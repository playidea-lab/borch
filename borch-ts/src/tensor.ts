/**
 * `Tensor` — one GPU buffer, a shape, and one node on the tape.
 *
 * The layout is **NCHW** and the dtype is **float32, only.** The sister
 * project (`borch_webgpu`) holding NHWC and packing int64 into float32 was
 * a detour around TF.js's constraints, and our kernels do not have those
 * constraints. Imitating it inherits someone else's detour for no reason.
 */

import { backward as tapeBackward, gradMode, type Node } from "./autograd.js";
import { Device, type DeviceKind, type InitOptions } from "./device.js";
import { type AxisPlan, isSlice, planAxis, type Slice } from "./indexing.js";
import { gauss, uniform } from "./random.js";
import {
  byRank, type DType, floatsPerElement, isComplexDType, promote, rankOf,
} from "./dtype.js";
import {
  IndexError, LinAlgError, NotImplementedError, RuntimeError, TORCH,
} from "./errors.js";

/**
 * `_ex` 계열이 상한 행렬에 담는 값.
 *
 * LAPACK 은 몇 번째 주피벗이 0 인지를 담는다. 여기서는 "상했다" 만 말하되 자릿수는
 * 맞춘다 — 진짜 torch 에 2×2 특이행렬을 주니 2 였다.
 */
const SINGULAR_INFO = 2;

/**
 * The four things `F.pad` accepts.
 */
export type PadMode = "constant" | "reflect" | "replicate" | "circular";

/**
 * How a loss folds. **Only `KLDivLoss` takes a fourth (`batchmean`).**
 */
export type Reduction = "none" | "mean" | "sum";

/**
 * SELU 의 고정점. `alphaDropout` 이 떨군 자리에 넣는 값이 여기서 나온다.
 *
 * 0 이 아니라 이 수를 넣어야 SELU 의 자기정규화가 유지된다 — 실측으로 확인했다
 * (입력이 전부 1 일 때 답이 `-0.779` 와 `1.666` 두 값이다).
 */
const ALPHA_PRIME = -1.7580993408473766;

/** 수 하나면 축마다 같은 값으로. */
function pairOf(v: number | readonly number[]): [number, number] {
  return typeof v === "number" ? [v, v] : [v[0] ?? 0, v[1] ?? 0];
}

/**
 * `(C·kh·kw, L)` 자리표. 값은 **덧댄** 입력의 평평한 자리다.
 *
 * `unfold` 는 이 자리를 모으고 `fold` 는 이 자리로 더해 넣는다 — 둘이 한 표를
 * 나눠 쓰므로 한쪽의 역방향이 곧 다른 쪽이 된다.
 */
function windowIndex(
  shape: [number, number, number],
  kernel: [number, number],
  dilation: [number, number],
  padding: [number, number],
  stride: [number, number],
): { idx: number[]; rows: number; cols: number } {
  const [c, h, w] = shape;
  const [kh, kw] = kernel;
  const [dh, dw] = dilation;
  const [ph, pw] = padding;
  const [sh, sw] = stride;
  const hp = h + 2 * ph;
  const wp = w + 2 * pw;
  const outH = Math.floor((hp - dh * (kh - 1) - 1) / sh) + 1;
  const outW = Math.floor((wp - dw * (kw - 1) - 1) / sw) + 1;
  const idx: number[] = [];
  for (let ch = 0; ch < c; ch++) {
    for (let i = 0; i < kh; i++) {
      for (let j = 0; j < kw; j++) {
        for (let oh = 0; oh < outH; oh++) {
          for (let ow = 0; ow < outW; ow++) {
            idx.push((ch * hp + oh * sh + i * dh) * wp + ow * sw + j * dw);
          }
        }
      }
    }
  }
  return { idx, rows: c * kh * kw, cols: outH * outW };
}

/** 겹선형 확대가 출력 자리마다 **어느 두 입력 자리를 얼마씩** 섞을지. */
function bilinearAxis(sizeIn: number, sizeOut: number, alignCorners: boolean) {
  const lo: number[] = [];
  const hi: number[] = [];
  const frac: number[] = [];
  for (let i = 0; i < sizeOut; i++) {
    const src = alignCorners
      ? i * ((sizeIn - 1) / Math.max(1, sizeOut - 1))
      : Math.max(0, (i + 0.5) * (sizeIn / sizeOut) - 0.5);
    const base = Math.floor(src);
    lo.push(base);
    hi.push(Math.min(base + 1, sizeIn - 1));
    frac.push(src - base);
  }
  return { lo, hi, frac };
}

/**
 * 출력 자리마다 **입력의 어느 자리를 읽는지.**
 *
 * 네 모드가 여기서만 갈린다. `[0,1,2]` 를 앞 2·뒤 1 로 늘리면(진짜 torch 에 물어
 * 자리마다 맞췄다):
 *
 *     reflect    2 1 [0 1 2] 1   ← 가장자리를 거울로 **하되 가장자리는 안 겹친다**
 *     replicate  0 0 [0 1 2] 2   ← 가장자리를 늘인다
 *     circular   1 2 [0 1 2] 0   ← 반대편에서 가져온다
 *
 * 색인 하나로 정리하면 순방향은 `indexSelect` 이고 역방향은 그것이 이미 하는
 * **모아 더하기**다 — 거울과 감기는 한 입력을 여러 번 읽으므로 덮어쓰면 그만큼이
 * 사라지는데, 그 자리를 새로 적을 필요가 없어진다.
 */
function padIndex(
  mode: PadMode, size: number, before: number, after: number,
): number[] {
  const idx: number[] = [];
  for (let i = -before; i < size + after; i++) {
    if (i >= 0 && i < size) { idx.push(i); continue; }
    if (mode === "replicate") { idx.push(i < 0 ? 0 : size - 1); continue; }
    if (mode === "circular") { idx.push(((i % size) + size) % size); continue; }
    let j = i;
    while (j < 0 || j >= size) j = j < 0 ? -j : 2 * (size - 1) - j;
    idx.push(j);
  }
  return idx;
}
import * as LA from "./linalg.js";
// **순환 가져오기다** — `special.ts` 가 `Tensor` 를 쓴다. 쓰는 자리가 메서드
// 몸통 안이라 모듈이 다 실린 뒤에 불리고, 그래서 돈다. 최상위에서 쓰면 안 된다.
import { polygamma } from "./special.js";
import { formatSize, formatTensor } from "./repr.js";
import {
  argReduce,
  type AxisRule,
  batchNormApply,
  batchNormBackwardApply,
  batchNormStats,
  batchNormStatsBackward,
  BINARY,
  binaryBackward,
  binaryForward,
  convGradInputGrid,
  convGradWeightGrid,
  convGradWeightSplit,
  convNDForwardTiled,
  convNDGradInputTiled,
  convNDGradWeightTiled,
  convNDKey,
  type ConvNDShape,
  convOut,
  convTiledGrid,
  cumExtreme,
  cumprodBackward,
  cumsumBackward,
  cumulative,
  diagflat,
  diagflatBackward,
  dropoutMask,
  uniformFill,
  expandDim,
  extremeBackward,
  fill,
  flatGather,
  flatGatherBackward,
  flatReduceInto,
  flatScatterInto,
  gather,
  gatherBackward,
  gatherIndex,
  gatherIndexBackward,
  indexSelect,
  indexSelectBackward,
  searchSorted,
  maskedScatterKernel,
  maskedScatterSourceBackward,
  matmul,
  padAxis,
  poolMaxIndexBackward,
  poolMaxWithIndex,
  poolNDBackward,
  poolNDBackwardNeedsInput,
  poolNDForward,
  poolNDKey,
  type PoolNDShape,
  type PoolWindows,
  poolWindowsKey,
  prodBackward,
  reduceBroadcast,
  reduceDim,
  type ReduceKind,
  ruleKey,
  scatterByIndex,
  scatterOverwrite,
  sortAxis,
  sumSplits,
  triangle,
  f32lit,
  hasUnary,
  UNARY,
  unaryBackward,
  unaryForward,
  unaryWith,
  unpoolFromIndex,
  unpoolFromIndexBackward,
  upsampleNearest,
  upsampleNearestBackward,
  whereBackward,
  whereKernel,
} from "./kernels.js";

/** 장치를 **객체 안에** 둔다. `autograd.ts` 의 `gradMode` 와 같은 이유다. */
const deviceHolder: { current: Device | null } = { current: null };

/**
 * 이만큼까지의 정수 지수는 곱셈으로 편다.
 *
 * 위로 갈수록 커널 호출이 그만큼 늘어나므로 무한정 펴지는 않는다. 이 위는 `pow`
 * 커널로 가고, 거기서는 음수 밑이 답이 없다.
 */
const MAX_UNROLLED_POWER = 8;

/**
 * float32 가 정수를 빠짐없이 셀 수 있는 한계(2^24).
 *
 * 이 위에서는 인접한 정수 둘이 같은 부동소수로 접힌다 — `randint` 가 거기서 멈추는
 * 이유다. 값이 조용히 반올림되면 라벨이 뒤섞이고 아무도 못 본다.
 */
const EXACT_INT_LIMIT = 16_777_216;

/**
 * 축약이 내는 형. **가르는 선은 "값을 만드는가" 다** — 모양·색인 연산에 그은 선과
 * 같은 선이고, torch 에게 서른세 자리를 물어 확인했다(`tests/test_reduce_dtype.py`).
 *
 * 누적(`sum`·`prod`·`cumsum`·`cumprod`)은 값을 **만든다** — 참·거짓 칸에 3 이 안
 * 들어가므로 bool 이 int64 로 올라간다. 고르기(`amax`·`amin`·`max`·`min`)는 있던 값을
 * **건네므로** 형이 그대로 간다. 축약이 예외였던 것이 아니라 둘이 서로 다른 것이다.
 */
function accumulated(from: DType): DType {
  return from === "bool" ? "int64" : from;
}

/**
 * What can go into one axis of `at()`. The syntax is written down in
 * `indexing.ts`.
 */
export type AtIndex = number | null | Slice | Tensor | readonly number[];

/** 계획 하나를 실제 연산으로. **여기서 값을 만들지 않는다** — 있는 문으로 보낸다. */
function applyPlan(t: Tensor, axis: number, plan: AxisPlan): Tensor {
  switch (plan.kind) {
    case "whole":
      return t;
    case "int":
      return t.select(axis, plan.at);
    case "range":
      return t.narrow(axis, plan.start, plan.length);
    case "picks":
      // 걸음이 있는 슬라이스와 번호표가 여기서 만난다 — 둘 다 "이 자리들" 이다.
      return t.indexSelect(
        axis,
        Tensor.from(plan.indices, [plan.indices.length], { dtype: "int64" }),
      );
  }
}

/**
 * 값 하나짜리 상수 텐서를 값으로 캐시한다.
 *
 * 학습 루프는 같은 상수를 매 스텝 다시 만든다 — 학습률, eps, 0.5, 게이트 수. 버퍼가
 * 4바이트이므로 들고 있는 값이 만드는 값보다 훨씬 싸다. 구역이 닫혀도 살아남게
 * 표시해 둔다(안 그러면 다음 스텝이 놓인 버퍼를 가리킨다).
 *
 * **쓰기는 안 간다.** 여기 오는 텐서는 상수이고, 제자리 연산은 자기 버퍼에만 쓴다.
 */
const scalarCache = new Map<number, GPUBuffer>();

/**
 * Acquires the WebGPU adapter. From the second call on it returns the one
 * already acquired.
 *
 * **`options` is heard on the first call only.** There is one device, and
 * once it is made there is nothing left to choose — better written here
 * than quietly ignored.
 */
export async function init(options: InitOptions = {}): Promise<Device> {
  if (!deviceHolder.current) deviceHolder.current = await Device.create(options);
  return deviceHolder.current;
}

/**
 * The device currently attached, or `null` if none. Where
 * `torch.accelerator` goes.
 *
 * **Unlike `dev()` it does not throw** — its purpose is to ask whether one
 * is attached, and that question itself must not fail.
 */
export function currentDevice(): DeviceKind | null {
  return deviceHolder.current ? "webgpu" : null;
}

function dev(): Device {
  const d = deviceHolder.current;
  if (!d) throw new Error("no device — call await init() first.");
  return d;
}

function numel(shape: readonly number[]): number {
  return shape.reduce((a, b) => a * b, 1);
}

/**
 * float32 가 담을 수 있는 가장 큰 유한한 값.
 *
 * `nanToNum` 이 무한대를 여기로 접는다 — torch 도 안 주면 그 형의 끝값을 쓴다.
 * `Number.MAX_VALUE` 는 배정도의 끝값이라 f32 버퍼에 넣으면 도로 무한대가 된다.
 */
const F32_MAX = 3.4028234663852886e38;

/**
 * torch's broadcasting rule. Aligned from the right, 1 stretches, and the
 * rest must match.
 */
export function broadcastShapes(
  a: readonly number[],
  b: readonly number[],
): number[] {
  const rank = Math.max(a.length, b.length);
  const out: number[] = new Array<number>(rank).fill(1);
  for (let i = 0; i < rank; i++) {
    const da = a[a.length - rank + i] ?? 1;
    const db = b[b.length - rank + i] ?? 1;
    if (da !== db && da !== 1 && db !== 1) {
      throw new RuntimeError(
        `The size of tensor a (${da}) ${TORCH.broadcast} b (${db}) at ` +
          `non-singleton dimension ${i}: [${a}] vs [${b}]`,
      );
    }
    out[i] = Math.max(da, db);
  }
  return out;
}

/**
 * The strides when `shape` is placed right-aligned against `out`'s rank.
 *
 * **A stretched axis is 0** — it reads the same value over and over.
 * Actually replicating to stretch costs memory, and that is exactly why
 * im2col lost to the fused kernel in the conv benchmark.
 */
export function alignStrides(
  shape: readonly number[],
  out: readonly number[],
): number[] {
  const rank = out.length;
  const own: number[] = new Array<number>(shape.length).fill(1);
  for (let d = shape.length - 2; d >= 0; d--) {
    own[d] = (own[d + 1] ?? 1) * (shape[d + 1] ?? 1);
  }
  const strides: number[] = new Array<number>(rank).fill(0);
  for (let i = 0; i < rank; i++) {
    const src = shape.length - rank + i;
    if (src < 0) continue;
    const dim = shape[src] ?? 1;
    strides[i] = dim === 1 && (out[i] ?? 1) !== 1 ? 0 : (own[src] ?? 1);
  }
  return strides;
}

// **분위수의 보간은 `quantileOver` 안으로 들어갔다.**
//
// 여기 값만 내는 헬퍼가 있었는데, 그 값으로 텐서를 만들면 **그래프가 없다.** 정렬
// 자리로 되짚어 뽑으면 값이 같고 기울기가 보간에 쓴 두 자리로 간다 — 그것이 이
// 연산의 규칙이다.

/**
 * 0 차 변형 베셀 함수. `kaiserWindow` 를 CPU 에서 만들므로 여기에도 한 벌 필요하다.
 *
 * 셰이더의 `i0_` 과 **같은 표**(Abramowitz & Stegun 9.8.1·9.8.2)를 쓴다. 두 벌을
 * 다르게 적으면 어느 쪽이 맞는지를 골든이 못 가른다.
 */
function besselI0(x: number): number {
  const a = Math.abs(x);
  if (a < 3.75) {
    const z = (x / 3.75) * (x / 3.75);
    return 1 + z * (3.5156229 + z * (3.0899424 + z * (1.2067492
      + z * (0.2659732 + z * (0.0360768 + z * 0.0045813)))));
  }
  const t = 3.75 / a;
  const poly = 0.39894228 + t * (0.01328592 + t * (0.00225319 + t * (-0.00157565
    + t * (0.00916281 + t * (-0.02057706 + t * (0.02635537
    + t * (-0.01647633 + t * 0.00392377)))))));
  return (Math.exp(a) / Math.sqrt(a)) * poly;
}

/**
 * ── 번호표 만들기 ──────────────────────────────────────────────────────────
 *
 * 모양만으로 정해지는 자리들을 **평평한 번호**로 뽑는다. 코어(numpy)는 같은 것을
 * `np.arange(size).reshape(shape)[...]` 로 만드는데, 여기에는 그런 뷰가 없어서 색인
 * 셈을 손으로 적는다. **두 벌이 되었으므로 골든이 심판이다** — 진짜 torch 를 세 번째
 * 답으로 두고 셋을 맞춘다.
 */

/** 행 우선 걸음. 축마다 한 칸 갈 때 평평한 번호가 얼마나 뛰는가. */
function rowStrides(shape: readonly number[]): number[] {
  const out = new Array<number>(shape.length).fill(1);
  for (let i = shape.length - 2; i >= 0; i--) {
    out[i] = out[i + 1]! * (shape[i + 1] ?? 1);
  }
  return out;
}

/** `shape` 위를 행 우선으로 훑으며 좌표마다 `at` 을 부른다. */
function eachCoord(
  shape: readonly number[],
  at: (coord: readonly number[], i: number) => void,
): void {
  const n = shape.reduce((a, b) => a * b, 1);
  const coord = new Array<number>(shape.length).fill(0);
  for (let i = 0; i < n; i++) {
    at(coord, i);
    for (let d = shape.length - 1; d >= 0; d--) {
      coord[d] = coord[d]! + 1;
      if (coord[d]! < shape[d]!) break;
      coord[d] = 0;
    }
  }
}

/** 걸음이 가리키는 자리들. 겹쳐도 되고 건너뛰어도 된다. */
function stridedSpots(
  size: readonly number[],
  stride: readonly number[],
  offset: number,
): Float32Array {
  const spots = new Float32Array(size.reduce((a, b) => a * b, 1));
  eachCoord(size, (c, i) => {
    let p = offset;
    for (let d = 0; d < size.length; d++) p += c[d]! * (stride[d] ?? 0);
    spots[i] = p;
  });
  return spots;
}

/** `x[..., start:stop:step, ...]` 의 자리들. */
function sliceSpots(
  shape: readonly number[],
  dim: number,
  start: number,
  stop: number,
  step: number,
): { spots: Float32Array; shape: number[] } {
  const st = rowStrides(shape);
  const count = Math.max(0, Math.ceil((stop - start) / step));
  const out = shape.map((s, d) => (d === dim ? count : s));
  const spots = new Float32Array(out.reduce((a, b) => a * b, 1));
  eachCoord(out, (c, i) => {
    let p = 0;
    for (let d = 0; d < shape.length; d++) {
      p += (d === dim ? start + c[d]! * step : c[d]!) * st[d]!;
    }
    spots[i] = p;
  });
  return { spots, shape: out };
}

/** `select(dim, index)` 의 자리들. 그 축은 결과에서 사라진다. */
function selectSpots(
  shape: readonly number[],
  dim: number,
  index: number,
): { spots: Float32Array; shape: number[] } {
  const st = rowStrides(shape);
  const out = shape.filter((_, d) => d !== dim);
  const spots = new Float32Array(out.reduce((a, b) => a * b, 1));
  eachCoord(out, (c, i) => {
    let p = index * st[dim]!;
    let k = 0;
    for (let d = 0; d < shape.length; d++) {
      if (d !== dim) p += c[k++]! * st[d]!;
    }
    spots[i] = p;
  });
  return { spots, shape: out };
}

/**
 * 대각선의 자리들. **그 축이 맨 뒤로 간다** — torch 도 numpy 도 그렇다.
 *
 * 배치 축이 있을 때 이 규약을 놓치면 값은 다 맞는데 순서만 갈린다. 2차원으로만 재면
 * 남는 축이 없어서 안 드러난다.
 */
function diagonalSpots(
  shape: readonly number[],
  offset: number,
  dim1: number,
  dim2: number,
): { spots: Float32Array; shape: number[] } {
  const st = rowStrides(shape);
  const n1 = shape[dim1] ?? 0;
  const n2 = shape[dim2] ?? 0;
  const k = offset >= 0
    ? Math.max(0, Math.min(n1, n2 - offset))
    : Math.max(0, Math.min(n1 + offset, n2));
  const r0 = offset >= 0 ? 0 : -offset;
  const c0 = offset >= 0 ? offset : 0;
  const rest: number[] = [];
  for (let d = 0; d < shape.length; d++) {
    if (d !== dim1 && d !== dim2) rest.push(d);
  }
  const out = rest.map((d) => shape[d]!).concat([k]);
  const spots = new Float32Array(out.reduce((a, b) => a * b, 1));
  eachCoord(out, (c, i) => {
    let p = 0;
    for (let j = 0; j < rest.length; j++) p += c[j]! * st[rest[j]!]!;
    const j = c[rest.length]!;
    spots[i] = p + (r0 + j) * st[dim1]! + (c0 + j) * st[dim2]!;
  });
  return { spots, shape: out };
}

/**
 * 히스토그램의 경계. **`min === max` 면 자료의 범위를 쓴다**(실측).
 *
 * 자료가 한 값뿐이면 그 범위가 0 이 되므로 양옆으로 반 칸씩 벌린다 — 안 그러면
 * 경계가 전부 같은 수가 되고 칸 너비가 0 이 된다.
 */
function histEdges(
  values: readonly number[],
  bins: number,
  min: number,
  max: number,
): number[] {
  let low = min;
  let high = max;
  if (low === high) {
    low = Math.min(...values);
    high = Math.max(...values);
    if (low === high) { low -= 0.5; high += 0.5; }
  }
  const step = (high - low) / bins;
  return Array.from({ length: bins + 1 }, (_, i) => low + step * i);
}

/** 값이 들어갈 칸. **범위 밖은 -1** 이고, 오른쪽 끝은 마지막 칸에 넣는다. */
function slotOf(value: number, edges: readonly number[]): number {
  const last = edges.length - 1;
  if (value < (edges[0] ?? 0) || value > (edges[last] ?? 0)) return -1;
  for (let i = 1; i <= last; i++) {
    if (value < (edges[i] ?? 0)) return i - 1;
  }
  return last - 1;
}

/** `edges` 가 나눈 칸에 센다. **범위 밖은 버린다** — torch 가 그렇다(실측). */
function countInto(
  values: readonly number[],
  edges: readonly number[],
  weights: readonly number[] | null,
): number[] {
  const out = new Array<number>(edges.length - 1).fill(0);
  values.forEach((value, i) => {
    const slot = slotOf(value, edges);
    if (slot < 0) return;
    out[slot] = (out[slot] ?? 0) + (weights === null ? 1 : (weights[i] ?? 0));
  });
  return out;
}

/** `dim` 축의 `at` 번째 줄이 차지하는 자리들. `uniqueConsecutive` 가 쓴다. */
function rowSpots(
  shape: readonly number[],
  dim: number,
  at: number,
  st: readonly number[],
): number[] {
  const out: number[] = [];
  eachCoord(shape.filter((_, d) => d !== dim), (c) => {
    let p = at * st[dim]!;
    let k = 0;
    for (let d = 0; d < shape.length; d++) if (d !== dim) p += c[k++]! * st[d]!;
    out.push(p);
  });
  return out;
}

/** 비교와 논리 연산은 입력이 무엇이든 참·거짓을 낸다. */
const BOOL_RESULT = new Set([
  "eq", "ne", "lt", "le", "gt", "ge", "logical_and", "logical_or",
]);

/** 산술 연산 이름을 승격 규칙의 기호로. 표에 없는 것은 높은 범주를 그대로 쓴다. */
const ARITH: Readonly<Record<string, "+" | "-" | "*" | "/">> = {
  add: "+", sub: "-", mul: "*", div: "/",
};

function resultDType(name: string, a: DType, b: DType): DType {
  if (BOOL_RESULT.has(name)) return "bool";
  const op = ARITH[name];
  if (op) return promote(a, b, op);
  // `maximum`·`pow` 처럼 표에 없는 것들. 형을 새로 만들지 않고 높은 쪽을 쓴다.
  return byRank(Math.max(rankOf(a), rankOf(b)));
}

/** `shape` 를 `out` 랭크에 오른쪽 맞춤한 것 — `reduceBroadcast` 가 쓴다. */
function padShape(shape: readonly number[], rank: number): number[] {
  const out: number[] = new Array<number>(rank).fill(1);
  for (let i = 0; i < rank; i++) {
    const src = shape.length - rank + i;
    if (src >= 0) out[i] = shape[src] ?? 1;
  }
  return out;
}

/**
 * 이 축소판에 **칸이 없는 형**의 거절. 파이썬 두 판의 문장을 그대로 쓴다 —
 * `borch/_base.py` 의 `_unsupported` 와 결속의 `_absent_dtype` 이 같은 글자다.
 */
function absentDType(name: string, shown: string): never {
  throw new RuntimeError(
    `\`.${name}()\`(${shown}) ${TORCH.absent}\n${TORCH.absentAdvice}`,
  );
}

export class Tensor implements Node<Tensor> {
  /**
   * **Not `readonly`** — one place, `mutate`, edits it.
   *
   * There are in-place operations that change the shape (`transpose_`,
   * `squeeze_`, `unsqueeze_`). Those edit the frame you view through rather
   * than the values, so the slot stays and only the shape is swapped.
   * Nowhere but that one place changes it.
   */
  shape: readonly number[];
  /**
   * **Not `readonly`** — the same reason as `shape`, and the same one place
   * edits it.
   *
   * `as_strided_` changes even the slot count. Other in-place operations
   * only change the frame, so "the slot count does not change" was true for
   * a long time, and then a name appeared for which it is not.
   */
  size: number;
  /**
   * 값이 GPU 에 있으면 그 버퍼, 호스트에 있으면 `null`. **직접 읽지 마라** — 밖에서
   * 보는 자리는 `buffer` 게터이고 그쪽이 장치를 확인한다.
   */
  private readonly gpu: GPUBuffer | null;
  /**
   * 태어날 때 그 버퍼가 **몇 번째 삶**이었는가. `device.ts` 의 `age` 와 짝이다.
   *
   * 구역이 닫히면 버퍼는 통으로 돌아가고 삶이 하나 오른다. 그 뒤로 이 텐서가 값에
   * 닿으려 하면 두 수가 어긋나고, 그러면 **이미 죽은 텐서**다.
   */
  private readonly age: number;
  /** 값이 호스트에 있으면 그 배열, GPU 에 있으면 `null`. */
  private readonly host: Float32Array | null;
  requiresGrad: boolean;
  grad: Tensor | null = null;
  freed = false;
  /**
   * **A label.** The values are always in a float32 buffer — the details
   * are in `src/dtype.ts`.
   */
  readonly dtype: DType;
  /**
   * Upstream in the graph. **Only `detach_` changes this** — hence not
   * `readonly`. Edited anywhere else, the backward of an already-made node
   * changes quietly.
   */
  parents: readonly Tensor[];
  backwardFn: ((grad: Tensor) => readonly (Tensor | null)[]) | null;
  gradName: string;

  constructor(
    storage: GPUBuffer | Float32Array,
    shape: readonly number[],
    options: {
      requiresGrad?: boolean;
      parents?: readonly Tensor[];
      backwardFn?: (grad: Tensor) => readonly (Tensor | null)[];
      gradName?: string;
      dtype?: DType;
    } = {},
  ) {
    // 저장소 둘을 한 자리로 받는다. 안쪽 47 군데가 버퍼를 넘기고 있고 그 자리를
    // 전부 고치는 것보다 여기서 갈라 두는 편이 낫다 — 갈림이 한 곳에 남는다.
    const onHost = storage instanceof Float32Array;
    this.gpu = onHost ? null : storage;
    this.host = onHost ? storage : null;
    this.age = this.gpu === null ? 0 : dev().age(this.gpu);
    this.shape = [...shape];
    this.size = numel(this.shape);
    this.parents = options.parents ?? [];
    this.gradName = options.gradName ?? "";
    this.dtype = options.dtype ?? "float32";
    // 부모 중 하나라도 흘리면 흘린다. no_grad 안에서는 아무도 안 흘린다.
    const inherited =
      gradMode.enabled && this.parents.some((p) => p.requiresGrad);
    this.requiresGrad = options.requiresGrad ?? inherited;
    this.backwardFn =
      this.requiresGrad && options.backwardFn ? options.backwardFn : null;
  }

  /**
   * The GPU buffer holding the values. **The only door through which an
   * operation touches them.**
   *
   * This getter is the device check. There are 176 operation entry points
   * and hanging a guard on each is not a thing that gets done, but all of
   * them pass through here in the end — 75 places inside the repository, 6
   * in the optimizers, 1 in `nn`. So feeding a host-side tensor into any
   * operation stops here, with torch's own wording.
   *
   * **Which operation it was is not recorded.** Recording that would mean
   * editing 176 places, and the stack trace already points at it.
   */
  get buffer(): GPUBuffer {
    if (this.gpu === null) {
      throw new RuntimeError(
        `${TORCH.crossDevice}, but found at least two devices, webgpu and cpu! ` +
          "a tensor on the host cannot take part in an operation — move it up with webgpu().",
      );
    }
    // **복소수도 여기서 막힌다 — 같은 문 하나를 두 번 쓴다.**
    //
    // 복소수 버퍼는 칸당 f32 두 개다. 그것을 모르는 커널이 받으면 앞쪽 절반만
    // 실수로 읽고 **예외 없이** 틀린 답을 낸다. 176 군데 진입점에 하나씩 가드를
    // 다는 것은 될 일이 아니지만, 그 전부가 결국 이 게터를 지난다.
    //
    // 복소수를 아는 코드는 `raw` 로 들어온다. 그래서 **기본값이 거절**이고, 새 연산을
    // 추가하는 사람이 아무것도 안 하면 그 연산은 복소수를 안 받는다 — 반대로 두면
    // 아무것도 안 한 연산이 복소수를 조용히 잘못 먹는다.
    if (isComplexDType(this.dtype)) {
      throw new RuntimeError(
        "this operation does not take complex64 yet — the storage is two f32 per slot " +
          "(interleaved), so a kernel that does not know about complex numbers reads it " +
          "and is quietly wrong. Use view_as_real to get a real tensor, or handle the " +
          "real and imaginary parts separately.",
      );
    }
    // **죽은 텐서도 여기서 막힌다 — 같은 문 하나를 세 번 쓴다.**
    this.refuseIfDead();
    return this.gpu;
  }

  /**
   * 구역이 닫힌 뒤에도 쓰이는 텐서를 멈춘다.
   *
   * **이걸 안 하면 조용히 남의 값이 나온다**(실측: `[1,2,3,4]` 를 흘리고 같은 크기를
   * 몇 번 더 잡은 뒤 읽으니 `9,9,9,9`). 버퍼가 파괴되지 않고 통에 돌아가기 때문이라
   * WebGPU 도 안 막아 준다 — 유효한 버퍼를 유효하게 읽는 것이 맞으니까.
   */
  private refuseIfDead(): void {
    if (this.gpu === null || dev().age(this.gpu) === this.age) return;
    throw new RuntimeError(
      "this tensor belongs to a closed scope — its buffer went back to the pool and " +
        "holds something else now.\n" +
        "  To carry a tensor out of scope(), mark it with keepAlive(t) or pass it as " +
        "the second argument of scope(body, () => [t]).",
    );
  }

  /**
   * The buffer holding the values — **without the complex check.** Only
   * code that knows complex uses it.
   *
   * The device check still happens. The difference between the two getters
   * is one line about complex, and that one line is the declaration "this
   * code knows the interleaved layout".
   */
  get raw(): GPUBuffer {
    if (this.gpu === null) {
      throw new RuntimeError(
        `${TORCH.crossDevice}, but found at least two devices, webgpu and cpu! ` +
          "a tensor on the host cannot take part in an operation — move it up with webgpu().",
      );
    }
    this.refuseIfDead();
    return this.gpu;
  }

  /**
   * How many f32 slots the buffer actually holds. For reals it equals
   * `size`; **for complex it is twice that.**
   *
   * Places that only ask "how many slots of value are there" — reads,
   * copies, lifetimes — use this. Using `size` directly cuts off the back
   * half of a complex quietly, with the shape and the element count both
   * looking plausible.
   */
  get floats(): number {
    return this.size * floatsPerElement(this.dtype);
  }

  /**
   * Where the values are. Where torch's `t.device` goes.
   *
   * **`'cpu'` is a place values are held, not a device that computes.**
   * borch has no CPU kernels — a tensor brought down with `cpu()` can only
   * be read (`toArray`, `item`, `repr`) and put back up, and feeding it to
   * an operation stops at the getter above with torch's own wording. It is
   * a partial implementation, but the failure is not quiet.
   */
  get device(): DeviceKind {
    return this.gpu === null ? "cpu" : "webgpu";
  }

  /**
   * 그래프에 마디를 하나 만든다. 코어의 `_make` 와 같은 자리다.
   *
   * `no_grad` 안이면 부모도 역방향도 안 달린다 — 달아 두고 안 쓰면 버퍼가 살아남아
   * 새지, 조용히 틀리지는 않지만 학습 루프에서는 그것도 치명적이다.
   */
  private static make(
    buffer: GPUBuffer,
    shape: readonly number[],
    parents: readonly Tensor[],
    backwardFn: (grad: Tensor) => readonly (Tensor | null)[],
    gradName: string,
    dtype: DType = "float32",
  ): Tensor {
    if (!gradMode.enabled || !parents.some((p) => p.requiresGrad)) {
      return new Tensor(buffer, shape, { requiresGrad: false, dtype });
    }
    return new Tensor(buffer, shape, { parents, backwardFn, gradName, dtype });
  }

  /**
   * The public door onto `make`. **Another module with kernels of its own**
   * (`fft.ts`) enters here.
   *
   * `make` itself is not exported because the name is too common — seen
   * from outside, `Tensor.make` does not read as making anything in
   * particular. `makeNode` at the end of the file wraps this.
   */
  static node(
    buffer: GPUBuffer,
    shape: readonly number[],
    parents: readonly Tensor[],
    backwardFn: (grad: Tensor) => readonly (Tensor | null)[],
    gradName: string,
    dtype: DType = "float32",
  ): Tensor {
    return Tensor.make(buffer, shape, parents, backwardFn, gradName, dtype);
  }

  // ── 만들기 ────────────────────────────────────────────────────────────

  /**
   * @param options taken by name rather than by position. A `device` was
   *   about to attach in fourth place, and `from(data, shape, false,
   *   "int64", "cpu")` is a line whose reader has to count the third and
   *   fourth arguments. Not yet published to npm is the moment this can be
   *   fixed.
   */
  static from(
    data: ArrayLike<number>,
    shape?: readonly number[],
    options: {
      requiresGrad?: boolean;
      dtype?: DType;
      device?: DeviceKind;
    } = {},
  ): Tensor {
    const { requiresGrad = false, dtype = "float32", device = "webgpu" } = options;
    const flat = data instanceof Float32Array ? data : Float32Array.from(data);
    const shp = shape ?? [flat.length];
    if (numel(shp) !== flat.length) {
      throw new Error(`shape [${shp}] does not match ${flat.length} elements.`);
    }
    // **복소수는 이 문으로 못 들어온다.** 이름표만 `complex64` 로 달면 칸 수는
    // `n` 인데 저장 규약은 `2n` 을 요구하므로, 뒤쪽 절반이 남의 메모리가 된다 —
    // 예외 없이 아무 값이나 읽힌다. 엮는 자리를 하나로 둔다.
    if (dtype === "complex64") {
      throw new RuntimeError(
        "Tensor.from cannot make complex64 — the storage is two f32 per slot. Use " +
          "Tensor.complex(re, im), Tensor.polar(r, theta), or x.viewAsComplex().",
      );
    }
    if (requiresGrad && dtype !== "float32") {
      // 정수와 참·거짓에는 기울기가 정의되지 않는다. torch 도 여기서 멈춘다 —
      // 흘려보내면 학습이 도는데 그 값이 아무 뜻도 없는 상태가 된다.
      throw new RuntimeError(
        "Only Tensors of floating point and complex dtype can require gradients",
      );
    }
    // 호스트에 두라면 사본을 든다. 넘겨받은 배열을 그대로 쥐면 부른 쪽이 그것을
    // 고칠 때 텐서 값이 같이 바뀐다 — GPU 쪽은 `upload` 가 복사하므로 그 자리가 없다.
    if (device === "cpu") return new Tensor(flat.slice(), shp, { requiresGrad, dtype });
    return new Tensor(dev().upload(flat), shp, { requiresGrad, dtype });
  }

  static full(shape: readonly number[], value: number): Tensor {
    const n = numel(shape);
    // **스칼라는 커널을 안 부른다.** 원소 하나를 쓰겠다고 dispatch 를 보내는 것은
    // 순수한 낭비인데, `x * 0.5` 같은 식이 전부 이리로 와서 ResNet 한 스텝에 286 번
    // 돌고 있었다(실측 — dispatch 1,636 개 중 17%). 게다가 학습 루프에서는 같은
    // 상수가 매 스텝 되풀이되므로 값으로 캐시한다.
    if (n === 1) {
      const hit = scalarCache.get(value);
      if (hit) return new Tensor(hit, shape);
      const buf = dev().upload(Float32Array.of(value));
      dev().keep(buf);
      scalarCache.set(value, buf);
      return new Tensor(buf, shape);
    }
    // **무한대와 NaN 은 셰이더로 못 굽는다.** WGSL 은 컴파일 시점에 계산되는 값이
    // inf 나 NaN 이 되는 것을 금지한다 — 리터럴도, `bitcast<f32>(0x7f800000u)` 도
    // 똑같이 `value inf cannot be represented as 'f32'` 로 거절당한다(실측). 그래서
    // 이쪽만 CPU 에서 채워 올린다. 업로드 한 번이라 커널보다 오히려 짧다.
    if (!Number.isFinite(value)) {
      return new Tensor(dev().upload(new Float32Array(n).fill(value)), shape);
    }
    const out = dev().alloc(n);
    dev().run1d(
      dev().pipeline(`fill:${n}:${value}`, () => fill(n, value)),
      [out],
      n,
    );
    return new Tensor(out, shape);
  }

  /**
   * Filled with a value but **without the cache — it gets its own buffer.**
   *
   * **Anything that will be modified in place has to come this way.**
   * `full` caches single-element tensors by value, so calling `zeros([1])`
   * twice returns **the same buffer.** That in itself is right — constants
   * are only read. The problem is something that is not a constant passing
   * through that door:
   *
   * - `Adam`'s m and v become the same buffer at a size-1 parameter, and
   *   WebGPU invalidates **the whole command buffer** for "writable storage
   *   buffer aliasing"
   * - `SGD`'s momentum buffer overwrites the zero constant the entire
   *   program uses
   * - `nn.PReLU()` defaults to a single parameter, so its weight *is* the
   *   global 0.25 constant, and the optimizer edits it during training
   * - `BatchNorm(1)`'s running statistics overwrite the global 0 and 1
   *   constants
   *
   * Only the first throws; the rest are **quietly wrong.** So parameters,
   * optimizer state and running statistics come here whatever their value.
   */
  static owned(shape: readonly number[], value = 0): Tensor {
    const data = new Float32Array(numel(shape));
    if (value !== 0) data.fill(value);
    return new Tensor(dev().upload(data), shape);
  }

  static zeros(shape: readonly number[]): Tensor {
    return Tensor.full(shape, 0);
  }

  static ones(shape: readonly number[]): Tensor {
    return Tensor.full(shape, 1);
  }

  /**
   * 단위 행렬. **CPU 에서 만들어 올린다** — 만드는 일은 한 번뿐이고, 이걸 위해
   * 셰이더를 하나 더 굽는 것은 얻는 것보다 비싸다.
   */
  /**
   * The identity matrix. **Not only square** — torch's `eye(n, m)` makes a
   * rectangle.
   */
  static eye(n: number, m: number = n): Tensor {
    const data = new Float32Array(n * m);
    for (let i = 0; i < Math.min(n, m); i++) data[i * m + i] = 1;
    return Tensor.from(data, [n, m]);
  }

  /** `0` 부터 `n-1` 까지. */
  /**
   * `[start, end)` in steps of `step`. **The end is excluded.**
   *
   * Given one argument it is `[0, n)`, as in torch, and for a long time
   * that was the only form. While start and step could not be given,
   * callers imitated it with `arange(n).mul(...).add(...)`, which is a
   * different computation whose rounding accumulates differently.
   */
  static arange(start: number, end?: number, step = 1): Tensor {
    const [from, to] = end === undefined ? [0, start] : [start, end];
    Tensor.needsStep(step, "arange");
    const n = Math.max(0, Math.ceil((to - from) / step));
    const data = new Float32Array(n);
    for (let i = 0; i < n; i++) data[i] = from + step * i;
    return Tensor.from(data, [n]);
  }

  /**
   * 걸음이 0 이면 **여기서 멈춘다.** torch 와 같은 자리다.
   *
   * 안 막으면 `(to - from) / 0` 이 Infinity 가 되어 배열을 잡는 자리에서 터지는데,
   * 그 문구는 메모리가 모자란 것과 구별이 안 된다 — 무엇을 잘못 넣었는지 안 보인다.
   * 걸음이 0 이면 값이 영원히 안 움직이므로 답이 없는 것이지 큰 것이 아니다.
   */
  private static needsStep(step: number, who: string): void {
    if (step === 0) {
      throw new RuntimeError(
        `${who}: step must be nonzero — with a step of 0 the value never moves and ` +
        "the range never ends.",
      );
    }
  }

  /**
   * The old name for `arange`, except that it **includes the end**
   * (measured). So it has one element more.
   *
   * torch marked it deprecated over this same difference. Quietly
   * forwarding to `arange` loses the last value, and that one slot is
   * invisible if you only measure sums or means.
   */
  static range(start: number, end: number, step = 1): Tensor {
    Tensor.needsStep(step, "range");
    const n = Math.max(0, Math.floor((end - start) / step) + 1);
    const data = new Float32Array(n);
    for (let i = 0; i < n; i++) data[i] = start + step * i;
    return Tensor.from(data, [n]);
  }

  /**
   * Reads the bytes **as they are.** `offset` is in bytes, not elements.
   *
   * `dtype` here decides **what to read them as.** That differs from
   * converting after reading — the same bytes become very different numbers
   * depending on the type, so fixing it later with `to()` comes after they
   * have already been read as something else.
   *
   * `complex64` is refused. The storage is interleaved so the bytes could
   * be laid on directly, but that arrangement is not nailed down as a
   * promise, which would leave **the reader not knowing what they
   * received** — better not to offer the name.
   */
  static frombuffer(
    buffer: ArrayBuffer,
    dtype: DType = "float32",
    count = -1,
    offset = 0,
  ): Tensor {
    const width = { float32: 4, int64: 8, bool: 1, complex64: 8 }[dtype];
    if (dtype === "complex64") {
      throw new RuntimeError(
        "frombuffer does not read complex64 — an interleaved layout is not part of the " +
        "contract. Read the real and imaginary parts separately and join them with " +
        "Tensor.complex.",
      );
    }
    const room = Math.floor((buffer.byteLength - offset) / width);
    const n = count < 0 ? room : count;
    if (n > room) {
      throw new RuntimeError(
        `frombuffer: asked for ${n} values but the buffer holds ${room} ` +
        `(bytes ${buffer.byteLength}, offset ${offset}, ${width} bytes per value)`,
      );
    }
    const data = new Float32Array(n);
    if (dtype === "float32") {
      data.set(new Float32Array(buffer, offset, n));
    } else if (dtype === "bool") {
      const raw = new Uint8Array(buffer, offset, n);
      for (let i = 0; i < n; i++) data[i] = raw[i] ? 1 : 0;
    } else {
      // **int64 는 f32 칸에 담긴다.** 2^24 를 넘으면 이미 못 세므로 거기서 멈춘다 —
      // 조용히 반올림되느니 낫다. `randint` 가 같은 자리에서 같은 말을 한다.
      const raw = new BigInt64Array(buffer, offset, n);
      for (let i = 0; i < n; i++) {
        const v = raw[i] ?? 0n;
        if (v > BigInt(EXACT_INT_LIMIT) || v < -BigInt(EXACT_INT_LIMIT)) {
          throw new RuntimeError(
            `frombuffer: ${v} is past what float32 counts exactly (${EXACT_INT_LIMIT}) — ` +
            "the value would be quietly rounded.",
          );
        }
        data[i] = Number(v);
      }
    }
    return Tensor.from(data, [n], { dtype });
  }

  /**
   * Evenly divided values, both ends included.
   */
  static linspace(start: number, end: number, count: number): Tensor {
    const data = new Float32Array(count);
    // 마지막 값을 계산으로 내면 반올림이 쌓여 end 에 정확히 안 닿는다. 못 박는다.
    const step = count > 1 ? (end - start) / (count - 1) : 0;
    for (let i = 0; i < count; i++) data[i] = start + step * i;
    if (count > 1) data[count - 1] = end;
    return Tensor.from(data, [count]);
  }

  /**
   * 창 함수 다섯의 공통 뼈대. **CPU 에서 만들어 올린다** — `eye` 와 같은 이유다.
   *
   * **`periodic` 이 기본이고 그것이 길이를 하나 늘린다.** 참이면 `N+1` 짜리 대칭
   * 창을 만들어 마지막을 버린다(실측: `hannWindow(5)` 가 대칭 6 의 앞 다섯과 정확히
   * 같다). 거짓으로만 물으면 그 규칙이 안 드러난다.
   *
   * `n === 1` 은 따로 둔다 — 나누는 자리(`total - 1`)가 0 이 된다.
   */
  private static window(
    n: number,
    periodic: boolean,
    at: (k: number, total: number) => number,
  ): Tensor {
    if (n <= 0) return Tensor.from(new Float32Array(0), [0]);
    if (n === 1) return Tensor.from(new Float32Array([1]), [1]);
    const total = periodic ? n + 1 : n;
    const data = new Float32Array(n);
    for (let k = 0; k < n; k++) data[k] = at(k, total);
    return Tensor.from(data, [n]);
  }

  /**
   * The triangular window. 1 in the middle, 0 at both ends.
   */
  static bartlettWindow(n: number, periodic = true): Tensor {
    return Tensor.window(n, periodic, (k, total) =>
      1 - Math.abs((2 * k) / (total - 1) - 1));
  }

  static hannWindow(n: number, periodic = true): Tensor {
    return Tensor.window(n, periodic, (k, total) =>
      0.5 - 0.5 * Math.cos((2 * Math.PI * k) / (total - 1)));
  }

  /**
   * `alpha - beta·cos`. torch's defaults are 0.54/0.46.
   */
  static hammingWindow(
    n: number, periodic = true, alpha = 0.54, beta = 0.46,
  ): Tensor {
    return Tensor.window(n, periodic, (k, total) =>
      alpha - beta * Math.cos((2 * Math.PI * k) / (total - 1)));
  }

  static blackmanWindow(n: number, periodic = true): Tensor {
    return Tensor.window(n, periodic, (k, total) => {
      const t = (2 * Math.PI * k) / (total - 1);
      return 0.42 - 0.5 * Math.cos(t) + 0.08 * Math.cos(2 * t);
    });
  }

  /**
   * `I₀(β√(1-((k-h)/h)²)) / I₀(β)`. torch's default `beta` is 12.0.
   */
  static kaiserWindow(n: number, periodic = true, beta = 12.0): Tensor {
    return Tensor.window(n, periodic, (k, total) => {
      const half = (total - 1) / 2;
      const r = (k - half) / half;
      return besselI0(beta * Math.sqrt(Math.max(0, 1 - r * r))) / besselI0(beta);
    });
  }

  /**
   * Uniform on `[0, 1)`. Where `torch.rand` goes, and **the door a user
   * should use.**
   *
   * **Made on the host and uploaded in one go.** That is what puts it on
   * the same stream as `randn`, `randint` and `randperm`, so one
   * `manualSeed` rewinds all four.
   *
   * There is one more door randomness comes out of — `Tensor.uniform` is a
   * GPU kernel and uses the dropout counter. That one belongs to places
   * drawing **a sample the size of the activations every step** (dropout,
   * `rrelu`, gumbel) and exists separately to avoid the upload.
   * `manualSeed` rewinds both, but **they are not the same stream.**
   */
  static rand(shape: readonly number[]): Tensor {
    const data = new Float32Array(numel(shape));
    for (let i = 0; i < data.length; i++) data[i] = uniform();
    return Tensor.from(data, shape);
  }

  /**
   * The standard normal. Where `torch.randn` goes.
   */
  static randn(shape: readonly number[]): Tensor {
    const data = new Float32Array(numel(shape));
    for (let i = 0; i < data.length; i++) data[i] = gauss();
    return Tensor.from(data, shape);
  }

  /**
   * Integers in `[low, high)`. **The upper end is excluded**, as in torch.
   *
   * The type comes back as `int64` while the values sit in a float32 buffer
   * (`dtype.ts`). Integers above 2^24 cannot be counted there already, so
   * it stops at that point — better than rounding quietly.
   */
  static randint(low: number, high: number, shape: readonly number[]): Tensor {
    if (!(high > low)) {
      throw new RuntimeError(`random_ expects 'from' to be less than 'to', but got from=${low} >= to=${high}`);
    }
    if (Math.max(Math.abs(low), Math.abs(high)) > EXACT_INT_LIMIT) {
      throw new RuntimeError(
        `the randint range runs past what float32 counts exactly (${EXACT_INT_LIMIT}) — ` +
          "values would be quietly rounded.",
      );
    }
    const span = high - low;
    const data = new Float32Array(numel(shape));
    for (let i = 0; i < data.length; i++) data[i] = low + Math.floor(uniform() * span);
    return Tensor.from(data, shape, { dtype: "int64" });
  }

  /**
   * `0..n-1` shuffled. Where `torch.randperm` goes. Fisher–Yates.
   */
  static randperm(n: number): Tensor {
    const data = new Float32Array(n);
    for (let i = 0; i < n; i++) data[i] = i;
    for (let i = n - 1; i > 0; i--) {
      const j = Math.floor(uniform() * (i + 1));
      const tmp = data[i] as number;
      data[i] = data[j] as number;
      data[j] = tmp;
    }
    return Tensor.from(data, [n], { dtype: "int64" });
  }

  /**
   * A sample from the normal distribution. **With `std` at 0 it is the mean
   * itself** (measured) — that endpoint is the only place that asks whether
   * this name really looks at the standard deviation.
   *
   * torch offers two forms: two tensors give a per-slot mean and standard
   * deviation, while two numbers and a shape give the same one everywhere.
   * The first **broadcasts** — a `(2,)` mean with a scalar standard
   * deviation is allowed.
   */
  static normal(
    mean: number | Tensor = 0,
    std: number | Tensor = 1,
    size?: readonly number[],
  ): Tensor {
    if (mean instanceof Tensor || std instanceof Tensor) {
      const m = mean instanceof Tensor ? mean : Tensor.full([], mean);
      const s = std instanceof Tensor ? std : Tensor.full([], std);
      return Tensor.randn(broadcastShapes(m.shape, s.shape)).mul(s).add(m);
    }
    return Tensor.randn(size ?? []).mul(Tensor.full([], std))
      .add(Tensor.full([], mean));
  }

  /**
   * Each slot **reads its own value as a probability** and lands on 0 or 1.
   * At `p=0` all zeros, at `p=1` all ones.
   *
   * **The comparison finishes on the device** — once the drawn randomness
   * is uploaded, no value comes back down. The binding (`borch_webgpu`)
   * builds the same name out of numpy, but that is because `get_rng_state`
   * over there has to serialise one stream, and there is no reason for this
   * side to do the same.
   */
  bernoulli(): Tensor {
    return Tensor.rand(this.shape).binary("lt", this).to(this.dtype);
  }

  /**
   * A Poisson sample at each slot's own mean. **A mean of 0 gives 0.**
   *
   * It is Knuth's multiplicative form, so **the number of iterations
   * differs per value** — a loop that diverges per slot cannot be moved
   * into a shader, so it reads once here. The same place as `histc` and
   * `mode`.
   */
  async poisson(): Promise<Tensor> {
    const lam = Array.from(await this.toArray());
    const got = lam.map((mean) => {
      if (!(mean > 0)) return 0;
      const limit = Math.exp(-mean);
      let count = 0;
      let product = uniform();
      while (product > limit) { count += 1; product *= uniform(); }
      return count;
    });
    return Tensor.from(got, [...this.shape], { dtype: this.dtype });
  }

  /**
   * The number of successes in `this` trials. **At probability 0 it is 0;
   * at 1 it is the trial count itself.**
   *
   * The trial count is a value, so the iteration count is a value too —
   * read for the same reason as `poisson`.
   */
  async binomial(prob: Tensor): Promise<Tensor> {
    const shape = broadcastShapes(this.shape, prob.shape);
    const counts = Array.from(await this.add(Tensor.zeros(shape)).toArray());
    const chances = Array.from(await prob.add(Tensor.zeros(shape)).toArray());
    const got = counts.map((n, i) => {
      const p = chances[i] ?? 0;
      let hits = 0;
      for (let k = 0; k < n; k++) if (uniform() < p) hits += 1;
      return hits;
    });
    return Tensor.from(got, shape, { dtype: this.dtype });
  }

  /**
   * 아래·위 삼각의 자리들. **`(2, 개수)` 짜리 표다**(실측) — 자리 쌍이 아니라 행 줄과
   * 열 줄로 나뉘어 온다. 쌍의 목록으로 읽으면 모양부터 다르다.
   */
  private static triangleIndices(
    row: number,
    col: number,
    offset: number,
    lower: boolean,
  ): Tensor {
    const rows: number[] = [];
    const cols: number[] = [];
    for (let r = 0; r < row; r++) {
      for (let c = 0; c < col; c++) {
        if (lower ? c - r <= offset : c - r >= offset) {
          rows.push(r);
          cols.push(c);
        }
      }
    }
    return Tensor.from(rows.concat(cols), [2, rows.length], { dtype: "int64" });
  }

  static trilIndices(row: number, col: number, offset = 0): Tensor {
    return Tensor.triangleIndices(row, col, offset, true);
  }

  static triuIndices(row: number, col: number, offset = 0): Tensor {
    return Tensor.triangleIndices(row, col, offset, false);
  }

  /**
   * Every pair. **Given only one it is just that** (measured) — it stays
   * one-dimensional.
   *
   * It does not read the values. Which slot goes where is decided **by the
   * counts alone**, so it gathers by index — reading would have made this
   * asynchronous too.
   */
  static cartesianProd(...tensors: readonly Tensor[]): Tensor {
    if (tensors.length === 1) return tensors[0]!.reshape([tensors[0]!.size]);
    const sizes = tensors.map((t) => t.size);
    const total = sizes.reduce((a, b) => a * b, 1);
    const columns = tensors.map((t, d) => {
      const after = sizes.slice(d + 1).reduce((a, b) => a * b, 1);
      const spots = new Float32Array(total);
      for (let i = 0; i < total; i++) {
        spots[i] = Math.floor(i / after) % sizes[d]!;
      }
      return t.reshape([t.size]).gatherSpots(Tensor.spotsTensor(spots), [total]);
    });
    return Tensor.stack(columns, 1);
  }

  /**
   * Combinations of `r` at a time. **Order does not matter**, and
   * with-replacement is separate.
   */
  static combinations(t: Tensor, r = 2, withReplacement = false): Tensor {
    const n = t.size;
    const rows: number[][] = [];
    const walk = (start: number, picked: number[]): void => {
      if (picked.length === r) {
        rows.push([...picked]);
        return;
      }
      for (let i = start; i < n; i++) {
        picked.push(i);
        walk(withReplacement ? i : i + 1, picked);
        picked.pop();
      }
    };
    walk(0, []);
    if (rows.length === 0) return Tensor.zeros([0, r]);
    const spots = new Float32Array(rows.length * r);
    rows.forEach((row, i) => row.forEach((v, j) => {
      spots[i * r + j] = v;
    }));
    return t.reshape([n]).gatherSpots(Tensor.spotsTensor(spots),
      [rows.length, r]);
  }

  /**
   * The Vandermonde matrix. **The default is decreasing powers** — the last
   * column is 1 (measured).
   *
   * The powers are not written with `pow`. WGSL's `pow` is
   * `exp2(y·log2(x))`, so **a negative base has no answer**, and negative
   * inputs are ordinary for a Vandermonde. Counting with a running product
   * is exact.
   */
  static vander(x: Tensor, N?: number, increasing = false): Tensor {
    const n = x.size;
    const cols = N === undefined ? n : N;
    if (cols === 0) return Tensor.zeros([n, 0]);
    // 첫 열은 1, 나머지는 x — 그 줄을 누적곱하면 0..N-1 차가 차례로 나온다.
    const spots = new Float32Array(n * cols);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < cols; j++) spots[i * cols + j] = i;
    }
    const spread = x.reshape([n]).gatherSpots(Tensor.spotsTensor(spots),
      [n, cols]);
    const head = new Float32Array(n * cols).fill(1);
    for (let i = 0; i < n; i++) {
      for (let j = 1; j < cols; j++) head[i * cols + j] = 0;
    }
    const isFirst = Tensor.from(head, [n, cols]);
    const base = isFirst.add(Tensor.full([], 1).sub(isFirst).mul(spread));
    const up = base.cumprod(1);
    return increasing ? up : up.flip(1);
  }

  zerosLike(): Tensor {
    return Tensor.zeros(this.shape);
  }

  onesLike(): Tensor {
    return Tensor.ones(this.shape);
  }

  /**
   * `torch.rand_like`. It borrows the shape only — neither the values nor
   * the type are inherited.
   */
  randLike(): Tensor {
    return Tensor.rand(this.shape);
  }

  /**
   * `torch.randn_like`.
   */
  randnLike(): Tensor {
    return Tensor.randn(this.shape);
  }

  // ── 원소별 ────────────────────────────────────────────────────────────

  unary(name: string): Tensor {
    if (!hasUnary(name)) throw new Error(`unknown unary op: ${name}`);
    const n = this.size;
    const out = dev().alloc(n);
    dev().run1d(
      dev().pipeline(`u:${name}:${n}`, () => unaryForward(name, n)),
      [this.buffer, out],
      n,
    );
    const result = Tensor.make(
      out,
      this.shape,
      [this],
      (g) => {
        const gi = dev().alloc(n);
        dev().run1d(
          dev().pipeline(`ub:${name}:${n}`, () => unaryBackward(name, n)),
          [this.buffer, result.buffer, g.buffer, gi],
          n,
        );
        return [new Tensor(gi, this.shape)];
      },
      `${name[0]?.toUpperCase()}${name.slice(1)}Backward0`,
    );
    return result;
  }

  /**
   * @param dtype the result's type. Left out, it follows the arithmetic
   *   promotion rules. Where the type is independent of the inputs, as in a
   *   comparison, it is pinned here.
   */
  binary(name: string, other: Tensor, dtype?: DType): Tensor {
    const spec = BINARY[name];
    if (!spec) throw new Error(`unknown binary op: ${name}`);
    // **복소수가 끼면 여기서 갈린다.** 이 메서드가 이항 연산의 유일한 문이라
    // `add`·`mul` 뿐 아니라 역전파의 기울기 누적까지 전부 여기를 지난다 — 누적이
    // 실수 커널로 새면 복소수 잎의 기울기가 앞쪽 절반만 더해진다.
    if (this.isComplex() || other.isComplex()) {
      if (name === "add" || name === "sub" || name === "mul" || name === "div") {
        return this.complexBinary(name, other);
      }
      // 나머지는 아래 실수 커널로 가면 안 된다. `buffer` 게터가 막겠지만, 문구가
      // "이 연산은 아직" 이라 어느 연산인지가 안 남는다.
      throw new RuntimeError(
        `complex64 does not have ${name} yet — what works now is ` +
          "add, sub, mul, div.",
      );
    }
    const outType = dtype ?? resultDType(name, this.dtype, other.dtype);
    const shape = broadcastShapes(this.shape, other.shape);
    const sa = alignStrides(this.shape, shape);
    const sb = alignStrides(other.shape, shape);
    const n = numel(shape);
    const key = `${shape}|${sa}|${sb}`;
    const out = dev().alloc(n);
    dev().run1d(
      dev().pipeline(`b:${name}:${key}`, () => binaryForward(name, shape, sa, sb)),
      [this.buffer, other.buffer, out],
      n,
    );
    const result = Tensor.make(
      out,
      shape,
      [this, other],
      (g) => {
        const side = (which: "a" | "b", self: Tensor): Tensor => {
          const wide = dev().alloc(n);
          dev().run1d(
            dev().pipeline(
              `bb:${name}:${which}:${key}`,
              () => binaryBackward(name, which, shape, sa, sb),
            ),
            [this.buffer, other.buffer, result.buffer, g.buffer, wide],
            n,
          );
          const wideTensor = new Tensor(wide, shape);
          return foldTo(wideTensor, self.shape);
        };
        return [
          this.requiresGrad ? side("a", this) : null,
          other.requiresGrad ? side("b", other) : null,
        ];
      },
      `${name[0]?.toUpperCase()}${name.slice(1)}Backward0`,
      outType,
    );
    return result;
  }

  /**
   * An argument torch offers that was missing here. Without it the caller
   * writes it out as `x.add(y.mul(a))` for the same answer, while **the
   * in-place form (`add_`) already had it** — so two names were taking
   * different things.
   *
   * @param alpha what the other side is multiplied by first — `this +
   *   alpha·other`.
   */
  add(other: Tensor, alpha = 1): Tensor {
    return this.binary("add", alpha === 1 ? other : other.mul(Tensor.full([], alpha)));
  }
  sub(other: Tensor, alpha = 1): Tensor {
    return this.binary("sub", alpha === 1 ? other : other.mul(Tensor.full([], alpha)));
  }
  mul(other: Tensor): Tensor {
    return this.binary("mul", other);
  }

  /**
   * **The type diverges.** True division is always float, but truncating or
   * flooring **comes back as the input's type** (measured: `int64 / int64`
   * with `trunc` is int64). Match only the value and leave the type float,
   * and it diverges later wherever indexing wants an integer.
   *
   * @param roundingMode `null` is true division; `"trunc"` and `"floor"` go
   *   to the integer side.
   */
  div(other: Tensor, roundingMode: "trunc" | "floor" | null = null): Tensor {
    const out = this.binary("div", other);
    if (roundingMode === null) return out;
    const rounded = roundingMode === "floor" ? out.floor() : out.trunc();
    const kind = resultDType("mul", this.dtype, other.dtype);
    return kind === "float32" ? rounded : rounded.to(kind);
  }

  // ── 행렬곱 ────────────────────────────────────────────────────────────

  /**
   * Two-dimensional only. Batched matrix multiply is T1 — a missing feature
   * beats a wrong answer.
   */
  mm(other: Tensor): Tensor {
    if (this.shape.length !== 2 || other.shape.length !== 2) {
      throw new Error(
        `mm is 2-D by 2-D: [${this.shape}] x [${other.shape}]. ` +
          "Batching is not here yet.",
      );
    }
    const M = this.shape[0] ?? 0;
    const K = this.shape[1] ?? 0;
    const K2 = other.shape[0] ?? 0;
    const N = other.shape[1] ?? 0;
    if (K !== K2) {
      throw new RuntimeError(
        `mat1 and mat2 ${TORCH.matmulShape} ` +
          `(${M}x${K} and ${K2}x${N})`,
      );
    }
    // **복소수는 실수 행렬곱 넷으로 쪼갠다.**
    //
    //   `(A + iB)(C + iD) = (AC − BD) + i(AD + BC)`
    //
    // 커널을 새로 안 쓴다 — `sum`·`diagflat` 이 같은 자리를 같은 방법으로 지난다.
    // 역방향도 공짜다: `real`·`imag`·`complex` 와 실수 `mm` 이 전부 자기 역방향을
    // 알고 있어서 이 식이 그대로 그래프가 된다.
    //
    // 한쪽만 복소수인 경우도 여기로 온다 — 실수 쪽의 허수부는 0 이라 두 곱이
    // 사라지지만, 그것을 특수화하는 것은 재보고 할 일이다. 맞는 것이 먼저다.
    if (this.isComplex() || other.isComplex()) {
      const a = this.isComplex() ? this : this.asComplexRe();
      const b = other.isComplex() ? other : other.asComplexRe();
      const [ar, ai] = [a.real(), a.imag()];
      const [br, bi] = [b.real(), b.imag()];
      return Tensor.complex(
        ar.mm(br).sub(ai.mm(bi)),
        ar.mm(bi).add(ai.mm(br)));
    }
    const out = dev().alloc(M * N);
    dev().run(
      dev().pipeline(`mm:${M}:${K}:${N}`, () => matmul(M, K, N)),
      [this.buffer, other.buffer, out],
      [Math.ceil(N / 64), Math.ceil(M / 64), 1],
    );
    return Tensor.make(
      out,
      [M, N],
      [this, other],
      (g) => [
        this.requiresGrad ? g.mm(other.transpose()) : null,
        other.requiresGrad ? this.transpose().mm(g) : null,
      ],
      "MmBackward0",
    );
  }

  /**
   * Two-dimensional transpose. For now it actually moves the data — views
   * are T1.
   */
  transpose(): Tensor {
    if (this.shape.length !== 2) {
      throw new Error(`transpose is 2-D only for now: [${this.shape}]`);
    }
    const M = this.shape[0] ?? 0;
    const N = this.shape[1] ?? 0;
    const out = dev().alloc(M * N);
    dev().run1d(
      dev().pipeline(`t:${M}:${N}`, () => transposeKernel(M, N)),
      [this.buffer, out],
      M * N,
    );
    return Tensor.make(out, [N, M], [this], (g) => [g.transpose()], "TBackward0",
      this.dtype);
  }

  // ── 축약 ──────────────────────────────────────────────────────────────

  /** 전부 더해 스칼라 하나로. `backward()` 의 출발점이다. */
  /**
   * Adds everything. Given `dtype` it converts **before** adding.
   *
   * **Every neighbour took that argument and only this one did not.**
   * `mean`, `prod` and `nansum` are `(dim?, keepdim, dtype?)`, and `cumsum`
   * and `cumprod` take it too. `sumDim`, the axis-taking side, takes it.
   * **The single most-called name among the reductions** was the one
   * missing it, so there was nowhere to carry `torch.sum(x,
   * dtype=torch.float32)` across to.
   *
   * One of four not listening is worse than none of them listening —
   * because no rule can be formed. `norm` had already shown this same place
   * once.
   *
   * **The order decides the value.** Convert first, then add — folding
   * `[1.7, −2.3, 0.9]` to int64 gives −1 truncating first and 0 truncating
   * last. torch does the former.
   */
  sum(dtype?: DType): Tensor {
    if (dtype !== undefined) return this.castFirst(dtype).sum().to(dtype);
    // **복소수는 실수 축약 둘로 쪼갠다.** 합은 실수부와 허수부에 각각 걸리므로
    // 새 커널이 필요 없다 — `real`·`imag`·`complex` 가 이미 있고 셋 다 역방향을 안다.
    //
    // 이 자리가 있어야 "복소 손실의 backward 는 거절" 이 **거절 자리에서** 거절된다.
    // 없으면 그 앞의 `sum()` 이 먼저 막고, 그러면 같은 예외 종류가 나와서 케이스는
    // 통과하는데 정작 물으려던 자리는 안 지난다 — 통과가 증명을 안 하는 모양이다.
    if (this.isComplex()) return Tensor.complex(this.real().sum(), this.imag().sum());
    const out = dev().sumAll(this.buffer, this.size);
    const shape = this.shape;
    return Tensor.make(
      out,
      [],
      [this],
      // d(sum)/dx 는 어디서나 1 이므로 씨앗을 모양대로 펴 준다.
      (g) => [foldFrom(g, shape)],
      "SumBackward0",
      accumulated(this.dtype),
    );
  }

  /**
   * 축 하나를 접는다. `dim` 이 없으면 전부 접어 스칼라로.
   *
   * 전체 합만 `Device.sumAll` 의 트리로 간다 — 축 축약 커널은 스레드 하나가 축을
   * 훑는 구조라 전체 축약에 쓰면 스레드 하나가 n 번 돈다.
   */
  private reduceOver(kind: ReduceKind, dim?: number, keepdim = false): Tensor {
    if (dim === undefined) {
      if (kind === "sum") return this.sum();
      // 전체 최대·최소는 평평하게 본 뒤 축 하나로 접는다.
      return this.flat().reduceOver(kind, 0, false);
    }
    const rank = this.shape.length;
    const axis = dim < 0 ? dim + rank : dim;
    if (axis < 0 || axis >= rank) {
      throw new Error(`dimension out of range: ${dim} (rank ${rank})`);
    }
    const red = this.shape[axis] ?? 1;
    const outer = this.shape.slice(0, axis).reduce((a, b) => a * b, 1);
    const inner = this.shape.slice(axis + 1).reduce((a, b) => a * b, 1);
    const outShape = [...this.shape];
    if (keepdim) outShape[axis] = 1;
    else outShape.splice(axis, 1);

    const n = outer * inner;
    const out = dev().alloc(n);
    const key = `${outer}:${red}:${inner}`;
    dev().run1d(
      dev().pipeline(`rd:${kind}:${key}`, () => reduceDim(kind, outer, red, inner)),
      [this.buffer, out],
      n,
    );
    const result = Tensor.make(
      out,
      outShape,
      [this],
      (g) => {
        const gi = dev().alloc(this.size);
        if (kind === "prod") {
          dev().run1d(
            dev().pipeline(`pb:${key}`, () => prodBackward(outer, red, inner)),
            [this.buffer, g.buffer, gi],
            this.size,
          );
          return [new Tensor(gi, this.shape)];
        }
        if (kind === "sum") {
          dev().run1d(
            dev().pipeline(`xd:${key}`, () => expandDim(outer, red, inner)),
            [g.buffer, gi],
            this.size,
          );
        } else {
          dev().run1d(
            dev().pipeline(`eb:${key}`, () => extremeBackward(outer, red, inner)),
            [this.buffer, result.buffer, g.buffer, gi],
            this.size,
          );
        }
        return [new Tensor(gi, this.shape)];
      },
      kind === "sum" ? "SumBackward1" : "AmaxBackward0",
      kind === "sum" || kind === "prod" ? accumulated(this.dtype) : this.dtype,
    );
    return result;
  }

  /** 같은 버퍼를 1차원으로 본다. 원소 순서가 그대로라 복사가 필요 없다. */
  private flat(): Tensor {
    if (this.shape.length === 1) return this;
    return Tensor.make(
      this.buffer,
      [this.size],
      [this],
      (g) => [new Tensor(g.buffer, this.shape)],
      "ViewBackward0",
      this.dtype,
    );
  }

  amax(dim?: number, keepdim = false): Tensor {
    return this.reduceOver("max", dim, keepdim);
  }

  amin(dim?: number, keepdim = false): Tensor {
    return this.reduceOver("min", dim, keepdim);
  }

  sumDim(dim: number, keepdim = false, dtype?: DType): Tensor {
    if (dtype !== undefined) return this.castFirst(dtype).sumDim(dim, keepdim).to(dtype);
    return this.reduceOver("sum", dim, keepdim);
  }

  /**
   * `dtype=` 를 받은 축약이 맨 앞에서 부른다.
   *
   * **규칙 한 줄이다: 넣기 전에 바꾼다.** 접고 나서가 아니다 — 실측이 그 둘을
   * 가른다: `[1.7, −2.3, 0.9].sum(dtype=int64)` 이 `−1` 이다. 먼저 접으면 `0.3`
   * 이라 깎아도 `0` 인데, 먼저 깎으면 `[1, −2, 0]` 이라 합이 `−1` 이다.
   *
   * 결과 형도 마지막에 못 박는다 — 안 그러면 누적 규칙이 다시 올려서
   * `sum(dtype=bool)` 이 int64 로 나온다(torch 는 `true` 다).
   */
  private castFirst(dtype: DType): Tensor {
    return this.dtype === dtype ? this : this.to(dtype);
  }

  mean(dim?: number, keepdim = false, dtype?: DType): Tensor {
    if (dtype !== undefined) {
      // **정수로 내리라는 것은 거절한다**(실측). `dtype=` 이 푸는 것은 **입력 쪽**
      // 거절뿐이다 — 결과가 정수인 평균은 여전히 답이 없다.
      if (dtype !== "float32" && dtype !== "complex64") {
        throw new RuntimeError(
          "mean(): could not infer output dtype. Input dtype must be either " +
            "a floating point or complex dtype");
      }
      return this.castFirst(dtype).mean(dim, keepdim).to(dtype);
    }
    // **torch 가 멈추는 자리에서 멈춘다**(실측). 나눗셈·제곱근이 정수 칸에
    // 답이 안 들어간다 — numpy 처럼 조용히 실수로 올리면 그 코드가 진짜
    // torch 에서 깨진다.
    this.needsFloat("mean is for floating point only", "mean(): could not infer output dtype. Input dtype must be either a floating point or complex dtype");
    const count = dim === undefined
      ? this.size
      : (this.shape[dim < 0 ? dim + this.shape.length : dim] ?? 1);
    const total = dim === undefined ? this.sum() : this.sumDim(dim, keepdim);
    return total.div(Tensor.full([], count));
  }

  /**
   * A copy cut from the graph. The buffer is **shared** — for places that
   * will only read the value.
   *
   * `logsumexp` uses it to detach the maximum. Undetached, `m` carries its
   * own gradient, and although that share cancels exactly on paper, in
   * floating point it becomes a subtraction of large numbers.
   */
  detach(): Tensor {
    // **형을 물려준다.** 안 물려주면 `x.to("int64").detach().dtype` 이 float32 다 —
    // 값은 같은 버퍼라 안 변하고 이름표만 조용히 갈린다. 복소수에서는 그 이름표가
    // 곧 저장 규약이라 잃으면 뒤쪽 절반이 사라진 것처럼 읽힌다.
    return new Tensor(this.raw, this.shape, {
      requiresGrad: false, dtype: this.dtype,
    });
  }

  /**
   * Cuts the graph at **the same tensor.** torch's `detach_`.
   *
   * `detach()` produces a new one, so the original is still attached
   * upstream — treating the two as the same means backpropagation keeps
   * flowing through `y` even after `y.detach_()`.
   */
  detach_(): Tensor {
    this.requiresGrad = false;
    this.parents = [];
    this.backwardFn = null;
    return this;
  }

  /**
   * `log(Σ exp(x))`. **Computed with the maximum subtracted** — used
   * plainly, the moment x passes 89 float32's exp becomes inf and
   * everything after it is inf.
   *
   * Left as an assembly. The backward falls out of the derivatives of
   * operations that already exist, so no new derivative is written by hand
   * — that was the most frequently wrong place this week.
   */
  logsumexp(dim?: number, keepdim = false): Tensor {
    // **정수·참거짓도 받고 float32 를 낸다**(실측). 실수로 올려 두면 아래 조립이
    // 그대로 돌고, 결과 형도 따라온다 — `logcumsumexp` 는 torch 가 거절하는 쪽이라
    // 여기와 갈린다(규칙이 아니라 torch 의 커널 구멍이다).
    if (this.dtype !== "float32") return this.to("float32").logsumexp(dim, keepdim);
    const m = (dim === undefined ? this.amax() : this.amax(dim, true)).detach();
    const shifted = this.sub(m);
    const summed = dim === undefined
      ? shifted.exp().sum()
      : shifted.exp().sumDim(dim, true);
    const logged = summed.log().add(m);
    if (dim === undefined || keepdim) return logged;
    return logged.squeeze(dim);
  }

  /**
   * A cumulative `logsumexp`. Counted **without overflowing** — the axis
   * maximum is subtracted, added, and restored.
   *
   * Assembled from the same place as `logsumexp`. With the maximum
   * detached, the backward falls straight out of the assembly, and one
   * hand-written derivative is gone.
   */
  logcumsumexp(dim: number): Tensor {
    const big = this.amax(dim, true).detach();
    return this.sub(big).exp().cumsum(dim).log().add(big);
  }

  /**
   * The multivariate log-gamma. `log Γ_p(x) = p(p−1)/4 · log π + Σᵢ log Γ(x
   * + (1−i)/2)`.
   *
   * At `p` of 1 it equals `lgamma` — asked with that value alone, whether
   * the sum runs is invisible.
   */
  mvlgamma(p: number): Tensor {
    let out = this.lgamma();
    for (let i = 2; i <= p; i++) {
      out = out.add(this.add(Tensor.full([], (1 - i) / 2)).lgamma());
    }
    return out.add(Tensor.full([], (p * (p - 1) / 4) * Math.log(Math.PI)));
  }

  /**
   * The mantissa and the exponent. `x = mantissa × 2^exponent`, with the
   * mantissa in [0.5, 1).
   *
   * torch returns the exponent as int32, and storage here is f32 only, so
   * it matches by value.
   */
  frexp(): { mantissa: Tensor; exponent: Tensor } {
    return { mantissa: this.frexpMantissa(), exponent: this.frexpExponent() };
  }

  /**
   * A **new** tensor of the same shape filled with one value. **Not in
   * place** — torch's `fill` is so, and it does something different from
   * `fill_`, one character away (measured).
   */
  fillWith(value: number): Tensor {
    return Tensor.full(this.shape, value);
  }

  /**
   * `‖x − y‖_p`. An assembly, so the backward follows on its own.
   *
   * **It did not take `p` for a long time** — it was always L2, and
   * `dist(a, b, 3)` produced a different value of plausible magnitude. The
   * core had the same place, and both surfaced only through comparing
   * values.
   */
  dist(other: Tensor, p = 2): Tensor {
    const diff = this.sub(other);
    if (p === 2) return diff.square().sum().sqrt();
    if (p === 1) return diff.abs().sum();
    if (p === Number.POSITIVE_INFINITY) return diff.abs().amax();
    if (p === Number.NEGATIVE_INFINITY) return diff.abs().amin();
    if (p === 0) return diff.binary("ne", Tensor.full([], 0), "float32").sum();
    return diff.abs().powScalar(p).sum().powScalar(1 / p);
  }

  /**
   * Adds, treating NaN as 0. No gradient goes to a NaN slot.
   */
  nansum(dim?: number, keepdim = false, dtype?: DType): Tensor {
    if (dtype !== undefined) return this.castFirst(dtype).nansum(dim, keepdim).to(dtype);
    const clean = this.unary("nanToZero");
    return dim === undefined ? clean.sum() : clean.sumDim(dim, keepdim);
  }

  /**
   * Averages excluding NaN. **The count excludes NaN too** — that is what
   * differs from mean.
   */
  nanmean(dim?: number, keepdim = false): Tensor {
    const total = this.nansum(dim, keepdim);
    const present = this.unary("notNan");
    const count = (dim === undefined
      ? present.sum()
      : present.sumDim(dim, keepdim)).detach();
    return total.div(count);
  }

  /**
   * Inserts an axis of size 1.
   */
  unsqueeze(dim: number): Tensor {
    const rank = this.shape.length;
    const axis = dim < 0 ? dim + rank + 1 : dim;
    const shape = [...this.shape];
    shape.splice(axis, 0, 1);
    return this.reshape(shape);
  }

  /**
   * Removes **every** axis of size 1. The argument-free `squeeze()`.
   */
  squeezeAll(): Tensor {
    return this.reshape(this.shape.filter((d) => d !== 1));
  }

  /**
   * A new tensor with the same values. The graph stays connected — that is
   * what differs from `detach`.
   */
  clone(): Tensor {
    return this.unary("positive");
  }

  /**
   * Joins along one axis.
   *
   * **There is no new kernel.** Padding each by the other's size and adding
   * does it — the padded region is zero and they do not overlap, so the sum
   * is the concatenation. The backward is `pad`'s, used as-is. It costs
   * twice the memory, and not making one more hand-written backward is
   * worth more.
   */
  static cat(parts: readonly Tensor[], dim = 0): Tensor {
    if (parts.length === 0) throw new Error("cat got nothing to concatenate");
    const first = parts[0];
    if (!first) throw new Error("cat got nothing to concatenate");
    const rank = first.shape.length;
    const axis = dim < 0 ? dim + rank : dim;
    const sizes = parts.map((p) => p.shape[axis] ?? 0);
    const total = sizes.reduce((a, b) => a + b, 0);
    let before = 0;
    let acc: Tensor | null = null;
    for (const [i, part] of parts.entries()) {
      const size = sizes[i] ?? 0;
      const padded = part.pad(axis, before, total - before - size);
      acc = acc === null ? padded : acc.add(padded);
      before += size;
    }
    if (!acc) throw new Error("cat got nothing to concatenate");
    return acc;
  }

  /**
   * Stacks by making a new axis. The same as `cat` with an axis inserted.
   */
  static stack(parts: readonly Tensor[], dim = 0): Tensor {
    return Tensor.cat(parts.map((p) => p.unsqueeze(dim)), dim);
  }

  /** 모자란 **앞**축을 1 로 채운다. `atleast_2d` 가 하는 일이다. */
  private static lift(t: Tensor, rank: number): Tensor {
    if (t.shape.length >= rank) return t;
    return t.reshape([...new Array<number>(rank - t.shape.length).fill(1), ...t.shape]);
  }

  /**
   * torch 의 `atleast_3d`. **뒤에 축을 붙인다** — 앞이 아니다.
   *
   * 1 차원 `(n,)` 은 `(1, n, 1)` 이고 2 차원 `(m, n)` 은 `(m, n, 1)` 이다. 앞에만
   * 채우면 `dstack` 이 셋째 축이 아니라 마지막 축으로 붙어 모양부터 달라진다.
   */
  private static lift3(t: Tensor): Tensor {
    const shape = t.shape;
    if (shape.length >= 3) return t;
    if (shape.length === 2) return t.reshape([shape[0] ?? 1, shape[1] ?? 1, 1]);
    return t.reshape([1, shape[0] ?? 1, 1]);
  }

  /**
   * One-dimensional inputs are joined end to end; above that,
   * **column-wise.**
   */
  static hstack(parts: readonly Tensor[]): Tensor {
    return Tensor.cat(parts, (parts[0]?.shape.length ?? 1) === 1 ? 0 : 1);
  }

  static vstack(parts: readonly Tensor[]): Tensor {
    return Tensor.cat(parts.map((p) => Tensor.lift(p, 2)), 0);
  }

  static dstack(parts: readonly Tensor[]): Tensor {
    return Tensor.cat(parts.map((p) => Tensor.lift3(p)), 2);
  }

  /**
   * Stands a one-dimensional input up **as a single column** and joins.
   * That is where it parts from `hstack`.
   */
  static columnStack(parts: readonly Tensor[]): Tensor {
    return Tensor.cat(
      parts.map((p) => (p.shape.length === 1 ? p.reshape([p.shape[0] ?? 0, 1]) : p)),
      1,
    );
  }

  /**
   * Blocks laid along the diagonal, zero elsewhere.
   */
  static blockDiag(parts: readonly Tensor[]): Tensor {
    const mats = parts.map((p) => Tensor.lift(p, 2));
    const widths = mats.map((m) => m.shape[1] ?? 0);
    const total = widths.reduce((a, b) => a + b, 0);
    const lines: Tensor[] = [];
    let at = 0;
    mats.forEach((m, i) => {
      const rows = m.shape[0] ?? 0;
      const width = widths[i] ?? 0;
      const pieces: Tensor[] = [];
      if (at) pieces.push(Tensor.zeros([rows, at]));
      pieces.push(m);
      if (total - at - width) pieces.push(Tensor.zeros([rows, total - at - width]));
      lines.push(pieces.length > 1 ? Tensor.cat(pieces, 1) : m);
      at += width;
    });
    return lines.length > 1 ? Tensor.cat(lines, 0) : (lines[0] ?? Tensor.zeros([0, 0]));
  }

  /**
   * Stretches everything to a common shape. **Copies, not views.**
   */
  static broadcastTensors(parts: readonly Tensor[]): Tensor[] {
    let shape: number[] = [];
    for (const p of parts) shape = broadcastShapes(shape, p.shape);
    return parts.map((p) => p.expand(...shape));
  }

  /**
   * A grid. **`xy` has the first two axes swapped**, so one rule cannot
   * cover both.
   *
   * Measured only with `ij`, that branch is invisible — on square input the
   * two have the same shape, and only the values separate them.
   */
  static meshgrid(parts: readonly Tensor[], indexing: "ij" | "xy" = "ij"): Tensor[] {
    if (indexing !== "ij" && indexing !== "xy") {
      throw new RuntimeError(`indexing must be 'ij' or 'xy': ${String(indexing)}`);
    }
    const order = parts.map((_, i) => i);
    if (indexing === "xy" && parts.length >= 2) {
      [order[0], order[1]] = [order[1]!, order[0]!];
    }
    const sizes = order.map((i) => parts[i]?.shape[0] ?? 0);
    const out = order.map((which, place) => {
      const lifted = new Array<number>(parts.length).fill(1);
      lifted[place] = sizes[place] ?? 1;
      return parts[which]!.reshape(lifted).expand(...sizes);
    });
    if (indexing === "xy" && parts.length >= 2) {
      [out[0], out[1]] = [out[1]!, out[0]!];
    }
    return out;
  }

  /**
   * Evenly spaced in powers of `base`. It uses `linspace` as the exponent.
   */
  static logspace(start: number, end: number, steps: number, base = 10.0): Tensor {
    return Tensor.full([], base).binary("pow", Tensor.linspace(start, end, steps));
  }

  /**
   * A tensor of one number. The same as `full([], v)`, given its own name
   * by torch.
   */
  static scalarTensor(value: number): Tensor {
    return Tensor.full([], value);
  }

  /**
   * **It leaves the values at 0** — torch gives you garbage, and that is
   * not a thing to learn.
   */
  static empty(shape: readonly number[]): Tensor {
    return Tensor.zeros(shape);
  }

  /**
   * Variance. **torch's default is the unbiased estimate (dividing by
   * n-1)** — left at `correction=0` the value comes out subtly smaller, and
   * that becomes the place it diverges quietly inside a normalisation
   * layer.
   */
  variance(correction = 1): Tensor {
    // **torch 가 멈추는 자리에서 멈춘다**(실측). 나눗셈·제곱근이 정수 칸에
    // 답이 안 들어간다 — numpy 처럼 조용히 실수로 올리면 그 코드가 진짜
    // torch 에서 깨진다.
    this.needsFloat("variance is for floating point only", "std and var only support floating point and complex dtypes");
    const n = this.size;
    // **평균을 떼도 기울기가 같다.** 평균을 통과하는 몫은 Σ(x−m) 에 비례하는데
    // 그 합이 정의상 0 이라 통째로 사라진다. 이어두면 큰 항 둘이 상쇄되는 계산이
    // 되므로, 떼는 쪽이 값도 더 정확하다.
    const centered = this.sub(this.mean().detach());
    return centered.square().sum().div(Tensor.full([], n - correction));
  }

  std(correction = 1): Tensor {
    // **torch 가 멈추는 자리에서 멈춘다**(실측). 나눗셈·제곱근이 정수 칸에
    // 답이 안 들어간다 — numpy 처럼 조용히 실수로 올리면 그 코드가 진짜
    // torch 에서 깨진다.
    this.needsFloat("std is for floating point only", "std and var only support floating point and complex dtypes");
    return this.variance(correction).sqrt();
  }

  /**
   * Removes an axis of size 1.
   */
  squeeze(dim: number): Tensor {
    const rank = this.shape.length;
    const axis = dim < 0 ? dim + rank : dim;
    if (this.shape[axis] !== 1) {
      throw new Error(`dimension ${dim} is not of size 1: [${this.shape}]`);
    }
    const outShape = [...this.shape];
    outShape.splice(axis, 1);
    const shape = this.shape;
    return Tensor.make(
      this.buffer,
      outShape,
      [this],
      (g) => [new Tensor(g.buffer, shape)],
      "SqueezeBackward0",
      this.dtype,
    );
  }

  // ── 모양 ──────────────────────────────────────────────────────────────

  /** 이 텐서의 연속 스트라이드. 모양 연산이 규칙을 짤 때 쓴다. */
  private strides(): number[] {
    const s: number[] = new Array<number>(this.shape.length).fill(1);
    for (let d = this.shape.length - 2; d >= 0; d--) {
      s[d] = (s[d + 1] ?? 1) * (this.shape[d + 1] ?? 1);
    }
    return s;
  }

  /**
   * 규칙대로 값을 모아 새 텐서를 만든다. 모양 연산이 전부 이리로 온다.
   *
   * 지금은 **실제로 옮겨 담는다.** 뷰로 두면 복사가 없어 빠르지만, 뷰가 생기는 순간
   * 제자리 연산이 어디까지 번지는지를 정해야 하고 그것은 아직 정할 때가 아니다.
   */
  private viewAs(
    rules: readonly AxisRule[],
    offset: number,
    outShape: readonly number[],
    gradName: string,
  ): Tensor {
    const n = outShape.reduce((a, b) => a * b, 1);
    // **빈 것도 답이다.** `x[5:99]` 처럼 범위 밖을 자르면 원소가 0 개인데, 셰이더는
    // 그 수로 나누므로 WGSL 이 "0 으로 나눈다" 며 통째로 거절한다. 그러면 명령 버퍼가
    // 같이 무효가 되어 **이 자리는 통과하고 뒤에 줄 선 것이 대신 틀린다** — 실제로
    // 그 다음 검사(`randn`)가 전부 0 을 받아서 드러났다.
    //
    // `indexSelect` 가 같은 갈래를 이미 막고 있다. 여기만 안 막혀 있었다 — 자르기로
    // 빈 것을 만드는 길이 그때는 없었기 때문이다.
    if (n === 0) {
      return new Tensor(dev().alloc(0), outShape, { dtype: this.dtype });
    }
    const key = ruleKey(rules, offset);
    const out = dev().alloc(n);
    dev().run1d(dev().pipeline(`gt:${key}`, () => gather(rules, offset)), [this.buffer, out], n);
    const inSize = this.size;
    const inShape = this.shape;
    return Tensor.make(
      out,
      outShape,
      [this],
      (g) => {
        const gi = dev().alloc(inSize);
        dev().run1d(
          dev().pipeline(`gb:${gradName}:${key}|${inSize}`,
                        () => gatherBackward(rules, offset, inSize)),
          [g.buffer, gi],
          inSize,
        );
        return [new Tensor(gi, inShape)];
      },
      gradName,
      this.dtype,
    );
  }

  /**
   * Views the same buffer as a different shape. The element order does not
   * change, so no kernel is needed.
   *
   * **`-1` means "the rest".** It is the form torch code always uses —
   * `x.reshape(-1)`, `x.view(n, -1)` — and it was missing here: the
   * size-matching check caught it and it stopped with `shape '[-1]' is
   * invalid`. It can be used in one position only, and must divide the rest
   * evenly.
   */
  reshape(want: readonly number[]): Tensor {
    const hole = want.indexOf(-1);
    let shape = want;
    if (hole >= 0) {
      if (want.indexOf(-1, hole + 1) >= 0) {
        throw new RuntimeError("only one dimension can be inferred");
      }
      const rest = want.reduce((a, b) => (b === -1 ? a : a * b), 1);
      if (rest <= 0 || this.size % rest !== 0) {
        throw new RuntimeError(
          `shape '[${want}]' ${TORCH.reshapeSize} ${this.size}`,
        );
      }
      shape = [...want.slice(0, hole), this.size / rest, ...want.slice(hole + 1)];
    }
    const n = shape.reduce((a, b) => a * b, 1);
    if (n !== this.size) {
      throw new RuntimeError(
        `shape '[${want}]' ${TORCH.reshapeSize} ${this.size}`,
      );
    }
    const from = this.shape;
    // **복소수도 지난다.** 이 연산은 칸을 안 옮기고 이름표만 바꾸므로 인터리브
    // 저장에 그대로 맞다 — 그래서 `raw` 로 들어온다. 칸을 **옮기는** 모양 연산
    // (`cat`·`select`·`transpose`…)은 f32 단위로 옮겨서 실·허가 어긋나므로
    // `buffer` 게터가 계속 막는다. 둘을 한 묶음으로 보면 안 된다.
    const dt = this.dtype;
    return Tensor.make(
      this.raw,
      shape,
      [this],
      (g) => [new Tensor(g.raw, from, { dtype: dt })],
      "ViewBackward0",
      dt,
    );
  }

  ravel(): Tensor {
    return this.reshape([this.size]);
  }

  /**
   * Spreads one axis into several.
   */
  unflatten(dim: number, sizes: readonly number[]): Tensor {
    const rank = this.shape.length;
    const axis = dim < 0 ? dim + rank : dim;
    const shape = [...this.shape.slice(0, axis), ...sizes, ...this.shape.slice(axis + 1)];
    return this.reshape(shape);
  }

  /**
   * At least two-dimensional. Already two or more, and it is unchanged.
   */
  atleast2d(): Tensor {
    if (this.shape.length >= 2) return this;
    if (this.shape.length === 1) return this.reshape([1, this.shape[0] ?? 0]);
    return this.reshape([1, 1]);
  }

  /**
   * Stretches axes of size 1. `-1` means "leave this one alone".
   *
   * Axes can also be added in front — those axes have stride 0, so
   * **nothing is replicated.**
   */
  expand(...sizes: number[]): Tensor {
    const rank = sizes.length;
    if (rank < this.shape.length) {
      throw new Error(`expand cannot shrink a dimension: [${this.shape}] -> [${sizes}]`);
    }
    const own = this.strides();
    const rules: AxisRule[] = [];
    const outShape: number[] = [];
    for (let i = 0; i < rank; i++) {
      const src = this.shape.length - rank + i;
      const dim = src >= 0 ? (this.shape[src] ?? 1) : 1;
      const want = sizes[i] ?? -1;
      const size = want === -1 ? dim : want;
      if (want !== -1 && dim !== 1 && want !== dim) {
        throw new Error(`dimension ${i} is ${dim} and cannot expand to ${want}.`);
      }
      const stride = src >= 0 && dim !== 1 ? (own[src] ?? 1) : 0;
      rules.push({ size, stride, kind: "lin", wrap: size });
      outShape.push(size);
    }
    return this.viewAs(rules, 0, outShape, "ExpandBackward0");
  }

  /**
   * Repeats an integer number of times per axis. Unlike `expand`, it really
   * does become several copies.
   */
  repeat(...times: number[]): Tensor {
    const rank = times.length;
    if (rank < this.shape.length) {
      throw new Error(`repeat cannot shrink a dimension: [${this.shape}] -> [${times}]`);
    }
    const own = this.strides();
    const rules: AxisRule[] = [];
    const outShape: number[] = [];
    for (let i = 0; i < rank; i++) {
      const src = this.shape.length - rank + i;
      const dim = src >= 0 ? (this.shape[src] ?? 1) : 1;
      const k = times[i] ?? 1;
      rules.push({
        size: dim * k,
        stride: src >= 0 ? (own[src] ?? 1) : 0,
        kind: "mod",
        wrap: dim,
      });
      outShape.push(dim * k);
    }
    return this.viewAs(rules, 0, outShape, "RepeatBackward0");
  }

  /**
   * Swaps two axes. The same as `swapdims` — torch carries both names.
   */
  swapaxes(a: number, b: number): Tensor {
    const rank = this.shape.length;
    const i = a < 0 ? a + rank : a;
    const j = b < 0 ? b + rank : b;
    const own = this.strides();
    const order = [...Array(rank).keys()];
    order[i] = j;
    order[j] = i;
    const rules: AxisRule[] = order.map((src) => ({
      size: this.shape[src] ?? 1,
      stride: own[src] ?? 1,
      kind: "lin" as const,
      wrap: this.shape[src] ?? 1,
    }));
    return this.viewAs(rules, 0, order.map((src) => this.shape[src] ?? 1), "TransposeBackward0");
  }

  /**
   * Picks one position along an axis and removes that axis.
   */
  select(dim: number, index: number): Tensor {
    const rank = this.shape.length;
    const axis = dim < 0 ? dim + rank : dim;
    const size = this.shape[axis] ?? 0;
    const at = index < 0 ? index + size : index;
    if (at < 0 || at >= size) {
      // 범위를 넘겨 읽으면 WGSL 은 던지지 않고 **가장자리 값이나 0 을 준다.**
      // 조용히 틀린 값을 내는 대신 여기서 멈춘다.
      throw new IndexError(
        `index ${index} is out of bounds for dimension ${dim} with size ${size}`,
      );
    }
    const own = this.strides();
    const offset = at * (own[axis] ?? 1);
    const rules: AxisRule[] = [];
    const outShape: number[] = [];
    for (let d = 0; d < rank; d++) {
      if (d === axis) continue;
      rules.push({
        size: this.shape[d] ?? 1,
        stride: own[d] ?? 1,
        kind: "lin",
        wrap: this.shape[d] ?? 1,
      });
      outShape.push(this.shape[d] ?? 1);
    }
    return this.viewAs(rules, offset, outShape, "SelectBackward0");
  }

  /**
   * The diagonal. A positive `offset` takes the one above, a negative one
   * the diagonal below.
   *
   * **Which two axes are read diverges.** `torch.diagonal` takes the first
   * two (`0, 1`) and `torch.linalg.diagonal` the last two (`-2, -1`) —
   * given three dimensions, `(2,3,4)` becomes `(4,2)` and `(2,3)`
   * respectively. The names are close enough to read as the same thing
   * while even the shape differs. So the axes are taken as arguments.
   *
   * **The extracted axis goes to the back.** The remaining axes come first
   * and the diagonal last — as in torch.
   */
  diagonal(offset = 0, dim1 = 0, dim2 = 1): Tensor {
    const rank = this.shape.length;
    if (rank < 2) {
      throw new Error(`diagonal needs two or more dimensions: [${this.shape}]`);
    }
    const a1 = dim1 < 0 ? dim1 + rank : dim1;
    const a2 = dim2 < 0 ? dim2 + rank : dim2;
    const rows = this.shape[a1] ?? 0;
    const cols = this.shape[a2] ?? 0;
    const own = this.strides();
    const rowStride = own[a1] ?? 1;
    const colStride = own[a2] ?? 1;
    const start = offset >= 0 ? offset * colStride : -offset * rowStride;
    const length = offset >= 0
      ? Math.max(0, Math.min(rows, cols - offset))
      : Math.max(0, Math.min(rows + offset, cols));
    const rest = [...Array(rank).keys()].filter((i) => i !== a1 && i !== a2);
    const rules: AxisRule[] = rest.map((i) => ({
      size: this.shape[i] ?? 1, stride: own[i] ?? 1, kind: "lin" as const,
      wrap: this.shape[i] ?? 1,
    }));
    // 한 걸음에 행과 열이 같이 하나씩 간다 — 그래서 걸음이 둘의 합이다.
    rules.push({
      size: length, stride: rowStride + colStride, kind: "lin", wrap: length,
    });
    const outShape = [...rest.map((i) => this.shape[i] ?? 1), length];
    return this.viewAs(rules, start, outShape, "DiagonalBackward0");
  }

  /**
   * A square matrix with the vector laid on its diagonal.
   *
   * @param offset which diagonal. Non-zero, **the matrix grows by that
   *   much** — its side is `n+|offset|`, so rather than calling the kernel
   *   again it is placed on the larger plate and moved.
   */
  diagflat(offset = 0): Tensor {
    if (offset !== 0) {
      // **앞에 0 을 채우고 굴린다.** 앞에 `k` 개를 채우면 값이 `(i+k, i+k)` 에
      // 놓이는데, 위쪽 대각선은 행을 `k` 만큼 당기면 `(i, i+k)` 가 되고 아래쪽은
      // 열을 당기면 `(i+k, i)` 가 된다. 굴림이 감아 넘기는 자리는 채운 0 이라
      // 해가 없다 — 그래서 커널을 새로 안 쓴다.
      const k = Math.abs(offset);
      const wide = this.padND([k, 0]).diagflat();
      return wide.roll(-k, offset > 0 ? 0 : 1);
    }
    // **복소수는 실수 둘로 쪼갠다.** 대각에 놓는 일은 값을 안 건드리므로 실수부와
    // 허수부에 따로 걸면 그대로다 — 새 커널이 필요 없다(`sum` 이 같은 자리를 같은
    // 방법으로 지난다). `linalg.eig` 의 `V·diag(λ)` 가 이 길로 들어온다.
    if (this.isComplex()) {
      return Tensor.complex(this.real().diagflat(), this.imag().diagflat());
    }
    const n = this.size;
    const out = dev().alloc(n * n);
    dev().run1d(dev().pipeline(`df:${n}`, () => diagflat(n)), [this.buffer, out], n * n);
    const shape = this.shape;
    return Tensor.make(
      out,
      [n, n],
      [this],
      (g) => {
        const gi = dev().alloc(n);
        dev().run1d(
          dev().pipeline(`dfb:${n}`, () => diagflatBackward(n)),
          [g.buffer, gi],
          n,
        );
        return [new Tensor(gi, shape)];
      },
      "DiagflatBackward0",
      this.dtype,
    );
  }

  /**
   * Reverses one axis.
   */
  flip(dim: number): Tensor {
    const rank = this.shape.length;
    const axis = dim < 0 ? dim + rank : dim;
    const own = this.strides();
    const rules: AxisRule[] = this.shape.map((size, d) => ({
      size,
      stride: own[d] ?? 1,
      kind: d === axis ? ("rev" as const) : ("lin" as const),
      wrap: size,
    }));
    return this.viewAs(rules, 0, this.shape, "FlipBackward0");
  }

  fliplr(): Tensor {
    return this.flip(1);
  }

  flipud(): Tensor {
    return this.flip(0);
  }

  /**
   * Rotates by 90° within a two-dimensional plane.
   *
   * At `k=1` it is `out[i][j] = in[j][C-1-i]` — swapping the axes while
   * reversing one, which writes down directly as a rule table.
   */
  rot90(k = 1): Tensor {
    if (this.shape.length !== 2) {
      throw new Error(`rot90 is 2-D only for now: [${this.shape}]`);
    }
    const turns = ((k % 4) + 4) % 4;
    if (turns === 0) return this.reshape(this.shape);
    const rows = this.shape[0] ?? 0;
    const cols = this.shape[1] ?? 0;
    const own = this.strides();
    const rowStride = own[0] ?? 1;
    const colStride = own[1] ?? 1;
    if (turns === 2) {
      const rules: AxisRule[] = [
        { size: rows, stride: rowStride, kind: "rev", wrap: rows },
        { size: cols, stride: colStride, kind: "rev", wrap: cols },
      ];
      return this.viewAs(rules, 0, [rows, cols], "Rot90Backward0");
    }
    if (turns === 1) {
      const rules: AxisRule[] = [
        { size: cols, stride: colStride, kind: "rev", wrap: cols },
        { size: rows, stride: rowStride, kind: "lin", wrap: rows },
      ];
      return this.viewAs(rules, 0, [cols, rows], "Rot90Backward0");
    }
    const rules: AxisRule[] = [
      { size: cols, stride: colStride, kind: "lin", wrap: cols },
      { size: rows, stride: rowStride, kind: "rev", wrap: rows },
    ];
    return this.viewAs(rules, 0, [cols, rows], "Rot90Backward0");
  }

  /**
   * A sliding window. With a step smaller than the window, the windows
   * overlap.
   *
   * **Overlaps accumulate in the backward** — spreading a length of 5 with
   * `unfold(0, 3, 1)` gives a gradient of `[1,2,3,2,1]`. Without the
   * addition they would all be 1, and a value check alone does not catch
   * it.
   */
  unfold(dim: number, size: number, step: number): Tensor {
    const rank = this.shape.length;
    const axis = dim < 0 ? dim + rank : dim;
    const own = this.strides();
    const axisSize = this.shape[axis] ?? 0;
    const windows = Math.floor((axisSize - size) / step) + 1;
    if (windows < 1) {
      throw new Error(`no window fits: size ${size}, step ${step}, over length ${axisSize}.`);
    }
    const axisStride = own[axis] ?? 1;
    const rules: AxisRule[] = [];
    const outShape: number[] = [];
    for (let d = 0; d < rank; d++) {
      const dim_ = d === axis ? windows : (this.shape[d] ?? 1);
      const stride = d === axis ? axisStride * step : (own[d] ?? 1);
      rules.push({ size: dim_, stride, kind: "lin", wrap: dim_ });
      outShape.push(dim_);
    }
    // 창 안쪽이 맨 뒤 축으로 붙는다.
    rules.push({ size, stride: axisStride, kind: "lin", wrap: size });
    outShape.push(size);
    return this.viewAs(rules, 0, outShape, "UnfoldBackward0");
  }

  /**
   * Divides one axis into equally sized pieces. Each piece is a new tensor.
   */
  split(dim: number, parts: number): Tensor[] {
    const rank = this.shape.length;
    const axis = dim < 0 ? dim + rank : dim;
    const axisSize = this.shape[axis] ?? 0;
    if (axisSize % parts !== 0) {
      throw new Error(`dimension ${dim} of size ${axisSize} does not divide into ${parts}.`);
    }
    const each = axisSize / parts;
    const own = this.strides();
    const out: Tensor[] = [];
    for (let k = 0; k < parts; k++) {
      const rules: AxisRule[] = this.shape.map((size, d) => ({
        size: d === axis ? each : size,
        stride: own[d] ?? 1,
        kind: "lin" as const,
        wrap: d === axis ? each : size,
      }));
      const outShape = this.shape.map((size, d) => (d === axis ? each : size));
      out.push(this.viewAs(rules, k * each * (own[axis] ?? 1), outShape, "SliceBackward0"));
    }
    return out;
  }

  /**
   * **The remainder is shared out from the front.** Splitting 10 into 4
   * gives 3·3·2·2 (measured).
   *
   * Different from `chunk` — that one fills the front pieces fully and the
   * last takes what is left. Measured only at sizes that divide evenly, the
   * two functions look the same.
   */
  tensorSplit(sections: number | readonly number[], dim = 0): Tensor[] {
    const rank = this.shape.length;
    const axis = dim < 0 ? dim + rank : dim;
    const len = this.shape[axis] ?? 0;
    if (Array.isArray(sections)) {
      return this.splitAt(sections as readonly number[], axis);
    }
    const k = sections as number;
    const base = Math.floor(len / k);
    const extra = len % k;
    const cuts: number[] = [];
    let at = 0;
    for (let i = 0; i < k - 1; i++) {
      at += base + (i < extra ? 1 : 0);
      cuts.push(at);
    }
    return this.splitAt(cuts, axis);
  }

  /**
   * Splits by **a list of piece sizes.** `tensorSplit` takes the
   * **positions** to cut at.
   */
  splitWithSizes(sizes: readonly number[], dim = 0): Tensor[] {
    const rank = this.shape.length;
    const axis = dim < 0 ? dim + rank : dim;
    const cuts: number[] = [];
    let at = 0;
    for (let i = 0; i < sizes.length - 1; i++) {
      at += sizes[i]!;
      cuts.push(at);
    }
    return this.splitAt(cuts, axis);
  }

  /** 자르는 **자리 목록**으로 쪼갠다. 두 쪼개기가 같은 밑동을 쓴다. */
  private splitAt(cuts: readonly number[], axis: number): Tensor[] {
    const len = this.shape[axis] ?? 0;
    const out: Tensor[] = [];
    let prev = 0;
    for (const stop of [...cuts, len]) {
      out.push(this.narrow(axis, prev, Math.max(0, stop - prev)));
      prev = stop;
    }
    return out;
  }

  hsplit(parts: number): Tensor[] {
    return this.split(1, parts);
  }

  vsplit(parts: number): Tensor[] {
    return this.split(0, parts);
  }

  /**
   * Moves an axis to a chosen position. Unlike `swapaxes`, it preserves the
   * order of the rest.
   */
  movedim(src: number, dst: number): Tensor {
    const rank = this.shape.length;
    const from = src < 0 ? src + rank : src;
    const to = dst < 0 ? dst + rank : dst;
    const order = [...Array(rank).keys()].filter((d) => d !== from);
    order.splice(to, 0, from);
    return this.permute(order);
  }

  /**
   * Reorders the axes wholesale.
   */
  permute(order: readonly number[]): Tensor {
    const own = this.strides();
    const rules: AxisRule[] = order.map((s) => ({
      size: this.shape[s] ?? 1,
      stride: own[s] ?? 1,
      kind: "lin" as const,
      wrap: this.shape[s] ?? 1,
    }));
    return this.viewAs(rules, 0, order.map((s) => this.shape[s] ?? 1), "PermuteBackward0");
  }

  /**
   * Only `length` entries from `start` along one axis.
   */
  narrow(dim: number, start: number, length: number): Tensor {
    const rank = this.shape.length;
    const axis = dim < 0 ? dim + rank : dim;
    const own = this.strides();
    const rules: AxisRule[] = this.shape.map((size, d) => ({
      size: d === axis ? length : size,
      stride: own[d] ?? 1,
      kind: "lin" as const,
      wrap: d === axis ? length : size,
    }));
    const outShape = this.shape.map((size, d) => (d === axis ? length : size));
    return this.viewAs(rules, start * (own[axis] ?? 1), outShape, "SliceBackward0");
  }

  /**
   * Shifts one axis. **What falls off the end comes back at the front.**
   *
   * It is `out[i] = in[(i - shift) mod n]`, so it is the rule table's `mod`
   * with a shift laid on top.
   */
  roll(shift: number, dim = 0): Tensor {
    const rank = this.shape.length;
    const axis = dim < 0 ? dim + rank : dim;
    const size = this.shape[axis] ?? 1;
    const own = this.strides();
    const bias = ((-shift % size) + size) % size;
    const rules: AxisRule[] = this.shape.map((s, d) => ({
      size: s,
      stride: own[d] ?? 1,
      kind: d === axis ? ("mod" as const) : ("lin" as const),
      wrap: s,
      ...(d === axis ? { bias } : {}),
    }));
    return this.viewAs(rules, 0, this.shape, "RollBackward0");
  }

  /**
   * The same work as `repeat`, but torch carries both names.
   */
  tile(...times: number[]): Tensor {
    return this.repeat(...times);
  }

  /**
   * Divides one axis by size. `split` takes the piece **size**, `chunk` the
   * piece **count**.
   */
  splitSize(dim: number, size: number): Tensor[] {
    const rank = this.shape.length;
    const axis = dim < 0 ? dim + rank : dim;
    const axisSize = this.shape[axis] ?? 0;
    const out: Tensor[] = [];
    for (let start = 0; start < axisSize; start += size) {
      out.push(this.narrow(axis, start, Math.min(size, axisSize - start)));
    }
    return out;
  }

  chunk(parts: number, dim = 0): Tensor[] {
    const rank = this.shape.length;
    const axis = dim < 0 ? dim + rank : dim;
    const axisSize = this.shape[axis] ?? 0;
    return this.splitSize(axis, Math.ceil(axisSize / parts));
  }

  /**
   * Tears an axis apart into singles. That axis disappears.
   */
  unbind(dim = 0): Tensor[] {
    const rank = this.shape.length;
    const axis = dim < 0 ? dim + rank : dim;
    const size = this.shape[axis] ?? 0;
    return Array.from({ length: size }, (_, i) => this.select(axis, i));
  }

  /**
   * The lower triangle. Zeroes everything above `diagonal`.
   */
  tril(diagonal = 0): Tensor {
    return this.triangleAs(true, diagonal);
  }

  triu(diagonal = 0): Tensor {
    return this.triangleAs(false, diagonal);
  }

  private triangleAs(lower: boolean, diagonal: number): Tensor {
    if (this.shape.length !== 2) {
      throw new Error(`tril/triu is 2-D: [${this.shape}]`);
    }
    const rows = this.shape[0] ?? 0;
    const cols = this.shape[1] ?? 0;
    const n = rows * cols;
    const key = `${lower ? "l" : "u"}:${rows}:${cols}:${diagonal}`;
    const out = dev().alloc(n);
    dev().run1d(
      dev().pipeline(`tri:${key}`, () => triangle(rows, cols, lower, diagonal)),
      [this.buffer, out],
      n,
    );
    const shape = this.shape;
    return Tensor.make(
      out,
      shape,
      [this],
      // 남긴 자리로만 흐른다. 0 으로 만든 자리는 결과에 안 들어갔다.
      (g) => {
        const gi = dev().alloc(n);
        dev().run1d(
          dev().pipeline(`tri:${key}`, () => triangle(rows, cols, lower, diagonal)),
          [g.buffer, gi],
          n,
        );
        return [new Tensor(gi, shape)];
      },
      lower ? "TrilBackward0" : "TriuBackward0",
      this.dtype,
    );
  }

  /**
   * The sum of the diagonal.
   */
  trace(): Tensor {
    return this.diagonal().sum();
  }

  /**
   * Two-dimensional, it extracts the diagonal; one-dimensional, it lays one
   * out — torch's `diag`.
   *
   * @param diagonal which diagonal. Positive is above, negative below.
   */
  diag(diagonal = 0): Tensor {
    return this.shape.length === 2
      ? this.diagonal(diagonal)
      : this.diagflat(diagonal);
  }

  /**
   * Accumulates along one axis.
   */
  cumsum(dim = 0, dtype?: DType): Tensor {
    if (dtype !== undefined) {
      noBoolAccumulate("cumsum", dtype);
      return this.castFirst(dtype).cumsum(dim).to(dtype);
    }
    return this.scan("sum", dim);
  }

  cumprod(dim = 0, dtype?: DType): Tensor {
    if (dtype !== undefined) {
      noBoolAccumulate("cumprod", dtype);
      return this.castFirst(dtype).cumprod(dim).to(dtype);
    }
    return this.scan("prod", dim);
  }

  private scan(kind: "sum" | "prod", dim: number): Tensor {
    const rank = this.shape.length;
    const axis = dim < 0 ? dim + rank : dim;
    const len = this.shape[axis] ?? 1;
    const outer = this.shape.slice(0, axis).reduce((a, b) => a * b, 1);
    const inner = this.shape.slice(axis + 1).reduce((a, b) => a * b, 1);
    const n = this.size;
    const key = `${outer}:${len}:${inner}`;
    const out = dev().alloc(n);
    dev().run1d(
      dev().pipeline(`cum:${kind}:${key}`, () => cumulative(kind, outer, len, inner)),
      [this.buffer, out],
      n,
    );
    const shape = this.shape;
    return Tensor.make(
      out,
      shape,
      [this],
      (g) => {
        const gi = dev().alloc(n);
        if (kind === "sum") {
          dev().run1d(
            dev().pipeline(`cumb:${key}`, () => cumsumBackward(outer, len, inner)),
            [g.buffer, gi],
            n,
          );
        } else {
          dev().run1d(
            dev().pipeline(`cumpb:${key}`, () => cumprodBackward(outer, len, inner)),
            [this.buffer, g.buffer, gi],
            n,
          );
        }
        return [new Tensor(gi, shape)];
      },
      kind === "sum" ? "CumsumBackward0" : "CumprodBackward0",
      accumulated(this.dtype),
    );
  }

  /**
   * Selects along one axis as an index tensor points. The indices arrive
   * held in float32.
   */
  gather(dim: number, index: Tensor): Tensor {
    const rank = this.shape.length;
    const axis = dim < 0 ? dim + rank : dim;
    const outer = this.shape.slice(0, axis).reduce((a, b) => a * b, 1);
    const inner = this.shape.slice(axis + 1).reduce((a, b) => a * b, 1);
    const axisSize = this.shape[axis] ?? 1;
    const outAxis = index.shape[axis] ?? 1;
    const n = index.size;
    const out = dev().alloc(n);
    dev().run1d(
      dev().pipeline(
        `gi:${outer}:${axisSize}:${inner}:${outAxis}`,
        () => gatherIndex(outer, axisSize, inner, outAxis),
      ),
      [this.buffer, index.buffer, out],
      n,
    );
    const shape = this.shape;
    return Tensor.make(
      out,
      index.shape,
      [this],
      (g) => {
        const gi = dev().alloc(this.size);
        dev().run1d(
          dev().pipeline(
            `gib:${outer}:${axisSize}:${inner}:${outAxis}`,
            () => gatherIndexBackward(outer, axisSize, inner, outAxis),
          ),
          [index.buffer, g.buffer, gi],
          this.size,
        );
        return [new Tensor(gi, shape)];
      },
      "GatherBackward0",
      this.dtype,
    );
  }

  /**
   * Where the brackets go. One door for carrying `x[...]` across — the
   * syntax and the reasoning are in `indexing.ts`.
   *
   * ```ts
   * x.at(0)                     // x[0]           the axis disappears
   * x.at([null, 1])             // x[:, 1]        null is Python's `:`
   * x.at(slice(1, 3))           // x[1:3]         the axis remains
   * x.at([0, slice(1, 3)])      // x[0, 1:3]
   * x.at(slice(null, null, 2))  // x[::2]
   * x.at([[0, 2]])              // x[[0, 2]]      two brackets — numpy's shape
   * x.at(idx)                   // x[idx]         an int64 tensor
   * ```
   *
   * **The outermost array is always the list of axes.** Give fewer and the
   * remaining axes come whole.
   *
   * ## No values are made here
   *
   * Everything is handed to existing methods — an integer to `select`, a
   * contiguous span to `narrow`, a strided one or an index list to
   * `indexSelect`. There is no new kernel, so **the golden cases already
   * protect those values.** What this method is responsible for is not the
   * value but **which door it is sent through.**
   *
   * ## Boolean masks are not accepted
   *
   * `x[mask]` remains `await x.maskedSelect(mask)`. The length of the
   * result **depends on the values**, so it takes one read off the GPU to
   * know, and making `at()` asynchronous for that alone would put an
   * `await` on every other use for no reason. It is the same reason
   * `unique`, `nonzero` and `bincount` are asynchronous, and keeping it in
   * the same place is better.
   */
  at(index: AtIndex | readonly AtIndex[]): Tensor {
    const list: readonly AtIndex[] = Array.isArray(index)
      && !isSlice(index) && !(index instanceof Tensor)
      ? index as readonly AtIndex[]
      : [index as AtIndex];
    if (list.length > this.shape.length) {
      throw new RuntimeError(
        `too many indices for tensor of dimension ${this.shape.length}: ` +
          `got ${list.length}`,
      );
    }
    let out: Tensor = this;
    // **축 번호가 밀린다.** 정수 인덱스는 축을 없애므로, 그 뒤의 인덱스는 원래
    // 자리보다 한 칸 앞을 가리킨다. 그래서 살아남은 축만 세는 자리를 따로 든다.
    let axis = 0;
    for (const [given, one] of list.entries()) {
      if (one instanceof Tensor) {
        out = out.indexSelect(axis, one);
        axis += 1;
        continue;
      }
      const plan = planAxis(one, out.shape[axis] ?? 0, given);
      out = applyPlan(out, axis, plan);
      if (plan.kind !== "int") axis += 1;
    }
    return out;
  }

  /**
   * 실수만 받는 자리에서 멈춘다. **torch 가 멈추는 곳에서 멈추는 것이 규칙이다** —
   * 관대한 쪽도 갈리는 것이고, 그 코드는 진짜 torch 에서 나중에 깨진다.
   */
  private needsFloat(what: string, phrase: string): void {
    if (this.dtype !== "float32") {
      throw new RuntimeError(`${what} — call \`.to("float32")\` first.\n(torch: ${phrase})`);
    }
  }

  /**
   * Where each of `values` would go inside the **sorted** boundaries
   * (`this`).
   *
   * It is `torch.searchsorted(sorted_sequence, values)`, and since the
   * receiver is the boundaries, here it is a method on the boundaries. The
   * answer is `int64` and its shape follows `values`.
   *
   * **Which way a tie goes is half of this function.** The default is left
   * — where the same value already exists, it stands **before** it (the
   * count of things smaller than me). Given `right` it stands after. The
   * Python side takes this under **two names**, `right` (boolean) and
   * `side` (string), and reconciling those two happens there — this side
   * knows only one.
   *
   * **It does not check that the input is sorted.** torch does not either
   * (checking costs another O(n) pass, and anywhere you would pay that you
   * would not be using this function). Given something unsorted the answer
   * does not quietly lose meaning — the binary search simply produces some
   * position.
   */
  searchSorted(values: Tensor, right = false): Tensor {
    const nSeq = this.size;
    const nVal = values.size;
    if (nVal === 0) {
      return new Tensor(dev().alloc(0), [...values.shape], { dtype: "int64" });
    }
    const out = dev().alloc(nVal);
    dev().run1d(
      dev().pipeline(
        `ss:${nSeq}:${nVal}:${right ? "r" : "l"}`,
        () => searchSorted(nSeq, nVal, right),
      ),
      [this.buffer, values.buffer, out],
      nVal,
    );
    // **자리는 값이 아니다** — 기울기가 안 흐른다. torch 도 그렇다.
    return new Tensor(out, [...values.shape], { dtype: "int64" });
  }

  /**
   * `searchSorted` **with the receiver flipped.** That is the whole
   * difference between the two names.
   *
   * torch has `bucketize(input, boundaries)` taking the values first, so
   * the values are the receiver here too. The computation exists only on
   * the other side — two copies and one of them gets fixed alone.
   */
  bucketize(boundaries: Tensor, right = false): Tensor {
    return boundaries.searchSorted(this, right);
  }

  /**
   * An index **vector** selects along one axis. Unlike `gather`, the index
   * does not differ per slot.
   */
  indexSelect(dim: number, index: Tensor): Tensor {
    const rank = this.shape.length;
    const axis = dim < 0 ? dim + rank : dim;
    const outer = this.shape.slice(0, axis).reduce((a, b) => a * b, 1);
    const inner = this.shape.slice(axis + 1).reduce((a, b) => a * b, 1);
    const axisSize = this.shape[axis] ?? 1;
    const count = index.size;
    const outShape = this.shape.map((s, d) => (d === axis ? count : s));
    const n = outer * count * inner;
    // **하나도 안 고르는 것이 정상이다.** `masked_select` 로 아무것도 안 걸리면
    // 여기 개수가 0 인데, 셰이더는 그 수로 나누므로 WGSL 이 "0 으로 나눈다" 며 통째로
    // 거절한다. 그러면 **이 케이스는 통과하고 다음 케이스가 대신 틀린다** — 명령
    // 버퍼가 같이 무효가 되기 때문이다. 그 앞에서 빈 텐서로 끝낸다.
    if (n === 0) {
      return new Tensor(dev().alloc(0), outShape, { dtype: this.dtype });
    }
    const key = `${outer}:${axisSize}:${inner}:${count}`;
    const out = dev().alloc(n);
    dev().run1d(
      dev().pipeline(`is:${key}`, () => indexSelect(outer, axisSize, inner, count)),
      [this.buffer, index.buffer, out],
      n,
    );
    const shape = this.shape;
    return Tensor.make(
      out,
      outShape,
      [this],
      (g) => {
        const gi = dev().alloc(this.size);
        dev().run1d(
          dev().pipeline(
            `isb:${key}`,
            () => indexSelectBackward(outer, axisSize, inner, count),
          ),
          [index.buffer, g.buffer, gi],
          this.size,
        );
        return [new Tensor(gi, shape)];
      },
      "IndexSelectBackward0",
      this.dtype,
    );
  }

  /**
   * Repeats per slot. `[a,b]` twice each gives `[a,a,b,b]` — different from
   * `tile`.
   */
  repeatInterleave(times: number, dim = 0): Tensor {
    const rank = this.shape.length;
    const axis = dim < 0 ? dim + rank : dim;
    const own = this.strides();
    const rules: AxisRule[] = this.shape.map((size, d) => ({
      size: d === axis ? size * times : size,
      stride: own[d] ?? 1,
      kind: d === axis ? ("div" as const) : ("lin" as const),
      wrap: d === axis ? times : size,
    }));
    const outShape = this.shape.map((s, d) => (d === axis ? s * times : s));
    return this.viewAs(rules, 0, outShape, "RepeatInterleaveBackward0");
  }

  /**
   * The difference between neighbours. Repeated `n` times it shortens by
   * that much.
   *
   * Prepending one makes the result **the same length** as the input — the
   * place used to avoid losing the first slot of a time series. The point
   * is that the prepend happens **before** the difference rather than
   * after; appending changes the last difference instead.
   *
   * @param prepend what to join on the front **before** the difference is
   *   taken.
   * @param append what to join on the end.
   */
  diff(n = 1, dim = 0, prepend?: Tensor, append?: Tensor): Tensor {
    const rank = this.shape.length;
    const axis = dim < 0 ? dim + rank : dim;
    let cur: Tensor = this;
    if (prepend !== undefined || append !== undefined) {
      const parts: Tensor[] = [];
      if (prepend !== undefined) parts.push(prepend);
      parts.push(this);
      if (append !== undefined) parts.push(append);
      cur = Tensor.cat(parts, axis);
    }
    for (let k = 0; k < n; k++) {
      const len = cur.shape[axis] ?? 0;
      if (len < 2) throw new Error(`dimension ${dim} is too short to diff again.`);
      cur = cur.narrow(axis, 1, len - 1).sub(cur.narrow(axis, 0, len - 1));
    }
    return cur;
  }

  /**
   * Overwrites the true positions with a value. **No gradient goes to an
   * overwritten slot.**
   */
  maskedFill(mask: Tensor, value: number): Tensor {
    return Tensor.full(this.shape, value).where(mask, this);
  }

  /**
   * An integer power. For now it repeats the multiplication — for small
   * exponents only.
   */
  matrixPower(k: number): Tensor {
    if (k < 1) throw new Error(`matrix_power supports 1 and up for now: ${k}`);
    // **곱셈을 이어 붙인다** — 그러면 역방향이 저절로 따라온다. 분해로 짜면 미분식을
    // 새로 써야 하고, 그건 틀릴 자리를 하나 더 만드는 것이다.
    //
    // 배치는 3 차원으로 접었다가 편다. `mm` 이 2 차원 전용이고 `bmm` 이 3 차원
    // 전용이라, `(2,3,4,4)` 같은 것은 둘 다 못 받는다 — 접으면 둘 다 필요 없다.
    const rank = this.shape.length;
    if (rank <= 2) {
      let out: Tensor = this;
      for (let i = 1; i < k; i++) out = out.mm(this);
      return out;
    }
    const n = this.shape[rank - 1] ?? 0;
    const batch = this.shape.slice(0, rank - 2).reduce((a, b) => a * b, 1);
    const flat = this.reshape([batch, n, n]);
    let out: Tensor = flat;
    for (let i = 1; i < k; i++) out = out.bmm(flat);
    return out.reshape(this.shape);
  }

  /**
   * This side or that, per position of the condition. torch's method form
   * is `x.where(cond, other)`.
   */
  where(cond: Tensor, other: Tensor): Tensor {
    const n = this.size;
    const out = dev().alloc(n);
    dev().run1d(
      dev().pipeline(`wh:${n}`, () => whereKernel(n)),
      [cond.buffer, this.buffer, other.buffer, out],
      n,
    );
    const shape = this.shape;
    const side = (g: Tensor, take: "a" | "b"): Tensor => {
      const gi = dev().alloc(n);
      dev().run1d(
        dev().pipeline(`whb:${take}:${n}`, () => whereBackward(n, take)),
        [cond.buffer, g.buffer, gi],
        n,
      );
      return new Tensor(gi, shape);
    };
    return Tensor.make(
      out,
      shape,
      [this, other],
      (g) => [
        this.requiresGrad ? side(g, "a") : null,
        other.requiresGrad ? side(g, "b") : null,
      ],
      "WhereBackward0",
    );
  }

  /**
   * Multiplies everything.
   */
  prod(dim?: number, keepdim = false, dtype?: DType): Tensor {
    if (dtype !== undefined) return this.castFirst(dtype).prod(dim, keepdim).to(dtype);
    if (dim === undefined) return this.flat().reduceOver("prod", 0, false);
    return this.reduceOver("prod", dim, keepdim);
  }

  /**
   * The L2 norm.
   */
  norm(): Tensor {
    // **torch 가 멈추는 자리에서 멈춘다**(실측). 나눗셈·제곱근이 정수 칸에
    // 답이 안 들어간다 — numpy 처럼 조용히 실수로 올리면 그 코드가 진짜
    // torch 에서 깨진다.
    this.needsFloat("norm is for floating point only", "linalg.vector_norm: Expected a floating point or complex tensor as input");
    return this.square().sum().sqrt();
  }

  /**
   * The inner product of two vectors.
   */
  dot(other: Tensor): Tensor {
    return this.mul(other).sum();
  }

  /**
   * The outer product of two vectors. It falls out of broadcasting — no new
   * kernel needed.
   */
  outer(other: Tensor): Tensor {
    return this.reshape([this.size, 1]).mul(other.reshape([1, other.size]));
  }

  /**
   * Cuts above and below.
   *
   * **It must not be laid on top of `maximum` and `minimum`.** Those two
   * split the gradient in half at a tie (torch does), while `clamp` passes
   * it whole at the boundary. Laid on top, the gradient halved exactly
   * where `x` sat on the boundary.
   */
  clamp(low: number, high: number): Tensor {
    const lo = f32lit(low), hi = f32lit(high);
    return this.unary(unaryWith(`clamp<${lo},${hi}>`, () => ({
      fwd: `clamp(x, ${lo}, ${hi})`,
      bwd: `select(0.0, 1.0, x >= ${lo} && x <= ${hi})`,
    })));
  }

  /**
   * A constant exponent.
   *
   * **Integer exponents go through multiplication.** WGSL's `pow(x, y)` is
   * `exp2(y·log2(x))`, so a negative base has no answer, and what actually
   * comes out looks like `|x|` was used. At even exponents the value
   * happens to be right, so `method::pow` passed and it was caught at
   * `grad::pow2` with the sign flipped — the kind where the value is right
   * and only the gradient is wrong.
   */
  powScalar(k: number): Tensor {
    if (Number.isInteger(k) && k >= 0 && k <= MAX_UNROLLED_POWER) {
      if (k === 0) return Tensor.ones(this.shape);
      let acc: Tensor = this;
      for (let i = 1; i < k; i++) acc = acc.mul(this);
      return acc;
    }
    // 정수가 아니면 음수 밑에서 답이 없는 것이 맞다. 그대로 커널에 맡긴다.
    return this.binary("pow", Tensor.full([], k));
  }

  /**
   * `exp(x) / Σ exp(x)`. **Computed with the maximum subtracted** —
   * otherwise it overflows at large values.
   */
  softmax(dim = 0): Tensor {
    const m = this.amax(dim, true).detach();
    const e = this.sub(m).exp();
    return e.div(e.sumDim(dim, true));
  }

  /**
   * `log(softmax(x))`. **It does not compute `softmax` and take the log** —
   * small probabilities become 0 and the log becomes -inf. Written directly
   * as a subtraction, that place does not exist.
   */
  logSoftmax(dim = 0): Tensor {
    return this.sub(this.logsumexp(dim, true));
  }

  /**
   * 접힌 축을 크기 1 로 되살린다. `keepdim` 을 받는 것들이 마지막에 부른다.
   *
   * **축이 사라진 모양은 브로드캐스팅에 자주 들어맞는다.** 그래서 `keepdim` 을 안
   * 받으면 시끄럽게 멈추는 대신 값만 틀린 채 끝까지 가는 일이 생긴다 —
   * `x.gather(1, x.argmax(1, true))` 가 그 꼴이다.
   */
  private liftAxis(out: Tensor, dim: number, keepdim: boolean): Tensor {
    if (!keepdim) return out;
    const axis = dim < 0 ? dim + this.shape.length : dim;
    const shape = [...out.shape];
    shape.splice(axis, 0, 1);
    return out.reshape(shape);
  }

  /**
   * **Where** the maximum or minimum is. On a tie, the first position.
   */
  argmax(dim = 0, keepdim = false): Tensor {
    return this.liftAxis(this.argReduceOver("max", dim), dim, keepdim);
  }

  argmin(dim = 0, keepdim = false): Tensor {
    return this.liftAxis(this.argReduceOver("min", dim), dim, keepdim);
  }

  /**
   * The value and the index together. torch's `x.max(dim)`.
   *
   * Calling `amax` and `argmax` separately does work, but this is the shape
   * torch code uses, and **whether the two point at the same place** is
   * confirmed only here. `amax` splits a tie evenly and `argmax` picks the
   * first single position, and re-fetching the value by index does no
   * splitting — torch's `max(dim)` is the latter.
   */
  max(dim = 0, keepdim = false): { values: Tensor; indices: Tensor } {
    return this.pickReduce("max", dim, keepdim);
  }

  min(dim = 0, keepdim = false): { values: Tensor; indices: Tensor } {
    return this.pickReduce("min", dim, keepdim);
  }

  private pickReduce(kind: "max" | "min", dim: number, keepdim = false):
    { values: Tensor; indices: Tensor } {
    const rank = this.shape.length;
    const axis = dim < 0 ? dim + rank : dim;
    const indices = this.argReduceOver(kind, axis);
    // 번호로 다시 뽑는다 — 그래야 기울기가 이긴 자리 **하나**로만 간다.
    // `gather` 는 랭크가 같아야 하므로 접혔던 축을 되살렸다가 다시 접는다.
    const lifted = [...this.shape];
    lifted[axis] = 1;
    const values = this.gather(axis, indices.reshape(lifted))
      .reshape(indices.shape);
    // **번호도 축을 지켜야 한다.** 값만 살리면 `x.gather(1, m.indices)` 가 랭크
    // 어긋남으로 멈추거나 — 더 나쁘게 — 브로드캐스팅으로 통과한다.
    return {
      values: this.liftAxis(values, axis, keepdim),
      indices: this.liftAxis(indices, axis, keepdim),
    };
  }

  private argReduceOver(kind: "max" | "min", dim: number): Tensor {
    const rank = this.shape.length;
    const axis = dim < 0 ? dim + rank : dim;
    const red = this.shape[axis] ?? 1;
    const outer = this.shape.slice(0, axis).reduce((a, b) => a * b, 1);
    const inner = this.shape.slice(axis + 1).reduce((a, b) => a * b, 1);
    const outShape = [...this.shape];
    outShape.splice(axis, 1);
    const n = outer * inner;
    const out = dev().alloc(n);
    dev().run1d(
      dev().pipeline(
        `ar:${kind}:${outer}:${red}:${inner}`,
        () => argReduce(kind, outer, red, inner),
      ),
      [this.buffer, out],
      n,
    );
    // 자리는 값이 아니다 — 기울기가 흐를 자리가 없다. torch 도 안 흘린다.
    // **형은 언제나 int64 다.** 고르기가 아니라 번호를 세는 것이라 원래 형과 무관하다.
    return new Tensor(out, outShape, { dtype: "int64" });
  }

  /**
   * The count of non-zeros. **It takes an axis** — this used to be a full
   * reduction only.
   *
   * Without the axis, `x.countNonzero(1)` quietly discards the argument and
   * produces a scalar, and that scalar broadcasts against anything.
   */
  countNonzero(dim?: number): Tensor {
    const flags = this.binary("ne", Tensor.full([], 0));
    return dim === undefined ? flags.sum() : flags.sumDim(dim);
  }

  /**
   * Whether all are true, or whether any is. It answers 0/1.
   */
  all(dim?: number, keepdim = false): Tensor {
    return this.boolReduce("amin", dim, keepdim);
  }

  any(dim?: number, keepdim = false): Tensor {
    return this.boolReduce("amax", dim, keepdim);
  }

  /** `all`·`any` 의 몸통. 0 이 아닌가로 바꾼 뒤 고르기로 접는다. */
  private boolReduce(pick: "amin" | "amax", dim: number | undefined,
                     keepdim: boolean): Tensor {
    const flags = this.binary("ne", Tensor.full([], 0));
    if (dim === undefined) return pick === "amin" ? flags.amin() : flags.amax();
    const out = pick === "amin" ? flags.amin(dim, keepdim)
                                : flags.amax(dim, keepdim);
    return out;
  }

  /**
   * The raw name for `padND(mode="constant")`. **From the last axis, in
   * (before, after) order.**
   *
   * It is a place where one computation has two names, so it **does not
   * compute here** and hands over — only one of the two being right has
   * happened three times in this repository.
   */
  constantPadNd(pad: readonly number[], value = 0): Tensor {
    return this.padND(pad, "constant", value);
  }

  /**
   * Imitates quantisation **on floats** — `clamp(round(x/s) + z)` and back.
   *
   * **No quantised dtype is needed.** The name had it counted as a refusal
   * for a long time, and measuring showed torch takes a float tensor and
   * produces a float too.
   *
   * **The gradient is 1 only inside the range** (measured). Rounding is a
   * staircase whose derivative is zero almost everywhere, and left as-is no
   * training goes below this layer at all — torch leaves that place as a
   * straight-through.
   */
  fakeQuantizePerTensorAffine(scale: number, zeroPoint: number,
                              quantMin: number, quantMax: number): Tensor {
    const s = Tensor.full([], scale);
    const z = Tensor.full([], zeroPoint);
    const raw = this.div(s).round().add(z);
    const clipped = raw.clamp(quantMin, quantMax);
    const value = clipped.sub(z).mul(s).detach();
    // 범위 안이면 기울기를 그대로, 밖이면 0. `where` 로 고른다.
    const inside = raw.detach().binary("ge", Tensor.full([], quantMin), "bool")
      .binary("logical_and",
              raw.detach().binary("le", Tensor.full([], quantMax), "bool"),
              "bool");
    return value.add(this.sub(this.detach()).where(inside,
                                                   Tensor.zeros(this.shape)));
  }

  /**
   * A different scale per slot. The scale varies along one axis.
   */
  fakeQuantizePerChannelAffine(scale: Tensor, zeroPoint: Tensor, axis: number,
                               quantMin: number, quantMax: number): Tensor {
    const line = new Array<number>(this.shape.length).fill(1);
    line[axis < 0 ? axis + this.shape.length : axis] = scale.size;
    const s = scale.reshape(line);
    const z = zeroPoint.reshape(line);
    const raw = this.div(s).round().add(z);
    const clipped = raw.clamp(quantMin, quantMax);
    const value = clipped.sub(z).mul(s).detach();
    const inside = raw.detach().binary("ge", Tensor.full([], quantMin), "bool")
      .binary("logical_and",
              raw.detach().binary("le", Tensor.full([], quantMax), "bool"),
              "bool");
    return value.add(this.sub(this.detach()).where(inside,
                                                   Tensor.zeros(this.shape)));
  }

  /**
   * A quantised tensor back to float. **For us it is always the identity.**
   *
   * The reason this differs from "an identity that happens to pass today"
   * is that having no quantised dtype is **already settled as permanent** —
   * the only input it can take is float, and on floats torch is the
   * identity too (measured). **It is not differentiable** — torch stops at
   * `backward`.
   */
  dequantize(): Tensor {
    return new Tensor(this.raw, this.shape, { dtype: this.dtype });
  }

  /**
   * Pads a constant before and after one axis. For several axes, call it
   * per axis.
   */
  pad(dim: number, before: number, after: number, value = 0): Tensor {
    const rank = this.shape.length;
    const axis = dim < 0 ? dim + rank : dim;
    const size = this.shape[axis] ?? 0;
    const outer = this.shape.slice(0, axis).reduce((a, b) => a * b, 1);
    const inner = this.shape.slice(axis + 1).reduce((a, b) => a * b, 1);
    const outShape = this.shape.map((s, d) => (d === axis ? s + before + after : s));
    const n = outShape.reduce((a, b) => a * b, 1);
    const out = dev().alloc(n);
    dev().run1d(
      dev().pipeline(
        `pad:${outer}:${before}:${size}:${after}:${inner}:${value}`,
        () => padAxis(outer, before, size, after, inner, value),
      ),
      [this.buffer, out],
      n,
    );
    return Tensor.make(
      out,
      outShape,
      [this],
      // 덧댄 자리는 입력에서 온 것이 아니다 — 가운데만 돌려준다.
      (g) => [g.narrow(axis, before, size)],
      "ConstantPadNdBackward0",
      this.dtype,
    );
  }

  // ── 창 펴기 ───────────────────────────────────────────────────────────
  //
  // **`unfold` 와 `fold` 는 서로의 역이 아니다.** 되접을 때 겹친 자리를 **더한다** —
  // 4×4 를 2×2 창으로 펴서 그대로 되접으면 가운데가 네 번 세어진다.
  //
  // 색인 하나로 둘을 만든다. 어디서 왔는지를 정리해 두면 펴는 것은 모으기이고
  // 되접는 것은 그 자리로 더해 넣기라, 한쪽의 역방향이 곧 다른 쪽이다.

  /**
   * Spreads windows into columns. `(N, C, H, W)` → `(N, C·kh·kw, L)`.
   */
  unfoldIm2col(kernel: number | [number, number], dilation = 1, padding = 0,
               stride = 1): Tensor {
    const [n, c, h, w] = this.shape as [number, number, number, number];
    const [kh, kw] = pairOf(kernel);
    const [ph, pw] = pairOf(padding);
    const padded = padding
      ? this.padND([pw, pw, ph, ph], "constant", 0)
      : this;
    const { idx, rows, cols } = windowIndex(
      [c, h, w], [kh, kw], pairOf(dilation), [ph, pw], pairOf(stride));
    return padded.reshape([n, padded.size / n])
      .indexSelect(1, Tensor.from(idx, [idx.length]))
      .reshape([n, rows, cols]);
  }

  /**
   * Folds the spread back. **Overlapping positions are added** — that is
   * what this function means.
   */
  fold(outputSize: number | [number, number], kernel: number | [number, number],
       dilation = 1, padding = 0, stride = 1): Tensor {
    const n = this.shape[0] ?? 1;
    const [kh, kw] = pairOf(kernel);
    const [oh, ow] = pairOf(outputSize);
    const [ph, pw] = pairOf(padding);
    const c = (this.shape[1] ?? 1) / (kh * kw);
    const { idx } = windowIndex(
      [c, oh, ow], [kh, kw], pairOf(dilation), [ph, pw], pairOf(stride));
    const hp = oh + 2 * ph;
    const wp = ow + 2 * pw;
    // 배치마다 같은 자리표를 쓴다 — `scatterAdd` 는 색인이 원본과 같은 모양이길 바란다.
    const wide = new Float32Array(n * idx.length);
    for (let b = 0; b < n; b++) wide.set(idx, b * idx.length);
    const flat = Tensor.zeros([n, c * hp * wp]).scatterAdd(
      1, Tensor.from(wide, [n, idx.length]), this.reshape([n, idx.length]));
    const made = flat.reshape([n, c, hp, wp]);
    if (!ph && !pw) return made;
    return made.narrow(2, ph, oh).narrow(3, pw, ow);
  }

  // ── 나머지 층이 쓰는 것들 ─────────────────────────────────────────────

  /**
   * `y[o] = x₁ᵀ·W[o]·x₂ + b[o]`. The weight has **three axes.**
   *
   * It does not call `einsum` — that side imports this file, so they would
   * bite each other. Split into two steps it writes down with what already
   * exists: first ride `x₂` through `W` to make `(B, O, I)`, then fold with
   * `x₁` along the last axis.
   */
  bilinear(other: Tensor, weight: Tensor, bias: Tensor | null = null): Tensor {
    const [o, i, j] = weight.shape as [number, number, number];
    const b = this.shape[0] ?? 1;
    const mixed = other.linear(weight.reshape([o * i, j])).reshape([b, o, i]);
    const out = this.reshape([b, 1, i]).mul(mixed).sumDim(2, false);
    return bias ? out.add(bias) : out;
  }

  /**
   * Divides by neighbouring channels.
   *
   * **The window is off-centre.** Channel `c`'s window is `[c − n//2, c + n
   * − 1 − n//2]`, so at `size=2` it is `{c−1, c}` — centring it shifts the
   * values by one slot, and the size being the same means the shape does
   * not show it.
   */
  localResponseNorm(size: number, alpha = 1e-4, beta = 0.75, k = 1.0): Tensor {
    const c = this.shape[1] ?? 1;
    const left = Math.floor(size / 2);
    const right = size - 1 - left;
    // 채널 축에 0 을 덧대면 가장자리가 저절로 잘린다. `padND` 는 마지막 축부터
    // 세므로 4 차원에서 채널은 셋째 짝이다.
    const tail = new Array<number>(2 * (this.shape.length - 2)).fill(0);
    const padded = this.square().padND([...tail, left, right], "constant", 0);
    let total: Tensor | null = null;
    for (let i = 0; i < size; i++) {
      const piece = padded.narrow(1, i, c);
      total = total ? total.add(piece) : piece;
    }
    const scaled = total!.binary("mul", Tensor.full([], alpha / size))
      .binary("add", Tensor.full([], k));
    return this.div(scaled.powScalar(beta));
  }

  /**
   * Random numbers between 0 and 1. `rrelu` and `gumbelSoftmax` use it.
   *
   * It uses the seed `manualSeed` catches — the same seed has to give the
   * same draw for anything to reproduce.
   */
  static uniform(shape: readonly number[], lower = 0, upper = 1): Tensor {
    const n = numel(shape);
    const out = dev().alloc(n);
    dev().run1d(
      dev().pipeline(`uni:${n}:${lower}:${upper}:${Tensor.dropoutSeed}`,
        () => uniformFill(n, lower, upper, Tensor.dropoutSeed)),
      [out],
      n,
    );
    Tensor.dropoutSeed = (Tensor.dropoutSeed + 1) >>> 0;
    return new Tensor(out, shape);
  }

  /**
   * A table with 1 only at the largest position along one axis.
   *
   * **A tie must not be settled with `eq`** — two equal values give two
   * ones, and then `gumbelSoftmax(hard=true)`'s answer is not one-hot. It
   * almost never happens with random numbers, but "almost" is not a
   * guarantee. It picks the index and places at that slot alone.
   */
  oneHotAlong(indices: Tensor, dim: number): Tensor {
    const shape = [...this.shape];
    shape[dim] = 1;
    // 번호를 자리로 펴는 일은 이미 `scatter` 가 한다 — 0 판 위에 1 을 놓는다.
    return Tensor.zeros(this.shape)
      .scatterSet(dim, indices.reshape(shape), Tensor.ones(shape));
  }

  /**
   * Draws the negative-side slope and uses it.
   *
   * **In eval mode it settles at the middle** — with the defaults, `(1/8 +
   * 1/3)/2 = 0.2292`. Randomness enters in training mode only.
   */
  rrelu(lower = 1 / 8, upper = 1 / 3, training = false): Tensor {
    if (!training) return this.leakyRelu((lower + upper) / 2);
    const n = this.size;
    const out = dev().alloc(n);
    dev().run1d(
      dev().pipeline(`uni:${n}:${lower}:${upper}:${Tensor.dropoutSeed}`,
        () => uniformFill(n, lower, upper, Tensor.dropoutSeed)),
      [out],
      n,
    );
    Tensor.dropoutSeed = (Tensor.dropoutSeed + 1) >>> 0;
    const slope = new Tensor(out, this.shape);
    const positive = this.binary("gt", Tensor.full([], 0));
    return this.where(positive, this.mul(slope));
  }

  /**
   * Bilinear enlargement.
   *
   * **`alignCorners` changes the values.** True pins both ends and divides
   * evenly between them; false measures from the centre of a cell.
   * `UpsamplingBilinear2d` is true and `Upsample(mode='bilinear')` defaults
   * to false, so treating them as aliases on the strength of the names
   * misaligns the edges — the interior is close enough that an eye does not
   * separate them.
   */
  interpolateBilinear(outH: number, outW: number, alignCorners: boolean): Tensor {
    const h = this.shape[2] ?? 1;
    const w = this.shape[3] ?? 1;
    const ys = bilinearAxis(h, outH, alignCorners);
    const xs = bilinearAxis(w, outW, alignCorners);
    const pick = (t: Tensor, axis: number, at: number[]) =>
      t.indexSelect(axis, Tensor.from(at, [at.length]));
    const wy = Tensor.from(ys.frac, [outH, 1]);
    const wx = Tensor.from(xs.frac, [1, outW]);
    const one = Tensor.full([], 1);
    const corner = (yi: number[], xi: number[]) =>
      pick(pick(this, 2, yi), 3, xi);
    const top = corner(ys.lo, xs.lo).mul(one.sub(wx))
      .add(corner(ys.lo, xs.hi).mul(wx));
    const bottom = corner(ys.hi, xs.lo).mul(one.sub(wx))
      .add(corner(ys.hi, xs.hi).mul(wx));
    return top.mul(one.sub(wy)).add(bottom.mul(wy));
  }

  // ── 자리 옮기기 ───────────────────────────────────────────────────────
  //
  // 셋 다 **값을 안 바꾸고 자리만 바꾼다.** 순방향이 `reshape` + 축 바꾸기이고
  // 역방향은 그 반대라, 이미 있는 것의 조합이다.

  /**
   * `(N, C·r², H, W)` → `(N, C, H·r, W·r)`. Cuts channels and plants them
   * in space.
   *
   * **The interleaving order is the whole of the value.** The channels
   * split into `(C, r, r)` and the two `r`s are inserted after `H` and
   * after `W` respectively. Change the order and the shape is the same
   * while the picture is scrambled.
   */
  pixelShuffle(upscaleFactor: number): Tensor {
    const r = upscaleFactor;
    const [n, c, h, w] = this.shape as [number, number, number, number];
    return this.reshape([n, c / (r * r), r, r, h, w])
      .permute([0, 1, 4, 2, 5, 3])
      .reshape([n, c / (r * r), h * r, w * r]);
  }

  /**
   * The inverse of `pixelShuffle`. Cuts space and stacks it into channels.
   */
  pixelUnshuffle(downscaleFactor: number): Tensor {
    const r = downscaleFactor;
    const [n, c, h, w] = this.shape as [number, number, number, number];
    return this.reshape([n, c, h / r, r, w / r, r])
      .permute([0, 1, 3, 5, 2, 4])
      .reshape([n, c * r * r, h / r, w / r]);
  }

  /**
   * Splits channels into groups and **interleaves them back.**
   *
   * Shuffling `[0,1,2,3]` in two groups gives `[0,2,1,3]` — it is the place
   * that releases information trapped inside groups after a grouped
   * convolution, so the direction of the interleave is the whole of the
   * value.
   */
  channelShuffle(groups: number): Tensor {
    const [n, c] = this.shape as [number, number];
    const rest = this.shape.slice(2);
    // `transpose` 는 2 차원 전용이라 `permute` 로 간다 — 뒤에 공간 축이 몇이든
    // 앞의 두 자리만 바꾸면 된다.
    const tail = rest.map((_, i) => i + 3);
    return this.reshape([n, groups, c / groups, ...rest])
      .permute([0, 2, 1, ...tail])
      .reshape([n, c, ...rest]);
  }

  /**
   * **Drops channels, not elements.**
   *
   * Sitting next to `dropout`, the name reads as "the N-dimensional one",
   * and the work is different — a whole channel goes to zero or stays
   * whole. Drawing the mask as `(N, C, 1, …)` and broadcasting writes that
   * meaning down directly.
   */
  featureDropout(p = 0.5, training = true): Tensor {
    if (!training || p === 0) return this;
    if (p >= 1) return this.mul(Tensor.full([], 0));
    const lead = this.shape.slice(0, 2);
    const ones = new Array<number>(this.shape.length - 2).fill(1);
    const mask = Tensor.ones([...lead, ...ones]).dropout(p, true);
    return this.mul(mask);
  }

  /**
   * The dropout used with SELU. **It does not put zeros where it drops.**
   *
   * It puts a negative constant and applies an affine transform over the
   * whole so that mean and variance are preserved — zeros break SELU's
   * self-normalisation, and the values are plausible enough that nothing
   * shows while training runs.
   */
  alphaDropout(p = 0.5, training = false, perChannel = false): Tensor {
    if (!training || p === 0) return this;
    const lead = perChannel
      ? [...this.shape.slice(0, 2), ...new Array<number>(this.shape.length - 2).fill(1)]
      : this.shape;
    // `dropout` 은 살아남은 자리를 `1/(1-p)` 로 키운다. 여기서는 0/1 이 필요하므로
    // 되돌려 곱한다 — 마스크를 따로 뽑는 커널을 하나 더 두지 않으려는 것이다.
    const keep = Tensor.ones(lead).dropout(p, true)
      .binary("mul", Tensor.full([], 1 - p));
    const a = ((1 - p) * (1 + p * ALPHA_PRIME ** 2)) ** -0.5;
    const b = -a * p * ALPHA_PRIME;
    const one = Tensor.full([], 1);
    const kept = this.mul(keep);
    const dropped = one.sub(keep).binary("mul", Tensor.full([], ALPHA_PRIME));
    return kept.add(dropped).binary("mul", Tensor.full([], a))
      .binary("add", Tensor.full([], b));
  }

  /**
   * Where `F.pad` goes — taken **from the last axis, in (before, after)
   * order.**
   *
   * **The pair count and the rank interlock.** One pair needs rank 2 or 3,
   * two pairs rank 3 or 4, three pairs rank 4 or 5 — torch refuses anything
   * else. Accepting any rank lets a wrong axis pass.
   *
   * Constant goes to the existing kernel because that is shorter; the other
   * three go through indexing.
   */
  padND(padding: readonly number[], mode: PadMode = "constant", value = 0): Tensor {
    const rank = this.shape.length;
    const pairs = Math.floor(padding.length / 2);
    if (mode !== "constant" && rank !== pairs + 1 && rank !== pairs + 2) {
      // **거절의 종류가 답의 일부다.** torch 는 여기를 `NotImplementedError` 로 내고,
      // 그것은 "부른 쪽이 틀렸다" 와 다른 말이다.
      throw new NotImplementedError(
        `Padding size ${padding.length} is not supported for ${rank}D input tensor`,
      );
    }
    let out: Tensor = this;
    for (let i = 0; i < pairs; i++) {
      const axis = rank - 1 - i;
      const before = padding[2 * i] ?? 0;
      const after = padding[2 * i + 1] ?? 0;
      if (before === 0 && after === 0) continue;
      if (mode === "constant") {
        out = out.pad(axis, before, after, value);
        continue;
      }
      const size = out.shape[axis] ?? 0;
      // **`reflect` 만 크기를 따진다.** 거울로 접으려면 접을 것이 있어야 한다.
      // `replicate` 는 다섯 칸을 늘려도 되는데, 늘일 값이 늘 있기 때문이다.
      if (mode === "reflect" && (before >= size || after >= size)) {
        throw new RuntimeError(
          "Argument #4: Padding size should be less than the corresponding input " +
          `dimension, but got: padding (${before}, ${after}) at dimension ${axis} ` +
          `of input ${rank}`,
        );
      }
      const idx = padIndex(mode, size, before, after);
      out = out.indexSelect(axis, Tensor.from(idx, [idx.length]));
    }
    return out;
  }

  /**
   * A gentle slope on the left.
   *
   * Written as `max(x, slope·x)` it was **wrong exactly where x is 0.**
   * There the two terms tie and `maximum` splits a tie in half, giving (1 +
   * slope)/2, while torch gives the slope — it splits on `x > 0` alone and
   * has no notion of a tie.
   */
  leakyRelu(slope = 0.01): Tensor {
    const s = f32lit(slope);
    return this.unary(unaryWith(`leakyRelu<${s}>`, () => ({
      fwd: `select(x * ${s}, x, x > 0.0)`,
      bwd: `select(${s}, 1.0, x > 0.0)`,
    })));
  }

  /**
   * The negative side lies down exponentially. **It takes α** — the table's
   * `elu` is baked at α=1.
   *
   * `nn.ELU(0.5)` is a form that exists in torch, so keeping only the
   * argument-free version stops that line. At α=1 the value equals the
   * table's, so **without varying α the two cannot be told apart.**
   */
  elu(alpha = 1.0): Tensor {
    const a = f32lit(alpha);
    return this.unary(unaryWith(`elu<${a}>`, () => ({
      fwd: `select(${a} * (exp(x) - 1.0), x, x > 0.0)`,
      bwd: `select(${a} * exp(x), 1.0, x > 0.0)`,
    })));
  }

  /**
   * The `approximate="tanh"` GELU. The table's `gelu` is **the exact form**
   * (erf).
   *
   * The maximum difference is around 1e-4, so it is a place where "near
   * enough, keep one" almost got through. torch keeps them apart because
   * the tanh one is faster, not because they are the same.
   */
  geluTanh(): Tensor {
    return this.unary(unaryWith("geluTanh", () => ({
      fwd: "0.5 * x * (1.0 + tanh(0.7978845608028654 * "
        + "(x + 0.044715 * x * x * x)))",
      bwd: "gelu_tanh_grad(x)",
      prelude: `
fn gelu_tanh_grad(x: f32) -> f32 {
  let inner = 0.7978845608028654 * (x + 0.044715 * x * x * x);
  let t = tanh(inner);
  let d = 0.7978845608028654 * (1.0 + 3.0 * 0.044715 * x * x);
  return 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * d;
}`,
    })));
  }

  /**
   * Unlike ELU it **divides** the negative side by α before taking the
   * exponential.
   *
   * At α=1 the value equals ELU's, so measuring without giving α cannot
   * tell them apart.
   */
  celu(alpha = 1.0): Tensor {
    const a = f32lit(alpha);
    return this.unary(unaryWith(`celu<${a}>`, () => ({
      fwd: `select(${a} * (exp(x / ${a}) - 1.0), x, x > 0.0)`,
      bwd: `select(exp(x / ${a}), 1.0, x > 0.0)`,
    })));
  }

  /**
   * |x| > λ passes through, otherwise 0. **The boundary goes to the zero side** (`>`, not `>=`).
   */
  hardshrink(lambd = 0.5): Tensor {
    const l = f32lit(lambd);
    return this.unary(unaryWith(`hardshrink<${l}>`, () => ({
      fwd: `select(0.0, x, abs(x) > ${l})`,
      bwd: `select(0.0, 1.0, abs(x) > ${l})`,
    })));
  }

  /**
   * Pulls **towards the origin** by λ. Unlike `hardshrink` the values stay
   * continuous.
   */
  softshrink(lambd = 0.5): Tensor {
    const l = f32lit(lambd);
    return this.unary(unaryWith(`softshrink<${l}>`, () => ({
      fwd: `select(select(0.0, x + ${l}, x < -${l}), x - ${l}, x > ${l})`,
      bwd: `select(0.0, 1.0, abs(x) > ${l})`,
    })));
  }

  hardtanh(minVal = -1.0, maxVal = 1.0): Tensor {
    const lo = f32lit(minVal);
    const hi = f32lit(maxVal);
    return this.unary(unaryWith(`hardtanh<${lo},${hi}>`, () => ({
      fwd: `clamp(x, ${lo}, ${hi})`,
      bwd: `select(0.0, 1.0, x > ${lo} && x < ${hi})`,
    })));
  }

  /**
   * (1/β)·log(1+e^{βx}). **Past the threshold, βx is simply x.**
   *
   * Leave that branch out and `exp` overflows on large inputs, and every
   * gradient after it is NaN.
   */
  softplus(beta = 1.0, threshold = 20.0): Tensor {
    const b = f32lit(beta);
    const t = f32lit(threshold);
    return this.unary(unaryWith(`softplus<${b},${t}>`, () => ({
      fwd: `select(log(1.0 + exp(-abs(${b} * x))) / ${b} + max(x, 0.0), x, ${b} * x > ${t})`,
      bwd: `select(1.0 / (1.0 + exp(-${b} * x)), 1.0, ${b} * x > ${t})`,
    })));
  }

  /**
   * x > t passes through, otherwise `value`. **The boundary goes to the
   * value side.**
   */
  threshold(t: number, value: number): Tensor {
    const th = f32lit(t);
    const v = f32lit(value);
    return this.unary(unaryWith(`threshold<${th},${v}>`, () => ({
      fwd: `select(${v}, x, x > ${th})`,
      bwd: `select(0.0, 1.0, x > ${th})`,
    })));
  }

  /**
   * The negative-side slope is **learned.**
   *
   * **Exactly zero belongs to the negative side.** The forward is 0 either
   * way so nothing shows, while the gradient diverges — torch gives 1 only
   * where `x > 0`. Splitting on `x < 0` in the core produced a maximum
   * difference of 3.75 at that single point.
   *
   * The weight is learned, so it is wired as a binary operation — baked as
   * a constant, no gradient flows.
   */
  prelu(weight: Tensor): Tensor {
    // `gt` 의 기울기는 0 이다 — 가림막이 기울기를 나르면 안 된다.
    // **float 으로 옮긴다.** 비교는 bool 을 내고 bool 로는 빼기가 거절된다(torch 도
    // 그렇다). 가림막을 수로 쓸 것이므로 여기서 형을 맞춘다.
    const pos = this.binary("gt", Tensor.full([], 0)).to("float32");
    const neg = Tensor.full([], 1).sub(pos);
    return this.mul(pos).add(this.mul(weight).mul(neg));
  }

  /**
   * Splits the axis in half and gives `a · σ(b)`. The only activation that
   * is not elementwise.
   */
  glu(dim = -1): Tensor {
    const rank = this.shape.length;
    const axis = dim < 0 ? dim + rank : dim;
    const n = this.shape[axis] ?? 0;
    if (n % 2 !== 0) {
      throw new Error(`glu needs an even length along dimension ${dim} (got ${n})`);
    }
    const half = n / 2;
    return this.narrow(axis, 0, half).mul(this.narrow(axis, half, half).sigmoid());
  }

  /**
   * softmax(−x). **Miss the sign and it becomes softmax** — only the values
   * separate them.
   */
  softmin(dim = -1): Tensor {
    return this.neg().softmax(dim);
  }

  /**
   * Picks slots to drop and **scales the survivors by `1/(1-p)`.**
   *
   * The scaling is the point — it is what makes the expectation during
   * training match inference, and leaving it out makes the two modes differ
   * in magnitude. `p=1` is branched separately: `1/(1-p)` becomes a
   * division by zero and produces NaN, and NaN differs even from itself, so
   * no comparison after it passes.
   *
   * **It drops different slots on every call.** The seed is advanced as it
   * goes — drawn once and cached, the same slots die every step, and that
   * is not dropout.
   */
  dropout(p = 0.5, training = true): Tensor {
    if (!training || p === 0) return this;
    if (p >= 1) return this.mul(Tensor.full([], 0));
    const n = this.size;
    const d = dev();
    const mask = d.alloc(n);
    d.run1d(
      // 씨앗이 이름에 들어가면 스텝마다 셰이더를 새로 굽게 된다. 상수로 굽는 것은
      // `p` 뿐이고 씨앗은 이름 밖에 둔다 — 대신 파이프라인 하나를 돌려 쓴다.
      d.pipeline(`drop:${n}:${p}:${Tensor.dropoutSeed}`,
        () => dropoutMask(n, p, Tensor.dropoutSeed)),
      [mask],
      n,
    );
    Tensor.dropoutSeed = (Tensor.dropoutSeed + 1) >>> 0;
    return this.mul(new Tensor(mask, this.shape));
  }

  /**
   * The seed dropout uses. It rises on every call, so as not to drop the
   * same slots twice.
   */
  static dropoutSeed = 1;

  /**
   * Length to 1 along an axis. `eps` stops the division blowing up on a
   * zero vector.
   */
  normalize(dim = 1, eps = 1e-12): Tensor {
    const len = this.square().sumDim(dim, true).sqrt();
    return this.div(len.binary("maximum", Tensor.full([], eps)));
  }

  /**
   * How closely two bundles point the same way.
   */
  cosineSimilarity(other: Tensor, dim = 1, eps = 1e-8): Tensor {
    const dotted = this.mul(other).sumDim(dim, false);
    const la = this.square().sumDim(dim, false).sqrt();
    const lb = other.square().sumDim(dim, false).sqrt();
    return dotted.div(la.mul(lb).binary("maximum", Tensor.full([], eps)));
  }

  /**
   * Absolute error.
   *
   * **`reduction` was missing here.** Thirteen rarely-used losses all took
   * it and the four most used did not — the ones written later followed
   * torch's signature and the ones written first were never fixed. The
   * tutorials use the default, so the table never asked either.
   */
  l1Loss(target: Tensor, reduction: Reduction = "mean"): Tensor {
    return this.sub(target).abs().reduceAs(reduction);
  }

  /**
   * 작을 때는 제곱, 클 때는 절대값. **원점에서 미분이 이어진다** — 그것이 이 손실을
   * 쓰는 이유이므로 `beta` 를 경계로 두 식을 붙인다.
   */
  /**
   * Squared error.
   */
  mseLoss(target: Tensor, reduction: Reduction = "mean"): Tensor {
    return this.sub(target).square().reduceAs(reduction);
  }

  /**
   * Binary cross-entropy taking logits directly.
   *
   * **It does not compute `sigmoid` first and take the log.** That gives
   * `log(0)` where the model is confident and the loss becomes infinite.
   * `max(x,0) − x·y + log(1+exp(−|x|))` gives the same value without
   * overflow — that is why this function exists separately.
   */
  bceWithLogits(target: Tensor, reduction: Reduction = "mean"): Tensor {
    const zero = Tensor.full([], 0);
    const hinge = this.binary("maximum", zero);
    const stable = this.abs().neg().exp().unary("log1p");
    return hinge.sub(this.mul(target)).add(stable).reduceAs(reduction);
  }

  /**
   * Trailing axes to mean 0, variance 1. **The variance is the biased
   * estimate (divided by n)** — that is torch's `layer_norm`, and it
   * differs from `var()`'s default.
   */
  layerNorm(dim = -1, eps = 1e-5): Tensor {
    const m = this.mean(dim, true);
    const centered = this.sub(m);
    const v = centered.square().mean(dim, true);
    return centered.div(v.binary("add", Tensor.full([], eps)).sqrt());
  }

  /**
   * Folds the **last `dims` axes together.** `layerNorm` folds one axis
   * only.
   *
   * torch's `LayerNorm(normalized_shape)` decides **how many** axes to fold
   * from the length of that shape, and measured with a single axis
   * (`LayerNorm(4)`) the two give the same answer — which is why it went
   * unseen for a long time. Given `(3, 4)` they diverge.
   */
  layerNormOver(dims = 1, eps = 1e-5): Tensor {
    if (dims <= 1) return this.layerNorm(-1, eps);
    const rank = this.shape.length;
    const lead = this.shape.slice(0, rank - dims).reduce((a, b) => a * b, 1);
    return this.reshape([lead, this.size / lead])
      .layerNorm(-1, eps).reshape(this.shape);
  }

  /**
   * Normalises along the batch axis. Training mode — it does not use
   * running statistics and counts from this batch.
   *
   * It differs from `layer_norm` only in which axis is folded. A different
   * axis means changing the folded axis, not standing up a separate
   * function.
   */
  batchNorm(dim = 0, eps = 1e-5): Tensor {
    return this.layerNorm(dim, eps);
  }

  /**
   * Normalises with the channels bundled into groups.
   *
   * It differs from `layerNorm` **only in the span it folds.** With one
   * group the whole channel set is one bundle, which is LayerNorm; with as
   * many groups as channels it is per channel, which is InstanceNorm — the
   * three are special cases of one another, and a wrong bundling rule makes
   * two of the three identical.
   *
   * The bundled span is laid down onto a single last axis and folded. That
   * way `layerNorm`'s formula is used directly, and the normalisation
   * formula lives in one place.
   */
  groupNorm(numGroups: number, eps = 1e-5): Tensor {
    const N = this.shape[0] ?? 1;
    const C = this.shape[1] ?? 1;
    if (C % numGroups !== 0) {
      throw new Error(`cannot split ${C} channels into ${numGroups} groups`);
    }
    const inner = this.size / (N * numGroups);
    return this.reshape([N, numGroups, inner]).layerNorm(-1, eps).reshape(this.shape);
  }

  /**
   * Per sample and per channel. `groupNorm` with the group count set to the
   * channel count.
   */
  instanceNorm(eps = 1e-5): Tensor {
    return this.groupNorm(this.shape[1] ?? 1, eps);
  }

  /**
   * **It does not subtract the mean.** That is the only difference from
   * `layerNorm`.
   *
   * The default eps is f32's machine epsilon rather than `1e-5` — as in
   * torch. Written as `1e-5` to match the other normalisation layers, the
   * forward stayed inside tolerance and **only the gradient** diverged, at
   * a maximum difference of 2.26e-02. It is amplified where the variance is
   * small.
   */
  rmsNorm(dims = 1, eps = 1.1920928955078125e-7): Tensor {
    const rank = this.shape.length;
    const lead = this.shape.slice(0, rank - dims).reduce((a, b) => a * b, 1);
    const flat = this.reshape([lead, this.size / lead]);
    const v = flat.square().mean(-1, true);
    const out = flat.div(v.binary("add", Tensor.full([], eps)).sqrt());
    return out.reshape(this.shape);
  }

  /**
   * `x @ Wᵀ`. torch's `F.linear` takes the weight as (out, in).
   */
  linear(weight: Tensor): Tensor {
    return this.mm(weight.transpose());
  }

  smoothL1Loss(target: Tensor, beta = 1.0,
               reduction: Reduction = "mean"): Tensor {
    const d = this.sub(target);
    const near = d.square().binary("mul", Tensor.full([], 0.5 / beta));
    const far = d.abs().binary("sub", Tensor.full([], 0.5 * beta));
    const isNear = d.abs().binary("lt", Tensor.full([], beta));
    return near.where(isNear, far).reduceAs(reduction);
  }

  // ── 손실과 거리 ───────────────────────────────────────────────────────
  //
  // **접는 방식이 손실의 일부다.** torch 의 손실은 전부 `reduction` 을 받고, 그 값에
  // 따라 원소별·평균·합이 된다. 한 자리에 모아 두면 열셋이 같은 규칙을 쓴다.

  private reduceAs(reduction: Reduction): Tensor {
    if (reduction === "none") return this;
    if (reduction === "sum") return this.sum();
    // **모르는 이름은 멈춘다.** `else` 로 평균에 흘려보내면 `"MEAN"` 을 적은 사람이
    // 자기가 고른 것이 쓰이는 줄 안다 — 값은 나오고 그것이 기본값과 같아서, 인자를
    // 준 적이 없는 것과 구별이 안 된다. torch 도 여기서 멈춘다.
    if (reduction !== "mean") {
      throw new RuntimeError(
        `${reduction} is not a valid value for reduction ` +
          "('none' | 'mean' | 'sum')");
    }
    return this.mean();
  }

  /**
   * **It equals `smoothL1Loss` only at δ=1.**
   *
   * The real relation is `huber(δ) = δ · smoothL1(β=δ)`. Measured at the
   * defaults alone, treating them as one function passes, so the golden
   * cases ask with δ varied.
   */
  huberLoss(target: Tensor, delta = 1.0, reduction: Reduction = "mean"): Tensor {
    const d = this.sub(target);
    const near = d.square().binary("mul", Tensor.full([], 0.5));
    const far = d.abs().binary("sub", Tensor.full([], 0.5 * delta))
      .binary("mul", Tensor.full([], delta));
    const isNear = d.abs().binary("lt", Tensor.full([], delta));
    return near.where(isNear, far).reduceAs(reduction);
  }

  /**
   * `target · (log target − this)`. **This side must already be in logs.**
   *
   * **There are four reductions.** `mean` divides by the element count and
   * `batchmean` by the batch size — the latter is the one that matches the
   * mathematical definition, and torch itself warns that it intends to
   * change.
   */
  klDiv(
    target: Tensor, reduction: Reduction | "batchmean" = "mean", logTarget = false,
  ): Tensor {
    const each = logTarget
      ? target.exp().mul(target.sub(this))
      : target.mul(target.log().sub(this));
    if (reduction === "batchmean") {
      return each.sum().binary("div", Tensor.full([], this.shape[0] ?? 1));
    }
    return each.reduceAs(reduction);
  }

  /**
   * The Poisson negative log-likelihood.
   *
   * **Stirling's correction is added only where `target > 1`** — added
   * unconditionally it is wrong only where the target is small (confirmed
   * by measurement).
   */
  poissonNllLoss(
    target: Tensor, logInput = true, full = false, eps = 1e-8,
    reduction: Reduction = "mean",
  ): Tensor {
    let out = logInput
      ? this.exp().sub(target.mul(this))
      : this.sub(target.mul(this.binary("add", Tensor.full([], eps)).log()));
    if (full) {
      const half = Tensor.full([], 0.5);
      const twoPi = Tensor.full([], 2 * Math.PI);
      const stirling = target.mul(target.log()).sub(target)
        .add(twoPi.mul(target).log().mul(half));
      const big = target.binary("gt", Tensor.full([], 1));
      out = out.add(stirling.where(big, Tensor.zeros(target.shape)));
    }
    return out.reduceAs(reduction);
  }

  /**
   * The Gaussian negative log-likelihood.
   *
   * **The variance is clamped by `eps`.** Unclamped it divides by zero and
   * goes to infinity.
   */
  gaussianNllLoss(
    target: Tensor, variance: Tensor, full = false, eps = 1e-6,
    reduction: Reduction = "mean",
  ): Tensor {
    const safe = variance.binary("maximum", Tensor.full([], eps));
    const d = this.sub(target);
    let out = safe.log().add(d.square().div(safe))
      .binary("mul", Tensor.full([], 0.5));
    if (full) {
      out = out.binary("add", Tensor.full([], 0.5 * Math.log(2 * Math.PI)));
    }
    return out.reduceAs(reduction);
  }

  /**
   * `max(0, −y·(x₁ − x₂) + margin)`.
   */
  marginRankingLoss(
    other: Tensor, target: Tensor, margin = 0.0, reduction: Reduction = "mean",
  ): Tensor {
    return target.neg().mul(this.sub(other))
      .binary("add", Tensor.full([], margin)).unary("relu").reduceAs(reduction);
  }

  /**
   * `1 − cos` at `y=1`, and `max(0, cos − margin)` at `y=−1`.
   */
  cosineEmbeddingLoss(
    other: Tensor, target: Tensor, margin = 0.0, reduction: Reduction = "mean",
  ): Tensor {
    const cos = this.cosineSimilarity(other, 1);
    const same = target.binary("gt", Tensor.full([], 0));
    const positive = Tensor.full([], 1).sub(cos);
    const negative = cos.binary("sub", Tensor.full([], margin)).unary("relu");
    return positive.where(same, negative).reduceAs(reduction);
  }

  /**
   * `x` itself at `y=1`, and `max(0, margin − x)` at `y=−1`.
   *
   * **It is not a split between the two but a sum of them.** torch puts the
   * margin term where `y ≠ 1` and `x` where `y ≠ −1` and **adds them** — at
   * ±1 only one is on and it matches the usual formula, but at `y=0`
   * **both** are on. Split on `y > 0` it diverged quietly there, and
   * `sign()` does produce 0.
   */
  hingeEmbeddingLoss(
    target: Tensor, margin = 1.0, reduction: Reduction = "mean",
  ): Tensor {
    const one = Tensor.full([], 1);
    const notOne = target.binary("ne", one);
    const notNeg = target.binary("ne", Tensor.full([], -1));
    const marginPart = Tensor.full([], margin).sub(this).unary("relu").mul(notOne);
    return marginPart.add(this.mul(notNeg)).reduceAs(reduction);
  }

  /**
   * `log(1 + e^{−y·x})`. Going through `softplus` keeps it from overflowing
   * at large values.
   */
  softMarginLoss(target: Tensor, reduction: Reduction = "mean"): Tensor {
    return target.mul(this).neg().softplus().reduceAs(reduction);
  }

  /**
   * The distance between two paired rows.
   *
   * **`eps` is added to the difference, not to the result.** Asked at `p=1`
   * where the difference is exactly 1.0, it gives 1.0000020 (= 1 + 2·1e-6)
   * — read as added to the result it would be 1.000001, one digit out.
   * Confirmed by measurement.
   */
  pairwiseDistance(other: Tensor, p = 2.0, eps = 1e-6, keepdim = false): Tensor {
    const diff = this.sub(other).binary("add", Tensor.full([], eps));
    const out = diff.vectorNorm(p, this.shape.length - 1);
    return keepdim ? out.reshape([...out.shape, 1]) : out;
  }

  /**
   * The distance between **every pair** within one bundle. It gives the
   * upper triangle only.
   */
  pdist(p = 2.0): Tensor {
    const n = this.shape[0] ?? 0;
    const rows: number[] = [];
    const cols: number[] = [];
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) { rows.push(i); cols.push(j); }
    }
    const pick = (idx: number[]) =>
      this.indexSelect(0, Tensor.from(idx, [idx.length]));
    return pick(rows).sub(pick(cols)).vectorNorm(p, 1);
  }

  /**
   * `max(0, d(a,p) − d(a,n) + margin)`.
   */
  tripletMarginLoss(
    positive: Tensor, negative: Tensor, margin = 1.0, p = 2.0, eps = 1e-6,
    swap = false, reduction: Reduction = "mean",
  ): Tensor {
    const dp = this.pairwiseDistance(positive, p, eps);
    let dn = this.pairwiseDistance(negative, p, eps);
    // 음성이 양성에 더 가까우면 그쪽이 더 어려운 짝이다.
    if (swap) dn = dn.binary("minimum", positive.pairwiseDistance(negative, p, eps));
    return dp.sub(dn).binary("add", Tensor.full([], margin)).unary("relu")
      .reduceAs(reduction);
  }

  /**
   * Independent binary classification per slot, **averaged over the whole
   * class set.**
   */
  multilabelSoftMarginLoss(
    target: Tensor, reduction: Reduction = "mean",
  ): Tensor {
    const each = target.mul(this.logsigmoid())
      .add(Tensor.full([], 1).sub(target).mul(this.neg().logsigmoid()));
    const dim = this.shape.length - 1;
    return each.neg().mean(dim).reduceAs(reduction);
  }

  /**
   * The margin between the target position and the rest.
   *
   * **It divides by the number of classes**, not the number of pairs
   * compared. That means the target position is in the denominator too, and
   * dividing by the pair count gives 3/2 times too much with three classes.
   */
  multiMarginLoss(
    target: Tensor, p = 1, margin = 1.0, weight: Tensor | null = null,
    reduction: Reduction = "mean",
  ): Tensor {
    const rows = this.shape[0] ?? 0;
    const classes = this.shape[1] ?? 0;
    const idx = target.reshape([rows, 1]);
    const correct = this.gather(1, idx);
    let each = Tensor.full([], margin).sub(correct).add(this).unary("relu");
    if (p === 2) each = each.square();
    if (weight) each = each.mul(weight.indexSelect(0, target).reshape([rows, 1]));
    // 정답 자리는 `margin` 이 그대로 남으므로 뺀다.
    const keep = Tensor.ones([rows, classes])
      .scatterSet(1, idx, Tensor.zeros([rows, 1]));
    return each.mul(keep).sumDim(1, false)
      .binary("div", Tensor.full([], classes)).reduceAs(reduction);
  }

  /**
   * **The target is a list of positions, and −1 marks the end.**
   *
   * `[3, 0, -1, 1]` means "3 and 0 are correct", and the trailing 1 is not
   * read. Break that convention and you either count −1 as one of the
   * classes or keep reading past the end.
   *
   * How far to read is decided by **a cumulative sum** — everything after a
   * `−1` is switched off. Scattering the target positions as 0/1 after that
   * means never reading back to the CPU. The reason it scatters by
   * **accumulating** rather than overwriting is that a slot past the end,
   * clamped, could otherwise erase a 1 in front of it.
   */
  multilabelMarginLoss(target: Tensor, reduction: Reduction = "mean"): Tensor {
    const rows = this.shape[0] ?? 0;
    const classes = this.shape[1] ?? 0;
    const stop = target.binary("eq", Tensor.full([], -1));
    const live = stop.cumsum(1).binary("eq", Tensor.full([], 0));
    const safe = target.binary("maximum", Tensor.full([], 0));
    const isPos = Tensor.zeros([rows, classes]).scatterAdd(1, safe, live);
    const isNeg = Tensor.full([], 1).sub(isPos);
    // `diff[r,i,j] = 1 − (x[r,i] − x[r,j])`, 정답 i 와 오답 j 만 센다.
    const asRow = this.reshape([rows, classes, 1]);
    const asCol = this.reshape([rows, 1, classes]);
    const term = Tensor.full([], 1).sub(asRow.sub(asCol)).unary("relu");
    const mask = isPos.reshape([rows, classes, 1])
      .mul(isNeg.reshape([rows, 1, classes]));
    return term.mul(mask).sumDim(2, false).sumDim(1, false)
      .binary("div", Tensor.full([], classes)).reduceAs(reduction);
  }

  /**
   * Batched matrix multiply. `(B, N, K) × (B, K, M)`.
   *
   * It unpacks the batch, repeats `mm`, and stacks — a separate batched
   * kernel would mean fixing two copies alongside `mm`. When batches grow
   * large, that is the place to stand a kernel up.
   */
  bmm(other: Tensor): Tensor {
    if (this.shape.length !== 3 || other.shape.length !== 3) {
      throw new Error(`bmm is 3-D by 3-D: [${this.shape}] x [${other.shape}]`);
    }
    const batch = this.shape[0] ?? 0;
    const parts: Tensor[] = [];
    for (let b = 0; b < batch; b++) {
      parts.push(this.select(0, b).mm(other.select(0, b)));
    }
    return Tensor.stack(parts, 0);
  }

  // ── addmm 계열 ────────────────────────────────────────────────────────
  //
  // 여덟이 전부 `β·this + α·(무슨 곱)` 한 꼴이다. 다른 것은 **곱이 무엇인가**뿐이라
  // 그 하나만 넘긴다.

  /**
   * `β·this + α·product`.
   *
   * **`β == 0` 은 값만 안 보고 그래프에는 남는다.** 둘 다여야 한다 —
   *
   * - `this.mul(0)` 으로 적으면 NaN 을 넣었을 때 결과가 NaN 이 된다. torch 는 멀쩡하다.
   * - 그렇다고 그래프에서 빼면 기울기가 0 이 아니라 **없다.** torch 는 0 을 준다(실측).
   *
   * 두 요구가 반대 방향이고, 평범한 입력으로는 **어느 쪽도** 안 보인다 — NaN 을 넣어야
   * 첫째가, 기울기를 물어야 둘째가 드러난다.
   */
  private blend(product: Tensor, beta: number, alpha: number): Tensor {
    const scaled = alpha === 1 ? product : product.mul(Tensor.full([], alpha));
    if (beta !== 0) {
      const kept = beta === 1 ? this : this.mul(Tensor.full([], beta));
      return kept.add(scaled);
    }
    const mine = this.shape;
    return Tensor.make(
      scaled.buffer,
      scaled.shape,
      [scaled, this],
      (g) => [g, Tensor.zeros(mine)],
      "AddmmBackward0",
    );
  }

  /**
   * `β·this + α·(mat1 @ mat2)`. `this` spreads to the result's shape.
   */
  addmm(mat1: Tensor, mat2: Tensor, beta = 1, alpha = 1): Tensor {
    return this.blend(mat1.mm(mat2), beta, alpha);
  }

  /**
   * **It merges the batch** — multiplies, then sums the batch axis to give
   * two dimensions.
   *
   * One character from `baddbmm` and a different result rank. With a batch
   * of 1 the two look the same, so the cases keep the batch above one.
   */
  addbmm(batch1: Tensor, batch2: Tensor, beta = 1, alpha = 1): Tensor {
    return this.blend(batch1.bmm(batch2).sumDim(0), beta, alpha);
  }

  /**
   * **It keeps the batch.** That is where it parts from `addbmm`.
   */
  baddbmm(batch1: Tensor, batch2: Tensor, beta = 1, alpha = 1): Tensor {
    return this.blend(batch1.bmm(batch2), beta, alpha);
  }

  /**
   * `β·this + α·(mat @ vec)`. The result is one-dimensional.
   */
  addmv(mat: Tensor, vec: Tensor, beta = 1, alpha = 1): Tensor {
    return this.blend(mat.mv(vec), beta, alpha);
  }

  /**
   * `β·this + α·(vec1 ⊗ vec2)`. An outer product, so the result is
   * two-dimensional.
   */
  addr(vec1: Tensor, vec2: Tensor, beta = 1, alpha = 1): Tensor {
    return this.blend(vec1.outer(vec2), beta, alpha);
  }

  /**
   * `this + value·(t1 · t2)`. **There is no `beta`** — `this`'s coefficient
   * is always 1.
   */
  addcmul(tensor1: Tensor, tensor2: Tensor, value = 1): Tensor {
    return this.blend(tensor1.mul(tensor2), 1, value);
  }

  /**
   * `this + value·(t1 / t2)`. The form optimizers use to write an update.
   */
  addcdiv(tensor1: Tensor, tensor2: Tensor, value = 1): Tensor {
    return this.blend(tensor1.div(tensor2), 1, value);
  }

  addmm_(mat1: Tensor, mat2: Tensor, beta = 1, alpha = 1): Tensor {
    return this.mutate(() => this.addmm(mat1, mat2, beta, alpha));
  }

  addbmm_(batch1: Tensor, batch2: Tensor, beta = 1, alpha = 1): Tensor {
    return this.mutate(() => this.addbmm(batch1, batch2, beta, alpha));
  }

  baddbmm_(batch1: Tensor, batch2: Tensor, beta = 1, alpha = 1): Tensor {
    return this.mutate(() => this.baddbmm(batch1, batch2, beta, alpha));
  }

  addmv_(mat: Tensor, vec: Tensor, beta = 1, alpha = 1): Tensor {
    return this.mutate(() => this.addmv(mat, vec, beta, alpha));
  }

  addr_(vec1: Tensor, vec2: Tensor, beta = 1, alpha = 1): Tensor {
    return this.mutate(() => this.addr(vec1, vec2, beta, alpha));
  }

  addcmul_(tensor1: Tensor, tensor2: Tensor, value = 1): Tensor {
    return this.mutate(() => this.addcmul(tensor1, tensor2, value));
  }

  addcdiv_(tensor1: Tensor, tensor2: Tensor, value = 1): Tensor {
    return this.mutate(() => this.addcdiv(tensor1, tensor2, value));
  }

  /**
   * The remainder. **torch's `fmod` follows the sign of the dividend** —
   * unlike Python's `%`.
   *
   * Gradient flows to the dividend as 1. The divisor side is a staircase,
   * so nothing flows.
   */
  fmod(divisor: number): Tensor {
    const d = Tensor.full([], divisor);
    const q = this.div(d).unary("trunc").detach();
    return this.sub(q.binary("mul", d));
  }

  /**
   * The remainder — **with a different sign from `fmod`.**
   *
   * torch's `remainder` (and Python's and torch's `%`) follows the sign of
   * **the divisor** while `fmod` follows the dividend. The remainder of
   * `-7` divided by `3` is 2 on this side and -1 on that. Both being called
   * "the remainder" is the trap, and JavaScript's `%` is the `fmod` one, so
   * using it directly diverges quietly on negative inputs only.
   *
   * The whole difference is truncation (`trunc`) versus flooring (`floor`).
   */
  remainder(divisor: Tensor | number): Tensor {
    // **텐서도 받는다.** torch 는 `x.remainder(y)` 를 쓰고, 수만 받으면 그 줄이
    // 그냥 안 돈다 — 있는데 좁은 이름은 없는 이름보다 찾기 어렵다(`lerpFrom` 이
    // 같은 자리였다).
    const d = Tensor.asTensor(divisor);
    const q = this.div(d).unary("floor").detach();
    return this.sub(q.binary("mul", d));
  }

  /**
   * Cuts from below only. torch's `clamp(min=…)`.
   */
  clampMin(low: number): Tensor {
    const lo = f32lit(low);
    return this.unary(unaryWith(`clampMin<${lo}>`, () => ({
      fwd: `max(x, ${lo})`,
      bwd: `select(0.0, 1.0, x >= ${lo})`,
    })));
  }

  /**
   * Cuts from above only. torch's `clamp(max=…)`.
   */
  clampMax(high: number): Tensor {
    const hi = f32lit(high);
    return this.unary(unaryWith(`clampMax<${hi}>`, () => ({
      fwd: `min(x, ${hi})`,
      bwd: `select(0.0, 1.0, x <= ${hi})`,
    })));
  }

  /**
   * Puts things of differing lengths into one batch.
   *
   * The default is `(longest length, count, …)`, and with `batchFirst` the
   * first two swap. **Gradient returns to each piece** — the padded region
   * is zero, so `pad`'s backward trims it away.
   */
  static padSequence(
    parts: readonly Tensor[],
    batchFirst = false,
    paddingValue = 0,
  ): Tensor {
    if (parts.length === 0) throw new Error("pad_sequence got nothing to pad");
    const longest = Math.max(...parts.map((p) => p.shape[0] ?? 0));
    const padded = parts.map((p) =>
      p.pad(0, 0, longest - (p.shape[0] ?? 0), paddingValue));
    const stacked = Tensor.stack(padded, 0); // (개수, 길이, …)
    return batchFirst ? stacked : stacked.swapaxes(0, 1);
  }

  /**
   * An index as a position marker. One of `n` slots, 1 at its own.
   */
  oneHot(classes: number): Tensor {
    const flat = this.reshape([this.size, 1]);
    const ids = Tensor.arange(classes).reshape([1, classes]);
    return flat.binary("eq", ids, "int64")
      .reshape([...this.shape, classes]);
  }

  /**
   * The negative log-likelihood. `this` must already have been through
   * `log_softmax`.
   *
   * **The graph breaks easily where the target position is extracted.**
   * Return the value detached and no gradient reaches the extracted slot,
   * making the whole classification loss non-differentiable.
   */
  nllLoss(target: Tensor, reduction: Reduction = "mean"): Tensor {
    // **접기 전에 표본별 값을 만든다.** 뽑자마자 평균을 내면 `reduction: "none"`
    // 을 만들 자리가 없어진다 — 스칼라에서 표본별 값을 되살릴 수는 없다.
    const each = this.gather(1, target.reshape([target.size, 1]))
      .reshape([target.size]).neg();
    return each.reduceAs(reduction);
  }

  /**
   * Straight from logits. `log_softmax` and `nll_loss` joined.
   */
  crossEntropy(target: Tensor, reduction: Reduction = "mean"): Tensor {
    return this.logSoftmax(-1).nllLoss(target, reduction);
  }

  // ── 결과 크기가 값에 달린 것들 ────────────────────────────────────────
  //
  // 이 무리는 **몇 개가 나올지를 값을 봐야 안다.** GPU 는 버퍼 크기를 미리 정해야
  // 하므로 값을 한 번 읽어 와야 하고, 그래서 전부 비동기다. 자매도 같은 이유로
  // 여기서 CPU 를 왕복한다.

  /**
   * Picks the true positions into one dimension. **Gradient flows to the
   * picked positions.**
   */
  async maskedSelect(mask: Tensor): Promise<Tensor> {
    const m = await mask.toArray();
    const picks: number[] = [];
    for (const [i, v] of m.entries()) if (v !== 0) picks.push(i);
    return this.flat().indexSelect(0, Tensor.from(picks, [picks.length]));
  }

  /**
   * The coordinates of the non-zeros. Positions are not values, so there is
   * no gradient.
   */
  async nonzero(): Promise<Tensor> {
    const values = await this.toArray();
    const rank = Math.max(1, this.shape.length);
    const dims = this.shape.length === 0 ? [1] : this.shape;
    const rows: number[] = [];
    let count = 0;
    for (const [i, v] of values.entries()) {
      if (v === 0) continue;
      count += 1;
      let rest = i;
      const coord: number[] = new Array<number>(rank).fill(0);
      for (let d = rank - 1; d >= 0; d--) {
        const size = dims[d] ?? 1;
        coord[d] = rest % size;
        rest = Math.floor(rest / size);
      }
      rows.push(...coord);
    }
    return Tensor.from(rows, [count, rank], { dtype: "int64" });
  }

  /**
   * The same as `nonzero` — torch carries both names.
   */
  async argwhere(): Promise<Tensor> {
    return this.nonzero();
  }

  /**
   * The non-zero positions, **a fixed number of them.** Short, it pads;
   * over, it cuts.
   *
   * `nonzero`'s result size depends on the values and so needs one read off
   * the GPU, whereas this one is given the size up front and **in
   * principle** needs no such round trip — that is the name's whole reason.
   * Here it still reads (which positions are non-zero is a value). A place
   * to move into a kernel.
   */
  async nonzeroStatic(size: number, fillValue = -1): Promise<Tensor> {
    const found = await this.nonzero();
    const rank = Math.max(1, this.shape.length);
    const rows = Array.from(await found.toArray());
    const out = new Array<number>(size * rank).fill(fillValue);
    for (let i = 0; i < Math.min(size * rank, rows.length); i++) {
      out[i] = rows[i] ?? fillValue;
    }
    return Tensor.from(out, [size, rank], { dtype: "int64" });
  }

  /**
   * How many in each bucket. **With `min === max` it uses the data's own
   * range** (measured).
   *
   * Given a range it **discards what falls outside** — it does not pile
   * them into the end buckets. Measured with data entirely inside the
   * range, that rule never surfaces.
   */
  async histc(bins = 100, min = 0, max = 0): Promise<Tensor> {
    const values = Array.from(await this.toArray());
    const edges = histEdges(values, bins, min, max);
    return Tensor.from(countInto(values, edges, null), [bins]);
  }

  /**
   * The same count as `histc`, **plus the boundaries.**
   *
   * Given a tensor for `bins`, that is the boundaries — the bucket widths
   * may differ, and then `density` divides by a different value per bucket.
   */
  async histogram(
    bins: number | Tensor = 100,
    range: readonly [number, number] | null = null,
    weight: Tensor | null = null,
    density = false,
  ): Promise<{ hist: Tensor; bin_edges: Tensor }> {
    const values = Array.from(await this.toArray());
    const edges = typeof bins === "number"
      ? histEdges(values, bins, range?.[0] ?? 0, range?.[1] ?? 0)
      : Array.from(await bins.toArray());
    const w = weight === null ? null : Array.from(await weight.toArray());
    let counts = countInto(values, edges, w);
    if (density) {
      const total = counts.reduce((a, b) => a + b, 0) || 1;
      counts = counts.map((c, i) => c / (((edges[i + 1] ?? 0) - (edges[i] ?? 0)) * total));
    }
    return {
      hist: Tensor.from(counts, [counts.length]),
      bin_edges: Tensor.from(edges, [edges.length]),
    };
  }

  /**
   * A histogram over several axes. `this` is `(samples, dimensions)`.
   */
  async histogramdd(bins: number | readonly number[] = 10): Promise<{
    hist: Tensor; bin_edges: Tensor[];
  }> {
    const values = Array.from(await this.toArray());
    const rows = this.shape[0] ?? 0;
    const dims = this.shape[1] ?? 1;
    const counts = typeof bins === "number"
      ? new Array<number>(dims).fill(bins)
      : [...bins];
    const edges = counts.map((n, d) => {
      const column: number[] = [];
      for (let r = 0; r < rows; r++) column.push(values[r * dims + d] ?? 0);
      return histEdges(column, n, 0, 0);
    });
    const total = counts.reduce((a, b) => a * b, 1);
    const hist = new Array<number>(total).fill(0);
    for (let r = 0; r < rows; r++) {
      let flat = 0;
      let inside = true;
      for (let d = 0; d < dims; d++) {
        const slot = slotOf(values[r * dims + d] ?? 0, edges[d]!);
        if (slot < 0) { inside = false; break; }
        flat = flat * (counts[d] ?? 1) + slot;
      }
      if (inside) hist[flat] = (hist[flat] ?? 0) + 1;
    }
    return {
      hist: Tensor.from(hist, counts),
      bin_edges: edges.map((e) => Tensor.from(e, [e.length])),
    };
  }

  /**
   * The most frequent value. **On an equal count the smaller value wins,
   * and the index is that value's last occurrence** (measured: `[4,4,5,5]`
   * gives value 4 at index 1).
   *
   * Measured with data that has no ties, that rule never surfaces.
   */
  async mode(dim = -1, keepdim = false): Promise<{
    values: Tensor; indices: Tensor;
  }> {
    return this.alongAxis(dim, keepdim, (line) => {
      const sorted = [...new Set(line)].sort((a, b) => a - b);
      let best = sorted[0] ?? 0;
      let bestCount = -1;
      for (const value of sorted) {
        const count = line.filter((v) => v === value).length;
        if (count > bestCount) { best = value; bestCount = count; }
      }
      return { value: best, index: line.lastIndexOf(best) };
    });
  }

  /**
   * The median counted **excluding** NaN. `median` returns NaN if there is
   * even one (measured).
   *
   * With an even count it **takes the lower** — it does not average.
   */
  async nanmedian(dim?: number, keepdim = false): Promise<
    Tensor | { values: Tensor; indices: Tensor }
  > {
    if (dim === undefined) {
      const clean = Array.from(await this.toArray()).filter((v) => !Number.isNaN(v));
      const sorted = [...clean].sort((a, b) => a - b);
      const pick = sorted[(sorted.length - 1) >> 1] ?? Number.NaN;
      if (Number.isNaN(pick)) return Tensor.from([pick], []);
      // **번호를 안 건네므로 고르게 나눈다.** NaN 칸은 `eq` 가 거짓이라 저절로 빠진다.
      return this.flat().spreadEqual(Tensor.full([], pick));
    }
    return this.alongAxis(dim, keepdim, (line) => {
      const keep = line.map((v, i) => [v, i] as const)
        .filter(([v]) => !Number.isNaN(v))
        .sort((a, b) => a[0] - b[0]);
      const at = keep[(keep.length - 1) >> 1];
      return { value: at?.[0] ?? Number.NaN, index: at?.[1] ?? 0 };
    });
  }

  /**
   * 축 하나를 따라 줄마다 값 하나와 자리 하나를 뽑는다.
   *
   * `mode` 와 `nanmedian` 이 같은 뼈대다 — 다른 것은 **줄 하나에서 무엇을 고르는가**
   * 뿐이라 그 하나만 넘긴다.
   */
  private async alongAxis(
    dim: number,
    keepdim: boolean,
    pick: (line: number[]) => { value: number; index: number },
  ): Promise<{ values: Tensor; indices: Tensor }> {
    const rank = this.shape.length;
    const axis = dim < 0 ? dim + rank : dim;
    const values = Array.from(await this.toArray());
    const st = rowStrides(this.shape);
    const outShape = this.shape.filter((_, d) => d !== axis);
    const len = this.shape[axis] ?? 0;
    const vals: number[] = [];
    const idx: number[] = [];
    eachCoord(outShape, (c) => {
      let base = 0;
      let k = 0;
      for (let d = 0; d < rank; d++) if (d !== axis) base += c[k++]! * st[d]!;
      const line: number[] = [];
      for (let i = 0; i < len; i++) line.push(values[base + i * st[axis]!] ?? 0);
      const got = pick(line);
      vals.push(got.value);
      idx.push(got.index);
    });
    const shape = keepdim
      ? this.shape.map((s, d) => (d === axis ? 1 : s))
      : outShape;
    void vals;
    // **값을 손으로 만들지 않고 뽑아 온다.** `Tensor.from(vals, …)` 는 값이 맞고
    // **그래프가 없다** — 값 검사는 전부 통과하고 `backward()` 에서야 드러나는데,
    // 그때 나오는 말이 "requires_grad 가 아니다" 라 **사용자를 가리킨다.**
    //
    // 번호를 건네는 연산이므로 규칙도 이쪽이 맞다: 기울기가 **고른 자리 하나로만**
    // 간다(실측). 뽑아 오면 그 규칙이 `gather` 의 역방향에서 저절로 나온다.
    const lifted = this.shape.map((s, d) => (d === axis ? 1 : s));
    const at = Tensor.from(idx, lifted, { dtype: "int64" });
    return {
      values: this.gather(axis, at).reshape(shape),
      indices: at.reshape(shape),
    };
  }

  /**
   * 값이 같은 칸에 기울기를 **고르게 나눈다.**
   *
   * 번호를 안 건네는 축약(`median()`·`max()`·`nanmedian()`)의 규칙이다(실측:
   * `[3,5,5,1,5]` 의 `median()` 기울기가 세 5 에 ⅓ 씩). 한 자리로 몰아주면 값은
   * 같고 기울기만 갈리는데, **동점이 없는 자료로 재면 어떤 규칙이든 같은 답**이라
   * 표가 아무것도 안 묻는다.
   *
   * 마스크로 곱해 더하고 개수로 나눈다 — 값은 그대로(같은 칸끼리 더해 개수로 나눔)
   * 이고 역방향이 `mask/개수` 라 규칙이 저절로 나온다. `mask` 는 비교라 기울기가
   * 없고, 개수도 끊어 둔다.
   */
  private spreadEqual(value: Tensor): Tensor {
    const hit = this.binary("eq", value, "bool").detach();
    const count = hit.to("float32").sum().detach();
    // **곱하면 안 된다 — `0 × NaN` 은 NaN 이다.** 마스크로 곱해 더하는 판이 먼저
    // 있었고, `nanmedian` 이 NaN 을 품은 자료에서 통째로 NaN 이 됐다. 이 저장소가
    // 같은 자리에서 세 번째로 물린 것이라(코어의 `median`, borch.ts 의 `median`,
    // 여기) 골라야 한다 — `where` 는 안 고른 칸을 **계산에 안 넣는다.**
    return this.where(hit, Tensor.zeros(this.shape)).sum().div(count);
  }

  /**
   * The central difference. **One per axis, returned as a bundle** — no
   * axis given, it is all of them.
   *
   * At `edgeOrder` 1 the ends use a one-sided difference; at 2 they are
   * fitted with a quadratic (measured: on `x²`, 2 gives the exact
   * derivative and 1 is off at both ends).
   */
  async gradient(spacing: number | readonly number[] = 1, dim?: number | readonly number[],
    edgeOrder = 1): Promise<Tensor[]> {
    const rank = this.shape.length;
    const axes = dim === undefined
      ? Array.from({ length: rank }, (_, i) => i)
      : (typeof dim === "number" ? [dim] : [...dim]);
    const steps = typeof spacing === "number"
      ? axes.map(() => spacing)
      : [...spacing];
    const values = Array.from(await this.toArray());
    const st = rowStrides(this.shape);
    const outs: Tensor[] = [];
    for (const [k, raw] of axes.entries()) {
      const axis = raw < 0 ? raw + rank : raw;
      const gap = steps[k] ?? 1;
      const len = this.shape[axis] ?? 0;
      const got = new Array<number>(values.length).fill(0);
      eachCoord(this.shape.filter((_, d) => d !== axis), (c) => {
        let base = 0;
        let j = 0;
        for (let d = 0; d < rank; d++) if (d !== axis) base += c[j++]! * st[d]!;
        const at = (i: number): number => values[base + i * st[axis]!] ?? 0;
        const put = (i: number, v: number): void => {
          got[base + i * st[axis]!] = v;
        };
        for (let i = 1; i < len - 1; i++) put(i, (at(i + 1) - at(i - 1)) / (2 * gap));
        if (len >= 2 && edgeOrder === 1) {
          put(0, (at(1) - at(0)) / gap);
          put(len - 1, (at(len - 1) - at(len - 2)) / gap);
        } else if (len >= 3) {
          // 이차식으로 맞춘 한쪽 차분. `x²` 에서 정확해진다.
          put(0, (-3 * at(0) + 4 * at(1) - at(2)) / (2 * gap));
          put(len - 1,
            (3 * at(len - 1) - 4 * at(len - 2) + at(len - 3)) / (2 * gap));
        }
      });
      outs.push(Tensor.from(got, [...this.shape]));
    }
    return outs;
  }

  /**
   * Adds **the mean of each neighbouring pair times the spacing.**
   * Trapezoidal integration.
   *
   * It was absent here for a long time. The binding was assembling the same
   * few lines in Python, and the comment said "making one more here would
   * be a second copy of the assembly". What that missed is that **the name
   * did not exist at all for anyone using borch.ts from TypeScript** — the
   * assembly was not one copy; it was only on the Python side.
   *
   * It is only slicing and adding, so **the gradient flows on its own.**
   * There is nothing to write by hand.
   *
   * @param x the positions given directly. Left out, the spacing is uniform
   *   at `dx`.
   */
  trapezoid(x?: Tensor, dx = 1, dim = -1): Tensor {
    const { pieces, axis } = this.trapezoidPieces(x, dx, dim);
    return pieces.sumDim(axis);
  }

  /**
   * The cumulative version. **The last value must equal `trapezoid`** —
   * that is the check.
   */
  cumulativeTrapezoid(x?: Tensor, dx = 1, dim = -1): Tensor {
    const { pieces, axis } = this.trapezoidPieces(x, dx, dim);
    return pieces.cumsum(axis);
  }

  private trapezoidPieces(
    x: Tensor | undefined, dx: number, dim: number,
  ): { pieces: Tensor; axis: number } {
    const rank = this.shape.length;
    const axis = dim < 0 ? dim + rank : dim;
    const n = this.shape[axis] ?? 0;
    const both = this.narrow(axis, 0, n - 1).add(this.narrow(axis, 1, n - 1));
    if (x === undefined) {
      return { pieces: both.mul(Tensor.full([], dx / 2)), axis };
    }
    const step = x.narrow(axis, 1, n - 1).sub(x.narrow(axis, 0, n - 1));
    return { pieces: both.mul(step).mul(Tensor.full([], 0.5)), axis };
  }

  /**
   * The distinct values, **ascending.** torch's default is to give them
   * sorted.
   */
  async unique(): Promise<Tensor> {
    const values = Array.from(await this.toArray());
    const seen = [...new Set(values)].sort((a, b) => a - b);
    return Tensor.from(seen, [seen.length], { dtype: this.dtype });
  }

  /**
   * Collapses **consecutive** duplicates only. Unlike `unique` it does not
   * sort — `[1,1,2,2,1]` becomes `[1,2,1]` (measured). Measured with sorted
   * input alone, the two look the same.
   *
   * The result's **length depends on the values**, so it reads once here.
   * The same place as `unique` and `bincount`, which is why those three
   * alone are asynchronous.
   */
  async uniqueConsecutive(
    returnInverse = false,
    returnCounts = false,
    dim?: number,
  ): Promise<Tensor | Tensor[]> {
    const values = Array.from(await this.toArray());
    const rank = this.shape.length;
    const axis = dim === undefined ? null : (dim < 0 ? dim + rank : dim);
    const rows = axis === null ? values.length : (this.shape[axis] ?? 0);
    const width = axis === null ? 1 : values.length / Math.max(1, rows);
    const st = rowStrides(this.shape);
    // 축을 앞으로 눕혀 줄 단위로 본다 — 축이 없으면 칸 하나가 곧 한 줄이다.
    const row = (r: number): number[] => {
      if (axis === null) return [values[r] ?? 0];
      const out: number[] = [];
      eachCoord(this.shape.filter((_, d) => d !== axis), (c) => {
        let p = r * st[axis]!;
        let k = 0;
        for (let d = 0; d < rank; d++) if (d !== axis) p += c[k++]! * st[d]!;
        out.push(values[p] ?? 0);
      });
      return out;
    };
    const keptRows: number[] = [];
    const inverse: number[] = [];
    const counts: number[] = [];
    let prev: number[] | null = null;
    for (let r = 0; r < rows; r++) {
      const cur = row(r);
      const same = prev !== null && cur.every((v, i) => v === prev![i]);
      if (!same) {
        keptRows.push(r);
        counts.push(0);
      }
      counts[counts.length - 1] = (counts[counts.length - 1] ?? 0) + 1;
      inverse.push(keptRows.length - 1);
      prev = cur;
    }
    let out: Tensor;
    if (axis === null) {
      out = Tensor.from(keptRows.map((r) => values[r] ?? 0), [keptRows.length],
        { dtype: this.dtype });
    } else {
      const spots = new Float32Array(keptRows.length * width);
      let at = 0;
      for (const r of keptRows) for (const v of rowSpots(this.shape, axis, r, st)) {
        spots[at++] = v;
      }
      const shape = this.shape.map((s, d) => (d === axis ? keptRows.length : s));
      out = this.gatherSpots(Tensor.spotsTensor(spots), shape);
    }
    if (!returnInverse && !returnCounts) return out;
    const extra: Tensor[] = [out];
    if (returnInverse) {
      extra.push(Tensor.from(inverse,
        axis === null ? [...this.shape] : [rows], { dtype: "int64" }));
    }
    if (returnCounts) {
      extra.push(Tensor.from(counts, [counts.length], { dtype: "int64" }));
    }
    return extra;
  }

  /** 정수 값마다 몇 번 나왔는가. 길이는 가장 큰 값이 정한다. */
  /**
   * How many times each bucket appears.
   *
   * **The type diverges** (measured): without weights it is `int64`; with
   * weights it is the weights' type — counting and summing are different
   * jobs.
   *
   * @param weights given, it **adds the weights** rather than counting.
   * @param minlength the result's minimum length. It reserves slots for
   *   buckets that never appeared.
   */
  async bincount(weights?: Tensor, minlength = 0): Promise<Tensor> {
    const values = await this.toArray();
    const w = weights === undefined ? null : await weights.toArray();
    let top = 0;
    for (const v of values) top = Math.max(top, Math.trunc(v));
    const size = Math.max(top + 1, minlength);
    const counts = new Float32Array(size);
    for (const [i, v] of values.entries()) {
      const at = Math.trunc(v);
      counts[at] = (counts[at] ?? 0) + (w === null ? 1 : (w[i] ?? 0));
    }
    return Tensor.from(counts, [counts.length],
      { dtype: w === null ? "int64" : (weights?.dtype ?? "float32") });
  }

  /**
   * Quantiles. **Linear interpolation** — that is torch's default, and it
   * differs from picking the nearest value.
   */
  async quantile(q: number | readonly number[]): Promise<Tensor> {
    return this.quantileOver(Array.from(await this.toArray()), q);
  }

  /**
   * 분위수의 몸통 — **정렬 자리로 되짚어 뽑는다.**
   *
   * 값을 손으로 만들면 그래프가 없다. 되짚어 뽑으면 기울기가 **보간에 쓴 두 자리로**
   * 가는데, 그것이 이 연산의 규칙이다(실측). 동점일 때 `median` 과 갈리는 자리가
   * 여기다 — `[1,5,5,5]` 에서 `median` 은 세 5 에 ⅓ 씩이고 `quantile(0.5)` 는
   * **앞의 두 5 에 ½ 씩**이다. 값은 둘 다 5 라, 되짚어야만 갈린다.
   */
  private quantileOver(host: number[], q: number | readonly number[]): Tensor {
    const order = host.map((_, i) => i).sort((a, b) => (host[a]! - host[b]!));
    const flat = this.flat();
    const wanted = typeof q === "number" ? [q] : [...q];
    const parts = wanted.map((p) => {
      const at = p * (order.length - 1);
      const lo = Math.floor(at);
      const hi = Math.min(lo + 1, order.length - 1);
      const w = at - lo;
      const pickLo = flat.select(0, order[lo] ?? 0);
      if (w === 0) return pickLo;
      const pickHi = flat.select(0, order[hi] ?? 0);
      return pickLo.mul(Tensor.full([], 1 - w))
        .add(pickHi.mul(Tensor.full([], w)));
    });
    if (typeof q === "number") return parts[0] as Tensor;
    return Tensor.stack(parts, 0);
  }

  /**
   * Quantiles counted excluding NaN.
   *
   * **Removing NaN shifts the positions.** So a small tensor of the intact
   * slots is built and the work is retraced on that — the way back to the
   * original positions is `indexSelect`, which is already differentiable,
   * so it stays connected.
   */
  async nanquantile(q: number | readonly number[]): Promise<Tensor> {
    const values = Array.from(await this.toArray());
    const keep = values.map((v, i) => [v, i] as const)
      .filter(([v]) => !Number.isNaN(v));
    const at = Tensor.from(keep.map(([, i]) => i), [keep.length],
      { dtype: "int64" });
    const clean = this.flat().indexSelect(0, at);
    return clean.quantileOver(keep.map(([v]) => v), q);
  }

  // ── 선형대수 ──────────────────────────────────────────────────────────

  /**
   * CPU 로 읽어 와 **행렬 묶음**으로 본다. 선형대수가 전부 여기를 지난다.
   *
   * **왕복이 생긴다.** 그래도 그렇게 하는 이유는 `src/linalg.ts` 에 적었다 —
   * 분해는 순차적이라 GPU 로 펼 자리가 거의 없고, 여기서 미는 크기에서는 커널을
   * 띄우는 값이 계산보다 비싸다.
   *
   * **마지막 두 축만 행렬이고 앞은 전부 배치다.** torch 의 `linalg` 가 그 규칙이라
   * `det((3,2,2))` 이 `(3,)` 을 낸다. 전에는 2 차원 정사각 하나만 받았는데, 그건
   * 흉내가 아니라 없는 것이었다 — 배치는 실제 코드가 늘 쓰는 모양이다.
   */
  private async asBatch(square = true): Promise<{
    mats: LA.Mat[]; rows: number; cols: number; batch: number; lead: number[];
  }> {
    const rank = this.shape.length;
    if (rank < 2) {
      throw new RuntimeError(
        `linalg: needs two or more dimensions — got [${this.shape}]`,
      );
    }
    const rows = this.shape[rank - 2] ?? 0;
    const cols = this.shape[rank - 1] ?? 0;
    if (square && rows !== cols) {
      throw new RuntimeError(
        `linalg: needs a square matrix — got [${this.shape}]`,
      );
    }
    const lead = [...this.shape.slice(0, rank - 2)];
    const batch = lead.reduce((a, b) => a * b, 1);
    const flat = LA.fromF32(await this.toArray());
    const size = rows * cols;
    const mats: LA.Mat[] = [];
    for (let b = 0; b < batch; b++) mats.push(flat.slice(b * size, (b + 1) * size));
    return { mats, rows, cols, batch, lead };
  }

  /** CPU 에서 만든 값을 다시 GPU 로. */
  private static fromMat(a: LA.Mat, shape: readonly number[]): Tensor {
    return Tensor.from(LA.toF32(a), shape);
  }

  /** 배치별로 나온 것들을 한 텐서로 잇는다. */
  private static fromBatch(mats: readonly LA.Mat[], shape: readonly number[]): Tensor {
    const total = shape.reduce((a, b) => a * b, 1);
    const out = new Float32Array(total);
    let off = 0;
    for (const m of mats) {
      out.set(LA.toF32(m), off);
      off += m.length;
    }
    return Tensor.from(out, shape);
  }

  /**
   * 배치마다 **다른 상수**를 쓰는 역방향.
   *
   * 값이 배치별로 나왔으니 역방향의 상수(역행렬·`L`·`L⁻¹`)도 배치별로 다르다. `g` 를
   * 배치 축 하나로 편 뒤 한 장씩 돌리고 다시 쌓는다. 배치가 하나면 그냥 넘긴다 —
   * 그쪽이 흔하고, 쌓기를 한 번 덜 하는 만큼 빠르다.
   */
  private static perBatch(
    g: Tensor,
    batch: number,
    gItemShape: readonly number[],
    outShape: readonly number[],
    fn: (gb: Tensor, b: number) => Tensor,
  ): Tensor {
    if (batch === 1) return fn(g.reshape(gItemShape), 0).reshape(outShape);
    const g2 = g.reshape([batch, ...gItemShape]);
    const parts: Tensor[] = [];
    for (let b = 0; b < batch; b++) parts.push(fn(g2.select(0, b), b));
    return Tensor.stack(parts, 0).reshape(outShape);
  }

  /** 배치별 역행렬. 특이행렬이면 조용히 NaN 을 내지 않고 던진다. */
  private static invAll(mats: readonly LA.Mat[], n: number, what: string): LA.Mat[] {
    return mats.map((a) => {
      if (LA.lu(a, n).singular) {
        throw new LinAlgError(
          `linalg.${what}: the matrix is singular — it has no inverse`,
        );
      }
      return LA.inverse(a, n);
    });
  }

  /**
   * The determinant. The backward is `det·A⁻ᵀ`.
   */
  async det(): Promise<Tensor> {
    const v = await this.asBatch();
    const vals = v.mats.map((a) => LA.det(LA.lu(a, v.rows)));
    const invTs = Tensor.invAll(v.mats, v.rows, "det")
      .map((m) => LA.transpose(m, v.rows, v.rows));
    return this.linalgNode(Tensor.from(vals, v.lead), (g) =>
      Tensor.perBatch(g, v.batch, [], this.shape, (gb, b) =>
        Tensor.fromMat(invTs[b]!, [v.rows, v.rows])
          .mul(gb.mul(Tensor.from([vals[b]!], [])))), "DetBackward0");
  }

  /**
   * `log|det|`. The backward is `A⁻ᵀ` — more stable, with no determinant
   * multiplied in.
   */
  async logdet(): Promise<Tensor> {
    const v = await this.asBatch();
    const vals = v.mats.map((a) => {
      const { sign, logabs } = LA.slogdet(LA.lu(a, v.rows));
      return sign > 0 ? logabs : NaN;
    });
    const invTs = Tensor.invAll(v.mats, v.rows, "logdet")
      .map((m) => LA.transpose(m, v.rows, v.rows));
    return this.linalgNode(Tensor.from(vals, v.lead), (g) =>
      Tensor.perBatch(g, v.batch, [], this.shape, (gb, b) =>
        Tensor.fromMat(invTs[b]!, [v.rows, v.rows]).mul(gb)), "LogdetBackward0");
  }

  /**
   * The sign and the log absolute value, separately. No digits are lost
   * even when the determinant is very small.
   */
  async slogdet(): Promise<{ sign: Tensor; logabs: Tensor }> {
    const v = await this.asBatch();
    const parts = v.mats.map((a) => LA.slogdet(LA.lu(a, v.rows)));
    const invTs = Tensor.invAll(v.mats, v.rows, "slogdet")
      .map((m) => LA.transpose(m, v.rows, v.rows));
    return {
      // 부호는 계단이라 안 흐른다.
      sign: Tensor.from(parts.map((p) => p.sign), v.lead),
      logabs: this.linalgNode(Tensor.from(parts.map((p) => p.logabs), v.lead), (g) =>
        Tensor.perBatch(g, v.batch, [], this.shape, (gb, b) =>
          Tensor.fromMat(invTs[b]!, [v.rows, v.rows]).mul(gb)), "SlogdetBackward0"),
    };
  }

  /**
   * The inverse. The backward is `-A⁻ᵀ·Ḡ·A⁻ᵀ`.
   */
  async inverse(): Promise<Tensor> {
    const v = await this.asBatch();
    const invs = Tensor.invAll(v.mats, v.rows, "inv");
    const n = v.rows;
    return this.linalgNode(Tensor.fromBatch(invs, this.shape), (g) =>
      Tensor.perBatch(g, v.batch, [n, n], this.shape, (gb, b) => {
        const t = Tensor.fromMat(LA.transpose(invs[b]!, n, n), [n, n]);
        return t.mm(gb).mm(t).neg();
      }), "InverseBackward0");
  }

  /**
   * The same as `inv` except that it **does not throw** — it puts a
   * non-zero number in `info` instead.
   */
  async invEx(): Promise<{ inverse: Tensor; info: Tensor }> {
    const v = await this.asBatch();
    try {
      return { inverse: await this.inverse(), info: Tensor.zeros(v.lead) };
    } catch (e) {
      if (!(e instanceof LinAlgError)) throw e;
      return {
        inverse: Tensor.full(this.shape, Infinity),
        info: Tensor.full(v.lead, SINGULAR_INFO),
      };
    }
  }

  /**
   * The pseudoinverse. **It has a gradient** — in three terms.
   *
   *     Ā = −Pᵀ·Ḡ·Pᵀ + (I − A·P)·Ḡᵀ·P·Pᵀ + Pᵀ·P·Ḡᵀ·(I − P·A)
   *
   * The last two terms **go to zero for a square invertible matrix.** There
   * `I − AP` and `I − PA` are both zero, only the first term remains, and
   * that first term is the same formula as the inverse's gradient. So
   * leaving the two out is right on squares and wrong only on rectangles —
   * the golden cases ask with rectangles too.
   */
  async pinverse(): Promise<Tensor> {
    const v = await this.asBatch(false);
    const { rows: m, cols: n } = v;
    const ps = v.mats.map((a) => LA.pinverse(a, m, n));
    const out = Tensor.fromBatch(ps, [...v.lead, n, m]);
    return this.linalgNode(out, (g) =>
      Tensor.perBatch(g, v.batch, [n, m], this.shape, (gb, b) => {
        const a = v.mats[b]!;
        const p = ps[b]!;
        const pt = Tensor.fromMat(LA.transpose(p, n, m), [m, n]);
        const eyeA = LA.matmul(a, p, m, n, m);
        const eyeB = LA.matmul(p, a, n, m, n);
        const c2 = new Float64Array(m * m);
        for (let i = 0; i < m * m; i++) c2[i] = -(eyeA[i] ?? 0);
        for (let i = 0; i < m; i++) c2[i * m + i] = (c2[i * m + i] ?? 0) + 1;
        const c4 = new Float64Array(n * n);
        for (let i = 0; i < n * n; i++) c4[i] = -(eyeB[i] ?? 0);
        for (let i = 0; i < n; i++) c4[i * n + i] = (c4[i * n + i] ?? 0) + 1;
        const ppt = Tensor.fromMat(
          LA.matmul(p, LA.transpose(p, n, m), n, m, n), [n, n]);
        const ptp = Tensor.fromMat(
          LA.matmul(LA.transpose(p, n, m), p, m, n, m), [m, m]);
        void b;
        const gt = gb.transpose();
        return pt.mm(gb).mm(pt).neg()
          .add(Tensor.fromMat(c2, [m, m]).mm(gt).mm(ppt))
          .add(ptp.mm(gt).mm(Tensor.fromMat(c4, [n, n])));
      }), "PinverseBackward0");
  }

  /**
   * 아래 삼각 콜레스키. 대칭 양정부호가 아니면 던진다.
   *
   * **역방향을 GPU 에서 쓴다.** 식은 `Ā = sym(L⁻ᵀ·Φ(Lᵀ·L̄)·L⁻¹)` 이고 `Φ` 는 아래
   * 삼각을 취하되 대각을 반으로 줄이는 것인데, 전부 행렬 연산이라 이미 있는 커널로
   * 적힌다. `L` 과 `L⁻¹` 만 순방향에서 CPU 로 구해 상수로 들고 온다 — 역방향 안에서는
   * GPU 를 기다릴 수가 없으므로 그 값이 미리 있어야 한다.
   */
  /**
   * **Written with a flip.** `Lᵀ` is `L` transposed and transposition is
   * already differentiable, so the factorisation is not written twice — the
   * same formula in two copies eventually gets fixed in one of them.
   *
   * @param upper true gives the upper triangle — `Lᵀ` instead of `L`.
   */
  async cholesky(upper = false): Promise<Tensor> {
    if (upper) return (await this.cholesky()).transpose();
    const v = await this.asBatch();
    const n = v.rows;
    const ls = v.mats.map((a) => {
      try {
        return LA.cholesky(a, n);
      } catch {
        throw new LinAlgError(
          "linalg.cholesky: the input is not symmetric positive definite (a leading principal minor is not positive)",
        );
      }
    });
    const linvs = ls.map((l) => LA.inverse(l, n));
    const eye = Tensor.eye(n);
    const half = Tensor.full([], 0.5);
    return this.linalgNode(Tensor.fromBatch(ls, this.shape), (g) =>
      Tensor.perBatch(g, v.batch, [n, n], this.shape, (gb, b) => {
        const lt = Tensor.fromMat(LA.transpose(ls[b]!, n, n), [n, n]);
        const linvT = Tensor.fromMat(LA.transpose(linvs[b]!, n, n), [n, n]);
        const linvC = Tensor.fromMat(linvs[b]!, [n, n]);
        const m = lt.mm(gb);
        // Φ — 아래 삼각을 남기고 대각만 반으로.
        const p = m.tril().sub(m.mul(eye).binary("mul", half));
        const abar = linvT.mm(p).mm(linvC);
        // A 가 대칭이라 위·아래 삼각이 같은 자유도를 나눠 갖는다 — 대칭화가 그 몫이다.
        return abar.add(abar.transpose()).binary("mul", half);
      }), "CholeskyBackward0");
  }

  /**
   * The non-throwing side of `cholesky`.
   */
  async choleskyEx(): Promise<{ L: Tensor; info: Tensor }> {
    const v = await this.asBatch();
    try {
      return { L: await this.cholesky(), info: Tensor.zeros(v.lead) };
    } catch (e) {
      if (!(e instanceof LinAlgError)) throw e;
      return {
        L: Tensor.full(this.shape, NaN),
        info: Tensor.full(v.lead, SINGULAR_INFO),
      };
    }
  }

  /**
   * `A x = b`. The backward is `A⁻ᵀ·Ḡ` towards `b` and `-A⁻ᵀ·Ḡ·xᵀ` towards
   * `A`.
   *
   * When `b` has one axis fewer than `A` it is read as **a bundle of
   * vectors** — torch's rule. The backward's outer product hangs on that
   * distinction.
   */
  async solve(b: Tensor): Promise<Tensor> {
    const v = await this.asBatch();
    const n = v.rows;
    const vector = b.shape.length === this.shape.length - 1;
    const cols = vector ? 1 : (b.shape[b.shape.length - 1] ?? 1);
    const rhsFlat = LA.fromF32(await b.toArray());
    const invTs = Tensor.invAll(v.mats, n, "solve")
      .map((m) => LA.transpose(m, n, n));
    const size = n * cols;
    const xs: LA.Mat[] = [];
    for (let i = 0; i < v.batch; i++) {
      xs.push(LA.luSolveFactored(
        LA.luFactor(v.mats[i]!, n, n), rhsFlat.slice(i * size, (i + 1) * size), cols));
    }
    const out = Tensor.fromBatch(xs, b.shape);
    const shape = this.shape;
    const gShape = vector ? [n] : [n, cols];
    return Tensor.make(
      out.buffer,
      b.shape,
      [this, b],
      (g) => {
        const gbs: Tensor[] = [];
        const ga = Tensor.perBatch(g, v.batch, gShape, shape, (gi, i) => {
          const at = Tensor.fromMat(invTs[i]!, [n, n]);
          const gb = at.mm(gi.reshape([n, cols]));
          gbs.push(gb.reshape(gShape));
          return gb.mm(Tensor.fromMat(xs[i]!, [n, cols]).transpose()).neg();
        });
        const gb = v.batch === 1
          ? gbs[0]!.reshape(b.shape)
          : Tensor.stack(gbs, 0).reshape(b.shape);
        return [
          this.requiresGrad ? ga : null,
          b.requiresGrad ? gb : null,
        ];
      },
      "SolveBackward0",
    );
  }

  /**
   * The non-throwing side of `solve`.
   */
  async solveEx(b: Tensor): Promise<{ result: Tensor; info: Tensor }> {
    const v = await this.asBatch();
    try {
      return { result: await this.solve(b), info: Tensor.zeros(v.lead) };
    } catch (e) {
      if (!(e instanceof LinAlgError)) throw e;
      return {
        result: Tensor.full(b.shape, Infinity),
        info: Tensor.full(v.lead, SINGULAR_INFO),
      };
    }
  }

  /**
   * QR factorisation. **`reduced` has a gradient.**
   *
   *     N = Qᵀ·Q̄ − R̄·Rᵀ
   *     Ā = [Q̄ + Q·(tril(N − Nᵀ, −1) − N)]·R⁻ᵀ
   *
   * Keeping only the lower triangle is the whole of this derivation —
   * differentiating `QᵀQ = I` makes `Qᵀ·dQ` antisymmetric, and the upper
   * half is the lower half's mirror, so there is nothing separate to count.
   *
   * `complete` has leftover columns in `Q` and a different derivation. That
   * side returns values only.
   */
  async qr(mode: "reduced" | "complete" = "reduced"): Promise<{ q: Tensor; r: Tensor }> {
    const v = await this.asBatch(false);
    const { rows, cols } = v;
    const k = Math.min(rows, cols);
    const qs: LA.Mat[] = [];
    const rs: LA.Mat[] = [];
    for (const a of v.mats) {
      const { q, r } = LA.qr(a, rows, cols);
      if (mode === "complete") {
        qs.push(q);
        rs.push(r);
        continue;
      }
      const qCut = new Float64Array(rows * k);
      for (let i = 0; i < rows; i++) {
        for (let j = 0; j < k; j++) qCut[i * k + j] = q[i * rows + j] ?? 0;
      }
      qs.push(qCut);
      rs.push(r.slice(0, k * cols));
    }
    const qShape = mode === "complete" ? [rows, rows] : [rows, k];
    const rShape = mode === "complete" ? [rows, cols] : [k, cols];
    const qOut = Tensor.fromBatch(qs, [...v.lead, ...qShape]);
    const rOut = Tensor.fromBatch(rs, [...v.lead, ...rShape]);
    if (mode === "complete" || rows < cols) return { q: qOut, r: rOut };

    const zeroQ = Tensor.zeros(qShape);
    const zeroR = Tensor.zeros(rShape);
    const back = (gq: Tensor, gr: Tensor, b: number): Tensor => {
      const q = Tensor.fromMat(qs[b]!, qShape);
      const rt = Tensor.fromMat(LA.transpose(rs[b]!, k, cols), [cols, k]);
      const rInvT = Tensor.fromMat(
        LA.transpose(LA.inverse(rs[b]!, k), k, k), [k, k]);
      const n = q.transpose().mm(gq).sub(gr.mm(rt));
      const inner = n.sub(n.transpose()).tril(-1).sub(n);
      return gq.add(q.mm(inner)).mm(rInvT);
    };
    return {
      q: this.linalgNode(qOut, (g) =>
        Tensor.perBatch(g, v.batch, qShape, this.shape,
          (gb, b) => back(gb, zeroR, b)), "QrBackward0"),
      r: this.linalgNode(rOut, (g) =>
        Tensor.perBatch(g, v.batch, rShape, this.shape,
          (gb, b) => back(zeroQ, gb, b)), "QrBackward0"),
    };
  }

  /**
   * Singular value decomposition.
   *
   * **The singular values have a gradient and `U` and `Vh` do not.** Since
   * `dS = diag(Uᵀ·dA·V)`, the singular-value side is the single line `Ā =
   * U·diag(Ḡ)·Vᵀ` with no degeneracy problem. The vector side brings in
   * `1/(sᵢ²−sⱼ²)` and blows up when singular values coincide, and that part
   * was left out.
   *
   * `fullMatrices` (true by default) fills `U` out to `rows×rows` — torch's
   * default. The direction of the fill is not unique once more than one
   * dimension is left over (see `completeBasis`). **What the backward uses
   * is the reduced form, before the fill.**
   */
  async svd(fullMatrices = true): Promise<{ u: Tensor; s: Tensor; vt: Tensor }> {
    const v = await this.asBatch(false);
    const { rows, cols } = v;
    const k = Math.min(rows, cols);
    const us: LA.Mat[] = [];
    const thin: LA.Mat[] = [];
    const ss: LA.Mat[] = [];
    const vts: LA.Mat[] = [];
    for (const a of v.mats) {
      const { u, s, vt } = LA.svd(a, rows, cols);
      thin.push(u);
      us.push(fullMatrices && rows > k ? LA.completeBasis(u, rows, k) : u);
      ss.push(s);
      vts.push(vt);
    }
    const uCols = fullMatrices && rows > k ? rows : k;
    const sOut = this.linalgNode(Tensor.fromBatch(ss, [...v.lead, k]), (g) =>
      Tensor.perBatch(g, v.batch, [k], this.shape, (gb, b) =>
        Tensor.fromMat(thin[b]!, [rows, k])
          .mm(gb.diagflat())
          .mm(Tensor.fromMat(vts[b]!, [k, cols]))), "SvdBackward0");
    return {
      u: Tensor.fromBatch(us, [...v.lead, rows, uCols]),
      s: sOut,
      vt: Tensor.fromBatch(vts, [...v.lead, k, cols]),
    };
  }

  /**
   * Eigenvalues and eigenvectors of a symmetric matrix. The eigenvalues are
   * ascending.
   *
   * **It reads one triangle only** — the lower by default. Jacobi looks at
   * the whole matrix, so it is mirrored first and then handed over. The
   * details are written down at `mirror` in `linalg.ts`.
   */
  async eigh(uplo: "L" | "U" = "L"): Promise<{ values: Tensor; vectors: Tensor }> {
    const v = await this.asBatch();
    const n = v.rows;
    const ws: LA.Mat[] = [];
    const vs: LA.Mat[] = [];
    const fs: LA.Mat[] = [];
    for (const a of v.mats) {
      const { values, vectors } = LA.eigh(LA.mirror(a, n, uplo === "U"), n);
      ws.push(values);
      vs.push(vectors);
      fs.push(LA.eighGap(values, n));
    }
    const half = Tensor.full([], 0.5);
    return {
      // 고윳값: `Ā = V·diag(Ḡ)·Vᵀ`. 한 줄이고 겹침 문제도 없다.
      values: this.linalgNode(Tensor.fromBatch(ws, [...v.lead, n]), (g) =>
        Tensor.perBatch(g, v.batch, [n], this.shape, (gb, b) => {
          const vec = Tensor.fromMat(vs[b]!, [n, n]);
          return vec.mm(gb.diagflat()).mm(vec.transpose());
        }), "EighBackward0"),
      // 고유벡터: `Ā = sym(V·(F∘(Vᵀ·Ḡ))·Vᵀ)`.
      //
      // **대칭화가 빠지면 안 된다.** `A` 가 대칭이라 위·아래 삼각이 같은 자유도를
      // 나눠 갖는데, 날 식은 그것을 한쪽에 몰아준다. 대각은 맞고 비대각만 갈려서
      // 값 대조 없이는 안 보인다 — 실측으로 골랐다.
      vectors: this.linalgNode(Tensor.fromBatch(vs, [...v.lead, n, n]), (g) =>
        Tensor.perBatch(g, v.batch, [n, n], this.shape, (gb, b) => {
          const vec = Tensor.fromMat(vs[b]!, [n, n]);
          const f = Tensor.fromMat(fs[b]!, [n, n]);
          const raw = vec.mm(f.mul(vec.transpose().mm(gb))).mm(vec.transpose());
          return raw.add(raw.transpose()).binary("mul", half);
        }), "EighBackward0"),
    };
  }

  /**
   * The singular values only. The middle of `svd`.
   */
  async svdvals(): Promise<Tensor> {
    return (await this.svd(false)).s;
  }

  /**
   * The eigenvalues of a symmetric matrix only.
   */
  async eigvalsh(uplo: "L" | "U" = "L"): Promise<Tensor> {
    return (await this.eigh(uplo)).values;
  }

  /**
   * Eigenvalues and eigenvectors of **any square matrix.** `linalg.eig`.
   *
   * This is not `eigh` with the symmetry requirement lifted; it is a different
   * function. `eigh` reads one triangle and answers in reals. This one reads
   * the whole matrix and **always answers in complex** — a rotation matrix has
   * no real eigenvalue at all, so the return type cannot depend on the input.
   *
   * Eigenvectors stand in **columns**: `V[:, k]` belongs to `values[k]`, which
   * is what makes `A·V = V·diag(λ)` hold.
   *
   * **The sign is not fixed and cannot be.** torch itself returns opposite
   * signs in float32 and float64 (measured). Anything depending on the sign of
   * an eigenvector is depending on something no implementation promises.
   *
   * There is no backward. The derivative of an eigendecomposition is undefined
   * where eigenvalues repeat, and a quietly wrong gradient is worse than an
   * absent one — the same line `qr` and `svd` draw here.
   */
  async eig(): Promise<{ values: Tensor; vectors: Tensor }> {
    const v = await this.asBatch();
    const n = v.rows;
    const wRe: LA.Mat[] = [];
    const wIm: LA.Mat[] = [];
    const vRe: LA.Mat[] = [];
    const vIm: LA.Mat[] = [];
    for (const a of v.mats) {
      const got = LA.eig(a, n);
      wRe.push(got.re);
      wIm.push(got.im);
      vRe.push(got.vecRe);
      vIm.push(got.vecIm);
    }
    return {
      values: Tensor.complex(
        Tensor.fromBatch(wRe, [...v.lead, n]), Tensor.fromBatch(wIm, [...v.lead, n])),
      vectors: Tensor.complex(
        Tensor.fromBatch(vRe, [...v.lead, n, n]),
        Tensor.fromBatch(vIm, [...v.lead, n, n])),
    };
  }

  /**
   * The eigenvalues of any square matrix. `linalg.eigvals` — the front half of
   * `eig`, and **it skips the eigenvectors** rather than computing and dropping
   * them.
   */
  async eigvals(): Promise<Tensor> {
    const v = await this.asBatch();
    const n = v.rows;
    const re: LA.Mat[] = [];
    const im: LA.Mat[] = [];
    for (const a of v.mats) {
      const got = LA.eigvals(a, n);
      re.push(got.re);
      im.push(got.im);
    }
    return Tensor.complex(
      Tensor.fromBatch(re, [...v.lead, n]), Tensor.fromBatch(im, [...v.lead, n]));
  }

  /**
   * The norm measured treating it as a matrix. **A different number per
   * branch.**
   *
   * The default is Frobenius; `2` is the largest singular value; `nuc` the
   * sum of singular values; `1` the maximum column absolute sum; and `inf`
   * the row equivalent. Only the three that need singular values make a CPU
   * round trip.
   */
  async matrixNorm(ord: number | string = "fro"): Promise<Tensor> {
    if (ord === "nuc" || ord === 2 || ord === -2) {
      const s = await this.svdvals();
      if (ord === "nuc") return s.sumDim(-1, false);
      return ord === 2 ? s.amax(-1, false) : s.amin(-1, false);
    }
    if (ord === "fro") return this.square().sumDim(-1, false).sumDim(-1, false).sqrt();
    // 1 은 열 방향(행을 더한다), inf 는 행 방향(열을 더한다).
    const axis = ord === 1 || ord === -1 ? -2 : -1;
    const sums = this.abs().sumDim(axis, false);
    return (ord as number) > 0 ? sums.amax(-1, false) : sums.amin(-1, false);
  }

  /**
   * The condition number. By default the ratio of singular values.
   */
  async cond(p: number | string | null = null): Promise<Tensor> {
    if (p === null || p === 2 || p === -2) {
      const s = await this.svdvals();
      const hi = s.amax(-1, false);
      const lo = s.amin(-1, false);
      return p === -2 ? lo.div(hi) : hi.div(lo);
    }
    const inv = await this.inverse();
    return (await this.matrixNorm(p)).mul(await inv.matrixNorm(p));
  }

  /**
   * Solves knowing the matrix is triangular.
   */
  async solveTriangular(
    b: Tensor, upper: boolean, left = true, unitriangular = false,
  ): Promise<Tensor> {
    const v = await this.asBatch();
    if (v.batch !== 1) throw new RuntimeError("solve_triangular: batching is not here yet");
    const n = v.rows;
    const width = b.shape.length === 1 ? 1 : (b.shape[b.shape.length - 1] ?? 1);
    const rhs = LA.fromF32(await b.toArray());
    if (!left) {
      // `X A = B` 는 양쪽을 전치하면 같은 길로 간다.
      const at = LA.transpose(v.mats[0]!, n, n);
      const bt = LA.transpose(rhs, width, n);
      const x = LA.solveTriangular(at, bt, n, width, !upper, unitriangular);
      return Tensor.fromMat(LA.transpose(x, n, width), b.shape);
    }
    return Tensor.fromMat(
      LA.solveTriangular(v.mats[0]!, rhs, n, width, upper, unitriangular), b.shape);
  }

  /**
   * The matrix exponential `e^A`. **It has a gradient.**
   *
   * The backward uses the table `matrixExpAdjointMap` in `linalg.ts` froze
   * during the forward — why a table is written down there (briefly: `Ḡ` is
   * on the GPU and `expm` is CPU). **It is built only when the gradient is
   * wanted.** One table is `n²` calls to `expm`, which is not free.
   */
  async matrixExp(): Promise<Tensor> {
    const v = await this.asBatch();
    const n = v.rows;
    const out = Tensor.fromBatch(
      v.mats.map((a) => LA.matrixExp(a, n)), this.shape);
    if (!this.requiresGrad) return out;
    const maps = v.mats.map((a) => LA.matrixExpAdjointMap(a, n));
    return this.linalgNode(out, (g) =>
      Tensor.perBatch(g, v.batch, [n, n], this.shape, (gb, b) =>
        Tensor.fromMat(maps[b]!, [n * n, n * n])
          .mm(gb.reshape([n * n, 1])).reshape([n, n])), "MatrixExpBackward0");
  }

  /**
   * The inner product, treating the last axis as a vector.
   */
  vecdot(other: Tensor, dim = -1): Tensor {
    return this.mul(other).sumDim(dim, false);
  }

  /**
   * The cross product of three-dimensional vectors. It splits into three
   * axes and recombines.
   */
  cross(other: Tensor, dim = -1): Tensor {
    const rank = this.shape.length;
    const axis = dim < 0 ? dim + rank : dim;
    const at = (t: Tensor, i: number) => t.narrow(axis, i, 1);
    return Tensor.cat([
      at(this, 1).mul(at(other, 2)).sub(at(this, 2).mul(at(other, 1))),
      at(this, 2).mul(at(other, 0)).sub(at(this, 0).mul(at(other, 2))),
      at(this, 0).mul(at(other, 1)).sub(at(this, 1).mul(at(other, 0))),
    ], axis);
  }

  /**
   * The norm measured treating it as a vector. **Given a matrix it flattens
   * the whole thing** — where it parts from `matrixNorm`.
   *
   * `ord=0` is the count of non-zeros, and `±Infinity` the maximum and
   * minimum absolute value. Branches that must not go through the power
   * formula, so they are written separately.
   */
  vectorNorm(ord = 2, dim?: number): Tensor {
    const flat = dim === undefined && this.shape.length > 1
      ? this.reshape([this.size])
      : this;
    const x = flat.abs();
    if (ord === Infinity) return x.amax(dim, false);
    if (ord === -Infinity) return x.amin(dim, false);
    if (ord === 0) return x.binary("ne", Tensor.full([], 0)).sumDim(dim ?? 0, false);
    if (ord === 1) return dim === undefined ? x.sum() : x.sumDim(dim, false);
    const powed = ord === 2 ? x.square() : x.powScalar(ord);
    const total = dim === undefined ? powed.sum() : powed.sumDim(dim, false);
    return ord === 2 ? total.sqrt() : total.powScalar(1 / ord);
  }

  /** 반데르몽드 행렬. 열이 **커지는 차수**다. */
  vander(N?: number): Tensor {
    const n = N ?? this.size;
    const cols: Tensor[] = [];
    for (let k = 0; k < n; k++) cols.push(this.powScalar(k));
    return Tensor.stack(cols, 1);
  }

  /**
   * Folds the tensor into a matrix, solves, and spreads it back.
   */
  async tensorSolve(b: Tensor): Promise<Tensor> {
    const n = b.size;
    const folded = this.reshape([n, this.size / n]);
    const x = await folded.solve(b.reshape([n]));
    return x.reshape(this.shape.slice(b.shape.length));
  }

  /**
   * Folds the tensor into a matrix, inverts, and returns the axis order.
   */
  async tensorInv(ind = 2): Promise<Tensor> {
    const lead = this.shape.slice(0, ind);
    const n = lead.reduce((a, b) => a * b, 1);
    const inv = await this.reshape([n, this.size / n]).inverse();
    return inv.reshape([...this.shape.slice(ind), ...lead]);
  }

  async matrixRank(): Promise<Tensor> {
    const v = await this.asBatch(false);
    return Tensor.from(v.mats.map((a) => LA.matrixRank(a, v.rows, v.cols)), v.lead);
  }

  /**
   * The least-squares solution. Square and invertible, it gives the same
   * answer as `solve`.
   */
  async lstsq(b: Tensor): Promise<Tensor> {
    const v = await this.asBatch(false);
    if (v.batch !== 1) throw new RuntimeError("lstsq: batching is not here yet");
    const { rows, cols } = v;
    const width = b.shape.length === 1 ? 1 : (b.shape[b.shape.length - 1] ?? 1);
    const rhs = LA.fromF32(await b.toArray());
    const sol = LA.matmul(
      LA.pinverse(v.mats[0]!, rows, cols), rhs, cols, rows, width);
    return Tensor.fromMat(sol, b.shape.length === 1 ? [cols] : [cols, width]);
  }

  /**
   * `L` and `U` packed into one plate, plus a pivot table **counting from
   * 1.**
   */
  async luFactor(): Promise<{ LU: Tensor; pivots: Tensor }> {
    const v = await this.asBatch(false);
    const k = Math.min(v.rows, v.cols);
    const packed = v.mats.map((a) => LA.luFactor(a, v.rows, v.cols));
    const pivots = new Float32Array(v.batch * k);
    packed.forEach((f, b) => pivots.set(Float32Array.from(f.piv), b * k));
    return {
      LU: Tensor.fromBatch(packed.map((f) => f.lu), this.shape),
      pivots: Tensor.from(pivots, [...v.lead, k]),
    };
  }

  /**
   * `luFactor` **plus one `info`.** Instead of throwing, it reports by
   * number — 0 means it went well, and `k` means the `k`-th pivot is zero
   * and the matrix is singular (counting from 1).
   */
  async luFactorEx(): Promise<{ LU: Tensor; pivots: Tensor; info: Tensor }> {
    const v = await this.asBatch(false);
    const k = Math.min(v.rows, v.cols);
    const got = await this.luFactor();
    const packed = v.mats.map((a) => LA.luFactor(a, v.rows, v.cols));
    const info = new Float32Array(v.batch);
    packed.forEach((f, b) => {
      for (let i = 0; i < k; i++) {
        if (f.lu[i * v.cols + i] === 0) { info[b] = i + 1; break; }
      }
    });
    return {
      LU: got.LU,
      pivots: got.pivots,
      info: Tensor.from(info, v.lead, { dtype: "int64" }),
    };
  }

  /**
   * A symmetric matrix as `L D Lᵀ`. **It does not pivot.**
   *
   * torch uses LAPACK's Bunch–Kaufman, which swaps positions when it has
   * to. Here only the cases with nothing to swap (positive definite and the
   * like) are handled, and a near-zero diagonal is refused loudly —
   * carrying on quietly gives a different answer with and without swapping,
   * and both are plausible.
   *
   * The answer is **packed into one plate** in torch's shape — the diagonal
   * is `D` and below it is `L`.
   */
  async ldlFactor(): Promise<{ LD: Tensor; pivots: Tensor }> {
    const v = await this.asBatch();
    const n = v.rows;
    const outs = v.mats.map((mat) => {
      const ld = new Float64Array(n * n);
      for (let j = 0; j < n; j++) {
        let d = mat[j * n + j] ?? 0;
        for (let k = 0; k < j; k++) {
          d -= (ld[j * n + k] ?? 0) ** 2 * (ld[k * n + k] ?? 0);
        }
        if (Math.abs(d) < 1e-12) {
          throw new RuntimeError("ldl_factor — this symmetric matrix needs pivoting (it is indefinite)");
        }
        ld[j * n + j] = d;
        for (let i = j + 1; i < n; i++) {
          let s = 0;
          for (let k = 0; k < j; k++) {
            s += (ld[i * n + k] ?? 0) * (ld[k * n + k] ?? 0) * (ld[j * n + k] ?? 0);
          }
          ld[i * n + j] = ((mat[i * n + j] ?? 0) - s) / d;
        }
      }
      return ld;
    });
    // 교환표는 1 부터 센 항등이다 — 자리를 안 바꿨으므로.
    const piv = new Float32Array(v.batch * n);
    for (let b = 0; b < v.batch; b++) {
      for (let i = 0; i < n; i++) piv[b * n + i] = i + 1;
    }
    return {
      LD: Tensor.fromBatch(outs, this.shape),
      pivots: Tensor.from(piv, [...v.lead, n], { dtype: "int64" }),
    };
  }

  /**
   * `ldlFactor` **plus one `info`.** The same place as `luFactorEx`.
   *
   * **Here it is always 0** — meeting a bad position makes `ldlFactor`
   * refuse on the spot, so nothing is left to report by number. The name is
   * kept anyway: torch offers it, and without it the caller has to invent
   * one of the three slots themselves — which the binding was actually
   * doing (standing up `_Fields` by hand and slotting a 0 into `info`).
   * Then the golden cases go through the binding and are green, and what is
   * missing is only for whoever writes TypeScript.
   */
  async ldlFactorEx(): Promise<{ LD: Tensor; pivots: Tensor; info: Tensor }> {
    const v = await this.asBatch();
    const got = await this.ldlFactor();
    return { ...got, info: Tensor.zeros(v.lead).to("int64") };
  }

  /**
   * Solves using the factorisation `ldlFactor` produced. Three passes: `L y
   * = b`, `D z = y`, `Lᵀ x = z`.
   */
  async ldlSolve(b: Tensor): Promise<Tensor> {
    const v = await this.asBatch();
    const n = v.rows;
    if (v.batch !== 1) throw new RuntimeError("ldl_solve: batching is not here yet");
    const ld = v.mats[0];
    if (!ld) throw new RuntimeError("ldl_solve: the factorization is empty");
    const width = b.shape.length === 1 ? 1 : (b.shape[b.shape.length - 1] ?? 1);
    const rhs = LA.fromF32(await b.toArray());
    const out = new Float64Array(n * width);
    for (let c = 0; c < width; c++) {
      const y = new Float64Array(n);
      for (let i = 0; i < n; i++) {
        let s = rhs[i * width + c] ?? 0;
        for (let k = 0; k < i; k++) s -= (ld[i * n + k] ?? 0) * (y[k] ?? 0);
        y[i] = s;
      }
      for (let i = 0; i < n; i++) y[i] = (y[i] ?? 0) / (ld[i * n + i] ?? 1);
      for (let i = n - 1; i >= 0; i--) {
        let s = y[i] ?? 0;
        for (let k = i + 1; k < n; k++) {
          s -= (ld[k * n + i] ?? 0) * (out[k * width + c] ?? 0);
        }
        out[i * width + c] = s;
      }
    }
    return Tensor.fromMat(out, b.shape);
  }

  /**
   * Produces QR **in reflector form.** `householderProduct` stands `Q` up
   * from it.
   *
   * **With everything below the diagonal already zero it does not reflect**
   * — LAPACK's `dlarfg` puts `tau = 0` there and leaves the values alone.
   * The last column of a square matrix is always that case.
   */
  async geqrf(): Promise<{ a: Tensor; tau: Tensor }> {
    const v = await this.asBatch(false);
    const { rows: m, cols: n } = v;
    const k = Math.min(m, n);
    const taus = new Float32Array(v.batch * k);
    const outs = v.mats.map((src, b) => {
      const mat = Float64Array.from(src);
      for (let j = 0; j < k; j++) {
        let tail = 0;
        for (let i = j + 1; i < m; i++) tail += (mat[i * n + j] ?? 0) ** 2;
        if (tail === 0) continue;
        const alpha = mat[j * n + j] ?? 0;
        const norm = Math.sqrt(alpha * alpha + tail);
        const beta = alpha !== 0 ? -Math.sign(alpha) * norm : -norm;
        const tau = (beta - alpha) / beta;
        const w = new Float64Array(m - j);
        w[0] = 1;
        for (let i = 1; i < m - j; i++) {
          w[i] = (mat[(j + i) * n + j] ?? 0) / (alpha - beta);
        }
        for (let c = j; c < n; c++) {
          let dot = 0;
          for (let i = 0; i < m - j; i++) dot += (w[i] ?? 0) * (mat[(j + i) * n + c] ?? 0);
          for (let i = 0; i < m - j; i++) {
            mat[(j + i) * n + c] = (mat[(j + i) * n + c] ?? 0) - tau * (w[i] ?? 0) * dot;
          }
        }
        mat[j * n + j] = beta;
        for (let i = 1; i < m - j; i++) mat[(j + i) * n + j] = w[i] ?? 0;
        taus[b * k + j] = tau;
      }
      return mat;
    });
    return {
      a: Tensor.fromBatch(outs, this.shape),
      tau: Tensor.from(taus, [...v.lead, k]),
    };
  }

  /**
   * Multiplies the reflectors to stand `Q` up. The counterpart of `geqrf`.
   *
   * `v_i` has 1 on the diagonal and `A[i+1:, i]` below it — the 1 on the
   * diagonal is **a promise not to store it**, so reading that slot
   * mistakes the `R` the factorisation put there for a reflector.
   */
  async householderProduct(tau: Tensor): Promise<Tensor> {
    const v = await this.asBatch(false);
    const { rows: m, cols: n } = v;
    const taus = await tau.toArray();
    const k = tau.shape[tau.shape.length - 1] ?? 0;
    const outs = v.mats.map((mat, b) => {
      const q = new Float64Array(m * m);
      for (let i = 0; i < m; i++) q[i * m + i] = 1;
      for (let j = k - 1; j >= 0; j--) {
        const t = taus[b * k + j] ?? 0;
        if (t === 0) continue;
        const w = new Float64Array(m);
        w[j] = 1;
        for (let i = j + 1; i < m; i++) w[i] = mat[i * n + j] ?? 0;
        for (let c = 0; c < m; c++) {
          let dot = 0;
          for (let i = 0; i < m; i++) dot += (w[i] ?? 0) * (q[i * m + c] ?? 0);
          for (let i = 0; i < m; i++) {
            q[i * m + c] = (q[i * m + c] ?? 0) - t * (w[i] ?? 0) * dot;
          }
        }
      }
      // torch 는 입력의 열 수만큼만 낸다.
      const cut = new Float64Array(m * n);
      for (let i = 0; i < m; i++) {
        for (let c = 0; c < n; c++) cut[i * n + c] = q[i * m + c] ?? 0;
      }
      return cut;
    });
    return Tensor.fromBatch(outs, this.shape);
  }

  /**
   * Spread into `P`, `L` and `U`. Easier to read than the packed plate.
   */
  async lu(): Promise<{ P: Tensor; L: Tensor; U: Tensor }> {
    const v = await this.asBatch(false);
    const { rows, cols } = v;
    const k = Math.min(rows, cols);
    const parts = v.mats.map((a) => LA.luExpand(LA.luFactor(a, rows, cols)));
    return {
      P: Tensor.fromBatch(parts.map((p) => p.p), [...v.lead, rows, rows]),
      L: Tensor.fromBatch(parts.map((p) => p.l), [...v.lead, rows, k]),
      U: Tensor.fromBatch(parts.map((p) => p.u), [...v.lead, k, cols]),
    };
  }

  /** `luFactor` 가 낸 것으로 `A x = b` 를 푼다. */
  /**
   * Solves `A x = b` with the factorisation. **`this` is the LU** — the
   * argument order of `linalg.lu_solve(LU, pivots, B)`, and a different
   * receiver from torch's `Tensor.lu_solve`.
   *
   * So **the torch claim was taken out of the name.** This used to be
   * `luSolve`, and someone carrying torch across writing `b.lu_solve(LU,
   * piv)` puts `LU` into the `pivots` slot — the name and the argument
   * count both match, so **nothing catches at that point** and only the
   * value is wrong.
   */
  async luSolveFactored(pivots: Tensor, b: Tensor): Promise<Tensor> {
    const v = await this.asBatch();
    if (v.batch !== 1) throw new RuntimeError("lu_solve: batching is not here yet");
    const n = v.rows;
    const piv = Int32Array.from(await pivots.toArray());
    const width = b.shape.length === 1 ? 1 : (b.shape[b.shape.length - 1] ?? 1);
    const rhs = LA.fromF32(await b.toArray());
    const x = LA.luSolveFactored(
      { lu: v.mats[0]!, piv, rows: n, cols: n }, rhs, width);
    return Tensor.fromMat(x, b.shape);
  }

  // ── 최상위 선형대수 ───────────────────────────────────────────────────
  //
  // **`linalg` 쪽과 인자 순서가 다르다.** torch 가 옛 이름들을 최상위에 남겨 뒀는데
  // 그것들은 대개 **오른쪽 변을 먼저** 받는다. 같은 계산이므로 계산은 한 벌만 두고
  // 자리만 옮긴다 — 그 옮김이 맞는지는 값으로만 확인된다.

  /**
   * 인수를 **아래 삼각으로** 세운다. `A = L Lᵀ` 가 되도록.
   *
   * 조립으로 둔다 — `tril`·`transpose` 를 지나면 **인수 쪽으로도 기울기가 흐른다.**
   * 값만 잘라 쓰면 역방향이 `b` 로만 가는데 torch 는 인수로도 흘린다(실측).
   */
  private static asLower(factor: Tensor, upper: boolean): Tensor {
    return upper ? factor.triu().transpose() : factor.tril();
  }

  /**
   * Solves `A x = b` **from the Cholesky factor.** `A = L Lᵀ` (or `Uᵀ U`).
   *
   * It rebuilds `A` and sends it to `solve`. Two triangular substitutions
   * would be cheaper, but written that way the backward has to be
   * hand-written and **the gradient towards the factor goes missing
   * quietly.**
   */
  async choleskySolve(factor: Tensor, upper = false): Promise<Tensor> {
    const low = Tensor.asLower(factor, upper);
    return low.mm(low.transpose()).solve(this);
  }

  /**
   * From a Cholesky factor it produces **the original matrix's inverse**,
   * not the factor's.
   */
  async choleskyInverse(upper = false): Promise<Tensor> {
    const low = Tensor.asLower(this, upper);
    return low.mm(low.transpose()).inverse();
  }

  /**
   * **It gives two** — the solution, and a **copy** of the coefficient
   * matrix handed in (measured).
   *
   * The same computation as `solveTriangular` with the argument order
   * flipped and **`upper` defaulting to true.** Miss those two and you
   * solve a different triangle with a plausible-looking value.
   */
  async triangularSolve(
    a: Tensor,
    upper = true,
    transpose = false,
    unitriangular = false,
  ): Promise<{ solution: Tensor; cloned_coefficient: Tensor }> {
    let tri = upper ? a.triu() : a.tril();
    if (unitriangular) {
      // **대각을 안 보고 1 로 친다.** 그대로 두면 조용히 다른 답이 나온다.
      const n = a.shape[a.shape.length - 1] ?? 0;
      const off = upper ? tri.triu(1) : tri.tril(-1);
      tri = off.add(Tensor.eye(n));
    }
    if (transpose) tri = tri.transpose();
    return { solution: await tri.solve(this), cloned_coefficient: a.add(Tensor.full([], 0)) };
  }

  /**
   * `(LU, pivots)`. **It produces something different from `lu()`** — that
   * one spreads into `P·L·U` and this one gives **one packed plate and a
   * swap list** (measured).
   *
   * With `getInfos` an info code is attached third. Ours is always 0 —
   * meeting a singular matrix it throws on the spot rather than reporting
   * quietly by code.
   */
  async luTop(pivot = true, getInfos = false): Promise<{
    LU: Tensor; pivots: Tensor; info?: Tensor;
  }> {
    if (!pivot) throw new RuntimeError("lu(pivot=false) is not here");
    const got = await this.luFactor();
    return getInfos ? { ...got, info: Tensor.zeros([]) } : got;
  }

  /**
   * `torch.Tensor.lu_solve` — **the receiver is the right-hand side `b`.**
   *
   * torch has `b.lu_solve(LU, piv)`, so it is the same here. The side that
   * takes the factorisation as receiver is `luSolveFactored`, which is
   * `linalg.lu_solve`'s argument order.
   */
  async luSolve(luData: Tensor, pivots: Tensor): Promise<Tensor> {
    return luData.luSolveFactored(pivots, this);
  }

  /**
   * The old name. Kept because the binding calls it — one line of
   * delegation.
   *
   * It was made to point at the opposite side back when `luSolve` took the
   * factorisation as receiver, and now that `luSolve` itself sits where
   * torch does, **there is no opposite side for `Top` to point at.** Once
   * the binding moves to `luSolve` it can be deleted.
   */
  async luSolveTop(luData: Tensor, pivots: Tensor): Promise<Tensor> {
    return this.luSolve(luData, pivots);
  }

  /**
   * Spreads one packed plate into `P·L·U`.
   *
   * **Turned off, what comes back is an empty tensor rather than `null`**
   * (measured: its shape is `[0]`). Returning `null` makes the receiving
   * side branch on it, and that is not torch code.
   */
  async luUnpack(
    pivots: Tensor,
    unpackData = true,
    unpackPivots = true,
  ): Promise<{ P: Tensor; L: Tensor; U: Tensor }> {
    const empty = Tensor.from(new Float32Array(0), [0]);
    if (!unpackData && !unpackPivots) return { P: empty, L: empty, U: empty };
    const v = await this.asBatch(false);
    const { rows, cols } = v;
    const piv = Int32Array.from(await pivots.toArray());
    const k = Math.min(rows, cols);
    const parts = v.mats.map((lu) => LA.luExpand({ lu, piv, rows, cols }));
    return {
      P: unpackPivots
        ? Tensor.fromBatch(parts.map((p) => p.p), [...v.lead, rows, rows])
        : empty,
      L: unpackData
        ? Tensor.fromBatch(parts.map((p) => p.l), [...v.lead, rows, k])
        : empty,
      U: unpackData
        ? Tensor.fromBatch(parts.map((p) => p.u), [...v.lead, k, cols])
        : empty,
    };
  }

  /**
   * Another name for `householderProduct`. torch offers both.
   */
  async orgqr(tau: Tensor): Promise<Tensor> {
    return this.householderProduct(tau);
  }

  /**
   * Multiplies into `C` **without standing Q up** — here it stands it up
   * and multiplies. The value is the same and nothing is saved at this
   * size.
   *
   * **It is a different Q from `orgqr`'s.** That one gives a Q **cut** to
   * `m×k`, while this uses the uncut `m×m` — the reflectors are maps on
   * `Rᵐ`, and cutting multiplies by only part of that map. The answer
   * diverges entirely on a tall matrix (measured). Measured on squares
   * alone, the two are the same and nothing shows.
   */
  async ormqr(
    tau: Tensor,
    other: Tensor,
    left = true,
    transpose = false,
  ): Promise<Tensor> {
    const v = await this.asBatch(false);
    const { rows: m, cols: n } = v;
    const taus = await tau.toArray();
    const k = tau.shape[tau.shape.length - 1] ?? 0;
    const mat = v.mats[0]!;
    const q = new Float64Array(m * m);
    for (let i = 0; i < m; i++) q[i * m + i] = 1;
    for (let j = k - 1; j >= 0; j--) {
      const t = taus[j] ?? 0;
      if (t === 0) continue;
      const w = new Float64Array(m);
      w[j] = 1;
      for (let i = j + 1; i < m; i++) w[i] = mat[i * n + j] ?? 0;
      for (let c = 0; c < m; c++) {
        let dot = 0;
        for (let i = 0; i < m; i++) dot += (w[i] ?? 0) * (q[i * m + c] ?? 0);
        for (let i = 0; i < m; i++) {
          q[i * m + c] = (q[i * m + c] ?? 0) - t * (w[i] ?? 0) * dot;
        }
      }
    }
    const qm = transpose ? LA.transpose(q, m, m) : q;
    const c = LA.fromF32(await other.toArray());
    const [cr, cc] = other.shape.length === 1
      ? [other.shape[0] ?? 0, 1]
      : [other.shape[0] ?? 0, other.shape[1] ?? 0];
    const out = left
      ? LA.matmul(qm, c, m, m, cc)
      : LA.matmul(c, qm, cr, m, m);
    return Tensor.fromMat(out, left ? [m, cc] : [cr, m]);
  }

  /**
   * The **`k` extreme eigenpairs** of a symmetric matrix.
   *
   * **torch is iterative and we are exact.** That side iterates to get a
   * few cheaply out of a large sparse matrix; we have no sparse and the
   * sizes are small. Measuring showed torch's answer converges to within
   * 7e-6 of the exact one and wobbles by about that much with the seed
   * (measured) — far below tolerance. The value is the same and only the
   * cost differs.
   *
   * **`largest` decides the order too** — true gives largest first, false
   * smallest first (measured).
   */
  async lobpcg(k = 1, largest = true): Promise<{
    eigenvalues: Tensor; eigenvectors: Tensor;
  }> {
    const { values, vectors } = await this.eigh();
    const n = values.shape[values.shape.length - 1] ?? 0;
    const picks: number[] = [];
    for (let i = 0; i < k; i++) picks.push(largest ? n - 1 - i : i);
    const idx = Tensor.from(picks, [picks.length]);
    return {
      eigenvalues: values.indexSelect(0, idx),
      eigenvectors: vectors.indexSelect(1, idx),
    };
  }

  /**
   * **A low-rank SVD** obtained by random projection. `(U, S, V)`, and **V
   * is not transposed.**
   *
   * **The answer is stable only on input that is exactly low-rank.** torch
   * projects with a random matrix, and once the rank exceeds `q` the
   * singular values move by around 0.5 depending on the seed (measured). At
   * rank `q` or below, changing the seed stays within 7e-7 — that is the
   * only place worth asking.
   *
   * We do not project. The full SVD is computed and the first `q` taken —
   * the same answer where the input is exactly low-rank, and **more
   * accurate than torch** where it is not.
   */
  async svdLowrank(q = 6, niter = 2, M: Tensor | null = null): Promise<{
    U: Tensor; S: Tensor; V: Tensor;
  }> {
    // **받되 안 쓴다 — 다듬을 것이 없어서다.** `niter` 는 torch 가 무작위 부분공간을
    // 몇 번 다듬는가이고, 그 반복이 있는 이유는 사영이 근사이기 때문이다. 우리는
    // 전체 SVD 를 구하므로 첫 답이 이미 수렴한 답이다 — 값은 torch 의 `niter` 를
    // 크게 키운 극한 쪽에 있다.
    //
    // 그래서 **작은 `niter` 에서는 torch 와 값이 갈린다.** 우리 쪽이 더 정확한
    // 갈림이지만 갈림은 갈림이라 README 에 적어 뒀다. 이유를 안 적으면 다음 사람이
    // "덜 구현됐다" 로 읽고 근사를 도로 넣는다.
    void niter;
    const src = M === null ? this : this.sub(M);
    const { u, s, vt } = await src.svd(false);
    const cut = Math.min(q, s.shape[0] ?? 0);
    const take = Tensor.from(
      Array.from({ length: cut }, (_, i) => i), [cut]);
    return {
      U: u.indexSelect(1, take),
      S: s.indexSelect(0, take),
      V: vt.indexSelect(0, take).transpose(),
    };
  }

  /**
   * Low-rank PCA. **With `center=false` it is the same thing as
   * `svdLowrank`** (measured).
   *
   * Centring is the whole difference between this function and that one.
   * Measured at true alone, that branch never surfaces.
   */
  async pcaLowrank(q?: number, center = true, niter = 2): Promise<{
    U: Tensor; S: Tensor; V: Tensor;
  }> {
    const rows = this.shape[0] ?? 0;
    const cols = this.shape[1] ?? 0;
    const want = q ?? Math.min(6, rows, cols);
    const src = center
      ? this.sub(this.sumDim(0, true).div(Tensor.full([], rows)))
      : this;
    return src.svdLowrank(want, niter);
  }

  /** 값은 CPU 에서 이미 나왔고, 여기서는 그래프만 잇는다. */
  private linalgNode(
    value: Tensor,
    backwardFn: (g: Tensor) => Tensor,
    gradName: string,
  ): Tensor {
    return Tensor.make(value.buffer, value.shape, [this],
      (g) => [backwardFn(g)], gradName);
  }

  // ── 제자리 연산 ───────────────────────────────────────────────────────

  /**
   * 제자리 연산의 공통 관문.
   *
   * **기울기가 켜진 잎은 거절한다.** torch 가 그렇고, 이유가 있다 — 잎의 값이 바뀌면
   * 이미 그 값을 쓴 역방향이 틀린 수를 쓰게 되는데 아무도 알 수가 없다. 옵티마이저가
   * 실제로 가중치를 제자리에서 고치는데, 그것은 `no_grad` 안이라 여기를 안 지난다.
   *
   * 결과를 새 버퍼에 만든 뒤 옮긴다. 읽으면서 같이 쓰면 스레드 순서가 없어 섞인다.
   *
   * **뷰로 번진다.** `reshape` 계열이 버퍼를 같이 쓰므로 `a.view(2,2).add_(10)` 이
   * `a` 까지 바꾼다 — torch 와 같다. 자매는 TF.js 텐서가 불변이라 그것을 거절한다.
   */
  private mutate(compute: () => Tensor): Tensor {
    if (gradMode.enabled && this.requiresGrad && this.parents.length === 0) {
      throw new RuntimeError(
        "a leaf Variable that requires grad is being used in an in-place operation.",
      );
    }
    const result = compute();
    // **결과가 같은 버퍼일 수 있다.** `squeeze`·`unsqueeze` 는 값을 안 옮기고 틀만
    // 바꾸므로 `reshape` 처럼 버퍼를 그대로 물려준다. 그 자리에 복사를 걸면 WebGPU 가
    // "원본과 사본이 같은 버퍼" 라며 명령 버퍼째 무효로 만들고, 그러면 **그 뒤에 줄
    // 서 있던 케이스가 대신 틀린다** — 실제로 다음 케이스가 엉뚱한 값으로 실패했다.
    // **칸 수가 줄 수 있다.** `as_strided_` 는 틀만이 아니라 크기까지 바꾼다. 원래
    // 크기로 복사하면 WebGPU 가 "원본 버퍼보다 크게 읽는다" 며 **명령 버퍼째** 무효로
    // 만들고, 그러면 이 케이스는 통과하고 **그 뒤에 줄 서 있던 케이스가 대신 틀린다** —
    // 실제로 다음 케이스가 엉뚱한 값으로 실패했다.
    if (result.size > this.size) {
      throw new RuntimeError(
        "an in-place operation cannot grow the element count — the buffer is not that large.",
      );
    }
    if (result.buffer !== this.buffer) {
      dev().copyInto(this.buffer, result.buffer, result.size);
    }
    this.size = result.size;
    // **모양이 바뀌는 것들이 있다.** `transpose_` 는 칸 수는 그대로 두고 틀을 바꾸고,
    // `squeeze_`·`unsqueeze_` 는 축만 넣고 뺀다. 값만 옮기고 모양을 그대로 두면
    // **정사각으로 물었을 때만 통과한다** — 코어에서 실제로 2×2 케이스가 그것을
    // 놓쳤다. 칸 수는 안 변하므로 `size` 는 그대로다.
    if (result.shape.length !== this.shape.length
      || result.shape.some((n, i) => n !== this.shape[i])) {
      this.shape = [...result.shape];
    }
    return this;
  }

  add_(other: number, alpha = 1): Tensor {
    return this.mutate(() => this.binary("add", Tensor.full([], other * alpha)));
  }

  sub_(other: number, alpha = 1): Tensor {
    return this.mutate(() => this.binary("sub", Tensor.full([], other * alpha)));
  }

  mul_(other: number): Tensor {
    return this.mutate(() => this.binary("mul", Tensor.full([], other)));
  }

  div_(other: number): Tensor {
    return this.mutate(() => this.binary("div", Tensor.full([], other)));
  }

  pow_(k: number): Tensor {
    return this.mutate(() => this.powScalar(k));
  }

  zero_(): Tensor {
    return this.mutate(() => Tensor.zeros(this.shape));
  }

  /**
   * A **copy** filled with one value. The counterpart of `fill_`, and torch
   * offers both.
   *
   * This name was missing and invisible because the check read `fill_` as
   * `fill` — the third place where stripping a trailing underscore erased
   * the meaning.
   */
  fill(value: number): Tensor {
    return Tensor.full(this.shape, value);
  }

  /**
   * The `n`-th derivative of the digamma function, in place. The order is
   * **the first argument.**
   */
  polygamma_(n: number): Tensor {
    return this.mutate(() => polygamma(n, this));
  }

  /**
   * Raises the gradient flag and **returns itself**, which torch does so it
   * can be chained (`x.requires_grad_().sum()`).
   */
  requiresGrad_(flag = true): Tensor {
    this.requiresGrad = flag;
    return this;
  }

  fill_(value: number): Tensor {
    return this.mutate(() => Tensor.full(this.shape, value));
  }

  clamp_(low: number, high: number): Tensor {
    return this.mutate(() => this.clamp(low, high));
  }

  /**
   * The same as `clamp_` — torch carries both names.
   */
  clip_(low: number, high: number): Tensor {
    return this.clamp_(low, high);
  }

  /**
   * Moves another tensor's values into this buffer. **It does not swap the
   * tensor.**
   *
   * The optimizer holds the parameter's handle, so making a new tensor and
   * swapping it in when putting weights in leaves the optimizer looking at
   * something else — a state where training runs and the parameters do not
   * move. So the slot stays and only the values move.
   */
  copyFrom(src: Tensor): Tensor {
    if (src.size !== this.size) {
      throw new Error(`size mismatch: [${this.shape}] <- [${src.shape}]`);
    }
    return this.mutate(() => src);
  }

  /**
   * `this ← (1-t)·this + t·other`. Running statistics use it.
   */
  lerpFrom(other: Tensor, t: number): Tensor {
    return this.mutate(() =>
      this.binary("mul", Tensor.full([], 1 - t))
        .add(other.binary("mul", Tensor.full([], t))));
  }

  /**
   * The table's unary operations, in place. Names like `abs_` come here.
   */
  inplaceUnary(name: string): Tensor {
    return this.mutate(() => this.unary(name));
  }

  // 인자를 받는 제자리 연산. 표로 못 도는 것들이라 하나씩 적되, **계산은 밑줄 없는
  // 쪽이 한다** — 같은 식을 두 벌로 두면 언젠가 갈리고 값이 그럴듯해서 안 보인다.

  transpose_(): Tensor {
    return this.mutate(() => this.transpose());
  }

  squeeze_(dim: number): Tensor {
    return this.mutate(() => this.squeeze(dim));
  }

  unsqueeze_(dim: number): Tensor {
    return this.mutate(() => this.unsqueeze(dim));
  }

  tril_(diagonal = 0): Tensor {
    return this.mutate(() => this.tril(diagonal));
  }

  triu_(diagonal = 0): Tensor {
    return this.mutate(() => this.triu(diagonal));
  }

  cumsum_(dim = 0): Tensor {
    return this.mutate(() => this.cumsum(dim));
  }

  cumprod_(dim = 0): Tensor {
    return this.mutate(() => this.cumprod(dim));
  }

  // ── 커널 표에는 있는데 이름이 없던 것들 ──────────────────────────────
  //
  // borch.ts 는 원소별 연산을 **이름마다 메서드로 주는 대신** `binary(이름, 저쪽)`·
  // `unary(이름)` 표로 준다. 커널이 하나뿐이니 그 편이 짧고, 새 연산을 넣을 때
  // 메서드를 안 늘려도 된다.
  //
  // **그런데 쓰는 사람이 치는 줄은 `x.gcd(y)` 다.** 표를 아는 사람만 `x.binary("gcd", y)`
  // 를 칠 수 있고, torch 에서 옮겨 온 코드는 그 이름을 모른다. 아래 열하나는 그래서
  // 있다 — 계산은 이미 있었고 없던 것은 **부르는 법**이다.
  //
  // 이 갈래가 `inplace::짝에서::` 40 건 뒤에 숨어 있었다. 갭 표의 까닭이 "별칭" 이라
  // 적혀 있었는데 별칭인 것은 그중 열뿐이었다.

  /**
   * 수도 텐서도 받는다.
   *
   * torch 는 `x.bitwise_and_(3)` 과 `x.gcd_(y)` 를 둘 다 쓴다 — 텐서만 받게 두면
   * 앞쪽이 그냥 안 돌고, 그 갈림은 서명에만 있어서 값 검사에 안 걸린다.
   */
  private static asTensor(v: Tensor | number): Tensor {
    return v instanceof Tensor ? v : Tensor.full([], v);
  }

  /**
   * The two-argument arctangent. **The argument order is `(y, x)`** —
   * reversed, it quietly gives a different angle.
   *
   * `arctan2` is torch's second spelling for the same thing.
   */
  atan2(other: Tensor | number): Tensor {
    return this.binary("atan2", Tensor.asTensor(other));
  }

  arctan2(other: Tensor | number): Tensor {
    return this.atan2(other);
  }

  arctan2_(other: Tensor | number): Tensor {
    return this.mutate(() => this.atan2(other));
  }

  // ── torch 의 둘째 철자들 ─────────────────────────────────────────────
  //
  // 계산은 이미 있고 **부르는 철자만** 없던 자리다. 교재와 옮겨 온 코드가 어느 쪽을
  // 치는지는 저자 취향이라, 한쪽만 있으면 그 코드가 `AttributeError` 로 멈춘다.

  multiply(other: Tensor): Tensor {
    return this.mul(other);
  }

  // 단항의 둘째 철자 다섯. 표에 없는 이름이라 루프가 안 달아 준다.
  absolute(): Tensor {
    return this.abs();
  }

  absolute_(): Tensor {
    return this.mutate(() => this.abs());
  }

  arctan(): Tensor {
    return this.atan();
  }

  arctan_(): Tensor {
    return this.mutate(() => this.atan());
  }

  arctanh(): Tensor {
    return this.atanh();
  }

  arctanh_(): Tensor {
    return this.mutate(() => this.atanh());
  }

  /**
   * Truncates towards zero. The same as `trunc` — not `floor`.
   */
  fix(): Tensor {
    return this.trunc();
  }

  fix_(): Tensor {
    return this.mutate(() => this.trunc());
  }

  negative(): Tensor {
    return this.neg();
  }

  negative_(): Tensor {
    return this.mutate(() => this.neg());
  }

  trueDivide(other: Tensor): Tensor {
    return this.div(other);
  }

  greater(other: Tensor | number): Tensor {
    return this.binary("gt", Tensor.asTensor(other));
  }

  greaterEqual(other: Tensor | number): Tensor {
    return this.binary("ge", Tensor.asTensor(other));
  }

  less(other: Tensor | number): Tensor {
    return this.binary("lt", Tensor.asTensor(other));
  }

  lessEqual(other: Tensor | number): Tensor {
    return this.binary("le", Tensor.asTensor(other));
  }

  notEqual(other: Tensor | number): Tensor {
    return this.binary("ne", Tensor.asTensor(other));
  }

  /**
   * **It skips NaN** — `maximum` carries NaN out with it. That divergence
   * is the whole of the name.
   *
   * There is no separate kernel. If one side is NaN the definition is to
   * take the other, and `where` writes that down directly — **written as a
   * multiplication, `0 × NaN = NaN` contaminates the very slots that were
   * filtered out.** The assembly lived only in the binding for a long time.
   */
  fmax(other: Tensor): Tensor {
    return this.nanExtreme(other, "maximum");
  }

  fmin(other: Tensor): Tensor {
    return this.nanExtreme(other, "minimum");
  }

  private nanExtreme(other: Tensor, better: string): Tensor {
    // **`x.where(조건, 저쪽)` 은 조건이 참일 때 `x` 를 낸다.** 이 세션에서 두 번째로
    // 뒤집어 적었다 — `nanToNum` 이 그 첫 번째였고 골든이 바로 말해 줬다.
    const picked = this.binary(better, other);
    const out = other.where(this.unary("isnan"), picked);   // a 가 NaN 이면 b
    return this.where(other.unary("isnan"), out);           // b 가 NaN 이면 a
  }

  /**
   * Moves an axis. The same as `movedim`, and torch offers both.
   */
  moveaxis(src: number, dst: number): Tensor {
    return this.movedim(src, dst);
  }

  /**
   * Two-dimensional transpose. **One dimension or fewer is left alone** —
   * as in torch.
   */
  t(): Tensor {
    return this.shape.length < 2 ? this : this.transpose();
  }

  /**
   * The vector inner product. **For complex it conjugates the left side** —
   * on reals it is `dot`.
   */
  vdot(other: Tensor): Tensor {
    return this.isComplex() ? this.conj().mul(other).sum() : this.dot(other);
  }

  /**
   * Stretches to that shape. The same as `expand`, taking the shape **as a
   * list.**
   */
  broadcastTo(shape: readonly number[]): Tensor {
    return this.expand(...shape);
  }

  /**
   * The greatest common divisor. **It discards signs** — torch's answer is
   * always non-negative.
   */
  gcd(other: Tensor | number): Tensor {
    return this.binary("gcd", Tensor.asTensor(other));
  }

  /**
   * The least common multiple. **Zero when `gcd` is zero** (measured: the
   * lcm of 0 and 7 is 0).
   */
  lcm(other: Tensor | number): Tensor {
    return this.binary("lcm", Tensor.asTensor(other));
  }

  /**
   * The next representable float towards `other`. The point is that the
   * spacing differs with the value.
   */
  nextafter(other: Tensor | number): Tensor {
    return this.binary("nextafter", Tensor.asTensor(other));
  }

  bitwiseAnd(other: Tensor | number): Tensor {
    return this.binary("bitwise_and", Tensor.asTensor(other));
  }

  bitwiseOr(other: Tensor | number): Tensor {
    return this.binary("bitwise_or", Tensor.asTensor(other));
  }

  bitwiseXor(other: Tensor | number): Tensor {
    return this.binary("bitwise_xor", Tensor.asTensor(other));
  }

  bitwiseLeftShift(other: Tensor | number): Tensor {
    return this.binary("bitwise_left_shift", Tensor.asTensor(other));
  }

  bitwiseRightShift(other: Tensor | number): Tensor {
    return this.binary("bitwise_right_shift", Tensor.asTensor(other));
  }

  /**
   * Logical and, **reading anything non-zero as true.** That is where it
   * parts from the bitwise operation.
   */
  logicalAnd(other: Tensor | number): Tensor {
    return this.binary("logical_and", Tensor.asTensor(other));
  }

  logicalOr(other: Tensor | number): Tensor {
    return this.binary("logical_or", Tensor.asTensor(other));
  }

  logicalNot(): Tensor {
    return this.unary("logical_not");
  }

  // ── 제자리 판 서른여덟 ───────────────────────────────────────────────
  //
  // torch 는 거의 모든 연산에 밑줄 짝을 준다. 여기 있던 것은 `i0_` 하나였고, 나머지
  // 서른여덟은 **계산이 이미 있는데 밑줄 이름만 없던** 자리다. `mutate` 가 제자리성을
  // 지키므로 한 줄씩이다.
  //
  // 열은 torch 의 **둘째 철자**다(`divide_`=`div_`). 옮기면 같은 질문이 두 번이지만,
  // 이름이 없으면 그 철자로 쓴 코드가 그냥 안 돈다 — 물음이 겹치는 것과 이름이 없는
  // 것은 다른 문제다.

  bitwiseAnd_(other: Tensor | number): Tensor {
    return this.mutate(() => this.bitwiseAnd(other));
  }

  bitwiseOr_(other: Tensor | number): Tensor {
    return this.mutate(() => this.bitwiseOr(other));
  }

  bitwiseXor_(other: Tensor | number): Tensor {
    return this.mutate(() => this.bitwiseXor(other));
  }

  bitwiseLeftShift_(other: Tensor | number): Tensor {
    return this.mutate(() => this.bitwiseLeftShift(other));
  }

  bitwiseRightShift_(other: Tensor | number): Tensor {
    return this.mutate(() => this.bitwiseRightShift(other));
  }

  bitwiseNot_(): Tensor {
    return this.mutate(() => this.bitwise_not());
  }

  logicalAnd_(other: Tensor | number): Tensor {
    return this.mutate(() => this.logicalAnd(other));
  }

  logicalOr_(other: Tensor | number): Tensor {
    return this.mutate(() => this.logicalOr(other));
  }

  logicalXor_(other: Tensor): Tensor {
    return this.mutate(() => this.logicalXor(other));
  }

  logicalNot_(): Tensor {
    return this.mutate(() => this.logicalNot());
  }

  gcd_(other: Tensor | number): Tensor {
    return this.mutate(() => this.gcd(other));
  }

  lcm_(other: Tensor | number): Tensor {
    return this.mutate(() => this.lcm(other));
  }

  nextafter_(other: Tensor | number): Tensor {
    return this.mutate(() => this.nextafter(other));
  }

  clampMax_(high: number): Tensor {
    return this.mutate(() => this.clampMax(high));
  }

  clampMin_(low: number): Tensor {
    return this.mutate(() => this.clampMin(low));
  }

  digamma_(): Tensor {
    return this.mutate(() => this.digamma());
  }

  erfinv_(): Tensor {
    return this.mutate(() => this.erfinv());
  }

  lgamma_(): Tensor {
    return this.mutate(() => this.lgamma());
  }

  mvlgamma_(p: number): Tensor {
    return this.mutate(() => this.mvlgamma(p));
  }

  floorDivide_(other: Tensor | number): Tensor {
    return this.mutate(() => this.floorDivide(Tensor.asTensor(other)));
  }

  fmod_(divisor: number): Tensor {
    return this.mutate(() => this.fmod(divisor));
  }

  remainder_(divisor: number): Tensor {
    return this.mutate(() => this.remainder(divisor));
  }

  lerp_(end: Tensor, weight: Tensor | number): Tensor {
    return this.mutate(() => this.lerp(end, weight));
  }

  // 떨구기 넷의 제자리 판. **`training` 이 거짓이면 항등이고**, 그래서 `p` 를 0 으로
  // 두면 값이 결정적이라 골든이 굳힐 수 있다 — 난수 자체는 못 굳혀도 이 자리는 된다.
  dropout_(p = 0.5, training = true): Tensor {
    return this.mutate(() => this.dropout(p, training));
  }

  /**
   * **Drops whole channels** — the same computation as `dropout2d`.
   */
  featureDropout_(p = 0.5, training = true): Tensor {
    return this.mutate(() => this.featureDropout(p, training));
  }

  alphaDropout_(p = 0.5, training = false): Tensor {
    return this.mutate(() => this.alphaDropout(p, training, false));
  }

  /**
   * The per-channel form of alpha dropout. `alphaDropout` with `perChannel`
   * turned on.
   */
  featureAlphaDropout(p = 0.5, training = false): Tensor {
    return this.alphaDropout(p, training, true);
  }

  featureAlphaDropout_(p = 0.5, training = false): Tensor {
    return this.mutate(() => this.alphaDropout(p, training, true));
  }

  // 활성 다섯의 제자리 판. `F.relu_` 처럼 **`F` 쪽에만 있는 밑줄 이름**들이고,
  // 밑동은 다 여기 있었는데 밑줄 쪽이 없었다.
  celu_(alpha = 1.0): Tensor {
    return this.mutate(() => this.celu(alpha));
  }

  hardtanh_(minVal = -1.0, maxVal = 1.0): Tensor {
    return this.mutate(() => this.hardtanh(minVal, maxVal));
  }

  leakyRelu_(slope = 0.01): Tensor {
    return this.mutate(() => this.leakyRelu(slope));
  }

  rrelu_(lower = 1 / 8, upper = 1 / 3, training = false): Tensor {
    return this.mutate(() => this.rrelu(lower, upper, training));
  }

  threshold_(t: number, value: number): Tensor {
    return this.mutate(() => this.threshold(t, value));
  }

  /**
   * The two deprecated upsampling names. torch still offers them and older textbooks
   * type them.
   *
   * **`upsample` alone cannot cover both** — nearest and bilinear are different
   * computations, and the names carry that split. Folding them into one becomes a
   * quietly different interpolation (a place this repository has already been bitten
   * once, at `Upsample(mode='bilinear')`).
   */
  upsampleNearest(scale: number): Tensor {
    return this.upsample(scale);
  }

  upsampleBilinear(scale: number): Tensor {
    // **`alignCorners` 를 켠다** — torch 의 `upsample_bilinear` 이 그 기본값이고,
    // 끄면 같은 배율에서 다른 값이 나온다.
    return this.interpolateBilinear(
      (this.shape[2] ?? 1) * scale, (this.shape[3] ?? 1) * scale, true);
  }

  nanToNum_(nan = 0, posinf?: number, neginf?: number): Tensor {
    return this.mutate(() => this.nanToNum(nan, posinf, neginf));
  }

  put_(index: Tensor, source: Tensor, accumulate = false): Tensor {
    return this.mutate(() => this.put(index, source, accumulate));
  }

  renorm_(p: number, dim: number, maxnorm: number): Tensor {
    return this.mutate(() => this.renorm(p, dim, maxnorm));
  }

  /**
   * **A different operation from its counterpart.** `bernoulli()` reads its
   * own values as probabilities; this one **ignores** them and fills with
   * `p` (measured: feed it `[1,4,9,2]` and at `p=0` it is all zeros).
   *
   * Built from the counterpart on the strength of the underscore alone, the
   * values agree wherever the probability is 0 or 1, because those are
   * certain, and **it is quietly wrong only at the probabilities in
   * between.** The core kept it out of the automatic table for the same
   * reason.
   */
  bernoulli_(p = 0.5): Tensor {
    return this.mutate(
      () => Tensor.rand(this.shape).binary("lt", Tensor.full([], p)).to(this.dtype));
  }

  /**
   * **It always refuses.** `float_power`'s result is double precision and
   * there is nowhere to write it back — torch stops at a float32 slot for
   * the same reason (measured).
   */
  floatPower_(exponent: Tensor | number): Tensor {
    void exponent;
    throw new RuntimeError(
      "`float_power_` cannot be used in place — the result is double precision and " +
      "there is nowhere to put it back. Use `x.floatPower(k)` for a new tensor. " +
      "(torch: the base given to float_power_ has dtype Float but the " +
      "operation's result requires dtype Double)");
  }

  // torch 의 둘째 철자들. 계산은 위에 있고 여기서는 이름만 잇는다.
  //
  // **수만 받는다** — 짝인 `div_`·`mul_`·`sub_` 가 그렇다. 텐서를 받는 제자리
  // 산술은 여기 없고, 별칭이 짝보다 넓으면 그것은 별칭이 아니라 새 약속이 된다.
  divide_(other: number): Tensor {
    return this.div_(other);
  }

  trueDivide_(other: number): Tensor {
    return this.div_(other);
  }

  multiply_(other: number): Tensor {
    return this.mul_(other);
  }

  subtract_(other: number, alpha = 1): Tensor {
    return this.sub_(other, alpha);
  }

  /**
   * Two-dimensional transpose in place. **The shape changes** — asked with
   * a square alone, it passes without changing anything.
   */
  t_(): Tensor {
    return this.transpose_();
  }

  greater_(other: Tensor | number): Tensor {
    return this.mutate(() => this.binary("gt", Tensor.asTensor(other)));
  }

  greaterEqual_(other: Tensor | number): Tensor {
    return this.mutate(() => this.binary("ge", Tensor.asTensor(other)));
  }

  less_(other: Tensor | number): Tensor {
    return this.mutate(() => this.binary("lt", Tensor.asTensor(other)));
  }

  lessEqual_(other: Tensor | number): Tensor {
    return this.mutate(() => this.binary("le", Tensor.asTensor(other)));
  }

  notEqual_(other: Tensor | number): Tensor {
    return this.mutate(() => this.binary("ne", Tensor.asTensor(other)));
  }

  /**
   * Views the same buffer as a different shape. The same as `reshape`, and
   * in-place operations carry across.
   */
  view(...shape: number[]): Tensor {
    return this.reshape(shape);
  }

  // ── 정렬 계열 ─────────────────────────────────────────────────────────

  /**
   * Sorts one axis and produces the values and the positions.
   *
   * **The gradient follows the values.** It flows only to the positions
   * taken and is zero elsewhere, and returning the values detached means no
   * gradient reaches those positions and the whole classification loss
   * becomes non-differentiable. The core went through this with `topk` and
   * `sort`, and the sister project was in the same state until review.
   */
  sort(dim = 0, descending = false): { values: Tensor; indices: Tensor } {
    const rank = this.shape.length;
    const axis = dim < 0 ? dim + rank : dim;
    const len = this.shape[axis] ?? 1;
    const outer = this.shape.slice(0, axis).reduce((a, b) => a * b, 1);
    const inner = this.shape.slice(axis + 1).reduce((a, b) => a * b, 1);
    const values = dev().alloc(this.size);
    const indices = dev().alloc(this.size);
    dev().run1d(
      dev().pipeline(
        `srt:${outer}:${len}:${inner}:${descending ? "d" : "a"}`,
        () => sortAxis(outer, len, inner, descending),
      ),
      [this.buffer, values, indices],
      outer * inner,
    );
    const idx = new Tensor(indices, this.shape);
    return {
      values: this.gatherBack(values, this.shape, idx, axis, len, len),
      indices: idx,
    };
  }

  /**
   * The sorted positions only. Used when the values are not needed.
   */
  argsort(dim = 0, descending = false): Tensor {
    return this.sort(dim, descending).indices;
  }

  /**
   * The largest `k`. The front of `sort` — torch gives them descending too.
   */
  topk(k: number, dim = 0): { values: Tensor; indices: Tensor } {
    const rank = this.shape.length;
    const axis = dim < 0 ? dim + rank : dim;
    const sorted = this.sort(axis, true);
    return {
      values: sorted.values.narrow(axis, 0, k),
      indices: sorted.indices.narrow(axis, 0, k),
    };
  }

  /**
   * The `k`-th smallest value. **Counting from 1**, as in torch.
   */
  kthvalue(k: number, dim = 0, keepdim = false):
    { values: Tensor; indices: Tensor } {
    const rank = this.shape.length;
    const axis = dim < 0 ? dim + rank : dim;
    const sorted = this.sort(axis, false);
    return {
      values: this.liftAxis(sorted.values.select(axis, k - 1), axis, keepdim),
      indices: this.liftAxis(sorted.indices.select(axis, k - 1), axis, keepdim),
    };
  }

  /**
   * The median. **With an even count it gives the lower** — torch does not
   * average the two.
   *
   * **One NaN anywhere makes it NaN** (measured). Sorting pushes NaN to one
   * end, so simply picking **skips the NaN** and returns an intact value —
   * that is `nanmedian`, and this is not. The core had the same defect, and
   * adding a case that asks the two side by side caught both at once.
   */
  median(dim?: number, keepdim = false): { values: Tensor; indices: Tensor } {
    const spoil = (got: { values: Tensor; indices: Tensor }, axis?: number):
      { values: Tensor; indices: Tensor } => {
      // NaN 이 든 줄은 통째로 NaN 이다. `isnan` 의 합이 0 보다 크면 그 줄이다.
      const sick = axis === undefined
        ? this.flat().unary("isnan").sum()
        : this.unary("isnan").sumDim(axis);
      const bad = sick.binary("gt", Tensor.full([], 0));
      // **산술로 섞으면 안 된다.** `0 * NaN` 이 NaN 이라, 성한 줄까지 NaN 이 된다 —
      // 기존 `median` 케이스 셋이 그렇게 빨개졌다. 골라야 한다.
      const nan = Tensor.zeros(got.values.shape).add(Tensor.full([], Number.NaN));
      return { values: nan.where(bad, got.values), indices: got.indices };
    };
    if (dim === undefined) {
      const flat = this.flat();
      const k = Math.floor((flat.size + 1) / 2);
      const got = spoil(flat.kthvalue(k, 0));
      // **번호를 안 건네므로 값이 같은 칸에 고르게 나눈다**(실측: `[3,5,5,1,5]` 의
      // 기울기가 세 5 에 ⅓ 씩). `kthvalue` 는 고른 자리 하나로만 흘리는데, 그것은
      // **번호를 건네는** 연산의 규칙이다 — 여기서는 그 번호를 안 내놓는다.
      return {
        values: flat.spreadEqual(got.values.detach()),
        indices: got.indices,
      };
    }
    const rank = this.shape.length;
    const axis = dim < 0 ? dim + rank : dim;
    const len = this.shape[axis] ?? 1;
    // **NaN 을 덮어씌우는 일이 먼저다.** `keepdim` 을 `kthvalue` 에 넘겨 버리면
    // `spoil` 이 만드는 `sick` 은 축이 접힌 모양이라 `where` 가 랭크에서 어긋난다.
    // 접힌 채로 고치고 마지막에 축을 되살린다.
    const got = spoil(this.kthvalue(Math.floor((len + 1) / 2), axis), axis);
    if (!keepdim) return got;
    return {
      values: this.liftAxis(got.values, axis, true),
      indices: this.liftAxis(got.indices, axis, true),
    };
  }

  /**
   * Sorts only; it does not give positions.
   */
  msort(): Tensor {
    return this.sort(0, false).values;
  }

  /**
   * The cumulative maximum and minimum. **On a tie it gives the later
   * position** — as in torch.
   */
  cummax(dim = 0): { values: Tensor; indices: Tensor } {
    return this.cumExtremeOver("max", dim);
  }

  cummin(dim = 0): { values: Tensor; indices: Tensor } {
    return this.cumExtremeOver("min", dim);
  }

  private cumExtremeOver(kind: "max" | "min", dim: number):
    { values: Tensor; indices: Tensor } {
    const rank = this.shape.length;
    const axis = dim < 0 ? dim + rank : dim;
    const len = this.shape[axis] ?? 1;
    const outer = this.shape.slice(0, axis).reduce((a, b) => a * b, 1);
    const inner = this.shape.slice(axis + 1).reduce((a, b) => a * b, 1);
    const values = dev().alloc(this.size);
    const indices = dev().alloc(this.size);
    dev().run1d(
      dev().pipeline(
        `cx:${kind}:${outer}:${len}:${inner}`,
        () => cumExtreme(kind, outer, len, inner),
      ),
      [this.buffer, values, indices],
      this.size,
    );
    const idx = new Tensor(indices, this.shape);
    return {
      values: this.gatherBack(values, this.shape, idx, axis, len, len),
      indices: idx,
    };
  }

  /**
   * 이미 계산해 둔 값 버퍼에 **자리 표를 통한 역방향**을 붙인다.
   *
   * 순방향은 커널이 이미 냈다. 여기서 하는 일은 그래프를 잇는 것뿐이고, 역방향은
   * 자리 표를 따라 원래 칸으로 되돌리는 것이다.
   */
  private gatherBack(
    values: GPUBuffer,
    shape: readonly number[],
    indices: Tensor,
    axis: number,
    len: number,
    taken: number,
  ): Tensor {
    const outer = this.shape.slice(0, axis).reduce((a, b) => a * b, 1);
    const inner = this.shape.slice(axis + 1).reduce((a, b) => a * b, 1);
    const inShape = this.shape;
    const inSize = this.size;
    return Tensor.make(
      values,
      shape,
      [this],
      (g) => {
        const gi = dev().alloc(inSize);
        dev().run1d(
          dev().pipeline(
            `sbi:${outer}:${len}:${inner}:${taken}`,
            () => scatterByIndex(outer, len, inner, taken),
          ),
          [indices.buffer, g.buffer, gi],
          inSize,
        );
        return [new Tensor(gi, inShape)];
      },
      "SortBackward0",
      this.dtype,
    );
  }

  /**
   * **Adds** at the positions the indices point to. Overlaps accumulate.
   *
   * The opposite of `gather`. With only one of them you can take out and
   * not put back, and code hand- building embeddings or one-hots meets that
   * immediately.
   */
  scatterAdd(dim: number, index: Tensor, src: Tensor): Tensor {
    return this.scatterWith(dim, index, src, "add");
  }

  /**
   * **Overwrites** at the positions the indices point to. On an overlap,
   * the last write stands.
   */
  scatterSet(dim: number, index: Tensor, src: Tensor): Tensor {
    return this.scatterWith(dim, index, src, "set");
  }

  private scatterWith(
    dim: number,
    index: Tensor,
    src: Tensor,
    kind: "add" | "set",
  ): Tensor {
    const rank = this.shape.length;
    const axis = dim < 0 ? dim + rank : dim;
    const outer = this.shape.slice(0, axis).reduce((a, b) => a * b, 1);
    const inner = this.shape.slice(axis + 1).reduce((a, b) => a * b, 1);
    const len = this.shape[axis] ?? 1;
    const taken = index.shape[axis] ?? 1;
    const d = dev();
    const out = d.alloc(this.size);
    if (kind === "add") {
      const spread = d.alloc(this.size);
      d.run1d(
        d.pipeline(`sbi:${outer}:${len}:${inner}:${taken}`,
          () => scatterByIndex(outer, len, inner, taken)),
        [index.buffer, src.buffer, spread],
        this.size,
      );
      // **그래프를 이어야 한다.** 맨 `new Tensor` 로 두면 `src` 로 기울기가 안
      // 가고, 증상은 "requires grad 가 아니다" 라는 먼 자리의 오류였다.
      // 흩뿌리기의 반대는 모으기이므로 역방향이 `gather` 다.
      const scattered = Tensor.make(
        spread, this.shape, [src],
        (g) => [g.gather(axis, index)],
        "ScatterAddBackward0",
      );
      return this.add(scattered);
    }
    d.run1d(
      d.pipeline(`sbo:${outer}:${len}:${inner}:${taken}`,
        () => scatterOverwrite(outer, len, inner, taken)),
      [index.buffer, src.buffer, this.buffer, out],
      this.size,
    );
    // 덮어쓴 자리는 원본과 끊긴다 — 그 자리에는 0 이 간다. 자리 표를 그대로
    // 쓰면 되므로 새 커널이 필요 없다.
    const written = Tensor.zeros(this.shape)
      .scatterSetMask(index, outer, len, inner, taken);
    const keep = Tensor.full([], 1).sub(written);
    return Tensor.make(
      out,
      this.shape,
      [this, src],
      (g) => [g.mul(keep), g.gather(axis, index)],
      "ScatterBackward0",
      this.dtype,
    );
  }

  /** 어느 칸이 쓰였는지의 표. 값이 1 인 자리가 덮어쓴 자리다. */
  private scatterSetMask(
    index: Tensor,
    outer: number, len: number, inner: number, taken: number,
  ): Tensor {
    const d = dev();
    const mask = d.alloc(this.size);
    d.run1d(
      d.pipeline(`sbo:${outer}:${len}:${inner}:${taken}`,
        () => scatterOverwrite(outer, len, inner, taken)),
      [index.buffer, Tensor.zeros(index.shape).add(Tensor.full([], 1)).buffer,
        this.buffer, mask],
      this.size,
    );
    return new Tensor(mask, this.shape);
  }

  // ── 번호표로 읽고 쓰기 ────────────────────────────────────────────────
  //
  // `as_strided`·`select_scatter`·`slice_scatter`·`diagonal_scatter`·`put`·
  // `index_put` 이 전부 이 두 문(`gatherSpots`·`scatterSpots`)을 지난다.

  /** 번호표를 GPU 로 올린다. 모양만으로 정해지는 자리라 기울기를 안 낸다. */
  private static spotsTensor(spots: Float32Array): Tensor {
    return Tensor.from(spots, [spots.length]);
  }

  /**
   * Reads the slots the index table points at and produces `shape`.
   *
   * **Overlapping indices accumulate gradient** — a slot read twice
   * receives twice. Measured with non-overlapping steps alone, that
   * accumulation never shows.
   */
  gatherSpots(spots: Tensor, shape: readonly number[]): Tensor {
    const n = spots.size;
    const out = dev().alloc(n);
    dev().run1d(
      dev().pipeline(`fg:${n}`, () => flatGather(n)),
      [this.buffer, spots.buffer, out],
      n,
    );
    const mine = this.shape;
    const size = this.size;
    return Tensor.make(
      out,
      shape,
      [this],
      (g) => {
        const gi = dev().alloc(size);
        dev().run1d(
          dev().pipeline(`fgb:${size}:${n}`, () => flatGatherBackward(size, n)),
          [spots.buffer, g.buffer, gi],
          size,
        );
        return [new Tensor(gi, mine)];
      },
      "AsStridedBackward0",
      this.dtype,
    );
  }

  /**
   * A **copy** with `src` written at the index table's positions.
   */
  scatterSpots(
    spots: Tensor,
    src: Tensor,
    accumulate = false,
    gradName = "SliceScatterBackward0",
  ): Tensor {
    const n = this.size;
    const count = spots.size;
    const out = dev().alloc(n);
    dev().run1d(
      dev().pipeline(
        `fs:${n}:${count}:${accumulate}`,
        () => flatScatterInto(n, count, accumulate),
      ),
      [this.buffer, spots.buffer, src.buffer, out],
      n,
    );
    const mine = this.shape;
    const srcShape = src.shape;
    return Tensor.make(
      out,
      mine,
      [this, src],
      // **쌓는 쪽은 원본이 안 끊긴다.** 덮어쓰는 쪽만 그 자리로 0 이 간다.
      (g) => [
        accumulate ? g : g.mul(Tensor.ones(mine).zeroAtSpots(spots)),
        g.gatherSpots(spots, srcShape),
      ],
      gradName,
      this.dtype,
    );
  }

  /** 번호표 자리만 0 인 표. 덮어쓴 자리로는 기울기가 안 간다. */
  private zeroAtSpots(spots: Tensor): Tensor {
    const n = this.size;
    const count = spots.size;
    const out = dev().alloc(n);
    dev().run1d(
      dev().pipeline(
        `fs:${n}:${count}:false`,
        () => flatScatterInto(n, count, false),
      ),
      [this.buffer, spots.buffer, Tensor.zeros([count]).buffer, out],
      n,
    );
    return new Tensor(out, this.shape);
  }

  /**
   * Reads flat storage **with different strides.** torch gives a view; here
   * it is a copy.
   */
  asStrided(
    size: readonly number[],
    stride: readonly number[],
    storageOffset = 0,
  ): Tensor {
    const spots = stridedSpots(size, stride, storageOffset);
    return this.gatherSpots(Tensor.spotsTensor(spots), [...size]);
  }

  asStridedScatter(
    src: Tensor,
    size: readonly number[],
    stride: readonly number[],
    storageOffset = 0,
  ): Tensor {
    const spots = stridedSpots(size, stride, storageOffset);
    return this.scatterSpots(Tensor.spotsTensor(spots), src, false,
      "AsStridedScatterBackward0");
  }

  /**
   * A **copy** with the plate `select` would have taken swapped out.
   */
  selectScatter(src: Tensor, dim: number, index: number): Tensor {
    const rank = this.shape.length;
    const axis = dim < 0 ? dim + rank : dim;
    const at = index < 0 ? index + (this.shape[axis] ?? 0) : index;
    const { spots } = selectSpots(this.shape, axis, at);
    return this.scatterSpots(Tensor.spotsTensor(spots), src, false,
      "SelectScatterBackward0");
  }

  /**
   * A **copy** with the `x[..., start:end:step]` region swapped out.
   */
  sliceScatter(
    src: Tensor,
    dim = 0,
    start?: number,
    end?: number,
    step = 1,
  ): Tensor {
    const rank = this.shape.length;
    const axis = dim < 0 ? dim + rank : dim;
    const len = this.shape[axis] ?? 0;
    const from = start === undefined || start === null ? 0 : start;
    const to = end === undefined || end === null ? len : Math.min(end, len);
    const { spots } = sliceSpots(this.shape, axis, from, to, step);
    return this.scatterSpots(Tensor.spotsTensor(spots), src, false,
      "SliceScatterBackward0");
  }

  /**
   * A **copy** with the diagonal swapped out.
   */
  diagonalScatter(src: Tensor, offset = 0, dim1 = 0, dim2 = 1): Tensor {
    const rank = this.shape.length;
    const d1 = dim1 < 0 ? dim1 + rank : dim1;
    const d2 = dim2 < 0 ? dim2 + rank : dim2;
    const { spots } = diagonalSpots(this.shape, offset, d1, d2);
    return this.scatterSpots(Tensor.spotsTensor(spots), src, false,
      "DiagonalScatterBackward0");
  }

  /**
   * **Spreads the last axis onto a diagonal**, adding an axis. The opposite
   * of `diagonal`.
   */
  diagEmbed(offset = 0, dim1 = -2, dim2 = -1): Tensor {
    const n = (this.shape[this.shape.length - 1] ?? 0) + Math.abs(offset);
    const rank = this.shape.length + 1;
    const d1 = dim1 < 0 ? dim1 + rank : dim1;
    const d2 = dim2 < 0 ? dim2 + rank : dim2;
    const shape = this.shape.slice(0, -1);
    for (const at of [d1, d2].sort((a, b) => a - b)) shape.splice(at, 0, n);
    const { spots } = diagonalSpots(shape, offset, d1, d2);
    // **바탕의 형이 결과의 형이다.** `zeros` 는 float32 라 그냥 두면 int64 를 넣어도
    // float32 가 나온다 — 값은 맞고 이름만 갈리는 자리다.
    return Tensor.zeros(shape).to(this.dtype).scatterSpots(
      Tensor.spotsTensor(spots), this.reshape([spots.length]), false,
      "DiagEmbedBackward0",
    );
  }

  /**
   * **Flattens** and writes by index — there is no notion of an axis. The
   * opposite of `take`.
   *
   * The indices arrive as values, so it uses the same door as a CPU-built
   * index table.
   */
  put(index: Tensor, source: Tensor, accumulate = false): Tensor {
    return this.scatterSpots(index.reshape([index.size]),
      source.reshape([source.size]), accumulate, "PutBackward0")
      .reshape(this.shape);
  }

  /**
   * Takes one index tensor per axis and writes at those positions.
   *
   * The per-axis indices are **folded into one flat index** — which makes
   * it the same door as `put`. The folding is tensor arithmetic, so no
   * kernel is added.
   */
  indexPut(
    indices: readonly Tensor[],
    values: Tensor,
    accumulate = false,
  ): Tensor {
    const st = rowStrides(this.shape);
    let flat: Tensor | null = null;
    indices.forEach((idx, d) => {
      const part = idx.mul(Tensor.full([], st[d] ?? 1));
      flat = flat === null ? part : flat.add(part);
    });
    const spots = flat ?? Tensor.zeros([0]);
    return this.scatterSpots(spots.reshape([spots.size]),
      values.reshape([values.size]), accumulate, "IndexPutBackward0");
  }

  /** 번호표 자리에 **합치며** 넣는다. `scatter_reduce`·`index_reduce` 의 밑동이다. */
  private reduceSpots(
    spots: Tensor,
    src: Tensor,
    reduce: string,
    includeSelf: boolean,
  ): Tensor {
    const n = this.size;
    const count = spots.size;
    const out = dev().alloc(n);
    dev().run1d(
      dev().pipeline(
        `fr:${n}:${count}:${reduce}:${includeSelf}`,
        () => flatReduceInto(n, count, reduce, includeSelf),
      ),
      [this.buffer, spots.buffer, src.buffer, out],
      n,
    );
    return new Tensor(out, this.shape, { dtype: this.dtype });
  }

  /**
   * The same place as `scatter`, but **combining instead of overwriting.**
   */
  scatterReduce(
    dim: number,
    index: Tensor,
    src: Tensor,
    reduce: string,
    includeSelf = true,
  ): Tensor {
    const rank = this.shape.length;
    const axis = dim < 0 ? dim + rank : dim;
    const st = rowStrides(this.shape);
    // 번호는 축 하나 몫이고 나머지 좌표는 제자리다 — 그 자리표를 더해 평평하게 만든다.
    const rest = new Float32Array(index.size);
    eachCoord(index.shape, (c, i) => {
      let p = 0;
      for (let d = 0; d < index.shape.length; d++) {
        if (d !== axis) p += c[d]! * st[d]!;
      }
      rest[i] = p;
    });
    const spots = index.mul(Tensor.full([], st[axis] ?? 1))
      .reshape([index.size])
      .add(Tensor.spotsTensor(rest));
    return this.reduceSpots(spots, src.reshape([src.size]), reduce, includeSelf);
  }

  /**
   * Combines the **rows** the indices point at. Unlike `scatterReduce`, the
   * indices are per row.
   */
  indexReduce(
    dim: number,
    index: Tensor,
    source: Tensor,
    reduce: string,
    includeSelf = true,
  ): Tensor {
    const rank = this.shape.length;
    const axis = dim < 0 ? dim + rank : dim;
    const st = rowStrides(this.shape);
    const shape = source.shape;
    const rest = new Float32Array(source.size);
    const which = new Float32Array(source.size);
    eachCoord(shape, (c, i) => {
      let p = 0;
      for (let d = 0; d < shape.length; d++) {
        if (d !== axis) p += c[d]! * st[d]!;
      }
      rest[i] = p;
      which[i] = c[axis]!;
    });
    // 줄 번호를 자리마다 펴서 더한다 — 그러면 `scatterReduce` 와 같은 번호표가 된다.
    const picked = index.reshape([index.size])
      .gatherSpots(Tensor.spotsTensor(which), [source.size])
      .mul(Tensor.full([], st[axis] ?? 1));
    return this.reduceSpots(picked.add(Tensor.spotsTensor(rest)),
      source.reshape([source.size]), reduce, includeSelf);
  }

  /**
   * Fills `source` **in flat order** into the positions where the mask is
   * true.
   *
   * Which value goes where depends on the mask's **values** while the
   * result shape is fixed — so rather than reading the values back, it is
   * solved with a kernel that counts, per position, how many trues came
   * before.
   */
  maskedScatter(mask: Tensor, source: Tensor): Tensor {
    const n = this.size;
    const wide = mask.size === n ? mask : mask.add(Tensor.zeros(this.shape));
    const flat = source.reshape([source.size]);
    const out = dev().alloc(n);
    dev().run1d(
      dev().pipeline(`ms:${n}`, () => maskedScatterKernel(n)),
      [this.buffer, wide.buffer, flat.buffer, out],
      n,
    );
    const mine = this.shape;
    const srcShape = source.shape;
    const count = source.size;
    return Tensor.make(
      out,
      mine,
      [this, source],
      (g) => {
        const gi = dev().alloc(count);
        dev().run1d(
          dev().pipeline(
            `msb:${count}:${n}`,
            () => maskedScatterSourceBackward(count, n),
          ),
          [wide.buffer, g.buffer, gi],
          count,
        );
        // 가면이 참인 자리는 원본과 끊긴다 — 거짓인 자리만 흘려보낸다.
        //
        // **형을 먼저 실수로 돌린다.** 가면은 `bool` 이고 뺄셈은 `bool` 을 거절한다 —
        // torch 도 그렇다. 값이 0/1 이라 셈은 되는데 그 앞에서 막힌다.
        const keep = Tensor.full([], 1).sub(wide.to("float32"));
        return [g.mul(keep), new Tensor(gi, srcShape)];
      },
      "MaskedScatterBackward0",
      this.dtype,
    );
  }

  // ── 결속에만 있던 이름들 ─────────────────────────────────────────────
  //
  // 아래는 전부 **torch 에 있고, 결속(`borch_webgpu`)이 파이썬에서 조립하고 있던**
  // 이름이다. 골든 케이스가 결속을 지나므로 표는 초록이었고, 없는 것은 borch.ts 를
  // TypeScript 에서 쓰는 쪽뿐이었다 — `tests/test_binding_fills_in.py` 가 그 자리를
  // 세어 준다. 조립 자체는 옳았으므로 셈은 그대로 옮기고, **이름이 사는 자리**만
  // 바꾼다. 결속은 이제 넘기기만 한다.

  /**
   * The quotient along `dim` by **floor division.** It rounds towards −∞,
   * not towards zero.
   */
  floorDivide(other: Tensor): Tensor {
    return this.div(other).unary("floor");
  }

  /**
   * A power. **torch computes it promoted to double** — we do not have that
   * type.
   *
   * The value is the same as `pow` and only the type differs, and having no
   * such type they are the same thing here. That is also why torch itself
   * refuses `float_power_` at a float32 slot — there is nowhere to write it
   * back.
   */
  floatPower(exponent: Tensor | number): Tensor {
    const k = exponent instanceof Tensor ? exponent : Tensor.full([], exponent);
    return this.binary("pow", k);
  }

  /**
   * `other - alpha * this`. The place where subtraction's sides are
   * swapped.
   */
  rsub(other: Tensor, alpha = 1): Tensor {
    return other.sub(this.mul(Tensor.full([], alpha)));
  }

  /**
   * The position `weight` of the way from `this` to `end`.
   *
   * **The weight may be a tensor.** `lerpFrom` takes numbers only, so it
   * cannot give a different weight per position — torch can.
   */
  lerp(end: Tensor, weight: Tensor | number): Tensor {
    const w = weight instanceof Tensor ? weight : Tensor.full([], weight);
    return this.add(end.sub(this).mul(w));
  }

  /**
   * Exclusive or. **Absent from the binary table**, so it is built by
   * asking twice whether it differs from zero.
   */
  logicalXor(other: Tensor): Tensor {
    const zero = Tensor.full([], 0);
    return this.binary("ne", zero).binary("ne", other.binary("ne", zero));
  }

  /**
   * NaN and infinities to finite numbers. **Unspecified, they are f32's
   * extremes** (as in torch).
   *
   * It selects with `where` — multiplying by a mask gives `0 × NaN = NaN`,
   * which contaminates the very slots that were filtered out. A place this
   * repository has been bitten by three times.
   */
  nanToNum(nan = 0, posinf?: number, neginf?: number): Tensor {
    const hi = posinf ?? F32_MAX;
    const lo = neginf ?? -F32_MAX;
    // **`x.where(조건, 저쪽)` 은 조건이 참일 때 `x` 를 낸다.** 채울 값이 수신자
    // 자리에 와야 한다 — 반대로 적으면 NaN 자리에 NaN 이 그대로 남는다(실측:
    // 골든 다섯이 `최대차 inf` 로 갈렸다).
    const spread = (v: number): Tensor =>
      Tensor.zeros(this.shape).add(Tensor.full([], v));
    let out = spread(nan).where(this.unary("isnan"), this);
    out = spread(hi).where(this.isposinf(), out);
    return spread(lo).where(this.isneginf(), out);
  }

  isposinf(): Tensor {
    return this.unary("isinf").binary("mul", this.binary("gt", Tensor.full([], 0)));
  }

  isneginf(): Tensor {
    return this.unary("isinf").binary("mul", this.binary("lt", Tensor.full([], 0)));
  }

  /**
   * Whether there is only a real part. **Non-complex, everything is true,
   * and that is a fact.**
   */
  isreal(): Tensor {
    if (!this.isComplex()) {
      return Tensor.ones(this.shape).binary("gt", Tensor.full([], 0));
    }
    return this.imag().binary("eq", Tensor.full([], 0));
  }

  /**
   * Whether each position is **close.** `|a−b| ≤ atol + rtol·|b|`.
   *
   * **It leans to the right** — it measures against `b`'s magnitude, so
   * swapping the two can change the answer. torch defines it that way.
   */
  isclose(other: Tensor, rtol = 1e-5, atol = 1e-8): Tensor {
    const room = other.abs().mul(Tensor.full([], rtol)).add(Tensor.full([], atol));
    return this.sub(other).abs().binary("le", room);
  }

  // `allclose` 는 여기 없다 — **이미 아래에 있고 그쪽이 낫다.** 모양을 먼저 보고
  // `equalNan` 까지 받는다. 짝이라고 나란히 새로 적었다가 중복으로 걸렸다.

  /**
   * Whether each element is in that list. It falls out of broadcasting
   * alone — no values are read.
   */
  isin(test: Tensor): Tensor {
    const grid = this.reshape([this.size, 1])
      .binary("eq", test.reshape([1, test.size]));
    return grid.to("float32").sumDim(1)
      .binary("gt", Tensor.full([], 0)).reshape(this.shape);
  }

  /**
   * The minimum and maximum **together.** Asking for one alone would not
   * catch the other being wrong — that is why torch gives this name
   * separately.
   */
  aminmax(dim?: number, keepdim = false): { min: Tensor; max: Tensor } {
    return {
      min: this.amin(dim, keepdim),
      max: this.amax(dim, keepdim),
    };
  }

  /**
   * The standard deviation and mean together. One name for the same reason
   * as `aminmax`.
   */
  stdMean(correction = 1): { std: Tensor; mean: Tensor } {
    return { std: this.std(correction), mean: this.mean() };
  }

  varMean(correction = 1): { variance: Tensor; mean: Tensor } {
    return { variance: this.variance(correction), mean: this.mean() };
  }

  /**
   * The conjugate transpose of the last two axes. On reals it is the
   * transpose.
   */
  adjoint(): Tensor {
    return this.isComplex() ? this.conj().swapaxes(-2, -1) : this.swapaxes(-2, -1);
  }

  /**
   * The distance between every pair. **It falls out of broadcasting alone**
   * — no kernel is added.
   *
   * `p === 2` is kept separate not because it is faster but because **the
   * square root has no derivative at zero.** Down the general path (`|d|^p`
   * then `^(1/p)`) the gradient becomes NaN at that same place.
   */
  cdist(other: Tensor, p = 2.0): Tensor {
    const [n, k] = [this.shape[0] ?? 0, this.shape[1] ?? 0];
    const m = other.shape[0] ?? 0;
    const diff = this.reshape([n, 1, k]).sub(other.reshape([1, m, k]));
    if (p === 2.0) return diff.mul(diff).sumDim(2).sqrt();
    return diff.abs().powScalar(p).sumDim(2).powScalar(1 / p);
  }

  /**
   * Covariance. **Rows are variables and columns are observations** — the
   * axes are the opposite of numpy's, which makes it a confusing place.
   */
  cov(correction = 1): Tensor {
    const wide = this.shape.length === 1
      ? this.reshape([1, this.shape[0] ?? 0])
      : this;
    const n = wide.shape[1] ?? 0;
    const centered = wide.sub(wide.mean(1, true));
    return centered.mm(centered.swapaxes(0, 1))
      .mul(Tensor.full([], 1 / Math.max(1, n - correction)));
  }

  /**
   * The Kronecker product. **One-dimensional only** — above that it
   * refuses.
   *
   * The version in the binding was looking at one axis only (two
   * `shape[0]`s), and given two dimensions the answer was quietly wrong. It
   * could have been fixed while carrying it across, but that writes down a
   * capability that is not here as though it were — **a missing feature
   * beats a wrong answer** is this repository's rule.
   */
  kron(other: Tensor): Tensor {
    if (this.shape.length !== 1 || other.shape.length !== 1) {
      throw new RuntimeError(
        "kron only does 1-D — two or more dimensions are not here yet. " +
        `(got shapes [${this.shape}] and [${other.shape}])`,
      );
    }
    const n = this.shape[0] ?? 0;
    const m = other.shape[0] ?? 0;
    return this.reshape([n, 1]).mul(other.reshape([1, m])).reshape([n * m]);
  }

  /**
   * The covariance divided by the standard deviations. **The diagonal comes
   * out 1** — that is the check.
   */
  corrcoef(): Tensor {
    const c = this.cov();
    const n = c.shape[0] ?? 0;
    const diag = c.diagonal();
    const scale = diag.reshape([n, 1]).mul(diag.reshape([1, n]));
    return c.div(scale.sqrt());
  }

  /**
   * Folds and multiplies along the named axes. It herds the folded axes and
   * finishes with **one matrix multiply.**
   *
   * @param dims a number means that many from the end; two lists mean the
   *   axes of each.
   */
  tensordot(other: Tensor, dims: number | readonly [readonly number[], readonly number[]] = 2): Tensor {
    const ash = this.shape;
    const bsh = other.shape;
    const [left, right] = typeof dims === "number"
      ? [
        Array.from({ length: dims }, (_, i) => ash.length - dims + i),
        Array.from({ length: dims }, (_, i) => i),
      ]
      : [[...dims[0]], [...dims[1]]];
    const aKeep = ash.map((_, i) => i).filter((i) => !left.includes(i));
    const bKeep = bsh.map((_, i) => i).filter((i) => !right.includes(i));
    const aShape = aKeep.map((i) => ash[i] ?? 1);
    const bShape = bKeep.map((i) => bsh[i] ?? 1);
    const inner = left.reduce((acc, i) => acc * (ash[i] ?? 1), 1);
    const rows = aShape.reduce((a, b) => a * b, 1);
    const cols = bShape.reduce((a, b) => a * b, 1);
    const am = this.permute([...aKeep, ...left]).reshape([rows, inner]);
    const bm = other.permute([...right, ...bKeep]).reshape([inner, cols]);
    return am.mm(bm).reshape([...aShape, ...bShape]);
  }

  /** 1 차원 번호를 `shape` 모양으로 편다 — `index_*` 셋이 같은 문을 쓴다. */
  private spreadIndex(index: Tensor, dim: number, shape: readonly number[]): Tensor {
    const lifted = shape.map((_, d) => (d === dim ? index.size : 1));
    return index.reshape(lifted).expand(...shape);
  }

  indexAdd(dim: number, index: Tensor, source: Tensor, alpha = 1): Tensor {
    const spread = this.spreadIndex(index, dim, source.shape);
    const src = alpha === 1 ? source : source.mul(Tensor.full([], alpha));
    return this.scatterAdd(dim, spread, src);
  }

  indexCopy(dim: number, index: Tensor, source: Tensor): Tensor {
    return this.scatterSet(dim, this.spreadIndex(index, dim, source.shape), source);
  }

  indexFill(dim: number, index: Tensor, value: number): Tensor {
    const shape = this.shape.map((n, d) => (d === dim ? index.size : n));
    return this.scatterSet(dim, this.spreadIndex(index, dim, shape),
      Tensor.full(shape, value));
  }

  /**
   * The diagonal to one value.
   *
   * **`wrap` only means something on a tall matrix.** The stride is `cols +
   * 1` either way, and all that differs is **where it stops** — off, it
   * stops at `cols²` slots and paints only the first block's diagonal; on,
   * it runs to the end and paints the lower blocks' diagonals too. numpy's
   * `fill_diagonal` is exactly that one line and torch takes it as-is.
   *
   * It was first written down as "the wrapped position skips a row", and
   * coded that way — no such rule exists. There was one case and it asked
   * with a square, so that branch was **passing through nothing** — a case
   * asking with a tall matrix goes in alongside.
   */
  fillDiagonal_(value: number, wrap = false): Tensor {
    const rows = this.shape[0] ?? 0;
    const cols = this.shape[1] ?? 0;
    const step = cols + 1;
    const limit = wrap ? rows * cols : Math.min(rows * cols, cols * cols);
    const spots: number[] = [];
    for (let at = 0; at < limit; at += step) spots.push(at);
    return this.mutate(() => this.reshape([this.size]).scatterSpots(
      Tensor.spotsTensor(Float32Array.from(spots)),
      Tensor.full([spots.length], value), false, "FillDiagonalBackward0",
    ).reshape(this.shape));
  }

  /**
   * 자리마다 새로 뽑아 **덮어쓴다.** 원래 값은 안 본다.
   *
   * 여섯이 한 문을 쓴다 — 분포마다 다른 것은 균등난수 하나를 무엇으로 바꾸느냐뿐이다.
   * 결속은 이 여섯을 numpy 로 만드는데, 그쪽은 `get_rng_state` 가 한 줄기를
   * 직렬화해야 해서 그런 것이고 이쪽이 그럴 이유는 없다.
   */
  private drawInto_(draw: (u: number) => number): Tensor {
    const data = new Float32Array(this.size);
    for (let i = 0; i < data.length; i++) data[i] = draw(uniform());
    return this.mutate(() => Tensor.from(data, this.shape, { dtype: this.dtype }));
  }

  /**
   * **연속 분포는 정수 칸에 답이 없다.** 다섯이 여기서 멈추고 `geometric_`·`random_`
   * 은 안 멈춘다 — 그 둘은 이산이라 정수 칸에 답이 있다.
   *
   * 이름만 보고 "난수는 실수만" 으로 묶으면 그 둘에서 틀린다. 처음 다섯을 넣을 때
   * 이 문을 안 달았고, 그러면 `zeros(6, int64).exponential_()` 이 **조용히 돌면서**
   * 정수 칸에 잘린 실수를 넣는다.
   *
   * **예외 종류가 torch 안에서도 갈린다**(실측): `normal_`·`uniform_`·`log_normal_`
   * 은 `NotImplementedError` 이고 `exponential_`·`cauchy_` 는 `RuntimeError` 다.
   * 하나로 묶으면 셋이 갈리고, 그 갈림은 값이 아니라 예외 이름이라 값으로 대조하는
   * 검사에는 안 걸린다. 부르는 쪽이 어느 쪽인지 적는다.
   */
  private needsFloatDraw(who: string, kind: "runtime" | "unimplemented"): void {
    if (this.dtype === "float32") return;
    const said = `"${who}" not implemented for '${this.dtype}' — ` +
      "연속 분포는 실수 칸에만 뽑습니다.";
    throw kind === "runtime"
      ? new RuntimeError(said)
      : new NotImplementedError(said);
  }

  /**
   * The exponential distribution. Its mean is `1/lambd`, and **lambd has to
   * be positive**.
   */
  exponential_(lambd = 1.0): Tensor {
    this.needsFloatDraw("exponential_", "runtime");
    if (!(lambd > 0)) {
      throw new RuntimeError(
        `exponential_ expects lambda > 0.0, but found lambda=${lambd}`);
    }
    // **`1 - u` 를 쓴다.** `uniform()` 이 0 을 낼 수 있고 `log(0)` 은 −∞ 다.
    return this.drawInto_((u) => -Math.log(1 - u) / lambd);
  }

  /**
   * The Cauchy distribution. **It has no mean** — the tails are heavy
   * enough that the sample mean does not converge.
   */
  cauchy_(median = 0.0, sigma = 1.0): Tensor {
    this.needsFloatDraw("cauchy_", "runtime");
    return this.drawInto_((u) => median + sigma * Math.tan(Math.PI * (u - 0.5)));
  }

  /**
   * The log-normal distribution. `mean` and `std` are the values **after
   * taking logs** (as in torch).
   */
  logNormal_(mean = 1.0, std = 2.0): Tensor {
    this.needsFloatDraw("log_normal_", "unimplemented");
    return this.drawInto_(() => Math.exp(mean + std * gauss()));
  }

  /**
   * Overwrites with the normal distribution. **`std` cannot be negative** —
   * at 0 it is the mean itself.
   */
  normal_(mean = 0.0, std = 1.0): Tensor {
    this.needsFloatDraw("normal_", "unimplemented");
    if (!(std >= 0)) {
      throw new RuntimeError(
        `normal_ expects std >= 0.0, but found std=${std}`);
    }
    return this.drawInto_(() => mean + std * gauss());
  }

  /**
   * Overwrites with reals in `[from, to)`. **`from < to` is required.**
   */
  uniform_(from = 0.0, to = 1.0): Tensor {
    this.needsFloatDraw("uniform_", "unimplemented");
    if (!(from < to)) {
      throw new RuntimeError(
        `uniform_ expects to return a [from, to) range, but found from=${from} > to=${to}`);
    }
    return this.drawInto_((u) => from + u * (to - from));
  }

  /**
   * The geometric distribution. **Being discrete, it runs on integer
   * tensors too** — that is what separates it from the continuous ones.
   *
   * `p` is an **open interval**. At 0 the event never happens and at 1 it
   * always happens on the first try, so neither is a distribution.
   */
  geometric_(p: number): Tensor {
    if (!(p > 0 && p < 1)) {
      throw new RuntimeError(
        `geometric_ expects p to be in (0, 1), but got p=${p}`);
    }
    return this.drawInto_((u) => Math.floor(Math.log(1 - u) / Math.log(1 - p)) + 1);
  }

  /**
   * Fills with integers from `[from, to)`.
   *
   * **Left out, the bound is as far as f32 counts exactly** (2^24). torch's
   * depends on the dtype and is 2^62 for int64, but int64 here lives in an
   * f32 cell and cannot count past that — drawn values above it have
   * neighbours that cannot be told apart. That is not imitated.
   */
  random_(from = 0, to?: number): Tensor {
    const high = to ?? EXACT_INT_LIMIT;
    if (!(from < high)) {
      throw new RuntimeError(
        `random_ expects 'from' to be less than 'to', but got from=${from} >= to=${high}`);
    }
    return this.drawInto_((u) => from + Math.floor(u * (high - from)));
  }

  /**
   * Whether the type is a floating-point one. **It asks about the type, not
   * the values** — it can be true with every value an integer.
   *
   * Looking at `dtype` would tell you, but the name exists in torch and did
   * not exist here. The binding was writing that branch in Python, which
   * leaves nowhere for the TypeScript side to ask.
   */
  isFloatingPoint(): boolean {
    return this.dtype === "float32";
  }

  /**
   * Whether the type can hold negatives. Only `bool` is false.
   */
  isSigned(): boolean {
    return this.dtype !== "bool";
  }

  /**
   * Whether this single-element tensor's value is non-zero.
   *
   * **With more than one it stops.** It is the place that keeps Python's
   * `if tensor:` from quietly looking at the first element, which is why
   * torch gives this name separately. It has to look at the value, so it is
   * asynchronous.
   */
  async isNonzero(): Promise<boolean> {
    if (this.size !== 1) {
      throw new RuntimeError(
        `Boolean value of Tensor with ${this.size} elements is ambiguous`);
    }
    return (await this.item()) !== 0;
  }

  fullLike(value: number): Tensor {
    return Tensor.full(this.shape, value);
  }

  /**
   * **It borrows the shape only.** Unlike torch it leaves the values at 0 —
   * garbage is not a thing to learn.
   */
  emptyLike(): Tensor {
    return Tensor.zeros(this.shape);
  }

  randintLike(low: number, high: number): Tensor {
    return Tensor.randint(low, high, this.shape);
  }

  /**
   * Pulls **each slice along `dim` down below `maxnorm`** in norm.
   *
   * Left as an assembly — **the scale has `x` inside it**, so the backward
   * is not `g·s`. Written by hand, the forward is right and the gradient
   * quietly wrong (which is how it went wrong once in the core).
   */
  renorm(p: number, dim: number, maxnorm: number): Tensor {
    const rank = this.shape.length;
    const axis = dim < 0 ? dim + rank : dim;
    // 재는 축을 앞으로 보내고 나머지를 한 줄로 눕히면 축약이 한 번으로 끝난다.
    const keep = this.shape[axis] ?? 1;
    const moved = this.movedim(axis, 0).reshape([keep, -1]);
    const norms = moved.abs().powScalar(p).sumDim(1, true).powScalar(1 / p);
    // `gt` 는 표에만 있는 이름이라 메서드가 아니다 — `binary` 로 부른다.
    const scale = norms.binary("gt", Tensor.full([], maxnorm))
      .mul(Tensor.full([], maxnorm).div(norms.add(Tensor.full([], 1e-7)))
        .sub(Tensor.full([], 1)))
      .add(Tensor.full([], 1));
    const restored: number[] = [keep];
    for (let d = 0; d < rank; d++) if (d !== axis) restored.push(this.shape[d]!);
    return moved.mul(scale).reshape(restored).movedim(0, axis);
  }

  /**
   * Unpacks a flat index into per-axis indices. **One tensor per axis,
   * returned as a bundle** (measured).
   *
   * The arithmetic is division and remainder only, so no values are read.
   */
  unravelIndex(shape: readonly number[]): Tensor[] {
    const st = rowStrides(shape);
    return shape.map((size, d) => this
      .div(Tensor.full([], st[d] ?? 1)).unary("floor")
      .remainder(size));
  }

  /**
   * Multiplies several matrices in sequence. `multiDot` takes the same
   * thing as a list.
   */
  static chainMatmul(...matrices: readonly Tensor[]): Tensor {
    let out = matrices[0]!;
    for (let i = 1; i < matrices.length; i++) out = out.mm(matrices[i]!);
    return out;
  }

  /**
   * The old name for the outer product. The same as `outer`.
   */
  ger(other: Tensor): Tensor {
    return this.outer(other);
  }

  /**
   * Folds and multiplies **along the last axes.** One-dimensional it is the
   * inner product; above that it is `a @ bᵀ`.
   *
   * Two dimensions is where it parts from `dot` — `dot` takes one dimension
   * only. This name was missing for a long time without surfacing, because
   * the check sweeping the source with a regular expression was counting
   * the local variable `const inner = …` as a public name.
   */
  inner(other: Tensor): Tensor {
    if (this.shape.length <= 1) return this.mul(other).sum();
    return this.mm(other.swapaxes(-2, -1));
  }

  /**
   * Matrix × vector. The work `mm` does, but torch gives it a name of its
   * own.
   *
   * It stands the vector up as a single column and then removes that axis
   * again — the result has to be one-dimensional.
   */
  mv(vec: Tensor): Tensor {
    return this.mm(vec.reshape([vec.size, 1])).reshape([this.shape[0] ?? 0]);
  }

  asStrided_(
    size: readonly number[],
    stride: readonly number[],
    storageOffset = 0,
  ): Tensor {
    return this.mutate(() => this.asStrided(size, stride, storageOffset));
  }

  maskedScatter_(mask: Tensor, source: Tensor): Tensor {
    return this.mutate(() => this.maskedScatter(mask, source));
  }

  indexPut_(
    indices: readonly Tensor[],
    values: Tensor,
    accumulate = false,
  ): Tensor {
    return this.mutate(() => this.indexPut(indices, values, accumulate));
  }

  // ── 합성곱·풀링 ───────────────────────────────────────────────────────

  /**
   * Two-dimensional convolution. `this` is `(N, C, H, W)` and the kernel
   * `(O, C, KH, KW)` — **NCHW.**
   *
   * The sister project carried NHWC around, and that was because TF.js's
   * conv was fast only in that layout. Here we write the kernels ourselves,
   * so torch's layout is used as-is.
   */
  conv2d(weight: Tensor, bias: Tensor | null = null, stride = 1, padding = 0): Tensor {
    if (this.shape.length !== 4 || weight.shape.length !== 4) {
      throw new Error(`conv2d is 4-D by 4-D: [${this.shape}] x [${weight.shape}]`);
    }
    return this.convND(weight, bias, stride, padding);
  }

  /**
   * One-dimensional convolution. `(N, C, L)`.
   */
  conv1d(weight: Tensor, bias: Tensor | null = null, stride = 1, padding = 0): Tensor {
    if (this.shape.length !== 3 || weight.shape.length !== 3) {
      throw new Error(`conv1d is 3-D by 3-D: [${this.shape}] x [${weight.shape}]`);
    }
    return this.convND(weight, bias, stride, padding);
  }

  /**
   * Three-dimensional convolution. `(N, C, D, H, W)`.
   */
  conv3d(weight: Tensor, bias: Tensor | null = null, stride = 1, padding = 0): Tensor {
    if (this.shape.length !== 5 || weight.shape.length !== 5) {
      throw new Error(`conv3d is 5-D by 5-D: [${this.shape}] x [${weight.shape}]`);
    }
    return this.convND(weight, bias, stride, padding);
  }

  maxPool1d(kernel = 2, stride?: number): Tensor {
    return this.poolND("max", kernel, stride);
  }

  maxPool3d(kernel = 2, stride?: number): Tensor {
    return this.poolND("max", kernel, stride);
  }

  // ── 이긴 자리를 함께 내는 풀링 ─────────────────────────────────────────
  //
  // 최대 풀링은 창마다 하나만 남기고 나머지를 버린다. **값 안에 "어느 칸이 이겼는가"
  // 가 없어서** `maxUnpool` 은 값만으로는 못 돌아간다. torch 는 풀링에게 자리표를
  // 같이 내게 하고 그것을 되돌리기에 넘긴다 — 자동 부호기에서 흔한 짝이다.

  /** 고정 창의 창 목록. 축마다 `[시작, 끝)`. */
  private fixedWindows(kernel: number, stride?: number): [number, number][][] {
    const step = stride ?? kernel;
    return this.shape.slice(2).map((n) => {
      const out: [number, number][] = [];
      for (let s = 0; s + kernel <= n; s += step) out.push([s, s + kernel]);
      return out;
    });
  }

  /** 적응형의 창 목록. 시작은 내림, 끝은 올림 — 자리마다 길이가 다르다. */
  private adaptiveWindows(outSize: number | readonly number[]): [number, number][][] {
    const spatial = this.shape.length - 2;
    const sizes = typeof outSize === "number"
      ? new Array<number>(spatial).fill(outSize)
      : [...outSize];
    return this.shape.slice(2).map((n, k) => {
      const want = sizes[k] ?? 1;
      const out: [number, number][] = [];
      for (let i = 0; i < want; i++) {
        out.push([Math.floor((i * n) / want), Math.ceil(((i + 1) * n) / want)]);
      }
      return out;
    });
  }

  private maxWithIndex(axes: [number, number][][]): {
    values: Tensor;
    indices: Tensor;
  } {
    const NC = (this.shape[0] ?? 1) * (this.shape[1] ?? 1);
    const inDims = this.shape.slice(2);
    const p: PoolWindows = { NC, inDims, axes };
    const outDims = axes.map((a) => a.length);
    const outShape = [this.shape[0] ?? 1, this.shape[1] ?? 1, ...outDims];
    const n = outShape.reduce((a, b) => a * b, 1);
    const key = poolWindowsKey(p);
    const out = dev().alloc(n);
    const idx = dev().alloc(n);
    dev().run1d(
      dev().pipeline(`pmi:${key}`, () => poolMaxWithIndex(p)),
      [this.buffer, out, idx],
      n,
    );
    const indices = new Tensor(idx, outShape, { dtype: "int64" });
    const shape = this.shape;
    const size = this.size;
    const values = Tensor.make(
      out,
      outShape,
      [this],
      (g) => {
        const gi = dev().alloc(size);
        dev().run1d(
          dev().pipeline(`pmib:${key}`, () => poolMaxIndexBackward(p)),
          [idx, g.buffer, gi],
          size,
        );
        return [new Tensor(gi, shape)];
      },
      "MaxPoolWithIndicesBackward0",
      this.dtype,
    );
    return { values, indices };
  }

  maxPoolWithIndices(kernel = 2, stride?: number): {
    values: Tensor;
    indices: Tensor;
  } {
    return this.maxWithIndex(this.fixedWindows(kernel, stride));
  }

  adaptiveMaxPoolWithIndices(outSize: number | readonly number[]): {
    values: Tensor;
    indices: Tensor;
  } {
    return this.maxWithIndex(this.adaptiveWindows(outSize));
  }

  // ── CTC ────────────────────────────────────────────────────────────────
  //
  // 소리와 글자를 **자리를 맞추지 않고** 잇는 손실. 가능한 정렬을 전부 더하는데 그
  // 수가 지수라, 표적 사이에 공백을 끼운 상태열을 두고 앞으로 훑어 접는다.
  //
  // `u` 축은 한 번에 민다. 시간만 돌므로 그래프가 `T` 에 비례한다 — 진짜 음성 길이
  // (수백 프레임)에서는 느리고, 정확한 쪽을 골랐다.

  /** 로그 확률의 "없음". `-Infinity` 는 logsumexp 에서 NaN 이 되므로 큰 음수를 쓴다. */
  private static readonly CTC_NEG = -1e30;

  private static ctcGap(n: number): Tensor {
    return Tensor.full([n], Tensor.CTC_NEG);
  }

  /**
   * One sample's `-log P(target | audio)`.
   *
   * @param lp `(T, C)` log probabilities.
   * @param labels the target tokens.
   */
  static ctcOne(
    lp: Tensor, labels: readonly number[], nTime: number, blank: number,
  ): Tensor {
    // `[l1, l2]` → `[_, l1, _, l2, _]`. **같은 글자가 이어지면 사이에 공백이 반드시
    // 있어야 한다** — 없으면 두 글자가 한 글자로 접힌다. 그 규칙이 `skip` 이다.
    const ext: number[] = [blank];
    for (const lab of labels) ext.push(lab, blank);
    const u = ext.length;

    const idx = Tensor.from(ext, [u], { dtype: "int64" });
    const skip = Tensor.from(
      ext.map((_, s) =>
        (s >= 2 && ext[s] !== blank && ext[s] !== ext[s - 2]) ? 0 : Tensor.CTC_NEG),
      [u]);
    const emit = lp.indexSelect(1, idx);          // (T, U)

    const head = Math.min(2, u);
    let alpha = emit.narrow(0, 0, 1).reshape([emit.shape[1] ?? u]).narrow(0, 0, head);
    if (u > head) alpha = Tensor.cat([alpha, Tensor.ctcGap(u - head)], 0);

    for (let t = 1; t < nTime; t++) {
      const same = alpha;
      const one = u > 1
        ? Tensor.cat([Tensor.ctcGap(1), alpha.narrow(0, 0, u - 1)], 0)
        : Tensor.ctcGap(1);
      const shifted = u > 2
        ? Tensor.cat([Tensor.ctcGap(2), alpha.narrow(0, 0, u - 2)], 0)
        : Tensor.ctcGap(u);
      const two = shifted.add(skip);
      const step = emit.narrow(0, t, 1).reshape([u]);
      alpha = Tensor.stack([same, one, two], 0).logsumexp(0).add(step);
    }
    const tail = u >= 2 ? alpha.narrow(0, u - 2, 2) : alpha;
    return tail.logsumexp(0).neg();
  }

  /**
   * 창 시작 자리들. **ATen 의 `generate_intervals` 그대로다.**
   *
   * `α = (입력 - 창) / (출력 - 1)` 에 `floor((i+u)·α) - floor(u·α)`. 마지막 창만
   * 오른쪽 끝에 붙여서 입력의 마지막 칸이 반드시 덮이게 한다.
   *
   * 나누어떨어지면 `α` 가 정수라 `u` 가 아무 일도 안 한다 — 6→3 으로 물으면 무작위
   * 부분이 통째로 안 보인다.
   */
  private static fractionalStarts(
    nIn: number, k: number, nOut: number, u: number,
  ): number[] {
    if (nOut <= 1) return [0];
    const alpha = (nIn - k) / (nOut - 1);
    const seq: number[] = [];
    for (let i = 0; i < nOut - 1; i++) {
      seq.push(Math.trunc((i + u) * alpha) - Math.trunc(u * alpha));
    }
    seq.push(nIn - k);
    return seq;
  }

  /**
   * Max pooling whose window positions are shaken by a sample.
   *
   * **The sample differs per plane** — torch's `_random_samples` is `(N, C,
   * axis)`, so the windows diverge per plane. Hence it loops over the
   * planes. Expensive, and the same as torch.
   *
   * **The axis order differs by dimensionality.** ATen's 2-D version reads
   * the sample as (width, height) and the 3-D version as (depth, height,
   * width) — the two functions are out of step with each other, and what is
   * imitated here is that inconsistency.
   *
   * @param samples the sample per plane. It is `samples[plane][i]`, and `i`
   *   follows the order above.
   */
  fractionalMaxPool(
    kernel: number,
    outDims: readonly number[],
    samples: readonly (readonly number[])[],
  ): { values: Tensor; indices: Tensor } {
    const spatial = this.shape.length - 2;
    const N = this.shape[0] ?? 1;
    const C = this.shape[1] ?? 1;
    const order = spatial === 3
      ? [0, 1, 2]
      : Array.from({ length: spatial }, (_, k) => spatial - 1 - k);
    const values: Tensor[] = [];
    const indices: Tensor[] = [];
    for (let p = 0; p < N * C; p++) {
      const plane = this.narrow(0, Math.floor(p / C), 1)
        .narrow(1, p % C, 1);
      const axes: [number, number][][] = [];
      for (let k = 0; k < spatial; k++) {
        const u = samples[p]?.[order[k] ?? k] ?? 0;
        const starts = Tensor.fractionalStarts(
          this.shape[2 + k] ?? 1, kernel, outDims[k] ?? 1, u,
        );
        axes.push(starts.map((s) => [s, s + kernel] as [number, number]));
      }
      const got = plane.maxWithIndex(axes);
      values.push(got.values);
      indices.push(got.indices);
    }
    const outShape = [N, C, ...outDims];
    return {
      values: Tensor.cat(values, 0).reshape(outShape),
      indices: Tensor.cat(indices, 0).reshape(outShape),
    };
  }

  /**
   * Places values at the slots the indices point at and leaves the rest
   * zero.
   *
   * What pooling discarded cannot be brought back, so the output size is
   * given from outside. The default is `(n-1)·stride - 2·padding + kernel`,
   * and `outSize` can name it directly.
   */
  maxUnpool(
    indices: Tensor,
    kernel: number,
    stride?: number,
    padding = 0,
    outSize?: readonly number[],
  ): Tensor {
    const step = stride ?? kernel;
    const inDims = this.shape.slice(2);
    const outDims = outSize
      ? [...outSize].slice(-inDims.length)
      : inDims.map((d) => (d - 1) * step - 2 * padding + kernel);
    const NC = (this.shape[0] ?? 1) * (this.shape[1] ?? 1);
    const inSpace = inDims.reduce((a, b) => a * b, 1);
    const outSpace = outDims.reduce((a, b) => a * b, 1);
    const outShape = [this.shape[0] ?? 1, this.shape[1] ?? 1, ...outDims];
    const key = `${NC}:${inSpace}:${outSpace}`;
    const out = dev().alloc(NC * outSpace);
    dev().run1d(
      dev().pipeline(`unp:${key}`, () => unpoolFromIndex(NC, inSpace, outSpace)),
      [this.buffer, indices.buffer, out],
      NC * outSpace,
    );
    const shape = this.shape;
    const size = this.size;
    return Tensor.make(
      out,
      outShape,
      [this],
      (g) => {
        const gi = dev().alloc(size);
        dev().run1d(
          dev().pipeline(`unpb:${key}`,
            () => unpoolFromIndexBackward(NC, inSpace, outSpace)),
          [indices.buffer, g.buffer, gi],
          size,
        );
        return [new Tensor(gi, shape)];
      },
      "MaxUnpoolBackward0",
    );
  }

  /**
   * Convolution independent of dimensionality. `conv1d`, `conv2d` and
   * `conv3d` all come here.
   *
   * One spatial axis is 1-D and two is 2-D — a separate function per
   * dimensionality makes three copies, and a day comes when only one of
   * them is fixed. The sister project was in exactly that state.
   */
  convND(
    weight: Tensor,
    bias: Tensor | null = null,
    stride: number | readonly number[] = 1,
    padding: number | readonly number[] = 0,
  ): Tensor {
    const spatial = this.shape.length - 2;
    if (spatial < 1 || weight.shape.length !== this.shape.length) {
      throw new Error(`conv: shapes do not match: [${this.shape}] x [${weight.shape}]`);
    }
    const spread = (v: number | readonly number[]): number[] =>
      typeof v === "number" ? new Array<number>(spatial).fill(v) : [...v];
    const inDims = this.shape.slice(2);
    const kernel = weight.shape.slice(2);
    const st = spread(stride);
    const pd = spread(padding);
    const C = this.shape[1] ?? 1;
    const WC = weight.shape[1] ?? 1;
    if (C !== WC) {
      throw new RuntimeError(
        `Given groups=1, weight of size [${weight.shape}], expected input` +
          `[${this.shape}] to have ${WC} channels, but got ${C} channels instead`,
      );
    }
    const s: ConvNDShape = {
      N: this.shape[0] ?? 1, C, O: weight.shape[0] ?? 1,
      inDims, kernel, stride: st, pad: pd,
      outDims: inDims.map((d, i) =>
        convOut(d, pd[i] ?? 0, kernel[i] ?? 1, st[i] ?? 1)),
    };
    const key = convNDKey(s);
    const outShape = [s.N, s.O, ...s.outDims];
    const n = outShape.reduce((a, b) => a * b, 1);
    const out = dev().alloc(n);
    // 타일링 판을 쓴다. 단순 판보다 셰이더가 길고 컴파일이 한 번 더 들지만, 모양
    // 서명으로 캐시되므로 그것은 한 번이고 스텝마다 도는 것은 커널 쪽이다.
    dev().run(
      dev().pipeline(`cnt:${key}:${bias ? "b" : "n"}`,
        () => convNDForwardTiled(s, bias !== null)),
      bias ? [this.buffer, weight.buffer, bias.buffer, out]
        : [this.buffer, weight.buffer, out],
      convTiledGrid(s),
    );
    const parents = bias ? [this, weight, bias] : [this, weight];
    return Tensor.make(
      out,
      outShape,
      parents,
      (g) => {
        const parts: (Tensor | null)[] = [];
        if (this.requiresGrad) {
          const gi = dev().alloc(this.size);
          dev().run(
            dev().pipeline(`cnxt:${key}`, () => convNDGradInputTiled(s)),
            [g.buffer, weight.buffer, gi],
            convGradInputGrid(s),
          );
          parts.push(new Tensor(gi, this.shape));
        } else parts.push(null);
        if (weight.requiresGrad) {
          // 축약을 쪼갰으면 부분합이 조각 수만큼 나오고, 한 번 더 더해야 한다.
          const splits = convGradWeightSplit(s);
          const parted = dev().alloc(weight.size * splits);
          dev().run(
            dev().pipeline(`cnwt:${key}`, () => convNDGradWeightTiled(s)),
            [this.buffer, g.buffer, parted],
            convGradWeightGrid(s),
          );
          let gw = parted;
          if (splits > 1) {
            gw = dev().alloc(weight.size);
            dev().run1d(
              dev().pipeline(`ss:${weight.size}:${splits}`,
                () => sumSplits(weight.size, splits)),
              [parted, gw],
              weight.size,
            );
          }
          parts.push(new Tensor(gw, weight.shape));
        } else parts.push(null);
        if (bias) {
          // 배치와 출력 자리를 전부 합친 것. 축약을 겹쳐 쓰면 새 커널이 없다.
          let acc = g.sumDim(0);
          for (let d = 0; d < spatial; d++) acc = acc.sumDim(1);
          parts.push(bias.requiresGrad ? acc : null);
        }
        return parts;
      },
      "ConvolutionBackward0",
    );
  }

  /**
   * Transposed convolution independent of dimensionality.
   *
   * **There is no new kernel.** A transposed convolution's forward is the
   * same computation as an ordinary convolution's flow towards its input,
   * so `convNDGradInputTiled` is used as-is. The backward flips likewise —
   * the input-side gradient is an ordinary convolution's forward, and the
   * weight side is the same kernel with two arguments exchanged. The same
   * computation in two copies means a day when one of them is fixed.
   *
   * **The weight axes are flipped relative to `convND`** — `(in, out, …)`.
   * With a square kernel the shape still matches when flipped, so only the
   * values diverge.
   */
  convTransposeND(
    weight: Tensor,
    bias: Tensor | null = null,
    stride: number | readonly number[] = 1,
    padding: number | readonly number[] = 0,
  ): Tensor {
    const spatial = this.shape.length - 2;
    if (spatial < 1 || weight.shape.length !== this.shape.length) {
      throw new Error(
        `convTranspose: shapes do not match: [${this.shape}] x [${weight.shape}]`);
    }
    const spread = (v: number | readonly number[]): number[] =>
      typeof v === "number" ? new Array<number>(spatial).fill(v) : [...v];
    const st = spread(stride);
    const pd = spread(padding);
    const Cin = this.shape[1] ?? 1;
    const Cout = weight.shape[1] ?? 1;
    if ((weight.shape[0] ?? 1) !== Cin) {
      throw new RuntimeError(
        `Given transposed=1, weight of size [${weight.shape}], expected input` +
          `[${this.shape}] to have ${weight.shape[0]} channels, but got ${Cin} channels instead`,
      );
    }
    const kernel = weight.shape.slice(2);
    const ourDims = this.shape.slice(2);
    // 보통 합성곱의 눈으로 본다: 우리 입력이 그쪽의 **출력**이고, 우리 출력이
    // 그쪽의 입력이다. 그래서 O 와 C 가 뒤바뀐 자리에 들어간다.
    const outDims = ourDims.map((d, i) =>
      (d - 1) * (st[i] ?? 1) + (kernel[i] ?? 1) - 2 * (pd[i] ?? 0));
    const s: ConvNDShape = {
      N: this.shape[0] ?? 1, C: Cout, O: Cin,
      inDims: outDims, kernel, stride: st, pad: pd, outDims: ourDims,
    };
    const key = convNDKey(s);
    const outShape = [s.N, Cout, ...outDims];
    const out = dev().alloc(outShape.reduce((a, b) => a * b, 1));
    dev().run(
      dev().pipeline(`cnxt:${key}`, () => convNDGradInputTiled(s)),
      [this.buffer, weight.buffer, out],
      convGradInputGrid(s),
    );
    let result = Tensor.make(
      out,
      outShape,
      [this, weight],
      (g) => {
        const parts: (Tensor | null)[] = [];
        if (this.requiresGrad) {
          // 우리 입력 쪽 기울기는 **보통 합성곱의 순방향**이다.
          const gi = dev().alloc(this.size);
          dev().run(
            dev().pipeline(`cnt:${key}:n`, () => convNDForwardTiled(s, false)),
            [g.buffer, weight.buffer, gi],
            convTiledGrid(s),
          );
          parts.push(new Tensor(gi, this.shape));
        } else parts.push(null);
        if (weight.requiresGrad) {
          const splits = convGradWeightSplit(s);
          const parted = dev().alloc(weight.size * splits);
          // 두 인자를 바꿔 넣는다 — 그쪽의 "입력" 이 우리 기울기이고 그쪽의
          // "출력 기울기" 가 우리 입력이다.
          dev().run(
            dev().pipeline(`cnwt:${key}`, () => convNDGradWeightTiled(s)),
            [g.buffer, this.buffer, parted],
            convGradWeightGrid(s),
          );
          let gw = parted;
          if (splits > 1) {
            gw = dev().alloc(weight.size);
            dev().run1d(
              dev().pipeline(`ss:${weight.size}:${splits}`,
                () => sumSplits(weight.size, splits)),
              [parted, gw],
              weight.size,
            );
          }
          parts.push(new Tensor(gw, weight.shape));
        } else parts.push(null);
        return parts;
      },
      "ConvTransposeBackward0",
    );
    if (bias) {
      const shape = [1, Cout, ...new Array<number>(spatial).fill(1)];
      result = result.add(bias.reshape(shape));
    }
    return result;
  }

  /**
   * Pooling independent of dimensionality.
   */
  poolND(kind: "max" | "avg", kernel: number, stride?: number): Tensor {
    const spatial = this.shape.length - 2;
    if (spatial < 1) throw new Error(`pooling: the shape does not match: [${this.shape}]`);
    const step = stride ?? kernel;
    const inDims = this.shape.slice(2);
    const p: PoolNDShape = {
      // 채널을 배치에 접어 넣는다 — 풀링은 평면마다 따로 도는 일이다.
      NC: (this.shape[0] ?? 1) * (this.shape[1] ?? 1),
      inDims,
      kernel: new Array<number>(spatial).fill(kernel),
      stride: new Array<number>(spatial).fill(step),
      outDims: inDims.map((d) => convOut(d, 0, kernel, step)),
    };
    const key = poolNDKey(p);
    const outShape = [this.shape[0] ?? 1, this.shape[1] ?? 1, ...p.outDims];
    const n = outShape.reduce((a, b) => a * b, 1);
    const out = dev().alloc(n);
    dev().run1d(
      dev().pipeline(`pn:${kind}:${key}`, () => poolNDForward(p, kind)),
      [this.buffer, out],
      n,
    );
    const shape = this.shape;
    return Tensor.make(
      out,
      outShape,
      [this],
      (g) => {
        const gi = dev().alloc(this.size);
        // 평균 풀링은 입력을 안 보므로 버퍼도 안 받는다 — 안 쓰는 바인딩을 넘기면
        // WebGPU 가 명령 버퍼를 통째로 무효로 만들고, 그때 역방향이 조용히 안 돈다.
        dev().run1d(
          dev().pipeline(`pnb:${kind}:${key}`, () => poolNDBackward(p, kind)),
          poolNDBackwardNeedsInput(kind)
            ? [this.buffer, g.buffer, gi]
            : [g.buffer, gi],
          this.size,
        );
        return [new Tensor(gi, shape)];
      },
      kind === "max" ? "MaxPoolNDBackward0" : "AvgPoolNDBackward0",
    );
  }

  /**
   * Enlargement taking `size`, `scaleFactor` and `mode` in one place. The
   * `Upsample` layer is this.
   *
   * **Taking `mode` and then not using it is the reason this function
   * exists.** Nearest and bilinear are entirely different kernels, and
   * without dispatching on it, asking for bilinear gives you nearest — not
   * an exception, a quietly different value.
   */
  interpolate(size: number | readonly number[] | null = null,
              scaleFactor: number | null = null,
              mode: "nearest" | "bilinear" = "nearest",
              alignCorners = false): Tensor {
    const h = this.shape[2] ?? 1;
    const w = this.shape[3] ?? 1;
    const pair = (v: number | readonly number[]): [number, number] =>
      typeof v === "number" ? [v, v] : [v[0] ?? 1, v[1] ?? v[0] ?? 1];
    if (mode === "nearest") {
      if (size === null) return this.upsample(scaleFactor ?? 2);
      // 저쪽 커널은 **배수만** 받는다. 배수가 아니면 조용히 근사하지 않고 멈춘다.
      const [oh, ow] = pair(size);
      if (oh % h || ow % w || oh / h !== ow / w) {
        throw new RuntimeError("interpolate(size=) — nearest upsampling by a non-integer factor");
      }
      return this.upsample(oh / h);
    }
    const [oh, ow] = size === null
      ? [h * (scaleFactor ?? 2), w * (scaleFactor ?? 2)] : pair(size);
    return this.interpolateBilinear(oh, ow, alignCorners);
  }

  /**
   * Nearest-neighbour enlargement. `Upsample` and `interpolate` are this.
   */
  upsample(scale: number): Tensor {
    const spatial = this.shape.length - 2;
    const inDims = this.shape.slice(2);
    const NC = (this.shape[0] ?? 1) * (this.shape[1] ?? 1);
    const outShape = [
      this.shape[0] ?? 1, this.shape[1] ?? 1, ...inDims.map((d) => d * scale),
    ];
    const n = outShape.reduce((a, b) => a * b, 1);
    const key = `${NC}:${inDims}:${scale}`;
    const out = dev().alloc(n);
    dev().run1d(
      dev().pipeline(`up:${key}`, () => upsampleNearest(NC, inDims, scale)),
      [this.buffer, out],
      n,
    );
    void spatial;
    const shape = this.shape;
    return Tensor.make(
      out,
      outShape,
      [this],
      (g) => {
        const gi = dev().alloc(this.size);
        dev().run1d(
          dev().pipeline(`upb:${key}`,
            () => upsampleNearestBackward(NC, inDims, scale)),
          [g.buffer, gi],
          this.size,
        );
        return [new Tensor(gi, shape)];
      },
      "UpsampleNearestBackward0",
    );
  }

  /**
   * Folds to a fixed output size. The window size comes out of the input.
   *
   * **It does not have to divide evenly.** It used to refuse, and torch
   * takes a different window per position — the refusal was not a form of
   * imitation but a different rule. The start floors and the end ceils, so
   * reducing 8 to 3 gives windows of 3·3·2.
   *
   * It folds one axis at a time. The window is rectangular, so doing it per
   * axis gives the same value — each row has the same length, which makes
   * the mean of means the overall mean, and the maximum is that way anyway.
   *
   * No dedicated kernel is used. With a window that differs per position
   * there is nothing to bake into the shader as a constant, and adaptive
   * pooling is usually a single final call rather than something running
   * per step.
   */
  adaptivePool(kind: "max" | "avg", outSize: number | readonly number[]): Tensor {
    const spatial = this.shape.length - 2;
    const sizes = typeof outSize === "number"
      ? new Array<number>(spatial).fill(outSize) : [...outSize];
    let out: Tensor = this;
    for (let k = 0; k < spatial; k++) {
      const axis = 2 + k;
      const n = out.shape[axis] ?? 1;
      const want = sizes[k] ?? 1;
      const parts: Tensor[] = [];
      for (let i = 0; i < want; i++) {
        const start = Math.floor((i * n) / want);
        const end = Math.ceil(((i + 1) * n) / want);
        const window = out.narrow(axis, start, end - start);
        parts.push(kind === "avg" ? window.mean(axis, true) : window.amax(axis, true));
      }
      out = Tensor.cat(parts, axis);
    }
    return out;
  }

  /**
   * The old name. The same as `adaptivePool("avg", …)` — kept because there
   * are callers.
   */
  adaptiveAvgPool(outSize: number): Tensor {
    return this.adaptivePool("avg", outSize);
  }

  /**
   * The `p`-th root of the sum of `p`-th powers.
   *
   * **It follows torch's assembly exactly** — average pooling, multiplied
   * by the window size to get back to a sum, then the root. The sign
   * handling and the `relu` in the middle are that implementation's too.
   */
  lpPool(normType: number, kernel: number, stride?: number): Tensor {
    const spatial = this.shape.length - 2;
    const count = kernel ** spatial;
    const powered = this.powScalar(normType);
    const out = powered.poolND("avg", kernel, stride ?? kernel);
    const signed = out.unary("sign").mul(out.abs().unary("relu"));
    return signed.mul(Tensor.full([], count)).powScalar(1 / normType);
  }

  /**
   * The maximum of non-overlapping windows. `this` is `(N, C, H, W)`.
   */
  maxPool2d(kernel = 2, stride?: number): Tensor {
    return this.pool2d("max", kernel, stride ?? kernel);
  }

  avgPool2d(kernel = 2, stride?: number): Tensor {
    return this.pool2d("avg", kernel, stride ?? kernel);
  }

  private pool2d(kind: "max" | "avg", kernel: number, stride: number): Tensor {
    if (this.shape.length !== 4) {
      throw new Error(`pooling is 4-D: [${this.shape}]`);
    }
    return this.poolND(kind, kernel, stride);
  }

  /**
   * `(N, C, H, W)` 를 채널마다 정규화한다. 학습 모드 — 이 배치로 통계를 센다.
   *
   * 축 셋을 한꺼번에 접어야 해서 `layerNorm` 을 못 쓴다. 축약을 겹쳐 쓰면 새 커널이
   * 필요 없다 — 대신 중간 텐서가 몇 개 생기고, 그게 지금 치르는 값이다.
   */
  /**
   * Fused batch normalisation. Statistics, normalisation, scale and shift
   * in three kernels.
   *
   * The assembled version cost over twenty dispatches for one layer, and
   * most of a ResNet step's 1,636 came from there (measured). Here it is
   * two forward and two backward.
   *
   * @returns the normalised result, and the mean and variance for updating
   *   the running statistics.
   */
  batchNormFused(
    weight: Tensor, bias: Tensor, eps = 1e-5,
  ): { out: Tensor; mean: Tensor; variance: Tensor } {
    const [N = 1, C = 1] = this.shape;
    const S = this.shape.slice(2).reduce((a, b) => a * b, 1);
    const key = `${N}:${C}:${S}`;
    const mean = dev().alloc(C);
    const variance = dev().alloc(C);
    dev().run1d(
      dev().pipeline(`bns:${key}`, () => batchNormStats(N, C, S)),
      [this.buffer, mean, variance],
      C,
    );
    const out = dev().alloc(this.size);
    dev().run1d(
      dev().pipeline(`bna:${key}:${eps}`, () => batchNormApply(N, C, S, eps)),
      [this.buffer, mean, variance, weight.buffer, bias.buffer, out],
      this.size,
    );
    // 표준화된 값은 역방향이 다시 쓴다. 여기서 한 번 만들어 들고 간다 —
    // 역방향에서 다시 세면 커널이 둘 더 는다.
    const meanT = new Tensor(mean, [C]);
    const varT = new Tensor(variance, [C]);
    const invStd = varT.binary("add", Tensor.full([], eps)).unary("rsqrt");
    const shape4 = [1, C, ...new Array<number>(this.shape.length - 2).fill(1)];
    const xhat = this.sub(meanT.reshape(shape4)).mul(invStd.reshape(shape4));
    const self = this;
    const result = Tensor.make(
      out,
      this.shape,
      [this, weight, bias],
      (g) => {
        const sumG = dev().alloc(C);
        const sumGXh = dev().alloc(C);
        dev().run1d(
          dev().pipeline(`bnsb:${key}`, () => batchNormStatsBackward(N, C, S)),
          [xhat.buffer, g.buffer, sumG, sumGXh],
          C,
        );
        const parts: (Tensor | null)[] = [];
        if (self.requiresGrad) {
          const gi = dev().alloc(self.size);
          dev().run1d(
            dev().pipeline(`bnba:${key}`, () => batchNormBackwardApply(N, C, S)),
            [xhat.buffer, g.buffer, sumG, sumGXh, weight.buffer, invStd.buffer, gi],
            self.size,
          );
          parts.push(new Tensor(gi, self.shape));
        } else parts.push(null);
        // 가중치·치우침의 기울기는 이미 센 두 합이다 — 새 커널이 필요 없다.
        parts.push(weight.requiresGrad ? new Tensor(sumGXh, [C]) : null);
        parts.push(bias.requiresGrad ? new Tensor(sumG, [C]) : null);
        return parts;
      },
      "NativeBatchNormBackward0",
    );
    return { out: result, mean: meanT, variance: varT };
  }

  batchNorm2d(eps = 1e-5): Tensor {
    const [N = 1, C = 1, H = 1, W = 1] = this.shape;
    const count = N * H * W;
    const perChannel = (t: Tensor): Tensor =>
      t.sumDim(0).sumDim(1).sumDim(1).reshape([1, C, 1, 1]);
    const mean = perChannel(this).div(Tensor.full([], count));
    const centered = this.sub(mean);
    // 분산은 편향추정(n 으로 나눔)이다 — torch 의 BatchNorm 이 그렇다.
    const varc = perChannel(centered.square()).div(Tensor.full([], count));
    return centered.div(varc.binary("add", Tensor.full([], eps)).sqrt());
  }

  // ── 복소수 ────────────────────────────────────────────────────────────
  //
  // 저장은 **인터리브** 다 — `[re, im, re, im, …]` 한 버퍼. 그래서 `viewAsReal` 과
  // `viewAsComplex` 가 **진짜 뷰**가 된다(버퍼를 그대로 들고 이름표와 모양만 바꾼다).
  // torch 도 그 둘이 뷰다. 실수부와 허수부를 텐서 둘로 나눠 들었다면 여기서 복사가
  // 났을 것이고, `Tensor` 가 버퍼를 둘 들어야 해서 수명·장치 경로가 전부 그것을
  // 배워야 했다. 버퍼 하나라는 불변식을 지키는 대신 **길이**만 두 배가 된다.
  //
  // ## 기울기 규약 (재서 못 박은 것)
  //
  // torch 는 복소 손실에 `backward()` 를 거절한다. 손실이 늘 실수이므로
  //
  //     z.grad = ∂L/∂re + i·∂L/∂im
  //
  // 이 잘 정의되고, 그 위에서 **정칙 함수의 역방향에 켤레가 붙는다** —
  // `mul`·`div` 가 그 자리다. 실수를 내는 `abs` 는 정칙이 아니라 안 붙고(`z/|z|`),
  // `conj` 자신은 `conj(g)` 다. 셋이 다른 규칙이고 실수 입력으로는 셋 다 구분이
  // 안 된다 — 켤레가 실수에서 항등이기 때문이다.

  /**
   * Whether this tensor is complex. Where `torch.is_complex` goes.
   */
  isComplex(): boolean {
    return isComplexDType(this.dtype);
  }

  /** 복소수 커널 하나. 입력들을 받아 `outFloats` 칸짜리 버퍼를 낸다. */
  private static cRun(
    key: string,
    source: () => string,
    inputs: readonly GPUBuffer[],
    outFloats: number,
    threads: number,
  ): GPUBuffer {
    const out = dev().alloc(outFloats);
    dev().run1d(dev().pipeline(key, source), [...inputs, out], threads);
    return out;
  }

  /**
   * Weaves a real and an imaginary part together. Where `torch.complex`
   * goes.
   *
   * The backward **splits the incoming complex gradient as it is** — the
   * convention *is* `(∂L/∂re, ∂L/∂im)`, so there is nothing to do but take
   * it apart. This place is the definition of the convention.
   */
  static complex(re: Tensor, im: Tensor): Tensor {
    if (re.shape.length !== im.shape.length
      || re.shape.some((d, i) => d !== im.shape[i])) {
      throw new RuntimeError(
        `complex requires matching shapes: [${re.shape}] vs [${im.shape}]`,
      );
    }
    const n = re.size;
    const out = Tensor.cRun(`cpack:${n}`, () => complexPack(n),
      [re.buffer, im.buffer], 2 * n, n);
    return Tensor.make(
      out, re.shape, [re, im],
      (g) => [
        re.requiresGrad ? g.real() : null,
        im.requiresGrad ? g.imag() : null,
      ],
      "ComplexBackward0", "complex64",
    );
  }

  /**
   * Builds from magnitude and angle. Where `torch.polar` goes.
   */
  static polar(abs: Tensor, angle: Tensor): Tensor {
    if (abs.shape.length !== angle.shape.length
      || abs.shape.some((d, i) => d !== angle.shape[i])) {
      throw new RuntimeError(
        `polar requires matching shapes: [${abs.shape}] vs [${angle.shape}]`,
      );
    }
    const n = abs.size;
    const out = Tensor.cRun(`cpolar:${n}`, () => complexPolar(n),
      [abs.buffer, angle.buffer], 2 * n, n);
    return new Tensor(out, abs.shape, { dtype: "complex64" });
  }

  /**
   * **Views a complex as a pair of reals** — a 2 is appended to the shape.
   * The buffer is not moved.
   *
   * It would be fair to say the interleaved storage exists for this one
   * line. It is a view in torch too.
   */
  viewAsReal(): Tensor {
    if (!this.isComplex()) {
      throw new RuntimeError(
        "view_as_real is for complex tensors only — this one is " +
          `torch.${this.dtype}.`,
      );
    }
    const shape = [...this.shape, 2];
    const back = this.shape;
    return Tensor.make(
      this.raw, shape, [this],
      (g) => [new Tensor(g.raw, back, { dtype: "complex64" })],
      "ViewAsRealBackward0", "float32",
    );
  }

  /**
   * Views a pair of reals as a complex. The opposite of `viewAsReal`, and a
   * view likewise.
   */
  viewAsComplex(): Tensor {
    const last = this.shape[this.shape.length - 1];
    if (this.isComplex() || last !== 2) {
      throw new RuntimeError(
        "view_as_complex is for real tensors whose last dimension is 2: " +
          `[${this.shape}] (dtype torch.${this.dtype})`,
      );
    }
    const shape = this.shape.slice(0, -1);
    const back = this.shape;
    return Tensor.make(
      this.raw, shape, [this],
      (g) => [new Tensor(g.raw, back, { dtype: "float32" })],
      "ViewAsComplexBackward0", "complex64",
    );
  }

  /** 복소수 하나를 요구한다. 실수를 받으면 어느 이름이 틀렸는지 말한다. */
  private needComplex(what: string): void {
    if (!this.isComplex()) {
      throw new RuntimeError(
        `${what} is for complex tensors only — this one is torch.${this.dtype}.`,
      );
    }
  }

  /**
   * The real part. **Gradient returns as `g + 0i`.**
   *
   * In torch this is a strided view; here it is **a copy** — our tensors
   * carry only a shape and no strides, so there is no way to build a frame
   * that reads every other slot. The values are the same.
   */
  real(): Tensor {
    // **`real` 과 `imag` 는 규칙이 다르다.** torch 는 실수 텐서에 `real` 을 주면
    // 그것을 그대로 돌려주고(문서에 그렇게 적혀 있다), `imag` 는 거절한다 —
    // "imag is not implemented for tensors with non-complex dtypes".
    //
    // 여기서는 둘 다 거절하고 있었다. 나란히 선 두 이름을 같은 규칙으로 묶은 것인데
    // torch 가 그 둘을 다르게 두었고, 그래서 `torch.real(x)` 를 쓰는 코드가 실수
    // 텐서에서 멈췄다. 형도 그대로여야 한다 — int64 를 넣으면 int64 가 나온다.
    if (!this.isComplex()) return this;
    const n = this.size;
    const out = Tensor.cRun(`cpart:0:${n}`, () => complexPart(n, 0),
      [this.raw], n, n);
    return Tensor.make(
      out, this.shape, [this], (g) => [g.asComplexRe()],
      "RealBackward0", "float32",
    );
  }

  /**
   * The imaginary part. **The gradient is `0 + gi`** — written as `−gi` it
   * runs plausibly with only the sign flipped. Measuring torch, the
   * gradient of `z.imag` is `0+1j`.
   */
  imag(): Tensor {
    this.needComplex("imag");
    const n = this.size;
    const out = Tensor.cRun(`cpart:1:${n}`, () => complexPart(n, 1),
      [this.raw], n, n);
    return Tensor.make(
      out, this.shape, [this], (g) => [g.asComplexIm()],
      "ImagBackward0", "float32",
    );
  }

  /** 실수를 `x + 0i` 로 올린다. 역방향이 실수부만 도로 꺼낸다. */
  private asComplexRe(): Tensor {
    const n = this.size;
    const out = Tensor.cRun(`cfromre:${n}`, () => complexFromReal(n),
      [this.buffer], 2 * n, n);
    return Tensor.make(
      out, this.shape, [this], (g) => [g.real()],
      "ToComplexBackward0", "complex64",
    );
  }

  /** 실수를 `0 + xi` 로 올린다. */
  private asComplexIm(): Tensor {
    const n = this.size;
    const out = Tensor.cRun(`cfromim:${n}`, () => complexFromImag(n),
      [this.buffer], 2 * n, n);
    return Tensor.make(
      out, this.shape, [this], (g) => [g.imag()],
      "ToComplexBackward0", "complex64",
    );
  }

  /**
   * The conjugate. **Not holomorphic, so the backward is `conj(g)`** — not
   * of the form `conj(f')·g`.
   *
   * **A place it parts from torch.** torch's `conj` is lazy and raises a
   * conjugate bit without flipping the values (which is why `is_conj` is
   * true and `view_as_real` refuses). Ours flips immediately, so that state
   * does not exist at all — the values are the same.
   */
  conjPhysical(): Tensor {
    if (!this.isComplex()) return this;
    const n = this.size;
    const out = Tensor.cRun(`cconj:${n}`, () => complexConj(n), [this.raw],
      2 * n, n);
    return Tensor.make(
      out, this.shape, [this], (g) => [g.conjPhysical()],
      "ConjBackward0", "complex64",
    );
  }

  conj(): Tensor {
    return this.conjPhysical();
  }

  /**
   * Whether the conjugate bit is raised. **Here it is always false.**
   *
   * torch's `conj()` does not change the values and **raises one bit** —
   * the actual conjugate is made later when needed (hence the separate
   * `resolve_conj`). `conj` here makes the conjugate on the spot, so there
   * is nothing to defer and no bit to raise.
   *
   * **The name is kept anyway.** Without it, `x.isConj()` stops with "no
   * such thing", and that question does have an answer — false. Having no
   * lazy bit is an implementation's business, not a reason for the question
   * to lose its meaning.
   */
  isConj(): boolean {
    return false;
  }

  /**
   * Whether the sign bit is raised. Always false, for the same reason as
   * `isConj`.
   */
  isNeg(): boolean {
    return false;
  }

  /**
   * A deferred conjugate made real. **Here it is itself** — nothing was
   * deferred.
   *
   * In torch this repays the bit `conj()` raised. Our `conj` arrives
   * already repaid, so there is nothing to repay and this is the identity.
   * Being the identity and **being absent are different** — putting this
   * line in before handing a conjugate on is a common idiom in torch code.
   */
  resolveConj(): Tensor {
    return this;
  }

  /**
   * A deferred negation made real. The identity, for the same reason as
   * `resolveConj`.
   */
  resolveNeg(): Tensor {
    return this;
  }

  /**
   * 부호를 뒤집는다. **스칼라 −1 을 곱하지 않는다** — 복소수 이항은 모양이 같아야
   * 해서 스칼라가 못 들어오고, 모양대로 −1 텐서를 만드는 것은 버퍼 한 벌을 더 쓴다.
   */
  private complexNeg(): Tensor {
    const n = this.size;
    const out = Tensor.cRun(`cneg:${n}`, () => complexNeg(n), [this.raw],
      2 * n, n);
    return Tensor.make(
      out, this.shape, [this], (g) => [g.complexNeg()],
      "NegBackward0", "complex64",
    );
  }

  /**
   * The magnitude of a complex. **The result is real and the gradient is
   * `z/|z|`** — with **no** conjugate attached.
   *
   * The real `abs` is in the unary table and does not come here. `abs()`
   * looks at the type and dispatches — **that is why this is public.** The
   * dispatch happens outside the class (after the unary table is laid on),
   * so a private one could not be called.
   */
  complexAbs(): Tensor {
    const n = this.size;
    const out = Tensor.cRun(`cabs:${n}`, () => complexAbs(n), [this.raw], n, n);
    return Tensor.make(
      out, this.shape, [this],
      (g) => [new Tensor(
        Tensor.cRun(`cabsb:${n}`, () => complexAbsBackward(n),
          [this.raw, g.buffer], 2 * n, n),
        this.shape, { dtype: "complex64" },
      )],
      "AbsBackward0", "float32",
    );
  }

  /**
   * The argument. For a complex, `atan2(im, re)`; for a real, π on
   * negatives (the type is always real).
   */
  angle(): Tensor {
    if (!this.isComplex()) {
      return this.binary("lt", Tensor.full([], 0), "float32")
        .mul(Tensor.full([], Math.PI));
    }
    const n = this.size;
    const out = Tensor.cRun(`cangle:${n}`, () => complexAngle(n), [this.raw],
      n, n);
    return new Tensor(out, this.shape, { dtype: "float32" });
  }

  /**
   * 복소수 이항. **모양이 같아야 한다** — 브로드캐스팅은 아직 없다.
   *
   * 실수가 한쪽에 오면 `x + 0i` 로 올려서 들인다. 실수 커널을 그대로 쓰는 길도
   * 있지만(덧셈은 평평한 2n 칸에서 그냥 맞는다) 곱셈이 안 맞고, 어떤 연산은 맞고
   * 어떤 연산은 안 맞는 규칙은 다음 사람이 틀리기 좋은 규칙이다.
   */
  private complexBinary(name: "add" | "sub" | "mul" | "div", other: Tensor): Tensor {
    // **실수 쪽만 모양을 맞춰 준다** — 복소수로 올리기 **전에** 늘리면 실수 쪽의
    // 브로드캐스팅(`expand`)을 그대로 빌릴 수 있다. 올린 뒤에 늘리려 하면 인터리브를
    // 아는 `expand` 가 따로 있어야 한다. 순서 하나로 커널 하나를 안 쓴다.
    const same = (p: readonly number[], q: readonly number[]): boolean =>
      p.length === q.length && p.every((d, i) => d === q[i]);
    let x: Tensor = this;
    let y: Tensor = other;
    if (!x.isComplex() && y.isComplex() && !same(x.shape, y.shape)) {
      x = x.expand(...y.shape);
    }
    if (!y.isComplex() && x.isComplex() && !same(x.shape, y.shape)) {
      y = y.expand(...x.shape);
    }
    const a: Tensor = x.isComplex() ? x : x.asComplexRe();
    const b: Tensor = y.isComplex() ? y : y.asComplexRe();
    if (!same(a.shape, b.shape)) {
      // 복소수끼리 모양이 다른 경우다. 여기는 아직 없다 — 값을 지어내지 않는다.
      throw new RuntimeError(
        `complex ${name} still requires matching shapes: [${a.shape}] vs [${b.shape}] ` +
          "— broadcasting between complex tensors is not here.",
      );
    }
    const n = a.size;
    const out = Tensor.cRun(`cbin:${name}:${n}`, () => complexBinary(name, n),
      [a.raw, b.raw], 2 * n, n);
    return Tensor.make(
      out, a.shape, [a, b],
      (g) => {
        // **켤레가 여기 붙는다.** `d(ab)/da = b` 가 아니라 `conj(b)` 이고,
        // 나눗셈도 같은 자리다. 실수만 넣어 보면 이 줄이 있는지 없는지 모른다.
        switch (name) {
          case "add":
            return [a.requiresGrad ? g : null, b.requiresGrad ? g : null];
          case "sub":
            return [
              a.requiresGrad ? g : null,
              b.requiresGrad ? g.complexNeg() : null,
            ];
          case "mul":
            return [
              a.requiresGrad ? g.complexBinary("mul", b.conjPhysical()) : null,
              b.requiresGrad ? g.complexBinary("mul", a.conjPhysical()) : null,
            ];
          default: {
            const cb = b.conjPhysical();
            return [
              a.requiresGrad ? g.complexBinary("div", cb) : null,
              b.requiresGrad
                ? g.complexBinary("mul", a.conjPhysical())
                  .complexBinary("div", cb.complexBinary("mul", cb))
                  .complexNeg()
                : null,
            ];
          }
        }
      },
      `${name[0]?.toUpperCase()}${name.slice(1)}Backward0`, "complex64",
    );
  }

  // ── 역전파 ────────────────────────────────────────────────────────────

  /**
   * @param retainGraph true keeps the graph. As in torch, **the default is
   *   to release it** — the intermediate values hold memory, and unreleased
   *   they accumulate through a training loop.
   */
  /**
   * Flows gradient.
   *
   * **The argument order matches torch** — `backward(gradient,
   * retainGraph)`. It used to have `retainGraph` first, which made
   * `y.backward(onesLike(y))` impossible to carry across. It is a common
   * line for a Jacobian-vector product, and **the core (numpy) had taken it
   * from the start** — this was the only one of the three that could not.
   *
   * Changing the first position broke no call site. Nowhere in the
   * repository passed an argument to `backward` at all, because the only
   * thing that could be passed was `retainGraph`, and nobody used it.
   *
   * @param gradient the seed. Left out, a 1 is placed on a scalar only.
   *   Given, **its shape has to match** — it is not reconciled by
   *   broadcasting. A mismatched seed gives a wrong gradient with plausible
   *   values, and that surfaces only as training failing to work.
   * @param retainGraph keeps the graph so it can be flowed through again.
   */
  backward(gradient?: Tensor, retainGraph = false): void {
    // **이 검사가 맨 앞이다.** torch 를 재보니 비스칼라이면서 requiresGrad 가 아닌
    // 텐서는 "스칼라가 아니다" 가 아니라 이쪽으로 거절한다 — 스칼라 여부보다 먼저
    // 본다. 코어(numpy)는 처음부터 그 차례였고 여기만 반대였다. 셋이 같은 자리에서
    // 다른 문구를 내면 옮겨 적은 코드가 어느 쪽을 잡아야 할지 모른다.
    if (!this.requiresGrad) {
      throw new RuntimeError(
        `element 0 of tensors ${TORCH.noGrad} and does not have a grad_fn: ` +
          "it was made under no_grad, or it passed through an operation that breaks the graph.",
      );
    }
    // **손실은 실수여야 한다.** torch 가 그 자리에서 멈춘다(실측).
    //
    // 이 한 줄이 복소수 기울기 규약 전체를 떠받친다 — 손실이 늘 실수라야
    // `z.grad = ∂L/∂re + i·∂L/∂im` 이 잘 정의된다. 복소 손실을 받아 주면
    // Wirtinger 의 나머지 절반을 정해야 하고, 그것은 안 정한 자리다.
    if (isComplexDType(this.dtype)) {
      throw new RuntimeError(
        "grad can be implicitly created only for real scalar outputs " +
          "but got torch.complex64: make it real with .real or .abs() first.",
      );
    }
    let seed: Tensor;
    if (gradient === undefined) {
      if (this.size !== 1) {
        throw new RuntimeError(
          `${TORCH.nonScalarBackward}: this shape is [${this.shape}] — ` +
            "pass a gradient, or call .sum() first.",
        );
      }
      seed = Tensor.full([], 1);
    } else {
      if (gradient.shape.length !== this.shape.length
        || gradient.shape.some((n, i) => n !== this.shape[i])) {
        throw new RuntimeError(
          `${TORCH.gradShape}: grad_output[0] has a shape of [${gradient.shape}] ` +
            `and output[0] has a shape of [${this.shape}].`,
        );
      }
      // **그래프를 끊어서 넣는다.** torch 의 기본이 `create_graph=False` 이고 이
      // 테이프는 이중 미분을 안 한다 — 안 끊으면 잎의 `grad` 가 그래프를 물고 남는다.
      seed = gradient.detach();
    }
    tapeBackward<Tensor>(this, seed, (a, b) => a.add(b), {
      retainGraph,
      onSecondPass: () => {
        throw new RuntimeError(
          `${TORCH.secondBackward}. To flow through it again, ` +
            "call backward(undefined, true) to keep the graph.",
        );
      },
    });
  }

  // ── 장치 옮기기 ───────────────────────────────────────────────────────

  /**
   * Brings the values down to the host. Where torch's `t.cpu()` goes, and
   * **asynchronous** — a round trip to fetch GPU memory back, with no way
   * around it.
   *
   * **It cuts the graph.** torch's `.cpu()` is differentiable and here it
   * cannot be — there are no host kernels, so there is no path for a
   * backward to take. Better cut than left attached and quietly flowing
   * zeros.
   */
  async cpu(): Promise<Tensor> {
    if (this.gpu === null) return this;
    // **`floats` 다.** 복소수는 칸당 둘이라 `size` 로 읽으면 뒤쪽 절반이 잘린다 —
    // 모양과 형은 그대로 붙어 나오므로 잘린 것이 안 보인다.
    return new Tensor(await dev().read(this.gpu, this.floats), this.shape, {
      dtype: this.dtype,
    });
  }

  /**
   * Puts the values up on the GPU. Where torch's `t.cuda()` goes.
   *
   * **Unlike `cpu()` it is synchronous.** Uploading is one queued write and
   * there is nothing to wait for — it looks asymmetric, and that is better
   * than inventing a round trip to dress up the symmetry.
   */
  webgpu(): Tensor {
    if (this.host === null) return this;
    return new Tensor(dev().upload(this.host), this.shape, { dtype: this.dtype });
  }

  // ── 읽기 ──────────────────────────────────────────────────────────────

  /**
   * The values as a flat f32 array.
   *
   * **A complex comes out interleaved as it is stored** — the length is `2
   * × size` rather than `size`, and it reads `[re, im, re, im, …]`. For the
   * real part alone call `real()` first; for pairs, `viewAsReal()`. Picking
   * out the real part here would make the length plausible and hide that
   * values were lost.
   */
  async toArray(): Promise<Float32Array> {
    // 이미 호스트에 있으면 왕복이 없다. **사본을 준다** — 안쪽 저장을 그대로 내보내면
    // 받은 쪽이 그것을 고칠 때 텐서 값이 같이 바뀐다.
    if (this.host !== null) return this.host.slice();
    return dev().read(this.raw, this.floats);
  }

  /**
   * Whether the shape and values are **exactly** equal. There is no
   * tolerance — that is what differs from `allclose`.
   *
   * It reads back from the GPU. Making a round trip for one verdict is a
   * shame, but holding a CPU copy and comparing that cannot see what
   * happened on the GPU.
   */
  async equal(other: Tensor): Promise<boolean> {
    if (this.shape.length !== other.shape.length ||
        this.shape.some((d, i) => d !== other.shape[i])) {
      return false;
    }
    const [a, b] = await Promise.all([this.toArray(), other.toArray()]);
    return a.every((v, i) => v === b[i]);
  }

  /** 허용 오차 안에서 같은가. torch 의 기본값과 같다. */
  /**
   * The golden harness does **not** turn this on — turned on, NaN passes
   * where NaN must not. Having the argument and turning it on are different
   * places: whether to is the caller's decision.
   *
   * @param equalNan true treats **NaN as equal to NaN.** The default is
   *   false, so they are not.
   */
  async allclose(other: Tensor, rtol = 1e-5, atol = 1e-8,
                 equalNan = false): Promise<boolean> {
    if (this.shape.length !== other.shape.length ||
        this.shape.some((d, i) => d !== other.shape[i])) {
      return false;
    }
    const [a, b] = await Promise.all([this.toArray(), other.toArray()]);
    return a.every((v, i) => {
      const w = b[i] ?? Number.NaN;
      if (Number.isNaN(v) && Number.isNaN(w)) return equalNan;
      return Math.abs(v - w) <= atol + rtol * Math.abs(w);
    });
  }

  /**
   * The text `print(t)` prints. **It reads back from the GPU, so it is
   * asynchronous.**
   *
   * Textbooks print this text verbatim, so this project treats it as a
   * specification too — the reasoning is in `src/repr.ts`.
   */
  async repr(): Promise<string> {
    // **복소수는 아직 못 찍는다.** `1.+2.j` 꼴을 torch 와 글자까지 맞추려면 그쪽
    // 자리맞춤 규칙을 재서 굳혀야 하고, 그건 아직 안 한 일이다. 반쯤 맞는 글자를
    // 내면 그것이 교재의 줄과 안 맞는데도 맞는 것처럼 보인다.
    if (isComplexDType(this.dtype)) {
      throw new RuntimeError(
        "there is no repr for complex64 yet — print the real pair with viewAsReal().",
      );
    }
    const values = Array.from(await this.toArray());
    return formatTensor({
      values,
      shape: this.shape,
      dtype: this.dtype,
      requiresGrad: this.requiresGrad,
      gradName: this.parents.length > 0 ? this.gradName : "",
    });
  }

  /**
   * `torch.Size([2, 2])`. The shape is printed too, so it is a
   * specification.
   */
  sizeRepr(): string {
    return formatSize(this.shape);
  }

  /** 형을 바꾼다. 값은 그대로다 — 저장이 float32 하나이므로 옮길 것이 없다. */
  /**
   * torch's **named type conversions.** The same work as `to(type)` under a
   * different name.
   *
   * Without these names, `x.float()` failed in borch.ts alone. Both Python
   * versions had them and the golden line stood as "a Python-side matter,
   * not carried across", when in fact it was not that the other side could
   * not — it was a name **nobody had asked about.**
   */
  float(): Tensor {
    return this.to("float32");
  }

  long(): Tensor {
    return this.to("int64");
  }

  bool(): Tensor {
    return this.to("bool");
  }

  /**
   * Real to complex. **`to("complex64")` will not do it** — a complex is
   * two f32 per slot, so relabelling is blocked, and that block is right.
   * It is built by attaching a zero imaginary part.
   */
  cfloat(): Tensor {
    if (isComplexDType(this.dtype)) return this;
    const re = this.dtype === "float32" ? this : this.to("float32");
    return Tensor.complex(re, Tensor.zeros(this.shape));
  }

  /**
   * To the same type as `other`. torch's `type_as`.
   */
  typeAs(other: Tensor): Tensor {
    return other.dtype === "complex64" ? this.cfloat() : this.to(other.dtype);
  }

  /**
   * **Types that do not exist.** The names are kept for the message — with
   * no name at all you get `'half' does not exist`, and that **cannot be
   * told apart from a typo.** Both Python versions stop with the same
   * sentence, so this one matches. It is a place matching text rather than
   * values, so comparing them against each other catches nothing, and a
   * learner reading divergent wording reads it as **differing per
   * implementation.**
   */
  half(): Tensor {
    return absentDType("half", "float16");
  }

  bfloat16(): Tensor {
    return absentDType("bfloat16", "bfloat16");
  }

  chalf(): Tensor {
    return absentDType("chalf", "complex32");
  }

  cdouble(): Tensor {
    return absentDType("cdouble", "complex128");
  }

  byte(): Tensor {
    return absentDType("byte", "uint8");
  }

  char(): Tensor {
    return absentDType("char", "int8");
  }

  short(): Tensor {
    return absentDType("short", "int16");
  }

  int(): Tensor {
    return absentDType("int", "int32");
  }

  /**
   * **Double precision is absent for a different reason.** The eight above
   * are slots we chose not to have; this one is that WebGPU shaders have no
   * `f64` at all. The wording differs accordingly — the Python binding
   * stops with the same sentence.
   */
  double(): Tensor {
    throw new RuntimeError(
      "Only Tensors of floating point dtype float32 are supported — "
        + "float64 is not in WebGPU shaders",
    );
  }

  to(dtype: DType): Tensor {
    if (dtype === this.dtype) return this;
    // **복소수는 이름표 갈이로 오갈 수 없다.** 다른 형끼리는 저장이 float32 하나로
    // 같아서 이름만 바꾸면 되는데, 복소수만 칸당 두 개다 — 양쪽 어느 방향으로든
    // 이름만 바꾸면 버퍼 길이와 `size` 가 어긋난 채로 남는다.
    if (dtype === "complex64" || isComplexDType(this.dtype)) {
      throw new RuntimeError(
        `torch.${this.dtype} -> torch.${dtype} is not a relabel — ` +
          "complex storage is two f32 per slot. " +
          "Move between them with Tensor.complex(re, im), viewAsComplex(), or real().",
      );
    }
    // **정수·참거짓으로 갈 때는 값도 바꾼다.** 형이 이름표라는 것이 "아무 값이나
    // 들어 있어도 된다" 는 뜻은 아니다 — torch 의 int64 텐서에는 정수가 들어 있다.
    //
    // 오래 이름만 갈고 있었다. `x.to("int64")` 뒤에도 버퍼에 `1.7` 이 남아서,
    // 읽어 갈 때만 깎이고 **GPU 위의 산술은 소수로 계속됐다.** `sum(dtype=int64)`
    // 케이스가 정확히 1 만큼 갈려서 드러났다 — torch 는 먼저 깎아 `−1` 인데
    // 우리는 안 깎아 `0.3` 을 접고 그것을 깎아 `0` 이었다.
    if (dtype === "int64" && this.dtype === "float32") {
      // **0 쪽으로 깎는다**(실측: `−2.3 → −2`). `floor` 면 `−3` 이라 갈린다.
      return this.unary("trunc").relabel(dtype);
    }
    if (dtype === "bool" && this.dtype !== "bool") {
      return this.binary("ne", Tensor.full([], 0), "bool");
    }
    return this.relabel(dtype);
  }

  /**
   * 값은 그대로 두고 이름표만 간다. **저장이 같은 형끼리만** 쓴다.
   *
   * `int64 → float32` 처럼 값이 이미 그 형의 것인 자리다. 기울기는 **끊지 않는다** —
   * 코어에서 `.float()` 이 조용히 끊겨 있던 자리가 정확히 이것이다.
   */
  private relabel(dtype: DType): Tensor {
    const from = this.dtype;
    return Tensor.make(
      this.buffer,
      this.shape,
      [this],
      (g) => [new Tensor(g.buffer, this.shape, { dtype: from })],
      "ToCopyBackward0",
      dtype,
    );
  }

  async item(): Promise<number> {
    // **자바스크립트에 복소수가 없다.** 실수부만 돌려주면 숫자 하나가 그럴듯하게
    // 나오면서 허수부가 사라진다 — 없는 것보다 나쁜 답이다.
    if (isComplexDType(this.dtype)) {
      throw new RuntimeError(
        "complex64 has no item() — JavaScript has no complex value. " +
          "Split it with real() and imag().",
      );
    }
    if (this.size !== 1) {
      throw new RuntimeError(
        `a Tensor with ${this.size} elements ${TORCH.itemScalar}`,
      );
    }
    const arr = await this.toArray();
    return arr[0] ?? Number.NaN;
  }
}

/**
 * 누적에 `dtype: "bool"` 은 torch 가 거절한다(실측 — `NotImplementedError` 다).
 *
 * **`sum(dtype=bool)` 은 되는데 `cumsum(dtype=bool)` 은 안 된다.** 규칙이 아니라
 * torch 가 그 커널을 안 만든 것이고, 관대한 쪽으로 갈리는 것도 갈리는 것이라
 * 따라간다 — 여기서 값을 내주면 그 코드가 진짜 torch 에서 깨진다.
 */
function noBoolAccumulate(name: string, dtype: DType): void {
  if (dtype === "bool") {
    throw new NotImplementedError(`"${name}_out_cpu" not implemented for 'Bool'`);
  }
}

/** 넓은 기울기를 목표 모양으로 접는다. 모양이 이미 같으면 그대로 둔다. */
function foldTo(wide: Tensor, target: readonly number[]): Tensor {
  if (numel(wide.shape) === numel(target) && wide.shape.length === target.length) {
    return new Tensor(wide.buffer, target);
  }
  const small = padShape(target, wide.shape.length);
  const n = numel(small);
  const out = dev().alloc(n);
  dev().run1d(
    dev().pipeline(
      `rb:${wide.shape}|${small}`,
      () => reduceBroadcast(wide.shape, small),
    ),
    [wide.buffer, out],
    n,
  );
  return new Tensor(out, target);
}

/** 스칼라 기울기를 모양대로 편다. `sum` 의 역방향이다. */
function foldFrom(g: Tensor, shape: readonly number[]): Tensor {
  const n = numel(shape);
  const out = dev().alloc(n);
  dev().run1d(
    dev().pipeline(`bcast1:${n}`, () => broadcastScalar(n)),
    [g.buffer, out],
    n,
  );
  return new Tensor(out, shape);
}

/** 스칼라 하나를 n 칸에 뿌린다. */
function broadcastScalar(n: number): string {
  return `
@group(0) @binding(0) var<storage, read> S: array<f32>;
@group(0) @binding(1) var<storage, read_write> Out: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let gid = g.y * ${Math.min(Math.max(1, Math.ceil(n / 64)), 65535) * 64}u + g.x;
  if (gid >= ${n}u) { return; }
  Out[gid] = S[0];
}`;
}

// ── 복소수 커널 ─────────────────────────────────────────────────────────
//
// 전부 **복소수 칸 하나에 스레드 하나**다. 그래서 `i` 는 언제나 복소수 색인이고,
// f32 자리는 `i*2`(실수부)와 `i*2+1`(허수부)이다. 스레드를 f32 칸에 붙이면 짝을
// 한꺼번에 못 읽어서 곱셈이 안 써진다.

/**
 * 1 차원 복소수 커널의 틀. **격자 접는 줄을 한 번만 적는다.**
 *
 * 열 개 가까운 커널이 같은 머리를 갖는데, 손으로 열 번 적으면 그중 하나가 다르게
 * 적히는 날이 온다 — 이 저장소가 이미 그 종류로 여러 번 물렸다.
 *
 * @param names the binding names. **The last is the output** and only it is
 *   writable.
 */
function complexShader(n: number, names: readonly string[], body: string): string {
  const decls = names.map((nm, i) => {
    const mode = i === names.length - 1 ? "read_write" : "read";
    return `@group(0) @binding(${i}) var<storage, ${mode}> ${nm}: array<f32>;`;
  }).join("\n");
  const stride = Math.min(Math.max(1, Math.ceil(n / 64)), 65535) * 64;
  return `
${decls}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let gid = g.y * ${stride}u + g.x;
  if (gid >= ${n}u) { return; }
  let i = gid * 2u;
${body}
}`;
}

/** 실수부·허수부를 엮어 인터리브로. */
function complexPack(n: number): string {
  return complexShader(n, ["Re", "Im", "Out"],
    "  Out[i] = Re[gid];\n  Out[i + 1u] = Im[gid];");
}

/** 크기·편각에서 인터리브로. */
function complexPolar(n: number): string {
  return complexShader(n, ["R", "T", "Out"],
    "  Out[i] = R[gid] * cos(T[gid]);\n  Out[i + 1u] = R[gid] * sin(T[gid]);");
}

/** 실수부(`off=0`)나 허수부(`off=1`)를 꺼낸다. */
function complexPart(n: number, off: 0 | 1): string {
  return complexShader(n, ["Z", "Out"], `  Out[gid] = Z[i + ${off}u];`);
}

/** 실수를 `x + 0i` 로. */
function complexFromReal(n: number): string {
  return complexShader(n, ["A", "Out"],
    "  Out[i] = A[gid];\n  Out[i + 1u] = 0.0;");
}

/** 실수를 `0 + xi` 로. */
function complexFromImag(n: number): string {
  return complexShader(n, ["A", "Out"],
    "  Out[i] = 0.0;\n  Out[i + 1u] = A[gid];");
}

/** 켤레. 허수부만 뒤집는다. */
function complexConj(n: number): string {
  return complexShader(n, ["Z", "Out"],
    "  Out[i] = Z[i];\n  Out[i + 1u] = -Z[i + 1u];");
}

/** 부호 뒤집기. 둘 다 뒤집는다 — 켤레와 헷갈리기 쉬운 자리다. */
function complexNeg(n: number): string {
  return complexShader(n, ["Z", "Out"],
    "  Out[i] = -Z[i];\n  Out[i + 1u] = -Z[i + 1u];");
}

/** 크기. 결과는 실수 한 칸이다. */
function complexAbs(n: number): string {
  return complexShader(n, ["Z", "Out"],
    "  Out[gid] = sqrt(Z[i] * Z[i] + Z[i + 1u] * Z[i + 1u]);");
}

/** 편각. `atan2(im, re)` — 인자 차례가 뒤집히면 조용히 다른 각이 된다. */
function complexAngle(n: number): string {
  return complexShader(n, ["Z", "Out"], "  Out[gid] = atan2(Z[i + 1u], Z[i]);");
}

/**
 * `abs` 의 역방향. **켤레가 안 붙는다** — `abs` 는 실수를 내므로 정칙이 아니다.
 *
 * 0 에서는 방향이 없다. torch 도 거기서 0 을 준다 — 나누는 값을 1 로 바꿔 두면
 * 분자가 0 이라 결과가 0 이 된다.
 */
function complexAbsBackward(n: number): string {
  return complexShader(n, ["Z", "G", "Out"], `
  let re = Z[i];
  let im = Z[i + 1u];
  let m = sqrt(re * re + im * im);
  let s = select(1.0, m, m > 0.0);
  Out[i] = G[gid] * re / s;
  Out[i + 1u] = G[gid] * im / s;`);
}

/** 복소수 사칙. 모양이 같은 것끼리만 온다. */
function complexBinary(name: "add" | "sub" | "mul" | "div", n: number): string {
  const head = `
  let ar = A[i];
  let ai = A[i + 1u];
  let br = B[i];
  let bi = B[i + 1u];`;
  const body: Readonly<Record<string, string>> = {
    add: "\n  Out[i] = ar + br;\n  Out[i + 1u] = ai + bi;",
    sub: "\n  Out[i] = ar - br;\n  Out[i + 1u] = ai - bi;",
    mul: "\n  Out[i] = ar * br - ai * bi;\n  Out[i + 1u] = ar * bi + ai * br;",
    div: `
  let d = br * br + bi * bi;
  Out[i] = (ar * br + ai * bi) / d;
  Out[i + 1u] = (ai * br - ar * bi) / d;`,
  };
  return complexShader(n, ["A", "B", "Out"], head + (body[name] ?? ""));
}

/** 2차원 전치 커널. 모양이 상수라 나눗셈이 안 남는다. */
function transposeKernel(M: number, N: number): string {
  const n = M * N;
  return `
@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read_write> Out: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let gid = g.y * ${Math.min(Math.max(1, Math.ceil(n / 64)), 65535) * 64}u + g.x;
  if (gid >= ${n}u) { return; }
  let r = gid / ${N}u;
  let c = gid % ${N}u;
  Out[c * ${M}u + r] = A[gid];
}`;
}

/**
 * 표에 있는 단항을 전부 메서드로 단다. 이름을 두 번 적지 않는다.
 *
 * 제자리 판(`abs_` 처럼 밑줄이 붙은 것)도 같이 단다 — 스물일곱 개를 손으로 적으면
 * 그중 하나가 다른 연산을 부르는 날이 온다.
 */
// **표가 덮기 전에 잡아 둔다.** 클래스 본문의 `elu` 는 α 를 받는데, 아래 루프가
// 표의 무인자 판으로 덮어 버린다 — `abs` 에서 이미 물린 순서 문제다. 루프 뒤에
// 되돌린다. 제자리 판(`elu_`)은 표 쪽 것을 그대로 쓴다(α=1).
const eluWithAlpha = Tensor.prototype.elu;

for (const name of Object.keys(UNARY)) {
  Object.defineProperty(Tensor.prototype, name, {
    value: function (this: Tensor): Tensor {
      return this.unary(name);
    },
    writable: true,
    configurable: true,
  });
  Object.defineProperty(Tensor.prototype, `${name}_`, {
    value: function (this: Tensor): Tensor {
      return this.inplaceUnary(name);
    },
    writable: true,
    configurable: true,
  });
}

/**
 * **`abs` 만 표 뒤에서 다시 단다.** 위 루프가 프로토타입에 얹고 나므로, 클래스
 * 본문에 적은 것은 덮인다 — 순서가 곧 규칙이다.
 *
 * 복소수의 `abs` 는 실수를 내고 기울기가 `z/|z|` 다. 실수의 것과 커널도 형도
 * 역방향도 달라서 표의 단항으로는 안 된다.
 */
// α 를 받는 `elu` 를 되돌린다. 위에서 잡아 둔 것이다.
Object.defineProperty(Tensor.prototype, "elu", {
  value: eluWithAlpha,
  writable: true,
  configurable: true,
});

{
  const realAbs = Tensor.prototype.abs;
  Object.defineProperty(Tensor.prototype, "abs", {
    value: function (this: Tensor): Tensor {
      return this.isComplex() ? this.complexAbs() : realAbs.call(this);
    },
    writable: true,
    configurable: true,
  });
}

/**
 * **`bitwise_not` 도 표 뒤에서 다시 단다.** 참거짓이면 논리 부정이다 — `~true` 는
 * `-2` 가 아니라 `false` 다(실측). 그 갈림이 오래 결속에만 있었고, 커널 주석은
 * "여기는 정수만 본다" 고 적어 두었다. 그러면 TypeScript 로 부르는 쪽은 **없는 답이
 * 아니라 틀린 답**을 받는다.
 *
 * 클래스 본문에 적었더니 위 루프가 덮었다 — `abs`·`elu` 가 이미 물린 순서 문제이고,
 * 세 번째다. 표에 이름이 있는 연산을 손으로 고치려면 **표 뒤**여야 한다.
 */
{
  const rawNot = Tensor.prototype.bitwise_not;
  Object.defineProperty(Tensor.prototype, "bitwise_not", {
    value: function (this: Tensor): Tensor {
      return this.dtype === "bool" ? this.logicalNot() : rawNot.call(this);
    },
    writable: true,
    configurable: true,
  });
}

/**
 * The types of the methods hung from the table above. Paired with the loop
 * above; fix one alone and they fall out of step.
 */
export interface Tensor {
  neg(): Tensor;
  abs(): Tensor;
  exp(): Tensor;
  log(): Tensor;
  sqrt(): Tensor;
  rsqrt(): Tensor;
  square(): Tensor;
  reciprocal(): Tensor;
  sin(): Tensor;
  cos(): Tensor;
  tan(): Tensor;
  sinh(): Tensor;
  cosh(): Tensor;
  tanh(): Tensor;
  asin(): Tensor;
  acos(): Tensor;
  atan(): Tensor;
  asinh(): Tensor;
  acosh(): Tensor;
  atanh(): Tensor;
  exp2(): Tensor;
  log2(): Tensor;
  log10(): Tensor;
  expm1(): Tensor;
  log1p(): Tensor;
  relu(): Tensor;
  sigmoid(): Tensor;
  sign(): Tensor;
  floor(): Tensor;
  ceil(): Tensor;
  round(): Tensor;
  trunc(): Tensor;
  frac(): Tensor;
  lgamma(): Tensor;
  digamma(): Tensor;
  erfinv(): Tensor;
  hardsigmoid(): Tensor;
  hardswish(): Tensor;
  logsigmoid(): Tensor;
  mish(): Tensor;
  relu6(): Tensor;
  selu(): Tensor;
  softsign(): Tensor;
  tanhshrink(): Tensor;
  bitwise_not(): Tensor;
  i0(): Tensor;
  i0_(): Tensor;
  frexpMantissa(): Tensor;
  frexpExponent(): Tensor;
  // ── 표가 다는 제자리 판 ──────────────────────────────────────────────
  //
  // 위 루프는 이름마다 **둘**을 단다(`abs` 와 `abs_`). 이 선언에는 오래 앞의 것만
  // 있었고, 그래서 예순다섯 개가 **런타임에는 있는데 타입에는 없었다** — TypeScript
  // 로 `x.acosh_()` 를 치면 컴파일이 막히고, 사이트 레퍼런스에도 안 나온다.
  //
  // 이 블록 머리에 "위 루프와 짝이고 하나만 고치면 어긋난다" 고 적혀 있었다.
  // 어긋난 것은 고친 쪽이 아니라 **처음부터 반쪽만 적은 쪽**이었다.
  neg_(): Tensor;
  abs_(): Tensor;
  exp_(): Tensor;
  log_(): Tensor;
  sqrt_(): Tensor;
  rsqrt_(): Tensor;
  square_(): Tensor;
  reciprocal_(): Tensor;
  sin_(): Tensor;
  cos_(): Tensor;
  tan_(): Tensor;
  sinh_(): Tensor;
  cosh_(): Tensor;
  tanh_(): Tensor;
  asin_(): Tensor;
  acos_(): Tensor;
  atan_(): Tensor;
  asinh_(): Tensor;
  acosh_(): Tensor;
  atanh_(): Tensor;
  exp2_(): Tensor;
  log2_(): Tensor;
  log10_(): Tensor;
  expm1_(): Tensor;
  log1p_(): Tensor;
  relu_(): Tensor;
  sigmoid_(): Tensor;
  sign_(): Tensor;
  floor_(): Tensor;
  ceil_(): Tensor;
  round_(): Tensor;
  trunc_(): Tensor;
  frac_(): Tensor;
  deg2rad_(): Tensor;
  rad2deg_(): Tensor;
  positive_(): Tensor;
  logit_(): Tensor;
  sinc_(): Tensor;
  erf_(): Tensor;
  erfc_(): Tensor;
  sgn_(): Tensor;
  signbit_(): Tensor;
  nanToZero_(): Tensor;
  notNan_(): Tensor;
  isnan_(): Tensor;
  isinf_(): Tensor;
  isfinite_(): Tensor;
  logical_not_(): Tensor;
  bitwise_not_(): Tensor;
  frexpMantissa_(): Tensor;
  frexpExponent_(): Tensor;
  gelu_(): Tensor;
  silu_(): Tensor;
  elu_(): Tensor;
  hardsigmoid_(): Tensor;
  hardswish_(): Tensor;
  logsigmoid_(): Tensor;
  mish_(): Tensor;
  relu6_(): Tensor;
  selu_(): Tensor;
  softsign_(): Tensor;
  tanhshrink_(): Tensor;
  lgamma_(): Tensor;
  digamma_(): Tensor;
  erfinv_(): Tensor;

}

export { noGrad } from "./autograd.js";

/**
 * One open scope. `using` closes it at the end of the block.
 *
 * @see scope
 */
export interface Scope extends Disposable {
  /**
   * Keeps something alive past this scope's close. The same work as the
   * callback form's second argument.
   *
   * What differs from `keepAlive(t)` is **where it is tied.** That one
   * keeps it alive forever; this hands it to the enclosing scope — closing
   * that releases it too.
   */
  keep(t: Tensor): Tensor;
}

/**
 * Opens and closes one scope. GPU buffers made inside are released on the
 * way out.
 *
 * **A training loop does not run without this.** One step makes thousands
 * of intermediate buffers, and JavaScript's garbage collector does not
 * return GPU memory in time.
 *
 * ## Two spellings — the same machine
 *
 * ```ts
 * await scope(async () => {           // callback
 *   ...
 *   return await loss.item();
 * });
 *
 * {
 *   using s = scope();                // block. closer to torch's `with`
 *   ...
 *   await loss.item();
 * }                                   // closes here
 * ```
 *
 * **The block form closes after the `await`s.** Inside an async function,
 * the block is left only after every `await` in it has finished, and
 * releasing (`endScope`) is itself synchronous, so `using` works rather
 * than `await using` — and that one is more widely supported.
 *
 * **The callback form is not removed.** Both stand on the same
 * `beginScope`/`endScope`, so there is nowhere for them to diverge, and
 * where you want the value handed straight back (`const loss = await
 * scope(...)`) the callback is shorter.
 *
 * @param keep the tensors to carry out of the scope. The rest are released.
 */
export function scope(): Scope;
export function scope<T>(
  body: () => Promise<T>,
  keep?: () => readonly Tensor[],
): Promise<T>;
export function scope<T>(
  body?: () => Promise<T>,
  keep: () => readonly Tensor[] = () => [],
): Promise<T> | Scope {
  const d = dev();
  d.beginScope();
  // **`raw` 다.** 살려 둘 것 중에 복소수가 있으면 `buffer` 가 거절하고, 그러면
  // 구역을 닫는 자리에서 예외가 난다 — 수명 관리는 값의 형을 알 필요가 없다.
  if (body === undefined) {
    const kept: Tensor[] = [];
    return {
      keep(t: Tensor): Tensor {
        kept.push(t);
        return t;
      },
      [Symbol.dispose](): void {
        d.endScope(kept.map((t) => t.raw));
      },
    };
  }
  return (async () => {
    try {
      return await body();
    } finally {
      d.endScope(keep().map((t) => t.raw));
    }
  })();
}

/**
 * Keeps something alive past a scope's close. Parameters and optimizer
 * state use it.
 */
export function keepAlive(t: Tensor): Tensor {
  // 호스트에 있는 것은 살릴 것이 없다 — 구역은 GPU 버퍼만 놓고, `Float32Array` 는
  // 자바스크립트의 쓰레기 수집이 알아서 가져간다. `keepAlive(await t.cpu())` 는
  // 자연스러운 줄이므로 여기서 거절하면 안 된다.
  if (t.device === "cpu") return t;
  dev().keep(t.raw);
  return t;
}

/**
 * Where released buffers are counted. Benchmarks watch leaks here.
 */
export function device(): Device {
  return dev();
}

/**
 * Makes one graph node from outside. **A door for places writing kernels in
 * another file.**
 *
 * `Tensor.make` is private — it is what this class's own operations use,
 * and that is right. But once a module with **kernels of its own** appears,
 * such as `fft.ts`, that door is needed. Rather than restating the
 * conditions there (two copies of the `gradMode` check and the
 * `requiresGrad` propagation eventually diverge), it hands over to the same
 * place.
 */
export function makeNode(
  buffer: GPUBuffer,
  shape: readonly number[],
  parents: readonly Tensor[],
  backwardFn: (grad: Tensor) => readonly (Tensor | null)[],
  gradName: string,
  dtype: DType = "float32",
): Tensor {
  return Tensor.node(buffer, shape, parents, backwardFn, gradName, dtype);
}
