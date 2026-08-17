/**
 * `Tensor` — GPU 버퍼 하나와 모양, 그리고 테이프의 마디 하나.
 *
 * 레이아웃은 **NCHW**, dtype 은 **float32 하나**다. 자매(`borch_webgpu`)가
 * NHWC 를 들고 int64 를 float32 에 담은 것은 TF.js 의 제약을 피한 우회였고, 우리
 * 커널에는 그 제약이 없다. 흉내 내면 이유 없이 남의 우회를 물려받는다.
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

/** `F.pad` 가 받는 네 가지. */
export type PadMode = "constant" | "reflect" | "replicate" | "circular";

/** 손실이 접는 방식. **`KLDivLoss` 만 넷째(`batchmean`)를 더 받는다.** */
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

/** `at()` 의 축 하나에 넣을 수 있는 것. 문법은 `indexing.ts` 에 적혀 있다. */
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
 * WebGPU 어댑터를 잡는다. 두 번째부터는 이미 잡은 것을 돌려준다.
 *
 * **`options` 는 첫 부름에서만 듣는다.** 장치는 하나이고 이미 만들어진 뒤에는 고를
 * 것이 없다 — 조용히 무시하느니 여기 적어 둔다.
 */
export async function init(options: InitOptions = {}): Promise<Device> {
  if (!deviceHolder.current) deviceHolder.current = await Device.create(options);
  return deviceHolder.current;
}

/**
 * 지금 붙어 있는 장치, 안 붙었으면 `null`. `torch.accelerator` 자리다.
 *
 * **`dev()` 와 달리 안 던진다** — 붙었는지 묻는 것이 목적이라 그 물음 자체가 실패하면
 * 안 된다.
 */
export function currentDevice(): DeviceKind | null {
  return deviceHolder.current ? "webgpu" : null;
}

function dev(): Device {
  const d = deviceHolder.current;
  if (!d) throw new Error("장치가 없다 — 먼저 `await init()` 을 불러라.");
  return d;
}

function numel(shape: readonly number[]): number {
  return shape.reduce((a, b) => a * b, 1);
}

/**
 * torch 의 브로드캐스팅 규칙. 오른쪽부터 맞추고, 1 은 늘어나고, 나머지는 같아야 한다.
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
          `non-singleton dimension ${i}: [${a}] 와 [${b}]`,
      );
    }
    out[i] = Math.max(da, db);
  }
  return out;
}

/**
 * `shape` 를 `out` 랭크에 오른쪽 맞춤으로 놓았을 때의 스트라이드.
 *
 * **늘어난 축은 0 이다** — 같은 값을 계속 읽는다. 실제로 복제해서 늘리면 메모리를
 * 쓰고, conv 벤치에서 im2col 이 융합 커널에 진 이유가 정확히 그것이었다.
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

export class Tensor implements Node<Tensor> {
  /**
   * **`readonly` 가 아니다** — `mutate` 하나가 고친다.
   *
   * 모양을 바꾸는 제자리 연산이 있다(`transpose_`·`squeeze_`·`unsqueeze_`). 그것들은
   * 값이 아니라 보는 틀을 고치므로, 자리를 그대로 두고 모양만 갈아 끼워야 한다.
   * 그 한 자리 말고는 아무 데서도 안 바뀐다.
   */
  shape: readonly number[];
  /**
   * **`readonly` 가 아니다** — `shape` 와 같은 이유이고, 같은 한 자리만 고친다.
   *
   * `as_strided_` 는 칸 수까지 바꾼다. 다른 제자리 연산들은 틀만 바꿔서 오랫동안
   * "칸 수는 안 변한다" 가 참이었는데, 그것이 참이 아닌 이름이 하나 생겼다.
   */
  size: number;
  /**
   * 값이 GPU 에 있으면 그 버퍼, 호스트에 있으면 `null`. **직접 읽지 마라** — 밖에서
   * 보는 자리는 `buffer` 게터이고 그쪽이 장치를 확인한다.
   */
  private readonly gpu: GPUBuffer | null;
  /** 값이 호스트에 있으면 그 배열, GPU 에 있으면 `null`. */
  private readonly host: Float32Array | null;
  requiresGrad: boolean;
  grad: Tensor | null = null;
  freed = false;
  /**
   * **이름표다.** 값은 언제나 float32 버퍼에 있다 — 자세한 사정은 `src/dtype.ts`.
   */
  readonly dtype: DType;
  /**
   * 그래프의 위쪽. **`detach_` 만 이것을 바꾼다** — 그래서 `readonly` 가 아니다.
   * 다른 자리에서 고치면 이미 만든 마디의 역방향이 조용히 달라진다.
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
   * 값이 든 GPU 버퍼. **연산이 값에 닿는 유일한 문이다.**
   *
   * 이 게터가 곧 장치 검사다. 연산 진입점은 176 군데이고 거기에 하나씩 가드를 다는
   * 것은 될 일이 아니지만, 그 전부가 결국 여기를 지난다 — 저장소 안 75 곳, 옵티마이저
   * 6 곳, `nn` 1 곳. 그래서 호스트에 있는 텐서를 연산에 넣으면 어느 연산이든 여기서
   * 멈추고, 문구는 torch 와 같은 것이 나온다.
   *
   * **어느 연산이었는지는 안 적힌다.** 그것을 적으려면 176 곳을 고쳐야 하고, 스택
   * 추적이 이미 그 자리를 가리킨다.
   */
  get buffer(): GPUBuffer {
    if (this.gpu === null) {
      throw new RuntimeError(
        `${TORCH.crossDevice}, but found at least two devices, webgpu and cpu! ` +
          "호스트에 있는 텐서는 연산에 못 쓴다 — `webgpu()` 로 올려라.",
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
        "이 연산은 complex64 를 아직 안 받는다 — 저장이 칸당 f32 두 개(인터리브)라 " +
          "복소수를 모르는 커널이 읽으면 조용히 틀린다. " +
          "`view_as_real` 로 실수 텐서를 만든 뒤 쓰거나, 실수부·허수부를 따로 다뤄라.",
      );
    }
    return this.gpu;
  }

  /**
   * 값이 든 버퍼 — **복소수 검사 없이.** 복소수를 아는 코드만 쓴다.
   *
   * 장치 검사는 그대로 한다. 두 게터의 차이는 복소수 한 줄뿐이고, 그 한 줄이 곧
   * "이 코드가 인터리브 저장을 안다" 는 선언이다.
   */
  get raw(): GPUBuffer {
    if (this.gpu === null) {
      throw new RuntimeError(
        `${TORCH.crossDevice}, but found at least two devices, webgpu and cpu! ` +
          "호스트에 있는 텐서는 연산에 못 쓴다 — `webgpu()` 로 올려라.",
      );
    }
    return this.gpu;
  }

  /**
   * 버퍼가 실제로 든 f32 칸 수. 실수는 `size` 와 같고 **복소수는 그 두 배다.**
   *
   * 읽기·복사·수명처럼 "값이 몇 칸인가" 만 묻는 자리가 이것을 쓴다. `size` 를 그대로
   * 쓰면 복소수의 뒤쪽 절반이 조용히 잘린다 — 모양도 원소 수도 그럴듯한 채로.
   */
  get floats(): number {
    return this.size * floatsPerElement(this.dtype);
  }

  /**
   * 값이 어디에 있는가. torch 의 `t.device` 자리다.
   *
   * **`'cpu'` 는 값이 담긴 그릇이지 연산되는 장치가 아니다.** borch 에 CPU 커널은
   * 없다 — `cpu()` 로 내린 텐서는 읽고(`toArray`·`item`·`repr`) 다시 올릴 수 있을
   * 뿐이고, 연산에 넣으면 위의 게터가 torch 와 같은 문구로 멈춘다. 부분 구현이지만
   * 실패가 조용하지 않다.
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
   * `make` 의 공개 문. **자기 커널을 가진 다른 모듈**(`fft.ts`)이 여기로 들어온다.
   *
   * `make` 자체를 공개하지 않는 이유는 이름이 너무 흔해서다 — 밖에서 `Tensor.make`
   * 를 보면 무엇을 만드는지가 안 읽힌다. 파일 끝의 `makeNode` 가 이 자리를 감싼다.
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
   * @param options 위치 인자가 아니라 이름으로 받는다. 넷째 자리에 `device` 가 붙을
   *   참이었는데, `from(data, shape, false, "int64", "cpu")` 는 읽는 사람이 셋째와
   *   넷째를 세어야 하는 줄이다. npm 에 아직 안 나간 지금이 고칠 수 있는 때다.
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
      throw new Error(`모양 [${shp}] 는 원소 ${flat.length}개와 안 맞는다.`);
    }
    // **복소수는 이 문으로 못 들어온다.** 이름표만 `complex64` 로 달면 칸 수는
    // `n` 인데 저장 규약은 `2n` 을 요구하므로, 뒤쪽 절반이 남의 메모리가 된다 —
    // 예외 없이 아무 값이나 읽힌다. 엮는 자리를 하나로 둔다.
    if (dtype === "complex64") {
      throw new RuntimeError(
        "Tensor.from 으로는 complex64 를 못 만든다 — 저장이 칸당 f32 두 개다. " +
          "`Tensor.complex(re, im)`·`Tensor.polar(r, θ)` 나 " +
          "`x.viewAsComplex()` 를 써라.",
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
   * 값으로 채우되 **캐시를 안 탄다 — 자기 버퍼를 갖는다.**
   *
   * **제자리로 고쳐질 것은 반드시 이쪽이어야 한다.** `full` 은 원소 하나짜리를 값으로
   * 캐시하므로 `zeros([1])` 을 두 번 부르면 **같은 버퍼**가 온다. 그 자체는 맞다 —
   * 상수는 읽기만 하니까. 문제는 상수가 아닌 것이 그 문을 지날 때다:
   *
   * - `Adam` 의 m·v 가 크기 1 파라미터에서 같은 버퍼가 되어, WebGPU 가 "writable
   *   storage buffer aliasing" 으로 **명령 버퍼째** 무효로 만든다
   * - `SGD` 의 모멘텀 버퍼가 프로그램 전체가 쓰는 0 상수를 덮어쓴다
   * - `nn.PReLU()` 는 기본이 파라미터 하나라 가중치가 곧 전역 0.25 상수이고,
   *   옵티마이저가 학습 중에 그것을 고친다
   * - `BatchNorm(1)` 의 이동 통계가 전역 0·1 상수를 덮어쓴다
   *
   * 앞의 하나만 예외로 터지고 나머지는 **조용히 틀린다.** 그래서 파라미터·옵티마이저
   * 상태·이동 통계는 값이 무엇이든 여기로 온다.
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
  static eye(n: number): Tensor {
    const data = new Float32Array(n * n);
    for (let i = 0; i < n; i++) data[i * n + i] = 1;
    return Tensor.from(data, [n, n]);
  }

  /** `0` 부터 `n-1` 까지. */
  static arange(n: number): Tensor {
    const data = new Float32Array(n);
    for (let i = 0; i < n; i++) data[i] = i;
    return Tensor.from(data, [n]);
  }

  /** 양끝을 포함해 고르게 나눈 값들. */
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

  /** 삼각창. 가운데가 1 이고 양끝이 0 이다. */
  static bartlettWindow(n: number, periodic = true): Tensor {
    return Tensor.window(n, periodic, (k, total) =>
      1 - Math.abs((2 * k) / (total - 1) - 1));
  }

  static hannWindow(n: number, periodic = true): Tensor {
    return Tensor.window(n, periodic, (k, total) =>
      0.5 - 0.5 * Math.cos((2 * Math.PI * k) / (total - 1)));
  }

  /** `alpha - beta·cos`. torch 의 기본은 0.54/0.46 이다. */
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

  /** `I₀(β√(1-((k-h)/h)²)) / I₀(β)`. torch 의 기본 `beta` 는 12.0 이다. */
  static kaiserWindow(n: number, periodic = true, beta = 12.0): Tensor {
    return Tensor.window(n, periodic, (k, total) => {
      const half = (total - 1) / 2;
      const r = (k - half) / half;
      return besselI0(beta * Math.sqrt(Math.max(0, 1 - r * r))) / besselI0(beta);
    });
  }

  /**
   * `[0, 1)` 균등분포. `torch.rand` 자리이고 **쓰는 사람이 쓸 문은 이쪽이다.**
   *
   * **호스트에서 만들어 한 번에 올린다.** 그래야 `randn`·`randint`·`randperm` 과 같은
   * 줄기에서 나오고 `manualSeed` 하나가 넷을 함께 되돌린다.
   *
   * 난수가 나오는 문이 하나 더 있다 — `Tensor.uniform` 은 GPU 커널이고 dropout
   * 계수기를 쓴다. 그쪽은 **활성값만 한 표본을 스텝마다 뽑는 자리**(dropout·`rrelu`·
   * gumbel)의 것이라 업로드를 피하려고 따로 있다. 둘 다 `manualSeed` 가 되돌리지만
   * **같은 줄기는 아니다.**
   */
  static rand(shape: readonly number[]): Tensor {
    const data = new Float32Array(numel(shape));
    for (let i = 0; i < data.length; i++) data[i] = uniform();
    return Tensor.from(data, shape);
  }

  /** 표준정규분포. `torch.randn` 자리다. */
  static randn(shape: readonly number[]): Tensor {
    const data = new Float32Array(numel(shape));
    for (let i = 0; i < data.length; i++) data[i] = gauss();
    return Tensor.from(data, shape);
  }

  /**
   * `[low, high)` 의 정수. **위끝은 안 들어간다** — torch 와 같다.
   *
   * 형은 `int64` 로 붙지만 값은 float32 버퍼에 있다(`dtype.ts`). 2^24 를 넘는
   * 정수는 그 안에서 이미 못 세므로 거기서 멈춘다 — 조용히 반올림되느니 낫다.
   */
  static randint(low: number, high: number, shape: readonly number[]): Tensor {
    if (!(high > low)) {
      throw new RuntimeError(`random_ expects 'from' to be less than 'to', but got from=${low} >= to=${high}`);
    }
    if (Math.max(Math.abs(low), Math.abs(high)) > EXACT_INT_LIMIT) {
      throw new RuntimeError(
        `randint 의 범위가 float32 로 셀 수 있는 한계(${EXACT_INT_LIMIT})를 넘는다 — ` +
          "값이 조용히 반올림된다.",
      );
    }
    const span = high - low;
    const data = new Float32Array(numel(shape));
    for (let i = 0; i < data.length; i++) data[i] = low + Math.floor(uniform() * span);
    return Tensor.from(data, shape, { dtype: "int64" });
  }

  /** `0..n-1` 을 섞은 것. `torch.randperm` 자리다. Fisher–Yates. */
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
   * 모든 짝. **하나만 주면 그냥 그것이다**(실측) — 1차원으로 남는다.
   *
   * 값을 안 읽는다. 어느 칸이 어디로 가는지는 **개수만으로** 정해지므로 번호표로
   * 뽑아 모은다 — 읽었으면 이것도 비동기가 됐다.
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

  /** `r` 개씩 고른 조합. **순서는 없고**, 중복 허용이 따로 있다. */
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
   * 판데르몬드 행렬. **기본은 차수가 줄어드는 쪽이다** — 마지막 열이 1 이다(실측).
   *
   * 거듭제곱을 `pow` 로 짜지 않는다. WGSL 의 `pow` 는 `exp2(y·log2(x))` 라 **밑이
   * 음수면 답이 없고**, 판데르몬드는 음수 입력이 예사다. 누적곱으로 세면 정확하다.
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

  /** `torch.rand_like`. 모양만 빌린다 — 값도 형도 안 물려받는다. */
  randLike(): Tensor {
    return Tensor.rand(this.shape);
  }

  /** `torch.randn_like`. */
  randnLike(): Tensor {
    return Tensor.randn(this.shape);
  }

  // ── 원소별 ────────────────────────────────────────────────────────────

  unary(name: string): Tensor {
    if (!hasUnary(name)) throw new Error(`모르는 단항 연산: ${name}`);
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
   * @param dtype 결과의 형. 안 주면 산술 승격 규칙을 따른다. 비교 연산처럼 형이
   *   입력과 무관한 것은 여기서 못 박는다.
   */
  binary(name: string, other: Tensor, dtype?: DType): Tensor {
    const spec = BINARY[name];
    if (!spec) throw new Error(`모르는 이항 연산: ${name}`);
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
        `complex64 에는 ${name} 이(가) 아직 없다 — 지금 되는 것은 ` +
          "add·sub·mul·div 다.",
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
   * @param alpha 상대 쪽에 먼저 곱하는 값 — `this + alpha·other` 다.
   *
   * torch 가 주는 인자인데 여기 없었다. 없으면 부르는 쪽이 `x.add(y.mul(a))` 로
   * 풀어 쓰게 되고 답은 같은데, **제자리 판(`add_`)에는 이미 있어서** 두 이름이
   * 서로 다른 것을 받고 있었다.
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
   * @param roundingMode `null` 이면 참나눗셈, `"trunc"`·`"floor"` 는 정수 쪽.
   *
   * **형이 갈린다.** 참나눗셈은 언제나 실수인데, 자르거나 내리면 **입력의 형으로
   * 돌아온다**(실측: `int64 / int64` 에 `trunc` 를 주면 int64 다). 값만 맞추고 형을
   * 실수로 두면 그 뒤 색인이 정수를 요구하는 자리에서 갈린다.
   */
  div(other: Tensor, roundingMode: "trunc" | "floor" | null = null): Tensor {
    const out = this.binary("div", other);
    if (roundingMode === null) return out;
    const rounded = roundingMode === "floor" ? out.floor() : out.trunc();
    const kind = resultDType("mul", this.dtype, other.dtype);
    return kind === "float32" ? rounded : rounded.to(kind);
  }

  // ── 행렬곱 ────────────────────────────────────────────────────────────

  /** 2차원만. 배치 행렬곱은 T1 이다 — 없는 기능이 틀린 답보다 낫다. */
  mm(other: Tensor): Tensor {
    if (this.shape.length !== 2 || other.shape.length !== 2) {
      throw new Error(
        `mm 은 2차원끼리다: [${this.shape}] × [${other.shape}]. ` +
          "배치는 아직 없다.",
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

  /** 2차원 전치. 지금은 실제로 옮겨 담는다 — 뷰는 T1 이다. */
  transpose(): Tensor {
    if (this.shape.length !== 2) {
      throw new Error(`transpose 는 아직 2차원만이다: [${this.shape}]`);
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
  sum(): Tensor {
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
      throw new Error(`축이 범위를 벗어났다: ${dim} (랭크 ${rank})`);
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
    this.needsFloat("mean 은 실수에만 있습니다", "mean(): could not infer output dtype. Input dtype must be either a floating point or complex dtype");
    const count = dim === undefined
      ? this.size
      : (this.shape[dim < 0 ? dim + this.shape.length : dim] ?? 1);
    const total = dim === undefined ? this.sum() : this.sumDim(dim, keepdim);
    return total.div(Tensor.full([], count));
  }

  /**
   * 그래프에서 뗀 사본. 버퍼는 **공유한다** — 값을 읽기만 할 자리에 쓴다.
   *
   * `logsumexp` 가 최대값을 뗄 때 쓴다. 안 떼면 `m` 이 자기 기울기를 갖고, 수식상
   * 그 몫이 정확히 상쇄되긴 하지만 부동소수에서는 큰 것끼리 빼는 꼴이 된다.
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
   * **같은 텐서**에서 그래프를 끊는다. torch 의 `detach_` 다.
   *
   * `detach()` 는 새것을 내므로 원본은 여전히 위쪽에 붙어 있다 — 그 둘을 같은 것으로
   * 보면 `y.detach_()` 뒤에도 `y` 를 지나 역전파가 계속 흐른다.
   */
  detach_(): Tensor {
    this.requiresGrad = false;
    this.parents = [];
    this.backwardFn = null;
    return this;
  }

  /**
   * `log(Σ exp(x))`. **최대값을 빼고 계산한다** — 그냥 쓰면 x 가 89 를 넘는 순간
   * float32 의 exp 가 inf 가 되고 그 뒤가 전부 inf 다.
   *
   * 조립으로 둔다. 역방향은 이미 있는 연산들의 미분에서 나오므로 새 미분식을 손으로
   * 쓰지 않는다 — 그 자리가 이번 주에 가장 자주 틀린 자리였다.
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
   * 누적 `logsumexp`. **넘치지 않게** 센다 — 축의 최대값을 빼고 더한 뒤 되돌린다.
   *
   * `logsumexp` 와 같은 자리로 조립한다. 최대값을 떼 두었으므로 역방향은 조립에서
   * 그대로 나오고, 손으로 쓴 미분식이 하나 줄었다.
   */
  logcumsumexp(dim: number): Tensor {
    const big = this.amax(dim, true).detach();
    return this.sub(big).exp().cumsum(dim).log().add(big);
  }

  /**
   * 다변량 로그감마. `log Γ_p(x) = p(p−1)/4 · log π + Σᵢ log Γ(x + (1−i)/2)`.
   *
   * `p` 가 1 이면 `lgamma` 와 같다 — 그 값으로만 물으면 합이 도는지 안 보인다.
   */
  mvlgamma(p: number): Tensor {
    let out = this.lgamma();
    for (let i = 2; i <= p; i++) {
      out = out.add(this.add(Tensor.full([], (1 - i) / 2)).lgamma());
    }
    return out.add(Tensor.full([], (p * (p - 1) / 4) * Math.log(Math.PI)));
  }

  /**
   * 가수와 지수. `x = 가수 × 2^지수` 이고 가수는 [0.5, 1) 이다.
   *
   * torch 는 지수를 int32 로 내는데 여기 저장은 f32 하나뿐이라 값으로만 맞춘다.
   */
  frexp(): { mantissa: Tensor; exponent: Tensor } {
    return { mantissa: this.frexpMantissa(), exponent: this.frexpExponent() };
  }

  /**
   * 같은 꼴을 한 값으로 채운 **새** 텐서. **제자리가 아니다** — torch 의 `fill` 이
   * 그렇고, 이름이 한 글자 다른 `fill_` 과 하는 일이 다르다(실측).
   */
  fillWith(value: number): Tensor {
    return Tensor.full(this.shape, value);
  }

  /**
   * `‖x − y‖_p`. 조립이라 역방향이 저절로 따라온다.
   *
   * **`p` 를 오래 안 받았다** — 언제나 L2 였고, `dist(a, b, 3)` 이 그럴듯한 크기의
   * 다른 값을 냈다. 코어에도 같은 자리가 있었고 둘 다 값 대조로만 드러났다.
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

  /** NaN 을 0 으로 보고 더한다. NaN 자리로는 기울기가 안 간다. */
  nansum(dim?: number, keepdim = false, dtype?: DType): Tensor {
    if (dtype !== undefined) return this.castFirst(dtype).nansum(dim, keepdim).to(dtype);
    const clean = this.unary("nanToZero");
    return dim === undefined ? clean.sum() : clean.sumDim(dim, keepdim);
  }

  /** NaN 을 빼고 평균낸다. **개수도 NaN 을 빼고 센다** — 그것이 mean 과 다른 점이다. */
  nanmean(dim?: number, keepdim = false): Tensor {
    const total = this.nansum(dim, keepdim);
    const present = this.unary("notNan");
    const count = (dim === undefined
      ? present.sum()
      : present.sumDim(dim, keepdim)).detach();
    return total.div(count);
  }

  /** 크기 1 인 축을 끼워 넣는다. */
  unsqueeze(dim: number): Tensor {
    const rank = this.shape.length;
    const axis = dim < 0 ? dim + rank + 1 : dim;
    const shape = [...this.shape];
    shape.splice(axis, 0, 1);
    return this.reshape(shape);
  }

  /** 크기 1 인 축을 **전부** 뺀다. 인자 없는 `squeeze()` 다. */
  squeezeAll(): Tensor {
    return this.reshape(this.shape.filter((d) => d !== 1));
  }

  /** 값이 같은 새 텐서. 그래프는 이어진다 — 그것이 `detach` 와 다른 점이다. */
  clone(): Tensor {
    return this.unary("positive");
  }

  /**
   * 축 하나를 따라 이어 붙인다.
   *
   * **새 커널이 없다.** 각자를 상대 크기만큼 덧대고 더하면 된다 — 덧댄 자리는 0 이고
   * 겹치지 않으므로 합이 곧 이어 붙인 것이다. 역방향도 `pad` 의 것이 그대로 쓰인다.
   * 메모리를 두 배 쓰지만, 손으로 쓴 역방향 하나를 안 만드는 값이 더 크다.
   */
  static cat(parts: readonly Tensor[], dim = 0): Tensor {
    if (parts.length === 0) throw new Error("cat 에 줄 것이 없다");
    const first = parts[0];
    if (!first) throw new Error("cat 에 줄 것이 없다");
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
    if (!acc) throw new Error("cat 에 줄 것이 없다");
    return acc;
  }

  /** 새 축을 만들어 쌓는다. `cat` 에 축을 하나 끼워 넣은 것과 같다. */
  static stack(parts: readonly Tensor[], dim = 0): Tensor {
    return Tensor.cat(parts.map((p) => p.unsqueeze(dim)), dim);
  }

  /**
   * 분산. **torch 의 기본은 불편추정(n-1 로 나눔)이다** — `correction=0` 으로 두면
   * 값이 미묘하게 작아지고, 그것이 정규화 층에서 조용히 갈리는 자리가 된다.
   */
  variance(correction = 1): Tensor {
    // **torch 가 멈추는 자리에서 멈춘다**(실측). 나눗셈·제곱근이 정수 칸에
    // 답이 안 들어간다 — numpy 처럼 조용히 실수로 올리면 그 코드가 진짜
    // torch 에서 깨진다.
    this.needsFloat("variance 는 실수에만 있습니다", "std and var only support floating point and complex dtypes");
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
    this.needsFloat("std 는 실수에만 있습니다", "std and var only support floating point and complex dtypes");
    return this.variance(correction).sqrt();
  }

  /** 크기 1 인 축을 뺀다. */
  squeeze(dim: number): Tensor {
    const rank = this.shape.length;
    const axis = dim < 0 ? dim + rank : dim;
    if (this.shape[axis] !== 1) {
      throw new Error(`축 ${dim} 의 크기가 1 이 아니다: [${this.shape}]`);
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
          dev().pipeline(`gb:${key}|${inSize}`, () => gatherBackward(rules, offset, inSize)),
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
   * 같은 버퍼를 다른 모양으로 본다. 원소 순서가 안 바뀌므로 커널이 필요 없다.
   *
   * **`-1` 은 "나머지" 다.** torch 코드가 `x.reshape(-1)`·`x.view(n, -1)` 로 늘 쓰는
   * 꼴인데 여기 없었다 — 곱해서 크기를 맞추는 검사에 걸려 `shape '[-1]' is invalid`
   * 로 멈췄다. 한 자리에만 쓸 수 있고, 나머지로 나누어떨어져야 한다.
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

  /** 축 하나를 여러 축으로 편다. */
  unflatten(dim: number, sizes: readonly number[]): Tensor {
    const rank = this.shape.length;
    const axis = dim < 0 ? dim + rank : dim;
    const shape = [...this.shape.slice(0, axis), ...sizes, ...this.shape.slice(axis + 1)];
    return this.reshape(shape);
  }

  /** 적어도 2차원으로. 이미 2차원 이상이면 그대로. */
  atleast2d(): Tensor {
    if (this.shape.length >= 2) return this;
    if (this.shape.length === 1) return this.reshape([1, this.shape[0] ?? 0]);
    return this.reshape([1, 1]);
  }

  /**
   * 크기 1 인 축을 늘린다. `-1` 은 "그대로 두라"는 뜻이다.
   *
   * 앞에 축을 더 붙일 수도 있다 — 그 축들은 걸음이 0 이라 **복제하지 않는다.**
   */
  expand(...sizes: number[]): Tensor {
    const rank = sizes.length;
    if (rank < this.shape.length) {
      throw new Error(`expand 는 축을 못 줄인다: [${this.shape}] → [${sizes}]`);
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
        throw new Error(`축 ${i} 는 ${dim} 이라 ${want} 로 못 늘린다.`);
      }
      const stride = src >= 0 && dim !== 1 ? (own[src] ?? 1) : 0;
      rules.push({ size, stride, kind: "lin", wrap: size });
      outShape.push(size);
    }
    return this.viewAs(rules, 0, outShape, "ExpandBackward0");
  }

  /** 축마다 정수 배로 되풀이한다. `expand` 와 달리 실제로 여러 벌이 된다. */
  repeat(...times: number[]): Tensor {
    const rank = times.length;
    if (rank < this.shape.length) {
      throw new Error(`repeat 는 축을 못 줄인다: [${this.shape}] → [${times}]`);
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

  /** 축 둘을 맞바꾼다. `swapdims` 와 같다 — torch 가 이름을 둘 다 갖는다. */
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

  /** 축 하나에서 한 자리를 고르고 그 축을 없앤다. */
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
   * 대각선. `offset` 이 양수면 위쪽, 음수면 아래쪽 대각선이다.
   *
   * **어느 두 축을 볼지가 갈린다.** `torch.diagonal` 은 앞의 두 축(`0, 1`)이고
   * `torch.linalg.diagonal` 은 마지막 두 축(`-2, -1`)이다 — 3 차원을 주면 `(2,3,4)`
   * 가 각각 `(4,2)` 와 `(2,3)` 으로 갈린다. 이름이 비슷해 같은 것으로 읽기 쉬운데
   * 모양부터 다르다. 그래서 축을 인자로 받는다.
   *
   * **뽑은 축은 뒤로 간다.** 남은 축이 앞이고 대각선이 마지막이다 — torch 가 그렇다.
   */
  diagonal(offset = 0, dim1 = 0, dim2 = 1): Tensor {
    const rank = this.shape.length;
    if (rank < 2) {
      throw new Error(`diagonal 은 2차원 이상이다: [${this.shape}]`);
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
   * 벡터를 대각선에 놓은 정사각 행렬.
   *
   * @param offset 어느 대각선인가. 0 이 아니면 **행렬이 그만큼 커진다** —
   *   `n+|offset|` 변이고, 그래서 커널을 다시 부르는 대신 큰 판에 넣고 옮긴다.
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

  /** 축 하나를 거꾸로. */
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
   * 2차원 평면 안에서 90° 씩 돌린다.
   *
   * `k=1` 이면 `out[i][j] = in[j][C-1-i]` 다 — 축을 바꾸면서 한쪽을 뒤집는 것이라,
   * 규칙 표로 그대로 적힌다.
   */
  rot90(k = 1): Tensor {
    if (this.shape.length !== 2) {
      throw new Error(`rot90 은 아직 2차원만이다: [${this.shape}]`);
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
   * 미끄러지는 창. 걸음이 창 크기보다 작으면 창끼리 겹친다.
   *
   * **겹치면 역방향에서 쌓인다** — 길이 5 를 `unfold(0, 3, 1)` 로 펴면 기울기가
   * `[1,2,3,2,1]` 이다. 안 더하면 전부 1 이 되고, 값 검사만으로는 안 걸린다.
   */
  unfold(dim: number, size: number, step: number): Tensor {
    const rank = this.shape.length;
    const axis = dim < 0 ? dim + rank : dim;
    const own = this.strides();
    const axisSize = this.shape[axis] ?? 0;
    const windows = Math.floor((axisSize - size) / step) + 1;
    if (windows < 1) {
      throw new Error(`창 ${size}, 걸음 ${step} 로는 길이 ${axisSize} 에서 창이 안 나온다.`);
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

  /** 축 하나를 같은 크기 조각으로 나눈다. 조각마다 새 텐서다. */
  split(dim: number, parts: number): Tensor[] {
    const rank = this.shape.length;
    const axis = dim < 0 ? dim + rank : dim;
    const axisSize = this.shape[axis] ?? 0;
    if (axisSize % parts !== 0) {
      throw new Error(`축 ${dim} 의 크기 ${axisSize} 는 ${parts} 로 안 나뉜다.`);
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
   * **나머지를 앞에서부터 나눠 갖는다.** 10 을 4 로 쪼개면 3·3·2·2 다(실측).
   *
   * `chunk` 와 다르다 — 그쪽은 앞을 크게 채우고 마지막이 남는 것을 받는다. 나눠
   * 떨어지는 크기로만 재면 두 함수가 같아 보인다.
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

  /** 조각 **크기 목록**으로 쪼갠다. `tensorSplit` 은 자르는 **자리**를 받는다. */
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

  /** 축을 원하는 자리로 옮긴다. `swapaxes` 와 달리 나머지 순서를 지킨다. */
  movedim(src: number, dst: number): Tensor {
    const rank = this.shape.length;
    const from = src < 0 ? src + rank : src;
    const to = dst < 0 ? dst + rank : dst;
    const order = [...Array(rank).keys()].filter((d) => d !== from);
    order.splice(to, 0, from);
    return this.permute(order);
  }

  /** 축 순서를 통째로 바꾼다. */
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

  /** 축 하나에서 `start` 부터 `length` 개만. */
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
   * 축 하나를 자리이동. **끝에서 빠진 것이 앞으로 돌아온다.**
   *
   * `out[i] = in[(i - shift) mod n]` 이라, 규칙 표의 `mod` 에 자리이동을 얹으면 된다.
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

  /** `repeat` 과 같은 일이되 torch 가 이름을 둘 다 갖는다. */
  tile(...times: number[]): Tensor {
    return this.repeat(...times);
  }

  /** 축 하나를 크기로 나눈다. `split` 은 조각 **크기**, `chunk` 는 조각 **수**다. */
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

  /** 축 하나를 따라 낱개로 뜯는다. 그 축이 사라진다. */
  unbind(dim = 0): Tensor[] {
    const rank = this.shape.length;
    const axis = dim < 0 ? dim + rank : dim;
    const size = this.shape[axis] ?? 0;
    return Array.from({ length: size }, (_, i) => this.select(axis, i));
  }

  /** 아래 삼각. `diagonal` 위쪽을 0 으로 만든다. */
  tril(diagonal = 0): Tensor {
    return this.triangleAs(true, diagonal);
  }

  triu(diagonal = 0): Tensor {
    return this.triangleAs(false, diagonal);
  }

  private triangleAs(lower: boolean, diagonal: number): Tensor {
    if (this.shape.length !== 2) {
      throw new Error(`tril/triu 는 2차원이다: [${this.shape}]`);
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

  /** 대각선의 합. */
  trace(): Tensor {
    return this.diagonal().sum();
  }

  /**
   * 2차원이면 대각선을 뽑고, 1차원이면 대각선에 놓는다 — torch 의 `diag` 다.
   *
   * @param diagonal 어느 대각선인가. 양수는 위쪽, 음수는 아래쪽.
   */
  diag(diagonal = 0): Tensor {
    return this.shape.length === 2
      ? this.diagonal(diagonal)
      : this.diagflat(diagonal);
  }

  /** 축 하나를 누적한다. */
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

  /** 축 하나를 색인 텐서가 가리키는 대로 고른다. 색인은 float32 에 담겨 온다. */
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
   * 대괄호 자리. `x[...]` 를 옮겨 적는 문 하나다 — 문법과 이유는 `indexing.ts`.
   *
   * ```ts
   * x.at(0)                     // x[0]           축이 사라진다
   * x.at([null, 1])             // x[:, 1]        null 이 파이썬의 `:` 다
   * x.at(slice(1, 3))           // x[1:3]         축이 남는다
   * x.at([0, slice(1, 3)])      // x[0, 1:3]
   * x.at(slice(null, null, 2))  // x[::2]
   * x.at([[0, 2]])              // x[[0, 2]]      대괄호 둘 — numpy 와 같은 모양
   * x.at(idx)                   // x[idx]         int64 텐서
   * ```
   *
   * **맨 바깥 배열은 언제나 축 목록이다.** 적게 주면 남은 축은 통째로 온다.
   *
   * ## 값은 여기서 안 만든다
   *
   * 전부 있는 메서드로 넘긴다 — 정수는 `select`, 이어진 구간은 `narrow`, 걸음이
   * 있거나 번호표면 `indexSelect`. 새 커널이 없으므로 **골든이 이미 그 값들을
   * 지키고 있다.** 이 메서드가 지는 책임은 값이 아니라 **어느 문으로 보내는가**다.
   *
   * ## 참·거짓 마스크는 안 받는다
   *
   * `x[mask]` 는 `await x.maskedSelect(mask)` 로 남는다. 결과의 길이가 **값에 달려
   * 있어서** GPU 에서 한 번 읽어야 알 수 있고, 그것 하나 때문에 `at()` 을 비동기로
   * 만들면 나머지 모든 쓰임이 이유 없이 `await` 를 달게 된다. `unique`·`nonzero`·
   * `bincount` 가 비동기인 것과 같은 이유이고, 같은 자리에 두는 편이 낫다.
   */
  at(index: AtIndex | readonly AtIndex[]): Tensor {
    const list: readonly AtIndex[] = Array.isArray(index)
      && !isSlice(index) && !(index instanceof Tensor)
      ? index as readonly AtIndex[]
      : [index as AtIndex];
    if (list.length > this.shape.length) {
      throw new RuntimeError(
        `too many indices for tensor of dimension ${this.shape.length}: ` +
          `${list.length} 개를 줬다`,
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
  private needsFloat(korean: string, phrase: string): void {
    if (this.dtype !== "float32") {
      throw new RuntimeError(`${korean} — \`.to("float32")\` 를 먼저 불러라.\n(torch: ${phrase})`);
    }
  }

  /** 축 하나를 색인 **벡터**가 고른다. `gather` 와 달리 색인이 자리마다 다르지 않다. */
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

  /** 자리마다 되풀이한다. `[a,b]` 를 2 번씩이면 `[a,a,b,b]` 다 — `tile` 과 다르다. */
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
   * 이웃 차. `n` 번 되풀이하면 그만큼 짧아진다.
   *
   * @param prepend 차를 구하기 **전에** 앞에 이어 붙일 것.
   * @param append 뒤에 이어 붙일 것.
   *
   * 하나를 붙이면 결과가 입력과 **같은 길이**가 된다 — 시계열에서 첫 칸을 잃지
   * 않으려고 쓰는 자리다. 붙이는 것이 차를 구한 뒤가 아니라 **전**이라는 것이 요점이고,
   * 뒤에 붙이면 마지막 차가 달라진다.
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
      if (len < 2) throw new Error(`축 ${dim} 가 짧아서 diff 를 더 못 한다.`);
      cur = cur.narrow(axis, 1, len - 1).sub(cur.narrow(axis, 0, len - 1));
    }
    return cur;
  }

  /** 참인 자리를 값으로 덮는다. **덮은 자리로는 기울기가 안 간다.** */
  maskedFill(mask: Tensor, value: number): Tensor {
    return Tensor.full(this.shape, value).where(mask, this);
  }

  /** 정수 거듭제곱. 지금은 곱셈을 되풀이한다 — 지수가 작을 때만 쓸 것이다. */
  matrixPower(k: number): Tensor {
    if (k < 1) throw new Error(`matrix_power 는 아직 1 이상만이다: ${k}`);
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

  /** 조건 자리마다 이쪽 아니면 저쪽. torch 의 메서드 형태는 `x.where(조건, 다른쪽)` 이다. */
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

  /** 전부 곱한다. */
  prod(dim?: number, keepdim = false, dtype?: DType): Tensor {
    if (dtype !== undefined) return this.castFirst(dtype).prod(dim, keepdim).to(dtype);
    if (dim === undefined) return this.flat().reduceOver("prod", 0, false);
    return this.reduceOver("prod", dim, keepdim);
  }

  /** L2 노름. */
  norm(): Tensor {
    // **torch 가 멈추는 자리에서 멈춘다**(실측). 나눗셈·제곱근이 정수 칸에
    // 답이 안 들어간다 — numpy 처럼 조용히 실수로 올리면 그 코드가 진짜
    // torch 에서 깨진다.
    this.needsFloat("norm 은 실수에만 있습니다", "linalg.vector_norm: Expected a floating point or complex tensor as input");
    return this.square().sum().sqrt();
  }

  /** 두 벡터의 안쪽 곱. */
  dot(other: Tensor): Tensor {
    return this.mul(other).sum();
  }

  /** 두 벡터의 바깥 곱. 브로드캐스팅으로 나온다 — 새 커널이 필요 없다. */
  outer(other: Tensor): Tensor {
    return this.reshape([this.size, 1]).mul(other.reshape([1, other.size]));
  }

  /**
   * 위아래로 자른다.
   *
   * **`maximum`·`minimum` 위에 얹으면 안 된다.** 그 둘은 동점에서 기울기를 반씩
   * 나누는데(torch 가 그렇다) `clamp` 는 경계에서 온전히 흘린다. 얹어 두었더니
   * `x` 가 정확히 경계에 앉은 자리에서만 기울기가 절반이 됐다.
   */
  clamp(low: number, high: number): Tensor {
    const lo = f32lit(low), hi = f32lit(high);
    return this.unary(unaryWith(`clamp<${lo},${hi}>`, () => ({
      fwd: `clamp(x, ${lo}, ${hi})`,
      bwd: `select(0.0, 1.0, x >= ${lo} && x <= ${hi})`,
    })));
  }

  /**
   * 상수 지수.
   *
   * **정수 지수는 곱셈으로 간다.** WGSL 의 `pow(x, y)` 는 `exp2(y·log2(x))` 라 밑이
   * 음수면 답이 없고, 실제로 `|x|` 를 쓴 것 같은 값이 나온다. 짝수 지수에서는 값이
   * 우연히 맞아서 `method::pow` 는 통과했고, `grad::pow2` 에서 부호가 뒤집힌 채로
   * 잡혔다 — 값은 맞고 기울기만 틀리는 그 종류다.
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

  /** `exp(x) / Σ exp(x)`. **최대값을 빼고 계산한다** — 안 그러면 큰 값에서 넘친다. */
  softmax(dim = 0): Tensor {
    const m = this.amax(dim, true).detach();
    const e = this.sub(m).exp();
    return e.div(e.sumDim(dim, true));
  }

  /**
   * `log(softmax(x))`. **`softmax` 를 구해 로그를 취하지 않는다** — 작은 확률에서
   * 0 이 되어 로그가 -inf 가 된다. 빼기로 바로 쓰면 그 자리가 없다.
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

  /** 최대·최소가 **어디에** 있는가. 동점이면 먼저 나온 자리다. */
  argmax(dim = 0, keepdim = false): Tensor {
    return this.liftAxis(this.argReduceOver("max", dim), dim, keepdim);
  }

  argmin(dim = 0, keepdim = false): Tensor {
    return this.liftAxis(this.argReduceOver("min", dim), dim, keepdim);
  }

  /**
   * 값과 번호를 함께. torch 의 `x.max(dim)` 이다.
   *
   * `amax` 와 `argmax` 로 따로 부르면 되기는 하는데, torch 코드가 쓰는 모양은
   * 이쪽이고 **둘이 같은 자리를 가리키는지**가 여기서만 확인된다. `amax` 는 동점을
   * 고르게 나누고 `argmax` 는 먼저 나온 자리 하나를 고르는데, 값을 번호로 다시
   * 뽑아 오면 나누는 일이 없다 — torch 의 `max(dim)` 이 그쪽이다.
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
   * 0 이 아닌 것의 개수. **축을 받는다** — 전체 축약만 되던 자리다.
   *
   * 축이 없으면 `x.countNonzero(1)` 이 인자를 조용히 버리고 스칼라를 내는데,
   * 그 스칼라는 어디에나 브로드캐스팅된다.
   */
  countNonzero(dim?: number): Tensor {
    const flags = this.binary("ne", Tensor.full([], 0));
    return dim === undefined ? flags.sum() : flags.sumDim(dim);
  }

  /** 전부 참인가 / 하나라도 참인가. 0/1 로 답한다. */
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

  /** 축 하나의 앞뒤에 상수를 덧댄다. 여러 축이면 축마다 부른다. */
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

  /** 창을 열로 편다. `(N, C, H, W)` → `(N, C·kh·kw, L)`. */
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

  /** 편 것을 되접는다. **겹친 자리는 더한다** — 그것이 이 함수의 뜻이다. */
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
   * `y[o] = x₁ᵀ·W[o]·x₂ + b[o]`. 가중치가 **세 축**이다.
   *
   * `einsum` 을 안 부른다 — 그쪽이 이 파일을 들여오므로 서로 물게 된다. 두 걸음으로
   * 나누면 이미 있는 것들로 적힌다: 먼저 `x₂` 를 `W` 에 태워 `(B, O, I)` 를 만들고,
   * 그 다음 `x₁` 과 마지막 축에서 접는다.
   */
  bilinear(other: Tensor, weight: Tensor, bias: Tensor | null = null): Tensor {
    const [o, i, j] = weight.shape as [number, number, number];
    const b = this.shape[0] ?? 1;
    const mixed = other.linear(weight.reshape([o * i, j])).reshape([b, o, i]);
    const out = this.reshape([b, 1, i]).mul(mixed).sumDim(2, false);
    return bias ? out.add(bias) : out;
  }

  /**
   * 이웃 채널로 나눈다.
   *
   * **창이 한쪽으로 치우쳐 있다.** 채널 `c` 의 창은 `[c − n//2, c + n − 1 − n//2]`
   * 이고 `size=2` 면 `{c−1, c}` 다 — 가운데를 잡으면 값이 한 칸씩 밀리는데 크기가
   * 같아서 모양으로는 안 보인다.
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
   * 0..1 사이 난수. `rrelu`·`gumbelSoftmax` 가 쓴다.
   *
   * `manualSeed` 가 잡는 그 씨앗을 쓴다 — 같은 씨앗에 같은 뽑기가 나와야 재현이 된다.
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
   * 축 하나에서 가장 큰 자리만 1 인 표.
   *
   * **동점을 `eq` 로 가르면 안 된다** — 같은 값이 둘이면 1 이 둘 나오고, 그러면
   * `gumbelSoftmax(hard=true)` 의 답이 one-hot 이 아니게 된다. 난수에서는 거의 안
   * 일어나지만 "거의" 는 보장이 아니다. 번호를 골라 그 자리에만 놓는다.
   */
  oneHotAlong(indices: Tensor, dim: number): Tensor {
    const shape = [...this.shape];
    shape[dim] = 1;
    // 번호를 자리로 펴는 일은 이미 `scatter` 가 한다 — 0 판 위에 1 을 놓는다.
    return Tensor.zeros(this.shape)
      .scatterSet(dim, indices.reshape(shape), Tensor.ones(shape));
  }

  /**
   * 음수 쪽 기울기를 뽑아 쓴다.
   *
   * **평가 모드에서는 가운데로 정해진다** — 기본값이면 `(1/8 + 1/3)/2 = 0.2292` 다.
   * 난수가 끼는 것은 학습 모드뿐이다.
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
   * 겹선형 확대.
   *
   * **`alignCorners` 가 값을 바꾼다.** 참이면 양 끝을 못 박고 그 사이를 고르게
   * 나누고, 거짓이면 칸의 가운데를 기준으로 잰다. `UpsamplingBilinear2d` 는 참이고
   * `Upsample(mode='bilinear')` 의 기본값은 거짓이라, 이름만 보고 별명으로 두면
   * 가장자리가 어긋난다 — 안쪽은 비슷해서 눈으로는 안 갈린다.
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
   * `(N, C·r², H, W)` → `(N, C, H·r, W·r)`. 채널을 잘라 공간에 심는다.
   *
   * **엇갈리는 순서가 값의 전부다.** 채널을 `(C, r, r)` 로 갈라 두 `r` 을 각각 `H` 와
   * `W` 뒤에 끼워 넣는다. 순서를 바꾸면 모양은 같고 그림만 뒤섞인다.
   */
  pixelShuffle(upscaleFactor: number): Tensor {
    const r = upscaleFactor;
    const [n, c, h, w] = this.shape as [number, number, number, number];
    return this.reshape([n, c / (r * r), r, r, h, w])
      .permute([0, 1, 4, 2, 5, 3])
      .reshape([n, c / (r * r), h * r, w * r]);
  }

  /** `pixelShuffle` 의 역. 공간을 잘라 채널에 쌓는다. */
  pixelUnshuffle(downscaleFactor: number): Tensor {
    const r = downscaleFactor;
    const [n, c, h, w] = this.shape as [number, number, number, number];
    return this.reshape([n, c, h / r, r, w / r, r])
      .permute([0, 1, 3, 5, 2, 4])
      .reshape([n, c * r * r, h / r, w / r]);
  }

  /**
   * 채널을 묶음으로 갈라 **엇갈려 다시 놓는다.**
   *
   * `[0,1,2,3]` 을 두 묶음으로 섞으면 `[0,2,1,3]` 이다 — 묶음별 합성곱 뒤에 정보가
   * 묶음 안에만 갇히는 것을 푸는 자리라, 엇갈리는 방향이 값의 전부다.
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
   * **원소가 아니라 채널을 떨군다.**
   *
   * 이름이 `dropout` 옆에 있어서 "N 차원용" 으로 읽기 쉬운데 하는 일이 다르다 —
   * 한 채널을 통째로 0 으로 만들거나 통째로 남긴다. 마스크를 `(N, C, 1, …)` 로 뽑아
   * 브로드캐스트하면 그 뜻이 그대로 적힌다.
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
   * SELU 와 함께 쓰는 dropout. **떨군 자리에 0 을 안 넣는다.**
   *
   * 음의 상수를 넣고 전체에 아핀 변환을 걸어 평균과 분산을 지킨다 — 0 을 넣으면
   * SELU 의 자기정규화가 깨지는데, 값이 그럴듯해서 학습이 도는 동안은 안 보인다.
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
   * `F.pad` 의 자리 — **마지막 축부터 (앞, 뒤) 순으로** 받는다.
   *
   * **짝의 개수와 랭크가 맞물린다.** 짝이 하나면 2·3 차원, 둘이면 3·4 차원, 셋이면
   * 4·5 차원이라야 한다 — torch 가 그 밖을 거절한다. 아무 랭크나 받으면 축을 잘못
   * 잡고도 통과한다.
   *
   * 상수는 이미 있는 커널이 짧아서 그쪽으로 보내고, 나머지 셋은 색인으로 간다.
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
   * 기울기가 완만한 왼쪽.
   *
   * `max(x, slope·x)` 로 두었더니 **x 가 정확히 0 인 자리에서 틀렸다.** 그 자리는 두
   * 항이 동점이고 `maximum` 은 동점에서 반씩 나누므로 (1 + slope)/2 가 나오는데,
   * torch 는 slope 를 준다 — `x > 0` 하나로 가르지 동점이라는 개념이 없다.
   */
  leakyRelu(slope = 0.01): Tensor {
    const s = f32lit(slope);
    return this.unary(unaryWith(`leakyRelu<${s}>`, () => ({
      fwd: `select(x * ${s}, x, x > 0.0)`,
      bwd: `select(${s}, 1.0, x > 0.0)`,
    })));
  }

  /**
   * ELU 와 달리 음수 쪽을 α 로 **나눈 뒤** 지수를 취한다.
   *
   * α=1 이면 ELU 와 같은 값이라, α 를 안 주고 재면 둘을 못 가른다.
   */
  celu(alpha = 1.0): Tensor {
    const a = f32lit(alpha);
    return this.unary(unaryWith(`celu<${a}>`, () => ({
      fwd: `select(${a} * (exp(x / ${a}) - 1.0), x, x > 0.0)`,
      bwd: `select(exp(x / ${a}), 1.0, x > 0.0)`,
    })));
  }

  /** |x| > λ 면 그대로, 아니면 0. **경계는 0 쪽이다**(`>` 이지 `>=` 가 아니다). */
  hardshrink(lambd = 0.5): Tensor {
    const l = f32lit(lambd);
    return this.unary(unaryWith(`hardshrink<${l}>`, () => ({
      fwd: `select(0.0, x, abs(x) > ${l})`,
      bwd: `select(0.0, 1.0, abs(x) > ${l})`,
    })));
  }

  /** λ 만큼 **원점 쪽으로 당긴다.** `hardshrink` 와 달리 값이 이어진다. */
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
   * (1/β)·log(1+e^{βx}). **βx 가 threshold 를 넘으면 그냥 x 다.**
   *
   * 그 갈래를 빼면 큰 입력에서 `exp` 가 넘치고, 그 뒤 기울기가 전부 NaN 이 된다.
   */
  softplus(beta = 1.0, threshold = 20.0): Tensor {
    const b = f32lit(beta);
    const t = f32lit(threshold);
    return this.unary(unaryWith(`softplus<${b},${t}>`, () => ({
      fwd: `select(log(1.0 + exp(-abs(${b} * x))) / ${b} + max(x, 0.0), x, ${b} * x > ${t})`,
      bwd: `select(1.0 / (1.0 + exp(-${b} * x)), 1.0, ${b} * x > ${t})`,
    })));
  }

  /** x > t 면 그대로, 아니면 `value`. **경계는 value 쪽이다.** */
  threshold(t: number, value: number): Tensor {
    const th = f32lit(t);
    const v = f32lit(value);
    return this.unary(unaryWith(`threshold<${th},${v}>`, () => ({
      fwd: `select(${v}, x, x > ${th})`,
      bwd: `select(0.0, 1.0, x > ${th})`,
    })));
  }

  /**
   * 음수 쪽 기울기가 **학습된다.**
   *
   * **정확히 0 은 음수 쪽이다.** 순방향은 어느 쪽으로 놓아도 0 이라 안 보이는데
   * 기울기는 갈린다 — torch 는 `x > 0` 일 때만 1 을 준다. 코어에서 `x < 0` 으로
   * 갈랐다가 그 한 점에서 최대차 3.75 를 냈다.
   *
   * 가중치가 학습되므로 이항으로 엮는다 — 상수로 구우면 기울기가 안 흐른다.
   */
  prelu(weight: Tensor): Tensor {
    // `gt` 의 기울기는 0 이다 — 가림막이 기울기를 나르면 안 된다.
    // **float 으로 옮긴다.** 비교는 bool 을 내고 bool 로는 빼기가 거절된다(torch 도
    // 그렇다). 가림막을 수로 쓸 것이므로 여기서 형을 맞춘다.
    const pos = this.binary("gt", Tensor.full([], 0)).to("float32");
    const neg = Tensor.full([], 1).sub(pos);
    return this.mul(pos).add(this.mul(weight).mul(neg));
  }

  /** 축을 반으로 갈라 `a · σ(b)`. 활성함수 중 유일하게 원소별이 아니다. */
  glu(dim = -1): Tensor {
    const rank = this.shape.length;
    const axis = dim < 0 ? dim + rank : dim;
    const n = this.shape[axis] ?? 0;
    if (n % 2 !== 0) {
      throw new Error(`glu 는 축 ${dim} 의 길이가 짝수여야 한다 (지금 ${n})`);
    }
    const half = n / 2;
    return this.narrow(axis, 0, half).mul(this.narrow(axis, half, half).sigmoid());
  }

  /** softmax(−x). **부호를 빠뜨리면 softmax 와 같아진다** — 값으로만 갈린다. */
  softmin(dim = -1): Tensor {
    return this.neg().softmax(dim);
  }

  /**
   * 자리를 골라 떨구고 **살아남은 값을 `1/(1-p)` 로 키운다.**
   *
   * 키우는 것이 요점이다 — 그래야 학습 때의 기댓값이 추론 때와 같고, 빼먹으면 두
   * 모드의 크기가 안 맞는다. `p=1` 은 따로 가른다: `1/(1-p)` 가 0 으로 나누기가 되어
   * NaN 이 나오고, NaN 은 자기 자신과도 달라서 그 뒤로 아무 비교도 통과 못 한다.
   *
   * **부를 때마다 다른 자리를 떨군다.** 씨앗을 하나 올려 가며 쓴다 — 한 번 뽑아
   * 캐시하면 매 스텝 같은 자리가 죽고, 그것은 dropout 이 아니다.
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

  /** dropout 이 쓰는 씨앗. 부를 때마다 오른다 — 같은 자리를 두 번 떨구지 않으려고. */
  static dropoutSeed = 1;

  /** 축을 따라 길이를 1 로. `eps` 는 0 벡터에서 나눗셈이 터지는 것을 막는다. */
  normalize(dim = 1, eps = 1e-12): Tensor {
    const len = this.square().sumDim(dim, true).sqrt();
    return this.div(len.binary("maximum", Tensor.full([], eps)));
  }

  /** 두 묶음의 방향이 얼마나 같은가. */
  cosineSimilarity(other: Tensor, dim = 1, eps = 1e-8): Tensor {
    const dotted = this.mul(other).sumDim(dim, false);
    const la = this.square().sumDim(dim, false).sqrt();
    const lb = other.square().sumDim(dim, false).sqrt();
    return dotted.div(la.mul(lb).binary("maximum", Tensor.full([], eps)));
  }

  /**
   * 절대 오차.
   *
   * **`reduction` 이 여기 없었다.** 드물게 쓰는 손실 열셋은 전부 받고 있었는데
   * 제일 많이 쓰는 넷이 안 받았다 — 나중에 쓴 것이 torch 서명을 따랐고 처음 쓴 것이
   * 안 고쳐졌다. 튜토리얼이 기본값만 쓰니 표도 안 물었다.
   */
  l1Loss(target: Tensor, reduction: Reduction = "mean"): Tensor {
    return this.sub(target).abs().reduceAs(reduction);
  }

  /**
   * 작을 때는 제곱, 클 때는 절대값. **원점에서 미분이 이어진다** — 그것이 이 손실을
   * 쓰는 이유이므로 `beta` 를 경계로 두 식을 붙인다.
   */
  /** 제곱 오차. */
  mseLoss(target: Tensor, reduction: Reduction = "mean"): Tensor {
    return this.sub(target).square().reduceAs(reduction);
  }

  /**
   * 로짓을 그대로 받는 이진 교차엔트로피.
   *
   * **`sigmoid` 를 먼저 구해서 로그를 취하지 않는다.** 그러면 확신이 큰 자리에서
   * `log(0)` 이 되어 손실이 무한대가 된다. `max(x,0) − x·y + log(1+exp(−|x|))` 는
   * 같은 값을 넘침 없이 낸다 — 이 함수가 따로 있는 이유가 그것이다.
   */
  bceWithLogits(target: Tensor, reduction: Reduction = "mean"): Tensor {
    const zero = Tensor.full([], 0);
    const hinge = this.binary("maximum", zero);
    const stable = this.abs().neg().exp().unary("log1p");
    return hinge.sub(this.mul(target)).add(stable).reduceAs(reduction);
  }

  /**
   * 마지막 축들을 평균 0, 분산 1 로. **분산은 편향추정(n 으로 나눔)이다** —
   * torch 의 `layer_norm` 이 그렇고, `var()` 의 기본과 다르다.
   */
  layerNorm(dim = -1, eps = 1e-5): Tensor {
    const m = this.mean(dim, true);
    const centered = this.sub(m);
    const v = centered.square().mean(dim, true);
    return centered.div(v.binary("add", Tensor.full([], eps)).sqrt());
  }

  /**
   * 배치 축을 따라 정규화. 학습 모드 — 이동 통계를 안 쓰고 이 배치로 센다.
   *
   * `layer_norm` 과 접는 축만 다르다. 축이 다르면 접는 축을 바꾸면 되지, 함수를
   * 따로 세울 일이 아니다.
   */
  batchNorm(dim = 0, eps = 1e-5): Tensor {
    return this.layerNorm(dim, eps);
  }

  /**
   * 채널을 그룹으로 묶어 정규화.
   *
   * `layerNorm` 과 **접는 구간만** 다르다. 그룹 수가 1 이면 채널 전체가 한 묶음이라
   * LayerNorm 이고, 채널 수와 같으면 채널마다 따로라 InstanceNorm 이다 — 셋은 서로의
   * 특수한 경우이고 묶는 규칙이 틀리면 셋 중 둘이 같아진다.
   *
   * 묶는 구간을 마지막 축 하나로 눕혀 놓고 접는다. 그러면 `layerNorm` 의 식을 그대로
   * 쓰므로 정규화 식이 한 군데에만 있다.
   */
  groupNorm(numGroups: number, eps = 1e-5): Tensor {
    const N = this.shape[0] ?? 1;
    const C = this.shape[1] ?? 1;
    if (C % numGroups !== 0) {
      throw new Error(`채널 ${C} 를 ${numGroups} 그룹으로 못 나눈다`);
    }
    const inner = this.size / (N * numGroups);
    return this.reshape([N, numGroups, inner]).layerNorm(-1, eps).reshape(this.shape);
  }

  /** 표본마다·채널마다 따로. 그룹 수를 채널 수로 준 `groupNorm` 이다. */
  instanceNorm(eps = 1e-5): Tensor {
    return this.groupNorm(this.shape[1] ?? 1, eps);
  }

  /**
   * **평균을 안 뺀다.** 그것이 `layerNorm` 과의 유일한 차이다.
   *
   * 기본 eps 가 `1e-5` 가 아니라 f32 의 기계 엡실론이다 — torch 가 그렇다. 다른
   * 정규화 층에 맞춰 `1e-5` 로 적었더니 순방향은 허용 오차 안이고 **기울기만**
   * 최대차 2.26e-02 로 갈렸다. 분산이 작은 자리에서 증폭된다.
   */
  rmsNorm(dims = 1, eps = 1.1920928955078125e-7): Tensor {
    const rank = this.shape.length;
    const lead = this.shape.slice(0, rank - dims).reduce((a, b) => a * b, 1);
    const flat = this.reshape([lead, this.size / lead]);
    const v = flat.square().mean(-1, true);
    const out = flat.div(v.binary("add", Tensor.full([], eps)).sqrt());
    return out.reshape(this.shape);
  }

  /** `x @ Wᵀ`. torch 의 `F.linear` 는 가중치를 (출력, 입력) 으로 받는다. */
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
   * **`smoothL1Loss` 와 δ=1 에서만 같다.**
   *
   * 실제 관계는 `huber(δ) = δ · smoothL1(β=δ)` 다. 기본값으로만 재면 둘을 같은
   * 함수로 두고도 통과하므로, 골든이 δ 를 바꿔 묻는다.
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
   * `target · (log target − this)`. **이쪽은 이미 로그**여야 한다.
   *
   * **`reduction` 이 넷이다.** `mean` 은 원소 수로 나누고 `batchmean` 은 배치 크기로
   * 나눈다 — 수학적 정의에 맞는 것은 뒤쪽이고, torch 자신도 바꾸겠다고 경고한다.
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
   * 포아송 음의 로그가능도.
   *
   * **스털링 보정은 `target > 1` 일 때만 더한다** — 조건 없이 늘 더하면 target 이
   * 작은 자리에서만 틀린다(실측으로 확인했다).
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
   * 가우스 음의 로그가능도.
   *
   * **분산을 `eps` 로 자른다.** 안 자르면 0 으로 나눠 무한대가 된다.
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

  /** `max(0, −y·(x₁ − x₂) + margin)`. */
  marginRankingLoss(
    other: Tensor, target: Tensor, margin = 0.0, reduction: Reduction = "mean",
  ): Tensor {
    return target.neg().mul(this.sub(other))
      .binary("add", Tensor.full([], margin)).unary("relu").reduceAs(reduction);
  }

  /** `y=1` 이면 `1 − cos`, `y=−1` 이면 `max(0, cos − margin)`. */
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
   * `y=1` 이면 `x` 그대로, `y=−1` 이면 `max(0, margin − x)`.
   *
   * **둘로 가르는 것이 아니라 둘을 더한다.** torch 는 `y ≠ 1` 인 자리에 여백 항을,
   * `y ≠ −1` 인 자리에 `x` 를 놓고 **합한다** — ±1 에서는 한쪽만 켜져 평소 식과
   * 같지만 `y=0` 에서는 **둘 다** 켜진다. `y > 0` 으로 갈라 놓았더니 거기서 조용히
   * 갈렸고, `sign()` 은 0 을 만든다.
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

  /** `log(1 + e^{−y·x})`. `softplus` 로 가면 큰 값에서 안 넘친다. */
  softMarginLoss(target: Tensor, reduction: Reduction = "mean"): Tensor {
    return target.mul(this).neg().softplus().reduceAs(reduction);
  }

  /**
   * 짝지어진 두 줄 사이의 거리.
   *
   * **`eps` 는 결과가 아니라 차에 더한다.** `p=1` 로 차가 정확히 1.0 인 자리를
   * 물으면 1.0000020 이 나온다(= 1 + 2·1e-6) — 결과에 더한다고 읽으면 1.000001 이
   * 되어 자릿수 하나가 갈린다. 실측으로 확인했다.
   */
  pairwiseDistance(other: Tensor, p = 2.0, eps = 1e-6, keepdim = false): Tensor {
    const diff = this.sub(other).binary("add", Tensor.full([], eps));
    const out = diff.vectorNorm(p, this.shape.length - 1);
    return keepdim ? out.reshape([...out.shape, 1]) : out;
  }

  /** 한 묶음 안의 **모든 짝** 사이 거리. 위 삼각만 준다. */
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

  /** `max(0, d(a,p) − d(a,n) + margin)`. */
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

  /** 자리마다 독립인 이진 분류를 **반 전체로 평균**한다. */
  multilabelSoftMarginLoss(
    target: Tensor, reduction: Reduction = "mean",
  ): Tensor {
    const each = target.mul(this.logsigmoid())
      .add(Tensor.full([], 1).sub(target).mul(this.neg().logsigmoid()));
    const dim = this.shape.length - 1;
    return each.neg().mean(dim).reduceAs(reduction);
  }

  /**
   * 정답 자리와 나머지 사이의 여백.
   *
   * **반의 개수로 나눈다** — 견준 짝의 수가 아니다. 정답 자리도 분모에 들어간다는
   * 뜻이고, 짝의 수로 나누면 반이 셋일 때 3/2 배가 나온다.
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
   * **표적이 자리 목록이고 −1 이 끝을 뜻한다.**
   *
   * `[3, 0, -1, 1]` 은 "3 번과 0 번이 정답" 이라는 뜻이고 뒤의 1 은 안 읽는다. 그
   * 규약을 안 지키면 −1 을 반의 하나로 세거나 끝난 뒤를 계속 읽는다.
   *
   * 어디까지 읽을지는 **누적합**으로 정한다 — `−1` 을 만난 뒤로는 전부 꺼진다. 그
   * 뒤 정답 자리를 0/1 로 흩어 담으면 CPU 로 읽어 올 일이 없다. 덮어쓰기가 아니라
   * **모아 더하기**로 담는 이유는, 끝난 뒤의 자리가 clamp 되어 앞의 1 을 지울 수
   * 있기 때문이다.
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
   * 배치 행렬곱. `(B, N, K) × (B, K, M)`.
   *
   * 배치를 풀어 `mm` 을 되풀이한 뒤 쌓는다 — 배치 커널을 따로 두면 `mm` 과 두 벌을
   * 고치게 된다. 배치가 커지면 그때 커널을 세울 자리다.
   */
  bmm(other: Tensor): Tensor {
    if (this.shape.length !== 3 || other.shape.length !== 3) {
      throw new Error(`bmm 은 3차원끼리다: [${this.shape}] × [${other.shape}]`);
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

  /** `β·this + α·(mat1 @ mat2)`. `this` 는 결과 모양으로 퍼진다. */
  addmm(mat1: Tensor, mat2: Tensor, beta = 1, alpha = 1): Tensor {
    return this.blend(mat1.mm(mat2), beta, alpha);
  }

  /**
   * **배치를 합친다** — 곱한 뒤 배치 축을 더해 2차원을 낸다.
   *
   * `baddbmm` 과 이름이 한 글자 다르고 결과 차수가 다르다. 배치가 1 이면 둘이 같아
   * 보이므로 케이스는 배치를 둘 이상으로 둔다.
   */
  addbmm(batch1: Tensor, batch2: Tensor, beta = 1, alpha = 1): Tensor {
    return this.blend(batch1.bmm(batch2).sumDim(0), beta, alpha);
  }

  /** **배치를 지킨다.** `addbmm` 과 여기서 갈린다. */
  baddbmm(batch1: Tensor, batch2: Tensor, beta = 1, alpha = 1): Tensor {
    return this.blend(batch1.bmm(batch2), beta, alpha);
  }

  /** `β·this + α·(mat @ vec)`. 결과가 1차원이다. */
  addmv(mat: Tensor, vec: Tensor, beta = 1, alpha = 1): Tensor {
    return this.blend(mat.mv(vec), beta, alpha);
  }

  /** `β·this + α·(vec1 ⊗ vec2)`. 바깥곱이라 결과가 2차원이다. */
  addr(vec1: Tensor, vec2: Tensor, beta = 1, alpha = 1): Tensor {
    return this.blend(vec1.outer(vec2), beta, alpha);
  }

  /** `this + value·(t1 · t2)`. **`beta` 가 없다** — `this` 의 계수는 늘 1 이다. */
  addcmul(tensor1: Tensor, tensor2: Tensor, value = 1): Tensor {
    return this.blend(tensor1.mul(tensor2), 1, value);
  }

  /** `this + value·(t1 / t2)`. 옵티마이저가 갱신을 적을 때 쓰는 꼴이다. */
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
   * 나머지. **torch 의 `fmod` 는 피제수의 부호를 따른다** — 파이썬의 `%` 와 다르다.
   *
   * 기울기는 피제수로 1 이 흐른다. 제수 쪽은 계단이라 안 흐른다.
   */
  fmod(divisor: number): Tensor {
    const d = Tensor.full([], divisor);
    const q = this.div(d).unary("trunc").detach();
    return this.sub(q.binary("mul", d));
  }

  /**
   * 나머지 — **`fmod` 와 부호가 다르다.**
   *
   * torch 의 `remainder`(그리고 파이썬·torch 의 `%`)는 **제수**의 부호를 따르고
   * `fmod` 는 피제수의 부호를 따른다. `-7` 을 `3` 으로 나눈 나머지가 이쪽은 2 이고
   * 저쪽은 -1 이다. 둘 다 "나머지" 라고 불리는 것이 함정이고, JS 의 `%` 는 `fmod`
   * 쪽이라 그것을 그대로 쓰면 음수 입력에서만 조용히 갈린다.
   *
   * 잘라내기(`trunc`)냐 내림(`floor`)이냐 하나 차이다.
   */
  remainder(divisor: number): Tensor {
    const d = Tensor.full([], divisor);
    const q = this.div(d).unary("floor").detach();
    return this.sub(q.binary("mul", d));
  }

  /** 아래로만 자른다. torch 의 `clamp(min=…)` 이다. */
  clampMin(low: number): Tensor {
    const lo = f32lit(low);
    return this.unary(unaryWith(`clampMin<${lo}>`, () => ({
      fwd: `max(x, ${lo})`,
      bwd: `select(0.0, 1.0, x >= ${lo})`,
    })));
  }

  /** 위로만 자른다. torch 의 `clamp(max=…)` 이다. */
  clampMax(high: number): Tensor {
    const hi = f32lit(high);
    return this.unary(unaryWith(`clampMax<${hi}>`, () => ({
      fwd: `min(x, ${hi})`,
      bwd: `select(0.0, 1.0, x <= ${hi})`,
    })));
  }

  /**
   * 길이가 다른 것들을 한 배치에 담는다.
   *
   * 기본은 `(가장 긴 길이, 개수, …)` 이고, `batchFirst` 면 앞의 둘이 바뀐다.
   * **기울기가 각 조각으로 되돌아간다** — 덧댄 자리는 0 이므로 `pad` 의 역방향이
   * 잘라내 준다.
   */
  static padSequence(
    parts: readonly Tensor[],
    batchFirst = false,
    paddingValue = 0,
  ): Tensor {
    if (parts.length === 0) throw new Error("pad_sequence 에 줄 것이 없다");
    const longest = Math.max(...parts.map((p) => p.shape[0] ?? 0));
    const padded = parts.map((p) =>
      p.pad(0, 0, longest - (p.shape[0] ?? 0), paddingValue));
    const stacked = Tensor.stack(padded, 0); // (개수, 길이, …)
    return batchFirst ? stacked : stacked.swapaxes(0, 1);
  }

  /** 번호를 자리 표시로. `n` 칸 중 자기 자리만 1 이다. */
  oneHot(classes: number): Tensor {
    const flat = this.reshape([this.size, 1]);
    const ids = Tensor.arange(classes).reshape([1, classes]);
    return flat.binary("eq", ids, "int64")
      .reshape([...this.shape, classes]);
  }

  /**
   * 음의 로그가능도. `this` 는 이미 `log_softmax` 를 지난 것이어야 한다.
   *
   * **정답 자리를 뽑는 곳에서 그래프가 끊기기 쉽다.** 값만 떼어 돌려주면 뽑은 자리로
   * 기울기가 안 가고 분류 손실이 통째로 미분 불가가 된다.
   */
  nllLoss(target: Tensor, reduction: Reduction = "mean"): Tensor {
    // **접기 전에 표본별 값을 만든다.** 뽑자마자 평균을 내면 `reduction: "none"`
    // 을 만들 자리가 없어진다 — 스칼라에서 표본별 값을 되살릴 수는 없다.
    const each = this.gather(1, target.reshape([target.size, 1]))
      .reshape([target.size]).neg();
    return each.reduceAs(reduction);
  }

  /** 로짓에서 바로. `log_softmax` 와 `nll_loss` 를 붙인 것이다. */
  crossEntropy(target: Tensor, reduction: Reduction = "mean"): Tensor {
    return this.logSoftmax(-1).nllLoss(target, reduction);
  }

  // ── 결과 크기가 값에 달린 것들 ────────────────────────────────────────
  //
  // 이 무리는 **몇 개가 나올지를 값을 봐야 안다.** GPU 는 버퍼 크기를 미리 정해야
  // 하므로 값을 한 번 읽어 와야 하고, 그래서 전부 비동기다. 자매도 같은 이유로
  // 여기서 CPU 를 왕복한다.

  /** 참인 자리만 골라 1차원으로. **기울기가 고른 자리로 흐른다.** */
  async maskedSelect(mask: Tensor): Promise<Tensor> {
    const m = await mask.toArray();
    const picks: number[] = [];
    for (const [i, v] of m.entries()) if (v !== 0) picks.push(i);
    return this.flat().indexSelect(0, Tensor.from(picks, [picks.length]));
  }

  /** 0 이 아닌 자리의 좌표. 자리는 값이 아니라 기울기가 없다. */
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

  /** `nonzero` 와 같다 — torch 가 이름을 둘 다 갖는다. */
  async argwhere(): Promise<Tensor> {
    return this.nonzero();
  }

  /**
   * 0 이 아닌 자리를 **정해진 개수만큼.** 모자라면 채우고 넘치면 자른다.
   *
   * `nonzero` 는 결과 크기가 값에 달려 GPU 를 한 번 읽어야 하는데, 이쪽은 크기를 미리
   * 주므로 그 왕복이 **원리상** 필요 없다 — 그 자리를 위해 있는 이름이다. 여기서는
   * 아직 읽는다(어느 자리가 0 이 아닌지는 값이라서). 커널로 옮길 자리다.
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
   * 칸마다 몇 개인가. **`min === max` 면 자료의 범위를 쓴다**(실측).
   *
   * 범위를 주면 **밖은 버린다** — 양끝 칸으로 몰아넣지 않는다. 전부 범위 안인 자료로
   * 재면 그 규칙이 안 드러난다.
   */
  async histc(bins = 100, min = 0, max = 0): Promise<Tensor> {
    const values = Array.from(await this.toArray());
    const edges = histEdges(values, bins, min, max);
    return Tensor.from(countInto(values, edges, null), [bins]);
  }

  /**
   * `histc` 와 같은 셈에 **경계까지.**
   *
   * `bins` 에 텐서를 주면 그것이 곧 경계다 — 칸 너비가 다를 수 있고, 그러면
   * `density` 가 칸마다 다른 값으로 나눈다.
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

  /** 축이 여럿인 히스토그램. `this` 는 `(표본 수, 차원)` 이다. */
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
   * 가장 자주 나온 값. **같은 횟수면 작은 값이 이기고, 자리는 그 값의 마지막이다**
   * (실측: `[4,4,5,5]` 가 값 4 · 자리 1 을 준다).
   *
   * 비긴 자리가 없는 자료로 재면 그 규칙이 안 드러난다.
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
   * NaN 을 **빼고** 센 중앙값. `median` 은 NaN 이 하나만 있어도 NaN 을 낸다(실측).
   *
   * 짝수 개면 **아래를 고른다** — 평균을 내지 않는다.
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
   * 중심 차분. **축마다 하나씩, 묶음으로 낸다** — 축을 안 주면 전부다.
   *
   * `edgeOrder` 가 1 이면 양끝을 한쪽 차분으로, 2 면 이차식으로 맞춘다(실측: `x²`
   * 에서 2 면 정확한 도함수가 나오고 1 이면 양끝이 어긋난다).
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

  // `trapz` 는 여기 없다. **`trapezoid` 자체가 borch.ts 의 메서드가 아니라**
  // 결속이 파이썬에서 조립하는 것이라(사다리꼴 조각을 더하는 몇 줄), 옛 이름도
  // 같은 자리에서 별칭으로 둔다 — 여기 하나 더 만들면 조립이 두 벌이 된다.

  /** 서로 다른 값을 **오름차순으로**. torch 의 기본이 정렬해서 주는 것이다. */
  async unique(): Promise<Tensor> {
    const values = Array.from(await this.toArray());
    const seen = [...new Set(values)].sort((a, b) => a - b);
    return Tensor.from(seen, [seen.length], { dtype: this.dtype });
  }

  /**
   * **이어진** 중복만 줄인다. `unique` 와 달리 정렬하지 않는다 — `[1,1,2,2,1]` 이
   * `[1,2,1]` 이 된다(실측). 정렬된 입력으로만 재면 둘이 같아 보인다.
   *
   * 결과의 **길이가 값에 달렸으므로** 여기서 한 번 읽는다. `unique`·`bincount` 와
   * 같은 자리이고, 그래서 이 셋만 비동기다.
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
   * 칸마다 몇 번 나왔는가.
   *
   * @param weights 주면 개수 대신 **무게를 더한다.**
   * @param minlength 결과의 최소 길이. 안 나온 칸까지 자리를 잡아 둔다.
   *
   * **형이 갈린다**(실측): 무게 없이는 `int64`, 무게가 있으면 그 무게의 형이다 —
   * 개수를 세는 것과 값을 더하는 것이 다른 일이기 때문이다.
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
   * 분위수. **선형 보간이다** — torch 의 기본이 그렇고, 가장 가까운 값을 고르는 것과
   * 다르다.
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
   * NaN 을 빼고 센 분위수.
   *
   * **NaN 을 빼면 자리가 밀린다.** 그래서 성한 칸만 모은 작은 텐서를 만들어 그 위에서
   * 되짚는다 — 원래 자리로 돌아가는 길은 `indexSelect` 가 이미 미분되므로 이어진다.
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
        `linalg: 2차원 이상이어야 한다 — 지금은 [${this.shape}] 다`,
      );
    }
    const rows = this.shape[rank - 2] ?? 0;
    const cols = this.shape[rank - 1] ?? 0;
    if (square && rows !== cols) {
      throw new RuntimeError(
        `linalg: 정사각 행렬이어야 한다 — 지금은 [${this.shape}] 다`,
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
          `linalg.${what}: 특이행렬이다 — 역행렬이 없다`,
        );
      }
      return LA.inverse(a, n);
    });
  }

  /** 행렬식. 역방향은 `det·A⁻ᵀ` 다. */
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

  /** `log|det|`. 역방향은 `A⁻ᵀ` — 행렬식이 곱해지지 않아 더 안정적이다. */
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

  /** 부호와 로그 절댓값을 따로. 행렬식이 아주 작아도 자릿수가 안 날아간다. */
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

  /** 역행렬. 역방향은 `-A⁻ᵀ·Ḡ·A⁻ᵀ` 다. */
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

  /** `inv` 와 같은데 **안 던진다** — 대신 `info` 에 0 이 아닌 수를 담는다. */
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
   * 유사역행렬. **기울기가 있다** — 항이 셋이다.
   *
   *     Ā = −Pᵀ·Ḡ·Pᵀ + (I − A·P)·Ḡᵀ·P·Pᵀ + Pᵀ·P·Ḡᵀ·(I − P·A)
   *
   * 뒤의 두 항은 **정사각 정칙에서 0 이 된다.** 그때는 `I − AP` 와 `I − PA` 가 둘 다
   * 0 이라 첫 항만 남고, 그 첫 항은 역행렬의 기울기와 같은 식이다. 그래서 둘을
   * 빠뜨려도 정사각에서는 맞고 직사각에서만 틀린다 — 골든이 직사각으로도 묻는다.
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
   * @param upper 참이면 위 삼각을 준다 — `L` 대신 `Lᵀ` 다.
   *
   * **뒤집기로 짠다.** `Lᵀ` 는 `L` 의 전치이고 전치는 이미 미분되므로, 분해를 두 벌
   * 쓰지 않는다 — 같은 식을 두 벌로 두면 언젠가 한쪽만 고쳐진다.
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
          "linalg.cholesky: 대칭 양정부호가 아니다 (주소행렬식이 0 이하다)",
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

  /** `cholesky` 의 안 던지는 쪽. */
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
   * `A x = b`. 역방향은 `b` 로는 `A⁻ᵀ·Ḡ`, `A` 로는 `-A⁻ᵀ·Ḡ·xᵀ` 다.
   *
   * `b` 가 `A` 보다 축이 하나 적으면 **벡터 묶음**으로 본다 — torch 의 규칙이다.
   * 역방향의 바깥곱이 그 구분에 걸린다.
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

  /** `solve` 의 안 던지는 쪽. */
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
   * QR 분해. **`reduced` 에는 기울기가 있다.**
   *
   *     N = Qᵀ·Q̄ − R̄·Rᵀ
   *     Ā = [Q̄ + Q·(tril(N − Nᵀ, −1) − N)]·R⁻ᵀ
   *
   * 아래 삼각만 남기는 자리가 이 유도의 전부다 — `QᵀQ = I` 를 미분하면 `Qᵀ·dQ` 가
   * 반대칭이 되고, 위쪽은 아래쪽의 거울이라 따로 셀 것이 없다.
   *
   * `complete` 는 `Q` 에 남는 열이 있어 유도가 다르다. 그쪽은 값만 낸다.
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
   * 특이값 분해.
   *
   * **특잇값에는 기울기가 있고 `U`·`Vh` 에는 없다.** `dS = diag(Uᵀ·dA·V)` 라 특잇값
   * 쪽은 `Ā = U·diag(Ḡ)·Vᵀ` 한 줄이고 겹침 문제도 없다. 벡터 쪽은 `1/(sᵢ²−sⱼ²)` 가
   * 들어가 특잇값이 겹치면 터지는데, 그 자리는 안 넣었다.
   *
   * `fullMatrices`(기본 참)는 `U` 를 `rows×rows` 로 채운다 — torch 의 기본값이다.
   * 채우는 방향은 남는 차원이 둘 이상이면 유일하지 않다(`completeBasis` 참고).
   * **역방향에 쓰는 것은 채우기 전의 축소본이다.**
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
   * 대칭 행렬의 고윳값·고유벡터. 고윳값은 오름차순이다.
   *
   * **한쪽 삼각만 읽는다** — 기본은 아래쪽이다. 야코비는 행렬 전체를 보므로 먼저
   * 거울을 만들어 넘긴다. 자세한 것은 `linalg.ts` 의 `mirror` 에 적었다.
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

  /** 특잇값만. `svd` 의 가운데다. */
  async svdvals(): Promise<Tensor> {
    return (await this.svd(false)).s;
  }

  /** 대칭 행렬의 고윳값만. */
  async eigvalsh(uplo: "L" | "U" = "L"): Promise<Tensor> {
    return (await this.eigh(uplo)).values;
  }

  /**
   * 행렬로 보고 재는 노름. **갈래마다 다른 수다.**
   *
   * 기본은 프로베니우스, `2` 는 최대 특잇값, `nuc` 는 특잇값의 합, `1` 은 열 절댓값
   * 합의 최대, `inf` 는 행 쪽이다. 특잇값이 필요한 셋만 CPU 를 왕복한다.
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

  /** 조건수. 기본은 특잇값의 비다. */
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

  /** 삼각행렬이라는 것을 알고 푼다. */
  async solveTriangular(
    b: Tensor, upper: boolean, left = true, unitriangular = false,
  ): Promise<Tensor> {
    const v = await this.asBatch();
    if (v.batch !== 1) throw new RuntimeError("solve_triangular: 배치는 아직 없다");
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
   * 행렬 지수 `e^A`. **기울기가 있다.**
   *
   * 역방향은 `linalg.ts` 의 `matrixExpAdjointMap` 이 순방향에서 굳혀 둔 표를 쓴다 —
   * 왜 표인지는 거기 적었다(짧게: `Ḡ` 가 GPU 에 있고 `expm` 은 CPU 다).
   * **기울기가 필요할 때만 만든다.** 표 하나에 `expm` 이 `n²` 번이라 공짜가 아니다.
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

  /** 마지막 축을 벡터로 보고 내적. */
  vecdot(other: Tensor, dim = -1): Tensor {
    return this.mul(other).sumDim(dim, false);
  }

  /** 3 차원 벡터의 외적. 축을 셋으로 갈라 조합한다. */
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
   * 벡터로 보고 재는 노름. **행렬을 줘도 통째로 편다** — `matrixNorm` 과 갈리는 자리다.
   *
   * `ord=0` 은 0 이 아닌 것의 개수, `±Infinity` 는 절댓값의 최대·최소다. 거듭제곱
   * 식에 넣으면 안 되는 갈래라 따로 적는다.
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

  /** 텐서를 행렬로 접어 풀고 다시 편다. */
  async tensorSolve(b: Tensor): Promise<Tensor> {
    const n = b.size;
    const folded = this.reshape([n, this.size / n]);
    const x = await folded.solve(b.reshape([n]));
    return x.reshape(this.shape.slice(b.shape.length));
  }

  /** 텐서를 행렬로 접어 뒤집고 축 순서를 돌려준다. */
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

  /** 최소제곱해. 정사각이고 정칙이면 `solve` 와 같은 답이다. */
  async lstsq(b: Tensor): Promise<Tensor> {
    const v = await this.asBatch(false);
    if (v.batch !== 1) throw new RuntimeError("lstsq: 배치는 아직 없다");
    const { rows, cols } = v;
    const width = b.shape.length === 1 ? 1 : (b.shape[b.shape.length - 1] ?? 1);
    const rhs = LA.fromF32(await b.toArray());
    const sol = LA.matmul(
      LA.pinverse(v.mats[0]!, rows, cols), rhs, cols, rows, width);
    return Tensor.fromMat(sol, b.shape.length === 1 ? [cols] : [cols, width]);
  }

  /** 한 장에 겹쳐 담은 `L`·`U` 와 **1 부터 세는** 교환표. */
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
   * `luFactor` 에 **`info` 를 하나 더.** 던지는 대신 번호로 알린다 — 0 이면 잘 됐고,
   * `k` 면 `k` 번째 피벗이 0 이라 특이행렬이다(1 부터 센다).
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
   * 대칭 행렬을 `L D Lᵀ` 로. **피벗을 안 한다.**
   *
   * torch 는 LAPACK 의 Bunch-Kaufman 을 쓰고 그것은 필요하면 자리를 바꾼다. 여기서는
   * 바꿀 일이 없는 자리(양의 정부호 같은 것)만 다루고, 대각이 0 에 가까우면 시끄럽게
   * 거절한다 — 조용히 이어 가면 바꾼 것과 안 바꾼 것이 다른 답을 내는데 둘 다
   * 그럴듯하다.
   *
   * 답은 torch 와 같은 모양으로 **한 장에 겹쳐 담는다** — 대각이 `D`, 그 아래가 `L`.
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
          throw new RuntimeError("ldl_factor — 피벗이 필요한 대칭 행렬 (부정부호)");
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

  /** `ldlFactor` 가 낸 분해로 푼다. `L y = b`, `D z = y`, `Lᵀ x = z` 세 번이다. */
  async ldlSolve(b: Tensor): Promise<Tensor> {
    const v = await this.asBatch();
    const n = v.rows;
    if (v.batch !== 1) throw new RuntimeError("ldl_solve: 배치는 아직 없다");
    const ld = v.mats[0];
    if (!ld) throw new RuntimeError("ldl_solve: 분해가 비었다");
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
   * QR 을 **반사자 꼴로** 낸다. `householderProduct` 가 그것으로 `Q` 를 세운다.
   *
   * **대각 아래가 전부 0 이면 반사를 안 한다** — LAPACK 의 `dlarfg` 가 그 자리에서
   * `tau = 0` 을 놓고 값을 그대로 둔다. 정사각의 마지막 열이 늘 그 자리다.
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
   * 반사자들을 곱해 `Q` 를 세운다. `geqrf` 의 짝이다.
   *
   * `v_i` 는 대각이 1 이고 그 아래가 `A[i+1:, i]` 다 — 대각의 1 은 **저장 안 하는
   * 약속**이라, 그 자리를 읽어 쓰면 분해가 담아 둔 `R` 을 반사자로 착각한다.
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

  /** `P`·`L`·`U` 셋으로 펴서. 겹쳐 담은 것보다 읽기 쉽다. */
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
   * 인수분해한 것으로 `A x = b` 를 푼다. **`this` 가 LU 다** — `linalg.lu_solve(LU,
   * pivots, B)` 의 자리 배치이고, torch 의 `Tensor.lu_solve` 와는 수신자가 다르다.
   *
   * 그래서 **이름에서 torch 주장을 뺐다.** 전에는 이것이 `luSolve` 였는데, torch 를
   * 옮겨 적는 사람이 `b.lu_solve(LU, piv)` 를 쓰면 `LU` 가 `pivots` 자리에 들어간다 —
   * 이름도 인자 개수도 맞아서 **그 자리에서는 안 걸리고** 값만 틀린다.
   */
  async luSolveFactored(pivots: Tensor, b: Tensor): Promise<Tensor> {
    const v = await this.asBatch();
    if (v.batch !== 1) throw new RuntimeError("lu_solve: 배치는 아직 없다");
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
   * `A x = b` 를 **촐레스키 인수로** 푼다. `A = L Lᵀ` (또는 `Uᵀ U`).
   *
   * `A` 를 다시 세워 `solve` 로 보낸다. 삼각 대입 두 번이 더 싸지만, 그 길로 적으면
   * 역방향을 손으로 써야 하고 **인수 쪽 기울기가 조용히 빠진다.**
   */
  async choleskySolve(factor: Tensor, upper = false): Promise<Tensor> {
    const low = Tensor.asLower(factor, upper);
    return low.mm(low.transpose()).solve(this);
  }

  /** 촐레스키 인수에서 **원래 행렬의 역행렬**을 낸다. 인수의 역이 아니다. */
  async choleskyInverse(upper = false): Promise<Tensor> {
    const low = Tensor.asLower(this, upper);
    return low.mm(low.transpose()).inverse();
  }

  /**
   * **둘을 준다** — 해와, 넘긴 계수 행렬의 **사본**(실측).
   *
   * `solveTriangular` 와 같은 계산인데 인자 순서가 뒤집혀 있고 **기본 `upper` 가
   * 참이다.** 그 둘을 놓치면 다른 삼각을 풀고도 값이 그럴듯하게 나온다.
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
   * `(LU, pivots)`. **`lu()` 와 다른 것을 낸다** — 그쪽은 `P·L·U` 셋으로 펴 주고
   * 이쪽은 **겹쳐 담은 한 판과 교환 목록**이다(실측).
   *
   * `getInfos` 면 셋째로 정보 코드가 붙는다. 우리는 늘 0 이다 — 특이 행렬을 만나면
   * 그 자리에서 던지지 조용히 코드로 알리지 않는다.
   */
  async luTop(pivot = true, getInfos = false): Promise<{
    LU: Tensor; pivots: Tensor; info?: Tensor;
  }> {
    if (!pivot) throw new RuntimeError("lu(pivot=false) 는 없다");
    const got = await this.luFactor();
    return getInfos ? { ...got, info: Tensor.zeros([]) } : got;
  }

  /**
   * `torch.Tensor.lu_solve` — **수신자가 오른쪽 변 `b` 다.**
   *
   * torch 가 `b.lu_solve(LU, piv)` 이므로 여기도 그렇다. 인수를 수신자로 받는 쪽은
   * `luSolveFactored` 이고, 그것이 `linalg.lu_solve` 의 자리 배치다.
   */
  async luSolve(luData: Tensor, pivots: Tensor): Promise<Tensor> {
    return luData.luSolveFactored(pivots, this);
  }

  /**
   * 옛 이름. 결속이 이것을 부르고 있어서 남긴다 — 위임 한 줄이다.
   *
   * `luSolve` 가 인수를 수신자로 받던 시절에 그 반대편을 가리키려고 만든 이름인데,
   * 이제 `luSolve` 자체가 torch 와 같은 자리이므로 **`Top` 이 가리킬 반대편이 없다.**
   * 결속이 `luSolve` 로 옮겨 가면 지워도 된다.
   */
  async luSolveTop(luData: Tensor, pivots: Tensor): Promise<Tensor> {
    return this.luSolve(luData, pivots);
  }

  /**
   * 겹쳐 담은 한 판을 `P·L·U` 로 편다.
   *
   * **끄면 `null` 이 아니라 빈 텐서가 온다**(실측: 모양이 `[0]` 이다). `null` 로 두면
   * 받는 쪽이 그것으로 갈라 쓰게 되고, 그것은 torch 코드가 아니다.
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

  /** `householderProduct` 의 다른 이름. torch 가 둘을 준다. */
  async orgqr(tau: Tensor): Promise<Tensor> {
    return this.householderProduct(tau);
  }

  /**
   * **Q 를 안 세우고** `C` 에 곱한다 — 여기서는 세워서 곱한다. 값이 같고 이 크기에서
   * 아끼는 것이 없다.
   *
   * **`orgqr` 과 다른 Q 다.** 그쪽은 `m×k` 로 **자른** Q 를 주는데, 이쪽은 자르지
   * 않은 `m×m` 을 쓴다 — 반사자들은 `Rᵐ` 위의 사상이고, 자르면 그 사상의 일부만
   * 곱하게 된다. 세로로 긴 행렬에서 답이 통째로 갈린다(실측). 정사각으로만 재면
   * 둘이 같아서 안 보인다.
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
   * 대칭 행렬의 **끝쪽 고유쌍 `k` 개.**
   *
   * **torch 는 반복법이고 우리는 정확해다.** 그쪽은 큰 희소 행렬에서 몇 개만 싸게
   * 얻으려고 반복하는데, 우리에게는 희소가 없고 크기도 작다. 재보니 torch 의 답이
   * 정확해로 7e-6 안까지 수렴하고 씨앗에도 그만큼만 흔들린다(실측) — 허용 오차
   * 한참 아래다. 값은 같고 비용만 다르다.
   *
   * **`largest` 가 순서까지 정한다** — 참이면 큰 것부터, 거짓이면 작은 것부터다(실측).
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
   * 무작위 사영으로 얻는 **저계수 SVD.** `(U, S, V)` 이고 **V 는 전치가 아니다.**
   *
   * **정확히 저계수인 입력에서만 답이 안 흔들린다.** torch 는 무작위 행렬로 사영하는데,
   * 계수가 `q` 를 넘으면 씨앗에 따라 특이값이 0.5 씩 움직인다(실측). 계수가 `q` 이하면
   * 씨앗을 바꿔도 7e-7 안이다 — 물을 수 있는 자리는 그쪽뿐이다.
   *
   * 우리는 사영을 안 한다. 전체 SVD 를 구해 앞의 `q` 개를 자른다 — 정확히 저계수인
   * 자리에서는 같은 답이고, 넘치는 자리에서는 **torch 보다 정확한** 답이다.
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
   * 저계수 PCA. **`center=false` 면 `svdLowrank` 와 같은 것이다**(실측).
   *
   * 가운데 맞추기가 이 함수와 저쪽의 차이 전부다. 참으로만 재면 그 갈래가 안 보인다.
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
        "제자리 연산이 칸 수를 늘릴 수는 없다 — 버퍼가 그만큼이 아니다.",
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

  fill_(value: number): Tensor {
    return this.mutate(() => Tensor.full(this.shape, value));
  }

  clamp_(low: number, high: number): Tensor {
    return this.mutate(() => this.clamp(low, high));
  }

  /** `clamp_` 와 같다 — torch 가 이름을 둘 다 갖는다. */
  clip_(low: number, high: number): Tensor {
    return this.clamp_(low, high);
  }

  /**
   * 다른 텐서의 값을 이 버퍼로 옮긴다. **텐서를 바꿔치지 않는다.**
   *
   * 옵티마이저가 파라미터의 손잡이를 들고 있으므로, 가중치를 넣을 때 새 텐서를
   * 만들어 갈아끼우면 옵티마이저가 다른 것을 보게 된다 — 학습이 도는데 파라미터가
   * 안 움직이는 상태다. 그래서 자리는 그대로 두고 값만 옮긴다.
   */
  copyFrom(src: Tensor): Tensor {
    if (src.size !== this.size) {
      throw new Error(`크기가 다르다: [${this.shape}] ← [${src.shape}]`);
    }
    return this.mutate(() => src);
  }

  /** `this ← (1-t)·this + t·other`. 이동 통계가 쓴다. */
  lerpFrom(other: Tensor, t: number): Tensor {
    return this.mutate(() =>
      this.binary("mul", Tensor.full([], 1 - t))
        .add(other.binary("mul", Tensor.full([], t))));
  }

  /** 표의 단항을 제자리로. `abs_` 같은 이름들이 이리로 온다. */
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

  /** 같은 버퍼를 다른 모양으로 본다. `reshape` 와 같고, 제자리 연산이 번진다. */
  view(...shape: number[]): Tensor {
    return this.reshape(shape);
  }

  // ── 정렬 계열 ─────────────────────────────────────────────────────────

  /**
   * 축 하나를 정렬해 값과 자리를 낸다.
   *
   * **기울기가 값을 따라간다.** 뽑아 온 자리로만 흘리고 나머지는 0 인데, 값만 떼어
   * 돌려주면 그 자리로 기울기가 안 가고 분류 손실이 통째로 미분 불가가 된다.
   * 코어가 `topk`·`sort` 에서 겪었고, 자매도 리뷰 전까지 같은 상태였다.
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

  /** 정렬한 자리만. 값이 필요 없을 때 쓴다. */
  argsort(dim = 0, descending = false): Tensor {
    return this.sort(dim, descending).indices;
  }

  /** 가장 큰 `k` 개. `sort` 의 앞부분이다 — torch 도 내림차순으로 준다. */
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
   * `k` 번째로 작은 값. **1 부터 센다** — torch 가 그렇다.
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
   * 중앙값. **짝수 개일 때 아래쪽을 준다** — torch 가 두 값을 평균내지 않는다.
   *
   * **NaN 이 하나라도 있으면 NaN 이다**(실측). 정렬은 NaN 을 한쪽 끝으로 밀어내므로
   * 그냥 골라 오면 **NaN 을 건너뛰고** 멀쩡한 값이 나온다 — 그것이 `nanmedian` 이고
   * 이쪽은 아니다. 코어에도 같은 결함이 있었고, 둘을 나란히 묻는 케이스를 넣으면서
   * 양쪽이 같이 걸렸다.
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

  /** 정렬만 하고 자리는 안 준다. */
  msort(): Tensor {
    return this.sort(0, false).values;
  }

  /** 누적 최대·최소. **동점이면 나중 자리**를 준다 — torch 가 그렇다. */
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
   * 번호가 가리키는 자리에 **더한다.** 겹치면 쌓인다.
   *
   * `gather` 의 반대다. 한쪽만 있으면 "꺼낼 수는 있는데 되돌려 넣을 수가 없는"
   * 상태이고, 임베딩이나 원-핫을 손으로 만드는 코드가 그 자리를 바로 만난다.
   */
  scatterAdd(dim: number, index: Tensor, src: Tensor): Tensor {
    return this.scatterWith(dim, index, src, "add");
  }

  /** 번호가 가리키는 자리에 **덮어쓴다.** 겹치면 마지막에 쓴 것이 남는다. */
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
   * 번호표가 가리키는 칸을 읽어 `shape` 로 낸다.
   *
   * **겹치는 번호로는 기울기가 쌓인다** — 한 칸을 두 번 읽었으면 두 번 온다. 안
   * 겹치는 걸음으로만 재면 그 쌓임이 안 보인다.
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

  /** 번호표 자리에 `src` 를 써 넣은 **사본.** */
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

  /** 평평한 저장소를 **다른 걸음으로** 읽는다. torch 는 뷰지만 여기서는 사본이다. */
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

  /** `select` 가 꺼내던 한 장을 **갈아끼운 사본.** */
  selectScatter(src: Tensor, dim: number, index: number): Tensor {
    const rank = this.shape.length;
    const axis = dim < 0 ? dim + rank : dim;
    const at = index < 0 ? index + (this.shape[axis] ?? 0) : index;
    const { spots } = selectSpots(this.shape, axis, at);
    return this.scatterSpots(Tensor.spotsTensor(spots), src, false,
      "SelectScatterBackward0");
  }

  /** `x[..., start:end:step]` 자리를 **갈아끼운 사본.** */
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

  /** 대각선 자리를 **갈아끼운 사본.** */
  diagonalScatter(src: Tensor, offset = 0, dim1 = 0, dim2 = 1): Tensor {
    const rank = this.shape.length;
    const d1 = dim1 < 0 ? dim1 + rank : dim1;
    const d2 = dim2 < 0 ? dim2 + rank : dim2;
    const { spots } = diagonalSpots(this.shape, offset, d1, d2);
    return this.scatterSpots(Tensor.spotsTensor(spots), src, false,
      "DiagonalScatterBackward0");
  }

  /** 마지막 축을 **대각선으로 펴서** 축을 하나 늘린다. `diagonal` 의 반대다. */
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
   * **평평하게 펴서** 번호대로 넣는다 — 축이라는 개념이 없다. `take` 의 반대다.
   *
   * 번호가 값으로 오므로 CPU 가 만드는 번호표와 같은 문을 그대로 쓴다.
   */
  put(index: Tensor, source: Tensor, accumulate = false): Tensor {
    return this.scatterSpots(index.reshape([index.size]),
      source.reshape([source.size]), accumulate, "PutBackward0")
      .reshape(this.shape);
  }

  /**
   * 축마다 번호 텐서를 하나씩 받아 그 자리에 넣는다.
   *
   * 축별 번호를 **평평한 번호 하나로 접는다** — 그러면 `put` 과 같은 문이 된다.
   * 접는 셈이 텐서 산술이라 커널이 안 늘어난다.
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

  /** `scatter` 와 같은 자리지만 **덮어쓰는 대신 합친다.** */
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

  /** 번호가 가리키는 **줄**을 합친다. `scatterReduce` 와 달리 번호가 줄 단위다. */
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
   * 가면이 참인 자리에 `source` 를 **평평한 차례대로** 채운다.
   *
   * 어느 값이 어디로 가는지가 가면의 **값**에 달렸는데도 결과 모양은 정해져 있다 —
   * 그래서 값을 읽어 오지 않고 자리마다 "앞에 참이 몇이었나" 를 세는 커널로 푼다.
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

  /**
   * `dim` 을 따라 잘라 본 **각 조각의 노름을 `maxnorm` 아래로** 끌어내린다.
   *
   * 조립으로 둔다 — **배율 안에 `x` 가 들어 있어서** 역방향이 `g·s` 가 아니다.
   * 손으로 적으면 순방향만 맞고 기울기가 조용히 틀린다(코어에서 그렇게 한 번 틀렸다).
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
   * 평평한 번호를 축별 번호로 푼다. **축마다 텐서 하나씩, 묶음으로 낸다**(실측).
   *
   * 셈이 나눗셈과 나머지뿐이라 값을 안 읽는다.
   */
  unravelIndex(shape: readonly number[]): Tensor[] {
    const st = rowStrides(shape);
    return shape.map((size, d) => this
      .div(Tensor.full([], st[d] ?? 1)).unary("floor")
      .remainder(size));
  }

  /** 여러 행렬을 잇달아 곱한다. `multiDot` 이 같은 것을 목록으로 받는다. */
  static chainMatmul(...matrices: readonly Tensor[]): Tensor {
    let out = matrices[0]!;
    for (let i = 1; i < matrices.length; i++) out = out.mm(matrices[i]!);
    return out;
  }

  /** 바깥곱의 옛 이름. `outer` 와 같은 것이다. */
  ger(other: Tensor): Tensor {
    return this.outer(other);
  }

  /**
   * 행렬 × 벡터. `mm` 이 하는 일이지만 torch 는 이름을 따로 준다.
   *
   * 벡터를 열 하나로 세웠다가 그 축을 도로 지운다 — 결과가 1차원이어야 한다.
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
   * 2차원 합성곱. `this` 는 `(N, C, H, W)`, 커널은 `(O, C, KH, KW)` 다 — **NCHW** 다.
   *
   * 자매는 NHWC 를 들고 다녔는데 그것은 TF.js 의 conv 가 그 배치에서만 빨라서였다.
   * 여기서는 커널을 우리가 쓰므로 torch 와 같은 배치를 그대로 쓴다.
   */
  conv2d(weight: Tensor, bias: Tensor | null = null, stride = 1, padding = 0): Tensor {
    if (this.shape.length !== 4 || weight.shape.length !== 4) {
      throw new Error(`conv2d 는 4차원끼리다: [${this.shape}] × [${weight.shape}]`);
    }
    return this.convND(weight, bias, stride, padding);
  }

  /** 1차원 합성곱. `(N, C, L)`. */
  conv1d(weight: Tensor, bias: Tensor | null = null, stride = 1, padding = 0): Tensor {
    if (this.shape.length !== 3 || weight.shape.length !== 3) {
      throw new Error(`conv1d 는 3차원끼리다: [${this.shape}] × [${weight.shape}]`);
    }
    return this.convND(weight, bias, stride, padding);
  }

  /** 3차원 합성곱. `(N, C, D, H, W)`. */
  conv3d(weight: Tensor, bias: Tensor | null = null, stride = 1, padding = 0): Tensor {
    if (this.shape.length !== 5 || weight.shape.length !== 5) {
      throw new Error(`conv3d 는 5차원끼리다: [${this.shape}] × [${weight.shape}]`);
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
   * 표본 하나의 `-log P(표적 | 소리)`.
   *
   * @param lp `(T, C)` 로그 확률.
   * @param labels 표적 글자들.
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
   * 창 자리를 표본이 흔드는 최대 풀링.
   *
   * **표본은 평면마다 다르다** — torch 의 `_random_samples` 가 `(N, C, 축)` 이라
   * 창이 평면마다 갈린다. 그래서 평면 수만큼 돈다. 비싼 대신 torch 와 같다.
   *
   * **축 순서가 차원마다 다르다.** ATen 의 2차원판은 표본을 (너비, 높이) 로 읽고
   * 3차원판은 (깊이, 높이, 너비) 로 읽는다 — 두 함수가 서로 어긋나 있고, 여기서
   * 흉내내는 것은 그 어긋남이다.
   *
   * @param samples 평면마다의 표본. `samples[plane][i]` 이고 `i` 는 위 순서다.
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
   * 자리표가 가리키는 칸에 값을 놓고 나머지는 0 으로 둔다.
   *
   * 풀링이 버린 자투리는 되살릴 수 없어서 출력 크기를 밖에서 정한다. 기본은
   * `(n-1)·stride - 2·padding + kernel` 이고, `outSize` 로 직접 줄 수도 있다.
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
   * 차원 수에 상관없는 합성곱. `conv1d`·`conv2d`·`conv3d` 가 전부 이리로 온다.
   *
   * 공간 축이 하나면 1차원, 둘이면 2차원이다 — 차원마다 함수를 따로 두면 세 벌이
   * 되고 그중 하나만 고치는 날이 온다. 자매가 실제로 그 상태였다.
   */
  convND(
    weight: Tensor,
    bias: Tensor | null = null,
    stride: number | readonly number[] = 1,
    padding: number | readonly number[] = 0,
  ): Tensor {
    const spatial = this.shape.length - 2;
    if (spatial < 1 || weight.shape.length !== this.shape.length) {
      throw new Error(`conv: 모양이 안 맞는다: [${this.shape}] × [${weight.shape}]`);
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
   * 차원 수에 상관없는 전치 합성곱.
   *
   * **새 커널이 없다.** 전치 합성곱의 순방향은 보통 합성곱이 입력 쪽으로 흘리는
   * 것과 같은 계산이라 `convNDGradInputTiled` 를 그대로 쓴다. 역방향도 마찬가지로
   * 뒤집힌다 — 입력 쪽 기울기가 보통 합성곱의 순방향이고, 가중치 쪽은 같은 커널에
   * 두 인자를 바꿔 넣은 것이다. 같은 계산을 두 벌 두면 한쪽만 고쳐지는 날이 온다.
   *
   * **가중치 축이 `convND` 와 뒤집혀 있다** — `(입력, 출력, …)` 다. 정사각 커널이면
   * 뒤집어 놓아도 모양이 맞아서 값으로만 갈린다.
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
        `convTranspose: 모양이 안 맞는다: [${this.shape}] × [${weight.shape}]`);
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

  /** 차원 수에 상관없는 풀링. */
  poolND(kind: "max" | "avg", kernel: number, stride?: number): Tensor {
    const spatial = this.shape.length - 2;
    if (spatial < 1) throw new Error(`풀링: 모양이 안 맞는다: [${this.shape}]`);
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

  /** 최근접 이웃으로 확대. `Upsample`·`interpolate` 가 이것이다. */
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
   * 출력 크기를 정해 놓고 접는다. 창 크기가 입력에서 나온다.
   *
   * **나눠떨어지지 않아도 된다.** 예전에는 거절했는데, torch 는 창을 자리마다 다르게
   * 잡는다 — 거절은 흉내의 한 방식이 아니라 다른 규칙이었다. 시작은 내림, 끝은
   * 올림이고, 8 을 3 으로 줄이면 창이 3·3·2 다.
   *
   * 축을 하나씩 접는다. 창이 직사각형이라 축별로 나눠 해도 같은 값이다 — 평균은 각
   * 줄의 길이가 같아서 평균의 평균이 전체 평균이고, 최댓값은 원래 그렇다.
   *
   * 전용 커널을 안 쓴다. 창이 자리마다 다르면 셰이더에 상수로 구울 것이 없고,
   * 적응형은 대개 마지막 한 번이라 스텝마다 도는 자리가 아니다.
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

  /** 예전 이름. `adaptivePool("avg", …)` 와 같다 — 부르던 자리가 있어 남긴다. */
  adaptiveAvgPool(outSize: number): Tensor {
    return this.adaptivePool("avg", outSize);
  }

  /**
   * `p` 승의 합을 `p` 제곱근한 것.
   *
   * **torch 의 조립을 그대로 따른다** — 평균 풀링을 쓰고 창 크기를 곱해 합으로
   * 되돌린 뒤 제곱근을 취한다. 부호와 `relu` 가 끼는 것도 그쪽 구현 그대로다.
   */
  lpPool(normType: number, kernel: number, stride?: number): Tensor {
    const spatial = this.shape.length - 2;
    const count = kernel ** spatial;
    const powered = this.powScalar(normType);
    const out = powered.poolND("avg", kernel, stride ?? kernel);
    const signed = out.unary("sign").mul(out.abs().unary("relu"));
    return signed.mul(Tensor.full([], count)).powScalar(1 / normType);
  }

  /** 겹치지 않는 창의 최대값. `this` 는 `(N, C, H, W)`. */
  maxPool2d(kernel = 2, stride?: number): Tensor {
    return this.pool2d("max", kernel, stride ?? kernel);
  }

  avgPool2d(kernel = 2, stride?: number): Tensor {
    return this.pool2d("avg", kernel, stride ?? kernel);
  }

  private pool2d(kind: "max" | "avg", kernel: number, stride: number): Tensor {
    if (this.shape.length !== 4) {
      throw new Error(`풀링은 4차원이다: [${this.shape}]`);
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
   * 융합 배치 정규화. 통계·정규화·크기·치우침을 커널 셋으로 끝낸다.
   *
   * 조립판은 층 하나에 dispatch 가 스무 개 넘게 들었고 ResNet 한 스텝의 1,636 개
   * 중 태반이 거기서 나왔다(실측). 여기서는 순방향 둘, 역방향 둘이다.
   *
   * @returns 정규화 결과와, 이동 통계를 갱신할 평균·분산.
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

  /** 이 텐서가 복소수인가. `torch.is_complex` 자리다. */
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
   * 실수부와 허수부를 엮는다. `torch.complex` 자리다.
   *
   * 역방향은 받은 복소 기울기를 **그대로 갈라 준다** — 규약이 곧
   * `(∂L/∂re, ∂L/∂im)` 이라 꺼내는 것 말고 할 일이 없다. 이 자리가 규약의 정의다.
   */
  static complex(re: Tensor, im: Tensor): Tensor {
    if (re.shape.length !== im.shape.length
      || re.shape.some((d, i) => d !== im.shape[i])) {
      throw new RuntimeError(
        `complex 는 모양이 같아야 한다: [${re.shape}] vs [${im.shape}]`,
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

  /** 크기와 편각으로 만든다. `torch.polar` 자리다. */
  static polar(abs: Tensor, angle: Tensor): Tensor {
    if (abs.shape.length !== angle.shape.length
      || abs.shape.some((d, i) => d !== angle.shape[i])) {
      throw new RuntimeError(
        `polar 는 모양이 같아야 한다: [${abs.shape}] vs [${angle.shape}]`,
      );
    }
    const n = abs.size;
    const out = Tensor.cRun(`cpolar:${n}`, () => complexPolar(n),
      [abs.buffer, angle.buffer], 2 * n, n);
    return new Tensor(out, abs.shape, { dtype: "complex64" });
  }

  /**
   * 복소수를 **실수 짝으로 본다** — 모양 끝에 2 가 붙는다. 버퍼를 안 옮긴다.
   *
   * 인터리브 저장이 이 한 줄을 위해 있다고 해도 된다. torch 에서도 뷰다.
   */
  viewAsReal(): Tensor {
    if (!this.isComplex()) {
      throw new RuntimeError(
        "view_as_real 은 복소수 텐서에만 쓴다 — 지금 형은 " +
          `torch.${this.dtype} 다.`,
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

  /** 실수 짝을 복소수로 본다. `viewAsReal` 의 반대이고 역시 뷰다. */
  viewAsComplex(): Tensor {
    const last = this.shape[this.shape.length - 1];
    if (this.isComplex() || last !== 2) {
      throw new RuntimeError(
        "view_as_complex 는 마지막 축이 2 인 실수 텐서에만 쓴다: " +
          `[${this.shape}] (형 torch.${this.dtype})`,
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
        `${what} 은(는) 복소수 텐서에만 쓴다 — 지금 형은 torch.${this.dtype} 다.`,
      );
    }
  }

  /**
   * 실수부. **기울기는 `g + 0i` 로 돌아간다.**
   *
   * torch 에서 이것은 스트라이드 뷰인데 여기서는 **복사**다 — 우리 텐서는 모양만
   * 들고 스트라이드를 안 들어서 칸 걸러 보는 틀을 만들 수가 없다. 값은 같다.
   */
  real(): Tensor {
    this.needComplex("real");
    const n = this.size;
    const out = Tensor.cRun(`cpart:0:${n}`, () => complexPart(n, 0),
      [this.raw], n, n);
    return Tensor.make(
      out, this.shape, [this], (g) => [g.asComplexRe()],
      "RealBackward0", "float32",
    );
  }

  /**
   * 허수부. **기울기는 `0 + gi` 다** — `−gi` 로 적으면 부호만 뒤집힌 채 그럴듯하게
   * 돈다. torch 를 재보면 `z.imag` 의 기울기가 `0+1j` 다.
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
   * 켤레. **정칙이 아니라 역방향이 `conj(g)`** 다 — `conj(f')·g` 꼴이 아니다.
   *
   * **torch 와 갈리는 자리다.** torch 의 `conj` 는 게을러서 켤레 비트만 세우고
   * 값을 안 뒤집는다(그래서 `is_conj` 가 참이고 `view_as_real` 이 거절한다).
   * 우리 것은 즉시 뒤집으므로 그 상태가 아예 없다 — 값은 같다.
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
   * 복소수의 크기. **결과가 실수이고 기울기가 `z/|z|`** 다 — 켤레가 **안** 붙는다.
   *
   * 실수 `abs` 는 단항 표에 있고 이쪽으로 안 온다. `abs()` 가 형을 보고 갈라 준다 —
   * **공개인 이유가 그것이다.** 갈라 주는 자리가 클래스 밖(단항 표를 얹은 뒤)이라
   * 비공개면 못 부른다.
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

  /** 편각. 복소수는 `atan2(im, re)`, 실수는 음수에 π 다(형은 언제나 실수). */
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
        `복소수 ${name} 은 아직 모양이 같아야 한다: [${a.shape}] vs [${b.shape}] ` +
          "— 복소수끼리의 브로드캐스팅은 없다.",
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
   * @param retainGraph 참이면 그래프를 놓지 않는다. torch 와 같이 **기본은 놓는 것**이다 —
   *   중간 값들이 메모리를 붙들고 있어서, 안 놓으면 학습 루프에서 계속 쌓인다.
   */
  /**
   * 기울기를 흘린다.
   *
   * **인자 차례가 torch 와 같다** — `backward(gradient, retainGraph)`. 전에는 첫
   * 자리가 `retainGraph` 였고, 그래서 `y.backward(onesLike(y))` 를 옮겨 적을 수가
   * 없었다. 야코비안-벡터 곱을 부르는 흔한 줄인데다, **코어(numpy)는 처음부터 받고
   * 있었다** — 셋 중 이쪽만 못 받았다.
   *
   * 첫 자리를 바꿔도 깨지는 호출부가 없었다. 저장소 어디에서도 `backward` 에 인자를
   * 넘기지 않고 있었는데, 넘길 수 있는 것이 `retainGraph` 뿐이라 아무도 안 쓴 것이다.
   *
   * @param gradient 씨앗. 안 주면 스칼라에만 1 이 놓인다. 주면 **모양이 같아야
   *   한다** — 브로드캐스팅으로 맞춰 주지 않는다. 어긋난 씨앗은 값이 그럴듯한 채로
   *   틀린 기울기를 내고, 그것은 학습이 안 되는 것으로만 드러난다.
   * @param retainGraph 그래프를 남겨 다시 흘릴 수 있게 한다.
   */
  backward(gradient?: Tensor, retainGraph = false): void {
    // **이 검사가 맨 앞이다.** torch 를 재보니 비스칼라이면서 requiresGrad 가 아닌
    // 텐서는 "스칼라가 아니다" 가 아니라 이쪽으로 거절한다 — 스칼라 여부보다 먼저
    // 본다. 코어(numpy)는 처음부터 그 차례였고 여기만 반대였다. 셋이 같은 자리에서
    // 다른 문구를 내면 옮겨 적은 코드가 어느 쪽을 잡아야 할지 모른다.
    if (!this.requiresGrad) {
      throw new RuntimeError(
        `element 0 of tensors ${TORCH.noGrad} and does not have a grad_fn: ` +
          "no_grad 안이었거나 흐름을 끊는 연산을 지났다.",
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
          "but got torch.complex64: `.real`·`.abs()` 로 실수를 만든 뒤 불러라.",
      );
    }
    let seed: Tensor;
    if (gradient === undefined) {
      if (this.size !== 1) {
        throw new RuntimeError(
          `${TORCH.nonScalarBackward}: 지금 모양은 [${this.shape}] 다 — ` +
            "기울기를 넘기거나 .sum() 을 먼저 불러라.",
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
          `${TORCH.secondBackward}. 다시 흘리려면 ` +
            "backward(undefined, true) 로 그래프를 남겨라.",
        );
      },
    });
  }

  // ── 장치 옮기기 ───────────────────────────────────────────────────────

  /**
   * 값을 호스트로 내린다. torch 의 `t.cpu()` 자리이고 **비동기다** — GPU 메모리를
   * 되가져오는 왕복이라 피할 길이 없다.
   *
   * **그래프를 끊는다.** torch 의 `.cpu()` 는 미분되지만 여기서는 그럴 수가 없다 —
   * 호스트에 커널이 없으므로 역방향이 지나갈 길이 없다. 이어 붙여 두고 조용히 0 을
   * 흘리느니 끊는다.
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
   * 값을 GPU 로 올린다. torch 의 `t.cuda()` 자리다.
   *
   * **`cpu()` 와 달리 동기다.** 올리는 것은 큐에 쓰기 하나여서 기다릴 것이 없다 —
   * 짝이 안 맞아 보이지만, 없는 왕복을 만들어 대칭을 꾸미는 것보다 낫다.
   */
  webgpu(): Tensor {
    if (this.host === null) return this;
    return new Tensor(dev().upload(this.host), this.shape, { dtype: this.dtype });
  }

  // ── 읽기 ──────────────────────────────────────────────────────────────

  /**
   * 값을 평평한 f32 배열로.
   *
   * **복소수는 인터리브 그대로 나온다** — 길이가 `size` 가 아니라 `2 × size` 이고
   * `[re, im, re, im, …]` 이다. 실수부만 원하면 `real()` 을, 짝으로 원하면
   * `viewAsReal()` 을 먼저 불러라. 여기서 실수부만 골라 주면 길이가 그럴듯해져서
   * 값을 잃은 것이 안 보인다.
   */
  async toArray(): Promise<Float32Array> {
    // 이미 호스트에 있으면 왕복이 없다. **사본을 준다** — 안쪽 저장을 그대로 내보내면
    // 받은 쪽이 그것을 고칠 때 텐서 값이 같이 바뀐다.
    if (this.host !== null) return this.host.slice();
    return dev().read(this.raw, this.floats);
  }

  /**
   * 모양과 값이 **정확히** 같은가. 허용 오차가 없다 — 그것이 `allclose` 와 다른 점이다.
   *
   * GPU 에서 읽어 온다. 판정 하나를 위해 왕복하는 것이 아깝지만, CPU 에 사본을 들고
   * 있다가 비교하면 GPU 에서 무슨 일이 있었는지를 못 본다.
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
   * @param equalNan 참이면 **NaN 끼리를 같다고 본다.** 기본은 거짓이라 안 같다.
   *
   * 골든 하네스는 이것을 **안 켠다** — 켜면 NaN 이 통과하면 안 되는 자리에서
   * 통과한다. 그것과 이 인자를 갖는 것은 다른 자리다: 켤지 말지는 부르는 쪽이 정한다.
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
   * `print(t)` 가 찍는 글자. **GPU 에서 읽어 오므로 비동기다.**
   *
   * 교재가 이 글자를 그대로 싣기 때문에 이 프로젝트는 이것도 명세로 본다 —
   * 자세한 이유는 `src/repr.ts` 에 있다.
   */
  async repr(): Promise<string> {
    // **복소수는 아직 못 찍는다.** `1.+2.j` 꼴을 torch 와 글자까지 맞추려면 그쪽
    // 자리맞춤 규칙을 재서 굳혀야 하고, 그건 아직 안 한 일이다. 반쯤 맞는 글자를
    // 내면 그것이 교재의 줄과 안 맞는데도 맞는 것처럼 보인다.
    if (isComplexDType(this.dtype)) {
      throw new RuntimeError(
        "complex64 의 repr 은 아직 없다 — `viewAsReal()` 로 실수 짝을 찍어라.",
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

  /** `torch.Size([2, 2])`. 모양도 찍히는 것이라 명세다. */
  sizeRepr(): string {
    return formatSize(this.shape);
  }

  /** 형을 바꾼다. 값은 그대로다 — 저장이 float32 하나이므로 옮길 것이 없다. */
  to(dtype: DType): Tensor {
    if (dtype === this.dtype) return this;
    // **복소수는 이름표 갈이로 오갈 수 없다.** 다른 형끼리는 저장이 float32 하나로
    // 같아서 이름만 바꾸면 되는데, 복소수만 칸당 두 개다 — 양쪽 어느 방향으로든
    // 이름만 바꾸면 버퍼 길이와 `size` 가 어긋난 채로 남는다.
    if (dtype === "complex64" || isComplexDType(this.dtype)) {
      throw new RuntimeError(
        `torch.${this.dtype} → torch.${dtype} 는 이름표 갈이로 안 된다 — ` +
          "복소수는 저장이 칸당 f32 두 개다. " +
          "`Tensor.complex(re, im)`·`viewAsComplex()`·`real()` 로 오가라.",
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
        "complex64 에는 item() 이 없다 — JS 에 복소수 값이 없다. " +
          "`real()`·`imag()` 로 갈라서 꺼내라.",
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
 * @param names 바인딩 이름들. **마지막이 출력**이고 그것만 쓰기 가능이다.
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

/** 표에서 단 메서드들의 타입. 위 루프와 짝이고, 하나만 고치면 어긋난다. */
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
}

export { noGrad } from "./autograd.js";

/**
 * 구역 하나를 열고 닫는다. 안에서 만든 GPU 버퍼는 나갈 때 놓는다.
 *
 * **학습 루프에 이것이 없으면 안 돈다.** 한 스텝이 중간 버퍼를 수천 개 만들고,
 * 자바스크립트의 쓰레기 수집은 GPU 메모리를 제때 안 놓아준다.
 *
 * @param keep 구역 밖으로 들고 나갈 텐서. 나머지는 놓는다.
 */
export async function scope<T>(
  body: () => Promise<T>,
  keep: () => readonly Tensor[] = () => [],
): Promise<T> {
  const d = dev();
  d.beginScope();
  try {
    return await body();
  } finally {
    // **`raw` 다.** 살려 둘 것 중에 복소수가 있으면 `buffer` 가 거절하고, 그러면
    // 구역을 닫는 자리에서 예외가 난다 — 수명 관리는 값의 형을 알 필요가 없다.
    d.endScope(keep().map((t) => t.raw));
  }
}

/** 구역이 닫혀도 살려 둔다. 파라미터와 옵티마이저 상태가 쓴다. */
export function keepAlive(t: Tensor): Tensor {
  // 호스트에 있는 것은 살릴 것이 없다 — 구역은 GPU 버퍼만 놓고, `Float32Array` 는
  // 자바스크립트의 쓰레기 수집이 알아서 가져간다. `keepAlive(await t.cpu())` 는
  // 자연스러운 줄이므로 여기서 거절하면 안 된다.
  if (t.device === "cpu") return t;
  dev().keep(t.raw);
  return t;
}

/** 놓은 버퍼 수를 세는 자리. 벤치가 누수를 본다. */
export function device(): Device {
  return dev();
}

/**
 * 그래프 마디 하나를 밖에서 만든다. **커널을 다른 파일에 쓰는 자리를 위한 문이다.**
 *
 * `Tensor.make` 는 비공개다 — 이 클래스 안의 연산들이 쓰는 것이라 그게 맞다. 그런데
 * `fft.ts` 처럼 **자기 커널을 가진 모듈**이 생기면 그 문이 필요해진다. 여기서 조건을
 * 다시 적는 대신(`gradMode` 검사와 `requiresGrad` 전파를 두 벌로 두면 언젠가 갈린다)
 * 같은 자리로 넘긴다.
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
