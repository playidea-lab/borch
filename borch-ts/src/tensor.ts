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
import { gauss, refuseGenerator, uniform } from "./random.js";
// **A cycle on purpose, and it holds because nothing is touched while loading.**
// `fft.ts` and `special.ts` import `Tensor` from here; these five methods import their
// bodies back. Every use is inside a method, so by the time one runs both modules have
// finished. The alternative was a `declare module` augmentation, which type-checks and
// runs — and which `site/build_api.py` does not read, so the five methods would exist
// and the API index would say they do not.
import {
  istft as istftImpl, type IstftOptions, stft as stftImpl, type StftOptions,
} from "./fft.js";
import {
  igamma as igammaImpl, igammac as igammacImpl, polygamma as polygammaImpl,
} from "./special.js";
import {
  byRank, type DType, floatsPerElement, isComplexDType, promote, rankOf,
} from "./dtype.js";
import {
  IndexError, LinAlgError, NotImplementedError, RuntimeError, TORCH,
} from "./errors.js";

/**
 * What the `_ex` family puts in `info` for a matrix that came out spoilt.
 *
 * LAPACK stores which leading pivot was zero. Here it says only "spoilt", while keeping
 * the digit count — asked of real torch with a 2×2 singular matrix, it was 2.
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
 * How `interpolate` resamples. **torch's `linear` and `trilinear` are absent because
 * they want a rank this function does not take** — it is written for `[N, C, H, W]`,
 * and those two are the 3-D and 5-D members of the same family. `nearest` and
 * `nearest-exact` differ by half an output cell; `area` is adaptive average pooling
 * under another name.
 */
export type InterpolateMode =
  "nearest" | "nearest-exact" | "area" | "bilinear" | "bicubic";

/**
 * SELU's fixed point. The value `alphaDropout` puts where it dropped comes from here.
 *
 * Filling with this number rather than 0 is what keeps SELU's self-normalisation —
 * confirmed by measurement (with an all-ones input the answers are the two values
 * `-0.779` and `1.666`).
 */
const ALPHA_PRIME = -1.7580993408473766;

/** One number means the same value on every axis. */
function pairOf(v: number | readonly number[]): [number, number] {
  return typeof v === "number" ? [v, v] : [v[0] ?? 0, v[1] ?? 0];
}

/**
 * A `(C·kh·kw, L)` table of positions. The values are flat positions in the **padded**
 * input.
 *
 * `unfold` gathers from these positions and `fold` adds into them — the two share one
 * table, so one's backward *is* the other.
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

/** For bilinear upsampling, **which two input positions in what proportion** each
 *  output position mixes. */
/**
 * @param given the scale the caller asked for, when it is to be used **instead of**
 *   `sizeIn / sizeOut`. Those two are the same number only when the output divides
 *   exactly; at `scale_factor = 1.5` on a 3-high input they are `0.6667` and `0.75`,
 *   and that is the whole of what `recompute_scale_factor` selects between. This
 *   function had only the second, so borch.ts answered as though the flag were always
 *   on — invisible at whole factors, which is what anybody tries first.
 */
function bilinearAxis(sizeIn: number, sizeOut: number, alignCorners: boolean,
                      given: number | null = null) {
  const lo: number[] = [];
  const hi: number[] = [];
  const frac: number[] = [];
  const step = given === null ? sizeIn / sizeOut : 1 / given;
  for (let i = 0; i < sizeOut; i++) {
    const src = alignCorners
      ? i * ((sizeIn - 1) / Math.max(1, sizeOut - 1))
      : Math.max(0, (i + 0.5) * step - 0.5);
    const base = Math.floor(src);
    lo.push(base);
    hi.push(Math.min(base + 1, sizeIn - 1));
    frac.push(src - base);
  }
  return { lo, hi, frac };
}

/**
 * For each output position, **which input position it reads.**
 *
 * The four modes differ here and nowhere else. Extending `[0,1,2]` by 2 in front and 1
 * behind (asked of real torch and matched position by position):
 *
 *     reflect    2 1 [0 1 2] 1   ← mirrors at the edge **without repeating the edge**
 *     replicate  0 0 [0 1 2] 2   ← stretches the edge
 *     circular   1 2 [0 1 2] 0   ← takes from the far side
 *
 * Reduced to one index list, the forward is `indexSelect` and the backward is the
 * **gathering sum** that already does — mirror and wrap read one input several times, so
 * overwriting would lose exactly that much, and none of it has to be written again here.
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
import * as LA from "./_linalg.js";
// **This is a circular import** — `special.ts` uses `Tensor`. It is used inside method
// bodies, so it is called after the modules have finished loading, and it works. It must
// not be used at the top level.
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
  poolOut,
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

/** The device is held **inside an object**. Same reason as `autograd.ts`'s `gradMode`. */
const deviceHolder: { current: Device | null } = { current: null };

/**
 * Integer exponents up to this are expanded into multiplications.
 *
 * The higher it goes the more kernel calls it costs, so it does not expand without
 * limit. Above this it goes to the `pow` kernel, and there a negative base has no
 * answer.
 */
const MAX_UNROLLED_POWER = 8;

/**
 * The limit up to which float32 counts every integer (2^24).
 *
 * Above it two adjacent integers fold onto the same float — which is why `randint`
 * stops there. A value quietly rounded shuffles the labels and nobody sees it.
 */
const EXACT_INT_LIMIT = 16_777_216;

/**
 * The dtype a reduction produces. **The dividing line is "does it make a value"** — the
 * same line drawn for the shape and indexing operations, confirmed by asking torch about
 * thirty-three places (`tests/test_reduce_dtype.py`).
 *
 * Accumulations (`sum`, `prod`, `cumsum`, `cumprod`) **make** a value — a true/false
 * cell cannot hold 3, so bool is promoted to int64. Selections (`amax`, `amin`, `max`,
 * `min`) **hand over** a value that was already there, so the dtype passes through
 * unchanged. Reductions were never the exception; the two are simply different things.
 */
function accumulated(from: DType): DType {
  return from === "bool" ? "int64" : from;
}

/**
 * What can go into one axis of `at()`. The syntax is written down in
 * `indexing.ts`.
 */
export type AtIndex = number | null | Slice | Tensor | readonly number[];

/** One plan into a real operation. **No value is made here** — it is sent through a
 *  door that already exists. */
function applyPlan(t: Tensor, axis: number, plan: AxisPlan): Tensor {
  switch (plan.kind) {
    case "whole":
      return t;
    case "int":
      return t.select(axis, plan.at);
    case "range":
      return t.narrow(axis, plan.start, plan.length);
    case "picks":
      // A strided slice and an index list meet here — both are "these positions".
      return t.indexSelect(
        axis,
        Tensor.from(plan.indices, [plan.indices.length], { dtype: "int64" }),
      );
  }
}

/**
 * Caches single-value constant tensors by their value.
 *
 * A training loop rebuilds the same constants every step — the learning rate, eps, 0.5,
 * a gate count. The buffer is four bytes, so holding one is far cheaper than making one.
 * They are marked to survive the scope closing (otherwise the next step points at a
 * released buffer).
 *
 * **Writes do not come through here.** What arrives is a constant, and an in-place
 * operation only ever writes to its own buffer.
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
 * The largest finite value float32 can hold.
 *
 * `nanToNum` folds infinity onto this — torch uses the dtype's extreme too when none is
 * given. `Number.MAX_VALUE` is double precision's extreme, and put into an f32 buffer it
 * becomes infinity again.
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

// **The quantile's interpolation moved inside `quantileOver`.**
//
// There was a helper here that produced the value alone, and a tensor built from that
// value **has no graph.** Gathering back through the sorted positions gives the same
// value and sends the gradient to the two positions the interpolation used — which is
// this operation's rule.

/**
 * The zeroth-order modified Bessel function. `kaiserWindow` is built on the CPU, so a
 * copy is needed here too.
 *
 * It uses **the same tables** as the shader's `i0_` (Abramowitz & Stegun 9.8.1 and
 * 9.8.2). Written differently in the two places, the golden cannot say which is right.
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
 * ── Building index lists ───────────────────────────────────────────────────
 *
 * Positions determined by the shape alone are produced as **flat indices**. The core
 * (numpy) builds the same thing with `np.arange(size).reshape(shape)[...]`; there is no
 * such view here, so the index arithmetic is written by hand. **That makes two copies,
 * so the golden is the judge** — with real torch as a third answer, and all three
 * matched.
 */

/** Row-major strides. How far the flat index jumps for one step along each axis. */
function rowStrides(shape: readonly number[]): number[] {
  const out = new Array<number>(shape.length).fill(1);
  for (let i = shape.length - 2; i >= 0; i--) {
    out[i] = out[i + 1]! * (shape[i + 1] ?? 1);
  }
  return out;
}

/** Walks `shape` in row-major order, calling `at` at every coordinate. */
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

/** The positions the strides point at. They may overlap and they may skip. */
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

/** The positions of `x[..., start:stop:step, ...]`. */
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

/** The positions of `select(dim, index)`. That axis disappears from the result. */
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
 * The diagonal's positions. **That axis goes to the end** — as in torch and in numpy.
 *
 * With a batch axis present, missing this convention leaves every value right and only
 * the order wrong. Measured at two dimensions alone it does not show, because no axis is
 * left over.
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
 * A histogram's edges. **When `min === max` it uses the data's own range** (measured).
 *
 * With the data at a single value that range is 0, so it opens out by half a bin on each
 * side — otherwise every edge is the same number and the bin width is 0.
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

/** The bin a value falls in. **Outside the range is -1**, and the right end goes into
 *  the last bin. */
function slotOf(value: number, edges: readonly number[]): number {
  const last = edges.length - 1;
  if (value < (edges[0] ?? 0) || value > (edges[last] ?? 0)) return -1;
  for (let i = 1; i <= last; i++) {
    if (value < (edges[i] ?? 0)) return i - 1;
  }
  return last - 1;
}

/** Counts into the bins `edges` divides. **Outside the range is discarded** — as torch
 *  does (measured). */
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

/** The positions taken by row `at` along axis `dim`. Used by `uniqueConsecutive`. */
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

/** Comparisons and logical operations produce true/false whatever the input was. */
const BOOL_RESULT = new Set([
  "eq", "ne", "lt", "le", "gt", "ge", "logical_and", "logical_or",
]);

/** An arithmetic operation's name into the promotion rule's symbol. What is not in the
 *  table keeps the higher category as it is. */
const ARITH: Readonly<Record<string, "+" | "-" | "*" | "/">> = {
  add: "+", sub: "-", mul: "*", div: "/",
};

function resultDType(name: string, a: DType, b: DType): DType {
  if (BOOL_RESULT.has(name)) return "bool";
  const op = ARITH[name];
  if (op) return promote(a, b, op);
  // The ones not in the table, such as `maximum` and `pow`. No new dtype is invented;
  // the higher side is used.
  return byRank(Math.max(rankOf(a), rankOf(b)));
}

/** `shape` right-aligned to `out`'s rank — used by `reduceBroadcast`. */
function padShape(shape: readonly number[], rank: number): number[] {
  const out: number[] = new Array<number>(rank).fill(1);
  for (let i = 0; i < rank; i++) {
    const src = shape.length - rank + i;
    if (src >= 0) out[i] = shape[src] ?? 1;
  }
  return out;
}

/**
 * The refusal for a dtype this subset **has no cell for**. It uses the two Python
 * versions' sentence verbatim — `borch/_base.py`'s `_unsupported` and the binding's
 * `_absent_dtype` are the same characters.
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
   * The buffer when the value is on the GPU, `null` when it is on the host. **Do not
   * read it directly** — the door from outside is the `buffer` getter, and that one
   * checks the device.
   */
  private readonly gpu: GPUBuffer | null;
  /**
   * **Which life** the buffer was on when this was born. The pair of `device.ts`'s
   * `age`.
   *
   * When a scope closes the buffer returns to the pool and its life count rises. If this
   * tensor reaches for its value after that, the two numbers disagree — and then it is
   * **a tensor that is already dead**.
   */
  private readonly age: number;
  /** The array when the value is on the host, `null` when it is on the GPU. */
  private readonly host: Float32Array | null;
  requiresGrad: boolean;
  grad: Tensor | null = null;
  freed = false;
  /** Set by `retainGrad()` — see there. */
  retainsGrad = false;
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
    // The two storages arrive through one slot. Forty-seven places inside pass a
    // buffer, and splitting here beats fixing all of them — the split stays in one
    // place.
    const onHost = storage instanceof Float32Array;
    this.gpu = onHost ? null : storage;
    this.host = onHost ? storage : null;
    this.age = this.gpu === null ? 0 : dev().age(this.gpu);
    this.shape = [...shape];
    this.size = numel(this.shape);
    this.parents = options.parents ?? [];
    this.gradName = options.gradName ?? "";
    this.dtype = options.dtype ?? "float32";
    // If any parent carries a gradient, this does. Inside no_grad nobody does.
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
    // **Complex numbers are stopped here too — one door used twice.**
    //
    // A complex buffer holds two f32 per cell. A kernel that does not know that reads
    // the first half as reals and produces a wrong answer **with no exception.** Putting
    // a guard on each of 176 entry points is not a thing that can be done, and every one
    // of them passes through this getter.
    //
    // Code that understands complex comes in through `raw`. So **the default is
    // refusal**, and somebody adding an operation who does nothing gets an operation
    // that does not accept complex — the other way round, an operation where nothing was
    // done eats complex quietly and wrongly.
    if (isComplexDType(this.dtype)) {
      throw new RuntimeError(
        "this operation does not take complex64 yet — the storage is two f32 per slot " +
          "(interleaved), so a kernel that does not know about complex numbers reads it " +
          "and is quietly wrong. Use view_as_real to get a real tensor, or handle the " +
          "real and imaginary parts separately.",
      );
    }
    // **Dead tensors are stopped here too — one door used three times.**
    this.refuseIfDead();
    return this.gpu;
  }

  /**
   * Stops a tensor that is used after its scope closed.
   *
   * **Without this, somebody else's values come out in silence** (measured: leak
   * `[1,2,3,4]`, take a few more allocations of the same size, read it back, and it is
   * `9,9,9,9`). WebGPU does not stop it either, because the buffer was not destroyed but
   * returned to the pool — reading a valid buffer validly is exactly what it is.
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
   * Makes one node in the graph. The same place as the core's `_make`.
   *
   * Inside `no_grad` neither parents nor a backward are attached — attaching them and
   * not using them keeps the buffers alive and leaks. It is not quietly wrong, and in a
   * training loop that is fatal all the same.
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

  // ── Creation ──────────────────────────────────────────────────────────

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
    // **Complex cannot come in through this door.** Labelling it `complex64` alone
    // gives `n` cells where the storage convention demands `2n`, so the second half is
    // somebody else's memory — read as any value at all, with no exception. There is one
    // place that assembles them.
    if (dtype === "complex64") {
      throw new RuntimeError(
        "Tensor.from cannot make complex64 — the storage is two f32 per slot. Use " +
          "Tensor.complex(re, im), Tensor.polar(r, theta), or x.viewAsComplex().",
      );
    }
    if (requiresGrad && dtype !== "float32") {
      // Gradients are not defined for integers and booleans. torch stops here too —
      // let it through and training runs while the values mean nothing at all.
      throw new RuntimeError(
        "Only Tensors of floating point and complex dtype can require gradients",
      );
    }
    // Asked to stay on the host, it holds a copy. Holding the given array as it is
    // means the tensor's value changes when the caller edits it — on the GPU side
    // `upload` copies, so that place does not exist.
    if (device === "cpu") return new Tensor(flat.slice(), shp, { requiresGrad, dtype });
    return new Tensor(dev().upload(flat), shp, { requiresGrad, dtype });
  }

  static full(shape: readonly number[], value: number): Tensor {
    const n = numel(shape);
    // **A scalar calls no kernel.** Sending a dispatch to write one element is pure
    // waste, and every expression like `x * 0.5` arrives here — 286 times in a single
    // ResNet step (measured: 17% of 1,636 dispatches). On top of that a training loop
    // repeats the same constant every step, so they are cached by value.
    if (n === 1) {
      const hit = scalarCache.get(value);
      if (hit) return new Tensor(hit, shape);
      const buf = dev().upload(Float32Array.of(value));
      dev().keep(buf);
      scalarCache.set(value, buf);
      return new Tensor(buf, shape);
    }
    // **Infinity and NaN cannot be baked into a shader.** WGSL forbids a
    // compile-time-evaluated value from becoming inf or NaN — a literal and
    // `bitcast<f32>(0x7f800000u)` are refused identically, as
    // `value inf cannot be represented as 'f32'` (measured). So this one case is filled
    // on the CPU and uploaded. It is one upload, which is shorter than the kernel path.
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
   * The identity matrix. **Built on the CPU and uploaded** — it is built once, and
   * baking another shader for it costs more than it buys.
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

  /** `0` through `n-1`. */
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
   * A step of 0 **stops here.** The same place torch stops.
   *
   * Unblocked, `(to - from) / 0` becomes Infinity and it blows up where the array is
   * allocated, and that message is indistinguishable from running out of memory — what
   * was passed wrongly is invisible. With a step of 0 the value never moves, so there is
   * no answer, not a large one.
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
      // **int64 lives in an f32 cell.** Past 2^24 it can no longer count, so it stops
      // there — better than rounding in silence. `randint` says the same thing at the
      // same place.
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
    // Computing the last value lets rounding accumulate so it never lands exactly on
    // end. It is pinned.
    const step = count > 1 ? (end - start) / (count - 1) : 0;
    for (let i = 0; i < count; i++) data[i] = start + step * i;
    if (count > 1) data[count - 1] = end;
    return Tensor.from(data, [count]);
  }

  /**
   * The shared skeleton of the five window functions. **Built on the CPU and
   * uploaded** — the same reason as `eye`.
   *
   * **`periodic` is the default and it adds one to the length.** When true it builds a
   * symmetric window of `N+1` and drops the last (measured: `hannWindow(5)` equals
   * exactly the first five of the symmetric 6). Asked only with false, that rule never
   * shows.
   *
   * `n === 1` is kept apart — the divisor (`total - 1`) becomes 0.
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
  /**
   * **`p` is torch's other form and it was not here.** Given nothing, the tensor's own
   * values are the probabilities; given a number, that probability is used everywhere
   * and the values are ignored. The core's docstring records both and this side had
   * only the first, so `x.bernoulli(0.5)` was a surplus argument — discarded, and the
   * draw came from the values instead of from `0.5`.
   *
   * `out` is refused rather than carried further: this library has no `out=` anywhere,
   * and a seat that silently allocates instead of writing where it was told is worse
   * than one that says so.
   */
  bernoulli(p?: number, generator?: never, out?: never): Tensor {
    // **`generator` is the second seat and `out` the third**, keyword-only in torch
    // (`bernoulli(p, *, generator, out)`). The first draft of this took `(p, out)` and
    // put `out` where `generator` sits — a drop from the middle rather than a short
    // tail, which the signature axis caught as `dropped` within the hour.
    if (generator !== undefined) {
      throw new Error(
        "bernoulli(generator=…) — there is one stream here; seed it with manualSeed");
    }
    if (out !== undefined) {
      throw new Error("bernoulli(out=…) — writing into a given tensor is not here");
    }
    const probs = p === undefined ? this : Tensor.full(this.shape, p);
    return Tensor.rand(this.shape).binary("lt", probs).to(this.dtype);
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
   * The positions of the lower and upper triangles. **It is a `(2, count)` table**
   * (measured) — not pairs of positions but a row of rows and a row of columns. Read as
   * a list of pairs, even the shape is different.
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
    // The first column is 1 and the rest are x — a cumulative product along that row
    // gives powers 0..N-1 in order.
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

  // ── Elementwise ───────────────────────────────────────────────────────

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
    // **With a complex operand it branches here.** This method is the only door for
    // binary operations, so not only `add` and `mul` but backward's gradient
    // accumulation passes through — and accumulation leaking into a real kernel adds
    // only the first half of a complex leaf's gradient.
    if (this.isComplex() || other.isComplex()) {
      if (name === "add" || name === "sub" || name === "mul" || name === "div") {
        return this.complexBinary(name, other);
      }
      // The rest must not fall through to the real kernel below. The `buffer` getter
      // would block it, and its wording is "this operation does not yet", which leaves
      // no record of which operation.
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

  // ── Matrix products ───────────────────────────────────────────────────

  /**
   * torch's `matmul`, which is what `@` maps to. Five cases, and they are
   * not variations of one rule — **each was measured against torch rather
   * than reasoned about**, because the 1-D ones do not follow from the
   * others:
   *
   * | this | other | result | what happens |
   * |------|-------|--------|--------------|
   * | 1-D  | 1-D   | scalar | inner product |
   * | 2-D  | 2-D   | 2-D    | plain `mm` |
   * | 1-D  | n-D   | drops one | a 1 is prepended, then removed after |
   * | n-D  | 1-D   | drops one | a 1 is appended, then removed after |
   * | n-D  | m-D   | broadcast | leading dims broadcast, last two multiply |
   *
   * The prepended and appended 1s are **removed from the result**, which is
   * why `[3] x [5,3,4]` gives `[5,4]` and not `[5,1,4]`. That asymmetry is
   * the part worth checking against torch before trusting any of it.
   *
   * ## Why this is built out of `bmm` rather than a new kernel
   *
   * `bmm` is `mm` per batch element and `mm` owns its backward, so the
   * expression written here *is* the graph — the same reasoning `mm` gives
   * for splitting a complex product into four real ones. A batched kernel
   * would be faster and is a separate job; being able to run a transformer
   * at all comes first.
   *
   * **The seat is `other`, and torch's own documentation says otherwise.**
   * `torch.Tensor.matmul.__doc__` opens with `matmul(tensor2) -> Tensor`, but calling
   * it with that keyword raises and `other=` is what works. Two sessions read the doc
   * and landed on `tensor2` separately, one of them after trying it and reverting;
   * neither had called it with a keyword. The name a caller can write is the name.
   */
  matmul(other: Tensor): Tensor {
    const a = this.shape.length;
    const b = other.shape.length;
    if (a === 0 || b === 0) {
      throw new RuntimeError(
        `both arguments to matmul need to be at least 1-D, ` +
          `but they are ${a}-D and ${b}-D`,
      );
    }
    if (a === 1 && b === 1) return this.dot(other);
    if (a === 2 && b === 2) return this.mm(other);
    // A 1-D side becomes 2-D for the duration and the axis it borrowed is
    // taken back at the end. `left`/`right` say which one to take back.
    if (a === 1) return other.mmBatched(this.unsqueeze(0), true).squeeze(-2);
    if (b === 1) return this.mmBatched(other.unsqueeze(1), false).squeeze(-1);
    return this.mmBatched(other, false);
  }

  /**
   * The n-D by n-D case of {@link matmul}: broadcast everything but the last
   * two axes, fold what is left into one batch axis, and hand it to `bmm`.
   *
   * `flipped` marks that the caller swapped the operands to reuse this path
   * (`1-D x n-D` is the only one that does), so the product is taken the
   * other way round.
   */
  private mmBatched(other: Tensor, flipped: boolean): Tensor {
    const left = flipped ? other : this;
    const right = flipped ? this : other;
    const ls = left.shape;
    const rs = right.shape;
    const M = ls[ls.length - 2] ?? 0;
    const K = ls[ls.length - 1] ?? 0;
    const K2 = rs[rs.length - 2] ?? 0;
    const N = rs[rs.length - 1] ?? 0;
    if (K !== K2) {
      throw new RuntimeError(
        `mat1 and mat2 ${TORCH.matmulShape} ` +
          `(${M}x${K} and ${K2}x${N})`,
      );
    }
    // The batch axes broadcast against each other exactly as elementwise
    // operations do, so the rule is not reinvented here.
    const lb = ls.slice(0, -2);
    const rb = rs.slice(0, -2);
    const rank = Math.max(lb.length, rb.length);
    const batch: number[] = [];
    for (let i = 0; i < rank; i++) {
      const x = lb[lb.length - rank + i] ?? 1;
      const y = rb[rb.length - rank + i] ?? 1;
      if (x !== y && x !== 1 && y !== 1) {
        throw new RuntimeError(
          `${TORCH.matmulShape}: batch dimensions [${lb}] and [${rb}] ` +
            "do not broadcast",
        );
      }
      batch.push(Math.max(x, y));
    }
    const total = batch.reduce((n, d) => n * d, 1);
    const l3 = left.broadcastTo([...batch, M, K]).reshape([total, M, K]);
    const r3 = right.broadcastTo([...batch, K, N]).reshape([total, K, N]);
    return l3.bmm(r3).reshape([...batch, M, N]);
  }

  /**
   * Two-dimensional only — {@link matmul} is the one that batches.
   */
  mm(mat2: Tensor): Tensor {
    if (this.shape.length !== 2 || mat2.shape.length !== 2) {
      throw new Error(
        `mm is 2-D by 2-D: [${this.shape}] x [${mat2.shape}]. ` +
          "Batching is not here yet.",
      );
    }
    const M = this.shape[0] ?? 0;
    const K = this.shape[1] ?? 0;
    const K2 = mat2.shape[0] ?? 0;
    const N = mat2.shape[1] ?? 0;
    if (K !== K2) {
      throw new RuntimeError(
        `mat1 and mat2 ${TORCH.matmulShape} ` +
          `(${M}x${K} and ${K2}x${N})`,
      );
    }
    // **Complex splits into four real matrix products.**
    //
    //   `(A + iB)(C + iD) = (AC − BD) + i(AD + BC)`
    //
    // No new kernel — `sum` and `diagflat` pass the same place the same way. The
    // backward is free too: `real`, `imag`, `complex` and the real `mm` all know their
    // own backward, so this expression *is* the graph.
    //
    // One-sided complex comes here as well — the real side's imaginary part is 0 so two
    // of the products vanish, and specialising that is a job for after measuring.
    // Correct comes first.
    if (this.isComplex() || mat2.isComplex()) {
      const a = this.isComplex() ? this : this.asComplexRe();
      const b = mat2.isComplex() ? mat2 : mat2.asComplexRe();
      const [ar, ai] = [a.real(), a.imag()];
      const [br, bi] = [b.real(), b.imag()];
      return Tensor.complex(
        ar.mm(br).sub(ai.mm(bi)),
        ar.mm(bi).add(ai.mm(br)));
    }
    const out = dev().alloc(M * N);
    dev().run(
      dev().pipeline(`mm:${M}:${K}:${N}`, () => matmul(M, K, N)),
      [this.buffer, mat2.buffer, out],
      [Math.ceil(N / 64), Math.ceil(M / 64), 1],
    );
    return Tensor.make(
      out,
      [M, N],
      [this, mat2],
      (g) => [
        this.requiresGrad ? g.mm(mat2.transpose()) : null,
        mat2.requiresGrad ? this.transpose().mm(g) : null,
      ],
      "MmBackward0",
    );
  }

  /**
   * Swaps two axes, as torch's `transpose(dim0, dim1)` does — **any rank.**
   *
   * It used to take no arguments and refuse anything but a matrix, while
   * `swapaxes(axis0, axis1)` — the name torch documents as this one's alias —
   * sat next to it handling every rank. So the operation a porting reader
   * reaches for first was the narrower of the two, and bimm's ViT wrote
   * `permute([0, 2, 1])` to swap the last two axes of an attention score
   * matrix because `transpose()` threw.
   *
   * **The seats are `dim0`/`dim1` because that is what torch answers to** —
   * `t.transpose(dim0=-2, dim1=-1)` works there. Checked by calling it, not
   * by reading the signature line: torch's doc and torch's runtime disagree
   * on other methods, so reading is not measuring.
   *
   * The no-argument call still means "swap the two axes of a matrix", which
   * is what every existing caller in this file meant by it.
   */
  transpose(dim0?: number, dim1?: number): Tensor {
    if (dim0 !== undefined || dim1 !== undefined) {
      return this.swapaxes(dim0 ?? 0, dim1 ?? 1);
    }
    if (this.shape.length !== 2) {
      throw new Error(
        `transpose() with no arguments is the 2-D swap: [${this.shape}]. ` +
        "Pass the two dimensions, as torch does — transpose(dim0, dim1).",
      );
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

  // ── Reductions ────────────────────────────────────────────────────────

  /** Adds everything into one scalar. The starting point of `backward()`. */
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
  sum(dim?: DType | number, keepdim = false, dtype?: DType): Tensor {
    // **torch has two overloads under this one name** — `sum(dtype)` folds the whole
    // tensor and `sum(dim, keepdim, dtype)` folds one axis — and only the first was
    // here, with the axis form living next door as `sumDim`. The neighbouring comment
    // on `variance` called that arrangement *safe by accident*: `x.sum(0)` would not
    // compile because the first parameter was a `DType`. Safe and wrong are different
    // things — the line a reader transcribes from torch has to work, not merely fail
    // loudly, and this was the last row in the axis's `shifted` bucket.
    //
    // Told apart the way torch tells them apart: a string is a dtype, a number is an
    // axis.
    if (typeof dim === "number") return this.sumDim(dim, keepdim, dtype);
    const only = dim ?? dtype;
    if (only !== undefined) return this.castFirst(only).sum().to(only);
    // **Complex splits into two real reductions.** The sum applies to the real and
    // imaginary parts separately, so no new kernel is needed — `real`, `imag` and
    // `complex` already exist and all three know their backward.
    //
    // This place has to exist for "backward on a complex loss is refused" to be refused
    // **at the refusing place.** Without it the `sum()` before it blocks first, and then
    // the same exception type comes out, the case passes, and the place it meant to ask
    // about is never reached — a pass that proves nothing.
    if (this.isComplex()) return Tensor.complex(this.real().sum(), this.imag().sum());
    const out = dev().sumAll(this.buffer, this.size);
    const shape = this.shape;
    return Tensor.make(
      out,
      [],
      [this],
      // d(sum)/dx is 1 everywhere, so the seed is expanded to the shape.
      (g) => [foldFrom(g, shape)],
      "SumBackward0",
      accumulated(this.dtype),
    );
  }

  /**
   * Folds one axis. Without `dim` it folds everything into a scalar.
   *
   * Only the full sum goes through `Device.sumAll`'s tree — the axis-reduction kernel
   * has one thread walking the axis, so used for a full reduction it is one thread going
   * round n times.
   */
  private reduceOver(kind: ReduceKind, dim?: number, keepdim = false): Tensor {
    if (dim === undefined) {
      if (kind === "sum") return this.sum();
      // A full max or min is viewed flat and then folded along one axis.
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

  /** Views the same buffer as 1-D. The element order is unchanged, so no copy. */
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
   * Called first by any reduction that was given `dtype=`.
   *
   * **The rule is one line: cast before folding.** Not after — measurement separates the
   * two: `[1.7, −2.3, 0.9].sum(dtype=int64)` is `−1`. Folding first gives `0.3`, which
   * truncates to `0`; casting first gives `[1, −2, 0]`, which sums to `−1`.
   *
   * The result dtype is pinned at the end too — otherwise the accumulation rule promotes
   * again and `sum(dtype=bool)` comes out int64 (torch gives `true`).
   */
  private castFirst(dtype: DType): Tensor {
    return this.dtype === dtype ? this : this.to(dtype);
  }

  mean(dim?: number, keepdim = false, dtype?: DType): Tensor {
    if (dtype !== undefined) {
      // **Being asked to land on an integer is refused** (measured). What `dtype=`
      // releases is the refusal on the **input** side alone — a mean whose result is an
      // integer still has no answer.
      if (dtype !== "float32" && dtype !== "complex64") {
        throw new RuntimeError(
          "mean(): could not infer output dtype. Input dtype must be either " +
            "a floating point or complex dtype");
      }
      return this.castFirst(dtype).mean(dim, keepdim).to(dtype);
    }
    // **It stops where torch stops** (measured). A division or a square root has no
    // answer that fits an integer cell — promoted quietly to float the way numpy does,
    // that code then breaks on real torch.
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
    // **The dtype is passed on.** Without it `x.to("int64").detach().dtype` is float32
    // — the value is the same buffer and does not change, and the label alone diverges
    // in silence. For complex that label *is* the storage convention, so losing it reads
    // as the second half having disappeared.
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
    // **Integers and booleans are accepted and float32 comes out** (measured).
    // Promoted to float, the assembly below runs unchanged and the result dtype follows
    // — `logcumsumexp` is one torch refuses, so it diverges here (a hole in torch's
    // kernels rather than a rule).
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
  nanmean(dim?: number, keepdim = false, dtype?: DType): Tensor {
    if (dtype !== undefined) {
      // As `mean` above: `dtype=` releases the refusal on the **input** side, and a
      // result that has to land on an integer still has no answer. torch stops in the
      // same place — measured, `softmax(dtype=int64)` is a `NotImplementedError` and
      // `nanmean(dtype=int64)` says it could not infer the output dtype.
      if (dtype !== "float32" && dtype !== "complex64") {
        throw new RuntimeError(
          "nanmean(): could not infer output dtype. Input dtype must be either " +
            "a floating point or complex dtype");
      }
      return this.castFirst(dtype).nanmean(dim, keepdim).to(dtype);
    }
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
  clone(memoryFormat: string | null = null): Tensor {
    if (memoryFormat !== null) {
      throw new Error("clone(memoryFormat) is not here — there is one layout.");
    }
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

  /** Fills the missing **leading** axes with 1. What `atleast_2d` does. */
  private static lift(t: Tensor, rank: number): Tensor {
    if (t.shape.length >= rank) return t;
    return t.reshape([...new Array<number>(rank - t.shape.length).fill(1), ...t.shape]);
  }

  /**
   * torch's `atleast_3d`. **It appends an axis** — it does not prepend one.
   *
   * 1-D `(n,)` becomes `(1, n, 1)` and 2-D `(m, n)` becomes `(m, n, 1)`. Filling only at
   * the front makes `dstack` join along the last axis rather than the third, and even
   * the shape differs.
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
   * Variance over `dim`, or over the whole tensor when `dim` is left out.
   *
   * **torch's default is the unbiased estimate (dividing by n-1)** — left at
   * `correction=0` the value comes out subtly smaller, and that becomes the
   * place it diverges quietly inside a normalisation layer.
   *
   * **`dim` is the first argument because torch puts it there**, and because
   * every neighbouring reduction here already does — `mean(dim, keepdim,
   * dtype)`, `sumDim`, `amax`. This pair took `correction` first, alone among
   * them, and `x.std(0)` is a line anybody transcribing torch writes: it
   * compiled, it ran, and it returned a scalar at correction 0 where torch
   * returns one value per column. Not a crash, an answer — and a different
   * rank, so it broke somewhere else entirely.
   *
   * The signature axis (`tests/ts_signatures.py`) is what found it. The
   * neighbour `sum(dim, keepdim, dtype)` has the same shape and is safe by
   * accident: its first argument is a `DType`, so `x.sum(0)` will not
   * compile. `number` is not narrow enough to catch anything.
   */
  variance(dim?: number, correction = 1, keepdim = false): Tensor {
    // **It stops where torch stops** (measured). A division or a square root has no
    // answer that fits an integer cell — promoted quietly to float the way numpy does,
    // that code then breaks on real torch.
    this.needsFloat("variance is for floating point only", "std and var only support floating point and complex dtypes");
    // **Detaching the mean leaves the gradient unchanged.** The share that passes
    // through the mean is proportional to Σ(x−m), and that sum is 0 by definition, so it
    // vanishes entirely. Left attached it becomes a computation where two large terms
    // cancel, so detaching is also the more accurate value.
    if (dim === undefined) {
      const n = this.size;
      const centered = this.sub(this.mean().detach());
      return centered.square().sum().div(Tensor.full([], n - correction));
    }
    const rank = this.shape.length;
    const axis = dim < 0 ? dim + rank : dim;
    const n = this.shape[axis] ?? 1;
    // The mean keeps the axis so it broadcasts back over the centred values; the fold
    // that follows is the one that answers to `keepdim`.
    const centered = this.sub(this.mean(axis, true).detach());
    return centered.square().sumDim(axis, keepdim).div(Tensor.full([], n - correction));
  }

  /** The standard deviation. See `variance` for why `dim` comes first. */
  std(dim?: number, correction = 1, keepdim = false, unbiased?: boolean): Tensor {
    if (unbiased !== undefined) return this.std(dim, unbiased ? 1 : 0, keepdim);
    // **It stops where torch stops** (measured). A division or a square root has no
    // answer that fits an integer cell — promoted quietly to float the way numpy does,
    // that code then breaks on real torch.
    this.needsFloat("std is for floating point only", "std and var only support floating point and complex dtypes");
    return this.variance(dim, correction, keepdim).sqrt();
  }

  /**
   * Removes the named axes when they are of size 1.
   *
   * **Two of torch's rules, both once absent.** `squeeze(0, 2)` names several axes,
   * and an axis whose length is not 1 is **left alone** rather than refused —
   * `x.squeeze(1)` on `[1, 2, 3]` gives `[1, 2, 3]` in torch and threw here.
   *
   * Refusing looks like the safer of the two and is not: it is torch's own answer
   * that is being refused, so a line copied from torch stops on a shape torch is
   * happy with. The core carried the same pair of gaps.
   */
  squeeze(...dim: number[]): Tensor {
    const rank = this.shape.length;
    // A 0-d tensor accepts -1 and 0, and both are no-ops — torch counts one axis
    // for the purpose of naming one.
    const span = Math.max(rank, 1);
    const axes = dim.map((d) => (d < 0 ? d + span : d));
    for (const one of axes) {
      if (one < 0 || one >= span) {
        throw new IndexError(
          `squeeze(): dimension ${one} is out of range for a tensor of rank ${rank}.`
          + `\n(torch: Dimension out of range (expected to be in range of [${-span}, `
          + `${span - 1}], but got ${one}))`);
      }
    }
    const keep = axes.filter((one) => rank > 0 && this.shape[one] === 1);
    if (keep.length === 0) return this.reshape([...this.shape]);
    const outShape = this.shape.filter((_d, i) => !keep.includes(i));
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

  // ── Shape ─────────────────────────────────────────────────────────────

  /** This tensor's contiguous strides. Used by the shape operations to build a plan. */
  private strides(): number[] {
    const s: number[] = new Array<number>(this.shape.length).fill(1);
    for (let d = this.shape.length - 2; d >= 0; d--) {
      s[d] = (s[d + 1] ?? 1) * (this.shape[d + 1] ?? 1);
    }
    return s;
  }

  /**
   * Gathers values by a plan into a new tensor. Every shape operation comes here.
   *
   * For now it **actually moves them.** A view would be faster with no copy, and the
   * moment views exist it has to be decided how far an in-place operation spreads —
   * which is not yet the time to decide.
   */
  private stridedView(
    rules: readonly AxisRule[],
    offset: number,
    outShape: readonly number[],
    gradName: string,
  ): Tensor {
    const n = outShape.reduce((a, b) => a * b, 1);
    // **Empty is an answer too.** Slicing outside the range, as in `x[5:99]`, gives 0
    // elements, and the shader divides by that count, so WGSL refuses the whole thing
    // for "division by zero". That invalidates the command buffer with it, so **this
    // place passes and whatever queued behind it is wrong instead** — it surfaced when
    // the next check (`randn`) received all zeros.
    //
    // `indexSelect` already blocks the same branch. Only this one was unblocked, because
    // at the time there was no way to make an empty result by slicing.
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
    // **Complex passes here too.** This operation moves no cells and only changes the
    // label, so it fits the interleaved storage as it is — which is why it comes in
    // through `raw`. Shape operations that **do move** cells (`cat`, `select`,
    // `transpose`…) move them in f32 units and pull real and imaginary out of step, so
    // the `buffer` getter goes on blocking those. The two must not be treated as one
    // group.
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
    return this.stridedView(rules, 0, outShape, "ExpandBackward0");
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
    return this.stridedView(rules, 0, outShape, "RepeatBackward0");
  }

  /**
   * Swaps two axes. The same as `swapdims` — torch carries both names.
   */
  swapaxes(axis0: number, axis1: number): Tensor {
    const rank = this.shape.length;
    const i = axis0 < 0 ? axis0 + rank : axis0;
    const j = axis1 < 0 ? axis1 + rank : axis1;
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
    return this.stridedView(rules, 0, order.map((src) => this.shape[src] ?? 1), "TransposeBackward0");
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
      // Reading past the range does not throw in WGSL; it **gives the edge value or
      // 0.** Rather than producing a quietly wrong value, it stops here.
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
    return this.stridedView(rules, offset, outShape, "SelectBackward0");
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
    // One step moves a row and a column together — so the stride is the sum of the two.
    rules.push({
      size: length, stride: rowStride + colStride, kind: "lin", wrap: length,
    });
    const outShape = [...rest.map((i) => this.shape[i] ?? 1), length];
    return this.stridedView(rules, start, outShape, "DiagonalBackward0");
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
      // **Pad with zeros in front and roll.** Padding `k` in front puts the value at
      // `(i+k, i+k)`; pulling the rows back by `k` gives the upper diagonal `(i, i+k)`
      // and pulling the columns back gives the lower `(i+k, i)`. What the roll wraps
      // around is the padded zeros, so it does no harm — which is why no new kernel is
      // written.
      const k = Math.abs(offset);
      const wide = this.padND([k, 0]).diagflat();
      return wide.roll(-k, offset > 0 ? 0 : 1);
    }
    // **Complex splits into two reals.** Placing on a diagonal touches no value, so
    // applying it to the real and imaginary parts separately leaves it as it was — no
    // new kernel (`sum` passes the same place the same way). `linalg.eig`'s `V·diag(λ)`
    // comes in this way.
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
    return this.stridedView(rules, 0, this.shape, "FlipBackward0");
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
  /**
   * **`dims` chooses the plane and there is only one plane here.** torch turns within
   * whichever two axes it is given, so `dims=[1, 2]` on a rank-3 tensor is a different
   * answer from the default `[0, 1]` — measured, different shapes. This is 2-D only,
   * so the pair can be nothing but `[0, 1]`; carried and refused rather than left out,
   * because a caller writing `dims` is asking for a plane and getting silence.
   */
  rot90(k = 1, dims: readonly number[] = [0, 1]): Tensor {
    if (this.shape.length !== 2) {
      throw new Error(`rot90 is 2-D only for now: [${this.shape}]`);
    }
    if (dims.length !== 2 || dims[0] !== 0 || dims[1] !== 1) {
      throw new Error(
        `rot90(dims=[${dims}]) — this side turns in the [0, 1] plane only`);
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
      return this.stridedView(rules, 0, [rows, cols], "Rot90Backward0");
    }
    if (turns === 1) {
      const rules: AxisRule[] = [
        { size: cols, stride: colStride, kind: "rev", wrap: cols },
        { size: rows, stride: rowStride, kind: "lin", wrap: rows },
      ];
      return this.stridedView(rules, 0, [cols, rows], "Rot90Backward0");
    }
    const rules: AxisRule[] = [
      { size: cols, stride: colStride, kind: "lin", wrap: cols },
      { size: rows, stride: rowStride, kind: "rev", wrap: rows },
    ];
    return this.stridedView(rules, 0, [cols, rows], "Rot90Backward0");
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
  unfold(dimension: number, size: number, step: number): Tensor {
    const rank = this.shape.length;
    const axis = dimension < 0 ? dimension + rank : dimension;
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
    // The inside of the window attaches as the last axis.
    rules.push({ size, stride: axisStride, kind: "lin", wrap: size });
    outShape.push(size);
    return this.stridedView(rules, 0, outShape, "UnfoldBackward0");
  }

  /**
   * Divides one axis into a **given number of** equally sized pieces.
   *
   * **This used to be called `split`, which is torch's name for the other one.**
   * torch's `split(split_size, dim)` says how big each piece is; this says how many
   * pieces there are, which is what torch calls `chunk` — and the arguments were in
   * the opposite order besides. A TypeScript caller writing `x.split(2, 0)` from
   * memory of torch got *axis 2, into 0 pieces*.
   *
   * Nothing had diverged, because the binding routes torch's `split` to `splitSize`
   * with a note saying why, and every golden case goes through the binding. **A name
   * that means something else is invisible to a value comparison** — it is only ever
   * met by somebody writing against this library directly.
   *
   * It is not `chunk`, which rounds up and takes a short last piece; this one
   * requires the axis to divide exactly, and `hsplit`/`vsplit`/`dsplit` want that.
   */
  splitParts(dim: number, parts: number): Tensor[] {
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
      out.push(this.stridedView(rules, k * each * (own[axis] ?? 1), outShape, "SliceBackward0"));
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
  splitWithSizes(splitSizes: readonly number[], dim = 0): Tensor[] {
    const rank = this.shape.length;
    const axis = dim < 0 ? dim + rank : dim;
    const cuts: number[] = [];
    let at = 0;
    for (let i = 0; i < splitSizes.length - 1; i++) {
      at += splitSizes[i]!;
      cuts.push(at);
    }
    return this.splitAt(cuts, axis);
  }

  /** Splits by a **list of cut positions**. Both splitters use this base. */
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

  hsplit(sections: number): Tensor[] {
    return this.splitParts(1, sections);
  }

  vsplit(sections: number): Tensor[] {
    return this.splitParts(0, sections);
  }

  /**
   * Moves an axis to a chosen position. Unlike `swapaxes`, it preserves the
   * order of the rest.
   */
  movedim(source: number, destination: number): Tensor {
    const rank = this.shape.length;
    const from = source < 0 ? source + rank : source;
    const to = destination < 0 ? destination + rank : destination;
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
    return this.stridedView(rules, 0, order.map((s) => this.shape[s] ?? 1), "PermuteBackward0");
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
    return this.stridedView(rules, start * (own[axis] ?? 1), outShape, "SliceBackward0");
  }

  /**
   * Shifts one axis. **What falls off the end comes back at the front.**
   *
   * It is `out[i] = in[(i - shift) mod n]`, so it is the rule table's `mod`
   * with a shift laid on top.
   */
  roll(shifts: number, dims = 0): Tensor {
    const rank = this.shape.length;
    const axis = dims < 0 ? dims + rank : dims;
    const size = this.shape[axis] ?? 1;
    const own = this.strides();
    const bias = ((-shifts % size) + size) % size;
    const rules: AxisRule[] = this.shape.map((s, d) => ({
      size: s,
      stride: own[d] ?? 1,
      kind: d === axis ? ("mod" as const) : ("lin" as const),
      wrap: s,
      ...(d === axis ? { bias } : {}),
    }));
    return this.stridedView(rules, 0, this.shape, "RollBackward0");
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
  /**
   * `torch.Tensor.split`. **The size of each piece**, not how many — the last one is
   * short when the axis does not divide. `splitSize` is the same thing with the two
   * arguments the other way round, and it stays because the binding and this file
   * both call it that.
   */
  split(splitSize: number, dim = 0): Tensor[] {
    return this.splitSize(dim, splitSize);
  }

  /**
   * Divides one axis by size, with the axis first. `split` is the same thing in
   * torch's argument order; `chunk` takes the piece **count** instead.
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

  chunk(chunks: number, dim = 0): Tensor[] {
    const rank = this.shape.length;
    const axis = dim < 0 ? dim + rank : dim;
    const axisSize = this.shape[axis] ?? 0;
    return this.splitSize(axis, Math.ceil(axisSize / chunks));
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
      // It flows only through the kept positions. What was zeroed never entered the
      // result.
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
  gather(dim: number, index: Tensor, sparseGrad = false): Tensor {
    if (sparseGrad) {
      throw new Error("gather(sparseGrad=true) is not here — there is no sparse layout.");
    }
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
    // **The axis numbers shift.** An integer index removes an axis, so every index
    // after it points one place earlier than it did. Hence a separate counter over the
    // surviving axes.
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
   * Stops where only floats are accepted. **The rule is to stop where torch stops** —
   * being the lenient one is also a divergence, and that code breaks on real torch
   * later.
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
    // **A position is not a value** — no gradient flows. torch is the same.
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
    // **Selecting nothing is normal.** When `masked_select` catches nothing the count
    // here is 0, and the shader divides by that count, so WGSL refuses the whole thing
    // for "division by zero". Then **this case passes and the next case is wrong
    // instead**, because the command buffer is invalidated with it. It ends as an empty
    // tensor before that.
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
  repeatInterleave(repeats: number, dim = 0, outputSize: number | null = null): Tensor {
    // torch takes this so the kernel need not read the repeat counts back off the
    // device. It changes no value; the length is already known here.
    void outputSize;
    const rank = this.shape.length;
    const axis = dim < 0 ? dim + rank : dim;
    const own = this.strides();
    const rules: AxisRule[] = this.shape.map((size, d) => ({
      size: d === axis ? size * repeats : size,
      stride: own[d] ?? 1,
      kind: d === axis ? ("div" as const) : ("lin" as const),
      wrap: d === axis ? repeats : size,
    }));
    const outShape = this.shape.map((s, d) => (d === axis ? s * repeats : s));
    return this.stridedView(rules, 0, outShape, "RepeatInterleaveBackward0");
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
  matrixPower(n: number): Tensor {
    if (n < 1) throw new Error(`matrix_power supports 1 and up for now: ${n}`);
    // **Multiplications are chained** — then the backward follows by itself. Writing
    // it as a decomposition means writing the derivative afresh, which is one more place
    // to be wrong.
    //
    // The batch is folded to 3-D and unfolded again. `mm` is 2-D only and `bmm` is 3-D
    // only, so something like `(2,3,4,4)` fits neither — folded, neither is needed.
    const rank = this.shape.length;
    if (rank <= 2) {
      let out: Tensor = this;
      for (let i = 1; i < n; i++) out = out.mm(this);
      return out;
    }
    // The side length, called `side` because the exponent took torch's name `n` and
    // the two used to be `k` and `n`.
    const side = this.shape[rank - 1] ?? 0;
    const batch = this.shape.slice(0, rank - 2).reduce((a, b) => a * b, 1);
    const flat = this.reshape([batch, side, side]);
    let out: Tensor = flat;
    for (let i = 1; i < n; i++) out = out.bmm(flat);
    return out.reshape(this.shape);
  }

  /**
   * This side or that, per position of the condition. torch's method form
   * is `x.where(cond, other)`.
   */
  where(condition: Tensor, other: Tensor): Tensor {
    // **The three broadcast against each other**, as torch's `where` does and as
    // every binary operation on this class already does. Without it the kernel runs
    // over *this* tensor's element count and reads the other two buffers at the same
    // positions, so a mask that is right but shorter — `[1,1,H,W,K,K]` against
    // `[N,C,H,W,K,K]`, which is what a per-position mask over channels looks like —
    // selects by whatever happened to sit at that offset. The shape came out right
    // and the values did not, which is the quietest way for this to be wrong:
    // twenty-one golden cases under `roi_align` and the samplers beside it.
    const wide = broadcastShapes(broadcastShapes(this.shape, other.shape),
                                 condition.shape);
    const fits = (s: readonly number[]): boolean =>
      s.length === wide.length && s.every((d, i) => d === wide[i]);
    if (!fits(this.shape) || !fits(other.shape) || !fits(condition.shape)) {
      return this.broadcastTo(wide)
        .where(condition.broadcastTo(wide), other.broadcastTo(wide));
    }
    const n = this.size;
    const out = dev().alloc(n);
    dev().run1d(
      dev().pipeline(`wh:${n}`, () => whereKernel(n)),
      [condition.buffer, this.buffer, other.buffer, out],
      n,
    );
    const shape = this.shape;
    const side = (g: Tensor, take: "a" | "b"): Tensor => {
      const gi = dev().alloc(n);
      dev().run1d(
        dev().pipeline(`whb:${take}:${n}`, () => whereBackward(n, take)),
        [condition.buffer, g.buffer, gi],
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
  norm(p: number = 2, dim?: number, keepdim = false, dtype?: DType): Tensor {
    if (dtype !== undefined) {
      // As `mean` above: `dtype=` releases the refusal on the **input** side, and a
      // result that has to land on an integer still has no answer. torch stops in the
      // same place — measured, `softmax(dtype=int64)` is a `NotImplementedError` and
      // `nanmean(dtype=int64)` says it could not infer the output dtype.
      if (dtype !== "float32" && dtype !== "complex64") {
        throw new RuntimeError(
          "norm(): could not infer output dtype. Input dtype must be either " +
            "a floating point or complex dtype");
      }
      return this.castFirst(dtype).norm(p, dim, keepdim).to(dtype);
    }
    // **It stops where torch stops** (measured). A division or a square root has no
    // answer that fits an integer cell — promoted quietly to float the way numpy does,
    // that code then breaks on real torch.
    this.needsFloat("norm is for floating point only", "linalg.vector_norm: Expected a floating point or complex tensor as input");
    // torch's order is `(p, dim, keepdim, dtype)` and this took nothing at all, so
    // every `x.norm(2, 1)` a reader transcribed was silently the norm over the whole
    // tensor. `p` reaches the same branches the core has: 1, 2, ±inf, 0, and the
    // general case.
    const fold = (t: Tensor): Tensor =>
      dim === undefined ? t.sum() : t.sumDim(dim, keepdim);
    if (p === 1) return fold(this.abs());
    if (p === 2) return fold(this.square()).sqrt();
    // `amax`/`amin` rather than `max`/`min`: those return `{values, indices}` and
    // torch's norm wants the value alone. With no `dim` they fold everything, which
    // is what `amax()` does with its default.
    if (p === Infinity) return this.abs().amax(dim, keepdim);
    if (p === -Infinity) return this.abs().amin(dim, keepdim);
    // **`p = 0` counts the non-zeros**, and the `* 0` term is what carries the graph
    // through: counting is a step, so the derivative is zero rather than absent, and
    // without it `norm(0).backward()` stops where torch keeps going.
    if (p === 0) {
      const nonzero = this.ne(Tensor.full([], 0)).float();
      return fold(nonzero).add(fold(this.mul(Tensor.full([], 0))));
    }
    return fold(this.abs().powScalar(p)).powScalar(1 / p);
  }

  /**
   * The inner product of two vectors.
   */
  dot(tensor: Tensor): Tensor {
    return this.mul(tensor).sum();
  }

  /**
   * The outer product of two vectors. It falls out of broadcasting — no new
   * kernel needed.
   */
  outer(vec2: Tensor): Tensor {
    return this.reshape([this.size, 1]).mul(vec2.reshape([1, vec2.size]));
  }

  /**
   * Cuts above and below.
   *
   * **It must not be laid on top of `maximum` and `minimum`.** Those two
   * split the gradient in half at a tie (torch does), while `clamp` passes
   * it whole at the boundary. Laid on top, the gradient halved exactly
   * where `x` sat on the boundary.
   */
  clamp(min: number, max: number): Tensor {
    const lo = f32lit(min), hi = f32lit(max);
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
    // For a non-integer exponent a negative base genuinely has no answer. It is left
    // to the kernel.
    return this.binary("pow", Tensor.full([], k));
  }

  /**
   * `exp(x) / Σ exp(x)`. **Computed with the maximum subtracted** —
   * otherwise it overflows at large values.
   */
  softmax(dim = 0, dtype?: DType): Tensor {
    if (dtype !== undefined) {
      // As `mean` above: `dtype=` releases the refusal on the **input** side, and a
      // result that has to land on an integer still has no answer. torch stops in the
      // same place — measured, `softmax(dtype=int64)` is a `NotImplementedError` and
      // `nanmean(dtype=int64)` says it could not infer the output dtype.
      if (dtype !== "float32" && dtype !== "complex64") {
        throw new RuntimeError(
          "softmax(): could not infer output dtype. Input dtype must be either " +
            "a floating point or complex dtype");
      }
      return this.castFirst(dtype).softmax(dim).to(dtype);
    }
    const m = this.amax(dim, true).detach();
    const e = this.sub(m).exp();
    return e.div(e.sumDim(dim, true));
  }

  /**
   * `log(softmax(x))`. **It does not compute `softmax` and take the log** —
   * small probabilities become 0 and the log becomes -inf. Written directly
   * as a subtraction, that place does not exist.
   */
  logSoftmax(dim = 0, dtype?: DType): Tensor {
    if (dtype !== undefined) {
      // As `mean` above: `dtype=` releases the refusal on the **input** side, and a
      // result that has to land on an integer still has no answer. torch stops in the
      // same place — measured, `softmax(dtype=int64)` is a `NotImplementedError` and
      // `nanmean(dtype=int64)` says it could not infer the output dtype.
      if (dtype !== "float32" && dtype !== "complex64") {
        throw new RuntimeError(
          "log_softmax(): could not infer output dtype. Input dtype must be either " +
            "a floating point or complex dtype");
      }
      return this.castFirst(dtype).logSoftmax(dim).to(dtype);
    }
    return this.sub(this.logsumexp(dim, true));
  }

  /**
   * Restores a folded axis at size 1. Everything taking `keepdim` calls this last.
   *
   * **A shape with an axis removed often still fits broadcasting.** So failing to take
   * `keepdim` produces, instead of a loud stop, a run that goes all the way through with
   * only the values wrong — `x.gather(1, x.argmax(1, true))` is that shape.
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
    // It gathers again by index — that is what sends the gradient to **the one**
    // winning position. `gather` needs equal rank, so the folded axis is restored and
    // folded again.
    const lifted = [...this.shape];
    lifted[axis] = 1;
    const values = this.gather(axis, indices.reshape(lifted))
      .reshape(indices.shape);
    // **The indices have to keep the axis too.** Restoring only the values makes
    // `x.gather(1, m.indices)` stop on a rank mismatch — or worse, pass by
    // broadcasting.
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
    // A position is not a value — there is nowhere for a gradient to flow. torch does
    // not flow one either.
    // **The dtype is always int64.** This counts indices rather than selecting values,
    // so it has nothing to do with the original dtype.
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

  /** The body of `all` and `any`. Converted to non-zero, then folded by selection. */
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
    // Inside the range the gradient passes through, outside it is 0. Chosen with
    // `where`.
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
      // The padded positions did not come from the input — only the middle is
      // returned.
      (g) => [g.narrow(axis, before, size)],
      "ConstantPadNdBackward0",
      this.dtype,
    );
  }

  // ── Windows ───────────────────────────────────────────────────────────
  //
  // **`unfold` and `fold` are not each other's inverse.** Folding back **adds** the
  // overlapping positions — unfold a 4×4 with a 2×2 window and fold it straight back,
  // and the middle is counted four times.
  //
  // Both are built from one index list. With where-it-came-from written down, unfolding
  // is a gather and folding back is an add into those positions, so one's backward *is*
  // the other.

  /**
   * Spreads windows into columns. `(N, C, H, W)` → `(N, C·kh·kw, L)`.
   *
   * **A 3-D input is one unbatched sample**, which is torch's rule and was missing on
   * both sides — the core refused *anything but 4-D*, half right. `(2, 3, 4)` with a
   * 2×2 kernel comes back as `(8, 6)`: `(C·kh·kw, L)` with no batch axis. Anything
   * else is refused with torch's own wording.
   */
  unfoldIm2col(kernel: number | [number, number], dilation = 1, padding = 0,
               stride = 1): Tensor {
    if (this.shape.length === 3) {
      const got = this.reshape([1, ...this.shape])
        .unfoldIm2col(kernel, dilation, padding, stride);
      return got.reshape(got.shape.slice(1));
    }
    if (this.shape.length !== 4) {
      throw new RuntimeError(
        "Expected 3D or 4D (batch mode) tensor with possibly 0 batch size and other "
        + `non-zero dimensions for input, but got: [${this.shape.join(", ")}]`);
    }
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
   *
   * **The unbatched form is 2-D here, one rank below `unfoldIm2col`'s**, because this
   * side has already folded the channel and the kernel into one axis. `(8, 6)` comes
   * back as `(2, 3, 4)`; 4-D is refused, with torch's own wording.
   */
  fold(outputSize: number | [number, number], kernel: number | [number, number],
       dilation = 1, padding = 0, stride = 1): Tensor {
    if (this.shape.length === 2) {
      const got = this.reshape([1, ...this.shape])
        .fold(outputSize, kernel, dilation, padding, stride);
      return got.reshape(got.shape.slice(1));
    }
    if (this.shape.length !== 3) {
      throw new RuntimeError(
        "Expected 2D or 3D (batch mode) tensor for input with possibly 0 batch size "
        + `and non-zero dimensions for input, but got: [${this.shape.join(", ")}]`);
    }
    const n = this.shape[0] ?? 1;
    const [kh, kw] = pairOf(kernel);
    const [oh, ow] = pairOf(outputSize);
    const [ph, pw] = pairOf(padding);
    const c = (this.shape[1] ?? 1) / (kh * kw);
    const { idx } = windowIndex(
      [c, oh, ow], [kh, kw], pairOf(dilation), [ph, pw], pairOf(stride));
    const hp = oh + 2 * ph;
    const wp = ow + 2 * pw;
    // Every batch uses the same position table — `scatterAdd` wants the index to have
    // the same shape as the source.
    const wide = new Float32Array(n * idx.length);
    for (let b = 0; b < n; b++) wide.set(idx, b * idx.length);
    const flat = Tensor.zeros([n, c * hp * wp]).scatterAdd(
      1, Tensor.from(wide, [n, idx.length]), this.reshape([n, idx.length]));
    const made = flat.reshape([n, c, hp, wp]);
    if (!ph && !pw) return made;
    return made.narrow(2, ph, oh).narrow(3, pw, ow);
  }

  // ── What the remaining layers use ─────────────────────────────────────

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
    // Padding the channel axis with zeros trims the edges by itself. `padND` counts
    // from the last axis, so at 4-D the channel is the third pair.
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
    // Spreading indices into positions is already what `scatter` does — it places 1s
    // onto a plate of zeros.
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
  interpolateBilinear(outH: number, outW: number, alignCorners: boolean,
                      given: number | null = null): Tensor {
    const h = this.shape[2] ?? 1;
    const w = this.shape[3] ?? 1;
    const ys = bilinearAxis(h, outH, alignCorners, given);
    const xs = bilinearAxis(w, outW, alignCorners, given);
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

  // ── Moving elements ───────────────────────────────────────────────────
  //
  // All three **change positions and not values.** The forward is `reshape` plus an axis
  // swap and the backward is the reverse, so it is a combination of what already exists.

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
    // `transpose` is 2-D only, so it goes through `permute` — however many spatial
    // axes follow, only the first two slots swap.
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
    // `dropout` scales the surviving positions by `1/(1-p)`. What is needed here is
    // 0/1, so it is multiplied back — rather than keeping one more kernel that produces
    // the mask on its own.
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
      // **The kind of refusal is part of the answer.** torch raises
      // `NotImplementedError` here, and that says something different from "the caller
      // was wrong".
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
      // **Only `reflect` cares about the size.** To mirror, there has to be something
      // to mirror. `replicate` may extend by five, because there is always a value to
      // extend with.
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
    // `gt`'s gradient is 0 — a mask must not carry one.
    // **Moved to float.** A comparison produces bool, and subtraction on bool is refused
    // (torch too). The mask is about to be used as a number, so the dtype is matched
    // here.
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
      // With the seed in the name, a new shader is baked every step. Only `p` is baked
      // as a constant and the seed stays out of the name — one pipeline is reused
      // instead.
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
  l1Loss(target: Tensor, reduction: Reduction = "mean",
         weight?: Tensor): Tensor {
    return this.sub(target).abs().weighTo(reduction, weight, true);
  }

  /**
   * Squared when small, absolute when large. **The derivative is continuous at the
   * origin** — which is the reason to use this loss, so the two expressions are joined
   * at `beta`.
   */
  /**
   * Squared error.
   */
  mseLoss(target: Tensor, reduction: Reduction = "mean",
          weight?: Tensor): Tensor {
    return this.sub(target).square().weighTo(reduction, weight, true);
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
   * Binary cross-entropy over **probabilities**, not logits. `torch.nn.BCELoss`.
   *
   * **torch clamps the log at −100, and the obvious guard is a different one.**
   * The core wrote `-(p + 1e-12).log()`, which cannot exceed 27.63 whatever `p`
   * is — so every confident-and-wrong prediction reported the same loss as every
   * other one, and at `p = 1e-8` the epsilon moved the answer in the fourth
   * decimal. Clamping the log's *output* caps the loss at 100 and leaves 1e-20
   * telling the truth (46.05).
   *
   * It is `CrossEntropyLoss`'s defect a second time. Both had the guard and the
   * defect on the same line.
   *
   * Prefer `bceWithLogits` where you have logits: it is the numerically stable
   * path and this one is reached only when the probabilities are what you hold.
   */
  bce(target: Tensor, reduction: Reduction = "mean"): Tensor {
    const floor = Tensor.full([], -100);
    const one = Tensor.full([], 1);
    const lo = this.unary("log").binary("maximum", floor);
    const hi = one.sub(this).unary("log").binary("maximum", floor);
    return target.mul(lo).add(one.sub(target).mul(hi)).neg().reduceAs(reduction);
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
   *
   * **Any number of leading dimensions**, as torch has. It used to be `mm`,
   * which meant 2-D only, and every model carrying `[batch, tokens, dim]`
   * had to fold the token axis into the batch itself before calling a
   * `Linear` — bimm's ViT shipped a `tokenwise()` helper to do exactly that.
   * `matmul` broadcasts, so the fold is no longer the caller's problem.
   */
  linear(weight: Tensor): Tensor {
    return this.matmul(weight.transpose());
  }

  smoothL1Loss(target: Tensor, beta = 1.0,
               reduction: Reduction = "mean"): Tensor {
    const d = this.sub(target);
    const near = d.square().binary("mul", Tensor.full([], 0.5 / beta));
    const far = d.abs().binary("sub", Tensor.full([], 0.5 * beta));
    const isNear = d.abs().binary("lt", Tensor.full([], beta));
    return near.where(isNear, far).reduceAs(reduction);
  }

  // ── Losses and distances ──────────────────────────────────────────────
  //
  // **How it folds is part of the loss.** Every torch loss takes `reduction`, and on
  // that value it becomes elementwise, mean or sum. Gathered in one place, thirteen of
  // them use one rule.

  private reduceAs(reduction: Reduction): Tensor {
    if (reduction === "none") return this;
    if (reduction === "sum") return this.sum();
    // **An unknown name stops.** Falling through an `else` into mean leaves whoever
    // wrote `"MEAN"` believing their choice was used — a value comes out, it equals the
    // default, and it is indistinguishable from never having passed the argument. torch
    // stops here too.
    if (reduction !== "mean") {
      throw new RuntimeError(
        `${reduction} is not a valid value for reduction ` +
          "('none' | 'mean' | 'sum')");
    }
    return this.mean();
  }

  /**
   * `reduceAs` with torch's per-element `weight` on the three losses that take one.
   *
   * Measured on three weight vectors:
   *
   *     none  w · ℓ                      all three
   *     sum   Σ w · ℓ                    all three
   *     mean  Σ w·ℓ / Σ w                `l1Loss` and `mseLoss`
   *     mean  Σ w·ℓ / n                  `huberLoss`
   *
   * **`huberLoss` divides by the count and the other two do not**, which is why the
   * divisor is a parameter rather than a rule — assuming the family agreed would
   * make huber's `mean` wrong by `Σw / n` with no exception anywhere.
   *
   * The shapes must match exactly; torch does not broadcast here, and the wording of
   * the refusal is torch's.
   */
  private weighTo(reduction: Reduction, weight: Tensor | undefined,
                  meanOverWeights: boolean): Tensor {
    if (weight === undefined) return this.reduceAs(reduction);
    if (weight.shape.join() !== this.shape.join()) {
      throw new RuntimeError("Weights and input must have the same size.");
    }
    const scaled = this.mul(weight);
    if (reduction !== "mean" || !meanOverWeights) return scaled.reduceAs(reduction);
    return scaled.sum().div(weight.sum());
  }

  /**
   * **It equals `smoothL1Loss` only at δ=1.**
   *
   * The real relation is `huber(δ) = δ · smoothL1(β=δ)`. Measured at the
   * defaults alone, treating them as one function passes, so the golden
   * cases ask with δ varied.
   *
   * **`weight`'s `mean` divides by the count here** and by the sum of the weights on
   * `l1Loss` and `mseLoss`. Measured, not inferred from the family.
   */
  huberLoss(target: Tensor, delta = 1.0, reduction: Reduction = "mean",
            weight?: Tensor): Tensor {
    const d = this.sub(target);
    const near = d.square().binary("mul", Tensor.full([], 0.5));
    const far = d.abs().binary("sub", Tensor.full([], 0.5 * delta))
      .binary("mul", Tensor.full([], delta));
    const isNear = d.abs().binary("lt", Tensor.full([], delta));
    return near.where(isNear, far).weighTo(reduction, weight, false);
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
   * `torch.nn.functional.triplet_margin_with_distance_loss`. **The distance is the
   * caller's**, which is the whole difference from `tripletMarginLoss` below — that
   * one is this with the pairwise distance, and torch says so by giving them the
   * same default.
   *
   * It is a method rather than a function in `functional.ts` because the reduction
   * is private to this class, and that file's rule is that every value comes from a
   * `Tensor` method.
   */
  tripletMarginWithDistanceLoss(
    positive: Tensor, negative: Tensor,
    distanceFunction: ((u: Tensor, v: Tensor) => Tensor) | null = null,
    margin = 1.0, swap = false, reduction: Reduction = "mean",
  ): Tensor {
    const dist = distanceFunction ?? ((u: Tensor, v: Tensor) => u.pairwiseDistance(v));
    const dp = dist(this, positive);
    let dn = dist(this, negative);
    // A negative closer to the positive than to the anchor is the harder pair, and
    // without this the loss never sees it.
    if (swap) dn = dn.binary("minimum", dist(positive, negative));
    return dp.sub(dn).add(Tensor.full([], margin)).unary("relu").reduceAs(reduction);
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
    // A negative closer than the positive is the harder pair.
    if (swap) dn = dn.binary("minimum", positive.pairwiseDistance(negative, p, eps));
    return dp.sub(dn).binary("add", Tensor.full([], margin)).unary("relu")
      .reduceAs(reduction);
  }

  /**
   * Independent binary classification per slot, **averaged over the whole
   * class set.**
   */
  multilabelSoftMarginLoss(
    target: Tensor, weight?: Tensor, reduction: Reduction = "mean",
  ): Tensor {
    // `weight` rescales each class before the mean, which is where torch applies
    // it. Multiplying before or after the negation gives the same number; the
    // order here follows the core so the two read alike side by side.
    let each = target.mul(this.logsigmoid())
      .add(Tensor.full([], 1).sub(target).mul(this.neg().logsigmoid()));
    if (weight !== undefined) each = each.mul(weight);
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
    // At the true position `margin` survives untouched, so it is subtracted.
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
    // `diff[r,i,j] = 1 − (x[r,i] − x[r,j])`, counting only true i against false j.
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
  bmm(mat2: Tensor): Tensor {
    if (this.shape.length !== 3 || mat2.shape.length !== 3) {
      throw new Error(`bmm is 3-D by 3-D: [${this.shape}] x [${mat2.shape}]`);
    }
    const batch = this.shape[0] ?? 0;
    const parts: Tensor[] = [];
    for (let b = 0; b < batch; b++) {
      parts.push(this.select(0, b).mm(mat2.select(0, b)));
    }
    return Tensor.stack(parts, 0);
  }

  // ── The addmm family ──────────────────────────────────────────────────
  //
  // All eight have the shape `β·this + α·(some product)`. The only difference is **what
  // the product is**, so that one thing is passed in.

  /**
   * `β·this + α·product`.
   *
   * **`β == 0` ignores the value and stays in the graph.** It has to be both —
   *
   * - written as `this.mul(0)`, a NaN input makes the result NaN. torch is fine.
   * - taken out of the graph instead, the gradient is not 0 but **absent.** torch gives
   *   0 (measured).
   *
   * The two requirements pull opposite ways, and with ordinary inputs **neither** shows
   * — the first needs a NaN and the second needs somebody to ask for a gradient.
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
  fmod(other: number): Tensor {
    const d = Tensor.full([], other);
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
  remainder(other: Tensor | number): Tensor {
    // **Tensors are accepted too.** torch writes `x.remainder(y)`, and accepting only
    // a number makes that line simply not run — a name that exists but is narrower is
    // harder to find than one that is missing (`lerpFrom` was the same place).
    const d = Tensor.asTensor(other);
    const q = this.div(d).unary("floor").detach();
    return this.sub(q.binary("mul", d));
  }

  /**
   * Cuts from below only. torch's `clamp(min=…)`.
   */
  clampMin(min: number): Tensor {
    const lo = f32lit(min);
    return this.unary(unaryWith(`clampMin<${lo}>`, () => ({
      fwd: `max(x, ${lo})`,
      bwd: `select(0.0, 1.0, x >= ${lo})`,
    })));
  }

  /**
   * Cuts from above only. torch's `clamp(max=…)`.
   */
  clampMax(max: number): Tensor {
    const hi = f32lit(max);
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
    const stacked = Tensor.stack(padded, 0); // (count, length, …)
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
  nllLoss(
    target: Tensor, ignoreIndex = -100, reduction: Reduction = "mean",
  ): Tensor {
    // **The per-sample values are made before folding.** Averaging as soon as they are
    // drawn leaves nowhere to build `reduction: "none"` — per-sample values cannot be
    // recovered from a scalar.
    //
    // `ignoreIndex` is not an index, so the gather has to see a real one: the
    // ignored rows are gathered from row 0 and then zeroed, and their value never
    // reaches the answer. Same ordering the core needs, for the same reason.
    const keep = target.ne(Tensor.full([], ignoreIndex));
    const safe = target.mul(keep);
    const each = this.gather(1, safe.reshape([target.size, 1]))
      .reshape([target.size]).neg().mul(keep.reshape([target.size]));
    return each.reduceIgnoring(keep, reduction);
  }

  /**
   * Straight from logits. `log_softmax` and `nll_loss` joined.
   */
  crossEntropy(
    target: Tensor, ignoreIndex = -100, reduction: Reduction = "mean",
    labelSmoothing = 0.0,
  ): Tensor {
    const logp = this.logSoftmax(-1);
    if (!labelSmoothing) return logp.nllLoss(target, ignoreIndex, reduction);
    // torch spreads ε over every class: the target term keeps 1-ε and the rest
    // share ε/C, which is the mean of every class's log-probability.
    const keep = target.ne(Tensor.full([], ignoreIndex));
    const safe = target.mul(keep);
    const picked = logp.gather(1, safe.reshape([target.size, 1]))
      .reshape([target.size]).neg();
    const spread = logp.neg().mean(logp.shape.length - 1);
    const each = picked.mul(Tensor.full([], 1 - labelSmoothing))
      .add(spread.mul(Tensor.full([], labelSmoothing)))
      .mul(keep.reshape([target.size]));
    return each.reduceIgnoring(keep, reduction);
  }

  /**
   * Fold, with the ignored rows already zeroed and `keep` marking which stayed.
   *
   * **The three reductions treat a skipped row differently and two of the three are
   * easy to get right by accident.** `sum` is unaffected either way. `mean` has to
   * divide by the rows that remain, not by all of them — averaging with the zeros in
   * gives a number too small by exactly that ratio and nothing about it looks wrong.
   * `none` **keeps them as zeros**, because the shape is part of that answer.
   */
  reduceIgnoring(keep: Tensor, reduction: Reduction): Tensor {
    if (reduction === "none") return this;
    const total = this.sum();
    if (reduction === "sum") return total;
    return total.div(keep.reshape([keep.size]).sum());
  }

  // ── Where the output size depends on the values ───────────────────────
  //
  // This family **only knows how many come out by looking at the values.** The GPU needs
  // the buffer size in advance, so the values have to be read back once, and that makes
  // all of them asynchronous. The sister library goes to the CPU and back here for the
  // same reason.

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
   *
   * `asTuple` gives **one 1-D tensor per axis** instead of one (count, rank)
   * table — the form indexing takes.
   *
   * `torch.nonzero` is read by `inspect` as "no signature found for builtin",
   * so the signature axis had never compared it. That bucket is not agreement;
   * it is *nothing was asked*.
   */
  async nonzero(asTuple?: false): Promise<Tensor>;
  async nonzero(asTuple: true): Promise<Tensor[]>;
  async nonzero(asTuple = false): Promise<Tensor | Tensor[]> {
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
    if (asTuple) {
      const cols: Tensor[] = [];
      for (let d = 0; d < rank; d++) {
        const one: number[] = [];
        for (let r = 0; r < count; r++) one.push(rows[r * rank + d] ?? 0);
        cols.push(Tensor.from(one, [count], { dtype: "int64" }));
      }
      return cols;
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
      // **It hands over no index, so it divides evenly.** NaN cells drop out by
      // themselves, since `eq` is false for them.
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
   * Along one axis, takes a value and a position per row.
   *
   * `mode` and `nanmedian` share this skeleton — the only difference is **what is chosen
   * within one row**, so that one thing is passed in.
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
    // **The values are gathered rather than built by hand.** `Tensor.from(vals, …)`
    // has the right values and **no graph** — every value check passes and it surfaces
    // only at `backward()`, where the message says "does not require grad" and so
    // **points at the user.**
    //
    // This operation hands over indices, so the rule belongs on this side too: the
    // gradient goes to **the one chosen position** (measured). Gathered, that rule falls
    // out of `gather`'s backward by itself.
    const lifted = this.shape.map((s, d) => (d === axis ? 1 : s));
    const at = Tensor.from(idx, lifted, { dtype: "int64" });
    return {
      values: this.gather(axis, at).reshape(shape),
      indices: at.reshape(shape),
    };
  }

  /**
   * **Divides the gradient evenly** across cells holding the same value.
   *
   * This is the rule for reductions that hand over no index (`median()`, `max()`,
   * `nanmedian()`) — measured: `[3,5,5,1,5]`'s `median()` gradient is ⅓ on each of the
   * three 5s. Sending it all to one position leaves the value identical and only the
   * gradient different, and **measured on data with no ties every rule gives the same
   * answer**, so the table asks nothing.
   *
   * It multiplies by the mask, sums, and divides by the count — the value is unchanged
   * (equal cells summed and divided by their count) and the backward is `mask/count`, so
   * the rule falls out by itself. `mask` is a comparison and carries no gradient, and
   * the count is detached as well.
   */
  private spreadEqual(value: Tensor): Tensor {
    const hit = this.binary("eq", value, "bool").detach();
    const count = hit.to("float32").sum().detach();
    // **It must not multiply — `0 × NaN` is NaN.** A version that multiplied by the
    // mask and summed came first, and `nanmedian` went wholly NaN on data containing a
    // NaN. This is the third time this repository has been bitten at the same place (the
    // core's `median`, borch.ts's `median`, and here), so it selects instead — `where`
    // **keeps the unchosen cells out of the computation.**
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
          // A one-sided difference fitted quadratically. Exact on `x²`.
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
  // **Overloaded so the plain call keeps its narrow type.** Widening the return to
  // `Tensor | Tensor[]` unconditionally made every existing `await x.unique()` stop
  // type-checking — `tsc` named both call sites at once — and the fix is to say what
  // torch's own shape is: one tensor when nothing extra is asked for, a tuple when it
  // is. A caller that asks for neither should not have to narrow.
  async unique(sorted?: boolean): Promise<Tensor>;
  async unique(
    sorted: boolean, returnInverse: boolean, returnCounts?: boolean,
    dim?: number | null,
  ): Promise<Tensor[]>;
  async unique(
    sorted = true, returnInverse = false, returnCounts = false,
    dim: number | null = null,
  ): Promise<Tensor | Tensor[]> {
    if (dim !== null) {
      return this.uniqueAlong(dim, returnInverse, returnCounts);
    }
    // torch's order — `returnInverse` **second**. Two arguments were missing from
    // the middle here, so `x.unique(true, true)` asked for the inverse in torch and
    // for nothing at all over here.
    //
    // `sorted` is accepted and changes nothing, which is what it does in torch on
    // the CPU as well: torch documents it as *may* return a different order and
    // sorts regardless. Said out loud, because an argument that is accepted and
    // silent otherwise reads as one that works.
    void sorted;
    const values = Array.from(await this.toArray());
    const seen = [...new Set(values)].sort((a, b) => a - b);
    const at = new Map(seen.map((v, i) => [v, i]));
    const out: Tensor[] = [
      Tensor.from(seen, [seen.length], { dtype: this.dtype }),
    ];
    if (returnInverse) {
      // **Flat, over the flattened input** — torch flattens before it looks when no
      // dimension is given, so the inverse is 1-D whatever the input's rank.
      const inverse = values.map((v) => at.get(v) ?? 0);
      out.push(Tensor.from(inverse, [inverse.length], { dtype: "int64" }));
    }
    if (returnCounts) {
      const counts = seen.map((v) => values.filter((w) => w === v).length);
      out.push(Tensor.from(counts, [counts.length], { dtype: "int64" }));
    }
    return out.length === 1 ? (out[0] as Tensor) : out;
  }

  /**
   * `unique(dim=…)` — **a different operation from the one beside it.**
   *
   * Without a dimension, `unique` folds *values*: a 3×2 of `[[1,2],[3,4],[1,2]]`
   * has four distinct numbers and comes back as a flat `[1, 2, 3, 4]`. With
   * `dim=0` it folds *whole slices*: two distinct rows, and the answer keeps its
   * rank at 2×2. The inverse changes meaning with it — one entry per slice along
   * `dim` rather than one per element.
   *
   * **This was written down as not carried across, and the reason outlived it.**
   * The gap table said the two forms were different operations, which is true and
   * was doing the work of an excuse: they are different operations and both are
   * torch's. Two golden rows sat red under that sentence.
   *
   * Slices are ordered lexicographically, which is what numpy's `unique(axis=)`
   * does and therefore what the core does. Equality is by value across the whole
   * slice — two rows are one row only if every element matches.
   */
  private async uniqueAlong(
    dim: number, returnInverse: boolean, returnCounts: boolean,
  ): Promise<Tensor | Tensor[]> {
    const rank = this.shape.length;
    const axis = dim < 0 ? dim + rank : dim;
    if (axis < 0 || axis >= rank) {
      throw new IndexError(
        `Dimension out of range (expected to be in range of [${-rank}, ${rank - 1}], but got ${dim})`,
      );
    }
    const flat = Array.from(await this.toArray());
    const along = this.shape[axis] ?? 0;
    // How far apart two neighbouring positions along `axis` sit in the flat array.
    // Everything below groups by `Math.floor(j / step) % along`, which is that
    // position's coordinate on the axis — and **ascending `j` within a group is
    // row-major order over the remaining axes**, so a slice needs no reshaping.
    let step = 1;
    for (let k = axis + 1; k < rank; k += 1) step *= this.shape[k] ?? 1;

    const slices: number[][] = Array.from({ length: along }, () => []);
    for (let j = 0; j < flat.length; j += 1) {
      const at = Math.floor(j / step) % along;
      (slices[at] as number[]).push(flat[j] as number);
    }

    const order = [...slices.keys()].sort((a, b) => {
      const x = slices[a] as number[];
      const y = slices[b] as number[];
      for (let k = 0; k < x.length; k += 1) {
        if (x[k] !== y[k]) return (x[k] as number) - (y[k] as number);
      }
      return 0;
    });

    const kept: number[][] = [];
    const counts: number[] = [];
    const inverse = new Array<number>(along).fill(0);
    for (const at of order) {
      const slice = slices[at] as number[];
      const last = kept[kept.length - 1];
      const same = last !== undefined && last.length === slice.length
        && last.every((v, k) => v === slice[k]);
      if (same) {
        counts[counts.length - 1] = (counts[counts.length - 1] as number) + 1;
      } else {
        kept.push(slice);
        counts.push(1);
      }
      inverse[at] = kept.length - 1;
    }

    const shape = [...this.shape];
    shape[axis] = kept.length;
    const values = new Array<number>(kept.length * (slices[0]?.length ?? 0));
    const width = kept.length;
    for (let j = 0; j < values.length; j += 1) {
      const at = Math.floor(j / step) % width;
      // The position inside the slice is the flat index with the axis removed, and
      // it counts up in exactly the order the slice was gathered in.
      const within = Math.floor(j / (step * width)) * step + (j % step);
      values[j] = (kept[at] as number[])[within] as number;
    }

    const out: Tensor[] = [Tensor.from(values, shape, { dtype: this.dtype })];
    if (returnInverse) {
      out.push(Tensor.from(inverse, [along], { dtype: "int64" }));
    }
    if (returnCounts) {
      out.push(Tensor.from(counts, [counts.length], { dtype: "int64" }));
    }
    return out.length === 1 ? (out[0] as Tensor) : out;
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
    // The axis is laid to the front and seen row by row — with no axis, one cell is
    // one row.
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

  /** How many times each integer value occurred. The largest value sets the length. */
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
  async quantile(q: number | readonly number[], dim: number | null = null,
                 keepdim = false, interpolation = "linear"): Promise<Tensor> {
    quantileSeats(dim, keepdim);
    return this.quantileOver(Array.from(await this.toArray()), q, interpolation);
  }

  /**
   * The body of the quantile — **gathered back through the sorted positions.**
   *
   * Values built by hand have no graph. Gathered back, the gradient goes to **the two
   * positions the interpolation used**, which is this operation's rule (measured). This
   * is where it parts from `median` on ties — for `[1,5,5,5]`, `median` gives ⅓ to each
   * of the three 5s and `quantile(0.5)` gives **½ to each of the first two.** Both
   * values are 5, so only gathering back tells them apart.
   */
  private quantileOver(host: number[], q: number | readonly number[],
                       interpolation = "linear"): Tensor {
    const order = host.map((_, i) => i).sort((a, b) => (host[a]! - host[b]!));
    const flat = this.flat();
    const wanted = typeof q === "number" ? [q] : [...q];
    const parts = wanted.map((p) => {
      const at = p * (order.length - 1);
      const lo = Math.floor(at);
      const hi = Math.min(lo + 1, order.length - 1);
      // **The rule that produced the value has to be the rule the gradient
      // splits by.** Written for `linear` alone, the two describe different
      // functions with only the backward wrong, and nothing shows it.
      const w = interpolationWeight(interpolation, at, lo);
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
  async nanquantile(q: number | readonly number[], dim: number | null = null,
                    keepdim = false, interpolation = "linear"): Promise<Tensor> {
    quantileSeats(dim, keepdim);
    const values = Array.from(await this.toArray());
    const keep = values.map((v, i) => [v, i] as const)
      .filter(([v]) => !Number.isNaN(v));
    const at = Tensor.from(keep.map(([, i]) => i), [keep.length],
      { dtype: "int64" });
    const clean = this.flat().indexSelect(0, at);
    return clean.quantileOver(keep.map(([v]) => v), q, interpolation);
  }

  // ── Linear algebra ────────────────────────────────────────────────────

  /**
   * Reads back to the CPU and sees it as **a bundle of matrices.** All of linear
   * algebra passes here.
   *
   * **This creates a round trip.** The reason for doing it anyway is written in
   * `src/linalg.ts` — a decomposition is sequential, so there is almost nothing to
   * spread across the GPU, and at the sizes pushed through here launching a kernel costs
   * more than the computation.
   *
   * **Only the last two axes are the matrix; everything before is batch.** That is
   * torch's `linalg` rule, so `det((3,2,2))` gives `(3,)`. It used to accept one 2-D
   * square alone, and that was not an imitation but an absence — batches are the shape
   * real code always uses.
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

  /** A value built on the CPU, back to the GPU. */
  private static fromMat(a: LA.Mat, shape: readonly number[]): Tensor {
    return Tensor.from(LA.toF32(a), shape);
  }

  /** Joins what came out per batch into one tensor. */
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
   * A backward that uses **a different constant per batch.**
   *
   * The values came out per batch, so the backward's constants (the inverse, `L`, `L⁻¹`)
   * differ per batch too. `g` is flattened to a single batch axis, run one plate at a
   * time, and stacked again. With one batch it passes straight through — that is the
   * common case, and it is faster by one stacking.
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

  /** The inverse per batch. A singular matrix throws rather than quietly giving NaN. */
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
      // The sign is a step, so nothing flows.
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
  /**
   * **`rcond` is the cut-off below which a singular value counts as zero**, and it
   * changes the answer rather than only its precision — measured on torch, a 2×2 at
   * `rcond=0.9` comes back entirely different from the default.
   *
   * The seat was carried and **refused** while `LA.pinverse` decided the cut-off
   * itself — carried because leaving it out made `pinverse(rcond=0.9)` a surplus
   * argument, and JavaScript discards those. It takes one now, so the refusal is
   * gone and the number is computed: `rcond` scales to the largest singular value,
   * which is numpy's convention and torch's, and the core has always matched it.
   */
  async pinverse(rcond?: number): Promise<Tensor> {
    const v = await this.asBatch(false);
    const { rows: m, cols: n } = v;
    const ps = v.mats.map((a) => LA.pinverse(a, m, n, rcond));
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
   * The lower-triangular Cholesky. Throws when the matrix is not symmetric positive
   * definite.
   *
   * **The backward runs on the GPU.** The expression is `Ā = sym(L⁻ᵀ·Φ(Lᵀ·L̄)·L⁻¹)`,
   * where `Φ` takes the lower triangle and halves the diagonal, and all of it is matrix
   * operations, so it is written with kernels that already exist. Only `L` and `L⁻¹` are
   * computed on the CPU in the forward and carried in as constants — inside a backward
   * there is no waiting for the GPU, so those values have to be there already.
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
        // Φ — keep the lower triangle and halve the diagonal alone.
        const p = m.tril().sub(m.mul(eye).binary("mul", half));
        const abar = linvT.mm(p).mm(linvC);
        // A is symmetric, so the upper and lower triangles share one degree of
        // freedom — symmetrising is that share.
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
  async solve(other: Tensor): Promise<Tensor> {
    const v = await this.asBatch();
    const n = v.rows;
    const vector = other.shape.length === this.shape.length - 1;
    const cols = vector ? 1 : (other.shape[other.shape.length - 1] ?? 1);
    const rhsFlat = LA.fromF32(await other.toArray());
    const invTs = Tensor.invAll(v.mats, n, "solve")
      .map((m) => LA.transpose(m, n, n));
    const size = n * cols;
    const xs: LA.Mat[] = [];
    for (let i = 0; i < v.batch; i++) {
      xs.push(LA.luSolveFactored(
        LA.luFactor(v.mats[i]!, n, n), rhsFlat.slice(i * size, (i + 1) * size), cols));
    }
    const out = Tensor.fromBatch(xs, other.shape);
    const shape = this.shape;
    const gShape = vector ? [n] : [n, cols];
    return Tensor.make(
      out.buffer,
      other.shape,
      [this, other],
      (g) => {
        const gbs: Tensor[] = [];
        const ga = Tensor.perBatch(g, v.batch, gShape, shape, (gi, i) => {
          const at = Tensor.fromMat(invTs[i]!, [n, n]);
          const gb = at.mm(gi.reshape([n, cols]));
          gbs.push(gb.reshape(gShape));
          return gb.mm(Tensor.fromMat(xs[i]!, [n, cols]).transpose()).neg();
        });
        const gb = v.batch === 1
          ? gbs[0]!.reshape(other.shape)
          : Tensor.stack(gbs, 0).reshape(other.shape);
        return [
          this.requiresGrad ? ga : null,
          other.requiresGrad ? gb : null,
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
  async qr(some = true): Promise<{ q: Tensor; r: Tensor }> {
    // **`torch.qr` is not `torch.linalg.qr`.** The method takes a boolean and
    // `linalg.qr` takes a string; both are torch's, and this took the string on
    // both doors, so `x.qr(some=false)` — a line torch code contains — did not
    // reach the complete form. `linalg.qr` translates on its way in.
    const mode: "reduced" | "complete" = some ? "reduced" : "complete";
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
  async linalgSvd(fullMatrices = true):
    Promise<{ u: Tensor; s: Tensor; vt: Tensor }> {
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
      // Eigenvalues: `Ā = V·diag(Ḡ)·Vᵀ`. One line, and no overlap problem.
      values: this.linalgNode(Tensor.fromBatch(ws, [...v.lead, n]), (g) =>
        Tensor.perBatch(g, v.batch, [n], this.shape, (gb, b) => {
          const vec = Tensor.fromMat(vs[b]!, [n, n]);
          return vec.mm(gb.diagflat()).mm(vec.transpose());
        }), "EighBackward0"),
      // Eigenvectors: `Ā = sym(V·(F∘(Vᵀ·Ḡ))·Vᵀ)`.
      //
      // **The symmetrising must not be dropped.** `A` is symmetric, so the upper and
      // lower triangles share one degree of freedom, and the raw expression sends all of
      // it to one side. The diagonal is right and only the off-diagonal differs, so it
      // is invisible without a value comparison — chosen by measurement.
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
   * `torch.svd` — **which is not `torch.linalg.svd`.** Two functions in torch, and
   * borch.ts had one of them under both names.
   *
   * - **The default is reduced**, so a 3×2 gives a 3×2 `U`; `linalgSvd` defaults to
   *   the full form and gives 3×3. The same call returned **different shapes** from
   *   the two libraries, and the overlapping block agreed, so anything reading only
   *   `S` or `U[:, :k]` saw nothing.
   * - **The third field is `v`, not `vt`.** They are transposes.
   * - **`some` is the opposite of `fullMatrices`**, so a caller porting from torch
   *   passes `false` for the reduced form and receives the full one.
   *
   * The core (`borch/_ops.py`) splits the two the same way and for the same reason.
   */
  async svd(some = true, computeUv = true):
    Promise<{ u: Tensor; s: Tensor; v: Tensor }> {
    const { u, s, vt } = await this.linalgSvd(!some);
    const v = vt.swapaxes(-1, -2);
    if (!computeUv) {
      // torch hands back zeros of the shape they would have had rather than a
      // shorter tuple, so the caller's destructuring keeps working.
      return { u: u.mul(Tensor.full([], 0)), s, v: v.mul(Tensor.full([], 0)) };
    }
    return { u, s, v };
  }

  /**
   * The singular values only. The middle of `svd`.
   */
  async svdvals(): Promise<Tensor> {
    return (await this.linalgSvd(false)).s;
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
  async matrixNorm(ord: number | string = "fro",
                   dim: readonly [number, number] = [-2, -1],
                   keepdim = false): Promise<Tensor> {
    // **The two axes are brought to the end rather than the arithmetic moved to
    // them.** Every branch below reduces the last two, so naming any other pair
    // is a permutation and not a second implementation — and `dim` given as the
    // trailing pair, which is the default, permutes to itself.
    const rank = this.shape.length;
    const at = dim.map((d) => (d < 0 ? d + rank : d));
    const [d0, d1] = at as [number, number];
    if (d0 === d1 || d0 >= rank || d1 >= rank || d0 < 0 || d1 < 0) {
      throw new RuntimeError(
        `matrixNorm: dim=[${dim[0]}, ${dim[1]}] is not two axes of a rank-${rank} tensor`);
    }
    const rest = [...Array(rank).keys()].filter((i) => i !== d0 && i !== d1);
    const moved = rest.length === 0 && d0 === rank - 2 && d1 === rank - 1
      ? this : this.permute([...rest, d0, d1]);
    const flat = await moved.matrixNormOfLast(ord);
    if (!keepdim) return flat;
    // Back in the caller's axis order, with a 1 where each named axis stood.
    const kept = this.shape.map((_, i) => (i === d0 || i === d1 ? 1 : this.shape[i]!));
    return flat.reshape(kept);
  }

  /** `matrixNorm`'s arithmetic, always on the last two axes. */
  private async matrixNormOfLast(ord: number | string): Promise<Tensor> {
    if (ord === "nuc" || ord === 2 || ord === -2) {
      const s = await this.svdvals();
      if (ord === "nuc") return s.sumDim(-1, false);
      return ord === 2 ? s.amax(-1, false) : s.amin(-1, false);
    }
    if (ord === "fro") return this.square().sumDim(-1, false).sumDim(-1, false).sqrt();
    // 1 goes down the columns (summing rows); inf goes along the rows (summing
    // columns).
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
      // `X A = B` takes the same path once both sides are transposed.
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
  vectorNorm(ord = 2, dim?: number, keepdim = false): Tensor {
    // **`keepdim` was missing and every branch passed `false`.** torch keeps the
    // reduced axes as length 1, and without it `vector_norm(x, dim=1, keepdim=True)`
    // came back one rank short — which broadcasts against the input differently and
    // is the shape of divergence that surfaces somewhere else entirely.
    //
    // With no `dim` torch reduces everything and, asked to keep, returns **all ones**
    // rather than a scalar (measured). The flatten below makes the rank unavailable by
    // then, so the original is remembered first.
    const rank = this.shape.length;
    const flat = dim === undefined && rank > 1 ? this.reshape([this.size]) : this;
    const x = flat.abs();
    const whole = (t: Tensor) => (keepdim ? t.reshape(new Array(rank).fill(1)) : t);
    if (ord === Infinity) return x.amax(dim, keepdim);
    if (ord === -Infinity) return x.amin(dim, keepdim);
    if (ord === 0) return x.binary("ne", Tensor.full([], 0)).sumDim(dim ?? 0, keepdim);
    if (ord === 1) return dim === undefined ? whole(x.sum()) : x.sumDim(dim, keepdim);
    const powed = ord === 2 ? x.square() : x.powScalar(ord);
    const total = dim === undefined ? powed.sum() : powed.sumDim(dim, keepdim);
    const done = ord === 2 ? total.sqrt() : total.powScalar(1 / ord);
    return dim === undefined ? whole(done) : done;
  }

  /** The Vandermonde matrix. The columns are **increasing powers**. */
  vander(N?: number): Tensor {
    const n = N ?? this.size;
    const cols: Tensor[] = [];
    for (let k = 0; k < n; k++) cols.push(this.powScalar(k));
    return Tensor.stack(cols, 1);
  }

  /**
   * Folds the tensor into a matrix, solves, and spreads it back.
   */
  async tensorSolve(b: Tensor, dims?: readonly number[]): Promise<Tensor> {
    // **`dims` moves those axes to the end before the fold**, in the order given,
    // with the rest keeping theirs. It therefore changes which axes become the
    // matrix — and the answer's shape is the moved array's trailing axes, not this
    // one's, which is the part a caller reading `slice(b.shape.length)` off the
    // receiver would get wrong.
    const a: Tensor = dims === undefined ? this : this.permute([
      ...this.shape.map((_, k) => k).filter((k) => !dims.includes(k)), ...dims]);
    const n = b.size;
    const folded = a.reshape([n, a.size / n]);
    const x = await folded.solve(b.reshape([n]));
    return x.reshape(a.shape.slice(b.shape.length));
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

  async matrixRank(tol?: number): Promise<Tensor> {
    const v = await this.asBatch(false);
    return Tensor.from(
      v.mats.map((a) => LA.matrixRank(a, v.rows, v.cols, tol)), v.lead);
  }

  /**
   * The least-squares solution. Square and invertible, it gives the same
   * answer as `solve`.
   */
  /**
   * **This method takes one argument and keeps taking one.** torch removed
   * `Tensor.lstsq` and the core carries it as a tombstone with a single `other`,
   * so `rcond` and `driver` — which live on `linalg.lstsq`, where torch puts them
   * — do not belong on it. Given them here the method read *longer than the core*
   * on the signature axis, which is exactly the report that argument was for.
   */
  async lstsq(other: Tensor): Promise<Tensor> {
    const v = await this.asBatch(false);
    if (v.batch !== 1) throw new RuntimeError("lstsq: batching is not here yet");
    const { rows, cols } = v;
    const width = other.shape.length === 1 ? 1 : (other.shape[other.shape.length - 1] ?? 1);
    const rhs = LA.fromF32(await other.toArray());
    const sol = LA.matmul(
      LA.pinverse(v.mats[0]!, rows, cols), rhs, cols, rows, width);
    return Tensor.fromMat(sol, other.shape.length === 1 ? [cols] : [cols, width]);
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
   * A symmetric matrix as `L D Lᵀ` — LAPACK's `dsytf2` on the lower triangle,
   * **with pivoting.**
   *
   * **This refused an indefinite matrix and the reason was accurate**: torch uses
   * Bunch–Kaufman, which swaps where it needs to, and a factorisation without the
   * swaps is a different one. So the way through was to write Bunch–Kaufman rather
   * than to loosen a tolerance — there are many valid `L D Lᵀ` decompositions and
   * only LAPACK's packing and swap table compare against torch's.
   *
   * The pivot table is LAPACK's own: a positive `k+1` is a 1×1 pivot with row `k`
   * swapped for row `pivots[k]−1`, and **a repeated negative pair is a 2×2 block**.
   *
   * **The swap is over columns, not rows**, and that one line is where the core's
   * copy of this first went wrong: written as a row swap, ten of thirteen matrices
   * still agreed, and the three that did not diverged first in their *pivot table*
   * two steps later.
   *
   * Checked against torch on 470 symmetric matrices, ranks 1 to 8.
   */
  async ldlFactor(): Promise<{ LD: Tensor; pivots: Tensor }> {
    const got = await this.ldlPacked();
    if (got.info.some((v) => v !== 0)) {
      const first = got.info.find((v) => v !== 0) ?? 0;
      throw new RuntimeError(
        `linalg.ldl_factor: the leading minor of order ${first} is singular — `
        + "`ldlFactorEx` reports it in `info` instead of stopping");
    }
    return { LD: got.LD, pivots: got.pivots };
  }

  /** Bunch–Kaufman itself, with `info` rather than a refusal. */
  private async ldlPacked():
    Promise<{ LD: Tensor; pivots: Tensor; info: number[] }> {
    const v = await this.asBatch();
    const n = v.rows;
    const ALPHA = (1 + Math.sqrt(17)) / 8;
    const piv = new Float32Array(v.batch * n);
    const info: number[] = [];
    const outs = v.mats.map((mat, plate) => {
      const a = new Float64Array(n * n);
      for (let i = 0; i < n; i++) {
        for (let j = 0; j <= i; j++) a[i * n + j] = mat[i * n + j] ?? 0;
      }
      const at = (i: number, j: number): number => a[i * n + j] ?? 0;
      let bad = 0;
      let k = 0;
      while (k < n) {
        let step = 1;
        const here = Math.abs(at(k, k));
        let imax = -1;
        let colmax = 0;
        for (let i = k + 1; i < n; i++) {
          if (Math.abs(at(i, k)) > colmax) { colmax = Math.abs(at(i, k)); imax = i; }
        }
        let kp: number;
        if (Math.max(here, colmax) === 0 || here >= ALPHA * colmax) {
          kp = k;
        } else {
          let rowmax = 0;
          for (let j = k; j < imax; j++) rowmax = Math.max(rowmax, Math.abs(at(imax, j)));
          for (let i = imax + 1; i < n; i++) {
            rowmax = Math.max(rowmax, Math.abs(at(i, imax)));
          }
          if (here >= ALPHA * colmax * (colmax / rowmax)) kp = k;
          else if (Math.abs(at(imax, imax)) >= ALPHA * rowmax) kp = imax;
          else { kp = imax; step = 2; }
        }
        const kk = k + step - 1;
        if (kp !== kk) {
          for (let i = kp + 1; i < n; i++) {
            const keep = at(i, kk);
            a[i * n + kk] = at(i, kp);
            a[i * n + kp] = keep;
          }
          for (let j = kk + 1; j < kp; j++) {
            const keep = at(j, kk);
            a[j * n + kk] = at(kp, j);
            a[kp * n + j] = keep;
          }
          const keepDiag = at(kk, kk);
          a[kk * n + kk] = at(kp, kp);
          a[kp * n + kp] = keepDiag;
          if (step === 2) {
            const keepPair = at(kk, k);
            a[kk * n + k] = at(kp, k);
            a[kp * n + k] = keepPair;
          }
        }
        if (step === 1) {
          const d11 = at(k, k);
          if (d11 === 0 && bad === 0) bad = k + 1;
          if (d11 !== 0) {
            for (let i = k + 1; i < n; i++) a[i * n + k] = at(i, k) / d11;
            for (let j = k + 1; j < n; j++) {
              for (let i = j; i < n; i++) {
                a[i * n + j] = at(i, j) - d11 * at(i, k) * at(j, k);
              }
            }
          }
          // **`kp + 1`, not `k + 1`.** A 1×1 pivot still records where the row came
          // from; writing the position instead is the identity whenever nothing
          // swapped, so every unswapped matrix agreed and only the swapped ones
          // parted — and they parted in the pivot table, not the values.
          piv[plate * n + k] = kp + 1;
        } else {
          if (k < n - 2) {
            const d21raw = at(k + 1, k);
            const d11 = at(k + 1, k + 1) / d21raw;
            const d22 = at(k, k) / d21raw;
            const d21 = (1 / (d11 * d22 - 1)) / d21raw;
            for (let j = k + 2; j < n; j++) {
              const wk = d21 * (d11 * at(j, k) - at(j, k + 1));
              const wkp1 = d21 * (d22 * at(j, k + 1) - at(j, k));
              for (let i = j; i < n; i++) {
                a[i * n + j] = at(i, j) - at(i, k) * wk - at(i, k + 1) * wkp1;
              }
              a[j * n + k] = wk;
              a[j * n + k + 1] = wkp1;
            }
          }
          piv[plate * n + k] = -(kp + 1);
          piv[plate * n + k + 1] = -(kp + 1);
        }
        k += step;
      }
      info.push(bad);
      return a;
    });
    return {
      LD: Tensor.fromBatch(outs, this.shape),
      pivots: Tensor.from(piv, [...v.lead, n], { dtype: "int64" }),
      info,
    };
  }

  /**
   * `ldlFactor` **plus one `info`** — the first zero pivot, counting from 1.
   *
   * It used to be always 0, with the note that a bad position makes `ldlFactor`
   * refuse on the spot. That was true while every such matrix was refused; now the
   * only ones left are singular, and this is where they are reported rather than
   * stopped. Measured against torch: `[[1,1],[1,1]]` gives 2 and a zero matrix 1.
   */
  async ldlFactorEx(): Promise<{ LD: Tensor; pivots: Tensor; info: Tensor }> {
    const v = await this.asBatch();
    const got = await this.ldlPacked();
    return {
      LD: got.LD,
      pivots: got.pivots,
      info: Tensor.from(got.info, v.lead, { dtype: "int64" }),
    };
  }

  /**
   * Solves using the factorisation `ldlFactor` produced — LAPACK's `dsytrs`,
   * **pivots and all.**
   *
   * `pivots` was not taken here at all, which was right only while nothing was ever
   * swapped: the factorisation refused every matrix that needed it. With
   * Bunch–Kaufman in place the old body was wrong on 47 of 80 random symmetric
   * matrices and returned a plausible number every time.
   *
   * **Swap, eliminate, divide — in that order, one step at a time.** Written as
   * *permute, then solve `L`, then solve `D`* it is still wrong, because `L` was
   * built in the swapped order and the two do not commute; that version was wrong on
   * 97 of 279. And **inside a 2×2 block the sub-diagonal entry belongs to `D`**, not
   * to the unit triangle around it.
   */
  async ldlSolve(pivots: Tensor, b: Tensor): Promise<Tensor> {
    const v = await this.asBatch();
    const n = v.rows;
    if (v.batch !== 1) throw new RuntimeError("ldl_solve: batching is not here yet");
    const ld = v.mats[0];
    if (!ld) throw new RuntimeError("ldl_solve: the factorization is empty");
    const piv = Array.from(await pivots.toArray()).map((v2) => Math.round(v2));
    const width = b.shape.length === 1 ? 1 : (b.shape[b.shape.length - 1] ?? 1);
    const rhs = LA.fromF32(await b.toArray());
    const x = Float64Array.from(rhs);
    const at = (i: number, j: number): number => ld[i * n + j] ?? 0;
    const row = (i: number, c: number): number => x[i * width + c] ?? 0;
    const swap = (i: number, j: number): void => {
      for (let c = 0; c < width; c++) {
        const keep = row(i, c);
        x[i * width + c] = row(j, c);
        x[j * width + c] = keep;
      }
    };
    let k = 0;
    while (k < n) {
      if ((piv[k] ?? 0) > 0) {
        const kp = (piv[k] ?? 0) - 1;
        if (kp !== k) swap(k, kp);
        for (let i = k + 1; i < n; i++) {
          for (let c = 0; c < width; c++) x[i * width + c] = row(i, c) - at(i, k) * row(k, c);
        }
        for (let c = 0; c < width; c++) x[k * width + c] = row(k, c) / at(k, k);
        k += 1;
      } else {
        const kp = -(piv[k] ?? 0) - 1;
        if (kp !== k + 1) swap(k + 1, kp);
        for (let i = k + 2; i < n; i++) {
          for (let c = 0; c < width; c++) {
            x[i * width + c] = row(i, c) - at(i, k) * row(k, c)
              - at(i, k + 1) * row(k + 1, c);
          }
        }
        const off = at(k + 1, k);
        const top = at(k, k) / off;
        const bot = at(k + 1, k + 1) / off;
        const denom = top * bot - 1;
        for (let c = 0; c < width; c++) {
          const b0 = row(k, c) / off;
          const b1 = row(k + 1, c) / off;
          x[k * width + c] = (bot * b0 - b1) / denom;
          x[(k + 1) * width + c] = (top * b1 - b0) / denom;
        }
        k += 2;
      }
    }
    // A 2×2 block is met at its **second** row on the way up, and both of its
    // columns are applied before the pair steps past.
    k = n - 1;
    while (k >= 0) {
      const back = (target: number, col: number): void => {
        let s = row(target, col);
        for (let i = k + 1; i < n; i++) s -= at(i, target) * row(i, col);
        x[target * width + col] = s;
      };
      if ((piv[k] ?? 0) > 0) {
        for (let c = 0; c < width; c++) back(k, c);
        const kp = (piv[k] ?? 0) - 1;
        if (kp !== k) swap(k, kp);
        k -= 1;
      } else {
        for (let c = 0; c < width; c++) { back(k, c); back(k - 1, c); }
        const kp = -(piv[k] ?? 0) - 1;
        if (kp !== k) swap(k, kp);
        k -= 2;
      }
    }
    return Tensor.fromMat(x, b.shape);
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
      // torch returns only as many columns as the input had.
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
  /**
   * **The two seats are carried so they can be refused.**
   *
   * `pivot=false` is LU without row swaps, which torch declines on the CPU too
   * (measured: `linalg.lu_factor: LU without pivoting is not implemented`), and
   * `getInfos=true` makes torch return a **third** value — so it changes the shape of
   * the answer rather than its contents, which is the kind of argument that must not
   * be dropped in silence.
   */
  async lu(pivot = true, getInfos = false):
      Promise<{ P: Tensor; L: Tensor; U: Tensor }> {
    if (!pivot) {
      throw new Error("lu(pivot=false) — LU without pivoting is not implemented");
    }
    if (getInfos) {
      throw new Error(
        "lu(getInfos=true) — the third value torch returns is not carried across");
    }
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

  /** Solves `A x = b` from what `luFactor` produced. */
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
  async luSolveFactored(pivots: Tensor, b: Tensor,
                        left = true, adjoint = false): Promise<Tensor> {
    if (!left) {
      // `X A = B` is `Aᵀ Xᵀ = Bᵀ`, so the right-hand solve is the left one on the
      // transposed sides with the adjoint flipped. Real storage only, so torch's
      // `Aᴴ` is a transpose.
      const t = await this.luSolveFactored(
        pivots, b.transpose(-1, -2), true, !adjoint);
      return t.transpose(-1, -2);
    }
    const v = await this.asBatch();
    const n = v.rows;
    // **Batched, by the same loop `solve` twenty lines up already uses.** This
    // refused a batch outright, which was honest; what made it worth closing is
    // that the core answered a batch and got it **wrong** — one permutation built
    // from the flattened pivots and applied to the batch axis. A refusal here and a
    // wrong number there is the pair a golden case cannot ask about, because the
    // case cannot be written while one side stops.
    const width = b.shape.length === this.shape.length - 1
      ? 1 : (b.shape[b.shape.length - 1] ?? 1);
    const piv = Int32Array.from(await pivots.toArray());
    const rhsFlat = LA.fromF32(await b.toArray());
    const size = n * width;
    const xs: LA.Mat[] = [];
    for (let i = 0; i < v.batch; i++) {
      xs.push(LA.luSolveFactored(
        // Each matrix has its own pivots — `n` of them, in the same order as the
        // matrices. Sharing one row of pivots across the batch is exactly the fault
        // the core had.
        { lu: v.mats[i]!, piv: piv.slice(i * n, (i + 1) * n), rows: n, cols: n },
        rhsFlat.slice(i * size, (i + 1) * size), width, adjoint));
    }
    return Tensor.fromBatch(xs, b.shape);
  }

  // ── Linear algebra at the top level ───────────────────────────────────
  //
  // **The argument order differs from `linalg`'s.** torch kept the old names at the top
  // level, and most of them take **the right-hand side first.** It is the same
  // computation, so there is one copy of it and only the slots move — and whether that
  // move is right is confirmed by values alone.

  /**
   * Stands the factor up **as a lower triangle**, so that `A = L Lᵀ`.
   *
   * Left as an assembly — passing through `tril` and `transpose` makes **the gradient
   * flow to the factor as well.** Slicing out the values alone sends the backward to `b`
   * only, while torch flows to the factor too (measured).
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
  async choleskySolve(input2: Tensor, upper = false): Promise<Tensor> {
    const low = Tensor.asLower(input2, upper);
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
      // **The diagonal is not read; it is taken as 1.** Left as it is, a quietly
      // different answer comes out.
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
  async luSolve(luData: Tensor, luPivots: Tensor): Promise<Tensor> {
    return luData.luSolveFactored(luPivots, this);
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
  async orgqr(input2: Tensor): Promise<Tensor> {
    return this.householderProduct(input2);
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
    input2: Tensor,
    input3: Tensor,
    left = true,
    transpose = false,
  ): Promise<Tensor> {
    const v = await this.asBatch(false);
    const { rows: m, cols: n } = v;
    const taus = await input2.toArray();
    const k = input2.shape[input2.shape.length - 1] ?? 0;
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
    const c = LA.fromF32(await input3.toArray());
    const [cr, cc] = input3.shape.length === 1
      ? [input3.shape[0] ?? 0, 1]
      : [input3.shape[0] ?? 0, input3.shape[1] ?? 0];
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
   *
   * **`b` is the generalised problem `A x = λ B x`, and it was refused.** With `B`
   * symmetric positive definite it reduces to a standard one in four lines:
   * `B = L Lᵀ`, the eigenvalues of `L⁻¹ A L⁻ᵀ` are the generalised ones, and
   * `x = L⁻ᵀ y` are the generalised vectors. Nothing iterative was needed.
   *
   * **Those vectors come out `B`-orthonormal**, not unit length — `xᵀBx = 1` and
   * `xᵀx = 0.996` on the fixture. It falls out of the reduction (`xᵀBx = yᵀy`) and
   * it is what torch returns; normalising them would look tidier and disagree.
   *
   * **`x` is a starting basis and this has nothing to start.** What it changes is
   * the count: given `x` and no `k`, torch takes `k` from its columns, which is why
   * `k` is `null` here rather than 1. The converged eigenvalues do not depend on it
   * — torch with and without agrees to 5e-6, the distance its own answer moves with
   * the seed.
   */
  async lobpcg(k: number | null = null, largest = true,
               b: Tensor | null = null, x: Tensor | null = null): Promise<{
    eigenvalues: Tensor; eigenvectors: Tensor;
  }> {
    const want = k ?? (x === null ? 1 : (x.shape[x.shape.length - 1] ?? 1));
    let values: Tensor;
    let vectors: Tensor;
    if (b === null) {
      ({ values, vectors } = await this.eigh());
    } else {
      const low = await b.cholesky();
      // `L⁻¹ A L⁻ᵀ`, symmetrised — the two solves are the reduction.
      const half = await low.solveTriangular(this, false);
      const inner = await low.solveTriangular(half.transpose(), false);
      const sym = inner.transpose().add(inner).mul(Tensor.full([], 0.5));
      const got = await sym.eigh();
      values = got.values;
      vectors = await low.transpose().solveTriangular(got.vectors, true);
    }
    const n = values.shape[values.shape.length - 1] ?? 0;
    const picks: number[] = [];
    for (let i = 0; i < want; i++) picks.push(largest ? n - 1 - i : i);
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
    // **Accepted and unused — because there is nothing to refine.** `niter` is how
    // many times torch refines a random subspace, and that iteration exists because the
    // projection is an approximation. We compute the full SVD, so the first answer is
    // already the converged one — the values sit where torch's would with `niter` taken
    // large.
    //
    // So **at small `niter` the values diverge from torch's.** It is a divergence in the
    // direction of being more accurate, and a divergence is still a divergence, so it is
    // written in the README. Without the reason written down, the next person reads it
    // as "not fully implemented" and puts the approximation back.
    void niter;
    const src = M === null ? this : this.sub(M);
    const { u, s, vt } = await src.linalgSvd(false);
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

  /** The values already came out on the CPU; this only joins up the graph. */
  private linalgNode(
    value: Tensor,
    backwardFn: (g: Tensor) => Tensor,
    gradName: string,
  ): Tensor {
    return Tensor.make(value.buffer, value.shape, [this],
      (g) => [backwardFn(g)], gradName);
  }

  // ── In-place operations ───────────────────────────────────────────────

  /**
   * The shared gate for in-place operations.
   *
   * **A leaf with gradients switched on is refused.** torch does the same, and there is
   * a reason — change a leaf's value and a backward that already used that value uses a
   * wrong number, with nobody able to tell. The optimiser does edit the weights in
   * place, and it does so inside `no_grad`, so it does not pass here.
   *
   * The result is built in a new buffer and then moved over. Writing while reading has
   * no thread ordering and mixes.
   *
   * **It spreads through views.** The `reshape` family shares the buffer, so
   * `a.view(2,2).add_(10)` changes `a` too — the same as torch. The sister library
   * refuses it, because TF.js tensors are immutable.
   */
  private mutate(compute: () => Tensor): Tensor {
    if (gradMode.enabled && this.requiresGrad && this.parents.length === 0) {
      throw new RuntimeError(
        "a leaf Variable that requires grad is being used in an in-place operation.",
      );
    }
    const result = compute();
    // **The result may be the same buffer.** `squeeze` and `unsqueeze` move no values
    // and change only the frame, so like `reshape` they pass the buffer straight on. A
    // copy issued there makes WebGPU invalidate the whole command buffer for "source and
    // destination are the same buffer", and then **a case queued behind it is wrong
    // instead** — the next case really did fail on a nonsense value.
    // **The cell count may shrink.** `as_strided_` changes not only the frame but the
    // size. Copying at the original size makes WebGPU invalidate **the whole command
    // buffer** for "reading beyond the source buffer", and then this case passes while
    // **a case queued behind it is wrong instead** — the next case really did fail on a
    // nonsense value.
    if (result.size > this.size) {
      throw new RuntimeError(
        "an in-place operation cannot grow the element count — the buffer is not that large.",
      );
    }
    if (result.buffer !== this.buffer) {
      dev().copyInto(this.buffer, result.buffer, result.size);
    }
    this.size = result.size;
    // **Some of them change the shape.** `transpose_` keeps the cell count and changes
    // the frame; `squeeze_` and `unsqueeze_` only add and remove an axis. Moving the
    // values while leaving the shape alone **passes only when asked with a square** —
    // the core's 2×2 case really did miss it. The cell count does not change, so `size`
    // stays.
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

  div_(other: number, roundingMode: "trunc" | "floor" | null = null): Tensor {
    // **`div` took `roundingMode` and `div_` did not**, so the two spellings of one
    // operation had different arguments and the in-place one quietly did true
    // division where torch floors. The core had the same pair, found in the same run.
    return this.mutate(() => this.div(Tensor.full([], other), roundingMode));
  }

  pow_(exponent: number): Tensor {
    return this.mutate(() => this.powScalar(exponent));
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
  requiresGrad_(requiresGrad = true): Tensor {
    this.requiresGrad = requiresGrad;
    return this;
  }

  fill_(value: number): Tensor {
    return this.mutate(() => Tensor.full(this.shape, value));
  }

  clamp_(min: number, max: number): Tensor {
    return this.mutate(() => this.clamp(min, max));
  }

  /**
   * The same as `clamp_` — torch carries both names.
   */
  clip_(min: number, max: number): Tensor {
    return this.clamp_(min, max);
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

  /**
   * Writes back whatever the callback computes. **The general form of
   * `inplaceUnary`**, for the two operations whose out-of-place half takes an
   * argument the `UNARY` table cannot give it — `round_(decimals)` and
   * `logit_(eps)`, both attached after the table for the ordering reason `abs` and
   * `elu` are.
   */
  inplaceFrom(compute: () => Tensor): Tensor {
    return this.mutate(compute);
  }

  /**
   * The table's **binary** operations, in place. Names like `hypot_` come here.
   *
   * It exists for the same reason `inplaceUnary` does: `mutate` is private, and the
   * loop that attaches these lives outside the class. Reaching into the private
   * from out there would have needed a cast, and a cast is how a class stops being
   * able to say what it owns.
   */
  inplaceBinary(name: string, other: Tensor): Tensor {
    return this.mutate(() => this.binary(name, other));
  }

  // In-place operations that take arguments. They cannot run off a table, so they are
  // written one by one, and **the computation belongs to the underscore-less side** —
  // one expression in two copies diverges eventually, and the values are plausible
  // enough that it stays invisible.

  // **Ten more, and these really are one by one.** The note above is right about
  // *this* family: `indexReduce_` takes five arguments and `swapaxes_` takes two,
  // so there is no shape for a table to hold. (The twelve elementwise binaries were
  // the exception and come off one — `INPLACE_BINARY` below.)
  //
  // Every partner was already here. What was missing was the underscore, which is
  // the sixth time in this session that a name read as absent while the work sat
  // beside it.

  conjPhysical_(): Tensor {
    return this.mutate(() => this.conjPhysical());
  }

  // **torch's names, which the partner three thousand lines up already uses.** This
  // took `a`/`b` where `swapaxes` takes `axis0`/`axis1`, so the two spellings of one
  // operation named their arguments differently and a keyword call reached one and
  // not the other.
  swapaxes_(axis0: number, axis1: number): Tensor {
    return this.swapaxesInPlace(axis0, axis1);
  }

  private swapaxesInPlace(a: number, b: number): Tensor {
    return this.mutate(() => this.swapaxes(a, b));
  }

  maskedFill_(mask: Tensor, value: number): Tensor {
    return this.mutate(() => this.maskedFill(mask, value));
  }

  indexAdd_(dim: number, index: Tensor, source: Tensor, alpha = 1): Tensor {
    return this.mutate(() => this.indexAdd(dim, index, source, alpha));
  }

  indexCopy_(dim: number, index: Tensor, source: Tensor): Tensor {
    return this.mutate(() => this.indexCopy(dim, index, source));
  }

  indexFill_(dim: number, index: Tensor, value: number): Tensor {
    return this.mutate(() => this.indexFill(dim, index, value));
  }

  indexReduce_(
    dim: number, index: Tensor, source: Tensor, reduce: string,
    includeSelf = true,
  ): Tensor {
    return this.mutate(() => this.indexReduce(dim, index, source, reduce, includeSelf));
  }

  /**
   * `torch.Tensor.scatter` — **overwrite at the indexed positions.** borch.ts calls
   * it `scatterSet`, so torch's own name was absent while the operation was there.
   *
   * `reduce` is torch's deprecated overload, kept because torch still answers it.
   * Given, it combines *onto what is there* instead of overwriting, and colliding
   * indices accumulate — which is `scatterReduce(…, includeSelf = true)` under
   * another two words, verified against torch rather than reasoned from the docs.
   */
  scatter(dim: number, index: Tensor, src: Tensor, reduce?: string): Tensor {
    if (reduce === undefined) return this.scatterSet(dim, index, src);
    return this.scatterOnto(dim, index, src, reduce);
  }

  scatter_(dim: number, index: Tensor, src: Tensor, reduce?: string): Tensor {
    return this.mutate(() => this.scatter(dim, index, src, reduce));
  }

  /**
   * The `reduce` half of `scatter`, and **it refuses to differentiate.**
   *
   * The value is `scatterReduce`'s, which does have a backward here — so the
   * refusal is a choice, not a limit. torch raises `derivative for aten::scatter is
   * not implemented` for this overload, and computing a slope where torch stops
   * would be the failure this repository keeps finding from the other side: the
   * number comes out, nothing marks it as ours alone, and it is wrong only in
   * training, where nobody is reading.
   */
  private scatterOnto(
    dim: number, index: Tensor, src: Tensor, reduce: string,
  ): Tensor {
    if (reduce !== "add" && reduce !== "multiply") {
      throw new RuntimeError(
        `scatter's \`reduce\` takes 'add' or 'multiply'; got ${JSON.stringify(reduce)}. `
        + "The wider set ('sum', 'prod', 'mean', 'amax', 'amin') belongs to "
        + "`scatterReduce`, which is the replacement torch points at."
        + "\n(torch: reduce argument must be either add or multiply.)");
    }
    const value = this.detach().scatterReduce(
      dim, index, src.detach(), reduce === "add" ? "sum" : "prod", true);
    return Tensor.make(value.raw, value.shape, [this, src], () => {
      throw new RuntimeError(
        "scatter with `reduce` has no gradient — torch does not define one either, "
        + "and a slope invented here would be wrong only in training."
        + "\n(torch: derivative for aten::scatter is not implemented)");
    }, "ScatterBackward0", value.dtype);
  }

  scatterAdd_(dim: number, index: Tensor, src: Tensor): Tensor {
    return this.mutate(() => this.scatterAdd(dim, index, src));
  }

  scatterReduce_(
    dim: number, index: Tensor, src: Tensor, reduce: string, includeSelf = true,
  ): Tensor {
    return this.mutate(() => this.scatterReduce(dim, index, src, reduce, includeSelf));
  }

  /**
   * `torch.Tensor.copy_` — the values of `src`, written into this one. borch.ts
   * spells it `copyFrom`, so torch's name was absent while the operation was here.
   */
  copy_(src: Tensor, nonBlocking = false): Tensor {
    // `nonBlocking` asks torch not to wait on an async device copy. Every write
    // here has landed by the time this returns — carried because the seat is
    // torch's, and a caller who passes it should not get a type error.
    void nonBlocking;
    return this.copyFrom(src);
  }

  /**
   * `torch.Tensor.contiguous`. **Always this tensor**, and here that is right.
   *
   * The core had to be fixed for handing back `self`: numpy holds genuinely
   * non-contiguous views, so after a transpose `is_contiguous()` gave the opposite
   * answer from torch. **This side has no such state** — `strides()` is computed
   * from the shape every time, and every shape operation materialises through a
   * plan rather than re-describing the same buffer.
   *
   * So the same two lines are a defect there and the truth here, and the difference
   * is a property of the storage rather than of the code. Written down because the
   * next person to read these will have read the core's comment first.
   */
  contiguous(memoryFormat: string | null = null): Tensor {
    noMemoryFormat("contiguous", memoryFormat);
    return this;
  }

  /** See `contiguous` — there is no non-contiguous tensor on this side. */
  isContiguous(memoryFormat: string | null = null): boolean {
    noMemoryFormat("isContiguous", memoryFormat);
    return true;
  }

  /** Split along **axis 2**. `torch.Tensor.dsplit`. */
  dsplit(sections: number): Tensor[] {
    return this.splitParts(2, sections);
  }

  /**
   * `torch.Tensor.type`. **With no argument it names the type** — the first thing
   * anybody writes to find out what they are holding — and with one it converts.
   */
  type(dtype?: DType, nonBlocking = false): string | Tensor {
    void nonBlocking;
    if (dtype === undefined) {
      const name = { float32: "Float", int64: "Long", bool: "Bool",
                     complex64: "ComplexFloat" }[this.dtype as string] ?? "Float";
      return `torch.${name}Tensor`;
    }
    return this.to(dtype);
  }

  // ── The `new_*` family, and four more built from what is here ─────────
  //
  // torch's `new_*` make a tensor **carrying this one's dtype**, which is the whole
  // point of the name — `Tensor.zeros(shape)` cannot know it. The core's take
  // `(*size, dtype, requires_grad)`; the shape is a list here because TypeScript has
  // no `*args` that also takes a list.

  /** Zeros of `shape`, in **this** tensor's dtype. `torch.Tensor.new_zeros`. */
  newZeros(shape: readonly number[]): Tensor {
    return Tensor.zeros([...shape]).to(this.dtype);
  }

  /** Ones of `shape`, in this tensor's dtype. */
  newOnes(shape: readonly number[]): Tensor {
    return Tensor.ones([...shape]).to(this.dtype);
  }

  /**
   * A tensor of `shape` in this dtype, **values undefined**. torch gives whatever
   * the allocation held; this gives zeros, and the golden asks about the shape and
   * the dtype only — pinning the values would make them the specification.
   */
  newEmpty(shape: readonly number[]): Tensor {
    return this.newZeros(shape);
  }

  /** `shape` filled with `value`, in this tensor's dtype. */
  newFull(
    size: readonly number[], fillValue: number, dtype: DType | null = null,
    device: string | null = null, requiresGrad = false,
    layout: string | null = null, pinMemory = false,
  ): Tensor {
    // torch's seven, in torch's order. The last four are **carried and refused**:
    // left out, a positional call that reaches them lands on nothing, and the core
    // refuses at the same seats for the same reason.
    if (device !== null) throw new Error("newFull(device) is not here — there is one device.");
    if (layout !== null) throw new Error("newFull(layout) is not here — there is one layout.");
    if (pinMemory) throw new Error("newFull(pinMemory) is not here — there is no host pinning.");
    const out = Tensor.full([...size], fillValue).broadcastTo([...size])
      .to(dtype ?? this.dtype);
    out.requiresGrad = requiresGrad;
    return out;
  }

  /** `torch.Tensor.is_same_size` — shapes equal, and nothing about the values. */
  isSameSize(other: Tensor): boolean {
    return this.shape.length === other.shape.length
      && this.shape.every((n, i) => n === other.shape[i]);
  }

  /**
   * The elements at flat positions, ignoring shape. `torch.Tensor.take`.
   *
   * **It is not `gather`.** `gather` walks one axis and keeps the others;
   * this reads the tensor as one long row and always gives back a 1-D answer.
   */
  take(index: Tensor): Tensor {
    return this.reshape([this.size]).indexSelect(0, index);
  }

  /**
   * `torch.Tensor.take_along_dim` — `gather` along `dim`, and **`take` when `dim` is
   * left out**, which is torch's rule and the reason the two names are one function.
   */
  takeAlongDim(indices: Tensor, dim: number | null = null): Tensor {
    return dim === null ? this.take(indices) : this.gather(dim, indices);
  }

    /** `torch.Tensor.var`. borch.ts spells the operation `variance`. */
  var(dim?: number, correction = 1, keepdim = false, unbiased?: boolean): Tensor {
    // **`unbiased` is torch's older spelling and torch still takes it**, so refusing
    // it would be a divergence rather than a tidy-up — even though its own docstring
    // no longer documents it. Given, it wins: that is what an older call meant.
    const take = unbiased === undefined ? correction : (unbiased ? 1 : 0);
    return this.variance(dim, take, keepdim);
  }

  // **Both took less than their partners one screen up.** `transpose(dim0, dim1)`
  // and `squeeze(...dim)` carry torch's whole list; the in-place twins took nothing
  // and one axis, so `x.transpose_(0, 1)` dropped both numbers and did the 2-D swap,
  // and `x.squeeze_(0, 2)` dropped the second. Same shape as `divide_`, which was
  // narrower than `div_` for the same reason: an in-place twin written by hand
  // rather than derived.
  transpose_(dim0?: number, dim1?: number): Tensor {
    return this.mutate(() => this.transpose(dim0, dim1));
  }

  squeeze_(...dim: number[]): Tensor {
    return this.mutate(() => this.squeeze(...dim));
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

  // **`dtype` was on the partner and not on these.** In place means *the same
  // arithmetic, written back* — so the two spellings of one operation must take one
  // list, and `x.cumsum_(0, "int64")` was a word JavaScript dropped. It could not be
  // seen until the core's forwarders declared what they forward: the signature axis
  // filed both rows as *no python signature*, one of ninety-seven in that bucket.
  cumsum_(dim = 0, dtype?: DType): Tensor {
    return this.mutate(() => this.cumsum(dim, dtype));
  }

  cumprod_(dim = 0, dtype?: DType): Tensor {
    return this.mutate(() => this.cumprod(dim, dtype));
  }

  // ── In the kernel tables, with no name to type ────────────────────────
  //
  // borch.ts gives the elementwise operations as the tables `binary(name, other)` and
  // `unary(name)` **rather than a method per name.** There is only one kernel, so that
  // is shorter, and adding an operation does not add a method.
  //
  // **But the line a user types is `x.gcd(y)`.** Only somebody who knows the table can
  // type `x.binary("gcd", y)`, and code ported from torch does not know that name. The
  // eleven below exist for that — the computation was already there and what was missing
  // was **how to call it.**
  //
  // This branch was hidden behind the 40 `inplace::짝에서::` cases (a golden case
  // name, so it stays as it is). The gap table's
  // reason read "alias", and only ten of them were aliases.

  /**
   * Takes a number or a tensor.
   *
   * torch writes both `x.bitwise_and_(3)` and `x.gcd_(y)` — accepting only a tensor
   * makes the first simply not run, and that divergence lives in the signature alone, so
   * no value check catches it.
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

  // ── torch's second spellings ──────────────────────────────────────────
  //
  // Places where the computation was already there and **only the spelling to call it**
  // was missing. Which one a textbook or ported code types is the author's taste, so
  // with one of them present the other stops that code with an `AttributeError`.

  multiply(other: Tensor): Tensor {
    return this.mul(other);
  }

  // Five second spellings for unary operations. The names are not in the table, so the
  // loop does not attach them.
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

  // ── The same six under torch's short names ──────────────────────────────
  //
  // torch carries both spellings — `gt` beside `greater`, `le` beside
  // `less_equal` — and the five above were the long half only. The short half is
  // what tutorial code is written in, so `x.gt(0)` raised while `x.greater(0)`
  // worked, and the kernels were the same kernels either way.
  //
  // **`eq` is the one that is not a rename.** torch's long form for `ne` is
  // `not_equal` and for `lt` is `less`, but `eq`'s neighbour `equal` is a
  // different function — it reduces two tensors to one boolean, where `eq`
  // compares position by position. Giving `eq` the elementwise body and leaving
  // `equal` alone is the whole of the care needed here; folding them together
  // would return a scalar where torch returns a mask.

  eq(other: Tensor | number): Tensor {
    return this.binary("eq", Tensor.asTensor(other));
  }

  ne(other: Tensor | number): Tensor {
    return this.notEqual(other);
  }

  lt(other: Tensor | number): Tensor {
    return this.less(other);
  }

  le(other: Tensor | number): Tensor {
    return this.lessEqual(other);
  }

  gt(other: Tensor | number): Tensor {
    return this.greater(other);
  }

  ge(other: Tensor | number): Tensor {
    return this.greaterEqual(other);
  }

  /**
   * The larger of the two, position by position.
   *
   * **Not `max`.** `max` reduces along an axis and this compares two tensors,
   * which is torch's split and not one invented here. `fmax` below is a third
   * thing again — it skips NaN where this carries it out.
   *
   * **A tie splits the gradient in half**, 0.5 to each side rather than 1 to
   * both. The kernel has done that since it was written and `edge::grad::` asks
   * about it; the forward is equally right either way, so a value comparison
   * cannot see which was meant.
   */
  maximum(other: Tensor | number): Tensor {
    return this.binary("maximum", Tensor.asTensor(other));
  }

  /** The smaller of the two, position by position. `maximum`'s twin. */
  minimum(other: Tensor | number): Tensor {
    return this.binary("minimum", Tensor.asTensor(other));
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
    // **`x.where(condition, other)` gives `x` where the condition is true.** This is
    // the second time in this session it was written the wrong way round — `nanToNum`
    // was the first, and the golden said so immediately.
    const picked = this.binary(better, other);
    const out = other.where(this.unary("isnan"), picked);   // b where a is NaN
    return this.where(other.unary("isnan"), out);           // a where b is NaN
  }

  /**
   * Moves an axis. The same as `movedim`, and torch offers both.
   */
  moveaxis(source: number, destination: number): Tensor {
    return this.movedim(source, destination);
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



  // ── Five more names over work that was already here ───────────────────
  //
  // Fourth time in one session that a name read as absent while the operation sat
  // beside it under a spelling borch.ts chose: `default_collate` was `stackItems`,
  // `avg_pool1d` was `poolND("avg", …)`, sixteen unaries were declared nowhere, and
  // these are `powScalar`, `broadcastTo`, `reshape`, `view` and `size`.

  /**
   * `torch.Tensor.pow`. **The exponent may be a tensor**, which is why this is not
   * simply `powScalar` renamed — that one takes a number, and the `pow` kernel next
   * to it takes the elementwise case with its own two-sided backward.
   */
  pow(exponent: Tensor | number): Tensor {
    return typeof exponent === "number"
      ? this.powScalar(exponent)
      : this.binary("pow", exponent);
  }

  /** How many elements. `torch.Tensor.numel`. */
  numel(): number {
    return this.size;
  }

  /** This, broadcast to `other`'s shape. `torch.Tensor.expand_as`. */
  expandAs(other: Tensor): Tensor {
    return this.broadcastTo(other.shape);
  }

  /** This, reshaped to `other`'s shape. `torch.Tensor.reshape_as`. */
  reshapeAs(other: Tensor): Tensor {
    return this.reshape(other.shape);
  }

  /**
   * This, viewed as `other`'s shape. `torch.Tensor.view_as`.
   *
   * **The name was taken.** A private helper that builds a strided view held
   * `viewAs`, which is a public torch name meaning something else entirely — so
   * `x.viewAs(y)` was either a type error or, from inside the class, a call into
   * the strider with an argument it does not take. It is `stridedView` now.
   */
  viewAs(other: Tensor): Tensor {
    return this.view(...other.shape);
  }

  // ── Six binary names the core has and this did not ────────────────────
  //
  // **The kernels were already here and only the method names were missing** —
  // `binary("hypot", …)` and its five neighbours are in `kernels.ts` with analytic
  // backwards, and the golden has been asking them through the module-level
  // functions the whole time. That is the third time today: `default_collate` was
  // `stackItems`, `avgPool1d` was `poolND("avg", …)`, and these were kernels
  // without a name on `Tensor`.
  //
  // **The first version of this block computed them by hand**, from the formulas,
  // and would have shipped six duplicate implementations with autograd-composed
  // gradients beside six analytic ones. Worse, it would have parted from the golden
  // at exactly the points the kernels are careful about — `logaddexp` written
  // literally overflows in f32 the moment an input passes 89, and `xlogy` written
  // literally gives NaN at `x = 0, y = 0` where the whole purpose of the name is
  // that it gives 0. **Not knowing the kernel was there is what would have caused
  // it**, which is the same not-knowing this axis exists to end.

  /** `sqrt(x² + y²)`. */
  hypot(other: Tensor): Tensor {
    return this.binary("hypot", other);
  }

  /** `x · 2^y`. `torch.Tensor.ldexp`. */
  ldexp(other: Tensor): Tensor {
    return this.binary("ldexp", other);
  }

  /** `log(exp(a) + exp(b))`, in the stable form — see the kernel. */
  logaddexp(other: Tensor): Tensor {
    return this.binary("logaddexp", other);
  }

  /** `log₂(2^a + 2^b)`. */
  logaddexp2(other: Tensor): Tensor {
    return this.binary("logaddexp2", other);
  }

  /** This magnitude with **that** sign. */
  copysign(other: Tensor): Tensor {
    return this.binary("copysign", other);
  }

  /** `0` below zero, `values` **at** zero, `1` above. */
  heaviside(values: Tensor): Tensor {
    return this.binary("heaviside", values);
  }

  /** `x·log(y)`, and **0 wherever `x` is 0** — including where `log(y)` is −∞. */
  xlogy(other: Tensor): Tensor {
    return this.binary("xlogy", other);
  }

  // ── The thirty-eight in-place forms ───────────────────────────────────
  //
  // torch gives almost every operation an underscore partner. What was here was `i0_`
  // alone, and the other thirty-eight are places where **the computation existed and
  // only the underscore name was missing.** `mutate` keeps the in-place-ness, so they
  // are one line each.
  //
  // Ten are torch's **second spellings** (`divide_` = `div_`). Porting them asks the
  // same question twice, and without the name, code written with that spelling simply
  // does not run — an overlapping question and a missing name are different problems.

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

  // **These four take their partner's word for the argument, and did not.** One line
  // up, `clampMax(max)`, `clampMin(min)`, `fmod(other)` and `remainder(other)` all use
  // torch's name; the in-place twins had invented `high`, `low` and `divisor`. Nothing
  // a caller writes changes — JavaScript has no keyword arguments — but the printed
  // API disagreed with torch's docs at four places, and the signature axis could not
  // say so while the core's own methods were `(self, *args, **kw)`.
  clampMax_(max: number): Tensor {
    return this.mutate(() => this.clampMax(max));
  }

  clampMin_(min: number): Tensor {
    return this.mutate(() => this.clampMin(min));
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

  fmod_(other: number): Tensor {
    return this.mutate(() => this.fmod(other));
  }

  remainder_(other: Tensor | number): Tensor {
    return this.mutate(() => this.remainder(other));
  }

  lerp_(end: Tensor, weight: Tensor | number): Tensor {
    return this.mutate(() => this.lerp(end, weight));
  }

  // The in-place forms of the four dropouts. **With `training` false it is the
  // identity**, so with `p` at 0 the values are deterministic and the golden can freeze
  // them — the draws themselves cannot be frozen, and this place can.
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

  // The in-place forms of the five activations. They are **underscore names that live
  // on the `F` side only**, like `F.relu_`; every base was here and the underscore side
  // was not.
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
    // **`alignCorners` is on** — that is torch's default for `upsample_bilinear`, and
    // off it gives different values at the same scale.
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
  bernoulli_(p = 0.5, generator?: null): Tensor {
    refuseGenerator("bernoulli_", generator);
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

  // torch's second spellings. The computation is above and this only joins the names.
  //
  // **Numbers only** — as their partners `div_`, `mul_` and `sub_` are. In-place
  // arithmetic taking a tensor is not here, and an alias wider than its partner is not
  // an alias but a new promise.
  // **An alias narrower than what it aliases drops an argument in silence.** `div_`
  // takes `roundingMode` and `divide_` did not, so `x.divide_(2, "floor")` handed the
  // mode to nothing and returned the true quotient — a number, and a plausible one.
  divide_(other: number, roundingMode: "trunc" | "floor" | null = null): Tensor {
    return this.div_(other, roundingMode);
  }

  trueDivide_(other: number): Tensor {
    // **No `roundingMode` here, and that is torch's own line.** `true_divide` is
    // always the true quotient; `divide` is `div`'s alias and carries the mode.
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

  // ── Sorting ───────────────────────────────────────────────────────────

  /**
   * Sorts one axis and produces the values and the positions.
   *
   * **The gradient follows the values.** It flows only to the positions
   * taken and is zero elsewhere, and returning the values detached means no
   * gradient reaches those positions and the whole classification loss
   * becomes non-differentiable. The core went through this with `topk` and
   * `sort`, and the sister project was in the same state until review.
   */
  sort(dim = 0, descending = false, stable = false):
    { values: Tensor; indices: Tensor } {
    // **`stable` asks for what the kernel already does.** `sortAxis` keeps equal
    // values in their original order, so the flag has nothing to switch on — it is
    // taken because the seat is torch's, and a call that reaches it must land on
    // this parameter rather than on the one after.
    void stable;
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
  argsort(dim = 0, descending = false, stable = false): Tensor {
    return this.sort(dim, descending, stable).indices;
  }

  /**
   * The largest `k`. The front of `sort` — torch gives them descending too.
   */
  topk(k: number, dim = 0, largest = true, sorted = true):
    { values: Tensor; indices: Tensor } {
    // **`largest` was missing**, so `topk(k, dim, false)` — torch's way of asking
    // for the smallest k — was a type error rather than the bottom of the sort.
    // `sorted` promises nothing about the order when false, so answering sorted
    // either way is within what torch allows.
    void sorted;
    const rank = this.shape.length;
    const axis = dim < 0 ? dim + rank : dim;
    const ordered = this.sort(axis, largest);
    return {
      values: ordered.values.narrow(axis, 0, k),
      indices: ordered.indices.narrow(axis, 0, k),
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
      // A row containing a NaN is wholly NaN. A row is one when `isnan` sums above 0.
      const sick = axis === undefined
        ? this.flat().unary("isnan").sum()
        : this.unary("isnan").sumDim(axis);
      const bad = sick.binary("gt", Tensor.full([], 0));
      // **It must not blend arithmetically.** `0 * NaN` is NaN, so even a healthy row
      // becomes NaN — three existing `median` cases went red that way. It has to
      // select.
      const nan = Tensor.zeros(got.values.shape).add(Tensor.full([], Number.NaN));
      return { values: nan.where(bad, got.values), indices: got.indices };
    };
    if (dim === undefined) {
      const flat = this.flat();
      const k = Math.floor((flat.size + 1) / 2);
      const got = spoil(flat.kthvalue(k, 0));
      // **It hands over no index, so it divides evenly across cells of equal value**
      // (measured: `[3,5,5,1,5]`'s gradient is ⅓ on each of the three 5s). `kthvalue`
      // flows to the one chosen position, and that is the rule for operations that
      // **do hand over an index** — this one does not produce it.
      return {
        values: flat.spreadEqual(got.values.detach()),
        indices: got.indices,
      };
    }
    const rank = this.shape.length;
    const axis = dim < 0 ? dim + rank : dim;
    const len = this.shape[axis] ?? 1;
    // **Overwriting the NaNs comes first.** Passing `keepdim` down to `kthvalue`
    // leaves the `sick` that `spoil` builds in the folded shape, so `where` disagrees on
    // rank. It is fixed while folded and the axis is restored at the end.
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
   * Attaches **a backward through the position table** to a value buffer that has
   * already been computed.
   *
   * The kernel produced the forward. All this does is join the graph, and the backward
   * follows the position table back to the original cells.
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
      // **The graph has to be joined.** Left as a bare `new Tensor`, no gradient
      // reaches `src`, and the symptom was an error far away saying "does not require
      // grad". The opposite of scattering is gathering, so the backward is `gather`.
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
    // An overwritten position is cut from the source — 0 goes there. The same position
    // table serves, so no new kernel is needed.
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

  /** A table of which cells were written. A 1 marks an overwritten position. */
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

  // ── Indexed read and write ────────────────────────────────────────────
  //
  // `as_strided`, `select_scatter`, `slice_scatter`, `diagonal_scatter`, `put` and
  // `index_put` all pass through these two doors (`gatherSpots` and `scatterSpots`).

  /** Uploads the index table to the GPU. The positions follow from the shape alone, so
   *  it produces no gradient. */
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
      // **The accumulating side does not cut the source.** Only the overwriting side
      // sends 0 to those positions.
      (g) => [
        accumulate ? g : g.mul(Tensor.ones(mine).zeroAtSpots(spots)),
        g.gatherSpots(spots, srcShape),
      ],
      gradName,
      this.dtype,
    );
  }

  /** A table that is 0 exactly at the indexed positions. No gradient goes to an
   *  overwritten position. */
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
    // **The base's dtype is the result's dtype.** `zeros` is float32, so left alone,
    // putting int64 in still gives float32 out — a place where the values are right and
    // only the label diverges.
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

  /** Writes into the indexed positions **while combining.** The base of
   *  `scatter_reduce` and `index_reduce`. */
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
    // The index covers one axis and the remaining coordinates stay put — that position
    // table is added to flatten it.
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
    // The row index is spread across the positions and added — which gives the same
    // index table `scatterReduce` uses.
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
        // Positions where the mask is true are cut from the source — only the false
        // ones flow through.
        //
        // **The dtype is turned to float first.** The mask is `bool` and subtraction
        // refuses `bool` — torch too. The values are 0/1 so the arithmetic would work,
        // and it is blocked before that.
        const keep = Tensor.full([], 1).sub(wide.to("float32"));
        return [g.mul(keep), new Tensor(gi, srcShape)];
      },
      "MaskedScatterBackward0",
      this.dtype,
    );
  }

  // ── Names that existed only in the binding ────────────────────────────
  //
  // Everything below is a name that **exists in torch and that the binding
  // (`borch_webgpu`) was assembling in Python.** The golden cases go through the
  // binding, so the table was green, and they were missing only for somebody using
  // borch.ts from TypeScript — `tests/test_binding_fills_in.py` counts that place. The
  // assembly itself was correct, so the arithmetic is carried over unchanged and only
  // **where the name lives** moves. The binding now only forwards.

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
    // **`x.where(condition, other)` gives `x` where the condition is true.** The fill
    // value has to be the receiver — written the other way round, the NaN positions keep
    // their NaN (measured: five golden cases diverged with `max diff inf`).
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
  /**
   * **`equalNan` decides whether NaN matches NaN**, and it was missing while
   * `allclose` twenty lines down has had it from the start. Two functions asking the
   * same question, one of them able to answer it.
   *
   * `allclose` reads the values back and can test with `Number.isNaN`; this one stays
   * on the device, so the same rule is written in tensor operations: a cell where
   * both sides are NaN is `x !== x && y !== y`, and `binary("ne", self)` is that test.
   */
  isclose(other: Tensor, rtol = 1e-5, atol = 1e-8, equalNan = false): Tensor {
    const room = other.abs().mul(Tensor.full([], rtol)).add(Tensor.full([], atol));
    const near = this.sub(other).abs().binary("le", room);
    if (!equalNan) return near;
    const bothNan = this.binary("ne", this).mul(other.binary("ne", other));
    return near.add(bothNan).binary("gt", Tensor.full([], 0));
  }

  // `allclose` is not here — **it is already below and that one is better.** It checks
  // the shape first and takes `equalNan` as well. Written again alongside as a partner,
  // it was caught as a duplicate.

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
  stdMean(
    dim?: number, correction = 1, keepdim = false,
  ): { std: Tensor; mean: Tensor } {
    return {
      std: this.std(dim, correction, keepdim),
      // **The mean has to fold the same axis.** It used to be `this.mean()`
      // regardless, so asking for a per-column standard deviation would have
      // returned it beside the mean of everything — two answers of different
      // rank in one object, and only one of them about the axis asked for.
      mean: this.mean(dim, keepdim),
    };
  }

  varMean(
    dim?: number, correction = 1, keepdim = false,
  ): { variance: Tensor; mean: Tensor } {
    return {
      variance: this.variance(dim, correction, keepdim),
      mean: this.mean(dim, keepdim),
    };
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
   * The Kronecker product, at any rank.
   *
   * **This was 1-D only, and the 1-D line was already the general one.** It read
   * `reshape([n, 1]).mul(other.reshape([1, m]))` and refused everything above one
   * axis — but interleaving the two shapes and letting broadcasting multiply *is*
   * `kron` at every rank:
   *
   *     out[(i, k), (j, l), …] = a[i, j, …] · b[k, l, …]
   *
   * The refusal was written because the binding's version looked at one axis only
   * and was quietly wrong above it, and *a missing feature beats a wrong answer*.
   * That was the right call on the day; what it hid is that the correct answer was
   * one loop away from the line it sat on.
   *
   * **The shorter operand is padded at the front** — numpy's rule and torch's,
   * measured against both across nine rank combinations including 0-D.
   */
  kron(other: Tensor): Tensor {
    const rank = Math.max(this.shape.length, other.shape.length);
    const pad = (s: readonly number[]): number[] =>
      [...new Array<number>(rank - s.length).fill(1), ...s];
    const a = pad(this.shape);
    const b = pad(other.shape);
    const aSpread: number[] = [];
    const bSpread: number[] = [];
    const folded: number[] = [];
    for (let i = 0; i < rank; i++) {
      aSpread.push(a[i] as number, 1);
      bSpread.push(1, b[i] as number);
      folded.push((a[i] as number) * (b[i] as number));
    }
    // Both 0-D: there is nothing to interleave and the product is the product.
    if (rank === 0) return this.mul(other);
    return this.reshape(aSpread).mul(other.reshape(bSpread)).reshape(folded);
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

  /** Spreads a 1-D index into `shape` — the three `index_*` use one door. */
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
  fillDiagonal_(fillValue: number, wrap = false): Tensor {
    const rows = this.shape[0] ?? 0;
    const cols = this.shape[1] ?? 0;
    const step = cols + 1;
    const limit = wrap ? rows * cols : Math.min(rows * cols, cols * cols);
    const spots: number[] = [];
    for (let at = 0; at < limit; at += step) spots.push(at);
    return this.mutate(() => this.reshape([this.size]).scatterSpots(
      Tensor.spotsTensor(Float32Array.from(spots)),
      Tensor.full([spots.length], fillValue), false, "FillDiagonalBackward0",
    ).reshape(this.shape));
  }

  /**
   * Draws afresh at every position and **overwrites.** The original values are not
   * read.
   *
   * Six use one door — all that differs per distribution is what a single uniform draw
   * is turned into. The binding builds these six with numpy, and that is because
   * `get_rng_state` there has to serialise one stream; there is no such reason here.
   */
  private drawInto_(draw: (u: number) => number): Tensor {
    const data = new Float32Array(this.size);
    for (let i = 0; i < data.length; i++) data[i] = draw(uniform());
    return this.mutate(() => Tensor.from(data, this.shape, { dtype: this.dtype }));
  }

  /**
   * **A continuous distribution has no answer in an integer cell.** Five stop here and
   * `geometric_` and `random_` do not — those two are discrete, so an integer cell does
   * hold their answer.
   *
   * Grouping them by name as "draws are floats only" is wrong for those two. This gate
   * was not attached when the first five went in, and without it
   * `zeros(6, int64).exponential_()` **runs quietly** and puts truncated floats into
   * integer cells.
   *
   * **The exception kind differs inside torch itself** (measured): `normal_`, `uniform_`
   * and `log_normal_` raise `NotImplementedError` while `exponential_` and `cauchy_`
   * raise `RuntimeError`. Grouped into one, three of them diverge, and that divergence
   * is an exception name rather than a value, so a check comparing values does not catch
   * it. The caller states which one it is.
   */
  private needsFloatDraw(who: string, kind: "runtime" | "unimplemented"): void {
    if (this.dtype === "float32") return;
    const said = `"${who}" not implemented for '${this.dtype}' — ` +
      "a continuous distribution draws into floating point cells only.";
    throw kind === "runtime"
      ? new RuntimeError(said)
      : new NotImplementedError(said);
  }

  /**
   * The exponential distribution. Its mean is `1/lambd`, and **lambd has to
   * be positive**.
   */
  exponential_(lambd = 1.0, generator?: null): Tensor {
    refuseGenerator("exponential_", generator);
    this.needsFloatDraw("exponential_", "runtime");
    if (!(lambd > 0)) {
      throw new RuntimeError(
        `exponential_ expects lambda > 0.0, but found lambda=${lambd}`);
    }
    // **It uses `1 - u`.** `uniform()` can give 0, and `log(0)` is −∞.
    return this.drawInto_((u) => -Math.log(1 - u) / lambd);
  }

  /**
   * The Cauchy distribution. **It has no mean** — the tails are heavy
   * enough that the sample mean does not converge.
   */
  cauchy_(median = 0.0, sigma = 1.0, generator?: null): Tensor {
    refuseGenerator("cauchy_", generator);
    this.needsFloatDraw("cauchy_", "runtime");
    return this.drawInto_((u) => median + sigma * Math.tan(Math.PI * (u - 0.5)));
  }

  /**
   * The log-normal distribution. `mean` and `std` are the values **after
   * taking logs** (as in torch).
   */
  logNormal_(mean = 1.0, std = 2.0, generator?: null): Tensor {
    refuseGenerator("log_normal_", generator);
    this.needsFloatDraw("log_normal_", "unimplemented");
    return this.drawInto_(() => Math.exp(mean + std * gauss()));
  }

  /**
   * Overwrites with the normal distribution. **`std` cannot be negative** —
   * at 0 it is the mean itself.
   */
  normal_(mean = 0.0, std = 1.0, generator?: null): Tensor {
    refuseGenerator("normal_", generator);
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
  uniform_(from = 0.0, to = 1.0, generator?: null): Tensor {
    refuseGenerator("uniform_", generator);
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
  geometric_(p: number, generator?: null): Tensor {
    refuseGenerator("geometric_", generator);
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
  random_(from = 0, to?: number, generator?: null): Tensor {
    refuseGenerator("random_", generator);
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
    // Sending the measured axis to the front and laying the rest into one row finishes
    // the reduction in a single pass.
    const keep = this.shape[axis] ?? 1;
    const moved = this.movedim(axis, 0).reshape([keep, -1]);
    const norms = moved.abs().powScalar(p).sumDim(1, true).powScalar(1 / p);
    // `gt` is a name that lives in the table alone, so it is not a method — it is
    // called through `binary`.
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
  ger(vec2: Tensor): Tensor {
    return this.outer(vec2);
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

  // ── Convolution and pooling ───────────────────────────────────────────

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

  /**
   * **`avgPool1d` and `avgPool3d` existed as a kernel and not as a name.**
   * `poolND("avg", …)` was doing the work for `avgPool2d` and for the layers at every
   * rank; only torch's two per-rank names were missing, exactly as `maxPool1d` and
   * `maxPool3d` are the same call three lines up.
   */
  avgPool1d(kernel = 2, stride?: number): Tensor {
    return this.poolND("avg", kernel, stride);
  }

  avgPool3d(kernel = 2, stride?: number): Tensor {
    return this.poolND("avg", kernel, stride);
  }

  // ── Pooling that also returns the winning positions ───────────────────
  //
  // Max pooling keeps one per window and discards the rest. **The values do not carry
  // "which cell won"**, so `maxUnpool` cannot go back from values alone. torch has the
  // pooling return the positions alongside and hands them to the unpooling — a common
  // pair in an autoencoder.

  /** The window list for a fixed window. `[start, end)` per axis. */
  private fixedWindows(kernel: number, stride?: number): [number, number][][] {
    const step = stride ?? kernel;
    return this.shape.slice(2).map((n) => {
      const out: [number, number][] = [];
      for (let s = 0; s + kernel <= n; s += step) out.push([s, s + kernel]);
      return out;
    });
  }

  /** The window list for the adaptive form. The start floors and the end ceils — the
   *  length differs per position. */
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

  // ── CTC ───────────────────────────────────────────────────────────────
  //
  // The loss that joins audio to characters **without aligning them.** It sums over
  // every possible alignment, and that count is exponential, so it lays out a state
  // sequence with blanks between the targets and folds forward through it.
  //
  // The `u` axis is pushed in one go. Only time is looped, so the graph is proportional
  // to `T` — slow at real speech lengths (hundreds of frames), and the accurate side was
  // chosen.

  /** "Absent" as a log probability. `-Infinity` becomes NaN in logsumexp, so a large
   *  negative is used. */
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
    // `[l1, l2]` → `[_, l1, _, l2, _]`. **Repeated characters must have a blank
    // between them** — without one, two characters fold into one. That rule is `skip`.
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
   * The window start positions. **ATen's `generate_intervals`, verbatim.**
   *
   * With `α = (input - window) / (output - 1)`, it is `floor((i+u)·α) - floor(u·α)`.
   * Only the last window is pushed against the right end, so the input's final cell is
   * certain to be covered.
   *
   * When it divides evenly `α` is an integer and `u` does nothing — asked as 6→3, the
   * random part is invisible in its entirety.
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
    dilation: number | readonly number[] = 1,
    groups = 1,
  ): Tensor {
    const spatial = this.shape.length - 2;
    if (groups !== 1) {
      // **Groups by slicing and joining, not inside the kernel.** The gradient
      // then follows from the pieces — `narrow` and `cat` carry theirs — where a
      // grouped path through the shader would be a second index arithmetic that
      // has to agree with the first, and the three conv shaders each carry one
      // already.
      const inCh = this.shape[1] ?? 1;
      const outCh = weight.shape[0] ?? 1;
      if (inCh % groups !== 0 || outCh % groups !== 0) {
        throw new RuntimeError(
          `groups=${groups} divides neither the input channels (${inCh}) nor the `
          + `filters (${outCh})`);
      }
      const cin = inCh / groups;
      const cout = outCh / groups;
      const parts: Tensor[] = [];
      for (let g = 0; g < groups; g++) {
        parts.push(this.narrow(1, g * cin, cin).convND(
          weight.narrow(0, g * cout, cout),
          bias === null ? null : bias.narrow(0, g * cout, cout),
          stride, padding, dilation));
      }
      return Tensor.cat(parts, 1);
    }
    if (spatial < 1 || weight.shape.length !== this.shape.length) {
      throw new Error(`conv: shapes do not match: [${this.shape}] x [${weight.shape}]`);
    }
    const spread = (v: number | readonly number[]): number[] =>
      typeof v === "number" ? new Array<number>(spatial).fill(v) : [...v];
    const inDims = this.shape.slice(2);
    const kernel = weight.shape.slice(2);
    const st = spread(stride);
    const pd = spread(padding);
    const dl = spread(dilation);
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
      inDims, kernel, stride: st, pad: pd, dilation: dl,
      outDims: inDims.map((d, i) =>
        convOut(d, pd[i] ?? 0, kernel[i] ?? 1, st[i] ?? 1, dl[i] ?? 1)),
    };
    const key = convNDKey(s);
    const outShape = [s.N, s.O, ...s.outDims];
    const n = outShape.reduce((a, b) => a * b, 1);
    const out = dev().alloc(n);
    // It uses the tiled version. The shader is longer than the simple one and costs one
    // more compilation, and it is cached by shape signature, so that happens once while
    // what runs every step is the kernel.
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
          // A split reduction leaves one partial sum per piece, and they have to be
          // added once more.
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
          // The batch and the output positions summed together. Stacking reductions
          // needs no new kernel.
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
    outputPadding: number | readonly number[] = 0,
    groups = 1,
    dilation: number | readonly number[] = 1,
  ): Tensor {
    const spatial = this.shape.length - 2;
    if (groups !== 1) {
      // **The weight axes are the other way round here** — `(in, out/groups, …)` —
      // so the slice that walks the groups runs down axis 0 for the input side and
      // the bias is cut by axis 1. Written apart from the convolution's for that
      // reason: with a square kernel the reversed weight fits, and it diverges only
      // in the values.
      const inCh = this.shape[1] ?? 1;
      const perGroupOut = weight.shape[1] ?? 1;
      if (inCh % groups !== 0) {
        throw new RuntimeError(
          `groups=${groups} does not divide the input channels (${inCh})`);
      }
      const cin = inCh / groups;
      const parts: Tensor[] = [];
      for (let g = 0; g < groups; g++) {
        parts.push(this.narrow(1, g * cin, cin).convTransposeND(
          weight.narrow(0, g * cin, cin),
          bias === null ? null : bias.narrow(0, g * perGroupOut, perGroupOut),
          stride, padding, outputPadding, 1, dilation));
      }
      return Tensor.cat(parts, 1);
    }
    if (spatial < 1 || weight.shape.length !== this.shape.length) {
      throw new Error(
        `convTranspose: shapes do not match: [${this.shape}] x [${weight.shape}]`);
    }
    const spread = (v: number | readonly number[]): number[] =>
      typeof v === "number" ? new Array<number>(spatial).fill(v) : [...v];
    const st = spread(stride);
    const pd = spread(padding);
    const op = spread(outputPadding);
    const dl = spread(dilation);
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
    // Seen through an ordinary convolution's eyes: our input is its **output** and our
    // output is its input. So O and C go into swapped slots.
    // **`outputPadding` is expressed as a longer output and nothing else.** The
    // shader finds, for each output cell, the input cells that reach it; ask it for
    // more cells and it answers by the same rule — which is what torch's extra rows
    // are. They are **not zeros**: they reach back into the part the padding trim
    // was about to throw away, and only past the untrimmed end is there nothing to
    // find, where the shader's guard already gives 0. Writing zeros instead matches
    // the shape exactly and differs in the values (measured on the core side, on
    // twelve of fifty-six configurations, every one of them padding and
    // outputPadding together).
    const outDims = ourDims.map((d, i) =>
      (d - 1) * (st[i] ?? 1) + ((kernel[i] ?? 1) - 1) * (dl[i] ?? 1) + 1
      - 2 * (pd[i] ?? 0) + (op[i] ?? 0));
    const s: ConvNDShape = {
      N: this.shape[0] ?? 1, C: Cout, O: Cin,
      inDims: outDims, kernel, stride: st, pad: pd, dilation: dl, outDims: ourDims,
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
          // The gradient on our input side is **an ordinary convolution's forward**.
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
          // The two arguments are swapped — its "input" is our gradient and its
          // "output gradient" is our input.
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
   * `torch.Tensor.stft`. **torch has both spellings and this had only the free one**,
   * so `x.stft(8, …)` — how torch's own tutorials write it — was a `TypeError`. The
   * name axis counted the free function as satisfying the method and reported no gap.
   *
   * The receiver is the first argument, **measured**: `sig.stft(8)` equals
   * `torch.stft(sig, 8)`. It is measured because the receiver's position is not a rule
   * — `polygamma`'s is the second, and the core has a note about copying that pair the
   * wrong way round out of a table.
   */
  stft(nFft: number, options: StftOptions = {}): Tensor {
    return stftImpl(this, nFft, options);
  }

  /** `torch.Tensor.istft`. The receiver is the first argument, measured. */
  istft(nFft: number, options: IstftOptions = {}): Tensor {
    return istftImpl(this, nFft, options);
  }

  /** `torch.Tensor.polygamma`. **The receiver is the second argument** —
   *  `x.polygamma(1)` is `torch.polygamma(1, x)`. */
  polygamma(n: number): Tensor {
    return polygammaImpl(n, this);
  }

  /** `torch.Tensor.igamma`. The receiver is the first. */
  igamma(other: Tensor): Tensor {
    return igammaImpl(this, other);
  }

  /** `torch.Tensor.igammac`. The receiver is the first. */
  igammac(other: Tensor): Tensor {
    return igammacImpl(this, other);
  }

  /**
   * Pooling independent of dimensionality.
   */
  poolND(kind: "max" | "avg", kernel: number, stride?: number,
         padding = 0, ceilMode = false, countIncludePad = true,
         divisorOverride: number | null = null): Tensor {
    const spatial = this.shape.length - 2;
    if (spatial < 1) throw new Error(`pooling: the shape does not match: [${this.shape}]`);
    // **The maximum used to refuse what only the average implements**, on the ground
    // that its backward reads the input at each window position and a padded position
    // has none to read. The average had the answer one function away the whole time:
    // take the padding off the coordinate and skip what falls outside. The maximum
    // needed that guard and a starting value below every real one, and both kernels
    // now carry it.
    const step = stride ?? kernel;
    const inDims = this.shape.slice(2);
    const p: PoolNDShape = {
      // The channels are folded into the batch — pooling runs per plane.
      NC: (this.shape[0] ?? 1) * (this.shape[1] ?? 1),
      inDims,
      kernel: new Array<number>(spatial).fill(kernel),
      stride: new Array<number>(spatial).fill(step),
      outDims: inDims.map((d) => poolOut(d, padding, kernel, step, ceilMode)),
      pad: new Array<number>(spatial).fill(padding),
      countIncludePad,
      divisorOverride,
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
        // Average pooling does not look at the input, so it takes no buffer either —
        // passing an unused binding makes WebGPU invalidate the whole command buffer,
        // and then the backward quietly does not run.
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
  /**
   * **A fractional `scaleFactor` has to be floored, and was not.**
   *
   * torch's output extent is `floor(in · scale)`. Multiplying without it gave
   * `Upsample(scale_factor=1.5)` on a 3×4 input a shape of `[1, 1, 4.5, 6]` — a
   * non-integer dimension, which for `nearest` produced a tensor summing to zero and
   * *did not throw*, and for `bilinear` blew up somewhere else entirely (`shape
   * [4.5,1] does not match 5 elements`). At 2.3 it reached WebGPU as `Size (253) must
   * be a multiple of 4`.
   *
   * Whole factors were right, which is why this stood: 2 and 3 are what anybody tries.
   *
   * `recomputeScaleFactor` is torch's fourth answer to the same arithmetic. With it
   * on, torch works out the integer output size and then **derives the scale back
   * from it** per axis, so a 3-high input at 1.5 samples at 4/3 rather than 1.5. Same
   * shape, different values — measured on torch: sum 120 against 132.
   */
  interpolate(size: number | readonly number[] | null = null,
              scaleFactor: number | null = null,
              mode: InterpolateMode = "nearest",
              alignCorners = false,
              recomputeScaleFactor: boolean | null = null,
              antialias = false): Tensor {
    const h = this.shape[2] ?? 1;
    const w = this.shape[3] ?? 1;
    const pair = (v: number | readonly number[]): [number, number] =>
      typeof v === "number" ? [v, v] : [v[0] ?? 1, v[1] ?? v[0] ?? 1];
    const scale = scaleFactor ?? 2;
    // torch's own rule, in one place so the two modes cannot part.
    const extent = (inp: number) => Math.floor(inp * scale);
    if (antialias) {
      if (mode !== "bilinear" && mode !== "bicubic") {
        throw new RuntimeError(
          "Anti-alias option is restricted to bilinear, bicubic, and lanczos modes "
          + "and requires a 4-D tensor as input");
      }
      const [oh, ow] = size === null ? [extent(h), extent(w)] : pair(size);
      const given = size === null && !recomputeScaleFactor ? scale : null;
      return this.interpolateAntialias(oh, ow, mode, alignCorners, given);
    }
    if (mode === "nearest") {
      // `resizeNearest` maps `src = floor(o · in / out)`, which *is* recomputing the
      // scale from the output size — so nearest gives the same answer either way and
      // torch says as much by ignoring the flag outside the fractional bilinear case.
      const [oh, ow] = size === null ? [extent(h), extent(w)] : pair(size);
      return this.resizeNearest([oh, ow]);
    }
    if (mode === "nearest-exact") {
      const [oh, ow] = size === null ? [extent(h), extent(w)] : pair(size);
      return this.resizeNearestExact(oh, ow);
    }
    // **`area` is `adaptivePool("avg", …)` and nothing else.** Measured against torch
    // on a shrink, an enlargement and a size dividing evenly into neither, the two
    // agree bit for bit — so this names what is here rather than writing a second
    // averaging, and the gradient comes with it.
    if (mode === "area") {
      const [oh, ow] = size === null ? [extent(h), extent(w)] : pair(size);
      return this.adaptivePool("avg", [oh, ow]);
    }
    const [oh, ow] = size === null ? [extent(h), extent(w)] : pair(size);
    if (mode === "bicubic") {
      const given = size === null && !recomputeScaleFactor ? scale : null;
      return this.interpolateBicubic(oh, ow, alignCorners, given);
    }
    // **The union protects TypeScript callers and not the binding.** `mode` reaching
    // here is a string, and Python hands one over without a type to stop it — so
    // everything unmatched fell through to bilinear and `mode="quadratic"` came back
    // as a bilinear answer under its name. The golden case asking that both sides stop
    // is what found it: `tsc` cannot, because in TypeScript the call does not compile.
    if (mode !== "bilinear") {
      throw new RuntimeError(
        `interpolate(mode=${JSON.stringify(mode)}) is not one of nearest, `
        + "nearest-exact, area, bilinear, bicubic");
    }
    // **The default is the *given* scale; recomputing is what the flag turns on.**
    // The kernel had only the recomputed rule (`in / out`), so borch.ts answered as
    // though the flag were always set. A size rather than a factor has no scale to
    // recompute from, so that path is unchanged.
    const given = size === null && !recomputeScaleFactor ? scale : null;
    return this.interpolateBilinear(oh, ow, alignCorners, given);
  }

  /**
   * `antialias=true` for `bilinear` and `bicubic` — the widened filter torch uses
   * when shrinking.
   *
   * The window is **widened by the shrink factor and the weights renormalised**, which
   * is the whole of what the flag means. Enlarging, the scale is below one, the
   * support stays at the kernel's own radius, and the weights are the plain ones —
   * which is why torch says the flag does nothing going up.
   *
   * **Two things here are torch disagreeing with itself, both measured rather than
   * reasoned.** The cubic constant is `a = −0.5` where plain `bicubic` uses `−0.75`;
   * fitted against torch, `−0.75` parts by 0.13 to 0.39 on a 4×5 and `−0.5` agrees to
   * noise. And `alignCorners` is half applied: the scale becomes `(in−1)/(out−1)`,
   * which is the align-corners rule, while the centre stays `scale·(i + 0.5)`, which
   * is the other one — taking the align-corners centre parts by 1.3 to 4.5.
   *
   * Each axis is a matrix multiply against a `(out, in)` weight matrix, so the
   * gradient is the multiply's own.
   */
  private interpolateAntialias(outH: number, outW: number,
                               mode: "bilinear" | "bicubic",
                               alignCorners: boolean,
                               given: number | null): Tensor {
    const radius = mode === "bilinear" ? 1 : 2;
    const filt = mode === "bilinear"
      ? (raw: number): number => {
        const t = Math.abs(raw);
        return t < 1 ? 1 - t : 0;
      }
      : (raw: number): number => {
        const t = Math.abs(raw);
        const a = -0.5;
        if (t <= 1) return ((a + 2) * t - (a + 3)) * t * t + 1;
        if (t < 2) return ((t - 5) * t + 8) * t * a - 4 * a;
        return 0;
      };
    const axis = (sizeIn: number, sizeOut: number): Tensor => {
      // The caller's scale is not read under `alignCorners` — measured: with it on,
      // the `scaleFactor` cases agree with `(in−1)/(out−1)` alone.
      const s = alignCorners
        ? (sizeOut > 1 ? (sizeIn - 1) / (sizeOut - 1) : 0)
        : (given === null ? sizeIn / sizeOut : 1 / given);
      const wide = s >= 1;
      const support = wide ? radius * s : radius;
      const inv = wide ? 1 / s : 1;
      const rows = new Float32Array(sizeOut * sizeIn);
      for (let i = 0; i < sizeOut; i++) {
        const centre = s * (i + 0.5);
        const lo = Math.max(Math.trunc(centre - support + 0.5), 0);
        const span = Math.min(Math.trunc(centre + support + 0.5), sizeIn) - lo;
        const taps = Array.from({ length: span },
                                (_, j) => filt((j + lo - centre + 0.5) * inv));
        const total = taps.reduce((a, b) => a + b, 0);
        for (let j = 0; j < span; j++) {
          rows[i * sizeIn + lo + j] = total ? (taps[j] as number) / total : 0;
        }
      }
      return Tensor.from(rows, [sizeOut, sizeIn]);
    };
    // `(o, h) · (n, c, h, w) · (w, p)` — the rows fold first, then the columns.
    return axis(this.shape[2] ?? 1, outH).matmul(this)
      .matmul(axis(this.shape[3] ?? 1, outW).transpose());
  }

  /**
   * `nearest-exact` — nearest measured from the **centre** of the output cell.
   *
   * `floor((o + 0.5)·in/out)` against `resizeNearest`'s `floor(o·in/out)`. Half a
   * cell, which is nothing when enlarging by a whole number and is a different row
   * entirely when shrinking: on a 4×5 to 2×3, `nearest` takes rows 0 and 2 and this
   * takes 1 and 3. torch keeps both because the plain one is what everybody else had
   * already shipped and is off by half.
   *
   * Built from `indexSelect` rather than a shader. The index rows are constants, so
   * the gradient is the gather's own — an output cell reading the same input cell
   * several times accumulates there, which is what `resizeNearest`'s kernel does by
   * hand.
   */
  private resizeNearestExact(outH: number, outW: number): Tensor {
    const h = this.shape[2] ?? 1;
    const w = this.shape[3] ?? 1;
    const pick = (sizeIn: number, sizeOut: number): Tensor => Tensor.from(
      Array.from({ length: sizeOut }, (_, i) =>
        Math.min(Math.floor(((i + 0.5) * sizeIn) / sizeOut), sizeIn - 1)),
      [sizeOut], { dtype: "int64" });
    return this.indexSelect(2, pick(h, outH)).indexSelect(3, pick(w, outW));
  }

  /**
   * Cubic convolution over a 4×4 neighbourhood — torch's `bicubic`.
   *
   * **`a = −0.75`**, which is a choice rather than a derivation: the family of cubic
   * kernels is parameterised by it, OpenCV uses −0.75 and Photoshop −0.5, and the
   * edges come out visibly different. torch uses −0.75 and that is the whole reason
   * this one does.
   *
   * The sixteen taps are gathers with constant indices times constant weights, so the
   * gradient is the graph's and nothing had to be written for it. **The edges clamp**,
   * which is torch's rule and what makes a window one cell outside the image safe.
   * Nothing is clamped on the way out — cubic convolution overshoots, so an image in
   * `[0, 1]` comes back slightly outside it, and torch does not clamp either.
   */
  private interpolateBicubic(outH: number, outW: number, alignCorners: boolean,
                             given: number | null): Tensor {
    const h = this.shape[2] ?? 1;
    const w = this.shape[3] ?? 1;
    const A = -0.75;
    const cubic = (raw: number): number => {
      const t = Math.abs(raw);
      if (t <= 1) return ((A + 2) * t - (A + 3)) * t * t + 1;
      if (t < 2) return ((t - 5) * t + 8) * t * A - 4 * A;
      return 0;
    };
    // The same two rules the bilinear path uses: `alignCorners` pins both ends,
    // otherwise the coordinate is measured from the centre of the output cell.
    const coords = (sizeIn: number, sizeOut: number): number[] => {
      if (alignCorners) {
        if (sizeOut === 1) return [0];
        return Array.from({ length: sizeOut },
                          (_, i) => (i * (sizeIn - 1)) / (sizeOut - 1));
      }
      const step = given === null ? sizeIn / sizeOut : 1 / given;
      return Array.from({ length: sizeOut }, (_, i) => (i + 0.5) * step - 0.5);
    };
    const ys = coords(h, outH);
    const xs = coords(w, outW);
    const clampRow = (base: number[], k: number, limit: number): Tensor =>
      Tensor.from(base.map((v) =>
        Math.min(Math.max(Math.floor(v) + k, 0), limit - 1)), [base.length],
      { dtype: "int64" });

    let out: Tensor | null = null;
    for (let ky = -1; ky < 3; ky++) {
      const wy = Tensor.from(
        ys.map((v) => cubic(ky - (v - Math.floor(v)))), [1, 1, outH, 1]);
      const rows = this.indexSelect(2, clampRow(ys, ky, h));
      for (let kx = -1; kx < 3; kx++) {
        const wx = Tensor.from(
          xs.map((v) => cubic(kx - (v - Math.floor(v)))), [1, 1, 1, outW]);
        const tap = rows.indexSelect(3, clampRow(xs, kx, w)).mul(wy).mul(wx);
        out = out === null ? tap : out.add(tap);
      }
    }
    return out ?? this;
  }

  /**
   * Nearest-neighbour enlargement. `Upsample` and `interpolate` are this.
   */
  upsample(scale: number): Tensor {
    return this.resizeNearest(this.shape.slice(2).map((d) => d * scale));
  }

  /**
   * Nearest resizing to given extents. `upsample(scale)` is this with the extents
   * worked out from a whole multiple, and **a target that is not a multiple is the
   * reason it takes extents at all** — `UpsamplingNearest2d(size=(3, 5))` from a 2×2
   * is an answer torch and the core both give and this used to refuse.
   */
  resizeNearest(outDims: readonly number[]): Tensor {
    const inDims = this.shape.slice(2);
    const NC = (this.shape[0] ?? 1) * (this.shape[1] ?? 1);
    const outShape = [this.shape[0] ?? 1, this.shape[1] ?? 1, ...outDims];
    const n = outShape.reduce((a, b) => a * b, 1);
    const key = `${NC}:${inDims}:${outDims}`;
    const out = dev().alloc(n);
    dev().run1d(
      dev().pipeline(`up:${key}`, () => upsampleNearest(NC, inDims, outDims)),
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
        dev().run1d(
          dev().pipeline(`upb:${key}`,
            () => upsampleNearestBackward(NC, inDims, outDims)),
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
  lpPool(normType: number, kernel: number, stride?: number, ceilMode = false): Tensor {
    // **The mean times the kernel volume, which is not the sum at a short window —
    // and torch is the one that wants it that way.** `lp_pool1d` is written as
    // `avg_pool1d(x^p, …).mul(kernel_size)`, so under `ceilMode` the average divides
    // an edge window by what it covered and the multiply puts back the whole volume.
    // The difference is exactly the windows the ceiling adds.
    //
    // This was rewritten to divide by one and take the true sum, on the grounds that
    // Lᵖ pooling is defined as a sum — true of the definition, not true of the
    // function being matched, and the two only agree while every window is full. The
    // two `LPPool` ceiling cases diverged and nothing else did. **Reading torch's own
    // source is what settled it**, and it should have come before the edit.
    const spatial = this.shape.length - 2;
    const count = kernel ** spatial;
    const powered = this.powScalar(normType);
    const out = powered.poolND("avg", kernel, stride ?? kernel, 0, ceilMode);
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
   * Normalises `(N, C, H, W)` per channel. Training mode — the statistics come from
   * this batch.
   *
   * Three axes have to fold at once, so `layerNorm` cannot be used. Stacking reductions
   * needs no new kernel — it makes a few intermediate tensors instead, and that is the
   * price being paid.
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
    // The backward uses the standardised values again. They are built once here and
    // carried — recomputing them in the backward costs two more kernels.
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
        // The weight and bias gradients are two sums that were already counted — no
        // new kernel.
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
    // The variance is the biased estimate (divided by n) — as torch's BatchNorm is.
    const varc = perChannel(centered.square()).div(Tensor.full([], count));
    return centered.div(varc.binary("add", Tensor.full([], eps)).sqrt());
  }

  // ── Complex ───────────────────────────────────────────────────────────
  //
  // The storage is **interleaved** — one buffer of `[re, im, re, im, …]`. That is what
  // makes `viewAsReal` and `viewAsComplex` **real views** (they keep the buffer and
  // change only the label and the shape). They are views in torch too. Holding the real
  // and imaginary parts as two tensors would have made a copy here, and `Tensor` would
  // have had to hold two buffers, so the lifetime and device paths would all have had to
  // learn about it. In exchange for keeping the one-buffer invariant, only the **length**
  // doubles.
  //
  // ## The gradient convention (pinned down by measurement)
  //
  // torch refuses `backward()` on a complex loss. Because a loss is always real,
  //
  //     z.grad = ∂L/∂re + i·∂L/∂im
  //
  // is well defined, and on top of that **a conjugate appears in the backward of a
  // holomorphic function** — `mul` and `div` are those places. `abs` produces a real and
  // is not holomorphic, so it gets none (`z/|z|`), and `conj` itself is `conj(g)`. Three
  // different rules, and with real inputs none of the three can be told apart — because
  // conjugation is the identity on reals.

  /**
   * Whether this tensor is complex. Where `torch.is_complex` goes.
   */
  isComplex(): boolean {
    return isComplexDType(this.dtype);
  }

  /** One complex kernel. It takes the inputs and produces a buffer of `outFloats`
   *  cells. */
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

  /** Demands a complex. Given a real, it says which name was wrong. */
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
    // **`real` and `imag` follow different rules.** Given a real tensor, torch returns
    // it unchanged from `real` (its documentation says so) and refuses `imag` —
    // "imag is not implemented for tensors with non-complex dtypes".
    //
    // Here both were refused. Two names standing side by side were grouped under one
    // rule while torch keeps them apart, so code using `torch.real(x)` stopped on a real
    // tensor. The dtype has to pass through too — int64 in, int64 out.
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

  /** Lifts a real to `x + 0i`. The backward takes the real part back out. */
  private asComplexRe(): Tensor {
    const n = this.size;
    const out = Tensor.cRun(`cfromre:${n}`, () => complexFromReal(n),
      [this.buffer], 2 * n, n);
    return Tensor.make(
      out, this.shape, [this], (g) => [g.real()],
      "ToComplexBackward0", "complex64",
    );
  }

  /** Lifts a real to `0 + xi`. */
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
   * Flips the sign. **It does not multiply by a scalar −1** — a complex binary needs
   * matching shapes so a scalar cannot come in, and building a −1 tensor of that shape
   * costs another buffer.
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
   * A complex binary. **The shapes have to match** — there is no broadcasting yet.
   *
   * A real on one side is lifted to `x + 0i` and let in. Using the real kernel directly
   * is possible (addition simply works over the flat 2n cells) and multiplication does
   * not, and a rule where some operations work and others do not is a good rule for the
   * next person to get wrong.
   */
  private complexBinary(name: "add" | "sub" | "mul" | "div", other: Tensor): Tensor {
    // **Only the real side has its shape matched** — expanding **before** lifting to
    // complex borrows the real side's broadcasting (`expand`) as it is. Expanding after
    // lifting would need a separate `expand` that understands the interleaving. One
    // ordering saves one kernel.
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
      // Two complex operands of different shapes. That does not exist here yet — no
      // value is invented.
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
        // **The conjugate belongs here.** `d(ab)/da` is not `b` but `conj(b)`, and
        // division is the same place. Tested with reals alone, there is no telling
        // whether this line exists.
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

  // ── Backward ──────────────────────────────────────────────────────────

  /**
   * `torch.Tensor.retain_grad` — **keep this one's gradient even though it is
   * derived.**
   *
   * A backward pass writes to leaves; everything in between is used and dropped,
   * because the intermediates are the bulk of a training loop's memory. This says
   * *keep that one*, which is how the gradient at a hidden activation is looked at.
   *
   * It was the same missing mechanism as `backward(…, inputs)`: both need a
   * derived node to be able to hold a `grad`, and neither could. Closing one
   * closed the other.
   *
   * On a leaf it is a no-op — a leaf already accumulates. torch is the same.
   */
  retainGrad(): void {
    if (!this.requiresGrad) {
      throw new RuntimeError(
        "retainGrad() on a tensor that does not require grad — there would be no "
        + "gradient to keep."
        + "\n(torch: can't retain_grad on Tensor that has requires_grad=False)",
      );
    }
    this.retainsGrad = true;
  }

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
   * @param createGraph **refused.** It asks for the backward pass itself to be
   *   recorded so the gradient can be differentiated again — what a second-order
   *   method needs. This tape does not record it: a backward is a closure over
   *   GPU buffers, not a node. Walking the same ungraphed pass and returning
   *   would answer a first-order question to a caller who asked a second-order
   *   one, and the answer would look right. The core refuses it in the same
   *   words.
   * @param inputs restricts which tensors get a gradient. torch walks the whole
   *   graph and writes only to the ones named, which is how a second loss is
   *   differentiated against one branch without disturbing the rest. It also
   *   **retains**: a derived tensor named here gets a `grad` the way
   *   `retainGrad()` gives one — which is why torch's refusal for a tensor that
   *   does not require grad says *can't retain_grad*.
   */
  backward(
    // **`null` counts as absent, and that is not tidiness.** The Python binding
    // fills every position to reach the fourth, and Pyodide turns `None` into
    // `null` — there is no way to send `undefined` from that side. Read strictly,
    // `loss.backward(inputs=[w])` would arrive with a `null` seed, miss the
    // implicit-1 branch, and go looking for `gradient.shape`.
    gradient?: Tensor | null,
    retainGraph = false,
    createGraph = false,
    inputs?: Tensor | readonly Tensor[] | null,
  ): void {
    if (createGraph) {
      throw new NotImplementedError(
        "backward(createGraph = true) — double backward is not in the browser subset. "
        + "This tape holds closures over GPU buffers rather than nodes, so the backward "
        + "pass is not itself recorded and the second derivative has nowhere to come from.",
      );
    }
    let only: Set<Tensor> | undefined;
    if (inputs !== undefined && inputs !== null) {
      const listed = inputs instanceof Tensor ? [inputs] : [...inputs];
      // **Before every other refusal**, which is torch's order: a tensor that
      // does not require grad, called with an empty `inputs`, still stops here
      // (measured).
      if (listed.length === 0) {
        throw new RuntimeError(
          "backward(inputs: []) names nothing to put a gradient on. Leave `inputs` "
          + "out to fill every leaf instead."
          + "\n(torch: `inputs` argument to `backward()` cannot be empty.)",
        );
      }
      only = new Set(listed);
    }
    // **This check comes first.** Measured against torch, a non-scalar tensor that
    // does not require grad is refused this way rather than with "not a scalar" — it is
    // looked at before scalarness. The core (numpy) had that order from the start and
    // only this side was reversed. Three implementations giving different wording at one
    // place leaves ported code not knowing which to catch.
    if (!this.requiresGrad) {
      throw new RuntimeError(
        `element 0 of tensors ${TORCH.noGrad} and does not have a grad_fn: ` +
          "it was made under no_grad, or it passed through an operation that breaks the graph.",
      );
    }
    // **The loss has to be real.** torch stops at that place (measured).
    //
    // This one line holds up the whole complex gradient convention — only a loss that is
    // always real makes `z.grad = ∂L/∂re + i·∂L/∂im` well defined. Accepting a complex
    // loss would mean deciding Wirtinger's other half, and that has not been decided.
    if (isComplexDType(this.dtype)) {
      throw new RuntimeError(
        "grad can be implicitly created only for real scalar outputs " +
          "but got torch.complex64: make it real with .real or .abs() first.",
      );
    }
    let seed: Tensor;
    if (gradient === undefined || gradient === null) {
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
      // **It goes in detached.** torch's default is `create_graph=False` and this tape
      // does not do second derivatives — left attached, a leaf's `grad` holds on to the
      // graph.
      seed = gradient.detach();
    }
    // **After the seed has been checked, not before it** — torch complains about a
    // name that cannot hold a gradient only once the shape is settled (measured).
    if (only !== undefined) {
      for (const one of only) {
        if (!one.requiresGrad) {
          throw new RuntimeError(
            "backward(inputs: …) was given a tensor that does not require grad, so "
            + "there is nowhere for a gradient to go."
            + "\n(torch: can't retain_grad on Tensor that has requires_grad=False)",
          );
        }
      }
    }
    tapeBackward<Tensor>(this, seed, (a, b) => a.add(b), {
      ...(only === undefined ? {} : { only }),
      retainGraph,
      onSecondPass: () => {
        throw new RuntimeError(
          `${TORCH.secondBackward}. To flow through it again, ` +
            "call backward(undefined, true) to keep the graph.",
        );
      },
    });
  }

  // ── Moving between devices ────────────────────────────────────────────

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
  async cpu(memoryFormat: string | null = null): Promise<Tensor> {
    noMemoryFormat("cpu", memoryFormat);
    if (this.gpu === null) return this;
    // **It is `floats`.** Complex holds two per cell, so reading by `size` truncates
    // the second half — and the shape and dtype come out attached unchanged, so the
    // truncation is invisible.
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

  // ── Reading values ────────────────────────────────────────────────────

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
    // Already on the host, there is no round trip. **A copy is given** — handing out
    // the internal storage means the tensor's value changes when the receiver edits
    // it.
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

  /** Equal within a tolerance. The same defaults as torch's. */
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
    // **Complex cannot be printed yet.** Matching the `1.+2.j` form to torch
    // character for character means measuring and freezing its alignment rules, and that
    // has not been done. Emitting half-right characters looks right while not matching
    // the line in a textbook.
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

  /** Changes the dtype. The values stay — the storage is float32 throughout, so there
   *  is nothing to move. */
  /**
   * torch's **named type conversions.** The same work as `to(type)` under a
   * different name.
   *
   * Without these names, `x.float()` failed in borch.ts alone. Both Python
   * versions had them and the golden line stood as "a Python-side matter,
   * not carried across", when in fact it was not that the other side could
   * not — it was a name **nobody had asked about.**
   */
  float(memoryFormat: string | null = null): Tensor {
    noMemoryFormat("float", memoryFormat);
    return this.to("float32");
  }

  long(memoryFormat: string | null = null): Tensor {
    noMemoryFormat("long", memoryFormat);
    return this.to("int64");
  }

  bool(memoryFormat: string | null = null): Tensor {
    noMemoryFormat("bool", memoryFormat);
    return this.to("bool");
  }

  /**
   * Real to complex. **`to("complex64")` will not do it** — a complex is
   * two f32 per slot, so relabelling is blocked, and that block is right.
   * It is built by attaching a zero imaginary part.
   */
  cfloat(memoryFormat: string | null = null): Tensor {
    noMemoryFormat("cfloat", memoryFormat);
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

  int(memoryFormat: string | null = null): Tensor {
    noMemoryFormat("int", memoryFormat);
    return absentDType("int", "int32");
  }

  /**
   * **Double precision is absent for a different reason.** The eight above
   * are slots we chose not to have; this one is that WebGPU shaders have no
   * `f64` at all. The wording differs accordingly — the Python binding
   * stops with the same sentence.
   */
  double(memoryFormat: string | null = null): Tensor {
    noMemoryFormat("double", memoryFormat);
    throw new RuntimeError(
      "Only Tensors of floating point dtype float32 are supported — "
        + "float64 is not in WebGPU shaders",
    );
  }

  to(dtype: DType): Tensor {
    if (dtype === this.dtype) return this;
    // **Complex cannot be reached by relabelling.** Between the other dtypes the
    // storage is the same single float32, so the name alone changes; complex alone holds
    // two per cell — relabelling in either direction leaves the buffer length and `size`
    // out of step.
    if (dtype === "complex64" || isComplexDType(this.dtype)) {
      throw new RuntimeError(
        `torch.${this.dtype} -> torch.${dtype} is not a relabel — ` +
          "complex storage is two f32 per slot. " +
          "Move between them with Tensor.complex(re, im), viewAsComplex(), or real().",
      );
    }
    // **Going to an integer or a boolean changes the values too.** A dtype being a
    // label does not mean "any value may be inside" — a torch int64 tensor holds
    // integers.
    //
    // For a long time only the label changed. After `x.to("int64")` the buffer still
    // held `1.7`, truncated only on the way out, and **the arithmetic on the GPU carried
    // on in fractions.** The `sum(dtype=int64)` case surfaced it by diverging by exactly
    // 1 — torch truncates first and gets `−1`, while we folded to `0.3` untruncated and
    // truncated that to `0`.
    if (dtype === "int64" && this.dtype === "float32") {
      // **It truncates toward 0** (measured: `−2.3 → −2`). `floor` would give `−3`
      // and diverge.
      return this.unary("trunc").relabel(dtype);
    }
    if (dtype === "bool" && this.dtype !== "bool") {
      return this.binary("ne", Tensor.full([], 0), "bool");
    }
    return this.relabel(dtype);
  }

  /**
   * Keeps the values and changes the label alone. **Only between dtypes with the same
   * storage.**
   *
   * Places like `int64 → float32`, where the values already belong to that dtype. The
   * gradient is **not cut** — that is exactly where the core's `.float()` was quietly
   * cutting it.
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
    // **JavaScript has no complex number.** Returning the real part alone gives one
    // plausible number while the imaginary part disappears — an answer worse than
    // none.
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
 * torch refuses `dtype: "bool"` on an accumulation (measured — a
 * `NotImplementedError`).
 *
 * **`sum(dtype=bool)` works and `cumsum(dtype=bool)` does not.** That is not a rule but
 * a kernel torch never built, and diverging in the lenient direction is still diverging,
 * so this follows it — producing a value here breaks that code on real torch.
 */
function noBoolAccumulate(name: string, dtype: DType): void {
  if (dtype === "bool") {
    throw new NotImplementedError(`"${name}_out_cpu" not implemented for 'Bool'`);
  }
}

/** Folds a broadcast gradient back to the target shape. Identical shapes pass
 *  through. */
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

/** Expands a scalar gradient to the shape. `sum`'s backward. */
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

/** Spreads one scalar across n cells. */
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

// ── Complex kernels ────────────────────────────────────────────────────
//
// Every one is **one thread per complex cell.** So `i` is always a complex index and
// the f32 positions are `i*2` (real) and `i*2+1` (imaginary). Attaching a thread to an
// f32 cell instead makes the pair unreadable at once, and then multiplication cannot be
// written.

/**
 * The frame for a 1-D complex kernel. **The grid-folding line is written once.**
 *
 * Close to ten kernels share this head, and written out ten times by hand, a day comes
 * when one of them is written differently — this repository has been bitten by that kind
 * several times already.
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

/** Weaves a real and an imaginary part into the interleaved form. */
function complexPack(n: number): string {
  return complexShader(n, ["Re", "Im", "Out"],
    "  Out[i] = Re[gid];\n  Out[i + 1u] = Im[gid];");
}

/** From a magnitude and an angle into the interleaved form. */
function complexPolar(n: number): string {
  return complexShader(n, ["R", "T", "Out"],
    "  Out[i] = R[gid] * cos(T[gid]);\n  Out[i + 1u] = R[gid] * sin(T[gid]);");
}

/** Takes out the real part (`off=0`) or the imaginary part (`off=1`). */
function complexPart(n: number, off: 0 | 1): string {
  return complexShader(n, ["Z", "Out"], `  Out[gid] = Z[i + ${off}u];`);
}

/** A real into `x + 0i`. */
function complexFromReal(n: number): string {
  return complexShader(n, ["A", "Out"],
    "  Out[i] = A[gid];\n  Out[i + 1u] = 0.0;");
}

/** A real into `0 + xi`. */
function complexFromImag(n: number): string {
  return complexShader(n, ["A", "Out"],
    "  Out[i] = 0.0;\n  Out[i + 1u] = A[gid];");
}

/** The conjugate. It flips the imaginary part alone. */
function complexConj(n: number): string {
  return complexShader(n, ["Z", "Out"],
    "  Out[i] = Z[i];\n  Out[i + 1u] = -Z[i + 1u];");
}

/** Sign flip. It flips both — an easy place to confuse with the conjugate. */
function complexNeg(n: number): string {
  return complexShader(n, ["Z", "Out"],
    "  Out[i] = -Z[i];\n  Out[i + 1u] = -Z[i + 1u];");
}

/** The magnitude. The result is one real cell. */
function complexAbs(n: number): string {
  return complexShader(n, ["Z", "Out"],
    "  Out[gid] = sqrt(Z[i] * Z[i] + Z[i + 1u] * Z[i + 1u]);");
}

/** The angle. `atan2(im, re)` — the arguments the other way round give a quietly
 *  different angle. */
function complexAngle(n: number): string {
  return complexShader(n, ["Z", "Out"], "  Out[gid] = atan2(Z[i + 1u], Z[i]);");
}

/**
 * `abs`'s backward. **No conjugate** — `abs` produces a real, so it is not holomorphic.
 *
 * At 0 there is no direction. torch gives 0 there too — with the divisor replaced by 1,
 * the numerator is 0 and the result comes out 0.
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

/** Complex arithmetic. Only operands of matching shape arrive. */
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

/** The 2-D transpose kernel. The shape is a constant, so no division survives. */
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
 * Attaches every unary in the table as a method. No name is written twice.
 *
 * The in-place forms (the underscored ones, like `abs_`) are attached with them —
 * writing twenty-seven by hand, a day comes when one of them calls a different
 * operation.
 */
// **Captured before the table overwrites it.** The class body's `elu` takes α, and the
// loop below overwrites it with the table's argument-less form — the ordering problem
// `abs` was already bitten by. It is restored after the loop. The in-place form (`elu_`)
// uses the table's as it is (α=1).
/**
 * torch's `memoryFormat` on the dtype and device methods — **carried and refused.**
 *
 * Seven of them take it and none had a seat for it, so torch's position landed on
 * nothing. There is one layout here; honouring it is impossible and swallowing it
 * would answer for a request that was not met. The core refuses at the same place
 * with the same sentence (`borch/_tensor.py`, `_memory_format`).
 */
function noMemoryFormat(name: string, memoryFormat: string | null): void {
  if (memoryFormat !== null) {
    throw new Error(`${name}(memoryFormat) is not here — there is one layout.`);
  }
}

/**
 * **The in-place binaries, from a table.** `eq_`, `lt_`, `atan2_`, `hypot_` and
 * eight more are each `mutate(() => this.binary(name, other))` and nothing else, so
 * the comment beside the hand-written in-place methods — *they cannot run off a
 * table, so they are written one by one* — is true of the ones taking an axis or an
 * index and not of these. Twelve written out by hand is twelve places that can
 * drift, which is the argument `_unary` makes on the Python side.
 *
 * The list is torch's: every name here has `name_` on `torch.Tensor` (measured).
 * `logaddexp` and `logaddexp2` are **not** here — they have no in-place form in
 * torch, and adding one would be a name this library has and torch does not.
 */
const INPLACE_BINARY = [
  "atan2", "copysign", "eq", "ge", "gt", "heaviside", "hypot", "ldexp",
  "le", "lt", "ne", "xlogy",
] as const;

for (const name of INPLACE_BINARY) {
  Object.defineProperty(Tensor.prototype, `${name}_`, {
    value: function (this: Tensor, other: Tensor): Tensor {
      // **The computation belongs to the underscore-less side.** One expression in
      // two copies diverges eventually, and the values stay plausible while it does.
      return this.inplaceBinary(name, other);
    },
    writable: true,
    configurable: true,
  });
}

const INTERPOLATIONS = ["linear", "lower", "higher", "midpoint", "nearest"];

/**
 * torch's `interpolation`, as the weight given to the upper of the two positions.
 *
 * **It changes the answer.** On `[1,2,3,4]` the 0.3 quantile is 1.9 under `linear`,
 * 1.0 under `lower` and 1.5 under `midpoint` — three plausible numbers with nothing
 * in the value to say which rule made it. The argument was absent on both sides, so
 * four of the five rules were unreachable.
 */
function interpolationWeight(how: string, at: number, lo: number): number {
  switch (how) {
    case "linear": return at - lo;
    case "lower": return 0;
    case "higher": return 1;
    case "midpoint": return 0.5;
    case "nearest": return Math.round(at) - lo;
    default:
      throw new Error(
        `quantile() interpolation must be one of ${INTERPOLATIONS.join(", ")}, got ${how}`);
  }
}

/**
 * `dim` and `keepdim` — **carried and refused.** The body here flattens, so an axis
 * cannot be honoured; accepting one and reducing everything would answer a question
 * the caller did not ask, with a plausible number. Left out, torch's second and
 * third positions land on somebody else.
 */
function quantileSeats(dim: number | null, keepdim: boolean): void {
  if (dim !== null) throw new Error("quantile(dim) is not here yet — it flattens.");
  if (keepdim) throw new Error("quantile(keepdim) is not here yet — it flattens.");
}

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
 * **`abs` alone is attached again after the table.** The loop above lays its own onto
 * the prototype, so whatever the class body wrote is overwritten — the ordering *is* the
 * rule.
 *
 * Complex `abs` produces a real and its gradient is `z/|z|`. Its kernel, its dtype and
 * its backward all differ from the real one, so it cannot be a table unary.
 */
// Restores the `elu` that takes α. It was captured above.
Object.defineProperty(Tensor.prototype, "elu", {
  value: eluWithAlpha,
  writable: true,
  configurable: true,
});

/**
 * **`round` takes `decimals` and the table cannot give it one.** The loop above
 * attaches a no-argument method for every table unary, so `round(2)` was a type
 * error — for the argument that is the reason most people reach for `round`.
 *
 * Half goes to even at every scale, which is the rule the kernel already has; the
 * scaling is by a power of ten either side of it, so nothing about the rounding
 * itself moves. Attached after the loop for the same reason `elu` is.
 */
/**
 * **`logit` takes `eps` and the table cannot give it one**, exactly as `round` takes
 * `decimals`. torch clamps the *input* into `[eps, 1 − eps]` before the division, so
 * `logit(0, 0.1)` is −2.197 rather than −∞. Without it, 0 and 1 give ∓∞ — which is
 * also torch's answer and the default.
 *
 * Attached after the table for the reason `abs`, `elu` and `bitwise_not` are: the
 * loop overwrites whatever the class body wrote, so the ordering is the rule.
 */
{
  const rawLogit = Tensor.prototype.logit;
  Object.defineProperty(Tensor.prototype, "logit", {
    value: function (this: Tensor, eps: number | null = null): Tensor {
      if (eps === null || eps === undefined) return rawLogit.call(this);
      return rawLogit.call(this.clamp(eps, 1 - eps));
    },
    writable: true,
    configurable: true,
  });
}

Object.defineProperty(Tensor.prototype, "round", {
  value: function (this: Tensor, decimals = 0): Tensor {
    if (!decimals) return this.unary("round");
    const scale = Tensor.full([], 10 ** decimals);
    return this.mul(scale).unary("round").div(scale);
  },
  writable: true,
  configurable: true,
});

/**
 * **The in-place halves of the two above, which the table gave no seats.**
 *
 * `round_` and `logit_` came out of the `UNARY` loop taking nothing, while `round`
 * and `logit` had been given `decimals` and `eps` by hand right here. So the two
 * spellings of one operation took different arguments — `x.round_(2)` was a word
 * JavaScript dropped and the answer came back rounded to whole numbers.
 *
 * It could not be seen from the Python side either: the core's `logit_` was built
 * nullary, so **both libraries agreed by being wrong the same way**, and the
 * signature axis filed the row as *no python signature* because the core's
 * forwarders declared `(*args, **kw)`. Teaching those forwarders to declare what
 * they forward is what surfaced all three.
 */
for (const [name, seats] of [["round", 1], ["logit", 1]] as const) {
  void seats;
  Object.defineProperty(Tensor.prototype, `${name}_`, {
    value: function (this: Tensor, arg?: number | null): Tensor {
      return this.inplaceFrom(() =>
        (this as unknown as Record<string, (a?: number | null) => Tensor>)[name]!
          .call(this, arg));
    },
    writable: true,
    configurable: true,
  });
}

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
 * **`bitwise_not` is attached again after the table too.** On a boolean it is logical
 * negation — `~true` is `false`, not `-2` (measured). That branch lived in the binding
 * alone for a long time, and the kernel's comment said "this only looks at integers".
 * Which leaves somebody calling from TypeScript with **a wrong answer rather than a
 * missing one.**
 *
 * Written in the class body, the loop above overwrote it — the ordering problem `abs`
 * and `elu` were already bitten by, and this is the third. Fixing by hand an operation
 * whose name is in the table has to happen **after the table.**
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
  round(decimals?: number): Tensor;
  // **Fourteen names that ran and could not be seen.** The loop over `UNARY`
  // attaches a method for every kernel, and this interface is what declares them —
  // eighteen of the sixty-six had no line here, so `x.sinc()` worked at runtime,
  // was frozen in the golden through the module-level path, and was absent from
  // TypeScript's view, from the API reference, and from `tests/ts_axis.py`.
  //
  // The axis counted them as **features borch.ts does not have**. They were
  // features borch.ts did not declare, which is not the same thing and repairs
  // differently: the fix is these lines, not a kernel.
  //
  // (`nanToZero` and `notNan` stay undeclared on purpose — the kernel file says
  // neither is a public torch name; they are the pieces `nansum` is built from.
  // `logical_not` and `elu` have declared spellings of their own.)
  ldexp_(other: Tensor): Tensor;
  atan2_(other: Tensor): Tensor;
  copysign_(other: Tensor): Tensor;
  eq_(other: Tensor): Tensor;
  ge_(other: Tensor): Tensor;
  gt_(other: Tensor): Tensor;
  // **torch calls it `values`**, and `heaviside` above already does — these
  // declarations are generated from one table and this row is the one where the
  // table's own name is not torch's.
  heaviside_(values: Tensor): Tensor;
  hypot_(other: Tensor): Tensor;
  le_(other: Tensor): Tensor;
  lt_(other: Tensor): Tensor;
  ne_(other: Tensor): Tensor;
  xlogy_(other: Tensor): Tensor;
  deg2rad(): Tensor;
  rad2deg(): Tensor;
  positive(): Tensor;
  sgn(): Tensor;
  sinc(): Tensor;
  signbit(): Tensor;
  isnan(): Tensor;
  isinf(): Tensor;
  isfinite(): Tensor;
  erf(): Tensor;
  erfc(): Tensor;
  gelu(): Tensor;
  silu(): Tensor;
  // `logit` takes torch's `eps`, so it is attached after the table like `round`.
  logit(eps?: number | null): Tensor;
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
  // ── The in-place forms the table attaches ─────────────────────────────
  //
  // The loop above attaches **two** per name (`abs` and `abs_`). This declaration held
  // only the first for a long time, so sixty-five of them **existed at runtime and not
  // in the types** — typing `x.acosh_()` in TypeScript fails to compile, and they do not
  // appear in the site's reference either.
  //
  // This block's head said "it pairs with the loop above, and fixing one alone puts them
  // out of step". What was out of step was not the side that was fixed but **the side
  // that only ever wrote half.**
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
  round_(decimals?: number): Tensor;
  trunc_(): Tensor;
  frac_(): Tensor;
  deg2rad_(): Tensor;
  rad2deg_(): Tensor;
  positive_(): Tensor;
  logit_(eps?: number | null): Tensor;
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
  // **It is `raw`.** With a complex among the things being kept alive, `buffer` would
  // refuse and an exception would come out where the scope closes — lifetime management
  // has no need to know a value's dtype.
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
  // Something on the host has nothing to keep alive — a scope releases GPU buffers
  // only, and JavaScript's garbage collector takes a `Float32Array` by itself.
  // `keepAlive(await t.cpu())` is a natural line to write, so it must not be refused
  // here.
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
