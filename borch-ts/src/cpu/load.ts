/**
 * Loads the CPU kernels — the wasm module `kernels.ts` carries as base64.
 *
 * ## What a caller gets
 *
 * A typed handle over the module's exports and its one linear memory. Offsets are
 * byte offsets into that memory; the tensor side (`borch-ts/src/cpu/`) allocates with
 * `alloc`, writes `f32` through a `Float32Array` view, and calls the kernels on offsets.
 * The view has to be re-taken after any `alloc`, because growing the memory detaches
 * the old buffer.
 *
 * ## The bytes are checked before they run
 *
 * `loadKernels` hashes the decoded bytes and compares with `KERNELS_WASM_SHA256`. The
 * hash was written by the same script that wrote the bytes, so the check is not
 * against tampering by a stranger; it is against the ordinary accident — a merge that
 * kept the constant and lost half the blob, an editor that rewrapped a line. Either
 * would otherwise instantiate something else under this name. `borch-hub` refuses a
 * model that cannot reproduce its own sample for the same reason.
 *
 * ## No `any`
 *
 * `WebAssembly.Exports` is a bag of unknowns. Each export is checked to be a function
 * (or, for `memory`, a `WebAssembly.Memory`) before it is given a signature, and the
 * one assertion in this file sits right after that check.
 */
import { KERNELS_EXPORTS, KERNELS_RELAXED_WASM_BASE64, KERNELS_RELAXED_WASM_SHA256, KERNELS_WASM_BASE64, KERNELS_WASM_SHA256 } from "./kernels.js";

/** The activation `biasAct` applies after adding the bias. */
export const ACT = { none: 0, swish: 1, sigmoid: 2, relu: 3 } as const;
export type Activation = (typeof ACT)[keyof typeof ACT];

/**
 * Which module is running. `strict` rounds every multiply and add separately, the same on
 * every machine; `relaxed` fuses them (`f32x4.relaxed_madd`) and is 1.7× faster on the
 * GEMM, with the last bit the hardware's.
 */
export type KernelFlavor = "strict" | "relaxed";

export interface CpuKernels {
  readonly flavor: KernelFlavor;
  readonly memory: WebAssembly.Memory;
  /** Bump-allocate `bytes`, 16-byte aligned. Returns a byte offset, or 0 when the memory would not grow. */
  alloc(bytes: number): number;
  /** Forget every allocation. */
  reset(): void;
  /** Where the heap stands, for `setHeap` to return to. */
  heap(): number;
  setHeap(pos: number): void;
  /** C[m×n] = A[m×k] · B[k×n], f32 row-major. `m % 4 === 0`, `n % 16 === 0`. */
  gemm(m: number, n: number, k: number, a: number, b: number, c: number): void;
  /** NHWC depthwise conv. `c % 16 === 0`. */
  dwconv(h: number, w: number, c: number, k: number, stride: number, pad: number, ho: number, wo: number, inp: number, wt: number, out: number): void;
  /** x ← x · sigmoid(x). `n % 4 === 0`. */
  swish(n: number, x: number): void;
  /** x[rows×c] += bias[c], then `act`. `c % 4 === 0`. */
  biasAct(rows: number, c: number, x: number, bias: number, act: Activation): void;
  /** out[c] = mean over rows. `c % 4 === 0`. */
  meanRows(rows: number, c: number, x: number, out: number): void;
  /** x[rows×c] *= s[c]. `c % 4 === 0`. */
  scaleRows(rows: number, c: number, x: number, s: number): void;
  /** a[n] += b[n]. `n % 4 === 0`. */
  addInplace(n: number, a: number, b: number): void;
  /** x ← max(x, 0). `n % 4 === 0`. */
  relu(n: number, x: number): void;
  /** NHWC im2col: in [h,w,c] → out [ho·wo, k·k·c], tap-major then channel. Any `c`. */
  im2col(h: number, w: number, c: number, k: number, stride: number, pad: number, ho: number, wo: number, inp: number, out: number): void;
  /** NHWC max pool, taps outside skipped. `c % 4 === 0`. */
  maxpool(h: number, w: number, c: number, k: number, stride: number, pad: number, ho: number, wo: number, inp: number, out: number): void;
  /**
   * Softmax cross-entropy, backward half: `grad = (softmax − onehot) / rows` over the first
   * `cReal` of `c` columns, and per row `stats[2r] = max`, `stats[2r+1] = Σexp` for the host to
   * finish the loss with `ln`. Labels are class indices stored as f32.
   */
  softmaxXentGrad(rows: number, c: number, cReal: number, logits: number, labels: number, grad: number, stats: number): void;
  /** out[d×k] += Xᵀ·G for X[n×d], G[n×k]. `k % 4 === 0`. Not zeroed first. */
  outerAcc(n: number, d: number, k: number, x: number, g: number, out: number): void;
  /** torch's SGD step: `g += wd·p; v = μ·v + g; p −= lr·v`. `n % 4 === 0`. */
  sgdStep(n: number, p: number, g: number, v: number, lr: number, momentum: number, weightDecay: number): void;
  /** Each row divided by its L2 norm; zero rows stay zero. `c % 4 === 0`. */
  l2NormalizeRows(rows: number, c: number, x: number): void;
  /** out[cols×rows] = in[rows×cols]ᵀ. */
  transpose(rows: number, cols: number, inp: number, out: number): void;
  /** x[n] ← 0. `n % 4 === 0`. */
  zero(n: number, x: number): void;
  /** `gemm`, then `+ bias[n]` and `act` before the store — the sum never leaves the registers. */
  gemmBiasAct(m: number, n: number, k: number, a: number, b: number, c: number, bias: number, act: Activation): void;
  /** `dwconv` with the bias as the starting value and `act` on each finished row. */
  dwconvBiasAct(h: number, w: number, c: number, k: number, stride: number, pad: number, ho: number, wo: number, inp: number, wt: number, out: number, bias: number, act: Activation): void;
  /** `im2col` for output rows `[row0, row0 + rows)` of one image — a block at a time against one buffer. */
  im2colRows(h: number, w: number, c: number, k: number, stride: number, pad: number, wo: number, row0: number, rows: number, inp: number, out: number): void;
}

// A module with one function, `f32x4.relaxed_madd` on its three v128 parameters — valid only
// where the relaxed SIMD proposal is. `WebAssembly.validate` neither instantiates nor runs it.
const RELAXED_PROBE = new Uint8Array([
  0, 97, 115, 109, 1, 0, 0, 0, 1, 8, 1, 96, 3, 123, 123, 123, 1, 123, 3, 2, 1, 0,
  10, 13, 1, 11, 0, 32, 0, 32, 1, 32, 2, 253, 133, 2, 11,
]);

/** Whether this engine accepts relaxed SIMD — Chrome 114+, Firefox 145+, Safari from Technology Preview 250. */
export function relaxedSimdAvailable(): boolean {
  try {
    return typeof WebAssembly !== "undefined" && WebAssembly.validate(RELAXED_PROBE);
  } catch {
    return false;
  }
}

/** A module's bytes, decoded. Pure; does not instantiate. */
export function kernelBytes(flavor: KernelFlavor = "strict"): Uint8Array<ArrayBuffer> {
  const text = atob(flavor === "relaxed" ? KERNELS_RELAXED_WASM_BASE64 : KERNELS_WASM_BASE64);
  const bytes = new Uint8Array(new ArrayBuffer(text.length));
  for (let i = 0; i < text.length; i++) bytes[i] = text.charCodeAt(i);
  return bytes;
}

async function sha256Hex(bytes: Uint8Array<ArrayBuffer>): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  return Array.from(new Uint8Array(digest), (b) => b.toString(16).padStart(2, "0")).join("");
}

type Fn = (...args: number[]) => number;

function fn(exports: WebAssembly.Exports, name: string): Fn {
  const value = exports[name];
  if (typeof value !== "function") throw new Error(`cpu kernels: export "${name}" is missing or not a function`);
  return value as Fn;
}

const loading: Partial<Record<KernelFlavor, Promise<CpuKernels>>> = {};

export interface LoadKernelsOptions {
  /**
   * `true` asks for the relaxed module and fails where the engine has no relaxed SIMD;
   * `false` asks for the strict one; absent, the relaxed one where the engine takes it
   * and the strict one elsewhere.
   */
  readonly relaxed?: boolean;
}

/**
 * Decode, verify, instantiate. Called twice for the same flavor, it returns the same
 * promise — a module has one memory and two copies would be two heaps. The two flavors
 * are two modules and may both be loaded, which is how a check compares them.
 */
export function loadKernels(opts: LoadKernelsOptions = {}): Promise<CpuKernels> {
  const flavor: KernelFlavor = opts.relaxed === true ? "relaxed" : opts.relaxed === false ? "strict" : (relaxedSimdAvailable() ? "relaxed" : "strict");
  const have = loading[flavor];
  if (have) return have;
  const promise: Promise<CpuKernels> = (async (): Promise<CpuKernels> => {
    const bytes = kernelBytes(flavor);
    const want = flavor === "relaxed" ? KERNELS_RELAXED_WASM_SHA256 : KERNELS_WASM_SHA256;
    const got = await sha256Hex(bytes);
    if (got !== want) {
      throw new Error(`cpu kernels (${flavor}): the embedded bytes hash to ${got.slice(0, 12)}…, kernels.ts says ${want.slice(0, 12)}… — run npm run build:wasm`);
    }
    const { instance } = await WebAssembly.instantiate(bytes, {});
    const ex = instance.exports;
    const memory = ex["memory"];
    if (!(memory instanceof WebAssembly.Memory)) throw new Error('cpu kernels: export "memory" is missing');
    for (const name of KERNELS_EXPORTS) fn(ex, name);
    const alloc = fn(ex, "alloc"), reset = fn(ex, "reset"), heap = fn(ex, "heap"), setHeap = fn(ex, "set_heap");
    const gemm = fn(ex, "gemm"), dwconv = fn(ex, "dwconv"), swish = fn(ex, "swish"), biasAct = fn(ex, "bias_act");
    const meanRows = fn(ex, "mean_rows"), scaleRows = fn(ex, "scale_rows"), addInplace = fn(ex, "add_inplace");
    const relu = fn(ex, "relu"), im2col = fn(ex, "im2col"), maxpool = fn(ex, "maxpool");
    const softmaxXentGrad = fn(ex, "softmax_xent_grad"), outerAcc = fn(ex, "outer_acc"), sgdStep = fn(ex, "sgd_step");
    const l2NormalizeRows = fn(ex, "l2_normalize_rows"), transpose = fn(ex, "transpose"), zero = fn(ex, "zero");
    const gemmBiasAct = fn(ex, "gemm_bias_act"), dwconvBiasAct = fn(ex, "dwconv_bias_act"), im2colRows = fn(ex, "im2col_rows");
    return {
      flavor,
      memory,
      alloc: (bytes) => alloc(bytes),
      reset: () => { reset(); },
      heap: () => heap(),
      setHeap: (pos) => { setHeap(pos); },
      gemm: (m, n, k, a, b, c) => { gemm(m, n, k, a, b, c); },
      dwconv: (h, w, c, k, stride, pad, ho, wo, inp, wt, out) => { dwconv(h, w, c, k, stride, pad, ho, wo, inp, wt, out); },
      swish: (n, x) => { swish(n, x); },
      biasAct: (rows, c, x, bias, act) => { biasAct(rows, c, x, bias, act); },
      meanRows: (rows, c, x, out) => { meanRows(rows, c, x, out); },
      scaleRows: (rows, c, x, s) => { scaleRows(rows, c, x, s); },
      addInplace: (n, a, b) => { addInplace(n, a, b); },
      relu: (n, x) => { relu(n, x); },
      im2col: (h, w, c, k, stride, pad, ho, wo, inp, out) => { im2col(h, w, c, k, stride, pad, ho, wo, inp, out); },
      maxpool: (h, w, c, k, stride, pad, ho, wo, inp, out) => { maxpool(h, w, c, k, stride, pad, ho, wo, inp, out); },
      softmaxXentGrad: (rows, c, cReal, logits, labels, grad, stats) => { softmaxXentGrad(rows, c, cReal, logits, labels, grad, stats); },
      outerAcc: (n, d, k, x, g, out) => { outerAcc(n, d, k, x, g, out); },
      sgdStep: (n, p, g, v, lr, momentum, weightDecay) => { sgdStep(n, p, g, v, lr, momentum, weightDecay); },
      l2NormalizeRows: (rows, c, x) => { l2NormalizeRows(rows, c, x); },
      transpose: (rows, cols, inp, out) => { transpose(rows, cols, inp, out); },
      zero: (n, x) => { zero(n, x); },
      gemmBiasAct: (m, n, k, a, b, c, bias, act) => { gemmBiasAct(m, n, k, a, b, c, bias, act); },
      dwconvBiasAct: (h, w, c, k, stride, pad, ho, wo, inp, wt, out, bias, act) => { dwconvBiasAct(h, w, c, k, stride, pad, ho, wo, inp, wt, out, bias, act); },
      im2colRows: (h, w, c, k, stride, pad, wo, row0, rows, inp, out) => { im2colRows(h, w, c, k, stride, pad, wo, row0, rows, inp, out); },
    };
  })();
  loading[flavor] = promise;
  return promise;
}
