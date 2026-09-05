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
import { KERNELS_EXPORTS, KERNELS_WASM_BASE64, KERNELS_WASM_SHA256 } from "./kernels.js";

/** The activation `biasAct` applies after adding the bias. */
export const ACT = { none: 0, swish: 1, sigmoid: 2 } as const;
export type Act = (typeof ACT)[keyof typeof ACT];

export interface CpuKernels {
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
  biasAct(rows: number, c: number, x: number, bias: number, act: Act): void;
  /** out[c] = mean over rows. `c % 4 === 0`. */
  meanRows(rows: number, c: number, x: number, out: number): void;
  /** x[rows×c] *= s[c]. `c % 4 === 0`. */
  scaleRows(rows: number, c: number, x: number, s: number): void;
  /** a[n] += b[n]. `n % 4 === 0`. */
  addInplace(n: number, a: number, b: number): void;
}

/** The module's bytes, decoded. Pure; does not instantiate. */
export function kernelBytes(): Uint8Array<ArrayBuffer> {
  const text = atob(KERNELS_WASM_BASE64);
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

let loading: Promise<CpuKernels> | null = null;

/**
 * Decode, verify, instantiate. Called twice, it returns the same promise — the module
 * has one memory and two copies would be two heaps.
 */
export function loadKernels(): Promise<CpuKernels> {
  if (loading) return loading;
  loading = (async () => {
    const bytes = kernelBytes();
    const got = await sha256Hex(bytes);
    if (got !== KERNELS_WASM_SHA256) {
      throw new Error(`cpu kernels: the embedded bytes hash to ${got.slice(0, 12)}…, kernels.ts says ${KERNELS_WASM_SHA256.slice(0, 12)}… — run npm run build:wasm`);
    }
    const { instance } = await WebAssembly.instantiate(bytes, {});
    const ex = instance.exports;
    const memory = ex["memory"];
    if (!(memory instanceof WebAssembly.Memory)) throw new Error('cpu kernels: export "memory" is missing');
    for (const name of KERNELS_EXPORTS) fn(ex, name);
    const alloc = fn(ex, "alloc"), reset = fn(ex, "reset"), heap = fn(ex, "heap"), setHeap = fn(ex, "set_heap");
    const gemm = fn(ex, "gemm"), dwconv = fn(ex, "dwconv"), swish = fn(ex, "swish"), biasAct = fn(ex, "bias_act");
    const meanRows = fn(ex, "mean_rows"), scaleRows = fn(ex, "scale_rows"), addInplace = fn(ex, "add_inplace");
    return {
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
    };
  })();
  return loading;
}
