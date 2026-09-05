/**
 * The `cpu` device — a classifier forward, a head on its features, and neighbours over
 * them, with no WebGPU adapter at all.
 *
 * ## What this is, and what it is not
 *
 * `borch-ts` proper is a `Tensor` on WebGPU. This namespace is not a second `Tensor`
 * backend: on the machines it exists for there is no adapter, so there is no `Tensor`
 * to make. It is a small set of things that run on WebAssembly SIMD kernels shipped
 * inside the package (`cpu/kernels`: two modules of 17 KB, strict and relaxed — see
 * `load.ts` —, no runtime, no imports):
 *
 * - `readSafetensors` — a checkpoint's bytes to host arrays, without a device
 * - `GraphBuilder` / `CpuRunner` — a short graph (conv, depthwise, pool, SE, add,
 *   linear; BatchNorm folded in) built from those arrays, and its forward
 * - `LinearHead` / `cosineNeighbours` — a head trained on cached features, and the
 *   nearest rows by cosine
 *
 * Everything else is absent by name. Measured against the WebGPU device on two hub
 * checkpoints (`borch-ts/test/cpu.py`; the numbers are in the book under "The `cpu`
 * device"): logits within 1e-4 relative, a head's losses within 3.5e-5 step for step.
 *
 * ## Choosing it
 *
 * The site's probe already says whether the adapter is hardware, software, or absent.
 * `available()` answers the one question this side adds — whether the engine runs
 * WebAssembly SIMD, which every browser that ships WebGPU does and a few older ones do
 * not. An application picks: hardware adapter → `borch-ts`; otherwise, if `available()`
 * → this.
 */
export { ACT, type Activation, type CpuKernels, type KernelFlavor, type LoadKernelsOptions, loadKernels, kernelBytes, kernelsFromExports, relaxedSimdAvailable, sharedMemoryAvailable } from "./cpu/load.js";
export { KERNELS_WASM_SHA256, KERNELS_WASM_BYTES, KERNELS_RUSTC } from "./cpu/kernels.js";
export { readSafetensors, type HostStateDict, type HostTensor } from "./cpu/safetensors.js";
export {
  GraphBuilder, CHANNEL_PAD,
  type CpuGraph, type BatchNorm, type ConvSpec, type Node,
} from "./cpu/graph.js";
export { CpuRunner } from "./cpu/runner.js";
export { LinearHead, cosineNeighbours, type HeadOptions, type HeadState, type Neighbours } from "./cpu/train.js";

// A module with one function that returns v128 zero — the smallest thing that is valid
// only where SIMD is. `WebAssembly.validate` neither instantiates nor runs it.
const SIMD_PROBE = new Uint8Array([
  0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 123, 3, 2, 1, 0, 10, 10, 1, 8, 0, 65, 0, 253, 15, 253, 98, 11,
]);

/** Whether this engine can run the kernels — WebAssembly with the SIMD proposal. */
export function available(): boolean {
  try {
    return typeof WebAssembly !== "undefined" && WebAssembly.validate(SIMD_PROBE);
  } catch {
    return false;
  }
}
export { MainSide, WorkerPool, CONTROL_LAYOUT, makeControl, workerLoop, threadsAvailable, defaultWorkers, type Dispatcher, type Task, type Call, type Layout } from "./cpu/threads.js";
