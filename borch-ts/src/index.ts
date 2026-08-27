/**
 * borch — a PyTorch-shaped tensor library running on WebGPU in the browser.
 *
 * ```ts
 * import { init, Tensor, nn, optim, scope, keepAlive } from "borch-ts";
 *
 * await init();                                  // acquire the WebGPU adapter
 *
 * const model = new nn.Sequential(
 *   new nn.Linear(784, 128), new nn.ReLU(), new nn.Linear(128, 10));
 * const opt = new optim.SGD(model.parameters(), 0.05, 0.9);
 * const crit = new nn.CrossEntropyLoss();
 *
 * const x = keepAlive(Tensor.from(pixels, [32, 784]));
 * const y = keepAlive(Tensor.from(labels, [32], { dtype: "int64" }));
 *
 * for (let i = 0; i < steps; i++) {
 *   await scope(async () => {                    // release one step's intermediates
 *     opt.zeroGrad();
 *     const loss = crit.call(model.call(x), y);
 *     loss.backward();
 *     opt.step();
 *     console.log(await loss.item());
 *   });
 * }
 * ```
 *
 * ## What this file does
 *
 * It gathers the public surface **into one place.** Without it, users have
 * to point at inner paths like `dist/src/tensor.js`, and then the moment we
 * move a file their code breaks. What is written here is the promise; the
 * rest is internal business.
 *
 * ## Where it parts from torch — written down in advance
 *
 * **`await init()` has to be called first.** Acquiring the WebGPU adapter
 * is asynchronous and there is no way around it. Make a tensor without
 * calling it and it stops there with a message. Where
 * `torch.cuda.is_available()` goes, use `await isAvailable()`, and if you
 * need to know *why* it is unavailable, `await probe()` separates
 * `'no-api'` (the browser is too old) from `'no-adapter'` (driver blocked,
 * headless).
 *
 * **`'cpu'` is a place values are held, not a device that computes.** Bring
 * values down to the host with `await t.cpu()` and back up with
 * `t.webgpu()`. A tensor that has come down can be read (`toArray`, `item`,
 * `repr`) but cannot be computed with — borch has no CPU kernels. Feed one
 * in and it stops with torch's own wording ("Expected all tensors to be on
 * the same device"). `t.device` answers where it is.
 *
 * **Reading values is asynchronous.** `await t.item()`, `await
 * t.toArray()`. It is a trip back for GPU memory. Forward and backward are
 * synchronous.
 *
 * **Training needs a scope open.** Intermediate buffers made inside
 * `scope()` are released on the way out. JavaScript's garbage collector
 * does not return GPU memory in time, so leaving the thousands one step
 * makes fills the device within a few steps. It is a concept torch does not
 * have, and it sits where TF.js's `tidy` sits. Anything that has to
 * survive, such as parameters, is marked with `keepAlive`.
 *
 * **There are two spellings — the one closer to Python's `with` is
 * recommended.**
 *
 * ```ts
 * for (let i = 0; i < steps; i++) {
 *   using s = scope();                 // closes at the end of the block
 *   opt.zeroGrad();
 *   const loss = crit.call(model.call(x), y);
 *   loss.backward();
 *   opt.step();
 *   console.log(await loss.item());
 * }
 * ```
 *
 * Releasing is synchronous, so it is `using` and not `await using`. The
 * block is left only after every `await` inside it has finished, which
 * makes the `await loss.item()` above safe (measured). The callback
 * spelling stays as well — it is shorter where you want the value handed
 * straight back.
 *
 * **Forgetting stops loudly.** Using a tensor carried out of a scope
 * without being marked raises at that point. For a while it did not, and
 * back then **whatever the next allocation had written over it** was read
 * quietly (measured: `[1,2,3,4]` came back as `9,9,9,9`).
 *
 * **A model is called as `model.call(x)`.** JavaScript cannot simply call
 * an object, so torch's `model(x)` cannot be carried over as written.
 * Calling `forward` directly gives the same value, but `call` is the
 * recommended one — it is where hooks would attach if they ever do.
 */

export {
  currentDevice,
  device,
  init,
  keepAlive,
  noGrad,
  scope,
  Tensor,
} from "./tensor.js";
// Somebody writing `using s = scope()` has to be able to name the handle's type.
export type { Scope } from "./tensor.js";

import { Tensor as TensorClass } from "./tensor.js";

export { Device, isAvailable, isSoftwareAdapter, probe } from "./device.js";
export type { Availability, DeviceKind, InitOptions } from "./device.js";
// The place `torch.manual_seed` occupies — layer initialisation, dropout and
// `Tensor.randn` all hang on one seed. `nn.manualSeed` gives the same thing (the old name
// is not broken).
export { manualSeed } from "./random.js";
export { einsum } from "./einsum.js";
// The place brackets occupy. `x[1:3]` resolves to `x[slice(1, 3)]` in Python too, so it
// is the same name.
export { slice } from "./indexing.js";
export type { Slice } from "./indexing.js";
// Checkpoints. **The format is safetensors** — Python `borch`, numpy and the HF tools
// read the same file. torch's `save`/`load` is pickle, which cannot and should not be
// ported.
// `save`/`load` carry **the nesting as it is** — the same place as torch and Python
// `borch`. `encode`/`decode` are the codec beneath them and deal only in a flat tensor
// table and bytes.
export {
  decode, encode, load, metaToNumbers, numbersToMeta, prefixed, save, unprefixed,
} from "./serialize.js";
export type { Bundle, Savable } from "./serialize.js";
// **It has to be openable from outside.** `noGrad(fn)` takes a function, so it cannot be
// ported to Python's `with` — the binding holds the switch directly.
export { gradMode } from "./autograd.js";

/**
 * Whether this is a tensor.
 *
 * Code wrapping this library from outside needs it — it has to treat a
 * method's result differently depending on whether it returns a tensor, a
 * number or a shape, and using `instanceof` means importing the class,
 * which other languages cannot do. Pyodide's Python binding is the first
 * user.
 */
export function isTensor(x: unknown): boolean {
  return x instanceof TensorClass;
}

/**
 * Empties one slot **to `null`.**
 *
 * ## Why this is exposed
 *
 * Python cannot make JavaScript's `null`. Pyodide hands `null` to Python as
 * `None`, and sending `None` back the other way produces **`undefined`** —
 * which in JavaScript is a different value.
 *
 * So `p.grad = None` left the slot on the other side as `undefined`, and
 * `autograd.ts` asks whether it is empty with `node.grad === null`. The
 * comparison came out false, it tried to accumulate gradient into something
 * that was not there, and what blew up was `Cannot read properties of
 * undefined (reading 'add')` — a long way from the Python line.
 *
 * **Loosening the strict comparison over there is not the fix.** That
 * weakens the TS invariant and leaves the real cause, the conversion,
 * exactly where it was. Put the way to make a null on the side that has
 * nulls.
 */
export function setNull(target: Record<string, unknown>, key: string): void {
  target[key] = null;
}
export { IndexError, RuntimeError } from "./errors.js";
import { RuntimeError } from "./errors.js";
export type { DType } from "./dtype.js";

// The namespaces are kept apart — both `nn` and `vision` have a `manualSeed`, and torch
// separates them as `torch.nn` and `torch.optim` too.
// The place `torch.utils.data` occupies. The namespace is kept apart for the same reason
// as `nn` and `optim`.
export * as data from "./data.js";
// The place `torch.fft` occupies. Its body carries its own kernel (the DFT shader), so it
// stands apart.
export * as fft from "./fft.js";
// The place `torch.linalg` occupies. The names forward to `Tensor` methods; what the
// namespace decides is the shape of the call. The numerics live in `_linalg.ts`, which is
// internal and takes flat `Float64Array`s rather than tensors.
export * as linalg from "./linalg.js";
export * as nn from "./nn.js";
export * as optim from "./optim.js";
export * as vision from "./vision.js";
// The place `torchvision.transforms.v2.functional` occupies. It is a separate module
// rather than a member of `vision` because it **imports** from it — nine new names and
// the rest bound straight through, so a cycle would be the price of nesting a namespace
// that is, on purpose, mostly the one it would nest inside.
export * as visionV2 from "./vision_v2.js";
// The place `torchvision.datasets` occupies — **the format half of it and not the
// address half.** Fetching, caching and checksums belong wherever the network is; a
// decoder takes bytes and is the part that goes wrong quietly.
export * as datasets from "./datasets.js";

// **`stft` and `istft` are top level** — torch puts them at `torch.stft`. The same
// functions are visible inside the `fft` namespace as well, and the form textbook code
// uses is the top-level one.
export { istft, stft } from "./fft.js";
export type { StftOptions } from "./fft.js";

// **The eight top-level recurrent names.** torch gives both the layer (`nn.LSTM`) and the
// function (`torch.lstm`), and what the layer calls internally is the function — the
// difference is that it takes the weights as a list.
export {
  gru, gruCell, lstm, lstmCell, rnnRelu, rnnReluCell, rnnTanh, rnnTanhCell,
} from "./rnn.js";
export type { RnnOptions } from "./rnn.js";

// Which build this is. A receiver comparing a manifest's `runtime.ts` range needs it —
// without it that field is written and never read.
export { VERSION } from "./version.js";

// The special functions. `n` is a shader constant, so they do not fit the unary table and
// stand apart.
export { igamma, igammac, polygamma } from "./special.js";

// ── four names whose Python rows said what they are for ─────────────────────────
//
// Each of these was declined in the core with a sentence that was **true about
// something else**. Written here so the three implementations answer one way.

/**
 * A `narrow` and a copy.
 *
 * torch's row for this said sparse tensors are why torch has it — a sparse tensor
 * has no view to narrow — which is true and is not why it was absent. On a dense
 * tensor it is a narrow and a clone, and **the clone is the whole difference**:
 * writing into a narrow reaches the original, writing into this one does not.
 */
export function narrowCopy(
  input: TensorClass, dim: number, start: number, length: number,
): TensorClass {
  return input.narrow(dim, start, length).clone();
}

/** The five reductions `segmentReduce` takes, by torch's spelling. */
export type SegmentReduction = "sum" | "prod" | "mean" | "max" | "min";

/**
 * Reduce a tensor **in runs**, one answer per run.
 *
 * torch's row read *for sparse and ragged bundles*, and the ragged half is dense:
 * a tensor and a list of run lengths. `lengths` and `offsets` are two spellings of
 * one thing — lengths are how long each run is, offsets are where each one starts,
 * so `offsets` carries one extra number, the end. A reader that took one for the
 * other shifts every boundary by the first run's length, which **runs of equal
 * size hide**, so the cases ask an uneven table too.
 *
 * `initial` seeds each run before reducing, which is what makes an empty run
 * answerable at all.
 */
export function segmentReduce(
  data: TensorClass,
  reduce: SegmentReduction,
  options: {
    lengths?: readonly number[] | readonly (readonly number[])[];
    offsets?: readonly number[] | readonly (readonly number[])[];
    axis?: number;
    initial?: number | null;
  } = {},
): TensorClass {
  const { lengths, offsets, axis = 0, initial = null } = options;
  if ((lengths === undefined) === (offsets === undefined)) {
    throw new RuntimeError(
      "segment_reduce(): Expected exactly one of lengths or offsets to be set");
  }
  // **A run table per row.** torch takes one, and the boundaries then differ between
  // the rows, so no single slice across the axis answers it — each row is its own
  // call and the answers stack.
  const table = (lengths ?? offsets) as readonly unknown[];
  if (Array.isArray(table[0])) {
    const rows = table.map((row, i) => segmentReduce(
      data.select(0, i), reduce,
      lengths === undefined
        ? { offsets: row as readonly number[], axis: axis - 1, initial }
        : { lengths: row as readonly number[], axis: axis - 1, initial }));
    return TensorClass.stack(rows, 0);
  }
  // One shape from here down: run `i` covers `[bound[i], bound[i + 1])`.
  const bound: number[] = [0];
  if (lengths !== undefined) {
    for (const run of lengths as readonly number[]) {
      bound.push((bound.at(-1) as number) + run);
    }
  } else {
    bound.length = 0;
    bound.push(...(offsets as readonly number[]));
  }
  const pieces: TensorClass[] = [];
  for (let i = 0; i < bound.length - 1; i++) {
    const at = bound[i] as number;
    let run = data.narrow(axis, at, (bound[i + 1] as number) - at);
    if (initial !== null) {
      const seed = TensorClass.full(
        run.shape.map((d, k) => (k === axis ? 1 : d)), initial);
      run = TensorClass.cat([seed, run], axis);
    }
    const one = reduce === "sum" ? run.sum(axis)
      : reduce === "prod" ? run.prod(axis)
      : reduce === "mean" ? run.mean(axis)
      : reduce === "max" ? run.amax(axis)
      : run.amin(axis);
    pieces.push(one.unsqueeze(axis));
  }
  return TensorClass.cat(pieces, axis);
}

/**
 * `false`, and **not a placeholder** — torch answers the same on any ordinary
 * build. Absent, a caller's `if (isVulkanAvailable())` stops here and runs there,
 * which is the one direction a compatibility layer cannot afford to be wrong in.
 */
export function isVulkanAvailable(): boolean {
  return false;
}

/** `false`, as torch answers on a build without cuDNN. As `isVulkanAvailable`. */
export function cudnnIsAcceptable(_tensor: TensorClass): boolean {
  return false;
}
