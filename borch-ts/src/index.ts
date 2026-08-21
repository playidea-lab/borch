/**
 * borch — a PyTorch-shaped tensor library running on WebGPU in the browser.
 *
 * ```ts
 * import { init, Tensor, nn, optim, scope, keepAlive } from "borch";
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
// `using s = scope()` 를 쓰는 쪽이 손잡이의 형을 적을 수 있어야 한다.
export type { Scope } from "./tensor.js";

import { Tensor as TensorClass } from "./tensor.js";

export { Device, isAvailable, probe } from "./device.js";
export type { Availability, DeviceKind, InitOptions } from "./device.js";
// `torch.manual_seed` 자리 — 층 초기화·dropout·`Tensor.randn` 이 한 씨앗에 걸린다.
// `nn.manualSeed` 로도 같은 것이 나온다(옛 이름을 안 깬다).
export { manualSeed } from "./random.js";
export { einsum } from "./einsum.js";
// 대괄호 자리. `x[1:3]` 은 파이썬에서도 `x[slice(1, 3)]` 으로 풀리므로 같은 이름이다.
export { slice } from "./indexing.js";
export type { Slice } from "./indexing.js";
// 체크포인트. **형식은 safetensors 다** — 파이썬 `borch`·numpy·HF 도구가 같은 파일을
// 읽는다. torch 의 `save`/`load` 는 pickle 이라 옮길 수도 옮겨서도 안 된다.
// `save`/`load` 는 **중첩을 그대로** 오간다 — torch·파이썬 `borch` 와 같은 자리다.
// `encode`/`decode` 는 그 밑의 코덱이고, 평평한 텐서 표와 바이트만 다룬다.
export {
  decode, encode, load, metaToNumbers, numbersToMeta, prefixed, save, unprefixed,
} from "./serialize.js";
export type { Bundle, Savable } from "./serialize.js";
// **밖에서 여닫을 수 있어야 한다.** `noGrad(fn)` 은 함수를 받는 모양이라 파이썬의
// `with` 로 옮길 수가 없다 — 결속이 스위치를 직접 쥔다.
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
export type { DType } from "./dtype.js";

// 이름 공간을 나눠 둔다 — `nn` 과 `vision` 둘 다 `manualSeed` 를 갖고 있고,
// torch 도 `torch.nn` · `torch.optim` 으로 갈라 부른다.
// `torch.utils.data` 자리. 이름 공간을 나눠 두는 것은 `nn`·`optim` 과 같은 이유다.
export * as data from "./data.js";
// `torch.fft` 자리. 몸통이 자기 커널(DFT 셰이더)을 갖고 있어 따로 선다.
export * as fft from "./fft.js";
export * as nn from "./nn.js";
export * as optim from "./optim.js";
export * as vision from "./vision.js";

// **`stft`·`istft` 는 최상위다** — torch 가 `torch.stft` 로 둔다. `fft` 이름 공간
// 안에도 같은 함수가 보이지만, 교재 코드가 쓰는 꼴은 최상위 쪽이다.
export { istft, stft } from "./fft.js";
export type { StftOptions } from "./fft.js";

// **최상위 순환 여덟.** torch 는 층(`nn.LSTM`)과 함수(`torch.lstm`)를 둘 다 주고,
// 층이 안에서 부르는 것이 함수 쪽이다 — 가중치를 목록으로 받는 것이 차이다.
export {
  gru, gruCell, lstm, lstmCell, rnnRelu, rnnReluCell, rnnTanh, rnnTanhCell,
} from "./rnn.js";
export type { RnnOptions } from "./rnn.js";

// 특수 함수. `n` 이 셰이더 상수라 단항 표에 안 들어가서 따로 선다.
export { igamma, igammac, polygamma } from "./special.js";
