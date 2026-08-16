/**
 * borch — 브라우저에서 WebGPU 위에 도는 PyTorch 모양의 텐서 라이브러리.
 *
 * ```ts
 * import { init, Tensor, nn, optim, scope, keepAlive } from "borch";
 *
 * await init();                                  // WebGPU 어댑터를 잡는다
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
 *   await scope(async () => {                    // 한 스텝의 중간 버퍼를 놓는다
 *     opt.zeroGrad();
 *     const loss = crit.call(model.call(x), y);
 *     loss.backward();
 *     opt.step();
 *     console.log(await loss.item());
 *   });
 * }
 * ```
 *
 * ## 이 파일이 하는 일
 *
 * 공개 표면을 **한 자리에 모은다.** 이것이 없으면 쓰는 사람이 `dist/src/tensor.js`
 * 같은 안쪽 경로를 직접 짚어야 하고, 그러면 우리가 파일을 옮기는 순간 남의 코드가
 * 깨진다. 여기 적힌 것만 약속이고 나머지는 안쪽 사정이다.
 *
 * ## torch 와 갈리는 자리 — 미리 적는다
 *
 * **`await init()` 을 먼저 불러야 한다.** WebGPU 어댑터를 잡는 것이 비동기라 피할
 * 길이 없다. 안 부르고 텐서를 만들면 그 자리에서 문구와 함께 멈춘다. `torch.cuda.
 * is_available()` 자리는 `await isAvailable()` 이고, 왜 안 되는지까지 알아야 하면
 * `await probe()` 가 `'no-api'`(브라우저가 낡음)와 `'no-adapter'`(드라이버 차단·
 * 헤드리스)를 갈라 준다.
 *
 * **`'cpu'` 는 값이 담긴 그릇이지 연산되는 장치가 아니다.** `await t.cpu()` 로 값을
 * 호스트로 내리고 `t.webgpu()` 로 올린다. 내려온 텐서는 읽을 수는 있어도(`toArray`·
 * `item`·`repr`) 연산에는 못 쓴다 — borch 에 CPU 커널이 없다. 넣으면 torch 와 같은
 * 문구("Expected all tensors to be on the same device")로 멈춘다. `t.device` 가
 * 지금 어디인지 답한다.
 *
 * **값을 읽는 것이 비동기다.** `await t.item()`, `await t.toArray()`. GPU 메모리를
 * 다시 가져오는 일이라 그렇다. 순방향·역방향은 동기다.
 *
 * **구역을 열어야 학습이 돈다.** `scope()` 안에서 만든 중간 버퍼는 나갈 때 놓는다.
 * 자바스크립트의 쓰레기 수집은 GPU 메모리를 제때 안 놓아주므로, 한 스텝이 만드는
 * 수천 개를 그냥 두면 몇 스텝 만에 장치가 찬다. torch 에 없는 개념이고 TF.js 의
 * `tidy` 와 같은 자리다. 파라미터처럼 살아남아야 하는 것은 `keepAlive` 로 표시한다.
 *
 * **모델을 부르는 것은 `model.call(x)` 다.** 자바스크립트는 객체를 그냥 부를 수
 * 없어서 torch 의 `model(x)` 를 그대로 옮길 수 없다. `forward` 를 직접 불러도 같은
 * 값이 나오지만 `call` 쪽을 권한다 — 나중에 훅이 붙는다면 그 자리다.
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

import { Tensor as TensorClass } from "./tensor.js";

export { Device, isAvailable, probe } from "./device.js";
export type { Availability, DeviceKind, InitOptions } from "./device.js";
// `torch.manual_seed` 자리 — 층 초기화·dropout·`Tensor.randn` 이 한 씨앗에 걸린다.
// `nn.manualSeed` 로도 같은 것이 나온다(옛 이름을 안 깬다).
export { manualSeed } from "./random.js";
export { einsum } from "./einsum.js";
// **밖에서 여닫을 수 있어야 한다.** `noGrad(fn)` 은 함수를 받는 모양이라 파이썬의
// `with` 로 옮길 수가 없다 — 결속이 스위치를 직접 쥔다.
export { gradMode } from "./autograd.js";

/**
 * 이것이 텐서인가.
 *
 * 밖에서 이 라이브러리를 감싸는 쪽이 필요로 한다 — 메서드가 텐서를 주는지 수를
 * 주는지 모양을 주는지에 따라 다르게 다뤄야 하는데, `instanceof` 를 쓰려면 클래스를
 * 들여와야 하고 다른 언어에서는 그것이 안 된다. Pyodide 의 파이썬 결속이 첫 사용자다.
 */
export function isTensor(x: unknown): boolean {
  return x instanceof TensorClass;
}
export { IndexError, RuntimeError } from "./errors.js";
export type { DType } from "./dtype.js";

// 이름 공간을 나눠 둔다 — `nn` 과 `vision` 둘 다 `manualSeed` 를 갖고 있고,
// torch 도 `torch.nn` · `torch.optim` 으로 갈라 부른다.
export * as nn from "./nn.js";
export * as optim from "./optim.js";
export * as vision from "./vision.js";
