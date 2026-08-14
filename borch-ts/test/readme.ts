/**
 * README 와 `index.ts` 에 적힌 예시를 **그대로 돌린다.**
 *
 * 문서의 코드는 안 돌리면 썩는다 — 이름이 바뀌어도, 인자 순서가 바뀌어도, `await` 가
 * 하나 빠져도 아무도 안 알려주고 첫 사용자가 거기서 막힌다. 이 저장소는 README 의
 * 설치 안내가 실제로 안 듣던 것을 두 번 잡았다(`1b5a1e9`, `e41c043`).
 *
 * 그래서 여기 있는 것은 예시의 **복사본이 아니라 원본**이어야 한다. 문서를 고치면
 * 이 파일도 같이 고친다. 값이 맞는지는 안 묻는다 — 골든이 그 일을 한다. 여기서 묻는
 * 것은 하나다: **적어 놓은 그대로 쳤을 때 도는가.**
 */

import { init, keepAlive, nn, optim, scope, Tensor } from "../src/index.js";

export async function report(): Promise<string> {
  await init();

  const model = new nn.Sequential(
    new nn.Linear(784, 128), new nn.ReLU(), new nn.Linear(128, 10));
  const opt = new optim.SGD(model.parameters(), 0.05, 0.9);
  const crit = new nn.CrossEntropyLoss();

  // 예시의 `pixels`·`labels` 자리. 값은 아무래도 좋고 모양만 맞으면 된다.
  const pixels = new Float32Array(32 * 784);
  for (let i = 0; i < pixels.length; i++) pixels[i] = (i % 17) / 17 - 0.5;
  const labels = new Float32Array(32);
  for (let i = 0; i < labels.length; i++) labels[i] = i % 10;

  const x = keepAlive(Tensor.from(pixels, [32, 784]));
  const y = keepAlive(Tensor.from(labels, [32], false, "int64"));

  const seen: number[] = [];
  for (let i = 0; i < 3; i++) {
    await scope(async () => {
      opt.zeroGrad();
      const loss = crit.call(model.call(x), y);
      loss.backward();
      opt.step();
      seen.push(await loss.item());
    });
  }

  // **손실이 내려가야 한다.** 도는 것과 배우는 것은 다르고, 예시가 보여주는 것은
  // 후자다 — 세 스텝이 전부 같은 수를 내면 그 예시는 거짓말이다.
  const [first, last] = [seen[0] ?? NaN, seen[seen.length - 1] ?? NaN];
  const ok = Number.isFinite(first) && Number.isFinite(last) && last < first;
  return [
    `README 예시 — 손실 ${seen.map((v) => v.toFixed(4)).join(" → ")}`,
    ok ? "예시가 적힌 그대로 돌고, 손실이 내려간다"
       : "**예시가 돌기는 하는데 손실이 안 내려간다** — 보여줄 것이 못 된다",
  ].join("\n");
}
