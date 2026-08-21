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
  const y = keepAlive(Tensor.from(labels, [32], { dtype: "int64" }));

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

  // ── README 의 `LBFGS` 예시 ────────────────────────────────────────────
  //
  // **이 파일의 첫 문단이 정한 규칙을 내가 어겼다.** "문서를 고치면 이 파일도 같이
  // 고친다" 라고 적혀 있는데, LBFGS 예시를 README 에 넣으면서 여기는 안 건드렸다.
  // 그래서 그 예시는 한 번도 안 돌아 본 채로 문서에 올라가 있었고, **실제로 틀려
  // 있었다** — `new LBFGS([p], …)` 로 파라미터 하나를 최적화하면서 손실은 `model`
  // 로 계산했다. `p` 가 그 모델의 것이 아니면 스텝이 아무것도 안 움직인다.
  //
  // 예외가 안 나는 종류라 읽어서는 안 보인다. 돌려야 보인다.
  const lb = new nn.Sequential(
    new nn.Linear(784, 128), new nn.ReLU(), new nn.Linear(128, 10));
  const lbOpt = new optim.LBFGS(lb.parameters(), 0.1);
  const before = await scope(async () => await crit.call(lb.call(x), y).item());
  await scope(async () => {
    await lbOpt.step(() => {
      lbOpt.zeroGrad();
      const loss = crit.call(lb.call(x), y);
      loss.backward();
      return loss;
    });
  });
  const after = await scope(async () => await crit.call(lb.call(x), y).item());
  const lbOk = Number.isFinite(before) && Number.isFinite(after) && after < before;

  return [
    `README 예시 — 손실 ${seen.map((v) => v.toFixed(4)).join(" → ")}`,
    ok ? "예시가 적힌 그대로 돌고, 손실이 내려간다"
       : "**예시가 돌기는 하는데 손실이 안 내려간다** — 보여줄 것이 못 된다",
    `README LBFGS 예시 — 손실 ${before.toFixed(4)} → ${after.toFixed(4)}`,
    lbOk ? "LBFGS 예시도 적힌 그대로 돌고, 한 스텝에 손실이 내려간다"
         : "**LBFGS 예시가 파라미터를 안 움직인다** — 최적화하는 것과 손실이 보는 것이 다르다",
  ].join("\n");
}
