/**
 * 한 스텝이 **무엇을 얼마나 쓰는가** — 시간이 아니라 **세는 것**으로.
 *
 *     npm run build:ts
 *     npm run cost:ts
 *
 * ## 왜 시간이 아닌가
 *
 * 벤치(`bench.ts`)는 벽시계를 재고, 그래서 **소프트웨어 어댑터에서는 답을 거부한다** —
 * CPU 래스터라이저에서 잰 ms 는 이 라이브러리의 수가 아니라 그 래스터라이저의 수다.
 * 그 판단이 맞고, 그래서 벤치는 GPU 가 있는 기계에서만 뜻이 있다.
 *
 * **여기서 세는 것들은 어댑터와 무관하다.** dispatch 를 몇 번 걸었는가, 구역이 몇
 * 개를 놓았는가, 버퍼를 몇 개 잡고 있는가 — 전부 코드 경로가 정하는 수이고 장치가
 * 바꾸지 않는다. 그래서 **벤치가 못 도는 자리에서 이것은 돈다.** CI 와 이 저장소의
 * 기본 실행 환경이 그 자리다.
 *
 * ## 무엇을 잡는가
 *
 * 골든은 **값만** 본다. 스텝마다 버퍼를 하나씩 흘리는 구현도, 커널을 두 배로 거는
 * 구현도 값은 똑같이 맞으므로 표는 전부 초록이다. 실제로 이 저장소가 그것을
 * 겪었고(`scope()` 가 있는 이유가 그것이다), 지금까지 그 자리를 지키는 검사가
 * 하나도 없었다 — `device.ts` 의 주석이 **"`survived` 가 0 이 아니면 학습 루프에서
 * 그것이 누수다"** 라고 적어 두었는데 그 수를 묻는 곳이 벤치뿐이었고, 벤치는
 * 사람이 손으로 돌리는 것이다.
 *
 * ## 굳힌 수를 고쳐야 할 때
 *
 * `EXPECT` 는 **재서 넣은 수다**(추정이 아니다). 늘었다면 셋 중 하나다 —
 * 커널을 더 걸게 만들었거나, 묶던 것을 안 묶거나, 누수가 생겼다. 줄었다면 좋은
 * 일이고 그때도 여기를 고쳐야 한다. **고칠 때 왜 바뀌었는지 함께 적는다** — 수만
 * 갈아 끼우면 다음 사람은 그 수가 무엇을 뜻하는지 모른 채 또 갈아 끼운다.
 *
 * ## 무는지 확인했다
 *
 * 학습 루프 안에 `keepAlive(loss.mul(loss))` 한 줄을 넣어 스텝마다 버퍼 하나를
 * 흘려 봤다. **세 검사가 각각 다른 각도에서 걸렸다** — dispatch 가 53→54,
 * `survived` 가 0→1, 잡은 버퍼가 26→36. 셋이 같은 것을 세고 있었다면 하나만
 * 빨개졌을 것이므로, 이 겹침은 낭비가 아니라 서로 다른 방식의 새는 자리를 덮는다.
 */

import * as nn from "../src/nn.js";
import { SGD } from "../src/optim.js";
import { device, keepAlive, scope, Tensor } from "../src/tensor.js";

/** 검사 하나. 통과 여부와 **실제로 본 수**를 같이 남긴다. */
interface Check {
  readonly name: string;
  readonly ok: boolean;
  readonly note: string;
}

/**
 * 재는 데 쓰는 모델. **작아야 한다** — 소프트웨어 어댑터에서도 도는 것이 이 검사의
 * 존재 이유다. 그래도 합성곱·정규화·선형·손실을 한 번씩 지나므로, 스텝 하나가
 * 건드리는 자리의 종류는 큰 모델과 같다.
 */
class Small extends nn.Module {
  private readonly conv = new nn.Conv2d(1, 4, 3, 1, 1, false);
  private readonly bn = new nn.BatchNormND(4);
  private readonly fc = new nn.Linear(4 * 8 * 8, 3);

  override forward(x: Tensor): Tensor {
    const h = this.bn.forward(this.conv.forward(x)).unary("relu");
    return this.fc.forward(h.reshape([x.shape[0] ?? 1, 4 * 8 * 8]));
  }
}

/**
 * 스텝 하나가 거는 dispatch 수와 제출 수. **재서 넣은 값이다.**
 *
 * 위의 `Small` 에 배치 4 로 잰 것이고, 모델이나 커널이 바뀌면 같이 바뀐다.
 */
const EXPECT = {
  dispatches: 53,
  // **스텝 하나에 제출이 한 번이다.** 명령을 쌓아 두었다가 손실을 읽을 때 한 번에
  // 보내기 때문이고, 이 수가 오르면 중간에 GPU 를 기다리는 자리가 생겼다는 뜻이다 —
  // 값은 그대로인 채 스텝이 느려지는 종류라 골든이 절대 못 본다.
  submits: 1,
};

export async function report(): Promise<string> {
  const checks: Check[] = [];
  const want = (name: string, ok: boolean, note = ""): void => {
    checks.push({ name, ok, note });
  };

  const dev = device();
  const batch = 4;
  const pixels = new Float32Array(batch * 1 * 8 * 8);
  for (let i = 0; i < pixels.length; i++) pixels[i] = (i % 13) / 13 - 0.5;
  const labels = new Float32Array(batch);
  for (let i = 0; i < batch; i++) labels[i] = i % 3;

  // 입력과 파라미터는 **구역 밖**이다. 안에서 만들면 첫 스텝 끝에 놓인다.
  const x = keepAlive(Tensor.from(pixels, [batch, 1, 8, 8]));
  const y = keepAlive(Tensor.from(labels, [batch], { dtype: "int64" }));
  const model = new Small();
  const opt = new SGD(model.parameters(), 0.05, 0.9);
  const crit = new nn.CrossEntropyLoss();

  /** **쓰는 사람이 칠 그대로.** 저수준 구역을 부르면 `scope()` 를 재는 게 아니다. */
  const step = async (): Promise<number> => scope(async () => {
    opt.zeroGrad();
    const loss = crit.call(model.call(x), y);
    loss.backward();
    opt.step();
    return await loss.item();      // 구역 안에서 읽어야 그 버퍼가 있다
  });

  // 워밍업. 첫 스텝은 셰이더를 굽고 통이 비어 있어서 뒤와 수가 다르다.
  for (let i = 0; i < 3; i++) await step();

  // ── 1. 스텝당 dispatch 수가 **스텝마다 같은가** ────────────────────────
  //
  // 늘어난다면 스텝이 스텝을 보고 있다는 뜻이다 — 그래프가 안 끊기거나, 캐시가
  // 열쇠를 잘못 잡아 셰이더를 다시 굽거나.
  const perStep: number[] = [];
  const perSubmit: number[] = [];
  for (let i = 0; i < 5; i++) {
    const d0 = dev.dispatches;
    const s0 = dev.submits;
    await step();
    perStep.push(dev.dispatches - d0);
    perSubmit.push(dev.submits - s0);
  }
  const first = perStep[0] ?? 0;
  want("스텝마다 dispatch 수가 같다", perStep.every((n) => n === first),
    perStep.join(" "));
  const firstSubmit = perSubmit[0] ?? 0;
  want("스텝마다 제출 수가 같다", perSubmit.every((n) => n === firstSubmit),
    perSubmit.join(" "));

  // ── 2. 굳힌 수와 같은가 ───────────────────────────────────────────────
  if (EXPECT.dispatches > 0) {
    want("스텝당 dispatch 가 굳힌 수와 같다", first === EXPECT.dispatches,
      `${first} (굳힌 것 ${EXPECT.dispatches})`);
    want("스텝당 제출이 굳힌 수와 같다", firstSubmit === EXPECT.submits,
      `${firstSubmit} (굳힌 것 ${EXPECT.submits})`);
  } else {
    // 굳힌 수를 지우고 돌리면 여기로 온다 — 새로 재는 자리다.
    want("스텝당 dispatch 를 아직 안 굳혔다", false,
      `재보니 dispatch ${first} · 제출 ${firstSubmit} — EXPECT 에 적어라`);
  }

  // ── 3. 구역이 아무것도 안 남기는가 ────────────────────────────────────
  //
  // **이것이 누수의 정의다.** `device.ts` 가 그렇게 적어 두었고, 스텝마다 하나씩
  // 남으면 긴 학습에서 장치가 찬다 — 값은 끝까지 맞은 채로.
  await step();
  want("구역이 버퍼를 안 남긴다", dev.lastScope.survived === 0,
    `살아남은 것 ${dev.lastScope.survived} · 놓은 것 ${dev.lastScope.freed}`);
  want("구역이 다 닫혔다", dev.scopeDepth === 0, `깊이 ${dev.scopeDepth}`);

  // ── 4. 잡고 있는 버퍼가 스텝 수와 함께 안 자라는가 ────────────────────
  //
  // 위의 `survived` 는 **한 구역**만 본다. 구역 밖에서 새는 것(예: 전역 캐시가
  // 스텝마다 항목을 늘리는 것)은 그 수에 안 잡히므로 따로 본다.
  const early = dev.memory;
  for (let i = 0; i < 10; i++) await step();
  const late = dev.memory;
  want("스텝을 열 번 더 돌려도 잡은 버퍼가 안 는다",
    late.tensors <= early.tensors,
    `${early.tensors} → ${late.tensors} 개 · ` +
    `${(early.bytes / 1024).toFixed(0)}KB → ${(late.bytes / 1024).toFixed(0)}KB`);

  // ── 5. 통이 실제로 도는가 ─────────────────────────────────────────────
  //
  // **`survived === 0` 만으로는 부족하다.** 매번 새로 만들고 매번 파괴해도 그 수는
  // 0 이다. 그때 값은 맞고 누수도 없는데 할당이 스텝마다 수백 번 돈다.
  // 잡고 있는 버퍼 수가 스텝당 dispatch 수보다 **한참 적으면** 통이 도는 것이다.
  want("버퍼를 스텝마다 새로 만들지 않는다", late.tensors < first,
    `잡은 것 ${late.tensors} 개 · 스텝당 dispatch ${first} 번`);

  const bad = checks.filter((c) => !c.ok);
  const lines = checks.map((c) =>
    `  ${c.ok ? "✓" : "✗"} ${c.name}${c.note ? ` — ${c.note}` : ""}`);
  lines.push("");
  lines.push(bad.length
    ? `**${bad.length}건이 갈렸다.**`
    : `비용 검사 ${checks.length}건 전부 통과`);
  return lines.join("\n");
}
