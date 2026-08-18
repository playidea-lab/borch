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
 *
 * 반대로 **겹치기만 하던 검사도 하나 있었고 그래서 지웠다** — 아래 "통이 도는가"
 * 자리에 이유가 있다. 같은 검사를 결속 쪽으로 옮겨 보고서야 알았다.
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

  // ── 통이 도는가는 **따로 안 묻는다** ──────────────────────────────────
  //
  // 처음에는 "잡은 버퍼 수가 스텝당 dispatch 수보다 적은가" 를 물었다. 매번 새로
  // 만들고 매번 놓는 구현도 `survived` 는 0 이니 그것만으로는 부족하다는 생각이었다.
  //
  // **결속 쪽에 같은 검사를 옮기고서 그것이 아무것도 새로 안 묻는다는 것을 알았다.**
  // 저쪽은 골든이 먼저 돈 페이지라 시작부터 4 만 개를 잡고 있어서, 절대값 비교가
  // 하네스의 잔여물을 학습 루프의 몫으로 읽었다. 고치려고 뜯어보니 `memory.tensors`
  // 는 `made - spare` 이고 **통에 안 돌려놓는 구현은 `spare` 가 0 이라 이 수가 그냥
  // 자란다** — 바로 위의 검사가 이미 그것을 잡는다. 두 검사가 다른 것을 세는 줄
  // 알았는데 같은 것을 세고 있었다.
  //
  // 다른 자리에서 돌려 보지 않았으면 그 겹침을 몰랐을 것이다. 검사 하나를 지우는
  // 것이 이 실행의 결과다.

  // ── 구역 밖으로 샌 텐서는 **시끄럽게 멈춘다** ─────────────────────────
  //
  // 전에는 조용히 남의 값이 나왔다(실측: `[1,2,3,4]` 가 `9,9,9,9` 로 읽혔다).
  // 버퍼가 파괴되지 않고 통에 돌아가서 다음 할당이 덮어쓰기 때문이고, WebGPU 는
  // 그것을 안 막아 준다 — 유효한 버퍼를 유효하게 읽는 것이니까.
  {
    let escaped: Tensor | null = null;
    await scope(async () => {
      escaped = Tensor.from([1, 2, 3, 4], [4]).mul(Tensor.full([], 1));
      return 0;
    });
    let note = "";
    let stopped = false;
    try {
      const got = await (escaped as unknown as Tensor).toArray();
      note = `조용히 읽혔다: ${Array.from(got).join(",")}`;
    } catch (err) {
      stopped = true;
      note = (err as Error).message.split("\n")[0] ?? "";
    }
    want("샌 텐서를 쓰면 멈춘다", stopped, note);
  }

  // ── 블록 꼴(`using`)이 콜백 꼴과 같은 일을 하는가 ─────────────────────
  //
  // 두 꼴이 같은 `beginScope`/`endScope` 위에 서지만, **그것을 말로만 두면 갈린다.**
  // 여기서 묻는 것은 세 가지다 — 블록을 벗어날 때 닫히는가, 그 시점이 안의 `await`
  // **뒤**인가, 그리고 `keep()` 한 것이 살아남는가.
  {
    const before = dev.scopeDepth;
    let inside = -1;
    let survived: Tensor | null = null;
    let awaited = -1;
    {
      using s = scope();
      inside = dev.scopeDepth;
      survived = s.keep(Tensor.from([5, 6], [2]).mul(Tensor.full([], 1)));
      // **`await` 이 블록 안에 있다.** 놓는 일이 이 기다림보다 먼저 일어나면
      // 여기서 죽은 텐서를 읽게 된다 — 그것이 `using` 으로 되는지의 핵심이다.
      awaited = (await survived.toArray())[0] ?? -1;
    }
    want("using 이 블록 안에서 구역을 연다", inside === before + 1,
      `깊이 ${before} → ${inside}`);
    want("using 이 블록 끝에서 닫는다", dev.scopeDepth === before,
      `깊이 ${dev.scopeDepth}`);
    want("블록 안의 await 이 닫히기 전에 끝난다", awaited === 5, `${awaited}`);
    // 살린 것은 바깥 구역으로 넘어갔으므로 블록 뒤에도 읽힌다. 안 넘겼으면 위에서
    // 만든 죽은 텐서 가드가 여기서 멈춘다 — 그래서 이 줄이 `keep()` 을 묻는다.
    const after = await survived.toArray();
    want("keep() 한 것이 블록 뒤에도 산다", after[1] === 6,
      Array.from(after).join(","));
  }

  const bad = checks.filter((c) => !c.ok);
  const lines = checks.map((c) =>
    `  ${c.ok ? "✓" : "✗"} ${c.name}${c.note ? ` — ${c.note}` : ""}`);
  lines.push("");
  lines.push(bad.length
    ? `**${bad.length}건이 갈렸다.**`
    : `비용 검사 ${checks.length}건 전부 통과`);
  return lines.join("\n");
}
