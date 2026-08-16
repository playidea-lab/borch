/**
 * 체크포인트 — 왕복과 **재개 동등성**.
 *
 * **왕복만 보면 부족하다.** 저장했다 읽어서 값이 같은지는 코덱만 묻는 것이고, 진짜
 * 물음은 그 뒤다: *끊었다 이은 학습이 안 끊고 돌린 학습과 같은가.* 모멘텀 하나,
 * 스텝 계수기 하나, 스케줄러의 에폭 하나만 빠져도 왕복은 초록인 채로 재개만 갈린다.
 *
 * 전부 결정론적이므로 **비트 단위로 같아야 한다.** 허용 오차를 두면 빠뜨린 상태를
 * 오차로 읽게 되므로 여기서는 정확히 같은지만 묻는다.
 */

import {
  init, keepAlive, load, manualSeed, metaToNumbers, nn, numbersToMeta, optim,
  prefixed, save, scope, Tensor, unprefixed,
} from "../src/index.js";

interface Check { name: string; ok: boolean; note: string }
const checks: Check[] = [];

function want(name: string, ok: boolean, note = ""): void {
  checks.push({ name, ok, note });
}

function same(a: Float32Array, b: Float32Array): boolean {
  return a.length === b.length && a.every((v, i) => v === b[i]);
}

class Net extends nn.Module {
  fc1 = new nn.Linear(6, 5);
  fc2 = new nn.Linear(5, 3);

  override forward(x: Tensor): Tensor {
    return this.fc2.call(this.fc1.call(x).relu());
  }
}

/** 같은 씨앗에서 같은 모델·옵티마이저·스케줄러를 세운다. */
function build(): {
  model: Net; opt: optim.Adam; sched: optim.StepLR; crit: nn.CrossEntropyLoss;
} {
  manualSeed(20260817);
  const model = new Net();
  for (const p of model.parameters()) keepAlive(p);
  const opt = new optim.Adam(model.parameters(), 0.05);
  return {
    model, opt,
    sched: new optim.StepLR(opt, 2, 0.5).start(),
    crit: new nn.CrossEntropyLoss(),
  };
}

const BATCH = 8;
const FEATURES = 6;

function inputs(): { x: Tensor; y: Tensor } {
  const pixels = new Float32Array(BATCH * FEATURES);
  for (let i = 0; i < pixels.length; i++) pixels[i] = (i % 11) / 11 - 0.5;
  const labels = new Float32Array(BATCH);
  for (let i = 0; i < BATCH; i++) labels[i] = i % 3;
  return {
    x: keepAlive(Tensor.from(pixels, [BATCH, FEATURES])),
    y: keepAlive(Tensor.from(labels, [BATCH], { dtype: "int64" })),
  };
}

async function train(
  kit: ReturnType<typeof build>, data: ReturnType<typeof inputs>, steps: number,
): Promise<number[]> {
  const seen: number[] = [];
  for (let i = 0; i < steps; i++) {
    await scope(async () => {
      kit.opt.zeroGrad();
      const loss = kit.crit.call(kit.model.call(data.x), data.y);
      loss.backward();
      kit.opt.step();
      seen.push(await loss.item());
    });
    kit.sched.step();
  }
  return seen;
}

/**
 * 파이썬 쪽이 뜯어볼 표본 하나.
 *
 * **이 형식을 고른 이유가 여기서만 증명된다.** 우리 코덱이 우리 코덱과 왕복하는 것은
 * 자체 형식으로도 되고, safetensors 를 든 값어치는 **남이 읽는다**는 데 있다. 러너가
 * 이 바이트를 받아 numpy 로 직접 파싱한다 — borch 코드를 한 줄도 안 쓰고.
 */
export async function sample(): Promise<number[]> {
  await init();
  const bytes = await save({
    "fc.weight": Tensor.from([1.5, -2.25, 0.5, 7, -0.125, 3], [2, 3]),
    "fc.labels": Tensor.from([3, 1, 4], [3], { dtype: "int64" }),
  }, { note: "cross-language" });
  return Array.from(bytes);
}

export async function report(): Promise<string> {
  await init();
  const data = inputs();

  // ── 코덱 ──────────────────────────────────────────────────────────────
  const original = {
    weight: Tensor.from([1.5, -2.25, 0, 7], [2, 2]),
    labels: Tensor.from([3, 1, 4], [3], { dtype: "int64" }),
    flags: Tensor.from([1, 0], [2], { dtype: "bool" }),
    empty: Tensor.from(new Float32Array(0), [0]),
  };
  const bytes = await save(original, { note: "borch 체크포인트" });
  const back = load(bytes);

  want("이름이 그대로 온다",
    Object.keys(back.tensors).sort().join(",") === "empty,flags,labels,weight",
    Object.keys(back.tensors).join(","));
  want("값이 정확히 같다",
    same(await original.weight.toArray(), await back.tensors.weight!.toArray()));
  want("모양이 그대로 온다",
    back.tensors.weight!.shape.join(",") === "2,2");
  // **이름표는 몸에 안 실린다.** 값은 언제나 float32 이고 int64·bool 은 머리에 적힌다 —
  // 안 그러면 4 바이트짜리 몸에 I64 라고 써 놓는 꼴이라 남의 리더가 깨진다.
  want("dtype 이름표가 살아 온다",
    back.tensors.labels!.dtype === "int64" && back.tensors.flags!.dtype === "bool",
    `${back.tensors.labels!.dtype} / ${back.tensors.flags!.dtype}`);
  want("빈 텐서도 왕복한다",
    back.tensors.empty!.size === 0 && back.tensors.empty!.shape.join(",") === "0");
  want("메타데이터가 실린다", back.metadata.note === "borch 체크포인트");

  // safetensors 는 머리 길이를 8 바이트 LE 로 앞에 적고 몸을 8 바이트에서 시작시킨다.
  const headerLength = Number(new DataView(bytes.buffer, bytes.byteOffset)
    .getBigUint64(0, true));
  want("머리가 8 바이트에 맞춰진다", (8 + headerLength) % 8 === 0,
    `머리 ${headerLength}`);
  want("두 번 저장하면 같은 바이트다",
    same(new Float32Array((await save(original)).buffer.slice(0)),
      new Float32Array((await save(original)).buffer.slice(0))));

  // 깨진 파일은 조용히 이상한 텐서가 되면 안 된다.
  for (const [name, broken] of [
    ["잘린 파일", bytes.subarray(0, 4)],
    ["몸이 모자란 파일", bytes.subarray(0, bytes.length - 4)],
  ] as [string, Uint8Array][]) {
    let threw = false;
    try { load(broken); } catch { threw = true; }
    want(`${name}을 거절한다`, threw);
  }

  // ── 모델 왕복 ─────────────────────────────────────────────────────────
  const trained = build();
  await train(trained, data, 3);
  const restored = build();
  restored.model.loadStateDict(load(await save(trained.model.stateDict())).tensors);
  want("모델 state_dict 가 바이트를 건넌다",
    same(await trained.model.parameters()[0]!.toArray(),
      await restored.model.parameters()[0]!.toArray()));

  // ── 재개 동등성 ───────────────────────────────────────────────────────
  // **이것이 이 러너의 이유다.** 열 스텝을 통으로 돌린 것과, 다섯에서 끊었다 이은
  // 것이 같은 손실 궤적을 내야 한다.
  const straightKit = build();
  const straight = await train(straightKit, data, 10);

  const first = build();
  const early = await train(first, data, 5);
  const optState = first.opt.stateDict();
  const checkpoint = await save(
    {
      ...prefixed("model", first.model.stateDict()),
      ...prefixed("opt", optState.tensors),
    },
    {
      ...numbersToMeta("opt", optState.numbers),
      ...numbersToMeta("sched", first.sched.stateDict()),
    },
  );

  const second = build();
  const read = load(checkpoint);
  second.model.loadStateDict(unprefixed("model", read.tensors));
  second.opt.loadStateDict({
    tensors: unprefixed("opt", read.tensors),
    numbers: metaToNumbers("opt", read.metadata),
  });
  second.sched.loadStateDict(metaToNumbers("sched", read.metadata));
  const resumed = await train(second, data, 5);

  const joined = [...early, ...resumed];
  const exact = joined.length === straight.length
    && joined.every((v, i) => v === straight[i]);
  want("끊었다 이은 학습이 통으로 돌린 것과 비트까지 같다", exact,
    exact ? `${straight.length} 스텝`
      : `통 ${straight.slice(4, 7).map((v) => v.toFixed(6)).join(" ")}\n` +
        `      이음 ${joined.slice(4, 7).map((v) => v.toFixed(6)).join(" ")}`);

  // 상태를 하나라도 빠뜨리면 위가 깨져야 한다 — 검사가 실제로 무엇을 재는지 확인한다.
  const naive = build();
  naive.model.loadStateDict(unprefixed("model", read.tensors));   // 가중치만 되돌린다
  const careless = await train(naive, data, 5);
  want("가중치만 되돌리면 궤적이 갈린다",
    !careless.every((v, i) => v === resumed[i]),
    "이 검사가 초록이면 위의 동등성은 아무것도 안 재고 있다");

  // ── 스케줄러 상태 ─────────────────────────────────────────────────────
  // **통으로 돌린 쪽과 비교해야 한다.** 처음에 끊긴 쪽(`first`, 5 스텝)과 이은
  // 쪽(`second`, 합쳐 10 스텝)을 견줬다가 걸렸는데, 둘이 다른 것이 당연했다 —
  // 검사가 스스로 틀린 자리였다.
  want("재개한 스케줄러가 통으로 돌린 것과 같은 학습률에 있다",
    second.opt.paramGroups[0]!.lr === straightKit.opt.paramGroups[0]!.lr,
    `이음 ${second.opt.paramGroups[0]!.lr} / 통 ${straightKit.opt.paramGroups[0]!.lr}`);

  // 무한대는 JSON 이 못 적는다. `ReduceLROnPlateau` 의 `best` 가 거기서 시작한다.
  const plateau = new optim.ReduceLROnPlateau(build().opt);
  const meta = numbersToMeta("p", plateau.stateDict());
  want("무한대가 머리를 건넌다",
    metaToNumbers("p", meta).best === Infinity, meta["p.best"] ?? "(없음)");

  const failed = checks.filter((c) => !c.ok);
  const lines = checks.map((c) =>
    `  ${c.ok ? "✓" : "✗"} ${c.name}${c.note ? ` — ${c.note}` : ""}`);
  lines.push(failed.length === 0
    ? `체크포인트 ${checks.length}건 전부 통과`
    : `**${failed.length}건 실패** / ${checks.length}건`);
  return lines.join("\n");
}
