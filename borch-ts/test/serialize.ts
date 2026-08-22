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
  decode, encode, init, keepAlive, load, manualSeed, metaToNumbers, nn,
  numbersToMeta, optim, prefixed, save, scope, Tensor, unprefixed,
} from "../src/index.js";

interface Check { name: string; ok: boolean; note: string }

/**
 * 이 보고의 정본은 `checks` 다. `text` 는 사람이 읽는 그림자다.
 *
 * **러너가 문장을 훑어 통과를 판정하고 있었다.** 그 방식은 문구가 바뀌면 조용히
 * 답을 바꾸고, `readme.py` 에서는 실제로 그랬다 — 두 예시 중 하나만 통과해도
 * 찾던 낱말이 다른 줄에 남아 있어 0 을 냈다. 상태를 그대로 넘기면 러너가 셀 수
 * 있고, 무엇이 실패했는지도 제 입으로 말한다.
 */
export interface Report { text: string; checks: Check[] }
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

/**
 * 중첩된 표본 하나. **나무 스킴이 두 벌이 됐고, 이것이 그 둘을 맞대는 유일한 자리다.**
 *
 * `serialize.ts` 와 파이썬 `_serialize.py` 가 같은 마디 종류(`T`/`d`/`l`/`j`)를 같은
 * 글자(`borch.tree`)에 적기로 되어 있는데, 그 약속을 지금까지 아무도 안 쟀다. 한쪽만
 * 고쳐지면 **한쪽이 쓴 체크포인트를 다른 쪽이 못 읽는다** — 그때 나오는 것은 예외가
 * 아니라 구조가 다른 사전이라 더 늦게 들킨다.
 *
 * 위의 `sample` 로는 안 보인다. 그쪽은 최상위가 텐서 사전이라 나무가 있으나 없으나
 * 같은 것이 나온다 — **평평한 것만 물으면 나무는 한 번도 안 밟힌다.**
 */
export async function sampleNested(): Promise<number[]> {
  await init();
  const bytes = await save({
    model: { "fc.weight": Tensor.from([1.5, -2.25], [2]) },
    steps: [Tensor.from([7], [1]), 3],
    epoch: 5,
    note: "nested",
    done: false,
    nothing: null,
  });
  return Array.from(bytes);
}

export async function report(): Promise<Report> {
  await init();
  const data = inputs();

  // ── 코덱 ──────────────────────────────────────────────────────────────
  const original = {
    weight: Tensor.from([1.5, -2.25, 0, 7], [2, 2]),
    labels: Tensor.from([3, 1, 4], [3], { dtype: "int64" }),
    flags: Tensor.from([1, 0], [2], { dtype: "bool" }),
    empty: Tensor.from(new Float32Array(0), [0]),
  };
  // **코덱을 직접 부른다.** `save`/`load` 는 그 위에 나무를 얹은 자리이고, 이 문단이
  // 묻는 것은 밑의 바이트다 — 나무까지 끼면 무엇이 깨졌는지가 흐려진다.
  const bytes = await encode(original, { note: "borch 체크포인트" });
  const back = decode(bytes);

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
    same(new Float32Array((await encode(original)).buffer.slice(0)),
      new Float32Array((await encode(original)).buffer.slice(0))));

  // 깨진 파일은 조용히 이상한 텐서가 되면 안 된다.
  for (const [name, broken] of [
    ["잘린 파일", bytes.subarray(0, 4)],
    ["몸이 모자란 파일", bytes.subarray(0, bytes.length - 4)],
  ] as [string, Uint8Array][]) {
    let threw = false;
    try { decode(broken); } catch { threw = true; }
    want(`${name}을 거절한다`, threw);
  }

  // ── 모델 왕복 ─────────────────────────────────────────────────────────
  const trained = build();
  await train(trained, data, 3);
  const restored = build();
  restored.model.loadStateDict(
    decode(await encode(trained.model.stateDict())).tensors);
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
  const checkpoint = await encode(
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
  const read = decode(checkpoint);
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

  // ── 중첩으로 같은 일 하기 ─────────────────────────────────────────────
  //
  // 위의 재개는 이름에 꼬리표를 붙여 평평하게 눕히고 숫자는 메타데이터로 뺐다. 그것이
  // 이 층이 생기기 전의 유일한 길이었고, **교재에 그렇게 안 적혀 있다** — 교재는
  // `{model: …, opt: …, epoch: 3}` 을 통째로 저장한다.
  //
  // 같은 궤적이 나와야 한다. 안 나오면 나무가 무언가를 흘린 것이고, **여기서 안 물으면
  // 그 흘림은 골든이 못 본다** — 골든은 열쇠 이름과 곁의 값만 보지 학습을 안 잇는다.
  const third = build();
  const early3 = await train(third, data, 5);
  const state3 = third.opt.stateDict();
  const nested = await save({
    model: third.model.stateDict(),
    opt: { tensors: state3.tensors, numbers: state3.numbers },
    sched: third.sched.stateDict(),
    epoch: 5,
    note: "half way",
  });

  const fourth = build();
  const back3 = load(nested) as {
    model: Record<string, Tensor>;
    opt: { tensors: Record<string, Tensor>; numbers: Record<string, number> };
    sched: Record<string, number>;
    epoch: number;
    note: string;
  };
  fourth.model.loadStateDict(back3.model);
  fourth.opt.loadStateDict(back3.opt);
  fourth.sched.loadStateDict(back3.sched);
  const resumed3 = await train(fourth, data, 5);

  const joined3 = [...early3, ...resumed3];
  want("중첩으로 저장한 재개도 통으로 돌린 것과 비트까지 같다",
    joined3.length === straight.length && joined3.every((v, i) => v === straight[i]),
    `${joined3.length} 스텝`);
  // 텐서가 아닌 것도 같이 실린다 — 그것이 평평한 표와 갈리는 자리다.
  want("나무가 숫자와 글자를 같이 나른다",
    back3.epoch === 5 && back3.note === "half way",
    `epoch=${String(back3.epoch)} note=${String(back3.note)}`);

  // **점 찍힌 열쇠를 다시 쪼개면 안 된다.** `stateDict` 의 이름에는 이미 점이 있다
  // (`fc1.weight`). 편 이름을 점으로 되쪼개 되돌리면 값은 다 있는데 구조가 달라진다.
  const dotted = load(await save({ model: third.model.stateDict() })) as
    { model: Record<string, Tensor> };
  want("중첩 안의 점 찍힌 열쇠가 안 쪼개진다",
    Object.keys(dotted.model).includes("fc1.weight"),
    Object.keys(dotted.model).sort().join(" "));

  // 나무가 없는 파일 — 남이 만든 safetensors 다. 평평한 표로 와야 한다.
  const foreign = load(await encode({ w: Tensor.from([1, 2], [2]) })) as
    Record<string, Tensor>;
  want("나무 없는 파일은 평평한 표로 온다",
    foreign.w !== undefined && foreign.w.shape.join(",") === "2",
    Object.keys(foreign).join(","));

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
  return { text: lines.join("\n"), checks };
}
