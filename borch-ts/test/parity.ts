/**
 * torch 를 옮겨 적었을 때 도는가 — 파라미터 등록·파라미터 그룹·난수 팩토리.
 *
 * **골든이 이 셋을 못 잡는다.** 골든은 케이스마다 가중치를 밖에서 넣어 주므로
 * 파라미터가 어떻게 모이는지를 안 묻고, 초기값도 안 본다. 여기서 잡으려는 것은
 * 값이 아니라 **배선**이다 — 특히 첫 번째는 틀려도 예외가 안 나고 학습만 조용히
 * 안 되는 종류라 러너가 없으면 영영 안 보인다.
 */

import {
  device, type DType, init, keepAlive, manualSeed, nn, noGrad, optim, scope,
  slice, Tensor,
} from "../src/index.js";

interface Check { name: string; ok: boolean; note: string }
const checks: Check[] = [];

function want(name: string, ok: boolean, note = ""): void {
  checks.push({ name, ok, note });
}

function near(a: number, b: number, tol: number): boolean {
  return Math.abs(a - b) <= tol;
}

/** 던져야 하는 자리. **안 던지는 것이 실패다** — 조용히 지나가면 값이 틀린다. */
function wantThrow(name: string, fragment: string, body: () => unknown): void {
  try {
    body();
    want(name, false, "안 던졌다");
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    want(name, message.includes(fragment),
      message.includes(fragment) ? "" : `문구가 다르다: ${message}`);
  }
}

function same(a: Float32Array, b: Float32Array): boolean {
  return a.length === b.length && a.every((v, i) => v === b[i]);
}

/** 밖에서 층을 만드는 사람이 쓸 법한 모양. 등록을 손으로 안 적는다. */
class Net extends nn.Module {
  fc1 = new nn.Linear(4, 8);
  fc2 = new nn.Linear(8, 2);

  override forward(x: Tensor): Tensor {
    return this.fc2.call(this.fc1.call(x).relu());
  }
}

/** 텐서 필드가 전부 파라미터인 것은 아니다. */
class WithConstant extends nn.Module {
  weight = Tensor.from([2, 3], [2], { requiresGrad: true });
  mask = Tensor.from([1, 0], [2]);          // 상수 — 옵티마이저가 밟으면 안 된다

  override forward(x: Tensor): Tensor {
    return x.mul(this.weight).mul(this.mask);
  }
}

export async function report(): Promise<string> {
  await init();

  // ── 1. 파라미터 자동 등록 ─────────────────────────────────────────────
  const net = new Net();
  want("자식 층이 필드에서 잡힌다", net.children().length === 2);
  want("파라미터가 전부 모인다", net.parameters().length === 4,
    `${net.parameters().length} 개`);

  const names = Object.keys(net.namedParameters()).sort();
  want("이름이 필드 이름을 쓴다",
    names.join(",") === "fc1.bias,fc1.weight,fc2.bias,fc2.weight", names.join(","));

  const wc = new WithConstant();
  want("requiresGrad 가 파라미터의 표식이다",
    Object.keys(wc.ownParameters()).join(",") === "weight",
    Object.keys(wc.ownParameters()).join(","));

  // ── `nn.Parameter` — **torch 와 갈리는 자리 둘을 값으로 붙잡는다** ────────
  //
  // 갈림을 주석에만 적으면 다음 사람이 그 주석을 안 읽고 고친다. 여기 있으면
  // 고치는 순간 빨개지고, 그때 갈림이 **의도였는지**를 다시 정하게 된다.
  const src = Tensor.from([1, 2, 3], [3]);
  const par = new nn.Parameter(src);
  want("Parameter 가 학습 대상 표식을 세운다", par.requiresGrad);
  // torch 는 원본을 안 건드린다 — 우리도 그렇다.
  want("Parameter 가 원본의 깃발을 안 건드린다", !src.requiresGrad);
  // **저장은 안 나눠 갖는다.** torch 는 나눠 갖지만 우리에게는 뷰가 없다.
  // 파이썬 결속도 같은 선택이라, 두 GPU 구현끼리는 안 갈린다.
  // 둘째 인자로 깃발을 끈다 — 안 끄면 잎에 제자리 쓰기를 못 해서 이 물음 자체가
  // 안 선다. 덤으로 그 인자가 닿는지도 여기서 물어진다.
  await (async () => {
    const p2 = new nn.Parameter(src, false);
    want("Parameter(t, false) 는 깃발을 안 세운다", !p2.requiresGrad);
    p2.copyFrom(Tensor.from([9, 9, 9], [3]));
    const after = await src.toArray();
    want("Parameter 는 저장을 안 나눠 갖는다 — torch 와 갈린다",
      after[0] === 1, `원본 첫 칸 ${after[0]}`);
  })();
  // **평범한 `requiresGrad` 텐서도 파라미터로 센다.** torch 는 `Parameter` 로 감싼
  // 것만 세고 이것은 안 센다(실측) — 규칙을 그쪽으로 바꾸면 `claim()` 으로 세워 둔
  // 지금 코드가 조용히 파라미터를 잃는다.
  class Bare extends nn.Module {
    marked = new nn.Parameter(Tensor.from([1, 1], [2]));
    flagged = Tensor.from([1, 1], [2], { requiresGrad: true });

    override forward(x: Tensor): Tensor {
      return x;
    }
  }
  want("깃발만 세운 텐서도 파라미터로 센다 — torch 와 갈린다",
    Object.keys(new Bare().ownParameters()).sort().join(",") === "flagged,marked",
    Object.keys(new Bare().ownParameters()).sort().join(","));

  // 컨테이너는 자리 번호를 지켜야 한다 — 골든이 그 이름으로 가중치를 넣는다.
  const seq = new nn.Sequential(new nn.Linear(2, 2), new nn.ReLU());
  want("Sequential 은 자리 번호를 지킨다",
    Object.keys(seq.namedParameters()).sort().join(",") === "0.bias,0.weight",
    Object.keys(seq.namedParameters()).join(","));
  const list = new nn.ModuleList([new nn.Linear(2, 2)]);
  want("ModuleList 도 자리 번호를 지킨다",
    Object.keys(list.namedParameters()).sort().join(",") === "0.bias,0.weight",
    Object.keys(list.namedParameters()).join(","));

  // state_dict 왕복. 이름이 갈리면 여기서 걸린다.
  const target = new Net();
  target.loadStateDict(net.stateDict());
  const a0 = await net.parameters()[0]!.toArray();
  const b0 = await target.parameters()[0]!.toArray();
  want("state_dict 왕복이 값을 옮긴다", same(a0, b0));

  // **이것이 진짜 물음이다.** 등록이 새면 손실이 안 내려간다 — 예외는 안 난다.
  const trained = new Net();
  for (const p of trained.parameters()) keepAlive(p);
  const opt = new optim.SGD(trained.parameters(), 0.1);
  const crit = new nn.CrossEntropyLoss();
  const x = keepAlive(Tensor.from(
    Array.from({ length: 8 * 4 }, (_, i) => (i % 7) / 7 - 0.5), [8, 4]));
  const y = keepAlive(Tensor.from([0, 1, 0, 1, 0, 1, 0, 1], [8], { dtype: "int64" }));
  const seen: number[] = [];
  for (let i = 0; i < 5; i++) {
    await scope(async () => {
      opt.zeroGrad();
      const loss = crit.call(trained.call(x), y);
      loss.backward();
      opt.step();
      seen.push(await loss.item());
    });
  }
  want("등록만으로 학습이 돈다",
    (seen[4] ?? NaN) < (seen[0] ?? NaN),
    seen.map((v) => v.toFixed(4)).join(" → "));

  // ── 2. 파라미터 그룹 ──────────────────────────────────────────────────
  const mk = () => keepAlive(Tensor.from([1], [1], { requiresGrad: true }));
  const slow = mk();
  const fast = mk();
  slow.grad = Tensor.from([1], [1]);
  fast.grad = Tensor.from([1], [1]);
  const two = new optim.SGD(
    [{ params: [slow], lr: 0.1 }, { params: [fast], lr: 0.5 }], 0.01);
  want("그룹이 둘로 잡힌다", two.paramGroups.length === 2);
  two.step();
  want("그룹마다 다른 학습률로 움직인다",
    near(await slow.item(), 0.9, 1e-6) && near(await fast.item(), 0.5, 1e-6),
    `${(await slow.item()).toFixed(4)} / ${(await fast.item()).toFixed(4)}`);

  // 그룹에 준 값과 생성자에 준 값이 같은 결과여야 한다. 식은 안 묻고 배선만 묻는다.
  const viaCtor = mk();
  const viaGroup = mk();
  viaCtor.grad = Tensor.from([1], [1]);
  viaGroup.grad = Tensor.from([1], [1]);
  new optim.SGD([viaCtor], 0.1, 0, 0.5).step();
  new optim.SGD([{ params: [viaGroup], weightDecay: 0.5 }], 0.1, 0, 0).step();
  want("그룹별 weightDecay 가 생성자 값과 같게 먹는다",
    near(await viaCtor.item(), await viaGroup.item(), 1e-6),
    `${(await viaCtor.item()).toFixed(6)} / ${(await viaGroup.item()).toFixed(6)}`);

  // 나중에 더한 그룹은 상태 은행까지 늘어야 한다 — 안 늘면 다음 스텝에서 터진다.
  const first = mk();
  const later = mk();
  const adam = new optim.Adam([first], 0.1);
  adam.addParamGroup({ params: [later], lr: 0.1 });
  first.grad = Tensor.from([1], [1]);
  later.grad = Tensor.from([1], [1]);
  let added = true;
  try {
    adam.step();
  } catch (err) {
    added = false;
    want("addParamGroup 뒤 step", false, String(err));
  }
  if (added) {
    want("addParamGroup 이 상태 은행까지 늘린다",
      (await later.item()) !== 1 && (await first.item()) !== 1,
      `${(await first.item()).toFixed(4)} / ${(await later.item()).toFixed(4)}`);
  }

  // 스케줄러는 그룹 전부를 몰아야 하고, 그룹 사이 비율을 지켜야 한다.
  const s1 = mk();
  const s2 = mk();
  const sched = new optim.SGD(
    [{ params: [s1], lr: 0.1 }, { params: [s2], lr: 1.0 }], 0.1);
  const step = new optim.StepLR(sched, 1, 0.1).start();
  step.step();
  want("스케줄러가 모든 그룹을 몰고 비율을 지킨다",
    near(sched.paramGroups[0]!.lr, 0.01, 1e-9)
      && near(sched.paramGroups[1]!.lr, 0.1, 1e-9),
    `${sched.paramGroups[0]!.lr} / ${sched.paramGroups[1]!.lr}`);

  // ── 제자리로 고쳐지는 것은 자기 버퍼를 가져야 한다 ────────────────────
  // 위의 `addParamGroup` 검사가 이것을 잡았다. `Tensor.zeros([1])` 은 값으로 캐시된
  // **전역 상수**를 돌려주는데, 옵티마이저와 이동 통계는 거기에 쓴다.
  // **옵티마이저 전부를 건다.** 이 결함은 한 클래스의 실수가 아니라 `Tensor.zeros`·
  // `Tensor.full` 로 상태를 만드는 습관 전체에 걸린 것이라, 새 옵티마이저가 붙을
  // 때마다 다시 들어온다 — 실제로 `Rprop` 이 그렇게 들어왔다.
  const makers: [string, (p: Tensor) => optim.Optimizer][] = [
    ["SGD(momentum)", (p) => new optim.SGD([p], 0.1, 0.9)],
    ["Adam", (p) => new optim.Adam([p], 0.1)],
    ["RMSprop", (p) => new optim.RMSprop([p], 0.1)],
    ["Adagrad", (p) => new optim.Adagrad([p], 0.1)],
    ["Adadelta", (p) => new optim.Adadelta([p], 0.1)],
    ["Adamax", (p) => new optim.Adamax([p], 0.1)],
    ["NAdam", (p) => new optim.NAdam([p], 0.1)],
    ["RAdam", (p) => new optim.RAdam([p], 0.1)],
    ["ASGD", (p) => new optim.ASGD([p], 0.1)],
    ["Rprop", (p) => new optim.Rprop([p], 0.1)],
    ["Adafactor", (p) => new optim.Adafactor([p], 0.1)],
  ];
  const before = device().faults.count;
  const moved: string[] = [];
  const stuck: string[] = [];
  for (const [label, make] of makers) {
    const p = mk();
    const o = make(p);
    // **두 번 밟는다.** 상태가 살아 있어야 하고, 첫 스텝이 상태를 망가뜨리면 둘째에서
    // 드러난다.
    for (let i = 0; i < 2; i++) {
      p.grad = Tensor.from([1], [1]);
      o.step();
    }
    ((await p.item()) !== 1 ? moved : stuck).push(label);
  }
  want("옵티마이저 전부가 원소 하나짜리 파라미터에서 돈다",
    stuck.length === 0, stuck.length ? `안 움직인 것: ${stuck.join(", ")}` : `${moved.length} 개`);
  want("그 사이 검증 오류가 안 났다", device().faults.count === before,
    device().faults.first);

  // 상태가 캐시를 탔으면 여기가 바뀐다. `0.1` 은 위에서 학습률로 쓴 값이다.
  //
  // **허용 오차가 필요하다.** 0.1 은 float32 로 정확히 안 떨어져서 읽어 오면
  // 0.10000000149… 다. 처음에 1e-9 로 물었다가 그 반올림에 걸렸다 — 오염이 아니라
  // 표현의 문제였고, 검사가 스스로 거짓 경보를 낸 자리다.
  const canary = [
    await Tensor.full([1], 0).item(),
    await Tensor.full([1], 1).item(),
    await Tensor.full([1], 0.1).item(),
  ];
  want("전역 0·1·0.1 상수가 안 더럽혀졌다",
    canary[0] === 0 && canary[1] === 1 && near(canary[2] ?? NaN, 0.1, 1e-6),
    canary.join(" / "));

  // 파라미터 모양이 아닌 상태를 드는 옵티마이저도 그룹이 늘어야 한다.
  const af1 = mk();
  const af2 = Tensor.from([1, 2, 3, 4], [2, 2], { requiresGrad: true });
  keepAlive(af2);
  const af = new optim.Adafactor([af1], 0.1);
  af.addParamGroup({ params: [af2], lr: 0.1 });
  af1.grad = Tensor.from([1], [1]);
  af2.grad = Tensor.from([1, 1, 1, 1], [2, 2]);
  af.step();
  want("Adafactor 가 addParamGroup 뒤에도 돈다",
    (await af1.item()) !== 1 && (await af2.toArray())[0] !== 1);

  const p1 = new nn.PReLU();
  const p2 = new nn.PReLU();
  noGrad(() => { p1.weight.fill_(9); });
  want("PReLU 기본 가중치가 서로 독립이다",
    (await p2.weight.item()) === 0.25, `${await p2.weight.item()}`);
  want("전역 0.25 상수가 안 더럽혀진다",
    (await Tensor.full([1], 0.25).item()) === 0.25);

  const bn = new nn.BatchNormND(1);
  noGrad(() => { bn.runningVar.fill_(5); });
  want("BatchNorm(1) 의 이동 통계가 전역 1 상수와 안 겹친다",
    (await Tensor.ones([1]).item()) === 1, `${await Tensor.ones([1]).item()}`);

  // ── 씨앗을 직접 주는 역방향 ───────────────────────────────────────────
  // 값이 맞는지는 골든이 진짜 torch 와 대조한다(`grad::vjp::*`, 결속 러너가 borch.ts 를
  // 지난다). 여기서 묻는 것은 **TS 표면**이다 — 인자 차례와 거절 문구.
  const leaf = keepAlive(Tensor.from([1, 2, 3], [3], { requiresGrad: true }));
  const out = leaf.mul(leaf);
  out.backward(Tensor.from([1, 10, 100], [3]));
  // d(x²)/dx · v = 2x·v = [2, 40, 600]
  want("씨앗을 준 역방향이 야코비안-벡터 곱을 낸다",
    same(await leaf.grad!.toArray(), Float32Array.from([2, 40, 600])),
    `${Array.from(await leaf.grad!.toArray()).join(",")}`);

  wantThrow("씨앗 없이 비스칼라는 거절한다",
    "grad can be implicitly created only for scalar outputs",
    () => Tensor.from([1, 2], [2], { requiresGrad: true }).backward());
  wantThrow("모양이 어긋난 씨앗을 거절한다", "Mismatch in shape",
    () => Tensor.from([1, 2], [2], { requiresGrad: true })
      .backward(Tensor.from([1, 2, 3], [3])));

  // 둘째 자리가 `retainGraph` 다 — torch 의 인자 차례와 같다.
  const twice = keepAlive(Tensor.from([2], [1], { requiresGrad: true }));
  const held = twice.mul(twice);
  held.backward(Tensor.from([1], [1]), true);
  held.backward(Tensor.from([1], [1]), true);
  want("둘째 자리가 retainGraph 다 — 두 번 흘리면 두 배",
    (await twice.grad!.item()) === 8, `${await twice.grad!.item()}`);

  // ── 대괄호 자리 ───────────────────────────────────────────────────────
  //
  // **`at()` 은 값을 안 만든다.** 전부 `select`·`narrow`·`indexSelect` 로 넘기고,
  // 그 셋은 골든이 이미 진짜 torch 와 대조하고 있다. 그러니 여기서 물을 것은
  // **`at()` 이 위임한 것과 같은 답을 내는가** 하나다 — 그러면 값은 골든에 얹힌다.
  //
  // 값을 손으로 적어 두고 비교하면 그 손이 틀렸을 때 검사가 같이 틀린다.
  const cube = keepAlive(Tensor.from(
    Array.from({ length: 24 }, (_, i) => i), [2, 3, 4]));

  const agrees = async (
    name: string, got: Tensor, expected: Tensor,
  ): Promise<void> => {
    const shapeOk = got.shape.join(",") === expected.shape.join(",");
    want(name, shapeOk && same(await got.toArray(), await expected.toArray()),
      shapeOk ? "" : `모양 [${got.shape}] vs [${expected.shape}]`);
  };

  await agrees("at(0) 은 select 다", cube.at(0), cube.select(0, 0));
  await agrees("at(-1) 은 뒤에서 센다", cube.at(-1), cube.select(0, 1));
  await agrees("at([null, 1]) 은 둘째 축의 select 다",
    cube.at([null, 1]), cube.select(1, 1));
  await agrees("at(slice(1, 3)) 은 narrow 다",
    cube.at(slice(1, 3)), cube.narrow(0, 1, 1));
  await agrees("열린 슬라이스가 끝까지 간다",
    cube.at([null, slice(1)]), cube.narrow(1, 1, 2));
  await agrees("걸음 있는 슬라이스는 indexSelect 로 간다",
    cube.at([null, null, slice(null, null, 2)]),
    cube.indexSelect(2, Tensor.from([0, 2], [2], { dtype: "int64" })));
  await agrees("번호표는 대괄호 둘이다",
    cube.at([[1, 0]]),
    cube.indexSelect(0, Tensor.from([1, 0], [2], { dtype: "int64" })));
  await agrees("텐서 번호표도 받는다",
    cube.at(Tensor.from([1], [1], { dtype: "int64" })),
    cube.indexSelect(0, Tensor.from([1], [1], { dtype: "int64" })));

  // **축 번호가 밀린다.** 정수는 축을 없애므로 둘째 인덱스는 원래 축 1 을 가리키는데,
  // 그때 남은 텐서에서는 그것이 축 0 이다. 이 자리를 안 세면 조용히 다른 축을 자른다.
  await agrees("정수 뒤의 인덱스가 원래 축을 가리킨다",
    cube.at([0, slice(1, 3)]), cube.select(0, 0).narrow(0, 1, 2));
  await agrees("정수 둘이 이어져도 밀림이 맞다",
    cube.at([1, 2]), cube.select(0, 1).select(0, 2));
  want("적게 주면 남은 축은 통째로",
    cube.at(0).shape.join(",") === "3,4", cube.at(0).shape.join(","));

  // 빈 것도 답이다 — 파이썬이 `x[5:99]` 를 빈 것으로 준다.
  want("범위를 넘는 슬라이스는 빈 것이 된다",
    cube.at(slice(5, 99)).shape.join(",") === "0,3,4",
    cube.at(slice(5, 99)).shape.join(","));

  wantThrow("범위를 넘는 정수는 거절한다", "out of bounds", () => cube.at(9));
  wantThrow("축보다 많은 인덱스를 거절한다", "too many indices",
    () => cube.at([0, 0, 0, 0]));
  wantThrow("음수 걸음은 flip 을 가리킨다", "flip()",
    () => slice(0, 3, -1));

  // ── 축약의 형 ─────────────────────────────────────────────────────────
  //
  // 표는 torch 에게 물어 굳혔고 코어 쪽은 `tests/test_reduce_dtype.py` 가 쥔다.
  // 여기는 **borch.ts 쪽 같은 표**다 — 셋이 같은 답을 내야 한다.
  //
  // **int64 와 bool 을 둘 다 묻는다.** int64 만 물으면 "형을 지킨다" 와 "bool 을
  // 올린다" 가 같아 보이고, 절반만 맞는 구현이 통과한다.
  const ints = Tensor.from([3, 1, 4], [3], { dtype: "int64" });
  const flags = Tensor.from([1, 0, 1], [3], { dtype: "bool" });
  const table: [string, DType, DType][] = [
    // 누적 — 값을 만든다. 참·거짓 칸에 3 이 안 들어가므로 bool 이 올라간다.
    ["sum", ints.sum().dtype, flags.sum().dtype],
    ["prod", ints.prod().dtype, flags.prod().dtype],
    ["cumsum", ints.cumsum(0).dtype, flags.cumsum(0).dtype],
    ["cumprod", ints.cumprod(0).dtype, flags.cumprod(0).dtype],
    // 고르기 — 있던 값을 건넨다. 형이 그대로 간다.
    ["amax", ints.amax().dtype, flags.amax().dtype],
    ["amin", ints.amin().dtype, flags.amin().dtype],
    // 고정
    ["any", ints.any().dtype, flags.any().dtype],
    ["all", ints.all().dtype, flags.all().dtype],
    ["countNonzero", ints.countNonzero().dtype, flags.countNonzero().dtype],
    ["argmax", ints.argmax().dtype, flags.argmax().dtype],
    ["logsumexp", ints.logsumexp(0).dtype, flags.logsumexp(0).dtype],
  ];
  const expected: Record<string, [DType, DType]> = {
    sum: ["int64", "int64"], prod: ["int64", "int64"],
    cumsum: ["int64", "int64"], cumprod: ["int64", "int64"],
    amax: ["int64", "bool"], amin: ["int64", "bool"],
    any: ["bool", "bool"], all: ["bool", "bool"],
    countNonzero: ["int64", "int64"], argmax: ["int64", "int64"],
    logsumexp: ["float32", "float32"],
  };
  const wrong = table.filter(([name, i, b]) => {
    const [wi, wb] = expected[name] as [DType, DType];
    return i !== wi || b !== wb;
  });
  want("축약의 형이 torch 표와 같다", wrong.length === 0,
    wrong.map(([n, i, b]) => `${n}: ${i}/${b}`).join(", ") || "11 개");

  // 누적과 고르기가 **갈리는지**를 따로 묻는다. 위의 표가 통째로 한 방향으로 틀려도
  // 이 줄은 살아남아 "둘이 서로 다른 것" 이라는 규칙을 지킨다.
  want("bool 에서 누적과 고르기가 갈린다",
    flags.sum().dtype === "int64" && flags.amax().dtype === "bool",
    `${flags.sum().dtype} / ${flags.amax().dtype}`);

  // 실수만 받는 넷. torch 가 멈추는 자리에서 멈춰야 한다.
  for (const [name, call] of [
    ["mean", () => ints.mean()], ["variance", () => ints.variance()],
    ["std", () => ints.std()], ["norm", () => ints.norm()],
  ] as [string, () => Tensor][]) {
    wantThrow(`${name} 은 정수를 거절한다`, "torch:", call);
  }

  // ── nn.functional ─────────────────────────────────────────────────────
  //
  // **값은 안 만든다.** 전부 `Tensor` 메서드로 넘기므로 골든이 이미 그 값들을 지킨다.
  // 여기서 묻는 것은 **위임한 것과 같은 답을 내는가**, 그리고 **이름으로 이어서는 안
  // 되는 자리가 안 이어졌는가** 둘이다.
  const F = nn.functional;
  const fx = keepAlive(Tensor.from([1, -2, 3, -4], [2, 2]));

  want("nn.functional 이 열린다", typeof F === "object" && F !== null);
  same(await F.relu(fx).toArray(), await fx.relu().toArray())
    ? want("F.relu 가 메서드와 같다", true)
    : want("F.relu 가 메서드와 같다", false);
  want("F.leakyRelu 가 메서드와 같다",
    same(await F.leakyRelu(fx, 0.2).toArray(), await fx.leakyRelu(0.2).toArray()));
  want("F.softmax 가 메서드와 같다",
    same(await F.softmax(fx, 1).toArray(), await fx.softmax(1).toArray()));

  // **이름이 같은데 연산이 다른 자리.** 자동으로 이었으면 조용히 다른 것이 걸린다.
  want("F.batchNorm 은 층의 자유 함수다 — Tensor.batchNorm 이 아니다",
    F.batchNorm.length >= 5, `인자 ${F.batchNorm.length} 개`);
  want("F.unfold 는 im2col 이다 — Tensor.unfold 가 아니다",
    same(await F.unfold(fx.reshape([1, 1, 2, 2]), 2).toArray(),
      await fx.reshape([1, 1, 2, 2]).unfoldIm2col(2).toArray()));
  // huberLoss 는 torch 가 (reduction, delta) 차례라 위치 인자가 뒤바뀐다.
  want("F.huberLoss 가 torch 의 인자 차례를 쓴다",
    same(await F.huberLoss(fx, fx.zerosLike(), "mean", 2).toArray(),
      await fx.huberLoss(fx.zerosLike(), 2, "mean").toArray()));

  // **수신자가 누구인가도 이름의 일부다.** `Tensor.lu_solve` 는 torch 에서 오른쪽
  // 변이 받는다 — 인수가 받도록 두면 이름도 인자 개수도 맞아서 그 자리에서는 안
  // 걸리고 값만 틀린다. 인수가 받는 쪽은 `luSolveFactored` 로 따로 있다.
  const lu = await Tensor.from([4, 3, 6, 3], [2, 2]).luFactor();
  const rhs = Tensor.from([1, 2], [2, 1]);
  const viaMethod = await rhs.luSolve(lu.LU, lu.pivots);
  const viaFactored = await lu.LU.luSolveFactored(lu.pivots, rhs);
  want("lu_solve 의 수신자가 b 다",
    same(await viaMethod.toArray(), await viaFactored.toArray()),
    `${Array.from(await viaMethod.toArray()).join(",")}`);

  // 이어서는 안 되는 것들은 **없어야** 한다. 있으면 조용히 다른 연산이다.
  for (const missing of ["layerNorm", "rmsNorm", "pad", "upsample"]) {
    want(`F.${missing} 은 안 낸다 — 연산이 다르다`,
      (F as Record<string, unknown>)[missing] === undefined);
  }

  // ── 3. 난수 팩토리 ────────────────────────────────────────────────────
  const N = 4096;
  const g = await Tensor.randn([N]).toArray();
  const mean = g.reduce((a, b) => a + b, 0) / N;
  const sd = Math.sqrt(g.reduce((a, b) => a + (b - mean) ** 2, 0) / N);
  want("randn 이 표준정규에 가깝다", near(mean, 0, 0.08) && near(sd, 1, 0.08),
    `평균 ${mean.toFixed(4)}, 표준편차 ${sd.toFixed(4)}`);

  const u = await Tensor.rand([N]).toArray();
  want("rand 가 [0, 1) 안에 든다", u.every((v) => v >= 0 && v < 1));

  const ri = Tensor.randint(3, 7, [N]);
  const riv = await ri.toArray();
  want("randint 가 [low, high) 안의 정수다",
    ri.dtype === "int64"
      && riv.every((v) => Number.isInteger(v) && v >= 3 && v < 7)
      && riv.some((v) => v === 6) && !riv.some((v) => v === 7));

  const perm = Array.from(await Tensor.randperm(64).toArray()).sort((a, b) => a - b);
  want("randperm 이 순열이다", perm.every((v, i) => v === i));

  want("randnLike 가 모양을 빌린다",
    Tensor.zeros([2, 3]).randnLike().shape.join(",") === "2,3");

  // 씨앗 하나가 텐서와 층을 같이 되돌려야 한다.
  manualSeed(7);
  const r1 = await Tensor.randn([8]).toArray();
  manualSeed(7);
  const r2 = await Tensor.randn([8]).toArray();
  want("같은 씨앗이면 randn 이 같다", same(r1, r2));

  manualSeed(11);
  const w1 = await new nn.Linear(3, 2).parameters()[0]!.toArray();
  manualSeed(11);
  const w2 = await new nn.Linear(3, 2).parameters()[0]!.toArray();
  want("같은 씨앗이면 층 초기화도 같다", same(w1, w2));

  // xorshift 는 상태가 0 이면 영원히 0 을 낸다. 씨앗 0 이 난수를 죽이면 안 된다.
  manualSeed(0);
  const z = await Tensor.rand([4]).toArray();
  want("manualSeed(0) 이 난수를 죽이지 않는다", new Set(z).size > 1);

  // **다른 씨앗은 다른 결과를 내야 한다.** 같은 씨앗에 같은 결과만 지키면 절반이다 —
  // dropout 계수기를 늘 1 로 되돌리던 동안 씨앗을 다섯 개 돌려도 마스크는 다섯 번 다
  // 같았고, 그러면 실험 분산이 가중치 초기화 하나에서만 나온다.
  const gpuDraw = async (seed: number): Promise<Float32Array> => {
    manualSeed(seed);
    return Tensor.uniform([16]).toArray();      // GPU 줄기 — dropout 계수기를 쓴다
  };
  want("다른 씨앗이면 GPU 줄기도 달라진다",
    !same(await gpuDraw(1), await gpuDraw(2)));
  want("같은 씨앗이면 GPU 줄기가 같다",
    same(await gpuDraw(7), await gpuDraw(7)));

  // ── 결속이 메꿔 주던 이름들 ─────────────────────────────────────────────
  //
  // **골든은 이 여섯을 구조적으로 못 본다.** 케이스가 전부 `borch_webgpu` 를
  // 지나는데, 그쪽이 텐서 메서드 위에 층을 **스스로 만들어** 놓아서 borch.ts 에
  // 클래스가 없어도 파이썬 쪽은 멀쩡했다. TypeScript 로 `new nn.MSELoss()` 를 쓰는
  // 사람에게만 없는 이름이었다.
  //
  // 여기서 묻는 것은 값이 아니라 **이름이 있는가와 인자가 닿는가**다.
  const lx = () => Tensor.from([0.5, -1, 2, 1.5], [2, 2]);
  const ly = () => Tensor.from([1, 0, -1, 0.5], [2, 2]);
  const label = () => Tensor.from([1, 0], [2], { dtype: "int64" as DType });
  const lossLayers: [string, (r: "none" | "sum") => Tensor][] = [
    ["MSELoss", (r) => new nn.MSELoss(r).call(lx(), ly())],
    ["L1Loss", (r) => new nn.L1Loss(r).call(lx(), ly())],
    ["SmoothL1Loss", (r) => new nn.SmoothL1Loss(r).call(lx(), ly())],
    ["BCEWithLogitsLoss", (r) => new nn.BCEWithLogitsLoss(r).call(
      lx(), Tensor.from([1, 0, 1, 0], [2, 2]))],
    ["NLLLoss", (r) => new nn.NLLLoss(r).call(
      Tensor.from([-1.6, -0.7, -0.5, -1.2], [2, 2]), label())],
    ["CrossEntropyLoss", (r) => new nn.CrossEntropyLoss(r).call(lx(), label())],
  ];
  for (const [name, call] of lossLayers) {
    // `none` 은 접기 전이라 원소가 여럿이고 `sum` 은 스칼라다. **둘의 모양이 달라야**
    // 인자가 실제로 닿은 것이다 — 값을 안 봐도 이것만으로 갈린다.
    want(`nn.${name} 이 있고 reduction 이 닿는다`,
      call("none").size > 1 && call("sum").size === 1,
      `none=${call("none").size} sum=${call("sum").size}`);
  }

  // **결속이 메꾸고 있는 층 이름 열일곱.** 위의 여섯과 같은 갈래인데 아직 안 고쳤다 —
  // 여기서는 **묻기만 한다.** 고치는 것은 `nn.ts` 이고, 묻는 줄을 먼저 붙여 두면
  // 그 사이에 하나가 조용히 생기거나 사라져도 잡힌다.
  //
  // 결속(`borch_webgpu/_nn.py`)이 텐서 메서드 위에 factory 로 만들어 두어서 파이썬
  // 쪽은 멀쩡하다. 골든 케이스는 전부 결속을 지나므로 **이 열일곱은 표가 구조적으로
  // 못 본다** — TypeScript 로 `new nn.GELU()` 를 쓰는 사람에게만 없는 이름이다.
  // **빨간 채로 두지 않는다.** 열일곱이 지금 없는 것은 아는 사실이고, 러너가 늘
  // 빨가면 다음 사람이 러너를 안 읽는다. 그래서 묻는 것은 "다 있는가" 가 아니라
  // **"아는 것 말고 새로 없어진 것이 있는가"** 다. 하나가 채워지면 아래 목록에서
  // 지우면 되고, 안 지워도 초록이 유지된다 — 늘어나는 쪽만 빨개진다.
  // 결속(`borch_webgpu/_nn.py`)이 손으로 쓴 층 factory 마흔일곱 — torch 에 있는 것만.
  const FILLED_IN = [
    "AdaptiveLogSoftmaxWithLoss", "AvgPool2d", "BCEWithLogitsLoss", "Bilinear",
    "CELU", "CTCLoss", "Conv1d", "Conv2d", "Conv3d", "CrossEntropyLoss", "ELU",
    "EmbeddingBag", "Flatten", "FractionalMaxPool2d", "FractionalMaxPool3d",
    "GELU", "GLU", "Hardshrink", "Hardtanh", "Identity", "L1Loss", "LPPool1d",
    "LayerNorm", "LeakyReLU", "Linear", "LogSoftmax", "MSELoss", "ModuleDict",
    "ModuleList", "MultiheadAttention", "NLLLoss", "Parameter", "ParameterDict",
    "ParameterList", "ReLU", "Sequential", "SiLU", "Sigmoid", "SmoothL1Loss",
    "Softmax", "Softmin", "Softplus", "Softshrink", "Tanh", "Threshold",
    "Unflatten", "Upsample",
  ];
  // **지금 없는 것.** 채워지면 여기서 지운다 — 안 지워도 초록이고, **늘어나는 쪽만**
  // 빨개진다. 목록에서 걸러 낸 것을 다시 목록과 대조하면 늘 초록이라 아무것도 안
  // 묻는다(처음에 그렇게 썼다가 고쳤다).
  // **열일곱 전부 채웠다.** 목록이 비어도 검사는 뜻이 있다 — 하나가 사라지면
  // `새로 없어진 것` 으로 빨개진다.
  const KNOWN_ABSENT = new Set<string>();
  const bag = nn as unknown as Record<string, unknown>;
  const absent = FILLED_IN.filter((n) => !(n in bag));
  const surprise = absent.filter((n) => !KNOWN_ABSENT.has(n));
  want("결속이 메꾸는 층 이름에 새 구멍이 없다", surprise.length === 0,
    `아는 것 ${absent.length}/${KNOWN_ABSENT.size}` +
    (surprise.length ? ` · **새로 없어진 것**: ${surprise.join(", ")}` : ""));

  // **`SmoothL1Loss` 의 첫 인자가 코어와 다르다.** 코어는 `(beta, reduction)` 이고
  // 여기는 `(reduction, beta)` 다. torch 자신은 둘 다 이름으로만 받는 자리라
  // "torch 와 갈렸다" 고는 못 하지만, **자매끼리 갈린 것**은 맞다 — 같은 코드를
  // 옮겨 적는 사람이 첫 인자에서 걸린다. 여기 적어 두어 다음에 정리할 때 안 잊는다.
  want("SmoothL1Loss 의 첫 인자는 reduction 이다",
    new nn.SmoothL1Loss("none").call(lx(), ly()).size > 1);

  // **검증 오류가 하나라도 났으면 위의 초록은 못 믿는다.** WebGPU 는 무효한 명령
  // 버퍼를 조용히 버리므로, 값이 안 바뀐 것을 "통과" 로 읽는 검사가 생길 수 있다.
  want("WebGPU 검증 오류가 없다", device().faults.count === 0,
    device().faults.first);

  const failed = checks.filter((c) => !c.ok);
  const lines = checks.map((c) =>
    `  ${c.ok ? "✓" : "✗"} ${c.name}${c.note ? ` — ${c.note}` : ""}`);
  lines.push(failed.length === 0
    ? `torch 배선 ${checks.length}건 전부 통과`
    : `**${failed.length}건 실패** / ${checks.length}건`);
  return lines.join("\n");
}
