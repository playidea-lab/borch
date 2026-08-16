/**
 * torch 를 옮겨 적었을 때 도는가 — 파라미터 등록·파라미터 그룹·난수 팩토리.
 *
 * **골든이 이 셋을 못 잡는다.** 골든은 케이스마다 가중치를 밖에서 넣어 주므로
 * 파라미터가 어떻게 모이는지를 안 묻고, 초기값도 안 본다. 여기서 잡으려는 것은
 * 값이 아니라 **배선**이다 — 특히 첫 번째는 틀려도 예외가 안 나고 학습만 조용히
 * 안 되는 종류라 러너가 없으면 영영 안 보인다.
 */

import {
  device, init, keepAlive, manualSeed, nn, noGrad, optim, scope, Tensor,
} from "../src/index.js";

interface Check { name: string; ok: boolean; note: string }
const checks: Check[] = [];

function want(name: string, ok: boolean, note = ""): void {
  checks.push({ name, ok, note });
}

function near(a: number, b: number, tol: number): boolean {
  return Math.abs(a - b) <= tol;
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
  const scalar = mk();
  scalar.grad = Tensor.from([1], [1]);
  new optim.Adam([scalar], 0.1).step();
  want("Adam 이 원소 하나짜리 파라미터에서 돈다", (await scalar.item()) !== 1,
    `${await scalar.item()}`);

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
