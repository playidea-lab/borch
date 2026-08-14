/**
 * ResNet-18(CIFAR 판)로 **진짜 학습 스텝**을 잰다.
 *
 * `tests/browser/bench.py` 와 **같은 잣대**다 — 같은 모델, 같은 옵티마이저, 같은
 * 배치, 같은 계산식. 잣대가 다르면 비교가 아니라 두 개의 다른 수다.
 *
 * 층 하나로 잰 값은 BN·ReLU·잔차 덧셈처럼 대역폭에 묶인 연산을 안 센다. "에폭 몇 분"
 * 은 그것들까지 포함한 실제 스텝에서만 나온다 — FLOPs 로 나눈 추정이 아니라.
 */

import * as nn from "../src/nn.js";
import { SGD } from "../src/optim.js";
import { Device } from "../src/device.js";
import { device, keepAlive, noGrad, scope, Tensor } from "../src/tensor.js";

/** 에폭 시간을 내는 데 쓰는 장수. 파이썬 벤치와 같은 수여야 비교가 된다. */
const CIFAR_TRAIN_IMAGES = 50000;

/** ResNet 의 기본 블록. 지름길이 모양을 바꿔야 할 때만 1×1 을 둔다. */
class Block extends nn.Module {
  private readonly conv1: nn.Conv2d;
  private readonly bn1: nn.BatchNormND;
  private readonly conv2: nn.Conv2d;
  private readonly bn2: nn.BatchNormND;
  private readonly down: { conv: nn.Conv2d; bn: nn.BatchNormND } | null;

  constructor(cin: number, cout: number, stride: number) {
    super();
    this.conv1 = new nn.Conv2d(cin, cout, 3, stride, 1, false);
    this.bn1 = new nn.BatchNormND(cout);
    this.conv2 = new nn.Conv2d(cout, cout, 3, 1, 1, false);
    this.bn2 = new nn.BatchNormND(cout);
    this.down = stride !== 1 || cin !== cout
      ? { conv: new nn.Conv2d(cin, cout, 1, stride, 0, false), bn: new nn.BatchNormND(cout) }
      : null;
  }

  override children(): nn.Module[] {
    const kids: nn.Module[] = [this.conv1, this.bn1, this.conv2, this.bn2];
    if (this.down) kids.push(this.down.conv, this.down.bn);
    return kids;
  }

  override forward(x: Tensor): Tensor {
    let out = this.bn1.forward(this.conv1.forward(x)).unary("relu");
    out = this.bn2.forward(this.conv2.forward(out));
    const side = this.down
      ? this.down.bn.forward(this.down.conv.forward(x))
      : x;
    return out.add(side).unary("relu");
  }
}

/**
 * CIFAR 판 — 3×3 스템에 맥스풀이 없다. 32×32 를 7×7 스템으로 받으면 너무 줄어든다.
 */
class ResNet18 extends nn.Module {
  private readonly stem: nn.Conv2d;
  private readonly bn: nn.BatchNormND;
  private readonly body: nn.Sequential;
  private readonly fc: nn.Linear;

  constructor(classes = 10) {
    super();
    this.stem = new nn.Conv2d(3, 64, 3, 1, 1, false);
    this.bn = new nn.BatchNormND(64);
    this.body = new nn.Sequential([
      new Block(64, 64, 1), new Block(64, 64, 1),
      new Block(64, 128, 2), new Block(128, 128, 1),
      new Block(128, 256, 2), new Block(256, 256, 1),
      new Block(256, 512, 2), new Block(512, 512, 1),
    ]);
    this.fc = new nn.Linear(512, classes);
  }

  override children(): nn.Module[] {
    return [this.stem, this.bn, this.body, this.fc];
  }

  override forward(x: Tensor): Tensor {
    let h = this.bn.forward(this.stem.forward(x)).unary("relu");
    h = this.body.forward(h).adaptiveAvgPool(1);
    return this.fc.forward(h.reshape([h.shape[0] ?? 1, 512]));
  }
}

export interface StepResult {
  batch: number;
  params: number;
  msPerStep: number;
  epochMin: number;
  /** 스텝당 안 놓인 버퍼 수. 0 이 아니면 새는 것이다. */
  leakPerStep: number;
  /** 스텝당 dispatch 수. 느린 것이 커널인지 부르는 횟수인지 가른다. */
  dispatches: number;
  /** dispatch 하나당 벽시계. 커널 시간이 아니라 **부르는 값**의 상한이다. */
  usPerDispatch: number;
  /** 커널 종류별 dispatch 수, 많은 것부터. 다음에 무엇을 고칠지가 여기서 나온다. */
  kinds: [string, number][];
  /** 스텝당 실제 제출 수. dispatch 수와 갈리는 만큼이 묶인 것이다. */
  submits: number;
  /** 순방향만 돌렸을 때의 벽시계. 나머지가 역방향과 옵티마이저의 몫이다. */
  msForward: number;
  /** 가중치 기울기를 끈 스텝. 전체에서 빼면 `gradWeight` 의 몫이다. */
  msNoWeightGrad: number;
  loss: number;
}

/**
 * 한 스텝의 벽시계 시간.
 *
 * **워밍업 뒤에 기준선을 잡는다.** 안 그러면 첫 스텝의 셰이더 컴파일이 측정 창에
 * 들어와 스텝당 비용으로 둔갑한다 — 모양마다 셰이더를 굽는 설계라 그 값이 크다.
 */
export async function runStep(
  batch = 32, steps = 5, warmup = 2,
): Promise<StepResult> {
  // 입력은 **구역 밖**에서 만든다. 안에서 만들면 첫 스텝이 끝날 때 놓인다.
  const rng = { s: 12345 };
  const next = (): number => {
    let x = rng.s; x ^= x << 13; x >>>= 0; x ^= x >> 17; x ^= x << 5; x >>>= 0;
    rng.s = x;
    return x / 0x100000000;
  };
  const pixels = new Float32Array(batch * 3 * 32 * 32);
  for (let i = 0; i < pixels.length; i++) pixels[i] = next() * 2 - 1;
  const labels = new Float32Array(batch);
  for (let i = 0; i < batch; i++) labels[i] = Math.floor(next() * 10);
  const x = keepAlive(Tensor.from(pixels, [batch, 3, 32, 32]));
  const y = keepAlive(Tensor.from(labels, [batch], false, "int64"));

  const model = new ResNet18();
  const params = model.parameters();
  const count = params.reduce((n, p) => n + p.size, 0);
  const opt = new SGD(params, 0.05, 0.9);
  const crit = new nn.CrossEntropyLoss();

  let survived = 0;
  const one = async (): Promise<number> => {
    const d = device();
    d.beginScope();
    let loss = 0;
    try {
      opt.zeroGrad();
      const out = crit.forward(model.forward(x), y);
      out.backward();
      opt.step();
      // **구역 안에서 읽어야 한다** — 나가면 그 버퍼가 없다.
      loss = await out.item();
    } finally {
      survived = d.endScope().survived;
    }
    return loss;
  };

  for (let i = 0; i < warmup; i++) await one();

  const t0 = performance.now();
  const d0 = device().dispatches;
  const s0 = device().submits;
  const k0 = new Map(device().byKind);
  let last = 0;
  for (let i = 0; i < steps; i++) last = await one();
  const perStep = (performance.now() - t0) / steps;
  const perStepDispatches = (device().dispatches - d0) / steps;
  const perStepSubmits = (device().submits - s0) / steps;
  // 종류별로 갈라 둔다 — 총수만으로는 다음에 무엇을 고칠지 안 나온다.
  const kinds: [string, number][] = [];
  for (const [kind, n] of device().byKind) {
    const grew = (n - (k0.get(kind) ?? 0)) / steps;
    if (grew > 0) kinds.push([kind, Math.round(grew)]);
  }
  kinds.sort((a, b) => b[1] - a[1]);

  // 누수는 **구역이 안 놓고 내보낸 버퍼 수**다.
  //
  // 처음에는 열린 구역의 깊이를 뺐는데, 그건 구역 밖에서 늘 0 이라 언제나 0 을 낸다 —
  // 이름은 누수인데 재는 것이 아무것도 없는 계측이다. 파이썬 벤치가 같은 함정을
  // 적어두었고("이름이 말하는 것과 다른 수를 내는 계측은 진짜 누수를 만났을 때도
  // 못 믿는다") 나는 그것을 읽고도 같은 것을 만들었다.
  await one();
  const leak = survived;

  /**
   * 순방향만.
   *
   * **역방향의 몫을 추측하지 않으려고 잰다.** 전체에서 이것을 빼면 남는 것이
   * 역방향과 옵티마이저이고, 그 둘 중 어느 쪽을 먼저 고칠지가 거기서 정해진다.
   * 손실을 읽는 것까지 같이 해야 GPU 를 기다리는 지점이 같다.
   */
  const forwardOnly = async (): Promise<void> => {
    const d = device();
    d.beginScope();
    try {
      // `noGrad` 는 **동기** 본문을 받는다 — 비동기를 넣으면 기울기 스위치가 읽기보다
      // 먼저 되돌아간다. 그래프 세우기만 감싸고 읽기는 밖에서 기다린다.
      const out = noGrad(() => crit.forward(model.forward(x), y));
      await out.item();
    } finally {
      d.endScope();
    }
  };
  await forwardOnly();
  const f0 = performance.now();
  for (let i = 0; i < steps; i++) await forwardOnly();
  const perForward = (performance.now() - f0) / steps;

  /**
   * 가중치 기울기를 끄고 한 스텝.
   *
   * **어느 커널이 무거운지를 추측 대신 뺄셈으로 낸다.** `gradWeight` 는 출력이 작고
   * 축약이 큰 GEMM 이라 층에 따라 워크그룹이 한 개까지 떨어진다 — 그것이 실제로
   * 비싼지는 켜고 끈 차이로만 알 수 있다. 학습은 이 동안 틀리지만 재는 것은 시간이다.
   */
  const convWeights = params.filter((p) => p.shape.length >= 3);
  for (const p of convWeights) p.requiresGrad = false;
  await one();
  const w0 = performance.now();
  for (let i = 0; i < steps; i++) await one();
  const perNoWeightGrad = (performance.now() - w0) / steps;
  for (const p of convWeights) p.requiresGrad = true;

  // **검증 오류가 하나라도 났으면 수를 안 낸다.**
  //
  // 무효한 명령 버퍼는 예외를 안 던지고 그냥 아무것도 안 한다. 그 상태에서도
  // 벽시계는 돌아서 ms/step 이 나오는데, 그건 측정이 아니라 학습이 안 되는 상태의
  // 벽시계다. 실제로 그렇게 됐다 — 평균 풀링의 역방향이 통째로 무효였고 손실이
  // 2.27 에서 안 움직이는 채로 "13070 ms/step" 이 나왔다.
  const faults = device().faults;
  if (faults.count > 0) {
    throw new Error(
      `WebGPU 검증 오류 ${faults.count}건이 났다 — 이 수는 측정이 아니다.\n` +
        `첫 건: ${faults.first}`,
    );
  }

  const stepsPerEpoch = Math.ceil(CIFAR_TRAIN_IMAGES / batch);
  return {
    batch,
    params: count,
    msPerStep: Math.round(perStep * 10) / 10,
    epochMin: Math.round((perStep * stepsPerEpoch) / 600) / 100,
    leakPerStep: leak,
    dispatches: Math.round(perStepDispatches),
    usPerDispatch: Math.round((perStep * 1000) / Math.max(1, perStepDispatches)),
    kinds,
    submits: Math.round(perStepSubmits),
    msForward: Math.round(perForward * 10) / 10,
    msNoWeightGrad: Math.round(perNoWeightGrad * 10) / 10,
    loss: Math.round(last * 10000) / 10000,
  };
}

/** 파이썬 벤치와 같은 배치들을 같은 차례로. */
export async function report(batches: readonly number[] = [16, 32, 64]): Promise<string> {
  const lines: string[] = [];
  for (const b of batches) {
    try {
      const r = await runStep(b);
      lines.push(
        `batch ${String(r.batch).padStart(3)}  ` +
        `${r.msPerStep.toFixed(1).padStart(8)} ms/step  ` +
        `(순방향 ${r.msForward.toFixed(1).padStart(7)} · dW뺀것 ` +
        `${r.msNoWeightGrad.toFixed(1).padStart(7)})  ` +
        `에폭 ${r.epochMin.toFixed(2).padStart(6)}분  ` +
        `dispatch ${String(r.dispatches).padStart(5)}  ` +
        `제출 ${String(r.submits).padStart(3)}  ` +
        `${String(r.usPerDispatch).padStart(5)}µs/dispatch  ` +
        `누수 ${r.leakPerStep.toFixed(1).padStart(5)}  ` +
        `손실 ${r.loss.toFixed(4)}`,
      );
      // 종류별 내역은 배치마다 같으므로 한 번만 찍는다.
      if (b === batches[0]) {
        const top = r.kinds.slice(0, 8)
          .map(([kind, n]) => `${kind} ${n}`).join(" · ");
        lines.push(`         dispatch 내역: ${top}`);
      }
    } catch (err) {
      lines.push(`batch ${String(b).padStart(3)}  실패: ` +
        `${err instanceof Error ? `${err.constructor.name}: ${err.message.slice(0, 120)}` : String(err)}`);
    }
  }
  // **어느 장치에서 잰 것인지가 수보다 먼저 온다.** 헤드리스 브라우저가 소프트웨어
  // 어댑터를 주면 예외 없이 느린 수가 나오고, 그것은 라이브러리의 성적이 아니다.
  return `ResNet-18(CIFAR) · 배치별 실제 학습 스텝\n어댑터: ${Device.adapterInfo}\n`
    + lines.join("\n");
}

/** 구역을 밖에서도 쓸 수 있게 — 러너가 전체를 한 겹 더 감싼다. */
export { scope };
