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
import { device, keepAlive, scope, Tensor } from "../src/tensor.js";

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

  const one = async (): Promise<{ loss: number; freed: number }> => {
    const d = device();
    d.beginScope();
    try {
      opt.zeroGrad();
      const loss = crit.forward(model.forward(x), y);
      loss.backward();
      opt.step();
      // **구역 안에서 읽어야 한다** — 나가면 그 버퍼가 없다.
      return { loss: await loss.item(), freed: 0 };
    } finally {
      d.endScope();
    }
  };

  for (let i = 0; i < warmup; i++) await one();

  const t0 = performance.now();
  let last = 0;
  for (let i = 0; i < steps; i++) last = (await one()).loss;
  const perStep = (performance.now() - t0) / steps;

  // 누수는 **구역이 안 놓은 것**으로 센다. 스텝을 하나 더 돌리고 그 구역이 몇 개를
  // 남겼는지 보는 것이 가장 곧은 측정이다.
  const before = device().scopeDepth;
  await one();
  const leak = device().scopeDepth - before;

  const stepsPerEpoch = Math.ceil(CIFAR_TRAIN_IMAGES / batch);
  return {
    batch,
    params: count,
    msPerStep: Math.round(perStep * 10) / 10,
    epochMin: Math.round((perStep * stepsPerEpoch) / 600) / 100,
    leakPerStep: leak,
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
        `에폭 ${r.epochMin.toFixed(2).padStart(5)}분  ` +
        `누수 ${r.leakPerStep.toFixed(1).padStart(5)}  ` +
        `손실 ${r.loss.toFixed(4)}`,
      );
    } catch (err) {
      lines.push(`batch ${String(b).padStart(3)}  실패: ` +
        `${err instanceof Error ? `${err.constructor.name}: ${err.message.slice(0, 120)}` : String(err)}`);
    }
  }
  return "ResNet-18(CIFAR) · 배치별 실제 학습 스텝\n" + lines.join("\n");
}

/** 구역을 밖에서도 쓸 수 있게 — 러너가 전체를 한 겹 더 감싼다. */
export { scope };
