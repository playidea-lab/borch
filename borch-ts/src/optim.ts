/**
 * 옵티마이저와 학습률 스케줄.
 *
 * ## 파라미터를 제자리에서 고친다
 *
 * 새 텐서를 만들어 갈아끼우지 않는다. 모델이 들고 있는 손잡이와 옵티마이저가 보는
 * 손잡이가 같아야 하고, 갈아끼우면 그 둘이 갈려서 **학습은 도는데 파라미터가 안
 * 움직이는** 상태가 된다. 제자리 수정은 기울기가 켜진 잎에서 막혀 있으므로
 * `no_grad` 안에서 한다 — torch 의 옵티마이저도 정확히 그렇게 한다.
 *
 * ## 상태를 텐서로 든다
 *
 * 모멘텀 버퍼 같은 것을 **기울기 텐서 그대로 물고 있으면 안 된다.** 다음 스텝에서
 * 그 기울기가 새것으로 바뀌면 버퍼가 엉뚱한 것을 가리킨다 — 이 저장소가 모멘텀
 * 없는 SGD 만 보다가 놓친 자리다. 그래서 버퍼는 자기 사본으로 시작한다.
 */

import { noGrad, Tensor } from "./tensor.js";

export interface ParamGroup {
  lr: number;
}

export abstract class Optimizer {
  /** torch 와 같은 모양 — 스케줄러가 여기 `lr` 을 고친다. */
  readonly paramGroups: ParamGroup[];

  constructor(protected readonly params: Tensor[], lr: number) {
    this.paramGroups = [{ lr }];
  }

  protected get lr(): number {
    return this.paramGroups[0]?.lr ?? 0;
  }

  /** 기울기를 비운다. **`null` 로 되돌린다** — 0 으로 채우면 잎 판정이 흐려진다. */
  zeroGrad(): void {
    for (const p of this.params) p.grad = null;
  }

  step(): void {
    noGrad(() => {
      for (const [i, p] of this.params.entries()) {
        const g = p.grad;
        if (!g) continue;
        this.update(i, p, g);
      }
    });
  }

  protected abstract update(index: number, param: Tensor, grad: Tensor): void;
}

export class SGD extends Optimizer {
  private readonly buffers = new Map<number, Tensor>();

  constructor(params: Tensor[], lr: number, private readonly momentum = 0) {
    super(params, lr);
  }

  protected override update(index: number, param: Tensor, grad: Tensor): void {
    if (this.momentum === 0) {
      param.copyFrom(param.sub(grad.binary("mul", Tensor.full([], this.lr))));
      return;
    }
    const held = this.buffers.get(index);
    // **첫 스텝에서 기울기를 그대로 물면 안 된다.** 다음 스텝에 그 텐서가 바뀐다.
    const next = held
      ? held.binary("mul", Tensor.full([], this.momentum)).add(grad)
      : grad.clone().detach();
    this.buffers.set(index, next);
    param.copyFrom(param.sub(next.binary("mul", Tensor.full([], this.lr))));
  }
}

export class Adam extends Optimizer {
  private readonly first = new Map<number, Tensor>();
  private readonly second = new Map<number, Tensor>();
  private stepCount = 0;

  constructor(
    params: Tensor[],
    lr: number,
    private readonly beta1 = 0.9,
    private readonly beta2 = 0.999,
    private readonly eps = 1e-8,
  ) {
    super(params, lr);
  }

  override step(): void {
    // 편향 보정이 스텝 수에 걸리므로 파라미터마다가 아니라 한 번만 센다.
    this.stepCount += 1;
    super.step();
  }

  protected override update(index: number, param: Tensor, grad: Tensor): void {
    const m = this.first.get(index) ?? Tensor.zeros(param.shape);
    const v = this.second.get(index) ?? Tensor.zeros(param.shape);
    const mNext = m.binary("mul", Tensor.full([], this.beta1))
      .add(grad.binary("mul", Tensor.full([], 1 - this.beta1)));
    const vNext = v.binary("mul", Tensor.full([], this.beta2))
      .add(grad.square().binary("mul", Tensor.full([], 1 - this.beta2)));
    this.first.set(index, mNext);
    this.second.set(index, vNext);
    const c1 = 1 - this.beta1 ** this.stepCount;
    const c2 = 1 - this.beta2 ** this.stepCount;
    const mHat = mNext.binary("div", Tensor.full([], c1));
    const vHat = vNext.binary("div", Tensor.full([], c2));
    const stepSize = mHat.div(vHat.sqrt().binary("add", Tensor.full([], this.eps)));
    param.copyFrom(param.sub(stepSize.binary("mul", Tensor.full([], this.lr))));
  }
}

export class RMSprop extends Optimizer {
  private readonly squares = new Map<number, Tensor>();

  constructor(
    params: Tensor[],
    lr: number,
    private readonly alpha = 0.99,
    private readonly eps = 1e-8,
  ) {
    super(params, lr);
  }

  protected override update(index: number, param: Tensor, grad: Tensor): void {
    const prev = this.squares.get(index) ?? Tensor.zeros(param.shape);
    const next = prev.binary("mul", Tensor.full([], this.alpha))
      .add(grad.square().binary("mul", Tensor.full([], 1 - this.alpha)));
    this.squares.set(index, next);
    const stepSize = grad.div(next.sqrt().binary("add", Tensor.full([], this.eps)));
    param.copyFrom(param.sub(stepSize.binary("mul", Tensor.full([], this.lr))));
  }
}

/**
 * 학습률 스케줄.
 *
 * **실수 연산뿐이라 torch 와 값이 그대로 같아야 한다** — 근사가 낄 자리가 없다.
 * 그래서 골든이 한 값이 아니라 **궤적 전체**를 굳혔고, 코어가 그렇게 하다가 `StepLR`
 * 의 차이를 잡았다.
 */
export abstract class LRScheduler {
  protected epoch = 0;
  protected readonly base: number;

  constructor(protected readonly opt: Optimizer) {
    this.base = opt.paramGroups[0]?.lr ?? 0;
  }

  step(): void {
    this.epoch += 1;
    const group = this.opt.paramGroups[0];
    if (group) group.lr = this.compute(this.epoch);
  }

  protected abstract compute(epoch: number): number;
}

export class StepLR extends LRScheduler {
  constructor(opt: Optimizer, private readonly stepSize: number,
              private readonly gamma = 0.1) {
    super(opt);
  }

  protected override compute(epoch: number): number {
    return this.base * this.gamma ** Math.floor(epoch / this.stepSize);
  }
}

export class MultiStepLR extends LRScheduler {
  constructor(opt: Optimizer, private readonly milestones: readonly number[],
              private readonly gamma = 0.1) {
    super(opt);
  }

  protected override compute(epoch: number): number {
    const passed = this.milestones.filter((m) => m <= epoch).length;
    return this.base * this.gamma ** passed;
  }
}

export class ExponentialLR extends LRScheduler {
  constructor(opt: Optimizer, private readonly gamma: number) {
    super(opt);
  }

  protected override compute(epoch: number): number {
    return this.base * this.gamma ** epoch;
  }
}

export class CosineAnnealingLR extends LRScheduler {
  constructor(opt: Optimizer, private readonly tMax: number,
              private readonly etaMin = 0) {
    super(opt);
  }

  protected override compute(epoch: number): number {
    return this.etaMin + (this.base - this.etaMin) *
      (1 + Math.cos((Math.PI * epoch) / this.tMax)) / 2;
  }
}

export class LambdaLR extends LRScheduler {
  constructor(opt: Optimizer, private readonly fn: (epoch: number) => number) {
    super(opt);
  }

  protected override compute(epoch: number): number {
    return this.base * this.fn(epoch);
  }
}

/**
 * 값이 나아지지 않으면 학습률을 줄인다.
 *
 * 다른 스케줄과 달리 **값을 받아야** 움직인다 — 그래서 `step(metric)` 이다.
 */
export class ReduceLROnPlateau {
  private best = Infinity;
  private bad = 0;

  constructor(
    private readonly opt: Optimizer,
    private readonly factor = 0.1,
    private readonly patience = 10,
    private readonly threshold = 1e-4,
  ) {}

  step(metric: number): void {
    // torch 의 기본은 `rel` 모드다 — 상대적으로 이만큼은 좋아져야 나아진 것으로 센다.
    if (metric < this.best * (1 - this.threshold)) {
      this.best = metric;
      this.bad = 0;
      return;
    }
    if (this.best === Infinity) this.best = metric;
    this.bad += 1;
    if (this.bad > this.patience) {
      const group = this.opt.paramGroups[0];
      if (group) group.lr *= this.factor;
      this.bad = 0;
    }
  }
}
