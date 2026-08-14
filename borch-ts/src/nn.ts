/**
 * 층 — `nn.Module` 의 최소 뼈대.
 *
 * ## 무엇을 위한 것인가
 *
 * 단위 대조는 연산 하나씩만 본다. 모듈·손실·옵티마이저가 엮여야만 갈리는 것이 있고,
 * 이 저장소가 통합 시나리오에서 잡은 결함들은 전부 그 자리에서 나왔다. 그래서
 * 층을 세우는 것은 편의가 아니라 **볼 수 없던 자리를 보는 일**이다.
 *
 * ## 이름이 곧 계약이다
 *
 * `state_dict` 의 열쇠가 `0.weight` 처럼 **자리 번호와 이름**으로 만들어진다. 골든이
 * 그 이름으로 가중치를 넣고 그 이름으로 결과를 꺼내므로, 이름이 갈리면 값이 아니라
 * 배선이 갈린다.
 */

import { noGrad, Tensor } from "./tensor.js";

/** 층 하나. 값을 지나가게 하고, 자기 파라미터를 이름과 함께 내놓는다. */
export abstract class Module {
  /** 학습 모드인가. `BatchNorm` 처럼 모드에 따라 다르게 구는 층이 본다. */
  training = true;

  abstract forward(x: Tensor): Tensor;

  /** 부를 수 있게 — torch 의 `model(x)` 와 같은 자리다. */
  call(x: Tensor): Tensor {
    return this.forward(x);
  }

  /** 이 층이 직접 가진 파라미터. 자식은 `children` 이 준다. */
  ownParameters(): Record<string, Tensor> {
    return {};
  }

  children(): Module[] {
    return [];
  }

  /** `0.weight` 처럼 자리 번호를 앞에 붙인 이름표. */
  namedParameters(prefix = ""): Record<string, Tensor> {
    const out: Record<string, Tensor> = {};
    for (const [name, p] of Object.entries(this.ownParameters())) {
      out[`${prefix}${name}`] = p;
    }
    for (const [i, child] of this.children().entries()) {
      Object.assign(out, child.namedParameters(`${prefix}${i}.`));
    }
    return out;
  }

  parameters(): Tensor[] {
    return Object.values(this.namedParameters());
  }

  /**
   * 밖에서 가중치를 넣는다.
   *
   * **값만 옮기고 텐서를 바꿔치지 않는다.** 바꿔치면 옵티마이저가 들고 있던 것과
   * 다른 텐서가 되어, 학습이 도는데 파라미터가 안 움직이는 상태가 된다.
   */
  loadStateDict(values: Readonly<Record<string, Tensor>>, strict = true): void {
    const own = this.namedParameters();
    for (const [name, src] of Object.entries(values)) {
      const dst = own[name];
      if (!dst) {
        if (strict) throw new Error(`load_state_dict: 모르는 이름 '${name}'`);
        continue;
      }
      noGrad(() => dst.copyFrom(src));
    }
  }

  stateDict(): Record<string, Tensor> {
    return this.namedParameters();
  }

  train(mode = true): this {
    this.training = mode;
    for (const c of this.children()) c.train(mode);
    return this;
  }

  eval(): this {
    return this.train(false);
  }
}

/** 층을 줄줄이 세운 것. 자리 번호가 곧 이름이다. */
export class Sequential extends Module {
  constructor(private readonly layers: Module[]) {
    super();
  }

  override children(): Module[] {
    return this.layers;
  }

  override forward(x: Tensor): Tensor {
    let cur = x;
    for (const layer of this.layers) cur = layer.forward(cur);
    return cur;
  }
}

/** `y = x·Wᵀ + b`. 가중치는 `(출력, 입력)` 이다 — torch 와 같다. */
export class Linear extends Module {
  readonly weight: Tensor;
  readonly bias: Tensor;

  constructor(inFeatures: number, outFeatures: number) {
    super();
    // 골든은 가중치를 밖에서 넣는다. 여기 초기값이 무엇이든 덮어쓰이지만,
    // 안 넣고 쓰는 경우를 위해 0 이 아닌 값을 둔다.
    this.weight = Tensor.zeros([outFeatures, inFeatures]);
    this.bias = Tensor.zeros([outFeatures]);
    this.weight.requiresGrad = true;
    this.bias.requiresGrad = true;
  }

  override ownParameters(): Record<string, Tensor> {
    return { weight: this.weight, bias: this.bias };
  }

  override forward(x: Tensor): Tensor {
    return x.linear(this.weight).add(this.bias);
  }
}

/**
 * 차원 수에 상관없는 합성곱 층. `spatial` 이 공간 축의 수다.
 *
 * `Conv1d`·`Conv2d`·`Conv3d` 가 그 수만 다르다 — 클래스를 셋 세우면 그중 하나만
 * 고치는 날이 온다.
 */
export class ConvND extends Module {
  readonly weight: Tensor;
  readonly bias: Tensor;

  constructor(
    inChannels: number,
    outChannels: number,
    kernel: number,
    spatial: number,
    private readonly stride = 1,
    private readonly padding = 0,
  ) {
    super();
    this.weight = Tensor.zeros([
      outChannels, inChannels, ...new Array<number>(spatial).fill(kernel),
    ]);
    this.bias = Tensor.zeros([outChannels]);
    this.weight.requiresGrad = true;
    this.bias.requiresGrad = true;
  }

  override ownParameters(): Record<string, Tensor> {
    return { weight: this.weight, bias: this.bias };
  }

  override forward(x: Tensor): Tensor {
    return x.convND(this.weight, this.bias, this.stride, this.padding);
  }
}

export class Conv1d extends ConvND {
  constructor(inC: number, outC: number, kernel: number, stride = 1, padding = 0) {
    super(inC, outC, kernel, 1, stride, padding);
  }
}

export class Conv2d extends ConvND {
  constructor(inC: number, outC: number, kernel: number, stride = 1, padding = 0) {
    super(inC, outC, kernel, 2, stride, padding);
  }
}

export class Conv3d extends ConvND {
  constructor(inC: number, outC: number, kernel: number, stride = 1, padding = 0) {
    super(inC, outC, kernel, 3, stride, padding);
  }
}

export class ReLU extends Module {
  override forward(x: Tensor): Tensor {
    return x.unary("relu");
  }
}

export class MaxPool2d extends Module {
  constructor(private readonly kernel = 2) {
    super();
  }

  override forward(x: Tensor): Tensor {
    return x.maxPool2d(this.kernel);
  }
}

/** 배치 축만 남기고 납작하게. */
export class Flatten extends Module {
  override forward(x: Tensor): Tensor {
    const batch = x.shape[0] ?? 1;
    return x.reshape([batch, x.size / batch]);
  }
}

/**
 * 채널마다 정규화. **학습 모드와 평가 모드가 다르다.**
 *
 * 학습 때는 이 배치의 통계로 정규화하면서 이동 통계를 갱신하고, 평가 때는 이동
 * 통계를 쓴다. 저장하고 복원한 뒤 평가 모드로 넘어가는 경로에서만 그 차이가
 * 드러나는데, 코어가 겪은 결함이 정확히 거기였다 — 이동 통계가 `state_dict` 에서
 * 빠져서 학습은 멀쩡하고 추론만 틀렸다.
 */
/**
 * 차원 수에 상관없는 배치 정규화. `BatchNorm1d`·`2d`·`3d` 가 전부 이것이다.
 *
 * 접는 축이 배치와 공간 전부이고 채널만 남는다 — 공간 축이 몇 개든 규칙이 같아서
 * 차원마다 클래스를 세울 이유가 없다.
 */
export class BatchNormND extends Module {
  readonly weight: Tensor;
  readonly bias: Tensor;
  readonly runningMean: Tensor;
  readonly runningVar: Tensor;

  constructor(
    private readonly channels: number,
    private readonly eps = 1e-5,
    private readonly momentum = 0.1,
  ) {
    super();
    this.weight = Tensor.ones([channels]);
    this.bias = Tensor.zeros([channels]);
    this.weight.requiresGrad = true;
    this.bias.requiresGrad = true;
    this.runningMean = Tensor.zeros([channels]);
    this.runningVar = Tensor.ones([channels]);
  }

  override ownParameters(): Record<string, Tensor> {
    return { weight: this.weight, bias: this.bias };
  }

  /** **이동 통계도 함께 나간다.** 빠지면 평가 모드에서만 조용히 틀린다. */
  override stateDict(): Record<string, Tensor> {
    return {
      ...this.namedParameters(),
      running_mean: this.runningMean,
      running_var: this.runningVar,
    };
  }

  override loadStateDict(
    values: Readonly<Record<string, Tensor>>,
    strict = true,
  ): void {
    const rest: Record<string, Tensor> = {};
    for (const [name, src] of Object.entries(values)) {
      if (name === "running_mean") noGrad(() => this.runningMean.copyFrom(src));
      else if (name === "running_var") noGrad(() => this.runningVar.copyFrom(src));
      else rest[name] = src;
    }
    super.loadStateDict(rest, strict);
  }

  override forward(x: Tensor): Tensor {
    const spatial = x.shape.length - 2;
    // 채널만 남기고 나머지를 1 로. 공간 축이 몇 개든 이 한 줄이 맞춰 준다.
    const shape = [1, this.channels, ...new Array<number>(spatial).fill(1)];
    const w = this.weight.reshape(shape);
    const b = this.bias.reshape(shape);
    if (!this.training) {
      const centered = x.sub(this.runningMean.reshape(shape));
      const scaled = centered.div(
        this.runningVar.reshape(shape).binary("add", Tensor.full([], this.eps)).sqrt());
      return scaled.mul(w).add(b);
    }
    const count = x.size / this.channels;
    const perChannel = (t: Tensor): Tensor => {
      // 배치를 접고, 남은 공간 축을 하나씩 접는다. 접을 때마다 채널이 앞으로 온다.
      let acc = t.sumDim(0);
      for (let d = 0; d < spatial; d++) acc = acc.sumDim(1);
      return acc.reshape(shape);
    };
    const mean = perChannel(x).div(Tensor.full([], count));
    const centered = x.sub(mean);
    const biased = perChannel(centered.square()).div(Tensor.full([], count));
    // 이동 통계에는 **불편추정**이 들어간다 — torch 가 그렇다. 정규화에 쓰는 것은
    // 편향추정이라 둘이 다르고, 하나로 합치면 평가 모드에서만 갈린다.
    noGrad(() => {
      const unbiased = biased.binary("mul", Tensor.full([], count / (count - 1)));
      this.runningMean.lerpFrom(mean.reshape([this.channels]), this.momentum);
      this.runningVar.lerpFrom(unbiased.reshape([this.channels]), this.momentum);
    });
    const norm = centered.div(biased.binary("add", Tensor.full([], this.eps)).sqrt());
    return norm.mul(w).add(b);
  }
}

/** 로짓과 정답 번호에서 바로. `log_softmax` 와 `nll_loss` 를 붙인 것이다. */
export class CrossEntropyLoss {
  forward(logits: Tensor, target: Tensor): Tensor {
    return logits.crossEntropy(target);
  }

  call(logits: Tensor, target: Tensor): Tensor {
    return this.forward(logits, target);
  }
}
