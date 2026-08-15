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

import { runningStats } from "./kernels.js";
import { device, keepAlive, noGrad, Tensor } from "./tensor.js";

/**
 * 가중치 초기화.
 *
 * **없으면 학습이 안 된다.** 0 으로 시작하면 같은 층의 뉴런들이 같은 기울기를 받아
 * 영원히 같은 값으로 움직인다 — ResNet 을 그렇게 돌리면 손실이 ln(10)=2.303 에
 * 붙은 채로 스텝만 지나간다. 실제로 그 상태로 벤치를 냈고, 골든 798건은 그것을
 * 못 잡았다: 케이스마다 가중치를 밖에서 넣어 주기 때문에 초기값을 아무도 안 본다.
 *
 * torch 의 기본과 같은 식이다 — `kaiming_uniform_(a=√5)` 는 경계가 `1/√fan_in` 으로
 * 접히고, 편향도 같은 경계의 균등분포다.
 *
 * **난수기는 torch 와 다르다.** 같을 수가 없고, 같은 척해서도 안 된다. 골든은 초기값을
 * 안 묻고 가중치를 늘 밖에서 넣으므로 여기서 갈릴 것이 없다.
 */
const initRng = { state: 0x9e3779b9 };

export function manualSeed(seed: number): void {
  initRng.state = (seed >>> 0) || 1;
}

function nextUniform(): number {
  let x = initRng.state;
  x ^= x << 13; x >>>= 0;
  x ^= x >> 17;
  x ^= x << 5; x >>>= 0;
  initRng.state = x;
  return x / 0x100000000;
}

/** `[-bound, bound]` 균등분포로 채운 텐서. */
function uniform(shape: readonly number[], bound: number): Tensor {
  const n = shape.reduce((a, b) => a * b, 1);
  const data = new Float32Array(n);
  for (let i = 0; i < n; i++) data[i] = (nextUniform() * 2 - 1) * bound;
  return Tensor.from(data, shape);
}

/** 들어오는 갈래 수. 가중치의 첫 축을 뺀 나머지의 곱이다. */
function fanIn(shape: readonly number[]): number {
  return shape.slice(1).reduce((a, b) => a * b, 1);
}

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

  /**
   * 자식을 **이름과 함께** 준다. 기본은 자리 번호다 — `Sequential` 이 그 모양이다.
   *
   * 이것이 없을 때는 `namedParameters` 가 자식을 번호로만 불렀고, 그러면
   * `fc1.weight` 같은 이름을 낼 수가 없었다. torch 는 속성 이름을 쓰는데
   * TypeScript 에는 속성을 훑어 층을 알아보는 자리가 없으므로, 이름을 붙이고
   * 싶은 층이 여기를 덮어쓴다.
   *
   * **`state_dict` 의 열쇠가 이 이름이다.** 갈리면 남의 체크포인트를 못 읽는다.
   */
  namedChildren(): Record<string, Module> {
    const out: Record<string, Module> = {};
    for (const [i, child] of this.children().entries()) out[String(i)] = child;
    return out;
  }

  /** `0.weight` 처럼 자리 이름을 앞에 붙인 이름표. */
  namedParameters(prefix = ""): Record<string, Tensor> {
    const out: Record<string, Tensor> = {};
    for (const [name, p] of Object.entries(this.ownParameters())) {
      out[`${prefix}${name}`] = p;
    }
    for (const [name, child] of Object.entries(this.namedChildren())) {
      Object.assign(out, child.namedParameters(`${prefix}${name}.`));
    }
    return out;
  }

  parameters(): Tensor[] {
    return Object.values(this.namedParameters());
  }

  /**
   * 파라미터를 구역 밖에서도 살려 둔다.
   *
   * 층을 학습 구역 **안에서** 세우면 그 구역이 닫힐 때 가중치가 놓인다. 밖에서
   * 세우는 것이 정석이지만, 정석을 안 지켰을 때 조용히 이상해지는 것보다 여기서
   * 못 박아 두는 편이 낫다.
   */
  protected claim(...params: Tensor[]): void {
    for (const p of params) {
      p.requiresGrad = true;
      keepAlive(p);
    }
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
    // **`namedChildren` 으로 돈다.** 이름을 붙인 층이 `children` 을 안 덮어써도
    // 모드가 내려가야 한다 — 안 그러면 학습은 멀쩡하고 추론만 틀린다.
    for (const c of Object.values(this.namedChildren())) c.train(mode);
    return this;
  }

  eval(): this {
    return this.train(false);
  }
}

/** 층을 줄줄이 세운 것. 자리 번호가 곧 이름이다. */
export class Sequential extends Module {
  private readonly layers: Module[];

  /**
   * **층을 그냥 나열한다** — `new Sequential(a, b, c)`. torch 가 그 모양이다.
   *
   * 배열 하나를 받게 두었더니 `index.ts` 에 적은 첫 예시부터 틀렸다. 쓰는 사람이
   * torch 코드를 옮겨 적을 때 대괄호를 빼먹는 것이 기본값이고, 그 기본값이 맞는
   * 쪽이어야 한다. 배열도 그대로 받는다 — 이미 그렇게 쓰던 자리가 있다.
   */
  constructor(...layers: readonly (Module | readonly Module[])[]) {
    super();
    this.layers = layers.flatMap((l) => (Array.isArray(l) ? [...l] : [l as Module]));
  }

  override children(): Module[] {
    return this.layers;
  }

  override forward(x: Tensor): Tensor {
    let cur = x;
    // 안에서도 `call` 로 지난다 — 권하는 길을 라이브러리 자신이 안 가면 그 권함은
    // 남이 볼 예시에서 지워진다.
    for (const layer of this.layers) cur = layer.call(cur);
    return cur;
  }
}

/**
 * 층 목록. 번호가 곧 이름이다 — `layers.0.weight`.
 *
 * `Sequential` 과 다른 점은 **부르지 않는다**는 것이다. 어떤 순서로 어떻게 쓸지는
 * 가진 쪽이 정하고, 이쪽은 파라미터가 보이게만 한다. 층 수가 정해지지 않은 모델이
 * 이것을 쓴다.
 */
export class ModuleList extends Module {
  private readonly items: Module[];

  constructor(mods: readonly Module[] = []) {
    super();
    this.items = [...mods];
  }

  override children(): Module[] {
    return this.items;
  }

  /** **부르는 층이 아니다.** 지나가려 하면 여기서 멈춘다 — torch 도 그렇다. */
  override forward(): Tensor {
    throw new Error("ModuleList 는 부르는 층이 아니다 — 안의 층을 골라 불러라");
  }

  append(module: Module): this {
    this.items.push(module);
    return this;
  }

  extend(mods: readonly Module[]): this {
    this.items.push(...mods);
    return this;
  }

  insert(index: number, module: Module): this {
    this.items.splice(index, 0, module);
    return this;
  }

  at(i: number): Module {
    const got = this.items.at(i);
    if (!got) throw new Error(`ModuleList 자리 ${i} 가 없다 (길이 ${this.items.length})`);
    return got;
  }

  get length(): number {
    return this.items.length;
  }

  [Symbol.iterator](): Iterator<Module> {
    return this.items[Symbol.iterator]();
  }
}

/**
 * torch 의 순서 규칙. **평범한 객체는 열쇠를 정렬해서 넣는다.**
 *
 * 안 맞추면 `namedParameters` 의 순서가 갈리고 그것이 곧 `stateDict` 의 순서다 —
 * 골든이 실제로 이 자리를 잡았다(`{w, b}` 에 torch 는 `ws.b ws.w` 를 냈다).
 */
function sortedEntries<T>(obj: Readonly<Record<string, T>>): [string, T][] {
  return Object.entries(obj).sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0));
}

/** 이름 붙은 층 묶음. 준 이름이 그대로 `stateDict` 열쇠가 된다. */
export class ModuleDict extends Module {
  private readonly items = new Map<string, Module>();

  constructor(mods: Readonly<Record<string, Module>> = {}) {
    super();
    for (const [name, m] of sortedEntries(mods)) this.items.set(name, m);
  }

  override children(): Module[] {
    return [...this.items.values()];
  }

  override namedChildren(): Record<string, Module> {
    return Object.fromEntries(this.items);
  }

  override forward(): Tensor {
    throw new Error("ModuleDict 는 부르는 층이 아니다 — 안의 층을 골라 불러라");
  }

  at(key: string): Module {
    const got = this.items.get(key);
    if (!got) throw new Error(`ModuleDict 에 '${key}' 가 없다`);
    return got;
  }

  set(key: string, module: Module): this {
    this.items.set(key, module);
    return this;
  }

  has(key: string): boolean {
    return this.items.has(key);
  }

  keys(): string[] {
    return [...this.items.keys()];
  }
}

/**
 * 학습되는 텐서 목록. **이것이 없으면 대신할 방법이 없다.**
 *
 * 층에 안 붙은 파라미터는 `ownParameters` 를 손으로 적지 않는 한 아무 데도 안
 * 잡힌다. 안 잡히면 옵티마이저가 못 보고, 못 보면 안 갱신하고, 그런데 **손실은
 * 내려간다** — 남은 파라미터가 대신 맞추기 때문이다. 예외도 경고도 없다.
 */
export class ParameterList extends Module {
  private readonly items: Tensor[];

  constructor(params: readonly Tensor[] = []) {
    super();
    this.items = [...params];
    this.claim(...this.items);
  }

  override ownParameters(): Record<string, Tensor> {
    return Object.fromEntries(this.items.map((p, i) => [String(i), p]));
  }

  override forward(): Tensor {
    throw new Error("ParameterList 는 부르는 층이 아니다");
  }

  append(param: Tensor): this {
    this.claim(param);
    this.items.push(param);
    return this;
  }

  at(i: number): Tensor {
    const got = this.items.at(i);
    if (!got) throw new Error(`ParameterList 자리 ${i} 가 없다 (길이 ${this.items.length})`);
    return got;
  }

  get length(): number {
    return this.items.length;
  }
}

/** 이름 붙은 파라미터 묶음. `ParameterList` 와 같은 이유로 있다. */
export class ParameterDict extends Module {
  private readonly items = new Map<string, Tensor>();

  constructor(params: Readonly<Record<string, Tensor>> = {}) {
    super();
    for (const [name, p] of sortedEntries(params)) this.items.set(name, p);
    this.claim(...this.items.values());
  }

  override ownParameters(): Record<string, Tensor> {
    return Object.fromEntries(this.items);
  }

  override forward(): Tensor {
    throw new Error("ParameterDict 는 부르는 층이 아니다");
  }

  at(key: string): Tensor {
    const got = this.items.get(key);
    if (!got) throw new Error(`ParameterDict 에 '${key}' 가 없다`);
    return got;
  }

  set(key: string, param: Tensor): this {
    this.claim(param);
    this.items.set(key, param);
    return this;
  }

  keys(): string[] {
    return [...this.items.keys()];
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
    const bound = 1 / Math.sqrt(Math.max(1, inFeatures));
    this.weight = uniform([outFeatures, inFeatures], bound);
    this.bias = uniform([outFeatures], bound);
    this.claim(this.weight, this.bias);
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
  /** 편향이 없을 수도 있다 — 뒤에 정규화가 오면 편향이 상수항으로 흡수된다. */
  readonly bias: Tensor | null;

  constructor(
    inChannels: number,
    outChannels: number,
    kernel: number,
    spatial: number,
    private readonly stride = 1,
    private readonly padding = 0,
    useBias = true,
  ) {
    super();
    const shape = [
      outChannels, inChannels, ...new Array<number>(spatial).fill(kernel),
    ];
    const bound = 1 / Math.sqrt(Math.max(1, fanIn(shape)));
    this.weight = uniform(shape, bound);
    this.bias = useBias ? uniform([outChannels], bound) : null;
    this.claim(this.weight);
    if (this.bias) this.claim(this.bias);
  }

  override ownParameters(): Record<string, Tensor> {
    return this.bias
      ? { weight: this.weight, bias: this.bias }
      : { weight: this.weight };
  }

  override forward(x: Tensor): Tensor {
    return x.convND(this.weight, this.bias, this.stride, this.padding);
  }
}

export class Conv1d extends ConvND {
  constructor(inC: number, outC: number, kernel: number, stride = 1, padding = 0,
              bias = true) {
    super(inC, outC, kernel, 1, stride, padding, bias);
  }
}

export class Conv2d extends ConvND {
  constructor(inC: number, outC: number, kernel: number, stride = 1, padding = 0,
              bias = true) {
    super(inC, outC, kernel, 2, stride, padding, bias);
  }
}

export class Conv3d extends ConvND {
  constructor(inC: number, outC: number, kernel: number, stride = 1, padding = 0,
              bias = true) {
    super(inC, outC, kernel, 3, stride, padding, bias);
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
    this.claim(this.weight, this.bias);
    this.runningMean = Tensor.zeros([channels]);
    this.runningVar = Tensor.ones([channels]);
    // 이동 통계는 기울기를 안 받지만 **구역이 닫혀도 살아야 한다.**
    keepAlive(this.runningMean);
    keepAlive(this.runningVar);
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
    void w;
    void b;
    // **융합 커널로 간다.** 조립판은 층 하나에 dispatch 스무 개가 넘었고, ResNet 한
    // 스텝의 1,636 개 중 태반이 거기서 나왔다(실측).
    const { out, mean, variance } = x.batchNormFused(this.weight, this.bias, this.eps);
    // 이동 통계에는 **불편추정**이 들어간다 — torch 가 그렇다. 정규화에 쓰는 것은
    // 편향추정이라 둘이 다르고, 하나로 합치면 평가 모드에서만 갈린다.
    //
    // 둘을 커널 하나로 갱신한다. 조립판은 층마다 여덟 dispatch 였고 층이 스무 개다.
    const unbias = count / (count - 1);
    const d = device();
    d.run1d(
      d.pipeline(`rs:${this.channels}:${this.momentum}:${unbias}`,
        () => runningStats(this.channels, this.momentum, unbias)),
      [this.runningMean.buffer, this.runningVar.buffer, mean.buffer, variance.buffer],
      this.channels,
    );
    return out;
  }
}

/**
 * 순환망 — `RNN`·`LSTM`·`GRU` 를 게이트 수만 바꿔 한 클래스로.
 *
 * 입력은 `(길이, 배치, 특징)` 이다 — torch 의 기본이고 `batch_first` 가 아니다.
 *
 * **시간 축은 순차적이라 펼 수가 없다.** 한 걸음의 출력이 다음 걸음의 입력이라
 * 병렬로 돌 자리가 없고, 그래서 걸음마다 커널을 부른다. 짧은 시퀀스에서 도는 값이며,
 * 길어지면 걸음당 호출 비용이 계산을 덮는다.
 */
export type RNNKind = "RNN" | "LSTM" | "GRU";

export class Recurrent extends Module {
  readonly weightIh: Tensor;
  readonly weightHh: Tensor;
  readonly biasIh: Tensor;
  readonly biasHh: Tensor;

  constructor(
    inputSize: number,
    readonly hidden: number,
    readonly kind: RNNKind,
  ) {
    super();
    const gates = kind === "LSTM" ? 4 : kind === "GRU" ? 3 : 1;
    const rows = hidden * gates;
    // torch 의 순환망은 은닉 크기로 경계를 잡는다 — 네 가중치가 같은 경계를 쓴다.
    const bound = 1 / Math.sqrt(Math.max(1, hidden));
    this.weightIh = uniform([rows, inputSize], bound);
    this.weightHh = uniform([rows, hidden], bound);
    this.biasIh = uniform([rows], bound);
    this.biasHh = uniform([rows], bound);
    this.claim(this.weightIh, this.weightHh, this.biasIh, this.biasHh);
  }

  override ownParameters(): Record<string, Tensor> {
    return {
      weight_ih_l0: this.weightIh, weight_hh_l0: this.weightHh,
      bias_ih_l0: this.biasIh, bias_hh_l0: this.biasHh,
    };
  }

  override forward(x: Tensor): Tensor {
    return this.run(x).output;
  }

  /** 출력 전체와 마지막 상태를 같이 낸다. */
  run(x: Tensor): { output: Tensor; hidden: Tensor; cell: Tensor } {
    const [steps = 0, batch = 0] = x.shape;
    const H = this.hidden;
    let h = Tensor.zeros([batch, H]);
    let c = Tensor.zeros([batch, H]);
    const outs: Tensor[] = [];
    for (let t = 0; t < steps; t++) {
      const xt = x.select(0, t);
      const gi = xt.linear(this.weightIh).add(this.biasIh);
      const gh = h.linear(this.weightHh).add(this.biasHh);
      if (this.kind === "RNN") {
        h = gi.add(gh).unary("tanh");
      } else if (this.kind === "LSTM") {
        // torch 의 게이트 순서는 i, f, g, o 다. 순서가 틀리면 값이 그럴듯하게 틀린다.
        const g = gi.add(gh);
        const i = slice(g, 0, H).unary("sigmoid");
        const f = slice(g, 1, H).unary("sigmoid");
        const gg = slice(g, 2, H).unary("tanh");
        const o = slice(g, 3, H).unary("sigmoid");
        c = f.mul(c).add(i.mul(gg));
        h = o.mul(c.unary("tanh"));
      } else {
        // GRU 는 r, z 까지만 더하고 **n 게이트에서 갈린다** — 은닉 쪽 몫에 r 을 곱한 뒤
        // 더한다. 다 더하고 나서 곱하면 값이 조용히 달라진다.
        const r = slice(gi, 0, H).add(slice(gh, 0, H)).unary("sigmoid");
        const z = slice(gi, 1, H).add(slice(gh, 1, H)).unary("sigmoid");
        const n = slice(gi, 2, H).add(r.mul(slice(gh, 2, H))).unary("tanh");
        const one = Tensor.full([], 1);
        h = one.sub(z).mul(n).add(z.mul(h));
      }
      outs.push(h);
    }
    return {
      output: Tensor.stack(outs, 0),
      hidden: h.reshape([1, batch, H]),
      cell: c.reshape([1, batch, H]),
    };
  }
}

/** 게이트가 세로로 이어 붙어 있다 — `k` 번째 `H` 줄. */
function slice(g: Tensor, k: number, H: number): Tensor {
  return g.narrow(1, k * H, H);
}

/**
 * 여러 머리로 나눠 보는 어텐션.
 *
 * 입력은 `(배치, 길이, 특징)` 이다(`batch_first=True`). 마스크는 **실수**다 —
 * 0 과 -inf 이고, "0 이 아니면 가림" 으로 뭉뚱그리면 여기서 갈린다.
 */
export class MultiheadAttention extends Module {
  readonly inWeight: Tensor;
  readonly inBias: Tensor;
  readonly outWeight: Tensor;
  readonly outBias: Tensor;

  constructor(private readonly embed: number, private readonly heads: number) {
    super();
    const bound = 1 / Math.sqrt(Math.max(1, embed));
    this.inWeight = uniform([3 * embed, embed], bound);
    // torch 의 어텐션은 편향을 0 에서 시작한다 — 여기는 대칭이 안 깨질 자리가 아니다.
    this.inBias = Tensor.zeros([3 * embed]);
    this.outWeight = uniform([embed, embed], bound);
    this.outBias = Tensor.zeros([embed]);
    this.claim(this.inWeight, this.inBias, this.outWeight, this.outBias);
  }

  override ownParameters(): Record<string, Tensor> {
    return {
      in_proj_weight: this.inWeight, in_proj_bias: this.inBias,
      "out_proj.weight": this.outWeight, "out_proj.bias": this.outBias,
    };
  }

  override forward(x: Tensor): Tensor {
    return this.attend(x, null);
  }

  attend(x: Tensor, mask: Tensor | null): Tensor {
    const [batch = 1, len = 1] = x.shape;
    const E = this.embed;
    const head = E / this.heads;
    const flat = x.reshape([batch * len, E]);
    const projected = flat.linear(this.inWeight).add(this.inBias);
    const parts = [0, 1, 2].map((k) => projected.narrow(1, k * E, E));
    const scale = Tensor.full([], 1 / Math.sqrt(head));
    const outs: Tensor[] = [];
    for (let b = 0; b < batch; b++) {
      const perHead: Tensor[] = [];
      for (let h = 0; h < this.heads; h++) {
        const take = (t: Tensor | undefined): Tensor => {
          if (!t) throw new Error("attention: 투영이 없다");
          return t.reshape([batch, len, E]).select(0, b).narrow(1, h * head, head);
        };
        const q = take(parts[0]);
        const k = take(parts[1]);
        const v = take(parts[2]);
        let scores = q.mm(k.transpose()).binary("mul", scale);
        if (mask) scores = scores.add(mask);
        perHead.push(scores.softmax(1).mm(v));
      }
      outs.push(Tensor.cat(perHead, 1));
    }
    const merged = Tensor.stack(outs, 0).reshape([batch * len, E]);
    return merged.linear(this.outWeight).add(this.outBias)
      .reshape([batch, len, E]);
  }

  /** 앞만 보게 하는 마스크. 위 삼각을 -inf 로 채운다. */
  static causalMask(len: number): Tensor {
    const data = new Float32Array(len * len);
    for (let i = 0; i < len; i++) {
      for (let j = i + 1; j < len; j++) data[i * len + j] = -Infinity;
    }
    return Tensor.from(data, [len, len]);
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
