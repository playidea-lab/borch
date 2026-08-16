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
import { onSeed, uniformArray } from "./random.js";
import {
  device, keepAlive, noGrad, type PadMode, type Reduction, Tensor,
} from "./tensor.js";

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
// **난수기는 `random.ts` 로 옮겼다.** 층 초기화와 `Tensor.randn` 이 같은 줄기를
// 써야 씨앗 하나가 전부를 되돌린다. 여기 갇혀 있으면 `tensor.ts` 가 못 부른다 —
// 부르면 순환이다.
//
// **dropout 도 같이 되돌린다.** 그쪽은 부를 때마다 오르는 계수기를 따로 들고 있어서,
// 안 건드리면 씨앗을 심어도 마스크가 매번 달라진다 — "같은 씨앗에 같은 결과" 를
// 기대하는 사람이 가장 먼저 확인하는 자리가 층 초기화와 dropout 이다. 코어도 같은
// 갈래의 결함을 갖고 있었고 같은 케이스가 둘 다 잡았다.
//
// **씨앗 값을 실어야 한다.** 여기서 늘 1 로 되돌렸었는데, 그러면 씨앗을 바꿔도
// 마스크가 안 바뀐다 — 씨앗 다섯 개로 분산을 재는 사람이 가중치 초기화만 흔들린 수를
// 실험 분산으로 읽는다. 황금비 상수로 한 번 섞는 것은 두 줄기(호스트 xorshift 와 GPU
// 해시)가 같은 수에서 출발해 우연히 붙어 움직이지 않게 하려는 것이다.
onSeed((seed) => { Tensor.dropoutSeed = ((seed ^ 0x9e3779b9) >>> 0) || 1; });

export { manualSeed } from "./random.js";

/** `[-bound, bound]` 균등분포로 채운 텐서. */
function uniform(shape: readonly number[], bound: number): Tensor {
  const n = shape.reduce((a, b) => a * b, 1);
  return Tensor.from(uniformArray(n, bound), shape);
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

  /**
   * 이 층이 직접 가진 파라미터. 자식은 `children` 이 준다.
   *
   * **기본이 자기 필드를 훑는 것이다.** 전에는 `{}` 를 돌려주고 층마다 덮어쓰게
   * 했는데, 그러면 밖에서 층을 만드는 사람이 덮어쓰기를 잊는 순간 그 파라미터가
   * `parameters()` 에 안 나오고 — **예외 없이** 학습만 안 된다. 손실이 안 내려가는데
   * 어디가 원인인지 가리키는 것이 아무것도 없는 종류다.
   *
   * 주석에 "TypeScript 에는 속성을 훑어 층을 알아보는 자리가 없다" 고 적혀 있었는데
   * 그것은 사실이 아니었다. `Object.entries(this)` 가 인스턴스 필드를 준다 —
   * torch 의 `__setattr__` 이 하는 일과 같은 자리다.
   *
   * **`requiresGrad` 가 표식이다.** 필드에 있는 텐서를 전부 파라미터로 세면 상수나
   * 마스크까지 옵티마이저가 밟는다. torch 에서 `nn.Parameter` 가 하는 구분을 여기서는
   * 이 깃발이 한다 — 층은 `claim()` 으로 세우고, 그것이 곧 "이건 파라미터다" 다.
   */
  ownParameters(): Record<string, Tensor> {
    const out: Record<string, Tensor> = {};
    for (const [name, value] of Object.entries(this)) {
      if (value instanceof Tensor && value.requiresGrad) out[name] = value;
    }
    return out;
  }

  children(): Module[] {
    return Object.values(this.namedChildren());
  }

  /**
   * 자식을 **이름과 함께** 준다. 기본은 **필드 이름**이다 — torch 와 같다.
   *
   * ```ts
   * class Net extends nn.Module {
   *   fc1 = new nn.Linear(4, 8);      // → "fc1.weight", "fc1.bias"
   *   fc2 = new nn.Linear(8, 2);
   *   forward(x) { return this.fc2.call(this.fc1.call(x)); }
   * }
   * ```
   *
   * **배열은 안 훑는다.** `layers = [a, b]` 는 자식이 아니다 — torch 도 파이썬 list 를
   * 등록하지 않고 `nn.ModuleList` 를 요구한다. 배열을 훑으면 "층이 아닌 배열" 과
   * 구분할 방법이 없고, 그 구분이 없으면 `state_dict` 열쇠가 조용히 바뀐다.
   *
   * 자리 번호로 부르고 싶은 컨테이너(`Sequential`·`ModuleList`)는 여기를 덮어쓴다.
   *
   * **`state_dict` 의 열쇠가 이 이름이다.** 갈리면 남의 체크포인트를 못 읽는다.
   */
  namedChildren(): Record<string, Module> {
    const out: Record<string, Module> = {};
    for (const [name, value] of Object.entries(this)) {
      if (value instanceof Module) out[name] = value;
    }
    return out;
  }

  /**
   * `print(model)` 이 찍는 글자. torch 의 `__repr__` 과 같은 모양이다.
   *
   * **자식의 여러 줄도 다시 들여쓴다.** 컨테이너가 컨테이너를 담을 때
   * (`ModuleList` 안의 `Sequential`) 안쪽이 왼쪽 끝에 붙으면 torch 와 그림이 갈린다 —
   * 값은 멀쩡하고 글자만 틀리는 종류라 눈으로만 잡힌다.
   *
   * 이름이나 인자를 찍고 싶은 층은 이것을 덮어쓴다.
   */
  describe(): string {
    const kids = Object.entries(this.namedChildren());
    const name = this.constructor.name;
    if (!kids.length) return `${name}()`;
    const lines = [`${name}(`];
    for (const [key, child] of kids) {
      const [first, ...rest] = child.describe().split("\n");
      lines.push(`  (${key}): ${first}`);
      for (const line of rest) lines.push(`  ${line}`);
    }
    lines.push(")");
    return lines.join("\n");
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

  /**
   * **자리 번호로 부른다** — `0.weight`. torch 의 `Sequential` 이 그 모양이고
   * 골든이 그 이름으로 가중치를 넣는다.
   *
   * 기본 구현은 필드 이름을 쓰는데 우리 층은 배열(`layers`) 안에 있어 거기 안 걸린다.
   * 그러면 `state_dict` 가 통째로 비므로 여기를 덮어써야 한다.
   */
  override namedChildren(): Record<string, Module> {
    const out: Record<string, Module> = {};
    for (const [i, child] of this.layers.entries()) out[String(i)] = child;
    return out;
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

  /** 자리 번호로 부른다. `Sequential` 과 같은 이유다 — 층이 배열 안에 있다. */
  override namedChildren(): Record<string, Module> {
    const out: Record<string, Module> = {};
    for (const [i, child] of this.items.entries()) out[String(i)] = child;
    return out;
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
  /** **없을 수 있다.** `AdaptiveLogSoftmaxWithLoss` 의 꼬리 층이 치우침을 안 쓴다. */
  readonly bias: Tensor | null;

  constructor(inFeatures: number, outFeatures: number, bias = true) {
    super();
    // 골든은 가중치를 밖에서 넣는다. 여기 초기값이 무엇이든 덮어쓰이지만,
    // 안 넣고 쓰는 경우를 위해 0 이 아닌 값을 둔다.
    const bound = 1 / Math.sqrt(Math.max(1, inFeatures));
    this.weight = uniform([outFeatures, inFeatures], bound);
    this.bias = bias ? uniform([outFeatures], bound) : null;
    this.claim(this.weight);
    if (this.bias) this.claim(this.bias);
  }

  override ownParameters(): Record<string, Tensor> {
    // **치우침이 없으면 열쇠도 없다.** 있는 척하면 `state_dict` 가 남의 것과 안 맞는다.
    return this.bias
      ? { weight: this.weight, bias: this.bias }
      : { weight: this.weight };
  }

  override forward(x: Tensor): Tensor {
    const out = x.linear(this.weight);
    return this.bias ? out.add(this.bias) : out;
  }

  /**
   * 파이썬이 찍는 그대로.
   *
   * **게으른 층이 굳으면 이 글자가 답이 된다** — 그 물건은 그때부터 `Linear` 이고,
   * 사용자가 `print(model)` 로 보는 것이 이것이다.
   */
  override describe(): string {
    const [out, inF] = [this.weight.shape[0] ?? 0, this.weight.shape[1] ?? 0];
    return `Linear(in_features=${inF}, out_features=${out}, `
      + `bias=${this.bias ? "True" : "False"})`;
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

// ── 활성함수 층. 전부 텐서 메서드 하나를 감싼다. ────────────────────────────
//
// 감싸개가 **다른 함수를 부르는** 것이 이 부류의 유일한 실패 방식이고, 그것은 눈으로
// 안 보이고 값으로만 갈린다 — 그래서 골든이 함수 꼴과 층 꼴을 따로 묻는다.

export class Hardsigmoid extends Module {
  override forward(x: Tensor): Tensor {
    return x.unary("hardsigmoid");
  }
}

export class Hardswish extends Module {
  override forward(x: Tensor): Tensor {
    return x.unary("hardswish");
  }
}

export class LogSigmoid extends Module {
  override forward(x: Tensor): Tensor {
    return x.unary("logsigmoid");
  }
}

export class Mish extends Module {
  override forward(x: Tensor): Tensor {
    return x.unary("mish");
  }
}

export class ReLU6 extends Module {
  override forward(x: Tensor): Tensor {
    return x.unary("relu6");
  }
}

export class SELU extends Module {
  override forward(x: Tensor): Tensor {
    return x.unary("selu");
  }
}

export class Softsign extends Module {
  override forward(x: Tensor): Tensor {
    return x.unary("softsign");
  }
}

export class Tanhshrink extends Module {
  override forward(x: Tensor): Tensor {
    return x.unary("tanhshrink");
  }
}

export class CELU extends Module {
  constructor(private readonly alpha = 1.0) {
    super();
  }

  override forward(x: Tensor): Tensor {
    return x.celu(this.alpha);
  }
}

export class Hardshrink extends Module {
  constructor(private readonly lambd = 0.5) {
    super();
  }

  override forward(x: Tensor): Tensor {
    return x.hardshrink(this.lambd);
  }
}

export class Softshrink extends Module {
  constructor(private readonly lambd = 0.5) {
    super();
  }

  override forward(x: Tensor): Tensor {
    return x.softshrink(this.lambd);
  }
}

export class Hardtanh extends Module {
  constructor(private readonly minVal = -1.0, private readonly maxVal = 1.0) {
    super();
  }

  override forward(x: Tensor): Tensor {
    return x.hardtanh(this.minVal, this.maxVal);
  }
}

export class Softplus extends Module {
  constructor(private readonly beta = 1.0, private readonly threshold = 20.0) {
    super();
  }

  override forward(x: Tensor): Tensor {
    return x.softplus(this.beta, this.threshold);
  }
}

export class Threshold extends Module {
  constructor(private readonly t: number, private readonly value: number) {
    super();
  }

  override forward(x: Tensor): Tensor {
    return x.threshold(this.t, this.value);
  }
}

export class Softmin extends Module {
  constructor(private readonly dim = -1) {
    super();
  }

  override forward(x: Tensor): Tensor {
    return x.softmin(this.dim);
  }
}

export class GLU extends Module {
  constructor(private readonly dim = -1) {
    super();
  }

  override forward(x: Tensor): Tensor {
    return x.glu(this.dim);
  }
}

/**
 * 음수 쪽 기울기를 **학습한다.** 이 부류에서 유일하게 파라미터가 있다.
 *
 * `weight` 라는 이름이 `stateDict` 열쇠가 되므로 torch 와 같아야 한다.
 */
export class PReLU extends Module {
  readonly weight: Tensor;

  constructor(numParameters = 1, init = 0.25) {
    super();
    this.weight = Tensor.owned([numParameters], init);
    this.claim(this.weight);
  }

  override ownParameters(): Record<string, Tensor> {
    return { weight: this.weight };
  }

  override forward(x: Tensor): Tensor {
    return x.prelu(this.weight);
  }
}

/**
 * 채널을 그룹으로 묶어 정규화. **배치가 작을 때 BatchNorm 대신 쓴다.**
 *
 * BatchNorm 은 배치 통계를 쓰므로 배치가 1~2 면 통계가 못 미덥다. 이쪽은 표본 하나
 * 안에서 묶으므로 배치 크기와 무관하다.
 */
export class GroupNorm extends Module {
  readonly weight: Tensor;
  readonly bias: Tensor;

  constructor(
    private readonly numGroups: number,
    numChannels: number,
    private readonly eps = 1e-5,
  ) {
    super();
    this.weight = Tensor.owned([numChannels], 1);
    this.bias = Tensor.owned([numChannels], 0);
    this.claim(this.weight, this.bias);
  }

  override ownParameters(): Record<string, Tensor> {
    return { weight: this.weight, bias: this.bias };
  }

  override forward(x: Tensor): Tensor {
    const shape = [1, this.weight.size, ...new Array<number>(x.shape.length - 2).fill(1)];
    return x.groupNorm(this.numGroups, this.eps)
      .mul(this.weight.reshape(shape)).add(this.bias.reshape(shape));
  }
}

/**
 * 표본마다·채널마다 따로. **기본이 파라미터 없음이다** — torch 가 그렇다.
 *
 * `BatchNorm` 과 반대라 헷갈리는 자리이고, 기본을 뒤집으면 `stateDict` 열쇠가
 * 통째로 갈린다.
 */
export class InstanceNormND extends Module {
  constructor(private readonly eps = 1e-5) {
    super();
  }

  override forward(x: Tensor): Tensor {
    return x.instanceNorm(this.eps);
  }
}

/** **평균을 안 뺀다.** 그것이 `LayerNorm` 과의 유일한 차이다. */
export class RMSNorm extends Module {
  readonly weight: Tensor;

  constructor(private readonly normalizedShape: number | readonly number[]) {
    super();
    const dims = typeof normalizedShape === "number"
      ? [normalizedShape] : [...normalizedShape];
    this.weight = Tensor.owned(dims, 1);
    this.claim(this.weight);
  }

  override ownParameters(): Record<string, Tensor> {
    return { weight: this.weight };
  }

  override forward(x: Tensor): Tensor {
    const dims = typeof this.normalizedShape === "number"
      ? 1 : this.normalizedShape.length;
    return x.rmsNorm(dims).mul(this.weight);
  }
}

/**
 * 전치 합성곱. **가중치가 `(입력, 출력, …)` 이다** — `ConvND` 와 뒤집혀 있다.
 *
 * 정사각 커널이면 뒤집어 놓아도 모양이 맞아서 값으로만 갈린다. `stateDict` 열쇠는
 * `weight`·`bias` 로 같으므로, 모양만 보고 넣으면 조용히 틀린다.
 */
export class ConvTransposeND extends Module {
  readonly weight: Tensor;
  readonly bias: Tensor | null;

  constructor(
    inC: number,
    outC: number,
    kernel: number,
    spatial: number,
    private readonly stride = 1,
    private readonly padding = 0,
    bias = true,
  ) {
    super();
    const bound = 1 / Math.sqrt(Math.max(1, outC * kernel ** spatial));
    this.weight = uniform([inC, outC, ...new Array<number>(spatial).fill(kernel)], bound);
    this.bias = bias ? uniform([outC], bound) : null;
    this.claim(...(this.bias ? [this.weight, this.bias] : [this.weight]));
  }

  override ownParameters(): Record<string, Tensor> {
    return this.bias
      ? { weight: this.weight, bias: this.bias }
      : { weight: this.weight };
  }

  override forward(x: Tensor): Tensor {
    return x.convTransposeND(this.weight, this.bias, this.stride, this.padding);
  }
}

/**
 * 학습 때만 자리를 떨군다. **평가 모드에서는 항등이다.**
 *
 * `training` 은 `Module` 이 들고 있고 `eval()` 이 컨테이너를 뚫고 내려가 끈다 —
 * 그 전파가 끊기면 학습은 멀쩡하고 추론만 틀린다.
 */
export class Dropout extends Module {
  constructor(private readonly p = 0.5) {
    super();
  }

  override forward(x: Tensor): Tensor {
    return x.dropout(this.p, this.training);
  }
}

/**
 * 어텐션의 알맹이. **`MultiheadAttention` 이 안에서 하던 계산을 이름으로 낸다.**
 *
 * 층은 있는데 이 함수가 없었다. 층을 안 쓰고 어텐션을 손으로 짜는 코드가 이 이름을
 * 부르고, 요즘 트랜스포머 코드의 기본형이 그것이다.
 *
 * **가림막은 곱하는 것이 아니라 더하는 것이다.** 큰 음수를 더해 softmax 가 0 을
 * 내게 하는 것이지 0 을 곱하는 것이 아니다 — 곱하면 softmax 가 이미 정규화한 뒤라
 * 남은 자리가 1 로 안 돌아간다.
 *
 * 배치 축이 있으면 표본마다 따로 돈다. `mm` 이 2 차원끼리라 그렇고, 묶음 행렬곱이
 * 생기면 그때 여기를 고치면 된다.
 */
export function scaledDotProductAttention(
  query: Tensor,
  key: Tensor,
  value: Tensor,
  attnMask: Tensor | null = null,
  isCausal = false,
): Tensor {
  const rank = query.shape.length;
  const dim = query.shape[rank - 1] ?? 1;
  const len = query.shape[rank - 2] ?? 1;
  const keyLen = key.shape[key.shape.length - 2] ?? 1;
  const scale = Tensor.full([], 1 / Math.sqrt(dim));

  // 위 삼각을 막는 가림막. 0 과 큰 음수로 만들어 **더한다.**
  let causal: Tensor | null = null;
  if (isCausal) {
    const rows: number[] = [];
    for (let i = 0; i < len; i++) {
      for (let j = 0; j < keyLen; j++) rows.push(j > i ? -1e30 : 0);
    }
    causal = Tensor.from(rows, [len, keyLen]);
  }

  const one = (q: Tensor, k: Tensor, v: Tensor): Tensor => {
    let scores = q.mm(k.transpose()).binary("mul", scale);
    if (causal) scores = scores.add(causal);
    if (attnMask) scores = scores.add(attnMask);
    return scores.softmax(-1).mm(v);
  };

  if (rank === 2) return one(query, key, value);
  const batch = query.shape[0] ?? 1;
  const outs: Tensor[] = [];
  for (let b = 0; b < batch; b++) {
    outs.push(one(query.select(0, b), key.select(0, b), value.select(0, b)));
  }
  return Tensor.stack(outs, 0);
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

// ── 되풀이의 한 걸음 ────────────────────────────────────────────────────
//
// **이름이 층 쪽과 다르다.** 층은 `weight_ih_l0` 처럼 층 번호를 붙이고 셀은
// `weight_ih` 다 — 셀에는 층이 없다. `stateDict` 열쇠가 그 이름이므로 틀리면
// 체크포인트가 안 맞는다.
//
// 게이트 순서는 `Recurrent` 와 같은 것을 쓴다 — GRU 는 `r, z, n`, LSTM 은
// `i, f, g, o` 다. 두 벌로 적으면 갈리는 날이 오고 그때 값만 조용히 틀린다.

export class RNNCellBase extends Module {
  readonly weightIh: Tensor;
  readonly weightHh: Tensor;
  readonly biasIh: Tensor | null;
  readonly biasHh: Tensor | null;

  constructor(readonly inputSize: number, readonly hidden: number,
              gates: number, readonly hasBias = true) {
    super();
    const rows = hidden * gates;
    const bound = 1 / Math.sqrt(Math.max(1, hidden));
    this.weightIh = uniform([rows, inputSize], bound);
    this.weightHh = uniform([rows, hidden], bound);
    this.biasIh = hasBias ? uniform([rows], bound) : null;
    this.biasHh = hasBias ? uniform([rows], bound) : null;
    this.claim(this.weightIh, this.weightHh);
    if (this.biasIh && this.biasHh) this.claim(this.biasIh, this.biasHh);
  }

  override ownParameters(): Record<string, Tensor> {
    const out: Record<string, Tensor> = {
      weight_ih: this.weightIh, weight_hh: this.weightHh,
    };
    if (this.biasIh && this.biasHh) {
      out["bias_ih"] = this.biasIh;
      out["bias_hh"] = this.biasHh;
    }
    return out;
  }

  protected gi(x: Tensor): Tensor {
    const out = x.linear(this.weightIh);
    return this.biasIh ? out.add(this.biasIh) : out;
  }

  protected gh(h: Tensor): Tensor {
    const out = h.linear(this.weightHh);
    return this.biasHh ? out.add(this.biasHh) : out;
  }

  protected zeros(x: Tensor): Tensor {
    return Tensor.zeros([x.shape[0] ?? 1, this.hidden]);
  }

  override forward(x: Tensor): Tensor { return x; }
}

export class RNNCell extends RNNCellBase {
  constructor(inputSize: number, hidden: number, hasBias = true,
              readonly nonlinearity: "tanh" | "relu" = "tanh") {
    super(inputSize, hidden, 1, hasBias);
  }

  step(x: Tensor, hx: Tensor | null = null): Tensor {
    const h = hx ?? this.zeros(x);
    return this.gi(x).add(this.gh(h)).unary(this.nonlinearity);
  }

  override forward(x: Tensor): Tensor { return this.step(x); }

  override describe(): string {
    let parts = `${this.inputSize}, ${this.hidden}`;
    if (!this.hasBias) parts += ", bias=False";
    if (this.nonlinearity !== "tanh") parts += `, nonlinearity=${this.nonlinearity}`;
    return `RNNCell(${parts})`;
  }
}

export class GRUCell extends RNNCellBase {
  constructor(inputSize: number, hidden: number, hasBias = true) {
    super(inputSize, hidden, 3, hasBias);
  }

  step(x: Tensor, hx: Tensor | null = null): Tensor {
    const h = hx ?? this.zeros(x);
    const H = this.hidden;
    const gi = this.gi(x);
    const gh = this.gh(h);
    const r = slice(gi, 0, H).add(slice(gh, 0, H)).unary("sigmoid");
    const z = slice(gi, 1, H).add(slice(gh, 1, H)).unary("sigmoid");
    const n = slice(gi, 2, H).add(r.mul(slice(gh, 2, H))).unary("tanh");
    return Tensor.full([], 1).sub(z).mul(n).add(z.mul(h));
  }

  override forward(x: Tensor): Tensor { return this.step(x); }

  override describe(): string {
    return `GRUCell(${this.inputSize}, ${this.hidden}` +
      `${this.hasBias ? "" : ", bias=False"})`;
  }
}

/** **혼자 둘을 돌려준다** — `(h, c)` 다. 셋을 한 모양으로 두면 기억 칸이 사라진다. */
export class LSTMCell extends RNNCellBase {
  constructor(inputSize: number, hidden: number, hasBias = true) {
    super(inputSize, hidden, 4, hasBias);
  }

  step(x: Tensor, hx: readonly [Tensor, Tensor] | null = null):
      [Tensor, Tensor] {
    const [h, c] = hx ?? [this.zeros(x), this.zeros(x)];
    const H = this.hidden;
    const g = this.gi(x).add(this.gh(h));
    const i = slice(g, 0, H).unary("sigmoid");
    const f = slice(g, 1, H).unary("sigmoid");
    const gg = slice(g, 2, H).unary("tanh");
    const o = slice(g, 3, H).unary("sigmoid");
    const cell = f.mul(c).add(i.mul(gg));
    return [o.mul(cell.unary("tanh")), cell];
  }

  override forward(x: Tensor): Tensor { return this.step(x)[0]; }

  override describe(): string {
    return `LSTMCell(${this.inputSize}, ${this.hidden}` +
      `${this.hasBias ? "" : ", bias=False"})`;
  }
}

// ── 나머지 층 ───────────────────────────────────────────────────────────

export class Unfold extends Module {
  constructor(readonly kernel: number, readonly dilation = 1,
              readonly padding = 0, readonly stride = 1) { super(); }

  override forward(x: Tensor): Tensor {
    return x.unfoldIm2col(this.kernel, this.dilation, this.padding, this.stride);
  }

  override describe(): string {
    return `Unfold(kernel_size=${this.kernel}, dilation=${this.dilation}, ` +
      `padding=${this.padding}, stride=${this.stride})`;
  }
}

export class Fold extends Module {
  constructor(readonly outputSize: [number, number], readonly kernel: number,
              readonly dilation = 1, readonly padding = 0,
              readonly stride = 1) { super(); }

  override forward(x: Tensor): Tensor {
    return x.fold(this.outputSize, this.kernel, this.dilation, this.padding,
      this.stride);
  }

  override describe(): string {
    return `Fold(output_size=(${this.outputSize.join(", ")}), ` +
      `kernel_size=${this.kernel}, dilation=${this.dilation}, ` +
      `padding=${this.padding}, stride=${this.stride})`;
  }
}

/** 두 입력을 **한꺼번에** 섞는다. 가중치가 `(out, in1, in2)` 세 축이다. */
export class Bilinear extends Module {
  readonly weight: Tensor;
  readonly bias: Tensor;

  constructor(readonly in1: number, readonly in2: number, readonly out: number) {
    super();
    const bound = 1 / Math.sqrt(Math.max(1, in1));
    this.weight = uniform([out, in1, in2], bound);
    this.bias = uniform([out], bound);
    this.claim(this.weight, this.bias);
  }

  override ownParameters(): Record<string, Tensor> {
    return { weight: this.weight, bias: this.bias };
  }

  override forward(x: Tensor): Tensor { return x; }

  call2(x1: Tensor, x2: Tensor): Tensor {
    return x1.bilinear(x2, this.weight, this.bias);
  }

  override describe(): string {
    return `Bilinear(in1_features=${this.in1}, in2_features=${this.in2}, ` +
      `out_features=${this.out}, bias=True)`;
  }
}

export class LocalResponseNorm extends Module {
  constructor(readonly size: number, readonly alpha = 1e-4,
              readonly beta = 0.75, readonly k = 1.0) { super(); }

  override forward(x: Tensor): Tensor {
    return x.localResponseNorm(this.size, this.alpha, this.beta, this.k);
  }

  override describe(): string {
    // **파이썬 꼴로 찍는다.** `k=1` 이 아니라 `k=1.0` 이다 — JS 는 정수처럼 보이는
    // 실수에서 소수점을 지우고, 골든은 글자를 굳혔다.
    const py = (v: number) => (Number.isInteger(v) ? `${v}.0` : String(v));
    return `LocalResponseNorm(${this.size}, alpha=${this.alpha}, ` +
      `beta=${py(this.beta)}, k=${py(this.k)})`;
  }
}

/** `(N, C, H, W)` 의 **채널 방향** softmax. `softmax(dim=1)` 과 같다. */
export class Softmax2d extends Module {
  override forward(x: Tensor): Tensor { return x.softmax(1); }
  override describe(): string { return "Softmax2d()"; }
}

export class RReLU extends Module {
  constructor(readonly lower = 1 / 8, readonly upper = 1 / 3) { super(); }

  override forward(x: Tensor): Tensor {
    return x.rrelu(this.lower, this.upper, this.training);
  }

  override describe(): string {
    return `RReLU(lower=${this.lower}, upper=${this.upper})`;
  }
}

/**
 * 옛 이름 둘.
 *
 * **`UpsamplingBilinear2d` 는 `alignCorners=true` 다** — `Upsample(bilinear)` 의
 * 기본값과 다르다. 이름만 보고 별명으로 두면 가장자리가 어긋난다.
 */
class UpsamplingBase extends Module {
  constructor(readonly label: string, readonly scale: number,
              readonly mode: "nearest" | "bilinear") { super(); }

  override forward(x: Tensor): Tensor {
    if (this.mode === "nearest") return x.upsample(this.scale);
    const h = (x.shape[2] ?? 1) * this.scale;
    const w = (x.shape[3] ?? 1) * this.scale;
    return x.interpolateBilinear(h, w, true);
  }

  override describe(): string {
    return `${this.label}(scale_factor=${this.scale.toFixed(1)}, ` +
      `mode='${this.mode}')`;
  }
}

export class UpsamplingNearest2d extends UpsamplingBase {
  constructor(scale = 2) { super("UpsamplingNearest2d", scale, "nearest"); }
}
export class UpsamplingBilinear2d extends UpsamplingBase {
  constructor(scale = 2) { super("UpsamplingBilinear2d", scale, "bilinear"); }
}

/** 가방마다 한 줄. 표에서 골라 **합치는 것**까지가 한 층이다. */
export class EmbeddingBag extends Module {
  readonly weight: Tensor;

  constructor(readonly num: number, readonly dim: number,
              readonly mode: "sum" | "mean" | "max" = "mean") {
    super();
    this.weight = uniform([num, dim], 1);
    this.claim(this.weight);
  }

  override ownParameters(): Record<string, Tensor> {
    return { weight: this.weight };
  }

  override forward(idx: Tensor): Tensor {
    // 계산은 `embeddingBag` 이 한다 — 층과 함수를 두 벌로 두지 않는다.
    return embeddingBag(idx, this.weight, null, this.mode);
  }

  /**
   * 1 차원 번호 줄을 `offsets` 로 잘라 가방을 만든다.
   *
   * **가방 길이가 제각각인 자리가 이 이름의 이유다.** 2 차원 입력은 길이가 같은
   * 가방만 되고, 실제로 쓰는 쪽(장바구니·문장)은 길이가 다르다.
   */
  callOffsets(idx: Tensor, offsets: readonly number[]): Tensor {
    return embeddingBag(idx, this.weight, offsets, this.mode);
  }

  override describe(): string {
    return `EmbeddingBag(${this.num}, ${this.dim}, mode='${this.mode}')`;
  }
}

// ── 자리 옮기기·채널째 dropout ──────────────────────────────────────────
//
// 여덟 층이 전부 텐서 메서드 하나를 부른다. 갈리는 것은 넘길 인자와 찍는 글자뿐이다.

export class PixelShuffle extends Module {
  constructor(readonly factor: number) { super(); }
  override forward(x: Tensor): Tensor { return x.pixelShuffle(this.factor); }
  override describe(): string { return `PixelShuffle(upscale_factor=${this.factor})`; }
}

export class PixelUnshuffle extends Module {
  constructor(readonly factor: number) { super(); }
  override forward(x: Tensor): Tensor { return x.pixelUnshuffle(this.factor); }
  override describe(): string { return `PixelUnshuffle(downscale_factor=${this.factor})`; }
}

export class ChannelShuffle extends Module {
  constructor(readonly groups: number) { super(); }
  override forward(x: Tensor): Tensor { return x.channelShuffle(this.groups); }
  override describe(): string { return `ChannelShuffle(groups=${this.groups})`; }
}

/** 채널째 떨구는 것들의 뿌리. **`inplace` 까지 찍는다** — torch 가 그렇다. */
class FeatureDropoutBase extends Module {
  constructor(
    readonly label: string,
    readonly p = 0.5,
    private readonly alpha = false,
    private readonly perChannel = true,
  ) { super(); }

  override forward(x: Tensor): Tensor {
    return this.alpha
      ? x.alphaDropout(this.p, this.training, this.perChannel)
      : x.featureDropout(this.p, this.training);
  }

  override describe(): string {
    return `${this.label}(p=${this.p}, inplace=False)`;
  }
}

export class Dropout1d extends FeatureDropoutBase {
  constructor(p?: number) { super("Dropout1d", p); }
}
export class Dropout2d extends FeatureDropoutBase {
  constructor(p?: number) { super("Dropout2d", p); }
}
export class Dropout3d extends FeatureDropoutBase {
  constructor(p?: number) { super("Dropout3d", p); }
}
export class AlphaDropout extends FeatureDropoutBase {
  constructor(p?: number) { super("AlphaDropout", p, true, false); }
}
export class FeatureAlphaDropout extends FeatureDropoutBase {
  constructor(p?: number) { super("FeatureAlphaDropout", p, true, true); }
}

// ── 게으른 층 ───────────────────────────────────────────────────────────
//
// **모양을 첫 forward 에서 알아낸다.** `new nn.LazyLinear(3)` 은 들어오는 크기를 안
// 받고 처음 지나가는 값을 보고 정한다 — 합성곱 뒤에 몇 채널이 나오는지를 손으로 세는
// 일이 사라진다.
//
// **굳으면 딴 것이 된다.** torch 는 첫 forward 뒤에 그 물건의 클래스를 바꿔 버린다.
// 여기서는 프로토타입을 갈아 끼우고 속을 옮겨 같은 자리를 짚는다 — 새 물건을
// 돌려주면 이미 붙잡아 둔 쪽이 옛것을 계속 들고 있게 된다.

export class LazyModule extends Module {
  private built: Module | null = null;

  constructor(
    /** 굳기 전에 찍을 글자. `describe()` 가 쓴다. */
    readonly label: string,
    /** 알아낸 크기로 진짜 층을 만든다. */
    private readonly build: (inferred: number) => Module,
    /** 입력에서 무엇을 읽을지. 선형은 마지막 축, 나머지는 채널이다. */
    private readonly read: (x: Tensor) => number,
  ) {
    super();
  }

  hasUninitializedParams(): boolean {
    return this.built === null;
  }

  override forward(x: Tensor): Tensor {
    if (!this.built) {
      const real = this.build(this.read(x));
      // 속을 옮기고 프로토타입을 갈아 끼운다 — 이 뒤로 이 물건은 진짜 층이다.
      Object.assign(this, real);
      Object.setPrototypeOf(this, Object.getPrototypeOf(real));
      return (this as unknown as Module).forward(x);
    }
    return this.built.forward(x);
  }

  override describe(): string {
    return this.label;
  }
}

/** 입력에서 무엇을 읽을지. */
const lastAxis = (x: Tensor) => x.shape[x.shape.length - 1] ?? 0;
const channels = (x: Tensor) => x.shape[1] ?? 0;

export class LazyLinear extends LazyModule {
  constructor(outFeatures: number) {
    super(`LazyLinear(in_features=0, out_features=${outFeatures}, bias=True)`,
      (inF) => new Linear(inF, outFeatures), lastAxis);
  }
}

// 이름을 붙여 열둘. **찍어내는 함수로 두지 않는다** — 익명 클래스를 내보내면
// TypeScript 가 `Module` 의 비공개 자리를 이유로 거절한다. 패딩 층에서 같은 벽을
// 만났고, 상속을 무너뜨리는 것보다 세 줄씩 적는 편이 싸다.
class LazyConvBase extends LazyModule {
  constructor(spatial: number, transpose: boolean, outC: number, kernel: number,
              stride = 1, padding = 0, bias = true) {
    super(`Lazy${transpose ? "ConvTranspose" : "Conv"}${spatial}d`,
      (inC) => (transpose
        ? new ConvTransposeND(inC, outC, kernel, spatial, stride, padding, bias)
        : new ConvND(inC, outC, kernel, spatial, stride, padding, bias)),
      channels);
  }
}

type ConvArgs = [number, number, number?, number?, boolean?];

export class LazyConv1d extends LazyConvBase {
  constructor(...a: ConvArgs) { super(1, false, ...a); }
}
export class LazyConv2d extends LazyConvBase {
  constructor(...a: ConvArgs) { super(2, false, ...a); }
}
export class LazyConv3d extends LazyConvBase {
  constructor(...a: ConvArgs) { super(3, false, ...a); }
}
export class LazyConvTranspose1d extends LazyConvBase {
  constructor(...a: ConvArgs) { super(1, true, ...a); }
}
export class LazyConvTranspose2d extends LazyConvBase {
  constructor(...a: ConvArgs) { super(2, true, ...a); }
}
export class LazyConvTranspose3d extends LazyConvBase {
  constructor(...a: ConvArgs) { super(3, true, ...a); }
}

class LazyNormBase extends LazyModule {
  constructor(kind: "batch" | "instance", spatial: number, eps = 1e-5,
              momentum = 0.1) {
    super(`Lazy${kind === "batch" ? "BatchNorm" : "InstanceNorm"}${spatial}d`,
      (c) => (kind === "batch"
        ? new BatchNormND(c, eps, momentum)
        : new InstanceNormND(eps)),
      channels);
  }
}

export class LazyBatchNorm1d extends LazyNormBase {
  constructor(eps?: number, m?: number) { super("batch", 1, eps, m); }
}
export class LazyBatchNorm2d extends LazyNormBase {
  constructor(eps?: number, m?: number) { super("batch", 2, eps, m); }
}
export class LazyBatchNorm3d extends LazyNormBase {
  constructor(eps?: number, m?: number) { super("batch", 3, eps, m); }
}
export class LazyInstanceNorm1d extends LazyNormBase {
  constructor(eps?: number) { super("instance", 1, eps); }
}
export class LazyInstanceNorm2d extends LazyNormBase {
  constructor(eps?: number) { super("instance", 2, eps); }
}
export class LazyInstanceNorm3d extends LazyNormBase {
  constructor(eps?: number) { super("instance", 3, eps); }
}

// ── 손실 층 ─────────────────────────────────────────────────────────────
//
// **전부 같은 모양이다** — 만들 때 인자를 받아 두고 부를 때 텐서 메서드로 넘긴다.
// torch 의 손실 층이 하는 일이 그것뿐이라, 층마다 `forward` 를 적으면 같은 두 줄을
// 열세 번 적는 것이 된다.
//
// **torch 는 손실 층을 인자 없이 찍는다** — `HuberLoss(delta=0.5)` 도 `HuberLoss()`
// 로 나온다(실측). 글자가 답의 일부라 그대로 따른다.

/**
 * 어휘가 아주 클 때의 softmax. **자주 나오는 글자를 싸게 낸다.**
 *
 * 글자를 빈도순으로 묶어, 앞쪽 뭉치는 머리에서 바로 내고 뒤쪽 뭉치는 **머리가 그
 * 뭉치를 고른 확률 × 뭉치 안의 확률**로 낸다. 뒤쪽일수록 중간 차원을 `divValue` 로
 * 나눠 좁힌다 — 드문 글자에 자리를 덜 쓴다.
 *
 * **기본값이 `divValue=4`·`headBias=false`** 다(재봤다: `tests/probe_asm.py`).
 * 중간 차원은 `inFeatures / divValue**(i+1)` 을 내림한 것이고 **0 이 될 수 있다** —
 * torch 도 거기서 빈 층을 만들고 넘어가므로 막지 않는다.
 */
export class AdaptiveLogSoftmaxWithLoss extends Module {
  readonly cutoffs: number[];
  readonly shortlistSize: number;
  readonly nClusters: number;
  readonly headSize: number;
  readonly head: Linear;
  readonly tail: ModuleList;

  constructor(
    readonly inFeatures: number,
    readonly nClasses: number,
    cutoffs: readonly number[],
    readonly divValue = 4.0,
    readonly headBias = false,
  ) {
    super();
    this.cutoffs = [...cutoffs, nClasses];
    this.shortlistSize = this.cutoffs[0] ?? 0;
    this.nClusters = this.cutoffs.length - 1;
    this.headSize = this.shortlistSize + this.nClusters;
    this.head = new Linear(inFeatures, this.headSize, headBias);
    const tail: Module[] = [];
    for (let i = 0; i < this.nClusters; i++) {
      const hidden = Math.floor(inFeatures / divValue ** (i + 1));
      const out = (this.cutoffs[i + 1] ?? 0) - (this.cutoffs[i] ?? 0);
      tail.push(new Sequential(
        new Linear(inFeatures, hidden, false),
        new Linear(hidden, out, false),
      ));
    }
    this.tail = new ModuleList(tail);
  }

  override namedChildren(): Record<string, Module> {
    return { head: this.head, tail: this.tail };
  }

  override children(): Module[] {
    return [this.head, this.tail];
  }

  /**
   * 모든 글자의 로그확률 `(N, nClasses)`.
   *
   * 뭉치 안의 확률에 **머리가 그 뭉치를 고른 로그확률을 더한다** — 곱셈이 로그에서
   * 덧셈이고, 그래서 행 전체의 합이 1 로 남는다.
   */
  logProb(x: Tensor): Tensor {
    const head = this.head.call(x).logSoftmax(1);
    const parts: Tensor[] = [head.narrow(1, 0, this.shortlistSize)];
    for (let i = 0; i < this.nClusters; i++) {
      const cluster = this.tail.at(i);
      if (!cluster) continue;
      const inside = cluster.call(x).logSoftmax(1);
      const picked = head.narrow(1, this.shortlistSize + i, 1);
      parts.push(inside.add(picked));
    }
    return Tensor.cat(parts, 1);
  }

  /** `{ output, loss }` — 정답 자리의 로그확률과 그 평균의 음수. */
  run(x: Tensor, target: Tensor): { output: Tensor; loss: Tensor } {
    const lp = this.logProb(x);
    const rows = target.reshape([target.size, 1]);
    const output = lp.gather(1, rows).reshape([target.size]);
    return { output, loss: output.mean().neg() };
  }

  /** **정답도 받는 층이다.** 지나가려 하면 여기서 멈춘다 — `run(x, target)` 이 그 자리다. */
  override forward(): Tensor {
    throw new Error(
      "AdaptiveLogSoftmaxWithLoss 는 정답도 받는다 — `run(x, target)` 을 써라",
    );
  }

  predict(x: Tensor): Tensor {
    return this.logProb(x).argmax(1);
  }
}

/**
 * 소리와 글자를 **자리를 맞추지 않고** 잇는 손실.
 *
 * `logProbs` 는 `(T, N, C)` — **시간이 앞이다.** 표본마다 길이가 다르고 그것이 이
 * 손실의 요점이라, 길이 두 벌을 함께 받는다.
 *
 * ## 값이 0 이고 기울기만 있는 항을 하나 더한다
 *
 * torch 가 `logProbs` 로 흘리는 기울기는 참도함수가 아니다 — 유한차분은 `-γ` 인데
 * torch 는 `exp(logProbs) - γ` 를 낸다(실측: `tests/probe_ctc3.py`). 쓰는 자리에서는
 * 앞에 `logSoftmax` 가 있어서 둘이 같은 답이 되지만(그 역방향의 고정점이다),
 * `logProbs` 를 바로 잎으로 두면 수가 갈린다. 맞추되 **왜 맞추는지**를 적는다.
 *
 * @param reduction `"mean"` 은 표본마다 **제 표적 길이로 나눈 뒤** 평균한다 —
 *   그냥 평균이 아니다.
 */
export function ctcLoss(
  logProbs: Tensor,
  targets: readonly (readonly number[])[],
  inputLengths: readonly number[],
  targetLengths: readonly number[],
  blank = 0,
  reduction: Reduction = "mean",
  zeroInfinity = false,
): Tensor {
  const parts: Tensor[] = [];
  const divisors: number[] = [];
  for (let i = 0; i < targets.length; i++) {
    const labels = (targets[i] ?? []).slice(0, targetLengths[i] ?? 0);
    const nTime = inputLengths[i] ?? 0;
    // 붙어 있는 같은 글자마다 공백 한 칸이 더 든다. 그보다 짧으면 정렬이 하나도
    // 없어서 확률이 0 이고 손실이 무한이다 — 문턱값이 아니라 실제 조건이다.
    let needs = labels.length;
    for (let k = 1; k < labels.length; k++) if (labels[k] === labels[k - 1]) needs++;
    divisors.push(Math.max(labels.length, 1));
    if (nTime < needs) {
      parts.push(Tensor.full([1], zeroInfinity ? 0 : Infinity));
      continue;
    }
    const plane = logProbs.narrow(1, i, 1)
      .reshape([logProbs.shape[0] ?? 1, logProbs.shape[2] ?? 1]);
    const one = Tensor.ctcOne(plane, labels, nTime, blank);
    const bias = plane.narrow(0, 0, nTime).exp().sum();
    parts.push(one.add(bias.sub(bias.detach())).reshape([1]));
  }
  const per = Tensor.cat(parts, 0);
  if (reduction === "none") return per;
  if (reduction === "sum") return per.sum();
  return per.div(Tensor.from(divisors, [divisors.length])).mean();
}

export class CTCLoss {
  constructor(
    readonly blank = 0,
    readonly reduction: Reduction = "mean",
    readonly zeroInfinity = false,
  ) {}

  forward(
    logProbs: Tensor,
    targets: readonly (readonly number[])[],
    inputLengths: readonly number[],
    targetLengths: readonly number[],
  ): Tensor {
    return ctcLoss(logProbs, targets, inputLengths, targetLengths,
      this.blank, this.reduction, this.zeroInfinity);
  }

  call(
    logProbs: Tensor,
    targets: readonly (readonly number[])[],
    inputLengths: readonly number[],
    targetLengths: readonly number[],
  ): Tensor {
    return this.forward(logProbs, targets, inputLengths, targetLengths);
  }

  /** torch 의 `extra_repr` 가 아무것도 안 낸다 — 그래서 비어 있다. */
  describe(): string { return "CTCLoss()"; }
}

export class HuberLoss {
  constructor(readonly delta = 1.0, readonly reduction: Reduction = "mean") {
  }

  forward(x: Tensor, target: Tensor): Tensor {
    return x.huberLoss(target, this.delta, this.reduction);
  }

  call(x: Tensor, target: Tensor): Tensor {
    return this.forward(x, target);
  }

  describe(): string { return "HuberLoss()"; }
}

export class KLDivLoss {
  constructor(
    readonly reduction: Reduction | "batchmean" = "mean",
    readonly logTarget = false,
  ) {}

  forward(x: Tensor, target: Tensor): Tensor {
    return x.klDiv(target, this.reduction, this.logTarget);
  }

  call(x: Tensor, target: Tensor): Tensor {
    return this.forward(x, target);
  }

  describe(): string { return "KLDivLoss()"; }
}

export class PoissonNLLLoss {
  constructor(
    readonly logInput = true, readonly full = false, readonly eps = 1e-8,
    readonly reduction: Reduction = "mean",
  ) {}

  forward(x: Tensor, target: Tensor): Tensor {
    return x.poissonNllLoss(target, this.logInput, this.full, this.eps,
      this.reduction);
  }

  call(x: Tensor, target: Tensor): Tensor {
    return this.forward(x, target);
  }

  describe(): string { return "PoissonNLLLoss()"; }
}

export class GaussianNLLLoss {
  constructor(
    readonly full = false, readonly eps = 1e-6,
    readonly reduction: Reduction = "mean",
  ) {}

  forward(x: Tensor, target: Tensor, variance: Tensor): Tensor {
    return x.gaussianNllLoss(target, variance, this.full, this.eps, this.reduction);
  }

  call(x: Tensor, target: Tensor, variance: Tensor): Tensor {
    return this.forward(x, target, variance);
  }

  describe(): string { return "GaussianNLLLoss()"; }
}

export class MarginRankingLoss {
  constructor(readonly margin = 0.0, readonly reduction: Reduction = "mean") {
  }

  forward(x1: Tensor, x2: Tensor, target: Tensor): Tensor {
    return x1.marginRankingLoss(x2, target, this.margin, this.reduction);
  }

  call(x1: Tensor, x2: Tensor, target: Tensor): Tensor {
    return this.forward(x1, x2, target);
  }

  describe(): string { return "MarginRankingLoss()"; }
}

export class CosineEmbeddingLoss {
  constructor(readonly margin = 0.0, readonly reduction: Reduction = "mean") {
  }

  forward(x1: Tensor, x2: Tensor, target: Tensor): Tensor {
    return x1.cosineEmbeddingLoss(x2, target, this.margin, this.reduction);
  }

  call(x1: Tensor, x2: Tensor, target: Tensor): Tensor {
    return this.forward(x1, x2, target);
  }

  describe(): string { return "CosineEmbeddingLoss()"; }
}

export class HingeEmbeddingLoss {
  constructor(readonly margin = 1.0, readonly reduction: Reduction = "mean") {
  }

  forward(x: Tensor, target: Tensor): Tensor {
    return x.hingeEmbeddingLoss(target, this.margin, this.reduction);
  }

  call(x: Tensor, target: Tensor): Tensor {
    return this.forward(x, target);
  }

  describe(): string { return "HingeEmbeddingLoss()"; }
}

export class SoftMarginLoss {
  constructor(readonly reduction: Reduction = "mean") {}

  forward(x: Tensor, target: Tensor): Tensor {
    return x.softMarginLoss(target, this.reduction);
  }

  call(x: Tensor, target: Tensor): Tensor {
    return this.forward(x, target);
  }

  describe(): string { return "SoftMarginLoss()"; }
}

export class TripletMarginLoss {
  constructor(
    readonly margin = 1.0, readonly p = 2.0, readonly eps = 1e-6,
    readonly swap = false, readonly reduction: Reduction = "mean",
  ) {}

  forward(anchor: Tensor, positive: Tensor, negative: Tensor): Tensor {
    return anchor.tripletMarginLoss(positive, negative, this.margin, this.p,
      this.eps, this.swap, this.reduction);
  }

  call(anchor: Tensor, positive: Tensor, negative: Tensor): Tensor {
    return this.forward(anchor, positive, negative);
  }

  describe(): string { return "TripletMarginLoss()"; }
}

/** 거리 함수를 받는 삼중항. 기본값이 쌍별 거리라 위와 같은 답이 나온다. */
export class TripletMarginWithDistanceLoss {
  constructor(
    readonly distanceFunction: ((a: Tensor, b: Tensor) => Tensor) | null = null,
    readonly margin = 1.0, readonly swap = false,
    readonly reduction: Reduction = "mean",
  ) {}

  forward(anchor: Tensor, positive: Tensor, negative: Tensor): Tensor {
    const dist = this.distanceFunction ?? ((a: Tensor, b: Tensor) =>
      a.pairwiseDistance(b));
    const dp = dist(anchor, positive);
    let dn = dist(anchor, negative);
    if (this.swap) dn = dn.binary("minimum", dist(positive, negative));
    const out = dp.sub(dn).binary("add", Tensor.full([], this.margin)).unary("relu");
    return this.reduction === "none" ? out
      : (this.reduction === "sum" ? out.sum() : out.mean());
  }

  call(anchor: Tensor, positive: Tensor, negative: Tensor): Tensor {
    return this.forward(anchor, positive, negative);
  }

  describe(): string { return "TripletMarginWithDistanceLoss()"; }
}

export class MultiLabelSoftMarginLoss {
  constructor(readonly reduction: Reduction = "mean") {}

  forward(x: Tensor, target: Tensor): Tensor {
    return x.multilabelSoftMarginLoss(target, this.reduction);
  }

  call(x: Tensor, target: Tensor): Tensor {
    return this.forward(x, target);
  }

  describe(): string { return "MultiLabelSoftMarginLoss()"; }
}

export class MultiMarginLoss {
  constructor(
    readonly p = 1, readonly margin = 1.0, readonly weight: Tensor | null = null,
    readonly reduction: Reduction = "mean",
  ) {}

  forward(x: Tensor, target: Tensor): Tensor {
    return x.multiMarginLoss(target, this.p, this.margin, this.weight,
      this.reduction);
  }

  call(x: Tensor, target: Tensor): Tensor {
    return this.forward(x, target);
  }

  describe(): string { return "MultiMarginLoss()"; }
}

export class MultiLabelMarginLoss {
  constructor(readonly reduction: Reduction = "mean") {}

  forward(x: Tensor, target: Tensor): Tensor {
    return x.multilabelMarginLoss(target, this.reduction);
  }

  call(x: Tensor, target: Tensor): Tensor {
    return this.forward(x, target);
  }

  describe(): string { return "MultiLabelMarginLoss()"; }
}

/** 짝지어진 두 줄 사이의 거리. **`eps` 는 차에 더한다** — 텐서 쪽에 적었다. */
export class PairwiseDistance {
  constructor(readonly p = 2.0, readonly eps = 1e-6, readonly keepdim = false) {
  }

  forward(x1: Tensor, x2: Tensor): Tensor {
    return x1.pairwiseDistance(x2, this.p, this.eps, this.keepdim);
  }

  call(x1: Tensor, x2: Tensor): Tensor {
    return this.forward(x1, x2);
  }

  describe(): string { return "PairwiseDistance()"; }
}

export class CosineSimilarity {
  constructor(readonly dim = 1, readonly eps = 1e-8) {}

  forward(x1: Tensor, x2: Tensor): Tensor {
    return x1.cosineSimilarity(x2, this.dim, this.eps);
  }

  call(x1: Tensor, x2: Tensor): Tensor {
    return this.forward(x1, x2);
  }

  describe(): string { return "CosineSimilarity()"; }
}

/**
 * 패딩 층 열다섯 개의 뿌리.
 *
 * **셋(1·2·3 차원) × 다섯(reflect·replicate·zero·constant·circular)이 한 기계에서
 * 나온다.** 갈리는 것은 모드 이름과 짝의 개수뿐이라, 손으로 열다섯 벌을 적으면
 * 실제로 다른 두 가지를 열다섯 자리에 흩어 놓는 것이 된다.
 */
export class PadNd extends Module {
  readonly padding: number[];

  constructor(
    /** 찍을 이름. **`constructor.name` 에 안 기댄다** — 묶는 도구가 이름을 줄이면
     * 그때부터 조용히 다른 글자가 나오고, 골든은 우리 빌드만 본다. */
    readonly label: string,
    padding: number | readonly number[],
    readonly mode: PadMode,
    dims: number,
    readonly value = 0,
  ) {
    super();
    this.padding = typeof padding === "number"
      ? new Array<number>(2 * dims).fill(padding)
      : [...padding];
  }

  override forward(x: Tensor): Tensor {
    return x.padND(this.padding, this.mode, this.value);
  }

  /**
   * 파이썬이 찍는 그대로.
   *
   * **`ConstantPad` 만 이름을 붙여 찍는다** — 나머지는 짝만 찍는다. 진짜 torch 가
   * 그렇고 골든이 글자를 굳혔으므로 그 차이가 답의 일부다. 값도 파이썬 꼴이라
   * 정수여도 소수점을 단다(`value=7.0`).
   */
  override describe(): string {
    const pairs = `(${this.padding.join(", ")})`;
    if (!this.label.startsWith("ConstantPad")) return `${this.label}(${pairs})`;
    const v = Number.isInteger(this.value) ? `${this.value}.0` : String(this.value);
    return `${this.label}(padding=${pairs}, value=${v})`;
  }
}

// 이름을 붙여 열다섯. **찍어내는 함수로 두지 않는다** — 익명 클래스를 내보내면
// TypeScript 가 `Module` 의 비공개 자리를 이유로 거절하고, 그 자리를 열려고 상속을
// 무너뜨리는 것보다 세 줄씩 적는 편이 싸다.
export class ReflectionPad1d extends PadNd {
  constructor(p: number | readonly number[]) { super("ReflectionPad1d", p, "reflect", 1); }
}
export class ReflectionPad2d extends PadNd {
  constructor(p: number | readonly number[]) { super("ReflectionPad2d", p, "reflect", 2); }
}
export class ReflectionPad3d extends PadNd {
  constructor(p: number | readonly number[]) { super("ReflectionPad3d", p, "reflect", 3); }
}
export class ReplicationPad1d extends PadNd {
  constructor(p: number | readonly number[]) {
    super("ReplicationPad1d", p, "replicate", 1);
  }
}
export class ReplicationPad2d extends PadNd {
  constructor(p: number | readonly number[]) {
    super("ReplicationPad2d", p, "replicate", 2);
  }
}
export class ReplicationPad3d extends PadNd {
  constructor(p: number | readonly number[]) {
    super("ReplicationPad3d", p, "replicate", 3);
  }
}
export class CircularPad1d extends PadNd {
  constructor(p: number | readonly number[]) { super("CircularPad1d", p, "circular", 1); }
}
export class CircularPad2d extends PadNd {
  constructor(p: number | readonly number[]) { super("CircularPad2d", p, "circular", 2); }
}
export class CircularPad3d extends PadNd {
  constructor(p: number | readonly number[]) { super("CircularPad3d", p, "circular", 3); }
}
export class ZeroPad1d extends PadNd {
  constructor(p: number | readonly number[]) { super("ZeroPad1d", p, "constant", 1); }
}
export class ZeroPad2d extends PadNd {
  constructor(p: number | readonly number[]) { super("ZeroPad2d", p, "constant", 2); }
}
export class ZeroPad3d extends PadNd {
  constructor(p: number | readonly number[]) { super("ZeroPad3d", p, "constant", 3); }
}
export class ConstantPad1d extends PadNd {
  constructor(p: number | readonly number[], v = 0) {
    super("ConstantPad1d", p, "constant", 1, v);
  }
}
export class ConstantPad2d extends PadNd {
  constructor(p: number | readonly number[], v = 0) {
    super("ConstantPad2d", p, "constant", 2, v);
  }
}
export class ConstantPad3d extends PadNd {
  constructor(p: number | readonly number[], v = 0) {
    super("ConstantPad3d", p, "constant", 3, v);
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
// ── 공간 변환기 ──────────────────────────────────────────────────────────
//
// `affineGrid` 가 "출력의 이 칸은 입력의 어디를 보는가" 를 적은 격자를 만들고,
// `gridSample` 이 그 자리에서 값을 떠 온다. 사이의 `theta` 가 학습된다.
//
// **새 커널을 안 쓴다.** 자리 번호도 텐서로 둘 수 있어서(`floor` 한 값을
// `indexSelect` 의 번호로 넘긴다) 있는 연산만으로 짜인다. 작은 커널이 여럿 도는
// 대신 입력과 격자 양쪽 기울기가 저절로 따라온다 — 흩뿌리는 역방향을 손으로 적으면
// 스레드끼리 같은 칸에 쓰게 되고, 그 답은 실행마다 달라질 수 있다.

/** `[-1, 1]` 위의 표본 자리. `alignCorners` 가 끝을 못 박느냐 칸 가운데를 잡느냐를 가른다. */
function gridBase(n: number, alignCorners: boolean): number[] {
  if (alignCorners) {
    if (n <= 1) return [0];
    return Array.from({ length: n }, (_, i) => -1 + (2 * i) / (n - 1));
  }
  return Array.from({ length: n }, (_, i) => (2 * i + 1) / n - 1);
}

/**
 * `theta` 가 그리는 표본 격자. `(N, 2, 3)` → `(N, H, W, 2)`.
 *
 * 마지막 축은 **`(x, y)` 순서다** — 모양의 `(H, W)` 와 뒤집혀 있다. 정사각에서는
 * 뒤집어 적어도 답이 같아서 직사각으로 물어야 드러난다.
 */
export function affineGrid(
  theta: Tensor,
  size: readonly number[],
  alignCorners = false,
): Tensor {
  const n = size[0] ?? 1;
  const h = size[2] ?? 1;
  const w = size[3] ?? 1;
  const xs = gridBase(w, alignCorners);
  const ys = gridBase(h, alignCorners);
  // 균질좌표 `(x, y, 1)` — 이동까지 한 번의 곱으로 끝낸다.
  const flat: number[] = [];
  for (let i = 0; i < h; i++) {
    for (let j = 0; j < w; j++) flat.push(xs[j] ?? 0, ys[i] ?? 0, 1);
  }
  // 배치마다 같은 격자에 제 `theta` 를 곱한다. `bmm` 이 3 차원끼리라 배치를 맞춰 편다.
  const base = Tensor.from(flat, [h * w, 3]);
  const parts: Tensor[] = [];
  for (let b = 0; b < n; b++) {
    parts.push(base.mm(theta.select(0, b).permute([1, 0])));
  }
  return Tensor.stack(parts, 0).reshape([n, h, w, 2]);
}

/** `[-1, 1]` 을 입력의 칸 번호로. `gridBase` 의 반대다. */
function gridDenorm(g: Tensor, n: number, alignCorners: boolean): Tensor {
  if (alignCorners) return g.add(Tensor.full([], 1)).mul(Tensor.full([], (n - 1) / 2));
  return g.add(Tensor.full([], 1)).mul(Tensor.full([], n)).sub(Tensor.full([], 1))
    .mul(Tensor.full([], 0.5));
}

/**
 * 범위 밖을 **되접는다.** 되접는 구간이 `alignCorners` 로 갈린다 —
 * 참이면 `[0, n−1]`, 거짓이면 `[−0.5, n−0.5]` 다(실측). 되접은 뒤 한 번 더 자른다.
 */
function gridReflect(v: Tensor, n: number, alignCorners: boolean): Tensor {
  const lo = alignCorners ? 0 : -0.5;
  const hi = alignCorners ? n - 1 : n - 0.5;
  if (hi <= lo) return v.mul(Tensor.full([], 0)).add(Tensor.full([], lo));
  const span = 2 * (hi - lo);
  const t = v.sub(Tensor.full([], lo)).remainder(span);
  const folded = t.binary("minimum", Tensor.full([], span).sub(t));
  return folded.add(Tensor.full([], lo)).clamp(0, n - 1);
}

/**
 * 격자가 가리키는 자리에서 값을 떠 온다. `affineGrid` 의 짝이다.
 *
 * **자리 번호는 상수, 무게는 텐서다.** 내림한 정수는 미분이 없고 그 나머지가 무게가
 * 되므로, 무게만 그래프에 두면 입력과 격자 양쪽으로 기울기가 흐른다.
 */
export function gridSample(
  x: Tensor,
  grid: Tensor,
  mode: "bilinear" | "nearest" = "bilinear",
  paddingMode: "zeros" | "border" | "reflection" = "zeros",
  alignCorners = false,
): Tensor {
  const N = x.shape[0] ?? 1;
  const C = x.shape[1] ?? 1;
  const H = x.shape[2] ?? 1;
  const W = x.shape[3] ?? 1;
  const OH = grid.shape[1] ?? 1;
  const OW = grid.shape[2] ?? 1;
  const cells = OH * OW;

  const g2 = grid.reshape([N, cells, 2]);
  let sx = gridDenorm(g2.narrow(2, 0, 1).reshape([N, cells]), W, alignCorners);
  let sy = gridDenorm(g2.narrow(2, 1, 1).reshape([N, cells]), H, alignCorners);
  if (paddingMode === "border") {
    sx = sx.clamp(0, W - 1);
    sy = sy.clamp(0, H - 1);
  } else if (paddingMode === "reflection") {
    sx = gridReflect(sx, W, alignCorners);
    sy = gridReflect(sy, H, alignCorners);
  }

  // 평면마다의 시작 번호. `(N, C)` 를 미리 펴 두면 모서리마다 더하기만 하면 된다.
  const planeStart: number[] = [];
  for (let n = 0; n < N; n++) {
    for (let c = 0; c < C; c++) planeStart.push((n * C + c) * H * W);
  }
  const starts = Tensor.from(planeStart, [N, C, 1]);
  const source = x.reshape([x.size]);

  /** 모서리 하나를 떠 온다. **범위 밖은 0 이되 번호는 잘라서 넘긴다.** */
  const pick = (iy: Tensor, ix: Tensor): Tensor => {
    const inside = iy.binary("ge", Tensor.full([], 0))
      .mul(iy.binary("lt", Tensor.full([], H)))
      .mul(ix.binary("ge", Tensor.full([], 0)))
      .mul(ix.binary("lt", Tensor.full([], W)));
    const cy = iy.clamp(0, H - 1);
    const cx = ix.clamp(0, W - 1);
    const offset = cy.mul(Tensor.full([], W)).add(cx).reshape([N, 1, cells]);
    const flat = starts.add(offset).reshape([N * C * cells]);
    const got = source.indexSelect(0, flat).reshape([N, C, cells]);
    return got.mul(inside.reshape([N, 1, cells]));
  };

  const shaped = (t: Tensor) => t.reshape([N, C, OH, OW]);
  if (mode === "nearest") {
    // torch 는 반올림한다. 무게가 없으므로 격자로는 기울기가 안 간다.
    return shaped(pick(sy.round(), sx.round()));
  }
  const x0 = sx.floor();
  const y0 = sy.floor();
  const wx = sx.sub(x0).reshape([N, 1, cells]);
  const wy = sy.sub(y0).reshape([N, 1, cells]);
  const one = Tensor.full([], 1);
  const x1 = x0.add(one);
  const y1 = y0.add(one);
  const out = pick(y0, x0).mul(one.sub(wy)).mul(one.sub(wx))
    .add(pick(y0, x1).mul(one.sub(wy)).mul(wx))
    .add(pick(y1, x0).mul(wy).mul(one.sub(wx)))
    .add(pick(y1, x1).mul(wy).mul(wx));
  return shaped(out);
}

/**
 * `BatchNormND` 의 함수 꼴. **층이 이것을 부른다** — 식을 한 벌만 둔다.
 *
 * **학습이면 이동 통계를 제자리에서 고친다.** torch 가 그렇고, 넘긴 텐서가 갱신되어
 * 돌아온다. 새것을 돌려주면 부르는 쪽의 버퍼가 안 움직여서 학습은 도는데 평가 모드의
 * 값만 틀린다.
 *
 * **분산을 두 가지로 쓴다.** 정규화는 편향추정, 이동 통계 갱신은 불편추정이다.
 * 하나로 합치면 평가 모드에서만 갈린다.
 */
export function batchNorm(
  x: Tensor,
  runningMean: Tensor | null,
  runningVar: Tensor | null,
  weight: Tensor | null,
  bias: Tensor | null,
  training = false,
  momentum = 0.1,
  eps = 1e-5,
): Tensor {
  const channels = x.shape[1] ?? 1;
  const spatial = x.shape.length - 2;
  const shape = [1, channels, ...new Array<number>(spatial).fill(1)];
  const w = weight ?? Tensor.ones([channels]);
  const b = bias ?? Tensor.zeros([channels]);
  if (!training) {
    if (!runningMean || !runningVar) {
      throw new Error("batchNorm: 평가 모드에는 이동 통계가 있어야 한다");
    }
    const centered = x.sub(runningMean.reshape(shape));
    const scaled = centered.div(
      runningVar.reshape(shape).binary("add", Tensor.full([], eps)).sqrt());
    return scaled.mul(w.reshape(shape)).add(b.reshape(shape));
  }
  // **융합 커널로 간다.** 조립판은 층 하나에 dispatch 스무 개가 넘었고, ResNet 한
  // 스텝의 1,636 개 중 태반이 거기서 나왔다(실측).
  const { out, mean, variance } = x.batchNormFused(w, b, eps);
  if (runningMean && runningVar) {
    // 둘을 커널 하나로 갱신한다. 조립판은 층마다 여덟 dispatch 였고 층이 스무 개다.
    const count = x.size / channels;
    const unbias = count / (count - 1);
    const d = device();
    d.run1d(
      d.pipeline(`rs:${channels}:${momentum}:${unbias}`,
        () => runningStats(channels, momentum, unbias)),
      [runningMean.buffer, runningVar.buffer, mean.buffer, variance.buffer],
      channels,
    );
  }
  return out;
}

/**
 * 가방마다 한 줄. 표에서 골라 **합치는 것**까지가 한 함수다.
 *
 * `offsets` 를 주면 1 차원 번호 줄을 가방으로 자른다 — 가방 길이가 제각각인 자리다.
 * `perSampleWeights` 는 torch 에서 `mode='sum'` 일 때만 쓴다.
 */
export function embeddingBag(
  idx: Tensor,
  weight: Tensor,
  offsets: readonly number[] | null = null,
  mode: "sum" | "mean" | "max" = "mean",
  perSampleWeights: Tensor | null = null,
): Tensor {
  const dim = weight.shape[1] ?? 1;
  const squash = (picked: Tensor, d: number) => {
    if (mode === "sum") return picked.sumDim(d, false);
    if (mode === "max") return picked.amax(d, false);
    return picked.mean(d, false);
  };
  if (offsets === null) {
    const bags = idx.shape[0] ?? 1;
    const each = idx.shape[1] ?? 1;
    let picked = weight.indexSelect(0, idx.reshape([bags * each]))
      .reshape([bags, each, dim]);
    if (perSampleWeights) {
      picked = picked.mul(perSampleWeights.reshape([bags, each, 1]));
    }
    return squash(picked, 1);
  }
  const bounds = [...offsets, idx.size];
  const parts: Tensor[] = [];
  for (let b = 0; b + 1 < bounds.length; b++) {
    const from = bounds[b] ?? 0;
    const len = (bounds[b + 1] ?? idx.size) - from;
    let picked = weight.indexSelect(0, idx.narrow(0, from, len));
    if (perSampleWeights) {
      picked = picked.mul(perSampleWeights.narrow(0, from, len).reshape([len, 1]));
    }
    parts.push(squash(picked, 0));
  }
  return Tensor.stack(parts, 0);
}

/**
 * 무작위로 하나를 고르되 **미분이 흐르게** 고른다.
 *
 * 범주 하나를 뽑는 것은 미분이 안 되는 일이라, Gumbel 잡음을 더해 `softmax` 로
 * 부드럽게 만든다. `tau` 가 작을수록 한쪽으로 몰린다.
 *
 * `hard=true` 면 답은 0/1 이지만 **기울기는 부드러운 쪽 것을 쓴다** —
 * `hard - soft.detach() + soft` 라는 흔한 수법이고, 값은 hard 이고 미분은 soft 다.
 * 그 둘을 갈라 두지 않으면 이 함수의 뜻이 없어진다.
 *
 * @param noise 이미 뽑아 둔 Gumbel 잡음. 안 주면 여기서 뽑는다 — 골든은 값을 못
 *   묻고 성질만 묻지만, 부르는 쪽이 정해진 잡음으로 재현하고 싶을 때가 있다.
 */
export function gumbelSoftmax(
  logits: Tensor,
  tau = 1.0,
  hard = false,
  dim = -1,
  noise: Tensor | null = null,
): Tensor {
  const axis = dim < 0 ? dim + logits.shape.length : dim;
  // Gumbel 잡음 — `-log(-log(u))`. 균등난수 하나에서 만든다.
  const eps = 1e-10;
  const g = noise ?? Tensor.uniform(logits.shape)
    .binary("add", Tensor.full([], eps)).log().neg()
    .binary("add", Tensor.full([], eps)).log().neg();
  const soft = logits.add(g).div(Tensor.full([], tau)).softmax(axis);
  if (!hard) return soft;
  const picked = soft.max(axis);
  const onehot = soft.oneHotAlong(picked.indices, axis);
  return onehot.sub(soft.detach()).add(soft);
}

export class BatchNormND extends Module {
  readonly weight: Tensor;
  readonly bias: Tensor;
  readonly runningMean: Tensor;
  readonly runningVar: Tensor;

  constructor(
    readonly channels: number,
    private readonly eps = 1e-5,
    private readonly momentum = 0.1,
  ) {
    super();
    this.weight = Tensor.owned([channels], 1);
    this.bias = Tensor.owned([channels], 0);
    this.claim(this.weight, this.bias);
    this.runningMean = Tensor.owned([channels], 0);
    this.runningVar = Tensor.owned([channels], 1);
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
    // **계산은 `batchNorm` 이 한다.** 층과 함수가 각자 적으면 언젠가 갈리고,
    // 갈리는 자리가 이동 통계라 학습은 멀쩡하고 평가만 틀린다.
    return batchNorm(x, this.runningMean, this.runningVar, this.weight,
      this.bias, this.training, this.momentum, this.eps);
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
/**
 * 어텐션의 함수 꼴. 가중치를 밖에서 받는다.
 *
 * **입력이 `(L, N, E)` 다 — 길이가 앞이다.** torch 의 같은 이름이 그렇고, 배치를
 * 앞에 두고 부르면 조용히 다른 축을 섞는다.
 *
 * **가림막은 더하는 실수다.** 참·거짓 표를 여기서 안 받는다 — `-inf` 를 더해
 * softmax 가 0 을 내게 하는 것이지 0 을 곱하는 것이 아니고, 곱하면 이미 정규화한
 * 뒤라 남은 자리가 1 로 안 돌아간다. 참·거짓을 실수로 바꾸는 일은 torch 의 계약을
 * 흉내내는 자리(파이썬 결속)에서 한다.
 *
 * @returns 출력 `(L, N, E)` 과 가중치. `averageWeights` 면 `(N, L, S)`, 아니면
 *   머리마다 `(N, H, L, S)` 다.
 */
export function multiHeadAttentionForward(
  query: Tensor,
  key: Tensor,
  value: Tensor,
  numHeads: number,
  inWeight: Tensor,
  inBias: Tensor | null,
  outWeight: Tensor,
  outBias: Tensor | null,
  attnMask: Tensor | null = null,
  keyPaddingMask: Tensor | null = null,
  averageWeights = true,
): { output: Tensor; weights: Tensor } {
  const L = query.shape[0] ?? 1;
  const N = query.shape[1] ?? 1;
  const E = query.shape[2] ?? 1;
  const S = key.shape[0] ?? 1;
  const head = E / numHeads;
  const scale = Tensor.full([], 1 / Math.sqrt(head));

  /** 길이를 앞에 둔 것을 배치 앞으로 돌리고 투영한다. */
  const project = (t: Tensor, len: number, slot: number): Tensor => {
    const flat = t.permute([1, 0, 2]).reshape([N * len, E]);
    const w = inWeight.narrow(0, slot * E, E);
    const out = flat.linear(w);
    return (inBias ? out.add(inBias.narrow(0, slot * E, E)) : out)
      .reshape([N, len, E]);
  };
  const q = project(query, L, 0);
  const k = project(key, S, 1);
  const v = project(value, S, 2);

  const rows: Tensor[] = [];
  const allWeights: Tensor[] = [];
  for (let n = 0; n < N; n++) {
    const perHead: Tensor[] = [];
    const perHeadWeights: Tensor[] = [];
    const pad = keyPaddingMask ? keyPaddingMask.select(0, n).reshape([1, S]) : null;
    for (let h = 0; h < numHeads; h++) {
      const cut = (t: Tensor, len: number) =>
        t.select(0, n).narrow(1, h * head, head).reshape([len, head]);
      let scores = cut(q, L).mm(cut(k, S).transpose()).binary("mul", scale);
      if (attnMask) scores = scores.add(attnMask);
      if (pad) scores = scores.add(pad);
      const w = scores.softmax(1);
      perHeadWeights.push(w.reshape([1, L, S]));
      perHead.push(w.mm(cut(v, S)));
    }
    rows.push(Tensor.cat(perHead, 1));                 // (L, E)
    allWeights.push(Tensor.cat(perHeadWeights, 0).reshape([1, numHeads, L, S]));
  }
  const merged = Tensor.stack(rows, 0).reshape([N * L, E]);
  const projected = merged.linear(outWeight);
  const out = (outBias ? projected.add(outBias) : projected)
    .reshape([N, L, E]).permute([1, 0, 2]);
  const weights = Tensor.cat(allWeights, 0);           // (N, H, L, S)
  return {
    output: out,
    weights: averageWeights ? weights.mean(1, false) : weights,
  };
}

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
    this.inBias = Tensor.owned([3 * embed], 0);
    this.outWeight = uniform([embed, embed], bound);
    this.outBias = Tensor.owned([embed], 0);
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
