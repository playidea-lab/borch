/**
 * Layers — the minimal skeleton of `nn.Module`.
 *
 * ## What it is for
 *
 * Unit comparison looks at one operation at a time. Some things diverge
 * only once modules, losses and optimizers are wired together, and every
 * defect this repository caught in an integrated scenario came from exactly
 * there. So building layers is not a convenience — it is **seeing a place
 * that could not be seen.**
 *
 * ## The name is the contract
 *
 * A `state_dict` key is built from **the position index and the name**, as
 * in `0.weight`. The golden cases put weights in by that name and take
 * results out by that name, so a divergent name means divergent wiring, not
 * divergent values.
 */

import { RuntimeError, ValueError } from "./errors.js";
import { runningStats } from "./kernels.js";
import { onSeed, uniform as uniform01, uniformArray } from "./random.js";
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

/**
 * One layer. It passes values through and hands out its parameters with
 * their names.
 */
export abstract class Module {
  /**
   * Whether it is in training mode. Layers that behave differently by mode,
   * such as `BatchNorm`, look at it.
   */
  training = true;

  abstract forward(x: Tensor): Tensor;

  /**
   * So it can be called — the same position as torch's `model(x)`.
   */
  call(x: Tensor): Tensor {
    return this.forward(x);
  }

  /**
   * The parameters this layer holds directly. Children come from
   * `children`.
   *
   * **The default is to sweep its own fields.** It used to return `{}` and
   * have each layer override, and then the moment someone building a layer
   * from outside forgets the override, that parameter is absent from
   * `parameters()` and — **with no exception** — training simply does not
   * happen. The kind where the loss will not come down and nothing points
   * at why.
   *
   * A comment said "TypeScript has no place to recognise layers by sweeping
   * properties", and that was not true. `Object.entries(this)` gives the
   * instance fields — the same position torch's `__setattr__` occupies.
   *
   * **`requiresGrad` is the mark.** Counting every tensor in a field as a
   * parameter would have the optimizer stepping constants and masks too.
   * The distinction torch draws with `nn.Parameter` is drawn here by this
   * flag — a layer raises it with `claim()`, and that is what "this is a
   * parameter" means.
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
   * Gives the children **with their names.** The default is **the field
   * name**, as in torch.
   *
   * ```ts
   * class Net extends nn.Module {
   *   fc1 = new nn.Linear(4, 8);      // → "fc1.weight", "fc1.bias"
   *   fc2 = new nn.Linear(8, 2);
   *   forward(x) { return this.fc2.call(this.fc1.call(x)); }
   * }
   * ```
   *
   * **Arrays are not swept.** `layers = [a, b]` is not a child — torch does
   * not register a Python list either, and asks for `nn.ModuleList`.
   * Sweeping arrays leaves no way to tell them from "an array that is not
   * layers", and without that distinction the `state_dict` keys change
   * quietly.
   *
   * Containers that want to be addressed by position (`Sequential`,
   * `ModuleList`) override here.
   *
   * **The `state_dict` keys are these names.** Diverge and you cannot read
   * anyone else's checkpoint.
   */
  namedChildren(): Record<string, Module> {
    const out: Record<string, Module> = {};
    for (const [name, value] of Object.entries(this)) {
      if (value instanceof Module) out[name] = value;
    }
    return out;
  }

  /**
   * The text `print(model)` prints. The same shape as torch's `__repr__`.
   *
   * **A child's several lines are re-indented too.** When a container holds
   * a container (a `Sequential` inside a `ModuleList`), the inner one
   * sitting flush left makes the picture differ from torch's — the kind
   * where the values are fine and only the text is wrong, so only an eye
   * catches it.
   *
   * A layer that wants to print a name or its arguments overrides this.
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

  /**
   * Labels with the position name prefixed, as in `0.weight`.
   */
  namedParameters(prefix = ""): Record<string, Tensor> {
    // **`children()` 만 덮어쓰면 그 자식은 안 배운다.**
    //
    // 파라미터를 모으는 것은 `namedChildren()` 뿐이다. 둘이 어긋나면 층은 눈에
    // 보이는데 파라미터가 안 잡히고, **예외도 경고도 없이 학습만** 그 자리에서
    // 멈춘다 — 손실은 내려간다. 나머지가 대신 맞추기 때문이다.
    //
    // 벤치의 ResNet-18 이 그 상태였다. 지름길을 평범한 객체(`{conv, bn}`)에 담고
    // `children()` 에만 적어 두어서, 지름길 층 여섯이 한 번도 안 배운 채로 에폭
    // 시간을 재고 있었다. 그것을 붙잡은 것은 값 검사가 아니라 **죽은 텐서 가드**다 —
    // 옵티마이저가 못 보는 잎은 `zeroGrad()` 도 못 받아 지난 스텝의 기울기가 남고,
    // 그 버퍼는 이미 통에 돌아간 것이었다.
    const kids = this.namedChildren();
    const listed = this.children().length;
    if (listed !== Object.keys(kids).length) {
      throw new RuntimeError(
        `${this.constructor.name}: children() lists ${listed}, but ` +
          `namedChildren() finds ${Object.keys(kids).length}.\n` +
          "  Parameters are gathered by namedChildren() — whatever is missing is " +
          "**silently left out of training**.\n" +
          "  If the children sit in a plain object or array, make them fields, or " +
          "put them in nn.ModuleList / nn.ModuleDict, and override " +
          "namedChildren() rather than children().",
      );
    }
    const out: Record<string, Tensor> = {};
    for (const [name, p] of Object.entries(this.ownParameters())) {
      out[`${prefix}${name}`] = p;
    }
    for (const [name, child] of Object.entries(kids)) {
      Object.assign(out, child.namedParameters(`${prefix}${name}.`));
    }
    return out;
  }

  parameters(): Tensor[] {
    return Object.values(this.namedParameters());
  }

  /**
   * Keeps parameters alive outside a scope.
   *
   * Build a layer **inside** a training scope and its weights are released
   * when that scope closes. Building outside is the correct way, but
   * nailing it down here is better than going quietly strange when the
   * correct way is not followed.
   */
  protected claim(...params: Tensor[]): void {
    for (const p of params) {
      p.requiresGrad = true;
      keepAlive(p);
    }
  }

  /**
   * Puts weights in from outside.
   *
   * **It moves values only; it does not swap the tensor.** Swapping makes
   * it a different tensor from the one the optimizer holds, and then
   * training runs while the parameters do not move.
   */
  loadStateDict(values: Readonly<Record<string, Tensor>>, strict = true): void {
    // **버퍼도 받는다.** 내보내는 것과 받는 것이 다르면 자기가 저장한 파일을
    // 자기가 못 읽는다 — `stateDict()` 는 버퍼를 실어 보내는데 여기는
    // `namedParameters()` 만 보고 있어서 strict 로는 "모르는 이름" 이 났다.
    // 저장과 복원은 **같은 목록**을 봐야 한다.
    const own = { ...this.namedParameters(), ...this.namedBuffers() };
    for (const [name, src] of Object.entries(values)) {
      const dst = own[name];
      if (!dst) {
        if (strict) throw new Error(`load_state_dict: unexpected key '${name}'`);
        continue;
      }
      noGrad(() => dst.copyFrom(src));
    }
  }

  stateDict(): Record<string, Tensor> {
    return { ...this.namedParameters(), ...this.namedBuffers(true) };
  }

  /**
   * A value that is not trained but is saved and restored. torch's
   * `register_buffer`.
   *
   * **There used to be no such thing.** The only route was `BatchNormND`
   * writing its own `stateDict` by hand to carry the running statistics,
   * which made buffers **that layer's special case.** Every model carrying
   * masks, position tables or normalisation constants uses this idiom, and
   * it was unavailable.
   *
   * With `persistent=false` it drops out of `stateDict` — the place for a
   * cache that can be rebuilt rather than checkpointed.
   */
  registerBuffer(name: string, value: Tensor, persistent = true): void {
    this.bufferNames.set(name, persistent);
    // 구역이 닫혀도 살아야 한다 — 파라미터가 아니라 아무도 안 잡아 준다.
    keepAlive(value);
    (this as unknown as Record<string, Tensor>)[name] = value;
  }

  /** 이름 → 저장되는가. 필드 자체는 층에 그대로 붙어 있고 여기는 표식만 든다. */
  private readonly bufferNames = new Map<string, boolean>();

  /**
   * A list differing from `namedParameters` by **exactly the buffers.**
   *
   * Its own come from the registered table and its children's by asking the
   * children. A layer that writes `stateDict` by hand (`BatchNormND`) has
   * to override this too, or the three fall out of step.
   */
  namedBuffers(persistentOnly = false): Record<string, Tensor> {
    const out: Record<string, Tensor> = {};
    for (const [name, persistent] of this.bufferNames) {
      if (persistentOnly && !persistent) continue;
      const got = (this as unknown as Record<string, unknown>)[name];
      if (got instanceof Tensor) out[name] = got;
    }
    for (const [prefix, child] of Object.entries(this.namedChildren())) {
      for (const [name, t] of Object.entries(child.namedBuffers(persistentOnly))) {
        out[`${prefix}.${name}`] = t;
      }
    }
    return out;
  }

  buffers(): Tensor[] {
    return Object.values(this.namedBuffers());
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

/**
 * Layers stood in a row. The position index is the name.
 */
export class Sequential extends Module {
  private readonly layers: Module[];

  /**
   * **Layers are simply listed** — `new Sequential(a, b, c)`. That is
   * torch's shape.
   *
   * Taking a single array made the very first example in `index.ts` wrong.
   * Someone copying torch code across leaves the brackets off by default,
   * and the default should be the one that is right. An array is still
   * accepted — there is already code written that way.
   */
  constructor(...layers: readonly (Module | readonly Module[])[]) {
    super();
    this.layers = layers.flatMap((l) => (Array.isArray(l) ? [...l] : [l as Module]));
  }

  override children(): Module[] {
    return this.layers;
  }

  /**
   * **Addressed by position index** — `0.weight`. That is torch's
   * `Sequential`'s shape, and the golden cases put weights in by that name.
   *
   * The default implementation uses field names, and our layers live inside
   * an array (`layers`) which that does not catch. That would leave
   * `state_dict` entirely empty, so this has to be overridden.
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
 * A list of layers. The index is the name — `layers.0.weight`.
 *
 * What differs from `Sequential` is that it **does not call them.** In what
 * order and how they are used is up to the holder; this side only makes the
 * parameters visible. Models whose layer count is not fixed use it.
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

  /**
   * Addressed by position index. Same reason as `Sequential` — the layers
   * live in an array.
   */
  override namedChildren(): Record<string, Module> {
    const out: Record<string, Module> = {};
    for (const [i, child] of this.items.entries()) out[String(i)] = child;
    return out;
  }

  /**
   * **It is not a layer you call.** Trying to pass through stops here — as
   * in torch.
   */
  override forward(): Tensor {
    throw new Error("ModuleList is not callable — pick a module inside it");
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
    if (!got) throw new Error(`ModuleList has no index ${i} (length ${this.items.length})`);
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

/**
 * A named bundle of layers. The name you give becomes the `stateDict` key
 * as-is.
 */
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
    throw new Error("ModuleDict is not callable — pick a module inside it");
  }

  at(key: string): Module {
    const got = this.items.get(key);
    if (!got) throw new Error(`ModuleDict has no key '${key}'`);
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
 * A list of trained tensors. **Without this there is no substitute.**
 *
 * A parameter not attached to a layer is caught nowhere unless
 * `ownParameters` is written by hand. Uncaught means the optimizer cannot
 * see it, unseen means it is not updated, and yet **the loss comes down** —
 * the remaining parameters compensate. No exception, no warning.
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
    throw new Error("ParameterList is not callable");
  }

  append(param: Tensor): this {
    this.claim(param);
    this.items.push(param);
    return this;
  }

  at(i: number): Tensor {
    const got = this.items.at(i);
    if (!got) throw new Error(`ParameterList has no index ${i} (length ${this.items.length})`);
    return got;
  }

  get length(): number {
    return this.items.length;
  }
}

/**
 * A named bundle of parameters. It exists for the same reason as
 * `ParameterList`.
 */
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
    throw new Error("ParameterDict is not callable");
  }

  at(key: string): Tensor {
    const got = this.items.get(key);
    if (!got) throw new Error(`ParameterDict has no key '${key}'`);
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

/**
 * Where `nn.Parameter` goes. **It stands up a new leaf.**
 *
 * ```ts
 * class Net extends nn.Module {
 *   w = new nn.Parameter(Tensor.zeros([4]));
 *   forward(x: Tensor) { return x.mul(this.w); }
 * }
 * ```
 *
 * ## Two places it parts from torch — measured
 *
 * **Storage is not shared.** torch's `nn.Parameter(t)` sees the same
 * storage as `t`, so editing one changes the other; having no views, we
 * **copy the values.** It is the same reason views are refused, and the
 * Python binding made the same choice at the same place — two GPU
 * implementations diverging from each other is worse than diverging from
 * torch.
 *
 * **`requiresGrad` alone makes something a parameter.** torch puts only
 * what is wrapped in `Parameter` into `named_parameters()` and leaves an
 * ordinary `requires_grad=True` tensor out (measured). Here
 * `ownParameters()` looks at the flag, so both come in. Moving the rule
 * towards torch would make today's code, which raises the flag with
 * `claim()`, **quietly lose parameters** — writing the divergence down is
 * cheaper. `parity.ts` pins these two by value.
 */
export const Parameter = function (t: Tensor, requiresGrad = true): Tensor {
  const p = t.clone().detach();
  p.requiresGrad = requiresGrad;
  keepAlive(p);
  return p;
} as unknown as { new (t: Tensor, requiresGrad?: boolean): Tensor };

/**
 * `y = x·Wᵀ + b`. The weight is `(out, in)`, as in torch.
 */
export class Linear extends Module {
  readonly weight: Tensor;
  /**
   * **It may be absent.** The tail layer of `AdaptiveLogSoftmaxWithLoss`
   * uses no bias.
   */
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
   * Exactly what Python prints.
   *
   * **When a lazy layer solidifies, this text is the answer** — from then
   * on the thing is a `Linear`, and this is what the user sees from
   * `print(model)`.
   */
  override describe(): string {
    const [out, inF] = [this.weight.shape[0] ?? 0, this.weight.shape[1] ?? 0];
    return `Linear(in_features=${inF}, out_features=${out}, `
      + `bias=${this.bias ? "True" : "False"})`;
  }
}

/**
 * A convolution layer independent of dimensionality. `spatial` is the
 * number of spatial axes.
 *
 * `Conv1d`, `Conv2d` and `Conv3d` differ only in that number — stand up
 * three classes and a day comes when only one of them is fixed.
 */
export class ConvND extends Module {
  readonly weight: Tensor;
  /**
   * The bias may be absent — with normalisation following, the bias is
   * absorbed as a constant term.
   */
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

// ── 결속이 손으로 메꾸고 있던 여덟. **이름은 껍데기인데 인자는 진짜다.** ─────
//
// 파이썬 결속(`borch_webgpu/_nn.py`)이 텐서 메서드 위에 factory 로 만들어 두어서
// 골든은 이 여덟이 없는 것을 **구조적으로 못 봤다** — 케이스가 전부 결속을 지나기
// 때문이다. TypeScript 로 `new nn.GELU()` 를 쓰는 사람에게만 없는 이름이었다.
//
// 옮기면서 셋이 인자를 갖고 있는 것이 드러났다(실측): `GELU(approximate)` 는 식이
// 통째로 다르고, `ELU(alpha)` 는 음수 쪽 크기를 정하고, `Softmax()` 의 기본 축은
// **`-1` 이 아니다.**

export class SiLU extends Module {
  override forward(x: Tensor): Tensor {
    return x.unary("silu");
  }
}

export class Sigmoid extends Module {
  override forward(x: Tensor): Tensor {
    return x.unary("sigmoid");
  }
}

export class Tanh extends Module {
  override forward(x: Tensor): Tensor {
    return x.unary("tanh");
  }
}

/**
 * Returns what it was given. Used as a placeholder in a `Sequential`.
 */
export class Identity extends Module {
  override forward(x: Tensor): Tensor {
    return x;
  }
}

export class LeakyReLU extends Module {
  constructor(private readonly negativeSlope = 0.01) {
    super();
  }

  override forward(x: Tensor): Tensor {
    return x.leakyRelu(this.negativeSlope);
  }
}

/**
 * The negative side lies down exponentially. **Without varying α it cannot
 * be told apart from the argument-free entry in the table.**
 */
export class ELU extends Module {
  constructor(private readonly alpha = 1.0) {
    super();
  }

  override forward(x: Tensor): Tensor {
    return x.elu(this.alpha);
  }
}

/**
 * **`approximate` is not accepted for show** — `"tanh"` is a different
 * formula and a different value.
 *
 * The maximum difference is around 1e-4, which is near this project's
 * tolerance, and so it is a place where "near enough, keep one" almost got
 * through. The golden cases ask about the two separately.
 */
export class GELU extends Module {
  constructor(private readonly approximate: "none" | "tanh" = "none") {
    super();
  }

  override forward(x: Tensor): Tensor {
    return this.approximate === "tanh" ? x.geluTanh() : x.unary("gelu");
  }
}

/**
 * `dim` 을 안 주었을 때 torch 가 고르는 축. **`-1` 이 아니다.**
 *
 * 랭크 1 → 0, 2 → 1, 3 → **0**, 4 → 1 (실측). torch 는 그 자리에서 경고까지 낸다.
 *
 * **랭크 2 로만 물으면 이 규칙이 안 보인다** — 거기서는 `dim=1` 과 `dim=-1` 이 같은
 * 축이라 `-1` 을 기본값으로 두어도 답이 같다. 코어가 실제로 그렇게 두고 있었고,
 * 랭크 3 에서 조용히 다른 축을 접고 있었다.
 */
function defaultSoftmaxDim(ndim: number): number {
  return ndim === 0 || ndim === 1 || ndim === 3 ? 0 : 1;
}

export class Softmax extends Module {
  constructor(private readonly dim: number | null = null) {
    super();
  }

  override forward(x: Tensor): Tensor {
    return x.softmax(this.dim ?? defaultSoftmaxDim(x.shape.length));
  }
}

export class LogSoftmax extends Module {
  constructor(private readonly dim: number | null = null) {
    super();
  }

  override forward(x: Tensor): Tensor {
    return x.logSoftmax(this.dim ?? defaultSoftmaxDim(x.shape.length));
  }
}

/**
 * **Learns** the slope on the negative side. The only one in this family
 * with a parameter.
 *
 * The name `weight` becomes a `stateDict` key, so it has to match torch.
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
 * Normalises with the channels bundled into groups. **Used instead of
 * BatchNorm when the batch is small.**
 *
 * BatchNorm uses batch statistics, so with a batch of 1–2 those statistics
 * cannot be trusted. This one bundles within a single sample, so it is
 * independent of batch size.
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
 * Per sample and per channel. **The default is no parameters** — as in
 * torch.
 *
 * It is the opposite of `BatchNorm`, which makes it a confusing place, and
 * flipping the default changes every `stateDict` key.
 */
export class InstanceNormND extends Module {
  constructor(private readonly eps = 1e-5) {
    super();
  }

  override forward(x: Tensor): Tensor {
    return x.instanceNorm(this.eps);
  }
}

/**
 * **It does not subtract the mean.** That is the only difference from
 * `LayerNorm`.
 */
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
 * Transposed convolution. **The weight is `(in, out, …)`** — flipped
 * relative to `ConvND`.
 *
 * With a square kernel the shape still matches when flipped, so only the
 * values diverge. The `stateDict` keys are the same, `weight` and `bias`,
 * so putting them in by shape alone is quietly wrong.
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

/* ── 차원을 고정한 이름들 ───────────────────────────────────────────────
 *
 * `ConvTransposeND` 는 `spatial` 을 **인자로** 받는다. torch 는 그 수를 이름에 넣으
 * 므로, 여기서는 그 자리 하나를 고정하는 것이 곧 torch 의 이름이다.
 *
 * 값은 이미 증명돼 있다 — 골든 `norm::nn.ConvTranspose{1,2,3}d` 세 건이 `ND` 판으로
 * 돌고 있었다. 없던 것은 계산이 아니라 이름이다.
 */

/** `torch.nn.ConvTranspose1d`. */
export class ConvTranspose1d extends ConvTransposeND {
  constructor(inC: number, outC: number, kernel: number,
              stride = 1, padding = 0, bias = true) {
    super(inC, outC, kernel, 1, stride, padding, bias);
  }
}

/** `torch.nn.ConvTranspose2d`. */
export class ConvTranspose2d extends ConvTransposeND {
  constructor(inC: number, outC: number, kernel: number,
              stride = 1, padding = 0, bias = true) {
    super(inC, outC, kernel, 2, stride, padding, bias);
  }
}

/** `torch.nn.ConvTranspose3d`. */
export class ConvTranspose3d extends ConvTransposeND {
  constructor(inC: number, outC: number, kernel: number,
              stride = 1, padding = 0, bias = true) {
    super(inC, outC, kernel, 3, stride, padding, bias);
  }
}

/**
 * Drops slots only while training. **In eval mode it is the identity.**
 *
 * `training` is held by `Module`, and `eval()` reaches down through
 * containers to turn it off — break that propagation and training is fine
 * while only inference is wrong.
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
 * The core of attention. **The computation `MultiheadAttention` was doing
 * inside, given a name.**
 *
 * The layer existed and this function did not. Code that writes attention
 * by hand rather than using the layer calls this name, and that is the
 * basic form of modern transformer code.
 *
 * **A mask is added, not multiplied.** A large negative number is added so
 * that softmax produces zero — it is not a multiplication by zero, because
 * multiplying after softmax has already normalised leaves the remaining
 * slots not summing back to 1.
 *
 * With a batch axis it runs per sample. That is because `mm` is
 * two-dimensional; when a batched matrix multiply exists, this is the place
 * to change.
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

/* ── 차원이 다른 짝들 ────────────────────────────────────────────────────
 *
 * **계산은 이미 있었고 층 이름만 없었다.** `x.maxPool1d(2)` 는 되는데
 * `new nn.MaxPool1d(2)` 는 안 되는 상태였다 — 교재를 그대로 따라 치면 거기서 멈춘다.
 *
 * 골든이 `ndim::nn.MaxPool1d` 로 값을 이미 붙잡고 있는데, borch.ts 쪽 케이스가 층이
 * 아니라 **텐서 메서드로** 답하고 있었다. 그래서 값은 증명돼 있고 이름은 없었다.
 * 케이스도 층을 지나가게 같이 고친다 — 안 그러면 이 이름들을 아무도 안 재게 된다.
 */

/** `torch.nn.MaxPool1d`. */
export class MaxPool1d extends Module {
  constructor(private readonly kernel = 2, private readonly stride?: number) {
    super();
  }

  override forward(x: Tensor): Tensor {
    return x.maxPool1d(this.kernel, this.stride);
  }
}

/** `torch.nn.MaxPool3d`. */
export class MaxPool3d extends Module {
  constructor(private readonly kernel = 2, private readonly stride?: number) {
    super();
  }

  override forward(x: Tensor): Tensor {
    return x.maxPool3d(this.kernel, this.stride);
  }
}

/**
 * Flattens, keeping only the batch axis.
 */
export class Flatten extends Module {
  override forward(x: Tensor): Tensor {
    const batch = x.shape[0] ?? 1;
    return x.reshape([batch, x.size / batch]);
  }
}

/**
 * The opposite of `Flatten`. Spreads one axis into several.
 */
export class Unflatten extends Module {
  constructor(private readonly dim: number,
              private readonly sizes: readonly number[]) {
    super();
  }

  override forward(x: Tensor): Tensor {
    return x.unflatten(this.dim, this.sizes);
  }
}

/**
 * Max pooling whose window positions are shaken by a sample.
 *
 * **The sample is drawn on the host.** The window positions have to be
 * baked into the shader, so the values must be on the CPU, and reading GPU
 * randomness back is asynchronous — `forward` has to be synchronous, so it
 * uses the stream in `random.ts`. `manualSeed` catches that stream too.
 *
 * Pass `randomSamples` and it uses those. It is where torch's
 * `_random_samples` goes, and it is needed to freeze values for comparison
 * — without it the three implementations' generators differ and the values
 * do not match.
 *
 * **`outputSize` and `outputRatio` are one or the other.** Giving both, or
 * neither, stops — as in torch.
 */
export class FractionalMaxPoolND extends Module {
  constructor(private readonly spatial: number,
              private readonly kernel: number,
              private readonly outputSize: number | readonly number[] | null = null,
              private readonly outputRatio: number | readonly number[] | null = null,
              returnIndices = false,
              private readonly randomSamples: readonly (readonly number[])[] | null
                = null) {
    super();
    if ((outputSize === null) === (outputRatio === null)) {
      throw new ValueError(
        "FractionalMaxPool takes either outputSize or outputRatio, not both.");
    }
    // **받아만 놓고 버리지 않는다.** torch 는 이 깃발이 참이면 `forward` 가 쌍을
    // 내는데, 여기 `forward` 는 텐서를 내기로 되어 있다(`Module` 의 약속). 조용히
    // 값만 주는 대신 멈추고 `pool()` 로 보낸다.
    if (returnIndices) {
      throw new RuntimeError(
        "returnIndices comes back from pool(x) — forward returns a single tensor.");
    }
  }

  private sizesFor(shape: readonly number[]): number[] {
    if (this.outputSize !== null) {
      return typeof this.outputSize === "number"
        ? new Array<number>(this.spatial).fill(this.outputSize)
        : [...this.outputSize];
    }
    const ratio = this.outputRatio as number | readonly number[];
    const each = typeof ratio === "number"
      ? new Array<number>(this.spatial).fill(ratio) : [...ratio];
    return each.map((r, k) => Math.trunc((shape[2 + k] ?? 1) * r));
  }

  pool(x: Tensor): { values: Tensor; indices: Tensor } {
    const planes = (x.shape[0] ?? 1) * (x.shape[1] ?? 1);
    const samples: readonly (readonly number[])[] = this.randomSamples
      ?? Array.from({ length: planes }, () =>
        Array.from({ length: this.spatial }, () => uniform01()));
    return x.fractionalMaxPool(this.kernel, this.sizesFor(x.shape), samples);
  }

  override forward(x: Tensor): Tensor {
    return this.pool(x).values;
  }
}

export class FractionalMaxPool2d extends FractionalMaxPoolND {
  constructor(kernel: number,
              outputSize: number | readonly number[] | null = null,
              outputRatio: number | readonly number[] | null = null,
              returnIndices = false,
              randomSamples: readonly (readonly number[])[] | null = null) {
    super(2, kernel, outputSize, outputRatio, returnIndices, randomSamples);
  }
}

export class FractionalMaxPool3d extends FractionalMaxPoolND {
  constructor(kernel: number,
              outputSize: number | readonly number[] | null = null,
              outputRatio: number | readonly number[] | null = null,
              returnIndices = false,
              randomSamples: readonly (readonly number[])[] | null = null) {
    super(3, kernel, outputSize, outputRatio, returnIndices, randomSamples);
  }
}

export class AvgPool2d extends Module {
  constructor(private readonly kernel: number,
              private readonly stride?: number) {
    super();
  }

  override forward(x: Tensor): Tensor {
    return x.avgPool2d(this.kernel, this.stride);
  }
}

/**
 * `torch.nn.AvgPool1d`. **It goes through `poolND`** — `avgPool2d` is a
 * two-dimensional kernel, and this one has a different number of remaining
 * axes, so it cannot use that path.
 */
export class AvgPool1d extends Module {
  constructor(private readonly kernel: number,
              private readonly stride?: number) {
    super();
  }

  override forward(x: Tensor): Tensor {
    return x.poolND("avg", this.kernel, this.stride);
  }
}

/** `torch.nn.AvgPool3d`. */
export class AvgPool3d extends Module {
  constructor(private readonly kernel: number,
              private readonly stride?: number) {
    super();
  }

  override forward(x: Tensor): Tensor {
    return x.poolND("avg", this.kernel, this.stride);
  }
}

/**
 * `torch.nn.AdaptiveAvgPool1d`. **It takes the output size**, not the kernel.
 *
 * It fixes the size and works the kernel backwards from it, so the argument
 * means the opposite of what it means above. Confusing the two gives a
 * different size quietly, which is why the name is separated by `Adaptive`.
 */
export class AdaptiveAvgPool1d extends Module {
  constructor(private readonly outSize: number | readonly number[]) {
    super();
  }

  override forward(x: Tensor): Tensor {
    return x.adaptivePool("avg", this.outSize);
  }
}

/** `torch.nn.AdaptiveAvgPool3d`. It takes the output size. */
export class AdaptiveAvgPool3d extends Module {
  constructor(private readonly outSize: number | readonly number[]) {
    super();
  }

  override forward(x: Tensor): Tensor {
    return x.adaptivePool("avg", this.outSize);
  }
}

/**
 * p-norm pooling. **The first argument is `normType`**, not the kernel
 * (that is torch's shape).
 */
export class LPPool1d extends Module {
  constructor(private readonly normType: number,
              private readonly kernel: number,
              private readonly stride?: number) {
    super();
  }

  override forward(x: Tensor): Tensor {
    return x.lpPool(this.normType, this.kernel, this.stride);
  }
}

export class LPPool2d extends LPPool1d {}
export class LPPool3d extends LPPool1d {}

/**
 * Trailing axes to mean 0, variance 1. **`normalizedShape` decides how many
 * axes are folded.**
 *
 * Measured only with a single axis (`new LayerNorm(4)`), the answer matches
 * "fold the last axis" and the rule stays invisible. Given `[3, 4]` it
 * folds the last two axes **as one block.**
 *
 * The default is with parameters — the opposite of `InstanceNorm`, which
 * makes it a confusing place, and flipping the default changes every
 * `stateDict` key.
 */
export class LayerNorm extends Module {
  readonly weight: Tensor | null = null;
  readonly bias: Tensor | null = null;
  private readonly dims: number;
  private readonly shape: number[];

  constructor(normalizedShape: number | readonly number[],
              private readonly eps = 1e-5,
              elementwiseAffine = true, useBias = true) {
    super();
    const shape = typeof normalizedShape === "number"
      ? [normalizedShape] : [...normalizedShape];
    this.shape = shape;
    this.dims = shape.length;
    if (elementwiseAffine) {
      this.weight = Tensor.owned(shape, 1);
      this.claim(this.weight);
      if (useBias) {
        this.bias = Tensor.owned(shape, 0);
        this.claim(this.bias);
      }
    }
  }

  override ownParameters(): Record<string, Tensor> {
    const out: Record<string, Tensor> = {};
    if (this.weight) out["weight"] = this.weight;
    if (this.bias) out["bias"] = this.bias;
    return out;
  }

  override forward(x: Tensor): Tensor {
    // **모양이 안 맞으면 멈춘다.** 관대하면 잘못된 축을 조용히 접는다.
    const tail = x.shape.slice(x.shape.length - this.dims);
    if (tail.length !== this.shape.length
      || tail.some((d, i) => d !== this.shape[i])) {
      throw new RuntimeError(
        `Given normalized_shape=[${this.shape}], expected input with shape `
        + `[*, ${this.shape}]`);
    }
    const normed = x.layerNormOver(this.dims, this.eps);
    if (!this.weight) return normed;
    const out = normed.mul(this.weight);
    return this.bias ? out.add(this.bias) : out;
  }
}

/**
 * Enlargement. **The first position is `size`**, as in torch.
 *
 * Putting the scale factor first makes `new Upsample(2)` ambiguous between
 * doubling and "output of 2", and the shape is plausible enough that only
 * values catch it.
 */
export class Upsample extends Module {
  constructor(private readonly size: number | readonly number[] | null = null,
              private readonly scaleFactor: number | null = null,
              private readonly mode: "nearest" | "bilinear" = "nearest",
              private readonly alignCorners: boolean | null = null) {
    super();
  }

  override forward(x: Tensor): Tensor {
    if (this.size === null && this.scaleFactor === null) {
      throw new RuntimeError("either size or scale_factor should be defined");
    }
    return x.interpolate(this.size, this.scaleFactor, this.mode,
                         this.alignCorners ?? false);
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

/**
 * **Alone in returning two** — `(h, c)`. Forcing the three into one shape
 * loses the memory cell.
 */
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

/**
 * Mixes two inputs **at once.** The weight has three axes, `(out, in1,
 * in2)`.
 */
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

/**
 * Softmax **along the channels** of `(N, C, H, W)`. The same as
 * `softmax(dim=1)`.
 */
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

/**
 * One row per bag. Selecting from the table **and combining** is one layer.
 */
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
   * Cuts a one-dimensional index run into bags with `offsets`.
   *
   * **Bags of differing lengths are the reason this name exists.** A
   * two-dimensional input only allows bags of equal length, and the actual
   * uses (shopping carts, sentences) have differing lengths.
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
    /**
     * The text to print before it solidifies. `describe()` uses it.
     */
    readonly label: string,
    /** 알아낸 크기로 진짜 층을 만든다. */
    private readonly build: (inferred: number) => Module,
    /**
     * What to read off the input. Linear reads the last axis; the rest read
     * channels.
     */
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
 * Softmax for a very large vocabulary. **It makes frequent tokens cheap.**
 *
 * Tokens are bundled by frequency; the leading bundle is produced straight
 * from the head, and the trailing bundles as **the probability the head
 * picked that bundle × the probability within it**. The further back, the
 * narrower the intermediate dimension gets, divided by `divValue` — less
 * room spent on rare tokens.
 *
 * **The defaults are `divValue=4` and `headBias=false`** (measured:
 * `tests/probe_asm.py`). The intermediate dimension is `inFeatures /
 * divValue**(i+1)` floored and **can reach 0** — torch makes an empty layer
 * there and carries on, so this does not block it either.
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
   * The log probability of every token, `(N, nClasses)`.
   *
   * To the probability within a bundle it **adds the log probability that
   * the head picked that bundle** — multiplication is addition in logs,
   * which is what leaves the whole row summing to 1.
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

  /**
   * `{ output, loss }` — the log probability at the target position and the
   * negative of its mean.
   */
  run(x: Tensor, target: Tensor): { output: Tensor; loss: Tensor } {
    const lp = this.logProb(x);
    const rows = target.reshape([target.size, 1]);
    const output = lp.gather(1, rows).reshape([target.size]);
    return { output, loss: output.mean().neg() };
  }

  /**
   * **This layer takes the target too.** Trying to pass through stops here
   * — `run(x, target)` is the place.
   */
  override forward(): Tensor {
    throw new Error(
      "AdaptiveLogSoftmaxWithLoss also takes the target — use run(x, target)",
    );
  }

  predict(x: Tensor): Tensor {
    return this.logProb(x).argmax(1);
  }
}

/**
 * A loss that joins audio to text **without aligning positions.**
 *
 * `logProbs` is `(T, N, C)` — **time first.** Each sample has its own
 * length and that is the point of this loss, so it receives two sets of
 * lengths alongside.
 *
 * ## It adds one term whose value is zero and whose gradient is not
 *
 * The gradient torch flows into `logProbs` is not the true derivative —
 * finite differences give `-γ` while torch produces `exp(logProbs) - γ`
 * (measured: `tests/probe_ctc3.py`). At the place it is used there is a
 * `logSoftmax` in front, which makes the two the same answer (it is a fixed
 * point of that backward), but with `logProbs` as a leaf directly the
 * numbers diverge. Matched — with **why it is matched** written down.
 *
 * @param reduction `"mean"` divides each sample **by its own target
 *   length** before averaging — not a plain mean.
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

  /**
   * torch's `extra_repr` produces nothing here — hence empty.
   */
  describe(): string { return "CTCLoss()"; }
}

// **흔한 손실 층 넷이 없었다.** `HuberLoss`·`KLDivLoss`·`TripletMarginLoss` 같은
// 드문 것은 있는데 `MSELoss`·`L1Loss`·`SmoothL1Loss`·`BCEWithLogitsLoss` 가 없었다 —
// `reduction` 때와 **같은 방향의 뒤집힘**이 층 이름에서 한 번 더 나온 자리다.
// 나중에 쓴 것이 torch 를 따랐고 처음 자리가 안 채워졌다.
//
// 파이썬 결속은 텐서 메서드 위에 스스로 만들어 두어 멀쩡했다. 그래서 **TypeScript
// 로 직접 쓰는 사람에게만** 없었고, 골든도 결속을 지나므로 안 물었다.

export class MSELoss {
  constructor(readonly reduction: Reduction = "mean") {}

  forward(x: Tensor, target: Tensor): Tensor {
    return x.mseLoss(target, this.reduction);
  }

  call(x: Tensor, target: Tensor): Tensor {
    return this.forward(x, target);
  }

  describe(): string { return "MSELoss()"; }
}

export class L1Loss {
  constructor(readonly reduction: Reduction = "mean") {}

  forward(x: Tensor, target: Tensor): Tensor {
    return x.l1Loss(target, this.reduction);
  }

  call(x: Tensor, target: Tensor): Tensor {
    return this.forward(x, target);
  }

  describe(): string { return "L1Loss()"; }
}

export class SmoothL1Loss {
  constructor(readonly reduction: Reduction = "mean", readonly beta = 1.0) {}

  forward(x: Tensor, target: Tensor): Tensor {
    return x.smoothL1Loss(target, this.beta, this.reduction);
  }

  call(x: Tensor, target: Tensor): Tensor {
    return this.forward(x, target);
  }

  describe(): string { return "SmoothL1Loss()"; }
}

export class BCEWithLogitsLoss {
  constructor(readonly reduction: Reduction = "mean") {}

  forward(x: Tensor, target: Tensor): Tensor {
    return x.bceWithLogits(target, this.reduction);
  }

  call(x: Tensor, target: Tensor): Tensor {
    return this.forward(x, target);
  }

  describe(): string { return "BCEWithLogitsLoss()"; }
}

export class NLLLoss {
  constructor(readonly reduction: Reduction = "mean") {}

  forward(x: Tensor, target: Tensor): Tensor {
    return x.nllLoss(target, this.reduction);
  }

  call(x: Tensor, target: Tensor): Tensor {
    return this.forward(x, target);
  }

  describe(): string { return "NLLLoss()"; }
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

/**
 * A triplet loss taking a distance function. The default is the pairwise
 * distance, so it gives the same answer as the one above.
 */
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

/**
 * The distance between two paired rows. **`eps` is added to the
 * difference** — written down on the tensor side.
 */
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
 * The root of fifteen padding layers.
 *
 * **Three (1-, 2-, 3-dimensional) × five (reflect, replicate, zero,
 * constant, circular) come out of one machine.** All that differs is the
 * mode name and the number of pairs, so writing fifteen copies by hand
 * scatters two genuinely different things across fifteen places.
 */
export class PadNd extends Module {
  readonly padding: number[];

  constructor(
    /**
     * The name to print. **It does not lean on `constructor.name`** — the
     * moment a bundler shortens names, different text comes out quietly,
     * and the golden cases only ever see our build.
     */
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
   * Exactly what Python prints.
   *
   * **Only `ConstantPad` prints with the name attached** — the rest print
   * only the pairs. Real torch does that and the golden cases froze the
   * text, so the difference is part of the answer. The value is in Python's
   * form too, so an integer still carries a decimal point (`value=7.0`).
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
 * The sampling grid `theta` draws. `(N, 2, 3)` → `(N, H, W, 2)`.
 *
 * The last axis is in **`(x, y)` order** — flipped relative to the shape's
 * `(H, W)`. On a square it gives the same answer written either way, so it
 * only surfaces when asked with a rectangle.
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
 * Lifts values from where the grid points. The counterpart of `affineGrid`.
 *
 * **The positions are constants and the weights are tensors.** A floored
 * integer has no derivative and its remainder becomes the weight, so
 * keeping only the weights in the graph lets gradient flow to both the
 * input and the grid.
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
 * The function form of `BatchNormND`. **The layer calls this** — one copy
 * of the formula.
 *
 * **In training it modifies the running statistics in place.** torch does,
 * and the tensor you hand in comes back updated. Returning a new one leaves
 * the caller's buffer unmoved, so training runs and only eval-mode values
 * are wrong.
 *
 * **It uses variance two ways.** Normalisation uses the biased estimate;
 * the running-statistics update uses the unbiased one. Merge them and it
 * diverges in eval mode only.
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
      throw new Error("batchNorm: eval mode needs running statistics");
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
 * Selects rows from a table by index. Where `F.embedding(indices, table)`
 * goes.
 *
 * **It is the definition itself** — the same thing `indexSelect` does, and
 * that side already knows the gradient. An index appearing several times
 * accumulates into that row several times. This is **naming something that
 * exists**, not imitating something that does not, so there is no place for
 * the values to diverge.
 *
 * And yet this name was missing for a long time. The binding held the same
 * three lines in Python and the golden cases went through that side, so it
 * was a name absent only for those writing TypeScript — the place
 * `tests/test_binding_fills_in.py` points at.
 *
 * **There is no `nn.Embedding` layer yet.** None of the three has it and
 * the golden cases do not ask. It would be a place that wraps this function
 * and holds the weights rather than a new computation, so writing down that
 * it is absent is better than putting the name there alone.
 */
export function embedding(idx: Tensor, weight: Tensor): Tensor {
  const dim = weight.shape[1] ?? 1;
  const picked = weight.indexSelect(0, idx.reshape([idx.size]));
  return picked.reshape([...idx.shape, dim]);
}

/**
 * One row per bag. Selecting from the table **and combining** is one
 * function.
 *
 * Given `offsets` it cuts a one-dimensional index run into bags — the place
 * where bag lengths differ. `perSampleWeights` is used by torch only when
 * `mode='sum'`.
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
 * Picks one at random **while letting derivatives through.**
 *
 * Drawing one category is not differentiable, so Gumbel noise is added and
 * `softmax` smooths it. The smaller `tau`, the more it concentrates on one
 * side.
 *
 * With `hard=true` the answer is 0/1 but **the gradient is the smooth
 * one's** — the common `hard - soft.detach() + soft` trick, where the value
 * is hard and the derivative is soft. Keeping those two apart is what this
 * function means.
 *
 * @param noise Gumbel noise already drawn. Undrawn, it is drawn here — the
 *   golden cases can only ask about properties rather than values, but a
 *   caller sometimes wants to reproduce with fixed noise.
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
  /**
   * 학습 모드로 지나간 횟수. **GPU 에 안 둔다 — 그냥 수다.**
   *
   * torch 는 이것을 0 차원 텐서 버퍼로 갖고 `state_dict` 에 넣는다. 우리는 이 값을
   * 계산에 안 쓰므로(`momentum` 이 늘 수다) 텐서로 둘 이유가 없고, 텐서로 두면
   * **BN 층마다 스텝당 dispatch 가 하나씩 는다** — ResNet-18 이면 스무 개다. 쓰지도
   * 않는 값에 그 값을 낼 수 없다.
   *
   * 그런데도 세는 이유는 `state_dict` 다. 이 열쇠가 없으면 torch·`borch` 가 낸
   * 체크포인트를 **기본(strict)으로 못 읽는다.** 실제로 그랬다 — 골든이
   * `container::BatchNorm/state_dict 열쇠` 로 그것을 잡았고, 그 전까지 열쇠를 묻는
   * 케이스가 `Linear` 뿐이라 버퍼 갈래를 한 번도 안 물었다.
   */
  private numBatchesTracked = 0;
  /**
   * 불러온 체크포인트가 갖고 있던 횟수. **텐서로 들고 있는다.**
   *
   * `item()` 이 비동기라 동기인 `loadStateDict` 안에서 수로 못 읽는다. 그렇다고
   * 버리면 불러온 것을 **다시 저장할 때 0 이 나온다** — 아무도 안 읽는 값이 조용히
   * 틀리는 것이라 제일 늦게 발견된다. 그래서 텐서인 채로 두고, 그 뒤로 지나간
   * 횟수만 수로 세어 내보낼 때 한 번 더한다.
   */
  private trackedBase: Tensor | null = null;

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

  /**
   * **The running statistics go out too.** Left out, it is quietly wrong in
   * eval mode only.
   *
   * A field named `runningMean` has to leave under the key `running_mean`,
   * which rules out `registerBuffer` (there the field name is the key).
   * Instead **the list lives here alone** and `stateDict` derives from it
   * the same way the base does — the three falling out of step has already
   * happened twice in this layer.
   */
  override namedBuffers(persistentOnly = false): Record<string, Tensor> {
    void persistentOnly;                      // 이 층에는 안 싣는 버퍼가 없다
    return {
      running_mean: this.runningMean,
      running_var: this.runningVar,
      // 셀 때는 수였지만 내보낼 때는 텐서여야 한다 — 저장 형식이 텐서 사전이다.
      // 부를 때마다 새로 만든다. `stateDict` 는 드물게 도는 길이라 값이 안 든다.
      num_batches_tracked: this.trackedBase === null
        ? Tensor.owned([], this.numBatchesTracked)
        : this.trackedBase.add(Tensor.owned([], this.numBatchesTracked)),
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
      else if (name === "num_batches_tracked") {
        // 준 텐서를 그대로 붙잡지 않는다 — 구역이 닫히면 그 버퍼는 재활용된다.
        const base = Tensor.owned([], 0);
        noGrad(() => base.copyFrom(src));
        keepAlive(base);
        this.trackedBase = base;
        this.numBatchesTracked = 0;
      }
      else rest[name] = src;
    }
    super.loadStateDict(rest, strict);
  }

  override forward(x: Tensor): Tensor {
    if (this.training) this.numBatchesTracked += 1;
    // **계산은 `batchNorm` 이 한다.** 층과 함수가 각자 적으면 언젠가 갈리고,
    // 갈리는 자리가 이동 통계라 학습은 멀쩡하고 평가만 틀린다.
    return batchNorm(x, this.runningMean, this.runningVar, this.weight,
      this.bias, this.training, this.momentum, this.eps);
  }
}

/**
 * `torch.nn.BatchNorm2d`. **It inherits `BatchNormND` unchanged.**
 *
 * Rank is not consulted, so the number in the name does not change the
 * computation — everything but the channel axis is reduced, which makes
 * (N,C,H,W) and (N,C,D,H,W) the same code. The core (`borch/_nn.py`) writes
 * the same reason at the same place, and three implementations giving one
 * answer is what this table is worth.
 *
 * **torch diverges here** — it rejects a 3-D input to `BatchNorm2d`. Adding
 * that check to borch.ts alone would split it from the core, so if it goes in
 * it is a change to all three, not something to slip in while standing this
 * name up.
 */
export class BatchNorm2d extends BatchNormND {}

/** `torch.nn.BatchNorm3d`. The computation is the same, for the reason above. */
export class BatchNorm3d extends BatchNormND {}

/**
 * Recurrent networks — `RNN`, `LSTM` and `GRU` as one class differing only
 * in gate count.
 *
 * Input is `(length, batch, features)` — torch's default, not
 * `batch_first`.
 *
 * **The time axis is sequential and cannot be unrolled.** One step's output
 * is the next step's input, so there is nowhere to run in parallel, and a
 * kernel is called per step. It holds for short sequences; longer, and the
 * per-step call cost buries the computation.
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

  /**
   * Produces the full output and the final state together.
   */
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

/* ── 종류를 고정한 이름들 ───────────────────────────────────────────────
 *
 * `Recurrent` 는 종류를 **인자로** 받고 torch 는 그것을 이름으로 가른다. 그래서 이
 * 셋은 인자 하나를 고정한 것이고, 그것이 곧 torch 가 쓰는 이름이다.
 *
 * 값은 이미 증명돼 있다 — 골든 `seq::{RNN,LSTM,GRU}/{출력,마지막상태}` 여섯 건이
 * `Recurrent` 로 돌고 있었다. 없던 것은 계산이 아니라 이름이고, **순환망 교재는
 * `nn.LSTM(...)` 으로 시작하므로** 그 첫 줄에서 멈추고 있었다.
 *
 * **torch 의 인자 전부를 받지는 않는다.** 저쪽은 `numLayers`·`batchFirst`·양방향·
 * 층간 드롭아웃도 받는데 여기 밑동은 한 층·시간 우선뿐이다. 받아 놓고 안 쓰면
 * 거짓말이 되므로 **아예 안 받는다** — 코어가 `InstanceNorm` 의
 * `track_running_stats` 에서 같은 자리를 같은 방법으로 지킨다.
 */

/** `torch.nn.RNN` — one layer, time-first. */
export class RNN extends Recurrent {
  constructor(inputSize: number, hidden: number) {
    super(inputSize, hidden, "RNN");
  }
}

/**
 * `torch.nn.LSTM` — one layer, time-first. It carries two states, so `cell`
 * comes back alongside `hidden`.
 */
export class LSTM extends Recurrent {
  constructor(inputSize: number, hidden: number) {
    super(inputSize, hidden, "LSTM");
  }
}

/** `torch.nn.GRU` — one layer, time-first. */
export class GRU extends Recurrent {
  constructor(inputSize: number, hidden: number) {
    super(inputSize, hidden, "GRU");
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
 * The function form of attention. Weights come from outside.
 *
 * **Input is `(L, N, E)` — length first.** torch's function of the same
 * name is, and calling it with batch first quietly mixes the wrong axes.
 *
 * **The mask is an added float.** A boolean table is not accepted here —
 * `-inf` is added so that softmax produces zero, rather than multiplying by
 * zero, and multiplying after normalisation leaves the remaining slots not
 * summing back to 1. Turning booleans into floats happens where torch's
 * contract is imitated (the Python binding).
 *
 * @returns the output `(L, N, E)` and the weights. With `averageWeights`
 *   they are `(N, L, S)`, otherwise `(N, H, L, S)`, one per head.
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
          if (!t) throw new Error("attention: the projections are missing");
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

  /**
   * A mask that only lets it look backwards. Fills the upper triangle with
   * -inf.
   */
  static causalMask(len: number): Tensor {
    const data = new Float32Array(len * len);
    for (let i = 0; i < len; i++) {
      for (let j = i + 1; j < len; j++) data[i * len + j] = -Infinity;
    }
    return Tensor.from(data, [len, len]);
  }
}

/**
 * Straight from logits and target indices. `log_softmax` and `nll_loss`
 * joined.
 *
 * **There was no `reduction`** — the layer side had the same hole.
 * Textbooks use layers more than functions, so this is the more frequently
 * touched side.
 */
export class CrossEntropyLoss {
  constructor(readonly reduction: Reduction = "mean") {}

  forward(logits: Tensor, target: Tensor): Tensor {
    return logits.crossEntropy(target, this.reduction);
  }

  call(logits: Tensor, target: Tensor): Tensor {
    return this.forward(logits, target);
  }

  describe(): string { return "CrossEntropyLoss()"; }
}

// ── torch.nn.functional 자리 ────────────────────────────────────────────
//
// **`nn.functional` 로 낸다** — torch 의 경로가 `torch.nn.functional` 이므로
// `const F = nn.functional` 한 줄이면 `F.conv2d(x, w, b)` 가 그대로 돈다.
//
// 이 파일이 이미 갖고 있던 자유 함수 여덟도 같이 모은다. 지금은 `nn.batchNorm` 과
// `nn.Linear` 가 한 이름 공간에 섞여 있는데, torch 에서 앞은 `F.` 이고 뒤는 `nn.` 이다.
// **옛 이름은 그대로 둔다** — 옮기는 것이 목적이지 깨는 것이 아니다.
//
// 방향이 하나다: `nn` → `functional` → `tensor`. `functional` 이 이 파일을 도로
// 부르면 순환이 되므로, 여기 있는 여덟은 옮기지 않고 **여기서 묶기만** 한다.
import * as delegated from "./functional.js";

export const functional = {
  ...delegated,
  affineGrid,
  batchNorm,
  ctcLoss,
  embedding,
  embeddingBag,
  gridSample,
  gumbelSoftmax,
  multiHeadAttentionForward,
  scaledDotProductAttention,
};
