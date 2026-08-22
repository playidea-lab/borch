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
 * Weight initialisation.
 *
 * **Without it nothing learns.** Starting at 0, the neurons of one layer receive the
 * same gradient and move to the same value forever — run ResNet that way and the loss
 * sits at ln(10)=2.303 while the steps go by. A bench was actually published in that
 * state, and 798 golden cases did not catch it: every case plants the weights from
 * outside, so nobody looks at the initial values.
 *
 * The formula is torch's default — `kaiming_uniform_(a=√5)` folds the bound down to
 * `1/√fan_in`, and the bias is uniform over the same bound.
 *
 * **The generator is not torch's.** It cannot be, and it must not pretend to be. The
 * golden does not ask about initial values and always plants the weights, so there is
 * nothing here to diverge.
 */
// **The generator moved to `random.ts`.** Layer initialisation and `Tensor.randn`
// have to draw from one stream for a single seed to reset everything. Kept in here,
// `tensor.ts` cannot call it — calling would be a cycle.
//
// **Dropout is reset with it.** That side holds its own counter that rises on every
// call, so left alone, planting a seed still gives a different mask each time — layer
// initialisation and dropout are the first two places somebody expecting "same seed,
// same result" looks. The core carried a defect of the same family, and one case
// caught both.
//
// **The seed value has to be carried in.** This used to reset to 1 every time, and
// then changing the seed does not change the mask — somebody measuring variance over
// five seeds reads a number that only the weight initialisation moved as experimental
// variance. Mixing once with the golden-ratio constant keeps the two streams (the host
// xorshift and the GPU hash) from starting at the same number and moving together by
// coincidence.
onSeed((seed) => { Tensor.dropoutSeed = ((seed ^ 0x9e3779b9) >>> 0) || 1; });

export { manualSeed } from "./random.js";

/** A tensor filled uniformly over `[-bound, bound]`. */
function uniform(shape: readonly number[], bound: number): Tensor {
  const n = shape.reduce((a, b) => a * b, 1);
  return Tensor.from(uniformArray(n, bound), shape);
}

/** How many paths arrive. The product of the weight's axes after the first. */
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
    // **Overriding `children()` alone leaves that child unlearned.**
    //
    // Only `namedChildren()` collects parameters. When the two disagree the layer is
    // visible while its parameters are not picked up, and **with no exception and no
    // warning, learning alone** stops at that place — the loss still goes down,
    // because the rest compensates.
    //
    // The bench's ResNet-18 was in that state. The shortcut was held in a plain object
    // (`{conv, bn}`) and written into `children()` only, so six shortcut layers were
    // never learning while the epoch time was being measured. What caught it was not a
    // value check but **the dead-tensor guard** — a leaf the optimiser cannot see never
    // receives `zeroGrad()`, so the previous step's gradient stays, and that buffer had
    // already gone back to the pool.
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
    // **Buffers are accepted too.** When what goes out differs from what comes in, a
    // file cannot be read back by the thing that wrote it — `stateDict()` sends the
    // buffers out while this looked at `namedParameters()` alone, so strict mode raised
    // "unknown name". Saving and restoring have to look at **the same list**.
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
    // It has to survive the scope closing — it is not a parameter, so nobody holds it.
    keepAlive(value);
    (this as unknown as Record<string, Tensor>)[name] = value;
  }

  /** Name → is it saved. The field itself stays on the layer; this holds only the mark. */
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
    // **It walks `namedChildren`.** The mode has to reach a named layer even when it
    // does not override `children` — otherwise training is fine and inference alone is
    // wrong.
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
    // It goes through `call` internally too — a recommended path the library itself
    // does not take is a recommendation erased from the example everybody reads.
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
 * torch's ordering rule. **A plain object goes in with its keys sorted.**
 *
 * Unmatched, `namedParameters` diverges in order and that is exactly `stateDict`'s
 * order — the golden caught this very place (for `{w, b}`, torch gave `ws.b ws.w`).
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
    // The golden plants the weights from outside, so whatever is here is overwritten
    // — a non-zero value is left for the case where nobody plants anything.
    const bound = 1 / Math.sqrt(Math.max(1, inFeatures));
    this.weight = uniform([outFeatures, inFeatures], bound);
    this.bias = bias ? uniform([outFeatures], bound) : null;
    this.claim(this.weight);
    if (this.bias) this.claim(this.bias);
  }

  override ownParameters(): Record<string, Tensor> {
    // **No bias, no key.** Pretending otherwise makes `state_dict` disagree with theirs.
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
    private readonly dilation = 1,
    private readonly groups = 1,
    private readonly paddingMode: PadMode | "zeros" = "zeros",
  ) {
    super();
    if (inChannels % groups !== 0 || outChannels % groups !== 0) {
      throw new RuntimeError(
        `groups=${groups} divides neither the input channels (${inChannels}) nor `
        + `the filters (${outChannels})`);
    }
    const shape = [
      outChannels, inChannels / groups, ...new Array<number>(spatial).fill(kernel),
    ];
    // **The fan-in is divided by `groups`.** Each filter sees only
    // `inChannels / groups` channels, and initialised as though it saw all of them
    // a grouped convolution starts too small — torch divides here too.
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
    // **A non-zero padding mode is padded here and the convolution called with 0**,
    // which is what torch's layer does and why `F.conv2d` has no such argument.
    // Putting it in both would be a second place for one decision.
    if (this.paddingMode !== "zeros" && this.padding !== 0) {
      const spatial = x.shape.length - 2;
      const widths: number[] = [];
      for (let d = 0; d < spatial; d++) widths.push(this.padding, this.padding);
      return x.padND(widths, this.paddingMode).convND(
        this.weight, this.bias, this.stride, 0, this.dilation, this.groups);
    }
    return x.convND(this.weight, this.bias, this.stride, this.padding,
                    this.dilation, this.groups);
  }
}

/**
 * **`bias` sits eighth, where torch has it**, and `dilation` sixth.
 *
 * `new Conv2d(3, 16, 3, 1, 1, false)` used to turn the bias off and now sets
 * `dilation`. The core moved first and this followed — while the two were apart,
 * the same line meant different things on the two sides, and both returned a
 * feature map of the right shape.
 */
export class Conv1d extends ConvND {
  constructor(inC: number, outC: number, kernel: number, stride = 1, padding = 0,
              dilation = 1, groups = 1, bias = true,
              paddingMode: PadMode | "zeros" = "zeros") {
    super(inC, outC, kernel, 1, stride, padding, bias, dilation, groups, paddingMode);
  }
}

export class Conv2d extends ConvND {
  constructor(inC: number, outC: number, kernel: number, stride = 1, padding = 0,
              dilation = 1, groups = 1, bias = true,
              paddingMode: PadMode | "zeros" = "zeros") {
    super(inC, outC, kernel, 2, stride, padding, bias, dilation, groups, paddingMode);
  }
}

export class Conv3d extends ConvND {
  constructor(inC: number, outC: number, kernel: number, stride = 1, padding = 0,
              dilation = 1, groups = 1, bias = true,
              paddingMode: PadMode | "zeros" = "zeros") {
    super(inC, outC, kernel, 3, stride, padding, bias, dilation, groups, paddingMode);
  }
}

export class ReLU extends Module {
  override forward(x: Tensor): Tensor {
    return x.unary("relu");
  }
}

// ── Activation layers. Each wraps exactly one tensor method. ────────────────
//
// The one way this family fails is a wrapper **calling a different function**, and
// that is invisible to the eye and diverges only in the values — which is why the
// golden asks about the functional form and the layer form separately.

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

// ── The eight the binding was filling in by hand. **The name is a shell and the
//    arguments are real.** ────────────────────────────────────────────────────
//
// The Python binding (`borch_webgpu/_nn.py`) built these as factories over the tensor
// methods, so the golden was **structurally unable to see** that these eight were
// missing — every case goes through the binding. They were absent only for somebody
// writing `new nn.GELU()` in TypeScript.
//
// Porting them showed that three carry arguments (measured): `GELU(approximate)` is a
// wholly different formula, `ELU(alpha)` sets the size of the negative side, and
// `Softmax()`'s default axis is **not `-1`.**

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
 * The axis torch picks when `dim` is not given. **It is not `-1`.**
 *
 * Rank 1 → 0, 2 → 1, 3 → **0**, 4 → 1 (measured). torch even warns at that point.
 *
 * **Asking only at rank 2 hides this rule** — there `dim=1` and `dim=-1` are the same
 * axis, so a default of `-1` gives the same answer. The core actually had it that way,
 * and was quietly folding a different axis at rank 3.
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
    private readonly outputPadding = 0,
    private readonly groups = 1,
    private readonly dilation = 1,
  ) {
    super();
    if (inC % groups !== 0 || outC % groups !== 0) {
      throw new RuntimeError(
        `groups=${groups} divides neither the input channels (${inC}) nor the `
        + `filters (${outC})`);
    }
    const bound = 1 / Math.sqrt(Math.max(1, (outC / groups) * kernel ** spatial));
    this.weight = uniform(
      [inC, outC / groups, ...new Array<number>(spatial).fill(kernel)], bound);
    this.bias = bias ? uniform([outC], bound) : null;
    this.claim(...(this.bias ? [this.weight, this.bias] : [this.weight]));
  }

  override ownParameters(): Record<string, Tensor> {
    return this.bias
      ? { weight: this.weight, bias: this.bias }
      : { weight: this.weight };
  }

  override forward(x: Tensor): Tensor {
    return x.convTransposeND(this.weight, this.bias, this.stride, this.padding,
                             this.outputPadding, this.groups, this.dilation);
  }
}

/* ── The names with the dimension fixed ─────────────────────────────────
 *
 * `ConvTransposeND` takes `spatial` **as an argument**. torch puts that number in the
 * name, so fixing that one slot here *is* torch's name.
 *
 * The values are already proven — the three golden cases
 * `norm::nn.ConvTranspose{1,2,3}d` were running on the `ND` form. What was missing was
 * the name, not the computation.
 */

/**
 * `torch.nn.ConvTranspose1d`.
 *
 * **torch puts `dilation` after `bias` here and before it in `Conv1d`.** The two
 * are not one list in a different spelling: the eighth position is `bias` in one
 * and `dilation` in the other. Following torch means following that too — a tidier
 * order of our own would read as agreement and land a positional call elsewhere.
 */
export class ConvTranspose1d extends ConvTransposeND {
  constructor(inC: number, outC: number, kernel: number, stride = 1, padding = 0,
              outputPadding = 0, groups = 1, bias = true, dilation = 1) {
    super(inC, outC, kernel, 1, stride, padding, bias, outputPadding, groups,
          dilation);
  }
}

/** `torch.nn.ConvTranspose2d`. See `ConvTranspose1d` on the argument order. */
export class ConvTranspose2d extends ConvTransposeND {
  constructor(inC: number, outC: number, kernel: number, stride = 1, padding = 0,
              outputPadding = 0, groups = 1, bias = true, dilation = 1) {
    super(inC, outC, kernel, 2, stride, padding, bias, outputPadding, groups,
          dilation);
  }
}

/** `torch.nn.ConvTranspose3d`. See `ConvTranspose1d` on the argument order. */
export class ConvTranspose3d extends ConvTransposeND {
  constructor(inC: number, outC: number, kernel: number, stride = 1, padding = 0,
              outputPadding = 0, groups = 1, bias = true, dilation = 1) {
    super(inC, outC, kernel, 3, stride, padding, bias, outputPadding, groups,
          dilation);
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

  // The mask that blocks the upper triangle. Built from 0 and a large negative, and
  // **added.**
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
  /**
   * **`returnIndices` was missing until the layer that consumes them arrived.**
   * The name was here, the argument was not, and that is invisible from a name
   * count — `MaxUnpool` is what makes the positions worth asking for.
   */
  constructor(private readonly kernel = 2,
              private readonly stride?: number,
              readonly returnIndices = false) {
    super();
  }

  override forward(x: Tensor): Tensor {
    return x.maxPool2d(this.kernel, this.stride);
  }

  /** The values and the positions that produced them. `MaxUnpool2d` takes both. */
  pick(x: Tensor): { values: Tensor; indices: Tensor } {
    return x.maxPoolWithIndices(this.kernel, this.stride);
  }
}

/* ── The pairs that differ only in dimension ────────────────────────────
 *
 * **The computation was already there and only the layer name was missing.**
 * `x.maxPool1d(2)` worked while `new nn.MaxPool1d(2)` did not — follow a textbook
 * literally and you stop right there.
 *
 * The golden was already holding the values as `ndim::nn.MaxPool1d`, but the borch.ts
 * case was answering **through the tensor method** rather than the layer. So the values
 * were proven and the name was absent. The cases are fixed to go through the layer as
 * well — otherwise nobody ever measures these names.
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
 * Adaptive max pooling. **It takes `returnIndices`, which the average side
 * does not** — and the argument changes what comes back.
 *
 * The underlying `adaptiveMaxPoolWithIndices` always returns both, so the layer
 * has to choose. Accepting the argument and returning the pair regardless would
 * be an argument taken and dropped, and the caller would meet the difference as
 * a shape error somewhere else.
 *
 * `forward` is the plain half so this fits in a `Sequential`; ask `pick()` when
 * the positions are wanted. `MaxUnpool` is the layer that consumes them.
 */
export class AdaptiveMaxPool1d extends Module {
  constructor(protected readonly outSize: number | readonly number[],
              readonly returnIndices = false) {
    super();
  }

  override forward(x: Tensor): Tensor {
    return x.adaptiveMaxPoolWithIndices(this.outSize).values;
  }

  /** The values and the positions that produced them. */
  pick(x: Tensor): { values: Tensor; indices: Tensor } {
    return x.adaptiveMaxPoolWithIndices(this.outSize);
  }
}

/** `torch.nn.AdaptiveMaxPool2d`. */
export class AdaptiveMaxPool2d extends AdaptiveMaxPool1d {}

/** `torch.nn.AdaptiveMaxPool3d`. */
export class AdaptiveMaxPool3d extends AdaptiveMaxPool1d {}

/**
 * Puts values back where `MaxPool` chose. `torch.nn.MaxUnpool1d`.
 *
 * **It needs two inputs, so `forward` is not the way in** — call `place()`.
 * The positions have to travel beside the values; hiding them inside the layer
 * means using somebody else's the second time the layer is applied. torch has
 * the same shape for the same reason, and it is why this one cannot go into a
 * `Sequential`.
 */
export class MaxUnpool1d extends Module {
  constructor(protected readonly kernel: number,
              protected readonly stride?: number,
              protected readonly padding = 0) {
    super();
  }

  /**
   * @param outSize the shape to restore. Pooling is not injective — a window
   *   that ran off the edge leaves an ambiguity only the caller can settle.
   */
  place(x: Tensor, indices: Tensor, outSize?: readonly number[]): Tensor {
    return x.maxUnpool(indices, this.kernel, this.stride, this.padding, outSize);
  }

  /** **There is no one-argument form.** Reaching here means the positions were lost. */
  override forward(_x: Tensor): Tensor {
    throw new RuntimeError(
      "MaxUnpool needs the positions MaxPool chose — call place(x, indices), " +
      "not forward(x). It cannot sit in a Sequential for that reason.");
  }
}

/** `torch.nn.MaxUnpool2d`. */
export class MaxUnpool2d extends MaxUnpool1d {}

/** `torch.nn.MaxUnpool3d`. */
export class MaxUnpool3d extends MaxUnpool1d {}

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
    // **Accepted and not discarded.** With this flag true torch's `forward` returns a
    // pair, and `forward` here is committed to returning a tensor (`Module`'s promise).
    // Rather than quietly handing back the value alone, it stops and sends you to
    // `pool()`.
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
    // **A shape that does not match stops.** Being lenient folds the wrong axis in
    // silence.
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

// ── One step of the recurrence ─────────────────────────────────────────
//
// **The names differ from the layer's.** A layer attaches the layer number, as in
// `weight_ih_l0`, and a cell is `weight_ih` — a cell has no layers. Those names are the
// `stateDict` keys, so getting them wrong makes checkpoints disagree.
//
// The gate order is the one `Recurrent` uses — `r, z, n` for GRU, `i, f, g, o` for
// LSTM. Written twice, a day comes when they diverge, and on that day only the values
// are quietly wrong.

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

// ── The remaining layers ───────────────────────────────────────────────

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
    // **Printed the way Python prints it.** `k=1.0`, not `k=1` — JS drops the decimal
    // point on a float that looks like an integer, and the golden froze the characters.
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
 * The two old names.
 *
 * **`UpsamplingBilinear2d` is `alignCorners=true`** — different from
 * `Upsample(bilinear)`'s default. Treated as an alias on the strength of the name
 * alone, the edges come out misaligned.
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
    // `embeddingBag` does the computation — the layer and the function are not two
    // copies of it.
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

// ── Moving elements, and dropout by channel ────────────────────────────
//
// All eight layers call exactly one tensor method. What differs is the arguments passed
// and the characters printed.

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

/** The root of the drop-by-channel family. **It prints `inplace` too** — torch does. */
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

// ── Lazy layers ────────────────────────────────────────────────────────
//
// **The shape is learned at the first forward.** `new nn.LazyLinear(3)` takes no input
// size and decides from the first value that passes through — which removes the job of
// counting by hand how many channels come out of a convolution.
//
// **Once it sets, it becomes something else.** After the first forward torch changes
// the object's class outright. Here the prototype is swapped and the innards moved, so
// the same object is still the one you hold — handing back a new object would leave
// whoever already captured it holding the old one.

export class LazyModule extends Module {
  private built: Module | null = null;

  constructor(
    /**
     * The text to print before it solidifies. `describe()` uses it.
     */
    readonly label: string,
    /** Builds the real layer from the size that was learned. */
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
      // Move the innards and swap the prototype — after this the object is the real
      // layer.
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

/** What to read out of the input. */
const lastAxis = (x: Tensor) => x.shape[x.shape.length - 1] ?? 0;
const channels = (x: Tensor) => x.shape[1] ?? 0;

export class LazyLinear extends LazyModule {
  constructor(outFeatures: number) {
    super(`LazyLinear(in_features=0, out_features=${outFeatures}, bias=True)`,
      (inF) => new Linear(inF, outFeatures), lastAxis);
  }
}

// Twelve, each with a name. **Not written as a factory** — exporting an anonymous
// class makes TypeScript refuse on the grounds of `Module`'s private members. The
// padding layers met the same wall, and three lines each is cheaper than breaking the
// inheritance to get around it.
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

// ── Loss layers ────────────────────────────────────────────────────────
//
// **They all have one shape** — hold the arguments at construction, hand them to a
// tensor method at the call. That is all torch's loss layers do, so writing a `forward`
// per layer would be writing the same two lines thirteen times.
//
// **torch prints a loss layer with no arguments** — even `HuberLoss(delta=0.5)` comes
// out as `HuberLoss()` (measured). The characters are part of the answer, so this
// follows.

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
    // Each repeated character that touches its neighbour costs one more blank. Shorter
    // than that and there is no alignment at all, so the probability is 0 and the loss
    // is infinite — an actual condition, not a threshold.
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

// **Four common loss layers were missing.** The rare ones — `HuberLoss`, `KLDivLoss`,
// `TripletMarginLoss` — were present while `MSELoss`, `L1Loss`, `SmoothL1Loss` and
// `BCEWithLogitsLoss` were not: **the same inversion as `reduction`**, appearing once
// more in the layer names. What was written later followed torch, and the first places
// were never filled in.
//
// The Python binding built them over the tensor methods itself and was fine. So they
// were missing **only for somebody writing TypeScript directly**, and the golden goes
// through the binding, so it never asked.

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
  /**
   * **`reduction` first, because that is torch's order** — `HuberLoss(reduction,
   * delta)`, and it is the odd one out among the losses that take a margin-like
   * number, which all put theirs first.
   *
   * This took `(delta, reduction)` until a signature check found it. `new
   * HuberLoss('sum')` set `delta` to a string, which is the same shape as
   * `ReduceLROnPlateau` losing its `mode` and as `std` taking the correction before
   * the axis: a line transcribed from torch that compiles and answers.
   */
  constructor(readonly reduction: Reduction = "mean", readonly delta = 1.0) {
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
  /**
   * **`weight` first, because that is torch's order.** This took `reduction`
   * alone until the core grew a real signature, and then the two axes disagreed:
   * `new MultiLabelSoftMarginLoss('sum')` set the reduction here and the class
   * weights in torch, and neither side raised.
   */
  constructor(
    readonly weight?: Tensor,
    readonly reduction: Reduction = "mean",
  ) {}

  forward(x: Tensor, target: Tensor): Tensor {
    return x.multilabelSoftMarginLoss(target, this.weight, this.reduction);
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

// Fifteen, each with a name. **Not written as a factory** — exporting an anonymous
// class makes TypeScript refuse on the grounds of `Module`'s private members, and three
// lines each is cheaper than breaking the inheritance to open that up.
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
 * Normalises per channel. **Training mode and evaluation mode differ.**
 *
 * In training it normalises with this batch's statistics while updating the running
 * ones; in evaluation it uses the running statistics. The difference shows only on the
 * path that saves, restores and then switches to evaluation, and that is exactly where
 * the core's defect sat — the running statistics were left out of `state_dict`, so
 * training was fine and inference alone was wrong.
 */
/**
 * Batch normalisation with no regard for the number of dimensions. `BatchNorm1d`, `2d`
 * and `3d` are all this.
 *
 * The axes folded are the batch and all the spatial ones, leaving the channel — the
 * rule is the same however many spatial axes there are, so there is no reason to stand
 * up a class per dimension.
 */
// ── The spatial transformer ────────────────────────────────────────────
//
// `affineGrid` builds a grid saying "this cell of the output looks at where in the
// input", and `gridSample` lifts the value from that place. The `theta` between them is
// what learns.
//
// **No new kernel.** The positions can be tensors too (a `floor`ed value is handed to
// `indexSelect` as its index), so it is built from operations that already exist. In
// exchange for several small kernels running, the gradients for both the input and the
// grid follow by themselves — writing the scattering backward by hand has threads
// writing to the same cell, and that answer can differ from run to run.

/** Sample positions over `[-1, 1]`. `alignCorners` decides between pinning the ends
 *  and taking cell centres. */
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
  // Homogeneous coordinates `(x, y, 1)` — the translation lands in the same product.
  const flat: number[] = [];
  for (let i = 0; i < h; i++) {
    for (let j = 0; j < w; j++) flat.push(xs[j] ?? 0, ys[i] ?? 0, 1);
  }
  // Each batch multiplies the same grid by its own `theta`. `bmm` works on 3-D pairs,
  // so the batch is expanded to match.
  const base = Tensor.from(flat, [h * w, 3]);
  const parts: Tensor[] = [];
  for (let b = 0; b < n; b++) {
    parts.push(base.mm(theta.select(0, b).permute([1, 0])));
  }
  return Tensor.stack(parts, 0).reshape([n, h, w, 2]);
}

/** `[-1, 1]` into the input's cell index. The reverse of `gridBase`. */
function gridDenorm(g: Tensor, n: number, alignCorners: boolean): Tensor {
  if (alignCorners) return g.add(Tensor.full([], 1)).mul(Tensor.full([], (n - 1) / 2));
  return g.add(Tensor.full([], 1)).mul(Tensor.full([], n)).sub(Tensor.full([], 1))
    .mul(Tensor.full([], 0.5));
}

/**
 * **Reflects** what falls outside. The interval reflected in depends on
 * `alignCorners` — `[0, n−1]` when true and `[−0.5, n−0.5]` when false (measured).
 * After reflecting it clamps once more.
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

  // The starting index per plane. With `(N, C)` flattened in advance, each corner is
  // just an addition.
  const planeStart: number[] = [];
  for (let n = 0; n < N; n++) {
    for (let c = 0; c < C; c++) planeStart.push((n * C + c) * H * W);
  }
  const starts = Tensor.from(planeStart, [N, C, 1]);
  const source = x.reshape([x.size]);

  /** Lifts one corner. **Outside the range gives 0, and the index is clamped before
   *  it is passed.** */
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
    // torch rounds. There are no weights, so no gradient reaches the grid.
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
  // **This goes through a fused kernel.** The assembled form cost more than twenty
  // dispatches for one layer, and most of the 1,636 in a single ResNet step came from
  // there (measured).
  const { out, mean, variance } = x.batchNormFused(w, b, eps);
  if (runningMean && runningVar) {
    // Both are updated by one kernel. The assembled form was eight dispatches per
    // layer, across twenty layers.
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
  // Gumbel noise — `-log(-log(u))`. Built from a single uniform draw.
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
   * How many times it has passed through in training mode. **Not on the GPU — it is
   * just a number.**
   *
   * torch keeps this as a 0-dimensional tensor buffer and puts it in `state_dict`. We
   * never use the value in a computation (`momentum` is always a number), so there is
   * no reason for it to be a tensor, and as a tensor it **adds one dispatch per step
   * per BN layer** — twenty of them in ResNet-18. A value nobody uses cannot be worth
   * that.
   *
   * It is counted anyway because of `state_dict`. Without this key a checkpoint written
   * by torch or by `borch` **cannot be read in the default (strict) mode.** That
   * actually happened — the golden caught it as
   * `container::BatchNorm/state_dict 열쇠`, and until then the only case asking about
   * keys was `Linear`, so the buffer branch had never been asked about.
   */
  private numBatchesTracked = 0;
  /**
   * The count the loaded checkpoint carried. **Held as a tensor.**
   *
   * `item()` is asynchronous, so it cannot be read as a number inside the synchronous
   * `loadStateDict`. Discarding it instead makes **saving what was loaded write 0** —
   * a value nobody reads going quietly wrong, which is the last kind to be found. So it
   * stays a tensor, and only the passes since then are counted as a number and added
   * once on the way out.
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
    // The running statistics take no gradient, and **still have to survive the scope
    // closing.**
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
    void persistentOnly;                      // no non-persistent buffer on this layer
    return {
      running_mean: this.runningMean,
      running_var: this.runningVar,
      // A number while counting and a tensor on the way out — the saved format is a
      // dictionary of tensors. Built fresh per call; `stateDict` is a rarely travelled
      // path, so it costs nothing.
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
        // The given tensor is not captured as it is — when the scope closes that
        // buffer is recycled.
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
    // **`batchNorm` does the computation.** With the layer and the function each
    // written out, a day comes when they diverge, and the place they diverge is the
    // running statistics — so training is fine and evaluation alone is wrong.
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
    // torch's recurrent nets take the bound from the hidden size — all four weights
    // use the same bound.
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
        // torch's gate order is i, f, g, o. Wrong order makes the values plausibly wrong.
        const g = gi.add(gh);
        const i = slice(g, 0, H).unary("sigmoid");
        const f = slice(g, 1, H).unary("sigmoid");
        const gg = slice(g, 2, H).unary("tanh");
        const o = slice(g, 3, H).unary("sigmoid");
        c = f.mul(c).add(i.mul(gg));
        h = o.mul(c.unary("tanh"));
      } else {
        // GRU adds only through r and z and **parts ways at the n gate** — the hidden
        // side's share is multiplied by r and then added. Adding first and multiplying
        // afterwards changes the value silently.
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

/* ── The names with the kind fixed ──────────────────────────────────────
 *
 * `Recurrent` takes the kind **as an argument** and torch splits it out into the name.
 * So these three fix one argument, and that is exactly the name torch uses.
 *
 * The values are already proven — the six golden cases
 * `seq::{RNN,LSTM,GRU}/{출력,마지막상태}` were running on `Recurrent`. What was missing
 * was the name rather than the computation, and **a recurrent-network textbook opens
 * with `nn.LSTM(...)`**, so it stopped people on the first line.
 *
 * **It does not take all of torch's arguments.** That side also takes `numLayers`,
 * `batchFirst`, bidirectional and inter-layer dropout, while the base here is one layer,
 * time-first. Accepting an argument and not using it is a lie, so **it is not accepted
 * at all** — the core holds the same line the same way at `InstanceNorm`'s
 * `track_running_stats`.
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

/** The gates are stacked vertically — the `k`th run of `H` rows. */
function slice(g: Tensor, k: number, H: number): Tensor {
  return g.narrow(1, k * H, H);
}

/**
 * Attention split across several heads.
 *
 * The input is `(batch, length, features)` (`batch_first=True`). The mask is a
 * **float** — 0 and -inf — and lumping it into "non-zero means masked" diverges here.
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

  /** Turns length-first into batch-first and projects. */
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
    // torch's attention starts the bias at 0 — this is not a place where symmetry
    // needs breaking.
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

// ── The place torch.nn.functional occupies ─────────────────────────────
//
// **It is exported as `nn.functional`** — torch's path is `torch.nn.functional`, so one
// line of `const F = nn.functional` makes `F.conv2d(x, w, b)` work unchanged.
//
// The eight free functions this file already had are gathered in too. As it stands
// `nn.batchNorm` and `nn.Linear` share one namespace, while in torch the first is `F.`
// and the second is `nn.`. **The old names stay** — the point is to move, not to break.
//
// There is one direction: `nn` → `functional` → `tensor`. `functional` calling back
// into this file would be a cycle, so the eight here are **only gathered**, not moved.
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
