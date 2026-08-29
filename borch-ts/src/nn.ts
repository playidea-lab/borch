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

import { NotImplementedError, RuntimeError, ValueError } from "./errors.js";
import { runningStats } from "./kernels.js";
import { onSeed, uniform as uniform01, uniformArray } from "./random.js";
import {
  device, keepAlive, noGrad, type PadMode, type Reduction, Tensor,
} from "./tensor.js";

/**
 * A number printed the way Python prints a float — `2.0`, not `2`.
 *
 * **JavaScript drops the decimal point on a float that happens to be integral**, and
 * `repr` cases freeze the characters, so `max_norm=2.0` came back as `max_norm=2` and a
 * golden row failed on nothing but punctuation.
 *
 * The same three lines had already been written twice in this file — inside
 * `LocalResponseNorm.describe` and inside the padding layers' — and a third copy was
 * about to go into `Embedding`. **Two copies is a coincidence and three is a rule that
 * belongs somewhere**, so both call sites now come here. Written once, the next
 * `describe` that needs it can be found by looking rather than by remembering.
 *
 * **Two layers never had it at all** — `RReLU`'s `lower` and `upper`, and
 * `LocalResponseNorm.alpha`. All three passed because their defaults (`1/8`, `1/3`,
 * `1e-4`) are never integral, so no golden case frozen at a default could see the
 * difference. A sweep that classified fields by their declared default missed them
 * for the same reason. `tests/test_describe_floats.py` asks torch's own signature
 * instead, which has no such accident to be fooled by.
 */
function pyFloat(v: number): string {
  return Number.isInteger(v) ? `${v}.0` : String(v);
}

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
 * What a `loadStateDict` found — the keys the layer asked for and did not get,
 * and the keys it was given and does not have.
 *
 * torch calls the same pair `missing_keys` and `unexpected_keys`. The names are
 * kept close because the person reading a failure here has usually just read one
 * there.
 */
export interface LoadReport {
  readonly missing: readonly string[];
  readonly unexpected: readonly string[];
}

/**
 * Stops a strict load that did not line up, saying **both sides** of the
 * difference.
 *
 * A list is cut at eight. A model with one wrong prefix reports every key it
 * has, and a screen of names buries the one line that says which prefix.
 */
function refuseIncompatible(
  missing: readonly string[],
  unexpected: readonly string[],
): void {
  if (missing.length === 0 && unexpected.length === 0) return;
  const show = (names: readonly string[]): string =>
    names.length > 8
      ? `${names.slice(0, 8).join(", ")} … and ${names.length - 8} more`
      : names.join(", ");
  const lines = ["load_state_dict: the checkpoint does not match the model"];
  if (missing.length > 0) {
    lines.push(`  the model asks for ${missing.length} key(s) the checkpoint`
      + ` does not have: ${show(missing)}`);
  }
  if (unexpected.length > 0) {
    lines.push(`  the checkpoint has ${unexpected.length} key(s) the model does`
      + ` not: ${show(unexpected)}`);
  }
  lines.push("  pass strict=false to load what lines up and weigh the rest yourself");
  throw new Error(lines.join("\n"));
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

  abstract forward(x: Tensor, ...rest: Tensor[]): Tensor;

  /**
   * So it can be called — the same position as torch's `model(x)`.
   *
   * **It forwards everything it was given.** It used to take one tensor, and a
   * decoder layer takes two: JavaScript discards a surplus argument without a
   * word, so `layer.call(tgt, memory)` ran the decoder against its own input and
   * never read the encoder. Every shape was right and the loss went down.
   *
   * The Python binding calls through here, so that is where it surfaced — the
   * borch.ts golden calls `forward` directly and was green.
   */
  call(x: Tensor, ...rest: Tensor[]): Tensor {
    return this.forward(x, ...rest);
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
   * The whole tree with dotted names, **this module first and named `""`.**
   *
   * `children` is one level and this is every level — a difference invisible on a
   * flat model, which is why the golden asks it of a `Sequential` inside a
   * `Sequential`. The names are the `state_dict` keys' prefixes, so a caller who
   * has a checkpoint key can find the layer it belongs to.
   *
   * `seen` drops a module reached twice. A model with tied weights holds one layer
   * under two names, and walked twice `apply` would initialise it twice.
   */
  namedModules(prefix = "", seen: Set<Module> = new Set()): [string, Module][] {
    if (seen.has(this)) return [];
    seen.add(this);
    const out: [string, Module][] = [[prefix, this]];
    for (const [name, child] of Object.entries(this.namedChildren())) {
      out.push(...child.namedModules(prefix ? `${prefix}.${name}` : name, seen));
    }
    return out;
  }

  /**
   * `fn` on every module in the tree, **children first**, then this one, and hands
   * back `this` so the line reads as a statement about the model.
   *
   * The order is torch's and it is the useful one: a container that reads its
   * children's shapes sees them already initialised.
   */
  apply(fn: (m: Module) => void): this {
    for (const child of Object.values(this.namedChildren())) child.apply(fn);
    fn(this);
    return this;
  }

  /**
   * One module by its dotted name — `"layer4.1.conv2"`, which is the shape of a
   * checkpoint key, so this is how a fine-tuning script reaches one layer.
   */
  getSubmodule(target: string): Module {
    if (target === "") return this;
    let at: Module = this;
    for (const part of target.split(".")) {
      const kid = at.namedChildren()[part];
      if (kid === undefined) {
        throw new RuntimeError(
          `${at.constructor.name} has no attribute \`${part}\` is not in the `
          + "browser subset.");
      }
      at = kid;
    }
    return at;
  }

  /** One parameter by its dotted name, as `getSubmodule` reaches a module. */
  getParameter(target: string): Tensor {
    return this.dotted(target, (m) => m.namedParameters(), "parameter");
  }

  /** One buffer by its dotted name. */
  getBuffer(target: string): Tensor {
    return this.dotted(target, (m) => m.namedBuffers(), "buffer");
  }

  private dotted(target: string, bank: (m: Module) => Record<string, Tensor>,
                 what: string): Tensor {
    const cut = target.lastIndexOf(".");
    const owner = this.getSubmodule(cut < 0 ? "" : target.slice(0, cut));
    const leaf = cut < 0 ? target : target.slice(cut + 1);
    const held = bank(owner)[leaf];
    if (held === undefined) {
      throw new RuntimeError(
        `${owner.constructor.name} has no ${what} named \`${leaf}\` is not in the `
        + "browser subset.");
    }
    return held;
  }

  /**
   * Attach a child under a name decided at run time.
   *
   * `namedChildren` reads this object's own fields, so assigning the property *is*
   * the registration — the method exists because a name held in a variable cannot
   * be written as a field. `registerModule` is torch's second spelling.
   */
  addModule(name: string, module: Module): void {
    (this as unknown as Record<string, Module>)[name] = module;
  }

  registerModule(name: string, module: Module): void {
    this.addModule(name, module);
  }

  /**
   * Freeze or unfreeze the whole tree. **This is how a backbone is frozen**, and it
   * returns `this` so the line reads as a statement about the model.
   */
  requiresGrad_(requiresGrad = true): this {
    for (const p of this.parameters()) p.requiresGrad = requiresGrad;
    return this;
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
   *
   * ## `strict` weighs both directions
   *
   * It used to weigh one. The loop walked **the values it was given** and
   * complained about names it did not recognise, and nothing ever asked which
   * of its own keys went unfilled. A checkpoint missing a weight loaded without
   * a word and that layer kept its initialisation — no exception, the model
   * stands, the numbers are plausible and wrong.
   *
   * Measured downstream: a cargo of 314 tensors with `conv_stem.weight` taken
   * out loaded strictly and reported 313 in. torch reports `missing_keys` and
   * `unexpected_keys` both, and this now does the same.
   *
   * The report comes back either way. `strict` decides whether a difference
   * stops the load or is left for the caller to weigh — a partial load is a
   * real thing to want (fine-tuning a head, warm-starting a trunk), and it
   * should be **asked for** rather than arrived at by accident.
   */
  loadStateDict(
    values: Readonly<Record<string, Tensor>>,
    strict = true,
  ): LoadReport {
    // **Buffers are accepted too.** When what goes out differs from what comes in, a
    // file cannot be read back by the thing that wrote it — `stateDict()` sends the
    // buffers out while this looked at `namedParameters()` alone, so strict mode raised
    // "unknown name". Saving and restoring have to look at **the same list**.
    const own = { ...this.namedParameters(), ...this.namedBuffers() };
    const unexpected: string[] = [];
    const filled = new Set<string>();
    for (const [name, src] of Object.entries(values)) {
      const dst = own[name];
      if (!dst) {
        unexpected.push(name);
        continue;
      }
      noGrad(() => dst.copyFrom(src));
      filled.add(name);
    }
    const missing = Object.keys(own).filter((name) => !filled.has(name));
    if (strict) refuseIncompatible(missing, unexpected);
    return { missing, unexpected };
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

  constructor(modules: readonly Module[] = []) {
    super();
    this.items = [...modules];
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

  extend(modules: readonly Module[]): this {
    this.items.push(...modules);
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

  constructor(modules: Readonly<Record<string, Module>> = {}) {
    super();
    for (const [name, m] of sortedEntries(modules)) this.items.set(name, m);
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

  constructor(values: readonly Tensor[] = []) {
    super();
    this.items = [...values];
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

  constructor(parameters: Readonly<Record<string, Tensor>> = {}) {
    super();
    for (const [name, p] of sortedEntries(parameters)) this.items.set(name, p);
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

  constructor(inFeatures: number, outFeatures: number, bias = true,
              device?: null, dtype?: null) {
    refuseDeviceDtype("Linear", device, dtype);
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
    // **Kept because `describe` prints them.** They were plain arguments: the shape
    // they build was stored and the numbers were not, so `print(model)` could say
    // `Conv2d()` and nothing else.
    private readonly inChannels: number,
    private readonly outChannels: number,
    private readonly kernelSize: number,
    private readonly spatial: number,
    private readonly stride = 1,
    private readonly padding = 0,
    bias = true,
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
      outChannels, inChannels / groups, ...new Array<number>(spatial).fill(kernelSize),
    ];
    // **The fan-in is divided by `groups`.** Each filter sees only
    // `inChannels / groups` channels, and initialised as though it saw all of them
    // a grouped convolution starts too small — torch divides here too.
    const bound = 1 / Math.sqrt(Math.max(1, fanIn(shape)));
    this.weight = uniform(shape, bound);
    this.bias = bias ? uniform([outChannels], bound) : null;
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

  /**
   * torch's `_ConvNd.extra_repr`, and **it prints the sizes as tuples** —
   * `kernel_size=(3, 3)` — because torch stores them that way. This library stores
   * one number and spreads it for the printing alone; changing what is stored would
   * reach into every forward pass for the sake of a string.
   *
   * **Several arguments print only when they are not the default.** A convolution
   * with no padding does not say `padding=(0, 0)`; one with padding does. Printing
   * them always is the same characters for a different layer, and reads as correct.
   */
  override describe(): string {
    return `${this.constructor.name}(${convExtra({
      spatial: this.spatial, inChannels: this.inChannels,
      outChannels: this.outChannels, kernelSize: this.kernelSize,
      stride: this.stride, padding: this.padding, dilation: this.dilation,
      groups: this.groups, hasBias: this.bias !== null,
      paddingMode: this.paddingMode,
    })})`;
  }
}

/**
 * The shared middle of every convolution's repr.
 *
 * **Written once because torch writes it once.** Six classes here print by the same
 * rule and the rule has five conditional clauses; six copies of it is six places for
 * one of them to go missing, and the missing one would only show at an argument
 * nobody passes in a test.
 */

/**
 * An adaptive pool's target, printed as Python would.
 *
 * **One number stays one number and a pair becomes a tuple**, which is what torch's
 * `output_size` holds — a reader who saw `(1,)` where torch says `1` would be looking
 * at a difference that is not there.
 */
function describeSize(value: number | readonly number[]): string {
  return typeof value === "number" ? `${value}` : `(${value.join(", ")})`;
}

/**
 * A number as Python's `repr` writes it.
 *
 * **The two languages switch to an exponent at different sizes.** JavaScript writes
 * `1e-5` as `0.00001` and only reaches for an exponent below `1e-7`; Python reaches
 * for one below `1e-4` and pads the exponent to two digits. Every normalisation layer
 * in this file prints `eps=1e-05`, so without this they all read as *nearly* right —
 * the same number, spelled the way the other language spells it.
 */
export function pyNumber(value: number): string {
  if (value !== 0 && Math.abs(value) < 1e-4) {
    const [mantissa = "0", exponent = "0"] = value.toExponential().split("e");
    const sign = exponent.startsWith("-") ? "-" : "+";
    const digits = exponent.replace(/^[-+]/, "").padStart(2, "0");
    return `${mantissa}e${sign}${digits}`;
  }
  return `${value}`;
}

interface ConvFields {
  spatial: number;
  inChannels: number;
  outChannels: number;
  kernelSize: number;
  stride: number;
  padding: number;
  dilation: number;
  groups: number;
  hasBias: boolean;
  paddingMode?: string;
  outputPadding?: number;
}

function convExtra(f: ConvFields): string {
  // **A one-element tuple carries a trailing comma in Python** — `(5,)`, not `(5)`.
  // The one-dimensional convolutions are the only place it shows, and without it they
  // print something that is not a tuple at all in the language being imitated.
  const spread = (value: number) =>
    f.spatial === 1 ? `(${value},)`
      : `(${new Array<number>(f.spatial).fill(value).join(", ")})`;
  let out = `${f.inChannels}, ${f.outChannels}, `
    + `kernel_size=${spread(f.kernelSize)}, stride=${spread(f.stride)}`;
  if (f.padding !== 0) out += `, padding=${spread(f.padding)}`;
  if (f.dilation !== 1) out += `, dilation=${spread(f.dilation)}`;
  if (f.outputPadding !== undefined && f.outputPadding !== 0) {
    out += `, output_padding=${spread(f.outputPadding)}`;
  }
  if (f.groups !== 1) out += `, groups=${f.groups}`;
  if (!f.hasBias) out += ", bias=False";
  if (f.paddingMode !== undefined && f.paddingMode !== "zeros") {
    out += `, padding_mode=${f.paddingMode}`;
  }
  return out;
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
  constructor(inChannels: number, outChannels: number, kernelSize: number, stride = 1, padding = 0,
              dilation = 1, groups = 1, bias = true,
              paddingMode: PadMode | "zeros" = "zeros",
              device?: null, dtype?: null) {
    refuseDeviceDtype("Conv1d", device, dtype);
    super(inChannels, outChannels, kernelSize, 1, stride, padding, bias, dilation, groups, paddingMode);
  }
}

export class Conv2d extends ConvND {
  constructor(inChannels: number, outChannels: number, kernelSize: number, stride = 1, padding = 0,
              dilation = 1, groups = 1, bias = true,
              paddingMode: PadMode | "zeros" = "zeros",
              device?: null, dtype?: null) {
    refuseDeviceDtype("Conv2d", device, dtype);
    super(inChannels, outChannels, kernelSize, 2, stride, padding, bias, dilation, groups, paddingMode);
  }
}

export class Conv3d extends ConvND {
  constructor(inChannels: number, outChannels: number, kernelSize: number, stride = 1, padding = 0,
              dilation = 1, groups = 1, bias = true,
              paddingMode: PadMode | "zeros" = "zeros",
              device?: null, dtype?: null) {
    refuseDeviceDtype("Conv3d", device, dtype);
    super(inChannels, outChannels, kernelSize, 3, stride, padding, bias, dilation, groups, paddingMode);
  }
}

/**
 * `out`'s values written into `x`'s buffer, handing back **`x` itself**.
 *
 * What `inplace: true` buys is not the value — `new ReLU(true).call(x)` returns exactly
 * what `new ReLU().call(x)` returns. It buys that the caller's tensor moved and the
 * thing handed back *is* the caller's tensor. So the flag is honoured through
 * `copyFrom` rather than by returning a new tensor, which would pass a value comparison
 * and fail the only thing the flag is for.
 *
 * The leaf refusal comes free with it: `copyFrom` goes through `mutate`, which stops on
 * a leaf that requires grad. torch stops there too, and letting it through means a
 * backward pass reads a value that has already moved.
 */
function writeBack(x: Tensor, out: Tensor): Tensor {
  return x.copyFrom(out);
}

/**
 * **`inplace` is not a convenience seat.** `nn.ReLU(inplace=True)` is a line most torch
 * models contain, and without the argument the call stops on the argument count rather
 * than doing something subtly different — which is the good kind of failure, but only
 * for someone reading the traceback.
 *
 * Six activations carry it here because six are what the golden asks about. torch gives
 * it to more (`ReLU6`, `Hardsigmoid`, `Hardswish`, `Mish`), and those seats are absent
 * rather than declared-and-unmeasured: a seat nothing asks about looks identical to a
 * working one from every axis this repository has.
 */
export class ReLU extends Module {
  constructor(private readonly inplace = false) {
    super();
  }

  override forward(x: Tensor): Tensor {
    const out = x.unary("relu");
    return this.inplace ? writeBack(x, out) : out;
  }
}

// ── Activation layers. Each wraps exactly one tensor method. ────────────────
//
// The one way this family fails is a wrapper **calling a different function**, and
// that is invisible to the eye and diverges only in the values — which is why the
// golden asks about the functional form and the layer form separately.

export class Hardsigmoid extends Module {
  constructor(private readonly inplace = false) {
    super();
  }

  override forward(x: Tensor): Tensor {
    const out = x.unary("hardsigmoid");
    return this.inplace ? writeBack(x, out) : out;
  }
}

export class Hardswish extends Module {
  constructor(private readonly inplace = false) {
    super();
  }

  override forward(x: Tensor): Tensor {
    const out = x.unary("hardswish");
    return this.inplace ? writeBack(x, out) : out;
  }
}

export class LogSigmoid extends Module {
  constructor(inplace = false) {
    super();
    if (inplace) {
      throw new RuntimeError(
        "LogSigmoid(inplace=true) got an unexpected keyword argument 'inplace' — "
        + "torch does not give this one an in-place form either");
    }
  }

  override forward(x: Tensor): Tensor {
    return x.unary("logsigmoid");
  }
}

export class Mish extends Module {
  constructor(private readonly inplace = false) {
    super();
  }

  override forward(x: Tensor): Tensor {
    const out = x.unary("mish");
    return this.inplace ? writeBack(x, out) : out;
  }
}

export class ReLU6 extends Module {
  constructor(private readonly inplace = false) {
    super();
  }

  override forward(x: Tensor): Tensor {
    const out = x.unary("relu6");
    return this.inplace ? writeBack(x, out) : out;
  }
}

export class SELU extends Module {
  constructor(private readonly inplace = false) {
    super();
  }

  override forward(x: Tensor): Tensor {
    const out = x.unary("selu");
    return this.inplace ? writeBack(x, out) : out;
  }
}

export class Softsign extends Module {
  constructor(inplace = false) {
    super();
    if (inplace) {
      throw new RuntimeError(
        "Softsign(inplace=true) got an unexpected keyword argument 'inplace' — "
        + "torch does not give this one an in-place form either");
    }
  }

  override forward(x: Tensor): Tensor {
    return x.unary("softsign");
  }
}

export class Tanhshrink extends Module {
  constructor(inplace = false) {
    super();
    if (inplace) {
      throw new RuntimeError(
        "Tanhshrink(inplace=true) got an unexpected keyword argument 'inplace' — "
        + "torch does not give this one an in-place form either");
    }
  }

  override forward(x: Tensor): Tensor {
    return x.unary("tanhshrink");
  }
}

export class CELU extends Module {
  constructor(private readonly alpha = 1.0, private readonly inplace = false) {
    super();
  }

  override forward(x: Tensor): Tensor {
    const out = x.celu(this.alpha);
    return this.inplace ? writeBack(x, out) : out;
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
  private readonly minVal: number;
  private readonly maxVal: number;

  /**
   * **`minValue` and `maxValue` are deprecated and they are not inert.**
   *
   * The signature table called this row short on purpose — "closing it means adding
   * two arguments torch itself tells you not to use". Measured, torch does more than
   * tolerate them: each **overrides** its replacement, even when both are given.
   *
   *     nn.Hardtanh(min_val=-1, min_value=-2)   ->   Hardtanh(min_val=-2, max_val=1.0)
   *
   * So they change the answer, and left out of a JavaScript signature they are not
   * refused — they are discarded in silence. That is the difference between a seat
   * torch says not to use and a seat a caller cannot tell went nowhere.
   *
   * Carried, warned about and honoured, in torch's order. `describe` prints `min_val`
   * whichever way it was set, as torch's does.
   */
  constructor(minVal = -1.0, maxVal = 1.0,
              private readonly inplace = false,
              minValue?: number, maxValue?: number) {
    super();
    for (const [old, now] of [["min_value", "min_val"],
                              ["max_value", "max_val"]] as const) {
      if ((old === "min_value" ? minValue : maxValue) !== undefined) {
        console.warn(`keyword argument \`${old}\` is deprecated and renamed to \`${now}\``);
      }
    }
    this.minVal = minValue ?? minVal;
    this.maxVal = maxValue ?? maxVal;
  }

  override forward(x: Tensor): Tensor {
    const out = x.hardtanh(this.minVal, this.maxVal);
    return this.inplace ? writeBack(x, out) : out;
  }

  override describe(): string {
    return `Hardtanh(min_val=${pyFloat(this.minVal)}, max_val=${pyFloat(this.maxVal)}`
      + `${this.inplace ? ", inplace=True" : ""})`;
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
  constructor(private readonly threshold: number, private readonly value: number,
              private readonly inplace = false) {
    super();
  }

  override forward(x: Tensor): Tensor {
    const out = x.threshold(this.threshold, this.value);
    return this.inplace ? writeBack(x, out) : out;
  }

  // `inplace` only when true, as `RReLU` — not the dropout family's always-print.
  override describe(): string {
    const pair = `threshold=${pyFloat(this.threshold)}, value=${pyFloat(this.value)}`;
    return `Threshold(${pair}${this.inplace ? ", inplace=True" : ""})`;
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
  constructor(private readonly inplace = false) {
    super();
  }

  override forward(x: Tensor): Tensor {
    const out = x.unary("silu");
    return this.inplace ? writeBack(x, out) : out;
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
  constructor(private readonly negativeSlope = 0.01,
              private readonly inplace = false) {
    super();
  }

  override forward(x: Tensor): Tensor {
    const out = x.leakyRelu(this.negativeSlope);
    return this.inplace ? writeBack(x, out) : out;
  }
}

/**
 * The negative side lies down exponentially. **Without varying α it cannot
 * be told apart from the argument-free entry in the table.**
 */
export class ELU extends Module {
  constructor(private readonly alpha = 1.0, private readonly inplace = false) {
    super();
  }

  override forward(x: Tensor): Tensor {
    const out = x.elu(this.alpha);
    return this.inplace ? writeBack(x, out) : out;
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

  override describe(): string {
    return `GELU(approximate='${this.approximate}')`;
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

  override describe(): string {
    return `Softmax(dim=${this.dim ?? "None"})`;
  }
}

export class LogSoftmax extends Module {
  constructor(private readonly dim: number | null = null) {
    super();
  }

  override forward(x: Tensor): Tensor {
    return x.logSoftmax(this.dim ?? defaultSoftmaxDim(x.shape.length));
  }

  override describe(): string {
    return `LogSoftmax(dim=${this.dim ?? "None"})`;
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

  constructor(numParameters = 1, init = 0.25, device?: null, dtype?: null) {
    refuseDeviceDtype("PReLU", device, dtype);
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

  /** **The count comes from the weight**, because that is where it landed — the
   * argument was not kept and the tensor is the same number. */
  override describe(): string {
    return `PReLU(num_parameters=${this.weight.shape[0]})`;
  }
}

/**
 * torch's `device` and `dtype`, **held so that what comes after them lands right.**
 *
 * borch.ts runs on one device in one precision, so these two are nothing here — and
 * leaving them out is free only while they are last. In `GroupNorm` and the three
 * `LazyBatchNorm`s torch puts `bias` *behind* them, keyword-only, so a fifth
 * positional argument reaches `device` in torch and reached `bias` here. That is a
 * shift rather than a short tail: both calls succeed and mean different things.
 *
 * Refusing at torch's own position is the trade the core made seventeen times for
 * the same two names. Where nothing follows them the seats stay out, and the row is
 * `shorter`, which is safe.
 */
function refuseDeviceDtype(layer: string, device: unknown, dtype: unknown): void {
  for (const [what, got] of [["device", device], ["dtype", dtype]] as const) {
    if (got !== undefined && got !== null) {
      throw new Error(
        `${layer}(${what}=…) is not in the browser subset — one device, one precision.`);
    }
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
  readonly weight: Tensor | null;
  readonly bias: Tensor | null;

  /**
   * **`affine` and `bias`, which this took neither of.** torch has both, and they
   * are two different halves of the same idea: `affine=false` is the layer with no
   * learnable scale or shift at all, and `bias=false` keeps the scale and drops the
   * shift. Without them the `stateDict` carried two keys torch's did not, so a
   * checkpoint written by a torch `GroupNorm(2, 4, affine=False)` could not be read
   * here in strict mode.
   *
   * The signature axis had this counted across thirteen layers before anybody wrote
   * it — the three `BatchNorm`s, the three `InstanceNorm`s, this, and the six lazy
   * variants. A counted absence is what makes the work happen.
   */
  constructor(
    private readonly numGroups: number,
    private readonly numChannels: number,
    private readonly eps = 1e-5,
    affine = true,
    device?: null,
    dtype?: null,
    bias = true,
  ) {
    refuseDeviceDtype("GroupNorm", device, dtype);
    super();
    this.weight = affine ? Tensor.owned([numChannels], 1) : null;
    this.bias = affine && bias ? Tensor.owned([numChannels], 0) : null;
    this.claim(...[this.weight, this.bias].filter((t): t is Tensor => t !== null));
  }

  override ownParameters(): Record<string, Tensor> {
    const out: Record<string, Tensor> = {};
    if (this.weight) out["weight"] = this.weight;
    if (this.bias) out["bias"] = this.bias;
    return out;
  }

  override forward(x: Tensor): Tensor {
    const width = this.weight?.size ?? this.bias?.size ?? 0;
    const shape = [1, width, ...new Array<number>(x.shape.length - 2).fill(1)];
    let out = x.groupNorm(this.numGroups, this.eps);
    if (this.weight) out = out.mul(this.weight.reshape(shape));
    return this.bias ? out.add(this.bias.reshape(shape)) : out;
  }

  /** torch's `GroupNorm.extra_repr`. **`bias` is not `affine`** — torch prints both,
   * because `affine=True, bias=False` is a scale with no shift and the two words
   * together are the only way to see it. */
  override describe(): string {
    return `GroupNorm(${this.numGroups}, ${this.numChannels}, eps=${pyNumber(this.eps)}, `
      + `affine=${this.weight !== null ? "True" : "False"}, `
      + `bias=${this.bias !== null ? "True" : "False"})`;
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
  readonly weight: Tensor | null;
  readonly bias: Tensor | null;

  /**
   * torch's list, in torch's seats. **`affine` defaults to false here and true
   * on `BatchNorm`** — that is torch, and flipping it swaps which layers carry
   * parameters, so the `state_dict` keys part wholesale.
   */
  constructor(
    private readonly numFeatures: number,
    private readonly eps = 1e-5,
    private readonly momentum = 0.1,
    // As `BatchNormND` — the other way round: here it is `trackRunningStats=true`
    // that is refused, so `false` is what this layer does.
    private readonly affine = false,
    private readonly trackRunningStats = false,
    device?: null,
    dtype?: null,
    bias = true,
  ) {
    super();
    // **`bias` sits behind `device` and `dtype`, and it did not.** torch declares it
    // keyword-only (`*, bias: bool = True`) after the pair, so a sixth positional
    // argument is `device` there and was `bias` here — a shift rather than a short
    // tail, and one the signature axis cannot see: `InstanceNorm1d/2d/3d` declare no
    // constructor of their own, so the emitted `.d.ts` has no argument list for them
    // and the axis reads `no argument list` rather than a disagreement.
    refuseDeviceDtype("InstanceNorm", device, dtype);
    if (trackRunningStats) {
      // torch registers three running statistics here and **reads them in eval
      // mode**. Accepting this and ignoring it leaves training right and
      // evaluation quietly wrong, and registering the buffers without the
      // forward using them only moves the parting to where the keys match and
      // the values do not. The core refuses at the same place for the same
      // reason (`borch/_nn.py`, `_InstanceNorm`).
      throw new Error(
        "InstanceNorm with trackRunningStats=true is not here yet.");
    }
    this.weight = affine ? Tensor.owned([numFeatures], 1) : null;
    this.bias = affine && bias ? Tensor.owned([numFeatures], 0) : null;
    this.claim(...[this.weight, this.bias].filter((t): t is Tensor => t !== null));
  }

  override ownParameters(): Record<string, Tensor> {
    const out: Record<string, Tensor> = {};
    if (this.weight) out["weight"] = this.weight;
    if (this.bias) out["bias"] = this.bias;
    return out;
  }

  override forward(x: Tensor): Tensor {
    let out = x.instanceNorm(this.eps);
    // Per channel, so the vector is broadcast along axis 1 and nowhere else.
    const spread = [1, -1, ...x.shape.slice(2).map(() => 1)];
    if (this.weight) out = out.mul(this.weight.reshape(spread));
    if (this.bias) out = out.add(this.bias.reshape(spread));
    return out;
  }

  /** The same wording as `BatchNormND`. **The two defaults are opposite** — a batch
   * norm is affine and tracks, an instance norm is neither — so the string is where
   * the difference shows without reading a constructor. */
  override describe(): string {
    return `${this.constructor.name}(${this.numFeatures}, eps=${pyNumber(this.eps)}, `
      + `momentum=${this.momentum}, affine=${this.affine ? "True" : "False"}, `
      + `bias=${this.bias !== null ? "True" : "False"}, `
      + `track_running_stats=${this.trackRunningStats ? "True" : "False"})`;
  }
}

/**
 * `torch.nn.InstanceNorm1d`. **It inherits `InstanceNormND` unchanged** — the
 * normalisation reduces everything but the channel axis, so the number in the
 * name does not reach the computation. The core says the same at the same place.
 *
 * These three and `BatchNorm1d` were **absent while all six lazy variants were
 * present**, so `LazyInstanceNorm2d` stood for the lazy form of a class that did
 * not exist. `BatchNormND`'s own comment promised "BatchNorm1d, 2d and 3d are
 * all this" with only two of the three written below it — documentation naming a
 * class nobody could import.
 *
 * Nothing could see it. The name axis in `tests/torch_gap.py` reads the python
 * core against torch; `borch-ts/test/run.py`'s ledger records golden cases. **The
 * core's `nn` against borch.ts's `nn` was not a question any file asked** —
 * `tests/test_nn_names.py` now asks it.
 */
// **They declare the list rather than inheriting it silently.** With an empty body
// the emitted `.d.ts` carries no argument list for these three, and the signature axis
// reads that as `no argument list` — not as agreement and not as a gap, just a row it
// skips. `InstanceNormND` had `bias` in `device`'s seat for exactly as long.
export class InstanceNorm1d extends InstanceNormND {
  constructor(numFeatures: number, eps?: number, momentum?: number, affine?: boolean,
              trackRunningStats?: boolean, device?: null, dtype?: null,
              bias?: boolean) {
    super(numFeatures, eps, momentum, affine, trackRunningStats, device, dtype, bias);
  }
}
export class InstanceNorm2d extends InstanceNormND {
  constructor(numFeatures: number, eps?: number, momentum?: number, affine?: boolean,
              trackRunningStats?: boolean, device?: null, dtype?: null,
              bias?: boolean) {
    super(numFeatures, eps, momentum, affine, trackRunningStats, device, dtype, bias);
  }
}
export class InstanceNorm3d extends InstanceNormND {
  constructor(numFeatures: number, eps?: number, momentum?: number, affine?: boolean,
              trackRunningStats?: boolean, device?: null, dtype?: null,
              bias?: boolean) {
    super(numFeatures, eps, momentum, affine, trackRunningStats, device, dtype, bias);
  }
}

/**
 * **It does not subtract the mean.** That is the only difference from
 * `LayerNorm`.
 */
export class RMSNorm extends Module {
  readonly weight: Tensor | null;

  /**
   * **`eps` was already in the kernel and the layer did not hand it over.**
   * `Tensor.rmsNorm(dims, eps)` has carried it from the start with torch's float32
   * default; this class took the shape alone, so `RMSNorm(4, 1e-3)` set nothing.
   *
   * `elementwiseAffine=false` is the layer with **no learnable scale at all** — not a
   * scale left at one. The difference shows in `stateDict`, where torch's has no
   * `weight` key, so a checkpoint written by `RMSNorm(4, elementwise_affine=False)`
   * could not be read here in strict mode. Same shape as `GroupNorm`'s `affine`.
   *
   * torch's `eps` defaults to `None`, meaning *use the dtype's epsilon*, and there is
   * one dtype here — so `null` and the kernel's default are the same answer.
   */
  constructor(private readonly normalizedShape: number | readonly number[],
              private readonly eps: number | null = null,
              elementwiseAffine = true,
              device?: null, dtype?: null) {
    super();
    refuseDeviceDtype("RMSNorm", device, dtype);
    const dims = typeof normalizedShape === "number"
      ? [normalizedShape] : [...normalizedShape];
    this.weight = elementwiseAffine ? Tensor.owned(dims, 1) : null;
    if (this.weight) this.claim(this.weight);
  }

  override ownParameters(): Record<string, Tensor> {
    return this.weight ? { weight: this.weight } : {};
  }

  override forward(x: Tensor): Tensor {
    const dims = typeof this.normalizedShape === "number"
      ? 1 : this.normalizedShape.length;
    const out = this.eps === null ? x.rmsNorm(dims) : x.rmsNorm(dims, this.eps);
    return this.weight ? out.mul(this.weight) : out;
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
    // As `ConvND` above — stored so that `describe` has them.
    private readonly inChannels: number,
    private readonly outChannels: number,
    private readonly kernelSize: number,
    private readonly spatial: number,
    private readonly stride = 1,
    private readonly padding = 0,
    bias = true,
    private readonly outputPadding = 0,
    private readonly groups = 1,
    private readonly dilation = 1,
    paddingMode: PadMode | "zeros" = "zeros",
  ) {
    super();
    // **torch refuses this itself, and the refusal is the whole of the argument.**
    // `ConvTranspose2d(..., padding_mode="reflect")` raises `Only "zeros" padding
    // mode is supported for ConvTranspose2d` — measured, not assumed. So this seat
    // is not an unimplemented feature: porting it means porting the error. Left out,
    // `paddingMode: "reflect"` was accepted in silence exactly where torch stops.
    if (paddingMode !== "zeros") {
      throw new RuntimeError(
        `Only "zeros" padding mode is supported for ConvTranspose${spatial}d`);
    }
    if (inChannels % groups !== 0 || outChannels % groups !== 0) {
      throw new RuntimeError(
        `groups=${groups} divides neither the input channels (${inChannels}) nor the `
        + `filters (${outChannels})`);
    }
    const bound = 1 / Math.sqrt(Math.max(1, (outChannels / groups) * kernelSize ** spatial));
    this.weight = uniform(
      [inChannels, outChannels / groups, ...new Array<number>(spatial).fill(kernelSize)], bound);
    this.bias = bias ? uniform([outChannels], bound) : null;
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

  /** As `ConvND`, plus `output_padding` — which prints only when it is set. */
  override describe(): string {
    return `${this.constructor.name}(${convExtra({
      spatial: this.spatial, inChannels: this.inChannels,
      outChannels: this.outChannels, kernelSize: this.kernelSize,
      stride: this.stride, padding: this.padding, dilation: this.dilation,
      groups: this.groups, hasBias: this.bias !== null,
      outputPadding: this.outputPadding,
    })})`;
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
  constructor(inChannels: number, outChannels: number, kernelSize: number, stride = 1, padding = 0,
              outputPadding = 0, groups = 1, bias = true, dilation = 1,
              paddingMode: PadMode | "zeros" = "zeros",
              device?: null, dtype?: null) {
    refuseDeviceDtype("ConvTranspose1d", device, dtype);
    super(inChannels, outChannels, kernelSize, 1, stride, padding, bias, outputPadding, groups,
          dilation, paddingMode);
  }
}

/** `torch.nn.ConvTranspose2d`. See `ConvTranspose1d` on the argument order. */
export class ConvTranspose2d extends ConvTransposeND {
  constructor(inChannels: number, outChannels: number, kernelSize: number, stride = 1, padding = 0,
              outputPadding = 0, groups = 1, bias = true, dilation = 1,
              paddingMode: PadMode | "zeros" = "zeros",
              device?: null, dtype?: null) {
    refuseDeviceDtype("ConvTranspose2d", device, dtype);
    super(inChannels, outChannels, kernelSize, 2, stride, padding, bias, outputPadding, groups,
          dilation, paddingMode);
  }
}

/** `torch.nn.ConvTranspose3d`. See `ConvTranspose1d` on the argument order. */
export class ConvTranspose3d extends ConvTransposeND {
  constructor(inChannels: number, outChannels: number, kernelSize: number, stride = 1, padding = 0,
              outputPadding = 0, groups = 1, bias = true, dilation = 1,
              paddingMode: PadMode | "zeros" = "zeros",
              device?: null, dtype?: null) {
    refuseDeviceDtype("ConvTranspose3d", device, dtype);
    super(inChannels, outChannels, kernelSize, 3, stride, padding, bias, outputPadding, groups,
          dilation, paddingMode);
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
  constructor(private readonly p = 0.5, private readonly inplace = false) {
    super();
  }

  override forward(x: Tensor): Tensor {
    const out = x.dropout(this.p, this.training);
    return this.inplace ? writeBack(x, out) : out;
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
  dropoutP = 0.0,
  isCausal = false,
  scale: number | null = null,
): Tensor {
  // **`dropoutP` and `scale` sit in torch's seats, and both were absent.**
  //
  // The binding accepted them and dropped them on the floor — `scaled_dot_product_
  // attention(q, k, v, mask, 0.1)` set no dropout and said nothing, and a `scale`
  // was ignored outright. `scale` is the sharper of the two: it *replaces*
  // `1/√dim`, so a caller who set it got the default back and a model whose
  // attention is scaled wrong trains to somewhere plausible.
  //
  // Their absence also put `isCausal` one seat early, which the signature axis read
  // as `dropped` — the bucket that means *a positional call lands on the wrong
  // parameter*. It was the last row in that bucket on either axis.
  const rank = query.shape.length;
  const dim = query.shape[rank - 1] ?? 1;
  const len = query.shape[rank - 2] ?? 1;
  const keyLen = key.shape[key.shape.length - 2] ?? 1;
  // The local yields the name; the parameter is torch's. Spelling the argument
  // `scaleOverride` to keep this line's `scale` is the wrong way round — the caller
  // reads the parameter and only this function reads the local.
  const scaleT = Tensor.full(
    [], scale === null ? 1 / Math.sqrt(dim) : scale);

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
    let scores = q.mm(k.transpose()).binary("mul", scaleT);
    if (causal) scores = scores.add(causal);
    if (attnMask) scores = scores.add(attnMask);
    return scores.softmax(-1).dropout(dropoutP, true).mm(v);
  };

  if (rank === 2) return one(query, key, value);
  const batch = query.shape[0] ?? 1;
  const outs: Tensor[] = [];
  for (let b = 0; b < batch; b++) {
    outs.push(one(query.select(0, b), key.select(0, b), value.select(0, b)));
  }
  return Tensor.stack(outs, 0);
}

/**
 * **`padding`, `dilation` and `ceilMode` hold torch's positions and refuse.**
 *
 * The core grew all three as working arguments; until the WGSL pooling kernel does
 * too, the choice is between an argument that is absent and one that is in the wrong
 * seat — and the wrong seat is what makes `new MaxPool2d(2, 2, 1)` set
 * `returnIndices` where torch and the core set `padding`. A refusal that names the
 * argument is the smaller wrong.
 *
 * It was written inline in `MaxPool2d` and the 1-D and 3-D layers had **no seats at
 * all** while printing `padding=0, dilation=1, ceil_mode=False` in their `describe`
 * — a repr for arguments they could not take, which is the tidiest way to look
 * finished. Three classes now share the one refusal, which is the third repetition
 * and the point at which this stops being a copy.
 *
 * **`padding` and `ceilMode` have left it.** They were refused on the ground that the
 * maximum's backward reads the input at each window position and a padded one has
 * none to read — which the average had answered one function away, by taking the
 * padding off the coordinate and skipping what falls outside. `poolND` does both for
 * the maximum now. `dilation` is the whole of what is left, and it keeps its seat for
 * the reason above.
 */
function refuseUnwiredPooling(layer: string, dilation: number): void {
  if (dilation !== 1) {
    throw new Error(
      `${layer}(dilation=…) is not carried across yet — the core implements it ` +
      "and this side does not. The argument is here so that it cannot take " +
      "another one's place.");
  }
}

export class MaxPool2d extends Module {
  /**
   * **`returnIndices` was missing until the layer that consumes them arrived.**
   * The name was here, the argument was not, and that is invisible from a name
   * count — `MaxUnpool` is what makes the positions worth asking for.
   */
  constructor(private readonly kernelSize = 2,
              private readonly stride?: number,
              readonly padding = 0,
              readonly dilation = 1,
              readonly returnIndices = false,
              readonly ceilMode = false) {
    super();
    refuseUnwiredPooling("MaxPool2d", dilation);
  }

  override forward(x: Tensor): Tensor {
    return x.poolND("max", this.kernelSize, this.stride, this.padding, this.ceilMode);
  }

  /** The values and the positions that produced them. `MaxUnpool2d` takes both. */
  pick(x: Tensor): { values: Tensor; indices: Tensor } {
    return x.maxPoolWithIndices(this.kernelSize, this.stride);
  }

  /** As `MaxPool1d`, and this one really holds the other three — they are refused
   * above unless left at their defaults, so what prints is what the layer does. */
  override describe(): string {
    return `MaxPool2d(kernel_size=${this.kernelSize}, stride=${this.stride ?? this.kernelSize}`
      + `, padding=${this.padding}, dilation=${this.dilation}`
      + `, ceil_mode=${this.ceilMode ? "True" : "False"})`;
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
  constructor(private readonly kernelSize = 2,
              private readonly stride?: number,
              readonly padding = 0,
              readonly dilation = 1,
              readonly returnIndices = false,
              readonly ceilMode = false) {
    super();
    refuseUnwiredPooling("MaxPool1d", dilation);
  }

  /** The values and the positions that produced them, as `MaxPool2d.pick`. */
  pick(x: Tensor): { values: Tensor; indices: Tensor } {
    return x.maxPoolWithIndices(this.kernelSize, this.stride);
  }

  override forward(x: Tensor): Tensor {
    return x.poolND("max", this.kernelSize, this.stride, this.padding, this.ceilMode);
  }

  /** torch's `_MaxPoolNd.extra_repr`. **A stride left unset prints the kernel**, which
   * is the step the layer actually takes — `stride=None` would be the argument rather
   * than the behaviour. */
  override describe(): string {
    return `MaxPool1d(kernel_size=${this.kernelSize}, stride=${this.stride ?? this.kernelSize}`
      + `, padding=${this.padding}, dilation=${this.dilation}`
      + `, ceil_mode=${this.ceilMode ? "True" : "False"})`;
  }
}

/** `torch.nn.MaxPool3d`. */
export class MaxPool3d extends Module {
  constructor(private readonly kernelSize = 2,
              private readonly stride?: number,
              readonly padding = 0,
              readonly dilation = 1,
              readonly returnIndices = false,
              readonly ceilMode = false) {
    super();
    refuseUnwiredPooling("MaxPool3d", dilation);
  }

  /** The values and the positions that produced them, as `MaxPool2d.pick`. */
  pick(x: Tensor): { values: Tensor; indices: Tensor } {
    return x.maxPoolWithIndices(this.kernelSize, this.stride);
  }

  override forward(x: Tensor): Tensor {
    return x.poolND("max", this.kernelSize, this.stride, this.padding, this.ceilMode);
  }

  /** As `MaxPool1d`. */
  override describe(): string {
    return `MaxPool3d(kernel_size=${this.kernelSize}, stride=${this.stride ?? this.kernelSize}`
      + `, padding=${this.padding}, dilation=${this.dilation}`
      + `, ceil_mode=${this.ceilMode ? "True" : "False"})`;
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
  constructor(protected readonly outputSize: number | readonly number[],
              readonly returnIndices = false) {
    super();
  }

  override forward(x: Tensor): Tensor {
    return x.adaptiveMaxPoolWithIndices(this.outputSize).values;
  }

  /** The values and the positions that produced them. */
  pick(x: Tensor): { values: Tensor; indices: Tensor } {
    return x.adaptiveMaxPoolWithIndices(this.outputSize);
  }

  /** Shared with the 2-D and 3-D names below, which subclass this one. */
  override describe(): string {
    return `${this.constructor.name}(output_size=${describeSize(this.outputSize)})`;
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
  constructor(protected readonly kernelSize: number,
              protected readonly stride?: number,
              protected readonly padding = 0) {
    super();
  }

  /**
   * @param outSize the shape to restore. Pooling is not injective — a window
   *   that ran off the edge leaves an ambiguity only the caller can settle.
   */
  place(x: Tensor, indices: Tensor, outSize?: readonly number[]): Tensor {
    return x.maxUnpool(indices, this.kernelSize, this.stride, this.padding, outSize);
  }

  /** **There is no one-argument form.** Reaching here means the positions were lost. */
  override forward(_x: Tensor): Tensor {
    throw new RuntimeError(
      "MaxUnpool needs the positions MaxPool chose — call place(x, indices), " +
      "not forward(x). It cannot sit in a Sequential for that reason.");
  }

  /** How many axes this one pools over — 1, 2 or 3. `describe` prints that many. */
  protected rank(): number {
    return 1;
  }

  /**
   * torch prints the three arguments **spread across the rank** — `MaxUnpool2d(2)` is
   * `kernel_size=(2, 2), stride=(2, 2), padding=(0, 0)`, and a `stride` left unset
   * prints the kernel rather than `None`.
   *
   * **One axis carries a trailing comma**: `MaxUnpool1d(2)` prints `(2,)`, which is
   * Python's one-element tuple and not a typo.
   */
  override describe(): string {
    const spread = (v: number): string => {
      const parts = Array.from({ length: this.rank() }, () => String(v));
      return parts.length === 1 ? `(${parts[0]},)` : `(${parts.join(", ")})`;
    };
    return `${this.constructor.name}(kernel_size=${spread(this.kernelSize)}, ` +
      `stride=${spread(this.stride ?? this.kernelSize)}, ` +
      `padding=${spread(this.padding)})`;
  }
}

/** `torch.nn.MaxUnpool2d`. */
export class MaxUnpool2d extends MaxUnpool1d {
  protected override rank(): number {
    return 2;
  }
}

/** `torch.nn.MaxUnpool3d`. */
export class MaxUnpool3d extends MaxUnpool1d {
  protected override rank(): number {
    return 3;
  }
}

/**
 * Flattens the axes from `startDim` to `endDim`, keeping the rest.
 *
 * **The two seats were missing and the defaults hid it.** torch's are `1` and `-1`,
 * which is "keep the batch axis and fold everything else" — exactly what this did
 * with no arguments at all, so every case that used the default passed while
 * `Flatten(0)` and `Flatten(1, 2)` were arguments JavaScript dropped. The class
 * declared no constructor, so the signature axis reported it as *unreadable* rather
 * than as short; that reader followed `extends` and found `Module`'s nothing.
 */
export class Flatten extends Module {
  constructor(private readonly startDim = 1, private readonly endDim = -1) {
    super();
  }

  override forward(x: Tensor): Tensor {
    const rank = x.shape.length;
    const from = this.startDim < 0 ? this.startDim + rank : this.startDim;
    const to = this.endDim < 0 ? this.endDim + rank : this.endDim;
    if (from > to || from < 0 || to >= rank) {
      throw new RuntimeError(
        `Flatten(${this.startDim}, ${this.endDim}) does not name a run of axes in a `
        + `rank-${rank} tensor.`);
    }
    let folded = 1;
    for (let i = from; i <= to; i++) folded *= x.shape[i] ?? 1;
    return x.reshape([...x.shape.slice(0, from), folded, ...x.shape.slice(to + 1)]);
  }

  override describe(): string {
    return `Flatten(start_dim=${this.startDim}, end_dim=${this.endDim})`;
  }
}

/**
 * The opposite of `Flatten`. Spreads one axis into several.
 */
export class Unflatten extends Module {
  constructor(private readonly dim: number,
              private readonly unflattenedSize: readonly number[]) {
    super();
  }

  override forward(x: Tensor): Tensor {
    return x.unflatten(this.dim, this.unflattenedSize);
  }

  override describe(): string {
    return `Unflatten(dim=${this.dim}, `
      + `unflattened_size=(${this.unflattenedSize.join(", ")}))`;
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
 * Pass `_randomSamples` and it uses those. It is where torch's
 * `_random_samples` goes, and it is needed to freeze values for comparison
 * — without it the three implementations' generators differ and the values
 * do not match.
 *
 * **`outputSize` and `outputRatio` are one or the other.** Giving both, or
 * neither, stops — as in torch.
 */
export class FractionalMaxPoolND extends Module {
  constructor(private readonly spatial: number,
              private readonly kernelSize: number,
              private readonly outputSize: number | readonly number[] | null = null,
              private readonly outputRatio: number | readonly number[] | null = null,
              returnIndices = false,
              private readonly _randomSamples: readonly (readonly number[])[] | null
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
    const samples: readonly (readonly number[])[] = this._randomSamples
      ?? Array.from({ length: planes }, () =>
        Array.from({ length: this.spatial }, () => uniform01()));
    return x.fractionalMaxPool(this.kernelSize, this.sizesFor(x.shape), samples);
  }

  override forward(x: Tensor): Tensor {
    return this.pool(x).values;
  }
}

export class FractionalMaxPool2d extends FractionalMaxPoolND {
  constructor(kernelSize: number,
              outputSize: number | readonly number[] | null = null,
              outputRatio: number | readonly number[] | null = null,
              returnIndices = false,
              _randomSamples: readonly (readonly number[])[] | null = null) {
    super(2, kernelSize, outputSize, outputRatio, returnIndices, _randomSamples);
  }
}

export class FractionalMaxPool3d extends FractionalMaxPoolND {
  constructor(kernelSize: number,
              outputSize: number | readonly number[] | null = null,
              outputRatio: number | readonly number[] | null = null,
              returnIndices = false,
              _randomSamples: readonly (readonly number[])[] | null = null) {
    super(3, kernelSize, outputSize, outputRatio, returnIndices, _randomSamples);
  }
}

export class AvgPool2d extends Module {
  /**
   * **The other four came from moving to the path its two siblings already use.**
   *
   * This took `(kernelSize, stride)` and called `avgPool2d`, the two-dimensional
   * kernel. `AvgPool1d` and `AvgPool3d` call `poolND`, which reads its rank off the
   * input (`shape.length - 2`) and has carried `padding`, `ceilMode`,
   * `countIncludePad` and `divisorOverride` from the start. So the missing arguments
   * were not missing arithmetic — they were one call away, in a function the file
   * next door was already using, and the 1-D layer's own comment explains
   * `divisorOverride`'s absence there in a sentence that only makes sense if the
   * 2-D form has it.
   */
  constructor(private readonly kernelSize: number,
              private readonly stride?: number,
              private readonly padding = 0,
              private readonly ceilMode = false,
              private readonly countIncludePad = true,
              private readonly divisorOverride: number | null = null) {
    super();
  }

  override forward(x: Tensor): Tensor {
    return x.poolND("avg", this.kernelSize, this.stride, this.padding, this.ceilMode,
                    this.countIncludePad, this.divisorOverride);
  }

  /** torch's `_AvgPoolNd.extra_repr` — three arguments and no `ceil_mode`. */
  override describe(): string {
    return `AvgPool2d(kernel_size=${this.kernelSize}, `
      + `stride=${this.stride ?? this.kernelSize}, padding=${this.padding})`;
  }
}

/**
 * `torch.nn.AvgPool1d`. **It goes through `poolND`** — `avgPool2d` is a
 * two-dimensional kernel, and this one has a different number of remaining
 * axes, so it cannot use that path.
 */
export class AvgPool1d extends Module {
  /**
   * **`AvgPool1d` has no `divisorOverride` and this is not an omission** — torch
   * gives that argument to the 2-D and 3-D forms alone, so a seat here would be one
   * this library invented.
   */
  constructor(private readonly kernelSize: number,
              private readonly stride?: number,
              private readonly padding = 0,
              private readonly ceilMode = false,
              private readonly countIncludePad = true) {
    super();
  }

  override forward(x: Tensor): Tensor {
    return x.poolND("avg", this.kernelSize, this.stride, this.padding, this.ceilMode,
                    this.countIncludePad);
  }

  /** **The one-dimensional average pool prints one-tuples** where its 2-D and 3-D
   * siblings print bare numbers. That is torch's own inconsistency between the three,
   * and it is copied rather than tidied. */
  override describe(): string {
    return `AvgPool1d(kernel_size=(${this.kernelSize},), `
      + `stride=(${this.stride ?? this.kernelSize},), padding=(${this.padding},))`;
  }
}

/** `torch.nn.AvgPool3d`. */
export class AvgPool3d extends Module {
  constructor(private readonly kernelSize: number,
              private readonly stride?: number,
              private readonly padding = 0,
              private readonly ceilMode = false,
              private readonly countIncludePad = true,
              private readonly divisorOverride: number | null = null) {
    super();
  }

  override forward(x: Tensor): Tensor {
    return x.poolND("avg", this.kernelSize, this.stride, this.padding, this.ceilMode,
                    this.countIncludePad, this.divisorOverride);
  }

  /** As `AvgPool2d`. */
  override describe(): string {
    return `AvgPool3d(kernel_size=${this.kernelSize}, `
      + `stride=${this.stride ?? this.kernelSize}, padding=${this.padding})`;
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
  constructor(private readonly outputSize: number | readonly number[]) {
    super();
  }

  override forward(x: Tensor): Tensor {
    return x.adaptivePool("avg", this.outputSize);
  }

  override describe(): string {
    return `AdaptiveAvgPool1d(output_size=${describeSize(this.outputSize)})`;
  }
}

/** `torch.nn.AdaptiveAvgPool3d`. It takes the output size. */
export class AdaptiveAvgPool3d extends Module {
  constructor(private readonly outputSize: number | readonly number[]) {
    super();
  }

  override forward(x: Tensor): Tensor {
    return x.adaptivePool("avg", this.outputSize);
  }

  override describe(): string {
    return `AdaptiveAvgPool3d(output_size=${describeSize(this.outputSize)})`;
  }
}

/**
 * p-norm pooling. **The first argument is `normType`**, not the kernel
 * (that is torch's shape).
 */
export class LPPool1d extends Module {
  constructor(private readonly normType: number,
              private readonly kernelSize: number,
              private readonly stride?: number,
              private readonly ceilMode = false) {
    super();
  }

  override forward(x: Tensor): Tensor {
    return x.lpPool(this.normType, this.kernelSize, this.stride, this.ceilMode);
  }

  /** torch's `_LPPoolNd.extra_repr`. Two things here:
   *
   * **The stride prints as `None`** and not as the kernel — the power pool is the one
   * place torch leaves it unfilled, and copying the neighbours' rule would be a
   * plausible string for a different layer.
   *
   * **`norm_type` prints what it was given**, and JavaScript cannot tell `2` from
   * `2.0`. torch preserves the type it received, so `LPPool1d(2)` says `2` there and
   * `LPPool1d(2.0)` says `2.0`; this side has one number and has to pick. It picks the
   * integer spelling, which is what every caller passes and what the frozen strings
   * hold — the other choice would be wrong for the common call and right for the rare
   * one. */
  override describe(): string {
    return `${this.constructor.name}(norm_type=${pyNumber(this.normType)}, `
      + `kernel_size=${this.kernelSize}, stride=${this.stride ?? "None"}, `
      + `ceil_mode=${this.ceilMode ? "True" : "False"})`;
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
              private readonly elementwiseAffine = true, bias = true,
              device?: null, dtype?: null) {
    refuseDeviceDtype("LayerNorm", device, dtype);
    super();
    const shape = typeof normalizedShape === "number"
      ? [normalizedShape] : [...normalizedShape];
    this.shape = shape;
    this.dims = shape.length;
    if (elementwiseAffine) {
      this.weight = Tensor.owned(shape, 1);
      this.claim(this.weight);
      if (bias) {
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

  /** torch's `LayerNorm.extra_repr`. The shape prints as a Python list. */
  override describe(): string {
    return `LayerNorm((${this.shape.join(", ")}${this.shape.length === 1 ? "," : ""}), `
      + `eps=${pyNumber(this.eps)}, elementwise_affine=${this.elementwiseAffine ? "True" : "False"}, `
      + `bias=${this.bias !== null ? "True" : "False"})`;
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
  /**
   * **`recomputeScaleFactor` was missing and the kernel behaved as though it were
   * on.** With a fractional factor torch works out the integer output size and then
   * either samples at the factor it was given (the default) or derives the scale back
   * from that size (the flag). Same shape, different values — measured on torch at
   * `scale_factor=1.5`: sums 120 and 132.
   */
  constructor(private readonly size: number | readonly number[] | null = null,
              private readonly scaleFactor: number | null = null,
              private readonly mode: "nearest" | "bilinear" = "nearest",
              private readonly alignCorners: boolean | null = null,
              private readonly recomputeScaleFactor: boolean | null = null) {
    super();
  }

  override forward(x: Tensor): Tensor {
    if (this.size === null && this.scaleFactor === null) {
      throw new RuntimeError("either size or scale_factor should be defined");
    }
    return x.interpolate(this.size, this.scaleFactor, this.mode,
                         this.alignCorners ?? false, this.recomputeScaleFactor);
  }

  /** **Whichever of the two was given**, not both — a layer built with a size prints
   * its size and one built with a factor prints its factor. Printing the other shows
   * `None` beside a number that is set. */
  override describe(): string {
    const head = this.scaleFactor !== null
      ? `scale_factor=${pyFloat(this.scaleFactor)}`
      : `size=${describeSize(this.size ?? 0)}`;
    return `Upsample(${head}, mode='${this.mode}')`;
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

  /**
   * **`bias` sits third, as it does in torch, and `numChunks` behind it.** The two
   * were the other way round — `(inputSize, hidden, gates, bias)` — so a positional
   * `new RNNCellBase(4, 8, false)` set the *gate count* to `false` here and the bias
   * flag in torch. Both build a layer; only the shapes differ.
   *
   * It was invisible while the third parameter was called `gates`: the lists could
   * not be lined up at all, so the row sat in the axis's `unaligned` bucket, which
   * reports nothing beneath it. Renaming it to torch's `numChunks` is what made the
   * order visible, and the order is the part that mattered.
   */
  constructor(readonly inputSize: number, readonly hidden: number,
              readonly bias = true, numChunks = 1,
              device?: null, dtype?: null) {
    refuseDeviceDtype("RNNCellBase", device, dtype);
    super();
    const rows = hidden * numChunks;
    const bound = 1 / Math.sqrt(Math.max(1, hidden));
    this.weightIh = uniform([rows, inputSize], bound);
    this.weightHh = uniform([rows, hidden], bound);
    this.biasIh = bias ? uniform([rows], bound) : null;
    this.biasHh = bias ? uniform([rows], bound) : null;
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
  constructor(inputSize: number, hidden: number, bias = true,
              readonly nonlinearity: "tanh" | "relu" = "tanh",
              device?: null, dtype?: null) {
    refuseDeviceDtype("RNNCell", device, dtype);
    super(inputSize, hidden, bias, 1);
  }

  step(x: Tensor, hx: Tensor | null = null): Tensor {
    const h = hx ?? this.zeros(x);
    return this.gi(x).add(this.gh(h)).unary(this.nonlinearity);
  }

  override forward(x: Tensor): Tensor { return this.step(x); }

  override describe(): string {
    let parts = `${this.inputSize}, ${this.hidden}`;
    if (!this.bias) parts += ", bias=False";
    if (this.nonlinearity !== "tanh") parts += `, nonlinearity=${this.nonlinearity}`;
    return `RNNCell(${parts})`;
  }
}

export class GRUCell extends RNNCellBase {
  constructor(inputSize: number, hidden: number, bias = true,
              device?: null, dtype?: null) {
    refuseDeviceDtype("GRUCell", device, dtype);
    super(inputSize, hidden, bias, 3);
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
      `${this.bias ? "" : ", bias=False"})`;
  }
}

/**
 * **Alone in returning two** — `(h, c)`. Forcing the three into one shape
 * loses the memory cell.
 */
export class LSTMCell extends RNNCellBase {
  constructor(inputSize: number, hidden: number, bias = true,
              device?: null, dtype?: null) {
    refuseDeviceDtype("LSTMCell", device, dtype);
    super(inputSize, hidden, bias, 4);
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
      `${this.bias ? "" : ", bias=False"})`;
  }
}

// ── The remaining layers ───────────────────────────────────────────────

export class Unfold extends Module {
  constructor(readonly kernelSize: number, readonly dilation = 1,
              readonly padding = 0, readonly stride = 1) { super(); }

  override forward(x: Tensor): Tensor {
    return x.unfoldIm2col(this.kernelSize, this.dilation, this.padding, this.stride);
  }

  override describe(): string {
    return `Unfold(kernel_size=${this.kernelSize}, dilation=${this.dilation}, ` +
      `padding=${this.padding}, stride=${this.stride})`;
  }
}

export class Fold extends Module {
  constructor(readonly outputSize: [number, number], readonly kernelSize: number,
              readonly dilation = 1, readonly padding = 0,
              readonly stride = 1) { super(); }

  override forward(x: Tensor): Tensor {
    return x.fold(this.outputSize, this.kernelSize, this.dilation, this.padding,
      this.stride);
  }

  override describe(): string {
    return `Fold(output_size=(${this.outputSize.join(", ")}), ` +
      `kernel_size=${this.kernelSize}, dilation=${this.dilation}, ` +
      `padding=${this.padding}, stride=${this.stride})`;
  }
}

/**
 * Mixes two inputs **at once.** The weight has three axes, `(out, in1,
 * in2)`.
 */
export class Bilinear extends Module {
  readonly weight: Tensor;
  readonly bias: Tensor | null;

  /**
   * **`bias` was accepted by the binding and had nowhere to go.** This constructor
   * took three numbers and always built a bias, so `Bilinear(a, b, c, bias=False)`
   * got one anyway — no error, a parameter the caller asked not to have, and a
   * gradient flowing into it every step. Found by the argument check once it was
   * pointed at `nn` as well as at `optim`.
   */
  constructor(readonly in1Features: number, readonly in2Features: number,
              readonly outFeatures: number, bias = true,
              device?: null, dtype?: null) {
    refuseDeviceDtype("Bilinear", device, dtype);
    super();
    const bound = 1 / Math.sqrt(Math.max(1, in1Features));
    this.weight = uniform([outFeatures, in1Features, in2Features], bound);
    this.bias = bias ? uniform([outFeatures], bound) : null;
    this.claim(...(this.bias ? [this.weight, this.bias] : [this.weight]));
  }

  override ownParameters(): Record<string, Tensor> {
    return this.bias
      ? { weight: this.weight, bias: this.bias }
      : { weight: this.weight };
  }

  override forward(x: Tensor): Tensor { return x; }

  call2(x1: Tensor, x2: Tensor): Tensor {
    return x1.bilinear(x2, this.weight, this.bias);
  }

  // **This printed `in1_features` while the constructor took `in1`.** The third of
  // that shape today — `PixelShuffle` and `PixelUnshuffle` were the others, and the
  // core's `_Rearrange` was the same thing in Python, where it was a `TypeError`
  // rather than a quiet mismatch. Here the repr claimed a name the class did not
  // answer to and nothing could notice, because JavaScript is positional.
  override describe(): string {
    return `Bilinear(in1_features=${this.in1Features}, ` +
      `in2_features=${this.in2Features}, out_features=${this.outFeatures}, ` +
      `bias=${this.bias ? "True" : "False"})`;
  }
}

export class LocalResponseNorm extends Module {
  constructor(readonly size: number, readonly alpha = 1e-4,
              readonly beta = 0.75, readonly k = 1.0) { super(); }

  override forward(x: Tensor): Tensor {
    return x.localResponseNorm(this.size, this.alpha, this.beta, this.k);
  }

  override describe(): string {
    // **`alpha` went unguarded for as long as the other two were guarded**, two lines
    // below a comment explaining exactly why they needed it. It passed only because
    // torch's default `alpha` is 1e-4, which is not integral and prints the same
    // either way — so the one golden case that existed could never see it. At
    // `alpha=1.0` torch says `alpha=1.0` and this said `alpha=1`.
    //
    // Knowing the rule and applying it to two of three arguments is not carelessness:
    // `beta` and `k` were the ones being edited. It is a claim of completeness scoped
    // to what the author had open, which is a shape this repository met twice today.
    //
    // `size` is torch's only integer here; the other three are floats and all three
    // go through `pyFloat` now. `tests/test_describe_floats.py` asks torch's own
    // signature which is which, so the next one cannot depend on anybody noticing.
    return `LocalResponseNorm(${this.size}, alpha=${pyFloat(this.alpha)}, ` +
      `beta=${pyFloat(this.beta)}, k=${pyFloat(this.k)})`;
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
  constructor(readonly lower = 1 / 8, readonly upper = 1 / 3,
              private readonly inplace = false) { super(); }

  override forward(x: Tensor): Tensor {
    const out = x.rrelu(this.lower, this.upper, this.training);
    return this.inplace ? writeBack(x, out) : out;
  }

  /**
   * **`inplace` appears only when it is true**, which is not how the dropout family
   * below prints it — those show `inplace=False` always. Two conventions inside one
   * feature, and the only way to know which applies where is to call `repr` on the
   * real thing. Measured: `RReLU(lower=0.125, upper=0.3333333333333333)` against
   * `RReLU(lower=0.125, upper=0.3333333333333333, inplace=True)`.
   */
  override describe(): string {
    const bounds = `lower=${pyFloat(this.lower)}, upper=${pyFloat(this.upper)}`;
    return `RReLU(${bounds}${this.inplace ? ", inplace=True" : ""})`;
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
  /**
   * **`size` was missing, and its absence was not a short tail.** torch takes
   * `(size, scale_factor)` and takes *exactly one of them*; this took a `scale`
   * defaulting to 2, so three calls parted at once:
   *
   *     new UpsamplingNearest2d()            here doubled · torch and the core refuse
   *     UpsamplingNearest2d(size=(5, 7))     here had no seat for it at all
   *
   * A default that answers where the authority refuses is the same fault as an
   * argument accepted and dropped: the call that stops on real torch runs here, and
   * the divergence shows up at the port rather than at the call.
   */
  constructor(readonly label: string,
              readonly mode: "nearest" | "bilinear",
              readonly size: number | readonly number[] | null = null,
              readonly scaleFactor: number | null = null) {
    super();
    if ((size === null) === (scaleFactor === null)) {
      throw new Error(
        `${label}: only one of size or scale_factor should be defined`);
    }
  }

  private target(x: Tensor): [number, number] {
    if (this.size !== null) {
      return typeof this.size === "number"
        ? [this.size, this.size]
        : [this.size[0] ?? 1, this.size[1] ?? this.size[0] ?? 1];
    }
    const k = this.scaleFactor ?? 1;
    return [(x.shape[2] ?? 1) * k, (x.shape[3] ?? 1) * k];
  }

  override forward(x: Tensor): Tensor {
    const [h, w] = this.target(x);
    if (this.mode === "nearest") return x.interpolate([h, w], null, "nearest");
    return x.interpolateBilinear(h, w, true);
  }

  override describe(): string {
    const what = this.size !== null
      ? `size=${JSON.stringify(this.size)}`
      : `scale_factor=${(this.scaleFactor ?? 1).toFixed(1)}`;
    return `${this.label}(${what}, mode='${this.mode}')`;
  }
}

export class UpsamplingNearest2d extends UpsamplingBase {
  constructor(size: number | readonly number[] | null = null,
              scaleFactor: number | null = null) {
    super("UpsamplingNearest2d", "nearest", size, scaleFactor);
  }
}
export class UpsamplingBilinear2d extends UpsamplingBase {
  constructor(size: number | readonly number[] | null = null,
              scaleFactor: number | null = null) {
    super("UpsamplingBilinear2d", "bilinear", size, scaleFactor);
  }
}

/**
 * A trainable table turning an index into a vector.
 *
 * **This description belongs on the class, and putting a class here took someone
 * else's.** The JSDoc above `EmbeddingBag` was "One row per bag…", and inserting a
 * new class in front of it silently handed that sentence to this one — the emitted
 * `.d.ts` attaches a comment to whatever declaration follows it, so the site would
 * have described `Embedding` as a bag and `EmbeddingBag` as nothing at all. Nothing
 * raises; `test_site.py` noticed that a Korean entry had lost its source.
 */
export class Embedding extends Module {
  readonly weight: Tensor;
  readonly paddingIdx: number | null;

  /**
   * torch's nine, of which two do real work here.
   *
   * **The comment on `embedding()` said this layer was absent from all three and
   * that the golden did not ask for it.** That was true when it was written. The
   * core then took torch's nine arguments, seven cases were added, and this side
   * was not touched — so the binding answered `module 'borch_webgpu._nn' has no
   * attribute 'Embedding'` seven times and the sentence explaining the absence was
   * the reason nobody looked.
   *
   * Two arguments do real work and each has a case that fails without it:
   *
   * - **`maxNorm` shortens the rows that were looked up, in the table itself.** The
   *   same side effect on a parameter that `embeddingBag` has, through the same
   *   `renormRows`, not a second copy of the rule.
   * - **`paddingIdx` stops that row learning.** The forward is untouched — torch
   *   returns the padding row's values like any other — so a version that leaves the
   *   gradient flowing passes every value case and fails only on the gradient one.
   *   Done here by splitting the table into a part the gradient flows through and a
   *   detached part, which needs no new autograd machinery.
   *
   *   **This used to say "an implementation that masks the output instead" passes
   *   every value case.** It does not: torch hands the padding row's real values
   *   back, so masking the output changes the answer and the *value* case catches it.
   *   Measured both ways — cut the gradient block and only the gradient case reddens;
   *   mask the output and only the forward case reddens. Two defects, two cases, and
   *   the sentence had merged them into one.
   *
   * And one line that is about **who supplied the weights** rather than about
   * padding: a fresh table has its padding row zeroed and a given one does not.
   * torch draws it in the same place, and both halves are asked because either
   * alone reads as a rule about padding.
   */
  constructor(readonly numEmbeddings: number, readonly embeddingDim: number,
              paddingIdx: number | null = null,
              readonly maxNorm: number | null = null,
              readonly normType = 2,
              readonly scaleGradByFreq = false,
              readonly sparse = false,
              _weight: Tensor | null = null,
              _freeze = false,
              device?: null, dtype?: null) {
    refuseDeviceDtype("Embedding", device, dtype);
    super();
    if (scaleGradByFreq) {
      throw new NotImplementedError("Embedding(scaleGradByFreq) is not carried across");
    }
    if (sparse) {
      throw new NotImplementedError(
        "Embedding(sparse) is not carried across — there is no sparse gradient here");
    }
    if (paddingIdx !== null) {
      if (paddingIdx >= numEmbeddings || paddingIdx < -numEmbeddings) {
        throw new Error("padding_idx must be within num_embeddings");
      }
      if (paddingIdx < 0) paddingIdx = numEmbeddings + paddingIdx;
    }
    this.paddingIdx = paddingIdx;
    if (_weight !== null) {
      this.weight = _weight;
    } else {
      const fresh = Tensor.randn([numEmbeddings, embeddingDim]);
      // Only a **fresh** table gets the padding row zeroed. A caller who handed
      // weights in meant them.
      this.weight = paddingIdx === null ? fresh : fresh.mul(this.rowMask(paddingIdx));
    }
    this.weight.requiresGrad = !_freeze;
    this.claim(this.weight);
  }

  /** `[numEmbeddings, 1]`, zero on `row` and one everywhere else. */
  private rowMask(row: number): Tensor {
    return Tensor.arange(0, this.numEmbeddings).ne(Tensor.full([], row))
      .to("float32").reshape([this.numEmbeddings, 1]);
  }

  override ownParameters(): Record<string, Tensor> {
    return { weight: this.weight };
  }

  override forward(idx: Tensor): Tensor {
    if (this.maxNorm !== null) renormRows(this.weight, idx, this.maxNorm, this.normType);
    // **The padding row keeps its values and loses its gradient.** `keep` carries the
    // gradient, `drop` carries the same numbers with the path cut, and the sum is the
    // table unchanged. Masking the output instead would pass every value case here
    // and fail only `grad::Embedding(padding_idx)`.
    let table = this.weight;
    if (this.paddingIdx !== null) {
      const keep = this.rowMask(this.paddingIdx);
      table = table.mul(keep)
        .add(table.detach().mul(Tensor.full([this.numEmbeddings, 1], 1).sub(keep)));
    }
    return embedding(idx, table);
  }

  override describe(): string {
    const parts = [`${this.numEmbeddings}, ${this.embeddingDim}`];
    if (this.paddingIdx !== null) parts.push(`padding_idx=${this.paddingIdx}`);
    // **`max_norm` is a float and `padding_idx` is an index.** `2.0` and `1`, and the
    // golden froze both spellings — this row failed on the punctuation alone.
    if (this.maxNorm !== null) parts.push(`max_norm=${pyFloat(this.maxNorm)}`);
    return `Embedding(${parts.join(", ")})`;
  }
}

/**
 * One row per bag. Selecting from the table **and combining** is one layer.
 */
export class EmbeddingBag extends Module {
  readonly weight: Tensor;

  /**
   * **`mode` sits sixth, where torch has it.**
   *
   * It used to be third, so `new EmbeddingBag(10, 3, "sum")` set `maxNorm` in
   * torch and the mode here. Both readings then build a layer and return bags of
   * the right shape, and only the numbers differ.
   */
  constructor(readonly numEmbeddings: number, readonly embeddingDim: number,
              readonly maxNorm: number | null = null,
              readonly normType = 2,
              readonly scaleGradByFreq = false,
              readonly mode: "sum" | "mean" | "max" = "mean",
              readonly sparse = false,
              _weight: Tensor | null = null,
              readonly includeLastOffset = false,
              readonly paddingIdx: number | null = null,
              device?: null, dtype?: null) {
    refuseDeviceDtype("EmbeddingBag", device, dtype);
    super();
    this.weight = _weight ?? uniform([numEmbeddings, embeddingDim], 1);
    this.claim(this.weight);
  }

  override ownParameters(): Record<string, Tensor> {
    return { weight: this.weight };
  }

  override forward(idx: Tensor): Tensor {
    // `embeddingBag` does the computation — the layer and the function are not two
    // copies of it.
    return embeddingBag(idx, this.weight, null, this.maxNorm, this.normType,
                        this.scaleGradByFreq, this.mode, this.sparse, null,
                        this.includeLastOffset, this.paddingIdx);
  }

  /**
   * Cuts a one-dimensional index run into bags with `offsets`.
   *
   * **Bags of differing lengths are the reason this name exists.** A
   * two-dimensional input only allows bags of equal length, and the actual
   * uses (shopping carts, sentences) have differing lengths.
   */
  callOffsets(idx: Tensor, offsets: readonly number[]): Tensor {
    return embeddingBag(idx, this.weight, offsets, this.maxNorm, this.normType,
                        this.scaleGradByFreq, this.mode, this.sparse, null,
                        this.includeLastOffset, this.paddingIdx);
  }

  override describe(): string {
    return `EmbeddingBag(${this.numEmbeddings}, ${this.embeddingDim}, mode='${this.mode}')`;
  }
}

// ── Moving elements, and dropout by channel ────────────────────────────
//
// All eight layers call exactly one tensor method. What differs is the arguments passed
// and the characters printed.

export class PixelShuffle extends Module {
  // **`describe` printed `upscale_factor` while the constructor took `factor`.** The
  // printed name was torch's and the callable one was not, which is the core's
  // `_Rearrange` defect in this file: a repr you cannot feed back in. Positional
  // construction hid it here — JavaScript never says the name out loud.
  constructor(readonly upscaleFactor: number) { super(); }
  override forward(x: Tensor): Tensor { return x.pixelShuffle(this.upscaleFactor); }
  override describe(): string {
    return `PixelShuffle(upscale_factor=${this.upscaleFactor})`;
  }
}

export class PixelUnshuffle extends Module {
  constructor(readonly downscaleFactor: number) { super(); }
  override forward(x: Tensor): Tensor { return x.pixelUnshuffle(this.downscaleFactor); }
  override describe(): string {
    return `PixelUnshuffle(downscale_factor=${this.downscaleFactor})`;
  }
}

export class ChannelShuffle extends Module {
  constructor(readonly groups: number) { super(); }
  override forward(x: Tensor): Tensor { return x.channelShuffle(this.groups); }
  override describe(): string { return `ChannelShuffle(groups=${this.groups})`; }
}

/**
 * The root of the drop-by-channel family. **It prints `inplace` always** — torch does,
 * unlike `RReLU` and `Threshold` above, which print it only when true.
 *
 * ## torch's two alpha layers accept the flag and drop it
 *
 * This is not a simplification on our side. Read torch's own `forward`:
 *
 * ```python
 * class Dropout2d:            return F.dropout2d(input, self.p, self.training, self.inplace)
 * class AlphaDropout:         return F.alpha_dropout(input, self.p, self.training)
 * class FeatureAlphaDropout:  return F.feature_alpha_dropout(input, self.p, self.training)
 * ```
 *
 * The functions honour `inplace`; the two alpha *layers* never hand it over. Measured
 * on real torch across four ranks: `nn.AlphaDropout(0.5, inplace=True)(x)` returns a
 * new tensor and leaves `x` untouched, while `F.alpha_dropout(x, inplace=True)` returns
 * `x` itself with the buffer moved.
 *
 * **So `honours` is per-class, and the two that do not still take the seat.** Honouring
 * it here would be the more sensible behaviour and the wrong port: code that works
 * against torch may read `x` after the call, and torch guarantees it survives. Being
 * better than the thing you are imitating is a way of being different from it.
 */
class FeatureDropoutBase extends Module {
  constructor(
    readonly label: string,
    readonly p = 0.5,
    private readonly alpha = false,
    private readonly perChannel = true,
    private readonly inplace = false,
    /** Whether torch's layer actually forwards the flag. False for the alpha pair. */
    private readonly honours = true,
  ) { super(); }

  override forward(x: Tensor): Tensor {
    const out = this.alpha
      ? x.alphaDropout(this.p, this.training, this.perChannel)
      : x.featureDropout(this.p, this.training);
    return this.inplace && this.honours ? writeBack(x, out) : out;
  }

  override describe(): string {
    return `${this.label}(p=${this.p}, inplace=${this.inplace ? "True" : "False"})`;
  }
}

export class Dropout1d extends FeatureDropoutBase {
  constructor(p?: number, inplace = false) { super("Dropout1d", p, false, true, inplace); }
}
export class Dropout2d extends FeatureDropoutBase {
  constructor(p?: number, inplace = false) { super("Dropout2d", p, false, true, inplace); }
}
export class Dropout3d extends FeatureDropoutBase {
  constructor(p?: number, inplace = false) { super("Dropout3d", p, false, true, inplace); }
}
export class AlphaDropout extends FeatureDropoutBase {
  constructor(p?: number, inplace = false) {
    super("AlphaDropout", p, true, false, inplace, false);
  }
}
export class FeatureAlphaDropout extends FeatureDropoutBase {
  constructor(p?: number, inplace = false) {
    super("FeatureAlphaDropout", p, true, true, inplace, false);
  }
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
  /** **`bias` reaches the `Linear` it builds** — the label printed `bias=True`
   * unconditionally, so a `bias=false` layer would have said otherwise. */
  constructor(outFeatures: number, bias = true, device?: null, dtype?: null) {
    refuseDeviceDtype("LazyLinear", device, dtype);
    super(`LazyLinear(in_features=0, out_features=${outFeatures}, `
      + `bias=${bias ? "True" : "False"})`,
      (inF) => new Linear(inF, outFeatures, bias), lastAxis);
  }
}

// Twelve, each with a name. **Not written as a factory** — exporting an anonymous
// class makes TypeScript refuse on the grounds of `Module`'s private members. The
// padding layers met the same wall, and three lines each is cheaper than breaking the
// inheritance to get around it.
/**
 * **One base per shape, because the two lists are different lists.** They shared one
 * before, with a `transpose` flag choosing the layer, and that is what let a single
 * five-long tuple stand for both — the arrangement the defect lived in.
 */
class LazyConvBase extends LazyModule {
  constructor(spatial: number, outChannels: number, kernelSize: number, stride = 1,
              padding = 0, dilation = 1, groups = 1, bias = true,
              paddingMode: PadMode | "zeros" = "zeros",
              device?: null, dtype?: null) {
    refuseDeviceDtype(`LazyConv${spatial}d`, device, dtype);
    super(`LazyConv${spatial}d`,
      (inChannels) => new ConvND(inChannels, outChannels, kernelSize, spatial, stride, padding, bias,
                          dilation, groups, paddingMode),
      channels);
  }
}

class LazyConvTransposeBase extends LazyModule {
  constructor(spatial: number, outChannels: number, kernelSize: number, stride = 1,
              padding = 0, outputPadding = 0, groups = 1, bias = true,
              dilation = 1, paddingMode: PadMode | "zeros" = "zeros",
              device?: null, dtype?: null) {
    refuseDeviceDtype(`LazyConvTranspose${spatial}d`, device, dtype);
    void paddingMode;
    super(`LazyConvTranspose${spatial}d`,
      (inChannels) => new ConvTransposeND(inChannels, outChannels, kernelSize, spatial, stride, padding,
                                   bias, outputPadding, groups, dilation),
      channels);
  }
}

/**
 * The six lazy convolutions share one argument list, and **the tuple was unlabelled**
 * — `[number, number, number?, number?, boolean?]`. A rest parameter takes its names
 * from the tuple, so with none there the editor offered `a_0`, `a_1`, the generated
 * declarations said `a`, and the signature axis could not line these six up against
 * anything. Labels cost nothing and are the names torch uses.
 */
/**
 * **Two lists, because torch has two.** The plain convolutions read `(…, padding,
 * dilation, groups, bias, padding_mode)` and the transposed ones
 * `(…, padding, output_padding, groups, bias, dilation, padding_mode)` — the fifth
 * seat is a dilation on one side and an output padding on the other, and `dilation`
 * moves to eighth.
 *
 * One shared tuple held `[outChannels, kernelSize, stride, padding, bias]`, which put
 * `bias` in the seat torch gives to `dilation` — `new LazyConv2d(16, 3, 1, 1, false)`
 * set the bias flag here and a dilation there, and both calls succeed. It stayed
 * invisible because the axis could not read a rest parameter's type alias at all;
 * once it could, the row did not move to `shorter` as either session expected. **The
 * wrong verdict was hiding a second defect, not merely a name.**
 */
type ConvArgs = [outChannels: number, kernelSize: number, stride?: number,
                 padding?: number, dilation?: number, groups?: number,
                 bias?: boolean, paddingMode?: PadMode | "zeros",
                 device?: null, dtype?: null];

type ConvTransposeArgs = [outChannels: number, kernelSize: number, stride?: number,
                          padding?: number, outputPadding?: number, groups?: number,
                          bias?: boolean, dilation?: number,
                          paddingMode?: PadMode | "zeros",
                          device?: null, dtype?: null];

export class LazyConv1d extends LazyConvBase {
  constructor(...a: ConvArgs) { super(1, ...a); }
}
export class LazyConv2d extends LazyConvBase {
  constructor(...a: ConvArgs) { super(2, ...a); }
}
export class LazyConv3d extends LazyConvBase {
  constructor(...a: ConvArgs) { super(3, ...a); }
}
export class LazyConvTranspose1d extends LazyConvTransposeBase {
  constructor(...a: ConvTransposeArgs) { super(1, ...a); }
}
export class LazyConvTranspose2d extends LazyConvTransposeBase {
  constructor(...a: ConvTransposeArgs) { super(2, ...a); }
}
export class LazyConvTranspose3d extends LazyConvTransposeBase {
  constructor(...a: ConvTransposeArgs) { super(3, ...a); }
}

/**
 * **A lazy layer takes its target's arguments, minus the one it infers.** These
 * took `(eps, m)` while the eager layers took five, so `LazyBatchNorm2d(1e-5, 0.1,
 * false)` — a layer built with no affine parameters — was a type error, and the
 * signature axis read six rows as unalignable rather than as short.
 *
 * The core derives the list from the target automatically; here it is written out,
 * so the two have to be kept level by hand. That is worth one comment: **the six
 * names below are the only place in this file where a signature is a copy of
 * another signature**, and `tests/ts_signatures.py` is what notices when the copy
 * stops matching.
 */
class LazyNormBase extends LazyModule {
  /**
   * **The instance branch took four arguments and handed over one.**
   * `new InstanceNormND(c, eps)` dropped `momentum`, `affine`, `trackRunningStats`
   * and `bias` on the floor, so `LazyInstanceNorm2d(1e-5, 0.1, true)` built a layer
   * with no parameters at all and said nothing. The batch branch beside it passed all
   * six from the start, which is what made the gap invisible: the signature next door
   * looked like the proof.
   *
   * torch defaults `affine` and `trackRunningStats` **the other way round** between
   * the two — true on batch, false on instance — so the arguments cannot be forwarded
   * with one set of defaults. They come in as `undefined` and each branch fills its
   * own.
   */
  constructor(kind: "batch" | "instance", spatial: number, eps = 1e-5,
              momentum = 0.1, affine?: boolean, trackRunningStats?: boolean,
              bias = true) {
    const batch = kind === "batch";
    super(`Lazy${batch ? "BatchNorm" : "InstanceNorm"}${spatial}d`,
      (c) => (batch
        ? new BatchNormND(c, eps, momentum, affine ?? true,
                          trackRunningStats ?? true, undefined, undefined, bias)
        : new InstanceNormND(c, eps, momentum, affine ?? false,
                             trackRunningStats ?? false, undefined, undefined, bias)),
      channels);
  }
}

export class LazyBatchNorm1d extends LazyNormBase {
  constructor(eps?: number, momentum?: number, affine?: boolean,
              trackRunningStats?: boolean, device?: null, dtype?: null,
              bias?: boolean) {
    refuseDeviceDtype("LazyBatchNorm1d", device, dtype);
    super("batch", 1, eps, momentum, affine, trackRunningStats, bias);
  }
}
export class LazyBatchNorm2d extends LazyNormBase {
  constructor(eps?: number, momentum?: number, affine?: boolean,
              trackRunningStats?: boolean, device?: null, dtype?: null,
              bias?: boolean) {
    refuseDeviceDtype("LazyBatchNorm2d", device, dtype);
    super("batch", 2, eps, momentum, affine, trackRunningStats, bias);
  }
}
export class LazyBatchNorm3d extends LazyNormBase {
  constructor(eps?: number, momentum?: number, affine?: boolean,
              trackRunningStats?: boolean, device?: null, dtype?: null,
              bias?: boolean) {
    refuseDeviceDtype("LazyBatchNorm3d", device, dtype);
    super("batch", 3, eps, momentum, affine, trackRunningStats, bias);
  }
}
export class LazyInstanceNorm1d extends LazyNormBase {
  constructor(eps?: number, momentum?: number, affine?: boolean,
              trackRunningStats?: boolean, device?: null, dtype?: null,
              bias?: boolean) {
    refuseDeviceDtype("LazyInstanceNorm1d", device, dtype);
    super("instance", 1, eps, momentum, affine, trackRunningStats,
          bias);
  }
}
export class LazyInstanceNorm2d extends LazyNormBase {
  constructor(eps?: number, momentum?: number, affine?: boolean,
              trackRunningStats?: boolean, device?: null, dtype?: null,
              bias?: boolean) {
    refuseDeviceDtype("LazyInstanceNorm2d", device, dtype);
    super("instance", 2, eps, momentum, affine, trackRunningStats,
          bias);
  }
}
export class LazyInstanceNorm3d extends LazyNormBase {
  constructor(eps?: number, momentum?: number, affine?: boolean,
              trackRunningStats?: boolean, device?: null, dtype?: null,
              bias?: boolean) {
    refuseDeviceDtype("LazyInstanceNorm3d", device, dtype);
    super("instance", 3, eps, momentum, affine, trackRunningStats,
          bias);
  }
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
    device?: null,
    dtype?: null,
  ) {
    refuseDeviceDtype("AdaptiveLogSoftmaxWithLoss", device, dtype);
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
 * One flat run of labels cut into rows by `lengths`.
 *
 * **It sits above the doc comment it belongs to, not below.** Slipped in between
 * `ctcLoss` and its own description, it inherits that description — and the check that
 * holds the Korean translations against their English sources reported `nn/ctcLoss` as
 * a name that had been renamed or deleted, which is what a doc comment silently
 * changing owner looks like from outside.
 */
function cutByLengths(flat: readonly number[],
                      lengths: readonly number[]): readonly number[][] {
  const rows: number[][] = [];
  let at = 0;
  for (const n of lengths) {
    rows.push([...flat.slice(at, at + n)]);
    at += n;
  }
  return rows;
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
  targets: readonly (readonly number[])[] | readonly number[],
  inputLengths: readonly number[],
  targetLengths: readonly number[],
  blank = 0,
  reduction: Reduction = "mean",
  zeroInfinity = false,
): Tensor {
  // **torch takes the targets two ways** — a padded `(N, S)` table, and one flat run cut
  // by `targetLengths`. The second is what a real loader produces, since padding a batch
  // to its longest label is work nobody does for a loss that already takes the lengths.
  // A caller who hands over the flat form and gets the padded reading back is comparing
  // the wrong letters and still gets a number.
  const rows: readonly (readonly number[])[] = typeof targets[0] === "number"
    ? cutByLengths(targets as readonly number[], targetLengths)
    : targets as readonly (readonly number[])[];
  const parts: Tensor[] = [];
  const divisors: number[] = [];
  for (let i = 0; i < rows.length; i++) {
    const labels = (rows[i] ?? []).slice(0, targetLengths[i] ?? 0);
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
  readonly reduction: Reduction;
  constructor(sizeAverage: boolean | null = null, reduce: boolean | null = null,
              reduction: Reduction = "mean") {
    this.reduction = legacyReduction(sizeAverage, reduce, reduction);
  }

  forward(x: Tensor, target: Tensor): Tensor {
    return x.mseLoss(target, this.reduction);
  }

  call(x: Tensor, target: Tensor): Tensor {
    return this.forward(x, target);
  }

  describe(): string { return "MSELoss()"; }
}

export class L1Loss {
  readonly reduction: Reduction;
  constructor(sizeAverage: boolean | null = null, reduce: boolean | null = null,
              reduction: Reduction = "mean") {
    this.reduction = legacyReduction(sizeAverage, reduce, reduction);
  }

  forward(x: Tensor, target: Tensor): Tensor {
    return x.l1Loss(target, this.reduction);
  }

  call(x: Tensor, target: Tensor): Tensor {
    return this.forward(x, target);
  }

  describe(): string { return "L1Loss()"; }
}

export class SmoothL1Loss {
  readonly reduction: Reduction;
  constructor(sizeAverage: boolean | null = null, reduce: boolean | null = null,
              reduction: Reduction = "mean", readonly beta = 1.0) {
    this.reduction = legacyReduction(sizeAverage, reduce, reduction);
  }

  forward(x: Tensor, target: Tensor): Tensor {
    return x.smoothL1Loss(target, this.beta, this.reduction);
  }

  call(x: Tensor, target: Tensor): Tensor {
    return this.forward(x, target);
  }

  describe(): string { return "SmoothL1Loss()"; }
}

/**
 * torch's deprecated `sizeAverage`/`reduce`, folded into a `reduction`.
 *
 * They were absent here, and absent is not neutral: every argument after them sat one
 * or two seats early, so the same positional call meant different things on the two
 * sides.
 *
 *     new nn.L1Loss("sum")   borch.ts → reduction = "sum"
 *                            the core → sizeAverage = "sum", which is truthy, so the
 *                                       legacy path folds it to the *mean*
 *
 * **The pair beats `reduction` when either is given**, which is the opposite of what
 * a reader expects from a deprecated argument and is what torch does: `reduce=false`
 * gives `none`, else `sizeAverage=false` gives `sum`, else `mean`, with `null`
 * reading as `true` on both. Written the expected way round, a caller passing all
 * three gets an answer torch does not produce.
 *
 * The message is torch's own wording so that a caller who hits it can search for the
 * same sentence in torch's issues.
 */
export function legacyReduction<R extends string>(
  sizeAverage: boolean | null,
  reduce: boolean | null,
  reduction: R,
): R {
  if (sizeAverage === null && reduce === null) return reduction;
  const got = (reduce === false ? "none"
    : sizeAverage === false ? "sum"
    : "mean") as R;
  console.warn(
    `size_average and reduce args will be deprecated, `
    + `please use reduction='${got}' instead.`);
  return got;
}

/**
 * Class weights are **refused rather than absent**, at the position torch puts them.
 *
 * The core refuses them for a reason worth repeating here: torch's `mean` divides by
 * the **sum of the weights** rather than by the sample count, so a `weight` accepted
 * and ignored changes the loss value quietly and leads to choosing the wrong learning
 * rate. What it must not do is take the seat of something else —
 * `new CrossEntropyLoss(classWeights)` set the *reduction* to a tensor until the core
 * grew torch's order and the two sides came apart.
 */
export function refuseWeight(layer: string, what: string, weight: unknown): void {
  if (weight !== undefined && weight !== null) {
    throw new Error(
      `${layer}(${what}=…) is not in the browser subset. Use real PyTorch on your ` +
      "own machine; imitating what is missing teaches the wrong thing.");
  }
}

/**
 * `torch.nn.BCELoss` — over **probabilities**, where `BCEWithLogitsLoss` takes
 * logits. It was absent while its logits form was here, so the name axis read the
 * whole feature as missing.
 *
 * **No `posWeight`.** That argument belongs to the logits form alone, and offering
 * it here would be an argument torch does not have — the core says the same at the
 * same place.
 */
export class BCELoss {
  readonly reduction: Reduction;
  constructor(weight?: Tensor, sizeAverage: boolean | null = null,
              reduce: boolean | null = null, reduction: Reduction = "mean") {
    refuseWeight("BCELoss", "weight", weight);
    this.reduction = legacyReduction(sizeAverage, reduce, reduction);
  }

  forward(x: Tensor, target: Tensor): Tensor {
    return x.bce(target, this.reduction);
  }

  call(x: Tensor, target: Tensor): Tensor {
    return this.forward(x, target);
  }

  describe(): string { return "BCELoss()"; }
}

export class BCEWithLogitsLoss {
  readonly reduction: Reduction;
  constructor(
    weight?: Tensor,
    sizeAverage: boolean | null = null,
    reduce: boolean | null = null,
    reduction: Reduction = "mean",
    posWeight?: Tensor,
  ) {
    refuseWeight("BCEWithLogitsLoss", "weight", weight);
    refuseWeight("BCEWithLogitsLoss", "posWeight", posWeight);
    this.reduction = legacyReduction(sizeAverage, reduce, reduction);
  }

  forward(x: Tensor, target: Tensor): Tensor {
    return x.bceWithLogits(target, this.reduction);
  }

  call(x: Tensor, target: Tensor): Tensor {
    return this.forward(x, target);
  }

  describe(): string { return "BCEWithLogitsLoss()"; }
}

export class NLLLoss {
  readonly reduction: Reduction;
  constructor(
    weight: Tensor | undefined = undefined,
    sizeAverage: boolean | null = null,
    readonly ignoreIndex = -100,
    reduce: boolean | null = null,
    reduction: Reduction = "mean",
  ) {
    refuseWeight("NLLLoss", "weight", weight);
    this.reduction = legacyReduction(sizeAverage, reduce, reduction);
  }

  forward(x: Tensor, target: Tensor): Tensor {
    return x.nllLoss(target, this.ignoreIndex, this.reduction);
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
  readonly reduction: Reduction | "batchmean";
  constructor(
    sizeAverage: boolean | null = null,
    reduce: boolean | null = null,
    reduction: Reduction | "batchmean" = "mean",
    readonly logTarget = false,
  ) {
    this.reduction = legacyReduction(sizeAverage, reduce, reduction);
  }

  forward(x: Tensor, target: Tensor): Tensor {
    return x.klDiv(target, this.reduction, this.logTarget);
  }

  call(x: Tensor, target: Tensor): Tensor {
    return this.forward(x, target);
  }

  describe(): string { return "KLDivLoss()"; }
}

export class PoissonNLLLoss {
  readonly reduction: Reduction;
  constructor(
    readonly logInput = true, readonly full = false,
    sizeAverage: boolean | null = null, readonly eps = 1e-8,
    reduce: boolean | null = null, reduction: Reduction = "mean",
  ) {
    this.reduction = legacyReduction(sizeAverage, reduce, reduction);
  }

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
  readonly reduction: Reduction;
  constructor(readonly margin = 0.0, sizeAverage: boolean | null = null,
              reduce: boolean | null = null, reduction: Reduction = "mean") {
    this.reduction = legacyReduction(sizeAverage, reduce, reduction);
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
  readonly reduction: Reduction;
  constructor(readonly margin = 0.0, sizeAverage: boolean | null = null,
              reduce: boolean | null = null, reduction: Reduction = "mean") {
    this.reduction = legacyReduction(sizeAverage, reduce, reduction);
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
  readonly reduction: Reduction;
  constructor(readonly margin = 1.0, sizeAverage: boolean | null = null,
              reduce: boolean | null = null, reduction: Reduction = "mean") {
    this.reduction = legacyReduction(sizeAverage, reduce, reduction);
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
  readonly reduction: Reduction;
  constructor(sizeAverage: boolean | null = null, reduce: boolean | null = null,
              reduction: Reduction = "mean") {
    this.reduction = legacyReduction(sizeAverage, reduce, reduction);
  }

  forward(x: Tensor, target: Tensor): Tensor {
    return x.softMarginLoss(target, this.reduction);
  }

  call(x: Tensor, target: Tensor): Tensor {
    return this.forward(x, target);
  }

  describe(): string { return "SoftMarginLoss()"; }
}

export class TripletMarginLoss {
  readonly reduction: Reduction;
  constructor(
    readonly margin = 1.0, readonly p = 2.0, readonly eps = 1e-6,
    readonly swap = false, sizeAverage: boolean | null = null,
    reduce: boolean | null = null, reduction: Reduction = "mean",
  ) {
    this.reduction = legacyReduction(sizeAverage, reduce, reduction);
  }

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
  readonly reduction: Reduction;
  constructor(
    readonly weight?: Tensor,
    sizeAverage: boolean | null = null,
    reduce: boolean | null = null,
    reduction: Reduction = "mean",
  ) {
    this.reduction = legacyReduction(sizeAverage, reduce, reduction);
  }

  forward(x: Tensor, target: Tensor): Tensor {
    return x.multilabelSoftMarginLoss(target, this.weight, this.reduction);
  }

  call(x: Tensor, target: Tensor): Tensor {
    return this.forward(x, target);
  }

  describe(): string { return "MultiLabelSoftMarginLoss()"; }
}

export class MultiMarginLoss {
  readonly reduction: Reduction;
  constructor(
    readonly p = 1, readonly margin = 1.0, readonly weight: Tensor | null = null,
    sizeAverage: boolean | null = null, reduce: boolean | null = null,
    reduction: Reduction = "mean",
  ) {
    this.reduction = legacyReduction(sizeAverage, reduce, reduction);
  }

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
  readonly reduction: Reduction;
  constructor(sizeAverage: boolean | null = null, reduce: boolean | null = null,
              reduction: Reduction = "mean") {
    this.reduction = legacyReduction(sizeAverage, reduce, reduction);
  }

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
    return `${this.label}(padding=${pairs}, value=${pyFloat(this.value)})`;
  }
}

// Fifteen, each with a name. **Not written as a factory** — exporting an anonymous
// class makes TypeScript refuse on the grounds of `Module`'s private members, and three
// lines each is cheaper than breaking the inheritance to open that up.
export class ReflectionPad1d extends PadNd {
  constructor(padding: number | readonly number[]) { super("ReflectionPad1d", padding, "reflect", 1); }
}
export class ReflectionPad2d extends PadNd {
  constructor(padding: number | readonly number[]) { super("ReflectionPad2d", padding, "reflect", 2); }
}
export class ReflectionPad3d extends PadNd {
  constructor(padding: number | readonly number[]) { super("ReflectionPad3d", padding, "reflect", 3); }
}
export class ReplicationPad1d extends PadNd {
  constructor(padding: number | readonly number[]) {
    super("ReplicationPad1d", padding, "replicate", 1);
  }
}
export class ReplicationPad2d extends PadNd {
  constructor(padding: number | readonly number[]) {
    super("ReplicationPad2d", padding, "replicate", 2);
  }
}
export class ReplicationPad3d extends PadNd {
  constructor(padding: number | readonly number[]) {
    super("ReplicationPad3d", padding, "replicate", 3);
  }
}
export class CircularPad1d extends PadNd {
  constructor(padding: number | readonly number[]) { super("CircularPad1d", padding, "circular", 1); }
}
export class CircularPad2d extends PadNd {
  constructor(padding: number | readonly number[]) { super("CircularPad2d", padding, "circular", 2); }
}
export class CircularPad3d extends PadNd {
  constructor(padding: number | readonly number[]) { super("CircularPad3d", padding, "circular", 3); }
}
export class ZeroPad1d extends PadNd {
  constructor(padding: number | readonly number[]) { super("ZeroPad1d", padding, "constant", 1); }
}
export class ZeroPad2d extends PadNd {
  constructor(padding: number | readonly number[]) { super("ZeroPad2d", padding, "constant", 2); }
}
export class ZeroPad3d extends PadNd {
  constructor(padding: number | readonly number[]) { super("ZeroPad3d", padding, "constant", 3); }
}
export class ConstantPad1d extends PadNd {
  constructor(padding: number | readonly number[], value = 0) {
    super("ConstantPad1d", padding, "constant", 1, value);
  }
}
export class ConstantPad2d extends PadNd {
  constructor(padding: number | readonly number[], value = 0) {
    super("ConstantPad2d", padding, "constant", 2, value);
  }
}
export class ConstantPad3d extends PadNd {
  constructor(padding: number | readonly number[], value = 0) {
    super("ConstantPad3d", padding, "constant", 3, value);
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
  input: Tensor,
  grid: Tensor,
  mode: "bilinear" | "nearest" = "bilinear",
  paddingMode: "zeros" | "border" | "reflection" = "zeros",
  alignCorners = false,
): Tensor {
  const N = input.shape[0] ?? 1;
  const C = input.shape[1] ?? 1;
  const H = input.shape[2] ?? 1;
  const W = input.shape[3] ?? 1;
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
  const source = input.reshape([input.size]);

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
  input: Tensor,
  runningMean: Tensor | null,
  runningVar: Tensor | null,
  weight: Tensor | null,
  bias: Tensor | null,
  training = false,
  momentum = 0.1,
  eps = 1e-5,
): Tensor {
  const channels = input.shape[1] ?? 1;
  const spatial = input.shape.length - 2;
  const shape = [1, channels, ...new Array<number>(spatial).fill(1)];
  const w = weight ?? Tensor.ones([channels]);
  const b = bias ?? Tensor.zeros([channels]);
  if (!training) {
    if (!runningMean || !runningVar) {
      throw new Error("batchNorm: eval mode needs running statistics");
    }
    const centered = input.sub(runningMean.reshape(shape));
    const scaled = centered.div(
      runningVar.reshape(shape).binary("add", Tensor.full([], eps)).sqrt());
    return scaled.mul(w.reshape(shape)).add(b.reshape(shape));
  }
  // **This goes through a fused kernel.** The assembled form cost more than twenty
  // dispatches for one layer, and most of the 1,636 in a single ResNet step came from
  // there (measured).
  const { out, mean, variance } = input.batchNormFused(w, b, eps);
  if (runningMean && runningVar) {
    // Both are updated by one kernel. The assembled form was eight dispatches per
    // layer, across twenty layers.
    const count = input.size / channels;
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
/**
 * **torch's five carried, and they split three ways.**
 *
 * `max_norm`/`norm_type` shorten the looked-up rows *in the table itself* — the same
 * side effect `embeddingBag` has, through the same `renormRows`, which is why that
 * function's long note applies here unchanged.
 *
 * `padding_idx` is **gradient-only**: measured on torch, the forward returns that row
 * exactly as it stands and only its gradient becomes zero. So it cannot be done by
 * changing what comes out — the padding row is detached on the way in instead, which
 * leaves the forward identical to the bit and cuts the path back. Left in, a pad token
 * drifts toward whatever the loss wants and the mask stops meaning *ignore this*.
 *
 * `scale_grad_by_freq` and `sparse` are refused, by name, because the core refuses
 * them by name. Left out of the signature all five were surplus arguments and
 * JavaScript discards those — `embedding(idx, w, 0)` returned the un-padded answer
 * under the name of a padded one.
 */
export function embedding(input: Tensor, weight: Tensor,
                          paddingIdx: number | null = null,
                          maxNorm: number | null = null, normType = 2.0,
                          scaleGradByFreq = false, sparse = false): Tensor {
  if (scaleGradByFreq) {
    throw new RuntimeError(
      "embedding(scale_grad_by_freq=true) is not in the browser subset.");
  }
  if (sparse) {
    throw new RuntimeError(
      "embedding(sparse=true) — there is no sparse gradient here is not in the "
      + "browser subset.");
  }
  const dim = weight.shape[1] ?? 1;
  if (maxNorm !== null) renormRows(weight, input, maxNorm, normType);
  const table = paddingIdx === null ? weight : detachRow(weight, paddingIdx);
  const picked = table.indexSelect(0, input.reshape([input.size]));
  return picked.reshape([...input.shape, dim]);
}

/**
 * The same table with one row cut out of the graph — value for value identical,
 * and nothing flows back through that row.
 */
function detachRow(weight: Tensor, row: number): Tensor {
  const rows = weight.shape[0] ?? 1;
  const mask = Tensor.zeros([rows])
    .indexAdd(0, Tensor.from([row], [1], { dtype: "int64" }), Tensor.ones([1]))
    .reshape([rows, 1]);
  const keep = Tensor.full([], 1).sub(mask);
  return weight.mul(keep).add(weight.detach().mul(mask));
}

/**
 * One row per bag. Selecting from the table **and combining** is one
 * function.
 *
 * Given `offsets` it cuts a one-dimensional index run into bags — the place
 * where bag lengths differ. `perSampleWeights` is used by torch only when
 * `mode='sum'`.
 */
/**
 * torch's `max_norm`: the rows that were looked up and are too long are shortened,
 * **in the table itself.**
 *
 * A side effect on a parameter, which is unusual enough to say out loud:
 * `embeddingBag(idx, w, null, 1.0)` leaves `w` changed, permanently. torch does
 * exactly this.
 *
 * **A version that renormalised a copy would never part on the output** — not on the
 * second call, not on the hundredth, because renormalising an already-short row is a
 * no-op. Measured against real torch three calls deep: identical to seven figures,
 * while the tables read `[0, 0.4472, 0.8944]` against `[0, 1, 2]`. They part on the
 * **state**, at once, and on the output only once training steps from the shortened
 * weights.
 *
 * That matters for what a check has to look at. Every instrument here compares a
 * returned value — the golden runs a case, the axes read a signature, parity weighs
 * an output — and **no number of repetitions turns a value comparison into a state
 * comparison.** The check below looks at `weight` after the call, which is the only
 * thing that separates the two.
 *
 * Only the rows `idx` names are touched. A whole-table renormalisation is the
 * obvious shortcut and it is wrong — a row nobody looked up stays long in torch.
 */
function renormRows(weight: Tensor, idx: Tensor, maxNorm: number,
                    normType: number): void {
  const rows = weight.shape[0] ?? 1;
  const flat = idx.reshape([idx.size]);
  const seen = Tensor.zeros([rows])
    .indexAdd(0, flat, Tensor.ones([flat.size]))
    .reshape([rows, 1]);
  const lengths = weight.abs().powScalar(normType).sumDim(1, true)
    .powScalar(1 / normType);
  // `min(1, maxNorm / (length + 1e-7))` where the row was looked up, and 1 everywhere
  // else. The epsilon is torch's own, in `renorm_`; the numpy core carries it too, so
  // leaving it out here made the two sides of this library disagree with each other by
  // about a part in ten million on exactly the rows this function exists to change.
  const wanted = lengths.add(Tensor.full([], 1e-7)).reciprocal()
    .mul(Tensor.full([], maxNorm)).minimum(1);
  const scale = wanted.mul(seen.gt(0).to("float32"))
    .add(seen.eq(0).to("float32"));
  // **The write has to be outside the graph.** `max_norm` shortens rows *in the table*,
  // and the table is a leaf that requires grad, so the in-place guard refuses it —
  // `a leaf Variable that requires grad is being used in an in-place operation`, from a
  // layer the caller only asked to look something up. torch does the same edit inside
  // its own `no_grad`, and the optimizer here already reaches this way; the numpy core
  // sidesteps the question entirely by writing to the raw array.
  //
  // Four binding-golden cases were failing on this and none of them said `max_norm`:
  // the message names the guard, which is correct and one layer below the mistake.
  noGrad(() => weight.copyFrom(weight.mul(scale)));
}

/**
 * One row per bag. Selecting from the table and **combining** is all one function.
 *
 * Given `offsets`, a one-dimensional index run is cut into bags — the shape for
 * when the bags have differing lengths. `perSampleWeights` is used in torch under
 * `mode='sum'` alone.
 *
 * **`mode` sits sixth, where torch has it.** It used to be fourth, so
 * `embeddingBag(idx, w, offsets, "sum")` handed a mode string to `maxNorm` — and
 * `maxNorm` rewrites the table, so the layer would have been rescaling its own
 * embeddings on every forward pass.
 */
export function embeddingBag(
  input: Tensor,
  weight: Tensor,
  offsets: readonly number[] | null = null,
  maxNorm: number | null = null,
  normType = 2,
  scaleGradByFreq = false,
  mode: "sum" | "mean" | "max" = "mean",
  sparse = false,
  perSampleWeights: Tensor | null = null,
  includeLastOffset = false,
  paddingIdx: number | null = null,
): Tensor {
  // **`mode` sits sixth, where torch has it.** It used to be fourth, so
  // `embeddingBag(idx, w, offsets, "sum")` handed a mode string to `maxNorm` — and
  // `maxNorm` rewrites the table, so the layer would have been rescaling its own
  // embeddings on every forward pass. The layer's call moved with it.
  if (scaleGradByFreq) {
    throw new NotImplementedError("embeddingBag(scaleGradByFreq) is not carried across");
  }
  if (sparse) {
    throw new NotImplementedError(
      "embeddingBag(sparse) is not carried across — there is no sparse gradient here");
  }
  if (maxNorm !== null) renormRows(weight, input, maxNorm, normType);
  const dim = weight.shape[1] ?? 1;
  // **`paddingIdx` leaves the bag rather than contributing zero to it.** Under `sum`
  // those are the same thing; under `mean` they are not, because the padded entry
  // has to leave the denominator too.
  const kept = paddingIdx === null
    ? null
    : input.ne(Tensor.full([], paddingIdx)).to("float32");
  const squash = (picked: Tensor, d: number, mask: Tensor | null) => {
    if (mode === "sum") return picked.sumDim(d, false);
    if (mode === "max") return picked.amax(d, false);
    if (mask === null) return picked.mean(d, false);
    return picked.sumDim(d, false)
      .div(mask.sumDim(d, false).maximum(1).reshape([...picked.shape.slice(0, d), 1]));
  };
  if (offsets === null) {
    const bags = input.shape[0] ?? 1;
    const each = input.shape[1] ?? 1;
    let picked = weight.indexSelect(0, input.reshape([bags * each]))
      .reshape([bags, each, dim]);
    if (perSampleWeights) {
      picked = picked.mul(perSampleWeights.reshape([bags, each, 1]));
    }
    if (kept) picked = picked.mul(kept.reshape([bags, each, 1]));
    return squash(picked, 1, kept);
  }
  // **`includeLastOffset` means the last entry closes the final bag** rather than
  // opening a new one, so the bag count is one fewer than the offsets rather than
  // one more than the gaps between them.
  const bounds = includeLastOffset ? [...offsets] : [...offsets, input.size];
  const parts: Tensor[] = [];
  for (let b = 0; b + 1 < bounds.length; b++) {
    const from = bounds[b] ?? 0;
    const len = (bounds[b + 1] ?? input.size) - from;
    let picked = weight.indexSelect(0, input.narrow(0, from, len));
    if (perSampleWeights) {
      picked = picked.mul(perSampleWeights.narrow(0, from, len).reshape([len, 1]));
    }
    let slice: Tensor | null = null;
    if (kept) {
      slice = kept.narrow(0, from, len);
      picked = picked.mul(slice.reshape([len, 1]));
    }
    if (mode === "mean" && slice) {
      parts.push(picked.sumDim(0, false).div(slice.sumDim(0, false).maximum(1)));
    } else {
      parts.push(squash(picked, 0, null));
    }
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
  eps = 1e-10,
  dim = -1,
  noise: Tensor | null = null,
): Tensor {
  // **`eps` is accepted, ignored, and warned about — which is exactly what torch does.**
  //
  // It was removed from this signature once, on the reasoning below: torch deprecated it
  // when its noise moved to an exponential draw needing no floor, so honouring a
  // caller's value would make this the one place the two libraries disagree on the same
  // call. All of that is still true and none of it argued for **dropping the
  // parameter**.
  //
  // Dropping it moved `dim` into `eps`'s position. A caller writing torch's own line —
  // `gumbelSoftmax(x, 1, false, 1e-10, -1)` — then bound `dim = 1e-10` and `noise = -1`,
  // silently, in the one implementation of this library that had rearranged the list.
  // The core and the binding both take `eps` and both warn; borch.ts was the odd one of
  // three, and `ts_signatures.py` is the axis that says so — it reported this as
  // `renamed`, its own docstring's example of the failure a name count cannot see.
  //
  // **An argument torch itself ignores is not a defect; a signature that shifts under
  // the caller is.**
  if (eps !== 1e-10) {
    console.warn("`eps` parameter is deprecated and has no effect.");
  }
  const axis = dim < 0 ? dim + logits.shape.length : dim;
  // Gumbel noise — `-log(-log(u))`. Built from a single uniform draw.
  //
  // **`eps` stays a constant here, and that is torch's answer too.** The binding was
  // taking `eps` and dropping it, which looked like an argument owed a home — so it
  // got one: a fourth parameter, threaded through, with a parity check proving a
  // caller's value changed the answer.
  //
  // Then torch was asked what it does with the same argument. It warns:
  // `eps` parameter is deprecated and has no effect`. Its noise comes from an
  // exponential draw, which needs no floor at all, and the parameter survives only so
  // that old calls keep parsing. Honouring it here would have made `gumbel_softmax`
  // one of the few places where this library and torch give different numbers for the
  // same call — while passing every structural check, because the argument would have
  // been visibly used.
  //
  // **The floor in the draw is this function's own and is not the caller's `eps`.**
  // It keeps `log(0)` out of a uniform draw that can return exactly 0; the parameter
  // above is torch's dead one and never reaches here.
  const floor = 1e-10;
  const g = noise ?? Tensor.uniform(logits.shape)
    .binary("add", Tensor.full([], floor)).log().neg()
    .binary("add", Tensor.full([], floor)).log().neg();
  const soft = logits.add(g).div(Tensor.full([], tau)).softmax(axis);
  if (!hard) return soft;
  const picked = soft.max(axis);
  const onehot = soft.oneHotAlong(picked.indices, axis);
  return onehot.sub(soft.detach()).add(soft);
}

export class BatchNormND extends Module {
  readonly weight: Tensor | null;
  readonly bias: Tensor | null;
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

  /**
   * See `GroupNorm` on `affine` and `bias`.
   *
   * **`numFeatures`, not `channels`, and `bias` behind `device` and `dtype`.** This is
   * the fix `InstanceNormND` took and this class did not, one file apart: torch declares
   * `bias` keyword-only after the pair, so a sixth positional argument is `device` there
   * and was `bias` here — **a shift, not a short tail.** The rename goes with it, because
   * torch and the core both call the first argument `num_features` and its sibling
   * already did.
   *
   * Two things next to it were already right and read like proof: `InstanceNormND`
   * twenty lines up takes the pair, and `LazyBatchNorm1d` — this layer's own lazy
   * spelling — declares `device` and `dtype` and refuses them. Only the eager batch
   * class was short, and the signature axis is what said so.
   */
  constructor(
    readonly numFeatures: number,
    private readonly eps = 1e-5,
    private readonly momentum = 0.1,
    // **Kept because `describe` prints them.** `trackRunningStats=false` is refused
    // below, so the only value it can hold is the one that was honoured — printing it
    // is a statement about the layer rather than about the argument.
    private readonly affine = true,
    private readonly trackRunningStats = true,
    device?: null,
    dtype?: null,
    bias = true,
  ) {
    super();
    refuseDeviceDtype("BatchNorm", device, dtype);
    if (!trackRunningStats) {
      // The forward pass reads the running statistics in eval mode, so accepting
      // this and ignoring it leaves training right and evaluation quietly wrong.
      throw new Error(
        "BatchNorm with trackRunningStats=false is not here yet.");
    }
    this.weight = affine ? Tensor.owned([numFeatures], 1) : null;
    this.bias = affine && bias ? Tensor.owned([numFeatures], 0) : null;
    this.claim(...[this.weight, this.bias].filter((t): t is Tensor => t !== null));
    this.runningMean = Tensor.owned([numFeatures], 0);
    this.runningVar = Tensor.owned([numFeatures], 1);
    // The running statistics take no gradient, and **still have to survive the scope
    // closing.**
    keepAlive(this.runningMean);
    keepAlive(this.runningVar);
  }

  override ownParameters(): Record<string, Tensor> {
    const out: Record<string, Tensor> = {};
    if (this.weight) out["weight"] = this.weight;
    if (this.bias) out["bias"] = this.bias;
    return out;
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

  /**
   * **The three this layer handles never reach the base.**
   *
   * That matters now that `strict` weighs missing keys: the base builds its list
   * from `namedBuffers()`, which holds all three, and it is handed `rest`, which
   * holds none of them. Left to judge on its own it would call every load of
   * this layer incompatible. So the base is asked not to judge, and what it
   * reports is corrected by what was handled here before the verdict.
   */
  override loadStateDict(
    values: Readonly<Record<string, Tensor>>,
    strict = true,
  ): LoadReport {
    const rest: Record<string, Tensor> = {};
    const handled: string[] = [];
    for (const [name, src] of Object.entries(values)) {
      if (name === "running_mean") {
        noGrad(() => this.runningMean.copyFrom(src));
        handled.push(name);
      } else if (name === "running_var") {
        noGrad(() => this.runningVar.copyFrom(src));
        handled.push(name);
      } else if (name === "num_batches_tracked") {
        // The given tensor is not captured as it is — when the scope closes that
        // buffer is recycled.
        const base = Tensor.owned([], 0);
        noGrad(() => base.copyFrom(src));
        keepAlive(base);
        this.trackedBase = base;
        this.numBatchesTracked = 0;
        handled.push(name);
      }
      else rest[name] = src;
    }
    const report = super.loadStateDict(rest, false);
    const missing = report.missing.filter((name) => !handled.includes(name));
    if (strict) refuseIncompatible(missing, report.unexpected);
    return { missing, unexpected: report.unexpected };
  }

  override forward(x: Tensor): Tensor {
    if (this.training) this.numBatchesTracked += 1;
    // **`batchNorm` does the computation.** With the layer and the function each
    // written out, a day comes when they diverge, and the place they diverge is the
    // running statistics — so training is fine and evaluation alone is wrong.
    return batchNorm(x, this.runningMean, this.runningVar, this.weight,
      this.bias, this.training, this.momentum, this.eps);
  }

  /** torch's `_BatchNorm.extra_repr`, shared by the instance norms next door. */
  override describe(): string {
    return `${this.constructor.name}(${this.numFeatures}, eps=${pyNumber(this.eps)}, `
      + `momentum=${this.momentum}, affine=${this.affine ? "True" : "False"}, `
      + `bias=${this.bias !== null ? "True" : "False"}, `
      + `track_running_stats=${this.trackRunningStats ? "True" : "False"})`;
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

/**
 * `torch.nn.BatchNorm1d`. The same inheritance as `BatchNorm2d`, and it was
 * **the one of the three that was never written** — see `InstanceNorm1d` for
 * how four absent names sat inside a counter that was pinned as expected.
 */
export class BatchNorm1d extends BatchNormND {}

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

/**
 * What `RNN`, `LSTM` and `GRU` share — the weights and the time loop.
 *
 * **torch calls this `RNNBase`**, and the name was `Recurrent` alone, so the name
 * axis counted the base class as a feature borch.ts does not have while `RNN
 * extends` it two hundred lines down. `RNNBase` is the exported name now and
 * `Recurrent` stays beside it, because `src/rnn.ts` and the golden's own comments
 * refer to it by that name and a rename that reaches into prose is how a comment
 * starts lying.
 */
export class RNNBase extends Module {
  readonly weightIh: Tensor;
  readonly weightHh: Tensor;
  readonly biasIh: Tensor;
  readonly biasHh: Tensor;

  /**
   * **`mode` comes first, as it does in torch.** It used to come last, so
   * `new RNNBase(10, 20, "LSTM")` here was `RNNBase("LSTM", 10, 20)` there — and the
   * row said nothing, because a name torch has at one end and we have at the other
   * cannot be lined up, which puts it in the bucket that reports no detail.
   */
  constructor(
    readonly mode: RNNKind,
    inputSize: number,
    readonly hidden: number,
    numLayers = 1,
    bias = true,
    readonly batchFirst = false,
    dropout = 0,
    bidirectional = false,
    projSize = 0,
    device?: null,
    dtype?: null,
  ) {
    super();
    refuseDeviceDtype("RNNBase", device, dtype);
    // **The three after `batchFirst` are a trailing tail, and a tail is not safe
    // here.** Python raises on a surplus positional and JavaScript discards it, so
    // `new RNNBase("LSTM", 2, 4, 1, true, false, 0, true)` built a one-directional net
    // and said nothing about the `true`. Carried and refused for the same reason
    // `numLayers` and `bias` below are: an argument that raises with its own name
    // beats one the caller cannot tell went nowhere.
    //
    // `dropout` is the exception and it is torch's own: at one layer torch warns and
    // ignores it, because the dropout goes *between* layers and there is no between.
    // Refusing it would stop a line torch accepts.
    if (dropout !== 0) {
      console.warn(
        "dropout option adds dropout after all but last recurrent layer, so "
        + `non-zero dropout expects num_layers greater than 1, but got dropout=${dropout} `
        + "and num_layers=1");
    }
    if (bidirectional) {
      throw new NotImplementedError(
        "RNNBase(bidirectional=true) — a second set of weights and a reversed pass, "
        + "neither of which is here");
    }
    if (projSize !== 0) {
      throw new NotImplementedError(
        `RNNBase(projSize=${projSize}) — the LSTM projection is not carried across`);
    }
    // **The seats between `hidden` and `batchFirst` exist so that `batchFirst` sits
    // where torch has it.** Left out, a line copied from torch positionally puts
    // `numLayers` into `batchFirst` and the net silently reads its axes the wrong way
    // round — which is the defect `EmbeddingBag`'s `mode` comment, forty lines up,
    // records having actually shipped.
    //
    // Accepted and refused rather than accepted and ignored. This base is one layer
    // and always biased; taking the argument and dropping it is the shape the `**kw`
    // sweep spent a day removing.
    if (numLayers !== 1) {
      throw new NotImplementedError(
        `RNNBase(numLayers=${numLayers}) — this side stacks one layer`);
    }
    if (!bias) {
      throw new NotImplementedError("RNNBase(bias=false) is not carried across");
    }
    const gates = mode === "LSTM" ? 4 : mode === "GRU" ? 3 : 1;
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
    // **`batchFirst` is a turn on the way in and a turn on the way out**, and nothing
    // else — the loop is time-first either way. Turning only on the way in gives an
    // answer of the right rank carrying the wrong layout, and `(2, 5, 4)` where the
    // answer is `(5, 2, 4)` cannot even be subtracted, which is what the golden's case
    // relies on.
    //
    // `hidden` and `cell` are **not** turned. torch leaves those `(layers, batch, H)`
    // whatever `batch_first` says, and turning them here would be the tidier-looking
    // choice that diverges.
    //
    // **Nothing in the golden checks that.** The three `batch_first` cases compare
    // `output` alone, and the last-state cases run with the flag off — so a version
    // that turned `hidden` only when `batchFirst` is set would pass every case here.
    // Read off torch's documented shape rather than measured, and said so, because a
    // sentence that sounds measured is the thing this file keeps having to correct.
    const src = this.batchFirst ? x.swapaxes(0, 1) : x;
    const [steps = 0, batch = 0] = src.shape;
    const H = this.hidden;
    let h = Tensor.zeros([batch, H]);
    let c = Tensor.zeros([batch, H]);
    const outs: Tensor[] = [];
    for (let t = 0; t < steps; t++) {
      const xt = src.select(0, t);
      const gi = xt.linear(this.weightIh).add(this.biasIh);
      const gh = h.linear(this.weightHh).add(this.biasHh);
      if (this.mode === "RNN") {
        h = gi.add(gh).unary("tanh");
      } else if (this.mode === "LSTM") {
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
    const stacked = Tensor.stack(outs, 0);
    return {
      output: this.batchFirst ? stacked.swapaxes(0, 1) : stacked,
      hidden: h.reshape([1, batch, H]),
      cell: c.reshape([1, batch, H]),
    };
  }
}

/* ── The names with the mode fixed ──────────────────────────────────────
 *
 * `Recurrent` takes the mode **as an argument** and torch splits it out into the name.
 * So these three fix one argument, and that is exactly the name torch uses.
 *
 * The values are already proven — the six golden cases
 * `seq::{RNN,LSTM,GRU}/{출력,마지막상태}` were running on `Recurrent`. What was missing
 * was the name rather than the computation, and **a recurrent-network textbook opens
 * with `nn.LSTM(...)`**, so it stopped people on the first line.
 *
 * **It still does not take all of torch's arguments** — `bidirectional`, `dropout` and
 * `projSize` are absent, and the base here is one layer. Accepting an argument and not
 * using it is a lie, which is the line the core holds the same way at `InstanceNorm`'s
 * `track_running_stats`.
 *
 * **`batchFirst` used to be absent under that same rule, and that was one step too
 * far.** It is not an argument this side cannot honour — it is a turn on the way in and
 * a turn on the way out — so leaving it out was refusing to do something possible. And
 * leaving it out is not neutral: `numLayers` and `bias` sit between it and `hidden` in
 * torch, so a line copied positionally lands `numLayers` in `batchFirst` and the net
 * reads its axes the wrong way round without saying so.
 *
 * So the three seats are here, at torch's indices, with the two that cannot be honoured
 * **refused rather than ignored**. `RNN` carries a fourth, `nonlinearity`, which is why
 * its `batchFirst` is at a different index from `LSTM`'s.
 */

/**
 * The former name of `RNNBase`. Kept because `src/rnn.ts` and the golden's own
 * comments name it in prose, and a rename that reaches into prose is how a comment
 * starts lying. **Both the value and the type**, because it was used as a type.
 */
export const Recurrent = RNNBase;
export type Recurrent = RNNBase;

/**
 * `torch.nn.RNN` — one layer.
 *
 * **`nonlinearity` sits fourth and `batchFirst` sixth, which is torch's order and not
 * `LSTM`'s.** This is the one of the three that takes an extra argument, so the same
 * flag lives at a different index here than next door; writing all three alike would
 * put `batchFirst` one seat early on exactly this class.
 */
export class RNN extends RNNBase {
  constructor(inputSize: number, hidden: number, numLayers = 1,
              nonlinearity: "tanh" | "relu" = "tanh", bias = true,
              batchFirst = false) {
    if (nonlinearity !== "tanh") {
      throw new NotImplementedError(
        `RNN(nonlinearity=${nonlinearity}) — this side computes tanh`);
    }
    super("RNN", inputSize, hidden, numLayers, bias, batchFirst);
  }
}

/**
 * `torch.nn.LSTM` — one layer. It carries two states, so `cell` comes back alongside
 * `hidden`.
 */
export class LSTM extends RNNBase {
  constructor(inputSize: number, hidden: number, numLayers = 1, bias = true,
              batchFirst = false) {
    super("LSTM", inputSize, hidden, numLayers, bias, batchFirst);
  }
}

/** `torch.nn.GRU` — one layer. */
export class GRU extends RNNBase {
  constructor(inputSize: number, hidden: number, numLayers = 1, bias = true,
              batchFirst = false) {
    super("GRU", inputSize, hidden, numLayers, bias, batchFirst);
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
  embedDimToCheck: number | null,
  numHeads: number,
  inProjWeight: Tensor,
  inProjBias: Tensor | null,
  biasK: Tensor | null = null,
  biasV: Tensor | null = null,
  addZeroAttn = false,
  dropoutP = 0.0,
  outProjWeight: Tensor = inProjWeight,
  outProjBias: Tensor | null = null,
  training = true,
  keyPaddingMask: Tensor | null = null,
  needWeights = true,
  attnMask: Tensor | null = null,
  useSeparateProjWeight = false,
  qProjWeight: Tensor | null = null,
  kProjWeight: Tensor | null = null,
  vProjWeight: Tensor | null = null,
  staticK: Tensor | null = null,
  staticV: Tensor | null = null,
  averageAttnWeights = true,
  isCausal = false,
): { output: Tensor; weights: Tensor | null } {
  // **torch's twenty-five, in torch's order.** This took thirteen, in an order of its
  // own, so `multiHeadAttentionForward(q, k, v, heads, …)` and torch's
  // `(query, key, value, embed_dim_to_check, num_heads, …)` mean different things from
  // the fourth argument on. Every one of the twelve that were missing sat *between*
  // ones that were present, so each shifted what followed.
  //
  // Six are refused rather than implemented, which is the trade the core makes at the
  // same place: quietly ignoring a branch like `biasK` or `staticK` makes the values
  // plausibly different, and plausible is the one thing a comparison cannot catch.
  for (const [what, given] of [["biasK", biasK], ["biasV", biasV],
                               ["staticK", staticK], ["staticV", staticV],
                               ["qProjWeight", qProjWeight],
                               ["kProjWeight", kProjWeight],
                               ["vProjWeight", vProjWeight]] as const) {
    if (given != null) {
      throw new NotImplementedError(`multiHeadAttentionForward(${what}=…)`);
    }
  }
  for (const [what, on] of [["addZeroAttn", addZeroAttn],
                            ["useSeparateProjWeight", useSeparateProjWeight]] as const) {
    if (on) throw new NotImplementedError(`multiHeadAttentionForward(${what}=true)`);
  }
  const inWeight = inProjWeight;
  const inBias = inProjBias;
  const outWeight = outProjWeight;
  const outBias = outProjBias;
  const averageWeights = averageAttnWeights;
  const L = query.shape[0] ?? 1;
  const N = query.shape[1] ?? 1;
  const E = query.shape[2] ?? 1;
  const S = key.shape[0] ?? 1;
  // **`!= null` on purpose, both spellings of absent.** Python's `None` arrives from
  // the binding as `undefined`, not `null`, so a strict `!== null` made the guard fire
  // on every call that declined it — *"was expecting embedding dimension of
  // undefined"*, three golden cases at once. A parameter that crosses a language
  // boundary has two empties, and the check has to know both.
  if (embedDimToCheck != null && embedDimToCheck !== E) {
    throw new Error(
      `was expecting embedding dimension of ${embedDimToCheck}, but got ${E}`);
  }
  const head = E / numHeads;
  const scale = Tensor.full([], 1 / Math.sqrt(head));
  // `isCausal` is torch's shorthand for the upper-triangular mask, and torch takes
  // the explicit one when both are given.
  if (isCausal && attnMask === null) attnMask = MultiheadAttention.causalMask(L);

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
      // **Dropout on the attention, torch's way round.** The binding took
      // `dropout_p` and `training` and dropped them both, so a layer built to
      // drop half its attention dropped none of it and said nothing — the kind
      // of difference that shows up as a model that trains a little too well.
      //
      // torch drops *after* the softmax and hands back the dropped weights, not
      // the ones before, so `need_weights=True` and the value matmul agree on
      // what was attended to. At `dropoutP = 0` this is the identity, which is
      // why every existing golden case is untouched by it.
      const w = scores.softmax(1).dropout(dropoutP, training);
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
    // torch hands back `None` rather than weights nobody asked for.
    weights: !needWeights ? null
      : averageWeights ? weights.mean(1, false) : weights,
  };
}

export class MultiheadAttention extends Module {
  readonly inWeight: Tensor;
  readonly inBias: Tensor | null;
  readonly outWeight: Tensor;
  readonly outBias: Tensor | null;

  /**
   * torch's eleven, of which three do work here and four are refused by name.
   *
   * **This took `(embedDim, numHeads)`, so seven arguments were missing from the
   * middle and every one of them shifted what followed.** `new MultiheadAttention(64,
   * 8, 0.1)` — torch's own way of writing a dropout of 0.1 — reached nothing at all
   * here, and the core had the same list a shift earlier where `0.1` landed on
   * `bias`. An absent feature beats a wrong answer, and **a wrong position is a wrong
   * answer wearing the shape of a feature.**
   *
   * `dropout`, `bias` and `batchFirst` work. `addBiasKv`, `addZeroAttn` and a
   * `kdim`/`vdim` unlike the embedding stop with their own name — the refusal exists
   * a layer down in `multiHeadAttentionForward` already, and carrying the argument
   * here is what makes it arrive with the right name attached.
   *
   * **`batchFirst` defaults to `false`, which flips what this class used to do.** It
   * read `(batch, len, E)` unconditionally, which is torch's `batch_first=True`, so
   * the default was the option torch does not take.
   */
  constructor(readonly embedDim: number, readonly numHeads: number,
              readonly dropout = 0,
              bias = true,
              addBiasKv = false,
              addZeroAttn = false,
              kdim: number | null = null,
              vdim: number | null = null,
              readonly batchFirst = false,
              device?: null, dtype?: null) {
    refuseDeviceDtype("MultiheadAttention", device, dtype);
    super();
    if (embedDim % numHeads) {
      throw new Error(
        `embed_dim(${embedDim}) is not divisible by num_heads(${numHeads}).`);
    }
    for (const [what, on] of [["addBiasKv", addBiasKv],
                              ["addZeroAttn", addZeroAttn]] as const) {
      if (on) throw new NotImplementedError(`MultiheadAttention(${what}=true)`);
    }
    // torch only takes the separate-projection path when these differ from
    // `embedDim`; the same number is the ordinary layer and asks for nothing.
    for (const [what, given] of [["kdim", kdim], ["vdim", vdim]] as const) {
      if (given !== null && given !== embedDim) {
        throw new NotImplementedError(
          `MultiheadAttention(${what}=${given}) — a key or value width unlike the `
          + "embedding's");
      }
    }
    const bound = 1 / Math.sqrt(Math.max(1, embedDim));
    this.inWeight = uniform([3 * embedDim, embedDim], bound);
    // torch's attention starts the bias at 0 — this is not a place where symmetry
    // needs breaking.
    this.inBias = bias ? Tensor.owned([3 * embedDim], 0) : null;
    this.outWeight = uniform([embedDim, embedDim], bound);
    this.outBias = bias ? Tensor.owned([embedDim], 0) : null;
    this.claim(...[this.inWeight, this.inBias, this.outWeight, this.outBias]
      .filter((t): t is Tensor => t !== null));
  }

  override ownParameters(): Record<string, Tensor> {
    const out: Record<string, Tensor> = {
      in_proj_weight: this.inWeight, "out_proj.weight": this.outWeight,
    };
    if (this.inBias) out.in_proj_bias = this.inBias;
    if (this.outBias) out["out_proj.bias"] = this.outBias;
    return out;
  }

  override forward(x: Tensor): Tensor {
    return this.attend(x, null);
  }

  attend(x: Tensor, mask: Tensor | null): Tensor {
    // **`batchFirst` decides which of the first two axes is which**, and torch's
    // default is length first. Flipped here so the body below stays one shape.
    const src = this.batchFirst ? x : x.swapaxes(0, 1);
    const [batch = 1, len = 1] = src.shape;
    const E = this.embedDim;
    const head = E / this.numHeads;
    const flat = src.reshape([batch * len, E]);
    const inBias = this.inBias;
    const projected = inBias ? flat.linear(this.inWeight).add(inBias)
      : flat.linear(this.inWeight);
    const parts = [0, 1, 2].map((k) => projected.narrow(1, k * E, E));
    const scale = Tensor.full([], 1 / Math.sqrt(head));
    const outs: Tensor[] = [];
    for (let b = 0; b < batch; b++) {
      const perHead: Tensor[] = [];
      for (let h = 0; h < this.numHeads; h++) {
        const take = (t: Tensor | undefined): Tensor => {
          if (!t) throw new Error("attention: the projections are missing");
          return t.reshape([batch, len, E]).select(0, b).narrow(1, h * head, head);
        };
        const q = take(parts[0]);
        const k = take(parts[1]);
        const v = take(parts[2]);
        let scores = q.mm(k.transpose()).binary("mul", scale);
        if (mask) scores = scores.add(mask);
        // torch drops attention weights, not the values — and while training only.
        perHead.push(scores.softmax(1).dropout(this.dropout, this.training).mm(v));
      }
      outs.push(Tensor.cat(perHead, 1));
    }
    const merged = Tensor.stack(outs, 0).reshape([batch * len, E]);
    const outBias = this.outBias;
    const out = outBias ? merged.linear(this.outWeight).add(outBias)
      : merged.linear(this.outWeight);
    const shaped = out.reshape([batch, len, E]);
    return this.batchFirst ? shaped : shaped.swapaxes(0, 1);
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
  /**
   * **`sizeAverage` and `reduce` are not adjacent here.** torch puts `ignoreIndex`
   * between them, and a rule that inserts the pair side by side would be right about
   * twelve of these classes and wrong about this one, `NLLLoss` and `PoissonNLLLoss`.
   * The order comes from torch, one class at a time.
   */
  readonly reduction: Reduction;
  constructor(
    weight: Tensor | undefined = undefined,
    sizeAverage: boolean | null = null,
    readonly ignoreIndex = -100,
    reduce: boolean | null = null,
    reduction: Reduction = "mean",
    readonly labelSmoothing = 0.0,
  ) {
    refuseWeight("CrossEntropyLoss", "weight", weight);
    this.reduction = legacyReduction(sizeAverage, reduce, reduction);
  }

  forward(logits: Tensor, target: Tensor): Tensor {
    return logits.crossEntropy(target, this.ignoreIndex, this.reduction,
                               this.labelSmoothing);
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

// ── The transformer, the last five names on the name axis ─────────────────
//
// Everything under them was already here: `MultiheadAttention`, `Linear`,
// `LayerNorm`, `Dropout`, `ModuleList`. What was missing is the assembly, and the
// field names below are **the core's field names on purpose** — `self_attn`,
// `linear1`, `norm1` and the rest become `state_dict` keys, and a checkpoint that
// cannot cross between the two implementations is a checkpoint that does not exist.

/**
 * Apply a 2-D-only layer **per position** of an `(N, L, E)` sequence.
 *
 * **borch.ts's `Linear` is 2-D by 2-D** — `mm is 2-D by 2-D: [2,3,4] x [4,8]` is
 * what the feed-forward said the first time these classes ran, and a peer had
 * already found and documented that exact limit from the other direction. The four
 * transformer cases were red on the borch.ts side while the core's were green: the
 * name axis cannot see a rank, and this is what that costs.
 *
 * Flattening is not a workaround here, it is the operation: a feed-forward and an
 * output projection are **position-wise by definition**, so `(N, L, E)` folded to
 * `(N·L, E)` and back is the same arithmetic with the batch spelled out. It is kept
 * local to these classes rather than pushed into `Linear`, because making `Linear`
 * batch is a change to a name a lesson page currently teaches around.
 */
function positionwise(x: Tensor, f: (flat: Tensor) => Tensor): Tensor {
  if (x.shape.length <= 2) return f(x);
  const lead = x.shape.slice(0, -1);
  const feat = x.shape[x.shape.length - 1] ?? 1;
  const out = f(x.reshape([lead.reduce((a, b) => a * b, 1), feat]));
  return out.reshape([...lead, out.shape[out.shape.length - 1] ?? 1]);
}

/** The activations torch names by string. */
function namedActivation(name: string): (x: Tensor) => Tensor {
  if (name === "relu") return (x) => x.relu();
  if (name === "gelu") return (x) => x.gelu();
  throw new Error(`activation must be "relu" or "gelu": ${name}`);
}

/**
 * Attention, then a feed-forward, each with a residual and a normalisation.
 *
 * **`normFirst` moves the normalisation inside the residual**, which is the
 * difference between the 2017 paper and every model since — post-norm adds and then
 * normalises, pre-norm normalises and then adds, and the second is what trains at
 * depth. Both are here because torch has both and the default is torch's (false).
 */
export class TransformerEncoderLayer extends Module {
  readonly self_attn: MultiheadAttention;
  readonly linear1: Linear;
  readonly linear2: Linear;
  readonly norm1: LayerNorm;
  readonly norm2: LayerNorm;
  readonly dropout: Dropout;
  private readonly activation: (x: Tensor) => Tensor;
  /** Its own arguments, so a stack can build more of it — see `TransformerEncoder`. */
  readonly config: readonly [number, number, number, number, string, number, boolean, boolean, boolean];

  /**
   * **torch's order**, which is not the order this was first written in:
   * `(…, activation, layerNormEps, batchFirst, normFirst, bias)`. With `batchFirst`
   * in the sixth seat, `new TransformerEncoderLayer(4, 2, 8, 0.1, "relu", true)` put
   * `true` into torch's epsilon and the layer normalised with eps = 1. Nothing
   * raises — the shapes are right and the loss goes down.
   */
  constructor(
    dModel: number, nhead: number, dimFeedforward = 2048, dropout = 0.1,
    activation = "relu", layerNormEps = 1e-5, batchFirst = false,
    normFirst = false, bias = true,
    device?: null, dtype?: null,
  ) {
    refuseDeviceDtype("TransformerEncoderLayer", device, dtype);
    super();
    // **`batchFirst` was taken and thrown away** — `void batchFirst;` sat here, in the
    // class whose own comment above is about an argument landing in the wrong seat.
    // It reaches the attention now, as it does in the core, which is the only place
    // the flag means anything.
    this.self_attn = new MultiheadAttention(dModel, nhead, 0, true, false, false,
                                            null, null, batchFirst);
    this.linear1 = new Linear(dModel, dimFeedforward, bias);
    this.linear2 = new Linear(dimFeedforward, dModel, bias);
    this.norm1 = new LayerNorm(dModel, layerNormEps);
    this.norm2 = new LayerNorm(dModel, layerNormEps);
    this.dropout = new Dropout(dropout);
    this.activation = namedActivation(activation);
    this.normFirst = normFirst;
    this.config = [dModel, nhead, dimFeedforward, dropout, activation, layerNormEps,
                   batchFirst, normFirst, bias];
  }

  readonly normFirst: boolean;

  private sa(x: Tensor): Tensor {
    return this.dropout.call(this.self_attn.call(x));
  }

  private ff(x: Tensor): Tensor {
    return this.dropout.call(positionwise(x, (v) =>
      this.linear2.call(this.dropout.call(this.activation(this.linear1.call(v))))));
  }

  override forward(src: Tensor): Tensor {
    if (this.normFirst) {
      const a = src.add(this.sa(this.norm1.call(src)));
      return a.add(this.ff(this.norm2.call(a)));
    }
    const a = this.norm1.call(src.add(this.sa(src)));
    return this.norm2.call(a.add(this.ff(a)));
  }
}

/**
 * The same layer stacked, named `layers.N.…` as in torch.
 *
 * **torch deep-copies the prototype and TypeScript cannot.** So the layer carries
 * its own arguments and the stack builds `numLayers` fresh ones from them — the
 * signature stays torch's, and the alternative (holding one object N times) would
 * share weights between every layer while looking exactly right.
 *
 * `enableNestedTensor` and `maskCheck` are accepted and change nothing, as in the
 * core: the first asks for a packed representation that does not exist here, and the
 * second guards a fast path that is not taken.
 */
export class TransformerEncoder extends Module {
  readonly layers: ModuleList;

  constructor(
    encoderLayer: TransformerEncoderLayer, numLayers: number,
    readonly norm: LayerNorm | null = null,
    enableNestedTensor = true, maskCheck = true,
  ) {
    super();
    void enableNestedTensor;
    void maskCheck;
    this.layers = new ModuleList(
      Array.from({ length: numLayers },
        () => new TransformerEncoderLayer(...encoderLayer.config)));
  }

  override forward(src: Tensor): Tensor {
    let x = src;
    for (const layer of this.layers.children()) x = layer.call(x);
    return this.norm === null ? x : this.norm.call(x);
  }
}

/**
 * Self-attention, then attention over the encoder's output, then a feed-forward.
 * The extra `norm3` is what makes it a decoder layer rather than an encoder one.
 */
export class TransformerDecoderLayer extends Module {
  readonly self_attn: MultiheadAttention;
  readonly multihead_attn: MultiheadAttention;
  readonly linear1: Linear;
  readonly linear2: Linear;
  readonly norm1: LayerNorm;
  readonly norm2: LayerNorm;
  readonly norm3: LayerNorm;
  readonly dropout: Dropout;
  readonly normFirst: boolean;
  private readonly activation: (x: Tensor) => Tensor;
  readonly config: readonly [number, number, number, number, string, number, boolean, boolean, boolean];

  /** torch's order — see `TransformerEncoderLayer`. */
  constructor(
    dModel: number, nhead: number, dimFeedforward = 2048, dropout = 0.1,
    activation = "relu", layerNormEps = 1e-5, batchFirst = false,
    normFirst = false, bias = true,
    device?: null, dtype?: null,
  ) {
    refuseDeviceDtype("TransformerDecoderLayer", device, dtype);
    super();
    // **`batchFirst` was taken and thrown away** — `void batchFirst;` sat here, in the
    // class whose own comment above is about an argument landing in the wrong seat.
    // It reaches the attention now, as it does in the core, which is the only place
    // the flag means anything.
    this.self_attn = new MultiheadAttention(dModel, nhead, 0, true, false, false,
                                            null, null, batchFirst);
    // The cross-attention goes through `multiHeadAttentionForward`, which takes the
    // length first regardless — but the flag belongs on the layer either way, so the
    // two attentions inside one decoder layer do not disagree about their input.
    this.multihead_attn = new MultiheadAttention(dModel, nhead, 0, true, false,
                                                 false, null, null, batchFirst);
    this.linear1 = new Linear(dModel, dimFeedforward, bias);
    this.linear2 = new Linear(dimFeedforward, dModel, bias);
    this.norm1 = new LayerNorm(dModel, layerNormEps);
    this.norm2 = new LayerNorm(dModel, layerNormEps);
    this.norm3 = new LayerNorm(dModel, layerNormEps);
    this.dropout = new Dropout(dropout);
    this.activation = namedActivation(activation);
    this.normFirst = normFirst;
    this.config = [dModel, nhead, dimFeedforward, dropout, activation, layerNormEps,
                   batchFirst, normFirst, bias];
  }

  private ff(x: Tensor): Tensor {
    return this.dropout.call(positionwise(x, (v) =>
      this.linear2.call(this.dropout.call(this.activation(this.linear1.call(v))))));
  }

  /**
   * **Attention over `memory`, not a second self-attention.** The first version of
   * this wrote `multihead_attn.attend(x, null)` and then added `memory * 0` so that
   * the argument would look used. That is the defect this repository has a whole
   * check for: an argument accepted, discarded, and made to *appear* consulted —
   * the decoder would have trained, converged to something, and never once read the
   * encoder.
   *
   * `multiHeadAttentionForward` takes query, key and value separately, which is what
   * makes the real thing possible: the query is the decoder's own state and the key
   * and value are the encoder's output.
   */
  private cross(x: Tensor, memory: Tensor): Tensor {
    const a = this.multihead_attn;
    // `null` is `embedDimToCheck` — torch's fourth, an assertion rather than a
    // setting, and the four `null`s after `dropoutP` are `biasK`, `biasV` and the
    // rest, which this call has nothing to say about.
    const { output } = multiHeadAttentionForward(
      x.swapaxes(0, 1), memory.swapaxes(0, 1), memory.swapaxes(0, 1),
      null, a.numHeads, a.inWeight, a.inBias, null, null, false, 0,
      a.outWeight, a.outBias);
    return output.swapaxes(0, 1);
  }

  override forward(tgt: Tensor, memory?: Tensor): Tensor {
    const mem = memory ?? tgt;
    if (this.normFirst) {
      let x = tgt.add(this.dropout.call(this.self_attn.call(this.norm1.call(tgt))));
      x = x.add(this.dropout.call(this.cross(this.norm2.call(x), mem)));
      return x.add(this.ff(this.norm3.call(x)));
    }
    let x = this.norm1.call(tgt.add(this.dropout.call(this.self_attn.call(tgt))));
    x = this.norm2.call(x.add(this.dropout.call(this.cross(x, mem))));
    return this.norm3.call(x.add(this.ff(x)));
  }
}

/** The decoder layer stacked. `layers.N.…`, as the encoder is. */
export class TransformerDecoder extends Module {
  readonly layers: ModuleList;
  /** The same objects the `ModuleList` holds, typed — `children()` gives `Module`. */
  private readonly stack: TransformerDecoderLayer[];

  constructor(
    decoderLayer: TransformerDecoderLayer, numLayers: number,
    readonly norm: LayerNorm | null = null,
  ) {
    super();
    this.stack = Array.from({ length: numLayers },
      () => new TransformerDecoderLayer(...decoderLayer.config));
    this.layers = new ModuleList(this.stack);
  }

  override forward(tgt: Tensor, memory?: Tensor): Tensor {
    let x = tgt;
    for (const layer of this.stack) x = layer.forward(x, memory);
    return this.norm === null ? x : this.norm.call(x);
  }
}

/** The encoder and the decoder together — the whole diagram from the 2017 paper. */
export class Transformer extends Module {
  readonly encoder: TransformerEncoder;
  readonly decoder: TransformerDecoder;

  /**
   * torch's order, with `customEncoder` and `customDecoder` in their seats — those
   * let a caller hand in an assembled stack instead of the one built here, and left
   * out, torch's eighth and ninth positions land on `layerNormEps` and `batchFirst`.
   */
  constructor(
    readonly dModel = 512, readonly nhead = 8, numEncoderLayers = 6,
    numDecoderLayers = 6, dimFeedforward = 2048, dropout = 0.1,
    activation = "relu", customEncoder: TransformerEncoder | null = null,
    customDecoder: TransformerDecoder | null = null, layerNormEps = 1e-5,
    batchFirst = false, normFirst = false, bias = true,
    device?: null, dtype?: null,
  ) {
    refuseDeviceDtype("Transformer", device, dtype);
    super();
    this.encoder = customEncoder ?? new TransformerEncoder(
      new TransformerEncoderLayer(dModel, nhead, dimFeedforward, dropout, activation,
        layerNormEps, batchFirst, normFirst, bias),
      numEncoderLayers, new LayerNorm(dModel, layerNormEps));
    this.decoder = customDecoder ?? new TransformerDecoder(
      new TransformerDecoderLayer(dModel, nhead, dimFeedforward, dropout, activation,
        layerNormEps, batchFirst, normFirst, bias),
      numDecoderLayers, new LayerNorm(dModel, layerNormEps));
  }

  override forward(src: Tensor, tgt?: Tensor): Tensor {
    const memory = this.encoder.call(src);
    return this.decoder.forward(tgt ?? src, memory);
  }

  /**
   * A **float** mask whose upper triangle is −∞. It is *added* to the scores, not
   * multiplied — which is why the masked positions are −∞ and not 0.
   */
  static generateSquareSubsequentMask(size: number): Tensor {
    const row: number[] = [];
    for (let i = 0; i < size; i++) {
      for (let j = 0; j < size; j++) row.push(j > i ? -Infinity : 0);
    }
    return Tensor.from(row, [size, size]);
  }
}
