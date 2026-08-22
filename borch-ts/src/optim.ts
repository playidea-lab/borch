/**
 * Optimizers and learning-rate schedules.
 *
 * ## Parameters are modified in place
 *
 * It does not make a new tensor and swap it in. The handle the model holds
 * and the handle the optimizer sees have to be the same one; swapping
 * splits them and produces a state where **training runs and the parameters
 * do not move.** In-place modification is blocked on a leaf with gradient
 * enabled, so it happens inside `no_grad` — which is exactly what torch's
 * optimizers do.
 *
 * ## State is held as tensors
 *
 * Something like a momentum buffer **must not hold the gradient tensor
 * itself.** When that gradient is replaced on the next step, the buffer
 * points at the wrong thing — a place this repository missed while only
 * ever looking at SGD without momentum. So the buffer starts as its own
 * copy.
 */

import { adamStep, rmspropStep, sgdStep } from "./kernels.js";
import { RuntimeError } from "./errors.js";
import { device, keepAlive, noGrad, Tensor } from "./tensor.js";

/**
 * One bundle of parameters and the hyperparameters attached to it.
 *
 * **There used to be no `params`.** The name is torch's `param_groups`, but
 * it was always a single `[{ lr }]` and every scheduler only ever looked at
 * `[0]` — no per-layer learning rate, no excluding bias and norm from
 * weight decay, while the name said otherwise. torch's shape with nothing
 * inside is worse than nothing: a user calling `paramGroups.push(...)` was
 * quietly ignored.
 */
export interface ParamGroup {
  /**
   * The parameters this group steps.
   */
  params: Tensor[];
  lr: number;
  /**
   * The value a scheduler takes as its baseline. The first scheduler stamps
   * it once.
   */
  initialLr?: number;
  /**
   * For when this group alone uses a different value. Absent, it is the
   * value given when the optimizer was built.
   */
  weightDecay?: number;
}

/**
 * What goes in when a group is made. Leave `lr` out to use the optimizer's
 * default.
 */
export interface ParamGroupInit {
  params: readonly Tensor[];
  lr?: number;
  weightDecay?: number;
}

/**
 * What an optimizer's constructor accepts — a list of tensors or a list of
 * groups.
 */
export type ParamsArg = readonly Tensor[] | readonly ParamGroupInit[];

function isGroups(arg: ParamsArg): arg is readonly ParamGroupInit[] {
  return arg.length > 0 && !(arg[0] instanceof Tensor);
}

export abstract class Optimizer {
  /**
   * torch's shape — this is where a scheduler edits `lr`.
   */
  readonly paramGroups: ParamGroup[] = [];

  /**
   * Every group's parameters, concatenated. The state bank is indexed by
   * this position.
   */
  protected readonly params: Tensor[] = [];

  /** Parameter position → group number. */
  private readonly groupOf: number[] = [];

  /** The group currently being walked while `step` runs. `lr` reads it. */
  private currentGroup = 0;

  /**
   * The banks `state()` created. Adding a group adds a slot to every one of them.
   *
   * The fill value is held with them — `Rprop`'s step size starts at `lr` rather than 0,
   * and a parameter added later has to start from the same place.
   */
  private readonly banks: { slots: Tensor[]; value: number }[] = [];

  constructor(params: ParamsArg, private readonly defaultLr: number) {
    const groups: readonly ParamGroupInit[] = isGroups(params)
      ? params
      : [{ params: params as readonly Tensor[] }];
    for (const g of groups) this.attach(g);
  }

  /** Attaches one group and appends it to the flat list. */
  private attach(init: ParamGroupInit): ParamGroup {
    const index = this.paramGroups.length;
    const group: ParamGroup = {
      params: [...init.params],
      lr: init.lr ?? this.defaultLr,
      ...(init.weightDecay === undefined ? {} : { weightDecay: init.weightDecay }),
    };
    this.paramGroups.push(group);
    for (const p of group.params) {
      this.params.push(p);
      this.groupOf.push(index);
    }
    return group;
  }

  /**
   * Adds a group later. Where `torch.optim.Optimizer.add_param_group` goes.
   *
   * **It grows the state bank too.** Without that, the next step blows up
   * with "no state for parameter N" — the bank is indexed by the flattened
   * parameter position.
   *
   * A scheduler already built does not know this group's baseline. torch
   * asks for `initial_lr` in the same place — if the scheduler came first,
   * build it again.
   */
  addParamGroup(init: ParamGroupInit): void {
    const group = this.attach(init);
    this.extendState(group.params);
  }

  /**
   * One state bank. **Going through here is what makes it grow when a group
   * is added.**
   *
   * It used to be built per optimizer with `params.map(...)`, which leaves
   * the slot for a later-added parameter empty.
   *
   * @param value the value to fill with. Only `Rprop`'s step size is
   *   non-zero.
   */
  protected state(shapes: readonly Tensor[], value = 0): Tensor[] {
    // **`Tensor.zeros` and `Tensor.full` must not be used.** A single-element tensor is
    // cached by value so the same buffer comes back, and an optimiser **writes** to its
    // state — `Adam`'s m and v collide and invalidate the whole command buffer, while
    // `SGD`'s momentum and `Rprop`'s step size overwrite a constant the entire program
    // uses. `owned` gives a buffer of its own (`tensor.ts`).
    //
    // **Gathering the place they are made into one is the point.** The first fix for this
    // defect changed `Composed.state()` alone, and `SGD`, `Adam` and `RMSprop` use their
    // own kernels and sit outside that base, so it never reached them — three were left
    // while the record said it was fixed. Kept as a rule like "be careful when the size
    // is 1", the next optimiser forgets again — and `Rprop` really did arrive that way.
    const slots = shapes.map((p) => keepAlive(Tensor.owned(p.shape, value)));
    this.banks.push({ slots, value });
    return slots;
  }

  /**
   * Grows the state when a group is added.
   *
   * By default it adds one **parameter-shaped** slot per registered bank.
   * An optimizer for which that premise does not hold (`Adafactor` holds
   * variance folded into rows and columns) overrides here.
   */
  protected extendState(added: readonly Tensor[]): void {
    for (const { slots, value } of this.banks) {
      for (const p of added) slots.push(keepAlive(Tensor.owned(p.shape, value)));
    }
  }

  /**
   * The learning rate of the group currently being stepped.
   */
  protected get lr(): number {
    return this.paramGroups[this.currentGroup]?.lr ?? this.defaultLr;
  }

  /**
   * What this group set for itself, or the optimizer's default if it set
   * nothing.
   */
  protected grouped(fallback: number): number {
    return this.paramGroups[this.currentGroup]?.weightDecay ?? fallback;
  }

  /**
   * Empties the gradients. **It returns them to `null`** — filling with
   * zeros blurs the leaf test.
   */
  zeroGrad(): void {
    for (const p of this.params) p.grad = null;
  }

  step(): void {
    noGrad(() => {
      for (const [i, p] of this.params.entries()) {
        const g = p.grad;
        if (!g) continue;
        // Which group a parameter belongs to has to be settled first for `this.lr` to
        // give the right value.
        this.currentGroup = this.groupOf[i] ?? 0;
        this.update(i, p, g);
      }
      this.currentGroup = 0;
    });
  }

  /**
   * The scalars this optimizer holds. Subclasses override it.
   *
   * **The bank does not go in here** — the base already holds it, so
   * `stateDict` picks it up on its own. What belongs here is the non-tensor
   * things such as `stepCount`, and leaving one out means **bias correction
   * restarts from the beginning** on resume. The values come out slightly
   * different with nothing pointing at where they diverged.
   */
  protected counters(): Record<string, number> {
    return {};
  }

  /**
   * The undo of the above. Whoever overrode it takes their own back out.
   */
  protected loadCounters(_values: Record<string, number>): void {
    /* an optimiser with no scalars has nothing to do */
  }

  /**
   * Everything needed to carry on training.
   *
   * **Momentum alone is not enough.** The group's learning rate (a
   * scheduler may already have cut it) and the step counter have to travel
   * with it for the step after a resume to give the same number as one that
   * never stopped.
   *
   * Tensors and numbers come back separately — the safetensors body carries
   * tensors and the header carries numbers (`serialize.ts`).
   */
  stateDict(): { tensors: Record<string, Tensor>; numbers: Record<string, number> } {
    const tensors: Record<string, Tensor> = {};
    for (const [b, bank] of this.banks.entries()) {
      for (const [i, slot] of bank.slots.entries()) tensors[`bank${b}.${i}`] = slot;
    }
    const numbers: Record<string, number> = { ...this.counters() };
    for (const [g, group] of this.paramGroups.entries()) {
      numbers[`group${g}.lr`] = group.lr;
      if (group.initialLr !== undefined) numbers[`group${g}.initialLr`] = group.initialLr;
      if (group.weightDecay !== undefined) {
        numbers[`group${g}.weightDecay`] = group.weightDecay;
      }
    }
    return { tensors, numbers };
  }

  /**
   * Restores what `stateDict()` gave. Call it **after rebuilding the
   * optimizer with the same arguments.**
   *
   * It throws if a bank slot is missing or a size differs. Adding or
   * removing parameters, or laying an old checkpoint over a model, can
   * otherwise leave training running with values shifted by one slot, and
   * that is far worse than an exception.
   */
  loadStateDict(state: {
    tensors: Record<string, Tensor>;
    numbers: Record<string, number>;
  }): void {
    noGrad(() => {
      for (const [b, bank] of this.banks.entries()) {
        for (const [i, slot] of bank.slots.entries()) {
          const saved = state.tensors[`bank${b}.${i}`];
          if (!saved) throw new RuntimeError(`the checkpoint has no bank${b}.${i}`);
          if (saved.size !== slot.size) {
            throw new RuntimeError(
              `bank${b}.${i} has a different size: saved ${saved.size}, now ${slot.size}`,
            );
          }
          slot.copyFrom(saved);
        }
      }
    });
    for (const [g, group] of this.paramGroups.entries()) {
      const lr = state.numbers[`group${g}.lr`];
      if (lr !== undefined) group.lr = lr;
      const initial = state.numbers[`group${g}.initialLr`];
      if (initial !== undefined) group.initialLr = initial;
      const decay = state.numbers[`group${g}.weightDecay`];
      if (decay !== undefined) group.weightDecay = decay;
    }
    this.loadCounters(state.numbers);
  }

  protected abstract update(index: number, param: Tensor, grad: Tensor): void;
}

export class SGD extends Optimizer {
  private readonly buffers: Tensor[];

  constructor(
    params: ParamsArg,
    lr: number,
    private readonly momentum = 0,
    private readonly weightDecay = 0,
  ) {
    super(params, lr);
    // **The state is allocated up front.** Building a new tensor every step has it
    // released when the scope closes, and the next step's buffer points at a place that
    // is gone. torch holds its state as a fixture too.
    //
    // Starting from 0 still matches torch's values: the first step's
    // `0·momentum + grad` is the same number as torch's `buf = grad.clone()`.
    this.buffers = momentum === 0 ? []
      : this.state(this.params);
  }

  protected override update(index: number, param: Tensor, grad: Tensor): void {
    const n = param.size;
    const d = device();
    const buffers = [param.buffer, grad.buffer];
    if (this.momentum !== 0) {
      const buf = this.buffers[index];
      if (!buf) throw new Error(`SGD: no buffer for parameter ${index}`);
      buffers.push(buf.buffer);
    }
    // **A group's own value wins where it set one.** Excluding bias and norm from weight
    // decay is this slot's representative use — that one thing is why groups exist.
    const decay = this.grouped(this.weightDecay);
    d.run1d(
      d.pipeline(`sgd:${n}:${this.lr}:${this.momentum}:${decay}`,
        () => sgdStep(n, this.lr, this.momentum, decay)),
      buffers,
      n,
    );
  }
}

export class Adam extends Optimizer {
  private readonly first: Tensor[];
  private readonly second: Tensor[];
  private stepCount = 0;

  /**
   * **`AdamW` is this class with the decay moved.** Coupled, it goes onto the
   * gradient before the moments see it; decoupled, onto the weights after the
   * update. That one placement is the whole difference between the two names.
   */
  protected readonly decoupled: boolean = false;

  constructor(
    params: ParamsArg,
    lr: number,
    private readonly beta1 = 0.9,
    private readonly beta2 = 0.999,
    private readonly eps = 1e-8,
    private readonly weightDecay = 0,
  ) {
    super(params, lr);
    this.first = this.state(this.params);
    this.second = this.state(this.params);
  }

  override step(): void {
    // The bias correction depends on the step count, so it is computed once rather than
    // per parameter.
    this.stepCount += 1;
    super.step();
  }

  protected override counters(): Record<string, number> {
    return { stepCount: this.stepCount };
  }

  protected override loadCounters(v: Record<string, number>): void {
    if (v.stepCount !== undefined) this.stepCount = v.stepCount;
  }

  protected override update(index: number, param: Tensor, grad: Tensor): void {
    const m = this.first[index];
    const v = this.second[index];
    if (!m || !v) throw new Error(`Adam: no state for parameter ${index}`);
    // The bias correction differs every step, so it is handed across in a small buffer
    // rather than baked into the shader.
    const corr = Tensor.from([
      1 - this.beta1 ** this.stepCount,
      1 - this.beta2 ** this.stepCount,
    ], [2]);

    // **The kernel is not touched.** Both decays can be written as tensor operations.
    //
    // The coupled one adds into the gradient before the moments see it — `g + λ·p`.
    //
    // The decoupled one applies to the original weight **after** the update:
    // `p − lr·m̂/(√v̂+ε) − lr·λ·p`. That is the same number as
    // `p·(1 − lr·λ) − lr·m̂/(√v̂+ε)`, so **shrink first** and then take an ordinary step.
    // The moments come from the gradient, so shrinking in advance does not disturb them.
    //
    // Baking one more argument into the kernel would also have worked, and then there is
    // one pipeline per `weightDecay` value — because that number goes into the baked
    // name.
    const decay = this.grouped(this.weightDecay);
    let g = grad;
    if (decay !== 0) {
      noGrad(() => {
        if (this.decoupled) {
          param.copyFrom(param.mul(Tensor.full([], 1 - this.lr * decay)));
        } else {
          g = grad.add(param.mul(Tensor.full([], decay)));
        }
      });
    }

    const n = param.size;
    const d = device();
    d.run1d(
      d.pipeline(`adam:${n}:${this.lr}:${this.beta1}:${this.beta2}:${this.eps}`,
        () => adamStep(n, this.lr, this.beta1, this.beta2, this.eps)),
      [param.buffer, g.buffer, m.buffer, v.buffer, corr.buffer],
      n,
    );
  }
}

/**
 * Adam with the weight decay applied **to the weights rather than to the
 * gradient.** `torch.optim.AdamW`.
 *
 * Coupled decay reaches the moments, so it is scaled by them and a parameter
 * with a large gradient history gets decayed less. Decoupled decay does not,
 * which is the correction the AdamW paper is about — and it is why the two
 * names cannot be one class with a flag the caller sets by accident.
 *
 * **The default is 0.01, not 0.** torch's is too. Left at zero the two
 * optimizers are the same one, so a default of zero would make the name mean
 * nothing until somebody passed an argument.
 */
export class AdamW extends Adam {
  protected override readonly decoupled = true;

  constructor(
    params: ParamsArg,
    lr = 1e-3,
    beta1 = 0.9,
    beta2 = 0.999,
    eps = 1e-8,
    weightDecay = 0.01,
  ) {
    super(params, lr, beta1, beta2, eps, weightDecay);
  }
}

export class RMSprop extends Optimizer {
  private readonly squares: Tensor[];

  constructor(
    params: ParamsArg,
    lr: number,
    private readonly alpha = 0.99,
    private readonly eps = 1e-8,
    private readonly weightDecay = 0,
  ) {
    super(params, lr);
    this.squares = this.state(this.params);
  }

  protected override update(index: number, param: Tensor, g: Tensor): void {
    const sq = this.squares[index];
    if (!sq) throw new Error(`RMSprop: no state for parameter ${index}`);
    // **The kernel is not touched.** Coupled decay adds into the gradient, so it is
    // finished as a tensor operation before the kernel — baking the value into the shader
    // gives one pipeline per `weightDecay`, since that number goes into the baked name
    // (the same judgement as `Adam`).
    const decay = this.grouped(this.weightDecay);
    const grad = decay === 0 ? g : g.add(param.mul(Tensor.full([], decay)));
    const n = param.size;
    const d = device();
    d.run1d(
      d.pipeline(`rms:${n}:${this.lr}:${this.alpha}:${this.eps}`,
        () => rmspropStep(n, this.lr, this.alpha, this.eps)),
      [param.buffer, grad.buffer, sq.buffer],
      n,
    );
  }
}

/**
 * The base for optimisers that hold their state as tensors and update it **with tensor
 * operations.**
 *
 * `SGD`, `Adam` and `RMSprop` use their own WGSL kernels — those three are nearly the
 * whole of a training loop, so fusing them was worth it. The rest are not, so they are
 * assembled from operations that exist. Correct comes first, and baking a kernel is a job
 * for **after measuring** that it is needed.
 *
 * The updated value is written back **in place** with `copyFrom`. Swapping in a new tensor
 * separates the handle the model holds from the one the optimiser holds, and then training
 * runs while the parameters do not move.
 */

abstract class Composed extends Optimizer {
  // `state()` moved up to the base — `SGD`, `Adam` and `RMSprop` have to register their
  // banks too for `addParamGroup` to grow them along with the rest.

  /**
   * **The decay is held here.** All five use the same coupled form (`g + λ·p`), so giving
   * each its own field and two lines makes five copies, and a day comes when one of them
   * is fixed.
   *
   * The decoupled form is `AdamW` alone, and that one sits on its own kernel and never
   * comes here.
   */
  constructor(params: ParamsArg, lr: number,
              protected readonly weightDecay = 0) {
    super(params, lr);
  }

  /**
   * The gradient with decay applied. **At 0 it hands back the original** — multiplying by
   * 0 and adding costs two more operations per parameter per step, and gives the same
   * value as not doing it.
   */
  protected decayed(param: Tensor, grad: Tensor): Tensor {
    const wd = this.grouped(this.weightDecay);
    return wd === 0 ? grad : grad.add(param.mul(this.k(wd)));
  }

  protected at(bank: Tensor[], index: number, what: string): Tensor {
    const got = bank[index];
    if (!got) throw new Error(`${what}: no state for parameter ${index}`);
    return got;
  }

  /** One constant as a tensor. Arithmetic against a scalar attaches by broadcasting. */
  protected k(v: number): Tensor {
    return Tensor.full([], v);
  }
}

/**
 * **Keeps adding** squared gradients — it only ever shrinks, never grows.
 */
export class Adagrad extends Composed {
  private readonly sums: Tensor[];
  private stepCount = 0;

  /**
   * **`weightDecay` sits fourth, before `eps`** — that is torch's order, and the
   * binding calls this positionally, so the position is the contract. It is one
   * of the two here inserted into the middle rather than appended; every call in
   * this repository stops at `lr`, so nothing moved under them, and that was
   * checked rather than assumed.
   */
  constructor(params: ParamsArg, lr = 0.01, private readonly lrDecay = 0,
              weightDecay = 0, private readonly eps = 1e-10) {
    super(params, lr, weightDecay);
    this.sums = this.state(this.params);
  }

  override step(): void {
    this.stepCount += 1;
    super.step();
  }

  protected override counters(): Record<string, number> {
    return { stepCount: this.stepCount };
  }

  protected override loadCounters(v: Record<string, number>): void {
    if (v.stepCount !== undefined) this.stepCount = v.stepCount;
  }

  protected override update(index: number, param: Tensor, g: Tensor): void {
    const grad = this.decayed(param, g);
    const sum = this.at(this.sums, index, "Adagrad");
    sum.copyFrom(sum.add(grad.square()));
    const lr = this.lr / (1 + (this.stepCount - 1) * this.lrDecay);
    param.copyFrom(param.sub(grad.mul(this.k(lr)).div(sum.sqrt().add(this.k(this.eps)))));
  }
}

/**
 * **The learning rate is barely used.** It builds the step size out of the
 * history of its own updates.
 */
export class Adadelta extends Composed {
  private readonly squares: Tensor[];
  private readonly deltas: Tensor[];

  constructor(params: ParamsArg, lr = 1.0, private readonly rho = 0.9,
              private readonly eps = 1e-6, weightDecay = 0) {
    super(params, lr, weightDecay);
    this.squares = this.state(this.params);
    this.deltas = this.state(this.params);
  }

  protected override update(index: number, param: Tensor, g: Tensor): void {
    const grad = this.decayed(param, g);
    const sq = this.at(this.squares, index, "Adadelta");
    const acc = this.at(this.deltas, index, "Adadelta");
    const rho = this.k(this.rho);
    const keep = this.k(1 - this.rho);
    const eps = this.k(this.eps);
    sq.copyFrom(sq.mul(rho).add(grad.square().mul(keep)));
    const delta = acc.add(eps).sqrt().div(sq.add(eps).sqrt()).mul(grad);
    acc.copyFrom(acc.mul(rho).add(delta.square().mul(keep)));
    param.copyFrom(param.sub(delta.mul(this.k(this.lr))));
  }
}

/**
 * Adam with the second moment taken as **the maximum instead of the root
 * mean square.**
 */
export class Adamax extends Composed {
  private readonly first: Tensor[];
  private readonly inf: Tensor[];
  private stepCount = 0;

  constructor(params: ParamsArg, lr = 2e-3, private readonly beta1 = 0.9,
              private readonly beta2 = 0.999, private readonly eps = 1e-8,
              weightDecay = 0) {
    super(params, lr, weightDecay);
    this.first = this.state(this.params);
    this.inf = this.state(this.params);
  }

  override step(): void {
    this.stepCount += 1;
    super.step();
  }

  protected override counters(): Record<string, number> {
    return { stepCount: this.stepCount };
  }

  protected override loadCounters(v: Record<string, number>): void {
    if (v.stepCount !== undefined) this.stepCount = v.stepCount;
  }

  protected override update(index: number, param: Tensor, g: Tensor): void {
    const grad = this.decayed(param, g);
    const m = this.at(this.first, index, "Adamax");
    const u = this.at(this.inf, index, "Adamax");
    m.copyFrom(m.mul(this.k(this.beta1)).add(grad.mul(this.k(1 - this.beta1))));
    u.copyFrom(u.mul(this.k(this.beta2)).binary("maximum", grad.abs().add(this.k(this.eps))));
    const bias = 1 - this.beta1 ** this.stepCount;
    param.copyFrom(param.sub(m.div(u).mul(this.k(this.lr / bias))));
  }
}

/**
 * Adam with Nesterov's look-ahead attached.
 *
 * **The momentum coefficient changes each step, and the running product of
 * that sequence has to be carried.** Held constant, the first few steps
 * diverge quietly.
 */
export class NAdam extends Composed {
  private readonly first: Tensor[];
  private readonly second: Tensor[];
  private muProduct = 1;
  private stepCount = 0;

  /**
   * **`weightDecay` goes before `momentumDecay`, not after** — torch's order,
   * and the binding calls positionally.
   *
   * This one is why an arity check cannot find a dropped argument. Before this,
   * the binding passed six and this took six, so the counts agreed perfectly
   * while the sixth was `momentumDecay` at both ends and `weight_decay` reached
   * nothing. It was the last of seven to be found, and the only one an
   * argument-count comparison could never have shown.
   */
  constructor(params: ParamsArg, lr = 2e-3, private readonly beta1 = 0.9,
              private readonly beta2 = 0.999, private readonly eps = 1e-8,
              weightDecay = 0, private readonly momentumDecay = 4e-3) {
    super(params, lr, weightDecay);
    this.first = this.state(this.params);
    this.second = this.state(this.params);
  }

  override step(): void {
    this.stepCount += 1;
    const t = this.stepCount;
    this.mu = this.beta1 * (1 - 0.5 * 0.96 ** (t * this.momentumDecay));
    this.muNext = this.beta1 * (1 - 0.5 * 0.96 ** ((t + 1) * this.momentumDecay));
    // The running product grows once per step rather than once per parameter.
    this.muProduct *= this.mu;
    super.step();
  }

  private mu = 0;
  private muNext = 0;

  protected override counters(): Record<string, number> {
    return { stepCount: this.stepCount, muProduct: this.muProduct, mu: this.mu, muNext: this.muNext };
  }

  protected override loadCounters(v: Record<string, number>): void {
    if (v.stepCount !== undefined) this.stepCount = v.stepCount;
    if (v.muProduct !== undefined) this.muProduct = v.muProduct;
    if (v.mu !== undefined) this.mu = v.mu;
    if (v.muNext !== undefined) this.muNext = v.muNext;
  }

  protected override update(index: number, param: Tensor, g: Tensor): void {
    const grad = this.decayed(param, g);
    const m = this.at(this.first, index, "NAdam");
    const v = this.at(this.second, index, "NAdam");
    m.copyFrom(m.mul(this.k(this.beta1)).add(grad.mul(this.k(1 - this.beta1))));
    v.copyFrom(v.mul(this.k(this.beta2)).add(grad.square().mul(this.k(1 - this.beta2))));
    const denom = v.div(this.k(1 - this.beta2 ** this.stepCount)).sqrt()
      .add(this.k(this.eps));
    const a = grad.div(denom).mul(this.k(this.lr * (1 - this.mu) / (1 - this.muProduct)));
    const b = m.div(denom).mul(
      this.k(this.lr * this.muNext / (1 - this.muProduct * this.muNext)));
    param.copyFrom(param.sub(a).sub(b));
  }
}

/**
 * Adam, but **without the adaptive step size early on.**
 *
 * Adam is known to be jumpy at the start because the second moment has few
 * samples and its variance is large; this one passes through that stretch
 * like SGD. Drop the threshold (`rho > 5`) and the values become Adam's.
 */
export class RAdam extends Composed {
  private readonly first: Tensor[];
  private readonly second: Tensor[];
  private stepCount = 0;

  constructor(params: ParamsArg, lr = 1e-3, private readonly beta1 = 0.9,
              private readonly beta2 = 0.999, private readonly eps = 1e-8,
              weightDecay = 0) {
    super(params, lr, weightDecay);
    this.first = this.state(this.params);
    this.second = this.state(this.params);
  }

  override step(): void {
    this.stepCount += 1;
    super.step();
  }

  protected override counters(): Record<string, number> {
    return { stepCount: this.stepCount };
  }

  protected override loadCounters(v: Record<string, number>): void {
    if (v.stepCount !== undefined) this.stepCount = v.stepCount;
  }

  protected override update(index: number, param: Tensor, g: Tensor): void {
    const grad = this.decayed(param, g);
    const m = this.at(this.first, index, "RAdam");
    const v = this.at(this.second, index, "RAdam");
    const t = this.stepCount;
    m.copyFrom(m.mul(this.k(this.beta1)).add(grad.mul(this.k(1 - this.beta1))));
    v.copyFrom(v.mul(this.k(this.beta2)).add(grad.square().mul(this.k(1 - this.beta2))));
    const mh = m.div(this.k(1 - this.beta1 ** t));
    const rhoInf = 2 / (1 - this.beta2) - 1;
    const rho = rhoInf - (2 * t * this.beta2 ** t) / (1 - this.beta2 ** t);
    if (rho > 5) {
      const rect = Math.sqrt(
        ((rho - 4) * (rho - 2) * rhoInf) / ((rhoInf - 4) * (rhoInf - 2) * rho));
      const denom = v.div(this.k(1 - this.beta2 ** t)).sqrt().add(this.k(this.eps));
      param.copyFrom(param.sub(mh.mul(this.k(this.lr * rect)).div(denom)));
    } else {
      param.copyFrom(param.sub(mh.mul(this.k(this.lr))));
    }
  }
}

/**
 * Averaged SGD. **The learning rate decays on its own each step**, and from
 * some point on the parameters are averaged.
 *
 * `eta` decays as `lr / (1 + lambd·lr·step)^alpha`, and `mu` is the weight
 * of the average. The default `t0` is a million, so in normal training `mu`
 * is always 1 and `ax` is a copy of the parameters — the averaging branch
 * only really runs once `t0` is lowered.
 *
 * **The decay is multiplicative.** `param *= (1 − lambd·eta)` happens
 * first, and the gradient is subtracted after.
 */
export class ASGD extends Composed {
  private readonly ax: Tensor[];
  private eta: number;
  private mu = 1;
  private stepCount = 0;

  constructor(params: ParamsArg, lr = 1e-2, private readonly lambd = 1e-4,
              private readonly alpha = 0.75, private readonly t0 = 1e6,
              weightDecay = 0) {
    super(params, lr, weightDecay);
    this.ax = this.state(this.params);
    this.eta = lr;
  }

  override step(): void {
    this.stepCount += 1;
    super.step();
    this.eta = this.lr / (1 + this.lambd * this.lr * this.stepCount) ** this.alpha;
    this.mu = 1 / Math.max(1, this.stepCount - this.t0);
  }

  protected override counters(): Record<string, number> {
    return { stepCount: this.stepCount, eta: this.eta, mu: this.mu };
  }

  protected override loadCounters(v: Record<string, number>): void {
    if (v.stepCount !== undefined) this.stepCount = v.stepCount;
    if (v.eta !== undefined) this.eta = v.eta;
    if (v.mu !== undefined) this.mu = v.mu;
  }

  protected override update(index: number, param: Tensor, grad: Tensor): void {
    const ax = this.at(this.ax, index, "ASGD");
    const eta = this.eta;
    const mu = this.mu;
    // **It uses the base's.** This was written out by hand here, and then it does not
    // see the per-group decay (`grouped`), so a value set differently per layer is ignored
    // in this optimiser alone.
    const g = this.decayed(param, grad);
    param.copyFrom(param.mul(this.k(1 - this.lambd * eta)).sub(g.mul(this.k(eta))));
    // With `mu` at 1 it is **a copy** rather than an average — adding gives twice.
    ax.copyFrom(mu === 1 ? param : ax.add(param.sub(ax).mul(this.k(mu))));
  }
}

/**
 * Looks at **the sign of the gradient only.** The magnitude is unused, and
 * the step width grows and shrinks per slot.
 *
 * If the sign holds, the width is multiplied by `etaPlus`; if it flips, by
 * `etaMinus`. **A flipped slot does not take that step at all** — its
 * gradient is set to zero, which also makes the next step's "previous
 * gradient" zero. Without those two, inputs whose sign never flips produce
 * a difference that is never caught.
 */
export class Rprop extends Composed {
  private readonly prev: Tensor[];
  private readonly stepSize: Tensor[];

  constructor(params: ParamsArg, lr = 1e-2, private readonly etaMinus = 0.5,
              private readonly etaPlus = 1.2, private readonly sizeMin = 1e-6,
              private readonly sizeMax = 50) {
    super(params, lr);
    this.prev = this.state(this.params);
    // **It must not be built with `Tensor.full`.** On a size-1 parameter that hands back
    // the value-cached global `lr` constant, and `update` below writes into it with
    // `size.copyFrom(...)` — that constant, for the whole program, changes during
    // training. Nothing is raised.
    this.stepSize = this.state(this.params, lr);
  }

  protected override update(index: number, param: Tensor, grad: Tensor): void {
    const prev = this.at(this.prev, index, "Rprop");
    const size = this.at(this.stepSize, index, "Rprop");
    const sign = grad.mul(prev).sign();
    // **The true/false table is turned back into floats.** A comparison gives bool and
    // borch.ts refuses arithmetic on bool — the values are 0/1 so it looks like it would
    // simply work, and the refusal is right.
    const rising = sign.binary("gt", this.k(0)).to("float32");
    const falling = sign.binary("lt", this.k(0)).to("float32");
    // A sign of 0 multiplies by 1 — that covers the first step and any cell zeroed by an
    // earlier reversal.
    const factor = this.k(1)
      .add(rising.mul(this.k(this.etaPlus - 1)))
      .add(falling.mul(this.k(this.etaMinus - 1)));
    size.copyFrom(size.mul(factor).clamp(this.sizeMin, this.sizeMax));
    const kept = grad.mul(this.k(1).sub(falling));
    param.copyFrom(param.sub(kept.sign().mul(size)));
    prev.copyFrom(kept);
  }
}

/**
 * A quasi-Newton method. **`step` takes a closure, and here it is async.**
 *
 * ## Why this one is different
 *
 * Every other optimizer walks one step from one set of gradients. This one loops
 * up to `maxIter` times inside a single step, asking for the loss and the
 * gradients again each time. **In torch too, this is the only name that takes a
 * closure** — the shape of the training loop is genuinely different.
 *
 * Being async is a second mark on top of that. Each iteration **reads scalars to
 * branch** — the gradient threshold, the curvature `y·s`, the directional
 * derivative `g·d`, the change in loss. They are all conditions of an `if` or a
 * `break`, so they cannot stay on the GPU, and reading a value is async here.
 *
 * **There was one way to remove those reads, and the golden closed it.** Drop the
 * early termination and run a fixed `maxIter` and this becomes synchronous — but
 * one of the `opt::LBFGS` cases asks precisely whether it stops near the tolerance.
 * What you get that way is not a synchronous LBFGS; it is **a different algorithm.**
 *
 * ## It is slow, and that is a property of the algorithm
 *
 * One `step()` costs something like a hundred GPU-to-host round trips (twenty
 * iterations × five scalars). Running a quasi-Newton method in a browser is that
 * kind of work, not a flaw in this implementation. Use `Adam` for a large model;
 * use this name to solve **a small problem precisely.**
 *
 * **There is no line search.** Pass `lineSearchFn` and it stops loudly — taking a
 * fixed step size quietly converges differently, and that difference shows up in
 * the curve, not in a value.
 */
export class LBFGS extends Optimizer {
  private readonly history: {
    dirs: Tensor[]; stps: Tensor[]; ro: number[];
  } = { dirs: [], stps: [], ro: [] };
  private d: Tensor | null = null;
  private t = 0;
  private hDiag = 1;
  private prevFlat: Tensor | null = null;
  private prevLoss = 0;
  private iterations = 0;

  constructor(
    params: ParamsArg,
    lr = 1,
    private readonly maxIter = 20,
    maxEval: number | null = null,
    private readonly toleranceGrad = 1e-7,
    private readonly toleranceChange = 1e-9,
    private readonly historySize = 100,
    lineSearchFn: string | null = null,
  ) {
    super(params, lr);
    if (lineSearchFn !== null) {
      throw new RuntimeError(
        `LBFGS(lineSearchFn=${JSON.stringify(lineSearchFn)}) is not implemented — ` +
        "taking a fixed step size instead would converge differently.");
    }
    if (this.paramGroups.length !== 1) {
      throw new RuntimeError("LBFGS takes a single parameter group.");
    }
    this.maxEval = maxEval ?? Math.floor((this.maxIter * 5) / 4);
  }

  private readonly maxEval: number;

  /**
   * **Nothing reaches here.** The base `step()` calls this once per parameter, and
   * this class overrides that `step()` whole — a step does not divide by parameter.
   * The slot is filled because the base demands it; being called means the override
   * came undone.
   */
  protected update(): void {
    throw new RuntimeError(
      "LBFGS has no per-parameter update — step(closure) takes its direction " +
      "from the concatenated gradient. Reaching here means step() was not overridden.");
  }

  /** Sees the parameters' gradients **joined into one row.** A missing one is 0. */
  private flatGrad(): Tensor {
    return Tensor.cat(
      this.params.map((p) => (p.grad ?? Tensor.zeros(p.shape)).reshape([p.size])), 0);
  }

  /** Cuts a one-row direction into the parameters' shapes and adds. */
  private addStep(size: number, direction: Tensor): void {
    let at = 0;
    noGrad(() => {
      for (const p of this.params) {
        const piece = direction.narrow(0, at, p.size).reshape(p.shape);
        p.copyFrom(p.add(piece.mul(Tensor.full([], size))));
        at += p.size;
      }
    });
  }

  /**
   * @param closure Re-evaluates the loss and fills the gradients. **Without it
   *   nothing can happen.**
   * @returns The loss measured on entry, as torch does — not the last one.
   */
  override async step(closure?: () => Tensor): Promise<Tensor> {
    if (!closure) {
      throw new RuntimeError(
        "LBFGS.step needs a closure — await opt.step(() => loss()). " +
        "It re-evaluates the loss several times within a single step.");
    }
    const orig = closure();
    let loss = await orig.item();
    let evals = 1;
    let flat = this.flatGrad();
    if (await flat.abs().max(0).values.item() <= this.toleranceGrad) return orig;

    let iter = 0;
    while (iter < this.maxIter) {
      iter += 1;
      this.iterations += 1;
      if (this.iterations === 1) {
        this.d = flat.neg();
        this.history.dirs = [];
        this.history.stps = [];
        this.history.ro = [];
        this.hDiag = 1;
      } else {
        const prev = this.prevFlat ?? flat;
        const y = flat.sub(prev);
        const s = (this.d ?? flat).mul(Tensor.full([], this.t));
        const ys = await y.mul(s).sum().item();
        if (ys > 1e-10) {
          if (this.history.dirs.length === this.historySize) {
            this.history.dirs.shift();
            this.history.stps.shift();
            this.history.ro.shift();
          }
          this.history.dirs.push(keepAlive(y));
          this.history.stps.push(keepAlive(s));
          this.history.ro.push(1 / ys);
          this.hDiag = ys / await y.mul(y).sum().item();
        }
        // The two-loop recursion — it produces the direction without building the
        // inverse Hessian.
        const n = this.history.dirs.length;
        const al = new Array<number>(n).fill(0);
        let q = flat.neg();
        for (let i = n - 1; i >= 0; i--) {
          al[i] = await this.history.stps[i]!.mul(q).sum().item() * this.history.ro[i]!;
          q = q.sub(this.history.dirs[i]!.mul(Tensor.full([], al[i]!)));
        }
        let r = q.mul(Tensor.full([], this.hDiag));
        for (let i = 0; i < n; i++) {
          const be = await this.history.dirs[i]!.mul(r).sum().item() * this.history.ro[i]!;
          r = r.add(this.history.stps[i]!.mul(Tensor.full([], al[i]! - be)));
        }
        this.d = r;
      }

      this.prevFlat = keepAlive(flat.clone());
      this.prevLoss = loss;
      this.t = this.iterations === 1
        ? Math.min(1, 1 / await flat.abs().sum().item()) * this.lr
        : this.lr;
      const gtd = await flat.mul(this.d ?? flat).sum().item();
      if (gtd > -this.toleranceChange) break;

      this.addStep(this.t, this.d ?? flat);
      if (iter !== this.maxIter) {
        // Nothing is measured again on the last iteration — as in torch.
        loss = await closure().item();
        flat = this.flatGrad();
        evals += 1;
        if (await flat.abs().max(0).values.item() <= this.toleranceGrad) break;
      }
      if (iter === this.maxIter || evals >= this.maxEval) break;
      const moved = await (this.d ?? flat).mul(Tensor.full([], this.t)).abs().max(0).values.item();
      if (moved <= this.toleranceChange) break;
      if (Math.abs(loss - this.prevLoss) < this.toleranceChange) break;
    }
    return orig;
  }
}

/**
 * Adam, but holding the second moment **split into rows and columns.**
 *
 * Adam holds one variance per parameter, so its memory costs as much as the
 * weights. Here only the row means and the column means are held and the
 * outer product restores them — `R + C` where `(R, C)` would have been.
 *
 * **One-dimensional parameters are not split** — there is only one axis to
 * split, so the variance is held directly. Ask with vectors alone and the
 * whole point of this optimization never runs.
 */
export class Adafactor extends Composed {
  private readonly rowVar: (Tensor | null)[] = [];
  private readonly colVar: (Tensor | null)[] = [];
  private readonly variance: (Tensor | null)[] = [];
  private stepCount = 0;

  constructor(params: ParamsArg, lr = 1e-2, private readonly beta2Decay = -0.8,
              private readonly eps1: number | null = null,
              private readonly eps2 = 1e-3, private readonly d = 1.0,
              weightDecay = 0) {
    super(params, lr, weightDecay);
    this.extendState(this.params);
  }

  /**
   * **The base's default cannot be used.** This optimizer's state is not
   * parameter-shaped but folded into rows and columns, and which of the two
   * is held depends on the rank. The slot count matches the parameters (the
   * unused side is filled with `null`), so the indexing still works.
   *
   * The constructor and `addParamGroup` both pass through this one place —
   * written in two places, one of them eventually gets fixed alone.
   */
  protected override extendState(added: readonly Tensor[]): void {
    for (const p of added) {
      const rank = p.shape.length;
      if (rank > 1) {
        const rows = [...p.shape.slice(0, -1), 1];
        const cols = [...p.shape.slice(0, -2), 1, p.shape[rank - 1] ?? 1];
        // It is updated in place, so it has to be `owned` and outside the cache.
        this.rowVar.push(keepAlive(Tensor.owned(rows)));
        this.colVar.push(keepAlive(Tensor.owned(cols)));
        this.variance.push(null);
      } else {
        this.rowVar.push(null);
        this.colVar.push(null);
        this.variance.push(keepAlive(Tensor.owned(p.shape)));
      }
    }
  }

  override step(): void {
    this.stepCount += 1;
    super.step();
  }

  protected override counters(): Record<string, number> {
    return { stepCount: this.stepCount };
  }

  protected override loadCounters(v: Record<string, number>): void {
    if (v.stepCount !== undefined) this.stepCount = v.stepCount;
  }

  protected override update(index: number, param: Tensor, grad: Tensor): void {
    // f32's machine epsilon. torch takes it from the dtype and here there is only
    // float32.
    const one = this.eps1 ?? 1.1920928955078125e-7;
    const step = this.stepCount;
    const blend = this.k(step ** this.beta2Decay);
    const rho = Math.min(this.lr, 1 / Math.sqrt(step));
    // **`alpha` and `denom` cannot be taken out as numbers** — WebGPU has no synchronous
    // read, so reading a value on the GPU here means waiting. They stay scalar **tensors**
    // and are multiplied. The core is numpy and computes them as numbers, and the
    // expression is the same.
    const norm = param.square().sum().sqrt();
    const alpha = norm.div(this.k(Math.sqrt(param.size)))
      .binary("maximum", this.k(this.eps2)).mul(this.k(rho));
    // **This one is the decoupled form** rather than `decayed()`'s coupled one. torch's
    // `Adafactor` applies it to the weight directly, the same place as `AdamW`. Swapping
    // in the base's helper quietly makes it a different optimiser.
    const wd = this.grouped(this.weightDecay);
    if (wd) param.copyFrom(param.mul(this.k(1 - this.lr * wd)));
    const rank = grad.shape.length;
    let variance: Tensor;
    if (rank > 1) {
      const row = this.rowVar[index];
      const col = this.colVar[index];
      if (!row || !col) throw new Error("Adafactor: the row/column state is missing");
      const sq = grad.square();
      row.copyFrom(row.add(sq.mean(rank - 1, true).sub(row).mul(blend)));
      col.copyFrom(col.add(sq.mean(rank - 2, true).sub(col).mul(blend)));
      // `(…, R, 1) × (…, 1, C)` is the outer product — broadcasting does it as it
      // stands.
      const outer = row.mul(col);
      variance = outer.div(row.mean(rank - 2, true).binary("maximum", this.k(one)));
    } else {
      const v = this.variance[index];
      if (!v) throw new Error("Adafactor: the variance state is missing");
      v.copyFrom(v.add(grad.square().sub(v).mul(blend)));
      variance = v;
    }
    const update = grad.div(variance.binary("maximum", this.k(one * one)).sqrt());
    const scale = update.square().sum().sqrt();
    const denom = scale.div(this.k(Math.sqrt(update.size) * this.d))
      .binary("maximum", this.k(1));
    param.copyFrom(param.sub(update.mul(alpha.div(denom))));
  }
}

/**
 * Learning-rate schedules.
 *
 * **It is all floating-point arithmetic, so the values have to equal
 * torch's exactly** — there is no room for an approximation. So the golden
 * cases froze not one value but **the whole trajectory**, and doing that is
 * how the core caught the difference in `StepLR`.
 *
 * **The baseline is `initialLr`, not the lr at construction time.** It is
 * stamped onto the optimizer once, and schedulers built later see the same
 * baseline. Used alone the two are equal and nothing catches, but chained,
 * the second one takes as its baseline what the first has already cut.
 */
export abstract class LRScheduler {
  protected epoch = 0;
  /** The base value per group. The `base` a subclass computes from is the first
   *  group's. */
  private readonly bases: number[];
  readonly base: number;

  constructor(protected readonly opt: Optimizer) {
    this.bases = opt.paramGroups.map((g) => {
      if (g.initialLr === undefined) g.initialLr = g.lr;
      return g.initialLr;
    });
    this.base = this.bases[0] ?? 0;
  }

  /**
   * Applies the computed value **to every group.**
   *
   * Where the groups have different bases it keeps their ratio — the same result as
   * torch holding `base_lrs` per group and recomputing from each. `compute` produces one
   * value from the first group's base, so the ratio is multiplied in here.
   *
   * **With one group the ratio is 1 and it does not differ from the old behaviour by a
   * bit.** The golden, which freezes whole trajectories, watches that.
   */
  private apply(value: number): void {
    const first = this.bases[0] ?? 0;
    for (const [i, group] of this.opt.paramGroups.entries()) {
      const mine = this.bases[i] ?? first;
      group.lr = first === 0 ? value : value * (mine / first);
    }
  }

  /**
   * **Applies epoch zero.** torch does this in the constructor and here it
   * cannot be — a TypeScript subclass's fields are filled in **after**
   * `super()` returns, so calling `compute` inside the constructor finds
   * things like `factor` still `undefined`.
   *
   * So it is called once, right after construction. Something like
   * `ConstantLR`, which changes the value from epoch zero, loses its entire
   * first term without it — measured at a maximum difference of 2.0e-01.
   */
  start(): this {
    this.epoch = 0;
    this.apply(this.compute(0));
    return this;
  }

  step(): void {
    this.epoch += 1;
    this.apply(this.compute(this.epoch));
  }

  /**
   * The learning rate now. Where a recursive scheduler reads the value
   * before its own.
   */
  protected get current(): number {
    return this.opt.paramGroups[0]?.lr ?? 0;
  }

  /**
   * Makes it step from the beginning again, for when `SequentialLR` hands
   * over.
   */
  restart(): void {
    this.apply(this.base);
    this.start();
  }

  /**
   * What resuming needs — **which epoch, and the baselines.**
   *
   * Restoring the optimizer's `lr` alone is not enough. A scheduler is a
   * thing that recomputes its value from `epoch`, so leaving that at zero
   * and carrying on means **the next `step()` puts the learning rate back
   * to its first value** — the optimizer side was restored perfectly and
   * one step erases it.
   *
   * The baselines travel too. `initialLr` comes from the optimizer, but
   * that is **the value the first scheduler stamped**, and chained
   * schedulers hold the baseline each of them saw.
   */
  stateDict(): Record<string, number> {
    const out: Record<string, number> = { epoch: this.epoch };
    for (const [i, base] of this.bases.entries()) out[`base${i}`] = base;
    return out;
  }

  /**
   * The undo of `stateDict()`. It does not touch the learning rate — the
   * optimizer holds that.
   */
  loadStateDict(values: Record<string, number>): void {
    if (values.epoch !== undefined) this.epoch = values.epoch;
    for (const i of this.bases.keys()) {
      const base = values[`base${i}`];
      if (base !== undefined) this.bases[i] = base;
    }
  }

  protected abstract compute(epoch: number): number;
}

/**
 * Multiplies by `gamma` every `stepSize` epochs.
 *
 * **It is recursive** — the same reason `ExponentialLR` below writes down.
 * This used to be the closed form `base * gamma ** floor(epoch /
 * stepSize)`.
 *
 * Stepped alone from the beginning, the two produce **the same sequence.**
 * That is why the golden cases, which freeze the whole trace, were green
 * for a long time. What diverges is building a new scheduler on an
 * optimizer whose lr has already moved — **resuming training.** At that
 * moment the closed form puts the learning rate back to its first value
 * (0.05 to 0.2). No error appears; the loss just jumps once.
 */
export class StepLR extends LRScheduler {
  constructor(opt: Optimizer, private readonly stepSize: number,
              private readonly gamma = 0.1) {
    super(opt);
  }

  protected override compute(epoch: number): number {
    if (epoch === 0 || epoch % this.stepSize !== 0) return this.current;
    return this.current * this.gamma;
  }
}

/**
 * Cuts only at the milestones. **It is recursive** — fixed for the same
 * reason as `StepLR`.
 *
 * A milestone written twice (`[3, 3]`) multiplies twice at that point — it
 * did under the closed form too, and torch does the same.
 */
export class MultiStepLR extends LRScheduler {
  constructor(opt: Optimizer, private readonly milestones: readonly number[],
              private readonly gamma = 0.1) {
    super(opt);
  }

  protected override compute(epoch: number): number {
    const hits = this.milestones.filter((m) => m === epoch).length;
    return hits === 0 ? this.current : this.current * this.gamma ** hits;
  }
}

/**
 * **It is recursive** — it multiplies the current learning rate. It does
 * not recount from the original.
 *
 * Used alone the two produce the same sequence. What diverges is another
 * scheduler touching the same lr — overlapped through `ChainedScheduler`,
 * the recursive form stacks on top of the other's result while the closed
 * form overwrites what the other did.
 */
export class ExponentialLR extends LRScheduler {
  constructor(opt: Optimizer, private readonly gamma: number) {
    super(opt);
  }

  protected override compute(epoch: number): number {
    return epoch === 0 ? this.current : this.current * this.gamma;
  }
}

/**
 * **Holds it cut until `totalIters`, then returns to the original.** The
 * simplest form of a warm-up.
 */
export class ConstantLR extends LRScheduler {
  constructor(opt: Optimizer, private readonly factor = 1 / 3,
              private readonly totalIters = 5) {
    super(opt);
  }

  protected override compute(epoch: number): number {
    if (epoch === 0) return this.current * this.factor;
    if (epoch !== this.totalIters) return this.current;
    return this.current / this.factor;
  }
}

/**
 * Moves **in a straight line** from the starting factor to the ending
 * factor.
 *
 * It meets `ConstantLR` at the end — past `totalIters` both are the
 * original learning rate. The last value alone cannot separate them, so the
 * golden cases ask about the whole trace.
 */
export class LinearLR extends LRScheduler {
  constructor(opt: Optimizer, private readonly startFactor = 1 / 3,
              private readonly endFactor = 1.0, private readonly totalIters = 5) {
    super(opt);
  }

  protected override compute(epoch: number): number {
    const t = Math.min(epoch, this.totalIters);
    const scale = this.startFactor
      + (this.endFactor - this.startFactor) * (this.totalIters ? t / this.totalIters : 1);
    return this.base * scale;
  }
}

/**
 * Descends as `(1 − t/T)^power`. At `power=1` it is a straight line.
 */
export class PolynomialLR extends LRScheduler {
  constructor(opt: Optimizer, private readonly totalIters = 5,
              private readonly power = 1.0) {
    super(opt);
  }

  protected override compute(epoch: number): number {
    if (epoch === 0 || epoch > this.totalIters) return this.current;
    const decay = ((1 - epoch / this.totalIters)
      / (1 - (epoch - 1) / this.totalIters)) ** this.power;
    return this.current * decay;
  }
}

/**
 * **Multiplies as it goes** — the baseline is the current learning rate,
 * not the original.
 */
export class MultiplicativeLR extends LRScheduler {
  constructor(opt: Optimizer, private readonly fn: (epoch: number) => number) {
    super(opt);
  }

  protected override compute(epoch: number): number {
    return epoch === 0 ? this.base : this.current * this.fn(epoch);
  }
}

/**
 * Descends as a cosine and then **returns to the start.** Each period is
 * `tMult` times longer.
 */
export class CosineAnnealingWarmRestarts extends LRScheduler {
  private tI: number;
  private tCur = -1;

  constructor(opt: Optimizer, t0: number,
              private readonly tMult = 1, private readonly etaMin = 0) {
    super(opt);
    this.tI = t0;
  }

  protected override compute(): number {
    this.tCur += 1;
    while (this.tCur >= this.tI) {
      this.tCur -= this.tI;
      this.tI *= this.tMult;
    }
    return this.etaMin + (this.base - this.etaMin)
      * (1 + Math.cos((Math.PI * this.tCur) / this.tI)) / 2;
  }
}

/**
 * Up and then down. **Close to the default of a modern training recipe.**
 *
 * The initial learning rate is `maxLr/divFactor` and the end is that
 * divided again, so **the learning rate given to the optimizer is never
 * used** — it is overwritten the moment this is built.
 */
export class OneCycleLR extends LRScheduler {
  private readonly initial: number;
  private readonly minLr: number;
  private readonly up: number;
  private readonly down: number;

  constructor(opt: Optimizer, private readonly maxLr: number,
              private readonly totalSteps: number, pctStart = 0.3,
              divFactor = 25, finalDivFactor = 1e4) {
    super(opt);
    this.initial = maxLr / divFactor;
    this.minLr = this.initial / finalDivFactor;
    // torch's arithmetic verbatim — `pctStart × totalSteps − 1`, not
    // `pctStart × (totalSteps − 1)`.
    this.up = pctStart * totalSteps - 1;
    this.down = totalSteps - this.up - 1;
  }

  protected override compute(epoch: number): number {
    const t = Math.min(epoch, this.totalSteps - 1);
    const rising = t <= this.up;
    const frac = rising
      ? (this.up ? t / this.up : 1)
      : (t - this.up) / Math.max(1e-12, this.down);
    const lo = rising ? this.initial : this.maxLr;
    const hi = rising ? this.maxLr : this.minLr;
    return lo + (hi - lo) * (1 - Math.cos(Math.PI * frac)) / 2;
  }
}

/**
 * Makes the learning rate **rise and fall.** It shakes it on purpose, to leave saddle
 * points.
 *
 * It climbs for `stepSizeUp` and descends for `stepSizeDown`. Left unset, it descends by
 * as much as it climbed — **with the rise and the fall equal, there is no seeing whether
 * that argument exists at all.**
 *
 * Of the three `mode`s, `expRange` alone measures against **the step rather than the
 * cycle.** That is where they diverge.
 */
/**
 * **With more than one group, the bounds become relative.** torch takes
 * `base_lr` and `max_lr` as a list per group; here they are single numbers.
 * Following the base's rule, the value computed against the first group is
 * multiplied by each group's baseline ratio, so group i cycles between
 * `baseLr·rᵢ` and `maxLr·rᵢ`. That is what someone who set per-layer
 * learning rates expects, but it **is not the same as torch.** With one
 * group the ratio is 1 and there is no difference.
 */
export class CyclicLR extends LRScheduler {
  private readonly down: number;

  constructor(
    opt: Optimizer,
    private readonly baseLr: number,
    private readonly maxLr: number,
    private readonly up = 2000,
    stepSizeDown: number | null = null,
    private readonly mode: "triangular" | "triangular2" | "exp_range" = "triangular",
    private readonly gamma = 1.0,
  ) {
    super(opt);
    this.down = stepSizeDown ?? this.up;
  }

  protected override compute(epoch: number): number {
    const total = this.up + this.down;
    const ratio = this.up / total;
    const cycle = Math.floor(1 + epoch / total);
    const x = 1 + epoch / total - cycle;
    // The climbing stretch and the descending stretch have different slopes.
    const rise = x <= ratio ? x / ratio : (x - 1) / (ratio - 1);
    const scale = this.mode === "triangular2"
      ? 1 / 2 ** (cycle - 1)
      : this.mode === "exp_range" ? this.gamma ** epoch : 1;
    return this.baseLr + (this.maxLr - this.baseLr) * rise * scale;
  }
}

/**
 * **Chains schedulers end to end.** On reaching a milestone it hands over
 * to the next.
 */
export class SequentialLR {
  private epoch = 0;

  constructor(
    readonly opt: Optimizer,
    private readonly schedulers: LRScheduler[],
    private readonly milestones: readonly number[],
  ) {
    const first = this.schedulers[0];
    if (first) first.restart();
  }

  step(): void {
    this.epoch += 1;
    const idx = this.milestones.filter((m) => m <= this.epoch).length;
    const sch = this.schedulers[Math.min(idx, this.schedulers.length - 1)];
    if (!sch) return;
    if (idx > 0 && this.milestones[idx - 1] === this.epoch) sch.restart();
    else sch.step();
  }

  getLastLr(): number[] {
    return this.opt.paramGroups.map((g) => g.lr);
  }
}

/**
 * Applies several **at once.** Their factors multiply.
 */
export class ChainedScheduler {
  constructor(private readonly schedulers: LRScheduler[]) {}

  step(): void {
    for (const s of this.schedulers) s.step();
  }

  getLastLr(): number[] {
    return [];
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
 * Cuts the learning rate when the value stops improving.
 *
 * Unlike other schedules it **has to be given a value** to move — hence
 * `step(metric)`.
 */
export class ReduceLROnPlateau {
  private best: number;
  private bad = 0;

  /**
   * `mode` decides which direction counts as improvement — `'min'` for a loss,
   * `'max'` for an accuracy.
   *
   * **It stands second because torch puts it second, and for a while it was not
   * here at all.** Without it `new ReduceLROnPlateau(opt, 'min', 0.1)` — the way
   * torch's own documentation writes the call — put the string into `factor` and
   * `0.1` into `patience`, and nothing raised: the schedule simply cut the learning
   * rate by `NaN` at the wrong time. An argument missing from the middle of a list
   * does not fail, it answers.
   */
  constructor(
    private readonly opt: Optimizer,
    private readonly mode: "min" | "max" = "min",
    private readonly factor = 0.1,
    private readonly patience = 10,
    private readonly threshold = 1e-4,
  ) {
    if (mode !== "min" && mode !== "max") {
      throw new Error(`mode must be 'min' or 'max', got ${JSON.stringify(mode)}`);
    }
    this.best = mode === "min" ? Infinity : -Infinity;
  }

  /**
   * Whether this value is an improvement, by torch's relative rule.
   *
   * The starting value carries the first call: `Infinity * (1 - threshold)` is
   * `Infinity` and `-Infinity * (1 + threshold)` is `-Infinity`, so whatever
   * arrives first is better than it. The core writes the same idea as `best is
   * None`; this way there is one comparison rather than two.
   */
  private better(metric: number): boolean {
    return this.mode === "min"
      ? metric < this.best * (1 - this.threshold)
      : metric > this.best * (1 + this.threshold);
  }

  /**
   * The two things a resume needs. **`best` starts at infinity** — which is
   * why putting these numbers in the header needed somewhere to write
   * infinity other than `JSON.stringify` (`numbersToMeta` in
   * `serialize.ts`).
   *
   * Without restoring them, whatever value arrives right after a resume
   * becomes "the first, therefore the best", and the patience counter
   * resets.
   */
  stateDict(): Record<string, number> {
    return { best: this.best, bad: this.bad };
  }

  loadStateDict(values: Record<string, number>): void {
    if (values.best !== undefined) this.best = values.best;
    if (values.bad !== undefined) this.bad = values.bad;
  }

  step(metric: number): void {
    // torch defaults to `rel` mode — it counts as an improvement only when it improves
    // relatively by at least this much.
    if (this.better(metric)) {
      this.best = metric;
      this.bad = 0;
      return;
    }
    this.bad += 1;
    if (this.bad > this.patience) {
      // **Every group is cut.** This one does not inherit `LRScheduler` (it multiplies
      // the current value rather than the base), so it cannot use `apply` above. Cutting
      // one group leaves the rest as they were, and the per-layer learning rates drift
      // apart as the schedule goes on.
      for (const group of this.opt.paramGroups) group.lr *= this.factor;
      this.bad = 0;
    }
  }
}
