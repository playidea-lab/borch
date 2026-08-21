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

  /** 파라미터 자리 → 그룹 번호. */
  private readonly groupOf: number[] = [];

  /** `step` 이 도는 동안 지금 밟고 있는 그룹. `lr` 이 이것을 본다. */
  private currentGroup = 0;

  /**
   * `state()` 가 만든 은행들. 그룹이 늘면 여기 전부에 자리를 더한다.
   *
   * 채운 값을 같이 든다 — `Rprop` 의 걸음 크기는 0 이 아니라 `lr` 에서 시작하고,
   * 나중에 더한 파라미터도 같은 자리에서 출발해야 한다.
   */
  private readonly banks: { slots: Tensor[]; value: number }[] = [];

  constructor(params: ParamsArg, private readonly defaultLr: number) {
    const groups: readonly ParamGroupInit[] = isGroups(params)
      ? params
      : [{ params: params as readonly Tensor[] }];
    for (const g of groups) this.attach(g);
  }

  /** 그룹 하나를 붙이고 평평한 목록에 이어 붙인다. */
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
    // **`Tensor.zeros`·`Tensor.full` 을 쓰면 안 된다.** 원소 하나짜리는 값으로
    // 캐시되어 있어 같은 버퍼가 오고, 옵티마이저는 상태에 **쓴다** — `Adam` 의 m·v 가
    // 겹쳐 명령 버퍼가 통째로 무효가 되고, `SGD` 의 모멘텀과 `Rprop` 의 걸음 크기는
    // 프로그램 전체가 쓰는 상수를 덮어쓴다. `owned` 는 자기 버퍼를 준다(`tensor.ts`).
    //
    // **만드는 자리를 하나로 모으는 것이 요점이다.** 이 결함을 처음 고칠 때 한쪽은
    // `Composed.state()` 만 고쳤는데 `SGD`·`Adam`·`RMSprop` 은 전용 커널을 써서 그
    // 밑동 밖이라 안 닿았고, 고쳤다고 적힌 채로 셋이 남아 있었다. "크기가 1 일 때만
    // 조심" 같은 규칙으로 두면 다음 옵티마이저에서 또 잊는다 — 실제로 `Rprop` 이
    // 그렇게 들어왔다.
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
        // 어느 그룹의 파라미터인지 먼저 정해야 `this.lr` 이 맞는 값을 준다.
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
    /* 스칼라가 없는 옵티마이저는 할 일이 없다 */
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
    // **상태를 미리 잡는다.** 스텝마다 새 텐서를 만들면 구역이 닫힐 때 그것이 놓이고,
    // 다음 스텝의 버퍼가 사라진 자리를 가리킨다. torch 도 상태를 붙박이로 든다.
    //
    // 0 에서 시작해도 torch 와 값이 같다: 첫 스텝의 `0·momentum + grad` 가 torch 의
    // `buf = grad.clone()` 과 같은 수다.
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
    // **그룹이 따로 정했으면 그것을 쓴다.** bias·norm 을 weight decay 에서 빼는 것이
    // 이 자리의 대표 용도다 — 그것 하나 때문에 그룹이 필요하다.
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
    // 편향 보정이 스텝 수에 걸리므로 파라미터마다가 아니라 한 번만 센다.
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
    // 편향 보정은 스텝마다 달라지므로 셰이더에 굽지 않고 작은 버퍼로 넘긴다.
    const corr = Tensor.from([
      1 - this.beta1 ** this.stepCount,
      1 - this.beta2 ** this.stepCount,
    ], [2]);

    // **커널은 안 건드린다.** 두 감쇠가 다 텐서 연산으로 적히기 때문이다.
    //
    // 붙은 쪽은 모멘트가 보기 전의 기울기에 더한다 — `g + λ·p`.
    //
    // 떨어진 쪽은 갱신 **뒤에** 원래 가중치에 걸린다: `p − lr·m̂/(√v̂+ε) − lr·λ·p`.
    // 그것은 `p·(1 − lr·λ) − lr·m̂/(√v̂+ε)` 와 같은 수이므로, **먼저 줄여 놓고**
    // 여느 때처럼 한 걸음 가면 된다. 모멘트는 기울기에서 나오므로 미리 줄인 것이
    // 그쪽을 흔들지 않는다.
    //
    // 커널에 인자를 하나 더 굽는 쪽도 됐지만, 그러면 `weightDecay` 값마다 파이프라인이
    // 하나씩 는다 — 굽는 이름에 그 수가 들어가기 때문이다.
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
  ) {
    super(params, lr);
    this.squares = this.state(this.params);
  }

  protected override update(index: number, param: Tensor, grad: Tensor): void {
    const sq = this.squares[index];
    if (!sq) throw new Error(`RMSprop: no state for parameter ${index}`);
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
 * 상태를 텐서로 들고 **텐서 연산으로** 갱신하는 옵티마이저의 밑동.
 *
 * `SGD`·`Adam`·`RMSprop` 은 전용 WGSL 커널을 쓴다 — 세 개가 학습 루프의 거의 전부라
 * 융합할 값어치가 있었다. 나머지는 그렇지 않으므로 있는 연산으로 조립한다. 맞는
 * 것이 먼저이고, 커널로 굽는 것은 **재보고** 필요할 때 할 일이다.
 *
 * 갱신한 값은 `copyFrom` 으로 **제자리에 되쓴다.** 새 텐서로 갈아끼우면 모델이 든
 * 손잡이와 옵티마이저가 든 손잡이가 갈려서 학습은 도는데 파라미터가 안 움직인다.
 */

abstract class Composed extends Optimizer {
  // `state()` 는 밑동으로 올라갔다 — `SGD`·`Adam`·`RMSprop` 도 은행을 등록해야
  // `addParamGroup` 이 그것들을 같이 늘린다.

  protected at(bank: Tensor[], index: number, what: string): Tensor {
    const got = bank[index];
    if (!got) throw new Error(`${what}: no state for parameter ${index}`);
    return got;
  }

  /** 상수 하나를 텐서로. 스칼라와의 연산은 브로드캐스팅으로 붙는다. */
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

  constructor(params: ParamsArg, lr = 0.01, private readonly lrDecay = 0,
              private readonly eps = 1e-10) {
    super(params, lr);
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

  protected override update(index: number, param: Tensor, grad: Tensor): void {
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
              private readonly eps = 1e-6) {
    super(params, lr);
    this.squares = this.state(this.params);
    this.deltas = this.state(this.params);
  }

  protected override update(index: number, param: Tensor, grad: Tensor): void {
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
              private readonly beta2 = 0.999, private readonly eps = 1e-8) {
    super(params, lr);
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

  protected override update(index: number, param: Tensor, grad: Tensor): void {
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

  constructor(params: ParamsArg, lr = 2e-3, private readonly beta1 = 0.9,
              private readonly beta2 = 0.999, private readonly eps = 1e-8,
              private readonly momentumDecay = 4e-3) {
    super(params, lr);
    this.first = this.state(this.params);
    this.second = this.state(this.params);
  }

  override step(): void {
    this.stepCount += 1;
    const t = this.stepCount;
    this.mu = this.beta1 * (1 - 0.5 * 0.96 ** (t * this.momentumDecay));
    this.muNext = this.beta1 * (1 - 0.5 * 0.96 ** ((t + 1) * this.momentumDecay));
    // 누적곱은 파라미터마다가 아니라 스텝마다 한 번 는다.
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

  protected override update(index: number, param: Tensor, grad: Tensor): void {
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
              private readonly beta2 = 0.999, private readonly eps = 1e-8) {
    super(params, lr);
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

  protected override update(index: number, param: Tensor, grad: Tensor): void {
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
              private readonly weightDecay = 0) {
    super(params, lr);
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
    const g = this.weightDecay
      ? grad.add(param.mul(this.k(this.weightDecay))) : grad;
    param.copyFrom(param.mul(this.k(1 - this.lambd * eta)).sub(g.mul(this.k(eta))));
    // `mu` 가 1 이면 평균이 아니라 **사본**이다 — 더하면 두 배가 된다.
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
    // **`Tensor.full` 로 만들면 안 된다.** 크기 1 파라미터에서 그것은 값으로 캐시된
    // 전역 `lr` 상수를 돌려주는데, 아래 `update` 가 `size.copyFrom(...)` 으로 거기에
    // 쓴다 — 프로그램 전체의 그 상수가 학습 중에 바뀐다. 예외는 안 난다.
    this.stepSize = this.state(this.params, lr);
  }

  protected override update(index: number, param: Tensor, grad: Tensor): void {
    const prev = this.at(this.prev, index, "Rprop");
    const size = this.at(this.stepSize, index, "Rprop");
    const sign = grad.mul(prev).sign();
    // **참·거짓 표를 실수로 되돌린다.** 비교는 bool 을 내고 borch.ts 는 bool 로 셈하는
    // 것을 거절한다 — 값이 0/1 이라 그냥 될 것 같지만 그 거절이 맞다.
    const rising = sign.binary("gt", this.k(0)).to("float32");
    const falling = sign.binary("lt", this.k(0)).to("float32");
    // 부호가 0 이면 1 을 곱한다 — 첫 걸음과, 앞서 뒤집혀 0 으로 만든 칸이 그렇다.
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

  /** 파라미터의 기울기를 **한 줄로 이어** 본다. 없는 자리는 0 이다. */
  private flatGrad(): Tensor {
    return Tensor.cat(
      this.params.map((p) => (p.grad ?? Tensor.zeros(p.shape)).reshape([p.size])), 0);
  }

  /** 한 줄짜리 방향을 파라미터 모양으로 잘라 더한다. */
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
        // 두 겹 되돌이 — 헤세 역행렬을 안 만들고 방향만 낸다.
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
        // 마지막 되돌이에서는 다시 안 잰다 — torch 도 그렇다.
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
              private readonly weightDecay = 0) {
    super(params, lr);
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
        // 제자리로 갱신되므로 캐시를 안 타는 `owned` 여야 한다.
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
    // f32 의 기계 입실론. torch 는 dtype 에서 가져오고 여기는 float32 하나뿐이다.
    const one = this.eps1 ?? 1.1920928955078125e-7;
    const step = this.stepCount;
    const blend = this.k(step ** this.beta2Decay);
    const rho = Math.min(this.lr, 1 / Math.sqrt(step));
    // **`alpha` 와 `denom` 을 수로 못 뺀다** — WebGPU 에 동기 읽기가 없어서 GPU 위의
    // 값을 여기서 읽으면 기다려야 한다. 스칼라 **텐서**로 남겨 곱한다. 코어는 numpy 라
    // 수로 계산하지만 식은 같다.
    const norm = param.square().sum().sqrt();
    const alpha = norm.div(this.k(Math.sqrt(param.size)))
      .binary("maximum", this.k(this.eps2)).mul(this.k(rho));
    if (this.weightDecay) {
      param.copyFrom(param.mul(this.k(1 - this.lr * this.weightDecay)));
    }
    const rank = grad.shape.length;
    let variance: Tensor;
    if (rank > 1) {
      const row = this.rowVar[index];
      const col = this.colVar[index];
      if (!row || !col) throw new Error("Adafactor: the row/column state is missing");
      const sq = grad.square();
      row.copyFrom(row.add(sq.mean(rank - 1, true).sub(row).mul(blend)));
      col.copyFrom(col.add(sq.mean(rank - 2, true).sub(col).mul(blend)));
      // `(…, R, 1) × (…, 1, C)` 는 바깥곱이다 — 브로드캐스팅이 그대로 해 준다.
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
  /** 그룹마다의 기준값. 하위 클래스가 세는 `base` 는 첫 그룹의 것이다. */
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
   * 계산한 값을 **모든 그룹에** 적용한다.
   *
   * 그룹마다 기준이 다르면 그 비율을 지킨다 — torch 가 `base_lrs` 를 그룹마다 들고
   * 각자에게서 다시 계산하는 것과 같은 결과다. `compute` 는 첫 그룹의 기준으로 한
   * 값만 내므로 여기서 비율을 곱한다.
   *
   * **그룹이 하나면 비율이 1 이라 옛 동작과 한 비트도 안 다르다.** 궤적을 통째로
   * 굳혀 둔 골든이 그것을 본다.
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
    // torch 의 셈 그대로 — `pctStart × totalSteps − 1` 이지
    // `pctStart × (totalSteps − 1)` 이 아니다.
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
 * 학습률을 **오르내리게** 한다. 안장점을 빠져나오라고 일부러 흔드는 방식이다.
 *
 * `stepSizeUp` 만큼 올라갔다가 `stepSizeDown` 만큼 내려온다. 안 주면 올라간 만큼
 * 내려온다 — **오르내림이 같으면 그 인자가 있는지도 안 보인다.**
 *
 * `mode` 셋 중 `expRange` 만 기준이 **주기가 아니라 걸음**이다. 거기가 갈리는 자리다.
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
    // 올라가는 구간과 내려오는 구간의 기울기가 다르다.
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
  private best = Infinity;
  private bad = 0;

  constructor(
    private readonly opt: Optimizer,
    private readonly factor = 0.1,
    private readonly patience = 10,
    private readonly threshold = 1e-4,
  ) {}

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
    // torch 의 기본은 `rel` 모드다 — 상대적으로 이만큼은 좋아져야 나아진 것으로 센다.
    if (metric < this.best * (1 - this.threshold)) {
      this.best = metric;
      this.bad = 0;
      return;
    }
    if (this.best === Infinity) this.best = metric;
    this.bad += 1;
    if (this.bad > this.patience) {
      // **그룹 전부를 깎는다.** 이쪽은 `LRScheduler` 를 안 물려받아서(기준값이 아니라
      // 지금 값에 곱한다) 위의 `apply` 를 못 쓴다. 한 그룹만 깎으면 나머지가 그대로
      // 남아 층별 학습률이 스케줄을 지날수록 어긋난다.
      for (const group of this.opt.paramGroups) group.lr *= this.factor;
      this.bad = 0;
    }
  }
}
