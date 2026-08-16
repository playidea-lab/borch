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

import { adamStep, rmspropStep, sgdStep } from "./kernels.js";
import { RuntimeError } from "./errors.js";
import { device, keepAlive, noGrad, Tensor } from "./tensor.js";

/**
 * 파라미터 묶음 하나와 거기 걸린 하이퍼파라미터.
 *
 * **전에는 `params` 가 없었다.** 이름은 torch 의 `param_groups` 인데 언제나
 * `[{ lr }]` 하나였고 스케줄러가 전부 `[0]` 만 봤다 — 층별 학습률도, bias·norm 에
 * weight decay 를 빼는 것도 안 되는데 이름은 된다고 말하고 있었다. 모양만 torch 인
 * 것이 없는 것보다 나쁘다: 쓰는 사람이 `paramGroups.push(...)` 를 쓰면 조용히
 * 무시됐다.
 */
export interface ParamGroup {
  /** 이 그룹이 밟는 파라미터. */
  params: Tensor[];
  lr: number;
  /** 스케줄러가 기준으로 삼는 값. 처음 스케줄러가 한 번만 찍는다. */
  initialLr?: number;
  /** 이 그룹만 다른 값을 쓸 때. 없으면 옵티마이저를 세울 때 준 값이다. */
  weightDecay?: number;
}

/** 그룹을 만들 때 넣는 것. `lr` 을 비우면 옵티마이저의 기본값을 쓴다. */
export interface ParamGroupInit {
  params: readonly Tensor[];
  lr?: number;
  weightDecay?: number;
}

/** 옵티마이저 생성자가 받는 것 — 텐서 목록이거나 그룹 목록이다. */
export type ParamsArg = readonly Tensor[] | readonly ParamGroupInit[];

function isGroups(arg: ParamsArg): arg is readonly ParamGroupInit[] {
  return arg.length > 0 && !(arg[0] instanceof Tensor);
}

export abstract class Optimizer {
  /** torch 와 같은 모양 — 스케줄러가 여기 `lr` 을 고친다. */
  readonly paramGroups: ParamGroup[] = [];

  /** 모든 그룹의 파라미터를 이어 붙인 것. 상태 은행이 이 자리로 색인된다. */
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
   * 그룹을 나중에 더한다. `torch.optim.Optimizer.add_param_group` 자리다.
   *
   * **상태 은행도 같이 늘린다.** 안 늘리면 다음 스텝에서 "파라미터 N 의 상태가
   * 없다" 로 터진다 — 은행이 평평한 파라미터 자리로 색인되기 때문이다.
   *
   * 이미 세워 둔 스케줄러는 이 그룹의 기준값을 모른다. torch 도 같은 자리에서
   * `initial_lr` 을 요구한다 — 스케줄러를 먼저 세웠다면 다시 세워라.
   */
  addParamGroup(init: ParamGroupInit): void {
    const group = this.attach(init);
    this.extendState(group.params);
  }

  /**
   * 상태 은행 하나. **여기를 지나야 그룹이 늘 때 같이 는다.**
   *
   * 옛날에는 옵티마이저마다 `params.map(...)` 으로 직접 만들었고, 그러면 나중에
   * 더해진 파라미터의 자리가 비어 있게 된다.
   *
   * @param value 채울 값. `Rprop` 의 걸음 크기만 0 이 아니다.
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
   * 그룹이 늘 때 상태를 같이 늘린다.
   *
   * 기본은 등록된 은행마다 **파라미터 모양** 자리를 더한다. 그 전제가 안 맞는
   * 옵티마이저(`Adafactor` 는 행·열로 접은 분산을 든다)는 여기를 덮어쓴다.
   */
  protected extendState(added: readonly Tensor[]): void {
    for (const { slots, value } of this.banks) {
      for (const p of added) slots.push(keepAlive(Tensor.owned(p.shape, value)));
    }
  }

  /** 지금 밟고 있는 그룹의 학습률. */
  protected get lr(): number {
    return this.paramGroups[this.currentGroup]?.lr ?? this.defaultLr;
  }

  /** 지금 그룹이 따로 정한 값, 없으면 옵티마이저의 기본값. */
  protected grouped(fallback: number): number {
    return this.paramGroups[this.currentGroup]?.weightDecay ?? fallback;
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
        // 어느 그룹의 파라미터인지 먼저 정해야 `this.lr` 이 맞는 값을 준다.
        this.currentGroup = this.groupOf[i] ?? 0;
        this.update(i, p, g);
      }
      this.currentGroup = 0;
    });
  }

  /**
   * 이 옵티마이저가 든 스칼라. 하위 클래스가 덮어쓴다.
   *
   * **은행은 여기 안 들어간다** — 밑동이 이미 들고 있으므로 `stateDict` 가 알아서
   * 담는다. 여기 적을 것은 `stepCount` 처럼 텐서가 아닌 것들이고, 빠뜨리면 재개했을
   * 때 **편향 보정이 처음부터 다시 시작한다.** 값이 조금씩 다르게 나오는데 어디서
   * 갈렸는지 가리키는 것이 없는 종류다.
   */
  protected counters(): Record<string, number> {
    return {};
  }

  /** 위의 되돌림. 덮어쓴 쪽이 자기 것만 꺼내 간다. */
  protected loadCounters(_values: Record<string, number>): void {
    /* 스칼라가 없는 옵티마이저는 할 일이 없다 */
  }

  /**
   * 학습을 이어서 하려면 있어야 하는 전부.
   *
   * **모멘텀만으로는 부족하다.** 그룹의 학습률(스케줄러가 이미 깎아 놓았을 수 있다)과
   * 스텝 계수기가 같이 가야 재개한 다음 스텝이 안 끊고 돌린 것과 같은 수를 낸다.
   *
   * 텐서와 수를 갈라 준다 — safetensors 의 몸에는 텐서가, 머리에는 수가 실린다
   * (`serialize.ts`).
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
   * `stateDict()` 가 준 것을 되돌린다. **옵티마이저를 같은 인자로 다시 세운 뒤** 부른다.
   *
   * 은행의 자리가 없거나 크기가 다르면 던진다. 파라미터가 늘거나 준 모델에 옛
   * 체크포인트를 얹으면 값이 한 칸씩 밀린 채로 학습이 돌 수 있고, 그것은 예외보다
   * 훨씬 나쁘다.
   */
  loadStateDict(state: {
    tensors: Record<string, Tensor>;
    numbers: Record<string, number>;
  }): void {
    noGrad(() => {
      for (const [b, bank] of this.banks.entries()) {
        for (const [i, slot] of bank.slots.entries()) {
          const saved = state.tensors[`bank${b}.${i}`];
          if (!saved) throw new RuntimeError(`체크포인트에 bank${b}.${i} 가 없다`);
          if (saved.size !== slot.size) {
            throw new RuntimeError(
              `bank${b}.${i} 의 크기가 다르다: 저장 ${saved.size}, 지금 ${slot.size}`,
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
      if (!buf) throw new Error(`SGD: 파라미터 ${index} 의 버퍼가 없다`);
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

  constructor(
    params: ParamsArg,
    lr: number,
    private readonly beta1 = 0.9,
    private readonly beta2 = 0.999,
    private readonly eps = 1e-8,
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
    if (!m || !v) throw new Error(`Adam: 파라미터 ${index} 의 상태가 없다`);
    // 편향 보정은 스텝마다 달라지므로 셰이더에 굽지 않고 작은 버퍼로 넘긴다.
    const corr = Tensor.from([
      1 - this.beta1 ** this.stepCount,
      1 - this.beta2 ** this.stepCount,
    ], [2]);
    const n = param.size;
    const d = device();
    d.run1d(
      d.pipeline(`adam:${n}:${this.lr}:${this.beta1}:${this.beta2}:${this.eps}`,
        () => adamStep(n, this.lr, this.beta1, this.beta2, this.eps)),
      [param.buffer, grad.buffer, m.buffer, v.buffer, corr.buffer],
      n,
    );
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
    if (!sq) throw new Error(`RMSprop: 파라미터 ${index} 의 상태가 없다`);
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
    if (!got) throw new Error(`${what}: 파라미터 ${index} 의 상태가 없다`);
    return got;
  }

  /** 상수 하나를 텐서로. 스칼라와의 연산은 브로드캐스팅으로 붙는다. */
  protected k(v: number): Tensor {
    return Tensor.full([], v);
  }
}

/** 기울기 제곱을 **계속 더한다** — 줄기만 하고 안 는다. */
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

/** **학습률이 거의 안 쓰인다.** 보폭을 갱신량의 이력에서 스스로 만든다. */
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

/** Adam 의 2차 모멘트를 **제곱평균 대신 최댓값**으로 둔 것. */
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
 * Adam 에 네스테로프의 앞보기를 붙인 것.
 *
 * **모멘텀 계수가 스텝마다 바뀌고, 그 수열의 누적곱을 들고 다녀야 한다.** 상수로
 * 두면 초반 몇 스텝이 조용히 갈린다.
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
 * Adam 인데 **초반에는 적응 보폭을 안 쓴다.**
 *
 * 2차 모멘트의 표본이 적을 때 분산이 커서 초반이 튀는 것이 Adam 의 알려진 성질이고,
 * 이쪽은 그 구간을 SGD 처럼 지나간다. 경계(`rho > 5`)를 빼면 값이 Adam 과 같아진다.
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
 * 평균 내는 SGD. **걸음마다 학습률이 스스로 줄고**, 어느 시점부터 파라미터를 평균낸다.
 *
 * `eta` 는 `lr / (1 + lambd·lr·step)^alpha` 로 줄고 `mu` 가 평균의 무게다. 기본 `t0`
 * 이 100만이라 보통 학습에서는 `mu` 가 늘 1 이고 `ax` 는 파라미터의 사본이다 —
 * 평균 갈래는 `t0` 을 낮춰야 실제로 돈다.
 *
 * **감쇠가 곱셈이다.** `param *= (1 − lambd·eta)` 를 먼저 하고 그다음 기울기를 뺀다.
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
 * 기울기의 **부호만** 본다. 크기는 안 쓰고 걸음 폭을 칸마다 따로 키우고 줄인다.
 *
 * 부호가 그대로면 폭에 `etaPlus` 를 곱하고, 뒤집히면 `etaMinus` 를 곱한다.
 * **뒤집힌 칸은 그 걸음을 아예 안 간다** — 기울기를 0 으로 만들어 두고, 그래서 다음
 * 걸음의 "이전 기울기" 도 0 이 된다. 그 둘이 없으면 부호가 안 바뀌는 입력으로는
 * 영원히 안 걸리는 차이가 생긴다.
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
 * Adam 인데 2차 모멘트를 **행과 열로 쪼개 든다.**
 *
 * Adam 은 파라미터마다 분산을 하나씩 들어 기억이 가중치만큼 든다. 여기서는 행 평균과
 * 열 평균만 들고 그 바깥곱으로 되살린다 — `(R, C)` 자리에 `R + C` 만 쓴다.
 *
 * **1 차원 파라미터는 안 쪼갠다** — 쪼갤 축이 하나뿐이라 그냥 분산을 든다. 벡터로만
 * 물으면 이 최적화의 요점이 통째로 안 돌아간다.
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
   * **밑동의 기본을 못 쓴다.** 이 옵티마이저의 상태는 파라미터 모양이 아니라 행·열로
   * 접은 것이고, 랭크에 따라 어느 쪽을 드는지도 갈린다. 자리 수는 파라미터와 맞으므로
   * (안 쓰는 쪽은 `null` 로 채운다) 색인은 그대로 통한다.
   *
   * 생성자와 `addParamGroup` 이 같은 이 자리를 지난다 — 둘로 나눠 적으면 언젠가
   * 한쪽만 고친다.
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
      if (!row || !col) throw new Error("Adafactor: 행·열 상태가 없다");
      const sq = grad.square();
      row.copyFrom(row.add(sq.mean(rank - 1, true).sub(row).mul(blend)));
      col.copyFrom(col.add(sq.mean(rank - 2, true).sub(col).mul(blend)));
      // `(…, R, 1) × (…, 1, C)` 는 바깥곱이다 — 브로드캐스팅이 그대로 해 준다.
      const outer = row.mul(col);
      variance = outer.div(row.mean(rank - 2, true).binary("maximum", this.k(one)));
    } else {
      const v = this.variance[index];
      if (!v) throw new Error("Adafactor: 분산 상태가 없다");
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
 * 학습률 스케줄.
 *
 * **실수 연산뿐이라 torch 와 값이 그대로 같아야 한다** — 근사가 낄 자리가 없다.
 * 그래서 골든이 한 값이 아니라 **궤적 전체**를 굳혔고, 코어가 그렇게 하다가 `StepLR`
 * 의 차이를 잡았다.
 *
 * **기준은 `initialLr` 이지 세울 때의 lr 이 아니다.** 옵티마이저에 한 번만 찍히고,
 * 나중에 세워지는 스케줄러들도 같은 기준을 본다. 혼자 쓰면 둘이 같아서 안 걸리는데,
 * 이어 붙이면 두 번째 것이 첫 번째가 이미 깎아 둔 값을 기준으로 잡는다.
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
   * **0 번째 에폭을 적용한다.** torch 는 이것을 생성자에서 하는데 여기서는 못 한다 —
   * TypeScript 의 하위 클래스 필드가 `super()` 가 끝난 **뒤에** 채워지므로, 생성자
   * 안에서 `compute` 를 부르면 `factor` 같은 것이 아직 `undefined` 다.
   *
   * 그래서 세운 직후 한 번 부른다. `ConstantLR` 처럼 0 번째부터 값을 바꾸는 것은
   * 이것이 없으면 첫 항이 통째로 갈린다 — 실제로 최대차 2.0e-01 이었다.
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

  /** 지금 학습률. 재귀식 스케줄러가 자기 앞의 값을 읽는 자리다. */
  protected get current(): number {
    return this.opt.paramGroups[0]?.lr ?? 0;
  }

  /** `SequentialLR` 이 넘어갈 때 처음부터 다시 밟게 한다. */
  restart(): void {
    this.apply(this.base);
    this.start();
  }

  /**
   * 재개하려면 있어야 하는 것 — **몇 번째 에폭인가와 기준값들.**
   *
   * 옵티마이저의 `lr` 만 되돌리면 안 된다. 스케줄러는 `epoch` 에서 값을 다시 계산하는
   * 물건이라, 그것을 0 으로 두고 이으면 **다음 `step()` 이 학습률을 처음 값으로
   * 되돌려 놓는다** — 옵티마이저 쪽은 멀쩡히 복원됐는데 한 스텝 만에 지워진다.
   *
   * 기준값도 같이 간다. `initialLr` 은 옵티마이저에서 오지만 그것은 **처음 세운
   * 스케줄러가 찍은 값**이고, 이어 붙인 스케줄러들은 자기가 본 기준을 따로 든다.
   */
  stateDict(): Record<string, number> {
    const out: Record<string, number> = { epoch: this.epoch };
    for (const [i, base] of this.bases.entries()) out[`base${i}`] = base;
    return out;
  }

  /** `stateDict()` 의 되돌림. 학습률은 안 건드린다 — 옵티마이저 쪽이 든다. */
  loadStateDict(values: Record<string, number>): void {
    if (values.epoch !== undefined) this.epoch = values.epoch;
    for (const i of this.bases.keys()) {
      const base = values[`base${i}`];
      if (base !== undefined) this.bases[i] = base;
    }
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

/**
 * **재귀식이다** — 지금 학습률에 곱한다. 원래 학습률에서 다시 세지 않는다.
 *
 * 혼자 쓰면 두 방식이 같은 수열을 낸다. 갈리는 것은 다른 스케줄러가 같은 lr 을 함께
 * 만질 때다 — `ChainedScheduler` 로 겹치면 재귀식은 서로의 결과 위에 쌓이고 닫힌
 * 식은 남이 한 일을 덮어쓴다.
 */
export class ExponentialLR extends LRScheduler {
  constructor(opt: Optimizer, private readonly gamma: number) {
    super(opt);
  }

  protected override compute(epoch: number): number {
    return epoch === 0 ? this.current : this.current * this.gamma;
  }
}

/** `totalIters` 까지 **깎아 두었다가 원래대로 돌아온다.** 워밍업의 가장 단순한 꼴. */
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
 * 시작 배율에서 끝 배율까지 **직선으로** 옮겨간다.
 *
 * `ConstantLR` 과 끝에서 만난다 — `totalIters` 를 지나면 둘 다 원래 학습률이다.
 * 마지막 값만 보면 둘을 못 가르므로 골든이 자취를 통째로 묻는다.
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

/** `(1 − t/T)^power` 로 내린다. `power=1` 이면 직선이다. */
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

/** **곱해 나간다** — 기준이 원래 학습률이 아니라 지금 학습률이다. */
export class MultiplicativeLR extends LRScheduler {
  constructor(opt: Optimizer, private readonly fn: (epoch: number) => number) {
    super(opt);
  }

  protected override compute(epoch: number): number {
    return epoch === 0 ? this.base : this.current * this.fn(epoch);
  }
}

/** 코사인으로 내리다가 **처음으로 되돌린다.** 주기가 `tMult` 배씩 길어진다. */
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
 * 올렸다가 내린다. **현대 학습 레시피의 기본값에 가깝다.**
 *
 * 초기 학습률은 `maxLr/divFactor` 이고 끝은 그것을 다시 나눈 값이라, **옵티마이저에
 * 준 학습률이 아예 안 쓰인다** — 세우는 순간 덮어쓴다.
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
 * **그룹이 여럿이면 경계가 상대값이 된다.** torch 는 `base_lr`·`max_lr` 을 그룹마다
 * 목록으로 받는데 여기서는 수 하나다. 밑동의 규칙대로 첫 그룹 기준으로 계산한 값에
 * 각 그룹의 기준 비율이 곱해지므로, 그룹 i 는 `baseLr·rᵢ` 와 `maxLr·rᵢ` 사이를 돈다.
 * 층별 학습률을 준 사람이 기대하는 쪽이지만, torch 와 **같은 것은 아니다.**
 * 그룹이 하나면 비율이 1 이라 차이가 없다.
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

/** 스케줄러를 **이어 붙인다.** 이정표에 닿으면 다음 것으로 넘어간다. */
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

/** 여럿을 **동시에** 건다. 각자의 배율이 곱해진다. */
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

  /**
   * 재개에 필요한 둘. **`best` 가 무한대에서 시작한다** — 그래서 이 수들을 머리에
   * 실을 때 `JSON.stringify` 대신 무한대를 따로 적는 자리가 필요했다
   * (`serialize.ts` 의 `numbersToMeta`).
   *
   * 안 되돌리면 재개 직후 어떤 값이 와도 "처음이라 최고" 가 되어 참을성이 초기화된다.
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
