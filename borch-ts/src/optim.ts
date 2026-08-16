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
import { device, keepAlive, noGrad, Tensor } from "./tensor.js";

export interface ParamGroup {
  lr: number;
  /** 스케줄러가 기준으로 삼는 값. 처음 스케줄러가 한 번만 찍는다. */
  initialLr?: number;
}

/**
 * **제자리로 고칠 버퍼는 자기 것이어야 한다.**
 *
 * `Tensor.full` 은 원소가 하나면 **값으로 캐시한 버퍼를 돌려준다**(`scalarCache`).
 * 학습 루프에서 `x * 0.5` 가 매 스텝 같은 상수를 만드는 것을 막으려는 것이고 거기서는
 * 옳다 — 아무도 그 버퍼에 안 쓰기 때문이다.
 *
 * 옵티마이저 상태는 다르다. `copyFrom` 으로 제자리에 쓰므로, 크기 1 파라미터에서는
 * **프로그램 전체가 공유하는 그 상수를 덮어쓴다.** 예외도 경고도 없고, 그때부터
 * `Tensor.full([], lr)` 이 다른 값을 낸다 — 원인에서 아주 먼 자리에서 틀리기 시작한다.
 *
 * `Tensor.zeros`·`ones` 도 `full` 을 지나므로 같은 자리다. **상태를 만드는 길은 전부
 * 여기를 지나야 한다** — 처음에 `Composed.state()` 만 고쳤는데 `SGD`·`Adam`·`RMSprop`
 * 은 전용 커널을 써서 그 밑동 밖이라 안 닿았다. "크기가 1 일 때만 조심" 같은 규칙으로
 * 두면 잊으므로, 만드는 자리를 하나로 모은다.
 */
function ownedBuffer(shape: readonly number[], value = 0): Tensor {
  const n = shape.reduce((a, b) => a * b, 1);
  return keepAlive(Tensor.from(new Float32Array(n).fill(value), shape));
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
  private readonly buffers: Tensor[];

  constructor(
    params: Tensor[],
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
      : params.map((p) => ownedBuffer(p.shape));
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
    d.run1d(
      d.pipeline(`sgd:${n}:${this.lr}:${this.momentum}:${this.weightDecay}`,
        () => sgdStep(n, this.lr, this.momentum, this.weightDecay)),
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
    params: Tensor[],
    lr: number,
    private readonly beta1 = 0.9,
    private readonly beta2 = 0.999,
    private readonly eps = 1e-8,
  ) {
    super(params, lr);
    // **둘이 같은 버퍼가 되면 안 된다.** 캐시를 타면 크기 1 파라미터에서 `m` 과 `v`
    // 가 같은 자리를 가리키고, WebGPU 가 "writable storage buffer aliasing" 으로
    // 명령 버퍼째 거절한다 — 이쪽은 조용하지 않고 시끄럽게 죽는 갈래다(실측).
    this.first = params.map((p) => ownedBuffer(p.shape));
    this.second = params.map((p) => ownedBuffer(p.shape));
  }

  override step(): void {
    // 편향 보정이 스텝 수에 걸리므로 파라미터마다가 아니라 한 번만 센다.
    this.stepCount += 1;
    super.step();
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
    params: Tensor[],
    lr: number,
    private readonly alpha = 0.99,
    private readonly eps = 1e-8,
  ) {
    super(params, lr);
    this.squares = params.map((p) => ownedBuffer(p.shape));
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
  protected state(shapes: Tensor[]): Tensor[] {
    return shapes.map((p) => ownedBuffer(p.shape));
  }

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

  constructor(params: Tensor[], lr = 0.01, private readonly lrDecay = 0,
              private readonly eps = 1e-10) {
    super(params, lr);
    this.sums = this.state(params);
  }

  override step(): void {
    this.stepCount += 1;
    super.step();
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

  constructor(params: Tensor[], lr = 1.0, private readonly rho = 0.9,
              private readonly eps = 1e-6) {
    super(params, lr);
    this.squares = this.state(params);
    this.deltas = this.state(params);
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

  constructor(params: Tensor[], lr = 2e-3, private readonly beta1 = 0.9,
              private readonly beta2 = 0.999, private readonly eps = 1e-8) {
    super(params, lr);
    this.first = this.state(params);
    this.inf = this.state(params);
  }

  override step(): void {
    this.stepCount += 1;
    super.step();
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

  constructor(params: Tensor[], lr = 2e-3, private readonly beta1 = 0.9,
              private readonly beta2 = 0.999, private readonly eps = 1e-8,
              private readonly momentumDecay = 4e-3) {
    super(params, lr);
    this.first = this.state(params);
    this.second = this.state(params);
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

  constructor(params: Tensor[], lr = 1e-3, private readonly beta1 = 0.9,
              private readonly beta2 = 0.999, private readonly eps = 1e-8) {
    super(params, lr);
    this.first = this.state(params);
    this.second = this.state(params);
  }

  override step(): void {
    this.stepCount += 1;
    super.step();
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

  constructor(params: Tensor[], lr = 1e-2, private readonly lambd = 1e-4,
              private readonly alpha = 0.75, private readonly t0 = 1e6,
              private readonly weightDecay = 0) {
    super(params, lr);
    this.ax = this.state(params);
    this.eta = lr;
  }

  override step(): void {
    this.stepCount += 1;
    super.step();
    this.eta = this.lr / (1 + this.lambd * this.lr * this.stepCount) ** this.alpha;
    this.mu = 1 / Math.max(1, this.stepCount - this.t0);
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

  constructor(params: Tensor[], lr = 1e-2, private readonly etaMinus = 0.5,
              private readonly etaPlus = 1.2, private readonly sizeMin = 1e-6,
              private readonly sizeMax = 50) {
    super(params, lr);
    this.prev = this.state(params);
    this.stepSize = params.map((p) => ownedBuffer(p.shape, lr));
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

  constructor(params: Tensor[], lr = 1e-2, private readonly beta2Decay = -0.8,
              private readonly eps1: number | null = null,
              private readonly eps2 = 1e-3, private readonly d = 1.0,
              private readonly weightDecay = 0) {
    super(params, lr);
    for (const p of params) {
      const rank = p.shape.length;
      if (rank > 1) {
        const rows = [...p.shape.slice(0, -1), 1];
        const cols = [...p.shape.slice(0, -2), 1, p.shape[rank - 1] ?? 1];
        // 행·열 은행은 파라미터 모양이 아니라 **접은 모양**이라 `state()` 를 못 쓴다.
        // 캐시를 피하는 것은 같은 이유다 — `(1, 1)` 이면 원소가 하나다.
        this.rowVar.push(ownedBuffer(rows));
        this.colVar.push(ownedBuffer(cols));
        this.variance.push(null);
      } else {
        this.rowVar.push(null);
        this.colVar.push(null);
        this.variance.push(ownedBuffer(p.shape));
      }
    }
  }

  override step(): void {
    this.stepCount += 1;
    super.step();
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
  readonly base: number;

  constructor(protected readonly opt: Optimizer) {
    const group = opt.paramGroups[0];
    if (group && group.initialLr === undefined) group.initialLr = group.lr;
    this.base = group?.initialLr ?? 0;
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
    const group = this.opt.paramGroups[0];
    if (group) group.lr = this.compute(0);
    return this;
  }

  step(): void {
    this.epoch += 1;
    const group = this.opt.paramGroups[0];
    if (group) group.lr = this.compute(this.epoch);
  }

  /** 지금 학습률. 재귀식 스케줄러가 자기 앞의 값을 읽는 자리다. */
  protected get current(): number {
    return this.opt.paramGroups[0]?.lr ?? 0;
  }

  /** `SequentialLR` 이 넘어갈 때 처음부터 다시 밟게 한다. */
  restart(): void {
    const group = this.opt.paramGroups[0];
    if (group) group.lr = this.base;
    this.start();
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
