/**
 * `Tensor` — GPU 버퍼 하나와 모양, 그리고 테이프의 마디 하나.
 *
 * 레이아웃은 **NCHW**, dtype 은 **float32 하나**다. 자매(`browsertorch_webgpu`)가
 * NHWC 를 들고 int64 를 float32 에 담은 것은 TF.js 의 제약을 피한 우회였고, 우리
 * 커널에는 그 제약이 없다. 흉내 내면 이유 없이 남의 우회를 물려받는다.
 */

import { backward as tapeBackward, gradMode, type Node } from "./autograd.js";
import { Device } from "./device.js";
import {
  type AxisRule,
  BINARY,
  binaryBackward,
  binaryForward,
  cumsumBackward,
  cumulative,
  diagflat,
  diagflatBackward,
  expandDim,
  extremeBackward,
  fill,
  gather,
  gatherBackward,
  gatherIndex,
  matmul,
  reduceBroadcast,
  reduceDim,
  type ReduceKind,
  ruleKey,
  triangle,
  UNARY,
  unaryBackward,
  unaryForward,
  whereBackward,
  whereKernel,
} from "./kernels.js";

/** 장치를 **객체 안에** 둔다. `autograd.ts` 의 `gradMode` 와 같은 이유다. */
const deviceHolder: { current: Device | null } = { current: null };

export async function init(): Promise<Device> {
  if (!deviceHolder.current) deviceHolder.current = await Device.create();
  return deviceHolder.current;
}

function dev(): Device {
  const d = deviceHolder.current;
  if (!d) throw new Error("장치가 없다 — 먼저 `await init()` 을 불러라.");
  return d;
}

function numel(shape: readonly number[]): number {
  return shape.reduce((a, b) => a * b, 1);
}

/**
 * torch 의 브로드캐스팅 규칙. 오른쪽부터 맞추고, 1 은 늘어나고, 나머지는 같아야 한다.
 */
export function broadcastShapes(
  a: readonly number[],
  b: readonly number[],
): number[] {
  const rank = Math.max(a.length, b.length);
  const out: number[] = new Array<number>(rank).fill(1);
  for (let i = 0; i < rank; i++) {
    const da = a[a.length - rank + i] ?? 1;
    const db = b[b.length - rank + i] ?? 1;
    if (da !== db && da !== 1 && db !== 1) {
      throw new Error(`브로드캐스팅이 안 된다: [${a}] 와 [${b}]`);
    }
    out[i] = Math.max(da, db);
  }
  return out;
}

/**
 * `shape` 를 `out` 랭크에 오른쪽 맞춤으로 놓았을 때의 스트라이드.
 *
 * **늘어난 축은 0 이다** — 같은 값을 계속 읽는다. 실제로 복제해서 늘리면 메모리를
 * 쓰고, conv 벤치에서 im2col 이 융합 커널에 진 이유가 정확히 그것이었다.
 */
export function alignStrides(
  shape: readonly number[],
  out: readonly number[],
): number[] {
  const rank = out.length;
  const own: number[] = new Array<number>(shape.length).fill(1);
  for (let d = shape.length - 2; d >= 0; d--) {
    own[d] = (own[d + 1] ?? 1) * (shape[d + 1] ?? 1);
  }
  const strides: number[] = new Array<number>(rank).fill(0);
  for (let i = 0; i < rank; i++) {
    const src = shape.length - rank + i;
    if (src < 0) continue;
    const dim = shape[src] ?? 1;
    strides[i] = dim === 1 && (out[i] ?? 1) !== 1 ? 0 : (own[src] ?? 1);
  }
  return strides;
}

/** `shape` 를 `out` 랭크에 오른쪽 맞춤한 것 — `reduceBroadcast` 가 쓴다. */
function padShape(shape: readonly number[], rank: number): number[] {
  const out: number[] = new Array<number>(rank).fill(1);
  for (let i = 0; i < rank; i++) {
    const src = shape.length - rank + i;
    if (src >= 0) out[i] = shape[src] ?? 1;
  }
  return out;
}

export class Tensor implements Node<Tensor> {
  readonly shape: readonly number[];
  readonly size: number;
  readonly buffer: GPUBuffer;
  requiresGrad: boolean;
  grad: Tensor | null = null;
  readonly parents: readonly Tensor[];
  readonly backwardFn: ((grad: Tensor) => readonly (Tensor | null)[]) | null;
  readonly gradName: string;

  constructor(
    buffer: GPUBuffer,
    shape: readonly number[],
    options: {
      requiresGrad?: boolean;
      parents?: readonly Tensor[];
      backwardFn?: (grad: Tensor) => readonly (Tensor | null)[];
      gradName?: string;
    } = {},
  ) {
    this.buffer = buffer;
    this.shape = [...shape];
    this.size = numel(this.shape);
    this.parents = options.parents ?? [];
    this.gradName = options.gradName ?? "";
    // 부모 중 하나라도 흘리면 흘린다. no_grad 안에서는 아무도 안 흘린다.
    const inherited =
      gradMode.enabled && this.parents.some((p) => p.requiresGrad);
    this.requiresGrad = options.requiresGrad ?? inherited;
    this.backwardFn =
      this.requiresGrad && options.backwardFn ? options.backwardFn : null;
  }

  /**
   * 그래프에 마디를 하나 만든다. 코어의 `_make` 와 같은 자리다.
   *
   * `no_grad` 안이면 부모도 역방향도 안 달린다 — 달아 두고 안 쓰면 버퍼가 살아남아
   * 새지, 조용히 틀리지는 않지만 학습 루프에서는 그것도 치명적이다.
   */
  private static make(
    buffer: GPUBuffer,
    shape: readonly number[],
    parents: readonly Tensor[],
    backwardFn: (grad: Tensor) => readonly (Tensor | null)[],
    gradName: string,
  ): Tensor {
    if (!gradMode.enabled || !parents.some((p) => p.requiresGrad)) {
      return new Tensor(buffer, shape, { requiresGrad: false });
    }
    return new Tensor(buffer, shape, { parents, backwardFn, gradName });
  }

  // ── 만들기 ────────────────────────────────────────────────────────────

  static from(
    data: ArrayLike<number>,
    shape?: readonly number[],
    requiresGrad = false,
  ): Tensor {
    const flat = data instanceof Float32Array ? data : Float32Array.from(data);
    const shp = shape ?? [flat.length];
    if (numel(shp) !== flat.length) {
      throw new Error(`모양 [${shp}] 는 원소 ${flat.length}개와 안 맞는다.`);
    }
    return new Tensor(dev().upload(flat), shp, { requiresGrad });
  }

  static full(shape: readonly number[], value: number): Tensor {
    const n = numel(shape);
    const out = dev().alloc(n);
    dev().run1d(
      dev().pipeline(`fill:${n}:${value}`, () => fill(n, value)),
      [out],
      n,
    );
    return new Tensor(out, shape);
  }

  static zeros(shape: readonly number[]): Tensor {
    return Tensor.full(shape, 0);
  }

  static ones(shape: readonly number[]): Tensor {
    return Tensor.full(shape, 1);
  }

  // ── 원소별 ────────────────────────────────────────────────────────────

  unary(name: string): Tensor {
    if (!UNARY[name]) throw new Error(`모르는 단항 연산: ${name}`);
    const n = this.size;
    const out = dev().alloc(n);
    dev().run1d(
      dev().pipeline(`u:${name}:${n}`, () => unaryForward(name, n)),
      [this.buffer, out],
      n,
    );
    const result = Tensor.make(
      out,
      this.shape,
      [this],
      (g) => {
        const gi = dev().alloc(n);
        dev().run1d(
          dev().pipeline(`ub:${name}:${n}`, () => unaryBackward(name, n)),
          [this.buffer, result.buffer, g.buffer, gi],
          n,
        );
        return [new Tensor(gi, this.shape)];
      },
      `${name[0]?.toUpperCase()}${name.slice(1)}Backward0`,
    );
    return result;
  }

  binary(name: string, other: Tensor): Tensor {
    const spec = BINARY[name];
    if (!spec) throw new Error(`모르는 이항 연산: ${name}`);
    const shape = broadcastShapes(this.shape, other.shape);
    const sa = alignStrides(this.shape, shape);
    const sb = alignStrides(other.shape, shape);
    const n = numel(shape);
    const key = `${shape}|${sa}|${sb}`;
    const out = dev().alloc(n);
    dev().run1d(
      dev().pipeline(`b:${name}:${key}`, () => binaryForward(name, shape, sa, sb)),
      [this.buffer, other.buffer, out],
      n,
    );
    const result = Tensor.make(
      out,
      shape,
      [this, other],
      (g) => {
        const side = (which: "a" | "b", self: Tensor): Tensor => {
          const wide = dev().alloc(n);
          dev().run1d(
            dev().pipeline(
              `bb:${name}:${which}:${key}`,
              () => binaryBackward(name, which, shape, sa, sb),
            ),
            [this.buffer, other.buffer, result.buffer, g.buffer, wide],
            n,
          );
          const wideTensor = new Tensor(wide, shape);
          return foldTo(wideTensor, self.shape);
        };
        return [
          this.requiresGrad ? side("a", this) : null,
          other.requiresGrad ? side("b", other) : null,
        ];
      },
      `${name[0]?.toUpperCase()}${name.slice(1)}Backward0`,
    );
    return result;
  }

  add(other: Tensor): Tensor {
    return this.binary("add", other);
  }
  sub(other: Tensor): Tensor {
    return this.binary("sub", other);
  }
  mul(other: Tensor): Tensor {
    return this.binary("mul", other);
  }
  div(other: Tensor): Tensor {
    return this.binary("div", other);
  }

  // ── 행렬곱 ────────────────────────────────────────────────────────────

  /** 2차원만. 배치 행렬곱은 T1 이다 — 없는 기능이 틀린 답보다 낫다. */
  mm(other: Tensor): Tensor {
    if (this.shape.length !== 2 || other.shape.length !== 2) {
      throw new Error(
        `mm 은 2차원끼리다: [${this.shape}] × [${other.shape}]. ` +
          "배치는 아직 없다.",
      );
    }
    const M = this.shape[0] ?? 0;
    const K = this.shape[1] ?? 0;
    const K2 = other.shape[0] ?? 0;
    const N = other.shape[1] ?? 0;
    if (K !== K2) {
      throw new Error(`안쪽 차원이 다르다: [${this.shape}] × [${other.shape}]`);
    }
    const out = dev().alloc(M * N);
    dev().run(
      dev().pipeline(`mm:${M}:${K}:${N}`, () => matmul(M, K, N)),
      [this.buffer, other.buffer, out],
      [Math.ceil(N / 64), Math.ceil(M / 64), 1],
    );
    return Tensor.make(
      out,
      [M, N],
      [this, other],
      (g) => [
        this.requiresGrad ? g.mm(other.transpose()) : null,
        other.requiresGrad ? this.transpose().mm(g) : null,
      ],
      "MmBackward0",
    );
  }

  /** 2차원 전치. 지금은 실제로 옮겨 담는다 — 뷰는 T1 이다. */
  transpose(): Tensor {
    if (this.shape.length !== 2) {
      throw new Error(`transpose 는 아직 2차원만이다: [${this.shape}]`);
    }
    const M = this.shape[0] ?? 0;
    const N = this.shape[1] ?? 0;
    const out = dev().alloc(M * N);
    dev().run1d(
      dev().pipeline(`t:${M}:${N}`, () => transposeKernel(M, N)),
      [this.buffer, out],
      M * N,
    );
    return Tensor.make(out, [N, M], [this], (g) => [g.transpose()], "TBackward0");
  }

  // ── 축약 ──────────────────────────────────────────────────────────────

  /** 전부 더해 스칼라 하나로. `backward()` 의 출발점이다. */
  sum(): Tensor {
    const out = dev().sumAll(this.buffer, this.size);
    const shape = this.shape;
    return Tensor.make(
      out,
      [],
      [this],
      // d(sum)/dx 는 어디서나 1 이므로 씨앗을 모양대로 펴 준다.
      (g) => [foldFrom(g, shape)],
      "SumBackward0",
    );
  }

  /**
   * 축 하나를 접는다. `dim` 이 없으면 전부 접어 스칼라로.
   *
   * 전체 합만 `Device.sumAll` 의 트리로 간다 — 축 축약 커널은 스레드 하나가 축을
   * 훑는 구조라 전체 축약에 쓰면 스레드 하나가 n 번 돈다.
   */
  private reduceOver(kind: ReduceKind, dim?: number, keepdim = false): Tensor {
    if (dim === undefined) {
      if (kind === "sum") return this.sum();
      // 전체 최대·최소는 평평하게 본 뒤 축 하나로 접는다.
      return this.flat().reduceOver(kind, 0, false);
    }
    const rank = this.shape.length;
    const axis = dim < 0 ? dim + rank : dim;
    if (axis < 0 || axis >= rank) {
      throw new Error(`축이 범위를 벗어났다: ${dim} (랭크 ${rank})`);
    }
    const red = this.shape[axis] ?? 1;
    const outer = this.shape.slice(0, axis).reduce((a, b) => a * b, 1);
    const inner = this.shape.slice(axis + 1).reduce((a, b) => a * b, 1);
    const outShape = [...this.shape];
    if (keepdim) outShape[axis] = 1;
    else outShape.splice(axis, 1);

    const n = outer * inner;
    const out = dev().alloc(n);
    const key = `${outer}:${red}:${inner}`;
    dev().run1d(
      dev().pipeline(`rd:${kind}:${key}`, () => reduceDim(kind, outer, red, inner)),
      [this.buffer, out],
      n,
    );
    const result = Tensor.make(
      out,
      outShape,
      [this],
      (g) => {
        if (kind === "prod") {
          // out/x 로 쓰면 x 에 0 이 하나라도 있는 축에서 조용히 틀린다. 제대로 하려면
          // 접두·접미 곱을 따로 들어야 하고, 아직 그것을 요구하는 케이스가 없다.
          throw new Error("prod 의 역방향은 아직 없다 — 틀린 답보다 없는 편이 낫다.");
        }
        const gi = dev().alloc(this.size);
        if (kind === "sum") {
          dev().run1d(
            dev().pipeline(`xd:${key}`, () => expandDim(outer, red, inner)),
            [g.buffer, gi],
            this.size,
          );
        } else {
          dev().run1d(
            dev().pipeline(`eb:${key}`, () => extremeBackward(outer, red, inner)),
            [this.buffer, result.buffer, g.buffer, gi],
            this.size,
          );
        }
        return [new Tensor(gi, this.shape)];
      },
      kind === "sum" ? "SumBackward1" : "AmaxBackward0",
    );
    return result;
  }

  /** 같은 버퍼를 1차원으로 본다. 원소 순서가 그대로라 복사가 필요 없다. */
  private flat(): Tensor {
    if (this.shape.length === 1) return this;
    return Tensor.make(
      this.buffer,
      [this.size],
      [this],
      (g) => [new Tensor(g.buffer, this.shape)],
      "ViewBackward0",
    );
  }

  amax(dim?: number, keepdim = false): Tensor {
    return this.reduceOver("max", dim, keepdim);
  }

  amin(dim?: number, keepdim = false): Tensor {
    return this.reduceOver("min", dim, keepdim);
  }

  sumDim(dim: number, keepdim = false): Tensor {
    return this.reduceOver("sum", dim, keepdim);
  }

  mean(dim?: number, keepdim = false): Tensor {
    const count = dim === undefined
      ? this.size
      : (this.shape[dim < 0 ? dim + this.shape.length : dim] ?? 1);
    const total = dim === undefined ? this.sum() : this.sumDim(dim, keepdim);
    return total.div(Tensor.full([], count));
  }

  /**
   * 그래프에서 뗀 사본. 버퍼는 **공유한다** — 값을 읽기만 할 자리에 쓴다.
   *
   * `logsumexp` 가 최대값을 뗄 때 쓴다. 안 떼면 `m` 이 자기 기울기를 갖고, 수식상
   * 그 몫이 정확히 상쇄되긴 하지만 부동소수에서는 큰 것끼리 빼는 꼴이 된다.
   */
  detach(): Tensor {
    return new Tensor(this.buffer, this.shape, { requiresGrad: false });
  }

  /**
   * `log(Σ exp(x))`. **최대값을 빼고 계산한다** — 그냥 쓰면 x 가 89 를 넘는 순간
   * float32 의 exp 가 inf 가 되고 그 뒤가 전부 inf 다.
   *
   * 조립으로 둔다. 역방향은 이미 있는 연산들의 미분에서 나오므로 새 미분식을 손으로
   * 쓰지 않는다 — 그 자리가 이번 주에 가장 자주 틀린 자리였다.
   */
  logsumexp(dim?: number, keepdim = false): Tensor {
    const m = (dim === undefined ? this.amax() : this.amax(dim, true)).detach();
    const shifted = this.sub(m);
    const summed = dim === undefined
      ? shifted.exp().sum()
      : shifted.exp().sumDim(dim, true);
    const logged = summed.log().add(m);
    if (dim === undefined || keepdim) return logged;
    return logged.squeeze(dim);
  }

  /** `‖x - y‖₂`. 조립이라 역방향이 저절로 따라온다. */
  dist(other: Tensor): Tensor {
    return this.sub(other).square().sum().sqrt();
  }

  /** NaN 을 0 으로 보고 더한다. NaN 자리로는 기울기가 안 간다. */
  nansum(dim?: number, keepdim = false): Tensor {
    const clean = this.unary("nanToZero");
    return dim === undefined ? clean.sum() : clean.sumDim(dim, keepdim);
  }

  /** NaN 을 빼고 평균낸다. **개수도 NaN 을 빼고 센다** — 그것이 mean 과 다른 점이다. */
  nanmean(dim?: number, keepdim = false): Tensor {
    const total = this.nansum(dim, keepdim);
    const present = this.unary("notNan");
    const count = (dim === undefined
      ? present.sum()
      : present.sumDim(dim, keepdim)).detach();
    return total.div(count);
  }

  /** 크기 1 인 축을 뺀다. */
  squeeze(dim: number): Tensor {
    const rank = this.shape.length;
    const axis = dim < 0 ? dim + rank : dim;
    if (this.shape[axis] !== 1) {
      throw new Error(`축 ${dim} 의 크기가 1 이 아니다: [${this.shape}]`);
    }
    const outShape = [...this.shape];
    outShape.splice(axis, 1);
    const shape = this.shape;
    return Tensor.make(
      this.buffer,
      outShape,
      [this],
      (g) => [new Tensor(g.buffer, shape)],
      "SqueezeBackward0",
    );
  }

  // ── 모양 ──────────────────────────────────────────────────────────────

  /** 이 텐서의 연속 스트라이드. 모양 연산이 규칙을 짤 때 쓴다. */
  private strides(): number[] {
    const s: number[] = new Array<number>(this.shape.length).fill(1);
    for (let d = this.shape.length - 2; d >= 0; d--) {
      s[d] = (s[d + 1] ?? 1) * (this.shape[d + 1] ?? 1);
    }
    return s;
  }

  /**
   * 규칙대로 값을 모아 새 텐서를 만든다. 모양 연산이 전부 이리로 온다.
   *
   * 지금은 **실제로 옮겨 담는다.** 뷰로 두면 복사가 없어 빠르지만, 뷰가 생기는 순간
   * 제자리 연산이 어디까지 번지는지를 정해야 하고 그것은 아직 정할 때가 아니다.
   */
  private viewAs(
    rules: readonly AxisRule[],
    offset: number,
    outShape: readonly number[],
    gradName: string,
  ): Tensor {
    const n = outShape.reduce((a, b) => a * b, 1);
    const key = ruleKey(rules, offset);
    const out = dev().alloc(n);
    dev().run1d(dev().pipeline(`gt:${key}`, () => gather(rules, offset)), [this.buffer, out], n);
    const inSize = this.size;
    const inShape = this.shape;
    return Tensor.make(
      out,
      outShape,
      [this],
      (g) => {
        const gi = dev().alloc(inSize);
        dev().run1d(
          dev().pipeline(`gb:${key}|${inSize}`, () => gatherBackward(rules, offset, inSize)),
          [g.buffer, gi],
          inSize,
        );
        return [new Tensor(gi, inShape)];
      },
      gradName,
    );
  }

  /** 같은 버퍼를 다른 모양으로 본다. 원소 순서가 안 바뀌므로 커널이 필요 없다. */
  reshape(shape: readonly number[]): Tensor {
    const n = shape.reduce((a, b) => a * b, 1);
    if (n !== this.size) {
      throw new Error(`모양 [${shape}] 는 원소 ${this.size}개와 안 맞는다.`);
    }
    const from = this.shape;
    return Tensor.make(
      this.buffer,
      shape,
      [this],
      (g) => [new Tensor(g.buffer, from)],
      "ViewBackward0",
    );
  }

  ravel(): Tensor {
    return this.reshape([this.size]);
  }

  /** 축 하나를 여러 축으로 편다. */
  unflatten(dim: number, sizes: readonly number[]): Tensor {
    const rank = this.shape.length;
    const axis = dim < 0 ? dim + rank : dim;
    const shape = [...this.shape.slice(0, axis), ...sizes, ...this.shape.slice(axis + 1)];
    return this.reshape(shape);
  }

  /** 적어도 2차원으로. 이미 2차원 이상이면 그대로. */
  atleast2d(): Tensor {
    if (this.shape.length >= 2) return this;
    if (this.shape.length === 1) return this.reshape([1, this.shape[0] ?? 0]);
    return this.reshape([1, 1]);
  }

  /**
   * 크기 1 인 축을 늘린다. `-1` 은 "그대로 두라"는 뜻이다.
   *
   * 앞에 축을 더 붙일 수도 있다 — 그 축들은 걸음이 0 이라 **복제하지 않는다.**
   */
  expand(...sizes: number[]): Tensor {
    const rank = sizes.length;
    if (rank < this.shape.length) {
      throw new Error(`expand 는 축을 못 줄인다: [${this.shape}] → [${sizes}]`);
    }
    const own = this.strides();
    const rules: AxisRule[] = [];
    const outShape: number[] = [];
    for (let i = 0; i < rank; i++) {
      const src = this.shape.length - rank + i;
      const dim = src >= 0 ? (this.shape[src] ?? 1) : 1;
      const want = sizes[i] ?? -1;
      const size = want === -1 ? dim : want;
      if (want !== -1 && dim !== 1 && want !== dim) {
        throw new Error(`축 ${i} 는 ${dim} 이라 ${want} 로 못 늘린다.`);
      }
      const stride = src >= 0 && dim !== 1 ? (own[src] ?? 1) : 0;
      rules.push({ size, stride, kind: "lin", wrap: size });
      outShape.push(size);
    }
    return this.viewAs(rules, 0, outShape, "ExpandBackward0");
  }

  /** 축마다 정수 배로 되풀이한다. `expand` 와 달리 실제로 여러 벌이 된다. */
  repeat(...times: number[]): Tensor {
    const rank = times.length;
    if (rank < this.shape.length) {
      throw new Error(`repeat 는 축을 못 줄인다: [${this.shape}] → [${times}]`);
    }
    const own = this.strides();
    const rules: AxisRule[] = [];
    const outShape: number[] = [];
    for (let i = 0; i < rank; i++) {
      const src = this.shape.length - rank + i;
      const dim = src >= 0 ? (this.shape[src] ?? 1) : 1;
      const k = times[i] ?? 1;
      rules.push({
        size: dim * k,
        stride: src >= 0 ? (own[src] ?? 1) : 0,
        kind: "mod",
        wrap: dim,
      });
      outShape.push(dim * k);
    }
    return this.viewAs(rules, 0, outShape, "RepeatBackward0");
  }

  /** 축 둘을 맞바꾼다. `swapdims` 와 같다 — torch 가 이름을 둘 다 갖는다. */
  swapaxes(a: number, b: number): Tensor {
    const rank = this.shape.length;
    const i = a < 0 ? a + rank : a;
    const j = b < 0 ? b + rank : b;
    const own = this.strides();
    const order = [...Array(rank).keys()];
    order[i] = j;
    order[j] = i;
    const rules: AxisRule[] = order.map((src) => ({
      size: this.shape[src] ?? 1,
      stride: own[src] ?? 1,
      kind: "lin" as const,
      wrap: this.shape[src] ?? 1,
    }));
    return this.viewAs(rules, 0, order.map((src) => this.shape[src] ?? 1), "TransposeBackward0");
  }

  /** 축 하나에서 한 자리를 고르고 그 축을 없앤다. */
  select(dim: number, index: number): Tensor {
    const rank = this.shape.length;
    const axis = dim < 0 ? dim + rank : dim;
    const own = this.strides();
    const offset = index * (own[axis] ?? 1);
    const rules: AxisRule[] = [];
    const outShape: number[] = [];
    for (let d = 0; d < rank; d++) {
      if (d === axis) continue;
      rules.push({
        size: this.shape[d] ?? 1,
        stride: own[d] ?? 1,
        kind: "lin",
        wrap: this.shape[d] ?? 1,
      });
      outShape.push(this.shape[d] ?? 1);
    }
    return this.viewAs(rules, offset, outShape, "SelectBackward0");
  }

  /** 2차원의 대각선. `offset` 이 양수면 위쪽, 음수면 아래쪽 대각선이다. */
  diagonal(offset = 0): Tensor {
    if (this.shape.length !== 2) {
      throw new Error(`diagonal 은 아직 2차원만이다: [${this.shape}]`);
    }
    const rows = this.shape[0] ?? 0;
    const cols = this.shape[1] ?? 0;
    const own = this.strides();
    const rowStride = own[0] ?? 1;
    const colStride = own[1] ?? 1;
    const start = offset >= 0 ? offset * colStride : -offset * rowStride;
    const length = offset >= 0
      ? Math.max(0, Math.min(rows, cols - offset))
      : Math.max(0, Math.min(rows + offset, cols));
    // 한 걸음에 행과 열이 같이 하나씩 간다 — 그래서 걸음이 둘의 합이다.
    const rules: AxisRule[] = [
      { size: length, stride: rowStride + colStride, kind: "lin", wrap: length },
    ];
    return this.viewAs(rules, start, [length], "DiagonalBackward0");
  }

  /** 벡터를 대각선에 놓은 정사각 행렬. */
  diagflat(): Tensor {
    const n = this.size;
    const out = dev().alloc(n * n);
    dev().run1d(dev().pipeline(`df:${n}`, () => diagflat(n)), [this.buffer, out], n * n);
    const shape = this.shape;
    return Tensor.make(
      out,
      [n, n],
      [this],
      (g) => {
        const gi = dev().alloc(n);
        dev().run1d(
          dev().pipeline(`dfb:${n}`, () => diagflatBackward(n)),
          [g.buffer, gi],
          n,
        );
        return [new Tensor(gi, shape)];
      },
      "DiagflatBackward0",
    );
  }

  /** 축 하나를 거꾸로. */
  flip(dim: number): Tensor {
    const rank = this.shape.length;
    const axis = dim < 0 ? dim + rank : dim;
    const own = this.strides();
    const rules: AxisRule[] = this.shape.map((size, d) => ({
      size,
      stride: own[d] ?? 1,
      kind: d === axis ? ("rev" as const) : ("lin" as const),
      wrap: size,
    }));
    return this.viewAs(rules, 0, this.shape, "FlipBackward0");
  }

  fliplr(): Tensor {
    return this.flip(1);
  }

  flipud(): Tensor {
    return this.flip(0);
  }

  /**
   * 2차원 평면 안에서 90° 씩 돌린다.
   *
   * `k=1` 이면 `out[i][j] = in[j][C-1-i]` 다 — 축을 바꾸면서 한쪽을 뒤집는 것이라,
   * 규칙 표로 그대로 적힌다.
   */
  rot90(k = 1): Tensor {
    if (this.shape.length !== 2) {
      throw new Error(`rot90 은 아직 2차원만이다: [${this.shape}]`);
    }
    const turns = ((k % 4) + 4) % 4;
    if (turns === 0) return this.reshape(this.shape);
    const rows = this.shape[0] ?? 0;
    const cols = this.shape[1] ?? 0;
    const own = this.strides();
    const rowStride = own[0] ?? 1;
    const colStride = own[1] ?? 1;
    if (turns === 2) {
      const rules: AxisRule[] = [
        { size: rows, stride: rowStride, kind: "rev", wrap: rows },
        { size: cols, stride: colStride, kind: "rev", wrap: cols },
      ];
      return this.viewAs(rules, 0, [rows, cols], "Rot90Backward0");
    }
    if (turns === 1) {
      const rules: AxisRule[] = [
        { size: cols, stride: colStride, kind: "rev", wrap: cols },
        { size: rows, stride: rowStride, kind: "lin", wrap: rows },
      ];
      return this.viewAs(rules, 0, [cols, rows], "Rot90Backward0");
    }
    const rules: AxisRule[] = [
      { size: cols, stride: colStride, kind: "lin", wrap: cols },
      { size: rows, stride: rowStride, kind: "rev", wrap: rows },
    ];
    return this.viewAs(rules, 0, [cols, rows], "Rot90Backward0");
  }

  /**
   * 미끄러지는 창. 걸음이 창 크기보다 작으면 창끼리 겹친다.
   *
   * **겹치면 역방향에서 쌓인다** — 길이 5 를 `unfold(0, 3, 1)` 로 펴면 기울기가
   * `[1,2,3,2,1]` 이다. 안 더하면 전부 1 이 되고, 값 검사만으로는 안 걸린다.
   */
  unfold(dim: number, size: number, step: number): Tensor {
    const rank = this.shape.length;
    const axis = dim < 0 ? dim + rank : dim;
    const own = this.strides();
    const axisSize = this.shape[axis] ?? 0;
    const windows = Math.floor((axisSize - size) / step) + 1;
    if (windows < 1) {
      throw new Error(`창 ${size}, 걸음 ${step} 로는 길이 ${axisSize} 에서 창이 안 나온다.`);
    }
    const axisStride = own[axis] ?? 1;
    const rules: AxisRule[] = [];
    const outShape: number[] = [];
    for (let d = 0; d < rank; d++) {
      const dim_ = d === axis ? windows : (this.shape[d] ?? 1);
      const stride = d === axis ? axisStride * step : (own[d] ?? 1);
      rules.push({ size: dim_, stride, kind: "lin", wrap: dim_ });
      outShape.push(dim_);
    }
    // 창 안쪽이 맨 뒤 축으로 붙는다.
    rules.push({ size, stride: axisStride, kind: "lin", wrap: size });
    outShape.push(size);
    return this.viewAs(rules, 0, outShape, "UnfoldBackward0");
  }

  /** 축 하나를 같은 크기 조각으로 나눈다. 조각마다 새 텐서다. */
  split(dim: number, parts: number): Tensor[] {
    const rank = this.shape.length;
    const axis = dim < 0 ? dim + rank : dim;
    const axisSize = this.shape[axis] ?? 0;
    if (axisSize % parts !== 0) {
      throw new Error(`축 ${dim} 의 크기 ${axisSize} 는 ${parts} 로 안 나뉜다.`);
    }
    const each = axisSize / parts;
    const own = this.strides();
    const out: Tensor[] = [];
    for (let k = 0; k < parts; k++) {
      const rules: AxisRule[] = this.shape.map((size, d) => ({
        size: d === axis ? each : size,
        stride: own[d] ?? 1,
        kind: "lin" as const,
        wrap: d === axis ? each : size,
      }));
      const outShape = this.shape.map((size, d) => (d === axis ? each : size));
      out.push(this.viewAs(rules, k * each * (own[axis] ?? 1), outShape, "SliceBackward0"));
    }
    return out;
  }

  hsplit(parts: number): Tensor[] {
    return this.split(1, parts);
  }

  vsplit(parts: number): Tensor[] {
    return this.split(0, parts);
  }

  /** 축을 원하는 자리로 옮긴다. `swapaxes` 와 달리 나머지 순서를 지킨다. */
  movedim(src: number, dst: number): Tensor {
    const rank = this.shape.length;
    const from = src < 0 ? src + rank : src;
    const to = dst < 0 ? dst + rank : dst;
    const order = [...Array(rank).keys()].filter((d) => d !== from);
    order.splice(to, 0, from);
    return this.permute(order);
  }

  /** 축 순서를 통째로 바꾼다. */
  permute(order: readonly number[]): Tensor {
    const own = this.strides();
    const rules: AxisRule[] = order.map((s) => ({
      size: this.shape[s] ?? 1,
      stride: own[s] ?? 1,
      kind: "lin" as const,
      wrap: this.shape[s] ?? 1,
    }));
    return this.viewAs(rules, 0, order.map((s) => this.shape[s] ?? 1), "PermuteBackward0");
  }

  /** 축 하나에서 `start` 부터 `length` 개만. */
  narrow(dim: number, start: number, length: number): Tensor {
    const rank = this.shape.length;
    const axis = dim < 0 ? dim + rank : dim;
    const own = this.strides();
    const rules: AxisRule[] = this.shape.map((size, d) => ({
      size: d === axis ? length : size,
      stride: own[d] ?? 1,
      kind: "lin" as const,
      wrap: d === axis ? length : size,
    }));
    const outShape = this.shape.map((size, d) => (d === axis ? length : size));
    return this.viewAs(rules, start * (own[axis] ?? 1), outShape, "SliceBackward0");
  }

  /**
   * 축 하나를 자리이동. **끝에서 빠진 것이 앞으로 돌아온다.**
   *
   * `out[i] = in[(i - shift) mod n]` 이라, 규칙 표의 `mod` 에 자리이동을 얹으면 된다.
   */
  roll(shift: number, dim = 0): Tensor {
    const rank = this.shape.length;
    const axis = dim < 0 ? dim + rank : dim;
    const size = this.shape[axis] ?? 1;
    const own = this.strides();
    const bias = ((-shift % size) + size) % size;
    const rules: AxisRule[] = this.shape.map((s, d) => ({
      size: s,
      stride: own[d] ?? 1,
      kind: d === axis ? ("mod" as const) : ("lin" as const),
      wrap: s,
      ...(d === axis ? { bias } : {}),
    }));
    return this.viewAs(rules, 0, this.shape, "RollBackward0");
  }

  /** `repeat` 과 같은 일이되 torch 가 이름을 둘 다 갖는다. */
  tile(...times: number[]): Tensor {
    return this.repeat(...times);
  }

  /** 축 하나를 크기로 나눈다. `split` 은 조각 **크기**, `chunk` 는 조각 **수**다. */
  splitSize(dim: number, size: number): Tensor[] {
    const rank = this.shape.length;
    const axis = dim < 0 ? dim + rank : dim;
    const axisSize = this.shape[axis] ?? 0;
    const out: Tensor[] = [];
    for (let start = 0; start < axisSize; start += size) {
      out.push(this.narrow(axis, start, Math.min(size, axisSize - start)));
    }
    return out;
  }

  chunk(parts: number, dim = 0): Tensor[] {
    const rank = this.shape.length;
    const axis = dim < 0 ? dim + rank : dim;
    const axisSize = this.shape[axis] ?? 0;
    return this.splitSize(axis, Math.ceil(axisSize / parts));
  }

  /** 축 하나를 따라 낱개로 뜯는다. 그 축이 사라진다. */
  unbind(dim = 0): Tensor[] {
    const rank = this.shape.length;
    const axis = dim < 0 ? dim + rank : dim;
    const size = this.shape[axis] ?? 0;
    return Array.from({ length: size }, (_, i) => this.select(axis, i));
  }

  /** 아래 삼각. `diagonal` 위쪽을 0 으로 만든다. */
  tril(diagonal = 0): Tensor {
    return this.triangleAs(true, diagonal);
  }

  triu(diagonal = 0): Tensor {
    return this.triangleAs(false, diagonal);
  }

  private triangleAs(lower: boolean, diagonal: number): Tensor {
    if (this.shape.length !== 2) {
      throw new Error(`tril/triu 는 2차원이다: [${this.shape}]`);
    }
    const rows = this.shape[0] ?? 0;
    const cols = this.shape[1] ?? 0;
    const n = rows * cols;
    const key = `${lower ? "l" : "u"}:${rows}:${cols}:${diagonal}`;
    const out = dev().alloc(n);
    dev().run1d(
      dev().pipeline(`tri:${key}`, () => triangle(rows, cols, lower, diagonal)),
      [this.buffer, out],
      n,
    );
    const shape = this.shape;
    return Tensor.make(
      out,
      shape,
      [this],
      // 남긴 자리로만 흐른다. 0 으로 만든 자리는 결과에 안 들어갔다.
      (g) => {
        const gi = dev().alloc(n);
        dev().run1d(
          dev().pipeline(`tri:${key}`, () => triangle(rows, cols, lower, diagonal)),
          [g.buffer, gi],
          n,
        );
        return [new Tensor(gi, shape)];
      },
      lower ? "TrilBackward0" : "TriuBackward0",
    );
  }

  /** 대각선의 합. */
  trace(): Tensor {
    return this.diagonal().sum();
  }

  /** 2차원이면 대각선을 뽑고, 1차원이면 대각선에 놓는다 — torch 의 `diag` 다. */
  diag(): Tensor {
    return this.shape.length === 2 ? this.diagonal() : this.diagflat();
  }

  /** 축 하나를 누적한다. */
  cumsum(dim = 0): Tensor {
    return this.scan("sum", dim);
  }

  cumprod(dim = 0): Tensor {
    return this.scan("prod", dim);
  }

  private scan(kind: "sum" | "prod", dim: number): Tensor {
    const rank = this.shape.length;
    const axis = dim < 0 ? dim + rank : dim;
    const len = this.shape[axis] ?? 1;
    const outer = this.shape.slice(0, axis).reduce((a, b) => a * b, 1);
    const inner = this.shape.slice(axis + 1).reduce((a, b) => a * b, 1);
    const n = this.size;
    const key = `${outer}:${len}:${inner}`;
    const out = dev().alloc(n);
    dev().run1d(
      dev().pipeline(`cum:${kind}:${key}`, () => cumulative(kind, outer, len, inner)),
      [this.buffer, out],
      n,
    );
    const shape = this.shape;
    return Tensor.make(
      out,
      shape,
      [this],
      (g) => {
        if (kind !== "sum") {
          // 0 이 섞인 곱의 역방향은 나눗셈으로 못 쓴다. 제대로 하려면 접두·접미
          // 곱을 따로 들어야 하고, 아직 그것을 요구하는 케이스가 없다.
          throw new Error("cumprod 의 역방향은 아직 없다 — 틀린 답보다 없는 편이 낫다.");
        }
        const gi = dev().alloc(n);
        dev().run1d(
          dev().pipeline(`cumb:${key}`, () => cumsumBackward(outer, len, inner)),
          [g.buffer, gi],
          n,
        );
        return [new Tensor(gi, shape)];
      },
      kind === "sum" ? "CumsumBackward0" : "CumprodBackward0",
    );
  }

  /** 축 하나를 색인 텐서가 가리키는 대로 고른다. 색인은 float32 에 담겨 온다. */
  gather(dim: number, index: Tensor): Tensor {
    const rank = this.shape.length;
    const axis = dim < 0 ? dim + rank : dim;
    const outer = this.shape.slice(0, axis).reduce((a, b) => a * b, 1);
    const inner = this.shape.slice(axis + 1).reduce((a, b) => a * b, 1);
    const axisSize = this.shape[axis] ?? 1;
    const outAxis = index.shape[axis] ?? 1;
    const n = index.size;
    const out = dev().alloc(n);
    dev().run1d(
      dev().pipeline(
        `gi:${outer}:${axisSize}:${inner}:${outAxis}`,
        () => gatherIndex(outer, axisSize, inner, outAxis),
      ),
      [this.buffer, index.buffer, out],
      n,
    );
    // 값만 낸다. 역방향은 흩뿌리기라 결정적으로 쓰려면 따로 세워야 하고,
    // 그것을 요구하는 케이스가 아직 없다.
    return new Tensor(out, index.shape);
  }

  /** 조건 자리마다 이쪽 아니면 저쪽. torch 의 메서드 형태는 `x.where(조건, 다른쪽)` 이다. */
  where(cond: Tensor, other: Tensor): Tensor {
    const n = this.size;
    const out = dev().alloc(n);
    dev().run1d(
      dev().pipeline(`wh:${n}`, () => whereKernel(n)),
      [cond.buffer, this.buffer, other.buffer, out],
      n,
    );
    const shape = this.shape;
    const side = (g: Tensor, take: "a" | "b"): Tensor => {
      const gi = dev().alloc(n);
      dev().run1d(
        dev().pipeline(`whb:${take}:${n}`, () => whereBackward(n, take)),
        [cond.buffer, g.buffer, gi],
        n,
      );
      return new Tensor(gi, shape);
    };
    return Tensor.make(
      out,
      shape,
      [this, other],
      (g) => [
        this.requiresGrad ? side(g, "a") : null,
        other.requiresGrad ? side(g, "b") : null,
      ],
      "WhereBackward0",
    );
  }

  /** 전부 곱한다. */
  prod(dim?: number, keepdim = false): Tensor {
    if (dim === undefined) return this.flat().reduceOver("prod", 0, false);
    return this.reduceOver("prod", dim, keepdim);
  }

  /** L2 노름. */
  norm(): Tensor {
    return this.square().sum().sqrt();
  }

  /** 두 벡터의 안쪽 곱. */
  dot(other: Tensor): Tensor {
    return this.mul(other).sum();
  }

  /** 두 벡터의 바깥 곱. 브로드캐스팅으로 나온다 — 새 커널이 필요 없다. */
  outer(other: Tensor): Tensor {
    return this.reshape([this.size, 1]).mul(other.reshape([1, other.size]));
  }

  /** 위아래로 자른다. */
  clamp(low: number, high: number): Tensor {
    return this.binary("maximum", Tensor.full([], low))
      .binary("minimum", Tensor.full([], high));
  }

  /** 상수 지수. */
  powScalar(k: number): Tensor {
    return this.binary("pow", Tensor.full([], k));
  }

  /** `exp(x) / Σ exp(x)`. **최대값을 빼고 계산한다** — 안 그러면 큰 값에서 넘친다. */
  softmax(dim = 0): Tensor {
    const m = this.amax(dim, true).detach();
    const e = this.sub(m).exp();
    return e.div(e.sumDim(dim, true));
  }

  // ── 역전파 ────────────────────────────────────────────────────────────

  backward(): void {
    if (this.size !== 1) {
      throw new Error(
        `backward 는 스칼라에서만 부른다. 지금 모양은 [${this.shape}] 다 — ` +
          ".sum() 을 먼저 불러라.",
      );
    }
    tapeBackward<Tensor>(this, Tensor.full([], 1), (a, b) => a.add(b));
  }

  // ── 읽기 ──────────────────────────────────────────────────────────────

  async toArray(): Promise<Float32Array> {
    return dev().read(this.buffer, this.size);
  }

  /**
   * 모양과 값이 **정확히** 같은가. 허용 오차가 없다 — 그것이 `allclose` 와 다른 점이다.
   *
   * GPU 에서 읽어 온다. 판정 하나를 위해 왕복하는 것이 아깝지만, CPU 에 사본을 들고
   * 있다가 비교하면 GPU 에서 무슨 일이 있었는지를 못 본다.
   */
  async equal(other: Tensor): Promise<boolean> {
    if (this.shape.length !== other.shape.length ||
        this.shape.some((d, i) => d !== other.shape[i])) {
      return false;
    }
    const [a, b] = await Promise.all([this.toArray(), other.toArray()]);
    return a.every((v, i) => v === b[i]);
  }

  /** 허용 오차 안에서 같은가. torch 의 기본값과 같다. */
  async allclose(other: Tensor, rtol = 1e-5, atol = 1e-8): Promise<boolean> {
    if (this.shape.length !== other.shape.length ||
        this.shape.some((d, i) => d !== other.shape[i])) {
      return false;
    }
    const [a, b] = await Promise.all([this.toArray(), other.toArray()]);
    return a.every((v, i) => {
      const w = b[i] ?? Number.NaN;
      return Math.abs(v - w) <= atol + rtol * Math.abs(w);
    });
  }

  async item(): Promise<number> {
    if (this.size !== 1) {
      throw new Error(`item 은 원소 하나짜리에서만이다: [${this.shape}]`);
    }
    const arr = await this.toArray();
    return arr[0] ?? Number.NaN;
  }
}

/** 넓은 기울기를 목표 모양으로 접는다. 모양이 이미 같으면 그대로 둔다. */
function foldTo(wide: Tensor, target: readonly number[]): Tensor {
  if (numel(wide.shape) === numel(target) && wide.shape.length === target.length) {
    return new Tensor(wide.buffer, target);
  }
  const small = padShape(target, wide.shape.length);
  const n = numel(small);
  const out = dev().alloc(n);
  dev().run1d(
    dev().pipeline(
      `rb:${wide.shape}|${small}`,
      () => reduceBroadcast(wide.shape, small),
    ),
    [wide.buffer, out],
    n,
  );
  return new Tensor(out, target);
}

/** 스칼라 기울기를 모양대로 편다. `sum` 의 역방향이다. */
function foldFrom(g: Tensor, shape: readonly number[]): Tensor {
  const n = numel(shape);
  const out = dev().alloc(n);
  dev().run1d(
    dev().pipeline(`bcast1:${n}`, () => broadcastScalar(n)),
    [g.buffer, out],
    n,
  );
  return new Tensor(out, shape);
}

/** 스칼라 하나를 n 칸에 뿌린다. */
function broadcastScalar(n: number): string {
  return `
@group(0) @binding(0) var<storage, read> S: array<f32>;
@group(0) @binding(1) var<storage, read_write> Out: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let gid = g.y * ${Math.min(Math.max(1, Math.ceil(n / 64)), 65535) * 64}u + g.x;
  if (gid >= ${n}u) { return; }
  Out[gid] = S[0];
}`;
}

/** 2차원 전치 커널. 모양이 상수라 나눗셈이 안 남는다. */
function transposeKernel(M: number, N: number): string {
  const n = M * N;
  return `
@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read_write> Out: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let gid = g.y * ${Math.min(Math.max(1, Math.ceil(n / 64)), 65535) * 64}u + g.x;
  if (gid >= ${n}u) { return; }
  let r = gid / ${N}u;
  let c = gid % ${N}u;
  Out[c * ${M}u + r] = A[gid];
}`;
}

/** 표에 있는 단항을 전부 메서드로 단다. 이름을 두 번 적지 않는다. */
for (const name of Object.keys(UNARY)) {
  Object.defineProperty(Tensor.prototype, name, {
    value: function (this: Tensor): Tensor {
      return this.unary(name);
    },
    writable: true,
    configurable: true,
  });
}

/** 표에서 단 메서드들의 타입. 위 루프와 짝이고, 하나만 고치면 어긋난다. */
export interface Tensor {
  neg(): Tensor;
  abs(): Tensor;
  exp(): Tensor;
  log(): Tensor;
  sqrt(): Tensor;
  rsqrt(): Tensor;
  square(): Tensor;
  reciprocal(): Tensor;
  sin(): Tensor;
  cos(): Tensor;
  tan(): Tensor;
  sinh(): Tensor;
  cosh(): Tensor;
  tanh(): Tensor;
  asin(): Tensor;
  acos(): Tensor;
  atan(): Tensor;
  asinh(): Tensor;
  acosh(): Tensor;
  atanh(): Tensor;
  exp2(): Tensor;
  log2(): Tensor;
  log10(): Tensor;
  expm1(): Tensor;
  log1p(): Tensor;
  relu(): Tensor;
  sigmoid(): Tensor;
  sign(): Tensor;
  floor(): Tensor;
  ceil(): Tensor;
  round(): Tensor;
  trunc(): Tensor;
  frac(): Tensor;
}

export { noGrad } from "./autograd.js";
