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
  BINARY,
  binaryBackward,
  binaryForward,
  fill,
  matmul,
  reduceBroadcast,
  UNARY,
  unaryBackward,
  unaryForward,
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
