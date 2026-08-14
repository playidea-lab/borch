/**
 * WGSL 커널을 **연산 표에서 생성한다.**
 *
 * ## 왜 표에서 생성하는가
 *
 * 자매 라이브러리를 만들면서 새 미분식을 손으로 여러 번 썼고, 그때마다 틀릴 자리를
 * 하나씩 만들었다. 콜레스키 역방향은 쓰기 전에 torch 와 대조해서 살았고,
 * `roll`·`masked_select` 는 값만 맞고 그래프가 끊긴 채로 골든 746건을 통과했다.
 * 이름·순방향식·역방향식을 **한 줄로 적고 커널이 나오게** 하면 그 자리가 줄어든다.
 *
 * ## 모양을 상수로 굽는다
 *
 * 나눗셈 제수를 유니폼으로 두면 컴파일러가 강도 축소를 못 하고, GPU 에는 정수 나눗셈
 * 하드웨어가 없다. 그것만으로 conv 가 TF.js 대비 43% → 284% 로 갈렸다
 * (실측, `tests/browser/wgsl_conv.js`). 그래서 모양이 셰이더 문자열에 들어가고,
 * **모양 서명 → 파이프라인 캐시**가 라이브러리 구조가 된다. 최적화가 아니다.
 */

/** 원소별 단항 — `fwd` 는 `x` 로 쓰고, `bwd` 는 **기울기에 곱할 것**이다. */
export interface UnarySpec {
  /** 순방향 WGSL 식. 입력은 `x`. */
  readonly fwd: string;
  /** 도함수 WGSL 식. `x` 는 입력, `o` 는 순방향 결과다 — 다시 안 센다. */
  readonly bwd: string;
  /** 식이 부르는 보조 함수의 WGSL 정의. 한 줄로 안 되는 것만 여기 온다. */
  readonly prelude?: string;
}

/** 원소별 이항 — `da`·`db` 는 각 입력으로 가는 기울기에 곱할 것이다. */
export interface BinarySpec {
  readonly fwd: string;
  readonly da: string;
  readonly db: string;
  readonly prelude?: string;
}

/**
 * erf 계열의 보조 함수.
 *
 * Abramowitz & Stegun 7.1.26 이고 코어(`browsertorch/_ops.py`)와 **같은 계수**다.
 * 셋이 다른 근사를 쓰면 값이 갈릴 때 구현이 갈린 것인지 근사가 갈린 것인지 못 가른다.
 *
 * 원형은 `erfc_pos` 다 — 다항식 × exp(-y²) 라 뺄셈이 없어서 꼬리에서 자릿수가 안 난다.
 *
 * **원점 근처는 코어와 다르게 간다.** 코어는 `1 - erfc_pos(|x|)` 를 float64 로 계산해
 * 상쇄를 피하는데, WGSL 에는 f64 가 없다. 그래서 |x| < 0.5 는 급수로 답한다 —
 * 그 구간에서 다음 항이 4e-7 이라 이 프로젝트의 허용 오차(1e-4) 한참 아래다.
 * f32 로 그냥 빼면 코어가 실측으로 확인한 그 자리(4.6만 점 중 5,124 점)가 되살아난다.
 */
/**
 * NaN 판정.
 *
 * **`x != x` 는 여기서 안 통했다.** WGSL 에 `isNan` 내장이 없어 그것을 썼는데,
 * `nansum` 이 NaN 을 그대로 더하고 `nanmean` 의 개수가 3 대신 4 로 나왔다 —
 * 셰이더 컴파일러가 부동소수 비교를 NaN 없는 것으로 접었다는 뜻이다.
 *
 * 지수부가 전부 1 이고 가수부가 0 이 아니면 NaN 이다. 비트로 보면 접힐 여지가 없다.
 */
const NAN_PRELUDE = `
fn is_nan(x: f32) -> bool {
  let b = bitcast<u32>(x);
  return (b & 0x7f800000u) == 0x7f800000u && (b & 0x007fffffu) != 0u;
}`;

const ERF_PRELUDE = `
fn erfc_pos(y: f32) -> f32 {
  let t = 1.0 / (1.0 + 0.3275911 * y);
  let poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741
           + t * (-1.453152027 + t * 1.061405429))));
  return poly * exp(-y * y);
}
fn erf_(x: f32) -> f32 {
  let a = abs(x);
  if (a < 0.5) {
    let z = x * x;
    return x * 1.1283791670955126 * (1.0 - z * (0.3333333333333333
         - z * (0.1 - z * (0.023809523809523808 - z * 0.004629629629629629))));
  }
  return sign(x) * (1.0 - erfc_pos(a));
}
fn erfc_(x: f32) -> f32 {
  // x >= 0 은 원형을 그대로 쓴다 — 뺄셈이 아예 없다.
  return select(2.0 - erfc_pos(abs(x)), erfc_pos(x), x >= 0.0);
}`;

/**
 * 도함수가 없는 것(`sign`·`floor` 같은 계단)은 `bwd: "0.0"` 이다.
 *
 * **그래프를 끊지 않는다.** torch 는 0 을 흘리고, 거절과 0 은 다르다 — 자매에서
 * 이것을 한 번 틀렸고, 계단을 낀 손실이 torch 에서는 도는데 우리에게서는 멈췄다.
 */
export const UNARY: Readonly<Record<string, UnarySpec>> = {
  neg: { fwd: "-x", bwd: "-1.0" },
  abs: { fwd: "abs(x)", bwd: "sign(x)" },
  exp: { fwd: "exp(x)", bwd: "o" },
  log: { fwd: "log(x)", bwd: "1.0 / x" },
  sqrt: { fwd: "sqrt(x)", bwd: "0.5 / o" },
  rsqrt: { fwd: "inverseSqrt(x)", bwd: "-0.5 * o / x" },
  square: { fwd: "x * x", bwd: "2.0 * x" },
  reciprocal: { fwd: "1.0 / x", bwd: "-o * o" },
  sin: { fwd: "sin(x)", bwd: "cos(x)" },
  cos: { fwd: "cos(x)", bwd: "-sin(x)" },
  tan: { fwd: "tan(x)", bwd: "1.0 + o * o" },
  sinh: { fwd: "sinh(x)", bwd: "cosh(x)" },
  cosh: { fwd: "cosh(x)", bwd: "sinh(x)" },
  tanh: { fwd: "tanh(x)", bwd: "1.0 - o * o" },
  asin: { fwd: "asin(x)", bwd: "inverseSqrt(1.0 - x * x)" },
  acos: { fwd: "acos(x)", bwd: "-inverseSqrt(1.0 - x * x)" },
  atan: { fwd: "atan(x)", bwd: "1.0 / (1.0 + x * x)" },
  asinh: { fwd: "asinh(x)", bwd: "inverseSqrt(x * x + 1.0)" },
  acosh: { fwd: "acosh(x)", bwd: "inverseSqrt(x * x - 1.0)" },
  atanh: { fwd: "atanh(x)", bwd: "1.0 / (1.0 - x * x)" },
  exp2: { fwd: "exp2(x)", bwd: "o * 0.6931471805599453" },
  log2: { fwd: "log2(x)", bwd: "1.0 / (x * 0.6931471805599453)" },
  log10: { fwd: "log(x) * 0.4342944819032518", bwd: "0.4342944819032518 / x" },
  expm1: { fwd: "exp(x) - 1.0", bwd: "o + 1.0" },
  log1p: { fwd: "log(1.0 + x)", bwd: "1.0 / (1.0 + x)" },
  relu: { fwd: "max(x, 0.0)", bwd: "step(0.0, x)" },
  sigmoid: { fwd: "1.0 / (1.0 + exp(-x))", bwd: "o * (1.0 - o)" },
  // 계단 — 0 을 흘린다. torch 가 그렇다.
  sign: { fwd: "sign(x)", bwd: "0.0" },
  floor: { fwd: "floor(x)", bwd: "0.0" },
  ceil: { fwd: "ceil(x)", bwd: "0.0" },
  round: { fwd: "round(x)", bwd: "0.0" },
  trunc: { fwd: "trunc(x)", bwd: "0.0" },
  frac: { fwd: "x - trunc(x)", bwd: "1.0" },
  deg2rad: { fwd: "x * 0.017453292519943295", bwd: "0.017453292519943295" },
  rad2deg: { fwd: "x * 57.29577951308232", bwd: "57.29577951308232" },
  positive: { fwd: "x", bwd: "1.0" },
  logit: { fwd: "log(x / (1.0 - x))", bwd: "1.0 / (x * (1.0 - x))" },
  // sinc(0) 은 1 이고 그 자리의 도함수는 0 이다. 0 으로 나누는 자리라 갈라 쓴다 —
  // WGSL 은 0/0 에서 NaN 을 내고, NaN 은 자기 자신과도 달라 대조가 통과할 수 없다.
  sinc: {
    fwd: "select(sin(3.141592653589793 * x) / (3.141592653589793 * x), 1.0, x == 0.0)",
    bwd:
      "select((cos(3.141592653589793 * x) - o) / x, 0.0, x == 0.0)",
  },
  erf: { fwd: "erf_(x)", bwd: "1.1283791670955126 * exp(-x * x)", prelude: ERF_PRELUDE },
  erfc: { fwd: "erfc_(x)", bwd: "-1.1283791670955126 * exp(-x * x)", prelude: ERF_PRELUDE },
  // sgn 은 실수에서 sign 과 같다. 별칭이지만 torch 가 둘 다 가지므로 이름을 남긴다.
  sgn: { fwd: "sign(x)", bwd: "0.0" },
  // 참·거짓을 0/1 로 낸다. dtype 이 float32 하나뿐이라 bool 을 따로 안 든다.
  // **-0.0 은 여기서 거짓이다** — torch 는 참으로 본다. 지금 케이스에 -0.0 이 없어
  // 안 갈리지만, 갈리는 날이 오면 이 줄이 원인이다.
  signbit: { fwd: "select(0.0, 1.0, x < 0.0)", bwd: "0.0" },
  // 둘 다 torch 의 공개 이름이 아니라 `nansum`·`nanmean` 을 조립하는 조각이다.
  nanToZero: {
    fwd: "select(x, 0.0, is_nan(x))",
    // NaN 자리로는 안 흘린다. 그 자리는 합에 안 들어갔으므로 0 이 맞다.
    bwd: "select(1.0, 0.0, is_nan(x))",
    prelude: NAN_PRELUDE,
  },
  notNan: { fwd: "select(1.0, 0.0, is_nan(x))", bwd: "0.0", prelude: NAN_PRELUDE },
  isnan: { fwd: "select(0.0, 1.0, is_nan(x))", bwd: "0.0", prelude: NAN_PRELUDE },
  // 무한대 판정도 비트로 본다 — 지수부가 전부 1 이고 가수부가 0 이다.
  isinf: {
    fwd: "select(0.0, 1.0, (bitcast<u32>(x) & 0x7fffffffu) == 0x7f800000u)",
    bwd: "0.0",
  },
  isfinite: {
    fwd: "select(0.0, 1.0, (bitcast<u32>(x) & 0x7f800000u) != 0x7f800000u)",
    bwd: "0.0",
  },
};

export const BINARY: Readonly<Record<string, BinarySpec>> = {
  add: { fwd: "x + y", da: "1.0", db: "1.0" },
  sub: { fwd: "x - y", da: "1.0", db: "-1.0" },
  mul: { fwd: "x * y", da: "y", db: "x" },
  div: { fwd: "x / y", da: "1.0 / y", db: "-x / (y * y)" },
  pow: { fwd: "pow(x, y)", da: "y * pow(x, y - 1.0)", db: "o * log(x)" },
  maximum: { fwd: "max(x, y)", da: "step(y, x)", db: "step(x, y)" },
  minimum: { fwd: "min(x, y)", da: "step(x, y)", db: "step(y, x)" },
  atan2: {
    fwd: "atan2(x, y)",
    da: "y / (x * x + y * y)",
    db: "-x / (x * x + y * y)",
  },
  hypot: { fwd: "sqrt(x * x + y * y)", da: "x / o", db: "y / o" },
  // 부호만 옮긴다. y 로는 안 흐른다 — 부호는 계단이다.
  copysign: {
    fwd: "select(-abs(x), abs(x), y >= 0.0)",
    da: "select(-sign(x), sign(x), y >= 0.0)",
    db: "0.0",
  },
  // **안정형으로 쓴다.** log(exp x + exp y) 를 그대로 쓰면 x 가 89 를 넘는 순간
  // float32 의 exp 가 inf 가 되고, 그 뒤 결과가 전부 inf 다. 큰 쪽을 빼내면 안 넘친다.
  logaddexp: {
    // WGSL 에 log1p 내장이 없다 — log(1+t) 로 적는다.
    fwd: "max(x, y) + log(1.0 + exp(-abs(x - y)))",
    da: "1.0 / (1.0 + exp(y - x))",
    db: "1.0 / (1.0 + exp(x - y))",
  },
  logaddexp2: {
    fwd: "max(x, y) + log2(1.0 + exp2(-abs(x - y)))",
    da: "1.0 / (1.0 + exp2(y - x))",
    db: "1.0 / (1.0 + exp2(x - y))",
  },
  // **x 가 0 이면 y 와 무관하게 0 이다.** y 가 0 이어도 그렇다 — 그러라고 있는 함수이고,
  // 그 자리를 안 보면 `x * log(y)` 와 구별이 안 된다.
  xlogy: {
    fwd: "select(x * log(y), 0.0, x == 0.0)",
    da: "log(y)",
    db: "x / y",
  },
  // x<0 이면 0, x>0 이면 1, x==0 이면 y 를 그대로. 계단이라 x 로는 안 흐른다.
  heaviside: {
    fwd: "select(select(0.0, 1.0, x > 0.0), y, x == 0.0)",
    da: "0.0",
    db: "select(0.0, 1.0, x == 0.0)",
  },
  ldexp: { fwd: "x * exp2(y)", da: "exp2(y)", db: "o * 0.6931471805599453" },
  // 비교는 0/1 을 낸다. dtype 이 float32 하나뿐이라 bool 을 따로 안 든다 —
  // 골든의 bool 케이스와는 0/1 로 맞는다. **기울기는 양쪽 다 0 이다.**
  eq: { fwd: "select(0.0, 1.0, x == y)", da: "0.0", db: "0.0" },
  ne: { fwd: "select(0.0, 1.0, x != y)", da: "0.0", db: "0.0" },
  lt: { fwd: "select(0.0, 1.0, x < y)", da: "0.0", db: "0.0" },
  le: { fwd: "select(0.0, 1.0, x <= y)", da: "0.0", db: "0.0" },
  gt: { fwd: "select(0.0, 1.0, x > y)", da: "0.0", db: "0.0" },
  ge: { fwd: "select(0.0, 1.0, x >= y)", da: "0.0", db: "0.0" },
};

/** 워크그룹 크기. 원소별과 축약은 1차원이다. */
export const WORKGROUP = 64;

/**
 * `dispatchWorkgroups` 축당 한계. **넘으면 던지지 않고 조용히 안 한다.**
 *
 * conv 벤치에서 589,824 개를 요청했고 WebGPU 는 아무 말 없이 일부만 돌렸다.
 * "TF.js 대비 144%" 로 보였고 값 6개 중 5개가 틀렸다 — 값을 같이 안 봤으면
 * 그대로 믿었을 수치다. 그래서 격자 계산이 커널 생성기 안에 있다.
 */
export const MAX_DISPATCH = 65535;

/** 1차원 작업을 한계 안의 2차원 격자로 편다. */
export interface Grid {
  /** `dispatchWorkgroups` 에 넣을 것. */
  readonly x: number;
  readonly y: number;
  /** 한 줄에 놓인 **스레드** 수 — 셰이더가 `g.y * GX + g.x` 로 쓴다. */
  readonly threadsX: number;
}

export function grid1d(n: number, workgroup: number = WORKGROUP): Grid {
  const groups = Math.max(1, Math.ceil(n / workgroup));
  const x = Math.min(groups, MAX_DISPATCH);
  const y = Math.ceil(groups / MAX_DISPATCH);
  return { x, y, threadsX: x * workgroup };
}

/** 2차원 격자에서 평평한 번호를 낸다. 한계를 안 넘어도 같은 식을 쓴다 — 경로가 하나여야 한다. */
function flatId(n: number): string {
  const { threadsX } = grid1d(n);
  return `  let gid = g.y * ${threadsX}u + g.x;\n  if (gid >= ${n}u) { return; }`;
}

export type UnaryName = keyof typeof UNARY & string;
export type BinaryName = keyof typeof BINARY & string;

function unarySpec(name: string): UnarySpec {
  const op = UNARY[name];
  if (!op) throw new Error(`모르는 단항 연산: ${name}`);
  return op;
}

/** 원소별 단항 순방향. 원소 수를 상수로 굽는다 — 경계 검사가 접힌다. */
export function unaryForward(name: string, n: number): string {
  const op = unarySpec(name);
  return `${op.prelude ?? ""}
@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read_write> Out: array<f32>;
@compute @workgroup_size(${WORKGROUP})
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
${flatId(n)}
  let x = A[gid];
  Out[gid] = ${op.fwd};
}`;
}

/** 원소별 단항 역방향. 순방향 결과를 받아 다시 안 센다. */
export function unaryBackward(name: string, n: number): string {
  const op = unarySpec(name);
  return `${op.prelude ?? ""}
@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> O: array<f32>;
@group(0) @binding(2) var<storage, read> G: array<f32>;
@group(0) @binding(3) var<storage, read_write> Out: array<f32>;
@compute @workgroup_size(${WORKGROUP})
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
${flatId(n)}
  let x = A[gid];
  let o = O[gid];
  Out[gid] = G[gid] * (${op.bwd});
}`;
}

/**
 * 평평한 출력 번호에서 두 입력의 번호를 낸다.
 *
 * **나눗셈이 하나도 안 남는다** — 축을 뒤에서부터 훑으며 나머지를 빼 나가고, 제수가
 * 전부 리터럴이라 컴파일러가 곱셈·시프트로 접는다. 유니폼으로 두면 이것이 안 접히고,
 * 그 차이가 conv 에서 4배였다.
 *
 * 크기 1 인 축은 스트라이드가 0 이라 같은 값을 계속 읽는다 — **늘려서 복제하지
 * 않는다.** 복제하는 판은 메모리를 쓰고, conv 에서 im2col 이 진 이유가 그것이다.
 */
function indexPair(
  shape: readonly number[],
  strideA: readonly number[],
  strideB: readonly number[],
): string {
  const lines = ["  var rest = gid;", "  var ia: u32 = 0u;", "  var ib: u32 = 0u;"];
  for (let d = shape.length - 1; d >= 0; d--) {
    lines.push(`  { let i = rest % ${shape[d]}u; rest = rest / ${shape[d]}u;`);
    if (strideA[d] !== 0) lines.push(`    ia = ia + i * ${strideA[d]}u;`);
    if (strideB[d] !== 0) lines.push(`    ib = ib + i * ${strideB[d]}u;`);
    lines.push("  }");
  }
  return lines.join("\n");
}

/** 원소별 이항 순방향. 브로드캐스팅을 스트라이드로 처리한다. */
export function binaryForward(
  name: string,
  shape: readonly number[],
  strideA: readonly number[],
  strideB: readonly number[],
): string {
  const op = BINARY[name];
  if (!op) throw new Error(`모르는 이항 연산: ${name}`);
  const n = shape.reduce((a, b) => a * b, 1);
  return `${op.prelude ?? ""}
@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> Out: array<f32>;
@compute @workgroup_size(${WORKGROUP})
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
${flatId(n)}
${indexPair(shape, strideA, strideB)}
  let x = A[ia];
  let y = B[ib];
  Out[gid] = ${op.fwd};
}`;
}

/**
 * 원소별 이항 역방향, 한쪽 몫. **출력 모양으로 낸다.**
 *
 * 브로드캐스팅이 있으면 여기서는 늘어난 채로 기여를 내고, 접는 일은
 * `reduceBroadcast` 가 따로 한다. 두 단계로 나눈 이유는 **원자 덧셈을 피하려는
 * 것**이다 — 여기서 입력 자리로 바로 더하면 순서가 매번 달라지고, 부동소수는
 * 순서가 바뀌면 값이 바뀌어 같은 씨앗의 학습이 두 번 다르게 간다.
 *
 * 순방향과 **같은 인덱싱 식**을 쓴다. 경로가 둘이면 한쪽만 고치게 된다.
 */
export function binaryBackward(
  name: string,
  which: "a" | "b",
  shape: readonly number[],
  strideA: readonly number[],
  strideB: readonly number[],
): string {
  const op = BINARY[name];
  if (!op) throw new Error(`모르는 이항 연산: ${name}`);
  const n = shape.reduce((a, b) => a * b, 1);
  return `${op.prelude ?? ""}
@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read> O: array<f32>;
@group(0) @binding(3) var<storage, read> G: array<f32>;
@group(0) @binding(4) var<storage, read_write> Out: array<f32>;
@compute @workgroup_size(${WORKGROUP})
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
${flatId(n)}
${indexPair(shape, strideA, strideB)}
  let x = A[ia];
  let y = B[ib];
  let o = O[gid];
  Out[gid] = G[gid] * (${which === "a" ? op.da : op.db});
}`;
}

/**
 * 브로드캐스팅으로 늘어난 축을 도로 접는다.
 *
 * `a(3,1) + b(3,4)` 에서 `a` 로 가는 기울기는 (3,4) 짜리를 축 1 로 합친 (3,1) 이다.
 * 이것을 안 하면 모양이 안 맞고, 모양만 맞추고 합을 빠뜨리면 **값이 그럴듯하게
 * 틀린다** — 골든을 통과하고 학습만 안 되는 종류다.
 *
 * 스레드 하나가 결과 한 칸을 맡아 자기 몫을 **정해진 순서로** 훑는다. 원자 덧셈을
 * 안 쓰므로 두 번 돌리면 같은 값이 나온다.
 *
 * @param full 기울기의 모양
 * @param small 접어서 만들 모양. 랭크가 같고 각 축은 1 이거나 `full` 과 같다.
 */
export function reduceBroadcast(
  full: readonly number[],
  small: readonly number[],
): string {
  if (full.length !== small.length) {
    throw new Error(`랭크가 다르다: ${full.length} vs ${small.length}`);
  }
  const rank = full.length;
  const fullStride: number[] = new Array<number>(rank).fill(1);
  for (let d = rank - 2; d >= 0; d--) {
    fullStride[d] = (fullStride[d + 1] ?? 1) * (full[d + 1] ?? 1);
  }

  const n = small.reduce((a, b) => a * b, 1);
  const decompose: string[] = [];
  const baseTerms: string[] = [];
  const broadcastAxes: number[] = [];
  for (let d = rank - 1; d >= 0; d--) {
    const sd = small[d] ?? 1;
    decompose.push(`  let i${d} = rest % ${sd}u; rest = rest / ${sd}u;`);
  }
  for (let d = 0; d < rank; d++) {
    const sd = small[d] ?? 1;
    const fd = full[d] ?? 1;
    if (sd === fd && fd !== 1) baseTerms.push(`i${d} * ${fullStride[d]}u`);
    else if (fd !== 1) broadcastAxes.push(d);
    // sd === fd === 1 이면 기여가 없다 — 항을 안 만든다.
  }

  const open: string[] = [];
  const close: string[] = [];
  const offTerms: string[] = ["base"];
  for (const d of broadcastAxes) {
    open.push(`  for (var j${d} = 0u; j${d} < ${full[d]}u; j${d} = j${d} + 1u) {`);
    close.push("  }");
    offTerms.push(`j${d} * ${fullStride[d]}u`);
  }

  return `
@group(0) @binding(0) var<storage, read> G: array<f32>;
@group(0) @binding(1) var<storage, read_write> Out: array<f32>;
@compute @workgroup_size(${WORKGROUP})
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
${flatId(n)}
  var rest = gid;
${decompose.join("\n")}
  let base = ${baseTerms.length > 0 ? baseTerms.join(" + ") : "0u"};
  var acc = 0.0;
${open.join("\n")}
    acc = acc + G[${offTerms.join(" + ")}];
${close.join("\n")}
  Out[gid] = acc;
}`;
}

/**
 * 행렬곱. **누산기 16개를 이름 붙인 스칼라로 펼친다.**
 *
 * `array<f32,16>` 에 담고 `acc[i*4+j]` 로 변수 인덱싱하면 WGSL 이 레지스터에 못 두고
 * 메모리로 떨어뜨린다. 같은 알고리즘·같은 타일 크기로 **182 vs 4,474 GFLOPS** 였다
 * (실측). 읽기는 나쁘지만 24배이고, 이 커널은 TF.js 의 115~217% 다.
 */
export function matmul(M: number, K: number, N: number): string {
  const decl: string[] = [];
  const zero: string[] = [];
  const fma: string[] = [];
  const store: string[] = [];
  for (let i = 0; i < 4; i++) {
    for (let j = 0; j < 4; j++) {
      decl.push(`  var c${i}${j}: f32;`);
      zero.push(`  c${i}${j} = 0.0;`);
      fma.push(`      c${i}${j} = fma(a${i}, b${j}, c${i}${j});`);
      store.push(
        `  { let r = row0 + ${i}u; let c = col0 + ${j}u;` +
          ` if (r < M && c < N) { Out[r * N + c] = c${i}${j}; } }`,
      );
    }
  }
  return `
@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> Out: array<f32>;
const M: u32 = ${M}u; const K: u32 = ${K}u; const N: u32 = ${N}u;
var<workgroup> As: array<f32, 1024>;
var<workgroup> Bs: array<f32, 1024>;
@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
  let tid = lid.y * 16u + lid.x;
  let row0 = wid.y * 64u + lid.y * 4u;
  let col0 = wid.x * 64u + lid.x * 4u;
${decl.join("\n")}
${zero.join("\n")}
  let tiles = (K + 15u) / 16u;
  for (var t = 0u; t < tiles; t = t + 1u) {
    for (var s = 0u; s < 4u; s = s + 1u) {
      let idx = s * 256u + tid;
      let ar = idx / 16u; let ak = idx % 16u;
      let arow = wid.y * 64u + ar; let acol = t * 16u + ak;
      As[idx] = select(0.0, A[arow * K + acol], arow < M && acol < K);
      let bk = idx / 64u; let bc = idx % 64u;
      let brow = t * 16u + bk; let bcol = wid.x * 64u + bc;
      Bs[idx] = select(0.0, B[brow * N + bcol], brow < K && bcol < N);
    }
    workgroupBarrier();
    for (var k = 0u; k < 16u; k = k + 1u) {
      let a0 = As[(lid.y * 4u + 0u) * 16u + k];
      let a1 = As[(lid.y * 4u + 1u) * 16u + k];
      let a2 = As[(lid.y * 4u + 2u) * 16u + k];
      let a3 = As[(lid.y * 4u + 3u) * 16u + k];
      let b0 = Bs[k * 64u + lid.x * 4u + 0u];
      let b1 = Bs[k * 64u + lid.x * 4u + 1u];
      let b2 = Bs[k * 64u + lid.x * 4u + 2u];
      let b3 = Bs[k * 64u + lid.x * 4u + 3u];
${fma.join("\n")}
    }
    workgroupBarrier();
  }
${store.join("\n")}
}`;
}

/**
 * 전체 합. 워크그룹 안에서 트리로 접고 워크그룹당 부분합 하나를 낸다.
 *
 * **원자 연산을 안 쓴다.** 부동소수 덧셈은 순서가 바뀌면 값이 달라지고, 그러면 같은
 * 씨앗으로 두 번 돌린 학습이 갈린다. 부분합이 하나가 될 때까지 다시 부르는 쪽이
 * 느리지만 **결정적**이고, 이 프로젝트는 재현되는 쪽을 고른다.
 */
export function reduceSum(n: number): string {
  const g = grid1d(n);
  return `
@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read_write> Out: array<f32>;
const N: u32 = ${n}u;
var<workgroup> part: array<f32, ${WORKGROUP}>;
@compute @workgroup_size(${WORKGROUP})
fn main(@builtin(global_invocation_id) g: vec3<u32>,
        @builtin(local_invocation_id) l: vec3<u32>,
        @builtin(workgroup_id) w: vec3<u32>) {
  // **여기서는 일찍 안 나간다.** 아래 배리어를 워크그룹 전원이 같이 만나야 하고,
  // 범위 밖 스레드가 return 하면 제어 흐름이 균일하지 않아 결과가 정의되지 않는다.
  let gid = g.y * ${g.threadsX}u + g.x;
  part[l.x] = select(0.0, A[gid], gid < N);
  workgroupBarrier();
  var span = ${WORKGROUP / 2}u;
  loop {
    if (span == 0u) { break; }
    if (l.x < span) { part[l.x] = part[l.x] + part[l.x + span]; }
    workgroupBarrier();
    span = span / 2u;
  }
  if (l.x == 0u) { Out[w.y * ${g.x}u + w.x] = part[0]; }
}`;
}

/** `reduceSum` 한 번이 내는 부분합 개수. 하나가 될 때까지 다시 부른다. */
export function reduceParts(n: number): number {
  return Math.max(1, Math.ceil(n / WORKGROUP));
}

/**
 * 축 하나를 접는다. 축약 축을 가운데 두고 `(바깥, 축약, 안쪽)` 으로 본다.
 *
 * 스레드 하나가 결과 한 칸을 맡아 축약 축을 **정해진 순서로** 훑는다. 원자 연산도
 * 트리도 없다 — 같은 입력이면 같은 값이 나온다.
 *
 * **전체 축약에는 이걸 쓰지 않는다.** `outer = inner = 1` 이면 스레드 하나가 n 번
 * 도는 꼴이라, 큰 텐서에서는 `Device.sumAll` 의 트리가 맞다. 여기 쓰는 것은 축이
 * 실제로 있을 때다.
 */
export type ReduceKind = "sum" | "max" | "min" | "prod";

/**
 * 축약의 시작값과 한 걸음.
 *
 * **최대·최소는 파수꾼 값을 안 쓴다.** 처음에 `-3.4028235e38` 을 넣었더니 WGSL 이
 * "f32 로 표현할 수 없다" 며 거부했고(JS 가 찍은 십진수가 f32 최대값보다 위로
 * 반올림된다), 비트캐스트로 -inf 를 만들었더니 그것도 거부했다 — WGSL 은 상수식에
 * 무한대를 못 담는다. 둘 다 예외가 아니라 결과 0 으로만 보였다.
 *
 * 첫 원소에서 시작해 나머지를 훑는 쪽이 그 문제가 아예 없고, 답도 더 정확하다.
 * 축약 길이는 항상 1 이상이라 첫 원소는 늘 있다.
 */
const REDUCE_INIT: Readonly<Record<ReduceKind, string>> = {
  sum: "0.0",
  prod: "1.0",
  max: "A[base]",
  min: "A[base]",
};

/** 시작값이 첫 원소면 그 자리를 두 번 세지 않는다. */
const REDUCE_FROM: Readonly<Record<ReduceKind, number>> = {
  sum: 0, prod: 0, max: 1, min: 1,
};

const REDUCE_STEP: Readonly<Record<ReduceKind, string>> = {
  sum: "acc = acc + v;",
  prod: "acc = acc * v;",
  max: "acc = max(acc, v);",
  min: "acc = min(acc, v);",
};

export function reduceDim(
  kind: ReduceKind,
  outer: number,
  red: number,
  inner: number,
): string {
  const n = outer * inner;
  return `
@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read_write> Out: array<f32>;
@compute @workgroup_size(${WORKGROUP})
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
${flatId(n)}
  let o = gid / ${inner}u;
  let i = gid % ${inner}u;
  let base = o * ${red * inner}u + i;
  var acc = ${REDUCE_INIT[kind]};
  for (var r = ${REDUCE_FROM[kind]}u; r < ${red}u; r = r + 1u) {
    let v = A[base + r * ${inner}u];
    ${REDUCE_STEP[kind]}
  }
  Out[gid] = acc;
}`;
}

/** `sum(dim)` 의 역방향 — 접힌 축으로 도로 편다. */
export function expandDim(outer: number, red: number, inner: number): string {
  const n = outer * red * inner;
  return `
@group(0) @binding(0) var<storage, read> G: array<f32>;
@group(0) @binding(1) var<storage, read_write> Out: array<f32>;
@compute @workgroup_size(${WORKGROUP})
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
${flatId(n)}
  let o = gid / ${red * inner}u;
  let i = gid % ${inner}u;
  Out[gid] = G[o * ${inner}u + i];
}`;
}

/**
 * `amax`/`amin` 의 역방향. **동점이면 고르게 나눈다.**
 *
 * torch 의 실측이 그렇다 — `[1,3,3,2]` 의 `amax` 기울기는 `[0,.5,.5,0]` 이다.
 * 하나만 골라 주면 값 검사는 통과하고 학습만 미묘하게 갈린다. 그래서 골든의 입력에
 * 동점이 일부러 들어 있고, 여기서 그 규칙을 지킨다.
 *
 * 자기 축을 한 번 더 훑어 동점 수를 센다. 축약 길이만큼의 추가 비용이고, 그 값을
 * 순방향에서 들고 오면 버퍼가 하나 더 필요하다 — 지금 크기에서는 다시 세는 쪽이 싸다.
 */
export function extremeBackward(
  outer: number,
  red: number,
  inner: number,
): string {
  const n = outer * red * inner;
  return `
@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> O: array<f32>;
@group(0) @binding(2) var<storage, read> G: array<f32>;
@group(0) @binding(3) var<storage, read_write> Out: array<f32>;
@compute @workgroup_size(${WORKGROUP})
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
${flatId(n)}
  let o = gid / ${red * inner}u;
  let i = gid % ${inner}u;
  let base = o * ${red * inner}u + i;
  let m = O[o * ${inner}u + i];
  var ties = 0.0;
  for (var r = 0u; r < ${red}u; r = r + 1u) {
    if (A[base + r * ${inner}u] == m) { ties = ties + 1.0; }
  }
  Out[gid] = select(0.0, G[o * ${inner}u + i] / ties, A[gid] == m);
}`;
}

/**
 * 출력 축 하나가 입력의 어디를 보는가.
 *
 * `expand`·`repeat`·`swapaxes`·`select`·`diagonal`·`rot90`·`unfold`·`flip`·`split` 이
 * 전부 이 세 규칙의 조합이다. 연산마다 커널을 쓰면 열 몇 개가 되고, 그중 하나만
 * 고치는 날이 온다.
 */
export interface AxisRule {
  /** 출력 축의 크기. */
  readonly size: number;
  /** 입력에서의 걸음. **0 이면 늘린 축이다** — 복제하지 않고 같은 값을 다시 읽는다. */
  readonly stride: number;
  /** `lin` 은 그대로, `mod` 는 되돌아가고(repeat), `rev` 는 거꾸로(flip). */
  readonly kind: "lin" | "mod" | "rev";
  /** `mod`·`rev` 의 주기. 보통 입력 축의 크기다. */
  readonly wrap: number;
  /** `mod` 에 더해지는 자리이동. `roll` 이 쓴다 — `repeat` 은 0 이다. */
  readonly bias?: number;
}

function ruleCoord(r: AxisRule, c: string): string {
  if (r.kind === "mod") {
    const bias = r.bias ?? 0;
    return bias === 0 ? `(${c} % ${r.wrap}u)` : `((${c} + ${bias}u) % ${r.wrap}u)`;
  }
  if (r.kind === "rev") return `(${r.wrap - 1}u - ${c})`;
  return c;
}

/** 출력 번호에서 입력 번호를 내는 WGSL. 제수가 전부 리터럴이라 나눗셈이 안 남는다. */
function sourceIndex(
  rules: readonly AxisRule[],
  offset: number,
  from: string,
  out: string,
): string {
  const lines = [`  var rest_${out} = ${from};`, `  var ${out} = ${offset}u;`];
  for (let d = rules.length - 1; d >= 0; d--) {
    const r = rules[d];
    if (!r) continue;
    lines.push(`  { let c = rest_${out} % ${r.size}u; rest_${out} = rest_${out} / ${r.size}u;`);
    if (r.stride !== 0) {
      lines.push(`    ${out} = ${out} + ${ruleCoord(r, "c")} * ${r.stride}u;`);
    }
    lines.push("  }");
  }
  return lines.join("\n");
}

function ruleCount(rules: readonly AxisRule[]): number {
  return rules.reduce((a, r) => a * r.size, 1);
}

/**
 * 규칙 묶음의 서명. **파이프라인 캐시의 열쇠다.**
 *
 * 규칙의 어느 한 자리라도 빠뜨리면 다른 연산이 같은 셰이더를 물려받는다 — 모양을
 * 굽는 설계라 그것은 조용히 틀린 답이 된다. 그래서 이 함수가 규칙 옆에 있다.
 */
export function ruleKey(rules: readonly AxisRule[], offset: number): string {
  const parts = rules.map(
    (r) => `${r.kind}:${r.size}:${r.stride}:${r.wrap}:${r.bias ?? 0}`,
  );
  return `${parts.join(",")}|${offset}`;
}

export function gather(rules: readonly AxisRule[], offset: number): string {
  const n = ruleCount(rules);
  return `
@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read_write> Out: array<f32>;
@compute @workgroup_size(${WORKGROUP})
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
${flatId(n)}
${sourceIndex(rules, offset, "gid", "src")}
  Out[gid] = A[src];
}`;
}

/**
 * gather 의 역방향 — 흩어진 것을 도로 모은다.
 *
 * **입력 자리마다 출력 전체를 훑는다.** 원자 덧셈으로 흩뿌리면 순서가 매번 달라지고,
 * 부동소수는 순서가 바뀌면 값이 바뀌어 같은 씨앗의 학습이 두 번 다르게 간다.
 * 여기서는 출력 번호를 오름차순으로 도니 몇 번을 돌려도 같은 값이다.
 *
 * 대신 비용이 **입력 크기 × 출력 크기**다. 지금 쓰는 자리(모양 연산의 역방향, 원소
 * 수십 개)에서는 문제가 없지만, 학습 루프 안쪽에 이 커널이 들어가면 안 된다.
 * 그때는 연산별로 접는 법이 따로 있다(`expand` 는 축약, `flip` 은 다시 뒤집기).
 *
 * 겹치는 자리를 제대로 더하는 것이 요점이다 — 길이 5 를 `unfold(3, 1)` 로 펴면
 * 기울기가 `[1,2,3,2,1]` 이 된다. 겹친 만큼 쌓이는 것이고, 안 더하면 전부 1 이 된다.
 */
export function gatherBackward(
  rules: readonly AxisRule[],
  offset: number,
  inSize: number,
): string {
  const outN = ruleCount(rules);
  return `
@group(0) @binding(0) var<storage, read> G: array<f32>;
@group(0) @binding(1) var<storage, read_write> Out: array<f32>;
@compute @workgroup_size(${WORKGROUP})
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
${flatId(inSize)}
  var acc = 0.0;
  for (var t = 0u; t < ${outN}u; t = t + 1u) {
${sourceIndex(rules, offset, "t", "src")}
    if (src == gid) { acc = acc + G[t]; }
  }
  Out[gid] = acc;
}`;
}

/**
 * `diagflat` — 벡터를 대각선에 놓고 나머지는 0.
 *
 * 이것만 gather 로 안 된다. 출력의 대부분이 입력의 **어느 자리도 안 보기** 때문이다.
 */
export function diagflat(n: number): string {
  const total = n * n;
  return `
@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read_write> Out: array<f32>;
@compute @workgroup_size(${WORKGROUP})
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
${flatId(total)}
  let i = gid / ${n}u;
  let j = gid % ${n}u;
  Out[gid] = select(0.0, A[i], i == j);
}`;
}

/** `diagflat` 의 역방향 — 대각선만 거둔다. */
export function diagflatBackward(n: number): string {
  return `
@group(0) @binding(0) var<storage, read> G: array<f32>;
@group(0) @binding(1) var<storage, read_write> Out: array<f32>;
@compute @workgroup_size(${WORKGROUP})
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
${flatId(n)}
  Out[gid] = G[gid * ${n + 1}u];
}`;
}

/**
 * `where(조건, x, y)` — 조건 자리마다 둘 중 하나.
 *
 * 기울기는 **고른 쪽으로만** 간다. 안 고른 쪽으로 0 을 보내는 것과 같은 값이지만,
 * 안 고른 쪽이 NaN 이면 다르다 — `0 * NaN` 은 NaN 이다. 그래서 곱하지 않고 고른다.
 */
export function whereKernel(n: number): string {
  return `
@group(0) @binding(0) var<storage, read> C: array<f32>;
@group(0) @binding(1) var<storage, read> A: array<f32>;
@group(0) @binding(2) var<storage, read> B: array<f32>;
@group(0) @binding(3) var<storage, read_write> Out: array<f32>;
@compute @workgroup_size(${WORKGROUP})
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
${flatId(n)}
  Out[gid] = select(B[gid], A[gid], C[gid] != 0.0);
}`;
}

/** `where` 의 역방향, 한쪽 몫. 고른 자리에만 기울기를 놓는다. */
export function whereBackward(n: number, take: "a" | "b"): string {
  const test = take === "a" ? "C[gid] != 0.0" : "C[gid] == 0.0";
  return `
@group(0) @binding(0) var<storage, read> C: array<f32>;
@group(0) @binding(1) var<storage, read> G: array<f32>;
@group(0) @binding(2) var<storage, read_write> Out: array<f32>;
@compute @workgroup_size(${WORKGROUP})
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
${flatId(n)}
  Out[gid] = select(0.0, G[gid], ${test});
}`;
}

/** 아래·위 삼각만 남기고 0. `diagonal` 이 아니라 **면**을 남긴다. */
export function triangle(rows: number, cols: number, lower: boolean, diagonal: number): string {
  const n = rows * cols;
  // tril 은 j - i <= diagonal, triu 는 j - i >= diagonal 을 남긴다. 정수 뺄셈이
  // 음수가 될 수 있어 i32 로 본다 — u32 로 두면 아래쪽 절반이 거대한 수가 된다.
  const test = lower
    ? `(i32(j) - i32(i)) <= ${diagonal}`
    : `(i32(j) - i32(i)) >= ${diagonal}`;
  return `
@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read_write> Out: array<f32>;
@compute @workgroup_size(${WORKGROUP})
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
${flatId(n)}
  let i = gid / ${cols}u;
  let j = gid % ${cols}u;
  Out[gid] = select(0.0, A[gid], ${test});
}`;
}

/**
 * 누적 합·곱. 스레드 하나가 자기 앞자리를 전부 훑는다.
 *
 * 병렬 스캔(Hillis-Steele 등)이 있지만 안 쓴다. 여기 필요한 길이가 짧고, 병렬 스캔은
 * **더하는 순서가 바뀌어** 같은 입력에서 다른 값이 나올 수 있다. 재현이 먼저다.
 */
export function cumulative(
  kind: "sum" | "prod",
  outer: number,
  len: number,
  inner: number,
): string {
  const n = outer * len * inner;
  const init = kind === "sum" ? "0.0" : "1.0";
  const step = kind === "sum" ? "acc = acc + v;" : "acc = acc * v;";
  return `
@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read_write> Out: array<f32>;
@compute @workgroup_size(${WORKGROUP})
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
${flatId(n)}
  let o = gid / ${len * inner}u;
  let rest = gid % ${len * inner}u;
  let k = rest / ${inner}u;
  let i = rest % ${inner}u;
  let base = o * ${len * inner}u + i;
  var acc = ${init};
  for (var t = 0u; t <= k; t = t + 1u) {
    let v = A[base + t * ${inner}u];
    ${step}
  }
  Out[gid] = acc;
}`;
}

/** `cumsum` 의 역방향 — 뒤에서부터 누적한다. 앞자리는 뒤 전부에 기여했다. */
export function cumsumBackward(outer: number, len: number, inner: number): string {
  const n = outer * len * inner;
  return `
@group(0) @binding(0) var<storage, read> G: array<f32>;
@group(0) @binding(1) var<storage, read_write> Out: array<f32>;
@compute @workgroup_size(${WORKGROUP})
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
${flatId(n)}
  let o = gid / ${len * inner}u;
  let rest = gid % ${len * inner}u;
  let k = rest / ${inner}u;
  let i = rest % ${inner}u;
  let base = o * ${len * inner}u + i;
  var acc = 0.0;
  for (var t = k; t < ${len}u; t = t + 1u) {
    acc = acc + G[base + t * ${inner}u];
  }
  Out[gid] = acc;
}`;
}

/**
 * `gather(dim, index)` — 축 하나를 색인 텐서가 가리키는 대로 고른다.
 *
 * 색인이 float32 에 담겨 온다. dtype 이 하나뿐이라 그런데, 정수 값이 float32 에
 * 정확히 담기는 범위(2²⁴)를 넘으면 조용히 틀린 자리를 읽는다 — 지금 쓰는 크기에서는
 * 한참 아래다.
 */
export function gatherIndex(
  outer: number,
  axis: number,
  inner: number,
  outAxis: number,
): string {
  const n = outer * outAxis * inner;
  return `
@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> I: array<f32>;
@group(0) @binding(2) var<storage, read_write> Out: array<f32>;
@compute @workgroup_size(${WORKGROUP})
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
${flatId(n)}
  let o = gid / ${outAxis * inner}u;
  let rest = gid % ${outAxis * inner}u;
  let i = rest % ${inner}u;
  let want = u32(I[gid]);
  Out[gid] = A[o * ${axis * inner}u + want * ${inner}u + i];
}`;
}

/** 값 하나로 채운다. 기울기 씨앗(`backward()` 의 1.0)과 `zeros` 가 쓴다. */
export function fill(n: number, value: number): string {
  return `
@group(0) @binding(0) var<storage, read_write> Out: array<f32>;
@compute @workgroup_size(${WORKGROUP})
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
${flatId(n)}
  Out[gid] = ${Number.isInteger(value) ? value.toFixed(1) : String(value)};
}`;
}
