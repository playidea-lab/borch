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
 * Abramowitz & Stegun 7.1.26 이고 코어(`borch/_ops.py`)와 **같은 계수**다.
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
 * 감마 계열. **코어(numpy)와 같은 식을 적는다** — 두 벌을 다르게 적으면 어느 쪽이
 * 맞는지를 골든이 못 가른다.
 *
 * `lgamma` 는 란초시(g=7, n=9), `digamma`·`trigamma` 는 되풀이 식으로 6 이상까지
 * 밀고 스털링 점근 전개를 쓴다. 작은 x 에서 점근식이 안 맞기 때문이고, 미는 자리를
 * 빼먹으면 0 근처에서만 조용히 틀린다.
 */
const GAMMA_PRELUDE = `
const LANCZOS = array<f32, 9>(
  0.9999999999998099, 676.5203681218851, -1259.1392167224028,
  771.3234287776531, -176.6150291621406, 12.507343278686905,
  -0.13857109526572012, 9.984369578019572e-6, 1.5056327351493116e-7);
fn lgamma_(x: f32) -> f32 {
  // 반사 공식으로 음수 쪽을 접는다: Γ(x)Γ(1−x) = π/sin(πx).
  let neg = x < 0.5;
  let z = select(x, 1.0 - x, neg) - 1.0;
  var acc = LANCZOS[0];
  for (var i = 1; i < 9; i = i + 1) { acc = acc + LANCZOS[i] / (z + f32(i)); }
  let t = z + 7.5;
  let out = 0.9189385332046727 + (z + 0.5) * log(t) - t + log(abs(acc));
  let flipped = log(3.141592653589793 / abs(sin(3.141592653589793 * x))) - out;
  return select(out, flipped, neg);
}
fn digamma_(x0: f32) -> f32 {
  var x = x0;
  var out = 0.0;
  // 6 이상으로 민다 — 점근식이 그 아래에서 안 맞는다.
  for (var i = 0; i < 8; i = i + 1) {
    if (x >= 6.0) { break; }
    out = out - 1.0 / x;
    x = x + 1.0;
  }
  let inv = 1.0 / x;
  let inv2 = inv * inv;
  return out + log(x) - 0.5 * inv
    - inv2 * (0.08333333333333333 - inv2 * (0.008333333333333333 - inv2 * 0.003968253968253968));
}
fn trigamma_(x0: f32) -> f32 {
  var x = x0;
  var out = 0.0;
  for (var i = 0; i < 8; i = i + 1) {
    if (x >= 6.0) { break; }
    out = out + 1.0 / (x * x);
    x = x + 1.0;
  }
  let inv = 1.0 / x;
  let inv2 = inv * inv;
  return out + inv * (1.0 + 0.5 * inv
    + inv2 * (0.16666666666666666 - inv2 * (0.03333333333333333 - inv2 * 0.023809523809523808)));
}`;

/**
 * `erf` 의 역함수. 구간을 둘로 가른다 — 가운데와 꼬리는 수렴이 달라서 한 식으로
 * 못 덮고, 덮으려 들면 한쪽이 허용 오차를 넘는다. 마지막에 뉴턴 한 번으로 조인다.
 */
const ERFINV_PRELUDE = `
fn erfinv_(x: f32) -> f32 {
  let z = x * x;
  let mid = x * (((-0.140543331 * z + 0.914624893) * z - 1.645349621) * z + 0.886226899)
    / ((((0.012229801 * z - 0.329097515) * z + 1.442710462) * z - 2.118377725) * z + 1.0);
  let safe = clamp(abs(x), 0.0, 0.999999);
  let w = sqrt(-log((1.0 - safe) / 2.0));
  let tail = sign(x) * (((1.641345311 * w + 3.429567803) * w - 1.624906493) * w - 1.970840454)
    / ((1.6370678 * w + 3.5438892) * w + 1.0);
  var out = select(tail, mid, abs(x) <= 0.7);
  // 근사식만으로는 허용 오차 언저리다. 뉴턴 한 번이 그것을 넘긴다.
  let err = erf_(out) - x;
  out = out - err / (1.1283791670955126 * exp(-out * out));
  return out;
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
  // **0 에서 안 흘린다.** `step(0.0, x)` 는 `x >= 0` 이라 정확히 0 인 자리에서 1 을
  // 주는데 torch 는 0 을 준다. 골든의 relu 케이스는 입력에 0 이 없어 이것을 못 봤고,
  // ResNet 을 진짜 torch 와 맞춰보다 드러났다.
  relu: { fwd: "max(x, 0.0)", bwd: "select(0.0, 1.0, x > 0.0)" },
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
  logical_not: { fwd: "select(0.0, 1.0, x == 0.0)", bwd: "0.0" },
  /**
   * torch 의 기본 `gelu` — 근사형이 아니라 **정확형**이다.
   *
   * `0.5·x·(1 + erf(x/√2))`. 왼쪽 꼬리에서 `1` 과 `erf` 가 상쇄되는데, f32 에서
   * `erf(-8)` 은 정확히 `-1` 이라 결과가 0 이 된다. torch 는 -4.9e-15 를 주므로
   * 차이가 5e-15 이고, 이 프로젝트의 허용 오차(1e-4) 한참 아래다. 더 정확히 하려면
   * 코어처럼 erfc 로 유도해야 하는데 지금 그럴 이유가 없다.
   */
  gelu: {
    fwd: "0.5 * x * (1.0 + erf_(x * 0.7071067811865476))",
    bwd:
      "0.5 * (1.0 + erf_(x * 0.7071067811865476)) " +
      "+ x * 0.3989422804014327 * exp(-0.5 * x * x)",
    prelude: ERF_PRELUDE,
  },
  silu: {
    fwd: "x / (1.0 + exp(-x))",
    // s 를 두 번 쓰므로 보조 함수로 뺀다 — 식 안에 두면 exp 를 네 번 부른다.
    bwd: "silu_grad(x)",
    prelude: `
fn silu_grad(x: f32) -> f32 {
  let s = 1.0 / (1.0 + exp(-x));
  return s * (1.0 + x * (1.0 - s));
}`,
  },
  elu: {
    fwd: "select(exp(x) - 1.0, x, x > 0.0)",
    bwd: "select(o + 1.0, 1.0, x > 0.0)",
  },
  // ── 인자 없는 활성함수. 인자를 받는 것들은 `tensor.ts` 가 상수를 구워 만든다. ──
  //
  // **꺾이는 점에서 어느 쪽인지가 전부다.** 식은 문서에 있지만 `x == ±3`·`x == 6`
  // 같은 정확한 경계에서 torch 가 무엇을 주는지는 재봐야 알고, 난수 입력은 그 점을
  // 절대 안 준다. 골든이 그 점들을 손으로 들고 있다.
  hardsigmoid: {
    fwd: "clamp(x / 6.0 + 0.5, 0.0, 1.0)",
    bwd: "select(0.0, 0.16666666666666666, x > -3.0 && x < 3.0)",
  },
  hardswish: {
    fwd: "select(select(x * (x + 3.0) / 6.0, x, x >= 3.0), 0.0, x <= -3.0)",
    bwd: "select(select((2.0 * x + 3.0) / 6.0, 1.0, x >= 3.0), 0.0, x <= -3.0)",
  },
  // log σ(x). **곧장 계산하면 큰 음수에서 log(0) 이 된다** — 안정형으로 쓴다.
  logsigmoid: {
    fwd: "-log(1.0 + exp(-abs(x))) + min(x, 0.0)",
    bwd: "1.0 / (1.0 + exp(x))",
  },
  mish: {
    fwd: "x * tanh(log(1.0 + exp(-abs(x))) + max(x, 0.0))",
    bwd: "mish_grad(x)",
    prelude: `
fn mish_grad(x: f32) -> f32 {
  let sp = log(1.0 + exp(-abs(x))) + max(x, 0.0);
  let th = tanh(sp);
  let s = 1.0 / (1.0 + exp(-x));
  return th + x * (1.0 - th * th) * s;
}`,
  },
  // **양쪽 경계에서 기울기가 0 이다.** `clamp` 를 그냥 미분하면 그 자리를 놓친다.
  relu6: {
    fwd: "clamp(x, 0.0, 6.0)",
    bwd: "select(0.0, 1.0, x > 0.0 && x < 6.0)",
  },
  selu: {
    fwd: "1.0507009873554805 * select(1.6732632423543772 * (exp(x) - 1.0), x, x > 0.0)",
    bwd:
      "1.0507009873554805 * select(1.6732632423543772 * exp(x), 1.0, x > 0.0)",
  },
  softsign: {
    fwd: "x / (1.0 + abs(x))",
    bwd: "softsign_grad(x)",
    prelude: `
fn softsign_grad(x: f32) -> f32 {
  let d = 1.0 + abs(x);
  return 1.0 / (d * d);
}`,
  },
  tanhshrink: {
    fwd: "x - tanh(x)",
    bwd: "tanh(x) * tanh(x)",
  },
  // ── 급수로 세는 것들. **닫힌 식이 없다.** ────────────────────────────────
  //
  // 계수는 잘 알려진 표를 그대로 쓴다 — 자릿수를 줄이면 그만큼 답이 틀린다.
  // 코어(numpy)와 **같은 식**을 적는다. 두 벌을 다르게 적으면 어느 쪽이 맞는지를
  // 골든이 못 가른다.
  lgamma: {
    fwd: "lgamma_(x)",
    bwd: "digamma_(x)",
    prelude: `${GAMMA_PRELUDE}`,
  },
  digamma: {
    fwd: "digamma_(x)",
    bwd: "trigamma_(x)",
    prelude: `${GAMMA_PRELUDE}`,
  },
  erfinv: {
    fwd: "erfinv_(x)",
    // d/dx erfinv(x) = √π/2 · exp(erfinv(x)²)
    bwd: "0.8862269254527580 * exp(o * o)",
    prelude: `${ERF_PRELUDE}${ERFINV_PRELUDE}`,
  },
};

export const BINARY: Readonly<Record<string, BinarySpec>> = {
  add: { fwd: "x + y", da: "1.0", db: "1.0" },
  sub: { fwd: "x - y", da: "1.0", db: "-1.0" },
  mul: { fwd: "x * y", da: "y", db: "x" },
  div: { fwd: "x / y", da: "1.0 / y", db: "-x / (y * y)" },
  // **밑이 음수면 답이 없다.** WGSL 의 pow 는 `exp2(y·log2(x))` 이고 log2 가 음수에서
  // 정의되지 않는다 — 실제로는 `|x|` 를 쓴 것 같은 값이 나와서, 짝수 지수의 순방향은
  // 우연히 맞고 역방향의 부호만 뒤집힌다. 정수 지수는 `Tensor.powScalar` 가 곱셈으로
  // 돌아가므로 이 자리를 안 지난다.
  pow: { fwd: "pow(x, y)", da: "y * pow(x, y - 1.0)", db: "o * log(x)" },
  // **동점이면 반씩 나눈다.** torch 가 그렇다 — `maximum(2, 2)` 의 기울기는 양쪽 다
  // 0.5 이지 1 이 아니다. `step(y, x)` 는 `x >= y` 라 동점에서 양쪽에 1 을 줬고,
  // 그러면 합이 torch 의 두 배가 된다. 순방향은 어느 쪽이든 똑같이 맞으므로 값 대조로는
  // 안 잡히고, `edge::grad::maximum(동점)` 이 이것을 묻는다.
  //
  // `clamp` 와 `leakyRelu` 는 여기 얹혀 있었는데 **torch 에서 그 둘은 나누지 않는다** —
  // 경계에서 기울기를 온전히 흘린다. 그래서 각자 커널을 갖게 했다(`clampScalar`·`leakyRelu`).
  maximum: {
    fwd: "max(x, y)",
    da: "select(select(0.0, 1.0, x > y), 0.5, x == y)",
    db: "select(select(0.0, 1.0, y > x), 0.5, x == y)",
  },
  minimum: {
    fwd: "min(x, y)",
    da: "select(select(0.0, 1.0, x < y), 0.5, x == y)",
    db: "select(select(0.0, 1.0, y < x), 0.5, x == y)",
  },
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
  logical_and: {
    fwd: "select(0.0, 1.0, x != 0.0 && y != 0.0)", da: "0.0", db: "0.0",
  },
  logical_or: {
    fwd: "select(0.0, 1.0, x != 0.0 || y != 0.0)", da: "0.0", db: "0.0",
  },
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

/**
 * 상수를 품은 단항 연산.
 *
 * `clamp(-1, 1)` 이나 `leakyRelu(0.1)` 처럼 인자가 식에 섞이는 것들이다. 인자를
 * 유니폼으로 넣으면 셰이더가 하나로 끝나지만 이 파일이 그 반대를 고르는 이유가 여기도
 * 그대로 적용된다 — 상수로 구우면 접힌다. 이름에 그 상수가 들어가므로 파이프라인
 * 캐시가 알아서 갈라지고, 같은 인자로 두 번 부르면 같은 셰이더를 쓴다.
 */
const DERIVED: Record<string, UnarySpec> = {};

/**
 * WGSL 의 f32 리터럴. 정수처럼 보이는 값도 소수점을 달아야 형이 안 갈린다.
 *
 * **무한대와 NaN 은 셰이더에 구울 수가 없다.** WGSL 은 컴파일 시점에 계산되는 값이
 * inf 나 NaN 이 되는 것을 금지한다 — 리터럴도, `bitcast<f32>(0x7f800000u)` 로 에둘러도
 * 똑같이 거절당한다(둘 다 실측). 그전에는 `String(Infinity)` 가 `Infinity` 라는
 * **글자**를 셰이더에 심어서 `unresolved value 'Infinity'` 로 멈췄다 — 값 하나
 * 채우려다 파이프라인 전체가 죽었고, 그 자리가 `Tensor.full(shape, Infinity)` 였다.
 *
 * 그쪽은 CPU 에서 채워 올리는 길로 갔다. 여기서는 **시끄럽게 거절한다** — 못 굽는
 * 것을 조용히 근사하면 셰이더는 도는데 답이 다른 상태가 된다.
 */
export function f32lit(v: number): string {
  if (!Number.isFinite(v)) {
    throw new Error(
      `WGSL 은 상수 ${v} 를 f32 로 못 적는다 — 무한대·NaN 은 컴파일 시점에 금지된다.\n` +
      "  CPU 에서 채워 올려라(`Tensor.full` 이 그 길로 간다).",
    );
  }
  return Number.isInteger(v) ? `${v}.0` : String(v);
}

/** 상수를 구운 단항 연산을 등록하고 그 이름을 준다. 이미 있으면 다시 안 만든다. */
export function unaryWith(key: string, make: () => UnarySpec): string {
  DERIVED[key] ??= make();
  return key;
}

/** 이 이름으로 단항 커널을 만들 수 있는가. 표에 있는 것과 구운 것을 함께 본다. */
export function hasUnary(name: string): boolean {
  return Boolean(UNARY[name] ?? DERIVED[name]);
}

function unarySpec(name: string): UnarySpec {
  const op = UNARY[name] ?? DERIVED[name];
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

/**
 * 축약하되 **값이 아니라 자리**를 낸다.
 *
 * 동점이면 **먼저 나온 자리**를 준다 — 뒤엣것으로 밀리지 않게 부등호를 엄격하게 쓴다.
 * torch 도 그렇게 답한다.
 */
export function argReduce(
  kind: "max" | "min",
  outer: number,
  red: number,
  inner: number,
): string {
  const n = outer * inner;
  const better = kind === "max" ? "v > best" : "v < best";
  return `
@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read_write> Out: array<f32>;
@compute @workgroup_size(${WORKGROUP})
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
${flatId(n)}
  let o = gid / ${inner}u;
  let i = gid % ${inner}u;
  let base = o * ${red * inner}u + i;
  var best = A[base];
  var at = 0u;
  for (var r = 1u; r < ${red}u; r = r + 1u) {
    let v = A[base + r * ${inner}u];
    if (${better}) { best = v; at = r; }
  }
  Out[gid] = f32(at);
}`;
}

/**
 * 축 하나의 앞뒤에 상수를 덧댄다.
 *
 * 덧댄 자리는 입력의 **어느 자리도 안 보므로** gather 로 안 된다. 여러 축을 채우려면
 * 이 커널을 축마다 부른다 — 한 번에 하는 커널을 따로 두면 두 벌을 고쳐야 한다.
 */
export function padAxis(
  outer: number,
  before: number,
  size: number,
  after: number,
  inner: number,
  value: number,
): string {
  const outSize = before + size + after;
  const n = outer * outSize * inner;
  const literal = f32lit(value);
  return `
@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read_write> Out: array<f32>;
@compute @workgroup_size(${WORKGROUP})
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
${flatId(n)}
  let o = gid / ${outSize * inner}u;
  let rest = gid % ${outSize * inner}u;
  let c = rest / ${inner}u;
  let i = rest % ${inner}u;
  if (c < ${before}u || c >= ${before + size}u) {
    Out[gid] = ${literal};
    return;
  }
  Out[gid] = A[o * ${size * inner}u + (c - ${before}u) * ${inner}u + i];
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
  /**
   * `lin` 은 그대로, `mod` 는 되돌아가고(repeat), `rev` 는 거꾸로(flip),
   * `div` 는 같은 자리에 머무른다(repeat_interleave).
   */
  readonly kind: "lin" | "mod" | "rev" | "div";
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
  // `wrap` 이 되풀이 횟수다. `[a,b]` 를 2 번씩이면 `[a,a,b,b]` 가 된다.
  if (r.kind === "div") return `(${c} / ${r.wrap}u)`;
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

/**
 * `prod` 의 역방향.
 *
 * **`out / x` 로 안 쓴다.** 그 식은 x 에 0 이 하나만 있어도 무너진다 — 0 인 자리에서
 * 0/0 이 되고, 나머지 자리에서는 out 이 0 이라 전부 0 이 된다. torch 는 그 축의
 * 다른 값들의 곱을 준다.
 *
 * 자기를 뺀 나머지를 그때그때 곱한다. 축 길이만큼 도는 값이고, 나눗셈이 없으니
 * 0 이 섞여도 맞는다.
 */
export function prodBackward(outer: number, red: number, inner: number): string {
  const n = outer * red * inner;
  return `
@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> G: array<f32>;
@group(0) @binding(2) var<storage, read_write> Out: array<f32>;
@compute @workgroup_size(${WORKGROUP})
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
${flatId(n)}
  let o = gid / ${red * inner}u;
  let rest = gid % ${red * inner}u;
  let r = rest / ${inner}u;
  let i = rest % ${inner}u;
  let base = o * ${red * inner}u + i;
  var others = 1.0;
  for (var t = 0u; t < ${red}u; t = t + 1u) {
    if (t != r) { others = others * A[base + t * ${inner}u]; }
  }
  Out[gid] = G[o * ${inner}u + i] * others;
}`;
}

/**
 * `cumprod` 의 역방향.
 *
 * `out[j]` 는 `A[0..j]` 의 곱이므로, `A[k]` 는 `j >= k` 인 모든 출력에 기여한다.
 * 그 기여가 **자기를 뺀 나머지의 곱**이라 여기도 나눗셈이 없다 — 0 이 섞여도 맞는다.
 *
 * 비용이 축 길이의 세제곱이다. 짧은 축에서만 쓸 것이고, 길어지면 접두·접미 곱을
 * 미리 들어야 한다. 지금 그것을 요구하는 자리가 없어서 안 한다.
 */
export function cumprodBackward(outer: number, len: number, inner: number): string {
  const n = outer * len * inner;
  return `
@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> G: array<f32>;
@group(0) @binding(2) var<storage, read_write> Out: array<f32>;
@compute @workgroup_size(${WORKGROUP})
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
${flatId(n)}
  let o = gid / ${len * inner}u;
  let rest = gid % ${len * inner}u;
  let k = rest / ${inner}u;
  let i = rest % ${inner}u;
  let base = o * ${len * inner}u + i;
  var acc = 0.0;
  for (var j = k; j < ${len}u; j = j + 1u) {
    var others = 1.0;
    for (var t = 0u; t <= j; t = t + 1u) {
      if (t != k) { others = others * A[base + t * ${inner}u]; }
    }
    acc = acc + G[base + j * ${inner}u] * others;
  }
  Out[gid] = acc;
}`;
}

/**
 * `gather(dim, index)` 의 역방향 — 읽어간 자리로 도로 모은다.
 *
 * 같은 자리를 여러 번 읽었으면 그만큼 쌓인다. 입력 자리마다 출력을 오름차순으로
 * 훑으므로 원자 연산이 없고, 두 번 돌리면 같은 값이다. 비용은 입력 × 출력이다.
 */
export function gatherIndexBackward(
  outer: number,
  axis: number,
  inner: number,
  outAxis: number,
): string {
  const inN = outer * axis * inner;
  const outN = outer * outAxis * inner;
  return `
@group(0) @binding(0) var<storage, read> I: array<f32>;
@group(0) @binding(1) var<storage, read> G: array<f32>;
@group(0) @binding(2) var<storage, read_write> Out: array<f32>;
@compute @workgroup_size(${WORKGROUP})
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
${flatId(inN)}
  var acc = 0.0;
  for (var t = 0u; t < ${outN}u; t = t + 1u) {
    let o = t / ${outAxis * inner}u;
    let rest = t % ${outAxis * inner}u;
    let i = rest % ${inner}u;
    let src = o * ${axis * inner}u + u32(I[t]) * ${inner}u + i;
    if (src == gid) { acc = acc + G[t]; }
  }
  Out[gid] = acc;
}`;
}

/** `index_select` — 축 하나를 색인 **벡터**가 고른다. 색인이 자리마다 다르지 않다. */
export function indexSelect(
  outer: number,
  axis: number,
  inner: number,
  count: number,
): string {
  const n = outer * count * inner;
  return `
@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> I: array<f32>;
@group(0) @binding(2) var<storage, read_write> Out: array<f32>;
@compute @workgroup_size(${WORKGROUP})
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
${flatId(n)}
  let o = gid / ${count * inner}u;
  let rest = gid % ${count * inner}u;
  let k = rest / ${inner}u;
  let i = rest % ${inner}u;
  Out[gid] = A[o * ${axis * inner}u + u32(I[k]) * ${inner}u + i];
}`;
}

/** `index_select` 의 역방향. 같은 자리를 여러 번 골랐으면 쌓인다. */
export function indexSelectBackward(
  outer: number,
  axis: number,
  inner: number,
  count: number,
): string {
  const inN = outer * axis * inner;
  return `
@group(0) @binding(0) var<storage, read> I: array<f32>;
@group(0) @binding(1) var<storage, read> G: array<f32>;
@group(0) @binding(2) var<storage, read_write> Out: array<f32>;
@compute @workgroup_size(${WORKGROUP})
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
${flatId(inN)}
  let o = gid / ${axis * inner}u;
  let rest = gid % ${axis * inner}u;
  let r = rest / ${inner}u;
  let i = rest % ${inner}u;
  var acc = 0.0;
  for (var k = 0u; k < ${count}u; k = k + 1u) {
    if (u32(I[k]) == r) { acc = acc + G[o * ${count * inner}u + k * ${inner}u + i]; }
  }
  Out[gid] = acc;
}`;
}

export function convOut(size: number, pad: number, kernel: number, stride: number): number {
  return Math.floor((size + 2 * pad - kernel) / stride) + 1;
}

/**
 * 차원 수에 상관없는 합성곱의 모양.
 *
 * 1·2·3 차원을 **한 커널 생성기로** 덮는다. 공간 축을 배열로 들면 conv1d 는 축이
 * 하나, conv3d 는 셋일 뿐이고 나머지 구조가 같다 — 차원마다 커널을 따로 쓰면
 * 세 벌이 되고, 그중 하나만 고치는 날이 온다. 실제로 자매가 그 상태였다.
 */
export interface ConvNDShape {
  readonly N: number;
  readonly C: number;
  readonly O: number;
  /** 입력의 공간 축들. */
  readonly inDims: readonly number[];
  readonly kernel: readonly number[];
  readonly stride: readonly number[];
  readonly pad: readonly number[];
  readonly outDims: readonly number[];
}

export function convNDKey(s: ConvNDShape): string {
  return [s.N, s.C, s.O, s.inDims, s.kernel, s.stride, s.pad].join("|");
}

/** 뒤에서부터 누적한 곱 — 축 하나를 한 칸 옮길 때 건너뛰는 원소 수다. */
function suffixStrides(dims: readonly number[]): number[] {
  const out: number[] = new Array<number>(dims.length).fill(1);
  for (let d = dims.length - 2; d >= 0; d--) {
    out[d] = (out[d + 1] ?? 1) * (dims[d + 1] ?? 1);
  }
  return out;
}


/**
 * 타일링 합성곱 — **암묵적 GEMM**.
 *
 * `tests/browser/wgsl_conv.js` 가 TF.js 의 72~284% 로 잰 커널을 옮긴 것이다. 옮기기
 * 전에는 아래의 단순 커널이 들어 있었고, ResNet 한 스텝이 자매의 272 배였다.
 *
 * ## 축을 뒤집었다
 *
 * 벤치 판은 결과를 `(N·OH·OW, O)` 로 쓴다. 우리는 NCHW 라 그대로 두면 전치가 한 번
 * 더 필요하다. 대신 GEMM 을 `(O, N·OH·OW)` 로 뒤집으면
 *
 * - 가중치 타일이 `W[f·K + k]` 로 이어진 자리를 읽고,
 * - 결과 타일의 이웃 스레드가 이웃 `ow` 를 써서 NCHW 로 바로 합쳐진다.
 *
 * ## 모양을 굽는 이유가 여기서 제일 크다
 *
 * 타일을 실을 때마다 원소당 나눗셈을 예닐곱 번 다시 한다. 제수가 유니폼이면 컴파일러가
 * 그것을 곱셈·시프트로 못 바꾸고 GPU 에는 정수 나눗셈 하드웨어가 없다 — 벤치에서 그
 * 하나가 43% 와 284% 를 갈랐다.
 */
export function convNDForwardTiled(s: ConvNDShape, hasBias: boolean): string {
  const inStride = suffixStrides(s.inDims);
  const outStride = suffixStrides(s.outDims);
  const kStride = suffixStrides(s.kernel);
  const inSpace = s.inDims.reduce((a, b) => a * b, 1);
  const outSpace = s.outDims.reduce((a, b) => a * b, 1);
  const kSpace = s.kernel.reduce((a, b) => a * b, 1);
  const K = s.C * kSpace;
  const P = s.N * outSpace;
  return tiledGemm({
    M: s.O, N: P, K,
    bindings: `@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> Wt: array<f32>;
${hasBias ? "@group(0) @binding(2) var<storage, read> B: array<f32>;" : ""}
@group(0) @binding(${hasBias ? 3 : 2}) var<storage, read_write> Out: array<f32>;`,
    // 가중치가 (O, K) 로 이어져 있어 행 하나가 통째로 붙어 있다.
    loadA: `          v = Wt[arow * ${K}u + kk];`,
    // im2col 을 **메모리에 안 펴고** 여기서 만든다.
    loadB: `          let ch = kk / ${kSpace}u;
${s.kernel.map((size, d) =>
      `          let kk${d} = (kk / ${kStride[d] ?? 1}u) % ${size}u;`).join("\n")}
          let bn = col / ${outSpace}u;
${s.outDims.map((size, d) =>
      `          let o${d} = (col / ${outStride[d] ?? 1}u) % ${size}u;`).join("\n")}
${s.outDims.map((_, d) =>
      `          let i${d} = i32(o${d} * ${s.stride[d] ?? 1}u + kk${d}) - ${s.pad[d] ?? 0};`)
      .join("\n")}
          if (${s.inDims.map((size, d) => `i${d} >= 0 && i${d} < ${size}`).join(" && ")}) {
            v = X[(bn * ${s.C}u + ch) * ${inSpace}u
              + ${s.inDims.map((_, d) => `u32(i${d}) * ${inStride[d] ?? 1}u`).join(" + ")}];
          }`,
    emit: `  let bn = col / ${outSpace}u;
${s.outDims.map((size, d) =>
      `  let o${d} = (col / ${outStride[d] ?? 1}u) % ${size}u;`).join("\n")}
  Out[(bn * ${s.O}u + f) * ${outSpace}u
    + ${s.outDims.map((_, d) => `o${d} * ${outStride[d] ?? 1}u`).join(" + ")}]
    = v${hasBias ? " + B[f]" : ""};`,
  });
}

/** 타일링 conv 가 쓸 dispatch 격자. 행이 출력 채널, 열이 배치·출력 자리다. */
export function convTiledGrid(s: ConvNDShape): [number, number, number] {
  const P = s.N * s.outDims.reduce((a, b) => a * b, 1);
  return [Math.ceil(P / 64), Math.ceil(s.O / 64), 1];
}

/**
 * 타일링 GEMM 의 뼈대.
 *
 * 순방향·역방향 셋이 **같은 구조에 색인만 다르다.** 뼈대를 세 번 베껴 적으면 그중
 * 하나만 고치는 날이 오고, 그 하나는 기울기 쪽일 것이다 — 값 검사가 못 보는 쪽.
 *
 * @param loadA 행 `arow`, 안쪽 `kk` 에서 왼쪽 타일 원소를 내는 WGSL(식 하나).
 * @param loadB 안쪽 `kk`, 열 `col` 에서 오른쪽 타일 원소를 내는 WGSL 블록. `v` 에 담는다.
 * @param emit 행 `f`, 열 `col`, 값 `v` 를 쓰는 WGSL 블록.
 */
function tiledGemm(opts: {
  readonly M: number;
  readonly N: number;
  readonly K: number;
  readonly bindings: string;
  readonly loadA: string;
  readonly loadB: string;
  readonly emit: string;
  /**
   * 축약을 몇 조각으로 나눌 것인가. 1 이면 안 나눈다.
   *
   * 나누면 `emit` 이 받는 것이 부분합이고, 어느 조각인지는 `part` 로 온다 —
   * 부르는 쪽이 그것을 어디에 쌓을지 정한다.
   */
  readonly splits?: number;
}): string {
  const decl: string[] = [];
  const zero: string[] = [];
  const fma: string[] = [];
  const store: string[] = [];
  for (let i = 0; i < 4; i++) {
    for (let j = 0; j < 4; j++) {
      decl.push(`  var c${i}${j}: f32;`);
      zero.push(`  c${i}${j} = 0.0;`);
      fma.push(`      c${i}${j} = fma(a${i}, b${j}, c${i}${j});`);
      store.push(`  emit(row0 + ${i}u, col0 + ${j}u, c${i}${j}, wid.z);`);
    }
  }
  const splits = opts.splits ?? 1;
  // 조각마다 맡는 타일 수. 마지막 조각이 조금 덜 맡을 수 있으므로 경계를 넘지 않게 센다.
  const allTiles = Math.ceil(opts.K / 16);
  const perSplit = Math.ceil(allTiles / splits);
  return `
${opts.bindings}

var<workgroup> As: array<f32, 1024>;
var<workgroup> Bs: array<f32, 1024>;

fn emit(f: u32, col: u32, v: f32, part: u32) {
  if (f >= ${opts.M}u || col >= ${opts.N}u) { return; }
${opts.emit}
}

@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
  let tid = lid.y * 16u + lid.x;
  let row0 = wid.y * 64u + lid.y * 4u;
  let col0 = wid.x * 64u + lid.x * 4u;
${decl.join("\n")}
${zero.join("\n")}
  let tFrom = wid.z * ${perSplit}u;
  let tTo = min(tFrom + ${perSplit}u, ${allTiles}u);
  for (var t = tFrom; t < tTo; t = t + 1u) {
    for (var sload = 0u; sload < 4u; sload = sload + 1u) {
      let idx = sload * 256u + tid;
      {
        let arow = wid.y * 64u + idx / 16u;
        let kk = t * 16u + idx % 16u;
        var v = 0.0;
        if (arow < ${opts.M}u && kk < ${opts.K}u) {
${opts.loadA}
        }
        As[idx] = v;
      }
      {
        let kk = t * 16u + idx / 64u;
        let col = wid.x * 64u + idx % 64u;
        var v = 0.0;
        if (kk < ${opts.K}u && col < ${opts.N}u) {
${opts.loadB}
        }
        Bs[idx] = v;
      }
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

/** 입력 자리(`col`)와 커널 자리(`kk`)를 좌표로 푸는 WGSL. */
function patchCoords(s: ConvNDShape, indent: string): {
  kParts: string; pParts: string; coords: string; guard: string; offset: string;
} {
  const inStride = suffixStrides(s.inDims);
  const outStride = suffixStrides(s.outDims);
  const kStride = suffixStrides(s.kernel);
  const kSpace = s.kernel.reduce((a, b) => a * b, 1);
  const outSpace = s.outDims.reduce((a, b) => a * b, 1);
  return {
    kParts: [`${indent}let ch = kk / ${kSpace}u;`,
      ...s.kernel.map((size, d) =>
        `${indent}let kk${d} = (kk / ${kStride[d] ?? 1}u) % ${size}u;`)].join("\n"),
    pParts: [`${indent}let bn = col / ${outSpace}u;`,
      ...s.outDims.map((size, d) =>
        `${indent}let o${d} = (col / ${outStride[d] ?? 1}u) % ${size}u;`)].join("\n"),
    coords: s.outDims.map((_, d) =>
      `${indent}let i${d} = i32(o${d} * ${s.stride[d] ?? 1}u + kk${d}) - ${s.pad[d] ?? 0};`)
      .join("\n"),
    guard: s.inDims.map((size, d) => `i${d} >= 0 && i${d} < ${size}`).join(" && "),
    offset: s.inDims.map((_, d) => `u32(i${d}) * ${inStride[d] ?? 1}u`).join(" + "),
  };
}

/**
 * 가중치로 가는 기울기 — 타일링 판.
 *
 * `dW[o, (c,k)] = Σ_p G[p, o] · X_col[p, (c,k)]` 인 GEMM 이다. 행이 출력 채널,
 * 열이 `(입력 채널, 커널 자리)`, 안쪽이 배치·출력 자리다. 결과가 `(O, K)` 로
 * 이어져 나오므로 가중치 모양 그대로다.
 */
export function convNDGradWeightTiled(s: ConvNDShape): string {
  const outSpace = s.outDims.reduce((a, b) => a * b, 1);
  const kSpace = s.kernel.reduce((a, b) => a * b, 1);
  const inSpace = s.inDims.reduce((a, b) => a * b, 1);
  const K = s.N * outSpace;
  const cols = s.C * kSpace;
  const splits = convGradWeightSplit(s);
  const c = patchCoords({ ...s }, "          ");
  return tiledGemm({
    M: s.O, N: cols, K, splits,
    bindings: `@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> G: array<f32>;
@group(0) @binding(2) var<storage, read_write> Out: array<f32>;`,
    // 왼쪽: G 를 (출력채널 × 배치·출력자리) 로 본다.
    loadA: `          let gn = kk / ${outSpace}u;
          let gp = kk % ${outSpace}u;
          v = G[(gn * ${s.O}u + arow) * ${outSpace}u + gp];`,
    // 오른쪽: im2col 을 **메모리에 안 펴고** 여기서 만든다. 열이 (채널, 커널 자리),
    // 안쪽이 (배치, 출력 자리)다 — 순방향과 축만 바뀐 같은 계산이다.
    loadB: `          let ch = col / ${kSpace}u;
${s.kernel.map((size, d) =>
      `          let kk${d} = (col / ${suffixStrides(s.kernel)[d] ?? 1}u) % ${size}u;`).join("\n")}
          let bn = kk / ${outSpace}u;
${s.outDims.map((size, d) =>
      `          let o${d} = (kk / ${suffixStrides(s.outDims)[d] ?? 1}u) % ${size}u;`).join("\n")}
${c.coords}
          if (${c.guard}) {
            v = X[(bn * ${s.C}u + ch) * ${inSpace}u + ${c.offset}];
          }`,
    // 조각을 나눴으면 부분합을 조각별 칸에 쓴다. 안 나눴으면 그 칸이 하나뿐이라
    // 그대로 결과다 — 부르는 쪽이 더하는 단계를 붙일지 말지 정한다.
    emit: `  Out[part * ${s.O * cols}u + f * ${cols}u + col] = v;`,
  });
}

/**
 * 입력으로 가는 기울기 — 타일링 판.
 *
 * `dX[(n,i), c] = Σ_{o,k} G[(n,o자리), o] · W[o, c, k]` 이고, 합의 짝 `(o, k)` 가
 * 안쪽 축이다. **걸음이 1 보다 크면 나눗셈이 안 떨어지는 자리가 있고 거기로는
 * 아무것도 안 온다** — 그 판정이 오른쪽 타일 안에 있다.
 */
export function convNDGradInputTiled(s: ConvNDShape): string {
  const outSpace = s.outDims.reduce((a, b) => a * b, 1);
  const kSpace = s.kernel.reduce((a, b) => a * b, 1);
  const inSpace = s.inDims.reduce((a, b) => a * b, 1);
  const inStride = suffixStrides(s.inDims);
  const outStride = suffixStrides(s.outDims);
  const kStride = suffixStrides(s.kernel);
  const K = s.O * kSpace;
  const cols = s.N * inSpace;
  return tiledGemm({
    M: s.C, N: cols, K,
    bindings: `@group(0) @binding(0) var<storage, read> G: array<f32>;
@group(0) @binding(1) var<storage, read> Wt: array<f32>;
@group(0) @binding(2) var<storage, read_write> Out: array<f32>;`,
    // 왼쪽: 가중치를 (입력채널 × (출력채널, 커널자리)) 로 본다.
    loadA: `          let oc = kk / ${kSpace}u;
          let kp = kk % ${kSpace}u;
          v = Wt[(oc * ${s.C}u + arow) * ${kSpace}u + kp];`,
    // 오른쪽: 이 입력 자리에 **닿는** 출력 자리를 찾는다. 안 닿으면 0 이다.
    loadB: `          let oc = kk / ${kSpace}u;
${s.kernel.map((size, d) =>
      `          let kk${d} = (kk / ${kStride[d] ?? 1}u) % ${size}u;`).join("\n")}
          let bn = col / ${inSpace}u;
${s.inDims.map((size, d) =>
      `          let i${d} = i32((col / ${inStride[d] ?? 1}u) % ${size}u);`).join("\n")}
          var ok = true;
          var off = 0u;
${s.inDims.map((_, d) => {
      const st = s.stride[d] ?? 1;
      return `          {
            let t${d} = i${d} + ${s.pad[d] ?? 0} - i32(kk${d});
            if (t${d} < 0 || t${d} % ${st} != 0) { ok = false; }
            else {
              let o${d} = t${d} / ${st};
              if (o${d} >= ${s.outDims[d] ?? 0}) { ok = false; }
              else { off = off + u32(o${d}) * ${outStride[d] ?? 1}u; }
            }
          }`;
    }).join("\n")}
          if (ok) {
            v = G[(bn * ${s.O}u + oc) * ${outSpace}u + off];
          }`,
    emit: `  Out[col / ${inSpace}u * ${s.C * inSpace}u + f * ${inSpace}u + col % ${inSpace}u] = v;`,
  });
}

/**
 * 가중치 기울기의 축약 축을 몇 조각으로 쪼갤 것인가.
 *
 * **이 GEMM 은 출력이 작고 축약이 크다.** 층 하나에서 출력이 `(64, 27)` 인데 축약이
 * `배치 × 32 × 32 = 16,384` 인 식이라, 타일 격자가 워크그룹 **한 개**까지 떨어진다 —
 * GPU 하나에 일감 하나다. 축약을 쪼개 여러 워크그룹에 나눠 주고 마지막에 더한다.
 *
 * 조각 수는 격자가 너무 작을 때만 늘린다. 쪼개면 부분합 버퍼와 더하는 단계가 붙으므로
 * 이미 격자가 넉넉한 층에서는 손해다.
 */
export function convGradWeightSplit(s: ConvNDShape): number {
  const cols = s.C * s.kernel.reduce((a, b) => a * b, 1);
  const tiles = Math.ceil(cols / 64) * Math.ceil(s.O / 64);
  const K = s.N * s.outDims.reduce((a, b) => a * b, 1);
  // 워크그룹이 이만큼은 돼야 GPU 가 논다는 소리를 안 듣는다. 넘으면 안 쪼갠다.
  const WANT = 64;
  if (tiles >= WANT) return 1;
  // 조각 하나가 최소 이만큼의 축약은 맡아야 나누는 값이 남는다.
  const MIN_PER_SPLIT = 256;
  return Math.max(1, Math.min(Math.ceil(WANT / tiles), Math.floor(K / MIN_PER_SPLIT)));
}

/** 가중치 기울기의 격자 — 행이 출력 채널, 열이 (입력 채널, 커널 자리), 깊이가 조각. */
export function convGradWeightGrid(s: ConvNDShape): [number, number, number] {
  const cols = s.C * s.kernel.reduce((a, b) => a * b, 1);
  return [Math.ceil(cols / 64), Math.ceil(s.O / 64), convGradWeightSplit(s)];
}

/** 쪼갠 부분합을 더한다. 순서가 정해져 있어 두 번 돌려도 같은 값이다. */
export function sumSplits(n: number, splits: number): string {
  return `
@group(0) @binding(0) var<storage, read> Parts: array<f32>;
@group(0) @binding(1) var<storage, read_write> Out: array<f32>;
@compute @workgroup_size(${WORKGROUP})
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
${flatId(n)}
  var acc = 0.0;
  for (var s = 0u; s < ${splits}u; s = s + 1u) {
    acc = acc + Parts[s * ${n}u + gid];
  }
  Out[gid] = acc;
}`;
}

/** 입력 기울기의 격자 — 행이 입력 채널, 열이 배치·입력 자리. */
export function convGradInputGrid(s: ConvNDShape): [number, number, number] {
  const cols = s.N * s.inDims.reduce((a, b) => a * b, 1);
  return [Math.ceil(cols / 64), Math.ceil(s.C / 64), 1];
}


/**
 * 차원 수에 상관없는 풀링. 채널을 배치에 접어 넣는다.
 *
 * `max` 는 이긴 자리 **하나**로만 보낸다 — 동점이면 먼저 나온 자리다. `amax` 가
 * 고르게 나누는 것과 다르고, torch 의 풀링이 그렇다.
 */
export interface PoolNDShape {
  readonly NC: number;
  readonly inDims: readonly number[];
  readonly kernel: readonly number[];
  readonly stride: readonly number[];
  readonly outDims: readonly number[];
}

export function poolNDKey(p: PoolNDShape): string {
  return [p.NC, p.inDims, p.kernel, p.stride].join("|");
}

export function poolNDForward(p: PoolNDShape, kind: "max" | "avg"): string {
  const inSpace = p.inDims.reduce((a, b) => a * b, 1);
  const outSpace = p.outDims.reduce((a, b) => a * b, 1);
  const kCount = p.kernel.reduce((a, b) => a * b, 1);
  const inStride = suffixStrides(p.inDims);
  const outStride = suffixStrides(p.outDims);
  const n = p.NC * outSpace;
  const decode = p.outDims.map((_, d) =>
    `  let o${d} = (r / ${outStride[d] ?? 1}u) % ${p.outDims[d] ?? 1}u;`).join("\n");
  const open: string[] = [];
  const close: string[] = [];
  const terms: string[] = [];
  for (const [d, size] of p.kernel.entries()) {
    open.push(`    for (var k${d} = 0u; k${d} < ${size}u; k${d} = k${d} + 1u) {`);
    close.push("    }");
    terms.push(`(o${d} * ${p.stride[d] ?? 1}u + k${d}) * ${inStride[d] ?? 1}u`);
  }
  const init = kind === "max" ? "X[base]" : "0.0";
  const step = kind === "max" ? "acc = max(acc, v);" : "acc = acc + v;";
  const done = kind === "max" ? "acc" : `acc / ${kCount.toFixed(1)}`;
  return `
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> Out: array<f32>;
@compute @workgroup_size(${WORKGROUP})
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
${flatId(n)}
  let plane = gid / ${outSpace}u;
  let r = gid % ${outSpace}u;
${decode}
  let base = plane * ${inSpace}u + ${terms.map((t) => t.replace(/k\d+/g, "0u")).join(" + ")};
  var acc = ${init};
${open.join("\n")}
      let v = X[plane * ${inSpace}u + ${terms.join(" + ")}];
      ${step}
${close.join("\n")}
  Out[gid] = ${done};
}`;
}

export function poolNDBackward(p: PoolNDShape, kind: "max" | "avg"): string {
  const inSpace = p.inDims.reduce((a, b) => a * b, 1);
  const outSpace = p.outDims.reduce((a, b) => a * b, 1);
  const kCount = p.kernel.reduce((a, b) => a * b, 1);
  const inStride = suffixStrides(p.inDims);
  const outStride = suffixStrides(p.outDims);
  const n = p.NC * inSpace;
  const decode = p.inDims.map((_, d) =>
    `  let i${d} = i32((r / ${inStride[d] ?? 1}u) % ${p.inDims[d] ?? 1}u);`).join("\n");
  const open: string[] = [];
  const close: string[] = [];
  const oTerms: string[] = [];
  const wTerms: string[] = [];
  for (const [d, size] of p.outDims.entries()) {
    const st = p.stride[d] ?? 1;
    open.push(`    for (var o${d} = 0; o${d} < ${size}; o${d} = o${d} + 1) {`);
    open.push(`      let d${d} = i${d} - o${d} * ${st};`);
    open.push(`      if (d${d} < 0 || d${d} >= ${p.kernel[d] ?? 1}) { continue; }`);
    close.push("    }");
    oTerms.push(`u32(o${d}) * ${outStride[d] ?? 1}u`);
    wTerms.push(`u32(o${d} * ${st}) * ${inStride[d] ?? 1}u`);
  }
  const kOpen: string[] = [];
  const kClose: string[] = [];
  const kTerms: string[] = [];
  const kMatch: string[] = [];
  for (const [d, size] of p.kernel.entries()) {
    kOpen.push(`        for (var m${d} = 0u; m${d} < ${size}u; m${d} = m${d} + 1u) {`);
    kClose.push("        }");
    kTerms.push(`m${d} * ${inStride[d] ?? 1}u`);
    kMatch.push(`m${d} == u32(d${d})`);
  }
  const body = kind === "avg"
    ? `      acc = acc + G[plane * ${outSpace}u + ${oTerms.join(" + ")}] / ${kCount.toFixed(1)};`
    : `      {
        let wbase = plane * ${inSpace}u + ${wTerms.join(" + ")};
        var best = X[wbase];
        var win = true;
${kOpen.join("\n")}
          let v = X[wbase + ${kTerms.join(" + ")}];
          if (v > best) { best = v; }
${kClose.join("\n")}
        // 동점이면 **먼저 나온 자리**가 이긴다. 앞자리에 같은 값이 있으면 진다.
        var earlier = false;
${kOpen.join("\n")}
          let idx = ${kTerms.join(" + ")};
          let mine = ${wTerms.map((_, d) => `u32(d${d}) * ${inStride[d] ?? 1}u`).join(" + ")};
          if (idx < mine && X[wbase + idx] == best) { earlier = true; }
${kClose.join("\n")}
        win = (X[wbase + ${wTerms.map((_, d) => `u32(d${d}) * ${inStride[d] ?? 1}u`).join(" + ")}] == best) && !earlier;
        if (win) { acc = acc + G[plane * ${outSpace}u + ${oTerms.join(" + ")}]; }
      }`;
  // **평균은 입력을 안 본다.** 그런데 `X` 를 선언만 해두면 `layout: "auto"` 가 안 쓰는
  // 바인딩을 빼버리고, 부르는 쪽이 버퍼 셋을 넘기면 "binding index 0 not present" 로
  // 거절한다. 그 거절은 예외가 아니라 **무효한 명령 버퍼**여서, 역방향이 통째로 안
  // 돌면서 학습만 조용히 멈춘다. 실제로 ResNet 벤치가 그 상태로 수를 냈다.
  const decl = kind === "max"
    ? `@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> G: array<f32>;
@group(0) @binding(2) var<storage, read_write> Out: array<f32>;`
    : `@group(0) @binding(0) var<storage, read> G: array<f32>;
@group(0) @binding(1) var<storage, read_write> Out: array<f32>;`;
  return `
${decl}
@compute @workgroup_size(${WORKGROUP})
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
${flatId(n)}
  let plane = gid / ${inSpace}u;
  let r = gid % ${inSpace}u;
${decode}
  var acc = 0.0;
${open.join("\n")}
${body}
${close.join("\n")}
  Out[gid] = acc;
}`;
}

/** 풀링 역방향이 받는 버퍼 수. 종류마다 다르므로 부르는 쪽이 여기서 받아 간다. */
export function poolNDBackwardNeedsInput(kind: "max" | "avg"): boolean {
  return kind === "max";
}

/**
 * 창 목록 — 축마다 각 출력 칸이 덮는 구간.
 *
 * 고정 창과 적응형을 **같은 모양으로** 넘기려고 만든 자리다. 고정은 `start = o·stride`
 * 에 길이가 일정하고, 적응형은 `floor(o·n/want)` 부터 `ceil((o+1)·n/want)` 까지라
 * 길이가 자리마다 다르다. 규칙을 셰이더 안에 둘로 적으면 한쪽만 고치는 날이 온다.
 */
export interface PoolWindows {
  readonly NC: number;
  readonly inDims: readonly number[];
  /** 축마다 `[시작, 끝)` 의 목록. 길이가 그 축의 출력 크기다. */
  readonly axes: readonly (readonly (readonly [number, number])[])[];
}

export function poolWindowsKey(p: PoolWindows): string {
  return [p.NC, p.inDims, p.axes.map((a) => a.map((w) => w.join(":")).join(","))]
    .join("|");
}

/**
 * 최댓값과 **이긴 자리**를 한 번에 낸다.
 *
 * 자리는 torch 의 규약대로 **평면 안의 평평한 번호**다 — 배치·채널마다 0 부터 다시
 * 센다. `maxUnpool` 이 이 번호를 그대로 되돌린다.
 *
 * **값을 여기서 같이 낸다.** 값을 다른 커널에서 구하면 "자리는 A 인데 값은 B" 인
 * 상태가 만들어질 수 있고, 둘 다 그럴듯해서 아무 눈에도 안 띈다.
 *
 * 동점이면 **먼저 나온 자리**가 이긴다 — torch 가 그렇다. 창을 도는 순서가 평평한
 * 번호가 커지는 순서이고 비교가 `>` 라서, 첫 최댓값이 그대로 남는다.
 */
export function poolMaxWithIndex(p: PoolWindows): string {
  const inSpace = p.inDims.reduce((a, b) => a * b, 1);
  const outDims = p.axes.map((a) => a.length);
  const outSpace = outDims.reduce((a, b) => a * b, 1);
  const inStride = suffixStrides(p.inDims);
  const outStride = suffixStrides(outDims);
  const n = p.NC * outSpace;

  // 창 표를 셰이더에 상수로 굽는다. 출력 칸 수가 작아서 값이 싸고, 자리마다 길이가
  // 다른 적응형도 같은 모양으로 실린다.
  const tables = p.axes.map((axis, d) => {
    const starts = axis.map((w) => `${w[0]}u`).join(", ");
    const ends = axis.map((w) => `${w[1]}u`).join(", ");
    return `var<private> S${d}: array<u32, ${axis.length}> = `
      + `array<u32, ${axis.length}>(${starts});\n`
      + `var<private> E${d}: array<u32, ${axis.length}> = `
      + `array<u32, ${axis.length}>(${ends});`;
  }).join("\n");

  const decode = outDims.map((size, d) =>
    `  let o${d} = (r / ${outStride[d] ?? 1}u) % ${size}u;`).join("\n");
  const open: string[] = [];
  const close: string[] = [];
  const terms: string[] = [];
  for (let d = 0; d < p.axes.length; d++) {
    open.push(`  for (var k${d} = S${d}[o${d}]; k${d} < E${d}[o${d}]; k${d} = k${d} + 1u) {`);
    close.push("  }");
    terms.push(`k${d} * ${inStride[d] ?? 1}u`);
  }
  const first = p.axes.map((_, d) => `S${d}[o${d}] * ${inStride[d] ?? 1}u`).join(" + ");

  return `
${tables}
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> Out: array<f32>;
@group(0) @binding(2) var<storage, read_write> Idx: array<f32>;
@compute @workgroup_size(${WORKGROUP})
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
${flatId(n)}
  let plane = gid / ${outSpace}u;
  let r = gid % ${outSpace}u;
${decode}
  var bestOff = ${first};
  var best = X[plane * ${inSpace}u + bestOff];
${open.join("\n")}
    let off = ${terms.join(" + ")};
    let v = X[plane * ${inSpace}u + off];
    if (v > best) { best = v; bestOff = off; }
${close.join("\n")}
  Out[gid] = best;
  Idx[gid] = f32(bestOff);
}`;
}

/**
 * 자리표를 따라 기울기를 되돌린다. 이긴 자리로만 간다.
 *
 * 순방향이 이미 자리를 정해 두었으므로 여기서 다시 고르지 않는다 — 다시 고르면 그
 * 고르기가 순방향과 갈릴 수 있고, 동점이 있을 때 정확히 그렇게 된다.
 *
 * 입력 자리마다 **자기를 가리키는 출력 칸을 찾아 더한다.** 흩뿌리기가 아니라 모으기라
 * 스레드끼리 같은 칸에 안 쓴다 — 창이 겹치면 한 입력이 여러 출력에게 이길 수 있다.
 */
export function poolMaxIndexBackward(p: PoolWindows): string {
  const inSpace = p.inDims.reduce((a, b) => a * b, 1);
  const outSpace = p.axes.map((a) => a.length).reduce((a, b) => a * b, 1);
  const n = p.NC * inSpace;
  return `
@group(0) @binding(0) var<storage, read> Idx: array<f32>;
@group(0) @binding(1) var<storage, read> G: array<f32>;
@group(0) @binding(2) var<storage, read_write> Out: array<f32>;
@compute @workgroup_size(${WORKGROUP})
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
${flatId(n)}
  let plane = gid / ${inSpace}u;
  let r = gid % ${inSpace}u;
  var acc = 0.0;
  for (var o = 0u; o < ${outSpace}u; o = o + 1u) {
    let at = plane * ${outSpace}u + o;
    if (u32(Idx[at]) == r) { acc = acc + G[at]; }
  }
  Out[gid] = acc;
}`;
}

/**
 * 자리표가 가리키는 칸에 값을 놓는다. 나머지는 0 — `MaxUnpool` 이다.
 *
 * **모으기로 쓴다.** 흩뿌리면 겹치는 자리에서 스레드 순서가 답을 정하는데, 그것은
 * 실행마다 달라질 수 있는 답이라 대조가 안 된다. 출력 칸마다 자기를 가리키는 입력을
 * 찾아 오면 순서가 정해진다 — 여럿이면 **마지막 것**이 남고, torch 의 흩뿌리기와
 * 같은 답이다.
 */
export function unpoolFromIndex(
  NC: number, inSpace: number, outSpace: number,
): string {
  const n = NC * outSpace;
  return `
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> Idx: array<f32>;
@group(0) @binding(2) var<storage, read_write> Out: array<f32>;
@compute @workgroup_size(${WORKGROUP})
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
${flatId(n)}
  let plane = gid / ${outSpace}u;
  let r = gid % ${outSpace}u;
  var got = 0.0;
  for (var i = 0u; i < ${inSpace}u; i = i + 1u) {
    let at = plane * ${inSpace}u + i;
    if (u32(Idx[at]) == r) { got = X[at]; }
  }
  Out[gid] = got;
}`;
}

/** `MaxUnpool` 의 역방향 — 값이 간 자리에서 그대로 받아 온다. 채우기의 반대다. */
export function unpoolFromIndexBackward(
  NC: number, inSpace: number, outSpace: number,
): string {
  const n = NC * inSpace;
  return `
@group(0) @binding(0) var<storage, read> Idx: array<f32>;
@group(0) @binding(1) var<storage, read> G: array<f32>;
@group(0) @binding(2) var<storage, read_write> Out: array<f32>;
@compute @workgroup_size(${WORKGROUP})
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
${flatId(n)}
  let plane = gid / ${inSpace}u;
  let r = gid % ${inSpace}u;
  Out[gid] = G[plane * ${outSpace}u + u32(Idx[plane * ${inSpace}u + r])];
}`;
}

/**
 * 최근접 이웃 확대. 각 출력 자리가 자기를 낳은 입력 자리를 그대로 읽는다.
 *
 * 역방향은 자기를 읽어 간 출력들을 모으는 것이고, 배율이 정수라 그 개수가 일정하다.
 */
export function upsampleNearest(
  NC: number,
  inDims: readonly number[],
  scale: number,
): string {
  const inSpace = inDims.reduce((a, b) => a * b, 1);
  const outDims = inDims.map((d) => d * scale);
  const outSpace = outDims.reduce((a, b) => a * b, 1);
  const inStride = suffixStrides(inDims);
  const outStride = suffixStrides(outDims);
  const n = NC * outSpace;
  const terms = inDims.map((_, d) =>
    `((r / ${outStride[d] ?? 1}u) % ${outDims[d] ?? 1}u / ${scale}u) * ${inStride[d] ?? 1}u`);
  return `
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> Out: array<f32>;
@compute @workgroup_size(${WORKGROUP})
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
${flatId(n)}
  let plane = gid / ${outSpace}u;
  let r = gid % ${outSpace}u;
  Out[gid] = X[plane * ${inSpace}u + ${terms.join(" + ")}];
}`;
}

/** 확대의 역방향 — 자기를 읽어 간 자리를 모은다. */
export function upsampleNearestBackward(
  NC: number,
  inDims: readonly number[],
  scale: number,
): string {
  const inSpace = inDims.reduce((a, b) => a * b, 1);
  const outDims = inDims.map((d) => d * scale);
  const outSpace = outDims.reduce((a, b) => a * b, 1);
  const inStride = suffixStrides(inDims);
  const outStride = suffixStrides(outDims);
  const n = NC * inSpace;
  const decode = inDims.map((_, d) =>
    `  let i${d} = (r / ${inStride[d] ?? 1}u) % ${inDims[d] ?? 1}u;`).join("\n");
  const open: string[] = [];
  const close: string[] = [];
  const terms: string[] = [];
  for (const [d] of inDims.entries()) {
    open.push(`    for (var s${d} = 0u; s${d} < ${scale}u; s${d} = s${d} + 1u) {`);
    close.push("    }");
    terms.push(`(i${d} * ${scale}u + s${d}) * ${outStride[d] ?? 1}u`);
  }
  return `
@group(0) @binding(0) var<storage, read> G: array<f32>;
@group(0) @binding(1) var<storage, read_write> Out: array<f32>;
@compute @workgroup_size(${WORKGROUP})
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
${flatId(n)}
  let plane = gid / ${inSpace}u;
  let r = gid % ${inSpace}u;
${decode}
  var acc = 0.0;
${open.join("\n")}
      acc = acc + G[plane * ${outSpace}u + ${terms.join(" + ")}];
${close.join("\n")}
  Out[gid] = acc;
}`;
}

/**
 * 축 하나를 정렬한다. **자리도 같이 옮긴다** — 값만 옮기면 `argsort` 를 못 만들고,
 * 기울기를 원래 자리로 되돌릴 수도 없다.
 *
 * 삽입 정렬이다. 축이 길어지면 나쁘지만, 여기서 미는 값이 축 길이 열 몇이고
 * **안정 정렬**이라 동점의 순서가 torch 와 같다 — 비토닉 정렬은 그 순서를 안 지킨다.
 * 스레드 하나가 축 하나를 통째로 맡으므로 원자 연산도 없다.
 */
export function sortAxis(
  outer: number,
  len: number,
  inner: number,
  descending: boolean,
): string {
  const n = outer * inner;
  // 안정성을 지키려면 **같은 값에서 멈춰야** 한다. 엄격 부등호를 쓰는 이유다.
  const test = descending ? "cur > prev" : "cur < prev";
  return `
@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read_write> V: array<f32>;
@group(0) @binding(2) var<storage, read_write> I: array<f32>;
@compute @workgroup_size(${WORKGROUP})
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
${flatId(n)}
  let o = gid / ${inner}u;
  let i = gid % ${inner}u;
  let base = o * ${len * inner}u + i;
  for (var k = 0u; k < ${len}u; k = k + 1u) {
    V[base + k * ${inner}u] = A[base + k * ${inner}u];
    I[base + k * ${inner}u] = f32(k);
  }
  for (var k = 1u; k < ${len}u; k = k + 1u) {
    var p = k;
    loop {
      if (p == 0u) { break; }
      let cur = V[base + p * ${inner}u];
      let prev = V[base + (p - 1u) * ${inner}u];
      if (!(${test})) { break; }
      V[base + p * ${inner}u] = prev;
      V[base + (p - 1u) * ${inner}u] = cur;
      let ci = I[base + p * ${inner}u];
      I[base + p * ${inner}u] = I[base + (p - 1u) * ${inner}u];
      I[base + (p - 1u) * ${inner}u] = ci;
      p = p - 1u;
    }
  }
}`;
}

/**
 * 자리 표를 따라 기울기를 원래 자리로 되돌린다.
 *
 * `sort`·`topk`·`median` 의 역방향이 전부 이것이다 — 뽑아 온 자리로만 흘리고
 * 나머지는 0. 값만 떼어 돌려주면 그 자리로 기울기가 안 가고 학습이 조용히 멈춘다.
 */
export function scatterByIndex(
  outer: number,
  len: number,
  inner: number,
  taken: number,
): string {
  const n = outer * len * inner;
  return `
@group(0) @binding(0) var<storage, read> I: array<f32>;
@group(0) @binding(1) var<storage, read> G: array<f32>;
@group(0) @binding(2) var<storage, read_write> Out: array<f32>;
@compute @workgroup_size(${WORKGROUP})
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
${flatId(n)}
  let o = gid / ${len * inner}u;
  let r = gid % ${len * inner}u;
  let k = r / ${inner}u;
  let i = r % ${inner}u;
  var acc = 0.0;
  for (var t = 0u; t < ${taken}u; t = t + 1u) {
    let at = o * ${taken * inner}u + t * ${inner}u + i;
    if (u32(I[at]) == k) { acc = acc + G[at]; }
  }
  Out[gid] = acc;
}`;
}

/**
 * `scatterByIndex` 의 **덮어쓰는** 판. 겹치는 번호에서 마지막에 쓴 것이 남는다.
 *
 * 쌓는 것과 덮는 것의 차이가 `scatter_add` 와 `scatter` 의 차이 전부다 — 번호가
 * 안 겹치면 두 함수가 같은 답을 내므로, 겹치는 번호로 재야만 갈린다.
 *
 * **출력 쪽에서 읽는다.** 입력 쪽에서 쓰면 같은 칸에 여러 스레드가 달려들어
 * 누가 마지막인지가 정해지지 않는다 — 여기서는 각 출력 칸이 자기에게 오는 것을
 * 훑으므로 순서가 정해진다.
 */
export function scatterOverwrite(
  outer: number,
  len: number,
  inner: number,
  taken: number,
): string {
  const n = outer * len * inner;
  return `
@group(0) @binding(0) var<storage, read> I: array<f32>;
@group(0) @binding(1) var<storage, read> G: array<f32>;
@group(0) @binding(2) var<storage, read> Base: array<f32>;
@group(0) @binding(3) var<storage, read_write> Out: array<f32>;
@compute @workgroup_size(${WORKGROUP})
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
${flatId(n)}
  let o = gid / ${len * inner}u;
  let r = gid % ${len * inner}u;
  let k = r / ${inner}u;
  let i = r % ${inner}u;
  var acc = Base[gid];
  for (var t = 0u; t < ${taken}u; t = t + 1u) {
    let at = o * ${taken * inner}u + t * ${inner}u + i;
    if (u32(I[at]) == k) { acc = G[at]; }
  }
  Out[gid] = acc;
}`;
}

/**
 * 누적 최대·최소. 값과 자리를 같이 낸다.
 *
 * **동점이면 나중 자리를 준다** — torch 의 `cummax` 가 그렇다(`argmax` 가 먼저
 * 자리를 주는 것과 반대다). 등호를 포함하는 부등호 하나가 그 차이다.
 */
export function cumExtreme(
  kind: "max" | "min",
  outer: number,
  len: number,
  inner: number,
): string {
  const n = outer * len * inner;
  const better = kind === "max" ? "v >= best" : "v <= best";
  return `
@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read_write> V: array<f32>;
@group(0) @binding(2) var<storage, read_write> I: array<f32>;
@compute @workgroup_size(${WORKGROUP})
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
${flatId(n)}
  let o = gid / ${len * inner}u;
  let r = gid % ${len * inner}u;
  let k = r / ${inner}u;
  let i = r % ${inner}u;
  let base = o * ${len * inner}u + i;
  var best = A[base];
  var at = 0u;
  for (var t = 1u; t <= k; t = t + 1u) {
    let v = A[base + t * ${inner}u];
    if (${better}) { best = v; at = t; }
  }
  V[gid] = best;
  I[gid] = f32(at);
}`;
}

/**
 * 배치 정규화의 채널 통계 — 합과 제곱합을 **한 번에** 낸다.
 *
 * 조립으로 두면 `sumDim` 셋 + `sub` + `square` + `sumDim` 셋 + 나눗셈 몇으로 열 몇
 * dispatch 가 되고, 그것이 층마다 스무 번이다. 실측에서 ResNet 한 스텝의 dispatch
 * 1,636 개 중 태반이 여기서 나왔다.
 *
 * 스레드 하나가 채널 하나를 맡아 배치·공간을 전부 훑는다. 순서가 정해져 있으므로
 * 원자 연산이 없고 두 번 돌리면 같은 값이다.
 */
export function batchNormStats(N: number, C: number, S: number): string {
  return `
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> Mean: array<f32>;
@group(0) @binding(2) var<storage, read_write> Var: array<f32>;
@compute @workgroup_size(${WORKGROUP})
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
${flatId(C)}
  var total = 0.0;
  var sq = 0.0;
  for (var n = 0u; n < ${N}u; n = n + 1u) {
    let base = (n * ${C}u + gid) * ${S}u;
    for (var i = 0u; i < ${S}u; i = i + 1u) {
      let v = X[base + i];
      total = total + v;
      sq = fma(v, v, sq);
    }
  }
  let m = total / ${(N * S).toFixed(1)};
  Mean[gid] = m;
  // **편향추정이다**(n 으로 나눈다) — torch 의 BatchNorm 이 정규화에 쓰는 것이 이것이고,
  // 이동 통계에 넣는 불편추정과는 다른 수다. 하나로 합치면 평가 모드에서만 갈린다.
  Var[gid] = sq / ${(N * S).toFixed(1)} - m * m;
}`;
}

/** 통계를 받아 정규화하고 크기·치우침까지 한 번에 먹인다. */
export function batchNormApply(N: number, C: number, S: number, eps: number): string {
  const n = N * C * S;
  return `
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> Mean: array<f32>;
@group(0) @binding(2) var<storage, read> Var: array<f32>;
@group(0) @binding(3) var<storage, read> Wt: array<f32>;
@group(0) @binding(4) var<storage, read> B: array<f32>;
@group(0) @binding(5) var<storage, read_write> Out: array<f32>;
@compute @workgroup_size(${WORKGROUP})
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
${flatId(n)}
  let c = (gid / ${S}u) % ${C}u;
  Out[gid] = (X[gid] - Mean[c]) * inverseSqrt(Var[c] + ${eps}) * Wt[c] + B[c];
}`;
}

/**
 * 배치 정규화의 역방향.
 *
 * **평균과 분산이 그래프 안에 있다.** 밖으로 빼면 입력 기울기가 어긋나고 `weight`
 * 에는 아예 안 온다 — 코어가 오래 겪은 자리다. 식은
 *
 *     dx = γ·σ⁻¹·(dy − mean(dy) − x̂·mean(dy·x̂))
 *
 * 이고, 여기 필요한 두 평균을 채널마다 한 번 세고 그 뒤에 원소별로 먹인다.
 */
export function batchNormStatsBackward(N: number, C: number, S: number): string {
  return `
@group(0) @binding(0) var<storage, read> Xh: array<f32>;
@group(0) @binding(1) var<storage, read> G: array<f32>;
@group(0) @binding(2) var<storage, read_write> SumG: array<f32>;
@group(0) @binding(3) var<storage, read_write> SumGXh: array<f32>;
@compute @workgroup_size(${WORKGROUP})
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
${flatId(C)}
  var sg = 0.0;
  var sgx = 0.0;
  for (var n = 0u; n < ${N}u; n = n + 1u) {
    let base = (n * ${C}u + gid) * ${S}u;
    for (var i = 0u; i < ${S}u; i = i + 1u) {
      let gv = G[base + i];
      sg = sg + gv;
      sgx = fma(gv, Xh[base + i], sgx);
    }
  }
  SumG[gid] = sg;
  SumGXh[gid] = sgx;
}`;
}

export function batchNormBackwardApply(
  N: number, C: number, S: number,
): string {
  const n = N * C * S;
  const count = (N * S).toFixed(1);
  return `
@group(0) @binding(0) var<storage, read> Xh: array<f32>;
@group(0) @binding(1) var<storage, read> G: array<f32>;
@group(0) @binding(2) var<storage, read> SumG: array<f32>;
@group(0) @binding(3) var<storage, read> SumGXh: array<f32>;
@group(0) @binding(4) var<storage, read> Wt: array<f32>;
@group(0) @binding(5) var<storage, read> InvStd: array<f32>;
@group(0) @binding(6) var<storage, read_write> Out: array<f32>;
@compute @workgroup_size(${WORKGROUP})
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
${flatId(n)}
  let c = (gid / ${S}u) % ${C}u;
  let xh = Xh[gid];
  Out[gid] = Wt[c] * InvStd[c] *
    (G[gid] - SumG[c] / ${count} - xh * SumGXh[c] / ${count});
}`;
}

/**
 * 옵티마이저 한 걸음 — **파라미터와 상태를 제자리에서 고친다.**
 *
 * 조립판은 파라미터 하나에 dispatch 넷이 들었다(모멘텀 곱, 기울기 더하기, 학습률 곱,
 * 빼기). ResNet-18 은 파라미터 텐서가 예순둘이라 그것만 이백사십 번이고, 실측에서
 * 원소별 dispatch 사백칠십 개의 절반을 넘었다.
 *
 * **읽으면서 같은 자리에 쓴다.** 스레드 하나가 자기 원소만 보므로 순서가 섞일 자리가
 * 없다 — 브로드캐스팅도 축약도 없는 원소별 갱신이라 가능한 일이다.
 */
export function sgdStep(
  n: number, lr: number, momentum: number, weightDecay = 0,
): string {
  const hasMomentum = momentum !== 0;
  // **가중치 감쇠는 기울기에 더한다.** 파라미터를 따로 줄이는 것과 다른 수다 —
  // 모멘텀 버퍼가 감쇠를 함께 들고 가느냐가 갈린다. torch 의 SGD 가 이쪽이다.
  const grad = weightDecay !== 0
    ? `G[gid] + P[gid] * ${weightDecay}`
    : "G[gid]";
  return `
@group(0) @binding(0) var<storage, read_write> P: array<f32>;
@group(0) @binding(1) var<storage, read> G: array<f32>;
${hasMomentum ? "@group(0) @binding(2) var<storage, read_write> Buf: array<f32>;" : ""}
@compute @workgroup_size(${WORKGROUP})
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
${flatId(n)}
  let gv = ${grad};
${hasMomentum
    ? `  let b = Buf[gid] * ${momentum} + gv;
  Buf[gid] = b;
  P[gid] = P[gid] - b * ${lr};`
    : `  P[gid] = P[gid] - gv * ${lr};`}
}`;
}

/** Adam 한 걸음. 편향 보정을 스텝 수로 받아 굽지 않는다 — 매 스텝 달라진다. */
export function adamStep(
  n: number, lr: number, beta1: number, beta2: number, eps: number,
): string {
  return `
@group(0) @binding(0) var<storage, read_write> P: array<f32>;
@group(0) @binding(1) var<storage, read> G: array<f32>;
@group(0) @binding(2) var<storage, read_write> M: array<f32>;
@group(0) @binding(3) var<storage, read_write> V: array<f32>;
@group(0) @binding(4) var<storage, read> Corr: array<f32>;
@compute @workgroup_size(${WORKGROUP})
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
${flatId(n)}
  let gv = G[gid];
  let m = M[gid] * ${beta1} + gv * ${1 - beta1};
  let v = V[gid] * ${beta2} + gv * gv * ${1 - beta2};
  M[gid] = m;
  V[gid] = v;
  // Corr[0] = 1-β₁ᵗ, Corr[1] = 1-β₂ᵗ. 스텝마다 달라지므로 굽지 않고 받는다.
  P[gid] = P[gid] - ${lr} * (m / Corr[0]) / (sqrt(v / Corr[1]) + ${eps});
}`;
}

export function rmspropStep(n: number, lr: number, alpha: number, eps: number): string {
  return `
@group(0) @binding(0) var<storage, read_write> P: array<f32>;
@group(0) @binding(1) var<storage, read> G: array<f32>;
@group(0) @binding(2) var<storage, read_write> S: array<f32>;
@compute @workgroup_size(${WORKGROUP})
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
${flatId(n)}
  let gv = G[gid];
  let s = S[gid] * ${alpha} + gv * gv * ${1 - alpha};
  S[gid] = s;
  P[gid] = P[gid] - ${lr} * gv / (sqrt(s) + ${eps});
}`;
}

/**
 * 이동 통계 갱신 — `running ← (1−t)·running + t·new`, 두 개를 한 번에.
 *
 * 조립판은 BatchNorm 하나에 여덟 dispatch 였고 층이 스무 개다. 채널 수만큼만 도는
 * 작은 일이라 커널 하나로 충분하다.
 */
export function runningStats(C: number, momentum: number, unbias: number): string {
  return `
@group(0) @binding(0) var<storage, read_write> RunMean: array<f32>;
@group(0) @binding(1) var<storage, read_write> RunVar: array<f32>;
@group(0) @binding(2) var<storage, read> Mean: array<f32>;
@group(0) @binding(3) var<storage, read> Var: array<f32>;
@compute @workgroup_size(${WORKGROUP})
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
${flatId(C)}
  RunMean[gid] = RunMean[gid] * ${1 - momentum} + Mean[gid] * ${momentum};
  // **이동 통계에는 불편추정이 들어간다** — 정규화에 쓰는 편향추정과 다른 수다.
  RunVar[gid] = RunVar[gid] * ${1 - momentum} + Var[gid] * ${unbias * momentum};
}`;
}

/** 값 하나로 채운다. 기울기 씨앗(`backward()` 의 1.0)과 `zeros` 가 쓴다. */
export function fill(n: number, value: number): string {
  return `
@group(0) @binding(0) var<storage, read_write> Out: array<f32>;
@compute @workgroup_size(${WORKGROUP})
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
${flatId(n)}
  Out[gid] = ${f32lit(value)};
}`;
}

/**
 * 자리마다 난수를 만드는 해시. **상태를 안 들고 자리와 씨앗만으로 뽑는다.**
 *
 * GPU 에는 순서라는 것이 없다. 스레드가 어떤 차례로 도는지 정해져 있지 않으므로
 * "다음 난수" 를 물려주는 방식은 여기서 뜻이 없고, 같은 입력에 같은 답이 나오지도
 * 않는다. 그래서 `해시(자리, 씨앗)` 로 뽑는다 — 자리마다 독립이고, 같은 씨앗이면
 * 같은 답이 나오고, 스레드 사이에 주고받을 것이 없다.
 *
 * 쓰는 것은 잘 알려진 정수 뒤섞기(Wang/Jenkins 계열)다. 암호용이 아니고 통계용도
 * 아니다 — dropout 이 자리를 고르는 데 쓰는 만큼이면 된다. 통계적 성질이 걸린 일이
 * 생기면 그때는 이것을 쓰면 안 되고, 그 사실을 여기 적어 둔다.
 */
const RANDOM_PRELUDE = `
fn hash_u32(v: u32) -> u32 {
  var x = v;
  x = (x ^ 61u) ^ (x >> 16u);
  x = x + (x << 3u);
  x = x ^ (x >> 4u);
  x = x * 0x27d4eb2du;
  x = x ^ (x >> 15u);
  return x;
}
fn rand01(gid: u32, seed: u32) -> f32 {
  // 24 비트만 쓴다 — f32 의 가수부가 그만큼이라 그 위는 어차피 안 실린다.
  let h = hash_u32(gid * 0x9e3779b9u + hash_u32(seed));
  return f32(h >> 8u) * (1.0 / 16777216.0);
}`;

/**
 * Dropout 의 가림막. **살아남은 자리에 `1/(1-p)` 를 적는다** — 0 아니면 그 값이다.
 *
 * 가림막을 따로 내놓는 이유는 역방향 때문이다. 순방향에서 뽑은 것과 **같은** 가림막을
 * 역방향이 봐야 하는데, 다시 뽑으면 씨앗이 같아도 그것을 보장하려고 씨앗을 들고
 * 다녀야 한다. 만들어 두고 곱하는 편이 짧고, 곱셈의 미분은 이미 있다.
 */
/**
 * `[lo, hi)` 균등난수.
 *
 * dropout 이 쓰던 해시를 그대로 쓴다 — 자리와 씨앗만으로 뽑으므로 GPU 에 순서가
 * 없어도 같은 답이 나온다. `rrelu` 가 학습 모드에서 기울기를 여기서 뽑는다.
 */
export function uniformFill(n: number, lo: number, hi: number,
                            seed: number): string {
  return `${RANDOM_PRELUDE}
@group(0) @binding(0) var<storage, read_write> Out: array<f32>;
@compute @workgroup_size(${WORKGROUP})
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
${flatId(n)}
  Out[gid] = ${f32lit(lo)} + rand01(gid, ${seed >>> 0}u) * ${f32lit(hi - lo)};
}`;
}

export function dropoutMask(n: number, p: number, seed: number): string {
  const keep = f32lit(1 / (1 - p));
  return `${RANDOM_PRELUDE}
@group(0) @binding(0) var<storage, read_write> Out: array<f32>;
@compute @workgroup_size(${WORKGROUP})
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
${flatId(n)}
  Out[gid] = select(0.0, ${keep}, rand01(gid, ${seed >>> 0}u) >= ${f32lit(p)});
}`;
}
