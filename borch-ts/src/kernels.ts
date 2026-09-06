/**
 * WGSL kernels are **generated from a table of operations.**
 *
 * ## Why generate from a table
 *
 * Building the sister library meant writing new derivative expressions by hand several
 * times, and each one created a place to be wrong. The Cholesky backward survived
 * because it was compared against torch before being written; `roll` and
 * `masked_select` passed 746 golden cases with the right values and a severed graph.
 * Writing **a name, a forward expression and a backward expression on one line and
 * letting the kernel come out** reduces that surface.
 *
 * ## The shape is baked in as a constant
 *
 * With the divisor in a uniform, the compiler cannot do strength reduction, and a GPU
 * has no integer division hardware. That alone moved conv from 43% to 284% of TF.js
 * (measured, `tests/browser/wgsl_conv.js`). So the shape goes into the shader string,
 * and **shape signature → pipeline cache** becomes the library's structure. It is not an
 * optimisation.
 */

/** An elementwise unary — `fwd` is written in terms of `x`, and `bwd` is **what the
 *  gradient is multiplied by.** */
export interface UnarySpec {
  /** The forward WGSL expression. The input is `x`. */
  readonly fwd: string;
  /** The derivative's WGSL expression. `x` is the input and `o` is the forward result
   *  — it is not recomputed. */
  readonly bwd: string;
  /** WGSL definitions for the helpers an expression calls. Only what does not fit on
   *  one line comes here. */
  readonly prelude?: string;
}

/** An elementwise binary — `da` and `db` are what the gradient to each input is
 *  multiplied by. */
export interface BinarySpec {
  readonly fwd: string;
  readonly da: string;
  readonly db: string;
  readonly prelude?: string;
}

/**
 * The helper behind the erf family.
 *
 * Abramowitz & Stegun 7.1.26, with **the same coefficients** as the core
 * (`borch/_ops.py`). Three implementations using three approximations means that when
 * the values diverge, there is no telling whether the implementations diverged or the
 * approximations did.
 *
 * The primitive is `erfc_pos` — a polynomial × exp(-y²), with no subtraction, so no
 * digits are lost in the tail.
 *
 * **Near the origin it parts from the core.** The core computes `1 - erfc_pos(|x|)` in
 * float64 to avoid the cancellation, and WGSL has no f64. So |x| < 0.5 is answered by a
 * series — the next term there is 4e-7, far below this project's tolerance (1e-4).
 * Subtracting in f32 instead brings back exactly the place the core confirmed by
 * measurement (5,124 points out of 46,000).
 */
/**
 * The NaN test.
 *
 * **`x != x` did not work here.** WGSL has no `isNan` builtin, so that was used, and
 * `nansum` went on adding NaNs while `nanmean`'s count came out 4 instead of 3 — which
 * means the shader compiler folded the float comparison as though NaN did not exist.
 *
 * An exponent of all ones with a non-zero mantissa is NaN. Seen as bits, there is
 * nothing to fold.
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
  // x >= 0 uses the primitive as it is — there is no subtraction at all.
  return select(2.0 - erfc_pos(abs(x)), erfc_pos(x), x >= 0.0);
}`;

/**
 * The gamma family. **The same expressions as the core (numpy)** — written differently
 * in the two places, the golden cannot say which is right.
 *
 * `lgamma` is Lanczos (g=7, n=9); `digamma` and `trigamma` push up past 6 with the
 * recurrence and then use Stirling's asymptotic expansion. The asymptotic form does not
 * hold at small x, and leaving out the pushing goes quietly wrong near 0 alone.
 */
const GAMMA_PRELUDE = `
const LANCZOS = array<f32, 9>(
  0.9999999999998099, 676.5203681218851, -1259.1392167224028,
  771.3234287776531, -176.6150291621406, 12.507343278686905,
  -0.13857109526572012, 9.984369578019572e-6, 1.5056327351493116e-7);
fn lgamma_(x: f32) -> f32 {
  // The reflection formula folds the negative side: Γ(x)Γ(1−x) = π/sin(πx).
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
  // Pushed above 6 — the asymptotic form does not hold below it.
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
 * The inverse of `erf`. It splits the range in two — the middle and the tail converge
 * differently, so one expression cannot cover both, and forcing it puts one side over
 * the tolerance. A single Newton step tightens it at the end.
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
  // The approximation alone sits at the edge of the tolerance. One Newton step clears
  // it.
  let err = erf_(out) - x;
  out = out - err / (1.1283791670955126 * exp(-out * out));
  return out;
}`;

/**
 * The zeroth-order modified Bessel function `I₀`. It uses the tables of Abramowitz &
 * Stegun 9.8.1 and 9.8.2 verbatim.
 *
 * **|x| splits at 3.75** — below it, an even series in `t = x/3.75`; above it,
 * `exp(|x|)/√|x|` in front multiplied by a series in `3.75/|x|`. Writing one of the two
 * alone is wholly wrong in the other range, and asking only with small values hides
 * that.
 */
const I0_PRELUDE = `
fn i0_(x: f32) -> f32 {
  let a = abs(x);
  if (a < 3.75) {
    let t = x / 3.75;
    let z = t * t;
    return 1.0 + z * (3.5156229 + z * (3.0899424 + z * (1.2067492
      + z * (0.2659732 + z * (0.0360768 + z * 0.0045813)))));
  }
  let t = 3.75 / a;
  let poly = 0.39894228 + t * (0.01328592 + t * (0.00225319 + t * (-0.00157565
    + t * (0.00916281 + t * (-0.02057706 + t * (0.02635537
    + t * (-0.01647633 + t * 0.00392377)))))));
  return exp(a) / sqrt(a) * poly;
}

// **i0's derivative — i1.** Written as a series:
//
//     i1(x) = Σ (x/2)^(2k+1) / (k! (k+1)!)
//
// Carrying each term forward by multiplying the previous one keeps the factorials from
// overflowing. **Every term is positive so none cancel**, and no digits are lost — this
// is convergence rather than approximation. The function is odd, so the sign is attached
// at the end. The core (numpy) uses the same series, and it was confirmed to agree with
// torch to a relative error of 1.6e-15.
//
// The loop count is a constant not because WGSL dislikes a conditional break but because
// **at f32, sixty terms already cover |x| ≤ 30.** Beyond that, i0 itself overruns f32.
//
// (The shader text lives inside a template literal — a backtick here closes the string.
//  This is the third time this repository has stepped on that: runner.html, fft.ts, and
//  here.)
fn i1_(x: f32) -> f32 {
  let half = abs(x) * 0.5;
  var term = half;
  var total = half;
  for (var k = 1u; k < 60u; k = k + 1u) {
    term = term * (half * half) / (f32(k) * (f32(k) + 1.0));
    total = total + term;
  }
  return select(-total, total, x >= 0.0);
}`;

/**
 * **The scaled Bessel family — `iₙ(x)·exp(-|x|)` and `kₙ(x)·exp(x)`.**
 *
 * Scaled from the first term of the series, never as a product. `i0(90)` is about 4e37
 * and `i0(200)` is past f32's ceiling entirely, so a shader that summed `iₙ` and then
 * multiplied by `exp(-|x|)` would answer `inf` and then `nan` where the true values are
 * 0.042111 and 0.0282272 — the scaled function is bounded by 1 everywhere. `k` runs the
 * other way: `k₀(20)` is 5.74e-10 and underflows long before the scaled form does.
 *
 * The same split as the core, at the same crossovers, and the crossovers were measured
 * rather than chosen: `k`'s series and asymptotic first met at 2 (where the textbooks
 * split the *unscaled* pair) and that was worth two digits — `k₀(2)` came back 0.0906
 * against 0.1139, on the two points either side of the seam and nowhere else.
 */
const BESSEL_SCALED_PRELUDE = `
fn i_scaled_(x: f32, order: u32) -> f32 {
  let a = abs(x);
  if (a < 15.0) {
    let half = a * 0.5;
    // The leading term carries exp(-a) from the start: (a/2)^order / order! · exp(-a).
    var term = exp(-a);
    for (var j = 1u; j <= order; j = j + 1u) { term = term * half / f32(j); }
    var total = term;
    for (var k = 1u; k < 200u; k = k + 1u) {
      term = term * (half * half) / (f32(k) * (f32(k) + f32(order)));
      total = total + term;
    }
    return total;
  }
  // Hankel's asymptotic for the scaled function, with mu = 4·order².
  let mu = 4.0 * f32(order) * f32(order);
  var series = 1.0;
  var term = 1.0;
  for (var k = 1u; k < 9u; k = k + 1u) {
    let m = 2.0 * f32(k) - 1.0;
    term = term * -(mu - m * m) / (8.0 * a * f32(k));
    series = series + term;
  }
  return series / sqrt(6.283185307179586 * a);
}

// **k is a minimax table here and a series in the core, and that is not a duplication
// to remove.** The core sums the harmonic-weighted series in float64 and subtracts
// (log(x/2) + gamma) * i0(x) from it; in f32 that subtraction is where the digits go.
// Measured: at x = 9.9 the shader answered 0.0015 where the true value is 1.97e-5,
// because the series peaks near 610 on its way to an answer of 2e-5 — seven digits of
// cancellation into a format that has seven.
//
// So this side uses the Abramowitz & Stegun 9.8 polynomials, which exist for exactly
// that reason: minimax fits in x^2/4 and 2/x with no cancelling sum in them, accurate
// to about 1e-7, splitting at 2 rather than at 10.
//
// **Two implementations of one function, and the golden is what holds them together.**
// The alternative was to make the core lose precision to match a shader, which is the
// wrong direction — the frozen answers come from torch, and both sides are held to
// those rather than to each other.
//
// (No backticks in this block. The shader text is a template literal and one closes
//  it — the note beside i1_ above says this repository has stepped on that three
//  times, and writing that note did not stop a fourth.)
fn k_scaled_(x: f32, order: u32) -> f32 {
  if (x <= 2.0) {
    let y = x * x * 0.25;
    if (order == 0u) {
      let poly = -0.57721566 + y * (0.42278420 + y * (0.23069756 + y * (0.03488590
        + y * (0.00262698 + y * (0.00010750 + y * 0.0000074)))));
      return (-log(x * 0.5) * i0_(x) + poly) * exp(x);
    }
    let poly = 1.0 + y * (0.15443144 + y * (-0.67278579 + y * (-0.18156897
      + y * (-0.01919402 + y * (-0.00110404 + y * -0.00004686)))));
    return (log(x * 0.5) * i1_(x) + poly / x) * exp(x);
  }
  let y = 2.0 / x;
  if (order == 0u) {
    let poly = 1.25331414 + y * (-0.07832358 + y * (0.02189568 + y * (-0.01062446
      + y * (0.00587872 + y * (-0.00251540 + y * 0.00053208)))));
    return poly / sqrt(x);
  }
  let poly = 1.25331414 + y * (0.23498619 + y * (-0.03655620 + y * (0.01504268
    + y * (-0.00780353 + y * (0.00325614 + y * -0.00068245)))));
  return poly / sqrt(x);
}`;

/**
 * **`erfcx` and `log_ndtr` — the two whose whole reason is the tail.**
 *
 * `erfc(x)·exp(x²)` is `inf` from x=10 and `nan` by x=26, against the true 0.056141 and
 * 0.0216836; `log(ndtr(x))` is `-inf` from x=-6, against -20.7368. Neither product is
 * formed here: `erfcx` is a continued fraction above |x| = 4 with no exponential in it,
 * and `log_ndtr` is a sum of logarithms rather than the logarithm of a product that has
 * already underflowed.
 *
 * **The split is on `|x|`, not on `x`.** The core's first draft split on `x >= 4` and
 * every negative argument took the small branch *and* the reflection on top of an answer
 * that was already right — `erfcx(-2)` came back 0.255 against 108.941.
 */
const ERFCX_PRELUDE = `
fn erfcx_(x: f32) -> f32 {
  if (abs(x) < 4.0) {
    // Nothing overflows here either way: exp(16) is 8.9e6 and erfc(-4) is nearly 2.
    return erfc_(x) * exp(x * x);
  }
  let a = abs(x);
  var frac = 0.0;
  for (var k = 60u; k >= 1u; k = k - 1u) { frac = (f32(k) * 0.5) / (a + frac); }
  let got = 1.0 / (1.7724538509055159 * (a + frac));
  // erfcx(-t) = 2·exp(t²) − erfcx(t), which overflows honestly for large t.
  return select(got, 2.0 * exp(a * a) - got, x < 0.0);
}
fn log_ndtr_(x: f32) -> f32 {
  if (x >= -1.0) { return log(erfc_(-x * 0.7071067811865476) * 0.5); }
  // ndtr(x) = erfcx(-x/√2)·exp(-x²/2)/2, so the logarithm is a sum rather than the
  // logarithm of a product that is already zero.
  let t = -x * 0.7071067811865476;
  return log(erfcx_(t) * 0.5) - x * x * 0.5;
}`;

/**
 * **The first- and second-kind Bessel functions, as minimax tables.**
 *
 * Not series: the ascending series for `J₀` alternates and cancels catastrophically past
 * x ≈ 8 — the terms reach 10⁴ before they turn over and the answer is under 1. So the
 * large-argument side is the asymptotic form and the two meet at 8 without a step.
 *
 * The coefficients are the standard ones (Abramowitz & Stegun 9.4), the same digits the
 * core carries, and **the transcription is what the golden checks**: a mistyped minimax
 * coefficient does not raise, it moves the answer in the fifth place.
 */
const BESSEL_JY_PRELUDE = `
fn bessel_j0_(x: f32) -> f32 {
  let a = abs(x);
  if (a < 8.0) {
    let y = a * a;
    let p = 57568490574.0 + y * (-13362590354.0 + y * (651619640.7
      + y * (-11214424.18 + y * (77392.33017 + y * -184.9052456))));
    let q = 57568490411.0 + y * (1029532985.0 + y * (9494680.718
      + y * (59272.64853 + y * (267.8532712 + y))));
    return p / q;
  }
  let z = 8.0 / a;
  let y = z * z;
  let xx = a - 0.785398164;
  let p1 = 1.0 + y * (-0.1098628627e-2 + y * (0.2734510407e-4
    + y * (-0.2073370639e-5 + y * 0.2093887211e-6)));
  let q1 = -0.1562499995e-1 + y * (0.1430488765e-3 + y * (-0.6911147651e-5
    + y * (0.7621095161e-6 + y * -0.934935152e-7)));
  return sqrt(0.636619772 / a) * (cos(xx) * p1 - z * sin(xx) * q1);
}
fn bessel_j1_(x: f32) -> f32 {
  let a = abs(x);
  var out = 0.0;
  if (a < 8.0) {
    let y = a * a;
    let p = a * (72362614232.0 + y * (-7895059235.0 + y * (242396853.1
      + y * (-2972611.439 + y * (15704.48260 + y * -30.16036606)))));
    let q = 144725228442.0 + y * (2300535178.0 + y * (18583304.74
      + y * (99447.43394 + y * (376.9991397 + y))));
    out = p / q;
  } else {
    let z = 8.0 / a;
    let y = z * z;
    let xx = a - 2.356194491;
    let p1 = 1.0 + y * (0.183105e-2 + y * (-0.3516396496e-4
      + y * (0.2457520174e-5 + y * -0.240337019e-6)));
    let q1 = 0.04687499995 + y * (-0.2002690873e-3 + y * (0.8449199096e-5
      + y * (-0.88228987e-6 + y * 0.105787412e-6)));
    out = sqrt(0.636619772 / a) * (cos(xx) * p1 - z * sin(xx) * q1);
  }
  // **J1 is odd where J0 is even**, so the sign of the argument comes back.
  return select(out, -out, x < 0.0);
}
fn bessel_y0_(x: f32) -> f32 {
  // **NaN and -inf are made from the argument, not written down.** WGSL refuses a
  // constant it cannot represent — bitcast<f32>(0x7fc00000u) is rejected at parse time
  // with "value nan cannot be represented as f32" — and a shader that fails to compile
  // does not raise: the dispatch quietly does nothing and whatever was in the buffer
  // comes back. Measured that way: bessel_y0(0.001) answered 99.97, which is k1's value
  // from the dispatch before it.
  let zero = x - x;
  if (x < 0.0) { return zero / zero; }
  if (x == 0.0) { return -1.0 / zero; }
  if (x < 8.0) {
    let y = x * x;
    let p = -2957821389.0 + y * (7062834065.0 + y * (-512359803.6
      + y * (10879881.29 + y * (-86327.92757 + y * 228.4622733))));
    let q = 40076544269.0 + y * (745249964.8 + y * (7189466.438
      + y * (47447.26470 + y * (226.1030244 + y))));
    return p / q + 0.636619772 * bessel_j0_(x) * log(x);
  }
  let z = 8.0 / x;
  let y = z * z;
  let xx = x - 0.785398164;
  let p1 = 1.0 + y * (-0.1098628627e-2 + y * (0.2734510407e-4
    + y * (-0.2073370639e-5 + y * 0.2093887211e-6)));
  let q1 = -0.1562499995e-1 + y * (0.1430488765e-3 + y * (-0.6911147651e-5
    + y * (0.7621095161e-6 + y * -0.934935152e-7)));
  return sqrt(0.636619772 / x) * (sin(xx) * p1 + z * cos(xx) * q1);
}
fn bessel_y1_(x: f32) -> f32 {
  // **NaN and -inf are made from the argument, not written down.** WGSL refuses a
  // constant it cannot represent — bitcast<f32>(0x7fc00000u) is rejected at parse time
  // with "value nan cannot be represented as f32" — and a shader that fails to compile
  // does not raise: the dispatch quietly does nothing and whatever was in the buffer
  // comes back. Measured that way: bessel_y0(0.001) answered 99.97, which is k1's value
  // from the dispatch before it.
  let zero = x - x;
  if (x < 0.0) { return zero / zero; }
  if (x == 0.0) { return -1.0 / zero; }
  if (x < 8.0) {
    let y = x * x;
    let p = x * (-4900604943000.0 + y * (1275274390000.0 + y * (-51534381390.0
      + y * (734926455.1 + y * (-4237922.726 + y * 8511.937935)))));
    let q = 24995805700000.0 + y * (424441966400.0 + y * (3733650367.0
      + y * (22459040.02 + y * (102042.605 + y * (354.9632885 + y)))));
    return p / q + 0.636619772 * (bessel_j1_(x) * log(x) - 1.0 / x);
  }
  let z = 8.0 / x;
  let y = z * z;
  let xx = x - 2.356194491;
  let p1 = 1.0 + y * (0.183105e-2 + y * (-0.3516396496e-4
    + y * (0.2457520174e-5 + y * -0.240337019e-6)));
  let q1 = 0.04687499995 + y * (-0.2002690873e-3 + y * (0.8449199096e-5
    + y * (-0.88228987e-6 + y * 0.105787412e-6)));
  return sqrt(0.636619772 / x) * (sin(xx) * p1 + z * cos(xx) * q1);
}`;

/**
 * **Airy's `Ai`** — the solution of `y″ = xy` that decays to the right.
 *
 * Two regimes, and the seam is where cancellation starts costing digits: the ascending
 * series converges for every argument, and past |x| ≈ 8 its largest term is a hundred
 * times the answer. Beyond that it is the standard asymptotic — exponential decay for
 * positive `x`, an oscillation with an `x^(−1/4)` envelope for negative.
 *
 * `Ai(0)` is `3^(−2/3)/Γ(2/3)` = 0.3550280539 and `−Ai′(0)` is `3^(−1/3)/Γ(1/3)`; both
 * were matched against torch to ten digits before the series was written, because a
 * normalisation constant wrong in the fourth place produces a curve of exactly the right
 * shape.
 */
const AIRY_PRELUDE = `
fn airy_ai_(x: f32) -> f32 {
  // **The seam is 6 here and 8 in the core, and the difference is f32.** The ascending
  // series converges everywhere and cancels as it goes: at x = -7.9 its largest term is
  // about 123 on the way to an answer of 0.04, which is three and a half digits gone
  // into a format with seven. Measured before this moved: 0.041377 against 0.041701,
  // outside the golden by three times. At 6 the largest term is 4.5 against an answer
  // near 0.33 and nothing is lost.
  if (abs(x) <= 6.0) {
    let cube = x * x * x;
    var f = 1.0;
    var term = 1.0;
    for (var k = 1u; k < 30u; k = k + 1u) {
      term = term * cube / ((3.0 * f32(k)) * (3.0 * f32(k) - 1.0));
      f = f + term;
    }
    var g = x;
    var t2 = x;
    for (var k = 1u; k < 30u; k = k + 1u) {
      t2 = t2 * cube / ((3.0 * f32(k) + 1.0) * (3.0 * f32(k)));
      g = g + t2;
    }
    return 0.3550280538878172 * f - 0.2588194037928068 * g;
  }
  let a = abs(x);
  let zeta = 0.6666666666666666 * a * sqrt(a);
  // u_k = u_{k-1}·(6k−5)(6k−3)(6k−1)/(216k(2k−1)); u_1 is 5/72 either way, which is
  // what pinned the form.
  var u = array<f32, 12>(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
  for (var k = 1u; k < 12u; k = k + 1u) {
    let n = f32(k);
    u[k] = u[k - 1u] * (6.0 * n - 5.0) * (6.0 * n - 3.0) * (6.0 * n - 1.0)
         / (216.0 * n * (2.0 * n - 1.0));
  }
  // **Eight terms, not twelve.** The expansion is asymptotic rather than convergent —
  // past its optimal truncation the terms turn and grow — and the seam moving from 8 to
  // 6 lowered the smallest zeta it is asked at from 15.1 to 9.8, which is where twelve
  // terms stops being past it and starts being over it.
  if (x > 0.0) {
    var series = 0.0;
    var power = 1.0;
    for (var k = 0u; k < 8u; k = k + 1u) {
      series = series + select(-u[k], u[k], (k & 1u) == 0u) * power;
      power = power / zeta;
    }
    return exp(-zeta) / (2.0 * 1.7724538509055159 * pow(a, 0.25)) * series;
  }
  var even = 0.0;
  var odd = 0.0;
  var power = 1.0;
  for (var k = 0u; k < 4u; k = k + 1u) {
    let s = select(-1.0, 1.0, (k & 1u) == 0u);
    even = even + s * u[2u * k] * power;
    odd = odd + s * u[2u * k + 1u] * power / zeta;
    power = power / (zeta * zeta);
  }
  let phase = zeta + 0.7853981633974483;
  return (sin(phase) * even - cos(phase) * odd)
       / (1.7724538509055159 * pow(a, 0.25));
}`;

/**
 * Anything with no derivative (a step, such as `sign` or `floor`) is `bwd: "0.0"`.
 *
 * **The graph is not severed.** torch flows a 0, and a refusal is not a 0 — the sister
 * library got this wrong once, and a loss with a step in it ran on torch and stopped on
 * ours.
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
  // **Nothing flows at 0.** `step(0.0, x)` is `x >= 0`, so it gives 1 at exactly 0
  // where torch gives 0. The golden's relu case had no 0 in its input and could not see
  // this; it surfaced while matching ResNet against real torch.
  relu: { fwd: "max(x, 0.0)", bwd: "select(0.0, 1.0, x > 0.0)" },
  sigmoid: { fwd: "1.0 / (1.0 + exp(-x))", bwd: "o * (1.0 - o)" },
  // A step — it flows 0. torch does the same.
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
  // sinc(0) is 1 and the derivative there is 0. It is a division by zero, so it is
  // split out — WGSL gives NaN for 0/0, and NaN differs even from itself, so no
  // comparison can pass.
  sinc: {
    fwd: "select(sin(3.141592653589793 * x) / (3.141592653589793 * x), 1.0, x == 0.0)",
    bwd:
      "select((cos(3.141592653589793 * x) - o) / x, 0.0, x == 0.0)",
  },
  erf: { fwd: "erf_(x)", bwd: "1.1283791670955126 * exp(-x * x)", prelude: ERF_PRELUDE },
  erfc: { fwd: "erfc_(x)", bwd: "-1.1283791670955126 * exp(-x * x)", prelude: ERF_PRELUDE },
  // On reals sgn is the same as sign. It is an alias, and torch has both, so the name
  // stays.
  sgn: { fwd: "sign(x)", bwd: "0.0" },
  // True and false come out as 0/1. The dtype is float32 alone, so no separate bool is
  // held.
  // **-0.0 is false here** — torch reads it as true. No current case contains -0.0 so
  // nothing diverges, and on the day something does, this line is the cause.
  signbit: { fwd: "select(0.0, 1.0, x < 0.0)", bwd: "0.0" },
  // Neither is a public torch name; they are the pieces `nansum` and `nanmean` are
  // assembled from.
  nanToZero: {
    fwd: "select(x, 0.0, is_nan(x))",
    // Nothing flows to a NaN position. It never entered the sum, so 0 is right.
    bwd: "select(1.0, 0.0, is_nan(x))",
    prelude: NAN_PRELUDE,
  },
  notNan: { fwd: "select(1.0, 0.0, is_nan(x))", bwd: "0.0", prelude: NAN_PRELUDE },
  isnan: { fwd: "select(0.0, 1.0, is_nan(x))", bwd: "0.0", prelude: NAN_PRELUDE },

  // ── `torch.special`'s fifteen that need a kernel ────────────────────────
  //
  // The other nineteen of that namespace forward or compose; these do not, and every
  // one of them exists **because** the obvious composition of what is already in this
  // table is `inf` or `nan` exactly where the name is reached for. Built as
  // arrangements they would agree with the core at every ordinary input and part in
  // the tail — which is the shape a value comparison against torch sees and a reader
  // does not.
  //
  // `bwd: "0.0"` throughout. torch differentiates several of these and this does not:
  // the derivations are a second body each, and a wrong one is wrong quietly. The
  // graph is not severed — a zero flows, which is what the note above this table is
  // about — and `backward()` through them gives 0 rather than refusing, which is the
  // one thing here that is worse than torch rather than absent.
  erfcx: { fwd: "erfcx_(x)", bwd: "0.0", prelude: ERF_PRELUDE + ERFCX_PRELUDE },
  logNdtr: { fwd: "log_ndtr_(x)", bwd: "0.0", prelude: ERF_PRELUDE + ERFCX_PRELUDE },

  // `i1` is `i0`'s derivative and has been in this file since `i0` arrived — it had no
  // public name until torch's `special.i1` needed one.
  i1: { fwd: "i1_(x)", bwd: "0.0", prelude: I0_PRELUDE },
  i0e: { fwd: "i_scaled_(x, 0u)", bwd: "0.0",
         prelude: I0_PRELUDE + BESSEL_SCALED_PRELUDE },
  // **Odd, where `i0e` is even.** The scaled series is computed on |x| and the sign is
  // put back — dropped, the negative half is silently the positive one.
  i1e: { fwd: "sign(x) * i_scaled_(abs(x), 1u)", bwd: "0.0",
         prelude: I0_PRELUDE + BESSEL_SCALED_PRELUDE },
  modifiedBesselK0: { fwd: "k_scaled_(x, 0u) * exp(-x)", bwd: "0.0",
                      prelude: I0_PRELUDE + BESSEL_SCALED_PRELUDE },
  modifiedBesselK1: { fwd: "k_scaled_(x, 1u) * exp(-x)", bwd: "0.0",
                      prelude: I0_PRELUDE + BESSEL_SCALED_PRELUDE },
  scaledModifiedBesselK0: { fwd: "k_scaled_(x, 0u)", bwd: "0.0",
                            prelude: I0_PRELUDE + BESSEL_SCALED_PRELUDE },
  scaledModifiedBesselK1: { fwd: "k_scaled_(x, 1u)", bwd: "0.0",
                            prelude: I0_PRELUDE + BESSEL_SCALED_PRELUDE },

  besselJ0: { fwd: "bessel_j0_(x)", bwd: "0.0", prelude: BESSEL_JY_PRELUDE },
  besselJ1: { fwd: "bessel_j1_(x)", bwd: "0.0", prelude: BESSEL_JY_PRELUDE },
  besselY0: { fwd: "bessel_y0_(x)", bwd: "0.0", prelude: BESSEL_JY_PRELUDE },
  besselY1: { fwd: "bessel_y1_(x)", bwd: "0.0", prelude: BESSEL_JY_PRELUDE },
  airyAi: { fwd: "airy_ai_(x)", bwd: "0.0", prelude: AIRY_PRELUDE },
  // The infinity test is bitwise too — an exponent of all ones with a zero mantissa.
  isinf: {
    fwd: "select(0.0, 1.0, (bitcast<u32>(x) & 0x7fffffffu) == 0x7f800000u)",
    bwd: "0.0",
  },
  isfinite: {
    fwd: "select(0.0, 1.0, (bitcast<u32>(x) & 0x7f800000u) != 0x7f800000u)",
    bwd: "0.0",
  },
  logical_not: { fwd: "select(0.0, 1.0, x == 0.0)", bwd: "0.0" },
  // The values are integers. The same arrangement as the binary bitwise operations —
  // stored as `f32` and computed as `i32`.
  // **On a boolean it is logical negation** (`~true` is `false`, not `-2`). That branch
  // is taken by the Python binding, which knows the dtype; this only looks at
  // integers.
  bitwise_not: { fwd: "f32(~i32(x))", bwd: "0.0" },
  /**
   * The zeroth-order modified Bessel function. `kaiser_window` stands on it. Its
   * derivative is `i1`.
   *
   * **It was `bwd: "0.0"` for a long time, and the justification was "the core severs
   * the graph too."** That premise disappeared when the core was fixed, and the
   * justification was wrong from the start — one side copying the other's hole and
   * writing that down as the reason means **the golden never asks that question again.**
   * It is the risk inherent in three implementations comparing against each other, and
   * it really was green.
   *
   * And this place was worse than the core's. **A severed graph makes a noise**
   * (`backward` stops with "does not require grad"). `0.0` makes none — a gradient whose
   * value is 0 and "there is no gradient here" say different things, and writing the
   * second as the first makes training quietly not happen.
   */
  i0: { fwd: "i0_(x)", bwd: "i1_(x)", prelude: I0_PRELUDE },
  /**
   * `frexp`'s two faces. The WGSL builtin returns a struct, so it is baked once per
   * slot.
   *
   * torch returns the exponent as int32 while the storage here is f32 alone, so only the
   * value is matched.
   */
  frexpMantissa: { fwd: "frexp(x).fract", bwd: "0.0" },
  frexpExponent: { fwd: "f32(frexp(x).exp)", bwd: "0.0" },
  /**
   * torch's default `gelu` — the **exact** form rather than the approximation.
   *
   * `0.5·x·(1 + erf(x/√2))`. In the left tail `1` and `erf` cancel, and in f32
   * `erf(-8)` is exactly `-1`, so the result is 0. torch gives -4.9e-15, a difference of
   * 5e-15, far below this project's tolerance (1e-4). Doing better would mean deriving
   * through erfc as the core does, and there is no reason to right now.
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
    // s is used twice, so it is pulled into a helper — inline it calls exp four
    // times.
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
  // ── Activations with no arguments. The ones taking arguments are baked as
  //    constants by `tensor.ts`. ─────────────────────────────────────────────
  //
  // **Which side of the kink you are on is the whole thing.** The expressions are in the
  // documentation, and what torch gives at exact boundaries like `x == ±3` or `x == 6`
  // has to be measured, and a random input never lands on those points. The golden holds
  // them by hand.
  hardsigmoid: {
    fwd: "clamp(x / 6.0 + 0.5, 0.0, 1.0)",
    bwd: "select(0.0, 0.16666666666666666, x > -3.0 && x < 3.0)",
  },
  hardswish: {
    fwd: "select(select(x * (x + 3.0) / 6.0, x, x >= 3.0), 0.0, x <= -3.0)",
    bwd: "select(select((2.0 * x + 3.0) / 6.0, 1.0, x >= 3.0), 0.0, x <= -3.0)",
  },
  // log σ(x). **Computed directly it becomes log(0) at large negatives** — written in
  // the stable form.
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
  // **The gradient is 0 at both boundaries.** Differentiating `clamp` naively misses
  // those points.
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
  // ── The ones computed by series. **There is no closed form.** ──────────────
  //
  // The coefficients come from well-known tables verbatim — trimming digits makes the
  // answer wrong by that much. **The same expressions** as the core (numpy). Written
  // differently in the two places, the golden cannot say which is right.
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
  // **A negative base has no answer.** WGSL's pow is `exp2(y·log2(x))` and log2 is
  // undefined for negatives — in practice a value comes out as though `|x|` had been
  // used, so an even exponent's forward is accidentally right and only the backward's
  // sign is flipped. Integer exponents go through `Tensor.powScalar` as multiplications
  // and never pass here.
  pow: { fwd: "pow(x, y)", da: "y * pow(x, y - 1.0)", db: "o * log(x)" },
  // **A tie splits in half.** torch does the same — `maximum(2, 2)`'s gradient is 0.5
  // on each side, not 1. `step(y, x)` is `x >= y`, so it gave 1 to both sides on a tie,
  // and then the sum is twice torch's. The forward is equally right either way so a
  // value comparison does not catch it, and `edge::grad::maximum(동점)` asks about it.
  //
  // `clamp` and `leakyRelu` used to sit on this, and **torch does not split those two**
  // — they flow the whole gradient at the boundary. So they were given their own kernels
  // (`clampScalar` and `leakyRelu`).
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
  // It carries the sign across. Nothing flows to y — a sign is a step.
  copysign: {
    fwd: "select(-abs(x), abs(x), y >= 0.0)",
    da: "select(-sign(x), sign(x), y >= 0.0)",
    db: "0.0",
  },
  // **Written in the stable form.** Taking log(exp x + exp y) literally makes
  // float32's exp overflow to inf the moment x passes 89, and every result after that is
  // inf. Pulling the larger one out keeps it from overflowing.
  logaddexp: {
    // WGSL has no log1p builtin — written as log(1+t).
    fwd: "max(x, y) + log(1.0 + exp(-abs(x - y)))",
    da: "1.0 / (1.0 + exp(y - x))",
    db: "1.0 / (1.0 + exp(x - y))",
  },
  logaddexp2: {
    fwd: "max(x, y) + log2(1.0 + exp2(-abs(x - y)))",
    da: "1.0 / (1.0 + exp2(y - x))",
    db: "1.0 / (1.0 + exp2(x - y))",
  },
  // **x at 0 gives 0 whatever y is.** Even when y is 0 — that is what the function is
  // for, and without looking at that point it is indistinguishable from `x * log(y)`.
  xlogy: {
    fwd: "select(x * log(y), 0.0, x == 0.0)",
    da: "log(y)",
    db: "x / y",
  },
  // 0 for x<0, 1 for x>0, and y itself at x==0. It is a step, so nothing flows to x.
  heaviside: {
    fwd: "select(select(0.0, 1.0, x > 0.0), y, x == 0.0)",
    da: "0.0",
    db: "select(0.0, 1.0, x == 0.0)",
  },
  ldexp: { fwd: "x * exp2(y)", da: "exp2(y)", db: "o * 0.6931471805599453" },
  // A comparison gives 0/1. The dtype is float32 alone, so no separate bool is held —
  // it matches the golden's bool cases as 0/1. **The gradient is 0 on both sides.**
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
  // ── Bitwise ─────────────────────────────────────────────────────────────
  //
  // The storage is a single f32 and the values are integers. They are converted to `i32`
  // for the arithmetic and back — f32 holds every integer exactly up to 2^24, so the
  // answers are right in that range.
  //
  // **The right shift is arithmetic.** WGSL's `i32 >>` extends the sign, so `-3 >> 5` is
  // `-1` — as in torch (measured). Converting to `u32` for a logical shift gives a
  // wholly different answer on negatives, and asking only with positives hides that.
  //
  // On booleans torch branches to the logical operation. That branch is taken by the
  // side that knows the dtype (the Python binding); here it is integer arithmetic
  // only.
  bitwise_and: { fwd: "f32(i32(x) & i32(y))", da: "0.0", db: "0.0" },
  bitwise_or: { fwd: "f32(i32(x) | i32(y))", da: "0.0", db: "0.0" },
  bitwise_xor: { fwd: "f32(i32(x) ^ i32(y))", da: "0.0", db: "0.0" },
  bitwise_left_shift: { fwd: "f32(i32(x) << u32(i32(y)))", da: "0.0", db: "0.0" },
  bitwise_right_shift: { fwd: "f32(i32(x) >> u32(i32(y)))", da: "0.0", db: "0.0" },
  // Euclid's algorithm. The sign is discarded — torch's gcd is always non-negative.
  gcd: {
    fwd: "gcd_(x, y)", da: "0.0", db: "0.0",
    prelude: `
fn gcd_(xa: f32, yb: f32) -> f32 {
  var a = abs(i32(xa));
  var b = abs(i32(yb));
  while (b != 0) { let t = a % b; a = b; b = t; }
  return f32(a);
}`,
  },
  // `|a·b| / gcd`. **A gcd of 0 gives 0** — as in torch (the lcm of 0 and 7 is 0).
  lcm: {
    fwd: "lcm_(x, y)", da: "0.0", db: "0.0",
    prelude: `
fn lcm_(xa: f32, yb: f32) -> f32 {
  var a = abs(i32(xa));
  var b = abs(i32(yb));
  let p = a * b;
  while (b != 0) { let t = a % b; a = b; b = t; }
  if (a == 0) { return 0.0; }
  return f32(p / a);
}`,
  },
  /**
   * **The next representable number** from `a` towards `b`. It moves by one ulp.
   *
   * Read as signed integers, f32 lays out in order of magnitude — a larger bit pattern
   * is a larger positive, and for negatives a larger bit pattern is **further from 0.**
   * So deciding the direction cannot be `b > a` alone; it is entangled with `a`'s sign.
   * Asked only with positives, that entanglement is invisible.
   *
   * `a == 0` is kept apart — adding 1 to the bits of 0 attaches magnitude rather than
   * sign, which misses the negative direction.
   */
  nextafter: {
    fwd: "nextafter_(x, y)", da: "1.0", db: "0.0",
    prelude: `
fn nextafter_(a: f32, b: f32) -> f32 {
  if (a == b) { return b; }
  if (a == 0.0) {
    return select(bitcast<f32>(0x80000001u), bitcast<f32>(1u), b > 0.0);
  }
  var i = bitcast<i32>(a);
  if ((b > a) == (a > 0.0)) { i = i + 1; } else { i = i - 1; }
  return bitcast<f32>(i);
}`,
  },
};

/** The workgroup size. Elementwise and reductions are 1-D. */
export const WORKGROUP = 64;

/**
 * The per-axis limit on `dispatchWorkgroups`. **Exceed it and it does not throw; it
 * quietly does not run.**
 *
 * The conv bench asked for 589,824 and WebGPU silently ran only some of them. It looked
 * like "144% of TF.js" and five of six values were wrong — a number that would have been
 * believed if the values had not been looked at alongside. Which is why the grid
 * arithmetic lives inside the kernel generator.
 */
export const MAX_DISPATCH = 65535;

/** Spreads 1-D work into a 2-D grid inside the limit. */
export interface Grid {
  /** What goes into `dispatchWorkgroups`. */
  readonly x: number;
  readonly y: number;
  /** The number of **threads** laid along one row — the shader writes
   *  `g.y * GX + g.x`. */
  readonly threadsX: number;
}

export function grid1d(n: number, workgroup: number = WORKGROUP): Grid {
  const groups = Math.max(1, Math.ceil(n / workgroup));
  const x = Math.min(groups, MAX_DISPATCH);
  const y = Math.ceil(groups / MAX_DISPATCH);
  return { x, y, threadsX: x * workgroup };
}

/** Produces the flat index from the 2-D grid. The same expression is used even below
 *  the limit — there has to be one path. */
function flatId(n: number): string {
  const { threadsX } = grid1d(n);
  return `  let gid = g.y * ${threadsX}u + g.x;\n  if (gid >= ${n}u) { return; }`;
}

export type UnaryName = keyof typeof UNARY & string;
export type BinaryName = keyof typeof BINARY & string;

/**
 * A unary operation carrying a constant.
 *
 * Things like `clamp(-1, 1)` or `leakyRelu(0.1)`, where the argument is mixed into the
 * expression. Putting the argument in a uniform would mean one shader, and the reason
 * this file chooses the opposite applies here too — baked as a constant, it folds. The
 * constant goes into the name, so the pipeline cache splits by itself and two calls with
 * the same argument use the same shader.
 */
const DERIVED: Record<string, UnarySpec> = {};

/**
 * WGSL's f32 literal. A value that looks like an integer still needs a decimal point or
 * the type diverges.
 *
 * **Infinity and NaN cannot be baked into a shader.** WGSL forbids a
 * compile-time-evaluated value from becoming inf or NaN — a literal and the detour
 * through `bitcast<f32>(0x7f800000u)` are refused identically (both measured). Before
 * that, `String(Infinity)` planted the **characters** `Infinity` into the shader and it
 * stopped with `unresolved value 'Infinity'` — filling one value killed the whole
 * pipeline, and the place was `Tensor.full(shape, Infinity)`.
 *
 * That side went the route of filling on the CPU and uploading. Here it **refuses
 * loudly** — quietly approximating what cannot be baked leaves the shader running with a
 * different answer.
 */
export function f32lit(v: number): string {
  if (!Number.isFinite(v)) {
    throw new Error(
      `WGSL cannot write the constant ${v} as f32 — infinity and NaN are rejected at ` +
      "compile time.\n  Fill it on the CPU and upload it (Tensor.full takes that path).",
    );
  }
  // **No decimal point is attached to exponent notation.** `String(-1e30)` is
  // `-1e+30`, `Number.isInteger` is true for it, so `.0` was appended — and WGSL refuses
  // `-1e+30.0` at parse time. Filling one value killed the whole pipeline. Exponent
  // notation is already a floating point literal by itself.
  const text = String(v);
  if (text.includes("e") || text.includes("E") || text.includes(".")) return text;
  return `${text}.0`;
}

/** Registers a unary with its constant baked in and hands back the name. Already
 *  present, it is not built again. */
export function unaryWith(key: string, make: () => UnarySpec): string {
  DERIVED[key] ??= make();
  return key;
}

/** Whether a unary kernel can be built for this name. It looks at the table's and the
 *  baked ones together. */
export function hasUnary(name: string): boolean {
  return Boolean(UNARY[name] ?? DERIVED[name]);
}

function unarySpec(name: string): UnarySpec {
  const op = UNARY[name] ?? DERIVED[name];
  if (!op) throw new Error(`unknown unary op: ${name}`);
  return op;
}

/** An elementwise unary forward. The element count is baked in as a constant — the
 *  bounds check folds away. */
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

/** An elementwise unary backward. It takes the forward result rather than recomputing
 *  it. */
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
 * From a flat output index, produces both inputs' indices.
 *
 * **Not one division survives** — the axes are walked from the back subtracting
 * remainders, and every divisor is a literal, so the compiler folds them into
 * multiplications and shifts. In a uniform they do not fold, and that difference was 4×
 * on conv.
 *
 * An axis of size 1 has stride 0 and keeps reading the same value — **it is not expanded
 * and copied.** The copying version costs memory, and that is why im2col lost at conv.
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

/** An elementwise binary forward. Broadcasting is handled through the strides. */
export function binaryForward(
  name: string,
  shape: readonly number[],
  strideA: readonly number[],
  strideB: readonly number[],
): string {
  const op = BINARY[name];
  if (!op) throw new Error(`unknown binary op: ${name}`);
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
 * One side's share of an elementwise binary backward. **It comes out in the output's
 * shape.**
 *
 * Where there is broadcasting, the contribution is produced expanded here and the
 * folding is done separately by `reduceBroadcast`. The reason for two stages is **to
 * avoid atomic addition** — adding straight into the input positions here would make the
 * order differ every time, and floating point changes value with order, so the same seed
 * would train two different ways.
 *
 * It uses **the same indexing expression** as the forward. Two paths means fixing only
 * one of them.
 */
export function binaryBackward(
  name: string,
  which: "a" | "b",
  shape: readonly number[],
  strideA: readonly number[],
  strideB: readonly number[],
): string {
  const op = BINARY[name];
  if (!op) throw new Error(`unknown binary op: ${name}`);
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
 * Folds a broadcast axis back down.
 *
 * For `a(3,1) + b(3,4)`, the gradient to `a` is the (3,4) summed along axis 1 into
 * (3,1). Without it the shape does not match, and matching the shape while omitting the
 * sum makes **the values plausibly wrong** — the kind that passes the golden and only
 * fails to train.
 *
 * One thread takes one output cell and walks its share **in a fixed order.** No atomic
 * addition, so running it twice gives the same value.
 *
 * @param full the gradient's shape
 * @param small the shape to fold down to. Same rank, and each axis is
 *   either 1 or equal to `full`.
 */
/** How many elements of `full` fold into one element of `small`. */
export function broadcastFold(full: readonly number[], small: readonly number[]): number {
  let fold = 1;
  for (let d = 0; d < full.length; d++) {
    if ((small[d] ?? 1) !== (full[d] ?? 1)) fold *= full[d] ?? 1;
  }
  return fold;
}

/** The threads one output element's fold is spread over in `reduceBroadcastWide`. */
export const FOLD_GROUP = 256;

/** Above this many elements per output, the fold takes a workgroup (see below). */
export const FOLD_WIDE = 64;

/**
 * Folds a broadcast gradient back to its shape — **a workgroup per output element**,
 * for folds that are long.
 *
 * `reduceBroadcast` below gives one thread one output element and has it walk every
 * element that broadcast from it. For a bias or a scalar that is the whole activation:
 * the U-Net step spent 2.5 ms — eleven per cent of its GPU time — in single threads
 * summing 147,456 floats each (measured with timestamps, 2026-09-06), the same disease
 * BatchNorm's statistics had. Here 256 threads stride the fold and meet in a tree; the
 * order is fixed, so two runs still give the same value. Short folds keep the thread
 * version: a workgroup for sixteen elements is the waste the other way.
 */
/** How many workgroups share one output element's fold in `reduceBroadcastWide`: pieces of
 *  at most 16,384, at most 64 — sixteen channels folding 147,456 each were sixteen
 *  workgroups on the whole GPU, 1.3 ms for a bias gradient (timestamps, 2026-09-07). */
export function foldPieces(full: readonly number[], small: readonly number[]): number {
  return Math.max(1, Math.min(64, Math.ceil(broadcastFold(full, small) / 16384)));
}

export function reduceBroadcastWide(full: readonly number[], small: readonly number[]): string {
  const rank = full.length;
  const fullStride: number[] = new Array<number>(rank).fill(1);
  for (let d = rank - 2; d >= 0; d--) fullStride[d] = (fullStride[d + 1] ?? 1) * (full[d + 1] ?? 1);
  // The output element's coordinates, and the part of the offset they fix.
  const decompose: string[] = [];
  const baseTerms: string[] = [];
  const axes: { d: number; size: number }[] = [];
  for (let d = rank - 1; d >= 0; d--) {
    decompose.push(`  let i${d} = rest % ${small[d] ?? 1}u; rest = rest / ${small[d] ?? 1}u;`);
  }
  for (let d = 0; d < rank; d++) {
    const sd = small[d] ?? 1;
    const fd = full[d] ?? 1;
    if (sd === fd && fd !== 1) baseTerms.push(`i${d} * ${fullStride[d]}u`);
    else if (fd !== 1) axes.push({ d, size: fd });
  }
  const fold = axes.reduce((a, x) => a * x.size, 1);
  // A fold index `f` in [0, fold) decomposes over the broadcast axes, last axis fastest.
  const foldTerms: string[] = [];
  let unit = 1;
  for (let a = axes.length - 1; a >= 0; a--) {
    const { d, size } = axes[a] as { d: number; size: number };
    foldTerms.push(`((f / ${unit}u) % ${size}u) * ${fullStride[d]}u`);
    unit *= size;
  }
  const pieces = foldPieces(full, small);
  const per = Math.ceil(fold / pieces);
  const n = small.reduce((a, b) => a * b, 1);
  // Cut into pieces, workgroup (element, piece) folds its slice and writes a partial at
  // Out[piece · n + element]; `sumSplits` adds the pieces. Unsplit, Out is the answer.
  return `
@group(0) @binding(0) var<storage, read> G: array<f32>;
@group(0) @binding(1) var<storage, read_write> Out: array<f32>;
var<workgroup> part: array<f32, ${FOLD_GROUP}>;
@compute @workgroup_size(${FOLD_GROUP})
fn main(@builtin(local_invocation_id) l: vec3<u32>, @builtin(workgroup_id) w: vec3<u32>) {
  var rest = w.x;
${decompose.join("\n")}
  let base = ${baseTerms.length ? baseTerms.join(" + ") : "0u"};
  let lo = w.y * ${per}u;
  let hi = min(lo + ${per}u, ${fold}u);
  var acc = 0.0;
  for (var f = lo + l.x; f < hi; f = f + ${FOLD_GROUP}u) {
    acc = acc + G[base + ${foldTerms.join(" + ")}];
  }
  part[l.x] = acc;
  workgroupBarrier();
  var span = ${FOLD_GROUP / 2}u;
  loop {
    if (span == 0u) { break; }
    if (l.x < span) { part[l.x] = part[l.x] + part[l.x + span]; }
    workgroupBarrier();
    span = span / 2u;
  }
  if (l.x == 0u) { Out[w.y * ${n}u + w.x] = part[0]; }
}`;
}

export function reduceBroadcast(
  full: readonly number[],
  small: readonly number[],
): string {
  if (full.length !== small.length) {
    throw new Error(`rank mismatch: ${full.length} vs ${small.length}`);
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
    // With sd === fd === 1 there is no contribution — no term is emitted.
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
 * The tile a subgroup-matrix GEMM of `M` by `N` takes: the largest of 8/16/32 rows and
 * 8/16/32/64 columns that divide the matrix, so no tile crosses an edge — a subgroup
 * matrix loads and stores whole 8 × 8 blocks and has no per-element guard.
 */
export function subgroupMatmulTile(M: number, N: number): { TM: number; TN: number } {
  const TM = [32, 16, 8].find((t) => M % t === 0) ?? 8;
  const TN = [64, 32, 16, 8].find((t) => N % t === 0) ?? 8;
  return { TM, TN };
}

/** Whether `matmulSubgroup` can take a shape: every dimension a multiple of eight. */
export function subgroupMatmulFits(M: number, K: number, N: number): boolean {
  return M % 8 === 0 && K % 8 === 0 && N % 8 === 0;
}

/**
 * How many pieces the subgroup GEMM's reduction is split into — the same reasoning as
 * `convForwardSplit`: a small output over a long K is a handful of workgroups walking
 * 147,456 steps each. Measured, 16 × 147,456 × 144 took 8.4 ms in nine workgroups; the
 * split brings the grid to at least sixty-four. Each piece takes whole eights of K.
 */
export function subgroupMatmulSplit(M: number, K: number, N: number): number {
  const { TM, TN } = subgroupMatmulTile(M, N);
  const tiles = (M / TM) * (N / TN);
  const WANT = 64;
  if (tiles >= WANT) return 1;
  const MIN_PER_SPLIT = 256;
  return Math.max(1, Math.min(Math.ceil(WANT / tiles), Math.floor(K / MIN_PER_SPLIT)));
}

/**
 * The matrix product on subgroup matrices — **the hardware's 8 × 8 multiply.**
 *
 * One subgroup per workgroup owns a TM × TN block of the result as (TM/8)·(TN/8)
 * accumulators and walks K eight at a time, loading both operands straight from storage.
 * Nothing is staged and there is no barrier. Measured on the M4 Max: 2048³ at 11.0
 * TFLOP/s against 4.5 for the scalar tile below and 10.8 for torch on Metal; the
 * skinny 16 × 144 × 147,456 at 3.4 against 0.44.
 *
 * **Every offset is derived from the workgroup id alone.** The load and store builtins
 * require a uniform offset, and one built from `local_invocation_id` is refused at
 * compile time (measured) — which is why the workgroup is exactly one subgroup.
 */
export function matmulSubgroup(M: number, K: number, N: number): string {
  const { TM, TN } = subgroupMatmulTile(M, N);
  const splits = subgroupMatmulSplit(M, K, N);
  // Whole eights per piece; the last piece is clipped to K.
  const perSplit = Math.ceil(K / 8 / splits) * 8;
  const am = TM / 8;
  const bn = TN / 8;
  const acc: string[] = [];
  const mma: string[] = [];
  const store: string[] = [];
  for (let i = 0; i < am; i++) {
    for (let j = 0; j < bn; j++) {
      acc.push(`  var c${i}${j}: subgroup_matrix_result<f32, 8, 8>;`);
      mma.push(`    c${i}${j} = subgroupMatrixMultiplyAccumulate(a${i}, b${j}, c${i}${j});`);
      // Split, each piece lands in its own slab of the output buffer, summed afterwards.
      store.push(`  subgroupMatrixStore(&Out, ${splits > 1 ? `wid.z * ${M * N}u + ` : ""}(row0 + ${i * 8}u) * ${N}u + col0 + ${j * 8}u, c${i}${j}, false, ${N}u);`);
    }
  }
  const loadA = Array.from({ length: am }, (_, i) =>
    `    let a${i} = subgroupMatrixLoad<subgroup_matrix_left<f32, 8, 8>>(&A, (row0 + ${i * 8}u) * ${K}u + k, false, ${K}u);`);
  const loadB = Array.from({ length: bn }, (_, j) =>
    `    let b${j} = subgroupMatrixLoad<subgroup_matrix_right<f32, 8, 8>>(&B, k * ${N}u + col0 + ${j * 8}u, false, ${N}u);`);
  return `enable chromium_experimental_subgroup_matrix;
@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> Out: array<f32>;
@compute @workgroup_size(32)
fn main(@builtin(workgroup_id) wid: vec3<u32>) {
  let row0 = wid.y * ${TM}u;
  let col0 = wid.x * ${TN}u;
  let kFrom = wid.z * ${perSplit}u;
  let kTo = min(kFrom + ${perSplit}u, ${K}u);
${acc.join("\n")}
  for (var k = kFrom; k < kTo; k = k + 8u) {
${loadA.join("\n")}
${loadB.join("\n")}
${mma.join("\n")}
  }
${store.join("\n")}
}`;
}

/**
 * The matrix product. **The sixteen accumulators are spread into named scalars.**
 *
 * Held in an `array<f32,16>` and indexed by a variable as `acc[i*4+j]`, WGSL cannot keep
 * them in registers and spills to memory. Same algorithm, same tile size:
 * **182 vs 4,474 GFLOPS** (measured). It reads badly and it is 24×, and this kernel runs
 * at 115–217% of TF.js.
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
 * The full sum. It folds as a tree inside a workgroup and produces one partial sum per
 * workgroup.
 *
 * **No atomics.** Floating point addition changes value with order, and then two runs
 * from the same seed train differently. Calling again until one partial sum is left is
 * slower and **deterministic**, and this project takes the reproducible side.
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
  // **Nothing returns early here.** The whole workgroup has to meet the barrier below
  // together, and a thread outside the range returning makes the control flow non-uniform
  // and the result undefined.
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

/** How many partial sums one `reduceSum` produces. It is called again until one is
 *  left. */
export function reduceParts(n: number): number {
  return Math.max(1, Math.ceil(n / WORKGROUP));
}

/**
 * Folds one axis. The reduced axis sits in the middle, seen as `(outer, reduced,
 * inner)`.
 *
 * One thread takes one output cell and walks the reduced axis **in a fixed order.** No
 * atomics and no tree — the same input gives the same value.
 *
 * **This is not used for a full reduction.** With `outer = inner = 1` it is one thread
 * going round n times, so on a large tensor `Device.sumAll`'s tree is the right one. This
 * is for when there really is an axis.
 */
export type ReduceKind = "sum" | "max" | "min" | "prod";

/**
 * A reduction's starting value and one step.
 *
 * **Max and min use no sentinel.** Putting in `-3.4028235e38` first, WGSL refused it as
 * "cannot be represented as f32" (the decimal JS prints rounds above f32's maximum), and
 * building -inf by bitcast was refused too — WGSL cannot hold an infinity in a constant
 * expression. Neither appeared as an exception; both appeared as a result of 0.
 *
 * Starting from the first element and walking the rest has none of that problem, and the
 * answer is more accurate. A reduction's length is always at least 1, so the first
 * element is always there.
 */
const REDUCE_INIT: Readonly<Record<ReduceKind, string>> = {
  sum: "0.0",
  prod: "1.0",
  max: "A[base]",
  min: "A[base]",
};

/** When the starting value is the first element, that position is not counted
 *  twice. */
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
 * Reduces, and produces **the position rather than the value.**
 *
 * A tie gives **the earlier position** — the comparison is strict so it does not slide
 * to the later one. torch answers the same way.
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
 * Pads one axis with a constant at each end.
 *
 * A padded position **looks at no input position**, so a gather cannot do it. Filling
 * several axes calls this kernel once per axis — a separate kernel doing them at once
 * would be two copies to fix.
 */
export function padAxis(
  outer: number,
  outSize: number,
  size: number,
  inner: number,
  value: number,
): string {
  // **`before` is read from `P[0]`, not baked** — see the note above `gather`. A
  // concatenation pads each piece to the full width with a different `before`, and a
  // grouped convolution concatenates once per group, so the padding shader was the second
  // half of the compile storm. `outSize`, `size` and `inner` are the divisors and stay.
  const n = outer * outSize * inner;
  const literal = f32lit(value);
  return `
@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read_write> Out: array<f32>;
@group(0) @binding(2) var<storage, read> P: array<u32>;
@compute @workgroup_size(${WORKGROUP})
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
${flatId(n)}
  let o = gid / ${outSize * inner}u;
  let rest = gid % ${outSize * inner}u;
  let c = rest / ${inner}u;
  let i = rest % ${inner}u;
  let before = P[0];
  if (c < before || c >= before + ${size}u) {
    Out[gid] = ${literal};
    return;
  }
  Out[gid] = A[o * ${size * inner}u + (c - before) * ${inner}u + i];
}`;
}

/**
 * One piece of a concatenation, copied into its place along the axis. **Nothing else is
 * written**, so `cat` is one copy per piece into one buffer — where it was one pad to the
 * full width per piece and an add between them: for the U-Net's two skip connections that
 * was 1.4 ms of a 23.6 ms step in pads and adds (timestamps, 2026-09-06), all of it
 * writing zeros and adding them back. `before` comes from `P[0]` for the reason `padAxis`
 * gives.
 */
export function catCopy(outer: number, outSize: number, size: number, inner: number): string {
  const n = outer * size * inner;
  return `
@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read_write> Out: array<f32>;
@group(0) @binding(2) var<storage, read> P: array<u32>;
@compute @workgroup_size(${WORKGROUP})
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
${flatId(n)}
  let o = gid / ${size * inner}u;
  let rest = gid % ${size * inner}u;
  let c = rest / ${inner}u;
  let i = rest % ${inner}u;
  Out[o * ${outSize * inner}u + (P[0] + c) * ${inner}u + i] = A[gid];
}`;
}

/**
 * Binary cross-entropy from logits, per element, in one pass:
 * `max(x, 0) − x·y + log(1 + e^−|x|)`. Assembled from tensor ops it was nine dispatches
 * forward and a dozen back for one loss, every one a pass over the whole activation
 * (timestamps, 2026-09-06: the unary chain alone was 1.2 ms of a U-Net step).
 */
export function bceLogitsForward(n: number): string {
  return `
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> Y: array<f32>;
@group(0) @binding(2) var<storage, read_write> Out: array<f32>;
@compute @workgroup_size(${WORKGROUP})
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
${flatId(n)}
  let x = X[gid];
  Out[gid] = max(x, 0.0) - x * Y[gid] + log(1.0 + exp(-abs(x)));
}`;
}

/** Its backward: `dx = (σ(x) − y) · g`. */
export function bceLogitsBackward(n: number): string {
  return `
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> Y: array<f32>;
@group(0) @binding(2) var<storage, read> G: array<f32>;
@group(0) @binding(3) var<storage, read_write> Out: array<f32>;
@compute @workgroup_size(${WORKGROUP})
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
${flatId(n)}
  let x = X[gid];
  Out[gid] = (1.0 / (1.0 + exp(-x)) - Y[gid]) * G[gid];
}`;
}

/** `sum(dim)`'s backward — expands back along the folded axis. */
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
 * `amax`/`amin`'s backward. **A tie divides evenly.**
 *
 * That is what torch measures as — `[1,3,3,2]`'s `amax` gradient is `[0,.5,.5,0]`.
 * Choosing one of them passes the value check and leaves training subtly different. So
 * the golden's inputs contain ties on purpose, and this holds the rule.
 *
 * It walks its axis once more to count the ties. That costs the reduction's length again,
 * and carrying the count from the forward would need another buffer — at these sizes,
 * counting again is cheaper.
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
 * Where one output axis looks in the input.
 *
 * `expand`, `repeat`, `swapaxes`, `select`, `diagonal`, `rot90`, `unfold`, `flip` and
 * `split` are all combinations of these three rules. A kernel per operation would be a
 * dozen of them, and a day comes when only one of them is fixed.
 */
export interface AxisRule {
  /** The output axis's size. */
  readonly size: number;
  /** The stride in the input. **0 means an expanded axis** — the same value is read
   *  again rather than copied. */
  readonly stride: number;
  /**
   * `lin` goes straight on, `mod` wraps back (repeat), `rev` runs backwards (flip), and
   * `div` stays in place (repeat_interleave).
   */
  readonly kind: "lin" | "mod" | "rev" | "div";
  /** The period for `mod` and `rev`. Usually the input axis's size. */
  readonly wrap: number;
  /** The shift added to `mod`. `roll` uses it — for `repeat` it is 0. */
  readonly bias?: number;
}

function ruleCoord(r: AxisRule, c: string): string {
  if (r.kind === "mod") {
    const bias = r.bias ?? 0;
    return bias === 0 ? `(${c} % ${r.wrap}u)` : `((${c} + ${bias}u) % ${r.wrap}u)`;
  }
  if (r.kind === "rev") return `(${r.wrap - 1}u - ${c})`;
  // `wrap` is the repeat count. `[a,b]` twice each gives `[a,a,b,b]`.
  if (r.kind === "div") return `(${c} / ${r.wrap}u)`;
  return c;
}

/** The WGSL producing the input index from the output index. Every divisor is a
 *  literal, so no division survives. */
function sourceIndex(
  rules: readonly AxisRule[],
  offset: string,
  from: string,
  out: string,
): string {
  const lines = [`  var rest_${out} = ${from};`, `  var ${out} = ${offset};`];
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
 * The signature of a rule set. **It is the pipeline cache's key.**
 *
 * Leaving out any one part of a rule lets a different operation inherit the same shader —
 * and with shapes baked in, that is a quietly wrong answer. Which is why this function
 * lives next to the rules.
 */
export function ruleKey(rules: readonly AxisRule[]): string {
  const parts = rules.map(
    (r) => `${r.kind}:${r.size}:${r.stride}:${r.wrap}:${r.bias ?? 0}`,
  );
  return parts.join(",");
}

// **The offset is not in the key, because it is not in the shader.** It was, and one
// EfficientNet-B4 forward baked 11,042 gather shaders whose rule sets were identical and
// whose offsets were not: a grouped convolution slices its input once per group, and
// every slice starts somewhere else. With 8,207 `pad` shaders of the same shape that was
// 19,249 of the model's 19,533 pipelines — the compile storm behind #121 — while the
// convolutions themselves were 52. The offset now arrives in a one-word storage buffer,
// `P[0]`, and the rule set alone names the pipeline: 19,533 → 349 on that model, measured
// before the change was written.
//
// The sizes and strides stay baked. They are the divisors, and a divisor the compiler
// can see is the difference `convNDForwardTiled` documents; an addend costs nothing
// either way.

export function gather(rules: readonly AxisRule[]): string {
  const n = ruleCount(rules);
  return `
@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read_write> Out: array<f32>;
@group(0) @binding(2) var<storage, read> P: array<u32>;
@compute @workgroup_size(${WORKGROUP})
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
${flatId(n)}
${sourceIndex(rules, "P[0]", "gid", "src")}
  Out[gid] = A[src];
}`;
}

/**
 * The gather's backward — it collects the scattered pieces back.
 *
 * **For each input position it walks the whole output.** Scattering with atomic addition
 * makes the order differ every time, and floating point changes value with order, so the
 * same seed would train two different ways. Here the output indices are walked in
 * ascending order, so any number of runs gives the same value.
 *
 * The cost instead is **input size × output size.** At the places it is used today (the
 * backward of a shape operation, tens of elements) that is fine, and this kernel must not
 * end up inside a training loop. For that there are per-operation ways to fold (`expand`
 * is a reduction, `flip` is another flip).
 *
 * Adding the overlapping positions properly is the point — unfolding a length of 5 with
 * `unfold(3, 1)` gives the gradient `[1,2,3,2,1]`. It piles up by the overlap, and
 * without the addition it would be all ones.
 */
/**
 * Whether a rule set **can be inverted** — whether the output position can be computed
 * directly from the input position.
 *
 * Two conditions. Every axis is `lin` (running straight on) with a non-zero stride —
 * `expand` has stride 0, so several outputs look at one input and the inverse is not
 * unique. And with the strides laid out largest first, **the blocks must not overlap**: a
 * large stride has to exceed the range the axes below it cover, or the division cannot
 * separate the coordinates.
 *
 * Slicing, transposing, `select` and `permute` all satisfy this. `repeat`, `expand`,
 * `flip` and `roll` do not, and those take the walking path.
 *
 * @returns the axes sorted by descending stride, or `null` where the
 *   conditions do not hold.
 */
function invertibleAxes(
  rules: readonly AxisRule[],
): { size: number; stride: number; outStride: number }[] | null {
  if (rules.length === 0) return null;
  // The strides in the output (row-major). Used to turn the recovered coordinates back
  // into an output index.
  const outStrides: number[] = new Array(rules.length).fill(1);
  for (let d = rules.length - 2; d >= 0; d--) {
    outStrides[d] = (outStrides[d + 1] ?? 1) * (rules[d + 1]?.size ?? 1);
  }
  const axes: { size: number; stride: number; outStride: number }[] = [];
  for (const [d, r] of rules.entries()) {
    if (r.kind !== "lin" || r.stride === 0) return null;
    // An axis of size 1 has coordinate 0 always, so it contributes nothing to the
    // inversion.
    if (r.size === 1) continue;
    axes.push({ size: r.size, stride: r.stride, outStride: outStrides[d] ?? 1 });
  }
  if (axes.length === 0) return [];
  axes.sort((a, b) => b.stride - a.stride);
  // The upper axis's stride has to exceed the width the axes below cover, or the
  // division cannot separate the coordinates.
  let span = 1;
  for (let i = axes.length - 1; i >= 0; i--) {
    const a = axes[i];
    if (!a) return null;
    if (a.stride < span) return null;
    span = a.stride * a.size;
  }
  return axes;
}

export function gatherBackward(
  rules: readonly AxisRule[],
  inSize: number,
): string {
  const outN = ruleCount(rules);
  const axes = invertibleAxes(rules);
  if (axes) {
    // **The inverting path.** It computes the output position for each input position
    // in one go — `O(input)`.
    //
    // The walking path is `O(input × output)`, so twice the batch is **four times** the
    // work. Measured: 94% of a single ResNet-18 step was this one kernel, where
    // `adaptiveAvgPool(1)` was slicing the whole 4×4 and so producing **a slice whose
    // input and output are the same size.**
    const steps = axes.map((a) => `
    { let c = rest / ${a.stride}u;
      if (c >= ${a.size}u) { ok = false; } else {
        rest = rest - c * ${a.stride}u;
        t = t + c * ${a.outStride}u;
      } }`).join("");
    return `
@group(0) @binding(0) var<storage, read> G: array<f32>;
@group(0) @binding(1) var<storage, read_write> Out: array<f32>;
@group(0) @binding(2) var<storage, read> P: array<u32>;
@compute @workgroup_size(${WORKGROUP})
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
${flatId(inSize)}
  var acc = 0.0;
  if (gid >= P[0]) {
    var rest = gid - P[0];
    var t = 0u;
    var ok = true;
${steps}
    // Only a remainder of 0 makes this input exactly that output position. Otherwise it
    // is a cell that was not selected.
    if (ok && rest == 0u && t < ${outN}u) { acc = G[t]; }
  }
  Out[gid] = acc;
}`;
  }
  return `
@group(0) @binding(0) var<storage, read> G: array<f32>;
@group(0) @binding(1) var<storage, read_write> Out: array<f32>;
@group(0) @binding(2) var<storage, read> P: array<u32>;
@compute @workgroup_size(${WORKGROUP})
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
${flatId(inSize)}
  var acc = 0.0;
  for (var t = 0u; t < ${outN}u; t = t + 1u) {
${sourceIndex(rules, "P[0]", "t", "src")}
    if (src == gid) { acc = acc + G[t]; }
  }
  Out[gid] = acc;
}`;
}

/**
 * `diagflat` — puts a vector on the diagonal and 0 everywhere else.
 *
 * This one alone cannot be a gather, because most of the output **looks at no input
 * position.**
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

/** `diagflat`'s backward — it collects the diagonal alone. */
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
 * `where(condition, x, y)` — one of the two at each position.
 *
 * The gradient goes **to the chosen side only.** That is the same value as sending 0 to
 * the unchosen side, and it differs when the unchosen side is NaN — `0 * NaN` is NaN. So
 * it selects rather than multiplies.
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

/** One side's share of `where`'s backward. It places the gradient only where that side
 *  was chosen. */
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

/** Keeps the lower or upper triangle and zeroes the rest. It keeps **an area**, not a
 *  `diagonal`. */
export function triangle(rows: number, cols: number, lower: boolean, diagonal: number): string {
  const n = rows * cols;
  // tril keeps j - i <= diagonal and triu keeps j - i >= diagonal. The integer
  // subtraction can go negative, so it is read as i32 — as u32 the lower half becomes an
  // enormous number.
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
 * The cumulative sum and product. One thread walks everything before its own position.
 *
 * A parallel scan (Hillis-Steele and the rest) exists and is not used. The lengths needed
 * here are short, and a parallel scan **changes the order of the additions**, so the same
 * input can give a different value. Reproducibility comes first.
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

/** `cumsum`'s backward — it accumulates from the back. An earlier position contributed
 *  to everything after it. */
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
 * ── Reading and writing through a flat index table ─────────────────────────
 *
 * `as_strided`, `select_scatter`, `slice_scatter`, `diagonal_scatter`, `put` and
 * `index_put` are all **one job** — they differ only in which cells of the storage they
 * look at. Extract that "which cells" into a single index table and three kernels
 * suffice.
 *
 * The table comes from two places. What follows from the shape alone (strides, slices,
 * diagonals) is built on the CPU and uploaded; what depends on values (`index_put`'s
 * index tensor) is computed on the GPU. **The kernels do not distinguish them** — either
 * way they are indices in a buffer.
 */

/** Reads the cells the index table points at. */
export function flatGather(n: number): string {
  return `
@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> I: array<f32>;
@group(0) @binding(2) var<storage, read_write> Out: array<f32>;
@compute @workgroup_size(${WORKGROUP})
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
${flatId(n)}
  Out[gid] = A[u32(I[gid])];
}`;
}

/**
 * Its backward. **Overlapping positions accumulate** — a cell read twice receives the
 * gradient twice.
 *
 * Rather than atomic addition, **the reading side counts.** The same technique as this
 * file's `gatherIndexBackward`, and no contention arises on overlapping indices at all.
 */
export function flatGatherBackward(n: number, count: number): string {
  return `
@group(0) @binding(0) var<storage, read> I: array<f32>;
@group(0) @binding(1) var<storage, read> G: array<f32>;
@group(0) @binding(2) var<storage, read_write> Out: array<f32>;
@compute @workgroup_size(${WORKGROUP})
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
${flatId(n)}
  var acc = 0.0;
  for (var t = 0u; t < ${count}u; t = t + 1u) {
    if (u32(I[t]) == gid) { acc = acc + G[t]; }
  }
  Out[gid] = acc;
}`;
}

/**
 * Writes into the indexed positions of a copy.
 *
 * **They diverge on repeated indices** — accumulating adds, and otherwise **the last
 * write wins.** Measured only with non-overlapping indices, the two branches look like
 * one function.
 */
export function flatScatterInto(
  n: number,
  count: number,
  accumulate: boolean,
): string {
  const body = accumulate ? "v = v + S[t];" : "v = S[t];";
  return `
@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> I: array<f32>;
@group(0) @binding(2) var<storage, read> S: array<f32>;
@group(0) @binding(3) var<storage, read_write> Out: array<f32>;
@compute @workgroup_size(${WORKGROUP})
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
${flatId(n)}
  var v = A[gid];
  for (var t = 0u; t < ${count}u; t = t + 1u) {
    if (u32(I[t]) == gid) { ${body} }
  }
  Out[gid] = v;
}`;
}

/** The five combining rules. `mean` alone counts separately, so it is split out
 *  here. */
export const REDUCE_START: Readonly<Record<string, string>> = {
  sum: "0.0",
  prod: "1.0",
  // **Writing f32's maximum literally makes WGSL refuse it.** `3.4028235e38` is the
  // decimal rounding and sits above the true maximum (3.40282347e38), so the parser
  // discards the whole shader as "cannot be represented as f32". It is written one digit
  // lower.
  amax: "-3.4028234e38",
  amin: "3.4028234e38",
  mean: "0.0",
};

/**
 * Writes into the indexed positions **while combining.** The base of `scatter_reduce` and
 * `index_reduce`.
 *
 * **`includeSelf` is whether the original value enters as the first term.** Multiplying
 * into a plate filled with the identity gives the same answer either way (measured), so
 * measured only there this flag is invisible.
 *
 * Cells nothing reached are **left as they are** — leaking the starting value (1 for
 * `prod`, -inf for `amax`) quietly makes it a different plate.
 */
export function flatReduceInto(
  n: number,
  count: number,
  reduce: string,
  includeSelf: boolean,
): string {
  const start = REDUCE_START[reduce] ?? "0.0";
  const step = {
    sum: "acc = acc + S[t];",
    mean: "acc = acc + S[t];",
    prod: "acc = acc * S[t];",
    amax: "acc = max(acc, S[t]);",
    amin: "acc = min(acc, S[t]);",
  }[reduce] ?? "acc = acc + S[t];";
  const fold = {
    sum: "acc = acc + A[gid];",
    mean: "acc = acc + A[gid]; hits = hits + 1.0;",
    prod: "acc = acc * A[gid];",
    amax: "acc = max(acc, A[gid]);",
    amin: "acc = min(acc, A[gid]);",
  }[reduce] ?? "acc = acc + A[gid];";
  return `
@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> I: array<f32>;
@group(0) @binding(2) var<storage, read> S: array<f32>;
@group(0) @binding(3) var<storage, read_write> Out: array<f32>;
@compute @workgroup_size(${WORKGROUP})
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
${flatId(n)}
  var acc = ${start};
  var hits = 0.0;
  for (var t = 0u; t < ${count}u; t = t + 1u) {
    if (u32(I[t]) == gid) { ${step} hits = hits + 1.0; }
  }
  if (hits == 0.0) { Out[gid] = A[gid]; return; }
  ${includeSelf ? fold : ""}
  Out[gid] = ${reduce === "mean" ? "acc / hits" : "acc"};
}`;
}

/**
 * Fills the positions where the mask is true from the source **in flat order.**
 *
 * Which element of the source lands at a position is decided by **how many trues came
 * before it.** That count is computed per position, so no value has to be read back — a
 * read would make this operation asynchronous, and then every caller would have to attach
 * an `await`.
 */
export function maskedScatterKernel(n: number): string {
  return `
@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> M: array<f32>;
@group(0) @binding(2) var<storage, read> S: array<f32>;
@group(0) @binding(3) var<storage, read_write> Out: array<f32>;
@compute @workgroup_size(${WORKGROUP})
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
${flatId(n)}
  if (M[gid] == 0.0) { Out[gid] = A[gid]; return; }
  var rank = 0u;
  for (var t = 0u; t < gid; t = t + 1u) {
    if (M[t] != 0.0) { rank = rank + 1u; }
  }
  Out[gid] = S[rank];
}`;
}

/** `masked_scatter`'s backward on the source side. Unused cells receive 0. */
export function maskedScatterSourceBackward(n: number, count: number): string {
  return `
@group(0) @binding(0) var<storage, read> M: array<f32>;
@group(0) @binding(1) var<storage, read> G: array<f32>;
@group(0) @binding(2) var<storage, read_write> Out: array<f32>;
@compute @workgroup_size(${WORKGROUP})
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
${flatId(n)}
  var rank = 0u;
  for (var t = 0u; t < ${count}u; t = t + 1u) {
    if (M[t] == 0.0) { continue; }
    if (rank == gid) { Out[gid] = G[t]; return; }
    rank = rank + 1u;
  }
  Out[gid] = 0.0;
}`;
}

/**
 * `gather(dim, index)` — selects along one axis as the index tensor points.
 *
 * The indices arrive in float32, because there is one dtype. Past the range where an
 * integer sits exactly in float32 (2²⁴) it quietly reads the wrong position — far above
 * the sizes in use.
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
 * `prod`'s backward.
 *
 * **It is not written as `out / x`.** That expression collapses the moment x contains a
 * single 0 — at the zero it becomes 0/0, and everywhere else out is 0 so everything comes
 * out 0. torch gives the product of the axis's other values.
 *
 * It multiplies the others as it goes. The cost is the axis's length, and with no
 * division it is right even with a 0 among them.
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
 * `cumprod`'s backward.
 *
 * `out[j]` is the product of `A[0..j]`, so `A[k]` contributes to every output with
 * `j >= k`. That contribution is **the product of the others**, so there is no division
 * here either — it is right even with a 0 among them.
 *
 * The cost is the cube of the axis's length. It is for short axes, and a longer one would
 * need prefix and suffix products held in advance. Nothing demands that today, so it is
 * not done.
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
 * `gather(dim, index)`'s backward — collects back to the positions that were read.
 *
 * A position read several times accumulates that many times. Each input position walks
 * the output in ascending order, so there are no atomics and two runs give the same
 * value. The cost is input × output.
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

/** `index_select` — one axis is chosen by an index **vector.** The index does not vary
 *  per position. */
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

/**
 * Where each value lands among sorted boundaries. `searchsorted` and `bucketize` share
 * it.
 *
 * **Counting by comparison also produces the answer.** The golden's TS version was
 * written that way for a long time — broadcasting `seq < want` and summing is exactly
 * this number. The values are right and that route builds an `n·m` intermediate tensor.
 * A thousand boundaries against a million values is 4GB, and then this name **stops at
 * the buffer limit** — precisely at the size somebody chooses `bucketize` to be cheap
 * at, it becomes unusable.
 *
 * So it is a binary search. One thread per value, `log2(n)` iterations, no intermediate
 * tensor.
 *
 * `right` decides which side of a tie — false gives **how many are smaller than me**
 * (standing before equals), true gives **how many are smaller than or equal to me**
 * (standing after them).
 */
export function searchSorted(nSeq: number, nVal: number, right: boolean): string {
  // Whether a tie goes left or right hangs on this one comparison.
  const goRight = right ? "A[mid] <= v" : "A[mid] < v";
  return `
@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> V: array<f32>;
@group(0) @binding(2) var<storage, read_write> Out: array<f32>;
@compute @workgroup_size(${WORKGROUP})
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
${flatId(nVal)}
  let v = V[gid];
  var lo = 0u;
  var hi = ${nSeq}u;
  while (lo < hi) {
    let mid = lo + (hi - lo) / 2u;
    if (${goRight}) { lo = mid + 1u; } else { hi = mid; }
  }
  Out[gid] = f32(lo);
}`;
}

/** `index_select`'s backward. A position chosen several times accumulates. */
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

export function convOut(size: number, pad: number, kernel: number, stride: number,
                        dilation = 1): number {
  return Math.floor((size + 2 * pad - ((kernel - 1) * dilation + 1)) / stride) + 1;
}

/**
 * Pooling's output extent, which is `convOut` **plus the ceiling rule.**
 *
 * `ceilMode` rounds up rather than down, and the rounding alone is not the rule: torch
 * then drops the last window if it *starts* in the padding past the end. Without that
 * second half, a ceiling can add a window made entirely of padded cells — an extra
 * output column that torch does not produce, so the two answers part in shape rather
 * than in value and every comparison after it is against the wrong extent.
 */
export function poolOut(size: number, pad: number, kernel: number, stride: number,
                        ceilMode = false, dilation = 1): number {
  // **A dilated window covers `dil·(k−1)+1` cells and reads `k` of them.** The extent
  // is what the output size is computed from, which is why it cannot be folded into
  // the kernel size — the count of reads stays `k`.
  const reach = dilation * (kernel - 1) + 1;
  if (!ceilMode) return Math.floor((size + 2 * pad - reach) / stride) + 1;
  let out = Math.ceil((size + 2 * pad - reach) / stride) + 1;
  if ((out - 1) * stride >= size + pad) out -= 1;
  return out;
}

/**
 * The shape of a convolution with no regard for the number of dimensions.
 *
 * One kernel generator covers 1, 2 and 3 dimensions. With the spatial axes held as an
 * array, conv1d simply has one axis and conv3d has three, and the rest of the structure
 * is the same — a kernel per dimension is three copies, and a day comes when only one of
 * them is fixed. The sister library really was in that state.
 */
export interface ConvNDShape {
  readonly N: number;
  readonly C: number;
  readonly O: number;
  /** The input's spatial axes. */
  readonly inDims: readonly number[];
  readonly kernel: readonly number[];
  readonly stride: readonly number[];
  readonly pad: readonly number[];
  /** How far apart the filter's cells sit. One is the ordinary convolution. */
  readonly dilation?: readonly number[];
  readonly outDims: readonly number[];
  /**
   * Channel groups, torch's `groups=`. One is the ordinary convolution; `C` is
   * depthwise. **Inside the kernel, not around it** — see `convND` for what the
   * slicing loop cost.
   */
  readonly groups?: number;
}

/** The three per-group sizes every conv kernel needs. */
function grouped(s: ConvNDShape): { groups: number; cin: number; cout: number } {
  const groups = s.groups ?? 1;
  return { groups, cin: s.C / groups, cout: s.O / groups };
}

/** The filter's spacing per axis, defaulting to one. */
function convDil(s: ConvNDShape, d: number): number {
  return s.dilation?.[d] ?? 1;
}

export function convNDKey(s: ConvNDShape): string {
  // **`dilation` belongs in the key.** The shader bakes the spacing in, so two calls
  // that differ only in dilation are two shaders — left out of the key the first one
  // is cached and the second silently reuses it, which is a wrong answer with no
  // exception anywhere near it.
  return [s.N, s.C, s.O, s.inDims, s.kernel, s.stride, s.pad,
    s.dilation ?? s.kernel.map(() => 1), s.groups ?? 1].join("|");
}

/** The product accumulated from the back — how many elements one step along an axis
 *  skips. */
function suffixStrides(dims: readonly number[]): number[] {
  const out: number[] = new Array<number>(dims.length).fill(1);
  for (let d = dims.length - 2; d >= 0; d--) {
    out[d] = (out[d + 1] ?? 1) * (dims[d + 1] ?? 1);
  }
  return out;
}


/**
 * The tiled convolution — **an implicit GEMM.**
 *
 * A port of the kernel `tests/browser/wgsl_conv.js` measured at 72–284% of TF.js. Before
 * the port, the simple kernel below was in place and one ResNet step was 272× the sister
 * library's.
 *
 * ## The axes are flipped
 *
 * The bench version writes its result as `(N·OH·OW, O)`. We are NCHW, so left that way it
 * needs one more transpose. Flipping the GEMM to `(O, N·OH·OW)` instead means
 *
 * - the weight tile reads contiguous positions as `W[f·K + k]`, and
 * - neighbouring threads in the result tile write neighbouring `ow`, landing directly in
 *   NCHW.
 *
 * ## This is where baking the shape pays most
 *
 * Every tile load redoes six or seven divisions per element. With the divisor in a
 * uniform the compiler cannot turn those into multiplications and shifts, and a GPU has
 * no integer division hardware — in the bench that one thing separated 43% from 284%.
 */
/**
 * What a convolution's forward may do to each value on the way out: add a residual
 * (the same shape as the output) and clamp at zero. **torch's `ConvAddReLU2d`**, as one
 * kernel — the calls this saves are the point: a ResNet-18 forward at batch 16 was 64
 * dispatches for 3.0 ms of GPU work on a 4090 (5.6 ms on the clock), and 26 of the
 * 64 were a relu or a residual add.
 */
export interface ConvEpilogue {
  readonly relu: boolean;
  readonly residual: boolean;
}

/** The epilogue's WGSL, over the variable `v` at output index `idx`. */
function epilogueWgsl(e: ConvEpilogue | undefined, v: string, idx: string): string {
  if (!e) return "";
  return (e.residual ? `  ${v} = ${v} + R[${idx}];\n` : "") + (e.relu ? `  ${v} = max(${v}, 0.0);\n` : "");
}

/** The residual binding, when the epilogue takes one. It sits after the bias. */
function residualBinding(e: ConvEpilogue | undefined, slot: number): string {
  return e?.residual ? `@group(0) @binding(${slot}) var<storage, read> R: array<f32>;` : "";
}

export function convNDForwardTiled(
  s: ConvNDShape, hasBias: boolean, epilogue?: ConvEpilogue,
): string {
  const inStride = suffixStrides(s.inDims);
  const outStride = suffixStrides(s.outDims);
  const inSpace = s.inDims.reduce((a, b) => a * b, 1);
  const outSpace = s.outDims.reduce((a, b) => a * b, 1);
  const kSpace = s.kernel.reduce((a, b) => a * b, 1);
  const { groups, cin, cout } = grouped(s);
  // Per group: `cout` filters over `cin` channels. The weight is `(O, cin, k…)` on every
  // path, so its row stride is the per-group `K` whatever `groups` is.
  const K = cin * kSpace;
  const P = s.N * outSpace;
  // Split, the partial sums land in a slab per piece and `sumSplitsConv` adds them (and
  // the bias) once more — the bias binding is not taken here in that case.
  const splits = convForwardSplit(s);
  const KT = tileDepth(tileShape(cout, P).TM);
  const withBias = hasBias && splits === 1;
  // Split, the epilogue waits for `sumSplitsConv` — a partial sum cannot be clamped.
  const ep = splits === 1 ? epilogue : undefined;
  const slot = withBias ? 3 : 2;
  return tiledGemm({
    M: cout, N: P, K, groups, splits,
    bindings: `@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> Wt: array<f32>;
${withBias ? "@group(0) @binding(2) var<storage, read> B: array<f32>;" : ""}
${residualBinding(ep, slot)}
@group(0) @binding(${ep?.residual ? slot + 1 : slot}) var<storage, read_write> Out: array<f32>;`,
    // The weights run as (O, K), so one row is contiguous in its entirety.
    loadA: `          v = Wt[(grp * ${cout}u + arow) * ${K}u + kk];`,
    // The column is (batch, output position) and constant per thread: resolved once.
    prepB: `  let pb_n = (bcol / ${outSpace}u) * ${s.C * inSpace}u;
${s.outDims.map((size, d) =>
      `  let pb_o${d} = i32(((bcol / ${outStride[d] ?? 1}u) % ${size}u) * ${s.stride[d] ?? 1}u) - ${s.pad[d] ?? 0};`).join("\n")}`,
    // im2col is built here **without being laid out in memory.** The kernel side of
    // `kk` — (channel, kernel position) — is split once per K-tile and carried.
    prepBTile: `    var kc = 0u;
${s.kernel.map((_, d) => `    var kd${d} = 0u;`).join("\n")}
${splitDigits(["kc", ...s.kernel.map((_, d) => `kd${d}`)], [cin, ...s.kernel], `t * ${KT}u + bkk0`, "    ")}`,
    stepB: carryDigits(["kc", ...s.kernel.map((_, d) => `kd${d}`)], [cin, ...s.kernel], "bstep", "      "),
    loadB: `          let ch = grp * ${cin}u + kc;
${s.outDims.map((_, d) =>
      `          let i${d} = pb_o${d} + i32(kd${d} * ${convDil(s, d)}u);`).join("\n")}
          if (${s.inDims.map((size, d) => `i${d} >= 0 && i${d} < ${size}`).join(" && ")}) {
            v = X[pb_n + ch * ${inSpace}u
              + ${s.inDims.map((_, d) => `u32(i${d}) * ${inStride[d] ?? 1}u`).join(" + ")}];
          }`,
    emit: `  let bn = col / ${outSpace}u;
${s.outDims.map((size, d) =>
      `  let o${d} = (col / ${outStride[d] ?? 1}u) % ${size}u;`).join("\n")}
  let fo = grp * ${cout}u + f;
  let idx = ${splits > 1 ? `part * ${s.N * s.O * outSpace}u + ` : ""}(bn * ${s.O}u + fo) * ${outSpace}u
    + ${s.outDims.map((_, d) => `o${d} * ${outStride[d] ?? 1}u`).join(" + ")};
  var r = v;
${withBias ? "  r = r + B[fo];\n" : ""}${epilogueWgsl(ep, "r", "idx")}  Out[idx] = r;`,
  });
}

/** The output columns one thread of `convDirect2d` computes. */
export const DIRECT_STRIP = 4;

/** The most output channels one thread accumulates — with the strip, 64 registers. */
const DIRECT_COUT = 16;

/** Workgroup storage the weights may take in `convDirect2d`: WebGPU's guaranteed floor. */
const DIRECT_WEIGHT_BYTES = 16384;

/**
 * How many output channels one thread of `convDirect2d` takes: the most that fit the
 * registers and whose weights fit the workgroup, and a divisor of nothing in particular
 * — the last slice is short.
 */
export function directCoutSlice(s: ConvNDShape): number {
  const kSpace = s.kernel.reduce((a, b) => a * b, 1);
  const perCout = s.C * kSpace * 4;
  return Math.max(1, Math.min(DIRECT_COUT, s.O, Math.floor(DIRECT_WEIGHT_BYTES / perCout)));
}

/**
 * Whether a convolution takes the direct kernel: two spatial axes, no groups, no
 * dilation, a stride of one or two, at most 32 input channels, a kernel no wider than
 * seven. Above 32 channels the GEMM's reuse wins; the boundary was measured, not chosen.
 */
export function directFits(s: ConvNDShape): boolean {
  return s.inDims.length === 2 && (s.groups ?? 1) === 1 && (s.dilation ?? [1, 1]).every((d) => d === 1)
    && s.stride.every((v) => v === 1 || v === 2) && s.C <= 32 && s.kernel.every((v) => v <= 7)
    && s.kernel.length === 2;
}

/** The grid for `convDirect2d`: strips over (batch, output row, strip), then channel slices. */
export function directGrid(s: ConvNDShape): [number, number, number] {
  const [OH = 1, OW = 1] = s.outDims;
  const strips = s.N * OH * Math.ceil(OW / DIRECT_STRIP);
  return [Math.ceil(strips / 256), Math.ceil(s.O / directCoutSlice(s)), 1];
}

/**
 * The direct 2-D convolution — **for the narrow layers the GEMM serves badly.**
 *
 * Measured on the M4 Max (2026-09-07): a 16 → 16 layer at 96 × 96, batch 16, ran its
 * 0.68 GFLOP in 0.5 ms through the implicit GEMM — 1.4 TFLOP/s against a peak above
 * ten, and against a memory floor of 0.06 ms. Neither arithmetic nor bandwidth: the
 * staging through workgroup memory, the two barriers per K-tile, and the gather were
 * the time. Winograd was costed and refused — its fourfold data expansion costs what
 * its 2.25× fewer multiplies save when there are sixteen channels.
 *
 * Here one thread owns four consecutive output columns of one row for up to sixteen
 * output channels — sixty-four accumulators in registers. The workgroup's 256 threads
 * share one copy of the weights in workgroup memory (16 × 16 × 9 floats is 9 KB), and
 * each thread reads the input row segment its strip needs straight into registers,
 * once per (input channel, kernel row). No staging of the activation, no barrier after
 * the weights land. Output channels beyond the slice are another dispatch along y.
 *
 * The same kernel serves the stride-1 input gradient (a forward on turned weights) and
 * takes the forward's epilogue (bias, residual, ReLU).
 *
 * **The weight gradient stays on the GEMM.** A direct version was written the same day
 * — a workgroup per 2 × 4 channel block and a piece of the positions, seventy-two
 * accumulators a thread, a tree fold — and it agreed with the core everywhere and ran
 * 0.4 ms slower than the GEMM over the U-Net's narrow layers (4.2 against 3.8 ms): a
 * strip of four columns gives 3.6 multiplies per global load, and the reduction over
 * every position is where the GEMM's tile reuse actually pays. Removed rather than
 * kept behind a switch.
 */
export function convDirect2d(s: ConvNDShape, hasBias: boolean, epilogue?: ConvEpilogue, turned = false): string {
  const [IH = 1, IW = 1] = s.inDims;
  const [OH = 1, OW = 1] = s.outDims;
  const [KH = 1, KW = 1] = s.kernel;
  const [SH = 1, SW = 1] = s.stride;
  const [PH = 0, PW = 0] = s.pad;
  const TS = DIRECT_STRIP;
  const slice = directCoutSlice(s);
  const kSpace = KH * KW;
  const wCells = slice * s.C * kSpace;
  // The input columns one strip touches on one row: (TS − 1)·stride + KW.
  const L = (TS - 1) * SW + KW;
  const stripsPerRow = Math.ceil(OW / TS);
  const strips = s.N * OH * stripsPerRow;
  const acc: string[] = [];
  for (let c = 0; c < slice; c++) for (let j = 0; j < TS; j++) acc.push(`  var a${c}_${j} = 0.0;`);
  const seg: string[] = [];
  for (let l = 0; l < L; l++) {
    seg.push(`      let iw${l} = iw0 + ${l};`);
    seg.push(`      let x${l} = select(0.0, X[rowBase + u32(iw${l})], inRow && iw${l} >= 0 && iw${l} < ${IW});`);
  }
  const fma: string[] = [];
  for (let kw = 0; kw < KW; kw++) {
    fma.push(`      { let wb = wRow + ${kw}u;`);
    for (let c = 0; c < slice; c++) {
      fma.push(`        let w${c} = Ws[wb + ${c * s.C * kSpace}u];`);
      for (let j = 0; j < TS; j++) {
        fma.push(`        a${c}_${j} = fma(w${c}, x${j * SW + kw}, a${c}_${j});`);
      }
    }
    fma.push("      }");
  }
  const store: string[] = [];
  for (let c = 0; c < slice; c++) {
    store.push(`  if (co + ${c}u < ${s.O}u) {`);
    store.push(`    let ob = (n * ${s.O}u + co + ${c}u) * ${OH * OW}u + oh * ${OW}u + ow0;`);
    for (let j = 0; j < TS; j++) {
      store.push(`    if (ow0 + ${j}u < ${OW}u) {`);
      store.push(`      var r = a${c}_${j};`);
      if (hasBias) store.push(`      r = r + B[co + ${c}u];`);
      store.push(epilogueWgsl(epilogue, "r", `ob + ${j}u`).replace(/^/gm, "    "));
      store.push(`      Out[ob + ${j}u] = r;`);
      store.push("    }");
    }
    store.push("  }");
  }
  const slot = hasBias ? 3 : 2;
  return `
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> Wt: array<f32>;
${hasBias ? "@group(0) @binding(2) var<storage, read> B: array<f32>;" : ""}
${residualBinding(epilogue, slot)}
@group(0) @binding(${epilogue?.residual ? slot + 1 : slot}) var<storage, read_write> Out: array<f32>;
var<workgroup> Ws: array<f32, ${wCells}>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let co = wid.y * ${slice}u;
  // This slice's weights, (slice, C, KH, KW), into the workgroup — rows past O are zero.
  // Turned, the buffer is the forward's (C, O, k…) weight read as (O, C, k…) with every
  // kernel axis reversed — the input gradient's weights, without a pass to make them.
  for (var i = lid.x; i < ${wCells}u; i = i + 256u) {
    let c = i / ${s.C * kSpace}u;
    let rest = i - c * ${s.C * kSpace}u;
    ${turned
      ? `let ci = rest / ${kSpace}u;
    let kk = rest - ci * ${kSpace}u;
    Ws[i] = select(0.0, Wt[(ci * ${s.O}u + co + c) * ${kSpace}u + (${kSpace - 1}u - kk)], co + c < ${s.O}u);`
      : `Ws[i] = select(0.0, Wt[(co + c) * ${s.C * kSpace}u + rest], co + c < ${s.O}u);`}
  }
  workgroupBarrier();
  let strip = wid.x * 256u + lid.x;
  // Nothing returns before the barrier above; past the last strip a thread idles here.
  if (strip >= ${strips}u) { return; }
  let n = strip / ${OH * stripsPerRow}u;
  let rest = strip - n * ${OH * stripsPerRow}u;
  let oh = rest / ${stripsPerRow}u;
  let ow0 = (rest - oh * ${stripsPerRow}u) * ${TS}u;
  let iw0 = i32(ow0 * ${SW}u) - ${PW};
${acc.join("\n")}
  for (var ci = 0u; ci < ${s.C}u; ci = ci + 1u) {
    for (var kh = 0u; kh < ${KH}u; kh = kh + 1u) {
      let ih = i32(oh * ${SH}u + kh) - ${PH};
      let inRow = ih >= 0 && ih < ${IH};
      let rowBase = (n * ${s.C}u + ci) * ${IH * IW}u + u32(max(ih, 0)) * ${IW}u;
${seg.join("\n")}
      let wRow = ci * ${kSpace}u + kh * ${KW}u;
${fma.join("\n")}
    }
  }
${store.join("\n")}
}`;
}

/**
 * How many pieces the forward's reduction is split into. The same policy as the weight
 * gradient's, for the same reason: **the late layers of a network make a tile grid too
 * small to fill the GPU.** ResNet-18's 512 → 512 convolution on a 4 × 4 plane at batch
 * 16 is 256 positions by 512 filters — 32 workgroups of 64 × 64 for a card with 128
 * SMs, and 8 at batch 1 — while its reduction is 4,608 long. Measured before the
 * split: 1.9 ms for that layer on a 4090, about 1 % of the card's peak.
 */
export function convForwardSplit(s: ConvNDShape): number {
  const { groups, cin, cout } = grouped(s);
  const P = s.N * s.outDims.reduce((a, b) => a * b, 1);
  const tiles = tileCount(cout, P) * groups;
  const K = cin * s.kernel.reduce((a, b) => a * b, 1);
  const WANT = 128;
  if (tiles >= WANT) return 1;
  const MIN_PER_SPLIT = 256;
  return Math.max(1, Math.min(Math.ceil(WANT / tiles), Math.floor(K / MIN_PER_SPLIT)));
}

/** Sums a split forward's partials and adds the bias per output channel. */
export function sumSplitsConv(
  s: ConvNDShape, splits: number, hasBias: boolean, epilogue?: ConvEpilogue,
): string {
  const outSpace = s.outDims.reduce((a, b) => a * b, 1);
  const n = s.N * s.O * outSpace;
  const slot = hasBias ? 2 : 1;
  return `
@group(0) @binding(0) var<storage, read> Parts: array<f32>;
${hasBias ? "@group(0) @binding(1) var<storage, read> B: array<f32>;" : ""}
${residualBinding(epilogue, slot)}
@group(0) @binding(${epilogue?.residual ? slot + 1 : slot}) var<storage, read_write> Out: array<f32>;
@compute @workgroup_size(${WORKGROUP})
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
${flatId(n)}
  var v = 0.0;
  for (var s = 0u; s < ${splits}u; s = s + 1u) {
    v = v + Parts[s * ${n}u + gid];
  }
${hasBias ? `  v = v + B[(gid / ${outSpace}u) % ${s.O}u];\n` : ""}${epilogueWgsl(epilogue, "v", "gid")}  Out[gid] = v;
}`;
}

/**
 * Depthwise forward — **one thread per output cell, no tile.**
 *
 * With `groups == C` and one channel per group, the tiled GEMM has one valid row in a
 * 64-row tile and a reduction of `kSpace` (nine, for a 3×3) inside a 16-wide tile: it
 * computes 1/64 of what it loads. Measured on EfficientNet-B4 at 64×64, the grouped GEMM
 * took a warm forward from 2.2 s (the old slice-and-join loop) to 3.1 s despite cutting
 * dispatches from 140,541 to 1,285 — the waste was the whole difference. Here each
 * invocation reads its `kSpace` weights and inputs and writes one cell, which is what a
 * depthwise convolution is.
 *
 * Forward only. The two backward kernels stay on the grouped GEMM, which is correct
 * (asked by the golden) and only slow, and training a depthwise network in a browser is
 * not the product's first number.
 */
export function depthwiseForward(s: ConvNDShape, hasBias: boolean): string {
  const inStride = suffixStrides(s.inDims);
  const outStride = suffixStrides(s.outDims);
  const kStride = suffixStrides(s.kernel);
  const inSpace = s.inDims.reduce((a, b) => a * b, 1);
  const outSpace = s.outDims.reduce((a, b) => a * b, 1);
  const kSpace = s.kernel.reduce((a, b) => a * b, 1);
  const n = s.N * s.C * outSpace;
  return `
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> Wt: array<f32>;
${hasBias ? "@group(0) @binding(2) var<storage, read> B: array<f32>;" : ""}
@group(0) @binding(${hasBias ? 3 : 2}) var<storage, read_write> Out: array<f32>;
@compute @workgroup_size(${WORKGROUP})
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
${flatId(n)}
  let p = gid % ${outSpace}u;
  let nc = gid / ${outSpace}u;
  let c = nc % ${s.C}u;
${s.outDims.map((size, d) =>
      `  let o${d} = (p / ${outStride[d] ?? 1}u) % ${size}u;`).join("\n")}
  var acc = 0.0;
  for (var k = 0u; k < ${kSpace}u; k = k + 1u) {
${s.kernel.map((size, d) =>
      `    let kk${d} = (k / ${kStride[d] ?? 1}u) % ${size}u;`).join("\n")}
${s.outDims.map((_, d) =>
      `    let i${d} = i32(o${d} * ${s.stride[d] ?? 1}u + kk${d} * ${convDil(s, d)}u) - ${s.pad[d] ?? 0};`)
      .join("\n")}
    if (${s.inDims.map((size, d) => `i${d} >= 0 && i${d} < ${size}`).join(" && ")}) {
      acc = fma(Wt[c * ${kSpace}u + k],
                X[nc * ${inSpace}u + ${s.inDims.map((_, d) => `u32(i${d}) * ${inStride[d] ?? 1}u`).join(" + ")}],
                acc);
    }
  }
  Out[gid] = acc${hasBias ? " + B[c]" : ""};
}`;
}

/** Whether a shape is depthwise — every channel its own group, one channel per filter. */
export function isDepthwise(s: ConvNDShape): boolean {
  const groups = s.groups ?? 1;
  return groups > 1 && groups === s.C && s.O === s.C;
}

/**
 * The tile's shape for a GEMM of `M` rows by `N` columns — **it follows the matrix.**
 *
 * The tile was 64 × 64 whatever the shape. A convolution with sixteen output channels
 * has M = 16, so three quarters of every tile were rows that do not exist: the threads
 * loaded zeros for them, multiplied them, and threw the result away at `emit`. Measured
 * on the M4 Max, the 16 → 16 layer of a U-Net at 96 × 96 ran at 261 GFLOP/s of useful
 * work against torch's 2.3 TFLOP/s, and the U-Net's early layers are all of that shape.
 *
 * Every shape here has 4,096 cells and 256 threads with a 4 × 4 micro-tile each, so the
 * inner loop is the same and only the loads and the grid change. The one with the least
 * padding wins; a tie goes to the square.
 */
export function tileShape(M: number, N: number): { TM: number; TN: number } {
  let best = { TM: 64, TN: 64 };
  let least = Math.ceil(M / 64) * Math.ceil(N / 64) * 4096;
  for (const [TM, TN] of [[32, 128], [16, 256]] as const) {
    const area = Math.ceil(M / TM) * Math.ceil(N / TN) * 4096;
    if (area < least) { least = area; best = { TM, TN }; }
  }
  return best;
}

/**
 * The inner tile's depth. Sixteen, except for the 16 × 256 shape: its B tile alone is
 * 16 KB at depth sixteen, and 16 KB is WebGPU's guaranteed workgroup storage — one byte
 * over and the pipeline is refused on the smallest adapters. Eight halves it.
 */
export function tileDepth(TM: number): number {
  return TM === 16 ? 8 : 16;
}

/** How many tiles a GEMM of `M` by `N` is — the grid before any split or group. */
function tileCount(M: number, N: number): number {
  const { TM, TN } = tileShape(M, N);
  return Math.ceil(M / TM) * Math.ceil(N / TN);
}

/** The dispatch grid for the tiled conv. Rows are output channels; columns are batch
 *  and output position. */
export function convTiledGrid(s: ConvNDShape): [number, number, number] {
  const P = s.N * s.outDims.reduce((a, b) => a * b, 1);
  const { groups, cout } = grouped(s);
  const { TM, TN } = tileShape(cout, P);
  return [Math.ceil(P / TN), Math.ceil(cout / TM), groups * convForwardSplit(s)];
}

/**
 * The skeleton of the tiled GEMM.
 *
 * The forward and the two backwards are **one structure with different indexing.**
 * Copying the skeleton three times means a day comes when only one of them is fixed, and
 * that one will be on the gradient side — the side a value check cannot see.
 *
 * @param loadA WGSL (a single expression) giving the left tile's element at
 *   row `arow`, inner `kk`.
 * @param loadB a WGSL block giving the right tile's element at inner `kk`,
 *   column `col`. It lands in `v`.
 * @param emit a WGSL block writing value `v` at row `f`, column `col`.
 */
/**
 * **The tiled GEMMs do not use subgroup matrices, and this was measured, not assumed.**
 * The inner loop was rewritten on them (2026-09-06: eight subgroups per workgroup over
 * the staged As/Bs, the result through an aliased half-tile slab so the footprint stayed
 * the scalar path's, every U-Net shape agreeing with the core to 1e-7) and the U-Net step
 * did not move — 27.3 ms against 26.3. The multiply is not where these kernels spend
 * their time: the im2col gather that fills the B tile is, and a faster multiply behind
 * the same gather buys nothing. `matmulSubgroup` keeps the hardware path where it pays —
 * a plain matrix product loads its operands straight from storage and there the same
 * instructions reach torch's speed.
 */
function tiledGemm(opts: {
  readonly M: number;
  readonly N: number;
  readonly K: number;
  readonly bindings: string;
  readonly loadA: string;
  readonly loadB: string;
  /**
   * A WGSL block run once per thread before the reduction loop, with `bcol` — the
   * column this thread loads for the right tile — in scope. **In every tile shape the
   * B column is constant per thread** (256 threads, `KT × TN` cells, `TN` divides 256),
   * so whatever `loadB` derives from `col` alone can be derived here once instead of
   * once per element. The convolution's gather spent six integer divisions per element
   * resolving the column into (batch, output position) — measured, more work than the
   * sixteen multiplies that element then feeds in a 16-row tile.
   */
  readonly prepB?: string;
  /**
   * A WGSL block run once per K-tile per thread, before the B loads, with `t` and
   * `bcol` in scope. Across the `sload` iterations that follow, this thread's `kk`
   * advances by a fixed `bstep` (= 256 / TN) — an arithmetic progression — so
   * whatever `loadB` derives from `kk` by division can be derived here once for the
   * first `kk` and **carried** forward by `stepB` after every load. The convolution's
   * gathers spent three integer divisions per element on the (batch, position) or
   * (channel, kernel) side of `kk`; carried, that is a few compares.
   */
  readonly prepBTile?: string;
  /** A WGSL block run after every B load (guarded or not), advancing what `prepBTile`
   *  set up by `bstep`. */
  readonly stepB?: string;
  readonly emit: string;
  /**
   * How many pieces to split the reduction into. 1 means no split.
   *
   * Split, what `emit` receives is a partial sum and which piece it is arrives as `part`
   * — the caller decides where to accumulate it.
   */
  readonly splits?: number;
  /**
   * Channel groups, each an independent GEMM of the same shape. They ride the z axis
   * with the splits: `wid.z = grp * splits + part`, and `grp` is in scope for `loadA`,
   * `loadB` and `emit`. One dispatch per layer instead of three per group — a
   * depthwise stack of 224 layers was 55,794 gathers, 27,930 pads and as many
   * convolutions, measured, before the group moved inside.
   */
  readonly groups?: number;
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
      store.push(`  emit(row0 + ${i}u, col0 + ${j}u, c${i}${j}, grp, part);`);
    }
  }
  const splits = opts.splits ?? 1;
  // The tile follows the matrix (see `tileShape`): TM × TN cells, TM/4 × TN/4 threads.
  const { TM, TN } = tileShape(opts.M, opts.N);
  const TX = TN / 4;
  const TY = TM / 4;
  const KT = tileDepth(TM);
  // How many tiles each piece takes. The last piece may take slightly fewer, so the
  // count is kept inside the boundary.
  const allTiles = Math.ceil(opts.K / KT);
  const perSplit = Math.ceil(allTiles / splits);
  const aCells = TM * KT;
  const bCells = KT * TN;
  const aLoads = Math.ceil(aCells / 256);
  const bLoads = Math.ceil(bCells / 256);
  const inner = `    for (var k = 0u; k < ${KT}u; k = k + 1u) {
      let a0 = As[(lid.y * 4u + 0u) * ${KT}u + k];
      let a1 = As[(lid.y * 4u + 1u) * ${KT}u + k];
      let a2 = As[(lid.y * 4u + 2u) * ${KT}u + k];
      let a3 = As[(lid.y * 4u + 3u) * ${KT}u + k];
      let b0 = Bs[k * ${TN}u + lid.x * 4u + 0u];
      let b1 = Bs[k * ${TN}u + lid.x * 4u + 1u];
      let b2 = Bs[k * ${TN}u + lid.x * 4u + 2u];
      let b3 = Bs[k * ${TN}u + lid.x * 4u + 3u];
${fma.join("\n")}
    }`;
  const finish = store.join("\n");
  return `
${opts.bindings}

var<workgroup> As: array<f32, ${aCells}>;
var<workgroup> Bs: array<f32, ${bCells}>;

fn emit(f: u32, col: u32, v: f32, grp: u32, part: u32) {
  if (f >= ${opts.M}u || col >= ${opts.N}u) { return; }
${opts.emit}
}

@compute @workgroup_size(${TX}, ${TY})
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
  let tid = lid.y * ${TX}u + lid.x;
  let row0 = wid.y * ${TM}u + lid.y * 4u;
  let col0 = wid.x * ${TN}u + lid.x * 4u;
${decl.join("\n")}
${zero.join("\n")}
  let grp = wid.z / ${splits}u;
  let part = wid.z % ${splits}u;
  let tFrom = part * ${perSplit}u;
  let tTo = min(tFrom + ${perSplit}u, ${allTiles}u);
  let bcol = wid.x * ${TN}u + tid % ${TN}u;
  let bstep = ${256 / TN}u;
  let bkk0 = tid / ${TN}u;
${opts.prepB ?? ""}
  for (var t = tFrom; t < tTo; t = t + 1u) {
    for (var sload = 0u; sload < ${aLoads}u; sload = sload + 1u) {
      let idx = sload * 256u + tid;
      if (idx < ${aCells}u) {
        let arow = wid.y * ${TM}u + idx / ${KT}u;
        let kk = t * ${KT}u + idx % ${KT}u;
        var v = 0.0;
        if (arow < ${opts.M}u && kk < ${opts.K}u) {
${opts.loadA}
        }
        As[idx] = v;
      }
    }
${opts.prepBTile ?? ""}
    for (var sload = 0u; sload < ${bLoads}u; sload = sload + 1u) {
      let idx = sload * 256u + tid;
      if (idx < ${bCells}u) {
        let kk = t * ${KT}u + idx / ${TN}u;
        let col = bcol;
        var v = 0.0;
        if (kk < ${opts.K}u && col < ${opts.N}u) {
${opts.loadB}
        }
        Bs[idx] = v;
      }
${opts.stepB ?? ""}
    }
    workgroupBarrier();
${inner}
    workgroupBarrier();
  }
${finish}
}`;
}


/**
 * WGSL for carrying a mixed-radix index forward. `names[d]` are `var<function>` digits
 * with radices `sizes[d]` (last fastest); `carry` is what to add. A while per digit, since
 * the step can exceed a small radix. The digits were set from a division once per K-tile;
 * this is what replaces the divisions on every element after (see `prepBTile`).
 */
function carryDigits(names: readonly string[], sizes: readonly number[], carry: string, indent: string): string {
  const lines: string[] = [`${indent}var cy = ${carry};`];
  for (let d = names.length - 1; d >= 0; d--) {
    const last = d === 0;
    lines.push(`${indent}${names[d]} = ${names[d]} + cy;`);
    if (!last) {
      lines.push(`${indent}cy = 0u;`);
      lines.push(`${indent}while (${names[d]} >= ${sizes[d]}u) { ${names[d]} = ${names[d]} - ${sizes[d]}u; cy = cy + 1u; }`);
    }
  }
  return lines.join("\n");
}

/** The digits of `value` in the mixed radix `sizes` (last fastest), assigned to `names`. */
function splitDigits(names: readonly string[], sizes: readonly number[], value: string, indent: string): string {
  const lines: string[] = [`${indent}var rest_ = ${value};`];
  for (let d = names.length - 1; d >= 0; d--) {
    if (d === 0) lines.push(`${indent}${names[d]} = rest_;`);
    else lines.push(`${indent}${names[d]} = rest_ % ${sizes[d]}u; rest_ = rest_ / ${sizes[d]}u;`);
  }
  return lines.join("\n");
}

/** The WGSL resolving an input position (`col`) and a kernel position (`kk`) into
 *  coordinates. */
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
      `${indent}let i${d} = i32(o${d} * ${s.stride[d] ?? 1}u + kk${d} * `
      + `${convDil(s, d)}u) - ${s.pad[d] ?? 0};`)
      .join("\n"),
    guard: s.inDims.map((size, d) => `i${d} >= 0 && i${d} < ${size}`).join(" && "),
    offset: s.inDims.map((_, d) => `u32(i${d}) * ${inStride[d] ?? 1}u`).join(" + "),
  };
}

/**
 * The gradient to the weights — the tiled version.
 *
 * A GEMM of `dW[o, (c,k)] = Σ_p G[p, o] · X_col[p, (c,k)]`. Rows are output channels,
 * columns are `(input channel, kernel position)`, and the inner axis is batch and output
 * position. The result comes out contiguous as `(O, K)`, which is the weights' shape as
 * it is.
 */
export function convNDGradWeightTiled(s: ConvNDShape): string {
  const outSpace = s.outDims.reduce((a, b) => a * b, 1);
  const kSpace = s.kernel.reduce((a, b) => a * b, 1);
  const inSpace = s.inDims.reduce((a, b) => a * b, 1);
  const K = s.N * outSpace;
  const { groups, cin, cout } = grouped(s);
  const cols = cin * kSpace;
  const splits = convGradWeightSplit(s);
  const c = patchCoords({ ...s }, "          ");
  return tiledGemm({
    M: cout, N: cols, K, splits, groups,
    bindings: `@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> G: array<f32>;
@group(0) @binding(2) var<storage, read_write> Out: array<f32>;`,
    // Left: G seen as (output channel × batch and output position).
    loadA: `          let gn = kk / ${outSpace}u;
          let gp = kk % ${outSpace}u;
          v = G[(gn * ${s.O}u + grp * ${cout}u + arow) * ${outSpace}u + gp];`,
    // Right: im2col is built here **without being laid out in memory.** Columns are
    // (channel, kernel position) and the inner axis is (batch, output position) — the
    // same computation as the forward with the axes swapped.
    prepB: `  let ch = grp * ${cin}u + bcol / ${kSpace}u;
${s.kernel.map((size, d) =>
      `  let kk${d} = (bcol / ${suffixStrides(s.kernel)[d] ?? 1}u) % ${size}u;`).join("\n")}`,
    // The (batch, output position) side of `kk` is split once per K-tile and carried.
    prepBTile: `    var bn = 0u;
${s.outDims.map((_, d) => `    var o${d} = 0u;`).join("\n")}
${splitDigits(["bn", ...s.outDims.map((_, d) => `o${d}`)], [s.N, ...s.outDims], `t * ${tileDepth(tileShape(cout, cols).TM)}u + bkk0`, "    ")}`,
    stepB: carryDigits(["bn", ...s.outDims.map((_, d) => `o${d}`)], [s.N, ...s.outDims], "bstep", "      "),
    loadB: `${c.coords}
          if (${c.guard}) {
            v = X[(bn * ${s.C}u + ch) * ${inSpace}u + ${c.offset}];
          }`,
    // Split, the partial sums are written into a cell per piece. Unsplit there is only
    // one such cell, so it is the result as it stands — the caller decides whether to
    // attach a summing stage.
    emit: `  Out[part * ${s.O * cols}u + (grp * ${cout}u + f) * ${cols}u + col] = v;`,
  });
}

/**
 * The weights turned for the input gradient: `(O, C, k…)` → `(C, O, k…)` with every
 * kernel axis reversed. Reversing the flattened kernel index reverses every axis at
 * once, so this is one permutation whatever the rank.
 *
 * **Why it exists.** With stride and dilation 1, the gradient to the input is itself a
 * convolution — of the output gradient with these turned weights, padded `k − 1 − pad`.
 * The dedicated input-gradient GEMM below has to test, per element, which output
 * positions reach an input position (the stride divisibility, the range), and measured
 * on the M4 Max it ran two to three times slower than the forward at equal FLOPs — 1.0
 * to 1.2 ms per U-Net layer against 0.3 to 0.75 for the forward. Turning the weights is
 * a few thousand elements; the forward's gather then does the rest at the forward's
 * speed. Strides above 1 and grouped convolutions keep the dedicated kernel.
 */
export function turnWeightsForGradInput(O: number, C: number, kSpace: number): string {
  const n = O * C * kSpace;
  return `
@group(0) @binding(0) var<storage, read> W: array<f32>;
@group(0) @binding(1) var<storage, read_write> Out: array<f32>;
@compute @workgroup_size(${WORKGROUP})
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
${flatId(n)}
  let k = gid % ${kSpace}u;
  let oc = gid / ${kSpace}u;
  let o = oc / ${C}u;
  let c = oc % ${C}u;
  Out[(c * ${O}u + o) * ${kSpace}u + (${kSpace - 1}u - k)] = W[gid];
}`;
}

/**
 * The gradient to the input — the tiled version.
 *
 * `dX[(n,i), c] = Σ_{o,k} G[(n, output position), o] · W[o, c, k]`, with the summed pair
 * `(o, k)` as the inner axis. **With a stride above 1 there are positions where the
 * division does not come out even, and nothing arrives at those** — that test lives
 * inside the right-hand tile.
 */
export function convNDGradInputTiled(s: ConvNDShape): string {
  const outSpace = s.outDims.reduce((a, b) => a * b, 1);
  const kSpace = s.kernel.reduce((a, b) => a * b, 1);
  const inSpace = s.inDims.reduce((a, b) => a * b, 1);
  const inStride = suffixStrides(s.inDims);
  const outStride = suffixStrides(s.outDims);
  const kStride = suffixStrides(s.kernel);
  const { groups, cin, cout } = grouped(s);
  const K = cout * kSpace;
  const cols = s.N * inSpace;
  return tiledGemm({
    M: cin, N: cols, K, groups,
    bindings: `@group(0) @binding(0) var<storage, read> G: array<f32>;
@group(0) @binding(1) var<storage, read> Wt: array<f32>;
@group(0) @binding(2) var<storage, read_write> Out: array<f32>;`,
    // Left: the weights seen as (input channel × (output channel, kernel position)).
    loadA: `          let oc = kk / ${kSpace}u;
          let kp = kk % ${kSpace}u;
          v = Wt[((grp * ${cout}u + oc) * ${cin}u + arow) * ${kSpace}u + kp];`,
    // Right: it finds the output positions that **reach** this input position. Nothing
    // reaching means 0.
    prepB: `  let pb_n = (bcol / ${inSpace}u) * ${s.O * outSpace}u;
${s.inDims.map((size, d) =>
      `  let pb_i${d} = i32((bcol / ${inStride[d] ?? 1}u) % ${size}u) + ${s.pad[d] ?? 0};`).join("\n")}`,
    loadB: `          let oc = kk / ${kSpace}u;
${s.kernel.map((size, d) =>
      `          let kk${d} = (kk / ${kStride[d] ?? 1}u) % ${size}u;`).join("\n")}
          var ok = true;
          var off = 0u;
${s.inDims.map((_, d) => {
      const st = s.stride[d] ?? 1;
      return `          {
            let t${d} = pb_i${d} - i32(kk${d} * ${convDil(s, d)}u);
            if (t${d} < 0 || t${d} % ${st} != 0) { ok = false; }
            else {
              let o${d} = t${d} / ${st};
              if (o${d} >= ${s.outDims[d] ?? 0}) { ok = false; }
              else { off = off + u32(o${d}) * ${outStride[d] ?? 1}u; }
            }
          }`;
    }).join("\n")}
          if (ok) {
            v = G[pb_n + (grp * ${cout}u + oc) * ${outSpace}u + off];
          }`,
    emit: `  Out[col / ${inSpace}u * ${s.C * inSpace}u + (grp * ${cin}u + f) * ${inSpace}u + col % ${inSpace}u] = v;`,
  });
}

/**
 * How many pieces to split the weight gradient's reduced axis into.
 *
 * **This GEMM has a small output and a large reduction.** In one layer the output is
 * `(64, 27)` while the reduction is `batch × 32 × 32 = 16,384`, so the tile grid falls to
 * **a single workgroup** — one piece of work for the whole GPU. The reduction is split
 * across several workgroups and summed at the end.
 *
 * The piece count only rises when the grid is too small. Splitting attaches a partial-sum
 * buffer and a summing stage, so on a layer whose grid is already ample it is a loss.
 */
export function convGradWeightSplit(s: ConvNDShape): number {
  const { groups, cin, cout } = grouped(s);
  const cols = cin * s.kernel.reduce((a, b) => a * b, 1);
  // Every group is a tile grid of its own, so the groups count towards keeping the GPU
  // busy before any split does.
  const tiles = tileCount(cout, cols) * groups;
  const K = s.N * s.outDims.reduce((a, b) => a * b, 1);
  // Below this many workgroups the GPU is said to be idle. Above it, no split.
  const WANT = 256;
  if (tiles >= WANT) return 1;
  // A piece has to take at least this much reduction for the split to be worth it.
  const MIN_PER_SPLIT = 256;
  return Math.max(1, Math.min(Math.ceil(WANT / tiles), Math.floor(K / MIN_PER_SPLIT)));
}

/** The weight gradient's grid — rows are output channels, columns are (input channel,
 *  kernel position), and depth is the piece. */
export function convGradWeightGrid(s: ConvNDShape): [number, number, number] {
  const { groups, cin, cout } = grouped(s);
  const cols = cin * s.kernel.reduce((a, b) => a * b, 1);
  const { TM, TN } = tileShape(cout, cols);
  return [Math.ceil(cols / TN), Math.ceil(cout / TM), groups * convGradWeightSplit(s)];
}

/** Sums the split partials. The order is fixed, so two runs give the same value. */
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

/** The input gradient's grid — rows are input channels, columns are batch and input
 *  position. */
export function convGradInputGrid(s: ConvNDShape): [number, number, number] {
  const cols = s.N * s.inDims.reduce((a, b) => a * b, 1);
  const { groups, cin } = grouped(s);
  const { TM, TN } = tileShape(cin, cols);
  return [Math.ceil(cols / TN), Math.ceil(cin / TM), groups];
}


/**
 * Pooling with no regard for the number of dimensions. The channels fold into the batch.
 *
 * `max` sends to **one** winning position — the earlier one on a tie. That differs from
 * `amax` dividing evenly, and it is what torch's pooling does.
 */
export interface PoolNDShape {
  readonly NC: number;
  readonly inDims: readonly number[];
  readonly kernel: readonly number[];
  readonly stride: readonly number[];
  readonly outDims: readonly number[];
  /** Per axis, and **only average pooling honours it** — the maximum's backward
   *  reads the input at window positions and a padded window has none. */
  readonly pad?: readonly number[];
  /** `torch`'s `count_include_pad`. The divisor counts the padded cells too. */
  readonly countIncludePad?: boolean;
  /** `torch`'s `divisor_override`. A fixed divisor, ignoring both counts. */
  readonly divisorOverride?: number | null;
  /**
   * Per axis, the gap between the cells one window reads. **Only the maximum takes
   * it** — torch's `avg_pool*d` has no such argument at all, so offering one on the
   * average would be a seat torch does not have.
   */
  readonly dilation?: readonly number[];
}

export function poolNDKey(p: PoolNDShape): string {
  // **The divisor settings belong in the key.** They change the generated source, and
  // a cache keyed without them hands back a pipeline compiled for the other divisor —
  // which is not a crash but a wrong number, from the second call onward.
  //
  // **`outDims` was not in the key and had to be.** It is derived from the other four
  // only while the rounding is fixed; `ceilMode` makes the same extents, kernel and
  // stride give two different output sizes, and the key could not tell them apart.
  // Nothing had gone wrong yet because nothing could ask for the ceiling — the field
  // was safe to leave out for exactly as long as the argument was missing.
  return [p.NC, p.inDims, p.kernel, p.stride, p.outDims,
          p.pad ?? [], p.countIncludePad !== false, p.divisorOverride ?? "-",
          p.dilation ?? []].join("|");
}

export function poolNDForward(p: PoolNDShape, kind: "max" | "avg"): string {
  if (kind === "avg") return avgNDForward(p);
  const inSpace = p.inDims.reduce((a, b) => a * b, 1);
  const outSpace = p.outDims.reduce((a, b) => a * b, 1);
  const inStride = suffixStrides(p.inDims);
  const outStride = suffixStrides(p.outDims);
  const pad = p.pad ?? p.kernel.map(() => 0);
  const n = p.NC * outSpace;
  const decode = p.outDims.map((_, d) =>
    `  let o${d} = i32((r / ${outStride[d] ?? 1}u) % ${p.outDims[d] ?? 1}u);`).join("\n");
  const open: string[] = [];
  const close: string[] = [];
  const guards: string[] = [];
  const terms: string[] = [];
  const dil = p.dilation ?? p.kernel.map(() => 1);
  for (const [d, size] of p.kernel.entries()) {
    const st = p.stride[d] ?? 1;
    const pd = pad[d] ?? 0;
    const dim = p.inDims[d] ?? 1;
    const dl = dil[d] ?? 1;
    open.push(`    for (var k${d} = 0; k${d} < ${size}; k${d} = k${d} + 1) {`);
    close.push("    }");
    // **The same guard the average has had all along.** The window is laid out in
    // padded coordinates and the buffer holds the real ones, so the padding comes off
    // before the read and anything outside is not read at all.
    //
    // **`dilation` is the step between the cells one window reads**, so it multiplies
    // the window index and nothing else. At 1 this is the line it has always been.
    guards.push(`      let a${d} = o${d} * ${st} + k${d} * ${dl} - ${pd};`);
    guards.push(`      if (a${d} < 0 || a${d} >= ${dim}) { continue; }`);
    terms.push(`u32(a${d}) * ${inStride[d] ?? 1}u`);
  }
  // **It starts below every real value rather than at the window's first cell.** With
  // padding that first cell can be outside the input, and reading it was the whole
  // reason this kernel refused the argument. torch's own answer is the same: a padded
  // position is −infinity and never wins. It cannot win everywhere either — torch
  // requires the padding to be at most half the kernel, so no window is all padding.
  //
  // **One digit lower than f32's maximum**, and the reduction table above says why:
  // `3.4028235e38` is the decimal rounding and sits *above* the true maximum, so WGSL
  // discards the whole shader as "cannot be represented as f32". Written with the
  // rounded value first, every max pool came back zero — a dispatch that never ran
  // looks exactly like a buffer nobody wrote.
  return `
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> Out: array<f32>;
@compute @workgroup_size(${WORKGROUP})
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
${flatId(n)}
  let plane = gid / ${outSpace}u;
  let r = gid % ${outSpace}u;
${decode}
  var acc = -3.4028234e38;
${open.join("\n")}
${guards.join("\n")}
      let v = X[plane * ${inSpace}u + ${terms.join(" + ")}];
      acc = max(acc, v);
${close.join("\n")}
  Out[gid] = acc;
}`;
}

/**
 * Average pooling's own source, because **the divisor is not a constant.**
 *
 * The maximum takes every cell in the window and reduces them; the average takes
 * every cell and then divides, and torch's divisor is three different things:
 *
 *   count_include_pad = true    the cells inside the *padded* extent
 *   count_include_pad = false   the cells inside the *real* input
 *   divisor_override = n        n, whatever the window covers
 *
 * The first two are not the kernel volume whenever a window hangs off an edge, which
 * is what `padding` and `ceilMode` both arrange. Each axis contributes its own count
 * and they multiply, so the whole thing is a handful of `min`/`max` per output cell
 * rather than a second pass.
 *
 * **A window may reach past the padded extent and never past it on the left**: torch
 * requires the last window to *start* inside, so the low end needs no clamp against
 * the padded extent — only against the real input, which is what `count_include_pad`
 * asks about.
 */
function avgNDForward(p: PoolNDShape): string {
  const inSpace = p.inDims.reduce((a, b) => a * b, 1);
  const outSpace = p.outDims.reduce((a, b) => a * b, 1);
  const inStride = suffixStrides(p.inDims);
  const outStride = suffixStrides(p.outDims);
  const pad = p.pad ?? p.kernel.map(() => 0);
  const n = p.NC * outSpace;
  const decode = p.outDims.map((_, d) =>
    `  let o${d} = i32((r / ${outStride[d] ?? 1}u) % ${p.outDims[d] ?? 1}u);`).join("\n");

  const open: string[] = [];
  const close: string[] = [];
  const guards: string[] = [];
  const terms: string[] = [];
  const counts: string[] = [];
  for (const [d, size] of p.kernel.entries()) {
    const st = p.stride[d] ?? 1;
    const pd = pad[d] ?? 0;
    const dim = p.inDims[d] ?? 1;
    open.push(`    for (var k${d} = 0; k${d} < ${size}; k${d} = k${d} + 1) {`);
    close.push("    }");
    // Padded coordinate minus the padding is the real one; outside is not read.
    guards.push(`      let a${d} = o${d} * ${st} + k${d} - ${pd};`);
    guards.push(`      if (a${d} < 0 || a${d} >= ${dim}) { continue; }`);
    terms.push(`u32(a${d}) * ${inStride[d] ?? 1}u`);
    const lo = `(o${d} * ${st})`;
    counts.push(p.countIncludePad === false
      ? `max(min(${lo} + ${size}, ${pd + dim}) - max(${lo}, ${pd}), 0)`
      : `max(min(${lo} + ${size}, ${dim + 2 * pd}) - ${lo}, 0)`);
  }
  const divisor = p.divisorOverride != null
    ? p.divisorOverride.toFixed(1)
    : `f32(${counts.join(" * ")})`;
  return `
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> Out: array<f32>;
@compute @workgroup_size(${WORKGROUP})
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
${flatId(n)}
  let plane = gid / ${outSpace}u;
  let r = gid % ${outSpace}u;
${decode}
  var acc = 0.0;
${open.join("\n")}
${guards.join("\n")}
      acc = acc + X[plane * ${inSpace}u + ${terms.join(" + ")}];
${close.join("\n")}
  Out[gid] = acc / ${divisor};
}`;
}

export function poolNDBackward(p: PoolNDShape, kind: "max" | "avg"): string {
  const inSpace = p.inDims.reduce((a, b) => a * b, 1);
  const outSpace = p.outDims.reduce((a, b) => a * b, 1);
  const inStride = suffixStrides(p.inDims);
  const outStride = suffixStrides(p.outDims);
  const n = p.NC * inSpace;
  const pad = p.pad ?? p.kernel.map(() => 0);
  const decode = p.inDims.map((_, d) =>
    `  let i${d} = i32((r / ${inStride[d] ?? 1}u) % ${p.inDims[d] ?? 1}u) + ${pad[d] ?? 0};`)
    .join("\n");
  const open: string[] = [];
  const close: string[] = [];
  const oTerms: string[] = [];
  const counts: string[] = [];
  const dil = p.dilation ?? p.kernel.map(() => 1);
  for (const [d, size] of p.outDims.entries()) {
    const st = p.stride[d] ?? 1;
    const pd = pad[d] ?? 0;
    const dim = p.inDims[d] ?? 1;
    const ks = p.kernel[d] ?? 1;
    const dl = dil[d] ?? 1;
    open.push(`    for (var o${d} = 0; o${d} < ${size}; o${d} = o${d} + 1) {`);
    open.push(`      let d${d} = i${d} - o${d} * ${st};`);
    // **A dilated window skips cells, so "inside the window" is not "inside the
    // extent".** The offset has to be a multiple of the dilation as well as under the
    // reach; at 1 both conditions collapse to the `d < kernel` this always had.
    open.push(`      if (d${d} < 0 || d${d} > ${(ks - 1) * dl}`
              + `${dl === 1 ? "" : ` || d${d} % ${dl} != 0`}) { continue; }`);
    close.push("    }");
    oTerms.push(`u32(o${d}) * ${outStride[d] ?? 1}u`);
    const lo = `(o${d} * ${st})`;
    counts.push(p.countIncludePad === false
      ? `max(min(${lo} + ${ks}, ${pd + dim}) - max(${lo}, ${pd}), 0)`
      : `max(min(${lo} + ${ks}, ${dim + 2 * pd}) - ${lo}, 0)`);
  }
  // **The same divisor as the forward, and it has to be recomputed here.** The
  // gradient a cell receives from a window is the window's own `1/divisor`, so an
  // edge window that divided by fewer cells hands back proportionally more. Dividing
  // by the kernel volume everywhere is right only when nothing hangs off an edge.
  const divisor = p.divisorOverride != null
    ? p.divisorOverride.toFixed(1)
    : `f32(${counts.join(" * ")})`;
  // **The window is scanned in padded coordinates and read in real ones.** Every read
  // below takes the padding off first and skips what falls outside — the same guard
  // the forward has. Written without it, `o·stride + m` was used straight as a memory
  // offset, which is right only while the padding is zero and is why the maximum
  // refused the argument rather than answering with it.
  const kOpen: string[] = [];
  const kClose: string[] = [];
  const kGuard: string[] = [];
  const kTerms: string[] = [];
  const kOrder: string[] = [];
  const mineTerms: string[] = [];
  const mineOrder: string[] = [];
  for (const [d, size] of p.kernel.entries()) {
    const st = p.stride[d] ?? 1;
    const pd = pad[d] ?? 0;
    const dim = p.inDims[d] ?? 1;
    const dl = dil[d] ?? 1;
    kOpen.push(`        for (var m${d} = 0; m${d} < ${size}; m${d} = m${d} + 1) {`);
    kClose.push("        }");
    kGuard.push(`          let b${d} = o${d} * ${st} + m${d} * ${dl} - ${pd};`);
    kGuard.push(`          if (b${d} < 0 || b${d} >= ${dim}) { continue; }`);
    kTerms.push(`u32(b${d}) * ${inStride[d] ?? 1}u`);
    // The tie-break runs over the window's own positions, so its ordering stays in
    // window coordinates while the reads are in real ones. **Both sides of the
    // comparison are offsets in padded coordinates** — `m·dilation` against `d` —
    // because `d` is already the padded difference and `m` is not.
    kOrder.push(`u32(m${d} * ${dl}) * ${inStride[d] ?? 1}u`);
    mineTerms.push(`u32(i${d} - ${pd}) * ${inStride[d] ?? 1}u`);
    mineOrder.push(`u32(d${d}) * ${inStride[d] ?? 1}u`);
  }
  const body = kind === "avg"
    ? `      acc = acc + G[plane * ${outSpace}u + ${oTerms.join(" + ")}] / ${divisor};`
    : `      {
        let mineAt = plane * ${inSpace}u + ${mineTerms.join(" + ")};
        var best = X[mineAt];
${kOpen.join("\n")}
${kGuard.join("\n")}
          let v = X[plane * ${inSpace}u + ${kTerms.join(" + ")}];
          if (v > best) { best = v; }
${kClose.join("\n")}
        // On a tie **the earlier position** wins. An equal value earlier beats it.
        var earlier = false;
${kOpen.join("\n")}
${kGuard.join("\n")}
          let idx = ${kOrder.join(" + ")};
          let mine = ${mineOrder.join(" + ")};
          if (idx < mine && X[plane * ${inSpace}u + ${kTerms.join(" + ")}] == best) {
            earlier = true;
          }
${kClose.join("\n")}
        if (X[mineAt] == best && !earlier) {
          acc = acc + G[plane * ${outSpace}u + ${oTerms.join(" + ")}];
        }
      }`;
  // **The average does not look at the input.** And merely declaring `X` makes
  // `layout: "auto"` drop the unused binding, so a caller passing three buffers is
  // refused with "binding index 0 not present". That refusal is not an exception but
  // **an invalid command buffer**, so the whole backward does not run and training alone
  // quietly stops. The ResNet bench really did produce numbers in that state.
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

/** How many buffers a pooling backward takes. It differs per kind, so the caller reads
 *  it from here. */
export function poolNDBackwardNeedsInput(kind: "max" | "avg"): boolean {
  return kind === "max";
}

/**
 * The window list — per axis, **the input positions each output cell reads.**
 *
 * It exists to hand the fixed and the adaptive forms across in one shape. The fixed one
 * is `start = o·stride` at a constant length; the adaptive one runs from
 * `floor(o·n/want)` to `ceil((o+1)·n/want)`, so the length differs per position.
 *
 * **It used to be `[start, end)` per cell**, and that shape could say neither of the two
 * things `max_pool(…, return_indices=True)` needs: a dilated window skips cells, so it
 * is not an interval, and a padded window hangs off the edge, so some of its positions
 * are not positions at all. Written as a list, both are the same thing — the positions
 * that exist — and the padded ones are simply absent from it. That is what the
 * refusal on those three arguments was about, and it was about this type.
 */
export interface PoolWindows {
  readonly NC: number;
  readonly inDims: readonly number[];
  /** Per axis, per output cell, the input positions read. */
  readonly axes: readonly (readonly (readonly number[])[])[];
}

export function poolWindowsKey(p: PoolWindows): string {
  return [p.NC, p.inDims, p.axes.map((a) => a.map((w) => w.join(":")).join(","))]
    .join("|");
}

/**
 * Produces the maximum and **the winning position** together.
 *
 * The position follows torch's convention: **a flat index within the plane** — counted
 * from 0 again per batch and channel. `maxUnpool` sends that index straight back.
 *
 * **The value comes out here with it.** Computing the value in another kernel allows a
 * state of "position A but value B", and both are plausible, so no eye catches it.
 *
 * On a tie **the earlier position** wins — as in torch. The window is walked in order of
 * increasing flat index and the comparison is `>`, so the first maximum stays.
 */
export function poolMaxWithIndex(p: PoolWindows): string {
  const inSpace = p.inDims.reduce((a, b) => a * b, 1);
  const outDims = p.axes.map((a) => a.length);
  const outSpace = outDims.reduce((a, b) => a * b, 1);
  const inStride = suffixStrides(p.inDims);
  const outStride = suffixStrides(outDims);
  const n = p.NC * outSpace;

  // The window table is baked into the shader as constants. The output cell count is
  // small so it is cheap, and every form — fixed, adaptive, dilated, padded — rides in
  // the same shape.
  //
  // **The positions are one flat run per axis with an offset table**, rather than a
  // start and an end. An interval cannot hold a dilated window (it skips cells) or a
  // padded one (some of its positions do not exist), and the flat run holds both:
  // `S[o]` to `S[o+1]` names this cell's slice of `P`.
  const tables = p.axes.map((axis, d) => {
    const flat: number[] = [];
    const offs: number[] = [0];
    for (const cell of axis) {
      flat.push(...cell);
      offs.push(flat.length);
    }
    return `var<private> P${d}: array<u32, ${Math.max(1, flat.length)}> = `
      + `array<u32, ${Math.max(1, flat.length)}>(`
      + `${(flat.length ? flat : [0]).map((v) => `${v}u`).join(", ")});\n`
      + `var<private> S${d}: array<u32, ${offs.length}> = `
      + `array<u32, ${offs.length}>(${offs.map((v) => `${v}u`).join(", ")});`;
  }).join("\n");

  const decode = outDims.map((size, d) =>
    `  let o${d} = (r / ${outStride[d] ?? 1}u) % ${size}u;`).join("\n");
  const open: string[] = [];
  const close: string[] = [];
  const terms: string[] = [];
  for (let d = 0; d < p.axes.length; d++) {
    open.push(`  for (var t${d} = S${d}[o${d}]; t${d} < S${d}[o${d} + 1u]; `
              + `t${d} = t${d} + 1u) {`);
    open.push(`    let k${d} = P${d}[t${d}];`);
    close.push("  }");
    terms.push(`k${d} * ${inStride[d] ?? 1}u`);
  }
  // The seed is the window's own first position, which exists because torch never
  // produces an empty window: the padding is at most half the reach.
  const first = p.axes.map((_, d) => `P${d}[S${d}[o${d}]] * ${inStride[d] ?? 1}u`)
    .join(" + ");

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
 * Sends the gradient back along the position table. It goes to the winning positions
 * only.
 *
 * The forward already fixed the positions, so nothing is chosen again here — choosing
 * again could diverge from the forward's choice, and with a tie present that is exactly
 * what happens.
 *
 * Each input position **finds the output cells pointing at it and adds.** It is a gather
 * rather than a scatter, so no two threads write the same cell — with overlapping windows
 * one input can win for several outputs.
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
 * Places values at the cells the position table points at, 0 elsewhere — this is
 * `MaxUnpool`.
 *
 * **Written as a gather.** Scattering lets thread order decide the answer at overlapping
 * positions, and that answer can differ from run to run, which cannot be compared. Having
 * each output cell find the inputs pointing at it fixes the order — with several, **the
 * last** survives, which is the same answer as torch's scatter.
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

/** `MaxUnpool`'s backward — it takes back from wherever the value went. The reverse of
 *  the filling. */
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
 * Nearest-neighbour upsampling. Each output position reads the input position that bore
 * it.
 *
 * The backward gathers the outputs that read it, and with an integer scale that count is
 * constant.
 */
/**
 * **It takes the output extents rather than a scale**, which is what lets a target
 * size that is not a whole multiple work at all. torch's rule is
 * `src = floor(o · in / out)`, and integer division in WGSL is that floor — with a
 * whole multiple it reduces to the division by the scale this used to do, so the
 * generated source for every existing call is unchanged in meaning.
 *
 * The scale-taking form refused a non-integer factor rather than approximating, which
 * was the right refusal while there was nothing behind it. `UpsamplingNearest2d(size=)`
 * is what put something behind it: torch answers there and so does the core.
 */
export function upsampleNearest(
  NC: number,
  inDims: readonly number[],
  outDims: readonly number[],
): string {
  const inSpace = inDims.reduce((a, b) => a * b, 1);
  const outSpace = outDims.reduce((a, b) => a * b, 1);
  const inStride = suffixStrides(inDims);
  const outStride = suffixStrides(outDims);
  const n = NC * outSpace;
  const terms = inDims.map((_, d) =>
    `((r / ${outStride[d] ?? 1}u) % ${outDims[d] ?? 1}u * ${inDims[d] ?? 1}u`
    + ` / ${outDims[d] ?? 1}u) * ${inStride[d] ?? 1}u`);
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

/** Upsampling's backward — it gathers the positions that read it. */
export function upsampleNearestBackward(
  NC: number,
  inDims: readonly number[],
  outDims: readonly number[],
): string {
  const inSpace = inDims.reduce((a, b) => a * b, 1);
  const outSpace = outDims.reduce((a, b) => a * b, 1);
  const inStride = suffixStrides(inDims);
  const outStride = suffixStrides(outDims);
  const n = NC * inSpace;
  const decode = inDims.map((_, d) =>
    `  let i${d} = (r / ${inStride[d] ?? 1}u) % ${inDims[d] ?? 1}u;`).join("\n");
  const open: string[] = [];
  const close: string[] = [];
  const terms: string[] = [];
  // **The window is not a fixed width any more.** With a whole multiple every input
  // cell is read by exactly `scale` outputs; at 2 → 5 the counts are 2 and 3, so the
  // loop runs the whole output axis and keeps the positions whose source is this cell.
  // Costlier per element and the only form that is right for both.
  for (const [d, size] of outDims.entries()) {
    open.push(`    for (var s${d} = 0u; s${d} < ${size}u; s${d} = s${d} + 1u) {`);
    open.push(`      if (s${d} * ${inDims[d] ?? 1}u / ${size}u != i${d}) { continue; }`);
    close.push("    }");
    terms.push(`s${d} * ${outStride[d] ?? 1}u`);
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
 * Sorts one axis. **The positions move with the values** — moving values alone cannot
 * build `argsort`, nor send the gradient back to where it came from.
 *
 * It is an insertion sort. That is bad for a long axis, and what is pushed through here
 * is an axis length in the tens, and it is a **stable sort**, so ties keep torch's order —
 * a bitonic sort does not preserve that. One thread takes a whole axis, so there are no
 * atomics either.
 */
export function sortAxis(
  outer: number,
  len: number,
  inner: number,
  descending: boolean,
): string {
  const n = outer * inner;
  // Stability requires **stopping at an equal value.** Hence the strict comparison.
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
 * Sends the gradient back to its original position along the position table.
 *
 * The backward of `sort`, `topk` and `median` is all this — it flows to the gathered
 * positions and 0 elsewhere. Returning the values detached sends no gradient there, and
 * training quietly stops.
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
 * The **overwriting** version of `scatterByIndex`. On repeated indices the last write
 * survives.
 *
 * The difference between accumulating and overwriting is the whole difference between
 * `scatter_add` and `scatter` — with non-repeating indices the two give the same answer,
 * so only repeated indices separate them.
 *
 * **It reads from the output side.** Writing from the input side has several threads
 * arriving at one cell with no fixed notion of which is last — here each output cell
 * walks what comes to it, which fixes the order.
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
 * The cumulative maximum and minimum. It produces the value and the position together.
 *
 * **A tie gives the later position** — as torch's `cummax` does (the opposite of `argmax`
 * giving the earlier one). One comparison including equality is the whole difference.
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

/** The threads one piece of a channel's statistics are spread over. */
export const BN_GROUP = 256;

/**
 * How many workgroups share one channel's reduction. **Sixteen channels were sixteen
 * workgroups**: the second version of these kernels gave a channel a workgroup, and on a
 * layer of sixteen channels at 96 × 96 that is 4,096 threads reading 2.4 million floats
 * — a memory-bound pass with a fortieth of the GPU awake, 0.8 ms forward and 1.2 ms
 * backward per U-Net step where torch spends 1.6 ms on all of BatchNorm (timestamps,
 * 2026-09-06). Now a channel is cut into pieces of at most 16,384 elements, each piece a
 * workgroup, and a pass over the pieces finishes the statistics. Every sum is in a fixed
 * order, so the value is the same on every run.
 */
export function bnPieces(N: number, S: number): number {
  return Math.max(1, Math.min(64, Math.ceil((N * S) / 16384)));
}

/**
 * BatchNorm's per-channel statistics — the sum and the sum of squares **in one pass.**
 *
 * Assembled, it is three `sumDim` plus `sub` plus `square` plus three `sumDim` plus a few
 * divisions, a dozen or so dispatches, twenty times over per layer. Measured, most of the
 * 1,636 dispatches in one ResNet step came from here.
 *
 * Workgroup `(channel, piece)` walks its slice of the channel with 256 threads striding
 * and folding as a tree, and writes two partial sums. The first version gave one *thread*
 * a channel (6.3 ms per layer, measured); the second a workgroup — see `bnPieces` for why
 * that was still too few.
 */
export function batchNormStats(N: number, C: number, S: number): string {
  const pieces = bnPieces(N, S);
  const per = Math.ceil((N * S) / pieces);
  return `
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> PartSum: array<f32>;
@group(0) @binding(2) var<storage, read_write> PartSq: array<f32>;
var<workgroup> pt: array<f32, ${BN_GROUP}>;
var<workgroup> pq: array<f32, ${BN_GROUP}>;
@compute @workgroup_size(${BN_GROUP})
fn main(@builtin(local_invocation_id) l: vec3<u32>, @builtin(workgroup_id) w: vec3<u32>) {
  let c = w.x;
  let piece = w.y;
  let lo = piece * ${per}u;
  let hi = min(lo + ${per}u, ${N * S}u);
  var total = 0.0;
  var sq = 0.0;
  for (var i = lo + l.x; i < hi; i = i + ${BN_GROUP}u) {
    let n = i / ${S}u;
    let v = X[(n * ${C}u + c) * ${S}u + (i - n * ${S}u)];
    total = total + v;
    sq = fma(v, v, sq);
  }
  pt[l.x] = total;
  pq[l.x] = sq;
  workgroupBarrier();
  var span = ${BN_GROUP / 2}u;
  loop {
    if (span == 0u) { break; }
    if (l.x < span) { pt[l.x] = pt[l.x] + pt[l.x + span]; pq[l.x] = pq[l.x] + pq[l.x + span]; }
    workgroupBarrier();
    span = span / 2u;
  }
  if (l.x == 0u) {
    PartSum[c * ${pieces}u + piece] = pt[0];
    PartSq[c * ${pieces}u + piece] = pq[0];
  }
}`;
}

/** Adds a channel's pieces and finishes the mean and the (biased) variance. */
export function batchNormFinish(N: number, C: number, S: number, eps: number): string {
  const pieces = bnPieces(N, S);
  return `
@group(0) @binding(0) var<storage, read> PartSum: array<f32>;
@group(0) @binding(1) var<storage, read> PartSq: array<f32>;
@group(0) @binding(2) var<storage, read_write> Mean: array<f32>;
@group(0) @binding(3) var<storage, read_write> Var: array<f32>;
@group(0) @binding(4) var<storage, read_write> InvStd: array<f32>;
@compute @workgroup_size(${WORKGROUP})
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
${flatId(C)}
  var total = 0.0;
  var sq = 0.0;
  for (var p = 0u; p < ${pieces}u; p = p + 1u) {
    total = total + PartSum[gid * ${pieces}u + p];
    sq = sq + PartSq[gid * ${pieces}u + p];
  }
  let m = total / ${(N * S).toFixed(1)};
  Mean[gid] = m;
  // **This is the biased estimate** (divided by n) — what torch's BatchNorm uses for
  // the normalisation, and a different number from the unbiased one that goes into the
  // running statistics. Merged into one, they diverge in evaluation mode alone.
  let v = sq / ${(N * S).toFixed(1)} - m * m;
  Var[gid] = v;
  // The backward's 1/σ as well — it was an add and an rsqrt over C elements after this
  // pass, two dispatches a layer for sixteen numbers (a captured step showed twenty).
  InvStd[gid] = inverseSqrt(v + ${f32lit(eps)});
}`;
}

/**
 * Takes the statistics, normalises, and applies the scale and shift in one pass.
 *
 * With `withXhat`, the standardised value goes out as a second output in the same pass —
 * the training backward needs it, and building it afterwards was two more full passes
 * over the activation (a broadcast subtract and a broadcast multiply).
 *
 * With `relu`, the ReLU that follows the layer is applied here, and the two backward
 * kernels mask the incoming gradient by this output — BatchNorm → ReLU as two kernels
 * forward and two back instead of three and three, each pass over the whole activation.
 * `Sequential` pairs the two layers (measured: the U-Net's ten pairs were 1.2 ms of
 * ReLU passes in a 21 ms step).
 */
export function batchNormApply(N: number, C: number, S: number, eps: number, withXhat = false, relu = false): string {
  const n = N * C * S;
  return `
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> Mean: array<f32>;
@group(0) @binding(2) var<storage, read> Var: array<f32>;
@group(0) @binding(3) var<storage, read> Wt: array<f32>;
@group(0) @binding(4) var<storage, read> B: array<f32>;
@group(0) @binding(5) var<storage, read_write> Out: array<f32>;
${withXhat ? "@group(0) @binding(6) var<storage, read_write> Xh: array<f32>;" : ""}
@compute @workgroup_size(${WORKGROUP})
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
${flatId(n)}
  let c = (gid / ${S}u) % ${C}u;
  let xh = (X[gid] - Mean[c]) * inverseSqrt(Var[c] + ${eps});
  ${relu ? "Out[gid] = max(xh * Wt[c] + B[c], 0.0);" : "Out[gid] = xh * Wt[c] + B[c];"}
${withXhat ? "  Xh[gid] = xh;" : ""}
}`;
}

/**
 * BatchNorm's backward.
 *
 * **The mean and the variance are inside the graph.** Taken outside it, the input
 * gradient comes out wrong and nothing reaches `weight` at all — a place the core lived
 * with for a long time. The expression is
 *
 *     dx = γ·σ⁻¹·(dy − mean(dy) − x̂·mean(dy·x̂))
 *
 * and the two means it needs are counted once per channel, then applied elementwise.
 */
export function batchNormStatsBackward(N: number, C: number, S: number, relu = false): string {
  const pieces = bnPieces(N, S);
  const per = Math.ceil((N * S) / pieces);
  return `
@group(0) @binding(0) var<storage, read> Xh: array<f32>;
@group(0) @binding(1) var<storage, read> G: array<f32>;
@group(0) @binding(2) var<storage, read_write> PartG: array<f32>;
@group(0) @binding(3) var<storage, read_write> PartGXh: array<f32>;
${relu ? "@group(0) @binding(4) var<storage, read> Y: array<f32>;" : ""}
var<workgroup> pg: array<f32, ${BN_GROUP}>;
var<workgroup> px: array<f32, ${BN_GROUP}>;
@compute @workgroup_size(${BN_GROUP})
fn main(@builtin(local_invocation_id) l: vec3<u32>, @builtin(workgroup_id) w: vec3<u32>) {
  let c = w.x;
  let piece = w.y;
  let lo = piece * ${per}u;
  let hi = min(lo + ${per}u, ${N * S}u);
  var sg = 0.0;
  var sgx = 0.0;
  for (var i = lo + l.x; i < hi; i = i + ${BN_GROUP}u) {
    let n = i / ${S}u;
    let at = (n * ${C}u + c) * ${S}u + (i - n * ${S}u);
    ${relu ? "let gv = select(0.0, G[at], Y[at] > 0.0);" : "let gv = G[at];"}
    sg = sg + gv;
    sgx = fma(gv, Xh[at], sgx);
  }
  pg[l.x] = sg;
  px[l.x] = sgx;
  workgroupBarrier();
  var span = ${BN_GROUP / 2}u;
  loop {
    if (span == 0u) { break; }
    if (l.x < span) { pg[l.x] = pg[l.x] + pg[l.x + span]; px[l.x] = px[l.x] + px[l.x + span]; }
    workgroupBarrier();
    span = span / 2u;
  }
  if (l.x == 0u) {
    PartG[c * ${pieces}u + piece] = pg[0];
    PartGXh[c * ${pieces}u + piece] = px[0];
  }
}`;
}

/** Adds the backward's pieces per channel — the two sums the apply kernel and the weight
 *  gradient read. */
export function batchNormFinishBackward(N: number, C: number, S: number): string {
  const pieces = bnPieces(N, S);
  return `
@group(0) @binding(0) var<storage, read> PartG: array<f32>;
@group(0) @binding(1) var<storage, read> PartGXh: array<f32>;
@group(0) @binding(2) var<storage, read_write> SumG: array<f32>;
@group(0) @binding(3) var<storage, read_write> SumGXh: array<f32>;
@compute @workgroup_size(${WORKGROUP})
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
${flatId(C)}
  var sg = 0.0;
  var sgx = 0.0;
  for (var p = 0u; p < ${pieces}u; p = p + 1u) {
    sg = sg + PartG[gid * ${pieces}u + p];
    sgx = sgx + PartGXh[gid * ${pieces}u + p];
  }
  SumG[gid] = sg;
  SumGXh[gid] = sgx;
}`;
}

export function batchNormBackwardApply(
  N: number, C: number, S: number, relu = false,
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
${relu ? "@group(0) @binding(7) var<storage, read> Y: array<f32>;" : ""}
@compute @workgroup_size(${WORKGROUP})
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
${flatId(n)}
  let c = (gid / ${S}u) % ${C}u;
  let xh = Xh[gid];
  ${relu ? "let gv = select(0.0, G[gid], Y[gid] > 0.0);" : "let gv = G[gid];"}
  Out[gid] = Wt[c] * InvStd[c] *
    (gv - SumG[c] / ${count} - xh * SumGXh[c] / ${count});
}`;
}

/**
 * Adam's step counter and bias corrections, **on the GPU.** One thread: the step goes up
 * by one and `Corr` becomes `[1 − β₁ᵗ, 1 − β₂ᵗ]`. The corrections used to be a fresh
 * two-float tensor built on the CPU every step — which a captured step would replay with
 * the values of the step it was recorded at. One dispatch per optimiser step.
 */
export function adamTick(beta1: number, beta2: number): string {
  return `
@group(0) @binding(0) var<storage, read_write> Step: array<f32>;
@group(0) @binding(1) var<storage, read_write> Corr: array<f32>;
@compute @workgroup_size(1)
fn main() {
  let t = Step[0] + 1.0;
  Step[0] = t;
  Corr[0] = 1.0 - pow(${f32lit(beta1)}, t);
  Corr[1] = 1.0 - pow(${f32lit(beta2)}, t);
}`;
}

/**
 * One optimiser step — **it edits the parameters and the state in place.**
 *
 * The assembled version cost four dispatches per parameter (multiply the momentum, add
 * the gradient, multiply the learning rate, subtract). ResNet-18 has sixty-two parameter
 * tensors, so that alone is two hundred and forty, and measured it was over half of the
 * four hundred and seventy elementwise dispatches.
 *
 * **It reads and writes the same position.** One thread sees only its own element, so
 * there is nowhere for the order to mix — possible because this is an elementwise update
 * with no broadcasting and no reduction.
 */
export function sgdStep(
  n: number, lr: number, momentum: number, weightDecay = 0,
  dampening = 0, nesterov = false, maximize = false, first = false,
): string {
  const hasMomentum = momentum !== 0;
  // **Weight decay is added into the gradient.** That is a different number from
  // shrinking the parameter separately — what differs is whether the momentum buffer
  // carries the decay along. torch's SGD is this side.
  const base = maximize ? "-G[gid]" : "G[gid]";
  const grad = weightDecay !== 0
    ? `${base} + P[gid] * ${weightDecay}`
    : base;
  // **The first step is the raw gradient, undamped.** torch seeds the buffer with
  // the gradient itself and only damps from the second step on, so a dampening of
  // 0.9 does not shrink the very first move. Baking `first` into the shader keeps
  // the branch out of the inner loop and out of the pipeline key's way — the key
  // carries it, so the two variants are two pipelines rather than one wrong one.
  const damped = first || dampening === 0 ? "gv" : `gv * ${1 - dampening}`;
  return `
@group(0) @binding(0) var<storage, read_write> P: array<f32>;
@group(0) @binding(1) var<storage, read> G: array<f32>;
${hasMomentum ? "@group(0) @binding(2) var<storage, read_write> Buf: array<f32>;" : ""}
@compute @workgroup_size(${WORKGROUP})
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
${flatId(n)}
  let gv = ${grad};
${hasMomentum
    ? `  let b = Buf[gid] * ${momentum} + ${damped};
  Buf[gid] = b;
  P[gid] = P[gid] - (${nesterov ? `gv + b * ${momentum}` : "b"}) * ${lr};`
    : `  P[gid] = P[gid] - gv * ${lr};`}
}`;
}

/** One Adam step. The bias correction arrives by step count rather than being baked —
 *  it differs every step.
 *
 *  `amsgrad` adds a fifth buffer holding the **running maximum of the second moment**,
 *  and divides by that instead of by the current one. The max is taken over the raw
 *  `v`, before the bias correction, which is where torch takes it — correcting first
 *  and then maximising is a different number, and both read plausible.
 */
export function adamStep(
  n: number, lr: number, beta1: number, beta2: number, eps: number,
  amsgrad = false,
): string {
  return `
@group(0) @binding(0) var<storage, read_write> P: array<f32>;
@group(0) @binding(1) var<storage, read> G: array<f32>;
@group(0) @binding(2) var<storage, read_write> M: array<f32>;
@group(0) @binding(3) var<storage, read_write> V: array<f32>;
@group(0) @binding(4) var<storage, read> Corr: array<f32>;
${amsgrad ? "@group(0) @binding(5) var<storage, read_write> Vmax: array<f32>;" : ""}
@compute @workgroup_size(${WORKGROUP})
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
${flatId(n)}
  let gv = G[gid];
  let m = M[gid] * ${beta1} + gv * ${1 - beta1};
  let v = V[gid] * ${beta2} + gv * gv * ${1 - beta2};
  M[gid] = m;
  V[gid] = v;
${amsgrad ? `  let vd = max(Vmax[gid], v);
  Vmax[gid] = vd;` : "  let vd = v;"}
  // Corr[0] = 1-β₁ᵗ, Corr[1] = 1-β₂ᵗ. They differ every step, so they arrive rather
  // than being baked.
  P[gid] = P[gid] - ${lr} * (m / Corr[0]) / (sqrt(vd / Corr[1]) + ${eps});
}`;
}

/**
 * One RMSprop step.
 *
 * `centered` subtracts the squared running *mean* of the gradient from the running mean
 * of the square — an estimate of the variance rather than of the second moment — and
 * `momentum` puts the normalised step through a buffer before it reaches the weight.
 * Both are torch's, and each adds one buffer; the bindings are numbered in the order
 * the caller pushes them, so **the two flags are part of the pipeline key**. Left out,
 * the cache would hand a centred optimizer the shader compiled for an uncentred one and
 * bind its `GradAvg` to nothing.
 *
 * `eps` goes **outside** the square root, as torch has it. Inside, it would act as a
 * floor on the variance instead of on the divisor, and the difference only shows where
 * the gradient is small — which is exactly where it matters.
 */
export function rmspropStep(
  n: number, lr: number, alpha: number, eps: number,
  momentum = 0, centered = false,
): string {
  let slot = 3;
  const gradAvg = centered ? slot++ : -1;
  const buf = momentum !== 0 ? slot++ : -1;
  return `
@group(0) @binding(0) var<storage, read_write> P: array<f32>;
@group(0) @binding(1) var<storage, read> G: array<f32>;
@group(0) @binding(2) var<storage, read_write> S: array<f32>;
${centered ? `@group(0) @binding(${gradAvg}) var<storage, read_write> A: array<f32>;` : ""}
${momentum !== 0 ? `@group(0) @binding(${buf}) var<storage, read_write> B: array<f32>;` : ""}
@compute @workgroup_size(${WORKGROUP})
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
${flatId(n)}
  let gv = G[gid];
  let s = S[gid] * ${alpha} + gv * gv * ${1 - alpha};
  S[gid] = s;
${centered ? `  let a = A[gid] * ${alpha} + gv * ${1 - alpha};
  A[gid] = a;
  let avg = s - a * a;` : "  let avg = s;"}
  let step = gv / (sqrt(avg) + ${eps});
${momentum !== 0 ? `  let b = B[gid] * ${momentum} + step;
  B[gid] = b;
  P[gid] = P[gid] - ${lr} * b;` : `  P[gid] = P[gid] - ${lr} * step;`}
}`;
}

/**
 * The running-statistics update — `running ← (1−t)·running + t·new`, both at once.
 *
 * The assembled version was eight dispatches per BatchNorm across twenty layers. It is a
 * small job running over the channel count alone, so one kernel is enough.
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
  // **The running statistics take the unbiased estimate** — a different number from the
  // biased one used for the normalisation.
  RunVar[gid] = RunVar[gid] * ${1 - momentum} + Var[gid] * ${unbias * momentum};
}`;
}

/** Fills with one value. The gradient seed (`backward()`'s 1.0) and `zeros` use it. */
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
 * The hash that makes a random number per position. **It holds no state and draws from
 * the position and the seed alone.**
 *
 * A GPU has no such thing as order. The sequence threads run in is not fixed, so handing
 * on "the next random number" means nothing here, and the same input would not give the
 * same answer. So it draws as `hash(position, seed)` — independent per position, the same
 * answer for the same seed, and nothing to pass between threads.
 *
 * What it uses is a well-known integer mix (the Wang/Jenkins family). It is not for
 * cryptography and it is not for statistics either — it is as much as dropout needs to
 * pick positions. If something arrives whose statistical properties matter, this must not
 * be used for it, and that fact is written down here.
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
  // Only 24 bits are used — that is f32's mantissa, and anything above it does not ride
  // along anyway.
  let h = hash_u32(gid * 0x9e3779b9u + hash_u32(seed));
  return f32(h >> 8u) * (1.0 / 16777216.0);
}`;

/**
 * Dropout's mask. **It writes `1/(1-p)` at the surviving positions** — either 0 or that
 * value.
 *
 * The mask is produced separately because of the backward. The backward has to see **the
 * same** mask the forward drew, and drawing again would mean carrying the seed around to
 * guarantee it even with the seed unchanged. Building it once and multiplying is shorter,
 * and multiplication's derivative already exists.
 */
/**
 * A uniform draw over `[lo, hi)`.
 *
 * It uses the hash dropout already uses — drawing from the position and the seed alone,
 * so the same answer comes out even though the GPU has no order. `rrelu` draws its slope
 * here in training mode.
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
