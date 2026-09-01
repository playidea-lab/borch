/**
 * `torch.special` — the namespace, which is twenty-two names and no new arithmetic.
 *
 * Every function here forwards to a method this library already has. torch spells a
 * handful of things twice, and this is the second spelling: `expit` is `sigmoid`,
 * `gammaln` is `lgamma`, `psi` is `digamma`, `modified_bessel_i0` is `i0`, and
 * `gammainc`/`gammaincc` are the regularised incomplete gammas next door in
 * `special.ts`. None of the six is guessable from the name, which is the whole reason
 * the namespace is worth carrying: `special.expit(x)` is the line a paper's appendix
 * writes, and without this module it stopped on a namespace that was not there while
 * `x.sigmoid()` sat one call away.
 *
 * **Why the file is not called `special.ts`.** That name is taken by the module holding
 * the incomplete-gamma and polygamma *kernels* — WGSL, and two of the names below stand
 * on it. Renaming that file would touch every importer for a cosmetic gain, so the
 * awkward name is here instead of there, with this paragraph as the apology.
 *
 * **This is deliberately thin.** Writing the bodies again would be a second copy of
 * arithmetic that is already tested, which is the defect this repository keeps finding
 * — a fix reaches one copy and the hand-written one is the copy it misses. If a line
 * below is longer than `return x.something()`, it is because the argument order differs
 * from the method's, and that is said where it happens.
 *
 * The 35 names of `torch.special` that are **not** here are real arithmetic — the
 * Bessel, Airy, `ndtr` and orthogonal-polynomial families — and `tests/torch_gap.py`
 * carries each with its reason.
 */

// **`Tensor` as a value, not only as a type.** The forwarding half of this file needs
// the name for its parameters alone; the arithmetic half calls `Tensor.full` to make the
// constants a recurrence multiplies by, and an `import type` is erased at run time.
import { Tensor } from "./tensor.js";
import { igamma, igammac } from "./special.js";

// ── the eleven spelled the same in both places ────────────────────────────────

export function digamma(input: Tensor): Tensor {
  return input.digamma();
}

export function erf(input: Tensor): Tensor {
  return input.erf();
}

export function erfc(input: Tensor): Tensor {
  return input.erfc();
}

export function erfinv(input: Tensor): Tensor {
  return input.erfinv();
}

export function exp2(input: Tensor): Tensor {
  return input.exp2();
}

export function expm1(input: Tensor): Tensor {
  return input.expm1();
}

export function i0(input: Tensor): Tensor {
  return input.i0();
}

export function log1p(input: Tensor): Tensor {
  return input.log1p();
}

export function sinc(input: Tensor): Tensor {
  return input.sinc();
}

/** `eps` clamps the input away from 0 and 1 before the log — torch takes it here too. */
export function logit(input: Tensor, eps?: number | null): Tensor {
  return input.logit(eps);
}

/**
 * **`decimals` is absent from torch's own docstring for this name and torch reads it.**
 * `special.round(0.34567, decimals=3)` answers `0.346` (measured), while the prose says
 * `round(input, *, out=None)`. Dropping the parameter to match the documentation would
 * take a working call away, so it is carried.
 */
export function round(input: Tensor, decimals?: number): Tensor {
  return input.round(decimals);
}

// ── the four torch spells differently here than at the top level ──────────────
//
// These are the reason a forwarding namespace can be wrong at all. Each one computes a
// real number under the wrong name if the pair is crossed, so nothing raises and the
// values are simply somebody else's.

/** The logistic sigmoid, under statistics' name for it. */
export function expit(input: Tensor): Tensor {
  return input.sigmoid();
}

/** The log-gamma, under statistics' name for it. */
export function gammaln(input: Tensor): Tensor {
  return input.lgamma();
}

/** The digamma, under its classical name. */
export function psi(input: Tensor): Tensor {
  return input.digamma();
}

/** The zeroth modified Bessel — the same function as `i0`, spelled out. */
export function modifiedBesselI0(input: Tensor): Tensor {
  return input.i0();
}

// ── the ones taking more than one argument ────────────────────────────────────

/**
 * **`(n, input)`, and that order is torch's.** `special.polygamma(x, 1)` is a
 * `TypeError` there (measured), so taking them the other way round would accept a call
 * torch refuses. The method is `x.polygamma(n)`, which is the pair reversed — the one
 * place in this file where forwarding is not a straight pass-through.
 */
export function polygamma(n: number, input: Tensor): Tensor {
  return input.polygamma(n);
}

export function xlogy(input: Tensor, other: Tensor): Tensor {
  return input.xlogy(other);
}

export function logsumexp(input: Tensor, dim?: number, keepdim = false): Tensor {
  return input.logsumexp(dim, keepdim);
}

/**
 * **`softmax` and `logSoftmax` take `dtype`, not a destination.** Eighteen of this
 * namespace's names accept `out=` in torch and these two do not — measured, where
 * `special.softmax(x, 1, out=…)` raises and the top-level `softmax` does not.
 */
export function softmax(input: Tensor, dim = 0, dtype?: Parameters<Tensor["softmax"]>[1]): Tensor {
  return input.softmax(dim, dtype);
}

export function logSoftmax(
  input: Tensor,
  dim = 0,
  dtype?: Parameters<Tensor["logSoftmax"]>[1],
): Tensor {
  return input.logSoftmax(dim, dtype);
}

// ── the two that were nearly counted as missing ───────────────────────────────
//
// The first sweep of this namespace put `gammainc` and `gammaincc` among the names
// that are not here. They are `igamma` and `igammac`, which this library has had all
// along — measured equal in torch, not assumed from the names.

/** The regularised lower incomplete gamma `P(a, x)`. */
export function gammainc(input: Tensor, other: Tensor): Tensor {
  return igamma(input, other);
}

/** The upper branch, `Q(a, x) = 1 − P(a, x)`. */
export function gammaincc(input: Tensor, other: Tensor): Tensor {
  return igammac(input, other);
}

// ── arithmetic of its own, built from what is already here ───────────────────
//
// The core has all thirty-four of `torch.special`'s remaining names in numpy. **Eighteen
// of them need no shader**, and this is those eighteen: twelve orthogonal recurrences,
// which are `mul` and `sub` in a loop, and six compositions whose safe form is a
// composition of pieces this library already has.
//
// The other sixteen are not here and are recorded by name in `tests/ts_axis.py`. They
// are the ones whose whole reason for existing is that the obvious composition breaks —
// `erfcx`, `log_ndtr`, `i0e`, the Bessel and Airy families, `zeta` — and each needs a
// kernel of its own rather than an arrangement of existing ones. Writing them as
// compositions here would agree with the core at every ordinary input and hand back
// `inf` in the tail, which is exactly what the core declined to do.
//
// **Every value below was checked against the core, not against a textbook.** The two
// implementations are held to one answer by the golden, whose arithmetic cases under
// this namespace were frozen from real torch before either was written.

/** A scalar as a tensor, for the arithmetic below. */
function k(value: number): Tensor {
  return Tensor.full([], value);
}

/**
 * The three-term recurrence, given its two opening terms and its step.
 *
 * **Twelve names, one engine**, exactly as in the core. Reading them as twelve families
 * is what made twelve one-line differences look like twelve pieces of work.
 *
 * **A negative order is 0** and orders 0 and 1 never enter the loop — measured against
 * torch, which answers 0 rather than raising and rather than `T₋ₙ = Tₙ`.
 */
function recurrence(
  x: Tensor, n: number,
  second: (x: Tensor) => Tensor,
  step: (k: number, prev: Tensor, prev2: Tensor, x: Tensor) => Tensor,
): Tensor {
  const order = Math.trunc(n);
  if (order < 0) return x.mul(k(0));
  const first = x.mul(k(0)).add(k(1));          // ones, in x's shape
  if (order === 0) return first;
  let prev2 = first;
  let prev = second(x);
  for (let i = 2; i <= order; i++) {
    const next = step(i, prev, prev2, x);
    prev2 = prev;
    prev = next;
  }
  return prev;
}

/** The four Chebyshev kinds differ in the second term alone; the step is shared. */
function chebyshev(kind: "t" | "u" | "v" | "w", shifted: boolean) {
  const opening = (xx: Tensor): Tensor => {
    switch (kind) {
      case "t": return xx;
      case "u": return xx.mul(k(2));
      case "v": return xx.mul(k(2)).sub(k(1));
      default: return xx.mul(k(2)).add(k(1));
    }
  };
  // **The shifted four are the plain four at `2x − 1`**, checked rather than assumed:
  // shifted `T₂` at -1.5 is 31, which is `T₂(-4)`.
  const inner = (x: Tensor): Tensor => (shifted ? x.mul(k(2)).sub(k(1)) : x);
  return (input: Tensor, n: number): Tensor =>
    recurrence(input, n,
               (x) => opening(inner(x)),
               (_i, prev, prev2, x) => inner(x).mul(k(2)).mul(prev).sub(prev2));
}

export const chebyshevPolynomialT = chebyshev("t", false);
export const chebyshevPolynomialU = chebyshev("u", false);
export const chebyshevPolynomialV = chebyshev("v", false);
export const chebyshevPolynomialW = chebyshev("w", false);
export const shiftedChebyshevPolynomialT = chebyshev("t", true);
export const shiftedChebyshevPolynomialU = chebyshev("u", true);
export const shiftedChebyshevPolynomialV = chebyshev("v", true);
export const shiftedChebyshevPolynomialW = chebyshev("w", true);

/** The **physicists'** Hermite — `H(k) = 2x·H(k−1) − 2(k−1)·H(k−2)`. */
export function hermitePolynomialH(input: Tensor, n: number): Tensor {
  return recurrence(input, n, (x) => x.mul(k(2)),
                    (i, prev, prev2, x) =>
                      x.mul(k(2)).mul(prev).sub(prev2.mul(k(2 * (i - 1)))));
}

/**
 * The **probabilists'** Hermite — `He(k) = x·He(k−1) − (k−1)·He(k−2)`.
 *
 * Two functions rather than one with a flag: `H₂(-1.5)` is 7 and `He₂(-1.5)` is 1.25,
 * and no constant relates the pair term by term.
 */
export function hermitePolynomialHe(input: Tensor, n: number): Tensor {
  return recurrence(input, n, (x) => x,
                    (i, prev, prev2, x) => x.mul(prev).sub(prev2.mul(k(i - 1))));
}

/** `L(k) = ((2k−1−x)·L(k−1) − (k−1)·L(k−2)) / k`. */
export function laguerrePolynomialL(input: Tensor, n: number): Tensor {
  return recurrence(input, n, (x) => k(1).sub(x),
                    (i, prev, prev2, x) =>
                      k(2 * i - 1).sub(x).mul(prev)
                        .sub(prev2.mul(k(i - 1))).div(k(i)));
}

/** `P(k) = ((2k−1)·x·P(k−1) − (k−1)·P(k−2)) / k`. */
export function legendrePolynomialP(input: Tensor, n: number): Tensor {
  return recurrence(input, n, (x) => x,
                    (i, prev, prev2, x) =>
                      x.mul(k(2 * i - 1)).mul(prev)
                        .sub(prev2.mul(k(i - 1))).div(k(i)));
}

// ── the six compositions whose safe form is a composition ────────────────────

const SQRT2 = Math.SQRT2;

/**
 * The standard normal CDF.
 *
 * **`erfc(-x/√2)/2` and not `(1 + erf(x/√2))/2`.** The two are one formula and not one
 * arithmetic: in the left tail the second is `1` plus something very close to `-1`, and
 * the digits that matter cancel. This form never builds the difference, which is why it
 * belongs with the compositions rather than with the sixteen that need a kernel.
 */
export function ndtr(input: Tensor): Tensor {
  return input.div(k(-SQRT2)).erfc().mul(k(0.5));
}

/** The normal quantile — `√2 · erfinv(2u − 1)`. */
export function ndtri(input: Tensor): Tensor {
  return input.mul(k(2)).sub(k(1)).erfinv().mul(k(SQRT2));
}

/**
 * `-x·log(x)`, **with the two boundaries an entropy is actually evaluated at**:
 * `entr(0)` is 0 and `entr(x < 0)` is `-inf`. The plain expression is `nan` at both.
 */
export function entr(input: Tensor): Tensor {
  const value = input.mul(input.log()).neg();
  const positive = input.binary("gt", k(0));
  const zero = input.binary("eq", k(0));
  return value.where(positive, k(0).sub(k(Infinity)).where(zero.binary("eq", k(0)),
                                                          k(0)));
}

/**
 * `x·log1p(y)` with `0·anything` defined as 0.
 *
 * **Not `xlogy(x, 1 + y)`**: the `1 + y` rounds a small `y` away before the logarithm
 * sees it — at y = 1e-12 torch answers 1e-12 and that composition answers 0.
 */
export function xlog1py(input: Tensor, other: Tensor): Tensor {
  return input.mul(other.log1p()).where(input.binary("ne", k(0)), k(0));
}

/** `sin(x)/x`, and **1 at the origin** rather than the `nan` the quotient gives. */
export function sphericalBesselJ0(input: Tensor): Tensor {
  const safe = input.where(input.binary("ne", k(0)), k(1));
  return input.sin().div(safe).where(input.binary("ne", k(0)), k(1));
}

/**
 * The multivariate log-gamma, `log Γ_p(x) = p(p−1)/4·log π + Σ log Γ(x + (1−i)/2)`.
 *
 * The same function as `mvlgamma` at torch's top level — measured equal, same argument
 * order, at p = 1, 2 and 3.
 */
export function multigammaln(input: Tensor, p: number): Tensor {
  let out = k((p * (p - 1) / 4) * Math.log(Math.PI));
  for (let i = 1; i <= p; i++) {
    out = out.add(input.add(k((1 - i) / 2)).lgamma());
  }
  return out;
}
