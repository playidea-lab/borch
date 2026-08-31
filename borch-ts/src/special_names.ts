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

import type { Tensor } from "./tensor.js";
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
