/**
 * What `vision.ts` and `v2.ts` both need and neither should own. **Internal** —
 * nothing here is a public name, and the reference generator's list does not
 * carry this file.
 *
 * ## Why it is its own file
 *
 * It used to be four lines at the top of `vision.ts`, which was right while
 * `vision.ts` was the only thing drawing. `v2.ts` draws too, and **two files
 * cannot share a module-private variable.** The choices were to export the draw
 * helpers from `vision.ts` — putting `nextFloat` in the public reference, where
 * it is nobody's business — or to move the stream somewhere both can reach.
 *
 * The alternative that looks simpler is a second generator in `v2.ts`, and it is
 * the one that breaks the promise: `manualSeed` is a single door, so a v1 crop
 * and a v2 zoom-out in one pipeline have to rewind together. Two streams make a
 * seeded run reproducible in halves.
 *
 * `_linalg.ts` is the precedent — internal, not in `index.ts`, and not on the
 * reference generator's list.
 *
 * ## It does not match numpy's, and must not pretend to
 *
 * The golden never compares a draw: it pins the probability at 0 or 1, or leaves
 * one place to crop, and asks about the deterministic part alone. So this is
 * xorshift32 and Box–Muller rather than an imitation of numpy's PCG64 — an
 * imitation would be a claim the values agree, and they cannot.
 */

import { Tensor } from "./tensor.js";
import { RuntimeError } from "./errors.js";
import type { Image, Subject } from "./vision.js";

const rng = { state: 12345 };

/** `vision.manualSeed` — the public door is there, the state is here. */
export function seed(value: number): void {
  rng.state = value >>> 0;
}

export function nextFloat(): number {
  // xorshift32. Not a place that measures a distribution — only that drawing runs.
  let x = rng.state || 1;
  x ^= x << 13; x >>>= 0;
  x ^= x >> 17;
  x ^= x << 5; x >>>= 0;
  rng.state = x;
  return x / 0x100000000;
}

export function nextInt(bound: number): number {
  return bound <= 1 ? 0 : Math.floor(nextFloat() * bound);
}

/**
 * One standard normal, Box–Muller off the same stream.
 *
 * **The floor under the logarithm is load-bearing.** `nextFloat` can return 0,
 * and `log(0)` is `-Infinity`, which comes out of the square root as `NaN` and
 * spreads through every pixel it touches. One draw in four billion, which is a
 * number small enough to never see in testing and large enough to meet in a
 * training run.
 */
export function nextNormal(): number {
  const u = Math.max(nextFloat(), Number.MIN_VALUE);
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * nextFloat());
}

/** A permutation of `0..n-1`, Fisher–Yates off the same stream. */
export function nextPermutation(n: number): number[] {
  const order = Array.from({ length: n }, (_, i) => i);
  for (let i = n - 1; i > 0; i--) {
    const j = nextInt(i + 1);
    [order[i], order[j]] = [order[j] as number, order[i] as number];
  }
  return order;
}


/** `FiveCrop` and `TenCrop` hand back several pictures; everything else takes one. */
export function isSeveral(x: Subject): x is readonly Image[] {
  return Array.isArray(x);
}

/**
 * One picture, or a refusal that says which of the two wrong things arrived.
 *
 * **Both branches were paid for.** An array reaches here because `Transform`
 * widened for `FiveCrop`, so `Compose([new FiveCrop(3), new ToTensor()])` type
 * checks; unblocked it destructures the array and blows up as
 * `shape [,,] does not match 0 elements` — an accident with no record of what was
 * done wrong. A tensor reaches here from a pipeline with `ToTensor` too early,
 * and the message says to move it rather than naming the axis it tripped on.
 */
export function asImage(x: Subject, who: string): Image {
  if (isSeveral(x)) {
    throw new RuntimeError(
      `${who} takes one picture — it received ${x.length} of them.\n` +
      "FiveCrop and TenCrop hand back several; take them apart with a Lambda first.");
  }
  if (x instanceof Tensor) {
    throw new Error(
      `${who} takes an (H,W,C) array — got a tensor.\n` +
        "Move ToTensor later in the pipeline: a tensor per image is a GPU buffer per image.",
    );
  }
  return x;
}


/**
 * Python's float repr: an integer still carries its decimal point (`1.0`, not
 * `1`), and that is what the frozen reprs hold.
 *
 * **Both sides of the library print through here.** It was `vision.ts`-private
 * while only v1 printed; `v2.ts` prints the same numbers under different field
 * names, and a second copy is how `side_range=(1, 4)` gets written where
 * `(1.0, 4.0)` is frozen.
 */
export function pyFloat(v: number): string {
  return Number.isInteger(v) ? `${v}.0` : String(v);
}


// ── The argument normalisations, shared because **both namespaces print them** ──
//
// v2's transforms delegate their behaviour to v1's and declare their own printed
// fields, and several of those fields are the *normalised* value rather than the one
// the caller passed: `ColorJitter(0.5)` prints `brightness=(0.5, 1.5)` and
// `RandomRotation(30)` prints `degrees=[-30.0, 30.0]`. A v1 object keeps those
// privately, so a twin can either read them — which TypeScript will not allow — or
// normalise again. Normalising again in a second place is how the two come to
// disagree about what `0` means, so the normalisation lives here and both call it.

/** `null` means it is not used, and that prints as `None`. */
export type Span = readonly [number, number] | null;

export function checkSpan(
  value: number | readonly [number, number] | undefined,
  name: string,
  center: number,
  lo: number,
  hi: number,
  clipFirstOnZero: boolean,
): Span {
  let pair: [number, number];
  if (value === undefined) {
    pair = [center, center];
  } else if (typeof value === "number") {
    if (value < 0) {
      throw new RuntimeError(
        `${name} as a single number must be non-negative — got ${value}.\n` +
        `(torch: If ${name} is a single number, it must be non negative.)`);
    }
    pair = [center - value, center + value];
    if (clipFirstOnZero) pair[0] = Math.max(pair[0], 0);
  } else if (Array.isArray(value) && value.length === 2) {
    pair = [Number(value[0]), Number(value[1])];
  } else {
    throw new RuntimeError(
      `${name} is a single number or a pair — got ${JSON.stringify(value)}.\n` +
      `(torch: ${name} should be a single number or a list/tuple with length 2.)`);
  }
  if (!(lo <= pair[0] && pair[0] <= pair[1] && pair[1] <= hi)) {
    throw new RuntimeError(
      `${name} values should be between (${lo}, ${hi}), but got [${pair[0]}, ${pair[1]}].\n` +
      `(torch: ${name} values should be between (${lo}, ${hi}), but got [${pair[0]}, ${pair[1]}].)`);
  }
  // **The identity is stored as `None` rather than as a range that does nothing.**
  // Applied anyway it costs one blend, and on a byte picture one rounding — "no jitter"
  // and "a jitter of exactly 1" are different things.
  return pair[0] === pair[1] && pair[0] === center ? null : [pair[0], pair[1]];
}

export function spanText(s: Span): string {
  return s === null ? "None" : `(${s[0]}, ${s[1]})`;
}

export function setupAngle(x: number | readonly number[], name: string): [number, number] {
  if (typeof x === "number") {
    if (x < 0) {
      throw new RuntimeError(
        `${name} as a single number must be positive — got ${x}.\n` +
        `(torch: If ${name} is a single number, it must be positive.)`);
    }
    return [-x, x];
  }
  if (x.length !== 2) {
    throw new RuntimeError(
      `${name} is a single number or a pair — got ${x.length} numbers.\n` +
      `(torch: ${name} should be a sequence of length 2.)`);
  }
  return [x[0] ?? 0, x[1] ?? 0];
}

export function pairOf(size: number | readonly [number, number]): [number, number] {
  return typeof size === "number" ? [size, size] : [size[0], size[1]];
}

/**
 * One draw from Gamma(shape, 1) — Marsaglia–Tsang, off the same stream.
 *
 * **Below shape 1 it is the shape-plus-one draw scaled down**, which is the
 * standard boost and not an approximation: the rejection step's bound only holds
 * for `d = shape - 1/3 > 0`, so a shape under 1 has to be lifted and corrected.
 */
function nextGamma(shape: number): number {
  if (shape < 1) return nextGamma(shape + 1) * Math.pow(nextFloat(), 1 / shape);
  const d = shape - 1 / 3;
  const c = 1 / Math.sqrt(9 * d);
  for (;;) {
    const x = nextNormal();
    const v = (1 + c * x) ** 3;
    if (v <= 0) continue;
    const u = Math.max(nextFloat(), Number.MIN_VALUE);
    if (Math.log(u) < 0.5 * x * x + d - d * v + d * Math.log(v)) return d * v;
  }
}

/**
 * One draw from Beta(alpha, alpha) — the blend weight `MixUp` and `CutMix` use.
 *
 * Two Gammas over their sum. **The symmetric case is the only one needed**, and
 * torchvision draws it the same way; what neither side can promise is the same
 * number, since the streams differ — which is why the golden holds these two
 * transforms' reprs and no value case.
 */
export function nextBeta(alpha: number): number {
  const x = nextGamma(alpha);
  const y = nextGamma(alpha);
  return x + y === 0 ? 0.5 : x / (x + y);
}
