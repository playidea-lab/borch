/**
 * The place brackets occupy — the syntax for copying `x[...]` across.
 *
 * ## Why it is needed
 *
 * Brackets appear everywhere in torch code and **JavaScript cannot overload `[]`.** So
 * here each line becomes a different method — `select`, `narrow`, `indexSelect`, `gather`,
 * `maskedSelect`… fifteen of them. Every value is right, and whoever is copying **has to
 * choose "which method is this" line by line.** Whether `x[1:3]` is `narrow` or `select`,
 * and whether the arguments are `(dim, start, length)` or `(dim, start, end)`.
 *
 * `at()` narrows those fifteen branches into one door. It removes nothing — it **opens one
 * more door**, and the existing methods stay.
 *
 * ## Why a slice is a function
 *
 * Whether `x.at([1, 3])` means "axis 0 at 1, axis 1 at 3" or "slice 1:3" **cannot be told
 * from an array alone.** Python's syntax separates them (`x[1, 3]` against `x[1:3]`) and
 * here both are arrays.
 *
 * So the slice becomes a named thing. Python's `x[1:3]` also resolves to
 * `x[slice(1, 3)]`, so **it uses the same name** — not something new to learn but the name
 * that was there all along.
 *
 * ```ts
 * x.at(0)                     // x[0]
 * x.at([null, 1])             // x[:, 1]        null is Python's `:`
 * x.at(slice(1, 3))           // x[1:3]
 * x.at([0, slice(1, 3)])      // x[0, 1:3]
 * x.at(slice(null, null, 2))  // x[::2]
 * x.at([[0, 2]])              // x[[0, 2]]      two brackets — the same shape as numpy
 * ```
 *
 * **The outermost array is always the list of axes.** Selecting by an index list wraps one
 * layer more — the same shape as the place numpy's `x[0, 1]` and `x[[0, 1]]` diverge.
 */

import { RuntimeError } from "./errors.js";

/**
 * `x[start:end:step]`. Do not build one directly — use `slice()`.
 */
export interface Slice {
  readonly kind: "slice";
  readonly start: number | null;
  readonly end: number | null;
  readonly step: number;
}

/**
 * The span to cut. An empty position runs to the end, as in Python.
 *
 * **A negative step is refused.** Python's `x[::-1]` reverses, but there is
 * a `flip` here that already does that job. Letting two spellings do one
 * thing blurs which is canonical, and that is a place this repository has
 * repeatedly stayed away from.
 */
export function slice(
  start: number | null = null,
  end: number | null = null,
  step = 1,
): Slice {
  if (!Number.isInteger(step) || step < 1) {
    throw new RuntimeError(
      `slice step must be a positive integer: ${step}` +
        (step < 0 ? " — use flip() to reverse." : ""),
    );
  }
  return { kind: "slice", start, end, step };
}

export function isSlice(v: unknown): v is Slice {
  return typeof v === "object" && v !== null
    && (v as Slice).kind === "slice";
}

/**
 * How one axis is to be read — **a plan, not a value.**
 *
 * Keeping parsing apart from execution means this file never calls `Tensor`
 * (no cycle). Ranges and ordering are all settled here, so the side that
 * executes only ever receives numbers that already hold.
 */
export type AxisPlan =
  | { kind: "int"; at: number }
  | { kind: "range"; start: number; length: number }
  | { kind: "picks"; indices: number[] }
  | { kind: "whole" };

/**
 * Python's slice arithmetic. A negative counts from the back and an overrun is clamped.
 *
 * **It does not throw here** — Python gives `x[5:99]` as empty too. Where an overrun is an
 * error is the integer-index side, and that is looked at separately.
 */
function resolveSlice(s: Slice, size: number): AxisPlan {
  const clamp = (v: number): number => Math.min(Math.max(v, 0), size);
  const from = clamp(s.start === null ? 0 : s.start < 0 ? s.start + size : s.start);
  const to = clamp(s.end === null ? size : s.end < 0 ? s.end + size : s.end);
  if (s.step === 1) {
    return { kind: "range", start: from, length: Math.max(0, to - from) };
  }
  const indices: number[] = [];
  for (let i = from; i < to; i += s.step) indices.push(i);
  return { kind: "picks", indices };
}

/**
 * A single-axis index, turned into a plan.
 *
 * @param axis which axis it is. Used only in the error message — without
 *   saying which position is wrong, someone has to count them by hand at
 *   around rank four.
 */
export function planAxis(
  index: number | null | Slice | readonly number[],
  size: number,
  axis: number,
): AxisPlan {
  if (index === null) return { kind: "whole" };
  if (typeof index === "number") {
    if (!Number.isInteger(index)) {
      throw new RuntimeError(`index for dimension ${axis} is not an integer: ${index}`);
    }
    const at = index < 0 ? index + size : index;
    if (at < 0 || at >= size) {
      throw new RuntimeError(
        `index ${index} is out of bounds for dimension ${axis} with size ${size}`,
      );
    }
    return { kind: "int", at };
  }
  if (isSlice(index)) return resolveSlice(index, size);
  if (Array.isArray(index)) {
    const picks = index.map((v) => {
      const at = v < 0 ? v + size : v;
      if (!Number.isInteger(v) || at < 0 || at >= size) {
        throw new RuntimeError(
          `index ${v} is out of bounds for dimension ${axis} with size ${size}`,
        );
      }
      return at;
    });
    return { kind: "picks", indices: picks };
  }
  throw new RuntimeError(`cannot read the index for dimension ${axis}: ${String(index)}`);
}
