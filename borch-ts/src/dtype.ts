/**
 * dtypes and the promotion rules.
 *
 * ## Storage is float32, and only that
 *
 * `int64` is a **label**. WGSL has no 64-bit integer, so there is no way to
 * actually hold one, and holding it in a 32-bit integer makes it `int32`,
 * not `int64`. So the values sit in float32 and the dtype is carried
 * separately — the sister project made the same choice, though its reason
 * (TF.js's cast was broken) differs from this one, which is that the means
 * of representation simply does not exist.
 *
 * **The price, written down**: integers above 2²⁴ are not exact in float32.
 * The ranges used for indices and counts are far below that, but a large
 * integer stored here drifts quietly.
 *
 * ## Why the split is by category
 *
 * torch has categories — `bool < integer < float` — and only promotes
 * **within** them. Inheriting numpy's rules would make `float32 + int64` a
 * float64, and then a learner learns the wrong rule. There is one width per
 * category here, so the rule comes out shorter still.
 */

import { RuntimeError } from "./errors.js";

export type DType = "float32" | "int64" | "bool" | "complex64";

/** 낮을수록 아래. 승격은 높은 쪽으로만 간다. */
const RANK: Readonly<Record<DType, number>> = {
  bool: 0, int64: 1, float32: 2, complex64: 3,
};

const BY_RANK: readonly DType[] = ["bool", "int64", "float32", "complex64"];

/**
 * How many float32 slots one element occupies.
 *
 * **This is the one place in this file where `int64` behaves differently.**
 * `int64` is a label, so its storage stays one float32; `complex64`
 * **really takes two** — real and imaginary interleaved.
 *
 * That breaks `tensor.ts`'s long-standing invariant, **"slot count = buffer
 * length".** The new invariant is `buffer length = size ×
 * floatsPerElement(dtype)`, and only code that knows it should touch a
 * complex buffer — a kernel that does not will read the first half as reals
 * and give a **quietly** wrong answer. The `Tensor.buffer` getter guards
 * that door.
 *
 * **`complex128` does not arrive here.** WGSL has no `f64`, so there is no
 * means at all of holding a double-precision complex — the core (numpy)
 * stops at that name for the same reason.
 */
export function floatsPerElement(d: DType): number {
  return d === "complex64" ? 2 : 1;
}

export function isComplexDType(d: DType): boolean {
  return d === "complex64";
}

/**
 * Every type name. **The place that decides whether a string arriving from
 * outside is a type** uses this — the label written in a checkpoint header
 * is someone else's text, and must not be taken on trust.
 *
 * It exports `BY_RANK` directly. Two copies of the list means one of them
 * grows when a type is added.
 */
export const DTYPES: readonly DType[] = BY_RANK;

export function dtypeName(d: DType): string {
  return `torch.${d}`;
}

/**
 * The type corresponding to a Python-side scalar. JavaScript does not
 * distinguish integer from float, so it is decided by the value.
 */
export function scalarDType(v: number | boolean): DType {
  if (typeof v === "boolean") return "bool";
  return Number.isInteger(v) ? "int64" : "float32";
}

/**
 * The result type of a binary operation.
 *
 * There are only four rules.
 * 1. **If a complex is involved, the result is complex.** Division included
 *    — this is read before rule 2 below.
 * 2. **Division is always float.** Even `int64 / int64` is float32 —
 *    torch's `/` is true division, and integer division is a separate call,
 *    `//`.
 * 3. **Subtraction involving a bool is refused.** There is no settled
 *    answer to what subtracting false from true means, so torch stops
 *    rather than invent one.
 * 4. Everything else promotes to the higher category.
 *
 * The point is that rule 1 sits **before** rule 2. Behind it, `z / z`
 * collapses to float32, and out comes a tensor that is half right — the
 * real part survives and the imaginary part is gone, so the shape and the
 * element count both look plausible.
 */
export function promote(a: DType, b: DType, op: "+" | "-" | "*" | "/"): DType {
  if (op === "-" && (a === "bool" || b === "bool")) throw BoolSubtraction();
  if (a === "complex64" || b === "complex64") return "complex64";
  if (op === "/") return "float32";
  return (RANK[a] >= RANK[b] ? a : b);
}

/**
 * Recovers a type from its rank. Used when a category has to be lifted a
 * step, as in a reduction.
 */
export function byRank(rank: number): DType {
  return BY_RANK[Math.min(Math.max(rank, 0), BY_RANK.length - 1)] ?? "float32";
}

export function rankOf(d: DType): number {
  return RANK[d];
}

/**
 * 뺄셈에 bool 이 낀 경우.
 *
 * torch 가 내는 문구를 그대로 담는다 — 막힌 사람이 검색할 것이 그 문구다.
 * **`RuntimeError` 를 그대로 쓴다**: 골든이 예외의 종류 이름을 굳혔고, 여기서 전용
 * 클래스를 세우면 그 이름이 갈린다.
 */
function BoolSubtraction(): RuntimeError {
  return new RuntimeError(
    "Subtraction, the `-` operator, with a bool tensor is not supported. " +
      "If you are trying to invert a mask, use the `~` or `logical_not()` operator instead.",
  );
}
