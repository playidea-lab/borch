/**
 * So that `print(t)` prints exactly as real torch does.
 *
 * ## Why the characters are treated as a specification
 *
 * What somebody learning does most is `print(tensor)`. Printed differently, the screen
 * does not match the textbook's example, and every time that happens they suspect
 * themselves. That holds even when the values are right.
 *
 * ## torch's rules — measured and ported
 *
 * - A float whose values are all integral prints as `1.`; otherwise to four decimals.
 * - A large ratio between the largest and smallest value, or very large values, switches
 *   to exponent form.
 * - Fields are right-aligned — with negatives among them, a positive gains a leading
 *   blank.
 * - A line over 80 characters wraps and indents by eight (the length of `tensor([`).
 * - From three dimensions on, a blank line separates the chunks.
 */

import type { DType } from "./dtype.js";

/** A line's maximum length. `tensor([` counts towards it. */
const LINE_WIDTH = 80;
const INDENT = "        "; // the same length as "tensor(["

/** The threshold for switching to exponent form — a larger ratio between the largest and
 *  smallest value than this. */
const SCI_RATIO = 1000;
const SCI_LARGE = 1e8;
const DECIMALS = 4;

interface Style {
  /** One element as characters. */
  readonly format: (v: number) => string;
  /** The width used for right alignment. */
  readonly width: number;
}

function styleFor(values: readonly number[], dtype: DType): Style {
  if (dtype === "bool") {
    const format = (v: number): string => (v !== 0 ? "True" : "False");
    return { width: widest(values, format), format };
  }
  if (dtype === "int64") {
    const format = (v: number): string => String(Math.trunc(v));
    return { width: widest(values, format), format };
  }
  const finite = values.filter((v) => Number.isFinite(v));
  const magnitudes = finite.map(Math.abs).filter((v) => v !== 0);
  const hi = magnitudes.length > 0 ? Math.max(...magnitudes) : 0;
  const lo = magnitudes.length > 0 ? Math.min(...magnitudes) : 0;
  // With all values integral torch prints the point and stops — `1.`, not `1.0000`.
  const allIntegral = finite.every((v) => Number.isInteger(v));
  const scientific = hi >= SCI_LARGE || (lo > 0 && hi / lo > SCI_RATIO);
  let format: (v: number) => string;
  if (scientific) {
    format = (v) => exponential(v);
  } else if (allIntegral) {
    format = (v) => (Number.isFinite(v) ? `${Math.trunc(v)}.` : special(v));
  } else {
    format = (v) => (Number.isFinite(v) ? v.toFixed(DECIMALS) : special(v));
  }
  return { width: widest(values, format), format };
}

function special(v: number): string {
  if (Number.isNaN(v)) return "nan";
  return v > 0 ? "inf" : "-inf";
}

/** `1.0000e+06` — the exponent is at least two digits. That is how torch prints it. */
function exponential(v: number): string {
  if (!Number.isFinite(v)) return special(v);
  const raw = v.toExponential(DECIMALS);
  return raw.replace(/e([+-])(\d)$/, "e$10$2");
}

function widest(values: readonly number[], format: (v: number) => string): number {
  let w = 0;
  for (const v of values) w = Math.max(w, format(v).length);
  return w;
}

/**
 * Lays out on one line, wrapping past 80 characters.
 *
 * @param depth how many opening brackets stand in front — the indentation
 *   grows by that much.
 */
function wrapRow(cells: readonly string[], depth: number): string {
  const pad = INDENT + " ".repeat(depth);
  const lines: string[] = [];
  let line = "";
  for (const [i, cell] of cells.entries()) {
    const piece = i === 0 ? cell : `, ${cell}`;
    const prefix = lines.length === 0 ? "tensor([".length + depth : pad.length;
    if (line !== "" && prefix + line.length + piece.length + 1 > LINE_WIDTH) {
      lines.push(line);
      line = cell;
    } else {
      line += piece;
    }
  }
  lines.push(line);
  return lines.join(`,\n${pad}`);
}

/** Builds the nested brackets. From three dimensions on, a blank line separates the
 *  chunks. */
function block(
  values: readonly number[],
  shape: readonly number[],
  style: Style,
  depth: number,
): string {
  if (shape.length === 0) return style.format(values[0] ?? 0);
  if (shape.length === 1) {
    const cells = values.map((v) => style.format(v).padStart(style.width));
    return `[${wrapRow(cells, depth)}]`;
  }
  const outer = shape[0] ?? 0;
  const inner = shape.slice(1);
  const stride = inner.reduce((a, b) => a * b, 1);
  const parts: string[] = [];
  for (let i = 0; i < outer; i++) {
    parts.push(block(values.slice(i * stride, (i + 1) * stride), inner, style, depth + 1));
  }
  // At rank 3 and above the chunks are separated by one more line.
  const gap = shape.length >= 3 ? "\n\n" : "\n";
  // The indent grows by exactly the one extra bracket. One more blank here shifts
  // everything by a column — hard to see by eye and caught only by a character
  // comparison.
  const pad = INDENT + " ".repeat(depth);
  return `[${parts.join(`,${gap}${pad}`)}]`;
}

export interface ReprInfo {
  readonly values: readonly number[];
  readonly shape: readonly number[];
  readonly dtype: DType;
  /** Whether it is a leaf that requires a gradient. */
  readonly requiresGrad: boolean;
  /** The `grad_fn` name when it is not a leaf. */
  readonly gradName: string;
}

export function formatTensor(info: ReprInfo): string {
  const style = styleFor(info.values, info.dtype);
  const body = info.values.length === 0
    ? "[]"
    : block(info.values, info.shape, style, 0);
  const parts = [body];
  if (info.gradName !== "") parts.push(`grad_fn=<${info.gradName}>`);
  else if (info.requiresGrad) parts.push("requires_grad=True");
  return `tensor(${parts.join(", ")})`;
}

/**
 * `torch.Size([2, 2])`. The shape is printed too, so it is a specification.
 */
export function formatSize(shape: readonly number[]): string {
  return `torch.Size([${shape.join(", ")}])`;
}
