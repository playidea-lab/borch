/**
 * `einsum` — contraction written in subscripts.
 *
 * ## How far it goes
 *
 * **One or two operands**, no ellipsis (`...`), and no subscript repeated
 * within a single operand (so no diagonal extraction). Anything else
 * throws.
 *
 * **What it cannot do, it does not quietly wave through.** einsum's
 * notation is short enough that an unsupported combination is easy to
 * write, and if a plausible answer comes back there is no way to find where
 * it went wrong.
 *
 * ## How it works
 *
 * There is no new kernel. It moves axes (`permute`), folds them (`sumDim`),
 * and drops to a matrix multiply where it can — all operations that already
 * exist and that the golden cases already watch. The value of doing it this
 * way is that not one hand-written backward appears.
 */

import type { Tensor } from "./tensor.js";

interface Plan {
  readonly inputs: string[];
  readonly output: string;
}

function parse(spec: string, operands: number): Plan {
  if (spec.includes("...")) {
    throw new Error("einsum: ellipsis (...) is not supported yet");
  }
  const [lhs = "", rhs] = spec.split("->");
  const inputs = lhs.split(",").map((s) => s.trim());
  if (inputs.length !== operands) {
    throw new Error(
      `einsum: ${inputs.length} subscript terms but ${operands} operands`,
    );
  }
  for (const term of inputs) {
    if (new Set(term).size !== term.length) {
      throw new Error(`einsum: repeated subscript inside one term ('${term}') is not supported yet`);
    }
  }
  if (rhs !== undefined) return { inputs, output: rhs.trim() };
  // With no output written, it keeps **the subscripts appearing once, in alphabetical
  // order.** The same rule as numpy.
  const seen = new Map<string, number>();
  for (const term of inputs) {
    for (const ch of term) seen.set(ch, (seen.get(ch) ?? 0) + 1);
  }
  const once = [...seen.entries()].filter(([, n]) => n === 1).map(([c]) => c);
  return { inputs, output: once.sort().join("") };
}

/** Rotates one term into the target subscript order and folds the axes left over. */
function alignOne(t: Tensor, term: string, output: string): Tensor {
  const keep = [...term].filter((c) => output.includes(c));
  const drop = [...term].filter((c) => !output.includes(c));
  let cur = t;
  let letters = term;
  // Folding from the back is what keeps the earlier axis numbers from shifting.
  for (const ch of [...drop].reverse()) {
    const at = letters.indexOf(ch);
    cur = cur.sumDim(at, false);
    letters = letters.slice(0, at) + letters.slice(at + 1);
  }
  void keep;
  const order = [...output].filter((c) => letters.includes(c))
    .map((c) => letters.indexOf(c));
  return order.length > 1 ? cur.permute(order) : cur;
}

export function einsum(spec: string, ...operands: Tensor[]): Tensor {
  const plan = parse(spec, operands.length);
  const first = operands[0];
  if (!first) throw new Error("einsum: no operands given");
  if (operands.length === 1) {
    return alignOne(first, plan.inputs[0] ?? "", plan.output);
  }
  if (operands.length !== 2) {
    throw new Error(`einsum: ${operands.length} operands are not supported yet (one or two)`);
  }
  const second = operands[1];
  if (!second) throw new Error("einsum: the second operand is missing");
  const [ta = "", tb = ""] = plan.inputs;
  const out = plan.output;

  // The subscripts split three ways: in both and in the output (batch), in both and not
  // in the output (to be contracted), and in one side only (to be kept).
  const batch = [...ta].filter((c) => tb.includes(c) && out.includes(c));
  const shrink = [...ta].filter((c) => tb.includes(c) && !out.includes(c));
  const keepA = [...ta].filter((c) => !tb.includes(c) && out.includes(c));
  const keepB = [...tb].filter((c) => !ta.includes(c) && out.includes(c));
  if (batch.length > 0) {
    throw new Error(`einsum: batch subscripts (${batch.join("")}) are not supported yet`);
  }

  const a = reorder(first, ta, [...keepA, ...shrink]);
  const b = reorder(second, tb, [...shrink, ...keepB]);
  const rowsA = keepA.reduce((n, c) => n * dimOf(first, ta, c), 1);
  const inner = shrink.reduce((n, c) => n * dimOf(first, ta, c), 1);
  const colsB = keepB.reduce((n, c) => n * dimOf(second, tb, c), 1);
  const product = a.reshape([rowsA, inner]).mm(b.reshape([inner, colsB]));

  const shape = [
    ...keepA.map((c) => dimOf(first, ta, c)),
    ...keepB.map((c) => dimOf(second, tb, c)),
  ];
  const natural = [...keepA, ...keepB].join("");
  const shaped = product.reshape(shape.length > 0 ? shape : []);
  if (natural === out) return shaped;
  const order = [...out].map((c) => natural.indexOf(c));
  return order.length > 1 ? shaped.permute(order) : shaped;
}

function dimOf(t: Tensor, term: string, letter: string): number {
  return t.shape[term.indexOf(letter)] ?? 1;
}

/** Rotates the axes into the target order. */
function reorder(t: Tensor, term: string, want: readonly string[]): Tensor {
  const order = want.map((c) => term.indexOf(c));
  return order.length > 1 ? t.permute(order) : t;
}
