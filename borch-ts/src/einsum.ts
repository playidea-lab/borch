/**
 * `einsum` — 첨자로 적는 축약.
 *
 * ## 무엇까지 하는가
 *
 * 피연산자 **한 개 또는 두 개**, 생략 부호(`...`) 없음, 한 피연산자 안에서 같은 첨자가
 * 두 번 나오지 않음(대각선 뽑기 없음). 그 밖은 던진다.
 *
 * **못 하는 것을 조용히 넘기지 않는다.** einsum 은 표기가 짧아서 안 되는 조합을
 * 넣기 쉽고, 그때 그럴듯한 답이 나오면 어디가 틀렸는지 찾을 길이 없다.
 *
 * ## 어떻게 하는가
 *
 * 새 커널이 없다. 축을 옮기고(`permute`), 접고(`sumDim`), 필요하면 행렬곱으로
 * 떨어뜨린다 — 전부 이미 있고 이미 골든이 보는 연산들이다. 손으로 쓴 역방향이
 * 하나도 안 생기는 것이 이 방식의 값이다.
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
  // 출력을 안 적으면 **한 번만 나온 첨자를 알파벳 순으로** 남긴다. numpy 와 같은 규칙이다.
  const seen = new Map<string, number>();
  for (const term of inputs) {
    for (const ch of term) seen.set(ch, (seen.get(ch) ?? 0) + 1);
  }
  const once = [...seen.entries()].filter(([, n]) => n === 1).map(([c]) => c);
  return { inputs, output: once.sort().join("") };
}

/** 한 항을 목표 첨자 순서로 돌리고, 남는 축은 접는다. */
function alignOne(t: Tensor, term: string, output: string): Tensor {
  const keep = [...term].filter((c) => output.includes(c));
  const drop = [...term].filter((c) => !output.includes(c));
  let cur = t;
  let letters = term;
  // 뒤에서부터 접어야 앞쪽 축 번호가 안 밀린다.
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

  // 첨자를 셋으로 가른다: 양쪽에 있고 출력에도 있는 것(배치), 양쪽에 있고 출력에
  // 없는 것(줄일 것), 한쪽에만 있는 것(남길 것).
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

/** 축을 목표 순서로 돌린다. */
function reorder(t: Tensor, term: string, want: readonly string[]): Tensor {
  const order = want.map((c) => term.indexOf(c));
  return order.length > 1 ? t.permute(order) : t;
}
