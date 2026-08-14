/**
 * 골든 케이스의 **TypeScript 쪽 본문**.
 *
 * `tests/golden.json` 은 답만 준다. 케이스 본문(`lambda L: L.acos(L.tensor(unit))`)은
 * 파이썬 코드이고 기계적으로 TS 가 되지 않으므로, 여기서 **같은 이름으로 다시 쓴다.**
 * 그 나눔이 골든을 언어 중립으로 만든 방식이다 — 비싼 절반(진짜 torch 를 돌려 얻은
 * 숫자)은 옮겨지고, 싼 절반(호출 한 줄)은 다시 쓴다.
 *
 * ## 이름이 맞아야 한다
 *
 * 여기 적은 이름이 `golden.json` 의 키와 한 글자라도 다르면 그 케이스는 **조용히
 * 안 돌아간다.** 러너가 "등록됐는데 골든에 없는 이름"을 따로 세서 알리는 이유다 —
 * 오타로 0건을 돌리고 "전부 통과"라고 말하는 것이 이 프로젝트에서 제일 나쁜 결과다.
 *
 * ## 입력이 여기 있는 이유
 *
 * `math::` 계열이 쓰는 배열은 `golden_inputs()` 에 없고 `tests/cases.py` 안에 직접
 * 적혀 있다. 그래서 JSON 의 `inputs` 로 안 넘어온다 — 여기서 같은 값을 다시 적는다.
 * 값이 갈리면 대조가 대조가 아니게 되므로, 옮길 때 그대로 옮겼다.
 */

import { Tensor } from "../src/tensor.js";

/** 케이스 하나. 결과 텐서를 낸다 — 러너가 읽어서 골든과 맞춘다. */
export type Case = () => Tensor;

// ── tests/cases.py 의 math_cases 가 쓰는 입력. 그대로 옮긴 것이다. ──────────
const plain = [0.5, 2.0, -1.5, 3.0];
const unit = [0.2, 0.6, -0.9, 0.45]; // (-1, 1) 안
const big = [1.5, 2.5, 3.0, 1.2]; // > 1
const pos = [0.5, 2.0, 1.5, 3.0];
const other = [1.0, 2.0, -3.0, 0.5];
const logitIn = [0.2, 0.6, 0.35, 0.45]; // (0, 1) 안
const weights = [1.0, 2.0, 3.0, 4.0]; // 자리마다 다른 가중치

/** 함수마다 정의역이 다르다. 밖에서 부르면 NaN 이고, NaN 은 대조가 안 된다. */
const DOMAIN: Readonly<Record<string, readonly number[]>> = {
  acos: unit, asin: unit, atanh: unit,
  arccos: unit, arcsin: unit, arctanh: unit,
  acosh: big, arccosh: big,
  log1p: pos,
  logit: logitIn,
};

function pick(name: string): readonly number[] {
  return DOMAIN[name] ?? plain;
}

/**
 * `math::` 가 쓰는 단항. **왼쪽이 골든의 이름, 오른쪽이 우리 표의 연산이다.**
 *
 * torch 의 별칭(`arccos` = `acos`)은 여기서 같은 커널로 간다. 별칭마다 커널을
 * 따로 두면 하나만 고치는 날이 온다.
 */
const MATH_UNARY: Readonly<Record<string, string>> = {
  acos: "acos", acosh: "acosh", asin: "asin", asinh: "asinh",
  atan: "atan", atanh: "atanh", expm1: "expm1", log1p: "log1p", exp2: "exp2",
  deg2rad: "deg2rad", rad2deg: "rad2deg", trunc: "trunc", frac: "frac",
  positive: "positive", logit: "logit", sinc: "sinc", erfc: "erfc",
  arccos: "acos", arccosh: "acosh", arcsin: "asin", arcsinh: "asinh",
  arctan: "atan", arctanh: "atanh", fix: "trunc", absolute: "abs",
};

const MATH_BINARY: readonly string[] = [
  "atan2", "hypot", "copysign", "logaddexp", "logaddexp2",
];

/** 계단 함수. **0 을 흘린다** — 없는 것과 0 인 것은 다르다. */
const STEPS: readonly string[] = ["sign", "floor", "ceil", "round", "trunc", "fix"];

/** `x.grad` 를 낸다. 안 도착했으면 조용히 넘기지 않고 던진다. */
function gradOf(leaf: Tensor, name: string): Tensor {
  const g = leaf.grad;
  if (!g) {
    throw new Error(`${name}: 기울기가 잎에 도착하지 않았다 — 그래프가 끊겼다`);
  }
  return g;
}

export function cases(): Map<string, Case> {
  const out = new Map<string, Case>();
  const w = () => Tensor.from(weights);

  for (const [name, op] of Object.entries(MATH_UNARY)) {
    out.set(`math::${name}`, () => Tensor.from(pick(name)).unary(op));
    out.set(`math::grad::${name}`, () => {
      const x = Tensor.from(pick(name), undefined, true);
      x.unary(op).mul(w()).sum().backward();
      return gradOf(x, name);
    });
  }

  for (const name of MATH_BINARY) {
    out.set(`math::${name}`, () =>
      Tensor.from(plain).binary(name, Tensor.from(other)));
    for (const [who, tag] of ["a", "b"].entries()) {
      out.set(`math::grad::${name}/${tag}`, () => {
        const leaves = [
          Tensor.from(plain, undefined, true),
          Tensor.from(other, undefined, true),
        ] as const;
        leaves[0].binary(name, leaves[1]).mul(w()).sum().backward();
        const leaf = leaves[who];
        if (!leaf) throw new Error(`${name}: 잎 ${who} 가 없다`);
        return gradOf(leaf, `${name}/${tag}`);
      });
    }
  }

  for (const name of [...STEPS, "sgn"]) {
    const op = name === "fix" ? "trunc" : name;
    out.set(`math::grad::${name}(0이어야)`, () => {
      const x = Tensor.from(plain, undefined, true);
      x.unary(op).mul(w()).sum().backward();
      return gradOf(x, name);
    });
  }

  // 값만 묻는 나머지. 참·거짓이거나 계단이라 기울기 케이스가 없다.
  out.set("math::sgn", () => Tensor.from(plain).unary("sgn"));
  out.set("math::signbit", () => Tensor.from(plain).unary("signbit"));
  // **x 에 0 이 있어야 이 함수를 시험하는 것이다.** 없으면 `x * log(y)` 와 구별이 안 된다.
  out.set("math::xlogy(x에 0 포함)", () =>
    Tensor.from([0.0, 2.0, 0.0, 3.0])
      .binary("xlogy", Tensor.from([1.0, 2.0, 0.5, 4.0])));
  out.set("math::heaviside", () =>
    Tensor.from([-1.0, 0.0, 1.0, 0.0])
      .binary("heaviside", Tensor.from([0.5, 0.5, 0.5, 0.5])));
  out.set("math::ldexp", () =>
    Tensor.from(plain).binary("ldexp", Tensor.from([1.0, 2.0, 0.0, -1.0])));

  addReduce(out);
  addShape(out);
  return out;
}

// ── tests/cases.py 의 reduce_cases 가 쓰는 입력. 그대로 옮긴 것이다. ────────
// **동점이 일부러 들어 있다.** amax 는 동점일 때 기울기를 고르게 나누고([1,3,3,2]
// → [0,.5,.5,0]), 동점 없는 입력으로 재면 그 규칙을 하나도 안 보게 된다.
const tie = [1.0, 3.0, 3.0, 2.0];
const mat = [1.0, 5.0, 3.0, 4.0, 2.0, 6.0]; // (2, 3)
const withnan = [1.0, Number.NaN, 3.0, 5.0];

/**
 * `reduce_cases` 의 기울기 케이스는 출력에 **자리마다 다른 가중치**를 곱한 뒤
 * 더해서 역전파한다. 스칼라 출력이면 가중치가 없다 — 곱할 자리가 없으므로.
 */
function seeded(out: Tensor): Tensor {
  if (out.shape.length === 0) return out.sum();
  const w = Array.from({ length: out.size }, (_, i) => i);
  return out.mul(Tensor.from(w, out.shape)).sum();
}

function addReduce(out: Map<string, Case>): void {
  /** 값 케이스와 기울기 케이스를 같이 단다 — 둘을 떼면 한쪽만 물어보게 된다. */
  const add = (
    name: string,
    fn: (x: Tensor) => Tensor,
    src: readonly number[],
    shape?: readonly number[],
    withGrad = true,
  ): void => {
    out.set(`reduce::${name}`, () => fn(Tensor.from(src, shape)));
    if (!withGrad) return;
    out.set(`reduce::grad::${name}`, () => {
      const x = Tensor.from(src, shape, true);
      seeded(fn(x)).backward();
      return gradOf(x, name);
    });
  };

  add("amax", (x) => x.amax(), tie);
  add("amin", (x) => x.amin(), tie);
  add("amax(dim)", (x) => x.amax(1), mat, [2, 3]);
  add("amin(keepdim)", (x) => x.amin(1, true), mat, [2, 3]);
  add("nansum", (x) => x.nansum(), withnan);
  add("nanmean", (x) => x.nanmean(), withnan);
  add("logsumexp", (x) => x.logsumexp(0), tie);
  add("logsumexp(dim1)", (x) => x.logsumexp(1), mat, [2, 3]);
  add("dist", (x) => x.dist(Tensor.zeros([4])), tie);

  // 기울기 케이스가 없는 것들. 골든도 값만 굳혔다.
  out.set("reduce::aminmax/최소", () => Tensor.from(tie).amin());
  out.set("reduce::aminmax/최대", () => Tensor.from(tie).amax());
}

// ── tests/cases.py 의 shape_cases 가 쓰는 입력. ────────────────────────────
const seq = (n: number): number[] => Array.from({ length: n }, (_, i) => i);
const SQUARE = seq(9); // (3, 3)
const LINE = seq(5);
const COL = [0.0, 3.0]; // mat[:, :1] — (2, 1)

function addShape(out: Map<string, Case>): void {
  /** 이 표에서 쓰는 (2,3) 짜리. 매번 새로 올려야 케이스끼리 상태를 안 나눈다. */
  const m = (grad = false) => Tensor.from(seq(6), [2, 3], grad);
  const sq = (grad = false) => Tensor.from(SQUARE, [3, 3], grad);
  const line = (grad = false) => Tensor.from(LINE, [5], grad);
  const col = (grad = false) => Tensor.from(COL, [2, 1], grad);
  const pair = (grad = false) => Tensor.from([1.0, 2.0], [2], grad);

  const value: [string, () => Tensor][] = [
    ["expand", () => col().expand(2, 3)],
    ["expand(-1)", () => col().expand(-1, 3)],
    ["expand(앞에 축 추가)", () => m().expand(2, 2, 3)],
    ["repeat", () => m().repeat(2, 1)],
    ["repeat(둘 다)", () => m().repeat(2, 3)],
    ["ravel", () => m().ravel()],
    ["swapaxes", () => m().swapaxes(0, 1)],
    ["swapdims", () => m().swapaxes(0, 1)],
    ["select", () => m().select(0, 1)],
    ["select(dim1)", () => m().select(1, 2)],
    ["diagonal", () => sq().diagonal()],
    ["diagonal(위로 1)", () => sq().diagonal(1)],
    ["diagonal(아래로 1)", () => sq().diagonal(-1)],
    ["diagflat", () => pair().diagflat()],
    ["rot90", () => m().rot90(1)],
    ["rot90(두 번)", () => m().rot90(2)],
    ["unfold", () => line().unfold(0, 3, 1)],
    ["unfold(걸음2)", () => line().unfold(0, 2, 2)],
    ["unflatten", () => Tensor.from(seq(6), [6]).unflatten(0, [2, 3])],
    ["fliplr", () => m().fliplr()],
    ["flipud", () => m().flipud()],
    ["atleast_2d", () => Tensor.from([1.0], []).atleast2d()],
  ];
  for (const [name, fn] of value) out.set(`shape::${name}`, fn);

  for (let k = 0; k < 3; k++) {
    out.set(`shape::hsplit[${k}]`, () => {
      const part = m().hsplit(3)[k];
      if (!part) throw new Error(`hsplit 조각 ${k} 가 없다`);
      return part;
    });
  }
  for (let k = 0; k < 2; k++) {
    out.set(`shape::vsplit[${k}]`, () => {
      const part = m().vsplit(2)[k];
      if (!part) throw new Error(`vsplit 조각 ${k} 가 없다`);
      return part;
    });
  }

  // **expand 와 unfold 가 여기서 갈린다.** expand 는 늘린 축을 도로 합치고, unfold 는
  // 겹친 창만큼 쌓는다 — 길이 5 를 3·1 로 펴면 [1,2,3,2,1] 이다.
  const grads: [string, () => Tensor, (g: boolean) => Tensor][] = [
    ["expand", () => col(true).expand(2, 3), col],
    ["repeat", () => m(true).repeat(2, 1), m],
    ["diagonal", () => sq(true).diagonal(), sq],
    ["diagonal(위로 1)", () => sq(true).diagonal(1), sq],
    ["diagflat", () => pair(true).diagflat(), pair],
    ["rot90", () => m(true).rot90(1), m],
    ["unfold(겹침)", () => line(true).unfold(0, 3, 1), line],
    ["select", () => m(true).select(0, 1), m],
    ["swapaxes", () => m(true).swapaxes(0, 1), m],
  ];
  for (const [name, build] of grads) {
    out.set(`shape::grad::${name}`, () => {
      // 잎을 다시 잡으려면 케이스 안에서 만들어야 한다 — 밖에서 만들면 케이스끼리
      // 같은 텐서를 나눠 쓰고 기울기가 쌓인다.
      const leaves: Tensor[] = [];
      const res = withLeafCapture(build, leaves);
      seeded(res).backward();
      const leaf = leaves[0];
      if (!leaf) throw new Error(`${name}: 잎을 못 잡았다`);
      return gradOf(leaf, name);
    });
  }
}

/**
 * `requires_grad` 인 잎을 붙잡는다.
 *
 * 케이스 본문이 `col(true).expand(...)` 처럼 잎을 그 자리에서 만들기 때문에, 나중에
 * `x.grad` 를 보려면 그 잎을 되찾아야 한다. 결과에서 그래프를 거슬러 올라가 잎을
 * 찾는 쪽이 본문을 둘로 쪼개는 것보다 낫다 — 본문이 골든 이름과 짝이어야 읽힌다.
 */
function withLeafCapture(build: () => Tensor, into: Tensor[]): Tensor {
  const result = build();
  const seen = new Set<Tensor>();
  const stack: Tensor[] = [result];
  while (stack.length > 0) {
    const node = stack.pop();
    if (!node || seen.has(node)) continue;
    seen.add(node);
    if (node.parents.length === 0 && node.requiresGrad) into.push(node);
    for (const p of node.parents) stack.push(p as Tensor);
  }
  return result;
}
