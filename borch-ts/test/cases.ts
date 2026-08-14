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

/**
 * 케이스 하나.
 *
 * 보통은 결과 텐서를 낸다. **문자열을 내는 것도 있다** — `equal` 이 참인가, 어떤
 * 예외가 나는가처럼 값이 아니라 판정을 굳힌 케이스다. 그런 것은 근사가 아니라
 * 정확히 같아야 한다.
 */
export type Case = () => Tensor | string | Promise<Tensor | string>;

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

/**
 * 참·거짓을 **골든이 굳힌 철자로** 적는다.
 *
 * 굳힐 때 파이썬의 `str(bool(...))` 을 썼으므로 `True`/`False` 다. JS 의 `String(true)`
 * 는 `true` 라 그대로 두면 안 맞는다. 이것은 판정이 다른 것이 아니라 적는 법이
 * 다른 것이고, 골든이 답을 들고 있는 쪽이니 이쪽이 맞춘다.
 */
function verdict(value: boolean): string {
  return value ? "True" : "False";
}

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
  addMethod(out);
  addFlow(out);
  addError(out);
  return out;
}

/**
 * 같은 조건에서 **같은 종류의 예외**가, 검색 가능한 문구와 함께 나는가.
 *
 * 답의 모양은 `종류|문구=참거짓` 이다. 종류 이름까지 굳혀 두었기 때문에 torch 의
 * `RuntimeError` 를 흉내 내야 맞는다 — 자세한 이유는 `src/errors.ts` 에 있다.
 *
 * **여기 없는 다섯은 기능이 없어서다.** 정수 dtype·`nn.Linear`·`conv2d`·색인·제자리
 * 연산이 아직 없다. 없는 것을 등록해 두면 러너가 "실패" 로 세는데, 그것은 틀린 것이
 * 아니라 없는 것이다.
 */
function addError(out: Map<string, Case>): void {
  // **비동기까지 받는다.** `item()` 은 GPU 에서 읽어 오므로 async 이고, 그 안의
  // throw 는 동기 try 로 안 잡힌다 — 그대로 두면 "예외가 안 났다" 로 답한다.
  const raised = async (
    fn: () => unknown | Promise<unknown>,
    phrase: string | null,
  ): Promise<string> => {
    try {
      await fn();
      return "예외가 안 났다";
    } catch (err) {
      const kind = err instanceof Error ? err.constructor.name : typeof err;
      const text = err instanceof Error ? err.message : String(err);
      const found = phrase === null ? true : text.includes(phrase);
      return `${kind}|문구=${verdict(found)}`;
    }
  };

  const cases: [string, () => unknown | Promise<unknown>, string | null][] = [
    ["행렬곱 모양 불일치",
      () => Tensor.zeros([3, 4]).mm(Tensor.zeros([3, 2])),
      "shapes cannot be multiplied"],
    ["브로드캐스트 불가",
      () => Tensor.zeros([3, 4]).add(Tensor.zeros([3, 2])),
      "must match the size of tensor"],
    ["reshape 원소수 불일치",
      () => Tensor.zeros([2, 3]).reshape([4, 2]),
      "is invalid for input of size"],
    ["스칼라 아닌 backward",
      () => Tensor.from([0, 0, 0], [3], true).backward(),
      "grad can be implicitly created only for scalar outputs"],
    ["requires_grad 없이 backward",
      () => Tensor.zeros([3]).sum().backward(),
      "does not require grad"],
    ["여러 원소에 item()",
      () => Tensor.zeros([3]).item(),
      "cannot be converted to Scalar"],
    ["backward 두 번", () => {
      const x = Tensor.from([1.0, 2.0], [2], true);
      const y = x.mul(Tensor.full([], 2)).sum();
      y.backward();
      y.backward();
    }, "backward through the graph a second time"],
  ];
  for (const [name, fn, phrase] of cases) {
    out.set(`error::${name}`, () => raised(fn, phrase));
  }
}

// ── tests/cases.py 의 flow_cases 가 쓰는 입력. ─────────────────────────────
const F_VEC = [0.5, 2.0, 1.5, 3.0];
const F_MAT = [1, 2, 3, 4, 5, 6, 7, 8, 9]; // (3, 3), 1 부터
const F_PAIR = [1, 2, 3, 4, 5, 6]; // (2, 3)
const F_SYM = [4, 1, 1, 3]; // (2, 2)
const F_MASK = [1, 0, 1, 0];

/**
 * 기울기가 **흐르는가**만 묻는 표.
 *
 * 값만 대조하는 검사는 그래프가 끊긴 것을 못 본다 — 값은 맞기 때문이다. 자매의
 * `roll` 과 `masked_select` 가 그렇게 조용히 끊겨 있었고 골든 746건이 전부 초록이었다.
 *
 * **두 가지를 함께 답한다.** `requires_grad` 만 물으면 부족하다 — `.float()` 이
 * 참이라고 말해놓고 `.grad` 를 비워둔 적이 있고, 그 검사만 있었으면 통과했다.
 */
function addFlow(out: Map<string, Case>): void {
  const vec = () => Tensor.from(F_VEC, [4], true);
  const mat = () => Tensor.from(F_MAT, [3, 3], true);
  const pair = () => Tensor.from(F_PAIR, [2, 3], true);
  const sym = () => Tensor.from(F_SYM, [2, 2], true);
  const idx1 = () => Tensor.from([1, 0], [2]);
  const idx2 = () => Tensor.from([0, 2, 1, 0], [2, 2]);
  const mask = () => Tensor.from(F_MASK, [4]);

  /**
   * 여기 없는 것은 **아직 없는 것**이다. `median`·`msort` 는 GPU 정렬이,
   * `masked_select` 는 결과 크기가 값에 달려서 CPU 왕복이, `einsum`·`det`·`logdet`·
   * `inverse`·`cholesky` 는 선형대수가 필요하다. 등록하고 던지게 하면 "실패" 로
   * 세는데, 그건 틀린 것이 아니라 없는 것이다.
   */
  // 이름이 곧 단항 연산인 것들.
  const unaries = ["exp", "log", "sqrt", "abs", "sin", "tanh", "sigmoid",
    "relu", "erf", "erfc", "sinc"];
  for (const name of unaries) {
    out.set(`flow::${name}`, () => asks(vec(), (x) => x.unary(name)));
  }

  const others: [string, () => Tensor, (x: Tensor) => Tensor][] = [
    ["sum", vec, (x) => x.sum()],
    ["mean", vec, (x) => x.mean()],
    ["prod", vec, (x) => x.prod()],
    ["norm", vec, (x) => x.norm()],
    ["amax", vec, (x) => x.amax()],
    ["amin", vec, (x) => x.amin()],
    ["nansum", vec, (x) => x.nansum()],
    ["nanmean", vec, (x) => x.nanmean()],
    ["logsumexp", vec, (x) => x.logsumexp(0)],
    ["cumsum", vec, (x) => x.cumsum(0)],
    ["cumprod", vec, (x) => x.cumprod(0)],
    ["diff", vec, (x) => x.diff()],
    ["flip", vec, (x) => x.flip(0)],
    ["roll", vec, (x) => x.roll(1)],
    ["tile", vec, (x) => x.tile(2)],
    ["repeat_interleave", vec, (x) => x.repeatInterleave(2)],
    ["narrow", vec, (x) => x.narrow(0, 0, 2)],
    ["index_select", vec, (x) => x.indexSelect(0, idx1())],
    ["masked_fill", vec, (x) => x.maskedFill(mask(), 0.0)],
    ["unbind", vec, (x) => {
      const part = x.unbind(0)[1];
      if (!part) throw new Error("unbind 조각 1 이 없다");
      return part;
    }],
    ["ravel", vec, (x) => x.ravel()],
    ["clamp", vec, (x) => x.clamp(1.0, 2.0)],
    ["softmax", vec, (x) => x.softmax(0)],
    ["diagflat", vec, (x) => x.diagflat()],
    ["diag", mat, (x) => x.diag()],
    ["trace", mat, (x) => x.trace()],
    ["tril", mat, (x) => x.tril()],
    ["diagonal", mat, (x) => x.diagonal()],
    ["rot90", mat, (x) => x.rot90(1)],
    ["select", mat, (x) => x.select(0, 1)],
    ["swapaxes", mat, (x) => x.swapaxes(0, 1)],
    ["movedim", mat, (x) => x.movedim(0, 1)],
    ["matrix_power", sym, (x) => x.matrixPower(2)],
    ["gather", pair, (x) => x.gather(1, idx2())],
  ];
  for (const [name, leaf, fn] of others) {
    out.set(`flow::${name}`, () => asks(leaf(), fn));
  }
}

/**
 * 흘렀는가, 그리고 실제로 기울기가 잎에 닿았는가.
 *
 * 답의 철자는 골든이 굳힌 그대로다 — 파이썬 쪽 케이스가 이 문자열을 만들었다.
 */
function asks(leaf: Tensor, fn: (x: Tensor) => Tensor): string {
  const result = fn(leaf);
  const flow = result.requiresGrad ? "흐름" : "안흐름";
  try {
    result.sum().backward();
  } catch {
    return `${flow}/역전파거절`;
  }
  return `${flow}/${leaf.grad !== null ? "기울기있음" : "조용히None"}`;
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

// ── tests/cases.py 의 method_cases 가 쓰는 입력. ───────────────────────────
const M_POS = [0.5, 2.0, 1.5, 3.0];
const M_VEC = [0.5, 2.0, -1.5, 3.0];
const M_OTHER = [1.0, 2.0, -3.0, 0.5];
const M_MAT = [1, 2, 3, 4, 5, 6, 7, 8, 9]; // (3, 3), 1 부터
const M_MASK = [1, 0, 1, 0]; // bool 을 0/1 로

/**
 * `x.f(...)` 로 부를 수 있어야 하는 것들.
 *
 * **아직 안 쓴 것은 등록하지 않는다.** `sort`·`argsort`·`topk`·`median`·`unique` 는
 * GPU 정렬이 필요하고 그것을 아직 안 세웠다. 이름만 올려두고 던지게 하면 러너가
 * "실패" 로 세는데, 그건 틀린 것이 아니라 없는 것이다 — 러너는 안 물은 수를 따로
 * 세므로 여기 없는 것이 그 수에 잡힌다.
 */
function addMethod(out: Map<string, Case>): void {
  const vec = (grad = false) => Tensor.from(M_VEC, [4], grad);
  const other = () => Tensor.from(M_OTHER, [4]);
  const mat = () => Tensor.from(M_MAT, [3, 3]);

  // 표에 있는 단항은 이름만 적으면 된다.
  const unaryOn: [readonly number[], readonly string[]][] = [
    [M_VEC, ["ceil", "cos", "cosh", "erf", "floor", "isfinite", "isinf", "isnan",
      "neg", "reciprocal", "relu", "round", "sigmoid", "sign", "sin", "sinh",
      "square", "tan", "tanh"]],
    [M_POS, ["log2", "log10", "rsqrt"]],
  ];
  for (const [src, names] of unaryOn) {
    for (const name of names) {
      out.set(`method::${name}`, () => Tensor.from(src, [4]).unary(name));
    }
  }

  // 짝이 필요한 것. 비교는 0/1 을 내고 골든의 bool 과 그대로 맞는다.
  for (const name of ["eq", "ne", "lt", "le", "gt", "ge", "maximum", "minimum"]) {
    out.set(`method::${name}`, () => vec().binary(name, other()));
  }
  out.set("method::dot", () => vec().dot(other()));
  out.set("method::outer", () => vec().outer(other()));

  const single: [string, () => Tensor][] = [
    ["prod", () => vec().prod()],
    ["norm", () => vec().norm()],
    ["clamp", () => vec().clamp(0.0, 1.0)],
    ["pow", () => vec().powScalar(2)],
    ["roll", () => vec().roll(1)],
    ["cumsum", () => vec().cumsum(0)],
    ["cumprod", () => vec().cumprod(0)],
    ["softmax", () => vec().softmax(0)],
    ["narrow", () => vec().narrow(0, 0, 2)],
    ["flip", () => vec().flip(0)],
    ["tile", () => vec().tile(2)],
    ["diag", () => mat().diag()],
    ["trace", () => mat().trace()],
    ["tril", () => mat().tril()],
    ["triu", () => mat().triu()],
    ["mm", () => mat().mm(mat())],
    // **인자 순서가 함수와 뒤집힌 유일한 자리다** — 메서드는 `x.where(조건, 저쪽)` 이다.
    ["where", () => vec().where(Tensor.from(M_MASK, [4]), other())],
    ["gather", () => mat().gather(1, Tensor.from([0, 2, 1, 0, 2, 1], [3, 2]))],
  ];
  for (const [name, fn] of single) out.set(`method::${name}`, fn);

  // 여럿을 돌려주는 것 — 조각마다 이름을 붙인다. 하나만 보면 나머지가 안 걸린다.
  const pieces: [string, () => Tensor[]][] = [
    ["chunk", () => vec().chunk(2)],
    ["split", () => vec().splitSize(0, 2)],
    ["unbind", () => vec().unbind(0)],
  ];
  for (const [name, fn] of pieces) {
    for (const k of [0, 1]) {
      out.set(`method::${name}[${k}]`, () => {
        const part = fn()[k];
        if (!part) throw new Error(`${name} 조각 ${k} 가 없다`);
        return part;
      });
    }
  }

  // **movedim 은 네 조합을 다 묻는다.** (0,0) 하나였을 때는 항등이라 아무것도 안
  // 물은 것과 같았고, 그 뒤에 자매의 movedim(0,-1) 이 조용히 항등으로 굴고 있었다.
  for (const [s, d] of [[0, -1], [-1, 0], [0, 1], [1, 0]] as const) {
    out.set(`method::movedim(${s},${d})`, () => mat().movedim(s, d));
  }

  // 값이 아니라 **판정**을 굳힌 것들. 라이브러리에게 물어야 한다 — 여기서 JS 배열을
  // 비교해 답하면 통과는 하는데 아무것도 시험하지 않는다.
  out.set("method::equal", async () => verdict(await vec().equal(vec())));
  out.set("method::equal(다른 것)", async () => verdict(await vec().equal(other())));
  out.set("method::allclose", async () => verdict(await vec().allclose(vec())));

  out.set("method::grad::square", () => {
    const x = vec(true);
    x.square().mul(Tensor.from([0, 1, 2, 3], [4])).sum().backward();
    return gradOf(x, "method::square");
  });

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
