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

import { type DType, dtypeName } from "../src/dtype.js";
import { einsum } from "../src/einsum.js";
import * as nn from "../src/nn.js";
import * as optim from "../src/optim.js";
import * as vision from "../src/vision.js";
import { noGrad, Tensor } from "../src/tensor.js";

/**
 * 케이스 하나.
 *
 * 보통은 결과 텐서를 낸다. **문자열을 내는 것도 있다** — `equal` 이 참인가, 어떤
 * 예외가 나는가처럼 값이 아니라 판정을 굳힌 케이스다. 그런 것은 근사가 아니라
 * 정확히 같아야 한다.
 */
export type Case = () => Tensor | string | Promise<Tensor | string>;

/** 골든이 실어 보낸 입력 하나. 값은 평평하고 모양이 따로 온다. */
export interface RawInput {
  readonly shape?: number[];
  readonly values?: (number | boolean | null)[];
}

/**
 * 케이스가 함께 쓰는 입력.
 *
 * **골든이 들고 온 것을 그대로 쓴다.** 여기서 배열을 다시 적으면 그 자리가 틀릴
 * 자리가 되고, 틀려도 화면에는 "우리 값이 다르다" 로만 뜬다. 굳힐 때 쓴 바로 그
 * 숫자를 쓰는 것이 대조를 대조로 만든다.
 */
export class Inputs {
  constructor(private readonly raw_: Readonly<Record<string, RawInput>>) {}

  /** 매번 새 텐서를 만든다 — 케이스끼리 텐서를 나눠 쓰면 기울기가 쌓인다. */
  get(name: string, requiresGrad = false): Tensor {
    const entry = this.raw_[name];
    if (!entry?.values) throw new Error(`골든에 입력 '${name}' 이 없다`);
    const flat = entry.values.map((v) =>
      typeof v === "boolean" ? (v ? 1 : 0) : (v ?? Number.NaN));
    return Tensor.from(flat, entry.shape ?? [flat.length], requiresGrad);
  }

  /** 텐서로 안 만들고 값만. 이미지처럼 GPU 에 안 올리는 것이 쓴다. */
  raw(name: string): number[] {
    const entry = this.raw_[name];
    if (!entry?.values) throw new Error(`골든에 입력 '${name}' 이 없다`);
    return entry.values.map((v) =>
      typeof v === "boolean" ? (v ? 1 : 0) : (v ?? Number.NaN));
  }

  shapeOf(name: string): number[] {
    const entry = this.raw_[name];
    if (!entry) throw new Error(`골든에 입력 '${name}' 이 없다`);
    return entry.shape ?? [entry.values?.length ?? 0];
  }
}

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
 * 접두사 없는 표 — 교재 범위 밖이지만 흔한 것들.
 *
 * 입력을 `Inputs` 에서 받는다. 여기 적힌 숫자가 하나도 없다는 것이 요점이다.
 *
 * **안 넣은 것들**: 정렬이 필요한 `median`·`topk`·`sort`·`unique`·`argsort`,
 * 결과 크기가 값에 달린 `masked_select`·`bincount`, 정수 dtype 이 필요한
 * `F.one_hot`·`F.nll_loss`, 그리고 합성곱·풀링·`BatchNorm2d`·`bmm`·`einsum`·
 * `pad_sequence` 는 T2 다.
 */
function addWide(out: Map<string, Case>, inp: Inputs): void {
  const xp = () => inp.get("xp");
  const x1 = () => inp.get("x1");
  const x2 = () => inp.get("x2");
  const tail = () => inp.get("tail");

  // xp 는 양수만 — log2·rsqrt 가 음수에서 NaN 이고, NaN 은 자기 자신과도 다르다.
  for (const name of ["log2", "log10", "rsqrt", "square", "reciprocal", "tan",
    "sinh", "cosh", "erf", "sign", "floor", "ceil", "round", "sqrt", "exp",
    "abs", "sin", "cos"]) {
    out.set(name, () => xp().unary(name));
  }

  const table: [string, () => Tensor][] = [
    ["prod", () => x1().prod()],
    ["count_nonzero", () => x1().countNonzero()],
    ["cumsum", () => x1().cumsum(0)],
    ["cumprod", () => x1().cumprod(0)],
    ["norm", () => x2().norm()],
    ["gather", () => x2().gather(1, inp.get("idx2"))],
    ["flip", () => x2().flip(0)],
    ["roll", () => x1().roll(2)],
    ["index_select", () => x2().indexSelect(0, Tensor.from([2, 0], [2]))],
    ["narrow", () => x2().narrow(1, 1, 2)],
    ["split", () => piece(x1().splitSize(0, 2), 1)],
    ["chunk", () => piece(x1().chunk(3), 2)],
    ["unbind", () => piece(x2().unbind(0), 1)],
    ["maximum", () => x1().binary("maximum", x1().neg())],
    ["minimum", () => x1().binary("minimum", x1().neg())],
    ["clamp", () => x1().clamp(-0.5, 0.5)],
    ["mm", () => x2().mm(x2().transpose())],
    ["dot", () => x1().dot(x1())],
    ["outer", () => x1().narrow(0, 0, 2).outer(x1().narrow(0, 0, 3))],
    ["diag", () => square3().diag()],
    ["trace", () => square3().trace()],
    ["F.gelu", () => x1().unary("gelu")],
    ["F.silu", () => x1().unary("silu")],
    ["F.leaky_relu", () => x1().leakyRelu(0.1)],
    ["F.elu", () => x1().unary("elu")],
    ["F.log_softmax", () => x2().logSoftmax(-1)],
    ["F.l1_loss", () => x1().l1Loss(x1().neg())],
    ["F.smooth_l1_loss", () => x1().smoothL1Loss(x1().neg())],
    ["F.pad", () => x2().pad(-1, 1, 1)],
    ["F.normalize", () => x2().normalize(1)],
    ["F.cosine_similarity",
      () => x2().cosineSimilarity(x2().binary("mul", Tensor.full([], 2)))],
    ["erf(꼬리)", () => tail().unary("erf")],
    ["F.gelu(꼬리)", () => tail().unary("gelu")],
    ["eye", () => Tensor.eye(3)],
    ["full", () => Tensor.full([2, 3], 2.5)],
    ["zeros_like", () => x2().zerosLike()],
    ["ones_like", () => x2().onesLike()],
    ["linspace", () => Tensor.linspace(0, 1, 5)],
    ["tril", () => square3().tril()],
    ["triu", () => square3().triu(1)],
    ["argmax", () => x2().argmax(1)],
    ["argmin", () => x2().argmin(1)],
    ["eq", () => x1().binary("eq", x1())],
    ["gt", () => x1().binary("gt", x1().neg())],
    ["logical_and", () => positive(x1()).binary("logical_and", positive(x1().neg()))],
    ["logical_not", () => positive(x1()).unary("logical_not")],
    ["isnan", () => x1().unary("isnan")],
    ["isfinite", () => x1().unary("isfinite")],
    ["all", () => x1().binary("gt", Tensor.full([], -99)).all()],
    ["any", () => x1().binary("gt", Tensor.full([], 99)).any()],
    ["repeat_interleave", () => x1().repeatInterleave(2)],
    ["tile", () => x1().tile(2)],
    ["movedim", () => x2().movedim(0, 1)],
    ["as_tensor", () => x1()],
    // 합성곱·풀링. **걸음 2 를 따로 묻는 것은 의도다** — 역방향에서 기울기 사이에
    // 0 을 끼우는 경로가 거기서만 돈다.
    ["F.conv2d", () => inp.get("img").conv2d(inp.get("cw"), inp.get("cb"), 1, 1)],
    ["F.conv2d(패딩0)", () => inp.get("img").conv2d(inp.get("cw"), null, 1, 0)],
    ["F.conv2d(스트라이드2)",
      () => inp.get("img").conv2d(inp.get("cw"), inp.get("cb"), 2, 1)],
    ["F.max_pool2d", () => inp.get("img").maxPool2d(2)],
    ["F.avg_pool2d", () => inp.get("img").avgPool2d(2)],
    ["BatchNorm2d(학습)", () => new nn.BatchNormND(3).forward(inp.get("img"))],
    // **저장·복원 뒤의 평가 모드.** 이동 통계가 state_dict 에서 빠지면 여기서만
    // 갈린다 — 학습은 멀쩡해 보이고 추론만 틀리는, 코어가 겪은 그 결함이다.
    ["BatchNorm2d(저장→복원→eval)", () => {
      const trained = new nn.BatchNormND(3);
      trained.forward(inp.get("img")); // 이동 통계가 갱신된다
      const fresh = new nn.BatchNormND(3);
      fresh.loadStateDict(trained.stateDict());
      fresh.eval();
      return fresh.forward(inp.get("img"));
    }],
    ["median", () => x1().median().values],
    ["median(dim)", () => x2().median(1).values],
    ["median(dim).indices", () => x2().median(1).indices],
    ["topk", () => x1().topk(3).values],
    ["sort", () => x1().sort(0).values],
    ["argsort", () => x1().argsort(0)],
    ["bmm", () => x2().reshape([1, 3, 4]).bmm(x2().transpose().reshape([1, 4, 3]))],
    ["einsum", () => einsum("ij,kj->ik", x2(), x2())],
    ["F.one_hot", () => Tensor.from([0, 2], [2], false, "int64").oneHot(3)],
    ["F.nll_loss",
      () => x2().logSoftmax(-1).nllLoss(Tensor.from([0, 1, 2], [3], false, "int64"))],
    // 길이가 다른 것을 한 배치에 담는 자리. 교재 ch05 가 이 경로를 그대로 쓴다.
    ["pad_sequence", () => Tensor.padSequence(ragged())],
    ["pad_sequence(batch_first)", () => Tensor.padSequence(ragged(), true)],
    ["pad_sequence(채움값)", () => Tensor.padSequence(ragged(), true, -1.0)],
    ["pad_sequence(2차원)",
      () => Tensor.padSequence([x2().narrow(0, 0, 3), x2().narrow(0, 0, 1)], true)],
  ];
  for (const [name, fn] of table) out.set(name, fn);

  // 결과 크기가 값에 달린 것들 — CPU 를 한 번 왕복하므로 비동기다.
  out.set("unique", async () => Tensor.from([1, 1, 2, 3], [4]).unique());
  out.set("masked_select",
    async () => x1().maskedSelect(x1().binary("gt", Tensor.full([], 0))));
  out.set("bincount",
    async () => Tensor.from([0, 1, 1, 3], [4], false, "int64").bincount());

  /** `x2[:3, :3]` — 골든이 그 자리만 쓰는 케이스들이 있다. */
  function square3(): Tensor {
    return x2().narrow(0, 0, 3).narrow(1, 0, 3);
  }
}

/**
 * 계산해서 얻은 텐서를 **잎으로** 세운다.
 *
 * 골든 쪽은 `L.tensor(x2.T.copy(), requires_grad=True)` 처럼 값을 미리 만들어 잎으로
 * 넣는다. 이쪽에서 `x2().transpose()` 를 그대로 쓰면 그것은 파생 텐서라 기울기가
 * 안 쌓이고, 케이스가 "기울기가 안 왔다" 로 죽는다 — 구현이 멀쩡한데도.
 *
 * 버퍼는 같이 쓴다. 값을 다시 계산할 이유가 없다.
 */
function asLeaf(t: Tensor): Tensor {
  return new Tensor(t.buffer, t.shape, { requiresGrad: true });
}

/**
 * 랭크가 올라갈 때 **어디서 무너지는지** 본다.
 *
 * 자매에게는 이것이 한계를 재는 표였다 — TF.js 가 랭크 7 부터 일부 연산을 거절해서
 * `=거절` 이 붙은 케이스들은 "거절하는 것이 정답" 이었다. borch.ts 에는 그 한계가
 * 없으므로 **torch 와 같이 성공이 정답**이고, 답도 torch 의 것을 그대로 쓴다.
 *
 * 값을 통째로 묻는다 — 스칼라로 줄이면 자리가 뒤바뀌어도 합이 같아 통과한다.
 * 기울기도 자리마다 다른 가중치를 곱해 받는다.
 */
function addHighRank(out: Map<string, Case>, inp: Inputs): void {
  for (const r of [6, 7, 8]) {
    const key = `rank${r}`;
    const tag = `랭크${r}`;
    const shape = inp.shapeOf(key);
    const axis = Math.floor(r / 2);
    const count = shape.reduce((a, b) => a * b, 1);
    const v = (g = false) => inp.get(key, g);
    const reversed = [...Array(r).keys()].reverse();
    const lowered = [...shape.slice(0, -2),
      (shape[r - 2] ?? 1) * (shape[r - 1] ?? 1)];

    const table: [string, () => Tensor][] = [
      [`${tag} 원소별`,
        () => v().binary("mul", Tensor.full([], 2)).binary("add", Tensor.full([], 1))],
      [`${tag} permute`, () => v().permute(reversed)],
      [`${tag} reshape(내림)`, () => v().reshape(lowered)],
      [`${tag} reshape(올림)`, () => Tensor.arange(count).reshape(shape)],
      [`F.pad(${tag})`, () => v().pad(-1, 1, 2)],
    ];
    for (const [name, fn] of table) out.set(`webgpu::${name}`, fn);

    // **셋 다 값으로 묻는다 — 랭크 6 이든 8 이든.** 예전에는 7·8 만 "거절하는 것이
    // 정답" 으로 굳혀 두었는데, 그것은 TF.js 의 천장이지 이 구현의 것이 아니었다.
    // 그쪽이 사라지면서 물어볼 수 있는 것이 "안 던졌는가" 에서 "맞는 값인가" 로
    // 올라갔다. 뒤가 훨씬 센 질문이다.
    out.set(`webgpu::${tag} 합(축)`, () => v().sumDim(axis));
    out.set(`webgpu::F.pad(${tag}, 값)`,
      () => v().pad(-1, 2, 1, -1.5).pad(-2, 1, 0, -1.5));
    out.set(`webgpu::grad::${tag} 원소별`, () => {
      const x = v(true);
      x.mul(x).add(x).sum().backward();
      return gradOf(x, `${tag} 원소별`);
    });

    if (r === 6) {
      for (const kind of ["narrow", "unbind", "split"] as const) {
        out.set(`webgpu::grad::${tag} ${kind}`, () => {
          const x = v(true);
          const res = kind === "narrow" ? x.narrow(axis, 1, 2)
            : kind === "unbind" ? piece(x.unbind(axis), 1)
              : piece(x.splitSize(axis, 1), 0);
          seeded(res).backward();
          return gradOf(x, `${tag} ${kind}`);
        });
      }
    }
  }

  // 넷을 따로 두는 이유는 이력이다. 자매에서 랭크 7 은 순방향도 기울기도 됐고 랭크 8 은
  // 값만 나오고 기울기가 없었다 — 경계가 연산 이름에도 입력 랭크에도 안 걸린다는 증거였다.
  // 지금은 넷 다 값을 내지만, 짐작으로 경계를 적으면 그 짐작이 문서가 된다는 자리로 남긴다.
  for (const r of [7, 8]) {
    const key = `rank${r}_unbind`;
    out.set(`webgpu::랭크${r} unbind(순방향)`,
      () => piece(inp.get(key).unbind(0), 1));
  }
  out.set("webgpu::grad::랭크7 unbind", () => {
    const x = inp.get("rank7_unbind", true);
    seeded(piece(x.unbind(0), 1)).backward();
    return gradOf(x, "랭크7 unbind");
  });
  out.set("webgpu::grad::랭크8 unbind", () => {
    const x = inp.get("rank8_unbind", true);
    seeded(piece(x.unbind(0), 1)).backward();
    return gradOf(x, "랭크8 unbind");
  });

  // 랭크 5 는 자매가 `tf.pad` 로 조용히 값을 깨뜨리던 자리다. 우리 것도 물어둔다.
  for (const kind of ["narrow", "unbind", "split"] as const) {
    out.set(`webgpu::grad::랭크5 ${kind}`, () => {
      const x = inp.get("vol5", true);
      const res = kind === "narrow" ? x.narrow(2, 1, 2)
        : kind === "unbind" ? piece(x.unbind(2), 1)
          : piece(x.splitSize(3, 2), 0);
      seeded(res).backward();
      return gradOf(x, `랭크5 ${kind}`);
    });
  }
  out.set("webgpu::F.pad(랭크5)", () => inp.get("vol5").pad(-1, 1, 2));
  out.set("webgpu::F.pad(랭크5, 값)",
    () => inp.get("vol5").pad(-1, 2, 1, -1.5).pad(-2, 1, 0, -1.5));
}

/** 길이 3·1·2 짜리 셋. 채운 자리가 어디인지 눈으로 보이는 최소 크기다. */
function ragged(grad = false): Tensor[] {
  return [[1, 2, 3], [4], [5, 6]].map((v) => Tensor.from(v, [v.length], grad));
}

/** `x > 0` 을 0/1 로. `logical_*` 케이스가 그 형태를 쓴다. */
function positive(t: Tensor): Tensor {
  return t.binary("gt", Tensor.full([], 0));
}

function piece(parts: Tensor[], k: number): Tensor {
  const part = parts[k];
  if (!part) throw new Error(`조각 ${k} 가 없다`);
  return part;
}

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

export function cases(inputs: Inputs): Map<string, Case> {
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
  addWide(out, inputs);
  addGrad(out, inputs);
  addInplace(out);
  addLinalg(out);
  addDType(out);
  addRepr(out);
  addNdim(out, inputs);
  addTrain(out, inputs);
  addVision(out, inputs);
  addSeq(out, inputs);
  addEdge(out);
  return out;
}

/**
 * 꺾이는 자리.
 *
 * 다른 표의 입력은 거의 다 정규분포 난수다. 좋은 기본값이지만 **특별한 값이 한 번도
 * 안 나온다** — 정확히 0, 정확히 같은 두 수, 정확히 경계값, 정확히 .5. 함수가 꺾이는
 * 자리가 전부 거기에 있고, `relu` 가 그래서 골든 798 건을 뚫고 나갔다.
 *
 * 접을 때 자리마다 다른 가중치를 곱하는 것이 조건이다. 균일하게 접으면 꺾인 한 자리의
 * 차이가 합계에 묻힌다.
 */
function addEdge(out: Map<string, Case>): void {
  const z = [-2, -1, 0, 1, 2, 0];              // 정확히 0 을 품는다
  const ta = [1, 2, 3, 2], tb = [1, 5, 3, 0];  // 자리 0·2 가 동점
  const half = [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5];
  const dup = [1, 3, 2, 3];

  const set = (name: string, fn: Case): void => { out.set(`edge::${name}`, fn); };
  const seed = (t: Tensor): Tensor =>
    t.mul(Tensor.from([...Array(t.size).keys()], t.shape));

  const grad = (name: string, src: readonly number[],
                fn: (x: Tensor) => Tensor): void => {
    set(`grad::${name}`, () => {
      const x = Tensor.from([...src], [src.length], true);
      seed(fn(x)).sum().backward();
      return gradOf(x, name);
    });
  };

  // ── 0 에서 꺾이는 것들 ──
  const kinks: ReadonlyArray<readonly [string, (x: Tensor) => Tensor]> = [
    ["abs", (x) => x.abs()],
    ["sign", (x) => x.sign()],
    ["relu", (x) => x.relu()],
    ["F.leaky_relu", (x) => x.leakyRelu(0.1)],
    ["F.elu", (x) => x.unary("elu")],
    ["F.gelu", (x) => x.unary("gelu")],
    ["F.silu", (x) => x.unary("silu")],
  ];
  for (const [name, fn] of kinks) {
    set(`${name}(0포함)`, () => fn(Tensor.from([...z])));
    grad(`${name}(0포함)`, z, fn);
  }

  // ── 경계에 정확히 닿는 clamp ──
  set("clamp(경계에서)", () => Tensor.from([...z]).clamp(-1, 1));
  grad("clamp(경계에서)", z, (x) => x.clamp(-1, 1));
  grad("clamp(위만)", z, (x) => x.clampMax(1));
  grad("clamp(아래만)", z, (x) => x.clampMin(-1));

  // ── 동점 ──
  // **torch 는 동점에서 기울기를 나눠 준다** — 두 입력이 같으면 각각 절반씩이다.
  // 한쪽에 몰아주거나 양쪽에 1 씩 주는 구현은 순방향이 완벽히 같아서 값으로는 안 잡힌다.
  set("maximum(동점)", () =>
    Tensor.from([...ta]).binary("maximum", Tensor.from([...tb])));
  set("minimum(동점)", () =>
    Tensor.from([...ta]).binary("minimum", Tensor.from([...tb])));
  for (const [who, tag] of ["a", "b"].entries()) {
    for (const op of ["maximum", "minimum"]) {
      set(`grad::${op}(동점)/${tag}`, () => {
        const leaves = [
          Tensor.from([...ta], [ta.length], true),
          Tensor.from([...tb], [tb.length], true),
        ] as const;
        seed(leaves[0].binary(op, leaves[1])).sum().backward();
        const leaf = leaves[who];
        if (!leaf) throw new Error(`${op}: 잎 ${who} 가 없다`);
        return gradOf(leaf, `${op}(동점)/${tag}`);
      });
    }
  }

  set("argmax(동점)", () => Tensor.from([...dup]).argmax());
  set("max(동점).indices", () => Tensor.from([...dup]).max(0).indices);
  set("min(동점).indices", () => Tensor.from([...dup]).neg().min(0).indices);
  set("grad::max(동점)", () => {
    const x = Tensor.from([...dup], [dup.length], true);
    seed(x.max(0).values.reshape([1])).sum().backward();
    return gradOf(x, "max(동점)");
  });
  set("sort(동점).values", () => Tensor.from([...dup]).sort(0).values);
  set("sort(동점).indices", () => Tensor.from([...dup]).sort(0).indices);
  set("topk(동점).indices", () => Tensor.from([...dup]).topk(3, 0).indices);

  // 창 안에 같은 값이 둘 있는 풀링. **`maximum` 과 답이 다르다** — torch 의 풀링은
  // 이긴 자리 하나를 골라 거기로만 흘리고 나누지 않는다. 이 라이브러리의 커널은
  // "동점이면 먼저 나온 자리" 라고 적혀 있는데 그것을 확인한 적이 없었다.
  const tied = [1, 1, 2, 0, 1, 0, 2, 2, 3, 3, 0, 1, 0, 3, 1, 1];
  set("max_pool2d(동점)", () => Tensor.from([...tied], [1, 1, 4, 4]).maxPool2d(2));
  set("grad::max_pool2d(동점)", () => {
    const x = Tensor.from([...tied], [1, 1, 4, 4], true);
    seed(x.maxPool2d(2)).sum().backward();
    return gradOf(x, "max_pool2d(동점)");
  });

  // ── 반올림 규칙 ──
  // **torch 는 .5 를 짝수로 붙인다.** `floor(x + 0.5)` 로 쓰면 전부 위로 올라가 갈린다.
  set("round(.5에서)", () => Tensor.from([...half]).round());
  set("floor(정수에서)", () => Tensor.from([...z]).floor());
  set("ceil(정수에서)", () => Tensor.from([...z]).ceil());
  set("trunc(음수)", () => Tensor.from([...half]).trunc());
  set("frac(음수)", () => Tensor.from([...half]).frac());

  // **`%` 는 제수의 부호를 따른다** — `-7 % 3` 이 2 이지 -1 이 아니다. JS 의 `%` 는
  // 반대이므로 그것을 그대로 쓰면 음수 입력에서만 갈린다. 양수로는 절대 안 드러난다.
  const neg = [-7, -3, 3, 7];
  set("%(음수)", () => Tensor.from([...neg]).remainder(3));
  set("%(음수로 나누기)", () => Tensor.from([...neg]).remainder(-3));
}

/**
 * 순환망과 어텐션.
 *
 * 가중치를 밖에서 넣어 **셋이 같은 자리에서 출발**하게 한다 — 각자 초기화하면
 * 무엇이 갈렸는지가 아니라 초기화가 갈렸는지를 보게 된다. 파라미터 **이름**이
 * torch 와 같아야 `state_dict` 로 넣을 수 있다는 것도 여기서 걸린다.
 */
function addSeq(out: Map<string, Case>, inp: Inputs): void {
  const build = (kind: nn.RNNKind): nn.Recurrent => {
    const m = new nn.Recurrent(3, 4, kind);
    const low = kind.toLowerCase();
    m.loadStateDict({
      weight_ih_l0: inp.get(`${low}_wih`), weight_hh_l0: inp.get(`${low}_whh`),
      bias_ih_l0: inp.get(`${low}_bih`), bias_hh_l0: inp.get(`${low}_bhh`),
    });
    return m;
  };
  for (const kind of ["RNN", "LSTM", "GRU"] as const) {
    out.set(`seq::${kind}/출력`, () => build(kind).run(inp.get("seq_x")).output);
    // LSTM 만 상태가 둘이라 골든이 은닉만 꺼낸다.
    out.set(`seq::${kind}/마지막상태`,
      () => build(kind).run(inp.get("seq_x")).hidden);
  }

  const attention = (mask: Tensor | null): Tensor => {
    const m = new nn.MultiheadAttention(4, 2);
    m.loadStateDict({
      in_proj_weight: inp.get("mha_in_w"), in_proj_bias: inp.get("mha_in_b"),
      "out_proj.weight": inp.get("mha_out_w"), "out_proj.bias": inp.get("mha_out_b"),
    });
    return m.attend(inp.get("attn_x"), mask);
  };
  out.set("seq::MultiheadAttention", () => attention(null));
  // 인과 마스크는 **실수**다(0/-inf). "0 이 아니면 가림" 으로 뭉뚱그리면 여기서 갈린다.
  out.set("seq::MultiheadAttention(인과 마스크)",
    () => attention(nn.MultiheadAttention.causalMask(5)));
}

/**
 * `torchvision.transforms` 모양의 변환.
 *
 * **무작위 변환은 뽑기를 대조할 수 없다** — torch 의 난수기를 우리가 못 쓰기 때문이다.
 * 그래서 골든이 확률을 0 이나 1 로 못 박거나 자를 자리가 하나뿐이게 만들어 결정적인
 * 자리만 묻는다. 여기서 "무작위니까 대조 못 한다"고 넘기면 그게 안 본 것을 봤다고
 * 적는 짓이다.
 */
function addVision(out: Map<string, Case>, inp: Inputs): void {
  const mean = [0.5, 0.4, 0.3];
  const std = [0.2, 0.3, 0.4];

  /** 골든의 (H,W,C) 입력을 그대로 이미지로 본다. */
  const pic = (name: string, isByte: boolean): vision.Image => {
    const shape = inp.shapeOf(name);
    const [h = 1, w = 1, c = 1] = shape;
    return vision.image(inp.raw(name), h, w, shape.length === 2 ? 1 : c, isByte);
  };
  const u8 = () => pic("vis_u8", true);
  const asTensor = (img: vision.Image): Tensor =>
    Tensor.from(img.data, [img.height, img.width, img.channels]);

  out.set("vision::ToTensor(uint8)", () => new vision.ToTensor().apply(u8()) as Tensor);
  out.set("vision::ToTensor(실수)",
    () => new vision.ToTensor().apply(pic("vis_f", false)) as Tensor);
  out.set("vision::ToTensor(2차원)",
    () => new vision.ToTensor().apply(pic("vis_gray", true)) as Tensor);
  out.set("vision::Normalize", () =>
    new vision.Normalize(mean, std).apply(new vision.ToTensor().apply(u8())));
  out.set("vision::Compose", () =>
    new vision.Compose([new vision.ToTensor(), new vision.Normalize(mean, std)])
      .apply(u8()) as Tensor);

  // 확률을 못 박아 뽑기와 무관하게 만든다.
  for (const p of [1.0, 0.0]) {
    out.set(`vision::Flip(p=${p === 1 ? 1 : 0})`, () =>
      asTensor(new vision.RandomHorizontalFlip(p).apply(u8()) as vision.Image));
  }
  // 자를 자리가 **하나뿐**이 되게 크기를 맞춘다. 그래야 뽑기와 무관하게 결정적이다.
  out.set("vision::Crop(패딩없음)",
    () => asTensor(new vision.RandomCrop([5, 4], 0).apply(u8()) as vision.Image));
  out.set("vision::Crop(패딩1)",
    () => asTensor(new vision.RandomCrop([7, 6], 1).apply(u8()) as vision.Image));

  // 이 프로젝트는 `repr` 도 명세로 본다 — 튜토리얼이 `print(transform)` 을 한다.
  const reprs: [string, () => vision.Transform][] = [
    ["ToTensor", () => new vision.ToTensor()],
    ["Normalize", () => new vision.Normalize(mean, std)],
    ["RandomHorizontalFlip", () => new vision.RandomHorizontalFlip(0.5)],
    ["RandomCrop", () => new vision.RandomCrop(32, 4)],
    ["Compose", () => new vision.Compose([
      new vision.ToTensor(), new vision.Normalize([0.5], [0.5]),
    ])],
  ];
  for (const [name, build] of reprs) {
    out.set(`vision::repr::${name}`, () => build().describe());
  }
}

/** 골든이 쓰는 스텝 수. 적게 두는 것은 의도다 — 길게 돌리면 무엇이 틀렸는지가 아니라
 * float32 가 갈라진 것을 보게 된다. */
const TRAIN_STEPS = 5;

/**
 * **학습이 도는가** — 조각이 엮였을 때를 본다.
 *
 * 단위 대조는 연산 하나씩만 본다. 모듈·손실·옵티마이저가 엮여야만 갈리는 것이 있고,
 * 이 저장소가 통합 시나리오에서 잡은 결함들은 전부 그 자리에서 나왔다.
 */
function addTrain(out: Map<string, Case>, inp: Inputs): void {
  const build = (kind: "SGD" | "SGD(모멘텀)" | "Adam" | "RMSprop"): nn.Sequential => {
    const model = new nn.Sequential([
      new nn.Linear(6, 8), new nn.ReLU(), new nn.Linear(8, 3),
    ]);
    model.loadStateDict({
      "0.weight": inp.get("w0"), "0.bias": inp.get("b0"),
      "2.weight": inp.get("w1"), "2.bias": inp.get("b1"),
    });
    void kind;
    return model;
  };

  const optimizerFor = (kind: string, params: Tensor[]): optim.Optimizer => {
    if (kind === "SGD") return new optim.SGD(params, 0.05);
    if (kind === "SGD(모멘텀)") return new optim.SGD(params, 0.05, 0.9);
    if (kind === "Adam") return new optim.Adam(params, 0.05);
    return new optim.RMSprop(params, 0.05);
  };

  const trained = (kind: "SGD" | "SGD(모멘텀)" | "Adam" | "RMSprop"): nn.Sequential => {
    const model = build(kind);
    const opt = optimizerFor(kind, model.parameters());
    const crit = new nn.CrossEntropyLoss();
    const x = inp.get("train_x");
    const y = inp.get("train_y");
    for (let i = 0; i < TRAIN_STEPS; i++) {
      opt.zeroGrad();
      crit.forward(model.forward(x), y).backward();
      opt.step();
    }
    return model;
  };

  for (const kind of ["SGD", "SGD(모멘텀)", "Adam"] as const) {
    out.set(`train::${kind}/손실`, () => {
      const model = trained(kind);
      return new nn.CrossEntropyLoss()
        .forward(model.forward(inp.get("train_x")), inp.get("train_y"));
    });
    // **가중치까지 본다.** 손실만 보면 파라미터가 안 움직여도 비슷해 보일 수 있다.
    out.set(`train::${kind}/0.weight`, () => {
      const w = trained(kind).namedParameters()["0.weight"];
      if (!w) throw new Error("0.weight 가 없다");
      return w;
    });
  }
  out.set("train::RMSprop/0.weight", () => {
    const w = trained("RMSprop").namedParameters()["0.weight"];
    if (!w) throw new Error("0.weight 가 없다");
    return w;
  });

  // 합성곱·풀링이 학습 루프 **안에서** 엮였을 때. 단위 대조는 이것을 못 본다.
  const cnnTrained = (): nn.Sequential => {
    const model = new nn.Sequential([
      new nn.Conv2d(1, 4, 3, 1, 1), new nn.ReLU(), new nn.MaxPool2d(2),
      new nn.Flatten(), new nn.Linear(4 * 4 * 4, 3),
    ]);
    model.loadStateDict({
      "0.weight": inp.get("ck"), "0.bias": inp.get("ckb"),
      "4.weight": inp.get("fw"), "4.bias": inp.get("fb"),
    }, false);
    const opt = new optim.SGD(model.parameters(), 0.05);
    const crit = new nn.CrossEntropyLoss();
    const x = inp.get("cnn_x");
    const y = inp.get("cnn_y");
    for (let i = 0; i < TRAIN_STEPS; i++) {
      opt.zeroGrad();
      crit.forward(model.forward(x), y).backward();
      opt.step();
    }
    return model;
  };
  out.set("train::CNN/손실", () => {
    const model = cnnTrained();
    return new nn.CrossEntropyLoss()
      .forward(model.forward(inp.get("cnn_x")), inp.get("cnn_y"));
  });
  out.set("train::CNN/conv.weight", () => {
    const w = cnnTrained().namedParameters()["0.weight"];
    if (!w) throw new Error("0.weight 가 없다");
    return w;
  });

  // 스케줄은 실수 연산뿐이라 값이 그대로 같아야 한다. **한 값이 아니라 궤적 전체**를
  // 본다 — 코어가 그렇게 하다가 StepLR 의 차이를 잡았다.
  const trajectory = (make: (o: optim.Optimizer) => optim.LRScheduler): Tensor => {
    const p = Tensor.from([1.0], [1], true);
    const opt = new optim.SGD([p], 1.0);
    const sch = make(opt);
    const seen = [opt.paramGroups[0]?.lr ?? 0];
    for (let i = 0; i < 6; i++) {
      sch.step();
      seen.push(opt.paramGroups[0]?.lr ?? 0);
    }
    return Tensor.from(seen, [seen.length]);
  };

  const schedules: [string, (o: optim.Optimizer) => optim.LRScheduler][] = [
    ["StepLR", (o) => new optim.StepLR(o, 2, 0.5)],
    ["MultiStepLR", (o) => new optim.MultiStepLR(o, [2, 4], 0.5)],
    ["ExponentialLR", (o) => new optim.ExponentialLR(o, 0.9)],
    ["CosineAnnealingLR", (o) => new optim.CosineAnnealingLR(o, 6)],
    ["LambdaLR", (o) => new optim.LambdaLR(o, (e) => 1.0 / (1 + e))],
  ];
  for (const [name, make] of schedules) {
    out.set(`sched::${name}`, () => trajectory(make));
  }

  out.set("sched::ReduceLROnPlateau", () => {
    const p = Tensor.from([1.0], [1], true);
    const opt = new optim.SGD([p], 1.0);
    const sch = new optim.ReduceLROnPlateau(opt, 0.5, 1);
    const seen: number[] = [];
    for (const metric of [1.0, 1.0, 1.0, 1.0, 0.1, 1.0, 1.0, 1.0]) {
      sch.step(metric);
      seen.push(opt.paramGroups[0]?.lr ?? 0);
    }
    return Tensor.from(seen, [seen.length]);
  });
}

/**
 * 함수 형태와 메서드 형태가 같은 것을 주는가.
 *
 * **여기 여덟 건만 등록한다.** 나머지 스무 건(`conv1d`·`conv3d`·풀링·`interpolate`)은
 * 입력을 `np.random.default_rng` 로 케이스 안에서 만들어서 `golden.json` 에 안 실린다 —
 * numpy 의 난수기를 TS 로 다시 만들지 않는 한 그 값을 얻을 방법이 없다.
 */
function addNdim(out: Map<string, Case>, inp: Inputs): void {
  const flat = () => Tensor.from([0, 1, 2, 3, 4, 5, 6, 7], [2, 4]);
  const mask = () => Tensor.from([1, 0, 1, 0, 1, 0, 1, 0], [2, 4], false, "bool");

  // 1·3차원 계열. 입력이 골든에 실리고 나서 열린 자리다.
  const nd = (name: string, g = false) => inp.get(name, g);
  const values: [string, () => Tensor][] = [
    ["F.conv1d", () => nd("nd_seq").conv1d(nd("nd_k1"), null, 1, 1)],
    ["F.conv1d(걸음2)", () => nd("nd_seq").conv1d(nd("nd_k1"), null, 2, 1)],
    ["F.conv1d(채움0)", () => nd("nd_seq").conv1d(nd("nd_k1"), null, 1, 0)],
    ["F.conv3d", () => nd("nd_vol").conv3d(nd("nd_k3"), null, 1, 1)],
    ["F.conv3d(채움0)", () => nd("nd_vol").conv3d(nd("nd_k3"), null, 1, 0)],
    ["F.max_pool1d", () => nd("nd_seq").maxPool1d(2)],
    ["F.max_pool3d", () => nd("nd_vol").maxPool3d(2)],
    ["F.interpolate", () => nd("nd_img").upsample(2)],
    ["F.adaptive_avg_pool2d", () => nd("nd_img").adaptiveAvgPool(2)],
    ["nn.Conv1d", () => {
      const m = new nn.Conv1d(3, 4, 3, 1, 1);
      m.loadStateDict({ weight: nd("nd_k1"), bias: Tensor.zeros([4]) });
      return m.forward(nd("nd_seq"));
    }],
    ["nn.MaxPool1d", () => nd("nd_seq").maxPool1d(2)],
    ["nn.MaxPool3d", () => nd("nd_vol").maxPool3d(2)],
    ["nn.BatchNorm3d", () => new nn.BatchNormND(2).forward(nd("nd_vol"))],
    ["nn.Upsample", () => nd("nd_img").upsample(2)],
  ];
  for (const [name, fn] of values) out.set(`ndim::${name}`, fn);

  const grads: [string, string, (x: Tensor) => Tensor][] = [
    ["conv1d", "nd_seq", (x) => x.conv1d(nd("nd_k1"), null, 1, 1)],
    ["conv3d", "nd_vol", (x) => x.conv3d(nd("nd_k3"), null, 1, 1)],
    ["max_pool1d", "nd_seq", (x) => x.maxPool1d(2)],
    ["max_pool3d", "nd_vol", (x) => x.maxPool3d(2)],
    ["interpolate", "nd_img", (x) => x.upsample(2)],
    ["BatchNorm3d", "nd_vol", (x) => new nn.BatchNormND(2).forward(x)],
  ];
  for (const [name, src, fn] of grads) {
    out.set(`ndim::grad::${name}`, () => {
      const x = nd(src, true);
      seeded(fn(x)).backward();
      return gradOf(x, name);
    });
  }

  out.set("ndim::torch.matmul", () => flat().mm(flat().transpose()));
  out.set("ndim::torch.reshape", () => flat().reshape([4, 2]));
  out.set("ndim::torch.unsqueeze", () => flat().unsqueeze(1));
  out.set("ndim::torch.masked_fill", () => flat().maskedFill(mask(), -1.0));
  out.set("ndim::x.masked_fill", () => flat().maskedFill(mask(), -1.0));
  out.set("ndim::x.index_select",
    () => flat().indexSelect(0, Tensor.from([1, 0], [2], false, "int64")));
  out.set("ndim::x.masked_select", async () => flat().maskedSelect(mask()));
  out.set("ndim::x.repeat_interleave",
    () => flat().ravel().repeatInterleave(2));

  addHighRank(out, inp);

  // `webgpu::` 중 입력이 골든에 실린 것들. 나머지는 위와 같은 이유로 닿을 수 없다.
  out.set("webgpu::F.pad(랭크4)", () => inp.get("img").pad(-1, 1, 2));
  out.set("webgpu::F.pad(랭크4, 값)",
    () => inp.get("img").pad(-1, 2, 1, -1.5).pad(-2, 1, 0, -1.5));
  // `seq_x` 를 (N, C, L) 로 돌린 것. 골든이 그렇게 만든다.
  // **잎이어야 기울기가 쌓인다.** `permute` 결과는 파생 텐서라 그대로 쓰면
  // "기울기가 안 왔다" 로 죽는다 — 구현이 아니라 케이스 탓으로.
  const wseq = (g = false) => {
    const t = inp.get("seq_x").permute([1, 2, 0]);
    return g ? asLeaf(t) : t;
  };
  const ck1 = () => inp.get("ck1");
  const vol = (g = false) => inp.get("vol5", g);
  const ck3 = () => inp.get("ck3");
  out.set("webgpu::F.conv1d", () => wseq().conv1d(ck1(), null, 1, 1));
  out.set("webgpu::F.conv1d(스트라이드2)", () => wseq().conv1d(ck1(), null, 2, 1));
  out.set("webgpu::F.max_pool1d", () => wseq().maxPool1d(2));
  out.set("webgpu::F.conv3d", () => vol().conv3d(ck3(), null, 1, 1));
  out.set("webgpu::F.max_pool3d", () => vol().maxPool3d(2));
  out.set("webgpu::Upsample(최근접)", () => inp.get("img").upsample(2));
  out.set("webgpu::BatchNorm3d(학습)",
    () => new nn.BatchNormND(2).forward(vol()));

  out.set("webgpu::grad::Upsample", () => {
    const x = inp.get("img", true);
    x.upsample(2).sum().backward();
    return gradOf(x, "Upsample");
  });
  out.set("webgpu::grad::max_pool3d", () => {
    const x = vol(true);
    x.maxPool3d(2).sum().backward();
    return gradOf(x, "max_pool3d");
  });
  out.set("webgpu::grad::BatchNorm3d", () => {
    const x = vol(true);
    new nn.BatchNormND(2).forward(x).sum().backward();
    return gradOf(x, "BatchNorm3d");
  });
  for (const [which, tag] of [["x", "x"], ["w", "w"]] as const) {
    out.set(`webgpu::grad::conv1d/${tag}`, () => {
      const x = wseq(true);
      const k = inp.get("ck1", true);
      x.conv1d(k, null, 1, 1).sum().backward();
      return gradOf(which === "x" ? x : k, `conv1d/${tag}`);
    });
    out.set(`webgpu::grad::conv3d/${tag}`, () => {
      const x = vol(true);
      const k = inp.get("ck3", true);
      x.conv3d(k, null, 1, 1).sum().backward();
      return gradOf(which === "x" ? x : k, `conv3d/${tag}`);
    });
  }

  out.set("webgpu::grad::pad_sequence", () => {
    const a = Tensor.from([1, 2, 3, 4], [2, 2], true);
    const b = Tensor.from([5, 6], [1, 2], true);
    seeded(Tensor.padSequence([a, b])).backward();
    return gradOf(a, "pad_sequence");
  });
}

const DTYPES: readonly DType[] = ["float32", "int64", "bool"];
const BIN_OPS = ["+", "-", "*", "/"] as const;
const OP_NAME: Readonly<Record<typeof BIN_OPS[number], string>> = {
  "+": "add", "-": "sub", "*": "mul", "/": "div",
};

/**
 * dtype 승격. **값이 아니라 어떤 형이 나오는가**를 묻는다.
 *
 * 거부하는 조합(뺄셈에 bool 이 낀 것)은 예외의 종류를 답으로 적는다 — 거부하는 것도
 * 명세이기 때문이다. 골든은 float64 가 빠진 세 종이다: 우리에게 배정도가 없다.
 */
function addDType(out: Map<string, Case>): void {
  const make = (d: DType) =>
    Tensor.from(d === "bool" ? [1, 0] : [1, 2], [2], false, d);

  const verdictOf = (fn: () => Tensor): string => {
    try {
      return dtypeName(fn().dtype);
    } catch (err) {
      return `<${err instanceof Error ? err.constructor.name : "?"}>`;
    }
  };

  for (const a of DTYPES) {
    for (const b of DTYPES) {
      for (const op of BIN_OPS) {
        out.set(`dtype::${a} ${op} ${b}`, () =>
          verdictOf(() => make(a).binary(OP_NAME[op] ?? "add", make(b))));
      }
    }
  }
  // 파이썬 스칼라는 **약하다** — 범주만 올리고 폭은 안 건드린다. 폭이 범주마다
  // 하나뿐인 여기서는 결과가 "높은 범주" 하나로 정리된다.
  //
  // **형을 값에서 유추하면 안 된다.** 파이썬의 `2.0` 은 float 인데 JS 의 `2.0` 은
  // 그냥 `2` 이고 `Number.isInteger` 가 참이다. 그 구분이 언어에 없으므로 여기서
  // 적어 준다 — 유추에 맡기면 `int64 + 파이썬 float` 이 조용히 int64 가 된다.
  const scalars: [string, number, DType][] = [
    ["파이썬 int", 2, "int64"],
    ["파이썬 float", 2, "float32"],
    ["파이썬 bool", 1, "bool"],
  ];
  for (const a of DTYPES) {
    for (const [label, value, kind] of scalars) {
      for (const op of BIN_OPS) {
        out.set(`dtype::${a} ${op} ${label}`, () =>
          verdictOf(() =>
            make(a).binary(OP_NAME[op] ?? "add",
              Tensor.from([value], [], false, kind))));
      }
    }
  }
}

/**
 * `print(t)` 가 진짜와 같은가. 값이 아니라 **글자**를 본다.
 *
 * 배우는 사람이 가장 많이 하는 일이 이것이고, 다르게 찍히면 교재의 예시와 화면이
 * 안 맞는다.
 */
function addRepr(out: Map<string, Case>): void {
  const t = (v: number[], shape?: number[], grad = false, d: DType = "float32") =>
    Tensor.from(v, shape ?? [v.length], grad, d);

  const cases: [string, () => Promise<string> | string][] = [
    ["스칼라", async () => t([3.14], []).repr()],
    ["정수값 float", async () => t([1, 2, 3]).repr()],
    ["소수", async () => t([0.1, 0.25]).repr()],
    ["음수 섞임", async () => t([-1.5, 2.0, -0.25]).repr()],
    ["2차원", async () => t([1, 2, 3, 4], [2, 2]).repr()],
    ["3차원", async () => Tensor.zeros([2, 1, 3]).repr()],
    ["정수", async () => t([1, 2, 3], undefined, false, "int64").repr()],
    ["불리언", async () => t([1, 0], undefined, false, "bool").repr()],
    ["빈 텐서", async () => t([], [0]).repr()],
    ["큰 값·작은 값", async () => t([1e6, 2e-6]).repr()],
    ["긴 1차원 줄바꿈", async () => Tensor.arange(30).repr()],
    ["requires_grad", async () => t([1, 2], undefined, true).repr()],
    ["비잎 노드 grad_fn",
      async () => t([1], undefined, true).binary("mul", Tensor.full([], 2)).repr()],
    ["합계 grad_fn", async () => t([1, 2], undefined, true).sum().repr()],
    ["relu grad_fn", async () => t([-1, 2], undefined, true).unary("relu").repr()],
    ["Size", () => t([1, 2, 3, 4], [2, 2]).sizeRepr()],
  ];
  for (const [name, fn] of cases) out.set(`repr::${name}`, fn);
}

/**
 * 선형대수. **CPU 를 한 번 왕복하므로 전부 비동기다.**
 *
 * 기울기는 닫힌 꼴이 있는 것만 있다. `qr`·`svd`·`pinverse`·`lstsq` 는 값만 준다 —
 * torch 는 미분하는데 우리는 안 한다. 유도가 까다롭고 틀리면 조용히 틀리므로,
 * 없는 것을 시끄럽게 둔다.
 */
function addLinalg(out: Map<string, Case>): void {
  const mat = (g = false) => Tensor.from([4, 1, 2, 3], [2, 2], g);
  const sym = (g = false) => Tensor.from([4, 1, 1, 3], [2, 2], g); // 대칭 양정부호
  const vec = (g = false) => Tensor.from([1, 2], [2], g);

  const value: [string, () => Promise<Tensor>][] = [
    ["det", async () => mat().det()],
    ["logdet", async () => sym().logdet()],
    ["slogdet/부호", async () => (await mat().slogdet()).sign],
    ["slogdet/로그", async () => (await mat().slogdet()).logabs],
    ["inverse", async () => mat().inverse()],
    ["pinverse", async () => mat().pinverse()],
    ["matrix_power", async () => mat().matrixPower(3)],
    ["matrix_power(음수)", async () => mat().inverse()],
    ["cholesky", async () => sym().cholesky()],
    ["solve", async () => mat().solve(vec())],
    ["matrix_rank", async () => mat().matrixRank()],
    ["lstsq", async () => mat().lstsq(vec())],
    ["eigh/고윳값", async () => (await sym().eigh()).values],
    ["linalg.det", async () => mat().det()],
    ["linalg.inv", async () => mat().inverse()],
    ["qr/R", async () => (await mat().qr()).r],
    // **부호 규약이 구현마다 다르다.** 열 부호를 뒤집어도 같은 분해라 절댓값으로 묻는다.
    ["qr/|Q|", async () => (await mat().qr()).q.abs()],
    ["svd/|U|", async () => (await mat().svd()).u.abs()],
    ["svd/S", async () => (await mat().svd()).s],
    ["svd/|Vh|", async () => (await mat().svd()).vt.abs()],
  ];
  for (const [name, fn] of value) out.set(`linalg::${name}`, fn);

  const grads: [string, (g: boolean) => Tensor, (x: Tensor) => Promise<Tensor>][] = [
    ["det", mat, async (x) => x.det()],
    ["logdet", sym, async (x) => x.logdet()],
    ["slogdet", mat, async (x) => (await x.slogdet()).logabs],
    ["inverse", mat, async (x) => x.inverse()],
    ["cholesky", sym, async (x) => x.cholesky()],
    ["matrix_power", mat, async (x) => x.matrixPower(3)],
  ];
  for (const [name, src, fn] of grads) {
    out.set(`linalg::grad::${name}`, async () => {
      const x = src(true);
      seeded(await fn(x)).backward();
      return gradOf(x, name);
    });
  }

  for (const [which, tag] of ["a", "b"].entries()) {
    out.set(`linalg::grad::solve/${tag}`, async () => {
      const a = mat(true);
      const b = vec(true);
      const res = await a.solve(b);
      res.mul(Tensor.from([1, 2], [2])).sum().backward();
      const leaf = which === 0 ? a : b;
      return gradOf(leaf, `solve/${tag}`);
    });
  }
}

/** `tests/cases.py` 의 inplace_cases 가 쓰는 입력. */
const IP_PLAIN = [1.0, 4.0, 9.0, 2.0];
const IP_SMALL = [0.5, 0.8, 0.3, 0.9]; // 정의역이 좁은 것들용

/** 정의역이 좁아 `small` 을 받아야 하는 것들. */
const IP_NARROW = new Set(["log", "log2", "log10", "sqrt", "rsqrt", "log1p"]);

/**
 * 제자리 연산.
 *
 * **되돌려받은 것이 아니라 원본을 본다** — 새 텐서를 만들어 돌려주면 제자리가 아니고,
 * 그래도 반환값만 보는 검사는 통과한다.
 */
function addInplace(out: Map<string, Case>): void {
  const each = (name: string, fn: (x: Tensor) => unknown, src = IP_PLAIN) => {
    out.set(`inplace::${name}`, () => {
      const x = Tensor.from(src, [src.length]);
      fn(x);
      return x;
    });
  };

  each("add_", (x) => x.add_(1));
  each("add_(alpha)", (x) => x.add_(1, 2));
  each("sub_", (x) => x.sub_(1));
  each("mul_", (x) => x.mul_(2));
  each("div_", (x) => x.div_(2));
  each("pow_", (x) => x.pow_(2));
  each("neg_", (x) => x.inplaceUnary("neg"));
  each("zero_", (x) => x.zero_());
  each("fill_", (x) => x.fill_(7));
  each("clamp_", (x) => x.clamp_(2, 5));
  each("clip_", (x) => x.clip_(2, 5));
  // **이어 부르기가 진짜 시험이다.** 돌려준 것이 자기 자신이어야 이어진다.
  each("이어 부르기", (x) => x.mul_(2).add_(1).clamp_(0, 10));

  for (const name of ["abs", "sqrt", "exp", "log", "sin", "cos", "tan", "tanh",
    "sigmoid", "relu", "erf", "floor", "ceil", "round", "sign", "reciprocal",
    "square", "trunc", "frac", "neg", "rsqrt", "log2", "log10", "expm1",
    "log1p", "sinh", "cosh"]) {
    each(`${name}_`, (x) => x.inplaceUnary(name),
      IP_NARROW.has(name) ? IP_SMALL : IP_PLAIN);
  }

  /**
   * **자매만 거절하는 자리다.**
   *
   * 골든이 값을 안 묻고 "문서에 적은 대로 굴었는가" 를 묻는다 — torch 는 성공이
   * 정답이고 자매는 거절이 정답이라, 값으로 물으면 영원히 갈린 채로 남기 때문이다.
   * borch.ts 는 자매가 아니다. 버퍼를 같이 쓰므로 번지고, 그래서 우리 답은 torch 와
   * 같은 "기대대로" 다.
   */
  out.set("inplace::뷰 전파=브라우저는거절", () => {
    try {
      const a = Tensor.arange(4);
      a.view(2, 2).add_(10);
    } catch (err) {
      return `뜻밖의 거절 <${err instanceof Error ? err.constructor.name : "?"}>`;
    }
    return "기대대로";
  });

  // 기울기가 켜진 잎은 셋 다 거절한다.
  out.set("inplace::잎 제자리 수정=거절", () => {
    const x = Tensor.from(IP_PLAIN, [4], true);
    try {
      x.add_(1);
    } catch {
      return "기대대로 거절";
    }
    return "뜻밖의 성공";
  });

  // `no_grad` 안에서는 잎도 고칠 수 있다 — 옵티마이저가 실제로 그렇게 한다.
  out.set("inplace::no_grad 안에서는 된다", () => {
    const x = Tensor.from(IP_PLAIN, [4], true);
    noGrad(() => x.add_(1));
    return x;
  });
}

/**
 * **기울기**를 대조하는 표.
 *
 * 순방향만 맞고 역방향이 틀리면 "학습은 돌고 손실도 내려가는데 값이 다른" 상태가
 * 된다. 코어가 BatchNorm 으로 오래 겪은 종류이고, 값 대조로는 안 잡힌다.
 *
 * 여기 없는 것: 정렬(`topk`·`sort`·`median`), `einsum`, `pad_sequence`, `fmod`,
 * dtype 이 필요한 `float()`·`double()`·`nll_loss`·`cross_entropy`, 그리고 합성곱.
 */
function addGrad(out: Map<string, Case>, inp: Inputs): void {
  const x1 = (g = false) => inp.get("x1", g);
  const xp = (g = false) => inp.get("xp", g);
  const x2 = (g = false) => inp.get("x2", g);

  /** 잎 하나를 만들고, 접어서 흘리고, 그 잎의 기울기를 낸다. */
  const one = (name: string, src: (g: boolean) => Tensor, fn: (x: Tensor) => Tensor) => {
    out.set(`grad::${name}`, () => {
      const x = src(true);
      fn(x).sum().backward();
      return gradOf(x, name);
    });
  };

  /**
   * 자리마다 다른 가중치를 곱해 받는다. 그냥 `sum()` 이면 기울기가 전부 1 이라
   * `movedim` 이 축을 뒤바꿔도, `tile` 이 조각을 엉뚱하게 겹쳐도 통과한다.
   */
  const weighted = (name: string, make: () => Tensor[], fn: (...xs: Tensor[]) => Tensor,
                    which = 0) => {
    out.set(`grad::${name}`, () => {
      const leaves = make();
      const res = fn(...leaves);
      seeded(res).backward();
      const leaf = leaves[which];
      if (!leaf) throw new Error(`${name}: 잎 ${which} 가 없다`);
      return gradOf(leaf, name);
    });
  };

  for (const name of ["exp", "abs", "sin", "cos", "tan", "sinh", "cosh", "tanh",
    "erf", "square", "relu", "sigmoid", "gelu", "silu", "elu", "neg"]) {
    one(name, x1, (x) => x.unary(name));
  }
  for (const name of ["log", "log2", "log10", "sqrt", "rsqrt", "reciprocal"]) {
    one(name, xp, (x) => x.unary(name));
  }
  one("leaky_relu", x1, (x) => x.leakyRelu(0.1));
  one("pow2", x1, (x) => x.powScalar(2));

  const mat = (g = false) => Tensor.from([1, 2, 3, 4, 5, 6, 7, 8, 9], [3, 3], g);
  const vec = (g = false) => Tensor.from([1, 2, 3, 4], [4], g);
  // 0 이 섞인 입력. 흔한 유도(out/x)는 여기서 나눗셈이 터져 조용히 NaN 을 흘린다.
  const zeroed = (g = false) => Tensor.from([2, 0, 3, 4], [4], g);
  const short = (g = false) => Tensor.from([1, 5, 2], [3], g);

  weighted("tril", () => [mat(true)], (x) => x.tril());
  weighted("triu(k=1)", () => [mat(true)], (x) => x.triu(1));
  weighted("diag(2차원)", () => [mat(true)], (x) => x.diag());
  weighted("diag(1차원)", () => [short(true)], (x) => x.diag());
  weighted("trace", () => [mat(true)], (x) => x.trace());
  weighted("cumprod", () => [vec(true)], (x) => x.cumprod(0));
  weighted("cumprod(0포함)", () => [zeroed(true)], (x) => x.cumprod(0));
  weighted("cumprod(2차원)", () => [mat(true)], (x) => x.cumprod(1));
  weighted("tile", () => [vec(true)], (x) => x.tile(2));
  weighted("tile(2차원)", () => [mat(true)], (x) => x.tile(2, 3));
  weighted("movedim", () => [mat(true)], (x) => x.movedim(0, 1));
  weighted("repeat_interleave", () => [vec(true)], (x) => x.repeatInterleave(3));
  weighted("repeat_interleave(dim)", () => [mat(true)], (x) => x.repeatInterleave(2, 0));

  one("sum", x2, (x) => x.sum());
  one("sum(dim)", x2, (x) => x.sumDim(1));
  one("mean", x2, (x) => x.mean());
  one("mean(dim)", x2, (x) => x.mean(0));
  one("softmax", x2, (x) => x.softmax(-1));
  one("log_softmax", x2, (x) => x.logSoftmax(-1));
  one("cumsum", x1, (x) => x.cumsum(0));
  one("flip", x1, (x) => x.flip(0));
  one("clamp", x1, (x) => x.clamp(-0.5, 0.5));
  one("norm", x2, (x) => x.norm());
  one("normalize", x2, (x) => x.normalize(1));
  one("gather", x2, (x) => x.gather(1, inp.get("idx2")));
  one("narrow", x1, (x) => x.narrow(0, 1, 3));
  one("split", x1, (x) => piece(x.splitSize(0, 2), 1));
  one("chunk", x1, (x) => piece(x.chunk(3), 2));
  one("unbind", x2, (x) => piece(x.unbind(0), 1));
  one("index_select", x2, (x) => x.indexSelect(0, Tensor.from([2, 0], [2])));
  one("pad", x2, (x) => x.pad(-1, 1, 1));
  one("prod", xp, (x) => x.prod());

  // 인덱싱 — torch 코드가 가장 자주 하는 일이고, 자르기와 같은 이유로 그래프를 잇는다.
  one("idx[0]", x2, (x) => x.select(0, 0));
  one("idx[-1]", x2, (x) => x.select(0, (x.shape[0] ?? 1) - 1));
  one("idx[1:3]", x1, (x) => x.narrow(0, 1, 2));
  one("idx[:, 1]", x2, (x) => x.select(1, 1));
  one("idx[1, 2]", x2, (x) => x.select(0, 1).select(0, 2));
  one("idx[0:2, 1:3]", x2, (x) => x.narrow(0, 0, 2).narrow(1, 1, 2));
  one("idx[목록]", x2, (x) => x.indexSelect(0, Tensor.from([2, 0], [2])));

  // 이어 붙이기·쌓기 — DataLoader 의 collate 가 이것 위에 선다.
  const twice = (x: Tensor) => x.binary("mul", Tensor.full([], 2));
  const thrice = (x: Tensor) => x.binary("mul", Tensor.full([], 3));
  one("cat", x1, (x) => Tensor.cat([x, twice(x)]));
  one("cat(dim=1)", x2, (x) => Tensor.cat([x, twice(x)], 1));
  one("stack", x1, (x) => Tensor.stack([x, thrice(x)]));
  one("stack(dim=1)", x2, (x) => Tensor.stack([x, thrice(x)], 1));

  one("메서드 x.abs()", x1, (x) => x.abs());
  one("메서드 x.exp()", x1, (x) => x.exp());
  one("메서드 x.sqrt()", xp, (x) => x.sqrt());

  one("LayerNorm", x2, (x) => x.layerNorm(-1));
  one("F.layer_norm", x2, (x) => x.layerNorm(-1));
  one("BatchNorm1d", x2, (x) => x.batchNorm(0));
  one("F.linear", x2, (x) => x.linear(inp.get("x2")));
  one("Softmax(층)", x2, (x) => x.softmax(-1));
  one("LogSoftmax(층)", x2, (x) => x.logSoftmax(-1));
  one("LeakyReLU(층)", x1, (x) => x.leakyRelu(0.1));
  one("ELU(층)", x1, (x) => x.unary("elu"));
  one("SiLU(층)", x1, (x) => x.unary("silu"));
  one("Identity", x1, (x) => x.clone());
  one("Unflatten", x1, (x) => x.unflatten(0, [3, 2]));

  one("where", x1, (x) => x.where(positive(x), x.binary("mul", Tensor.full([], 0.1))));
  one("masked_fill", x1, (x) => x.maskedFill(positive(x), -1.0));
  one("clone", x1, (x) => x.clone());
  one("permute", x2, (x) => x.permute([1, 0]));
  one("squeeze", x1, (x) => x.unsqueeze(0).squeezeAll());
  one("max(dim)", x2, (x) => x.amax(1));
  one("min(dim)", x2, (x) => x.amin(1));
  one("var", x1, (x) => x.variance());
  one("std", x1, (x) => x.std());

  // **같은 번호가 여러 번 나오면 그 행에 기울기가 쌓여야 한다.**
  out.set("grad::embedding(중복 번호)", () => {
    const w = asLeaf(inp.get("w0").narrow(0, 0, 5)); // (5, 6)
    w.indexSelect(0, Tensor.from([0, 2, 0, 4], [4])).sum().backward();
    return gradOf(w, "embedding");
  });

  // 이항 — 양쪽 잎을 다 본다. 한쪽만 보면 반대쪽 끊김을 못 잡는다.
  const pairs: [string, (a: Tensor, b: Tensor) => Tensor, () => [Tensor, Tensor]][] = [
    ["add", (a, b) => a.add(b), () => [x1(true), x1(true)]],
    ["sub", (a, b) => a.sub(b), () => [x1(true), x1(true)]],
    ["mul", (a, b) => a.mul(b), () => [x1(true), x1(true)]],
    ["div", (a, b) => a.div(b), () => [xp(true), xp(true)]],
    // **오른쪽도 잎이어야 한다.** `x1().neg()` 은 파생 텐서라 기울기가 안 쌓이고,
    // 그러면 `/b` 케이스가 "기울기가 안 왔다" 로 죽는다 — 구현이 아니라 케이스 탓으로.
    ["maximum", (a, b) => a.binary("maximum", b), () => [x1(true), asLeaf(x1().neg())]],
    ["minimum", (a, b) => a.binary("minimum", b), () => [x1(true), asLeaf(x1().neg())]],
    ["matmul", (a, b) => a.mm(b), () => [x2(true), asLeaf(x2().transpose())]],
    ["l1_loss", (a, b) => a.l1Loss(b), () => [x1(true), x1(true)]],
    ["mse_loss", (a, b) => a.mseLoss(b), () => [x1(true), x1(true)]],
    ["smooth_l1_loss", (a, b) => a.smoothL1Loss(b), () => [x1(true), x1(true)]],
    ["cosine_similarity", (a, b) => a.cosineSimilarity(b),
      () => [x2(true), asLeaf(x2().binary("mul", Tensor.full([], 2)))]],
  ];
  for (const [name, fn, make] of pairs) {
    for (const [which, tag] of ["a", "b"].entries()) {
      out.set(`grad::${name}/${tag}`, () => {
        const leaves = make();
        fn(leaves[0], leaves[1]).sum().backward();
        const leaf = leaves[which];
        if (!leaf) throw new Error(`${name}: 잎 ${tag} 가 없다`);
        return gradOf(leaf, `${name}/${tag}`);
      });
    }
  }
  // 합성곱 — **역방향을 직접 짠 자리다.** 입력·가중치·편향 셋 다 본다.
  // 걸음 2 를 같이 보는 것은 의도다: 기울기 사이에 0 을 끼우는 경로가 거기서만 돈다.
  const convGrad = (
    label: string, which: "x" | "w" | "b", stride: number, padding: number,
    useBias: boolean,
  ) => {
    out.set(`grad::${label}/${which}`, () => {
      const x = inp.get("img", true);
      const k = inp.get("cw", true);
      const b = useBias ? inp.get("cb", true) : null;
      x.conv2d(k, b, stride, padding).sum().backward();
      const leaf = which === "x" ? x : which === "w" ? k : b;
      if (!leaf) throw new Error(`${label}: 잎 ${which} 가 없다`);
      return gradOf(leaf, `${label}/${which}`);
    });
  };
  for (const which of ["x", "w", "b"] as const) convGrad("conv2d", which, 1, 1, true);
  for (const which of ["x", "w"] as const) {
    convGrad("conv2d(패딩0)", which, 1, 0, false);
    convGrad("conv2d(스트라이드2)", which, 2, 1, false);
  }
  one("max_pool2d", () => inp.get("img", true), (x) => x.maxPool2d(2));

  // **평균 풀링의 역방향을 아무도 안 묻고 있었다.**
  //
  // 이 라이브러리에서 평균 풀링의 역방향이 아예 안 도는 것을 통합 시험이 잡았다 —
  // 쓰지 않는 바인딩이 레이아웃에서 빠지면서 커맨드 버퍼가 통째로 무효가 됐는데
  // WebGPU 는 그것을 예외로 안 던진다. 손실이 ln 10 에 앉아 있는데 ms/step 은 계속
  // 나왔다. 표가 이것을 물었다면 통합까지 갈 일이 아니었다.
  //
  // 균일하게 접으면 안 된다. max 는 이긴 자리 하나에 흘리고 avg 는 창 전체에 1/n 씩
  // 나누는데, 상류가 전부 1 이면 **입력 기울기의 합이 같아서** 둘을 바꿔 놔도 통과한다.
  const pooled = (name: string, fn: (x: Tensor) => Tensor) => {
    out.set(`grad::${name}`, () => {
      const x = inp.get("img", true);
      seeded(fn(x)).backward();
      return gradOf(x, name);
    });
  };
  pooled("avg_pool2d", (x) => x.avgPool2d(2));
  pooled("avg_pool2d(스트라이드1)", (x) => x.avgPool2d(2, 1));
  pooled("adaptive_avg_pool2d", (x) => x.adaptiveAvgPool(1));
  pooled("max_pool2d(가중치)", (x) => x.maxPool2d(2));

  // **평균·분산이 그래프 안에 있어야 한다.** 밖으로 빼면 입력 기울기가 어긋나고
  // weight 에는 아예 안 온다(None). 그래서 둘 다 본다.
  for (const which of ["x", "weight"] as const) {
    out.set(`grad::BatchNorm2d/${which}`, () => {
      const x = inp.get("img", true);
      const bn = new nn.BatchNormND(3);
      bn.forward(x).sum().backward();
      const leaf = which === "x" ? x : bn.weight;
      return gradOf(leaf, `BatchNorm2d/${which}`);
    });
    // **위의 `sum()` 이 BatchNorm 역방향의 절반을 가린다.** 입력 기울기는 곧바로
    // 오는 항 하나와, 평균·분산이 입력에 의존해서 생기는 보정항 둘로 되어 있다.
    // 상류가 전부 1 이면 그 보정항 둘이 정확히 상쇄되어(기대값 4.7e-10) 위 케이스는
    // 보정항을 아예 안 묻는다. 자리마다 다른 가중치를 주면 상쇄가 깨진다.
    out.set(`grad::BatchNorm2d(가중치)/${which}`, () => {
      const x = inp.get("img", true);
      const bn = new nn.BatchNormND(3);
      seeded(bn.forward(x)).backward();
      const leaf = which === "x" ? x : bn.weight;
      return gradOf(leaf, `BatchNorm2d(가중치)/${which}`);
    });
  }
  // **뽑기 계열이 그래프를 끊기 가장 쉬운 자리다.** 값만 떼어 돌려주면 뽑은 자리로
  // 기울기가 안 가고 학습이 조용히 멈춘다.
  one("topk", x1, (x) => x.topk(3).values);
  one("sort", x1, (x) => x.sort(0).values);
  one("sort(내림차순)", x1, (x) => x.sort(0, true).values);
  // **꺾이는 자리에서 흘리는가.** torch 의 relu 는 입력이 정확히 0 이면 기울기를
  // 0 으로 준다. `x1` 은 무작위 정규분포라 0 이 없어서 골든이 이 자리를 안 보고
  // 있었고, 이 라이브러리가 거기서 1 을 흘리고 있었다.
  weighted("relu(0에서)",
    () => [Tensor.from([-1, 0, 1, 0], [4], true)], (x) => x.unary("relu"));

  weighted("median()", () => [vec(true)], (x) => x.median().values);
  weighted("median(dim)", () => [mat(true)], (x) => x.median(1).values);

  // 골든이 이항 형태로 굳혔으므로 이름 뒤에 잎 표시가 붙는다.
  one("L1Loss(층)/a", x1, (x) => x.l1Loss(inp.get("x1")));
  one("SmoothL1Loss(층)/a", x1, (x) => x.smoothL1Loss(inp.get("x1")));
  one("BCEWithLogitsLoss/a", x1, (x) => x.bceWithLogits(inp.get("x1")));

  // **형만 바꾼 자리에서 그래프가 끊긴 적이 있다.** 코어의 `.float()` 이 결과에
  // requires_grad 를 붙여놓고 부모를 안 달아서, backward 가 예외 없이 돌고 잎의
  // grad 만 None 으로 남았다. 경고도 예외도 없이.
  one("float()", x1, (x) => x.to("float32"));
  // 골든이 이 둘을 **가중치를 곱하는** 무리에 넣었다. 그냥 sum() 이면 기울기가 전부
  // 1 이라 자리가 뒤바뀌어도 안 걸린다 — 그러라고 가중치를 준 것이다.
  weighted("einsum(ij->i)", () => [mat(true)], (x) => einsum("ij->i", x));
  weighted("fmod(%)", () => [vec(true)], (x) => x.fmod(2));
  one("nll_loss", x2,
    (x) => x.logSoftmax(-1).nllLoss(Tensor.from([0, 1, 2], [3], false, "int64")));
  one("cross_entropy", x2,
    (x) => x.crossEntropy(Tensor.from([0, 1, 2], [3], false, "int64")));

  // 자매만 거절하는 자리. 우리는 배정도가 없지만 **형을 바꾸는 것 자체는 되고**,
  // float32 로 되돌아오는 것이라 torch 처럼 성공이 정답이다.
  out.set("grad::double()=브라우저는거절", () => {
    try {
      const x = x1(true);
      x.to("float32").sum().backward();
      gradOf(x, "double()");
    } catch (err) {
      return `뜻밖의 거절 <${err instanceof Error ? err.constructor.name : "?"}>`;
    }
    return "기대대로";
  });

  const mat2 = () => Tensor.from([2, 0, 1, 1, 3, 2, 0, 1, 4], [3, 3], true);
  for (const [which, tag] of ["a", "b"].entries()) {
    out.set(`grad::einsum(ij,jk->ik)/${tag}`, () => {
      const leaves = [mat(true), mat2()];
      const a = leaves[0];
      const b = leaves[1];
      if (!a || !b) throw new Error("einsum: 잎이 없다");
      seeded(einsum("ij,jk->ik", a, b)).backward();
      const leaf = leaves[which];
      if (!leaf) throw new Error(`einsum: 잎 ${tag} 가 없다`);
      return gradOf(leaf, `einsum/${tag}`);
    });
    out.set(`grad::pad_sequence/${tag}`, () => {
      const leaves = [
        Tensor.from([1, 2, 3, 4], [4], true),
        Tensor.from([1, 5, 2], [3], true),
      ];
      const a = leaves[0];
      const b = leaves[1];
      if (!a || !b) throw new Error("pad_sequence: 잎이 없다");
      seeded(Tensor.padSequence([a, b])).backward();
      const leaf = leaves[which];
      if (!leaf) throw new Error(`pad_sequence: 잎 ${tag} 가 없다`);
      return gradOf(leaf, `pad_sequence/${tag}`);
    });
  }
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
    ["Linear 입력 차원 불일치", () => {
      const layer = new nn.Linear(4, 2);
      layer.forward(Tensor.zeros([3, 5]));
    }, "shapes cannot be multiplied"],
    ["Conv2d 채널 불일치",
      () => Tensor.zeros([1, 3, 8, 8]).conv2d(Tensor.zeros([4, 1, 3, 3])),
      null],
    ["leaf 제자리 수정", () => {
      const x = Tensor.from([1, 2, 3], [3], true);
      x.add_(1);
    }, null],
    ["인덱스 범위 초과", () => Tensor.zeros([3]).select(0, 5), "out of bounds"],
    ["정수 텐서에 requires_grad",
      () => Tensor.from([1, 2, 3], [3], true, "int64"), null],
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

  // CPU 를 왕복하는 것들. 흐르는지 묻는 것은 같고, 결과를 기다려야 할 뿐이다.
  const slow: [string, () => Tensor, (x: Tensor) => Promise<Tensor>][] = [
    ["median", vec, async (x) => x.median().values],
    ["msort", vec, async (x) => x.msort()],
    ["masked_select", vec,
      async (x) => x.maskedSelect(Tensor.from(F_MASK, [4]))],
    ["einsum", mat, async (x) => einsum("ij->i", x)],
    ["det", sym, async (x) => x.det()],
    ["logdet", sym, async (x) => x.logdet()],
    ["inverse", sym, async (x) => x.inverse()],
    ["cholesky", sym, async (x) => x.cholesky()],
  ];
  for (const [name, leaf, fn] of slow) {
    out.set(`flow::${name}`, async () => asksSlow(leaf(), fn));
  }
}

/** `asks` 와 같은 질문이되 결과를 기다린다. */
async function asksSlow(
  leaf: Tensor,
  fn: (x: Tensor) => Promise<Tensor>,
): Promise<string> {
  const result = await fn(leaf);
  const flow = result.requiresGrad ? "흐름" : "안흐름";
  try {
    result.sum().backward();
  } catch {
    return `${flow}/역전파거절`;
  }
  return `${flow}/${leaf.grad !== null ? "기울기있음" : "조용히None"}`;
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
  add("cummax", (x) => x.cummax(0).values, tie);
  add("cummin", (x) => x.cummin(0).values, tie);
  add("kthvalue", (x) => x.kthvalue(2).values, tie);
  add("msort", (x) => x.msort(), mat, [2, 3], false);
  add("diff", (x) => x.diff(), tie);
  add("diff(n=2)", (x) => x.diff(2), tie);

  // **축을 받는 것은 값으로 묻는다.** 기울기로만 물으면 축을 통째로 무시해도
  // 통과한다 — `sum(dim=1).sum()` 과 `sum().sum()` 의 기울기가 둘 다 전부 1 이라
  // 답이 같기 때문이다. 파이썬 결속이 실제로 그 구멍으로 축을 버리고 있었다.
  add("sum(dim)", (x) => x.sumDim(1), mat, [2, 3]);
  add("sum(dim0)", (x) => x.sumDim(0), mat, [2, 3]);
  add("sum(dim,keepdim)", (x) => x.sumDim(1, true), mat, [2, 3]);
  add("norm(dim)", (x) => x.square().sumDim(1).sqrt(), mat, [2, 3]);
  add("norm(p=1,dim)", (x) => x.abs().sumDim(0), mat, [2, 3]);

  // 기울기 케이스가 없는 것들. 골든도 값만 굳혔다.
  out.set("reduce::aminmax/최소", () => Tensor.from(tie).amin());
  out.set("reduce::aminmax/최대", () => Tensor.from(tie).amax());
  // **번호를 따로 묻는다** — 값만 보면 번호가 틀려도 통과한다.
  out.set("reduce::cummax 번호", () => Tensor.from(tie).cummax(0).indices);
  out.set("reduce::cummin 번호", () => Tensor.from(tie).cummin(0).indices);
  out.set("reduce::kthvalue 번호", () => Tensor.from(tie).kthvalue(2).indices);
  out.set("reduce::quantile", async () => Tensor.from(tie).quantile(0.5));
  out.set("reduce::quantile(여럿)",
    async () => Tensor.from(tie).quantile([0.25, 0.75]));
  out.set("reduce::nanquantile",
    async () => Tensor.from([1, Number.NaN, 3, 5], [4]).nanquantile(0.5));
  out.set("reduce::nonzero", async () => Tensor.from([0, 1, 0, 2], [4]).nonzero());
  out.set("reduce::argwhere", async () => Tensor.from([0, 1, 0, 2], [4]).argwhere());
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
  // **랭크 3.** 축을 바꾸는 것을 2차원으로만 물으면 `(0,1)` 밖의 자리를 못 본다 —
  // 2차원에서는 어느 두 축을 골라도 답이 하나뿐이라 축 인자를 버리는 구현도 통과한다.
  // 여기서는 `permute` 로 적는다. borch.ts 의 `transpose()` 는 2차원 전용이고 축을
  // 안 받는다 — 파이썬 쪽이 두 축을 받아 이 순서를 만들어 넘긴다.
  const cube = (grad = false) => Tensor.from(seq(24), [2, 3, 4], grad);

  const value: [string, () => Tensor][] = [
    ["expand", () => col().expand(2, 3)],
    ["expand(-1)", () => col().expand(-1, 3)],
    ["expand(앞에 축 추가)", () => m().expand(2, 2, 3)],
    ["repeat", () => m().repeat(2, 1)],
    ["repeat(둘 다)", () => m().repeat(2, 3)],
    ["ravel", () => m().ravel()],
    ["swapaxes", () => m().swapaxes(0, 1)],
    ["swapdims", () => m().swapaxes(0, 1)],
    ["transpose(랭크3)", () => cube().permute([0, 2, 1])],
    ["transpose(랭크3, 0과2)", () => cube().permute([2, 1, 0])],
    ["transpose(랭크3, 음수축)", () => cube().permute([2, 1, 0])],
    ["swapdims(랭크3)", () => cube().permute([1, 0, 2])],
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
    ["argsort", () => vec().argsort(0)],
    ["sort", () => vec().sort(0).values],
    ["topk", () => vec().topk(2).values],
    ["median", () => vec().median().values],
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
  out.set("method::unique", async () => vec().unique());

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
