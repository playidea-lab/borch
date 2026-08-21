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
import * as fft from "../src/fft.js";
import { istft, stft } from "../src/fft.js";
import * as nn from "../src/nn.js";
import * as rnn from "../src/rnn.js";
import { igamma, igammac, polygamma } from "../src/special.js";
import * as optim from "../src/optim.js";
import { load, save, type Savable } from "../src/serialize.js";
import * as vision from "../src/vision.js";
import { LinAlgError } from "../src/errors.js";
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
    return Tensor.from(flat, entry.shape ?? [flat.length], { requiresGrad });
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
    ["BatchNorm2d(학습)", () => new nn.BatchNorm2d(3).call(inp.get("img"))],
    // **저장·복원 뒤의 평가 모드.** 이동 통계가 state_dict 에서 빠지면 여기서만
    // 갈린다 — 학습은 멀쩡해 보이고 추론만 틀리는, 코어가 겪은 그 결함이다.
    ["BatchNorm2d(저장→복원→eval)", () => {
      const trained = new nn.BatchNorm2d(3);
      trained.call(inp.get("img")); // 이동 통계가 갱신된다
      const fresh = new nn.BatchNorm2d(3);
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
    ["F.one_hot", () => Tensor.from([0, 2], [2], { dtype: "int64" }).oneHot(3)],
    ["F.nll_loss",
      () => x2().logSoftmax(-1).nllLoss(Tensor.from([0, 1, 2], [3], { dtype: "int64" }))],
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
    async () => Tensor.from([0, 1, 1, 3], [4], { dtype: "int64" }).bincount());

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
  return [[1, 2, 3], [4], [5, 6]].map((v) => Tensor.from(v, [v.length], { requiresGrad: grad }));
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
      const x = Tensor.from(pick(name), undefined, { requiresGrad: true });
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
          Tensor.from(plain, undefined, { requiresGrad: true }),
          Tensor.from(other, undefined, { requiresGrad: true }),
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
      const x = Tensor.from(plain, undefined, { requiresGrad: true });
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
  addContainer(out, inputs);
  addAct(out, inputs);
  addNorm(out, inputs);
  addPad(out);
  addLoss(out);
  addLazy(out);
  addShuffle(out);
  addMisc(out);
  addCell(out);
  addUnpool(out);
  addRnnTop(out, inputs);
  addOpt(out, inputs);
  addDropout(out, inputs);
  addSdpa(out, inputs);
  addModFn(out, inputs);
  addPool(out, inputs);
  addNewFn(out, inputs);
  addIndex(out, inputs);
  addNumeric(out, inputs);
  addRecent(out);
  addVision(out, inputs);
  addSeq(out, inputs);
  addEdge(out);
  addComplex(out);
  addFft(out);
  addKeepdim(out);
  addTopRest(out);
  return out;
}

/**
 * 최상위에 남아 있던 이름들 — `top::`.
 *
 * **골든의 `top::` 가운데 여기서 안 묻는 것이 둘 있다.** `device::` 는 borch.ts 에
 * 같은 것이 없고(우리 쪽 `device()` 는 어댑터를 내는 다른 함수다), `resize_as_` 는
 * 파이썬 결속이 손잡이를 갈아 끼워 해내는 것이라 TS 표면에 없다. 이름을 안 맞춰
 * 두면 러너가 "골든에 없는 이름"으로 세어 주므로, 안 쓰는 쪽을 고른 것이다.
 */
function addTopRest(out: Map<string, Case>): void {
  const P = "top::";
  const GRID = [-1.7, 0.3, 2.9, 5.5];
  const SHAPES = [0.5, 1.0, 2.0, 3.0];
  const SPOTS = [0.25, 1.5, 0.5, 4.0];
  const STEPS = [1.0, 2.0, 3.5];
  const x = (grad = false): Tensor => Tensor.from(GRID, undefined, { requiresGrad: grad });
  const a = (): Tensor => Tensor.from(SHAPES);

  // ── 최상위 전용 제자리들 ────────────────────────────────────────────
  //
  // 난수는 못 굳히지만 **`p=0` 이면 항등이라** 값이 결정적이다. 그리고 `is` 를 따로
  // 묻는다 — 같은 텐서를 안 돌려주면 이어 부르는 코드가 원본을 못 고친다.
  const holes23 = (): Tensor => Tensor.from(
    [-1.0, 0.5, Number.NaN, 0.25, Infinity, 1.0], [2, 3]);
  const img1234 = (): Tensor => Tensor.from(
    Array.from({ length: 24 }, (_, i) => i), [1, 2, 3, 4]);
  const topInplace: [string, () => Tensor, (t: Tensor) => Tensor][] = [
    ["nan_to_num_", holes23, (v) => v.nanToNum_()],
    ["dropout_", img1234, (v) => v.dropout_(0.0, true)],
    ["feature_dropout_", img1234, (v) => v.featureDropout_(0.0, true)],
    ["alpha_dropout_", img1234, (v) => v.alphaDropout_(0.0, false)],
    ["feature_alpha_dropout_", img1234, (v) => v.featureAlphaDropout_(0.0, false)],
  ];
  for (const [name, src, run] of topInplace) {
    out.set(`${P}제자리::${name}`, () => {
      const v = src();
      run(v);
      return v;
    });
    out.set(`${P}제자리::${name}(같은 텐서)`, () => {
      const v = src();
      return verdict(run(v) === v);
    });
  }
  // `feature_dropout` 은 **채널째** 떨군다 — `dropout2d` 와 같은 계산이다.
  out.set(`${P}feature_dropout(p=0)`, () => img1234().featureDropout(0.0, true));

  out.set(`${P}igamma`, () => igamma(a(), Tensor.from(SPOTS)));
  // **한 식으로 못 덮는다** — `x < a+1` 은 급수, 그 밖은 연분수다.
  out.set(`${P}igamma(큰 x)`, () =>
    igamma(a(), Tensor.from(SHAPES.map((v) => v * 8))));
  out.set(`${P}igammac`, () => igammac(a(), Tensor.from(SPOTS)));
  out.set(`${P}igamma + igammac = 1`, () =>
    igamma(a(), Tensor.from(SPOTS)).add(igammac(a(), Tensor.from(SPOTS))));
  for (const n of [0, 1, 2, 3]) {
    out.set(`${P}polygamma(${n})`, () => polygamma(n, Tensor.from(STEPS)));
  }
  out.set(`${P}constant_pad_nd`, () => x().constantPadNd([1, 2], 9.0));
  out.set(`${P}fake_quantize(per_tensor)`, () =>
    x().fakeQuantizePerTensorAffine(0.5, 0, 0, 7));
  // 영점을 옮기면 자르는 자리가 바뀐다.
  out.set(`${P}fake_quantize(zp=2)`, () =>
    x().fakeQuantizePerTensorAffine(0.5, 2, 0, 7));
  out.set(`${P}fake_quantize(per_channel)`, () =>
    Tensor.from(GRID, [2, 2]).fakeQuantizePerChannelAffine(
      Tensor.from([0.5, 0.25]), Tensor.from([0.0, 1.0]), 0, 0, 7));
  out.set(`${P}dequantize`, () => x().dequantize());

  const grad = (name: string, values: number[],
                fn: (leaf: Tensor) => Tensor): void => {
    out.set(`${P}grad::${name}`, () => {
      const leaf = Tensor.from(values, undefined, { requiresGrad: true });
      fn(leaf).sum().backward();
      return gradOf(leaf, name);
    });
  };
  grad("igamma / x", SPOTS, (t) => igamma(a(), t));
  grad("igammac / x", SPOTS, (t) => igammac(a(), t));
  grad("polygamma(1)", STEPS, (t) => polygamma(1, t));
  grad("constant_pad_nd", GRID, (t) => t.constantPadNd([1, 2], 9.0));
  // **범위 밖은 0 이다** — 반올림이 계단인데 범위 안은 곧바로 통과시킨다.
  grad("fake_quantize", GRID, (t) => t.fakeQuantizePerTensorAffine(0.5, 0, 0, 7));

  // 첫 인자로는 안 미분한다 — torch 자신이 거절한다(닫힌 꼴이 없다).
  out.set(`${P}igamma 는 a 로 안 미분한다`, () => {
    try {
      const leaf = Tensor.from(SHAPES, undefined, { requiresGrad: true });
      igamma(leaf, Tensor.from(SPOTS)).sum().backward();
      return "예외가 안 났다";
    } catch (err) {
      return err instanceof Error ? err.constructor.name : typeof err;
    }
  });
}

/**
 * `keepdim` — `keep::`.
 *
 * **축이 조용히 사라지는 자리다.** 모양이 안 맞으면 시끄럽게 멈추는데, 축 하나가
 * 빠진 모양은 브로드캐스팅에 **자주 들어맞는다** — 그러면 값만 틀린 채 끝까지 간다.
 *
 * `all`·`any`·`countNonzero` 는 축 자체가 없었다. 인자를 주면 조용히 버려지고
 * 전체 축약이 나왔다.
 */
function addKeepdim(out: Map<string, Case>): void {
  const P = "keep::";
  const GRID = [1.0, 4.0, 2.0, 3.0, 0.5, 5.0];
  const FLAGS = [1, 0, 1, 0, 0, 1];
  const g = (grad = false): Tensor =>
    Tensor.from(GRID, [2, 3], { requiresGrad: grad });
  const b = (): Tensor => Tensor.from(FLAGS, [2, 3], { dtype: "bool" });
  const shapeOf = (fn: () => Tensor): Case => () => `(${fn().shape.join(", ")})`;

  // 축을 접는 것들. 골든의 이름이 파이썬 쪽 철자라 그대로 쓴다.
  const folds: [string, (keep: boolean) => Tensor][] = [
    ["sum", (k) => g().sumDim(1, k)],
    ["mean", (k) => g().mean(1, k)],
    ["amax", (k) => g().amax(1, k)],
    ["amin", (k) => g().amin(1, k)],
    ["prod", (k) => g().prod(1, k)],
    ["logsumexp", (k) => g().logsumexp(1, k)],
    ["argmax", (k) => g().argmax(1, k)],
    ["argmin", (k) => g().argmin(1, k)],
  ];
  for (const [name, fn] of folds) {
    out.set(`${P}${name}(dim=1, keepdim)`, shapeOf(() => fn(true)));
    out.set(`${P}${name}(dim=1) 값`, () => fn(true));
  }

  // 쌍을 내는 것들 — **둘 다 축이 살아야 한다.**
  const pairs: [string, (keep: boolean) => { values: Tensor; indices: Tensor }][] = [
    ["max", (k) => g().max(1, k)],
    ["min", (k) => g().min(1, k)],
    ["median", (k) => g().median(1, k)],
  ];
  for (const [name, fn] of pairs) {
    out.set(`${P}${name}(dim=1, keepdim) 값`, () => fn(true).values);
    out.set(`${P}${name}(dim=1, keepdim) 번호`, () => fn(true).indices);
    out.set(`${P}${name}(dim=1, keepdim) 모양`, shapeOf(() => fn(true).values));
  }
  out.set(`${P}kthvalue(2, dim=1, keepdim) 값`,
    () => g().kthvalue(2, 1, true).values);
  out.set(`${P}kthvalue(2, dim=1, keepdim) 모양`,
    shapeOf(() => g().kthvalue(2, 1, true).values));

  for (const name of ["all", "any"] as const) {
    out.set(`${P}${name}(dim=1)`, () => b()[name](1));
    out.set(`${P}${name}(dim=1, keepdim) 모양`, shapeOf(() => b()[name](1, true)));
    out.set(`${P}${name}(dim=1, keepdim) 값`, () => b()[name](1, true));
    out.set(`${P}${name}() 전체`, () => b()[name]());
  }
  out.set(`${P}count_nonzero(dim=1)`, () => g().countNonzero(1));
  out.set(`${P}count_nonzero() 전체`, () => g().countNonzero());

  // 기울기도 축을 살린 채 와야 한다. 어긋나면 잎에서 터지거나 — 더 나쁘게 —
  // 브로드캐스팅으로 **번져서** 값이 커진다.
  const grad = (name: string, body: (t: Tensor) => Tensor): void => {
    out.set(`${P}grad::${name}`, () => {
      const leaf = g(true);
      body(leaf).sum().backward();
      return gradOf(leaf, name);
    });
  };
  grad("sum(keepdim)", (t) => t.sumDim(1, true));
  grad("prod(keepdim)", (t) => t.prod(1, true));
  grad("amax(keepdim)", (t) => t.amax(1, true));
  grad("max(keepdim)", (t) => t.max(1, true).values);
  grad("median(keepdim)", (t) => t.median(1, true).values);
  grad("mean(keepdim)", (t) => t.mean(1, true));
  addReduceDtype(out);
  addNamedCasts(out);
  addArgs(out);
}

/**
 * 축약의 `dtype=` — `keep::dtype::`.
 *
 * **규칙 한 줄이다: 넣기 전에 바꾼다.** 접고 나서가 아니다. 형만 물으면 두 순서가
 * 구별이 안 되므로 값도 묻는다 — `[1.7, −2.3, 0.9]` 의 합이 먼저 깎으면 −1,
 * 나중에 깎으면 0 이다.
 *
 * **이 갈래가 조용히 틀리는 자리라는 것은 이미 실측됐다.** 축약 중 `norm` 하나만
 * `dtype=` 을 안 듣고 있었고, `sum`·`mean`·`prod` 는 들었다 — 넷 중 하나만 안 듣는
 * 것이 하나도 안 듣는 것보다 나쁘다. 파이썬 쪽에서 그것을 잡았고, 여기 서른다섯은
 * **같은 물음을 borch.ts 에 처음 하는 것**이다.
 */
/**
 * 이름 있는 형 바꾸기 — `dtype::형바꾸기::`.
 *
 * **`to(형)` 이 있는데 왜 이름이 또 필요한가.** torch 가 둘 다 주기 때문이고, 교재가
 * `x.float()` 을 쓰기 때문이다. 이 이름들이 borch.ts 에 없어서 파이썬 두 판만 답하고
 * 있었고, 골든의 그 열넷은 "파이썬 쪽 이야기" 로 안 옮겨져 있었다. **실제로는
 * 아무도 안 물어본 이름이었다** — `spot::` 마흔일곱이 같은 꼴이었다.
 *
 * 여덟은 **거절이 답**이다. 그 자리에서 무엇이 중요한지가 값이 아니라 **문구**라는
 * 것이 이 묶음의 요점이다. 이름이 아예 없으면 `'half' 이 없다` 가 나오는데 그것은
 * 오타와 구별이 안 되고, 파이썬 두 판은 `.half()`(float16) 은(는) 브라우저 축소판에
 * 없습니다 로 멈춘다. 셋이 다른 문장을 말하면 배우는 사람은 **구현마다 다른 것**으로
 * 읽는다. 값 대조로는 절대 안 걸리는 갈래다.
 */
function addNamedCasts(out: Map<string, Case>): void {
  const P = "dtype::형바꾸기::";
  const src = (): Tensor => Tensor.from([1.5, -2.5, 3.0], [3]);

  out.set(`${P}float`, () => dtypeName(src().float().dtype));
  out.set(`${P}long`, () => dtypeName(src().long().dtype));
  out.set(`${P}bool`, () => dtypeName(src().bool().dtype));
  out.set(`${P}cfloat`, () => dtypeName(src().cfloat().dtype));
  // **`type_as` 는 상대의 형을 따른다** — 상대가 정수면 정수다.
  out.set(`${P}type_as`, () => dtypeName(
    src().typeAs(Tensor.from([1, 2, 3], [3], { dtype: "int64" })).dtype));

  // 파이썬 쪽 판정이 문구의 **조각**을 본다. 여기서도 같은 조각을 확인하고 같은
  // 낱말을 돌려준다 — 문장 전체를 굳히면 한 글자만 바뀌어도 갈린다.
  // **파이썬 쪽 `tests/cases.py` 와 같은 조각이어야 한다.** 여기가 갈리면 양쪽이 각자
  // 자기 문장만 확인하고, 스물한 건이 전부 초록인 채로 서로 다른 말을 하게 된다 —
  // 실제로 그랬다. 굳혀진 답이 `기대대로` 라는 판정 낱말이라 문장이 벌어져도 안 움직인다.
  const MARK = "is not in the browser subset";
  const weRefuse = (name: string, body: () => unknown): void => {
    out.set(`${P}${name}=우리는거절`, () => {
      try {
        body();
      } catch (err) {
        return (err as Error).message.includes(MARK)
          ? "기대대로"
          : `다른 문구 <${(err as Error).message.slice(0, 44)}>`;
      }
      return "뜻밖의 성공";
    });
  };
  weRefuse("half", () => src().half());
  weRefuse("bfloat16", () => src().bfloat16());
  weRefuse("chalf", () => src().chalf());
  weRefuse("cdouble", () => src().cdouble());
  weRefuse("byte", () => src().byte());
  weRefuse("char", () => src().char());
  weRefuse("short", () => src().short());
  weRefuse("int", () => src().int());

  // **`double` 은 갈래가 다르다** — 코어에는 float64 가 있고 브라우저 쪽에만 없다.
  // 파이썬 쪽은 `_as_expected` 로 접었고 답이 같은 낱말이다.
  out.set(`${P}double=브라우저는거절`, () => {
    try {
      src().double();
    } catch {
      return "기대대로";
    }
    return "뜻밖의 성공";
  });
}


function addReduceDtype(out: Map<string, Case>): void {
  const P = "keep::dtype::";
  // 실수를 정수로 접는 자리가 순서를 가른다. 정수·참거짓은 올림 쪽을 본다.
  const SLANT = [1.7, -2.3, 0.9];
  const COUNTS = [3, 1, 4];
  const MARKS = [1, 0, 1];
  const src = (name: string): Tensor => {
    if (name === "실수") return Tensor.from(SLANT, [3]);
    if (name === "정수") return Tensor.from(COUNTS, [3], { dtype: "int64" });
    return Tensor.from(MARKS, [3], { dtype: "bool" });
  };

  for (const kind of ["실수", "정수", "참거짓"]) {
    for (const want of ["float32", "int64"] as const) {
      out.set(`${P}sum(${kind}→${want})`, () => src(kind).sum(want));
      // **형 이름은 `dtypeName` 을 지난다.** 골든은 파이썬의 `str(dtype)` 인
      // `torch.float32` 를 굳혔고 borch.ts 의 `.dtype` 은 `float32` 다 — 그대로
      // 내면 여덟 건이 이름 표기 하나로 빨개진다.
      out.set(`${P}sum(${kind}→${want}) 의 형`,
        () => dtypeName(src(kind).sum(want).dtype));
      out.set(`${P}cumsum(${kind}→${want})`, () => src(kind).cumsum(0, want));
    }
    // **`sum(dtype=bool)` 은 되는데 `cumsum(dtype=bool)` 은 안 된다** — 규칙이
    // 아니라 torch 가 그 커널을 안 만든 것이라, 따로 묻지 않으면 안 보인다.
    out.set(`${P}sum(${kind}→참거짓)`, () => src(kind).sum("bool"));
    out.set(`${P}prod(${kind}→float32)`, () => src(kind).prod(undefined, false, "float32"));
  }
  out.set(`${P}mean(정수→float32)`,
    () => src("정수").mean(undefined, false, "float32"));
  out.set(`${P}mean(참거짓→float32)`,
    () => src("참거짓").mean(undefined, false, "float32"));
  out.set(`${P}sum(dim=1→float32)`,
    () => Tensor.from([1, 2, 3, 4], [2, 2], { dtype: "int64" })
      .sumDim(1, false, "float32"));
  out.set(`${P}nansum(실수→int64)`, () => src("실수").nansum(undefined, false, "int64"));

  // `dtype=` 이 **모든** 거절을 풀지는 않는다. 셋은 그대로다(실측).
  const refuses = (name: string, body: () => unknown): void => {
    out.set(`${P}${name}`, () => {
      try {
        body();
      } catch (err) {
        // 파이썬 쪽은 예외의 **종류 이름**을 굳혔다. 저쪽 `RuntimeError` 가
        // 여기서도 같은 이름이라 그대로 맞는다.
        return (err as Error).constructor.name;
      }
      return "예외가 안 났다";
    });
  };
  refuses("mean(→int64)는 거절", () => src("실수").mean(undefined, false, "int64"));
  refuses("cumsum(→참거짓)은 거절", () => src("정수").cumsum(0, "bool"));
  refuses("cumprod(→참거짓)은 거절", () => src("정수").cumprod(0, "bool"));

  // **`to` 가 형을 진짜 바꾼다.** 오래 장치 문자열만 보고 형을 조용히 버렸다.
  out.set(`${P}to(float32) 의 형`, () => dtypeName(src("정수").to("float32").dtype));
  out.set(`${P}to(int64) 의 형`, () => dtypeName(src("실수").to("int64").dtype));
  out.set(`${P}to(int64) 의 값`, () => src("실수").to("int64"));

  // **이름은 `keep::grad::` 다** — 파이썬 쪽에서 `grad()` 헬퍼가 붙이는 접두어가
  // `dtype::` 이 아니다. 여기서 한 칸 더 넣었더니 "골든에 없는 이름" 으로 나왔다.
  out.set("keep::grad::sum(dtype=float32)", () => {
    const leaf = Tensor.from([1.0, 4.0, 2.0, 3.0, 0.5, 5.0], [2, 3],
      { requiresGrad: true });
    leaf.sumDim(1, true, "float32").sum().backward();
    return gradOf(leaf, "sum(dtype=float32)");
  });
}

/**
 * 나머지 선택 인자 — `keep::arg::`.
 *
 * **둘은 받는 척하고 버리던 자리였다.** `dist(p)` 가 `p` 를 무시하고 늘 L2 를 냈고
 * (값이 그럴듯한 크기라 안 보였다), `div(roundingMode)` 는 값만 맞추고 형을 실수로
 * 뒀다. 나머지는 인자 자체가 없어 시끄럽게 멈추던 것들이다.
 */
function addArgs(out: Map<string, Case>): void {
  const P = "keep::arg::";
  const A = (): Tensor => Tensor.from([1.0, 4.0, -2.0, 3.0], [4]);
  const B = (): Tensor => Tensor.from([2.0, 3.0, 5.0, -1.0], [4]);
  const tops = (): Tensor => Tensor.from([7, -7, 8, -8], [4], { dtype: "int64" });
  const bots = (): Tensor => Tensor.from([2, 2, 3, 3], [4], { dtype: "int64" });
  const tally = (): Tensor => Tensor.from([1, 2, 2, 5], [4], { dtype: "int64" });
  const spd = (): Tensor => Tensor.from([4.0, 1.0, 1.0, 3.0], [2, 2]);
  const grid3 = (): Tensor =>
    Tensor.from(Array.from({ length: 9 }, (_, i) => i), [3, 3]);
  const zero = (): Tensor => Tensor.from([0.0], [1]);
  const nanPair = (): Tensor => Tensor.from([1.0, Number.NaN], [2]);

  out.set(`${P}add(alpha=2)`, () => A().add(B(), 2));
  out.set(`${P}sub(alpha=2)`, () => A().sub(B(), 2));
  for (const mode of ["trunc", "floor"] as const) {
    out.set(`${P}div(정수, ${mode})`, () => tops().div(bots(), mode));
    out.set(`${P}div(정수, ${mode}) 의 형`,
      () => dtypeName(tops().div(bots(), mode).dtype));
    out.set(`${P}div(실수, ${mode})`, () => A().div(B(), mode));
  }
  for (const p of [1, 3]) out.set(`${P}dist(p=${p})`, () => A().dist(B(), p));
  out.set(`${P}cholesky(upper)`, async () => spd().cholesky(true));
  out.set(`${P}diag(diagonal=1)`, () => grid3().diag(1));
  out.set(`${P}diag(diagonal=-1)`, () => grid3().diag(-1));
  out.set(`${P}diagflat(offset=1)`, () => A().diagflat(1));
  out.set(`${P}diagflat(offset=-1)`, () => A().diagflat(-1));
  out.set(`${P}diff(prepend)`, () => A().diff(1, 0, zero()));
  out.set(`${P}diff(append)`, () => A().diff(1, 0, undefined, zero()));
  out.set(`${P}bincount(weights)`, async () => tally().bincount(A()));
  out.set(`${P}bincount(weights) 의 형`,
    async () => dtypeName((await tally().bincount(A())).dtype));
  out.set(`${P}bincount(minlength=8)`,
    async () => tally().bincount(undefined, 8));
  out.set(`${P}allclose(equal_nan=False)`,
    async () => verdict(await nanPair().allclose(nanPair())));
  out.set(`${P}allclose(equal_nan=True)`,
    async () => verdict(await nanPair().allclose(nanPair(), 1e-5, 1e-8, true)));
}

/**
 * 푸리에 — `fft::`.
 *
 * **커널이 하나다.** 정변환·역변환·반쪽 변환과 그 셋의 역방향이 전부 같은 셰이더를
 * 부호와 배율만 바꿔 부른다. 그래서 이 표가 실제로 묻는 것은 **그 인자 조합**이다.
 *
 * 값보다 기울기가 요점이다. 변환은 선형이라 순방향은 맞히기 쉽고, 어려운 자리는
 * **어느 쪽 반쪽을 세는가** 다 — `rfft` 는 켤레 짝을 안 더하고(더하면 두 배),
 * `irfft` 는 가장자리만 한 번 가운데는 두 번 센다. 둘 다 **순방향 값은 멀쩡한 채로**
 * 틀릴 수 있어서, 값 케이스만 있으면 초록인 채로 지나간다.
 */
function addFft(out: Map<string, Case>): void {
  const P = "fft::";
  const XS = [1.0, -2.0, 0.5, 3.0, -1.0, 0.25];
  const YS = [0.5, 1.0, -1.5, 0.25, 2.0, -0.5];
  // **칼날을 피한 신호다.** 경사 신호(`arange/8 − 1`)는 나이퀴스트 칸이 정확히 0 이
  // 되는데 거기서 `abs` 가 미분 불가능하고 부호가 반올림에 달린다 — 값이 아니라
  // 케이스가 문제인 자리라, 0 인 칸이 없는 수로 바꿔 두었다.
  const SIG = [0.3, -1.2, 0.7, 2.1, -0.4, 1.5, -2.3, 0.9,
               1.1, -0.6, 0.25, -1.7, 2.4, 0.05, -0.8, 1.35];
  const MAT = Array.from({ length: 12 }, (_, i) => i);

  const x = (grad = false): Tensor => Tensor.from(XS, [6], { requiresGrad: grad });
  const z = (): Tensor => Tensor.complex(Tensor.from(XS, [6]), Tensor.from(YS, [6]));
  const mat = (): Tensor => Tensor.from(MAT, [3, 4]);
  const sig = (grad = false): Tensor =>
    Tensor.from(SIG, [16], { requiresGrad: grad });
  const hann = (n = 8): Tensor => Tensor.hannWindow(n);
  const pair = (fn: () => Tensor): Case => () => fn().viewAsReal();

  // ── 여러 축 · 에르미트 ─────────────────────────────────────────────────
  //
  // **이 묶음이 커널의 결함 하나를 잡았다.** 복소수를 마지막이 아닌 축으로 변환할 때
  // 허수부를 엉뚱한 칸에서 읽고 있었다 — 셰이더가 쓸 때는 `(re, im)` 을 끼워 넣고
  // 읽을 때는 안쪽 크기만큼 떨어져 있다고 봤는데, **마지막 축에서는 그 둘이 우연히
  // 같다.** 위의 1 차원 케이스가 전부 통과한 이유이고, 여기 2 차원이 없었으면
  // 지금도 초록이었을 자리다.
  const grid2 = [0.31, -1.2, 0.75, 2.1, -0.4, 1.55, -2.3, 0.9,
                 1.1, -0.62, 0.25, -1.7];
  const rgrid = (): Tensor => Tensor.from(grid2, [3, 4]);
  const cgrid = (): Tensor => Tensor.complex(
    rgrid(), Tensor.from([...grid2].slice(8).concat(grid2.slice(4, 8),
                                                   grid2.slice(0, 4)), [3, 4]));

  out.set(`${P}여러축::fft2`, pair(() => fft.fft2(cgrid())));
  out.set(`${P}여러축::fftn`, pair(() => fft.fftn(cgrid())));
  out.set(`${P}여러축::ifft2`, pair(() => fft.ifft2(cgrid())));
  out.set(`${P}여러축::ifftn`, pair(() => fft.ifftn(cgrid())));
  out.set(`${P}여러축::rfft2`, pair(() => fft.rfft2(rgrid())));
  out.set(`${P}여러축::rfftn`, pair(() => fft.rfftn(rgrid())));
  out.set(`${P}여러축::irfft2`, () => fft.irfft2(cgrid()));
  out.set(`${P}여러축::irfftn`, () => fft.irfftn(cgrid()));
  out.set(`${P}여러축::hfft`, () => fft.hfft(cgrid()));
  out.set(`${P}여러축::hfft2`, () => fft.hfft2(cgrid()));
  out.set(`${P}여러축::hfftn`, () => fft.hfftn(cgrid()));
  out.set(`${P}여러축::ihfft2`, pair(() => fft.ihfft2(rgrid())));
  out.set(`${P}여러축::ihfftn`, pair(() => fft.ihfftn(rgrid())));
  out.set(`${P}여러축::fft2(norm=ortho)`,
          pair(() => fft.fft2(cgrid(), null, [-2, -1], "ortho")));
  out.set(`${P}여러축::fft2(norm=forward)`,
          pair(() => fft.fft2(cgrid(), null, [-2, -1], "forward")));
  out.set(`${P}여러축::fft2(s)`, pair(() => fft.fft2(cgrid(), [2, 8])));
  out.set(`${P}여러축::fftn(dim 하나만)`,
          pair(() => fft.fftn(cgrid(), null, [0])));

  out.set(`${P}fft(실수)`, pair(() => fft.fft(x())));
  out.set(`${P}fft(복소)`, pair(() => fft.fft(z())));
  out.set(`${P}fft 의 형`, () => dtypeName(fft.fft(x()).dtype));
  out.set(`${P}ifft(fft)`, pair(() => fft.ifft(fft.fft(x()))));
  out.set(`${P}ifft(복소)`, pair(() => fft.ifft(z())));
  out.set(`${P}rfft`, pair(() => fft.rfft(x())));
  out.set(`${P}irfft(rfft)`, () => fft.irfft(fft.rfft(x())));
  out.set(`${P}irfft 의 형`, () => dtypeName(fft.irfft(fft.rfft(x())).dtype));
  out.set(`${P}irfft(n=5)`, () => fft.irfft(fft.rfft(x()), 5));
  out.set(`${P}irfft(n=7)`, () => fft.irfft(fft.rfft(x()), 7));
  for (const norm of ["forward", "backward", "ortho"]) {
    out.set(`${P}fft norm=${norm}`, pair(() => fft.fft(x(), null, -1, norm)));
    out.set(`${P}ifft norm=${norm}`, pair(() => fft.ifft(z(), null, -1, norm)));
  }
  for (const n of [4, 8]) {
    out.set(`${P}fft(n=${n})`, pair(() => fft.fft(x(), n)));
    out.set(`${P}rfft(n=${n})`, pair(() => fft.rfft(x(), n)));
  }
  out.set(`${P}fft(dim=0)`, pair(() => fft.fft(mat(), null, 0)));
  out.set(`${P}rfft(dim=0)`, pair(() => fft.rfft(mat(), null, 0)));

  for (const n of [5, 6]) {
    out.set(`${P}fftfreq(${n})`, () => fft.fftfreq(n));
    out.set(`${P}rfftfreq(${n})`, () => fft.rfftfreq(n));
    out.set(`${P}fftshift(${n})`, () => fft.fftshift(fft.fftfreq(n)));
    out.set(`${P}ifftshift(fftshift(${n}))`,
      () => fft.ifftshift(fft.fftshift(fft.fftfreq(n))));
  }
  out.set(`${P}fftfreq(6, d=0.5)`, () => fft.fftfreq(6, 0.5));

  const grad = (name: string, body: (t: Tensor) => Tensor): void => {
    out.set(`${P}grad::${name}`, () => {
      const leaf = x(true);
      body(leaf).sum().backward();
      return gradOf(leaf, name);
    });
  };
  grad("fft 실수부", (t) => fft.fft(t).real());
  grad("fft 크기", (t) => fft.fft(t).abs());
  grad("rfft 실수부", (t) => fft.rfft(t).real());
  grad("rfft 허수부", (t) => fft.rfft(t).imag());
  grad("rfft 크기", (t) => fft.rfft(t).abs());
  grad("irfft(rfft)", (t) => fft.irfft(fft.rfft(t)));
  grad("irfft 가중", (t) => fft.irfft(fft.rfft(t))
    .mul(Tensor.from([0, 1, 2, 3, 4, 5], [6])));
  grad("ifft(fft) 실수부", (t) => fft.ifft(fft.fft(t)).real());
  grad("fftshift(rfft) 크기", (t) => fft.fftshift(fft.rfft(t)).abs());

  for (const center of [true, false]) {
    for (const hop of [2, 4]) {
      // **이름은 파이썬 쪽 글자다.** JS 의 `true` 를 그대로 끼우면 `center=true` 가
      // 되어 골든의 `center=True` 와 안 맞고, 그 케이스는 **조용히 안 돌아간다** —
      // 러너가 "이름이 골든에 없다" 로 따로 세는 이유가 이것이다.
      const tag = center ? "True" : "False";
      out.set(`${P}stft center=${tag} hop=${hop}`, pair(() => stft(sig(), 8, {
        hopLength: hop, window: hann(), center, returnComplex: true,
      })));
    }
  }
  out.set(`${P}stft 기본 hop`,
    pair(() => stft(sig(), 8, { window: hann(), returnComplex: true })));
  out.set(`${P}stft 창 없이`,
    pair(() => stft(sig(), 8, { hopLength: 4, returnComplex: true })));
  out.set(`${P}stft win_length=6`, pair(() => stft(sig(), 8, {
    hopLength: 4, winLength: 6, window: hann(6), returnComplex: true,
  })));
  out.set(`${P}stft onesided=False`, pair(() => stft(sig(), 8, {
    hopLength: 4, window: hann(), onesided: false, returnComplex: true,
  })));
  out.set(`${P}stft normalized`, pair(() => stft(sig(), 8, {
    hopLength: 4, window: hann(), normalized: true, returnComplex: true,
  })));
  for (const mode of ["reflect", "constant", "replicate"] as const) {
    out.set(`${P}stft pad_mode=${mode}`, pair(() =>
      stft(Tensor.from([1, 2, 3, 4], [4]), 4, {
        hopLength: 2, window: Tensor.ones([4]), padMode: mode,
        returnComplex: true,
      })));
  }
  out.set(`${P}stft 배치`, pair(() => stft(sig().reshape([1, 16]), 8, {
    hopLength: 4, window: hann(), returnComplex: true,
  })));
  out.set(`${P}istft(length=16)`, () => istft(
    stft(sig(), 8, { hopLength: 4, window: hann(), returnComplex: true }),
    8, { hopLength: 4, window: hann(), length: 16 }));
  out.set(`${P}istft 길이 없이`, () => istft(
    stft(sig(), 8, { hopLength: 4, window: hann(), returnComplex: true }),
    8, { hopLength: 4, window: hann() }));

  const sgrad = (name: string, body: (t: Tensor) => Tensor): void => {
    out.set(`${P}grad::${name}`, () => {
      const leaf = sig(true);
      body(leaf).sum().backward();
      return gradOf(leaf, name);
    });
  };
  sgrad("stft 크기", (t) => stft(t, 8, {
    hopLength: 4, window: hann(), returnComplex: true,
  }).abs());
  sgrad("stft center=False 크기", (t) => stft(t, 8, {
    hopLength: 4, window: hann(), center: false, returnComplex: true,
  }).abs());
  sgrad("istft(stft)", (t) => istft(
    stft(t, 8, { hopLength: 4, window: hann(), returnComplex: true }),
    8, { hopLength: 4, window: hann(), length: 16 }));

  const refuses = (name: string, body: () => unknown): void => {
    out.set(`${P}${name}`, () => {
      try {
        body();
        return "예외가 안 났다";
      } catch {
        return "RuntimeError";
      }
    });
  };
  refuses("rfft(복소)는 거절", () => fft.rfft(z()));
  refuses("stft 는 return_complex 를 요구",
    () => stft(sig(), 8, { hopLength: 4, window: hann() }));
  refuses("복소 스펙트럼의 backward 는 거절",
    () => fft.fft(x(true)).sum().backward());
}

/**
 * 복소수 — `cplx::`.
 *
 * **파이썬 코어(numpy)가 먼저 갔고 여기가 뒤따른다.** 그동안 이 케이스들은 `golden.py`
 * 의 `CORE_ONLY_PREFIXES` 에 걸려 브라우저 쪽이 통째로 건너뛰었다. 여기 본문이 생기는
 * 순간 그 건너뜀이 끝나는 것이 아니라, **결속(`borch_webgpu`)은 아직 건너뛴다** —
 * 파이썬 결속에 복소수 이름이 아직 없기 때문이다. 셋의 범위가 한 줄로 안 움직인다.
 *
 * ## 무엇을 묻는가
 *
 * 값보다 **기울기 쪽이 이 표의 요점**이다. 규약이
 *
 *     z.grad = ∂L/∂re + i·∂L/∂im
 *
 * 이라 정칙 함수(`mul`·`div`)의 역방향에는 켤레가 붙고, 실수를 내는 `abs` 에는 안
 * 붙는다. **실수 입력으로는 그 차이가 안 보인다** — 켤레가 실수에서 항등이라서다.
 * 그래서 셋을 한 표에서 묻는다.
 *
 * `(z*z).real` 은 그중에서도 **규약 자체를 가른다**: 이 규약이면 `2−4j`, 보통의 복소
 * 미분이면 `2+4j` 다. 값만 맞히는 구현은 여기서 갈린다.
 */
function addComplex(out: Map<string, Case>): void {
  const re = [1.0, -3.0];
  const im = [2.0, 0.5];
  const z = (): Tensor => Tensor.complex(Tensor.from(re), Tensor.from(im));
  const P = "cplx::";

  out.set(`${P}complex(re, im)`, () => z().viewAsReal());
  out.set(`${P}complex 의 형`, () => dtypeName(z().dtype));
  out.set(`${P}polar`, () =>
    Tensor.polar(Tensor.from([1.0, 2.0]), Tensor.from([0.0, 1.5708]))
      .viewAsReal());
  out.set(`${P}view_as_complex 왕복`, () =>
    z().viewAsReal().viewAsComplex().viewAsReal());
  out.set(`${P}real`, () => z().real());
  out.set(`${P}imag`, () => z().imag());
  out.set(`${P}conj_physical`, () => z().conjPhysical().viewAsReal());
  out.set(`${P}angle`, () => z().angle());
  out.set(`${P}abs`, () => z().abs());
  out.set(`${P}abs 의 형`, () => dtypeName(z().abs().dtype));
  out.set(`${P}is_complex`, () => verdict(z().isComplex()));

  out.set(`${P}z * z`, () => z().mul(z()).viewAsReal());
  out.set(`${P}z + z`, () => z().add(z()).viewAsReal());
  out.set(`${P}z / z`, () => z().div(z()).viewAsReal());
  out.set(`${P}z * 실수`, () => z().mul(Tensor.from(re)).viewAsReal());
  // 승격은 **형 이름**을 묻는다. 실수가 끼어도 복소수로 남아야 한다.
  out.set(`${P}complex64 + float32 의 형`, () =>
    dtypeName(z().add(Tensor.from([1.0])).dtype));
  out.set(`${P}complex64 + int64 의 형`, () =>
    dtypeName(z().add(Tensor.from([1]).to("int64")).dtype));

  /**
   * 기울기를 **실수 잎 둘**에서 받는다. 복소수 잎을 직접 만들지 않는 것이 요점이다 —
   * 값이 `(∂L/∂re, ∂L/∂im)` 로 갈려 나와서 어느 쪽이 틀렸는지가 보인다.
   */
  const grad = (name: string, body: (w: Tensor) => Tensor): void => {
    out.set(`${P}grad::${name}`, () => {
      const r = Tensor.from(re, undefined, { requiresGrad: true });
      const i = Tensor.from(im, undefined, { requiresGrad: true });
      body(Tensor.complex(r, i)).sum().backward();
      return Tensor.cat([gradOf(r, name), gradOf(i, name)], 0);
    });
  };

  grad("z.real", (w) => w.real());
  grad("z.imag", (w) => w.imag());
  grad("abs(z)", (w) => w.abs());
  grad("abs(z) 제곱", (w) => w.abs().mul(w.abs()));
  grad("(z*z).real", (w) => w.mul(w).real());
  grad("(z*conj(z)).real", (w) => w.mul(w.conjPhysical()).real());
  grad("view_as_real 합", (w) => w.viewAsReal());

  out.set(`${P}복소 손실의 backward 는 거절`, () => {
    const r = Tensor.from(re, undefined, { requiresGrad: true });
    const i = Tensor.from(im, undefined, { requiresGrad: true });
    try {
      Tensor.complex(r, i).mul(Tensor.complex(r, i)).sum().backward();
      return "예외가 안 났다";
    } catch {
      // 골든은 **예외의 종류 이름**을 굳혔다. 코어(numpy)가 `RuntimeError` 를 내고
      // 여기도 같은 이름이라야 옮겨 적은 코드가 같은 것을 잡는다.
      return "RuntimeError";
    }
  });
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
  // **가중치가 1 부터다.** 0 부터면 첫 자리의 몫이 0 이고, 출력이 한 칸인 케이스는
  // 그 하나가 전부라 **기울기가 통째로 0** 이 된다 — 균일 접기를 피하려는 장치가
  // 그 케이스를 아무것도 안 묻는 상태로 만든다.
  const seed = (t: Tensor): Tensor =>
    t.mul(Tensor.from([...Array(t.size).keys()].map((i) => i + 1), t.shape));

  const grad = (name: string, src: readonly number[],
                fn: (x: Tensor) => Tensor): void => {
    set(`grad::${name}`, () => {
      const x = Tensor.from([...src], [src.length], { requiresGrad: true });
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
          Tensor.from([...ta], [ta.length], { requiresGrad: true }),
          Tensor.from([...tb], [tb.length], { requiresGrad: true }),
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
    const x = Tensor.from([...dup], [dup.length], { requiresGrad: true });
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
    const x = Tensor.from([...tied], [1, 1, 4, 4], { requiresGrad: true });
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
  // **torch 의 이름을 지나간다.** 밑동은 `Recurrent(in, hidden, kind)` 하나인데,
  // 그것만 부르면 `nn.LSTM` 이라는 이름을 아무도 안 재게 된다 — 교재의 첫 줄이 그것이다.
  const KINDS = { RNN: nn.RNN, LSTM: nn.LSTM, GRU: nn.GRU } as const;
  const build = (kind: nn.RNNKind): nn.Recurrent => {
    const m = new KINDS[kind](3, 4);
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

  // 크기 바꾸기. **실수 이미지로 한다** — uint8 로 하면 `ToTensor` 가 255 로 나누는
  // 자리가 파이썬 판과 순서 때문에 갈린다(저쪽은 텐서를 받아 먼저 나뉜다).
  const f = () => pic("vis_f", false);
  // **`ToTensor` 를 거친다** — 파이썬 쪽 케이스가 `ToTensor(Resize(...))` 이고,
  // 그것이 (H,W,C) 를 (C,H,W) 로 돌린다. 옆의 `asTensor` 는 전치를 안 하는
  // 다른 헬퍼다(그 케이스들은 파이썬 쪽도 전치를 안 한다).
  const toTensor = (img: vision.Image): Tensor =>
    new vision.ToTensor().apply(img) as Tensor;
  out.set("vision::Resize(줄임·겹선형)", () =>
    toTensor(new vision.Resize([4, 3]).apply(f()) as vision.Image));
  out.set("vision::Resize(늘림·겹선형)", () =>
    toTensor(new vision.Resize([11, 9]).apply(f()) as vision.Image));
  out.set("vision::Resize(짧은변)", () =>
    toTensor(new vision.Resize(4).apply(f()) as vision.Image));
  out.set("vision::Resize(최근접)", () =>
    toTensor(new vision.Resize([4, 3], "nearest").apply(f()) as vision.Image));
  // **상한이 실제로 문다.** 5×4 를 짧은 변 8 로 늘리면 긴 변이 10 인데 9 로 잘리고
  // 짧은 변이 7 로 따라 준다 — 두 나눗셈이 다 버림이라야 (9, 7) 이다.
  out.set("vision::Resize(long side capped)", () =>
    toTensor(new vision.Resize(8, "bilinear", 9).apply(f()) as vision.Image));
  // **자를 자리가 홀수인 것을 넣는다.** 파이썬의 round 는 절반을 짝수로 보내고
  // `Math.round` 는 위로 올린다 — 그 차이로 여기가 한 칸 어긋나 최대 0.837 이
  // 갈렸다(실측). 짝수만 시험하면 그 자리가 통째로 안 걸린다.
  out.set("vision::CenterCrop(짝수)", () =>
    toTensor(new vision.CenterCrop([4, 4]).apply(f()) as vision.Image));
  out.set("vision::CenterCrop(홀수)", () =>
    toTensor(new vision.CenterCrop([5, 3]).apply(f()) as vision.Image));
  out.set("vision::CenterCrop(원본보다 큼)", () =>
    toTensor(new vision.CenterCrop([13, 11]).apply(f()) as vision.Image));

  // 이 프로젝트는 `repr` 도 명세로 본다 — 튜토리얼이 `print(transform)` 을 한다.
  const reprs: [string, () => vision.Transform][] = [
    ["ToTensor", () => new vision.ToTensor()],
    ["Normalize", () => new vision.Normalize(mean, std)],
    ["RandomHorizontalFlip", () => new vision.RandomHorizontalFlip(0.5)],
    ["RandomCrop", () => new vision.RandomCrop(32, 4)],
    // **기본값을 따로 묻는다.** 위의 것은 `padding=4` 를 주므로 기본이 무엇으로
    // 찍히는지를 한 번도 안 본다 — 그래서 `padding=0` 과 torchvision 의 `None` 이
    // 갈린 채로 남아 있었다. 인자를 준 케이스는 그 인자의 기본값을 못 잰다.
    ["RandomCrop(the default)", () => new vision.RandomCrop(32)],
    ["CenterCrop", () => new vision.CenterCrop(24)],
    // **네 칸 전부 찍는다.** 여기가 두 칸이었고 파이썬 쪽도 두 칸이라 양쪽이 맞았다 —
    // repr 이 다른 그 변환이 골든에 케이스가 없던 유일한 변환이었다.
    ["Resize", () => new vision.Resize(4)],
    ["Resize(a pair)", () => new vision.Resize([4, 3])],
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
/**
 * 합성 구조를 뚫고 **파라미터가 보이는가.**
 *
 * 나머지 케이스는 값을 묻는다 — 틀리면 숫자가 다르고 바로 보인다. 여기서 묻는 것은
 * 순회다. `parameters()` 가 어떤 파라미터를 안 내놓으면 옵티마이저가 그것을 못 보고,
 * 못 보면 안 갱신하고, **손실은 그래도 내려간다**(남은 파라미터가 대신 맞춘다).
 *
 * 그래서 자리마다 둘을 짝으로 둔다 — `namedParameters` 의 **이름 목록**과, SGD 를
 * 세 스텝 돌린 뒤의 **파라미터 값**. 등록이 빠지면 값이 출발점 그대로 남아 갈린다.
 */
function addContainer(out: Map<string, Case>, inp: Inputs): void {
  const STEPS = 3;

  const run = (
    name: string,
    build: () => nn.Module,
    load: (m: nn.Module) => void,
    forward: (m: nn.Module, x: Tensor) => Tensor,
    want: string,
  ): void => {
    out.set(`container::${name}/이름`,
      () => Object.keys(build().namedParameters()).join(" "));
    out.set(`container::${name}/학습`, () => {
      const m = build();
      load(m);
      const opt = new optim.SGD(m.parameters(), 0.05);
      const x = inp.get("train_x");
      for (let i = 0; i < STEPS; i++) {
        opt.zeroGrad();
        const o = forward(m, x);
        // 자리마다 다른 가중치를 곱해 접는다 — 그냥 `sum()` 이면 기울기가 전부 1 이라
        // 어느 자리가 안 움직였는지가 값에 안 남는다.
        o.mul(Tensor.arange(o.size).reshape(o.shape)).sum().backward();
        opt.step();
      }
      const got = m.namedParameters()[want];
      if (!got) throw new Error(`${want} 가 없다`);
      return got;
    });
  };

  // ── 이름 붙인 자식. torch 의 `self.fc1 = …` 자리다. ──────────────────────
  class Named extends nn.Module {
    readonly fc1 = new nn.Linear(6, 8);
    readonly fc2 = new nn.Linear(8, 3);

    override namedChildren(): Record<string, nn.Module> {
      return { fc1: this.fc1, fc2: this.fc2 };
    }

    override forward(x: Tensor): Tensor {
      return this.fc2.call(this.fc1.call(x).relu());
    }
  }

  const loadTwo = (m: nn.Module, a: string, b: string): void => {
    m.loadStateDict({
      [`${a}.weight`]: inp.get("w0"), [`${a}.bias`]: inp.get("b0"),
      [`${b}.weight`]: inp.get("w1"), [`${b}.bias`]: inp.get("b1"),
    });
  };

  run("상속", () => new Named(), (m) => loadTwo(m, "fc1", "fc2"),
    (m, x) => m.forward(x), "fc1.weight");

  // ── ModuleList — 생성자로 세운 것과 `append` 로 세운 것. ─────────────────
  class Listed extends nn.Module {
    readonly layers: nn.ModuleList;

    constructor(appended: boolean) {
      super();
      this.layers = appended ? new nn.ModuleList() : new nn.ModuleList([
        new nn.Linear(6, 8), new nn.Linear(8, 3),
      ]);
      if (appended) {
        this.layers.append(new nn.Linear(6, 8));
        this.layers.append(new nn.Linear(8, 3));
      }
    }

    override namedChildren(): Record<string, nn.Module> {
      return { layers: this.layers };
    }

    override forward(x: Tensor): Tensor {
      return this.layers.at(1).call(this.layers.at(0).call(x).relu());
    }
  }

  run("ModuleList", () => new Listed(false),
    (m) => loadTwo(m, "layers.0", "layers.1"),
    (m, x) => m.forward(x), "layers.0.weight");
  run("ModuleList(append)", () => new Listed(true),
    (m) => loadTwo(m, "layers.0", "layers.1"),
    (m, x) => m.forward(x), "layers.1.weight");

  // ── ModuleDict — 이름으로 갈래를 고른다. ────────────────────────────────
  class Dicted extends nn.Module {
    readonly blocks = new nn.ModuleDict({
      down: new nn.Linear(6, 8), up: new nn.Linear(8, 3),
    });

    override namedChildren(): Record<string, nn.Module> {
      return { blocks: this.blocks };
    }

    override forward(x: Tensor): Tensor {
      return this.blocks.at("up").call(this.blocks.at("down").call(x).relu());
    }
  }

  run("ModuleDict", () => new Dicted(),
    (m) => loadTwo(m, "blocks.down", "blocks.up"),
    (m, x) => m.forward(x), "blocks.down.weight");

  // ── ParameterList·ParameterDict — 층에 안 붙은 파라미터. ────────────────
  //
  // `w0` 를 눕혀(`(6,8)`) `x @ w` 가 되게 한다. 잎으로 다시 세우는 것이 요점이다 —
  // 전치한 결과를 그대로 쓰면 부모가 달린 텐서라 파라미터가 아니다.
  const flatW = (): Tensor => asLeaf(inp.get("w0").transpose());
  const bias = (): Tensor => asLeaf(inp.get("b0"));

  class PList extends nn.Module {
    readonly ws = new nn.ParameterList([flatW(), bias()]);

    override namedChildren(): Record<string, nn.Module> {
      return { ws: this.ws };
    }

    override forward(x: Tensor): Tensor {
      return x.mm(this.ws.at(0)).add(this.ws.at(1));
    }
  }

  run("ParameterList", () => new PList(), () => undefined,
    (m, x) => m.forward(x), "ws.0");

  class PDict extends nn.Module {
    readonly ws = new nn.ParameterDict({ w: flatW(), b: bias() });

    override namedChildren(): Record<string, nn.Module> {
      return { ws: this.ws };
    }

    override forward(x: Tensor): Tensor {
      return x.mm(this.ws.at("w")).add(this.ws.at("b"));
    }
  }

  run("ParameterDict", () => new PDict(), () => undefined,
    (m, x) => m.forward(x), "ws.w");

  // ── `stateDict` 의 열쇠. 갈리면 남의 체크포인트를 못 읽는다. ────────────
  out.set("container::상속/state_dict 열쇠",
    () => Object.keys(new Named().stateDict()).sort().join(" "));
  out.set("container::ModuleDict/state_dict 열쇠",
    () => Object.keys(new Dicted().stateDict()).sort().join(" "));

  // **버퍼를 가진 층으로도 물어야 한다.** 위의 둘은 `Linear` 뿐이라 파라미터만 나오고,
  // 그래서 `stateDict` 와 `namedParameters` 가 같은지 다른지가 안 보인다. 둘은 정확히
  // 버퍼만큼 달라야 한다 — 같으면 이동 통계가 파라미터 행세를 하고 옵티마이저로 간다.
  out.set("container::BatchNorm/state_dict 열쇠",
    () => Object.keys(new nn.BatchNormND(3).stateDict()).sort().join(" "));
  out.set("container::BatchNorm/named_parameters 열쇠",
    () => Object.keys(new nn.BatchNormND(3).namedParameters()).sort().join(" "));
  out.set("container::BatchNorm/named_buffers 열쇠",
    () => Object.keys(new nn.BatchNormND(3).namedBuffers()).sort().join(" "));

  // `registerBuffer` 는 층이 아니라 **사용자가 쓰는 문법**이다. 마스크·위치표를
  // 들고 다니는 모델이 전부 이것을 쓴다. borch.ts 에도 있어야 파이썬 쪽과 같은
  // 모델을 세울 수 있다.
  class Buffered extends nn.Module {
    fc = new nn.Linear(6, 8);
    constructor() {
      super();
      this.registerBuffer("mask", Tensor.owned([4], 1));
    }
    override forward(x: Tensor): Tensor { return this.fc.forward(x); }
  }

  out.set("container::register_buffer/state_dict 열쇠",
    () => Object.keys(new Buffered().stateDict()).sort().join(" "));
  out.set("container::register_buffer/named_parameters 열쇠",
    () => Object.keys(new Buffered().namedParameters()).sort().join(" "));

  // `persistent=false` 는 **저장에서 빠진다.** 무시하면 남의 체크포인트와 열쇠가
  // 어긋나고, 받는 쪽이 strict 로 읽으면 그대로 거절이다.
  class Cached extends nn.Module {
    constructor() {
      super();
      this.registerBuffer("kept", Tensor.owned([2], 1));
      this.registerBuffer("cache", Tensor.owned([2], 1), false);
    }
    override forward(x: Tensor): Tensor { return x; }
  }

  out.set("container::register_buffer(persistent=False)",
    () => Object.keys(new Cached().stateDict()).sort().join(" "));

  // **열쇠가 맞아도 값이 안 건너가면 소용없다.** 내보내는 목록과 받는 목록이
  // 갈리면 자기가 저장한 파일을 자기가 못 읽는다 — 실제로 그 상태였다.
  class Masked extends nn.Module {
    constructor() {
      super();
      this.registerBuffer("mask", Tensor.owned([3], 1));
    }
    override forward(x: Tensor): Tensor { return x; }
  }

  // **등록 안 한 텐서 속성은 버퍼가 아니다.** torch 는 어느 목록에도 안 넣는다.
  // 여기는 `ownParameters` 가 깃발을, `namedBuffers` 가 등록을 보므로 이미 그렇다 —
  // 결속은 안 그랬고(속성에 붙은 텐서를 전부 실었다) 그래서 이 규칙을 못 박는다.
  class Plain extends nn.Module {
    fc = new nn.Linear(6, 8);
    plain = Tensor.owned([3], 1);              // 등록 안 했다
    override forward(x: Tensor): Tensor { return this.fc.forward(x); }
  }

  out.set("container::등록 안 한 텐서 속성/state_dict 열쇠",
    () => Object.keys(new Plain().stateDict()).sort().join(" "));
  out.set("container::등록 안 한 텐서 속성/named_buffers 열쇠",
    () => Object.keys(new Plain().namedBuffers()).sort().join(" "));

  out.set("container::버퍼 값이 왕복한다", () => {
    const src = new Masked();
    src.loadStateDict({ mask: Tensor.from([2, 5, 9]) });
    const dst = new Masked();
    dst.loadStateDict(src.stateDict());
    return dst.namedBuffers()["mask"] as Tensor;
  });

  // ── `eval()` 이 컨테이너를 뚫고 내려가는가. ─────────────────────────────
  //
  // 갓 세운 BatchNorm 은 `running_mean=0`·`running_var=1` 이라 평가 모드의 출력이
  // 입력과 거의 같고, 학습 모드는 배치 통계로 정규화해 눈에 띄게 다른 값이 된다.
  class Normed extends nn.Module {
    readonly layers = new nn.ModuleList([
      new nn.Linear(6, 8), new nn.BatchNormND(8),
    ]);

    override namedChildren(): Record<string, nn.Module> {
      return { layers: this.layers };
    }

    override forward(x: Tensor): Tensor {
      return this.layers.at(1).call(this.layers.at(0).call(x));
    }
  }

  out.set("container::eval 이 컨테이너를 뚫는다", () => {
    const m = new Normed();
    m.loadStateDict({
      "layers.0.weight": inp.get("w0"), "layers.0.bias": inp.get("b0"),
    }, false);
    m.eval();
    return m.forward(inp.get("train_x"));
  });
}

/**
 * 활성함수 열일곱. **꺾이는 자리에서 묻는다.**
 *
 * 난수 입력은 특별한 값을 안 준다 — 정확히 0, ±1, ±3, 6 은 뽑히지 않는데 활성함수는
 * 바로 그 점에서 꺾인다. 골든의 `kinks` 가 그 점들을 손으로 들고 있다.
 *
 * 함수 꼴(텐서 메서드)과 층 꼴을 둘 다 묻는다. 층은 한 줄짜리 감싸개라 틀릴 데가
 * 없어 보이지만 틀리는 방식이 하나 있다 — 다른 함수를 부르는 것. 값으로만 잡힌다.
 */
function addAct(out: Map<string, Case>, inp: Inputs): void {
  const add = (
    name: string,
    fn: (x: Tensor) => Tensor,
    key: "kinks" | "x1" | "x2" = "kinks",
  ): void => {
    out.set(`act::${name}`, () => fn(inp.get(key)));
    out.set(`act::grad::${name}`, () => {
      const x = inp.get(key, true);
      seeded(fn(x)).backward();
      return gradOf(x, name);
    });
  };

  // 인자 없는 것들 — 표에서 자동으로 메서드가 된 이름을 그대로 부른다.
  const plain: [string, string][] = [
    ["celu", "CELU"], ["hardshrink", "Hardshrink"], ["hardsigmoid", "Hardsigmoid"],
    ["hardswish", "Hardswish"], ["hardtanh", "Hardtanh"], ["logsigmoid", "LogSigmoid"],
    ["mish", "Mish"], ["relu6", "ReLU6"], ["selu", "SELU"], ["softplus", "Softplus"],
    ["softshrink", "Softshrink"], ["softsign", "Softsign"],
    ["tanhshrink", "Tanhshrink"],
  ];
  const call = (x: Tensor, name: string): Tensor => {
    // 인자를 받는 넷은 기본값으로 부른다. 나머지는 단항 표에 있다.
    if (name === "celu") return x.celu();
    if (name === "hardshrink") return x.hardshrink();
    if (name === "softshrink") return x.softshrink();
    if (name === "hardtanh") return x.hardtanh();
    if (name === "softplus") return x.softplus();
    return x.unary(name);
  };
  const layer = (cls: string): nn.Module => {
    const table: Record<string, () => nn.Module> = {
      CELU: () => new nn.CELU(), Hardshrink: () => new nn.Hardshrink(),
      Hardsigmoid: () => new nn.Hardsigmoid(), Hardswish: () => new nn.Hardswish(),
      Hardtanh: () => new nn.Hardtanh(), LogSigmoid: () => new nn.LogSigmoid(),
      Mish: () => new nn.Mish(), ReLU6: () => new nn.ReLU6(),
      SELU: () => new nn.SELU(), Softplus: () => new nn.Softplus(),
      Softshrink: () => new nn.Softshrink(), Softsign: () => new nn.Softsign(),
      Tanhshrink: () => new nn.Tanhshrink(),
    };
    const make = table[cls];
    if (!make) throw new Error(`모르는 활성함수 층: ${cls}`);
    return make();
  };
  for (const [fname, cls] of plain) {
    add(`F.${fname}`, (x) => call(x, fname));
    out.set(`act::nn.${cls}`, () => layer(cls).call(inp.get("kinks")));
  }

  // 인자를 받는 것들. **기본값만 물으면 그 인자가 아예 안 쓰여도 통과한다.**
  add("F.hardtanh(범위)", (x) => x.hardtanh(-0.5, 0.5));
  add("F.softplus(beta)", (x) => x.softplus(2.0));
  add("F.celu(alpha)", (x) => x.celu(0.5));
  add("F.hardshrink(람다)", (x) => x.hardshrink(1.0));
  add("F.softshrink(람다)", (x) => x.softshrink(1.0));
  add("F.threshold", (x) => x.threshold(0.5, -1.0));
  out.set("act::nn.Threshold",
    () => new nn.Threshold(0.5, -1.0).call(inp.get("kinks")));
  out.set("act::nn.Hardtanh(범위)",
    () => new nn.Hardtanh(-0.5, 0.5).call(inp.get("kinks")));

  add("F.softmin", (x) => x.softmin(-1), "x2");
  out.set("act::nn.Softmin", () => new nn.Softmin(-1).call(inp.get("x2")));

  add("F.glu", (x) => x.glu(-1), "x1");
  out.set("act::nn.GLU", () => new nn.GLU(-1).call(inp.get("x1")));

  add("F.prelu", (x) => x.prelu(Tensor.full([1], 0.25)));
  out.set("act::nn.PReLU", () => new nn.PReLU().call(inp.get("kinks")));
  out.set("act::nn.PReLU/파라미터 이름",
    () => Object.keys(new nn.PReLU().namedParameters()).join(" "));

  // ── 결속이 메꾸고 있던 여덟. **여기서 처음 물어진다.** ──────────────────
  //
  // 케이스가 전부 결속을 지나므로 이 층들이 borch.ts 에 없다는 것을 골든이 구조적으로
  // 못 봤다. 옮기면서 셋이 인자를 갖고 있는 것이 드러났다.
  for (const [fname, cls] of [
    ["silu", "SiLU"], ["sigmoid", "Sigmoid"], ["tanh", "Tanh"], ["gelu", "GELU"],
  ] as [string, string][]) {
    add(`F.${fname}`, (x) => x.unary(fname));
    out.set(`act::nn.${cls}`, () => {
      const table: Record<string, () => nn.Module> = {
        SiLU: () => new nn.SiLU(), Sigmoid: () => new nn.Sigmoid(),
        Tanh: () => new nn.Tanh(), GELU: () => new nn.GELU(),
      };
      const make = table[cls];
      if (!make) throw new Error(`모르는 활성함수 층: ${cls}`);
      return make().call(inp.get("kinks"));
    });
  }

  add("F.gelu(tanh)", (x) => x.geluTanh());
  out.set("act::nn.GELU(tanh)", () => new nn.GELU("tanh").call(inp.get("kinks")));
  out.set("act::GELU 두 꼴은 다르다", async () => {
    const k = inp.get("kinks");
    // `max()` 는 값과 자리를 함께 낸다 — 축 없이 접는 것은 `amax` 다.
    const gap = await k.unary("gelu").sub(k.geluTanh()).abs().amax().item();
    return verdict(gap > 1e-6);
  });

  add("F.elu(alpha)", (x) => x.elu(0.5));
  out.set("act::nn.ELU", () => new nn.ELU().call(inp.get("kinks")));
  out.set("act::nn.ELU(alpha)", () => new nn.ELU(0.5).call(inp.get("kinks")));
  out.set("act::nn.LeakyReLU", () => new nn.LeakyReLU().call(inp.get("kinks")));
  out.set("act::nn.LeakyReLU(기울기)",
    () => new nn.LeakyReLU(0.2).call(inp.get("kinks")));
  out.set("act::nn.Identity", () => new nn.Identity().call(inp.get("kinks")));
  // torch 의 `Identity` 는 아무 인자나 받아 버린다. 자바스크립트는 남는 인자를 그냥
  // 버리므로 이쪽은 저절로 그렇게 되지만, **저절로 되는 것도 물어 둔다** —
  // 생성자에 인자를 넣는 날 조용히 갈릴 자리다.
  out.set("act::nn.Identity(인자를 삼킨다)",
    () => new nn.Identity().call(inp.get("kinks")));

  // `Softmax()` 의 기본 축은 **`-1` 이 아니다.** 랭크 2 로만 물으면 `dim=1` 과
  // `dim=-1` 이 같은 축이라 그 규칙이 안 보인다.
  const ranked = Array.from({ length: 24 }, (_, i) => i * 0.1);
  const folds: [string, (x: Tensor, dim: number | null) => Tensor][] = [
    ["Softmax", (x, d) => new nn.Softmax(d).call(x)],
    ["LogSoftmax", (x, d) => new nn.LogSoftmax(d).call(x)],
  ];
  for (const [cls, make] of folds) {
    out.set(`act::nn.${cls}(dim 지정)`, () => make(inp.get("x2"), -1));
    out.set(`act::nn.${cls}(기본 축/랭크2)`,
      () => make(inp.get("x2").reshape([3, 4]), null));
    out.set(`act::nn.${cls}(기본 축/랭크3)`,
      () => make(Tensor.from(ranked, [2, 3, 4]), null));
    out.set(`act::nn.${cls}(기본 축/랭크4)`,
      () => make(Tensor.from(ranked, [2, 3, 2, 2]), null));
  }
}

/**
 * 정규화 셋과 전치 합성곱. **모양이 맞아도 값이 틀리는 자리들이다.**
 *
 * 정규화 넷은 식이 같고 묶는 축만 다르다 — 축을 잘못 고르면 모양은 그대로이고 값만
 * 갈리는데, 학습은 그래도 돌아서 한참 뒤에야 안다. 전치 합성곱은 가중치 축이
 * `(입력, 출력, …)` 로 뒤집혀 있어서, 정사각 커널이면 뒤집어도 모양이 맞는다.
 */
/**
 * RNN 셀 셋 — 되풀이의 한 걸음.
 *
 * 게이트 순서가 값의 전부라 가중치를 못 박고 값을 묻는다. 자세한 것은
 * `tests/cases.py` 의 `cell_cases` 에 적었다.
 */
function addCell(out: Map<string, Case>): void {
  const x = () => Tensor.from([1, 2], [1, 2]);
  const h = () => Tensor.from([0.5, 0.5], [1, 2]);
  const c0 = () => Tensor.from([0.2, 0.3], [1, 2]);
  const eye = [1, 0, 0, 1];

  const load = (cell: nn.RNNCellBase, gates: number) => {
    const rep = (scale: number) => {
      const got: number[] = [];
      for (let g = 0; g < gates; g++) for (const v of eye) got.push(v * scale);
      return Tensor.from(got, [gates * 2, 2]);
    };
    cell.loadStateDict({
      weight_ih: rep(1), weight_hh: rep(0.5),
      bias_ih: Tensor.zeros([gates * 2]), bias_hh: Tensor.zeros([gates * 2]),
    });
    return cell;
  };

  out.set("cell::RNNCell",
    () => (load(new nn.RNNCell(2, 2), 1) as nn.RNNCell).step(x(), h()));
  out.set("cell::RNNCell(relu)",
    () => (load(new nn.RNNCell(2, 2, true, "relu"), 1) as nn.RNNCell)
      .step(x(), h()));
  out.set("cell::RNNCell(상태 없이)",
    () => (load(new nn.RNNCell(2, 2), 1) as nn.RNNCell).step(x()));
  out.set("cell::GRUCell",
    () => (load(new nn.GRUCell(2, 2), 3) as nn.GRUCell).step(x(), h()));
  out.set("cell::LSTMCell/h",
    () => (load(new nn.LSTMCell(2, 2), 4) as nn.LSTMCell)
      .step(x(), [h(), c0()])[0]);
  out.set("cell::LSTMCell/c",
    () => (load(new nn.LSTMCell(2, 2), 4) as nn.LSTMCell)
      .step(x(), [h(), c0()])[1]);
  out.set("cell::LSTMCell(상태 없이)",
    () => (load(new nn.LSTMCell(2, 2), 4) as nn.LSTMCell).step(x())[0]);

  out.set("cell::state_dict 열쇠",
    async () => Object.keys(new nn.RNNCell(3, 2).stateDict()).join(","));
  out.set("cell::state_dict 열쇠(bias 없이)",
    async () => Object.keys(new nn.RNNCell(3, 2, false).stateDict()).join(","));

  for (const [name, make] of [
    ["RNNCell", () => new nn.RNNCell(3, 2)],
    ["GRUCell", () => new nn.GRUCell(3, 2)],
    ["LSTMCell", () => new nn.LSTMCell(3, 2)],
  ] as const) {
    out.set(`cell::repr::${name}`, async () => make().describe());
  }
  out.set("cell::repr::RNNCell(relu)",
    async () => new nn.RNNCell(3, 2, true, "relu").describe());
  out.set("cell::repr::RNNCell(bias 없이)",
    async () => new nn.RNNCell(3, 2, false).describe());

  for (const [name, make] of [
    ["RNNCell", () => new nn.RNNCell(3, 2)],
    ["GRUCell", () => new nn.GRUCell(3, 2)],
    ["LSTMCell", () => new nn.LSTMCell(3, 2)],
  ] as const) {
    out.set(`cell::모양::${name}`,
      async () => `(${make().weightIh.shape.join(", ")})`);
  }

  const grads: [string, number, (c: nn.RNNCellBase, x: Tensor) => Tensor][] = [
    ["RNNCell", 1, (c, xi) => (c as nn.RNNCell).step(xi, h())],
    ["GRUCell", 3, (c, xi) => (c as nn.GRUCell).step(xi, h())],
    ["LSTMCell", 4, (c, xi) => (c as nn.LSTMCell).step(xi, [h(), c0()])[0]],
  ];
  for (const [name, gates, run] of grads) {
    out.set(`cell::grad::${name}`, () => {
      const make = name === "GRUCell"
        ? new nn.GRUCell(2, 2)
        : name === "LSTMCell" ? new nn.LSTMCell(2, 2) : new nn.RNNCell(2, 2);
      const cell = load(make, gates);
      const inp = Tensor.from([1, 2], [1, 2], { requiresGrad: true });
      seeded(run(cell, inp)).backward();
      return gradOf(inp, name);
    });
  }
}

/**
 * 이긴 자리를 함께 내는 풀링과, 그 자리로 되돌리는 짝.
 *
 * 최대 풀링은 창마다 하나만 남기므로 **값 안에 "어느 칸이 이겼는가" 가 없다.**
 * `maxUnpool` 은 그래서 값만으로 못 돌아간다. 자리는 torch 의 규약대로 **평면 안의
 * 평평한 번호**이고, 배치·채널마다 0 부터 다시 센다.
 *
 * 함정 둘은 파이썬 쪽 `unpool_cases` 에 길게 적었다. 짧게: 같은 값이 둘이면 평평한
 * 번호가 작은 쪽이 이기고, 나누어떨어지지 않는 적응형은 창 길이가 자리마다 다르다.
 */
function addUnpool(out: Map<string, Case>): void {
  const grid = (shape: number[]) =>
    Tensor.from(
      Array.from({ length: shape.reduce((a, b) => a * b, 1) }, (_, i) => i),
      shape);
  const plane = () => grid([1, 1, 4, 4]);
  const planes = () => grid([2, 2, 4, 4]);
  const line = () => grid([1, 1, 8]);
  const cube = () => grid([1, 1, 4, 4, 4]);
  const odd = () => grid([1, 1, 3, 3]);

  const pools: [string, () => { values: Tensor; indices: Tensor }][] = [
    ["max_pool1d", () => line().maxPoolWithIndices(2)],
    ["max_pool2d", () => plane().maxPoolWithIndices(2)],
    ["max_pool2d(stride=1)", () => plane().maxPoolWithIndices(2, 1)],
    ["여러 평면", () => planes().maxPoolWithIndices(2)],
    ["max_pool3d", () => cube().maxPoolWithIndices(2)],
    ["적응형", () => plane().adaptiveMaxPoolWithIndices(2)],
    ["적응형(3→2)", () => odd().adaptiveMaxPoolWithIndices(2)],
    ["적응형 1차원", () => line().adaptiveMaxPoolWithIndices(4)],
    ["적응형 3차원", () => cube().adaptiveMaxPoolWithIndices(2)],
  ];
  for (const [name, run] of pools) {
    out.set(`unpool::자리::${name}`, () => run().indices);
    out.set(`unpool::값::${name}`, () => run().values);
  }

  // 자리를 켠 길과 안 켠 길이 **같은 값**이어야 한다. 커널이 둘이라 갈릴 수 있다.
  out.set("unpool::자리를 켜도 값은 같다",
    () => plane().maxPool2d(2).sub(plane().maxPoolWithIndices(2).values));

  const back = (src: () => Tensor, kernel = 2, stride?: number) => {
    const got = src().maxPoolWithIndices(kernel, stride);
    return got.values.maxUnpool(got.indices, kernel, stride);
  };
  out.set("unpool::되돌리기::1차원", () => back(line));
  out.set("unpool::되돌리기::2차원", () => back(plane));
  out.set("unpool::되돌리기::3차원", () => back(cube));
  out.set("unpool::되돌리기::여러 평면", () => back(planes));
  out.set("unpool::되돌리기::겹치는 창", () => back(plane, 2, 1));
  out.set("unpool::되돌리기::output_size", () => {
    const got = plane().maxPoolWithIndices(2);
    return got.values.maxUnpool(got.indices, 2, undefined, 0, [5, 5]);
  });

  // ── 층 ────────────────────────────────────────────────────────────────
  //
  // **계산은 위에서 이미 다 재고 있었고 층 이름이 없었다.** `되돌리기::` 여섯 건이
  // 텐서 메서드로 돌고 있었는데, `nn.MaxUnpool2d` 라는 이름은 아무도 안 물었다.
  //
  // `MaxUnpool` 은 인자가 둘이라 `Sequential` 에 못 들어간다 — 자리가 값과 나란히
  // 흘러야 하고, 층 안에 숨기면 같은 층을 두 번 쓸 때 남의 자리를 쓴다. torch 도 같은
  // 모양이고, 그래서 여기도 `forward` 가 아니라 `place(x, indices)` 다.
  out.set("unpool::층::MaxPool2d → MaxUnpool2d", () => {
    const pool = new nn.MaxPool2d(2, undefined, true);
    const got = pool.pick(plane());
    return new nn.MaxUnpool2d(2).place(got.values, got.indices);
  });

  // `returnIndices` 가 **반환형을 바꾸는** 자리라 층이 그것을 보고 골라야 한다.
  out.set("unpool::층::AdaptiveMaxPool2d 자리",
    () => new nn.AdaptiveMaxPool2d(2, true).pick(plane()).indices);

  out.set("unpool::grad::자리 판의 풀링", () => {
    const x = grid([1, 1, 4, 4]);
    x.requiresGrad = true;
    const got = x.maxPoolWithIndices(2);
    got.values.mul(Tensor.full([], 2)).sum().backward();
    return gradOf(x, "maxPoolWithIndices");
  });

  out.set("unpool::grad::되돌리기", () => {
    const pooled = grid([1, 1, 2, 2]);
    pooled.requiresGrad = true;
    const idx = plane().maxPoolWithIndices(2).indices;
    pooled.maxUnpool(idx, 2).sum().backward();
    return gradOf(pooled, "maxUnpool");
  });

  // ── 보폭을 창에서 떼어 놓는다 ─────────────────────────────────────────
  //
  // `stride` 를 안 주면 `kernel` 이 된다 — **기본값이 겹치는 짝**이라, 둘을 같게 두고
  // 묻는 케이스만으로는 보폭을 흘리는 구현도 통과한다. 파이썬 쪽에 이유를 길게 적었다.
  const div = (t: Tensor, by: number) => t.div(Tensor.full([], by));
  out.set("unpool::겹치는 창::max_pool1d", () => line().maxPool1d(3, 1));
  out.set("unpool::겹치는 창::max_pool3d", () => cube().maxPool3d(3, 1));
  out.set("unpool::겹치는 창::avg_pool1d", () => line().poolND("avg", 3, 1));
  out.set("unpool::겹치는 창::avg_pool3d", () => cube().poolND("avg", 3, 1));
  out.set("unpool::겹치는 창::lp_pool1d", () => div(line(), 8).lpPool(2, 3, 1));
  out.set("unpool::겹치는 창::lp_pool2d", () => div(plane(), 8).lpPool(2, 3, 1));
  out.set("unpool::겹치는 창::lp_pool3d", () => div(cube(), 64).lpPool(2, 3, 1));
  out.set("unpool::겹치는 창::max_unpool1d", () => back(line, 3, 1));
  out.set("unpool::겹치는 창::max_unpool3d", () => back(cube, 3, 1));

  const small = () => grid([1, 1, 4, 4, 4]).div(Tensor.full([], 8));
  out.set("unpool::lp_pool3d", () => small().lpPool(2, 2));
  out.set("unpool::lp_pool3d(p=1)", () => small().lpPool(1, 2));

  // ── 분수 최대 풀링 ─────────────────────────────────────────────────────
  //
  // **7→3 으로 묻는다.** 6→3 은 α 가 정수라 표본이 아무 일도 안 하고, 그러면
  // 무작위 부분이 통째로 안 보인다. 축마다 다른 표본을 주는 케이스도 하나 둔다 —
  // ATen 이 2차원판과 3차원판에서 표본을 **다른 순서로** 읽기 때문이다.
  const frac = () => grid([1, 1, 7, 7]);
  const frac3 = () => grid([1, 1, 7, 7, 7]);
  const planes7 = () => grid([2, 2, 7, 7]);

  // 이름의 철자는 **파이썬이 찍은 것**이다 — JS 의 `${0.0}` 은 "0" 이고 파이썬은
  // "0.0" 이라, 수를 그대로 끼우면 이름이 갈린다. 러너가 그것을 세서 알려줬다.
  for (const [label, u] of [["0.0", 0], ["0.25", 0.25], ["0.5", 0.5],
    ["0.75", 0.75], ["0.99", 0.99]] as const) {
    out.set(`unpool::분수::값(u=${label})`,
      () => frac().fractionalMaxPool(2, [3, 3], [[u, u]]).values);
    out.set(`unpool::분수::자리(u=${label})`,
      () => frac().fractionalMaxPool(2, [3, 3], [[u, u]]).indices);
  }
  out.set("unpool::분수::축마다 다른 표본",
    () => frac().fractionalMaxPool(2, [3, 3], [[0.0, 0.75]]).indices);
  out.set("unpool::분수::평면마다 다른 표본",
    () => planes7().fractionalMaxPool(
      2, [3, 3], [[0.0, 0.0], [0.3, 0.7], [0.9, 0.1], [0.5, 0.5]]).values);
  out.set("unpool::분수::평면마다 다른 표본 자리",
    () => planes7().fractionalMaxPool(
      2, [3, 3], [[0.0, 0.0], [0.3, 0.7], [0.9, 0.1], [0.5, 0.5]]).indices);
  out.set("unpool::분수::output_ratio",
    () => frac().fractionalMaxPool(2, [3, 3], [[0, 0]]).values);
  out.set("unpool::분수::겹치는 창",
    () => frac().fractionalMaxPool(3, [3, 3], [[0, 0]]).indices);
  out.set("unpool::분수::3차원 값",
    () => frac3().fractionalMaxPool(2, [3, 3, 3], [[0.2, 0.0, 0.25]]).values);
  out.set("unpool::분수::3차원 자리",
    () => frac3().fractionalMaxPool(2, [3, 3, 3], [[0.2, 0.0, 0.25]]).indices);
  out.set("unpool::분수::grad", () => {
    const x = grid([1, 1, 7, 7]);
    x.requiresGrad = true;
    x.fractionalMaxPool(2, [3, 3], [[0.25, 0.75]]).values.sum().backward();
    return gradOf(x, "fractionalMaxPool");
  });

  // 층 꼴. **위의 `output_ratio` 케이스는 비율을 안 묻는다** — 본문이 크기를 직접
  // 적고 있어서(`[3, 3]`) 비율을 크기로 바꾸는 규칙이 그 자리에 없다. 층에는 그
  // 인자가 있으므로 여기서 비로소 물어진다. 7×0.5 는 3.5 라 버림과 반올림이 갈린다.
  out.set("unpool::층::FractionalMaxPool2d",
    () => new nn.FractionalMaxPool2d(2, [4, 4], null, false, [[0.0, 0.75]])
      .call(frac()));
  out.set("unpool::층::FractionalMaxPool2d(비율)",
    () => new nn.FractionalMaxPool2d(2, null, [0.5, 0.5], false, [[0.0, 0.75]])
      .call(frac()));
  const fracRefuses = (
    size: readonly number[] | null, ratio: readonly number[] | null,
  ): string => {
    try {
      new nn.FractionalMaxPool2d(2, size, ratio);
      return "예외가 안 났다";
    } catch (err) {
      // **종류 이름을 그대로 답한다.** "멈췄는가" 로만 접으면 오타가 낸 `TypeError`
      // 도 통과한다 — 그건 아무것도 안 묻는 검사다.
      return err instanceof Error ? err.constructor.name : typeof err;
    }
  };
  out.set("unpool::층::FractionalMaxPool2d(둘 다 주면)",
    () => fracRefuses([3, 3], [0.5, 0.5]));
  out.set("unpool::층::FractionalMaxPool2d(둘 다 없으면)",
    () => fracRefuses(null, null));

  // ── CTC ────────────────────────────────────────────────────────────────
  //
  // `reduction="mean"` 은 표본마다 **제 표적 길이로 나눈 뒤** 평균한다. 길이가 다
  // 같으면 그냥 평균과 답이 같아 그 나눗셈이 안 보이므로, 2 와 1 로 어긋나게 준다.
  const T = 5, NB = 2, CC = 4;
  const logits = () => Tensor.from(
    Array.from({ length: T * NB * CC }, (_, i) => i / 10), [T, NB, CC]);
  const lp = () => logits().logSoftmax(2);
  const tgt = [[1, 2], [3, 0]];
  const inLen = [5, 5];
  const tgtLen = [2, 1];

  for (const red of ["mean", "sum", "none"] as const) {
    out.set(`unpool::ctc::reduction=${red}`,
      () => nn.ctcLoss(lp(), tgt, inLen, tgtLen, 0, red));
  }
  out.set("unpool::ctc::blank=3",
    () => nn.ctcLoss(lp(), [[1, 2], [0, 0]], inLen, tgtLen, 3, "none"));
  out.set("unpool::ctc::입력 길이가 다를 때",
    () => nn.ctcLoss(lp(), tgt, [5, 3], tgtLen, 0, "none"));
  out.set("unpool::ctc::반복 글자",
    () => nn.ctcLoss(lp(), [[1, 1], [1, 1]], inLen, [2, 2], 0, "none"));
  const tooLong = [[1, 2, 3, 1, 2, 3], [1, 2, 3, 1, 2, 3]];
  out.set("unpool::ctc::표적이 입력보다 길 때",
    () => nn.ctcLoss(lp(), tooLong, [2, 2], [6, 6], 0, "none"));
  out.set("unpool::ctc::zero_infinity",
    () => nn.ctcLoss(lp(), tooLong, [2, 2], [6, 6], 0, "none", true));

  out.set("unpool::ctc::grad(로짓까지)", () => {
    const x = logits();
    x.requiresGrad = true;
    nn.ctcLoss(x.logSoftmax(2), tgt, inLen, tgtLen, 0, "sum").backward();
    return gradOf(x, "ctcLoss");
  });

  // ── AdaptiveLogSoftmaxWithLoss ─────────────────────────────────────────
  //
  // 가중치는 **난수가 아니다.** 파이썬 쪽 케이스와 같은 값을 여기서도 적어야 하는데
  // 난수 생성기는 언어를 못 건넌다. 세는 값이라 양쪽이 같은 것을 만든다.
  const asmW = (shape: number[]) => {
    const n = shape.reduce((a, b) => a * b, 1);
    return Tensor.from(Array.from({ length: n }, (_, i) => i / n - 0.5), shape);
  };
  const asmX = () => Tensor.from(
    Array.from({ length: 24 }, (_, i) => i / 10 - 1), [6, 4]);
  const asmY = () => Tensor.from([0, 1, 5, 7, 10, 11], [6], { dtype: "int64" });

  const asm = () => {
    const m = new nn.AdaptiveLogSoftmaxWithLoss(4, 12, [3, 7], 2.0);
    m.loadStateDict({
      "head.weight": asmW([5, 4]),
      "tail.0.0.weight": asmW([2, 4]),
      "tail.0.1.weight": asmW([4, 2]),
      "tail.1.0.weight": asmW([1, 4]),
      "tail.1.1.weight": asmW([5, 1]),
    });
    return m;
  };

  out.set("unpool::적응형softmax::log_prob", () => asm().logProb(asmX()));
  // **행마다 확률의 합이 1** — 뭉치를 고른 확률을 안 더하면 여기서 깨진다.
  out.set("unpool::적응형softmax::행 합이 1",
    () => asm().logProb(asmX()).exp().sumDim(1, false));
  out.set("unpool::적응형softmax::output",
    () => asm().run(asmX(), asmY()).output);
  out.set("unpool::적응형softmax::loss",
    () => asm().run(asmX(), asmY()).loss);
  out.set("unpool::적응형softmax::predict", () => asm().predict(asmX()));
  out.set("unpool::층::repr::AdaptiveLogSoftmaxWithLoss",
    async () => asm().describe());

  // ── `max`·`min` 의 세 얼굴 ──────────────────────────────────────────────
  //
  // torch 는 인자에 따라 다른 것을 낸다: 전부의 최댓값 하나 · `(값, 번호)` 짝 ·
  // 칸마다의 최댓값. borch.ts 는 셋을 **다른 이름**으로 갖고 있어서(`amax`·`max`·
  // `binary("maximum")`) 여기서는 갈릴 자리가 없지만, 파이썬 쪽이 한 이름으로
  // 받으므로 그 세 답을 여기서도 굳혀 둔다.
  const grid2 = () => Tensor.from([3, 1, 4, 1, 5, 9], [2, 3]);
  const other2 = () => Tensor.from([2, 2, 2, 7, 0, 7], [2, 3]);
  out.set("fname::max::전부", () => grid2().amax());
  out.set("fname::min::전부", () => grid2().amin());
  out.set("fname::max::전부(모양)",
    async () => `(${grid2().amax().shape.join(", ")})`);
  out.set("fname::max::축 하나의 값", () => grid2().max(1).values);
  out.set("fname::max::축 하나의 번호", () => grid2().max(1).indices);
  out.set("fname::min::축 하나의 값", () => grid2().min(0).values);
  out.set("fname::max::칸마다", () => grid2().binary("maximum", other2()));
  out.set("fname::min::칸마다", () => grid2().binary("minimum", other2()));

  // ── batch_norm ─────────────────────────────────────────────────────────
  //
  // **학습이면 이동 통계를 제자리에서 고친다.** 새것을 돌려주는 구현은 출력 케이스를
  // 전부 지나고 평가 모드의 값만 틀리므로, 갱신된 통계 자체를 답으로 굳힌다.
  const bnX = () => Tensor.from(
    Array.from({ length: 24 }, (_, i) => i / 10 - 1), [2, 3, 4]);
  const bnRm = () => Tensor.from([0.1, 0.2, 0.3], [3]);
  const bnRv = () => Tensor.from([1.0, 2.0, 0.5], [3]);
  const bnW = () => Tensor.from([1.5, 0.5, 2.0], [3]);
  const bnB = () => Tensor.from([0.1, -0.1, 0.2], [3]);

  out.set("fname::batch_norm::평가",
    () => nn.batchNorm(bnX(), bnRm(), bnRv(), bnW(), bnB(), false));
  out.set("fname::batch_norm::eps=0.1",
    () => nn.batchNorm(bnX(), bnRm(), bnRv(), bnW(), bnB(), false, 0.1, 0.1));
  out.set("fname::batch_norm::학습",
    () => nn.batchNorm(bnX(), bnRm(), bnRv(), bnW(), bnB(), true));
  out.set("fname::batch_norm::가중치 없이",
    () => nn.batchNorm(bnX(), bnRm(), bnRv(), null, null, false));
  out.set("fname::batch_norm::통계 없이 학습",
    () => nn.batchNorm(bnX(), null, null, bnW(), bnB(), true));
  const bnUpdate = (momentum: number) => () => {
    const rm = bnRm();
    const rv = bnRv();
    nn.batchNorm(bnX(), rm, rv, null, null, true, momentum);
    return Tensor.cat([rm, rv], 0);
  };
  out.set("fname::batch_norm::갱신된 통계", bnUpdate(0.1));
  out.set("fname::batch_norm::갱신된 통계(momentum=0.5)", bnUpdate(0.5));
  out.set("fname::batch_norm::grad", () => {
    const x = bnX();
    x.requiresGrad = true;
    nn.batchNorm(x, bnRm(), bnRv(), bnW(), bnB(), true).sum().backward();
    return gradOf(x, "batchNorm");
  });

  // ── embedding_bag ──────────────────────────────────────────────────────
  const ebTable = () => Tensor.from(
    Array.from({ length: 20 }, (_, i) => i / 10), [5, 4]);
  const ebIdx = () => Tensor.from([0, 2, 1, 4], [2, 2], { dtype: "int64" });
  for (const mode of ["mean", "sum", "max"] as const) {
    out.set(`fname::embedding_bag::${mode}`,
      () => nn.embeddingBag(ebIdx(), ebTable(), null, mode));
  }
  out.set("fname::embedding_bag::offsets",
    () => nn.embeddingBag(
      Tensor.from([0, 2, 1, 4, 3], [5], { dtype: "int64" }), ebTable(), [0, 2], "sum"));
  out.set("fname::embedding_bag::per_sample_weights",
    () => nn.embeddingBag(ebIdx(), ebTable(), null, "sum",
      Tensor.from([1, 2, 0.5, 0.5], [2, 2])));
  out.set("fname::embedding_bag::grad", () => {
    const table = ebTable();
    table.requiresGrad = true;
    nn.embeddingBag(ebIdx(), table, null, "sum").sum().backward();
    return gradOf(table, "embeddingBag");
  });

  // ── 공간 변환기 ────────────────────────────────────────────────────────
  //
  // 함정 둘은 파이썬 쪽에 길게 적었다. 짧게: 정사각으로만 물으면 격자의 `(x, y)`
  // 순서를 못 보고, **기울기는 칸 안쪽에서 물어야 한다** — 90° 회전은 격자가 칸
  // 경계에 정확히 떨어져 `floor` 가 6e-8 차이에 뒤집힌다.
  const th = (v: number[]) => Tensor.from(v, [1, 2, 3]);
  const eye3 = () => th([1, 0, 0, 0, 1, 0]);
  const shift3 = () => th([1, 0, 0.5, 0, 1, -0.5]);
  const flip3 = () => th([-1, 0, 0, 0, -1, 0]);
  const rot3 = () => th([0, -1, 0, 1, 0, 0]);
  const tilt3 = () => th([0.8, 0.2, 0.05, -0.15, 0.9, -0.1]);
  const img3 = () => grid([1, 1, 3, 3]);
  const rect24 = () => grid([1, 1, 2, 4]);

  const gridCases: [string, () => Tensor, number[]][] = [
    ["항등", eye3, [1, 1, 3, 3]],
    ["이동", shift3, [1, 1, 2, 2]],
    ["뒤집기", flip3, [1, 1, 3, 3]],
    ["회전", rot3, [1, 1, 3, 3]],
    ["직사각 2x4", eye3, [1, 1, 2, 4]],
  ];
  for (const [name, make, size] of gridCases) {
    for (const ac of [false, true]) {
      out.set(`fname::affine_grid::${name}(align=${ac ? "True" : "False"})`,
        () => nn.affineGrid(make(), size, ac));
    }
  }

  for (const ac of [false, true]) {
    const tag = ac ? "True" : "False";
    for (const mode of ["bilinear", "nearest"] as const) {
      out.set(`fname::grid_sample::항등(${mode}, align=${tag})`,
        () => nn.gridSample(img3(), nn.affineGrid(eye3(), [1, 1, 3, 3], ac),
          mode, "zeros", ac));
    }
    out.set(`fname::grid_sample::뒤집기(align=${tag})`,
      () => nn.gridSample(img3(), nn.affineGrid(flip3(), [1, 1, 3, 3], ac),
        "bilinear", "zeros", ac));
  }

  const outGrid = () => Tensor.from([-2, -2, 2, 2, 0, 0, -1, 1], [1, 2, 2, 2]);
  for (const pad of ["zeros", "border", "reflection"] as const) {
    for (const ac of [false, true]) {
      out.set(`fname::grid_sample::padding=${pad}(align=${ac ? "True" : "False"})`,
        () => nn.gridSample(img3(), outGrid(), "bilinear", pad, ac));
    }
  }

  const halfGrid = () => Tensor.from([0.25, -0.3, -0.6, 0.4], [1, 1, 2, 2]);
  out.set("fname::grid_sample::반 칸",
    () => nn.gridSample(img3(), halfGrid(), "bilinear", "zeros", false));
  out.set("fname::grid_sample::직사각 입력",
    () => nn.gridSample(rect24(), halfGrid(), "bilinear", "zeros", false));
  out.set("fname::grid_sample::여러 평면", () => {
    const g = Tensor.cat([outGrid(), outGrid()], 0);
    return nn.gridSample(grid([2, 2, 3, 3]), g, "bilinear", "zeros", false);
  });

  out.set("fname::grid_sample::grad(입력)", () => {
    const x = img3();
    x.requiresGrad = true;
    nn.gridSample(x, halfGrid(), "bilinear", "zeros", false).sum().backward();
    return gradOf(x, "gridSample");
  });
  out.set("fname::grid_sample::grad(격자)", () => {
    const g = halfGrid();
    g.requiresGrad = true;
    nn.gridSample(img3(), g, "bilinear", "zeros", false).sum().backward();
    return gradOf(g, "gridSample");
  });
  out.set("fname::grid_sample::grad(theta 까지)", () => {
    const t = tilt3();
    t.requiresGrad = true;
    nn.gridSample(img3(), nn.affineGrid(t, [1, 1, 3, 3], false),
      "bilinear", "zeros", false).sum().backward();
    return gradOf(t, "affineGrid");
  });

  // ── multi_head_attention_forward ───────────────────────────────────────
  //
  // **입력이 `(L, N, E)` — 길이가 앞이다.** 파이썬 쪽과 같은 수를 써야 하므로
  // 가중치는 골든이 들고 온 것을 쓴다(`inputs` 에 있다).
  //
  // 여기서는 가림막이 **더하는 실수**뿐이다. 참·거짓을 실수로 바꾸는 일은 torch 의
  // 계약을 흉내내는 파이썬 결속이 한다 — 라이브러리 자신은 한 가지만 받는다.
  // 가중치는 **난수가 아니다** — 세는 값이라 파이썬 쪽과 같은 것을 만든다.
  const mhaW = (shape: number[], spin: number, grad = false) => {
    const n = shape.reduce((a, b) => a * b, 1);
    return Tensor.from(
      Array.from({ length: n }, (_, i) => Math.sin(i + spin) * 0.5), shape, { requiresGrad: grad });
  };
  const mhaQ = (grad = false) => mhaW([3, 2, 4], 0.0, grad);
  const mhaK = () => mhaW([3, 2, 4], 0.7);
  const mhaV = () => mhaW([3, 2, 4], 1.3);
  const mhaInW = () => mhaW([12, 4], 2.1);
  const mhaInB = () => mhaW([12], 0.4);
  const mhaOutW = () => mhaW([4, 4], 1.9);
  const mhaOutB = () => mhaW([4], 2.6);
  const runMha = (opts: {
    mask?: Tensor | null;
    pad?: Tensor | null;
    average?: boolean;
  } = {}) => nn.multiHeadAttentionForward(
    mhaQ(), mhaK(), mhaV(), 2, mhaInW(), mhaInB(), mhaOutW(), mhaOutB(),
    opts.mask ?? null, opts.pad ?? null, opts.average ?? true);

  out.set("fname::mha::출력", () => runMha().output);
  out.set("fname::mha::가중치(머리 평균)", () => runMha().weights);
  out.set("fname::mha::가중치(머리마다)",
    () => runMha({ average: false }).weights);
  // 인과 가림막 — 위 삼각을 -inf 로. 참·거짓이 아니라 더하는 실수다.
  const causalAdd = () => Tensor.from(
    [0, -Infinity, -Infinity, 0, 0, -Infinity, 0, 0, 0], [3, 3]);
  out.set("fname::mha::실수 가림막",
    () => runMha({ mask: causalAdd() }).output);
  out.set("fname::mha::불리언 가림막",
    () => runMha({ mask: causalAdd() }).output);
  out.set("fname::mha::is_causal",
    () => runMha({ mask: causalAdd() }).output);
  const padAdd = () => Tensor.from([0, 0, -Infinity, 0, -Infinity, -Infinity], [2, 3]);
  out.set("fname::mha::key_padding_mask",
    () => runMha({ pad: padAdd() }).output);
  out.set("fname::mha::key_padding_mask 가중치",
    () => runMha({ pad: padAdd() }).weights);
  out.set("fname::mha::grad(query)", () => {
    const q = mhaQ(true);
    nn.multiHeadAttentionForward(q, mhaK(), mhaV(), 2, mhaInW(), mhaInB(),
      mhaOutW(), mhaOutB()).output.sum().backward();
    return gradOf(q, "multiHeadAttentionForward");
  });
}

/**
 * 남은 층 아홉 — 창을 펴는 둘과 나머지.
 *
 * 함정 넷은 파이썬 쪽 `misc_cases` 에 적었다 — 짧게: `fold` 는 겹친 자리를 더하고,
 * `LocalResponseNorm` 의 창은 왼쪽으로 치우쳐 있으며, `RReLU` 는 평가 모드에서
 * 기울기가 정해지고, `UpsamplingBilinear2d` 는 `alignCorners=true` 다.
 */
function addMisc(out: Map<string, Case>): void {
  const seq = (n: number, shape: number[]) =>
    Tensor.from(Array.from({ length: n }, (_, i) => i), shape);
  const img = () => seq(16, [1, 1, 4, 4]);
  const img3 = () => seq(48, [1, 3, 4, 4]);
  const small = () => seq(4, [1, 1, 2, 2]);
  const chans = () => Tensor.from([1, 2, 3, 4], [1, 4, 1, 1]);
  const cube = () => seq(12, [1, 3, 2, 2]);

  const value: [string, () => Tensor][] = [
    ["unfold", () => img().unfoldIm2col(2)],
    ["unfold(stride=2)", () => img().unfoldIm2col(2, 1, 0, 2)],
    ["unfold(padding=1)", () => img().unfoldIm2col(2, 1, 1)],
    ["unfold(채널 셋)", () => img3().unfoldIm2col(2)],
    ["fold(겹친 자리는 더한다)", () => img().unfoldIm2col(2).fold([4, 4], 2)],
    ["fold(stride=2 면 안 겹친다)",
      () => img().unfoldIm2col(2, 1, 0, 2).fold([4, 4], 2, 1, 0, 2)],
    ["층::Unfold", () => new nn.Unfold(2).call(img())],
    ["층::Fold",
      () => new nn.Fold([4, 4], 2).call(new nn.Unfold(2).call(img()))],

    ["local_response_norm", () => chans().localResponseNorm(2)],
    ["local_response_norm(alpha=1)",
      () => chans().localResponseNorm(2, 1, 1, 1)],
    ["local_response_norm(size=3)",
      () => chans().localResponseNorm(3, 1, 1, 1)],
    ["층::LocalResponseNorm", () => new nn.LocalResponseNorm(2).call(chans())],

    ["층::Softmax2d", () => new nn.Softmax2d().call(cube())],
    ["Softmax2d 는 softmax(dim=1)",
      () => new nn.Softmax2d().call(cube()).sub(cube().softmax(1))],

    ["rrelu(eval)", () => Tensor.from([-1, -2, 1], [1, 3]).rrelu()],
    ["층::RReLU(eval)",
      () => new nn.RReLU().eval().call(Tensor.from([-1, -2, 1], [1, 3]))],
    ["rrelu(eval, 범위 지정)",
      () => Tensor.from([-1, -2, 1], [1, 3]).rrelu(0.2, 0.4, false)],

    ["층::UpsamplingNearest2d",
      () => new nn.UpsamplingNearest2d(2).call(small())],
    ["층::UpsamplingBilinear2d",
      () => new nn.UpsamplingBilinear2d(2).call(small())],
    ["UpsamplingBilinear2d 는 align_corners=True",
      () => new nn.UpsamplingBilinear2d(2).call(small())
        .sub(small().interpolateBilinear(4, 4, true))],
  ];
  for (const [name, fn] of value) out.set(`misc::${name}`, fn);

  out.set("misc::grad::unfold", () => {
    const x = Tensor.from(Array.from({ length: 16 }, (_, i) => i),
      [1, 1, 4, 4], { requiresGrad: true });
    seeded(x.unfoldIm2col(2)).backward();
    return gradOf(x, "unfold");
  });

  const w = Array.from({ length: 24 }, (_, i) => i / 10);
  const bias = [0.5, -0.25];
  const a1 = () => Tensor.from([1, 2, 3], [1, 3]);
  const a2 = () => Tensor.from([1, 1, 1, 1], [1, 4]);
  out.set("misc::bilinear", () =>
    a1().bilinear(a2(), Tensor.from(w, [2, 3, 4]), Tensor.from(bias, [2])));
  out.set("misc::bilinear(편향 없이)", () =>
    a1().bilinear(a2(), Tensor.from(w, [2, 3, 4])));
  out.set("misc::층::Bilinear", () => {
    const layer = new nn.Bilinear(3, 4, 2);
    layer.loadStateDict({
      weight: Tensor.from(w, [2, 3, 4]), bias: Tensor.from(bias, [2]),
    });
    return layer.call2(a1(), a2());
  });

  const table = Array.from({ length: 15 }, (_, i) => i);
  const bags = () => Tensor.from([0, 1, 2, 3], [2, 2]);
  for (const mode of ["sum", "mean", "max"] as const) {
    out.set(`misc::층::EmbeddingBag(${mode})`, () => {
      const layer = new nn.EmbeddingBag(5, 3, mode);
      layer.loadStateDict({ weight: Tensor.from(table, [5, 3]) });
      return layer.call(bags());
    });
  }

  out.set("misc::층::EmbeddingBag(offsets)", () => {
    const layer = new nn.EmbeddingBag(5, 3, "sum");
    layer.loadStateDict({ weight: Tensor.from(table, [5, 3]) });
    return layer.callOffsets(Tensor.from([0, 1, 2, 3], [4]), [0, 2]);
  });

  for (const [name, make] of [
    ["Bilinear", () => new nn.Bilinear(3, 4, 2)],
    ["LocalResponseNorm", () => new nn.LocalResponseNorm(2)],
    ["Softmax2d", () => new nn.Softmax2d()],
    ["RReLU", () => new nn.RReLU()],
    ["EmbeddingBag", () => new nn.EmbeddingBag(5, 3)],
  ] as const) {
    out.set(`misc::repr::${name}`,
      async () => (make() as unknown as { describe(): string }).describe());
  }
}

/**
 * 자리를 옮기는 층 셋과 채널째 떨구는 dropout 다섯.
 *
 * 난수가 안 끼는 자리는 값으로, 끼는 자리는 성질로 묻는다 — 자세한 것은
 * `tests/cases.py` 의 `shuffle_cases` 에 적었다.
 */
function addShuffle(out: Map<string, Case>): void {
  const seq = (n: number, shape: number[]) =>
    Tensor.from(Array.from({ length: n }, (_, i) => i), shape);
  const pix = () => seq(32, [1, 8, 2, 2]);
  const flat = () => seq(16, [1, 1, 4, 4]);
  const chan = () => seq(4, [1, 4, 1, 1]);
  const chan6 = () => seq(12, [1, 6, 2, 1]);
  const img = () => seq(16, [1, 4, 2, 2]);

  const value: [string, () => Tensor][] = [
    ["pixel_shuffle", () => pix().pixelShuffle(2)],
    ["pixel_unshuffle", () => flat().pixelUnshuffle(2)],
    ["pixel 왕복", () => pix().pixelShuffle(2).pixelUnshuffle(2)],
    ["channel_shuffle(2)", () => chan().channelShuffle(2)],
    ["channel_shuffle(3)", () => chan6().channelShuffle(3)],
    ["층::PixelShuffle", () => new nn.PixelShuffle(2).call(pix())],
    ["층::PixelUnshuffle", () => new nn.PixelUnshuffle(2).call(flat())],
    ["층::ChannelShuffle", () => new nn.ChannelShuffle(2).call(chan())],
  ];
  for (const [name, fn] of value) out.set(`shuffle::${name}`, fn);

  out.set("shuffle::grad::pixel_shuffle", () => {
    const x = Tensor.from(Array.from({ length: 32 }, (_, i) => i),
      [1, 8, 2, 2], { requiresGrad: true });
    seeded(x.pixelShuffle(2)).backward();
    return gradOf(x, "pixel_shuffle");
  });

  for (const [name, make] of [
    ["PixelShuffle", () => new nn.PixelShuffle(2)],
    ["PixelUnshuffle", () => new nn.PixelUnshuffle(2)],
    ["ChannelShuffle", () => new nn.ChannelShuffle(2)],
  ] as const) {
    out.set(`shuffle::repr::${name}`, async () => make().describe());
  }

  // 난수가 안 끼는 자리는 값으로.
  const ranks: Record<string, () => Tensor> = {
    dropout1d: () => seq(12, [1, 4, 3]),
    dropout2d: img,
    dropout3d: () => seq(6, [1, 3, 2, 1, 1]),
    alpha_dropout: img,
    feature_alpha_dropout: img,
  };
  for (const [name, src] of Object.entries(ranks)) {
    const alpha = name.includes("alpha");
    const perChannel = name !== "alpha_dropout";
    out.set(`shuffle::${name}::eval 은 항등`, () => (alpha
      ? src().alphaDropout(0.5, false, perChannel)
      : src().featureDropout(0.5, false)));
    out.set(`shuffle::${name}::p=0 은 항등`, () => (alpha
      ? src().alphaDropout(0, true, perChannel)
      : src().featureDropout(0, true)));
  }

  // 난수가 끼는 자리는 성질로.
  const big = () => Tensor.ones([200, 8, 2, 2]);
  const perChannelSame = async (make: () => Tensor, label: string) => {
    const got = await make().toArray();
    let uniform = true;
    for (let i = 0; i < 200 * 8 && uniform; i++) {
      const base = got[i * 4] ?? 0;
      for (let k = 1; k < 4; k++) if (got[i * 4 + k] !== base) uniform = false;
    }
    void label;
    return `채널마다 한 덩어리=${uniform ? "True" : "False"}`;
  };
  out.set("shuffle::dropout2d::채널째 떨군다",
    async () => perChannelSame(() => big().featureDropout(0.5, true), "d2"));
  out.set("shuffle::feature_alpha_dropout::채널째 떨군다",
    async () => perChannelSame(
      () => big().alphaDropout(0.5, true, true), "fa"));

  out.set("shuffle::dropout2d::살아남은 배율", async () => {
    const got = await big().featureDropout(0.5, true).toArray();
    const kept = [...got].filter((v) => v !== 0);
    const mean = kept.reduce((a, b) => a + b, 0) / Math.max(1, kept.length);
    // 자릿수를 못 박는다 — 파이썬은 `2.0` 을, JS 는 `2` 를 낸다.
    return kept.length ? `배율=${mean.toFixed(3)}` : "배율=none";
  });
  out.set("shuffle::dropout2d::떨구는 비율", async () => {
    const got = await big().featureDropout(0.5, true).toArray();
    let zeros = 0;
    for (let i = 0; i < 200 * 8; i++) if ((got[i * 4] ?? 0) === 0) zeros += 1;
    const rate = zeros / (200 * 8);
    return `대략 절반=${rate > 0.4 && rate < 0.6 ? "True" : "False"}`;
  });
  out.set("shuffle::alpha_dropout::떨군 자리가 0 이 아니다", async () => {
    const got = await Tensor.ones([400, 8]).alphaDropout(0.5, true, false)
      .toArray();
    const seen = new Set([...got].map((v) => Math.round(v * 1e4) / 1e4));
    const vals = [...seen].sort((a, b) => a - b);
    const lo = vals[0] ?? 0;
    const hi = vals[vals.length - 1] ?? 0;
    return `값이 둘=${vals.length === 2 ? "True" : "False"} ` +
      `낮은쪽=${Math.round(lo * 1000) / 1000} 높은쪽=${Math.round(hi * 1000) / 1000}`;
  });

  const layers: [string, () => nn.Module, () => Tensor][] = [
    ["Dropout1d", () => new nn.Dropout1d(0.5), () => seq(12, [1, 4, 3])],
    ["Dropout2d", () => new nn.Dropout2d(0.5), img],
    ["Dropout3d", () => new nn.Dropout3d(0.5), () => seq(6, [1, 3, 2, 1, 1])],
    ["AlphaDropout", () => new nn.AlphaDropout(0.5), img],
    ["FeatureAlphaDropout", () => new nn.FeatureAlphaDropout(0.5), img],
  ];
  for (const [name, make, src] of layers) {
    out.set(`shuffle::층::${name}(eval)`, () => make().eval().call(src()));
    out.set(`shuffle::repr::${name}`,
      async () => (make() as unknown as { describe(): string }).describe());
  }
}

/**
 * 모양을 첫 forward 에서 알아내는 층들.
 *
 * **굳으면 딴 것이 된다.** 파이썬 쪽은 클래스를 바꾸고 이쪽은 프로토타입을 갈아
 * 끼운다 — 같은 자리를 다른 언어로 짚은 것이다. 사용자가 보는 것은 `print(model)`
 * 이므로 골든이 그 글자를 묻는다. 자세한 것은 `tests/cases.py` 의 `lazy_cases` 에.
 */
function addLazy(out: Map<string, Case>): void {
  const x2d = () => Tensor.from(Array.from({ length: 10 }, (_, i) => i), [2, 5]);
  const img = () => Tensor.from(
    Array.from({ length: 2 * 2 * 8 * 8 }, (_, i) => i / 100), [2, 2, 8, 8]);

  out.set("lazy::굳기전::repr", async () => new nn.LazyLinear(3).describe());
  out.set("lazy::굳은뒤::repr", async () => {
    const m = new nn.LazyLinear(3);
    m.call(x2d());
    return (m as unknown as nn.Linear).describe();
  });
  out.set("lazy::has_uninitialized_params 가 사라진다", async () => {
    const m = new nn.LazyLinear(3);
    const has = (o: object) =>
      typeof (o as { hasUninitializedParams?: unknown })
        .hasUninitializedParams === "function";
    const before = has(m);
    m.call(x2d());
    return `전 ${before ? "True" : "False"} 후 ${has(m) ? "True" : "False"}`;
  });

  const shapes: [string, () => nn.LazyModule, () => Tensor][] = [
    ["LazyLinear", () => new nn.LazyLinear(3), x2d],
    ["LazyConv2d", () => new nn.LazyConv2d(4, 3), img],
    ["LazyBatchNorm2d", () => new nn.LazyBatchNorm2d(), img],
    ["LazyInstanceNorm2d", () => new nn.LazyInstanceNorm2d(), img],
    ["LazyConvTranspose2d", () => new nn.LazyConvTranspose2d(4, 3), img],
  ];
  for (const [name, make, src] of shapes) {
    out.set(`lazy::굳은뒤::${name}`,
      async () => `(${make().call(src()).shape.join(", ")})`);
  }

  out.set("lazy::굳은뒤::가중치 모양", async () => {
    const m = new nn.LazyLinear(3);
    m.call(x2d());
    return `(${(m as unknown as nn.Linear).weight.shape.join(", ")})`;
  });

  out.set("lazy::성질::같은 씨앗이면 같은 초기화", async () => {
    nn.manualSeed(0);
    const lazy = new nn.LazyLinear(3);
    const got = await lazy.call(x2d()).toArray();
    nn.manualSeed(0);
    const eager = new nn.Linear(5, 3);
    const want = await eager.call(x2d()).toArray();
    const same = got.every((v, i) => Math.abs(v - (want[i] ?? 0)) < 1e-5);
    return `같다=${same ? "True" : "False"}`;
  });

  out.set("lazy::성질::굳은 뒤 학습이 돈다", async () => {
    nn.manualSeed(0);
    const m = new nn.LazyLinear(2);
    const opt = new optim.SGD((m as unknown as nn.Linear).parameters(), 0.1);
    const target = Tensor.zeros([2, 2]);
    let first = 0;
    for (let step = 0; step < 3; step++) {
      const loss = m.call(x2d()).sub(target).square().mean();
      if (step === 0) first = (await loss.toArray())[0] ?? 0;
      opt.zeroGrad();
      loss.backward();
      opt.step();
    }
    const last = (await m.call(x2d()).sub(target).square().mean().toArray())[0] ?? 0;
    return `손실이 내려갔다=${last < first ? "True" : "False"}`;
  });

  // 씨앗이 층 초기화와 dropout 에 닿는가. 코어가 여기서 결함을 하나 냈다 —
  // 파이썬 쪽 주석에 그 이야기를 적었다.
  const same = async (make: () => Tensor) => {
    nn.manualSeed(0);
    const a = await make().toArray();
    nn.manualSeed(0);
    const b = await make().toArray();
    const ok = a.every((v, i) => v === b[i]);
    return `재현된다=${ok ? "True" : "False"}`;
  };
  out.set("lazy::씨앗::Linear 초기화",
    async () => same(() => new nn.Linear(4, 3).weight));
  out.set("lazy::씨앗::Conv2d 초기화",
    async () => same(() => new nn.Conv2d(2, 3, 3).weight));
  out.set("lazy::씨앗::dropout 마스크",
    async () => same(() => Tensor.ones([8]).dropout(0.5, true)));
}

/**
 * 손실 열셋과 거리 셋.
 *
 * 함정 셋은 파이썬 쪽 `loss_cases` 에 적었다 — 짧게: `huber(δ)` 는 `smooth_l1(β=δ)`
 * 의 δ 배이고(δ=1 에서만 같다), `KLDivLoss` 의 `mean` 과 `batchmean` 은 나누는 수가
 * 다르며, `pairwise_distance` 의 `eps` 는 결과가 아니라 **차에** 더해진다.
 */
function addLoss(out: Map<string, Case>): void {
  const X = [0.5, -1.0, 2.0, 1.5, 0.25, -0.5];
  const Y = [1.0, 0.0, -1.0, 0.5, 1.0, 0.25];
  const x = (g = false) => Tensor.from(X, [2, 3], { requiresGrad: g });
  const y = () => Tensor.from(Y, [2, 3]);
  const sgn = () => Tensor.from(Y.map(Math.sign), [2, 3]);
  const counts = () => Tensor.from([1, 2, 0, 3, 0.5, 1], [2, 3]);
  const variance = () => Tensor.from([1, 0.5, 2, 0.25, 1.5, 1], [2, 3]);
  const positive = (g = false) => Tensor.from(X.map(Math.abs).map((v) => v + 0.5),
    [2, 3], { requiresGrad: g });
  const a = (g = false) => Tensor.from([1, 2, 0.5, -1], [2, 2], { requiresGrad: g });
  const b = () => Tensor.from([0.5, 1.5, 1, -0.5], [2, 2]);
  const sign2 = () => Tensor.from([1, -1], [2]);
  const anc = (g = false) => Tensor.from([1, 0, 0, 1], [2, 2], { requiresGrad: g });
  const pos = () => Tensor.from([2, 0.5, 1.5, 1], [2, 2]);
  const neg = () => Tensor.from([1.1, 0.1, 0.2, 0.9], [2, 2]);
  const hinge = () => Tensor.from([0.5, 1.5, 2, 0.25], [2, 2]);
  const htgt = () => Tensor.from([1, -1, -1, 1], [2, 2]);
  const mm = () => Tensor.from([0.1, 0.2, 0.4, 0.8, 0.3, 0.1], [2, 3]);
  const mmt = () => Tensor.from([2, 0], [2]);
  const logp = () => x().logSoftmax(1);
  const tgtp = () => y().softmax(1);

  const value: [string, () => Tensor][] = [
    ["huber(기본)", () => x().huberLoss(y())],
    ["huber(δ=0.5)", () => x().huberLoss(y(), 0.5)],
    ["huber(δ=2)", () => x().huberLoss(y(), 2.0)],
    ["huber(none)", () => x().huberLoss(y(), 1.0, "none")],
    ["huber(sum)", () => x().huberLoss(y(), 1.0, "sum")],
    ["huber(δ=0.5)/smooth_l1(β=0.5)",
      () => x().huberLoss(y(), 0.5).div(x().smoothL1Loss(y(), 0.5))],

    ["kl_div(none)", () => logp().klDiv(tgtp(), "none")],
    ["kl_div(mean)", () => logp().klDiv(tgtp(), "mean")],
    ["kl_div(sum)", () => logp().klDiv(tgtp(), "sum")],
    ["kl_div(batchmean)", () => logp().klDiv(tgtp(), "batchmean")],
    ["kl_div(log_target)", () => logp().klDiv(tgtp().log(), "mean", true)],

    ["poisson(log_input=True,full=False)",
      () => positive().poissonNllLoss(counts(), true, false)],
    ["poisson(log_input=True,full=True)",
      () => positive().poissonNllLoss(counts(), true, true)],
    ["poisson(log_input=False,full=False)",
      () => positive().poissonNllLoss(counts(), false, false)],
    ["poisson(log_input=False,full=True)",
      () => positive().poissonNllLoss(counts(), false, true)],
    ["poisson(none)",
      () => positive().poissonNllLoss(counts(), true, false, 1e-8, "none")],

    ["gaussian(full=False)", () => x().gaussianNllLoss(y(), variance(), false)],
    ["gaussian(full=True)", () => x().gaussianNllLoss(y(), variance(), true)],
    ["gaussian(var<eps)",
      () => x().gaussianNllLoss(y(), Tensor.from([1e-9, 1, 1, 1, 1, 1], [2, 3]),
        false, 1e-6, "none")],
    ["gaussian(eps=1e-2)",
      () => x().gaussianNllLoss(y(), Tensor.from([1e-9, 1, 1, 1, 1, 1], [2, 3]),
        false, 1e-2, "none")],

    ["margin_ranking", () => Tensor.from([1, 2], [2])
      .marginRankingLoss(Tensor.from([2, 1], [2]), sign2(), 0.5)],
    ["margin_ranking(none)", () => Tensor.from([1, 2], [2])
      .marginRankingLoss(Tensor.from([2, 1], [2]), sign2(), 0.5, "none")],
    ["cosine_embedding(margin=0.0)",
      () => a().cosineEmbeddingLoss(b(), sign2(), 0.0, "none")],
    ["cosine_embedding(margin=0.5)",
      () => a().cosineEmbeddingLoss(b(), sign2(), 0.5, "none")],
    ["hinge_embedding(margin=1.0)",
      () => hinge().hingeEmbeddingLoss(htgt(), 1.0, "none")],
    ["hinge_embedding(margin=2.0)",
      () => hinge().hingeEmbeddingLoss(htgt(), 2.0, "none")],
    // 표적이 ±1 이 아닌 자리 — 두 항이 **둘 다** 켜진다.
    ["hinge_embedding(y=0)",
      () => Tensor.from([-1, 0.5, 2], [1, 3])
        .hingeEmbeddingLoss(Tensor.zeros([1, 3]), 1.0, "none")],
    ["soft_margin", () => x().softMarginLoss(sgn())],
    ["soft_margin(none)", () => x().softMarginLoss(sgn(), "none")],

    ["triplet(기본)", () => anc().tripletMarginLoss(pos(), neg())],
    ["triplet(margin=2)", () => anc().tripletMarginLoss(pos(), neg(), 2.0)],
    ["triplet(p=1)", () => anc().tripletMarginLoss(pos(), neg(), 1.0, 1)],
    ["triplet(swap)",
      () => anc().tripletMarginLoss(pos(), neg(), 1.0, 2.0, 1e-6, true)],
    ["triplet(none)",
      () => anc().tripletMarginLoss(pos(), neg(), 1.0, 2.0, 1e-6, false, "none")],
    ["triplet_with_distance(기본)",
      () => new nn.TripletMarginWithDistanceLoss().call(anc(), pos(), neg())],
    ["triplet_with_distance(margin=2)",
      () => new nn.TripletMarginWithDistanceLoss(null, 2.0).call(anc(), pos(), neg())],

    ["multilabel_soft_margin", () => Tensor.from([0.5, -1, 2], [1, 3])
      .multilabelSoftMarginLoss(Tensor.from([1, 0, 1], [1, 3]))],
    ["multi_margin(기본)", () => mm().multiMarginLoss(mmt(), 1, 1.0, null, "none")],
    ["multi_margin(margin=0.5)",
      () => mm().multiMarginLoss(mmt(), 1, 0.5, null, "none")],
    ["multi_margin(p=2)", () => mm().multiMarginLoss(mmt(), 2, 1.0, null, "none")],
    ["multi_margin(weight)",
      () => mm().multiMarginLoss(mmt(), 1, 1.0, Tensor.from([1, 2, 0.5], [3]))],
    ["multilabel_margin", () => Tensor.from([0.1, 0.2, 0.4, 0.8], [1, 4])
      .multilabelMarginLoss(Tensor.from([3, 0, -1, 1], [1, 4]))],

    ["pairwise_distance", () => a().pairwiseDistance(b())],
    ["pairwise_distance(p=1)", () => a().pairwiseDistance(b(), 1)],
    ["pairwise_distance(eps=0)", () => a().pairwiseDistance(b(), 1, 0)],
    ["pairwise_distance(keepdim)", () => a().pairwiseDistance(b(), 2, 1e-6, true)],
    ["pdist", () => Tensor.from([0, 0, 3, 4, 1, 1], [3, 2]).pdist()],
    // torch 에서 이 둘은 최상위와 `F` 가 **글자 그대로 같은 함수**다. 같이 드러난
    // 손실 일곱은 최상위 쪽이 날 ATen 연산이라 서명이 달라서 안 낸다.
    ["최상위::pairwise_distance", () => a().pairwiseDistance(b())],
    ["최상위::pdist", () => Tensor.from([0, 0, 3, 4, 1, 1], [3, 2]).pdist()],
    // 원소 하나를 접기 — 파이썬 쪽 주석에 이유를 적었다. 여기가 0 을 내던 자리다.
    ["원소 하나를 mean",
      () => Tensor.from([1, 2, 3], [3]).sum().binary("mul", Tensor.full([], 1))
        .reshape([1]).mean()],
    ["원소 하나를 sum",
      () => Tensor.from([1, 2, 3], [3]).sum().binary("mul", Tensor.full([], 1))
        .reshape([1]).sum()],
  ];
  for (const [name, fn] of value) out.set(`loss::${name}`, fn);

  const layers: [string, () => Tensor][] = [
    ["HuberLoss", () => new nn.HuberLoss(0.5).call(x(), y())],
    ["KLDivLoss", () => new nn.KLDivLoss("batchmean").call(logp(), tgtp())],
    ["PoissonNLLLoss", () => new nn.PoissonNLLLoss().call(positive(), counts())],
    ["GaussianNLLLoss",
      () => new nn.GaussianNLLLoss().call(x(), y(), variance())],
    ["MarginRankingLoss", () => new nn.MarginRankingLoss(0.5)
      .call(Tensor.from([1, 2], [2]), Tensor.from([2, 1], [2]), sign2())],
    ["CosineEmbeddingLoss",
      () => new nn.CosineEmbeddingLoss().call(a(), b(), sign2())],
    ["HingeEmbeddingLoss",
      () => new nn.HingeEmbeddingLoss().call(hinge(), htgt())],
    ["SoftMarginLoss", () => new nn.SoftMarginLoss().call(x(), sgn())],
    ["TripletMarginLoss",
      () => new nn.TripletMarginLoss().call(anc(), pos(), neg())],
    ["TripletMarginWithDistanceLoss",
      () => new nn.TripletMarginWithDistanceLoss().call(anc(), pos(), neg())],
    ["MultiLabelSoftMarginLoss", () => new nn.MultiLabelSoftMarginLoss()
      .call(Tensor.from([0.5, -1, 2], [1, 3]), Tensor.from([1, 0, 1], [1, 3]))],
    ["MultiMarginLoss", () => new nn.MultiMarginLoss().call(mm(), mmt())],
    ["MultiLabelMarginLoss", () => new nn.MultiLabelMarginLoss()
      .call(Tensor.from([0.1, 0.2, 0.4, 0.8], [1, 4]),
        Tensor.from([3, 0, -1, 1], [1, 4]))],
    ["PairwiseDistance", () => new nn.PairwiseDistance().call(a(), b())],
    ["CosineSimilarity", () => new nn.CosineSimilarity(1).call(a(), b())],
  ];
  for (const [name, fn] of layers) out.set(`loss::층::${name}`, fn);

  // ── 접는 방식은 손실의 일부다 ─────────────────────────────────────────
  const ceX = () => Tensor.from([0.5, -1, 2, 1.5, 0.25, -0.5], [2, 3]);
  const ceT = () => Tensor.from([2, 0], [2], { dtype: "int64" });
  const ceLogp = () => Tensor.from(
    [0.2, 0.5, 0.3, 0.6, 0.1, 0.3].map(Math.log), [2, 3]);
  //
  // `reduceAs` 는 오래 있었는데 **`huberLoss`·`klDiv` 만 쓰고 있었다.** 흔한 넷은
  // `.mean()` 이 박혀 있었다. 코어도 같은 자리에 같은 구멍이었고, 표가 못 본 이유는
  // 교재가 기본값 `mean` 만 쓰기 때문이다.
  //
  // `nllLoss`·`crossEntropy` 는 아직 스칼라만 내므로 여기 없다.
  for (const reduction of ["none", "mean", "sum"] as const) {
    const fns: [string, () => Tensor][] = [
      [`mse_loss(${reduction})`, () => x().mseLoss(y(), reduction)],
      [`l1_loss(${reduction})`, () => x().l1Loss(y(), reduction)],
      [`smooth_l1_loss(${reduction})`,
        () => x().smoothL1Loss(y(), 1.0, reduction)],
      [`huber_loss(${reduction})`, () => x().huberLoss(y(), 1.0, reduction)],
      // 분류 손실 둘. `nllLoss` 는 **뽑자마자 평균을 내고 있어서** `none` 을 만들
      // 자리가 없었다 — 스칼라에서 표본별 값은 되살릴 수 없다.
      [`cross_entropy(${reduction})`,
        () => ceX().crossEntropy(ceT(), reduction)],
      [`nll_loss(${reduction})`, () => ceLogp().nllLoss(ceT(), reduction)],
      // 층 여섯도 한동안 **없었다.** borch.ts 의 `nn` 에 `HuberLoss` 같은 드문 것만
      // 있고 흔한 것이 빠져 있었는데, **결속이 텐서 메서드 위에 스스로 층을 만들어
      // 메꾸고 있어서** 골든이 하나도 못 봤다 — 케이스가 전부 결속을 지나기 때문이다.
      // TypeScript 로 직접 쓰는 사람에게만 없는 이름이었다.
      [`nn.MSELoss(${reduction})`,
        () => new nn.MSELoss(reduction).call(x(), y())],
      [`nn.L1Loss(${reduction})`,
        () => new nn.L1Loss(reduction).call(x(), y())],
      [`nn.SmoothL1Loss(${reduction})`,
        () => new nn.SmoothL1Loss(reduction, 1.0).call(x(), y())],
      [`nn.CrossEntropyLoss(${reduction})`,
        () => new nn.CrossEntropyLoss(reduction).call(ceX(), ceT())],
      [`nn.NLLLoss(${reduction})`,
        () => new nn.NLLLoss(reduction).call(ceLogp(), ceT())],
    ];
    for (const [name, fn] of fns) out.set(`loss::reduction::${name}`, fn);
  }
  // 모르는 값을 평균으로 삼키지 않는다. `batchmean` 은 `klDiv` **에만** 있는 값이라
  // 다른 손실에서는 틀린 이름이다.
  for (const bad of ["MEAN", "batchmean"]) {
    out.set(`loss::reduction::거절::${bad}`, () => {
      try {
        x().l1Loss(y(), bad as "mean");
      } catch (err) {
        return String(err).includes(bad) ? "멈췄다" : `다른 문구 <${err}>`;
      }
      return "안 던졌다";
    });
  }

  // **손실은 기울기가 전부다.** 값이 맞고 기울기가 틀리면 학습이 조용히 다른 데로 간다.
  const grads: [string, (p: Tensor) => Tensor][] = [
    ["huber", (p) => p.huberLoss(y(), 0.5)],
    ["kl_div", (p) => p.logSoftmax(1).klDiv(tgtp())],
    ["poisson", (p) => p.poissonNllLoss(counts())],
    ["gaussian", (p) => p.gaussianNllLoss(y(), variance())],
    ["soft_margin", (p) => p.softMarginLoss(sgn())],
    ["hinge_embedding", (p) => p.hingeEmbeddingLoss(sgn())],
    ["multilabel_soft_margin",
      (p) => p.multilabelSoftMarginLoss(
        Tensor.from(Y.map((v) => (v > 0 ? 1 : 0)), [2, 3]))],
  ];
  for (const [name, fn] of grads) {
    out.set(`loss::grad::${name}`, () => {
      const p = x(true);
      fn(p).backward();
      return gradOf(p, name);
    });
  }
  out.set("loss::grad::triplet", () => {
    const p = anc(true);
    p.tripletMarginLoss(pos(), neg()).backward();
    return gradOf(p, "triplet");
  });
  out.set("loss::grad::cosine_embedding", () => {
    const p = a(true);
    p.cosineEmbeddingLoss(b(), sign2()).backward();
    return gradOf(p, "cosine_embedding");
  });
}

/**
 * 패딩 — 네 모드와 층 열다섯.
 *
 * 입력이 `arange` 라 **답만 보고 어디서 온 값인지** 알 수 있다. 거울인지 되풀이인지
 * 감기인지가 값에 그대로 드러나므로, 이 케이스들이 규약을 통째로 붙잡는다.
 * 자세한 것은 `tests/cases.py` 의 `pad_cases` 에 적었다.
 */
function addPad(out: Map<string, Case>): void {
  const seq = (n: number) => Array.from({ length: n }, (_, i) => i);
  const p1 = (g = false) => Tensor.from(seq(6), [1, 2, 3], { requiresGrad: g });
  const p2 = (g = false) => Tensor.from(seq(12), [1, 1, 3, 4], { requiresGrad: g });
  const p3 = (g = false) => Tensor.from(seq(24), [1, 1, 2, 3, 4], { requiresGrad: g });
  const shapes: [string, (g?: boolean) => Tensor, number[]][] = [
    ["1d", p1, [2, 1]],
    ["2d", p2, [1, 1, 1, 1]],
    ["3d", p3, [1, 1, 1, 1, 1, 1]],
  ];
  const modes = ["constant", "reflect", "replicate", "circular"] as const;
  for (const [tag, src, pads] of shapes) {
    for (const mode of modes) {
      const value = mode === "constant" ? 9 : 0;
      out.set(`pad::${tag}::${mode}`, () => src().padND(pads, mode, value));
      out.set(`pad::grad::${tag}::${mode}`, () => {
        const x = src(true);
        seeded(x.padND(pads, mode)).backward();
        return gradOf(x, `pad ${mode}`);
      });
    }
  }

  out.set("pad::비대칭::reflect", () => p2().padND([1, 2, 0, 1], "reflect"));
  out.set("pad::비대칭::circular", () => p2().padND([2, 1, 1, 0], "circular"));
  out.set("pad::replicate(크게)", () => p1().padND([5, 0], "replicate"));
  out.set("pad::2차원 입력::reflect",
    () => Tensor.arange(6).reshape([2, 3]).padND([1, 1], "reflect"));

  const layers: [string, () => nn.PadNd, () => Tensor][] = [
    ["ReflectionPad1d", () => new nn.ReflectionPad1d(2), p1],
    ["ReflectionPad2d", () => new nn.ReflectionPad2d(1), p2],
    ["ReflectionPad2d(비대칭)", () => new nn.ReflectionPad2d([1, 2, 0, 1]), p2],
    ["ReflectionPad3d", () => new nn.ReflectionPad3d(1), p3],
    ["ReplicationPad1d", () => new nn.ReplicationPad1d(2), p1],
    ["ReplicationPad2d", () => new nn.ReplicationPad2d(1), p2],
    ["ReplicationPad3d", () => new nn.ReplicationPad3d(1), p3],
    ["ZeroPad1d", () => new nn.ZeroPad1d(2), p1],
    ["ZeroPad2d", () => new nn.ZeroPad2d(1), p2],
    ["ZeroPad3d", () => new nn.ZeroPad3d(1), p3],
    ["CircularPad1d", () => new nn.CircularPad1d(2), p1],
    ["CircularPad2d", () => new nn.CircularPad2d(1), p2],
    ["CircularPad3d", () => new nn.CircularPad3d(1), p3],
    ["ConstantPad1d", () => new nn.ConstantPad1d(2, 7), p1],
    ["ConstantPad2d", () => new nn.ConstantPad2d(1, 7), p2],
    ["ConstantPad3d", () => new nn.ConstantPad3d(1, 7), p3],
  ];
  for (const [name, make, src] of layers) {
    out.set(`pad::층::${name}`, () => make().call(src()));
    out.set(`pad::repr::${name}`, async () => make().describe());
  }

  out.set("pad::grad::층::ReflectionPad2d", () => {
    const x = p2(true);
    seeded(new nn.ReflectionPad2d(1).call(x)).backward();
    return gradOf(x, "ReflectionPad2d");
  });

  const refuses = (fn: () => Tensor): string => {
    try {
      fn();
    } catch (e) {
      return (e as Error).name;
    }
    return "예외가 안 났다";
  };
  out.set("pad::거절::reflect(크기 초과)",
    async () => refuses(() => p1().padND([3, 0], "reflect")));
  out.set("pad::거절::짝 개수가 랭크와 안 맞음",
    async () => refuses(() => p2().padND([1, 1], "reflect")));
}

function addNorm(out: Map<string, Case>, inp: Inputs): void {
  const add = (name: string, fn: (x: Tensor) => Tensor, key: string): void => {
    out.set(`norm::${name}`, () => fn(inp.get(key)));
    out.set(`norm::grad::${name}`, () => {
      const x = inp.get(key, true);
      seeded(fn(x)).backward();
      return gradOf(x, name);
    });
  };

  add("F.group_norm(1)", (x) => x.groupNorm(1), "img");
  add("F.group_norm(3)", (x) => x.groupNorm(3), "img");
  const gn = (groups: number): nn.Module => new nn.GroupNorm(groups, 3);
  out.set("norm::nn.GroupNorm(1,3)", () => gn(1).call(inp.get("img")));
  out.set("norm::nn.GroupNorm(3,3)", () => gn(3).call(inp.get("img")));
  out.set("norm::nn.GroupNorm/파라미터 이름",
    () => Object.keys(gn(3).namedParameters()).join(" "));

  add("F.instance_norm", (x) => x.instanceNorm(), "img");
  for (const [nd, key] of [["1d", "nd_seq"], ["2d", "img"], ["3d", "nd_vol"]] as const) {
    out.set(`norm::nn.InstanceNorm${nd}`,
      () => new nn.InstanceNormND().call(inp.get(key)));
  }

  add("F.rms_norm", (x) => x.rmsNorm(1), "img");
  out.set("norm::nn.RMSNorm", () => new nn.RMSNorm(4).call(inp.get("img")));

  // **`normalizedShape` 는 접는 축의 개수다.** 축 하나짜리로만 물으면 "마지막 축을
  // 접는다" 와 답이 같아서 그 규칙이 안 보인다 — 셋 다 그렇게 적혀 있었다.
  out.set("norm::nn.LayerNorm(축 하나)",
    () => new nn.LayerNorm(4).call(inp.get("img")));
  out.set("norm::nn.LayerNorm(축 둘)",
    () => new nn.LayerNorm([4, 4]).call(inp.get("img")));
  out.set("norm::grad::nn.LayerNorm(축 둘)", () => {
    const x = inp.get("img", true);
    seeded(new nn.LayerNorm([4, 4]).call(x)).backward();
    return gradOf(x, "LayerNorm(축 둘)");
  });
  // 골든이 굳힌 것은 **예외 종류의 이름**이다. "멈췄는가" 로 접으면 오타가 낸
  // `TypeError` 도 통과한다.
  out.set("norm::nn.LayerNorm(모양 불일치)", () => {
    try {
      new nn.LayerNorm([3, 4]).call(inp.get("img"));
    } catch (e) {
      return e instanceof Error ? e.constructor.name : typeof e;
    }
    return "예외가 안 났다";
  });
  out.set("norm::nn.LayerNorm/파라미터 이름",
    () => Object.keys(new nn.LayerNorm(4).namedParameters()).join(" "));
  out.set("norm::nn.LayerNorm(affine 끄면)",
    () => Object.keys(
      new nn.LayerNorm(4, 1e-5, false).namedParameters()).join(" ") || "없음");

  add("F.conv_transpose1d", (x) => x.convTransposeND(inp.get("tw1")), "nd_seq");
  add("F.conv_transpose2d", (x) => x.convTransposeND(inp.get("tw2")), "img");
  add("F.conv_transpose2d(스트라이드2)",
    (x) => x.convTransposeND(inp.get("tw2"), null, 2), "img");
  add("F.conv_transpose2d(패딩1)",
    (x) => x.convTransposeND(inp.get("tw2"), null, 1, 1), "img");
  add("F.conv_transpose2d(편향)",
    (x) => x.convTransposeND(inp.get("tw2"), inp.get("tb")), "img");
  add("F.conv_transpose3d", (x) => x.convTransposeND(inp.get("tw3")), "nd_vol");

  // **가중치 쪽 기울기도 본다.** 입력 쪽만 보면 축이 뒤집힌 것을 놓친다.
  out.set("norm::grad::conv_transpose2d/가중치", () => {
    const w = inp.get("tw2", true);
    seeded(inp.get("img").convTransposeND(w)).backward();
    return gradOf(w, "conv_transpose2d 가중치");
  });

  for (const [nd, key, wk, bk, spatial] of [
    ["1d", "nd_seq", "tw1", "tb", 1],
    ["2d", "img", "tw2", "tb", 2],
    ["3d", "nd_vol", "tw3", "tb3", 3],
  ] as const) {
    out.set(`norm::nn.ConvTranspose${nd}`, () => {
      const w = inp.get(wk);
      // 차원을 고정한 이름을 지나간다 — `ND` 판만 부르면 그 셋을 아무도 안 잰다.
      const Cls = { 1: nn.ConvTranspose1d, 2: nn.ConvTranspose2d,
        3: nn.ConvTranspose3d }[spatial] ?? nn.ConvTranspose2d;
      const layer = new Cls(w.shape[0] ?? 1, w.shape[1] ?? 1, w.shape[2] ?? 1);
      layer.loadStateDict({ weight: w, bias: inp.get(bk) });
      return layer.call(inp.get(key));
    });
  }
}

/**
 * 옵티마이저 다섯과 스케줄러 여덟. **여러 스텝 뒤를 묻는다.**
 *
 * 옵티마이저는 상태를 쌓으므로 첫 스텝에서는 서로 비슷하게 군다. 한 스텝만 재면
 * 다섯 개를 전부 SGD 로 구현해도 통과한다.
 *
 * 스케줄러가 하는 일은 학습률의 **수열**이라 그 수열을 통째로 굳힌다. 마지막 값만
 * 보면 가는 길이 달라도 통과하고, 실제로 `LinearLR` 과 `ConstantLR` 은 끝에서 만난다.
 */
function addOpt(out: Map<string, Case>, inp: Inputs): void {
  const model = (): nn.Sequential => {
    const m = new nn.Sequential([
      new nn.Linear(6, 8), new nn.ReLU(), new nn.Linear(8, 3),
    ]);
    m.loadStateDict({
      "0.weight": inp.get("w0"), "0.bias": inp.get("b0"),
      "2.weight": inp.get("w1"), "2.bias": inp.get("b1"),
    });
    return m;
  };

  const trained = (make: (ps: Tensor[]) => optim.Optimizer): nn.Sequential => {
    const m = model();
    const opt = make(m.parameters());
    const crit = new nn.CrossEntropyLoss();
    for (let i = 0; i < 5; i++) {
      opt.zeroGrad();
      crit.forward(m.forward(inp.get("train_x")), inp.get("train_y")).backward();
      opt.step();
    }
    return m;
  };

  const kinds: [string, (ps: Tensor[]) => optim.Optimizer][] = [
    ["Adagrad", (ps) => new optim.Adagrad(ps, 0.1)],
    ["Adadelta", (ps) => new optim.Adadelta(ps, 0.5)],
    ["Adamax", (ps) => new optim.Adamax(ps, 0.05)],
    ["NAdam", (ps) => new optim.NAdam(ps, 0.05)],
    ["RAdam", (ps) => new optim.RAdam(ps, 0.05)],
    ["ASGD", (ps) => new optim.ASGD(ps, 0.05)],
    ["Rprop", (ps) => new optim.Rprop(ps, 0.05)],
    // **2 차원 가중치라야 Adafactor 의 요점이 돈다** — 여기 `0.weight` 가 (8, 6) 이다.
    ["Adafactor", (ps) => new optim.Adafactor(ps, 0.05)],
    // **감쇠가 물어야 `AdamW` 가 `Adam` 과 갈린다.** 둘을 가르는 것은 감쇠가 놓이는
    // 자리 하나뿐이다 — 모멘트가 보기 전의 기울기냐(`Adam`), 갱신 뒤의 가중치냐
    // (`AdamW`). `weightDecay=0` 이면 두 갈래가 다 사라져 같은 옵티마이저가 되므로,
    // 기본값으로 만든 케이스는 어느 구현으로도 통과한다.
    ["Adam(weight_decay)",
      (ps) => new optim.Adam(ps, 0.05, 0.9, 0.999, 1e-8, 0.1)],
    ["AdamW", (ps) => new optim.AdamW(ps, 0.05, 0.9, 0.999, 1e-8, 0.1)],
    // **`weightDecay` 를 받는 옵티마이저 전부를 0 이 아닌 값으로 묻는다.**
    //
    // 오늘까지 그런 케이스가 하나도 없었고, 그 부재가 결손 일곱을 숨겼다 — 결속이
    // 이 인자를 받아서 일곱 자리에서 버리고 있었고, 그중 `NAdam` 은 **인자 수까지
    // 맞아서** arity 검사로도 안 보였다.
    //
    // 자리에 주의: `Adagrad` 는 `eps` **앞**, `NAdam` 은 `momentumDecay` **앞**이다.
    // 끝에 붙이면 조용히 다른 인자가 된다.
    ["Adagrad(weight_decay)",
      (ps) => new optim.Adagrad(ps, 0.1, 0, 0.1)],
    ["Adadelta(weight_decay)",
      (ps) => new optim.Adadelta(ps, 0.5, 0.9, 1e-6, 0.1)],
    ["Adamax(weight_decay)",
      (ps) => new optim.Adamax(ps, 0.05, 0.9, 0.999, 1e-8, 0.1)],
    ["NAdam(weight_decay)",
      (ps) => new optim.NAdam(ps, 0.05, 0.9, 0.999, 1e-8, 0.1)],
    ["RAdam(weight_decay)",
      (ps) => new optim.RAdam(ps, 0.05, 0.9, 0.999, 1e-8, 0.1)],
    ["RMSprop(weight_decay)",
      (ps) => new optim.RMSprop(ps, 0.01, 0.99, 1e-8, 0.1)],
    ["ASGD(weight_decay)",
      (ps) => new optim.ASGD(ps, 0.05, 1e-4, 0.75, 1e6, 0.1)],
    ["SGD(weight_decay)", (ps) => new optim.SGD(ps, 0.05, 0, 0.1)],
  ];
  for (const [name, make] of kinds) {
    out.set(`opt::${name}/0.weight`, () => {
      const w = trained(make).namedParameters()["0.weight"];
      if (!w) throw new Error("0.weight 가 없다");
      return w;
    });
    out.set(`opt::${name}/손실`, () => new nn.CrossEntropyLoss()
      .forward(trained(make).forward(inp.get("train_x")), inp.get("train_y")));
  }

  // ── 옵티마이저의 `state_dict` — 이어서 학습하기가 여기 걸려 있다 ─────
  //
  // 모델 가중치만 저장하고 옵티마이저를 안 저장하면, 이어 붙인 학습이 **모멘텀과
  // 2 차 모먼트를 잃은 채로** 다시 시작한다. 예외는 안 나고 손실 곡선만 한 번 튄다.
  //
  // **열쇠 이름이 아니라 값을 묻는다.** torch 는 `{state, param_groups}` 이고
  // 이쪽은 은행 구조라 모양이 다르다 — 모양을 물으면 영원히 갈리는데, 정작 중요한
  // "끊었다 이으면 안 끊은 것과 같은가" 는 모양과 무관하게 물을 수 있다.
  const stepOnce = (m: nn.Sequential, o: optim.Optimizer): void => {
    o.zeroGrad();
    new nn.CrossEntropyLoss()
      .forward(m.forward(inp.get("train_x")), inp.get("train_y")).backward();
    o.step();
  };
  const resume = (make: (ps: Tensor[]) => optim.Optimizer, carry = true): Tensor => {
    const m = model();
    const opt = make(m.parameters());
    for (let i = 0; i < 3; i++) stepOnce(m, opt);
    const saved = opt.stateDict();
    // **새 옵티마이저에 얹는다.** 같은 것에 도로 넣으면 아무것도 안 물은 것이 된다.
    const fresh = make(m.parameters());
    if (carry) fresh.loadStateDict(saved);
    for (let i = 0; i < 2; i++) stepOnce(m, fresh);
    const w = m.namedParameters()["0.weight"];
    if (!w) throw new Error("0.weight 가 없다");
    return w;
  };
  // `SGD(momentum=0)` 은 상태가 없어서 저장을 통째로 빼먹어도 지난다 — 상태를 가진
  // 것으로 물어야 한다.
  const resumable: [string, (ps: Tensor[]) => optim.Optimizer][] = [
    ["SGD", (ps) => new optim.SGD(ps, 0.1, 0.9)],
    ["Adam", (ps) => new optim.Adam(ps, 0.05)],
    ["RMSprop", (ps) => new optim.RMSprop(ps, 0.05)],
  ];
  for (const [name, make] of resumable) {
    out.set(`opt::${name}/이어서 학습하기`, () => resume(make));
  }
  // **위의 셋이 정말 무언가를 옮겼는가.** `loadStateDict` 가 빈 일을 해도 위 셋은
  // 통과할 수 있다 — 그래서 **둘의 차이**를 답으로 굳힌다. 얹기가 아무 일도 안 하는
  // 구현에서는 이것이 0 이 되고, 그때 위의 셋은 초록인 채로 아무것도 안 재고 있다.
  out.set("opt::상태를 안 옮기면 갈린다", () => {
    const make = (ps: Tensor[]): optim.Optimizer => new optim.Adam(ps, 0.05);
    return resume(make, true).sub(resume(make, false));
  });

  // **스케줄러도 이어져야 한다.** 옵티마이저만 되돌리고 스케줄러를 새로 세우면
  // 학습률이 **처음 값으로 돌아간다** — 반쯤 식혀 놓은 학습이 다시 뜨거워진다.
  //
  // **글자로 돌려준다.** 학습률의 자취는 수열이라, 갈렸을 때 어느 칸에서 갈렸는지가
  // 그대로 보여야 한다 — 텐서로 주면 "최대차 1.5e-01" 만 남는다.
  const schedResume = (carry: boolean): string => {
    const m = model();
    const opt = new optim.SGD(m.parameters(), 0.2);
    const sch = new optim.StepLR(opt, 2, 0.5);
    const seen: string[] = [];
    const lrNow = (): string => {
      const group = opt.paramGroups[0];
      return `${Number((group?.lr ?? 0).toFixed(6))}`;
    };
    for (let i = 0; i < 4; i++) {
      seen.push(lrNow());
      opt.step();
      sch.step();
    }
    const saved = sch.stateDict();
    const fresh = new optim.StepLR(opt, 2, 0.5);
    if (carry) fresh.loadStateDict(saved);
    for (let i = 0; i < 4; i++) {
      seen.push(lrNow());
      opt.step();
      fresh.step();
    }
    return seen.join(" ");
  };
  out.set("opt::StepLR/이어서 학습하기", () => schedResume(true));
  // 얹기가 빈 일이면 두 자취가 같아지고, 그러면 위 케이스는 아무것도 안 재고 있다.
  out.set("opt::스케줄러 상태를 안 옮기면 갈린다",
    () => `${schedResume(true)} | ${schedResume(false)}`);

  // ── `save`/`load` — 위의 것들이 쓸모가 있으려면 파일이 되어야 한다 ────
  //
  // `stateDict()` 를 셋 다 맞춰 놓고도 그것을 **쓸 방법이 없으면** 이어서 학습하기는
  // 한 세션 안의 이야기다. 탭을 새로고침하면 사라진다.
  //
  // 형식이 서로 다르므로(torch 는 pickle, 이쪽은 safetensors) **바이트가 아니라
  // 왕복을 묻는다.** 파이썬 쪽은 임시 파일을 거치고 여기는 바이트 그대로인데,
  // 묻는 것은 같다 — 쓴 것을 되읽으면 같은 것이 나오는가.
  /** 되읽은 것에서 텐서 표를 꺼낸다. 나무는 `Savable` 이라 좁혀야 쓴다. */
  const asTensors = (got: Savable): Record<string, Tensor> => {
    if (got === null || typeof got !== "object" || Array.isArray(got)
        || got instanceof Tensor) {
      throw new Error("되읽은 것이 텐서 표가 아니다");
    }
    return got as Record<string, Tensor>;
  };

  out.set("opt::save/load 가 state_dict 를 왕복한다", async () => {
    const w = asTensors(load(await save(model().stateDict())))["0.weight"];
    if (!w) throw new Error("0.weight 가 없다");
    return w;
  });
  // 되읽은 것이 **정말 쓸 수 있는가.** 열쇠와 값이 맞아도 그대로 못 얹으면 소용없다.
  out.set("opt::되읽은 것을 그대로 얹을 수 있다", async () => {
    const bytes = await save(model().stateDict());
    const dst = new nn.Sequential(
      new nn.Linear(6, 8), new nn.ReLU(), new nn.Linear(8, 3));
    dst.loadStateDict(asTensors(load(bytes)));
    return dst.forward(inp.get("train_x"));
  });

  // **교재의 관용구는 중첩이다.** `{model: …, opt: …, epoch: 3}` 를 통째로 저장한다.
  // 평평한 텐서 표만 되면 그 코드가 안 돈다 — 오래 안 됐다.
  out.set("opt::save/load 가 중첩을 왕복한다", async () => {
    const m = model();
    const opt = new optim.Adam(m.parameters(), 0.05);
    const sd = opt.stateDict();
    const bytes = await save({
      model: m.stateDict(), opt: sd, epoch: 3, note: "half way",
    });
    const got = load(bytes);
    if (got === null || typeof got !== "object" || Array.isArray(got)
        || got instanceof Tensor) {
      throw new Error("되읽은 것이 사전이 아니다");
    }
    const keys = Object.keys(got).sort().join(" ");
    return `${keys} epoch=${String(got.epoch)} note=${String(got.note)}`;
  });

  // **`stateDict` 의 열쇠에는 이미 점이 들어 있다**(`0.weight`). 편 이름을 점으로
  // 다시 쪼개 되돌리면 `{model: {0: {weight: …}}}` 가 나온다 — 값은 다 있는데 구조가
  // 달라서, 되읽은 것을 `loadStateDict` 에 넣으면 그때 터진다.
  out.set("opt::중첩 안의 점 찍힌 열쇠가 안 쪼개진다", async () => {
    const got = load(await save({ model: model().stateDict() }));
    if (got === null || typeof got !== "object" || Array.isArray(got)
        || got instanceof Tensor) {
      throw new Error("되읽은 것이 사전이 아니다");
    }
    return Object.keys(asTensors(got.model as Savable)).sort().join(" ");
  });

  /** 학습률의 자취. **옵티마이저를 실제로 밟는다** — 순서가 값을 정한다. */
  const trace = (
    make: (o: optim.Optimizer) => { step: () => void },
    steps: number,
  ): Tensor => {
    const opt = new optim.SGD(model().parameters(), 0.2);
    const sch = make(opt);
    const seen: number[] = [];
    for (let i = 0; i < steps; i++) {
      seen.push(opt.paramGroups[0]?.lr ?? 0);
      opt.step();
      sch.step();
    }
    return Tensor.from(seen, [steps]);
  };

  out.set("opt::ConstantLR/자취",
    () => trace((o) => new optim.ConstantLR(o, 0.5, 3).start(), 8));
  out.set("opt::LinearLR/자취",
    () => trace((o) => new optim.LinearLR(o, 0.5, 1.0, 4).start(), 8));
  out.set("opt::PolynomialLR/자취",
    () => trace((o) => new optim.PolynomialLR(o, 5, 2.0).start(), 8));
  out.set("opt::MultiplicativeLR/자취",
    () => trace((o) => new optim.MultiplicativeLR(o, () => 0.9).start(), 6));
  out.set("opt::CosineAnnealingWarmRestarts/자취",
    () => trace((o) => new optim.CosineAnnealingWarmRestarts(o, 3, 2).start(), 10));
  out.set("opt::OneCycleLR/자취",
    () => trace((o) => new optim.OneCycleLR(o, 0.4, 10).start(), 10));

  out.set("opt::SequentialLR/자취", () => trace((o) => {
    const a = new optim.ConstantLR(o, 0.25, 3).start();
    const b = new optim.ExponentialLR(o, 0.8).start();
    return new optim.SequentialLR(o, [a, b], [3]);
  }, 8));
  out.set("opt::ChainedScheduler/자취", () => trace((o) => {
    const a = new optim.ConstantLR(o, 0.5, 2).start();
    const b = new optim.ExponentialLR(o, 0.9).start();
    return new optim.ChainedScheduler([a, b]);
  }, 6));

  // ── 갈래를 좁혀 묻는 자리 ────────────────────────────────────────────────
  //
  // 위의 모델 학습은 옵티마이저가 **대충 맞으면** 지난다. 갈래를 정하는 인자들은
  // 파라미터 하나에 기울기를 손으로 먹여야 드러난다.
  const start = () => Tensor.from([1, -2, 0.5], [3], { requiresGrad: true });
  const ramp = (i: number) => Tensor.from(
    [0.1 * (i + 1), -0.3 * (i + 1), 0.2 * (i + 1)], [3]);
  // **부호가 뒤집히는 기울기.** Rprop 의 `etas` 와 "뒤집힌 칸은 안 간다" 규칙이
  // 여기서만 보인다.
  const flipGrads = [[0.1, -0.3, 0.2], [-0.1, -0.3, 0.2],
    [-0.2, -0.3, 0.2], [-0.2, 0.3, 0.2]];

  const walk = (
    make: (ps: Tensor[]) => optim.Optimizer,
    grads: (i: number) => Tensor,
    steps = 4,
  ) => () => {
    const p = start();
    const opt = make([p]);
    const seen: Tensor[] = [];
    for (let i = 0; i < steps; i++) {
      opt.zeroGrad();
      p.grad = grads(i);
      opt.step();
      // **여기서 값을 베껴야 한다.** `reshape` 는 버퍼를 그대로 물려주므로 그냥
      // 담으면 네 줄이 전부 같은 자리를 가리키고, `cat` 이 나중에 읽을 때는 마지막
      // 값만 넷 나온다 — 자취가 아니라 한 점이 된다. 0 을 더해 새 버퍼로 옮긴다.
      seen.push(p.reshape([1, 3]).detach().add(Tensor.full([], 0)));
    }
    return Tensor.cat(seen, 0);
  };
  const flip = (i: number) => Tensor.from(flipGrads[i] ?? [0, 0, 0], [3]);

  out.set("opt::ASGD/기본값", walk((ps) => new optim.ASGD(ps), ramp));
  out.set("opt::ASGD/lambd",
    walk((ps) => new optim.ASGD(ps, 0.1, 0.01), ramp));
  out.set("opt::ASGD/alpha",
    walk((ps) => new optim.ASGD(ps, 0.1, 1e-4, 0.5), ramp));
  // **`t0` 을 낮춰야 평균이 실제로 돈다** — 기본값 100만에서 `mu` 는 늘 1 이다.
  out.set("opt::ASGD/t0(평균이 도는 자리)",
    walk((ps) => new optim.ASGD(ps, 0.1, 1e-4, 0.75, 2), ramp));
  out.set("opt::ASGD/weight_decay",
    walk((ps) => new optim.ASGD(ps, 0.1, 1e-4, 0.75, 1e6, 0.1), ramp));

  out.set("opt::Rprop/기본값", walk((ps) => new optim.Rprop(ps), ramp));
  out.set("opt::Rprop/부호 바뀜",
    walk((ps) => new optim.Rprop(ps, 0.1), flip));
  out.set("opt::Rprop/etas",
    walk((ps) => new optim.Rprop(ps, 0.1, 0.4, 1.5), flip));
  out.set("opt::Rprop/step_sizes 상한",
    walk((ps) => new optim.Rprop(ps, 0.1, 0.5, 1.2, 1e-6, 0.11), ramp));

  out.set("opt::Adafactor/기본값", walk((ps) => new optim.Adafactor(ps), ramp));
  out.set("opt::Adafactor/weight_decay",
    walk((ps) => new optim.Adafactor(ps, 0.1, -0.8, null, 1e-3, 1.0, 0.1), ramp));
  out.set("opt::Adafactor/d",
    walk((ps) => new optim.Adafactor(ps, 0.1, -0.8, null, 1e-3, 2.0), ramp));

  // **2 차원부터 행·열로 쪼갠다** — 벡터로만 물으면 그 길이 한 번도 안 돌아간다.
  const matrixWalk = (shape: number[]) => () => {
    const n = shape.reduce((a, b) => a * b, 1);
    const p = Tensor.from(
      Array.from({ length: n }, (_, i) => i / 4 - 0.5), shape, { requiresGrad: true });
    const opt = new optim.Adafactor([p], 0.1);
    for (let i = 0; i < 3; i++) {
      opt.zeroGrad();
      p.grad = Tensor.from(
        Array.from({ length: n }, (_, k) => (k / 8 - 0.2) * (i + 1)), shape);
      opt.step();
    }
    return p;
  };
  out.set("opt::Adafactor/2차원", matrixWalk([3, 4]));
  out.set("opt::Adafactor/3차원", matrixWalk([2, 3, 4]));

  // **원소 하나짜리 텐서는 값으로 캐시된다** — `Tensor.owned` 가 있는 까닭이다.
  // 옵티마이저 상태는 그 버퍼에 제자리로 쓰므로, 크기 1 파라미터에서 둘이 만나면
  // 프로그램 전체가 쓰는 상수가 조용히 덮어써진다. 그래서 **학습을 시킨 뒤에** 같은
  // 상수로 곱해 본다. 답은 자명하고, 그 자명한 답이 이 결함을 잡는다.
  //
  // 파이썬 쪽 짝과 같은 이유로 **한 걸음으로는 못 잡고**(Rprop 의 첫 걸음은 폭을 안
  // 바꾼다), 상수도 0·1 을 함께 본다(상태 은행이 0 에서 시작한다), 그리고 전용 커널을
  // 쓰는 `SGD`·`Adam`·`RMSprop` 을 꼭 섞는다 — 그 셋은 공통 밑동 밖이다.
  out.set("opt::크기 1 파라미터가 상수를 안 더럽힌다", () => {
    const makers: ((ps: Tensor[]) => optim.Optimizer)[] = [
      (ps) => new optim.Rprop(ps, 0.05),
      (ps) => new optim.Adafactor(ps, 0.05),
      (ps) => new optim.ASGD(ps, 0.05),
      (ps) => new optim.Adagrad(ps, 0.05),
      (ps) => new optim.SGD(ps, 0.05, 0.9),
      (ps) => new optim.Adam(ps, 0.05),
      (ps) => new optim.RMSprop(ps, 0.05),
    ];
    for (const make of makers) {
      const p = Tensor.from([0.5], [1], { requiresGrad: true });
      const opt = make([p]);
      for (let i = 0; i < 3; i++) {
        opt.zeroGrad();
        p.grad = Tensor.from([0.1 * (i + 1)], [1]);
        opt.step();
      }
    }
    const probe = Tensor.from([1, 2], [2]);
    return Tensor.cat([0, 1, 0.05].map((k) => probe.mul(Tensor.full([], k))), 0);
  });

  // **이 여섯만 케이스 본문이 비동기다.** `LBFGS.step` 이 한 걸음 안에서 스칼라를
  // 읽어 분기하기 때문이다(`optim.ts` 의 설명 참조). 골든 러너는 본문을 `await`
  // 하므로 여기서만 `async` 를 써도 나머지는 그대로다.
  const lbfgs = (steps: number, make: (ps: Tensor[]) => optim.LBFGS) => async () => {
    const p = start();
    const opt = make([p]);
    const seen: Tensor[] = [];
    for (let i = 0; i < steps; i++) {
      // **닫힘이 기울기를 넣어 준다** — 미분하지 않는다. 그러면 반복마다 `flat` 이
      // 그대로라 이력이 안 쌓이고, 남는 것은 첫 반복의 경사하강과 보폭 규칙뿐이다.
      // 파이썬 쪽 주석이 말하는 바로 그 반쪽이고, 여기서도 같은 반쪽을 잰다.
      await opt.step(() => {
        p.grad = Tensor.from([0.1, -0.3, 0.2], [3]);
        return p.mul(p).sum();
      });
      seen.push(p.detach().clone());
    }
    return Tensor.stack(seen);
  };
  out.set("opt::LBFGS/기본값", lbfgs(3, (ps) => new optim.LBFGS(ps, 0.1)));
  out.set("opt::LBFGS/max_iter", lbfgs(3, (ps) => new optim.LBFGS(ps, 0.1, 3)));
  out.set("opt::LBFGS/history_size",
    lbfgs(3, (ps) => new optim.LBFGS(ps, 0.5, 5, null, 1e-7, 1e-9, 2)));

  // 자리마다 곡률이 다른 이차식 — 준뉴턴법이 이기는 모양이고, 이력이 실제로 찬다.
  const curve = () => Tensor.from([1, 4, 9], [3]);
  const lbfgsReal = (steps: number, make: (ps: Tensor[]) => optim.LBFGS) => async () => {
    const p = start();
    const w = curve();
    const opt = make([p]);
    const seen: Tensor[] = [];
    for (let i = 0; i < steps; i++) {
      await opt.step(() => {
        p.grad = null;
        const out = p.mul(p).mul(w).sum();
        out.backward();
        return out;
      });
      seen.push(p.detach().clone());
    }
    return Tensor.stack(seen);
  };
  out.set("opt::LBFGS/진짜 기울기", lbfgsReal(3, (ps) => new optim.LBFGS(ps, 0.1)));
  out.set("opt::LBFGS/이력이 밀려난다",
    lbfgsReal(2, (ps) => new optim.LBFGS(ps, 0.5, 8, null, 1e-7, 1e-9, 2)));
  out.set("opt::LBFGS/문턱 근처에서 멈춘다",
    lbfgsReal(2, (ps) => new optim.LBFGS(ps, 0.3, 12, null, 1e-7, 1e-3)));

  // **오르내림을 다르게, 주기를 여러 번.** 같은 폭에 한 주기만 밟으면 `stepSizeDown`
  // 도 `triangular2` 도 있는지 안 보인다.
  out.set("opt::CyclicLR/자취",
    () => trace((o) => new optim.CyclicLR(o, 0.01, 0.1, 3).start(), 14));
  out.set("opt::CyclicLR(위아래 다름)/자취",
    () => trace((o) => new optim.CyclicLR(o, 0.01, 0.1, 2, 4).start(), 14));
  out.set("opt::CyclicLR(triangular2)/자취",
    () => trace((o) => new optim.CyclicLR(
      o, 0.01, 0.1, 3, null, "triangular2").start(), 14));
  // **`exp_range` 의 기준은 주기가 아니라 걸음이다.**
  out.set("opt::CyclicLR(exp_range)/자취",
    () => trace((o) => new optim.CyclicLR(
      o, 0.01, 0.1, 3, null, "exp_range", 0.9).start(), 14));
}

/**
 * Dropout. **값이 아니라 성질을 묻는다.**
 *
 * 답이 난수기에 달려 있고 우리 난수기가 torch 의 것과 같을 이유가 없다. 그렇다고
 * 안 물으면 층 하나가 통째로 검사 밖에 남으므로, **양쪽이 똑같이 답할 수 있는 것**만
 * 묻는다 — 평가 모드는 항등인가, 살아남은 값이 `1/(1-p)` 배인가, 대략 `p` 만큼
 * 떨구는가, 기울기가 살아남은 자리로만 흐르는가, 두 번 부르면 다른가.
 */
function addDropout(out: Map<string, Case>, inp: Inputs): void {
  const big = (grad = false): Tensor => {
    // 골든 쪽과 같은 입력 — `train_x` 를 40 번 쌓은 것이다. 비율을 재려면 표본이
    // 많아야 하고, 작으면 난수의 흔들림이 답을 흔든다.
    const src = inp.get("train_x");
    const rows = src.shape[0] ?? 1;
    const cols = src.shape[1] ?? 1;
    const tiled = src.expand(40, rows, cols).reshape([40 * rows, cols]);
    return grad ? asLeaf(tiled) : tiled;
  };

  out.set("dropout::eval 은 항등", () => inp.get("x2").dropout(0.5, false));
  out.set("dropout::p=0 은 항등", () => inp.get("x2").dropout(0, true));
  out.set("dropout::p=1 은 전부 0", () => inp.get("x2").dropout(1, true));
  out.set("dropout::nn.Dropout(eval) 은 항등",
    () => new nn.Dropout(0.5).eval().call(inp.get("x2")));

  out.set("dropout::살아남은 값은 1/(1-p) 배", async () => {
    const x = big();
    const made = await x.dropout(0.5, true).toArray();
    const src = await x.toArray();
    let worst = 0;
    let any = false;
    for (let i = 0; i < made.length; i++) {
      const o = made[i] ?? 0;
      const s = src[i] ?? 0;
      if (o === 0 || s === 0) continue;
      any = true;
      worst = Math.max(worst, Math.abs(o / s - 2));
    }
    if (!any) return "아무것도 안 남았다";
    return worst < 1e-4 ? "맞다" : `배율이 ${worst.toPrecision(3)} 만큼 어긋난다`;
  });

  out.set("dropout::대략 p 만큼 떨군다", async () => {
    const made = await big().dropout(0.5, true).toArray();
    const zeros = made.reduce((a, v) => a + (v === 0 ? 1 : 0), 0) / made.length;
    return Math.abs(zeros - 0.5) < 0.05 ? "대략 맞다" : `${zeros.toFixed(3)} 이 떨어졌다`;
  });

  out.set("dropout::기울기는 살아남은 자리로만", async () => {
    const x = big(true);
    const made = x.dropout(0.5, true);
    made.sum().backward();
    const values = await made.toArray();
    const grad = x.grad;
    if (!grad) return "기울기가 없다";
    const got = await grad.toArray();
    let stray = 0;
    for (let i = 0; i < values.length; i++) {
      if ((values[i] ?? 0) === 0 && (got[i] ?? 0) !== 0) stray += 1;
    }
    return stray === 0 ? "살아남은 자리로만" : `떨군 자리 ${stray} 곳에 흘렀다`;
  });

  out.set("dropout::두 번 부르면 다른 자리", async () => {
    const x = big();
    const a = await x.dropout(0.5, true).toArray();
    const b = await x.dropout(0.5, true).toArray();
    for (let i = 0; i < a.length; i++) {
      if (((a[i] ?? 0) === 0) !== ((b[i] ?? 0) === 0)) return "다르다";
    }
    return "두 번이 같다";
  });
}

/**
 * `scaledDotProductAttention`. **요즘 트랜스포머 코드가 직접 부르는 이름이다.**
 *
 * 가림막이 곱셈이 아니라 덧셈이라는 것이 가장 흔한 오해다 — 큰 음수를 더해 softmax
 * 가 0 을 내게 하는 것이지 0 을 곱하는 것이 아니다.
 */
function addSdpa(out: Map<string, Case>, inp: Inputs): void {
  const mask = (): Tensor => {
    const rows: number[] = [];
    for (let i = 0; i < 5; i++) {
      for (let j = 0; j < 5; j++) rows.push(j >= 3 ? -1e9 : 0);
    }
    return Tensor.from(rows, [5, 5]);
  };

  const shapes: [string, (x: Tensor) => Tensor][] = [
    ["맨 것", (x) => nn.scaledDotProductAttention(x, x, x)],
    ["더하는 가림막", (x) => nn.scaledDotProductAttention(x, x, x, mask())],
    ["인과", (x) => nn.scaledDotProductAttention(x, x, x, null, true)],
  ];
  for (const [name, fn] of shapes) {
    out.set(`sdpa::${name}`, () => fn(inp.get("attn_x")));
    out.set(`sdpa::grad::${name}`, () => {
      const q = inp.get("attn_x", true);
      seeded(fn(q)).backward();
      return gradOf(q, name);
    });
  }

  // **셋을 같은 것으로만 주면 인자를 뒤바꿔 써도 값이 같아서 안 걸린다.**
  out.set("sdpa::q·k·v 가 다를 때", () => {
    const q = inp.get("attn_x");
    const k = q.mul(Tensor.full([], 0.5)).add(Tensor.full([], 0.1));
    const v = q.flip(0);
    return nn.scaledDotProductAttention(q, k, v);
  });
}

/**
 * 파이썬 쪽에서 `torch.sum(x)` 처럼 **모듈 함수로** 부르는 꼴.
 *
 * TypeScript 에는 그런 두 번째 이름이 없다 — 여기서는 메서드가 유일한 부르는 법이고,
 * 자유 함수를 따로 두면 표면만 는다. 그래서 이 케이스들은 **같은 답을 메서드로**
 * 낸다. 골든이 묻는 것은 값이고, 부르는 문법은 언어마다 다를 수 있다.
 */
function addModFn(out: Map<string, Case>, inp: Inputs): void {
  const m = (): Tensor => inp.get("x2");
  const table: [string, () => Tensor][] = [
    ["sum", () => m().sum()],
    ["sum(dim)", () => m().sumDim(1)],
    ["mean", () => m().mean()],
    ["mean(dim)", () => m().mean(0)],
    ["std", () => m().std()],
    ["var", () => m().variance()],
    ["numel", () => Tensor.from([m().size], [])],
    // `flat` 은 바깥에 안 열려 있다 — `reshape` 로 같은 것을 한다.
    ["argmax", () => m().reshape([m().size]).argmax(0)],
    ["argmin(dim)", () => m().argmin(1)],
    ["clone", () => m().clone()],
    ["detach", () => m().detach()],
    ["flatten", () => m().reshape([m().size])],
    ["permute", () => m().permute([1, 0])],
    ["transpose", () => m().transpose()],
    ["squeeze", () => inp.get("x1").reshape([1, 6, 1]).squeeze(2).squeeze(0)],
    // `max` 는 축을 주면 **짝**을 낸다. 값 쪽을 꺼낸다.
    ["max", () => m().reshape([m().size]).max(0).values],
    ["max(dim)/값", () => m().max(1).values],
    ["min(dim)/번호", () => m().min(1).indices],
  ];
  for (const [name, fn] of table) out.set(`modfn::${name}`, fn);

  out.set("modfn::relu_(원본이 바뀐다)", () => {
    const t = inp.get("x1").clone();
    t.inplaceUnary("relu");
    return t;
  });

  // ── 파이썬 쪽에서 `torch.add(a, b)` 처럼 부르는 **두 번째 이름들.** ────────
  //
  // TypeScript 에는 그 두 번째 이름이 없다 — 메서드가 유일한 부르는 법이고, 자유
  // 함수를 나란히 두면 표면만 는다. 여기서는 **같은 답을 메서드로** 낸다.
  const a2 = (): Tensor => inp.get("x2");
  const b2 = (): Tensor => a2().mul(Tensor.full([], 0.5)).add(Tensor.full([], 1));
  const line = (): Tensor => inp.get("x1").narrow(0, 0, 4);
  const neg = (): Tensor => Tensor.from([-5, -3, 3, 5], [1, 4]);
  const three = (): Tensor => Tensor.full([], 3);

  const aliases: [string, () => Tensor][] = [
    ["add", () => a2().add(b2())],
    ["add(alpha)", () => a2().add(b2().mul(Tensor.full([], 2)))],
    ["sub", () => a2().sub(b2())],
    ["mul", () => a2().mul(b2())],
    ["div", () => a2().div(b2())],
    ["div(floor)", () => a2().div(b2()).unary("floor")],
    ["rsub", () => b2().sub(a2())],
    // **`remainder` 와 `fmod` 는 음수에서 갈린다** — 부호가 반대쪽을 따른다.
    ["remainder(음수)", () => neg().sub(neg().div(three()).unary("floor").mul(three()))],
    ["fmod(음수)", () => neg().sub(neg().div(three()).unary("trunc").mul(three()))],
    ["floor_divide(음수)", () => neg().div(three()).unary("floor")],
    ["greater", () => a2().binary("gt", b2())],
    ["greater_equal", () => a2().binary("ge", b2())],
    ["less", () => a2().binary("lt", b2())],
    ["less_equal", () => a2().binary("le", b2())],
    ["not_equal", () => a2().binary("ne", b2())],
    ["hstack(1차원)", () => Tensor.cat([line(), line()], 0)],
    ["hstack(2차원)", () => Tensor.cat([a2(), b2()], 1)],
    ["vstack(1차원)", () => Tensor.cat([line().reshape([1, 4]), line().reshape([1, 4])], 0)],
    ["column_stack(1차원)",
      () => Tensor.cat([line().reshape([4, 1]), line().reshape([4, 1])], 1)],
    // **`dstack` 은 뒤에 축을 붙인다** — 앞이 아니다.
    ["dstack", () => Tensor.cat([a2().reshape([3, 4, 1]), b2().reshape([3, 4, 1])], 2)],
    ["concat", () => Tensor.cat([a2(), b2()], 0)],
    ["t(2차원)", () => a2().transpose()],
    ["t(1차원은 그대로)", () => line()],
    ["adjoint", () => a2().transpose()],
    ["moveaxis", () => a2().movedim(0, 1)],
    ["broadcast_to", () => line().reshape([1, 4]).expand(3, 4)],
    ["broadcast_tensors", () => line().reshape([1, 4]).expand(3, 4)],
  ];
  for (const [name, fn] of aliases) out.set(`modfn::${name}`, fn);

  // 대각선에 블록을 늘어놓고 나머지는 0.
  out.set("modfn::block_diag", () => {
    const a = a2();
    const b = b2().narrow(0, 0, 1);
    const top = Tensor.cat([a, Tensor.zeros([3, 4])], 1);
    const bottom = Tensor.cat([Tensor.zeros([1, 4]), b], 1);
    return Tensor.cat([top, bottom], 0);
  });
}

/**
 * 풀링의 나머지 차원과 나머지 종류.
 *
 * **적응형은 나눠떨어지지 않을 때가 요점이다.** 창을 자리마다 다르게 잡는 규칙이
 * torch 와 갈리면 값이 조용히 다르고, 떨어지는 경우만 물으면 그 규칙을 한 번도
 * 안 보게 된다.
 */
function addPool(out: Map<string, Case>, inp: Inputs): void {
  const add = (name: string, fn: (x: Tensor) => Tensor, key: string): void => {
    out.set(`pool::${name}`, () => fn(inp.get(key)));
    out.set(`pool::grad::${name}`, () => {
      const x = inp.get(key, true);
      seeded(fn(x)).backward();
      return gradOf(x, name);
    });
  };

  add("F.avg_pool1d", (x) => x.poolND("avg", 2, 2), "nd_seq");
  add("F.avg_pool3d", (x) => x.poolND("avg", 2, 2), "nd_vol");
  // **`nn.` 이 붙은 케이스는 층을 지나가야 한다.** 바로 위 `F.` 짝이 이미 텐서
  // 메서드를 재고 있으므로, 여기서 같은 메서드를 부르면 두 케이스가 같은 것을 두 번
  // 묻고 층 이름은 아무도 안 재는 상태가 된다 — 실제로 그랬다.
  out.set("pool::nn.AvgPool1d",
    () => new nn.AvgPool1d(2, 2).call(inp.get("nd_seq")));
  out.set("pool::nn.AvgPool3d",
    () => new nn.AvgPool3d(2, 2).call(inp.get("nd_vol")));

  add("F.adaptive_avg_pool1d(4)", (x) => x.adaptivePool("avg", 4), "nd_seq");
  add("F.adaptive_avg_pool1d(3)", (x) => x.adaptivePool("avg", 3), "nd_seq");
  add("F.adaptive_avg_pool3d", (x) => x.adaptivePool("avg", 2), "nd_vol");
  out.set("pool::nn.AdaptiveAvgPool1d",
    () => new nn.AdaptiveAvgPool1d(4).call(inp.get("nd_seq")));
  out.set("pool::nn.AdaptiveAvgPool3d",
    () => new nn.AdaptiveAvgPool3d(2).call(inp.get("nd_vol")));

  add("F.adaptive_max_pool1d", (x) => x.adaptivePool("max", 4), "nd_seq");
  add("F.adaptive_max_pool2d", (x) => x.adaptivePool("max", 2), "img");
  add("F.adaptive_max_pool2d(안 떨어짐)", (x) => x.adaptivePool("max", 3), "img");
  add("F.adaptive_max_pool3d", (x) => x.adaptivePool("max", 2), "nd_vol");
  for (const [nd, key, size] of [
    ["1d", "nd_seq", 4], ["2d", "img", 2], ["3d", "nd_vol", 2],
  ] as const) {
    out.set(`pool::nn.AdaptiveMaxPool${nd}`,
      () => inp.get(key).adaptivePool("max", size));
  }

  add("F.lp_pool1d(p=2)", (x) => x.lpPool(2, 2), "nd_seq");
  add("F.lp_pool2d(p=2)", (x) => x.lpPool(2, 2), "img");
  add("F.lp_pool2d(p=1)", (x) => x.lpPool(1, 2), "img");
  out.set("pool::nn.LPPool2d", () => inp.get("img").lpPool(2, 2));
}

/**
 * 파이썬 쪽에 새로 생긴 함수들. **여기서는 조합으로 같은 답을 낸다.**
 *
 * borch.ts 에 이름을 늘리지 않는다 — 계산이 느는 것이 아니라 파이썬이 부를 철자가
 * 느는 것이고, 그것은 파이썬 쪽 일이다.
 */
function addNewFn(out: Map<string, Case>, inp: Inputs): void {
  const x1 = (): Tensor => inp.get("x1");
  const x2 = (): Tensor => inp.get("x2");
  const withnan = (): Tensor =>
    Tensor.from([1, Number.NaN, -Infinity, Infinity, 3], [5]);
  const zeros5 = (): Tensor => Tensor.zeros([5]);

  // 난수 계열은 값이 같을 수 없다 — **모양을 답으로 굳힌다.**
  for (const name of ["empty_like", "rand_like", "randn_like"]) {
    out.set(`newfn::${name}/모양`, () => x2().shape.join(" "));
  }
  out.set("newfn::randint_like/모양", () => x2().shape.join(" "));

  const table: [string, () => Tensor][] = [
    ["logspace", () => Tensor.full([], 10).binary("pow", Tensor.linspace(0, 2, 5))],
    ["scalar_tensor", () => Tensor.full([], 2.5)],
    // `xy` 는 앞의 두 축이 뒤바뀐 것이라 한 규칙으로 못 덮는다.
    ["meshgrid/0", () => x1().narrow(0, 0, 3).reshape([3, 1]).expand(3, 2)],
    ["meshgrid/1", () => x1().narrow(0, 0, 2).reshape([1, 2]).expand(3, 2)],
    ["meshgrid(xy)", () => x1().narrow(0, 0, 3).reshape([1, 3]).expand(2, 3)],
    ["lerp", () => x1().add(x1().mul(Tensor.full([], 2)).sub(x1())
      .mul(Tensor.full([], 0.25)))],
    ["nan_to_num", () => nanFix(withnan(), 0, 3.4028234663852886e38,
      -3.4028234663852886e38)],
    ["nan_to_num(값 지정)", () => nanFix(withnan(), 0.5, 9, -9)],
    ["isclose", () => x1().sub(x1()).abs()
      .binary("le", x1().abs().mul(Tensor.full([], 1e-5)).add(Tensor.full([], 1e-8)))],
    // 실수만 있으므로 전부 참이다 — 거짓말이 아니라 사실이다.
    ["isreal", () => Tensor.ones([5]).binary("gt", Tensor.full([], 0))],
    ["isposinf", () => withnan().unary("isinf")
      .mul(withnan().binary("gt", Tensor.full([], 0)).to("float32"))
      .binary("gt", Tensor.full([], 0))],
    ["isneginf", () => withnan().unary("isinf")
      .mul(withnan().binary("lt", Tensor.full([], 0)).to("float32"))
      .binary("gt", Tensor.full([], 0))],
    // **NaN 을 건너뛴다** — `maximum` 은 NaN 을 물고 나온다.
    ["fmax(NaN 건너뜀)", () => nanSkip(withnan(), zeros5(), "maximum")],
    ["fmin(NaN 건너뜀)", () => nanSkip(withnan(), zeros5(), "minimum")],
    ["float_power", () => inp.get("xp").powScalar(2)],
    ["logical_xor", () => {
      const a = Tensor.from([1, 0, 1, 0], [4]).binary("ne", Tensor.full([], 0));
      const b = Tensor.from([1, 1, 0, 0], [4]).binary("ne", Tensor.full([], 0));
      return a.binary("ne", b);
    }],
    ["isin", () => {
      const e = Tensor.from([1, 2, 3, 4], [4]).reshape([4, 1]);
      const t = Tensor.from([2, 4], [2]).reshape([1, 2]);
      return e.binary("eq", t).to("float32").sumDim(1)
        .binary("gt", Tensor.full([], 0)).reshape([4]);
    }],
    ["var_mean/분산", () => x2().variance()],
    ["var_mean/평균", () => x2().mean()],
    ["std_mean/표준편차", () => x2().std()],
    ["inner", () => x2().mm(x2().transpose())],
    ["vdot", () => x1().mul(x1()).sum()],
    ["kron", () => x1().narrow(0, 0, 2).reshape([2, 1])
      .mul(x1().narrow(0, 2, 2).reshape([1, 2])).reshape([4])],
    ["cross", () => crossOf(x1().narrow(0, 0, 3).reshape([1, 3]),
      x1().narrow(0, 3, 3).reshape([1, 3]))],
  ];
  for (const [name, fn] of table) out.set(`newfn::${name}`, fn);
}

/** NaN·무한대를 유한한 수로. 채울 값도 **같은 모양으로 펴서** 넘긴다. */
function nanFix(t: Tensor, nan: number, hi: number, lo: number): Tensor {
  const like = (v: number): Tensor => Tensor.zeros(t.shape).add(Tensor.full([], v));
  const isInf = t.unary("isinf");
  const pos = isInf.mul(t.binary("gt", Tensor.full([], 0)).to("float32"))
    .binary("gt", Tensor.full([], 0));
  const negInf = isInf.mul(t.binary("lt", Tensor.full([], 0)).to("float32"))
    .binary("gt", Tensor.full([], 0));
  let outv = like(nan).where(t.unary("isnan"), t);
  outv = like(hi).where(pos, outv);
  return like(lo).where(negInf, outv);
}

function nanSkip(a: Tensor, b: Tensor, kind: "maximum" | "minimum"): Tensor {
  const picked = a.binary(kind, b);
  const first = b.where(a.unary("isnan"), picked);
  return a.where(b.unary("isnan"), first);
}

function crossOf(a: Tensor, b: Tensor): Tensor {
  const p = (t: Tensor, i: number): Tensor => t.narrow(1, i, 1);
  return Tensor.cat([
    p(a, 1).mul(p(b, 2)).sub(p(a, 2).mul(p(b, 1))),
    p(a, 2).mul(p(b, 0)).sub(p(a, 0).mul(p(b, 2))),
    p(a, 0).mul(p(b, 1)).sub(p(a, 1).mul(p(b, 0))),
  ], 1);
}

/**
 * 색인으로 **쓰는** 쪽. 읽는 쪽(`gather`)은 이미 있었다.
 *
 * **번호가 겹칠 때가 요점이다.** `scatterSet` 은 마지막에 쓴 것이 남고
 * `scatterAdd` 는 더한다 — 안 겹치는 번호로만 재면 둘이 같아 보인다.
 */
function addIndex(out: Map<string, Case>, inp: Inputs): void {
  const base = (): Tensor => Tensor.zeros([3, 4]);
  const src = (grad = false): Tensor => {
    const s = inp.get("x2").mul(Tensor.full([], 10));
    return grad ? asLeaf(s) : s;
  };
  // 0 이 두 번 나온다 — 겹치는 자리가 여기다.
  const dup = (): Tensor =>
    Tensor.from([0, 0, 1, 2, 1, 1, 2, 3, 2, 2, 3, 0], [3, 4]);
  const rows = (values: number[]): Tensor => Tensor.from(values, [values.length]);

  /** 1 차원 번호를 줄 단위로 편다 — `index_add` 류가 쓰는 모양이다. */
  const spread = (index: Tensor, dim: number, shape: number[]): Tensor => {
    const lifted = shape.map(() => 1);
    lifted[dim] = index.size;
    return index.reshape(lifted).expand(...shape);
  };

  const table: [string, () => Tensor][] = [
    ["scatter(겹치는 번호)", () => base().scatterSet(1, dup(), src())],
    ["scatter_add(겹치는 번호)", () => base().scatterAdd(1, dup(), src())],
    ["scatter(스칼라)",
      () => base().scatterSet(1, dup(), Tensor.zeros([3, 4]).add(Tensor.full([], 7)))],
    ["index_add", () => base().scatterAdd(0, spread(rows([0, 0, 2]), 0, [3, 4]),
      inp.get("x2"))],
    ["index_copy", () => base().scatterSet(0, spread(rows([2, 1, 0]), 0, [3, 4]),
      inp.get("x2"))],
    ["index_fill", () => inp.get("x2").scatterSet(
      1, spread(rows([0, 2]), 1, [3, 2]),
      Tensor.zeros([3, 2]).add(Tensor.full([], -1)))],
    // `take` 는 평평하게 펴서 뽑는다 — 축이라는 개념이 없다.
    ["take", () => inp.get("x2").reshape([12]).indexSelect(0, rows([0, 2, 2, 5]))],
    ["take_along_dim", () => inp.get("x2").gather(1, dup())],
  ];
  for (const [name, fn] of table) out.set(`index::${name}`, fn);

  out.set("index::grad::scatter_add", () => {
    const s = src(true);
    seeded(base().scatterAdd(1, dup(), s)).backward();
    return gradOf(s, "scatter_add");
  });

  // 정렬된 것 안에서 자리를 찾는다.
  //
  // **이 셋이 오래 `searchSorted` 를 안 불렀다.** `seq < want` 를 퍼뜨려 더하는
  // 것으로 적혀 있었고, 값은 정확히 같아서 초록이었다 — 다만 그때 재던 것은
  // 퍼뜨리기와 축약이었지 이 이름이 아니었다. borch.ts 에 그 이름이 없었으므로
  // 그것이 그때 할 수 있는 전부이기도 했다.
  const seq = (): Tensor => Tensor.from([1, 3, 5, 7], [4]);
  const want = (): Tensor => Tensor.from([0, 3, 6, 9], [4]);
  out.set("index::searchsorted", () => seq().searchSorted(want()));
  out.set("index::searchsorted(right)", () => seq().searchSorted(want(), true));
  out.set("index::bucketize", () => want().bucketize(seq()));

  // **같은 것을 두 이름으로 받는 자리**는 파이썬 쪽 이야기다 — `right`(참거짓)와
  // `side`(글자)를 맞춰 보는 일, 그리고 둘이 어긋날 때 멈추는 일. borch.ts 는
  // 하나만 알고, 그 하나가 맞는지를 여기서 묻는다.
  out.set("index::searchsorted(side=left)", () => seq().searchSorted(want(), false));
  out.set("index::searchsorted(side=right)", () => seq().searchSorted(want(), true));
  out.set("index::searchsorted(side=right, right=True)",
    () => seq().searchSorted(want(), true));

  // 경계가 하나뿐이거나 값이 경계를 벗어나는 자리. **이진 탐색의 양 끝**이고,
  // 가운데만 물으면 `lo`·`hi` 의 초기값이 틀려도 답이 맞는다.
  out.set("index::searchsorted(끝 밖)", () =>
    Tensor.from([2, 4], [2]).searchSorted(Tensor.from([0, 1, 2, 3, 4, 5], [6])));
  out.set("index::searchsorted(경계 하나)", () =>
    Tensor.from([3], [1]).searchSorted(Tensor.from([1, 3, 5], [3]), true));
}

/**
 * 최근에 늘어난 이름들 중 **TS 쪽에서 실제로 부를 것들.**
 *
 * ## 왜 전부가 아닌가
 *
 * 파이썬 골든이 2,173 건인데 여기 본문이 있는 것은 그 일부다. 남는 수가 크게
 * 벌어졌는데(500건 넘게), 그것을 전부 옮기는 것은 **같은 질문을 두 번 하는 일**이다 —
 * 결속 러너가 이미 그 케이스들에서 borch.ts 커널을 지나므로 **값은 검증되고 있고**,
 * 실제로 이번 묶음들의 셰이더 오류와 `mutate` 초과 복사를 잡은 것도 결속 러너였다.
 *
 * TS 케이스가 **추가로** 증명하는 것은 값이 아니라 **이 쪽 표면**이다 — `asStrided`
 * 라는 이름이 그 자리에 있고 인자 순서가 그러한가. 그것은 값어치가 있지만 500 건에
 * 걸릴 값어치는 아니고, 남는 것 중 상당수는 `borch.i0` 같은 **파이썬 이름 별칭**을
 * 묻는 케이스라 옮기면 정말로 같은 질문이 두 번이 된다.
 *
 * 그래서 골랐다. 기준은 **TS 로 코드를 쓰는 사람이 이 이름을 부르는가**다.
 *
 * ## 입력이 문자 그대로인 것만
 *
 * 파이썬 쪽 케이스 몇은 입력을 `numpy.random.default_rng(0)` 로 만든다. 그 수열은
 * 여기서 못 만들므로 그 케이스는 **안 옮긴다** — 비슷한 값으로 채우면 이름은 등록되고
 * 답은 갈리는데, 그것은 결함을 알리는 것이 아니라 **없는 결함을 만드는 것**이다.
 */
function addRecent(out: Map<string, Case>): void {
  // ── 비트·정수 (`bit::`) ─────────────────────────────────────────────
  //
  // 음수와 0 이 함께 있어야 한다 — 오른쪽 시프트가 산술인지, `gcd` 가 부호를 버리는지,
  // `lcm(0, 7)` 이 0 으로 안 나누는지가 전부 여기 달렸다.
  const ints = (): Tensor => Tensor.from([12, 10, -3, 0], [4], { dtype: "int64" });
  const rhs = (): Tensor => Tensor.from([10, 3, 5, 7], [4], { dtype: "int64" });
  for (const name of ["bitwise_and", "bitwise_or", "bitwise_xor",
    "bitwise_left_shift", "bitwise_right_shift", "gcd", "lcm"]) {
    out.set(`bit::${name}`, () => ints().binary(name, rhs()));
  }

  // **메서드 이름으로도 묻는다.** 위의 일곱은 `binary(이름)` 표를 거치는데, torch 에서
  // 옮겨 온 코드가 치는 줄은 `x.gcd(y)` 다 — 그 이름이 오래 없었다.
  const asMethod: [string, (a: Tensor, b: Tensor) => Tensor][] = [
    ["bitwise_and", (a, b) => a.bitwiseAnd(b)],
    ["bitwise_or", (a, b) => a.bitwiseOr(b)],
    ["bitwise_xor", (a, b) => a.bitwiseXor(b)],
    ["bitwise_left_shift", (a, b) => a.bitwiseLeftShift(b)],
    ["bitwise_right_shift", (a, b) => a.bitwiseRightShift(b)],
    ["gcd", (a, b) => a.gcd(b)],
    ["lcm", (a, b) => a.lcm(b)],
  ];
  for (const [name, fn] of asMethod) {
    out.set(`bit::메서드::${name}`, () => fn(ints(), rhs()));
  }

  out.set("bit::bitwise_not", () => ints().bitwise_not());
  // **참거짓은 다른 계산이다** — `~True` 는 `-2` 가 아니라 `False` 다. 정수로만
  // 물으면 이 갈래가 통째로 안 돌아간다.
  const flags4b = (): Tensor => Tensor.from([1, 1, 0, 0], [4], { dtype: "bool" });
  const notFlags = (): Tensor => Tensor.from([0, 0, 1, 1], [4], { dtype: "bool" });
  out.set("bit::bitwise_and(참거짓)", () => flags4b().bitwiseAnd(notFlags()));
  out.set("bit::bitwise_or(참거짓)", () => flags4b().bitwiseOr(notFlags()));
  out.set("bit::bitwise_not(참거짓)", () => flags4b().bitwise_not());

  // 제자리 판. **같은 텐서를 돌려줘야** 이어 부르는 코드가 원본을 고친다.
  const inPlacePairs: [string, (x: Tensor) => Tensor][] = [
    ["gcd_", (x) => x.gcd_(rhs())],
    ["lcm_", (x) => x.lcm_(rhs())],
  ];
  for (const [name, fn] of inPlacePairs) {
    out.set(`bit::제자리::${name}`, () => {
      const x = ints();
      fn(x);
      return x;
    });
    out.set(`bit::제자리::${name}(같은 텐서)`, () => {
      const x = ints();
      return verdict(fn(x) === x);
    });
  }

  const reals = (): Tensor => Tensor.from([-2.5, 0.5, 1.5, 3.0], [4]);
  out.set("bit::clamp_max", () => reals().clampMax(1.0));
  out.set("bit::clamp_min", () => reals().clampMin(0.0));
  out.set("bit::제자리::clamp_max_", () => {
    const x = reals();
    x.clampMax_(1.0);
    return x;
  });
  out.set("bit::제자리::clamp_min_", () => {
    const x = reals();
    x.clampMin_(0.0);
    return x;
  });
  out.set("bit::arctan2", () => reals().arctan2(reals().add(Tensor.full([], 1))));
  // **같은 텐서**에서 그래프를 끊는다. `detach()` 로 잘못 지으면 `===` 가 거짓이
  // 되고, 원본은 여전히 위쪽에 붙어 있어 역전파가 계속 흐른다.
  out.set("bit::detach_", () => {
    const x = Tensor.from([-2.5, 0.5, 1.5, 3.0], [4], { requiresGrad: true });
    const y = x.mul(Tensor.full([], 2));
    const z = y.detach_();
    return `${verdict(z === y)} ${verdict(y.requiresGrad)}`;
  });
  // **`fill` 은 `fill_` 과 달리 제자리가 아니다.** 한 글자 차이라 값만 보면
  // 그럴듯하고, 원본이 그대로인지를 따로 물어야 드러난다.
  out.set("bit::fill", () => reals().fill(7.0));
  out.set("bit::fill(원본은 그대로)", () => {
    const x = reals();
    x.fill(7.0);
    return x;
  });
  out.set("bit::i0", () => reals().i0());
  // 급수가 3.75 에서 갈린다 — 그 너머를 따로 묻는다.
  out.set("bit::i0(큰 값)", () => Tensor.from([4.0, 8.0, 12.0], [3]).i0());
  out.set("bit::nextafter", () =>
    Tensor.from([1.0, 2.0], [2]).binary("nextafter", Tensor.from([2.0, 1.0], [2])));
  const frexpIn = (): Tensor => Tensor.from([1.0, 0.5, 8.0, -3.0], [4]);
  out.set("bit::frexp(가수)", () => frexpIn().frexp().mantissa);
  out.set("bit::frexp(지수)", () => frexpIn().frexp().exponent);
  const gam3 = (): Tensor => Tensor.from([2.0, 3.0, 4.5], [3]);
  out.set("bit::mvlgamma(p=2)", () => gam3().mvlgamma(2));
  out.set("bit::mvlgamma(p=3)", () => gam3().mvlgamma(3));

  // **축을 가로로도 세로로도 묻는다** — 정사각이 아닌 것으로 물어야 축이 바뀌면
  // 모양에서 먼저 걸린다.
  const grid23 = (): Tensor => Tensor.from([1.0, 2.0, -1.0, 3.0, 4.0, 0.5], [2, 3]);
  for (const dim of [0, 1]) {
    out.set(`bit::logcumsumexp(dim=${dim})`, () => grid23().logcumsumexp(dim));
  }
  // **고르지 않은 무게로 센다.** 전부 1 이면 누적의 순서가 상쇄되어 뒤에서부터
  // 쌓이는 규칙이 안 드러난다.
  out.set("bit::grad::logcumsumexp", () => {
    const x = Tensor.from([1.0, 2.0, -1.0, 3.0, 4.0, 0.5], [2, 3],
      { requiresGrad: true });
    x.logcumsumexp(1)
      .mul(Tensor.from([1.0, 2.0, 0.5, 0.5, 3.0, 1.5], [2, 3])).sum().backward();
    return gradOf(x, "logcumsumexp");
  });

  // 창 함수. **`periodic` 이 기본이고 그것이 길이를 하나 늘린다.**
  const windows: [string, (n: number, p: boolean) => Tensor][] = [
    ["bartlett_window", (n, p) => Tensor.bartlettWindow(n, p)],
    ["blackman_window", (n, p) => Tensor.blackmanWindow(n, p)],
    ["hamming_window", (n, p) => Tensor.hammingWindow(n, p)],
    ["hann_window", (n, p) => Tensor.hannWindow(n, p)],
    ["kaiser_window", (n, p) => Tensor.kaiserWindow(n, p)],
  ];
  for (const [name, make] of windows) {
    for (const periodic of [true, false]) {
      out.set(`bit::${name}(6, periodic=${periodic ? "True" : "False"})`,
        () => make(6, periodic));
    }
    // 나누는 자리가 0 이 되는 유일한 크기.
    out.set(`bit::${name}(1)`, () => make(1, true));
  }
  out.set("bit::hamming_window(alpha, beta)",
    () => Tensor.hammingWindow(6, true, 0.5, 0.5));
  out.set("bit::kaiser_window(beta=8)", () => Tensor.kaiserWindow(6, true, 8.0));

  // ── 모양·색인 (`spot::`) ────────────────────────────────────────────
  const grid = (): Tensor => Tensor.from(
    Array.from({ length: 12 }, (_, i) => i), [3, 4]);
  const line = (): Tensor => Tensor.from(
    Array.from({ length: 10 }, (_, i) => i), [10]);

  out.set("spot::as_strided", () => grid().asStrided([2, 2], [1, 2]));
  out.set("spot::as_strided(offset)", () => grid().asStrided([2, 2], [1, 2], 3));
  // **겹치는 걸음.** 안 겹치면 한 칸을 두 번 읽는 자리가 없다.
  out.set("spot::as_strided(겹침)", () => grid().asStrided([3, 3], [1, 1]));
  out.set("spot::as_strided_scatter",
    () => grid().asStridedScatter(Tensor.zeros([2, 2]), [2, 2], [1, 2], 3));

  out.set("spot::select_scatter",
    () => grid().selectScatter(Tensor.zeros([4]), 0, 1));
  out.set("spot::slice_scatter",
    () => grid().sliceScatter(Tensor.zeros([3, 2]), 1, 1, 3));
  // **`step` 이 1 이 아니어야** 건너뛰는 자리를 안 건드리는지 드러난다.
  out.set("spot::slice_scatter(step=2)",
    () => grid().sliceScatter(Tensor.zeros([3, 2]), 1, 0, 4, 2));
  // 길이가 offset 을 따라 변한다 — (3,4) 에서 0·1 은 셋이고 -1 은 둘이다.
  for (const [offset, k] of [[-1, 2], [0, 3], [1, 3]] as const) {
    out.set(`spot::diagonal_scatter(offset=${offset})`,
      () => grid().diagonalScatter(Tensor.zeros([k]), offset));
  }
  // **배치 축이 있어야** 대각선 축이 맨 뒤로 가는 규약이 드러난다.
  out.set("spot::diag_embed(2차)", () => grid().diagEmbed());

  // **나머지를 앞에서부터 나눠 갖는다** — 10 을 4 로 쪼개면 3·3·2·2 다.
  for (const k of [3, 4, 5]) {
    out.set(`spot::tensor_split(${k})`, () => Tensor.cat(line().tensorSplit(k), 0));
    // 이어 붙이면 어떻게 나눴는지가 사라진다. 조각 크기 자체를 묻는다.
    out.set(`spot::tensor_split(${k}, 조각 크기)`,
      () => Tensor.from(line().tensorSplit(k).map((p) => p.shape[0] ?? 0), [k]));
  }
  out.set("spot::split_with_sizes",
    () => line().splitWithSizes([2, 3, 5])[1] ?? Tensor.zeros([0]));

  const mask = (): Tensor => Tensor.from(
    [1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0], [3, 4], { dtype: "bool" });
  const feed = (): Tensor => Tensor.from(
    Array.from({ length: 12 }, (_, i) => 100 + i), [12]);
  out.set("spot::masked_scatter", () => grid().maskedScatter(mask(), feed()));

  // **번호가 겹친다** — 0 이 두 번 나온다. 여기서만 두 갈래가 갈린다.
  const flatIdx = (): Tensor => Tensor.from([0, 0, 5], [3]);
  const flatVal = (): Tensor => Tensor.from([-1.0, -2.0, -3.0], [3]);
  for (const acc of [false, true]) {
    out.set(`spot::put(accumulate=${acc ? "True" : "False"})`,
      () => grid().put(flatIdx(), flatVal(), acc));
    out.set(`spot::index_put(accumulate=${acc ? "True" : "False"})`,
      () => grid().indexPut(
        [Tensor.from([0, 1, 0], [3]), Tensor.from([1, 2, 1], [3])],
        Tensor.from([10.0, 20.0, 30.0], [3]), acc));
  }

  // **밑판이 2.5 다.** 1 이면 곱하기에서 항등원이라 `include_self` 가 안 보인다.
  const base34 = (): Tensor => Tensor.zeros([3, 4]).add(Tensor.full([], 2.5));
  const dup34 = (): Tensor =>
    Tensor.from([0, 0, 1, 2, 1, 1, 2, 3, 2, 2, 3, 0], [3, 4]);
  for (const reduce of ["sum", "prod", "amax", "amin", "mean"]) {
    for (const self of [true, false]) {
      out.set(`spot::scatter_reduce(${reduce}, include_self=${self ? "True" : "False"})`,
        () => base34().scatterReduce(1, dup34(), grid(), reduce, self));
    }
  }

  // 첫 줄은 이미 작아서 **안 건드려야** 한다. 나머지 둘은 깎인다.
  const tall32 = (): Tensor => Tensor.from([3, 4, 6, 8, 30, 40], [3, 2]);
  for (const p of [1, 2, 3]) {
    out.set(`spot::renorm(p=${p})`, () => tall32().renorm(p, 0, 5.0));
  }
  out.set("spot::renorm(dim=1)", () => tall32().renorm(2, 1, 5.0));

  // **음수가 섞인 입력이다.** 거듭제곱으로 짜면 WGSL 의 `pow` 가 여기서 NaN 이 된다.
  const trio = (): Tensor => Tensor.from([1.0, -2.0, 3.0], [3]);
  out.set("spot::vander", () => Tensor.vander(trio()));
  out.set("spot::vander(N=2)", () => Tensor.vander(trio(), 2));
  out.set("spot::vander(increasing)", () => Tensor.vander(trio(), undefined, true));
  out.set("spot::vander(N=5)", () => Tensor.vander(trio(), 5));
  for (const offset of [-1, 0, 1]) {
    out.set(`spot::tril_indices(offset=${offset})`,
      () => Tensor.trilIndices(3, 4, offset));
    out.set(`spot::triu_indices(offset=${offset})`,
      () => Tensor.triuIndices(3, 4, offset));
  }
  out.set("spot::ger", () => trio().ger(Tensor.from([4.0, 5.0], [2])));
  out.set("spot::mv", () => grid().mv(Tensor.from([1, 0, 0, 2], [4])));

  // **고르지 않은 무게.** 전부 1 이면 자리마다 다른 몫이 상쇄되어 안 보인다.
  const spotW = (): Tensor => Tensor.from(
    [1.0, 2.0, 0.5, 3.0, 2.0, 0.5, 1.5, 1.0, 0.25, 3.0, 2.0, 0.75], [3, 4]);
  const spotGrad = (
    name: string, fn: (x: Tensor) => Tensor, w: () => Tensor = spotW,
  ): void => {
    out.set(`spot::grad::${name}`, () => {
      const x = Tensor.from(Array.from({ length: 12 }, (_, i) => i), [3, 4],
        { requiresGrad: true });
      const got = fn(x);
      got.mul(w().reshape(got.shape)).sum().backward();
      return gradOf(x, name);
    });
  };
  /** 넣은 값 쪽 기울기. **넣은 자리로만** 흘러야 한다. */
  const spotSrcGrad = (
    name: string, fn: (t: Tensor, v: Tensor) => Tensor, src: () => Tensor,
  ): void => {
    out.set(`spot::grad(넣는 값)::${name}`, () => {
      const v = src();
      const got = fn(grid(), v);
      got.mul(spotW().reshape(got.shape)).sum().backward();
      return gradOf(v, name);
    });
  };
  const leafOnes = (shape: number[]): Tensor => Tensor.from(
    Array.from({ length: shape.reduce((a, b) => a * b, 1) }, () => 1), shape,
    { requiresGrad: true });

  spotGrad("as_strided", (x) => x.asStrided([3, 4], [1, 3]));
  // 겹치는 걸음의 기울기 — 한 칸으로 여러 번 온다.
  spotGrad("as_strided(겹침)", (x) => x.asStrided([3, 3], [1, 1]),
    () => Tensor.from(Array.from({ length: 9 }, (_, i) => i + 1), [3, 3]));
  spotGrad("select_scatter", (x) => x.selectScatter(Tensor.zeros([4]), 0, 1));
  spotGrad("slice_scatter",
    (x) => x.sliceScatter(Tensor.zeros([3, 2]), 1, 0, 4, 2));
  spotGrad("diagonal_scatter",
    (x) => x.diagonalScatter(Tensor.zeros([3]), 1));
  spotGrad("diag_embed", (x) => x.diagEmbed(),
    () => Tensor.from(Array.from({ length: 48 }, (_, i) => i + 1), [3, 4, 4]));
  // (3,4) 를 3 으로 쪼개면 2·1·1 이라 가운데 조각이 (3,1) 이다.
  spotGrad("tensor_split", (x) => x.tensorSplit(3, 1)[1] ?? Tensor.zeros([3, 1]),
    () => Tensor.from([1.0, 3.0, 5.0], [3, 1]));
  spotGrad("masked_scatter", (x) => x.maskedScatter(mask(), feed()));
  spotGrad("put", (x) => x.put(flatIdx(), flatVal()));
  spotGrad("index_put", (x) => x.indexPut(
    [Tensor.from([0, 1, 0], [3]), Tensor.from([1, 2, 1], [3])],
    Tensor.from([10.0, 20.0, 30.0], [3])));
  // **깎인 줄의 기울기.** 배율 안에 x 가 있어서 `g·s` 로 적으면 여기서 갈린다.
  spotGrad("renorm", (x) => x.renorm(2, 0, 5.0));
  // `mv` 는 1차원이 낀 행렬곱이다 — 그 역방향이 코어에서 축 하나를 놓치고 있었다.
  spotGrad("mv", (x) => x.mv(Tensor.from([1, 0, 0, 2], [4])),
    () => Tensor.from([1.0, 2.0, 0.5], [3]));

  spotSrcGrad("select_scatter", (t, v) => t.selectScatter(v, 0, 1),
    () => leafOnes([4]));
  spotSrcGrad("diagonal_scatter", (t, v) => t.diagonalScatter(v, 1),
    () => leafOnes([3]));
  spotSrcGrad("as_strided_scatter",
    (t, v) => t.asStridedScatter(v, [2, 2], [1, 2], 3),
    () => leafOnes([2, 2]));
  spotSrcGrad("masked_scatter", (t, v) => t.maskedScatter(mask(), v),
    () => Tensor.from(Array.from({ length: 12 }, (_, i) => 100 + i), [12],
      { requiresGrad: true }));

  // ── diag_embed ─────────────────────────────────────────────────────
  for (const offset of [-1, 0, 1]) {
    out.set(`spot::diag_embed(1차, offset=${offset})`,
      () => trio().diagEmbed(offset));
  }
  out.set("spot::diag_embed(dim1=0, dim2=1)", () => grid().diagEmbed(0, 0, 1));

  // ── 쪼개기·번호 풀기 ────────────────────────────────────────────────
  out.set("spot::tensor_split(자리 목록)",
    () => Tensor.cat(line().tensorSplit([2, 5]), 0));
  out.set("spot::tensor_split(dim=1)",
    () => grid().tensorSplit(3, 1)[1] ?? Tensor.zeros([3, 1]));
  out.set("spot::unravel_index",
    () => Tensor.cat(Tensor.from([0, 5, 11], [3]).unravelIndex([3, 4]), 0));

  // ── 이어진 중복 ─────────────────────────────────────────────────────
  //
  // **정렬하지 않는다** — `[1,1,2,2,2,1,3]` 에서 1 이 두 번 남는다. 정렬된 입력으로만
  // 재면 `unique` 와 구분이 안 간다.
  const runs = (): Tensor => Tensor.from([1, 1, 2, 2, 2, 1, 3], [7],
    { dtype: "int64" });
  const rowRuns = (): Tensor => Tensor.from([1, 1, 1, 1, 1, 2, 3, 3], [4, 2],
    { dtype: "int64" });
  out.set("spot::unique_consecutive",
    async () => await runs().uniqueConsecutive() as Tensor);
  out.set("spot::unique_consecutive(inverse)",
    async () => (await runs().uniqueConsecutive(true) as Tensor[])[1]!);
  out.set("spot::unique_consecutive(counts)",
    async () => (await runs().uniqueConsecutive(false, true) as Tensor[])[1]!);
  out.set("spot::unique_consecutive(dim=0)",
    async () => await rowRuns().uniqueConsecutive(false, false, 0) as Tensor);
  out.set("spot::unique_consecutive(dim=0, counts)",
    async () => (await rowRuns().uniqueConsecutive(false, true, 0) as Tensor[])[1]!);

  // ── 줄이며 넣기 ─────────────────────────────────────────────────────
  //
  // `index_reduce` 에 `sum` 은 없다 — 그 자리는 `index_add` 다(실측).
  for (const reduce of ["prod", "mean", "amax", "amin"]) {
    for (const self of [true, false]) {
      out.set(`spot::index_reduce(${reduce}, include_self=${self ? "True" : "False"})`,
        () => base34().indexReduce(0, Tensor.from([0, 0, 2], [3]), grid(),
          reduce, self));
    }
  }

  // ── 조합·행렬 ───────────────────────────────────────────────────────
  const duo = (): Tensor => Tensor.from([4.0, 5.0], [2]);
  out.set("spot::cartesian_prod(둘)",
    () => Tensor.cartesianProd(trio(), duo()));
  // **하나만 주면 그냥 그것이다**(실측) — 1차원으로 남는다.
  out.set("spot::cartesian_prod(하나)", () => Tensor.cartesianProd(trio()));
  out.set("spot::cartesian_prod(셋)",
    () => Tensor.cartesianProd(trio(), duo(), duo()));
  for (const r of [1, 2, 3]) {
    out.set(`spot::combinations(r=${r})`, () => Tensor.combinations(trio(), r));
  }
  out.set("spot::combinations(중복 허용)",
    () => Tensor.combinations(trio(), 2, true));
  out.set("spot::chain_matmul", () => Tensor.chainMatmul(
    Tensor.from(Array.from({ length: 6 }, (_, i) => i), [2, 3]),
    Tensor.from(Array.from({ length: 12 }, (_, i) => i), [3, 4]),
    Tensor.from(Array.from({ length: 8 }, (_, i) => i), [4, 2])));

  // ── 제자리 ──────────────────────────────────────────────────────────
  //
  // **모양까지 따라가야 한다.** 값만 옮기면 정사각으로 물었을 때만 통과한다.
  out.set("spot::제자리::as_strided_", () => {
    const x = grid();
    const got = x.asStrided_([2, 3], [1, 2]);
    return `${verdict(got === x)} (${x.shape.join(", ")})`;
  });
  out.set("spot::제자리::masked_scatter_", async () => {
    const x = grid();
    const got = x.maskedScatter_(mask(), feed());
    return `${verdict(got === x)} ${(await x.toArray())[0]!.toFixed(1)}`;
  });
  out.set("spot::제자리::index_put_", async () => {
    const x = grid();
    const got = x.indexPut_(
      [Tensor.from([0, 1, 0], [3]), Tensor.from([1, 2, 1], [3])],
      Tensor.from([10.0, 20.0, 30.0], [3]));
    return `${verdict(got === x)} ${(await x.toArray())[1]!.toFixed(1)}`;
  });

  // ── 통계 (`stat::`) ─────────────────────────────────────────────────
  //
  // **여기 있는 것 거의 전부가 비동기다.** 히스토그램은 어느 칸에 들어가는지가 값이고,
  // `mode`·`nanmedian` 은 어느 값이 이기는지가 값이라 한 번 읽어야 한다 — 케이스 꼴이
  // 다른 자리와 갈리는 것은 그 때문이지 이름이 달라서가 아니다.
  const sample = (): Tensor => Tensor.from([0.5, 2.0, 2.0, 3.5, 1.0, 4.0, 2.0], [7]);
  const sampleW = (): Tensor => Tensor.from([1.0, 2.0, 1.0, 1.0, 3.0, 1.0, 1.0], [7]);
  out.set("stat::histc(bins=4)", async () => await sample().histc(4));
  out.set("stat::histc(min/max)", async () => await sample().histc(4, 0.0, 4.0));
  // **범위 밖은 버린다** — 양끝 칸으로 몰아넣지 않는다.
  out.set("stat::histc(범위 밖은 버림)",
    async () => await sample().histc(2, 1.0, 3.0));
  out.set("stat::histogram 의 hist",
    async () => (await sample().histogram(4)).hist);
  out.set("stat::histogram 의 edges",
    async () => (await sample().histogram(4)).bin_edges);
  out.set("stat::histogram(weight)",
    async () => (await sample().histogram(4, null, sampleW())).hist);
  out.set("stat::histogram(density)",
    async () => (await sample().histogram(4, null, null, true)).hist);
  out.set("stat::histogram(range)",
    async () => (await sample().histogram(4, [0.0, 4.0])).hist);
  // **칸 너비가 다르다** — `density` 가 칸마다 다른 값으로 나누는지 여기서만 보인다.
  out.set("stat::histogram(경계를 직접)",
    async () => (await sample().histogram(Tensor.from([0.0, 1.0, 2.0, 4.0], [4]))).hist);

  const pts = (): Tensor => Tensor.from(
    [0.5, 1.0, 1.5, 1.5, 2.5, 0.5, 0.2, 2.5], [4, 2]);
  out.set("stat::histogramdd 의 hist",
    async () => (await pts().histogramdd([2, 2])).hist);
  out.set("stat::histogramdd 의 edges",
    async () => Tensor.cat((await pts().histogramdd([2, 2])).bin_edges, 0));

  // **비긴 자리가 있다** — 없으면 `mode` 의 규칙(작은 값이 이기고 자리는 마지막)이
  // 안 드러난다.
  const tie = (): Tensor => Tensor.from([1.0, 2.0, 2.0, 3.0, 4.0, 4.0, 5.0, 5.0], [2, 4]);
  for (const dim of [0, 1]) {
    out.set(`stat::mode(dim=${dim}) 값`, async () => (await tie().mode(dim)).values);
    out.set(`stat::mode(dim=${dim}) 자리`, async () => (await tie().mode(dim)).indices);
  }
  out.set("stat::mode(keepdim) 모양", async () => {
    const got = await tie().mode(1, true);
    return `(${got.values.shape.join(", ")})`;
  });

  const holes = (): Tensor => Tensor.from(
    [1.0, Number.NaN, 3.0, 5.0, 2.0, 4.0, Number.NaN, Number.NaN], [2, 4]);
  out.set("stat::nanmedian(전체)",
    async () => await holes().nanmedian() as Tensor);
  out.set("stat::nanmedian(dim=1) 값", async () =>
    (await holes().nanmedian(1) as { values: Tensor }).values);
  out.set("stat::nanmedian(dim=1) 자리", async () =>
    (await holes().nanmedian(1) as { indices: Tensor }).indices);
  // **짝수 개면 아래를 고른다** — 평균을 내면 여기서 갈린다.
  out.set("stat::nanmedian(짝수 개)",
    async () => await Tensor.from([1.0, 2.0, 3.0, 4.0], [4]).nanmedian() as Tensor);
  // `median` 은 NaN 하나에도 NaN 이다 — 나란히 둬야 `nanmedian` 이 무엇인지 보인다.
  //
  // **값이 아니라 판정을 굳힌다.** 대조가 `allclose` 라 NaN 은 자기 자신과도 다르다.
  out.set("stat::median(NaN 이 섞이면 NaN 이다)", async () => {
    const got = await holes().median(1).values.toArray();
    return Array.from(got).map((v) => verdict(Number.isNaN(v))).join(" ");
  });

  // `x²` 이다 — `edge_order=2` 가 정확해지는 자리.
  const curve = (): Tensor => Tensor.from([1.0, 4.0, 9.0, 16.0, 25.0], [5]);
  const mat33 = (): Tensor => Tensor.from(
    [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0], [3, 3]);
  out.set("stat::gradient(기본)",
    async () => Tensor.cat(await curve().gradient(), 0));
  out.set("stat::gradient(spacing=2)",
    async () => Tensor.cat(await curve().gradient(2.0), 0));
  out.set("stat::gradient(edge_order=2)",
    async () => Tensor.cat(await curve().gradient(1, undefined, 2), 0));
  for (const axis of [0, 1]) {
    out.set(`stat::gradient(2차)[${axis}]`,
      async () => (await mat33().gradient())[axis]!);
  }
  out.set("stat::gradient(dim=1)",
    async () => (await mat33().gradient(1, 1))[0]!);

  // **모자라면 채우고 넘치면 자른다.** 딱 맞는 크기로만 재면 두 갈래가 안 드러난다.
  const sparse = (): Tensor => Tensor.from([0.0, 3.0, 0.0, 5.0, 0.0], [5]);
  for (const size of [1, 2, 5]) {
    out.set(`stat::nonzero_static(size=${size})`,
      async () => await sparse().nonzeroStatic(size));
  }
  out.set("stat::nonzero_static(fill=-9)",
    async () => await sparse().nonzeroStatic(5, -9));

  // **`wrap` 은 세로로 긴 행렬에서만 뜻이 있다.** 정사각으로만 묻던 동안 이쪽 판이
  // 없는 규칙(되돌아 올 때 줄을 하나 건너뛴다)을 지어냈는데도 초록이었다. 결속을
  // 거쳐서도 묻지만, **틀렸던 자리는 이쪽에서 직접** 묻는 편이 낫다.
  for (const wrap of [false, true]) {
    out.set(`inplace::짝없이::fill_diagonal_(세로, wrap=${verdict(wrap)})`, () => {
      const x = Tensor.from(Array.from({ length: 18 }, (_, i) => i), [6, 3]);
      x.fillDiagonal_(9, wrap);
      return x;
    });
  }

  // ── 짝에서 만든 제자리 판 (`inplace::짝에서::`) ──────────────────────
  //
  // torch 는 거의 모든 연산에 밑줄 짝을 준다. 여기 있던 것은 `i0_` 하나였고, 나머지
  // 서른여덟은 **계산이 이미 있는데 이름만 없던** 자리였다 — 열은 torch 의 둘째
  // 철자(`divide_`=`div_`)이고, 열하나는 커널 표에만 있어서 `binary("gcd", …)` 로만
  // 닿았다. 갭 표가 마흔을 통째로 "별칭" 이라 적었는데 별칭인 것은 열뿐이었다.
  //
  // **둘은 짝에서 만들면 안 된다.** `bernoulli_(p)` 는 자기 값을 확률로 읽는
  // `bernoulli()` 와 달리 자기 값을 무시하고 `p` 로 채우고, `float_power_` 는 결과가
  // 배정도라 torch 도 거절한다.
  const ints4 = (): Tensor => Tensor.from([6, -4, 3, 9], [4], { dtype: "int64" });
  const flags4 = (): Tensor => Tensor.from([1, 0, 1, 0], [4], { dtype: "bool" });
  const other4 = (): Tensor => Tensor.from([1, 1, 0, 0], [4], { dtype: "bool" });
  const plain4 = (): Tensor => Tensor.from([1, 4, 9, 2], [4]);
  const twos4 = (): Tensor => Tensor.from([2, 2, 2, 2], [4]);
  const pos4 = (): Tensor => Tensor.from([2, 3, 4, 5], [4]);
  const ramp23 = (): Tensor => Tensor.from([0, 1, 2, 3, 4, 5], [2, 3]);
  const pairs: [string, () => Tensor, (t: Tensor) => Tensor][] = [
    ["bitwise_and_", ints4, (x) => x.bitwiseAnd_(3)],
    ["bitwise_or_", ints4, (x) => x.bitwiseOr_(3)],
    ["bitwise_xor_", ints4, (x) => x.bitwiseXor_(3)],
    ["bitwise_not_", ints4, (x) => x.bitwiseNot_()],
    ["bitwise_left_shift_", ints4, (x) => x.bitwiseLeftShift_(1)],
    ["bitwise_right_shift_", ints4, (x) => x.bitwiseRightShift_(1)],
    ["logical_and_", flags4, (x) => x.logicalAnd_(other4())],
    ["logical_or_", flags4, (x) => x.logicalOr_(other4())],
    ["logical_xor_", flags4, (x) => x.logicalXor_(other4())],
    ["logical_not_", flags4, (x) => x.logicalNot_()],
    ["clamp_max_", plain4, (x) => x.clampMax_(4)],
    ["clamp_min_", plain4, (x) => x.clampMin_(3)],
    ["digamma_", pos4, (x) => x.digamma_()],
    ["divide_", plain4, (x) => x.divide_(2)],
    ["erfinv_", () => Tensor.from([0.0, 0.3, -0.2, 0.4], [4]), (x) => x.erfinv_()],
    ["floor_divide_", plain4, (x) => x.floorDivide_(2)],
    ["fmod_", plain4, (x) => x.fmod_(2)],
    ["gcd_", ints4, (x) => x.gcd_(Tensor.from([2, 2, 3, 3], [4], { dtype: "int64" }))],
    ["lcm_", ints4, (x) => x.lcm_(Tensor.from([2, 2, 3, 3], [4], { dtype: "int64" }))],
    ["greater_", plain4, (x) => x.greater_(3)],
    ["greater_equal_", plain4, (x) => x.greaterEqual_(4)],
    ["less_", plain4, (x) => x.less_(3)],
    ["less_equal_", plain4, (x) => x.lessEqual_(4)],
    ["not_equal_", plain4, (x) => x.notEqual_(4)],
    ["i0_", plain4, (x) => x.i0_()],
    ["lgamma_", pos4, (x) => x.lgamma_()],
    ["lerp_", plain4, (x) => x.lerp_(twos4(), 0.5)],
    ["mvlgamma_", pos4, (x) => x.mvlgamma_(1)],
    ["multiply_", plain4, (x) => x.multiply_(3)],
    ["nan_to_num_",
      () => Tensor.from([1.0, Number.NaN, Infinity, -Infinity], [4]),
      (x) => x.nanToNum_()],
    ["nextafter_", plain4, (x) => x.nextafter_(twos4())],
    ["put_", plain4,
      (x) => x.put_(Tensor.from([0, 2], [2], { dtype: "int64" }),
        Tensor.from([9, 9], [2]))],
    ["remainder_", plain4, (x) => x.remainder_(2)],
    ["renorm_", ramp23, (x) => x.renorm_(2, 0, 1.0)],
    ["subtract_", plain4, (x) => x.subtract_(1)],
    ["true_divide_", plain4, (x) => x.trueDivide_(2)],
    // **모양이 바뀌는 제자리 연산.** 정사각으로만 물으면 안 바뀐 채 통과한다.
    ["t_", ramp23, (x) => x.t_()],
  ];
  for (const [name, src, run] of pairs) {
    out.set(`inplace::짝에서::${name}`, () => {
      const x = src();
      run(x);
      return x;
    });
  }
  // **자기 값을 확률로 안 읽는다** — 입력이 [1,4,9,2] 인데 `p=0` 이면 전부 0 이다.
  for (const p of [0.0, 1.0]) {
    out.set(`inplace::짝에서::bernoulli_(p=${p.toFixed(1)})`, () => {
      const x = plain4();
      x.bernoulli_(p);
      return x;
    });
  }
  out.set("inplace::짝에서::float_power_ 는 거절", () => {
    try {
      plain4().floatPower_(2);
    } catch (err) {
      const said = err instanceof Error ? err.message : String(err);
      return said.includes("Double") ? "Double" : `다른 문구 <${said}>`;
    }
    return "안 던졌다";
  });

  // ── 분포에서 뽑아 채우는 일곱 (`inplace::분포::`) ─────────────────────
  //
  // 값은 못 굳힌다. 굳힐 수 있는 것은 **모양·형이 안 바뀐다**는 것과 **거절**이다.
  //
  // 이 묶음이 갭 표에서 "별칭·파이썬" 으로 덮여 있었고, 그 아래에 **여기 코드가 안
  // 하던 것 열다섯**이 들어 있었다. 다섯을 넣을 때 `parity.ts` 에는 분포의 중심만
  // 재는 검사를 넣고 거절은 하나도 안 쟀다 — `normal` 의 `std=0` 만 재면 절반이라고
  // 적어 놓고 같은 실수를 했다.
  const drawn = (
    name: string, fill: (t: Tensor) => Tensor,
  ): void => {
    out.set(`inplace::분포::${name} 는 모양과 형을 지킨다`, () => {
      const x = Tensor.zeros([2, 3]);
      fill(x);
      return `(${x.shape.join(", ")}) ${dtypeName(x.dtype)}`;
    });
    // **연속 다섯은 정수를 거절하고 `geometric_`·`random_` 은 안 한다.** 이름만
    // 보고 "난수는 실수만" 으로 묶으면 그 둘에서 틀린다.
    out.set(`inplace::분포::${name}(int64)`, () => {
      const x = Tensor.zeros([6]).to("int64");
      try {
        fill(x);
      } catch (err) {
        return `거절(${err instanceof Error ? err.constructor.name : "?"})`;
      }
      return `돈다 ${dtypeName(x.dtype)}`;
    });
  };
  drawn("normal_", (t) => t.normal_(0.0, 2.0));
  drawn("uniform_", (t) => t.uniform_(-1.0, 3.0));
  drawn("exponential_", (t) => t.exponential_(2.0));
  drawn("cauchy_", (t) => t.cauchy_(1.0, 0.5));
  drawn("log_normal_", (t) => t.logNormal_(0.0, 1.0));
  drawn("geometric_", (t) => t.geometric_(0.3));
  drawn("random_", (t) => t.random_(0, 5));

  // **인자의 정의역은 분포마다 다르다** — `p` 는 열린 구간, `lambda` 는 양수,
  // `from < to`, `std >= 0`. 문구의 조각을 묻는다: 값이 아니라 글자라 구현끼리
  // 대조해도 안 걸리는 자리다.
  const refusesArg = (label: string, call: () => Tensor, fragment: string): void => {
    out.set(`inplace::분포::거절::${label}`, () => {
      try {
        call();
      } catch (err) {
        const said = err instanceof Error ? err.message : String(err);
        return said.includes(fragment) ? "멈췄다" : `다른 문구 <${said}>`;
      }
      return "안 던졌다";
    });
  };
  const zeros3 = (): Tensor => Tensor.zeros([3]);
  refusesArg("geometric_(0)", () => zeros3().geometric_(0), "p to be in (0, 1)");
  refusesArg("geometric_(1)", () => zeros3().geometric_(1), "p to be in (0, 1)");
  refusesArg("exponential_(0)", () => zeros3().exponential_(0), "lambda > 0.0");
  refusesArg("uniform_(3,1)", () => zeros3().uniform_(3, 1), "[from, to) range");
  refusesArg("normal_(0,-1)", () => zeros3().normal_(0, -1), "std >= 0.0");
  refusesArg("random_(5,2)",
    () => Tensor.zeros([3]).to("int64").random_(5, 2),
    "'from' to be less than 'to'");

  // **범위가 형이 정확히 셀 수 있는 데까지다.** float32 는 2²⁴ 를 넘으면 이웃한
  // 정수를 구별 못 해 값이 뭉친다. int64 쪽은 여기 없다 — torch 는 2⁶² 인데 이쪽의
  // int64 는 f32 칸에 담기므로 그 위를 셀 수가 없고, 못 세는 것을 흉내 내지 않는다.
  out.set("inplace::분포::random_(float32) 의 상한", async () => {
    const x = Tensor.zeros([3000]);
    x.random_();
    const got = Array.from(await x.toArray()).reduce((a, b) => Math.max(a, Math.abs(b)), 0);
    const cap = 1 << 24;
    return verdict(cap / 2 <= got && got <= cap);
  });

  out.set("stat::trapz(y)", () => curve().trapezoid());
  out.set("stat::trapz(dx=2)", () => curve().trapezoid(undefined, 2.0));
  out.set("stat::trapz(y, x)",
    () => curve().trapezoid(Tensor.from([0.0, 1.0, 3.0, 6.0, 10.0], [5])));

  // ── 난수 넷 — 결정적인 끝값만 ───────────────────────────────────────
  //
  // 값은 못 굳힌다(torch 의 난수 줄기와 우리 것이 다르고 같게 만들 길이 없다).
  // **그래서 끝값을 묻는다** — 안 물으면 `bernoulli` 가 확률을 아예 안 보고 있어도
  // 통과한다. "난수라 못 묻는다" 와 "안 묻는다" 는 다르다.
  out.set("stat::bernoulli(p=0)", () => Tensor.zeros([4]).bernoulli());
  out.set("stat::bernoulli(p=1)", () => Tensor.ones([4]).bernoulli());
  out.set("stat::poisson(0)", async () => await Tensor.zeros([4]).poisson());
  const ten = (): Tensor => Tensor.from([10.0, 10.0], [2]);
  out.set("stat::binomial(p=0)",
    async () => await ten().binomial(Tensor.zeros([2])));
  out.set("stat::binomial(p=1)",
    async () => await ten().binomial(Tensor.ones([2])));
  out.set("stat::normal(std=0)",
    () => Tensor.normal(Tensor.from([1.0, 100.0], [2]), Tensor.zeros([2])));
  // 값은 못 묻지만 **모양은 묻는다** — 그것마저 안 물으면 이름만 있는 것과 같다.
  out.set("stat::normal(size) 모양",
    () => `(${Tensor.normal(0.0, 1.0, [2, 3]).shape.join(", ")})`);
  out.set("stat::bernoulli 모양",
    () => `(${Tensor.zeros([2, 3]).bernoulli().shape.join(", ")})`);

  // ── addmm 계열 (`blend::`) ──────────────────────────────────────────
  //
  // **`beta=0` 은 값만 안 보고 그래프에는 남는다.** 여기서는 값만 묻는다 — 기울기
  // 쪽은 파이썬 케이스가 갖고 있다.
  const m1 = (): Tensor => Tensor.from(
    Array.from({ length: 6 }, (_, i) => i), [2, 3]);
  const m2 = (): Tensor => Tensor.from(
    Array.from({ length: 12 }, (_, i) => i), [3, 4]);
  const base24 = (): Tensor => Tensor.zeros([2, 4]).add(Tensor.full([], 10));
  const b1 = (): Tensor => Tensor.from(
    Array.from({ length: 12 }, (_, i) => i), [2, 2, 3]);
  const b2 = (): Tensor => Tensor.from(
    Array.from({ length: 24 }, (_, i) => i), [2, 3, 4]);
  const deep = (): Tensor => Tensor.zeros([2, 2, 4]).add(Tensor.full([], 10));
  for (const [beta, alpha] of [[1, 1], [2, 3], [0, 1], [1, 0], [-1, 0.5]] as const) {
    out.set(`blend::addmm(beta=${beta}, alpha=${alpha})`,
      () => base24().addmm(m1(), m2(), beta, alpha));
  }
  // **NaN 을 넣어야** `input * 0` 으로 적은 것이 드러난다.
  out.set("blend::addmm(beta=0, input=NaN)", () =>
    Tensor.zeros([2, 4]).add(Tensor.full([], Number.NaN)).addmm(m1(), m2(), 0));
  // **배치가 둘 이상이어야** 합치는 쪽과 지키는 쪽이 갈린다.
  for (const [beta, alpha] of [[1, 1], [2, 3], [0, 1]] as const) {
    out.set(`blend::addbmm(beta=${beta}, alpha=${alpha})`,
      () => base24().addbmm(b1(), b2(), beta, alpha));
    out.set(`blend::baddbmm(beta=${beta}, alpha=${alpha})`,
      () => deep().baddbmm(b1(), b2(), beta, alpha));
  }
  const t0 = (): Tensor => Tensor.from([1, 2, 3, 4], [2, 2]);
  const t1 = (): Tensor => Tensor.from([2, 3, 4, 5], [2, 2]);
  const t2 = (): Tensor => Tensor.from([5, 2, 2, 4], [2, 2]);
  for (const value of [1, 2, -1, 0]) {
    out.set(`blend::addcmul(value=${value})`, () => t0().addcmul(t1(), t2(), value));
    out.set(`blend::addcdiv(value=${value})`, () => t0().addcdiv(t1(), t2(), value));
  }

  // ── 남은 서른넷 ─────────────────────────────────────────────────────
  //
  // 위는 값만 물었고, 이 아래가 **인자가 실제로 닿는지**를 묻는 자리다.
  //
  // - `addmv`·`addr` 은 `beta`·`alpha` 를 한 번도 안 물었다.
  // - `input` 이 결과보다 **작아야** 퍼지는 것이 보인다 — torch 는 `(4,)` 도
  //   스칼라도 받는다.
  // - **기울기가 `beta=0` 을 가른다.** 값만 보면 그래프에서 뺀 것과 구별이 안 되는데,
  //   빼 두면 `input.grad` 가 0 이 아니라 **없다**. torch 는 0 을 준다.
  // - 제자리 판은 **자기를 돌려줘야** 한다. 새 텐서를 주면 `x.addmm_(a, b).add_(1)`
  //   이 원본이 아닌 사본을 고치기 시작한다.
  const vecv = (): Tensor => Tensor.from([1, 0, 2], [3]);
  const v1 = (): Tensor => Tensor.from([1, 2], [2]);
  const v2 = (): Tensor => Tensor.from([3, 4, 5], [3]);
  const ones = (shape: number[]): Tensor =>
    Tensor.zeros(shape).add(Tensor.full([], 1));

  for (const [beta, alpha] of [[1, 1], [2, 3], [0, 1]] as const) {
    out.set(`blend::addmv(beta=${beta}, alpha=${alpha})`,
      () => ones([2]).addmv(m1(), vecv(), beta, alpha));
    out.set(`blend::addr(beta=${beta}, alpha=${alpha})`,
      () => ones([2, 3]).addr(v1(), v2(), beta, alpha));
  }

  // 퍼지는 `input`. 결과는 (2,4) 인데 받는 쪽이 (4,) 이거나 스칼라다.
  out.set("blend::addmm(input 이 (4,))", () => ones([4]).addmm(m1(), m2()));
  out.set("blend::addmm(input 이 스칼라)", () => ones([]).addmm(m1(), m2()));
  out.set("blend::baddbmm(input 이 (2,4))", () => base24().baddbmm(b1(), b2()));
  out.set("blend::addcmul(브로드캐스트)",
    () => t0().addcmul(Tensor.from([1, 10], [2]), t2()));

  // 기울기. 무게를 **고르지 않게** 줘야 자리마다 다른 몫이 안 상쇄된다.
  const WEIGHT = [1.0, 2.0, 0.5, 3.0, 2.0, 0.5, 1.5, 1.0];
  const blendGrad = (
    name: string, src: () => Tensor, body: (t: Tensor) => Tensor,
    weight: () => Tensor = () => Tensor.from(WEIGHT, [2, 4]),
  ): void => {
    out.set(`blend::grad::${name}`, () => {
      const leaf = src();
      leaf.requiresGrad = true;
      body(leaf).mul(weight()).sum().backward();
      return gradOf(leaf, name);
    });
  };

  blendGrad("addmm(beta=2, alpha=3)", base24, (x) => x.addmm(m1(), m2(), 2, 3));
  // **여기서 그래프에서 뺀 구현이 멈춘다** — `requires_grad 가 아니다` 로.
  blendGrad("addmm(beta=0)", base24, (x) => x.addmm(m1(), m2(), 0));
  blendGrad("addmm(퍼지는 input)", () => ones([4]), (x) => x.addmm(m1(), m2()));
  blendGrad("addmm(mat1)", m1, (x) => base24().addmm(x, m2(), 1, 3));
  blendGrad("addbmm", base24, (x) => x.addbmm(b1(), b2()));
  blendGrad("addbmm(batch1)", b1, (x) => base24().addbmm(x, b2(), 1, 2));
  blendGrad("baddbmm", deep, (x) => x.baddbmm(b1(), b2()),
    () => Tensor.from(Array.from({ length: 16 }, (_, i) => i + 1), [2, 2, 4]));
  blendGrad("addmv(mat)", m1, (x) => ones([2]).addmv(x, vecv(), 1, 2),
    () => Tensor.from([1, 2], [2]));
  blendGrad("addr(vec1)", v1, (x) => ones([2, 3]).addr(x, v2(), 1, 2),
    () => Tensor.from([1, 2, 3, 4, 5, 6], [2, 3]));
  blendGrad("addcdiv", t0, (x) => x.addcdiv(t1(), t2(), 2),
    () => Tensor.from([1.0, 2.0, 0.5, 3.0], [2, 2]));

  // 제자리. **값과 "자기를 돌려주는가" 를 따로 묻는다** — 값만 물으면 사본을
  // 돌려주는 구현이 통과한다.
  const inplace: [string, () => Tensor, (t: Tensor) => Tensor][] = [
    ["addmm_", base24, (t) => t.addmm_(m1(), m2())],
    ["addbmm_", base24, (t) => t.addbmm_(b1(), b2())],
    ["baddbmm_", deep, (t) => t.baddbmm_(b1(), b2())],
    ["addmv_", () => ones([2]), (t) => t.addmv_(m1(), vecv())],
    ["addr_", () => ones([2, 3]), (t) => t.addr_(v1(), v2())],
    ["addcmul_", t0, (t) => t.addcmul_(t1(), t2())],
    ["addcdiv_", t0, (t) => t.addcdiv_(t1(), t2())],
  ];
  for (const [name, src, run] of inplace) {
    out.set(`blend::제자리::${name}`, () => {
      const x = src();
      run(x);
      return x;
    });
    // **`verdict` 를 지나야 한다.** 골든은 파이썬의 `str(bool)` 이라 `True` 이고
    // JS 의 `String(true)` 는 `true` 다 — 판정이 다른 것이 아니라 적는 법이 다르다.
    out.set(`blend::제자리::${name}(같은 텐서)`, () => {
      const x = src();
      return verdict(run(x) === x);
    });
  }

  // ── 복소수의 이웃 (`make::`) ────────────────────────────────────────
  //
  // **복소수가 없어도 답이 있는 이름들이다.** 실수를 주면 `real`·`conj` 는 자기
  // 자신이고 `angle` 은 0(음수면 π)이다. 항등인 것과 **없는 것은 다르다** — torch
  // 코드는 켤레를 넘기기 전에 `resolve_conj()` 를 넣는 관용구를 쓴다.
  //
  // **형까지 지켜야 한다.** 항등이라고 아무거나 돌려주면 int64 가 float32 로
  // 새는데, 값이 같아서 값만 묻는 케이스는 통과한다.
  const kinds: [string, () => Tensor][] = [
    ["float32", () => Tensor.from([-1.5, 0.0, 2.0, 3.0, -4.0, 0.5], [2, 3])],
    ["int64", () => Tensor.from([1, -2, 3], [3], { dtype: "int64" })],
    ["bool", () => Tensor.from([1, 0, 1], [3], { dtype: "bool" })],
  ];
  const idents: [string, (t: Tensor) => Tensor][] = [
    ["real", (t) => t.real()],
    ["conj", (t) => t.conj()],
    ["conj_physical", (t) => t.conjPhysical()],
    ["resolve_conj", (t) => t.resolveConj()],
    ["resolve_neg", (t) => t.resolveNeg()],
  ];
  for (const [name, fn] of idents) {
    for (const [tag, src] of kinds) {
      out.set(`make::${name}(${tag})`, () => fn(src()));
      out.set(`make::${name}(${tag}) 형`, () => dtypeName(fn(src()).dtype));
    }
  }
  // `angle` 만 형이 **언제나 float32** 다 — 각도는 실수라서.
  for (const [tag, src] of kinds) {
    out.set(`make::angle(${tag})`, () => src().angle());
    out.set(`make::angle(${tag}) 형`, () => dtypeName(src().angle().dtype));
  }
  // 판정 셋 — 전부 거짓. **게으른 비트가 없다는 것이 물음이 뜻을 잃는 이유는 아니다.**
  const predicates: [string, (t: Tensor) => boolean][] = [
    ["is_complex", (t) => t.isComplex()],
    ["is_conj", (t) => t.isConj()],
    ["is_neg", (t) => t.isNeg()],
  ];
  for (const [name, fn] of predicates) {
    out.set(`make::${name}`,
      () => kinds.map(([, src]) => verdict(fn(src()))).join(" "));
  }

  // 바이트를 그대로 읽는다. 파이썬 쪽은 `bytes` 를 주고 이쪽은 `ArrayBuffer` 다 —
  // 같은 바이트 열두 개이고, 그것이 이 이름이 묻는 전부다.
  const rawBytes = (): ArrayBuffer => {
    const view = new Float32Array([1.0, 2.0, 3.0]);
    return view.buffer;
  };
  out.set("make::frombuffer", () => Tensor.frombuffer(rawBytes()));
  out.set("make::frombuffer(count=2)",
    () => Tensor.frombuffer(rawBytes(), "float32", 2));
  // **`offset` 은 바이트다** — 원소 수로 읽으면 여기서 갈린다.
  out.set("make::frombuffer(offset=4)",
    () => Tensor.frombuffer(rawBytes(), "float32", -1, 4));

  // **끝을 포함한다** — `arange` 와 한 칸 다르다.
  out.set("make::range(0, 4)", () => Tensor.range(0, 4));
  out.set("make::range(1, 7, 2)", () => Tensor.range(1, 7, 2));
  out.set("make::range(0, 1, 0.25)", () => Tensor.range(0, 1, 0.25));
  // 그 한 칸은 **개수를 물어야** 드러난다 — 합이나 평균으로는 안 보인다.
  out.set("make::range 와 arange 의 개수",
    () => `${Tensor.range(0, 4).size} ${Tensor.arange(0, 4).size}`);

  // **걸음이 0 이면 멈춘다.** 안 막으면 `(끝-시작)/0` 이 Infinity 가 되어 배열을
  // 잡는 자리에서 터지고, 그 문구는 메모리가 모자란 것과 구별이 안 간다.
  //
  // 문구의 **조각**을 묻는다 — 값이 아니라 글자라 서로 대조해도 안 걸리는 자리다.
  const refusesZeroStep = (make: () => Tensor): string => {
    try {
      make();
    } catch (err) {
      const said = err instanceof Error ? err.message : String(err);
      return said.includes("nonzero") ? "nonzero" : `다른 문구 <${said}>`;
    }
    return "안 던졌다";
  };
  out.set("make::arange(step=0)=거절",
    () => refusesZeroStep(() => Tensor.arange(0, 5, 0)));
  out.set("make::range(step=0)=거절",
    () => refusesZeroStep(() => Tensor.range(0, 5, 0)));

  // ── 최상위 선형대수 (`toplin::`) ────────────────────────────────────
  //
  // **인자 순서가 `linalg` 쪽과 뒤집혀 있다.** 그 자리를 TS 에서도 물어 둔다.
  const spd = (): Tensor => Tensor.from([4, 2, 1, 2, 5, 3, 1, 3, 6], [3, 3]);
  const tri = (): Tensor => Tensor.from([2, 0, 0, 1, 3, 0, 4, 2, 5], [3, 3]);
  const rhs32 = (): Tensor => Tensor.from([1, 2, 3, 1, 2, 4], [3, 2]);
  for (const upper of [false, true]) {
    out.set(`toplin::cholesky_solve(upper=${upper ? "True" : "False"})`, async () => {
      const low = await spd().cholesky();
      return rhs32().choleskySolve(upper ? low.transpose() : low, upper);
    });
    out.set(`toplin::cholesky_inverse(upper=${upper ? "True" : "False"})`,
      async () => {
        const low = await spd().cholesky();
        return (upper ? low.transpose() : low).choleskyInverse(upper);
      });
  }
  for (const upper of [false, true]) {
    for (const trans of [false, true]) {
      for (const unit of [false, true]) {
        out.set(
          `toplin::triangular_solve(u=${upper ? "True" : "False"},` +
          `t=${trans ? "True" : "False"},unit=${unit ? "True" : "False"})`,
          async () => (await rhs32().triangularSolve(tri(), upper, trans, unit))
            .solution);
      }
    }
  }
  // **`orgqr` 은 자른 Q 이고 `ormqr` 은 자르지 않은 Q 를 쓴다.** 세로로 긴 것으로
  // 물어야 그 갈림이 보인다 — 정사각으로 재면 둘이 같다.
  const tall = (): Tensor => Tensor.from([1, 2, 3, 4, 5, 6], [3, 2]);
  const side = (): Tensor => Tensor.from([1, 0, 0, 1, 1, 1], [3, 2]);
  out.set("toplin::orgqr", async () => {
    const { a, tau } = await tall().geqrf();
    return a.orgqr(tau);
  });
  for (const left of [true, false]) {
    for (const trans of [true, false]) {
      out.set(
        `toplin::ormqr(left=${left ? "True" : "False"}, ` +
        `transpose=${trans ? "True" : "False"})`,
        async () => {
          const { a, tau } = await tall().geqrf();
          return a.ormqr(tau, left ? side() : side().transpose(), left, trans);
        });
    }
  }
}

/**
 * 수치 계열. **조합되는 것과 급수로 세는 것이 섞여 있다.**
 *
 * `lgamma`·`digamma`·`erfinv` 는 닫힌 식이 없어서 근사식을 WGSL 에 적었다. 여기서
 * 묻는 것은 **f32 로 센 것이 torch 의 배정도와 허용 오차 안인가**이고, 그것이
 * 이 케이스들의 값어치 전부다.
 */
function addNumeric(out: Map<string, Case>, inp: Inputs): void {
  const mat = (): Tensor => inp.get("x2");
  const other = (): Tensor =>
    mat().mul(Tensor.full([], 0.5)).add(Tensor.full([], 1));
  const pos = (grad = false): Tensor => {
    const t = inp.get("xp");
    return grad ? asLeaf(t) : t;
  };
  // 감마 계열은 양수에서만 본다 — 음의 정수에서 발산하는 것이 정의다.
  const gam = (grad = false): Tensor => {
    const t = Tensor.from([0.1, 0.5, 1, 1.5, 2, 3, 5, 8.5], [8]);
    return grad ? asLeaf(t) : t;
  };
  const unit = (grad = false): Tensor => {
    const t = Tensor.from([-0.9, -0.5, -0.1, 0, 0.1, 0.5, 0.9], [7]);
    return grad ? asLeaf(t) : t;
  };

  const cov = (t: Tensor): Tensor => {
    const n = t.shape[1] ?? 1;
    const centered = t.sub(t.mean(1, true));
    return centered.mm(centered.transpose()).mul(Tensor.full([], 1 / (n - 1)));
  };

  /** 사다리꼴 조각. 이웃한 두 점의 평균에 간격을 곱한 것. */
  const pieces = (y: Tensor, dx: number): Tensor => {
    const n = y.shape[0] ?? 1;
    return y.narrow(0, 0, n - 1).add(y.narrow(0, 1, n - 1))
      .mul(Tensor.full([], dx / 2));
  };

  const table: [string, () => Tensor][] = [
    ["cdist", () => {
      const [n = 1, k = 1] = mat().shape;
      const m = other().shape[0] ?? 1;
      const diff = mat().reshape([n, 1, k]).sub(other().reshape([1, m, k]));
      return diff.mul(diff).sumDim(2).sqrt();
    }],
    ["cov", () => cov(mat())],
    ["corrcoef", () => {
      const c = cov(mat());
      const n = c.shape[0] ?? 1;
      const d = c.diagonal();
      return c.div(d.reshape([n, 1]).mul(d.reshape([1, n])).sqrt());
    }],
    ["tensordot", () => mat().mm(other().transpose())],
    ["trapezoid", () => pieces(pos(), 1).sum()],
    ["trapezoid(dx)", () => pieces(pos(), 0.5).sum()],
    ["cumulative_trapezoid", () => pieces(pos(), 1).cumsum(0)],
    ["lgamma", () => gam().lgamma()],
    ["digamma", () => gam().digamma()],
    ["erfinv", () => unit().erfinv()],
  ];
  for (const [name, fn] of table) out.set(`num::${name}`, fn);

  const grads: [string, (x: Tensor) => Tensor, (g?: boolean) => Tensor][] = [
    ["lgamma", (x) => x.lgamma(), gam],
    ["digamma", (x) => x.digamma(), gam],
    ["erfinv", (x) => x.erfinv(), unit],
  ];
  for (const [name, fn, make] of grads) {
    out.set(`num::grad::${name}`, () => {
      const x = make(true);
      seeded(fn(x)).backward();
      return gradOf(x, name);
    });
  }
}

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
    const p = Tensor.from([1.0], [1], { requiresGrad: true });
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
    const p = Tensor.from([1.0], [1], { requiresGrad: true });
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
  const mask = () => Tensor.from([1, 0, 1, 0, 1, 0, 1, 0], [2, 4], { dtype: "bool" });

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
    ["nn.MaxPool1d", () => new nn.MaxPool1d(2).call(nd("nd_seq"))],
    ["nn.MaxPool3d", () => new nn.MaxPool3d(2).call(nd("nd_vol"))],
    ["nn.BatchNorm3d", () => new nn.BatchNorm3d(2).call(nd("nd_vol"))],
    ["nn.Upsample", () => nd("nd_img").upsample(2)],
    // **`mode` 를 받아만 놓고 안 쓰던 자리다** — 겹선형을 달라고 해도 최근접이
    // 나왔다. 예외가 아니라 조용히 다른 값이다.
    ["nn.Upsample(겹선형)",
      () => new nn.Upsample(null, 2, "bilinear").call(nd("nd_img"))],
    // **첫 자리는 `size` 다.** 배율을 첫 자리에 두면 같은 줄이 늘리는 것과 줄이는
    // 것으로 갈리는데 모양이 양쪽 다 그럴듯하다. 12 는 3 배라 기본값 2 와 갈린다.
    ["nn.Upsample(첫 자리는 size)", () => new nn.Upsample(12).call(nd("nd_img"))],
    ["nn.AvgPool2d", () => new nn.AvgPool2d(2).call(nd("nd_img"))],
    ["nn.AvgPool2d(보폭)", () => new nn.AvgPool2d(2, 1).call(nd("nd_img"))],
    ["nn.LPPool1d", () => new nn.LPPool1d(2, 2).call(nd("nd_seq"))],
    ["nn.Unflatten",
      () => new nn.Unflatten(1, [1, 3]).call(nd("nd_img").reshape([2, 3, 16]))],
  ];
  for (const [name, fn] of values) out.set(`ndim::${name}`, fn);

  const grads: [string, string, (x: Tensor) => Tensor][] = [
    ["conv1d", "nd_seq", (x) => x.conv1d(nd("nd_k1"), null, 1, 1)],
    ["conv3d", "nd_vol", (x) => x.conv3d(nd("nd_k3"), null, 1, 1)],
    ["max_pool1d", "nd_seq", (x) => x.maxPool1d(2)],
    ["max_pool3d", "nd_vol", (x) => x.maxPool3d(2)],
    ["interpolate", "nd_img", (x) => x.upsample(2)],
    ["BatchNorm3d", "nd_vol", (x) => new nn.BatchNorm3d(2).call(x)],
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
    () => flat().indexSelect(0, Tensor.from([1, 0], [2], { dtype: "int64" })));
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
    () => new nn.BatchNorm3d(2).call(vol()));

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
    new nn.BatchNorm3d(2).call(x).sum().backward();
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
    const a = Tensor.from([1, 2, 3, 4], [2, 2], { requiresGrad: true });
    const b = Tensor.from([5, 6], [1, 2], { requiresGrad: true });
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
    Tensor.from(d === "bool" ? [1, 0] : [1, 2], [2], { dtype: d });

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
              Tensor.from([value], [], { dtype: kind }))));
      }
    }
  }
  addDTypeKept(out);
}

/**
 * **자리만 옮기는 연산은 형을 그대로 둔다** — `dtype::자리만::`.
 *
 * 경계는 값을 만드는가다. 고르기·자르기·이어붙이기·갈아끼우기는 원래 형이 그대로
 * 나오고, 셈을 하는 것은 승격 규칙을 따른다.
 *
 * 이 표가 없어서 결함이 하나 살아 있었다 — 데이터셋이 `int64` 라벨에서 표본을 꺼내니
 * `float32` 가 나왔는데 **값이 맞아서** 골든 어디에도 안 걸렸다. 값으로 서로를
 * 대조하는 것만으로는 이름표가 떨어지는 것을 영원히 못 본다.
 *
 * **여태 결속을 통해서만 재고 있었다.** 갭 표가 이 묶음을 "파이썬 서명 이야기" 로
 * 적어 두었는데 그것은 이웃한 `공장::`·`별칭::` 의 사정이었고, 이쪽은 `t.dtype` 하나만
 * 묻는 순수한 이쪽 성질이다.
 */
function addDTypeKept(out: Map<string, Case>): void {
  const ints = (): Tensor => Tensor.from(
    Array.from({ length: 12 }, (_, i) => i), [3, 4], { dtype: "int64" });
  const flags = (): Tensor => Tensor.from(
    Array.from({ length: 12 }, (_, i) => (i % 2 === 0 ? 1 : 0)), [3, 4],
    { dtype: "bool" });
  const pick = (): Tensor => Tensor.from([0, 2], [2], { dtype: "int64" });
  const spread = (): Tensor => Tensor.from(
    [0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1], [3, 4], { dtype: "int64" });

  const kept = (fn: (t: Tensor) => Tensor): (t: Tensor) => string =>
    (t) => {
      try {
        return dtypeName(fn(t).dtype);
      } catch (err) {
        return `<${err instanceof Error ? err.constructor.name : "?"}>`;
      }
    };

  const moves: [string, (t: Tensor) => Tensor][] = [
    ["reshape", (t) => t.reshape([4, 3])],
    ["ravel", (t) => t.ravel()],
    ["squeeze", (t) => t.reshape([1, 12]).squeeze(0)],
    ["unsqueeze", (t) => t.unsqueeze(0)],
    ["transpose", (t) => t.swapaxes(0, 1)],
    ["t", (t) => t.transpose()],
    ["permute", (t) => t.permute([1, 0])],
    ["flip", (t) => t.flip(0)],
    ["select", (t) => t.select(0, 1)],
    ["narrow", (t) => t.narrow(1, 1, 2)],
    ["diagonal", (t) => t.diagonal()],
    ["chunk[0]", (t) => t.chunk(2, 0)[0] ?? t],
    ["unbind[0]", (t) => t.unbind(0)[0] ?? t],
    ["tensor_split[0]", (t) => t.tensorSplit(3, 1)[0] ?? t],
    ["index_select", (t) => t.indexSelect(0, pick())],
    ["gather", (t) => t.gather(1, spread())],
    // `take` 는 저쪽에 그 이름이 없다 — 평평하게 편 뒤 고르는 것이 같은 것이다.
    ["take", (t) => t.reshape([t.size]).indexSelect(0, pick())],
    ["cat", (t) => Tensor.cat([t, t], 0)],
    ["stack", (t) => Tensor.stack([t, t], 0)],
    ["repeat", (t) => t.repeat(2, 1)],
    ["roll", (t) => t.roll(1, 0)],
    ["tril", (t) => t.tril()],
    ["triu", (t) => t.triu()],
    ["pad", (t) => t.pad(1, 1, 1)],
    ["as_strided", (t) => t.asStrided([2, 2], [1, 2])],
    ["diag_embed", (t) => t.diagEmbed()],
    ["slice_scatter",
      (t) => t.sliceScatter(Tensor.zeros([3, 2]).to("int64"), 1, 0, 2)],
    ["select_scatter",
      (t) => t.selectScatter(Tensor.zeros([4]).to("int64"), 0, 1)],
    ["scatter", (t) => t.scatterSet(1, spread(), t)],
    ["sort[0]", (t) => t.sort(1).values],
  ];
  for (const [name, fn] of moves) {
    // **정수와 참거짓 둘 다 묻는다.** 하나만 물으면 "float32 로 떨어뜨리는" 결함이
    // 나머지 하나에서만 살아남을 수 있다.
    out.set(`dtype::자리만::${name}(int64)`, () => kept(fn)(ints()));
    out.set(`dtype::자리만::${name}(bool)`, () => kept(fn)(flags()));
  }
  // **`topk` 는 정수만 묻는다.** 참거짓에서는 torch 가 거절하는데 예외 종류가 우리와
  // 달라, 그것은 형 보존이 아니라 거절 문구의 이야기다.
  out.set("dtype::자리만::topk[0](int64)",
    () => kept((t) => t.topk(2, 1).values)(ints()));

  // `maskedSelect` 만 위의 표 밖이다 — **저쪽이 비동기**라서다(결과 길이가 값에
  // 달렸다). 표에 억지로 끼우면 서른한 줄 전부가 비동기가 된다.
  for (const [tag, src] of [["int64", ints], ["bool", flags]] as const) {
    out.set(`dtype::자리만::masked_select(${tag})`, async () => {
      const t = src();
      return dtypeName((await t.maskedSelect(t.binary("gt", Tensor.full([], 5)))).dtype);
    });
  }

  // ── 묻는 것 셋 — **거짓이 나오는 입력을 먼저 쟀다** ──────────────────
  //
  // 늘 참인 술어는 케이스로 물어도 묻는 게 아니다. 셋 다 torch 에서 실제로 거짓이
  // 나오는 입력이 있고, 그 입력으로 묻는다.
  const floats = (): Tensor => Tensor.from([1.5, -2.5, 3.0], [3]);
  const predicates: [string, () => string][] = [
    ["is_floating_point(float32)", () => verdict(floats().isFloatingPoint())],
    ["is_floating_point(int64)", () => verdict(ints().isFloatingPoint())],
    ["is_floating_point(bool)", () => verdict(flags().isFloatingPoint())],
    ["is_signed(float32)", () => verdict(floats().isSigned())],
    ["is_signed(int64)", () => verdict(ints().isSigned())],
    ["is_signed(bool)", () => verdict(flags().isSigned())],
  ];
  for (const [label, fn] of predicates) out.set(`dtype::묻는것::${label}`, fn);

  out.set("dtype::묻는것::is_nonzero(0)",
    async () => verdict(await Tensor.zeros([1]).isNonzero()));
  out.set("dtype::묻는것::is_nonzero(3)",
    async () => verdict(await Tensor.full([1], 3).isNonzero()));
  // 여럿이면 멈춘다 — `if tensor:` 가 조용히 첫 원소를 보는 일을 막는 자리다.
  out.set("dtype::묻는것::is_nonzero(여럿)은 멈춘다", async () => {
    try {
      await floats().isNonzero();
    } catch (err) {
      const said = err instanceof Error ? err.message : String(err);
      return said.includes("ambiguous") ? "멈췄다" : `다른 문구 <${said}>`;
    }
    return "안 던졌다";
  });
}

/**
 * `print(t)` 가 진짜와 같은가. 값이 아니라 **글자**를 본다.
 *
 * 배우는 사람이 가장 많이 하는 일이 이것이고, 다르게 찍히면 교재의 예시와 화면이
 * 안 맞는다.
 */
function addRepr(out: Map<string, Case>): void {
  const t = (v: number[], shape?: number[], grad = false, d: DType = "float32") =>
    Tensor.from(v, shape ?? [v.length], { requiresGrad: grad, dtype: d });

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
  const mat = (g = false) => Tensor.from([4, 1, 2, 3], [2, 2], { requiresGrad: g });
  const sym = (g = false) => Tensor.from([4, 1, 1, 3], [2, 2], { requiresGrad: g }); // 대칭 양정부호
  const vec = (g = false) => Tensor.from([1, 2], [2], { requiresGrad: g });

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
  addLinalgStruct(out);
}

/** `tests/cases.py` 의 `linalg_struct_cases` 가 쓰는 입력. 손으로 적은 것 그대로. */
const LA_BATCH = [4, 1, 2, 3, 2, 0, 1, 5, 3, -1, 1, 2];
const LA_BATCH_SYM = [4, 1, 1, 3, 9, 2, 2, 5, 2, 0.5, 0.5, 1];
const LA_BATCH_VEC = [1, 2, 3, 1, 0, 4];
const LA_BATCH_RHS = [1, 0, 2, 1, 0, 3, 1, 1, 2, 2, 0, 1];
const LA_RECT = [1, 2, 3, 4, 5, 7];
const LA_SYM3 = [4, 1, 0, 1, 3, 1, 0, 1, 2];
const LA_PIVOT = [1, 2, 3, 4];
const LA_SINGULAR = [1, 2, 2, 4];

/**
 * `linalg` 의 구조 — 배치·직사각·`_ex`·LU.
 *
 * 앞의 `addLinalg` 는 2×2 한 장만 묻는다. torch 의 `linalg` 는 전부 배치이고
 * `qr`·`svd`·`pinv` 는 직사각도 받는다. 한 장만 묻는 골든은 그 둘을 못 본다.
 */
function addLinalgStruct(out: Map<string, Case>): void {
  const bat = (g = false) => Tensor.from(LA_BATCH, [3, 2, 2], { requiresGrad: g });
  const sym = (g = false) => Tensor.from(LA_BATCH_SYM, [3, 2, 2], { requiresGrad: g });
  const vecB = (g = false) => Tensor.from(LA_BATCH_VEC, [3, 2], { requiresGrad: g });
  const rhsB = (g = false) => Tensor.from(LA_BATCH_RHS, [3, 2, 2], { requiresGrad: g });
  const rect = () => Tensor.from(LA_RECT, [3, 2]);
  const sym3 = (g = false) => Tensor.from(LA_SYM3, [3, 3], { requiresGrad: g });
  const mat2 = () => Tensor.from([4, 1, 2, 3], [2, 2]);

  const value: [string, () => Promise<Tensor>][] = [
    // ── 배치 ──────────────────────────────────────────────────────────
    ["batch::det", async () => bat().det()],
    ["batch::inv", async () => bat().inverse()],
    ["batch::solve(벡터)", async () => bat().solve(vecB())],
    ["batch::solve(행렬)", async () => bat().solve(rhsB())],
    ["batch::cholesky", async () => sym().cholesky()],
    ["batch::slogdet/부호", async () => (await bat().slogdet()).sign],
    ["batch::slogdet/로그", async () => (await bat().slogdet()).logabs],
    ["batch::matrix_rank", async () => bat().matrixRank()],
    ["batch::matrix_power", async () => bat().matrixPower(3)],
    ["batch::qr/R", async () => (await bat().qr()).r],
    ["batch::svd/S", async () => (await bat().svd()).s],
    ["batch::eigh/값", async () => (await sym().eigh()).values],
    ["batch::pinv", async () => bat().pinverse()],
    ["batch::logdet", async () => sym().logdet()],
    // 3×3 — 2×2 는 야코비 회전이 한 번뿐이라 쓸어담기 반복을 안 지난다.
    ["3x3::eigh/값", async () => (await sym3().eigh()).values],
    ["3x3::svd/S", async () => (await sym3().svd()).s],
    ["3x3::det", async () => sym3().det()],
    ["3x3::inv", async () => sym3().inverse()],

    // ── 직사각 ────────────────────────────────────────────────────────
    ["rect::qr/R", async () => (await rect().qr()).r],
    ["rect::qr/|Q|", async () => (await rect().qr()).q.abs()],
    ["rect::qr(complete)/|Q|", async () => (await rect().qr("complete")).q.abs()],
    ["rect::svd/S", async () => (await rect().svd()).s],
    ["rect::svd/|U|", async () => (await rect().svd()).u.abs()],
    ["rect::svd(축소)/|U|", async () => (await rect().svd(false)).u.abs()],
    ["rect::pinv", async () => rect().pinverse()],
    ["rect::matrix_rank", async () => rect().matrixRank()],
    ["rect::lstsq", async () => rect().lstsq(Tensor.from([1, 2, 3], [3]))],

    // ── 이름으로 묻기 ─────────────────────────────────────────────────
    // 파이썬 쪽은 `.logabsdet`·`.Q`·`.eigenvalues` 로 묻는다. 여기서는 JS 이름이
    // 그 자리를 채우고, 결속이 둘을 잇는다.
    ["name::slogdet.sign", async () => (await bat().slogdet()).sign],
    ["name::slogdet.logabsdet", async () => (await bat().slogdet()).logabs],
    ["name::qr.R", async () => (await rect().qr()).r],
    ["name::qr.|Q|", async () => (await rect().qr()).q.abs()],
    ["name::svd.S", async () => (await rect().svd()).s],
    ["name::svd.|Vh|", async () => (await rect().svd()).vt.abs()],
    ["name::eigh.eigenvalues", async () => (await sym3().eigh()).values],
    ["name::eigh.|eigenvectors|", async () => (await sym3().eigh()).vectors.abs()],

    // ── `_ex` — 던지는 대신 info 를 준다 ──────────────────────────────
    ["ex::inv_ex/값", async () => (await mat2().invEx()).inverse],
    ["ex::inv_ex/info", async () => (await mat2().invEx()).info],
    ["ex::inv_ex(특이)/info",
      async () => (await Tensor.from(LA_SINGULAR, [2, 2]).invEx()).info],
    ["ex::cholesky_ex/L",
      async () => (await Tensor.from([4, 1, 1, 3], [2, 2]).choleskyEx()).L],
    ["ex::cholesky_ex(비양정)/info",
      async () => (await Tensor.from(LA_SINGULAR, [2, 2]).choleskyEx()).info],
    ["ex::solve_ex/값",
      async () => (await mat2().solveEx(Tensor.from([1, 2], [2]))).result],
    ["ex::solve_ex/info",
      async () => (await mat2().solveEx(Tensor.from([1, 2], [2]))).info],
  ];
  for (const [name, fn] of value) out.set(`linalg::${name}`, fn);

  // ── LU ──────────────────────────────────────────────────────────────
  for (const [tag, src] of [["교환없음", [4, 1, 2, 3]], ["교환", LA_PIVOT]] as const) {
    const a = () => Tensor.from(src as readonly number[], [2, 2]);
    out.set(`linalg::lu::lu_factor/${tag}/LU`, async () => (await a().luFactor()).LU);
    out.set(`linalg::lu::lu_factor/${tag}/pivots`,
      async () => (await a().luFactor()).pivots);
    out.set(`linalg::lu::lu/${tag}/P`, async () => (await a().lu()).P);
    out.set(`linalg::lu::lu/${tag}/L`, async () => (await a().lu()).L);
    out.set(`linalg::lu::lu/${tag}/U`, async () => (await a().lu()).U);
  }
  out.set("linalg::lu::lu_solve(교환)", async () => {
    const f = await Tensor.from(LA_PIVOT, [2, 2]).luFactor();
    return f.LU.luSolveFactored(f.pivots, Tensor.from([1, 2], [2, 1]));
  });

  out.set("linalg::ex::inv(특이)가 던지는 것", async () => {
    try {
      await Tensor.from(LA_SINGULAR, [2, 2]).inverse();
    } catch (e) {
      return e instanceof LinAlgError
        ? "LinAlgError 로 잡힌다"
        : `다른 것이 났다: ${(e as Error).name}`;
    }
    return "예외가 안 났다";
  });

  // ── 배치의 기울기 ───────────────────────────────────────────────────
  // **값이 맞는데 기울기가 안 맞는 자리가 여기다.** 역방향 상수가 배치마다 다른데
  // 한 장 것을 돌려 쓰면 첫 장만 맞고 나머지가 조용히 틀린다.
  const grads: [string, (g: boolean) => Tensor, (x: Tensor) => Promise<Tensor>][] = [
    ["det", bat, async (x) => x.det()],
    ["logdet", sym, async (x) => x.logdet()],
    ["slogdet", bat, async (x) => (await x.slogdet()).logabs],
    ["inv", bat, async (x) => x.inverse()],
    ["cholesky", sym, async (x) => x.cholesky()],
    ["matrix_power", bat, async (x) => x.matrixPower(3)],
    ["3x3/inv", sym3, async (x) => x.inverse()],
    ["3x3/cholesky", sym3, async (x) => x.cholesky()],
  ];
  for (const [name, src, fn] of grads) {
    out.set(`linalg::batch::grad::${name}`, async () => {
      const x = src(true);
      seeded(await fn(x)).backward();
      return gradOf(x, name);
    });
  }

  for (const [tag, rhs] of [["벡터", vecB], ["행렬", rhsB]] as const) {
    for (const [which, who] of ["a", "b"].entries()) {
      out.set(`linalg::batch::grad::solve(${tag})/${who}`, async () => {
        const a = bat(true);
        const b = rhs(true);
        seeded(await a.solve(b)).backward();
        return gradOf(which === 0 ? a : b, `batch solve ${tag}/${who}`);
      });
    }
  }
  addLinalgEx(out);
  addLinalgNames(out);
}

/**
 * `_ex` 변종과 LDL·반사자.
 *
 * **`_ex` 는 던지는 대신 `info` 로 알린다** — 0 이면 잘 됐고, `k` 면 `k` 번째 피벗이
 * 0 이다(1 부터 센다). 잘 되는 행렬로만 재면 그 수가 늘 0 이라 자리만 있는 것과
 * 구분이 안 가므로 특이행렬로도 묻는다.
 */
function addLinalgEx(out: Map<string, Case>): void {
  const pivot2 = (): Tensor => Tensor.from([1, 2, 3, 4], [2, 2]);
  // 대칭이다 — LDL 은 대칭에만 뜻이 있다.
  const lin4 = (): Tensor => Tensor.from(
    [2.0, 1.0, 0.5, -1.0, 1.0, 3.0, -0.5, 0.25,
      0.5, -0.5, 2.5, 0.75, -1.0, 0.25, 0.75, 4.0], [4, 4]);
  const singular2 = (): Tensor => Tensor.from([1, 2, 2, 4], [2, 2]);
  // 대각을 3 만큼 올려 계수를 채운다 — 안 올리면 열이 서로 비례해서 특이행렬이다.
  const rect53 = (): Tensor => Tensor.from(
    Array.from({ length: 15 }, (_, i) =>
      i / 4 - 1.5 + (Math.floor(i / 3) === i % 3 ? 3 : 0)), [5, 3]);

  const shapes: [string, () => Tensor][] = [
    ["정사각", pivot2], ["직사각", rect53],
  ];
  for (const [tag, src] of shapes) {
    out.set(`linalg::ex::lu_factor_ex/${tag}/LU`,
      async () => (await src().luFactorEx()).LU);
    out.set(`linalg::ex::lu_factor_ex/${tag}/pivots`,
      async () => (await src().luFactorEx()).pivots);
    out.set(`linalg::ex::lu_factor_ex/${tag}/info`,
      async () => (await src().luFactorEx()).info);
  }
  // **특이행렬로도 물어야** `info` 가 자리만 지키는 0 이 아니라는 것이 드러난다.
  out.set("linalg::ex::lu_factor_ex/특이행렬 info",
    async () => (await singular2().luFactorEx()).info);

  out.set("linalg::ex::ldl_factor/LD",
    async () => (await lin4().ldlFactor()).LD);
  out.set("linalg::ex::ldl_factor/pivots",
    async () => (await lin4().ldlFactor()).pivots);
  out.set("linalg::ex::ldl_factor_ex/info",
    async () => (await lin4().ldlFactorEx()).info);
  out.set("linalg::ex::ldl_solve", async () => {
    const got = await lin4().ldlFactor();
    return got.LD.ldlSolve(Tensor.from(
      [1.0, -2.0, 0.5, 0.25, -1.5, 3.0, 2.0, 0.5], [4, 2]));
  });

  // **반사자 꼴 QR.** `geqrf` 가 담고 `householderProduct` 가 `Q` 로 편다.
  //
  // **정사각으로도 물어야 한다.** 대각 아래가 전부 0 이면 반사를 안 하고(`tau = 0`)
  // 값을 그대로 두는데, 정사각의 마지막 열이 늘 그 자리다 — 직사각으로만 물으면
  // 그 열이 안 나와서 거기서 부호를 뒤집어도 안 걸린다.
  for (const [tag, src] of [["정사각", lin4], ["직사각", rect53]] as const) {
    out.set(`linalg::ex::geqrf/${tag}/a`, async () => (await src().geqrf()).a);
    out.set(`linalg::ex::geqrf/${tag}/tau`,
      async () => (await src().geqrf()).tau);
    out.set(`linalg::ex::householder_product/${tag}`, async () => {
      const got = await src().geqrf();
      return got.a.householderProduct(got.tau);
    });
  }
}

/**
 * `linalg` 의 조합층 — 있는 것에 이름을 붙이는 자리와, 갈래가 있는 노름.
 *
 * 계산이 새로 필요한 것은 `matrixExp` 하나뿐이다. 나머지는 조합인데, **조합이
 * 자명하지 않은 자리**가 셋 있다: 노름의 갈래, `diagonal` 의 축, `eigh` 가 한쪽
 * 삼각만 읽는다는 것. 셋 다 파이썬 쪽 주석에 이유를 적었다.
 */
function addLinalgNames(out: Map<string, Case>): void {
  const mat = () => Tensor.from([4, 1, 2, 3], [2, 2]);
  const sym = () => Tensor.from([4, 1, 1, 3], [2, 2]);
  const sym3 = () => Tensor.from(LA_SYM3, [3, 3]);
  const rect = () => Tensor.from(LA_RECT, [3, 2]);
  const vec3 = () => Tensor.from([3, -4, 0], [3]);
  const upper = () => Tensor.from([2, 1, 0, 3], [2, 2]);
  const cube = () => Tensor.arange(24).reshape([2, 3, 4]);
  // 위쪽에 99 를 넣어도 답이 안 바뀌어야 한다 — 아래 삼각만 읽는지 묻는 자리.
  const skew = () => Tensor.from([4, 99, 1, 3], [2, 2]);
  // `eig` 용. 회전은 **실수 고윳값이 없고**(±i), 일반은 셋 다 실수인데 대칭이 아니다.
  const rot = () => Tensor.from([0, -1, 1, 0], [2, 2]);
  const gen = () => Tensor.from([4, 1, 2, 0, 3, -1, 1, 0, 2], [3, 3]);

  const value: [string, () => Promise<Tensor>][] = [
    ["name2::matmul", async () => mat().mm(mat())],
    ["name2::vecdot", async () => mat().vecdot(mat())],
    ["name2::cross", async () => Tensor.from([1, 2, 3], [3])
      .cross(Tensor.from([4, 5, 6], [3]))],
    ["name2::svdvals", async () => mat().svdvals()],
    ["name2::svdvals(직사각)", async () => rect().svdvals()],
    ["name2::eigvalsh", async () => sym().eigvalsh()],
    ["name2::eigvalsh(3x3)", async () => sym3().eigvalsh()],
    ["name2::eigvalsh(아래삼각만)", async () => skew().eigvalsh()],
    ["name2::eigh(아래삼각만)/값", async () => (await skew().eigh()).values],

    // ── `eig` — 대칭이 아닌 것도 받는다 ────────────────────────────────
    //
    // **차례에 안 기댄다.** LAPACK 이 고윳값 차례를 안 정하고, torch 는 복소수를
    // 정렬조차 못 한다(파이썬 쪽 실측). 그래서 대칭함수로 접어 묻는다.
    ["eig::eigvals(회전)/크기",
      async () => (await rot().eigvals()).abs().sort().values],
    ["eig::eigvals(비대칭)/크기",
      async () => (await gen().eigvals()).abs().sort().values],

    ["name2::linalg.diagonal", async () => cube().diagonal(0, -2, -1)],
    ["name2::torch.diagonal(다른 축)", async () => cube().diagonal(0, 0, 1)],
    ["name2::linalg.diagonal(offset)", async () => mat().diagonal(1, -2, -1)],

    ["name2::vector_norm", async () => vec3().vectorNorm()],
    ["name2::vector_norm(행렬을 통째로)", async () => mat().vectorNorm()],
    ["name2::vector_norm(dim)", async () => mat().vectorNorm(2, 1)],

    ["name2::multi_dot", async () => mat().mm(mat()).mm(mat())],
    ["name2::multi_dot(둘)", async () => mat().mm(mat())],
    ["name2::vander", async () => Tensor.from([1, 2, 3], [3]).vander()],
    ["name2::vander(N)", async () => Tensor.from([2, 3], [2]).vander(4)],
    ["name2::solve_triangular(위)",
      async () => upper().solveTriangular(Tensor.from([1, 3], [2, 1]), true)],
    ["name2::solve_triangular(아래)",
      async () => Tensor.from([2, 0, 1, 3], [2, 2])
        .solveTriangular(Tensor.from([1, 2], [2, 1]), false)],
    ["name2::solve_triangular(단위대각)",
      async () => upper().solveTriangular(Tensor.from([1, 3], [2, 1]), true, true, true)],
    ["name2::tensorsolve", async () => Tensor.eye(4).reshape([2, 2, 2, 2])
      .tensorSolve(Tensor.from([1, 2, 3, 4], [2, 2]))],
    ["name2::tensorinv", async () => Tensor.eye(4).reshape([2, 2, 2, 2]).tensorInv(2)],

    ["name2::matrix_exp(멱영)",
      async () => Tensor.from([0, 1, 0, 0], [2, 2]).matrixExp()],
    ["name2::matrix_exp", async () => mat().matrixExp()],
    ["name2::matrix_exp(큰 값)",
      async () => Tensor.from([20, 5, 10, 15], [2, 2]).matrixExp()],
    ["name2::matrix_exp(3x3)", async () => sym3().matrixExp()],
    ["name2::matrix_exp(배치)",
      async () => Tensor.from(LA_BATCH, [3, 2, 2]).matrixExp()],
    ["name2::torch.matrix_exp", async () => mat().matrixExp()],
  ];
  for (const [name, fn] of value) out.set(`linalg::${name}`, fn);

  for (const [tag, ordv] of [["1", 1], ["inf", Infinity], ["-inf", -Infinity],
    ["0", 0], ["3", 3]] as const) {
    out.set(`linalg::name2::vector_norm(ord=${tag})`,
      async () => vec3().vectorNorm(ordv));
  }
  for (const [tag, ordv] of [["fro", "fro"], ["nuc", "nuc"], ["2", 2], ["-2", -2],
    ["1", 1], ["-1", -1], ["inf", Infinity]] as const) {
    out.set(`linalg::name2::matrix_norm(ord=${tag})`,
      async () => mat().matrixNorm(ordv));
  }
  out.set("linalg::name2::matrix_norm(기본)", async () => mat().matrixNorm());
  out.set("linalg::name2::matrix_norm(배치)",
    async () => Tensor.from(LA_BATCH, [3, 2, 2]).matrixNorm());
  for (const [tag, pv] of [["기본", null], ["fro", "fro"], ["nuc", "nuc"], ["2", 2],
    ["-2", -2], ["1", 1], ["inf", Infinity]] as const) {
    out.set(`linalg::name2::cond(p=${tag})`, async () => mat().cond(pv));
  }
  // **답이 늘 복소수다.** 대칭 행렬을 줘도 그렇다 — 회전 행렬처럼 실수 고윳값이
  // 아예 없는 것이 있어서, 반환형이 입력에 따라 달라질 수가 없다.
  out.set("linalg::eig::eigvals(대칭이어도 복소수형)",
    async () => `torch.${(await sym().eigvals()).dtype}`);

  // **합은 대각합이다.** 차례와 무관하고 값이 맞는지를 수학으로 묻는다 — 고윳값을
  // 아무렇게나 내는 구현은 크기 정렬은 통과해도 여기서 걸린다.
  out.set("linalg::eig::eigvals(비대칭)/합=대각합", async () => {
    const sum = await (await gen().eigvals()).sum().real().item();
    const tr = await gen().trace().item();
    return `${sum.toFixed(4)} ${tr.toFixed(4)}`;
  });

  // **고유벡터는 못 굳힌다** — 부호가 안 정해진다(torch 자신도 float32 와 float64
  // 에서 반대 부호를 낸다). 대신 정의를 묻는다: `A·V` 와 `V·diag(λ)` 가 같은가.
  // 부호가 뒤집혀도 양쪽이 같이 뒤집혀 답이 안 변한다.
  out.set("linalg::eig::eig(정의를 지키나)", async () => {
    const a = gen();
    const { values, vectors } = await a.eig();
    const left = a.cfloat().mm(vectors);
    const right = vectors.mm(values.diagflat());
    return (await left.sub(right).abs().max(0).values.max(0).values.item()).toFixed(4);
  });

  addLinalgGrads(out);
}

/**
 * 분해의 기울기.
 *
 * 오래 안 넣었다 — 유도가 까다롭고 틀리면 조용히 틀린다. 바뀐 것은 유도가 쉬워진
 * 것이 아니라 **대조할 것이 생긴 것**이다. 골든이 진짜 torch 의 수를 자리마다 들고
 * 있어서, 틀리면 조용히가 아니라 크게 틀린다.
 */
function addLinalgGrads(out: Map<string, Case>): void {
  const MAT = [4, 1, 2, 3];
  const SYM = [4, 1, 1, 3];
  const src: Record<string, [readonly number[], readonly number[]]> = {
    mat: [MAT, [2, 2]],
    sym: [SYM, [2, 2]],
    sym3: [LA_SYM3, [3, 3]],
    rect: [LA_RECT, [3, 2]],
    small: [MAT.map((v) => v * 0.1), [2, 2]],
  };
  const grads: [string, string, (x: Tensor) => Promise<Tensor>][] = [
    ["svdvals", "mat", async (x) => x.svdvals()],
    ["svd/S", "mat", async (x) => (await x.svd()).s],
    ["svd/S(직사각)", "rect", async (x) => (await x.svd(false)).s],
    ["eigvalsh", "sym", async (x) => x.eigvalsh()],
    ["eigh/값", "sym", async (x) => (await x.eigh()).values],
    ["eigh/값(3x3)", "sym3", async (x) => (await x.eigh()).values],
    // 고유벡터는 **제곱해서** 묻는다 — 열 부호는 구현이 정하고 야코비와 LAPACK 이
    // 다르게 고른다. `V∘V` 는 그것과 무관하다. 파이썬 쪽 주석에 이유를 적었다.
    ["eigh/벡터²", "sym", async (x) => (await x.eigh()).vectors.square()],
    ["eigh/벡터²(3x3)", "sym3", async (x) => (await x.eigh()).vectors.square()],
    ["qr/R", "mat", async (x) => (await x.qr()).r],
    ["qr/Q", "mat", async (x) => (await x.qr()).q],
    ["qr/R(직사각)", "rect", async (x) => (await x.qr()).r],
    ["qr/Q(직사각)", "rect", async (x) => (await x.qr()).q],
    ["pinv", "mat", async (x) => x.pinverse()],
    // **직사각이 진짜 시험이다.** 정사각에서는 빠뜨린 항이 0 이 되어 안 드러난다.
    ["pinv(직사각)", "rect", async (x) => x.pinverse()],
    ["pinv(3x3)", "sym3", async (x) => x.pinverse()],
    ["matrix_exp", "mat", async (x) => x.matrixExp()],
    ["matrix_exp(3x3)", "sym3", async (x) => x.matrixExp()],
    ["matrix_exp(작은 값)", "small", async (x) => x.matrixExp()],
  ];
  for (const [name, key, fn] of grads) {
    out.set(`linalg::grad2::${name}`, async () => {
      const [data, shape] = src[key]!;
      const x = Tensor.from(data, shape, { requiresGrad: true });
      seeded(await fn(x)).backward();
      return gradOf(x, name);
    });
  }

  out.set("linalg::grad2::이어 붙이기", async () => {
    const x = Tensor.from(MAT, [2, 2], { requiresGrad: true });
    const s = await x.svdvals();
    const loss = s.mul(s).sum().add(await x.matrixNorm("nuc"));
    loss.backward();
    return gradOf(x, "svdvals→노름");
  });
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
    const x = Tensor.from(IP_PLAIN, [4], { requiresGrad: true });
    try {
      x.add_(1);
    } catch {
      return "기대대로 거절";
    }
    return "뜻밖의 성공";
  });

  // `no_grad` 안에서는 잎도 고칠 수 있다 — 옵티마이저가 실제로 그렇게 한다.
  out.set("inplace::no_grad 안에서는 된다", () => {
    const x = Tensor.from(IP_PLAIN, [4], { requiresGrad: true });
    noGrad(() => x.add_(1));
    return x;
  });

  // ── 같은 계산에 이름이 둘 (`method2::`) ─────────────────────────────
  //
  // `torch.add(x, y)` 와 `x.add(y)`. 이 저장소는 한 방향 고리만 갖고 있었고, 반대가
  // 없어서 **계산은 다 해 놓고 이름이 한쪽에서만 닿았다.** 교재가 치는 꼴은 메서드
  // 쪽이고, 그때 나는 것은 `AttributeError` 다.
  //
  // **이름이 닿는지만 보면 껍데기도 통과한다** — 값을 물어야 그 이름이 진짜 그 계산에
  // 닿았는지가 드러난다.
  const m2a = (): Tensor => Tensor.from([1, 2, 3, 4], [2, 2]);
  const m2b = (): Tensor => Tensor.from([0.5, 1.5, 2.5, 3.5], [2, 2]);
  const m2sym = (): Tensor => Tensor.from([4, 1, 1, 3], [2, 2]);
  const m2neg = (): Tensor => Tensor.from([-1, 2, -3, 0.5], [2, 2]);
  const vec3a = (): Tensor => Tensor.from([1, 2, 3], [3]);
  const vec3b = (): Tensor => Tensor.from([4, 5, 6], [3]);

  const named: [string, () => Tensor][] = [
    ["add", () => m2a().add(m2b())],
    ["sub", () => m2a().sub(m2b())],
    ["mul", () => m2a().mul(m2b())],
    ["div", () => m2a().div(m2b())],
    ["multiply", () => m2a().multiply(m2b())],
    ["true_divide", () => m2a().trueDivide(m2b())],
    ["floor_divide", () => m2a().floorDivide(m2b())],
    ["remainder", () => m2a().remainder(m2b())],
    ["fmod", () => m2a().fmod(2.0)],
    ["lerp", () => m2a().lerp(m2b(), 0.5)],
    ["greater", () => m2a().greater(m2b())],
    ["less_equal", () => m2a().lessEqual(m2b())],
    ["logical_and", () => m2a().logicalAnd(m2b())],
    ["logical_not", () => m2a().logicalNot()],
    ["isclose", () => m2a().isclose(m2b())],
    ["nan_to_num", () => m2a().nanToNum()],
    ["fmax", () => m2a().fmax(m2b())],
    ["inner", () => m2a().inner(m2b())],
    ["count_nonzero", () => m2a().countNonzero()],
    ["adjoint", () => m2a().adjoint()],
    ["moveaxis", () => m2a().moveaxis(0, 1)],
    ["t", () => m2a().t()],
    ["lgamma", () => m2a().lgamma()],
    ["digamma", () => m2a().digamma()],
    ["log_softmax", () => m2a().logSoftmax(1)],
    ["hardshrink", () => m2a().hardshrink()],
    ["corrcoef", () => m2a().corrcoef()],
    ["cov", () => m2a().cov()],
    ["cross", () => vec3a().cross(vec3b())],
    ["vdot", () => vec3a().vdot(vec3b())],
    ["kron", () => vec3a().kron(Tensor.from([4, 5], [2]))],
    ["broadcast_to", () => Tensor.from([1, 2], [2]).broadcastTo([3, 2])],
    ["prelu", () => m2neg().prelu(Tensor.from([0.25], [1]))],
  ];
  for (const [name, fn] of named) out.set(`method2::${name}`, fn);

  // 값을 읽어야 하는 넷은 따로 — 저쪽이 비동기다.
  out.set("method2::inverse", async () => m2a().inverse());
  out.set("method2::pinverse", async () => m2a().pinverse());
  out.set("method2::qr", async () => (await m2a().qr()).r);
  out.set("method2::svd", async () => (await m2a().svd()).s);
  out.set("method2::cholesky", async () => m2sym().cholesky());
  out.set("method2::slogdet", async () => (await m2a().slogdet()).logabs);
  out.set("method2::det", async () => m2a().det());
  out.set("method2::logdet", async () => m2sym().logdet());
  out.set("method2::matrix_exp", async () => m2a().matrixExp());
  out.set("method2::matrix_power", async () => m2a().matrixPower(2));

  // **함수로 부른 것과 같아야 한다.** 이름만 닿고 다른 계산이면 여기서 갈린다.
  out.set("method2::함수와 같은 답", async () => {
    const x = m2a();
    const y = m2b();
    const det = await x.det();
    const expm = await x.matrixExp();
    return Tensor.cat([
      x.add(y).sub(x.add(y)).reshape([4]),
      x.mul(y).sub(x.mul(y)).reshape([4]),
      det.sub(det).reshape([1]),
      expm.sub(expm).reshape([4]),
    ], 0);
  });

  // 제자리 단항 열다섯. **`acosh` 는 1 이상, `logit` 은 0..1 안에서만 답이 있다** —
  // 밖은 양쪽 다 NaN 이고 NaN 은 서로 같다고 못 하므로 물어도 아무것도 안 드러난다.
  const m2small = (): Tensor => Tensor.from([0.25, 0.5, 0.75, -0.5], [2, 2]);
  const inplaceUnaries: [string, () => Tensor][] = [
    ["absolute", () => m2small().absolute_()],
    ["acosh", () => m2small().abs().add(Tensor.full([], 1)).acosh_()],
    ["arctan", () => m2small().arctan_()],
    ["arctanh", () => m2small().arctanh_()],
    ["asinh", () => m2small().asinh_()],
    ["atanh", () => m2small().atanh_()],
    ["deg2rad", () => m2small().deg2rad_()],
    ["erfc", () => m2small().erfc_()],
    ["exp2", () => m2small().exp2_()],
    ["fix", () => m2small().fix_()],
    ["negative", () => m2small().negative_()],
    ["rad2deg", () => m2small().rad2deg_()],
    ["sgn", () => m2small().sgn_()],
    ["sinc", () => m2small().sinc_()],
    ["logit", () => m2small().abs().mul(Tensor.full([], 0.8))
      .add(Tensor.full([], 0.1)).logit_()],
  ];
  for (const [name, fn] of inplaceUnaries) {
    out.set(`method2::제자리::${name}_`, fn);
  }

  // 잎에 기울기가 켜져 있으면 제자리 연산을 거절한다. **예외의 종류를 묻는다** —
  // 문구가 아니라 이름이라, 구현끼리 값으로 대조해도 안 걸리는 자리다.
  out.set("method2::제자리::기울기 켜진 잎은 거절", () => {
    try {
      Tensor.from([0.25, 0.5, 0.75, -0.5], [2, 2], { requiresGrad: true })
        .absolute_();
    } catch (err) {
      return err instanceof Error ? err.constructor.name : "?";
    }
    return "예외가 안 났다";
  });

  // ── 인자를 받는 제자리 연산 ─────────────────────────────────────────────
  //
  // **모양이 바뀌는 것들이 여기 있다.** `transpose_`·`squeeze_`·`unsqueeze_` 는 값이
  // 아니라 보는 틀을 고친다. 파이썬 쪽은 이것을 2×2 로 물어서 **모양이 안 바뀐 채
  // 통과했다** — 정사각에서는 틀이 그대로여도 답이 같기 때문이다. 그래서 여기서도
  // 직사각으로 묻는다.
  const square = () => Tensor.from([1, 2, 3, 4], [2, 2]);
  const rect = () => Tensor.from([1, 2, 3, 4, 5, 6], [2, 3]);

  out.set("method2::제자리::인자를 받는 것", () => {
    const x = square();
    x.transpose_();
    const y = square();
    y.tril_();
    const z = square();
    z.cumsum_(1);
    return Tensor.cat([x.reshape([4]), y.reshape([4]), z.reshape([4])]);
  });

  out.set("method2::제자리::모양이 바뀐다", async () => {
    const x = rect();
    x.transpose_();
    const y = rect();
    y.unsqueeze_(0);
    const z = rect().reshape([1, 2, 3]);
    z.squeeze_(0);
    return [x, y, z].map((t) => `(${t.shape.join(", ")})`).join(" ");
  });

  out.set("method2::제자리::transpose_ 의 값", () => {
    const x = rect();
    x.transpose_();
    return x;
  });

  // 답의 철자는 파이썬의 `str(True)` 다 — 골든을 굳힌 쪽이 파이썬이라 "true" 로는
  // 안 맞는다. 같은 질문에 같은 철자로 답해야 대조가 대조다.
  out.set("method2::제자리::같은 텐서인가", async () => {
    const x = Tensor.from([0.25, 0.5, 0.75, -0.5], [2, 2]);
    return `같은 것=${x.inplaceUnary("abs") === x ? "True" : "False"}`;
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

  const mat = (g = false) => Tensor.from([1, 2, 3, 4, 5, 6, 7, 8, 9], [3, 3], { requiresGrad: g });
  const vec = (g = false) => Tensor.from([1, 2, 3, 4], [4], { requiresGrad: g });
  // 0 이 섞인 입력. 흔한 유도(out/x)는 여기서 나눗셈이 터져 조용히 NaN 을 흘린다.
  const zeroed = (g = false) => Tensor.from([2, 0, 3, 4], [4], { requiresGrad: g });
  const short = (g = false) => Tensor.from([1, 5, 2], [3], { requiresGrad: g });

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
    // **예측과 표적을 다르게 준다.** 같으면 기울기가 전부 0 이라 부호가 뒤집힌
    // 구현도, 개수로 안 나눈 구현도 통과한다.
    ["l1_loss", (a, b) => a.l1Loss(b), () => [x1(true), asLeaf(x1().neg())]],
    ["mse_loss", (a, b) => a.mseLoss(b), () => [x1(true), xp(true)]],
    ["smooth_l1_loss", (a, b) => a.smoothL1Loss(b), () => [x1(true), xp(true)]],
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
    () => [Tensor.from([-1, 0, 1, 0], [4], { requiresGrad: true })], (x) => x.unary("relu"));

  weighted("median()", () => [vec(true)], (x) => x.median().values);
  weighted("median(dim)", () => [mat(true)], (x) => x.median(1).values);

  // 골든이 이항 형태로 굳혔으므로 이름 뒤에 잎 표시가 붙는다.
  // **예측과 표적을 다르게 준다** — 같으면 기울기가 전부 0 이다.
  one("L1Loss(층)/a", x1, (x) => x.l1Loss(inp.get("x1").neg()));
  one("SmoothL1Loss(층)/a", x1, (x) => x.smoothL1Loss(inp.get("xp")));
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
    (x) => x.logSoftmax(-1).nllLoss(Tensor.from([0, 1, 2], [3], { dtype: "int64" })));
  one("cross_entropy", x2,
    (x) => x.crossEntropy(Tensor.from([0, 1, 2], [3], { dtype: "int64" })));

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

  const mat2 = () => Tensor.from([2, 0, 1, 1, 3, 2, 0, 1, 4], [3, 3], { requiresGrad: true });
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
        Tensor.from([1, 2, 3, 4], [4], { requiresGrad: true }),
        Tensor.from([1, 5, 2], [3], { requiresGrad: true }),
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

  // ── 접히는 자리 — **동점이 있어야 열린다** ──────────────────────────────
  //
  // 여러 칸이 한 칸으로 접히면, 되돌릴 때 어디로 가는가가 갈린다. 값이 전부
  // 다르면 어떤 규칙을 써도 같은 답이 나와서 이 물음이 안 열린다. 이유는 파이썬
  // 쪽 `grad_cases` 에 길게 적었다.
  //
  // **여기 없는 것이 있다.** `max()`·`min()` 의 축 없는 꼴은 borch.ts 에 없다 —
  // `max(dim)` 만 있고 그것은 번호를 건네므로 규칙이 반대다. 축 없는 꼴은 결속이
  // 만들고, 그래서 그 두 케이스는 코어와 결속만 답한다.
  const tied = () => Tensor.from([3, 5, 5, 1, 5], [5], { requiresGrad: true });
  const fold = (name: string, fn: (x: Tensor) => Tensor) => {
    out.set(`grad::접힘::${name}`, () => {
      const x = tied();
      fn(x).sum().backward();
      return gradOf(x, name);
    });
  };
  fold("amax() 동점 셋", (x) => x.amax());
  fold("amin() 동점 없음", (x) => x.amin());
  fold("max(dim=0) 은 한 자리로", (x) => x.max(0).values);
  fold("norm(inf)", (x) => x.vectorNorm(Infinity));
  fold("norm(-inf)", (x) => x.vectorNorm(-Infinity));
  fold("norm(3)", (x) => x.vectorNorm(3));
  out.set("grad::접힘::angle() 은 0 을 흘린다", () => {
    const x = Tensor.from([0.5, -1, 2], [3], { requiresGrad: true });
    x.angle().sum().backward();
    return gradOf(x, "angle");
  });

  // 아래는 한동안 코어 쪽 `tests/test_fold_grad.py` 에만 있었다 — borch.ts 가 그때
  // 답을 못 해서 셋을 함께 묻는 자리에 못 올렸던 것들이다. 이제 셋 다 답한다.
  const leaf = (v: number[]) => Tensor.from(v, [v.length], { requiresGrad: true });
  const even = () => leaf([1, 5, 5, 5]);
  const dup = () => leaf([1, 1, 2, 2, 2]);
  const nanTie = () => leaf([1, NaN, 5, 5, 5]);
  const back = (x: Tensor, got: Tensor, tag: string) => {
    got.sum().backward();
    return gradOf(x, tag);
  };

  // **축이 없으면 번호도 없다.** `median()` 은 텐서 하나를 주고 `median(0)` 은
  // 값·번호 쌍을 준다 — 규칙이 반대인 것이 서명에도 그대로 드러나 있다.
  const vals = (r: Tensor | { values: Tensor; indices: Tensor }) =>
    r instanceof Tensor ? r : r.values;
  fold("median() 동점 셋", (x) => vals(x.median()));
  out.set("grad::접힘::median() 짝수·동점",
    () => { const x = even(); return back(x, vals(x.median()), "median 짝수"); });
  fold("median(dim=0) 은 한 자리로", (x) => x.median(0).values);
  out.set("grad::접힘::nanmedian() 동점", async () => {
    const x = nanTie();
    return back(x, vals(await x.nanmedian()), "nanmedian");
  });
  out.set("grad::접힘::nanmedian(dim=0)", async () => {
    const x = nanTie();
    return back(x, vals(await x.nanmedian(0)), "nanmedian(0)");
  });
  out.set("grad::접힘::mode() 는 마지막 자리로", async () => {
    const x = dup();
    return back(x, (await x.mode()).values, "mode");
  });
  fold("kthvalue(2)", (x) => x.kthvalue(2).values);
  for (const [tag, q, src] of [
    ["quantile(0.5) 정확히 맞음", 0.5, tied],
    ["quantile(0.3) 보간", 0.3, tied],
    ["quantile(0.5) 짝수는 둘로", 0.5, even],
    ["quantile(0.75) 짝수", 0.75, even],
  ] as [string, number, () => Tensor][]) {
    out.set(`grad::접힘::${tag}`, async () => {
      const x = src();
      return back(x, await x.quantile(q), tag);
    });
  }
  // 도함수가 `i1` 이다. 여기가 **0 을 흘리고 있었고**, 그 주석이 코어의 구멍을
  // 근거로 대고 있었다 — 값이 0 인 기울기와 기울기가 없는 것은 다른 말인데,
  // 베낄 때 뒤가 앞으로 바뀌었다.
  out.set("grad::접힘::i0() 의 도함수는 i1", () => {
    const x = leaf([0.5, -1, 2]);
    x.i0().sum().backward();
    return gradOf(x, "i0");
  });
  fold("topk(3) 는 셋 다", (x) => x.topk(3).values);
  fold("sort() 는 전부 하나씩", (x) => x.sort().values);
  fold("cummax(0) 은 늦은 자리를", (x) => x.cummax(0).values);
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
      () => Tensor.from([0, 0, 0], [3], { requiresGrad: true }).backward(),
      "grad can be implicitly created only for scalar outputs"],
    ["requires_grad 없이 backward",
      () => Tensor.zeros([3]).sum().backward(),
      "does not require grad"],
    ["여러 원소에 item()",
      () => Tensor.zeros([3]).item(),
      "cannot be converted to Scalar"],
    ["backward 두 번", () => {
      const x = Tensor.from([1.0, 2.0], [2], { requiresGrad: true });
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
      const x = Tensor.from([1, 2, 3], [3], { requiresGrad: true });
      x.add_(1);
    }, null],
    ["인덱스 범위 초과", () => Tensor.zeros([3]).select(0, 5), "out of bounds"],
    ["정수 텐서에 requires_grad",
      () => Tensor.from([1, 2, 3], [3], { requiresGrad: true, dtype: "int64" }), null],
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
  const vec = () => Tensor.from(F_VEC, [4], { requiresGrad: true });
  const mat = () => Tensor.from(F_MAT, [3, 3], { requiresGrad: true });
  const pair = () => Tensor.from(F_PAIR, [2, 3], { requiresGrad: true });
  const sym = () => Tensor.from(F_SYM, [2, 2], { requiresGrad: true });
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
      const x = Tensor.from(src, shape, { requiresGrad: true });
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
  const m = (grad = false) => Tensor.from(seq(6), [2, 3], { requiresGrad: grad });
  const sq = (grad = false) => Tensor.from(SQUARE, [3, 3], { requiresGrad: grad });
  const line = (grad = false) => Tensor.from(LINE, [5], { requiresGrad: grad });
  const col = (grad = false) => Tensor.from(COL, [2, 1], { requiresGrad: grad });
  const pair = (grad = false) => Tensor.from([1.0, 2.0], [2], { requiresGrad: grad });
  // **랭크 3.** 축을 바꾸는 것을 2차원으로만 물으면 `(0,1)` 밖의 자리를 못 본다 —
  // 2차원에서는 어느 두 축을 골라도 답이 하나뿐이라 축 인자를 버리는 구현도 통과한다.
  // 여기서는 `permute` 로 적는다. borch.ts 의 `transpose()` 는 2차원 전용이고 축을
  // 안 받는다 — 파이썬 쪽이 두 축을 받아 이 순서를 만들어 넘긴다.
  const cube = (grad = false) => Tensor.from(seq(24), [2, 3, 4], { requiresGrad: grad });

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
    // **자기 역이 아닌 순열.** 위의 넷은 전부 대합이라(두 축 맞바꾸기·뒤집기)
    // 순열을 거꾸로 적용하는 구현도 통과한다. `(1,2,0)` 은 역이 `(2,0,1)` 이고
    // 모양부터 다르다 — [3,4,2] 대 [4,2,3].
    ["permute(비가역)", () => cube().permute([1, 2, 0])],
    ["permute(비가역의 역)", () => cube().permute([2, 0, 1])],
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
    // **역방향에서 더 그렇다** — 되돌리려면 역순열이 필요한데, 순열이 자기 역이면
    // 앞뒤로 같은 배열을 쓰고도 답이 맞는다.
    ["permute(비가역)", () => cube(true).permute([1, 2, 0]), cube],
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
  const vec = (grad = false) => Tensor.from(M_VEC, [4], { requiresGrad: grad });
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

/**
 * 최상위 순환 여덟 — `torch.lstm` 과 그 형제들.
 *
 * **한동안 여기 없었다.** 결속이 borch.ts 를 그대로 부르므로 값은 대조됐지만,
 * borch.ts 의 **직접 표면**(가중치 목록의 차례, 내놓는 것의 개수)은 아무도 안
 * 물었다. 못 옮긴 이유는 가중치가 `cases.py` 안에서 뽑혀 `golden.json` 에 안
 * 실렸기 때문이고, `golden_inputs()` 로 옮기고서 열렸다.
 *
 * 묻는 것이 값이 아니라 **배선**이라 조각마다 이름을 붙인다 — `lstm` 은 셋을
 * 펴고(`출력, h_n, c_n`) 나머지는 둘이다. 하나만 보면 나머지가 안 걸린다.
 */
function addRnnTop(out: Map<string, Case>, inp: Inputs): void {
  const w = (prefix: string, count: number) =>
    Array.from({ length: count }, (_, i) => inp.get(`rt_${prefix}${i}`));
  const x = () => inp.get("rt_x");
  const xb = () => inp.get("rt_xb");
  const h1 = () => inp.get("rt_h1");
  const c1 = () => inp.get("rt_c1");
  const h2 = () => inp.get("rt_h2");
  const c2 = () => inp.get("rt_c2");

  // 이름이 하나라도 어긋나면 그 케이스는 **조용히 안 돈다** — 러너의 "골든에 없는
  // 이름" 줄이 아니면 못 본다. 여기 이름에는 한글 꼬리와 공백이 있어 더 잘 어긋난다.
  const many: [string, (o: rnn.RnnOptions, two: boolean) => Tensor[]][] = [
    ["lstm", (o, two) => rnn.lstm(o.batchFirst ? xb() : x(),
      two ? [h2(), c2()] : [h1(), c1()],
      w(two ? "lstm_two" : "lstm_w", o.hasBiases === false ? 2 : (two ? 8 : 4)), o)],
    ["gru", (o, two) => rnn.gru(o.batchFirst ? xb() : x(), two ? h2() : h1(),
      w(two ? "gru_two" : "gru_w", o.hasBiases === false ? 2 : (two ? 8 : 4)), o)],
    ["rnn_tanh", (o, two) => rnn.rnnTanh(o.batchFirst ? xb() : x(), two ? h2() : h1(),
      w(two ? "rnn_tanh_two" : "rnn_tanh_w", o.hasBiases === false ? 2 : (two ? 8 : 4)), o)],
    ["rnn_relu", (o, two) => rnn.rnnRelu(o.batchFirst ? xb() : x(), two ? h2() : h1(),
      w(two ? "rnn_relu_two" : "rnn_relu_w", o.hasBiases === false ? 2 : (two ? 8 : 4)), o)],
  ];
  for (const [name, call] of many) {
    const pieces = name === "lstm" ? 3 : 2;
    for (let k = 0; k < pieces; k++) {
      out.set(`rnntop::${name}[${k}]`, () => {
        const got = call({}, false)[k];
        if (!got) throw new Error(`${name}[${k}] 이 없다`);
        return got;
      });
    }
    out.set(`rnntop::${name}(batch_first)`, () => call({ batchFirst: true }, false)[0]!);
    out.set(`rnntop::${name}(has_biases=False)`, () => call({ hasBiases: false }, false)[0]!);
    out.set(`rnntop::${name}(num_layers=2)`, () => call({ numLayers: 2 }, true)[0]!);
    out.set(`rnntop::${name}(num_layers=2) 마지막 상태`,
      () => call({ numLayers: 2 }, true)[1]!);
  }

  // 셀 넷. **한 걸음**이라 목록이 아니라 텐서 넷을 낱개로 받는다 — 목록으로 받는
  // 위쪽과 인자 꼴이 다르고, 그 차이가 여기서만 드러난다.
  const hs = () => inp.get("rt_hs");
  const cs = () => inp.get("rt_cs");
  const xs = () => inp.get("rt_xs");
  const cw = (n: string) => w(`${n}_w`, 4) as [Tensor, Tensor, Tensor, Tensor];

  for (let k = 0; k < 2; k++) {
    out.set(`rnntop::lstm_cell[${k}]`, () => {
      const [a, b, ci, di] = cw("lstm_cell");
      return rnn.lstmCell(xs(), [hs(), cs()], a, b, ci, di)[k]!;
    });
  }
  out.set("rnntop::lstm_cell(편향 없이)", () => {
    const [a, b] = cw("lstm_cell");
    return rnn.lstmCell(xs(), [hs(), cs()], a, b)[0];
  });
  const plain: [string, typeof rnn.gruCell][] = [
    ["gru_cell", rnn.gruCell],
    ["rnn_tanh_cell", rnn.rnnTanhCell],
    ["rnn_relu_cell", rnn.rnnReluCell],
  ];
  for (const [name, fn] of plain) {
    out.set(`rnntop::${name}`, () => {
      const [a, b, ci, di] = cw(name);
      return fn(xs(), hs(), a, b, ci, di);
    });
    out.set(`rnntop::${name}(편향 없이)`, () => {
      const [a, b] = cw(name);
      return fn(xs(), hs(), a, b);
    });
  }

  // 드롭아웃 0 은 **학습 중에도 아무것도 안 버린다.** 0 이 아닌 값은 우리 층에
  // 없어서 거절하는데, 0 까지 막으면 정상 경로가 닫힌다.
  out.set("rnntop::dropout=0 이면 돈다", () =>
    rnn.lstm(x(), [h1(), c1()], w("drop_w", 4),
      { dropout: 0, train: true })[0]);
}
