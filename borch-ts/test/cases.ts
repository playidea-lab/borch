/**
 * The golden cases' **bodies, on the TypeScript side.**
 *
 * `tests/golden.json` carries the answers alone. A case body
 * (`lambda L: L.acos(L.tensor(unit))`) is Python and does not mechanically become TS, so
 * it is **rewritten here under the same name.** That division is what makes the golden
 * language-neutral — the expensive half (numbers obtained by running real torch) is
 * carried across and the cheap half (one line of calls) is rewritten.
 *
 * ## The names have to match
 *
 * A name written here that differs from `golden.json`'s key by one character makes that
 * case **quietly not run.** It is why the runner separately counts and reports "registered
 * and absent from the golden" — running 0 cases through a typo and saying "all passed" is
 * the worst outcome in this project.
 *
 * ## Why the inputs are here
 *
 * The arrays the `math::` family uses are not in `golden_inputs()`; they are written
 * directly inside `tests/cases.py`. So they do not arrive through the JSON's `inputs` —
 * the same values are written again here. Divergent values stop the comparison from being
 * a comparison, so they were carried across verbatim.
 */

import {
  getAutocastCpuDtype, getAutocastDtype, getAutocastGpuDtype, getAutocastIpuDtype,
  getAutocastXlaDtype, isAutocastCacheEnabled, isAutocastCpuEnabled,
  isAutocastEnabled, isAutocastIpuEnabled, isAutocastXlaEnabled,
  areDeterministicAlgorithmsEnabled, getDeterministicDebugMode,
  getFloat32MatmulPrecision, isAnomalyCheckNanEnabled, isAnomalyEnabled,
  isDeterministicAlgorithmsWarnOnlyEnabled, isWarnAlwaysEnabled,
  getNumInteropThreads, getNumThreads,
} from "../src/index.js";
import { cudnnIsAcceptable, isVulkanAvailable, narrowCopy, segmentReduce }
  from "../src/index.js";
import { type DType, dtypeName } from "../src/dtype.js";
import { einsum } from "../src/einsum.js";
import * as fft from "../src/fft.js";
import { istft, stft } from "../src/fft.js";
import * as nn from "../src/nn.js";
import * as rnn from "../src/rnn.js";
import { igamma, igammac, polygamma } from "../src/special.js";
// **The namespace, not the kernels.** `special.ts` above holds the incomplete-gamma
// and polygamma shaders; this is `torch.special`'s twenty-two forwarding names, which
// is a different thing under a confusingly close file name — see its header.
import * as special from "../src/special_names.js";
import * as optim from "../src/optim.js";
import { load, save, type Savable } from "../src/serialize.js";
import * as vision from "../src/vision.js";
import * as v2f from "../src/vision_v2.js";
import * as v2twins from "../src/vision_v2_twins.js";
import * as datasets from "../src/datasets.js";
import * as ops from "../src/ops.js";
import * as data from "../src/data.js";
import * as F from "../src/functional.js";
import { LinAlgError } from "../src/errors.js";
// **The namespace, not the methods behind it.** Every `linalg` case here reaches a
// method directly, which is why `linalg.norm` could sit taking one argument where the
// method takes four and no case noticed: the door was never the thing being opened.
import * as linalg from "../src/linalg.js";
import { noGrad, Tensor } from "../src/tensor.js";

/**
 * One case.
 *
 * Usually it produces the resulting tensor. **Some produce a string** — cases that froze a
 * verdict rather than a value, such as whether `equal` is true or which exception is
 * raised. Those have to be exactly equal rather than approximately.
 */
export type Case = () => Tensor | string | Promise<Tensor | string>;

/** One input the golden carried across. The values are flat and the shape arrives
 *  separately. */
export interface RawInput {
  readonly shape?: number[];
  readonly values?: (number | boolean | null)[];
}

/**
 * The inputs the cases share.
 *
 * **What the golden brought is used as it is.** Writing the arrays out again here makes
 * that a place to be wrong, and being wrong shows on screen only as "our values differ".
 * Using the very numbers the answers were frozen with is what makes the comparison a
 * comparison.
 */
export class Inputs {
  constructor(private readonly raw_: Readonly<Record<string, RawInput>>) {}

  /** A new tensor every time — cases sharing a tensor accumulate gradients. */
  get(name: string, requiresGrad = false): Tensor {
    const entry = this.raw_[name];
    if (!entry?.values) throw new Error(`골든에 입력 '${name}' 이 없다`);
    const flat = entry.values.map((v) =>
      typeof v === "boolean" ? (v ? 1 : 0) : (v ?? Number.NaN));
    return Tensor.from(flat, entry.shape ?? [flat.length], { requiresGrad });
  }

  /** The values without making a tensor. Used by things that do not go to the GPU, such
   *  as images. */
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

// ── The inputs `math_cases` uses in tests/cases.py. Carried across verbatim. ──
const plain = [0.5, 2.0, -1.5, 3.0];
const unit = [0.2, 0.6, -0.9, 0.45]; // (-1, 1) 안
const big = [1.5, 2.5, 3.0, 1.2]; // > 1
const pos = [0.5, 2.0, 1.5, 3.0];
const other = [1.0, 2.0, -3.0, 0.5];
const logitIn = [0.2, 0.6, 0.35, 0.45]; // (0, 1) 안
const weights = [1.0, 2.0, 3.0, 4.0]; // 자리마다 다른 가중치

/** Each function has its own domain. Called outside it the answer is NaN, and NaN cannot
 *  be compared. */
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
 * The unaries `math::` uses. **The left is the golden's name and the right is the
 * operation in our table.**
 *
 * torch's aliases (`arccos` = `acos`) go to the same kernel here. A kernel per alias means
 * a day when only one of them is fixed.
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

/** A step function. **It flows 0** — absent and zero are different. */
const STEPS: readonly string[] = ["sign", "floor", "ceil", "round", "trunc", "fix"];

/**
 * The table with no prefix — outside the textbook's range and common all the same.
 *
 * The inputs come from `Inputs`. That not one number is written here is the point.
 *
 * **What is left out**: the ones needing a sort (`median`, `topk`, `sort`, `unique`,
 * `argsort`), the ones whose output size depends on the values (`masked_select`,
 * `bincount`), the ones needing an integer dtype (`F.one_hot`, `F.nll_loss`), and
 * convolution, pooling, `BatchNorm2d`, `bmm`, `einsum` and `pad_sequence`, which are T2.
 */
function addWide(out: Map<string, Case>, inp: Inputs): void {
  const xp = () => inp.get("xp");
  const x1 = () => inp.get("x1");
  const x2 = () => inp.get("x2");
  const tail = () => inp.get("tail");

  // xp is positives only — log2 and rsqrt are NaN on a negative, and NaN differs even
  // from itself.
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
    // **A negative dimension counts from the end**, and the core did not: it built the
    // axis as `dim` leading slices, `range(-1)` is empty, and the selection landed on
    // axis 0. This side had it right from the start — `dim < 0 ? dim + rank : dim`, one
    // line in `indexSelect` — so the case is here to keep the two together.
    ["index_select(dim=-1)",
      () => x2().indexSelect(-1, Tensor.from([3, 1], [2]))],
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
    // Convolution and pooling. **Asking about stride 2 separately is deliberate** — the
    // backward path that inserts zeros between the gradients runs only there.
    ["F.conv2d", () => inp.get("img").conv2d(inp.get("cw"), inp.get("cb"), 1, 1)],
    ["F.conv2d(패딩0)", () => inp.get("img").conv2d(inp.get("cw"), null, 1, 0)],
    ["F.conv2d(스트라이드2)",
      () => inp.get("img").conv2d(inp.get("cw"), inp.get("cb"), 2, 1)],
    ["F.max_pool2d", () => inp.get("img").maxPool2d(2)],
    ["F.avg_pool2d", () => inp.get("img").avgPool2d(2)],
    ["BatchNorm2d(학습)", () => new nn.BatchNorm2d(3).call(inp.get("img"))],
    // **Evaluation mode after a save and a restore.** With the running statistics left
    // out of state_dict it diverges here and nowhere else — training looks fine and
    // inference alone is wrong, the defect the core lived through.
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
    // Where different lengths go into one batch. The textbook's ch05 uses this path as
    // it is.
    ["pad_sequence", () => Tensor.padSequence(ragged())],
    ["pad_sequence(batch_first)", () => Tensor.padSequence(ragged(), true)],
    ["pad_sequence(채움값)", () => Tensor.padSequence(ragged(), true, -1.0)],
    ["pad_sequence(2차원)",
      () => Tensor.padSequence([x2().narrow(0, 0, 3), x2().narrow(0, 0, 1)], true)],
  ];
  for (const [name, fn] of table) out.set(name, fn);

  // The ones whose output size depends on the values — they go to the CPU and back once,
  // so they are asynchronous.
  out.set("unique", async () => Tensor.from([1, 1, 2, 3], [4]).unique());
  // `returnInverse` and `dim` were missing from the middle on both sides, so
  // `x.unique(true, true)` asked for the inverse in torch and for nothing here.
  // Asked by position rather than by keyword because that is the call the gap made
  // wrong.
  {
    const dups = (): Tensor =>
      Tensor.from([3, 1, 2, 1, 3, 3], [6], { dtype: "int64" });
    out.set("spot::unique(inverse)",
      async () => (await dups().unique(true, true))[1] as Tensor);
    out.set("spot::unique(inverse, counts)",
      async () => (await dups().unique(true, true, true))[2] as Tensor);
    // **`dim` folds whole slices, which is a different operation from the two rows
    // above**, and it was written down as not carried across for exactly that
    // reason. Both are torch's, so the reason was true and was doing an excuse's
    // work; these two rows sat red under it.
    //
    // The grid has a repeated row on purpose: without one, an implementation that
    // returns every slice passes, and folding is the whole point.
    const grid3 = (): Tensor =>
      Tensor.from([1, 2, 3, 4, 1, 2], [3, 2], { dtype: "int64" });
    out.set("spot::unique(dim=0)",
      async () => (await grid3().unique(true, false, false, 0)) as unknown as Tensor);
    out.set("spot::unique(dim=0, inverse)",
      async () => (await grid3().unique(true, true, false, 0))[1] as Tensor);
    // `norm`'s `p`, `dim` and `keepdim` arrived in the same edit — this took no
    // arguments at all, so every `x.norm(2, 1)` was silently the norm of everything.
    const plane = (): Tensor =>
      Tensor.from([1, -2, 3, 4, 0, -1], [2, 3]);
    out.set("spot::norm(p=2, dim=1, keepdim)", () => plane().norm(2, 1, true));
    out.set("spot::norm(p=inf, dim=1, keepdim)", () => plane().norm(Infinity, 1, true));
  }
  out.set("masked_select",
    async () => x1().maskedSelect(x1().binary("gt", Tensor.full([], 0))));
  out.set("bincount",
    async () => Tensor.from([0, 1, 1, 3], [4], { dtype: "int64" }).bincount());

  /** `x2[:3, :3]` — some golden cases use that corner alone. */
  function square3(): Tensor {
    return x2().narrow(0, 0, 3).narrow(1, 0, 3);
  }
}

/**
 * Stands a computed tensor up **as a leaf.**
 *
 * The golden side builds the value in advance and puts it in as a leaf, as in
 * `L.tensor(x2.T.copy(), requires_grad=True)`. Using `x2().transpose()` directly here
 * makes it a derived tensor, no gradient accumulates, and the case dies as "no gradient
 * arrived" — with the implementation perfectly fine.
 *
 * The buffer is shared. There is no reason to recompute the value.
 */
function asLeaf(t: Tensor): Tensor {
  return new Tensor(t.buffer, t.shape, { requiresGrad: true });
}

/**
 * **Where it breaks** as the rank rises.
 *
 * For the sister library this was a table measuring a limit — TF.js refuses some
 * operations from rank 7, so cases marked `=거절` had "refusing is the right answer".
 * borch.ts has no such limit, so **succeeding, as torch does, is the right answer**, and
 * the answers are torch's verbatim.
 *
 * The whole value is asked about — reduced to a scalar, swapped positions still sum the
 * same and it passes. The gradients are received multiplied by a different weight per
 * position too.
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

    // **All three are asked by value — at rank 6 as at rank 8.** 7 and 8 used to be
    // frozen with "refusing is the right answer", and that was TF.js's ceiling rather than
    // this implementation's. With that side gone, the question that can be asked rose from
    // "did it avoid throwing" to "is the value right". The second is far stronger.
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

  // The four are kept apart for historical reasons. In the sister library rank 7 had both
  // the forward and the gradient while rank 8 produced a value with no gradient — evidence
  // that the boundary hung on neither the operation's name nor the input's rank. All four
  // produce values now, and they stay as the place recording that a boundary written from
  // a guess becomes documentation of that guess.
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

  // Rank 5 is where the sister library quietly broke values through `tf.pad`. Ours is
  // asked about too.
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

/** Three of lengths 3, 1 and 2. The smallest size at which the padded positions are
 *  visible by eye. */
function ragged(grad = false): Tensor[] {
  return [[1, 2, 3], [4], [5, 6]].map((v) => Tensor.from(v, [v.length], { requiresGrad: grad }));
}

/** `x > 0` as 0/1. The `logical_*` cases use that form. */
function positive(t: Tensor): Tensor {
  return t.binary("gt", Tensor.full([], 0));
}

function piece(parts: Tensor[], k: number): Tensor {
  const part = parts[k];
  if (!part) throw new Error(`조각 ${k} 가 없다`);
  return part;
}

/**
 * Writes true and false **in the spelling the golden froze.**
 *
 * The freezing used Python's `str(bool(...))`, so it is `True`/`False`. JS's
 * `String(true)` is `true` and left alone it does not match. This is a difference in how
 * it is written rather than in the verdict, and the golden is the side holding the
 * answers, so this side matches it.
 */
function verdict(value: boolean): string {
  return value ? "True" : "False";
}

/** Produces `x.grad`. If none arrived it throws rather than passing quietly. */
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
    // **The method spelling reaches the same place.** These were kernels with no
    // name on `Tensor`, so `x.hypot(y)` was a type error here while `binary("hypot")`
    // had been frozen in the golden for months. Asking both is what says the two
    // spellings are one operation rather than two that agree today.
    out.set(`math::${name}/메서드`, () => {
      const lhs = Tensor.from(plain) as unknown as Record<string, (t: Tensor) => Tensor>;
      return lhs[name]!(Tensor.from(other));
    });
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

  // The rest, asked by value alone. They are booleans or steps, so they have no gradient
  // case.
  out.set("math::sgn", () => Tensor.from(plain).unary("sgn"));
  out.set("math::signbit", () => Tensor.from(plain).unary("signbit"));
  // **A 0 in x is what makes this a test of this function.** Without one it is
  // indistinguishable from `x * log(y)`.
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
  addV2Functional(out, inputs);
  addDatasets(out);
  addOps(out);
  addSeq(out, inputs);
  addEdge(out);
  addComplex(out);
  addFft(out);
  addSpecial(out);
  addKeepdim(out);
  addTopRest(out);
  return out;
}

/**
 * The names left at the top level — `top::`.
 *
 * **Two of the golden's `top::` are not asked here.** `device::` has no counterpart in
 * borch.ts (our `device()` is a different function returning the adapter), and
 * `resize_as_` is something the Python binding achieves by swapping handles, so it has no
 * TS surface. Leaving the names unmatched has the runner count them as "absent from the
 * golden", so not using them was the choice made.
 */
function addTopRest(out: Map<string, Case>): void {
  const P = "top::";
  const GRID = [-1.7, 0.3, 2.9, 5.5];
  const SHAPES = [0.5, 1.0, 2.0, 3.0];
  const SPOTS = [0.25, 1.5, 0.5, 4.0];
  const STEPS = [1.0, 2.0, 3.5];
  const x = (grad = false): Tensor => Tensor.from(GRID, undefined, { requiresGrad: grad });
  const a = (): Tensor => Tensor.from(SHAPES);

  // ── The in-place forms that exist only at the top level ──────────────
  //
  // Draws cannot be frozen, and **at `p=0` it is the identity**, so the values are
  // deterministic. And `is` is asked separately — without the same tensor coming back,
  // code that chains on cannot edit the original.
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
  // `feature_dropout` drops **by channel** — the same computation as `dropout2d`.
  out.set(`${P}feature_dropout(p=0)`, () => img1234().featureDropout(0.0, true));

  out.set(`${P}igamma`, () => igamma(a(), Tensor.from(SPOTS)));
  // **One expression cannot cover it** — `x < a+1` is the series and the rest is the
  // continued fraction.
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
  // Moving the zero point moves where it truncates.
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
  // **Outside the range it is 0** — rounding is a step, and inside the range it passes
  // straight through.
  grad("fake_quantize", GRID, (t) => t.fakeQuantizePerTensorAffine(0.5, 0, 0, 7));

  // It does not differentiate with respect to the first argument — torch itself refuses
  // (there is no closed form).
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
 * **This is where an axis quietly disappears.** A shape that does not match stops loudly,
 * and a shape with one axis missing **often still fits** broadcasting — and then it runs
 * all the way through with only the values wrong.
 *
 * `all`, `any` and `countNonzero` had no axis at all. An argument passed to them was
 * quietly discarded and a full reduction came out.
 */
function addKeepdim(out: Map<string, Case>): void {
  const P = "keep::";
  const GRID = [1.0, 4.0, 2.0, 3.0, 0.5, 5.0];
  const FLAGS = [1, 0, 1, 0, 0, 1];
  const g = (grad = false): Tensor =>
    Tensor.from(GRID, [2, 3], { requiresGrad: grad });
  const b = (): Tensor => Tensor.from(FLAGS, [2, 3], { dtype: "bool" });
  const shapeOf = (fn: () => Tensor): Case => () => `(${fn().shape.join(", ")})`;

  // The ones that fold an axis. The golden's names are the Python spellings, so they are
  // used as they are.
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

  // The ones producing a pair — **the axis has to survive in both.**
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

  // The gradient has to arrive with the axis alive too. Mismatched, it blows up at the
  // leaf — or worse, **spreads** by broadcasting and the values grow.
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
 * A reduction's `dtype=` — `keep::dtype::`.
 *
 * **The rule is one line: cast before folding.** Not after. Asking about the dtype alone
 * cannot separate the two orders, so the value is asked about too — the sum of
 * `[1.7, −2.3, 0.9]` is −1 casting first and 0 casting after.
 *
 * **That this branch is a place to be quietly wrong is already measured.** Among the
 * reductions only `norm` was not listening to `dtype=` while `sum`, `mean` and `prod`
 * were — one of four not listening is worse than none of them listening. The Python side
 * caught that, and the thirty-five here are **the first time the same question is asked of
 * borch.ts.**
 */
/**
 * The named casts — `dtype::형바꾸기::`.
 *
 * **Why another name when `to(dtype)` exists.** Because torch gives both, and because the
 * textbook writes `x.float()`. These names were absent from borch.ts, so only the two
 * Python versions answered, and those fourteen golden cases had gone uncarried as "a
 * Python-side matter". **In fact they were names nobody had ever asked about** — the
 * forty-seven `spot::` were the same shape.
 *
 * For eight of them **the refusal is the answer.** That what matters there is the
 * **wording** rather than the value is this group's point. With the name simply absent the
 * message is `'half' is not defined`, which is indistinguishable from a typo, while the
 * two Python versions stop with `.half()`(float16) is not in the browser subset. Three
 * implementations saying three sentences read to a learner as **three different things.**
 * It is a branch a value comparison can never catch.
 */
function addNamedCasts(out: Map<string, Case>): void {
  const P = "dtype::형바꾸기::";
  const src = (): Tensor => Tensor.from([1.5, -2.5, 3.0], [3]);

  out.set(`${P}float`, () => dtypeName(src().float().dtype));
  out.set(`${P}long`, () => dtypeName(src().long().dtype));
  out.set(`${P}bool`, () => dtypeName(src().bool().dtype));
  out.set(`${P}cfloat`, () => dtypeName(src().cfloat().dtype));
  // **`type_as` follows the other's dtype** — an integer partner gives an integer.
  out.set(`${P}type_as`, () => dtypeName(
    src().typeAs(Tensor.from([1, 2, 3], [3], { dtype: "int64" })).dtype));

  // The Python side's verdict looks at **a fragment** of the message. The same fragment
  // is checked here and the same word returned — freezing the whole sentence diverges on a
  // single character.
  // **It has to be the same fragment as the Python side's `tests/cases.py`.** Diverge here
  // and each side checks only its own sentence, and twenty-one cases stay green while
  // saying different things — which is what happened. The frozen answer is the verdict word
  // `기대대로`, so it does not move however far the sentences drift.
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

  // **`double` does not branch — all three refuse.** This comment said "the core has
  // float64 and only the browser side does not", which was true until the core began
  // narrowing double precision at construction and stopped being true without anything
  // noticing: the case asked whether the call was turned away, and a silent downcast is
  // not a refusal. The core refuses at `_cast` now, so the name says `우리는거절`.
  out.set(`${P}double=우리는거절`, () => {
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
  // Folding a float into an integer is what separates the orders. Integers and booleans
  // look at the promoting side.
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
      // **The dtype name goes through `dtypeName`.** The golden froze Python's
      // `str(dtype)`, which is `torch.float32`, and borch.ts's `.dtype` is `float32` —
      // handed over unchanged, eight cases go red on one spelling.
      out.set(`${P}sum(${kind}→${want}) 의 형`,
        () => dtypeName(src(kind).sum(want).dtype));
      out.set(`${P}cumsum(${kind}→${want})`, () => src(kind).cumsum(0, want));
    }
    // **`sum(dtype=bool)` works and `cumsum(dtype=bool)` does not** — not a rule but a
    // kernel torch never built, so it is invisible unless asked about separately.
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

  // `dtype=` does not release **every** refusal. Three stand (measured).
  const refuses = (name: string, body: () => unknown): void => {
    out.set(`${P}${name}`, () => {
      try {
        body();
      } catch (err) {
        // The Python side froze the exception's **kind name.** Their `RuntimeError` is
        // the same name here, so it matches as it stands.
        return (err as Error).constructor.name;
      }
      return "예외가 안 났다";
    });
  };
  refuses("mean(→int64)는 거절", () => src("실수").mean(undefined, false, "int64"));
  refuses("cumsum(→참거짓)은 거절", () => src("정수").cumsum(0, "bool"));
  refuses("cumprod(→참거짓)은 거절", () => src("정수").cumprod(0, "bool"));

  // **`to` really changes the dtype.** For a long time it read the device string alone
  // and quietly discarded the dtype.
  out.set(`${P}to(float32) 의 형`, () => dtypeName(src("정수").to("float32").dtype));
  out.set(`${P}to(int64) 의 형`, () => dtypeName(src("실수").to("int64").dtype));
  out.set(`${P}to(int64) 의 값`, () => src("실수").to("int64"));

  // **The name is `keep::grad::`** — the prefix the `grad()` helper attaches on the
  // Python side is not `dtype::`. Adding one more segment here produced "a name absent
  // from the golden".
  out.set("keep::grad::sum(dtype=float32)", () => {
    const leaf = Tensor.from([1.0, 4.0, 2.0, 3.0, 0.5, 5.0], [2, 3],
      { requiresGrad: true });
    leaf.sumDim(1, true, "float32").sum().backward();
    return gradOf(leaf, "sum(dtype=float32)");
  });
}

/**
 * The remaining optional arguments — `keep::arg::`.
 *
 * **Two of them were accepting and discarding.** `dist(p)` ignored `p` and always produced
 * L2 (invisible, because the value was of a plausible size), and `div(roundingMode)`
 * matched the value while leaving the dtype a float. The rest had no such argument at all
 * and stopped loudly.
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
 * Fourier — `fft::`.
 *
 * **There is one kernel.** The forward, the inverse, the half transform and all three
 * backwards call the same shader with only the sign and the factor changed. So what this
 * table really asks about is **that combination of arguments.**
 *
 * The gradients matter more than the values. The transform is linear so the forward is
 * easy to get right, and the hard part is **which half is counted** — `rfft` does not add
 * the conjugate partner (adding it doubles), and `irfft` counts the edges once and the
 * middle twice. Both can be wrong **with the forward values perfectly fine**, so with only
 * value cases they pass green.
 */
/**
 * `torch.special` — `special::`.
 *
 * **Twenty-two names and no arithmetic of its own**, so the values are not what is in
 * doubt: `erf` has had a case since the beginning. What this table asks is that the
 * second spelling reaches the same body, and a forwarding namespace has exactly three
 * ways to fail at that.
 *
 * *It can point at the wrong function.* `expit` is `sigmoid`, `gammaln` is `lgamma`,
 * `psi` is `digamma`, `modifiedBesselI0` is `i0`, `gammainc` is `igamma`. Cross any pair
 * and a real number comes back under the wrong name — nothing raises, and only a frozen
 * answer says so.
 *
 * *It can lose an argument.* `polygamma` is `(n, input)` in this namespace and
 * `x.polygamma(n)` as a method — the pair reversed — and `round`'s `decimals` is absent
 * from torch's own docstring while torch reads it.
 *
 * *It can drop a dimension.* `softmax(x, 0)` and `softmax(x, 1)` differ, so a forwarder
 * passing `dim` along wrongly is caught by having both.
 *
 * **A bare unary call would pass with all three mistakes in place**, which is why the
 * arguments are half of this table.
 */
function addSpecial(out: Map<string, Case>): void {
  const P = "special::";
  const XS = [0.3, 0.7, 1.2, 2.5];
  const UNIT = [0.2, 0.8];
  const x = (): Tensor => Tensor.from(XS, [2, 2]);
  const unit = (): Tensor => Tensor.from(UNIT, [2]);

  // The eleven spelled the same, and the four that are not — from here they look
  // identical, which is the point: a crossed pair is invisible in this list and shows
  // up only in the frozen answer.
  out.set(`${P}digamma`, () => special.digamma(x()));
  out.set(`${P}erf`, () => special.erf(x()));
  out.set(`${P}erfc`, () => special.erfc(x()));
  out.set(`${P}exp2`, () => special.exp2(x()));
  out.set(`${P}expm1`, () => special.expm1(x()));
  out.set(`${P}i0`, () => special.i0(x()));
  out.set(`${P}log1p`, () => special.log1p(x()));
  out.set(`${P}sinc`, () => special.sinc(x()));
  out.set(`${P}psi`, () => special.psi(x()));
  out.set(`${P}gammaln`, () => special.gammaln(x()));
  out.set(`${P}expit`, () => special.expit(x()));
  out.set(`${P}modified_bessel_i0`, () => special.modifiedBesselI0(x()));

  // **The open unit interval.** Outside it `erfinv` is ±∞ and `logit` is undefined, and
  // a case standing there is a case about infinity rather than about forwarding.
  out.set(`${P}erfinv`, () => special.erfinv(unit()));
  out.set(`${P}logit`, () => special.logit(unit()));
  out.set(`${P}logit(eps)`, () => special.logit(unit(), 0.3));

  out.set(`${P}round`, () => special.round(x()));
  // The input has to actually move under rounding or the case says nothing.
  out.set(`${P}round(decimals)`,
          () => special.round(Tensor.from([0.34567, 1.98765, -2.55555], [3]), 3));

  out.set(`${P}polygamma`, () => special.polygamma(1, x()));
  out.set(`${P}polygamma(n=2)`, () => special.polygamma(2, x()));
  out.set(`${P}xlogy`, () => special.xlogy(x(), x()));
  out.set(`${P}logsumexp`, () => special.logsumexp(x(), 1));
  out.set(`${P}logsumexp(keepdim)`, () => special.logsumexp(x(), 1, true));
  out.set(`${P}softmax`, () => special.softmax(x(), 1));
  out.set(`${P}log_softmax`, () => special.logSoftmax(x(), 1));
  out.set(`${P}softmax(dim=0)`, () => special.softmax(x(), 0));
  out.set(`${P}log_softmax(dim=0)`, () => special.logSoftmax(x(), 0));

  // **The two that were nearly counted as missing.** The first sweep of this namespace
  // put `gammainc` and `gammaincc` among the names not here; they are `igamma` and
  // `igammac`, which this library has had all along.
  const a = (): Tensor => Tensor.from([0.5, 1.5, 2.5], [3]);
  const z = (): Tensor => Tensor.from([0.3, 1.0, 2.0], [3]);
  out.set(`${P}gammainc`, () => special.gammainc(a(), z()));
  out.set(`${P}gammaincc`, () => special.gammaincc(a(), z()));
  // They must sum to one — a statement about the pair that neither row above makes,
  // since both could be the lower branch and both would still freeze cleanly.
  out.set(`${P}gammainc+gammaincc`,
          () => special.gammainc(a(), z()).add(special.gammaincc(a(), z())));
}

function addFft(out: Map<string, Case>): void {
  const P = "fft::";
  const XS = [1.0, -2.0, 0.5, 3.0, -1.0, 0.25];
  const YS = [0.5, 1.0, -1.5, 0.25, 2.0, -0.5];
  // **A signal that avoids the blade.** A ramp (`arange/8 − 1`) makes the Nyquist bin
  // exactly 0, and there `abs` is not differentiable and the sign hangs on rounding — a
  // place where the case rather than the value is the problem, so it was replaced with
  // numbers that have no zero bin.
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

  // ── Several axes and the Hermitian forms ───────────────────────────────
  //
  // **This group caught a defect in the kernel.** Transforming a complex input along any
  // axis but the last read the imaginary part from the wrong cell — the shader interleaves
  // `(re, im)` when writing and assumed they were an inner-size apart when reading, and
  // **on the last axis those two are accidentally the same.** That is why every 1-D case
  // above passed, and without the 2-D cases here it would still be green.
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
      // **The name is the Python spelling.** Interpolating JS's `true` gives
      // `center=true`, which does not match the golden's `center=True`, and that case
      // **quietly does not run** — which is why the runner separately counts "the name is
      // absent from the golden".
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
  // `alignToWindow` has no value case anywhere: torch's whole behaviour for it is the
  // refusal, and with `center` false it answers the same at every setting.
  out.set(`${P}stft align_to_window(center 이면 거절)`, () => {
    try {
      stft(sig(), 8, {
        hopLength: 4, window: hann(), returnComplex: true, alignToWindow: true,
      });
    } catch (e) {
      return String(e).includes("center = false") ? "거절" : `다른 말 <${String(e)}>`;
    }
    return "안 던졌다";
  });
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

  // **`returnComplex` was in the options type and nothing read it**, and the
  // two-sided branch beside it ran `fft` where the inverse belongs — the function
  // whose name is *inverse* did the forward one. Both defects were in the core as
  // well, and both are invisible on the default path, which uses `irfft` and is
  // right. The reconstruction itself cannot be asked here: the two-sided branch is
  // complex all the way through the overlap-add and this library's kernels stop at
  // complex on purpose, so that row is the Python side's.
  // The Python side spells a dtype `torch.float32`; over here it is the bare name,
  // and the golden is keyed by the string both must produce.
  out.set(`${P}istft 의 형은 실수다`, () => `torch.${istft(
    stft(sig(), 8, { hopLength: 4, window: hann(), returnComplex: true }),
    8, { hopLength: 4, window: hann(), length: 16 }).dtype}`);
  out.set(`${P}istft(onesided 인데 복소를 달라면 거절)`, () => {
    try {
      istft(stft(sig(), 8, { hopLength: 4, window: hann(), returnComplex: true }),
            8, { hopLength: 4, window: hann(), length: 16, returnComplex: true });
    } catch (e) {
      return String(e).includes("Cannot have onesided output")
        ? "문구대로" : `다른 문구 <${String(e).slice(0, 44)}>`;
    }
    return "안 던졌다";
  });

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
 * Complex — `cplx::`.
 *
 * **The Python core (numpy) went first and this follows.** Until now these cases were
 * caught by `golden.py`'s `CORE_ONLY_PREFIXES` and the browser side skipped them entirely.
 * Bodies existing here does not end that skipping outright — **the binding
 * (`borch_webgpu`) still skips**, because the Python binding has no complex names yet. The
 * three implementations' coverage does not move in one line.
 *
 * ## What it asks
 *
 * **The gradients are this table's point** rather than the values. The convention is
 *
 *     z.grad = ∂L/∂re + i·∂L/∂im
 *
 * so a conjugate appears in the backward of a holomorphic function (`mul`, `div`) and does
 * not for `abs`, which produces a real. **With real inputs that difference is invisible**,
 * because conjugation is the identity on reals. So all three are asked in one table.
 *
 * `(z*z).real` **separates the conventions themselves**: under this one it is `2−4j`, and
 * under ordinary complex differentiation `2+4j`. An implementation that only matches
 * values diverges here.
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
  // Promotion asks about the **dtype name.** A real among them still has to come out
  // complex.
  out.set(`${P}complex64 + float32 의 형`, () =>
    dtypeName(z().add(Tensor.from([1.0])).dtype));
  out.set(`${P}complex64 + int64 의 형`, () =>
    dtypeName(z().add(Tensor.from([1]).to("int64")).dtype));

  /**
   * Receives the gradient at **two real leaves.** Not building a complex leaf directly is
   * the point — the value comes apart as `(∂L/∂re, ∂L/∂im)` and which side is wrong is
   * visible.
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
      // The golden froze the **exception's kind name.** The core (numpy) raises
      // `RuntimeError`, and only the same name here lets ported code catch the same
      // thing.
      return "RuntimeError";
    }
  });
}

/**
 * The kinks.
 *
 * Almost every other table's input is a normal draw. That is a good default and **a
 * special value never appears** — exactly 0, exactly two equal numbers, exactly the
 * boundary, exactly .5. Every place a function kinks is there, and that is how `relu` got
 * through 798 golden cases.
 *
 * Folding with a different weight per position is the condition. Folded uniformly, the
 * difference at one kinked position is buried in the sum.
 */
function addEdge(out: Map<string, Case>): void {
  const z = [-2, -1, 0, 1, 2, 0];              // 정확히 0 을 품는다
  const ta = [1, 2, 3, 2], tb = [1, 5, 3, 0];  // 자리 0·2 가 동점
  const half = [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5];
  const dup = [1, 3, 2, 3];

  const set = (name: string, fn: Case): void => { out.set(`edge::${name}`, fn); };
  // **The weights start at 1.** Starting at 0 makes the first position's share 0, and a
  // case whose output is a single cell has that one as its entirety, so **the gradient is
  // 0 outright** — the device meant to avoid uniform folding leaves that case asking
  // nothing.
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

  // ── The ones that kink at 0 ──
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

  // ── clamp landing exactly on the boundary ──
  set("clamp(경계에서)", () => Tensor.from([...z]).clamp(-1, 1));
  grad("clamp(경계에서)", z, (x) => x.clamp(-1, 1));
  grad("clamp(위만)", z, (x) => x.clampMax(1));
  grad("clamp(아래만)", z, (x) => x.clampMin(-1));

  // ── Ties ──
  // **torch divides the gradient on a tie** — two equal inputs get half each. An
  // implementation sending it all to one side, or 1 to both, has a perfectly identical
  // forward and is not caught by values.
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

  // Pooling with two equal values inside the window. **The answer differs from
  // `maximum`'s** — torch's pooling picks one winning position and flows only there rather
  // than dividing. This library's kernel says "the earlier position on a tie", and that
  // had never been confirmed.
  const tied = [1, 1, 2, 0, 1, 0, 2, 2, 3, 3, 0, 1, 0, 3, 1, 1];
  set("max_pool2d(동점)", () => Tensor.from([...tied], [1, 1, 4, 4]).maxPool2d(2));
  set("grad::max_pool2d(동점)", () => {
    const x = Tensor.from([...tied], [1, 1, 4, 4], { requiresGrad: true });
    seed(x.maxPool2d(2)).sum().backward();
    return gradOf(x, "max_pool2d(동점)");
  });

  // ── Rounding rules ──
  // **torch sends .5 to the even side.** Written as `floor(x + 0.5)` everything rounds up
  // and diverges.
  set("round(.5에서)", () => Tensor.from([...half]).round());
  set("floor(정수에서)", () => Tensor.from([...z]).floor());
  set("ceil(정수에서)", () => Tensor.from([...z]).ceil());
  set("trunc(음수)", () => Tensor.from([...half]).trunc());
  set("frac(음수)", () => Tensor.from([...half]).frac());

  // **`%` follows the divisor's sign** — `-7 % 3` is 2 rather than -1. JS's `%` is the
  // opposite, so using it directly diverges on negative inputs alone. It never shows with
  // positives.
  const neg = [-7, -3, 3, 7];
  set("%(음수)", () => Tensor.from([...neg]).remainder(3));
  set("%(음수로 나누기)", () => Tensor.from([...neg]).remainder(-3));
}

/**
 * Recurrent networks and attention.
 *
 * The weights are planted from outside so that **all three start from the same place** —
 * initialising separately shows whether the initialisation diverged rather than what
 * diverged. That the parameter **names** have to match torch's for `state_dict` to plant
 * them is caught here too.
 */
function addSeq(out: Map<string, Case>, inp: Inputs): void {
  // **It goes through torch's names.** The base is the single
  // `Recurrent(in, hidden, kind)`, and calling that alone leaves nobody measuring the name
  // `nn.LSTM` — which is a textbook's first line.
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
  // `batchFirst` at torch's index — sixth on `RNN`, which carries `nonlinearity`, and
  // fifth on the other two. Writing all three alike would put it one seat early on
  // exactly one of them.
  const buildTurned = (kind: nn.RNNKind): nn.Recurrent => {
    const m = kind === "RNN"
      ? new nn.RNN(3, 4, 1, "tanh", true, true)
      : new KINDS[kind](3, 4, 1, true, true);
    const low = kind.toLowerCase();
    m.loadStateDict({
      weight_ih_l0: inp.get(`${low}_wih`), weight_hh_l0: inp.get(`${low}_whh`),
      bias_ih_l0: inp.get(`${low}_bih`), bias_hh_l0: inp.get(`${low}_bhh`),
    });
    return m;
  };
  for (const kind of ["RNN", "LSTM", "GRU"] as const) {
    out.set(`seq::${kind}/출력`, () => build(kind).run(inp.get("seq_x")).output);
    // LSTM alone has two states, so the golden takes the hidden one.
    out.set(`seq::${kind}/마지막상태`,
      () => build(kind).run(inp.get("seq_x")).hidden);
    // **The flag is honoured exactly when feeding `(batch, length, …)` gives back what
    // `(length, batch, …)` gave, turned.** A layer ignoring it reads `seq_x`'s `(5, 2)`
    // as length 5 batch 2 either way and hands back a differently shaped answer — the
    // two cannot even be subtracted, which is where the teeth are.
    //
    // That needs **both axes not 1 and different from each other**. `[1, 6, 8]` read
    // the wrong way round still comes back `[1, 6, 8]` after the turn, so the shapes
    // round-trip and nothing separates. 5×2 satisfies the condition.
    out.set(`seq::${kind}/batch_first 는 같은 답을 돌려놓는다`, () => {
      const turned = buildTurned(kind)
        .run(inp.get("seq_x").swapaxes(0, 1)).output.swapaxes(0, 1);
      return turned.sub(build(kind).run(inp.get("seq_x")).output).abs().amax();
    });
  }

  const attention = (mask: Tensor | null): Tensor => {
    // **`batchFirst` is the ninth argument and it is `true` here**, as the Python
    // case says. It read `(4, 2)` and leaned on the class defaulting to batch-first,
    // which is the option torch does *not* default to — the same answer for the
    // wrong reason, and only while the class disagreed with torch.
    const m = new nn.MultiheadAttention(4, 2, 0, true, false, false, null, null, true);
    m.loadStateDict({
      in_proj_weight: inp.get("mha_in_w"), in_proj_bias: inp.get("mha_in_b"),
      "out_proj.weight": inp.get("mha_out_w"), "out_proj.bias": inp.get("mha_out_b"),
    });
    return m.attend(inp.get("attn_x"), mask);
  };
  out.set("seq::MultiheadAttention", () => attention(null));
  // A causal mask is a **float** (0/-inf). Lumping it into "non-zero means masked"
  // diverges here.
  out.set("seq::MultiheadAttention(인과 마스크)",
    () => attention(nn.MultiheadAttention.causalMask(5)));

  // ── the attention flags, refused here until today ──────────────────────────
  //
  // The layer takes one tensor (`attend` is self-attention), so these go through
  // `multiHeadAttentionForward` with the layer's own weights — which is exactly what
  // the binding does, and what makes the three sides one computation.
  //
  //     project → concat biasK/biasV → split heads → staticK/staticV
  //             → concat the zero step → scores
  const mhaX = (len: number, width: number): Tensor => {
    const v: number[] = [];
    for (let i = 0; i < len * 2 * width; i++) v.push(i * 0.1 - 1.0);
    return Tensor.from(v, [len, 2, width]);
  };
  interface MhaFlags {
    addBiasKv?: boolean;
    addZeroAttn?: boolean;
    kdim?: number;
    vdim?: number;
  }
  const mhaBuild = (f: MhaFlags): nn.MultiheadAttention => {
    const m = new nn.MultiheadAttention(
      4, 2, 0, true, f.addBiasKv ?? false, f.addZeroAttn ?? false,
      f.kdim ?? null, f.vdim ?? null, false);
    m.eval();
    return m;
  };
  const mhaRun = (f: MhaFlags, part: "출력" | "가중치",
                  mask: "attn" | "pad" | null = null) => () => {
    const m = ramp(mhaBuild(f));
    const kd = f.kdim ?? 4, vd = f.vdim ?? 4;
    let attnMask: Tensor | null = null;
    let padMask: Tensor | null = null;
    if (mask === "attn") {
      const g: number[] = new Array(15).fill(0);
      g[0] = -1e9;
      attnMask = Tensor.from(g, [3, 5]);
    } else if (mask === "pad") {
      // The Python side hands a boolean mask; the binding turns it into the additive
      // one this function takes, so the case carries the additive form here.
      const g: number[] = new Array(10).fill(0);
      g[1] = -Infinity;
      padMask = Tensor.from(g, [2, 5]);
    }
    const got = nn.multiHeadAttentionForward(
      mhaX(3, 4), mhaX(5, kd), mhaX(5, vd), 4, 2, m.inWeight, m.inBias,
      m.biasK, m.biasV, m.addZeroAttn, 0, m.outWeight, m.outBias, false,
      padMask, true, attnMask, !m.qkvSameEmbedDim, m.qWeight, m.kWeight,
      m.vWeight);
    return part === "출력" ? got.output : got.weights!;
  };
  const MHA_FLAGS: Array<[string, MhaFlags]> = [
    ["add_bias_kv", { addBiasKv: true }],
    ["add_zero_attn", { addZeroAttn: true }],
    ["둘 다", { addBiasKv: true, addZeroAttn: true }],
    ["kdim, vdim", { kdim: 6, vdim: 7 }],
  ];
  for (const [label, flags] of MHA_FLAGS) {
    for (const part of ["출력", "가중치"] as const) {
      out.set(`seq::MultiheadAttention(${label})/${part}`, mhaRun(flags, part));
    }
    out.set(`seq::MultiheadAttention(${label})/state_dict 열쇠`,
      () => Object.keys(mhaBuild(flags).stateDict()).sort().join(" "));
  }
  for (const [label, flags] of MHA_FLAGS.slice(0, 3)) {
    for (const mask of ["attn", "pad"] as const) {
      out.set(`seq::MultiheadAttention(${label}, ${mask} 마스크)`,
        mhaRun(flags, "출력", mask));
    }
  }

  // ── The transformer ──────────────────────────────────────────────────────
  //
  // **The weights come from each parameter's own shape**, by the same rule as
  // `tests/cases.py`: `0.1·((i mod 7) − 3)` down the flat parameter. No array
  // crosses the boundary, and the two sides only agree if their `stateDict` keys
  // **and shapes** agree — which is half of what these ask. The other half is the
  // wiring, in an order that `normFirst` reverses.
  const ramp = <T extends nn.Module>(mod: T): T => {
    const filled: Record<string, Tensor> = {};
    for (const [name, value] of Object.entries(mod.stateDict())) {
      const n = value.shape.reduce((a, b) => a * b, 1);
      const flat: number[] = [];
      for (let i = 0; i < n; i++) flat.push(0.1 * ((i % 7) - 3));
      filled[name] = Tensor.from(flat, [...value.shape]);
    }
    mod.loadStateDict(filled);
    return mod;
  };
  const tseq = (scale = 1): Tensor => {
    const v: number[] = [];
    for (let i = 0; i < 24; i++) v.push((i * 0.1 - 1.0) * scale);
    return Tensor.from(v, [2, 3, 4]);
  };

  // ── the recurrent flags, refused on all three sides until today ───────────
  //
  // Carried verbatim from `tests/cases.py`: the input is `0.1·i − 1.0` over (5, 2, 3)
  // and every parameter comes from `ramp` above, so nothing crosses the boundary and
  // the two sides agree only if their `stateDict` keys **and shapes** do.
  const rseq = (): Tensor => {
    const v: number[] = [];
    for (let i = 0; i < 30; i++) v.push(i * 0.1 - 1.0);
    return Tensor.from(v, [5, 2, 3]);
  };
  interface RNNFlags {
    numLayers?: number;
    bias?: boolean;
    dropout?: number;
    bidirectional?: boolean;
    projSize?: number;
    nonlinearity?: "tanh" | "relu";
  }
  const rnnBuild = (kind: nn.RNNKind, f: RNNFlags): nn.RNNBase => {
    const {
      numLayers = 1, bias = true, dropout = 0, bidirectional = false,
      projSize = 0, nonlinearity = "tanh",
    } = f;
    // **The fourth seat is not the same argument on all three.** torch's `RNN` puts
    // `nonlinearity` there and the other two put `bias`, so one call written for all
    // three lands a string in `bias`, where it is true.
    const m = kind === "RNN"
      ? new nn.RNN(3, 4, numLayers, nonlinearity, bias, false, dropout, bidirectional)
      : kind === "LSTM"
        ? new nn.LSTM(3, 4, numLayers, bias, false, dropout, bidirectional, projSize)
        : new nn.GRU(3, 4, numLayers, bias, false, dropout, bidirectional);
    m.eval();
    return m;
  };
  const RNN_FLAGS: Array<[string, RNNFlags, readonly nn.RNNKind[]]> = [
    ["양방향", { bidirectional: true }, ["RNN", "LSTM", "GRU"]],
    ["2층", { numLayers: 2 }, ["RNN", "LSTM", "GRU"]],
    ["2층양방향", { numLayers: 2, bidirectional: true }, ["RNN", "LSTM", "GRU"]],
    ["편향없음", { bias: false }, ["RNN", "LSTM", "GRU"]],
    // `eval()` is what makes this deterministic and is also the claim: the dropout
    // goes between the layers while training and is the identity afterwards.
    ["dropout 은 평가에서 항등", { numLayers: 2, dropout: 0.5 },
      ["RNN", "LSTM", "GRU"]],
    ["proj_size", { projSize: 2 }, ["LSTM"]],
    ["proj_size 와 양방향", { numLayers: 2, projSize: 2, bidirectional: true },
      ["LSTM"]],
    ["relu", { nonlinearity: "relu" }, ["RNN"]],
  ];
  for (const [label, flags, kinds] of RNN_FLAGS) {
    for (const kind of kinds) {
      const parts = kind === "LSTM"
        ? ["출력", "상태", "셀"] : ["출력", "상태"];
      for (const part of parts) {
        out.set(`seq::${kind}(${label})/${part}`, () => {
          const got = ramp(rnnBuild(kind, flags)).run(rseq());
          return part === "출력" ? got.output
            : part === "상태" ? got.hidden : got.cell;
        });
      }
      out.set(`seq::${kind}(${label})/state_dict 열쇠`,
        () => Object.keys(rnnBuild(kind, flags).stateDict()).sort().join(" "));
    }
  }
  // The message alone, not the type: Python answers with `ValueError` and JavaScript
  // with `RangeError`, and neither is the other's mistake.
  const rnnRefuses = (build: () => unknown): string => {
    try {
      build();
    } catch (err) {
      return err instanceof Error ? err.message : String(err);
    }
    return "(거절 없음)";
  };
  // `RNN` and `GRU` have no `projSize` seat here, for the reason their classes give,
  // so the refusal is asked through the base — which is where torch raises it too.
  out.set("seq::거절::RNN(proj_size)",
    () => rnnRefuses(() => new nn.RNNBase("RNN", 3, 4, 1, true, false, 0, false, 2)));
  out.set("seq::거절::GRU(proj_size)",
    () => rnnRefuses(() => new nn.RNNBase("GRU", 3, 4, 1, true, false, 0, false, 2)));
  out.set("seq::거절::LSTM(proj_size=hidden)",
    () => rnnRefuses(() => new nn.LSTM(3, 4, 1, true, false, 0, false, 4)));
  out.set("seq::거절::LSTM(proj_size<0)",
    () => rnnRefuses(() => new nn.LSTM(3, 4, 1, true, false, 0, false, -1)));

  const encoderLayer = (normFirst: boolean): Tensor => {
    const m = new nn.TransformerEncoderLayer(4, 2, 8, 0.0, "relu", 1e-5, true, normFirst);
    m.eval();
    return ramp(m).call(tseq());
  };
  out.set("seq::TransformerEncoderLayer", () => encoderLayer(false));
  // **`normFirst` moves the normalisation inside the residual.** Asked at one
  // setting alone the other branch is never run.
  out.set("seq::TransformerEncoderLayer(norm_first)", () => encoderLayer(true));

  out.set("seq::TransformerEncoder(2층)", () => {
    const proto = new nn.TransformerEncoderLayer(4, 2, 8, 0.0, "relu", 1e-5, true);
    const m = new nn.TransformerEncoder(proto, 2);
    m.eval();
    return ramp(m).call(tseq());
  });

  // **The three above are `eval()` and forward only**, which is the shape of the
  // first defect this repository ever found: BatchNorm's backward was wrong and
  // survived a long time because only the forward was being compared. The encoder is
  // the largest block here and the one where a wrong gradient is least visible — a
  // mask, a softmax and three projections meet inside attention, and a forward sum
  // agrees whatever the backward does.
  const rampedLayer = (): nn.TransformerEncoderLayer => {
    const m = new nn.TransformerEncoderLayer(4, 2, 8, 0.0, "relu", 1e-5, true);
    m.eval();
    return ramp(m);
  };
  // The weights start at 1 here, not 0 — a first slot weighted 0 drops the whole of
  // that position out of the fold.
  const foldWeights = (t: Tensor): Tensor =>
    t.mul(Tensor.from(Array.from({ length: t.size }, (_, i) => i + 1), t.shape)).sum();

  out.set("seq::grad::TransformerEncoderLayer/입력", () => {
    const x = asLeaf(tseq());
    foldWeights(rampedLayer().call(x)).backward();
    return gradOf(x, "TransformerEncoderLayer");
  });

  out.set("seq::grad::TransformerEncoderLayer/파라미터 합", async () => {
    const m = rampedLayer();
    foldWeights(m.call(tseq())).backward();
    let total = 0;
    for (const p of m.parameters()) {
      if (p.grad === null) continue;
      total += await p.grad.abs().sum().item();
    }
    // A `Case` answers with a tensor or a string, and this is one number — so it goes
    // back as a 0-d tensor and is compared numerically, which is what the Python side
    // returning a bare float gets as well.
    return Tensor.full([], total);
  });

  // **The one that reads `memory`.** A decoder layer that quietly ignored it would
  // train, converge, and never look at the encoder — with every shape right. So the
  // memory here is a *different* tensor from the target.
  out.set("seq::TransformerDecoderLayer", () => {
    const m = new nn.TransformerDecoderLayer(4, 2, 8, 0.0, "relu", 1e-5, true);
    m.eval();
    return ramp(m).forward(tseq(), tseq(-0.5));
  });

  out.set("seq::Transformer/마스크",
    () => nn.Transformer.generateSquareSubsequentMask(4));
}

/**
 * Transforms shaped like `torchvision.transforms`.
 *
 * **A random transform's draws cannot be compared** — because torch's generator is not
 * available to us. So the golden pins the probability at 0 or 1, or leaves only one place
 * to crop, and asks about the deterministic part alone. Waving it away here as "random, so
 * it cannot be compared" would be recording something unlooked-at as looked-at.
 */
/**
 * `ops::` — box geometry.
 *
 * **The boxes are written out rather than drawn**, and that is the whole design of this
 * block. Overlaps have to be *arranged*: random boxes in a field large enough to hold
 * them mostly miss each other, and a table of zeros passes against an implementation
 * computing entirely the wrong thing. The five below overlap in three different amounts,
 * one sits far away from everything, and one is **a duplicate** — which is what gives
 * `nms` at a low threshold anything to do.
 *
 * Nothing here is drawn, so unlike the rest of the vision side there is no distribution
 * half and no seed. The numbers are the case.
 */
function addOps(out: Map<string, Case>): void {
  const boxes = () => Tensor.from(
    [0, 0, 10, 10, 1, 1, 11, 11, 5, 5, 15, 15, 30, 30, 40, 40, 0, 0, 10, 10], [5, 4]);
  const others = () => Tensor.from([2, 2, 8, 8, 12, 0, 22, 10, 30, 31, 41, 39], [3, 4]);
  const scores = () => Tensor.from([0.9, 0.75, 0.6, 0.95, 0.5]);
  const labels = () => Tensor.from([0, 0, 1, 1, 0]);
  // Pairs for the losses, chosen so the aspect-ratio term is **not** zero. The flat box
  // against the tall one of equal area is the pair that separates the complete loss from
  // the distance loss; the last pair is identical to itself, the one input for which
  // every one of these must return exactly 0.
  const shapes = () => Tensor.from(
    [0, 0, 10, 10, 5, 5, 15, 15, 0, 0, 20, 4, 0, 0, 4, 20, 1, 1, 3, 3], [5, 4]);
  const targets = () => Tensor.from(
    [1, 1, 11, 11, 20, 20, 30, 30, 0, 0, 4, 20, 0, 0, 20, 4, 1, 1, 3, 3], [5, 4]);
  const logits = () => Tensor.from(
    [-1, 2, 0, 0.5, -3, 8, -40, 40, -0.25], [3, 3]);
  const hits = () => Tensor.from([0, 1, 1, 1, 0, 0, 1, 0, 1], [3, 3]);

  // ── sampling a feature map at coordinates that are not integers ────────────
  //
  // **The boxes are arranged here too, and three of the five are why.** One fits, one
  // hangs off the top-left, one off the bottom-right, one is smaller than a cell, and
  // one is in the second image of the batch. Written with the inside ones alone this
  // whole block passes against a reader that clamps a sample to the edge instead of
  // zeroing it — which is what torchvision's *own Python reference* does, and what the
  // first version of the Python side did.
  const featureMap = () => {
    const values: number[] = [];
    for (let i = 0; i < 2 * 3 * 7 * 9; i++) values.push(((i % 17) * 0.3));
    return Tensor.from(values, [2, 3, 7, 9]);
  };
  const rois = () => Tensor.from(
    [0, 1, 1, 6, 5, 0, -3, -2, 4, 3, 0, 6, 4, 20, 20, 0, 2, 2, 2.4, 2.4,
     1, 0, 0, 8, 6], [5, 5]);
  // Eight channels so that `pooled = 2` leaves two out — the position-sensitive pair
  // needs `out * ph * pw`, and one output channel would hide the channel ordering.
  const psMap = () => {
    const values: number[] = [];
    for (let i = 0; i < 1 * 8 * 7 * 9; i++) values.push(((i % 13) * 0.4));
    return Tensor.from(values, [1, 8, 7, 9]);
  };
  const psRois = () => Tensor.from(
    [0, 1, 1, 6, 5, 0, -3, -2, 4, 3, 0, 6, 4, 20, 20, 0, 2, 2, 2.4, 2.4], [4, 5]);

  // **`aligned` moves every value**, so both settings are asked.
  // **The Python cases flatten before freezing**, so these do too — what is compared is
  // the values in order, and the shape is asked on that side.
  const flat = (make: () => Promise<Tensor>) =>
    async () => (await make()).reshape([-1]);

  out.set("ops::roi_align(aligned=False)",
    flat(() => ops.roiAlign(featureMap(), rois(), [3, 2])));
  out.set("ops::roi_align(aligned=True)",
    flat(() => ops.roiAlign(featureMap(), rois(), [3, 2], 1, -1, true)));
  out.set("ops::roi_align(sampling_ratio=2)",
    flat(() => ops.roiAlign(featureMap(), rois(), [3, 2], 1, 2)));
  out.set("ops::roi_align(sampling_ratio=1)",
    flat(() => ops.roiAlign(featureMap(), rois(), [3, 2], 1, 1)));
  out.set("ops::roi_align(spatial_scale=0.5)",
    flat(() => ops.roiAlign(featureMap(), rois(), [3, 2], 0.5)));
  out.set("ops::roi_align(output_size=1)",
    flat(() => ops.roiAlign(featureMap(), rois(), 1)));
  // **The rounding is C's, not JavaScript's.** `Math.round(-0.5)` is `-0` where C gives
  // `-1`, and `spatial_scale = 0.5` puts every odd coordinate on a half.
  out.set("ops::roi_pool", flat(() => ops.roiPool(featureMap(), rois(), [3, 2])));
  out.set("ops::roi_pool(spatial_scale=0.5, halves)",
    flat(() => ops.roiPool(featureMap(), rois(), [3, 2], 0.5)));
  // **The output channel is the slow axis and the bin is the fast one.**
  out.set("ops::ps_roi_align", flat(() => ops.psRoiAlign(psMap(), psRois(), 2)));
  out.set("ops::ps_roi_align(sampling_ratio=2)",
    flat(() => ops.psRoiAlign(psMap(), psRois(), 2, 1, 2)));
  // **An average, not a maximum** — the one thing here that reads like a typo.
  out.set("ops::ps_roi_pool", flat(() => ops.psRoiPool(psMap(), psRois(), 2)));
  out.set("ops::ps_roi_pool(spatial_scale=0.5)",
    flat(() => ops.psRoiPool(psMap(), psRois(), 2, 0.5)));

  // ── the convolution whose sampling positions are learned ────────────────────
  //
  // **The fixture is computed, not drawn.** The Python side built these from
  // `numpy.random.default_rng`, and there is none of that here — which is the whole of
  // why these eight were never asked on this side. `minstd` is the same three numbers
  // on both, and its modulus is small enough that a float64 multiply stays exact.
  const minstd = (count: number, scale: number, seed: number): number[] => {
    const out: number[] = [];
    let state = seed % 2147483647;
    if (state <= 0) state = 1;
    for (let i = 0; i < count; i++) {
      state = (state * 48271) % 2147483647;
      out.push(Math.fround((state / 2147483647 - 0.5) * 2 * scale));
    }
    return out;
  };
  const spread = (shape: number[], scale: number, seed: number): Tensor =>
    Tensor.from(minstd(shape.reduce((a, b) => a * b, 1), scale, seed), shape);

  interface Deform {
    seed: number; batch: number; inC: number; outC: number;
    kh: number; kw: number; h: number; w: number;
    wgroups: number; ogroups: number; useMask: boolean;
    stride: [number, number]; padding: [number, number]; dilation: [number, number];
  }
  const DEFORM: Deform = {
    seed: 3, batch: 2, inC: 4, outC: 6, kh: 3, kw: 3, h: 7, w: 8,
    wgroups: 1, ogroups: 1, useMask: false,
    stride: [1, 1], padding: [0, 0], dilation: [1, 1],
  };
  const deform = (over: Partial<Deform>) => {
    const d = { ...DEFORM, ...over };
    const outH = Math.floor(
      (d.h + 2 * d.padding[0] - (d.dilation[0] * (d.kh - 1) + 1)) / d.stride[0]) + 1;
    const outW = Math.floor(
      (d.w + 2 * d.padding[1] - (d.dilation[1] * (d.kw - 1) + 1)) / d.stride[1]) + 1;
    // **The mask is a gain in [0, 1)**, which is what the Python side builds by adding
    // half to a spread of half — written as a spread of one it would go negative and
    // stop being a gain.
    const maskValues = minstd(d.batch * d.ogroups * d.kh * d.kw * outH * outW,
                              0.5, d.seed + 3).map((v) => v + 0.5);
    return {
      input: spread([d.batch, d.inC, d.h, d.w], 2.0, d.seed),
      offset: spread([d.batch, d.ogroups * 2 * d.kh * d.kw, outH, outW],
                     2.4, d.seed + 2),
      weight: spread([d.outC, d.inC / d.wgroups, d.kh, d.kw], 0.6, d.seed + 1),
      bias: spread([d.outC], 2.0, d.seed + 4),
      mask: d.useMask
        ? Tensor.from(maskValues,
                      [d.batch, d.ogroups * d.kh * d.kw, outH, outW])
        : null,
      settings: d,
    };
  };
  const deformCase = (over: Partial<Deform>) => flat(async () => {
    const g = deform(over);
    return ops.deformConv2d(g.input, g.offset, g.weight, g.bias,
                            g.settings.stride, g.settings.padding,
                            g.settings.dilation, g.mask);
  });

  out.set("ops::deform_conv2d", deformCase({}));
  // **v2 adds a learned weight per position**, not only a learned place.
  out.set("ops::deform_conv2d(mask, v2)", deformCase({ useMask: true }));
  // **Two kinds of group, and they may differ** — one offset field can steer several
  // groups of filters. The mask is indexed by the offset group too, which a fixture
  // with one group cannot show.
  out.set("ops::deform_conv2d(weight groups)", deformCase({ wgroups: 2 }));
  out.set("ops::deform_conv2d(offset groups)", deformCase({ ogroups: 2 }));
  out.set("ops::deform_conv2d(both groups, mask)",
    deformCase({ wgroups: 2, ogroups: 2, useMask: true }));
  // **The kernel's dilation is applied before the offset**, so a zero displacement
  // leaves an ordinary dilated convolution.
  out.set("ops::deform_conv2d(stride, padding, dilation)",
    deformCase({ h: 9, w: 9, stride: [2, 2], padding: [1, 1], dilation: [2, 2] }));
  out.set("ops::deform_conv2d(1x1 kernel, mask)",
    deformCase({ kh: 1, kw: 1, h: 5, w: 5, useMask: true }));
  out.set("ops::deform_conv2d(non-square kernel)",
    deformCase({ kw: 1, h: 6, w: 6 }));

  // ── the structured dropouts, at the settings that draw nothing ──────────────
  //
  // **What the coin does has no shared answer to compare against**, so only the two
  // ways each of these declines to draw are asked: `training = false`, and a rate of
  // zero. That is a fact about the comparison and not about the functions — the
  // drawing halves are written and only these branches can be frozen.
  // **Two ramps, because the case table has two.** `_passthrough` builds
  // `(i·0.017) % 1.7 − 0.8` and `_layer_values` builds `(i·0.013) % 1.9 − 0.9`, and the
  // layers below go through the second. Written with one, `Permute` and
  // `DropBlock2d(eval)` came back off by a tenth at every position — a difference small
  // enough to read as a tolerance and large enough not to be one.
  const layerRamp = (shape: number[]): Tensor => {
    const n = shape.reduce((a, b) => a * b, 1);
    const values: number[] = [];
    const f = Math.fround;
    for (let i = 0; i < n; i++) {
      values.push(f(f(f(i * f(0.013)) % f(1.9)) - f(0.9)));
    }
    return Tensor.from(values, shape);
  };
  const ramp = (shape: number[]): Tensor => {
    const n = shape.reduce((a, b) => a * b, 1);
    const values: number[] = [];
    // **Rounded to f32 at every step, because numpy is.** `(i * 0.017) % 1.7` in
    // float64 and then rounded gives a different number wherever the product lands on
    // a multiple of the modulus: at `i = 500` the product is 8.5, which is five 1.7s
    // exactly in double and a hair under five in f32 — so one side answered −0.8 and
    // the other 0.9. One case in 750 values, and only that one.
    const f = Math.fround;
    for (let i = 0; i < n; i++) {
      values.push(f(f(f(i * f(0.017)) % f(1.7)) - f(0.8)));
    }
    return Tensor.from(values, shape);
  };
  out.set("ops::stochastic_depth(training=False)",
    flat(async () => ops.stochasticDepth(ramp([2, 3, 4, 5]), 0.5, "row", false)));
  out.set("ops::stochastic_depth(p=0)",
    flat(async () => ops.stochasticDepth(ramp([2, 3, 4, 5]), 0, "batch")));
  out.set("ops::drop_block2d(training=False)",
    flat(async () => ops.dropBlock2d(ramp([2, 3, 5, 5]), 0.3, 3, false, 1e-6, false)));
  out.set("ops::drop_block3d(training=False)",
    flat(async () => ops.dropBlock3d(ramp([2, 3, 5, 5, 5]), 0.3, 3, false, 1e-6, false)));

  // ── the ops that are layers ─────────────────────────────────────────────────
  //
  // **A layer's repr is an answer with the same standing as a value.** These hold
  // settings and no weights, so nothing has to be written into them first — which is
  // the half of the `ops::` ledger row that its one sentence about a harness did not
  // describe.
  const layerReprs: [string, () => { describe(): string }][] = [
    ["RoIAlign(2, 1.0, 2, aligned)", () => new ops.RoIAlign(2, 1.0, 2, true)],
    ["RoIPool(2, 0.5)", () => new ops.RoIPool(2, 0.5)],
    ["PSRoIAlign(2, 1.0, 2)", () => new ops.PSRoIAlign(2, 1.0, 2)],
    ["PSRoIPool(2, 1.0)", () => new ops.PSRoIPool(2, 1.0)],
    ["DropBlock2d(0.3, 3)", () => new ops.DropBlock2d(0.3, 3)],
    ["DropBlock3d(0.2, 5, inplace)", () => new ops.DropBlock3d(0.2, 5, true)],
    ["StochasticDepth(0.5, row)", () => new ops.StochasticDepth(0.5, "row")],
  ];
  for (const [name, make] of layerReprs) {
    out.set(`ops::${name}=repr`, () => make().describe());
  }

  // **The modules take the same arguments the functions were asked with**, which is
  // `(2, 1.0, 2)` rather than the function cases' `[3, 2]` — a layer built with the
  // function's settings would be a second copy of the function case.
  out.set("ops::RoIAlign(aligned)",
    flat(() => new ops.RoIAlign(2, 1.0, 2, true).forward(featureMap(), rois())));
  out.set("ops::RoIPool",
    flat(() => new ops.RoIPool(2, 1.0).forward(featureMap(), rois())));
  out.set("ops::PSRoIAlign",
    flat(() => new ops.PSRoIAlign(2, 1.0, 2).forward(psMap(), psRois())));
  out.set("ops::PSRoIPool",
    flat(() => new ops.PSRoIPool(2, 1.0).forward(psMap(), psRois())));
  out.set("ops::Permute([0, 2, 3, 1])",
    flat(async () => new ops.Permute([0, 2, 3, 1]).call(layerRamp([2, 3, 4, 5]))));
  // **`eval()` is the whole case.** In training mode this draws, and a draw has no
  // shared answer; the layer is here to be asked whether it stops.
  out.set("ops::DropBlock2d(eval)", flat(async () => {
    const layer = new ops.DropBlock2d(0.3, 3);
    layer.training = false;
    return layer.call(layerRamp([2, 3, 5, 5]));
  }));

  // ── the pyramid's level-picker ──────────────────────────────────────────────
  //
  // **The three boxes are 10, 200 and 56 pixels a side against a 64-pixel image**, so
  // they land on three different levels. A fixture whose boxes were all one size would
  // pass against a reader that always took the first map.
  const fpnShapes: number[][] = [[1, 4, 16, 16], [1, 6, 8, 8], [1, 8, 4, 4]];
  const fpnInput = (widths?: number): Map<string, Tensor> => {
    const out = new Map<string, Tensor>();
    fpnShapes.forEach((shape, i) => {
      const s = widths === undefined ? shape : [1, widths, ...shape.slice(2)];
      const n = s.reduce((a, b) => a * b, 1);
      const values: number[] = [];
      for (let k = 0; k < n; k++) values.push(Math.fround((k % 11) * 0.2));
      out.set(`feat${i}`, Tensor.from(values, s));
    });
    return out;
  };
  const multiscale = (samplingRatio: number) => flat(async () => {
    const model = new ops.MultiScaleRoIAlign(
      ["feat0", "feat1", "feat2"], 3, samplingRatio);
    const box = Tensor.from(
      [0, 0, 10, 10, 0, 0, 200, 200, 4, 4, 60, 60], [3, 4]);
    return model.forward(fpnInput(5), [box], [[64, 64]]);
  });
  out.set("ops::MultiScaleRoIAlign(a level per box)", multiscale(2));
  out.set("ops::MultiScaleRoIAlign(sampling_ratio=-1)", multiscale(-1));
  out.set("ops::MultiScaleRoIAlign(names, 3, 2)=repr",
    () => new ops.MultiScaleRoIAlign(["feat0", "feat1"], 3, 2).describe());

  // **These two hold weights and the repr does not read them** — only the shapes they
  // were built from. So the reprs cross before the values do.
  out.set("ops::FrozenBatchNorm2d(3)=repr",
    () => new ops.FrozenBatchNorm2d(3).describe());
  out.set("ops::DeformConv2d(4, 6, 3, padding=1)=repr",
    () => new ops.DeformConv2d(4, 6, 3, 1, 1).describe());
  out.set("ops::DeformConv2d(groups, no bias)=repr",
    () => new ops.DeformConv2d(4, 6, 3, 1, 0, 2, 2, false).describe());

  // ── the blocks, with the weights written rather than drawn ──────────────────
  //
  // **This is the harness the `ops::` ledger row has been waiting for.** The two
  // libraries initialise from different generators, so a layer compared as built
  // compares two draws; every parameter and buffer is written here from one ramp
  // instead, in sorted name order so that both sides give the same slot the same turn.
  //
  // `running_var` is made positive because a variance is, and **one channel's is very
  // nearly zero** — the only place epsilon shows. With every variance comfortably
  // positive, `(v + eps).rsqrt()` and `v.rsqrt()` agree to five decimals.
  interface Weighted {
    namedParameters(prefix?: string): Record<string, Tensor>;
    namedBuffers(persistent?: boolean, prefix?: string): Record<string, Tensor>;
  }
  const fill = <T extends Weighted>(module: T): T => {
    const seen: Record<string, Tensor> = {
      ...module.namedParameters(), ...module.namedBuffers(),
    };
    const f = Math.fround;
    Object.keys(seen).sort().forEach((name, turn) => {
      if (name.includes("num_batches")) return;
      const target = seen[name] as Tensor;
      const n = target.size;
      const values: number[] = [];
      for (let i = 0; i < n; i++) {
        values.push(f(f(f(f(i * f(0.037)) + f(turn * 0.11)) % f(1.7)) - f(0.6)));
      }
      if (name.includes("running_var")) {
        for (let i = 0; i < n; i++) values[i] = f(Math.abs(values[i] ?? 0) + f(0.5));
        values[0] = f(1e-9);
      }
      noGrad(() => target.copyFrom(Tensor.from(values, [...target.shape])));
    });
    return module;
  };
  const blockCase = (make: () => nn.Module, shape: number[]) => flat(async () => {
    const layer = fill(make());
    layer.eval();
    return noGrad(() => layer.call(layerRamp(shape)));
  });

  out.set("ops::Conv2dNormActivation(3→4, k3)",
    blockCase(() => new ops.Conv2dNormActivation(3, 4), [2, 3, 6, 7]));
  // **`convLayer` is the whole difference**, which is what that last seat is for. The
  // norm goes with it — a 2-D norm refuses five axes.
  out.set("ops::Conv3dNormActivation(2→3, the last seat)",
    blockCase(() => new ops.Conv3dNormActivation(2, 3), [2, 2, 4, 5, 6]));
  // **`bias = null` means "only when there is no norm"** — with the norm dropped the
  // convolution grows a bias, and the parameter list changes shape.
  out.set("ops::Conv2dNormActivation(k5, dilation 2, no norm)",
    blockCase(() => new ops.Conv2dNormActivation(3, 4, 5, 1, null, 1, null,
                                                 () => new nn.ReLU(), 2),
              [2, 3, 9, 9]));
  out.set("ops::SqueezeExcitation(6, 2)",
    blockCase(() => new ops.SqueezeExcitation(6, 2), [2, 6, 5, 4]));
  out.set("ops::MLP(4 → [6, 3])",
    blockCase(() => new ops.MLP(4, [6, 3]), [5, 4]));
  out.set("ops::MLP(with a norm between)",
    blockCase(() => new ops.MLP(4, [6, 3], (w) => new nn.BatchNorm1d(w)), [5, 4]));
  out.set("ops::FrozenBatchNorm2d(3)",
    blockCase(() => new ops.FrozenBatchNorm2d(3), [2, 3, 4, 5]));

  // **The count goes in front of the values.** The pyramid answers a named set, and a
  // reader that dropped one map would otherwise agree on everything it did return.
  out.set("ops::FeaturePyramidNetwork(three widths, three sizes)", flat(async () => {
    const model = fill(new ops.FeaturePyramidNetwork([4, 6, 8], 5));
    model.eval();
    const got = noGrad(() => model.forwardMaps(fpnInput()));
    const parts = [Tensor.from([got.size], [1])];
    for (const value of got.values()) parts.push(value.reshape([-1]));
    return Tensor.cat(parts, 0);
  }));

  // **The offsets still come from outside.** That is the shape of the layer: it holds a
  // weight and takes a displacement field, because the field is produced by another
  // convolution the caller writes.
  out.set("ops::DeformConv2d(4, 6, 3, padding=1)", flat(async () => {
    const layer = fill(new ops.DeformConv2d(4, 6, 3, 1, 1));
    const g = deform({ seed: 11, batch: 2, inC: 4, outC: 6, kh: 3, kw: 3, h: 6, w: 6,
                       padding: [1, 1] });
    return layer.forward(g.input, g.offset);
  }));

  // ── the frame axis, which is the only thing here that is about video ────────
  //
  // Thirty-seven names were declined on the Python side for *there is no video anywhere
  // in this project*. In torchvision thirty-three of them are one line delegating to
  // the image kernel — no container, no codec — and they stay out there for a different
  // reason: that file's image kernels are v1's and take an `(H, W, C)` array.
  //
  // These two touch no picture kernel at all. **A linspace, not a stride**: seven
  // frames sampled to three takes 0, 3, 6, where `T / n` takes 0, 2, 4 and never
  // reaches the end. The clip's frame count is not a multiple of the sample count,
  // which is the only arrangement where the two part.
  const clip = () => {
    const values: number[] = [];
    const f = Math.fround;
    for (let i = 0; i < 2 * 7 * 3 * 4 * 5; i++) values.push(f(i * f(0.05)));
    return Tensor.from(values, [2, 7, 3, 4, 5]);
  };
  out.set("v2f::uniform_temporal_subsample(linspace, not a stride)",
    flat(async () => ops.uniformTemporalSubsample(clip(), 3)));
  out.set("v2f::get_num_frames(the fourth axis from the end)",
    () => String(ops.getNumFrames(clip())));
  // **The transform around the function.** The arithmetic above was here and the v2
  // class was not, so the name axis reported it missing — and was right to, because
  // that axis refuses to fold a capital-initial name onto a lowercase one:
  // `nn.Embedding` is not `F.embedding`, and a transform is not its function. A
  // pipeline is built out of transforms, so `Compose([…, UniformTemporalSubsample(8)])`
  // was the line that could not be written.
  out.set("v2::UniformTemporalSubsample(the transform around it)",
    flat(async () => new v2twins.UniformTemporalSubsample(3).apply(clip()) as Tensor));

  // ── two whose Python rows named what they are for ───────────────────────────
  //
  // `narrow_copy`'s row explained why **torch** has it — sparse tensors have no
  // view-narrow — which is true and is not why it was absent. On a dense tensor it is a
  // narrow and a copy, and **the copy is the whole of the difference**: writing into a
  // narrow reaches the original and writing into this does not. A fixture that never
  // writes cannot tell them apart, so this one writes and then reads the input.
  out.set("top::narrow_copy(writing into it leaves the input alone)", async () => {
    const base = Tensor.from(
      Array.from({ length: 6 }, (_, i) => i), [2, 3]);
    const taken = narrowCopy(base, 1, 1, 2);
    taken.copyFrom(Tensor.zeros([2, 2]));
    const a = Array.from(await taken.toArray()).map((v) => String(Math.trunc(v)));
    const b = Array.from(await base.toArray()).map((v) => String(Math.trunc(v)));
    return `${a.join(" ")} | ${b.join(" ")}`;
  });

  // The same thing spelled as a method — torch keeps both, and the two are separate
  // seats: the function went in first and the method was absent for a while.
  out.set("top::narrow_copy(as a method)", async () => {
    const base = Tensor.from(Array.from({ length: 6 }, (_, k) => k), [2, 3]);
    return Array.from(await base.narrowCopy(1, 1, 2).toArray())
      .map((v) => String(Math.trunc(v))).join(" ");
  });

  // `segment_reduce`'s row read *for sparse and ragged bundles*, and the ragged half is
  // the part that is here: a dense tensor and a list of run lengths.
  //
  // **`lengths` and `offsets` are two spellings of one thing** and both are asked — a
  // reader that took one for the other shifts every boundary by the first run's length,
  // which on runs of equal size is invisible. So the uneven table is here beside the even
  // one. The library's `segmentReduce` is what answers; a helper written in this file
  // would have proved the helper.
  const SEG = (): Tensor => Tensor.from([1, 5, 3, 4], [4]);
  for (const [tag, kind, bounds] of [
    ["sum", "sum", { lengths: [2, 2] }],
    ["mean, uneven runs", "mean", { lengths: [1, 3] }],
    ["max", "max", { lengths: [2, 2] }],
    ["min", "min", { lengths: [1, 3] }],
    ["prod", "prod", { lengths: [2, 2] }],
    ["offsets say the same thing", "sum", { offsets: [0, 1, 4] }],
    ["initial seeds every run", "sum", { lengths: [2, 2], initial: 10 }],
  ] as const) {
    out.set(`top::segment_reduce(${tag})`,
      () => segmentReduce(SEG(), kind, bounds));
  }
  // A run table per row, where the boundaries differ between the rows.
  out.set("top::segment_reduce(a run table per row)", () => segmentReduce(
    Tensor.from(Array.from({ length: 8 }, (_, i) => i), [2, 4]),
    "sum", { lengths: [[2, 2], [1, 3]], axis: 1 }));

  // **Python spells a boolean `False` and JavaScript spells it `false`.** The golden was
  // frozen from real torch, so its answer carries Python's lettering; rendered here the
  // JavaScript way, the case would compare two spellings and never the two answers.
  const asPython = (b: boolean): string => (b ? "True" : "False");
  out.set("top::is_vulkan_available", () => asPython(isVulkanAvailable()));
  out.set("top::cudnn_is_acceptable", () => asPython(cudnnIsAcceptable(SEG())));

  // **The ten autocast questions, and this side is what their reason is about.**
  // Mixed precision is declined because *our shaders use f32 only*, and the shaders
  // are here. Four answer false; `isAutocastCacheEnabled` answers **true**, which is
  // why the five are asked one at a time rather than folded into one row.
  for (const [name, fn] of [
    ["is_autocast_enabled", isAutocastEnabled],
    ["is_autocast_cpu_enabled", isAutocastCpuEnabled],
    ["is_autocast_ipu_enabled", isAutocastIpuEnabled],
    ["is_autocast_xla_enabled", isAutocastXlaEnabled],
    ["is_autocast_cache_enabled", isAutocastCacheEnabled],
  ] as [string, () => boolean][]) {
    out.set(`top::${name}`, () => asPython(fn()));
  }

  // The getters answer with a **name**, not a dtype this side has — `DType` is the
  // four that exist and half precision is deliberately not among them. Python spells
  // the same answer `torch.bfloat16`, which is an `_AbsentDtype` there: a name kept so
  // that using it says what is missing. So the case carries torch's spelling.
  const asTorchDtype = (name: string): string => `torch.${name}`;
  for (const [name, fn] of [
    ["get_autocast_cpu_dtype", getAutocastCpuDtype],
    ["get_autocast_gpu_dtype", getAutocastGpuDtype],
    ["get_autocast_ipu_dtype", getAutocastIpuDtype],
    ["get_autocast_xla_dtype", getAutocastXlaDtype],
  ] as [string, () => string][]) {
    out.set(`top::${name}`, () => asTorchDtype(fn()));
  }
  for (const dev of ["cpu", "cuda", "xla", "ipu"]) {
    out.set(`top::get_autocast_dtype(${dev})`,
      () => asTorchDtype(getAutocastDtype(dev)));
  }
  out.set("top::get_autocast_dtype(모르는 장치)=거절", () => {
    try {
      getAutocastDtype("nonsense");
    } catch (e) {
      // The Python case answers with the exception's class name, and torch's is a
      // `RuntimeError` — which borch.ts raises under that name too.
      return (e as Error).name;
    }
    return "받았다";
  });

  // **The seven switch questions**, the same split as autocast above: three get/set
  // pairs declined whole for what the switch is for, and the reading half answers
  // anywhere. `isAnomalyCheckNanEnabled` is the `true` among them.
  for (const [name, fn] of [
    ["are_deterministic_algorithms_enabled", areDeterministicAlgorithmsEnabled],
    ["is_deterministic_algorithms_warn_only_enabled",
      isDeterministicAlgorithmsWarnOnlyEnabled],
    ["is_anomaly_enabled", isAnomalyEnabled],
    ["is_anomaly_check_nan_enabled", isAnomalyCheckNanEnabled],
    ["is_warn_always_enabled", isWarnAlwaysEnabled],
  ] as [string, () => boolean][]) {
    out.set(`top::${name}`, () => asPython(fn()));
  }
  out.set("top::get_deterministic_debug_mode",
    () => String(getDeterministicDebugMode()));
  // `"highest"` is full float32 and no TF32 — what these shaders do, rather than a
  // stand-in for a switch that is missing.
  out.set("top::get_float32_matmul_precision", () => getFloat32MatmulPrecision());

  // **The two thread counts, asked for the part that holds on both sides.** The value
  // itself parts from torch — 1 here against the machine's core count there — so it is
  // pinned in `parity.ts` beside `nn.Parameter`, not frozen here where it would pass on
  // one laptop and fail on the next. `int` is Python's spelling for what JS calls a
  // number, so the case carries Python's word.
  for (const [name, fn] of [
    ["get_num_threads", getNumThreads],
    ["get_num_interop_threads", getNumInteropThreads],
  ] as [string, () => number][]) {
    out.set(`top::${name}(양수 정수인가)`, () => `int ${verdict(fn() >= 1)}`);
  }

  out.set("ops::box_area", () => ops.boxArea(boxes()));
  // The same boxes read three ways. **`fmt` is a claim about four numbers that look
  // identical either way**, so a wrong one is a wrong answer with nothing raised — and
  // the round trip is what pins the pair of conversions to each other.
  out.set("ops::box_convert(xyxy to xywh)", () => ops.boxConvert(boxes(), "xyxy", "xywh"));
  out.set("ops::box_convert(xyxy to cxcywh)",
    () => ops.boxConvert(boxes(), "xyxy", "cxcywh"));
  out.set("ops::box_convert(cxcywh back to xyxy)", async () =>
    ops.boxConvert(await ops.boxConvert(boxes(), "xyxy", "cxcywh"), "cxcywh", "xyxy"));
  out.set("ops::box_area(cxcywh)", async () =>
    ops.boxArea(await ops.boxConvert(boxes(), "xyxy", "cxcywh"), "cxcywh"));
  // **N by M and not a paired list.** Five boxes against three gives fifteen numbers,
  // and an implementation that pairs them off returns three.
  out.set("ops::box_iou", () => ops.boxIou(boxes(), others()));
  // The three penalised IoUs. They agree with plain IoU wherever the boxes overlap and
  // part from it where they do not, which is why `others` holds one box that misses
  // everything — without it all four functions would return the same table.
  out.set("ops::generalized_box_iou", () => ops.generalizedBoxIou(boxes(), others()));
  out.set("ops::distance_box_iou", () => ops.distanceBoxIou(boxes(), others()));
  out.set("ops::complete_box_iou", () => ops.completeBoxIou(boxes(), others()));
  // **The losses are paired, where the IoUs above are N by M.** Five boxes against five,
  // one answer each — an implementation that reached for the matrix and took its diagonal
  // would agree here and compute `N²` to keep `N`, so the shape is the case as much as
  // the values are. (Written that way here first, and corrected before it shipped.)
  //
  // `shapes` is deliberately not `boxes`: those pairs have matched aspect ratios, and
  // `completeBoxIouLoss`'s extra term is exactly zero whenever they do. On these it
  // differs from the distance loss by 0.217.
  out.set("ops::generalized_box_iou_loss",
    () => ops.generalizedBoxIouLoss(shapes(), targets()));
  out.set("ops::distance_box_iou_loss", () => ops.distanceBoxIouLoss(shapes(), targets()));
  out.set("ops::complete_box_iou_loss", () => ops.completeBoxIouLoss(shapes(), targets()));
  out.set("ops::generalized_box_iou_loss(mean)",
    () => ops.generalizedBoxIouLoss(shapes(), targets(), "mean"));
  out.set("ops::complete_box_iou_loss(sum)",
    () => ops.completeBoxIouLoss(shapes(), targets(), "sum"));
  // Focal loss takes **logits**, so the values run either side of zero and one pair sits
  // at ±40 — past where a plain sigmoid or a plain `log(1 + exp(-x))` stops being finite
  // while the loss is not.
  out.set("ops::sigmoid_focal_loss", () => ops.sigmoidFocalLoss(logits(), hits()));
  // **`alpha = -1` is a switch, not a weight** — it turns the class balancing off.
  out.set("ops::sigmoid_focal_loss(alpha off)",
    () => ops.sigmoidFocalLoss(logits(), hits(), -1));
  // `gamma = 0` removes the focusing term, leaving the weighted cross-entropy this whole
  // loss is a modulation of.
  out.set("ops::sigmoid_focal_loss(gamma 0)",
    () => ops.sigmoidFocalLoss(logits(), hits(), 0.25, 0));
  out.set("ops::sigmoid_focal_loss(mean)",
    () => ops.sigmoidFocalLoss(logits(), hits(), 0.25, 2, "mean"));
  out.set("ops::clip_boxes_to_image", () => ops.clipBoxesToImage(boxes(), [20, 25]));
  out.set("ops::remove_small_boxes", () => ops.removeSmallBoxes(boxes(), 10.5));
  // **`> threshold` and not `>=`.** At zero, boxes that merely touch both survive; the
  // duplicate does not. Both ends of the range are asked.
  out.set("ops::nms(nothing may overlap)", () => ops.nms(boxes(), scores(), 0.0));
  out.set("ops::nms(half)", () => ops.nms(boxes(), scores(), 0.5));
  out.set("ops::nms(everything survives)", () => ops.nms(boxes(), scores(), 1.0));
  // Per class, by moving each class out of the others' reach.
  //
  // **The first row does not test the offset.** Measured, by deleting the shift from
  // `batchedNms` and running these sixteen: it stayed green. A duplicate is suppressed
  // whether or not the classes were separated first, and with these labels no pair in
  // *different* classes overlaps past the threshold — the largest such is 0.22 against
  // 0.5. It is kept because the frozen value is worth keeping; the row after it is what
  // asks the question.
  out.set("ops::batched_nms", () => ops.batchedNms(boxes(), scores(), labels(), 0.5));
  // Box 1 moved to the other class. It overlaps box 0 at 0.68, so one pass over
  // unshifted boxes drops it and the per-class answer keeps it — the two spellings
  // differ in **length**, which no tolerance can absorb.
  out.set("ops::batched_nms(classes that would suppress each other)", () =>
    ops.batchedNms(boxes(), scores(), Tensor.from([0, 1, 1, 1, 0]), 0.5));
  // A stack with **one blank plane — and the blank is the case**: torchvision answers
  // zeros rather than raising, which is what lets a batch carrying an empty annotation
  // still stack.
  out.set("ops::masks_to_boxes", () => {
    const masks = new Float32Array(3 * 6 * 8);
    for (let i = 1; i < 4; i++) for (let j = 2; j < 6; j++) masks[i * 8 + j] = 1;
    for (let i = 0; i < 2; i++) masks[6 * 8 + i * 8] = 1;
    return ops.masksToBoxes(Tensor.from(masks, [3, 6, 8]));
  });
}

/**
 * The elastic warp's displacement field, `(1, 5, 4, 2)` flattened.
 *
 * **Written out because it cannot be drawn twice.** The Python side builds it with
 * numpy's `default_rng(7)`, and PCG64 has no counterpart here — a field generated by a
 * different generator is a different input, and comparing two answers to two different
 * questions is not a comparison. So it is transcribed, which is the same choice the box
 * fixtures in `addOps` make for a different reason.
 */
/**
 * `Lambda`'s repr prints the function's name, so it has to have one — and **it has to
 * be the same name on both sides.**
 *
 * The golden froze `Lambda(_v2_named, …)` from the Python case's helper, so this one is
 * named to match rather than named well. That coupling is real and worth seeing: a case
 * comparing a repr is comparing a *string*, and a string can contain an identifier from
 * the other language's source.
 */
// eslint-disable-next-line @typescript-eslint/naming-convention
function _v2_named(x: vision.Subject): vision.Subject {
  return x;
}

const ELASTIC_SHIFT = new Float32Array([
  0.025019101798534393, 0.07944276183843613, 0.055137135088443756,
  -0.054958563297986984, -0.03996674343943596, 0.07471068948507309,
  -0.0989469438791275, 0.06424569338560104, 0.059413887560367584,
  -0.006413005292415619, -0.03939351439476013, -0.044314879924058914,
  -0.04902608320116997, -0.010984733700752258, 0.0009096488356590271,
  0.010699473321437836, 0.09910006076097488, 0.05853237956762314,
  0.02443584054708481, 0.09779203683137894, -0.05693826079368591,
  -0.06795759499073029, 0.0225079208612442, -0.09121160209178925,
  -0.09286394715309143, 0.0029777660965919495, -0.006758794188499451,
  0.08343356102705002, 0.025845251977443695, 0.0028235316276550293,
  -0.000625312328338623, -0.05049701780080795, -0.09764119982719421,
  -0.061519574373960495, 0.03840642422437668, -0.05987865850329399,
  -0.026092737913131714, -0.09925315529108047, 0.06600954383611679,
  -0.069107785820961,
]);

/**
 * `v2f::` — `transforms.v2.functional`.
 *
 * **Nine names v2 adds, and eight of v1's reached through the v2 spelling.** The second
 * group is the interesting one: those eight are re-exports, and a re-export cannot be
 * got wrong in a way a value case sees — except by quietly becoming a second
 * implementation, which is exactly what these catch, because they are held against v1's
 * own frozen answers.
 */
/**
 * `dataset::` — **the decoders, asked without a network.**
 *
 * A dataset is an address and a format. The address half cannot be a golden case: a
 * case that downloads is a case that fails on a train, and freezing MNIST's answer
 * means shipping MNIST. The format half is the part that goes wrong quietly, and the
 * bytes for it are **built here** so that both sides read the same file.
 */
function addDatasets(out: Map<string, Case>): void {
  /**
   * An IDX file. **The header is the case**: two zero bytes, a type code, the number of
   * axes, then one big-endian length each.
   */
  const idx = (kind: number, shape: readonly number[],
               payload: Uint8Array): Uint8Array => {
    const head = new Uint8Array(4 + shape.length * 4);
    head[2] = kind;
    head[3] = shape.length;
    const view = new DataView(head.buffer);
    shape.forEach((n, i) => view.setUint32(4 + i * 4, n, false));
    return new Uint8Array([...head, ...payload]);
  };
  /** Big-endian int32 values, which is what a QMNIST label table holds. */
  const be32 = (values: readonly number[]): Uint8Array => {
    const bytes = new Uint8Array(values.length * 4);
    const view = new DataView(bytes.buffer);
    values.forEach((v, i) => view.setInt32(i * 4, v, false));
    return bytes;
  };

  // Pictures: two 3×4 frames of bytes counting up, so a transposed or mis-strided read
  // lands somewhere visibly different.
  const pixels = new Uint8Array(Array.from({ length: 24 }, (_, i) => i));
  // Labels: ten of them, including 0 and 255 — the ends are where a signed read shows.
  const labels = new Uint8Array([0, 1, 2, 9, 200, 255, 3, 4, 5, 6]);

  out.set("dataset::IDX images",
    () => datasets.readIdxImages(idx(8, [2, 3, 4], pixels)));
  out.set("dataset::IDX labels",
    () => datasets.readIdxLabels(idx(8, [10], labels)));
  out.set("dataset::IDX images(one frame)",
    () => datasets.readIdxImages(idx(8, [1, 4, 6], pixels)));
  // **A header that promises more than the file carries.** Both sides refuse, and what
  // is frozen is that they refuse the same way — torch's words are `shape '[12]' is
  // invalid for input of size 10`, which is the phrase this searches for.
  out.set("dataset::IDX labels(short by two)=거절", async () => {
    try {
      datasets.readIdxLabels(idx(8, [12], labels));
    } catch (err) {
      const said = err instanceof Error ? err.message : String(err);
      return `거절|문구=${said.includes("invalid for input of size 10") ? "True" : "False"}`;
    }
    return "예외가 안 났다";
  });

  // Eight columns, three rows, and **values past what a byte holds** — 279260 is a real
  // QMNIST field. Read as bytes it becomes something else entirely, and read
  // little-endian it becomes something else again.
  const qmnist = [7, 4, 2578, 69, 37, 279260, 0, 0,
    2, 4, 2359, 55, 32, 253328, 0, 0,
    1, 4, 2530, 80, 31, 273542, 0, 0];
  out.set("dataset::IDX int32 table(QMNIST labels)",
    () => datasets.readIdxTensor(idx(12, [3, 8], be32(qmnist))));
  // A negative, because int32 is signed and the reader must not widen it wrong.
  out.set("dataset::IDX int32 table(negative)",
    () => datasets.readIdxTensor(
      idx(12, [2, 2], be32([-1, 2147483647, -2147483648, 0]))));

  // Two 4×4×3 pictures rather than 96×96 — the arrangement is the question and the size
  // is not. Values count up, so a transposed read lands somewhere visible.
  const stlPixels = new Uint8Array(Array.from({ length: 96 }, (_, i) => i));
  out.set("dataset::STL10 bytes(pictures)",
    () => datasets.readStl10Images(stlPixels));
  // **The same bytes without the swap**, frozen beside the one above so the difference
  // is a value in this table rather than a claim about one — the two agree on shape and
  // on every summary statistic.
  out.set("dataset::STL10 bytes(no transpose)",
    () => datasets.readStl10Images(stlPixels, 4, false));
  out.set("dataset::STL10 bytes(labels, 1-based)",
    () => datasets.readStl10Labels(new Uint8Array([1, 10, 5, 3])));

  // **`.data` holds `(N,C,H,W)` and an item is `(H,W,C)`.** They disagree on purpose:
  // torchvision transposes at that line because its other datasets hold HWC and this
  // one does not. The case exists because the door was wrong while the store was right.
  out.set("dataset::STL10 item(H,W,C)", () => {
    const raw = new Uint8Array(Array.from({ length: 2 * 3 * 96 * 96 }, (_, i) => i % 251));
    const pictures = datasets.readStl10Images(raw, 96);
    const target = datasets.readStl10Labels(new Uint8Array([1, 10]));
    // Item 1, moved to (H, W, C), with its label after it.
    const item = pictures.select(0, 1).permute([1, 2, 0]);
    return Tensor.cat([item.reshape([96 * 96 * 3]), target.select(0, 1).reshape([1])], 0);
  });

  /** A `.npy` holding `(frames, clips, H, W)` of bytes counting up. */
  const npy = (shape: readonly number[]): Uint8Array => {
    const header = `{'descr': '|u1', 'fortran_order': False, 'shape': (${
      shape.join(", ")},), }`;
    const padded = header + " ".repeat((64 - (10 + header.length + 1) % 64) % 64) + "\n";
    const n = shape.reduce((a, b) => a * b, 1);
    const out2 = new Uint8Array(10 + padded.length + n);
    out2.set([0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59, 1, 0], 0);
    new DataView(out2.buffer).setUint16(8, padded.length, true);
    for (let i = 0; i < padded.length; i++) out2[10 + i] = padded.charCodeAt(i);
    for (let i = 0; i < n; i++) out2[10 + padded.length + i] = i;
    return out2;
  };
  const clip = () => npy([6, 2, 4, 4]);
  out.set("dataset::MovingMNIST(whole)", () => datasets.readMovingMnist(clip(), null, 3));
  out.set("dataset::MovingMNIST(train)",
    () => datasets.readMovingMnist(clip(), "train", 3));
  out.set("dataset::MovingMNIST(test)",
    () => datasets.readMovingMnist(clip(), "test", 3));

  /**
   * FER2013's CSV, in both of its layouts. **The ICML one puts a leading space in the
   * column names**, and a reader written against the other file finds out with a
   * missing column rather than a wrong number.
   */
  const fer = (icml: boolean): string => {
    const space = icml ? " " : "";
    const rows = [`emotion,${space}pixels,${space}Usage`];
    const usages: readonly [number, string][] = [
      [3, "Training"], [0, "Training"], [6, "PublicTest"], [2, "PrivateTest"]];
    usages.forEach(([emotion, usage], i) => {
      const cells = Array.from({ length: 48 * 48 }, (_, k) => (i * 13 + k) % 256);
      rows.push(`${emotion},${cells.join(" ")},${usage}`);
    });
    return `${rows.join("\n")}\n`;
  };
  const ferTensor = (icml: boolean, split: "train" | "test") => () => {
    const rows = datasets.readFer2013(fer(icml), split);
    const flat: number[] = [];
    for (const row of rows) flat.push(...row.pixels, row.emotion);
    return Tensor.from(flat, [rows.length, 48 * 48 + 1]);
  };
  out.set("dataset::FER2013(train)", ferTensor(false, "train"));
  // **`Usage` decides, and test is two values** — `PublicTest` and `PrivateTest` both
  // land there. A reader matching only the first returns half the set and no error.
  out.set("dataset::FER2013(test, two usages)", ferTensor(false, "test"));
  out.set("dataset::FER2013(icml, spaced headers)", ferTensor(true, "train"));
}

function addV2Functional(out: Map<string, Case>, inp: Inputs): void {
  const pic = (name: string, isByte: boolean): vision.Image => {
    const shape = inp.shapeOf(name);
    const [h = 1, w = 1, c = 1] = shape;
    return vision.image(inp.raw(name), h, w, shape.length === 2 ? 1 : c, isByte);
  };
  const f = () => pic("vis_f", false);
  const u8 = () => pic("vis_u8", true);
  const asTensor = (img: vision.Image): Tensor =>
    Tensor.from(img.data, [img.height, img.width, img.channels]);
  // The first channel alone, which is the Python side's `img_f[:, :, :1]`.
  const grey = (): vision.Image => {
    const src = f();
    const one = new Float64Array(src.height * src.width);
    for (let i = 0; i < one.length; i++) one[i] = src.data[i * src.channels] ?? 0;
    return vision.image(one, src.height, src.width, 1, false);
  };
  const set = (name: string, make: () => vision.Image): void => {
    out.set(`v2f::${name}`, () => asTensor(make()));
  };

  // ── the bounding-box kernels ───────────────────────────────────────────────
  //
  // **Frozen as text**, because five of these answer `[boxes, canvasSize]` and two
  // answer boxes alone, and the shape is part of what is being checked. The Python side
  // folds the sign of zero for the same reason it does here: mirroring a box whose far
  // edge is the canvas edge gives `32 - 32`, torchvision spells that `-0.0`, and a text
  // freeze would make two spellings of one number look like a disagreement.
  const boxes = () => Tensor.from(
    [0, 0, 10, 10, 5, 5, 15, 20, 2, 3, 4, 9, 28, 20, 32, 24, 30, 22, 31, 23], [5, 4]);
  const CANVAS: [number, number] = [24, 32];
  const show = async (t: Tensor): Promise<string> => {
    const flat = Array.from(await t.toArray());
    const rows: string[] = [];
    for (let i = 0; i < flat.length; i += 4) {
      rows.push("[" + flat.slice(i, i + 4)
        .map((v) => {
          const r = Math.round((v === 0 ? 0 : v) * 1e3) / 1e3;
          return (r === 0 ? 0 : r).toString();
        })
        .map((v) => (v.includes(".") ? v : v + ".0"))
        .join(", ") + "]");
    }
    return "[" + rows.join(", ") + "]";
  };
  const pair = async (r: Promise<[Tensor, [number, number]]>): Promise<string> => {
    const [t, canvas] = await r;
    return `${await show(t)} ${canvas[0]} ${canvas[1]}`;
  };

  // ── masks and keypoints ───────────────────────────────────────────────────
  //
  // A label map with **three** classes, not two: with 0/1 alone a bilinear resample
  // rounds back to the right answer half the time, so nearest and not-nearest would
  // still agree. The dtype travels in the answer because a mask is labels — widening
  // `uint8` makes `=== 3` start failing on values that are exactly equal.
  const labels = () => {
    const v: number[] = [];
    for (let i = 0; i < 2 * 24 * 32; i++) v.push(i % 3);
    return Tensor.from(v, [2, 24, 32]);
  };
  // Three groups of two. **The last group has a point outside the canvas**, which is
  // what makes sanitize drop a group rather than a point.
  const pts = () => Tensor.from(
    [0, 0, 1, 2, 5, 5, 31, 23, 10, 10, 33, 26], [3, 2, 2]);
  // **`[shape] [values]`, flat, and the same flat the core writes.** The nested
  // spelling is presentation and the two sides reached it differently — `show` chunks
  // four to a row, which is right for `(N, 4)` boxes, wrong for a `(2, 24, 32)` mask and
  // wrong again for `(N, K, 2)` keypoints. Eighteen cases failed on that and not one was
  // a wrong number.
  const nums = (v: number[]): string => "[" + v.join(", ") + "]";
  const flatWhole = async (t: Tensor): Promise<string> =>
    nums(t.shape as number[]) + " " +
    nums(Array.from(await t.toArray()).map((v) => Math.round(v)));
  const flatReal = async (t: Tensor): Promise<string> => {
    const body = Array.from(await t.toArray()).map((v) => {
      const r = Math.round(v * 1e3) / 1e3;
      const z = r === 0 ? 0 : r;
      const text = z.toString();
      return text.includes(".") || text.includes("e") ? text : text + ".0";
    });
    return nums(t.shape as number[]) + " [" + body.join(", ") + "]";
  };
  const withDtype = async (t: Promise<Tensor>): Promise<string> =>
    (await flatWhole(await t)) + " uint8";
  const kpPair = async (r: Promise<[Tensor, [number, number]]>): Promise<string> => {
    const [t, canvas] = await r;
    return (await flatReal(t)) + " " + canvas[0] + " " + canvas[1];
  };

  out.set("v2f::horizontal_flip_mask", () => withDtype(ops.horizontalFlipMask(labels())));
  out.set("v2f::vertical_flip_mask", () => withDtype(ops.verticalFlipMask(labels())));
  out.set("v2f::crop_mask(inside)",
    () => withDtype(ops.cropMask(labels(), 1, 2, 12, 14)));
  out.set("v2f::crop_mask(off the corner)",
    () => withDtype(ops.cropMask(labels(), -3, -4, 10, 10)));
  out.set("v2f::center_crop_mask(odd)",
    () => withDtype(ops.centerCropMask(labels(), [11, 13])));
  out.set("v2f::pad_mask(four sides, fill=7)",
    () => withDtype(ops.padMask(labels(), [1, 2, 3, 4], 7)));
  out.set("v2f::resize_mask(pair)",
    () => withDtype(ops.resizeMask(labels(), [12, 15])));
  out.set("v2f::resize_mask(short edge)",
    () => withDtype(ops.resizeMask(labels(), [9])));
  out.set("v2f::resized_crop_mask",
    () => withDtype(ops.resizedCropMask(labels(), 1, 2, 12, 14, [6, 8])));

  out.set("v2f::horizontal_flip_keypoints",
    async () => flatReal(await ops.horizontalFlipKeypoints(pts(), [24, 32])));
  out.set("v2f::vertical_flip_keypoints",
    async () => flatReal(await ops.verticalFlipKeypoints(pts(), [24, 32])));
  out.set("v2f::crop_keypoints", () => kpPair(ops.cropKeypoints(pts(), 1, 2, 12, 14)));
  out.set("v2f::center_crop_keypoints(odd)",
    () => kpPair(ops.centerCropKeypoints(pts(), [24, 32], [11, 13])));
  out.set("v2f::pad_keypoints(four sides)",
    () => kpPair(ops.padKeypoints(pts(), [24, 32], [1, 2, 3, 4])));
  out.set("v2f::resize_keypoints(pair)",
    () => kpPair(ops.resizeKeypoints(pts(), [12, 16], [24, 32])));
  out.set("v2f::resized_crop_keypoints",
    () => kpPair(ops.resizedCropKeypoints(pts(), 1, 2, 12, 14, [6, 8])));
  out.set("v2f::clamp_keypoints",
    async () => flatReal(await ops.clampKeypoints(pts(), [24, 32])));
  out.set("v2f::sanitize_keypoints", async () => {
    const [kept, mask] = await ops.sanitizeKeypoints(pts(), [24, 32]);
    return (await flatReal(kept)) + " " + (await flatWhole(mask));
  });

  // ── the corner warps ───────────────────────────────────────────────────────
  //
  // **A rotated box grows** — the upright hull of a tilted rectangle is larger than the
  // rectangle. 90 degrees is asked because it is the one angle where the hull is the box
  // again, so it catches a transform right about extents and wrong about direction.
  //
  // `center: [0, 0]` is asked because it is **the corner, not the middle**: the default
  // centre is `[w/2, h/2]`, and an implementation that reuses the image grid's
  // convention answers this case for the default one.
  // **An integer formula and not a drawn field**, and the core builds it the same way.
  // No seeded generator matches across two runtimes — numpy's PCG64 and anything
  // JavaScript can write are different streams — so a formula both sides evaluate is
  // the same field by construction rather than by hope.
  const FIELD: number[] = (() => {
    const v: number[] = [];
    for (let i = 0; i < 24 * 32; i++) {
      v.push(((i % 17) - 8) / 100, ((i % 23) - 11) / 100);
    }
    return v;
  })();
  const SP = [[0, 0], [31, 0], [31, 23], [0, 23]];
  const EP = [[2, 1], [29, 2], [30, 22], [1, 21]];

  out.set("v2f::affine_bounding_boxes(30)", async () =>
    show(await ops.affineBoundingBoxes(boxes(), "xyxy", [24, 32], 30, [0, 0], 1, [0, 0])));
  out.set("v2f::affine_bounding_boxes(90)", async () =>
    show(await ops.affineBoundingBoxes(boxes(), "xyxy", [24, 32], 90, [0, 0], 1, [0, 0])));
  out.set("v2f::affine_bounding_boxes(all four)", async () =>
    show(await ops.affineBoundingBoxes(boxes(), "xyxy", [24, 32], -15, [3, -2], 1.3,
      [10, 5])));
  out.set("v2f::affine_bounding_boxes(center at the corner)", async () =>
    show(await ops.affineBoundingBoxes(boxes(), "xyxy", [24, 32], 30, [0, 0], 1, [0, 0],
      [0, 0])));
  out.set("v2f::rotate_bounding_boxes(30)",
    () => pair(ops.rotateBoundingBoxes(boxes(), "xyxy", [24, 32], 30)));
  out.set("v2f::rotate_bounding_boxes(30, expand)",
    () => pair(ops.rotateBoundingBoxes(boxes(), "xyxy", [24, 32], 30, true)));
  out.set("v2f::rotate_bounding_boxes(-47, expand)",
    () => pair(ops.rotateBoundingBoxes(boxes(), "xyxy", [24, 32], -47, true)));
  out.set("v2f::perspective_bounding_boxes", async () =>
    show(await ops.perspectiveBoundingBoxes(boxes(), "xyxy", [24, 32], SP, EP)));
  out.set("v2f::elastic_bounding_boxes", async () =>
    show(await ops.elasticBoundingBoxes(boxes(), "xyxy", [24, 32], FIELD, 24, 32)));

  out.set("v2f::affine_keypoints(30)",
    () => kpPair(ops.affineKeypoints(pts(), [24, 32], 30, [0, 0], 1, [0, 0])));
  out.set("v2f::affine_keypoints(all four)",
    () => kpPair(ops.affineKeypoints(pts(), [24, 32], -15, [3, -2], 1.3, [10, 5])));
  out.set("v2f::rotate_keypoints(30, expand)",
    () => kpPair(ops.rotateKeypoints(pts(), [24, 32], 30, true)));
  out.set("v2f::perspective_keypoints", async () =>
    flatReal(await ops.perspectiveKeypoints(pts(), [24, 32], SP, EP)));
  out.set("v2f::elastic_keypoints", async () =>
    flatReal(await ops.elasticKeypoints(pts(), [24, 32], FIELD, 24, 32)));

  out.set("v2f::affine_mask(30)",
    () => withDtype(ops.affineMask(labels(), 30, [0, 0], 1, [0, 0])));
  out.set("v2f::affine_mask(all four)",
    () => withDtype(ops.affineMask(labels(), -15, [3, -2], 1.3, [10, 5])));
  out.set("v2f::rotate_mask(30)", () => withDtype(ops.rotateMask(labels(), 30)));
  out.set("v2f::rotate_mask(30, expand)",
    () => withDtype(ops.rotateMask(labels(), 30, true)));
  out.set("v2f::perspective_mask",
    () => withDtype(ops.perspectiveMask(labels(), SP, EP)));
  out.set("v2f::elastic_mask", () => withDtype(ops.elasticMask(labels(), FIELD)));

  for (const fmt of ["xyxy", "xywh", "cxcywh"] as const) {
    const b = async () => ops.boxConvert(boxes(), "xyxy", fmt);
    out.set(`v2f::horizontal_flip_bounding_boxes(${fmt})`,
      async () => show(await ops.horizontalFlipBoundingBoxes(await b(), fmt, CANVAS)));
    out.set(`v2f::vertical_flip_bounding_boxes(${fmt})`,
      async () => show(await ops.verticalFlipBoundingBoxes(await b(), fmt, CANVAS)));
    out.set(`v2f::crop_bounding_boxes(${fmt})`,
      async () => pair(ops.cropBoundingBoxes(await b(), fmt, 1, 2, 12, 14)));
    out.set(`v2f::center_crop_bounding_boxes(odd, ${fmt})`,
      async () => pair(ops.centerCropBoundingBoxes(await b(), fmt, CANVAS, [11, 13])));
    out.set(`v2f::pad_bounding_boxes(four sides, ${fmt})`,
      async () => pair(ops.padBoundingBoxes(await b(), fmt, CANVAS, [1, 2, 3, 4])));
    out.set(`v2f::resize_bounding_boxes(pair, ${fmt})`,
      async () => pair(ops.resizeBoundingBoxes(await b(), CANVAS, [12, 16], undefined, fmt)));
    out.set(`v2f::resize_bounding_boxes(short edge, ${fmt})`,
      async () => pair(ops.resizeBoundingBoxes(await b(), CANVAS, [12], undefined, fmt)));
    out.set(`v2f::resized_crop_bounding_boxes(${fmt})`,
      async () => pair(ops.resizedCropBoundingBoxes(await b(), fmt, 1, 2, 12, 14, [6, 8])));
  }

  set("horizontal_flip", () => v2f.horizontalFlip(f()));
  set("vertical_flip", () => v2f.verticalFlip(f()));
  set("grayscale_to_rgb(one channel)", () => v2f.grayscaleToRgb(grey()));
  // Three channels **pass through** rather than raising, so a mixed pipeline needs no
  // branch — and a version that stacked them again would answer nine.
  set("grayscale_to_rgb(three channels)", () => v2f.grayscaleToRgb(f()));
  set("permute_channels", () => v2f.permuteChannels(f(), [2, 0, 1]));
  set("to_dtype(scaling)", () => v2f.toDtype(u8(), "float32", true));
  set("to_dtype(not scaling)", () => v2f.toDtype(u8(), "float32"));
  // **`sigma` is zero in both**, which is what makes them askable: every other setting
  // draws, and a drawn value has no frozen answer. So what is measured is the mean and
  // the clipping, not the noise.
  set("gaussian_noise(sigma=0)", () => v2f.gaussianNoise(f(), 0.0, 0.0));
  set("gaussian_noise(clipping)", () => v2f.gaussianNoise(f(), 5.0, 0.0, true));
  // **The displacement is written out, not drawn.** The Python side builds it with
  // numpy's PCG64, which has no counterpart here, and the same values on both sides is
  // what makes this a comparison at all. A field small enough to keep the picture
  // recognisable and large enough that every pixel moves — a zero field passes against
  // a warp that does nothing.
  set("elastic", () => v2f.elastic(f(), ELASTIC_SHIFT));

  // **The two that answer a size, side by side.** v2 reversed the pair on purpose and
  // the names are one namespace apart, so a reader who takes the wrong one gets a
  // picture that is transposed and still perfectly plausible. Frozen as text, because
  // `[5, 4]` and `[4, 5]` compare equal to nothing else.
  out.set("v2f::get_size(height first)", () => `[${v2f.getSize(f()).join(", ")}]`);
  out.set("v2f::get_image_size(width first)",
    () => `[${vision.getImageSize(f()).join(", ")}]`);
  out.set("v2f::get_num_channels", () => String(v2f.getNumChannels(f())));

  const v2set = (name: string, make: () => vision.Image): void => {
    out.set(`v2::${name}`, () => asTensor(make()));
  };

  // ── `v2::` — the transform classes ──
  //
  // **The repr is the larger half of that block and it is not cosmetic.** A tutorial's
  // `print(transform)` is how a reader checks that the thing they built is the thing
  // they meant, and on the Python side these came out wrong four times before they came
  // out right — every one found by comparing rather than by reading.
  //
  // Two rules do most of the work. A field of a kind v2 does not print **disappears
  // from the line** rather than printing as `None`, which is why `ToDtype` shows only
  // `scale` and not the dtype that is its main argument. And Python's float carries its
  // decimal point, so `mean=0.0` and not `mean=0`.
  // The arguments are chosen to make the repr **say something** — a default-everything
  // constructor prints the same text no matter what the constructor did with what it
  // was given.
  for (const [name, make] of [
    ["Identity", () => new v2f.Identity()],
    ["RGB", () => new v2f.RGB()],
    ["ToImage", () => new v2f.ToImage()],
    ["ToPureTensor", () => new v2f.ToPureTensor()],
    ["ToDtype", () => new v2f.ToDtype("float32", true)],
    ["GaussianNoise", () => new v2f.GaussianNoise()],
    ["GaussianNoise(three arguments)", () => new v2f.GaussianNoise(0.1, 0.5, false)],
    ["RandomChannelPermutation", () => new v2f.RandomChannelPermutation()],
    // The twins.
    ["Resize", () => new v2f.Resize([4, 3])],
    ["Resize(one number)", () => new v2f.Resize(5)],
    ["CenterCrop", () => new v2f.CenterCrop(4)],
    ["RandomCrop", () => new v2f.RandomCrop(4)],
    ["RandomResizedCrop", () => new v2f.RandomResizedCrop(4)],
    ["FiveCrop", () => new v2f.FiveCrop(3)],
    ["TenCrop", () => new v2f.TenCrop(3)],
    ["Pad", () => new v2f.Pad(2)],
    ["RandomHorizontalFlip", () => new v2f.RandomHorizontalFlip()],
    ["RandomVerticalFlip", () => new v2f.RandomVerticalFlip()],
    ["Grayscale", () => new v2f.Grayscale(3)],
    ["RandomGrayscale", () => new v2f.RandomGrayscale()],
    ["Normalize", () => new v2f.Normalize([0.5], [0.5])],
    ["RandomErasing", () => new v2f.RandomErasing()],
    ["ColorJitter", () => new v2f.ColorJitter(0.5)],
    ["ColorJitter(all four)", () => new v2f.ColorJitter(0.5, 0.3, 0.2, 0.1)],
    ["RandomInvert", () => new v2f.RandomInvert()],
    ["RandomPosterize", () => new v2f.RandomPosterize(4)],
    ["RandomSolarize", () => new v2f.RandomSolarize(0.5)],
    ["RandomAutocontrast", () => new v2f.RandomAutocontrast()],
    ["RandomEqualize", () => new v2f.RandomEqualize()],
    ["RandomAdjustSharpness", () => new v2f.RandomAdjustSharpness(2)],
    ["RandomRotation", () => new v2f.RandomRotation(30)],
    ["RandomAffine", () => new v2f.RandomAffine(30)],
    ["RandomPerspective", () => new v2f.RandomPerspective()],
    ["ElasticTransform", () => new v2f.ElasticTransform()],
    ["GaussianBlur", () => new v2f.GaussianBlur(3)],
    ["AutoAugment", () => new v2f.AutoAugment()],
    ["RandAugment", () => new v2f.RandAugment()],
    ["TrivialAugmentWide", () => new v2f.TrivialAugmentWide()],
    ["AugMix", () => new v2f.AugMix()],
    ["RandomOrder", () => new v2f.RandomOrder([new v2f.Identity(), new v2f.RGB()])],
    ["RandomChoice", () => new v2f.RandomChoice([new v2f.Identity(), new v2f.RGB()])],
    // **One child and two children print differently** — torch's `nn.Module` puts one
    // on the same line and breaks two across lines. Both spellings are asked.
    ["Compose(one)", () => new v2f.Compose([new v2f.Identity()])],
    ["Compose(two)", () => new v2f.Compose([new v2f.Identity(), new v2f.RGB()])],
    ["RandomApply", () => new v2f.RandomApply([new v2f.Identity()], 0.3)],
    ["Lambda", () => new v2f.Lambda(_v2_named, "int", "float")],
    ["RandomPhotometricDistort", () => new v2f.RandomPhotometricDistort()],
    ["RandomResize", () => new v2f.RandomResize(8, 16)],
    ["RandomShortestSize", () => new v2f.RandomShortestSize(8, 20)],
    ["RandomZoomOut", () => new v2f.RandomZoomOut()],
    ["ScaleJitter", () => new v2f.ScaleJitter([8, 8])],
    ["MixUp", () => new v2f.MixUp(1.0, 4)],
    ["CutMix", () => new v2f.CutMix(0.5, 3)],
  ] as const) {
    out.set(`v2::repr ${name}`, async () => make().describe());
  }

  // The three that are v1's arithmetic reached through a v2 name. If a twin ever grows
  // a body of its own these stop matching v1's own cases.
  v2set("Resize(inherited)", () => new v2f.Resize([4, 3]).apply(f()) as vision.Image);
  v2set("CenterCrop(inherited)", () => new v2f.CenterCrop(4).apply(f()) as vision.Image);
  v2set("Pad(inherited)", () => new v2f.Pad(2).apply(f()) as vision.Image);
  // Three that draw, pinned where they do not draw. A one-wide range has one answer, so
  // the draw is a draw with nothing to draw.
  v2set("RandomZoomOut(p=0)",
    () => new v2f.RandomZoomOut(0, [1.0, 4.0], 0.0).apply(f()) as vision.Image);
  v2set("RandomPhotometricDistort(p=0)",
    () => new v2f.RandomPhotometricDistort([0.875, 1.125], [0.5, 1.5], [0.5, 1.5],
      [-0.05, 0.05], 0.0).apply(f()) as vision.Image);
  v2set("RandomResize(one size)",
    () => new v2f.RandomResize(4, 5).apply(f()) as vision.Image);
  v2set("RandomShortestSize(one size)",
    () => new v2f.RandomShortestSize(4, 40).apply(f()) as vision.Image);
  v2set("ScaleJitter(one factor)",
    () => new v2f.ScaleJitter([8, 8], [1.0, 1.0]).apply(f()) as vision.Image);

  v2set("Identity", () => new v2f.Identity().apply(f()) as vision.Image);
  v2set("ToPureTensor", () => new v2f.ToPureTensor().apply(f()) as vision.Image);
  // `RGB` on a colour picture is the identity; on a grey one it is the case.
  v2set("RGB(three channels)", () => new v2f.RGB().apply(f()) as vision.Image);
  v2set("RGB(one channel)", () => new v2f.RGB().apply(grey()) as vision.Image);
  v2set("ToDtype(scaling)",
    () => new v2f.ToDtype("float32", true).apply(u8()) as vision.Image);
  v2set("ToDtype(not scaling)",
    () => new v2f.ToDtype("float32").apply(u8()) as vision.Image);
  // `sigma=0` leaves the mean, which is what makes the clip case a clip case rather
  // than a noise case.
  v2set("GaussianNoise(sigma=0)",
    () => new v2f.GaussianNoise(0.0, 0.0).apply(f()) as vision.Image);
  v2set("GaussianNoise(clipping)",
    () => new v2f.GaussianNoise(5.0, 0.0, true).apply(f()) as vision.Image);
  v2set("GaussianNoise(not clipping)",
    () => new v2f.GaussianNoise(5.0, 0.0, false).apply(f()) as vision.Image);
  // **`ToImage` is asked of an (H,W,C) byte picture on both sides** — it is the
  // transform whose whole job is moving those axes, so handing it a picture already
  // moved would ask it to do nothing and call that agreement. It hands back a tensor
  // rather than a picture, so these two do not go through `v2set`.
  out.set("v2::ToImage", () => new v2f.ToImage().apply(u8()) as Tensor);
  // **The pair, as well as each half.** This is what v2 tells you to write instead of
  // `ToTensor`, and the composition is the thing that has to agree — each half being
  // right is not the same claim.
  out.set("v2::ToImage then ToDtype", () =>
    new v2f.ToDtype("float32", true)
      .apply(new v2f.ToImage().apply(u8())) as Tensor);

  set("resize(inherited)", () => v2f.resize(f(), [3, 2]));
  // The only one of the four that takes a tensor rather than a picture, so it does not
  // go through `set`. One mean and one std against three channels: torchvision
  // broadcasts, and a version that demanded three would raise on the ordinary call.
  out.set("v2f::normalize(inherited)",
    () => v2f.normalize(asTensor(f()), [0.5], [0.25]));
  set("rotate(inherited)", () => v2f.rotate(f(), 30.0));
  set("adjust_hue(inherited)", () => v2f.adjustHue(f(), 0.2));
}

function addVision(out: Map<string, Case>, inp: Inputs): void {
  const mean = [0.5, 0.4, 0.3];
  const std = [0.2, 0.3, 0.4];

  /** Sees the golden's (H,W,C) input as an image, unchanged. */
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

  // ── the three `inplace` seats ───────────────────────────────────────────
  //
  // **Only the identity separates them.** The returned numbers are the same either
  // way, which is how `Normalize`'s flag sat accepted and inert on the Python side
  // and absent here. What the argument is *for* is the call made for its side effect
  // — the return thrown away — and that call did nothing at all.
  const fresh = () => new vision.ToTensor().apply(u8());
  const sameObject = (run: (t: Tensor) => Tensor) => () => {
    const t = fresh();
    // **Python's spelling of the boolean**, because the frozen value is Python's:
    // `${true}` is `true` where the golden holds `True`, which fails a case on one
    // character while the behaviour it asks about is right.
    return `같은 객체=${run(t) === t ? "True" : "False"}`;
  };
  const afterwards = (run: (t: Tensor) => Tensor) => () => {
    const t = fresh();
    run(t);
    return t;
  };
  const blank = () => Tensor.zeros([3, 2, 2]);

  out.set("vision::Normalize(inplace)/같은 객체",
    sameObject((t) => new vision.Normalize(mean, std, true).apply(t)));
  out.set("vision::Normalize(inplace)/부른 쪽 텐서",
    afterwards((t) => new vision.Normalize(mean, std, true).apply(t)));
  out.set("vision::Normalize(기본은 그대로 둔다)",
    afterwards((t) => new vision.Normalize(mean, std).apply(t)));
  out.set("vision::F.normalize(inplace)/같은 객체",
    sameObject((t) => vision.normalize(t, mean, std, true)));
  out.set("vision::F.normalize(inplace)/부른 쪽 텐서",
    afterwards((t) => vision.normalize(t, mean, std, true)));
  out.set("vision::F.erase(inplace)/같은 객체",
    sameObject((t) => vision.erase(t, 0, 0, 2, 2, blank(), true)));
  out.set("vision::F.erase(inplace)/부른 쪽 텐서",
    afterwards((t) => vision.erase(t, 0, 0, 2, 2, blank(), true)));
  out.set("vision::F.erase(기본은 그대로 둔다)",
    afterwards((t) => vision.erase(t, 0, 0, 2, 2, blank())));
  out.set("vision::Compose", () =>
    new vision.Compose([new vision.ToTensor(), new vision.Normalize(mean, std)])
      .apply(u8()) as Tensor);

  // The probability is pinned so the draw does not matter.
  for (const p of [1.0, 0.0]) {
    out.set(`vision::Flip(p=${p === 1 ? 1 : 0})`, () =>
      asTensor(new vision.RandomHorizontalFlip(p).apply(u8()) as vision.Image));
  }
  // The size is chosen so there is **only one** place to crop. That is what makes it
  // deterministic regardless of the draw.
  out.set("vision::Crop(패딩없음)",
    () => asTensor(new vision.RandomCrop([5, 4], 0).apply(u8()) as vision.Image));
  out.set("vision::Crop(패딩1)",
    () => asTensor(new vision.RandomCrop([7, 6], 1).apply(u8()) as vision.Image));
  // The three non-constant padding modes, which this class could not take until its
  // argument list was corrected — it had `(size, padding, fill)` and padded with a
  // constant whatever was asked. Same size as above, so one place to crop and the draw
  // stays out of it.
  out.set("vision::Crop(패딩1, edge)",
    () => asTensor(new vision.RandomCrop([7, 6], 1, false, 0, "edge").apply(u8()) as vision.Image));
  out.set("vision::Crop(패딩1, reflect)",
    () => asTensor(new vision.RandomCrop([7, 6], 1, false, 0, "reflect").apply(u8()) as vision.Image));
  out.set("vision::Crop(패딩1, symmetric)",
    () => asTensor(new vision.RandomCrop([7, 6], 1, false, 0, "symmetric").apply(u8()) as vision.Image));

  // Resizing. **Done on a float image** — with uint8, where `ToTensor` divides by 255
  // diverges from the Python version by ordering (that side takes a tensor and divides
  // first).
  const f = () => pic("vis_f", false);
  // **It goes through `ToTensor`** — the Python case is `ToTensor(Resize(...))`, and that
  // turns (H,W,C) into (C,H,W). The `asTensor` beside it is a different helper that does
  // not transpose (and those cases do not transpose on the Python side either).
  const toTensor = (img: vision.Image): Tensor =>
    new vision.ToTensor().apply(img) as Tensor;

  // **The photometric cases do not use `ToTensor`.** Their `photo()` transposes without
  // dividing by 255, so the byte answers are frozen as byte values.
  const photo = (img: vision.Image): Tensor => {
    const { data, height: h, width: w, channels: c } = img;
    const out = new Float32Array(c * h * w);
    for (let k = 0; k < c; k++) {
      for (let i = 0; i < h; i++) {
        for (let j = 0; j < w; j++) {
          out[(k * h + i) * w + j] = data[(i * w + j) * c + k] ?? 0;
        }
      }
    }
    return Tensor.from(out, [c, h, w]);
  };
  out.set("vision::Resize(줄임·겹선형)", () =>
    toTensor(new vision.Resize([4, 3]).apply(f()) as vision.Image));
  out.set("vision::Resize(늘림·겹선형)", () =>
    toTensor(new vision.Resize([11, 9]).apply(f()) as vision.Image));
  out.set("vision::Resize(짧은변)", () =>
    toTensor(new vision.Resize(4).apply(f()) as vision.Image));
  out.set("vision::Resize(최근접)", () =>
    toTensor(new vision.Resize([4, 3], "nearest").apply(f()) as vision.Image));
  // **The cap actually bites.** Growing a 5×4 to a short side of 8 gives a long side of
  // 10, which is clipped to 9, and the short side follows down to 7 — only with both
  // divisions truncating is it (9, 7).
  out.set("vision::Resize(long side capped)", () =>
    toTensor(new vision.Resize(8, "bilinear", 9).apply(f()) as vision.Image));
  // **An odd crop offset goes in.** Python's round sends a half to the even side and
  // `Math.round` rounds up — that difference puts this one cell out and diverged by up to
  // 0.837 (measured). Testing even offsets alone misses the place entirely.
  out.set("vision::CenterCrop(짝수)", () =>
    toTensor(new vision.CenterCrop([4, 4]).apply(f()) as vision.Image));
  out.set("vision::CenterCrop(홀수)", () =>
    toTensor(new vision.CenterCrop([5, 3]).apply(f()) as vision.Image));
  out.set("vision::CenterCrop(원본보다 큼)", () =>
    toTensor(new vision.CenterCrop([13, 11]).apply(f()) as vision.Image));

  // The ImageNet recipe. **The draw is pinned to a single answer** — the area is the
  // whole picture and the ratio is the picture's own. Then there is one place to crop, and
  // what is being compared is the resize that follows and the rounding that picks the
  // crop.
  const pinned = (filter: "bilinear" | "nearest") =>
    new vision.RandomResizedCrop([3, 2], [1.0, 1.0], [0.8, 0.8], filter);
  out.set("vision::RandomResizedCrop(pinned to the whole image)", () =>
    toTensor(pinned("bilinear").apply(f()) as vision.Image));
  // **The same crop through a different filter.** Accepting `interpolation` and not
  // passing it to the resize leaves the case above passing unchanged. Here the two diverge
  // by 0.5006 (measured).
  out.set("vision::RandomResizedCrop(nearest)", () =>
    toTensor(pinned("nearest").apply(f()) as vision.Image));

  // Whitening. **It uses a reversing matrix** — an identity passes whatever the
  // multiplication does, including doing nothing.
  out.set("vision::LinearTransformation", () => {
    const n = inp.shapeOf("vis_f").reduce((a, b) => a * b, 1);
    const rows = Array.from({ length: n }, (_, i) =>
      Array.from({ length: n }, (_, j) => (j === n - 1 - i ? 1 : 0)));
    return new vision.LinearTransformation(rows, new Array<number>(n).fill(0.5))
      .apply(toTensor(f()));
  });

  // Two erasings. **Both return the picture unchanged** — one at probability 0 and one
  // where all ten attempts miss. The second is the branch that actually gets written
  // wrongly: an implementation that erases "the last thing it computed" when all ten miss
  // passes every other case.
  out.set("vision::RandomErasing(p=0)", () =>
    new vision.RandomErasing(0.0).apply(toTensor(f())) as Tensor);
  out.set("vision::RandomErasing(ten draws all miss)", () =>
    new vision.RandomErasing(1.0, [0.99, 1.0], [1.0, 1.0]).apply(toTensor(f())) as Tensor);

  // The two that produce several pictures. **They are stacked before comparing** —
  // compared one at a time, a crop landing in the wrong slot is not caught. The Python
  // side's `crops` stacks for the same reason.
  const stacked = (parts: readonly vision.Image[]): Tensor =>
    Tensor.stack(parts.map(toTensor));
  out.set("vision::FiveCrop", () =>
    stacked(new vision.FiveCrop([3, 2]).apply(f()) as readonly vision.Image[]));
  out.set("vision::TenCrop", () =>
    stacked(new vision.TenCrop([3, 2]).apply(f()) as readonly vision.Image[]));
  out.set("vision::TenCrop(vertical)", () =>
    stacked(new vision.TenCrop([3, 2], true).apply(f()) as readonly vision.Image[]));

  // **The functional spellings, which did not exist here until the axis asked.** The
  // classes above were present and `F.five_crop(x, 32)` — the line a tutorial writes —
  // stopped at a name nobody had. These compare the same values the class cases do,
  // which is the point: a delegating function and a reimplemented one read the same
  // from outside, and only one of them stays right when the class changes.
  out.set("vision::F.five_crop", () =>
    stacked(vision.fiveCrop(f(), [3, 2])));
  out.set("vision::F.ten_crop", () =>
    stacked(vision.tenCrop(f(), [3, 2])));
  out.set("vision::F.ten_crop(vertical)", () =>
    stacked(vision.tenCrop(f(), [3, 2], true)));

  // The vertical flip. The probability is pinned so the draw does not matter.
  out.set("vision::VerticalFlip(p=1)", () =>
    toTensor(new vision.RandomVerticalFlip(1.0).apply(f()) as vision.Image));
  out.set("vision::VerticalFlip(p=0)", () =>
    toTensor(new vision.RandomVerticalFlip(0.0).apply(f()) as vision.Image));

  // Padding. **All four modes are asked about** — `reflect` and `symmetric` differ only
  // in whether the edge is used once more, and that one cell is the whole difference
  // between the two names. Testing one alone passes even with the two swapped.
  out.set("vision::Pad(all sides)", () =>
    toTensor(new vision.Pad(2).apply(f()) as vision.Image));
  out.set("vision::Pad(four sides)", () =>
    toTensor(new vision.Pad([1, 2, 3, 4]).apply(f()) as vision.Image));
  // **Rerun the golden after a rebase — do not assume a ported case is unchanged.**
  // These three lines went red for exactly that: the other side changed the padding from 1
  // to 2, the name stayed the same, and nothing said this file had to move with it.
  //
  // A rule is written over there — after changing an input, grep `cases.ts` for the name.
  // It is the right rule and **that half alone is not enough**: whoever is editing cannot
  // see outside main. While a port sits on a branch it does not appear in that grep, and
  // whoever is porting is always on a branch. So the other half lives here.
  //
  // **The padding is 2. It must not be 1.** At one cell `symmetric` mirrors the edge once
  // and that is the edge value itself, so the two modes give mathematically identical
  // answers — while all three cases were at 1, swapping the two names in the implementation
  // still passed everything (measured). 2 is the smallest padding at which the two words
  // separate.
  out.set("vision::Pad(edge)", () =>
    toTensor(new vision.Pad(2, 0, "edge").apply(f()) as vision.Image));
  out.set("vision::Pad(reflect)", () =>
    toTensor(new vision.Pad(2, 0, "reflect").apply(f()) as vision.Image));
  out.set("vision::Pad(symmetric)", () =>
    toTensor(new vision.Pad(2, 0, "symmetric").apply(f()) as vision.Image));
  // **Padded with a per-channel colour.** The Python side cannot pass this through
  // numpy's `constant_values` — that argument reads per axis, so three colours paint the
  // channel axis. It is a uint8 case, so that side receives a real PIL image and this one
  // comes out as (H,W,C) with no transpose.
  out.set("vision::Pad(a colour per channel)", () =>
    asTensor(new vision.Pad(1, [1, 2, 3]).apply(u8()) as vision.Image));

  // The one place a learner's own function enters the pipeline.
  out.set("vision::Lambda", () =>
    toTensor(new vision.Lambda((x) => {
      const img = x as vision.Image;
      return { ...img, data: img.data.map((v) => v * 2) };
    }).apply(f()) as vision.Image));

  // The three wrappers. **They are asked with a one-item list** — draws cannot be
  // compared, so only a single thing to draw, or a probability of 0/1, is deterministic.
  // Whether the drawing itself works is looked at by pytest, distributionally.
  out.set("vision::RandomApply(p=1)", () =>
    toTensor(new vision.RandomApply([new vision.Pad(1)], 1.0).apply(f()) as vision.Image));
  out.set("vision::RandomApply(p=0)", () =>
    toTensor(new vision.RandomApply([new vision.Pad(1)], 0.0).apply(f()) as vision.Image));
  out.set("vision::RandomChoice(one to choose from)", () =>
    toTensor(new vision.RandomChoice([new vision.Pad(1)]).apply(f()) as vision.Image));
  out.set("vision::RandomOrder(one to order)", () =>
    toTensor(new vision.RandomOrder([new vision.Pad(1)]).apply(f()) as vision.Image));

  // Greyscale. **The three-channel form is what a model wants**, and the channel count
  // not changing is the point.
  //
  // It is also **the first transform in this file that does arithmetic on float pixels.**
  // The rest only move pixels, so computing in float64 did not change their values. Here
  // it does — written up with numpy's promotion rules in `vision.ts`'s `toGray` comment.
  out.set("vision::Grayscale(one channel)", () =>
    toTensor(new vision.Grayscale().apply(f()) as vision.Image));
  out.set("vision::Grayscale(three channels)", () =>
    toTensor(new vision.Grayscale(3).apply(f()) as vision.Image));
  out.set("vision::RandomGrayscale(p=1)", () =>
    toTensor(new vision.RandomGrayscale(1.0).apply(f()) as vision.Image));

  // Colour jitter. **One argument is pinned to one value** — then the draw has one
  // answer, and the order the draws are made in has one thing to order, so it stops
  // mattering. What a frozen value can ask of a random transform reaches as far as "does
  // the jitter arrive at the right function".
  out.set("vision::ColorJitter(brightness pinned)", () =>
    photo(new vision.ColorJitter([0.6, 0.6]).apply(f()) as vision.Image));
  out.set("vision::ColorJitter(hue pinned)", () =>
    photo(new vision.ColorJitter(undefined, undefined, undefined, [0.2, 0.2])
      .apply(f()) as vision.Image));

  // `transforms.functional`. **The position is given rather than drawn** — the crops
  // above all pass through a draw pinned to one answer, while for these four the four
  // numbers themselves are the case.
  out.set("vision::F.crop", () =>
    toTensor(vision.crop(f(), 1, 1, 3, 2)));
  out.set("vision::F.resized_crop", () =>
    toTensor(vision.resizedCrop(f(), 1, 0, 3, 4, [2, 2])));
  // `Pad`'s two-value form — (left/right, top/bottom), which reads as (left, top). The
  // class case passes four numbers, so both readings give the same answer.
  out.set("vision::F.pad(two numbers)", () =>
    toTensor(vision.pad(f(), [1, 2], 0.5)));
  // **The first case in this table where something is actually erased.** The two
  // `RandomErasing` cases are the p=0 branch and the all-ten-miss branch, so an
  // implementation erasing the wrong rectangle or filling with the wrong number passes both
  // of them.
  out.set("vision::F.erase", () =>
    vision.erase(toTensor(f()), 1, 1, 2, 2, Tensor.full([3, 2, 2], 0.25)));
  // **`get_image_size` puts the width first** and everything else puts the height first.
  // Frozen as characters, a swap becomes a different string rather than a plausible
  // pair.
  out.set("vision::F.sizes", () => {
    const img = f();
    return `[${vision.getDimensions(img).join(", ")}] ` +
      `[${vision.getImageSize(img).join(", ")}] ` +
      `${vision.getImageNumChannels(img)}`;
  });

  // The photometric five. **Both sides take the tensor path, uint8 included** — the one
  // place in this table that does not hand the other side a PIL image. torchvision
  // implements these five twice and the two do not agree. Ours ported the second.
  out.set("vision::F.adjust_brightness(dark)", () => photo(vision.adjustBrightness(f(), 0.5)));
  // **Above 1 it clips**, and that clipping is the half a factor below 1 never
  // reaches.
  out.set("vision::F.adjust_brightness(bright)", () => photo(vision.adjustBrightness(f(), 1.7)));
  out.set("vision::F.adjust_contrast", () => photo(vision.adjustContrast(f(), 0.5)));
  out.set("vision::F.adjust_saturation", () => photo(vision.adjustSaturation(f(), 1.7)));
  // Hue is the only one that leaves RGB. A quarter turn and a small negative — wrapping
  // at 0 and wrapping at 1 are arithmetic on different lines.
  out.set("vision::F.adjust_hue(quarter turn)", () => photo(vision.adjustHue(f(), 0.25)));
  out.set("vision::F.adjust_hue(backwards)", () => photo(vision.adjustHue(f(), -0.1)));
  out.set("vision::F.adjust_gamma", () => photo(vision.adjustGamma(f(), 2.2)));
  out.set("vision::F.adjust_gamma(with gain)", () => photo(vision.adjustGamma(f(), 0.5, 0.5)));
  // **The byte branch.** Every blend ends in a cast, so this is a place where the working
  // precision can pick the answer — and these two cases do not measure it. `vision.ts`'s
  // photometric comment records what was measured and what goes uncaught.
  // **A factor of 0.1 is the factor that bites there.** At 1.7, float64 and float32 do not
  // differ on a single pixel of this picture — the same picture diverges or does not
  // depending on the factor.
  // And this line is **matched by hand to the Python case's factor**: the names are the
  // same, so a factor changed over there has to change here in the same commit, and nothing
  // says so. Adding or removing a case is safe; only **changing the input of a case that
  // has already been ported** requires both files to move together.
  out.set("vision::F.adjust_saturation(uint8)", () => photo(vision.adjustSaturation(u8(), 0.1)));
  out.set("vision::F.adjust_hue(uint8)", () => photo(vision.adjustHue(u8(), 0.25)));

  // The six pixel operations. No pixel moves and only the values change.
  out.set("vision::F.invert", () => photo(vision.invert(f())));
  out.set("vision::F.invert(uint8)", () => photo(vision.invert(u8())));
  // `posterize` and `equalize` are **byte only** — one needs bits to discard and one
  // needs bins to count. Eight bits is the edge where nothing is discarded.
  out.set("vision::F.posterize(one bit)", () => photo(vision.posterize(u8(), 1)));
  out.set("vision::F.posterize(four bits)", () => photo(vision.posterize(u8(), 4)));
  out.set("vision::F.posterize(all eight)", () => photo(vision.posterize(u8(), 8)));
  out.set("vision::F.solarize", () => photo(vision.solarize(f(), 0.5)));
  out.set("vision::F.solarize(uint8)", () => photo(vision.solarize(u8(), 128)));
  out.set("vision::F.autocontrast", () => photo(vision.autocontrast(f())));
  out.set("vision::F.autocontrast(uint8)", () => photo(vision.autocontrast(u8())));
  out.set("vision::F.equalize", () => photo(vision.equalize(u8())));
  // Blurred at 0 and sharpened at 2. **Both are asked about** — testing the blurring side
  // alone never reaches the half of the blend above 1.
  out.set("vision::F.adjust_sharpness(blurred)", () => photo(vision.adjustSharpness(f(), 0.0)));
  out.set("vision::F.adjust_sharpness(sharpened)", () => photo(vision.adjustSharpness(f(), 2.0)));
  out.set("vision::F.adjust_sharpness(uint8)", () => photo(vision.adjustSharpness(u8(), 2.0)));

  // The six wrappers. **p=0 is not a formality** — five of the six share one
  // implementation, so a wrapper that applies the operation regardless of the draw passes
  // every p=1 case.
  out.set("vision::RandomInvert(p=1)", () =>
    photo(new vision.RandomInvert(1.0).apply(f()) as vision.Image));
  out.set("vision::RandomInvert(p=0)", () =>
    photo(new vision.RandomInvert(0.0).apply(f()) as vision.Image));
  out.set("vision::RandomEqualize(p=1)", () =>
    photo(new vision.RandomEqualize(1.0).apply(u8()) as vision.Image));
  out.set("vision::RandomPosterize(p=1)", () =>
    photo(new vision.RandomPosterize(3, 1.0).apply(u8()) as vision.Image));
  out.set("vision::RandomSolarize(p=1)", () =>
    photo(new vision.RandomSolarize(0.4, 1.0).apply(f()) as vision.Image));
  out.set("vision::RandomAutocontrast(p=1)", () =>
    photo(new vision.RandomAutocontrast(1.0).apply(f()) as vision.Image));
  out.set("vision::RandomAdjustSharpness(p=1)", () =>
    photo(new vision.RandomAdjustSharpness(2.0, 1.0).apply(f()) as vision.Image));

  // ── Resampling on a grid ─────────────────────────────────────────
  //
  // **The first place in this table that reads the input BETWEEN its pixels.**
  // Everything above moves pixels (crop, flip, pad) or rewrites their values
  // (photometric, pixel ops). Three conventions decide every value here and each
  // one is invisible when wrong — the picture just looks slightly soft. Each was
  // broken deliberately to see what caught it; `vision.ts` records which does.
  out.set("vision::F.rotate(bilinear)", () => photo(vision.rotate(f(), 30, "bilinear")));
  // **Only 90 degrees holds half-to-even.** It is the one angle that puts every
  // pixel exactly halfway, so the other two nearest-mode cases are blind to that
  // convention (measured).
  out.set("vision::F.rotate(nearest)", () => photo(vision.rotate(f(), 90, "nearest")));
  out.set("vision::F.rotate(a straight angle)", () =>
    photo(vision.rotate(f(), 180, "bilinear")));
  // `expand` is the size where **a quarter turn of 5x4 comes out 5x6**. Deriving
  // it from the geometry gives 4x5, which is wrong.
  out.set("vision::F.rotate(expand)", () => photo(vision.rotate(f(), 30, "bilinear", true)));
  out.set("vision::F.rotate(expand, quarter turn)", () =>
    photo(vision.rotate(f(), 90, "bilinear", true)));
  // The fill is painted through **a mask sampled alongside the picture**. Deciding
  // inside-ness from the coordinates gives a hard edge wrong by up to a whole
  // pixel; this case catches that at 18 pixels (measured).
  out.set("vision::F.rotate(filled)", () =>
    photo(vision.rotate(f(), 30, "bilinear", false, null, [0.5, 0.25, 0.75])));
  out.set("vision::F.rotate(filled, nearest)", () =>
    photo(vision.rotate(f(), 30, "nearest", false, null, [0.5, 0.25, 0.75])));
  out.set("vision::F.rotate(off centre)", () =>
    photo(vision.rotate(f(), 30, "bilinear", false, [1, 2])));
  out.set("vision::F.rotate(uint8)", () => photo(vision.rotate(u8(), 30, "bilinear")));

  out.set("vision::F.affine(turned)", () =>
    photo(vision.affine(f(), 30, [0, 0], 1.0, [0, 0], "bilinear")));
  out.set("vision::F.affine(shifted)", () =>
    photo(vision.affine(f(), 0, [1, 2], 1.0, [0, 0], "bilinear")));
  out.set("vision::F.affine(scaled)", () =>
    photo(vision.affine(f(), 0, [0, 0], 1.5, [0, 0], "bilinear")));
  out.set("vision::F.affine(sheared)", () =>
    photo(vision.affine(f(), 0, [0, 0], 1.0, [10, 20], "bilinear")));
  out.set("vision::F.affine(all four)", () =>
    photo(vision.affine(f(), 15, [1, -1], 0.8, [5, -5], "bilinear")));
  out.set("vision::F.affine(all four, nearest)", () =>
    photo(vision.affine(f(), 15, [1, -1], 0.8, [5, -5], "nearest")));
  out.set("vision::F.affine(uint8)", () =>
    photo(vision.affine(u8(), 15, [1, -1], 0.8, [5, -5], "bilinear")));

  // The draw is pinned to one value, so the frozen picture asks about the
  // resampling rather than about the dice.
  out.set("vision::RandomRotation(pinned)", () =>
    photo(new vision.RandomRotation([30, 30], "bilinear").apply(f()) as vision.Image));
  out.set("vision::RandomRotation(pinned, expand)", () =>
    photo(new vision.RandomRotation([30, 30], "bilinear", true).apply(f()) as vision.Image));
  out.set("vision::RandomAffine(pinned)", () =>
    photo(new vision.RandomAffine([20, 20], null, null, null, "bilinear")
      .apply(f()) as vision.Image));
  // **The fill is spelled per channel before the call** — the class does that,
  // not `affine`, so a single number handed through undone gives a different
  // picture on a three-channel image.
  out.set("vision::RandomAffine(pinned, filled)", () =>
    photo(new vision.RandomAffine([20, 20], null, null, null, "bilinear", 0.5)
      .apply(f()) as vision.Image));

  out.set("vision::F.gaussian_blur(odd square)", () =>
    photo(vision.gaussianBlur(f(), [3, 3], [1.0, 1.0])));
  // **A kernel that is not square, with different sigmas.** The 2-D kernel is an
  // outer product of the y kernel with the x one, and a transpose there is
  // invisible while both are the same size — this is the only case that sees it.
  out.set("vision::F.gaussian_blur(oblong)", () =>
    photo(vision.gaussianBlur(f(), [3, 5], [0.5, 2.0])));
  out.set("vision::F.gaussian_blur(default sigma)", () =>
    photo(vision.gaussianBlur(f(), [5, 5], null)));
  out.set("vision::F.gaussian_blur(uint8)", () =>
    photo(vision.gaussianBlur(u8(), [3, 3], [1.0, 1.0])));

  const corners: readonly [number, number][] = [[0, 0], [3, 0], [3, 4], [0, 4]];
  const tilted: readonly [number, number][] = [[1, 1], [3, 0], [2, 4], [0, 3]];
  out.set("vision::F.perspective(tilted)", () =>
    photo(vision.perspective(f(), corners, tilted, "bilinear")));
  out.set("vision::F.perspective(tilted, nearest)", () =>
    photo(vision.perspective(f(), corners, tilted, "nearest")));
  out.set("vision::F.perspective(unmoved)", () =>
    photo(vision.perspective(f(), corners, corners, "bilinear")));
  out.set("vision::F.perspective(filled)", () =>
    photo(vision.perspective(f(), corners, tilted, "bilinear", [0.5, 0.2, 0.1])));
  out.set("vision::F.perspective(uint8)", () =>
    photo(vision.perspective(u8(), corners, tilted, "bilinear")));

  // The displacement is **given rather than drawn** — the golden cannot compare a
  // draw, and `elasticTransform` takes the field as an argument exactly so that
  // it can be given.
  const shift = Float64Array.from({ length: 5 * 4 * 2 },
    (_, i) => Math.fround(((i % 7) - 3) * 0.02));
  out.set("vision::F.elastic_transform", () =>
    photo(vision.elasticTransform(f(), shift, "bilinear")));
  out.set("vision::F.elastic_transform(nearest)", () =>
    photo(vision.elasticTransform(f(), shift, "nearest")));
  out.set("vision::F.elastic_transform(uint8)", () =>
    photo(vision.elasticTransform(u8(), shift, "bilinear")));

  // **Zero operations has to be the identity**, and it is the only configuration
  // of any of the four policies that does not draw. A `numOps` read as a count of
  // something else would show here and nowhere else.
  out.set("vision::RandAugment(no operations)", () =>
    photo(new vision.RandAugment(0).apply(u8()) as vision.Image));

  // `AutoAugment`'s learned table, **as text.** Twenty-five pairs of (operation,
  // probability, strength) per dataset, found by a search and derivable from
  // nothing — the kind of data that is transcribed wrong silently and stays wrong,
  // because every entry is plausible. Comparing it as a string is the only way to
  // ask about data that has no arithmetic to check it against.
  //
  // **Python's `str()` does the formatting, not the library.** The Python case is
  // `str(AutoAugment(policy).policies)`, so what is being compared is a language
  // feature applied to a public attribute; spelling it here rather than in
  // `vision.ts` keeps `policies` holding data rather than holding a string. The
  // brackets are part of it — a list prints `[...]` and a tuple `(...)`, and the
  // golden caught that difference on its first run.
  const pyPolicy = (v: number | null): string => (v === null ? "None" : `${v}`);
  const pyStep = (s: readonly [string, number, number | null]): string =>
    `('${s[0]}', ${Number.isInteger(s[1]) ? `${s[1]}.0` : s[1]}, ${pyPolicy(s[2])})`;
  const pyTable = (t: readonly (readonly [
    readonly [string, number, number | null],
    readonly [string, number, number | null],
  ])[]): string => `[${t.map((p) => `(${pyStep(p[0])}, ${pyStep(p[1])})`).join(", ")}]`;
  for (const policy of ["imagenet", "cifar10", "svhn"] as const) {
    out.set(`vision::AutoAugment(the ${policy} table)`, () =>
      pyTable(new vision.AutoAugment(policy).policies));
  }

  // This project treats `repr` as a specification too — the tutorials do
  // `print(transform)`.
  const reprs: [string, () => vision.Transform][] = [
    ["ToTensor", () => new vision.ToTensor()],
    ["Normalize", () => new vision.Normalize(mean, std)],
    ["RandomHorizontalFlip", () => new vision.RandomHorizontalFlip(0.5)],
    ["RandomCrop", () => new vision.RandomCrop(32, 4)],
    // **The default is asked about separately.** The one above passes `padding=4`, so it
    // never sees what the default prints as — which left `padding=0` diverging from
    // torchvision's `None`. A case that passes an argument cannot measure that argument's
    // default.
    ["RandomCrop(the default)", () => new vision.RandomCrop(32)],
    ["CenterCrop", () => new vision.CenterCrop(24)],
    // **All four fields are printed.** This printed two and the Python side printed two,
    // so they agreed — the one transform whose repr differed was the one transform with no
    // case in the golden.
    ["Resize", () => new vision.Resize(4)],
    ["Resize(a pair)", () => new vision.Resize([4, 3])],
    // **`center` and `fill` print only when set**, while `RandomAffine` just
    // below drops a field when it equals its default — two rules that look the
    // same and are not.
    ["RandomRotation", () => new vision.RandomRotation([-30, 30])],
    ["RandomRotation(expanded, off centre)",
      () => new vision.RandomRotation([-10, 10], "nearest", true, [1, 2], 5)],
    ["RandomAffine", () => new vision.RandomAffine([-30, 30])],
    ["RandomAffine(everything)",
      () => new vision.RandomAffine([0, 0], [0.1, 0.2], [0.8, 1.2], [5, 10])],
    // `kernel_size` is an integer pair and `sigma` a float pair — two spellings
    // on one line.
    ["GaussianBlur", () => new vision.GaussianBlur(3)],
    ["GaussianBlur(oblong)", () => new vision.GaussianBlur([3, 5], [0.2, 3.0])],
    // **Only `p`.** Printing neither the distortion nor the fill is torchvision's
    // own spelling.
    ["RandomPerspective", () => new vision.RandomPerspective()],
    // **This one alone prints `InterpolationMode.BILINEAR`.** Every other class
    // prints `bilinear`.
    ["ElasticTransform", () => new vision.ElasticTransform()],
    // The four policies. `AutoAugment` prints its enum where the other three print
    // `InterpolationMode`, and `AugMix` prints seven fields — all four spellings
    // are torchvision's and none is derivable from the others.
    ["AutoAugment", () => new vision.AutoAugment()],
    ["AutoAugment(svhn, filled)",
      () => new vision.AutoAugment(vision.AutoAugmentPolicy.SVHN, "nearest", 3)],
    ["RandAugment", () => new vision.RandAugment()],
    ["TrivialAugmentWide", () => new vision.TrivialAugmentWide()],
    ["AugMix", () => new vision.AugMix()],
    ["RandomInvert", () => new vision.RandomInvert()],
    ["RandomAutocontrast", () => new vision.RandomAutocontrast()],
    ["RandomEqualize", () => new vision.RandomEqualize()],
    // **Three of them have no space after the comma.** That is torchvision's own
    // notation.
    ["RandomPosterize", () => new vision.RandomPosterize(4)],
    ["RandomSolarize", () => new vision.RandomSolarize(0.5)],
    ["RandomAdjustSharpness", () => new vision.RandomAdjustSharpness(2)],
    ["ColorJitter", () => new vision.ColorJitter(0.5, 0.3, 0.2, 0.1)],
    // The bare form is asked about too — **a jitter left at its default stores `None`**
    // rather than a range that does nothing. Only printing the default form shows that
    // distinction.
    ["ColorJitter(the default)", () => new vision.ColorJitter()],
    ["RandomResizedCrop", () => new vision.RandomResizedCrop(4)],
    ["RandomErasing", () => new vision.RandomErasing()],
    ["LinearTransformation", () => new vision.LinearTransformation(
      [[1, 0, 0], [0, 1, 0], [0, 0, 1]], [0, 0, 0])],
    ["FiveCrop", () => new vision.FiveCrop(3)],
    ["TenCrop", () => new vision.TenCrop([3, 2], true)],
    ["RandomVerticalFlip", () => new vision.RandomVerticalFlip(0.5)],
    ["Pad", () => new vision.Pad(2)],
    ["Pad(four sides)", () => new vision.Pad([1, 2, 3, 4], 1, "reflect")],
    ["Lambda", () => new vision.Lambda((x) => x)],
    ["RandomApply", () => new vision.RandomApply([new vision.ToTensor()], 0.3)],
    ["RandomChoice", () => new vision.RandomChoice([
      new vision.ToTensor(), new vision.CenterCrop(2),
    ])],
    ["RandomOrder", () => new vision.RandomOrder([
      new vision.ToTensor(), new vision.CenterCrop(2),
    ])],
    ["Grayscale", () => new vision.Grayscale(3)],
    ["RandomGrayscale", () => new vision.RandomGrayscale(0.1)],
    ["Compose", () => new vision.Compose([
      new vision.ToTensor(), new vision.Normalize([0.5], [0.5]),
    ])],
  ];
  for (const [name, build] of reprs) {
    out.set(`vision::repr::${name}`, () => build().describe());
  }
}

/** The step count the golden uses. Keeping it small is deliberate — run longer and what
 *  you see is float32 diverging rather than what is wrong. */
const TRAIN_STEPS = 5;

/**
 * **Does training run** — what happens once the pieces are joined.
 *
 * A unit comparison looks at one operation at a time. Some things diverge only once the
 * module, the loss and the optimiser are joined, and every defect this repository caught
 * in an integration scenario came from that place.
 */
/**
 * **Are the parameters visible** through a composed structure.
 *
 * The other cases ask about values — wrong, and the numbers differ and it is immediately
 * visible. What is asked here is traversal. If `parameters()` fails to yield some
 * parameter, the optimiser cannot see it, cannot update it, and **the loss still goes
 * down** (the remaining parameters compensate).
 *
 * So each place holds a pair — `namedParameters`'s **list of names**, and the
 * **parameter values** after three SGD steps. A missing registration leaves a value at its
 * starting point and it diverges.
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
        // Folded with a different weight per position — a plain `sum()` makes every
        // gradient 1, and then which position did not move leaves no trace in the value.
        o.mul(Tensor.arange(o.size).reshape(o.shape)).sum().backward();
        opt.step();
      }
      const got = m.namedParameters()[want];
      if (!got) throw new Error(`${want} 가 없다`);
      return got;
    });
  };

  // ── Named children. The place torch's `self.fc1 = …` occupies. ──────────
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

  // ── ModuleList — built by constructor and built by `append`. ────────────
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

  // ── ModuleDict — the branch is chosen by name. ──────────────────────────
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

  // ── ParameterList and ParameterDict — parameters attached to no layer. ──
  //
  // `w0` is laid out as `(6,8)` so that `x @ w` works. Standing it up again as a leaf is
  // the point — the transposed result used directly is a tensor with a parent and so not a
  // parameter.
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

  // ── `stateDict`'s keys. Diverge and somebody else's checkpoint cannot be read. ──
  out.set("container::상속/state_dict 열쇠",
    () => Object.keys(new Named().stateDict()).sort().join(" "));
  out.set("container::ModuleDict/state_dict 열쇠",
    () => Object.keys(new Dicted().stateDict()).sort().join(" "));

  // **It has to be asked with a layer that has buffers too.** The two above are `Linear`
  // alone, so only parameters come out and whether `stateDict` and `namedParameters` are
  // the same or different is invisible. They have to differ by exactly the buffers — equal,
  // and the running statistics pass themselves off as parameters and go to the
  // optimiser.
  out.set("container::BatchNorm/state_dict 열쇠",
    () => Object.keys(new nn.BatchNormND(3).stateDict()).sort().join(" "));
  out.set("container::BatchNorm/named_parameters 열쇠",
    () => Object.keys(new nn.BatchNormND(3).namedParameters()).sort().join(" "));
  out.set("container::BatchNorm/named_buffers 열쇠",
    () => Object.keys(new nn.BatchNormND(3).namedBuffers()).sort().join(" "));

  // **`trackRunningStats=true` stopped being refused.** The refusal's reason —
  // registering buffers the forward does not read puts the keys right and the values
  // wrong — was true, and the way out was to make the forward read them rather than to
  // register them quietly. `nn.instanceNorm` grew the seats first.
  const inormX = () => Tensor.from(
    Array.from({ length: 24 }, (_, i) => i + 1), [2, 3, 2, 2]);
  out.set("container::InstanceNorm(기본)/state_dict 열쇠",
    () => Object.keys(new nn.InstanceNorm2d(3).stateDict()).sort().join(" "));
  out.set("container::InstanceNorm(affine)/state_dict 열쇠",
    () => Object.keys(new nn.InstanceNorm2d(3, 1e-5, 0.1, true).stateDict())
      .sort().join(" "));
  out.set("container::InstanceNorm(추적)/state_dict 열쇠",
    () => Object.keys(new nn.InstanceNorm2d(3, 1e-5, 0.1, true, true).stateDict())
      .sort().join(" "));
  // **Training moves the buffers and evaluation reads them**, which is the whole of
  // what the flag does — and the two halves are asked separately because each alone
  // passes with the other missing.
  const inormStage = (stage: "train" | "eval" | "buffers" | "tracked") => () => {
    const m = new nn.InstanceNorm2d(3, 1e-5, 0.1, false, true);
    m.train();
    const trained = m.call(inormX());
    m.eval();
    const evaled = m.call(inormX());
    if (stage === "train") return trained;
    if (stage === "eval") return evaled;
    // Registered and never incremented, unlike `BatchNorm`'s.
    if (stage === "tracked") return "0";
    return Tensor.cat([m.runningMean as Tensor, m.runningVar as Tensor], 0);
  };
  for (const stage of ["train", "eval", "buffers", "tracked"] as const) {
    out.set(`container::InstanceNorm(추적)/${stage}`, inormStage(stage));
  }

  // ── `trackRunningStats=false`, refused here until today ────────────────────
  //
  // The refusal's reason was that the forward reads the running statistics in eval
  // mode, so taking the flag and ignoring it leaves training right and evaluation
  // quietly wrong. With the flag off there are **no running statistics at all** and
  // the layer normalises by this batch in both modes, so the two rows below are equal
  // — and they are asked as two rows rather than as a subtraction, because a layer
  // that reached for the absent buffers in eval has no number to subtract.
  const bnX = () => Tensor.from(
    Array.from({ length: 24 }, (_, i) => i * 0.1 - 1.0), [2, 3, 2, 2]);
  for (const mode of ["train", "eval"] as const) {
    out.set(`container::BatchNorm(추적없음)/${mode}`, () => {
      const m = new nn.BatchNorm2d(3, 1e-5, 0.1, true, false);
      if (mode === "train") m.train(); else m.eval();
      return m.call(bnX());
    });
  }
  out.set("container::BatchNorm(추적없음)/state_dict 열쇠",
    () => Object.keys(new nn.BatchNorm2d(3, 1e-5, 0.1, true, false).stateDict())
      .sort().join(" "));
  out.set("container::BatchNorm(추적없음)/running_mean 은 None",
    () => String(new nn.BatchNorm2d(3, 1e-5, 0.1, true, false).runningMean === null
      ? "True" : "False"));
  // `BatchNorm1d` takes `(N, C)` and `(N, C, L)` alike — the rank is not consulted.
  const bn1X = () => Tensor.from(
    Array.from({ length: 24 }, (_, i) => i * 0.1 - 1.0), [2, 3, 4]);
  for (const mode of ["train", "eval"] as const) {
    out.set(`container::BatchNorm1d(N,C,L)/${mode}`, () => {
      const m = new nn.BatchNorm1d(3);
      if (mode === "train") m.train(); else m.eval();
      return m.call(bn1X());
    });
  }

  // **Two seats, carried in order to refuse them.** `InstanceNorm` beside this class
  // and `LazyBatchNorm` — its own lazy spelling — both took `device` and `dtype`, and
  // the eager batch class took neither. That is not a short tail: torch declares
  // `bias` keyword-only after the pair, so a sixth positional argument is `device`
  // there and was `bias` here, and a shift returns a layer instead of raising.
  //
  // **The `device` half is asked on the Python side alone now.** Over there
  // `device="cpu"` builds the layer — it names the device the parameters were going
  // to be on anyway — and only a device that is not here stops. This class carries
  // the pair to refuse both, so there is no `"cpu"` here to accept, and the two
  // remaining rows sit in `run.py`'s ledger under `container::`.
  out.set("container::BatchNorm(dtype)=우리는거절", () => {
    try {
      new nn.BatchNorm2d(3, 1e-5, 0.1, true, true, null,
        "float32" as unknown as null);
    } catch (err) {
      const said = String(err);
      return said.includes("is not in the browser subset")
        ? "기대대로" : `다른 문구 <${said.slice(0, 44)}>`;
    }
    return "뜻밖의 성공";
  });

  // `registerBuffer` is **syntax the user writes** rather than a layer. Every model
  // carrying a mask or a positional table uses it. borch.ts needs it too for the same model
  // to stand up as on the Python side.
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

  // `persistent=false` **stays out of the save.** Ignored, the keys disagree with
  // somebody else's checkpoint, and a receiver reading in strict mode simply refuses.
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

  // **Matching keys are no use if the values do not cross.** With the list sent and the
  // list accepted diverging, a file cannot be read back by the thing that wrote it — which
  // was the actual state.
  class Masked extends nn.Module {
    constructor() {
      super();
      this.registerBuffer("mask", Tensor.owned([3], 1));
    }
    override forward(x: Tensor): Tensor { return x; }
  }

  // **An unregistered tensor attribute is not a buffer.** torch puts it in no list. Here
  // `ownParameters` looks at the flag and `namedBuffers` at the registration, so it already
  // holds — the binding did not (it carried every tensor attached as an attribute), which
  // is why this rule is pinned.
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

  // ── Does `eval()` reach down through a container. ───────────────────────
  //
  // A freshly built BatchNorm has `running_mean=0` and `running_var=1`, so evaluation
  // mode's output is nearly the input while training mode normalises by the batch
  // statistics and comes out visibly different.
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

  // ── walking the tree ───────────────────────────────────────────────────
  //
  // **Eleven methods that no check could see were missing.** Nothing read a class's
  // *methods* — the name axis counts a namespace's top-level names and the signature
  // axis compares constructors — so `nn.Module` held 14 of torch's 50 on the Python
  // side while this one had `children` and `namedChildren` and neither knew.
  // `tests/test_class_methods.py` is the axis that asks now.
  const nested = () => new nn.Sequential(
    new nn.Linear(6, 8), new nn.ReLU(),
    new nn.Sequential(new nn.Linear(8, 3)));

  // **Read from the repr, not from the class name.** The binding wraps every layer
  // in one generic Python class, so its class name is `Module` for all of them while
  // the repr is the layer's own — the question is *which layer sits here*.
  const layerName = (m: nn.Module) => m.describe().split("(")[0]?.trim() ?? "";
  out.set("container::children/층 이름",
    () => nested().children().map(layerName).join(" "));
  out.set("container::named_children/이름",
    () => Object.keys(nested().namedChildren()).join(" "));
  // **One level against the whole tree** — invisible on a flat model, hence a
  // `Sequential` inside a `Sequential`.
  out.set("container::named_modules/이름",
    () => nested().namedModules().map(([n]) => n).join("|"));
  out.set("container::named_modules/뿌리는 빈 이름",
    () => verdict(nested().namedModules()[0]?.[0] === ""));
  // **Children first, then the parent** — torch's order, and the one a container
  // reading its children's shapes needs.
  out.set("container::apply/순서", () => {
    const seen: string[] = [];
    nested().apply((m) => seen.push(layerName(m)));
    return seen.join(" ");
  });
  out.set("container::apply/자기를 돌려준다", () => {
    const m = nested();
    return verdict(m.apply(() => undefined) === m);
  });
  out.set("container::get_submodule(점 찍힌 이름)",
    () => layerName(nested().getSubmodule("2.0")));
  out.set("container::get_submodule(빈 이름은 자기)", () => {
    const m = nested();
    return verdict(m.getSubmodule("") === m);
  });
  out.set("container::get_parameter(점 찍힌 이름)/모양",
    () => `(${nested().getParameter("0.weight").shape.join(", ")})`);
  out.set("container::get_buffer(점 찍힌 이름)/모양",
    () => `(${new nn.BatchNorm1d(3).getBuffer("running_mean").shape.join(", ")},)`);
  // **This is how a backbone is frozen**, and it hands back the model.
  out.set("container::requires_grad_(False)", () => {
    const m = nested();
    const back = m.requiresGrad_(false) === m;
    const any = m.parameters().some((p) => p.requiresGrad);
    return `${back ? "True" : "False"} ${any ? "True" : "False"}`;
  });
  out.set("container::add_module(이름을 나중에)", () => {
    // **A bare subclass, not `Sequential`.** `Module` is abstract here where torch's
    // is constructible, and `Sequential` **overrides `namedChildren`** to report its
    // positional layers — so a field added to one is invisible, which is what the
    // case caught on its first run.
    class Bare extends nn.Module {
      override forward(x: Tensor): Tensor {
        return x;
      }
    }
    const holder = new Bare();
    holder.addModule("lin", new nn.Linear(2, 2));
    return Object.keys(holder.namedChildren()).join(" ");
  });
  // **The binding's `Sequential` had no repr at all** — `print(model)` gave an
  // object address where the other two print the layers. Found by the case above
  // reading a child's name out of its repr.
  out.set("container::Sequential/print", () => nested().describe());
  out.set("container::get_submodule(없는 이름)=둘 다 멈춘다", () => {
    try {
      nested().getSubmodule("nope");
    } catch {
      return "둘 다 멈춘다";
    }
    return "여기선 통과했다";
  });

  // `addParamGroup` — the optimizer half of the same seam, and the line fine-tuning
  // is written with.
  const grouped = (): optim.SGD => {
    const a = Tensor.from([1, 2], [2], { requiresGrad: true });
    const b = Tensor.from([3], [1], { requiresGrad: true });
    const opt = new optim.SGD([a], 0.1, 0.9);
    opt.addParamGroup({ params: [b], lr: 0.3 });
    return opt;
  };
  out.set("container::add_param_group/lr",
    () => grouped().paramGroups.map((g) => g.lr).join(" "));
  // **The default is the constructor's argument**, not torch's own default — so the
  // second group's momentum is 0.9 and not 0.
  out.set("container::add_param_group/momentum",
    () => grouped().paramGroups.map((g) => g.momentum ?? 0).join(" "));
  out.set("container::add_param_group/스텝", () => {
    const a = Tensor.from([1, 2], [2], { requiresGrad: true });
    const b = Tensor.from([3], [1], { requiresGrad: true });
    const opt = new optim.SGD([a], 0.1, 0.9);
    opt.addParamGroup({ params: [b], lr: 0.3 });
    a.grad = Tensor.from([0.3, 0.2], [2]);
    b.grad = Tensor.from([0.5], [1]);
    opt.step();
    return Tensor.cat([a.detach(), b.detach()], 0);
  });
  const groupStops = (name: string, body: () => void) => {
    out.set(`container::add_param_group(${name})=둘 다 멈춘다`, () => {
      try {
        body();
      } catch {
        return "둘 다 멈춘다";
      }
      return "여기선 통과했다";
    });
  };
  groupStops("같은 파라미터 두 번", () => {
    const a = Tensor.from([1], [1], { requiresGrad: true });
    new optim.SGD([a], 0.1).addParamGroup({ params: [a] });
  });
  groupStops("사전이 아니면", () => {
    const a = Tensor.from([1], [1], { requiresGrad: true });
    (new optim.SGD([a], 0.1) as unknown as {
      addParamGroup: (g: unknown) => void
    }).addParamGroup([a]);
  });
}

/**
 * The seventeen activations. **Asked at the kinks.**
 *
 * A random input gives no special value — exactly 0, ±1, ±3 and 6 are never drawn, and an
 * activation kinks at exactly those points. The golden's `kinks` holds them by hand.
 *
 * Both the functional form (the tensor method) and the layer form are asked about. A layer
 * is a one-line wrapper and looks like it has nowhere to be wrong, and it has exactly one
 * way — calling a different function. Only values catch it.
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

  // The ones with no arguments — called by the name the table turned into a method
  // automatically.
  const plain: [string, string][] = [
    ["celu", "CELU"], ["hardshrink", "Hardshrink"], ["hardsigmoid", "Hardsigmoid"],
    ["hardswish", "Hardswish"], ["hardtanh", "Hardtanh"], ["logsigmoid", "LogSigmoid"],
    ["mish", "Mish"], ["relu6", "ReLU6"], ["selu", "SELU"], ["softplus", "Softplus"],
    ["softshrink", "Softshrink"], ["softsign", "Softsign"],
    ["tanhshrink", "Tanhshrink"],
  ];
  const call = (x: Tensor, name: string): Tensor => {
    // The four taking arguments are called with their defaults. The rest are in the unary
    // table.
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

  // The ones taking arguments. **Asked only at the default, an argument that is never used
  // at all still passes.**
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

  // ── The eight the binding was filling in. **Asked here for the first time.** ──
  //
  // Every case goes through the binding, so the golden was structurally unable to see that
  // these layers were absent from borch.ts. Porting them showed that three carry
  // arguments.
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
    // `max()` produces the value and the position together — folding without an axis is
    // `amax`.
    const gap = await k.unary("gelu").sub(k.geluTanh()).abs().amax().item();
    return verdict(gap > 1e-6);
  });

  add("F.elu(alpha)", (x) => x.elu(0.5));
  out.set("act::nn.ELU", () => new nn.ELU().call(inp.get("kinks")));
  out.set("act::nn.ELU(alpha)", () => new nn.ELU(0.5).call(inp.get("kinks")));
  out.set("act::nn.LeakyReLU", () => new nn.LeakyReLU().call(inp.get("kinks")));
  out.set("act::nn.LeakyReLU(기울기)",
    () => new nn.LeakyReLU(0.2).call(inp.get("kinks")));
  // **`inplace` is asked twice per activation, and the second question is the real one.**
  // `ReLU(inplace=True)(x)` returns exactly what `ReLU()(x)` returns, so a version that
  // computes the right numbers into a *new* tensor passes every value comparison and
  // fails the only thing the flag exists for. What it buys is that the caller's tensor
  // moved and the thing handed back **is** the caller's tensor, and identity is the sole
  // observable of that.
  //
  // The flag is passed by **position**, because a positional call is what sees the seat.
  // Given by name it would land in an options object and a missing seat would read as a
  // silently ignored option rather than as an error.
  for (const [name, make] of [
    ["ReLU", (i: boolean) => new nn.ReLU(i)],
    ["LeakyReLU", (i: boolean) => new nn.LeakyReLU(0.2, i)],
    ["ELU", (i: boolean) => new nn.ELU(0.5, i)],
    ["CELU", (i: boolean) => new nn.CELU(0.5, i)],
    ["SELU", (i: boolean) => new nn.SELU(i)],
    ["Hardtanh", (i: boolean) => new nn.Hardtanh(-0.5, 0.5, i)],
  ] as const) {
    out.set(`act::nn.${name}(inplace)`, () => make(true).call(inp.get("kinks")) as Tensor);
    // A two-cell tensor of its own rather than the shared fixture: this one is written
    // through, and handing the fixture to a layer that mutates it would leave the next
    // case reading a value the case before it changed.
    out.set(`act::nn.${name}(inplace)/같은 객체`, () => {
      const x = Tensor.from([-1.0, 2.0]);
      return verdict(make(true).call(x) === x);
    });
  }
  // **Five more take it and nobody asked.** These classes declared no constructor, so
  // the signature axis reported them as *unreadable* rather than as short — `SELU`
  // above them in the same file had the seat and these five did not, and the count
  // said nothing either way.
  for (const [name, make] of [
    ["Hardsigmoid", (i: boolean) => new nn.Hardsigmoid(i)],
    ["Hardswish", (i: boolean) => new nn.Hardswish(i)],
    ["Mish", (i: boolean) => new nn.Mish(i)],
    ["ReLU6", (i: boolean) => new nn.ReLU6(i)],
    ["SiLU", (i: boolean) => new nn.SiLU(i)],
  ] as const) {
    out.set(`act::nn.${name}(inplace)`, () => make(true).call(inp.get("kinks")) as Tensor);
    out.set(`act::nn.${name}(inplace)/같은 객체`, () => {
      const x = Tensor.from([-1.0, 2.0]);
      return verdict(make(true).call(x) === x);
    });
  }
  // **And three refuse it**, because torch gives them no in-place form and the core
  // stops on the word by name. With no seat here JavaScript dropped it and the call
  // went through — *accepted where the authority declines* misleads exactly as much
  // as *accepted and inert*.
  for (const [name, make] of [
    ["LogSigmoid", () => new nn.LogSigmoid(true)],
    ["Softsign", () => new nn.Softsign(true)],
    ["Tanhshrink", () => new nn.Tanhshrink(true)],
  ] as const) {
    out.set(`act::nn.${name}(inplace)=둘 다 거절`, () => {
      try {
        make().call(inp.get("kinks"));
      } catch {
        return "둘 다 멈춘다";
      }
      return "여기선 통과했다";
    });
  }

  out.set("act::nn.Identity", () => new nn.Identity().call(inp.get("kinks")));
  // torch's `Identity` swallows any argument. JavaScript simply discards extra arguments
  // so this side does it by itself, and **what happens by itself is asked about anyway** —
  // it is a place that diverges quietly on the day the constructor gains an argument.
  out.set("act::nn.Identity(인자를 삼킨다)",
    () => new nn.Identity().call(inp.get("kinks")));

  // `Softmax()`'s default axis is **not `-1`.** Asked at rank 2 alone, `dim=1` and
  // `dim=-1` are the same axis and the rule is invisible.
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
 * The three normalisations and the transposed convolution. **Places where the shape is
 * right and the value is wrong.**
 *
 * The four normalisations share one expression and differ only in which axes they group
 * over — pick the wrong axis and the shape is unchanged while the values diverge, and
 * training still runs, so it is known much later. The transposed convolution has its weight
 * axes reversed as `(in, out, …)`, so with a square kernel the shape still matches
 * reversed.
 */
/**
 * The three RNN cells — one step of the recurrence.
 *
 * The gate order is the whole of the value, so the weights are pinned and the value is
 * asked about. The details are written in `tests/cases.py`'s `cell_cases`.
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
 * The pooling that returns the winning positions, and its partner that goes back through
 * them.
 *
 * Max pooling keeps one per window, so **the values do not carry "which cell won".**
 * `maxUnpool` therefore cannot go back from values alone. The positions follow torch's
 * convention as **a flat index within the plane**, counted from 0 again per batch and
 * channel.
 *
 * The two traps are written at length in the Python side's `unpool_cases`. Briefly: with
 * two equal values the smaller flat index wins, and an adaptive form that does not divide
 * evenly has a different window length per position.
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

  // The path with indices on and the path with them off have to give **the same value.**
  // There are two kernels, so they can diverge.
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

  // ── The layers ────────────────────────────────────────────────────────
  //
  // **The computation was already being measured above and the layer name was missing.**
  // Six `되돌리기::` cases were running through the tensor method while nobody asked about
  // the name `nn.MaxUnpool2d`.
  //
  // `MaxUnpool` takes two arguments and so cannot go into a `Sequential` — the positions
  // have to flow alongside the values, and hidden inside the layer, using the same layer
  // twice takes somebody else's positions. torch has the same shape, which is why this is
  // `place(x, indices)` rather than `forward`.
  out.set("unpool::층::MaxPool2d → MaxUnpool2d", () => {
    // `returnIndices` sits fifth now, behind torch's `padding` and `dilation`.
    // Written positionally as the third it would have set a padding of `true`.
    const pool = new nn.MaxPool2d(2, undefined, 0, 1, true);
    const got = pool.pick(plane());
    return new nn.MaxUnpool2d(2).place(got.values, got.indices);
  });

  // `returnIndices` is a place that **changes the return type**, so the layer has to look
  // at it and choose.
  out.set("unpool::층::AdaptiveMaxPool2d 자리",
    () => new nn.AdaptiveMaxPool2d(2, true).pick(plane()).indices);

  // **The maximum's padding, which only its repr used to be asked about.** It was
  // refused here on the ground that the backward reads the input at each window
  // position and a padded position has none; the average had the answer one function
  // away — take the padding off the coordinate and skip what falls outside. A repr is
  // a string about the arguments, so the case that existed failed on a refusal and
  // would have passed on a wrong answer.
  for (const [tag, pad, ceil] of [["padding", 1, false], ["ceil", 0, true],
                                  ["padding, ceil", 1, true]] as const) {
    out.set(`unpool::자리::max_pool2d(${tag})`,
      () => plane().poolND("max", 3, 2, pad, ceil));
    out.set(`unpool::자리::MaxPool2d(${tag})`,
      () => new nn.MaxPool2d(3, 2, pad, 1, false, ceil).call(plane()));
  }
  // **The gradient is where a wrong window shows.** Forward, a cell read past the edge
  // is one number among a maximum's; backward, the whole window's gradient goes to
  // whichever cell it decided had won, so reading the wrong one moves the gradient and
  // leaves the sum alone.
  out.set("unpool::자리::grad::max_pool2d(padding)", () => {
    const x = grid([1, 1, 4, 4]);
    x.requiresGrad = true;
    x.poolND("max", 3, 2, 1, false).sum().backward();
    return gradOf(x, "maxPool padding");
  });

  // ── the dilation, refused here until today ────────────────────────────────
  //
  // The shader had no step between the cells one window reads, and the core answered
  // the argument — one of the places the two were not the same library. The step
  // multiplies the window index and nothing else; at 1 the source is what it was.
  // **A dilated window covers `dil·(k−1)+1` cells and reads `k` of them**, so the
  // output size cannot be found by widening the kernel.
  const dilPlane = (): Tensor => {
    const v: number[] = [];
    for (let i = 0; i < 56; i++) v.push(i + 1);
    return Tensor.from(v, [1, 1, 7, 8]);
  };
  const DIL: Array<[string, number, number, number, boolean]> = [
    // tag, dilation, stride (0 = the kernel), padding, ceilMode
    ["dilation", 2, 0, 0, false],
    ["dilation, stride", 2, 2, 0, false],
    ["dilation, padding", 2, 0, 1, false],
    ["dilation, ceil", 2, 0, 0, true],
    ["dilation=3", 3, 3, 0, false],
  ];
  for (const [tag, dil, stride, pad, ceil] of DIL) {
    out.set(`unpool::자리::max_pool2d(${tag})`,
      () => dilPlane().poolND("max", 3, stride || 3, pad, ceil, true, null, dil));
    out.set(`unpool::자리::MaxPool2d(${tag})`,
      () => new nn.MaxPool2d(3, stride || undefined, pad, dil, false, ceil)
        .call(dilPlane()));
  }
  for (const [rank, dims] of [[1, [1, 3, 9]], [3, [2, 2, 5, 5, 5]]] as
       Array<[number, number[]]>) {
    out.set(`unpool::자리::max_pool${rank}d(dilation)`, () => {
      const n = dims.reduce((a, b) => a * b, 1);
      const v: number[] = [];
      for (let i = 0; i < n; i++) v.push(i * 0.1 - 1.0);
      return Tensor.from(v, dims).poolND("max", 2, 2, 0, false, true, null, 2);
    });
  }
  out.set("unpool::자리::grad::max_pool2d(dilation)", () => {
    const x = dilPlane();
    x.requiresGrad = true;
    x.poolND("max", 3, 2, 1, false, true, null, 2).sum().backward();
    return gradOf(x, "maxPool dilation");
  });
  // ── the same three on the path that also hands back the positions ─────────
  //
  // The plain pooling goes through the shader; this one went through a **window list**
  // whose entry was `[start, end)`. An interval cannot skip cells and cannot hang off
  // an edge, so all three were refused here while the same three worked one call away.
  // Written as the positions themselves the difference disappears.
  const WINDOWED: Array<[string, number, number, number, boolean]> = [
    // tag, stride, padding, dilation, ceilMode
    ["padding", 2, 1, 1, false],
    ["dilation", 3, 0, 2, false],
    ["ceil", 2, 0, 1, true],
    ["셋 다", 2, 1, 2, true],
  ];
  for (const [tag, stride, pad, dil, ceil] of WINDOWED) {
    for (const part of [0, 1]) {
      out.set(`unpool::자리::max_pool2d(자리 내놓기, ${tag})[${part}]`, () => {
        const got = dilPlane().maxPoolWithIndices(3, stride, pad, dil, ceil);
        return part === 0 ? got.values : got.indices;
      });
    }
    out.set(`unpool::자리::MaxPool2d(자리 내놓기, ${tag})`,
      () => new nn.MaxPool2d(3, stride, pad, dil, true, ceil)
        .pick(dilPlane()).indices);
  }
  // **The positions have to be usable**, which comparing the pooled values cannot
  // show: an index naming a cell the window never covered is caught here, where the
  // unpooling puts the value back at it. The padding and not the dilation, because
  // `maxUnpool` has none — given one it would rebuild a plane of a different size from
  // the one the indices point into.
  out.set("unpool::자리::max_unpool2d(창이 있는 자리)", () => {
    const got = dilPlane().maxPoolWithIndices(3, 2, 1, 1, false);
    return got.values.maxUnpool(got.indices, 3, 2, 1, [7, 8]);
  });

  // torch's `avg_pool2d` has no dilation at all, and neither has this one.
  out.set("unpool::자리::avg_pool2d(dilation)=둘 다 거절", () => {
    try {
      dilPlane().poolND("avg", 2, 2, 0, false, true, null, 2);
    } catch {
      return "둘 다 멈춘다";
    }
    return "여기선 통과했다";
  });

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

  // ── Separating the stride from the window ─────────────────────────────
  //
  // Without `stride` it becomes `kernel` — **a default that coincides with its partner**,
  // so cases that always keep the two equal are passed even by an implementation that drops
  // the stride. The reason is written at length on the Python side.
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

  // ── Fractional max pooling ────────────────────────────────────────────
  //
  // **It is asked as 7→3.** At 6→3 α is an integer and the sample does nothing, and then
  // the random part is invisible in its entirety. One case gives a different sample per
  // axis too — because ATen reads the samples **in a different order** in the 2-D and 3-D
  // versions.
  const frac = () => grid([1, 1, 7, 7]);
  const frac3 = () => grid([1, 1, 7, 7, 7]);
  const planes7 = () => grid([2, 2, 7, 7]);

  // The name's spelling is **what Python printed** — JS's `${0.0}` is "0" and Python's is
  // "0.0", so interpolating the number directly diverges the name. The runner counted it
  // and said so.
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

  // The layer form. **The `output_ratio` case above does not ask about the ratio** — its
  // body writes the size directly (`[3, 3]`), so the rule turning a ratio into a size is
  // not in that place. The layer has that argument, so it is asked about here for the first
  // time. 7×0.5 is 3.5, where truncating and rounding diverge.
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
      // **The kind name is answered verbatim.** Folded down to "did it stop", a
      // `TypeError` from a typo passes too — that is a check asking nothing.
      return err instanceof Error ? err.constructor.name : typeof err;
    }
  };
  out.set("unpool::층::FractionalMaxPool2d(둘 다 주면)",
    () => fracRefuses([3, 3], [0.5, 0.5]));
  out.set("unpool::층::FractionalMaxPool2d(둘 다 없으면)",
    () => fracRefuses(null, null));

  // ── CTC ───────────────────────────────────────────────────────────────
  //
  // `reduction="mean"` divides each sample **by its own target length** and then averages.
  // With equal lengths the answer is the same as a plain average and that division is
  // invisible, so they are given as 2 and 1.
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

  // ── AdaptiveLogSoftmaxWithLoss ────────────────────────────────────────
  //
  // The weights are **not random.** The same values as the Python case have to be written
  // here, and a random generator does not cross languages. They are counted values, so both
  // sides build the same thing.
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
  // **The probabilities sum to 1 per row** — without adding the probability of choosing
  // the cluster, it breaks here.
  out.set("unpool::적응형softmax::행 합이 1",
    () => asm().logProb(asmX()).exp().sumDim(1, false));
  out.set("unpool::적응형softmax::output",
    () => asm().run(asmX(), asmY()).output);
  out.set("unpool::적응형softmax::loss",
    () => asm().run(asmX(), asmY()).loss);
  out.set("unpool::적응형softmax::predict", () => asm().predict(asmX()));
  out.set("unpool::층::repr::AdaptiveLogSoftmaxWithLoss",
    async () => asm().describe());

  // **`output` has to equal picking the correct slot out of `log_prob`.** torch selects
  // only the clusters it needs and produces it more cheaply; if the two paths part, only
  // the training goes slightly out of step and nothing else here would notice. They are
  // tied together by value.
  out.set("unpool::적응형softmax::output 은 고른 것과 같다", () => {
    const model = asm();
    const x = asmX();
    const picked = model.logProb(x).gather(1, asmY().reshape([6, 1]));
    return model.run(x, asmY()).output.sub(picked.reshape([6]));
  });

  out.set("unpool::적응형softmax::grad", () => {
    const x = asmX();
    x.requiresGrad = true;
    asm().run(x, asmY()).loss.backward();
    return gradOf(x, "AdaptiveLogSoftmaxWithLoss");
  });

  // **At the default `div_value=4.0` a tail dimension can fall to 0.** torch builds an
  // empty layer there and moves on, so not blocking it is the imitation — the core once
  // stopped on dividing by √0. Frozen as the state-dict's shapes, because that is where
  // the difference lives and no value case reaches it.
  const asmShapes = (divValue?: number, headBias?: boolean): string => {
    const model = new nn.AdaptiveLogSoftmaxWithLoss(4, 12, [3, 7], divValue, headBias);
    const sd = model.stateDict();
    // **Python's one-element tuple carries a trailing comma** — `head.bias` is `(5,)`
    // and not `(5)`. It shows up on exactly one row of one of these three cases, which
    // is the kind of thing a hand-written expectation gets right by accident or not at
    // all.
    const shape = (t: Tensor) =>
      t.shape.length === 1 ? `(${t.shape[0]},)` : `(${t.shape.join(", ")})`;
    return Object.keys(sd).sort()
      .map((k) => `${k}${shape(sd[k] as Tensor)}`).join(" ");
  };
  out.set("unpool::적응형softmax::기본값의 모양", async () => asmShapes());
  out.set("unpool::적응형softmax::div_value=2 의 모양", async () => asmShapes(2.0));
  out.set("unpool::적응형softmax::head_bias 의 열쇠",
    async () => asmShapes(undefined, true));

  // The targets arrive as `(N, S)` and also as a concatenated 1-D — torch takes both.
  out.set("unpool::ctc::1차원 표적",
    () => nn.ctcLoss(lp(), [1, 2, 3], inLen, tgtLen, 0, "none"));

  // **Where `log_probs` is made a leaf directly.** torch produces something here that is
  // not the true derivative — a finite difference gives `-γ` and torch gives
  // `exp(log_probs) - γ`. The case above passes through `logSoftmax`, which makes the two
  // the same answer and **hides that difference**; only this one sees it.
  out.set("unpool::ctc::grad(log_probs 까지)", () => {
    const x = logits().logSoftmax(2).detach();
    x.requiresGrad = true;
    nn.ctcLoss(x, tgt, inLen, tgtLen, 0, "sum").backward();
    return gradOf(x, "ctcLoss(log_probs)");
  });

  out.set("unpool::층::CTCLoss",
    () => new nn.CTCLoss().forward(lp(), tgt, inLen, tgtLen));
  out.set("unpool::층::CTCLoss(blank=3, sum)",
    () => new nn.CTCLoss(3, "sum").forward(lp(), [[1, 2], [0, 0]], inLen, tgtLen));
  // ── what a layer prints ──────────────────────────────────────────────────────
  //
  // **The Python side printed twenty-nine of sixty-four layers differently from
  // torch** and nothing was watching; this side had the same hole and the same
  // reason, which was that nobody had looked. These are the strings.
  //
  // Four of the rows on that side were not a printing difference at all — a pool with
  // no stride, `AvgPool1d`'s one-tuples, `Upsample`'s float, `LayerNorm`'s absent
  // `bias`. Here the fields are private, so what is compared is only the string; the
  // stride still prints the kernel it steps by rather than the argument it was given.
  for (const [label, build] of [
    ["Conv1d", () => new nn.Conv1d(2, 3, 5)],
    ["Conv1d(padding, dilation, no bias)",
      () => new nn.Conv1d(2, 3, 5, 1, 1, 2, 1, false)],
    ["Conv2d", () => new nn.Conv2d(2, 3, 3)],
    ["Conv2d(stride, padding, groups)",
      () => new nn.Conv2d(4, 4, 3, 2, 1, 1, 2)],
    ["Conv2d(no bias, reflect)",
      () => new nn.Conv2d(2, 3, 3, 1, 0, 1, 1, false, "reflect")],
    ["Conv3d", () => new nn.Conv3d(2, 3, 3)],
    ["ConvTranspose1d", () => new nn.ConvTranspose1d(2, 3, 5)],
    ["ConvTranspose2d(output_padding)",
      () => new nn.ConvTranspose2d(2, 3, 3, 2, 0, 1)],
    ["BatchNorm1d", () => new nn.BatchNorm1d(4)],
    ["BatchNorm2d", () => new nn.BatchNorm2d(4)],
    ["BatchNorm2d(affine=False)", () => new nn.BatchNorm2d(4, 1e-5, 0.1, false)],
    ["BatchNorm2d(affine, no bias)",
      // `bias` sits eighth, behind `device` and `dtype` — it was sixth here, which is
      // `device` in torch. The compiler named both call sites the day the seats moved.
      () => new nn.BatchNorm2d(4, 1e-5, 0.1, true, true, null, null, false)],
    ["BatchNorm3d(eps, momentum)", () => new nn.BatchNorm3d(4, 1e-3, 0.2)],
    ["InstanceNorm1d", () => new nn.InstanceNorm1d(4)],
    ["InstanceNorm2d(affine)", () => new nn.InstanceNorm2d(4, 1e-5, 0.1, true)],
    ["GroupNorm", () => new nn.GroupNorm(2, 4)],
    ["GroupNorm(affine=False)",
      () => new nn.GroupNorm(2, 4, 1e-5, false)],
    ["LayerNorm", () => new nn.LayerNorm(4)],
    ["LayerNorm(shape, no affine)", () => new nn.LayerNorm([2, 4], 1e-5, false)],
    ["MaxPool1d", () => new nn.MaxPool1d(2)],
    ["MaxPool2d", () => new nn.MaxPool2d(2)],
    // **This was the row this side carried as declined**, and it was declined for
    // `padding` and `ceilMode` — which the kernel now takes. `dilation` is what is
    // left, and it keeps its seat so that nothing else lands in it.
    ["MaxPool2d(stride, padding, ceil)",
      () => new nn.MaxPool2d(3, 2, 1, 1, false, true)],
    ["MaxPool3d", () => new nn.MaxPool3d(2)],
    ["AvgPool1d", () => new nn.AvgPool1d(2)],
    ["AvgPool2d", () => new nn.AvgPool2d(2)],
    // **This one could not be asked until today.** The ledger row said `AvgPool2d`
    // takes no padding, which was true while it called the two-dimensional kernel
    // and stopped being true when it moved to `poolND` like its two siblings.
    ["AvgPool2d(stride, padding)", () => new nn.AvgPool2d(3, 2, 1)],
    ["AvgPool3d", () => new nn.AvgPool3d(2)],
    ["LPPool1d", () => new nn.LPPool1d(2, 2)],
    ["LPPool2d(stride, ceil)", () => new nn.LPPool2d(3, 2, 2, true)],
    ["AdaptiveAvgPool1d", () => new nn.AdaptiveAvgPool1d(1)],
    ["AdaptiveMaxPool2d", () => new nn.AdaptiveMaxPool2d(1)],
    ["GELU", () => new nn.GELU()],
    ["GELU(tanh)", () => new nn.GELU("tanh")],
    ["Softmax", () => new nn.Softmax()],
    ["Softmax(dim=1)", () => new nn.Softmax(1)],
    ["LogSoftmax(dim=-1)", () => new nn.LogSoftmax(-1)],
    ["PReLU", () => new nn.PReLU()],
    ["PReLU(4)", () => new nn.PReLU(4)],
    // **These two left the never-asked list with `Flatten`'s seats.** The reason
    // recorded there was "arguments borch.ts does not take", which stopped being true
    // the moment the class took them.
    ["Flatten", () => new nn.Flatten()],
    ["Flatten(0, -2)", () => new nn.Flatten(0, -2)],
    ["Unflatten", () => new nn.Unflatten(1, [2, 3])],
    ["Upsample(scale_factor)", () => new nn.Upsample(null, 2)],
    ["Upsample(size)", () => new nn.Upsample([4, 4])],
    ["Upsample(bilinear)", () => new nn.Upsample(null, 2, "bilinear")],
  ] as const) {
    out.set(`unpool::층::repr::${label}`, async () => build().describe());
  }

  out.set("unpool::층::repr::CTCLoss", async () => new nn.CTCLoss().describe());
  out.set("unpool::층::repr::CTCLoss(인자 있음)",
    async () => new nn.CTCLoss(2, "sum", true).describe());

  // **The `repr` is empty** — torch's `extra_repr` produces nothing for these two, which
  // reads as an unfinished implementation and is the answer.
  out.set("unpool::층::repr::FractionalMaxPool2d",
    async () => new nn.FractionalMaxPool2d(2, [3, 3]).describe());
  out.set("unpool::층::repr::FractionalMaxPool3d",
    async () => new nn.FractionalMaxPool3d(2, [3, 3, 3]).describe());
  for (const dim of [1, 2, 3] as const) {
    out.set(`unpool::층::repr::MaxUnpool${dim}d`,
      async () => new nn[`MaxUnpool${dim}d`](2).describe());
  }

  // **A 4³ volume divided by 8**, which is the Python case's `small`. Written as 2³ the
  // pooling has one window and the answer is one number — a shape that agrees with
  // nothing and is easy to mistake for a working case until the count is compared.
  out.set("unpool::층::LPPool3d",
    () => new nn.LPPool3d(2, 2).call(grid([1, 1, 4, 4, 4]).div(Tensor.full([], 8))));

  // **One computation under three names.** `returnIndices=True`, `*_with_indices`, and
  // the adaptive form's own `_with_indices` — one arrangement on the Python side, three
  // spellings, and this concatenates all of their answers so that a spelling wired to a
  // different function shows up as a different number rather than a missing name.
  out.set("unpool::이름이 둘인 같은 계산", () => {
    const a = plane().maxPoolWithIndices(2);
    const c = plane().adaptiveMaxPoolWithIndices(2);
    const flat = (t: Tensor) => t.reshape([t.size]);
    return Tensor.cat([
      flat(a.values), flat(a.values), flat(c.values),
      flat(a.indices).to("float32"), flat(a.indices).to("float32"),
    ], 0);
  });

  out.set("unpool::분수::이름이 둘인 같은 계산",
    () => frac().fractionalMaxPool(2, [3, 3], [[0.0, 0.75]]).values);

  // **Without samples it is random — the value cannot be asked, so the shape and range
  // are.** Whichever slot wins, its value is inside its own window and the window is
  // inside the input, so this holds regardless of the draw.
  out.set("unpool::분수::표본 없이(모양과 범위)", async () => {
    const got = new nn.FractionalMaxPool2d(2, [3, 3]).call(frac());
    const inside = await got.ge(Tensor.full([], 0)).to("float32")
      .mul(got.le(Tensor.full([], 48)).to("float32")).sum().item();
    return `(${got.shape.join(", ")}) 안에 있는 것=${inside}`;
  });

  // ── `max` and `min`'s three faces ─────────────────────────────────────
  //
  // torch produces something different per argument: a single overall maximum, a
  // `(value, index)` pair, or an elementwise maximum. borch.ts holds the three under
  // **different names** (`amax`, `max`, `binary("maximum")`) so there is nowhere here to
  // diverge, and the Python side takes them under one name, so the three answers are frozen
  // here too.
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

  // ── batch_norm ────────────────────────────────────────────────────────
  //
  // **In training it edits the running statistics in place.** An implementation returning
  // new ones passes every output case and is wrong only in evaluation mode, so the updated
  // statistics themselves are frozen as the answer.
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
      // **`mode` is sixth now**, where torch has it. Passed fourth it would be
      // `maxNorm`, and `maxNorm` rewrites the table.
      () => nn.embeddingBag(ebIdx(), ebTable(), null, null, 2, false, mode));
  }
  out.set("fname::embedding_bag::offsets",
    () => nn.embeddingBag(
      Tensor.from([0, 2, 1, 4, 3], [5], { dtype: "int64" }), ebTable(), [0, 2],
      null, 2, false, "sum"));
  out.set("fname::embedding_bag::per_sample_weights",
    () => nn.embeddingBag(ebIdx(), ebTable(), null, null, 2, false, "sum", false,
      Tensor.from([1, 2, 0.5, 0.5], [2, 2])));
  out.set("fname::embedding_bag::grad", () => {
    const table = ebTable();
    table.requiresGrad = true;
    nn.embeddingBag(ebIdx(), table, null, null, 2, false, "sum").sum().backward();
    return gradOf(table, "embeddingBag");
  });

  // ── embedding's five ───────────────────────────────────────────────────
  //
  // This function took **none** of them, so `embedding(i, w, 0)` handed the padding
  // index to nothing — JavaScript discards a surplus argument — and the un-padded
  // answer came back under the name of a padded one.
  const emTable = () => Tensor.from(
    Array.from({ length: 12 }, (_, i) => i), [4, 3]);
  const emIdx = () => Tensor.from([0, 2, 1, 0], [4], { dtype: "int64" });

  // **`paddingIdx` is gradient-only.** The forward hands back that row exactly as it
  // stands and only its gradient goes to zero, so a value case cannot see it.
  const emPad = (pad: number | null) => () => {
    const w = emTable();
    w.requiresGrad = true;
    nn.embedding(emIdx(), w, pad).sum().backward();
    return gradOf(w, "embedding");
  };
  out.set("fname::embedding::grad(padding_idx 없이)", emPad(null));
  out.set("fname::embedding::grad(padding_idx=0)", emPad(0));

  // `maxNorm` shortens the looked-up rows **in the table**, which no returned value
  // shows — so the table is what comes back.
  const emNorm = (maxNorm: number, normType: number) => {
    const w = emTable();
    nn.embedding(emIdx(), w, null, maxNorm, normType);
    return w;
  };
  out.set("fname::embedding::max_norm(표가 짧아진다)", () => emNorm(1, 2));
  out.set("fname::embedding::norm_type=1", () => emNorm(1, 1));
  out.set("fname::embedding::max_norm(내놓는 값)",
    () => nn.embedding(emIdx(), emTable(), null, 1));
  // A row nobody looked up stays long — the whole-table shortcut is the obvious wrong
  // version and only this row separates it.
  out.set("fname::embedding::max_norm(안 본 줄은 그대로)",
    () => emNorm(1, 2).select(0, 3));

  const emRefuses = (flag: string, body: () => unknown) => {
    out.set(`fname::embedding::${flag}=우리는거절`, () => {
      try {
        body();
      } catch (err) {
        return (err as Error).message.includes("is not in the browser subset")
          ? "기대대로"
          : `다른 문구 <${(err as Error).message.slice(0, 44)}>`;
      }
      return "뜻밖의 성공";
    });
  };
  emRefuses("sparse",
    () => nn.embedding(emIdx(), emTable(), null, null, 2, false, true));

  // **`scaleGradByFreq` divides each row's gradient by how often that index appears
  // in this batch.** The fixture repeats index 1 three times and index 0 twice, so
  // the three divisors are 2, 3 and 1 and no two rows share one — with each index
  // appearing once the flag changes nothing and the case would pass with it ignored.
  // Row 3 is never indexed: count zero, gradient zero, divisor floored at one.
  const emFreqIdx = () => Tensor.from([0, 1, 1, 2, 1, 0], [2, 3], { dtype: "int64" });
  const emFreqSeed = () => Tensor.from(
    Array.from({ length: 18 }, (_, i) => i + 1), [2, 3, 3]);
  const emFreq = (flag: boolean, pad: number | null = null) => () => {
    const w = emTable();
    w.requiresGrad = true;
    nn.embedding(emFreqIdx(), w, pad, null, 2, flag).mul(emFreqSeed()).sum().backward();
    return gradOf(w, "embedding/scale_grad_by_freq");
  };
  out.set("fname::embedding::scale_grad_by_freq 없이", emFreq(false));
  out.set("fname::embedding::scale_grad_by_freq", emFreq(true));
  // The padding row is zeroed as well as scaled, and zero over any count is zero — so
  // the two are asked together to say that neither undoes the other.
  out.set("fname::embedding::scale_grad_by_freq(padding_idx=1)", emFreq(true, 1));

  // **The layer stored the flag, printed it in `describe()`, and never read it.** The
  // constructor refused it, so the unread field could not be seen.
  const emFreqLayer = (flag: boolean) => () => {
    const e = new nn.Embedding(4, 3, null, null, 2, flag, false, emTable());
    e.call(emFreqIdx()).mul(emFreqSeed()).sum().backward();
    return gradOf(e.weight, "nn.Embedding/scale_grad_by_freq");
  };
  out.set("fname::embedding::Embedding(scale_grad_by_freq) 없이",
    emFreqLayer(false));
  out.set("fname::embedding::Embedding(scale_grad_by_freq)", emFreqLayer(true));

  // **The flag touches the gradient and must leave the forward alone.** The scaling
  // here puts the table through an expression that is the identity in value, so a
  // wrong expression would move the values too, and nothing else looks at that.
  //
  // **What this cannot see is the last bit.** The obvious alternative expression,
  // `w·k + w.detach()·(1−k)`, is not the identity in float32 — at `k = 1/3` a row
  // holding 15 comes back as 14.999999 — and that is 1e-7 relative against a golden
  // compared at `rtol = 1e-4`, so the wrong form was planted and passed. This table
  // holds the values where the perturbation is largest anyway.
  out.set("fname::embedding::scale_grad_by_freq 는 값을 안 건드린다",
    () => nn.embedding(
      emFreqIdx(),
      Tensor.from([1, 2, 3, 15, 27, 45, 7, 8, 9, 0, 0, 0], [4, 3]),
      null, null, 2, true));

  // **torch disagrees with itself on the bag**, so this one stays a refusal:
  // `embeddingBag(mode: "sum")` is `embedding(...).sum(1)` by definition, and under
  // this flag the two give different gradients on torch 2.13.0.
  const ebRefuses = (flag: string, body: () => unknown) => {
    out.set(`fname::embedding_bag::${flag}=우리는거절`, () => {
      try {
        body();
      } catch (err) {
        return (err as Error).message.includes("is not in the browser subset")
          ? "기대대로"
          : `다른 문구 <${(err as Error).message.slice(0, 44)}>`;
      }
      return "뜻밖의 성공";
    });
  };
  ebRefuses("scale_grad_by_freq",
    () => nn.embeddingBag(ebIdx(), ebTable(), null, null, 2, true, "sum"));
  ebRefuses("sparse",
    () => nn.embeddingBag(ebIdx(), ebTable(), null, null, 2, false, "mean", true));

  // ── `interpolate`'s other three modes ─────────────────────────────────
  //
  // `mode` was `"nearest" | "bilinear"` — a union, so anything else was a compile
  // error rather than a wrong answer. Three of torch's rest were a short distance
  // away: `area` is `adaptivePool("avg", …)` bit for bit, `nearest-exact` is nearest
  // measured from the output cell's centre, and `bicubic` is Keys' kernel at torch's
  // `a = −0.75`. `linear` and `trilinear` want a rank this function does not take.
  //
  // **The output size is asked three ways.** At a whole-number enlargement `nearest`,
  // `nearest-exact` and `area` all agree, so a case asking only `scaleFactor: 2` would
  // pass with the three wired to one body; the size that divides evenly into neither
  // is what separates them.
  const interpImg = () => Tensor.from(
    Array.from({ length: 20 }, (_, i) => i + 1), [1, 1, 4, 5]);
  const interpSizes: [number, number][] = [[2, 3], [8, 10], [3, 7]];
  for (const mode of ["area", "nearest-exact"] as const) {
    for (const s of interpSizes) {
      out.set(`fname::interpolate(${mode}, size=(${s[0]}, ${s[1]}))`,
        () => interpImg().interpolate(s, null, mode));
    }
    for (const sf of [0.5, 1.5]) {
      out.set(`fname::interpolate(${mode}, scale=${sf})`,
        () => interpImg().interpolate(null, sf, mode));
    }
  }
  // `alignCorners` both ways: the two coordinate rules part at the edges and agree
  // closely in the middle, which is the kind of difference only a value sees.
  for (const ac of [false, true]) {
    const tag = ac ? "True" : "False";
    for (const s of interpSizes) {
      out.set(`fname::interpolate(bicubic, align=${tag}, size=(${s[0]}, ${s[1]}))`,
        () => interpImg().interpolate(s, null, "bicubic", ac));
    }
    out.set(`fname::interpolate(bicubic, align=${tag}, scale=1.5)`,
      () => interpImg().interpolate(null, 1.5, "bicubic", ac));
  }
  // **The gradient is where a resampling is usually wrong**, and each of the three is
  // a gather with constant weights, so nothing was written for the backward — which is
  // exactly when a wrong forward index goes unnoticed, the same wrong index being used
  // on the way back.
  const interpGrad = (mode: "area" | "nearest-exact" | "bicubic", ac = false) => () => {
    const x = interpImg();
    x.requiresGrad = true;
    x.interpolate([3, 7], null, mode, ac)
      .mul(Tensor.from(Array.from({ length: 21 }, (_, i) => i + 1), [1, 1, 3, 7]))
      .sum().backward();
    return gradOf(x, `interpolate/${mode}`);
  };
  out.set("fname::interpolate(area) 의 기울기", interpGrad("area"));
  out.set("fname::interpolate(nearest-exact) 의 기울기", interpGrad("nearest-exact"));
  out.set("fname::interpolate(bicubic) 의 기울기", interpGrad("bicubic"));
  out.set("fname::interpolate(bicubic, align=True) 의 기울기",
    interpGrad("bicubic", true));

  // ── `antialias`, the last of `interpolate`'s seats ─────────────────────
  //
  // It widens the filter by the shrink factor and renormalises the weights, so every
  // input cell reaches the output instead of being skipped. Enlarging, the scale is
  // below one and the weights are the plain ones — which is why torch says the flag
  // does nothing going up, and why the sizes include one.
  //
  // **Two things in torch's own implementation disagree with the rest of torch**, and
  // both are asked rather than reasoned about: the anti-aliased `bicubic` uses
  // `a = −0.5` where the plain one uses `−0.75`, and `alignCorners` reaches the scale
  // but not the centre. Fitting either the other way parts by more than a tolerance.
  for (const mode of ["bilinear", "bicubic"] as const) {
    for (const ac of [false, true]) {
      const tag = ac ? "True" : "False";
      for (const s of interpSizes) {
        out.set(
          `fname::interpolate(${mode}, antialias, align=${tag}, size=(${s[0]}, ${s[1]}))`,
          () => interpImg().interpolate(s, null, mode, ac, null, true));
      }
      // The caller's scale is read when `alignCorners` is off and not when it is on —
      // the pair is what says so, and a size alone cannot.
      out.set(`fname::interpolate(${mode}, antialias, align=${tag}, scale=1.5)`,
        () => interpImg().interpolate(null, 1.5, mode, ac, null, true));
    }
    // A matrix multiply per axis, so the backward is the multiply's own — and that is
    // when a wrong weight matrix goes unnoticed, the same weights being used back.
    out.set(`fname::interpolate(${mode}, antialias) 의 기울기`, () => {
      const x = interpImg();
      x.requiresGrad = true;
      x.interpolate([3, 7], null, mode, false, null, true)
        .mul(Tensor.from(Array.from({ length: 21 }, (_, i) => i + 1), [1, 1, 3, 7]))
        .sum().backward();
      return gradOf(x, `interpolate/antialias/${mode}`);
    });
  }
  // torch restricts the flag to the two smooth modes and refuses the rest by name.
  // Unlike the unknown-`mode` case, this one compiles — `antialias` is a boolean and
  // `"nearest"` is in the union — so it can be asked here.
  out.set("fname::interpolate(nearest 에 antialias)=둘 다 거절", () => {
    try {
      interpImg().interpolate([2, 3], null, "nearest", false, null, true);
    } catch {
      return "둘 다 멈춘다";
    }
    return "여기선 통과했다";
  });

  // ── the ranks, which were an unpacking error on the Python side ────────────
  //
  // Carried verbatim from `tests/cases.py`. `linear` and `trilinear` were refused as
  // wanting *a rank this function does not take*; they are one separable kernel at
  // one and three axes, and `nearest`/`area` at those ranks were never refused at all
  // — the core failed at them with an unpacking error and this side read
  // `shape[2]`/`shape[3]` and answered about a rank it had not been given.
  const rankImg = (dims: number[]): Tensor => {
    const n = dims.reduce((a, b) => a * b, 1);
    return Tensor.from(Array.from({ length: n }, (_, i) => i + 1), dims);
  };
  const RANKS: Array<[number, number[], "linear" | "trilinear"]> = [
    [3, [2, 3, 5], "linear"],
    [5, [2, 3, 2, 3, 4], "trilinear"],
  ];
  for (const [rank, dims, lin] of RANKS) {
    for (const mode of ["nearest", "nearest-exact", "area"] as const) {
      for (const sf of [2, 1.7, 0.5]) {
        out.set(`fname::interpolate(${rank}차원 ${mode}, scale=${sf})`,
          () => rankImg(dims).interpolate(null, sf, mode));
      }
      out.set(`fname::interpolate(${rank}차원 ${mode}, size=3)`,
        () => rankImg(dims).interpolate(3, null, mode));
    }
    for (const ac of [false, true]) {
      const tag = ac ? "True" : "False";
      for (const sf of [2, 1.7]) {
        out.set(`fname::interpolate(${lin}, align=${tag}, scale=${sf})`,
          () => rankImg(dims).interpolate(null, sf, lin, ac));
      }
      out.set(`fname::interpolate(${lin}, align=${tag}, size=3)`,
        () => rankImg(dims).interpolate(3, null, lin, ac));
    }
    for (const mode of [lin, "nearest", "area"] as const) {
      out.set(`fname::interpolate(${rank}차원 ${mode}) 의 기울기`, () => {
        const x = rankImg(dims);
        x.requiresGrad = true;
        const got = x.interpolate(null, 2, mode, false);
        got.mul(got).sum().backward();
        return gradOf(x, `interpolate/${mode}`);
      });
    }
  }
  // **`recomputeScaleFactor` reaches `nearest` too.** The kernel maps
  // `floor(o·in/out)`, which is the *recomputed* rule, and the comment beside it said
  // the flag made no difference here. It does whenever the flooring loses something:
  // 5 at 1.7 gives 8, and 8/5 is 1.6.
  for (const mode of ["nearest", "nearest-exact"] as const) {
    for (const rsf of [true, false]) {
      out.set(
        `fname::interpolate(${mode}, scale=1.7, recompute=${rsf ? "True" : "False"})`,
        () => interpImg().interpolate(null, 1.7, mode, false, rsf));
    }
  }
  // The mode-against-rank refusals. `alignCorners` on a mode with no corners is a
  // Python-side rule — borch.ts takes a boolean, so *not given* and *given as false*
  // are the same word by the time they arrive.
  for (const [rank, dims] of [[3, [2, 3, 5]], [4, [1, 1, 4, 5]],
                              [5, [2, 3, 2, 3, 4]]] as Array<[number, number[]]>) {
    for (const mode of ["linear", "bilinear", "bicubic", "trilinear"] as const) {
      const fits = (rank === 3 && mode === "linear")
        || (rank === 4 && (mode === "bilinear" || mode === "bicubic"))
        || (rank === 5 && mode === "trilinear");
      if (fits) continue;
      out.set(`fname::interpolate(${rank}차원 ${mode})=둘 다 거절`, () => {
        try {
          rankImg(dims).interpolate(null, 2, mode);
        } catch {
          return "둘 다 멈춘다";
        }
        return "여기선 통과했다";
      });
    }
  }

  // **One scale per axis.** `scaleFactor` held a single number here, so the second
  // axis silently took the first one's factor. `(2, 1.5)` on a 4×5 gives 8×7 and one
  // number would give 8×10 or 6×7 — loud, but only because the fixture's two axes
  // differ, and every case above passes the same scale to both.
  for (const mode of ["nearest", "nearest-exact", "area", "bilinear",
                      "bicubic"] as const) {
    out.set(`fname::interpolate(${mode}, 축마다 다른 배율)`,
      () => interpImg().interpolate(null, [2, 1.5], mode, false));
  }
  out.set("fname::interpolate(bilinear, antialias, 축마다 다른 배율)",
    () => interpImg().interpolate(null, [0.6, 0.4], "bilinear", false, null, true));

  // ── The spatial transformer ───────────────────────────────────────────
  //
  // The two traps are written at length on the Python side. Briefly: asked with squares
  // alone the grid's `(x, y)` order is invisible, and **the gradient has to be asked inside
  // a cell** — a 90° rotation lands the grid exactly on a cell boundary and `floor` flips
  // on a difference of 6e-8.
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

  // ── `bicubic`, the third mode ─────────────────────────────────────────
  //
  // The same Keys kernel `interpolate`'s plain `bicubic` uses, at `a = −0.75` — not
  // the anti-aliased path's `−0.5`, which is the constant next door.
  //
  // **The padding lands on the tap, not on the centre.** Bilinear clamps the
  // continuous coordinate once and both its corners are inside; a 4×4 window steps
  // one cell further, and clamping the centre and then masking gave `border` the same
  // numbers as `zeros`. So the out-of-range grid is asked under all three modes.
  const gridPads = ["zeros", "border", "reflection"] as const;
  for (const pad of gridPads) {
    for (const ac of [false, true]) {
      out.set(
        `fname::grid_sample(bicubic, padding=${pad}, align=${ac ? "True" : "False"})`,
        () => nn.gridSample(img3(), outGrid(), "bicubic", pad, ac));
    }
  }
  out.set("fname::grid_sample(bicubic, 반 칸)",
    () => nn.gridSample(img3(), halfGrid(), "bicubic", "zeros", false));
  // **The gradient has to reach the grid** — the path a spatial transformer learns
  // `theta` along. The cubic weights are tensor expressions for that reason alone;
  // as constants the values would all still be right and this would be zero.
  for (const which of ["input", "grid"] as const) {
    for (const pad of gridPads) {
      out.set(`fname::grid_sample(bicubic, grad ${which}, padding=${pad})`, () => {
        const x = img3();
        const g = outGrid();
        x.requiresGrad = true;
        g.requiresGrad = true;
        nn.gridSample(x, g, "bicubic", pad, false).sum().backward();
        return gradOf(which === "input" ? x : g, `grid_sample/bicubic/${which}`);
      });
    }
  }

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

  // ── multi_head_attention_forward ──────────────────────────────────────
  //
  // **The input is `(L, N, E)` — length first.** The same numbers as the Python side are
  // required, so the weights are the ones the golden brought (they are in `inputs`).
  //
  // Here the mask is **an added float** and nothing else. Turning a boolean into a float is
  // done by the Python binding, which imitates torch's contract — the library itself takes
  // one kind.
  // The weights are **not random** — they are counted values, so both sides build the
  // same.
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
    mhaQ(), mhaK(), mhaV(), null, 2, mhaInW(), mhaInB(), null, null, false, 0,
    mhaOutW(), mhaOutB(), true, opts.pad ?? null, true, opts.mask ?? null,
    false, null, null, null, null, null, opts.average ?? true);

  out.set("fname::mha::출력", () => runMha().output);
  out.set("fname::mha::가중치(머리 평균)", () => runMha().weights!);
  out.set("fname::mha::가중치(머리마다)",
    () => runMha({ average: false }).weights!);
  // The causal mask — the upper triangle at -inf. An added float rather than a
  // boolean.
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
    () => runMha({ pad: padAdd() }).weights!);
  out.set("fname::mha::grad(query)", () => {
    const q = mhaQ(true);
    nn.multiHeadAttentionForward(q, mhaK(), mhaV(), null, 2, mhaInW(), mhaInB(),
      null, null, false, 0, mhaOutW(), mhaOutB()).output.sum().backward();
    return gradOf(q, "multiHeadAttentionForward");
  });
}

/**
 * The nine remaining layers — the two that unfold windows, and the rest.
 *
 * The four traps are written in the Python side's `misc_cases` — briefly: `fold` adds the
 * overlapping positions, `LocalResponseNorm`'s window leans left, `RReLU`'s slope is fixed
 * in evaluation mode, and `UpsamplingBilinear2d` is `alignCorners=true`.
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

    // **`null, 2` is `size` then `scaleFactor`.** These read `(2)` while the
    // constructor took a scale alone; `size` leads now, as in torch, and `2` on its
    // own became a 2×2 target. Both are numbers, so the compiler had nothing to say —
    // the row `test_binding_arguments` calls *number into a number slot*.
    // 3×5 from 2×2 — not a whole multiple, so it is an answer no scale gives.
    ["층::UpsamplingNearest2d(크기)",
      () => new nn.UpsamplingNearest2d([3, 5]).call(small())],
    ["층::UpsamplingBilinear2d(크기)",
      () => new nn.UpsamplingBilinear2d([3, 5]).call(small())],
    ["층::UpsamplingNearest2d",
      () => new nn.UpsamplingNearest2d(null, 2).call(small())],
    ["층::UpsamplingBilinear2d",
      () => new nn.UpsamplingBilinear2d(null, 2).call(small())],
    ["UpsamplingBilinear2d 는 align_corners=True",
      () => new nn.UpsamplingBilinear2d(null, 2).call(small())
        .sub(small().interpolateBilinear(4, 4, true))],
  ];
  for (const [name, fn] of value) out.set(`misc::${name}`, fn);

  out.set("misc::grad::unfold", () => {
    const x = Tensor.from(Array.from({ length: 16 }, (_, i) => i),
      [1, 1, 4, 4], { requiresGrad: true });
    seeded(x.unfoldIm2col(2)).backward();
    return gradOf(x, "unfold");
  });

  // ── the unbatched rank, which the refusal had half right ──────────────
  //
  // The core said *anything but 4-D*, and torch takes 3-D as one unbatched sample:
  // `(C, H, W)` comes back as `(C·kh·kw, L)` with no batch axis. `fold`'s unbatched
  // form is one rank lower again — 2-D, because the channel and the kernel are already
  // one axis there — and the pair is asked together because taking them for the same
  // number is the mistake available.
  const uf3d = () => Tensor.from(
    Array.from({ length: 24 }, (_, i) => i), [2, 3, 4]);
  const ufOpts: [string, [number, number, number]][] = [
    ["기본", [1, 0, 1]], ["stride=2", [1, 0, 2]],
    ["padding=1", [1, 1, 1]], ["dilation=2", [2, 0, 1]],
  ];
  for (const [tag, [dil, pad, stride]] of ufOpts) {
    out.set(`misc::unfold(배치 없이, ${tag})`,
      () => uf3d().unfoldIm2col(2, dil, pad, stride));
  }
  out.set("misc::fold(배치 없이)",
    () => uf3d().unfoldIm2col(2).fold([3, 4], 2));
  // The unbatched path goes through the batched one and reshapes, so the gradient has
  // to come back out of that reshape with the batch axis gone — a value case cannot
  // see a gradient that kept it.
  out.set("misc::grad::unfold(배치 없이)", () => {
    const x = uf3d();
    x.requiresGrad = true;
    seeded(x.unfoldIm2col(2)).backward();
    return gradOf(x, "unfold/unbatched");
  });
  // torch's wording, compared as a string — an invented message would pass a check
  // that only asks whether something was raised.
  const rankRefusal = (name: string, body: () => unknown) => {
    out.set(`misc::${name}=거절 문구`, () => {
      try {
        body();
      } catch (err) {
        const head = (err as Error).message.split(",")[0] ?? "";
        return head.startsWith("Expected")
          ? head : `다른 문구 <${(err as Error).message.slice(0, 44)}>`;
      }
      return "안 던졌다";
    });
  };
  rankRefusal("unfold(5차원)", () => Tensor.zeros([2, 3, 4, 5, 6]).unfoldIm2col(2));
  rankRefusal("fold(4차원)", () => Tensor.zeros([2, 3, 8, 6]).fold([3, 4], 2));

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
      const layer = new nn.EmbeddingBag(5, 3, null, 2, false, mode);
      layer.loadStateDict({ weight: Tensor.from(table, [5, 3]) });
      return layer.call(bags());
    });
  }

  out.set("misc::층::EmbeddingBag(offsets)", () => {
    const layer = new nn.EmbeddingBag(5, 3, null, 2, false, "sum");
    layer.loadStateDict({ weight: Tensor.from(table, [5, 3]) });
    return layer.callOffsets(Tensor.from([0, 1, 2, 3], [4]), [0, 2]);
  });

  // ── `maxNorm`, and the case that has to look at the table ──
  //
  // **`maxNorm` shortens rows in the table itself, and every other instrument here
  // compares what a call returns.** A call that also changes the thing it was called on
  // is invisible to all of them.
  //
  // And calling twice does not reveal it. Re-normalising a row already short enough is
  // a no-op, so an implementation that normalised a **copy** would return these same
  // numbers forever — the divergence lives in `weight` and nowhere else, until
  // something trains on it. That is why the second row of each pair reads the table
  // after the call rather than the output.
  const bagRenorm = (what: "out" | "weight", normType = 2): Tensor => {
    const layer = new nn.EmbeddingBag(5, 3, 1.0, normType, false, "sum");
    layer.loadStateDict({ weight: Tensor.from(table, [5, 3]) });
    const out2 = layer.call(bags());
    return what === "weight" ? layer.weight : out2;
  };
  out.set("misc::층::EmbeddingBag(max_norm)", () => bagRenorm("out"));
  out.set("misc::층::EmbeddingBag(max_norm)/표가 줄었다", () => bagRenorm("weight"));
  // **Only the rows that were looked up shrink.** Row 4 is never asked for and keeps
  // its own length — an implementation that normalised the whole table agrees on every
  // output and differs only here.
  out.set("misc::층::EmbeddingBag(max_norm)/색인 안 된 행", () => {
    const layer = new nn.EmbeddingBag(5, 3, 1.0, 2, false, "sum");
    layer.loadStateDict({ weight: Tensor.from(table, [5, 3]) });
    layer.call(Tensor.from([0, 1], [1, 2]));
    return layer.weight;
  });
  // `normType=1` measures a row differently, so a different amount comes off. Left at
  // the default this argument is unasked, and an implementation ignoring it passes both
  // rows above.
  out.set("misc::층::EmbeddingBag(max_norm, norm_type=1)", () => bagRenorm("weight", 1));

  // ── Embedding: the same two arguments, on the layer next door ──
  const emb = (what: "out" | "weight", paddingIdx: number | null = null,
               maxNorm: number | null = null, normType = 2): Tensor => {
    const layer = new nn.Embedding(4, 3, paddingIdx, maxNorm, normType);
    layer.loadStateDict({ weight: Tensor.from(table.slice(0, 12), [4, 3]) });
    const out2 = layer.call(Tensor.from([0, 1, 2], [3]));
    return what === "weight" ? layer.weight : out2;
  };
  out.set("misc::층::Embedding(padding_idx)", () => emb("out", 1));
  out.set("misc::층::Embedding(max_norm)", () => emb("out", null, 1.0));
  out.set("misc::층::Embedding(max_norm)/표가 줄었다", () => emb("weight", null, 1.0));
  out.set("misc::층::Embedding(max_norm, norm_type=1)/표",
    () => emb("weight", null, 1.0, 1));
  out.set("misc::repr::Embedding(전부)",
    async () => new nn.Embedding(4, 3, 1, 2.0).describe());

  // **The padding row learns nothing.** Left in, a pad token drifts toward whatever the
  // loss wants and the mask stops meaning "ignore this" — while the forward stays
  // right, so an implementation that masks the *output* instead passes every value case
  // above and fails only here.
  out.set("misc::grad::Embedding(padding_idx)", () => {
    const layer = new nn.Embedding(4, 3, 1);
    layer.loadStateDict({ weight: Tensor.from(table.slice(0, 12), [4, 3]) });
    layer.call(Tensor.from([0, 1, 2, 1], [4])).sum().backward();
    return gradOf(layer.weight, "Embedding(padding_idx)");
  });

  // A **fresh** table zeroes the padding row and a **given** one does not. torch draws
  // the line at the same place, and both halves are asked because either alone reads as
  // a rule about padding rather than about who supplied the weights.
  out.set("misc::층::Embedding(padding_idx)/새 표와 준 표", async () => {
    const fresh = new nn.Embedding(4, 3, 2).weight;
    const given = new nn.Embedding(4, 3, 2, null, 2, false, false,
      Tensor.from(table.slice(0, 12), [4, 3])).weight;
    return Tensor.cat([fresh.select(0, 2), given.select(0, 2)], 0);
  });

  for (const [name, make] of [
    ["Bilinear", () => new nn.Bilinear(3, 4, 2)],
    ["LocalResponseNorm", () => new nn.LocalResponseNorm(2)],
    ["Softmax2d", () => new nn.Softmax2d()],
    ["RReLU", () => new nn.RReLU()],
    // **`RReLU`'s defaults are `1/8` and `1/3`, so the row above can never see a
    // missing decimal point.** JavaScript prints `1.0` as `1`, and both of these
    // were interpolated bare — the row stayed green because no value it was given
    // was ever integral. That is not a case for `lower` and `upper`; it is a case
    // for torch having picked fractional defaults.
    ["RReLU(정수 경계)", () => new nn.RReLU(1.0, 2.0)],
    ["EmbeddingBag", () => new nn.EmbeddingBag(5, 3)],
  ] as const) {
    out.set(`misc::repr::${name}`,
      async () => (make() as unknown as { describe(): string }).describe());
  }

  // **The row above builds it on the defaults, and the default `alpha` is 1e-4.**
  // That prints the same in both languages, so it could never see that `describe`
  // interpolated `alpha` without the decimal-point guard `beta` and `k` went through.
  // At a whole number torch says `alpha=1.0` and this said `alpha=1`.
  out.set("misc::repr::LocalResponseNorm(alpha=1.0)",
    async () => new nn.LocalResponseNorm(2, 1.0, 2.0, 2.0).describe());
}

/**
 * The three layers that move elements, and the five dropouts that drop by channel.
 *
 * Where no draw is involved it is asked by value, and where one is it is asked by property
 * — the details are written in `tests/cases.py`'s `shuffle_cases`.
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
    // **`Flatten`'s two seats, hidden by their own defaults.** torch's are `1` and
    // `-1` — keep the batch axis, fold the rest — which is what this did with no
    // arguments at all, so every case using the default passed while `Flatten(0)`
    // and `Flatten(1, 2)` were words JavaScript dropped.
    ["층::Flatten기본", () => new nn.Flatten().call(pix())],
    ["층::Flatten(0, -1)", () => new nn.Flatten(0, -1).call(pix())],
    ["층::Flatten(1, 2)", () => new nn.Flatten(1, 2).call(pix())],
    ["층::Flatten(2, 3)", () => new nn.Flatten(2, 3).call(pix())],
    ["층::Flatten(0, 1)", () => new nn.Flatten(0, 1).call(pix())],
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

  // Where no draw is involved, by value.
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

  // Where a draw is involved, by property.
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
    // The digits are pinned — Python gives `2.0` and JS gives `2`.
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
 * The layers that learn their shape at the first forward.
 *
 * **Once it sets, it becomes something else.** The Python side changes the class and this
 * side swaps the prototype — the same place pointed at in two languages. What a user sees
 * is `print(model)`, so the golden asks about those characters. The details are in
 * `tests/cases.py`'s `lazy_cases`.
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

  // Whether the seed reaches layer initialisation and dropout. The core produced a defect
  // here — the account is in the Python side's comment.
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
 * Thirteen losses and three distances.
 *
 * The three traps are written in the Python side's `loss_cases` — briefly: `huber(δ)` is δ
 * times `smooth_l1(β=δ)` (they coincide only at δ=1), `KLDivLoss`'s `mean` and `batchmean`
 * divide by different numbers, and `pairwise_distance`'s `eps` is added **to the
 * difference** rather than to the result.
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
    // Where the target is not ±1 — **both** terms switch on.
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
    // **A distance that is not the default**, which is the only reason this name
    // exists apart from `tripletMarginLoss`. The two above pass null and get the
    // pairwise distance, so they agree with the plain form — and would agree just as
    // well with an implementation that ignored the argument.
    ["triplet_with_distance(L1)",
      () => F.tripletMarginWithDistanceLoss(
        anc(), pos(), neg(), (u, v) => u.sub(v).abs().sumDim(-1))],

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
    // In torch these two are **literally the same function** at the top level and under
    // `F`. The seven losses that surfaced alongside are raw ATen operations at the top
    // level with different signatures, so they are not exported.
    ["최상위::pairwise_distance", () => a().pairwiseDistance(b())],
    ["최상위::pdist", () => Tensor.from([0, 0, 3, 4, 1, 1], [3, 2]).pdist()],
    // Folding a single element — the reason is written in the Python side's comment. This
    // is where 0 used to come out.
    ["원소 하나를 mean",
      () => Tensor.from([1, 2, 3], [3]).sum().binary("mul", Tensor.full([], 1))
        .reshape([1]).mean()],
    ["원소 하나를 sum",
      () => Tensor.from([1, 2, 3], [3]).sum().binary("mul", Tensor.full([], 1))
        .reshape([1]).sum()],
  ];
  for (const [name, fn] of value) out.set(`loss::${name}`, fn);

  // ── `linearCrossEntropy`, which is a fusion and not a formula ────────────
  //
  // It equals `crossEntropy(linear(x, w, b), t)` and the cases are about the arguments
  // arriving on the right side of that. **`ignoreIndex` is the one a lazy port loses**:
  // torch's default here is `null`, `crossEntropy`'s is `-100`, and `null` is measured
  // to *mean* -100 — so passing it through is the same answer on every target that
  // holds no -100 and a different one on the targets the argument exists for.
  const lceX = [0.5, -1.2, 0.3, 2.0, -0.7,
                1.1, 0.4, -0.9, 0.2, 1.5,
                -0.3, 0.8, 1.7, -1.1, 0.6,
                2.2, -0.5, 0.1, 0.9, -1.4,
                0.7, 1.3, -0.2, -0.8, 0.4,
                -1.0, 0.2, 0.9, 1.6, -0.3];
  const lceW = [0.3, -0.7, 1.1, 0.2, -0.4,
                -1.2, 0.5, 0.8, -0.9, 1.3,
                0.6, 1.0, -0.3, 0.7, -1.1,
                -0.8, -0.2, 0.4, 1.5, 0.9];
  const lceB = [0.2, -0.5, 0.9, -0.1];
  const lceCw = [0.7, 1.4, 0.9, 1.1];
  const lx = (): Tensor => Tensor.from(lceX, [6, 5]);
  const lw = (): Tensor => Tensor.from(lceW, [4, 5]);
  const lt = (): Tensor => Tensor.from([0, 1, 2, 3, 0, 1], [6], { dtype: "int64" });

  const lce: [string, () => Tensor][] = [
    ["기본", () => F.linearCrossEntropy(lx(), lw(), lt())],
    ["linear_bias",
      () => F.linearCrossEntropy(lx(), lw(), lt(), Tensor.from(lceB, [4]))],
    ["weight", () => F.linearCrossEntropy(lx(), lw(), lt(), undefined,
                                          Tensor.from(lceCw, [4]))],
    ["reduction=sum",
      () => F.linearCrossEntropy(lx(), lw(), lt(), undefined, undefined, "sum")],
    ["reduction=none",
      () => F.linearCrossEntropy(lx(), lw(), lt(), undefined, undefined, "none")],
    ["label_smoothing",
      () => F.linearCrossEntropy(lx(), lw(), lt(), undefined, undefined, "mean",
                                 null, 0.1)],
    ["ignore_index",
      () => F.linearCrossEntropy(lx(), lw(), lt(), undefined, undefined, "mean", 1)],
    ["전부",
      () => F.linearCrossEntropy(lx(), lw(), lt(), Tensor.from(lceB, [4]),
                                 Tensor.from(lceCw, [4]), "sum", 2, 0.1)],
    ["options",
      () => F.linearCrossEntropy(lx(), lw(), lt(), undefined, undefined, "mean",
                                 null, 0.0,
                                 new nn.LinearCrossEntropyOptions(false, 2))],
    // The target that actually holds -100. Without it the pair above says nothing
    // about the default, because *ignore nothing* and *ignore class -100* agree
    // everywhere else.
    ["기본이 -100 을 건너뛴다",
      () => F.linearCrossEntropy(
        lx(), lw(),
        Tensor.from([0, -100, 2, 3, -100, 1], [6], { dtype: "int64" }))],
  ];
  for (const [name, fn] of lce) out.set(`loss::linear_cross_entropy(${name})`, fn);

  // ── `special::수학::`, the eighteen that need no shader ──────────────────
  //
  // Twelve orthogonal recurrences and six compositions. **The polynomials are asked
  // outside their orthogonality interval** — Chebyshev's `T` is defined on [-1, 1] and
  // torch evaluates the polynomial anywhere, so an implementation that clamped would
  // agree on every textbook input and part at x = 2. A negative order is 0 and is asked.
  const polyX = [-2.0, -1.5, -0.6, 0.0, 0.3, 0.9, 1.0, 2.0, 3.7];
  const px = (): Tensor => Tensor.from(polyX, [9]);
  const polys: [string, (x: Tensor, n: number) => Tensor][] = [
    ["chebyshev_polynomial_t", special.chebyshevPolynomialT],
    ["chebyshev_polynomial_u", special.chebyshevPolynomialU],
    ["chebyshev_polynomial_v", special.chebyshevPolynomialV],
    ["chebyshev_polynomial_w", special.chebyshevPolynomialW],
    ["shifted_chebyshev_polynomial_t", special.shiftedChebyshevPolynomialT],
    ["shifted_chebyshev_polynomial_u", special.shiftedChebyshevPolynomialU],
    ["shifted_chebyshev_polynomial_v", special.shiftedChebyshevPolynomialV],
    ["shifted_chebyshev_polynomial_w", special.shiftedChebyshevPolynomialW],
    ["hermite_polynomial_h", special.hermitePolynomialH],
    ["hermite_polynomial_he", special.hermitePolynomialHe],
    ["laguerre_polynomial_l", special.laguerrePolynomialL],
    ["legendre_polynomial_p", special.legendrePolynomialP],
  ];
  for (const [name, fn] of polys) {
    // n = 5 is past where a wrong second term or step coefficient has compounded into
    // something visible; 0, 1 and -1 are the three the recurrence never reaches.
    for (const n of [0, 1, 5, -1]) {
      out.set(`special::수학::${name}(n=${n})`, () => fn(px(), n));
    }
  }

  out.set("special::수학::ndtr",
          () => special.ndtr(Tensor.from([-8.0, -3.0, 0.0, 1.0, 5.0], [5])));
  out.set("special::수학::ndtri",
          () => special.ndtri(Tensor.from([0.001, 0.1, 0.5, 0.9, 0.999], [5])));
  out.set("special::수학::entr",
          () => special.entr(Tensor.from([-1.0, 0.0, 0.25, 0.5, 1.0, 2.0], [6])));
  out.set("special::수학::spherical_bessel_j0",
          () => special.sphericalBesselJ0(Tensor.from([-3.0, 0.0, 0.5, 2.0, 10.0], [5])));
  out.set("special::수학::xlog1py",
          () => special.xlog1py(Tensor.from([1.0, 1.0, 1.0, 0.0, 2.0, -1.0], [6]),
                                Tensor.from([1.0, 1e-8, 1e-12, -1.0, 0.5, 3.0], [6])));
  for (const p of [1, 2, 3]) {
    out.set(`special::수학::multigammaln(p=${p})`,
            () => special.multigammaln(Tensor.from([2.5, 3.0, 4.5], [3]), p));
  }

  // **The sixteen that carry a kernel, at the inputs the kernels exist for.**
  // A middle-of-the-range value passes against the composition each of these replaces,
  // so the numbers here are the ones where that composition is `inf` or `nan`:
  // `erfc(x)·exp(x²)` from x=10, `log(ndtr(x))` from x=-6, `i0(x)·exp(-|x|)` at x=90.
  // The `k` pair is asked either side of its seam at 10 — the first crossover tried was
  // 2 and was worth two digits, on those two points and nowhere else.
  const tails: [string, number[], (x: Tensor) => Tensor][] = [
    ["erfcx", [-5, -1, 0, 1, 10, 26, 100], special.erfcx],
    ["log_ndtr", [-40, -10, -6, -1, 0, 2], special.logNdtr],
    ["i0e", [0, 1, 15, 50, 90, 200], special.i0e],
    ["i1", [-5, -1, 0, 0.5, 2, 10], special.i1],
    ["i1e", [-200, -1, 0, 1, 15, 90], special.i1e],
    ["modified_bessel_i1", [-3, 0, 0.5, 2, 10], special.modifiedBesselI1],
    ["modified_bessel_k0", [0.01, 0.5, 2, 9.9, 10, 10.1, 20],
     special.modifiedBesselK0],
    ["modified_bessel_k1", [0.01, 0.5, 2, 9.9, 10, 10.1, 20],
     special.modifiedBesselK1],
    ["scaled_modified_bessel_k0", [0.01, 0.5, 2, 9.9, 10.1, 80],
     special.scaledModifiedBesselK0],
    ["scaled_modified_bessel_k1", [0.01, 0.5, 2, 9.9, 10.1, 80],
     special.scaledModifiedBesselK1],
    // `J` and `Y` change method at 8. `Y` has a pole at 0 and is `nan` below it;
    // `J₀` is even and `J₁` is odd, and a sign dropped there is invisible on the
    // positive half alone.
    ["bessel_j0", [-20, -8.1, -7.9, -1, 0, 2.4, 7.9, 8.1, 20], special.besselJ0],
    ["bessel_j1", [-20, -8.1, -7.9, -1, 0, 2.4, 7.9, 8.1, 20], special.besselJ1],
    ["bessel_y0", [0.001, 0.5, 2.4, 7.9, 8, 8.1, 20], special.besselY0],
    ["bessel_y1", [0.001, 0.5, 2.4, 7.9, 8, 8.1, 20], special.besselY1],
    // Airy's seam is at 8 too, and the negative side oscillates — the envelope and the
    // phase are two ways to be wrong and only the negative arguments show the second.
    ["airy_ai", [-30, -12, -8.1, -7.9, -1, 0, 1, 7.9, 8.1, 12], special.airyAi],
  ];
  for (const [name, xs, fn] of tails) {
    out.set(`special::수학::${name}`,
            () => fn(Tensor.from(xs, [xs.length])));
  }

  // `ζ(2,1)` is π²/6 and `ζ(4,1)` is π⁴/90, so two of these rows are checked by
  // arithmetic as well as by torch. `x ≤ 1` is `nan` and `ζ(1,1)` is `inf`.
  out.set("special::수학::zeta", () => special.zeta(
    Tensor.from([2, 3, 4, 2.5, 1.5, 6, 10, 2, 1, 0.5], [10]),
    Tensor.from([1, 1, 1, 2, 1, 3, 0.5, 10, 1, 1], [10])));

  // The layer, with the weight planted — its initialisation is not torch's.
  //
  // **Inside `noGrad`, because a parameter is a leaf that wants a gradient** and
  // writing into one is refused: *a leaf Variable that requires grad is being used in
  // an in-place operation*. Every other planted case in this file goes through the same
  // door; these two did not at first and said so on the first run.
  out.set("loss::nn.LinearCrossEntropyLoss", () => {
    const m = new nn.LinearCrossEntropyLoss(5, 4);
    noGrad(() => m.linear.weight.copyFrom(lw()));
    return m.call(lx(), lt());
  });
  out.set("loss::nn.LinearCrossEntropyLoss(bias)", () => {
    const m = new nn.LinearCrossEntropyLoss(5, 4, [], true);
    noGrad(() => {
      m.linear.weight.copyFrom(lw());
      if (m.linear.bias) m.linear.bias.copyFrom(Tensor.from(lceB, [4]));
    });
    return m.call(lx(), lt());
  });

  // **torch's deprecated pair, and it beats `reduction`.** The last two read wrongly
  // at a glance and are the point: all three given folds to the *mean*, and the
  // positional string lands on `sizeAverage`, which is truthy, so it folds there too.
  // The `자리::` group below runs the same fold under `F`.
  const legacy: [string, () => Tensor][] = [
    ["size_average=False", () => new nn.L1Loss(false).call(a(), b())],
    ["reduce=False", () => new nn.L1Loss(null, false).call(a(), b())],
    ["셋 다", () => new nn.L1Loss(true, true, "sum").call(a(), b())],
    // `"sum"` is not a boolean, so this one needs the cast TypeScript would
    // otherwise refuse — which is the whole hazard stated in the type system.
    ["L1Loss('sum') 위치",
      () => new nn.L1Loss("sum" as unknown as boolean).call(a(), b())],
  ];
  for (const [name, fn] of legacy) out.set(`loss::가장자리::${name}`, fn);

  // **The same pair, under `F` rather than on a layer.** Everything in `value`
  // above gives `reduction` by name, and a keyword survives a signature whose
  // seats are wrong — a position does not. `F`'s exports were short of torch's
  // list by exactly these two seats, so torch's own line was a type error here
  // and, on the Python binding reading the same table, a value silently at the
  // default: `F.soft_margin_loss(a, b, "sum")` set `size_average`, which is
  // neither null nor false, so the fold answered `mean`.
  const seats: [string, (r: "none" | "sum" | "mean") => Tensor][] = [
    ["soft_margin", (r) => F.softMarginLoss(x(), sgn(), null, null, r)],
    ["hinge_embedding",
      (r) => F.hingeEmbeddingLoss(x(), sgn(), 1.0, null, null, r)],
    ["multilabel_margin", (r) => F.multilabelMarginLoss(
      mm(), Tensor.from([2, -1, -1, 0, -1, -1], [2, 3]), null, null, r)],
    ["kl_div", (r) => F.klDiv(logp(), tgtp(), null, null, r)],
    // torch's third seat here **is** `reduction` — this one is newer and never
    // carried the pair, so nothing steps over anything and it agreed all along.
    // It is asked beside the others as the control.
    ["huber", (r) => F.huberLoss(x(), y(), r)],
  ];
  for (const [name, fn] of seats) {
    for (const r of ["none", "sum", "mean"] as const) {
      out.set(`loss::자리::${name}(${r})`, () => fn(r));
    }
  }
  // `multilabelSoftMarginLoss` takes a **`weight` in the third seat** on both
  // sides, so this is the one place in the group where the seat is filled rather
  // than stepped over.
  out.set("loss::자리::multilabel_soft_margin(weight)",
    () => F.multilabelSoftMarginLoss(
      Tensor.from([0.5, -1, 2], [1, 3]), Tensor.from([1, 0, 1], [1, 3]),
      Tensor.from([1, 2, 3], [3])));
  out.set("loss::자리::multilabel_soft_margin(weight, none)",
    () => F.multilabelSoftMarginLoss(
      Tensor.from([0.5, -1, 2], [1, 3]), Tensor.from([1, 0, 1], [1, 3]),
      Tensor.from([1, 2, 3], [3]), null, null, "none"));
  // **One row could not tell a kept batch from a collapsed one.** The single
  // `multilabel_margin` case above has one row, and the core meanwhile summed the
  // whole batch into one number — `none` handed back the sum and `mean` divided it
  // by one. This side kept the rows the whole time and had nobody to disagree with.
  out.set("loss::multilabel_margin(두 행, none)",
    () => F.multilabelMarginLoss(
      mm(), Tensor.from([2, -1, -1, 0, -1, -1], [2, 3]), null, null, "none"));
  // A row whose target starts at −1 has **no labels at all**: its loss is zero and
  // the rows around it must not shift up to fill the seat.
  out.set("loss::multilabel_margin(빈 행)",
    () => F.multilabelMarginLoss(
      mm(), Tensor.from([-1, -1, -1, 0, 1, -1], [2, 3]), null, null, "none"));

  // The class weights are unequal and none of them is 1: an equal weight makes the
  // two candidate denominators agree, and the case would pass either way.
  const clsW = () => Tensor.from([0.5, 2.0, 3.0], [3]);
  const clsLogits = () =>
    Tensor.from([2, 1, 0.1, 0.5, 2.5, 0.3, 1, 0.2, 3], [3, 3]);
  const clsT = () => Tensor.from([0, 1, 2], [3], { dtype: "int64" });
  const clsTIgn = () => Tensor.from([0, -100, 2], [3], { dtype: "int64" });
  const bwZ = () => Tensor.from([-1.0, 0.5, 2.0, -0.25], [4]);
  const bwT = () => Tensor.from([1, 0, 1, 0], [4]);
  const bwW = () => Tensor.from([0.5, 1.5, 2.0, 0.25], [4]);
  const bwPW = () => Tensor.from([2.5, 0.5, 1.0, 3.0], [4]);
  const bwP = () => Tensor.from(
    [-1.0, 0.5, 2.0, -0.25].map((v) => 1 / (1 + Math.exp(-v))), [4]);

  const layers: [string, () => Tensor][] = [
    // `delta` moved behind `reduction` to match torch. The Python case passes
    // `delta=0.5` by keyword, so its value never moved and the golden is unchanged —
    // which is exactly why nothing caught the order being wrong.
    ["HuberLoss", () => new nn.HuberLoss("mean", 0.5).call(x(), y())],
    ["KLDivLoss", () => new nn.KLDivLoss(null, null, "batchmean").call(logp(), tgtp())],
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
    ["MultiLabelSoftMarginLoss(weight)",
      () => new nn.MultiLabelSoftMarginLoss(Tensor.from([0.5, 2, 1.5], [3]))
        .call(Tensor.from([0.5, -1, 2], [1, 3]), Tensor.from([1, 0, 1], [1, 3]))],
    // `ignoreIndex` across all three folds. The three treat a skipped row
    // differently — `mean` drops it from the denominator, `sum` is unaffected,
    // `none` keeps it as a zero so the shape survives — and only the `none` case
    // can show the third, which is how the core's first version was caught coming
    // back one element short.
    ...(["mean", "sum", "none"] as const).map((r) =>
      [`CrossEntropyLoss(ignore_index, ${r})`,
        () => new nn.CrossEntropyLoss(undefined, null, -100, null, r).call(
          Tensor.from([2, 1, 0.1, 0.5, 2.5, 0.3, 1, 0.2, 3], [3, 3]),
          Tensor.from([0, -100, 2], [3], { dtype: "int64" }))] as
        [string, () => Tensor]),
    ["CrossEntropyLoss(label_smoothing)",
      () => new nn.CrossEntropyLoss(undefined, null, -100, null, "mean", 0.1).call(
        Tensor.from([2, 1, 0.1, 0.5, 2.5, 0.3, 1, 0.2, 3], [3, 3]),
        Tensor.from([0, 1, 2], [3], { dtype: "int64" }))],
    ["NLLLoss(ignore_index)",
      () => new nn.NLLLoss(undefined, null, -100).call(
        Tensor.from([0.7, 0.2, 0.1, 0.1, 0.8, 0.1, 0.2, 0.2, 0.6], [3, 3]).log(),
        Tensor.from([0, -100, 2], [3], { dtype: "int64" }))],
    // ── the class weight ──────────────────────────────────────────────────
    //
    // Refused on both sides until today, with a reason that was true: `mean` divides
    // by the sum of the weights, so a weight taken and dropped moves the loss
    // quietly. Carried verbatim from `_CLS_*` and `_BW_*` in `tests/cases.py` —
    // divergent values stop the comparison from being a comparison.
    ...(["mean", "sum", "none"] as const).map((r) =>
      [`CrossEntropyLoss(weight, ${r})`,
        () => new nn.CrossEntropyLoss(clsW(), null, -100, null, r)
          .call(clsLogits(), clsT())] as [string, () => Tensor]),
    ["CrossEntropyLoss(weight, ignore_index)",
      () => new nn.CrossEntropyLoss(clsW(), null, -100).call(
        clsLogits(), clsTIgn())],
    // The weight enters **inside** the spread: `(1−ε)·w[t]·nll + ε·mean_c(w_c·−log p_c)`.
    ["CrossEntropyLoss(weight, label_smoothing)",
      () => new nn.CrossEntropyLoss(clsW(), null, -100, null, "mean", 0.1)
        .call(clsLogits(), clsT())],
    ["NLLLoss(weight, ignore_index)",
      () => new nn.NLLLoss(clsW(), null, -100).call(
        Tensor.from([0.7, 0.2, 0.1, 0.1, 0.8, 0.1, 0.2, 0.2, 0.6], [3, 3]).log(),
        clsTIgn())],
    // The binary pair divides by the **count** instead, weights or no weights.
    ["BCEWithLogitsLoss(pos_weight)",
      () => new nn.BCEWithLogitsLoss(undefined, null, null, "mean", bwPW())
        .call(bwZ(), bwT())],
    ["BCEWithLogitsLoss(weight)",
      () => new nn.BCEWithLogitsLoss(bwW()).call(bwZ(), bwT())],
    ["BCEWithLogitsLoss(weight, pos_weight)",
      () => new nn.BCEWithLogitsLoss(bwW(), null, null, "mean", bwPW())
        .call(bwZ(), bwT())],
    ["BCELoss(weight)", () => new nn.BCELoss(bwW()).call(bwP(), bwT())],
    ["MultiMarginLoss", () => new nn.MultiMarginLoss().call(mm(), mmt())],
    ["MultiLabelMarginLoss", () => new nn.MultiLabelMarginLoss()
      .call(Tensor.from([0.1, 0.2, 0.4, 0.8], [1, 4]),
        Tensor.from([3, 0, -1, 1], [1, 4]))],
    ["PairwiseDistance", () => new nn.PairwiseDistance().call(a(), b())],
    ["CosineSimilarity", () => new nn.CosineSimilarity(1).call(a(), b())],
  ];
  for (const [name, fn] of layers) out.set(`loss::층::${name}`, fn);

  // ── How it folds is part of the loss ──────────────────────────────────
  const ceX = () => Tensor.from([0.5, -1, 2, 1.5, 0.25, -0.5], [2, 3]);
  // Carried verbatim from `_BCE_P` / `_BCE_T` in `tests/cases.py`. Divergent values
  // stop the comparison from being a comparison.
  const bceP = () => Tensor.from([0.2, 0.7, 0.9, 0.4, 0.15, 0.6], [2, 3]);
  const bceT = () => Tensor.from([0.0, 1.0, 1.0, 0.0, 0.0, 1.0], [2, 3]);
  const ceT = () => Tensor.from([2, 0], [2], { dtype: "int64" });
  const ceLogp = () => Tensor.from(
    [0.2, 0.5, 0.3, 0.6, 0.1, 0.3].map(Math.log), [2, 3]);
  //
  // `reduceAs` existed for a long time and **only `huberLoss` and `klDiv` used it.** The
  // four common ones had `.mean()` baked in. The core had the same hole in the same place,
  // and the reason the table could not see it is that textbooks use the default `mean`
  // alone.
  //
  // `nllLoss` and `crossEntropy` still produce a scalar only, so they are not here.
  for (const reduction of ["none", "mean", "sum"] as const) {
    const fns: [string, () => Tensor][] = [
      [`mse_loss(${reduction})`, () => x().mseLoss(y(), reduction)],
      [`l1_loss(${reduction})`, () => x().l1Loss(y(), reduction)],
      [`smooth_l1_loss(${reduction})`,
        () => x().smoothL1Loss(y(), 1.0, reduction)],
      [`huber_loss(${reduction})`, () => x().huberLoss(y(), 1.0, reduction)],
      // The two classification losses. `nllLoss` **averaged as soon as it gathered**, so
      // there was nowhere to build `none` — per-sample values cannot be recovered from a
      // scalar.
      // `ignoreIndex` sits between the target and the reduction now, as it does in
      // torch and in the core. Passing the reduction positionally used to work and
      // would now set an index, which is why `tsc` names every one of these.
      [`cross_entropy(${reduction})`,
        () => ceX().crossEntropy(ceT(), -100, reduction)],
      [`nll_loss(${reduction})`, () => ceLogp().nllLoss(ceT(), -100, reduction)],
      // **Binary cross-entropy over probabilities.** Its logits form has been asked
      // for a long time and this one could not be, because borch.ts had no method
      // taking probabilities. The core's forward was `-(p + 1e-12).log()`, which
      // caps at 27.63 whatever `p` is — `CrossEntropyLoss`'s defect a second time,
      // with the guard and the defect on the same line. torch clamps the log's
      // *output* at −100. These are comfortable probabilities and would not find
      // that; the saturating ones are asked on the Python side.
      [`binary_cross_entropy(${reduction})`,
        () => bceP().bce(bceT(), reduction)],
      [`nn.BCELoss(${reduction})`,
        () => new nn.BCELoss(undefined, null, null, reduction).call(bceP(), bceT())],
      // Six layers were **missing** for a while too. borch.ts's `nn` had the rare ones
      // such as `HuberLoss` and was missing the common ones, and **the binding was filling
      // in by building layers over the tensor methods itself**, so the golden saw none of
      // it — every case goes through the binding. They were names missing only for somebody
      // writing TypeScript directly.
      // **`null, null` is torch's deprecated pair, not padding.** These read
      // `new MSELoss(reduction)` until the two legacy seats went in ahead of it, at
      // which point the string landed on `sizeAverage` — the very shift the seats
      // were added to close. TypeScript refused to compile it, which is the whole
      // difference between this file and the Python side, where the same mistake
      // ran and had to be caught by comparing values against torch.
      [`nn.MSELoss(${reduction})`,
        () => new nn.MSELoss(null, null, reduction).call(x(), y())],
      [`nn.L1Loss(${reduction})`,
        () => new nn.L1Loss(null, null, reduction).call(x(), y())],
      [`nn.SmoothL1Loss(${reduction})`,
        () => new nn.SmoothL1Loss(null, null, reduction, 1.0).call(x(), y())],
      [`nn.CrossEntropyLoss(${reduction})`,
        () => new nn.CrossEntropyLoss(undefined, null, -100, null, reduction)
          .call(ceX(), ceT())],
      [`nn.NLLLoss(${reduction})`,
        () => new nn.NLLLoss(undefined, null, -100, null, reduction).call(ceLogp(), ceT())],
    ];
    for (const [name, fn] of fns) out.set(`loss::reduction::${name}`, fn);
  }
  // **`weight` on the three elementwise losses, which was a refusal nothing asked.**
  // The core carried the seat and stopped the value, on the ground that `mean`
  // divides by the sum of the weights rather than by the count — right, and also the
  // specification. Measured:
  //
  //     none  w · ℓ                      all three
  //     sum   Σ w · ℓ                    all three
  //     mean  Σ w·ℓ / Σ w                `l1Loss` and `mseLoss`
  //     mean  Σ w·ℓ / n                  `huberLoss`
  //
  // Its sum is 12 against 6 elements, so the two divisors are two different numbers
  // and huber can be told from the other two — with all-ones, or with all-twos over
  // twelve elements, the rules agree and nothing shows.
  const lossW = () => Tensor.from([1, 2, 3, 4, 1, 1], [2, 3]);
  for (const reduction of ["none", "sum", "mean"] as const) {
    out.set(`loss::weight::l1_loss(${reduction})`,
      () => x().l1Loss(y(), reduction, lossW()));
    out.set(`loss::weight::mse_loss(${reduction})`,
      () => x().mseLoss(y(), reduction, lossW()));
    out.set(`loss::weight::huber_loss(${reduction})`,
      () => x().huberLoss(y(), 1.0, reduction, lossW()));
  }
  // The backward carries the weight because it is a `mul` in the graph rather than a
  // number applied afterwards — nothing had to be written for it, and nothing would
  // say so if it had been dropped.
  out.set("loss::weight::mse_loss 의 기울기", () => {
    const p = x(true);
    p.mseLoss(y(), "mean", lossW()).backward();
    return gradOf(p, "mse_loss/weight");
  });
  // torch does not broadcast the weight: a `[6]` against a `[2, 3]` input raises.
  out.set("loss::weight::모양이 다르면 거절", () => {
    try {
      x().l1Loss(y(), "mean", Tensor.from([1, 2, 3, 4, 1, 1], [6]));
    } catch (err) {
      return String(err).includes("must have the same size")
        ? "문구대로" : `다른 문구 <${err}>`;
    }
    return "안 던졌다";
  });

  // An unknown value is not swallowed into mean. `batchmean` exists **only** for
  // `klDiv`, so on another loss it is a wrong name.
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

  // **For a loss the gradient is everything.** With the value right and the gradient
  // wrong, training quietly goes somewhere else.
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
  // BCE's gradient has its own formula in the core — torch floors the denominator at
  // 1e-12 rather than differentiating through the −100 clamp. This asks at comfortable
  // probabilities, where the two agree; the saturating end is asked in Python.
  out.set("loss::grad::bce", () => {
    const q = bceP();
    q.requiresGrad = true;
    q.bce(bceT()).backward();
    return gradOf(q, "bce");
  });

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
 * Padding — the four modes and fifteen layers.
 *
 * The input is an `arange`, so **the answer alone says where each value came from.**
 * Whether it mirrored, replicated or wrapped is written straight into the values, so these
 * cases hold the whole convention. The details are written in `tests/cases.py`'s
 * `pad_cases`.
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

  // **`layerNormOver` was here the whole time and the Python side did not use it.**
  // `F.layer_norm` folded the last axis whatever `normalized_shape` said, because
  // that argument was in the signature and the body never read it; the binding
  // folded `layerNorm(-len(shape))`, which is one axis *that far from the end*.
  // Both agree with this at one axis, which is every case that asked.
  const ln = (grad = false): Tensor =>
    Tensor.from(Array.from({ length: 24 }, (_, i) => i), [2, 3, 4],
                grad ? { requiresGrad: true } : {});
  for (const [tag, dims] of [["(4,)", 1], ["(3, 4)", 2], ["(2, 3, 4)", 3]] as
       [string, number][]) {
    out.set(`norm::F.layer_norm${tag}`, () => ln().layerNormOver(dims));
    // The gradient too — a fold over the wrong axes has a plausible forward and a
    // backward that trains the wrong thing.
    out.set(`norm::grad::F.layer_norm${tag}`, () => {
      const x = ln(true);
      seeded(x.layerNormOver(dims)).backward();
      return gradOf(x, `layerNormOver(${dims})`);
    });
  }
  const lnw = (): Tensor =>
    Tensor.from(Array.from({ length: 12 }, (_, i) => i + 1), [3, 4]);
  out.set("norm::F.layer_norm(weight)", () => ln().layerNormOver(2).mul(lnw()));
  out.set("norm::F.layer_norm(weight, bias)",
    () => ln().layerNormOver(2).mul(lnw()).add(Tensor.full([3, 4], 2)));
  // The layer beside the function, on one input — one of them being wrong cannot
  // hide behind the other, which is exactly how this went unseen: the golden asked
  // the layer for the multi-axis case and the function for the single-axis one.
  for (const [tag, shape, dims] of [["(4,)", [4], 1], ["(3, 4)", [3, 4], 2]] as
       [string, number[], number][]) {
    out.set(`norm::nn.LayerNorm${tag} 는 F 와 같다`,
      () => new nn.LayerNorm(shape).call(ln()).sub(ln().layerNormOver(dims)));
  }

  add("F.group_norm(1)", (x) => x.groupNorm(1), "img");
  add("F.group_norm(3)", (x) => x.groupNorm(3), "img");
  const gn = (groups: number): nn.Module => new nn.GroupNorm(groups, 3);
  out.set("norm::nn.GroupNorm(1,3)", () => gn(1).call(inp.get("img")));
  out.set("norm::nn.GroupNorm(3,3)", () => gn(3).call(inp.get("img")));
  out.set("norm::nn.GroupNorm/파라미터 이름",
    () => Object.keys(gn(3).namedParameters()).join(" "));
  // **`affine` and `bias` are two different halves and the row above sees neither.**
  // `affine=false` is no learnable scale or shift; `bias=false` keeps the scale and
  // drops the shift. Neither existed on any of the thirteen normalisation layers
  // here, so a layer built either way was a type error and no case had ever tried.
  //
  // The parameter names are asked as well as the values, because that is the half a
  // value case cannot see: the arithmetic is right either way, and what goes wrong
  // is a `stateDict` carrying a key torch's does not.
  // **`null, null` is torch's `device` and `dtype`, not padding.** `GroupNorm` puts
  // `bias` behind them — keyword-only over there, so no torch call reaches it by
  // position — and these lines used to pass the flag straight into what is now the
  // `device` seat. `tsc` named both the moment the seats went in.
  for (const [flag, affine, bias] of
    [["affine=False", false, true], ["bias=False", true, false]] as const) {
    out.set(`norm::nn.GroupNorm(${flag})`,
      () => new nn.GroupNorm(3, 3, 1e-5, affine, null, null, bias)
        .call(inp.get("img")));
    out.set(`norm::nn.GroupNorm(${flag})/파라미터 이름`,
      () => Object.keys(new nn.GroupNorm(3, 3, 1e-5, affine, null, null, bias)
        .namedParameters()).join(" "));
    out.set(`norm::nn.BatchNorm2d(${flag})/파라미터 이름`,
      () => Object.keys(new nn.BatchNormND(3, 1e-5, 0.1, affine, true, null, null, bias)
        .namedParameters()).join(" "));
  }

  add("F.instance_norm", (x) => x.instanceNorm(), "img");
  // The three names, not the base three times. Each of these cases carried the
  // name of a class that did not exist and built `InstanceNormND` instead — the
  // case table promising a name the same way the docstring did.
  for (const [nd, key] of [["1d", "nd_seq"], ["2d", "img"], ["3d", "nd_vol"]] as const) {
    const Cls = { "1d": nn.InstanceNorm1d, "2d": nn.InstanceNorm2d, "3d": nn.InstanceNorm3d }[nd];
    out.set(`norm::nn.InstanceNorm${nd}`,
      () => new Cls(inp.get(key).shape[1] ?? 1).call(inp.get(key)));
  }

  add("F.rms_norm", (x) => x.rmsNorm(1), "img");
  out.set("norm::nn.RMSNorm", () => new nn.RMSNorm(4).call(inp.get("img")));
  // **`eps` reached the kernel and not the binding's copy of the layer.** This
  // class has carried it from the start — its own note records fixing exactly that
  // — and the Python class beside it repeated the defect. Asked at an `eps` big
  // enough to move the answer, because at the default the two agree.
  out.set("norm::nn.RMSNorm(eps 를 크게)",
    () => new nn.RMSNorm(4, 10.0).call(inp.get("img")));
  out.set("norm::F.rms_norm(eps 를 크게)",
    () => inp.get("img").rmsNorm(1, 10.0));

  // **`normalizedShape` is how many axes are folded.** Asked with a single axis alone the
  // answer equals "fold the last axis" and the rule is invisible — all three were written
  // that way.
  out.set("norm::nn.LayerNorm(축 하나)",
    () => new nn.LayerNorm(4).call(inp.get("img")));
  out.set("norm::nn.LayerNorm(축 둘)",
    () => new nn.LayerNorm([4, 4]).call(inp.get("img")));
  out.set("norm::grad::nn.LayerNorm(축 둘)", () => {
    const x = inp.get("img", true);
    seeded(new nn.LayerNorm([4, 4]).call(x)).backward();
    return gradOf(x, "LayerNorm(축 둘)");
  });
  // What the golden froze is **the exception's kind name.** Folded down to "did it stop",
  // a `TypeError` from a typo passes too.
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

  // **The weight-side gradient is looked at too.** The input side alone misses reversed
  // axes.
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
      // It goes through the dimension-fixed names — calling the `ND` form alone leaves
      // nobody measuring those three.
      const Cls = { 1: nn.ConvTranspose1d, 2: nn.ConvTranspose2d,
        3: nn.ConvTranspose3d }[spatial] ?? nn.ConvTranspose2d;
      const layer = new Cls(w.shape[0] ?? 1, w.shape[1] ?? 1, w.shape[2] ?? 1);
      layer.loadStateDict({ weight: w, bias: inp.get(bk) });
      return layer.call(inp.get(key));
    });
  }
}

/**
 * Five optimisers and eight schedulers. **Asked several steps in.**
 *
 * An optimiser accumulates state, so at the first step they all behave alike. Measured at
 * one step, implementing all five as SGD passes.
 *
 * What a scheduler produces is a **sequence** of learning rates, so the whole sequence is
 * frozen. Looking at the final value alone passes even when the route differs, and
 * `LinearLR` and `ConstantLR` really do meet at the end.
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
    // **Adafactor's point only runs on a 2-D weight** — `0.weight` here is (8, 6).
    ["Adafactor", (ps) => new optim.Adafactor(ps, 0.05)],
    // **`AdamW` diverges from `Adam` only once the decay bites.** One thing separates
    // them: where the decay lands — on the gradient before the moments see it (`Adam`) or
    // on the weight after the update (`AdamW`). At `weightDecay=0` both branches vanish and
    // they are the same optimiser, so a case built with the defaults passes under either
    // implementation.
    ["Adam(weight_decay)",
      (ps) => new optim.Adam(ps, 0.05, [0.9, 0.999], 1e-8, 0.1)],
    ["AdamW", (ps) => new optim.AdamW(ps, 0.05, [0.9, 0.999], 1e-8, 0.1)],
    // **Every optimiser taking `weightDecay` is asked with a non-zero value.**
    //
    // Until today there was no such case at all, and that absence hid seven defects — the
    // binding accepted this argument and discarded it in seven places, and one of them,
    // `NAdam`, **matched even on argument count**, so an arity check could not see it
    // either.
    //
    // Mind the position: for `Adagrad` it comes **before** `eps` and for `NAdam` **before**
    // `momentumDecay`. Appended at the end it quietly becomes a different argument.
    ["Adagrad(weight_decay)",
      (ps) => new optim.Adagrad(ps, 0.1, 0, 0.1)],
    ["Adadelta(weight_decay)",
      (ps) => new optim.Adadelta(ps, 0.5, 0.9, 1e-6, 0.1)],
    ["Adamax(weight_decay)",
      (ps) => new optim.Adamax(ps, 0.05, [0.9, 0.999], 1e-8, 0.1)],
    ["NAdam(weight_decay)",
      (ps) => new optim.NAdam(ps, 0.05, [0.9, 0.999], 1e-8, 0.1)],
    ["RAdam(weight_decay)",
      (ps) => new optim.RAdam(ps, 0.05, [0.9, 0.999], 1e-8, 0.1)],
    ["RMSprop(weight_decay)",
      (ps) => new optim.RMSprop(ps, 0.01, 0.99, 1e-8, 0.1)],
    ["ASGD(weight_decay)",
      (ps) => new optim.ASGD(ps, 0.05, 1e-4, 0.75, 1e6, 0.1)],
    ["SGD(weight_decay)", (ps) => new optim.SGD(ps, 0.05, 0, 0, 0.1)],
    // The three that were not parameters until the core took torch's order.
    ["SGD(dampening)", (ps) => new optim.SGD(ps, 0.1, 0.9, 0.5)],
    ["SGD(nesterov)", (ps) => new optim.SGD(ps, 0.1, 0.9, 0, 0, true)],
    ["SGD(maximize)", (ps) => new optim.SGD(ps, 0.01, 0, 0, 0, false,
      { maximize: true })],
    ["Adagrad(initial_accumulator_value)",
      (ps) => new optim.Adagrad(ps, 0.1, 0, 0, 0.5)],
    // **The six that borch.ts could not be asked about until it had them.** Each
    // changes the values rather than the speed, so each waited on real work over
    // here — `maximize` on the base, and one buffer apiece for the other four.
    //
    // The arguments are the Python table's, value for value. Picking different
    // ones would ask a question torch never froze an answer to, and the row would
    // pass by comparing borch.ts against itself.
    ["Adam(maximize)", (ps) => new optim.Adam(ps, 0.05, [0.9, 0.999], 1e-8, 0,
      false, { maximize: true })],
    ["RMSprop(maximize)", (ps) => new optim.RMSprop(ps, 0.05, 0.99, 1e-8, 0, 0,
      false, { maximize: true })],
    ["Adam(amsgrad)",
      (ps) => new optim.Adam(ps, 0.05, [0.9, 0.999], 1e-8, 0.1, true)],
    ["RMSprop(centered)",
      (ps) => new optim.RMSprop(ps, 0.05, 0.99, 1e-8, 0, 0, true)],
    ["RMSprop(momentum)",
      (ps) => new optim.RMSprop(ps, 0.05, 0.99, 1e-8, 0, 0.9)],
    ["NAdam(decoupled_weight_decay)",
      (ps) => new optim.NAdam(ps, 0.05, [0.9, 0.999], 1e-8, 0.1, 4e-3, true)],
    // **The same flag on the other two torch gives it to.** Absent here the word was
    // taken and discarded, and the coupled answer came back under the decoupled name.
    ["Adam(decoupled_weight_decay)",
      (ps) => new optim.Adam(ps, 0.05, [0.9, 0.999], 1e-8, 0.1, false,
        { decoupledWeightDecay: true })],
    ["RAdam(decoupled_weight_decay)",
      (ps) => new optim.RAdam(ps, 0.05, [0.9, 0.999], 1e-8, 0.1, true)],
    // **A default is the one value a case cannot check by using it**, because every
    // row above names its own rate. So this one names none: `new SGD(params)` is the
    // line a first tutorial writes, and until `lr` had torch's default it stopped
    // here on an argument count.
    ["SGD(the default rate)", (ps) => new optim.SGD(ps)],
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

  // ── The optimiser's `state_dict` — resuming a training run hangs on it ──
  //
  // Saving the model weights and not the optimiser restarts a resumed run **without its
  // momentum and second moments.** Nothing is raised and the loss curve simply jumps once.
  //
  // **It asks about the values rather than the key names.** torch's is
  // `{state, param_groups}` and this side is a bank structure, so the shapes differ — asking
  // about the shape diverges forever, while the thing that matters, "does stopping and
  // resuming equal not stopping", can be asked regardless of shape.
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
    // **It is loaded into a new optimiser.** Loading back into the same one asks
    // nothing.
    const fresh = make(m.parameters());
    if (carry) fresh.loadStateDict(saved);
    for (let i = 0; i < 2; i++) stepOnce(m, fresh);
    const w = m.namedParameters()["0.weight"];
    if (!w) throw new Error("0.weight 가 없다");
    return w;
  };
  // `SGD(momentum=0)` has no state and passes even with the save omitted entirely — it
  // has to be asked with something that carries state.
  const resumable: [string, (ps: Tensor[]) => optim.Optimizer][] = [
    ["SGD", (ps) => new optim.SGD(ps, 0.1, 0.9)],
    ["Adam", (ps) => new optim.Adam(ps, 0.05)],
    ["RMSprop", (ps) => new optim.RMSprop(ps, 0.05)],
  ];
  for (const [name, make] of resumable) {
    out.set(`opt::${name}/이어서 학습하기`, () => resume(make));
  }
  // **Did the three above actually move anything.** The three can pass even with
  // `loadStateDict` doing nothing — so **the difference between the two** is frozen as the
  // answer. Under an implementation where loading does nothing this comes out 0, and at
  // that moment the three above are green and measuring nothing.
  out.set("opt::상태를 안 옮기면 갈린다", () => {
    const make = (ps: Tensor[]): optim.Optimizer => new optim.Adam(ps, 0.05);
    return resume(make, true).sub(resume(make, false));
  });

  // ── `zeroGrad(setToNone)` — the seat did not exist here at all ─────────────
  //
  // On both Python sides it existed and was never read, so the ordinary call
  // agreed everywhere and `false` quietly did the other thing. It is invisible
  // until something reads `grad` between two calls: `null` has no shape and
  // nothing to add into, which is what the argument is for.
  // The name carries Python's `True`/`False` — the golden is keyed by the string,
  // and JavaScript prints a boolean in lower case.
  for (const setToNone of [true, false]) {
    out.set(`opt::zero_grad(opt, set_to_none=${setToNone ? "True" : "False"})`, async () => {
      const m = model();
      const opt = new optim.SGD(m.parameters(), 0.1);
      new nn.CrossEntropyLoss()
        .forward(m.forward(inp.get("train_x")), inp.get("train_y")).backward();
      opt.zeroGrad(setToNone);
      const w = m.namedParameters()["0.weight"];
      const got = w?.grad ?? null;
      // The **kind** of what is left, not a value — `null` and a zero tensor are
      // what the flag chooses between, and only one of them has values at all.
      // Reading a number back is asynchronous here, so the case awaits it; the
      // Python side writes `True`/`False` and this must spell them the same way.
      if (got === null) return "None";
      const total = await got.abs().sum().item();
      return `0 텐서=${total === 0 ? "True" : "False"}`;
    });
  }

  // **The scheduler has to resume too.** Restoring the optimiser alone and standing a new
  // scheduler up sends the learning rate **back to its first value** — a half-cooled run
  // heats up again.
  //
  // **It comes back as characters.** A learning-rate trace is a sequence, so a divergence
  // has to show which position diverged — handed over as a tensor, all that is left is
  // "max diff 1.5e-01".
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
  // With loading doing nothing the two traces become equal, and then the case above is
  // measuring nothing.
  out.set("opt::스케줄러 상태를 안 옮기면 갈린다",
    () => `${schedResume(true)} | ${schedResume(false)}`);

  // **`ReduceLROnPlateau` had no `state_dict` on the Python side at all** — seven
  // mutable attributes and nowhere to put one of them, so a run that saved everything
  // it could still came back with the patience, the best value and the cooldown at
  // their starting points. Asked **with a cooldown**, because that is the attribute a
  // resume loses most visibly: without one the patience is re-earned in a step or two
  // and the two traces rejoin.
  const plateauResume = (carry: boolean): string => {
    const p = Tensor.from([1.0], [1], { requiresGrad: true });
    const opt = new optim.SGD([p], 1.0);
    const build = (): optim.ReduceLROnPlateau =>
      new optim.ReduceLROnPlateau(opt, "min", 0.5, 1, 1e-4, "rel", 2);
    const sch = build();
    const seen: string[] = [];
    for (const metric of [1, 1, 0.5, 0.5, 0.5]) {
      sch.step(metric);
      seen.push(`${opt.paramGroups[0]?.lr ?? 0}`);
    }
    const saved = sch.stateDict();
    const fresh = build();
    if (carry) fresh.loadStateDict(saved);
    for (const metric of [0.5, 0.5, 0.5, 0.5, 0.5]) {
      fresh.step(metric);
      seen.push(`${opt.paramGroups[0]?.lr ?? 0}`);
    }
    return seen.join(" ");
  };
  out.set("opt::ReduceLROnPlateau/이어서 학습하기", () => plateauResume(true));
  out.set("opt::ReduceLROnPlateau 상태를 안 옮기면 갈린다",
    () => `${plateauResume(true)} | ${plateauResume(false)}`);

  // ── `save`/`load` — for the above to be of use it has to become a file ──
  //
  // With all three `stateDict()`s matched but **no way to use them**, resuming a training
  // run is a within-one-session story. Refresh the tab and it is gone.
  //
  // The formats differ (torch is pickle and this side is safetensors), so **it asks about
  // the round trip rather than the bytes.** The Python side goes through a temporary file
  // and this one stays in bytes, and the question is the same — does reading back what was
  // written give the same thing.
  /** Takes the tensor table out of what was read back. A tree is `Savable`, so it has to be
   *  narrowed before use. */
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
  // Is what was read back **actually usable.** Matching keys and values are no use if it
  // cannot be loaded as it stands.
  out.set("opt::되읽은 것을 그대로 얹을 수 있다", async () => {
    const bytes = await save(model().stateDict());
    const dst = new nn.Sequential(
      new nn.Linear(6, 8), new nn.ReLU(), new nn.Linear(8, 3));
    dst.loadStateDict(asTensors(load(bytes)));
    return dst.forward(inp.get("train_x"));
  });

  // **The textbook idiom is nested.** It saves `{model: …, opt: …, epoch: 3}` whole. With
  // a flat tensor table alone that code does not run — and for a long time it did not.
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

  // **`stateDict`'s keys already contain dots** (`0.weight`). Splitting the flattened
  // name on dots again to restore gives `{model: {0: {weight: …}}}` — every value is there
  // and the structure differs, so it blows up when what was read back is passed to
  // `loadStateDict`.
  out.set("opt::중첩 안의 점 찍힌 열쇠가 안 쪼개진다", async () => {
    const got = load(await save({ model: model().stateDict() }));
    if (got === null || typeof got !== "object" || Array.isArray(got)
        || got instanceof Tensor) {
      throw new Error("되읽은 것이 사전이 아니다");
    }
    return Object.keys(asTensors(got.model as Savable)).sort().join(" ");
  });

  /** The learning rate's trace. **It really steps the optimiser** — the order decides the
   *  values. */
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

  // ── the two chaining classes checked nothing about their schedulers ────────
  //
  // `SequentialLR(a, [ConstantLR(a), ConstantLR(b)])` was built without complaint
  // and then, at the milestone, stepped `b`'s rate while `getLastLr` read `a`'s.
  // The rate that trains and the rate that is printed part company, and nothing
  // raises. torch checks it in both classes and neither of these did.
  const schedulerRefuses = (fragment: string, body: () => void): string => {
    try {
      body();
    } catch (err) {
      const said = String(err);
      return said.includes(fragment) ? "문구대로" : `다른 문구 <${said.slice(0, 50)}>`;
    }
    return "안 던졌다";
  };
  out.set("opt::SequentialLR(다른 optimizer)=거절",
    () => schedulerRefuses("belong to the same optimizer", () => {
      const a = new optim.SGD(model().parameters(), 0.2);
      const b = new optim.SGD(model().parameters(), 0.2);
      new optim.SequentialLR(a, [new optim.ConstantLR(a, 0.5, 2).start(),
                                 new optim.ConstantLR(b, 0.1, 2).start()], [2]);
    }));
  out.set("opt::ChainedScheduler(다른 optimizer)=거절",
    () => schedulerRefuses("belong to the same optimizer", () => {
      const a = new optim.SGD(model().parameters(), 0.2);
      const b = new optim.SGD(model().parameters(), 0.2);
      new optim.ChainedScheduler([new optim.ConstantLR(a).start(),
                                  new optim.ConstantLR(b).start()]);
    }));
  // One scheduler per interval and one interval more than there are milestones.
  // Given two of each the last is never reached and `step` walks the wrong one.
  out.set("opt::SequentialLR(milestone 개수가 안 맞으면)=거절",
    () => schedulerRefuses("one more than the number of milestone", () => {
      const o = new optim.SGD(model().parameters(), 0.2);
      new optim.SequentialLR(o, [new optim.ConstantLR(o).start(),
                                 new optim.ConstantLR(o).start()], [1, 2]);
    }));
  // **`last_epoch` was a seat the binding never read** — borch.ts's constructor has
  // taken it all along and the word stopped one call short, so resuming put the
  // chain back at its first interval however far it had got.
  for (const lastEpoch of [-1, 0, 2]) {
    out.set(`opt::SequentialLR(last_epoch=${lastEpoch})/자취`, () => {
      const opt = new optim.SGD(model().parameters(), 0.2);
      const sch = new optim.SequentialLR(
        opt, [new optim.ConstantLR(opt, 0.5, 2).start(),
              new optim.ExponentialLR(opt, 0.5).start()], [2], lastEpoch);
      const seen: number[] = [];
      for (let i = 0; i < 5; i++) {
        seen.push(opt.paramGroups[0]?.lr ?? 0);
        sch.step();
      }
      return Tensor.from(seen, [5]);
    });
  }

  // ── Where the branches are asked about narrowly ─────────────────────────
  //
  // The model training above passes as long as the optimiser is **roughly right.** The
  // arguments that decide a branch only show when a gradient is fed to a single parameter
  // by hand.
  const start = () => Tensor.from([1, -2, 0.5], [3], { requiresGrad: true });
  const ramp = (i: number) => Tensor.from(
    [0.1 * (i + 1), -0.3 * (i + 1), 0.2 * (i + 1)], [3]);
  // **A gradient whose sign flips.** Rprop's `etas` and the "a flipped cell does not
  // move" rule are visible only here.
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
      // **The value has to be copied here.** `reshape` passes the buffer straight on, so
      // stored directly all four rows point at the same place, and when `cat` reads later
      // the last value comes out four times — a point rather than a trace. Adding 0 moves
      // it to a new buffer.
      seen.push(p.reshape([1, 3]).detach().add(Tensor.full([], 0)));
    }
    return Tensor.cat(seen, 0);
  };
  const flip = (i: number) => Tensor.from(flipGrads[i] ?? [0, 0, 0], [3]);

  // ── the four words every `torch.optim.*` carries and none of the three of us compute ──
  //
  // **torch draws the line, not us.** `foreach` and `fused` pick a kernel: measured on both
  // classes, torch returns the very same numbers with them on, so ours take the word and
  // ignore it. `capturable` and `differentiable` change what a step means, and torch itself
  // stops on both without CUDA, so ours stop too. The verdict is the category and not the
  // wording — torch's own two classes refuse `capturable` with different exception types,
  // and asking for a phrase would freeze which torch build froze the file.
  const carries = (
    make: (ps: Tensor[], o: optim.OptimizerOptions) => optim.Optimizer,
    word: keyof optim.OptimizerOptions,
  ) => async () => {
    const stepOnce = (o: optim.OptimizerOptions) => {
      const p = start();
      const opt = make([p], o);
      opt.zeroGrad();
      p.grad = Tensor.from([0.3, 0.2, -0.4], [3]);
      opt.step();
      return p.detach();
    };
    const plain = stepOnce({});
    let got: Tensor;
    try {
      got = stepOnce({ [word]: true });
    } catch {
      return "거절";
    }
    return await got.sub(plain).abs().amax().item() === 0
      ? "받고 값이 같다" : "받는데 값이 다르다";
  };

  for (const word of ["foreach", "fused", "capturable", "differentiable"] as const) {
    out.set(`opt::낱말::SGD(${word})`,
      carries((ps, o) => new optim.SGD(ps, 0.1, 0, 0, 0, false, o), word));
    out.set(`opt::낱말::Adam(${word})`,
      carries(
        (ps, o) => new optim.Adam(ps, 0.1, [0.9, 0.999], 1e-8, 0, false, o),
        word));
  }

  out.set("opt::ASGD/기본값", walk((ps) => new optim.ASGD(ps), ramp));
  out.set("opt::ASGD/lambd",
    walk((ps) => new optim.ASGD(ps, 0.1, 0.01), ramp));
  out.set("opt::ASGD/alpha",
    walk((ps) => new optim.ASGD(ps, 0.1, 1e-4, 0.5), ramp));
  // **The averaging only runs with `t0` lowered** — at the default of a million, `mu` is
  // always 1.
  out.set("opt::ASGD/t0(평균이 도는 자리)",
    walk((ps) => new optim.ASGD(ps, 0.1, 1e-4, 0.75, 2), ramp));
  out.set("opt::ASGD/weight_decay",
    walk((ps) => new optim.ASGD(ps, 0.1, 1e-4, 0.75, 1e6, 0.1), ramp));

  out.set("opt::Rprop/기본값", walk((ps) => new optim.Rprop(ps), ramp));
  out.set("opt::Rprop/부호 바뀜",
    walk((ps) => new optim.Rprop(ps, 0.1), flip));
  out.set("opt::Rprop/etas",
    walk((ps) => new optim.Rprop(ps, 0.1, [0.4, 1.5]), flip));
  out.set("opt::Rprop/step_sizes 상한",
    walk((ps) => new optim.Rprop(ps, 0.1, [0.5, 1.2], [1e-6, 0.11]), ramp));

  out.set("opt::Adafactor/기본값", walk((ps) => new optim.Adafactor(ps), ramp));
  out.set("opt::Adafactor/weight_decay",
    walk((ps) => new optim.Adafactor(ps, 0.1, -0.8, [null, 1e-3], 1.0, 0.1), ramp));
  out.set("opt::Adafactor/d",
    walk((ps) => new optim.Adafactor(ps, 0.1, -0.8, [null, 1e-3], 2.0), ramp));

  // **From 2-D it factors into rows and columns** — asked with a vector alone that path
  // never runs.
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

  // **A single-element tensor is cached by value** — which is why `Tensor.owned` exists.
  // Optimiser state writes into that buffer in place, so where the two meet on a size-1
  // parameter, a constant the whole program uses is quietly overwritten. So it multiplies
  // by that same constant **after training.** The answer is obvious, and that obvious answer
  // catches this defect.
  //
  // For the same reasons as its Python partner it **cannot be caught in one step** (Rprop's
  // first step does not change the step size), the constants looked at include 0 and 1 (the
  // state bank starts at 0), and `SGD`, `Adam` and `RMSprop` are always among them — those
  // three sit outside the shared base.
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

  // **These six alone have asynchronous case bodies.** `LBFGS.step` reads a scalar and
  // branches on it within one step (see the explanation in `optim.ts`). The golden runner
  // awaits the body, so using `async` here alone leaves the rest unchanged.
  const lbfgs = (steps: number, make: (ps: Tensor[]) => optim.LBFGS) => async () => {
    const p = start();
    const opt = make([p]);
    const seen: Tensor[] = [];
    for (let i = 0; i < steps; i++) {
      // **The closure plants the gradient** rather than differentiating. Then `flat` is
      // unchanged from iteration to iteration, no history accumulates, and what is left is
      // the first iteration's gradient descent and the step-size rule. It is exactly the
      // half the Python side's comment names, and the same half is measured here.
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

  // A quadratic with a different curvature per coordinate — the shape a quasi-Newton
  // method wins on, and the history actually fills.
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

  // ── the strong-Wolfe line search, refused here until today ────────────────
  //
  // Carried verbatim from `tests/cases.py`, whose comment records what these rows do
  // and do not defend: the **structure** (skipping the zoom, or keeping Armijo
  // without the curvature test, moves them by units) and not the float32 width, whose
  // effect measures 1.65e-04 — under the threshold.
  out.set("opt::LBFGS(strong_wolfe)/진짜 기울기",
    lbfgsReal(3, (ps) => new optim.LBFGS(ps, 0.8, 20, null, 1e-7, 1e-9, 100,
                                         "strong_wolfe")));
  out.set("opt::LBFGS(strong_wolfe)/이력이 밀려난다",
    lbfgsReal(2, (ps) => new optim.LBFGS(ps, 0.5, 8, null, 1e-7, 1e-9, 2,
                                         "strong_wolfe")));
  // The budget the search gets is what is left of `maxEval`.
  out.set("opt::LBFGS(strong_wolfe)/평가 예산이 짧다",
    lbfgsReal(3, (ps) => new optim.LBFGS(ps, 0.8, 5, 6, 1e-7, 1e-9, 100,
                                         "strong_wolfe")));
  // A name torch does not have, refused where torch refuses it — inside the loop,
  // so a gradient already inside the tolerance never reaches the check.
  out.set("opt::LBFGS(없는 line_search_fn)=문구", async () => {
    const p = start();
    const w = curve();
    const opt = new optim.LBFGS([p], 1, 20, null, 1e-7, 1e-9, 100, "backtracking");
    try {
      await opt.step(() => {
        p.grad = null;
        const out = p.mul(p).mul(w).sum();
        out.backward();
        return out;
      });
    } catch (err) {
      return err instanceof Error ? err.message : String(err);
    }
    return "(거절 없음)";
  });

  // **Coupled quadratics — the off-diagonal term is the point.** `sum(w·p²)` above
  // is separable, so its search direction is always an axis and the zoom phase never
  // runs; at `lr=2` on these the first probe overshoots and the zoom is what produces
  // the step, and at `lr=0.1` it undershoots and the bracketing walks outwards.
  const lbA2 = () => Tensor.from([3, 0.5, 0.5, 2], [2, 2]);
  const lbB2 = () => Tensor.from([1, -2], [2]);
  const lbA3 = () => Tensor.from([4, 1, 0.5, 1, 3, -1, 0.5, -1, 2], [3, 3]);
  const lbB3 = () => Tensor.from([1, -2, 0.5], [3]);
  const lbX2 = () => Tensor.from([2.5, -1.5], [2], { requiresGrad: true });
  const lbX3 = () => Tensor.from([2.5, -1.5, 0.75], [3], { requiresGrad: true });
  const lbfgsShape = (
    x0: () => Tensor, mat: () => Tensor, vec: () => Tensor,
    make: (ps: Tensor[]) => optim.LBFGS, steps = 3,
  ) => async () => {
    const p = x0();
    const opt = make([p]);
    const seen: Tensor[] = [];
    for (let i = 0; i < steps; i++) {
      await opt.step(() => {
        p.grad = null;
        const out = p.mul(mat().matmul(p)).sum().mul(Tensor.full([], 0.5))
          .add(vec().mul(p).sum());
        out.backward();
        return out;
      });
      seen.push(p.detach().clone());
    }
    return Tensor.stack(seen);
  };
  out.set("opt::LBFGS(strong_wolfe)/얽힌 이차형식",
    lbfgsShape(lbX2, lbA2, lbB2,
      (ps) => new optim.LBFGS(ps, 2, 20, null, 1e-7, 1e-9, 100, "strong_wolfe")));
  out.set("opt::LBFGS(strong_wolfe)/얽힌 이차형식(3변수)",
    lbfgsShape(lbX3, lbA3, lbB3,
      (ps) => new optim.LBFGS(ps, 2, 20, null, 1e-7, 1e-9, 100, "strong_wolfe")));
  out.set("opt::LBFGS(strong_wolfe)/처음이 모자란다",
    lbfgsShape(lbX2, lbA2, lbB2,
      (ps) => new optim.LBFGS(ps, 0.1, 2, null, 1e-7, 1e-9, 100, "strong_wolfe")));
  // The same shape without the search, at a rate a fixed step survives — `lr=2` with
  // no search overshoots and the iteration goes chaotic.
  out.set("opt::LBFGS/얽힌 이차형식",
    lbfgsShape(lbX2, lbA2, lbB2, (ps) => new optim.LBFGS(ps, 0.8)));

  // **Different rise and fall, several cycles.** With equal widths and one cycle, neither
  // `stepSizeDown` nor `triangular2` is visible at all.
  out.set("opt::CyclicLR/자취",
    () => trace((o) => new optim.CyclicLR(o, 0.01, 0.1, 3).start(), 14));
  out.set("opt::CyclicLR(위아래 다름)/자취",
    () => trace((o) => new optim.CyclicLR(o, 0.01, 0.1, 2, 4).start(), 14));
  out.set("opt::CyclicLR(triangular2)/자취",
    () => trace((o) => new optim.CyclicLR(
      o, 0.01, 0.1, 3, null, "triangular2").start(), 14));
  // **`exp_range` measures against the step rather than the cycle.**
  out.set("opt::CyclicLR(exp_range)/자취",
    () => trace((o) => new optim.CyclicLR(
      o, 0.01, 0.1, 3, null, "exp_range", 0.9).start(), 14));

  // ── the six seats `CyclicLR` was short of ──────────────────────────────
  //
  // **`scaleMode` alone does nothing, and that is torch's rule** — the branch that
  // derives a curve from `mode` sets both, so a `scaleMode` beside a bare `mode` is
  // overwritten. This one let it win and `triangular2` halved by step: 0.04 where
  // torch says 0.07 at the third entry, caught the first time the case ran.
  out.set("opt::CyclicLR(scale_mode 만으로는 안 바뀐다)/자취",
    () => trace((o) => new optim.CyclicLR(
      o, 0.01, 0.1, 3, null, "triangular2", 1.0, null, "iterations").start(), 14));
  // `scaleFn` **overrides** the mode's own curve — a version multiplying the two
  // together passes every case above and fails this one.
  out.set("opt::CyclicLR(scale_fn)/자취",
    () => trace((o) => new optim.CyclicLR(
      o, 0.01, 0.1, 3, null, "triangular2", 1.0, () => 0.5).start(), 14));
  // **Together they act**, which is what makes the pair worth a seat.
  out.set("opt::CyclicLR(scale_fn, iterations)/자취",
    () => trace((o) => new optim.CyclicLR(
      o, 0.01, 0.1, 3, null, "triangular", 1.0,
      (i) => 1 / (1 + i), "iterations").start(), 14));
  out.set("opt::CyclicLR(scale_fn, cycle)/자취",
    () => trace((o) => new optim.CyclicLR(
      o, 0.01, 0.1, 3, null, "triangular", 1.0,
      (c) => 1 / (1 + c), "cycle").start(), 14));

  // **`cycleMomentum` is the one of the six that changes trained values**, and no
  // `자취` case can see it: those compare the printed learning rate and this moves
  // the momentum, against the rate. torch's default is on.
  const cyclicMomentum = (
    base = 0.8, max = 0.9, on = true,
  ): { param: Tensor; trail: Tensor } => {
    const p = Tensor.from([1, -2, 0.5], [3], { requiresGrad: true });
    const opt = new optim.SGD([p], 0.1, 0.9);
    const sch = new optim.CyclicLR(
      opt, 0.01, 0.1, 3, null, "triangular", 1.0, null, null, on, base, max,
    ).start();
    const seen: number[] = [];
    for (let i = 0; i < 8; i++) {
      opt.zeroGrad();
      p.grad = Tensor.from([0.3, 0.2, -0.4], [3]);
      opt.step();
      sch.step();
      seen.push(opt.paramGroups[0]?.momentum ?? 0);
    }
    return { param: p.detach(), trail: Tensor.from(seen, [8]) };
  };
  out.set("opt::CyclicLR/momentum 자취", () => cyclicMomentum().trail);
  out.set("opt::CyclicLR/momentum 이 값을 바꾼다", () => cyclicMomentum().param);
  out.set("opt::CyclicLR(cycle_momentum=False)/자취",
    () => cyclicMomentum(0.8, 0.9, false).trail);
  out.set("opt::CyclicLR(base/max_momentum)",
    () => cyclicMomentum(0.7, 0.95).trail);

  // `lastEpoch` resumes mid-schedule. **The Python case sets `initial_lr` equal to
  // `base_lr` on purpose** — torch assigns its own `base_lrs` after the base class
  // has already taken one step, so the first value after a resume is measured from
  // `initial_lr`; equal, that ordering cannot show. Nothing here carries an
  // `initialLr` before the scheduler stamps one, so this side has no such seam.
  out.set("opt::CyclicLR(last_epoch)/자취", () => {
    const p = Tensor.from([0, 0, 0], [3], { requiresGrad: true });
    const opt = new optim.SGD([p], 0.1);
    const sch = new optim.CyclicLR(
      opt, 0.01, 0.1, 3, null, "triangular", 1.0, null, null, false, 0.8, 0.9, 4,
    ).start();
    const seen: number[] = [];
    for (let i = 0; i < 6; i++) {
      seen.push(opt.paramGroups[0]?.lr ?? 0);
      opt.step();
      sch.step();
    }
    return Tensor.from(seen, [6]);
  });
}

/**
 * Dropout. **Asked by property rather than by value.**
 *
 * The answer depends on the generator, and there is no reason ours matches torch's. Not
 * asking would leave a whole layer outside every check, so it asks **only what both sides
 * can answer identically** — is evaluation mode the identity, are the survivors scaled by
 * `1/(1-p)`, does it drop roughly `p`, does the gradient flow only to the survivors, and
 * do two calls differ.
 */
function addDropout(out: Map<string, Case>, inp: Inputs): void {
  const big = (grad = false): Tensor => {
    // The same input as the golden's — `train_x` stacked forty times. Measuring a
    // proportion needs many samples, and with few, the draw's own wobble moves the
    // answer.
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
 * `scaledDotProductAttention`. **The name modern transformer code calls directly.**
 *
 * The commonest misunderstanding is that the mask multiplies rather than adds — it adds a
 * large negative so that softmax produces 0, rather than multiplying by 0.
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
    // **`isCausal` moved from the fifth seat to the sixth** when `dropoutP` took
    // torch's place, and this line is what caught it — `tsc` refused a boolean in a
    // number slot. The *lucky* direction: had the two been the same type it would
    // have compiled and the case would have asked about a dropout of 1.
    ["인과", (x) => nn.scaledDotProductAttention(x, x, x, null, 0, true)],
    // **`scale` replaces `1/√dim`**, so a caller who sets it and is ignored gets a
    // model whose attention is weighted wrong and trains to somewhere plausible.
    // Asked at a value the default cannot produce.
    ["배율", (x) => nn.scaledDotProductAttention(x, x, x, null, 0, false, 0.25)],
  ];
  for (const [name, fn] of shapes) {
    out.set(`sdpa::${name}`, () => fn(inp.get("attn_x")));
    out.set(`sdpa::grad::${name}`, () => {
      const q = inp.get("attn_x", true);
      seeded(fn(q)).backward();
      return gradOf(q, name);
    });
  }

  // **Given the same thing for all three, swapped arguments give the same value and go
  // uncaught.**
  out.set("sdpa::q·k·v 가 다를 때", () => {
    const q = inp.get("attn_x");
    const k = q.mul(Tensor.full([], 0.5)).add(Tensor.full([], 0.1));
    const v = q.flip(0);
    return nn.scaledDotProductAttention(q, k, v);
  });
}

/**
 * The form the Python side calls **as a module function**, as in `torch.sum(x)`.
 *
 * TypeScript has no such second name — the method is the only way to call it here, and a
 * free function alongside would only widen the surface. So these cases **produce the same
 * answer through the method.** What the golden asks about is the value, and the calling
 * syntax may differ per language.
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
    // `flat` is not exposed — `reshape` does the same thing.
    ["argmax", () => m().reshape([m().size]).argmax(0)],
    ["argmin(dim)", () => m().argmin(1)],
    ["clone", () => m().clone()],
    ["detach", () => m().detach()],
    ["flatten", () => m().reshape([m().size])],
    ["permute", () => m().permute([1, 0])],
    ["transpose", () => m().transpose()],
    ["squeeze", () => inp.get("x1").reshape([1, 6, 1]).squeeze(2).squeeze(0)],
    // Given an axis, `max` produces **a pair.** The value side is taken.
    ["max", () => m().reshape([m().size]).max(0).values],
    ["max(dim)/값", () => m().max(1).values],
    ["min(dim)/번호", () => m().min(1).indices],
  ];
  for (const [name, fn] of table) out.set(`modfn::${name}`, fn);

  // **Two of torch's rules that were absent on both sides.** Several axes at once,
  // and an axis whose length is not 1 left alone rather than refused — refusing looks
  // like the safer answer and is not, because it is torch's own answer being refused.
  const boxed = (): Tensor => inp.get("x1").reshape([1, 6, 1]);
  for (const [name, call] of [
    ["squeeze(0, 2)", (t: Tensor) => t.squeeze(0, 2)],
    // borch.ts has no tuple spelling — JavaScript spreads instead, and the golden
    // answer is the shape, which is the same either way.
    ["squeeze((0, 2))", (t: Tensor) => t.squeeze(0, 2)],
    ["squeeze(0, 1)", (t: Tensor) => t.squeeze(0, 1)],
    ["squeeze(길이가 1 이 아닌 축)", (t: Tensor) => t.squeeze(1)],
  ] as [string, (t: Tensor) => Tensor][]) {
    out.set(`modfn::모양::${name}`, () => `(${call(boxed()).shape.join(", ")}${
      call(boxed()).shape.length === 1 ? "," : ""})`);
  }

  out.set("modfn::relu_(원본이 바뀐다)", () => {
    const t = inp.get("x1").clone();
    t.inplaceUnary("relu");
    return t;
  });

  // ── **The second names** the Python side calls, as in `torch.add(a, b)`. ──
  //
  // TypeScript has no such second name — the method is the only way to call it, and a free
  // function alongside would only widen the surface. Here they **produce the same answer
  // through the method.**
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
    // **`remainder` and `fmod` diverge on negatives** — the sign follows the other
    // side.
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
    // **`dstack` appends an axis** — it does not prepend one.
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

  // The blocks are laid along the diagonal and the rest is 0.
  out.set("modfn::block_diag", () => {
    const a = a2();
    const b = b2().narrow(0, 0, 1);
    const top = Tensor.cat([a, Tensor.zeros([3, 4])], 1);
    const bottom = Tensor.cat([Tensor.zeros([1, 4]), b], 1);
    return Tensor.cat([top, bottom], 0);
  });
}

/**
 * The remaining pooling dimensions and kinds.
 *
 * **The adaptive form's point is when it does not divide evenly.** A rule that sizes the
 * window differently per position, diverging from torch's, gives quietly different values,
 * and asked only where it divides evenly, that rule is never seen.
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
  // **A case named `nn.` has to go through the layer.** The `F.` partner just above
  // already measures the tensor method, so calling the same method here has two cases asking
  // one thing twice while nobody measures the layer name — which is what happened.
  out.set("pool::nn.AvgPool1d",
    () => new nn.AvgPool1d(2, 2).call(inp.get("nd_seq")));
  out.set("pool::nn.AvgPool3d",
    () => new nn.AvgPool3d(2, 2).call(inp.get("nd_vol")));

  // **Through the `F.` names, as the Python side does.** The computation is
  // `adaptivePool` either way; what these ask about is the wrapper — its name and
  // its rank check, which is the only part of it that is not one line.
  add("F.adaptive_avg_pool1d(4)", (x) => F.adaptiveAvgPool1d(x, 4), "nd_seq");
  add("F.adaptive_avg_pool1d(3)", (x) => F.adaptiveAvgPool1d(x, 3), "nd_seq");
  add("F.adaptive_avg_pool3d", (x) => F.adaptiveAvgPool3d(x, 2), "nd_vol");
  out.set("pool::nn.AdaptiveAvgPool1d",
    () => new nn.AdaptiveAvgPool1d(4).call(inp.get("nd_seq")));
  out.set("pool::nn.AdaptiveAvgPool3d",
    () => new nn.AdaptiveAvgPool3d(2).call(inp.get("nd_vol")));

  add("F.adaptive_max_pool1d", (x) => F.adaptiveMaxPool1d(x, 4), "nd_seq");
  add("F.adaptive_max_pool2d", (x) => F.adaptiveMaxPool2d(x, 2), "img");
  add("F.adaptive_max_pool2d(안 떨어짐)", (x) => F.adaptiveMaxPool2d(x, 3), "img");
  add("F.adaptive_max_pool3d", (x) => F.adaptiveMaxPool3d(x, 2), "nd_vol");
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

  // ── the pooling arguments that had a seat and no case ──
  //
  // `padding`, `countIncludePad`, `divisorOverride` and `ceilMode` were all reachable
  // here before any of these lines existed. **That is exactly why they needed cases**:
  // a seat that is declared and a seat that works are indistinguishable to the name
  // axis, which counts declared names, so the arguments being present was never
  // evidence that anything happened when they were passed.
  //
  // Every one changes the answer, which is how they were chosen. Four change the
  // **shape** — the padding and the three `ceilMode`s — so a wrong one cannot hide
  // under a tolerance; the other two change the divisor, where a wrong one is a
  // perfectly plausible number.
  //
  // **The argument order is torch's, and it is not the obvious one.** `AvgPool` takes
  // `ceilMode` before `countIncludePad`, and `LPPool` leads with the norm rather than
  // the kernel. Getting either backwards gives a run that works and answers something
  // else, so the Python spellings are transcribed here position by position.
  const layer = (name: string, make: () => nn.Module, key: string, grad: boolean): void => {
    out.set(`pool::nn.${name}`, () => make().call(inp.get(key)) as Tensor);
    if (!grad) return;
    out.set(`pool::grad::nn.${name}`, () => {
      const x = inp.get(key, true);
      seeded(make().call(x) as Tensor).backward();
      return gradOf(x, name);
    });
  };

  // `countIncludePad` needs a padding to have anything to include — hence the 1 — and
  // the pair differs only in that flag, which is what makes the pair the measurement.
  layer("AvgPool1d(테두리 채움)", () => new nn.AvgPool1d(2, 2, 1), "nd_seq", true);
  layer("AvgPool1d(가장자리 빼기)",
    () => new nn.AvgPool1d(2, 2, 1, false, false), "nd_seq", true);
  layer("AvgPool3d(테두리 채움)", () => new nn.AvgPool3d(2, 2, 1), "nd_vol", false);
  // `divisorOverride` needs **overlapping windows** for the difference to show, which
  // is the stride of 1 rather than the kernel's 2.
  layer("AvgPool3d(나눗수 지정)",
    () => new nn.AvgPool3d(2, 1, 0, false, true, 4), "nd_vol", true);
  // **A 3 and not a 2, because the volume is 4³.** Written as a 2 it divides evenly,
  // `ceilMode` changes nothing, and the case freezes what its neighbour already froze.
  layer("AvgPool3d(올림)", () => new nn.AvgPool3d(3, 3, 0, true), "nd_vol", true);
  layer("LPPool1d(올림)", () => new nn.LPPool1d(2, 3, 3, true), "nd_seq", false);
  layer("LPPool2d(올림)", () => new nn.LPPool2d(2, 3, 3, true), "img", false);
  // **The layer honoured `ceilMode` and the function refused it**, on the binding
  // side — with a reason that was true when written and stopped being true the day
  // `lpPool` grew the seat. Here the method takes it directly, so these three rows are
  // what the Python `F.lp_pool*d(…, ceil_mode=True)` reaches.
  for (const [rank, key] of [[1, "nd_seq"], [2, "img"], [3, "nd_vol"]] as
       Array<[number, string]>) {
    out.set(`pool::F.lp_pool${rank}d(올림)`,
      () => inp.get(key).lpPool(2, 3, 3, true));
  }
}

/**
 * Functions newly added on the Python side. **Here they produce the same answer by
 * composition.**
 *
 * No name is added to borch.ts — what grows is not a computation but a spelling for Python
 * to call, and that is the Python side's business.
 */
function addNewFn(out: Map<string, Case>, inp: Inputs): void {
  const x1 = (): Tensor => inp.get("x1");
  const x2 = (): Tensor => inp.get("x2");
  const withnan = (): Tensor =>
    Tensor.from([1, Number.NaN, -Infinity, Infinity, 3], [5]);
  const zeros5 = (): Tensor => Tensor.zeros([5]);

  // The random family cannot have equal values — **the shape is frozen as the answer.**
  for (const name of ["empty_like", "rand_like", "randn_like"]) {
    out.set(`newfn::${name}/모양`, () => x2().shape.join(" "));
  }
  out.set("newfn::randint_like/모양", () => x2().shape.join(" "));

  const table: [string, () => Tensor][] = [
    ["logspace", () => Tensor.full([], 10).binary("pow", Tensor.linspace(0, 2, 5))],
    ["scalar_tensor", () => Tensor.full([], 2.5)],
    // `xy` has its first two axes swapped, so one rule cannot cover it.
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
    // There are only floats, so they are all true — a fact rather than a lie.
    ["isreal", () => Tensor.ones([5]).binary("gt", Tensor.full([], 0))],
    ["isposinf", () => withnan().unary("isinf")
      .mul(withnan().binary("gt", Tensor.full([], 0)).to("float32"))
      .binary("gt", Tensor.full([], 0))],
    ["isneginf", () => withnan().unary("isinf")
      .mul(withnan().binary("lt", Tensor.full([], 0)).to("float32"))
      .binary("gt", Tensor.full([], 0))],
    // **It skips NaN** — `maximum` comes back holding it.
    ["fmax(NaN 건너뜀)", () => nanSkip(withnan(), zeros5(), "maximum")],
    ["fmin(NaN 건너뜀)", () => nanSkip(withnan(), zeros5(), "minimum")],
    ["float_power", () => inp.get("xp").powScalar(2)],
    // **`pow` with a tensor exponent** — the half `powScalar` cannot do, and the
    // reason `pow` is not that method renamed: the elementwise kernel carries a
    // two-sided backward and the scalar one does not.
    ["pow(텐서 지수)",
      () => inp.get("xp").pow(Tensor.from([2, 3, 0.5, 1, 2, 0], [6]))],
    ["pow(수 지수)", () => inp.get("xp").pow(3)],
    ["expand_as", () => Tensor.from([1, 2], [2, 1]).expandAs(Tensor.zeros([2, 3]))],
    ["reshape_as", () => inp.get("xp").reshapeAs(Tensor.zeros([2, 3]))],
    ["view_as", () => inp.get("xp").viewAs(Tensor.zeros([3, 2]))],
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

  // **`equalNan` was taken here and stopped at the binding's boundary.** This
  // method has carried it from the start and `allclose` beside it forwarded it, so
  // the check existed and the word never reached it — NaN against NaN came back
  // `false` where torch says `true`. A fixture without a NaN cannot tell the two
  // apart, which is why the `isclose` row above did not.
  // ── the indices are `int64`, and six of the eight were float32 ─────────────
  //
  // A position is not a value, and `argReduceOver` says exactly that about its own
  // output — which is why `max(dim)` and `min(dim)` were right while `sort`,
  // `topk`, `kthvalue`, `median`, `cummax` and `cummin` handed back floats: those
  // six build their index tensor in two other places, and both left the dtype at
  // its default. It went unseen because the pair had no `repr` on either Python
  // side, so the label was never on screen, and `gather` takes either.
  const pairSrc = (): Tensor => Tensor.from([3, 1, 2, 0.5], [4]);
  const indexKinds: [string, () => Tensor][] = [
    ["topk", () => pairSrc().topk(2).indices],
    ["sort", () => pairSrc().sort().indices],
    ["max(dim)", () => pairSrc().max(0).indices],
    ["min(dim)", () => pairSrc().min(0).indices],
    ["median(dim)", () => pairSrc().median(0).indices],
    ["kthvalue", () => pairSrc().kthvalue(2).indices],
    ["cummax", () => pairSrc().cummax(0).indices],
    ["cummin", () => pairSrc().cummin(0).indices],
  ];
  for (const [label, get] of indexKinds) {
    // The Python side spells a dtype `torch.int64`; over here it is the bare name,
    // and the golden is keyed by the string both must produce.
    out.set(`top::살펴보기::짝::${label} 의 자리는 int64`, () => `torch.${get().dtype}`);
  }

  const nanTrio = (): Tensor => Tensor.from([1, NaN, 3], [3]);
  out.set("newfn::isclose(equal_nan)", () => nanTrio().isclose(nanTrio(), 1e-5, 1e-8, true));
  out.set("newfn::isclose(equal_nan 없이)", () => nanTrio().isclose(nanTrio()));
  // Python's `True`, not JavaScript's `true` — the golden is keyed by the string.
  out.set("newfn::allclose(equal_nan)",
    async () => (await nanTrio().allclose(nanTrio(), 1e-5, 1e-8, true))
      ? "True" : "False");
}

/** NaN and infinity into finite numbers. The fill value is handed over **expanded to the
 *  same shape** too. */
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
 * The **writing** side of indexing. The reading side (`gather`) already existed.
 *
 * **Repeated indices are the point.** `scatterSet` keeps the last write and `scatterAdd`
 * adds — measured with non-repeating indices alone, the two look identical.
 */
function addIndex(out: Map<string, Case>, inp: Inputs): void {
  const base = (): Tensor => Tensor.zeros([3, 4]);
  const src = (grad = false): Tensor => {
    const s = inp.get("x2").mul(Tensor.full([], 10));
    return grad ? asLeaf(s) : s;
  };
  // 0 appears twice — that is the repeated position.
  const dup = (): Tensor =>
    Tensor.from([0, 0, 1, 2, 1, 1, 2, 3, 2, 2, 3, 0], [3, 4]);
  const rows = (values: number[]): Tensor => Tensor.from(values, [values.length]);

  /** Spreads a 1-D index by rows — the shape the `index_add` family uses. */
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
    // `take` gathers from the flattened view — it has no notion of an axis.
    ["take", () => inp.get("x2").reshape([12]).indexSelect(0, rows([0, 2, 2, 5]))],
    ["take_along_dim", () => inp.get("x2").gather(1, dup())],
  ];
  for (const [name, fn] of table) out.set(`index::${name}`, fn);

  out.set("index::grad::scatter_add", () => {
    const s = src(true);
    seeded(base().scatterAdd(1, dup(), s)).backward();
    return gradOf(s, "scatter_add");
  });

  // ── `reduce`, torch's deprecated overload ──
  //
  // **Asked on a non-zero base.** `reduce` combines onto what is already there, so
  // started from `base()` (zeros) `add` would be `scatterAdd` and the case would
  // pass with nothing implemented. `x2` is the base and `dup` repeats an index, so
  // the collisions accumulate here too.
  for (const red of ["add", "multiply"]) {
    out.set(`index::scatter(reduce=${red})`,
      () => inp.get("x2").scatter(1, dup(), src(), red));
    out.set(`index::제자리::scatter_(reduce=${red})`,
      () => inp.get("x2").clone().scatter_(1, dup(), src(), red));
  }

  out.set("index::거절::scatter(reduce=sum)", () => {
    try {
      inp.get("x2").scatter(1, dup(), src(), "sum");
    } catch (e) {
      const said = String(e);
      return said.includes("must be either add or multiply")
        ? "must be either add or multiply" : `다른 문구 <${said.slice(0, 44)}>`;
    }
    return "안 던졌다";
  });

  // **The forward is not where it stops.** torch hands back a tensor that carries
  // `requiresGrad`, and only `backward()` raises — so the refusal lives in the node,
  // not in the call. Written the other way it would read the same to any check that
  // only calls the function, and would take away a value torch returns.
  out.set("index::거절::scatter(reduce) 의 기울기", () => {
    const s = src(true);
    try {
      seeded(inp.get("x2").scatter(1, dup(), s, "add")).backward();
    } catch (e) {
      const said = String(e);
      return said.includes("derivative for aten::scatter is not implemented")
        ? "derivative for aten::scatter is not implemented"
        : `다른 문구 <${said.slice(0, 44)}>`;
    }
    return "안 던졌다";
  });

  // Finding a position inside something sorted.
  //
  // **These three did not call `searchSorted` for a long time.** They were written as
  // broadcasting `seq < want` and summing, the values were exactly equal, and they were
  // green — except that what was being measured then was broadcasting and reduction rather
  // than this name. borch.ts had no such name, so it was also all that could be done at the
  // time.
  const seq = (): Tensor => Tensor.from([1, 3, 5, 7], [4]);
  const want = (): Tensor => Tensor.from([0, 3, 6, 9], [4]);
  out.set("index::searchsorted", () => seq().searchSorted(want()));
  out.set("index::searchsorted(right)", () => seq().searchSorted(want(), true));
  out.set("index::bucketize", () => want().bucketize(seq()));

  // **Taking one thing under two names** is a Python-side matter — reconciling `right` (a
  // boolean) with `side` (a string), and stopping when the two disagree. borch.ts knows one
  // of them, and whether that one is right is what is asked here.
  out.set("index::searchsorted(side=left)", () => seq().searchSorted(want(), false));
  out.set("index::searchsorted(side=right)", () => seq().searchSorted(want(), true));
  out.set("index::searchsorted(side=right, right=True)",
    () => seq().searchSorted(want(), true));

  // Where there is a single boundary, or the value falls outside them. **The two ends of
  // the binary search** — asked in the middle alone, a wrong initial `lo` or `hi` still
  // gives the right answer.
  out.set("index::searchsorted(끝 밖)", () =>
    Tensor.from([2, 4], [2]).searchSorted(Tensor.from([0, 1, 2, 3, 4, 5], [6])));
  out.set("index::searchsorted(경계 하나)", () =>
    Tensor.from([3], [1]).searchSorted(Tensor.from([1, 3, 5], [3]), true));
}

/**
 * Of the names added recently, **the ones a TS user will actually call.**
 *
 * ## Why not all of them
 *
 * The Python golden holds 2,173 cases and only a part of them have bodies here. The gap
 * grew large (over 500), and carrying all of it across would be **asking one question
 * twice** — the binding runner already passes through borch.ts's kernels on those cases, so
 * **the values are being verified**, and it was the binding runner that caught this round's
 * shader errors and `mutate`'s over-copying.
 *
 * What a TS case proves **in addition** is not the value but **this side's surface** —
 * whether the name `asStrided` is in that place and whether the argument order is as
 * stated. That is worth something and not worth 500 cases, and a good many of the remainder
 * ask about **Python name aliases** such as `borch.i0`, where carrying them across really
 * would be one question asked twice.
 *
 * So they were chosen. The criterion is **would somebody writing TS call this name.**
 *
 * ## Only where the inputs are literal
 *
 * A few Python cases build their inputs with `numpy.random.default_rng(0)`. That sequence
 * cannot be built here, so those cases are **not carried across** — filling in with similar
 * values registers the name and diverges the answer, and that is not reporting a defect but
 * **manufacturing one.**
 */
function addRecent(out: Map<string, Case>): void {
  // ── Bitwise and integer (`bit::`) ───────────────────────────────────
  //
  // Negatives and zero have to be present together — whether the right shift is arithmetic,
  // whether `gcd` discards the sign, and whether `lcm(0, 7)` avoids dividing by zero all
  // hang on it.
  const ints = (): Tensor => Tensor.from([12, 10, -3, 0], [4], { dtype: "int64" });
  const rhs = (): Tensor => Tensor.from([10, 3, 5, 7], [4], { dtype: "int64" });
  for (const name of ["bitwise_and", "bitwise_or", "bitwise_xor",
    "bitwise_left_shift", "bitwise_right_shift", "gcd", "lcm"]) {
    out.set(`bit::${name}`, () => ints().binary(name, rhs()));
  }

  // **They are asked by method name too.** The seven above go through the `binary(name)`
  // table, and the line code ported from torch types is `x.gcd(y)` — that name was missing
  // for a long time.
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
  // **Booleans are a different computation** — `~True` is `False` rather than `-2`. Asked
  // with integers alone, this branch never runs.
  const flags4b = (): Tensor => Tensor.from([1, 1, 0, 0], [4], { dtype: "bool" });
  const notFlags = (): Tensor => Tensor.from([0, 0, 1, 1], [4], { dtype: "bool" });
  out.set("bit::bitwise_and(참거짓)", () => flags4b().bitwiseAnd(notFlags()));
  out.set("bit::bitwise_or(참거짓)", () => flags4b().bitwiseOr(notFlags()));
  out.set("bit::bitwise_not(참거짓)", () => flags4b().bitwise_not());

  // The in-place forms. **The same tensor has to come back** for chained code to edit the
  // original.
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
  // It severs the graph **on the same tensor.** Built wrongly with `detach()`, `===`
  // comes out false and the original is still attached above, so backpropagation keeps
  // flowing.
  out.set("bit::detach_", () => {
    const x = Tensor.from([-2.5, 0.5, 1.5, 3.0], [4], { requiresGrad: true });
    const y = x.mul(Tensor.full([], 2));
    const z = y.detach_();
    return `${verdict(z === y)} ${verdict(y.requiresGrad)}`;
  });
  // **`fill` is not in place, unlike `fill_`.** One character apart, the values alone look
  // plausible, and it shows only when whether the original is unchanged is asked
  // separately.
  out.set("bit::fill", () => reals().fill(7.0));
  out.set("bit::fill(원본은 그대로)", () => {
    const x = reals();
    x.fill(7.0);
    return x;
  });
  out.set("bit::i0", () => reals().i0());
  // The series splits at 3.75 — beyond it is asked about separately.
  out.set("bit::i0(큰 값)", () => Tensor.from([4.0, 8.0, 12.0], [3]).i0());
  out.set("bit::nextafter", () =>
    Tensor.from([1.0, 2.0], [2]).binary("nextafter", Tensor.from([2.0, 1.0], [2])));
  const frexpIn = (): Tensor => Tensor.from([1.0, 0.5, 8.0, -3.0], [4]);
  out.set("bit::frexp(가수)", () => frexpIn().frexp().mantissa);
  out.set("bit::frexp(지수)", () => frexpIn().frexp().exponent);
  const gam3 = (): Tensor => Tensor.from([2.0, 3.0, 4.5], [3]);
  out.set("bit::mvlgamma(p=2)", () => gam3().mvlgamma(2));
  out.set("bit::mvlgamma(p=3)", () => gam3().mvlgamma(3));

  // **The axis is asked about both ways** — only with something non-square does a swapped
  // axis get caught on the shape first.
  const grid23 = (): Tensor => Tensor.from([1.0, 2.0, -1.0, 3.0, 4.0, 0.5], [2, 3]);
  for (const dim of [0, 1]) {
    out.set(`bit::logcumsumexp(dim=${dim})`, () => grid23().logcumsumexp(dim));
  }
  // **It counts with uneven weights.** All ones and the accumulation's order cancels, so
  // the rule about accumulating from the back does not show.
  out.set("bit::grad::logcumsumexp", () => {
    const x = Tensor.from([1.0, 2.0, -1.0, 3.0, 4.0, 0.5], [2, 3],
      { requiresGrad: true });
    x.logcumsumexp(1)
      .mul(Tensor.from([1.0, 2.0, 0.5, 0.5, 3.0, 1.5], [2, 3])).sum().backward();
    return gradOf(x, "logcumsumexp");
  });

  // The window functions. **`periodic` is the default and it adds one to the length.**
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
    // The one size at which the divisor becomes 0.
    out.set(`bit::${name}(1)`, () => make(1, true));
  }
  out.set("bit::hamming_window(alpha, beta)",
    () => Tensor.hammingWindow(6, true, 0.5, 0.5));
  out.set("bit::kaiser_window(beta=8)", () => Tensor.kaiserWindow(6, true, 8.0));

  // ── Shape and indexing (`spot::`) ───────────────────────────────────
  const grid = (): Tensor => Tensor.from(
    Array.from({ length: 12 }, (_, i) => i), [3, 4]);
  const line = (): Tensor => Tensor.from(
    Array.from({ length: 10 }, (_, i) => i), [10]);

  out.set("spot::as_strided", () => grid().asStrided([2, 2], [1, 2]));
  out.set("spot::as_strided(offset)", () => grid().asStrided([2, 2], [1, 2], 3));
  // **Overlapping strides.** Without overlap there is nowhere a cell is read twice.
  out.set("spot::as_strided(겹침)", () => grid().asStrided([3, 3], [1, 1]));
  out.set("spot::as_strided_scatter",
    () => grid().asStridedScatter(Tensor.zeros([2, 2]), [2, 2], [1, 2], 3));

  out.set("spot::select_scatter",
    () => grid().selectScatter(Tensor.zeros([4]), 0, 1));
  out.set("spot::slice_scatter",
    () => grid().sliceScatter(Tensor.zeros([3, 2]), 1, 1, 3));
  // **Only with `step` other than 1** does whether the skipped positions are left alone
  // show.
  out.set("spot::slice_scatter(step=2)",
    () => grid().sliceScatter(Tensor.zeros([3, 2]), 1, 0, 4, 2));
  // The length varies with the offset — at (3,4), 0 and 1 give three and -1 gives two.
  for (const [offset, k] of [[-1, 2], [0, 3], [1, 3]] as const) {
    out.set(`spot::diagonal_scatter(offset=${offset})`,
      () => grid().diagonalScatter(Tensor.zeros([k]), offset));
  }
  // **Only with a batch axis** does the convention of the diagonal axis going to the end
  // show.
  out.set("spot::diag_embed(2차)", () => grid().diagEmbed());

  // **The remainder is shared out from the front** — 10 split into 4 is 3, 3, 2, 2.
  for (const k of [3, 4, 5]) {
    out.set(`spot::tensor_split(${k})`, () => Tensor.cat(line().tensorSplit(k), 0));
    // Concatenated, how it was split disappears. The piece sizes themselves are asked
    // about.
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

  // **The indices repeat** — 0 appears twice. The two branches diverge here and nowhere
  // else.
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

  // **The base plate is 2.5.** At 1 it is the identity for multiplication and
  // `include_self` is invisible.
  const base34 = (): Tensor => Tensor.zeros([3, 4]).add(Tensor.full([], 2.5));
  const dup34 = (): Tensor =>
    Tensor.from([0, 0, 1, 2, 1, 1, 2, 3, 2, 2, 3, 0], [3, 4]);
  for (const reduce of ["sum", "prod", "amax", "amin", "mean"]) {
    for (const self of [true, false]) {
      out.set(`spot::scatter_reduce(${reduce}, include_self=${self ? "True" : "False"})`,
        () => base34().scatterReduce(1, dup34(), grid(), reduce, self));
    }
  }

  // The first row is already small and **has to be left alone.** The other two are
  // clipped.
  const tall32 = (): Tensor => Tensor.from([3, 4, 6, 8, 30, 40], [3, 2]);
  for (const p of [1, 2, 3]) {
    out.set(`spot::renorm(p=${p})`, () => tall32().renorm(p, 0, 5.0));
  }
  out.set("spot::renorm(dim=1)", () => tall32().renorm(2, 1, 5.0));

  // **The input contains negatives.** Written with a power, WGSL's `pow` becomes NaN
  // here.
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

  // **Uneven weights.** All ones and the differing shares per position cancel and become
  // invisible.
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
  /** The gradient on the written value's side. It has to flow **only to the positions
   *  written.** */
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
  // The gradient of overlapping strides — it arrives at one cell several times.
  spotGrad("as_strided(겹침)", (x) => x.asStrided([3, 3], [1, 1]),
    () => Tensor.from(Array.from({ length: 9 }, (_, i) => i + 1), [3, 3]));
  spotGrad("select_scatter", (x) => x.selectScatter(Tensor.zeros([4]), 0, 1));
  spotGrad("slice_scatter",
    (x) => x.sliceScatter(Tensor.zeros([3, 2]), 1, 0, 4, 2));
  spotGrad("diagonal_scatter",
    (x) => x.diagonalScatter(Tensor.zeros([3]), 1));
  spotGrad("diag_embed", (x) => x.diagEmbed(),
    () => Tensor.from(Array.from({ length: 48 }, (_, i) => i + 1), [3, 4, 4]));
  // (3,4) split into 3 gives 2, 1, 1, so the middle piece is (3,1).
  spotGrad("tensor_split", (x) => x.tensorSplit(3, 1)[1] ?? Tensor.zeros([3, 1]),
    () => Tensor.from([1.0, 3.0, 5.0], [3, 1]));
  spotGrad("masked_scatter", (x) => x.maskedScatter(mask(), feed()));
  spotGrad("put", (x) => x.put(flatIdx(), flatVal()));
  spotGrad("index_put", (x) => x.indexPut(
    [Tensor.from([0, 1, 0], [3]), Tensor.from([1, 2, 1], [3])],
    Tensor.from([10.0, 20.0, 30.0], [3])));
  // **The gradient of a clipped row.** x is inside the scale, so writing it as `g·s`
  // diverges here.
  spotGrad("renorm", (x) => x.renorm(2, 0, 5.0));
  // `mv` is a matrix product with a 1-D operand — its backward was missing an axis in the
  // core.
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

  // ── Splitting and unravelling indices ───────────────────────────────
  out.set("spot::tensor_split(자리 목록)",
    () => Tensor.cat(line().tensorSplit([2, 5]), 0));
  out.set("spot::tensor_split(dim=1)",
    () => grid().tensorSplit(3, 1)[1] ?? Tensor.zeros([3, 1]));
  out.set("spot::unravel_index",
    () => Tensor.cat(Tensor.from([0, 5, 11], [3]).unravelIndex([3, 4]), 0));

  // ── Consecutive duplicates ──────────────────────────────────────────
  //
  // **It does not sort** — in `[1,1,2,2,2,1,3]` the 1 survives twice. Measured with sorted
  // input alone it is indistinguishable from `unique`.
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

  // ── Writing while combining ─────────────────────────────────────────
  //
  // `index_reduce` has no `sum` — that place is `index_add` (measured).
  for (const reduce of ["prod", "mean", "amax", "amin"]) {
    for (const self of [true, false]) {
      out.set(`spot::index_reduce(${reduce}, include_self=${self ? "True" : "False"})`,
        () => base34().indexReduce(0, Tensor.from([0, 0, 2], [3]), grid(),
          reduce, self));
    }
  }

  // ── Combinations and matrices ───────────────────────────────────────
  const duo = (): Tensor => Tensor.from([4.0, 5.0], [2]);
  out.set("spot::cartesian_prod(둘)",
    () => Tensor.cartesianProd(trio(), duo()));
  // **Given one thing it is simply that** (measured) — it stays 1-D.
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

  // ── In place ────────────────────────────────────────────────────────
  //
  // **The shape has to follow too.** Moving the values alone passes only when asked with a
  // square.
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

  // ── Statistics (`stat::`) ───────────────────────────────────────────
  //
  // **Almost everything here is asynchronous.** For a histogram which bin a value falls in
  // *is* the value, and for `mode` and `nanmedian` which value wins *is* the value, so one
  // read back is required — the case shape differing from elsewhere is that rather than a
  // difference of name.
  const sample = (): Tensor => Tensor.from([0.5, 2.0, 2.0, 3.5, 1.0, 4.0, 2.0], [7]);
  const sampleW = (): Tensor => Tensor.from([1.0, 2.0, 1.0, 1.0, 3.0, 1.0, 1.0], [7]);
  out.set("stat::histc(bins=4)", async () => await sample().histc(4));
  out.set("stat::histc(min/max)", async () => await sample().histc(4, 0.0, 4.0));
  // **Outside the range is discarded** — it is not herded into the end bins.
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
  // **The bin widths differ** — whether `density` divides by a different value per bin is
  // visible here alone.
  out.set("stat::histogram(경계를 직접)",
    async () => (await sample().histogram(Tensor.from([0.0, 1.0, 2.0, 4.0], [4]))).hist);

  const pts = (): Tensor => Tensor.from(
    [0.5, 1.0, 1.5, 1.5, 2.5, 0.5, 0.2, 2.5], [4, 2]);
  out.set("stat::histogramdd 의 hist",
    async () => (await pts().histogramdd([2, 2])).hist);
  out.set("stat::histogramdd 의 edges",
    async () => Tensor.cat((await pts().histogramdd([2, 2])).bin_edges, 0));

  // **There is a tie** — without one, `mode`'s rule (the smaller value wins and the
  // position is the last) does not show.
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
  // **With an even count it takes the lower** — averaging diverges here.
  out.set("stat::nanmedian(짝수 개)",
    async () => await Tensor.from([1.0, 2.0, 3.0, 4.0], [4]).nanmedian() as Tensor);
  // `median` is NaN on a single NaN — side by side is what shows what `nanmedian` is.
  //
  // **A verdict is frozen rather than a value.** The comparison is `allclose`, and NaN
  // differs even from itself.
  out.set("stat::median(NaN 이 섞이면 NaN 이다)", async () => {
    const got = await holes().median(1).values.toArray();
    return Array.from(got).map((v) => verdict(Number.isNaN(v))).join(" ");
  });

  // It is `x²` — where `edge_order=2` becomes exact.
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

  // **Short, it pads; long, it cuts.** Measured at exactly the right size alone, neither
  // branch shows.
  const sparse = (): Tensor => Tensor.from([0.0, 3.0, 0.0, 5.0, 0.0], [5]);
  for (const size of [1, 2, 5]) {
    out.set(`stat::nonzero_static(size=${size})`,
      async () => await sparse().nonzeroStatic(size));
  }
  out.set("stat::nonzero_static(fill=-9)",
    async () => await sparse().nonzeroStatic(5, -9));

  // **The tuple form is what indexing takes** — one 1-D tensor per axis rather than
  // one (count, rank) table. `torch.nonzero` reads as "no signature found for
  // builtin", so the signature axis had never compared it.
  out.set("stat::nonzero(as_tuple)/행",
    async () => (await sparse().nonzero(true))[0]!);
  out.set("stat::nonzero 표", async () => await sparse().nonzero());

  // ── dataconv:: ──────────────────────────────────────────────────────────
  //
  // **This prefix was recorded as unaskable because "borch.ts has no DataLoader".**
  // It has had one since 2026-08-17; the sentence was written on the 23rd. The row
  // now says what is actually missing, and this case is the part that stopped being
  // missing when the samplers were written.
  //
  // A batch sampler decides the grouping, so this asks the thing only it can answer:
  // ten items, batches of four, `dropLast` — **two batches, and the last two items
  // gone.** A loader that re-sliced a flattened order would give three.
  out.set("dataconv::DataLoader(batch_sampler)", async () => {
    const items = new data.TensorDataset(
      Tensor.from([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [10]));
    const loader = new data.DataLoader(items, {
      batchSampler: new data.BatchSampler(new data.SequentialSampler(items), 4, true),
    });
    const seen: string[] = [];
    for (const [batch] of loader) {
      seen.push(Array.from(await (batch as Tensor).toArray())
        .map((v) => String(Math.trunc(v))).join(","));
    }
    return seen.join(" | ");
  });

  // **`DistributedSampler` was declined for the wrong reason and is here now.**
  // It read as *this is one tab* because of its name; given `numReplicas` and
  // `rank` outright it never asks a process group anything.
  //
  // The padding is asked by value because that is where the surprise is: ten rows
  // over three ranks is 4/4/4, not 4/3/3, so rank 1 ends `…, 0`.
  const sliced = (n: number, k: number, r: number, dropLast: boolean): string => {
    const s = new data.DistributedSampler({ length: n }, k, r, false, 0, dropLast);
    return `[${Array.from(s).join(", ")}] len=${s.length} total=${s.totalSize}`;
  };
  for (const [n, k, dropLast] of [
    [10, 3, false], [10, 3, true], [9, 3, false],
    [2, 5, false], [2, 5, true], [7, 1, false], [0, 3, false],
  ] as [number, number, boolean][]) {
    for (let r = 0; r < k; r++) {
      const rank = r;
      out.set(
        `dataconv::DistributedSampler(${n}행 ${k}랭크 drop_last=${verdict(dropLast)})`
        + `/rank ${rank}`,
        () => sliced(n, k, rank, dropLast));
    }
  }

  // The shuffled order is this library's stream rather than torch's, so what is
  // asked is the contract: **the ranks together cover the dataset exactly once**,
  // which holds only if all of them drew the same permutation.
  for (const [n, k] of [[10, 3], [12, 4], [100, 7]] as [number, number][]) {
    out.set(`dataconv::DistributedSampler(shuffle)/${n}행 ${k}랭크가 전부를 덮는다`,
      () => {
        const seen: number[] = [];
        for (let r = 0; r < k; r++) {
          const s = new data.DistributedSampler({ length: n }, k, r, true, 7);
          s.setEpoch(0);
          seen.push(...s);
        }
        const got = Array.from(new Set(seen)).sort((a, b) => a - b);
        return verdict(got.length === n && got.every((v, i) => v === i));
      });
  }

  const epochOrder = (epoch: number): string => {
    const s = new data.DistributedSampler({ length: 12 }, 3, 0, true, 0);
    s.setEpoch(epoch);
    return Array.from(s).join(",");
  };
  out.set("dataconv::DistributedSampler(set_epoch)/같은 epoch 은 재현",
    () => verdict(epochOrder(3) === epochOrder(3)));
  out.set("dataconv::DistributedSampler(set_epoch)/epoch 이 순서를 바꾼다",
    () => verdict(epochOrder(0) !== epochOrder(1)));

  const refuses = (build: () => unknown): string => {
    try {
      build();
    } catch {
      return "거절";
    }
    return "받았다";
  };
  out.set("dataconv::DistributedSampler(범위 밖 rank)",
    () => refuses(() => new data.DistributedSampler({ length: 10 }, 3, 3)));
  out.set("dataconv::DistributedSampler(음수 rank)",
    () => refuses(() => new data.DistributedSampler({ length: 10 }, 3, -1)));

  // **A second session wrote these six the same day, from the Python side.** Its ledger
  // row said they could not be asked here because *borch.ts has no `utils.data` at
  // all*, which the file next door disproves — the samplers, the loader and this class
  // are in `src/data.ts`. Two sessions closing one name from two ends is why the row
  // was written before the other end existed; the answer is to ask them, not to explain
  // them.
  //
  // Their shape differs from the rows above and is worth keeping: **every rank in one
  // answer.** Asked one rank at a time the interleave and a contiguous split look alike
  // on rank 0 — `[0, 2, 4]` one way and `[0, 1, 2]` the other, both starting at zero
  // with the right length.
  const allRanks = (n: number, k: number, dropLast: boolean): string =>
    Array.from({ length: k }, (_, rank) =>
      Array.from(new data.DistributedSampler({ length: n }, k, rank, false, 0, dropLast))
        .join(",")).join(" | ");
  for (const [tag, n, k, dropLast] of [
    ["10 over 2", 10, 2, false],
    ["10 over 3, padded", 10, 3, false],
    ["10 over 3, dropped", 10, 3, true],
    ["7 over 4, padded past its own length", 7, 4, false],
  ] as [string, number, number, boolean][]) {
    out.set(`dataconv::DistributedSampler(${tag})`, () => allRanks(n, k, dropLast));
  }

  // These two answer with the **name of the error class**, so they only line up because
  // borch.ts has a `ValueError` of its own and this class raises it — which it did not
  // until the Python side's measurement said which of torch's two it should be.
  const refusalName = (build: () => unknown): string => {
    try {
      build();
    } catch (e) {
      return (e as Error).name;
    }
    return "예외가 안 났다";
  };
  out.set("dataconv::DistributedSampler(no ranks given)=거절",
    () => refusalName(() => new (data.DistributedSampler as unknown as
      new (...a: unknown[]) => unknown)({ length: 4 })));
  out.set("dataconv::DistributedSampler(rank out of range)=거절",
    () => refusalName(() => new data.DistributedSampler({ length: 4 }, 2, 2)));


  // **`wrap` means something only on a tall matrix.** While it was asked with squares
  // alone, this version invented a rule that does not exist (skipping a row on wrapping
  // round) and was still green. It is asked through the binding as well, and **a place that
  // was wrong is better asked here directly.**
  for (const wrap of [false, true]) {
    out.set(`inplace::짝없이::fill_diagonal_(세로, wrap=${verdict(wrap)})`, () => {
      const x = Tensor.from(Array.from({ length: 18 }, (_, i) => i), [6, 3]);
      x.fillDiagonal_(9, wrap);
      return x;
    });
  }

  // ── The in-place forms built from their partners (`inplace::짝에서::`) ──
  //
  // torch gives almost every operation an underscore partner. What was here was `i0_`
  // alone, and the other thirty-eight were places where **the computation existed and only
  // the name was missing** — ten are torch's second spellings (`divide_` = `div_`), and
  // eleven lived in the kernel table alone and were reachable only as
  // `binary("gcd", …)`. The gap table wrote all forty down as "alias", and only ten were
  // aliases.
  //
  // **Two of them must not be built from their partners.** `bernoulli_(p)` ignores its own
  // value and fills with `p`, unlike `bernoulli()`, which reads its own value as the
  // probability; and `float_power_`'s result is double precision, so torch refuses it
  // too.
  const ints4 = (): Tensor => Tensor.from([6, -4, 3, 9], [4], { dtype: "int64" });
  const flags4 = (): Tensor => Tensor.from([1, 0, 1, 0], [4], { dtype: "bool" });
  const other4 = (): Tensor => Tensor.from([1, 1, 0, 0], [4], { dtype: "bool" });
  const plain4 = (): Tensor => Tensor.from([1, 4, 9, 2], [4]);
  const twos4 = (): Tensor => Tensor.from([2, 2, 2, 2], [4]);
  // Carried verbatim from `at` / `feed` / `mask4` in `tests/cases.py`.
  const at2 = (): Tensor => Tensor.from([0, 2], [2], { dtype: "int64" });
  const feed2 = (): Tensor => Tensor.from([10, 20], [2]);
  const mask4 = (): Tensor => Tensor.from([1, 0, 1, 0], [4], { dtype: "bool" });
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
    // **An alias narrower than what it aliases drops an argument in silence.** `div_`
    // one line up carries `roundingMode` and `divide_` did not, so this call handed
    // the mode to nothing and returned the true quotient — a number, and a plausible
    // one.
    ["divide_(rounding_mode)", plain4, (x) => x.divide_(2, "floor")],
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
    // **The twelve in-place binaries**, each `mutate(binary(name, other))` over a
    // kernel that was already here. They are attached from a table in `tensor.ts`,
    // and a table is what a case list cannot check — the same table would write the
    // cases. So they are written out, here and in `tests/cases.py`.
    ["atan2_", plain4, (x) => x.atan2_(twos4())],
    ["copysign_", plain4, (x) => x.copysign_(twos4())],
    ["eq_", plain4, (x) => x.eq_(twos4())],
    ["ge_", plain4, (x) => x.ge_(twos4())],
    ["gt_", plain4, (x) => x.gt_(twos4())],
    ["heaviside_", plain4, (x) => x.heaviside_(twos4())],
    ["hypot_", plain4, (x) => x.hypot_(twos4())],
    ["ldexp_", plain4, (x) => x.ldexp_(twos4())],
    ["le_", plain4, (x) => x.le_(twos4())],
    ["lt_", plain4, (x) => x.lt_(twos4())],
    ["ne_", plain4, (x) => x.ne_(twos4())],
    ["xlogy_", plain4, (x) => x.xlogy_(twos4())],
    // **Ten in-place forms whose partner was already here.** Every one mutates, and
    // none was held by a case — they went into `tensor.ts` and the only thing that
    // would notice a wrong write is this table.
    ["index_add_", plain4, (x) => x.indexAdd_(0, at2(), feed2())],
    ["index_copy_", plain4, (x) => x.indexCopy_(0, at2(), feed2())],
    ["index_fill_", plain4, (x) => x.indexFill_(0, at2(), 5.0)],
    ["index_reduce_", plain4,
      (x) => x.indexReduce_(0, at2(), feed2(), "amax")],
    ["masked_fill_", plain4, (x) => x.maskedFill_(mask4(), 9.0)],
    ["scatter_", plain4, (x) => x.scatter_(0, at2(), feed2())],
    ["scatter_add_", plain4, (x) => x.scatterAdd_(0, at2(), feed2())],
    ["scatter_reduce_", plain4,
      (x) => x.scatterReduce_(0, at2(), feed2(), "sum")],
    ["swapaxes_", ramp23, (x) => x.swapaxes_(0, 1)],
    ["conj_physical_", plain4, (x) => x.conjPhysical_()],
    // **In-place operations that change the shape.** Asked with squares alone they pass
    // unchanged.
    ["t_", ramp23, (x) => x.t_()],
  ];
  for (const [name, src, run] of pairs) {
    out.set(`inplace::짝에서::${name}`, () => {
      const x = src();
      run(x);
      return x;
    });
  }
  // **It does not read its own value as the probability** — the input is [1,4,9,2] and at
  // `p=0` everything is 0.
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

  // ── The seven that fill by drawing from a distribution (`inplace::분포::`) ──
  //
  // The values cannot be frozen. What can be frozen is that **the shape and dtype do not
  // change**, and the **refusals.**
  //
  // This group was covered in the gap table as "alias / Python", and underneath it sat
  // **fifteen things this code was not doing.** When the first five went in, `parity.ts`
  // received a check measuring the distribution's centre alone and measured no refusal at
  // all — having written down that measuring `normal`'s `std=0` alone is half of it, and
  // then making the same mistake.
  const drawn = (
    name: string, fill: (t: Tensor) => Tensor,
  ): void => {
    out.set(`inplace::분포::${name} 는 모양과 형을 지킨다`, () => {
      const x = Tensor.zeros([2, 3]);
      fill(x);
      return `(${x.shape.join(", ")}) ${dtypeName(x.dtype)}`;
    });
    // **The five continuous ones refuse integers and `geometric_` and `random_` do
    // not.** Grouping them by name as "draws are floats only" is wrong for those two.
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

  // **Each distribution's argument has its own domain** — `p` an open interval, `lambda`
  // positive, `from < to`, `std >= 0`. A fragment of the message is asked about: it is
  // characters rather than a value, so comparing implementations does not catch it.
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

  // **The range reaches as far as the dtype counts exactly.** Past 2²⁴, float32 cannot
  // separate adjacent integers and the values clump. The int64 side is not here — torch's
  // is 2⁶², and this int64 lives in an f32 cell so it cannot count above that, and what
  // cannot be counted is not imitated.
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

  // ── The four random ones — the deterministic extremes alone ─────────
  //
  // The values cannot be frozen (torch's stream differs from ours and there is no way to
  // make them match). **So the extremes are asked about** — without them, `bernoulli` passes
  // even while never looking at the probability at all. "Random, so it cannot be asked" and
  // "it is not asked" are different things.
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
  // The value cannot be asked and **the shape is** — without even that, having the name is
  // all there is.
  out.set("stat::normal(size) 모양",
    () => `(${Tensor.normal(0.0, 1.0, [2, 3]).shape.join(", ")})`);
  out.set("stat::bernoulli 모양",
    () => `(${Tensor.zeros([2, 3]).bernoulli().shape.join(", ")})`);

  // ── The addmm family (`blend::`) ────────────────────────────────────
  //
  // **`beta=0` ignores the value and stays in the graph.** Only the value is asked about
  // here — the gradient side is held by the Python cases.
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
  // **Only a NaN input** shows an implementation written as `input * 0`.
  out.set("blend::addmm(beta=0, input=NaN)", () =>
    Tensor.zeros([2, 4]).add(Tensor.full([], Number.NaN)).addmm(m1(), m2(), 0));
  // **Only with more than one batch** do the combining side and the preserving side
  // diverge.
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

  // ── The remaining thirty-four ───────────────────────────────────────
  //
  // The ones above ask about values alone; below is where **whether an argument actually
  // arrives** is asked.
  //
  // - `addmv` and `addr` never asked about `beta` or `alpha` at all.
  // - Only an `input` **smaller** than the result shows the broadcasting — torch accepts
  //   `(4,)` and a scalar too.
  // - **The gradient separates `beta=0`.** By value alone it is indistinguishable from
  //   being taken out of the graph, and taken out, `input.grad` is not 0 but **absent.**
  //   torch gives 0.
  // - An in-place form has to **return itself.** Given a new tensor,
  //   `x.addmm_(a, b).add_(1)` starts editing a copy rather than the original.
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

  // A broadcast `input`. The result is (2,4) while what it receives is (4,) or a
  // scalar.
  out.set("blend::addmm(input 이 (4,))", () => ones([4]).addmm(m1(), m2()));
  out.set("blend::addmm(input 이 스칼라)", () => ones([]).addmm(m1(), m2()));
  out.set("blend::baddbmm(input 이 (2,4))", () => base24().baddbmm(b1(), b2()));
  out.set("blend::addcmul(브로드캐스트)",
    () => t0().addcmul(Tensor.from([1, 10], [2]), t2()));

  // The gradient. Only **uneven** weights keep the differing shares per position from
  // cancelling.
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
  // **An implementation that took it out of the graph stops here** — with "does not
  // require grad".
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

  // In place. **The value and "does it return itself" are asked separately** — by value
  // alone an implementation returning a copy passes.
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
    // **It has to go through `verdict`.** The golden is Python's `str(bool)`, so `True`,
    // and JS's `String(true)` is `true` — a difference in how it is written rather than in
    // the verdict.
    out.set(`blend::제자리::${name}(같은 텐서)`, () => {
      const x = src();
      return verdict(run(x) === x);
    });
  }

  // ── Complex's neighbours (`make::`) ─────────────────────────────────
  //
  // **Names that have an answer even with no complex number in sight.** Given a real,
  // `real` and `conj` are the thing itself and `angle` is 0 (π for a negative). Being the
  // identity and **being absent are different** — torch code uses the idiom of inserting
  // `resolve_conj()` before handing on a conjugate.
  //
  // **The dtype has to be preserved too.** Returning anything at all on the grounds that it
  // is the identity leaks int64 into float32, and since the values are equal, a case that
  // asks about values alone passes.
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
  // `angle` alone is **always float32** — an angle is a real number.
  for (const [tag, src] of kinds) {
    out.set(`make::angle(${tag})`, () => src().angle());
    out.set(`make::angle(${tag}) 형`, () => dtypeName(src().angle().dtype));
  }
  // Three predicates — all false. **Having no lazy bits is not a reason for the question
  // to lose its meaning.**
  const predicates: [string, (t: Tensor) => boolean][] = [
    ["is_complex", (t) => t.isComplex()],
    ["is_conj", (t) => t.isConj()],
    ["is_neg", (t) => t.isNeg()],
  ];
  for (const [name, fn] of predicates) {
    out.set(`make::${name}`,
      () => kinds.map(([, src]) => verdict(fn(src()))).join(" "));
  }

  // It reads the bytes as they are. The Python side passes `bytes` and this one passes an
  // `ArrayBuffer` — the same twelve bytes, and that is the whole of what this name asks.
  const rawBytes = (): ArrayBuffer => {
    const view = new Float32Array([1.0, 2.0, 3.0]);
    return view.buffer;
  };
  out.set("make::frombuffer", () => Tensor.frombuffer(rawBytes()));
  out.set("make::frombuffer(count=2)",
    () => Tensor.frombuffer(rawBytes(), "float32", 2));
  // **`offset` is in bytes** — read as an element count it diverges here.
  out.set("make::frombuffer(offset=4)",
    () => Tensor.frombuffer(rawBytes(), "float32", -1, 4));

  // **It includes the end** — one cell different from `arange`.
  out.set("make::range(0, 4)", () => Tensor.range(0, 4));
  out.set("make::range(1, 7, 2)", () => Tensor.range(1, 7, 2));
  out.set("make::range(0, 1, 0.25)", () => Tensor.range(0, 1, 0.25));
  // That one cell shows only when **the count is asked about** — a sum or a mean hides
  // it.
  out.set("make::range 와 arange 의 개수",
    () => `${Tensor.range(0, 4).size} ${Tensor.arange(0, 4).size}`);

  // **A step of 0 stops.** Unblocked, `(end-start)/0` becomes Infinity and it blows up
  // where the array is allocated, and that message is indistinguishable from running out of
  // memory.
  //
  // **A fragment** of the message is asked about — it is characters rather than a value, so
  // comparing the implementations does not catch it.
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

  // ── Linear algebra at the top level (`toplin::`) ────────────────────
  //
  // **The argument order is reversed from `linalg`'s.** That place is asked about from TS
  // too.
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
  // **`orgqr` uses the reduced Q and `ormqr` the full one.** Only asked with something
  // tall does that divergence show — measured on a square the two are the same.
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
 * The numerical family. **A mixture of what composes and what is computed by series.**
 *
 * `lgamma`, `digamma` and `erfinv` have no closed form, so approximations are written in
 * WGSL. What is asked here is **whether the f32 computation lands inside the tolerance
 * against torch's double precision**, and that is the whole worth of these cases.
 */
function addNumeric(out: Map<string, Case>, inp: Inputs): void {
  const mat = (): Tensor => inp.get("x2");
  const other = (): Tensor =>
    mat().mul(Tensor.full([], 0.5)).add(Tensor.full([], 1));
  const pos = (grad = false): Tensor => {
    const t = inp.get("xp");
    return grad ? asLeaf(t) : t;
  };
  // The gamma family is looked at on positives alone — diverging at the negative integers
  // is its definition.
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

  /** One trapezoid — the mean of two neighbouring points times the spacing. */
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
    // **The weights are looked at too.** By loss alone it can look similar even with the
    // parameters not moving.
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

  // Convolution and pooling joined **inside** a training loop. A unit comparison cannot
  // see this.
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

  // A schedule is float arithmetic alone, so the values have to be exactly equal. It looks
  // at **the whole trajectory rather than one value** — doing that is how the core caught
  // StepLR's difference.
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
    // `mode` was not a parameter when this case was written, so `0.5` and `1` stood
    // one place to the left. The Python side has always passed them by keyword, which
    // is why the two sides agreed on a value while disagreeing about the call.
    const sch = new optim.ReduceLROnPlateau(opt, "min", 0.5, 1);
    const seen: number[] = [];
    for (const metric of [1.0, 1.0, 1.0, 1.0, 0.1, 1.0, 1.0, 1.0]) {
      sch.step(metric);
      seen.push(opt.paramGroups[0]?.lr ?? 0);
    }
    return Tensor.from(seen, [seen.length]);
  });

  out.set("sched::ReduceLROnPlateau(max)", () => {
    const p = Tensor.from([1.0], [1], { requiresGrad: true });
    const opt = new optim.SGD([p], 1.0);
    const sch = new optim.ReduceLROnPlateau(opt, "max", 0.5, 1);
    const seen: number[] = [];
    for (const metric of [0.1, 0.2, 0.2, 0.2, 0.2, 0.9, 0.2, 0.2]) {
      sch.step(metric);
      seen.push(opt.paramGroups[0]?.lr ?? 0);
    }
    return Tensor.from(seen, [seen.length]);
  });

  // `thresholdMode`, `cooldown` and `eps` — not parameters at all until both sides
  // took torch's argument list. The metrics fall then stall, so a cut happens twice
  // and a cooldown has something to suppress.
  const plateauArgs: [string, (o: optim.SGD) => optim.ReduceLROnPlateau][] = [
    ["threshold_mode=abs", (o) => new optim.ReduceLROnPlateau(
      o, "min", 0.5, 1, 0.1, "abs")],
    ["cooldown", (o) => new optim.ReduceLROnPlateau(
      o, "min", 0.5, 1, 1e-4, "rel", 2)],
    ["eps", (o) => new optim.ReduceLROnPlateau(
      o, "min", 0.5, 1, 1e-4, "rel", 0, 0, 0.4)],
    ["min_lr", (o) => new optim.ReduceLROnPlateau(
      o, "min", 0.5, 1, 1e-4, "rel", 0, 0.3)],
  ];
  for (const [label, make] of plateauArgs) {
    out.set(`sched::ReduceLROnPlateau(${label})`, () => {
      const p = Tensor.from([1.0], [1], { requiresGrad: true });
      const opt = new optim.SGD([p], 1.0);
      const sch = make(opt);
      const seen: number[] = [];
      for (const metric of [1, 1, 0.5, 0.5, 0.5, 0.5, 0.2, 0.5, 0.5, 0.5, 0.5, 0.5]) {
        sch.step(metric);
        seen.push(opt.paramGroups[0]?.lr ?? 0);
      }
      return Tensor.from(seen, [seen.length]);
    });
  }
}

/**
 * Whether the functional form and the method form give the same thing.
 *
 * **Only eight are registered here.** The other twenty (`conv1d`, `conv3d`, pooling,
 * `interpolate`) build their inputs inside the case with `np.random.default_rng`, so they
 * do not ride in `golden.json` — and short of rebuilding numpy's generator in TS there is
 * no way to obtain those values.
 */
function addNdim(out: Map<string, Case>, inp: Inputs): void {
  const flat = () => Tensor.from([0, 1, 2, 3, 4, 5, 6, 7], [2, 4]);
  const mask = () => Tensor.from([1, 0, 1, 0, 1, 0, 1, 0], [2, 4], { dtype: "bool" });

  // The 1-D and 3-D family. A place that opened once the inputs rode in the golden.
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
    // **A place where `mode` was accepted and unused** — asking for bilinear produced
    // nearest. Not an exception but a quietly different value.
    ["nn.Upsample(겹선형)",
      () => new nn.Upsample(null, 2, "bilinear").call(nd("nd_img"))],
    // **The first slot is `size`.** Putting the scale in the first slot makes the same
    // line enlarge in one reading and shrink in the other, and both shapes are plausible.
    // 12 is 3× and so diverges from the default of 2.
    ["nn.Upsample(첫 자리는 size)", () => new nn.Upsample(12).call(nd("nd_img"))],
    // The three the whole-factor cases above were blind to. `interpolate` multiplied
    // without flooring, so 1.5 asked for a fractional number of rows — a shape with a
    // .5 in it, which nearest returned as zeros without throwing.
    // **Three rows, not four.** `nd_img` is 4×4 and 4 × 1.5 = 6 exactly, so flooring
    // and not flooring agree on it — the first draft of these three passed with the
    // floor taken out. Cropped to three rows they part: 3 × 1.5 = 4.5.
    ["nn.Upsample(분수 배율)",
      () => new nn.Upsample(null, 1.5).call(nd("nd_img").narrow(2, 0, 3))],
    ["nn.Upsample(분수 배율, 겹선형)",
      () => new nn.Upsample(null, 1.5, "bilinear")
        .call(nd("nd_img").narrow(2, 0, 3))],
    ["nn.Upsample(분수 배율, recompute)",
      () => new nn.Upsample(null, 1.5, "bilinear", null, true)
        .call(nd("nd_img").narrow(2, 0, 3))],
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

  // **Whether matmul batches and broadcasts.** Every matmul case in this
  // table used to be 2-D, where `matmul` and `mm` are the same function, so
  // this side answered all of them with `mm` and **the absence of a batched
  // path never showed.**
  //
  // The values are the same numbers written again from `tests/cases.py`.
  // Divergent values stop the comparison from being a comparison.
  const m3a = () => Tensor.from([0.0, 0.0333333351, 0.0666666701, 0.1000000015, 0.1333333403, 0.1666666716, 0.200000003, 0.2333333343, 0.2666666806, 0.3000000119, 0.3333333433, 0.3666666746, 0.400000006, 0.4333333373, 0.4666666687, 0.5, 0.5333333611, 0.5666666627, 0.6000000238, 0.6333333254, 0.6666666865, 0.6999999881, 0.7333333492, 0.7666666508, 0.8000000119, 0.8333333135, 0.8666666746, 0.8999999762, 0.9333333373, 0.9666666389], [5, 2, 3]);
  const m3b = () => Tensor.from([0.0, 0.0166666675, 0.0333333351, 0.0500000007, 0.0666666701, 0.0833333358, 0.1000000015, 0.1166666672, 0.1333333403, 0.150000006, 0.1666666716, 0.1833333373, 0.200000003, 0.2166666687, 0.2333333343, 0.25, 0.2666666806, 0.2833333313, 0.3000000119, 0.3166666627, 0.3333333433, 0.349999994, 0.3666666746, 0.3833333254, 0.400000006, 0.4166666567, 0.4333333373, 0.4499999881, 0.4666666687, 0.4833333194, 0.5, 0.5166666508, 0.5333333611, 0.5500000119, 0.5666666627, 0.5833333135, 0.6000000238, 0.6166666746, 0.6333333254, 0.6499999762, 0.6666666865, 0.6833333373, 0.6999999881, 0.7166666389, 0.7333333492, 0.75, 0.7666666508, 0.7833333611, 0.8000000119, 0.8166666627, 0.8333333135, 0.8500000238, 0.8666666746, 0.8833333254, 0.8999999762, 0.9166666865, 0.9333333373, 0.9499999881, 0.9666666389, 0.9833333492], [5, 3, 4]);
  const m2t = () => Tensor.from([0.0, 0.0833333358, 0.1666666716, 0.25, 0.3333333433, 0.4166666567, 0.5, 0.5833333135, 0.6666666865, 0.75, 0.8333333135, 0.9166666865], [3, 4]);
  const v3t = () => Tensor.from([0.0, 0.3333333433, 0.6666666865], [3]);
  const v4t = () => Tensor.from([0.0, 0.25, 0.5, 0.75], [4]);
  const m4a = () => Tensor.from([0.0, 0.0833333358, 0.1666666716, 0.25, 0.3333333433, 0.4166666567, 0.5, 0.5833333135, 0.6666666865, 0.75, 0.8333333135, 0.9166666865], [2, 1, 2, 3]);
  const m4b = () => Tensor.from([0.0, 0.020833334, 0.0416666679, 0.0625, 0.0833333358, 0.1041666642, 0.125, 0.1458333284, 0.1666666716, 0.1875, 0.2083333284, 0.2291666716, 0.25, 0.2708333433, 0.2916666567, 0.3125, 0.3333333433, 0.3541666567, 0.375, 0.3958333433, 0.4166666567, 0.4375, 0.4583333433, 0.4791666567, 0.5, 0.5208333135, 0.5416666865, 0.5625, 0.5833333135, 0.6041666865, 0.625, 0.6458333135, 0.6666666865, 0.6875, 0.7083333135, 0.7291666865, 0.75, 0.7708333135, 0.7916666865, 0.8125, 0.8333333135, 0.8541666865, 0.875, 0.8958333135, 0.9166666865, 0.9375, 0.9583333135, 0.9791666865], [1, 4, 3, 4]);

  // **transpose takes two dimensions and works at any rank.** It took none
  // and refused anything but a matrix until now, and every case here was 2-D,
  // so nothing asked. The negative pair is what models write.
  out.set("transpose::3d", () => m3a().transpose(1, 2));
  out.set("transpose::3d_negative", () => m3a().transpose(-2, -1));
  // Axes 0 and 2 are both 2 here, so the shape does not move and only the
  // values do — a case that a shape check alone would pass.
  out.set("transpose::4d_outer", () => m4a().transpose(0, 2));
  out.set("transpose::swapaxes_agrees", () => m3a().swapaxes(-2, -1));

  out.set("matmul::3d@3d", () => m3a().matmul(m3b()));
  out.set("matmul::3d@2d", () => m3a().matmul(m2t()));
  out.set("matmul::2d@3d", () => m2t().transpose().matmul(m3b()));
  // The borrowed axis comes back out: [3] @ [5,3,4] is [5,4], not [5,1,4].
  out.set("matmul::1d@3d", () => v3t().matmul(m3b()));
  out.set("matmul::3d@1d", () => m3b().matmul(v4t()));
  out.set("matmul::4d_broadcast", () => m4a().matmul(m4b()));
  // The place bimm's ViT had to work around — a Linear taking a 3-D input.
  out.set("matmul::linear_3d", () => m3a().linear(m2t().transpose()));

  out.set("ndim::torch.matmul", () => flat().matmul(flat().transpose()));
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

  // The `webgpu::` cases whose inputs ride in the golden. The rest are unreachable for the
  // reason above.
  out.set("webgpu::F.pad(랭크4)", () => inp.get("img").pad(-1, 1, 2));
  out.set("webgpu::F.pad(랭크4, 값)",
    () => inp.get("img").pad(-1, 2, 1, -1.5).pad(-2, 1, 0, -1.5));
  // `seq_x` turned into (N, C, L). That is how the golden builds it.
  // **Only a leaf accumulates a gradient.** A `permute` result is a derived tensor, so used
  // directly it dies as "no gradient arrived" — through the case rather than the
  // implementation.
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
 * dtype promotion. **It asks which dtype comes out rather than the value.**
 *
 * A refused combination (a subtraction with a bool in it) records the exception's kind as
 * the answer — because refusing is a specification too. The golden holds three kinds with
 * float64 left out: we have no double precision.
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
  // A Python scalar is **weak** — it raises the category and leaves the width alone. Here,
  // where each category has one width, the result reduces to "the higher category".
  //
  // **The dtype must not be inferred from the value.** Python's `2.0` is a float and JS's
  // `2.0` is simply `2`, for which `Number.isInteger` is true. The language has no such
  // distinction, so it is written down here — left to inference,
  // `int64 + a Python float` quietly becomes int64.
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
 * **An operation that only moves elements leaves the dtype alone** — `dtype::자리만::`.
 *
 * The boundary is whether it makes a value. Selecting, slicing, concatenating and
 * relabelling come out in the original dtype, and anything doing arithmetic follows the
 * promotion rules.
 *
 * A defect survived for want of this table — a dataset pulling a sample from `int64` labels
 * produced `float32`, and **because the values were right** nothing in the golden caught
 * it. Comparing implementations by value alone can never see a label falling off.
 *
 * **Until now it was measured only through the binding.** The gap table wrote this group
 * down as "a Python signature matter", and that was the circumstance of the neighbouring
 * `공장::` and `별칭::`; this one asks about `t.dtype` alone and is purely a property of
 * this side.
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
    // The other side has no name `take` — flattening and then selecting is the same
    // thing.
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
    // **Both integers and booleans are asked about.** Asked with one alone, a defect that
    // drops to float32 can survive in the other.
    out.set(`dtype::자리만::${name}(int64)`, () => kept(fn)(ints()));
    out.set(`dtype::자리만::${name}(bool)`, () => kept(fn)(flags()));
  }
  // **`topk` is asked with integers alone.** torch refuses on booleans and its exception
  // kind differs from ours, so that is a matter of refusal wording rather than dtype
  // preservation.
  out.set("dtype::자리만::topk[0](int64)",
    () => kept((t) => t.topk(2, 1).values)(ints()));

  // `maskedSelect` alone sits outside the table above — because **that side is
  // asynchronous** (the result length depends on the values). Forced into the table, all
  // thirty-one rows become asynchronous.
  for (const [tag, src] of [["int64", ints], ["bool", flags]] as const) {
    out.set(`dtype::자리만::masked_select(${tag})`, async () => {
      const t = src();
      return dtypeName((await t.maskedSelect(t.binary("gt", Tensor.full([], 5)))).dtype);
    });
  }

  // ── Three predicates — **an input producing false was measured first** ──
  //
  // A predicate that is always true is not asked about by having a case. All three have an
  // input for which torch really produces false, and they are asked with that input.
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
  // With several it stops — the place that keeps `if tensor:` from quietly looking at the
  // first element.
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
 * Whether `print(t)` matches the real thing. It looks at **characters** rather than values.
 *
 * This is what somebody learning does most, and printed differently the screen does not
 * match the textbook's example.
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
 * Linear algebra. **Everything is asynchronous, because it goes to the CPU and back.**
 *
 * Gradients exist only where there is a closed form. `qr`, `svd`, `pinverse` and `lstsq`
 * give values alone — torch differentiates them and we do not. The derivation is delicate
 * and wrong, it is quietly wrong, so what is absent is left loudly absent.
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
    // **The sign convention differs per implementation.** Flipping a column's sign is the
    // same decomposition, so it is asked about in absolute value.
    ["qr/|Q|", async () => (await mat().qr()).q.abs()],
    ["svd/|U|", async () => (await mat().linalgSvd()).u.abs()],
    ["svd/S", async () => (await mat().linalgSvd()).s],
    ["svd/|Vh|", async () => (await mat().linalgSvd()).vt.abs()],
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

  // ── seven seats the argument axis could not see until this week ────────────
  //
  // torch writes `linalg.solve(A, B, *, left=True, out=None)` with the namespace in
  // front, and the reader that judges these matched the bare name — so `linalg` was
  // judged on 5 of 42 rows and ten divergences sat inside the rest. These are the
  // seven that closed. All ten are keyword-only in torch, so nothing was landing in
  // a wrong seat; the calls simply did not run.
  const rect = (): Tensor => Tensor.from([1, 2, 3, 4, 5, 6, 7, 8, 10], [3, 3]);
  const rhs = (): Tensor => Tensor.from([9, 2, 8, 3], [2, 2]);

  // `left=false` solves `X A = B`, which is a different matrix. Both are asked
  // because a reader ignoring the word answers the first for both.
  for (const left of [true, false]) {
    out.set(`linalg::solve(left=${verdict(left)})`,
      async () => linalg.solve(mat(), rhs(), left));
    out.set(`linalg::solve_ex(left=${verdict(left)})/해`,
      async () => (await linalg.solveEx(mat(), rhs(), left)).result);
  }
  out.set("linalg::solve(left=False, 벡터)=거절", async () => {
    try {
      await linalg.solve(mat(), vec(), false);
    } catch (e) {
      return (e as Error).name;
    }
    return "받았다";
  });

  // `driver=` names a cuSOLVER routine and torch refuses it off CUDA — the sentence
  // is torch's, to its last clause, and `svdvals` answers the same one naming `svd`.
  for (const name of ["svd", "svdvals"]) {
    out.set(`linalg::${name}(driver)=거절`, async () => {
      try {
        if (name === "svd") await linalg.svd(rect(), true, "gesvd");
        else await linalg.svdvals(rect(), "gesvd");
      } catch (e) {
        return `${(e as Error).name}: ${(e as Error).message}`;
      }
      return "받았다";
    });
    out.set(`linalg::${name}(driver=None)/특잇값`, async () =>
      (name === "svd" ? (await linalg.svd(rect(), true, null)).s
        : await linalg.svdvals(rect(), null)));
  }

  // `dtype=` had a seat on `linalg.norm` and not on the two it dispatches to.
  out.set("linalg::vector_norm(dtype)",
    () => linalg.vectorNorm(vec(), 2, undefined, false, "float32"));
  out.set("linalg::matrix_norm(dtype)",
    async () => linalg.matrixNorm(rect(), "fro", [-2, -1], false, "float32"));

  addLinalgStruct(out);
}

/** The inputs `linalg_struct_cases` uses in `tests/cases.py`. Written by hand, verbatim. */
const LA_BATCH = [4, 1, 2, 3, 2, 0, 1, 5, 3, -1, 1, 2];
const LA_BATCH_SYM = [4, 1, 1, 3, 9, 2, 2, 5, 2, 0.5, 0.5, 1];
const LA_BATCH_VEC = [1, 2, 3, 1, 0, 4];
const LA_BATCH_RHS = [1, 0, 2, 1, 0, 3, 1, 1, 2, 2, 0, 1];
// Rank 2 by construction — the second row is twice the first — so `matrix_rank(3.0)`
// cuts it to 1 and the tolerance is asked where it changes the answer.
const LA_RANK2 = [1, 2, 3, 2, 4, 6, 1, 0, 1];
const LA_RECT = [1, 2, 3, 4, 5, 7];
const LA_SYM3 = [4, 1, 0, 1, 3, 1, 0, 1, 2];
const LA_PIVOT = [1, 2, 3, 4];
const LA_SINGULAR = [1, 2, 2, 4];

/**
 * `linalg`'s structure — batches, rectangles, `_ex` and LU.
 *
 * The `addLinalg` above asks with one 2×2 plate. torch's `linalg` is batched throughout and
 * `qr`, `svd` and `pinv` accept rectangles. A golden that asks with one plate can see
 * neither.
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
    // ── Batches ───────────────────────────────────────────────────────
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
    ["batch::svd/S", async () => (await bat().linalgSvd()).s],
    ["batch::eigh/값", async () => (await sym().eigh()).values],
    ["batch::pinv", async () => bat().pinverse()],
    ["batch::logdet", async () => sym().logdet()],
    // 3×3 — at 2×2 there is one Jacobi rotation and the sweeping iteration never runs.
    ["3x3::eigh/값", async () => (await sym3().eigh()).values],
    ["3x3::svd/S", async () => (await sym3().linalgSvd()).s],
    ["3x3::det", async () => sym3().det()],
    ["3x3::inv", async () => sym3().inverse()],

    // ── Rectangles ────────────────────────────────────────────────────
    ["rect::qr/R", async () => (await rect().qr()).r],
    ["rect::qr/|Q|", async () => (await rect().qr()).q.abs()],
    // `qr(false)` is `some=False`, which is `linalg.qr(mode="complete")` — the Python
    // side of this case asks through that other door. Two spellings, one answer, and
    // the golden holds the answer, so it is the place that would catch them parting.
    ["rect::qr(complete)/|Q|", async () => (await rect().qr(false)).q.abs()],
    ["rect::svd/S", async () => (await rect().linalgSvd()).s],
    ["rect::svd/|U|", async () => (await rect().linalgSvd()).u.abs()],
    ["rect::svd(축소)/|U|", async () => (await rect().linalgSvd(false)).u.abs()],
    ["rect::pinv", async () => rect().pinverse()],
    ["rect::matrix_rank", async () => rect().matrixRank()],
    ["rect::lstsq", async () => rect().lstsq(Tensor.from([1, 2, 3], [3]))],

    // ── Asking by name ────────────────────────────────────────────────
    // The Python side asks with `.logabsdet`, `.Q` and `.eigenvalues`. Here the JS names
    // fill those slots and the binding joins the two.
    ["name::slogdet.sign", async () => (await bat().slogdet()).sign],
    ["name::slogdet.logabsdet", async () => (await bat().slogdet()).logabs],
    ["name::qr.R", async () => (await rect().qr()).r],
    ["name::qr.|Q|", async () => (await rect().qr()).q.abs()],
    ["name::svd.S", async () => (await rect().linalgSvd()).s],
    ["name::svd.|Vh|", async () => (await rect().linalgSvd()).vt.abs()],
    ["name::eigh.eigenvalues", async () => (await sym3().eigh()).values],
    ["name::eigh.|eigenvectors|", async () => (await sym3().eigh()).vectors.abs()],

    // ── `_ex` — it gives info instead of throwing ─────────────────────
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

  // **A batch whose first matrix swaps a row and whose second does not.** This side
  // refused a batch outright until now, which was honest and also meant this case
  // could not be written — and the core, which did answer, answered wrongly: one
  // permutation shared across the batch and applied to the batch axis. A refusal on
  // one side and a wrong number on the other is a pair no golden row can hold.
  out.set("linalg::batch::lu_solve(한쪽만 교환)", async () => {
    const a = Tensor.from([1, 2, 3, 4, 5, 1, 1, 2], [2, 2, 2]);
    const f = await a.luFactor();
    return f.LU.luSolveFactored(f.pivots, Tensor.from([1, 0, 0, 1], [2, 2, 1]));
  });

  // **`left` and `adjoint`, which were carried here in order to refuse.** The refusal
  // said each solves a different system than the one these factors were made for —
  // true, and also the reason it could be closed: `Aᵀ = Uᵀ Lᵀ Pᵀ` is the same three
  // pieces in the other order with the permutation on the answer, and `X A = B` is
  // `Aᵀ Xᵀ = Bᵀ`. The matrix is unsymmetric because a symmetric one is its own
  // transpose and would pass with the flag ignored.
  // Its permutation is a three-cycle on purpose: a single row swap is its own
  // inverse, and the adjoint's scatter and the forward's gather agree under one.
  const asym3 = () => Tensor.from([4, 2, -3, 9, -1, -5, 7, -6, 7], [3, 3]);
  const asymRhs = () => Tensor.from([5, 1, -2, 3, 9, 0], [3, 2]);
  const asymRhsT = () => Tensor.from([5, -2, 9, 1, 3, 0], [2, 3]);
  const luModes: [string, boolean, boolean, () => Tensor][] = [
    ["adjoint", true, true, asymRhs],
    ["left=False", false, false, asymRhsT],
    ["left=False, adjoint", false, true, asymRhsT],
  ];
  for (const [tag, left, adjoint, rhs] of luModes) {
    out.set(`linalg::batch::lu_solve(${tag})`, async () => {
      const f = await asym3().luFactor();
      return linalg.luSolve(f.LU, f.pivots, rhs(), left, adjoint);
    });
  }
  // **The adjoint's permutation goes on the answer, and this is where that shows.**
  // The first matrix swaps and the second does not, so a permutation built once for
  // the batch parts from a per-matrix one here too. Forward the swaps are `Pᵀ`; the
  // adjoint wants `P`, which is the same swaps in reverse order.
  out.set("linalg::batch::lu_solve(adjoint, 한쪽만 교환)", async () => {
    const f = await Tensor.from([1, 2, 3, 4, 5, 1, 1, 2], [2, 2, 2]).luFactor();
    return linalg.luSolve(f.LU, f.pivots,
                          Tensor.from([1, 0, 0, 1], [2, 2, 1]), true, true);
  });
  out.set("linalg::batch::lu_solve(left=False, 한쪽만 교환)", async () => {
    const f = await Tensor.from([1, 2, 3, 4, 5, 1, 1, 2], [2, 2, 2]).luFactor();
    return linalg.luSolve(f.LU, f.pivots,
                          Tensor.from([1, 0, 0, 1], [2, 1, 2]), false, false);
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

  // ── A batch's gradients ─────────────────────────────────────────────
  // **This is the place where the values are right and the gradients are not.** The
  // backward's constants differ per batch, and reusing one plate's leaves the first plate
  // right and the rest quietly wrong.
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
 * The `_ex` variants, LDL and the reflectors.
 *
 * **`_ex` reports through `info` rather than throwing** — 0 means it succeeded and `k`
 * means the `k`th pivot was 0 (counted from 1). Measured with matrices that succeed alone
 * that number is always 0 and is indistinguishable from a field that is merely present, so
 * it is asked with a singular matrix too.
 */
function addLinalgEx(out: Map<string, Case>): void {
  const pivot2 = (): Tensor => Tensor.from([1, 2, 3, 4], [2, 2]);
  // It is symmetric — LDL means something only on a symmetric matrix.
  const lin4 = (): Tensor => Tensor.from(
    [2.0, 1.0, 0.5, -1.0, 1.0, 3.0, -0.5, 0.25,
      0.5, -0.5, 2.5, 0.75, -1.0, 0.25, 0.75, 4.0], [4, 4]);
  const singular2 = (): Tensor => Tensor.from([1, 2, 2, 4], [2, 2]);
  // The diagonal is raised by 3 to fill the rank — unraised, the columns are proportional
  // and it is singular.
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
  // **Only asked with a singular matrix** does it show that `info` is not a 0 merely
  // holding a slot.
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
    return got.LD.ldlSolve(got.pivots, Tensor.from(
      [1.0, -2.0, 0.5, 0.25, -1.5, 3.0, 2.0, 0.5], [4, 2]));
  });

  // ── the pivoting, which was refused ───────────────────────────────────
  //
  // `ldlFactor` stopped on any matrix needing a swap, and the reason was accurate:
  // torch uses LAPACK's Bunch–Kaufman and a factorisation without the swaps is a
  // different one. **Both halves are asked**: the first fixture takes a 2×2 block
  // over rows 0 and 1 (`pivots` comes back `[-3, -3, 3]`), the second a 1×1 pivot
  // with a row swap. Writing the swap over rows instead of columns left ten of
  // thirteen matrices agreeing, and the three that did not diverged first in their
  // **pivot table**, two steps after the wrong line.
  const ldlBlock = () => Tensor.from([0, 1, 2, 1, 0, 3, 2, 3, 0], [3, 3]);
  const ldlSwap = () => Tensor.from(
    [-0.31, 0.96, -1.07, 0.96, 0.29, 0.01, -1.07, 0.01, 0.69], [3, 3]);
  // **A 6×6, because the column swap is empty on a 3×3.** The swap runs over the rows
  // below `kp`, and on a 3×3 every pivot that is not the diagonal lands on the last
  // row — dropping the swap entirely changes no answer there. Both plants passed
  // against the two 3×3s.
  const ldlWide = () => Tensor.from(
    [-1.32, -0.52, 1.01, 0.32, 0.47, -0.27,
      -0.52, 0.75, 0.92, -0.22, -1.44, -0.50,
      1.01, 0.92, -1.73, 0.23, -0.71, -0.47,
      0.32, -0.22, 0.23, -0.06, -0.79, -0.32,
      0.47, -1.44, -0.71, -0.79, -0.17, -0.84,
      -0.27, -0.50, -0.47, -0.32, -0.84, -1.09], [6, 6]);
  const ldlPivoted: [string, () => Tensor, number][] = [
    ["2x2 블록", ldlBlock, 3], ["교환", ldlSwap, 3], ["6x6 열 교환", ldlWide, 6],
  ];
  for (const [tag, make, n] of ldlPivoted) {
    out.set(`linalg::ex::ldl_factor(${tag})/LD`,
      async () => (await make().ldlFactor()).LD);
    out.set(`linalg::ex::ldl_factor(${tag})/pivots`,
      async () => (await make().ldlFactor()).pivots);
    // The solve reads the pivot table too, and it did not — it took the packed
    // matrix alone, right only while nothing was ever swapped.
    out.set(`linalg::ex::ldl_solve(${tag})`, async () => {
      const got = await make().ldlFactor();
      return got.LD.ldlSolve(got.pivots, Tensor.from(
        Array.from({ length: n * 2 }, (_, i) => i * 0.5 - 1), [n, 2]));
    });
  }
  // **`info` stopped being a constant.** The only bad cases left are singular, and it
  // is the first zero pivot counting from 1.
  out.set("linalg::ex::ldl_factor_ex(특이)/info",
    async () => (await Tensor.from([1, 1, 1, 1], [2, 2]).ldlFactorEx()).info);
  out.set("linalg::ex::ldl_factor_ex(영행렬)/info",
    async () => (await Tensor.zeros([2, 2]).ldlFactorEx()).info);
  // torch's own `ldl_factor` breaks here with an internal assert rather than saying
  // anything about the matrix, so what is asked is that both stop.
  out.set("linalg::ex::ldl_factor(특이)=둘 다 거절", async () => {
    try {
      await Tensor.from([1, 1, 1, 1], [2, 2]).ldlFactor();
    } catch {
      return "둘 다 멈춘다";
    }
    return "여기선 통과했다";
  });

  // **QR in reflector form.** `geqrf` stores them and `householderProduct` expands them
  // into `Q`.
  //
  // **It has to be asked with a square too.** With everything below the diagonal already 0
  // there is no reflection (`tau = 0`) and the values are left alone, and a square's last
  // column is always in that position — asked with rectangles alone that column never
  // appears, and flipping a sign there goes uncaught.
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
 * `linalg`'s composition layer — where existing things are given a name, and the norms with
 * branches.
 *
 * `matrixExp` alone needs a new computation. The rest are compositions, and there are three
 * places where **the composition is not obvious**: the norm's branches, `diagonal`'s axis,
 * and `eigh` reading one triangle only. The reasons for all three are written in the Python
 * side's comments.
 */
function addLinalgNames(out: Map<string, Case>): void {
  const mat = () => Tensor.from([4, 1, 2, 3], [2, 2]);
  const sym = () => Tensor.from([4, 1, 1, 3], [2, 2]);
  const sym3 = () => Tensor.from(LA_SYM3, [3, 3]);
  const rect = () => Tensor.from(LA_RECT, [3, 2]);
  const vec3 = () => Tensor.from([3, -4, 0], [3]);
  const upper = () => Tensor.from([2, 1, 0, 3], [2, 2]);
  // `tensorsolve(dims)`'s fixture, built the way `tests/cases.py` builds it: a ramp
  // with the folded diagonal raised so every reordering asked stays invertible. The
  // arithmetic runs on tensors rather than in JS so the rounding is float32 on both
  // sides. 2×3×2×3 rather than a cube, because a cube hides what `dims` does to the
  // answer's shape.
  const t6 = () => Tensor.from(Array.from({ length: 36 }, (_, i) => i), [2, 3, 2, 3])
    .mul(Tensor.full([], 0.1))
    .add(Tensor.eye(6).reshape([2, 3, 2, 3]).mul(Tensor.full([], 5)));
  const b6 = () => Tensor.from([1, 2, 3, 4, 5, 6], [2, 3]);
  const cube = () => Tensor.arange(24).reshape([2, 3, 4]);
  // Putting 99 in the upper triangle must not change the answer — the place asking whether
  // it reads the lower triangle alone.
  const skew = () => Tensor.from([4, 99, 1, 3], [2, 2]);
  // For `eig`. The rotation **has no real eigenvalue** (±i), and the general one has three
  // real ones while not being symmetric.
  const rot = () => Tensor.from([0, -1, 1, 0], [2, 2]);
  const gen = () => Tensor.from([4, 1, 2, 0, 3, -1, 1, 0, 2], [3, 3]);

  const value: [string, () => Promise<Tensor>][] = [
    ["name2::matmul", async () => mat().matmul(mat())],
    ["name2::vecdot", async () => mat().vecdot(mat())],
    ["name2::cross", async () => Tensor.from([1, 2, 3], [3])
      .cross(Tensor.from([4, 5, 6], [3]))],
    ["name2::svdvals", async () => mat().svdvals()],
    ["name2::svdvals(직사각)", async () => rect().svdvals()],
    ["name2::eigvalsh", async () => sym().eigvalsh()],
    ["name2::eigvalsh(3x3)", async () => sym3().eigvalsh()],
    ["name2::eigvalsh(아래삼각만)", async () => skew().eigvalsh()],
    ["name2::eigh(아래삼각만)/값", async () => (await skew().eigh()).values],

    // ── `eig` — it accepts non-symmetric matrices too ─────────────────
    //
    // **It does not lean on the order.** LAPACK does not fix the eigenvalue order and torch
    // cannot even sort complex numbers (measured on the Python side). So it is asked by
    // folding through symmetric functions.
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
    // `keepdim` keeps the reduced axis at length 1; with no `dim` torch keeps every
    // axis rather than handing back a scalar. Both branches are asked because they are
    // written apart — one threads the flag through the reduction, the other reshapes.
    ["name2::vector_norm(dim, keepdim)", async () => mat().vectorNorm(2, 1, true)],
    ["name2::vector_norm(keepdim, no dim)",
      async () => mat().vectorNorm(2, undefined, true)],
    ["name2::norm(ord)", async () => linalg.norm(mat(), 2)],
    ["name2::norm(ord, dim)", async () => linalg.norm(mat(), 2, 1)],
    ["name2::norm(ord, dim, keepdim)", async () => linalg.norm(mat(), 2, 1, true)],

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
    // **`dims` was refused here and it is a permute away.** It moves those axes to
    // the end before the fold, in the order given, so it changes which axes become
    // the matrix — the values and the answer's *shape*. The fixture is 2×3×2×3
    // rather than the cubic one above, because a cube hides the shape half.
    ["name2::tensorsolve(dims 없이, 2×3×2×3)",
      async () => t6().tensorSolve(b6())],
    ...(([["(0, 1)", [0, 1]], ["(1, 0)", [1, 0]], ["(0,)", [0]], ["(2, 3)", [2, 3]],
          ["(3,)", [3]], ["(1, 2)", [1, 2]]] as [string, number[]][])
      .map(([tag, dims]): [string, () => Promise<Tensor>] =>
        [`name2::tensorsolve(dims=${tag})`,
          async () => linalg.tensorsolve(t6(), b6(), dims)])),

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

  // ── `torch.norm`'s two words ────────────────────────────────────────────
  //
  // `matrixNorm(ord="nuc")` above has been here for a long time and the top-level
  // `torch.norm(A, "nuc")` had not — the binding refused it as *needing an SVD*, a
  // reason that named a computation this side already had. `"fro"` is the elementwise
  // 2-norm under another name, so it is `vectorNorm`, not a per-matrix anything.
  //
  // These are the rows the Python `torch.norm` reaches through the binding, which
  // does its own dispatch on `p`; the two shapes are what separate the words.
  out.set("linalg::name2::torch.norm(nuc)", async () => mat().matrixNorm("nuc"));
  out.set("linalg::name2::torch.norm(nuc, keepdim)",
    async () => mat().matrixNorm("nuc", [-2, -1], true));
  out.set("linalg::name2::torch.norm(nuc, dim 뒤집기)",
    async () => mat().matrixNorm("nuc", [1, 0]));
  out.set("linalg::name2::torch.norm(fro)", () => mat().vectorNorm(2));
  out.set("linalg::name2::torch.norm(fro, keepdim)",
    () => mat().vectorNorm(2).reshape([1, 1]));
  out.set("linalg::name2::torch.norm(배치, nuc, dim=(0,1))",
    async () => Tensor.from(LA_BATCH, [3, 2, 2]).matrixNorm("nuc", [0, 1]));
  out.set("linalg::name2::torch.norm(배치, nuc, dim=(1,2), keepdim)",
    async () => Tensor.from(LA_BATCH, [3, 2, 2]).matrixNorm("nuc", [1, 2], true));
  out.set("linalg::name2::torch.norm(배치, fro)",
    () => Tensor.from(LA_BATCH, [3, 2, 2]).vectorNorm(2));
  out.set("linalg::name2::torch.norm(배치, fro, dim=(1,2))",
    () => Tensor.from(LA_BATCH, [3, 2, 2]).square().sumDim(1).sumDim(1).sqrt());

  // ── the six seats `linalg` was short of ─────────────────────────────────
  //
  // Each was an argument **JavaScript received and dropped**, so the default answer
  // came back under the name of a computation nobody ran. Three of them are computed
  // here and three refused; both halves are asked, because a seat that only ever
  // refuses and a seat that is not there look the same to a caller who never passes
  // the argument.
  out.set("linalg::name2::matrix_norm(dim)",
    async () => linalg.matrixNorm(Tensor.from(LA_BATCH, [3, 2, 2]), "fro", [0, 1]));
  out.set("linalg::name2::matrix_norm(keepdim)",
    async () => linalg.matrixNorm(mat(), "fro", [-2, -1], true));
  out.set("linalg::name2::matrix_rank(tol)",
    async () => linalg.matrixRank(Tensor.from(LA_RANK2, [3, 3]), 3.0));
  // **`pinv`'s cut-off stopped being a refusal here** — the solver takes one now, and
  // the core has matched torch on it from the start.
  out.set("linalg::name2::pinv(rcond)",
    async () => linalg.pinv(Tensor.from(LA_RANK2, [3, 3]), 0.5));

  // **Where torch's four drivers separate.** At full rank all four agree to float
  // noise; with `rcond=0.9` on this fixture `gels` gives 3.50/1.40, `gelsy` 0.77/2.31
  // and `gelsd`/`gelss` 0.79/2.32. This is the SVD, so `gels` takes no cutoff and
  // `gelsd` is exact, while `gelsy` — *the default* — is refused where the cutoff
  // bites, there being no pivoted QR here to produce its answer with.
  const tall = () => Tensor.from([1, 1, 1, 2, 1, 3, 1, 4], [4, 2]);
  const tallRhs = () => Tensor.from([6, 5, 7, 10], [4, 1]);
  for (const drv of ["gels", "gelsd", "gelss"] as const) {
    out.set(`linalg::name2::lstsq(rcond, ${drv})`,
      async () => linalg.lstsq(tall(), tallRhs(), 0.9, drv));
  }
  out.set("linalg::name2::lstsq(rcond 이 안 자를 때)",
    async () => linalg.lstsq(tall(), tallRhs(), 1e-6));

  // **A batch, which was refused here until `pinverse`'s own batching was noticed.**
  // The second matrix is not the first, so a solve that computes one and stretches it
  // agrees with a batch of copies and with nothing else. The free function gives the
  // solution only — the other three fields are the core's `_Lstsq`, which has no seat
  // on this side — so those cases stay Python's.
  const batchTall = () => Tensor.from(
    [1, 1, 1, 2, 1, 3, 1, 4, 2, 0, 0, 1, 1, 1, 3, 2], [2, 4, 2]);
  const batchRhs = () => Tensor.from(
    [6, 1, 5, 2, 7, 3, 10, 4, 1, 0, 2, 1, 3, 2, 4, 3], [2, 4, 2]);
  out.set("linalg::batch::lstsq(gelsd).solution",
    async () => linalg.lstsq(
      batchTall(), Tensor.from([6, 5, 7, 10, 1, 2, 3, 4], [2, 4]),
      undefined, "gelsd"));
  out.set("linalg::batch::lstsq(행렬 우변)",
    async () => linalg.lstsq(batchTall(), batchRhs(), undefined, "gelsd"));
  // The matrix reading broadcasts its leading dimensions and the vector reading does
  // not — the pair is what tells the two apart.
  out.set("linalg::batch::lstsq(우변 하나를 늘린다)",
    async () => linalg.lstsq(
      batchTall(), Tensor.from([6, 1, 5, 2, 7, 3, 10, 4], [1, 4, 2]),
      undefined, "gelsd"));
  // **The cut-off is per matrix.** Singular values 10/9.5 and 1/0.95 — at `rcond=0.9`
  // neither matrix loses one, so `gelsy` goes through. Scale the batch against one
  // shared largest and the second keeps nothing and the call is refused where torch
  // answers, which is why this pair is *un*refused rather than refused.
  out.set("linalg::batch::lstsq(잘림은 행렬마다 본다)",
    async () => linalg.lstsq(
      Tensor.from([10, 0, 0, 9.5, 0, 0, 0, 0, 1, 0, 0, 0.95, 0, 0, 0, 0], [2, 4, 2]),
      Tensor.from([6, 5, 7, 10, 1, 2, 3, 4], [2, 4]), 0.9));
  // Both matrices lose a singular value here, so `gelsy` is refused — the wording is
  // the one `refusal_case` freezes. `laRefuses` below writes a `name2::` prefix, so
  // this one is spelled out rather than borrowed.
  out.set("linalg::batch::lstsq(둘 다 잘리면)=우리는거절", async () => {
    try {
      await linalg.lstsq(batchTall(),
                         Tensor.from([6, 5, 7, 10, 1, 2, 3, 4], [2, 4]), 0.9);
    } catch (err) {
      return (err as Error).message.includes("is not in the browser subset")
        ? "기대대로"
        : `다른 문구 <${(err as Error).message.slice(0, 44)}>`;
    }
    return "뜻밖의 성공";
  });
  // **A batch of one, because two is the number that hid a defect.** The rank bound is
  // `min(m, n)`; written as `Math.min(...shape)` over a `[2, 4, 2]` batch it comes to 2
  // either way and the plant survived the case above untouched. At `[1, 4, 2]` the same
  // expression gives 1, the bitten matrix keeps 1, and `1 < 1` is false — the refusal
  // vanishes and torch's answer comes back from an algorithm nobody ran.
  out.set("linalg::batch::lstsq(하나짜리 배치도 잘린다)=우리는거절", async () => {
    try {
      await linalg.lstsq(Tensor.from([1, 1, 1, 2, 1, 3, 1, 4], [1, 4, 2]),
                         Tensor.from([6, 5, 7, 10], [1, 4]), 0.9);
    } catch (err) {
      return (err as Error).message.includes("is not in the browser subset")
        ? "기대대로"
        : `다른 문구 <${(err as Error).message.slice(0, 44)}>`;
    }
    return "뜻밖의 성공";
  });
  out.set("linalg::batch::lstsq(벡터 우변은 안 늘어난다)=둘 다 거절", async () => {
    try {
      await linalg.lstsq(batchTall(), Tensor.from([6, 5, 7, 10], [1, 4]),
                         undefined, "gelsd");
    } catch {
      return "둘 다 멈춘다";
    }
    return "여기선 통과했다";
  });

  const laRefuses = (name: string, body: () => unknown) => {
    out.set(`linalg::name2::${name}=우리는거절`, async () => {
      try {
        await body();
      } catch (err) {
        return (err as Error).message.includes("is not in the browser subset")
          ? "기대대로"
          : `다른 문구 <${(err as Error).message.slice(0, 44)}>`;
      }
      return "뜻밖의 성공";
    });
  };
  laRefuses("lstsq(기본 driver 가 자르면)",
    () => linalg.lstsq(tall(), tallRhs(), 0.9));
  // **torch refuses this one too** — *LU without pivoting is not implemented on the
  // CPU* — so what is asked is that both stop, not that this one does.
  out.set("linalg::name2::lu(pivot=False)=둘 다 거절", async () => {
    try {
      await linalg.lu(mat(), false);
    } catch {
      return "둘 다 멈춘다";
    }
    return "여기선 통과했다";
  });
  for (const [tag, pv] of [["기본", null], ["fro", "fro"], ["nuc", "nuc"], ["2", 2],
    ["-2", -2], ["1", 1], ["inf", Infinity]] as const) {
    out.set(`linalg::name2::cond(p=${tag})`, async () => mat().cond(pv));
  }
  // **The answer is always complex.** Even given a symmetric matrix — because things like
  // a rotation matrix have no real eigenvalue at all, the return type cannot vary with the
  // input.
  out.set("linalg::eig::eigvals(대칭이어도 복소수형)",
    async () => `torch.${(await sym().eigvals()).dtype}`);

  // **The sum is the trace.** It is independent of order and asks mathematically whether
  // the values are right — an implementation producing arbitrary eigenvalues can pass a
  // magnitude sort and is caught here.
  out.set("linalg::eig::eigvals(비대칭)/합=대각합", async () => {
    const sum = await (await gen().eigvals()).sum().real().item();
    const tr = await gen().trace().item();
    return `${sum.toFixed(4)} ${tr.toFixed(4)}`;
  });

  // **The eigenvectors cannot be frozen** — the sign is not fixed (torch itself gives
  // opposite signs at float32 and float64). The definition is asked instead: does `A·V`
  // equal `V·diag(λ)`. A flipped sign flips both sides together and the answer does not
  // change.
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
 * The decompositions' gradients.
 *
 * They were left out for a long time — the derivation is delicate and wrong, it is quietly
 * wrong. What changed is not that the derivation got easier but that **there is now
 * something to compare against.** The golden holds real torch's numbers position by
 * position, so being wrong is loud rather than quiet.
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
    // The eigenvectors are asked about **squared** — the column signs are the
    // implementation's choice and Jacobi and LAPACK choose differently. `V∘V` is
    // independent of that. The reason is written in the Python side's comment.
    ["eigh/벡터²", "sym", async (x) => (await x.eigh()).vectors.square()],
    ["eigh/벡터²(3x3)", "sym3", async (x) => (await x.eigh()).vectors.square()],
    ["qr/R", "mat", async (x) => (await x.qr()).r],
    ["qr/Q", "mat", async (x) => (await x.qr()).q],
    ["qr/R(직사각)", "rect", async (x) => (await x.qr()).r],
    ["qr/Q(직사각)", "rect", async (x) => (await x.qr()).q],
    ["pinv", "mat", async (x) => x.pinverse()],
    // **The rectangle is the real test.** On a square an omitted term becomes 0 and does
    // not show.
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

/** The inputs `inplace_cases` uses in `tests/cases.py`. */
const IP_PLAIN = [1.0, 4.0, 9.0, 2.0];
const IP_SMALL = [0.5, 0.8, 0.3, 0.9]; // 정의역이 좁은 것들용

/** The ones whose domain is narrow enough to need `small`. */
const IP_NARROW = new Set(["log", "log2", "log10", "sqrt", "rsqrt", "log1p"]);

/**
 * In-place operations.
 *
 * **It looks at the original rather than at what came back** — building a new tensor and
 * returning it is not in place, and a check looking at the return value alone still
 * passes.
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
  // **Chaining is the real test.** Only what it returns being itself keeps the chain.
  each("이어 부르기", (x) => x.mul_(2).add_(1).clamp_(0, 10));

  // ── the seats the in-place halves were short of ──
  //
  // **In place means the same arithmetic written back**, so the two spellings of one
  // operation take one list — and these took different ones. `round` and `logit` were
  // given their argument by hand after the `UNARY` table (which cannot give one) and
  // `round_`/`logit_` were left with the table's empty signature, so `x.round_(2)`
  // was a word JavaScript dropped and the answer came back rounded to whole numbers.
  //
  // The Python side had it too — `logit_` was built nullary there — so **both
  // libraries agreed by being wrong the same way**, which is the one arrangement the
  // golden cannot catch on its own. What found it was teaching the core's generated
  // forwarders to declare what they forward; until then the signature axis filed
  // ninety-seven `Tensor` rows as *no python signature*.
  each("round_(decimals)", (x) => x.round_(1), [1.234, 4.567, 9.876, 2.345]);
  each("logit_(eps)", (x) => x.logit_(0.1), [0.0, 0.2, 0.5, 1.0]);
  each("cumsum_(dim)", (x) => x.cumsum_(0));
  each("cumprod_(dim)", (x) => x.cumprod_(0));
  // torch calls it `values`; this side called it `other`. JavaScript has no keyword
  // arguments, so only the axis can see that — the value is asked here all the same.
  out.set("inplace::heaviside_(values 라는 이름)", () => {
    const x = Tensor.from([-1, 0, 1, 0], [4]);
    x.heaviside_(Tensor.from([1, 1, 1, 1], [4]));
    return x;
  });

  for (const name of ["abs", "sqrt", "exp", "log", "sin", "cos", "tan", "tanh",
    "sigmoid", "relu", "erf", "floor", "ceil", "round", "sign", "reciprocal",
    "square", "trunc", "frac", "neg", "rsqrt", "log2", "log10", "expm1",
    "log1p", "sinh", "cosh"]) {
    each(`${name}_`, (x) => x.inplaceUnary(name),
      IP_NARROW.has(name) ? IP_SMALL : IP_PLAIN);
  }

  /**
   * **A place only the sister library refuses.**
   *
   * The golden asks not about the value but about "did it behave as documented" — for torch
   * succeeding is the right answer and for the sister library refusing is, so asked by value
   * it would stay diverged forever. borch.ts is not the sister library. It shares the
   * buffer so it spreads, and our answer is torch's — `기대대로`.
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

  // All three refuse a leaf with gradients switched on.
  out.set("inplace::잎 제자리 수정=거절", () => {
    const x = Tensor.from(IP_PLAIN, [4], { requiresGrad: true });
    try {
      x.add_(1);
    } catch {
      return "기대대로 거절";
    }
    return "뜻밖의 성공";
  });

  // Inside `no_grad` even a leaf can be edited — the optimiser really does that.
  out.set("inplace::no_grad 안에서는 된다", () => {
    const x = Tensor.from(IP_PLAIN, [4], { requiresGrad: true });
    noGrad(() => x.add_(1));
    return x;
  });

  // ── keeping a derived tensor's gradient (`inplace::기울기::`) ──
  //
  // `retainGrad()` and `backward(…, inputs)` are **one mechanism**: a node that is
  // not a leaf being able to hold a `grad`. This side wrote into leaves only, so
  // `retain_grad` in the binding raised the flag and did nothing, and the Python
  // case table said in as many words that the value "cannot be asked together".
  // It can now, and these are that question.
  const WIDE = [0, 1, 2, 3, 4, 5];
  const wide = (): Tensor => Tensor.from(WIDE, [2, 3], { requiresGrad: true });

  out.set("inplace::기울기::retain_grad 가 값을 정말 남긴다", () => {
    const x = wide();
    const m = x.mul(Tensor.full([], 3));
    m.retainGrad();
    m.mul(m).sum().backward();
    return gradOf(m, "retain_grad");
  });

  out.set("inplace::기울기::backward(inputs) 는 중간 노드도 채운다", () => {
    const x = wide();
    const m = x.mul(Tensor.full([], 3));
    m.mul(m).sum().backward(undefined, false, false, [m]);
    return gradOf(m, "inputs");
  });

  out.set("inplace::기울기::backward(inputs) 는 안 부른 잎을 안 건드린다", () => {
    const a = wide();
    const b = wide();
    a.mul(b).sum().backward(undefined, false, false, [a]);
    return `a=${a.grad === null ? "없다" : "있다"} b=${b.grad === null ? "없다" : "있다"}`;
  });

  out.set("inplace::기울기::backward(inputs) 로 부른 잎의 값", () => {
    const a = wide();
    a.mul(wide()).sum().backward(undefined, false, false, [a]);
    return gradOf(a, "a");
  });

  // **The two rules do not cancel** — `inputs` adds retention, it does not take it
  // away from a node that asked for it.
  out.set("inplace::기울기::retain_grad 는 inputs 밖에서도 남는다", () => {
    const x = wide();
    const m = x.mul(Tensor.full([], 3));
    m.retainGrad();
    m.mul(m).sum().backward(undefined, false, false, [x]);
    gradOf(x, "x");
    return gradOf(m, "m");
  });

  // ── `Tensor.grad` — the gradient handed back, not stored ───────────────────
  //
  // The tape's own walk with the accumulation left off. **It was absent from all
  // three** and nobody had written a reason; on the Python side the miss came out
  // as `AttributeError: module 'borch' has no attribute 'autograd'`, which names
  // neither gradients nor what is missing.
  const gradTaken = (t: Tensor | null): Tensor =>
    t === null ? Tensor.full([], -999) : t;

  out.set("inplace::기울기::autograd.grad", () => {
    const x = wide();
    return gradTaken(Tensor.grad(x.mul(x).sum(), x)[0] ?? null);
  });

  // **The whole reason the function exists.** Asked by the returned value alone,
  // an implementation that also accumulated would pass.
  out.set("inplace::기울기::autograd.grad 는 .grad 를 안 건드린다", () => {
    const x = wide();
    Tensor.grad(x.mul(x).sum(), x);
    return `grad=${x.grad === null ? "None" : "있다"}`;
  });

  out.set("inplace::기울기::autograd.grad(입력 둘)", () => {
    const a = wide();
    const b = wide();
    const got = Tensor.grad(a.mul(b).sum(), [a, b]);
    return Tensor.stack([gradTaken(got[0] ?? null), gradTaken(got[1] ?? null)]);
  });

  // **Not only a leaf** — torch differentiates against an intermediate, which is
  // how a saliency map is taken against an activation.
  out.set("inplace::기울기::autograd.grad(중간 텐서)", () => {
    const m = wide().mul(Tensor.full([], 3));
    return gradTaken(Tensor.grad(m.mul(m).sum(), m)[0] ?? null);
  });

  // Several outputs are seeded together and their gradients **sum where the graphs
  // meet** (measured: `2x + 5`). A walk written for one root gives one of the two
  // halves and looks entirely reasonable.
  out.set("inplace::기울기::autograd.grad(출력 둘)", () => {
    const x = wide();
    return gradTaken(
      Tensor.grad([x.mul(x).sum(), x.mul(Tensor.full([], 5)).sum()], x)[0] ?? null);
  });

  out.set("inplace::기울기::autograd.grad(grad_outputs)", () => {
    const x = wide();
    return gradTaken(
      Tensor.grad(x.mul(Tensor.full([], 2)), x, Tensor.ones([2, 3]))[0] ?? null);
  });

  out.set("inplace::기울기::autograd.grad(allow_unused)", () => {
    const a = wide();
    const b = wide();
    const got = Tensor.grad(a.mul(Tensor.full([], 2)).sum(), [a, b],
                            undefined, false, true);
    return `둘째=${got[1] === null ? "None" : "있다"} 첫째=${
      got[0] ? "있다" : "None"}`;
  });

  // Zeros rather than null — and measured: torch does **not** ask for
  // `allow_unused` alongside it, though the documentation reads as though it would.
  out.set("inplace::기울기::autograd.grad(materialize_grads)", () => {
    const a = wide();
    const b = wide();
    const got = Tensor.grad(a.mul(Tensor.full([], 2)).sum(), [a, b],
                            undefined, false, false, true);
    return Tensor.stack([gradTaken(got[0] ?? null), gradTaken(got[1] ?? null)]);
  });

  // It releases the graph by default, as `backward()` does. **The two walks are
  // summed** — the second one alone is `2x`, the same answer the plain case gives,
  // so the name would hang an argument that changed nothing.
  out.set("inplace::기울기::autograd.grad(retain_graph)", () => {
    const x = wide();
    const y = x.mul(x).sum();
    const first = gradTaken(Tensor.grad(y, x, undefined, true)[0] ?? null);
    return first.add(gradTaken(Tensor.grad(y, x)[0] ?? null));
  });

  // The pair to the case two above: one fills `grad` and one must not.
  out.set("inplace::기울기::autograd.backward 는 쌓는다", () => {
    const x = wide();
    x.mul(x).sum().backward();
    return gradOf(x, "autograd.backward");
  });

  const refusesBackward = (name: string, fragment: string, body: () => void) => {
    out.set(`inplace::기울기::거절::${name}`, () => {
      try {
        body();
      } catch (err) {
        const said = String(err);
        return said.includes(fragment)
          ? fragment : `다른 문구 <${said.slice(0, 44)}>`;
      }
      return "안 던졌다";
    });
  };

  // Empty stops ahead of every other refusal, which is torch's order.
  refusesBackward("빈 inputs", "cannot be empty",
    () => wide().mul(Tensor.full([], 2)).sum().backward(undefined, false, false, []));
  refusesBackward("grad 없는 것을 inputs 에", "requires_grad=False",
    () => wide().mul(Tensor.full([], 2)).sum()
      .backward(undefined, false, false, [Tensor.from(WIDE, [2, 3])]));

  refusesBackward("autograd.grad(안 쓰인 입력)", "not have been used in the graph",
    () => { Tensor.grad(wide().mul(Tensor.full([], 2)).sum(), [wide()]); });
  refusesBackward("autograd.grad(빈 inputs)", "cannot be empty",
    () => { Tensor.grad(wide().mul(Tensor.full([], 2)).sum(), []); });
  refusesBackward("autograd.grad(grad 없는 입력)", "does not require grad",
    () => {
      Tensor.grad(wide().mul(Tensor.full([], 2)).sum(), Tensor.from(WIDE, [2, 3]));
    });
  refusesBackward("autograd.grad(씨앗 없는 벡터 출력)", "only for scalar outputs",
    () => { Tensor.grad(wide().mul(Tensor.full([], 2)), wide()); });
  refusesBackward("autograd.grad(모양이 틀린 grad_outputs)", "Mismatch in shape",
    () => {
      const x = wide();
      Tensor.grad(x.mul(Tensor.full([], 2)), x, Tensor.ones([9]));
    });

  // torch computes this one; all three of ours refuse in the specified words.
  out.set("inplace::기울기::backward(create_graph)=우리는거절", () => {
    try {
      wide().mul(Tensor.full([], 2)).sum().backward(undefined, false, true);
    } catch (err) {
      const said = String(err);
      return said.includes("is not in the browser subset")
        ? "기대대로" : `다른 문구 <${said.slice(0, 44)}>`;
    }
    return "뜻밖의 성공";
  });

  // ── One computation under two names (`method2::`) ───────────────────
  //
  // `torch.add(x, y)` and `x.add(y)`. This repository had the loop in one direction only,
  // and without the other **the computation was all there and the name reached it from one
  // side alone.** The form a textbook types is the method, and what comes out then is an
  // `AttributeError`.
  //
  // **Checking only that the name resolves passes a shell too** — asking about the value is
  // what shows whether the name really reached that computation.
  const m2a = (): Tensor => Tensor.from([1, 2, 3, 4], [2, 2]);
  const m2b = (): Tensor => Tensor.from([0.5, 1.5, 2.5, 3.5], [2, 2]);
  const m2sym = (): Tensor => Tensor.from([4, 1, 1, 3], [2, 2]);
  const m2neg = (): Tensor => Tensor.from([-1, 2, -3, 0.5], [2, 2]);
  const vec3a = (): Tensor => Tensor.from([1, 2, 3], [3]);
  // `kron`'s two-dimensional pair, carried verbatim from `tests/cases.py`.
  const k2a = (): Tensor => Tensor.from([1, 2, 3, 4], [2, 2]);
  const k2b = (): Tensor => Tensor.from([0, 5, 6, 7], [2, 2]);
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
    // **Above one axis, where it used to refuse.** A 1-D case cannot tell the
    // general rule from a one-axis version that is wrong above it — both agree on
    // vectors, which is why the value that was quietly wrong in the binding got as
    // far as it did. These make the interleaving visible: two squares, a mixed rank
    // (the shorter is padded at the *front*), and a rectangle whose axes all differ.
    ["kron(2차원)", () => k2a().kron(k2b())],
    ["kron(2차원 × 1차원)", () => k2a().kron(Tensor.from([5, 6], [2]))],
    ["kron(직사각)", () => Tensor.from([1, 2, 3], [1, 3])
      .kron(Tensor.from([4, 5], [2, 1]))],
    // **The backward is what the general form had to keep.** It is `reshape` and
    // `mul`, both of which already carry one, so a wrong interleaving shows here as
    // much as in the value. Both gradients come back as one tensor.
    ["kron(2차원)의 기울기", () => {
      const a = asLeaf(k2a());
      const b = asLeaf(k2b());
      const out = a.kron(b);
      out.mul(Tensor.from(
        Array.from({ length: out.size }, (_, i) => i + 1), out.shape)).sum().backward();
      return Tensor.cat([
        gradOf(a, "kron/a").reshape([4]), gradOf(b, "kron/b").reshape([4])]);
    }],
    ["broadcast_to", () => Tensor.from([1, 2], [2]).broadcastTo([3, 2])],
    ["prelu", () => m2neg().prelu(Tensor.from([0.25], [1]))],
  ];
  for (const [name, fn] of named) out.set(`method2::${name}`, fn);

  // The four needing a read back are separate — that side is asynchronous.
  out.set("method2::inverse", async () => m2a().inverse());
  out.set("method2::pinverse", async () => m2a().pinverse());
  out.set("method2::qr", async () => (await m2a().qr()).r);
  out.set("method2::svd", async () => (await m2a().linalgSvd()).s);
  out.set("method2::cholesky", async () => m2sym().cholesky());
  out.set("method2::slogdet", async () => (await m2a().slogdet()).logabs);
  out.set("method2::det", async () => m2a().det());
  out.set("method2::logdet", async () => m2sym().logdet());
  out.set("method2::matrix_exp", async () => m2a().matrixExp());
  out.set("method2::matrix_power", async () => m2a().matrixPower(2));

  // **It has to equal the functional call.** A name that resolves to a different
  // computation diverges here.
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

  // Fifteen in-place unaries. **`acosh` has an answer from 1 up and `logit` inside 0..1**
  // — outside, both are NaN, and NaN cannot be called equal to anything, so asking shows
  // nothing.
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

  // With gradients switched on at a leaf, an in-place operation is refused. **The
  // exception's kind is asked about** — a name rather than a message, so comparing
  // implementations by value does not catch it.
  out.set("method2::제자리::기울기 켜진 잎은 거절", () => {
    try {
      Tensor.from([0.25, 0.5, 0.75, -0.5], [2, 2], { requiresGrad: true })
        .absolute_();
    } catch (err) {
      return err instanceof Error ? err.constructor.name : "?";
    }
    return "예외가 안 났다";
  });

  // ── In-place operations that take arguments ─────────────────────────────
  //
  // **The ones that change the shape are here.** `transpose_`, `squeeze_` and `unsqueeze_`
  // edit the frame rather than the values. The Python side asked this with a 2×2 and
  // **passed with the shape unchanged** — on a square the answer is the same even with the
  // frame left alone. So this side asks with a rectangle.
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

  // **Rank 3, where a dropped pair of axes shows.** Every case above is 2-D, and at
  // 2-D `transpose_(0, 1)` and `transpose_()` are the same answer — which is why
  // `transpose_()` taking no axes at all passed all of them.
  out.set("method2::제자리::transpose_(1, 2) 는 3차원에서", () => {
    const x = Tensor.from([...Array(24).keys()], [2, 3, 4]);
    x.transpose_(1, 2);
    return x.reshape([-1]).at([[0, 1, 2, 3, 4, 5]]).toArray()
      .then((v) => `(${x.shape.join(", ")}) [${[...v].map((n) => `${n}.0`).join(", ")}]`);
  });

  out.set("method2::제자리::squeeze_(0, 2)", () => {
    const x = Tensor.from([...Array(6).keys()], [1, 6, 1]);
    x.squeeze_(0, 2);
    return `(${x.shape.join(", ")},)`;
  });

  // The answer's spelling is Python's `str(True)` — the golden was frozen by Python, so
  // "true" does not match. Answering one question in one spelling is what makes the
  // comparison a comparison.
  out.set("method2::제자리::같은 텐서인가", async () => {
    const x = Tensor.from([0.25, 0.5, 0.75, -0.5], [2, 2]);
    return `같은 것=${x.inplaceUnary("abs") === x ? "True" : "False"}`;
  });
}

/**
 * The table that compares **gradients.**
 *
 * With the forward right and the backward wrong, the state is "training runs, the loss goes
 * down, and the values differ". It is the kind the core lived with for a long time through
 * BatchNorm, and a value comparison does not catch it.
 *
 * What is not here: the sorts (`topk`, `sort`, `median`), `einsum`, `pad_sequence`, `fmod`,
 * the dtype-needing `float()`, `double()`, `nll_loss` and `cross_entropy`, and
 * convolution.
 */
function addGrad(out: Map<string, Case>, inp: Inputs): void {
  const x1 = (g = false) => inp.get("x1", g);
  const xp = (g = false) => inp.get("xp", g);
  const x2 = (g = false) => inp.get("x2", g);

  /** Builds one leaf, folds and flows, and produces that leaf's gradient. */
  const one = (name: string, src: (g: boolean) => Tensor, fn: (x: Tensor) => Tensor) => {
    out.set(`grad::${name}`, () => {
      const x = src(true);
      fn(x).sum().backward();
      return gradOf(x, name);
    });
  };

  /**
   * Received multiplied by a different weight per position. With a plain `sum()` every
   * gradient is 1, and then `movedim` swapping axes, or `tile` overlapping its pieces
   * wrongly, still passes.
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
  // An input containing a 0. The common derivation (out/x) blows up on the division here
  // and quietly flows NaN.
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

  // Indexing — the thing torch code does most, and it joins the graph for the same reason
  // slicing does.
  one("idx[0]", x2, (x) => x.select(0, 0));
  one("idx[-1]", x2, (x) => x.select(0, (x.shape[0] ?? 1) - 1));
  one("idx[1:3]", x1, (x) => x.narrow(0, 1, 2));
  one("idx[:, 1]", x2, (x) => x.select(1, 1));
  one("idx[1, 2]", x2, (x) => x.select(0, 1).select(0, 2));
  one("idx[0:2, 1:3]", x2, (x) => x.narrow(0, 0, 2).narrow(1, 1, 2));
  one("idx[목록]", x2, (x) => x.indexSelect(0, Tensor.from([2, 0], [2])));

  // Concatenating and stacking — a DataLoader's collate stands on this.
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

  // **A repeated index has to accumulate the gradient into that row.**
  out.set("grad::embedding(중복 번호)", () => {
    const w = asLeaf(inp.get("w0").narrow(0, 0, 5)); // (5, 6)
    w.indexSelect(0, Tensor.from([0, 2, 0, 4], [4])).sum().backward();
    return gradOf(w, "embedding");
  });

  // Binary — both leaves are looked at. One alone cannot catch a severance on the
  // other.
  const pairs: [string, (a: Tensor, b: Tensor) => Tensor, () => [Tensor, Tensor]][] = [
    ["add", (a, b) => a.add(b), () => [x1(true), x1(true)]],
    ["sub", (a, b) => a.sub(b), () => [x1(true), x1(true)]],
    ["mul", (a, b) => a.mul(b), () => [x1(true), x1(true)]],
    ["div", (a, b) => a.div(b), () => [xp(true), xp(true)]],
    // **The right side has to be a leaf too.** `x1().neg()` is a derived tensor and
    // accumulates no gradient, and then the `/b` case dies as "no gradient arrived" —
    // through the case rather than the implementation.
    ["maximum", (a, b) => a.binary("maximum", b), () => [x1(true), asLeaf(x1().neg())]],
    ["minimum", (a, b) => a.binary("minimum", b), () => [x1(true), asLeaf(x1().neg())]],
    ["matmul", (a, b) => a.matmul(b), () => [x2(true), asLeaf(x2().transpose())]],
    // **The prediction and the target are given differently.** Equal, every gradient is 0
    // and an implementation with a flipped sign, or one that never divided by the count,
    // passes.
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
  // Convolution — **a place where the backward is written by hand.** The input, the weight
  // and the bias are all looked at. Looking at stride 2 alongside is deliberate: the path
  // that inserts zeros between the gradients runs only there.
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

  // **Nobody was asking about average pooling's backward.**
  //
  // An integration test caught that this library's average-pooling backward was not running
  // at all — an unused binding dropped out of the layout and invalidated the whole command
  // buffer, and WebGPU does not throw on that. The loss sat at ln 10 while ms/step kept
  // coming out. Had the table asked this, it would never have reached integration.
  //
  // It must not fold uniformly. max flows to one winning position and avg divides 1/n
  // across the window, and with the upstream all ones **the input gradients sum to the
  // same**, so swapping the two still passes.
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

  // **The mean and the variance have to be inside the graph.** Taken out, the input
  // gradient comes out wrong and nothing reaches weight at all (None). So both are looked
  // at.
  for (const which of ["x", "weight"] as const) {
    out.set(`grad::BatchNorm2d/${which}`, () => {
      const x = inp.get("img", true);
      const bn = new nn.BatchNormND(3);
      bn.forward(x).sum().backward();
      const leaf = which === "x" ? x : bn.weight!;
      return gradOf(leaf, `BatchNorm2d/${which}`);
    });
    // **The `sum()` above hides half of BatchNorm's backward.** The input gradient is one
    // term arriving directly plus two correction terms that arise because the mean and the
    // variance depend on the input. With the upstream all ones those two corrections cancel
    // exactly (expected 4.7e-10) and the case above never asks about them at all. A
    // different weight per position breaks the cancellation.
    out.set(`grad::BatchNorm2d(가중치)/${which}`, () => {
      const x = inp.get("img", true);
      const bn = new nn.BatchNormND(3);
      seeded(bn.forward(x)).backward();
      const leaf = which === "x" ? x : bn.weight!;
      return gradOf(leaf, `BatchNorm2d(가중치)/${which}`);
    });
  }
  // **The gathering family is where the graph is easiest to sever.** Returning the values
  // detached sends no gradient to the gathered positions and training quietly stops.
  one("topk", x1, (x) => x.topk(3).values);
  one("sort", x1, (x) => x.sort(0).values);
  one("sort(내림차순)", x1, (x) => x.sort(0, true).values);
  // **Does it flow at the kink.** torch's relu gives a gradient of 0 when the input is
  // exactly 0. `x1` is a normal draw and contains no 0, so the golden was not looking at
  // this place, and this library was flowing 1 there.
  weighted("relu(0에서)",
    () => [Tensor.from([-1, 0, 1, 0], [4], { requiresGrad: true })], (x) => x.unary("relu"));

  weighted("median()", () => [vec(true)], (x) => x.median().values);
  weighted("median(dim)", () => [mat(true)], (x) => x.median(1).values);

  // The golden froze it in the binary form, so a leaf marker follows the name.
  // **The prediction and the target are given differently** — equal, every gradient is
  // 0.
  one("L1Loss(층)/a", x1, (x) => x.l1Loss(inp.get("x1").neg()));
  one("SmoothL1Loss(층)/a", x1, (x) => x.smoothL1Loss(inp.get("xp")));
  one("BCEWithLogitsLoss/a", x1, (x) => x.bceWithLogits(inp.get("x1")));

  // **A graph was once severed where only the dtype changed.** The core's `.float()`
  // attached requires_grad to the result without attaching a parent, so backward ran with no
  // exception and only the leaf's grad was left None. With no warning and no exception.
  one("float()", x1, (x) => x.to("float32"));
  // The golden put these two in the group that **multiplies by weights.** With a plain
  // sum() every gradient is 1 and swapped positions go uncaught — which is what the weights
  // are for.
  weighted("einsum(ij->i)", () => [mat(true)], (x) => einsum("ij->i", x));
  weighted("fmod(%)", () => [vec(true)], (x) => x.fmod(2));
  one("nll_loss", x2,
    (x) => x.logSoftmax(-1).nllLoss(Tensor.from([0, 1, 2], [3], { dtype: "int64" })));
  one("cross_entropy", x2,
    (x) => x.crossEntropy(Tensor.from([0, 1, 2], [3], { dtype: "int64" })));

  // **Not a place only this side refuses, any more.** The core handed back float32 for
  // the same call and now stops at the same gate, so all three agree and the expected
  // word is the refusal one.
  out.set("grad::double()=우리는거절", () => {
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

  // ── The folding places — **only a tie opens the question** ───────────────
  //
  // When several cells fold into one, where the gradient goes on the way back diverges.
  // With all values distinct every rule gives the same answer and the question never opens.
  // The reason is written at length in the Python side's `grad_cases`.
  //
  // **Something is missing here.** The axis-less forms of `max()` and `min()` do not exist
  // in borch.ts — only `max(dim)`, which hands over an index, so its rule is the opposite.
  // The axis-less forms are built by the binding, so those two cases are answered by the
  // core and the binding alone.
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

  // The ones below lived in the core's `tests/test_fold_grad.py` alone for a while —
  // borch.ts could not answer them then, so they could not be raised to the place where all
  // three are asked together. All three answer now.
  const leaf = (v: number[]) => Tensor.from(v, [v.length], { requiresGrad: true });
  const even = () => leaf([1, 5, 5, 5]);
  const dup = () => leaf([1, 1, 2, 2, 2]);
  const nanTie = () => leaf([1, NaN, 5, 5, 5]);
  const back = (x: Tensor, got: Tensor, tag: string) => {
    got.sum().backward();
    return gradOf(x, tag);
  };

  // **No axis means no index either.** `median()` gives one tensor and `median(0)` gives a
  // value/index pair — the opposing rules are visible in the signatures themselves.
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
  // **The four rules the default hides.** The four above pass no `interpolation`,
  // so they exercise `linear` alone — and the other four are different answers
  // *and different gradients*: the split follows the rule that produced the value,
  // or the forward stops interpolating while the backward goes on doing it.
  for (const how of ["lower", "higher", "midpoint", "nearest"]) {
    out.set(`grad::접힘::quantile(0.3, ${how})`, async () => {
      const x = tied();
      return back(x, await x.quantile(0.3, null, false, how),
        `quantile(0.3, ${how})`);
    });
  }
  // The derivative is `i1`. This **was flowing 0**, and its comment cited the core's hole
  // as the justification — a gradient whose value is 0 and no gradient say different things,
  // and in the copying the second became the first.
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
 * Whether **the same kind of exception** arises under the same condition, with a searchable
 * message.
 *
 * The answer's shape is `kind|fragment=boolean`. The kind name is frozen too, so imitating
 * torch's `RuntimeError` is what matches — the reasons are in `src/errors.ts`.
 *
 * **The five missing here are missing because the features are.** Integer dtypes,
 * `nn.Linear`, `conv2d`, indexing and in-place operations do not exist yet. Registering
 * what does not exist has the runner count it as "failed", and that is absent rather than
 * wrong.
 */
function addError(out: Map<string, Case>): void {
  // **Asynchronous bodies are accepted too.** `item()` reads back from the GPU and so is
  // async, and a throw inside it is not caught by a synchronous try — left as it is, the
  // answer becomes "no exception arose".
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

// ── The inputs `flow_cases` uses in tests/cases.py. ───────────────────────
const F_VEC = [0.5, 2.0, 1.5, 3.0];
const F_MAT = [1, 2, 3, 4, 5, 6, 7, 8, 9]; // (3, 3), 1 부터
const F_PAIR = [1, 2, 3, 4, 5, 6]; // (2, 3)
const F_SYM = [4, 1, 1, 3]; // (2, 2)
const F_MASK = [1, 0, 1, 0];

/**
 * The table that asks only whether the gradient **flows.**
 *
 * A check comparing values alone cannot see a severed graph — because the values are right.
 * The sister library's `roll` and `masked_select` were quietly severed that way while all
 * 746 golden cases were green.
 *
 * **Two things are answered together.** Asking about `requires_grad` alone is not enough —
 * `.float()` once said true and left `.grad` empty, and with that check alone it would have
 * passed.
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
   * What is not here is **what does not exist yet.** `median` and `msort` need a GPU sort,
   * `masked_select` needs a CPU round trip because its output size depends on the values,
   * and `einsum`, `det`, `logdet`, `inverse` and `cholesky` need linear algebra. Registering
   * them and letting them throw has it counted as "failed", and that is absent rather than
   * wrong.
   */
  // The ones whose name is itself a unary operation.
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

  // The ones that go to the CPU and back. The question is the same and the result simply
  // has to be awaited.
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

/** The same question as `asks`, awaiting the result. */
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
 * Did it flow, and did a gradient actually reach the leaf.
 *
 * The answer's spelling is the golden's verbatim — the Python case built this string.
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

// ── The inputs `reduce_cases` uses in tests/cases.py. Carried across verbatim. ──
// **A tie is in there on purpose.** amax divides the gradient evenly on a tie
// ([1,3,3,2] → [0,.5,.5,0]), and measured with tie-free input that rule is never seen.
const tie = [1.0, 3.0, 3.0, 2.0];
const mat = [1.0, 5.0, 3.0, 4.0, 2.0, 6.0]; // (2, 3)
const withnan = [1.0, Number.NaN, 3.0, 5.0];

/**
 * `reduce_cases`'s gradient cases multiply the output by **a different weight per
 * position**, sum, and backpropagate. A scalar output has no weights — there is no position
 * to multiply.
 */
function seeded(out: Tensor): Tensor {
  if (out.shape.length === 0) return out.sum();
  const w = Array.from({ length: out.size }, (_, i) => i);
  return out.mul(Tensor.from(w, out.shape)).sum();
}

function addReduce(out: Map<string, Case>): void {
  /** The value case and the gradient case are attached together — separated, only one of
   *  them ends up being asked. */
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

  // **The ones taking an axis are asked by value.** Asked by gradient alone, ignoring the
  // axis entirely still passes — the gradients of `sum(dim=1).sum()` and `sum().sum()` are
  // both all ones, so the answers coincide. The Python binding really was discarding the
  // axis through that hole.
  add("sum(dim)", (x) => x.sumDim(1), mat, [2, 3]);
  add("sum(dim0)", (x) => x.sumDim(0), mat, [2, 3]);
  add("sum(dim,keepdim)", (x) => x.sumDim(1, true), mat, [2, 3]);
  add("norm(dim)", (x) => x.square().sumDim(1).sqrt(), mat, [2, 3]);
  add("norm(p=1,dim)", (x) => x.abs().sumDim(0), mat, [2, 3]);

  // The ones with no gradient case. The golden froze values alone for them too.
  out.set("reduce::aminmax/최소", () => Tensor.from(tie).amin());
  out.set("reduce::aminmax/최대", () => Tensor.from(tie).amax());
  // **The indices are asked about separately** — by value alone, wrong indices pass.
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

// ── The inputs `shape_cases` uses in tests/cases.py. ──────────────────────
const seq = (n: number): number[] => Array.from({ length: n }, (_, i) => i);
const SQUARE = seq(9); // (3, 3)
const LINE = seq(5);
const COL = [0.0, 3.0]; // mat[:, :1] — (2, 1)

function addShape(out: Map<string, Case>): void {
  /** The (2,3) this table uses. Uploaded fresh each time so the cases share no state. */
  const m = (grad = false) => Tensor.from(seq(6), [2, 3], { requiresGrad: grad });
  const sq = (grad = false) => Tensor.from(SQUARE, [3, 3], { requiresGrad: grad });
  const line = (grad = false) => Tensor.from(LINE, [5], { requiresGrad: grad });
  const col = (grad = false) => Tensor.from(COL, [2, 1], { requiresGrad: grad });
  const pair = (grad = false) => Tensor.from([1.0, 2.0], [2], { requiresGrad: grad });
  // **Rank 3.** Asking about swapping axes at two dimensions alone cannot see anywhere
  // outside `(0,1)` — at two dimensions any pair of axes gives one answer, so an
  // implementation discarding the axis arguments passes. It is written here with `permute`.
  // borch.ts's `transpose()` is 2-D only and takes no axes — the Python side takes two axes
  // and builds this order to hand over.
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
    // **A permutation that is not its own inverse.** The four above are all involutions
    // (swapping two axes, reversing), so an implementation applying the permutation
    // backwards passes. `(1,2,0)`'s inverse is `(2,0,1)` and even the shapes differ —
    // [3,4,2] against [4,2,3].
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

  // **expand and unfold diverge here.** expand folds the expanded axis back together and
  // unfold accumulates by the window overlap — unfolding a length of 5 by 3 and 1 gives
  // [1,2,3,2,1].
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
    // **It is more so in the backward** — reversing needs the inverse permutation, and
    // when a permutation is its own inverse, using the same array both ways still gives the
    // right answer.
    ["permute(비가역)", () => cube(true).permute([1, 2, 0]), cube],
  ];
  for (const [name, build] of grads) {
    out.set(`shape::grad::${name}`, () => {
      // Recovering the leaf means building it inside the case — built outside, the cases
      // share one tensor and the gradients accumulate.
      const leaves: Tensor[] = [];
      const res = withLeafCapture(build, leaves);
      seeded(res).backward();
      const leaf = leaves[0];
      if (!leaf) throw new Error(`${name}: 잎을 못 잡았다`);
      return gradOf(leaf, name);
    });
  }
}

// ── The inputs `method_cases` uses in tests/cases.py. ─────────────────────
const M_POS = [0.5, 2.0, 1.5, 3.0];
const M_VEC = [0.5, 2.0, -1.5, 3.0];
const M_OTHER = [1.0, 2.0, -3.0, 0.5];
const M_MAT = [1, 2, 3, 4, 5, 6, 7, 8, 9]; // (3, 3), 1 부터
const M_MASK = [1, 0, 1, 0]; // bool 을 0/1 로

/**
 * The ones that have to be callable as `x.f(...)`.
 *
 * **What is not written yet is not registered.** `sort`, `argsort`, `topk`, `median` and
 * `unique` need a GPU sort and that has not been stood up. Listing the name and letting it
 * throw has the runner count it as "failed", and that is absent rather than wrong — the
 * runner counts the unasked separately, so what is missing here shows up in that count.
 */
function addMethod(out: Map<string, Case>): void {
  const vec = (grad = false) => Tensor.from(M_VEC, [4], { requiresGrad: grad });
  const other = () => Tensor.from(M_OTHER, [4]);
  const mat = () => Tensor.from(M_MAT, [3, 3]);

  // A unary in the table needs only its name written.
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

  // The ones needing a partner. A comparison gives 0/1 and matches the golden's bool as it
  // stands.
  //
  // **All eight go through the public method, and until today none of them did.**
  // They called `vec().binary("gt", other())` — the internal helper — so the kernel
  // was measured, the gradient a tie splits was measured, and the eight names a user
  // types were measured by nothing. Two of them (`maximum`, `minimum`) did not exist
  // at all while their cases were green, and the other six existed only under
  // torch's long spellings: `greater` was here and `gt` was not.
  //
  // A case that calls the internal helper is green about the internal helper. The
  // core-to-borch.ts name axis is what found it; no value comparison can, because
  // both sides of the comparison were the same helper.
  const paired: [string, (a: Tensor, b: Tensor) => Tensor][] = [
    ["eq", (a, b) => a.eq(b)], ["ne", (a, b) => a.ne(b)],
    ["lt", (a, b) => a.lt(b)], ["le", (a, b) => a.le(b)],
    ["gt", (a, b) => a.gt(b)], ["ge", (a, b) => a.ge(b)],
    ["maximum", (a, b) => a.maximum(b)], ["minimum", (a, b) => a.minimum(b)],
  ];
  for (const [name, call] of paired) {
    out.set(`method::${name}`, () => call(vec(), other()));
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
    // **The one place whose argument order is reversed from the function's** — the method
    // is `x.where(condition, other)`.
    ["where", () => vec().where(Tensor.from(M_MASK, [4]), other())],
    // **The three broadcast against each other.** Every `where` above hands all
    // three the same shape, which is the one arrangement that cannot tell a
    // broadcasting `where` from one that walks three buffers at the same offset —
    // and the second is what this was.
    ["where(마스크가 좁다)", () => {
      const wide = Tensor.from(
        Array.from({ length: 24 }, (_, i) => i), [2, 3, 4]);
      const thin = Tensor.from(
        Array.from({ length: 8 }, (_, i) => (i % 3 > 0 ? 1 : 0)), [2, 1, 4]);
      return wide.where(thin, Tensor.zeros([2, 3, 4]));
    }],
    ["where(가지가 좁다)", () => {
      const wide = Tensor.from(
        Array.from({ length: 24 }, (_, i) => i), [2, 3, 4]);
      const narrow = Tensor.from([0, 1, 2, 3], [4]);
      return narrow.where(wide.binary("gt", Tensor.full([], 5)), wide);
    }],
    ["gather", () => mat().gather(1, Tensor.from([0, 2, 1, 0, 2, 1], [3, 2]))],
    ["argsort", () => vec().argsort(0)],
    ["sort", () => vec().sort(0).values],
    ["topk", () => vec().topk(2).values],
    ["median", () => vec().median().values],
  ];
  for (const [name, fn] of single) out.set(`method::${name}`, fn);

  // The ones returning several things — each piece gets a name. Looking at one leaves the
  // rest uncaught.
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

  // **movedim is asked with all four combinations.** With (0,0) alone it is the identity
  // and equal to asking nothing, and behind that the sister library's movedim(0,-1) was
  // quietly behaving as the identity.
  for (const [s, d] of [[0, -1], [-1, 0], [0, 1], [1, 0]] as const) {
    out.set(`method::movedim(${s},${d})`, () => mat().movedim(s, d));
  }
  out.set("method::unique", async () => vec().unique());

  // The ones that froze **a verdict** rather than a value. The library has to be asked —
  // answering here by comparing JS arrays passes and tests nothing.
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
 * Captures the leaf that requires grad.
 *
 * A case body builds its leaf in place, as in `col(true).expand(...)`, so seeing `x.grad`
 * later means recovering that leaf. Walking the graph back from the result to find it beats
 * splitting the body in two — a body reads only while it pairs with the golden's name.
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
 * The eight top-level recurrent names — `torch.lstm` and its siblings.
 *
 * **They were not here for a while.** The binding calls borch.ts directly so the values
 * were compared, and nobody asked about borch.ts's **direct surface** (the order of the
 * weight list, how many things come back). They could not be carried across because the
 * weights were drawn inside `cases.py` and did not ride in `golden.json`, and moving them
 * into `golden_inputs()` opened it.
 *
 * What is asked is the **wiring** rather than the values, so each piece gets a name —
 * `lstm` unfolds three (`output, h_n, c_n`) and the rest two. Looking at one leaves the
 * rest uncaught.
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

  // One mismatched name and that case **quietly does not run** — invisible except through
  // the runner's "absent from the golden" line. The names here carry a Korean tail and a
  // space, so they mismatch more easily.
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

  // The four cells. They are **one step**, so they take four tensors individually rather
  // than a list — the argument shape differs from the list-taking ones above, and that
  // difference shows here alone.
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

  // A dropout of 0 **discards nothing even during training.** A non-zero value does not
  // exist in our layer and is refused, and blocking 0 as well would close the normal
  // path.
  out.set("rnntop::dropout=0 이면 돈다", () =>
    rnn.lstm(x(), [h1(), c1()], w("drop_w", 4),
      { dropout: 0, train: true })[0]);

  // ── bidirectional, stacked and projected, refused here until today ──────────
  //
  // Carried verbatim from `tests/cases.py`: the input is `0.1·i − 1.0` over (5, 2, 3)
  // and every weight is `0.1·((i mod 7) − 3)` down its own shape, so nothing crosses
  // the boundary. **The flat list is `namedParameters()`'s order** — four per
  // direction, the reverse one second, layers outermost, `weight_hr` last within a
  // direction.
  const rtGates: Record<string, number> = {
    lstm: 4, gru: 3, rnn_tanh: 1, rnn_relu: 1,
  };
  const rtRamp = (shape: number[]): Tensor => {
    const n = shape.reduce((a, b) => a * b, 1);
    const flat: number[] = [];
    for (let i = 0; i < n; i++) flat.push(0.1 * ((i % 7) - 3));
    return Tensor.from(flat, shape);
  };
  const rtX = (): Tensor => {
    const v: number[] = [];
    for (let i = 0; i < 30; i++) v.push(i * 0.1 - 1.0);
    return Tensor.from(v, [5, 2, 3]);
  };
  const topFlag = (
    name: string, part: number,
    { layers = 1, bidirectional = false, dropout = 0, proj = 0 } = {},
  ): Tensor => {
    const gates = rtGates[name]!, hidden = 4, width = 3;
    const dirs = bidirectional ? 2 : 1;
    const real = proj || hidden;
    const flat: Tensor[] = [];
    for (let layer = 0; layer < layers; layer++) {
      const inSize = layer === 0 ? width : real * dirs;
      for (let d = 0; d < dirs; d++) {
        flat.push(rtRamp([gates * hidden, inSize]));
        flat.push(rtRamp([gates * hidden, real]));
        flat.push(rtRamp([gates * hidden]));
        flat.push(rtRamp([gates * hidden]));
        if (proj) flat.push(rtRamp([proj, hidden]));
      }
    }
    const rows = layers * dirs;
    const opts = { numLayers: layers, dropout, bidirectional };
    const h0 = Tensor.zeros([rows, 2, real]);
    const got = name === "lstm"
      ? rnn.lstm(rtX(), [h0, Tensor.zeros([rows, 2, hidden])], flat, opts)
      : name === "gru" ? rnn.gru(rtX(), h0, flat, opts)
        : name === "rnn_tanh" ? rnn.rnnTanh(rtX(), h0, flat, opts)
          : rnn.rnnRelu(rtX(), h0, flat, opts);
    return got[part]!;
  };
  for (const name of ["lstm", "gru", "rnn_tanh", "rnn_relu"]) {
    for (let p = 0; p < (name === "lstm" ? 3 : 2); p++) {
      out.set(`rnntop::${name}(양방향)[${p}]`,
        () => topFlag(name, p, { bidirectional: true }));
      out.set(`rnntop::${name}(2층양방향)[${p}]`,
        () => topFlag(name, p, { layers: 2, bidirectional: true }));
    }
    // `train` is false, so the dropout is the identity — which is the claim.
    out.set(`rnntop::${name}(2층 dropout, 평가)`,
      () => topFlag(name, 0, { layers: 2, dropout: 0.5 }));
  }
  for (let p = 0; p < 3; p++) {
    out.set(`rnntop::lstm(proj_size)[${p}]`,
      () => topFlag("lstm", p, { proj: 2 }));
    out.set(`rnntop::lstm(proj_size 와 양방향)[${p}]`,
      () => topFlag("lstm", p, { layers: 2, proj: 2, bidirectional: true }));
  }
}
