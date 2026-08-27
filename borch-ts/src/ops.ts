/**
 * `torchvision.ops` — **box geometry, and only that.**
 *
 * Eleven of torchvision's thirty-nine. The other twenty-eight want a detector's
 * feature maps or its predictions, and there is no detector here; these eleven want
 * nothing but four numbers a box. That makes every one of them **deterministic**,
 * which is unusual for the vision side — there is no distribution half to this file.
 *
 * ## Why the arithmetic happens on the CPU
 *
 * Boxes come in tens, not millions, and every function here either sorts, iterates
 * until a set stops shrinking, or reads a mask's extent — shapes a kernel is bad at
 * and a loop is good at. So each entry reads its tensors back once and returns a new
 * one, the same shape `linalg` takes for `det` and `qr`. Being `async` is the price of
 * a readback in WebGPU and it is charged once per call rather than per box.
 *
 * ## Two conventions that are easy to get backwards
 *
 * **`(N, 4)` against `(M, 4)` gives `N x M`**, not a paired list of length `N`. Every
 * box is measured against every box, because that is what a detector needs. An
 * implementation that pairs them off returns `min(N, M)` numbers that all look
 * plausible.
 *
 * **A size is `(height, width)`** while a box is `(x, y, x, y)`. The two orders are
 * opposite and torchvision keeps both, so `clipBoxesToImage` is the one place where
 * reading the tuple in box order silently clips against the wrong edge.
 */

import { Tensor } from "./tensor.js";
// **The warp kernels borrow three helpers from `vision.ts`** rather than carrying a
// second copy of the affine formula. They are exported there for this and nothing else.
import * as vision from "./vision.js";
import {
  affineOutputSize,
  inverseAffineMatrix,
  perspectiveCoefficients,
  shearPair,
} from "./vision.js";

/** The three spellings of a box. */
export type BoxFormat = "xyxy" | "xywh" | "cxcywh";

const FORMATS: readonly BoxFormat[] = ["xyxy", "xywh", "cxcywh"];

function checkFormat(fmt: string, where: string): asserts fmt is BoxFormat {
  if (!FORMATS.includes(fmt as BoxFormat)) {
    throw new Error(
      `Unsupported Bounding Box format ${fmt} — it is one of ${FORMATS.join(", ")}. ` +
        `(${where})`,
    );
  }
}

/**
 * Boxes as an array of rows, from a tensor of shape `(N, 4)`.
 *
 * **The row count comes from the length and not from the shape.** A caller who hands
 * over a `(4,)` tensor means one box, and torchvision's own functions accept that;
 * dividing by four is what makes both spellings arrive here the same.
 */
async function rows(boxes: Tensor): Promise<number[][]> {
  const flat = await boxes.toArray();
  if (flat.length % 4 !== 0) {
    throw new Error(
      `boxes has ${flat.length} numbers, which is not a whole number of (x1, y1, x2, y2) rows.`,
    );
  }
  const out: number[][] = [];
  for (let i = 0; i < flat.length; i += 4) {
    out.push([flat[i] ?? 0, flat[i + 1] ?? 0, flat[i + 2] ?? 0, flat[i + 3] ?? 0]);
  }
  return out;
}

/** One row, read in `fmt`, written as `xyxy`. */
function toXyxy(box: number[], fmt: BoxFormat): number[] {
  const [a = 0, b = 0, c = 0, d = 0] = box;
  if (fmt === "xyxy") return [a, b, c, d];
  if (fmt === "xywh") return [a, b, a + c, b + d];
  return [a - 0.5 * c, b - 0.5 * d, a + 0.5 * c, b + 0.5 * d];
}

/**
 * Between the three spellings.
 *
 * **The identity is a copy and not the same tensor** — torchvision returns a new one
 * even when the formats match, and a caller who writes into the result must not reach
 * the caller's boxes.
 */
export async function boxConvert(
  boxes: Tensor,
  inFmt: string,
  outFmt: string,
): Promise<Tensor> {
  checkFormat(inFmt, "in_fmt");
  checkFormat(outFmt, "out_fmt");
  const src = await rows(boxes);
  const out: number[] = [];
  for (const box of src) {
    const [x1 = 0, y1 = 0, x2 = 0, y2 = 0] = toXyxy(box, inFmt);
    if (outFmt === "xyxy") {
      out.push(x1, y1, x2, y2);
    } else {
      const w = x2 - x1;
      const h = y2 - y1;
      if (outFmt === "xywh") out.push(x1, y1, w, h);
      else out.push(x1 + 0.5 * w, y1 + 0.5 * h, w, h);
    }
  }
  return Tensor.from(out, [src.length, 4]);
}

/**
 * Width times height.
 *
 * **A box with `x2 < x1` gets a negative area** rather than zero. torchvision does not
 * clamp here, and clamping would hide a box built the wrong way round — which is the
 * one thing an area is good for catching.
 */
export async function boxArea(boxes: Tensor, fmt: string = "xyxy"): Promise<Tensor> {
  checkFormat(fmt, "format");
  const src = await rows(boxes);
  const out = src.map((box) => {
    const [x1 = 0, y1 = 0, x2 = 0, y2 = 0] = toXyxy(box, fmt);
    return (x2 - x1) * (y2 - y1);
  });
  return Tensor.from(out, [src.length]);
}

/** Intersection and union of every box in `a` against every box in `b`. */
function interUnion(a: number[][], b: number[][]): { inter: number[][]; union: number[][] } {
  const areaOf = (r: number[]) =>
    ((r[2] ?? 0) - (r[0] ?? 0)) * ((r[3] ?? 0) - (r[1] ?? 0));
  const inter: number[][] = [];
  const union: number[][] = [];
  for (const ra of a) {
    const rowI: number[] = [];
    const rowU: number[] = [];
    for (const rb of b) {
      const w = Math.max(Math.min(ra[2] ?? 0, rb[2] ?? 0) - Math.max(ra[0] ?? 0, rb[0] ?? 0), 0);
      const h = Math.max(Math.min(ra[3] ?? 0, rb[3] ?? 0) - Math.max(ra[1] ?? 0, rb[1] ?? 0), 0);
      const i = w * h;
      rowI.push(i);
      rowU.push(areaOf(ra) + areaOf(rb) - i);
    }
    inter.push(rowI);
    union.push(rowU);
  }
  return { inter, union };
}

/** The smallest box enclosing both, as its area and its squared diagonal. */
function enclosing(ra: number[], rb: number[]): { area: number; diagonal: number } {
  const w = Math.max(Math.max(ra[2] ?? 0, rb[2] ?? 0) - Math.min(ra[0] ?? 0, rb[0] ?? 0), 0);
  const h = Math.max(Math.max(ra[3] ?? 0, rb[3] ?? 0) - Math.min(ra[1] ?? 0, rb[1] ?? 0), 0);
  return { area: w * h, diagonal: w * w + h * h };
}

/** Squared distance between the two centres. */
function centreGap(ra: number[], rb: number[]): number {
  const dx = ((ra[0] ?? 0) + (ra[2] ?? 0)) / 2 - ((rb[0] ?? 0) + (rb[2] ?? 0)) / 2;
  const dy = ((ra[1] ?? 0) + (ra[3] ?? 0)) / 2 - ((rb[1] ?? 0) + (rb[3] ?? 0)) / 2;
  return dx * dx + dy * dy;
}

/** Fills an `N x M` table from a function of the two rows, as a flat tensor. */
function table(
  a: number[][],
  b: number[][],
  cell: (ra: number[], rb: number[], i: number, j: number) => number,
): Tensor {
  const out: number[] = [];
  for (let i = 0; i < a.length; i++) {
    for (let j = 0; j < b.length; j++) out.push(cell(a[i]!, b[j]!, i, j));
  }
  return Tensor.from(out, [a.length, b.length]);
}

/**
 * **An `N x M` matrix, not a paired list.** Every box against every box, which is what
 * a detector needs and what surprises everyone the first time.
 */
export async function boxIou(
  boxes1: Tensor,
  boxes2: Tensor,
  fmt: string = "xyxy",
): Promise<Tensor> {
  checkFormat(fmt, "format");
  const a = (await rows(boxes1)).map((r) => toXyxy(r, fmt));
  const b = (await rows(boxes2)).map((r) => toXyxy(r, fmt));
  const { inter, union } = interUnion(a, b);
  return table(a, b, (_ra, _rb, i, j) => (inter[i]![j] ?? 0) / (union[i]![j] ?? 0));
}

/**
 * IoU, **minus what the smallest enclosing box wastes.**
 *
 * Two boxes that do not touch have an IoU of 0 however far apart they are; this one
 * keeps falling, towards -1. That is the whole reason a loss can be built on it and
 * not on plain IoU — a gradient that is flat everywhere it matters teaches nothing.
 */
export async function generalizedBoxIou(boxes1: Tensor, boxes2: Tensor): Promise<Tensor> {
  const a = await rows(boxes1);
  const b = await rows(boxes2);
  const { inter, union } = interUnion(a, b);
  return table(a, b, (ra, rb, i, j) => {
    const u = union[i]![j] ?? 0;
    const { area } = enclosing(ra, rb);
    return (inter[i]![j] ?? 0) / u - (area - u) / area;
  });
}

/**
 * IoU penalised by **how far apart the centres are**, as a fraction of the enclosing
 * box's squared diagonal.
 */
export async function distanceBoxIou(
  boxes1: Tensor,
  boxes2: Tensor,
  eps = 1e-7,
): Promise<Tensor> {
  const a = await rows(boxes1);
  const b = await rows(boxes2);
  const { inter, union } = interUnion(a, b);
  return table(a, b, (ra, rb, i, j) => {
    const { diagonal } = enclosing(ra, rb);
    return (inter[i]![j] ?? 0) / (union[i]![j] ?? 0) - centreGap(ra, rb) / (diagonal + eps);
  });
}

/**
 * `distanceBoxIou` and **one more term for the aspect ratio** — two boxes sharing a
 * centre and an area but not a shape score lower here, and identically under the
 * distance one.
 */
export async function completeBoxIou(
  boxes1: Tensor,
  boxes2: Tensor,
  eps = 1e-7,
): Promise<Tensor> {
  const a = await rows(boxes1);
  const b = await rows(boxes2);
  const { inter, union } = interUnion(a, b);
  const FOUR_OVER_PI_SQ = 4 / (Math.PI * Math.PI);
  return table(a, b, (ra, rb, i, j) => {
    const iou = (inter[i]![j] ?? 0) / (union[i]![j] ?? 0);
    const { diagonal } = enclosing(ra, rb);
    const diou = iou - centreGap(ra, rb) / (diagonal + eps);
    const ratio = (r: number[]) =>
      Math.atan(((r[2] ?? 0) - (r[0] ?? 0)) / ((r[3] ?? 0) - (r[1] ?? 0)));
    const v = FOUR_OVER_PI_SQ * (ratio(rb) - ratio(ra)) ** 2;
    return diou - (v / (1 - iou + v + eps)) * v;
  });
}

export type Reduction = "none" | "mean" | "sum";

/**
 * `none`, `mean` or `sum`, as every loss in torch takes.
 *
 * **An unknown name is refused rather than quietly meaning `none`.** A typo that means
 * "no reduction" hands back a vector where a scalar was wanted, and the shape error then
 * surfaces somewhere else entirely, with nothing pointing here.
 */
function reduce(values: number[], reduction: string): Tensor {
  if (reduction === "none") return Tensor.from(values, [values.length]);
  const total = values.reduce((s, v) => s + v, 0);
  if (reduction === "sum") return Tensor.from([total], []);
  if (reduction === "mean") {
    return Tensor.from([values.length ? total / values.length : 0], []);
  }
  throw new Error(
    `${reduction} is not a valid value for reduction — it is one of none, mean, sum.`,
  );
}

/**
 * The two box sets, **matched one to one.**
 *
 * The IoU functions above answer *every box against every box*, because that is what a
 * detector's assignment step needs. **A loss is the other question**: these arrive
 * already paired, one prediction against its own target, and the answer is one number
 * per pair rather than a matrix.
 *
 * Taking the diagonal of the matrix gives the same numbers and computes `N²` of them to
 * keep `N`, which on a real batch is the entire cost of the loss.
 */
async function paired(
  boxes1: Tensor,
  boxes2: Tensor,
): Promise<{ a: number[][]; b: number[][] }> {
  const a = await rows(boxes1);
  const b = await rows(boxes2);
  if (a.length !== b.length) {
    throw new Error(
      `the two box sets must be the same length to be paired — got ${a.length} and ${b.length}.`,
    );
  }
  return { a, b };
}

/** Intersection and union of **one** pair — the `N` of them, not the `N x M`. */
function pairInterUnion(ra: number[], rb: number[]): { inter: number; union: number } {
  const areaOf = (r: number[]) =>
    ((r[2] ?? 0) - (r[0] ?? 0)) * ((r[3] ?? 0) - (r[1] ?? 0));
  const w = Math.max(Math.min(ra[2] ?? 0, rb[2] ?? 0) - Math.max(ra[0] ?? 0, rb[0] ?? 0), 0);
  const h = Math.max(Math.min(ra[3] ?? 0, rb[3] ?? 0) - Math.max(ra[1] ?? 0, rb[1] ?? 0), 0);
  const inter = w * h;
  return { inter, union: areaOf(ra) + areaOf(rb) - inter };
}

/** `1 - giou`, pair by pair. */
export async function generalizedBoxIouLoss(
  boxes1: Tensor,
  boxes2: Tensor,
  reduction: string = "none",
  eps = 1e-7,
): Promise<Tensor> {
  const { a, b } = await paired(boxes1, boxes2);
  return reduce(
    a.map((ra, i) => {
      const { inter, union } = pairInterUnion(ra, b[i]!);
      const iou = inter / (union + eps);
      const { area } = enclosing(ra, b[i]!);
      return 1 - (iou - (area - union) / (area + eps));
    }),
    reduction,
  );
}

/** `1 - diou` — IoU penalised by how far apart the centres are. */
export async function distanceBoxIouLoss(
  boxes1: Tensor,
  boxes2: Tensor,
  reduction: string = "none",
  eps = 1e-7,
): Promise<Tensor> {
  const { a, b } = await paired(boxes1, boxes2);
  return reduce(
    a.map((ra, i) => {
      const { inter, union } = pairInterUnion(ra, b[i]!);
      const { diagonal } = enclosing(ra, b[i]!);
      return 1 - (inter / (union + eps) - centreGap(ra, b[i]!) / (diagonal + eps));
    }),
    reduction,
  );
}

/**
 * `1 - ciou` — the distance term and **one more for the aspect ratio.**
 *
 * Two boxes sharing a centre and an area but not a shape score the same under the
 * distance loss and differently here. The golden case was chosen so the two disagree:
 * with matched aspect ratios this extra term is exactly zero, and a case like that
 * passes while asking nothing about the one thing this function adds.
 */
export async function completeBoxIouLoss(
  boxes1: Tensor,
  boxes2: Tensor,
  reduction: string = "none",
  eps = 1e-7,
): Promise<Tensor> {
  const { a, b } = await paired(boxes1, boxes2);
  const FOUR_OVER_PI_SQ = 4 / (Math.PI * Math.PI);
  return reduce(
    a.map((ra, i) => {
      const rb = b[i]!;
      const { inter, union } = pairInterUnion(ra, rb);
      const iou = inter / (union + eps);
      const { diagonal } = enclosing(ra, rb);
      const ratio = (r: number[]) =>
        Math.atan(((r[2] ?? 0) - (r[0] ?? 0)) / ((r[3] ?? 0) - (r[1] ?? 0) + eps));
      const v = FOUR_OVER_PI_SQ * (ratio(rb) - ratio(ra)) ** 2;
      const alpha = v / (1 - iou + v + eps);
      return 1 - (iou - (centreGap(ra, rb) / (diagonal + eps) + alpha * v));
    }),
    reduction,
  );
}

/**
 * Cross-entropy with **the easy examples turned down.**
 *
 * A detector looks at tens of thousands of boxes and almost all of them are plainly
 * background. Summed with equal weight that majority drowns out the few that are hard,
 * so each term is scaled by `(1 - p_t) ** gamma` — near zero once the model is already
 * confident and right.
 *
 * `alpha` is the separate, older fix for the same imbalance: one weight for the positive
 * class and `1 - alpha` for the negative. **`alpha = -1` turns it off**, which is
 * torchvision's own switch rather than a value.
 *
 * The inputs are **logits**, not probabilities. Something already through a sigmoid
 * gives a number rather than an error, which is why it is said here.
 *
 * Both halves are written around `Math.exp` overflowing: the plain sigmoid and a plain
 * `log(1 + exp(-x))` both stop being finite around ±710, where the loss itself is not.
 */
export async function sigmoidFocalLoss(
  inputs: Tensor,
  targets: Tensor,
  alpha = 0.25,
  gamma = 2,
  reduction: string = "none",
): Promise<Tensor> {
  const x = Array.from(await inputs.toArray());
  const y = Array.from(await targets.toArray());
  if (x.length !== y.length) {
    throw new Error(
      `inputs and targets must hold the same number of values — got ${x.length} and ${y.length}.`,
    );
  }
  const out = x.map((v, i) => {
    const t = y[i] ?? 0;
    const z = Math.exp(-Math.abs(v));
    const p = v >= 0 ? 1 / (1 + z) : z / (1 + z);
    const ce = Math.max(v, 0) - v * t + Math.log1p(z);
    const pt = p * t + (1 - p) * (1 - t);
    const loss = ce * (1 - pt) ** gamma;
    return alpha >= 0 ? (alpha * t + (1 - alpha) * (1 - t)) * loss : loss;
  });
  if (reduction === "none") {
    return Tensor.from(out, inputs.shape as number[]);
  }
  return reduce(out, reduction);
}

/**
 * Push every corner back inside a picture of `size`.
 *
 * **`size` is `(height, width)`** — the opposite order to a box's own `(x, y)`, and
 * torchvision's own convention. Reading it in box order clips x against the height and
 * gives boxes that are wrong without being obviously wrong.
 */
export async function clipBoxesToImage(
  boxes: Tensor,
  size: readonly [number, number],
): Promise<Tensor> {
  const [height, width] = size;
  const src = await rows(boxes);
  const out: number[] = [];
  for (const r of src) {
    out.push(
      Math.min(Math.max(r[0] ?? 0, 0), width),
      Math.min(Math.max(r[1] ?? 0, 0), height),
      Math.min(Math.max(r[2] ?? 0, 0), width),
      Math.min(Math.max(r[3] ?? 0, 0), height),
    );
  }
  return Tensor.from(out, [src.length, 4]);
}

/**
 * **Indices, not boxes.**
 *
 * Every filter in this file returns positions rather than survivors, because the
 * caller almost always holds scores and labels that have to be cut at the same places.
 */
export async function removeSmallBoxes(boxes: Tensor, minSize: number): Promise<Tensor> {
  const src = await rows(boxes);
  const keep: number[] = [];
  src.forEach((r, i) => {
    if ((r[2] ?? 0) - (r[0] ?? 0) >= minSize && (r[3] ?? 0) - (r[1] ?? 0) >= minSize) {
      keep.push(i);
    }
  });
  return Tensor.from(keep, [keep.length]);
}

/**
 * The tightest box around each mask, from a `(N, H, W)` stack.
 *
 * **An empty plane gives all zeros** rather than an error. That is torchvision's
 * behaviour and it is the one that lets a batch with one blank mask still stack —
 * raising here would make a single empty annotation kill a whole batch.
 */
export async function masksToBoxes(masks: Tensor): Promise<Tensor> {
  const [n = 0, h = 0, w = 0] = masks.shape;
  const flat = await masks.toArray();
  const out: number[] = [];
  for (let k = 0; k < n; k++) {
    let x1 = Infinity;
    let y1 = Infinity;
    let x2 = -Infinity;
    let y2 = -Infinity;
    for (let i = 0; i < h; i++) {
      for (let j = 0; j < w; j++) {
        if ((flat[(k * h + i) * w + j] ?? 0) !== 0) {
          if (j < x1) x1 = j;
          if (i < y1) y1 = i;
          if (j > x2) x2 = j;
          if (i > y2) y2 = i;
        }
      }
    }
    if (x1 === Infinity) out.push(0, 0, 0, 0);
    else out.push(x1, y1, x2, y2);
  }
  return Tensor.from(out, [n, 4]);
}

/**
 * Non-maximum suppression: keep the best-scoring box, discard everything overlapping
 * it too much, repeat.
 *
 * **`> iouThreshold` and not `>=`.** At a threshold of 0, two boxes that merely touch —
 * zero overlap — both survive. That is the boundary anybody testing this reaches for
 * first, and the two spellings differ there and nowhere else.
 *
 * Ties in the score are **not decided here, and torchvision does not decide them
 * either** — its own documentation says the winner is not guaranteed to match between
 * CPU and GPU. So a case built on tied scores is a case with no answer, and the sort
 * below is stable only so that repeated runs here agree with each other.
 */
export async function nms(
  boxes: Tensor,
  scores: Tensor,
  iouThreshold: number,
): Promise<Tensor> {
  const src = await rows(boxes);
  const values = await scores.toArray();
  return Tensor.from(nmsOn(src, Array.from(values), iouThreshold));
}

/** The loop itself, on rows already read back — `batchedNms` reuses it after shifting. */
function nmsOn(src: number[][], values: number[], iouThreshold: number): number[] {
  let order = src
    .map((_, i) => i)
    .sort((p, q) => (values[q] ?? 0) - (values[p] ?? 0) || p - q);
  const kept: number[] = [];
  while (order.length) {
    const best = order[0]!;
    kept.push(best);
    const rest = order.slice(1);
    if (!rest.length) break;
    const { inter, union } = interUnion([src[best]!], rest.map((i) => src[i]!));
    order = rest.filter((_, j) => (inter[0]![j] ?? 0) / (union[0]![j] ?? 0) <= iouThreshold);
  }
  return kept;
}

/**
 * NMS **per class**, done by moving each class's boxes somewhere the others cannot
 * reach.
 *
 * The offset trick is torchvision's and it is worth reading twice: every box is shifted
 * by its class index times more than the largest coordinate present, so boxes of
 * different classes can no longer overlap at all and **one pass does the lot**. It
 * costs a single addition where the obvious implementation costs a loop over classes.
 */
export async function batchedNms(
  boxes: Tensor,
  scores: Tensor,
  idxs: Tensor,
  iouThreshold: number,
): Promise<Tensor> {
  const src = await rows(boxes);
  if (!src.length) return Tensor.from([], [0]);
  const values = Array.from(await scores.toArray());
  const labels = await idxs.toArray();
  let most = -Infinity;
  for (const r of src) for (const v of r) if (v > most) most = v;
  const shifted = src.map((r, i) => {
    const by = (labels[i] ?? 0) * (most + 1);
    return r.map((v) => v + by);
  });
  return Tensor.from(nmsOn(shifted, values, iouThreshold));
}

// ── v2's bounding-box kernels ────────────────────────────────────────────────
//
// **Declined as tv_tensor kernels, and they take a plain tensor.** torchvision's own
// error on a bare tensor is *"For pure tensor inputs, `format`, `canvas_size` and
// `clamping_mode` have to be passed"* — the plain-tensor path is documented and
// supported, and what the tv_tensor would have carried arrives as ordinary arguments.
//
// **Five of the seven return `[boxes, canvasSize]`** and the two flips return boxes
// alone, because a flip does not move the canvas. The core got this wrong first, from a
// probe that read `out[0] if isinstance(out, tuple) else out` — a helper that normalised
// away the exact property being measured.
//
// `format` is a string here, as everywhere else in this file. torchvision wants its
// `BoundingBoxFormat` enum and answers a string with `IndexError: index 4 is out of
// bounds`, having iterated it.

/** `(height, width)`, checked rather than unpacked — a box runs `(x, y)` and a canvas does not. */
function canvasOf(canvasSize: readonly number[], where: string): [number, number] {
  const [h, w] = canvasSize;
  if (h === undefined || w === undefined) {
    throw new Error(`${where} wants canvasSize as [height, width].`);
  }
  return [h, w];
}

/** Corners back inside a canvas — torchvision's `clamping_mode: "soft"`. */
function clampTo(box: number[], height: number, width: number): number[] {
  return [
    Math.min(Math.max(box[0] ?? 0, 0), width),
    Math.min(Math.max(box[1] ?? 0, 0), height),
    Math.min(Math.max(box[2] ?? 0, 0), width),
    Math.min(Math.max(box[3] ?? 0, 0), height),
  ];
}

function fromXyxy(box: number[], fmt: BoxFormat): number[] {
  const [x1 = 0, y1 = 0, x2 = 0, y2 = 0] = box;
  if (fmt === "xyxy") return [x1, y1, x2, y2];
  const w = x2 - x1;
  const h = y2 - y1;
  if (fmt === "xywh") return [x1, y1, w, h];
  return [x1 + 0.5 * w, y1 + 0.5 * h, w, h];
}

async function mapBoxes(
  boxes: Tensor,
  fmt: string,
  move: (xyxy: number[]) => number[],
): Promise<Tensor> {
  checkFormat(fmt, "format");
  const src = await rows(boxes);
  const out: number[] = [];
  for (const box of src) out.push(...fromXyxy(move(toXyxy(box, fmt)), fmt));
  return Tensor.from(out, [src.length, 4]);
}

/** Boxes mirrored left-to-right. */
export async function horizontalFlipBoundingBoxes(
  boundingBoxes: Tensor,
  format: string,
  canvasSize: readonly number[],
): Promise<Tensor> {
  const [, width] = canvasOf(canvasSize, "horizontalFlipBoundingBoxes");
  return mapBoxes(boundingBoxes, format,
    (b) => [width - (b[2] ?? 0), b[1] ?? 0, width - (b[0] ?? 0), b[3] ?? 0]);
}

/** Boxes mirrored top-to-bottom. */
export async function verticalFlipBoundingBoxes(
  boundingBoxes: Tensor,
  format: string,
  canvasSize: readonly number[],
): Promise<Tensor> {
  const [height] = canvasOf(canvasSize, "verticalFlipBoundingBoxes");
  return mapBoxes(boundingBoxes, format,
    (b) => [b[0] ?? 0, height - (b[3] ?? 0), b[2] ?? 0, height - (b[1] ?? 0)]);
}

/** Boxes moved into a crop's frame and clipped to it, as `[boxes, canvasSize]`. */
export async function cropBoundingBoxes(
  boundingBoxes: Tensor,
  format: string,
  top: number,
  left: number,
  height: number,
  width: number,
): Promise<[Tensor, [number, number]]> {
  const out = await mapBoxes(boundingBoxes, format,
    (b) => clampTo(
      [(b[0] ?? 0) - left, (b[1] ?? 0) - top, (b[2] ?? 0) - left, (b[3] ?? 0) - top],
      height, width));
  return [out, [height, width]];
}

/**
 * `crop` about the middle, as `[boxes, canvasSize]`.
 *
 * **The offset is `round`, and torchvision's `round` breaks ties to even.** A margin of
 * 19 gives 10 and one of 13 gives 6, from the same rule. Written as a floor division
 * this agrees on every even output size and is off by a whole pixel on odd ones, which
 * is what the golden's odd case is for — the core shipped the floor version until an
 * odd size was measured.
 */
export async function centerCropBoundingBoxes(
  boundingBoxes: Tensor,
  format: string,
  canvasSize: readonly number[],
  outputSize: readonly number[],
): Promise<[Tensor, [number, number]]> {
  const [height, width] = canvasOf(canvasSize, "centerCropBoundingBoxes");
  const [outH, outW] = canvasOf(outputSize, "centerCropBoundingBoxes");
  return cropBoundingBoxes(boundingBoxes, format,
    roundHalfToEven((height - outH) / 2), roundHalfToEven((width - outW) / 2),
    outH, outW);
}

/**
 * **`Math.round` is not Python's `round`.** `Math.round(6.5)` is 7 and Python's is 6:
 * one rounds halves up, the other to the nearer even. torchvision is written in Python,
 * so the Python rule is the one that has to be here.
 */
function roundHalfToEven(x: number): number {
  const floor = Math.floor(x);
  if (x - floor !== 0.5) return Math.round(x);
  return floor % 2 === 0 ? floor : floor + 1;
}

/** Boxes shifted by a pad, as `[boxes, canvasSize]` — the canvas grew. */
export async function padBoundingBoxes(
  boundingBoxes: Tensor,
  format: string,
  canvasSize: readonly number[],
  padding: number | readonly number[],
): Promise<[Tensor, [number, number]]> {
  const [height, width] = canvasOf(canvasSize, "padBoundingBoxes");
  let pad = typeof padding === "number" ? [padding, padding, padding, padding] : [...padding];
  if (pad.length === 2) pad = [pad[0]!, pad[1]!, pad[0]!, pad[1]!];
  if (pad.length !== 4) {
    throw new Error(`padding wants 1, 2 or 4 numbers — got ${pad.length}.`);
  }
  const [l = 0, t = 0, r = 0, b = 0] = pad;
  const out = await mapBoxes(boundingBoxes, format,
    (x) => [(x[0] ?? 0) + l, (x[1] ?? 0) + t, (x[2] ?? 0) + l, (x[3] ?? 0) + t]);
  return [out, [height + t + b, width + l + r]];
}

/**
 * Boxes scaled to a new canvas, as `[boxes, canvasSize]`.
 *
 * **`size` as a single number keeps the aspect ratio**, matching the shorter edge; as a
 * pair it does not. `maxSize` caps the longer edge and only bites in the first case, so
 * giving it with a pair raises rather than being ignored.
 */
export async function resizeBoundingBoxes(
  boundingBoxes: Tensor,
  canvasSize: readonly number[],
  size: number | readonly number[],
  maxSize?: number,
  format: string = "xyxy",
): Promise<[Tensor, [number, number]]> {
  const [height, width] = canvasOf(canvasSize, "resizeBoundingBoxes");
  // **One rule, read from one place.** The mask and keypoint kernels ask the same
  // question about `size` and `maxSize`, and this rule was written out inline here
  // first — three copies of *a single number keeps the aspect ratio* is exactly the
  // shape this repository keeps catching in its tables.
  const [newH, newW] = resizeTarget(height, width, size, maxSize, "resizeBoundingBoxes");
  const sy = newH / height;
  const sx = newW / width;
  const out = await mapBoxes(boundingBoxes, format,
    (b) => [(b[0] ?? 0) * sx, (b[1] ?? 0) * sy, (b[2] ?? 0) * sx, (b[3] ?? 0) * sy]);
  return [out, [newH, newW]];
}

/** `crop` then `resize`, as `[boxes, canvasSize]`. */
export async function resizedCropBoundingBoxes(
  boundingBoxes: Tensor,
  format: string,
  top: number,
  left: number,
  height: number,
  width: number,
  size: number | readonly number[],
): Promise<[Tensor, [number, number]]> {
  const [cropped] = await cropBoundingBoxes(boundingBoxes, format, top, left, height, width);
  return resizeBoundingBoxes(cropped, [height, width], size, undefined, format);
}

/**
 * The boxes worth keeping, as `[boxes, mask]` — **a boolean per row, not indices.**
 *
 * A mask is the shape the caller already has: labels and scores sit in parallel arrays
 * and get filtered by the same one. `removeSmallBoxes` answers with indices for the same
 * reason from the other side, and both spellings are torchvision's.
 */
export async function sanitizeBoundingBoxes(
  boundingBoxes: Tensor,
  format: string = "xyxy",
  canvasSize?: readonly number[],
  minSize = 1.0,
  minArea = 1.0,
): Promise<[Tensor, Tensor]> {
  checkFormat(format, "format");
  const src = await rows(boundingBoxes);
  const keep: number[] = [];
  const out: number[] = [];
  for (const box of src) {
    const xyxy = toXyxy(box, format);
    const w = (xyxy[2] ?? 0) - (xyxy[0] ?? 0);
    const h = (xyxy[3] ?? 0) - (xyxy[1] ?? 0);
    let ok = w >= minSize && h >= minSize && w * h >= minArea;
    if (ok && canvasSize !== undefined) {
      const [height, width] = canvasOf(canvasSize, "sanitizeBoundingBoxes");
      ok = (xyxy[0] ?? 0) >= 0 && (xyxy[1] ?? 0) >= 0
        && (xyxy[2] ?? 0) <= width && (xyxy[3] ?? 0) <= height;
    }
    keep.push(ok ? 1 : 0);
    if (ok) out.push(...box);
  }
  return [Tensor.from(out, [out.length / 4, 4]), Tensor.from(keep, [keep.length])];
}

// ── v2's mask and keypoint kernels ───────────────────────────────────────────
//
// **A mask is not an image with a different name.** Every one of these is the image
// operation with **nearest** sampling and nothing else — measured against torchvision,
// `resizeMask` equals `resize_image(interpolation=NEAREST)` to the last value, and
// `NEAREST_EXACT` is a different answer. Sampling a label map any other way averages
// class 3 and class 5 into class 4: a picture that looks right and means nothing.
//
// A mask arrives as a flat tensor whose last two dimensions are the picture; the leading
// dimensions are carried through untouched.

function hw(t: Tensor, where: string): [number, number] {
  const shape = t.shape as number[];
  if (shape.length < 2) throw new Error(`${where} wants a mask of at least two dimensions.`);
  return [shape[shape.length - 2]!, shape[shape.length - 1]!];
}

function planes(t: Tensor): number {
  const shape = t.shape as number[];
  return shape.slice(0, -2).reduce((a, b) => a * b, 1);
}

/** Which source row each destination row reads — `floor(i * src / dst)`, torch's `nearest`. */
function nearestAxis(srcLen: number, dstLen: number): number[] {
  const out: number[] = [];
  for (let i = 0; i < dstLen; i++) {
    out.push(Math.min(Math.floor((i * srcLen) / dstLen), srcLen - 1));
  }
  return out;
}

/** Resamples every plane by an explicit row and column index list. */
async function gatherMask(
  mask: Tensor,
  rows: number[],
  cols: number[],
): Promise<Tensor> {
  const flat = Array.from(await mask.toArray());
  const [h, w] = hw(mask, "gatherMask");
  const n = planes(mask);
  const out = new Array<number>(n * rows.length * cols.length);
  let k = 0;
  for (let p = 0; p < n; p++) {
    for (const r of rows) {
      for (const c of cols) out[k++] = flat[p * h * w + r * w + c] ?? 0;
    }
  }
  const lead = (mask.shape as number[]).slice(0, -2);
  return Tensor.from(out, [...lead, rows.length, cols.length]);
}

/** A mask mirrored left-to-right. */
export async function horizontalFlipMask(mask: Tensor): Promise<Tensor> {
  const [h, w] = hw(mask, "horizontalFlipMask");
  const cols: number[] = [];
  for (let c = w - 1; c >= 0; c--) cols.push(c);
  return gatherMask(mask, nearestAxis(h, h), cols);
}

/** A mask mirrored top-to-bottom. */
export async function verticalFlipMask(mask: Tensor): Promise<Tensor> {
  const [h, w] = hw(mask, "verticalFlipMask");
  const rows: number[] = [];
  for (let r = h - 1; r >= 0; r--) rows.push(r);
  return gatherMask(mask, rows, nearestAxis(w, w));
}

/**
 * A window out of a mask.
 *
 * **Outside the picture is zero rather than an error** — torchvision pads there, and a
 * crop that half-leaves the frame is ordinary in a detection pipeline.
 */
export async function cropMask(
  mask: Tensor,
  top: number,
  left: number,
  height: number,
  width: number,
): Promise<Tensor> {
  const flat = Array.from(await mask.toArray());
  const [h, w] = hw(mask, "cropMask");
  const n = planes(mask);
  const out = new Array<number>(n * height * width).fill(0);
  let k = 0;
  for (let p = 0; p < n; p++) {
    for (let i = 0; i < height; i++) {
      for (let j = 0; j < width; j++) {
        const r = top + i;
        const c = left + j;
        out[k++] = r >= 0 && r < h && c >= 0 && c < w ? flat[p * h * w + r * w + c] ?? 0 : 0;
      }
    }
  }
  const lead = (mask.shape as number[]).slice(0, -2);
  return Tensor.from(out, [...lead, height, width]);
}

/**
 * `cropMask` about the middle.
 *
 * **The offset is `round`, breaking ties to even**, as the box kernels do. A floor
 * division agrees on every even output size and is off by a whole row on odd ones.
 */
export async function centerCropMask(
  mask: Tensor,
  outputSize: readonly number[],
): Promise<Tensor> {
  const [h, w] = hw(mask, "centerCropMask");
  const [outH, outW] = canvasOf(outputSize, "centerCropMask");
  return cropMask(mask, roundHalfToEven((h - outH) / 2), roundHalfToEven((w - outW) / 2),
    outH, outW);
}

/**
 * A mask with a border.
 *
 * **`fill` is 0 by default and 0 is a class**, so padding a label map writes background
 * around it — torchvision's behaviour, and what a segmentation loss then has to be told
 * to ignore. Only `constant` is taken: reflecting or repeating an edge invents labels
 * that were never annotated.
 */
export async function padMask(
  mask: Tensor,
  padding: number | readonly number[],
  fill = 0,
  paddingMode: string = "constant",
): Promise<Tensor> {
  if (paddingMode !== "constant") {
    throw new Error(
      `padMask takes paddingMode "constant" here — got ${JSON.stringify(paddingMode)}.`,
    );
  }
  let pad = typeof padding === "number" ? [padding, padding, padding, padding] : [...padding];
  if (pad.length === 2) pad = [pad[0]!, pad[1]!, pad[0]!, pad[1]!];
  if (pad.length !== 4) throw new Error(`padding wants 1, 2 or 4 numbers — got ${pad.length}.`);
  const [l = 0, t = 0, r = 0, b = 0] = pad;
  const flat = Array.from(await mask.toArray());
  const [h, w] = hw(mask, "padMask");
  const n = planes(mask);
  const outH = h + t + b;
  const outW = w + l + r;
  const out = new Array<number>(n * outH * outW).fill(fill);
  for (let p = 0; p < n; p++) {
    for (let i = 0; i < h; i++) {
      for (let j = 0; j < w; j++) {
        out[p * outH * outW + (i + t) * outW + (j + l)] = flat[p * h * w + i * w + j] ?? 0;
      }
    }
  }
  const lead = (mask.shape as number[]).slice(0, -2);
  return Tensor.from(out, [...lead, outH, outW]);
}

/** A mask resampled, **nearest and only nearest.** */
export async function resizeMask(
  mask: Tensor,
  size: number | readonly number[],
  maxSize?: number,
): Promise<Tensor> {
  const [h, w] = hw(mask, "resizeMask");
  const [newH, newW] = resizeTarget(h, w, size, maxSize, "resizeMask");
  return gatherMask(mask, nearestAxis(h, newH), nearestAxis(w, newW));
}

/** `cropMask` then `resizeMask`. */
export async function resizedCropMask(
  mask: Tensor,
  top: number,
  left: number,
  height: number,
  width: number,
  size: number | readonly number[],
): Promise<Tensor> {
  return resizeMask(await cropMask(mask, top, left, height, width), size);
}

/** The `(height, width)` a `size` argument asks for — one rule, used by three families. */
function resizeTarget(
  srcH: number,
  srcW: number,
  size: number | readonly number[],
  maxSize: number | undefined,
  where: string,
): [number, number] {
  const want = typeof size === "number" ? [size] : [...size];
  if (want.length === 1) {
    const short = Math.min(srcH, srcW);
    const long = Math.max(srcH, srcW);
    let newShort = want[0]!;
    let newLong = (long * newShort) / short;
    if (maxSize !== undefined && newLong > maxSize) {
      newShort = (newShort * maxSize) / newLong;
      newLong = maxSize;
    }
    const [a, b] = srcH <= srcW ? [newShort, newLong] : [newLong, newShort];
    return [Math.trunc(a), Math.trunc(b)];
  }
  if (want.length === 2) {
    if (maxSize !== undefined) {
      throw new Error("maxSize is only used when size is a single number.");
    }
    return [Math.trunc(want[0]!), Math.trunc(want[1]!)];
  }
  throw new Error(`${where}: size wants one or two numbers — got ${want.length}.`);
}

// ── keypoints ────────────────────────────────────────────────────────────────
//
// **A keypoint is not a box with two numbers instead of four.** A box's `x2` is an
// exclusive edge living on `[0, width]`; a point is a pixel index living on
// `[0, width - 1]`, so the same canvas mirrors them differently and clamps them
// differently. Written the box way every one of these is off by exactly one pixel —
// a skeleton that is still a skeleton, drawn one column from where it belongs.

async function points(t: Tensor): Promise<number[]> {
  return Array.from(await t.toArray());
}

async function mapPoints(
  keypoints: Tensor,
  move: (x: number, y: number) => [number, number],
): Promise<Tensor> {
  const flat = await points(keypoints);
  const out = new Array<number>(flat.length);
  for (let i = 0; i < flat.length; i += 2) {
    const [x, y] = move(flat[i] ?? 0, flat[i + 1] ?? 0);
    out[i] = x;
    out[i + 1] = y;
  }
  return Tensor.from(out, keypoints.shape as number[]);
}

/** Points mirrored left-to-right — `(width - 1) - x`, **not** `width - x`. */
export async function horizontalFlipKeypoints(
  keypoints: Tensor,
  canvasSize: readonly number[],
): Promise<Tensor> {
  const [, width] = canvasOf(canvasSize, "horizontalFlipKeypoints");
  return mapPoints(keypoints, (x, y) => [width - 1 - x, y]);
}

/** Points mirrored top-to-bottom — `(height - 1) - y`. */
export async function verticalFlipKeypoints(
  keypoints: Tensor,
  canvasSize: readonly number[],
): Promise<Tensor> {
  const [height] = canvasOf(canvasSize, "verticalFlipKeypoints");
  return mapPoints(keypoints, (x, y) => [x, height - 1 - y]);
}

/**
 * Points moved into a crop's frame, as `[keypoints, canvasSize]`.
 *
 * **Nothing is clipped and nothing is dropped.** A point outside the crop keeps its
 * negative coordinate — it is still that keypoint, and whether it counts is
 * `sanitizeKeypoints`' decision.
 */
export async function cropKeypoints(
  keypoints: Tensor,
  top: number,
  left: number,
  height: number,
  width: number,
): Promise<[Tensor, [number, number]]> {
  const out = await mapPoints(keypoints, (x, y) => [x - left, y - top]);
  return [out, [height, width]];
}

/** `cropKeypoints` about the middle, as `[keypoints, canvasSize]`. */
export async function centerCropKeypoints(
  inpt: Tensor,
  canvasSize: readonly number[],
  outputSize: readonly number[],
): Promise<[Tensor, [number, number]]> {
  const [height, width] = canvasOf(canvasSize, "centerCropKeypoints");
  const [outH, outW] = canvasOf(outputSize, "centerCropKeypoints");
  return cropKeypoints(inpt, roundHalfToEven((height - outH) / 2),
    roundHalfToEven((width - outW) / 2), outH, outW);
}

/** Points shifted by a pad, as `[keypoints, canvasSize]`. */
export async function padKeypoints(
  keypoints: Tensor,
  canvasSize: readonly number[],
  padding: number | readonly number[],
): Promise<[Tensor, [number, number]]> {
  const [height, width] = canvasOf(canvasSize, "padKeypoints");
  let pad = typeof padding === "number" ? [padding, padding, padding, padding] : [...padding];
  if (pad.length === 2) pad = [pad[0]!, pad[1]!, pad[0]!, pad[1]!];
  if (pad.length !== 4) throw new Error(`padding wants 1, 2 or 4 numbers — got ${pad.length}.`);
  const [l = 0, t = 0, r = 0, b = 0] = pad;
  const out = await mapPoints(keypoints, (x, y) => [x + l, y + t]);
  return [out, [height + t + b, width + l + r]];
}

/** Points scaled to a new canvas, as `[keypoints, canvasSize]`. */
export async function resizeKeypoints(
  keypoints: Tensor,
  size: number | readonly number[],
  canvasSize: readonly number[],
  maxSize?: number,
): Promise<[Tensor, [number, number]]> {
  const [height, width] = canvasOf(canvasSize, "resizeKeypoints");
  const [newH, newW] = resizeTarget(height, width, size, maxSize, "resizeKeypoints");
  const out = await mapPoints(keypoints, (x, y) => [(x * newW) / width, (y * newH) / height]);
  return [out, [newH, newW]];
}

/** `cropKeypoints` then `resizeKeypoints`, as `[keypoints, canvasSize]`. */
export async function resizedCropKeypoints(
  keypoints: Tensor,
  top: number,
  left: number,
  height: number,
  width: number,
  size: number | readonly number[],
): Promise<[Tensor, [number, number]]> {
  const [cropped, canvas] = await cropKeypoints(keypoints, top, left, height, width);
  return resizeKeypoints(cropped, size, canvas);
}

/** Points pushed back inside the canvas — **the last pixel is `width - 1`.** */
export async function clampKeypoints(
  keypoints: Tensor,
  canvasSize: readonly number[],
): Promise<Tensor> {
  const [height, width] = canvasOf(canvasSize, "clampKeypoints");
  return mapPoints(keypoints, (x, y) => [
    Math.min(Math.max(x, 0), width - 1),
    Math.min(Math.max(y, 0), height - 1),
  ]);
}

/**
 * The point groups worth keeping, as `[keypoints, mask]`.
 *
 * **The unit is the group, not the point.** These arrive as `(N, K, 2)` — `N` skeletons
 * of `K` points each — and a group survives only if **every** one of its points is
 * inside the canvas. Filtering point by point is the obvious reading and returns half a
 * pose with the joints renumbered.
 */
export async function sanitizeKeypoints(
  keyPoints: Tensor,
  canvasSize?: readonly number[],
): Promise<[Tensor, Tensor]> {
  if (canvasSize === undefined) {
    throw new Error(
      "canvasSize cannot be undefined if keyPoints is a pure tensor. Set it to an appropriate value.",
    );
  }
  const [height, width] = canvasOf(canvasSize, "sanitizeKeypoints");
  const flat = await points(keyPoints);
  const shape = keyPoints.shape as number[];
  const groups = shape[0] ?? 1;
  const per = flat.length / Math.max(groups, 1);
  const keep: number[] = [];
  const out: number[] = [];
  for (let g = 0; g < groups; g++) {
    let ok = true;
    for (let i = 0; i < per; i += 2) {
      const x = flat[g * per + i] ?? 0;
      const y = flat[g * per + i + 1] ?? 0;
      if (x < 0 || x > width || y < 0 || y > height) ok = false;
    }
    keep.push(ok ? 1 : 0);
    if (ok) out.push(...flat.slice(g * per, (g + 1) * per));
  }
  const kept = keep.reduce((a, b) => a + b, 0);
  return [
    Tensor.from(out, [kept, ...shape.slice(1)]),
    Tensor.from(keep, [groups]),
  ];
}

/* ── sampling a feature map at coordinates that are not integers ──────────────
 *
 * **Eleven names in `ops` waited on one piece of arithmetic**, and their rows on the
 * Python side read *it crops from a feature map. A feature map comes from a model* —
 * true of where the tensor came from, not of what the function needs.
 *
 * **This side computes in JavaScript numbers, as the rest of this file does.** The
 * Python `roi_align` is written in the backend's own operations so that a gradient
 * reaches the feature map; here `boxIou` and the losses beside it all read to an array
 * and write a tensor back, and one differentiable function among value-level neighbours
 * would be the odder thing. What that costs is written into the gap table rather than
 * left for somebody to discover: a detector trained in the browser would want the
 * gradient, and this does not give it.
 */

interface Feature {
  values: number[];
  batch: number;
  channels: number;
  height: number;
  width: number;
}

async function featureOf(input: Tensor, where: string): Promise<Feature> {
  const shape = input.shape as number[];
  if (shape.length !== 4) {
    throw new Error(`${where} expects a (N, C, H, W) tensor — got ${shape.length} axes.`);
  }
  return {
    values: Array.from(await input.toArray()),
    batch: shape[0] ?? 0,
    channels: shape[1] ?? 0,
    height: shape[2] ?? 0,
    width: shape[3] ?? 0,
  };
}

/**
 * One value out of a feature map at a fractional `(y, x)`.
 *
 * **A sample outside the map contributes zero, and is still counted.** Not the clamped
 * edge value — torchvision's own Python reference clamps and its compiled kernel does
 * not, and the kernel is what a caller meets. Measured on the Python side: a box hanging
 * half off the left edge comes back at exactly twice torchvision's answer when the
 * clamping rule is used, one real sample and one zero averaged over two.
 *
 * The last row and column have no neighbour above them, so the high index is held there
 * and the coordinate is snapped with it — moving one without the other weighs a
 * neighbour that is not the one being read.
 */
function sampleAt(f: Feature, image: number, channel: number, y: number, x: number): number {
  if (y < -1 || y > f.height || x < -1 || x > f.width) return 0;
  const yy = Math.max(y, 0);
  const xx = Math.max(x, 0);
  let yLow = Math.floor(yy);
  let xLow = Math.floor(xx);
  let yHigh = yLow + 1;
  let xHigh = xLow + 1;
  let dy = yy;
  let dx = xx;
  if (yLow >= f.height - 1) {
    yHigh = f.height - 1;
    yLow = f.height - 1;
    dy = yLow;
  }
  if (xLow >= f.width - 1) {
    xHigh = f.width - 1;
    xLow = f.width - 1;
    dx = xLow;
  }
  const ly = dy - yLow;
  const lx = dx - xLow;
  const hy = 1 - ly;
  const hx = 1 - lx;
  const plane = (image * f.channels + channel) * f.height * f.width;
  const at = (r: number, c: number) => f.values[plane + r * f.width + c] ?? 0;
  return hy * hx * at(yLow, xLow) + hy * lx * at(yLow, xHigh)
    + ly * hx * at(yHigh, xLow) + ly * lx * at(yHigh, xHigh);
}

/** `(k, 5)` rows — a batch index and four coordinates. */
async function roiRows(boxes: Tensor): Promise<number[][]> {
  const flat = Array.from(await boxes.toArray());
  if (flat.length % 5 !== 0) {
    throw new Error(
      `boxes has ${flat.length} numbers, which is not a whole number of `
      + "(batch, x1, y1, x2, y2) rows.",
    );
  }
  const out: number[][] = [];
  for (let i = 0; i < flat.length; i += 5) {
    out.push([flat[i] ?? 0, flat[i + 1] ?? 0, flat[i + 2] ?? 0,
              flat[i + 3] ?? 0, flat[i + 4] ?? 0]);
  }
  return out;
}

function pairOfSizes(outputSize: number | readonly number[]): [number, number] {
  if (typeof outputSize === "number") return [outputSize, outputSize];
  return [outputSize[0] ?? 1, outputSize[1] ?? outputSize[0] ?? 1];
}

/**
 * Crop each box out of the feature map at a fixed size, **without rounding it.**
 *
 * That is the whole difference from `roiPool` below and what the paper is named for: a
 * box's edges land between pixels, and rounding them to the grid moves the crop by up to
 * half a cell.
 *
 * - **`spatialScale` is how much smaller the map is than the image.** A stride-16
 *   backbone wants `1/16`; left at 1 this samples the top-left sixteenth of everything.
 * - **`samplingRatio = -1` means as many samples as the bin is wide**, computed per box.
 * - **`aligned` shifts by half a pixel** and stops clamping the box to a minimum size of
 *   one. It is the correction to the original implementation and it moves every output;
 *   `false` is the default because that is what the released weights were trained with.
 */
export async function roiAlign(
  input: Tensor,
  boxes: Tensor,
  outputSize: number | readonly number[],
  spatialScale = 1,
  samplingRatio = -1,
  aligned = false,
): Promise<Tensor> {
  const f = await featureOf(input, "roiAlign");
  const rois = await roiRows(boxes);
  const [ph, pw] = pairOfSizes(outputSize);
  const out: number[] = [];
  const shift = aligned ? 0.5 : 0;
  for (const roi of rois) {
    const image = Math.trunc(roi[0] ?? 0);
    const startW = (roi[1] ?? 0) * spatialScale - shift;
    const startH = (roi[2] ?? 0) * spatialScale - shift;
    let roiW = (roi[3] ?? 0) * spatialScale - shift - startW;
    let roiH = (roi[4] ?? 0) * spatialScale - shift - startH;
    if (!aligned) {
      roiW = Math.max(roiW, 1);
      roiH = Math.max(roiH, 1);
    }
    const binH = roiH / ph;
    const binW = roiW / pw;
    const gridH = samplingRatio > 0 ? samplingRatio : Math.ceil(roiH / ph);
    const gridW = samplingRatio > 0 ? samplingRatio : Math.ceil(roiW / pw);
    const count = Math.max(gridH * gridW, 1);
    for (let c = 0; c < f.channels; c++) {
      for (let i = 0; i < ph; i++) {
        for (let j = 0; j < pw; j++) {
          let total = 0;
          for (let iy = 0; iy < gridH; iy++) {
            const y = startH + i * binH + (iy + 0.5) * (binH / gridH);
            for (let ix = 0; ix < gridW; ix++) {
              const x = startW + j * binW + (ix + 0.5) * (binW / gridW);
              total += sampleAt(f, image, c, y, x);
            }
          }
          out.push(total / count);
        }
      }
    }
  }
  return Tensor.from(out, [rois.length, f.channels, ph, pw]);
}

/**
 * The **maximum** in each bin, with the box rounded to the grid.
 *
 * Faster R-CNN's pooling, and the thing `roiAlign` was written to replace.
 *
 * **An empty bin answers zero rather than negative infinity.** A box small enough that a
 * bin covers no cell at all is real — a distant object on a stride-32 map — and a
 * maximum over nothing has no value to give.
 *
 * **The rounding is C's, halves away from zero**, where JavaScript's `Math.round` sends
 * `-0.5` to `-0` and `0.5` to `1`. At `spatialScale = 1` nothing ever lands on a half;
 * at `0.5` every odd coordinate does.
 */
export async function roiPool(
  input: Tensor,
  boxes: Tensor,
  outputSize: number | readonly number[],
  spatialScale = 1,
): Promise<Tensor> {
  const f = await featureOf(input, "roiPool");
  const rois = await roiRows(boxes);
  const [ph, pw] = pairOfSizes(outputSize);
  const out: number[] = [];
  for (const roi of rois) {
    const image = Math.trunc(roi[0] ?? 0);
    const left = cRound((roi[1] ?? 0) * spatialScale);
    const top = cRound((roi[2] ?? 0) * spatialScale);
    const right = cRound((roi[3] ?? 0) * spatialScale);
    const bottom = cRound((roi[4] ?? 0) * spatialScale);
    const boxH = Math.max(bottom - top + 1, 1);
    const boxW = Math.max(right - left + 1, 1);
    for (let c = 0; c < f.channels; c++) {
      const plane = (image * f.channels + c) * f.height * f.width;
      for (let i = 0; i < ph; i++) {
        for (let j = 0; j < pw; j++) {
          const y0 = Math.min(Math.max(top + Math.floor((i * boxH) / ph), 0), f.height);
          const y1 = Math.min(Math.max(top + Math.ceil(((i + 1) * boxH) / ph), 0), f.height);
          const x0 = Math.min(Math.max(left + Math.floor((j * boxW) / pw), 0), f.width);
          const x1 = Math.min(Math.max(left + Math.ceil(((j + 1) * boxW) / pw), 0), f.width);
          if (y1 <= y0 || x1 <= x0) {
            out.push(0);
            continue;
          }
          let best = -Infinity;
          for (let r = y0; r < y1; r++) {
            for (let cc = x0; cc < x1; cc++) {
              best = Math.max(best, f.values[plane + r * f.width + cc] ?? 0);
            }
          }
          out.push(best);
        }
      }
    }
  }
  return Tensor.from(out, [rois.length, f.channels, ph, pw]);
}

/** C's `round` — to the nearest, halves **away from zero**. */
function cRound(value: number): number {
  return Math.sign(value) * Math.floor(Math.abs(value) + 0.5);
}

function psChannels(f: Feature, ph: number, pw: number): number {
  if (f.channels % (ph * pw) !== 0) {
    throw new Error("input channels must be a multiple of pooling height * pooling width");
  }
  return f.channels / (ph * pw);
}

/**
 * `roiAlign`, **but each bin reads its own bank of channels.**
 *
 * R-FCN's pooling: the convolutions before it produce a map that already knows *where*
 * in the object each channel is looking, so the position is in the channel index.
 *
 * **The output channel is the slow axis and the bin is the fast one** —
 * `c * ph * pw + (i * pw + j)`. The other order gives an output of exactly the right
 * shape built from the wrong channels.
 *
 * **It has no `aligned` flag and the correction is on.** `roiAlign` grew a flag and
 * defaults it to `false`; this always subtracts the half pixel and never clamps the box
 * to a minimum size. The absent argument is not the absent behaviour.
 */
export async function psRoiAlign(
  input: Tensor,
  boxes: Tensor,
  outputSize: number,
  spatialScale = 1,
  samplingRatio = -1,
): Promise<Tensor> {
  const f = await featureOf(input, "psRoiAlign");
  const rois = await roiRows(boxes);
  const [ph, pw] = pairOfSizes(outputSize);
  const outChannels = psChannels(f, ph, pw);
  const out: number[] = [];
  for (const roi of rois) {
    const image = Math.trunc(roi[0] ?? 0);
    const startW = (roi[1] ?? 0) * spatialScale - 0.5;
    const startH = (roi[2] ?? 0) * spatialScale - 0.5;
    const roiW = (roi[3] ?? 0) * spatialScale - 0.5 - startW;
    const roiH = (roi[4] ?? 0) * spatialScale - 0.5 - startH;
    const binH = roiH / ph;
    const binW = roiW / pw;
    const gridH = samplingRatio > 0 ? samplingRatio : Math.ceil(binH);
    const gridW = samplingRatio > 0 ? samplingRatio : Math.ceil(binW);
    const count = Math.max(gridH * gridW, 1);
    for (let c = 0; c < outChannels; c++) {
      for (let i = 0; i < ph; i++) {
        for (let j = 0; j < pw; j++) {
          const bank = c * ph * pw + i * pw + j;
          let total = 0;
          for (let iy = 0; iy < gridH; iy++) {
            const y = startH + i * binH + (iy + 0.5) * (binH / gridH);
            for (let ix = 0; ix < gridW; ix++) {
              const x = startW + j * binW + (ix + 0.5) * (binW / gridW);
              total += sampleAt(f, image, bank, y, x);
            }
          }
          out.push(total / count);
        }
      }
    }
  }
  return Tensor.from(out, [rois.length, outChannels, ph, pw]);
}

/**
 * `roiPool`'s grid with `psRoiAlign`'s channel banks — and **an average, not a maximum.**
 *
 * That is the one thing here that reads like a typo and is not: a maximum over a
 * position-sensitive bank would pick the single most confident position and throw away
 * the vote, which is the opposite of what the bank is for.
 *
 * **Two things differ from `roiPool`** and both only show on a box that runs off the
 * map: the rounding takes the start inside it, and the clip is to the last index rather
 * than to the size.
 */
export async function psRoiPool(
  input: Tensor,
  boxes: Tensor,
  outputSize: number,
  spatialScale = 1,
): Promise<Tensor> {
  const f = await featureOf(input, "psRoiPool");
  const rois = await roiRows(boxes);
  const [ph, pw] = pairOfSizes(outputSize);
  const outChannels = psChannels(f, ph, pw);
  const out: number[] = [];
  for (const roi of rois) {
    const image = Math.trunc(roi[0] ?? 0);
    const left = cRound((roi[1] ?? 0) * spatialScale);
    const top = cRound((roi[2] ?? 0) * spatialScale);
    const right = cRound((roi[3] ?? 0) * spatialScale);
    const bottom = cRound((roi[4] ?? 0) * spatialScale);
    const boxH = Math.max(bottom - top, 0.1);
    const boxW = Math.max(right - left, 0.1);
    for (let c = 0; c < outChannels; c++) {
      for (let i = 0; i < ph; i++) {
        for (let j = 0; j < pw; j++) {
          const y0 = Math.min(Math.max(Math.floor((i * boxH) / ph + top), 0), f.height - 1);
          const y1 = Math.min(Math.max(Math.ceil(((i + 1) * boxH) / ph + top), 0), f.height - 1);
          const x0 = Math.min(Math.max(Math.floor((j * boxW) / pw + left), 0), f.width - 1);
          const x1 = Math.min(Math.max(Math.ceil(((j + 1) * boxW) / pw + left), 0), f.width - 1);
          const bank = c * ph * pw + i * pw + j;
          const plane = (image * f.channels + bank) * f.height * f.width;
          if (y1 <= y0 || x1 <= x0) {
            out.push(0);
            continue;
          }
          let total = 0;
          let seen = 0;
          for (let r = y0; r < y1; r++) {
            for (let cc = x0; cc < x1; cc++) {
              total += f.values[plane + r * f.width + cc] ?? 0;
              seen += 1;
            }
          }
          out.push(seen ? total / seen : 0);
        }
      }
    }
  }
  return Tensor.from(out, [rois.length, outChannels, ph, pw]);
}

// ── v2's corner warps ────────────────────────────────────────────────────────
//
// `affine`, `rotate`, `perspective` and `elastic`, for boxes, masks and keypoints —
// twelve names and three quite different jobs.
//
// **A mask is the image kernel with `nearest`**, measured for all four. **A keypoint is
// the transform applied to the point. A box is the transform applied to its four
// corners, then the axis-aligned box around them** — which is why a rotated box grows:
// the hull of a tilted rectangle is larger than the rectangle.
//
// The return shapes are not uniform and were measured one at a time:
//
//     affine        boxes: bare              keypoints: [points, canvas]
//     rotate        boxes: [boxes, canvas]   keypoints: [points, canvas]
//     perspective   boxes: bare              keypoints: bare
//     elastic       boxes: bare              keypoints: bare
//
// **Boxes clamp and keypoints do not.** A box that half-leaves the picture is still a
// region of it; a keypoint that leaves is outside, and saying so is the only way
// `sanitizeKeypoints` can mean anything.

type Affine = {
  readonly m: readonly [number, number, number, number];
  readonly c: readonly [number, number];
  readonly t: readonly [number, number];
};

/**
 * The transform that moves a **point**, from the one that moves a grid.
 *
 * `inverseAffineMatrix` answers what image resampling needs — output positions mapped
 * back to input ones — so its 2x2 is inverted here.
 *
 * **The centre is absolute here and an offset there.** An image grid is already centred,
 * so torch passes a displacement from the middle and defaults it to `[0, 0]`; a point
 * lives in pixel space, where the origin is the corner and the default centre is
 * `[w/2, h/2]`. Using the grid convention for points is off by exactly one centring —
 * measured, its answer for the default centre equalled torchvision's answer for
 * `center: [0, 0]`, all the way through.
 *
 * `translate` is applied **after** the linear part, which is what makes
 * `translate: [3, -2]` at angle 0 move a box by exactly three and two.
 */
function forwardAffine(
  center: readonly number[] | null | undefined,
  angle: number,
  translate: readonly number[],
  scale: number,
  shear: number | readonly number[],
  w: number,
  h: number,
): Affine {
  const inv = inverseAffineMatrix([0, 0], angle, [0, 0], scale, shearPair(shear));
  const [a = 0, b = 0, , d = 0, e = 0] = inv;
  const det = a * e - b * d;
  if (Math.abs(det) < 1e-12) {
    throw new Error("this affine collapses the picture to a line.");
  }
  const cxy: readonly [number, number] = center === null || center === undefined
    ? [w * 0.5, h * 0.5]
    : [center[0] ?? 0, center[1] ?? 0];
  return {
    m: [e / det, -b / det, -d / det, a / det],
    c: cxy,
    t: [translate[0] ?? 0, translate[1] ?? 0],
  };
}

function applyAffine(p: readonly [number, number], f: Affine): [number, number] {
  const x = p[0] - f.c[0];
  const y = p[1] - f.c[1];
  return [
    f.m[0] * x + f.m[1] * y + f.c[0] + f.t[0],
    f.m[2] * x + f.m[3] * y + f.c[1] + f.t[1],
  ];
}

function applyPerspective(
  p: readonly [number, number],
  k: readonly number[],
): [number, number] {
  const [a = 0, b = 0, c = 0, d = 0, e = 0, f = 0, g = 0, h = 0] = k;
  const den = g * p[0] + h * p[1] + 1;
  return [(a * p[0] + b * p[1] + c) / den, (d * p[0] + e * p[1] + f) / den];
}

/** Corners through `move`, then the hull of each four, then clipped to `canvas`. */
async function warpBoxes(
  boundingBoxes: Tensor,
  format: string,
  canvas: readonly [number, number],
  move: (p: readonly [number, number]) => [number, number],
): Promise<Tensor> {
  checkFormat(format, "format");
  const src = await rows(boundingBoxes);
  const out: number[] = [];
  for (const box of src) {
    const [x1 = 0, y1 = 0, x2 = 0, y2 = 0] = toXyxy(box, format);
    const corners: [number, number][] =
      [move([x1, y1]), move([x2, y1]), move([x2, y2]), move([x1, y2])];
    const xs = corners.map((p) => p[0]);
    const ys = corners.map((p) => p[1]);
    const hull = clampTo(
      [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)],
      canvas[0], canvas[1]);
    out.push(...fromXyxy(hull, format));
  }
  return Tensor.from(out, [src.length, 4]);
}

async function warpPoints(
  keypoints: Tensor,
  move: (p: readonly [number, number]) => [number, number],
): Promise<Tensor> {
  const flat = Array.from(await keypoints.toArray());
  const out = new Array<number>(flat.length);
  for (let i = 0; i < flat.length; i += 2) {
    const [x, y] = move([flat[i] ?? 0, flat[i + 1] ?? 0]);
    out[i] = x;
    out[i + 1] = y;
  }
  return Tensor.from(out, keypoints.shape as number[]);
}

/** A mask through an image warp — **the layout moves and nothing else does.** */
async function maskThrough(
  mask: Tensor,
  call: (img: vision.Image) => vision.Image,
): Promise<Tensor> {
  const flat = Array.from(await mask.toArray());
  const [h, w] = hw(mask, "maskThrough");
  const n = planes(mask);
  const hwc = new Float64Array(h * w * n);
  for (let p = 0; p < n; p++) {
    for (let i = 0; i < h * w; i++) hwc[i * n + p] = flat[p * h * w + i] ?? 0;
  }
  const got = call(vision.image(hwc, h, w, n, false));
  const out = new Array<number>(got.height * got.width * n);
  for (let p = 0; p < n; p++) {
    for (let i = 0; i < got.height * got.width; i++) {
      out[p * got.height * got.width + i] = got.data[i * n + p] ?? 0;
    }
  }
  const lead = (mask.shape as number[]).slice(0, -2);
  return Tensor.from(out, [...lead, got.height, got.width]);
}

/** `affine` on a label map — nearest, and nearest is not a default here. */
export async function affineMask(
  mask: Tensor,
  angle: number,
  translate: readonly number[],
  scale: number,
  shear: number | readonly number[],
  fill?: number | readonly number[],
  center?: readonly number[] | null,
): Promise<Tensor> {
  return maskThrough(mask, (img) => vision.affine(
    img, angle, translate as [number, number], scale, shear as [number, number],
    "nearest", fill, center as [number, number] | null | undefined));
}

/** `rotate` on a label map. */
export async function rotateMask(
  mask: Tensor,
  angle: number,
  expand = false,
  center?: readonly number[] | null,
  fill?: number | readonly number[],
): Promise<Tensor> {
  return maskThrough(mask, (img) => vision.rotate(
    img, angle, "nearest", expand, center as [number, number] | null | undefined, fill));
}

/** `perspective` on a label map. */
export async function perspectiveMask(
  mask: Tensor,
  startpoints: readonly (readonly number[])[],
  endpoints: readonly (readonly number[])[],
  fill?: number | readonly number[],
): Promise<Tensor> {
  return maskThrough(mask, (img) => vision.perspective(
    img, startpoints as [number, number][], endpoints as [number, number][],
    "nearest", fill));
}

/** `elastic` on a label map. */
export async function elasticMask(
  mask: Tensor,
  displacement: readonly number[],
  fill?: number | readonly number[],
): Promise<Tensor> {
  // **No canvas argument here and one on the box and keypoint versions**, because a
  // mask carries its own size in its shape and a box does not. torchvision's signature
  // says the same, and matching it is what `ts_signatures.py` compares.
  return maskThrough(mask, (img) => vision.elasticTransform(
    img, displacement as number[], "nearest", fill));
}

/** Points through an affine, as `[keypoints, canvasSize]`. Not clamped. */
export async function affineKeypoints(
  keypoints: Tensor,
  canvasSize: readonly number[],
  angle: number,
  translate: readonly number[],
  scale: number,
  shear: number | readonly number[],
  center?: readonly number[] | null,
): Promise<[Tensor, [number, number]]> {
  const [height, width] = canvasOf(canvasSize, "affineKeypoints");
  const f = forwardAffine(center, angle, translate, scale, shear, width, height);
  return [await warpPoints(keypoints, (p) => applyAffine(p, f)), [height, width]];
}

/**
 * Box corners through an affine, then the axis-aligned box around them.
 *
 * **A rotated box grows**, and that is the transform being honest rather than a defect:
 * the smallest upright rectangle containing a tilted one is larger than it. Repeated
 * rotation therefore inflates a box, which is why detection pipelines rotate once.
 */
export async function affineBoundingBoxes(
  boundingBoxes: Tensor,
  format: string,
  canvasSize: readonly number[],
  angle: number,
  translate: readonly number[],
  scale: number,
  shear: number | readonly number[],
  center?: readonly number[] | null,
): Promise<Tensor> {
  const [height, width] = canvasOf(canvasSize, "affineBoundingBoxes");
  const f = forwardAffine(center, angle, translate, scale, shear, width, height);
  return warpBoxes(boundingBoxes, format, [height, width], (p) => applyAffine(p, f));
}

/**
 * `[newHeight, newWidth]` for `expand`, and **this is not the image's answer.**
 *
 * torchvision's coordinate path and its image path disagree about the expanded size,
 * because they hand the same size function different matrices: the image path passes a
 * centre **offset** (`[0, 0]` by default, a grid being already centred) and this path
 * passes the **absolute** centre. Measured across four canvases and six angles they part
 * on most, in both directions — a 24x32 at 30 degrees is 38 rows to the image and 37 to
 * the boxes. Both are reproduced rather than reconciled.
 */
function expandedCanvas(
  angle: number,
  center: readonly number[] | null | undefined,
  width: number,
  height: number,
): [number, number] {
  const abs: [number, number] = center === null || center === undefined
    ? [width * 0.5, height * 0.5]
    : [center[0] ?? 0, center[1] ?? 0];
  const matrix = inverseAffineMatrix(abs, -angle, [0, 0], 1, [0, 0]);
  const [ow, oh] = affineOutputSize(matrix, Math.trunc(width), Math.trunc(height));
  return [oh, ow];
}

/**
 * The rotation, and the shift that puts the turned picture's own corner at the origin.
 *
 * **The shift is the minimum of the transformed canvas corners**, not half the growth in
 * each direction — half the growth is wrong by a fraction of a pixel whenever the
 * picture is not square about its centre, which reads as a rounding difference and is
 * not one.
 */
function rotationMove(
  angle: number,
  center: readonly number[] | null | undefined,
  width: number,
  height: number,
  expand: boolean,
): (p: readonly [number, number]) => [number, number] {
  const f = forwardAffine(center, -angle, [0, 0], 1, [0, 0], width, height);
  const turn = (p: readonly [number, number]): [number, number] => applyAffine(p, f);
  if (!expand) return turn;
  const frame: [number, number][] = [[0, 0], [0, height], [width, height], [width, 0]];
  const moved = frame.map(turn);
  const sx = Math.min(...moved.map((p) => p[0]));
  const sy = Math.min(...moved.map((p) => p[1]));
  return (p) => {
    const q = turn(p);
    return [q[0] - sx, q[1] - sy];
  };
}

/** Points turned about the centre, as `[keypoints, canvasSize]`. */
export async function rotateKeypoints(
  keypoints: Tensor,
  canvasSize: readonly number[],
  angle: number,
  expand = false,
  center?: readonly number[] | null,
): Promise<[Tensor, [number, number]]> {
  const [height, width] = canvasOf(canvasSize, "rotateKeypoints");
  const canvas = expand ? expandedCanvas(angle, center, width, height) : [height, width];
  const move = rotationMove(angle, center, width, height, expand);
  return [await warpPoints(keypoints, move), canvas as [number, number]];
}

/** Box corners turned about the centre, as `[boxes, canvasSize]`. */
export async function rotateBoundingBoxes(
  boundingBoxes: Tensor,
  format: string,
  canvasSize: readonly number[],
  angle: number,
  expand = false,
  center?: readonly number[] | null,
): Promise<[Tensor, [number, number]]> {
  const [height, width] = canvasOf(canvasSize, "rotateBoundingBoxes");
  const canvas = (expand ? expandedCanvas(angle, center, width, height)
    : [height, width]) as [number, number];
  const move = rotationMove(angle, center, width, height, expand);
  return [await warpBoxes(boundingBoxes, format, canvas, move), canvas];
}

/**
 * **The arguments go in the other order than for an image.**
 *
 * `perspectiveCoefficients` solves the map an image needs — output positions back to
 * input ones, so endpoints to startpoints. A corner travels the other way, and swapping
 * the two lists is the whole difference.
 */
function forwardPerspective(
  startpoints: readonly (readonly number[])[],
  endpoints: readonly (readonly number[])[],
  coefficients?: readonly number[] | null,
): readonly number[] {
  if (coefficients !== null && coefficients !== undefined) return coefficients;
  return perspectiveCoefficients(endpoints as [number, number][],
    startpoints as [number, number][]);
}

/** Points through a projective map. Not clamped, and not a pair. */
export async function perspectiveKeypoints(
  keypoints: Tensor,
  canvasSize: readonly number[],
  startpoints: readonly (readonly number[])[],
  endpoints: readonly (readonly number[])[],
  coefficients?: readonly number[] | null,
): Promise<Tensor> {
  canvasOf(canvasSize, "perspectiveKeypoints");
  const k = forwardPerspective(startpoints, endpoints, coefficients);
  return warpPoints(keypoints, (p) => applyPerspective(p, k));
}

/** Box corners through a projective map, then the hull. */
export async function perspectiveBoundingBoxes(
  boundingBoxes: Tensor,
  format: string,
  canvasSize: readonly number[],
  startpoints: readonly (readonly number[])[],
  endpoints: readonly (readonly number[])[],
  coefficients?: readonly number[] | null,
): Promise<Tensor> {
  const [height, width] = canvasOf(canvasSize, "perspectiveBoundingBoxes");
  const k = forwardPerspective(startpoints, endpoints, coefficients);
  return warpBoxes(boundingBoxes, format, [height, width],
    (p) => applyPerspective(p, k));
}

/**
 * The displacement field, read at each point's own pixel.
 *
 * **The field is in normalised coordinates**, `[-1, 1]` across the picture, so `0.1` is
 * `0.1 * width / 2` pixels. And it is **subtracted**: the field says where a destination
 * pixel reads *from*, so a feature at `p` ends up at `p` minus what the field holds
 * there.
 */
function elasticMove(
  displacement: readonly number[],
  fieldH: number,
  fieldW: number,
  width: number,
  height: number,
): (p: readonly [number, number]) => [number, number] {
  // **The point snaps to that pixel too** — floored, clamped to the canvas, and the
  // displacement applied to the snapped point rather than the original. The fractional
  // part is discarded. Measured against torchvision with a field whose every entry names
  // its own coordinate: `(10.4, 5.6)` and `(10.6, 5.4)` come back identical, and
  // `(33, 26)` on a 24x32 canvas comes back identical to `(31, 23)`.
  //
  // Clamping only the lookup index is the obvious reading and agrees on every point
  // inside the picture. The core shipped that version until a case put a corner at
  // `(32, 24)`.
  return (p) => {
    const col = Math.min(Math.max(Math.floor(p[0]), 0), fieldW - 1);
    const row = Math.min(Math.max(Math.floor(p[1]), 0), fieldH - 1);
    const at = (row * fieldW + col) * 2;
    return [
      col - (displacement[at] ?? 0) * width / 2,
      row - (displacement[at + 1] ?? 0) * height / 2,
    ];
  };
}

/** Points moved by a displacement field. Not clamped, and not a pair. */
export async function elasticKeypoints(
  keypoints: Tensor,
  canvasSize: readonly number[],
  displacement: readonly number[],
  fieldHeight: number,
  fieldWidth: number,
): Promise<Tensor> {
  const [height, width] = canvasOf(canvasSize, "elasticKeypoints");
  return warpPoints(keypoints,
    elasticMove(displacement, fieldHeight, fieldWidth, width, height));
}

/** Box corners moved by a displacement field, then the hull. */
export async function elasticBoundingBoxes(
  boundingBoxes: Tensor,
  format: string,
  canvasSize: readonly number[],
  displacement: readonly number[],
  fieldHeight: number,
  fieldWidth: number,
): Promise<Tensor> {
  const [height, width] = canvasOf(canvasSize, "elasticBoundingBoxes");
  return warpBoxes(boundingBoxes, format, [height, width],
    elasticMove(displacement, fieldHeight, fieldWidth, width, height));
}

/**
 * A convolution whose sampling positions are **learned.**
 *
 * <https://arxiv.org/abs/1703.06211> — and v2, with `mask`,
 * <https://arxiv.org/abs/1811.11168>
 *
 * **Its sampling is not `sampleAt`'s rule with different bounds — it is a different
 * rule.** That one clamps the coordinate to the edge and reads four real neighbours;
 * this one keeps the coordinate and **drops each of the four corners separately** when
 * that corner is off the map. At `y = -0.5` the two disagree completely: clamped, the
 * sample is read at rows 0 and 1; here the row above does not exist, contributes
 * nothing, and the row below carries half its weight.
 *
 * Written with a clamp it is right in the middle of the map and wrong along every
 * border — which is exactly where a deformable convolution spends its offsets.
 *
 * **Two group counts, and they are allowed to differ.** The weight's second axis
 * against the input's channels gives the weight groups; the offset's channel count
 * against `2 * kh * kw` gives the offset groups, and one offset field can steer several
 * groups of filters.
 *
 * **The kernel's own dilation is applied before the offset**, so a displacement of zero
 * leaves an ordinary dilated convolution. Added afterwards the offsets would mean
 * something different at every dilation.
 */
export async function deformConv2d(
  input: Tensor,
  offset: Tensor,
  weight: Tensor,
  bias: Tensor | null = null,
  stride: number | readonly number[] = 1,
  padding: number | readonly number[] = 0,
  dilation: number | readonly number[] = 1,
  mask: Tensor | null = null,
): Promise<Tensor> {
  const [strideH, strideW] = pairOfSizes(stride);
  const [padH, padW] = pairOfSizes(padding);
  const [dilH, dilW] = pairOfSizes(dilation);
  const f = await featureOf(input, "deformConv2d");
  const off = await featureOf(offset, "deformConv2d");
  const wShape = weight.shape as number[];
  const w = Array.from(await weight.toArray());
  const outChannels = wShape[0] ?? 0;
  const groupChannels = wShape[1] ?? 0;
  const kernelH = wShape[2] ?? 0;
  const kernelW = wShape[3] ?? 0;
  const offsetGroups = Math.floor(off.channels / (2 * kernelH * kernelW));
  if (offsetGroups === 0) {
    throw new Error(
      "the shape of the offset tensor at dimension 1 is not valid. It should be a "
      + "multiple of 2 * weight.size[2] * weight.size[3].\n"
      + `Got offset.shape[1]=${off.channels}, while 2 * weight.size[2] * `
      + `weight.size[3]=${2 * kernelH * kernelW}`);
  }
  const weightGroups = groupChannels === 0 ? 1 : Math.floor(f.channels / groupChannels);
  const outH = Math.floor(
    (f.height + 2 * padH - (dilH * (kernelH - 1) + 1)) / strideH) + 1;
  const outW = Math.floor(
    (f.width + 2 * padW - (dilW * (kernelW - 1) + 1)) / strideW) + 1;

  const m = mask === null ? null : await featureOf(mask, "deformConv2d");
  const b = bias === null ? null : Array.from(await bias.toArray());
  const perOffsetGroup = Math.floor(f.channels / offsetGroups);
  const perOut = Math.floor(outChannels / weightGroups);
  const out = new Array<number>(f.batch * outChannels * outH * outW).fill(0);

  for (let n = 0; n < f.batch; n++) {
    for (let i = 0; i < outH; i++) {
      for (let j = 0; j < outW; j++) {
        for (let p = 0; p < kernelH; p++) {
          for (let q = 0; q < kernelW; q++) {
            const which = p * kernelW + q;
            for (let g = 0; g < offsetGroups; g++) {
              const start = (g * kernelH * kernelW + which) * 2;
              const at = (ch: number): number =>
                off.values[((n * off.channels + ch) * off.height + i) * off.width + j]
                ?? 0;
              const y = i * strideH - padH + p * dilH + at(start);
              const x = j * strideW - padW + q * dilW + at(start + 1);
              const gain = m === null ? 1
                : m.values[((n * m.channels + g * kernelH * kernelW + which)
                            * m.height + i) * m.width + j] ?? 0;
              for (let k = 0; k < perOffsetGroup; k++) {
                const cin = g * perOffsetGroup + k;
                const taken = deformSample(f, n, cin, y, x) * gain;
                if (taken === 0) continue;
                // **The weight groups fold here**, not in the sampling: filter `co`
                // sees only its own group's channels. Letting every filter see every
                // channel gives the right shape from `weightGroups` times as many terms.
                const wg = groupChannels === 0 ? 0 : Math.floor(cin / groupChannels);
                const inGroup = cin - wg * groupChannels;
                for (let o = 0; o < perOut; o++) {
                  const co = wg * perOut + o;
                  const tap = w[((co * groupChannels + inGroup) * kernelH + p)
                                * kernelW + q] ?? 0;
                  const seat = ((n * outChannels + co) * outH + i) * outW + j;
                  out[seat] = (out[seat] ?? 0) + taken * tap;
                }
              }
            }
          }
        }
      }
    }
  }
  if (b !== null) {
    for (let n = 0; n < f.batch; n++) {
      for (let c = 0; c < outChannels; c++) {
        const add = b[c] ?? 0;
        for (let k = 0; k < outH * outW; k++) {
          out[(n * outChannels + c) * outH * outW + k] =
            (out[(n * outChannels + c) * outH * outW + k] ?? 0) + add;
        }
      }
    }
  }
  return Tensor.from(out, [f.batch, outChannels, outH, outW]);
}

/**
 * One value at a fractional `(y, x)`, **zero outside the map** — and each of the four
 * corners dropped on its own when that corner is off it.
 *
 * The outer guard is the compiled kernel's own: a coordinate at or past the edge by a
 * whole pixel contributes nothing at all, whichever corners would have been valid.
 */
function deformSample(f: Feature, image: number, channel: number,
                      y: number, x: number): number {
  if (y <= -1 || y >= f.height || x <= -1 || x >= f.width) return 0;
  const yLow = Math.floor(y);
  const xLow = Math.floor(x);
  const ly = y - yLow;
  const lx = x - xLow;
  const plane = (image * f.channels + channel) * f.height * f.width;
  const read = (r: number, c: number): number =>
    (r < 0 || r >= f.height || c < 0 || c >= f.width)
      ? 0
      : f.values[plane + r * f.width + c] ?? 0;
  return (1 - ly) * (1 - lx) * read(yLow, xLow)
    + (1 - ly) * lx * read(yLow, xLow + 1)
    + ly * (1 - lx) * read(yLow + 1, xLow)
    + ly * lx * read(yLow + 1, xLow + 1);
}

/**
 * Whole residual branches dropped at random.
 *
 * <https://arxiv.org/abs/1603.09382>
 *
 * **`mode` decides what a coin is tossed for**: `batch` drops the branch for the whole
 * batch at once, `row` drops it per example. The two differ only in the shape of the
 * noise, and getting it wrong gives a network that trains at a different effective
 * depth — nothing raises and the curve moves.
 *
 * **The survivors are divided by the survival rate**, so the expected value is
 * unchanged and evaluation needs no rescaling. Dropped without it, every layer is
 * quieter than the next one expects.
 *
 * The golden asks the two settings that draw nothing — `training = false` and `p = 0`.
 * What the coin does has no shared answer to compare against, which is a fact about
 * the comparison rather than about this function.
 */
export function stochasticDepth(
  input: Tensor, p: number, mode: "batch" | "row", training = true,
): Tensor {
  if (p < 0 || p > 1) {
    throw new Error(`drop probability has to be between 0 and 1, but got ${p}`);
  }
  if (mode !== "batch" && mode !== "row") {
    throw new Error(`mode has to be either 'batch' or 'row', but got ${mode}`);
  }
  if (!training || p === 0) return input;
  const survival = 1 - p;
  const shape = input.shape.map((d, i) => (mode === "row" && i === 0 ? d : 1));
  let noise = Tensor.full([...shape], survival).bernoulli();
  if (survival > 0) noise = noise.binary("div", Tensor.full([], survival));
  return input.mul(noise);
}

/**
 * DropBlock in `spatial` dimensions — **contiguous regions, not single pixels.**
 *
 * <https://arxiv.org/abs/1810.12890>
 *
 * Dropping pixels one at a time does little to a convolutional map, because the
 * neighbours carry the same information. So a seed is drawn per position and then
 * **grown to a block by a max pool**, which is the whole trick: the pool spreads each
 * surviving one over its window, and one minus that is the mask.
 *
 * **The seeds are drawn on a smaller grid than the map.** A block centred at the very
 * edge would hang off it, so the positions a seed may take are `size - block + 1` per
 * axis and the field is padded back out with zeros afterwards. Drawn at full size, the
 * rate is right and the blocks at the border are clipped, which drops less than asked.
 */
function dropBlock(
  input: Tensor, p: number, blockSize: number, inplace: boolean, eps: number,
  training: boolean, spatial: 2 | 3,
): Tensor {
  if (p < 0 || p > 1) {
    throw new Error(`drop probability has to be between 0 and 1, but got ${p}.`);
  }
  if (input.shape.length !== spatial + 2) {
    throw new Error(`input should be ${spatial + 2} dimensional. Got `
      + `${input.shape.length} dimensions.`);
  }
  if (!training || p === 0) return input;
  const sizes = input.shape.slice(2);
  const block = Math.min(blockSize, ...sizes);
  if (block % 2 === 0) {
    throw new Error(`block size should be odd. Got ${block} which is even.`);
  }
  const seeds = sizes.map((one) => one - block + 1);
  const total = sizes.reduce((a, b) => a * b, 1);
  const positions = seeds.reduce((a, b) => a * b, 1);
  const gamma = (p * total) / (block ** spatial * positions);
  const drawn = Tensor.full([input.shape[0] ?? 1, input.shape[1] ?? 1, ...seeds], gamma)
    .bernoulli();
  // **`pad` here is one axis at a time**, where torch takes the whole list. The spatial
  // axes are the last `spatial` of them, and the half-block goes on both ends of each.
  const half = Math.floor(block / 2);
  let field = drawn;
  for (let axis = 2; axis < 2 + spatial; axis++) {
    field = field.pad(axis, half, half, 0);
  }
  const grown = field.poolND("max", block, 1);
  const noise = Tensor.full([], 1).sub(grown);
  // **The scale is the survivors' share, and `eps` is on the sum rather than after
  // it.** Divided afterwards, a field that dropped everything gives infinity instead of
  // a very large number, and the difference shows only in the one case nobody writes.
  const scale = Tensor.full([], noise.numel()).binary(
    "div", noise.sum().binary("add", Tensor.full([], eps)));
  const out = input.mul(noise).binary("mul", scale);
  return inplace ? input.copyFrom(out) : out;
}

/** `dropBlock` over height and width.
 *
 * **`inplace` sits fourth, where torch puts it.** Written without that seat, `eps` took
 * it — so `drop_block2d(x, 0.3, 3, 1e-6)` set `inplace` to a number, which is truthy.
 * The signature axis said so in the run that added this function, which is what that
 * axis is: a count of positions rather than of names, and it caught the same defect
 * this file spent the day on an hour after it was written about.
 */
export function dropBlock2d(
  input: Tensor, p: number, blockSize: number, inplace = false, eps = 1e-6,
  training = true,
): Tensor {
  return dropBlock(input, p, blockSize, inplace, eps, training, 2);
}

/** `dropBlock` over depth, height and width. */
export function dropBlock3d(
  input: Tensor, p: number, blockSize: number, inplace = false, eps = 1e-6,
  training = true,
): Tensor {
  return dropBlock(input, p, blockSize, inplace, eps, training, 3);
}
