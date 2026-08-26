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
  const want = typeof size === "number" ? [size] : [...size];
  let newH: number;
  let newW: number;
  if (want.length === 1) {
    const short = Math.min(height, width);
    const long = Math.max(height, width);
    let newShort = want[0]!;
    let newLong = (long * newShort) / short;
    if (maxSize !== undefined && newLong > maxSize) {
      newShort = (newShort * maxSize) / newLong;
      newLong = maxSize;
    }
    [newH, newW] = height <= width ? [newShort, newLong] : [newLong, newShort];
    newH = Math.trunc(newH);
    newW = Math.trunc(newW);
  } else if (want.length === 2) {
    if (maxSize !== undefined) {
      throw new Error("maxSize is only used when size is a single number.");
    }
    newH = Math.trunc(want[0]!);
    newW = Math.trunc(want[1]!);
  } else {
    throw new Error(`size wants one or two numbers — got ${want.length}.`);
  }
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
