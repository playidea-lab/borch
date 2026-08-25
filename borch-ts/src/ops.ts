/**
 * Box geometry shaped like `torchvision.ops`.
 *
 * ## Eleven of thirty-nine, and which eleven
 *
 * The other twenty-eight are `nn.Module` layers and functions that need a
 * model's feature maps or its predictions, and there is no detector in this
 * catalogue. These eleven need **nothing but four numbers a box.** So unlike
 * almost everything in `vision.ts` they draw nothing, which is why the golden
 * can hold every one of them — there is no distribution half to this file.
 *
 * The same eleven are in the repository's Python side (`borchvision.ops`) and
 * both are held against the same frozen answers. If the two ever part, the
 * same boxes give different geometry per library, and that is caught only by
 * comparing values.
 *
 * ## Why this is a namespace and not more names in `vision`
 *
 * `vision.ts` carries `torchvision.transforms` with its names flattened to the
 * top, because transforms were all there was. **`ops` cannot join them there.**
 * `torchvision.ops` is a top-level namespace beside `transforms`, not under it,
 * so a flattened `vision.nms` would be a name torchvision does not have — and
 * inventing a place the imitated library has not got is the one move this
 * project cannot make. It is reached as `vision.ops.nms`, which is
 * `torchvision.ops.nms` with the library's own name in front.
 *
 * ## Boxes are arrays, not tensors
 *
 * torchvision takes and returns tensors here. This side takes `(N, 4)` numbers
 * and hands numbers back, for the reason `vision.ts` gives about images and one
 * more. A box is four numbers: one GPU buffer per box list buys nothing and
 * costs a round trip. And **reading a tensor back is asynchronous here**, so a
 * tensor-taking `boxIou` would have to be `await`ed — an `await` in front of a
 * question about rectangles that torch does not make you write.
 *
 * ```ts
 * import { vision } from "borch";
 * const keep = vision.ops.nms(boxes, scores, 0.5);
 * ```
 */

import { RuntimeError } from "./errors.js";

/** One box. The four numbers mean different things per format — see `BoxFormat`. */
export type Box = readonly number[];

/** `(N, 4)`. */
export type Boxes = readonly Box[];

/**
 * How to read a box's four numbers.
 *
 * **The three are indistinguishable by inspection** — four numbers either way —
 * so a wrong format is a wrong answer that raises nothing. It is an argument
 * rather than a guess for exactly that reason.
 *
 * - `xyxy` two corners · `xywh` a corner and a size · `cxcywh` a centre and a size
 */
export type BoxFormat = "xyxy" | "xywh" | "cxcywh";

const FORMATS: readonly BoxFormat[] = ["xyxy", "xywh", "cxcywh"];

function isFormat(fmt: string): fmt is BoxFormat {
  return (FORMATS as readonly string[]).includes(fmt);
}

/** `(N, 4)` of plain numbers, whatever kind of array arrived. */
function rows(boxes: Boxes, who: string): number[][] {
  const out: number[][] = [];
  for (const box of boxes) {
    if (box.length !== 4) {
      throw new RuntimeError(
        `${who} takes boxes of four numbers — one of them has ${box.length}.\n` +
        "(torch: Tensor shape is expected to be [N, 4])");
    }
    const [a = 0, b = 0, c = 0, d = 0] = box;
    out.push([a, b, c, d]);
  }
  return out;
}

function toXyxy(boxes: number[][], fmt: string): number[][] {
  if (!isFormat(fmt)) {
    throw new RuntimeError(
      `Unsupported Bounding Box format ${fmt} — it is one of ${FORMATS.join(", ")}.\n` +
      `(torch: Unsupported Bounding Box area for given format ${fmt})`);
  }
  if (fmt === "xyxy") return boxes;
  if (fmt === "xywh") {
    return boxes.map(([x = 0, y = 0, w = 0, h = 0]) => [x, y, x + w, y + h]);
  }
  return boxes.map(([cx = 0, cy = 0, w = 0, h = 0]) =>
    [cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h]);
}

/**
 * Between the three spellings of a box.
 *
 * **The identity is a copy and not the same array** — torchvision returns a new
 * tensor even when the formats match, and a caller who writes into the result
 * should not reach the caller's boxes.
 */
export function boxConvert(boxes: Boxes, inFmt: string, outFmt: string): number[][] {
  for (const [name, fmt] of [["in_fmt", inFmt], ["out_fmt", outFmt]] as const) {
    if (!isFormat(fmt)) {
      throw new RuntimeError(
        `Unsupported Bounding Box Conversions for given ${name} ${fmt}.\n` +
        "(torch: Unsupported Bounding Box Conversions for given in_fmt and out_fmt)");
    }
  }
  const xyxy = toXyxy(rows(boxes, "boxConvert"), inFmt);
  if (outFmt === "xyxy") return xyxy.map((box) => box.slice());
  return xyxy.map(([x1 = 0, y1 = 0, x2 = 0, y2 = 0]) => {
    const w = x2 - x1;
    const h = y2 - y1;
    return outFmt === "xywh" ? [x1, y1, w, h] : [x1 + 0.5 * w, y1 + 0.5 * h, w, h];
  });
}

/**
 * Width times height.
 *
 * **A box with `x2 < x1` gets a negative area** rather than zero — torchvision
 * does not clamp here, and clamping would hide a box built the wrong way round.
 */
export function boxArea(boxes: Boxes, fmt: string = "xyxy"): number[] {
  return toXyxy(rows(boxes, "boxArea"), fmt)
    .map(([x1 = 0, y1 = 0, x2 = 0, y2 = 0]) => (x2 - x1) * (y2 - y1));
}

/** Intersection and union of every box in `a` against every box in `b`. */
function interUnion(a: number[][], b: number[][]): { inter: number[][]; union: number[][] } {
  const areaOf = (box: number[]): number => {
    const [x1 = 0, y1 = 0, x2 = 0, y2 = 0] = box;
    return (x2 - x1) * (y2 - y1);
  };
  const inter: number[][] = [];
  const union: number[][] = [];
  for (const one of a) {
    const [ax1 = 0, ay1 = 0, ax2 = 0, ay2 = 0] = one;
    const interRow: number[] = [];
    const unionRow: number[] = [];
    for (const other of b) {
      const [bx1 = 0, by1 = 0, bx2 = 0, by2 = 0] = other;
      const w = Math.max(Math.min(ax2, bx2) - Math.max(ax1, bx1), 0);
      const h = Math.max(Math.min(ay2, by2) - Math.max(ay1, by1), 0);
      const overlap = w * h;
      interRow.push(overlap);
      unionRow.push(areaOf(one) + areaOf(other) - overlap);
    }
    inter.push(interRow);
    union.push(unionRow);
  }
  return { inter, union };
}

/** The smallest box enclosing both, as width and height. */
function enclosing(one: number[], other: number[]): { w: number; h: number } {
  const [ax1 = 0, ay1 = 0, ax2 = 0, ay2 = 0] = one;
  const [bx1 = 0, by1 = 0, bx2 = 0, by2 = 0] = other;
  return {
    w: Math.max(Math.max(ax2, bx2) - Math.min(ax1, bx1), 0),
    h: Math.max(Math.max(ay2, by2) - Math.min(ay1, by1), 0),
  };
}

/** Squared distance between the two centres. */
function centreDistance(one: number[], other: number[]): number {
  const [ax1 = 0, ay1 = 0, ax2 = 0, ay2 = 0] = one;
  const [bx1 = 0, by1 = 0, bx2 = 0, by2 = 0] = other;
  const dx = (ax1 + ax2) / 2 - (bx1 + bx2) / 2;
  const dy = (ay1 + ay2) / 2 - (by1 + by2) / 2;
  return dx * dx + dy * dy;
}

/**
 * **An `N x M` matrix, not a paired list.** Every box against every box, which
 * is what a detector needs and what surprises everyone the first time.
 */
export function boxIou(boxes1: Boxes, boxes2: Boxes, fmt: string = "xyxy"): number[][] {
  const a = toXyxy(rows(boxes1, "boxIou"), fmt);
  const b = toXyxy(rows(boxes2, "boxIou"), fmt);
  const { inter, union } = interUnion(a, b);
  return inter.map((row, i) => row.map((v, j) => v / (union[i]?.[j] ?? 1)));
}

/**
 * IoU, **minus what the smallest enclosing box wastes.** Two boxes that do not
 * touch have an IoU of 0 whatever the distance between them; this one keeps
 * falling to -1, which is why a loss can be built on it and not on IoU.
 */
export function generalizedBoxIou(boxes1: Boxes, boxes2: Boxes): number[][] {
  const a = rows(boxes1, "generalizedBoxIou");
  const b = rows(boxes2, "generalizedBoxIou");
  const { inter, union } = interUnion(a, b);
  return a.map((one, i) => b.map((other, j) => {
    const u = union[i]?.[j] ?? 0;
    const iou = (inter[i]?.[j] ?? 0) / u;
    const { w, h } = enclosing(one, other);
    const area = w * h;
    return iou - (area - u) / area;
  }));
}

/**
 * IoU penalised by **how far apart the centres are**, as a fraction of the
 * enclosing box's diagonal.
 */
export function distanceBoxIou(boxes1: Boxes, boxes2: Boxes, eps = 1e-7): number[][] {
  const a = rows(boxes1, "distanceBoxIou");
  const b = rows(boxes2, "distanceBoxIou");
  const { inter, union } = interUnion(a, b);
  return a.map((one, i) => b.map((other, j) => {
    const iou = (inter[i]?.[j] ?? 0) / (union[i]?.[j] ?? 1);
    const { w, h } = enclosing(one, other);
    return iou - centreDistance(one, other) / (w * w + h * h + eps);
  }));
}

/**
 * `distanceBoxIou` and **one more term for the aspect ratio** — two boxes with
 * the same centre and area but different shapes score lower here and
 * identically under the distance one.
 */
export function completeBoxIou(boxes1: Boxes, boxes2: Boxes, eps = 1e-7): number[][] {
  const a = rows(boxes1, "completeBoxIou");
  const b = rows(boxes2, "completeBoxIou");
  const { inter, union } = interUnion(a, b);
  return a.map((one, i) => b.map((other, j) => {
    const iou = (inter[i]?.[j] ?? 0) / (union[i]?.[j] ?? 1);
    const { w, h } = enclosing(one, other);
    const diou = iou - centreDistance(one, other) / (w * w + h * h + eps);
    const [ax1 = 0, ay1 = 0, ax2 = 0, ay2 = 0] = one;
    const [bx1 = 0, by1 = 0, bx2 = 0, by2 = 0] = other;
    const angle = Math.atan((bx2 - bx1) / (by2 - by1)) - Math.atan((ax2 - ax1) / (ay2 - ay1));
    const v = (4 / (Math.PI * Math.PI)) * angle * angle;
    // **The division is left to produce what it produces.** Two identical boxes
    // give `1 - iou + v` of zero, and numpy answers `nan` there with the invalid
    // warning silenced rather than raising — so a guard here would be this side
    // inventing a number the other side does not have.
    const alpha = v / (1 - iou + v + eps);
    return diou - alpha * v;
  }));
}

/**
 * Push every corner back inside a picture of `size`, which is **(height,
 * width)** — the opposite order to a box's own `(x, y)`, and torchvision's own
 * convention.
 */
export function clipBoxesToImage(boxes: Boxes, size: readonly number[]): number[][] {
  const [height = 0, width = 0] = size;
  return rows(boxes, "clipBoxesToImage").map(([x1 = 0, y1 = 0, x2 = 0, y2 = 0]) => [
    Math.min(Math.max(x1, 0), width), Math.min(Math.max(y1, 0), height),
    Math.min(Math.max(x2, 0), width), Math.min(Math.max(y2, 0), height),
  ]);
}

/**
 * **Indices, not boxes.** Every one of these that filters returns the positions
 * rather than the survivors, because the caller almost always has scores and
 * labels to filter by the same positions.
 */
export function removeSmallBoxes(boxes: Boxes, minSize: number): number[] {
  const keep: number[] = [];
  rows(boxes, "removeSmallBoxes").forEach(([x1 = 0, y1 = 0, x2 = 0, y2 = 0], i) => {
    if (x2 - x1 >= minSize && y2 - y1 >= minSize) keep.push(i);
  });
  return keep;
}

/**
 * The tightest box around each mask, which is `(N, H, W)`.
 *
 * **An empty mask gives all zeros** rather than an error — torchvision's
 * behaviour, and the one that lets a batch with a blank mask in it still stack.
 */
export function masksToBoxes(masks: readonly (readonly number[][])[]): number[][] {
  return masks.map((mask) => {
    let x1 = Infinity;
    let y1 = Infinity;
    let x2 = -Infinity;
    let y2 = -Infinity;
    mask.forEach((row, y) => row.forEach((v, x) => {
      if (!v) return;
      if (x < x1) x1 = x;
      if (x > x2) x2 = x;
      if (y < y1) y1 = y;
      if (y > y2) y2 = y;
    }));
    return Number.isFinite(x1) ? [x1, y1, x2, y2] : [0, 0, 0, 0];
  });
}

/**
 * Non-maximum suppression: keep the best-scoring box, throw away everything
 * that overlaps it too much, repeat.
 *
 * **`> iouThreshold` and not `>=`.** At a threshold of 0 two boxes that merely
 * touch — zero overlap — both survive, and that is the boundary anybody testing
 * this reaches for first.
 *
 * Ties in the score are **not decided here and torchvision does not decide them
 * either**; its own documentation says the choice is not guaranteed to match
 * between CPU and GPU. So a case built on tied scores is a case with no answer.
 */
export function nms(boxes: Boxes, scores: readonly number[], iouThreshold: number): number[] {
  const arr = rows(boxes, "nms");
  // Descending by score. **The sort has to be stable** — JavaScript's has been
  // since ES2019 and numpy's `argsort` is asked for it by name, so equal scores
  // keep the order they arrived in on both sides.
  let order = arr.map((_, i) => i)
    .sort((i, j) => (scores[j] ?? 0) - (scores[i] ?? 0));
  const kept: number[] = [];
  while (order.length) {
    const best = order[0] as number;
    kept.push(best);
    if (order.length === 1) break;
    const rest = order.slice(1);
    const overlap = boxIou([arr[best] as number[]], rest.map((i) => arr[i] as number[]))[0] ?? [];
    order = rest.filter((_, k) => (overlap[k] ?? 0) <= iouThreshold);
  }
  return kept;
}

/**
 * NMS **per class**, done by moving each class's boxes somewhere the others
 * cannot reach.
 *
 * The offset trick is torchvision's and it is worth reading twice: every box is
 * shifted by its class index times more than the largest coordinate, so boxes
 * of different classes can no longer overlap and a single pass of `nms` does
 * the lot.
 */
export function batchedNms(boxes: Boxes, scores: readonly number[],
                           idxs: readonly number[], iouThreshold: number): number[] {
  const arr = rows(boxes, "batchedNms");
  if (!arr.length) return [];
  const largest = Math.max(...arr.map((box) => Math.max(...box)));
  const moved = arr.map((box, i) => {
    const offset = (idxs[i] ?? 0) * (largest + 1);
    return box.map((v) => v + offset);
  });
  return nms(moved, scores, iouThreshold);
}
