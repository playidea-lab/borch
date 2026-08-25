/**
 * The dataset **decoders**, and only those.
 *
 * A dataset is two things: an address and a format. The address half is not here —
 * fetching, caching and checksums belong to whatever is holding the network, and none
 * of it can be frozen in a comparison anyway, because a case that downloads is a case
 * that fails on a train.
 *
 * The format half is the part that goes wrong **quietly**, and every function in this
 * file is one of those. Each one is a conversion that leaves a dataset which still
 * trains when it is reversed:
 *
 * - STL10 is stored **column-major** and its labels start at **1**. Skip the transpose
 *   and every picture comes out rotated — still a picture, still trains.
 * - MovingMNIST's split cuts the **frames** and not the clips, so a reader that cuts
 *   clips instead produces the right rank and half the count.
 * - FER2013's pixels are a string of integers in a cell, and one of its two layouts
 *   puts **a leading space in the column names**.
 * - CIFAR's batch is planar — 1024 red, then 1024 green, then 1024 blue — and a
 *   channel-swapped CIFAR trains to a perfectly plausible number.
 *
 * So they take bytes and give tensors, with nothing else in the way.
 */

import { Tensor } from "./tensor.js";

// ── IDX ──────────────────────────────────────────────────────────────────────

/**
 * The type codes IDX uses. **Everything is big-endian**, which is the whole trap: a
 * QMNIST field of 279260 read the other way round is 469516288, and a label table wrong
 * in its later columns still trains, because the digit is column zero.
 */
const IDX_TYPES: Readonly<Record<number, { bytes: number; signed: boolean; float: boolean }>> = {
  8: { bytes: 1, signed: false, float: false },   // uint8
  9: { bytes: 1, signed: true, float: false },    // int8
  11: { bytes: 2, signed: true, float: false },   // int16
  12: { bytes: 4, signed: true, float: false },   // int32
  13: { bytes: 4, signed: false, float: true },   // float32
  14: { bytes: 8, signed: false, float: true },   // float64
};

/**
 * An IDX file's header and its numbers, unpacked.
 *
 * **The field is `numbers` and not `values` on purpose.** Called `values` it collided
 * with `Tensor.values` — torch's sparse-tensor method — and the name axis, which counts
 * declared names and cannot tell an interface property from a method, reported that
 * borch.ts had grown one. A name this library does not implement was being claimed by a
 * field on a plain object.
 */
export interface IdxFile {
  readonly shape: readonly number[];
  readonly numbers: Float64Array;
}

/**
 * The header is **two zero bytes, a type code, and the number of axes**, then one
 * big-endian length per axis. Sixteen bytes for a file that is otherwise eleven
 * megabytes, which is why the golden builds one here rather than downloading MNIST.
 */
export function readIdx(bytes: Uint8Array): IdxFile {
  if (bytes.length < 4) {
    throw new Error(`an IDX file is at least four bytes; this is ${bytes.length}.`);
  }
  const kind = IDX_TYPES[bytes[2] ?? 0];
  if (!kind) throw new Error(`unknown IDX type code ${bytes[2]}.`);
  const rank = bytes[3] ?? 0;
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const shape: number[] = [];
  for (let i = 0; i < rank; i++) shape.push(view.getUint32(4 + i * 4, false));
  const want = shape.reduce((a, b) => a * b, 1);
  const at = 4 + rank * 4;
  const have = Math.floor((bytes.length - at) / kind.bytes);
  // **The header promising more than the file carries is a case.** torchvision relaxes
  // an assert under `strict=False` and not the reshape underneath it, so both sides
  // still refuse — and this is worded to match, because the phrase is what the golden
  // compares.
  if (have < want) {
    throw new Error(`shape '[${want}]' is invalid for input of size ${have}`);
  }
  const values = new Float64Array(want);
  for (let i = 0; i < want; i++) {
    const off = at + i * kind.bytes;
    if (kind.float) {
      values[i] = kind.bytes === 4 ? view.getFloat32(off, false) : view.getFloat64(off, false);
    } else if (kind.bytes === 1) {
      values[i] = kind.signed ? view.getInt8(off) : view.getUint8(off);
    } else if (kind.bytes === 2) {
      values[i] = view.getInt16(off, false);
    } else {
      values[i] = view.getInt32(off, false);
    }
  }
  return { shape, numbers: values };
}

/** An IDX file as a tensor, whatever its rank. */
export function readIdxTensor(bytes: Uint8Array): Tensor {
  const { shape, numbers } = readIdx(bytes);
  return Tensor.from(numbers, shape);
}

/** MNIST's images — an IDX of rank three. */
export function readIdxImages(bytes: Uint8Array): Tensor {
  const got = readIdx(bytes);
  if (got.shape.length !== 3) {
    throw new Error(`image file should have rank 3, got ${got.shape.length}.`);
  }
  return Tensor.from(got.numbers, got.shape);
}

/** MNIST's labels — an IDX of rank one. */
export function readIdxLabels(bytes: Uint8Array): Tensor {
  const got = readIdx(bytes);
  if (got.shape.length !== 1) {
    throw new Error(`label file should have rank 1, got ${got.shape.length}.`);
  }
  return Tensor.from(got.numbers, got.shape);
}

// ── STL10 ────────────────────────────────────────────────────────────────────

/**
 * STL10's pictures, `(N, 3, H, W)`.
 *
 * **Stored column-major**, so the last two axes are swapped on the way out. Skipping
 * that gives every picture rotated a quarter turn — the shape is right, every summary
 * statistic is right, and the model trains. The golden freezes the un-swapped read
 * beside the swapped one so the difference is a value in the table rather than a claim
 * about one.
 */
export function readStl10Images(bytes: Uint8Array, side = 4, transpose = true): Tensor {
  const perPicture = 3 * side * side;
  const n = bytes.length / perPicture;
  if (!Number.isInteger(n)) {
    throw new Error(`${bytes.length} bytes is not a whole number of ${side}×${side} pictures.`);
  }
  const out = new Float64Array(bytes.length);
  for (let p = 0; p < n; p++) {
    for (let c = 0; c < 3; c++) {
      for (let i = 0; i < side; i++) {
        for (let j = 0; j < side; j++) {
          const from = ((p * 3 + c) * side + i) * side + j;
          const to = transpose
            ? ((p * 3 + c) * side + j) * side + i
            : from;
          out[to] = bytes[from] ?? 0;
        }
      }
    }
  }
  return Tensor.from(out, [n, 3, side, side]);
}

/** STL10's labels. **They start at 1 on disk**, and a model does not say so. */
export function readStl10Labels(bytes: Uint8Array): Tensor {
  return Tensor.from(Array.from(bytes, (v) => v - 1), [bytes.length]);
}

// ── .npy, for MovingMNIST ────────────────────────────────────────────────────

/** A `.npy` file's header and its numbers. */
export interface NpyFile {
  readonly shape: readonly number[];
  readonly numbers: Float64Array;
}

/**
 * numpy's own container: a magic string, a version, then **an ASCII dict** giving the
 * dtype, the order and the shape, and the raw block after it.
 *
 * Only what MovingMNIST ships is read — little-endian, C order, one of the plain
 * numeric dtypes. Fortran order is refused rather than read as C, because reading it as
 * C transposes the array and gives something that is still an array.
 */
export function readNpy(bytes: Uint8Array): NpyFile {
  const magic = String.fromCharCode(...bytes.slice(0, 6));
  if (magic !== "\x93NUMPY") throw new Error("not a .npy file — the magic is wrong.");
  const major = bytes[6] ?? 1;
  const headerLength = major === 1
    ? new DataView(bytes.buffer, bytes.byteOffset).getUint16(8, true)
    : new DataView(bytes.buffer, bytes.byteOffset).getUint32(8, true);
  const at = (major === 1 ? 10 : 12) + headerLength;
  const header = String.fromCharCode(...bytes.slice(major === 1 ? 10 : 12, at));
  if (/'fortran_order'\s*:\s*True/.test(header)) {
    throw new Error(".npy in Fortran order is not read here — it would transpose silently.");
  }
  const descr = /'descr'\s*:\s*'([^']+)'/.exec(header)?.[1] ?? "";
  const shape = [...(/'shape'\s*:\s*\(([^)]*)\)/.exec(header)?.[1] ?? "").matchAll(/\d+/g)]
    .map((m) => Number(m[0]));
  const want = shape.reduce((a, b) => a * b, 1);
  const view = new DataView(bytes.buffer, bytes.byteOffset + at, bytes.byteLength - at);
  const little = !descr.startsWith(">");
  const code = descr.replace(/^[<>|=]/, "");
  const values = new Float64Array(want);
  for (let i = 0; i < want; i++) {
    if (code === "u1") values[i] = view.getUint8(i);
    else if (code === "i1") values[i] = view.getInt8(i);
    else if (code === "u2") values[i] = view.getUint16(i * 2, little);
    else if (code === "i2") values[i] = view.getInt16(i * 2, little);
    else if (code === "u4") values[i] = view.getUint32(i * 4, little);
    else if (code === "i4") values[i] = view.getInt32(i * 4, little);
    else if (code === "f4") values[i] = view.getFloat32(i * 4, little);
    else if (code === "f8") values[i] = view.getFloat64(i * 8, little);
    else throw new Error(`.npy dtype '${descr}' is not read here.`);
  }
  return { shape, numbers: values };
}

/**
 * MovingMNIST, from the `.npy` it ships as.
 *
 * The file is `(frames, clips, H, W)` and what comes out is `(clips, frames, 1, H, W)`
 * — the axes swap and a channel appears.
 *
 * **The split cuts the frames, not the clips**, so both halves keep every clip. A
 * reader that cut the clips instead produces the right rank and half the count, which
 * is a dataset that loads, trains, and is half missing.
 */
export function readMovingMnist(bytes: Uint8Array, split: "train" | "test" | null = null,
                                ratio = 10): Tensor {
  const { shape, numbers } = readNpy(bytes);
  const [frames = 0, clips = 0, h = 0, w = 0] = shape;
  const from = split === "test" ? ratio : 0;
  const to = split === "train" ? ratio : frames;
  const kept = to - from;
  const out = new Float64Array(clips * kept * h * w);
  for (let c = 0; c < clips; c++) {
    for (let f = 0; f < kept; f++) {
      for (let i = 0; i < h * w; i++) {
        out[((c * kept + f) * h * w) + i] =
          numbers[(((from + f) * clips + c) * h * w) + i] ?? 0;
      }
    }
  }
  return Tensor.from(out, [clips, kept, 1, h, w]);
}

// ── FER2013 ──────────────────────────────────────────────────────────────────

/** One row: the 48×48 picture flattened, and the emotion. */
export interface Fer2013Row {
  readonly pixels: readonly number[];
  readonly emotion: number;
}

/**
 * FER2013's CSV, which is **two CSVs with the same name**.
 *
 * The `fer2013.csv` layout has an `emotion,pixels,Usage` header; the `icml_face_data.csv`
 * layout writes the same three names with **a leading space** on the second and third,
 * and a reader written against the other file finds out with a missing column rather
 * than a wrong number. Both are read here by trimming, which is what makes the two one
 * function.
 *
 * **`Usage` decides the split and `test` is two values**, `PublicTest` and
 * `PrivateTest`. A reader matching only the first returns half the set and no error.
 */
export function readFer2013(text: string, split: "train" | "test" = "train"): Fer2013Row[] {
  const lines = text.split("\n").filter((line) => line.length > 0);
  const header = (lines[0] ?? "").split(",").map((name) => name.trim());
  const emotionAt = header.indexOf("emotion");
  const pixelsAt = header.indexOf("pixels");
  const usageAt = header.indexOf("Usage");
  if (pixelsAt < 0) throw new Error(`no 'pixels' column in [${header.join(", ")}]`);
  const wanted = split === "train" ? ["Training"] : ["PublicTest", "PrivateTest"];
  const rows: Fer2013Row[] = [];
  for (const line of lines.slice(1)) {
    const cells = line.split(",").map((cell) => cell.trim());
    if (usageAt >= 0 && !wanted.includes(cells[usageAt] ?? "")) continue;
    rows.push({
      pixels: (cells[pixelsAt] ?? "").split(" ").filter((s) => s).map(Number),
      emotion: Number(cells[emotionAt] ?? 0),
    });
  }
  return rows;
}
