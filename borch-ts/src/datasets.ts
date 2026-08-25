/**
 * `torchvision.datasets` — **the decoders, and only those.**
 *
 * ## A dataset is an address and a format, and only one of them can cross
 *
 * The address half does not come to the browser: torchvision's own hosts send no
 * CORS header (`cs.toronto.edu` and `ossci-datasets.s3.amazonaws.com`, measured),
 * and the sites that do are somebody else's mirror. The Python side works around
 * that with `download=True` over a mirror list; here the page already has its own
 * fetch-and-cache path in `borch_webgpu` and the tutorials use it.
 *
 * The **format** half is the part that goes wrong quietly, and it is what this
 * file carries. An IDX header is sixteen bytes and a wrong read of it gives a
 * dataset that still trains — transposed, or off by a row, or with every label
 * shifted. That is the kind of defect a frozen comparison catches and nothing
 * else does.
 *
 * ## Bytes in, not a path
 *
 * torchvision's readers take a filename. There are no paths here, so these take
 * the bytes — which is what the caller has after a `fetch`, and what the Python
 * side's own `_read_idx` takes one layer under its public reader.
 *
 * ## What is not here, and why each one is not
 *
 * - **CIFAR's batches** — a batch is a *Python pickle* with a numpy array inside
 *   it. Reading one here means an opcode interpreter plus numpy's own
 *   reconstruction protocol, which is the opposite end of "no dependency we could
 *   not have written in an afternoon". The tutorials read CIFAR from plain binary
 *   instead, which is the same pictures without the format.
 * - **`FER2013` and the folder datasets** — their readers take a *directory*.
 *   There is no filesystem in a page, so what would be ported is not a decoder
 *   but a filesystem.
 */

import { RuntimeError } from "./errors.js";

/**
 * An array with its shape, which is what an IDX file holds.
 *
 * **The field is `data` and not `values`, and that is not a taste.** The name axis
 * asks whether borch.ts has a name *anywhere*, so an interface member called
 * `values` answers yes to a question about `Tensor.values` — the sparse-tensor
 * accessor the core carries only in order to refuse it. Named `values` this
 * interface silently turned a refusal into a feature in two separate measures;
 * `data` is a name `Image` already has, so it adds nothing to answer with.
 */
export interface IdxArray {
  readonly shape: readonly number[];
  readonly data: Float64Array;
}

// Type code → how many bytes each value takes and how to read it. **Big endian,
// always** — the format is Pascal Vincent's and it predates the machines that would
// have written it the other way.
const IDX_TYPES: Readonly<Record<number, string>> = {
  8: "u1", 9: "i1", 11: "i2", 12: "i4", 13: "f4", 14: "f8",
};

const IDX_WIDTH: Readonly<Record<string, number>> = {
  u1: 1, i1: 1, i2: 2, i4: 4, f4: 4, f8: 8,
};

/**
 * IDX bytes to an array.
 *
 * The header is four bytes — two zero, then the type, then the number of axes —
 * and then one big-endian 32-bit length per axis.
 *
 * **The lengths are not trusted to match the payload.** torchvision passes
 * `strict=False` from both its readers, so a file carrying more than the header
 * promises is truncated and one carrying less refuses. Both behaviours are kept,
 * because a divergence in what a malformed file does is the kind that shows up on
 * the day one arrives.
 */
export function readIdx(data: Uint8Array): IdxArray {
  const kind = data[2] as number;
  const axes = data[3] as number;
  const code = IDX_TYPES[kind];
  if (code === undefined) {
    throw new RuntimeError(
      `unknown IDX type code ${kind} — expected one of ` +
      `${Object.keys(IDX_TYPES).map(Number).sort((a, b) => a - b).join(", ")}`);
  }
  if (!(axes >= 1 && axes <= 3)) {
    throw new RuntimeError(`IDX header says ${axes} axes; 1 to 3 is the format`);
  }
  const view = new DataView(data.buffer, data.byteOffset, data.byteLength);
  const shape: number[] = [];
  for (let i = 0; i < axes; i++) shape.push(view.getUint32(4 * (i + 1), false));

  const width = IDX_WIDTH[code] as number;
  const start = 4 * (axes + 1);
  const carried = Math.floor((data.byteLength - start) / width);
  const want = shape.reduce((a, b) => a * b, 1);
  if (carried < want) {
    // **torch's sentence, because that is the one people search for.** Its readers
    // pass `strict=False`, which relaxes an assert and not the reshape underneath
    // it, so a short file refuses there too — with these words.
    throw new RuntimeError(
      `shape '[${shape.join(", ")}]' is invalid for input of size ${carried}\n` +
      `  the IDX header promises ${want} values and the file carries ${carried} ` +
      "— it is truncated.");
  }

  const out = new Float64Array(want);
  for (let i = 0; i < want; i++) {
    const at = start + i * width;
    switch (code) {
      case "u1": out[i] = view.getUint8(at); break;
      case "i1": out[i] = view.getInt8(at); break;
      case "i2": out[i] = view.getInt16(at, false); break;
      case "i4": out[i] = view.getInt32(at, false); break;
      case "f4": out[i] = view.getFloat32(at, false); break;
      default: out[i] = view.getFloat64(at, false); break;
    }
  }
  return { shape, data: out };
}

/** The three-axis half, checked. The messages are torchvision's. */
export function readImageFile(data: Uint8Array): IdxArray {
  if (data[2] !== 8) {
    throw new RuntimeError(
      `x should be of dtype uint8 instead of ${IDX_TYPES[data[2] as number] ?? "?"}`);
  }
  const out = readIdx(data);
  if (out.shape.length !== 3) {
    throw new RuntimeError(`x should have 3 dimensions instead of ${out.shape.length}`);
  }
  return out;
}

/**
 * The one-axis half. **Widened to int64**, as torchvision's `.long()` does — the
 * labels are bytes on disk and an index everywhere else.
 *
 * There is no int64 storage here, so the widening is a statement about what the
 * values mean rather than about how they are held; the numbers are the same either
 * way, and a label read as signed would turn 200 into -56.
 */
export function readLabelFile(data: Uint8Array): IdxArray {
  if (data[2] !== 8) {
    throw new RuntimeError(
      `x should be of dtype uint8 instead of ${IDX_TYPES[data[2] as number] ?? "?"}`);
  }
  const out = readIdx(data);
  if (out.shape.length !== 1) {
    throw new RuntimeError(`x should have 1 dimension instead of ${out.shape.length}`);
  }
  return out;
}
