/**
 * Checkpoint to bytes, bytes to checkpoint — the **safetensors** format.
 *
 * ## Why this is needed
 *
 * `stateDict()` existed, but **there was no way to save it.** Refreshing
 * the tab lost the training. A browser is a place where the process dies
 * far more often than on a desktop, so this gap was felt more sharply here
 * than it is in torch.
 *
 * ## Why no format of our own
 *
 * torch's `save`/`load` is pickle. It is a format that unpacks by executing
 * Python objects, which cannot be carried into a browser and should not be.
 *
 * safetensors is a JSON header and a body of contiguous bytes. **The codec
 * is this one file, and for that price Python `borch`, numpy and the HF
 * tools read the same file.** A private format loses that — training in the
 * browser and taking the result back to your own machine is half of this
 * project's story, and a format only we can read throws that half away.
 *
 * ```
 * [8 bytes LE u64: header length N][N bytes of JSON header][body: tensor bytes, concatenated]
 * ```
 *
 * ## Writing the dtype honestly
 *
 * borch's `int64` and `bool` are **labels; the values live in a float32
 * buffer** (`dtype.ts`). Writing `I64` in the header contradicts a
 * four-byte body and **breaks other people's readers.**
 *
 * So `dtype` is always written as `F32`, and borch's label rides separately
 * in `__metadata__`. Someone else reads a float32 array (which is true); we
 * read it and the label comes back too.
 */

import { RuntimeError } from "./errors.js";
import { DTYPES, type DType } from "./dtype.js";
import { Tensor } from "./tensor.js";

const BYTES_PER_F32 = 4;

/** Where the header length is written. The size safetensors fixed. */
const LENGTH_FIELD = 8;

/**
 * Pads the header to this multiple so the body starts on 8 bytes.
 *
 * The specification does not require it and the reference implementation pads with spaces
 * to match. Misaligned, the numpy side cannot map the body directly and it costs another
 * copy.
 */
const ALIGN = 8;

/** The prefix of the key carrying borch's labels. float32 is not written — it is the
 *  default. */
const DTYPE_KEY = "borch.dtype:";

export interface Bundle {
  tensors: Record<string, Tensor>;
  metadata: Record<string, string>;
}

interface Entry {
  dtype: string;
  shape: number[];
  data_offsets: [number, number];
}

/**
 * A checkpoint, as bytes.
 *
 * **It is async** — the values are on the GPU, so there is a round trip to
 * fetch them back. Hand it tensors already brought down to the host (`await
 * t.cpu()`) and that round trip is gone.
 */
export async function encode(
  tensors: Record<string, Tensor>,
  metadata: Record<string, string> = {},
): Promise<Uint8Array> {
  const names = Object.keys(tensors);
  // **The name order is fixed.** An object's key order follows creation order, so the
  // promise that saving one model twice gives the same bytes hangs on that alone.
  names.sort();

  const header: Record<string, Entry | Record<string, string>> = {};
  const meta: Record<string, string> = { ...metadata };
  const bodies: Float32Array[] = [];
  let offset = 0;

  for (const name of names) {
    const t = tensors[name] as Tensor;
    // **Complex does not go into this file format yet.** The storage is two f32 per
    // cell, so `shape` and the body length disagree, and in that state it saves and blows
    // up **on reading** — a checkpoint that cannot be read hours later is the worst
    // failure there is.
    if (t.dtype === "complex64") {
      throw new RuntimeError(
        `'${name}' is complex64 — saving that is not supported yet. ` +
          "Store the real pair with viewAsReal() and restore it with viewAsComplex() on load.",
      );
    }
    const values = await t.toArray();
    const bytes = values.length * BYTES_PER_F32;
    header[name] = {
      dtype: "F32",
      shape: [...t.shape],
      data_offsets: [offset, offset + bytes],
    };
    if (t.dtype !== "float32") meta[DTYPE_KEY + name] = t.dtype;
    bodies.push(values);
    offset += bytes;
  }

  if (Object.keys(meta).length > 0) header.__metadata__ = meta;

  const json = new TextEncoder().encode(JSON.stringify(header));
  const padding = (ALIGN - ((LENGTH_FIELD + json.length) % ALIGN)) % ALIGN;
  const headerLength = json.length + padding;

  const out = new Uint8Array(LENGTH_FIELD + headerLength + offset);
  new DataView(out.buffer).setBigUint64(0, BigInt(headerLength), true);
  out.set(json, LENGTH_FIELD);
  // The remainder is spaces — a JSON parser walks past trailing whitespace.
  out.fill(0x20, LENGTH_FIELD + json.length, LENGTH_FIELD + headerLength);

  let at = LENGTH_FIELD + headerLength;
  for (const body of bodies) {
    out.set(new Uint8Array(body.buffer, body.byteOffset, body.byteLength), at);
    at += body.byteLength;
  }
  return out;
}

/**
 * Bytes, as a checkpoint. **Synchronous** — uploading is one queued write.
 *
 * It will not quietly manufacture odd tensors out of a damaged file.
 * Lengths, offsets and types are all checked, and it throws when they do
 * not agree — a checkpoint that is quietly wrong robs every training run
 * after it of meaning.
 */
export function decode(bytes: Uint8Array): Bundle {
  if (bytes.byteLength < LENGTH_FIELD) {
    throw new RuntimeError(`checkpoint is too short: ${bytes.byteLength} bytes`);
  }
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const headerLength = Number(view.getBigUint64(0, true));
  const bodyAt = LENGTH_FIELD + headerLength;
  if (!Number.isSafeInteger(headerLength) || bodyAt > bytes.byteLength) {
    throw new RuntimeError(
      `header length runs past the file: ${headerLength} (file ${bytes.byteLength})`,
    );
  }

  const text = new TextDecoder().decode(
    bytes.subarray(LENGTH_FIELD, LENGTH_FIELD + headerLength),
  );
  let header: Record<string, unknown>;
  try {
    header = JSON.parse(text) as Record<string, unknown>;
  } catch {
    throw new RuntimeError("checkpoint header is not JSON");
  }

  const raw = header.__metadata__;
  const metadata: Record<string, string> = isStringMap(raw) ? raw : {};
  const tensors: Record<string, Tensor> = {};

  for (const [name, value] of Object.entries(header)) {
    if (name === "__metadata__") continue;
    const entry = asEntry(name, value);
    const [begin, end] = entry.data_offsets;
    if (begin > end || bodyAt + end > bytes.byteLength) {
      throw new RuntimeError(
        `'${name}' points past the end of the file: [${begin}, ${end}]`,
      );
    }
    const count = (end - begin) / BYTES_PER_F32;
    const size = entry.shape.reduce((a, b) => a * b, 1);
    if (count !== size) {
      throw new RuntimeError(
        `'${name}' has shape [${entry.shape}], which does not match ${count} elements`,
      );
    }
    // **A copy is taken.** The original byte array may not be 8-byte aligned, and then
    // `new Float32Array(buffer, offset)` simply throws. One copy here removes the
    // alignment worry, and it copies again on the way to the GPU regardless.
    const values = new Float32Array(
      bytes.slice(bodyAt + begin, bodyAt + end).buffer,
    );
    const label = metadata[DTYPE_KEY + name];
    tensors[name] = Tensor.from(values, entry.shape, {
      ...(label === undefined ? {} : { dtype: asDType(name, label) }),
    });
  }
  return { tensors, metadata };
}

// ── Nesting ────────────────────────────────────────────────────────────────
//
// **The textbook idiom is nested** — `{model: …, opt: …, epoch: 3}` saved whole. With a
// flat tensor table alone, that code does not run.
//
// The file format is unchanged. The structure is written as a tree and carried in a
// metadata key called `borch.tree`, and the tensors are laid out flat as before. **It is
// the same scheme as Python's `_serialize.py`, and that is the point** — two copies means
// one gets fixed and then a file one side wrote cannot be read by the other. A file with
// no tree (somebody else's safetensors) comes back as a flat table.

/** `borch.tree` — where the structure is written. It has to be **the same characters**
 *  as the Python side's. */
const TREE_KEY = "borch.tree";

/** What this format can hold. It is not pickle, so not any object at all. */
export type Savable =
  | Tensor | null | boolean | number | string | Savable[] | { [key: string]: Savable };

/** A node of the tree. `T`=tensor, `d`=dict, `l`=list, `j`=a plain value. */
type Node =
  | { t: "T"; v: string }
  | { t: "d"; v: Record<string, Node> }
  | { t: "l"; v: Node[] }
  | { t: "j"; v: null | boolean | number | string };

// The Python side also has `u` (tuple). **JS has no such slot** — there is only the
// array, so nothing writes one, and on reading it is accepted and handed back as an array
// (below). Otherwise a file written by Python fails, quietly and only where it contains a
// tuple.

function flatten(
  obj: Savable, path: string[], tensors: Record<string, Tensor>, seen: Set<string>,
): Node {
  if (obj instanceof Tensor) {
    const name = path.join(".") || "tensor";
    if (seen.has(name)) {
      // Two different places flattened to the same name. One overwriting the other gives
      // both places the same value on restore, and that is worse than an exception.
      throw new RuntimeError(
        `'${name}' appears twice — the flattened names collide and cannot be stored.`);
    }
    seen.add(name);
    tensors[name] = obj;
    return { t: "T", v: name };
  }
  if (Array.isArray(obj)) {
    return { t: "l", v: obj.map((v, i) => flatten(v, [...path, String(i)], tensors, seen)) };
  }
  if (obj === null || typeof obj === "boolean" || typeof obj === "number"
      || typeof obj === "string") {
    return { t: "j", v: obj };
  }
  if (typeof obj === "object") {
    const v: Record<string, Node> = {};
    for (const [k, child] of Object.entries(obj)) {
      v[k] = flatten(child, [...path, k], tensors, seen);
    }
    return { t: "d", v };
  }
  throw new RuntimeError(
    `${typeof obj} cannot be stored — only tensors, objects, arrays, numbers and strings.\n` +
    "This format is not pickle, so it cannot hold arbitrary objects.");
}

function unflatten(node: Node, tensors: Record<string, Tensor>): Savable {
  if (node.t === "T") {
    const t = tensors[node.v];
    if (!t) throw new RuntimeError(`the tree names '${node.v}', which the file does not hold`);
    return t;
  }
  if (node.t === "d") {
    const out: Record<string, Savable> = {};
    for (const [k, child] of Object.entries(node.v)) out[k] = unflatten(child, tensors);
    return out;
  }
  // A tuple written by Python (`u`) arrives here too — JS has no such slot, so it comes
  // back as an array.
  if (node.t === "l" || (node as { t: string }).t === "u") {
    return (node.v as Node[]).map((child) => unflatten(child, tensors));
  }
  return node.v;
}

/**
 * A checkpoint, as bytes — **it takes nesting as it is.** This is where
 * `torch.save` sits.
 *
 * ```ts
 * const bytes = await save({ model: m.stateDict(), opt: o.stateDict(), epoch: 3 });
 * ```
 *
 * **It is async** — the values are on the GPU, so there is a round trip to fetch
 * them back.
 *
 * Hand it a flat table of tensors and it behaves as it did before — that is one
 * case of nesting too. One tree rides along in the file, though, so **the bytes
 * differ from before.** Older files still read.
 */
export async function save(
  obj: Savable, metadata: Record<string, string> = {},
): Promise<Uint8Array> {
  const tensors: Record<string, Tensor> = {};
  const tree = flatten(obj, [], tensors, new Set());
  return encode(tensors, { ...metadata, [TREE_KEY]: JSON.stringify(tree) });
}

/**
 * Bytes, as a checkpoint — with the structure it was saved under. This is where
 * `torch.load` sits.
 *
 * **A file with no tree comes back as a flat table.** Someone else's safetensors is
 * like that, and so is a file borch.ts wrote before it had this layer.
 */
export function load(bytes: Uint8Array): Savable {
  const { tensors, metadata } = decode(bytes);
  const tree = metadata[TREE_KEY];
  if (tree === undefined) return tensors;
  let node: Node;
  try {
    node = JSON.parse(tree) as Node;
  } catch {
    throw new RuntimeError(`checkpoint has a '${TREE_KEY}' that is not JSON`);
  }
  return unflatten(node, tensors);
}

/** Whether what the header holds is a shape we know. Otherwise it throws, naming it. */
function asEntry(name: string, value: unknown): Entry {
  const e = value as Partial<Entry> | null;
  if (
    !e || typeof e !== "object"
    || !Array.isArray(e.shape) || !e.shape.every((n) => Number.isInteger(n))
    || !Array.isArray(e.data_offsets) || e.data_offsets.length !== 2
    || typeof e.data_offsets[0] !== "number" || typeof e.data_offsets[1] !== "number"
  ) {
    throw new RuntimeError(`'${name}' has a malformed header entry`);
  }
  if (e.dtype !== "F32") {
    // **It does not approximate.** Reading somebody else's F16 or I64 file as float32
    // produces values, and nobody knows what those values are.
    throw new RuntimeError(
      `'${name}' has dtype ${String(e.dtype)} — borch only reads F32`,
    );
  }
  return { dtype: "F32", shape: e.shape, data_offsets: [e.data_offsets[0], e.data_offsets[1]] };
}

function asDType(name: string, label: string): DType {
  if (!(DTYPES as readonly string[]).includes(label)) {
    throw new RuntimeError(`'${name}' has an unfamiliar dtype label: ${label}`);
  }
  return label as DType;
}

function isStringMap(v: unknown): v is Record<string, string> {
  return typeof v === "object" && v !== null
    && Object.values(v).every((x) => typeof x === "string");
}

/**
 * Puts a tag in front of each name. Used to hold a model and an optimizer
 * **in one file.**
 *
 * ```ts
 * const bytes = await encode({
 *   ...prefixed("model", model.stateDict()),
 *   ...prefixed("opt", opt.stateDict().tensors),
 * }, { ...numbersToMeta("opt", opt.stateDict().numbers) });
 * ```
 *
 * **This is a tool for using the flat codec directly.** `save` takes nesting as it
 * is, so no tag is needed — write `save({ model: …, opt: … })` and names cannot
 * collide.
 */
export function prefixed<T>(
  prefix: string, entries: Record<string, T>,
): Record<string, T> {
  const out: Record<string, T> = {};
  for (const [name, value] of Object.entries(entries)) out[`${prefix}.${name}`] = value;
  return out;
}

/**
 * Takes the tag off. Anything with a different prefix is not returned.
 */
export function unprefixed<T>(
  prefix: string, entries: Record<string, T>,
): Record<string, T> {
  const head = `${prefix}.`;
  const out: Record<string, T> = {};
  for (const [name, value] of Object.entries(entries)) {
    if (name.startsWith(head)) out[name.slice(head.length)] = value as T;
  }
  return out;
}

/**
 * Numbers as strings, so they can ride in the header. safetensors'
 * `__metadata__` takes strings only.
 *
 * **It writes with `JSON.stringify`.** `String(0.1)` is `"0.1"` and reading
 * that back gives the same double, but `Infinity` becomes `"Infinity"`,
 * which `JSON.parse` refuses — and `ReduceLROnPlateau`'s `best` starts at
 * exactly infinity.
 */
export function numbersToMeta(
  prefix: string, numbers: Record<string, number>,
): Record<string, string> {
  const out: Record<string, string> = {};
  for (const [name, value] of Object.entries(numbers)) {
    out[`${prefix}.${name}`] = Number.isFinite(value) ? String(value)
      : value > 0 ? "Infinity" : value < 0 ? "-Infinity" : "NaN";
  }
  return out;
}

/**
 * The undo of the above. Only entries whose prefix matches are returned.
 */
export function metaToNumbers(
  prefix: string, metadata: Record<string, string>,
): Record<string, number> {
  const head = `${prefix}.`;
  const out: Record<string, number> = {};
  for (const [name, text] of Object.entries(metadata)) {
    if (!name.startsWith(head)) continue;
    out[name.slice(head.length)] = text === "Infinity" ? Infinity
      : text === "-Infinity" ? -Infinity : Number(text);
  }
  return out;
}
