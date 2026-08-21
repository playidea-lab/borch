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

/** 머리 길이를 적는 자리. safetensors 가 정한 크기다. */
const LENGTH_FIELD = 8;

/**
 * 머리를 이 배수로 맞춰 몸이 8 바이트에서 시작하게 한다.
 *
 * 사양이 요구하지는 않지만 참조 구현이 공백으로 채워 맞춘다. 어긋나면 numpy 쪽에서
 * 몸을 그대로 매핑하지 못하고 복사가 한 번 더 든다.
 */
const ALIGN = 8;

/** borch 이름표를 싣는 열쇠의 앞머리. float32 인 것은 안 적는다 — 기본값이다. */
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
  // **이름 순서를 고정한다.** 객체의 열쇠 순서는 만든 차례를 따르므로, 같은 모델을
  // 두 번 저장하면 같은 바이트가 나와야 한다는 약속이 그것 하나에 걸린다.
  names.sort();

  const header: Record<string, Entry | Record<string, string>> = {};
  const meta: Record<string, string> = { ...metadata };
  const bodies: Float32Array[] = [];
  let offset = 0;

  for (const name of names) {
    const t = tensors[name] as Tensor;
    // **복소수는 아직 이 파일 형식에 안 들어간다.** 저장이 칸당 f32 두 개라
    // `shape` 와 몸 길이가 어긋나고, 그 상태는 저장은 되는데 **읽을 때** 터진다 —
    // 체크포인트가 몇 시간 뒤에 못 읽히는 것이 제일 나쁜 실패다.
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
  // 남는 자리는 공백이다 — JSON 파서가 뒤에 붙은 공백을 그냥 지난다.
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
    // **사본을 뜬다.** 원본 바이트 배열이 8 바이트 정렬이 아닐 수 있고, 그때
    // `new Float32Array(buffer, offset)` 은 그냥 던진다. 여기서 한 번 복사하면
    // 정렬 걱정이 사라지고, 어차피 GPU 로 올리며 또 복사한다.
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

// ── 중첩 ────────────────────────────────────────────────────────────────────
//
// **교재의 관용구는 중첩이다** — `{model: …, opt: …, epoch: 3}` 를 통째로 저장한다.
// 평평한 텐서 표만 되면 그 코드가 안 돈다.
//
// 파일 형식은 그대로다. 구조를 나무로 적어 `borch.tree` 라는 메타데이터 열쇠에 싣고,
// 텐서는 지금까지처럼 평평하게 눕힌다. **파이썬 `_serialize.py` 와 같은 스킴이고,
// 그것이 요점이다** — 두 벌로 두면 한쪽만 고쳐지고 그때 한쪽이 쓴 파일을 다른 쪽이
// 못 읽는다. 나무가 없는 파일(남이 만든 safetensors)은 평평한 표로 준다.

/** `borch.tree` — 구조를 적는 자리. 파이썬 쪽과 **같은 글자여야 한다.** */
const TREE_KEY = "borch.tree";

/** What this format can hold. It is not pickle, so not any object at all. */
export type Savable =
  | Tensor | null | boolean | number | string | Savable[] | { [key: string]: Savable };

/** 나무의 마디. `T`=텐서 · `d`=사전 · `l`=배열 · `j`=그냥 값. */
type Node =
  | { t: "T"; v: string }
  | { t: "d"; v: Record<string, Node> }
  | { t: "l"; v: Node[] }
  | { t: "j"; v: null | boolean | number | string };

// 파이썬 쪽에는 `u`(튜플)도 있다. **JS 에는 그 자리가 없다** — 배열 하나뿐이라
// 쓸 일이 없고, 읽을 때는 받아서 배열로 준다(아래). 안 그러면 파이썬이 쓴 파일에서
// 튜플이 든 것만 조용히 못 읽힌다.

function flatten(
  obj: Savable, path: string[], tensors: Record<string, Tensor>, seen: Set<string>,
): Node {
  if (obj instanceof Tensor) {
    const name = path.join(".") || "tensor";
    if (seen.has(name)) {
      // 서로 다른 자리가 같은 이름으로 펴졌다. 하나가 다른 하나를 덮으면 되돌릴 때
      // 두 자리가 같은 값을 갖는데, 그것은 예외보다 나쁘다.
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
  // 파이썬이 쓴 튜플(`u`)도 여기로 온다 — JS 에는 그 자리가 없어 배열로 준다.
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

/** 머리에 든 것이 우리가 아는 모양인가. 아니면 그 이름을 대고 던진다. */
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
    // **근사하지 않는다.** 남이 만든 F16·I64 파일을 float32 로 읽어 주면 값은
    // 나오는데 그 값이 무엇인지는 아무도 모른다.
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
