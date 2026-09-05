/**
 * Reads a safetensors file into host arrays — without a device.
 *
 * `serialize.decode` does the same job and hands back `Tensor`s, which is right for the
 * WebGPU side and impossible here: on the machine this device exists for there is no
 * adapter, so there is no `Tensor` to hand back. This reader stops at `Float32Array`.
 *
 * The format is small enough to read in full: eight bytes of little-endian header
 * length, a JSON header naming each tensor's dtype, shape and byte range, then the
 * bytes. Only `F32` is accepted — the hub's checkpoints are F32 throughout, and a
 * silently widened `F16` would be a different model under the same name.
 */

export interface HostTensor {
  readonly shape: readonly number[];
  readonly data: Float32Array;
}

export interface HostStateDict {
  readonly tensors: ReadonlyMap<string, HostTensor>;
  readonly metadata: Readonly<Record<string, string>>;
}

interface Entry {
  readonly dtype: string;
  readonly shape: readonly number[];
  readonly data_offsets: readonly [number, number];
}

function isEntry(v: unknown): v is Entry {
  if (typeof v !== "object" || v === null) return false;
  const o = v as Record<string, unknown>;
  return typeof o["dtype"] === "string" && Array.isArray(o["shape"]) && Array.isArray(o["data_offsets"]) && o["data_offsets"].length === 2;
}

const LENGTH_FIELD = 8;

export function readSafetensors(bytes: Uint8Array): HostStateDict {
  if (bytes.byteLength < LENGTH_FIELD) throw new Error(`safetensors: ${bytes.byteLength} bytes is too short for a header`);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const headerLength = Number(view.getBigUint64(0, true));
  const bodyAt = LENGTH_FIELD + headerLength;
  if (!Number.isSafeInteger(headerLength) || bodyAt > bytes.byteLength) throw new Error(`safetensors: header length ${headerLength} runs past the file`);
  const parsed: unknown = JSON.parse(new TextDecoder().decode(bytes.subarray(LENGTH_FIELD, bodyAt)));
  if (typeof parsed !== "object" || parsed === null) throw new Error("safetensors: the header is not an object");
  const header = parsed as Record<string, unknown>;
  const tensors = new Map<string, HostTensor>();
  let metadata: Record<string, string> = {};
  for (const [name, value] of Object.entries(header)) {
    if (name === "__metadata__") {
      if (typeof value === "object" && value !== null) {
        metadata = Object.fromEntries(Object.entries(value as Record<string, unknown>).filter((kv): kv is [string, string] => typeof kv[1] === "string"));
      }
      continue;
    }
    if (!isEntry(value)) throw new Error(`safetensors: "${name}" has no dtype/shape/data_offsets`);
    if (value.dtype !== "F32") throw new Error(`safetensors: "${name}" is ${value.dtype}; this reader takes F32 only`);
    const [begin, end] = value.data_offsets;
    const count = value.shape.reduce((a, d) => a * d, 1);
    if (end - begin !== count * 4 || bodyAt + end > bytes.byteLength) throw new Error(`safetensors: "${name}" byte range does not match its shape`);
    // The body is rarely 4-byte aligned relative to the buffer, and a Float32Array view
    // needs it to be — so copy. The weights are copied once more into the module's memory
    // anyway; this is not the copy that matters.
    const data = new Float32Array(count);
    const src = bytes.subarray(bodyAt + begin, bodyAt + end);
    new Uint8Array(data.buffer).set(src);
    tensors.set(name, { shape: [...value.shape], data });
  }
  return { tensors, metadata };
}
