/**
 * **Host per-channel int8 quantisation** — for storing a frozen weight in a window at a quarter
 * of the bytes (`docs/SCALE.md` Step 5). Like `half.ts`, this is **not** a dtype: the bytes are
 * packed int8, held as raw window bytes (four per `u32`), and turned back into f32 on the GPU by
 * the matmul's int8 read (`weightInt8`) or a dequant pass — the core axiom "storage is float32"
 * is untouched (ADR-003 decision 5).
 *
 * **Symmetric, per output channel.** A weight `[out, in]` gets one scale per `out` row:
 * `scale[o] = max|W[o,:]| / 127`, and `q[o,i] = round(W[o,i] / scale[o])` clamped to `[-127, 127]`.
 * Per-channel (rather than one scale for the whole tensor) is what keeps top-1 within a point of
 * f32 — a single outlier row would otherwise crush every other row's resolution. The values pack
 * in the weight's own row-major order, so element `o*in + i` is byte `(o*in + i) % 4` of word
 * `(o*in + i) / 4` — the same index the matmul reads with `transB` (`bcol*K + brow`, `bcol = o`).
 */

const I8_MAX = 127;

/**
 * Quantises `data` (a row-major `[outChannels, inPerChannel]` weight) to per-channel int8,
 * packed four to a `u32`. Returns the packed words (padded so the byte count is a whole number
 * of words) and one f32 scale per output channel. A zero row gets scale 0 and quantises to zeros.
 */
export function quantizeInt8PerChannel(
  data: Float32Array,
  outChannels: number,
): { packed: Uint32Array; scales: Float32Array } {
  const total = data.length;
  const inPer = outChannels > 0 ? total / outChannels : 0;
  const scales = new Float32Array(outChannels);
  const q = new Int8Array(total);
  for (let o = 0; o < outChannels; o++) {
    let peak = 0;
    const base = o * inPer;
    for (let i = 0; i < inPer; i++) { const a = Math.abs(data[base + i] as number); if (a > peak) peak = a; }
    const scale = peak / I8_MAX;
    scales[o] = scale;
    if (scale === 0) continue;                       // a zero row stays zero
    const inv = 1 / scale;
    for (let i = 0; i < inPer; i++) {
      const r = Math.round((data[base + i] as number) * inv);
      q[base + i] = r > I8_MAX ? I8_MAX : r < -I8_MAX ? -I8_MAX : r;
    }
  }
  // Pack four signed bytes per word, low byte first — the order `unpack` reads back.
  const words = Math.ceil(total / 4);
  const packed = new Uint32Array(words);
  const bytes = new Uint8Array(q.buffer);            // reinterpret int8 as the raw bytes
  for (let w = 0; w < words; w++) {
    const b0 = bytes[w * 4] ?? 0, b1 = bytes[w * 4 + 1] ?? 0, b2 = bytes[w * 4 + 2] ?? 0, b3 = bytes[w * 4 + 3] ?? 0;
    packed[w] = (b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)) >>> 0;
  }
  return { packed, scales };
}

/**
 * The exact f32 values a per-channel int8 window weight holds — `quantize` then reconstruct
 * (`q * scale`). The reference for a test: feeding these to an f32 matmul matches the int8
 * matmul's own reconstruction bit for bit.
 */
export function dequantInt8PerChannel(data: Float32Array, outChannels: number): Float32Array {
  const { packed, scales } = quantizeInt8PerChannel(data, outChannels);
  const inPer = outChannels > 0 ? data.length / outChannels : data.length;
  const out = new Float32Array(data.length);
  for (let e = 0; e < data.length; e++) {
    const word = packed[e >> 2] as number;
    const byte = (word >>> ((e & 3) * 8)) & 0xff;
    const q = byte > 127 ? byte - 256 : byte;          // sign-extend
    out[e] = q * (scales[Math.floor(e / inPer)] as number);
  }
  return out;
}
