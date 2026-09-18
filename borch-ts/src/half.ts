/**
 * **Host f32 ↔ f16 bit conversion** — for storing frozen weights in a window at half the
 * bytes (`docs/SCALE.md` Step 4, folded into Step 3). This is not a `float16` dtype: the
 * bytes are packed IEEE half-precision, held as raw window bytes, and unpacked to f32 on
 * the GPU (`unpackHalf`) when a weight funnel reads them. The core axiom "storage is
 * float32" is untouched — a window buffer is outside the Tensor storage model (ADR-003
 * decision 5). WGSL's `pack2x16float` would do the same on the GPU; packing on the host
 * keeps the place path (staging → copyRange) as it is, with no temporary GPU buffers.
 *
 * The conversion is the standard round-to-nearest-even one (Fabian Giesen's), handling
 * subnormals, overflow to inf, and NaN — the same rounding `pack2x16float` does, so a
 * host-packed weight and a GPU-unpacked read agree (checked in `window_probe`).
 */

const _f32 = new Float32Array(1);
const _i32 = new Int32Array(_f32.buffer);

/** One f32 value to its IEEE half-precision bit pattern (a `u16`). */
export function f32ToF16Bit(val: number): number {
  _f32[0] = val;
  const x = _i32[0] ?? 0;
  let bits = (x >> 16) & 0x8000;          // sign
  let m = (x >> 12) & 0x07ff;             // mantissa with the round bit
  const e = (x >> 23) & 0xff;             // exponent
  if (e < 103) return bits;               // too small — flushes to signed zero
  if (e > 142) {                          // overflow to inf, or NaN
    bits |= 0x7c00;
    // A NaN must stay a NaN (non-zero mantissa), not become inf.
    bits |= (e === 255 ? 0 : 1) && (x & 0x007fffff) ? 0x0200 : 0;
    return bits;
  }
  if (e < 113) {                          // subnormal half
    m |= 0x0800;
    bits |= (m >> (114 - e)) + ((m >> (113 - e)) & 1);
    return bits;
  }
  bits |= ((e - 112) << 10) | (m >> 1);
  bits += m & 1;                          // round to nearest even
  return bits & 0xffff;
}

/** An IEEE half-precision bit pattern (a `u16`) to its f32 value. */
export function f16BitToF32(h: number): number {
  const s = (h & 0x8000) >> 15;
  const e = (h & 0x7c00) >> 10;
  const f = h & 0x03ff;
  const sign = s ? -1 : 1;
  if (e === 0) return sign * 2 ** -14 * (f / 1024);
  if (e === 0x1f) return f ? NaN : sign * Infinity;
  return sign * 2 ** (e - 15) * (1 + f / 1024);
}

/** Pack an f32 array to half-precision bits. Length is preserved; each value is 2 bytes. */
export function f32ToF16Bits(a: Float32Array): Uint16Array {
  const out = new Uint16Array(a.length);
  for (let i = 0; i < a.length; i++) out[i] = f32ToF16Bit(a[i] ?? 0);
  return out;
}

/** Unpack half-precision bits back to an f32 array — the host mirror of `unpackHalf`. */
export function f16BitsToF32(a: Uint16Array): Float32Array {
  const out = new Float32Array(a.length);
  for (let i = 0; i < a.length; i++) out[i] = f16BitToF32(a[i] ?? 0);
  return out;
}
