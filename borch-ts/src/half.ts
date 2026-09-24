/**
 * **Host f32 ↔ f16 bit conversion** — for storing frozen weights in a window at half the
 * bytes (`docs/SCALE.md` Step 4, folded into Step 3). This is not a `float16` dtype: the
 * bytes are packed IEEE half-precision, held as raw window bytes, and unpacked to f32 on
 * the GPU (`unpackHalf`) when a weight funnel reads them. The core axiom "storage is
 * float32" is untouched — a window buffer is outside the Tensor storage model (ADR-003
 * decision 5). WGSL's `pack2x16float` would do the same on the GPU; packing on the host
 * keeps the place path (staging → copyRange) as it is, with no temporary GPU buffers.
 *
 * The conversion rounds to nearest, ties to even, with subnormals, overflow to inf and
 * NaN kept — the same rounding `pack2x16float` does, so a host-packed weight and a
 * GPU-unpacked read agree (checked in `window_probe`, and against `Float16Array` in
 * `tests/test_half.py`).
 */

const _f32 = new Float32Array(1);
const _i32 = new Int32Array(_f32.buffer);

/** One f32 value to its IEEE half-precision bit pattern (a `u16`), rounded to nearest,
 *  ties to even — what `pack2x16float` and `Float16Array` do.
 *
 *  The version before this (2026-09-24 review) had NaN and overflow the wrong way round —
 *  a NaN packed as inf, and a finite value past 65504 with mantissa bits as NaN — and
 *  rounded halves up, so `1 + 2⁻¹¹` came out one step high. It is now held bit for bit
 *  against `Float16Array` (`tests/test_half.py`). */
export function f32ToF16Bit(val: number): number {
  _f32[0] = val;
  const x = _i32[0] ?? 0;
  const sign = (x >>> 16) & 0x8000;
  const exp = (x >>> 23) & 0xff;
  const mant = x & 0x7fffff;
  // NaN stays NaN (the quiet bit set, so no payload can clear it); inf stays inf.
  if (exp === 0xff) return sign | 0x7c00 | (mant !== 0 ? 0x0200 | (mant >>> 13) : 0);
  const e = exp - 112;                    // the half's biased exponent
  if (e >= 0x1f) return sign | 0x7c00;    // past the largest half — inf
  if (e <= 0) {
    // A subnormal half, or zero: below 2⁻²⁵ every value rounds to zero.
    if (e < -10) return sign;
    const m = mant | 0x800000;             // the implicit bit made explicit
    const shift = 14 - e;
    let h = m >>> shift;
    const rest = m & ((1 << shift) - 1), half = 1 << (shift - 1);
    if (rest > half || (rest === half && (h & 1) === 1)) h += 1;
    return sign | h;                       // a carry into 0x400 is the smallest normal — right
  }
  let h = (e << 10) | (mant >>> 13);
  const rest = mant & 0x1fff;
  if (rest > 0x1000 || (rest === 0x1000 && (h & 1) === 1)) h += 1;   // a carry past 0x7bff is inf — right
  return sign | h;
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
