//! The CPU kernels behind borch-ts's `cpu` device.
//!
//! ## What this is
//!
//! WebAssembly SIMD implementations of the handful of operations a model forward and a
//! head's training actually spend time in: GEMM (every 1×1 convolution and linear layer),
//! depthwise convolution, and the elementwise passes around them. `borch-ts/src/cpu/`
//! is the TypeScript side that owns tensors and calls these on offsets into the one
//! linear memory.
//!
//! ## Why a wasm module, and why it has no runtime
//!
//! Measured on 2026-09-05, on an Apple M4 Max, one thread: the same EfficientNet-B0
//! forward took 20 ms an image through these kernels and 520–565 ms through
//! SwiftShader, WebGPU's own CPU implementation — and SwiftShader compiles every shader
//! for every new shape first, 64 s for that network. A hand-written kernel has no
//! compile step and no dispatch model to imitate. Plain JavaScript on typed arrays sat
//! at SwiftShader's speed (2.3–2.6 GFLOPS), so the speed is the SIMD, not the language.
//!
//! There is no allocator library, no `wasm-bindgen`, no standard library. The module
//! exports functions over its own memory and a bump allocator, and the TypeScript side
//! does the rest. Two reasons: a second runtime would be a second heap beside Pyodide's
//! when the Python binding runs on this device, and every byte of this file ends up
//! base64 in `kernels.ts`, so the module is kept to a few kilobytes on purpose.
//!
//! ## Layout contract with the caller
//!
//! Row-major `f32`, activations NHWC with the batch folded into rows. The caller pads:
//! `gemm` wants `m % 4 == 0` and `n % 16 == 0`; the channel-wise kernels want
//! `c % 16 == 0` (`dwconv`) or `c % 4 == 0` (the rest). Padding columns are zero in the
//! weights and therefore stay zero in every activation. `tests/test_cpu_kernels.py`
//! holds the module's shape (no imports, a few kilobytes, exactly the exports the loader
//! asks for); the values are the `cpu` device's job, checked against the WebGPU device
//! when that device arrives.
//!
//! The comments are in English because this repository is public; the reasons live
//! beside the code, as everywhere else here.
#![no_std]

use core::arch::wasm32::*;

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    core::arch::wasm32::unreachable()
}

const PAGE: usize = 65536;
static mut BASE: usize = 0;
static mut HEAP: usize = 0;

/// Bump allocation, 16-byte aligned, starting after the static data. Grows the memory
/// when it has to; returns 0 when the engine refuses to grow.
#[no_mangle]
pub unsafe extern "C" fn alloc(bytes: usize) -> usize {
    if BASE == 0 {
        BASE = memory_size::<0>() * PAGE;
        HEAP = BASE;
    }
    let p = (HEAP + 15) & !15;
    let end = p + bytes;
    let cur = memory_size::<0>() * PAGE;
    if end > cur {
        let pages = (end - cur + PAGE - 1) / PAGE;
        if memory_grow::<0>(pages) == usize::MAX {
            return 0;
        }
    }
    HEAP = end;
    p
}

/// Forget every allocation. The caller runs `reset → alloc → compute` per network so
/// there is nothing to fragment.
#[no_mangle]
pub unsafe extern "C" fn reset() {
    HEAP = BASE;
}

/// Where the heap stands. Paired with `set_heap` to drop a block's temporaries while
/// keeping its output.
#[no_mangle]
pub unsafe extern "C" fn heap() -> usize {
    HEAP
}

#[no_mangle]
pub unsafe extern "C" fn set_heap(pos: usize) {
    HEAP = pos;
}

/// C[m×n] = A[m×k] · B[k×n]. Register block of 4 rows × 16 columns (four v128 per row),
/// so one pass over `k` does 64 multiply-adds from 4 loads of B and 4 splats of A.
/// A 1×1 convolution in NHWC is exactly this call: A the activations `[HW × Cin]`,
/// B the weights `[Cin × Cout]`.
#[no_mangle]
pub unsafe extern "C" fn gemm(m: usize, n: usize, k: usize, a: *const f32, b: *const f32, c: *mut f32) {
    let mut i = 0;
    while i < m {
        let a0 = a.add(i * k);
        let a1 = a0.add(k);
        let a2 = a1.add(k);
        let a3 = a2.add(k);
        let mut j = 0;
        while j < n {
            let mut c00 = f32x4_splat(0.0); let mut c01 = c00; let mut c02 = c00; let mut c03 = c00;
            let mut c10 = c00; let mut c11 = c00; let mut c12 = c00; let mut c13 = c00;
            let mut c20 = c00; let mut c21 = c00; let mut c22 = c00; let mut c23 = c00;
            let mut c30 = c00; let mut c31 = c00; let mut c32 = c00; let mut c33 = c00;
            let mut p = 0;
            while p < k {
                let bp = b.add(p * n + j);
                let b0 = v128_load(bp as *const v128);
                let b1 = v128_load(bp.add(4) as *const v128);
                let b2 = v128_load(bp.add(8) as *const v128);
                let b3 = v128_load(bp.add(12) as *const v128);
                let s0 = f32x4_splat(*a0.add(p));
                let s1 = f32x4_splat(*a1.add(p));
                let s2 = f32x4_splat(*a2.add(p));
                let s3 = f32x4_splat(*a3.add(p));
                c00 = f32x4_add(c00, f32x4_mul(s0, b0)); c01 = f32x4_add(c01, f32x4_mul(s0, b1));
                c02 = f32x4_add(c02, f32x4_mul(s0, b2)); c03 = f32x4_add(c03, f32x4_mul(s0, b3));
                c10 = f32x4_add(c10, f32x4_mul(s1, b0)); c11 = f32x4_add(c11, f32x4_mul(s1, b1));
                c12 = f32x4_add(c12, f32x4_mul(s1, b2)); c13 = f32x4_add(c13, f32x4_mul(s1, b3));
                c20 = f32x4_add(c20, f32x4_mul(s2, b0)); c21 = f32x4_add(c21, f32x4_mul(s2, b1));
                c22 = f32x4_add(c22, f32x4_mul(s2, b2)); c23 = f32x4_add(c23, f32x4_mul(s2, b3));
                c30 = f32x4_add(c30, f32x4_mul(s3, b0)); c31 = f32x4_add(c31, f32x4_mul(s3, b1));
                c32 = f32x4_add(c32, f32x4_mul(s3, b2)); c33 = f32x4_add(c33, f32x4_mul(s3, b3));
                p += 1;
            }
            let r0 = c.add(i * n + j);
            let r1 = r0.add(n);
            let r2 = r1.add(n);
            let r3 = r2.add(n);
            v128_store(r0 as *mut v128, c00); v128_store(r0.add(4) as *mut v128, c01);
            v128_store(r0.add(8) as *mut v128, c02); v128_store(r0.add(12) as *mut v128, c03);
            v128_store(r1 as *mut v128, c10); v128_store(r1.add(4) as *mut v128, c11);
            v128_store(r1.add(8) as *mut v128, c12); v128_store(r1.add(12) as *mut v128, c13);
            v128_store(r2 as *mut v128, c20); v128_store(r2.add(4) as *mut v128, c21);
            v128_store(r2.add(8) as *mut v128, c22); v128_store(r2.add(12) as *mut v128, c23);
            v128_store(r3 as *mut v128, c30); v128_store(r3.add(4) as *mut v128, c31);
            v128_store(r3.add(8) as *mut v128, c32); v128_store(r3.add(12) as *mut v128, c33);
            j += 16;
        }
        i += 4;
    }
}

/// NHWC depthwise convolution: in `[h,w,c]`, weights `[k,k,c]`, out `[ho,wo,c]`, channels
/// sixteen at a time. The bounds check sits outside the channel loop — for each output
/// pixel the valid taps are chosen once, then accumulated straight into the output row.
#[no_mangle]
pub unsafe extern "C" fn dwconv(
    h: usize, w: usize, c: usize, k: usize, stride: usize, pad: usize,
    ho: usize, wo: usize, inp: *const f32, wt: *const f32, out: *mut f32,
) {
    let zero = f32x4_splat(0.0);
    for oy in 0..ho {
        for ox in 0..wo {
            let o = out.add((oy * wo + ox) * c);
            let mut ch = 0;
            while ch < c {
                v128_store(o.add(ch) as *mut v128, zero);
                v128_store(o.add(ch + 4) as *mut v128, zero);
                v128_store(o.add(ch + 8) as *mut v128, zero);
                v128_store(o.add(ch + 12) as *mut v128, zero);
                ch += 16;
            }
            for ky in 0..k {
                let iy = (oy * stride + ky) as isize - pad as isize;
                if iy < 0 || iy >= h as isize { continue; }
                for kx in 0..k {
                    let ix = (ox * stride + kx) as isize - pad as isize;
                    if ix < 0 || ix >= w as isize { continue; }
                    let ip = inp.add(((iy as usize) * w + ix as usize) * c);
                    let wp = wt.add((ky * k + kx) * c);
                    let mut ch = 0;
                    while ch < c {
                        let o0 = o.add(ch) as *mut v128;
                        let o1 = o.add(ch + 4) as *mut v128;
                        let o2 = o.add(ch + 8) as *mut v128;
                        let o3 = o.add(ch + 12) as *mut v128;
                        v128_store(o0, f32x4_add(v128_load(o0), f32x4_mul(v128_load(ip.add(ch) as *const v128), v128_load(wp.add(ch) as *const v128))));
                        v128_store(o1, f32x4_add(v128_load(o1), f32x4_mul(v128_load(ip.add(ch + 4) as *const v128), v128_load(wp.add(ch + 4) as *const v128))));
                        v128_store(o2, f32x4_add(v128_load(o2), f32x4_mul(v128_load(ip.add(ch + 8) as *const v128), v128_load(wp.add(ch + 8) as *const v128))));
                        v128_store(o3, f32x4_add(v128_load(o3), f32x4_mul(v128_load(ip.add(ch + 12) as *const v128), v128_load(wp.add(ch + 12) as *const v128))));
                        ch += 16;
                    }
                }
            }
        }
    }
}

/// e^x for four lanes. 2^y is split into an integer part that goes straight into the
/// exponent bits and a fraction handled by a fifth-order polynomial. Relative error is
/// about 1e-5 — inside the golden's 1e-4, and `swish` is checked against `Math.exp` in
/// `borch-ts/test/cpu.ts` rather than trusted.
#[inline(always)]
unsafe fn exp_f32x4(x: v128) -> v128 {
    let x = f32x4_pmin(f32x4_pmax(x, f32x4_splat(-87.0)), f32x4_splat(88.0));
    let y = f32x4_mul(x, f32x4_splat(1.442_695_04));
    let yi = f32x4_floor(y);
    let f = f32x4_sub(y, yi);
    // 2^f for f in [0,1): the Taylor series of e^(f·ln2) to the fifth term.
    let mut p = f32x4_splat(0.001_333_36);
    p = f32x4_add(f32x4_mul(p, f), f32x4_splat(0.009_618_13));
    p = f32x4_add(f32x4_mul(p, f), f32x4_splat(0.055_504_11));
    p = f32x4_add(f32x4_mul(p, f), f32x4_splat(0.240_226_51));
    p = f32x4_add(f32x4_mul(p, f), f32x4_splat(0.693_147_18));
    p = f32x4_add(f32x4_mul(p, f), f32x4_splat(1.0));
    let e = i32x4_shl(i32x4_add(i32x4_trunc_sat_f32x4(yi), i32x4_splat(127)), 23);
    f32x4_mul(p, e)
}

#[inline(always)]
unsafe fn sigmoid_f32x4(v: v128) -> v128 {
    let one = f32x4_splat(1.0);
    f32x4_div(one, f32x4_add(one, exp_f32x4(f32x4_neg(v))))
}

/// x ← x · sigmoid(x), in place. `n % 4 == 0`.
#[no_mangle]
pub unsafe extern "C" fn swish(n: usize, x: *mut f32) {
    let mut i = 0;
    while i < n {
        let p = x.add(i) as *mut v128;
        let v = v128_load(p);
        v128_store(p, f32x4_mul(v, sigmoid_f32x4(v)));
        i += 4;
    }
}

/// x[rows×c] += bias[c], then an activation: 0 none, 1 swish, 2 sigmoid, 3 relu. `c % 4 == 0`.
/// A folded BatchNorm's bias and the activation in one pass over memory — measured at a
/// fifth of the forward, which is why folding it into `gemm`'s epilogue is the next lever.
#[no_mangle]
pub unsafe extern "C" fn bias_act(rows: usize, c: usize, x: *mut f32, bias: *const f32, act: u32) {
    for r in 0..rows {
        let row = x.add(r * c);
        let mut ch = 0;
        while ch < c {
            let p = row.add(ch) as *mut v128;
            let mut v = f32x4_add(v128_load(p), v128_load(bias.add(ch) as *const v128));
            if act == 1 { v = f32x4_mul(v, sigmoid_f32x4(v)); }
            else if act == 2 { v = sigmoid_f32x4(v); }
            else if act == 3 { v = f32x4_pmax(v, f32x4_splat(0.0)); }
            v128_store(p, v);
            ch += 4;
        }
    }
}

/// out[c] = mean over rows of x[rows×c]. Squeeze-and-excite's squeeze and the global
/// average pool. `c % 4 == 0`.
#[no_mangle]
pub unsafe extern "C" fn mean_rows(rows: usize, c: usize, x: *const f32, out: *mut f32) {
    let mut ch = 0;
    while ch < c {
        v128_store(out.add(ch) as *mut v128, f32x4_splat(0.0));
        ch += 4;
    }
    for r in 0..rows {
        let row = x.add(r * c);
        let mut ch = 0;
        while ch < c {
            let o = out.add(ch) as *mut v128;
            v128_store(o, f32x4_add(v128_load(o), v128_load(row.add(ch) as *const v128)));
            ch += 4;
        }
    }
    let inv = f32x4_splat(1.0 / rows as f32);
    let mut ch = 0;
    while ch < c {
        let o = out.add(ch) as *mut v128;
        v128_store(o, f32x4_mul(v128_load(o), inv));
        ch += 4;
    }
}

/// x[rows×c] *= s[c]. Squeeze-and-excite's excite. `c % 4 == 0`.
#[no_mangle]
pub unsafe extern "C" fn scale_rows(rows: usize, c: usize, x: *mut f32, s: *const f32) {
    for r in 0..rows {
        let row = x.add(r * c);
        let mut ch = 0;
        while ch < c {
            let p = row.add(ch) as *mut v128;
            v128_store(p, f32x4_mul(v128_load(p), v128_load(s.add(ch) as *const v128)));
            ch += 4;
        }
    }
}

/// a[n] += b[n]. The residual connection. `n % 4 == 0`.
#[no_mangle]
pub unsafe extern "C" fn add_inplace(n: usize, a: *mut f32, b: *const f32) {
    let mut i = 0;
    while i < n {
        let p = a.add(i) as *mut v128;
        v128_store(p, f32x4_add(v128_load(p), v128_load(b.add(i) as *const v128)));
        i += 4;
    }
}

/// x ← max(x, 0), in place. `n % 4 == 0`. ResNet's residual add is followed by this.
#[no_mangle]
pub unsafe extern "C" fn relu(n: usize, x: *mut f32) {
    let zero = f32x4_splat(0.0);
    let mut i = 0;
    while i < n {
        let p = x.add(i) as *mut v128;
        v128_store(p, f32x4_pmax(v128_load(p), zero));
        i += 4;
    }
}

/// NHWC im2col: in `[h,w,c]` → out `[ho·wo, k·k·c]`, tap-major then channel, zeros where
/// a tap falls outside. A dense k×k convolution is then one `gemm` against weights packed
/// in the same tap-major order. Channels are copied four at a time when `c % 4 == 0`,
/// one at a time otherwise — the stem's three input channels take the second path.
#[no_mangle]
pub unsafe extern "C" fn im2col(
    h: usize, w: usize, c: usize, k: usize, stride: usize, pad: usize,
    ho: usize, wo: usize, inp: *const f32, out: *mut f32,
) {
    let kk = k * k;
    let zero = f32x4_splat(0.0);
    for oy in 0..ho {
        for ox in 0..wo {
            let row = out.add((oy * wo + ox) * kk * c);
            for ky in 0..k {
                let iy = (oy * stride + ky) as isize - pad as isize;
                for kx in 0..k {
                    let ix = (ox * stride + kx) as isize - pad as isize;
                    let dst = row.add((ky * k + kx) * c);
                    let inside = iy >= 0 && iy < h as isize && ix >= 0 && ix < w as isize;
                    if c % 4 == 0 {
                        let mut ch = 0;
                        if inside {
                            let src = inp.add(((iy as usize) * w + ix as usize) * c);
                            while ch < c { v128_store(dst.add(ch) as *mut v128, v128_load(src.add(ch) as *const v128)); ch += 4; }
                        } else {
                            while ch < c { v128_store(dst.add(ch) as *mut v128, zero); ch += 4; }
                        }
                    } else if inside {
                        let src = inp.add(((iy as usize) * w + ix as usize) * c);
                        for ch in 0..c { *dst.add(ch) = *src.add(ch); }
                    } else {
                        for ch in 0..c { *dst.add(ch) = 0.0; }
                    }
                }
            }
        }
    }
}

/// NHWC max pool, `c % 4 == 0`. Taps outside the input are skipped, which is what padding
/// with −∞ means — torch's `MaxPool2d(3, 2, 1)` on the ResNet stem.
#[no_mangle]
pub unsafe extern "C" fn maxpool(
    h: usize, w: usize, c: usize, k: usize, stride: usize, pad: usize,
    ho: usize, wo: usize, inp: *const f32, out: *mut f32,
) {
    let neg = f32x4_splat(f32::NEG_INFINITY);
    for oy in 0..ho {
        for ox in 0..wo {
            let o = out.add((oy * wo + ox) * c);
            let mut ch = 0;
            while ch < c {
                let mut acc = neg;
                for ky in 0..k {
                    let iy = (oy * stride + ky) as isize - pad as isize;
                    if iy < 0 || iy >= h as isize { continue; }
                    for kx in 0..k {
                        let ix = (ox * stride + kx) as isize - pad as isize;
                        if ix < 0 || ix >= w as isize { continue; }
                        acc = f32x4_pmax(acc, v128_load(inp.add(((iy as usize) * w + ix as usize) * c + ch) as *const v128));
                    }
                }
                v128_store(o.add(ch) as *mut v128, acc);
                ch += 4;
            }
        }
    }
}
