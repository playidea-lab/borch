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

/// `a·b + c` on four lanes. With the `relaxed` feature this is one instruction,
/// `f32x4.relaxed_madd`, whose rounding the hardware decides (fused or not — the last bit
/// may differ by machine, which is what the proposal's name means). Without it, the
/// multiply and the add are two instructions with one rounding each, the same on every
/// machine — the module the golden is measured with. Every multiply-add in this file goes
/// through here, so the two modules differ in exactly this.
#[inline(always)]
unsafe fn madd(a: v128, b: v128, c: v128) -> v128 {
    #[cfg(feature = "relaxed")]
    { f32x4_relaxed_madd(a, b, c) }
    #[cfg(not(feature = "relaxed"))]
    { f32x4_add(c, f32x4_mul(a, b)) }
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

/// C[m×n] = A[m×k] · B[k×n], row-major f32. Register block of 4 rows × 16 columns (four
/// v128 per row), so one pass over `k` does 64 multiply-adds from 4 loads of B and 4 splats
/// of A. A 1×1 convolution in NHWC is exactly this call: A the activations `[HW × Cin]`,
/// B the weights `[Cin × Cout]`. One body with `gemm_bias_act` — this is that with no epilogue.
#[no_mangle]
pub unsafe extern "C" fn gemm(m: usize, n: usize, k: usize, a: *const f32, b: *const f32, c: *mut f32) {
    gemm_bias_act(m, n, k, a, b, c, core::ptr::null(), 0)
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
                        v128_store(o0, madd(v128_load(ip.add(ch) as *const v128), v128_load(wp.add(ch) as *const v128), v128_load(o0)));
                        v128_store(o1, madd(v128_load(ip.add(ch + 4) as *const v128), v128_load(wp.add(ch + 4) as *const v128), v128_load(o1)));
                        v128_store(o2, madd(v128_load(ip.add(ch + 8) as *const v128), v128_load(wp.add(ch + 8) as *const v128), v128_load(o2)));
                        v128_store(o3, madd(v128_load(ip.add(ch + 12) as *const v128), v128_load(wp.add(ch + 12) as *const v128), v128_load(o3)));
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

// ---- a head on cached features: forward is `gemm` + `bias_act`; these are the rest ----

/// Softmax cross-entropy, backward half. For each of `rows` rows of `logits[rows×c]`
/// (the first `c_real` columns are classes, the rest padding): `grad[row] = (softmax − onehot) / rows`
/// with the padding columns written zero, and `stats[row·2] = max`, `stats[row·2+1] = Σ exp(l−max)`
/// so the caller can take `loss = mean(−(l[label] − max − ln Σ))` — this module has no `ln`,
/// on purpose: one scalar per row is the host's to finish. `labels` are class indices as f32.
#[no_mangle]
pub unsafe extern "C" fn softmax_xent_grad(
    rows: usize, c: usize, c_real: usize, logits: *const f32, labels: *const f32, grad: *mut f32, stats: *mut f32,
) {
    let neg = f32x4_splat(f32::NEG_INFINITY);
    let zero = f32x4_splat(0.0);
    let inv_rows = 1.0 / rows as f32;
    for r in 0..rows {
        let row = logits.add(r * c);
        let out = grad.add(r * c);
        // max over the real columns
        let mut m = neg;
        let mut j = 0;
        while j < c {
            let lane = u32x4_lt(u32x4(j as u32, j as u32 + 1, j as u32 + 2, j as u32 + 3), u32x4_splat(c_real as u32));
            m = f32x4_pmax(m, v128_bitselect(v128_load(row.add(j) as *const v128), neg, lane));
            j += 4;
        }
        let mx = f32x4_extract_lane::<0>(m).max(f32x4_extract_lane::<1>(m)).max(f32x4_extract_lane::<2>(m)).max(f32x4_extract_lane::<3>(m));
        let mv = f32x4_splat(mx);
        // exp(l − max) into grad, sum it
        let mut sum = zero;
        j = 0;
        while j < c {
            let lane = u32x4_lt(u32x4(j as u32, j as u32 + 1, j as u32 + 2, j as u32 + 3), u32x4_splat(c_real as u32));
            let e = v128_bitselect(exp_f32x4(f32x4_sub(v128_load(row.add(j) as *const v128), mv)), zero, lane);
            v128_store(out.add(j) as *mut v128, e);
            sum = f32x4_add(sum, e);
            j += 4;
        }
        let s = f32x4_extract_lane::<0>(sum) + f32x4_extract_lane::<1>(sum) + f32x4_extract_lane::<2>(sum) + f32x4_extract_lane::<3>(sum);
        *stats.add(r * 2) = mx;
        *stats.add(r * 2 + 1) = s;
        // (p − onehot) / rows
        let scale = f32x4_splat(inv_rows / s);
        j = 0;
        while j < c {
            let p = out.add(j) as *mut v128;
            v128_store(p, f32x4_mul(v128_load(p), scale));
            j += 4;
        }
        let label = *labels.add(r) as usize;
        if label < c_real { *out.add(label) -= inv_rows; }
    }
}

/// out[d×k] += Σ_n x[n,d] · g[n,k] — the weight gradient `Xᵀ·G` of a linear layer, as rank-one
/// updates so nothing has to be transposed. `k % 4 == 0`; `out` is not zeroed here.
#[no_mangle]
pub unsafe extern "C" fn outer_acc(n: usize, d: usize, k: usize, x: *const f32, g: *const f32, out: *mut f32) {
    for row in 0..n {
        let gr = g.add(row * k);
        let xr = x.add(row * d);
        for dd in 0..d {
            let s = f32x4_splat(*xr.add(dd));
            let o = out.add(dd * k);
            let mut kk = 0;
            while kk < k {
                let p = o.add(kk) as *mut v128;
                v128_store(p, madd(s, v128_load(gr.add(kk) as *const v128), v128_load(p)));
                kk += 4;
            }
        }
    }
}

/// SGD with momentum and weight decay, torch's order: `g += wd·p; v = μ·v + g; p −= lr·v`. `n % 4 == 0`.
#[no_mangle]
pub unsafe extern "C" fn sgd_step(n: usize, p: *mut f32, g: *const f32, v: *mut f32, lr: f32, momentum: f32, weight_decay: f32) {
    let lrv = f32x4_splat(lr);
    let mu = f32x4_splat(momentum);
    let wd = f32x4_splat(weight_decay);
    let mut i = 0;
    while i < n {
        let pp = p.add(i) as *mut v128;
        let vp = v.add(i) as *mut v128;
        let pv = v128_load(pp);
        let gv = f32x4_add(v128_load(g.add(i) as *const v128), f32x4_mul(wd, pv));
        let nv = f32x4_add(f32x4_mul(mu, v128_load(vp)), gv);
        v128_store(vp, nv);
        v128_store(pp, f32x4_sub(pv, f32x4_mul(lrv, nv)));
        i += 4;
    }
}

/// Each row of x[rows×c] divided by its L2 norm (rows of zeros stay zero). `c % 4 == 0`.
/// Cosine similarity is then plain `gemm` against the transpose.
#[no_mangle]
pub unsafe extern "C" fn l2_normalize_rows(rows: usize, c: usize, x: *mut f32) {
    for r in 0..rows {
        let row = x.add(r * c);
        let mut acc = f32x4_splat(0.0);
        let mut j = 0;
        while j < c {
            let v = v128_load(row.add(j) as *const v128);
            acc = f32x4_add(acc, f32x4_mul(v, v));
            j += 4;
        }
        let ss = f32x4_extract_lane::<0>(acc) + f32x4_extract_lane::<1>(acc) + f32x4_extract_lane::<2>(acc) + f32x4_extract_lane::<3>(acc);
        if ss == 0.0 { continue; }
        let inv = f32x4_splat(1.0 / f32x4_extract_lane::<0>(f32x4_sqrt(f32x4_splat(ss))));
        j = 0;
        while j < c {
            let p = row.add(j) as *mut v128;
            v128_store(p, f32x4_mul(v128_load(p), inv));
            j += 4;
        }
    }
}

/// out[cols×rows] = in[rows×cols]ᵀ. Scalar; the matrices this serves are a few megabytes.
#[no_mangle]
pub unsafe extern "C" fn transpose(rows: usize, cols: usize, inp: *const f32, out: *mut f32) {
    for r in 0..rows {
        for cidx in 0..cols {
            *out.add(cidx * rows + r) = *inp.add(r * cols + cidx);
        }
    }
}

/// x[n] ← 0. For accumulators before `outer_acc`. `n % 4 == 0`.
#[no_mangle]
pub unsafe extern "C" fn zero(n: usize, x: *mut f32) {
    let z = f32x4_splat(0.0);
    let mut i = 0;
    while i < n {
        v128_store(x.add(i) as *mut v128, z);
        i += 4;
    }
}

// ---- epilogues: the bias and the activation applied where the result is still in registers ----

/// The activation `bias_act` applies, on four lanes. 0 none, 1 swish, 2 sigmoid, 3 relu.
#[inline(always)]
unsafe fn act_f32x4(v: v128, act: u32) -> v128 {
    if act == 1 { f32x4_mul(v, sigmoid_f32x4(v)) }
    else if act == 2 { sigmoid_f32x4(v) }
    else if act == 3 { f32x4_pmax(v, f32x4_splat(0.0)) }
    else { v }
}

/// One register block of the GEMM: `R` rows of A against sixteen columns of B, the
/// `R × 4` accumulators in registers across the whole of `k`, bias and activation applied
/// on the way out. `R` is a compile-time constant so the arrays below are registers, not
/// memory — measured: at `R = 4` this runs at the speed of the hand-unrolled block it
/// replaced, in a third of the lines.
#[inline(always)]
unsafe fn micro<const R: usize>(k: usize, n: usize, a: *const f32, b: *const f32, c: *mut f32, j: usize, bias: *const f32, act: u32) {
    let z = f32x4_splat(0.0);
    let mut acc = [[z; 4]; R];
    let mut p = 0;
    while p < k {
        let bp = b.add(p * n + j);
        let b0 = v128_load(bp as *const v128);
        let b1 = v128_load(bp.add(4) as *const v128);
        let b2 = v128_load(bp.add(8) as *const v128);
        let b3 = v128_load(bp.add(12) as *const v128);
        let mut r = 0;
        while r < R {
            let s = f32x4_splat(*a.add(r * k + p));
            acc[r][0] = madd(s, b0, acc[r][0]);
            acc[r][1] = madd(s, b1, acc[r][1]);
            acc[r][2] = madd(s, b2, acc[r][2]);
            acc[r][3] = madd(s, b3, acc[r][3]);
            r += 1;
        }
        p += 1;
    }
    let (bb0, bb1, bb2, bb3) = if bias.is_null() { (z, z, z, z) } else {
        (v128_load(bias.add(j) as *const v128), v128_load(bias.add(j + 4) as *const v128),
         v128_load(bias.add(j + 8) as *const v128), v128_load(bias.add(j + 12) as *const v128))
    };
    let mut r = 0;
    while r < R {
        let row = c.add(r * n + j);
        v128_store(row as *mut v128, act_f32x4(f32x4_add(acc[r][0], bb0), act));
        v128_store(row.add(4) as *mut v128, act_f32x4(f32x4_add(acc[r][1], bb1), act));
        v128_store(row.add(8) as *mut v128, act_f32x4(f32x4_add(acc[r][2], bb2), act));
        v128_store(row.add(12) as *mut v128, act_f32x4(f32x4_add(acc[r][3], bb3), act));
        r += 1;
    }
}

/// `gemm`, then `+ bias[n]` (skipped when `bias` is null) and the activation before the
/// store — the sum never leaves the registers. `n % 16 == 0`; `m` is any count: four rows
/// at a time, then the tail (at most three rows) one row at a time. Writing exactly `m` rows
/// matters once two threads write neighbouring row ranges of one buffer (`threads.ts`) —
/// a padded write over the last block's tail used to land in the next image's first rows,
/// harmless in sequence, a race in parallel.
///
/// **Four rows by sixteen columns, and that was measured twice.** A wider block would
/// load less per multiply-add — six rows is ten loads for twenty-four against eight for
/// sixteen — and on paper 24 + 4 + 1 v128 fit in arm64's thirty-two. Measured on the
/// relaxed module (M4 Max, Chrome's V8 in node): six rows 60–84 GFLOPS, five rows 55–69,
/// four rows 97, on every ResNet-18 and B0 shape. The engine's register allocator spills
/// past about twenty live vectors, and a spilled accumulator costs more than the loads it
/// saves. A cache-blocked variant (`k` in chunks, rows in blocks) was measured earlier at
/// no gain. Both are recorded here so neither is written a third time.
#[no_mangle]
pub unsafe extern "C" fn gemm_bias_act(m: usize, n: usize, k: usize, a: *const f32, b: *const f32, c: *mut f32, bias: *const f32, act: u32) {
    let mut i = 0;
    while i + 4 <= m {
        let mut j = 0;
        while j < n { micro::<4>(k, n, a.add(i * k), b, c.add(i * n), j, bias, act); j += 16; }
        i += 4;
    }
    // The tail (at most three rows) one row at a time: one more instantiation of `micro`,
    // not three — the module is measured in kilobytes, and a tail is a block's last rows.
    while i < m {
        let mut j = 0;
        while j < n { micro::<1>(k, n, a.add(i * k), b, c.add(i * n), j, bias, act); j += 16; }
        i += 1;
    }
}

/// `dwconv` with the bias and the activation folded in — and **the accumulator in registers**.
///
/// The first version walked pixel → tap → channel and kept the running sum in the output
/// row: every tap read the row, added, and wrote it back — twelve loads and four stores per
/// four multiply-adds, and fusing the multiply-add (the relaxed module) moved it by 2 %,
/// which said the arithmetic was never the bound. This walks pixel → sixteen channels →
/// tap, holds the four accumulators in registers from the bias to the store, and applies the
/// activation before that one store. The valid tap range is decided once per row and once
/// per column, so the tap loop has no branch. Same contract as `dwconv`: `c % 16 == 0`. The
/// taps are summed in the same order as before, so the strict module's values are unchanged.
#[no_mangle]
pub unsafe extern "C" fn dwconv_bias_act(
    h: usize, w: usize, c: usize, k: usize, stride: usize, pad: usize,
    ho: usize, wo: usize, inp: *const f32, wt: *const f32, out: *mut f32, bias: *const f32, act: u32,
) {
    for oy in 0..ho {
        let y0 = oy * stride;                                   // iy = y0 + ky − pad
        let ky_lo = if pad > y0 { pad - y0 } else { 0 };
        let ky_hi = { let m = h + pad - y0; if m < k { m } else { k } };
        for ox in 0..wo {
            let x0 = ox * stride;
            let kx_lo = if pad > x0 { pad - x0 } else { 0 };
            let kx_hi = { let m = w + pad - x0; if m < k { m } else { k } };
            let o = out.add((oy * wo + ox) * c);
            let mut ch = 0;
            while ch < c {
                let mut a0 = v128_load(bias.add(ch) as *const v128);
                let mut a1 = v128_load(bias.add(ch + 4) as *const v128);
                let mut a2 = v128_load(bias.add(ch + 8) as *const v128);
                let mut a3 = v128_load(bias.add(ch + 12) as *const v128);
                for ky in ky_lo..ky_hi {
                    let row_in = inp.add((y0 + ky - pad) * w * c + ch);
                    let row_w = wt.add(ky * k * c + ch);
                    for kx in kx_lo..kx_hi {
                        let ip = row_in.add((x0 + kx - pad) * c);
                        let wp = row_w.add(kx * c);
                        a0 = madd(v128_load(ip as *const v128), v128_load(wp as *const v128), a0);
                        a1 = madd(v128_load(ip.add(4) as *const v128), v128_load(wp.add(4) as *const v128), a1);
                        a2 = madd(v128_load(ip.add(8) as *const v128), v128_load(wp.add(8) as *const v128), a2);
                        a3 = madd(v128_load(ip.add(12) as *const v128), v128_load(wp.add(12) as *const v128), a3);
                    }
                }
                v128_store(o.add(ch) as *mut v128, act_f32x4(a0, act));
                v128_store(o.add(ch + 4) as *mut v128, act_f32x4(a1, act));
                v128_store(o.add(ch + 8) as *mut v128, act_f32x4(a2, act));
                v128_store(o.add(ch + 12) as *mut v128, act_f32x4(a3, act));
                ch += 16;
            }
        }
    }
}

/// `im2col` for output rows `[row0, row0 + rows)` of one image only — so a convolution can be
/// done a block of rows at a time against one reused buffer instead of unrolling the whole
/// image (ResNet-18's first stage at batch 16 was 116 MB of columns per layer).
#[no_mangle]
pub unsafe extern "C" fn im2col_rows(
    h: usize, w: usize, c: usize, k: usize, stride: usize, pad: usize,
    wo: usize, row0: usize, rows: usize, inp: *const f32, out: *mut f32,
) {
    let kk = k * k;
    let zero = f32x4_splat(0.0);
    for r in 0..rows {
        let pix = row0 + r;
        let oy = pix / wo;
        let ox = pix % wo;
        let row = out.add(r * kk * c);
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
