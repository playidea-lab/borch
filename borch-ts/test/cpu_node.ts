/**
 * The CPU device against a scalar reference — no browser, no adapter, no network.
 *
 * `cpu.py` proves the device against the WebGPU device on real checkpoints, and needs a
 * GPU and 68 MB from the hub to do it. This file proves the same wiring against a plain
 * JavaScript reference on a small random network, and needs node. `tests/test_cpu_device.py`
 * runs it under `pytest`, so the device is checked on every machine the suite runs on —
 * which is the point: this device exists for machines with no GPU, and until this file the
 * only check of it required one.
 *
 * ## What the network exercises
 *
 * Every node kind the graph has, on shapes small enough to reference in float64 in a few
 * hundred milliseconds: a strided 3×3 convolution with three input channels (the stem's
 * scalar im2col path), a depthwise 3×3 with swish, squeeze-and-excite, two 1×1
 * convolutions joined by a residual add with relu, a max pool, a dense 3×3 on sixteen-wide
 * channels (the vector im2col path, blocked), global average pool, a linear layer. Batch
 * two, so the row padding is exercised at 2·25 = 50 rows → 52.
 *
 * The reference folds nothing: it runs conv, then BatchNorm in eval form, then the
 * activation, from the same torch-layout arrays the builder was given. Agreement to 1e-4
 * relative says the folding, the repacking, the padding and the in-place operands are
 * right; the swish carries the module's `exp` approximation, which is why the bound is
 * not 1e-6.
 */
import { ACT, loadKernels } from "../src/cpu/load.js";
import { GraphBuilder, type BatchNorm } from "../src/cpu/graph.js";
import { CpuRunner } from "../src/cpu/runner.js";
import { LinearHead, cosineNeighbours } from "../src/cpu/train.js";

interface Check { name: string; ok: boolean; note: string }

let seed = 0x9e3779b9;
function rnd(): number { seed ^= seed << 13; seed >>>= 0; seed ^= seed >>> 17; seed ^= seed << 5; seed >>>= 0; return seed / 4294967296; }
const uni = (n: number, lo: number, hi: number): Float32Array => Float32Array.from({ length: n }, () => lo + (hi - lo) * rnd());
const gauss = (n: number, scale: number): Float32Array => Float32Array.from({ length: n }, () => scale * Math.sqrt(-2 * Math.log(Math.max(rnd(), 1e-9))) * Math.cos(2 * Math.PI * rnd()));
function bnParams(c: number): BatchNorm {
  return { weight: uni(c, 0.5, 1.5), bias: uni(c, -0.2, 0.2), runningMean: uni(c, -0.2, 0.2), runningVar: uni(c, 0.5, 1.5) };
}

// ---- the reference: NHWC arrays as plain Float64Array with a shape ----
interface T { h: number; w: number; c: number; d: Float64Array }
const at = (t: T, y: number, x: number, ch: number): number => t.d[(y * t.w + x) * t.c + ch] ?? 0;
function conv(x: T, W: Float32Array, cout: number, k: number, stride: number, pad: number, groups: 1 | "dw"): T {
  const ho = Math.floor((x.h + 2 * pad - k) / stride) + 1, wo = Math.floor((x.w + 2 * pad - k) / stride) + 1;
  const y: T = { h: ho, w: wo, c: cout, d: new Float64Array(ho * wo * cout) };
  for (let oy = 0; oy < ho; oy++) for (let ox = 0; ox < wo; ox++) for (let o = 0; o < cout; o++) {
    let acc = 0;
    for (let ky = 0; ky < k; ky++) for (let kx = 0; kx < k; kx++) {
      const iy = oy * stride + ky - pad, ix = ox * stride + kx - pad;
      if (iy < 0 || iy >= x.h || ix < 0 || ix >= x.w) continue;
      if (groups === "dw") acc += at(x, iy, ix, o) * (W[(o * k + ky) * k + kx] ?? 0);
      else for (let i = 0; i < x.c; i++) acc += at(x, iy, ix, i) * (W[((o * x.c + i) * k + ky) * k + kx] ?? 0);
    }
    y.d[(oy * wo + ox) * cout + o] = acc;
  }
  return y;
}
function bnAct(t: T, bn: BatchNorm | null, act: number): T {
  const d = new Float64Array(t.d.length);
  for (let p = 0; p < t.h * t.w; p++) for (let ch = 0; ch < t.c; ch++) {
    let v = t.d[p * t.c + ch] ?? 0;
    if (bn) v = ((v - (bn.runningMean[ch] ?? 0)) / Math.sqrt((bn.runningVar[ch] ?? 1) + 1e-5)) * (bn.weight[ch] ?? 1) + (bn.bias[ch] ?? 0);
    if (act === ACT.swish) v = v / (1 + Math.exp(-v)); else if (act === ACT.sigmoid) v = 1 / (1 + Math.exp(-v)); else if (act === ACT.relu) v = Math.max(v, 0);
    d[p * t.c + ch] = v;
  }
  return { ...t, d };
}
function se(t: T, w1: Float32Array, b1: Float32Array, w2: Float32Array, b2: Float32Array, cse: number): T {
  const mean = new Float64Array(t.c);
  for (let p = 0; p < t.h * t.w; p++) for (let ch = 0; ch < t.c; ch++) mean[ch] = (mean[ch] ?? 0) + (t.d[p * t.c + ch] ?? 0) / (t.h * t.w);
  const r = new Float64Array(cse);
  for (let j = 0; j < cse; j++) { let a = b1[j] ?? 0; for (let i = 0; i < t.c; i++) a += (mean[i] ?? 0) * (w1[j * t.c + i] ?? 0); r[j] = a / (1 + Math.exp(-a)); }
  const g = new Float64Array(t.c);
  for (let i = 0; i < t.c; i++) { let a = b2[i] ?? 0; for (let j = 0; j < cse; j++) a += (r[j] ?? 0) * (w2[i * cse + j] ?? 0); g[i] = 1 / (1 + Math.exp(-a)); }
  const d = new Float64Array(t.d.length);
  for (let p = 0; p < t.h * t.w; p++) for (let ch = 0; ch < t.c; ch++) d[p * t.c + ch] = (t.d[p * t.c + ch] ?? 0) * (g[ch] ?? 0);
  return { ...t, d };
}
function maxpool(t: T, k: number, stride: number, pad: number): T {
  const ho = Math.floor((t.h + 2 * pad - k) / stride) + 1, wo = Math.floor((t.w + 2 * pad - k) / stride) + 1;
  const y: T = { h: ho, w: wo, c: t.c, d: new Float64Array(ho * wo * t.c) };
  for (let oy = 0; oy < ho; oy++) for (let ox = 0; ox < wo; ox++) for (let ch = 0; ch < t.c; ch++) {
    let m = -Infinity;
    for (let ky = 0; ky < k; ky++) for (let kx = 0; kx < k; kx++) { const iy = oy * stride + ky - pad, ix = ox * stride + kx - pad; if (iy < 0 || iy >= t.h || ix < 0 || ix >= t.w) continue; m = Math.max(m, at(t, iy, ix, ch)); }
    y.d[(oy * wo + ox) * t.c + ch] = m;
  }
  return y;
}

export async function check(): Promise<{ checks: Check[]; text: string }> {
  const checks: Check[] = [];
  const lines: string[] = [];
  const say = (s: string): void => { lines.push(s); };
  const K = await loadKernels();
  say(`kernels: ${K.flavor}`);

  // ---- the network ----
  const B = 2, H = 20, Wd = 20;
  const p = {
    stemW: gauss(16 * 3 * 9, 0.3), stemBn: bnParams(16),
    dwW: gauss(16 * 9, 0.3), dwBn: bnParams(16),
    seW1: gauss(4 * 16, 0.3), seB1: uni(4, -0.1, 0.1), seW2: gauss(16 * 4, 0.3), seB2: uni(16, -0.1, 0.1),
    mainW: gauss(32 * 16, 0.3), mainBn: bnParams(32),
    shortW: gauss(32 * 16, 0.3), shortBn: bnParams(32),
    denseW: gauss(32 * 32 * 9, 0.08), denseBn: bnParams(32),
    fcW: gauss(10 * 32, 0.3), fcB: uni(10, -0.1, 0.1),
  };
  const g = new GraphBuilder();
  const x0 = g.input(3);
  const s1 = g.conv(x0, { weight: p.stemW, cout: 16, cin: 3, k: 3, stride: 2, pad: 1, bn: p.stemBn, act: ACT.relu });
  const s2 = g.dwconv(s1, { weight: p.dwW, cout: 16, cin: 16, k: 3, stride: 1, pad: 1, bn: p.dwBn, act: ACT.swish });
  const s3 = g.se(s2, p.seW1, p.seB1, p.seW2, p.seB2, 4);
  const main = g.conv(s3, { weight: p.mainW, cout: 32, cin: 16, k: 1, stride: 1, pad: 0, bn: p.mainBn });
  const shortcut = g.conv(s3, { weight: p.shortW, cout: 32, cin: 16, k: 1, stride: 1, pad: 0, bn: p.shortBn });
  const s4 = g.add(main, shortcut, ACT.relu);
  const s5 = g.maxpool(s4, 3, 2, 1);
  const s6 = g.conv(s5, { weight: p.denseW, cout: 32, cin: 32, k: 3, stride: 1, pad: 1, bn: p.denseBn, act: ACT.swish });
  const s7 = g.gap(s6);
  const out = g.linear(s7, p.fcW, p.fcB, 10);
  const runner = new CpuRunner(K, g.finish(out));

  const input = gauss(B * 3 * H * Wd, 1);
  const t0 = performance.now();
  const got = runner.forward(input, B, H, Wd);
  const ms = performance.now() - t0;

  // the reference, image by image
  let worst = 0, scale = 0;
  for (let b = 0; b < B; b++) {
    const x: T = { h: H, w: Wd, c: 3, d: new Float64Array(H * Wd * 3) };
    for (let ch = 0; ch < 3; ch++) for (let q = 0; q < H * Wd; q++) x.d[q * 3 + ch] = input[(b * 3 + ch) * H * Wd + q] ?? 0;
    let t = bnAct(conv(x, p.stemW, 16, 3, 2, 1, 1), p.stemBn, ACT.relu);
    t = bnAct(conv(t, p.dwW, 16, 3, 1, 1, "dw"), p.dwBn, ACT.swish);
    t = se(t, p.seW1, p.seB1, p.seW2, p.seB2, 4);
    const m = bnAct(conv(t, p.mainW, 32, 1, 1, 0, 1), p.mainBn, ACT.none);
    const sc = bnAct(conv(t, p.shortW, 32, 1, 1, 0, 1), p.shortBn, ACT.none);
    const added: T = { ...m, d: m.d.map((v, i) => Math.max(v + (sc.d[i] ?? 0), 0)) };
    let u = maxpool(added, 3, 2, 1);
    u = bnAct(conv(u, p.denseW, 32, 3, 1, 1, 1), p.denseBn, ACT.swish);
    const pooled = new Float64Array(32);
    for (let q = 0; q < u.h * u.w; q++) for (let ch = 0; ch < 32; ch++) pooled[ch] = (pooled[ch] ?? 0) + (u.d[q * 32 + ch] ?? 0) / (u.h * u.w);
    for (let o = 0; o < 10; o++) {
      let a = p.fcB[o] ?? 0; for (let i = 0; i < 32; i++) a += (pooled[i] ?? 0) * (p.fcW[o * 32 + i] ?? 0);
      worst = Math.max(worst, Math.abs(a - (got[b * 10 + o] ?? 0))); scale = Math.max(scale, Math.abs(a));
    }
  }
  const rel = worst / scale;
  say(`network: ${runner ? 10 : 0} nodes · batch ${B} · max|Δ| ${worst.toExponential(2)} on logits up to ${scale.toFixed(3)} → relative ${rel.toExponential(2)} · ${ms.toFixed(1)} ms`);
  checks.push({ name: "cpu forward matches the scalar reference", ok: rel < 1e-4 && got.length === B * 10, note: `relative ${rel.toExponential(2)}` });

  // ---- the head: full-batch SGD against a float64 reference ----
  {
    const N = 24, D = 16, Kc = 3, STEPS = 30, LR = 0.1, MU = 0.9;
    const feats = gauss(N * D, 1);
    const labels = Array.from({ length: N }, (_, i) => i % Kc);
    const head = new LinearHead(K, D, Kc, { lr: LR, momentum: MU });
    const losses = head.fit(feats, labels, N, STEPS);
    // reference
    const W = new Float64Array(Kc * D), bb = new Float64Array(Kc), vW = new Float64Array(Kc * D), vb = new Float64Array(Kc);
    let worstLoss = 0;
    for (let step = 0; step < STEPS; step++) {
      const gW = new Float64Array(Kc * D), gb = new Float64Array(Kc);
      let loss = 0;
      for (let r = 0; r < N; r++) {
        const z = new Float64Array(Kc);
        for (let k = 0; k < Kc; k++) { let a = bb[k] ?? 0; for (let d = 0; d < D; d++) a += (feats[r * D + d] ?? 0) * (W[k * D + d] ?? 0); z[k] = a; }
        const mx = Math.max(...z); let s = 0; for (let k = 0; k < Kc; k++) s += Math.exp((z[k] ?? 0) - mx);
        const y = labels[r] ?? 0;
        loss += -((z[y] ?? 0) - mx - Math.log(s));
        for (let k = 0; k < Kc; k++) {
          const gk = (Math.exp((z[k] ?? 0) - mx) / s - (k === y ? 1 : 0)) / N;
          gb[k] = (gb[k] ?? 0) + gk;
          for (let d = 0; d < D; d++) gW[k * D + d] = (gW[k * D + d] ?? 0) + gk * (feats[r * D + d] ?? 0);
        }
      }
      loss /= N;
      worstLoss = Math.max(worstLoss, Math.abs(loss - (losses[step] ?? 0)) / Math.max(1e-9, Math.abs(loss)));
      for (let i = 0; i < Kc * D; i++) { vW[i] = MU * (vW[i] ?? 0) + (gW[i] ?? 0); W[i] = (W[i] ?? 0) - LR * (vW[i] ?? 0); }
      for (let k = 0; k < Kc; k++) { vb[k] = MU * (vb[k] ?? 0) + (gb[k] ?? 0); bb[k] = (bb[k] ?? 0) - LR * (vb[k] ?? 0); }
    }
    const st = head.stateDict();
    let worstW = 0; for (let i = 0; i < Kc * D; i++) worstW = Math.max(worstW, Math.abs((st.weight[i] ?? 0) - (W[i] ?? 0)));
    say(`head: ${STEPS} steps · loss ${losses[0]?.toFixed(4)} → ${losses[STEPS - 1]?.toFixed(4)} · worst relative loss gap ${worstLoss.toExponential(2)} · weights within ${worstW.toExponential(2)}`);
    checks.push({ name: "linear head matches the float64 SGD reference", ok: worstLoss < 1e-4 && worstW < 1e-4, note: `loss ${worstLoss.toExponential(2)}, weights ${worstW.toExponential(2)}` });
  }

  // ---- neighbours ----
  {
    const N = 30, D = 16, k = 3;
    const feats = gauss(N * D, 1);
    const got = cosineNeighbours(K, feats, N, D, k);
    let mismatched = 0, worstSim = 0;
    const norm = (i: number): number => { let s = 0; for (let d = 0; d < D; d++) s += (feats[i * D + d] ?? 0) ** 2; return Math.sqrt(s); };
    for (let i = 0; i < N; i++) {
      const sims: { j: number; s: number }[] = [];
      for (let j = 0; j < N; j++) { if (j === i) continue; let dot = 0; for (let d = 0; d < D; d++) dot += (feats[i * D + d] ?? 0) * (feats[j * D + d] ?? 0); sims.push({ j, s: dot / (norm(i) * norm(j)) }); }
      sims.sort((a, b) => b.s - a.s);
      for (let q = 0; q < k; q++) { if (got.indices[i * k + q] !== sims[q]?.j) mismatched++; worstSim = Math.max(worstSim, Math.abs((got.sims[i * k + q] ?? 0) - (sims[q]?.s ?? 0))); }
    }
    say(`neighbours: ${N} rows, k=${k} · ${mismatched} of ${N * k} differ · sims within ${worstSim.toExponential(2)}`);
    checks.push({ name: "cosine neighbours match the reference", ok: mismatched === 0 && worstSim < 1e-5, note: `${mismatched} differ, sims ${worstSim.toExponential(2)}` });
  }
  return { checks, text: lines.join("\n") };
}
