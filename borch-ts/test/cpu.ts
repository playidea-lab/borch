/**
 * The CPU device against the WebGPU device — the same checkpoint, the same input, the
 * logits compared.
 *
 * ## What is asked
 *
 * Two classifiers from the hub, EfficientNet-B0 and ResNet-18, are loaded twice: once
 * through `borch-hub` onto the WebGPU device (the path the site uses), and once through
 * `cpu/safetensors` + `cpu/graph` + `cpu/runner` with no device at all. The same seeded
 * image goes through both. If the CPU graph is wired right — BatchNorm folded, weights
 * repacked, padding channels held at zero — the two answer the same to float32 noise.
 *
 * ## The plans are written here, for now
 *
 * `bimm-ts` owns the architectures and builds them as `nn.Module`s; its plan tables are
 * not exported. The two conversions below carry the tables themselves, which is a copy,
 * and a copy is the shape that drifts. This is the stopgap for proving the device; the
 * conversions move into bimm beside the plans they mirror, and this file then imports
 * them. Until then, a checkpoint key this file spells wrong fails loudly with the key
 * list — that is the guard against the copy being wrong in a way that passes.
 *
 * ## The tolerance is measured, not chosen
 *
 * Two float32 implementations summing in different orders land apart by float32 noise:
 * measured 2026-09-05 on `apple / metal-3`, EfficientNet-B0 8.8e-5 and ResNet-18 4e-7
 * relative to the largest logit (swish carries an exp approximation, relu carries
 * nothing). The check passes under 1e-3 and prints the number, so the margin is
 * visible rather than assumed.
 */
import { init, Device, Tensor, noGrad, scope, keepAlive } from "../src/index.js";
import { ACT } from "../src/cpu/load.js";
import { loadKernels } from "../src/cpu/load.js";
import { GraphBuilder, type CpuGraph, type BatchNorm } from "../src/cpu/graph.js";
import { CpuRunner } from "../src/cpu/runner.js";
import { readSafetensors, type HostStateDict } from "../src/cpu/safetensors.js";

interface Check { name: string; ok: boolean; note: string }

/** What this check uses of `borch-hub` — passed in by the page, which imports the real thing. */
export interface HubLike {
  load(manifestUrl: string, opts: { verify: boolean }): Promise<{ manifest: unknown; model: ModelLike }>;
  fetchWeights(manifest: unknown, manifestUrl: string): Promise<Uint8Array>;
}
export interface ModelLike { eval(): unknown; forward(x: Tensor): Tensor }

interface IndexEntry { readonly name: string; readonly version: string; readonly manifestUrl: string }

// ---- the two conversions ----

function need(st: HostStateDict, key: string): Float32Array {
  const t = st.tensors.get(key);
  if (!t) {
    const near = [...st.tensors.keys()].filter((k) => k.startsWith(key.split(".").slice(0, 2).join("."))).slice(0, 8);
    throw new Error(`checkpoint has no "${key}" — nearby: ${near.join(", ")}`);
  }
  return t.data;
}
function bn(st: HostStateDict, prefix: string): BatchNorm {
  return { weight: need(st, `${prefix}.weight`), bias: need(st, `${prefix}.bias`), runningMean: need(st, `${prefix}.running_mean`), runningVar: need(st, `${prefix}.running_var`) };
}

/** timm `efficientnet_b0`: the table bimm's `efficientnetPlan(1, 1)` produces. */
export function efficientnetB0Graph(st: HostStateDict): CpuGraph {
  const stages: readonly [number, number, number, number, number][] = [
    // kernel, expansion, cout, repeats, stride
    [3, 1, 16, 1, 1], [3, 6, 24, 2, 2], [5, 6, 40, 2, 2], [3, 6, 80, 3, 2], [5, 6, 112, 3, 1], [5, 6, 192, 4, 2], [3, 6, 320, 1, 1],
  ];
  const g = new GraphBuilder();
  const x = g.input(3);
  let h = g.conv(x, { weight: need(st, "conv_stem.weight"), cout: 32, cin: 3, k: 3, stride: 2, pad: 1, bn: bn(st, "bn1"), act: ACT.swish });
  let cin = 32;
  stages.forEach(([k, e, cout, repeats, s], si) => {
    for (let i = 0; i < repeats; i++) {
      const stride = i === 0 ? s : 1, p = `blocks.${si}.${i}`, se = Math.round(cin * 0.25), pad = (k - 1) / 2;
      const skip = stride === 1 && cin === cout;
      let d: number, out: number;
      if (si === 0) {
        d = g.dwconv(h, { weight: need(st, `${p}.conv_dw.weight`), cout: cin, cin, k, stride, pad, bn: bn(st, `${p}.bn1`), act: ACT.swish });
        d = g.se(d, need(st, `${p}.se.conv_reduce.weight`), need(st, `${p}.se.conv_reduce.bias`), need(st, `${p}.se.conv_expand.weight`), need(st, `${p}.se.conv_expand.bias`), se);
        out = g.conv(d, { weight: need(st, `${p}.conv_pw.weight`), cout, cin, k: 1, stride: 1, pad: 0, bn: bn(st, `${p}.bn2`) });
      } else {
        const mid = cin * e;
        const ex = g.conv(h, { weight: need(st, `${p}.conv_pw.weight`), cout: mid, cin, k: 1, stride: 1, pad: 0, bn: bn(st, `${p}.bn1`), act: ACT.swish });
        d = g.dwconv(ex, { weight: need(st, `${p}.conv_dw.weight`), cout: mid, cin: mid, k, stride, pad, bn: bn(st, `${p}.bn2`), act: ACT.swish });
        d = g.se(d, need(st, `${p}.se.conv_reduce.weight`), need(st, `${p}.se.conv_reduce.bias`), need(st, `${p}.se.conv_expand.weight`), need(st, `${p}.se.conv_expand.bias`), se);
        out = g.conv(d, { weight: need(st, `${p}.conv_pwl.weight`), cout, cin: mid, k: 1, stride: 1, pad: 0, bn: bn(st, `${p}.bn3`) });
      }
      h = skip ? g.add(out, h) : out;
      cin = cout;
    }
  });
  h = g.conv(h, { weight: need(st, "conv_head.weight"), cout: 1280, cin, k: 1, stride: 1, pad: 0, bn: bn(st, "bn2"), act: ACT.swish });
  const pooled = g.gap(h);
  return g.finish(g.linear(pooled, need(st, "classifier.weight"), need(st, "classifier.bias"), 1000));
}

/** timm `resnet18`: BasicBlock ×[2,2,2,2], the table bimm's `resnetPlan("resnet18")` produces. */
export function resnet18Graph(st: HostStateDict): CpuGraph {
  const g = new GraphBuilder();
  const x = g.input(3);
  let h = g.conv(x, { weight: need(st, "conv1.weight"), cout: 64, cin: 3, k: 7, stride: 2, pad: 3, bn: bn(st, "bn1"), act: ACT.relu });
  h = g.maxpool(h, 3, 2, 1);
  let cin = 64;
  [64, 128, 256, 512].forEach((width, li) => {
    for (let i = 0; i < 2; i++) {
      const stride = i === 0 ? (li === 0 ? 1 : 2) : 1, p = `layer${li + 1}.${i}`;
      const a = g.conv(h, { weight: need(st, `${p}.conv1.weight`), cout: width, cin, k: 3, stride, pad: 1, bn: bn(st, `${p}.bn1`), act: ACT.relu });
      const b = g.conv(a, { weight: need(st, `${p}.conv2.weight`), cout: width, cin: width, k: 3, stride: 1, pad: 1, bn: bn(st, `${p}.bn2`) });
      const shortcut = (stride !== 1 || cin !== width)
        ? g.conv(h, { weight: need(st, `${p}.downsample.0.weight`), cout: width, cin, k: 1, stride, pad: 0, bn: bn(st, `${p}.downsample.1`) })
        : h;
      h = g.add(b, shortcut, ACT.relu);
      cin = width;
    }
  });
  const pooled = g.gap(h);
  return g.finish(g.linear(pooled, need(st, "fc.weight"), need(st, "fc.bias"), 1000));
}

const MODELS: readonly { hub: string; convert: (st: HostStateDict) => CpuGraph }[] = [
  { hub: "imagenet-efficientnet-b0", convert: efficientnetB0Graph },
  { hub: "imagenet-resnet18", convert: resnet18Graph },
];

// ---- the check ----

function seeded(n: number, seed: number): Float32Array {
  let s = seed >>> 0;
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    s ^= s << 13; s >>>= 0; s ^= s >>> 17; s ^= s << 5; s >>>= 0;
    const u = Math.max(s / 4294967296, 1e-9);
    s ^= s << 13; s >>>= 0; s ^= s >>> 17; s ^= s << 5; s >>>= 0;
    out[i] = Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * (s / 4294967296));
  }
  return out;
}

async function median(fn: () => Promise<void> | void, n: number): Promise<number> {
  await fn();
  const t: number[] = [];
  for (let i = 0; i < n; i++) { const s = performance.now(); await fn(); t.push(performance.now() - s); }
  t.sort((a, b) => a - b);
  return t[Math.floor(t.length / 2)] ?? NaN;
}

export async function report(hub: HubLike, indexUrl: string, only: string | null): Promise<{ text: string; checks: Check[] }> {
  const lines: string[] = [];
  const checks: Check[] = [];
  const say = (s: string): void => { lines.push(s); console.log(`[cpu] ${s}`); };
  await init();
  say(`gpu adapter: ${Device.adapterInfo}`);
  const K = await loadKernels();
  say(`cpu kernels: loaded, memory ${(K.memory.buffer.byteLength / 1e6).toFixed(1)} MB`);
  const raw: unknown = await (await fetch(indexUrl)).json();
  // The index is `{ models: [...] }` today and was a bare array once; take either.
  const list: unknown = Array.isArray(raw) ? raw : (typeof raw === "object" && raw !== null ? ((raw as Record<string, unknown>)["models"] ?? (raw as Record<string, unknown>)["entries"]) : []);
  const entries: IndexEntry[] = (Array.isArray(list) ? list : []).filter((r): r is IndexEntry =>
    typeof r === "object" && r !== null && typeof (r as IndexEntry).name === "string" && typeof (r as IndexEntry).manifestUrl === "string");

  for (const m of MODELS) {
    if (only && m.hub !== only) continue;
    const rows = entries.filter((e) => e.name === m.hub);
    const entry = rows[rows.length - 1];
    if (!entry) { checks.push({ name: `${m.hub}: in the registry`, ok: false, note: "not in the index" }); continue; }
    say(`\n== ${entry.name} ${entry.version} ==`);
    const t0 = performance.now();
    const { manifest, model } = await hub.load(entry.manifestUrl, { verify: false });
    model.eval();
    const bytes = await hub.fetchWeights(manifest, entry.manifestUrl);
    say(`loaded on gpu and fetched ${(bytes.length / 1e6).toFixed(1)} MB in ${(performance.now() - t0).toFixed(0)} ms`);
    const st = readSafetensors(bytes);
    const t1 = performance.now();
    const graph = m.convert(st);
    const runner = new CpuRunner(K, graph);
    say(`cpu graph: ${graph.nodes.length} nodes, built and uploaded in ${(performance.now() - t1).toFixed(0)} ms`);

    for (const B of [1, 2]) {
      const arr = seeded(B * 3 * 224 * 224, 7 + B);
      const x = keepAlive(Tensor.from(arr, [B, 3, 224, 224]));
      const gpu = await scope(async () => noGrad(() => model.forward(x)).toArray());
      const cpu = runner.forward(arr, B, 224, 224);
      if (cpu.length !== gpu.length) { checks.push({ name: `${m.hub} b${B}: shape`, ok: false, note: `cpu ${cpu.length} vs gpu ${gpu.length}` }); continue; }
      let worst = 0, scale = 0, argAgree = 0;
      for (let b = 0; b < B; b++) {
        let ag = -1, ac = -1, mg = -Infinity, mc = -Infinity;
        for (let i = 0; i < 1000; i++) {
          const gv = gpu[b * 1000 + i] ?? 0, cv = cpu[b * 1000 + i] ?? 0;
          worst = Math.max(worst, Math.abs(gv - cv)); scale = Math.max(scale, Math.abs(gv));
          if (gv > mg) { mg = gv; ag = i; } if (cv > mc) { mc = cv; ac = i; }
        }
        if (ag === ac) argAgree++;
      }
      const rel = worst / scale;
      const ok = rel < 1e-3 && argAgree === B;
      const note = `max|Δ| ${worst.toExponential(2)} on logits up to ${scale.toFixed(2)} → relative ${rel.toExponential(2)} · argmax agrees ${argAgree}/${B}`;
      say(`batch ${B}: ${note}`);
      checks.push({ name: `${m.hub} b${B}: cpu matches gpu`, ok, note });
    }
    // Time both, batch 1 and 16, medians of five.
    for (const B of [1, 16]) {
      const arr = seeded(B * 3 * 224 * 224, 99);
      const x = keepAlive(Tensor.from(arr, [B, 3, 224, 224]));
      const gpuMs = await median(async () => { await scope(async () => noGrad(() => model.forward(x)).toArray()); }, 5);
      const cpuMs = await median(() => { runner.forward(arr, B, 224, 224); }, B === 1 ? 5 : 3);
      say(`batch ${B}: gpu ${gpuMs.toFixed(1)} ms (${(gpuMs / B).toFixed(2)} ms/image) · cpu ${cpuMs.toFixed(1)} ms (${(cpuMs / B).toFixed(1)} ms/image) · wasm memory ${(K.memory.buffer.byteLength / 1e6).toFixed(0)} MB`);
    }
  }
  const failed = checks.filter((c) => !c.ok).length;
  say(`\n${checks.length - failed} / ${checks.length} checks passed`);
  return { text: lines.join("\n"), checks };
}
