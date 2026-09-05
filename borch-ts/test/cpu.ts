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
 * ## The plans live in bimm
 *
 * `bimm-ts` owns the architectures, and since 0.11.0 its plan tables build the CPU
 * graph too: `cpuGraphFor(name, st, opts)` beside the `efficientnetPlan` and
 * `resnetPlan` that build the `nn.Module`s. This file carried a copy of those tables
 * for a day to prove the device before bimm could depend on it; the copy is gone, and
 * what is checked here is bimm's conversion against the GPU model bimm also built.
 * The page imports bimm (esm.sh, pinned) and hands the function in, the way it hands
 * the hub in — this package does not depend on bimm.
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
import { loadKernels, relaxedSimdAvailable } from "../src/cpu/load.js";
import { WorkerPool, threadsAvailable, defaultWorkers } from "../src/cpu/threads.js";
import type { CpuGraph } from "../src/cpu/graph.js";
import { CpuRunner } from "../src/cpu/runner.js";
import { readSafetensors, type HostStateDict } from "../src/cpu/safetensors.js";
import { LinearHead, cosineNeighbours } from "../src/cpu/train.js";
import { nn, optim } from "../src/index.js";

interface Check { name: string; ok: boolean; note: string }

/** What this check uses of `borch-hub` — passed in by the page, which imports the real thing. */
export interface HubLike {
  load(manifestUrl: string, opts: { verify: boolean }): Promise<{ manifest: unknown; model: ModelLike }>;
  fetchWeights(manifest: unknown, manifestUrl: string): Promise<Uint8Array>;
}
export interface ModelLike { eval(): unknown; forward(x: Tensor): Tensor; forwardFeatures(x: Tensor): Tensor; forwardHead(h: Tensor, preLogits: boolean): Tensor }

interface IndexEntry { readonly name: string; readonly version: string; readonly manifestUrl: string }

// ---- what the page hands in from bimm ----

/** `cpuGraphFor` as bimm-ts 0.11.0 exports it. Typed here so this package does not import bimm. */
export interface BimmLike {
  cpuGraphFor(name: { library: string; factory: string }, st: HostStateDict, opts: { numClasses: number; features?: boolean }): CpuGraph;
}

const MODELS: readonly { hub: string; factory: string }[] = [
  { hub: "imagenet-efficientnet-b0", factory: "efficientnet_b0" },
  { hub: "imagenet-resnet18", factory: "resnet18" },
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

export async function report(hub: HubLike, bimm: BimmLike, indexUrl: string, only: string | null): Promise<{ text: string; checks: Check[] }> {
  const lines: string[] = [];
  const checks: Check[] = [];
  const say = (s: string): void => { lines.push(s); console.log(`[cpu] ${s}`); };
  await init();
  say(`gpu adapter: ${Device.adapterInfo}`);
  const K = await loadKernels();
  say(`cpu kernels: ${K.flavor} (relaxed SIMD ${relaxedSimdAvailable() ? "accepted" : "not accepted"} by this engine), memory ${(K.memory.buffer.byteLength / 1e6).toFixed(1)} MB`);
  // The other flavor too, so the two are timed side by side below.
  const Kother = relaxedSimdAvailable() ? await loadKernels({ relaxed: K.flavor !== "relaxed" }) : null;
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
    const graph = bimm.cpuGraphFor({ library: "timm", factory: m.factory }, st, { numClasses: 1000 });
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
    // Time both, batch 1 and 16, medians of five — and the other kernel flavor beside it.
    const other = Kother ? new CpuRunner(Kother, graph) : null;
    for (const B of [1, 16]) {
      const arr = seeded(B * 3 * 224 * 224, 99);
      const x = keepAlive(Tensor.from(arr, [B, 3, 224, 224]));
      const gpuMs = await median(async () => { await scope(async () => noGrad(() => model.forward(x)).toArray()); }, 5);
      const cpuMs = await median(() => { runner.forward(arr, B, 224, 224); }, B === 1 ? 5 : 3);
      const otherMs = other ? await median(() => { other.forward(arr, B, 224, 224); }, B === 1 ? 5 : 3) : NaN;
      const otherNote = other && Kother ? ` · cpu ${Kother.flavor} ${otherMs.toFixed(1)} ms (${(otherMs / B).toFixed(1)} ms/image)` : "";
      say(`batch ${B}: gpu ${gpuMs.toFixed(1)} ms (${(gpuMs / B).toFixed(2)} ms/image) · cpu ${K.flavor} ${cpuMs.toFixed(1)} ms (${(cpuMs / B).toFixed(1)} ms/image)${otherNote} · wasm memory ${(K.memory.buffer.byteLength / 1e6).toFixed(0)} MB`);
    }
    if (other && Kother) {
      // The two flavors have to agree with each other as closely as either agrees with the GPU.
      const arr = seeded(2 * 3 * 224 * 224, 5);
      const a = runner.forward(arr, 2, 224, 224), b = other.forward(arr, 2, 224, 224);
      let worst = 0, scale = 0;
      for (let i = 0; i < a.length; i++) { worst = Math.max(worst, Math.abs((a[i] ?? 0) - (b[i] ?? 0))); scale = Math.max(scale, Math.abs(a[i] ?? 0)); }
      const rel = worst / scale;
      say(`${K.flavor} vs ${Kother.flavor}: max|Δ| ${worst.toExponential(2)} → relative ${rel.toExponential(2)}`);
      checks.push({ name: `${m.hub}: the two kernel flavors agree`, ok: rel < 1e-3, note: `relative ${rel.toExponential(2)}` });
    }
  }
  // ---- the workbench's second half: features → a head → neighbours ----
  if (!only || only === "imagenet-efficientnet-b0") {
    say(`\n== features, a head, neighbours (imagenet-efficientnet-b0) ==`);
    const entry = entries.filter((e) => e.name === "imagenet-efficientnet-b0").pop();
    if (!entry) throw new Error("imagenet-efficientnet-b0 left the index between two paragraphs");
    const { manifest, model } = await hub.load(entry.manifestUrl, { verify: false });
    model.eval();
    const st = readSafetensors(await hub.fetchWeights(manifest, entry.manifestUrl));
    const featureRunner = new CpuRunner(K, bimm.cpuGraphFor({ library: "timm", factory: "efficientnet_b0" }, st, { numClasses: 1000, features: true }));
    const N = 48, D = 1280, Kc = 5;
    const arr = seeded(N * 3 * 224 * 224, 1234);
    const x = keepAlive(Tensor.from(arr, [N, 3, 224, 224]));
    const gpuFeat = await scope(async () => noGrad(() => model.forwardHead(model.forwardFeatures(x), true)).toArray());
    const t0 = performance.now();
    const cpuFeat = featureRunner.forward(arr, N, 224, 224);
    const featMs = performance.now() - t0;
    {
      let worst = 0, scale = 0;
      for (let i = 0; i < N * D; i++) { worst = Math.max(worst, Math.abs((gpuFeat[i] ?? 0) - (cpuFeat[i] ?? 0))); scale = Math.max(scale, Math.abs(gpuFeat[i] ?? 0)); }
      const rel = worst / scale;
      const note = `${N} images · max|Δ| ${worst.toExponential(2)} on features up to ${scale.toFixed(2)} → relative ${rel.toExponential(2)} · cpu ${featMs.toFixed(0)} ms`;
      say(`features: ${note}`);
      checks.push({ name: "b0 features: cpu matches gpu", ok: rel < 1e-3 && cpuFeat.length === N * D, note });
    }
    // The head: the same features (the CPU's), the same zero start, the same SGD, on both devices.
    const labels = Array.from({ length: N }, (_, i) => i % Kc);
    const STEPS = 40, LR = 0.05, MU = 0.9;
    const gpuLosses: number[] = [];
    {
      const head = new nn.Linear(D, Kc);
      head.loadStateDict({ weight: Tensor.from(new Float32Array(Kc * D), [Kc, D]), bias: Tensor.from(new Float32Array(Kc), [Kc]) });
      const opt = new optim.SGD(head.parameters(), LR, MU);
      const crit = new nn.CrossEntropyLoss();
      const feats = keepAlive(Tensor.from(cpuFeat, [N, D]));
      const y = keepAlive(Tensor.from(labels, [N], { dtype: "int64" }));
      for (let step = 0; step < STEPS; step++) {
        const l = await scope(async () => { const loss = crit.forward(head.forward(feats), y); opt.zeroGrad(); loss.backward(); opt.step(); return (await loss.toArray())[0] ?? NaN; });
        gpuLosses.push(l);
      }
    }
    const cpuHead = new LinearHead(K, D, Kc, { lr: LR, momentum: MU });
    const t1 = performance.now();
    const cpuLosses = cpuHead.fit(cpuFeat, labels, N, STEPS);
    const fitMs = performance.now() - t1;
    {
      let worst = 0;
      for (let i = 0; i < STEPS; i++) worst = Math.max(worst, Math.abs((gpuLosses[i] ?? 0) - (cpuLosses[i] ?? 0)) / Math.max(1e-6, Math.abs(gpuLosses[i] ?? 0)));
      const note = `${STEPS} steps · loss ${cpuLosses[0]?.toFixed(4)} → ${cpuLosses[STEPS - 1]?.toFixed(4)} (gpu ${gpuLosses[0]?.toFixed(4)} → ${gpuLosses[STEPS - 1]?.toFixed(4)}) · worst relative step gap ${worst.toExponential(2)} · cpu fit ${fitMs.toFixed(0)} ms`;
      say(`head: ${note}`);
      checks.push({ name: "linear head: cpu loss trajectory matches gpu", ok: worst < 1e-3, note });
      const pred = cpuHead.predict(cpuFeat, N);
      let hit = 0; for (let r = 0; r < N; r++) { let best = 0; for (let k = 1; k < Kc; k++) if ((pred[r * Kc + k] ?? 0) > (pred[r * Kc + best] ?? 0)) best = k; if (best === labels[r]) hit++; }
      say(`head: predicts its own training set ${hit}/${N}`);
    }
    // Neighbours against a plain reference on the same features.
    {
      const k = 5;
      const t2 = performance.now();
      const got = cosineNeighbours(K, cpuFeat, N, D, k);
      const nnMs = performance.now() - t2;
      const norms = new Float64Array(N);
      for (let i = 0; i < N; i++) { let ss = 0; for (let d = 0; d < D; d++) ss += (cpuFeat[i * D + d] ?? 0) ** 2; norms[i] = Math.sqrt(ss); }
      let mismatched = 0, worstSim = 0;
      for (let i = 0; i < N; i++) {
        const sims: { j: number; s: number }[] = [];
        for (let j = 0; j < N; j++) { if (j === i) continue; let dot = 0; for (let d = 0; d < D; d++) dot += (cpuFeat[i * D + d] ?? 0) * (cpuFeat[j * D + d] ?? 0); sims.push({ j, s: dot / ((norms[i] ?? 1) * (norms[j] ?? 1)) }); }
        sims.sort((a, b) => b.s - a.s);
        const ref = new Set(sims.slice(0, k).map((e) => e.j));
        for (let q = 0; q < k; q++) {
          if (!ref.has(got.indices[i * k + q] ?? -1)) mismatched++;
          worstSim = Math.max(worstSim, Math.abs((got.sims[i * k + q] ?? 0) - (sims[q]?.s ?? 0)));
        }
      }
      const note = `${N} rows, k=${k} · ${mismatched} of ${N * k} neighbours differ from the reference · sims within ${worstSim.toExponential(2)} · ${nnMs.toFixed(1)} ms`;
      say(`neighbours: ${note}`);
      checks.push({ name: "cosine neighbours: cpu matches reference", ok: mismatched === 0 && worstSim < 1e-4, note });
    }
  }

  // ---- the worker pool: the same forward on N workers must match the single thread to the bit ----
  say(`\n== worker pool ==`);
  say(`cross-origin isolated: ${typeof crossOriginIsolated !== "undefined" ? crossOriginIsolated : "n/a"} · threads available: ${threadsAvailable()} · hardware threads ${navigator.hardwareConcurrency} · default workers ${defaultWorkers()}`);
  if (threadsAvailable() && (!only || only === "imagenet-efficientnet-b0")) {
    const entry = entries.filter((e) => e.name === "imagenet-efficientnet-b0").pop();
    if (!entry) throw new Error("imagenet-efficientnet-b0 left the index");
    const { manifest } = await hub.load(entry.manifestUrl, { verify: false });
    const st = readSafetensors(await hub.fetchWeights(manifest, entry.manifestUrl));
    const graph = bimm.cpuGraphFor({ library: "timm", factory: "efficientnet_b0" }, st, { numClasses: 1000 });
    const direct = new CpuRunner(K, graph);
    const B = 16, arr = seeded(B * 3 * 224 * 224, 7);
    const ref = direct.forward(arr, B, 224, 224);
    const directMs = await median(() => { direct.forward(arr, B, 224, 224); }, 3);
    for (const P of [2, defaultWorkers()].filter((v, i, a) => a.indexOf(v) === i)) {
      const t0 = performance.now();
      const pool = await WorkerPool.spawn(P);
      const spawnMs = performance.now() - t0;
      try {
        const runner = new CpuRunner(pool.kernels, graph, pool);
        const out = runner.forward(arr, B, 224, 224);
        let worst = 0;
        for (let i = 0; i < out.length; i++) worst = Math.max(worst, Math.abs((out[i] ?? 0) - (ref[i] ?? 0)));
        const ms = await median(() => { runner.forward(arr, B, 224, 224); }, 3);
        say(`${P} workers: spawn ${spawnMs.toFixed(0)} ms · b${B} ${ms.toFixed(1)} ms (${(ms / B).toFixed(1)} ms/image, ×${(directMs / ms).toFixed(2)} vs ${K.flavor} on one thread ${directMs.toFixed(1)} ms) · max|Δ| ${worst.toExponential(1)} · shared memory ${(pool.kernels.memory.buffer.byteLength / 1e6).toFixed(0)} MB`);
        checks.push({ name: `worker pool ×${P}: matches one thread to the bit`, ok: worst === 0, note: `max|Δ| ${worst.toExponential(1)}` });
        checks.push({ name: `worker pool ×${P}: faster than one thread`, ok: ms < directMs, note: `×${(directMs / ms).toFixed(2)}` });
      } finally {
        pool.terminate();
      }
    }
  } else {
    say("no worker pool here — the page is not cross-origin isolated (needs COOP same-origin + COEP require-corp)");
    checks.push({ name: "worker pool: available on this page", ok: false, note: "not cross-origin isolated" });
  }

  const failed = checks.filter((c) => !c.ok).length;
  say(`\n${checks.length - failed} / ${checks.length} checks passed`);
  return { text: lines.join("\n"), checks };
}
