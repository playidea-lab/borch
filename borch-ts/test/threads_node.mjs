// The cpu device on a pool of workers over one shared memory, measured in node. EfficientNet-B0
// and ResNet-18 forwards with random weights, direct and on pools of 1/2/4/8 workers, each pool's
// output compared with the direct one (they must match to the bit).
//
// Node's worker_threads stand in for Web Workers; the protocol (src/cpu/threads.ts) is the one a
// page uses through WorkerPool. The kernel module is the `shared` flavor from kernels.ts — the
// relaxed kernels linked over an imported, shared memory (build.py, SHARED_LINK). `__stack_pointer`
// is exported so each worker is given its own stack region before it runs: Rust's shadow stack is
// in linear memory and every instance starts it at the same address. Without that the workers
// spill over each other's frames — measured as results that differed from the direct forward by
// 1e-5 and changed between runs.
//
//   node borch-ts/test/threads_node.mjs        (env: WORKERS=1,2,4,8 BATCHES=16,4 REPS=5)
import { Worker, isMainThread, workerData } from "node:worker_threads";
import { fileURLToPath } from "node:url";
import { kernelBytes, kernelsFromExports } from "../dist/src/cpu/load.js";
import { CpuRunner } from "../dist/src/cpu/runner.js";
import { CONTROL_LAYOUT, MainSide, makeControl, workerLoop } from "../dist/src/cpu/threads.js";
import { cpuGraphFor } from "bimm-ts";

if (!isMainThread) {
  const { ctrl, data, memory, bytes, id, workers, stackTop } = workerData;
  const instance = new WebAssembly.Instance(new WebAssembly.Module(bytes), { env: { memory } });
  // Rust's shadow stack lives in linear memory; every instance's __stack_pointer starts at the
  // same address, so without this the workers would spill registers over each other's frames.
  instance.exports.__stack_pointer.value = stackTop;
  workerLoop(ctrl, data, instance.exports, id, workers, CONTROL_LAYOUT);
  process.exit(0);
}

function fake(shapes) { const t = new Map(); let seed = 1; const rnd = () => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed / 0x7fffffff; }; for (const [k, shape] of Object.entries(shapes)) t.set(k, { shape, data: Float32Array.from({ length: shape.reduce((a, b) => a * b, 1) }, () => (rnd() - 0.5) * 0.1) }); for (const k of [...t.keys()]) if (k.endsWith("running_var")) t.get(k).data.fill(1); return { tensors: t, metadata: {} }; }
function bn(p, c) { return { [`${p}.weight`]: [c], [`${p}.bias`]: [c], [`${p}.running_mean`]: [c], [`${p}.running_var`]: [c] }; }
const r = { "conv1.weight": [64, 3, 7, 7], ...bn("bn1", 64), "fc.weight": [1000, 512], "fc.bias": [1000] };
{ let cin = 64; [64, 128, 256, 512].forEach((w, li) => { for (let i = 0; i < 2; i++) { const p = `layer${li + 1}.${i}`; Object.assign(r, { [`${p}.conv1.weight`]: [w, cin, 3, 3], ...bn(`${p}.bn1`, w), [`${p}.conv2.weight`]: [w, w, 3, 3], ...bn(`${p}.bn2`, w) }); if (i === 0 && (li > 0)) Object.assign(r, { [`${p}.downsample.0.weight`]: [w, cin, 1, 1], ...bn(`${p}.downsample.1`, w) }); cin = w; } }); }
const e = { "conv_stem.weight": [32, 3, 3, 3], ...bn("bn1", 32), "conv_head.weight": [1280, 320, 1, 1], ...bn("bn2", 1280), "classifier.weight": [1000, 1280], "classifier.bias": [1000] };
{ let cin = 32; const ST = [[3, 1, 16, 1, 1], [3, 6, 24, 2, 2], [5, 6, 40, 2, 2], [3, 6, 80, 3, 2], [5, 6, 112, 3, 1], [5, 6, 192, 4, 2], [3, 6, 320, 1, 1]];
  ST.forEach(([k, ex, cout, reps, s], si) => { for (let i = 0; i < reps; i++) { const p = `blocks.${si}.${i}`, se = Math.round(cin * 0.25);
    if (si === 0) Object.assign(e, { [`${p}.conv_dw.weight`]: [cin, 1, k, k], ...bn(`${p}.bn1`, cin), [`${p}.se.conv_reduce.weight`]: [se, cin, 1, 1], [`${p}.se.conv_reduce.bias`]: [se], [`${p}.se.conv_expand.weight`]: [cin, se, 1, 1], [`${p}.se.conv_expand.bias`]: [cin], [`${p}.conv_pw.weight`]: [cout, cin, 1, 1], ...bn(`${p}.bn2`, cout) });
    else { const mid = cin * ex; Object.assign(e, { [`${p}.conv_pw.weight`]: [mid, cin, 1, 1], ...bn(`${p}.bn1`, mid), [`${p}.conv_dw.weight`]: [mid, 1, k, k], ...bn(`${p}.bn2`, mid), [`${p}.se.conv_reduce.weight`]: [se, mid, 1, 1], [`${p}.se.conv_reduce.bias`]: [se], [`${p}.se.conv_expand.weight`]: [mid, se, 1, 1], [`${p}.se.conv_expand.bias`]: [mid], [`${p}.conv_pwl.weight`]: [cout, mid, 1, 1], ...bn(`${p}.bn3`, cout) }); }
    cin = cout; } }); }

const bytes = kernelBytes("shared");
const memory = new WebAssembly.Memory({ initial: 1024, maximum: 16384, shared: true });
const main = new WebAssembly.Instance(new WebAssembly.Module(bytes), { env: { memory } });
const K = kernelsFromExports(main.exports, "relaxed", memory);
const graphs = { "effnet-b0": cpuGraphFor({ library: "timm", factory: "efficientnet_b0" }, fake(e), { numClasses: 1000 }), "resnet-18": cpuGraphFor({ library: "timm", factory: "resnet18" }, fake(r), { numClasses: 1000 }) };
const BATCHES = (process.env.BATCHES ?? "16,4").split(",").map(Number);
const WORKERS = (process.env.WORKERS ?? "1,2,4,8").split(",").map(Number);
const REPS = Number(process.env.REPS ?? 5);
const median = (a) => a.slice().sort((x, y) => x - y)[a.length >> 1];

const inputs = {}; for (const B of BATCHES) inputs[B] = new Float32Array(B * 3 * 224 * 224).map(() => Math.random());
const single = {}; const ref = {};
for (const [name, g] of Object.entries(graphs)) { const runner = new CpuRunner(K, g); for (const B of BATCHES) { runner.forward(inputs[B], B, 224, 224); const ts = []; for (let i = 0; i < REPS; i++) { const t0 = performance.now(); const out = runner.forward(inputs[B], B, 224, 224); ts.push(performance.now() - t0); if (i === 0) ref[`${name}/${B}`] = out; } single[`${name}/${B}`] = median(ts); } }
console.log(`direct (no pool) · ${new Intl.NumberFormat().format(memory.buffer.byteLength / 1e6)} MB`);
for (const [k, v] of Object.entries(single)) console.log(`  ${k.padEnd(14)} ${v.toFixed(0).padStart(6)} ms  ${(v / Number(k.split("/")[1])).toFixed(1).padStart(6)} ms/image`);

for (const P of WORKERS) {
  const { ctrl, data } = makeControl();
  const STACK = 1 << 20; // 1 MB, the same as the linker's default for the main instance
  const workers = Array.from({ length: P }, (_, id) => new Worker(fileURLToPath(import.meta.url), { workerData: { ctrl, data, memory, bytes, id, workers: P, stackTop: K.alloc(STACK) + STACK } }));
  const pool = new MainSide(ctrl, data, P);
  // give the workers a moment to instantiate — the first run's wait covers it either way
  await new Promise((res) => setTimeout(res, 200));
  console.log(`pool ${P} worker${P > 1 ? "s" : ""}`);
  for (const [name, g] of Object.entries(graphs)) {
    const runner = new CpuRunner(K, g, pool);
    for (const B of BATCHES) {
      runner.forward(inputs[B], B, 224, 224);
      const ts = []; let maxd = 0;
      for (let i = 0; i < REPS; i++) { const t0 = performance.now(); const out = runner.forward(inputs[B], B, 224, 224); ts.push(performance.now() - t0); if (i === 0) { const r0 = ref[`${name}/${B}`]; for (let j = 0; j < out.length; j++) maxd = Math.max(maxd, Math.abs(out[j] - r0[j])); } }
      const t = median(ts), s = single[`${name}/${B}`];
      console.log(`  ${`${name}/${B}`.padEnd(14)} ${t.toFixed(0).padStart(6)} ms  ${(t / B).toFixed(1).padStart(6)} ms/image  ×${(s / t).toFixed(2)} vs direct  max|Δ| ${maxd.toExponential(1)}`);
    }
  }
  pool.stop();
  await Promise.all(workers.map((w) => new Promise((res) => w.on("exit", res))));
}
console.log(`wasm memory ${(memory.buffer.byteLength / 1e6).toFixed(0)} MB`);
