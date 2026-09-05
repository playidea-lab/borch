// Two things the pool has to survive, checked in node (worker_threads stand in for Web Workers):
//
//   1. **More tasks than one dispatch holds, on a worker count that does not divide it.** A
//      convolution is cut into im2col blocks, each worker owns one column buffer, and the runner
//      hands blocks to the pool MAX_TASKS at a time. Worker `t % P` must use buffer `t % P` in
//      *every* chunk — with three workers and 300 blocks the second chunk once mismatched, and
//      two workers wrote one buffer. The pool's output has to equal one thread's to the bit.
//   2. **A worker that traps.** A bad offset traps inside the worker; the main side must be told
//      and throw, not wait for a count that never arrives.
//
//   node borch-ts/test/threads_check.mjs      → one JSON line: {"checks":[{name, ok, note}]}
import { Worker, isMainThread, workerData } from "node:worker_threads";
import { fileURLToPath } from "node:url";
import { ACT, kernelBytes, kernelsFromExports } from "../dist/src/cpu/load.js";
import { GraphBuilder } from "../dist/src/cpu/graph.js";
import { CpuRunner } from "../dist/src/cpu/runner.js";
import { CONTROL_LAYOUT, MainSide, makeControl, workerLoop } from "../dist/src/cpu/threads.js";

if (!isMainThread) {
  const { ctrl, data, memory, bytes, id, workers, stackTop } = workerData;
  const instance = new WebAssembly.Instance(new WebAssembly.Module(bytes), { env: { memory } });
  instance.exports.__stack_pointer.value = stackTop;
  try { workerLoop(ctrl, data, instance.exports, id, workers, CONTROL_LAYOUT); } catch { /* the flag was raised; the worker leaves */ }
  process.exit(0);
}

const checks = [];
const bytes = kernelBytes("shared");
const memory = new WebAssembly.Memory({ initial: 1024, maximum: 16384, shared: true });
const K = kernelsFromExports(new WebAssembly.Instance(new WebAssembly.Module(bytes), { env: { memory } }).exports, "shared", memory);

function spawn(P) {
  const { ctrl, data } = makeControl();
  const STACK = 256 * 1024;
  const workers = Array.from({ length: P }, (_, id) => new Worker(fileURLToPath(import.meta.url), { workerData: { ctrl, data, memory, bytes, id, workers: P, stackTop: K.alloc(STACK) + STACK } }));
  const main = new MainSide(ctrl, data, P);
  return { pool: main, stop: async () => { main.stop(); for (const w of workers) await w.terminate(); }, ctrl, workers };
}

let seed = 3;
const rnd = () => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed / 0x7fffffff - 0.5; };
const gauss = (n, s) => Float32Array.from({ length: n }, () => rnd() * s);

if (!process.argv.includes("--trap-child")) // ---- 1. one 3×3 convolution, 300 images of 96×96×16: 300 blocks, three workers ----
{
  const B = 300, H = 96, C = 16;
  const g = new GraphBuilder();
  const x0 = g.input(C);
  const y = g.conv(x0, { weight: gauss(C * C * 9, 0.1), cout: C, cin: C, k: 3, stride: 1, pad: 1, act: ACT.relu });
  const graph = g.finish(g.gap(y));
  const input = gauss(B * C * H * H, 1);
  const direct = new CpuRunner(K, graph).forward(input, B, H, H);
  const { pool, stop } = spawn(3);
  await new Promise((r) => setTimeout(r, 100));
  const got = new CpuRunner(K, graph, pool).forward(input, B, H, H);
  let worst = 0;
  for (let i = 0; i < got.length; i++) worst = Math.max(worst, Math.abs(got[i] - direct[i]));
  checks.push({ name: "300 conv blocks on 3 workers match one thread to the bit", ok: worst === 0 && got.length === direct.length, note: `max|Δ| ${worst.toExponential(1)} over ${got.length} values` });
  await stop();
}

// ---- 2. a worker traps: the main side throws instead of waiting forever ----
// In a child process: `run` spins synchronously, so no timer in this process could catch a
// hang. A child that does not answer within ten seconds is the hang, reported as such.
if (process.argv.includes("--trap-child")) {
  const { pool, stop } = spawn(2);
  await new Promise((r) => setTimeout(r, 100));
  const t0 = performance.now();
  let threw = null;
  try { pool.run([[["gemm_bias_act", 4, 16, 16, 0x7fff0000, 0, 0x7fff0000, 0, 0]]]); } catch (e) { threw = e; }
  process.stdout.write(JSON.stringify({ threw: threw ? String(threw.message) : null, ms: Math.round(performance.now() - t0) }) + "\n");
  await stop();
  process.exit(0);
}
{
  const { spawnSync } = await import("node:child_process");
  const child = spawnSync(process.execPath, [fileURLToPath(import.meta.url), "--trap-child"], { encoding: "utf8", timeout: 10_000 });
  const line = (child.stdout || "").trim().split("\n").pop();
  let answer = null;
  try { answer = line ? JSON.parse(line) : null; } catch { answer = null; }
  if (child.error && child.error.code === "ETIMEDOUT") checks.push({ name: "a trapping worker is reported", ok: false, note: "the main side waited 10 s — it hangs" });
  else checks.push({ name: "a trapping worker is reported", ok: !!(answer && answer.threw), note: answer ? (answer.threw ? `threw in ${answer.ms} ms: ${answer.threw.slice(0, 90)}` : "returned normally") : `no answer: ${(child.stderr || "").slice(-200)}` });
}

process.stdout.write(JSON.stringify({ checks }) + "\n");
process.exit(checks.every((c) => c.ok) ? 0 : 1);
