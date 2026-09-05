/**
 * Many workers, one memory: the protocol by which a forward hands kernel calls to a pool.
 *
 * ## The shape
 *
 * The kernels are stateless functions over offsets into one linear memory, so a forward
 * parallelises by rows: a GEMM's rows in `P` slices, a depthwise convolution's images one
 * per worker, an im2col block and its GEMM as one unit. What this file adds is the least
 * that makes that possible in a browser: the memory is a **shared** `WebAssembly.Memory`
 * that every worker instantiates the same module over, the main side allocates and the
 * workers only compute, and the hand-off is `Atomics` on a small control block rather
 * than `postMessage` — a forward is a hundred-odd kernel calls, and a hundred message
 * round trips at 50–100 µs each would cost more than the forward.
 *
 * ## The control block
 *
 * `ctrl` (Int32Array on a SharedArrayBuffer): `[0]` generation, `[1]` workers done with
 * the current generation, `[2]` task count, `[3]` stop. `data` (Float64Array on another):
 * `MAX_TASKS` slots of `SLOT` doubles — a call count, then up to `MAX_CALLS` calls of
 * `[fnId, nArgs, args…]`, `fnId` an index into `KERNELS_EXPORTS`. The main side writes the
 * tasks, bumps the generation and notifies; worker `id` takes every task `t` with
 * `t % workers === id`, runs its calls in order, adds one to done and notifies; the main
 * side waits until done equals the worker count. Task `t` always lands on worker
 * `t % workers`, and the runner relies on that to give each worker its own scratch buffer.
 *
 * ## What it needs from the page
 *
 * `SharedArrayBuffer`, which a browser gives only to pages served with
 * `Cross-Origin-Opener-Policy: same-origin` and `Cross-Origin-Embedder-Policy: require-corp`.
 * A page opened from `file://` cannot have them. `Atomics.wait` is refused on a browser's
 * main thread, so the main side there spins on `Atomics.load` instead — the runner usually
 * lives in a worker anyway (Pyodide's), where waiting is allowed.
 *
 * This file is the protocol and the two loops. Spawning the workers is the host's: Web
 * Workers in a page, `worker_threads` in node — `borch-ts/test/threads_node.mjs` does the
 * second for the measurement.
 */
import { KERNELS_EXPORTS } from "./kernels.js";

export type KernelName = (typeof KERNELS_EXPORTS)[number];
/** One kernel call: the export's name, then its numeric arguments. */
export type Call = readonly [KernelName, ...number[]];
/** Calls one worker runs in order — an im2col block and its GEMM, for instance. */
export type Task = readonly Call[];

export interface Dispatcher {
  readonly workers: number;
  /** Runs every task, task `t` on worker `t % workers`, and returns when all are done. */
  run(tasks: readonly Task[]): void;
}

export const MAX_TASKS = 256;
export const MAX_CALLS = 4;
export const MAX_ARGS = 14;
const CALL = 2 + MAX_ARGS;
export const SLOT = 1 + MAX_CALLS * CALL;
const GEN = 0, DONE = 1, COUNT = 2, STOP = 3;

const FN_ID: ReadonlyMap<string, number> = new Map(KERNELS_EXPORTS.map((name, i) => [name, i]));

/** The two shared buffers a pool is built on. */
export function makeControl(): { ctrl: Int32Array<SharedArrayBuffer>; data: Float64Array<SharedArrayBuffer> } {
  return {
    ctrl: new Int32Array(new SharedArrayBuffer(4 * 8)),
    data: new Float64Array(new SharedArrayBuffer(8 * MAX_TASKS * SLOT)),
  };
}

/** Main side: encode the tasks, wake the workers, wait for all of them. */
export class MainSide implements Dispatcher {
  constructor(
    private readonly ctrl: Int32Array<SharedArrayBuffer>,
    private readonly data: Float64Array<SharedArrayBuffer>,
    readonly workers: number,
  ) {}

  run(tasks: readonly Task[]): void {
    if (tasks.length > MAX_TASKS) throw new Error(`cpu threads: ${tasks.length} tasks, the block holds ${MAX_TASKS}`);
    const { ctrl, data } = this;
    tasks.forEach((task, t) => {
      if (task.length > MAX_CALLS) throw new Error(`cpu threads: task of ${task.length} calls, at most ${MAX_CALLS}`);
      const base = t * SLOT;
      data[base] = task.length;
      task.forEach((call, c) => {
        const [name, ...args] = call;
        const id = FN_ID.get(name);
        if (id === undefined) throw new Error(`cpu threads: no kernel named ${name}`);
        if (args.length > MAX_ARGS) throw new Error(`cpu threads: ${name} with ${args.length} arguments, at most ${MAX_ARGS}`);
        const at = base + 1 + c * CALL;
        data[at] = id; data[at + 1] = args.length;
        for (let i = 0; i < args.length; i++) data[at + 2 + i] = args[i] ?? 0;
      });
    });
    Atomics.store(ctrl, DONE, 0);
    Atomics.store(ctrl, COUNT, tasks.length);
    Atomics.add(ctrl, GEN, 1);
    Atomics.notify(ctrl, GEN);
    // Wait for every worker. A browser's main thread may not `Atomics.wait`; it spins.
    for (;;) {
      const done = Atomics.load(ctrl, DONE);
      if (done >= this.workers) break;
      try { Atomics.wait(ctrl, DONE, done, 50); } catch { /* main thread: spin */ }
    }
  }

  /** Tell the workers to leave their loops. */
  stop(): void {
    Atomics.store(this.ctrl, STOP, 1);
    Atomics.add(this.ctrl, GEN, 1);
    Atomics.notify(this.ctrl, GEN);
  }
}

/**
 * Worker side: the loop a worker runs for its whole life. `exports` is its own instance of
 * the kernel module over the shared memory; `id` decides which tasks are its.
 */
export function workerLoop(ctrl: Int32Array<SharedArrayBuffer>, data: Float64Array<SharedArrayBuffer>, exports: WebAssembly.Exports, id: number, workers: number): void {
  const fns: ((...args: number[]) => number)[] = KERNELS_EXPORTS.map((name) => {
    const f = exports[name];
    if (typeof f !== "function") throw new Error(`cpu threads: worker ${id} has no export ${name}`);
    return f as (...args: number[]) => number;
  });
  let seen = Atomics.load(ctrl, GEN);
  for (;;) {
    Atomics.wait(ctrl, GEN, seen);
    if (Atomics.load(ctrl, STOP)) return;
    const gen = Atomics.load(ctrl, GEN);
    if (gen === seen) continue;
    seen = gen;
    const count = Atomics.load(ctrl, COUNT);
    for (let t = id; t < count; t += workers) {
      const base = t * SLOT;
      const calls = data[base] ?? 0;
      for (let c = 0; c < calls; c++) {
        const at = base + 1 + c * CALL;
        const f = fns[data[at] ?? 0];
        const n = data[at + 1] ?? 0;
        const args: number[] = [];
        for (let i = 0; i < n; i++) args.push(data[at + 2 + i] ?? 0);
        if (f) f(...args);
      }
    }
    Atomics.add(ctrl, DONE, 1);
    Atomics.notify(ctrl, DONE);
  }
}
