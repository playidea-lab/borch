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
 * This file is the protocol, the two loops, and `WorkerPool`, which spawns Web Workers from
 * a blob URL and hands each the compiled module, the memory and its own stack. In node the
 * loop is driven from `worker_threads` instead — `borch-ts/test/threads_node.mjs`.
 */
import { KERNELS_EXPORTS } from "./kernels.js";
import { loadKernels, sharedMemoryAvailable, type CpuKernels, type LoadKernelsOptions } from "./load.js";

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
export const MAX_ARGS = 16;
const CALL = 2 + MAX_ARGS;
export const SLOT = 1 + MAX_CALLS * CALL;
const GEN = 0, DONE = 1, COUNT = 2, STOP = 3;

const FN_ID: ReadonlyMap<string, number> = new Map(KERNELS_EXPORTS.map((name, i) => [name, i]));

/**
 * Everything the worker loop needs to know about the control block, as data. The loop
 * takes it as a parameter and touches nothing else from this module, so that its source
 * (`workerLoop.toString()`) can be put in a worker script as it is.
 */
export interface Layout {
  readonly names: readonly string[];
  readonly slot: number; readonly call: number;
  readonly gen: number; readonly done: number; readonly count: number; readonly stop: number;
}
export const CONTROL_LAYOUT: Layout = { names: KERNELS_EXPORTS, slot: SLOT, call: CALL, gen: GEN, done: DONE, count: COUNT, stop: STOP };

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
 * the kernel module over the shared memory; `id` decides which tasks are its. Only its
 * parameters and the globals `Atomics` — see `Layout`.
 */
export function workerLoop(ctrl: Int32Array<SharedArrayBuffer>, data: Float64Array<SharedArrayBuffer>, exports: WebAssembly.Exports, id: number, workers: number, layout: Layout): void {
  const fns: ((...args: number[]) => number)[] = layout.names.map((name) => {
    const f = exports[name];
    if (typeof f !== "function") throw new Error(`cpu threads: worker ${id} has no export ${name}`);
    return f as (...args: number[]) => number;
  });
  let seen = Atomics.load(ctrl, layout.gen);
  for (;;) {
    Atomics.wait(ctrl, layout.gen, seen);
    if (Atomics.load(ctrl, layout.stop)) return;
    const gen = Atomics.load(ctrl, layout.gen);
    if (gen === seen) continue;
    seen = gen;
    const count = Atomics.load(ctrl, layout.count);
    for (let t = id; t < count; t += workers) {
      const base = t * layout.slot;
      const calls = data[base] ?? 0;
      for (let c = 0; c < calls; c++) {
        const at = base + 1 + c * layout.call;
        const f = fns[data[at] ?? 0];
        const n = data[at + 1] ?? 0;
        const args: number[] = [];
        for (let i = 0; i < n; i++) args.push(data[at + 2 + i] ?? 0);
        if (f) f(...args);
      }
    }
    Atomics.add(ctrl, layout.done, 1);
    Atomics.notify(ctrl, layout.done);
  }
}

/** Each worker's shadow stack. The kernels keep almost everything in registers; 256 KB is generous. */
export const WORKER_STACK_BYTES = 256 * 1024;

/**
 * Whether a pool can be spawned here: a shared memory (`sharedMemoryAvailable`) and
 * `Worker`. Node has the first and not the second — there `worker_threads` drives
 * `workerLoop` directly.
 */
export function threadsAvailable(): boolean {
  return typeof Worker !== "undefined" && sharedMemoryAvailable();
}

/** Half the hardware threads, at least one, at most eight — the pools measured past eight gained little. */
export function defaultWorkers(): number {
  const hw = typeof navigator !== "undefined" ? navigator.hardwareConcurrency : undefined;
  return Math.max(1, Math.min(8, Math.floor((hw ?? 2) / 2)));
}

/**
 * The worker's whole script. The loop's source is pasted in — it depends on nothing but its
 * parameters, which is what `Layout` is for. The message carries the compiled module, the
 * shared memory, the two control buffers, the worker's id and its stack top; the worker
 * sets `__stack_pointer`, says it is ready, and enters the loop for good.
 */
function workerSource(): string {
  return `"use strict";
const loop = (${workerLoop.toString()});
self.onmessage = (e) => {
  const m = e.data;
  const instance = new WebAssembly.Instance(m.module, { env: { memory: m.memory } });
  instance.exports.__stack_pointer.value = m.stackTop;
  self.postMessage("ready");
  loop(m.ctrl, m.data, instance.exports, m.id, m.workers, m.layout);
};
`;
}

/**
 * A pool of Web Workers over the shared kernel module. `spawn` loads (or is given) the
 * `shared` kernels, allocates a stack per worker out of that memory, starts the workers
 * from a blob URL and waits until every one has instantiated. Hand the pool to
 * `new CpuRunner(pool.kernels, graph, pool)`; `terminate()` when done with it.
 */
export class WorkerPool implements Dispatcher {
  private constructor(
    readonly kernels: CpuKernels,
    private readonly main: MainSide,
    private readonly handles: Worker[],
    private readonly url: string,
  ) {}

  get workers(): number { return this.main.workers; }

  static async spawn(workers: number, opts: LoadKernelsOptions & { kernels?: CpuKernels } = {}): Promise<WorkerPool> {
    if (!threadsAvailable()) throw new Error("cpu threads: no Worker or no shared memory in this context — a cross-origin isolated page is needed (COOP same-origin, COEP require-corp)");
    if (!Number.isInteger(workers) || workers < 1) throw new Error(`cpu threads: ${workers} workers`);
    const kernels = opts.kernels ?? await loadKernels({ ...opts, shared: true });
    if (kernels.flavor !== "shared" || !kernels.module) throw new Error(`cpu threads: the pool needs the shared kernels, got ${kernels.flavor}`);
    const { ctrl, data } = makeControl();
    const url = URL.createObjectURL(new Blob([workerSource()], { type: "text/javascript" }));
    const handles: Worker[] = [];
    try {
      const ready = Array.from({ length: workers }, (_, id) => new Promise<void>((resolve, reject) => {
        const w = new Worker(url);
        handles.push(w);
        w.onmessage = () => resolve();
        w.onerror = (e) => reject(new Error(`cpu threads: worker ${id} failed to start — ${e.message}`));
        const stackTop = kernels.alloc(WORKER_STACK_BYTES) + WORKER_STACK_BYTES;
        w.postMessage({ ctrl, data, module: kernels.module, memory: kernels.memory, id, workers, stackTop, layout: CONTROL_LAYOUT });
      }));
      await Promise.all(ready);
    } catch (e) {
      for (const w of handles) w.terminate();
      URL.revokeObjectURL(url);
      throw e;
    }
    return new WorkerPool(kernels, new MainSide(ctrl, data, workers), handles, url);
  }

  run(tasks: readonly Task[]): void { this.main.run(tasks); }

  /** Stops the loops and the workers. The kernels and their memory stay usable on this thread. */
  terminate(): void {
    this.main.stop();
    for (const w of this.handles) w.terminate();
    URL.revokeObjectURL(this.url);
  }
}
