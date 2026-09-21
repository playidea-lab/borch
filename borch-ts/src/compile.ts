/**
 * `torch.compiled` and `torch.capture` for JavaScript — **the bookkeeping the Python
 * binding does around a recording, done here** (`docs/COMPILER.md` Step 6).
 *
 * A training step is the same dispatches with the same buffers every time; recorded once
 * (`Device.beginCapture`/`endCapture`, `Capture`), it replays without the JavaScript that
 * built it. What `compiled` adds to that: the first call for a set of input shapes runs
 * the function under a capture on **copies** of its tensor arguments (uploads, so they
 * are the recording's live-ins and are never moved by the memory plan), keeps what it
 * returned, and every later call with those shapes copies its arguments into the same
 * buffers and replays — handing back the same returned objects, which now read the new
 * step. A new shape — the last, shorter batch — is recorded once more and kept beside
 * the first. Numbers among the arguments are part of the shape key: a value the step
 * baked into a kernel cannot change on replay.
 *
 * **What the caller holds.** Python knows every tensor still referenced and tells the
 * fusion and memory passes so; JavaScript cannot. Here what the function *returns* is
 * held — its buffers are neither left unwritten by `fuse` nor moved by `plan` — and
 * anything else made inside the function may be. A value the caller wants after the
 * step is returned from the step; the parameters, the optimizer's state and the inputs
 * are live-ins and stay where they are regardless.
 *
 * `check: true` holds each recording to the eager step on the caller's own function,
 * as the binding does: the live-ins are read back after the recording ran, the
 * recording replayed, the live-ins and the result read again and put back, the function
 * run eagerly once more from the same place, and the two compared buffer by buffer —
 * bit for bit plain; fused, the outputs within `tol` and the state within `stateTol`.
 * A difference throws and names the worst buffer.
 */
import { Capture, Device, type TuneCandidate, type TuneReport } from "./device.js";
import { fuseForInference, Module, quantizeForInt8 } from "./nn.js";
import { device, noGrad, scope, Tensor } from "./tensor.js";

/** What a compiled function may take: tensors, and plain values that become part of the key. */
export type CompiledArg = Tensor | number | string | boolean | null | undefined;

export interface CompiledOptions {
  /** Merge elementwise trees into single kernels (`Capture.fuse`). Default on. */
  readonly fuse?: boolean;
  /** Lay the intermediates into arenas (`Capture.plan`). Default on. */
  readonly plan?: boolean;
  /** Hold each recording to the eager step; costs two steps and a readback per shape. */
  readonly check?: boolean;
  /** Fused, the outputs' relative tolerance against eager. */
  readonly tol?: number;
  /** Fused, the state's — Adam magnifies a rounding on a parameter near zero. */
  readonly stateTol?: number;
  /**
   * **A model's convolutions on the int8 subgroup path** (`docs/INT8.md`), where the
   * adapter has the configuration — refused by name where it has not. Off by default:
   * the int8 forward is held to accuracy, not to torch's logits, and a caller asks for
   * that trade. `Compiled.int8Layers` says how many layers took it.
   */
  readonly int8?: boolean;
  /**
   * **Kernel selection by measurement** (`docs/COMPILER.md` Step 5): before a shape's
   * first recording, one pass collects every choice a rule makes among kernels, each
   * candidate is timed under the profiler, and the fastest is cached for this adapter
   * (in memory and in `localStorage`); the recording then uses the decisions. Default on
   * where the device has `timestamp-query`; the hand rules decide where it has not.
   */
  readonly tune?: boolean;
  /**
   * The step is a pure function of its arguments — it writes no parameter, no optimiser
   * state, no buffer that outlives the call. `compiled(model)` sets it; a function form
   * says so itself. A pure step's first call compiles its kernels side by side instead of
   * one after another, and its tuning runs on a throwaway recording in idle time; a step
   * that writes state is compiled as it runs and tuned at the start of its next call
   * (`docs/FIRST.md` 1b).
   */
  readonly pure?: boolean;
}

export interface CheckReport {
  readonly buffers: number;
  readonly outputs: number;
  readonly differ: number;
  readonly worstRel: number;
}

export interface PlanReport {
  readonly moved: number; readonly released: number; readonly bytesBefore: number; readonly bytesAfter: number; readonly arenas: number;
}

interface Recording<R> { readonly cap: Capture; readonly inputs: readonly CompiledArg[]; readonly out: R }

/** Idle time, where the page offers it (`requestIdleCallback`); the next turn otherwise. */
function idle(f: () => void): void {
  const g = globalThis as { requestIdleCallback?: (cb: () => void) => unknown };
  if (typeof g.requestIdleCallback === "function") g.requestIdleCallback(f);
  else setTimeout(f, 0);
}

/** The tensors in a returned value: one, or an array or object of them. */
function tensorsOf(value: unknown): Tensor[] {
  if (value instanceof Tensor) return [value];
  if (Array.isArray(value)) return value.filter((v): v is Tensor => v instanceof Tensor);
  if (value && typeof value === "object") return Object.values(value as Record<string, unknown>).filter((v): v is Tensor => v instanceof Tensor);
  return [];
}

function keyOf(args: readonly CompiledArg[]): string {
  return args.map((a) => {
    if (a instanceof Tensor) return `t[${a.shape.join(",")}]${a.dtype}`;
    if (a === null) return "null";
    if (a === undefined) return "undef";
    if (typeof a === "number" || typeof a === "string" || typeof a === "boolean") return `v:${typeof a}:${String(a)}`;
    throw new TypeError(`compiled: arguments are tensors or plain values — got ${typeof a}`);
  }).join("|");
}

/**
 * Runs `fn` under a capture and returns the recording with the function's result. The
 * hand-driven form: the caller made the inputs, writes the next batch into them
 * (`copyFrom`) and calls `step.replay()`; the tensors `fn` made read the new step.
 */
export function capture<T>(fn: () => T): { step: Capture; result: T } {
  const d = device();
  d.beginCapture();
  let result: T;
  try {
    result = fn();
  } catch (err) {
    d.endCapture().dispose();
    throw err;
  }
  return { step: d.endCapture(), result };
}

/**
 * `capture` for a step that awaits — one that streams a frozen backbone through a window
 * (`streamTrainStep`), whose refills are staging maps. Nothing else may dispatch on the
 * device while the recording is open across those awaits; the recording carries the
 * refills (`Recorded.refill`) and replays through `step.replayAsync()`.
 */
export async function captureAsync<T>(fn: () => Promise<T>): Promise<{ step: Capture; result: T }> {
  const d = device();
  d.beginCapture();
  let result: T;
  try {
    result = await fn();
  } catch (err) {
    d.endCapture().dispose();
    throw err;
  }
  return { step: d.endCapture(), result };
}

/** The costs of a first call, ms of wall — `Compiled.firstCall`. */
export interface FirstCallCost {
  /** Recording the step: the JavaScript of one eager run, its dispatches encoded — and,
   *  for a pure step, the waves of its own kernels compiling side by side. */
  record: number;
  /** Whether the tuning ran after the first call returned (a pure step, idle time; a
   *  state-writing step, its next call) — then `tuning`, `compile`, `rerecord` and
   *  `candidates` are filled in when it has. */
  deferred: boolean;
  /** Waiting for that run's GPU work and reading its outputs back — the step's own first
   *  execution, paid here before the tuner runs on the same buffers; without a tuner the
   *  same wait comes when the caller reads the result. */
  wait: number;
  /** The tuning pass: every candidate warmed once, then timed. Includes `compile`. */
  tuning: number;
  /** Of `tuning`, the warm wave — the candidates' pipelines compiling (the platform's
   *  cost: ~10 ms a pipeline on NVIDIA Vulkan, ~200 on D3D12, measured 2026-09-21). */
  compile: number;
  /** Recording again with the chosen kernels, where a decision changed a pure step. */
  rerecord: number;
  candidates: number;
}

export class Compiled<A extends CompiledArg[], R> {
  private readonly records = new Map<string, Recording<Awaited<R>>>();
  private readonly fuse: boolean;
  private readonly plan: boolean;
  private readonly check: boolean;
  private readonly tol: number;
  private readonly stateTol: number;
  /** One report per recording when `check` is on. */
  readonly checked: CheckReport[] = [];
  /** One report per recording when `plan` is on. */
  readonly planned: PlanReport[] = [];
  /** Per recording, how many dispatches were replay-invariant and left the replay. */
  readonly hoisted: number[] = [];

  /** How many convolutions took the int8 path (`int8: true`), once prepared; −1 before. */
  int8Layers = -1;
  /** The tuner's decisions, one list per recording (empty where nothing was collected). */
  readonly tuned: TuneReport[][] = [];
  /** What the first call for a shape cost, per recording, in ms of wall: the recording
   *  itself, the tuning pass (0 without one), the re-recording where a decision changed a
   *  pure step (0 where none) — and how many candidates the pass timed. The number the
   *  Step 5 gate is about (`docs/COMPILER.md`): a first load pays this once per shape. */
  readonly firstCall: FirstCallCost[] = [];
  private readonly pure: boolean;
  /** Tuning owed per shape key: the recording's candidate lists (a state-writing step —
   *  closures over its live buffers) or none (a pure step tunes a throwaway recording),
   *  with what the first call had. */
  private readonly pending = new Map<string, { queue: Map<string, readonly TuneCandidate[]> | null; args: A; datas: (Float32Array | null)[]; cost: FirstCallCost }>();
  private readonly tune: boolean;
  /** Runs once before the first recording — the int8 quantisation of a model's weights. */
  private prepare: (() => Promise<void>) | null = null;

  constructor(private readonly fn: (...args: A) => R, opts: CompiledOptions = {}, prepare?: () => Promise<void>) {
    this.prepare = prepare ?? null;
    this.tune = opts.tune ?? true;
    this.pure = opts.pure ?? false;
    this.fuse = opts.fuse ?? true;
    this.plan = opts.plan ?? true;
    this.check = opts.check ?? false;
    this.tol = opts.tol ?? 1e-5;
    this.stateTol = opts.stateTol ?? 1e-2;
  }

  /** How many input shapes have been recorded. */
  get shapes(): number {
    return this.records.size;
  }

  /** The recording for the shapes of `args`, if one exists — for `explain`. */
  recordingOf(...args: A): Capture | undefined {
    return this.records.get(keyOf(args))?.cap;
  }

  /**
   * The step for `args`. Async because the first call for a shape reads the tensor
   * arguments back to make the recording's own copies of them; a replay is synchronous
   * underneath and the promise resolves at once.
   */
  async call(...args: A): Promise<Awaited<R>> {
    const key = keyOf(args);
    let rec = this.records.get(key);
    if (rec) {
      // A state-writing step owes its tuning: it runs here, before the replay, on the
      // recording's own buffers — the caller is done with the last call's outputs by now,
      // and the replay overwrites them anyway.
      if (this.pending.get(key)?.queue) await this.tuneNow(key);
      // A re-recording with the tuner's kernels waits here too: the last call's outputs
      // are the caller's until they ask again, and swapping under them would kill them.
      const fresh = this.replacement.get(key);
      if (fresh) {
        this.replacement.delete(key);
        this.records.get(key)?.cap.dispose();
        this.records.set(key, fresh);
      }
      rec = this.records.get(key) as Recording<Awaited<R>>;
      rec.inputs.forEach((held, i) => {
        if (held instanceof Tensor) held.copyFrom(args[i] as Tensor);
      });
      if (rec.cap.hasRefills) await rec.cap.replayAsync(); else rec.cap.replay();
      return rec.out;
    }
    if (this.prepare) { const p = this.prepare; this.prepare = null; await p(); }
    const datas = await Promise.all(args.map((a) => (a instanceof Tensor ? a.toArray() : Promise.resolve(null))));
    const d = device();
    const tuning = this.tune && Device.canTime;
    const t0 = performance.now();
    // A pure step's first recording compiles ahead and collects nothing (its tuning runs
    // on a throwaway recording later); a state-writing step collects its candidates now,
    // closures over the live recording, and is compiled as it runs.
    const made = await this.record(args, datas, tuning && !this.pure, this.pure);
    const cost: FirstCallCost = { record: performance.now() - t0, deferred: tuning, wait: 0, tuning: 0, compile: 0, rerecord: 0, candidates: 0 };
    this.firstCall.push(cost);
    // A state-writing step's candidates are closures over this recording's buffers, and
    // the passes (fusion, hoisting, the plan) move and release those buffers — so the
    // raw recording stands until its tuning has run at the next call, and the passes
    // follow it there. Measured the other way round (2026-09-21, both NVIDIA cards): a
    // candidate wrote a buffer the plan had released and the pool had handed out again,
    // twice in one dispatch — a validation fault.
    if (tuning && !this.pure) {
      this.records.set(key, made);
      this.pending.set(key, { queue: d.takeTuneQueue(), args, datas, cost });
    } else {
      this.finish(key, made);
      if (this.check) this.checked.push(await this.verify(made.cap, made.inputs, made.out));
      if (tuning) {
        this.pending.set(key, { queue: null, args, datas, cost });
        idle(() => { void this.tuneNow(key); });
      }
    }
    return made.out;
  }

  /** Runs the tuning still owed — every shape's — now. Tests call it; a page may, to
   *  have the decisions before it measures. */
  async settle(): Promise<void> {
    for (const key of [...this.pending.keys()]) await this.tuneNow(key);
    await Promise.all([...this.inflight.values()]);
  }

  /** The tuning owed for `key`, started once — a second asker waits on the first. */
  private tuneNow(key: string): Promise<void> {
    const running = this.inflight.get(key);
    if (running) return running;
    const p = this.tunePending(key).finally(() => { this.inflight.delete(key); });
    this.inflight.set(key, p);
    return p;
  }
  private readonly inflight = new Map<string, Promise<void>>();
  private disposed = false;
  /** A re-recording with the chosen kernels, swapped in at the next call. */
  private readonly replacement = new Map<string, Recording<Awaited<R>>>();

  /**
   * One recording of the step for `args`. `collect` queues every kernel choice for the
   * tuner (the rule's pick runs); `ahead` compiles the kernels the step meets for the
   * first time side by side — a miss throws out of the step, the wave is awaited, the
   * step is run again from the start — which only a pure step can afford.
   */
  private async record(args: A, datas: (Float32Array | null)[], collect: boolean, ahead: boolean): Promise<Recording<Awaited<R>>> {
    const d = device();
    const make = (): CompiledArg[] => args.map((a, i) => (a instanceof Tensor
      ? Tensor.from(datas[i] as Float32Array, a.shape, { dtype: a.dtype })
      : a));
    // A pure step's kernels, compiled side by side before the recording (`Device.dryRunAhead`).
    if (ahead) await d.dryRunAhead(async () => { await this.fn(...(make() as A)); });
    d.beginCapture();
    if (collect) d.tuneMode = "collect";
    let inputs: CompiledArg[];
    let out: Awaited<R>;
    try {
      inputs = make();
      // A step that awaits (a streamed backbone's refills) is recorded across its awaits.
      out = await this.fn(...(inputs as A));
    } catch (err) {
      d.tuneMode = null;
      d.endCapture().dispose();
      throw err;
    }
    d.tuneMode = null;
    return { cap: d.endCapture(), inputs, out };
  }

  /** The passes over a fresh recording — fusion, hoisting, the plan — and its place in
   *  the table. */
  private finish(key: string, made: Recording<Awaited<R>>): void {
    this.passes(made);
    this.records.set(key, made);
  }

  private passes(made: Recording<Awaited<R>>): void {
    const { cap, out } = made;
    const held = tensorsOf(out).map((t) => t.buffer);
    if (this.fuse) cap.fuse(held);
    // The replay-invariant dispatches — a frozen weight's repack — run once and leave the
    // replay (`Capture.hoist`); then the plan, before the check, so the check holds the
    // recording that will replay.
    this.hoisted.push(cap.hoist().hoisted);
    if (this.plan) {
      const p = cap.plan(held);
      this.planned.push({ moved: p.moved, released: p.released, bytesBefore: p.bytesBefore, bytesAfter: p.bytesAfter, arenas: p.arenas });
    }
  }

  /**
   * **The tuning pass** (`docs/COMPILER.md` Step 5), owed from the first call. A pure
   * step: the step is recorded once more, collecting, into a throwaway recording whose
   * buffers the candidates may write, the candidates are timed there, and if a decision
   * changed the step is recorded a third time with the chosen kernels and swapped in.
   * A state-writing step: its candidates were collected on the live recording, and they
   * run there — at the start of the next call, when the last outputs are spent — with
   * nothing to re-record (its decisions serve the next recording). Either way the
   * fastest is cached for this adapter, in memory and in `localStorage`.
   */
  private async tunePending(key: string): Promise<void> {
    const p = this.pending.get(key);
    if (!p) return;
    this.pending.delete(key);
    const rec = this.records.get(key);
    if (!rec) return;
    const d = device();
    const t0 = performance.now();
    let report: TuneReport[];
    if (p.queue) {
      const outs = tensorsOf(rec.out);
      d.flush();
      const outBefore = await Promise.all(outs.map((o) => o.toArray()));
      if (this.disposed) return;
      p.cost.wait = performance.now() - t0;
      const queue = p.queue;
      report = await scope(async () => d.runTuning(queue));
      if (this.disposed) return;
      d.flush();
      outs.forEach((o, i) => d.writeWords(o.buffer, words(outBefore[i] as Float32Array)));
      // The passes the raw recording was waiting for.
      this.finish(key, rec);
      if (this.check) this.checked.push(await this.verify(rec.cap, rec.inputs, rec.out));
      if (this.disposed) return;
    } else {
      // In idle time the step may be disposed under this pass; every wait checks.
      const scratch = await this.record(p.args, p.datas, true, false);
      const queue = d.takeTuneQueue();
      if (this.disposed) { scratch.cap.dispose(); return; }
      report = await scope(async () => d.runTuning(queue));
      scratch.cap.dispose();
      if (this.disposed) return;
    }
    this.tuned.push(report);
    p.cost.tuning = performance.now() - t0 - p.cost.wait;
    p.cost.compile = d.tuneWarmMs;
    p.cost.candidates = report.reduce((a, t) => a + t.candidates.length, 0);
    // Where a decision changed and the recording is a pure function of its inputs —
    // said so, or found so (`mutatesState`) — it is made again with the chosen kernels
    // and swapped in; a step that writes state is made once.
    if (report.some((t) => t.chosen !== t.prior) && (this.pure || !rec.cap.mutatesState())) {
      const t1 = performance.now();
      const fresh = await this.record(p.args, p.datas, false, false);
      if (this.disposed) { fresh.cap.dispose(); return; }
      this.passes(fresh);
      if (this.check) this.checked.push(await this.verify(fresh.cap, fresh.inputs, fresh.out));
      this.replacement.get(key)?.cap.dispose();
      this.replacement.set(key, fresh);
      p.cost.rerecord = performance.now() - t1;
    }
  }

  /** Returns every recording's memory. The returned tensors are not to be used after. */
  dispose(): void {
    this.disposed = true;
    this.pending.clear();
    for (const r of this.replacement.values()) r.cap.dispose();
    this.replacement.clear();
    for (const r of this.records.values()) r.cap.dispose();
    this.records.clear();
  }

  private async verify(cap: Capture, inputs: readonly CompiledArg[], out: Awaited<R>): Promise<CheckReport> {
    const d = device();
    const live = cap.liveIns();
    // The commands encoded so far go out first — a read copies through its own encoder.
    const snapshot = async (): Promise<Float32Array[]> => { d.flush(); return Promise.all(live.map((b) => d.read(b, b.size / 4))); };
    const outs = tensorsOf(out);
    const before = await snapshot();
    const outBefore = await Promise.all(outs.map((o) => o.toArray()));
    if (cap.hasRefills) await cap.replayAsync(); else cap.replay();
    const replayedState = await snapshot();
    const replayed = await Promise.all(outs.map((o) => o.toArray()));
    d.flush();
    live.forEach((b, i) => d.writeWords(b, words(before[i] as Float32Array)));
    // **The rerun takes the path the recording did.** A captured step takes the
    // optimizer's per-parameter path; the single-group arena the uncaptured rerun would
    // take keeps its momentum elsewhere and would read as a spurious difference.
    d.suppressArena = true;
    let eagerState: Float32Array[] = [];
    let eager: Float32Array[] = [];
    try {
      await scope(async () => {
        const eo = tensorsOf(await this.fn(...(inputs as A)));
        eagerState = await snapshot();
        eager = await Promise.all(eo.map((o) => o.toArray()));
      });
    } finally {
      d.suppressArena = false;
    }
    // The call stays one step: the state goes back to where the recording left it.
    d.flush();
    live.forEach((b, i) => d.writeWords(b, words(before[i] as Float32Array)));
    outs.forEach((o, i) => d.writeWords(o.buffer, words(outBefore[i] as Float32Array)));
    let worst = 0, differ = 0;
    const offenders: string[] = [];
    const all = [...replayedState, ...replayed];
    const ref = [...eagerState, ...eager];
    all.forEach((a, i) => {
      const b = ref[i] as Float32Array;
      const limit = !this.fuse ? 0 : (i < live.length ? this.stateTol : this.tol);
      if (a.length !== b.length) { differ++; worst = Infinity; return; }
      let num = 0, den = 0;
      for (let k = 0; k < a.length; k++) { num = Math.max(num, Math.abs((a[k] as number) - (b[k] as number))); den = Math.max(den, Math.abs(b[k] as number)); }
      const gap = a.length ? num / (den + 1e-12) : 0;
      if (gap > limit) {
        differ++; worst = Math.max(worst, gap);
        offenders.push(`#${i} (${a.length} floats, ${i < live.length ? "state" : "an output"}) rel ${gap.toExponential(1)}`);
      }
    });
    const report = { buffers: live.length, outputs: outs.length, differ, worstRel: worst };
    if (differ) {
      throw new Error(
        `compiled(check: true): the replay is not the eager step — ${differ} of ${live.length + outs.length} buffers differ, ` +
        `the worst by ${worst.toExponential(2)} relative (${this.fuse ? `fused, tolerance ${this.tol} on the outputs and ${this.stateTol} on the state` : "plain, bit for bit"}). ` +
        "Something the step does is not in the recording: a value read back and used, a branch on the step's values, " +
        `a hyperparameter set outside the group. Worst: ${offenders.slice(0, 3).join("; ")}`);
    }
    return report;
  }
}

/**
 * `torch.compiled(fn, opts)` — see `Compiled`. Given a **module** instead of a function,
 * it is the inference form (`docs/INFER.md` Step 6): the module is put in eval mode,
 * its batch norms folded into the convolutions before them and the relus into their
 * epilogues (`fuseForInference` — the module's own `fuse()` where it has one, the
 * `Sequential` pass otherwise), and its `noGrad` forward recorded: the first call for a
 * shape runs it, every call after replays — the repacks hoisted, the intermediates in
 * arenas. The returned step takes the input tensor and hands back the logits.
 */
/** A float array's bits as the words `writeWords` takes. */
function words(a: Float32Array): Uint32Array<ArrayBuffer> {
  const copy = new Uint32Array(new ArrayBuffer(a.byteLength));
  new Float32Array(copy.buffer).set(a);
  return copy;
}

export function compiled<A extends CompiledArg[], R>(fn: (...args: A) => R, opts?: CompiledOptions): Compiled<A, R>;
export function compiled(model: Module, opts?: CompiledOptions): Compiled<[Tensor], Tensor>;
export function compiled(target: Module | ((...args: never[]) => unknown), opts: CompiledOptions = {}): unknown {
  if (target instanceof Module) {
    // Eval first: the fold reads the running statistics and refuses a training module.
    const model = target.eval();
    fuseForInference(model);
    const step = new Compiled((x: Tensor) => noGrad(() => model.forward(x)), { ...opts, pure: true },
      opts.int8 ? async () => { step.int8Layers = await quantizeForInt8(model); } : undefined);
    return step;
  }
  return new Compiled(target as (...args: CompiledArg[]) => unknown, opts);
}
