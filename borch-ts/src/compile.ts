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
import { Capture } from "./device.js";
import { device, scope, Tensor } from "./tensor.js";

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

  constructor(private readonly fn: (...args: A) => R, opts: CompiledOptions = {}) {
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
    const rec = this.records.get(key);
    if (rec) {
      rec.inputs.forEach((held, i) => {
        if (held instanceof Tensor) held.copyFrom(args[i] as Tensor);
      });
      if (rec.cap.hasRefills) await rec.cap.replayAsync(); else rec.cap.replay();
      return rec.out;
    }
    const datas = await Promise.all(args.map((a) => (a instanceof Tensor ? a.toArray() : Promise.resolve(null))));
    const d = device();
    d.beginCapture();
    let inputs: CompiledArg[];
    let out: Awaited<R>;
    try {
      inputs = args.map((a, i) => (a instanceof Tensor
        ? Tensor.from(datas[i] as Float32Array, a.shape, { dtype: a.dtype })
        : a));
      // A step that awaits (a streamed backbone's refills) is recorded across its awaits.
      out = await this.fn(...(inputs as A));
    } catch (err) {
      d.endCapture().dispose();
      throw err;
    }
    const cap = d.endCapture();
    const held = tensorsOf(out).map((t) => t.buffer);
    if (this.fuse) cap.fuse(held);
    // Planned before the check, so the check holds the recording that will replay.
    if (this.plan) {
      const p = cap.plan(held);
      this.planned.push({ moved: p.moved, released: p.released, bytesBefore: p.bytesBefore, bytesAfter: p.bytesAfter, arenas: p.arenas });
    }
    if (this.check) this.checked.push(await this.verify(cap, inputs, out));
    this.records.set(key, { cap, inputs, out });
    return out;
  }

  /** Returns every recording's memory. The returned tensors are not to be used after. */
  dispose(): void {
    for (const r of this.records.values()) r.cap.dispose();
    this.records.clear();
  }

  private async verify(cap: Capture, inputs: readonly CompiledArg[], out: Awaited<R>): Promise<CheckReport> {
    const d = device();
    const live = cap.liveIns();
    const words = (a: Float32Array): Uint32Array<ArrayBuffer> => {
      const copy = new Uint32Array(new ArrayBuffer(a.byteLength));
      new Float32Array(copy.buffer).set(a);
      return copy;
    };
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

/** `torch.compiled(fn, opts)` — see `Compiled`. */
export function compiled<A extends CompiledArg[], R>(fn: (...args: A) => R, opts: CompiledOptions = {}): Compiled<A, R> {
  return new Compiled(fn, opts);
}
