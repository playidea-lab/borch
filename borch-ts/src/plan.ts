/**
 * Memory planning over a captured step — **intermediates whose lives do not overlap share
 * bytes.**
 *
 * A capture pins every buffer the step allocates for as long as the capture lives, each
 * in its own buffer: the recorded bind groups point at them, and the eager step's habit
 * of handing a scope's buffers back to the pool cannot apply. Measured (`bench.ts`,
 * ResNet-18 CIFAR, 2026-09-19): 233 MB live at batch 16 with 517 MB of pool beside it,
 * 700 MB and 1,755 MB at batch 64. The recording knows the order every buffer is written
 * and read in (`Recorded.access`, `docs/COMPILER.md` Step 0), so each intermediate is an
 * interval — from the dispatch that first writes it to the last that touches it — and
 * intervals that do not overlap can be laid at the same offset of one arena. That is the
 * whole pass: a list of intervals, a first-fit over offsets, and the bind groups rebuilt
 * against `{buffer, offset, size}` slots (`BindSlot` already carries them for the
 * optimizer arena).
 *
 * **What is never moved** — because something outside the recording may still read it
 * where it was: a buffer uploaded under the capture (an input the caller writes into
 * before each replay, a constant), a buffer the caller says it holds (`held` — the same
 * set the fusion pass is given), a buffer bound as a sub-range anywhere (it belongs to an
 * arena of its own), and a buffer whose first touch is a read — its contents come from
 * before the recording (a parameter, a running statistic, or a fresh buffer a kernel
 * accumulates into, whose zeros are WebGPU's zero-initialisation and would not be zeros
 * in a reused slot). And a buffer no record touches at all — an intermediate the fusion
 * pass left unwritten — is simply released.
 *
 * **What the plan assumes of a kernel**: a binding it declares writable and never reads
 * is written *whole* by the dispatch — every element the buffer is later read for. A
 * kernel that wrote part of a fresh buffer and relied on WebGPU's zeros for the rest
 * would read a stale slot after planning. The capture probes hold the planned recordings
 * bit for bit against eager on two models; a kernel of that shape would show there.
 *
 * The plan is pure: it takes the records and three answers about buffers and returns
 * placements. `Capture.plan` in `device.ts` allocates the arenas, rebinds and releases.
 */
import { type BindSlot, bufOf, type Recorded } from "./device.js";

/** Where one buffer goes: which arena, at what byte offset, for which dispatches. */
export interface Placement {
  readonly buf: GPUBuffer;
  readonly arena: number;
  readonly offset: number;
  readonly size: number;
  readonly first: number;
  readonly last: number;
}

export interface Plan {
  /** Buffers laid into arenas, by the buffer they replace. */
  readonly placements: ReadonlyMap<GPUBuffer, Placement>;
  /** Bytes of each arena to allocate. */
  readonly arenas: readonly number[];
  /** Candidate buffers no record touches — releasable, nothing to rebind. */
  readonly untouched: readonly GPUBuffer[];
  /** Why candidates were left where they are. */
  readonly kept: { readonly liveIn: number; readonly subRange: number };
}

export interface PlanInput {
  readonly records: readonly Recorded[];
  /** Whether a buffer may be moved at all: the capture made it, nobody uploaded into it,
   *  nobody outside holds it. */
  readonly movable: (buf: GPUBuffer) => boolean;
  /** Every buffer that answers `movable` — so the untouched ones can be found. */
  readonly candidates: Iterable<GPUBuffer>;
  readonly sizeOf: (buf: GPUBuffer) => number;
  /** `minStorageBufferOffsetAlignment` — every offset is a multiple of it. */
  readonly align: number;
  /** `maxStorageBufferBindingSize` — no arena grows past it; a second one is opened. */
  readonly arenaMax: number;
}

/** What one record reads and writes, as buffers (a sub-range slot names its buffer). */
export function touchesOf(r: Recorded): { reads: GPUBuffer[]; writes: GPUBuffer[]; subRange: GPUBuffer[] } {
  const reads: GPUBuffer[] = [];
  const writes: GPUBuffer[] = [];
  const subRange: GPUBuffer[] = [];
  for (const b of r.buffers) if (!(b instanceof GPUBuffer)) subRange.push(b.buffer);
  const slot = (k: number): GPUBuffer => bufOf(r.buffers[k] as BindSlot);
  if (r.copy) {
    reads.push(slot(0)); writes.push(slot(1));
  } else if (r.refill) {
    writes.push(slot(0));
  } else if (r.meta && "expr" in r.meta) {
    for (const inp of r.meta.inputs) reads.push(slot(inp.binding));
    writes.push(slot(r.meta.out));
  } else if (r.meta && "input" in r.meta) {
    const input = r.meta.input;
    r.buffers.forEach((_, k) => { if (k === input) reads.push(slot(k)); else writes.push(slot(k)); });
  } else if (r.access) {
    r.buffers.forEach((_, k) => {
      const a = r.access?.[k] ?? "rw";
      if (a !== "w") reads.push(slot(k));
      if (a !== "r") writes.push(slot(k));
    });
  } else {
    r.buffers.forEach((_, k) => { reads.push(slot(k)); writes.push(slot(k)); });
  }
  return { reads, writes, subRange };
}

const roundUp = (n: number, align: number): number => Math.ceil(n / align) * align;

/**
 * Which buffers may not share an arena. **WebGPU validates a buffer's usages per
 * dispatch, not per range**: one buffer bound read-only at one offset and read-write at
 * another in the same dispatch is a fault ("includes writable usage and another usage in
 * the same synchronization scope" — measured, the first plan's every dispatch), and a
 * copy's source and destination may not be one buffer at all. So within each record the
 * read-only-bound buffers and the read-write-bound ones conflict pairwise, and a copy's
 * two ends conflict; conflicting buffers are given different arenas (a greedy colouring
 * in interval order — a chain of kernels alternates between two).
 */
function conflictsOf(records: readonly Recorded[], movable: (b: GPUBuffer) => boolean): Map<GPUBuffer, Set<GPUBuffer>> {
  const edges = new Map<GPUBuffer, Set<GPUBuffer>>();
  const join = (a: GPUBuffer, b: GPUBuffer): void => {
    if (a === b) return;
    let s = edges.get(a); if (!s) { s = new Set(); edges.set(a, s); } s.add(b);
    let t = edges.get(b); if (!t) { t = new Set(); edges.set(b, t); } t.add(a);
  };
  for (const r of records) {
    const ro: GPUBuffer[] = [];
    const rw: GPUBuffer[] = [];
    r.buffers.forEach((slot, k) => {
      const b = bufOf(slot);
      if (!movable(b)) return;
      // A copy: its source is read, its destination written. A dispatch: what the WGSL
      // declares — `r` is `var<storage, read>` or a uniform, anything else `read_write`.
      // Without a declaration the binding is taken as writable, which conflicts most.
      const kind = r.copy ? (k === 0 ? "r" : "w") : (r.access?.[k] ?? "w");
      (kind === "r" ? ro : rw).push(b);
    });
    for (const a of ro) for (const b of rw) join(a, b);
    if (r.copy && ro[0] && rw[0]) join(ro[0], rw[0]);
  }
  return edges;
}

/**
 * Lays the movable intermediates of `records` into arenas. First-fit by offset over the
 * intervals already placed that overlap in time; candidates are taken in order of first
 * write, the larger first among equals, which keeps the big activations low in the arena
 * and the small ones in the gaps above them.
 */
export function planRecords(input: PlanInput): Plan {
  const first = new Map<GPUBuffer, number>();
  const last = new Map<GPUBuffer, number>();
  const readFirst = new Set<GPUBuffer>();
  const subRange = new Set<GPUBuffer>();
  input.records.forEach((r, i) => {
    const t = touchesOf(r);
    for (const b of t.subRange) subRange.add(b);
    // Reads before writes within one dispatch: a kernel that reads and writes the same
    // buffer reads what was there first.
    for (const b of t.reads) {
      if (!first.has(b)) { first.set(b, i); readFirst.add(b); }
      last.set(b, i);
    }
    for (const b of t.writes) {
      if (!first.has(b)) first.set(b, i);
      last.set(b, i);
    }
  });

  const kept = { liveIn: 0, subRange: 0 };
  const untouched: GPUBuffer[] = [];
  const intervals: { buf: GPUBuffer; size: number; first: number; last: number }[] = [];
  for (const buf of input.candidates) {
    if (!input.movable(buf)) continue;
    const f = first.get(buf);
    if (f === undefined) { untouched.push(buf); continue; }
    if (subRange.has(buf)) { kept.subRange++; continue; }
    if (readFirst.has(buf)) { kept.liveIn++; continue; }
    intervals.push({ buf, size: input.sizeOf(buf), first: f, last: last.get(buf) ?? f });
  }
  intervals.sort((a, b) => a.first - b.first || b.size - a.size);

  const conflicts = conflictsOf(input.records, input.movable);
  const placements = new Map<GPUBuffer, Placement>();
  const arenas: number[] = [];
  const placed: Placement[][] = [];
  const members: Set<GPUBuffer>[] = [];
  const clashes = (buf: GPUBuffer, a: number): boolean => {
    const c = conflicts.get(buf);
    if (!c) return false;
    for (const m of members[a] as Set<GPUBuffer>) if (c.has(m)) return true;
    return false;
  };
  for (const it of intervals) {
    const size = roundUp(it.size, input.align);
    let done = false;
    for (let a = 0; a < arenas.length && !done; a++) {
      if (clashes(it.buf, a)) continue;
      const busy = (placed[a] as Placement[])
        .filter((p) => p.first <= it.last && it.first <= p.last)
        .sort((x, y) => x.offset - y.offset);
      let offset = 0;
      for (const p of busy) {
        if (offset + size <= p.offset) break;
        offset = Math.max(offset, p.offset + p.size);
      }
      if (offset + size <= input.arenaMax) {
        const pl: Placement = { buf: it.buf, arena: a, offset, size, first: it.first, last: it.last };
        placements.set(it.buf, pl);
        (placed[a] as Placement[]).push(pl);
        (members[a] as Set<GPUBuffer>).add(it.buf);
        arenas[a] = Math.max(arenas[a] as number, offset + size);
        done = true;
      }
    }
    if (!done) {
      const a = arenas.length;
      arenas.push(size);
      placed.push([{ buf: it.buf, arena: a, offset: 0, size, first: it.first, last: it.last }]);
      members.push(new Set([it.buf]));
      placements.set(it.buf, (placed[a] as Placement[])[0] as Placement);
    }
  }
  return { placements, arenas, untouched, kept };
}
