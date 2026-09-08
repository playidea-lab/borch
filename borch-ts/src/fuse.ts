/**
 * Fusion over a captured step — **elementwise kernels that feed each other become one.**
 *
 * A capture (`device.ts`) is the step as the device saw it: a list of dispatches with
 * their buffers, and, for an elementwise one, what it computed (`Elementwise` in
 * `kernels.ts`). This pass reads that list as a graph and, for each elementwise dispatch
 * — the root — pulls its elementwise producers into it: a producer whose output the root
 * reads contiguously over the same element count, whose output nobody reads *between*
 * the two (moving it to the root's position would make such a reader see an unwritten
 * buffer), and whose output is written exactly once. The producers' producers follow by
 * the same rule, so what fuses is a tree of expressions — a GELU written out by hand, a
 * loss built from squares and means, a normalisation someone typed — computed in one
 * kernel from its leaves.
 *
 * **A fused node writes its output unless nothing can read it.** A buffer no later
 * dispatch reads may still be held — by Python (a prediction kept for an accuracy, a
 * value printed after the step) or by autograd (a forward value saved for a backward run
 * after the capture) — and a replay that silently stopped writing it would read as a
 * wrong number, not as an error. So what goes unwritten is what the pass can see is
 * unreadable: autograd's own intermediates (`internal`), and, when the caller says which
 * buffers it holds (`Capture.fuse(held)`), values made with autograd off that it does not
 * hold (`detached`) — an inference pass keeps nothing but its answer. Everything else is
 * left exactly as the eager step left it. The expressions are the same strings the single
 * kernels use, on the same IEEE operations in the same order, so the values are the same
 * bit for bit (`capture:py`).
 *
 * **A reduction takes its producers in too.** A sum, a fold along an axis, a broadcast
 * gradient folded back — each reads every element of its input exactly once, so an
 * elementwise tree feeding it is evaluated inside the reduction (`Reduce` in
 * `kernels.ts`: the kernel rebuilt around a `load(i)`) under the same rules, and the
 * tree's own output is written only when something else reads it — a loss's squared
 * difference, the `x²` under a variance, a gradient on its way to a bias never touch
 * memory.
 *
 * **Not an epilogue of the matmul.** Fusing the tree that consumes a product into the
 * product's kernel — the lanes walking the tile they stored, the tile still in cache —
 * was written and measured (2026-09-07): the hand-written network's fused replay went
 * from 76 to 62 dispatches and from 0.72 to 1.69 ms, the GPT step not at all. The
 * subgroup matmul is one subgroup a workgroup, so the elementwise work fell on thirty-two
 * lanes per tile, sixty-four cells each in series, where an elementwise kernel gives
 * every cell a thread; the launch it saved cost less than the parallelism it lost.
 * Removed rather than kept behind a switch. A product's consumers stay a kernel of
 * their own, fused among themselves as above.
 *
 * Measured on the M4 Max (2026-09-07): the U-Net step, already fused by hand where it
 * counts (BatchNorm with its ReLU, the loss with its sigmoid), has ten elementwise
 * dispatches of 223 and gives this pass three; a hand-written GELU network gives it
 * sixty-three of 139 — see `fuse:py`. `fuse()` itself is a millisecond; the shaders it
 * made are compiled when first dispatched, which the first replay pays — ten
 * milliseconds on the U-Net, nothing visible on the small network (measured).
 */
import { Device, type Recorded } from "./device.js";
import { contiguousStrides, type Elementwise, elementLanes, grid1d, laneable, laneMode, type Reduce, type Source, WORKGROUP } from "./kernels.js";

function elementwise(meta: Elementwise | Reduce | undefined): meta is Elementwise {
  return meta !== undefined && "expr" in meta;
}

function reduction(meta: Elementwise | Reduce | undefined): meta is Reduce {
  return meta !== undefined && "make" in meta;
}

/** A short, stable hash of a string — the fused kernels' cache key. */
function hashOf(text: string): string {
  let h1 = 0x811c9dc5, h2 = 0x01000193;
  for (let i = 0; i < text.length; i++) {
    const c = text.charCodeAt(i);
    h1 = Math.imul(h1 ^ c, 16777619) >>> 0;
    h2 = Math.imul(h2 + c, 2246822519) >>> 0;
  }
  return h1.toString(16) + h2.toString(16) + text.length.toString(16);
}

/**
 * The longest walk one reduction thread may take with a tree inside it. A fold along an
 * axis is one thread per output cell walking the axis; a tree evaluated in that walk
 * runs serially where the elementwise kernel ran it across every thread, and past this
 * many elements the reduction is slower with the tree than the two were apart. Measured
 * on the M4 Max (2026-09-07, `fuse:py`): a LayerNorm's mean over 256 taken in — the
 * training replay 0.95 ms, the inference pass 0.70; left out, 0.72 and 0.43. The full
 * sum (one element a thread) and the wide broadcast fold (a piece over 256 threads)
 * stay under the limit; the axis folds of a small network do not.
 */
const SERIAL_LIMIT = 64;

/** The WGSL of every fused kernel built, by key — for reading what the pass made. */
export const fusedCodes = new Map<string, string>();

interface Node {
  readonly index: number;
  readonly rec: Recorded;
  readonly meta: Elementwise;
}

/** The recording as a graph: who reads and who writes each buffer, by record index, and
 *  which buffers the caller holds. */
interface Graph {
  readonly readers: Map<GPUBuffer, number[]>;
  readonly writers: Map<GPUBuffer, number[]>;
  readonly held?: ReadonlySet<GPUBuffer>;
}

/** Rewrites `records` with elementwise trees fused. Returns the new list, how many
 *  dispatches were folded away, and how many intermediates go unwritten. `held` — the
 *  buffers the caller still holds; see `Capture.fuse`. */
export function fuseRecords(dev: Device, records: readonly Recorded[], held?: ReadonlySet<GPUBuffer>): { records: Recorded[]; fused: number; unwritten: number } {
  // A dispatch without a recipe is taken to read and write every buffer it binds.
  const readers = new Map<GPUBuffer, number[]>();
  const writers = new Map<GPUBuffer, number[]>();
  const push = (map: Map<GPUBuffer, number[]>, b: GPUBuffer, i: number): void => {
    const list = map.get(b);
    if (list) list.push(i); else map.set(b, [i]);
  };
  records.forEach((r, i) => {
    if (elementwise(r.meta)) {
      for (const inp of r.meta.inputs) push(readers, r.buffers[inp.binding] as GPUBuffer, i);
      push(writers, r.buffers[r.meta.out] as GPUBuffer, i);
    } else if (reduction(r.meta)) {
      const input = r.meta.input;
      push(readers, r.buffers[input] as GPUBuffer, i);
      r.buffers.forEach((b, k) => { if (k !== input) push(writers, b, i); });
    } else {
      for (const b of r.buffers) { push(readers, b, i); push(writers, b, i); }
    }
  });
  const graph: Graph = held ? { readers, writers, held } : { readers, writers };
  const producerOf = (b: GPUBuffer, before: number): number | undefined => {
    const ws = writers.get(b) ?? [];
    let last: number | undefined;
    for (const w of ws) if (w < before) last = w;
    return last;
  };

  const absorbed = new Set<number>();
  const out: Recorded[] = [];
  const replaced = new Map<number, Recorded>();   // root index → the fused dispatch
  let fused = 0;
  let unwritten = 0;
  for (let i = records.length - 1; i >= 0; i--) {
    const root = records[i] as Recorded;
    if (absorbed.has(i)) continue;
    if (elementwise(root.meta)) {
      const tree = gather(records, i, i, 0, graph, producerOf, absorbed);
      if (tree.length < 2) continue;
      for (const node of tree) if (node.index !== i) absorbed.add(node.index);
      fused += tree.length - 1;
      unwritten += tree.filter((node) => !mustWrite(node, tree, graph)).length;
      replaced.set(i, build(dev, tree, graph));
    } else if (reduction(root.meta)) {
      const top = feeder(records, i, root.meta, graph, producerOf, absorbed);
      if (top === undefined) continue;
      const tree = gather(records, top, i, root.buffers.length - 1, graph, producerOf, absorbed);
      for (const node of tree) absorbed.add(node.index);
      fused += tree.length;
      unwritten += tree.filter((node) => !mustWrite(node, tree, graph, i)).length;
      replaced.set(i, buildReduce(dev, tree, root, i, root.meta, graph));
    }
  }
  records.forEach((r, i) => {
    if (absorbed.has(i)) return;
    out.push(replaced.get(i) ?? r);
  });
  return { records: out, fused, unwritten };
}

/**
 * Whether the producer `p` of `buf` can be evaluated at `position` instead: an
 * elementwise dispatch of `n` elements writing `buf` and nothing else writing it, with no
 * reader of `buf` between the two except `except`.
 */
function movable(
  records: readonly Recorded[], p: number, buf: GPUBuffer, n: number, position: number,
  graph: Graph, except: ReadonlySet<number>,
): Node | undefined {
  const prod = records[p] as Recorded;
  if (!elementwise(prod.meta) || prod.meta.n !== n) return undefined;
  if (prod.buffers[prod.meta.out] !== buf) return undefined;
  if ((graph.writers.get(buf) ?? []).length !== 1) return undefined;
  const between = (graph.readers.get(buf) ?? []).filter((r) => r > p && r < position && !except.has(r));
  if (between.length) return undefined;
  return { index: p, rec: prod, meta: prod.meta };
}

/** The elementwise producer a reduction at `at` can take in, if it has one. */
function feeder(
  records: readonly Recorded[], at: number, meta: Reduce, graph: Graph,
  producerOf: (b: GPUBuffer, before: number) => number | undefined,
  absorbed: Set<number>,
): number | undefined {
  const rec = records[at] as Recorded;
  const buf = rec.buffers[meta.input] as GPUBuffer;
  if (meta.serial > SERIAL_LIMIT) return undefined;
  const p = producerOf(buf, at);
  if (p === undefined || absorbed.has(p)) return undefined;
  const node = movable(records, p, buf, meta.n, at, graph, new Set([at]));
  if (!node) return undefined;
  // The reduction's own buffers count against the binding budget alongside the tree's.
  if (bindingsOf([node], graph, at) + rec.buffers.length - 1 > Device.storageBuffersPerStage) return undefined;
  return p;
}

/**
 * The root and every producer it can pull in, in original order. The tree is evaluated
 * at `position` — the root's own index, or that of the reduction consuming the root —
 * with `reserve` bindings of the consumer's own to leave room for.
 */
function gather(
  records: readonly Recorded[], rootIndex: number, position: number, reserve: number,
  graph: Graph,
  producerOf: (b: GPUBuffer, before: number) => number | undefined,
  absorbed: Set<number>,
): Node[] {
  const root = records[rootIndex] as Recorded;
  const consumer = position === rootIndex ? undefined : position;
  const nodes = new Map<number, Node>();
  nodes.set(rootIndex, { index: rootIndex, rec: root, meta: root.meta as Elementwise });
  const queue = [rootIndex];
  while (queue.length) {
    const at = queue.pop() as number;
    const node = nodes.get(at) as Node;
    for (const inp of node.meta.inputs) {
      if (inp.strides && !contiguousStrides(node.meta.shape, inp.strides)) continue;
      const buf = node.rec.buffers[inp.binding] as GPUBuffer;
      const p = producerOf(buf, at);
      if (p === undefined || nodes.has(p) || absorbed.has(p)) continue;
      // Written once, and read by nobody between the producer and the position except
      // nodes already in the tree (and the consumer).
      const except = new Set(nodes.keys());
      if (consumer !== undefined) except.add(consumer);
      const prod = movable(records, p, buf, node.meta.n, position, graph, except);
      if (!prod) continue;
      // Within the device's binding budget: every leaf and every node is a buffer.
      const trial = new Map(nodes); trial.set(p, prod);
      if (bindingsOf([...trial.values()], graph, consumer) + reserve > Device.storageBuffersPerStage) continue;
      nodes.set(p, prod);
      queue.push(p);
    }
  }
  return [...nodes.values()].sort((a, b) => a.index - b.index);
}

/**
 * Whether a node's output has to be written: when something outside the tree reads it,
 * or when it could be read by what the pass cannot see — a forward value autograd
 * saved, or a tensor the caller holds (with `held` given, an unheld value made with
 * autograd off is safe; without, only autograd's own intermediates are). The root of a
 * tree that is itself the kernel is always written; the root feeding a reduction
 * (`consumer`) is an intermediate like the rest.
 */
function mustWrite(node: Node, tree: readonly Node[], graph: Graph, consumer?: number): boolean {
  const root = tree[tree.length - 1] as Node;
  if (node.index === root.index && consumer === undefined) return true;
  const out = node.rec.buffers[node.meta.out] as GPUBuffer;
  const unseen = node.meta.internal || (graph.held !== undefined && node.meta.detached === true && !graph.held.has(out));
  if (!unseen) return true;
  const inTree = new Set(tree.map((t) => t.index));
  if (consumer !== undefined) inTree.add(consumer);
  return (graph.readers.get(out) ?? []).some((r) => !inTree.has(r));
}

/** How many buffers a kernel for `tree` binds: its distinct leaves plus the outputs it writes. */
function bindingsOf(tree: readonly Node[], graph: Graph, consumer?: number): number {
  const outputs = new Set<GPUBuffer>();
  for (const node of tree) outputs.add(node.rec.buffers[node.meta.out] as GPUBuffer);
  const leaves = new Set<GPUBuffer>();
  for (const node of tree) {
    for (const inp of node.meta.inputs) {
      const b = node.rec.buffers[inp.binding] as GPUBuffer;
      if (!outputs.has(b)) leaves.add(b);
    }
  }
  return leaves.size + tree.filter((node) => mustWrite(node, tree, graph, consumer)).length;
}

/** The tree as WGSL: its bindings and buffers, the preludes, the body computing every
 *  node at element `gid`, and the variable holding the root's value. */
interface Emitted {
  readonly bindings: string[];
  readonly buffers: GPUBuffer[];
  readonly prelude: string;
  readonly body: string;
  readonly result: string;
  /** Leaves read as one broadcast value (`L[0]`) — bound as scalars even when the
   *  kernel takes four cells a thread: their buffers may be four bytes. */
  readonly scalarLeaves: Set<number>;
}

function emit(tree: Node[], graph: Graph, consumer?: number): Emitted {
  const inTree = new Map<GPUBuffer, Node>();
  for (const node of tree) inTree.set(node.rec.buffers[node.meta.out] as GPUBuffer, node);
  // Bindings: leaves first (dedup by buffer), then every node's output.
  const leaves: { buffer: GPUBuffer; index: number }[] = [];
  const leafIndex = new Map<GPUBuffer, number>();
  const leafCode: string[] = [];
  const nodeCode: string[] = [];
  const preludes = new Set<string>();
  const scalarLeaves = new Set<number>();
  const value = new Map<number, string>();    // node index → the WGSL variable holding its value
  tree.forEach((node, k) => {
    if (node.meta.prelude) preludes.add(node.meta.prelude);
    const locals: string[] = [];
    for (const inp of node.meta.inputs) {
      const buf = node.rec.buffers[inp.binding] as GPUBuffer;
      const producer = inTree.get(buf);
      if (producer && producer.index !== node.index && producer.index < node.index) {
        locals.push(`let ${inp.local} = ${value.get(producer.index) as string};`);
        continue;
      }
      let li = leafIndex.get(buf);
      if (li === undefined) {
        li = leaves.length;
        leaves.push({ buffer: buf, index: li });
        leafIndex.set(buf, li);
      }
      // The leaf's element for this node: contiguous is `gid`; broadcast follows the strides
      // over this node's shape.
      if (inp.strides && inp.strides.every((v) => v === 0)) {
        // One value broadcast to the whole output.
        scalarLeaves.add(li);
        locals.push(`let ${inp.local} = L${li}[0];`);
      } else if (inp.strides && !contiguousStrides(node.meta.shape, inp.strides)) {
        // Four cells a thread: a leaf whose last stride is zero shares one value across
        // the four (a scalar binding, the index of the first cell); one whose last
        // stride is one has them consecutive (a `vec4` at the index over four).
        if (laneMode(node.meta.shape, inp.strides) === "same") scalarLeaves.add(li);
        const lines = [`  var rest_${k}_${inp.binding} = gid;`, `  var ix_${k}_${inp.binding}: u32 = 0u;`];
        for (let d = node.meta.shape.length - 1; d >= 0; d--) {
          const size = node.meta.shape[d] ?? 1;
          lines.push(`  { let i = rest_${k}_${inp.binding} % ${size}u; rest_${k}_${inp.binding} = rest_${k}_${inp.binding} / ${size}u;`);
          if ((inp.strides[d] ?? 0) !== 0) lines.push(`    ix_${k}_${inp.binding} = ix_${k}_${inp.binding} + i * ${inp.strides[d]}u;`);
          lines.push("  }");
        }
        leafCode.push(lines.join("\n"));
        locals.push(`let ${inp.local} = L${li}[ix_${k}_${inp.binding}];`);
      } else {
        locals.push(`let ${inp.local} = L${li}[gid];`);
      }
    }
    value.set(node.index, `v${k}`);
    nodeCode.push(`  var v${k}: f32;\n  { ${locals.join(" ")} v${k} = ${node.meta.expr}; }` + (mustWrite(node, tree, graph, consumer) ? `\n  O${k}[gid] = v${k};` : ""));
  });
  const bindings: string[] = [];
  const buffers: GPUBuffer[] = [];
  leaves.forEach((leaf, i) => { bindings.push(`@group(0) @binding(${i}) var<storage, read> L${i}: array<f32>;`); buffers.push(leaf.buffer); });
  tree.forEach((node, k) => {
    if (!mustWrite(node, tree, graph, consumer)) return;
    bindings.push(`@group(0) @binding(${buffers.length}) var<storage, read_write> O${k}: array<f32>;`);
    buffers.push(node.rec.buffers[node.meta.out] as GPUBuffer);
  });
  return { bindings, buffers, prelude: [...preludes].join("\n"), body: [...leafCode, ...nodeCode].join("\n"), result: `v${tree.length - 1}`, scalarLeaves };
}

/** Whether every operand of every node in the tree is read contiguously — then the
 *  kernel can take four cells a thread (`elementLanes`). */
function allContiguous(tree: readonly Node[]): boolean {
  return tree.every((node) => node.meta.inputs.every((inp) => !inp.strides || laneable(node.meta.shape, inp.strides)));
}

/**
 * One kernel for the tree: the leaves read, every node's output written. Four cells a
 * thread when every operand is contiguous and the count allows: the scalar body is
 * repeated per component of the `vec4` loads, so the operations and their order are
 * those of one cell a thread, and the outputs are stored as one `vec4` each.
 */
function build(dev: Device, tree: Node[], graph: Graph): Recorded {
  const root = tree[tree.length - 1] as Node;
  const n = root.meta.n;
  const e = emit(tree, graph);
  const lanes = elementLanes(n, allContiguous(tree));
  const grid = grid1d(n / lanes);
  let bindings = e.bindings.join("\n");
  let body = e.body;
  if (lanes === 4) {
    bindings = e.bindings.map((line, i) => i < e.buffers.length && e.scalarLeaves.has(i) && line.includes(`L${i}:`) ? line : line.replace("array<f32>", "array<vec4<f32>>")).join("\n");
    const outs = [...new Set([...e.body.matchAll(/O(\d+)\[gid\] = /g)].map((m) => m[1]))];
    const lane = (c: string): string => `  {\n${e.body
      .replace(/var (rest_\w+) = gid;/g, "var $1 = gid * 4u;")
      .replace(/L(\d+)\[gid\]/g, `L$1[gid].${c}`)
      .replace(/L(\d+)\[(ix_\w+)\]/g, (m, li: string, ix: string) => e.scalarLeaves.has(Number(li)) ? m : `L${li}[${ix} / 4u].${c}`)
      .replace(/O(\d+)\[gid\] = /g, `o$1.${c} = `)}\n  }`;
    body = [...outs.map((k) => `  var o${k}: vec4<f32>;`), ...["x", "y", "z", "w"].map(lane), ...outs.map((k) => `  O${k}[gid] = o${k};`)].join("\n");
  }
  const code = `${e.prelude}
${bindings}
@compute @workgroup_size(${WORKGROUP})
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let gid = g.y * ${grid.threadsX}u + g.x;
  if (gid >= ${n / lanes}u) { return; }
${body}
}`;
  // The key is the code: two trees of the same ops can bind differently (a leaf used
  // twice, an intermediate written or not), and a pipeline cached by op names alone was
  // handed a bind group of another shape (measured: "binding index 7 not present").
  const key = `fused:${hashOf(code)}:${n}`;
  const pipeline = dev.pipeline(key, () => code);
  fusedCodes.set(key, code);
  return { pipeline, bindGroup: dev.bindGroupFor(pipeline, e.buffers), groups: [grid.x, grid.y, 1], buffers: e.buffers, sig: key };
}

/** The reduction `root` rebuilt around the tree feeding it: the tree's bindings first,
 *  the reduction's own (its input left out) after. */
function buildReduce(dev: Device, tree: Node[], root: Recorded, at: number, meta: Reduce, graph: Graph): Recorded {
  const e = emit(tree, graph, at);
  const source: Source = {
    bindings: e.bindings.join("\n"), prelude: e.prelude, count: e.buffers.length,
    load: `fn load(gid: u32) -> f32 {\n${e.body}\n  return ${e.result};\n}`,
  };
  const code = meta.make(source);
  const buffers = [...e.buffers, ...root.buffers.filter((_, k) => k !== meta.input)];
  const key = `fused:${hashOf(code)}:${meta.n}`;
  const pipeline = dev.pipeline(key, () => code);
  fusedCodes.set(key, code);
  return { pipeline, bindGroup: dev.bindGroupFor(pipeline, buffers), groups: root.groups, buffers, sig: key.replace("fused:", "fusedr:") };
}
