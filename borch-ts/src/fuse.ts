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
 * **Every fused node still writes its output.** A dead-intermediate elimination would
 * be the larger saving, and it cannot be done from here: a buffer that no later dispatch
 * reads may still be held by Python — a prediction kept for an accuracy, a value
 * printed after the step — and a replay that silently stopped writing it would read as
 * a wrong number, not as an error. What is saved is the launch and the read of each
 * intermediate; what is kept is that a replay leaves every buffer exactly as the eager
 * step did. The expressions are the same strings the single kernels use, on the same
 * IEEE operations in the same order, so the values are the same bit for bit (`capture:py`).
 *
 * Measured on the M4 Max (2026-09-07): the U-Net step, already fused by hand where it
 * counts, gives this pass fifteen of 224 dispatches; a hand-written GELU network gives
 * it far more — see `fuse:py`.
 */
import { Device, type Recorded } from "./device.js";
import { contiguousStrides, type Elementwise, grid1d, WORKGROUP } from "./kernels.js";

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

/** The WGSL of every fused kernel built, by key — for reading what the pass made. */
export const fusedCodes = new Map<string, string>();

interface Node {
  readonly index: number;
  readonly rec: Recorded;
  readonly meta: Elementwise;
}

/** Rewrites `records` with elementwise trees fused. Returns the new list and how many
 *  dispatches were folded away. */
export function fuseRecords(dev: Device, records: readonly Recorded[]): { records: Recorded[]; fused: number } {
  // Who reads and who writes each buffer, by record index. A dispatch without a recipe
  // is taken to read and write every buffer it binds.
  const readers = new Map<GPUBuffer, number[]>();
  const writers = new Map<GPUBuffer, number[]>();
  const push = (map: Map<GPUBuffer, number[]>, b: GPUBuffer, i: number): void => {
    const list = map.get(b);
    if (list) list.push(i); else map.set(b, [i]);
  };
  records.forEach((r, i) => {
    if (r.meta) {
      for (const inp of r.meta.inputs) push(readers, r.buffers[inp.binding] as GPUBuffer, i);
      push(writers, r.buffers[r.meta.out] as GPUBuffer, i);
    } else {
      for (const b of r.buffers) { push(readers, b, i); push(writers, b, i); }
    }
  });
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
  for (let i = records.length - 1; i >= 0; i--) {
    const root = records[i] as Recorded;
    if (!root.meta || absorbed.has(i)) continue;
    const tree = gather(records, i, readers, writers, producerOf, absorbed);
    if (tree.length < 2) continue;
    for (const node of tree) if (node.index !== i) absorbed.add(node.index);
    fused += tree.length - 1;
    replaced.set(i, build(dev, tree, readers));
  }
  records.forEach((r, i) => {
    if (absorbed.has(i)) return;
    out.push(replaced.get(i) ?? r);
  });
  return { records: out, fused };
}

/** The root and every producer it can pull in, in original order. */
function gather(
  records: readonly Recorded[], rootIndex: number,
  readers: Map<GPUBuffer, number[]>, writers: Map<GPUBuffer, number[]>,
  producerOf: (b: GPUBuffer, before: number) => number | undefined,
  absorbed: Set<number>,
): Node[] {
  const root = records[rootIndex] as Recorded;
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
      const prod = records[p] as Recorded;
      if (!prod.meta || prod.meta.n !== node.meta.n) continue;
      if (prod.buffers[prod.meta.out] !== buf) continue;
      // Written once, and read by nobody between the producer and the root except nodes
      // already in the tree.
      if ((writers.get(buf) ?? []).length !== 1) continue;
      const between = (readers.get(buf) ?? []).filter((r) => r > p && r < rootIndex && !nodes.has(r));
      if (between.length) continue;
      // Within the device's binding budget: every leaf and every node is a buffer.
      const trial = new Map(nodes); trial.set(p, { index: p, rec: prod, meta: prod.meta });
      if (bindingsOf([...trial.values()], readers) > Device.storageBuffersPerStage) continue;
      nodes.set(p, { index: p, rec: prod, meta: prod.meta });
      queue.push(p);
    }
  }
  return [...nodes.values()].sort((a, b) => a.index - b.index);
}

/** Whether a node's output has to be written: the root's always; an intermediate's
 *  when it is not internal, or when something outside the tree reads it. */
function mustWrite(node: Node, tree: readonly Node[], readers: Map<GPUBuffer, number[]>): boolean {
  const root = tree[tree.length - 1] as Node;
  if (node.index === root.index || !node.meta.internal) return true;
  const inTree = new Set(tree.map((t) => t.index));
  return (readers.get(node.rec.buffers[node.meta.out] as GPUBuffer) ?? []).some((r) => !inTree.has(r));
}

/** How many buffers a kernel for `tree` binds: its distinct leaves plus the outputs it writes. */
function bindingsOf(tree: readonly Node[], readers: Map<GPUBuffer, number[]>): number {
  const outputs = new Set<GPUBuffer>();
  for (const node of tree) outputs.add(node.rec.buffers[node.meta.out] as GPUBuffer);
  const leaves = new Set<GPUBuffer>();
  for (const node of tree) {
    for (const inp of node.meta.inputs) {
      const b = node.rec.buffers[inp.binding] as GPUBuffer;
      if (!outputs.has(b)) leaves.add(b);
    }
  }
  return leaves.size + tree.filter((node) => mustWrite(node, tree, readers)).length;
}

/** One kernel for the tree: the leaves read, every node's output written. */
function build(dev: Device, tree: Node[], readers: Map<GPUBuffer, number[]>): Recorded {
  const root = tree[tree.length - 1] as Node;
  const n = root.meta.n;
  const inTree = new Map<GPUBuffer, Node>();
  for (const node of tree) inTree.set(node.rec.buffers[node.meta.out] as GPUBuffer, node);
  // Bindings: leaves first (dedup by buffer), then every node's output.
  const leaves: { buffer: GPUBuffer; index: number }[] = [];
  const leafIndex = new Map<GPUBuffer, number>();
  const leafCode: string[] = [];
  const nodeCode: string[] = [];
  const preludes = new Set<string>();
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
      if (inp.strides && !contiguousStrides(node.meta.shape, inp.strides)) {
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
    nodeCode.push(`  var v${k}: f32;\n  { ${locals.join(" ")} v${k} = ${node.meta.expr}; }` + (mustWrite(node, tree, readers) ? `\n  O${k}[gid] = v${k};` : ""));
  });
  const bindings: string[] = [];
  const buffers: GPUBuffer[] = [];
  leaves.forEach((leaf, i) => { bindings.push(`@group(0) @binding(${i}) var<storage, read> L${i}: array<f32>;`); buffers.push(leaf.buffer); });
  tree.forEach((node, k) => {
    if (!mustWrite(node, tree, readers)) return;
    bindings.push(`@group(0) @binding(${buffers.length}) var<storage, read_write> O${k}: array<f32>;`);
    buffers.push(node.rec.buffers[node.meta.out] as GPUBuffer);
  });
  const grid = grid1d(n);
  const code = `${[...preludes].join("\n")}
${bindings.join("\n")}
@compute @workgroup_size(${WORKGROUP})
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let gid = g.y * ${grid.threadsX}u + g.x;
  if (gid >= ${n}u) { return; }
${leafCode.join("\n")}
${nodeCode.join("\n")}
}`;
  // The key is the code: two trees of the same ops can bind differently (a leaf used
  // twice, an intermediate written or not), and a pipeline cached by op names alone was
  // handed a bind group of another shape (measured: "binding index 7 not present").
  const key = `fused:${hashOf(code)}:${n}`;
  const pipeline = dev.pipeline(key, () => code);
  fusedCodes.set(key, code);
  return { pipeline, bindGroup: dev.bindGroupFor(pipeline, buffers), groups: [grid.x, grid.y, 1], buffers };
}
