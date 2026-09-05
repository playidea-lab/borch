/**
 * Runs a `CpuGraph` on the wasm kernels.
 *
 * ## Memory
 *
 * The module has one linear memory and a bump allocator. Weights go in once, at
 * construction, and stay. Activations come and go per forward, and a bump allocator
 * cannot free one in the middle — so each forward opens a pool keyed by byte size on
 * top of it, starting wherever the heap stands (so a head trained beside this runner
 * keeps its buffers): a buffer freed at its value's last use goes back to the pool, the
 * next allocation of the same size takes it, and the forward's end returns the heap to
 * where it began. Shapes repeat block after block, so the pool absorbs most of a
 * forward; the peak is roughly the largest live set, not the sum. Dense convolutions
 * unroll their taps a block of rows at a time into one 8 MB buffer rather than the whole
 * image — the book carries the measured peaks.
 *
 * ## Epilogues
 *
 * The bias and the activation ride on the GEMM and the depthwise kernel (`gemm_bias_act`,
 * `dwconv_bias_act`) instead of a second pass over the result; measured before the fold,
 * that pass was a fifth of a forward.
 *
 * ## Layout
 *
 * NHWC with the batch folded into rows, every channel count padded to sixteen. The
 * image arrives NCHW (torch's convention, and `Tensor`'s) and is transposed on the way
 * in; the result leaves as `[B, channels]` with the padding trimmed. Rows are padded to a
 * multiple of four for `gemm`, and the padding rows are zeroed at allocation so nothing
 * stale flows into them.
 *
 * ## In place
 *
 * Squeeze-and-excite scales its input where it stands and `add` accumulates into its
 * first operand — both would otherwise cost one more pass over the largest activations.
 * The value produced takes over the buffer, and the operand that gave it up is not
 * freed on its own. That is the one subtlety in the liveness bookkeeping below.
 */
import { ACT, type CpuKernels } from "./load.js";
import type { CpuGraph, Node } from "./graph.js";

/** The im2col buffer a blocked convolution reuses. 8 MB holds 3,640 rows of a ResNet-18 first-stage column (576 wide). */
const COL_BYTES = 8 * 1024 * 1024;

interface Value {
  off: number; bytes: number;
  rows: number; rowsP: number; h: number; w: number; cP: number;
  /** Whether freeing this value frees `off`. False after the buffer was handed to another value. */
  owns: boolean;
}

class Pool {
  private readonly free = new Map<number, number[]>();
  private readonly base: number;
  /** Starts at wherever the heap stands — everything below it (weights, other users' buffers) is left alone. */
  constructor(private readonly K: CpuKernels) { this.base = K.heap(); }
  alloc(bytes: number): number {
    const list = this.free.get(bytes);
    const reused = list?.pop();
    if (reused !== undefined) return reused;
    const off = this.K.alloc(bytes);
    if (off === 0) throw new Error(`cpu runner: the wasm memory would not grow by ${bytes} bytes`);
    return off;
  }
  give(off: number, bytes: number): void {
    const list = this.free.get(bytes);
    if (list) list.push(off); else this.free.set(bytes, [off]);
  }
  /** Return the heap to where this pool began. Only what the pool handed out is gone. */
  release(): void { this.free.clear(); this.K.setHeap(this.base); }
}

export class CpuRunner {
  private readonly weights: number[] = [];
  private readonly lastUse: number[];

  constructor(private readonly K: CpuKernels, private readonly graph: CpuGraph) {
    // Weights first, so the activation arena starts above them and `reset` never touches them.
    for (const node of graph.nodes) this.weights.push(...this.upload(node));
    this.lastUse = graph.nodes.map(() => -1);
    graph.nodes.forEach((node, i) => { for (const dep of inputsOf(node)) this.lastUse[dep] = i; });
    this.lastUse[graph.output] = graph.nodes.length; // the output outlives the loop
  }

  /** Copies a node's packed weights into the module and records where they went. */
  private upload(node: Node): number[] {
    const put = (a: Float32Array): number => {
      const off = this.K.alloc(a.byteLength);
      if (off === 0) throw new Error("cpu runner: the wasm memory would not grow for the weights");
      new Float32Array(this.K.memory.buffer).set(a, off / 4);
      return off;
    };
    switch (node.kind) {
      case "conv": case "dwconv": case "linear": return [put(node.weight), put(node.bias)];
      case "se": return [put(node.w1), put(node.b1), put(node.w2), put(node.b2)];
      default: return [];
    }
  }

  private weightOffsets(): Map<number, number[]> {
    const map = new Map<number, number[]>();
    let at = 0;
    this.graph.nodes.forEach((node, i) => {
      const n = node.kind === "se" ? 4 : (node.kind === "conv" || node.kind === "dwconv" || node.kind === "linear") ? 2 : 0;
      map.set(i, this.weights.slice(at, at + n)); at += n;
    });
    return map;
  }

  private f32(): Float32Array { return new Float32Array(this.K.memory.buffer); }

  /**
   * `slack` extra rows on top of the padding — a blocked convolution's last `gemm` may write
   * up to three rows past an image's own, and for the last image that is past the value.
   */
  private value(pool: Pool, rows: number, h: number, w: number, cP: number, slack = 0): Value {
    const rowsP = Math.ceil(rows / 4) * 4 + slack;
    const bytes = rowsP * cP * 4;
    const off = pool.alloc(bytes);
    if (rowsP > rows) this.f32().fill(0, off / 4 + rows * cP, off / 4 + rowsP * cP);
    return { off, bytes, rows, rowsP, h, w, cP, owns: true };
  }

  /**
   * One forward. `input` is NCHW, `[B, channels, H, W]`; the result is `[B, outputChannels]`.
   */
  forward(input: Float32Array, B: number, H: number, W: number): Float32Array {
    const { K, graph } = this;
    const offsets = this.weightOffsets();
    const values = new Map<number, Value>();
    const at = (id: number): Value => { const v = values.get(id); if (!v) throw new Error(`cpu runner: node ${id} has no value`); return v; };
    const outShape = (v: Value, k: number, stride: number, pad: number): [number, number] => [Math.floor((v.h + 2 * pad - k) / stride) + 1, Math.floor((v.w + 2 * pad - k) / stride) + 1];

    // The activation arena begins wherever the heap stands now and ends there again.
    const pool = new Pool(K);
    try {
      graph.nodes.forEach((node, i) => {
        const wts = offsets.get(i) ?? [];
        const w = (j: number): number => { const o = wts[j]; if (o === undefined) throw new Error(`cpu runner: node ${i} has no weight ${j}`); return o; };
        switch (node.kind) {
          case "input": {
            const c = node.channels;
            if (input.length !== B * c * H * W) throw new Error(`cpu runner: input has ${input.length} values, expected ${B}×${c}×${H}×${W}`);
            const v = this.value(pool, B * H * W, H, W, c);
            const f = this.f32(); const base = v.off / 4; const HW = H * W;
            for (let b = 0; b < B; b++) for (let ch = 0; ch < c; ch++) {
              const src = (b * c + ch) * HW; const dst = base + b * HW * c + ch;
              for (let p = 0; p < HW; p++) f[dst + p * c] = input[src + p] ?? 0;
            }
            values.set(i, v);
            break;
          }
          case "conv": {
            const x = at(node.input);
            const [ho, wo] = outShape(x, node.k, node.stride, node.pad);
            if (node.k === 1 && node.stride === 1 && node.pad === 0) {
              const y = this.value(pool, B * ho * wo, ho, wo, node.coutP);
              K.gemmBiasAct(y.rowsP, node.coutP, node.cinP, x.off, w(0), y.off, w(1), node.act);
              values.set(i, y);
            } else {
              // A block of output rows at a time: unroll those rows' taps into one reused
              // buffer, multiply, write the rows in place. ResNet-18's first stage at batch 16
              // was 116 MB of columns per layer unrolled whole; a block is COL_BYTES.
              const kk = node.k * node.k * node.cinP, per = ho * wo;
              const R = Math.max(4, Math.min(Math.ceil(per / 4) * 4, Math.floor(COL_BYTES / (kk * 4) / 4) * 4));
              const y = this.value(pool, B * per, ho, wo, node.coutP, 4);
              const col = this.value(pool, R, 1, 1, kk);
              for (let b = 0; b < B; b++) {
                for (let r0 = 0; r0 < per; r0 += R) {
                  const rows = Math.min(R, per - r0);
                  K.im2colRows(x.h, x.w, node.cinP, node.k, node.stride, node.pad, wo, r0, rows, x.off + b * x.h * x.w * node.cinP * 4, col.off);
                  K.gemmBiasAct(Math.ceil(rows / 4) * 4, node.coutP, kk, col.off, w(0), y.off + (b * per + r0) * node.coutP * 4, w(1), node.act);
                }
              }
              pool.give(col.off, col.bytes);
              // the last block may have written into the slack rows; keep them zero for whoever reads rowsP
              if (y.rowsP > y.rows) this.f32().fill(0, y.off / 4 + y.rows * y.cP, y.off / 4 + y.rowsP * y.cP);
              values.set(i, y);
            }
            break;
          }
          case "dwconv": {
            const x = at(node.input);
            const [ho, wo] = outShape(x, node.k, node.stride, node.pad);
            const y = this.value(pool, B * ho * wo, ho, wo, node.cP);
            for (let b = 0; b < B; b++) K.dwconvBiasAct(x.h, x.w, node.cP, node.k, node.stride, node.pad, ho, wo, x.off + b * x.h * x.w * node.cP * 4, w(0), y.off + b * ho * wo * node.cP * 4, w(1), node.act);
            values.set(i, y);
            break;
          }
          case "maxpool": {
            const x = at(node.input);
            const [ho, wo] = outShape(x, node.k, node.stride, node.pad);
            const y = this.value(pool, B * ho * wo, ho, wo, x.cP);
            for (let b = 0; b < B; b++) K.maxpool(x.h, x.w, x.cP, node.k, node.stride, node.pad, ho, wo, x.off + b * x.h * x.w * x.cP * 4, y.off + b * ho * wo * x.cP * 4);
            values.set(i, y);
            break;
          }
          case "se": {
            const x = at(node.input);
            const HW = x.h * x.w;
            const sq = this.value(pool, B, 1, 1, node.cP);
            for (let b = 0; b < B; b++) K.meanRows(HW, node.cP, x.off + b * HW * node.cP * 4, sq.off + b * node.cP * 4);
            const r1 = this.value(pool, B, 1, 1, node.cseP);
            K.gemmBiasAct(sq.rowsP, node.cseP, node.cP, sq.off, w(0), r1.off, w(1), ACT.swish);
            const g = this.value(pool, B, 1, 1, node.cP);
            K.gemmBiasAct(r1.rowsP, node.cP, node.cseP, r1.off, w(2), g.off, w(3), ACT.sigmoid);
            for (let b = 0; b < B; b++) K.scaleRows(HW, node.cP, x.off + b * HW * node.cP * 4, g.off + b * node.cP * 4);
            for (const t of [sq, r1, g]) pool.give(t.off, t.bytes);
            values.set(i, { ...x, owns: true }); x.owns = false; // the value moves into x's buffer
            break;
          }
          case "add": {
            const a = at(node.input), b = at(node.other);
            if (a.rows !== b.rows || a.cP !== b.cP) throw new Error(`cpu runner: add of ${a.rows}×${a.cP} and ${b.rows}×${b.cP}`);
            K.addInplace(a.rowsP * a.cP, a.off, b.off);
            if (node.act === ACT.relu) K.relu(a.rowsP * a.cP, a.off);
            else if (node.act !== ACT.none) throw new Error(`cpu runner: add supports relu or nothing, got act ${node.act}`);
            values.set(i, { ...a, owns: true }); a.owns = false;
            break;
          }
          case "gap": {
            const x = at(node.input);
            const HW = x.h * x.w;
            const p = this.value(pool, B, 1, 1, x.cP);
            for (let b = 0; b < B; b++) K.meanRows(HW, x.cP, x.off + b * HW * x.cP * 4, p.off + b * x.cP * 4);
            values.set(i, p);
            break;
          }
          case "linear": {
            const x = at(node.input);
            const y = this.value(pool, x.rows, 1, 1, node.coutP);
            K.gemmBiasAct(y.rowsP, node.coutP, node.cinP, x.off, w(0), y.off, w(1), node.act);
            values.set(i, y);
            break;
          }
        }
        // Free whatever saw its last use here — except a buffer that a later value now owns.
        for (const dep of inputsOf(node)) {
          if (this.lastUse[dep] === i) { const v = at(dep); if (v.owns) pool.give(v.off, v.bytes); }
        }
      });
      const out = at(graph.output);
      const f = this.f32();
      const result = new Float32Array(out.rows * graph.outputChannels);
      for (let r = 0; r < out.rows; r++) result.set(f.subarray(out.off / 4 + r * out.cP, out.off / 4 + r * out.cP + graph.outputChannels), r * graph.outputChannels);
      return result;
    } finally {
      pool.release();
    }
  }
}

function inputsOf(node: Node): number[] {
  switch (node.kind) {
    case "input": return [];
    case "add": return [node.input, node.other];
    default: return [node.input];
  }
}
