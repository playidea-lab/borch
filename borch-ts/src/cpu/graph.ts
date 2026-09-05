/**
 * The graph the CPU device runs — a short list of nodes, weights already packed for the
 * kernels, BatchNorm already folded in.
 *
 * ## Why a graph and not `nn.Module`
 *
 * On the machine this device is for there is no WebGPU adapter, so there is no `Tensor`
 * and no `nn.Module` instance to walk. What there is: the checkpoint's bytes and the
 * knowledge of what shape the network has. This file turns the two into a list the
 * runner executes. The list is deliberately small — a conv, a depthwise conv, a max
 * pool, squeeze-and-excite, an add, a global average pool, a linear. Every classifier
 * in the catalogue is made of these; anything else is absent by name.
 *
 * ## What the builder does at build time
 *
 * - **Folds BatchNorm into the convolution before it** (eval mode; `scale = γ/√(σ²+ε)`,
 *   `W' = W·scale`, `b' = β − μ·scale`). One pass fewer over memory per layer.
 * - **Repacks weights** from torch's `[Cout, Cin, kh, kw]` into the tap-major
 *   `[kh·kw·Cin, Cout]` that `im2col` + `gemm` want, NHWC throughout.
 * - **Pads channels to sixteen** and remembers both counts. Padded weight columns are
 *   zero, so padded activation channels are exactly zero at every layer — which is what
 *   lets the output be compared with the unpadded network digit for digit.
 */
import { ACT, type Act } from "./load.js";

/** Channels are padded to this. `dwconv` wants 16; everything else would take 4. */
export const CHANNEL_PAD = 16;
export const pad = (v: number, m: number): number => Math.ceil(v / m) * m;

export interface BatchNorm {
  readonly weight: Float32Array;
  readonly bias: Float32Array;
  readonly runningMean: Float32Array;
  readonly runningVar: Float32Array;
  readonly eps?: number;
}

interface Meta { readonly channels: number; readonly channelsP: number }

export interface InputNode { readonly kind: "input"; readonly channels: number; readonly channelsP: number }
export interface ConvNode {
  readonly kind: "conv"; readonly input: number;
  readonly k: number; readonly stride: number; readonly pad: number;
  readonly cinP: number; readonly cout: number; readonly coutP: number;
  /** `[k·k·cinP × coutP]`, tap-major. */ readonly weight: Float32Array;
  readonly bias: Float32Array; readonly act: Act;
}
export interface DwConvNode {
  readonly kind: "dwconv"; readonly input: number;
  readonly k: number; readonly stride: number; readonly pad: number; readonly cP: number;
  /** `[k·k × cP]`. */ readonly weight: Float32Array; readonly bias: Float32Array; readonly act: Act;
}
export interface MaxPoolNode { readonly kind: "maxpool"; readonly input: number; readonly k: number; readonly stride: number; readonly pad: number }
export interface SeNode {
  readonly kind: "se"; readonly input: number; readonly cP: number; readonly cseP: number;
  readonly w1: Float32Array; readonly b1: Float32Array; readonly w2: Float32Array; readonly b2: Float32Array;
}
export interface AddNode { readonly kind: "add"; readonly input: number; readonly other: number; readonly act: Act }
export interface GapNode { readonly kind: "gap"; readonly input: number }
export interface LinearNode {
  readonly kind: "linear"; readonly input: number; readonly cinP: number; readonly cout: number; readonly coutP: number;
  /** `[cinP × coutP]`. */ readonly weight: Float32Array; readonly bias: Float32Array; readonly act: Act;
}
export type Node = InputNode | ConvNode | DwConvNode | MaxPoolNode | SeNode | AddNode | GapNode | LinearNode;

export interface CpuGraph {
  readonly nodes: readonly Node[];
  /** The node whose value `forward` returns. */
  readonly output: number;
  /** Channels of the output, unpadded. */
  readonly outputChannels: number;
}

export interface ConvSpec {
  /** torch layout `[cout, cin, k, k]` (or `[c, 1, k, k]` for depthwise). */
  readonly weight: Float32Array; readonly cout: number; readonly cin: number; readonly k: number;
  readonly stride: number; readonly pad: number;
  readonly bias?: Float32Array; readonly bn?: BatchNorm; readonly act?: Act;
}

function fold(cout: number, bn: BatchNorm | undefined, bias: Float32Array | undefined): { scale: Float32Array; shift: Float32Array } {
  const scale = new Float32Array(cout).fill(1);
  const shift = new Float32Array(cout);
  if (bias) shift.set(bias.subarray(0, cout));
  if (bn) {
    const eps = bn.eps ?? 1e-5;
    for (let o = 0; o < cout; o++) {
      const s = (bn.weight[o] ?? 1) / Math.sqrt((bn.runningVar[o] ?? 1) + eps);
      scale[o] = s;
      shift[o] = (bn.bias[o] ?? 0) + ((shift[o] ?? 0) - (bn.runningMean[o] ?? 0)) * s;
    }
  }
  return { scale, shift };
}

/** Builds a graph node by node, tracking each value's channel count and its padding. */
export class GraphBuilder {
  private readonly nodes: Node[] = [];
  private readonly meta: Meta[] = [];

  private push(node: Node, channels: number, channelsP: number): number {
    this.nodes.push(node);
    this.meta.push({ channels, channelsP });
    return this.nodes.length - 1;
  }

  private metaOf(id: number): Meta {
    const m = this.meta[id];
    if (!m) throw new Error(`cpu graph: no node ${id}`);
    return m;
  }

  /** The image, NCHW at run time, `channels` deep (three for RGB). Never padded. */
  input(channels: number): number {
    return this.push({ kind: "input", channels, channelsP: channels }, channels, channels);
  }

  /** A dense convolution, BatchNorm folded, weights repacked tap-major. */
  conv(x: number, s: ConvSpec): number {
    const { channelsP: cinP, channels: cin } = this.metaOf(x);
    if (cin !== s.cin) throw new Error(`cpu graph: conv expects ${s.cin} channels, input has ${cin}`);
    const coutP = pad(s.cout, CHANNEL_PAD);
    const { scale, shift } = fold(s.cout, s.bn, s.bias);
    const weight = new Float32Array(s.k * s.k * cinP * coutP);
    for (let o = 0; o < s.cout; o++) for (let i = 0; i < s.cin; i++) for (let ky = 0; ky < s.k; ky++) for (let kx = 0; kx < s.k; kx++) {
      weight[((ky * s.k + kx) * cinP + i) * coutP + o] = (s.weight[((o * s.cin + i) * s.k + ky) * s.k + kx] ?? 0) * (scale[o] ?? 1);
    }
    const bias = new Float32Array(coutP); bias.set(shift);
    return this.push({ kind: "conv", input: x, k: s.k, stride: s.stride, pad: s.pad, cinP, cout: s.cout, coutP, weight, bias, act: s.act ?? ACT.none }, s.cout, coutP);
  }

  /** A depthwise convolution (`groups == channels`), BatchNorm folded. */
  dwconv(x: number, s: ConvSpec): number {
    const { channelsP: cP, channels: c } = this.metaOf(x);
    if (c !== s.cin || s.cout !== s.cin) throw new Error(`cpu graph: depthwise conv on ${c} channels got ${s.cin}→${s.cout}`);
    const { scale, shift } = fold(c, s.bn, s.bias);
    const weight = new Float32Array(s.k * s.k * cP);
    for (let i = 0; i < c; i++) for (let ky = 0; ky < s.k; ky++) for (let kx = 0; kx < s.k; kx++) {
      weight[(ky * s.k + kx) * cP + i] = (s.weight[(i * s.k + ky) * s.k + kx] ?? 0) * (scale[i] ?? 1);
    }
    const bias = new Float32Array(cP); bias.set(shift);
    return this.push({ kind: "dwconv", input: x, k: s.k, stride: s.stride, pad: s.pad, cP, weight, bias, act: s.act ?? ACT.none }, c, cP);
  }

  maxpool(x: number, k: number, stride: number, padding: number): number {
    const m = this.metaOf(x);
    return this.push({ kind: "maxpool", input: x, k, stride, pad: padding }, m.channels, m.channelsP);
  }

  /**
   * Squeeze-and-excite: mean over the image → 1×1 `reduce` (+bias, swish) → 1×1 `expand`
   * (+bias, sigmoid) → scale the input by it. Weights in torch's `[out, in, 1, 1]`.
   */
  se(x: number, reduceW: Float32Array, reduceB: Float32Array, expandW: Float32Array, expandB: Float32Array, cse: number): number {
    const { channels: c, channelsP: cP } = this.metaOf(x);
    const cseP = pad(cse, CHANNEL_PAD);
    const w1 = new Float32Array(cP * cseP), w2 = new Float32Array(cseP * cP);
    for (let r = 0; r < cse; r++) for (let i = 0; i < c; i++) {
      w1[i * cseP + r] = reduceW[r * c + i] ?? 0;
      w2[r * cP + i] = expandW[i * cse + r] ?? 0;
    }
    const b1 = new Float32Array(cseP); b1.set(reduceB.subarray(0, cse));
    const b2 = new Float32Array(cP); b2.set(expandB.subarray(0, c));
    return this.push({ kind: "se", input: x, cP, cseP, w1, b1, w2, b2 }, c, cP);
  }

  /** `a + b`, then `act`. Both the same shape. */
  add(a: number, b: number, act: Act = ACT.none): number {
    const ma = this.metaOf(a), mb = this.metaOf(b);
    if (ma.channels !== mb.channels) throw new Error(`cpu graph: add of ${ma.channels} and ${mb.channels} channels`);
    return this.push({ kind: "add", input: a, other: b, act }, ma.channels, ma.channelsP);
  }

  /** Global average pool: `[B, H, W, C]` → `[B, C]`. */
  gap(x: number): number {
    const m = this.metaOf(x);
    return this.push({ kind: "gap", input: x }, m.channels, m.channelsP);
  }

  /** A linear layer on `[B, cin]`. Weight in torch's `[out, in]`. */
  linear(x: number, weight: Float32Array, bias: Float32Array | undefined, cout: number, act: Act = ACT.none): number {
    const { channels: cin, channelsP: cinP } = this.metaOf(x);
    const coutP = pad(cout, CHANNEL_PAD);
    const w = new Float32Array(cinP * coutP);
    for (let o = 0; o < cout; o++) for (let i = 0; i < cin; i++) w[i * coutP + o] = weight[o * cin + i] ?? 0;
    const b = new Float32Array(coutP); if (bias) b.set(bias.subarray(0, cout));
    return this.push({ kind: "linear", input: x, cinP, cout, coutP, weight: w, bias: b, act }, cout, coutP);
  }

  /** Seal the graph with the node whose value is the network's output. */
  finish(output: number): CpuGraph {
    const m = this.metaOf(output);
    return { nodes: [...this.nodes], output, outputChannels: m.channels };
  }
}
