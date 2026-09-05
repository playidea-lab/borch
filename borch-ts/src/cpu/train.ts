/**
 * The other half of the workbench on the CPU device: a linear head trained on cached
 * features, and cosine neighbours over the same features.
 *
 * ## Why only this much training
 *
 * The device runs a frozen backbone forward (`runner.ts`) and stops. What a labelling
 * tool trains is a head on top of the features that forward produced — a linear layer,
 * softmax cross-entropy, SGD with momentum — and that is the whole of the training this
 * file knows. The backward of a linear layer is two products; the backward of
 * cross-entropy is `softmax − onehot`. Everything past that (a second layer, a different
 * loss, Adam) is absent by name, and `torch_gap` is where the reasons for absences live.
 *
 * ## Layout
 *
 * Features `[N × D]` are uploaded once per `fit`, rows padded to four (zero) for `gemm`.
 * The weight is kept as `[D × K]` — the transpose of torch's `[K × D]` — because that is
 * what `gemm` consumes; `stateDict()` hands it back in torch's order. Classes are padded
 * to sixteen and the padding columns never receive gradient (the cross-entropy kernel
 * writes them zero), so they stay at their zero initialisation.
 *
 * ## Memory
 *
 * The head's own buffers are allocated on construction and live until the module is
 * `reset`; everything a `fit` or `predict` needs is allocated above the heap mark and the
 * heap is returned there after. A page that trains many heads should reuse one.
 */
import { ACT, type CpuKernels } from "./load.js";
import { CHANNEL_PAD, pad } from "./graph.js";

export interface HeadOptions {
  readonly lr: number;
  readonly momentum?: number;
  readonly weightDecay?: number;
}

export interface HeadState {
  /** torch layout, `[numClasses × inFeatures]`. */
  readonly weight: Float32Array;
  readonly bias: Float32Array;
}

export class LinearHead {
  readonly kP: number;
  private readonly w: number;
  private readonly b: number;
  private readonly vw: number;
  private readonly vb: number;

  constructor(
    private readonly K: CpuKernels,
    readonly inFeatures: number,
    readonly numClasses: number,
    private readonly opts: HeadOptions,
    init?: HeadState,
  ) {
    if (inFeatures % 4 !== 0) throw new Error(`cpu head: inFeatures must be a multiple of 4, got ${inFeatures}`);
    this.kP = pad(numClasses, CHANNEL_PAD);
    const D = inFeatures, KP = this.kP;
    const grab = (n: number): number => { const off = K.alloc(n * 4); if (off === 0) throw new Error("cpu head: the wasm memory would not grow"); K.zero(n, off); return off; };
    this.w = grab(D * KP); this.b = grab(KP); this.vw = grab(D * KP); this.vb = grab(KP);
    if (init) this.load(init);
  }

  private f32(): Float32Array { return new Float32Array(this.K.memory.buffer); }

  /** Overwrite the parameters from torch-layout arrays. Momentum is not reset. */
  load(state: HeadState): void {
    const { inFeatures: D, numClasses: Kc, kP: KP } = this;
    if (state.weight.length !== Kc * D || state.bias.length !== Kc) throw new Error(`cpu head: state is [${state.weight.length}], [${state.bias.length}]; expected [${Kc}×${D}], [${Kc}]`);
    const f = this.f32();
    for (let k = 0; k < Kc; k++) {
      for (let d = 0; d < D; d++) f[this.w / 4 + d * KP + k] = state.weight[k * D + d] ?? 0;
      f[this.b / 4 + k] = state.bias[k] ?? 0;
    }
  }

  stateDict(): HeadState {
    const { inFeatures: D, numClasses: Kc, kP: KP } = this;
    const f = this.f32();
    const weight = new Float32Array(Kc * D), bias = new Float32Array(Kc);
    for (let k = 0; k < Kc; k++) {
      for (let d = 0; d < D; d++) weight[k * D + d] = f[this.w / 4 + d * KP + k] ?? 0;
      bias[k] = f[this.b / 4 + k] ?? 0;
    }
    return { weight, bias };
  }

  /** Upload `[N × D]` features as `[NP × D]`, padding rows zero. Returns the offset; caller owns the heap mark. */
  private upload(features: Float32Array, N: number): { off: number; NP: number } {
    const D = this.inFeatures;
    if (features.length !== N * D) throw new Error(`cpu head: ${features.length} values is not ${N}×${D}`);
    const NP = pad(N, 4);
    const off = this.K.alloc(NP * D * 4);
    if (off === 0) throw new Error("cpu head: the wasm memory would not grow for the features");
    const f = this.f32();
    f.set(features, off / 4);
    if (NP > N) f.fill(0, off / 4 + N * D, off / 4 + NP * D);
    return { off, NP };
  }

  /** Logits `[N × numClasses]` for `[N × D]` features. */
  predict(features: Float32Array, N: number): Float32Array {
    const { K, inFeatures: D, numClasses: Kc, kP: KP } = this;
    const mark = K.heap();
    try {
      const x = this.upload(features, N);
      const y = K.alloc(x.NP * KP * 4);
      if (y === 0) throw new Error("cpu head: the wasm memory would not grow");
      K.gemmBiasAct(x.NP, KP, D, x.off, this.w, y, this.b, ACT.none);
      const f = this.f32();
      const out = new Float32Array(N * Kc);
      for (let r = 0; r < N; r++) out.set(f.subarray(y / 4 + r * KP, y / 4 + r * KP + Kc), r * Kc);
      return out;
    } finally {
      K.setHeap(mark);
    }
  }

  /**
   * Full-batch SGD for `steps` steps on `[N × D]` features with integer class labels.
   * Returns the mean cross-entropy before each step — the same numbers torch prints.
   */
  fit(features: Float32Array, labels: ArrayLike<number>, N: number, steps: number): Float32Array {
    const { K, inFeatures: D, numClasses: Kc, kP: KP } = this;
    if (labels.length !== N) throw new Error(`cpu head: ${labels.length} labels for ${N} rows`);
    const lr = this.opts.lr, momentum = this.opts.momentum ?? 0, wd = this.opts.weightDecay ?? 0;
    const mark = K.heap();
    try {
      const x = this.upload(features, N);
      const grab = (n: number): number => { const off = K.alloc(n * 4); if (off === 0) throw new Error("cpu head: the wasm memory would not grow"); return off; };
      const lab = grab(pad(N, 4)), y = grab(x.NP * KP), g = grab(x.NP * KP), stats = grab(pad(2 * N, 4)), dw = grab(D * KP), db = grab(KP);
      {
        const f = this.f32();
        for (let r = 0; r < N; r++) f[lab / 4 + r] = labels[r] ?? 0;
        f.fill(0, g / 4 + N * KP, g / 4 + x.NP * KP);
      }
      const losses = new Float32Array(steps);
      for (let step = 0; step < steps; step++) {
        K.gemmBiasAct(x.NP, KP, D, x.off, this.w, y, this.b, ACT.none);
        K.softmaxXentGrad(N, KP, Kc, y, lab, g, stats);
        // loss = mean(−(l[label] − max − ln Σexp)), finished here because the module has no ln
        const f = this.f32();
        let loss = 0;
        for (let r = 0; r < N; r++) {
          const label = labels[r] ?? 0;
          const l = f[y / 4 + r * KP + label] ?? 0, mx = f[stats / 4 + 2 * r] ?? 0, s = f[stats / 4 + 2 * r + 1] ?? 1;
          loss -= l - mx - Math.log(s);
        }
        losses[step] = loss / N;
        // dW = Xᵀ G ; db = Σ_rows G
        K.zero(D * KP, dw);
        K.outerAcc(N, D, KP, x.off, g, dw);
        K.meanRows(N, KP, g, db);
        { const fb = this.f32(); for (let k = 0; k < KP; k++) fb[db / 4 + k] = (fb[db / 4 + k] ?? 0) * N; }
        K.sgdStep(D * KP, this.w, dw, this.vw, lr, momentum, wd);
        K.sgdStep(KP, this.b, db, this.vb, lr, momentum, wd);
      }
      return losses;
    } finally {
      K.setHeap(mark);
    }
  }
}

export interface Neighbours {
  /** `[N × k]` indices of the nearest rows by cosine similarity, self excluded, nearest first. */
  readonly indices: Int32Array;
  /** `[N × k]` the similarities, same order. */
  readonly sims: Float32Array;
}

/**
 * The `k` nearest rows to every row of `[N × D]` features, by cosine similarity, self
 * excluded. Rows are normalised once; the similarity matrix is `gemm` against the
 * transpose, produced a block of rows at a time so 5,000 features do not mean a 100 MB
 * matrix — each block is read for its top `k` and dropped.
 */
export function cosineNeighbours(K: CpuKernels, features: Float32Array, N: number, D: number, k: number, block = 256): Neighbours {
  if (features.length !== N * D) throw new Error(`cpu neighbours: ${features.length} values is not ${N}×${D}`);
  if (D % 4 !== 0) throw new Error(`cpu neighbours: D must be a multiple of 4, got ${D}`);
  if (k >= N) throw new Error(`cpu neighbours: k=${k} needs more than ${N} rows`);
  const NP = pad(N, 16);
  const mark = K.heap();
  try {
    const grab = (n: number): number => { const off = K.alloc(n * 4); if (off === 0) throw new Error("cpu neighbours: the wasm memory would not grow"); return off; };
    const x = grab(NP * D), xt = grab(D * NP);
    {
      const f = new Float32Array(K.memory.buffer);
      f.set(features, x / 4);
      f.fill(0, x / 4 + N * D, x / 4 + NP * D);
    }
    K.l2NormalizeRows(N, D, x);
    K.transpose(NP, D, x, xt);
    const R = pad(Math.min(block, NP), 4);
    const s = grab(R * NP);
    const indices = new Int32Array(N * k), sims = new Float32Array(N * k);
    for (let r0 = 0; r0 < N; r0 += R) {
      const rows = Math.min(R, NP - r0);
      K.gemm(pad(rows, 4), NP, D, x + r0 * D * 4, xt, s);
      const f = new Float32Array(K.memory.buffer);
      for (let r = 0; r < rows && r0 + r < N; r++) {
        const row = f.subarray(s / 4 + r * NP, s / 4 + r * NP + N);
        const self = r0 + r;
        // partial selection: keep the k best seen so far, insertion-sorted (k is small)
        const bestI = new Int32Array(k).fill(-1), bestS = new Float32Array(k).fill(-Infinity);
        for (let j = 0; j < N; j++) {
          if (j === self) continue;
          const v = row[j] ?? -Infinity;
          if (v <= (bestS[k - 1] ?? -Infinity)) continue;
          let pos = k - 1;
          while (pos > 0 && (bestS[pos - 1] ?? -Infinity) < v) { bestS[pos] = bestS[pos - 1] ?? -Infinity; bestI[pos] = bestI[pos - 1] ?? -1; pos--; }
          bestS[pos] = v; bestI[pos] = j;
        }
        indices.set(bestI, self * k); sims.set(bestS, self * k);
      }
    }
    return { indices, sims };
  } finally {
    K.setHeap(mark);
  }
}
