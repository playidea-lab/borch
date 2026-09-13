/**
 * Parameter-efficient fine-tuning — borch's own namespace, **not torch's.**
 *
 * torch.nn has no LoRA, so these do not live in `nn`: a learner who typed
 * `nn.LoRALinear` would meet `AttributeError` on real torch, which is the one thing
 * this project positions against (`tests/test_gap.py`). The ecosystem name for this is
 * `peft` (HuggingFace), and it is honestly non-torch, so it sits here as `borch.peft`.
 *
 * It lands in borch.ts first. The numpy core and the Pyodide binding mirror it when the
 * transfer-learning / federated direction earns the build — declared, not a silent split.
 *
 * Each layer freezes a base and trains a low-rank adapter beside it; `B` (the second
 * factor) starts at zero, so before the first step the layer is exactly its base. The
 * adapter is what {@link LoRALinear.adapterState} hands out — the KB-sized thing a
 * federated round sends instead of a whole model — and `merge()` folds it back into a
 * plain layer for inference or ONNX export. torch.nn has no LoRA to match, so these are
 * checked by their invariants (borch-ts/test/lora.py), not a golden.
 */
import { ValueError } from "./errors.js";
import { Conv2d, Linear, Module } from "./nn.js";
import { uniformArray } from "./random.js";
import { noGrad, Tensor } from "./tensor.js";

/** A uniform tensor in [-bound, bound] — the same init `nn` uses, kept local to peft. */
function uniform(shape: readonly number[], bound: number): Tensor {
  const n = shape.reduce((a, b) => a * b, 1);
  return Tensor.from(uniformArray(n, bound), shape);
}

/** Options for {@link LoRALinear} and {@link LoRAConv2d}. `alpha` defaults to `r`. */
export interface LoRAOptions { r?: number; alpha?: number; bias?: boolean; }

/**
 * A `Linear` whose base weight is frozen and adapted by a trainable low-rank pair:
 *
 *     y = x·Wᵀ + b  +  (x·Aᵀ)·Bᵀ · (alpha / r)
 *
 * `A` is `(r, in)` and `B` is `(out, r)` with `r ≪ min(in, out)`; only `A` and `B`
 * train. The base `W`, `b` are frozen — registered as buffers, so they stay in
 * `stateDict()` but out of `parameters()`, and an optimiser touches only the adapter.
 * `B` starts at zero (the standard LoRA init, Hu et al. 2021).
 */
export class LoRALinear extends Module {
  readonly weight: Tensor;       // frozen base (out, in)
  readonly bias: Tensor | null;  // frozen base
  readonly loraA: Tensor;        // (r, in), trained
  readonly loraB: Tensor;        // (out, r), trained, starts at zero
  readonly r: number;
  readonly alpha: number;
  readonly scaling: number;      // alpha / r

  constructor(inFeatures: number, outFeatures: number, options: LoRAOptions = {}) {
    super();
    const { r = 8, alpha = r, bias = true } = options;
    if (r <= 0) throw new ValueError(`LoRALinear needs r >= 1, got ${r}`);
    const bound = 1 / Math.sqrt(Math.max(1, inFeatures));
    this.weight = uniform([outFeatures, inFeatures], bound);
    this.bias = bias ? uniform([outFeatures], bound) : null;
    this.weight.requiresGrad = false;
    this.registerBuffer("weight", this.weight);
    if (this.bias) { this.bias.requiresGrad = false; this.registerBuffer("bias", this.bias); }
    this.loraA = uniform([r, inFeatures], bound);
    this.loraB = Tensor.zeros([outFeatures, r]);
    this.claim(this.loraA, this.loraB);
    this.r = r; this.alpha = alpha; this.scaling = alpha / r;
  }

  /** Wrap an existing (trained) `Linear`: its weight/bias become the frozen base. */
  static fromLinear(linear: Linear, options: LoRAOptions = {}): LoRALinear {
    const [out, inF] = [linear.weight.shape[0] ?? 0, linear.weight.shape[1] ?? 0];
    const lora = new LoRALinear(inF, out, { ...options, bias: linear.bias != null });
    const w = lora as { weight: Tensor; bias: Tensor | null };
    w.weight = linear.weight; w.weight.requiresGrad = false; lora.registerBuffer("weight", w.weight);
    if (linear.bias) { w.bias = linear.bias; w.bias.requiresGrad = false; lora.registerBuffer("bias", w.bias); }
    return lora;
  }

  /** The adapter alone — the KB-sized pair a federated round sends. */
  adapterState(): Record<string, Tensor> {
    return { lora_A: this.loraA, lora_B: this.loraB };
  }

  override ownParameters(): Record<string, Tensor> {
    return { lora_A: this.loraA, lora_B: this.loraB };
  }

  override forward(x: Tensor): Tensor {
    const base = this.bias ? x.linear(this.weight).add(this.bias) : x.linear(this.weight);
    const delta = x.linear(this.loraA).linear(this.loraB).mul(Tensor.full([], this.scaling));
    return base.add(delta);
  }

  /** Fold the adapter into the base and return a plain `Linear` (for inference / export):
   *  W' = W + (alpha/r)·(B·A), b' = b. */
  merge(): Linear {
    const [out, inF] = [this.weight.shape[0] ?? 0, this.weight.shape[1] ?? 0];
    const merged = new Linear(inF, out, this.bias != null);
    const m = merged as { weight: Tensor; bias: Tensor | null };
    noGrad(() => {
      const dW = this.loraB.matmul(this.loraA).mul(Tensor.full([], this.scaling)); // (out,in)
      m.weight = this.weight.add(dW);
      if (this.bias) m.bias = this.bias;
    });
    return merged;
  }

  override describe(): string {
    const [out, inF] = [this.weight.shape[0] ?? 0, this.weight.shape[1] ?? 0];
    return `LoRALinear(in_features=${inF}, out_features=${out}, r=${this.r}, `
      + `alpha=${this.alpha}, bias=${this.bias ? "True" : "False"})`;
  }
}

/** Options for {@link LoRAConv2d}. The convolution's own geometry plus the LoRA rank. */
export interface LoRAConv2dOptions extends LoRAOptions {
  stride?: number; padding?: number; dilation?: number; groups?: number;
}

/**
 * A 2-D convolution whose base kernel is frozen and adapted by a trainable low-rank
 * bottleneck of two convolutions:
 *
 *     y = conv(x, W) + b  +  up(down(x)) · (alpha / r)
 *
 * `down` is `(r, in, kH, kW)` — the full spatial kernel, `r` channels — and `up` is a
 * `1×1` `(out, r, 1, 1)`. Only `down` and `up` train; the base `W`, `b` are frozen
 * buffers. `up` starts at zero, so the layer begins as its base. The base may itself be
 * grouped (a depthwise stage can pass through untouched); the adapter is a plain
 * `groups=1` path added beside it, and `merge()` folds it into one kernel only when the
 * base is also `groups=1`.
 *
 * This is where a frozen CNN backbone (e.g. MobileNet's pointwise convolutions) is
 * adapted without touching the base — the conv analogue of {@link LoRALinear}.
 */
export class LoRAConv2d extends Module {
  readonly weight: Tensor;       // frozen base (out, in/groups, kH, kW)
  readonly bias: Tensor | null;  // frozen base
  readonly down: Tensor;         // (r, in, kH, kW), trained
  readonly up: Tensor;           // (out, r, 1, 1), trained, starts at zero
  readonly r: number;
  readonly alpha: number;
  readonly scaling: number;
  readonly kernelSize: number;
  readonly stride: number;
  readonly padding: number;
  readonly dilation: number;
  readonly groups: number;

  constructor(inChannels: number, outChannels: number, kernelSize: number, options: LoRAConv2dOptions = {}) {
    super();
    const { r = 8, alpha = r, bias = true, stride = 1, padding = 0, dilation = 1, groups = 1 } = options;
    if (r <= 0) throw new ValueError(`LoRAConv2d needs r >= 1, got ${r}`);
    if (inChannels % groups !== 0 || outChannels % groups !== 0) {
      throw new ValueError(`groups=${groups} divides neither in (${inChannels}) nor out (${outChannels})`);
    }
    const fanIn = (inChannels / groups) * kernelSize * kernelSize;
    const bound = 1 / Math.sqrt(Math.max(1, fanIn));
    this.weight = uniform([outChannels, inChannels / groups, kernelSize, kernelSize], bound);
    this.bias = bias ? uniform([outChannels], bound) : null;
    this.weight.requiresGrad = false;
    this.registerBuffer("weight", this.weight);
    if (this.bias) { this.bias.requiresGrad = false; this.registerBuffer("bias", this.bias); }
    // The adapter is a plain groups=1 low-rank bottleneck beside the (possibly grouped) base.
    const dBound = 1 / Math.sqrt(Math.max(1, inChannels * kernelSize * kernelSize));
    this.down = uniform([r, inChannels, kernelSize, kernelSize], dBound);
    this.up = Tensor.zeros([outChannels, r, 1, 1]);
    this.claim(this.down, this.up);
    this.r = r; this.alpha = alpha; this.scaling = alpha / r;
    this.kernelSize = kernelSize; this.stride = stride; this.padding = padding;
    this.dilation = dilation; this.groups = groups;
  }

  adapterState(): Record<string, Tensor> {
    return { lora_down: this.down, lora_up: this.up };
  }

  override ownParameters(): Record<string, Tensor> {
    return { lora_down: this.down, lora_up: this.up };
  }

  override forward(x: Tensor): Tensor {
    const base = x.conv2d(this.weight, this.bias, this.stride, this.padding, this.dilation, this.groups);
    const d = x.conv2d(this.down, null, this.stride, this.padding, this.dilation, 1);   // (.., r, H', W')
    const delta = d.conv2d(this.up, null, 1, 0, 1, 1).mul(Tensor.full([], this.scaling)); // 1×1 up
    return base.add(delta);
  }

  /** Fold the adapter into one kernel and return a plain `Conv2d`. Only when the base is
   *  `groups=1`: a grouped base and a `groups=1` adapter are different convolutions and
   *  cannot become one. W'[o,i,·,·] = W + (alpha/r)·Σ_r up[o,r]·down[r,i,·,·]. */
  merge(): Conv2d {
    if (this.groups !== 1) {
      throw new ValueError("cannot merge a grouped LoRAConv2d — the groups=1 adapter is a different convolution");
    }
    const [out, inC, kh, kw] = [this.weight.shape[0] ?? 0, this.weight.shape[1] ?? 0,
      this.weight.shape[2] ?? 0, this.weight.shape[3] ?? 0];
    const merged = new Conv2d(inC, out, this.kernelSize, this.stride, this.padding,
      this.dilation, 1, this.bias != null);
    const m = merged as { weight: Tensor; bias: Tensor | null };
    noGrad(() => {
      const up2d = this.up.reshape([out, this.r]);          // (out, r)
      const down2d = this.down.reshape([this.r, inC * kh * kw]); // (r, in*kh*kw)
      const dW = up2d.matmul(down2d).reshape([out, inC, kh, kw]).mul(Tensor.full([], this.scaling));
      m.weight = this.weight.add(dW);
      if (this.bias) m.bias = this.bias;
    });
    return merged;
  }

  override describe(): string {
    const [out, inC] = [this.weight.shape[0] ?? 0, (this.weight.shape[1] ?? 0) * this.groups];
    return `LoRAConv2d(${inC}, ${out}, kernel_size=${this.kernelSize}, r=${this.r}, `
      + `alpha=${this.alpha}, stride=${this.stride}, padding=${this.padding}, groups=${this.groups})`;
  }
}
