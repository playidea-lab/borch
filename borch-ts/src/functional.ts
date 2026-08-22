/**
 * The place `torch.nn.functional` occupies — **so that `F.conv2d(x, w, b)` can be copied
 * across.**
 *
 * torch gives the same operation two names. `x.relu()` works and so does `F.relu(x)`, and
 * textbook code uses the first for layers and the second when calling losses and
 * convolutions directly. borch had the first alone, so any line written with `F.` had to
 * be rewritten entirely.
 *
 * ## What is here is delegation and nothing else
 *
 * Every value comes from a `Tensor` method. **The golden already holds those values**, so
 * this file's responsibility is one name and one argument order — no new kernel and no new
 * derivative. The same place `at()` occupies for brackets.
 *
 * ## The methods are not removed
 *
 * torch has both, so we have both. The point is to port rather than to break, and code
 * written as `x.relu()` has no reason to stop because of this change.
 *
 * **`Tensor` does not get smaller.** Of its 401 methods, about sixty get a name here and
 * the rest are ones torch keeps as methods too. This file is not untangling a god object;
 * it is **opening a door that was not there** — saying the two together sets an
 * expectation the result does not meet.
 */

import { type Reduction, Tensor } from "./tensor.js";

export function alphaDropout(input: Tensor, p = 0.5, training = false, perChannel = false): Tensor {
  return input.alphaDropout(p, training, perChannel);
}

export function avgPool2d(input: Tensor, kernel = 2, stride?: number): Tensor {
  return input.avgPool2d(kernel, stride);
}


export function bilinear(input: Tensor, other: Tensor, weight: Tensor, bias: Tensor | null = null): Tensor {
  return input.bilinear(other, weight, bias);
}

export function celu(input: Tensor, alpha = 1.0): Tensor {
  return input.celu(alpha);
}

export function channelShuffle(input: Tensor, groups: number): Tensor {
  return input.channelShuffle(groups);
}

export function conv1d(input: Tensor, weight: Tensor, bias: Tensor | null = null, stride = 1, padding = 0): Tensor {
  return input.conv1d(weight, bias, stride, padding);
}

export function conv2d(input: Tensor, weight: Tensor, bias: Tensor | null = null, stride = 1, padding = 0): Tensor {
  return input.conv2d(weight, bias, stride, padding);
}

export function conv3d(input: Tensor, weight: Tensor, bias: Tensor | null = null, stride = 1, padding = 0): Tensor {
  return input.conv3d(weight, bias, stride, padding);
}

export function cosineEmbeddingLoss(input: Tensor, other: Tensor, target: Tensor, margin = 0.0, reduction: Reduction = "mean"): Tensor {
  return input.cosineEmbeddingLoss(other, target, margin, reduction);
}

export function cosineSimilarity(input: Tensor, other: Tensor, dim = 1, eps = 1e-8): Tensor {
  return input.cosineSimilarity(other, dim, eps);
}

export function crossEntropy(input: Tensor, target: Tensor): Tensor {
  return input.crossEntropy(target);
}

export function dropout(input: Tensor, p = 0.5, training = true): Tensor {
  return input.dropout(p, training);
}

export function fold(input: Tensor, outputSize: number | [number, number], kernel: number | [number, number], dilation = 1, padding = 0, stride = 1): Tensor {
  return input.fold(outputSize, kernel, dilation, padding, stride);
}

export function gaussianNllLoss(input: Tensor, target: Tensor, variance: Tensor, full = false, eps = 1e-6, reduction: Reduction = "mean"): Tensor {
  return input.gaussianNllLoss(target, variance, full, eps, reduction);
}

export function glu(input: Tensor, dim = -1): Tensor {
  return input.glu(dim);
}

export function groupNorm(input: Tensor, numGroups: number, eps = 1e-5): Tensor {
  return input.groupNorm(numGroups, eps);
}

export function hardshrink(input: Tensor, lambd = 0.5): Tensor {
  return input.hardshrink(lambd);
}

export function hardtanh(input: Tensor, minVal = -1.0, maxVal = 1.0): Tensor {
  return input.hardtanh(minVal, maxVal);
}

export function hingeEmbeddingLoss(input: Tensor, target: Tensor, margin = 1.0, reduction: Reduction = "mean"): Tensor {
  return input.hingeEmbeddingLoss(target, margin, reduction);
}


export function instanceNorm(input: Tensor, eps = 1e-5): Tensor {
  return input.instanceNorm(eps);
}

export function klDiv(input: Tensor, target: Tensor, reduction: Reduction | "batchmean" = "mean", logTarget = false): Tensor {
  return input.klDiv(target, reduction, logTarget);
}

export function l1Loss(input: Tensor, target: Tensor): Tensor {
  return input.l1Loss(target);
}


export function leakyRelu(input: Tensor, slope = 0.01): Tensor {
  return input.leakyRelu(slope);
}

export function linear(input: Tensor, weight: Tensor): Tensor {
  return input.linear(weight);
}

export function localResponseNorm(input: Tensor, size: number, alpha = 1e-4, beta = 0.75, k = 1.0): Tensor {
  return input.localResponseNorm(size, alpha, beta, k);
}

export function logSoftmax(input: Tensor, dim = 0): Tensor {
  return input.logSoftmax(dim);
}

export function marginRankingLoss(input: Tensor, other: Tensor, target: Tensor, margin = 0.0, reduction: Reduction = "mean"): Tensor {
  return input.marginRankingLoss(other, target, margin, reduction);
}

export function maxPool1d(input: Tensor, kernel = 2, stride?: number): Tensor {
  return input.maxPool1d(kernel, stride);
}

export function maxPool2d(input: Tensor, kernel = 2, stride?: number): Tensor {
  return input.maxPool2d(kernel, stride);
}

export function maxPool3d(input: Tensor, kernel = 2, stride?: number): Tensor {
  return input.maxPool3d(kernel, stride);
}

export function mseLoss(input: Tensor, target: Tensor): Tensor {
  return input.mseLoss(target);
}

export function multiMarginLoss(input: Tensor, target: Tensor, p = 1, margin = 1.0, weight: Tensor | null = null, reduction: Reduction = "mean"): Tensor {
  return input.multiMarginLoss(target, p, margin, weight, reduction);
}

export function multilabelMarginLoss(input: Tensor, target: Tensor, reduction: Reduction = "mean"): Tensor {
  return input.multilabelMarginLoss(target, reduction);
}

export function multilabelSoftMarginLoss(input: Tensor, target: Tensor, reduction: Reduction = "mean"): Tensor {
  return input.multilabelSoftMarginLoss(target, reduction);
}

export function nllLoss(input: Tensor, target: Tensor): Tensor {
  return input.nllLoss(target);
}

export function normalize(input: Tensor, dim = 1, eps = 1e-12): Tensor {
  return input.normalize(dim, eps);
}

export function oneHot(input: Tensor, classes: number): Tensor {
  return input.oneHot(classes);
}


export function pairwiseDistance(input: Tensor, other: Tensor, p = 2.0, eps = 1e-6, keepdim = false): Tensor {
  return input.pairwiseDistance(other, p, eps, keepdim);
}

export function pdist(input: Tensor, p = 2.0): Tensor {
  return input.pdist(p);
}

export function pixelShuffle(input: Tensor, upscaleFactor: number): Tensor {
  return input.pixelShuffle(upscaleFactor);
}

export function pixelUnshuffle(input: Tensor, downscaleFactor: number): Tensor {
  return input.pixelUnshuffle(downscaleFactor);
}

export function poissonNllLoss(input: Tensor, target: Tensor, logInput = true, full = false, eps = 1e-8, reduction: Reduction = "mean"): Tensor {
  return input.poissonNllLoss(target, logInput, full, eps, reduction);
}

export function prelu(input: Tensor, weight: Tensor): Tensor {
  return input.prelu(weight);
}


export function rrelu(input: Tensor, lower = 1 / 8, upper = 1 / 3, training = false): Tensor {
  return input.rrelu(lower, upper, training);
}

export function smoothL1Loss(input: Tensor, target: Tensor, beta = 1.0): Tensor {
  return input.smoothL1Loss(target, beta);
}

export function softMarginLoss(input: Tensor, target: Tensor, reduction: Reduction = "mean"): Tensor {
  return input.softMarginLoss(target, reduction);
}

export function softmax(input: Tensor, dim = 0): Tensor {
  return input.softmax(dim);
}

export function softmin(input: Tensor, dim = -1): Tensor {
  return input.softmin(dim);
}

export function softplus(input: Tensor, beta = 1.0, threshold = 20.0): Tensor {
  return input.softplus(beta, threshold);
}

export function softshrink(input: Tensor, lambd = 0.5): Tensor {
  return input.softshrink(lambd);
}

export function threshold(input: Tensor, t: number, value: number): Tensor {
  return input.threshold(t, value);
}

export function tripletMarginLoss(input: Tensor, positive: Tensor, negative: Tensor, margin = 1.0, p = 2.0, eps = 1e-6, swap = false, reduction: Reduction = "mean"): Tensor {
  return input.tripletMarginLoss(positive, negative, margin, p, eps, swap, reduction);
}



/**
 * `F.unfold` — im2col. **A different operation from `Tensor.unfold`.**
 *
 * In torch, `x.unfold(dim, size, step)` slides a window along one axis while
 * `F.unfold(x, kernel)` lays it out for a convolution. The names match, so wiring it up
 * automatically **quietly attaches the wrong operation** — which is what happened when
 * this was first generated.
 */
export function unfold(
  input: Tensor, kernelSize: number | [number, number], dilation = 1,
  padding = 0, stride = 1,
): Tensor {
  return input.unfoldIm2col(kernelSize, dilation, padding, stride);
}

/**
 * `F.huber_loss` — **the argument order differs.** torch is `(input, target, reduction,
 * delta)` and our method is `(target, delta, reduction)`. Delegated automatically, code
 * using positional arguments swaps delta and reduction.
 */
export function huberLoss(
  input: Tensor, target: Tensor, reduction: Reduction = "mean", delta = 1.0,
): Tensor {
  return input.huberLoss(target, delta, reduction);
}

// ── What is not here — **the name matches and the operation does not** ────
//
// Wired up by name, five of them quietly attach to something else. So they are not
// exported. Leaving something absent is better than a torch name with different insides.
//
//   F.batch_norm      `Tensor.batchNorm(dim, eps)` is `layerNorm` with the axis changed.
//                     The real one goes out as `nn.functional.batchNorm` (a free function
//                     in `nn.ts`)
//   F.layer_norm      torch takes `normalized_shape`; we take one axis to fold
//   F.rms_norm        the same reason
//   F.pad             torch takes every axis's padding as a list; we take one axis at a time
//   F.upsample        deprecated in torch too — `interpolate` is the authoritative one,
//                     and ours takes a single scale
//
// **This list does not grow by itself.** Wiring a new method here means checking it
// against torch's signature, argument names and order — eighteen were off when this was
// first generated and seven of those were genuinely different operations. `F.unfold` is
// im2col rather than `Tensor.unfold`, and `F.huber_loss` has delta and reduction in
// swapped positions.

// ── The unaries generated from the table ──────────────────────────────────
//
// Things like `relu` and `sigmoid` are attached to the prototype by `tensor.ts` walking
// the `UNARY` table. **They have a declaration and no body**, so a signature sweep does
// not catch them — when this was first generated `F.relu` nearly went missing entirely,
// and the parity runner caught it.
//
// The eleven here are the ones torch also has under `F.`. The other thirty-eight (`exp`,
// `log` and so on) are not under torch's `F.` either, so they are not exported.
export function relu(input: Tensor): Tensor { return input.relu(); }
export function relu6(input: Tensor): Tensor { return input.relu6(); }
export function sigmoid(input: Tensor): Tensor { return input.sigmoid(); }
export function hardsigmoid(input: Tensor): Tensor { return input.hardsigmoid(); }
export function hardswish(input: Tensor): Tensor { return input.hardswish(); }
export function logsigmoid(input: Tensor): Tensor { return input.logsigmoid(); }
export function mish(input: Tensor): Tensor { return input.mish(); }
export function selu(input: Tensor): Tensor { return input.selu(); }
export function softsign(input: Tensor): Tensor { return input.softsign(); }
export function tanh(input: Tensor): Tensor { return input.tanh(); }
export function tanhshrink(input: Tensor): Tensor { return input.tanhshrink(); }
