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
// **This is a cycle, and it is the deliberate kind.** `nn.ts` imports this module at
// its foot to bind the delegated names; these two travel the other way so that the
// functional losses refuse `weight` in the same words their layers do, rather than
// carrying a second copy of the message.
//
// Both are **function declarations**, so they are hoisted and are only ever called at
// call time — never while either module is still initialising, which is the case a
// cycle actually breaks. The alternative was a third module for six lines, and a module
// that is not in `site/build_api.py`'s list is a module nobody counts.
import { legacyReduction, refuseWeight } from "./nn.js";

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

export function cosineEmbeddingLoss(
  input: Tensor, other: Tensor, target: Tensor, margin = 0.0,
  sizeAverage: boolean | null = null, reduce: boolean | null = null,
  reduction: Reduction = "mean",
): Tensor {
  reduction = legacyReduction(sizeAverage, reduce, reduction);
  return input.cosineEmbeddingLoss(other, target, margin, reduction);
}

export function cosineSimilarity(input: Tensor, other: Tensor, dim = 1, eps = 1e-8): Tensor {
  return input.cosineSimilarity(other, dim, eps);
}

export function crossEntropy(
  input: Tensor, target: Tensor, weight?: Tensor,
  sizeAverage: boolean | null = null, ignoreIndex = -100,
  reduce: boolean | null = null, reduction: Reduction = "mean",
  labelSmoothing = 0.0,
): Tensor {
  // **`ignoreIndex` sits between the pair**, which is torch's order and not a tidy one.
  // `nn.ts` says the same about the classes: a rule that puts `sizeAverage` and
  // `reduce` side by side is right about twelve of them and wrong about this one,
  // `NLLLoss` and `PoissonNLLLoss`.
  refuseWeight("cross_entropy", "weight", weight);
  reduction = legacyReduction(sizeAverage, reduce, reduction);
  return input.crossEntropy(target, ignoreIndex, reduction, labelSmoothing);
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

export function hingeEmbeddingLoss(
  input: Tensor, target: Tensor, margin = 1.0,
  sizeAverage: boolean | null = null, reduce: boolean | null = null,
  reduction: Reduction = "mean",
): Tensor {
  return input.hingeEmbeddingLoss(
    target, margin, legacyReduction(sizeAverage, reduce, reduction));
}


export function instanceNorm(input: Tensor, eps = 1e-5): Tensor {
  return input.instanceNorm(eps);
}

export function klDiv(
  input: Tensor, target: Tensor,
  sizeAverage: boolean | null = null, reduce: boolean | null = null,
  reduction: Reduction | "batchmean" = "mean", logTarget = false,
): Tensor {
  return input.klDiv(
    target, legacyReduction(sizeAverage, reduce, reduction), logTarget);
}

export function l1Loss(
  input: Tensor, target: Tensor, sizeAverage: boolean | null = null,
  reduce: boolean | null = null, reduction: Reduction = "mean", weight?: Tensor,
): Tensor {
  refuseWeight("l1_loss", "weight", weight);
  reduction = legacyReduction(sizeAverage, reduce, reduction);
  return input.l1Loss(target, reduction);
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

export function marginRankingLoss(
  input: Tensor, other: Tensor, target: Tensor, margin = 0.0,
  sizeAverage: boolean | null = null, reduce: boolean | null = null,
  reduction: Reduction = "mean",
): Tensor {
  reduction = legacyReduction(sizeAverage, reduce, reduction);
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

export function mseLoss(
  input: Tensor, target: Tensor, sizeAverage: boolean | null = null,
  reduce: boolean | null = null, reduction: Reduction = "mean", weight?: Tensor,
): Tensor {
  refuseWeight("mse_loss", "weight", weight);
  reduction = legacyReduction(sizeAverage, reduce, reduction);
  return input.mseLoss(target, reduction);
}

export function multiMarginLoss(
  input: Tensor, target: Tensor, p = 1, margin = 1.0,
  weight: Tensor | null = null, sizeAverage: boolean | null = null,
  reduce: boolean | null = null, reduction: Reduction = "mean",
): Tensor {
  reduction = legacyReduction(sizeAverage, reduce, reduction);
  return input.multiMarginLoss(target, p, margin, weight, reduction);
}

export function multilabelMarginLoss(
  input: Tensor, target: Tensor,
  sizeAverage: boolean | null = null, reduce: boolean | null = null,
  reduction: Reduction = "mean",
): Tensor {
  return input.multilabelMarginLoss(
    target, legacyReduction(sizeAverage, reduce, reduction));
}

// `weight` sits between `target` and `reduction`, which is where torch's
// `F.multilabel_soft_margin_loss(input, target, weight, size_average, reduce,
// reduction)` puts it.
export function multilabelSoftMarginLoss(
  input: Tensor, target: Tensor, weight?: Tensor,
  sizeAverage: boolean | null = null, reduce: boolean | null = null,
  reduction: Reduction = "mean",
): Tensor {
  return input.multilabelSoftMarginLoss(
    target, weight, legacyReduction(sizeAverage, reduce, reduction));
}

export function nllLoss(
  input: Tensor, target: Tensor, weight?: Tensor,
  sizeAverage: boolean | null = null, ignoreIndex = -100,
  reduce: boolean | null = null, reduction: Reduction = "mean",
): Tensor {
  // `ignoreIndex` between the pair, as `crossEntropy` above and for the same reason.
  refuseWeight("nll_loss", "weight", weight);
  reduction = legacyReduction(sizeAverage, reduce, reduction);
  return input.nllLoss(target, ignoreIndex, reduction);
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

export function poissonNllLoss(
  input: Tensor, target: Tensor, logInput = true, full = false,
  sizeAverage: boolean | null = null, eps = 1e-8, reduce: boolean | null = null,
  reduction: Reduction = "mean",
): Tensor {
  // **`eps` sat in `sizeAverage`'s seat.** torch puts the pair either side of `eps`
  // here, so appending the two would have been wrong twice over — a number was landing
  // where a boolean goes, and `reduction` where `eps` does.
  reduction = legacyReduction(sizeAverage, reduce, reduction);
  return input.poissonNllLoss(target, logInput, full, eps, reduction);
}

export function prelu(input: Tensor, weight: Tensor): Tensor {
  return input.prelu(weight);
}


export function rrelu(input: Tensor, lower = 1 / 8, upper = 1 / 3, training = false): Tensor {
  return input.rrelu(lower, upper, training);
}

export function smoothL1Loss(
  input: Tensor, target: Tensor, sizeAverage: boolean | null = null,
  reduce: boolean | null = null, reduction: Reduction = "mean", beta = 1.0,
): Tensor {
  // **`beta` sat in `sizeAverage`'s seat**, and there was no `reduction` at all — so
  // `smoothL1Loss(x, t, "sum")` set the beta to a string, which compares false against
  // every difference and quietly gives the `beta = 1` answer. That is the shape a peer
  // found in the binding's `huber_loss` on the same afternoon.
  reduction = legacyReduction(sizeAverage, reduce, reduction);
  return input.smoothL1Loss(target, beta, reduction);
}

/**
 * torch's list, with **the deprecated `size_average`/`reduce` in the seats it keeps
 * them in.** Left out, everything after moves forward and torch's own line stops
 * compiling — see `binaryCrossEntropyWithLogits` above, where the same two seats
 * were the whole defect. `legacyReduction` folds the pair and warns, which is what
 * torch does with it.
 */
export function softMarginLoss(
  input: Tensor, target: Tensor,
  sizeAverage: boolean | null = null, reduce: boolean | null = null,
  reduction: Reduction = "mean",
): Tensor {
  return input.softMarginLoss(
    target, legacyReduction(sizeAverage, reduce, reduction));
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

export function tripletMarginLoss(
  input: Tensor, positive: Tensor, negative: Tensor, margin = 1.0, p = 2.0,
  eps = 1e-6, swap = false, sizeAverage: boolean | null = null,
  reduce: boolean | null = null, reduction: Reduction = "mean",
): Tensor {
  reduction = legacyReduction(sizeAverage, reduce, reduction);
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

// ── The names torch keeps per rank, over kernels that were already here ────
//
// **Eighteen names, no new arithmetic.** `poolND`, `lpPool`, `maxUnpool`,
// `convTransposeND`, `maxPoolWithIndices` and `fractionalMaxPool` were all doing this
// work already, under generic ND names that borch.ts chose and torch does not have. So
// the name axis counted eighteen absent *features* while every one of them was one
// delegation away — the same shape as `BatchNorm1d` (absent while all six lazy variants
// were present) and `default_collate` (present, under the name `stackItems`, with a
// comment saying what it would be called).
//
// The rank is not consulted by the kernels, so it is checked here: torch refuses a 4-D
// input to `conv_transpose1d`, and passing it through would answer a question torch
// calls undefined — the quiet direction, and one the rank axis measures.

function atRank(input: Tensor, want: number, who: string): Tensor {
  if (input.shape.length !== want) {
    throw new Error(`${who} expects a ${want}-D input, got [${input.shape}]`);
  }
  return input;
}

export function avgPool1d(input: Tensor, kernel = 2, stride?: number): Tensor {
  return input.avgPool1d(kernel, stride);
}

export function avgPool3d(input: Tensor, kernel = 2, stride?: number): Tensor {
  return input.avgPool3d(kernel, stride);
}

export function convTranspose1d(
  input: Tensor, weight: Tensor, bias: Tensor | null = null,
  stride: number | readonly number[] = 1, padding: number | readonly number[] = 0,
  outputPadding: number | readonly number[] = 0, groups = 1,
  dilation: number | readonly number[] = 1,
): Tensor {
  return atRank(input, 3, "conv_transpose1d")
    .convTransposeND(weight, bias, stride, padding, outputPadding, groups, dilation);
}

export function convTranspose2d(
  input: Tensor, weight: Tensor, bias: Tensor | null = null,
  stride: number | readonly number[] = 1, padding: number | readonly number[] = 0,
  outputPadding: number | readonly number[] = 0, groups = 1,
  dilation: number | readonly number[] = 1,
): Tensor {
  return atRank(input, 4, "conv_transpose2d")
    .convTransposeND(weight, bias, stride, padding, outputPadding, groups, dilation);
}

export function convTranspose3d(
  input: Tensor, weight: Tensor, bias: Tensor | null = null,
  stride: number | readonly number[] = 1, padding: number | readonly number[] = 0,
  outputPadding: number | readonly number[] = 0, groups = 1,
  dilation: number | readonly number[] = 1,
): Tensor {
  return atRank(input, 5, "conv_transpose3d")
    .convTransposeND(weight, bias, stride, padding, outputPadding, groups, dilation);
}

export function lpPool1d(
  input: Tensor, normType: number, kernel: number, stride?: number,
): Tensor {
  return atRank(input, 3, "lp_pool1d").lpPool(normType, kernel, stride);
}

export function lpPool2d(
  input: Tensor, normType: number, kernel: number, stride?: number,
): Tensor {
  return atRank(input, 4, "lp_pool2d").lpPool(normType, kernel, stride);
}

export function lpPool3d(
  input: Tensor, normType: number, kernel: number, stride?: number,
): Tensor {
  return atRank(input, 5, "lp_pool3d").lpPool(normType, kernel, stride);
}

export function maxPool1dWithIndices(
  input: Tensor, kernel = 2, stride?: number,
): { values: Tensor; indices: Tensor } {
  return atRank(input, 3, "max_pool1d_with_indices").maxPoolWithIndices(kernel, stride);
}

export function maxPool2dWithIndices(
  input: Tensor, kernel = 2, stride?: number,
): { values: Tensor; indices: Tensor } {
  return atRank(input, 4, "max_pool2d_with_indices").maxPoolWithIndices(kernel, stride);
}

export function maxPool3dWithIndices(
  input: Tensor, kernel = 2, stride?: number,
): { values: Tensor; indices: Tensor } {
  return atRank(input, 5, "max_pool3d_with_indices").maxPoolWithIndices(kernel, stride);
}

export function maxUnpool1d(
  input: Tensor, indices: Tensor, kernel: number, stride?: number, padding = 0,
  outputSize?: readonly number[],
): Tensor {
  return atRank(input, 3, "max_unpool1d")
    .maxUnpool(indices, kernel, stride, padding, outputSize);
}

export function maxUnpool2d(
  input: Tensor, indices: Tensor, kernel: number, stride?: number, padding = 0,
  outputSize?: readonly number[],
): Tensor {
  return atRank(input, 4, "max_unpool2d")
    .maxUnpool(indices, kernel, stride, padding, outputSize);
}

export function maxUnpool3d(
  input: Tensor, indices: Tensor, kernel: number, stride?: number, padding = 0,
  outputSize?: readonly number[],
): Tensor {
  return atRank(input, 5, "max_unpool3d")
    .maxUnpool(indices, kernel, stride, padding, outputSize);
}

export function fractionalMaxPool2d(
  input: Tensor, kernel: number, outputSize: readonly number[],
  randomSamples: readonly (readonly number[])[],
): { values: Tensor; indices: Tensor } {
  return atRank(input, 4, "fractional_max_pool2d")
    .fractionalMaxPool(kernel, outputSize, randomSamples);
}

export function fractionalMaxPool2dWithIndices(
  input: Tensor, kernel: number, outputSize: readonly number[],
  randomSamples: readonly (readonly number[])[],
): { values: Tensor; indices: Tensor } {
  return fractionalMaxPool2d(input, kernel, outputSize, randomSamples);
}

export function fractionalMaxPool3d(
  input: Tensor, kernel: number, outputSize: readonly number[],
  randomSamples: readonly (readonly number[])[],
): { values: Tensor; indices: Tensor } {
  return atRank(input, 5, "fractional_max_pool3d")
    .fractionalMaxPool(kernel, outputSize, randomSamples);
}

export function fractionalMaxPool3dWithIndices(
  input: Tensor, kernel: number, outputSize: readonly number[],
  randomSamples: readonly (readonly number[])[],
): { values: Tensor; indices: Tensor } {
  return fractionalMaxPool3d(input, kernel, outputSize, randomSamples);
}

/**
 * `F.binary_cross_entropy_with_logits`. The method has been here as `bceWithLogits`
 * — this is torch's name for it.
 *
 * **torch's whole list, and it took three of seven.** `reduction` sat in `weight`'s
 * seat, so a caller writing torch's own line — or the binding unrolling it
 * positionally — put a tensor where a string belongs. The layer next door,
 * `BCEWithLogitsLoss`, has taken all seven and refused two of them for as long as it
 * has existed; only the functional form was short.
 *
 * `weight` and `posWeight` are **refused rather than ignored**, which is what the
 * layer does and what this repository's rule says: an argument accepted and unused is
 * worse than one that is absent, because the caller cannot tell.
 */
export function binaryCrossEntropyWithLogits(
  input: Tensor,
  target: Tensor,
  weight?: Tensor,
  sizeAverage: boolean | null = null,
  reduce: boolean | null = null,
  reduction: Reduction = "mean",
  posWeight?: Tensor,
): Tensor {
  refuseWeight("binary_cross_entropy_with_logits", "weight", weight);
  refuseWeight("binary_cross_entropy_with_logits", "pos_weight", posWeight);
  return input.bceWithLogits(target, legacyReduction(sizeAverage, reduce, reduction));
}

/**
 * `F.binary_cross_entropy` — over probabilities. See `Tensor.bce` on why the
 * clamp is on the log's output rather than an epsilon on the probability.
 *
 * **No `posWeight`.** That argument belongs to the logits form alone, and offering it
 * here would be an argument torch does not have — the core says the same at the same
 * place.
 */
export function binaryCrossEntropy(
  input: Tensor,
  target: Tensor,
  weight?: Tensor,
  sizeAverage: boolean | null = null,
  reduce: boolean | null = null,
  reduction: Reduction = "mean",
): Tensor {
  refuseWeight("binary_cross_entropy", "weight", weight);
  return input.bce(target, legacyReduction(sizeAverage, reduce, reduction));
}

// ── The adaptive family ───────────────────────────────────────────────────
//
// `adaptivePool` and `adaptiveMaxPoolWithIndices` do the work and do not consult
// the rank — a peer measured that `AdaptiveAvgPool1d` and `AdaptiveAvgPool3d` have
// byte-identical bodies over the same helper. So these nine are torch's per-rank
// names over one computation, and the rank check is here because torch has one.
//
// **`nn.AdaptiveAvgPool2d` is deliberately still absent.** A peer's lesson page
// teaches a reader what to do without it and pins the absence at both ends; these
// are `nn.functional` names in camelCase, a different string from the class, so the
// pin is untouched (checked, not assumed: `_folds_onto` returns false for a
// capitalised name, so the class cannot fold onto any of these).

export function adaptiveAvgPool1d(input: Tensor, outputSize: number | readonly number[]): Tensor {
  return atRank(input, 3, "adaptive_avg_pool1d").adaptivePool("avg", outputSize);
}

export function adaptiveAvgPool2d(input: Tensor, outputSize: number | readonly number[]): Tensor {
  return atRank(input, 4, "adaptive_avg_pool2d").adaptivePool("avg", outputSize);
}

export function adaptiveAvgPool3d(input: Tensor, outputSize: number | readonly number[]): Tensor {
  return atRank(input, 5, "adaptive_avg_pool3d").adaptivePool("avg", outputSize);
}

export function adaptiveMaxPool1d(input: Tensor, outputSize: number | readonly number[]): Tensor {
  return atRank(input, 3, "adaptive_max_pool1d").adaptivePool("max", outputSize);
}

export function adaptiveMaxPool2d(input: Tensor, outputSize: number | readonly number[]): Tensor {
  return atRank(input, 4, "adaptive_max_pool2d").adaptivePool("max", outputSize);
}

export function adaptiveMaxPool3d(input: Tensor, outputSize: number | readonly number[]): Tensor {
  return atRank(input, 5, "adaptive_max_pool3d").adaptivePool("max", outputSize);
}

export function adaptiveMaxPool1dWithIndices(
  input: Tensor, outputSize: number | readonly number[],
): { values: Tensor; indices: Tensor } {
  return atRank(input, 3, "adaptive_max_pool1d_with_indices")
    .adaptiveMaxPoolWithIndices(outputSize);
}

export function adaptiveMaxPool2dWithIndices(
  input: Tensor, outputSize: number | readonly number[],
): { values: Tensor; indices: Tensor } {
  return atRank(input, 4, "adaptive_max_pool2d_with_indices")
    .adaptiveMaxPoolWithIndices(outputSize);
}

export function adaptiveMaxPool3dWithIndices(
  input: Tensor, outputSize: number | readonly number[],
): { values: Tensor; indices: Tensor } {
  return atRank(input, 5, "adaptive_max_pool3d_with_indices")
    .adaptiveMaxPoolWithIndices(outputSize);
}

/**
 * `F.triplet_margin_with_distance_loss` — see `Tensor.tripletMarginWithDistanceLoss`
 * on why the body is a method.
 */
export function tripletMarginWithDistanceLoss(
  anchor: Tensor,
  positive: Tensor,
  negative: Tensor,
  distanceFunction: ((u: Tensor, v: Tensor) => Tensor) | null = null,
  margin = 1.0,
  swap = false,
  reduction: Reduction = "mean",
): Tensor {
  return anchor.tripletMarginWithDistanceLoss(
    positive, negative, distanceFunction, margin, swap, reduction);
}
