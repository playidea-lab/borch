/**
 * `torch.nn.functional` 자리 — **`F.conv2d(x, w, b)` 를 옮겨 적을 수 있게 한다.**
 *
 * torch 는 같은 연산을 두 이름으로 갖는다. `x.relu()` 도 되고 `F.relu(x)` 도 되며,
 * 교재 코드는 층을 쓸 때는 앞쪽을 쓰고 손실·합성곱을 직접 부를 때는 뒤쪽을 쓴다.
 * borch 에는 앞쪽만 있어서 `F.` 로 적힌 줄을 통째로 다시 써야 했다.
 *
 * ## 여기 있는 것은 위임뿐이다
 *
 * 값은 전부 `Tensor` 의 메서드가 낸다. **골든이 이미 그 값들을 지키고 있으므로**
 * 이 파일이 지는 책임은 이름과 인자 차례 하나다 — 새 커널도, 새 미분식도 없다.
 * `at()` 이 대괄호에 해준 것과 같은 자리다.
 *
 * ## 메서드를 안 없앤다
 *
 * torch 가 둘 다 갖고 있으므로 우리도 둘 다 갖는다. 옮기는 것이 목적이지 깨는 것이
 * 아니고, `x.relu()` 로 적힌 코드가 이 변경으로 멈출 이유가 없다.
 *
 * **`Tensor` 가 작아지지는 않는다.** 메서드 401 개 중 여기로 이름이 나는 것은
 * 예순쯤이고 나머지는 torch 도 메서드로 두는 것들이다. 이 파일은 god object 를
 * 푸는 것이 아니라 **없던 문을 내는 것**이다 — 그 둘을 섞어 말하면 결과가 기대에
 * 못 미친다.
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
 * `F.unfold` — im2col 이다. **`Tensor.unfold` 와 다른 연산이다.**
 *
 * torch 에서 `x.unfold(dim, size, step)` 은 축 하나에 창을 미는 것이고
 * `F.unfold(x, kernel)` 은 합성곱용으로 펴는 것이다. 이름이 같아서 자동으로 이으면
 * **다른 연산이 조용히 걸린다** — 실제로 처음 생성했을 때 그렇게 걸렸다.
 */
export function unfold(
  input: Tensor, kernelSize: number | [number, number], dilation = 1,
  padding = 0, stride = 1,
): Tensor {
  return input.unfoldIm2col(kernelSize, dilation, padding, stride);
}

/**
 * `F.huber_loss` — **인자 차례가 다르다.** torch 는 `(input, target, reduction,
 * delta)` 이고 우리 메서드는 `(target, delta, reduction)` 이다. 자동 위임이면 위치
 * 인자를 쓴 코드가 델타와 축약을 바꿔 넣는다.
 */
export function huberLoss(
  input: Tensor, target: Tensor, reduction: Reduction = "mean", delta = 1.0,
): Tensor {
  return input.huberLoss(target, delta, reduction);
}

// ── 여기 없는 것 — **이름이 같은데 연산이 다르다** ──────────────────────
//
// 이름으로 이으면 다섯이 조용히 다른 것에 걸린다. 그래서 안 낸다. 없는 것은
// 없다고 두는 편이, 이름은 torch 인데 속이 다른 것보다 낫다.
//
//   F.batch_norm      `Tensor.batchNorm(dim, eps)` 은 축만 바꾼 `layerNorm` 이다.
//                     진짜는 `nn.functional.batchNorm` 으로 나간다(`nn.ts` 의 자유 함수)
//   F.layer_norm      torch 는 `normalized_shape` 를, 우리는 접을 축 하나를 받는다
//   F.rms_norm        같은 이유
//   F.pad             torch 는 축 전부의 덧댐을 목록으로, 우리는 축 하나씩
//   F.upsample        torch 에서도 폐기 예정이다 — `interpolate` 쪽이 정본이고
//                     우리 것은 배율 하나만 받는다
//
// **이 목록은 자동으로 안 는다.** 새 메서드를 여기 이을 때는 torch 의 시그니처와
// 인자 이름·차례를 맞춰 봐야 한다 — 처음 생성했을 때 열여덟이 어긋났고 그중
// 일곱이 진짜 다른 연산이었다. `F.unfold` 는 `Tensor.unfold` 가 아니라 im2col 이고,
// `F.huber_loss` 는 델타와 축약의 자리가 우리와 뒤바뀌어 있다.

// ── 표에서 생성되는 단항들 ──────────────────────────────────────────────
//
// `relu`·`sigmoid` 같은 것은 `tensor.ts` 가 `UNARY` 표를 돌며 프로토타입에 단다.
// **선언만 있고 본문이 없어서** 시그니처를 훑는 방식으로는 안 잡힌다 — 처음 생성했을
// 때 `F.relu` 가 통째로 빠진 채로 지나갈 뻔했고, parity 러너가 그것을 잡았다.
//
// 여기 있는 열하나가 torch 의 `F.` 에도 있는 것들이다. 나머지 서른여덟(`exp`·`log`
// 따위)은 torch 도 `F.` 에 안 두므로 안 낸다.
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
