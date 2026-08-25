/**
 * `torchvision.transforms.v2.functional`.
 *
 * ## 165 names, and 114 of them are one operation counted five times
 *
 * `affine_image`, `affine_mask`, `affine_bounding_boxes`, `affine_keypoints` and
 * `affine_video` are v2's dispatch kernels: the type of what arrives decides
 * which runs, and that type system is the half of v2 this project declines. What
 * is left is 51 names, 34 of which v1 already has.
 *
 * So **most of this file is a re-export**, and what is written out is the nine
 * v2 adds that need no tv_tensors. They are thin on purpose. A second body for
 * `hflip` under the name `horizontalFlip` is two implementations of one thing,
 * and the one nobody is looking at is the one that drifts.
 *
 * ## The pair that answers opposite things
 *
 * `getSize` gives `[height, width]` and v1's `getImageSize` gives
 * `[width, height]`. Reversing it was one of v2's deliberate corrections, so the
 * two names sit one namespace apart giving opposite answers — and taking the
 * wrong one gives a transposed picture that is still plausible. Both are frozen
 * side by side in the golden for exactly that reason.
 */

import type { DType } from "./dtype.js";
import { GaussianNoise, RGB, ToDtype } from "./v2.js";
import {
  elasticTransform, getDimensions, getImageNumChannels,
  hflip, image, vflip,
  type Image, type Subject,
} from "./vision.js";
import { asImage, nextPermutation } from "./_vision_util.js";

/** v2's spelling of `hflip`. **The same function, not a copy of it.** */
export function horizontalFlip(img: Image): Image {
  return hflip(img);
}

/** v2's spelling of `vflip`. */
export function verticalFlip(img: Image): Image {
  return vflip(img);
}

/**
 * v2's spelling of `elasticTransform`.
 *
 * `fill` defaults to `null` and not `0`. Writing the zero looks harmless and is
 * not: `null` leaves the outside of the warp untouched and `0` paints it black,
 * so on a picture whose edges barely move the difference is a thin dark rim that
 * reads as the warp working.
 */
export function elastic(
  img: Image,
  displacement: ArrayLike<number>,
  interpolation: "bilinear" | "nearest" = "bilinear",
  fill: number | readonly number[] | null = null,
): Image {
  return elasticTransform(img, displacement, interpolation, fill);
}

/** `[height, width]` — **a list, and in that order.** v1's `getImageSize` is the
 *  other way round; this is the place to say which is which. */
export function getSize(img: Image): [number, number] {
  const [, height, width] = getDimensions(img);
  return [height, width];
}

/** v2's spelling of `getImageNumChannels`. */
export function getNumChannels(img: Image): number {
  return getImageNumChannels(img);
}

/** One channel to three. **Three channels pass through untouched** — torchvision
 *  returns the input rather than raising, so a pipeline can carry mixed pictures
 *  without a branch. */
export function grayscaleToRgb(img: Image): Image {
  return new RGB().apply(img);
}

/**
 * Reorder the channels. The list is **positions to take from**, so `[2, 0, 1]`
 * puts the old third channel first — the same direction as indexing, and the
 * opposite of "where each channel goes".
 */
export function permuteChannels(img: Image, permutation: readonly number[]): Image {
  const src = asImage(img, "permute_channels");
  const sorted = [...permutation].sort((a, b) => a - b);
  const ok = sorted.length === src.channels && sorted.every((v, i) => v === i);
  if (!ok) {
    throw new RangeError(
      `Invalid permutation ${JSON.stringify([...permutation])} for ` +
      `${src.channels} channels\n  (torch: Invalid permutation)`);
  }
  const out = new Float64Array(src.data.length);
  for (let i = 0; i < src.height * src.width; i++) {
    for (let c = 0; c < src.channels; c++) {
      out[i * src.channels + c] =
        src.data[i * src.channels + (permutation[c] as number)] as number;
    }
  }
  return image(out, src.height, src.width, src.channels, src.isByte);
}

/** A permutation drawn from the shared stream — `RandomChannelPermutation` as a
 *  function, kept here so the class and the function cannot disagree. */
export function randomPermuteChannels(img: Image): Image {
  return permuteChannels(img, nextPermutation(asImage(img, "permute_channels").channels));
}

/**
 * Cast, and **optionally scale on the way** — the half of `ToTensor` that v2
 * split out, as a function. `scale` is `false` by default here as it is there,
 * which is the trap: `toDtype(bytes, "float32")` gives 0..255 floats and looks
 * like it worked.
 */
export function toDtype(img: Subject, dtype: DType = "float32", scale = false): Subject {
  return new ToDtype(dtype, scale).apply(img);
}

/** Add normal noise. Float pictures only, for the reason `GaussianNoise` gives. */
export function gaussianNoise(img: Subject, mean = 0.0, sigma = 0.1, clip = true): Image {
  return new GaussianNoise(mean, sigma, clip).apply(img);
}

// v1's, re-exported rather than rewritten — v2 changed what its transforms print,
// not what its functions compute, and a second body under a second name is the one
// that drifts because nobody is looking at it.
export {
  adjustBrightness, adjustContrast, adjustGamma, adjustHue, adjustSaturation,
  adjustSharpness, affine, autocontrast, centerCrop, crop, elasticTransform,
  equalize, erase, fiveCrop, gaussianBlur, getDimensions, getImageNumChannels,
  getImageSize, hflip, invert, normalize, pad, perspective, posterize, resize,
  resizedCrop, rgbToGrayscale, rotate, solarize, tenCrop, toGrayscale, toTensor,
  vflip,
} from "./vision.js";
