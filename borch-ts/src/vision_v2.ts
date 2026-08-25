/**
 * `torchvision.transforms.v2.functional` — **nine new names, and the rest re-exported.**
 *
 * v2 is torchvision's current recommended surface and it is v1 with a different front.
 * Measured before any of this was written: on a plain image v2's transforms give what
 * v1's give, to the last bit. So the arithmetic is not written twice — everything here
 * that v1 already had is a **re-export and not a copy**, which is the one property this
 * file has to keep and the one a value comparison cannot see on its own. The golden
 * asks four of v1's through the v2 spelling for exactly that reason: if one ever grows
 * a body of its own, those four stop matching v1's frozen answers.
 *
 * ## The pair that answers backwards from its neighbour
 *
 * `getSize` answers `[height, width]`. `getImageSize`, one namespace over, answers
 * `[width, height]`. v2 reversed it deliberately and torchvision deprecated the old
 * one rather than changing it, so the two live side by side giving opposite answers.
 * Both are here, next to each other, because a reader who reaches for the wrong one
 * gets a picture that is transposed and entirely plausible.
 */

import {
  adjustHue,
  elasticTransform,
  getImageNumChannels,
  getImageSize,
  hflip,
  type Image,
  image,
  normalize,
  resize,
  rotate,
  vflip,
} from "./vision.js";

// ── v1's names, reached through v2's spelling ────────────────────────────────
//
// Bound rather than wrapped. A wrapper is a body, and a body is the thing these are
// here to prove does not exist.
export { adjustHue, getImageSize, normalize, resize, rotate };

/** v2's name for `hflip`. */
export const horizontalFlip = hflip;
/** v2's name for `vflip`. */
export const verticalFlip = vflip;
/** v2's name for `getImageNumChannels`. */
export const getNumChannels = getImageNumChannels;

/**
 * `[height, width]` — **the opposite order to `getImageSize`**, which is one namespace
 * away and answers `[width, height]`.
 */
export function getSize(img: Image): [number, number] {
  const [w = 0, h = 0] = getImageSize(img);
  return [h, w];
}

/**
 * One channel becomes three.
 *
 * **Three channels pass through rather than raising**, so a pipeline mixing grey and
 * colour needs no branch. A version that stacked regardless would answer nine, which is
 * a shape error three calls later rather than here.
 */
export function grayscaleToRgb(img: Image): Image {
  if (img.channels === 3) return img;
  if (img.channels !== 1) {
    throw new Error(`grayscale_to_rgb wants 1 or 3 channels, got ${img.channels}.`);
  }
  const out = new Float64Array(img.height * img.width * 3);
  for (let i = 0; i < img.height * img.width; i++) {
    const v = img.data[i] ?? 0;
    out[i * 3] = v;
    out[i * 3 + 1] = v;
    out[i * 3 + 2] = v;
  }
  return image(out, img.height, img.width, 3, img.isByte);
}

/**
 * Channels reordered. `[2, 0, 1]` moves the third to the front.
 *
 * **The list is where each output channel reads from**, not where each input channel
 * goes. The two readings differ for every permutation that is not its own inverse, and
 * both produce a plausible picture.
 */
export function permuteChannels(img: Image, order: readonly number[]): Image {
  if (order.length !== img.channels) {
    throw new Error(
      `permute_channels wants ${img.channels} positions, got ${order.length}.`,
    );
  }
  const out = new Float64Array(img.data.length);
  for (let i = 0; i < img.height * img.width; i++) {
    for (let c = 0; c < img.channels; c++) {
      out[i * img.channels + c] = img.data[i * img.channels + (order[c] ?? 0)] ?? 0;
    }
  }
  return image(out, img.height, img.width, img.channels, img.isByte);
}

/**
 * To a float image, **scaling or not**, and the flag is the whole question.
 *
 * With `scale`, a byte image's 0–255 becomes 0–1 the way `ToTensor` does it. Without,
 * the numbers are carried across unchanged and a 255 stays a 255. Both are ordinary
 * requests — the first for a network, the second for anything that still means to
 * measure pixels — and picking the wrong one gives values off by 255× that are
 * perfectly well-formed.
 */
export function toDtype(img: Image, _dtype: unknown = "float32", scale = false): Image {
  const divide = scale && img.isByte;
  const out = new Float64Array(img.data.length);
  for (let i = 0; i < out.length; i++) out[i] = (img.data[i] ?? 0) / (divide ? 255 : 1);
  return image(out, img.height, img.width, img.channels, false);
}

/**
 * Gaussian noise added to a float image.
 *
 * **`clip` is applied after the addition**, to `[0, 1]`. With `sigma` at zero this is
 * `x + mean`, which is what makes it askable at all — every other setting draws, and a
 * drawn value has no frozen answer.
 */
export function gaussianNoise(img: Image, mean = 0.0, sigma = 0.1, clip = true): Image {
  const out = new Float64Array(img.data.length);
  for (let i = 0; i < out.length; i++) {
    let v = (img.data[i] ?? 0) + mean;
    if (sigma !== 0) v += sigma * gaussianDraw();
    out[i] = clip ? Math.min(Math.max(v, 0), 1) : v;
  }
  return image(out, img.height, img.width, img.channels, img.isByte);
}

/** Box–Muller, for the `sigma > 0` path. Nothing frozen reaches it. */
function gaussianDraw(): number {
  const u = 1 - Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * Math.random());
}

/**
 * v2's name for the warp. **`fill` defaults to `null` and not to `0`.**
 *
 * Written as `0` it paints the outside of the warp black, and on a picture whose edges
 * barely move that reads as the warp working — a defect visible only at the border of
 * an image that mostly did not move. The Python side had it as `0` first and comparing
 * is what caught it.
 *
 * **The displacement is a flat array and not a tensor**, which is where this parts from
 * torchvision's spelling. Reading a tensor here means `await`, and `elasticTransform`
 * next door is synchronous — one async name in a file of synchronous ones is a worse
 * mismatch than this one, since it changes how every caller writes the line.
 */
export function elastic(
  img: Image,
  displacement: ArrayLike<number>,
  interpolation: "bilinear" | "nearest" = "bilinear",
  fill: number | readonly number[] | null = null,
): Image {
  return elasticTransform(img, displacement, interpolation, fill);
}
