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

import { Tensor } from "./tensor.js";
import type { DType } from "./dtype.js";
import { RuntimeError } from "./errors.js";
import {
  adjustHue,
  elasticTransform,
  getImageNumChannels,
  getImageSize,
  hflip,
  type Image,
  image,
  normalize,
  pyBool,
  pyFloat,
  resize,
  rotate,
  type Subject,
  type Transform,
  vflip,
} from "./vision.js";

// **The thirty-six twins live next door**, in `vision_v2_twins.ts`, and are re-exported
// here so that `v2` is one namespace to a caller. They are split off because they are a
// different kind of thing — v1's behaviour under v2's printed surface, where everything
// in this file is either new or a straight binding.
export * from "./vision_v2_twins.js";

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
export function permuteChannels(img: Image, permutation: readonly number[]): Image {
  if (permutation.length !== img.channels) {
    throw new Error(
      `permute_channels wants ${img.channels} positions, got ${permutation.length}.`,
    );
  }
  const out = new Float64Array(img.data.length);
  for (let i = 0; i < img.height * img.width; i++) {
    for (let c = 0; c < img.channels; c++) {
      out[i * img.channels + c] = img.data[i * img.channels + (permutation[c] ?? 0)] ?? 0;
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
export function toDtype(img: Image, dtype: DType = "float32", scale = false): Image {
  // **The argument is torch's `dtype` and it is read rather than ignored.** There is
  // one float type in this subset, so any other request cannot be honoured — and a
  // parameter named after torch's that silently accepts anything is worse than one
  // that is missing, because the caller's `"float64"` looks like it worked.
  if (dtype !== "float32") {
    throw new RuntimeError(
      `to_dtype takes float32 here — got ${JSON.stringify(dtype)}.\n` +
      "  Storage is float32 in this subset, so there is nothing else to cast into.");
  }
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

// ── The transform classes v2 adds ────────────────────────────────────────────
//
// **v2's printing rule, and it is most of what separates the two namespaces.** Each
// class declares the fields it shows, in the order its constructor assigns them, and a
// field left unset **disappears from the line** rather than printing as `None` —
// torchvision filters by type and `None` is not among the kinds it keeps.
//
// The fields are declared rather than read off the object because the behaviour comes
// from v1, whose attributes have their own names and their own order. Declaring keeps
// one implementation of the arithmetic and one statement of the surface.

/** One shown field, already rendered the way Python would print it. */
type Field = readonly [string, string];

function v2Repr(name: string, fields: readonly Field[] = []): string {
  return `${name}(${fields.map(([k, v]) => `${k}=${v}`).join(", ")})`;
}

/**
 * Does nothing, and **that is a transform** — it is what a policy draws when it draws
 * no operation, and what a `Compose` holds when a branch is switched off.
 */
export class Identity implements Transform {
  apply(x: Subject): Subject {
    return x;
  }

  describe(): string {
    return v2Repr("Identity");
  }
}

/**
 * Strips the tv_tensor wrappers off a sample. **Here there are none**, so it is the
 * identity — kept because a pipeline copied from torchvision ends with it and should
 * not stop, and named rather than aliased to `Identity` because the two mean different
 * things the day tv_tensors arrive.
 */
export class ToPureTensor implements Transform {
  apply(x: Subject): Subject {
    return x;
  }

  describe(): string {
    return v2Repr("ToPureTensor");
  }
}

/**
 * One channel to three. A three-channel picture passes through, which is what makes it
 * safe to put in front of a model that needs three.
 */
export class RGB implements Transform {
  apply(x: Subject): Subject {
    return grayscaleToRgb(x as Image);
  }

  describe(): string {
    return v2Repr("RGB");
  }
}

/**
 * `(H, W, C)` to a `(C, H, W)` tensor — **and it does not divide by 255.**
 *
 * That is the whole reason v2 split `ToTensor` in two. `ToTensor` both moved the axes
 * and scaled, so a float image was scaled a second time by anyone who did not know;
 * here the moving is one transform and the scaling is `ToDtype(scale: true)`, and each
 * says which it does.
 */
export class ToImage implements Transform {
  apply(x: Subject): Subject {
    const img = x as Image;
    const { data, height: h, width: w, channels: c } = img;
    const out = new Float32Array(c * h * w);
    for (let k = 0; k < c; k++) {
      for (let i = 0; i < h; i++) {
        for (let j = 0; j < w; j++) out[(k * h + i) * w + j] = data[(i * w + j) * c + k] ?? 0;
      }
    }
    return Tensor.from(out, [c, h, w]);
  }

  describe(): string {
    return v2Repr("ToImage");
  }
}

/**
 * Cast, and **optionally scale on the way.**
 *
 * `scale: true` is the half of the old `ToTensor` that divided; without it this only
 * changes the dtype, which is why the flag is not a default. **The dtype does not
 * appear in the repr** — it is not one of the kinds v2's filter keeps, so a line that
 * looks like it forgot to print its main argument is printing exactly what torchvision
 * prints.
 */
export class ToDtype implements Transform {
  constructor(private readonly dtype: DType = "float32",
              private readonly scale = false) {}

  /**
   * **It takes a tensor as well as a picture**, because the pair v2 tells you to write
   * instead of `ToTensor` is `Compose([ToImage(), ToDtype(float32, scale=true)])` — and
   * `ToImage` hands back a tensor. A version that only took pictures would refuse the
   * one composition this class exists for.
   */
  apply(x: Subject): Subject {
    if (x instanceof Tensor) {
      return this.scale ? x.div(Tensor.full([], 255)) : x;
    }
    return toDtype(x as Image, this.dtype, this.scale);
  }

  describe(): string {
    return v2Repr("ToDtype", [["scale", pyBool(this.scale)]]);
  }
}

/**
 * Add normal noise. **Float pictures only** — torchvision has an integer path that
 * works in int16 and clamps, and a byte picture with sigma in units of `[0, 1]` is a
 * different question, so it is refused rather than answered differently.
 */
export class GaussianNoise implements Transform {
  constructor(private readonly mean = 0.0, private readonly sigma = 0.1,
              private readonly clip = true) {
    if (sigma < 0) {
      throw new Error(`sigma shouldn't be negative. Got ${pyFloat(sigma)}`);
    }
  }

  apply(x: Subject): Subject {
    return gaussianNoise(x as Image, this.mean, this.sigma, this.clip);
  }

  describe(): string {
    return v2Repr("GaussianNoise", [
      ["mean", pyFloat(this.mean)],
      ["sigma", pyFloat(this.sigma)],
      ["clip", pyBool(this.clip)],
    ]);
  }
}

/**
 * Shuffle the channels. **Every ordering including the original** — it is a draw over
 * permutations, not a guarantee of change.
 */
export class RandomChannelPermutation implements Transform {
  apply(x: Subject): Subject {
    const img = x as Image;
    const order = [...Array(img.channels).keys()];
    for (let i = order.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [order[i], order[j]] = [order[j] as number, order[i] as number];
    }
    return permuteChannels(img, order);
  }

  describe(): string {
    return v2Repr("RandomChannelPermutation");
  }
}
