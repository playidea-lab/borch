/**
 * `torchvision.transforms.v2` — torchvision's current recommended API.
 *
 * ## What v2 changes, and what it does not
 *
 * **It changed what its transforms print, not what they compute.** Measured
 * across the comparable names, the values agree everywhere; the reprs differ in
 * 21 of 33. `Resize(5)` keeps its size as `[5]`, `ColorJitter` drops the
 * arguments left unset rather than printing them. So the names v1 already has
 * are the v1 arithmetic reached through a v2 spelling, and this file's job for
 * those is the printing.
 *
 * On top of them v2 adds a set that v1 has no answer for at all, and those are
 * what this file first grew: `Identity`, `RGB`, `ToImage`, `ToDtype`,
 * `ToPureTensor`, `GaussianNoise`, `RandomChannelPermutation`,
 * `RandomPhotometricDistort`, `RandomResize`, `RandomShortestSize`,
 * `RandomZoomOut` and `ScaleJitter`.
 *
 * ## Why the shared ones delegate rather than inherit
 *
 * The Python side subclasses v1 and overrides the repr. That is the shorter
 * spelling and **it does not survive the crossing.** `vision.ts` has to name
 * this namespace to expose it as `vision.v2`, so the two modules import each
 * other, and in a cycle a class body running `extends Resize` reads `Resize`
 * before its own module has finished evaluating — a temporal dead zone, which is
 * a crash at import rather than a wrong value. Delegation moves that read to
 * call time, when both modules are up.
 *
 * What inheritance was buying is still bought: the v1 object *is* the
 * implementation, so a v2 name cannot drift from its v1 twin without the twin
 * moving too. The golden asks exactly that — `v2::Resize(inherited)` holds the
 * v2 spelling against v1's own frozen answer.
 *
 * ## What is not here
 *
 * The tv_tensor half — boxes, masks, keypoints and video travelling beside the
 * picture, and the dispatch that routes on their type. It pays off with a
 * detector and there is none in this catalogue.
 */

import { RuntimeError } from "./errors.js";
import { Tensor } from "./tensor.js";
import type { DType } from "./dtype.js";
import {
  asImage, nextFloat, nextInt, nextNormal, nextPermutation, pyFloat,
} from "./_vision_util.js";
import {
  adjustBrightness, adjustContrast, adjustHue, adjustSaturation,
  image, Pad, Resize,
  type Image, type Subject, type Transform,
} from "./vision.js";

// The composition names are v1's, re-exported. **A re-export is safe in the cycle
// where a subclass is not**: nothing reads the binding while the modules are still
// evaluating. Their printing is v1's too until the repr layer lands, which is the
// one place this file is knowingly incomplete rather than deliberately narrow.
export { Compose, Lambda, RandomApply } from "./vision.js";

// **The v1 names, re-exported rather than twinned — for now.** v2 prints these
// differently and computes them identically, so what is missing is the printing and
// not the arithmetic. Re-exporting says that plainly; a twin whose `describe` still
// answered v1's text would say it in a way nothing could measure. Three are here
// because the golden asks their values through the v2 spelling — the rest arrive with
// the repr layer.
export { CenterCrop, Pad, Resize } from "./vision.js";

// `v2.functional`, in its own file for the reason `functional.ts` is in one: it is a
// namespace of its own on the other side, and a name that lives only inside another
// module's object is invisible to the reference generator. The cycle here is the same
// shape as the one with `vision.ts` and is safe for the same reason — `v2f` calls these
// classes from inside functions, never at module scope.
export * as functional from "./v2f.js";

/** Does nothing, and **that is a transform** — it is what a policy draws when it
 *  draws no operation, and what a `Compose` holds when a branch is switched off. */
export class Identity implements Transform {
  apply(x: Subject): Subject {
    return x;
  }

  describe(): string {
    return "Identity()";
  }
}

/**
 * Strips the tv_tensor wrappers off a sample. **Here there are none**, so it is
 * the identity — kept because a pipeline copied from torchvision ends with it and
 * should not stop, and named rather than aliased to `Identity` because the two
 * mean different things the day tv_tensors arrive.
 */
export class ToPureTensor implements Transform {
  apply(x: Subject): Subject {
    return x;
  }

  describe(): string {
    return "ToPureTensor()";
  }
}

/**
 * One channel to three. **A three-channel picture passes through**, which is what
 * makes it safe to put in front of a model that needs three.
 */
export class RGB implements Transform {
  apply(x: Subject): Image {
    const img = asImage(x, "RGB");
    if (img.channels === 3) return img;
    if (img.channels !== 1) {
      throw new RuntimeError(
        `RGB takes a 1- or 3-channel picture — it received ${img.channels}.`);
    }
    const out = new Float64Array(img.height * img.width * 3);
    for (let i = 0; i < img.height * img.width; i++) {
      const v = img.data[i] as number;
      out[i * 3] = v; out[i * 3 + 1] = v; out[i * 3 + 2] = v;
    }
    return image(out, img.height, img.width, 3, img.isByte);
  }

  describe(): string {
    return "RGB()";
  }
}

/**
 * `(H,W,C)` to a `(C,H,W)` tensor — **and it does not divide by 255.**
 *
 * That is the whole reason v2 split `ToTensor` in two. `ToTensor` both moved the
 * axes and scaled, so a float image was scaled a second time by anyone who did
 * not know; here the moving is one transform and the scaling is
 * `ToDtype(scale=true)`, and each says which it does.
 */
export class ToImage implements Transform {
  apply(x: Subject): Tensor {
    if (x instanceof Tensor) return x;
    const img = asImage(x, "ToImage");
    const { data, height, width, channels } = img;
    const out = new Float32Array(channels * height * width);
    for (let c = 0; c < channels; c++) {
      for (let h = 0; h < height; h++) {
        for (let w = 0; w < width; w++) {
          out[(c * height + h) * width + w] = data[(h * width + w) * channels + c] as number;
        }
      }
    }
    // **`int64` where torchvision hands back `uint8`**, and it is the core's limit
    // rather than this file's: there is no uint8 storage to hand back. The numbers
    // are the same 0..255 in a wider box, and it stops mattering the moment
    // `ToDtype("float32", true)` runs — the pair v2 tells you to write.
    return Tensor.from(out, [channels, height, width],
                       img.isByte ? { dtype: "int64" } : {});
  }

  describe(): string {
    return "ToImage()";
  }
}

/**
 * Cast, and **optionally scale on the way** — the half of `ToTensor` that v2
 * split out.
 *
 * `scale` defaults to `false` as it does there, and that is the trap:
 * `new ToDtype("float32").apply(bytes)` gives 0..255 floats and looks like it
 * worked.
 */
export class ToDtype implements Transform {
  constructor(
    private readonly dtype: DType,
    private readonly scale: boolean = false,
  ) {}

  apply(x: Subject): Subject {
    const wantsFloat = this.dtype === "float32";
    if (x instanceof Tensor) {
      const fromInt = x.dtype !== "float32";
      if (this.scale && fromInt && wantsFloat) return x.to("float32").div(Tensor.from([255]));
      if (this.scale && !fromInt && !wantsFloat) return x.mul(Tensor.from([255])).to(this.dtype);
      return x.to(this.dtype);
    }
    const img = asImage(x, "ToDtype");
    // On a picture the dtype is the `isByte` flag: that is what the rest of this
    // library reads to decide whether values run 0..255 or 0..1, and a cast that
    // changed the numbers without moving the flag would leave the two disagreeing.
    if (this.scale && img.isByte && wantsFloat) {
      return image(img.data.map((v) => v / 255), img.height, img.width, img.channels, false);
    }
    if (this.scale && !img.isByte && !wantsFloat) {
      return image(img.data.map((v) => Math.trunc(v * 255)), img.height, img.width,
                   img.channels, true);
    }
    return image(wantsFloat ? img.data : img.data.map(Math.trunc),
                 img.height, img.width, img.channels, !wantsFloat);
  }

  describe(): string {
    return `ToDtype(scale=${this.scale ? "True" : "False"})`;
  }
}

/**
 * Add normal noise. **Float pictures only** — torchvision has an integer path
 * that works in int16 and clamps, and a uint8 picture with sigma in units of
 * [0,1] is a different question, so it is refused here rather than answered
 * differently.
 */
export class GaussianNoise implements Transform {
  constructor(
    private readonly mean: number = 0.0,
    private readonly sigma: number = 0.1,
    private readonly clip: boolean = true,
  ) {
    if (sigma < 0) {
      throw new RuntimeError(
        `sigma shouldn't be negative. Got ${sigma}\n` +
        "(torch: sigma shouldn't be negative)");
    }
  }

  apply(x: Subject): Image {
    const img = asImage(x, "GaussianNoise");
    if (img.isByte) {
      throw new RuntimeError(
        "GaussianNoise takes a float picture — it received bytes.\n" +
        "  Its sigma is in the units of a normalised image; on bytes the same " +
        "number means something else.");
    }
    const out = img.data.map((v) => {
      const noised = v + this.mean + nextNormal() * this.sigma;
      return this.clip ? Math.min(Math.max(noised, 0), 1) : noised;
    });
    return image(out, img.height, img.width, img.channels, false);
  }

  describe(): string {
    return `GaussianNoise(mean=${pyFloat(this.mean)}, sigma=${pyFloat(this.sigma)}, ` +
      `clip=${this.clip ? "True" : "False"})`;
  }
}

/** Reorder the channels at random. **Every ordering including the original** — it
 *  is a draw over permutations, not a guarantee of change. */
export class RandomChannelPermutation implements Transform {
  apply(x: Subject): Image {
    const img = asImage(x, "RandomChannelPermutation");
    return permuted(img, nextPermutation(img.channels));
  }

  describe(): string {
    return "RandomChannelPermutation()";
  }
}

/** Takes channel `order[i]` into position `i` — the direction indexing reads in. */
function permuted(img: Image, order: readonly number[]): Image {
  const out = new Float64Array(img.data.length);
  for (let i = 0; i < img.height * img.width; i++) {
    for (let c = 0; c < img.channels; c++) {
      out[i * img.channels + c] = img.data[i * img.channels + (order[c] as number)] as number;
    }
  }
  return image(out, img.height, img.width, img.channels, img.isByte);
}

/**
 * The SSD recipe: each of four adjustments applied with probability `p`, **the
 * contrast either before or after the other two**, and then maybe a channel
 * shuffle.
 *
 * The contrast's position is itself a coin flip, which reads as a detail and is
 * not: contrast measures the picture's mean, so doing it first and doing it last
 * are different pictures.
 */
export class RandomPhotometricDistort implements Transform {
  constructor(
    private readonly brightness: readonly [number, number] = [0.875, 1.125],
    private readonly contrast: readonly [number, number] = [0.5, 1.5],
    private readonly saturation: readonly [number, number] = [0.5, 1.5],
    private readonly hue: readonly [number, number] = [-0.05, 0.05],
    private readonly p: number = 0.5,
  ) {}

  apply(x: Subject): Image {
    let img = asImage(x, "RandomPhotometricDistort");
    const draw = (span: readonly [number, number]): number | null =>
      (nextFloat() < this.p ? span[0] + nextFloat() * (span[1] - span[0]) : null);
    const brightness = draw(this.brightness);
    const contrast = draw(this.contrast);
    const saturation = draw(this.saturation);
    const hue = draw(this.hue);
    const contrastFirst = nextFloat() < 0.5;
    const shuffle = nextFloat() < this.p;
    if (brightness !== null) img = adjustBrightness(img, brightness);
    if (contrast !== null && contrastFirst) img = adjustContrast(img, contrast);
    if (saturation !== null) img = adjustSaturation(img, saturation);
    if (hue !== null) img = adjustHue(img, hue);
    if (contrast !== null && !contrastFirst) img = adjustContrast(img, contrast);
    return shuffle ? permuted(img, nextPermutation(img.channels)) : img;
  }

  describe(): string {
    return `RandomPhotometricDistort(brightness=${pair(this.brightness)}, ` +
      `contrast=${pair(this.contrast)}, hue=${pair(this.hue)}, ` +
      `saturation=${pair(this.saturation)}, p=${this.p})`;
  }
}

/** Python's tuple spelling, which is what the frozen reprs hold. */
function pair(values: readonly number[]): string {
  return `(${values.map((v) => String(v)).join(", ")})`;
}

/** The same, for fields torchvision holds as floats — `(1.0, 4.0)` and not `(1, 4)`. */
function floatPair(values: readonly number[]): string {
  return `(${values.map(pyFloat).join(", ")})`;
}

/**
 * Resize the short side to a number drawn from `[minSize, maxSize)`. **A range of
 * sizes rather than one**, which is what multi-scale training wants.
 */
export class RandomResize implements Transform {
  constructor(
    private readonly minSize: number,
    private readonly maxSize: number,
    private readonly interpolation: "bilinear" | "nearest" = "bilinear",
    private readonly antialias: boolean = true,
  ) {}

  apply(x: Subject): Subject {
    const img = asImage(x, "RandomResize");
    const size = this.minSize + nextInt(this.maxSize - this.minSize);
    return new Resize(size, this.interpolation).apply(img);
  }

  describe(): string {
    return `RandomResize(min_size=${this.minSize}, max_size=${this.maxSize}, ` +
      `interpolation=${this.interpolation}, ` +
      `antialias=${this.antialias ? "True" : "False"})`;
  }
}

/**
 * The short side to one of `minSize`, **with the long side capped** by `maxSize`
 * — so a very wide picture is scaled by whichever of the two constraints binds
 * first, rather than by the short side alone.
 */
export class RandomShortestSize implements Transform {
  private readonly sizes: readonly number[];

  constructor(
    minSize: number | readonly number[],
    private readonly maxSize: number | null = null,
    private readonly interpolation: "bilinear" | "nearest" = "bilinear",
    private readonly antialias: boolean = true,
  ) {
    this.sizes = typeof minSize === "number" ? [Math.trunc(minSize)]
                                             : minSize.map((s) => Math.trunc(s));
  }

  apply(x: Subject): Subject {
    const img = asImage(x, "RandomShortestSize");
    const drawn = this.sizes[nextInt(this.sizes.length)] as number;
    let ratio = drawn / Math.min(img.height, img.width);
    if (this.maxSize !== null) {
      ratio = Math.min(ratio, this.maxSize / Math.max(img.height, img.width));
    }
    return new Resize([Math.trunc(img.height * ratio), Math.trunc(img.width * ratio)],
                      this.interpolation).apply(img);
  }

  describe(): string {
    return `RandomShortestSize(min_size=[${this.sizes.join(", ")}], ` +
      `max_size=${this.maxSize === null ? "None" : this.maxSize}, ` +
      `interpolation=${this.interpolation}, ` +
      `antialias=${this.antialias ? "True" : "False"})`;
  }
}

/**
 * Put the picture on a **larger canvas**, somewhere random on it, with the rest
 * filled. The picture shrinks relative to the frame without being resampled —
 * which is why detection recipes reach for it rather than for a scale-down.
 */
export class RandomZoomOut implements Transform {
  constructor(
    private readonly fill: number | readonly number[] = 0,
    private readonly sideRange: readonly [number, number] = [1.0, 4.0],
    private readonly p: number = 0.5,
  ) {
    if (sideRange[0] < 1.0 || sideRange[0] > sideRange[1]) {
      throw new RuntimeError(
        `Invalid side range provided ${pair(sideRange)}.\n` +
        "(torch: Invalid canvas side range provided)");
    }
  }

  apply(x: Subject): Subject {
    const img = asImage(x, "RandomZoomOut");
    if (nextFloat() >= this.p) return img;
    const ratio = this.sideRange[0] + nextFloat() * (this.sideRange[1] - this.sideRange[0]);
    const canvasW = Math.trunc(img.width * ratio);
    const canvasH = Math.trunc(img.height * ratio);
    const left = Math.trunc((canvasW - img.width) * nextFloat());
    const top = Math.trunc((canvasH - img.height) * nextFloat());
    return new Pad([left, top, canvasW - (left + img.width), canvasH - (top + img.height)],
                   this.fill).apply(img);
  }

  describe(): string {
    return `RandomZoomOut(p=${this.p}, fill=${this.fill}, ` +
      `side_range=${floatPair(this.sideRange)})`;
  }
}

/**
 * Resize toward `targetSize` by a **drawn factor** — the large-scale jitter of
 * the detection recipes, where the same picture is seen at a tenth and at twice
 * its size across an epoch.
 */
export class ScaleJitter implements Transform {
  constructor(
    private readonly targetSize: readonly [number, number],
    private readonly scaleRange: readonly [number, number] = [0.1, 2.0],
    private readonly interpolation: "bilinear" | "nearest" = "bilinear",
    private readonly antialias: boolean = true,
  ) {}

  apply(x: Subject): Subject {
    const img = asImage(x, "ScaleJitter");
    const scale = this.scaleRange[0]
      + nextFloat() * (this.scaleRange[1] - this.scaleRange[0]);
    const ratio = Math.min(this.targetSize[1] / img.height,
                           this.targetSize[0] / img.width) * scale;
    return new Resize([Math.trunc(img.height * ratio), Math.trunc(img.width * ratio)],
                      this.interpolation).apply(img);
  }

  describe(): string {
    return `ScaleJitter(target_size=${pair(this.targetSize)}, ` +
      `scale_range=${floatPair(this.scaleRange)}, ` +
      `interpolation=${this.interpolation}, ` +
      `antialias=${this.antialias ? "True" : "False"})`;
  }
}
