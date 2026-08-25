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
  asImage, checkSpan, nextBeta, nextFloat, nextInt, nextNormal, nextPermutation,
  pairOf, pyFloat, setupAngle, spanText, type Span,
} from "./_vision_util.js";
import {
  adjustBrightness, adjustContrast, adjustHue, adjustSaturation, image,
  type AutoAugmentPolicyName, type Image, type PaddingMode, type Subject,
  type Transform,
} from "./vision.js";
// **The v1 classes arrive under a namespace and not by name.** Every twin below is
// called what its v1 original is called, so a named import would be shadowed by the
// class it is meant to wrap — silently, since both are constructible.
import * as v1 from "./vision.js";

// `AutoAugmentPolicy` and `InterpolationMode`'s place are v1's — v2 re-exports the
// same objects rather than making second ones, because two enums equal by value and
// not by identity is the kind of difference that bites once and takes an hour.
export { AutoAugmentPolicy } from "./vision.js";

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
    return new v1.Resize(size, this.interpolation).apply(img);
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
    return new v1.Resize([Math.trunc(img.height * ratio), Math.trunc(img.width * ratio)],
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
    return new v1.Pad([left, top, canvasW - (left + img.width),
                       canvasH - (top + img.height)], this.fill).apply(img);
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
    return new v1.Resize([Math.trunc(img.height * ratio), Math.trunc(img.width * ratio)],
                         this.interpolation).apply(img);
  }

  describe(): string {
    return `ScaleJitter(target_size=${pair(this.targetSize)}, ` +
      `scale_range=${floatPair(this.scaleRange)}, ` +
      `interpolation=${this.interpolation}, ` +
      `antialias=${this.antialias ? "True" : "False"})`;
  }
}

// ── The thirty-eight twins ────────────────────────────────────────────────
//
// **What differs between the two namespaces is what each class stores, under what
// name and in what order** — v2 does not hand-write these strings either. Its
// `Transform.extra_repr` walks the instance, keeps whatever is a bool, number,
// string, tuple, list or enum, and joins them; `None` is not among the kinds it
// keeps, which is why `ColorJitter(0.5)` prints one field and v1's prints four.
//
// So each twin below **declares its printed fields** rather than reading them off
// the v1 object. Two reasons, and the second is the one that matters. TypeScript
// cannot read another class's private state at all — but even where it could,
// twelve of these agree with v1's line today by coincidence, and inheriting a
// coincidence is how a printed surface changes without anyone deciding to.
//
// The values are the **normalised** ones where v2 prints those: `RandomRotation(30)`
// says `degrees=[-30.0, 30.0]`, `ColorJitter(0.5)` says `brightness=(0.5, 1.5)`.
// Those normalisations are in `_vision_util.ts` so that the twin and the original
// call the same code — normalising twice is how two files come to disagree about
// what `0` means.

/** Python prints these capitalised, and the golden holds the string. */
function bool2(v: boolean): string {
  return v ? "True" : "False";
}

/** `[a, b]` — v2 keeps as a list several things v1 keeps as a tuple. */
function listOf(values: readonly number[], asFloat = false): string {
  return `[${values.map((v) => (asFloat ? pyFloat(v) : String(v))).join(", ")}]`;
}

/** A size argument as v2 stores it: one number is a one-element list. */
function sizeList(size: number | readonly number[]): string {
  return listOf(typeof size === "number" ? [size] : [...size]);
}

/** `fill` prints as it was given — a number stays a number, a sequence is a list. */
function fillText(fill: number | readonly number[] | null): string {
  if (fill === null) return "None";
  return typeof fill === "number" ? String(fill) : listOf(fill);
}

export class Resize implements Transform {
  private readonly inner: v1.Resize;

  constructor(
    private readonly size: number | readonly [number, number],
    private readonly interpolation: "bilinear" | "nearest" = "bilinear",
    maxSize: number | null = null,
    private readonly antialias = true,
  ) {
    this.inner = new v1.Resize(size, interpolation, maxSize, antialias);
  }

  apply(x: Image | Tensor): Image {
    return this.inner.apply(x);
  }

  // **`max_size` is stored and not printed**, which is v2's list and not an
  // oversight — v1 prints it and that is one of the 21 lines that differ.
  describe(): string {
    return `Resize(size=${sizeList(this.size)}, ` +
      `interpolation=${this.interpolation}, antialias=${bool2(this.antialias)})`;
  }
}

export class CenterCrop implements Transform {
  private readonly inner: v1.CenterCrop;

  constructor(private readonly size: number | readonly [number, number]) {
    this.inner = new v1.CenterCrop(size);
  }

  apply(x: Image | Tensor): Image {
    return this.inner.apply(x);
  }

  describe(): string {
    return `CenterCrop(size=${pair(pairOf(this.size))})`;
  }
}

export class RandomCrop implements Transform {
  private readonly inner: v1.RandomCrop;

  constructor(
    private readonly size: number | readonly [number, number],
    padding: number | readonly number[] | null = null,
    private readonly padIfNeeded = false,
    private readonly fill: number | readonly number[] = 0,
    private readonly paddingMode: PaddingMode = "constant",
  ) {
    this.inner = new v1.RandomCrop(size, padding, padIfNeeded, fill, paddingMode);
  }

  apply(x: Image | Tensor): Image {
    return this.inner.apply(x);
  }

  // `padding` is stored and not printed — v2 keeps it under a private name.
  describe(): string {
    return `RandomCrop(size=${pair(pairOf(this.size))}, ` +
      `pad_if_needed=${bool2(this.padIfNeeded)}, fill=${fillText(this.fill)}, ` +
      `padding_mode=${this.paddingMode})`;
  }
}

export class RandomResizedCrop implements Transform {
  private readonly inner: v1.RandomResizedCrop;

  constructor(
    private readonly size: number | readonly [number, number],
    private readonly scale: readonly [number, number] = [0.08, 1.0],
    private readonly ratio: readonly [number, number] = [3 / 4, 4 / 3],
    private readonly interpolation: "bilinear" | "nearest" = "bilinear",
    private readonly antialias = true,
  ) {
    this.inner = new v1.RandomResizedCrop(size, scale, ratio, interpolation, antialias);
  }

  apply(x: Subject): Subject {
    return this.inner.apply(x);
  }

  describe(): string {
    return `RandomResizedCrop(size=${pair(pairOf(this.size))}, ` +
      `scale=${floatPair(this.scale)}, ratio=${pair(this.ratio)}, ` +
      `interpolation=${this.interpolation}, antialias=${bool2(this.antialias)})`;
  }
}

export class FiveCrop implements Transform {
  private readonly inner: v1.FiveCrop;

  constructor(private readonly size: number | readonly [number, number]) {
    this.inner = new v1.FiveCrop(size);
  }

  apply(x: Subject): Subject {
    return this.inner.apply(x);
  }

  describe(): string {
    return `FiveCrop(size=${pair(pairOf(this.size))})`;
  }
}

export class TenCrop implements Transform {
  private readonly inner: v1.TenCrop;

  constructor(
    private readonly size: number | readonly [number, number],
    private readonly verticalFlip = false,
  ) {
    this.inner = new v1.TenCrop(size, verticalFlip);
  }

  apply(x: Subject): Subject {
    return this.inner.apply(x);
  }

  describe(): string {
    return `TenCrop(size=${pair(pairOf(this.size))}, ` +
      `vertical_flip=${bool2(this.verticalFlip)})`;
  }
}

export class Pad implements Transform {
  private readonly inner: v1.Pad;

  constructor(
    private readonly padding: number | readonly number[],
    private readonly fill: number | readonly number[] = 0,
    private readonly paddingMode: PaddingMode = "constant",
  ) {
    this.inner = new v1.Pad(padding, fill, paddingMode);
  }

  apply(x: Image | Tensor): Image {
    return this.inner.apply(x);
  }

  describe(): string {
    return `Pad(padding=${fillText(this.padding)}, fill=${fillText(this.fill)}, ` +
      `padding_mode=${this.paddingMode})`;
  }
}

export class RandomHorizontalFlip implements Transform {
  private readonly inner: v1.RandomHorizontalFlip;

  constructor(private readonly p = 0.5) {
    this.inner = new v1.RandomHorizontalFlip(p);
  }

  apply(x: Image | Tensor): Image {
    return this.inner.apply(x);
  }

  describe(): string {
    return `RandomHorizontalFlip(p=${this.p})`;
  }
}

export class RandomVerticalFlip implements Transform {
  private readonly inner: v1.RandomVerticalFlip;

  constructor(private readonly p = 0.5) {
    this.inner = new v1.RandomVerticalFlip(p);
  }

  apply(x: Image | Tensor): Image {
    return this.inner.apply(x);
  }

  describe(): string {
    return `RandomVerticalFlip(p=${this.p})`;
  }
}

export class Grayscale implements Transform {
  private readonly inner: v1.Grayscale;

  constructor(private readonly numOutputChannels = 1) {
    this.inner = new v1.Grayscale(numOutputChannels);
  }

  apply(x: Image | Tensor): Image {
    return this.inner.apply(x);
  }

  describe(): string {
    return `Grayscale(num_output_channels=${this.numOutputChannels})`;
  }
}

export class RandomGrayscale implements Transform {
  private readonly inner: v1.RandomGrayscale;

  constructor(private readonly p = 0.1) {
    this.inner = new v1.RandomGrayscale(p);
  }

  apply(x: Image | Tensor): Image {
    return this.inner.apply(x);
  }

  describe(): string {
    return `RandomGrayscale(p=${this.p})`;
  }
}

export class Normalize implements Transform {
  private readonly inner: v1.Normalize;

  // **`inplace` is taken and not acted on.** There is no in-place path on this side;
  // v2 stores the flag and prints it, so a pipeline copied from torchvision keeps its
  // argument list and its printed line rather than losing one of them.
  constructor(
    private readonly mean: readonly number[],
    private readonly std: readonly number[],
    private readonly inplace = false,
  ) {
    this.inner = new v1.Normalize(mean, std);
  }

  apply(x: Subject): Subject {
    return this.inner.apply(x as never);
  }

  describe(): string {
    return `Normalize(mean=${listOf(this.mean)}, std=${listOf(this.std)}, ` +
      `inplace=${bool2(this.inplace)})`;
  }
}

export class RandomErasing implements Transform {
  private readonly inner: v1.RandomErasing;

  constructor(
    private readonly p = 0.5,
    private readonly scale: readonly [number, number] = [0.02, 0.33],
    private readonly ratio: readonly [number, number] = [0.3, 3.3],
    private readonly value: number | readonly number[] | "random" = 0,
    private readonly inplace = false,
  ) {
    this.inner = new v1.RandomErasing(p, scale, ratio, value, inplace);
  }

  apply(x: Subject): Subject {
    return this.inner.apply(x);
  }

  // **`value=0` prints as `[0.0]`.** v2 stores the fill as a list of floats whatever
  // came in, and `"random"` stays the string it was.
  describe(): string {
    const value = typeof this.value === "string" ? this.value
      : listOf(typeof this.value === "number" ? [this.value] : this.value, true);
    return `RandomErasing(p=${this.p}, scale=${pair(this.scale)}, ` +
      `ratio=${pair(this.ratio)}, value=${value}, inplace=${bool2(this.inplace)})`;
  }
}

export class ColorJitter implements Transform {
  private readonly inner: v1.ColorJitter;
  private readonly spans: readonly (readonly [string, Span])[];

  constructor(
    brightness?: number | readonly [number, number],
    contrast?: number | readonly [number, number],
    saturation?: number | readonly [number, number],
    hue?: number | readonly [number, number],
  ) {
    this.inner = new v1.ColorJitter(brightness, contrast, saturation, hue);
    this.spans = [
      ["brightness", checkSpan(brightness, "brightness", 1, 0, Infinity, true)],
      ["contrast", checkSpan(contrast, "contrast", 1, 0, Infinity, true)],
      ["saturation", checkSpan(saturation, "saturation", 1, 0, Infinity, true)],
      ["hue", checkSpan(hue, "hue", 0, -0.5, 0.5, false)],
    ];
  }

  apply(x: Subject): Subject {
    return this.inner.apply(x);
  }

  // **The clearest case of v2's type filter doing the work.** A factor nobody asked
  // for is `None`, and `None` is not a kind v2 prints — so the default constructor
  // prints its own name and nothing else, where v1 prints four `None`s.
  describe(): string {
    const parts = this.spans
      .filter(([, span]) => span !== null)
      .map(([name, span]) => `${name}=${spanText(span)}`);
    return `ColorJitter(${parts.join(", ")})`;
  }
}

export class RandomInvert implements Transform {
  private readonly inner: v1.RandomInvert;

  constructor(private readonly p = 0.5) {
    this.inner = new v1.RandomInvert(p);
  }

  apply(x: Subject): Subject {
    return this.inner.apply(x);
  }

  describe(): string {
    return `RandomInvert(p=${this.p})`;
  }
}

export class RandomAutocontrast implements Transform {
  private readonly inner: v1.RandomAutocontrast;

  constructor(private readonly p = 0.5) {
    this.inner = new v1.RandomAutocontrast(p);
  }

  apply(x: Subject): Subject {
    return this.inner.apply(x);
  }

  describe(): string {
    return `RandomAutocontrast(p=${this.p})`;
  }
}

export class RandomEqualize implements Transform {
  private readonly inner: v1.RandomEqualize;

  constructor(private readonly p = 0.5) {
    this.inner = new v1.RandomEqualize(p);
  }

  apply(x: Subject): Subject {
    return this.inner.apply(x);
  }

  describe(): string {
    return `RandomEqualize(p=${this.p})`;
  }
}

// **`p` comes first in the printed line and second in the call.** That is v2's
// assignment order, not a tidier one, and the three below share it.
export class RandomPosterize implements Transform {
  private readonly inner: v1.RandomPosterize;

  constructor(private readonly bits: number, private readonly p = 0.5) {
    this.inner = new v1.RandomPosterize(bits, p);
  }

  apply(x: Subject): Subject {
    return this.inner.apply(x);
  }

  describe(): string {
    return `RandomPosterize(p=${this.p}, bits=${this.bits})`;
  }
}

export class RandomSolarize implements Transform {
  private readonly inner: v1.RandomSolarize;

  constructor(private readonly threshold: number, private readonly p = 0.5) {
    this.inner = new v1.RandomSolarize(threshold, p);
  }

  apply(x: Subject): Subject {
    return this.inner.apply(x);
  }

  describe(): string {
    return `RandomSolarize(p=${this.p}, threshold=${this.threshold})`;
  }
}

export class RandomAdjustSharpness implements Transform {
  private readonly inner: v1.RandomAdjustSharpness;

  constructor(private readonly sharpnessFactor: number, private readonly p = 0.5) {
    this.inner = new v1.RandomAdjustSharpness(sharpnessFactor, p);
  }

  apply(x: Subject): Subject {
    return this.inner.apply(x);
  }

  describe(): string {
    return `RandomAdjustSharpness(p=${this.p}, ` +
      `sharpness_factor=${this.sharpnessFactor})`;
  }
}

export class RandomRotation implements Transform {
  private readonly inner: v1.RandomRotation;
  private readonly degrees: [number, number];

  constructor(
    degrees: number | readonly number[],
    private readonly interpolation: "bilinear" | "nearest" = "nearest",
    private readonly expand = false,
    center: readonly [number, number] | null = null,
    private readonly fill: number | readonly number[] | null = 0,
  ) {
    this.inner = new v1.RandomRotation(degrees, interpolation, expand, center, fill);
    this.degrees = setupAngle(degrees, "degrees");
  }

  apply(x: Subject): Subject {
    return this.inner.apply(x);
  }

  describe(): string {
    return `RandomRotation(degrees=${listOf(this.degrees, true)}, ` +
      `interpolation=${this.interpolation}, expand=${bool2(this.expand)}, ` +
      `fill=${fillText(this.fill)})`;
  }
}

export class RandomAffine implements Transform {
  private readonly inner: v1.RandomAffine;
  private readonly degrees: [number, number];

  constructor(
    degrees: number | readonly number[],
    translate: readonly [number, number] | null = null,
    scale: readonly [number, number] | null = null,
    shear: number | readonly number[] | null = null,
    private readonly interpolation: "bilinear" | "nearest" = "nearest",
    private readonly fill: number | readonly number[] = 0,
    center: readonly [number, number] | null = null,
  ) {
    this.inner = new v1.RandomAffine(degrees, translate, scale, shear, interpolation,
                                     fill, center);
    this.degrees = setupAngle(degrees, "degrees");
  }

  apply(x: Subject): Subject {
    return this.inner.apply(x);
  }

  // `translate`, `scale` and `shear` are stored and left out: unset they are `None`,
  // and `None` is not a kind v2 prints.
  describe(): string {
    return `RandomAffine(degrees=${listOf(this.degrees, true)}, ` +
      `interpolation=${this.interpolation}, fill=${fillText(this.fill)})`;
  }
}

export class RandomPerspective implements Transform {
  private readonly inner: v1.RandomPerspective;

  constructor(
    private readonly distortionScale = 0.5,
    private readonly p = 0.5,
    private readonly interpolation: "bilinear" | "nearest" = "bilinear",
    private readonly fill: number | readonly number[] = 0,
  ) {
    this.inner = new v1.RandomPerspective(distortionScale, p, interpolation, fill);
  }

  apply(x: Subject): Subject {
    return this.inner.apply(x);
  }

  describe(): string {
    return `RandomPerspective(p=${this.p}, ` +
      `distortion_scale=${this.distortionScale}, ` +
      `interpolation=${this.interpolation}, fill=${fillText(this.fill)})`;
  }
}

export class ElasticTransform implements Transform {
  private readonly inner: v1.ElasticTransform;
  private readonly alphaPair: [number, number];
  private readonly sigmaPair: [number, number];

  // **The printed `fill` cannot be read back off the object.** v1 normalises it to a
  // list in its constructor and v2 prints the number as it was given, so the argument
  // is kept rather than recovered — recovering it would turn `0` into `[0.0]`.
  constructor(
    alpha: number | readonly number[] = 50,
    sigma: number | readonly number[] = 5,
    private readonly interpolation: "bilinear" | "nearest" = "bilinear",
    private readonly fill: number | readonly number[] = 0,
  ) {
    this.inner = new v1.ElasticTransform(alpha, sigma, interpolation, fill);
    this.alphaPair = typeof alpha === "number" ? [alpha, alpha] : [alpha[0] ?? 0, alpha[1] ?? 0];
    this.sigmaPair = typeof sigma === "number" ? [sigma, sigma] : [sigma[0] ?? 0, sigma[1] ?? 0];
  }

  apply(x: Subject): Subject {
    return this.inner.apply(x);
  }

  describe(): string {
    return `ElasticTransform(alpha=${listOf(this.alphaPair, true)}, ` +
      `sigma=${listOf(this.sigmaPair, true)}, interpolation=${this.interpolation}, ` +
      `fill=${fillText(this.fill)})`;
  }
}

export class GaussianBlur implements Transform {
  private readonly inner: v1.GaussianBlur;
  private readonly kernel: [number, number];
  private readonly sigmaPair: [number, number];

  constructor(
    kernelSize: number | readonly number[],
    sigma: number | readonly [number, number] = [0.1, 2.0],
  ) {
    this.inner = new v1.GaussianBlur(kernelSize, sigma);
    this.kernel = typeof kernelSize === "number" ? [kernelSize, kernelSize]
      : [kernelSize[0] ?? 0, kernelSize[1] ?? 0];
    this.sigmaPair = typeof sigma === "number" ? [sigma, sigma]
      : [sigma[0] ?? 0, sigma[1] ?? 0];
  }

  apply(x: Subject): Subject {
    return this.inner.apply(x);
  }

  describe(): string {
    // **`sigma` is a float pair and `kernel_size` is an int pair**, so one carries a
    // decimal point and the other does not — `[0.1, 2.0]` beside `(3, 3)`. torchvision
    // stores the sigma through `float()` whatever came in.
    return `GaussianBlur(kernel_size=${pair(this.kernel)}, ` +
      `sigma=${listOf(this.sigmaPair, true)})`;
  }
}

// **The policies put `interpolation` first**, where every other class here has it
// after the arguments that decide the picture. That is v2's assignment order.
export class AutoAugment implements Transform {
  private readonly inner: v1.AutoAugment;

  constructor(
    private readonly policy: AutoAugmentPolicyName = "imagenet",
    private readonly interpolation: "bilinear" | "nearest" = "nearest",
    fill: number | readonly number[] | null = null,
  ) {
    this.inner = new v1.AutoAugment(policy, interpolation, fill);
  }

  apply(x: Subject): Subject {
    return this.inner.apply(x);
  }

  // The policy prints as the enum member and not as its value — `IMAGENET`, not
  // `imagenet`, because Python's enum repr is what the golden froze.
  describe(): string {
    return `AutoAugment(interpolation=${this.interpolation}, ` +
      `policy=AutoAugmentPolicy.${this.policy.toUpperCase()})`;
  }
}

export class RandAugment implements Transform {
  private readonly inner: v1.RandAugment;

  constructor(
    private readonly numOps = 2,
    private readonly magnitude = 9,
    private readonly numMagnitudeBins = 31,
    private readonly interpolation: "bilinear" | "nearest" = "nearest",
    fill: number | readonly number[] | null = null,
  ) {
    this.inner = new v1.RandAugment(numOps, magnitude, numMagnitudeBins,
                                    interpolation, fill);
  }

  apply(x: Subject): Subject {
    return this.inner.apply(x);
  }

  describe(): string {
    return `RandAugment(interpolation=${this.interpolation}, ` +
      `num_ops=${this.numOps}, magnitude=${this.magnitude}, ` +
      `num_magnitude_bins=${this.numMagnitudeBins})`;
  }
}

export class TrivialAugmentWide implements Transform {
  private readonly inner: v1.TrivialAugmentWide;

  constructor(
    private readonly numMagnitudeBins = 31,
    private readonly interpolation: "bilinear" | "nearest" = "nearest",
    fill: number | readonly number[] | null = null,
  ) {
    this.inner = new v1.TrivialAugmentWide(numMagnitudeBins, interpolation, fill);
  }

  apply(x: Subject): Subject {
    return this.inner.apply(x);
  }

  describe(): string {
    return `TrivialAugmentWide(interpolation=${this.interpolation}, ` +
      `num_magnitude_bins=${this.numMagnitudeBins})`;
  }
}

export class AugMix implements Transform {
  private readonly inner: v1.AugMix;

  constructor(
    private readonly severity = 3,
    private readonly mixtureWidth = 3,
    private readonly chainDepth = -1,
    private readonly alpha = 1.0,
    private readonly allOps = true,
    private readonly interpolation: "bilinear" | "nearest" = "bilinear",
    fill: number | readonly number[] | null = null,
  ) {
    this.inner = new v1.AugMix(severity, mixtureWidth, chainDepth, alpha, allOps,
                               interpolation, fill);
  }

  apply(x: Subject): Subject {
    return this.inner.apply(x);
  }

  describe(): string {
    return `AugMix(interpolation=${this.interpolation}, severity=${this.severity}, ` +
      `mixture_width=${this.mixtureWidth}, chain_depth=${this.chainDepth}, ` +
      `alpha=${pyFloat(this.alpha)}, all_ops=${bool2(this.allOps)})`;
  }
}

export class LinearTransformation implements Transform {
  private readonly inner: v1.LinearTransformation;

  constructor(matrix: readonly (readonly number[])[], mean: readonly number[]) {
    this.inner = new v1.LinearTransformation(matrix, mean);
  }

  apply(x: Subject): Subject {
    return this.inner.apply(x as never);
  }

  // **It prints nothing at all**, and neither does `ToTensor` below. Their state is
  // arrays, and an array of arrays is not a kind v2's rule keeps — easy to mistake
  // for an unfinished repr, which is why it says so here.
  describe(): string {
    return "LinearTransformation()";
  }
}

export class ToTensor implements Transform {
  private readonly inner = new v1.ToTensor();

  apply(x: Subject): Subject {
    return this.inner.apply(x);
  }

  describe(): string {
    return "ToTensor()";
  }
}

// ── The containers, which print torch's `nn.Module` way ───────────────────

/**
 * torch's `nn.Module` printing, which is what v2's containers inherit.
 *
 * **One transform prints inline and two print over several lines**, and the indent
 * is not the same in the two cases — four spaces inline, six once it breaks, because
 * torch indents each line of `extra_repr` by two more when it wraps. Measured rather
 * than derived; it is the kind of thing nobody would guess and everybody would get
 * almost right.
 */
function moduleRepr(name: string, lines: readonly string[]): string {
  const body = lines.map((line) => `    ${line}`).join("\n");
  if (!body.includes("\n")) return `${name}(${body})`;
  const inner = body.split("\n").map((line) => `  ${line}`).join("\n");
  return `${name}(\n${inner}\n)`;
}

export class Compose implements Transform {
  private readonly inner: v1.Compose;

  constructor(private readonly transforms: readonly Transform[]) {
    this.inner = new v1.Compose(transforms);
  }

  apply(x: Subject): Subject {
    return this.inner.apply(x);
  }

  describe(): string {
    return moduleRepr("Compose", this.transforms.map((t) => t.describe()));
  }
}

export class RandomApply implements Transform {
  private readonly inner: v1.RandomApply;

  constructor(private readonly transforms: readonly Transform[], p = 0.5) {
    this.inner = new v1.RandomApply(transforms, p);
  }

  apply(x: Subject): Subject {
    return this.inner.apply(x);
  }

  // **`p` is stored and not printed**, unlike v1's — torch's module repr shows only
  // what `extra_repr` returns, and v2 leaves the probability out of it.
  describe(): string {
    return moduleRepr("RandomApply", this.transforms.map((t) => t.describe()));
  }
}

export class RandomOrder implements Transform {
  private readonly inner: v1.RandomOrder;

  constructor(private readonly transforms: readonly Transform[]) {
    this.inner = new v1.RandomOrder(transforms);
  }

  apply(x: Subject): Subject {
    return this.inner.apply(x);
  }

  describe(): string {
    return `RandomOrder(transforms=[${
      this.transforms.map((t) => t.describe()).join(", ")}])`;
  }
}

export class RandomChoice implements Transform {
  private readonly inner: v1.RandomChoice;
  private readonly weights: readonly number[];

  constructor(private readonly transforms: readonly Transform[],
              weights: readonly number[] | null = null) {
    this.inner = new v1.RandomChoice(transforms, weights);
    // **v2 fills `p` in.** v1 leaves it unset; v2 builds the uniform distribution and
    // stores it, so two transforms given no probabilities print `p=[0.5, 0.5]`.
    this.weights = weights ?? transforms.map(() => 1 / transforms.length);
  }

  apply(x: Subject): Subject {
    return this.inner.apply(x);
  }

  describe(): string {
    return `RandomChoice(transforms=[${
      this.transforms.map((t) => t.describe()).join(", ")}], ` +
      `p=${listOf(this.weights)})`;
  }
}

/**
 * v2's `Lambda` takes **the types it applies to** as well as the function — the one
 * place in this namespace where the constructor differs and not just the printing.
 *
 * Given tv_tensors it would run only on the kinds named; there is one kind here, so
 * the argument is kept and printed rather than acted on. Taking it and dropping it
 * silently would read as support.
 */
export class Lambda implements Transform {
  private readonly inner: v1.Lambda;
  private readonly types: readonly string[];

  constructor(private readonly fn: (x: Subject) => Subject,
              ...types: readonly string[]) {
    this.inner = new v1.Lambda(fn);
    this.types = types.length ? types : ["object"];
  }

  apply(x: Subject): Subject {
    return this.inner.apply(x);
  }

  describe(): string {
    return `Lambda(${this.fn.name}, types=[${
      this.types.map((t) => `'${t}'`).join(", ")}])`;
  }
}

// ── MixUp and CutMix: a batch, and labels that move with it ───────────────
//
// **The only two transforms in v2 whose input is a training pair rather than a
// picture.** Everything above takes `(H,W,C)` and gives `(H,W,C)` back; these take
// a batch with a label per row and give both back changed, because mixing two
// pictures without mixing their labels teaches the wrong thing.
//
// The pairing is **row `i` with row `i-1`** — a roll by one, not a random partner.
// torchvision says outright that this is an implementation detail and that the
// batch is expected to be shuffled already; kept the same here, because a recipe
// that shuffles for torchvision's sake would otherwise be silently unnecessary.
//
// `labels_getter` is **not taken.** In torchvision it exists to find the labels
// inside a nested sample — a dict, a tuple of tv_tensors — and there are none here,
// so the labels arrive as the second argument and nothing has to go looking. Taking
// the argument and ignoring it would read as support.

/** A batch of pictures, and a label per row: indices or one row of probabilities. */
export type Batch = readonly Image[];
export type Labels = readonly number[] | readonly (readonly number[])[];

/** What both share: the two checks torchvision makes, and the label blend. */
abstract class MixBase {
  constructor(readonly alpha: number = 1.0,
              readonly numClasses: number | null = null) {}

  /** **A wrong batch here is a silent mis-train**, since every shape involved is
   *  plausible — so the shapes are checked in torchvision's own words. */
  protected read(images: Batch, labels: Labels): readonly number[][] {
    const rows = labels as readonly (number | readonly number[])[];
    const twoD = rows.length > 0 && Array.isArray(rows[0]);
    if (!twoD && this.numClasses === null) {
      throw new RuntimeError(
        "num_classes must be passed if the labels are index-based (1D)");
    }
    if (images.length !== rows.length) {
      throw new RuntimeError(
        "The batch size of the image or video does not match the batch size " +
        `of the labels: ${images.length} != ${rows.length}.`);
    }
    if (twoD) return rows.map((r) => [...(r as readonly number[])]);
    // One-hot first if they came in as indices.
    return rows.map((r) => {
      const hot = new Array<number>(this.numClasses as number).fill(0);
      hot[Math.trunc(r as number)] = 1;
      return hot;
    });
  }

  /** **`lam` weights the row itself and `1-lam` its partner** — the way round that
   *  matters, and the way round torchvision has it. */
  protected mixLabels(labels: readonly number[][], lam: number): number[][] {
    return labels.map((row, i) => {
      const partner = labels[(i - 1 + labels.length) % labels.length] as number[];
      return row.map((v, k) => (partner[k] as number) * (1 - lam) + v * lam);
    });
  }

  protected partnerOf(images: Batch, i: number): Image {
    return images[(i - 1 + images.length) % images.length] as Image;
  }
}

/**
 * Blend each picture with the one before it, and their labels by the same weight.
 *
 * <https://arxiv.org/abs/1710.09412>. The whole transform is one weighted average,
 * which is what makes it worth having: no crop, no resample, nothing to get subtly
 * wrong, and it still moves a classifier's calibration.
 */
export class MixUp extends MixBase {
  apply(images: Batch, labels: Labels): [Image[], number[][]] {
    const hot = this.read(images, labels);
    const lam = nextBeta(this.alpha);
    const mixed = images.map((img, i) => {
      const partner = this.partnerOf(images, i);
      const out = img.data.map((v, k) => (partner.data[k] as number) * (1 - lam) + v * lam);
      return image(out, img.height, img.width, img.channels, img.isByte);
    });
    return [mixed, this.mixLabels(hot, lam)];
  }

  describe(): string {
    return `MixUp(alpha=${pyFloat(this.alpha)}, num_classes=${this.numClasses})`;
  }
}

/**
 * Paste a rectangle of the previous picture into this one, and mix the labels by
 * **the area actually pasted** rather than by the weight that was drawn.
 *
 * <https://arxiv.org/abs/1905.04899>. That adjustment is the part worth pointing at:
 * the box is centred on a random point and clipped at the edges, so a box near a
 * corner loses half its area, and a label mixed by the drawn weight would then claim
 * more of the other class than the picture contains.
 */
export class CutMix extends MixBase {
  apply(images: Batch, labels: Labels): [Image[], number[][]] {
    const hot = this.read(images, labels);
    const lam = nextBeta(this.alpha);
    const first = images[0] as Image;
    const h = first.height;
    const w = first.width;
    const centreX = nextInt(w);
    const centreY = nextInt(h);
    const half = 0.5 * Math.sqrt(1 - lam);
    const halfW = Math.trunc(half * w);
    const halfH = Math.trunc(half * h);
    const x1 = Math.max(centreX - halfW, 0);
    const y1 = Math.max(centreY - halfH, 0);
    const x2 = Math.min(centreX + halfW, w);
    const y2 = Math.min(centreY + halfH, h);
    const pasted = images.map((img, i) => {
      const partner = this.partnerOf(images, i);
      const out = Float64Array.from(img.data);
      for (let y = y1; y < y2; y++) {
        for (let x = x1; x < x2; x++) {
          for (let c = 0; c < img.channels; c++) {
            const at = (y * img.width + x) * img.channels + c;
            out[at] = partner.data[at] as number;
          }
        }
      }
      return image(out, img.height, img.width, img.channels, img.isByte);
    });
    const byArea = 1 - ((x2 - x1) * (y2 - y1)) / (w * h);
    return [pasted, this.mixLabels(hot, byArea)];
  }

  describe(): string {
    return `CutMix(alpha=${pyFloat(this.alpha)}, num_classes=${this.numClasses})`;
  }
}
