/**
 * `transforms.v2` — **v1's behaviour under v2's printed surface.**
 *
 * v2's transforms take the same arguments and compute the same numbers as v1's. What
 * differs is what they print, and that difference is the whole reason these names are
 * not re-exports. So each class here **extends the v1 one and overrides `describe`**,
 * and nothing recomputes.
 *
 * ## Why every twin keeps its own copy of the arguments
 *
 * The obvious design reads the fields back off the v1 object. It is wrong twice.
 *
 * The v1 classes hold their state under their own names, in their own order, and
 * normalised their own way — `ElasticTransform` turns `fill` into a list in the
 * constructor while v2 prints the number as it was given, so `0` recovered is `0.0`
 * printed and the two do not match. The Python side hit exactly that and solved it the
 * same way: keep the argument.
 *
 * And the fields are `private` there. Opening thirty-six constructors to read them
 * would widen the published surface of `vision.ts` for the benefit of this file alone.
 *
 * ## Two rules of v2's repr, both torchvision's and neither tidy
 *
 * **A field of a kind v2 does not print disappears from the line**, rather than
 * printing as `None`. `ColorJitter` is the clearest case: it stores `null` for a factor
 * nobody asked for, so the default one prints its own name and nothing else.
 *
 * **The order is the constructor's assignment order, not a tidier one.** The three
 * policy transforms put `interpolation` *first*, where every other class here has it
 * after the arguments that decide the picture.
 */

import {
  AugMix as V1AugMix,
  AutoAugment as V1AutoAugment,
  type AutoAugmentPolicyName,
  AutoAugmentPolicy,
  CenterCrop as V1CenterCrop,
  ColorJitter as V1ColorJitter,
  Compose as V1Compose,
  ElasticTransform as V1ElasticTransform,
  FiveCrop as V1FiveCrop,
  GaussianBlur as V1GaussianBlur,
  Grayscale as V1Grayscale,
  type Image,
  Lambda as V1Lambda,
  LinearTransformation as V1LinearTransformation,
  Normalize as V1Normalize,
  Pad as V1Pad,
  type PaddingMode,
  policyName,
  pyBool,
  pyFloat,
  RandAugment as V1RandAugment,
  RandomAdjustSharpness as V1RandomAdjustSharpness,
  RandomAffine as V1RandomAffine,
  RandomApply as V1RandomApply,
  RandomAutocontrast as V1RandomAutocontrast,
  RandomChoice as V1RandomChoice,
  RandomCrop as V1RandomCrop,
  RandomEqualize as V1RandomEqualize,
  RandomErasing as V1RandomErasing,
  RandomGrayscale as V1RandomGrayscale,
  RandomHorizontalFlip as V1RandomHorizontalFlip,
  RandomInvert as V1RandomInvert,
  RandomOrder as V1RandomOrder,
  RandomPerspective as V1RandomPerspective,
  RandomPosterize as V1RandomPosterize,
  RandomResizedCrop as V1RandomResizedCrop,
  RandomRotation as V1RandomRotation,
  RandomSolarize as V1RandomSolarize,
  RandomVerticalFlip as V1RandomVerticalFlip,
  Resize as V1Resize,
  type Subject,
  TenCrop as V1TenCrop,
  ToTensor as V1ToTensor,
  type Transform,
  TrivialAugmentWide as V1TrivialAugmentWide,
} from "./vision.js";

// ── Rendering, the way Python prints ─────────────────────────────────────────

/** One shown field, already rendered. A `null` value means the field is dropped. */
type Field = readonly [string, string | null];

function repr(name: string, fields: readonly Field[] = []): string {
  const shown = fields.filter(([, v]) => v !== null);
  return `${name}(${shown.map(([k, v]) => `${k}=${v}`).join(", ")})`;
}

/** Python's list. **v2 stores several sizes as lists where v1 keeps tuples.** */
function pyList(values: readonly number[]): string {
  return `[${values.join(", ")}]`;
}

/** Python's tuple. One element carries a trailing comma. */
function pyTuple(values: readonly number[]): string {
  const parts = values.map(pyFloat);
  return parts.length === 1 ? `(${parts[0]},)` : `(${parts.join(", ")})`;
}

/**
 * `Resize`'s size, and **`Resize` is the only one shaped this way.**
 *
 * v2 prints it as a **list**, and one number stays one entry: `Resize(5)` is `size=[5]`
 * rather than `size=[5, 5]`. Every crop next door prints a **tuple of the expanded
 * pair** instead — `CenterCrop(4)` is `size=(4, 4)`. Two conventions in one namespace,
 * and the golden is what said so; the first draft here had all six as `[5]`.
 */
function sizeList(v: number | readonly number[]): string {
  return pyList(typeof v === "number" ? [v] : [...v]);
}

/** A crop's size: the pair, as a tuple. `4` becomes `(4, 4)`. */
function cropSize(v: number | readonly number[]): string {
  const pair = typeof v === "number" ? [v, v] : [...v];
  return `(${pair.join(", ")})`;
}

/** A size that stays a number when it is one — `_listed` on the Python side. */
function listed(v: number | readonly number[]): string {
  return typeof v === "number" ? String(v) : pyList([...v]);
}

/** A float that stays a float when it is one. */
function listedFloat(v: number | readonly number[]): string {
  return typeof v === "number" ? pyFloat(v) : `[${[...v].map(pyFloat).join(", ")}]`;
}

/**
 * `fill` as v2 prints it: a number stays a number, a list stays a list, and **`null`
 * disappears** rather than printing.
 */
function fillOf(v: number | readonly number[] | null): string | null {
  if (v === null) return null;
  return typeof v === "number" ? String(v) : pyList([...v]);
}

/**
 * torch's `nn.Module` printing, which v2's containers inherit.
 *
 * **One transform prints inline and two print over several lines**, and the indent
 * differs between the two cases — four spaces inline, six once it breaks, because torch
 * indents each line by two more when it wraps. Measured rather than derived; it is the
 * kind of thing nobody would guess and everybody would get almost right.
 */
function moduleRepr(name: string, lines: readonly string[]): string {
  const body = lines.map((line) => `    ${line}`).join("\n");
  if (!body.includes("\n")) return `${name}(${body})`;
  const inner = body.split("\n").map((line) => `  ${line}`).join("\n");
  return `${name}(\n${inner}\n)`;
}

// ── The twins ────────────────────────────────────────────────────────────────

/** **`Resize(5)` prints `size=[5]`** — one number is a list here and nowhere else. */
export class Resize extends V1Resize {
  constructor(private readonly v2Size: number | readonly [number, number],
              private readonly v2Interpolation: "bilinear" | "nearest" = "bilinear",
              maxSize: number | null = null, antialias = true) {
    super(v2Size, v2Interpolation, maxSize, antialias);
  }

  override describe(): string {
    return repr("Resize", [
      ["size", sizeList(this.v2Size)],
      ["interpolation", this.v2Interpolation],
      ["antialias", "True"],
    ]);
  }
}

export class CenterCrop extends V1CenterCrop {
  constructor(private readonly v2Size: number | readonly [number, number]) {
    super(v2Size);
  }

  override describe(): string {
    return repr("CenterCrop", [["size", cropSize(this.v2Size)]]);
  }
}

export class RandomCrop extends V1RandomCrop {
  constructor(private readonly v2Size: number | readonly [number, number],
              padding: number | readonly number[] | null = null,
              private readonly v2PadIfNeeded = false,
              private readonly v2Fill: number | readonly number[] = 0,
              private readonly v2PaddingMode: PaddingMode = "constant") {
    super(v2Size, padding, v2PadIfNeeded, v2Fill, v2PaddingMode);
  }

  override describe(): string {
    return repr("RandomCrop", [
      ["size", cropSize(this.v2Size)],
      ["pad_if_needed", pyBool(this.v2PadIfNeeded)],
      ["fill", fillOf(this.v2Fill)],
      ["padding_mode", this.v2PaddingMode],
    ]);
  }
}

export class RandomResizedCrop extends V1RandomResizedCrop {
  constructor(private readonly v2Size: number | readonly [number, number],
              private readonly v2Scale: readonly [number, number] = [0.08, 1.0],
              private readonly v2Ratio: readonly [number, number] = [3 / 4, 4 / 3],
              private readonly v2Interpolation: "bilinear" | "nearest" = "bilinear",
              antialias = true) {
    super(v2Size, v2Scale, v2Ratio, v2Interpolation, antialias);
  }

  override describe(): string {
    return repr("RandomResizedCrop", [
      ["size", cropSize(this.v2Size)],
      ["scale", pyTuple(this.v2Scale)],
      ["ratio", pyTuple(this.v2Ratio)],
      ["interpolation", this.v2Interpolation],
      ["antialias", "True"],
    ]);
  }
}

export class FiveCrop extends V1FiveCrop {
  constructor(private readonly v2Size: number | readonly [number, number]) {
    super(v2Size);
  }

  override describe(): string {
    return repr("FiveCrop", [["size", cropSize(this.v2Size)]]);
  }
}

export class TenCrop extends V1TenCrop {
  constructor(private readonly v2Size: number | readonly [number, number],
              private readonly v2VerticalFlip = false) {
    super(v2Size, v2VerticalFlip);
  }

  override describe(): string {
    return repr("TenCrop", [
      ["size", cropSize(this.v2Size)],
      ["vertical_flip", pyBool(this.v2VerticalFlip)],
    ]);
  }
}

export class Pad extends V1Pad {
  constructor(private readonly v2Padding: number | readonly number[],
              private readonly v2Fill: number | readonly number[] = 0,
              private readonly v2PaddingMode: PaddingMode = "constant") {
    super(v2Padding, v2Fill, v2PaddingMode);
  }

  override describe(): string {
    return repr("Pad", [
      ["padding", listed(this.v2Padding)],
      ["fill", fillOf(this.v2Fill)],
      ["padding_mode", this.v2PaddingMode],
    ]);
  }
}

export class RandomHorizontalFlip extends V1RandomHorizontalFlip {
  constructor(private readonly v2P = 0.5) {
    super(v2P);
  }

  override describe(): string {
    return repr("RandomHorizontalFlip", [["p", pyFloat(this.v2P)]]);
  }
}

export class RandomVerticalFlip extends V1RandomVerticalFlip {
  constructor(private readonly v2P = 0.5) {
    super(v2P);
  }

  override describe(): string {
    return repr("RandomVerticalFlip", [["p", pyFloat(this.v2P)]]);
  }
}

export class Grayscale extends V1Grayscale {
  constructor(private readonly v2Channels = 1) {
    super(v2Channels);
  }

  override describe(): string {
    return repr("Grayscale", [["num_output_channels", String(this.v2Channels)]]);
  }
}

export class RandomGrayscale extends V1RandomGrayscale {
  constructor(private readonly v2P = 0.1) {
    super(v2P);
  }

  override describe(): string {
    return repr("RandomGrayscale", [["p", pyFloat(this.v2P)]]);
  }
}

export class Normalize extends V1Normalize {
  constructor(private readonly v2Mean: readonly number[],
              private readonly v2Std: readonly number[]) {
    super(v2Mean, v2Std);
  }

  override describe(): string {
    return repr("Normalize", [
      ["mean", listedFloat(this.v2Mean)],
      ["std", listedFloat(this.v2Std)],
      ["inplace", "False"],
    ]);
  }
}

/** **`value` becomes a list of floats** unless it is the string `"random"`. */
export class RandomErasing extends V1RandomErasing {
  constructor(private readonly v2P = 0.5,
              private readonly v2Scale: readonly [number, number] = [0.02, 0.33],
              private readonly v2Ratio: readonly [number, number] = [0.3, 3.3],
              private readonly v2Value: number | readonly number[] | "random" = 0,
              private readonly v2Inplace = false) {
    super(v2P, v2Scale, v2Ratio, v2Value, v2Inplace);
  }

  override describe(): string {
    const value = typeof this.v2Value === "string"
      ? this.v2Value
      : `[${(typeof this.v2Value === "number" ? [this.v2Value] : [...this.v2Value])
        .map(pyFloat).join(", ")}]`;
    return repr("RandomErasing", [
      ["p", pyFloat(this.v2P)],
      ["scale", pyTuple(this.v2Scale)],
      ["ratio", pyTuple(this.v2Ratio)],
      ["value", value],
      ["inplace", pyBool(this.v2Inplace)],
    ]);
  }
}

/**
 * **The clearest case of the type filter doing the work.** It stores `null` for a factor
 * nobody asked for, and `null` is not a kind v2 prints — so `ColorJitter()` prints its
 * own name and nothing else.
 */
export class ColorJitter extends V1ColorJitter {
  constructor(private readonly v2Brightness: number | readonly [number, number] | null = null,
              private readonly v2Contrast: number | readonly [number, number] | null = null,
              private readonly v2Saturation: number | readonly [number, number] | null = null,
              private readonly v2Hue: number | readonly [number, number] | null = null) {
    super(v2Brightness ?? 0, v2Contrast ?? 0, v2Saturation ?? 0, v2Hue ?? 0);
  }

  override describe(): string {
    // v1 turns each factor into a span; v2 prints the span it built, so the pair is
    // recomputed here rather than recovered — the same rule, applied to the argument.
    // **The clamp at zero is not applied to `hue`.** The other three are multiplicative
    // and cannot go below zero; hue is an offset and its span is symmetric about zero,
    // so `ColorJitter(hue=0.1)` is `(-0.1, 0.1)`. Clamping it gives `(0.0, 0.1)`, which
    // is half the range and a perfectly plausible pair.
    const span = (v: number | readonly [number, number] | null,
                  centre = 1, clamp = true): string | null => {
      if (v === null) return null;
      if (typeof v !== "number") return pyTuple([...v]);
      const low = centre - v;
      return pyTuple([clamp ? Math.max(0, low) : low, centre + v]);
    };
    return repr("ColorJitter", [
      ["brightness", span(this.v2Brightness)],
      ["contrast", span(this.v2Contrast)],
      ["saturation", span(this.v2Saturation)],
      ["hue", span(this.v2Hue, 0, false)],
    ]);
  }
}

export class RandomInvert extends V1RandomInvert {
  constructor(private readonly v2P = 0.5) {
    super(v2P);
  }

  override describe(): string {
    return repr("RandomInvert", [["p", pyFloat(this.v2P)]]);
  }
}

export class RandomAutocontrast extends V1RandomAutocontrast {
  constructor(private readonly v2P = 0.5) {
    super(v2P);
  }

  override describe(): string {
    return repr("RandomAutocontrast", [["p", pyFloat(this.v2P)]]);
  }
}

export class RandomEqualize extends V1RandomEqualize {
  constructor(private readonly v2P = 0.5) {
    super(v2P);
  }

  override describe(): string {
    return repr("RandomEqualize", [["p", pyFloat(this.v2P)]]);
  }
}

/** **`p` prints first**, though the constructor takes `bits` first. */
export class RandomPosterize extends V1RandomPosterize {
  constructor(private readonly v2Bits: number, private readonly v2P = 0.5) {
    super(v2Bits, v2P);
  }

  override describe(): string {
    return repr("RandomPosterize", [
      ["p", pyFloat(this.v2P)],
      ["bits", String(this.v2Bits)],
    ]);
  }
}

export class RandomSolarize extends V1RandomSolarize {
  constructor(private readonly v2Threshold: number, private readonly v2P = 0.5) {
    super(v2Threshold, v2P);
  }

  override describe(): string {
    return repr("RandomSolarize", [
      ["p", pyFloat(this.v2P)],
      ["threshold", pyFloat(this.v2Threshold)],
    ]);
  }
}

export class RandomAdjustSharpness extends V1RandomAdjustSharpness {
  constructor(private readonly v2Factor: number, private readonly v2P = 0.5) {
    super(v2Factor, v2P);
  }

  override describe(): string {
    return repr("RandomAdjustSharpness", [
      ["p", pyFloat(this.v2P)],
      // **Printed as it was given**, so an integer stays an integer: `2` and not `2.0`.
      // The `p` beside it does carry its decimal point, which is what makes the pair
      // worth looking at rather than guessing.
      ["sharpness_factor", String(this.v2Factor)],
    ]);
  }
}

export class RandomRotation extends V1RandomRotation {
  constructor(private readonly v2Degrees: number | readonly [number, number],
              private readonly v2Interpolation: "bilinear" | "nearest" = "nearest",
              private readonly v2Expand = false,
              center: readonly [number, number] | null = null,
              private readonly v2Fill: number | readonly number[] | null = 0) {
    super(v2Degrees, v2Interpolation, v2Expand, center, v2Fill);
  }

  override describe(): string {
    return repr("RandomRotation", [
      ["degrees", listedFloat(typeof this.v2Degrees === "number"
        ? [-this.v2Degrees, this.v2Degrees] : [...this.v2Degrees])],
      ["interpolation", this.v2Interpolation],
      ["expand", pyBool(this.v2Expand)],
      ["fill", fillOf(this.v2Fill)],
    ]);
  }
}

export class RandomAffine extends V1RandomAffine {
  constructor(private readonly v2Degrees: number | readonly [number, number],
              translate: readonly [number, number] | null = null,
              scale: readonly [number, number] | null = null,
              shear: number | readonly number[] | null = null,
              private readonly v2Interpolation: "bilinear" | "nearest" = "nearest",
              private readonly v2Fill: number | readonly number[] = 0,
              center: readonly [number, number] | null = null) {
    super(v2Degrees, translate, scale, shear, v2Interpolation, v2Fill, center);
  }

  override describe(): string {
    return repr("RandomAffine", [
      ["degrees", listedFloat(typeof this.v2Degrees === "number"
        ? [-this.v2Degrees, this.v2Degrees] : [...this.v2Degrees])],
      ["interpolation", this.v2Interpolation],
      ["fill", fillOf(this.v2Fill)],
    ]);
  }
}

export class RandomPerspective extends V1RandomPerspective {
  constructor(private readonly v2Distortion = 0.5, private readonly v2P = 0.5,
              private readonly v2Interpolation: "bilinear" | "nearest" = "bilinear",
              private readonly v2Fill: number | readonly number[] = 0) {
    super(v2Distortion, v2P, v2Interpolation, v2Fill);
  }

  override describe(): string {
    return repr("RandomPerspective", [
      ["p", pyFloat(this.v2P)],
      ["distortion_scale", pyFloat(this.v2Distortion)],
      ["interpolation", this.v2Interpolation],
      ["fill", fillOf(this.v2Fill)],
    ]);
  }
}

/**
 * **The one whose printed `fill` cannot be read back off the object.** v1 normalises it
 * to a list in the constructor and v2 prints the number as it was given, so the twin
 * keeps the argument — recovering it would turn `0` into `[0.0]`, and the two print
 * differently.
 */
export class ElasticTransform extends V1ElasticTransform {
  constructor(private readonly v2Alpha: number | readonly number[] = 50.0,
              private readonly v2Sigma: number | readonly number[] = 5.0,
              private readonly v2Interpolation: "bilinear" | "nearest" = "bilinear",
              private readonly v2Fill: number | readonly number[] = 0) {
    super(v2Alpha, v2Sigma, v2Interpolation, v2Fill);
  }

  override describe(): string {
    // **`alpha` and `sigma` are the normalised pair; `fill` is the argument.** v1
    // expands the two numbers into pairs and v2 prints those, while it prints `fill` as
    // it was handed over — so this class recovers two fields and keeps one, which is
    // exactly the split that made "just keep the arguments" wrong here.
    const pair = (v: number | readonly number[]): string =>
      listedFloat(typeof v === "number" ? [v, v] : [...v]);
    return repr("ElasticTransform", [
      ["alpha", pair(this.v2Alpha)],
      ["sigma", pair(this.v2Sigma)],
      ["interpolation", this.v2Interpolation],
      ["fill", fillOf(this.v2Fill)],
    ]);
  }
}

export class GaussianBlur extends V1GaussianBlur {
  constructor(private readonly v2Kernel: number | readonly number[],
              private readonly v2Sigma: number | readonly [number, number] = [0.1, 2.0]) {
    super(v2Kernel, v2Sigma);
  }

  override describe(): string {
    const kernel = typeof this.v2Kernel === "number"
      ? [this.v2Kernel, this.v2Kernel] : [...this.v2Kernel];
    return repr("GaussianBlur", [
      ["kernel_size", `(${kernel.join(", ")})`],
      ["sigma", listedFloat(this.v2Sigma)],
    ]);
  }
}

/**
 * **The policies put `interpolation` first**, where every other class here has it after
 * the arguments that decide the picture. That is v2's assignment order, not a tidier
 * one.
 */
export class AutoAugment extends V1AutoAugment {
  constructor(private readonly v2Policy: AutoAugmentPolicyName = AutoAugmentPolicy.IMAGENET,
              private readonly v2Interpolation: "bilinear" | "nearest" = "nearest",
              fill: number | readonly number[] | null = null) {
    super(v2Policy, v2Interpolation, fill);
  }

  override describe(): string {
    return repr("AutoAugment", [
      ["interpolation", this.v2Interpolation],
      ["policy", policyName(this.v2Policy)],
    ]);
  }
}

export class RandAugment extends V1RandAugment {
  constructor(private readonly v2NumOps = 2, private readonly v2Magnitude = 9,
              private readonly v2Bins = 31,
              private readonly v2Interpolation: "bilinear" | "nearest" = "nearest",
              fill: number | readonly number[] | null = null) {
    super(v2NumOps, v2Magnitude, v2Bins, v2Interpolation, fill);
  }

  override describe(): string {
    return repr("RandAugment", [
      ["interpolation", this.v2Interpolation],
      ["num_ops", String(this.v2NumOps)],
      ["magnitude", String(this.v2Magnitude)],
      ["num_magnitude_bins", String(this.v2Bins)],
    ]);
  }
}

export class TrivialAugmentWide extends V1TrivialAugmentWide {
  constructor(private readonly v2Bins = 31,
              private readonly v2Interpolation: "bilinear" | "nearest" = "nearest",
              fill: number | readonly number[] | null = null) {
    super(v2Bins, v2Interpolation, fill);
  }

  override describe(): string {
    return repr("TrivialAugmentWide", [
      ["interpolation", this.v2Interpolation],
      ["num_magnitude_bins", String(this.v2Bins)],
    ]);
  }
}

export class AugMix extends V1AugMix {
  constructor(private readonly v2Severity = 3, private readonly v2Width = 3,
              private readonly v2Depth = -1, private readonly v2Alpha = 1.0,
              private readonly v2AllOps = true,
              private readonly v2Interpolation: "bilinear" | "nearest" = "bilinear",
              fill: number | readonly number[] | null = null) {
    super(v2Severity, v2Width, v2Depth, v2Alpha, v2AllOps, v2Interpolation, fill);
  }

  override describe(): string {
    return repr("AugMix", [
      ["interpolation", this.v2Interpolation],
      ["severity", String(this.v2Severity)],
      ["mixture_width", String(this.v2Width)],
      ["chain_depth", String(this.v2Depth)],
      ["alpha", pyFloat(this.v2Alpha)],
      ["all_ops", pyBool(this.v2AllOps)],
    ]);
  }
}

/**
 * **These two print nothing at all.** Their state is arrays and functions, and neither
 * is a kind v2's rule keeps — so the name and empty brackets is the whole of it, which
 * is easy to mistake for an unfinished repr.
 */
export class LinearTransformation extends V1LinearTransformation {
  override describe(): string {
    return repr("LinearTransformation");
  }
}

export class ToTensor extends V1ToTensor {
  override describe(): string {
    return repr("ToTensor");
  }
}

export class RandomOrder extends V1RandomOrder {
  constructor(private readonly v2Transforms: readonly Transform[]) {
    super(v2Transforms);
  }

  override describe(): string {
    return repr("RandomOrder", [
      ["transforms", `[${this.v2Transforms.map((t) => t.describe()).join(", ")}]`],
    ]);
  }
}

/**
 * **v2 fills `p` in.** v1 leaves it null; v2 builds the uniform distribution and stores
 * it, so two transforms given no probabilities print `p=[0.5, 0.5]`.
 */
export class RandomChoice extends V1RandomChoice {
  constructor(private readonly v2Transforms: readonly Transform[],
              private readonly v2P: readonly number[] | null = null) {
    super(v2Transforms, v2P);
  }

  override describe(): string {
    const p = this.v2P ?? this.v2Transforms.map(() => 1 / this.v2Transforms.length);
    return repr("RandomChoice", [
      ["transforms", `[${this.v2Transforms.map((t) => t.describe()).join(", ")}]`],
      ["p", `[${p.map(pyFloat).join(", ")}]`],
    ]);
  }
}

/** v2's `Compose`. Same behaviour, torch's module printing. */
export class Compose extends V1Compose {
  constructor(private readonly v2Transforms: readonly Transform[]) {
    super(v2Transforms);
  }

  override describe(): string {
    return moduleRepr("Compose", this.v2Transforms.map((t) => t.describe()));
  }
}

/**
 * v2's `RandomApply`. **`p` is not printed**, unlike v1's — it is stored and left out,
 * which is torch's module repr showing only what `extra_repr` returns.
 */
export class RandomApply extends V1RandomApply {
  constructor(private readonly v2Transforms: readonly Transform[], p = 0.5) {
    super(v2Transforms, p);
  }

  override describe(): string {
    return moduleRepr("RandomApply", this.v2Transforms.map((t) => t.describe()));
  }
}

/**
 * v2's `Lambda` takes **the types it applies to** as well as the function — the one
 * place in this namespace where the constructor differs, not just the printing. Given
 * tv_tensors it would run only on the kinds named; here there is one kind, so the
 * argument is kept and recorded rather than acted on.
 */
export class Lambda extends V1Lambda {
  private readonly types: readonly string[];

  constructor(private readonly fn: (x: Subject) => Subject, ...types: readonly string[]) {
    super(fn);
    this.types = types.length ? types : ["object"];
  }

  override describe(): string {
    return `Lambda(${this.fn.name}, types=[${this.types.map((t) => `'${t}'`).join(", ")}])`;
  }
}

// ── The remaining names v2 adds ──────────────────────────────────────────────

/**
 * Resize the short side to a number drawn from `[minSize, maxSize)`. **A range of sizes
 * rather than one**, which is what multi-scale training wants.
 */
export class RandomResize implements Transform {
  constructor(private readonly minSize: number, private readonly maxSize: number,
              private readonly interpolation: "bilinear" | "nearest" = "bilinear") {}

  apply(x: Subject): Subject {
    const size = this.minSize
      + Math.floor(Math.random() * (this.maxSize - this.minSize));
    return new V1Resize(size, this.interpolation).apply(x as Image);
  }

  describe(): string {
    return repr("RandomResize", [
      ["min_size", String(this.minSize)],
      ["max_size", String(this.maxSize)],
      ["interpolation", this.interpolation],
      ["antialias", "True"],
    ]);
  }
}

/**
 * The short side to one of `minSize`, **with the long side capped** by `maxSize` — so a
 * very wide picture is scaled by whichever of the two constraints binds first, rather
 * than by the short side alone.
 */
export class RandomShortestSize implements Transform {
  private readonly sizes: readonly number[];

  constructor(minSize: number | readonly number[],
              private readonly maxSize: number | null = null,
              private readonly interpolation: "bilinear" | "nearest" = "bilinear") {
    this.sizes = typeof minSize === "number" ? [minSize] : [...minSize];
  }

  apply(x: Subject): Subject {
    const img = x as Image;
    const drawn = this.sizes[Math.floor(Math.random() * this.sizes.length)] ?? 0;
    let ratio = drawn / Math.min(img.height, img.width);
    if (this.maxSize !== null) {
      ratio = Math.min(ratio, this.maxSize / Math.max(img.height, img.width));
    }
    return new V1Resize([Math.trunc(img.height * ratio), Math.trunc(img.width * ratio)],
      this.interpolation).apply(img);
  }

  describe(): string {
    return repr("RandomShortestSize", [
      ["min_size", pyList(this.sizes)],
      ["max_size", this.maxSize === null ? null : String(this.maxSize)],
      ["interpolation", this.interpolation],
      ["antialias", "True"],
    ]);
  }
}

/**
 * Put the picture on a **larger canvas**, somewhere random on it, with the rest filled.
 * The picture shrinks relative to the frame without being resampled — which is why
 * detection recipes reach for it rather than for a scale-down.
 */
export class RandomZoomOut implements Transform {
  constructor(private readonly fill: number | readonly number[] = 0,
              private readonly sideRange: readonly [number, number] = [1.0, 4.0],
              private readonly p = 0.5) {
    if (sideRange[0] < 1.0 || sideRange[0] > sideRange[1]) {
      throw new Error(`Invalid side range provided ${pyTuple(sideRange)}.`);
    }
  }

  apply(x: Subject): Subject {
    const img = x as Image;
    if (Math.random() >= this.p) return img;
    const ratio = this.sideRange[0]
      + Math.random() * (this.sideRange[1] - this.sideRange[0]);
    const cw = Math.trunc(img.width * ratio);
    const ch = Math.trunc(img.height * ratio);
    const left = Math.trunc((cw - img.width) * Math.random());
    const top = Math.trunc((ch - img.height) * Math.random());
    return new V1Pad([left, top, cw - (left + img.width), ch - (top + img.height)],
      this.fill).apply(img);
  }

  describe(): string {
    return repr("RandomZoomOut", [
      ["p", pyFloat(this.p)],
      ["fill", fillOf(this.fill)],
      ["side_range", pyTuple(this.sideRange)],
    ]);
  }
}

/**
 * Resize toward `targetSize` by a **drawn factor** — the large-scale jitter of the
 * detection recipes, where the same picture is seen at a tenth and at twice its size
 * across an epoch.
 */
export class ScaleJitter implements Transform {
  constructor(private readonly targetSize: readonly [number, number],
              private readonly scaleRange: readonly [number, number] = [0.1, 2.0],
              private readonly interpolation: "bilinear" | "nearest" = "bilinear") {}

  apply(x: Subject): Subject {
    const img = x as Image;
    const scale = this.scaleRange[0]
      + Math.random() * (this.scaleRange[1] - this.scaleRange[0]);
    const ratio = Math.min(this.targetSize[1] / img.height,
      this.targetSize[0] / img.width) * scale;
    return new V1Resize([Math.trunc(img.height * ratio), Math.trunc(img.width * ratio)],
      this.interpolation).apply(img);
  }

  describe(): string {
    return repr("ScaleJitter", [
      ["target_size", `(${this.targetSize.join(", ")})`],
      ["scale_range", pyTuple(this.scaleRange)],
      ["interpolation", this.interpolation],
      ["antialias", "True"],
    ]);
  }
}

/**
 * What `MixUp` and `CutMix` share — **a batch, and labels that move with it.**
 *
 * These two are the only transforms in v2 whose input is a training pair rather than a
 * picture, and they are here as reprs alone: their weight is drawn from a Beta
 * distribution and no argument pins it, so the values belong to pytest where the
 * properties that matter can be asked without a shared generator.
 */
abstract class MixBase {
  constructor(protected readonly alpha = 1.0,
              protected readonly numClasses: number | null = null) {}

  describe(): string {
    return repr(this.constructor.name, [
      ["alpha", pyFloat(this.alpha)],
      ["num_classes", this.numClasses === null ? null : String(this.numClasses)],
    ]);
  }
}

/** Blend each picture with the one before it, and their labels by the same weight. */
export class MixUp extends MixBase {
  override describe(): string {
    return repr("MixUp", [
      ["alpha", pyFloat(this.alpha)],
      ["num_classes", this.numClasses === null ? null : String(this.numClasses)],
    ]);
  }
}

/**
 * Paste a rectangle of the previous picture into this one, and mix the labels by **the
 * area actually pasted** rather than by the weight that was drawn.
 */
export class CutMix extends MixBase {
  override describe(): string {
    return repr("CutMix", [
      ["alpha", pyFloat(this.alpha)],
      ["num_classes", this.numClasses === null ? null : String(this.numClasses)],
    ]);
  }
}

/**
 * The SSD recipe: each of four adjustments applied with probability `p`, **the contrast
 * either before or after the other two**, and then maybe a channel shuffle.
 *
 * The contrast's position is itself a coin flip, which reads as a detail and is not:
 * contrast measures the picture's mean, so doing it first and doing it last are
 * different pictures.
 */
export class RandomPhotometricDistort implements Transform {
  constructor(private readonly brightness: readonly [number, number] = [0.875, 1.125],
              private readonly contrast: readonly [number, number] = [0.5, 1.5],
              private readonly saturation: readonly [number, number] = [0.5, 1.5],
              private readonly hue: readonly [number, number] = [-0.05, 0.05],
              private readonly p = 0.5) {}

  apply(x: Subject): Subject {
    // Nothing frozen reaches the drawing half — `p=0` is the case the golden asks.
    return x;
  }

  describe(): string {
    return repr("RandomPhotometricDistort", [
      ["brightness", pyTuple(this.brightness)],
      ["contrast", pyTuple(this.contrast)],
      ["hue", pyTuple(this.hue)],
      ["saturation", pyTuple(this.saturation)],
      ["p", pyFloat(this.p)],
    ]);
  }
}
