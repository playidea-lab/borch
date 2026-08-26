/**
 * Transforms shaped like `torchvision.transforms`.
 *
 * ## Why this is inside a tensor library
 *
 * Running training requires a place where images become tensors, and there
 * are things that go quietly wrong at that place — the comment on
 * `ToTensor` below is one of them. It uses **the same rules** as the
 * repository's Python side (`borchvision.py`). If the rules diverge, the
 * same data becomes different values per library, and that is caught only
 * by comparing values.
 *
 * ## An image is an array, not a tensor
 *
 * `RandomHorizontalFlip` and `RandomCrop` take an `(H, W, C)` array. What
 * arrives here in torchvision is a PIL image, and having no PIL, an array
 * stands in its place. Making a tensor per image makes one GPU buffer per
 * image, which looks like it works right up until it collapses on memory.
 */

import { RuntimeError } from "./errors.js";
import { Tensor } from "./tensor.js";

/**
 * An image laid out as `(H, W, C)`. The values are uint8 or float.
 */
export interface Image {
  readonly data: Float64Array;
  readonly height: number;
  readonly width: number;
  readonly channels: number;
  /**
   * If uint8, `ToTensor` divides by 255.
   */
  readonly isByte: boolean;
}

export function image(
  data: ArrayLike<number>,
  height: number,
  width: number,
  channels: number,
  isByte: boolean,
): Image {
  return { data: Float64Array.from(data), height, width, channels, isByte };
}

/**
 * What a transform can be handed, and what it can hand back.
 *
 * The array is `FiveCrop` and `TenCrop`, which turn one picture into several.
 * torchvision has the same shape and says so in its own documentation: the
 * tuple has to be taken apart by a `Lambda` before anything else can read it.
 * Every other transform refuses an array loudly rather than picking the first.
 */
export type Subject = Image | Tensor | readonly Image[];

export interface Transform {
  apply(x: Subject): Subject;
  describe(): string;
}

/**
 * Transforms in sequence. `repr` is several lines with the inside indented,
 * so that shape is matched too.
 */
export class Compose implements Transform {
  constructor(protected readonly transforms: readonly Transform[]) {}

  apply(x: Subject): Subject {
    let cur = x;
    for (const t of this.transforms) cur = t.apply(cur);
    return cur;
  }

  describe(): string {
    const inner = this.transforms.map((t) => `\n    ${t.describe()}`).join("");
    return `Compose(${inner}\n)`;
  }
}

/**
 * Wraps a function so it can stand inside a `Compose`.
 *
 * It looks like nothing, and it is the only place a learner's own function can
 * enter the pipeline — without it a one-line `x * 2` has to become a class.
 */
export class Lambda implements Transform {
  constructor(private readonly lambd: (x: Subject) => Subject) {
    if (typeof lambd !== "function") {
      throw new RuntimeError(
        "Lambda takes a function — it received " + typeof lambd + ".\n" +
        "(torch: Argument lambd should be callable)");
    }
  }

  apply(x: Subject): Subject {
    return this.lambd(x);
  }

  describe(): string {
    // **The function is not printed.** torchvision prints an empty pair of
    // brackets here, and a function's source would make the same pipeline print
    // differently depending on how it was written.
    return "Lambda()";
  }
}

/** What the three below share — the list, and how the list prints. */
abstract class RandomTransforms implements Transform {
  // **The name is handed in.** Written as `this.constructor.name` or
  // `new.target.name`, `repr` changes quietly the day a bundler shortens the class name —
  // and this project treats `repr` as a specification, so that is the specification
  // changing without a sound.
  constructor(
    private readonly who: string,
    protected readonly transforms: readonly Transform[],
  ) {
    if (!Array.isArray(transforms)) {
      throw new RuntimeError(
        `${who} takes a sequence of transforms.\n` +
        "(torch: Argument transforms should be a sequence)");
    }
  }

  abstract apply(x: Subject): Subject;

  protected inner(): string {
    return this.transforms.map((t) => `\n    ${t.describe()}`).join("");
  }

  describe(): string {
    return `${this.who}(${this.inner()}\n)`;
  }
}

/**
 * Applies the whole list, or none of it, with probability `p`.
 *
 * **All of them or none** — not each with its own draw. One draw decides the
 * lot, which is what makes it different from putting a `p` on each transform.
 */
export class RandomApply extends RandomTransforms {
  constructor(transforms: readonly Transform[], protected readonly p = 0.5) {
    super("RandomApply", transforms);
  }

  apply(x: Subject): Subject {
    if (nextFloat() >= this.p) return x;
    let cur = x;
    for (const t of this.transforms) cur = t.apply(cur);
    return cur;
  }

  override describe(): string {
    return `RandomApply(\n    p=${this.p}${this.inner()}\n)`;
  }
}

/** Draws **one** of the list and applies it. `p` weights the draw. */
export class RandomChoice extends RandomTransforms {
  constructor(
    transforms: readonly Transform[],
    protected readonly p: readonly number[] | null = null,
  ) {
    super("RandomChoice", transforms);
  }

  apply(x: Subject): Subject {
    const t = this.transforms[this.draw()];
    if (t === undefined) {
      throw new RuntimeError("RandomChoice was given an empty list of transforms.");
    }
    return t.apply(x);
  }

  private draw(): number {
    const n = this.transforms.length;
    if (this.p === null) return nextInt(n);
    // **The weights are normalised here.** torch's `random.choices` takes
    // relative weights; handing them straight to a cumulative draw would make
    // `p=[1, 1]` mean something other than "evenly".
    let total = 0;
    for (const w of this.p) total += w;
    let r = nextFloat() * total;
    for (let i = 0; i < n; i++) {
      r -= this.p[i] ?? 0;
      if (r < 0) return i;
    }
    return n - 1;
  }

  override describe(): string {
    const p = this.p === null ? "None" : tuple(this.p);
    return `RandomChoice(${this.inner()}\n)(p=${p})`;
  }
}

/** Applies every one of them, in a shuffled order. */
export class RandomOrder extends RandomTransforms {
  constructor(transforms: readonly Transform[]) {
    super("RandomOrder", transforms);
  }

  apply(x: Subject): Subject {
    const order = this.transforms.map((_, i) => i);
    for (let i = order.length - 1; i > 0; i--) {
      const j = nextInt(i + 1);
      const a = order[i] ?? 0, b = order[j] ?? 0;
      order[i] = b; order[j] = a;
    }
    let cur = x;
    for (const i of order) {
      const t = this.transforms[i];
      if (t !== undefined) cur = t.apply(cur);
    }
    return cur;
  }
}

/**
 * `(H,W,C)` or `(H,W)` → a `(C,H,W)` tensor.
 *
 * **It divides by 255 only when the input is uint8** — that is
 * torchvision's rule. Pass a float array and it is carried over undivided.
 * Miss this distinction and data that is already in [0,1] gets divided once
 * more and comes out 255× darker, with **no exception raised and only the
 * training failing.**
 */
export class ToTensor implements Transform {
  apply(x: Subject): Tensor {
    if (x instanceof Tensor) return x;
    // **Several pictures arriving is caught here.** Since `Transform` widened to accept
    // an array, `Compose([new FiveCrop(3), new ToTensor()])` passes the type check.
    // Unblocked, it destructures the array, `data` and `height` all come out `undefined`,
    // and it does blow up — as `shape [,,] does not match 0 elements`, an accident with
    // no record of what was done wrong (measured). torchvision says to use a `Lambda` at
    // the same place.
    const { data, height, width, channels, isByte } = asImage(x, "ToTensor");
    const out = new Float32Array(channels * height * width);
    const scale = isByte ? 1 / 255 : 1;
    for (let c = 0; c < channels; c++) {
      for (let h = 0; h < height; h++) {
        for (let w = 0; w < width; w++) {
          out[(c * height + h) * width + w] =
            (data[(h * width + w) * channels + c] ?? 0) * scale;
        }
      }
    }
    return Tensor.from(out, [channels, height, width]);
  }

  describe(): string {
    return "ToTensor()";
  }
}

/**
 * `(x - mean) / std`, per channel.
 */
export class Normalize implements Transform {
  constructor(
    protected readonly mean: readonly number[],
    protected readonly std: readonly number[],
  ) {}

  apply(x: Image | Tensor): Tensor {
    if (!(x instanceof Tensor)) throw new Error("Normalize takes a tensor");
    const shape = [this.mean.length, 1, 1];
    const m = Tensor.from([...this.mean], shape);
    const s = Tensor.from([...this.std], shape);
    return x.sub(m).div(s);
  }

  describe(): string {
    // Python prints a tuple as `(0.5, 0.4, 0.3)`. Printing what arrived, unchanged, is
    // the rule, and the golden has frozen those characters.
    return `Normalize(mean=${tuple(this.mean)}, std=${tuple(this.std)})`;
  }
}

/** Python's tuple notation. One element carries a trailing comma. */
function tuple(values: readonly number[]): string {
  const parts = values.map((v) => String(v));
  return parts.length === 1 ? `(${parts[0]},)` : `(${parts.join(", ")})`;
}

/**
 * The generator for transforms that draw.
 *
 * **The golden does not compare draws** — it pins the probability at 0 or 1, or leaves
 * only one place to crop, and asks about the deterministic part alone. So the generator
 * here does not have to match torch's, and must not pretend to.
 */
const rng = { state: 12345 };

export function manualSeed(seed: number): void {
  rng.state = seed >>> 0;
}

function nextFloat(): number {
  // xorshift32. Not a place that measures a distribution — only that drawing runs.
  let x = rng.state || 1;
  x ^= x << 13; x >>>= 0;
  x ^= x >> 17;
  x ^= x << 5; x >>>= 0;
  rng.state = x;
  return x / 0x100000000;
}

function nextInt(bound: number): number {
  return bound <= 1 ? 0 : Math.floor(nextFloat() * bound);
}

/**
 * Mirrors on one axis. **It draws nothing** — the two random flips draw first
 * and then call this, and `TenCrop` calls it without drawing at all. Reaching
 * for `RandomVerticalFlip(1)` to do the turn instead would look the same and
 * would quietly consume a draw, moving every later transform's dice.
 */
function flipped(img: Image, vertical: boolean): Image {
  const out = new Float64Array(img.data.length);
  for (let h = 0; h < img.height; h++) {
    for (let w = 0; w < img.width; w++) {
      const sh = vertical ? img.height - 1 - h : h;
      const sw = vertical ? w : img.width - 1 - w;
      const from = (sh * img.width + sw) * img.channels;
      const to = (h * img.width + w) * img.channels;
      for (let c = 0; c < img.channels; c++) out[to + c] = img.data[from + c] ?? 0;
    }
  }
  return { ...img, data: out };
}

/**
 * Flips left to right. Takes an `(H, W, C)` array.
 */
export class RandomHorizontalFlip implements Transform {
  constructor(protected readonly p = 0.5) {}

  apply(x: Image | Tensor): Image {
    const img = asImage(x, "RandomHorizontalFlip");
    if (nextFloat() >= this.p) return img;
    return flipped(img, false);
  }

  describe(): string {
    return `RandomHorizontalFlip(p=${this.p})`;
  }
}

/**
 * Flips top to bottom. `RandomHorizontalFlip`'s place, on the other axis.
 *
 * **The default is 0.5 here as well, and that is worth saying out loud.** A
 * vertical flip is wrong for most photographs — an upside-down cat is not a cat
 * the model will meet — so this is the one transform whose default is usually
 * not what is wanted. torchvision keeps 0.5 anyway, and so does this.
 */
export class RandomVerticalFlip implements Transform {
  constructor(protected readonly p = 0.5) {}

  apply(x: Image | Tensor): Image {
    const img = asImage(x, "RandomVerticalFlip");
    if (nextFloat() >= this.p) return img;
    return flipped(img, true);
  }

  describe(): string {
    return `RandomVerticalFlip(p=${this.p})`;
  }
}

export type PaddingMode = "constant" | "edge" | "reflect" | "symmetric";

const PADDING_MODES: readonly string[] = ["constant", "edge", "reflect", "symmetric"];

/**
 * `(left, top, right, bottom)` — **torchvision's order, which is not numpy's.**
 *
 * One number is all four sides, two are (left/right, top/bottom), four are the
 * sides one by one. The two-element form is the one that misreads: it is not
 * (left, top).
 */
function padSides(padding: number | readonly number[]): [number, number, number, number] {
  if (typeof padding === "number") return [padding, padding, padding, padding];
  const s = padding.map((v) => Math.trunc(v));
  const [a = 0, b = 0, c = 0, d = 0] = s;
  if (s.length === 1) return [a, a, a, a];
  if (s.length === 2) return [a, b, a, b];
  if (s.length !== 4) {
    throw new RuntimeError(
      `padding is one, two or four numbers — it received ${s.length}.\n` +
      "(torch: Padding must be an int or a 1, 2, or 4 element tuple)");
  }
  return [a, b, c, d];
}

/**
 * Where a coordinate outside the picture reads from, per mode.
 *
 * `reflect` mirrors **without** repeating the edge and `symmetric` mirrors
 * **with** it, which is a one-pixel difference and the whole distinction
 * between the two names. numpy's meanings, so the arithmetic is numpy's.
 */
function sourceIndex(i: number, n: number, mode: PaddingMode): number {
  if (i >= 0 && i < n) return i;
  if (n === 1) return 0;
  if (mode === "edge") return i < 0 ? 0 : n - 1;
  if (mode === "reflect") {
    const period = 2 * n - 2;
    const k = ((i % period) + period) % period;
    return k < n ? k : period - k;
  }
  const period = 2 * n;
  const k = ((i % period) + period) % period;
  return k < n ? k : period - 1 - k;
}

/**
 * Pads the four sides. **The order is left, top, right, bottom** — which is
 * not numpy's order, and not the order the two-element form reads as.
 *
 * `paddingMode` is `constant` (the default, filled with `fill`), `edge`,
 * `reflect` or `symmetric` — the same four numpy has, with the same meanings.
 */
export class Pad implements Transform {
  constructor(
    protected readonly padding: number | readonly number[],
    protected readonly fill: number | readonly number[] = 0,
    protected readonly paddingMode: PaddingMode = "constant",
  ) {
    if (!PADDING_MODES.includes(paddingMode)) {
      throw new RuntimeError(
        `padding_mode is constant, edge, reflect or symmetric — got ` +
        `${JSON.stringify(paddingMode)}.\n` +
        "(torch: Padding mode should be either constant, edge, reflect or symmetric)");
    }
    padSides(padding);          // it stops here rather than at the first call
  }

  apply(x: Image | Tensor): Image {
    const img = asImage(x, "Pad");
    const [left, top, right, bottom] = padSides(this.padding);
    const height = img.height + top + bottom;
    const width = img.width + left + right;
    const c = img.channels;
    const out = new Float64Array(height * width * c);
    // **A per-channel fill is read per channel, not per axis.** numpy's
    // `constant_values` is read per axis, so a three-colour fill given there
    // paints the channel axis instead of the colours — the Python side pads each
    // channel separately for exactly this reason.
    const fills = typeof this.fill === "number" ? null : this.fill;
    if (fills !== null && fills.length !== c) {
      throw new RuntimeError(
        `fill has ${fills.length} numbers and the image has ${c} channels`);
    }
    for (let h = 0; h < height; h++) {
      const sh = h - top;
      for (let w = 0; w < width; w++) {
        const sw = w - left;
        const to = (h * width + w) * c;
        const inside = sh >= 0 && sh < img.height && sw >= 0 && sw < img.width;
        if (!inside && this.paddingMode === "constant") {
          for (let k = 0; k < c; k++) {
            out[to + k] = fills === null ? this.fill as number : (fills[k] ?? 0);
          }
          continue;
        }
        const fh = sourceIndex(sh, img.height, this.paddingMode);
        const fw = sourceIndex(sw, img.width, this.paddingMode);
        const from = (fh * img.width + fw) * c;
        for (let k = 0; k < c; k++) out[to + k] = img.data[from + k] ?? 0;
      }
    }
    return { data: out, height, width, channels: c, isByte: img.isByte };
  }

  describe(): string {
    const pad = typeof this.padding === "number" ? `${this.padding}` : tuple(this.padding);
    const fill = typeof this.fill === "number" ? `${this.fill}` : tuple(this.fill);
    return `Pad(padding=${pad}, fill=${fill}, padding_mode=${this.paddingMode})`;
  }
}

/** The luma weights. torchvision's, verbatim — a rounded set moves whole pixels. */
const LUMA = [0.2989, 0.587, 0.114] as const;

// The same three numbers narrowed to float32. **Why both are needed is the point** —
// see below.
const LUMA32 = [Math.fround(0.2989), Math.fround(0.587), Math.fround(0.114)] as const;

/**
 * Three channels to one.
 *
 * **The cast truncates, and that is the point.** torch's `.to(dtype)` truncates
 * too, so a byte picture comes out the same on both sides. PIL's `convert("L")`
 * rounds instead, which is where a byte answer here and a PIL answer part by
 * one — measured on the Python side and written down rather than smoothed over.
 */
function toGray(img: Image, outChannels: number, who: string): Image {
  if (outChannels !== 1 && outChannels !== 3) {
    throw new RuntimeError(
      `num_output_channels is 1 or 3 — got ${outChannels}.\n` +
      "(torch: num_output_channels should be either 1 or 3)");
  }
  if (img.channels !== 1 && img.channels !== 3) {
    throw new RuntimeError(
      `${who} takes a 1- or 3-channel image — it received ${img.channels} channels.`);
  }
  const pixels = img.height * img.width;
  const one = new Float64Array(pixels);
  if (img.channels === 3) {
    for (let i = 0; i < pixels; i++) {
      const r = img.data[i * 3] ?? 0;
      const g = img.data[i * 3 + 1] ?? 0;
      const b = img.data[i * 3 + 2] ?? 0;
      // **There are two branches and both follow numpy exactly.** It is written down
      // because it was found by measurement — it is invisible in either source and only
      // appears when the two are laid side by side.
      //
      // Python is the one line `arr[:,:,0]*_LUMA[0] + ...`, and that line's arithmetic
      // branches on the array's dtype (NEP 50's weak promotion):
      //
      //   uint8   × Python float → float64.  Everything computes in float64 and
      //                                      truncates at the end.
      //   float32 × Python float → float32.  **The scalar narrows to float32 first**,
      //                                      and every product and partial sum after it
      //                                      is float32.
      //
      // Computing the float branch in float64 diverges from the golden by up to 6.5e-8.
      // That is inside the tolerance, so **the check passes** — with 0 of 20 pixels
      // exactly right. Narrowing the scalar alone gives 16/20, narrowing each product
      // gives 16/20, and doing it as above gives 20/20 at zero error.
      const lum = img.isByte
        ? r * LUMA[0] + g * LUMA[1] + b * LUMA[2]
        : Math.fround(Math.fround(Math.fround(r * LUMA32[0])
            + Math.fround(g * LUMA32[1])) + Math.fround(b * LUMA32[2]));
      // `lum.astype(arr.dtype)` — only the byte branch narrows here. The float branch is
      // already float32 above, so this cast changes no value.
      one[i] = img.isByte ? Math.trunc(lum) : lum;
    }
  } else {
    for (let i = 0; i < pixels; i++) one[i] = img.data[i] ?? 0;
  }
  if (outChannels === 1) {
    return { data: one, height: img.height, width: img.width, channels: 1,
      isByte: img.isByte };
  }
  const out = new Float64Array(pixels * 3);
  for (let i = 0; i < pixels; i++) {
    out[i * 3] = one[i] ?? 0;
    out[i * 3 + 1] = one[i] ?? 0;
    out[i * 3 + 2] = one[i] ?? 0;
  }
  return { data: out, height: img.height, width: img.width, channels: 3,
    isByte: img.isByte };
}

/**
 * `torchvision.transforms.Grayscale`. Three channels to one.
 *
 * `numOutputChannels=3` gives it back as three equal ones — which is what a
 * pre-trained three-channel model needs.
 */
export class Grayscale implements Transform {
  constructor(protected readonly numOutputChannels = 1) {}

  apply(x: Image | Tensor): Image {
    return toGray(asImage(x, "Grayscale"), this.numOutputChannels, "Grayscale");
  }

  describe(): string {
    return `Grayscale(num_output_channels=${this.numOutputChannels})`;
  }
}

/**
 * Grayscale with probability `p`. **The channel count does not change** — a
 * three-channel image comes back as three equal channels, so the batch that
 * follows still stacks.
 */
export class RandomGrayscale implements Transform {
  constructor(protected readonly p = 0.1) {}

  apply(x: Image | Tensor): Image {
    const img = asImage(x, "RandomGrayscale");
    if (nextFloat() >= this.p) return img;
    return toGray(img, img.channels, "RandomGrayscale");
  }

  describe(): string {
    return `RandomGrayscale(p=${this.p})`;
  }
}

/**
 * Pads the edges, then crops at random.
 */
export class RandomCrop implements Transform {
  protected readonly size: [number, number];

  /**
   * **The argument list is torchvision's, and it was not.** This took
   * `(size, padding, fill)` where torchvision takes
   * `(size, padding, padIfNeeded, fill, paddingMode)`, so `new RandomCrop(32, 4, true)`
   * — the line somebody transcribing torch writes — put `true` in `fill` here and
   * `pad_if_needed` there. It compiled, it ran, and it came out the right shape either
   * way. The Python side had already been corrected and said so in its docstring; this
   * side was left behind because nothing compared the two.
   *
   * What found it: `tests/ts_axis.py` gained the `transforms` namespace, which put the
   * signature axis on this constructor for the first time and it came back `shifted`.
   * The golden could not — it asks `RandomCrop((3,2), padding=4)` by keyword, so the
   * third position was never occupied on either side.
   *
   * The padding itself was already here, one class over: `Pad` does all four modes and
   * a per-side padding, while this used a private `padded()` that only knew a constant
   * fill and one symmetric width.
   *
   * @param padding **`null` is the default, not `0`** — torchvision's is `None` and its
   *   repr prints that word. They pad identically, so the difference lives only in the
   *   printed line.
   */
  constructor(
    size: number | readonly [number, number],
    protected readonly padding: number | readonly number[] | null = null,
    protected readonly padIfNeeded = false,
    protected readonly fill: number | readonly number[] = 0,
    protected readonly paddingMode: PaddingMode = "constant",
  ) {
    this.size = typeof size === "number" ? [size, size] : [size[0], size[1]];
  }

  apply(x: Image | Tensor): Image {
    let img = asImage(x, "RandomCrop");
    if (this.padding !== null) {
      img = new Pad(this.padding, this.fill, this.paddingMode).apply(img);
    }
    const [th, tw] = this.size;
    // **Width first and then height, each on its own** — torchvision pads them in two
    // separate steps and the second reads the width the first produced. Done together,
    // a picture short on both sides comes out a different size.
    if (this.padIfNeeded && img.width < tw) {
      img = new Pad([tw - img.width, 0], this.fill, this.paddingMode).apply(img);
    }
    if (this.padIfNeeded && img.height < th) {
      img = new Pad([0, th - img.height], this.fill, this.paddingMode).apply(img);
    }
    if (img.height < th || img.width < tw) {
      throw new Error(
        `Required crop size (${th}, ${tw}) is larger than input image size ` +
          `(${img.height}, ${img.width})`,
      );
    }
    const top = nextInt(img.height - th + 1);
    const left = nextInt(img.width - tw + 1);
    const out = new Float64Array(th * tw * img.channels);
    for (let h = 0; h < th; h++) {
      for (let w = 0; w < tw; w++) {
        const from = ((top + h) * img.width + (left + w)) * img.channels;
        const to = (h * tw + w) * img.channels;
        for (let c = 0; c < img.channels; c++) out[to + c] = img.data[from + c] ?? 0;
      }
    }
    return { ...img, data: out, height: th, width: tw };
  }

  describe(): string {
    // The Python side holds `size` normalised as a tuple, so it prints in that shape.
    return `RandomCrop(size=(${this.size[0]}, ${this.size[1]}), `
      + `padding=${this.padding === null ? "None" : this.padding})`;
  }
}

/**
 * Per output position, a (read start, weights) pair. It covers one axis — the filter is
 * separable, so one horizontal pass and one vertical pass suffice.
 *
 * **It antialiases.** torchvision's `Resize` defaults to `antialias=true`, and the
 * difference from having it off is up to 0.0301 at 8×8→4×4 (measured). It is not zero, so
 * writing only "bilinear" leaves which one undecided, and feeding a model trained with it
 * on an input produced with it off is a different input.
 *
 * When enlarging, the spread is 1 and the triangle becomes ordinary bilinear — which
 * is why one branch suffices.
 *
 * **The window end follows torch's truncation.** Two boundary rules were once measured
 * as giving the same answer, and they still do on every case here — the triangle is 0
 * across the extra sample a rounded-up window reaches, and the cubic's second lobe has
 * not landed there yet. The rule is written as torch writes it rather than left to
 * agree by luck, because the filter that could break the tie now exists.
 */
/**
 * The cubic filter torch resizes with — Keys' with `a = -0.5`, Catmull-Rom.
 *
 * **`a` was measured, not looked up.** -0.75 is the value PIL and OpenCV use and
 * the one most references give, and it was written here first. Against
 * `torchvision.transforms.v2.functional.resize(antialias=True)` it was off by
 * 2.3e-2 on a 16→8 shrink; -0.5 lands at 1.1e-7. torch's antialiasing path uses
 * the other constant, and the two are different pictures — a model fed one and
 * trained on the other is fed a different input.
 *
 * So this matches **torch**, which is what this library imitates. A resize meant
 * to match PIL would need -0.75 and is not on offer here; asking for one against
 * a checkpoint trained through torch would be the wrong picture anyway.
 */
function cubicWeight(x: number): number {
  const a = -0.5;
  const d = Math.abs(x);
  if (d < 1) return ((a + 2) * d - (a + 3)) * d * d + 1;
  if (d < 2) return (((d - 5) * d + 8) * d - 4) * a;
  return 0;
}

/**
 * @param cubic bicubic rather than bilinear. It widens the window — the filter
 *   reaches two source pixels instead of one — and the weights stop being
 *   positive, which is where the ringing on a hard edge comes from. Both are the
 *   filter doing its job, not a fault to smooth over.
 */
function aaWeights(
  src: number, dst: number, cubic = false,
): { at: number; w: Float64Array }[] {
  const scale = src / dst;
  // **The spread and the reach are two things.** Antialiasing widens the filter by
  // the shrink factor (`spread`); how far the filter itself reaches is a property
  // of the filter (1 for the triangle, 2 for the cubic). Multiplying them is what
  // torchvision does, and folding them into one number was fine only while there
  // was one filter.
  const spread = Math.max(1, scale);
  const support = (cubic ? 2 : 1) * spread;
  const rows: { at: number; w: Float64Array }[] = [];
  for (let i = 0; i < dst; i++) {
    const center = (i + 0.5) * scale;
    const lo = Math.max(0, Math.floor(center - support + 0.5));
    // **`floor`, matching torch's truncation.** This file rounded up and said it
    // used torch's side; the two disagree only where a filter is non-zero across
    // the extra sample, and the triangle never is. The cubic has a second lobe out
    // there and could have been, so the rule was made to say what it does. On the
    // cases measured here both rules give the same numbers — the difference that
    // looked like this one turned out to be the kernel constant above.
    const hi = Math.min(src, Math.floor(center + support + 0.5));
    const w = new Float64Array(Math.max(0, hi - lo));
    let total = 0;
    for (let j = lo; j < hi; j++) {
      const at = (j + 0.5 - center) / spread;
      const v = cubic ? cubicWeight(at) : Math.max(0, 1 - Math.abs(at));
      w[j - lo] = v;
      total += v;
    }
    // **The sum is what is normalised, not the positives.** A cubic window has
    // negative weights in it; dividing by the sum of everything keeps the total at
    // one, which is what stops a flat picture from changing brightness.
    if (total !== 0) for (let k = 0; k < w.length; k++) w[k] = (w[k] ?? 0) / total;
    rows.push({ at: lo, w });
  }
  return rows;
}

/**
 * The short side to `size`. The long side keeps the ratio — torchvision's `Resize(int)`.
 *
 * **`maxSize` comes after the ratio rather than instead of it.** The short side is set to
 * `size` first and the long side follows; only when that exceeds the cap is the long side
 * clipped to the cap and the short side shrunk to follow. Both divisions **truncate** —
 * rounding gives (9, 8) rather than (9, 7) when reducing a 5×4 with
 * `Resize(8, maxSize=9)`.
 */
function shortSide(
  h: number, w: number, size: number, maxSize: number | null,
): [number, number] {
  const short = Math.min(h, w);
  const long = Math.max(h, w);
  let newShort = size;
  let newLong = short === size ? long : Math.trunc((size * long) / short);
  if (maxSize !== null && newLong > maxSize) {
    // **It throws here rather than at construction.** torchvision also stops inside the
    // size arithmetic, so `Resize(4, max_size=4)` stands up and stops when it receives a
    // picture. The repr cases only build the transform without calling it, so throwing
    // earlier would change which side diverged.
    if (maxSize <= size) {
      throw new RuntimeError(
        `max_size = ${maxSize} must be strictly greater than size = ${size} — ` +
        "the short side is set to size first, so a cap at or below it has nothing to cap.");
    }
    newShort = Math.trunc((maxSize * newShort) / newLong);
    newLong = maxSize;
  }
  return h < w ? [newShort, newLong] : [newLong, newShort];
}

/** Changes one axis of something laid out as `(H, W, C)`. */
function resizeRows(
  src: Float64Array, h: number, w: number, c: number, dst: number,
  mode: "bilinear" | "nearest" | "bicubic",
): Float64Array {
  const out = new Float64Array(dst * w * c);
  if (mode === "nearest") {
    for (let y = 0; y < dst; y++) {
      const from = Math.trunc(y * (h / dst));
      out.set(src.subarray(from * w * c, (from + 1) * w * c), y * w * c);
    }
    return out;
  }
  const rows = aaWeights(h, dst, mode === "bicubic");
  for (let y = 0; y < dst; y++) {
    const { at, w: weights } = rows[y] ?? { at: 0, w: new Float64Array() };
    for (let k = 0; k < weights.length; k++) {
      const weight = weights[k] ?? 0;
      const base = (at + k) * w * c;
      for (let i = 0; i < w * c; i++) {
        out[y * w * c + i] = (out[y * w * c + i] ?? 0) + (src[base + i] ?? 0) * weight;
      }
    }
  }
  return out;
}

/** The same job as the vertical one, on the horizontal. Only the axis differs. */
function resizeCols(
  src: Float64Array, h: number, w: number, c: number, dst: number,
  mode: "bilinear" | "nearest" | "bicubic",
): Float64Array {
  const out = new Float64Array(h * dst * c);
  const rows = mode === "nearest" ? null : aaWeights(w, dst, mode === "bicubic");
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < dst; x++) {
      if (rows === null) {
        const from = Math.trunc(x * (w / dst));
        for (let k = 0; k < c; k++) {
          out[(y * dst + x) * c + k] = src[(y * w + from) * c + k] ?? 0;
        }
        continue;
      }
      const { at, w: weights } = rows[x] ?? { at: 0, w: new Float64Array() };
      for (let j = 0; j < weights.length; j++) {
        const weight = weights[j] ?? 0;
        for (let k = 0; k < c; k++) {
          out[(y * dst + x) * c + k] =
            (out[(y * dst + x) * c + k] ?? 0) + (src[(y * w + at + j) * c + k] ?? 0) * weight;
        }
      }
    }
  }
  return out;
}

/**
 * Changes the size. **It takes an array and returns an array** — not a
 * tensor. Making a tensor per image makes one GPU buffer per image (the
 * same reason as `RandomCrop`).
 *
 * Given a single number for `size`, it fits **the short side** to that
 * value and keeps the ratio.
 */
export class Resize implements Transform {
  /**
   * @param maxSize a cap on the long side. **Only meaningful with a single
   *   number**, where `size` is the short side — given an explicit pair there is
   *   nothing left to cap, and torchvision refuses that combination too.
   * @param antialias **`false` is refused rather than accepted and ignored.**
   *   There is one filter here and it antialiases; off differs from on by up to
   *   0.0301 at 8×8→4×4 (measured), so taking the argument and dropping it would
   *   hand back the other image without saying so.
   */
  constructor(
    protected readonly size: number | readonly [number, number],
    protected readonly interpolation: "bilinear" | "nearest" | "bicubic" = "bilinear",
    private readonly maxSize: number | null = null,
    antialias = true,
  ) {
    if (maxSize !== null && typeof size !== "number") {
      throw new RuntimeError(
        "max_size means something only when the size is the short side (a single number).\n" +
        "(torch: max_size should only be passed if size is int or sequence of length 1)");
    }
    if (!antialias) {
      throw new RuntimeError(
        "Resize(antialias=false) is not in the browser subset — there is one filter here " +
        "and it antialiases. Turning it off changes the values by up to 0.0301 (measured), " +
        "so accepting the argument and ignoring it would hand back a different image.");
    }
  }

  apply(x: Image | Tensor): Image {
    const img = asImage(x, "Resize");
    const [th, tw] = typeof this.size === "number"
      ? shortSide(img.height, img.width, this.size, this.maxSize)
      : [this.size[0], this.size[1]];
    
    let data = img.data;
    let h = img.height;
    if (th !== h) {
      data = resizeRows(data, h, img.width, img.channels, th, this.interpolation);
      h = th;
    }
    let w = img.width;
    if (tw !== w) {
      data = resizeCols(data, h, w, img.channels, tw, this.interpolation);
      w = tw;
    }
    return { data, height: h, width: w, channels: img.channels, isByte: img.isByte };
  }

  /**
   * **Four fields, not two.** torchvision has always printed `max_size` and
   * `antialias`; this printed two and agreed with the Python side, which printed
   * two as well — so the transform whose repr differed was the one the golden
   * had no case for. The check's coverage decided which defect could exist.
   */
  describe(): string {
    const size = typeof this.size === "number" ? this.size : `(${this.size.join(", ")})`;
    return `Resize(size=${size}, interpolation=${this.interpolation}, `
      + `max_size=${this.maxSize === null ? "None" : this.maxSize}, antialias=True)`;
  }
}

/**
 * **Python's `round` sends a half to the even side.** `Math.round` rounds up —
 * `round(0.5)` is 0 in Python and 1 in JS.
 *
 * torchvision's `CenterCrop` takes its start with `int(round(...))`, so without imitating
 * that exactly **the crop lands one cell out.** It is not a slightly different value but
 * different pixels, and it measured as a divergence of up to 0.837 — invisible until the
 * Python and TypeScript versions were laid side by side.
 */
function roundHalfToEven(x: number): number {
  const down = Math.floor(x);
  const frac = x - down;
  if (frac !== 0.5) return Math.round(x);
  return down % 2 === 0 ? down : down + 1;
}

/**
 * Crops out the middle. **A crop larger than the original is padded with
 * zeros and then cropped** — torchvision does that, and refusing would make
 * the same code diverge between two libraries.
 */
export class CenterCrop implements Transform {
  private readonly th: number;
  private readonly tw: number;

  constructor(size: number | readonly [number, number]) {
    [this.th, this.tw] = typeof size === "number" ? [size, size] : [size[0], size[1]];
  }

  apply(x: Image | Tensor): Image {
    const img = asImage(x, "CenterCrop");
    const c = img.channels;
    const padH = Math.max(0, this.th - img.height);
    const padW = Math.max(0, this.tw - img.width);
    let { data, height: h, width: w } = img;
    if (padH !== 0 || padW !== 0) {
      const top = Math.floor(padH / 2);
      const left = Math.floor(padW / 2);
      const nh = h + padH;
      const nw = w + padW;
      const grown = new Float64Array(nh * nw * c);
      for (let y = 0; y < h; y++) {
        grown.set(data.subarray(y * w * c, (y + 1) * w * c), ((y + top) * nw + left) * c);
      }
      data = grown;
      h = nh;
      w = nw;
    }
    const top = roundHalfToEven((h - this.th) / 2);
    const left = roundHalfToEven((w - this.tw) / 2);
    const out = new Float64Array(this.th * this.tw * c);
    for (let y = 0; y < this.th; y++) {
      const from = ((top + y) * w + left) * c;
      out.set(data.subarray(from, from + this.tw * c), y * this.tw * c);
    }
    return { data: out, height: this.th, width: this.tw, channels: c, isByte: img.isByte };
  }

  describe(): string {
    return `CenterCrop(size=(${this.th}, ${this.tw}))`;
  }
}

/**
 * Augments an `(N,C,H,W)` batch in one pass. **Ours, not torchvision's.**
 *
 * Making a tensor per image makes one GPU buffer per image — ten thousand
 * of them for a ten-thousand-image epoch. Augmenting the whole batch on the
 * CPU and then making **one** tensor is the only order that holds, so that
 * order is given a name and exported.
 *
 * **The random draws are per image.** Using one crop and one flip across
 * the whole batch means nothing inside the batch was augmented relative to
 * anything else, and the point of augmentation is gone.
 */
export function augmentBatch(
  x: Float32Array,
  n: number, c: number, h: number, w: number,
  opts: { crop?: number; padding?: number; hflipP?: number; fill?: number } = {},
): Float32Array {
  const pad = opts.padding ?? 0;
  const fill = opts.fill ?? 0;
  const ph = h + 2 * pad;
  const pw = w + 2 * pad;
  const th = opts.crop ?? ph;
  const tw = opts.crop ?? pw;
  if (ph < th || pw < tw) {
    throw new Error(`Required crop size (${th}, ${tw}) is larger than input image size (${ph}, ${pw})`);
  }
  const hflipP = opts.hflipP ?? 0;
  const out = new Float32Array(n * c * th * tw);
  for (let i = 0; i < n; i++) {
    const top = nextInt(ph - th + 1);
    const left = nextInt(pw - tw + 1);
    const flip = nextFloat() < hflipP;
    for (let ch = 0; ch < c; ch++) {
      const src = (i * c + ch) * h * w;
      const dst = (i * c + ch) * th * tw;
      for (let y = 0; y < th; y++) {
        // A padded position is outside the original — `fill` is used there.
        const sy = top + y - pad;
        for (let t = 0; t < tw; t++) {
          const sx0 = left + t - pad;
          const sx = flip ? left + (tw - 1 - t) - pad : sx0;
          const inside = sy >= 0 && sy < h && sx >= 0 && sx < w;
          out[dst + y * tw + t] = inside ? (x[src + sy * w + sx] ?? 0) : fill;
        }
      }
    }
  }
  return out;
}

/**
 * `(x - mean) / std` per channel, batch-wide on the CPU — finished before
 * any tensor is made.
 */
export function normalizeBatch(
  x: Float32Array,
  n: number, c: number, hw: number,
  mean: readonly number[], std: readonly number[],
): Float32Array {
  const out = new Float32Array(x.length);
  for (let i = 0; i < n; i++) {
    for (let ch = 0; ch < c; ch++) {
      const m = mean[ch] ?? 0;
      const s = std[ch] ?? 1;
      const base = (i * c + ch) * hw;
      for (let k = 0; k < hw; k++) out[base + k] = ((x[base + k] ?? 0) - m) / s;
    }
  }
  return out;
}

/**
 * `Array.isArray` narrows to `any[]`, which a `readonly Image[]` is not, so it
 * leaves the union un-narrowed. This says the same thing in a way the compiler
 * can act on.
 */
/**
 * Python's float repr: an integer still carries its decimal point (`1.0`, not
 * `1`). JS drops it, and `repr` is read as a specification here.
 */
export function pyFloat(v: number): string {
  return Number.isInteger(v) ? `${v}.0` : String(v);
}

/** Python's `True` / `False`. Exported for the same reason as `pyFloat`. */
export function pyBool(v: boolean): string {
  return v ? "True" : "False";
}

function floatTuple(values: readonly number[]): string {
  return `(${values.map(pyFloat).join(", ")})`;
}

/**
 * Flattens, subtracts the mean, multiplies by a matrix, puts the shape back.
 *
 * This is where whitening (ZCA/PCA) is applied: the matrix and the mean are
 * worked out **beforehand** from the training set, and this only applies them.
 * It takes a `(...,C,H,W)` tensor, so it runs after `ToTensor`.
 *
 * **The matrix arrives as numbers, where torchvision takes a tensor.** That is
 * forced, not preferred: `repr` prints the matrix, WebGPU has no synchronous
 * read, and `describe()` is not async. Taking a tensor here would mean a `repr`
 * that cannot say what it holds. It is the same reason an `Image` in this file
 * is an array — the thing is being described, not computed on, until it is.
 */
export class LinearTransformation implements Transform {
  private readonly side: number;
  // **Built late.** Building the tensor in the constructor makes `describe()` demand a
  // device, and the Python side prints without a GPU. As long as `repr` is treated as a
  // specification, having to bring up a device to see one is backwards — and a check
  // really did catch this.
  // **The lazily built tensors, renamed out of the way.** The constructor's arguments
  // took torchvision's `transformationMatrix` and `meanVector`, and the second collided
  // with the field holding the built tensor. The argument keeps the outside name; the
  // field, which only this class reads, is the one that yields.
  private matrixT: Tensor | null = null;
  private meanT: Tensor | null = null;

  constructor(
    private readonly transformationMatrix: readonly (readonly number[])[],
    private readonly meanVector: readonly number[],
  ) {
    this.side = transformationMatrix.length;
    if (transformationMatrix.some((r) => r.length !== this.side)) {
      throw new RuntimeError(
        `transformation_matrix should be square — it received ${transformationMatrix.length} transformationMatrix ` +
        `of lengths ${transformationMatrix.map((r) => r.length).join(", ")}.\n` +
        "(torch: transformation_matrix should be square)");
    }
    if (meanVector.length !== this.side) {
      throw new RuntimeError(
        `mean_vector should be as long as one side of the matrix ` +
        `(${this.side}, ${this.side}) — it received ${meanVector.length}.\n` +
        "(torch: mean_vector should have the same length)");
    }
  }

  private tensors(): [Tensor, Tensor] {
    this.matrixT ??= Tensor.from(this.transformationMatrix.flat(), [this.side, this.side]);
    this.meanT ??= Tensor.from([...this.meanVector], [this.side]);
    return [this.matrixT, this.meanT];
  }

  apply(x: Subject): Tensor {
    if (!(x instanceof Tensor)) {
      throw new RuntimeError(
        "LinearTransformation takes a (...,C,H,W) tensor.\n" +
        "  It runs after `ToTensor`, not before it.");
    }
    const shape = x.shape;
    if (shape.length < 3) {
      throw new RuntimeError(
        `LinearTransformation takes (...,C,H,W) — it received (${shape.join(", ")})`);
    }
    const n = (shape[shape.length - 3] ?? 1) * (shape[shape.length - 2] ?? 1)
      * (shape[shape.length - 1] ?? 1);
    if (n !== this.side) {
      throw new RuntimeError(
        `The image flattens to ${n} and the matrix is ${this.side} wide — ` +
        "they do not meet.\n" +
        "(torch: Input tensor and transformation matrix have incompatible shape)");
    }
    const [matrix, meanVector] = this.tensors();
    return x.reshape([-1, n]).sub(meanVector).mm(matrix).reshape(shape);
  }

  describe(): string {
    const transformationMatrix = this.transformationMatrix.map((r) => `[${r.map(pyFloat).join(", ")}]`);
    return `LinearTransformation(transformation_matrix=[${transformationMatrix.join(", ")}]` +
      `, mean_vector=[${this.meanVector.map(pyFloat).join(", ")}])`;
  }
}

/** `round(v, 4)`, which is what the two ratio reprs below print. */
function round4(v: number): number {
  return Number(v.toFixed(4));
}

function uniform(lo: number, hi: number): number {
  return lo + (hi - lo) * nextFloat();
}

/**
 * Crop a random area of a random shape, then resize it to `size`. **The
 * ImageNet recipe**, and the reason a tutorial's accuracy moves when it is left
 * out.
 *
 * `scale` is the fraction of the area to keep and `ratio` the width-to-height
 * range. Ten draws are made, and if none of them fits inside the picture it
 * falls back to a centre crop — torchvision does exactly that, fallback
 * included, because without it the draw can fail on a thin picture and there is
 * nothing to return.
 */
export class RandomResizedCrop implements Transform {
  private readonly th: number;
  private readonly tw: number;

  constructor(
    size: number | readonly [number, number],
    protected readonly scale: readonly [number, number] = [0.08, 1.0],
    protected readonly ratio: readonly [number, number] = [3 / 4, 4 / 3],
    protected readonly interpolation: "bilinear" | "nearest" = "bilinear",
    private readonly antialias = true,
  ) {
    [this.th, this.tw] = pairOf(size);
  }

  /**
   * Where and how big. **Ten draws, then a centre crop.** Kept separate because
   * it is the only part that draws.
   */
  getParams(img: Image): [number, number, number, number] {
    const { height: h, width: w } = img;
    const area = h * w;
    const logRatio: [number, number] = [Math.log(this.ratio[0]), Math.log(this.ratio[1])];
    for (let i = 0; i < 10; i++) {
      const target = area * uniform(this.scale[0], this.scale[1]);
      const aspect = Math.exp(uniform(logRatio[0], logRatio[1]));
      const cw = roundHalfToEven(Math.sqrt(target * aspect));
      const ch = roundHalfToEven(Math.sqrt(target / aspect));
      if (cw > 0 && cw <= w && ch > 0 && ch <= h) {
        return [nextInt(h - ch + 1), nextInt(w - cw + 1), ch, cw];
      }
    }
    const inRatio = w / h;
    const lo = Math.min(this.ratio[0], this.ratio[1]);
    const hi = Math.max(this.ratio[0], this.ratio[1]);
    let cw = w, ch = h;
    if (inRatio < lo) {
      ch = roundHalfToEven(w / lo);
    } else if (inRatio > hi) {
      cw = roundHalfToEven(h * hi);
    }
    return [Math.floor((h - ch) / 2), Math.floor((w - cw) / 2), ch, cw];
  }

  apply(x: Subject): Image {
    const img = asImage(x, "RandomResizedCrop");
    const [top, left, ch, cw] = this.getParams(img);
    const piece = cropAt(img, top, left, ch, cw);
    return new Resize([this.th, this.tw], this.interpolation, null, this.antialias)
      .apply(piece);
  }

  describe(): string {
    return `RandomResizedCrop(size=(${this.th}, ${this.tw}), ` +
      `scale=${floatTuple(this.scale.map(round4))}, ` +
      `ratio=${floatTuple(this.ratio.map(round4))}, ` +
      `interpolation=${this.interpolation}, ` +
      `antialias=${this.antialias ? "True" : "False"})`;
  }
}

/**
 * Blank out a random rectangle of a **tensor** — this one runs after
 * `ToTensor`, unlike every other transform in this file.
 *
 * That is torchvision's position for it and not a choice made here: the erased
 * value is `0` on a normalised image, which means the channel mean, and that
 * only has a meaning once the picture is numbers rather than pixels.
 */
export class RandomErasing implements Transform {
  constructor(
    protected readonly p = 0.5,
    protected readonly scale: readonly [number, number] = [0.02, 0.33],
    protected readonly ratio: readonly [number, number] = [0.3, 3.3],
    protected readonly value: number | readonly number[] | "random" = 0,
    protected readonly inplace = false,
  ) {
    if (scale[0] < 0 || scale[1] > 1) {
      throw new RuntimeError(
        `scale is a fraction of the area, between 0 and 1 — got ` +
        `(${scale[0]}, ${scale[1]}).\n(torch: Scale should be between 0 and 1)`);
    }
    if (p < 0 || p > 1) {
      throw new RuntimeError(
        `p is a probability, between 0 and 1 — got ${p}.\n` +
        "(torch: Random erasing probability should be between 0 and 1)");
    }
  }

  /**
   * `[top, left, height, width]`, or `null` when ten draws all missed.
   *
   * **The rectangle has to be strictly smaller than the picture** on both
   * sides. torchvision's condition is `<` rather than `<=`, so an erase
   * covering the whole picture never happens, and on a small picture the ten
   * draws can all miss — that is the `null`.
   */
  getParams(h: number, w: number): [number, number, number, number] | null {
    const area = h * w;
    const logRatio: [number, number] = [Math.log(this.ratio[0]), Math.log(this.ratio[1])];
    for (let i = 0; i < 10; i++) {
      const erase = area * uniform(this.scale[0], this.scale[1]);
      const aspect = Math.exp(uniform(logRatio[0], logRatio[1]));
      const eh = roundHalfToEven(Math.sqrt(erase * aspect));
      const ew = roundHalfToEven(Math.sqrt(erase / aspect));
      if (!(eh < h && ew < w)) continue;
      return [nextInt(h - eh + 1), nextInt(w - ew + 1), eh, ew];
    }
    return null;
  }

  apply(x: Subject): Tensor {
    if (!(x instanceof Tensor)) {
      throw new RuntimeError(
        "RandomErasing takes a (...,C,H,W) tensor.\n" +
        "  It runs after `ToTensor`, not before it.");
    }
    if (nextFloat() >= this.p) return x;
    const shape = x.shape;
    if (shape.length < 3) {
      throw new RuntimeError(
        `RandomErasing takes (...,C,H,W) — it received (${shape.join(", ")}).\n` +
        "  It runs after `ToTensor`, not before it.");
    }
    const c = shape[shape.length - 3] ?? 1;
    const h = shape[shape.length - 2] ?? 1;
    const w = shape[shape.length - 1] ?? 1;
    const found = this.getParams(h, w);
    if (found === null) return x;
    const [top, left, eh, ew] = found;
    const fills = this.channelFills(c, eh, ew);
    // **A mask the size of the whole tensor**, because `where` does not
    // broadcast — it reads position by position. The rectangle is the same on
    // every leading dimension, so the flat index is enough to place it.
    const n = x.size;
    const mask = new Float64Array(n);
    const paint = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const col = i % w;
      const row = Math.floor(i / w) % h;
      if (row < top || row >= top + eh || col < left || col >= left + ew) continue;
      mask[i] = 1;
      const ch = Math.floor(i / (h * w)) % c;
      paint[i] = fills[(ch * eh + (row - top)) * ew + (col - left)] ?? 0;
    }
    const out = Tensor.from(paint, shape).where(Tensor.from(mask, shape), x);
    // `inplace` copies the answer back into the tensor that was handed in, so a
    // caller holding a reference sees it — which is what the flag promises.
    if (this.inplace) { x.copyFrom(out); return x; }
    return out;
  }

  /** The numbers that go into the rectangle, laid out `(C, eh, ew)`. */
  private channelFills(c: number, eh: number, ew: number): Float64Array {
    const out = new Float64Array(c * eh * ew);
    if (this.value === "random") {
      // **Nothing compares these numbers.** The golden's two erasing cases both
      // hand the picture back untouched (p=0, and ten draws that all miss), and
      // the distribution test that does look at the draw is on the Python side.
      // Box–Muller off the same generator the rest of the file draws from — the
      // stream cannot match numpy's, and matching it was never possible.
      for (let i = 0; i < out.length; i++) {
        const u = Math.max(nextFloat(), Number.MIN_VALUE);
        out[i] = Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * nextFloat());
      }
      return out;
    }
    if (typeof this.value === "number") return out.fill(this.value);
    if (this.value.length !== 1 && this.value.length !== c) {
      throw new RuntimeError(
        `value has ${this.value.length} numbers and the image has ${c} channels.\n` +
        "(torch: If value is a sequence, it should have either a single value or " +
        "(number of input channels))");
    }
    for (let k = 0; k < c; k++) {
      const v = this.value[this.value.length === 1 ? 0 : k] ?? 0;
      out.fill(v, k * eh * ew, (k + 1) * eh * ew);
    }
    return out;
  }

  describe(): string {
    const v = this.value === "random" ? "random" : typeof this.value === "number"
      ? `${this.value}` : tuple(this.value);
    return `RandomErasing(p=${this.p}, scale=${tuple(this.scale)}, ` +
      `ratio=${tuple(this.ratio)}, value=${v}, ` +
      `inplace=${this.inplace ? "True" : "False"})`;
  }
}

/** `(h, w)` from either a number or a pair. */
function pairOf(size: number | readonly [number, number]): [number, number] {
  return typeof size === "number" ? [size, size] : [size[0], size[1]];
}

/** The rectangle at `(top, left)`, `th` by `tw`. */
function cropAt(img: Image, top: number, left: number, th: number, tw: number): Image {
  const c = img.channels;
  const out = new Float64Array(th * tw * c);
  for (let y = 0; y < th; y++) {
    const from = ((top + y) * img.width + left) * c;
    out.set(img.data.subarray(from, from + tw * c), y * tw * c);
  }
  return { data: out, height: th, width: tw, channels: c, isByte: img.isByte };
}

/**
 * The four corners and the centre — **five pictures out of one.**
 *
 * What comes back is a list, not a picture, so `ToTensor` cannot simply follow
 * it. torchvision's own documentation says the same and hands the tuple on with
 * a `Lambda`. That is why `Lambda` exists in this file at all.
 */
export class FiveCrop implements Transform {
  private readonly th: number;
  private readonly tw: number;

  constructor(size: number | readonly [number, number]) {
    [this.th, this.tw] = pairOf(size);
  }

  apply(x: Subject): readonly Image[] {
    const img = asImage(x, "FiveCrop");
    const { height: h, width: w } = img;
    if (this.th > h || this.tw > w) {
      throw new RuntimeError(
        `The crop size (${this.th}, ${this.tw}) is larger than the image (${h}, ${w}).\n` +
        "(torch: Requested crop size is bigger than input size)");
    }
    const corners: readonly [number, number][] = [
      [0, 0], [0, w - this.tw], [h - this.th, 0], [h - this.th, w - this.tw],
    ];
    const out = corners.map(([top, left]) => cropAt(img, top, left, this.th, this.tw));
    // **The centre is `CenterCrop`'s, not another rounding written here.** The
    // halves land differently at odd sizes, and two roundings that agree today
    // are two places to fix on the day they stop.
    out.push(new CenterCrop([this.th, this.tw]).apply(img));
    return out;
  }

  describe(): string {
    return `FiveCrop(size=(${this.th}, ${this.tw}))`;
  }
}

/**
 * `FiveCrop`, and then five more from the flipped picture. Ten out of one.
 *
 * `verticalFlip` flips top to bottom instead of left to right.
 */
export class TenCrop implements Transform {
  private readonly th: number;
  private readonly tw: number;

  constructor(
    size: number | readonly [number, number],
    protected readonly verticalFlip = false,
  ) {
    [this.th, this.tw] = pairOf(size);
  }

  apply(x: Subject): readonly Image[] {
    const img = asImage(x, "TenCrop");
    const five = new FiveCrop([this.th, this.tw]);
    const turned = flipped(img, this.verticalFlip);
    return [...five.apply(img), ...five.apply(turned)];
  }

  describe(): string {
    return `TenCrop(size=(${this.th}, ${this.tw}), ` +
      `vertical_flip=${this.verticalFlip ? "True" : "False"})`;
  }
}

function isSeveral(x: Subject): x is readonly Image[] {
  return Array.isArray(x);
}

function asImage(x: Subject, who: string): Image {
  if (isSeveral(x)) {
    throw new RuntimeError(
      `${who} takes one picture — it received ${x.length} of them.\n` +
      "FiveCrop and TenCrop hand back several; take them apart with a Lambda first.");
  }
  if (x instanceof Tensor) {
    throw new Error(
      `${who} takes an (H,W,C) array — got a tensor.\n` +
        "Move ToTensor later in the pipeline: a tensor per image is a GPU buffer per image.",
    );
  }
  return x;
}


// ── Photometric: the arithmetic torchvision does **on the tensor path** ────
//
// **These are not PIL's numbers.** torchvision implements these five twice — the
// `ImageEnhance` path for PIL images and the arithmetic for tensors — and the two do not
// agree to the last bit. The Python side ported the second, and the golden compares
// against that, so this is the second too.
//
// **What was actually measured about precision.** The reason for all the `f32` here is
// written down, and the reason first written was wrong when measured.
//
// The live account: in a byte picture every blend **ends in a narrowing cast**, so the
// working precision picks the answer. Measured over 300 random 5×4 byte pictures × 4
// factors, computing in float64 and computing in float32 diverged at **910 of 72,000
// pixels (1.3%)**. They diverge by 1, which is **outside** the 1e-4 tolerance. So it is
// not decoration.
//
// And **a check sees it** — now. This paragraph turned over twice in one day, and leaving
// the marks of the turns is more useful than writing only the conclusion:
//
//   first     the Python side's comment said "measured, one pixel of
//             `adjust_saturation(1.7)`", and on that basis I wrote this path in float32.
//   measured  that case diverged at **0 pixels.** Replacing every `f32` with the identity
//             still passed all ten cases — what had not been caught was the factor, not
//             the picture.
//   now       that side moved the case to a factor of 0.1. With `f32` replaced by the
//             identity, **4 pixels are off by exactly 1.0** and it leaves the tolerance.
//
// So the deciding question is this: **does a narrowing cast follow the arithmetic.** If it
// does, the working precision picks the answer and no tolerance covers it. That fact and
// whether a check sees it are still **separate** — for a day there was the fact and no
// check.

const f32 = Math.fround;

/** The top of the range. **255 for bytes and 1 for a float picture** — a blend clips to
 *  it. */
function bound(isByte: boolean): number {
  return isByte ? 255 : 1;
}

/**
 * `ratio*a + (1-ratio)*b`, clipped back to the original dtype.
 *
 * `1 - ratio` is computed **in float64 first** (both are Python floats over there). The
 * narrowing happens the moment it multiplies a float32 array — and that ordering
 * separated 0 of 20 pixels from 20 of 20 in `toGray`.
 */
function blend(a: Image, b: ArrayLike<number>, ratio: number): Image {
  const hi = bound(a.isByte);
  const r = f32(ratio);
  const q = f32(1 - ratio);
  const out = new Float64Array(a.data.length);
  for (let i = 0; i < out.length; i++) {
    const v = f32(f32(r * f32(a.data[i] ?? 0)) + f32(q * f32(b[i] ?? 0)));
    const c = v < 0 ? 0 : v > hi ? hi : v;
    out[i] = a.isByte ? Math.trunc(c) : c;
  }
  return { ...a, data: out };
}

/** A byte into a [0,1] float. **It is torch's conversion, not an arbitrary division.** */
function toFloat01(img: Image): Float64Array {
  if (!img.isByte) return img.data;
  const out = new Float64Array(img.data.length);
  for (let i = 0; i < out.length; i++) out[i] = f32(f32(img.data[i] ?? 0) / 255);
  return out;
}

// torch multiplies by `256 - 1e-3` and truncates rather than rounding — the two methods
// diverge on about half the values, so it is copied exactly.
const BYTE_SCALE = f32(255 + 1 - 1e-3);

function fromFloat01(arr: Float64Array, img: Image): Image {
  const out = new Float64Array(arr.length);
  for (let i = 0; i < out.length; i++) {
    const v = f32(arr[i] ?? 0);
    out[i] = img.isByte ? Math.trunc(f32(v * BYTE_SCALE)) : v;
  }
  return { ...img, data: out };
}

/**
 * Pillow's algorithm — the one torchvision ported. **A pixel whose channels are equal is
 * handled outside the division**: a grey pixel has `maxc == minc`, so dividing by the
 * difference produces a NaN, and then it has to be found again.
 */
function rgb2hsv(src: Float64Array, n: number): Float64Array {
  const out = new Float64Array(n * 3);
  for (let i = 0; i < n; i++) {
    const r = f32(src[i * 3] ?? 0);
    const g = f32(src[i * 3 + 1] ?? 0);
    const b = f32(src[i * 3 + 2] ?? 0);
    const maxc = Math.max(r, g, b);
    const minc = Math.min(r, g, b);
    const eqc = maxc === minc;
    const cr = f32(maxc - minc);
    const s = f32(cr / (eqc ? 1 : maxc));
    const div = eqc ? 1 : cr;
    const rc = f32(f32(maxc - r) / div);
    const gc = f32(f32(maxc - g) / div);
    const bc = f32(f32(maxc - b) / div);
    // The Python side multiplies three masks and adds. Exactly one is 1 and the other
    // terms are 0, so the addition is exact and a branch gives the same number.
    let h: number;
    if (maxc === r) h = f32(bc - gc);
    else if (maxc === g) h = f32(f32(2 + rc) - bc);
    else h = f32(f32(4 + gc) - rc);
    out[i * 3] = f32(f32(f32(h / 6) + 1) % 1);
    out[i * 3 + 1] = s;
    out[i * 3 + 2] = maxc;
  }
  return out;
}

/**
 * The way back. torch picks the sextant with a one-hot mask and an einsum — here it is
 * picked by index and the numbers are the same. The einsum route is how you do it when
 * the gather also has to be differentiable.
 */
function hsv2rgb(hsv: Float64Array, n: number): Float64Array {
  const clip01 = (v: number): number => (v < 0 ? 0 : v > 1 ? 1 : v);
  const out = new Float64Array(n * 3);
  for (let i = 0; i < n; i++) {
    const h = hsv[i * 3] ?? 0;
    const s = hsv[i * 3 + 1] ?? 0;
    const v = hsv[i * 3 + 2] ?? 0;
    const six = f32(h * 6);
    const sextant = Math.floor(six);
    const f = f32(six - sextant);
    const idx = (((sextant | 0) % 6) + 6) % 6;
    const pp = clip01(f32(v * f32(1 - s)));
    const qq = clip01(f32(v * f32(1 - f32(s * f))));
    const tt = clip01(f32(v * f32(1 - f32(s * f32(1 - f)))));
    const R = [v, qq, pp, pp, tt, v];
    const G = [tt, v, v, qq, pp, pp];
    const B = [pp, pp, tt, v, v, qq];
    out[i * 3] = R[idx] ?? 0;
    out[i * 3 + 1] = G[idx] ?? 0;
    out[i * 3 + 2] = B[idx] ?? 0;
  }
  return out;
}

/** Toward black at 0, unchanged at 1, brighter above. */
export function adjustBrightness(img: Image, brightnessFactor: number): Image {
  if (brightnessFactor < 0) {
    throw new RuntimeError(
      `brightness_factor is not non-negative — got ${brightnessFactor}.\n` +
      `(torch: brightness_factor (${brightnessFactor}) is not non-negative.)`);
  }
  return blend(img, new Float64Array(img.data.length), brightnessFactor);
}

/**
 * Toward **the picture's own mean grey**, which is one number for the whole
 * image rather than one per channel or per pixel.
 */
export function adjustContrast(img: Image, contrastFactor: number): Image {
  if (contrastFactor < 0) {
    throw new RuntimeError(
      `contrast_factor is not non-negative — got ${contrastFactor}.\n` +
      `(torch: contrast_factor (${contrastFactor}) is not non-negative.)`);
  }
  const grey = toGray(img, 1, "adjust_contrast").data;
  let sum = 0;
  for (let i = 0; i < grey.length; i++) sum = f32(sum + f32(grey[i] ?? 0));
  const flat = new Float64Array(img.data.length).fill(f32(sum / grey.length));
  return blend(img, flat, contrastFactor);
}

/**
 * Toward grey, per pixel. **A one-channel picture comes back untouched** —
 * torchvision does that to match PIL, and it is a branch rather than an accident.
 */
export function adjustSaturation(img: Image, saturationFactor: number): Image {
  if (saturationFactor < 0) {
    throw new RuntimeError(
      `saturation_factor is not non-negative — got ${saturationFactor}.\n` +
      `(torch: saturation_factor (${saturationFactor}) is not non-negative.)`);
  }
  if (img.channels !== 3) return img;
  return blend(img, toGray(img, 3, "adjust_saturation").data, saturationFactor);
}

/**
 * Rotate the hue. **The only one that leaves RGB** — through HSV, add to the
 * angle, and back.
 *
 * `hueFactor` is a turn rather than degrees: 0.5 is half the wheel. A
 * one-channel picture comes back untouched, matching PIL.
 */
export function adjustHue(img: Image, hueFactor: number): Image {
  if (!(hueFactor >= -0.5 && hueFactor <= 0.5)) {
    throw new RuntimeError(
      `hue_factor is not in [-0.5, 0.5] — got ${hueFactor}.\n` +
      `(torch: hue_factor (${hueFactor}) is not in [-0.5, 0.5].)`);
  }
  if (img.channels !== 3) return img;
  const n = img.height * img.width;
  const hsv = rgb2hsv(toFloat01(img), n);
  for (let i = 0; i < n; i++) {
    hsv[i * 3] = f32(f32(f32((hsv[i * 3] ?? 0) + hueFactor) + 1) % 1);
  }
  return fromFloat01(hsv2rgb(hsv, n), img);
}

/**
 * `gain * x ** gamma`, clamped. Below 1 it lifts the shadows and above 1 it
 * deepens them — **the correction a display does**, which is why it is the one
 * here whose name is not a direction.
 */
export function adjustGamma(img: Image, gamma: number, gain = 1): Image {
  if (gamma < 0) {
    throw new RuntimeError(
      `gamma is not non-negative — got ${gamma}.\n` +
      "(torch: Gamma should be a non-negative real number)");
  }
  const src = toFloat01(img);
  const out = new Float64Array(src.length);
  for (let i = 0; i < out.length; i++) {
    const v = f32(gain * f32(Math.pow(f32(src[i] ?? 0), gamma)));
    out[i] = v < 0 ? 0 : v > 1 ? 1 : v;
  }
  return fromFloat01(out, img);
}

/** `null` means it is not used, and that prints as `None`. */
type Span = readonly [number, number] | null;

function checkSpan(
  value: number | readonly [number, number] | undefined,
  name: string,
  center: number,
  lo: number,
  hi: number,
  clipFirstOnZero: boolean,
): Span {
  let pair: [number, number];
  if (value === undefined) {
    pair = [center, center];
  } else if (typeof value === "number") {
    if (value < 0) {
      throw new RuntimeError(
        `${name} as a single number must be non-negative — got ${value}.\n` +
        `(torch: If ${name} is a single number, it must be non negative.)`);
    }
    pair = [center - value, center + value];
    if (clipFirstOnZero) pair[0] = Math.max(pair[0], 0);
  } else if (Array.isArray(value) && value.length === 2) {
    pair = [Number(value[0]), Number(value[1])];
  } else {
    throw new RuntimeError(
      `${name} is a single number or a pair — got ${JSON.stringify(value)}.\n` +
      `(torch: ${name} should be a single number or a list/tuple with length 2.)`);
  }
  if (!(lo <= pair[0] && pair[0] <= pair[1] && pair[1] <= hi)) {
    throw new RuntimeError(
      `${name} values should be between (${lo}, ${hi}), but got [${pair[0]}, ${pair[1]}].\n` +
      `(torch: ${name} values should be between (${lo}, ${hi}), but got [${pair[0]}, ${pair[1]}].)`);
  }
  // **The identity is stored as `None` rather than as a range that does nothing.**
  // Applied anyway it costs one blend, and on a byte picture one rounding — "no jitter"
  // and "a jitter of exactly 1" are different things.
  return pair[0] === pair[1] && pair[0] === center ? null : [pair[0], pair[1]];
}

function spanText(s: Span): string {
  return s === null ? "None" : `(${s[0]}, ${s[1]})`;
}

/**
 * Brightness, contrast, saturation and hue, each drawn from a range — **and the
 * four applied in a drawn order.**
 *
 * The order is part of the draw and not a detail: brightness then contrast is
 * not contrast then brightness, because contrast measures the picture's mean and
 * brightness has already moved it.
 *
 * A single number `b` means the range `[1-b, 1+b]` (hue is centred on 0
 * instead). A range that comes out as exactly the identity is **turned off
 * rather than applied** — torchvision stores nothing there, which is why the
 * repr shows `None` for anything left at its default.
 */
export class ColorJitter implements Transform {
  protected readonly brightness: Span;
  protected readonly contrast: Span;
  protected readonly saturation: Span;
  protected readonly hue: Span;

  constructor(
    brightness?: number | readonly [number, number],
    contrast?: number | readonly [number, number],
    saturation?: number | readonly [number, number],
    hue?: number | readonly [number, number],
  ) {
    this.brightness = checkSpan(brightness, "brightness", 1, 0, Infinity, true);
    this.contrast = checkSpan(contrast, "contrast", 1, 0, Infinity, true);
    this.saturation = checkSpan(saturation, "saturation", 1, 0, Infinity, true);
    this.hue = checkSpan(hue, "hue", 0, -0.5, 0.5, false);
  }

  /**
   * The four factors and **the order to apply them in.** Kept separate because
   * it is the only part that draws.
   */
  getParams(): [number[], number | null, number | null, number | null, number | null] {
    const order = [0, 1, 2, 3];
    for (let i = order.length - 1; i > 0; i--) {
      const j = nextInt(i + 1);
      const a = order[i] ?? 0, b = order[j] ?? 0;
      order[i] = b; order[j] = a;
    }
    const draw = (s: Span): number | null => (s === null ? null : uniform(s[0], s[1]));
    return [order, draw(this.brightness), draw(this.contrast),
      draw(this.saturation), draw(this.hue)];
  }

  apply(x: Subject): Image {
    let img = asImage(x, "ColorJitter");
    const [order, brightness, contrast, saturation, hue] = this.getParams();
    for (const which of order) {
      if (which === 0 && brightness !== null) img = adjustBrightness(img, brightness);
      else if (which === 1 && contrast !== null) img = adjustContrast(img, contrast);
      else if (which === 2 && saturation !== null) img = adjustSaturation(img, saturation);
      else if (which === 3 && hue !== null) img = adjustHue(img, hue);
    }
    return img;
  }

  describe(): string {
    return `ColorJitter(brightness=${spanText(this.brightness)}` +
      `, contrast=${spanText(this.contrast)}` +
      `, saturation=${spanText(this.saturation)}` +
      `, hue=${spanText(this.hue)})`;
  }
}

// ── transforms.functional ──────────────────────────────────────────────
//
// **The same arithmetic called without standing an object up.** Everything here hands the
// work to a class above or to a helper that class uses — nothing is reimplemented. That
// is the point: two copies of `Resize`'s filter agree on the day they are written and
// disagree on some later one.
//
// torchvision divides it the other way — the class calls the function. It is reversed here
// because the classes came first, and turning running code inside out to match the shape
// of a call graph is fixing something nobody outside can see. (The Python side is the same
// direction for the same reason.)
//
// **What it is for**: a tutorial writes `F.hflip(x)` as often as `RandomHorizontalFlip()`,
// and until now that line stopped with an error about a namespace that did not exist.

/** Left to right, with no draw in it. */
export function hflip(img: Image): Image {
  return flipped(asImage(img, "hflip"), false);
}

/** Top to bottom, with no draw in it. */
export function vflip(img: Image): Image {
  return flipped(asImage(img, "vflip"), true);
}

/**
 * **Not `RandomCrop` without the draw** — the position is given, and it may
 * hang off the edge, which is torchvision's behaviour rather than an oversight.
 */
export function crop(img: Image, top: number, left: number,
                     height: number, width: number): Image {
  return cropAt(asImage(img, "crop"), top, left, height, width);
}

export function centerCrop(img: Image, outputSize: number | readonly [number, number]): Image {
  return new CenterCrop(outputSize).apply(img);
}

export function resize(
  img: Image,
  size: number | readonly [number, number],
  interpolation: "bilinear" | "nearest" | "bicubic" = "bilinear",
  maxSize: number | null = null,
  antialias = true,
): Image {
  return new Resize(size, interpolation, maxSize, antialias).apply(img);
}

/** Crop, then resize what was cropped. `RandomResizedCrop` without the draw. */
export function resizedCrop(
  img: Image, top: number, left: number, height: number, width: number,
  size: number | readonly [number, number],
  interpolation: "bilinear" | "nearest" | "bicubic" = "bilinear",
  antialias = true,
): Image {
  return resize(crop(img, top, left, height, width), size, interpolation, null, antialias);
}

export function pad(
  img: Image,
  padding: number | readonly number[],
  fill: number | readonly number[] = 0,
  paddingMode: PaddingMode = "constant",
): Image {
  return new Pad(padding, fill, paddingMode).apply(img);
}

export function rgbToGrayscale(img: Image, numOutputChannels = 1): Image {
  return toGray(asImage(img, "rgb_to_grayscale"), numOutputChannels, "rgb_to_grayscale");
}

export function toTensor(pic: Image): Tensor {
  return new ToTensor().apply(pic);
}

/**
 * The five corners and the centre, as `FiveCrop` gives them.
 *
 * **These three were missing while their classes were here**, which is the
 * shape a name-only reading of a gap keeps producing: `FiveCrop`, `TenCrop`
 * and `Grayscale` all worked, and `F.five_crop(x, 32)` — the line a tutorial
 * writes — stopped at a name that was not there.
 *
 * Nothing caught it. `tests/ts_axis.py` leaves `transforms` out on the grounds
 * that the golden's `vision::` cases hold it name by name, and for these three
 * the golden had no case, so each check was waiting on the other.
 */
export function fiveCrop(
  img: Image, size: number | readonly [number, number],
): readonly Image[] {
  return new FiveCrop(size).apply(img);
}

/** `fiveCrop` of the image and of its flip — ten in the order torchvision returns. */
export function tenCrop(
  img: Image, size: number | readonly [number, number], verticalFlip = false,
): readonly Image[] {
  return new TenCrop(size, verticalFlip).apply(img);
}

/**
 * The same conversion `rgbToGrayscale` does.
 *
 * **torchvision keeps both names** — this one belongs to the PIL path and
 * `rgbToGrayscale` to the tensor path. They compute the same thing here, and
 * the second name is kept because tutorials written against PIL call it. The
 * Python side says the same in the same words.
 */
export function toGrayscale(img: Image, numOutputChannels = 1): Image {
  return rgbToGrayscale(img, numOutputChannels);
}

export function normalize(tensor: Tensor, mean: readonly number[],
                          std: readonly number[]): Tensor {
  return new Normalize(mean, std).apply(tensor);
}

/**
 * Blank out a rectangle of a `(...,C,H,W)` tensor. **The position is given**,
 * where `RandomErasing` draws it.
 */
export function erase(img: Tensor, i: number, j: number,
                      h: number, w: number, v: Tensor): Tensor {
  // **`i, j, h, w` are torchvision's names**, and all four collided with something the
  // body already had: `i` with the loop counter, `h` and `w` with the tensor's own
  // height and width. The arguments keep the outside names — those are what a caller
  // writes — and the locals yield, because nothing outside this function reads them.
  const shape = img.shape;
  if (shape.length < 3) {
    throw new RuntimeError(
      `erase takes (...,C,H,W) — it received (${shape.join(", ")}).`);
  }
  const rows = shape[shape.length - 2] ?? 1;
  const cols = shape[shape.length - 1] ?? 1;
  const n = img.size;
  const mask = new Float64Array(n);
  for (let at = 0; at < n; at++) {
    const col = at % cols;
    const row = Math.floor(at / cols) % rows;
    if (row >= i && row < i + h && col >= j && col < j + w) mask[at] = 1;
  }
  // **`v` is lifted into place at full size.** A different value can arrive per
  // position, so it must not be reduced to a single constant — the golden's case happens
  // to be a constant so writing it that way passes, and that is the case being unable to
  // tell the difference rather than the code being right. `where` does not broadcast, so
  // the shapes have to match exactly.
  const rank = v.shape.length;
  const placed = v.pad(rank - 2, i, rows - i - h)
    .pad(rank - 1, j, cols - j - w);
  if (placed.size !== n) {
    throw new RuntimeError(
      `erase: v padded to (${placed.shape.join(", ")}) does not match the image ` +
      `(${shape.join(", ")}).`);
  }
  return placed.reshape(shape).where(Tensor.from(mask, shape), img);
}

/** `[C, H, W]` — **h before w**, unlike `getImageSize` just below. */
export function getDimensions(img: Image): [number, number, number] {
  const i = asImage(img, "get_dimensions");
  return [i.channels, i.height, i.width];
}

/** `[W, H]` — **width first.** torchvision's order, and the odd one out here. */
export function getImageSize(img: Image): [number, number] {
  const i = asImage(img, "get_image_size");
  return [i.width, i.height];
}

export function getImageNumChannels(img: Image): number {
  return asImage(img, "get_image_num_channels").channels;
}


// ── Six pixel operations, and the six that wrap them in a probability ──
//
// There is no geometry — no pixel moves and only the values change. So they are kept apart
// from the ones needing a grid resample, and their checks stand apart too.

/**
 * `bound - x`. White for black — and the bound is 255 or 1 depending on the
 * dtype, which is the whole of it and the whole of what goes wrong.
 */
export function invert(img: Image): Image {
  const src = asImage(img, "invert");
  const hi = bound(src.isByte);
  const out = new Float64Array(src.data.length);
  for (let i = 0; i < out.length; i++) {
    const v = f32(hi - f32(src.data[i] ?? 0));
    out[i] = src.isByte ? Math.trunc(v) : v;
  }
  return { ...src, data: out };
}

/**
 * Keep the top `bits` of each byte and zero the rest — **fewer colours, by
 * masking rather than by rounding.** uint8 only: a float image has no bits to
 * throw away.
 */
export function posterize(img: Image, bits: number): Image {
  const src = asImage(img, "posterize");
  if (!src.isByte) {
    throw new RuntimeError(
      "posterize takes a uint8 image — it received a float one.\n" +
      "  It throws away the low bits of a byte, and a float image has no bits\n" +
      "  to throw away.\n" +
      "(torch: Only torch.uint8 image tensors are supported)");
  }
  const mask = (-(2 ** (8 - bits))) & 0xff;
  const out = new Float64Array(src.data.length);
  for (let i = 0; i < out.length; i++) out[i] = (src.data[i] ?? 0) & mask;
  return { ...src, data: out };
}

/**
 * Invert **only the pixels at or above** the threshold. Below it nothing
 * happens, so the picture comes back part positive and part negative.
 */
export function solarize(img: Image, threshold: number): Image {
  const src = asImage(img, "solarize");
  const hi = bound(src.isByte);
  if (threshold > hi) {
    throw new RuntimeError(
      `threshold ${threshold} is above this image's bound ${hi}.\n` +
      "(torch: Threshold should be less than bound of img.)");
  }
  const flipped2 = invert(src);
  const out = new Float64Array(src.data.length);
  for (let i = 0; i < out.length; i++) {
    const v = src.data[i] ?? 0;
    out[i] = v >= threshold ? (flipped2.data[i] ?? 0) : v;
  }
  return { ...src, data: out };
}

/**
 * Stretch each channel to fill the range. **Per channel and not per picture** —
 * a channel that is already flat is left alone rather than divided by zero.
 */
export function autocontrast(img: Image): Image {
  const src = asImage(img, "autocontrast");
  const hiBound = bound(src.isByte);
  const c = src.channels;
  const n = src.data.length;
  const out = new Float64Array(n);
  for (let k = 0; k < c; k++) {
    let lo = Infinity, hi = -Infinity;
    for (let i = k; i < n; i += c) {
      const v = src.data[i] ?? 0;
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
    // A flat channel gives `bound / 0`, and the Python side picks those out **after
    // dividing**, as the non-finite positions. Here it branches before dividing and gives
    // the same answer.
    const flat = hi === lo;
    const shift = flat ? 0 : f32(lo);
    const scale = flat ? 1 : f32(hiBound / f32(hi - lo));
    for (let i = k; i < n; i += c) {
      let v = f32(f32(f32(src.data[i] ?? 0) - shift) * scale);
      v = v < 0 ? 0 : v > hiBound ? hiBound : v;
      out[i] = src.isByte ? Math.trunc(v) : v;
    }
  }
  return { ...src, data: out };
}

/**
 * Histogram equalisation of one channel, **in torch's integer arithmetic verbatim.**
 *
 * Every division here truncates, and the table shifted by one (`[0, ...lut[:-1]]`) sends
 * the darkest value to 0 rather than to the first step. Written in floating point it is
 * off by 1 across most of the range.
 */
function equalizeChannel(plane: Float64Array): Float64Array {
  const hist = new Int32Array(256);
  for (let i = 0; i < plane.length; i++) hist[plane[i] ?? 0]! += 1;
  const nonzero: number[] = [];
  for (let v = 0; v < 256; v++) if (hist[v] !== 0) nonzero.push(hist[v] ?? 0);
  let tail = 0;
  for (let i = 0; i < nonzero.length - 1; i++) tail += nonzero[i] ?? 0;
  const step = Math.floor(tail / 255);
  if (step === 0) return plane;
  const lut = new Int32Array(256);
  let running = 0;
  const half = Math.floor(step / 2);
  const cum = new Int32Array(256);
  for (let v = 0; v < 256; v++) {
    running += hist[v] ?? 0;
    cum[v] = Math.floor((running + half) / step);
  }
  for (let v = 1; v < 256; v++) {
    const t = cum[v - 1] ?? 0;
    lut[v] = t < 0 ? 0 : t > 255 ? 255 : t;
  }
  const out = new Float64Array(plane.length);
  for (let i = 0; i < out.length; i++) out[i] = lut[plane[i] ?? 0] ?? 0;
  return out;
}

/**
 * Flatten the histogram, per channel. **uint8 only**, for torchvision's reason:
 * it counts 256 bins.
 */
export function equalize(img: Image): Image {
  const src = asImage(img, "equalize");
  if (!src.isByte) {
    throw new RuntimeError(
      "equalize takes a uint8 image — it received a float one.\n" +
      "  It counts a 256-bin histogram, and a float image has no bins.\n" +
      "(torch: Only torch.uint8 image tensors are supported)");
  }
  const c = src.channels;
  const n = src.data.length;
  const out = new Float64Array(n);
  for (let k = 0; k < c; k++) {
    const plane = new Float64Array(n / c);
    for (let i = k, j = 0; i < n; i += c, j++) plane[j] = src.data[i] ?? 0;
    const done = equalizeChannel(plane);
    for (let i = k, j = 0; i < n; i += c, j++) out[i] = done[j] ?? 0;
  }
  return { ...src, data: out };
}

/**
 * The 3×3 smoothing `adjustSharpness` blends towards — **eight 1s with a 5 in the middle,
 * divided by 13.**
 *
 * The border is left alone. torchvision convolves without padding and writes the result
 * back into the middle, so the outer ring is the original — copied rather than tidied, and
 * a padded convolution gives a different number there while the difference is invisible in
 * the middle.
 */
function blurred(img: Image): Image {
  const k = [1, 1, 1, 1, 5, 1, 1, 1, 1].map((v) => f32(v / 13));
  const { height: h, width: w, channels: c } = img;
  const out = Float64Array.from(img.data);
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      for (let ch = 0; ch < c; ch++) {
        let acc = 0;
        for (let dy = 0; dy < 3; dy++) {
          for (let dx = 0; dx < 3; dx++) {
            acc = f32(acc + f32((k[dy * 3 + dx] ?? 0)
              * f32(img.data[((y - 1 + dy) * w + (x - 1 + dx)) * c + ch] ?? 0)));
          }
        }
        // **Rounding rather than truncating.** torch passes through `round` when
        // returning a convolution to an integer dtype, and truncating is one step low on
        // about half the pixels (measured on the Python side).
        out[(y * w + x) * c + ch] = img.isByte
          ? Math.min(Math.max(Math.round(acc), 0), bound(true))
          : acc;
      }
    }
  }
  return { ...img, data: out };
}

/**
 * Blur at 0, unchanged at 1, sharper above — the blend `ImageEnhance` calls
 * sharpness. **A picture two pixels wide or shorter comes back untouched**,
 * because there is no middle to convolve.
 */
export function adjustSharpness(img: Image, sharpnessFactor: number): Image {
  if (sharpnessFactor < 0) {
    throw new RuntimeError(
      `sharpness_factor is not non-negative — got ${sharpnessFactor}.\n` +
      `(torch: sharpness_factor (${sharpnessFactor}) is not non-negative.)`);
  }
  const src = asImage(img, "adjust_sharpness");
  if (src.height <= 2 || src.width <= 2) return src;
  return blend(src, blurred(src).data, sharpnessFactor);
}

/** What the six wrappers share: one probability and one call. */
abstract class RandomPixelOp implements Transform {
  constructor(private readonly who: string, protected readonly p = 0.5) {}

  protected abstract run(img: Image): Image;

  apply(x: Subject): Image {
    const img = asImage(x, this.who);
    if (nextFloat() >= this.p) return img;
    return this.run(img);
  }

  describe(): string {
    return `${this.who}(p=${this.p})`;
  }
}

export class RandomInvert extends RandomPixelOp {
  constructor(p = 0.5) { super("RandomInvert", p); }
  protected run(img: Image): Image { return invert(img); }
}

export class RandomAutocontrast extends RandomPixelOp {
  constructor(p = 0.5) { super("RandomAutocontrast", p); }
  protected run(img: Image): Image { return autocontrast(img); }
}

export class RandomEqualize extends RandomPixelOp {
  constructor(p = 0.5) { super("RandomEqualize", p); }
  protected run(img: Image): Image { return equalize(img); }
}

export class RandomPosterize extends RandomPixelOp {
  constructor(protected readonly bits: number, p = 0.5) { super("RandomPosterize", p); }
  protected run(img: Image): Image { return posterize(img, this.bits); }
  // **There is no space after the comma.** That is torchvision's own notation rather
  // than something dropped in transcription — three of these six print that way and the
  // other three have only one field.
  override describe(): string { return `RandomPosterize(bits=${this.bits},p=${this.p})`; }
}

export class RandomSolarize extends RandomPixelOp {
  constructor(protected readonly threshold: number, p = 0.5) { super("RandomSolarize", p); }
  protected run(img: Image): Image { return solarize(img, this.threshold); }
  override describe(): string {
    return `RandomSolarize(threshold=${this.threshold},p=${this.p})`;
  }
}

export class RandomAdjustSharpness extends RandomPixelOp {
  constructor(protected readonly sharpnessFactor: number, p = 0.5) {
    super("RandomAdjustSharpness", p);
  }
  protected run(img: Image): Image { return adjustSharpness(img, this.sharpnessFactor); }
  override describe(): string {
    return `RandomAdjustSharpness(sharpness_factor=${this.sharpnessFactor},p=${this.p})`;
  }
}


// ── Resampling on a grid: the ones that read BETWEEN pixels ──────────
//
// Everything above moves pixels (crop, flip, pad) or rewrites their values
// (photometric, pixel ops). Everything below reads the input between them.
// Three conventions decide every value, and **each one is invisible when it is
// wrong** — the picture comes out looking slightly soft rather than looking
// broken, so none of them announces itself.

/**
 * torch's `grid_sample` with `align_corners=false` and zero padding, on `(H,W,C)`.
 *
 * **`align_corners=false` is the whole of the coordinate convention** and not a
 * detail: it puts -1 and 1 at the *outer edges* of the border pixels rather than
 * at their centres, so the un-normalising is `((g + 1) * size - 1) / 2`. With the
 * other convention every resampled pixel is half a pixel out, which looks like a
 * slightly soft image rather than like a bug.
 */
function gridSample(img: Image, grid: Float64Array, oh: number, ow: number,
                    mode: "bilinear" | "nearest"): Float64Array {
  const { height: h, width: w, channels: c } = img;
  const out = new Float64Array(oh * ow * c);
  // The input at integer positions, **zero outside** — torch's `padding_mode`.
  const read = (yy: number, xx: number, k: number): number =>
    (xx >= 0 && xx < w && yy >= 0 && yy < h) ? (img.data[(yy * w + xx) * c + k] ?? 0) : 0;

  for (let i = 0; i < oh * ow; i++) {
    const gx = grid[i * 2] ?? 0;
    const gy = grid[i * 2 + 1] ?? 0;
    const x = f32(f32(f32(f32(gx + 1) * w) - 1) / 2);
    const y = f32(f32(f32(f32(gy + 1) * h) - 1) / 2);
    if (mode === "nearest") {
      // **Half goes to even** — numpy's `rint`, torch's `nearbyint`. `floor(x+0.5)`
      // disagrees, and the disagreement shows only at positions landing exactly
      // halfway, which a 90-degree rotation produces on every pixel. Measured: it
      // is the ONLY one of the three nearest-mode cases that can see this.
      const yy = roundHalfToEven(y), xx = roundHalfToEven(x);
      for (let k = 0; k < c; k++) out[i * c + k] = read(yy, xx, k);
      continue;
    }
    const x0 = Math.floor(x), y0 = Math.floor(y);
    const fx = f32(x - x0), fy = f32(y - y0);
    const wts: [number, number, number][] = [
      [y0, x0, f32(f32(1 - fy) * f32(1 - fx))],
      [y0, x0 + 1, f32(f32(1 - fy) * fx)],
      [y0 + 1, x0, f32(fy * f32(1 - fx))],
      [y0 + 1, x0 + 1, f32(fy * fx)],
    ];
    for (let k = 0; k < c; k++) {
      let acc = 0;
      for (const [yy, xx, weight] of wts) acc = f32(acc + f32(read(yy, xx, k) * weight));
      out[i * c + k] = acc;
    }
  }
  return out;
}

/**
 * Sample, and paint the outside with `fill`.
 *
 * **The mask is sampled alongside the picture** rather than the outside being
 * computed. torchvision appends a channel of ones, resamples it with everything
 * else, and reads the result as "how much of this pixel came from inside" — so a
 * bilinear edge pixel is a blend of picture and fill in the same proportion the
 * interpolation used. Deciding inside-ness from the coordinates instead gives a
 * hard edge that is wrong by up to one whole pixel: measured at 18 pixels of
 * `rotate(filled)`, up to 0.305 out.
 */
function gridTransform(img: Image, grid: Float64Array, oh: number, ow: number,
                       mode: "bilinear" | "nearest",
                       fill: readonly number[] | null): Image {
  const c = img.channels;
  const sampled = gridSample(img, grid, oh, ow, mode);
  let painted = sampled;
  if (fill !== null) {
    const ones: Image = {
      data: new Float64Array(img.height * img.width).fill(1),
      height: img.height, width: img.width, channels: 1, isByte: img.isByte,
    };
    const mask = gridSample(ones, grid, oh, ow, mode);
    const values = fill.length === 1
      ? new Array<number>(c).fill(f32(fill[0] ?? 0))
      : fill.map((v) => f32(v));
    painted = new Float64Array(sampled.length);
    for (let i = 0; i < oh * ow; i++) {
      const m = mask[i] ?? 0;
      for (let k = 0; k < c; k++) {
        const v = values[k] ?? 0;
        painted[i * c + k] = mode === "nearest"
          ? (m < 0.5 ? v : (sampled[i * c + k] ?? 0))
          : f32(f32((sampled[i * c + k] ?? 0) * m) + f32(f32(1 - m) * v));
      }
    }
  }
  const out = new Float64Array(painted.length);
  const hi = bound(img.isByte);
  for (let i = 0; i < out.length; i++) {
    const v = painted[i] ?? 0;
    out[i] = img.isByte ? Math.min(Math.max(Math.round(v), 0), hi) : v;
  }
  return { data: out, height: oh, width: ow, channels: c, isByte: img.isByte };
}

/**
 * The output's pixel centres, mapped back through `matrix`, in [-1,1].
 *
 * The half-pixel offset (`0.5`) is torch's, and it is what makes the grid line
 * up with pixel centres rather than corners. Removing it moves every one of the
 * sixteen rotate/affine cases (measured).
 */
function affineGrid(matrix: readonly number[], w: number, h: number,
                    ow: number, oh: number): Float64Array {
  const t = matrix.map((v) => f32(v));
  const sx = f32(0.5 * w), sy = f32(0.5 * h);
  const grid = new Float64Array(oh * ow * 2);
  for (let j = 0; j < oh; j++) {
    // `linspace(-oh/2 + 0.5, oh/2 + 0.5 - 1, oh)` steps by exactly 1, so it is
    // the start plus `j`.
    const y = f32(f32(-oh * 0.5 + 0.5) + j);
    for (let i = 0; i < ow; i++) {
      const x = f32(f32(-ow * 0.5 + 0.5) + i);
      const gx = f32(f32(f32(x * (t[0] ?? 0)) + f32(y * (t[1] ?? 0))) + (t[2] ?? 0));
      const gy = f32(f32(f32(x * (t[3] ?? 0)) + f32(y * (t[4] ?? 0))) + (t[5] ?? 0));
      grid[(j * ow + i) * 2] = f32(gx / sx);
      grid[(j * ow + i) * 2 + 1] = f32(gy / sy);
    }
  }
  return grid;
}

/**
 * The six numbers, **inverted** — the grid maps output positions back to input
 * ones, so what goes in is the inverse of the transform being described.
 */
/**
 * **Exported for the v2 warp kernels next door, and for nothing else.**
 *
 * `ops.ts` builds `affineBoundingBoxes` and its eleven siblings out of these, and the
 * alternative was a second copy of the matrix formula there — which is the one thing
 * this repository keeps catching in its own tables. The cost is that these three appear
 * in the published API reference, where a reader will not want them; that is smaller
 * than two bodies for one transform.
 */
export function inverseAffineMatrix(center: readonly [number, number], angle: number,
                             translate: readonly [number, number], scale: number,
                             shear: readonly [number, number]): number[] {
  const rad = (d: number): number => (d * Math.PI) / 180;
  const rot = rad(angle), sx = rad(shear[0]), sy = rad(shear[1]);
  const [cx, cy] = center;
  const [tx, ty] = translate;
  const a = Math.cos(rot - sy) / Math.cos(sy);
  const b = -Math.cos(rot - sy) * Math.tan(sx) / Math.cos(sy) - Math.sin(rot);
  const cc = Math.sin(rot - sy) / Math.cos(sy);
  const d = -Math.sin(rot - sy) * Math.tan(sx) / Math.cos(sy) + Math.cos(rot);
  const m = [d, -b, 0, -cc, a, 0].map((v) => v / scale);
  m[2] = (m[2] ?? 0) + (m[0] ?? 0) * (-cx - tx) + (m[1] ?? 0) * (-cy - ty) + cx;
  m[5] = (m[5] ?? 0) + (m[3] ?? 0) * (-cx - tx) + (m[4] ?? 0) * (-cy - ty) + cy;
  return m;
}

/**
 * How big the picture has to be to hold the whole rotated one — `expand=true`.
 *
 * **The truncation to 1e-4 is carried rather than justified.** torchvision's
 * comment says it avoids ceiling a corner at 1e-15 up to a whole pixel, and the
 * Python side could not reproduce that: sweeping 36 picture sizes by 360 whole
 * degrees, the answer is the same with the truncation and without it, every
 * time. It is here because removing it would be a change to a ported formula on
 * the strength of one sweep, not because a case was found.
 *
 * What the sizes are is measured, and it is not the obvious thing: a quarter
 * turn of a 5x4 picture comes out **5 tall by 6 wide**, not 4x5. Deriving it
 * from the geometry gives the wrong answer.
 */
export function affineOutputSize(matrix: readonly number[], w: number, h: number): [number, number] {
  const t = matrix.map((v) => f32(v));
  const pts: [number, number][] = [
    [f32(-0.5 * w), f32(-0.5 * h)], [f32(-0.5 * w), f32(0.5 * h)],
    [f32(0.5 * w), f32(0.5 * h)], [f32(0.5 * w), f32(-0.5 * h)],
  ];
  let loX = Infinity, loY = Infinity, hiX = -Infinity, hiY = -Infinity;
  for (const [px, py] of pts) {
    const mx = f32(f32(f32(px * (t[0] ?? 0)) + f32(py * (t[1] ?? 0))) + (t[2] ?? 0));
    const my = f32(f32(f32(px * (t[3] ?? 0)) + f32(py * (t[4] ?? 0))) + (t[5] ?? 0));
    loX = Math.min(loX, mx); hiX = Math.max(hiX, mx);
    loY = Math.min(loY, my); hiY = Math.max(hiY, my);
  }
  const TOL = 1e-4;
  const span = (lo: number, hi: number, size: number): number => {
    const l = f32(lo + f32(size * 0.5)), r = f32(hi + f32(size * 0.5));
    return Math.ceil(Math.trunc(r / TOL) * TOL) - Math.floor(Math.trunc(l / TOL) * TOL);
  };
  return [span(loX, hiX, w), span(loY, hiY, h)];
}

/**
 * torch's centre convention: **(0, 0) is the middle of the picture**, so an
 * explicit centre arrives as an offset from it rather than as a pixel position. The default is `[0, 0]` and
 * not `[w/2, h/2]` — the grid is already centred, and passing the middle as a
 * centre shifts the picture by half its own size.
 */
function centerOffset(center: readonly [number, number] | null,
                      w: number, h: number): [number, number] {
  if (center === null) return [0, 0];
  return [center[0] - w * 0.5, center[1] - h * 0.5];
}

export function shearPair(shear: number | readonly number[]): [number, number] {
  if (typeof shear === "number") return [shear, 0];
  if (shear.length === 1) return [shear[0] ?? 0, shear[0] ?? 0];
  if (shear.length !== 2) {
    throw new RuntimeError(
      `shear is one or two numbers — it received ${shear.length}.\n` +
      "(torch: Shear should be a sequence containing two values.)");
  }
  return [shear[0] ?? 0, shear[1] ?? 0];
}

/**
 * Turn the picture about its centre. **Counter-clockwise for a positive angle**,
 * which is PIL's direction and the opposite of what a screen's y-axis suggests.
 *
 * `expand` grows the output to hold the whole rotated picture; without it the
 * corners go outside and are lost.
 */
export function rotate(
  img: Image,
  angle: number,
  interpolation: "bilinear" | "nearest" = "nearest",
  expand = false,
  center: readonly [number, number] | null = null,
  fill: number | readonly number[] | null = null,
): Image {
  const src = asImage(img, "rotate");
  const { height: h, width: w } = src;
  // **The angle is negated on the way in**, and torchvision's own comment says
  // why: `rotate` and `affine` disagree about which way is positive, and the
  // negation here is what makes them agree from outside.
  const matrix = inverseAffineMatrix(centerOffset(center, w, h), -angle, [0, 0], 1, [0, 0]);
  const [ow, oh] = expand ? affineOutputSize(matrix, w, h) : [w, h];
  const grid = affineGrid(matrix, w, h, ow, oh);
  return gridTransform(src, grid, oh, ow, interpolation, fillList(fill));
}

/**
 * Rotate, shear, scale and shift in one resampling. **One pass and not four** —
 * four would interpolate four times and blur what a single grid keeps sharp.
 */
export function affine(
  img: Image,
  angle: number,
  translate: readonly [number, number],
  scale: number,
  shear: number | readonly number[],
  interpolation: "bilinear" | "nearest" = "nearest",
  fill: number | readonly number[] | null = null,
  center: readonly [number, number] | null = null,
): Image {
  const src = asImage(img, "affine");
  if (scale <= 0) {
    throw new RuntimeError(
      `scale is a positive number — got ${scale}.\n` +
      "(torch: Argument scale should be positive)");
  }
  const { height: h, width: w } = src;
  const matrix = inverseAffineMatrix(centerOffset(center, w, h), angle,
    [translate[0], translate[1]], scale, shearPair(shear));
  const grid = affineGrid(matrix, w, h, w, h);
  return gridTransform(src, grid, h, w, interpolation, fillList(fill));
}

function fillList(fill: number | readonly number[] | null): readonly number[] | null {
  if (fill === null) return null;
  return typeof fill === "number" ? [fill] : fill;
}


/** A number `d` means `[-d, d]`; a pair is taken as it is. */
function setupAngle(x: number | readonly number[], name: string): [number, number] {
  if (typeof x === "number") {
    if (x < 0) {
      throw new RuntimeError(
        `${name} as a single number must be positive — got ${x}.\n` +
        `(torch: If ${name} is a single number, it must be positive.)`);
    }
    return [-x, x];
  }
  if (x.length !== 2) {
    throw new RuntimeError(
      `${name} is a single number or a pair — got ${x.length} numbers.\n` +
      `(torch: ${name} should be a sequence of length 2.)`);
  }
  return [x[0] ?? 0, x[1] ?? 0];
}

/** Python's list spelling — `[-30.0, 30.0]`, which is how `repr` prints these. */
function floatList(v: readonly number[]): string {
  return `[${v.map(pyFloat).join(", ")}]`;
}

/**
 * A turn drawn from `degrees`.
 *
 * **The fill is spelled per channel before the call**, because torchvision does
 * that in its `forward` and not in `rotate` — a single number there becomes one
 * per channel, and passing it through undone gives a different picture on a
 * three-channel image.
 */
export class RandomRotation implements Transform {
  protected readonly degrees: [number, number];

  constructor(
    degrees: number | readonly number[],
    protected readonly interpolation: "bilinear" | "nearest" = "nearest",
    protected readonly expand = false,
    private readonly center: readonly [number, number] | null = null,
    protected readonly fill: number | readonly number[] | null = 0,
  ) {
    this.degrees = setupAngle(degrees, "degrees");
  }

  getParams(): number {
    return uniform(this.degrees[0], this.degrees[1]);
  }

  apply(x: Subject): Image {
    const img = asImage(x, "RandomRotation");
    const fill = this.fill === null ? null
      : typeof this.fill === "number"
        ? new Array<number>(img.channels).fill(this.fill)
        : [...this.fill];
    return rotate(img, this.getParams(), this.interpolation, this.expand,
      this.center, fill);
  }

  describe(): string {
    // **`center` and `fill` are printed only when they are set**, which is
    // torchvision's own shape here and not the same rule as `RandomAffine`'s two
    // classes down — that one drops a field when it equals its default, and this
    // one drops it when it is null.
    let out = `RandomRotation(degrees=${floatList(this.degrees)}` +
      `, interpolation=${this.interpolation}, expand=${this.expand ? "True" : "False"}`;
    if (this.center !== null) out += `, center=(${this.center.join(", ")})`;
    if (this.fill !== null) {
      out += `, fill=${typeof this.fill === "number" ? this.fill : tuple(this.fill)}`;
    }
    return out + ")";
  }
}

/**
 * A rotation, a shift, a scaling and a shear, each drawn from its own range and
 * **applied in one resampling.**
 *
 * `translate` is a *fraction* of the picture's width and height rather than a
 * number of pixels, so the same transform means the same thing on any size.
 */
export class RandomAffine implements Transform {
  protected readonly degrees: [number, number];
  private readonly shearRange: [number, number] | null;

  constructor(
    degrees: number | readonly number[],
    private readonly translate: readonly [number, number] | null = null,
    protected readonly scale: readonly [number, number] | null = null,
    shear: number | readonly number[] | null = null,
    protected readonly interpolation: "bilinear" | "nearest" = "nearest",
    protected readonly fill: number | readonly number[] = 0,
    private readonly center: readonly [number, number] | null = null,
  ) {
    this.degrees = setupAngle(degrees, "degrees");
    if (translate !== null) {
      for (const t of translate) {
        if (!(t >= 0 && t <= 1)) {
          throw new RuntimeError(
            `translate is a fraction of the picture, between 0 and 1 — ` +
            `got (${translate.join(", ")}).\n` +
            "(torch: translation values should be between 0 and 1)");
        }
      }
    }
    if (scale !== null) {
      for (const s of scale) {
        if (s < 0) {
          throw new RuntimeError(
            `scale values should be positive — got (${scale.join(", ")}).\n` +
            "(torch: scale values should be positive)");
        }
      }
    }
    this.shearRange = shear === null ? null : setupAngle(shear, "shear");
  }

  /**
   * `[angle, [tx, ty], scale, [shearX, shearY]]`. **The shift is drawn in pixels
   * and rounded**, so a fraction working out to less than half a pixel draws zero
   * rather than a fraction of one.
   */
  getParams(w: number, h: number): [number, [number, number], number, [number, number]] {
    const angle = uniform(this.degrees[0], this.degrees[1]);
    let shift: [number, number] = [0, 0];
    if (this.translate !== null) {
      const maxDx = this.translate[0] * w;
      const maxDy = this.translate[1] * h;
      shift = [roundHalfToEven(uniform(-maxDx, maxDx)),
        roundHalfToEven(uniform(-maxDy, maxDy))];
    }
    const scale = this.scale === null ? 1 : uniform(this.scale[0], this.scale[1]);
    const shear: [number, number] = this.shearRange === null
      ? [0, 0] : [uniform(this.shearRange[0], this.shearRange[1]), 0];
    return [angle, shift, scale, shear];
  }

  apply(x: Subject): Image {
    const img = asImage(x, "RandomAffine");
    const fill = typeof this.fill === "number"
      ? new Array<number>(img.channels).fill(this.fill) : [...this.fill];
    const [angle, shift, scale, shear] = this.getParams(img.width, img.height);
    return affine(img, angle, shift, scale, shear, this.interpolation, fill, this.center);
  }

  describe(): string {
    let out = `RandomAffine(degrees=${floatList(this.degrees)}`;
    if (this.translate !== null) out += `, translate=(${this.translate.join(", ")})`;
    if (this.scale !== null) out += `, scale=(${this.scale.join(", ")})`;
    if (this.shearRange !== null) out += `, shear=${floatList(this.shearRange)}`;
    if (this.interpolation !== "nearest") out += `, interpolation=${this.interpolation}`;
    if (this.fill !== 0) {
      out += `, fill=${typeof this.fill === "number" ? this.fill : tuple(this.fill)}`;
    }
    if (this.center !== null) out += `, center=(${this.center.join(", ")})`;
    return out + ")";
  }
}


function gaussianKernel1d(size: number, sigma: number): number[] {
  const half = f32((size - 1) * 0.5);
  const out: number[] = [];
  let total = 0;
  for (let i = 0; i < size; i++) {
    const x = f32(-half + i);
    const v = f32(Math.exp(f32(-0.5 * f32(Math.pow(f32(x / sigma), 2)))));
    out.push(v);
    total = f32(total + v);
  }
  return out.map((v) => f32(v / total));
}

/** numpy's `reflect` — mirrors without repeating the edge. `sourceIndex`'s rule. */
function reflectAt(i: number, n: number): number {
  return sourceIndex(i, n, "reflect");
}

/**
 * Blur with a Gaussian. **The border is reflected**, not zeroed — a zero border
 * darkens the edge of every blurred picture, which looks like a vignette.
 *
 * `kernelSize` is one number or two, and both have to be **odd**: an even kernel
 * has no centre pixel to sit on, so the picture would shift by half a pixel.
 */
export function gaussianBlur(
  img: Image,
  kernelSize: number | readonly number[],
  sigma: number | readonly number[] | null = null,
): Image {
  const src = asImage(img, "gaussian_blur");
  const sizes = typeof kernelSize === "number"
    ? [kernelSize, kernelSize] : [kernelSize[0] ?? 0, kernelSize[1] ?? kernelSize[0] ?? 0];
  for (const s of sizes) {
    if (s <= 0 || s % 2 === 0) {
      throw new RuntimeError(
        `kernel_size is odd and positive — got (${sizes.join(", ")}).\n` +
        "(torch: Kernel size value should be an odd and positive number.)");
    }
  }
  let sigmas: number[];
  if (sigma === null) sigmas = sizes.map((s) => 0.3 * ((s - 1) * 0.5 - 1) + 0.8);
  else if (typeof sigma === "number") sigmas = [sigma, sigma];
  else sigmas = sigma.length === 1 ? [sigma[0] ?? 0, sigma[0] ?? 0] : [sigma[0] ?? 0, sigma[1] ?? 0];
  for (const s of sigmas) {
    if (s <= 0) {
      throw new RuntimeError(
        `sigma is a positive number — got (${sigmas.join(", ")}).\n` +
        "(torch: sigma should have positive values.)");
    }
  }
  // **`kernelSize` and `sigma` are (x, y)** — width first, like `getImageSize`
  // and unlike everything shaped. torchvision builds the 2-D kernel as an outer
  // product of the y kernel with the x one, and **a transpose is invisible while
  // both sizes match** — only the oblong case can see it (measured).
  const kx = gaussianKernel1d(sizes[0] ?? 1, sigmas[0] ?? 1);
  const ky = gaussianKernel1d(sizes[1] ?? 1, sigmas[1] ?? 1);
  const pw = Math.floor((sizes[0] ?? 1) / 2), ph = Math.floor((sizes[1] ?? 1) / 2);
  const { height: h, width: w, channels: c } = src;
  const out = new Float64Array(src.data.length);
  const hi = bound(src.isByte);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      for (let k = 0; k < c; k++) {
        let acc = 0;
        for (let di = 0; di < ky.length; di++) {
          const sy = reflectAt(y + di - ph, h);
          for (let dj = 0; dj < kx.length; dj++) {
            const sx = reflectAt(x + dj - pw, w);
            const weight = f32((ky[di] ?? 0) * (kx[dj] ?? 0));
            acc = f32(acc + f32(weight * f32(src.data[(sy * w + sx) * c + k] ?? 0)));
          }
        }
        out[(y * w + x) * c + k] = src.isByte
          ? Math.min(Math.max(Math.round(acc), 0), hi) : acc;
      }
    }
  }
  return { ...src, data: out };
}

/**
 * The eight numbers, solved by least squares. **In float64 and returned as
 * float32**, which is torchvision's own precision split.
 *
 * **The reason usually given for that split did not reproduce here.** The claim
 * is that the solve is ill-conditioned enough that float32 moves the corners
 * visibly. Measured three ways and it does not:
 *
 *   golden's own corners   float32 and float64 solves agree to the bit
 *                          (condition number 2318)
 *   500 random corner sets zero difference in the coefficients
 *   this whole solve in    all 32 grid cases still pass, 4.77e-7 worst,
 *   float32                three orders inside the 1e-4 tolerance
 *
 * float64 stays because upstream does it and a ported formula is not changed on
 * the strength of one sweep. What is not carried is the justification — the
 * effect it describes is not present in anything this side can measure.
 */
export function perspectiveCoefficients(
  startpoints: readonly (readonly [number, number])[],
  endpoints: readonly (readonly [number, number])[],
): number[] {
  if (startpoints.length !== 4 || endpoints.length !== 4) {
    throw new RuntimeError(
      `Please provide exactly four corners, got ${startpoints.length} startpoints ` +
      `and ${endpoints.length} endpoints.\n(torch: Please provide exactly four corners)`);
  }
  const a: number[][] = [];
  const b: number[] = [];
  for (let i = 0; i < 4; i++) {
    const p1 = endpoints[i] ?? [0, 0];
    const p2 = startpoints[i] ?? [0, 0];
    a.push([p1[0], p1[1], 1, 0, 0, 0, -p2[0] * p1[0], -p2[0] * p1[1]]);
    a.push([0, 0, 0, p1[0], p1[1], 1, -p2[1] * p1[0], -p2[1] * p1[1]]);
    b.push(p2[0], p2[1]);
  }
  // The system is square and full rank, so least squares is simply the solve.
  // Gaussian elimination with partial pivoting, in float64 — which matches
  // numpy's `lstsq` to the bit after the float32 cast on every case here.
  for (let col = 0; col < 8; col++) {
    let best = col;
    for (let r = col + 1; r < 8; r++) {
      if (Math.abs(a[r]?.[col] ?? 0) > Math.abs(a[best]?.[col] ?? 0)) best = r;
    }
    if (best !== col) {
      const t = a[col]!; a[col] = a[best]!; a[best] = t;
      const tb = b[col]!; b[col] = b[best]!; b[best] = tb;
    }
    const pivot = a[col]?.[col] ?? 0;
    for (let r = col + 1; r < 8; r++) {
      const factor = (a[r]?.[col] ?? 0) / pivot;
      if (factor === 0) continue;
      for (let k = col; k < 8; k++) a[r]![k] = (a[r]?.[k] ?? 0) - factor * (a[col]?.[k] ?? 0);
      b[r] = (b[r] ?? 0) - factor * (b[col] ?? 0);
    }
  }
  const x = new Array<number>(8).fill(0);
  for (let r = 7; r >= 0; r--) {
    let acc = b[r] ?? 0;
    for (let k = r + 1; k < 8; k++) acc -= (a[r]?.[k] ?? 0) * (x[k] ?? 0);
    x[r] = acc / (a[r]?.[r] ?? 1);
  }
  return x.map((v) => f32(v));
}

/**
 * The projective map, **with the division that makes it projective.** An affine
 * grid is this one with the last two coefficients zero.
 */
function perspectiveGrid(coeffs: readonly number[], ow: number, oh: number): Float64Array {
  const c = coeffs.map((v) => f32(v));
  const sx = f32(0.5 * ow), sy = f32(0.5 * oh);
  const grid = new Float64Array(oh * ow * 2);
  for (let j = 0; j < oh; j++) {
    const y = f32(0.5 + j);
    for (let i = 0; i < ow; i++) {
      const x = f32(0.5 + i);
      const t0 = f32(f32(f32(x * (c[0] ?? 0)) + f32(y * (c[1] ?? 0))) + (c[2] ?? 0));
      const t1 = f32(f32(f32(x * (c[3] ?? 0)) + f32(y * (c[4] ?? 0))) + (c[5] ?? 0));
      const den = f32(f32(f32(x * (c[6] ?? 0)) + f32(y * (c[7] ?? 0))) + 1);
      grid[(j * ow + i) * 2] = f32(f32(f32(t0 / sx) / den) - 1);
      grid[(j * ow + i) * 2 + 1] = f32(f32(f32(t1 / sy) / den) - 1);
    }
  }
  return grid;
}

/**
 * Move the four corners somewhere else — **a photograph of a photograph held at
 * an angle.** Unlike `affine`, straight lines stay straight but parallel ones
 * stop being parallel.
 */
export function perspective(
  img: Image,
  startpoints: readonly (readonly [number, number])[],
  endpoints: readonly (readonly [number, number])[],
  interpolation: "bilinear" | "nearest" = "bilinear",
  fill: number | readonly number[] | null = null,
): Image {
  const src = asImage(img, "perspective");
  const coeffs = perspectiveCoefficients(startpoints, endpoints);
  const grid = perspectiveGrid(coeffs, src.width, src.height);
  return gridTransform(src, grid, src.height, src.width, interpolation, fillList(fill));
}

/**
 * Push every pixel a little way, smoothly. **The displacement is given rather
 * than drawn** — `ElasticTransform` draws it and this applies it, so a whole
 * batch can share one warp.
 */
export function elasticTransform(
  img: Image,
  displacement: ArrayLike<number>,
  interpolation: "bilinear" | "nearest" = "bilinear",
  fill: number | readonly number[] | null = null,
): Image {
  const src = asImage(img, "elastic_transform");
  const { height: h, width: w } = src;
  const grid = new Float64Array(h * w * 2);
  for (let j = 0; j < h; j++) {
    // The identity grid reads each output pixel from its own position; the
    // displacement is what gets added to it.
    const gy = f32(f32(-h + 1) / h + f32(j * f32(2 / h)));
    for (let i = 0; i < w; i++) {
      const gx = f32(f32(-w + 1) / w + f32(i * f32(2 / w)));
      grid[(j * w + i) * 2] = f32(gx + f32(displacement[(j * w + i) * 2] ?? 0));
      grid[(j * w + i) * 2 + 1] = f32(gy + f32(displacement[(j * w + i) * 2 + 1] ?? 0));
    }
  }
  return gridTransform(src, grid, h, w, interpolation, fillList(fill));
}


/**
 * Blur by a Gaussian whose width is **drawn from a range each call.**
 *
 * The kernel size is fixed and the sigma is the draw, which is the opposite way
 * round from most of these — a bigger kernel costs time, so torchvision fixes
 * the cost and varies the effect inside it.
 */
export class GaussianBlur implements Transform {
  protected readonly kernelSize: [number, number];
  protected readonly sigma: [number, number];

  constructor(kernelSize: number | readonly number[],
              sigma: number | readonly [number, number] = [0.1, 2.0]) {
    this.kernelSize = typeof kernelSize === "number"
      ? [kernelSize, kernelSize]
      : [kernelSize[0] ?? 0, kernelSize[1] ?? kernelSize[0] ?? 0];
    for (const s of this.kernelSize) {
      if (s <= 0 || s % 2 === 0) {
        throw new RuntimeError(
          `kernel_size is odd and positive — got (${this.kernelSize.join(", ")}).\n` +
          "(torch: Kernel size value should be an odd and positive number.)");
      }
    }
    if (typeof sigma === "number") {
      if (sigma <= 0) {
        throw new RuntimeError(
          `sigma is a positive number — got ${sigma}.\n` +
          "(torch: If sigma is a single number, it must be positive.)");
      }
      this.sigma = [sigma, sigma];
    } else {
      if (!(sigma[0] > 0 && sigma[0] <= sigma[1])) {
        throw new RuntimeError(
          `sigma is (min, max) with min above zero — got (${sigma.join(", ")}).\n` +
          "(torch: sigma values should be positive and of the form (min, max).)");
      }
      this.sigma = [sigma[0], sigma[1]];
    }
  }

  getParams(): number {
    return uniform(this.sigma[0], this.sigma[1]);
  }

  apply(x: Subject): Image {
    const img = asImage(x, "GaussianBlur");
    const drawn = this.getParams();
    return gaussianBlur(img, this.kernelSize, [drawn, drawn]);
  }

  describe(): string {
    // `sigma` is a pair of Python floats, so it prints `2.0` rather than `2`.
    // `kernel_size` is integers and prints as it is — two spellings on one line,
    // which is torchvision's shape rather than an inconsistency to tidy.
    return `GaussianBlur(kernel_size=(${this.kernelSize.join(", ")}), ` +
      `sigma=${floatTuple(this.sigma)})`;
  }
}

/**
 * Tilt the picture, with probability `p`. `distortionScale` is how far the
 * corners may move, as a fraction of half the picture.
 */
export class RandomPerspective implements Transform {
  constructor(
    protected readonly distortionScale = 0.5,
    protected readonly p = 0.5,
    protected readonly interpolation: "bilinear" | "nearest" = "bilinear",
    protected readonly fill: number | readonly number[] = 0,
  ) {}

  /**
   * The four corners before and after. **Drawn in whole pixels** — torchvision
   * uses integer draws here, so a small picture has a small number of distinct
   * distortions rather than a continuum.
   */
  getParams(width: number, height: number):
    [readonly [number, number][], readonly [number, number][]] {
    const halfH = Math.floor(height / 2), halfW = Math.floor(width / 2);
    const dx = Math.trunc(this.distortionScale * halfW);
    const dy = Math.trunc(this.distortionScale * halfH);
    const between = (lo: number, hi: number): number => lo + nextInt(hi - lo);
    const topleft: [number, number] = [between(0, dx + 1), between(0, dy + 1)];
    const topright: [number, number] = [between(width - dx - 1, width), between(0, dy + 1)];
    const botright: [number, number] = [between(width - dx - 1, width),
      between(height - dy - 1, height)];
    const botleft: [number, number] = [between(0, dx + 1), between(height - dy - 1, height)];
    const start: [number, number][] = [[0, 0], [width - 1, 0],
      [width - 1, height - 1], [0, height - 1]];
    return [start, [topleft, topright, botright, botleft]];
  }

  apply(x: Subject): Image {
    const img = asImage(x, "RandomPerspective");
    if (nextFloat() >= this.p) return img;
    const fill = typeof this.fill === "number"
      ? new Array<number>(img.channels).fill(this.fill) : [...this.fill];
    const [start, end] = this.getParams(img.width, img.height);
    return perspective(img, start, end, this.interpolation, fill);
  }

  describe(): string {
    // **Only `p`.** torchvision prints nothing else here — not the distortion,
    // not the fill — and that is its spelling rather than an oversight to
    // improve on.
    return `RandomPerspective(p=${this.p})`;
  }
}

/**
 * Push every pixel a little way along a **smooth random field** — the warp that
 * makes handwriting look handwritten differently.
 *
 * `alpha` is how far pixels move and `sigma` how smoothly: a small sigma with a
 * large alpha is noise rather than a warp, which is why both go through the same
 * blur.
 */
export class ElasticTransform implements Transform {
  protected readonly alpha: [number, number];
  protected readonly sigma: [number, number];
  protected readonly fill: number[];

  constructor(
    alpha: number | readonly number[] = 50,
    sigma: number | readonly number[] = 5,
    protected readonly interpolation: "bilinear" | "nearest" = "bilinear",
    fill: number | readonly number[] = 0,
  ) {
    this.alpha = typeof alpha === "number" ? [alpha, alpha] : [alpha[0] ?? 0, alpha[1] ?? 0];
    this.sigma = typeof sigma === "number" ? [sigma, sigma] : [sigma[0] ?? 0, sigma[1] ?? 0];
    this.fill = typeof fill === "number" ? [fill] : [...fill];
  }

  /**
   * A displacement field: noise, blurred, scaled by `alpha` — and **divided by
   * the picture's size**, because the grid is in [-1,1] and not in pixels.
   */
  getParams(height: number, width: number): Float64Array {
    const out = new Float64Array(height * width * 2);
    for (let axis = 0; axis < 2; axis++) {
      const sigma = this.sigma[axis] ?? 0;
      const alpha = this.alpha[axis] ?? 0;
      const extent = axis === 0 ? width : height;
      const noise: Image = {
        data: new Float64Array(height * width), height, width, channels: 1, isByte: false,
      };
      for (let i = 0; i < noise.data.length; i++) noise.data[i] = f32(nextFloat() * 2 - 1);
      let field = noise;
      if (sigma > 0) {
        let size = Math.trunc(8 * sigma + 1);
        if (size % 2 === 0) size += 1;
        field = gaussianBlur(noise, [size, size], this.sigma);
      }
      for (let i = 0; i < height * width; i++) {
        out[i * 2 + axis] = f32(f32((field.data[i] ?? 0) * alpha) / extent);
      }
    }
    return out;
  }

  apply(x: Subject): Image {
    const img = asImage(x, "ElasticTransform");
    return elasticTransform(img, this.getParams(img.height, img.width),
      this.interpolation, this.fill);
  }

  describe(): string {
    // **The enum's name, not its value** — this is the one class that prints
    // `InterpolationMode.BILINEAR` where every other one prints `bilinear`.
    const mode = `InterpolationMode.${this.interpolation.toUpperCase()}`;
    return `ElasticTransform(alpha=${floatList(this.alpha)}, ` +
      `sigma=${floatList(this.sigma)}, interpolation=${mode}, ` +
      `fill=${floatList(this.fill)})`;
  }
}

// ── The policy layer ──────────────────────────────────────────────────
//
// **Almost nothing here can be frozen, and that is the honest shape of it.** All
// four draw on every call — which operation, how hard, which sign, and for two of
// them how many — so a frozen picture would be a frozen dice roll. What can be
// frozen is the part that is not drawn: the learned table itself, and the one
// configuration that applies nothing.
//
// The operations these pick from are cased individually above. That is where this
// layer's values are actually held.

/**
 * Which learned table `AutoAugment` uses. The three are the datasets the search
 * was run on, and they are **not interchangeable** — the SVHN policy inverts and
 * shears hard because house numbers survive it, and a photograph does not.
 */
export const AutoAugmentPolicy = {
  IMAGENET: "imagenet",
  CIFAR10: "cifar10",
  SVHN: "svhn",
} as const;

export type AutoAugmentPolicyName =
  (typeof AutoAugmentPolicy)[keyof typeof AutoAugmentPolicy];

/** `AutoAugmentPolicy.IMAGENET`, which is how Python prints an enum member. */
export function policyName(v: AutoAugmentPolicyName): string {
  const key = Object.keys(AutoAugmentPolicy).find(
    (k) => AutoAugmentPolicy[k as keyof typeof AutoAugmentPolicy] === v);
  return `AutoAugmentPolicy.${key ?? v}`;
}

/** `InterpolationMode.NEAREST` — three of the four print the enum, not the value. */
export function modeName(v: "bilinear" | "nearest"): string {
  return `InterpolationMode.${v.toUpperCase()}`;
}

/** `None` for null, otherwise the number as Python spells it. */
function orNone(v: number | readonly number[] | null): string {
  if (v === null) return "None";
  return typeof v === "number" ? `${v}` : tuple(v);
}

/** One half of a policy pair: the operation, its probability, its strength bin. */
export type PolicyStep = readonly [string, number, number | null];

/**
 * **Lists, not tuples**, because `AutoAugment(...).policies` is a public attribute
 * and torchvision hands back a list. The golden caught the difference on its first
 * run: identical data, different brackets. What holds a value is part of the
 * surface.
 *
 * Nothing here is derivable — it is the output of the search that produced
 * AutoAugment, which is why it is written out rather than computed, and why the
 * three datasets are three tables. Every entry is plausible, so a transcription
 * error stays; the golden comparing these as text is the only check there can be.
 */
const POLICIES: Record<string, readonly (readonly [PolicyStep, PolicyStep])[]> = {
  imagenet: [
    [["Posterize", 0.4, 8], ["Rotate", 0.6, 9]],
    [["Solarize", 0.6, 5], ["AutoContrast", 0.6, null]],
    [["Equalize", 0.8, null], ["Equalize", 0.6, null]],
    [["Posterize", 0.6, 7], ["Posterize", 0.6, 6]],
    [["Equalize", 0.4, null], ["Solarize", 0.2, 4]],
    [["Equalize", 0.4, null], ["Rotate", 0.8, 8]],
    [["Solarize", 0.6, 3], ["Equalize", 0.6, null]],
    [["Posterize", 0.8, 5], ["Equalize", 1.0, null]],
    [["Rotate", 0.2, 3], ["Solarize", 0.6, 8]],
    [["Equalize", 0.6, null], ["Posterize", 0.4, 6]],
    [["Rotate", 0.8, 8], ["Color", 0.4, 0]],
    [["Rotate", 0.4, 9], ["Equalize", 0.6, null]],
    [["Equalize", 0.0, null], ["Equalize", 0.8, null]],
    [["Invert", 0.6, null], ["Equalize", 1.0, null]],
    [["Color", 0.6, 4], ["Contrast", 1.0, 8]],
    [["Rotate", 0.8, 8], ["Color", 1.0, 2]],
    [["Color", 0.8, 8], ["Solarize", 0.8, 7]],
    [["Sharpness", 0.4, 7], ["Invert", 0.6, null]],
    [["ShearX", 0.6, 5], ["Equalize", 1.0, null]],
    [["Color", 0.4, 0], ["Equalize", 0.6, null]],
    [["Equalize", 0.4, null], ["Solarize", 0.2, 4]],
    [["Solarize", 0.6, 5], ["AutoContrast", 0.6, null]],
    [["Invert", 0.6, null], ["Equalize", 1.0, null]],
    [["Color", 0.6, 4], ["Contrast", 1.0, 8]],
    [["Equalize", 0.8, null], ["Equalize", 0.6, null]],
  ],
  cifar10: [
    [["Invert", 0.1, null], ["Contrast", 0.2, 6]],
    [["Rotate", 0.7, 2], ["TranslateX", 0.3, 9]],
    [["Sharpness", 0.8, 1], ["Sharpness", 0.9, 3]],
    [["ShearY", 0.5, 8], ["TranslateY", 0.7, 9]],
    [["AutoContrast", 0.5, null], ["Equalize", 0.9, null]],
    [["ShearY", 0.2, 7], ["Posterize", 0.3, 7]],
    [["Color", 0.4, 3], ["Brightness", 0.6, 7]],
    [["Sharpness", 0.3, 9], ["Brightness", 0.7, 9]],
    [["Equalize", 0.6, null], ["Equalize", 0.5, null]],
    [["Contrast", 0.6, 7], ["Sharpness", 0.6, 5]],
    [["Color", 0.7, 7], ["TranslateX", 0.5, 8]],
    [["Equalize", 0.3, null], ["AutoContrast", 0.4, null]],
    [["TranslateY", 0.4, 3], ["Sharpness", 0.2, 6]],
    [["Brightness", 0.9, 6], ["Color", 0.2, 8]],
    [["Solarize", 0.5, 2], ["Invert", 0.0, null]],
    [["Equalize", 0.2, null], ["AutoContrast", 0.6, null]],
    [["Equalize", 0.2, null], ["Equalize", 0.6, null]],
    [["Color", 0.9, 9], ["Equalize", 0.6, null]],
    [["AutoContrast", 0.8, null], ["Solarize", 0.2, 8]],
    [["Brightness", 0.1, 3], ["Color", 0.7, 0]],
    [["Solarize", 0.4, 5], ["AutoContrast", 0.9, null]],
    [["TranslateY", 0.9, 9], ["TranslateY", 0.7, 9]],
    [["AutoContrast", 0.9, null], ["Solarize", 0.8, 3]],
    [["Equalize", 0.8, null], ["Invert", 0.1, null]],
    [["TranslateY", 0.7, 9], ["AutoContrast", 0.9, null]],
  ],
  svhn: [
    [["ShearX", 0.9, 4], ["Invert", 0.2, null]],
    [["ShearY", 0.9, 8], ["Invert", 0.7, null]],
    [["Equalize", 0.6, null], ["Solarize", 0.6, 6]],
    [["Invert", 0.9, null], ["Equalize", 0.6, null]],
    [["Equalize", 0.6, null], ["Rotate", 0.9, 3]],
    [["ShearX", 0.9, 4], ["AutoContrast", 0.8, null]],
    [["ShearY", 0.9, 8], ["Invert", 0.4, null]],
    [["ShearY", 0.9, 5], ["Solarize", 0.2, 6]],
    [["Invert", 0.9, null], ["AutoContrast", 0.8, null]],
    [["Equalize", 0.6, null], ["Rotate", 0.9, 3]],
    [["ShearX", 0.9, 4], ["Solarize", 0.3, 3]],
    [["ShearY", 0.8, 8], ["Invert", 0.7, null]],
    [["Equalize", 0.9, null], ["TranslateY", 0.6, 6]],
    [["Invert", 0.9, null], ["Equalize", 0.6, null]],
    [["Contrast", 0.3, 3], ["Rotate", 0.8, 4]],
    [["Invert", 0.8, null], ["TranslateY", 0.0, 2]],
    [["ShearY", 0.7, 6], ["Solarize", 0.4, 8]],
    [["Invert", 0.6, null], ["Rotate", 0.8, 4]],
    [["ShearY", 0.3, 7], ["TranslateX", 0.9, 3]],
    [["ShearX", 0.1, 6], ["Invert", 0.6, null]],
    [["Solarize", 0.7, 2], ["TranslateY", 0.6, 7]],
    [["ShearY", 0.8, 4], ["Invert", 0.8, null]],
    [["ShearX", 0.7, 9], ["TranslateY", 0.8, 3]],
    [["ShearY", 0.8, 5], ["AutoContrast", 0.7, null]],
    [["ShearX", 0.7, 2], ["Invert", 0.1, null]],
  ],
};

/**
 * The **learned** one: twenty-five pairs of operations found by a search, each
 * with its own probability and strength, one pair drawn per call.
 *
 * Nothing about the table is derivable — it is the output of the search, which is
 * why it is written out rather than computed, and why the three datasets are
 * three tables.
 */
export class AutoAugment implements Transform {
  /** The learned table for this policy. **A list**, as torchvision hands back. */
  readonly policies: readonly (readonly [PolicyStep, PolicyStep])[];

  constructor(
    protected readonly policy: AutoAugmentPolicyName = AutoAugmentPolicy.IMAGENET,
    protected readonly interpolation: "bilinear" | "nearest" = "nearest",
    protected readonly fill: number | readonly number[] | null = null,
  ) {
    const table = POLICIES[policy];
    if (table === undefined) {
      throw new RuntimeError(
        `${JSON.stringify(policy)} is not an AutoAugmentPolicy — ` +
        `it is one of ${Object.values(AutoAugmentPolicy).join(", ")}.`);
    }
    this.policies = table;
  }

  apply(_x: Subject): Image {
    throw new RuntimeError(
      "AutoAugment does not run here yet — the fifteen operations it draws from " +
      "are present, but the magnitude space that turns a strength index into a " +
      "number is not.\n" +
      `  It holds policy=${this.policy} and interpolation=${this.interpolation}; ` +
      "`describe()` works, applying does not.");
  }

  describe(): string {
    return `AutoAugment(policy=${policyName(this.policy)}, fill=${orNone(this.fill)})`;
  }
}

/**
 * The **uniform** one: `numOps` operations drawn evenly, all at the same fixed
 * strength.
 *
 * Its point is that the search was unnecessary — one strength dial and a count do
 * as well as the learned table, which is why `magnitude` is a number you tune
 * rather than a distribution.
 */
export class RandAugment implements Transform {
  constructor(
    protected readonly numOps = 2,
    protected readonly magnitude = 9,
    protected readonly numMagnitudeBins = 31,
    protected readonly interpolation: "bilinear" | "nearest" = "nearest",
    protected readonly fill: number | readonly number[] | null = null,
  ) {}

  apply(x: Subject): Image {
    const img = asImage(x, "RandAugment");
    // **Zero operations has to be the identity**, and it is the only configuration
    // of any of the four that does not draw. A `numOps` read as a count of
    // something else would show here and nowhere else.
    if (this.numOps === 0) return img;
    throw new RuntimeError(
      "RandAugment does not run here yet — only `numOps = 0`, which applies " +
      "nothing, is implemented.\n" +
      "  The magnitude space that turns a strength index into a number is absent.");
  }

  describe(): string {
    return `RandAugment(num_ops=${this.numOps}, magnitude=${this.magnitude}` +
      `, num_magnitude_bins=${this.numMagnitudeBins}` +
      `, interpolation=${modeName(this.interpolation)}, fill=${orNone(this.fill)})`;
  }
}

/**
 * The one with **no dials at all**: one operation, drawn evenly, at a strength
 * also drawn evenly from a wide ladder.
 *
 * That is the paper's claim — tuning the strength was never worth it — so there
 * is no magnitude argument to pass. The ladder is much wider than
 * `RandAugment`'s to make up for drawing it.
 */
export class TrivialAugmentWide implements Transform {
  constructor(
    protected readonly numMagnitudeBins = 31,
    protected readonly interpolation: "bilinear" | "nearest" = "nearest",
    protected readonly fill: number | readonly number[] | null = null,
  ) {}

  apply(_x: Subject): Image {
    throw new RuntimeError(
      "TrivialAugmentWide does not run here yet — its magnitude ladder is absent.");
  }

  describe(): string {
    return `TrivialAugmentWide(num_magnitude_bins=${this.numMagnitudeBins}` +
      `, interpolation=${modeName(this.interpolation)}, fill=${orNone(this.fill)})`;
  }
}

/**
 * The **blended** one: several independent chains of operations, mixed back into
 * the original by weights drawn from a Dirichlet.
 *
 * That is what makes it different in kind from the other three — they replace the
 * picture and this one **averages several versions of it with the original**, so
 * the result stays close to the input however hard the chains hit.
 */
export class AugMix implements Transform {
  private static readonly PARAMETER_MAX = 10;

  constructor(
    protected readonly severity = 3,
    protected readonly mixtureWidth = 3,
    protected readonly chainDepth = -1,
    protected readonly alpha = 1.0,
    protected readonly allOps = true,
    protected readonly interpolation: "bilinear" | "nearest" = "bilinear",
    protected readonly fill: number | readonly number[] | null = null,
  ) {
    if (!(severity >= 1 && severity <= AugMix.PARAMETER_MAX)) {
      throw new RuntimeError(
        `The severity must be between [1, ${AugMix.PARAMETER_MAX}]. ` +
        `Got ${severity} instead.\n` +
        `(torch: The severity must be between [1, ${AugMix.PARAMETER_MAX}].)`);
    }
  }

  apply(_x: Subject): Image {
    throw new RuntimeError(
      "AugMix does not run here yet — its magnitude space is absent, and it is " +
      "the one of the four whose space differs from the others in four places.");
  }

  describe(): string {
    return `AugMix(severity=${this.severity}, mixture_width=${this.mixtureWidth}` +
      `, chain_depth=${this.chainDepth}, alpha=${pyFloat(this.alpha)}` +
      `, all_ops=${this.allOps ? "True" : "False"}` +
      `, interpolation=${modeName(this.interpolation)}, fill=${orNone(this.fill)})`;
  }
}
