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
  constructor(private readonly transforms: readonly Transform[]) {}

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
  constructor(private readonly fn: (x: Subject) => Subject) {
    if (typeof fn !== "function") {
      throw new RuntimeError(
        "Lambda takes a function — it received " + typeof fn + ".\n" +
        "(torch: Argument lambd should be callable)");
    }
  }

  apply(x: Subject): Subject {
    return this.fn(x);
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
  // **이름을 넘겨받는다.** `this.constructor.name` 이나 `new.target.name` 으로 쓰면
  // 번들러가 클래스 이름을 줄이는 날 `repr` 이 조용히 바뀐다 — 이 프로젝트는 `repr`
  // 을 명세로 보므로 그것은 사양이 소리 없이 바뀌는 것이다.
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
  constructor(transforms: readonly Transform[], private readonly p = 0.5) {
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
    private readonly weights: readonly number[] | null = null,
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
    if (this.weights === null) return nextInt(n);
    // **The weights are normalised here.** torch's `random.choices` takes
    // relative weights; handing them straight to a cumulative draw would make
    // `p=[1, 1]` mean something other than "evenly".
    let total = 0;
    for (const w of this.weights) total += w;
    let r = nextFloat() * total;
    for (let i = 0; i < n; i++) {
      r -= this.weights[i] ?? 0;
      if (r < 0) return i;
    }
    return n - 1;
  }

  override describe(): string {
    const p = this.weights === null ? "None" : tuple(this.weights);
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
    // **여러 장이 온 것을 여기서 잡는다.** `Transform` 이 배열까지 받도록 넓어진 뒤로
    // `Compose([new FiveCrop(3), new ToTensor()])` 가 타입 검사를 통과한다. 막지
    // 않으면 배열을 구조분해해 `data`·`height` 가 전부 `undefined` 가 되고, 터지긴
    // 하는데 `shape [,,] does not match 0 elements` 라고 터진다 — 무엇을 잘못했는지
    // 안 적힌 사고다(실측). torchvision 도 같은 자리에서 `Lambda` 를 쓰라고 말한다.
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
    private readonly mean: readonly number[],
    private readonly std: readonly number[],
  ) {}

  apply(x: Image | Tensor): Tensor {
    if (!(x instanceof Tensor)) throw new Error("Normalize takes a tensor");
    const shape = [this.mean.length, 1, 1];
    const m = Tensor.from([...this.mean], shape);
    const s = Tensor.from([...this.std], shape);
    return x.sub(m).div(s);
  }

  describe(): string {
    // 파이썬이 튜플을 `(0.5, 0.4, 0.3)` 로 찍는다. 받은 그대로를 찍는 것이 규칙이고,
    // 골든이 그 글자를 굳혔다.
    return `Normalize(mean=${tuple(this.mean)}, std=${tuple(this.std)})`;
  }
}

/** 파이썬의 튜플 표기. 원소가 하나면 뒤에 쉼표가 붙는다. */
function tuple(values: readonly number[]): string {
  const parts = values.map((v) => String(v));
  return parts.length === 1 ? `(${parts[0]},)` : `(${parts.join(", ")})`;
}

/**
 * 뽑기를 쓰는 변환의 난수기.
 *
 * **골든은 뽑기를 대조하지 않는다** — 확률을 0 이나 1 로 못 박거나 자를 자리가
 * 하나뿐이게 만들어 결정적인 자리만 묻는다. 그래서 여기 난수기는 torch 와 같을
 * 필요가 없고, 같은 척해서도 안 된다.
 */
const rng = { state: 12345 };

export function manualSeed(seed: number): void {
  rng.state = seed >>> 0;
}

function nextFloat(): number {
  // xorshift32. 분포를 재는 자리가 아니라 뽑기가 도는지만 보는 자리다.
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
  constructor(private readonly p = 0.5) {}

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
  constructor(private readonly p = 0.5) {}

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
    private readonly padding: number | readonly number[],
    private readonly fill: number | readonly number[] = 0,
    private readonly paddingMode: PaddingMode = "constant",
  ) {
    if (!PADDING_MODES.includes(paddingMode)) {
      throw new RuntimeError(
        `padding_mode is constant, edge, reflect or symmetric — got ` +
        `${JSON.stringify(paddingMode)}.\n` +
        "(torch: Padding mode should be either constant, edge, reflect or symmetric)");
    }
    padSides(padding);          // 첫 호출이 아니라 여기서 멈춘다
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

// 같은 세 수를 float32 로 좁힌 것. **왜 둘 다 필요한지가 요점이다** — 아래 참조.
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
      // **두 갈래가 있고, 둘 다 numpy 를 그대로 따라간 것이다.** 재서 알아낸 자리라
      // 적어 둔다 — 어느 쪽 소스를 읽어도 안 보이고, 두 쪽을 맞대야만 나온다.
      //
      // 파이썬은 `arr[:,:,0]*_LUMA[0] + ...` 한 줄이고, 그 한 줄의 산술이 배열의
      // dtype 에 따라 갈린다 (NEP 50 의 약한 승격):
      //
      //   uint8  × 파이썬 실수 → float64.  전부 float64 로 계산하고 끝에서 잘린다.
      //   float32 × 파이썬 실수 → float32.  **스칼라가 먼저 float32 로 좁혀지고**,
      //                                     그 뒤 곱과 부분합이 전부 float32 다.
      //
      // 실수 쪽을 float64 로 계산하면 골든과 최대 6.5e-8 갈린다. 허용오차 안이라
      // **검사는 통과한다** — 20 개 화소 중 0 개가 정확히 맞는데도. 스칼라만 좁히면
      // 16/20, 곱마다 좁히면 16/20, 위대로 하면 20/20 에 오차 0 이다.
      const lum = img.isByte
        ? r * LUMA[0] + g * LUMA[1] + b * LUMA[2]
        : Math.fround(Math.fround(Math.fround(r * LUMA32[0])
            + Math.fround(g * LUMA32[1])) + Math.fround(b * LUMA32[2]));
      // `lum.astype(arr.dtype)` — 바이트 쪽만 여기서 좁힌다. 실수 쪽은 위에서
      // 이미 float32 이므로 이 캐스트가 값을 안 바꾼다.
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
  constructor(private readonly numOutputChannels = 1) {}

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
  constructor(private readonly p = 0.1) {}

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
  private readonly size: [number, number];

  /**
   * @param padding **`null` is the default, not `0`** — torchvision's is `None`
   *   and its repr prints that word. They pad identically, so the difference
   *   lives only in the printed line, and the golden's repr case passed
   *   `padding=4` and therefore never printed a default.
   */
  constructor(
    size: number | readonly [number, number],
    private readonly padding: number | null = null,
    private readonly fill = 0,
  ) {
    this.size = typeof size === "number" ? [size, size] : [size[0], size[1]];
  }

  apply(x: Image | Tensor): Image {
    const img = padded(asImage(x, "RandomCrop"), this.padding ?? 0, this.fill);
    const [th, tw] = this.size;
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
    // 파이썬 쪽이 `size` 를 튜플로 정규화해서 들고 있으므로 그 모양으로 찍는다.
    return `RandomCrop(size=(${this.size[0]}, ${this.size[1]}), `
      + `padding=${this.padding === null ? "None" : this.padding})`;
  }
}

/**
 * 출력 자리마다 (읽기 시작점, 가중치). 축 하나에 대한 것이다 — 이 필터는 분리
 * 가능하므로 가로 한 번, 세로 한 번 지나면 된다.
 *
 * **안티에일리어싱을 한다.** torchvision 의 `Resize` 는 기본이 `antialias=true`
 * 이고 끈 것과의 차이가 8×8→4×4 에서 최대 0.0301 이다(실측). 0 이 아니므로
 * "bilinear" 라고만 적으면 어느 쪽인지 안 정해진 것이고, 켠 것으로 학습한 모델에
 * 끈 것을 넣으면 입력이 다르다.
 *
 * 확대일 때는 `support` 가 1 이라 보통의 겹선형과 같아진다 — 갈래가 하나로 족한
 * 이유다. 경계 규칙을 두 가지로 재봤는데 **모든 케이스에서 답이 같았다**(넓힌
 * 자리에서 삼각 필터가 0 이다). torch 의 C 구현과 같은 쪽을 쓴다.
 */
function aaWeights(src: number, dst: number): { at: number; w: Float64Array }[] {
  const scale = src / dst;
  const support = Math.max(1, scale);
  const rows: { at: number; w: Float64Array }[] = [];
  for (let i = 0; i < dst; i++) {
    const center = (i + 0.5) * scale;
    const lo = Math.max(0, Math.floor(center - support + 0.5));
    const hi = Math.min(src, Math.ceil(center + support + 0.5));
    const w = new Float64Array(Math.max(0, hi - lo));
    let total = 0;
    for (let j = lo; j < hi; j++) {
      const v = Math.max(0, 1 - Math.abs((j + 0.5 - center) / support));
      w[j - lo] = v;
      total += v;
    }
    if (total !== 0) for (let k = 0; k < w.length; k++) w[k] = (w[k] ?? 0) / total;
    rows.push({ at: lo, w });
  }
  return rows;
}

/**
 * 짧은 변을 `size` 로. 긴 변은 비율을 지킨다 — torchvision 의 `Resize(int)` 다.
 *
 * **`maxSize` 는 비율 대신이 아니라 비율 뒤에 온다.** 먼저 짧은 변을 `size` 로 맞춰
 * 긴 변을 구하고, 그것이 상한을 넘을 때만 긴 변을 상한으로 자른 뒤 짧은 변이 따라
 * 줄어든다. 두 번의 나눗셈이 **둘 다 버림**이다 — 반올림하면 5×4 를
 * `Resize(8, maxSize=9)` 로 줄일 때 (9, 7) 이 아니라 (9, 8) 이 나온다.
 */
function shortSide(
  h: number, w: number, size: number, maxSize: number | null,
): [number, number] {
  const short = Math.min(h, w);
  const long = Math.max(h, w);
  let newShort = size;
  let newLong = short === size ? long : Math.trunc((size * long) / short);
  if (maxSize !== null && newLong > maxSize) {
    // **여기서 던진다, 만들 때가 아니라.** torchvision 도 크기를 셈하는 안쪽에서
    // 멈추므로 `Resize(4, max_size=4)` 는 세워지고 그림을 받을 때 선다. repr 케이스는
    // 변환을 세우기만 하고 안 부르므로, 앞당겨 던지면 어느 쪽이 갈렸는지가 바뀐다.
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

/** `(H, W, C)` 로 늘어놓은 것의 한 축만 바꾼다. */
function resizeRows(
  src: Float64Array, h: number, w: number, c: number, dst: number, nearest: boolean,
): Float64Array {
  const out = new Float64Array(dst * w * c);
  if (nearest) {
    for (let y = 0; y < dst; y++) {
      const from = Math.trunc(y * (h / dst));
      out.set(src.subarray(from * w * c, (from + 1) * w * c), y * w * c);
    }
    return out;
  }
  const rows = aaWeights(h, dst);
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

/** 세로를 바꾸는 것과 같은 일을 가로에. 축만 다르다. */
function resizeCols(
  src: Float64Array, h: number, w: number, c: number, dst: number, nearest: boolean,
): Float64Array {
  const out = new Float64Array(h * dst * c);
  const rows = nearest ? null : aaWeights(w, dst);
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
    private readonly size: number | readonly [number, number],
    private readonly interpolation: "bilinear" | "nearest" = "bilinear",
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
    const nearest = this.interpolation === "nearest";
    let data = img.data;
    let h = img.height;
    if (th !== h) {
      data = resizeRows(data, h, img.width, img.channels, th, nearest);
      h = th;
    }
    let w = img.width;
    if (tw !== w) {
      data = resizeCols(data, h, w, img.channels, tw, nearest);
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
 * **파이썬의 `round` 는 절반을 짝수로 보낸다.** `Math.round` 는 위로 올린다 —
 * `round(0.5)` 가 파이썬에서 0, JS 에서 1 이다.
 *
 * torchvision 의 `CenterCrop` 이 `int(round(...))` 로 시작점을 잡으므로, 그것을
 * 그대로 흉내 내지 않으면 **자르는 자리가 한 칸 어긋난다.** 값이 조금 다른 것이
 * 아니라 다른 화소가 나오고, 실측으로 최대 0.837 갈렸다 — 파이썬 판과 TypeScript
 * 판을 나란히 대보기 전에는 안 보였다.
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
        // 채운 자리는 원본 밖이다 — 거기서는 `fill` 을 쓴다.
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
function pyFloat(v: number): string {
  return Number.isInteger(v) ? `${v}.0` : String(v);
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
  // **늦게 만든다.** 생성자에서 텐서를 만들면 `describe()` 가 장치를 요구하게 되고,
  // 파이썬 쪽은 GPU 없이 찍힌다. `repr` 을 명세로 보는 이상 그것을 보려고 장치를
  // 띄워야 하는 것은 뒤집힌 것이다 — 실제로 검사가 여기서 걸렸다.
  private matrix: Tensor | null = null;
  private meanVector: Tensor | null = null;

  constructor(
    private readonly rows: readonly (readonly number[])[],
    private readonly mean: readonly number[],
  ) {
    this.side = rows.length;
    if (rows.some((r) => r.length !== this.side)) {
      throw new RuntimeError(
        `transformation_matrix should be square — it received ${rows.length} rows ` +
        `of lengths ${rows.map((r) => r.length).join(", ")}.\n` +
        "(torch: transformation_matrix should be square)");
    }
    if (mean.length !== this.side) {
      throw new RuntimeError(
        `mean_vector should be as long as one side of the matrix ` +
        `(${this.side}, ${this.side}) — it received ${mean.length}.\n` +
        "(torch: mean_vector should have the same length)");
    }
  }

  private tensors(): [Tensor, Tensor] {
    this.matrix ??= Tensor.from(this.rows.flat(), [this.side, this.side]);
    this.meanVector ??= Tensor.from([...this.mean], [this.side]);
    return [this.matrix, this.meanVector];
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
    const rows = this.rows.map((r) => `[${r.map(pyFloat).join(", ")}]`);
    return `LinearTransformation(transformation_matrix=[${rows.join(", ")}]` +
      `, mean_vector=[${this.mean.map(pyFloat).join(", ")}])`;
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
    private readonly scale: readonly [number, number] = [0.08, 1.0],
    private readonly ratio: readonly [number, number] = [3 / 4, 4 / 3],
    private readonly interpolation: "bilinear" | "nearest" = "bilinear",
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
    private readonly p = 0.5,
    private readonly scale: readonly [number, number] = [0.02, 0.33],
    private readonly ratio: readonly [number, number] = [0.3, 3.3],
    private readonly value: number | readonly number[] | "random" = 0,
    private readonly inplace = false,
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
    private readonly verticalFlip = false,
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

function padded(img: Image, pad: number, fill: number): Image {
  if (pad === 0) return img;
  const height = img.height + 2 * pad;
  const width = img.width + 2 * pad;
  const out = new Float64Array(height * width * img.channels).fill(fill);
  for (let h = 0; h < img.height; h++) {
    for (let w = 0; w < img.width; w++) {
      const from = (h * img.width + w) * img.channels;
      const to = ((h + pad) * width + (w + pad)) * img.channels;
      for (let c = 0; c < img.channels; c++) out[to + c] = img.data[from + c] ?? 0;
    }
  }
  return { ...img, data: out, height, width };
}
