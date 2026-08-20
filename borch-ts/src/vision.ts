/**
 * `torchvision.transforms` 모양의 변환.
 *
 * ## 왜 텐서 라이브러리 안에 있나
 *
 * 학습을 돌리려면 이미지를 텐서로 바꾸는 자리가 반드시 있고, 그 자리에서 조용히
 * 틀리는 것들이 있다 — 아래 `ToTensor` 의 주석이 그 중 하나다. 저장소의 파이썬 판
 * (`borchvision.py`)과 **같은 규칙**을 쓴다. 규칙이 갈리면 같은 데이터가
 * 라이브러리마다 다른 값이 되고, 그것은 값 대조로만 잡힌다.
 *
 * ## 이미지는 배열이지 텐서가 아니다
 *
 * `RandomHorizontalFlip`·`RandomCrop` 은 `(H, W, C)` 배열을 받는다. torchvision 에서
 * 이 자리에 오는 것이 PIL 이미지이고 우리에게 PIL 이 없어서 배열이 그 자리를 대신한다.
 * 장당 텐서를 만들면 GPU 버퍼가 장당 하나씩 생기는데, 그건 되는 것처럼 보이다가
 * 메모리로 무너진다.
 */

import { Tensor } from "./tensor.js";

/** `(H, W, C)` 로 늘어놓은 이미지. 값은 uint8 이거나 실수다. */
export interface Image {
  readonly data: Float64Array;
  readonly height: number;
  readonly width: number;
  readonly channels: number;
  /** uint8 이면 `ToTensor` 가 255 로 나눈다. */
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

export interface Transform {
  apply(x: Image | Tensor): Image | Tensor;
  describe(): string;
}

/** 변환을 줄줄이. `repr` 이 안쪽을 들여쓴 여러 줄이라 그 모양까지 맞춘다. */
export class Compose implements Transform {
  constructor(private readonly transforms: readonly Transform[]) {}

  apply(x: Image | Tensor): Image | Tensor {
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
 * `(H,W,C)` 또는 `(H,W)` → `(C,H,W)` 텐서.
 *
 * **uint8 일 때만 255 로 나눈다** — torchvision 의 규칙이 그렇다. 실수 배열을 넣으면
 * 나누지 않고 그대로 옮긴다. 이 구분을 놓치면 이미 [0,1] 인 데이터가 한 번 더 나뉘어
 * 255 배 어두워지는데, **예외는 안 나고 학습만 안 된다.**
 */
export class ToTensor implements Transform {
  apply(x: Image | Tensor): Tensor {
    if (x instanceof Tensor) return x;
    const { data, height, width, channels, isByte } = x;
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

/** 채널마다 `(x - mean) / std`. */
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

/** 좌우로 뒤집는다. `(H, W, C)` 배열을 받는다. */
export class RandomHorizontalFlip implements Transform {
  constructor(private readonly p = 0.5) {}

  apply(x: Image | Tensor): Image {
    const img = asImage(x, "RandomHorizontalFlip");
    if (nextFloat() >= this.p) return img;
    const out = new Float64Array(img.data.length);
    for (let h = 0; h < img.height; h++) {
      for (let w = 0; w < img.width; w++) {
        const from = (h * img.width + (img.width - 1 - w)) * img.channels;
        const to = (h * img.width + w) * img.channels;
        for (let c = 0; c < img.channels; c++) out[to + c] = img.data[from + c] ?? 0;
      }
    }
    return { ...img, data: out };
  }

  describe(): string {
    return `RandomHorizontalFlip(p=${this.p})`;
  }
}

/** 가장자리를 채운 뒤 무작위로 잘라낸다. */
export class RandomCrop implements Transform {
  private readonly size: [number, number];

  constructor(
    size: number | readonly [number, number],
    private readonly padding = 0,
    private readonly fill = 0,
  ) {
    this.size = typeof size === "number" ? [size, size] : [size[0], size[1]];
  }

  apply(x: Image | Tensor): Image {
    const img = padded(asImage(x, "RandomCrop"), this.padding, this.fill);
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
    return `RandomCrop(size=(${this.size[0]}, ${this.size[1]}), padding=${this.padding})`;
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

/** 짧은 변을 `size` 로. 긴 변은 비율을 지킨다 — torchvision 의 `Resize(int)` 다. */
function shortSide(h: number, w: number, size: number): [number, number] {
  const short = Math.min(h, w);
  const long = Math.max(h, w);
  if (short === size) return [h, w];
  const grown = Math.trunc((size * long) / short);
  return h < w ? [size, grown] : [grown, size];
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
 * 크기를 바꾼다. **배열을 받아 배열을 낸다** — 텐서가 아니다. 장당 텐서를 만들면
 * GPU 버퍼가 장당 하나씩 생긴다(`RandomCrop` 과 같은 이유).
 *
 * `size` 가 수 하나면 **짧은 변**을 그 값으로 맞추고 비율을 지킨다.
 */
export class Resize implements Transform {
  constructor(
    private readonly size: number | readonly [number, number],
    private readonly interpolation: "bilinear" | "nearest" = "bilinear",
  ) {}

  apply(x: Image | Tensor): Image {
    const img = asImage(x, "Resize");
    const [th, tw] = typeof this.size === "number"
      ? shortSide(img.height, img.width, this.size)
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

  describe(): string {
    const size = typeof this.size === "number" ? this.size : `(${this.size.join(", ")})`;
    return `Resize(size=${size}, interpolation=${this.interpolation})`;
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
 * 가운데를 잘라낸다. **자를 크기가 원본보다 크면 0 으로 채운 뒤 자른다** —
 * torchvision 이 그렇게 하고, 거절하면 같은 코드가 두 라이브러리에서 갈린다.
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
 * `(N,C,H,W)` 배치를 한 번에 늘린다. **torchvision 에 없는 우리 것이다.**
 *
 * 장당 텐서를 만들면 GPU 버퍼가 장당 하나씩 생긴다 — 한 에폭이 만 장이면 만 개다.
 * 배치를 CPU 에서 다 늘린 뒤 텐서를 **하나** 만드는 것이 감당되는 유일한 순서라,
 * 그 순서를 이름 붙여 내놓는다.
 *
 * **뽑기는 장마다 따로 한다.** 배치 전체에 같은 자르기·뒤집기를 쓰면 배치 안에서
 * 늘어난 것이 없어 augmentation 의 효과가 사라진다.
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

/** 채널마다 `(x - mean) / std`. CPU 에서 배치째 — 텐서를 만들기 전에 끝낸다. */
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

function asImage(x: Image | Tensor, who: string): Image {
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
