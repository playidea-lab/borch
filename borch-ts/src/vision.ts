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
