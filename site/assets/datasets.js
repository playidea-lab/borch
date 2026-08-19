/**
 * 튜토리얼이 쓰는 데이터 — 브라우저에서 읽어 텐서로.
 *
 * 그림 한 장에 타일로 담아 두고 캔버스로 되읽는다. 원본 바이너리는 배치당 29MB 라
 * 페이지 하나 열자고 받게 할 수 없고, 스프라이트 + JPEG 로 1MB 아래가 된다.
 * 만드는 쪽은 `site/fetch_data.py` 이고, 그 파일들은 `.gitignore` 다 —
 * `vendor/pyodide` 와 같은 자리다.
 *
 * **픽셀이 원본과 완전히 같지 않다**(JPEG). 여기서 나온 정확도를 논문의 수와
 * 비교하면 안 된다. 이 데이터가 답하는 질문은 "학습이 되는가" 다.
 */

import { t } from "./i18n.js";

const HERE = new URL("./data/", import.meta.url).href;
const cache = new Map();

/**
 * CIFAR-10 부분집합.
 *
 * @param split "train"(2000장) 또는 "test"(500장)
 * @param options `{ count, normalize }` — normalize 가 참이면 채널당 평균 0.5·표준편차 0.5
 * @returns `{ x: [N,3,32,32], y: [N] int64, classes, count }`
 */
export async function cifar10(Tensor, split = "train", options = {}) {
  const { count = 0, normalize = false } = options;
  const key = `${split}:${count}:${normalize}`;
  if (cache.has(key)) return cache.get(key);

  const meta = await grab(`${HERE}cifar-${split}.json`, "json");
  const blob = await grab(`${HERE}cifar-${split}.jpg`, "blob");
  const bitmap = await createImageBitmap(blob);

  const n = count > 0 ? Math.min(count, meta.count) : meta.count;
  const { cols, tile } = meta;

  // 한 번에 다 그려 놓고 픽셀을 읽는다. 타일마다 읽으면 호출이 2000 번이다.
  const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(bitmap, 0, 0);
  const sheet = ctx.getImageData(0, 0, bitmap.width, bitmap.height).data;

  const plane = tile * tile;
  const pixels = new Float32Array(n * 3 * plane);
  for (let i = 0; i < n; i++) {
    const row = Math.floor(i / cols) * tile;
    const col = (i % cols) * tile;
    for (let y = 0; y < tile; y++) {
      for (let x = 0; x < tile; x++) {
        const from = ((row + y) * bitmap.width + col + x) * 4;
        const to = i * 3 * plane + y * tile + x;
        for (let c = 0; c < 3; c++) {
          let v = sheet[from + c] / 255;
          if (normalize) v = (v - 0.5) / 0.5;
          pixels[to + c * plane] = v;
        }
      }
    }
  }

  const out = {
    x: Tensor.from(pixels, [n, 3, tile, tile]),
    y: Tensor.from(Float32Array.from(meta.labels.slice(0, n)), [n], { dtype: "int64" }),
    classes: meta.classes,
    count: n,
    note: meta.note,
  };
  cache.set(key, out);
  return out;
}

async function grab(url, as) {
  const res = await fetch(url);
  if (!res.ok) {
    // **없는 것과 못 읽은 것을 가른다.** 데이터는 `.gitignore` 라 클론만 한 사람에게는
    // 없는 것이 정상이고, 그때 필요한 것은 오류가 아니라 다음에 칠 명령이다.
    //
    // 이 문구가 한국어로만 박혀 있었다. 영어 페이지를 연 사람이 데이터를 안 만들었을
    // 때 **읽을 수 없는 문장**을 받았다 — 데이터가 없는 것은 클론 직후의 정상 상태라
    // 그 사람이 첫 방문에서 만나는 문장이기도 하다.
    throw new Error(t("data.missing", res.status));
  }
  return as === "json" ? res.json() : res.blob();
}
