/**
 * The data the tutorials use — read in the browser, turned into tensors.
 *
 * It sits as tiles on one image and is read back through a canvas. The original
 * binary is 29MB per batch, which is not a thing to make someone download to open
 * one page; a sprite plus JPEG comes in under 1MB. `site/fetch_data.py` makes them,
 * and they are committed — the same place `vendor/pyodide` sits.
 *
 * **The pixels are not exactly the original** (JPEG). Accuracy from here is not to be
 * compared against a paper's numbers. The question this data answers is whether
 * training happens at all.
 */

import { pick, t } from "./i18n.js";

const HERE = new URL("./data/", import.meta.url).href;
const cache = new Map();

/**
 * A subset of CIFAR-10.
 *
 * @param split "train" (2,000 images) or "test" (500)
 * @param options `{ count, normalize }` — normalize true means mean 0.5, std 0.5 per channel
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

  // Draw once and read the pixels. Reading per tile is two thousand calls.
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
    note: pick(meta.note),
  };
  cache.set(key, out);
  return out;
}

async function grab(url, as) {
  const res = await fetch(url);
  if (!res.ok) {
    // **Absent and unreadable are different things.** The data is committed now, so
    // absent means something went wrong with the fetch rather than "you have not built
    // it yet" — but the message still has to be the next command to type, not a status
    // code on its own.
    //
    // This wording was once hardcoded in Korean. Someone opening the English page
    // without the data got **a sentence they could not read**, and back then a missing
    // file was the normal state right after a clone, so it was the sentence they met
    // on a first visit.
    throw new Error(t("data.missing", res.status));
  }
  return as === "json" ? res.json() : res.blob();
}
