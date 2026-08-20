/**
 * Draws tensors so they can be seen — images and curves.
 *
 * **The tutorials do not hold up without this.** Showing the image a classifier got
 * wrong is shorter than describing it, and a line coming down is more exact than the
 * sentence "the loss goes down". The lesson blocks and the playground use the same
 * code.
 */

import { t } from "./i18n.js";

/** Spreads a tensor's shape into (count, channels, height, width), saying what is wrong if it cannot. */
function layout(shape) {
  if (shape.length === 4) return { n: shape[0], c: shape[1], h: shape[2], w: shape[3] };
  if (shape.length === 3) return { n: 1, c: shape[0], h: shape[1], w: shape[2] };
  if (shape.length === 2) return { n: 1, c: 1, h: shape[0], w: shape[1] };
  throw new Error(
    t("draw.rank", shape));
}

/**
 * Draws a tensor onto a canvas and returns it.
 *
 * @param tensor  a borch tensor (reading its values is awaited here)
 * @param options `{ scale, labels, max, range, width }`
 *                `range` is the value span — by default it stretches to the real
 *                minimum and maximum. To see a normalised image as it was, pass
 *                something like `[-1, 1]`.
 */
export async function drawTensor(tensor, options = {}) {
  const { scale = 3, labels = null, max = 64, range = null, width = 720 } = options;
  const shape = tensor.shape;
  const { n, c, h, w } = layout(shape);
  if (c !== 1 && c !== 3) {
    throw new Error(t("draw.channels", c));
  }

  const values = await tensor.toArray();
  const count = Math.min(n, max);
  // How many per row. It wraps once the row grows too long.
  const cols = Math.min(count, Math.max(1, Math.floor(width / (w * scale + 4))));
  const rows = Math.ceil(count / cols);
  const labelRoom = labels ? 14 : 0;

  const canvas = document.createElement("canvas");
  const dpr = window.devicePixelRatio || 1;
  const cw = cols * (w * scale + 4) + 4;
  const ch = rows * (h * scale + 4 + labelRoom) + 4;
  canvas.width = cw * dpr;
  canvas.height = ch * dpr;
  canvas.style.width = `${cw}px`;
  canvas.style.height = `${ch}px`;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.imageSmoothingEnabled = false;

  // The value span. **Not stretched per image** — stretching each one differently
  // leaves them at different brightnesses and nothing can be compared. They are drawn
  // side by side in order to be compared, so the span covers all of them.
  let lo = 0, hi = 1;
  if (range) {
    [lo, hi] = range;
  } else {
    lo = Infinity; hi = -Infinity;
    for (const v of values) { if (v < lo) lo = v; if (v > hi) hi = v; }
    if (!(hi > lo)) { lo = 0; hi = 1; }
  }
  const span = hi - lo || 1;

  const plane = h * w;
  for (let i = 0; i < count; i++) {
    const tile = ctx.createImageData(w, h);
    for (let p = 0; p < plane; p++) {
      const at = i * c * plane + p;
      const to = (v) => Math.max(0, Math.min(255, Math.round(((v - lo) / span) * 255)));
      const r = to(values[at]);
      const g = c === 3 ? to(values[at + plane]) : r;
      const b = c === 3 ? to(values[at + 2 * plane]) : r;
      tile.data[p * 4] = r;
      tile.data[p * 4 + 1] = g;
      tile.data[p * 4 + 2] = b;
      tile.data[p * 4 + 3] = 255;
    }
    // createImageData cannot scale directly, so it is drawn once and then scaled.
    const small = document.createElement("canvas");
    small.width = w;
    small.height = h;
    small.getContext("2d").putImageData(tile, 0, 0);

    const col = i % cols, row = Math.floor(i / cols);
    const x = 4 + col * (w * scale + 4);
    const y = 4 + row * (h * scale + 4 + labelRoom);
    ctx.drawImage(small, x, y, w * scale, h * scale);

    if (labels) {
      ctx.font = "11px ui-monospace, monospace";
      ctx.fillStyle = getComputedStyle(document.documentElement)
        .getPropertyValue("--fg-dim").trim() || "#888";
      ctx.textAlign = "center";
      const text = String(labels[i] ?? "");
      ctx.fillText(text.slice(0, 12), x + (w * scale) / 2, y + h * scale + 11);
    }
  }
  return canvas;
}

/** A run of numbers as a line. The loss curve is its first user. */
export function drawSeries(series, options = {}) {
  const { width = 520, height = 120, name = "" } = options;
  const canvas = document.createElement("canvas");
  const dpr = window.devicePixelRatio || 1;
  canvas.width = width * dpr;
  canvas.height = height * dpr;
  canvas.style.width = `${width}px`;
  canvas.style.height = `${height}px`;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  const css = getComputedStyle(document.documentElement);
  const ink = css.getPropertyValue("--ember").trim() || "#ff7a3d";
  const faint = css.getPropertyValue("--fg-faint").trim() || "#6b788c";
  ctx.font = "11px ui-monospace, monospace";

  if (series.length < 2) {
    ctx.fillStyle = faint;
    ctx.fillText(t("draw.empty"), 10, height / 2);
    return canvas;
  }

  const lo = Math.min(...series), hi = Math.max(...series);
  const pad = 16, span = hi - lo || 1;
  const xAt = (i) => pad + (i / (series.length - 1)) * (width - pad * 2);
  const yAt = (v) => height - pad - ((v - lo) / span) * (height - pad * 2);

  ctx.strokeStyle = faint;
  ctx.globalAlpha = 0.3;
  ctx.beginPath();
  ctx.moveTo(pad, height - pad);
  ctx.lineTo(width - pad, height - pad);
  ctx.stroke();
  ctx.globalAlpha = 1;

  ctx.strokeStyle = ink;
  ctx.lineWidth = 1.6;
  ctx.beginPath();
  series.forEach((v, i) => (i ? ctx.lineTo(xAt(i), yAt(v)) : ctx.moveTo(xAt(i), yAt(v))));
  ctx.stroke();

  ctx.fillStyle = faint;
  ctx.fillText(`${name} ${series[series.length - 1].toPrecision(4)}`, pad, 12);
  ctx.textAlign = "right";
  ctx.fillText(hi.toPrecision(3), width - pad, 12);
  ctx.fillText(lo.toPrecision(3), width - pad, height - pad - 3);
  return canvas;
}
