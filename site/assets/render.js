/**
 * 텐서를 눈에 보이게 그린다 — 이미지와 곡선.
 *
 * **튜토리얼이 이것 없이는 성립하지 않는다.** 분류기가 무엇을 틀렸는지 글로 적는
 * 것보다 그 이미지를 보여주는 편이 짧고, 손실이 내려간다는 말보다 내려가는 선이
 * 정확하다. 강의의 실행 블록과 플레이그라운드가 같은 것을 쓴다.
 */

/** 텐서 모양을 (장수, 채널, 높이, 너비) 로 편다. 못 펴면 무엇이 문제인지 말한다. */
function layout(shape) {
  if (shape.length === 4) return { n: shape[0], c: shape[1], h: shape[2], w: shape[3] };
  if (shape.length === 3) return { n: 1, c: shape[0], h: shape[1], w: shape[2] };
  if (shape.length === 2) return { n: 1, c: 1, h: shape[0], w: shape[1] };
  throw new Error(
    `show() 는 [H,W]·[C,H,W]·[N,C,H,W] 를 그린다 — 받은 것은 [${shape}] 다.`);
}

/**
 * 텐서를 캔버스에 그려 돌려준다.
 *
 * @param tensor  borch 텐서 (값 읽기는 여기서 await 한다)
 * @param options `{ scale, labels, max, range, width }`
 *                `range` 는 값의 범위 — 기본은 실제 최소·최대로 늘린다.
 *                정규화된 이미지를 원래대로 보고 싶으면 `[-1, 1]` 처럼 준다.
 */
export async function drawTensor(tensor, options = {}) {
  const { scale = 3, labels = null, max = 64, range = null, width = 720 } = options;
  const shape = tensor.shape;
  const { n, c, h, w } = layout(shape);
  if (c !== 1 && c !== 3) {
    throw new Error(`채널이 1 이나 3 이어야 그린다 — 받은 것은 ${c} 다.`);
  }

  const values = await tensor.toArray();
  const count = Math.min(n, max);
  // 한 줄에 몇 장. 너무 길어지면 접는다.
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

  // 값의 범위. **한 장씩 늘리지 않는다** — 장마다 다르게 늘리면 밝기가 서로
  // 달라져서 비교가 안 된다. 비교하라고 나란히 그리는 것이므로 전체로 늘린다.
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
    // createImageData 는 바로 못 키운다. 한 번 그려서 키운다.
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

/** 수의 흐름을 선으로. 손실 곡선이 첫 사용자다. */
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
    ctx.fillText("아직 그릴 것이 없다", 10, height / 2);
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
