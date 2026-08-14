/**
 * `print(t)` 가 진짜 torch 와 같게 찍히도록.
 *
 * ## 왜 글자를 명세로 두는가
 *
 * 배우는 사람이 가장 많이 하는 일이 `print(tensor)` 다. 다르게 찍히면 교재의 예시와
 * 화면이 안 맞고, 그때마다 "내가 뭘 잘못했나" 를 의심하게 된다. 값이 맞아도 그렇다.
 *
 * ## torch 의 규칙 — 재서 옮긴 것이다
 *
 * - 실수인데 값이 전부 정수면 `1.` 로 찍는다. 아니면 소수 넷째 자리까지.
 * - 가장 큰 값과 가장 작은 값의 비가 크거나 값이 아주 크면 지수형으로 바꾼다.
 * - 자리를 오른쪽 맞춤으로 채운다 — 음수가 섞이면 양수 앞에 빈칸이 하나 생긴다.
 * - 한 줄이 80 자를 넘으면 자르고 여덟 칸 들여쓴다(`tensor([` 의 길이다).
 * - 3차원부터는 덩어리 사이에 빈 줄이 하나 들어간다.
 */

import type { DType } from "./dtype.js";

/** 한 줄의 최대 길이. `tensor([` 를 포함해서 센다. */
const LINE_WIDTH = 80;
const INDENT = "        "; // "tensor([" 와 같은 길이

/** 지수형으로 넘어가는 문턱. 큰 값과 작은 값의 비가 이보다 크면 바꾼다. */
const SCI_RATIO = 1000;
const SCI_LARGE = 1e8;
const DECIMALS = 4;

interface Style {
  /** 원소 하나를 글자로. */
  readonly format: (v: number) => string;
  /** 오른쪽 맞춤에 쓸 너비. */
  readonly width: number;
}

function styleFor(values: readonly number[], dtype: DType): Style {
  if (dtype === "bool") {
    const format = (v: number): string => (v !== 0 ? "True" : "False");
    return { width: widest(values, format), format };
  }
  if (dtype === "int64") {
    const format = (v: number): string => String(Math.trunc(v));
    return { width: widest(values, format), format };
  }
  const finite = values.filter((v) => Number.isFinite(v));
  const magnitudes = finite.map(Math.abs).filter((v) => v !== 0);
  const hi = magnitudes.length > 0 ? Math.max(...magnitudes) : 0;
  const lo = magnitudes.length > 0 ? Math.min(...magnitudes) : 0;
  // 값이 전부 정수면 torch 는 소수점만 찍고 끝낸다 — `1.` 이지 `1.0000` 이 아니다.
  const allIntegral = finite.every((v) => Number.isInteger(v));
  const scientific = hi >= SCI_LARGE || (lo > 0 && hi / lo > SCI_RATIO);
  let format: (v: number) => string;
  if (scientific) {
    format = (v) => exponential(v);
  } else if (allIntegral) {
    format = (v) => (Number.isFinite(v) ? `${Math.trunc(v)}.` : special(v));
  } else {
    format = (v) => (Number.isFinite(v) ? v.toFixed(DECIMALS) : special(v));
  }
  return { width: widest(values, format), format };
}

function special(v: number): string {
  if (Number.isNaN(v)) return "nan";
  return v > 0 ? "inf" : "-inf";
}

/** `1.0000e+06` — 지수는 최소 두 자리다. torch 가 그렇게 찍는다. */
function exponential(v: number): string {
  if (!Number.isFinite(v)) return special(v);
  const raw = v.toExponential(DECIMALS);
  return raw.replace(/e([+-])(\d)$/, "e$10$2");
}

function widest(values: readonly number[], format: (v: number) => string): number {
  let w = 0;
  for (const v of values) w = Math.max(w, format(v).length);
  return w;
}

/**
 * 한 줄에 늘어놓되 80 자를 넘으면 자른다.
 *
 * @param depth 여는 대괄호가 몇 개나 앞에 있는가 — 들여쓰기가 그만큼 더 들어간다.
 */
function wrapRow(cells: readonly string[], depth: number): string {
  const pad = INDENT + " ".repeat(depth);
  const lines: string[] = [];
  let line = "";
  for (const [i, cell] of cells.entries()) {
    const piece = i === 0 ? cell : `, ${cell}`;
    const prefix = lines.length === 0 ? "tensor([".length + depth : pad.length;
    if (line !== "" && prefix + line.length + piece.length + 1 > LINE_WIDTH) {
      lines.push(line);
      line = cell;
    } else {
      line += piece;
    }
  }
  lines.push(line);
  return lines.join(`,\n${pad}`);
}

/** 중첩된 대괄호를 만든다. 3차원부터는 덩어리 사이에 빈 줄이 하나 들어간다. */
function block(
  values: readonly number[],
  shape: readonly number[],
  style: Style,
  depth: number,
): string {
  if (shape.length === 0) return style.format(values[0] ?? 0);
  if (shape.length === 1) {
    const cells = values.map((v) => style.format(v).padStart(style.width));
    return `[${wrapRow(cells, depth)}]`;
  }
  const outer = shape[0] ?? 0;
  const inner = shape.slice(1);
  const stride = inner.reduce((a, b) => a * b, 1);
  const parts: string[] = [];
  for (let i = 0; i < outer; i++) {
    parts.push(block(values.slice(i * stride, (i + 1) * stride), inner, style, depth + 1));
  }
  // 랭크가 3 이상이면 덩어리 사이가 한 줄 더 벌어진다.
  const gap = shape.length >= 3 ? "\n\n" : "\n";
  // 대괄호가 하나 더 있는 만큼만 더 들여쓴다. 여기에 빈칸을 하나 더 붙이면 한 칸씩
  // 밀리는데, 눈으로는 잘 안 보이고 글자 대조에서만 걸린다.
  const pad = INDENT + " ".repeat(depth);
  return `[${parts.join(`,${gap}${pad}`)}]`;
}

export interface ReprInfo {
  readonly values: readonly number[];
  readonly shape: readonly number[];
  readonly dtype: DType;
  /** 잎이면서 기울기를 받는가. */
  readonly requiresGrad: boolean;
  /** 잎이 아니면 `grad_fn` 이름. */
  readonly gradName: string;
}

export function formatTensor(info: ReprInfo): string {
  const style = styleFor(info.values, info.dtype);
  const body = info.values.length === 0
    ? "[]"
    : block(info.values, info.shape, style, 0);
  const parts = [body];
  if (info.gradName !== "") parts.push(`grad_fn=<${info.gradName}>`);
  else if (info.requiresGrad) parts.push("requires_grad=True");
  return `tensor(${parts.join(", ")})`;
}

/** `torch.Size([2, 2])`. 모양도 찍히는 것이라 명세다. */
export function formatSize(shape: readonly number[]): string {
  return `torch.Size([${shape.join(", ")}])`;
}
