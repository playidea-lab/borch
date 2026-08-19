/**
 * `torch.fft` — 이산 푸리에 변환, 그리고 그 위의 `stft`/`istft`.
 *
 * **커널이 하나다.** 정변환·역변환·반쪽 변환·그 셋의 역방향이 전부 같은 셰이더를
 * 부호와 배율만 바꿔 부른다. 나눠 쓰면 여섯 벌이 되고, 그중 하나가 다르게 고쳐지는
 * 날이 온다 — 이 저장소가 반복해서 진 자리가 그 종류다.
 *
 * ## O(n²) 다 — 이름은 `fft` 인데
 *
 * 직접 DFT 를 돈다. 쿨리-튜키를 쓰면 2 의 거듭제곱에서만 빠르고, 아닌 길이는
 * 블루스타인이 따로 필요하다. **값은 어느 쪽이든 같고**, 이 프로젝트의 천장(교재
 * 크기의 신호)에서는 차이가 안 보인다. 빨라야 하는 날이 오면 그때 바꾸되,
 * **지금 없는 속도를 있는 것처럼 적지 않는다.**
 *
 * ## 축을 안 옮긴다
 *
 * `dim` 이 어디든 자료를 옮기지 않는다. `(바깥, 축, 안쪽)` 세 수로 자리를 계산해
 * 셰이더가 바로 집는다 — 옮기고 되돌리면 버퍼 두 벌이 더 들고, 복소수 옮기기는
 * 아직 없는 연산이라 그 자리를 만들어야 했을 것이다.
 *
 * ## 기울기
 *
 * 변환은 **선형**이라 역방향이 규약에서 바로 나온다. 정칙 함수의 역방향이
 * `conj(f')·g` 이므로 `grad_x[j] = Σ_k e^{+2πijk/n}·g[k]` — **정규화 없는 역변환**이다.
 * 어려운 자리는 값이 아니라 **어느 쪽 반쪽을 세는가** 다:
 *
 * * `rfft` 의 역방향은 저장된 반쪽에만 기울기가 온다 — 켤레 짝을 더하면 두 배가 된다.
 * * `irfft` 의 역방향은 **가장자리만 한 번, 가운데는 두 번** 센다 — 되살린 켤레 짝이
 *   같은 저장 칸에서 왔기 때문이다.
 *
 * 둘 다 **순방향 값은 멀쩡한 채로** 틀릴 수 있다. 코어(numpy)가 먼저 같은 유도를
 * 지났고 골든이 둘을 따로 묻는다.
 */

import { RuntimeError } from "./errors.js";
import { device, makeNode, Tensor } from "./tensor.js";

/** `n` 은 변환 길이, `nIn`·`nOut` 은 실제로 든 칸과 낼 칸. */
interface DftPlan {
  readonly n: number;
  readonly nIn: number;
  readonly nOut: number;
  /** −1 이면 정변환, +1 이면 역변환. */
  readonly sign: number;
  readonly scale: number;
  /** 입력이 인터리브 복소수인가. */
  readonly inComplex: boolean;
  /** 저장된 반쪽을 켤레로 되살릴 것인가(`irfft`). */
  readonly hermitian: boolean;
  /** 결과의 실수부만 쓸 것인가(`irfft`). */
  readonly realOut: boolean;
  /** 축 앞쪽 칸 수와 뒤쪽 칸 수. 축을 안 옮기려고 든다. */
  readonly outer: number;
  readonly inner: number;
}

function shader(p: DftPlan): string {
  const threads = p.outer * p.nOut * p.inner;
  const stride = Math.min(Math.max(1, Math.ceil(threads / 64)), 65535) * 64;
  const step = p.inComplex ? 2 : 1;
  // **입력 한 칸 집기.** 없는 칸은 0 이고(길이를 늘려 물은 자리), 허미시안이면
  // 반대쪽 짝을 켤레로 되살린다.
  // **허수부는 실수부 바로 옆이다.** 복소수 저장은 칸마다 `(re, im)` 을 끼워
  // 넣는다 — 쓰는 쪽이 `Out[o*2]`·`Out[o*2+1]` 로 그렇게 적고 있다.
  //
  // 여기 오래 `A[at + inner]` 라고 적혀 있었다. **마지막 축에서는 `inner` 가 1 이라
  // 우연히 같다.** 그래서 1 차원 케이스는 전부 통과했고, 복소수를 **마지막이 아닌
  // 축**으로 변환하는 순간 허수부를 엉뚱한 칸에서 읽었다. 실수 입력은 이 줄을
  // 아예 안 지나므로 두 축 다 맞았다 — 세 조건이 겹쳐야 드러나는 자리였다.
  const fetch = p.hermitian
    ? `
    var xr = 0.0;
    var xi = 0.0;
    if (j < ${p.nIn}u) {
      let at = base + j * ${p.inner * step}u;
      xr = A[at];
      xi = A[at + 1u];
    } else if (${p.n}u - j < ${p.nIn}u) {
      let at = base + (${p.n}u - j) * ${p.inner * step}u;
      xr = A[at];
      xi = -A[at + 1u];
    }`
    : `
    var xr = 0.0;
    var xi = 0.0;
    if (j < ${p.nIn}u) {
      let at = base + j * ${p.inner * step}u;
      xr = A[at];
      ${p.inComplex ? "xi = A[at + 1u];" : ""}
    }`;
  const write = p.realOut
    ? `  Out[o] = re * ${p.scale};`
    : `  Out[o * 2u] = re * ${p.scale};\n  Out[o * 2u + 1u] = im * ${p.scale};`;
  return `
@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> Tw: array<f32>;
@group(0) @binding(2) var<storage, read_write> Out: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let gid = g.y * ${stride}u + g.x;
  if (gid >= ${threads}u) { return; }
  let outerIdx = gid / ${p.nOut * p.inner}u;
  let rest = gid % ${p.nOut * p.inner}u;
  let k = rest / ${p.inner}u;
  let innerIdx = rest % ${p.inner}u;
  // 입력에서 이 줄이 시작하는 자리. 복소수면 칸당 둘이라 안쪽 걸음이 두 배다.
  // (셰이더 글은 템플릿 문자열 안에 있다 — 여기 역따옴표를 쓰면 문자열이 닫힌다.)
  let base = outerIdx * ${p.nIn * p.inner * step}u + innerIdx * ${step}u;
  var re = 0.0;
  var im = 0.0;
  for (var j = 0u; j < ${p.n}u; j = j + 1u) {
${fetch}
    // **회전인자는 표에서 읽는다 — 셰이더에서 cos/sin 을 안 부른다.**
    //
    // 각을 계산해 부르는 판이 먼저 있었는데, 사각창 stft 한 자리가 상대오차
    // 2.7e-4 로 골든을 벗어났다. f32 반올림으로는 설명이 안 되는 크기이고,
    // WGSL 의 삼각함수는 정확도가 구현에 맡겨져 있다(소프트웨어 어댑터에서 특히).
    // 표를 호스트에서 **배정도로** 만들어 올리면 그 자리가 아예 없어지고, 안쪽
    // 고리에서 초월함수도 사라진다.
    let m = (j * k) % ${p.n}u;
    let c = Tw[m * 2u];
    let s = ${p.sign > 0 ? "" : "-"}Tw[m * 2u + 1u];
    re = re + xr * c - xi * s;
    im = im + xr * s + xi * c;
  }
  let o = outerIdx * ${p.nOut * p.inner}u + k * ${p.inner}u + innerIdx;
${write}
}`;
}

/**
 * 회전인자 표. `m = 0…n−1` 에 대한 `cos(2πm/n)`·`sin(2πm/n)` 을 번갈아 담는다.
 *
 * **호스트에서 배정도로 만든다.** 부호는 셰이더가 붙이므로 표는 한 벌이면 되고,
 * 같은 `n` 이면 같은 버퍼를 다시 쓴다 — 학습 고리에서 매 스텝 새로 올리면 그것이
 * 곧 누수처럼 보인다.
 */
const twiddles = new Map<number, GPUBuffer>();

function twiddleOf(n: number): GPUBuffer {
  const had = twiddles.get(n);
  if (had) return had;
  const table = new Float32Array(n * 2);
  for (let m = 0; m < n; m++) {
    const ang = (2 * Math.PI * m) / n;
    table[m * 2] = Math.cos(ang);
    table[m * 2 + 1] = Math.sin(ang);
  }
  const buf = device().upload(table);
  device().keep(buf);
  twiddles.set(n, buf);
  return buf;
}

/** 커널 한 번. 모양은 부르는 쪽이 안다. */
function run(input: Tensor, p: DftPlan): Tensor {
  const count = p.outer * p.nOut * p.inner;
  const dev = device();
  const out = dev.alloc(count * (p.realOut ? 1 : 2));
  const key = `dft:${p.n}:${p.nIn}:${p.nOut}:${p.sign}:${p.scale}:` +
    `${p.inComplex}:${p.hermitian}:${p.realOut}:${p.outer}:${p.inner}`;
  dev.run1d(dev.pipeline(key, () => shader(p)),
            [input.raw, twiddleOf(p.n), out], count);
  return new Tensor(out, [count], {
    dtype: p.realOut ? "float32" : "complex64",
  });
}

function axisOf(dim: number, rank: number): number {
  const axis = dim < 0 ? dim + rank : dim;
  if (axis < 0 || axis >= rank) {
    throw new RuntimeError(`축 ${dim} 은 랭크 ${rank} 에 없다.`);
  }
  return axis;
}

function split(shape: readonly number[], axis: number): [number, number] {
  let outer = 1;
  let inner = 1;
  for (let i = 0; i < axis; i++) outer *= shape[i] ?? 1;
  for (let i = axis + 1; i < shape.length; i++) inner *= shape[i] ?? 1;
  return [outer, inner];
}

function replaced(shape: readonly number[], axis: number, size: number): number[] {
  const out = [...shape];
  out[axis] = size;
  return out;
}

/**
 * 정규화 이름 → 곱할 값. **틀린 이름은 멈춘다** — 조용히 1 을 쓰면 값이 갈린다.
 */
function normScale(norm: string | null | undefined, n: number,
                   inverse: boolean): number {
  if (norm === undefined || norm === null || norm === "backward") {
    return inverse ? 1 / n : 1;
  }
  if (norm === "forward") return inverse ? 1 : 1 / n;
  if (norm === "ortho") return 1 / Math.sqrt(n);
  throw new RuntimeError(`Invalid normalization mode: "${norm}"`);
}

/** 한 판을 돌리고 모양을 되붙인다. 네 이름이 전부 이 자리를 지난다. */
function transform(
  input: Tensor, plan: Omit<DftPlan, "outer" | "inner">, axis: number,
  gradName: string, back: (g: Tensor) => Tensor,
): Tensor {
  const [outer, inner] = split(input.shape, axis);
  const flat = run(input, { ...plan, outer, inner });
  const shape = replaced(input.shape, axis, plan.nOut);
  return makeNode(flat.raw, shape, [input], (g) => [back(g)], gradName,
                  plan.realOut ? "float32" : "complex64");
}

export function fft(input: Tensor, n?: number | null, dim = -1,
                    norm?: string | null): Tensor {
  const axis = axisOf(dim, input.shape.length);
  const have = input.shape[axis] ?? 0;
  const len = n === undefined || n === null ? have : n;
  const scale = normScale(norm, len, false);
  const wasComplex = input.isComplex();
  return transform(input, {
    n: len, nIn: Math.min(have, len), nOut: len, sign: -1, scale,
    inComplex: wasComplex, hermitian: false, realOut: false,
  }, axis, "FftC2CBackward0", (g) => {
    // **정규화 없는 역변환**에 순방향의 배율이 그대로 곱해진다 — 선형이라.
    const [outer, inner] = split(g.shape, axis);
    const wide = run(g, {
      n: len, nIn: len, nOut: len, sign: +1, scale,
      inComplex: true, hermitian: false, realOut: !wasComplex, outer, inner,
    });
    return trim(new Tensor(wide.raw, replaced(g.shape, axis, len),
                           { dtype: wasComplex ? "complex64" : "float32" }),
                axis, have);
  });
}

export function ifft(input: Tensor, n?: number | null, dim = -1,
                     norm?: string | null): Tensor {
  const axis = axisOf(dim, input.shape.length);
  const have = input.shape[axis] ?? 0;
  const len = n === undefined || n === null ? have : n;
  const scale = normScale(norm, len, true);
  const wasComplex = input.isComplex();
  return transform(input, {
    n: len, nIn: Math.min(have, len), nOut: len, sign: +1, scale,
    inComplex: wasComplex, hermitian: false, realOut: false,
  }, axis, "FftC2CBackward0", (g) => {
    const [outer, inner] = split(g.shape, axis);
    const wide = run(g, {
      n: len, nIn: len, nOut: len, sign: -1, scale,
      inComplex: true, hermitian: false, realOut: !wasComplex, outer, inner,
    });
    return trim(new Tensor(wide.raw, replaced(g.shape, axis, len),
                           { dtype: wasComplex ? "complex64" : "float32" }),
                axis, have);
  });
}

export function rfft(input: Tensor, n?: number | null, dim = -1,
                     norm?: string | null): Tensor {
  if (input.isComplex()) {
    throw new RuntimeError("rfft expects a real input tensor, but got complex");
  }
  const axis = axisOf(dim, input.shape.length);
  const have = input.shape[axis] ?? 0;
  const len = n === undefined || n === null ? have : n;
  const scale = normScale(norm, len, false);
  const bins = Math.floor(len / 2) + 1;
  return transform(input, {
    n: len, nIn: Math.min(have, len), nOut: bins, sign: -1, scale,
    inComplex: false, hermitian: false, realOut: false,
  }, axis, "FftR2CBackward0", (g) => {
    // **켤레 짝을 안 더한다.** 저장 안 된 반쪽은 애초에 손실에 안 들어갔다.
    const [outer, inner] = split(g.shape, axis);
    const wide = run(g, {
      n: len, nIn: bins, nOut: len, sign: +1, scale,
      inComplex: true, hermitian: false, realOut: true, outer, inner,
    });
    return trim(new Tensor(wide.raw, replaced(g.shape, axis, len)), axis, have);
  });
}

export function irfft(input: Tensor, n?: number | null, dim = -1,
                      norm?: string | null): Tensor {
  const axis = axisOf(dim, input.shape.length);
  const have = input.shape[axis] ?? 0;
  const len = n === undefined || n === null ? 2 * (have - 1) : n;
  const scale = normScale(norm, len, true);
  // **`n` 을 주면 앞쪽 `n//2+1` 칸만 쓴다**(실측). 든 칸을 다 쓰면 n 이 작을 때
  // 없는 주파수까지 되살려서, 모양은 맞고 값만 틀린다.
  const used = Math.min(have, Math.floor(len / 2) + 1);
  return transform(input, {
    n: len, nIn: used, nOut: len, sign: +1, scale,
    inComplex: true, hermitian: true, realOut: true,
  }, axis, "FftC2RBackward0", (g) => {
    // **쓴 칸만큼만 낸다.** `nOut = used` 라, 안 쓴 칸에는 기울기가 애초에 안 생긴다 —
    // 그 자리를 나중에 0 으로 지우는 대신 만들지 않는 편이 짧고 틀릴 자리가 없다.
    const [outer, inner] = split(g.shape, axis);
    const wide = run(g, {
      n: len, nIn: len, nOut: used, sign: -1, scale,
      inComplex: false, hermitian: false, realOut: false, outer, inner,
    });
    let full = new Tensor(wide.raw, replaced(g.shape, axis, used),
                          { dtype: "complex64" });
    // **가장자리는 한 번, 가운데는 두 번.** 되살린 짝이 같은 칸에서 왔으므로 그
    // 칸에 기울기가 두 번 도착한다. `k=0` 과 짝수 n 의 `k=n/2` 만 자기 켤레다.
    const weight = new Float32Array(used).fill(2);
    weight[0] = 1;
    if (len % 2 === 0 && used > len / 2) weight[len / 2] = 1;
    const line = new Array<number>(full.shape.length).fill(1);
    line[axis] = used;
    full = full.mul(Tensor.from(weight, line));
    if (used === have) return full;
    // 안 쓴 칸은 0 이다. 모양은 입력과 같아야 하므로 뒤에 채운다.
    return overReal(full, (r) => padAxis(r, axis, have - used));
  });
}

/**
 * 축 하나를 앞에서부터 `keep` 칸으로 자른다. 이미 같으면 그대로.
 *
 * **복소수도 지나야 한다** — 역방향이 길이를 늘려 물은 자리를 도로 줄일 때 여기를
 * 쓰는데, 그때 손에 든 것이 복소수 기울기다. `narrow` 를 그냥 부르면 복소수 문에서
 * 멈추고, 그 문구는 "이 연산은 아직" 이라 원인이 fft 안쪽을 안 가리킨다.
 */
function trim(t: Tensor, axis: number, keep: number): Tensor {
  if ((t.shape[axis] ?? 0) === keep) return t;
  return overReal(t, (r) => r.narrow(axis, 0, keep));
}

/** 축 하나의 뒤에 0 을 `count` 칸 붙인다. `padND` 는 마지막 축부터 세므로 짝을 채운다. */
function padAxis(t: Tensor, axis: number, count: number): Tensor {
  const pairs: number[] = [];
  for (let i = t.shape.length - 1; i > axis; i--) pairs.push(0, 0);
  pairs.push(0, count);
  return t.padND(pairs);
}

export function fftfreq(n: number, d = 1.0): Tensor {
  const out = new Float32Array(n);
  const half = Math.floor((n - 1) / 2) + 1;
  for (let i = 0; i < half; i++) out[i] = i / (n * d);
  for (let i = half; i < n; i++) out[i] = (i - n) / (n * d);
  return Tensor.from(out, [n]);
}

export function rfftfreq(n: number, d = 1.0): Tensor {
  const bins = Math.floor(n / 2) + 1;
  const out = new Float32Array(bins);
  for (let i = 0; i < bins; i++) out[i] = i / (n * d);
  return Tensor.from(out, [bins]);
}

/**
 * 복소수 텐서에 **칸을 옮기는 연산**을 걸어 준다.
 *
 * `viewAsReal` 이 마지막에 크기 2 축을 붙이고 그것이 늘 안쪽에 남으므로, 실수
 * 텐서로 보고 축 번호만 한 칸 밀어 두면 그대로 통한다. 옮기고 나서 `viewAsComplex`
 * 로 되돌린다 — **둘 다 뷰라 버퍼 사본이 안 는다.**
 *
 * 인터리브 저장을 아는 자리이므로 여기 둔다. 복소수를 아는 옮기기 커널을 따로
 * 쓰는 대신, 이미 있는 실수 커널을 빌린다.
 */
function overReal(z: Tensor, fn: (t: Tensor, shift: number) => Tensor): Tensor {
  if (!z.isComplex()) return fn(z, 0);
  return fn(z.viewAsReal(), 1).viewAsComplex();
}

function rollBy(input: Tensor, dim: number | readonly number[] | null | undefined,
                by: (n: number) => number): Tensor {
  return overReal(input, (t, shift) => {
    const rank = t.shape.length - shift;
    const axes = dim === null || dim === undefined
      ? Array.from({ length: rank }, (_, i) => i)
      : (Array.isArray(dim) ? dim.map((d) => axisOf(d, rank))
                            : [axisOf(dim as number, rank)]);
    let out = t;
    for (const a of axes) out = out.roll(by(t.shape[a] ?? 0), a);
    return out;
  });
}

/** 0 주파수를 가운데로. **`n//2` 만큼 민다**(실측 — 홀수에서도 그렇다). */
export function fftshift(input: Tensor,
                         dim?: number | readonly number[] | null): Tensor {
  return rollBy(input, dim, (n) => Math.floor(n / 2));
}

/** 되돌리기. **홀수에서 `n//2` 로 되돌리면 안 맞는다** — 반대 방향으로 같은 만큼. */
export function ifftshift(input: Tensor,
                          dim?: number | readonly number[] | null): Tensor {
  return rollBy(input, dim, (n) => -Math.floor(n / 2));
}

// ── 짧은 시간 변환 ────────────────────────────────────────────────────────
//
// **새 커널이 아니라 조립이다.** 자르고 · 창을 곱하고 · `rfft`. 셋 다 이미
// 미분되는 이름이라 **기울기가 저절로 맞는다.**

/**
 * 창을 **`nFft` 길이로** 맞춘다. `winLength` 는 받기만 하고 맞추는 길이는 `nFft` 다.
 *
 * 한동안 `winLength` 로 맞추고 있었다 — `win_length=6, n_fft=8` 에서 창이 6 칸으로
 * 남아 곱셈이 모양에서 멈췄다. **시끄럽게 멈춰서 다행인 자리**다. 두 수가 우연히
 * 같은 케이스만 있었으면 조용히 지나갔다.
 */
function windowOf(window: Tensor | null | undefined, nFft: number,
                  winLength?: number | null): Tensor {
  if (winLength !== null && winLength !== undefined && winLength > nFft) {
    throw new RuntimeError(
      "window length should be less than or equal to n_fft");
  }
  if (window === null || window === undefined) return Tensor.ones([nFft]);
  const have = window.shape[window.shape.length - 1] ?? 0;
  if (have === nFft) return window;
  if (have > nFft) {
    throw new RuntimeError(
      "window length should be less than or equal to n_fft");
  }
  // **가운데에 놓는다**(실측). 왼쪽 정렬이면 값이 갈린다.
  const left = Math.floor((nFft - have) / 2);
  return window.padND([left, nFft - have - left]);
}

export interface StftOptions {
  hopLength?: number | null;
  winLength?: number | null;
  window?: Tensor | null;
  center?: boolean;
  padMode?: "constant" | "reflect" | "replicate" | "circular";
  normalized?: boolean;
  onesided?: boolean | null;
  returnComplex?: boolean | null;
  length?: number | null;
}

export function stft(input: Tensor, nFft: number, options: StftOptions = {}): Tensor {
  const {
    hopLength = null, winLength = null, window = null, center = true,
    padMode = "reflect", normalized = false, onesided = null,
    returnComplex = null,
  } = options;
  // **`returnComplex` 를 안 주면 거절한다**(실측). 실수 `(…, 2)` 로 내는 옛 길은
  // torch 에서 폐기 예정이라, 기본값을 정해 주면 곧 사라질 모양을 가르치게 된다.
  if (returnComplex === null && !input.isComplex()) {
    throw new RuntimeError(
      "stft requires the return_complex parameter be given for real inputs");
  }
  if (returnComplex === false) {
    throw new RuntimeError("stft with return_complex=False is deprecated");
  }
  const hop = hopLength ?? Math.floor(nFft / 4);
  const half = onesided ?? !input.isComplex();

  let x = input;
  if (center) {
    const pad = Math.floor(nFft / 2);
    const flat = x.shape.length === 1;
    if (flat) x = x.reshape([1, -1]);
    x = x.padND([pad, pad], padMode);
    if (flat) x = x.reshape([-1]);
  }
  const length = x.shape[x.shape.length - 1] ?? 0;
  if (length < nFft) {
    throw new RuntimeError("Expected size of signal to be at least n_fft");
  }
  const count = 1 + Math.floor((length - nFft) / hop);
  const parts: Tensor[] = [];
  for (let k = 0; k < count; k++) parts.push(x.narrow(-1, k * hop, nFft));
  let frames = Tensor.stack(parts, -2)
    .mul(windowOf(window, nFft, winLength));
  let spec = half ? rfft(frames, null, -1) : fft(frames, null, -1);
  if (normalized) spec = spec.mul(Tensor.full([], 1 / Math.sqrt(nFft)));
  // `(…, 틀, 칸)` → `(…, 칸, 틀)`. torch 가 칸을 앞에 둔다.
  return swapLastTwo(spec);
}

/** 마지막 두 축을 바꾼다. 복소수면 실수 짝으로 보고 축을 한 칸 밀어서. */
function swapLastTwo(t: Tensor): Tensor {
  return overReal(t, (r, shift) => {
    const rank = r.shape.length;
    return r.movedim(rank - 2 - shift, rank - 1 - shift);
  });
}

export function istft(input: Tensor, nFft: number,
                      options: StftOptions = {}): Tensor {
  const {
    hopLength = null, winLength = null, window = null, center = true,
    normalized = false, onesided = null, length = null,
  } = options;
  const hop = hopLength ?? Math.floor(nFft / 4);
  const bins = input.shape[input.shape.length - 2] ?? 0;
  const half = onesided ?? (bins === Math.floor(nFft / 2) + 1);
  const count = input.shape[input.shape.length - 1] ?? 0;
  const spec = swapLastTwo(input);                       // (…, 틀, 칸)
  let frames = half ? irfft(spec, nFft, -1) : fft(spec, null, -1);
  if (normalized) frames = frames.mul(Tensor.full([], Math.sqrt(nFft)));
  const win = windowOf(window, nFft, winLength);
  frames = frames.mul(win);

  const total = nFft + hop * (count - 1);
  // **겹쳐 더하기.** 틀마다 자리를 맞춰 0 으로 두르고 전부 더한다 — 흩뿌리는
  // 커널 없이 되고, 역방향이 그대로 따라온다.
  let out: Tensor | null = null;
  for (let k = 0; k < count; k++) {
    const piece = frames.select(-2, k)
      .padND([k * hop, total - nFft - k * hop]);
    out = out === null ? piece : out.add(piece);
  }
  if (out === null) throw new RuntimeError("istft: 틀이 하나도 없다");
  // **창의 제곱 겹침으로 나눈다.** 그 나눗셈이 없으면 겹친 자리가 창 무게만큼
  // 부풀어 오른다. 0 에 가까운 자리는 1 로 두어 나눗셈을 피한다.
  const envelope = win.mul(win);
  let cover: Tensor | null = null;
  for (let k = 0; k < count; k++) {
    const piece = envelope.padND([k * hop, total - nFft - k * hop]);
    cover = cover === null ? piece : cover.add(piece);
  }
  const safe = cover === null ? Tensor.ones([total])
    : cover.where(cover.binary("gt", Tensor.full([], 1e-11)),
                  Tensor.ones([total]));
  out = out.div(safe);
  if (center) {
    out = out.narrow(-1, Math.floor(nFft / 2),
                     total - 2 * Math.floor(nFft / 2));
  }
  if (length !== null) out = out.narrow(-1, 0, length);
  return out;
}

// ── 여러 축 · 에르미트 — **전부 위 넷의 조립이다** ───────────────────────────
//
// 새 커널이 없다. `fft2` 는 축을 하나씩 도는 것이고(실측: torch 의 `fft2` 와 정확히
// 같다) 에르미트 갈래는 켤레와 배율로 풀린다. 그래서 기울기도 따로 안 쓴다 — 테이프가
// 그대로 이어진다. 여기 역방향을 손으로 적으면 위 넷과 두 벌이 되고, 두 벌은 갈린다.
//
// **파이썬이 대신 채우지 않게 여기 둔다.** 결속이 조립하면 골든은 초록이 되는데
// borch.ts 를 쓰는 쪽에는 그 이름이 여전히 없다 — 이 저장소가 일곱 번 겪은 자리다.

function axesAndSizes(
  t: Tensor, s: readonly (number | null)[] | null | undefined,
  dim: readonly number[] | number | null | undefined,
): [number[], number[]] {
  const rank = t.shape.length;
  let dims: readonly number[];
  if (dim === null || dim === undefined) {
    dims = s === null || s === undefined
      ? Array.from({ length: rank }, (_, i) => i)
      : Array.from({ length: s.length }, (_, i) => rank - s.length + i);
  } else {
    dims = typeof dim === "number" ? [dim] : dim;
  }
  const axes = dims.map((d) => axisOf(d, rank));
  const sizes = axes.map((a, i) => {
    const want = s === null || s === undefined ? null : s[i];
    return want === null || want === undefined ? (t.shape[a] ?? 0) : want;
  });
  return [axes, sizes];
}

/** 에르미트 갈래는 정·역이 뒤바뀐다 — 정규화 이름도 같이 뒤집는다. */
function flipNorm(norm?: string | null): string | null | undefined {
  if (norm === "forward") return "backward";
  if (norm === "backward" || norm === null || norm === undefined) return "forward";
  return norm;
}

export function fftn(input: Tensor, s?: (number | null)[] | null,
                     dim?: number[] | number | null, norm?: string | null): Tensor {
  const [axes, sizes] = axesAndSizes(input, s, dim);
  let out = input;
  axes.forEach((a, i) => { out = fft(out, sizes[i], a, norm); });
  return out;
}

export function ifftn(input: Tensor, s?: (number | null)[] | null,
                      dim?: number[] | number | null, norm?: string | null): Tensor {
  const [axes, sizes] = axesAndSizes(input, s, dim);
  let out = input;
  axes.forEach((a, i) => { out = ifft(out, sizes[i], a, norm); });
  return out;
}

export function fft2(input: Tensor, s?: (number | null)[] | null,
                     dim: number[] = [-2, -1], norm?: string | null): Tensor {
  return fftn(input, s, dim, norm);
}

export function ifft2(input: Tensor, s?: (number | null)[] | null,
                      dim: number[] = [-2, -1], norm?: string | null): Tensor {
  return ifftn(input, s, dim, norm);
}

/** 실수 입력. **마지막 축만 `rfft` 이고 나머지는 `fft`** 다 — 차례가 답을 정한다. */
export function rfftn(input: Tensor, s?: (number | null)[] | null,
                      dim?: number[] | number | null, norm?: string | null): Tensor {
  const [axes, sizes] = axesAndSizes(input, s, dim);
  const last = axes.length - 1;
  let out = rfft(input, sizes[last], axes[last], norm);
  for (let i = 0; i < last; i += 1) out = fft(out, sizes[i], axes[i], norm);
  return out;
}

/** `rfftn` 의 역. **`ifft` 를 먼저 돌고 마지막에 `irfft`** 다. */
export function irfftn(input: Tensor, s?: (number | null)[] | null,
                       dim?: number[] | number | null, norm?: string | null): Tensor {
  const [axes, sizes] = axesAndSizes(input, s, dim);
  const last = axes.length - 1;
  if (s === null || s === undefined) {
    sizes[last] = 2 * ((input.shape[axes[last] as number] ?? 1) - 1);
  }
  let out = input;
  for (let i = 0; i < last; i += 1) out = ifft(out, sizes[i], axes[i], norm);
  return irfft(out, sizes[last], axes[last], norm);
}

export function rfft2(input: Tensor, s?: (number | null)[] | null,
                      dim: number[] = [-2, -1], norm?: string | null): Tensor {
  return rfftn(input, s, dim, norm);
}

export function irfft2(input: Tensor, s?: (number | null)[] | null,
                       dim: number[] = [-2, -1], norm?: string | null): Tensor {
  return irfftn(input, s, dim, norm);
}

/** 에르미트 대칭인 복소수 → **실수.** `irfft` 의 켤레 관계다(실측). */
export function hfft(input: Tensor, n?: number | null, dim = -1,
                     norm?: string | null): Tensor {
  const axis = axisOf(dim, input.shape.length);
  const len = n === undefined || n === null
    ? 2 * ((input.shape[axis] ?? 1) - 1) : n;
  return irfft(input.conj(), len, axis, flipNorm(norm));
}

/** 실수 → **에르미트 대칭인 복소수.** `rfft` 의 켤레다. */
export function ihfft(input: Tensor, n?: number | null, dim = -1,
                      norm?: string | null): Tensor {
  const axis = axisOf(dim, input.shape.length);
  return rfft(input, n, axis, flipNorm(norm)).conj();
}

/**
 * 마지막 축이 `hfft` 이고 **앞 축은 `fft`** 다.
 *
 * `rfftn` 의 거울이니 `ifft` 일 것 같은데 torch 는 `fft` 다(실측 — 후보를 둘 다
 * 만들어 대 봤다). **모양은 양쪽 다 맞아서** 값을 안 재면 안 드러난다.
 */
export function hfftn(input: Tensor, s?: (number | null)[] | null,
                      dim?: number[] | number | null, norm?: string | null): Tensor {
  const [axes, sizes] = axesAndSizes(input, s, dim);
  const last = axes.length - 1;
  if (s === null || s === undefined) {
    sizes[last] = 2 * ((input.shape[axes[last] as number] ?? 1) - 1);
  }
  let out = input;
  for (let i = 0; i < last; i += 1) out = fft(out, sizes[i], axes[i], norm);
  return hfft(out, sizes[last], axes[last], norm);
}

export function ihfftn(input: Tensor, s?: (number | null)[] | null,
                       dim?: number[] | number | null, norm?: string | null): Tensor {
  const [axes, sizes] = axesAndSizes(input, s, dim);
  const last = axes.length - 1;
  let out = ihfft(input, sizes[last], axes[last], norm);
  for (let i = 0; i < last; i += 1) out = ifft(out, sizes[i], axes[i], norm);
  return out;
}

export function hfft2(input: Tensor, s?: (number | null)[] | null,
                      dim: number[] = [-2, -1], norm?: string | null): Tensor {
  return hfftn(input, s, dim, norm);
}

export function ihfft2(input: Tensor, s?: (number | null)[] | null,
                       dim: number[] = [-2, -1], norm?: string | null): Tensor {
  return ihfftn(input, s, dim, norm);
}
