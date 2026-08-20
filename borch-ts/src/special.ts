/**
 * Special functions — the incomplete gamma and the polygammas.
 *
 * **The reason these kernels live apart** is that `n` is a shader constant.
 * `polygamma(n, x)` bakes a different shader per `n`, and the unary table
 * (`UNARY` in `kernels.ts`) hangs one shader off one name, so it does not
 * fit there.
 *
 * The formulas are carried over from the core (numpy) unchanged — two
 * copies means a day when only one gets fixed. The core side was confirmed
 * against torch for both values and gradients first; this is that carried
 * across.
 */

import { NotImplementedError, RuntimeError } from "./errors.js";
import { device, makeNode, Tensor } from "./tensor.js";

/**
 * `lgamma` — 두 특수 함수가 공통으로 쓴다. 램초스 근사다.
 *
 * **반사식을 재귀로 쓰면 안 된다.** WGSL 은 함수의 재귀를 금지하고, 셰이더를 굽는
 * 자리에서 `cyclic dependency found` 로 멈춘다 — 값이 틀린 것이 아니라 커널이
 * 아예 안 만들어진다. 그래서 근사 몸통(`lgammaCore_`)과 반사(`lgamma_`)를
 * 두 함수로 갈랐다.
 */
const LGAMMA = `
fn lgammaCore_(x: f32) -> f32 {
  let z = x - 1.0;
  var a = 0.99999999999980993;
  let g = array<f32, 8>(
    676.5203681218851, -1259.1392167224028, 771.32342877765313,
    -176.61502916214059, 12.507343278686905, -0.13857109526572012,
    9.9843695780195716e-6, 1.5056327351493116e-7);
  for (var i = 0u; i < 8u; i = i + 1u) {
    a = a + g[i] / (z + f32(i) + 1.0);
  }
  let t = z + 7.5;
  return 0.9189385332046727 + (z + 0.5) * log(t) - t + log(a);
}
fn lgamma_(x: f32) -> f32 {
  if (x < 0.5) {
    // 반사식. 작은 쪽은 근사가 안 맞아서 큰 쪽으로 옮겨 센다.
    return log(3.14159265358979 / abs(sin(3.14159265358979 * x))) - lgammaCore_(1.0 - x);
  }
  return lgammaCore_(x);
}`;

/**
 * 되풀이 + 점근으로 `ψ^(n)`. `n` 이 상수라 계승을 호스트에서 미리 접어 넣는다.
 *
 *     ψ^(n)(x) = ψ^(n)(x+1) + (−1)^(n+1) n! / x^(n+1)
 *     ψ^(n)(x) ≈ (−1)^(n+1) [ (n−1)!/xⁿ + n!/(2x^(n+1)) + Σ B_2k … ]
 */
function polygammaSource(n: number, count: number): string {
  const fact = (k: number): number => {
    let out = 1;
    for (let i = 2; i <= k; i++) out *= i;
    return out;
  };
  const sign = (n + 1) % 2 === 0 ? 1 : -1;
  const bern = [1 / 6, -1 / 30, 1 / 42, -1 / 30];
  const tail = bern.map((b, i) => {
    const k = i + 1;
    return `    s = s + ${(b * fact(2 * k + n - 1) / fact(2 * k)).toExponential(12)}`
      + ` / pow(y, ${2 * k + n}.0);`;
  }).join("\n");
  const stride = Math.min(Math.max(1, Math.ceil(count / 64)), 65535) * 64;
  return `
@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read_write> Out: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let gid = g.y * ${stride}u + g.x;
  if (gid >= ${count}u) { return; }
  var y = A[gid];
  var acc = 0.0;
  // **큰 쪽으로 민다.** 점근식은 작은 x 에서 안 맞는다.
  for (var i = 0u; i < 64u; i = i + 1u) {
    if (y >= 20.0) { break; }
    acc = acc + ${sign}.0 * ${fact(n).toExponential(12)} / pow(y, ${n + 1}.0);
    y = y + 1.0;
  }
  var s = ${fact(n - 1).toExponential(12)} / pow(y, ${n}.0)
    + ${fact(n).toExponential(12)} / (2.0 * pow(y, ${n + 1}.0));
${tail}
  Out[gid] = acc + ${sign}.0 * s;
}`;
}

/**
 * 정규화된 하부 불완전 감마 `P(a, x)`.
 *
 * **한 식으로 못 덮는다** — `x < a+1` 은 급수, 그 밖은 연분수다. 반대로 쓰면 항이
 * 서로 지워 자릿수를 잃는다.
 */
function igammaSource(count: number): string {
  const stride = Math.min(Math.max(1, Math.ceil(count / 64)), 65535) * 64;
  return `
${LGAMMA}
@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read_write> Out: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let gid = g.y * ${stride}u + g.x;
  if (gid >= ${count}u) { return; }
  let a = A[gid];
  let x = X[gid];
  if (x <= 0.0) { Out[gid] = 0.0; return; }
  let head = exp(-x + a * log(x) - lgamma_(a));
  if (x < a + 1.0) {
    var ap = a;
    var term = 1.0 / a;
    var total = term;
    for (var i = 0u; i < 200u; i = i + 1u) {
      ap = ap + 1.0;
      term = term * x / ap;
      total = total + term;
      if (abs(term) <= abs(total) * 1e-8) { break; }
    }
    Out[gid] = total * head;
    return;
  }
  // 연분수 — 상부 Q 를 구하고 1 에서 뺀다.
  let tiny = 1e-30;
  var b = x + 1.0 - a;
  var c = 1.0 / tiny;
  var d = 1.0 / b;
  var h = d;
  for (var i = 1u; i < 200u; i = i + 1u) {
    let an = -f32(i) * (f32(i) - a);
    b = b + 2.0;
    d = an * d + b;
    if (abs(d) < tiny) { d = tiny; }
    c = b + an / c;
    if (abs(c) < tiny) { c = tiny; }
    d = 1.0 / d;
    let delta = d * c;
    h = h * delta;
    if (abs(delta - 1.0) <= 1e-8) { break; }
  }
  Out[gid] = 1.0 - head * h;
}`;
}

/** `dP/dx = x^(a−1)·e^(−x) / Γ(a)`. 기울기가 이쪽에만 있다. */
function igammaSlopeSource(count: number): string {
  const stride = Math.min(Math.max(1, Math.ceil(count / 64)), 65535) * 64;
  return `
${LGAMMA}
@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read_write> Out: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let gid = g.y * ${stride}u + g.x;
  if (gid >= ${count}u) { return; }
  let a = A[gid];
  let x = X[gid];
  if (x <= 0.0) { Out[gid] = 0.0; return; }
  Out[gid] = exp((a - 1.0) * log(x) - x - lgamma_(a));
}`;
}

function sameShape(a: Tensor, b: Tensor): void {
  if (a.shape.length !== b.shape.length
    || a.shape.some((d, i) => d !== b.shape[i])) {
    throw new RuntimeError(
      `igamma requires matching shapes: [${a.shape}] vs [${b.shape}]`);
  }
}

export function igamma(a: Tensor, x: Tensor): Tensor {
  sameShape(a, x);
  const dev = device();
  const n = x.size;
  const out = dev.alloc(n);
  dev.run1d(dev.pipeline(`igamma:${n}`, () => igammaSource(n)),
            [a.buffer, x.buffer, out], n);
  return makeNode(out, x.shape, [a, x], (g) => {
    // **첫 인자로는 안 미분한다** — torch 도 닫힌 꼴이 없어 거절한다(실측).
    // 종류는 `NotImplementedError` 다. torch 가 그 종류로 내고, "아직 없다" 와
    // "부른 쪽이 틀렸다" 는 잡는 코드가 다르다.
    if (a.requiresGrad) {
      throw new NotImplementedError(
        "the derivative for 'igamma: input' is not implemented.");
    }
    const slope = dev.alloc(n);
    dev.run1d(dev.pipeline(`igammaD:${n}`, () => igammaSlopeSource(n)),
              [a.buffer, x.buffer, slope], n);
    return [null, g.mul(new Tensor(slope, x.shape))];
  }, "IgammaBackward0");
}

/**
 * The upper branch, `Q = 1 − P`. **The two sum to exactly 1** (measured).
 */
export function igammac(a: Tensor, x: Tensor): Tensor {
  return Tensor.ones(x.shape).sub(igamma(a, x));
}

export function polygamma(n: number, x: Tensor): Tensor {
  const k = Math.trunc(n);
  if (k < 0) {
    throw new RuntimeError("polygamma(n, x) does not support negative n.");
  }
  // `n = 0` 은 이미 단항 표에 있다 — 같은 식을 두 벌로 두지 않는다.
  if (k === 0) return x.digamma();
  const dev = device();
  const count = x.size;
  const out = dev.alloc(count);
  dev.run1d(dev.pipeline(`polygamma:${k}:${count}`,
                         () => polygammaSource(k, count)),
            [x.buffer, out], count);
  // 미분은 다음 차수다(실측: `polygamma(1)` 의 기울기가 `polygamma(2)`).
  return makeNode(out, x.shape, [x],
                  (g) => [g.mul(polygamma(k + 1, x).detach())],
                  "PolygammaBackward0");
}
