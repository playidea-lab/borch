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
 * `lgamma` — shared by two special functions. The Lanczos approximation.
 *
 * **The reflection formula must not be written recursively.** WGSL forbids function
 * recursion and stops while baking the shader with `cyclic dependency found` — not a wrong
 * value but no kernel at all. So the approximation body (`lgammaCore_`) and the reflection
 * (`lgamma_`) are two functions.
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
    // The reflection formula. The approximation does not hold on the small side, so it
    // is moved to the large side and computed there.
    return log(3.14159265358979 / abs(sin(3.14159265358979 * x))) - lgammaCore_(1.0 - x);
  }
  return lgammaCore_(x);
}`;

/**
 * `ψ^(n)` by recurrence plus asymptotics. `n` is a constant, so the factorials are folded
 * in on the host in advance.
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
  // **Pushed to the large side.** The asymptotic form does not hold at small x.
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
 * The regularised lower incomplete gamma `P(a, x)`.
 *
 * **One expression cannot cover it** — `x < a+1` is the series and the rest is the
 * continued fraction. The other way round, the terms cancel and digits are lost.
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
  // The continued fraction — it computes the upper Q and subtracts from 1.
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

/** `dP/dx = x^(a−1)·e^(−x) / Γ(a)`. The gradient exists on this side only. */
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
    // **It does not differentiate with respect to the first argument** — torch has no
    // closed form for it either and refuses (measured). The kind is `NotImplementedError`.
    // torch raises that kind, and "it does not exist yet" and "the caller was wrong" are
    // caught by different code.
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
  // `n = 0` is already in the unary table — the same expression is not kept twice.
  if (k === 0) return x.digamma();
  const dev = device();
  const count = x.size;
  const out = dev.alloc(count);
  dev.run1d(dev.pipeline(`polygamma:${k}:${count}`,
                         () => polygammaSource(k, count)),
            [x.buffer, out], count);
  // The derivative is the next order (measured: `polygamma(1)`'s gradient is
  // `polygamma(2)`).
  return makeNode(out, x.shape, [x],
                  (g) => [g.mul(polygamma(k + 1, x).detach())],
                  "PolygammaBackward0");
}

