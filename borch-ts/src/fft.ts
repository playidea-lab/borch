/**
 * `torch.fft` — the discrete Fourier transform, and `stft`/`istft` on top
 * of it.
 *
 * **There is one kernel.** Forward, inverse, half transforms and the
 * inverses of those three all call the same shader with a different sign
 * and scale. Split apart they become six copies, and the day comes when one
 * of them is fixed differently — that is the kind of ground this repository
 * has repeatedly lost.
 *
 * ## It is O(n²), for all that it is called `fft`
 *
 * It runs the DFT directly. Cooley–Tukey is only fast at powers of two, and
 * other lengths need Bluestein on the side. **The values are the same
 * either way**, and at this project's ceiling — textbook-sized signals —
 * the difference does not show. If a day comes when it has to be fast,
 * change it then, but **do not write down speed that is not there today.**
 *
 * ## Axes are not moved
 *
 * Wherever `dim` points, no data is moved. Three numbers — `(outer, axis,
 * inner)` — locate the element and the shader reads it directly. Moving and
 * moving back costs two more buffers, and moving complex data is an
 * operation that does not exist yet, so it would have had to be built for
 * this.
 *
 * ## Gradients
 *
 * The transform is **linear**, so the backward falls straight out of the
 * convention. The backward of a holomorphic function is `conj(f')·g`, which
 * gives `grad_x[j] = Σ_k e^{+2πijk/n}·g[k]` — **an unnormalised inverse
 * transform.** The hard part is not the value but **which half gets
 * counted**:
 *
 * * `rfft`'s backward receives gradient only on the stored half — adding
 *   the conjugate pair doubles it.
 * * `irfft`'s backward counts **the edges once and the middle twice**,
 *   because the conjugate pair it restored came out of the same stored
 *   slot.
 *
 * Either can be wrong **while the forward values stay perfectly fine**. The
 * core (numpy) went through the same derivation first, and the golden cases
 * ask about the two separately.
 */

import { RuntimeError } from "./errors.js";
import { device, makeNode, Tensor } from "./tensor.js";

/** `n` is the transform length; `nIn` and `nOut` are the cells actually held and
 *  produced. */
interface DftPlan {
  readonly n: number;
  readonly nIn: number;
  readonly nOut: number;
  /** −1 is the forward transform and +1 the inverse. */
  readonly sign: number;
  readonly scale: number;
  /** Whether the input is interleaved complex. */
  readonly inComplex: boolean;
  /** Whether to restore the stored half by conjugation (`irfft`). */
  readonly hermitian: boolean;
  /** Whether to keep only the result's real part (`irfft`). */
  readonly realOut: boolean;
  /** The cell counts before and after the axis. Held so the axis need not be moved. */
  readonly outer: number;
  readonly inner: number;
}

function shader(p: DftPlan): string {
  const threads = p.outer * p.nOut * p.inner;
  const stride = Math.min(Math.max(1, Math.ceil(threads / 64)), 65535) * 64;
  const step = p.inComplex ? 2 : 1;
  // **Picking one input cell.** A cell that is not there is 0 (where the length was
  // extended in the request), and in the Hermitian case the opposite partner is restored
  // by conjugation.
  // **The imaginary part sits immediately beside the real one.** Complex storage
  // interleaves `(re, im)` per cell — the writing side says so with `Out[o*2]` and
  // `Out[o*2+1]`.
  //
  // This said `A[at + inner]` for a long time. **On the last axis `inner` is 1, so it
  // happens to be the same.** Every 1-D case passed, and transforming a complex input
  // along **any axis but the last** read the imaginary part from the wrong cell. A real
  // input never passes this line at all, so both axes were right for it — it took three
  // conditions together to show.
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
  // Where this row starts in the input. Complex holds two per cell, so the inner stride
  // doubles.
  // (The shader text lives inside a template literal — a backtick here closes the
  //  string.)
  let base = outerIdx * ${p.nIn * p.inner * step}u + innerIdx * ${step}u;
  var re = 0.0;
  var im = 0.0;
  for (var j = 0u; j < ${p.n}u; j = j + 1u) {
${fetch}
    // **The twiddle factors are read from a table — no cos/sin is called in the
    // shader.**
    //
    // A version computing the angle and calling them came first, and one rectangular-window
    // stft left the golden at a relative error of 2.7e-4. That is too large to explain by
    // f32 rounding, and WGSL leaves its trigonometric accuracy to the implementation
    // (particularly on a software adapter). Building the table **in double precision** on
    // the host and uploading it removes that place entirely, and the transcendentals leave
    // the inner loop with it.
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
 * The twiddle table. It holds `cos(2πm/n)` and `sin(2πm/n)` alternately for
 * `m = 0…n−1`.
 *
 * **Built in double precision on the host.** The shader attaches the sign, so one table
 * suffices, and the same `n` reuses the same buffer — uploading a new one every step in a
 * training loop looks exactly like a leak.
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

/** One kernel call. The caller knows the shape. */
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
    throw new RuntimeError(`Dimension ${dim} is out of range for a tensor of rank ${rank}.`);
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
 * Normalisation name → the factor. **An unknown name stops** — quietly using 1 diverges
 * in the values.
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

/** Runs one plate and reattaches the shape. All four names pass through here. */
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
    // **An unnormalised inverse** takes the forward's factor as it is — it is linear.
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
    // **The conjugate partner is not added.** The unstored half never entered the loss.
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
  // **Given `n`, only the first `n//2+1` cells are used** (measured). Using every cell
  // held restores frequencies that are not there when n is small — the shape is right and
  // only the values are wrong.
  const used = Math.min(have, Math.floor(len / 2) + 1);
  return transform(input, {
    n: len, nIn: used, nOut: len, sign: +1, scale,
    inComplex: true, hermitian: true, realOut: true,
  }, axis, "FftC2RBackward0", (g) => {
    // **It produces only as many cells as were used.** With `nOut = used`, no gradient
    // ever arises at an unused cell — not making it beats zeroing it afterwards, and there
    // is nowhere to be wrong.
    const [outer, inner] = split(g.shape, axis);
    const wide = run(g, {
      n: len, nIn: len, nOut: used, sign: -1, scale,
      inComplex: false, hermitian: false, realOut: false, outer, inner,
    });
    let full = new Tensor(wide.raw, replaced(g.shape, axis, used),
                          { dtype: "complex64" });
    // **The edges once and the middle twice.** The restored partner came from the same
    // cell, so the gradient arrives at that cell twice. Only `k=0`, and `k=n/2` at even n,
    // are their own conjugates.
    const weight = new Float32Array(used).fill(2);
    weight[0] = 1;
    if (len % 2 === 0 && used > len / 2) weight[len / 2] = 1;
    const line = new Array<number>(full.shape.length).fill(1);
    line[axis] = used;
    full = full.mul(Tensor.from(weight, line));
    if (used === have) return full;
    // Unused cells are 0. The shape has to match the input, so they are filled at the
    // end.
    return overReal(full, (r) => padAxis(r, axis, have - used));
  });
}

/**
 * Cuts one axis to `keep` cells from the front. Already equal, it passes through.
 *
 * **Complex has to pass here too** — the backward uses it to shrink a request that
 * extended the length, and what it holds then is a complex gradient. Calling `narrow`
 * directly stops at the complex door, and that message says "this operation does not yet",
 * which does not point the cause anywhere inside fft.
 */
function trim(t: Tensor, axis: number, keep: number): Tensor {
  if ((t.shape[axis] ?? 0) === keep) return t;
  return overReal(t, (r) => r.narrow(axis, 0, keep));
}

/** Appends `count` zeros to one axis. `padND` counts from the last axis, so the pairs
 *  are filled accordingly. */
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
 * Applies **an operation that moves cells** to a complex tensor.
 *
 * `viewAsReal` appends an axis of size 2 at the end and it always stays innermost, so
 * seeing it as a real tensor and shifting the axis numbers by one works as it is. After
 * the move, `viewAsComplex` turns it back — **both are views, so no buffer copy is
 * added.**
 *
 * It lives here because this is the place that knows the interleaved storage. Rather than
 * a moving kernel that understands complex, it borrows the real kernel that already
 * exists.
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

/**
 * Zero frequency to the middle. **It shifts by `n//2`** (measured —
 * including at odd lengths).
 */
export function fftshift(input: Tensor,
                         dim?: number | readonly number[] | null): Tensor {
  return rollBy(input, dim, (n) => Math.floor(n / 2));
}

/**
 * The undo. **Shifting back by `n//2` does not land at odd lengths** — the
 * same amount, the other way.
 */
export function ifftshift(input: Tensor,
                          dim?: number | readonly number[] | null): Tensor {
  return rollBy(input, dim, (n) => -Math.floor(n / 2));
}

// ── The short-time transform ──────────────────────────────────────────────
//
// **An assembly rather than a new kernel.** Slice, multiply by the window, `rfft`. All
// three are already differentiable names, so **the gradient comes out right by itself.**

/**
 * Fits the window **to `nFft`.** `winLength` is only received; the length fitted to is
 * `nFft`.
 *
 * It fitted to `winLength` for a while — at `win_length=6, n_fft=8` the window stayed 6
 * cells and the multiplication stopped on the shape. **A place it is fortunate to stop
 * loudly.** With only cases where the two numbers happen to be equal, it would have gone
 * by in silence.
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
  // **It is centred** (measured). Left-aligned diverges in the values.
  const left = Math.floor((nFft - have) / 2);
  return window.padND([left, nFft - have - left]);
}

/**
 * **Two interfaces, because torch has two lists.** One stood for both, so `stft`
 * offered a `length` that belongs to `istft` and `istft` offered a `padMode` that
 * belongs to `stft` — the same shape as one shared tuple standing for the plain and
 * transposed convolutions, and it hides the same way: a caller who passes the wrong
 * one is not told, because the type accepts it.
 */
export interface StftOptions {
  hopLength?: number | null;
  winLength?: number | null;
  window?: Tensor | null;
  center?: boolean;
  padMode?: "constant" | "reflect" | "replicate" | "circular";
  normalized?: boolean;
  onesided?: boolean | null;
  returnComplex?: boolean | null;
  /**
   * **Accepted, and torch's own refusal is the whole of what it does.** torch rejects
   * it unless `center` is false, and with `center` false it answered the same at every
   * setting — so there is nothing to imitate but the refusal. The core says the same
   * at the same place, and the seat is torch's either way.
   */
  alignToWindow?: boolean | null;
}

export interface IstftOptions {
  hopLength?: number | null;
  winLength?: number | null;
  window?: Tensor | null;
  center?: boolean;
  normalized?: boolean;
  onesided?: boolean | null;
  length?: number | null;
  returnComplex?: boolean | null;
}

export function stft(input: Tensor, nFft: number, options: StftOptions = {}): Tensor {
  // `!= null` covers both empties — Python's `None` crosses as `undefined`.
  if (options.alignToWindow != null && (options.center ?? true)) {
    throw new RuntimeError(
      "stft align_to_window should only be set when center = false");
  }
  const {
    hopLength = null, winLength = null, window = null, center = true,
    padMode = "reflect", normalized = false, onesided = null,
    returnComplex = null,
  } = options;
  // **Without `returnComplex` it refuses** (measured). The old route producing a real
  // `(…, 2)` is deprecated in torch, so choosing a default would teach a shape that is
  // about to disappear.
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
  // `(…, frame, bin)` → `(…, bin, frame)`. torch puts the bins first.
  return swapLastTwo(spec);
}

/** Swaps the last two axes. For complex, seen as its real pair with the axes shifted by
 *  one. */
function swapLastTwo(t: Tensor): Tensor {
  return overReal(t, (r, shift) => {
    const rank = r.shape.length;
    return r.movedim(rank - 2 - shift, rank - 1 - shift);
  });
}

export function istft(input: Tensor, nFft: number,
                      options: IstftOptions = {}): Tensor {
  const {
    hopLength = null, winLength = null, window = null, center = true,
    normalized = false, onesided = null, length = null,
  } = options;
  const hop = hopLength ?? Math.floor(nFft / 4);
  const bins = input.shape[input.shape.length - 2] ?? 0;
  const half = onesided ?? (bins === Math.floor(nFft / 2) + 1);
  const count = input.shape[input.shape.length - 1] ?? 0;
  const spec = swapLastTwo(input);                       // (…, frame, bin)
  let frames = half ? irfft(spec, nFft, -1) : fft(spec, null, -1);
  if (normalized) frames = frames.mul(Tensor.full([], Math.sqrt(nFft)));
  const win = windowOf(window, nFft, winLength);
  frames = frames.mul(win);

  const total = nFft + hop * (count - 1);
  // **Overlap-add.** Each frame is placed by padding with zeros and they are all summed
  // — it works without a scattering kernel and the backward follows as it is.
  let out: Tensor | null = null;
  for (let k = 0; k < count; k++) {
    const piece = frames.select(-2, k)
      .padND([k * hop, total - nFft - k * hop]);
    out = out === null ? piece : out.add(piece);
  }
  if (out === null) throw new RuntimeError("istft: the input has no frames");
  // **Divided by the overlapped window squared.** Without that division the overlapping
  // positions swell by the window's weight. Positions near 0 are set to 1 to avoid the
  // division.
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

// ── Several axes and the Hermitian forms — **all assemblies of the four above** ──
//
// No new kernel. `fft2` walks the axes one at a time (measured: exactly torch's `fft2`),
// and the Hermitian branch resolves into conjugates and factors. So no gradient is written
// either — the tape carries straight through. Writing a backward by hand here would make
// two copies of the four above, and two copies diverge.
//
// **They live here so that Python does not fill them in instead.** With the binding
// assembling them the golden goes green while the name is still absent for anybody using
// borch.ts — a place this repository has met seven times.

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

/** The Hermitian branch swaps forward and inverse — the normalisation name flips with
 *  them. */
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

/**
 * Real input. **Only the last axis is `rfft`; the rest are `fft`** — the
 * order decides the answer.
 */
export function rfftn(input: Tensor, s?: (number | null)[] | null,
                      dim?: number[] | number | null, norm?: string | null): Tensor {
  const [axes, sizes] = axesAndSizes(input, s, dim);
  const last = axes.length - 1;
  let out = rfft(input, sizes[last], axes[last], norm);
  for (let i = 0; i < last; i += 1) out = fft(out, sizes[i], axes[i], norm);
  return out;
}

/**
 * The inverse of `rfftn`. **`ifft` first, `irfft` last.**
 */
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

/**
 * Hermitian-symmetric complex → **real.** The conjugate relation of `irfft`
 * (measured).
 */
export function hfft(input: Tensor, n?: number | null, dim = -1,
                     norm?: string | null): Tensor {
  const axis = axisOf(dim, input.shape.length);
  const len = n === undefined || n === null
    ? 2 * ((input.shape[axis] ?? 1) - 1) : n;
  return irfft(input.conj(), len, axis, flipNorm(norm));
}

/**
 * Real → **Hermitian-symmetric complex.** The conjugate of `rfft`.
 */
export function ihfft(input: Tensor, n?: number | null, dim = -1,
                      norm?: string | null): Tensor {
  const axis = axisOf(dim, input.shape.length);
  return rfft(input, n, axis, flipNorm(norm)).conj();
}

/**
 * The last axis is `hfft` and **the axes before it are `fft`.**
 *
 * Being the mirror of `rfftn` it looks like it should be `ifft`, but torch
 * uses `fft` (measured — both candidates were built and compared). **The
 * shape is right either way**, so it does not surface unless the values are
 * measured.
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

