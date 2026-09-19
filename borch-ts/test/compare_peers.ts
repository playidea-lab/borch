/**
 * The peers bench — the ResNet-18 (CIFAR) training step of `bench.ts`, run in the two
 * other libraries that train in a browser on WebGPU, on the same page, same machine,
 * same batch sizes, same seeded data:
 *
 * - **jax-js** (`@jax-js/jax` 0.1.25 + `@jax-js/optax` 0.1.2, ekzhang) — JavaScript, a JAX
 *   in the browser with `grad`/`jit` and a WebGPU backend. It has no BatchNorm module and
 *   no cross-entropy loss, so both are written from its primitives, as its own MNIST example
 *   writes them. Layout is NCHW (its `lax.conv` has no other), so the seeded pixels are the
 *   NCHW draw of `bench.ts` exactly.
 * - **Burn** (Rust, wgpu backend + autodiff) — compiled to wasm from a scratch crate whose
 *   source and build are recorded in `docs/BOOK.md`; loaded from
 *   `tests/browser/burn_resnet18/pkg`. Absent bundle → the row says so and the rest stands.
 *
 * **Why it exists.** The 2026-09-19 survey found these are the only other libraries that
 * train on WebGPU in a browser, and that nobody had published a training ms/step on Apple
 * silicon — so the training comparison rested on TF.js alone, a framework that stopped
 * releasing in 2024. These two rows are the missing ones.
 *
 * **What is held equal** (as `compare.ts` holds it against TF.js): architecture (stem
 * 3×3·64, four stages of two blocks, 1×1 stride-2 shortcuts where the shape changes, global
 * average pool, dense 10, convs without bias), SGD 0.05/0.9, cross-entropy on integer labels,
 * the xorshift-seeded pixels and labels, two warm-up steps then five timed, and a readback of
 * the loss every step so the clock includes the GPU finishing. **What is not**: parameter
 * initialisation (each library's own draw — a speed, not a trained result), kernel fusion
 * (jax-js's `jit` fuses what it can, as borch's eager step does not — each at its best), and
 * jax-js's reference counting, which this file follows by hand (`.ref` on every use but the
 * last) because a leaked array there would be a leak, not a slowdown.
 *
 * **Burn's autotune, ruled out.** Its first browser load tunes shaders, so its two warm-ups were
 * suspected of being too few. Re-run with eight (2026-09-19, apple/metal-3): 264.4 / 516.2 / 1017.3
 * ms against 262.8 / 512.0 / 1014.7 with two — within 1 %. The default stays two, equal to the rest.
 */

// Bare specifiers, resolved by compare_peers.html's import map to pinned jsDelivr URLs — so that
// optax and this file share one instance of jax-js (see the map's comment on why not esm.sh).
const JAX_URL = "@jax-js/jax";
const OPTAX_URL = "@jax-js/optax";
// Absolute from the served root — the compiled module lives at /borch-ts/dist/test/, and a
// relative path from there walked into /borch-ts/tests/… (measured, a 404).
const BURN_PKG = "/tests/browser/burn_resnet18/pkg/burn_resnet18.js";

export interface PeerStep { batch: number; msPerStep: number; lastLoss: number }

/** The same numbers `bench.ts` draws: xorshift32 from 12345, pixels in [-1, 1), labels 0..9 — NCHW. */
function seededNCHW(batch: number): { pixels: Float32Array; labels: Int32Array } {
  const rng = { s: 12345 };
  const next = (): number => {
    let x = rng.s; x ^= x << 13; x >>>= 0; x ^= x >> 17; x ^= x << 5; x >>>= 0;
    rng.s = x;
    return x / 0x100000000;
  };
  const pixels = new Float32Array(batch * 3 * 32 * 32);
  for (let i = 0; i < pixels.length; i++) pixels[i] = next() * 2 - 1;
  const labels = new Int32Array(batch);
  for (let i = 0; i < batch; i++) labels[i] = Math.floor(next() * 10);
  return { pixels, labels };
}

// ── jax-js ─────────────────────────────────────────────────────────────────────────

/** The slice of jax-js this file uses, typed from its published `index.d.ts`. */
interface JArr {
  readonly ref: JArr;
  readonly shape: number[];
  dispose(): void;
  add(o: JArr | number): JArr; sub(o: JArr | number): JArr; mul(o: JArr | number): JArr; div(o: JArr | number): JArr;
  mean(axis?: number | number[], opts?: { keepdims?: boolean }): JArr;
  sum(axis?: number | number[]): JArr;
  neg(): JArr;
  reshape(shape: number[]): JArr;
  jsAsync(): Promise<unknown>;
}
/** jax-js's pytrees are any nesting of arrays; the optimizer state is its own opaque one. */
type OptState = unknown;
interface JaxModule {
  init(...devices: string[]): Promise<string[]>;
  defaultDevice(d?: string): string;
  blockUntilReady<T>(t: T): Promise<void>;
  numpy: {
    array(v: Float32Array | Int32Array, opts?: { shape?: number[]; dtype?: unknown }): JArr;
    int32: unknown;
    var_(x: JArr, axis?: number[], opts?: { mean?: JArr; keepdims?: boolean }): JArr;
    sqrt(x: JArr): JArr;
    dot(a: JArr, b: JArr): JArr;
  };
  lax: { conv(lhs: JArr, rhs: JArr, strides: number[], padding: string | [number, number][]): JArr };
  nn: { relu(x: JArr): JArr; logSoftmax(x: JArr, axis?: number): JArr; oneHot(x: JArr, n: number): JArr };
  random: { key(seed: number): unknown; split(key: unknown, n: number): unknown[]; normal(key: unknown, shape: number[]): JArr };
  tree: { ref<T>(t: T): T; dispose<T>(t: T): void };
  jit<F>(f: F): F & { dispose(): void };
  /** The gradient has the shape of the first argument's tree. */
  valueAndGrad<P, R extends unknown[]>(f: (p: P, ...rest: R) => JArr): (p: P, ...rest: R) => [JArr, P];
}
interface OptaxModule {
  sgd(lr: number, opts?: { momentum?: number | null; nesterov?: boolean }): {
    init<P>(p: P): OptState; update<P>(g: P, s: OptState, p?: P): [P, OptState];
  };
  applyUpdates<P>(p: P, u: P): P;
}

/** Arrays only — a number leaf (the stride) in a jit-traced tree becomes a tracer and `conv`
 *  refuses it, so the strides live in `STAGES` and are indexed by block position. */
interface JaxBlock { w1: JArr; g1: JArr; b1: JArr; w2: JArr; g2: JArr; b2: JArr; wd?: JArr; gd?: JArr; bd?: JArr }
interface JaxParams { stem: { w: JArr; g: JArr; b: JArr }; blocks: JaxBlock[]; fc: { w: JArr; b: JArr } }

const EPS = 1e-5;
const STAGES = [[64, 64, 1], [64, 64, 1], [64, 128, 2], [128, 128, 1],
                [128, 256, 2], [256, 256, 1], [256, 512, 2], [512, 512, 1]] as const;

/** He-scaled normal weights drawn on the host (xorshift + Box–Muller) and uploaded, ones/zeros
 *  for the norms. **Not `random.normal`**: in 0.1.25 that op fails to compile for the WebGPU
 *  device ("Receiver must be an instance of class M" out of its shape compiler — bisected op by
 *  op, 2026-09-19), and the draw is not what is being timed. */
function jaxParams(J: JaxModule): JaxParams {
  let seed = 12345;
  const normal = (shape: number[], fan: number): JArr => {
    const n = shape.reduce((a, b) => a * b, 1);
    const out = new Float32Array(n);
    let st = (seed++) >>> 0;
    const u = (): number => { st ^= st << 13; st >>>= 0; st ^= st >>> 17; st ^= st << 5; st >>>= 0; return (st + 1) / 4294967297; };
    const scale = Math.sqrt(2 / fan);
    for (let i = 0; i < n; i++) out[i] = Math.sqrt(-2 * Math.log(u())) * Math.cos(2 * Math.PI * u()) * scale;
    return J.numpy.array(out, { shape });
  };
  const ones = (c: number): JArr => J.numpy.array(new Float32Array(c).fill(1), { shape: [c] });
  const zeros = (c: number): JArr => J.numpy.array(new Float32Array(c), { shape: [c] });
  const blocks: JaxBlock[] = STAGES.map(([cin, cout, stride]) => {
    const b: JaxBlock = {
      w1: normal([cout, cin, 3, 3], cin * 9), g1: ones(cout), b1: zeros(cout),
      w2: normal([cout, cout, 3, 3], cout * 9), g2: ones(cout), b2: zeros(cout),
    };
    if (stride !== 1 || cin !== cout) { b.wd = normal([cout, cin, 1, 1], cin); b.gd = ones(cout); b.bd = zeros(cout); }
    return b;
  });
  return {
    stem: { w: normal([64, 3, 3, 3], 27), g: ones(64), b: zeros(64) },
    blocks,
    fc: { w: normal([512, 10], 512), b: zeros(10) },
  };
}

/** BatchNorm in training mode from the primitives — batch statistics, affine. `x` used thrice. */
function jaxBN(J: JaxModule, x: JArr, g: JArr, b: JArr): JArr {
  const mu = x.ref.mean([0, 2, 3], { keepdims: true });
  const v = J.numpy.var_(x.ref, [0, 2, 3], { mean: mu.ref, keepdims: true });
  return x.sub(mu).div(J.numpy.sqrt(v.add(EPS))).mul(g.reshape([1, -1, 1, 1])).add(b.reshape([1, -1, 1, 1]));
}

export async function runStepJax(batch = 32, steps = 5, warmup = 2): Promise<PeerStep> {
  const J = (await import(/* @vite-ignore */ JAX_URL)) as unknown as JaxModule;
  const O = (await import(/* @vite-ignore */ OPTAX_URL)) as unknown as OptaxModule;
  const devices = await J.init("webgpu");
  if (!devices.includes("webgpu")) throw new Error(`jax-js: no WebGPU device (${devices.join(",")})`);
  J.defaultDevice("webgpu");
  const { numpy: np, lax, nn } = J;
  const { pixels, labels } = seededNCHW(batch);

  const conv3 = (x: JArr, w: JArr, s: number): JArr => lax.conv(x, w, [s, s], [[1, 1], [1, 1]]);
  const block = (p: JaxBlock, x: JArr, stride: number): JArr => {
    const shortcut = p.wd !== undefined;
    let h = nn.relu(jaxBN(J, conv3(x.ref, p.w1, stride), p.g1, p.b1));
    h = jaxBN(J, conv3(h, p.w2, 1), p.g2, p.b2);
    const sc = shortcut
      ? jaxBN(J, lax.conv(x, p.wd as JArr, [stride, stride], "VALID"), p.gd as JArr, p.bd as JArr)
      : x;
    return nn.relu(h.add(sc));
  };
  // `jit` compiles the forward once per shape and fuses what it can — jax-js at its best.
  // The strides come from `STAGES` (plain numbers in the closure), not from the traced tree.
  const forward = J.jit((params: JaxParams, x: JArr): JArr => {
    let h = nn.relu(jaxBN(J, conv3(x, params.stem.w, 1), params.stem.g, params.stem.b));
    params.blocks.forEach((blk, i) => { h = block(blk, h, STAGES[i]?.[2] ?? 1); });
    return np.dot(h.mean([2, 3]), params.fc.w).add(params.fc.b);
  });
  const loss = (params: JaxParams, x: JArr, y: JArr): JArr =>
    nn.logSoftmax(forward(params, x), -1).mul(nn.oneHot(y, 10)).sum(1).mean().neg();

  let params = jaxParams(J);
  const opt = O.sgd(0.05, { momentum: 0.9 });
  let state = opt.init(J.tree.ref(params));
  const one = async (): Promise<number> => {
    const x = np.array(pixels, { shape: [batch, 3, 32, 32] });
    const y = np.array(labels);                     // an Int32Array is int32 on its own
    const [lossVal, grads] = J.valueAndGrad(loss)(J.tree.ref(params), x, y);
    const [updates, next] = opt.update(grads, state);
    state = next;
    params = O.applyUpdates(params, updates);
    await J.blockUntilReady(params);                 // the GPU is done here — the sync, as TF.js's data()
    const v = await lossVal.jsAsync();
    return typeof v === "number" ? v : NaN;
  };
  for (let i = 0; i < warmup; i++) await one();
  const t0 = performance.now();
  let last = NaN;
  for (let i = 0; i < steps; i++) last = await one();
  const msPerStep = (performance.now() - t0) / steps;
  J.tree.dispose(params); J.tree.dispose(state); forward.dispose();
  return { batch, msPerStep, lastLoss: last };
}

export async function reportJax(batches: readonly number[] = [16, 32, 64]): Promise<string> {
  const lines = [`jax-js @jax-js/jax 0.1.25 · @jax-js/optax 0.1.2 · device webgpu · NCHW · jit forward`];
  try {
    for (const b of batches) {
      const r = await runStepJax(b);
      lines.push(`batch ${String(r.batch).padStart(3)}  ${r.msPerStep.toFixed(1).padStart(8)} ms/step  loss ${r.lastLoss.toFixed(4)}`);
    }
  } catch (e) {
    const err = e as Error;
    lines.push(`  jax-js could not run: ${String(err && err.message || e).slice(0, 300)}`);
    lines.push(`  ${String(err && err.stack || "").split("\n").slice(0, 6).join("\n  ")}`);
  }
  return lines.join("\n");
}

// ── Burn (Rust → wasm, wgpu) ──────────────────────────────────────────────────────

interface BurnModule {
  default(input?: unknown): Promise<unknown>;         // wasm-bindgen's module init
  init(): Promise<void>;                                // the crate's: device + model + optimizer
  train_step(x: Float32Array, y: Int32Array, batch: number): Promise<number>;
  backend(): string;
}

export async function runStepBurn(B: BurnModule, batch = 32, steps = 5, warmup = 2): Promise<PeerStep> {
  const { pixels, labels } = seededNCHW(batch);
  for (let i = 0; i < warmup; i++) await B.train_step(pixels, labels, batch);
  const t0 = performance.now();
  let last = NaN;
  for (let i = 0; i < steps; i++) last = await B.train_step(pixels, labels, batch);   // resolves after the GPU finishes
  return { batch, msPerStep: (performance.now() - t0) / steps, lastLoss: last };
}

export async function reportBurn(batches: readonly number[] = [16, 32, 64]): Promise<string> {
  let B: BurnModule;
  try {
    B = (await import(/* @vite-ignore */ BURN_PKG)) as unknown as BurnModule;
    await B.default();
    await B.init();
  } catch (e) {
    return `Burn: bundle not loadable (${String(e && (e as Error).message || e).slice(0, 200)}) — build tests/browser/burn_resnet18 first (docs/BOOK.md)`;
  }
  const lines = [`Burn · ${B.backend()}`];
  try {
    for (const b of batches) {
      const r = await runStepBurn(B, b);
      lines.push(`batch ${String(r.batch).padStart(3)}  ${r.msPerStep.toFixed(1).padStart(8)} ms/step  loss ${r.lastLoss.toFixed(4)}`);
    }
  } catch (e) {
    lines.push(`  Burn could not run: ${String(e && (e as Error).message || e).slice(0, 300)}`);
  }
  return lines.join("\n");
}
