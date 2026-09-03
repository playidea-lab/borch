/**
 * The comparison bench — the ResNet-18 (CIFAR) training step of `bench.ts`, run in
 * **TF.js** on the same page, same machine, same batch sizes, same seeded data.
 *
 * **Why this exists.** The README's table (154.9 vs 118.5 ms/step) was measured against a
 * TF.js implementation that no longer exists, on a machine nobody wrote down, so it
 * survives only as a ratio. This is the measurement made again, reproducibly: the
 * competitor's bytes are pinned by `tests/browser/assets.lock` (TF.js 4.22.0, UMD, global
 * `tf`), the model is written with TF.js's own layers API in its own native layout
 * (NHWC), and it uses TF.js's own optimizer and loss — each library at its best, not one
 * library emulating the other.
 *
 * **What is held equal**: architecture (stem 3×3·64, four stages of two blocks, 1×1
 * shortcuts where the shape changes, global average pool, dense 10), SGD with momentum
 * 0.05/0.9, cross-entropy, the xorshift-seeded pixels and labels of `bench.ts`, two
 * warm-up steps then five timed, and a readback of the loss every step so the clock
 * includes the GPU finishing.
 *
 * **What is not**: BatchNorm momentum (TF.js 0.99, torch 0.1 — no effect on speed),
 * memory layout (each library's native one — that *is* part of the comparison), and
 * shader compilation, which the warm-up pays for on both sides.
 */

// The UMD build puts `tf` on the window; this file is typed against what it uses.
type TfTensor = { dataSync(): Float32Array; data(): Promise<Float32Array>; dispose(): void };
type TfLayer = { apply(x: unknown): unknown };
interface Tf {
  setBackend(name: string): Promise<boolean>;
  ready(): Promise<void>;
  getBackend(): string;
  tensor(values: Float32Array, shape: number[]): TfTensor;
  oneHot(indices: unknown, depth: number): TfTensor;
  tensor1d(values: Int32Array, dtype: string): unknown;
  input(config: { shape: number[] }): unknown;
  model(config: { inputs: unknown; outputs: unknown }): {
    trainableWeights: { read(): TfTensor }[];
    apply(x: unknown, opts?: { training: boolean }): unknown;
    countParams(): number;
  };
  layers: {
    conv2d(c: Record<string, unknown>): TfLayer;
    batchNormalization(c?: Record<string, unknown>): TfLayer;
    reLU(): TfLayer;
    add(): TfLayer;
    globalAveragePooling2d(c?: Record<string, unknown>): TfLayer;
    dense(c: Record<string, unknown>): TfLayer;
  };
  train: { momentum(lr: number, m: number): { minimize(f: () => TfTensor, returnCost: boolean): TfTensor | null } };
  losses: { softmaxCrossEntropy(labels: unknown, logits: unknown): TfTensor };
  tidy<T>(f: () => T): T;
  memory(): { numTensors: number; numBytes: number };
  version: { tfjs: string };
}

function tf(): Tf {
  const got = (globalThis as { tf?: Tf }).tf;
  if (!got) throw new Error("TF.js is not on the page — compare.html loads vendor/tf.min.js first");
  return got;
}

function block(t: Tf, x: unknown, cin: number, cout: number, stride: number): unknown {
  const conv = (filters: number, k: number, s: number) =>
    t.layers.conv2d({ filters, kernelSize: k, strides: s, padding: "same", useBias: false });
  let out: unknown = conv(cout, 3, stride).apply(x);
  out = t.layers.reLU().apply(t.layers.batchNormalization().apply(out));
  out = t.layers.batchNormalization().apply(conv(cout, 3, 1).apply(out));
  const side = stride !== 1 || cin !== cout
    ? t.layers.batchNormalization().apply(conv(cout, 1, stride).apply(x))
    : x;
  return t.layers.reLU().apply(t.layers.add().apply([out, side]));
}

export function resnet18(t: Tf, classes = 10) {
  const input = t.input({ shape: [32, 32, 3] });
  let h: unknown = t.layers.conv2d({ filters: 64, kernelSize: 3, strides: 1, padding: "same", useBias: false }).apply(input);
  h = t.layers.reLU().apply(t.layers.batchNormalization().apply(h));
  for (const [cin, cout, stride] of [[64, 64, 1], [64, 64, 1], [64, 128, 2], [128, 128, 1],
                                     [128, 256, 2], [256, 256, 1], [256, 512, 2], [512, 512, 1]] as const) {
    h = block(t, h, cin, cout, stride);
  }
  h = t.layers.globalAveragePooling2d({}).apply(h);
  const out = t.layers.dense({ units: classes }).apply(h);
  return t.model({ inputs: input, outputs: out });
}

export interface CompareStep { batch: number; msPerStep: number; params: number; lastLoss: number }

/** The same numbers `bench.ts` draws: xorshift32 from 12345, pixels in [-1, 1), labels 0..9. */
function seeded(batch: number): { pixels: Float32Array; labels: Int32Array } {
  const rng = { s: 12345 };
  const next = (): number => {
    let x = rng.s; x ^= x << 13; x >>>= 0; x ^= x >> 17; x ^= x << 5; x >>>= 0;
    rng.s = x;
    return x / 0x100000000;
  };
  // bench.ts draws NCHW; the same draw is laid out NHWC here, which is TF.js's own layout.
  const nchw = new Float32Array(batch * 3 * 32 * 32);
  for (let i = 0; i < nchw.length; i++) nchw[i] = next() * 2 - 1;
  const labels = new Int32Array(batch);
  for (let i = 0; i < batch; i++) labels[i] = Math.floor(next() * 10);
  const pixels = new Float32Array(batch * 32 * 32 * 3);
  for (let n = 0; n < batch; n++) for (let c = 0; c < 3; c++) for (let p = 0; p < 1024; p++) {
    pixels[(n * 1024 + p) * 3 + c] = nchw[(n * 3 + c) * 1024 + p] ?? 0;
  }
  return { pixels, labels };
}

export async function runStepTf(batch = 32, steps = 5, warmup = 2): Promise<CompareStep> {
  const t = tf();
  const { pixels, labels } = seeded(batch);
  const x = t.tensor(pixels, [batch, 32, 32, 3]);
  const y = t.oneHot(t.tensor1d(labels, "int32"), 10);
  const model = resnet18(t);
  const opt = t.train.momentum(0.05, 0.9);
  const one = async (): Promise<number> => {
    const cost = t.tidy(() => opt.minimize(() => t.tidy(() =>
      t.losses.softmaxCrossEntropy(y, model.apply(x, { training: true }))), true));
    if (!cost) return NaN;
    const v = (await cost.data())[0] ?? NaN;   // the readback is the sync, as bench.ts's item()
    cost.dispose();
    return v;
  };
  for (let i = 0; i < warmup; i++) await one();
  const t0 = performance.now();
  let last = NaN;
  for (let i = 0; i < steps; i++) last = await one();
  const msPerStep = (performance.now() - t0) / steps;
  x.dispose(); y.dispose();
  return { batch, msPerStep, params: model.countParams(), lastLoss: last };
}

export async function reportTf(batches: readonly number[] = [16, 32, 64]): Promise<string> {
  const t = tf();
  await t.setBackend("webgpu");
  await t.ready();
  const lines = [`TF.js ${t.version.tfjs} · backend ${t.getBackend()}`];
  for (const b of batches) {
    const r = await runStepTf(b);
    lines.push(`batch ${String(r.batch).padStart(3)}  ${r.msPerStep.toFixed(1).padStart(8)} ms/step  ` +
               `params ${r.params}  loss ${r.lastLoss.toFixed(4)}`);
  }
  return lines.join("\n");
}
