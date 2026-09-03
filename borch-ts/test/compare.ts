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

import { Tensor, noGrad } from "../src/tensor.js";
import { load } from "../src/serialize.js";
import { exportOnnx } from "../src/onnx.js";
import { device as dev } from "../src/tensor.js";
import { ResNet18 } from "./bench.js";

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


// ── inference: the same weights in borch.ts and in ONNX Runtime Web ────────────────
//
// **Training is the claim; inference is the honest neighbour.** ORT Web is an
// inference runtime with years of kernel work behind it, and this library expects to
// lose here. The point is to say by how much, on the same page, with the same weights —
// exported once from torch by `tests/browser/export_resnet18.py` as safetensors for
// borch.ts and ONNX for ORT — and only after both runtimes reproduce torch's logits on a
// seeded input to 1e-3. A speed without that gate is a speed of something else.

interface OrtTensorLike { data: Float32Array }
interface OrtSession { run(feeds: Record<string, unknown>): Promise<Record<string, OrtTensorLike>> }
interface Ort {
  env: { wasm: { wasmPaths: string } };
  Tensor: new (type: string, data: Float32Array, dims: number[]) => unknown;
  InferenceSession: { create(url: string | Uint8Array, opts: { executionProviders: string[] }): Promise<OrtSession> };
}

function ort(): Ort {
  const got = (globalThis as { ort?: Ort }).ort;
  if (!got) throw new Error("ONNX Runtime Web is not on the page — compare.html loads vendor/ort.webgpu.min.js first");
  return got;
}

interface Probe { input: number[]; shape: number[]; logits: number[]; torch: string }

const OUT = "../test/out/";

async function bytes(url: string): Promise<Uint8Array> {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: ${r.status} — run tests/browser/export_resnet18.py first`);
  return new Uint8Array(await r.arrayBuffer());
}

function maxAbsDiff(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let m = 0;
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs((a[i] ?? 0) - (b[i] ?? 0)));
  return m;
}

/** Warm, time, mean — the bench's shape. Inference forwards are short (single-digit ms), so
 * they take twenty timed runs where a training step takes five. `run` must include the readback. */
async function timed(run: () => Promise<unknown>, warmup = 2, steps = 5): Promise<number> {
  for (let i = 0; i < warmup; i++) await run();
  const t0 = performance.now();
  for (let i = 0; i < steps; i++) await run();
  return (performance.now() - t0) / steps;
}

export async function reportInfer(batches: readonly number[] = [1, 16]): Promise<string> {
  const probe = JSON.parse(new TextDecoder().decode(await bytes(OUT + "resnet18_cifar.probe.json"))) as Probe;
  const lines: string[] = [`weights from torch ${probe.torch}, exported by tests/browser/export_resnet18.py`];

  // borch.ts: the bench's ResNet18 with the exported state, in eval mode.
  const table = load(await bytes(OUT + "resnet18_cifar.safetensors")) as Record<string, Tensor>;
  const model = new ResNet18();
  // Not strict: `num_batches_tracked` is not in the file (an int64 counter with no part
  // in inference), and it must be the only thing missing.
  const report = model.loadStateDict(table, false);
  const missing = report.missing.filter((k) => !k.endsWith("num_batches_tracked"));
  if (missing.length || report.unexpected.length) {
    throw new Error(`state dict did not fit: missing ${missing.join(",")} unexpected ${report.unexpected.join(",")}`);
  }
  model.eval();
  const x1 = Tensor.from(Float32Array.from(probe.input), probe.shape);
  const ours = await noGrad(() => model.forward(x1)).toArray();
  const oursGap = maxAbsDiff(ours, probe.logits);

  // ORT Web, WebGPU execution provider, the ONNX twin of the same weights.
  const o = ort();
  o.env.wasm.wasmPaths = "../../vendor/";
  const session = await o.InferenceSession.create(OUT + "resnet18_cifar.onnx", { executionProviders: ["webgpu"] });
  const feed = (data: Float32Array, b: number) => ({ input: new o.Tensor("float32", data, [b, 3, 32, 32]) });
  const theirsOut = await session.run(feed(Float32Array.from(probe.input), 1));
  const theirs = theirsOut["logits"]?.data ?? new Float32Array();
  const theirsGap = maxAbsDiff(theirs, probe.logits);

  const GATE = 1e-3;
  lines.push(`gate: max |logits − torch| on the probe input — borch.ts ${oursGap.toExponential(2)} · ORT Web ${theirsGap.toExponential(2)} · limit ${GATE}`);
  if (oursGap > GATE || theirsGap > GATE) {
    lines.push("**a runtime does not reproduce torch's logits — its speed below is a speed of something else**");
  }

  for (const b of batches) {
    const data = new Float32Array(b * 3 * 32 * 32);
    for (let i = 0; i < b; i++) data.set(probe.input, i * 3 * 32 * 32);
    const xb = Tensor.from(data, [b, 3, 32, 32]);
    const oursMs = await timed(() => noGrad(() => model.forward(xb)).toArray(), 3, 20);
    const theirsMs = await timed(() => session.run(feed(data, b)), 3, 20);
    lines.push(`batch ${String(b).padStart(3)}  forward  borch.ts ${oursMs.toFixed(2).padStart(8)} ms · ORT Web ${theirsMs.toFixed(2).padStart(8)} ms · ratio ${(oursMs / theirsMs).toFixed(2)}× (borch/ORT)`);
    // Where our forward spends itself: dispatches per forward, and GPU time by kind of
    // kernel (a pass per dispatch while profiling, so read the share, not the total).
    // The count separates "too many calls" from "a slow kernel" — the eval-mode batch
    // norm was the former, six dispatches a layer over twenty layers (measured here).
    const d = dev();
    const d0 = d.dispatches;
    await noGrad(() => model.forward(xb)).toArray();
    const dispatches = d.dispatches - d0;
    await d.profile(() => noGrad(() => model.forward(xb)).toArray());
    const hot: [string, number][] = [];
    for (const [kind, ns] of d.nsByKind) hot.push([kind, ns / 1e6]);
    hot.sort((p, q) => q[1] - p[1]);
    const total = hot.reduce((a, [, ms]) => a + ms, 0);
    lines.push(`           borch.ts ${dispatches} dispatches/forward · GPU time (ms, total ${total.toFixed(1)}): `
      + hot.slice(0, 8).map(([k, ms]) => `${k} ${ms.toFixed(1)}`).join(" · ")
      + (d.profileDropped ? ` · ${d.profileDropped} dropped` : ""));
  }

  // The same network with every batch norm folded into the convolution before it —
  // `nn.fuseConvBnEval`, torch's `fuse_conv_bn_eval`. Gated the same way first.
  model.fuse();
  const fusedGap = maxAbsDiff(await noGrad(() => model.forward(x1)).toArray(), probe.logits);
  lines.push(`fused (batch norms folded into the convolutions): max |logits − torch| ${fusedGap.toExponential(2)}`);
  if (fusedGap > GATE) lines.push("**the fused network does not reproduce torch's logits**");
  for (const b of batches) {
    const data = new Float32Array(b * 3 * 32 * 32);
    for (let i = 0; i < b; i++) data.set(probe.input, i * 3 * 32 * 32);
    const xb = Tensor.from(data, [b, 3, 32, 32]);
    const ms = await timed(() => noGrad(() => model.forward(xb)).toArray(), 3, 20);
    const d = dev();
    const d0 = d.dispatches;
    await noGrad(() => model.forward(xb)).toArray();
    lines.push(`batch ${String(b).padStart(3)}  forward  borch.ts fused ${ms.toFixed(2).padStart(8)} ms · ${d.dispatches - d0} dispatches/forward`);
  }

  // The whole story on one page: the fused network leaves as ONNX — borch's own file,
  // not torch's — and ORT Web runs it, gated against torch's logits like everything
  // above. Training here, serving anywhere.
  const exported = await exportOnnx(model, x1);
  const ownSession = await o.InferenceSession.create(exported.bytes, { executionProviders: ["webgpu"] });
  const ownOut = await ownSession.run(feed(Float32Array.from(probe.input), 1));
  const ownGap = maxAbsDiff(ownOut["output"]?.data ?? new Float32Array(), probe.logits);
  lines.push(`borch's own ONNX export (${exported.ops.length} nodes, ${(exported.bytes.length / 1e6).toFixed(1)} MB) run by ORT Web: max |logits − torch| ${ownGap.toExponential(2)}`);
  if (ownGap > GATE) lines.push("**ORT running borch's export does not reproduce torch's logits**");
  for (const b of batches) {
    const data = new Float32Array(b * 3 * 32 * 32);
    for (let i = 0; i < b; i++) data.set(probe.input, i * 3 * 32 * 32);
    const ms = await timed(() => ownSession.run(feed(data, b)), 3, 20);
    lines.push(`batch ${String(b).padStart(3)}  forward  ORT Web on borch's export ${ms.toFixed(2).padStart(8)} ms`);
  }
  return lines.join("\n");
}
