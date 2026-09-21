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

import { Tensor, noGrad, scope } from "../src/tensor.js";
import { Device } from "../src/device.js";
import { calibrateInt8, clearInt8, fuseForInference, quantizeForInt8 } from "../src/nn.js";
import { compiled } from "../src/compile.js";
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
interface OrtSession { run(feeds: Record<string, unknown>): Promise<Record<string, OrtTensorLike>>; release(): Promise<void> }
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
/** The bytes at `url`, or `null` when there is no such file. */
async function maybeBytes(url: string): Promise<Uint8Array | null> {
  try { return await bytes(url); } catch { return null; }
}

/** Top-1 of `forward` over the slice, batches of `batch`; the logits' argmax on the host. */
async function top1(forward: (x: Tensor) => Tensor, pixels: Float32Array, labels: Uint32Array, batch: number): Promise<number> {
  const n = labels.length;
  let correct = 0;
  for (let i = 0; i < n; i += batch) {
    const b = Math.min(batch, n - i);
    // A scope a batch: thirty forwards of a hundred images without one filled the card
    // (the allocation's out-of-memory is reported late, as an invalid buffer in a bind group).
    const out = await scope(async () => {
      const x = Tensor.from(pixels.subarray(i * 3072, (i + b) * 3072), [b, 3, 32, 32]);
      return noGrad(() => forward(x)).toArray();
    });
    for (let j = 0; j < b; j++) {
      let best = 0;
      for (let c = 1; c < 10; c++) if ((out[j * 10 + c] ?? 0) > (out[j * 10 + best] ?? 0)) best = c;
      if (best === labels[i + j]) correct++;
    }
  }
  return correct / n;
}

async function reportAccuracy(): Promise<string[]> {
  const weights = await maybeBytes(OUT + "resnet18_cifar_trained.safetensors");
  const slice = await maybeBytes(OUT + "cifar10_test.bin");
  if (!weights || !slice) return ["accuracy: no trained network beside the seed-0 one (tests/browser/train_resnet18_cifar.py writes it) — the rows above are speed only"];
  const view = new DataView(slice.buffer, slice.byteOffset, slice.byteLength);
  const n = view.getUint32(0, true);
  const pixels = new Float32Array(slice.buffer.slice(slice.byteOffset + 4, slice.byteOffset + 4 + n * 3072 * 4));
  const labels = new Uint32Array(slice.buffer.slice(slice.byteOffset + 4 + n * 3072 * 4, slice.byteOffset + 4 + n * 3072 * 4 + n * 4));
  const model = new ResNet18();
  const report = model.loadStateDict(load(weights) as Record<string, Tensor>, false);
  const missing = report.missing.filter((k) => !k.endsWith("num_batches_tracked"));
  if (missing.length || report.unexpected.length) throw new Error(`trained state dict did not fit: missing ${missing.join(",")} unexpected ${report.unexpected.join(",")}`);
  model.eval();
  fuseForInference(model);
  const BATCH = 100;
  // The slice is split: the second half calibrates the static scales, the first half is
  // scored — so the static int8 forward is not measured on the images that set its scales.
  const half = Math.floor(n / 2);
  const scorePix = pixels.subarray(0, half * 3072), scoreLab = labels.subarray(0, half);
  const calibPix = pixels.subarray(half * 3072);
  const f32 = await top1((x) => model.forward(x), scorePix, scoreLab, BATCH);
  const lines = [`accuracy on ${half} CIFAR-10 test images (trained weights): borch.ts f32 fused ${(100 * f32).toFixed(2)}%`];
  if (Device.subgroupInt8) {
    const layers = await quantizeForInt8(model);
    const q = await top1((x) => model.forward(x), scorePix, scoreLab, BATCH);
    await calibrateInt8(model, calibPix, BATCH, [3, 32, 32]);
    const qs = await top1((x) => model.forward(x), scorePix, scoreLab, BATCH);
    clearInt8(model);
    const GATE_POINTS = 0.5;
    const verdict = (drop: number) => `${drop >= 0 ? "−" : "+"}${Math.abs(drop).toFixed(2)} points · gate within ${GATE_POINTS}: ${drop <= GATE_POINTS ? "passed" : "**FAILED — the int8 path is not routed to**"}`;
    lines.push(`accuracy on the same images: borch.ts int8 dynamic (${layers} layers) ${(100 * q).toFixed(2)}% · ${verdict(100 * (f32 - q))}`);
    lines.push(`accuracy on the same images: borch.ts int8 static (scales from the other ${n - half}) ${(100 * qs).toFixed(2)}% · ${verdict(100 * (f32 - qs))}`);
  } else {
    lines.push("accuracy: no int8 configuration on this adapter — the int8 gate runs where there is one");
  }
  // **ORT's int8 on the same images** — quantised by onnxruntime from the trained network,
  // calibrated on the other half of the slice as borch's static scales are. Its f32 file
  // first, so the drop is ORT's own, not a difference between the two f32 forwards.
  const trainedF32 = await maybeBytes(OUT + "resnet18_cifar_trained.onnx");
  const trainedInt8 = await maybeBytes(OUT + "resnet18_cifar_trained_int8.onnx");
  if (trainedF32 && trainedInt8) {
    const o = ort();
    const feed: Feed = (data, b) => ({ input: new o.Tensor("float32", data, [b, 3, 32, 32]) });
    const s32 = await o.InferenceSession.create(trainedF32, { executionProviders: ["webgpu"] });
    const a32 = await ortTop1(s32, feed, scorePix, scoreLab, BATCH);
    const s8 = await o.InferenceSession.create(trainedInt8, { executionProviders: ["webgpu"] });
    const a8 = await ortTop1(s8, feed, scorePix, scoreLab, BATCH);
    lines.push(`accuracy on the same images: ORT Web f32 ${(100 * a32).toFixed(2)}% · ORT Web int8 (QDQ, per-channel, calibrated on the other ${n - half}) ${(100 * a8).toFixed(2)}% · ${a32 - a8 >= 0 ? "−" : "+"}${Math.abs(100 * (a32 - a8)).toFixed(2)} points`);
  } else {
    lines.push("accuracy: no trained ONNX pair for ORT (resnet18_cifar_trained.onnx + _int8.onnx) — ORT's int8 accuracy not measured");
  }
  return lines;
}

type Feed = (data: Float32Array, b: number) => Record<string, unknown>;

/** ORT's f16 and int8 (QDQ) forms of the same ResNet-18, on WebGPU — and the int8 on the
 * wasm provider as well: **a WebGPU time that is the wasm time is a fallback**, whatever
 * the session said about its providers, and the row has to be able to show it. Each is
 * held to torch's logits like the f32 file; the gaps are printed, not gated (f16 and int8
 * are not held to 1e-3 — the number that stands beside them is how far they are). */
async function reportOrtVariants(o: Ort, probe: Probe, batches: readonly number[], feed: Feed): Promise<string[]> {
  const lines: string[] = [];
  let scale = 0; for (const v of probe.logits) scale = Math.max(scale, Math.abs(v));
  for (const variant of ["f16", "int8"] as const) {
    const file = await maybeBytes(OUT + `resnet18_cifar_${variant}.onnx`);
    if (!file) { lines.push(`ORT Web ${variant}: no resnet18_cifar_${variant}.onnx (tests/browser/export_ort_variants.py writes it) — not measured`); continue; }
    const providers = variant === "int8" ? ["webgpu", "wasm"] : ["webgpu"];
    for (const ep of providers) {
      // **A file ORT cannot run is a row, not a crash**: on the RTX 5080's Chrome (Linux,
      // Vulkan) the f16 session refuses at creation — "Program Transpose requires f16 but
      // the device does not support it" — and that sentence is the measurement.
      let session: OrtSession;
      try {
        session = await o.InferenceSession.create(file, { executionProviders: [ep] });
      } catch (err) {
        const why = String(err instanceof Error ? err.message : err).split("\n")[0] ?? "";
        lines.push(`ORT Web ${variant} on ${ep}: could not run — ${why.replace(/^Can't create a session\. ERROR_CODE: \d+, ERROR_MESSAGE: /, "")}`);
        continue;
      }
      const out = (await session.run(feed(Float32Array.from(probe.input), 1)))["logits"]?.data ?? new Float32Array();
      const gap = maxAbsDiff(out, probe.logits);
      const cells: string[] = [];
      for (const b of batches) {
        const data = new Float32Array(b * 3 * 32 * 32);
        for (let i = 0; i < b; i++) data.set(probe.input, i * 3 * 32 * 32);
        const ms = await timed(() => session.run(feed(data, b)), 3, 20);
        cells.push(`batch ${b} ${ms.toFixed(2)} ms`);
      }
      lines.push(`ORT Web ${variant} on ${ep}: ${cells.join(" · ")} · max |logits − torch| ${gap.toExponential(1)} (${(gap / scale).toExponential(1)} of the logits' scale)`);
      // **Released, not left.** The rows after these read 3× slower on the 5080 with the
      // sessions alive — the wasm provider's threads, or the WebGPU provider's device,
      // keep the machine busy after `run` returns (`docs/FIRST.md` 2a).
      await session.release();
    }
  }
  return lines;
}

/** Top-1 of an ORT session over the slice — the same images and batches as `top1`. */
async function ortTop1(session: OrtSession, feed: Feed, pixels: Float32Array, labels: Uint32Array, batch: number): Promise<number> {
  const n = labels.length;
  let correct = 0;
  for (let i = 0; i < n; i += batch) {
    const b = Math.min(batch, n - i);
    const out = (await session.run(feed(pixels.slice(i * 3072, (i + b) * 3072), b)))["logits"]?.data ?? new Float32Array();
    for (let j = 0; j < b; j++) {
      let best = 0;
      for (let c = 1; c < 10; c++) if ((out[j * 10 + c] ?? 0) > (out[j * 10 + best] ?? 0)) best = c;
      if (best === labels[i + j]) correct++;
    }
  }
  return correct / n;
}

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
    // **A scope a forward, as a page would run it.** Without one every forward makes its
    // intermediates on fresh buffers, and on Direct3D 12 a buffer is milliseconds to make:
    // the RTX 5050 Laptop read 152 ms for a batch-16 forward of 22 ms of GPU (2026-09-21)
    // while its training step, scoped, read 32 for 28. The clock here is the forward's,
    // not the allocator's.
    const oursMs = await timed(() => scope(async () => noGrad(() => model.forward(xb)).toArray()), 3, 20);
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
    // **A kind's time is a sum over its dispatches, and the count is printed with it**: the
    // ResNet-18 has three 512 → 512 convolutions at 4 × 4, and their summed 0.37 ms was read
    // for a day as one slow dispatch 3.5× the bench's — it was three at the bench's speed.
    const hot: [string, number, number][] = [];
    for (const [kind, ns] of d.nsByKind) hot.push([kind, ns / 1e6, d.countByKind.get(kind) ?? 1]);
    hot.sort((p, q) => q[1] - p[1]);
    const total = hot.reduce((a, [, ms]) => a + ms, 0);
    lines.push(`           borch.ts ${dispatches} dispatches/forward · GPU time (ms, total ${total.toFixed(1)}, ×count): `
      + hot.slice(0, 8).map(([k, ms, n]) => `${k} ${ms.toFixed(2)}${n > 1 ? `×${n}` : ""}`).join(" · ")
      + (d.profileDropped ? ` · ${d.profileDropped} dropped` : ""));
  }

  // **ORT at its own reduced precisions** — the int8 rows below stand beside these too,
  // not only beside ORT's f32 (`tests/browser/export_ort_variants.py` writes the files).
  lines.push(...await reportOrtVariants(o, probe, batches, feed));

  // The same network with every batch norm folded into the convolution before it —
  // `nn.fuseConvBnEval`, torch's `fuse_conv_bn_eval`. Gated the same way first.
  model.fuse();
  const faults0 = dev().faults.count;
  const fusedGap = maxAbsDiff(await noGrad(() => model.forward(x1)).toArray(), probe.logits);
  // **A validation fault and the number is not a number.** An invalid pipeline does
  // nothing, and the output buffer then holds whatever the pool last held — measured:
  // a fused forward "reproduced" torch at 6.7e-8 in 0.5 ms while its kernels had not
  // compiled. The fault count is the only witness.
  const faults = dev().faults.count - faults0;
  lines.push(`fused (batch norms, relu and the residual add folded into the convolutions): max |logits − torch| ${fusedGap.toExponential(2)}`
    + (faults ? ` · **${faults} validation fault(s) — the numbers below are not measurements**` : ""));
  if (fusedGap > GATE || faults) lines.push("**the fused network does not reproduce torch's logits**");
  for (const b of batches) {
    const data = new Float32Array(b * 3 * 32 * 32);
    for (let i = 0; i < b; i++) data.set(probe.input, i * 3 * 32 * 32);
    const xb = Tensor.from(data, [b, 3, 32, 32]);
    const ms = await timed(() => scope(async () => noGrad(() => model.forward(xb)).toArray()), 3, 20);
    const d = dev();
    const d0 = d.dispatches;
    await noGrad(() => model.forward(xb)).toArray();
    const perForward = d.dispatches - d0;
    // **Where the eager wall goes** (`docs/FIRST.md` 2a): the JavaScript of the forward
    // — encoding, bind groups, tensor bookkeeping — timed on its own (the forward is
    // synchronous; the readback after it is the GPU's), and the bind groups it made.
    let js = 0; const bg0 = d.bindGroups;
    for (let i = 0; i < 10; i++) {
      await scope(async () => {
        const t = performance.now();
        const y = noGrad(() => model.forward(xb));
        js += performance.now() - t;
        await y.toArray();
      });
    }
    lines.push(`batch ${String(b).padStart(3)}  forward  borch.ts fused ${ms.toFixed(2).padStart(8)} ms · ${perForward} dispatches/forward · JavaScript ${(js / 10).toFixed(2)} ms of it (${Math.round((d.bindGroups - bg0) / 10)} bind groups)`);
    // The fused forward's GPU time by kind — what of the wall is the GPU's (INFER Step 0).
    const ph0 = d.packHits, pm0 = d.packMisses;
    await d.profile(() => noGrad(() => model.forward(xb)).toArray());
    const packed = `packs ${d.packHits - ph0} hit / ${d.packMisses - pm0} made`;
    // **The same forward without the pack cache** (a forward outside `noGrad` packs into
    // scope buffers as before the cache): on the 5080 the fused eager kernels read 9×
    // their unfused time with the cache (`docs/FIRST.md` 2a) — this row says whether the
    // kept pack buffers are what is slow.
    let noCacheTotal = 0;
    await scope(async () => {
      await d.profile(() => model.forward(xb).toArray());
      for (const ns of d.nsByKind.values()) noCacheTotal += ns / 1e6;
    });
    await d.profile(() => noGrad(() => model.forward(xb)).toArray());
    const hot: [string, number, number][] = [];
    for (const [kind, ns] of d.nsByKind) hot.push([kind, ns / 1e6, d.countByKind.get(kind) ?? 1]);
    hot.sort((p, q) => q[1] - p[1]);
    const total = hot.reduce((a, [, v]) => a + v, 0);
    lines.push(`           fused GPU time (ms, total ${total.toFixed(1)}, ×count; ${packed}; without the pack cache ${noCacheTotal.toFixed(1)}): ` + hot.slice(0, 8).map(([k, v, n]) => `${k} ${v.toFixed(2)}${n > 1 ? `×${n}` : ""}`).join(" · "));
    // **The eval forward recorded and replayed** (INFER Step 1): `compiled` over the fused
    // network's `noGrad` forward — the intermediates fused, laid into arenas, and the
    // JavaScript that encodes thirty-eight dispatches paid once. The replay's logits are
    // held to the eager fused forward's, bit for bit, before its clock is printed.
    const step = compiled((x: Tensor) => noGrad(() => model.forward(x)));
    const captured = await (await step.call(xb)).toArray();
    const eagerFused = await noGrad(() => model.forward(xb)).toArray();
    const capGap = maxAbsDiff(captured, eagerFused);
    const capMs = await timed(async () => (await step.call(xb)).toArray(), 3, 20);
    const rec = step.recordingOf(xb);
    lines.push(`batch ${String(b).padStart(3)}  forward  borch.ts fused + captured ${capMs.toFixed(2).padStart(8)} ms · ${rec ? rec.dispatches : 0} dispatches/replay · max |replay − eager| ${capGap.toExponential(1)}`
      + (capGap > 0 ? " **— the replay is not the eager forward**" : ""));
    step.dispose();
    // **The int8 forward** (`docs/INT8.md` Step 6) where the adapter has the configuration:
    // `compiled(model, { int8: true })`, its logits against the f32 fused forward's — the
    // number that stands beside its time, since this path is held to accuracy, not to
    // torch. How many layers took int8 is printed too; the rest ran f32.
    if (Device.subgroupInt8) {
      // The model is already folded (its `fuse()` ran for the rows above), so the weights
      // are quantised here directly and the step is the function form.
      const layers = await quantizeForInt8(model);
      const q = compiled((x: Tensor) => noGrad(() => model.forward(x)));
      q.int8Layers = layers;
      const qOut = await (await q.call(xb)).toArray();
      const qGap = maxAbsDiff(qOut, eagerFused);
      let scale = 0; for (const v of eagerFused) scale = Math.max(scale, Math.abs(v));
      const qMs = await timed(async () => (await q.call(xb)).toArray(), 3, 20);
      const qRec = q.recordingOf(xb);
      lines.push(`batch ${String(b).padStart(3)}  forward  borch.ts int8 + captured ${qMs.toFixed(2).padStart(8)} ms · ${qRec ? qRec.dispatches : 0} dispatches/replay · ${q.int8Layers} layers int8 · max |int8 − f32| ${qGap.toExponential(1)} (${(qGap / scale).toExponential(1)} of the logits' scale)`);
      q.dispose();
      // **The static scales**: calibrated on seeded pixels (the scales only have to be
      // finite for a clock; the accuracy section calibrates on real images), then the
      // same forward with the quantise passes gone where a layer feeds a layer.
      const calib = seeded(64);
      await calibrateInt8(model, calib.pixels, 16, [3, 32, 32]);
      const st = compiled((x: Tensor) => noGrad(() => model.forward(x)));
      const stOut = await (await st.call(xb)).toArray();
      const stGap = maxAbsDiff(stOut, eagerFused);
      const stMs = await timed(async () => (await st.call(xb)).toArray(), 3, 20);
      const stRec = st.recordingOf(xb);
      lines.push(`batch ${String(b).padStart(3)}  forward  borch.ts int8 static + captured ${stMs.toFixed(2).padStart(8)} ms · ${stRec ? stRec.dispatches : 0} dispatches/replay · max |int8 − f32| ${stGap.toExponential(1)} (${(stGap / scale).toExponential(1)} of the logits' scale)`);
      st.dispose();
      // The quantisation is a property of the model: back to f32 for the rows after this
      // one (the first run left it on, and the batch-16 f32 rows measured int8).
      clearInt8(model);
    }
  }

  // **Accuracy, where a trained network exists** (`docs/INT8.md` Step 4): the trained
  // weights and a labelled test slice written by `tests/browser/train_resnet18_cifar.py`
  // beside the seed-0 files. The f32 fused forward's top-1 on the slice, then the int8
  // forward's on the same images, and the gate — within half a point — beside the times
  // above. Absent files mean the seed-0 weights only, and the line says so.
  lines.push(...await reportAccuracy());
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


// ── transformer inference: ViT-Tiny/16 in borch.ts (bimm-ts) and in ORT Web ─────────
//
// The ResNet above is convolutions; a transformer forward is batched matmuls, softmax,
// layer norms and GELU over a token row — a different set of kernels, and ORT has ones
// written for each (a fused attention among them). The weights are timm's
// `vit_tiny_patch16_224` at seed 0, exported by `tests/browser/export_vit_tiny.py`; the
// borch side is bimm-ts's `vitTinyPatch16`, which keys its state dict as timm does. The
// page imports bimm (this package does not depend on it, as `cpu.ts` says) and passes the
// factory in.

/** What this file needs of bimm-ts: the ViT-Tiny factory and the module it returns. */
export interface VitModule {
  loadStateDict(table: Record<string, Tensor>, strict: boolean): { missing: string[]; unexpected: string[] };
  eval(): unknown;
  forward(x: Tensor): Tensor;
  alignTokens: boolean;
}
export interface Bimm { vitTinyPatch16(numClasses: number): VitModule }

interface VitProbe extends Probe { timm: string; model: string }

const VIT_CLASSES = 1000;
const VIT_PIXELS = 3 * 224 * 224;

/** GPU time by kind of kernel over one profiled forward, hottest first, with counts. */
function gpuByKind(d: Device): string {
  const hot: [string, number, number][] = [];
  for (const [kind, ns] of d.nsByKind) hot.push([kind, ns / 1e6, d.countByKind.get(kind) ?? 1]);
  hot.sort((p, q) => q[1] - p[1]);
  const total = hot.reduce((a, [, ms]) => a + ms, 0);
  return `GPU time (ms, total ${total.toFixed(1)}, ×count): `
    + hot.slice(0, 16).map(([k, ms, n]) => `${k} ${ms.toFixed(2)}${n > 1 ? `×${n}` : ""}`).join(" · ")
    + (d.profileDropped ? ` · ${d.profileDropped} dropped` : "");
}

export async function reportInferVit(bimm: Bimm, batches: readonly number[] = [1, 16]): Promise<string> {
  const probeBytes = await maybeBytes(OUT + "vit_tiny.probe.json");
  const weights = await maybeBytes(OUT + "vit_tiny.safetensors");
  if (!probeBytes || !weights) return "ViT-Tiny/16: no exported weights (tests/browser/export_vit_tiny.py writes them) — not measured";
  const probe = JSON.parse(new TextDecoder().decode(probeBytes)) as VitProbe;
  const lines: string[] = [`${probe.model} — weights from timm ${probe.timm} / torch ${probe.torch}, exported by tests/browser/export_vit_tiny.py`];

  const model = bimm.vitTinyPatch16(VIT_CLASSES);
  const report = model.loadStateDict(load(weights) as Record<string, Tensor>, true);
  if (report.missing.length || report.unexpected.length) {
    throw new Error(`ViT state dict did not fit: missing ${report.missing.join(",")} unexpected ${report.unexpected.join(",")}`);
  }
  model.eval();
  const x1 = Tensor.from(Float32Array.from(probe.input), probe.shape);
  const ours = await noGrad(() => model.forward(x1)).toArray();
  const oursGap = maxAbsDiff(ours, probe.logits);

  const o = ort();
  o.env.wasm.wasmPaths = "../../vendor/";
  const session = await o.InferenceSession.create(OUT + "vit_tiny.onnx", { executionProviders: ["webgpu"] });
  const feed = (data: Float32Array, b: number) => ({ input: new o.Tensor("float32", data, [b, 3, 224, 224]) });
  const theirs = (await session.run(feed(Float32Array.from(probe.input), 1)))["logits"]?.data ?? new Float32Array();
  const theirsGap = maxAbsDiff(theirs, probe.logits);

  const GATE = 1e-3;
  let scale = 0; for (const v of probe.logits) scale = Math.max(scale, Math.abs(v));
  lines.push(`gate: max |logits − torch| on the probe input — borch.ts ${oursGap.toExponential(2)} · ORT Web ${theirsGap.toExponential(2)} · limit ${GATE} (logits' scale ${scale.toFixed(2)})`
    + ` · tokens ${model.alignTokens && Device.subgroupMatrix ? "padded 197 → 200 for the subgroup matrices" : "197, unpadded"}`);
  if (oursGap > GATE || theirsGap > GATE) {
    lines.push("**a runtime does not reproduce torch's logits — its speed below is a speed of something else**");
  }

  for (const b of batches) {
    const data = new Float32Array(b * VIT_PIXELS);
    for (let i = 0; i < b; i++) data.set(probe.input, i * VIT_PIXELS);
    const xb = Tensor.from(data, [b, 3, 224, 224]);
    // A scope a forward, as the ResNet rows (the allocator's time is not the forward's).
    const oursMs = await timed(() => scope(async () => noGrad(() => model.forward(xb)).toArray()), 3, 20);
    const theirsMs = await timed(() => session.run(feed(data, b)), 3, 20);
    lines.push(`batch ${String(b).padStart(3)}  forward  borch.ts eager ${oursMs.toFixed(2).padStart(8)} ms · ORT Web ${theirsMs.toFixed(2).padStart(8)} ms · ratio ${(oursMs / theirsMs).toFixed(2)}× (borch/ORT)`);
    const d = dev();
    const d0 = d.dispatches;
    await scope(async () => noGrad(() => model.forward(xb)).toArray());
    const dispatches = d.dispatches - d0;
    await scope(() => d.profile(() => noGrad(() => model.forward(xb)).toArray()));
    lines.push(`           borch.ts eager ${dispatches} dispatches/forward · ${gpuByKind(d)}`);
    // The forward recorded and replayed, held to torch's logits like the eager one (the
    // tuner may pick another matmul configuration for the replay, so it is not held to
    // the eager forward bit for bit — the gate is the same for both).
    const step = compiled((x: Tensor) => noGrad(() => model.forward(x)));
    const captured = await (await step.call(xb)).toArray();
    const capGap = maxAbsDiff(captured.subarray(0, VIT_CLASSES), probe.logits);
    const capMs = await timed(async () => (await step.call(xb)).toArray(), 3, 20);
    const rec = step.recordingOf(xb);
    await scope(() => d.profile(async () => (await step.call(xb)).toArray()));
    lines.push(`batch ${String(b).padStart(3)}  forward  borch.ts captured ${capMs.toFixed(2).padStart(8)} ms · ORT Web ${theirsMs.toFixed(2).padStart(8)} ms · ratio ${(capMs / theirsMs).toFixed(2)}× (borch/ORT)`
      + ` · ${rec ? rec.dispatches : 0} dispatches/replay · max |replay − torch| ${capGap.toExponential(1)}` + (capGap > GATE ? " **— the replay does not reproduce torch's logits**" : ""));
    lines.push(`           borch.ts captured ${gpuByKind(d)}`);
    step.dispose();
  }
  return lines.join("\n");
}
