/**
 * Does the ONNX file borch.ts writes **run somewhere else** — and answer the same?
 *
 * The exporter round-tripping through our own reader would prove a format of our own.
 * What `exportOnnx` claims is that ONNX Runtime opens the file and reproduces the
 * forward, so that is what is asked here: the bench's ResNet-18, traced, handed to
 * ORT Web as bytes, run at the traced batch and at another one, and compared to our
 * own forward. Then the same after `fuse()`, where the graph has no batch norms.
 */
import { Tensor, noGrad } from "../src/tensor.js";
import { manualSeed } from "../src/random.js";
import { exportOnnx } from "../src/onnx.js";
import { ResNet18 } from "./bench.js";

interface OrtTensorLike { data: Float32Array }
interface OrtSession { run(feeds: Record<string, unknown>): Promise<Record<string, OrtTensorLike>> }
interface Ort {
  env: { wasm: { wasmPaths: string } };
  InferenceSession: { create(model: Uint8Array, options: { executionProviders: string[] }): Promise<OrtSession> };
  Tensor: new (type: string, data: Float32Array, dims: number[]) => unknown;
}
function ort(): Ort {
  return (globalThis as unknown as { ort: Ort }).ort;
}

export interface Check { name: string; ok: boolean; note: string }

function pixels(batch: number, seed: number): Float32Array {
  // xorshift32 — the same pixels on every machine.
  let s = seed >>> 0;
  const out = new Float32Array(batch * 3 * 32 * 32);
  for (let i = 0; i < out.length; i++) {
    s ^= s << 13; s >>>= 0; s ^= s >>> 17; s ^= s << 5; s >>>= 0;
    out[i] = (s / 4294967296) * 2 - 1;
  }
  return out;
}

function maxAbsDiff(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let worst = 0;
  for (let i = 0; i < a.length; i++) worst = Math.max(worst, Math.abs((a[i] ?? 0) - (b[i] ?? 0)));
  return worst;
}

const GATE = 1e-4;

async function agree(session: OrtSession, model: ResNet18, batch: number, seed: number): Promise<number> {
  const data = pixels(batch, seed);
  const ours = await noGrad(() => model.forward(Tensor.from(data, [batch, 3, 32, 32]))).toArray();
  const o = ort();
  const theirs = (await session.run({ input: new o.Tensor("float32", data, [batch, 3, 32, 32]) }))["output"]?.data
    ?? new Float32Array();
  if (theirs.length !== ours.length) return Infinity;
  return maxAbsDiff(ours, theirs);
}

export async function report(): Promise<{ text: string; checks: Check[] }> {
  const lines: string[] = [];
  const checks: Check[] = [];
  const o = ort();
  o.env.wasm.wasmPaths = "../../vendor/";

  manualSeed(7);
  const model = new ResNet18();
  model.eval();
  // The norms' running statistics are fresh (0 and 1) on an untrained network, where a
  // wrong mean would hide — so one training step moves them first.
  model.train();
  noGrad(() => model.forward(Tensor.from(pixels(8, 99), [8, 3, 32, 32])));
  model.eval();

  const sample = Tensor.from(pixels(2, 1), [2, 3, 32, 32]);
  const t0 = performance.now();
  const plain = await exportOnnx(model, sample);
  const exportMs = performance.now() - t0;
  lines.push(`exported ResNet-18: ${(plain.bytes.length / 1e6).toFixed(1)} MB · ${plain.ops.length} nodes · `
    + `${plain.initializers.length} initializers · ${exportMs.toFixed(0)} ms`);
  const kinds = new Map<string, number>();
  for (const op of plain.ops) kinds.set(op, (kinds.get(op) ?? 0) + 1);
  lines.push("  " + [...kinds].map(([k, n]) => `${k} ×${n}`).join(" · "));

  const session = await o.InferenceSession.create(plain.bytes, { executionProviders: ["webgpu"] });
  checks.push({ name: "ORT Web opens the file", ok: true, note: "" });
  const gap2 = await agree(session, model, 2, 1);
  checks.push({ name: "ORT reproduces the forward at the traced batch (2)", ok: gap2 <= GATE,
                note: `max |Δ| ${gap2.toExponential(2)} against ${GATE}` });
  const gap5 = await agree(session, model, 5, 3);
  checks.push({ name: "and at a batch it was not traced at (5)", ok: gap5 <= GATE,
                note: `max |Δ| ${gap5.toExponential(2)}` });
  lines.push(`ORT Web vs borch.ts: batch 2 max |Δ| ${gap2.toExponential(2)} · batch 5 ${gap5.toExponential(2)}`);

  model.fuse();
  const fused = await exportOnnx(model, sample);
  const session2 = await o.InferenceSession.create(fused.bytes, { executionProviders: ["webgpu"] });
  const gapF = await agree(session2, model, 3, 5);
  checks.push({ name: "the fused network exports without batch norms and agrees", ok: gapF <= GATE && !fused.ops.includes("BatchNormalization"),
                note: `${fused.ops.length} nodes · max |Δ| ${gapF.toExponential(2)}` });
  lines.push(`after fuse(): ${fused.ops.length} nodes (${plain.ops.length} before) · max |Δ| ${gapF.toExponential(2)}`);

  // A refusal names the op rather than writing a file that will not run.
  let refusal = "";
  try {
    const odd = { training: false, eval() { return this; }, train() { return this; },
      namedParameters: () => ({}), namedBuffers: () => ({}),
      forward: (x: Tensor) => x.unary("erf") };
    await exportOnnx(odd as unknown as ResNet18, sample);
  } catch (err) {
    refusal = String(err instanceof Error ? err.message : err);
  }
  checks.push({ name: "an op with no ONNX spelling is refused by name", ok: refusal.includes("cannot export erf"), note: refusal });

  const failed = checks.filter((c) => !c.ok);
  lines.push(failed.length ? `${failed.length} check(s) failed` : `all ${checks.length} ONNX checks passed`);
  return { text: lines.join("\n"), checks };
}
