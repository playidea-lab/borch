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
import { AdaptiveAvgPool2d, Conv2d, Embedding, Flatten, Hardsigmoid, Hardswish, LayerNorm, Linear, MaxPool2d, Module } from "../src/nn.js";
import { ResNet18 } from "./bench.js";

interface OrtTensorLike { data: Float32Array }
interface OrtSession { run(feeds: Record<string, unknown>): Promise<Record<string, OrtTensorLike>> }
interface Ort {
  env: { wasm: { wasmPaths: string } };
  InferenceSession: { create(model: Uint8Array, options: { executionProviders: string[] }): Promise<OrtSession> };
  Tensor: new (type: string, data: Float32Array | BigInt64Array, dims: number[]) => unknown;
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

/** A transformer encoder layer on `[B, T, D]` — the ops a CNN never uses: layer_norm,
 *  the projections as batched `linear`, attention's batched matmuls and softmax, and an
 *  FFN. Every one now traces, so the whole layer exports and ORT reproduces it. */
class EncoderLayer extends Module {
  ln1 = new LayerNorm(8);
  ln2 = new LayerNorm(8);
  wq = new Linear(8, 8);
  wk = new Linear(8, 8);
  wv = new Linear(8, 8);
  wo = new Linear(8, 8);
  fc1 = new Linear(8, 16);
  fc2 = new Linear(16, 8);
  override forward(x: Tensor): Tensor {
    const h = this.ln1.forward(x);
    const q = this.wq.forward(h), k = this.wk.forward(h), v = this.wv.forward(h);  // [B, T, 8]
    const ctx = q.bmm(k.transpose(1, 2)).softmax(-1).bmm(v);                        // [B, T, 8]
    const a = x.add(this.wo.forward(ctx));                                         // residual
    const f = this.fc2.forward(this.fc1.forward(this.ln2.forward(a)).relu());       // FFN
    return a.add(f);
  }
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

  // ── A transformer encoder layer on [B, T, D] — the ops a CNN never uses ──
  // layer_norm, batched `linear` projections, attention's batched matmuls and softmax, an
  // FFN. None was traced until now, so this whole layer exported with them frozen out;
  // now it runs as LayerNormalization / MatMul / Transpose / Softmax / Relu / Add / Gemm
  // and ORT reproduces the forward.
  manualSeed(11);
  const enc = new EncoderLayer();
  enc.eval();
  const B = 2, T = 4, D = 8;
  const adata = pixels(B, 11).subarray(0, B * T * D);
  const asample = Tensor.from(adata, [B, T, D]);
  const encPlan = await exportOnnx(enc, asample);
  lines.push(`exported encoder layer: ${encPlan.ops.length} nodes · ${[...new Set(encPlan.ops)].join(", ")}`);
  const esession = await o.InferenceSession.create(encPlan.bytes, { executionProviders: ["webgpu"] });
  const eours = await noGrad(() => enc.forward(Tensor.from(adata, [B, T, D]))).toArray();
  const etheirs = (await esession.run({ input: new o.Tensor("float32", adata, [B, T, D]) }))["output"]?.data
    ?? new Float32Array();
  const egap = etheirs.length === eours.length ? maxAbsDiff(eours, etheirs) : Infinity;
  const wanted = ["LayerNormalization", "MatMul", "Transpose", "Softmax"];
  const present = wanted.every((k) => encPlan.ops.includes(k));
  checks.push({ name: "ORT reproduces a transformer encoder layer", ok: egap <= GATE && present,
                note: `${encPlan.ops.length} nodes · max |Δ| ${egap.toExponential(2)}` });
  lines.push(`encoder layer ORT vs borch.ts: max |Δ| ${egap.toExponential(2)}`);

  // ── GELU, the exact (erf) form a transformer's FFN uses — one opset-20 `Gelu` node ──
  // The encoder above runs a ReLU FFN; the real one is GELU, and torch's default `gelu`
  // is the erf form, which `Gelu(approximate="none")` is. This exports it and ORT runs it.
  manualSeed(13);
  const gnet = new (class extends Module {
    fc = new Linear(8, 8);
    override forward(x: Tensor): Tensor { return this.fc.forward(x).gelu(); }
  })();
  gnet.eval();
  const gdata = pixels(2, 13).subarray(0, 2 * 8);
  const gsample = Tensor.from(gdata, [2, 8]);
  const gPlan = await exportOnnx(gnet, gsample);
  const gsession = await o.InferenceSession.create(gPlan.bytes, { executionProviders: ["webgpu"] });
  const gours = await noGrad(() => gnet.forward(Tensor.from(gdata, [2, 8]))).toArray();
  const gtheirs = (await gsession.run({ input: new o.Tensor("float32", gdata, [2, 8]) }))["output"]?.data
    ?? new Float32Array();
  const ggap = gtheirs.length === gours.length ? maxAbsDiff(gours, gtheirs) : Infinity;
  checks.push({ name: "ORT reproduces the exact GELU (an opset-20 Gelu node)",
                ok: ggap <= GATE && gPlan.ops.includes("Gelu"),
                note: `${gPlan.ops.join(", ")} · max |Δ| ${ggap.toExponential(2)}` });
  lines.push(`gelu ORT vs borch.ts: ${gPlan.ops.join(", ")} · max |Δ| ${ggap.toExponential(2)}`);

  // ── Embedding — a `Gather` on an int64 input, a text transformer's first layer ──
  // The input is token indices, not floats, so the file declares it int64; ORT is handed a
  // BigInt64Array to match. The table is the weight; one Gather reproduces the lookup.
  manualSeed(17);
  const VOCAB = 16, D2 = 8;
  const enet = new (class extends Module {
    emb = new Embedding(VOCAB, D2);
    override forward(idx: Tensor): Tensor { return this.emb.forward(idx); }
  })();
  enet.eval();
  const tokens = [3, 1, 14, 0, 7, 2, 9, 5];   // [4, 2] of int64 indices < VOCAB
  const idx = Tensor.from(tokens, [4, 2], { dtype: "int64" });
  const embPlan = await exportOnnx(enet, idx);
  const embSession = await o.InferenceSession.create(embPlan.bytes, { executionProviders: ["webgpu"] });
  const embOurs = await noGrad(() => enet.forward(Tensor.from(tokens, [4, 2], { dtype: "int64" }))).toArray();
  const int64Input = new o.Tensor("int64", BigInt64Array.from(tokens.map((t) => BigInt(t))), [4, 2]);
  const embTheirs = (await embSession.run({ input: int64Input }))["output"]?.data ?? new Float32Array();
  const embGap = embTheirs.length === embOurs.length ? maxAbsDiff(embOurs, embTheirs) : Infinity;
  checks.push({ name: "ORT reproduces Embedding as a Gather on an int64 input",
                ok: embGap <= GATE && embPlan.ops.includes("Gather"),
                note: `${embPlan.ops.join(", ")} · max |Δ| ${embGap.toExponential(2)}` });
  lines.push(`embedding ORT vs borch.ts: ${embPlan.ops.join(", ")} · max |Δ| ${embGap.toExponential(2)}`);

  // ── A small CNN whose global pool is `AdaptiveAvgPool2d` on an intermediate ──
  // The workbench's section-4 model. `AdaptiveAvgPool2d` slices its input into bins and
  // averages each — untraced, that slice reached the export as a `SliceBackward0` with no
  // producer and the exporter refused (or, before it refused, froze it wrong). Now the
  // pool is one traced `GlobalAveragePool` node and the whole CNN exports.
  manualSeed(19);
  const cnn = new (class extends Module {
    conv = new Conv2d(3, 4, 3, 1, 1);
    pool = new MaxPool2d(2);
    gap = new AdaptiveAvgPool2d(1);
    flat = new Flatten();
    fc = new Linear(4, 3);
    override forward(x: Tensor): Tensor {
      const h = this.pool.forward(this.conv.forward(x).unary("relu"));
      return this.fc.forward(this.flat.forward(this.gap.forward(h)));
    }
  })();
  cnn.eval();
  const cdata = pixels(1, 19).subarray(0, 1 * 3 * 8 * 8);
  const csample = Tensor.from(cdata, [1, 3, 8, 8]);
  const cPlan = await exportOnnx(cnn, csample);
  const csession = await o.InferenceSession.create(cPlan.bytes, { executionProviders: ["webgpu"] });
  const cours = await noGrad(() => cnn.forward(Tensor.from(cdata, [1, 3, 8, 8]))).toArray();
  const ctheirs = (await csession.run({ input: new o.Tensor("float32", cdata, [1, 3, 8, 8]) }))["output"]?.data
    ?? new Float32Array();
  const cgap = ctheirs.length === cours.length ? maxAbsDiff(cours, ctheirs) : Infinity;
  checks.push({ name: "a small CNN with AdaptiveAvgPool2d on an intermediate exports and agrees",
                ok: cgap <= GATE && cPlan.ops.includes("GlobalAveragePool"),
                note: `${cPlan.ops.join(", ")} · max |Δ| ${cgap.toExponential(2)}` });
  lines.push(`small CNN ORT vs borch.ts: max |Δ| ${cgap.toExponential(2)}`);

  // ── MobileNetV3's activations: HardSwish and HardSigmoid ──
  // ONNX HardSigmoid is `max(0, min(1, alpha·x + beta))` and defaults to alpha=0.2, but
  // torch's (and borch's) is `x/6 + 0.5` clamped — alpha=1/6. Emitted without the
  // attribute, the file loaded and the throw-guard was happy, yet ORT ran a steeper gate
  // and the hub MobileNetV3's features came out ~5 off. The exporter now writes alpha/beta.
  const hdata = pixels(1, 23).subarray(0, 1 * 3 * 8 * 8).map((v) => v * 4);   // span the clip regions
  const act = new (class extends Module {
    hs = new Hardsigmoid();
    hw = new Hardswish();
    override forward(x: Tensor): Tensor { return this.hw.forward(this.hs.forward(x)); }
  })();
  act.eval();
  const hsample = Tensor.from(hdata, [1, 3, 8, 8]);
  const hPlan = await exportOnnx(act, hsample);
  const hsession = await o.InferenceSession.create(hPlan.bytes, { executionProviders: ["webgpu"] });
  const hours = await noGrad(() => act.forward(Tensor.from(hdata, [1, 3, 8, 8]))).toArray();
  const htheirs = (await hsession.run({ input: new o.Tensor("float32", hdata, [1, 3, 8, 8]) }))["output"]?.data
    ?? new Float32Array();
  const hgap = htheirs.length === hours.length ? maxAbsDiff(hours, htheirs) : Infinity;
  checks.push({ name: "ORT reproduces HardSigmoid (alpha=1/6) and HardSwish",
                ok: hgap <= GATE && hPlan.ops.includes("HardSigmoid") && hPlan.ops.includes("HardSwish"),
                note: `${hPlan.ops.join(", ")} · max |Δ| ${hgap.toExponential(2)}` });
  lines.push(`hard activations ORT vs borch.ts: max |Δ| ${hgap.toExponential(2)}`);

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
