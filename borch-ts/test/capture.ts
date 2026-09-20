/**
 * `torch.compiled` in JavaScript, held to the eager step — the binding's `capture:py`
 * for borch.ts (`docs/COMPILER.md` Step 6), and the measurement Step 1 waited on.
 *
 * Two networks. A small MLP over a sequence of batches whose seventh is shorter: the
 * compiled step must record twice (one per shape) and give the eager losses bit for bit
 * under `fuse` and `plan`, and `check: true` must pass on both recordings. Then the
 * ResNet-18 of `bench.ts` at batch 16 — the model the memory prediction was written for:
 * three eager steps against three replays on the same data, bit for bit, and the plan's
 * numbers beside the eager step's: how many intermediates moved, from how many bytes to
 * how many, and the replay's clock against the eager one.
 */
import * as nn from "../src/nn.js";
import { SGD } from "../src/optim.js";
import { compiled } from "../src/compile.js";
import { Device } from "../src/device.js";
import { device, keepAlive, scope, Tensor } from "../src/tensor.js";
import { ResNet18 } from "./bench.js";

interface Check { name: string; ok: boolean; note: string }
export interface Report { text: string; checks: Check[] }

const checks: Check[] = [];
function want(name: string, ok: boolean, note = ""): void {
  checks.push({ name, ok, note });
}

function seeded(seed: number): () => number {
  let s = seed >>> 0;
  return () => { s ^= s << 13; s >>>= 0; s ^= s >>> 17; s ^= s << 5; s >>>= 0; return s / 0x100000000; };
}

function batch(next: () => number, n: number, feats: number, classes: number): { x: Float32Array; y: Float32Array } {
  const x = new Float32Array(n * feats);
  for (let i = 0; i < x.length; i++) x[i] = next() * 2 - 1;
  const y = new Float32Array(n);
  for (let i = 0; i < n; i++) y[i] = Math.floor(next() * classes);
  return { x, y };
}

class MLP extends nn.Module {
  readonly fc1: nn.Linear;
  readonly fc2: nn.Linear;
  constructor(feats: number, hidden: number, classes: number) {
    super();
    this.fc1 = new nn.Linear(feats, hidden);
    this.fc2 = new nn.Linear(hidden, classes);
  }
  override forward(x: Tensor): Tensor {
    return this.fc2.call(this.fc1.call(x).relu());
  }
}

async function mlp(): Promise<void> {
  const FEATS = 32, HIDDEN = 64, CLASSES = 10, STEPS = 14;
  const batches = Array.from({ length: STEPS }, (_, i) => batch(seeded(100 + i), i % 7 === 6 ? 10 : 16, FEATS, CLASSES));
  const make = (): { model: MLP; opt: SGD; crit: nn.CrossEntropyLoss } => {
    nn.manualSeed(7);
    const model = new MLP(FEATS, HIDDEN, CLASSES);
    return { model, opt: new SGD(model.parameters(), 0.05, 0.9), crit: new nn.CrossEntropyLoss() };
  };
  const eager: number[] = [];
  {
    const { model, opt, crit } = make();
    for (const b of batches) {
      eager.push(await scope(async () => {
        opt.zeroGrad();
        const loss = crit.call(model.call(Tensor.from(b.x, [b.y.length, FEATS])), Tensor.from(b.y, [b.y.length], { dtype: "int64" }));
        loss.backward(); opt.step();
        return await loss.item();
      }));
    }
  }
  const replayed: number[] = [];
  const { model, opt, crit } = make();
  const step = compiled((x: Tensor, y: Tensor) => {
    opt.zeroGrad();
    const loss = crit.call(model.call(x), y);
    loss.backward(); opt.step();
    return loss;
  }, { check: true });
  for (const b of batches) {
    replayed.push(await scope(async () => {
      const loss = await step.call(Tensor.from(b.x, [b.y.length, FEATS]), Tensor.from(b.y, [b.y.length], { dtype: "int64" }));
      return await loss.item();
    }));
  }
  const worst = Math.max(...eager.map((e, i) => Math.abs(e - (replayed[i] as number))));
  want("MLP: the compiled losses are the eager losses, bit for bit, over fourteen batches", worst === 0, `max |Δloss| ${worst.toExponential(1)}`);
  want("MLP: two shapes, two recordings", step.shapes === 2, `${step.shapes} recordings`);
  want("MLP: check=true passed on both recordings", step.checked.length === 2 && step.checked.every((c) => c.differ === 0),
    step.checked.map((c) => `${c.buffers} live-ins, ${c.differ} differ`).join(" · "));
  want("MLP: the plan moved intermediates in both recordings", step.planned.length === 2 && step.planned.every((p) => p.moved > 0 && p.bytesAfter <= p.bytesBefore),
    step.planned.map((p) => `${p.moved} moved, ${(p.bytesBefore / 1024).toFixed(0)}K → ${(p.bytesAfter / 1024).toFixed(0)}K`).join(" · "));
  step.dispose();
}

async function resnet(lines: string[]): Promise<void> {
  const BATCH = 16, STEPS = 3, TIMED = 5;
  const data = Array.from({ length: STEPS + TIMED + 2 }, (_, i) => batch(seeded(500 + i), BATCH, 3 * 32 * 32, 10));
  const make = (): { model: ResNet18; opt: SGD; crit: nn.CrossEntropyLoss } => {
    nn.manualSeed(11);
    const model = new ResNet18();
    return { model, opt: new SGD(model.parameters(), 0.05, 0.9), crit: new nn.CrossEntropyLoss() };
  };
  const eagerOne = async (m: { model: ResNet18; opt: SGD; crit: nn.CrossEntropyLoss }, b: { x: Float32Array; y: Float32Array }): Promise<number> =>
    scope(async () => {
      m.opt.zeroGrad();
      const loss = m.crit.call(m.model.call(Tensor.from(b.x, [BATCH, 3, 32, 32])), Tensor.from(b.y, [BATCH], { dtype: "int64" }));
      loss.backward(); m.opt.step();
      return await loss.item();
    });
  const eager: number[] = [];
  const e = make();
  for (let i = 0; i < STEPS; i++) eager.push(await eagerOne(e, data[i] as { x: Float32Array; y: Float32Array }));
  // The eager clock and its footprint, on this model — what the replay is held against.
  const heldEager = device().memory.bytes;
  const pooledEager = device().pooled.bytes;
  let t0 = performance.now();
  for (let i = 0; i < TIMED; i++) await eagerOne(e, data[STEPS + i] as { x: Float32Array; y: Float32Array });
  const eagerMs = (performance.now() - t0) / TIMED;

  const c = make();
  const x = keepAlive(Tensor.from((data[0] as { x: Float32Array }).x, [BATCH, 3, 32, 32]));
  const y = keepAlive(Tensor.from((data[0] as { y: Float32Array }).y, [BATCH], { dtype: "int64" }));
  const step = compiled((xb: Tensor, yb: Tensor) => {
    c.opt.zeroGrad();
    const loss = c.crit.call(c.model.call(xb), yb);
    loss.backward(); c.opt.step();
    return loss;
  });
  const replayed: number[] = [];
  const d0 = device().dispatches;
  for (let i = 0; i < STEPS; i++) {
    const b = data[i] as { x: Float32Array; y: Float32Array };
    replayed.push(await scope(async () => {
      x.copyFrom(Tensor.from(b.x, [BATCH, 3, 32, 32]));
      y.copyFrom(Tensor.from(b.y, [BATCH], { dtype: "int64" }));
      const loss = await step.call(x, y);
      return await loss.item();
    }));
  }
  const worst = Math.max(...eager.map((v, i) => Math.abs(v - (replayed[i] as number))));
  want("ResNet-18: three replays are the three eager steps, bit for bit", worst === 0, `max |Δloss| ${worst.toExponential(1)}`);
  const plan = step.planned[0];
  want("ResNet-18: the plan laid the step's intermediates into fewer bytes", !!plan && plan.moved > 0 && plan.bytesAfter < plan.bytesBefore,
    plan ? `${plan.moved} moved, ${(plan.bytesBefore / 1048576).toFixed(1)} MB → ${(plan.bytesAfter / 1048576).toFixed(1)} MB in ${plan.arenas} arenas` : "no plan");
  t0 = performance.now();
  for (let i = 0; i < TIMED; i++) {
    await scope(async () => {
      const loss = await step.call(x, y);
      return await loss.item();
    });
  }
  const replayMs = (performance.now() - t0) / TIMED;
  const heldCompiled = device().memory.bytes;
  const pooledCompiled = device().pooled.bytes;
  const rec = step.recordingOf(x, y);
  const dispatches = rec ? rec.dispatches : 0;
  lines.push(`ResNet-18 (CIFAR) batch ${BATCH}, SGD 0.05/0.9 — eager ${eagerMs.toFixed(1)} ms/step, compiled replay ${replayMs.toFixed(1)} ms/step (${dispatches} dispatches, a submit each)`);
  lines.push(`  memory: eager held ${(heldEager / 1048576).toFixed(1)} MB (+pool ${(pooledEager / 1048576).toFixed(1)}) · compiled held ${(heldCompiled / 1048576).toFixed(1)} MB (+pool ${(pooledCompiled / 1048576).toFixed(1)})`);
  if (plan) lines.push(`  plan: ${plan.moved} intermediates ${(plan.bytesBefore / 1048576).toFixed(1)} MB → ${(plan.bytesAfter / 1048576).toFixed(1)} MB in ${plan.arenas} arenas, ${plan.released} released`);
  if (rec) {
    await device().profile(async () => { rec.replay(); await (await step.call(x, y)).item(); });
    lines.push("  explain (first eight of the recording, GPU ms per dispatch of its kind):");
    lines.push(rec.explain(device().nsByKind, 8).split("\n").map((l) => "    " + l).join("\n"));
  }
  want("ResNet-18: the replay is not slower than the eager step", replayMs <= eagerMs * 1.05, `eager ${eagerMs.toFixed(1)} · replay ${replayMs.toFixed(1)} ms`);
  step.dispose();
  const d1 = device().dispatches;
  want("ResNet-18: dispatches were counted", d1 > d0, `${d1 - d0} dispatches over the compiled section`);
}

export async function report(): Promise<Report> {
  const lines: string[] = [];
  await mlp();
  await resnet(lines);
  const faults = device().faults.count;
  want("no WebGPU faults", faults === 0, `${faults}`);
  const failed = checks.filter((c) => !c.ok);
  const out = [
    `adapter: ${Device.adapterInfo}`,
    ...lines,
    ...checks.map((c) => `  ${c.ok ? "✓" : "✗"} ${c.name}${c.note ? ` — ${c.note}` : ""}`),
    failed.length === 0 ? `all ${checks.length} compiled-step checks passed` : `**${failed.length} failed** / ${checks.length}`,
  ];
  return { text: out.join("\n"), checks };
}
