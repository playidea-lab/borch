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
import { type CheckReport, compiled } from "../src/compile.js";
import { Device } from "../src/device.js";
import { VERSION } from "../src/version.js";
import { streamTrainStep, type TrainBlock } from "../src/stream_train.js";
import { device, keepAlive, noGrad, scope, Tensor } from "../src/tensor.js";
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

/** `n` seeded values in `(−scale/2, scale/2)`. */
function array(n: number, seed: number, scale: number): Float32Array {
  const next = seeded(seed);
  const a = new Float32Array(n);
  for (let i = 0; i < n; i++) a[i] = (next() - 0.5) * scale;
  return a;
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
  // The tuning a state-writing step owes ran at its second call (`docs/FIRST.md` 1b);
  // `settle` makes sure of it before the decisions are read.
  await step.settle();
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
  // **The tuner's decisions** (`docs/COMPILER.md` Step 5): every choice collected on the
  // first recording, each candidate timed, the fastest kept. Held to "nothing tuned is
  // slower than the rule's pick", and the report says what changed and by how much.
  const tuned = step.tuned.flat();
  const changed = tuned.filter((t) => t.chosen !== t.prior);
  const saved = changed.reduce((a, t) => a + (t.priorMs - t.chosenMs), 0);
  lines.push(`autotune: ${tuned.length} decisions on the first recording, ${changed.length} changed from the rule's pick, ${saved.toFixed(3)} ms of GPU a step saved`
    + (changed.length ? ":\n" + changed.slice(0, 8).map((t) => `    ${t.key.split("|")[1] ?? t.key}: ${t.prior} ${t.priorMs.toFixed(3)} → ${t.chosen} ${t.chosenMs.toFixed(3)} ms`).join("\n") : ""));
  // **The compile wave, by pipeline** (`docs/FIRST.md` 1a): which candidates' shaders
  // cost the first load, with their WGSL size — the number a first visit on D3D12 pays.
  const comp = [...device().tuneCompiles].sort((p, q) => q.ms - p.ms);
  if (comp.length) lines.push(`compile wave: ${comp.length} pipelines, ${comp.reduce((a, c) => a + c.ms, 0).toFixed(0)} ms summed (overlapping), ${(comp.reduce((a, c) => a + c.bytes, 0) / 1024).toFixed(0)} KB of WGSL — dearest: `
    + comp.slice(0, 8).map((c) => `${c.sig.split(":").slice(0, 2).join(":")} ${c.ms.toFixed(0)} ms/${(c.bytes / 1024).toFixed(0)}K`).join(" · "));
  want("autotune: no tuned decision is slower than the rule's pick", tuned.every((t) => t.chosenMs <= t.priorMs), `${tuned.length} decisions`);
  want("autotune: decisions were collected where the device can time", !Device.canTime || tuned.length > 0, `${tuned.length} decisions · timestamps ${Device.canTime}`);
  // **What the first call paid** (the Step 5 gate's other number): the recording, the
  // wait for the step's own first run (paid here rather than at the caller's read), the
  // tuning pass, the re-recording — against the same step recorded with the tuner off,
  // on its own model, so the cost is the tuner's and not the recording's. The first
  // measurement (metal-3, 2026-09-21) read 45–50 ms of tuning for ten candidates, a
  // profiled round trip per candidate per round with the shader compiles inside the
  // timed rounds; `runTuning` now warms every candidate once and profiles a round as one
  // pass, and the prediction for that is 25–35 ms — the GPU time of the repetitions
  // plus one compile wave, against the plan's bound of 2 ms a candidate.
  const cost = step.firstCall[0];
  const u = make();
  const untuned = compiled((xb: Tensor, yb: Tensor) => {
    u.opt.zeroGrad();
    const loss = u.crit.call(u.model.call(xb), yb);
    loss.backward(); u.opt.step();
    return loss;
  }, { tune: false });
  await scope(async () => (await untuned.call(x, y)).item());
  await untuned.settle();
  const plain = untuned.firstCall[0];
  untuned.dispose();
  if (cost && plain) {
    const perCand = cost.candidates ? cost.tuning / cost.candidates : 0;
    lines.push(`autotune overhead: first call ${cost.record.toFixed(0)} ms (the recording; the tuning is ${cost.deferred ? "owed to the second call" : "not deferred"}) · then the step's own first run ${cost.wait.toFixed(0)} + tuning ${cost.tuning.toFixed(0)} (of which the candidates' pipelines compiling ${cost.compileWave.toFixed(0)}; ${cost.candidates} candidates, ${perCand.toFixed(1)} ms each) + re-record ${cost.rerecord.toFixed(0)} · the same step recorded with tune: false ${plain.record.toFixed(0)} ms · a replay ${replayMs.toFixed(1)} ms`);
  }
  // **The plan's bound (2 ms a candidate) and two after it (three replays of the step,
  // with and without the compile wave) were all guesses the measurement refused**: the
  // pass is one compile wave — the platform's, 19 ms on metal-3, 69 on the 5080's
  // Vulkan, ~900 on the laptop's D3D12 with the pipelines made side by side — plus ten
  // repetitions of every candidate's GPU time, 11 / 39 / ~350 ms on the same three. What
  // a user must be protected from is not that number but paying it again (Burn's
  // autotune, 262 ms on every step): the decisions are cached by adapter and key, so a
  // second recording of the same step times nothing. That is the gate.
  const again = make();
  const cached = compiled((xb: Tensor, yb: Tensor) => {
    again.opt.zeroGrad();
    const loss = again.crit.call(again.model.call(xb), yb);
    loss.backward(); again.opt.step();
    return loss;
  });
  await scope(async () => (await cached.call(x, y)).item());
  await cached.settle();
  const second = cached.firstCall[0];
  cached.dispose();
  want("autotune: a second recording of the same step times nothing (the decisions are cached by adapter and key)",
    !!second && second.candidates === 0 && second.compileWave === 0, second ? `${second.candidates} candidates, no compile wave, ${second.tuning.toFixed(1)} ms of readback and passes` : "no first call");
  step.dispose();
  const d1 = device().dispatches;
  want("ResNet-18: dispatches were counted", d1 > d0, `${d1 - d0} dispatches over the compiled section`);
}

/**
 * A LoRA chain over a streamed frozen base — `streamTrainStep` under `compiled`
 * (`docs/COMPILER.md` Step 7). The base weights live on the host and pass through a window
 * two slots wide; the recording keeps each placement as a refill and replays through
 * `replayAsync`. Three eager streamed steps against one recording replayed twice, on the
 * same initial adapters: the losses and the adapters bit for bit, and the window never
 * wider than it was eagerly.
 */
async function streamed(lines: string[]): Promise<void> {
  const D = 32, BLOCKS = 4, R = 4, BATCH = 8, STEPS = 3;
  const bases = Array.from({ length: BLOCKS }, (_, i) => array(D * D, 1000 + i, 0.5));
  const inputs = Array.from({ length: STEPS }, (_, s) => array(BATCH * D, 42 + s, 2));
  const make = (): { blocks: TrainBlock[]; params: Tensor[] } => {
    const params: Tensor[] = [];
    const blocks = bases.map((w, i) => {
      const a = keepAlive(Tensor.from(array(R * D, 2000 + i, 0.3), [R, D])); a.requiresGrad = true;
      const b = keepAlive(Tensor.from(array(D * R, 3000 + i, 0.3), [D, R])); b.requiresGrad = true;
      params.push(a, b);
      const blk: TrainBlock = {
        weights: [w], shapes: [[D, D]], params: [a, b],
        run: (h, windowed) => h.matmul(windowed[0] as Tensor).add(h.matmul(a.transpose(0, 1)).matmul(b.transpose(0, 1))).relu(),
      };
      return blk;
    });
    return { blocks, params };
  };
  const loss = (o: Tensor): Tensor => o.mul(o).mean();
  const slotBytes = D * D * 4;

  const eager: number[] = [];
  const e = make();
  const winE = device().window(2 * slotBytes);
  const optE = new SGD(e.params, 0.05, 0.9);
  for (let s = 0; s < STEPS; s++) {
    eager.push(await scope(async () => {
      optE.zeroGrad();
      const l = await streamTrainStep(winE, Tensor.from(inputs[s] as Float32Array, [BATCH, D]), e.blocks, loss);
      optE.step();
      return await l.item();
    }));
  }
  const eagerParams = await Promise.all(e.params.map((p) => p.toArray()));

  const c = make();
  const winC = device().window(2 * slotBytes);
  const optC = new SGD(c.params, 0.05, 0.9);
  const step = compiled(async (x: Tensor) => {
    optC.zeroGrad();
    const l = await streamTrainStep(winC, x, c.blocks, loss);
    optC.step();
    return l;
  }, { check: true });
  const replayed: number[] = [];
  for (let s = 0; s < STEPS; s++) {
    replayed.push(await scope(async () => {
      const l = await step.call(Tensor.from(inputs[s] as Float32Array, [BATCH, D]));
      return await l.item();
    }));
  }
  const compiledParams = await Promise.all(c.params.map((p) => p.toArray()));
  const worstLoss = Math.max(...eager.map((v, i) => Math.abs(v - (replayed[i] as number))));
  let worstParam = 0;
  eagerParams.forEach((a, i) => { const b = compiledParams[i] as Float32Array; for (let k = 0; k < a.length; k++) worstParam = Math.max(worstParam, Math.abs((a[k] as number) - (b[k] as number))); });
  const rec = step.recordingOf(Tensor.from(inputs[0] as Float32Array, [BATCH, D]));
  const cov = rec ? rec.coverage() : null;
  want("streamed: one recording, replayed twice, gives the eager streamed losses bit for bit", worstLoss === 0 && step.shapes === 1, `max |Δloss| ${worstLoss.toExponential(1)}, ${step.shapes} recording`);
  want("streamed: the adapters agree bit for bit after three steps", worstParam === 0, `max |Δparam| ${worstParam.toExponential(1)}`);
  want("streamed: the recording carries a refill per placement, forward and backward", !!cov && cov.refills === 2 * BLOCKS, cov ? `${cov.refills} refills` : "no recording");
  // Every dispatch of the fused recording says what it touches — a fused kernel included
  // (nine a step were "guessed" before fuse.ts carried the pipeline's access sets).
  want("streamed: no dispatch of the fused recording is guessed at", !!cov && cov.guessed === 0, cov ? `${cov.guessed} guessed of ${cov.dispatches}` : "no recording");
  want("streamed: check=true passed on the streamed recording", step.checked.length === 1 && (step.checked[0] as CheckReport).differ === 0, step.checked.map((r) => `${r.buffers} live-ins, ${r.differ} differ`).join(""));
  want("streamed: the window stayed two slots wide under the recording", winC.used <= 2 * slotBytes, `${winC.used} of ${2 * slotBytes} bytes used`);
  const plan = step.planned[0];
  if (plan) lines.push(`streamed LoRA chain (${BLOCKS} blocks, window ${2 * slotBytes / 1024} KB): plan ${plan.moved} intermediates ${(plan.bytesBefore / 1024).toFixed(0)}K → ${(plan.bytesAfter / 1024).toFixed(0)}K in ${plan.arenas} arenas · ${cov?.refills ?? 0} refills a step`);
  step.dispose();
  winE.free(); winC.free();
}

const maxAbs = (a: Float32Array, b: Float32Array): number => {
  let m = 0;
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs((a[i] as number) - (b[i] as number)));
  return m;
};

/**
 * `compiled(model)` — the inference form (`docs/INFER.md` Step 6): eval, the folds, the
 * `noGrad` forward recorded. Held two ways: a model with its own `fuse()` (the ResNet-18)
 * gives the hand-fused eager forward bit for bit; a plain `Sequential` is folded by the
 * generic pass — `Conv → BN → ReLU → Conv → BN` becomes `ConvReLU2d → Conv2d` — and gives
 * the unfused eval forward to a rounding in fewer dispatches.
 */
/**
 * **Tuning in idle time leaves the caller's scopes alone** (the 2026-09-24 review). The
 * tuning pass awaited compiles and timings inside one `scope`, and a scope's close pops
 * whatever frame is on top — so a scope the page opened while the tuning waited was the
 * one closed, its tensors pooled under it. And a tuning that fails falls back to the
 * rule's kernels: the step still answers.
 */
async function background(): Promise<void> {
  const dv = device() as unknown as { scopes: Set<GPUBuffer>[]; profiling: boolean };
  // Shapes nothing else on this page has tuned, so the pass has candidates to time.
  nn.manualSeed(4);
  const net = new nn.Sequential(new nn.Conv2d(3, 24, 3, 1, 1), new nn.ReLU(), new nn.Conv2d(24, 40, 3, 1, 1));
  const xi = Tensor.from(array(4 * 3 * 20 * 20, 55, 2), [4, 3, 20, 20]);
  const step = compiled(net);
  await (await step.call(xi)).toArray();
  const depth0 = dv.scopes.length;
  const tuning = step.settle();
  // And the timestamps: a tuning pass that kept profiling on across its waits timed the
  // page's own dispatches in that window into the candidates' numbers.
  let heldAcross = false, profiledAcross = false;
  for (let t = 0; t < 400 && !heldAcross; t++) {
    await new Promise((r) => setTimeout(r, 2));
    if (dv.scopes.length > depth0) heldAcross = true;
    if (dv.profiling) profiledAcross = true;
  }
  let stillMine = false, intact = false, refused = "";
  const data = array(1024, 66, 1);
  await scope(async () => {
    const mine = Tensor.from(data, [1024]);
    await tuning;                                   // the tuning ends while this scope is open
    try {
      stillMine = dv.scopes[dv.scopes.length - 1]?.has(mine.buffer) ?? false;
      intact = maxAbs(await mine.toArray(), data) === 0;
    } catch (err) {
      refused = String(err).split("\n")[0] ?? "";
    }
  });
  // **A decision is this adapter's, in this browser, for this build** (2026-09-24 review):
  // keyed on the adapter alone it outlived a browser update that changed the compiler and
  // a library update that changed the candidates, in `localStorage`, for good.
  const browser = /(?:Chrome|Firefox|Version)\/\d+/.exec(globalThis.navigator?.userAgent ?? "")?.[0] ?? "";
  const keys = [...Device.tune.keys()];
  want("tuning decisions are keyed by adapter, browser version and library version",
    keys.length > 0 && keys.every((k) => k.includes(`|${browser}|`) && k.includes(`|${VERSION}|`)),
    `${keys.length} decisions · ${keys[0]?.split("|").slice(0, 4).join(" | ") ?? "none"}`);
  want("compiled: tuning in idle time times nothing but its own candidates — profiling is never left on across an await",
    !profiledAcross, profiledAcross ? "profiling was on while the tuning waited" : "off at every await");
  want("compiled: tuning in idle time leaves a scope the page opened meanwhile its own",
    stillMine && intact && dv.scopes.length === depth0,
    `${heldAcross ? "the tuning held a scope across its awaits" : "no scope held across an await"} · the page's frame ${stillMine ? "kept" : "TAKEN"}${refused ? ` (${refused.slice(0, 90)})` : ""} · depth ${depth0} → ${dv.scopes.length}`);
  step.dispose();
}

/**
 * **`check: true` with `fuse: false` on a training step the tuner changes** (the 2026-09-24
 * review). A state-writing step's check runs after its tuning, against an eager rerun; the
 * recording holds the rule's kernels and the rerun took the tuner's, and with no fusion the
 * tolerance is none — the check refused a correct step. A batch of 8 is shapes nothing on
 * this page has tuned, so the step is tuned here.
 */
async function checkedTraining(): Promise<void> {
  const BATCH = 8;
  nn.manualSeed(12);
  const model = new ResNet18();
  const opt = new SGD(model.parameters(), 0.05, 0.9);
  const crit = new nn.CrossEntropyLoss();
  const b = batch(seeded(900), BATCH, 3 * 32 * 32, 10);
  const x = keepAlive(Tensor.from(b.x, [BATCH, 3, 32, 32]));
  const y = keepAlive(Tensor.from(b.y, [BATCH], { dtype: "int64" }));
  const step = compiled((xb: Tensor, yb: Tensor) => {
    opt.zeroGrad();
    const loss = crit.call(model.call(xb), yb);
    loss.backward(); opt.step();
    return loss;
  }, { check: true, fuse: false });
  let refused = "";
  try {
    for (let i = 0; i < 3; i++) await scope(async () => { await (await step.call(x, y)).item(); });
  } catch (err) {
    refused = String(err).split("\n")[0] ?? "";
  }
  const changed = step.tuned.flat().filter((t) => t.chosen !== t.prior).length;
  want("compiled(check, fuse: false): a training step the tuner changed passes its own check",
    refused === "" && step.checked.every((c) => c.differ === 0),
    `${changed} decision(s) changed · ${refused ? refused.slice(0, 100) : `${step.checked.length} checks`}`);
  step.dispose();
}

async function inference(lines: string[]): Promise<void> {
  const x = Tensor.from(array(16 * 3 * 32 * 32, 77, 2), [16, 3, 32, 32]);
  nn.manualSeed(3);
  const byHand = new ResNet18();
  nn.manualSeed(3);
  const byCall = new ResNet18();
  byHand.eval();
  byHand.fuse();
  const step = compiled(byCall);
  const first = await step.call(x);
  const got = await first.toArray();
  // A pure step tunes in idle time on a throwaway recording; settled here so the second
  // call replays the chosen kernels and the numbers below are its.
  await step.settle();
  const second = await step.call(x);
  const again = await second.toArray();
  // **The object the first call returned is the one every later call returns** — the
  // contract on `compiled`. Where the tuner changed a decision the step is recorded again
  // and swapped in; that swap used to hand back new objects and pool the old ones, so a
  // caller who kept the first read a released buffer (2026-09-24 review).
  let heldReads = "";
  try { heldReads = maxAbs(await first.toArray(), again) === 0 ? "reads the latest step" : "reads other values"; } catch (err) { heldReads = String(err).split("\n")[0] ?? ""; }
  const swapped = step.tuned.flat().some((t) => t.chosen !== t.prior);
  want("compiled(model): the object the first call returned is the one later calls return, and it reads the latest step",
    first === second && heldReads === "reads the latest step",
    `${swapped ? "re-recorded after a changed decision" : "no decision changed here — no swap to test"} · same object ${first === second} · the first ${heldReads.slice(0, 90)}`);
  const rec = step.recordingOf(x);
  // The eager reference after the first call: the tuner may have decided a kernel on
  // that call (`docs/COMPILER.md` Step 5), and eager then takes the same decision — the
  // contract is eager and replay bit for bit, not the rule's kernels and the tuned ones,
  // which differ by a rounding.
  const ref = await noGrad(() => byHand.forward(x)).toArray();
  // The first call answered with the rule's kernels before the tuner ran (`docs/FIRST.md`
  // 1b), so it may differ from the settled forward by a rounding; every call after the
  // settle replays the chosen kernels, and eager makes the same decision — bit for bit.
  let refScale = 0; for (const v of ref) refScale = Math.max(refScale, Math.abs(v));
  want("compiled(model) on a model with its own fuse() is the hand-fused eval forward, bit for bit once settled",
    maxAbs(ref, again) === 0 && maxAbs(ref, got) / refScale < 1e-5, `max |Δ| settled ${maxAbs(ref, again).toExponential(1)} · first call ${(maxAbs(ref, got) / refScale).toExponential(1)} rel · ${rec ? rec.dispatches : 0} dispatches a replay`);
  // **The eager pack cache** (`Device.packed`): the fused eval forward's deep layers read
  // the weight tap-major, and an eager forward repacked it every call. Under `noGrad` the
  // pack is kept beside the weight's write epoch — a second forward repacks nothing,
  // and a weight written in place (an in-place multiply, as an optimiser would) is
  // repacked on the next. The reference forward above made the packs.
  const dv = device();
  const m0 = dv.packMisses, h0 = dv.packHits;
  await scope(async () => noGrad(() => byHand.forward(x)).toArray());
  const missesAgain = dv.packMisses - m0, hitsAgain = dv.packHits - h0;
  want("eager pack cache: a second noGrad forward of the fused network repacks no weight", missesAgain === 0, `${missesAgain} repacks, ${hitsAgain} packs reused`);
  // Every convolution weight written in place: which layers pack is the tuner's to
  // decide per adapter (on the 5080 it sent the 512 → 512 layers to the tiled GEMM,
  // which reads the weight as it is — a test that picked one weight found no pack), so
  // the count that must come back is the count that was reused.
  noGrad(() => { for (const p of byHand.parameters()) if (p.shape.length === 4) p.mul_(1); });
  const m1 = dv.packMisses;
  await scope(async () => noGrad(() => byHand.forward(x)).toArray());
  want("eager pack cache: the weights written in place are repacked on the next forward, and only those", dv.packMisses - m1 === hitsAgain, `${dv.packMisses - m1} repacks after the write, ${hitsAgain} packs in use`);
  // **The compiled model itself, eagerly** — the case the byHand reference never ran.
  // The first call's dry run made packs that nothing wrote, and they were cached: this
  // forward read zeros or a recycled buffer's bytes (2026-09-24 review).
  const onItself = await scope(async () => noGrad(() => byCall.forward(x)).toArray());
  want("compiled(model): an eager forward of the same model afterwards is the eval forward, bit for bit",
    maxAbs(ref, onItself) === 0, `max |Δ| ${maxAbs(ref, onItself).toExponential(1)}`);
  const fc = step.firstCall[0];
  if (fc) lines.push(`compiled(model) first call: recording ${fc.record.toFixed(0)} ms (its kernels compiled side by side; the answer is out) · then in idle time: tuning ${fc.tuning.toFixed(0)} (compile wave ${fc.compileWave.toFixed(0)}; ${fc.candidates} candidates) + re-record ${fc.rerecord.toFixed(0)} ms${step.tuned.flat().some((t) => t.chosen !== t.prior) ? " (a decision changed; the pure forward was recorded again)" : ""}`);
  step.dispose();

  const xs = Tensor.from(array(8 * 3 * 16 * 16, 78, 2), [8, 3, 16, 16]);
  nn.manualSeed(5);
  const seq = new nn.Sequential(
    new nn.Conv2d(3, 8, 3, 1, 1), new nn.BatchNorm2d(8), new nn.ReLU(),
    new nn.Conv2d(8, 8, 3, 1, 1), new nn.BatchNorm2d(8));
  // Running statistics worth folding: a few training-mode forwards move them off 0 / 1.
  for (let i = 0; i < 3; i++) await scope(async () => { await seq.forward(Tensor.from(array(8 * 3 * 16 * 16, 90 + i, 2), [8, 3, 16, 16])).toArray(); });
  seq.eval();
  const refSeq = await noGrad(() => seq.forward(xs)).toArray();
  const d0 = device().dispatches;
  await noGrad(() => seq.forward(xs)).toArray();
  const eagerDispatches = device().dispatches - d0;
  const cs = compiled(seq);
  const gotSeq = await (await cs.call(xs)).toArray();
  // Relative to the output's scale, not per element: a fold that rounds once where the
  // eager path rounded twice differs by 1e-7 on a value near zero, which per element is
  // not a rounding but a ratio of two small numbers (measured: 7e-4 that way, 0 this way).
  let scale = 0;
  for (let i = 0; i < refSeq.length; i++) scale = Math.max(scale, Math.abs(refSeq[i] as number));
  const rel = maxAbs(refSeq, gotSeq) / (scale + 1e-6);
  const recSeq = cs.recordingOf(xs);
  const kinds = seq.children().map((c) => c.constructor.name).join(" → ");
  want("compiled(model) folds a Sequential's Conv→BN→ReLU into ConvReLU2d and gives the eval forward to a rounding",
    rel <= 1e-5 && kinds === "ConvReLU2d → Conv2d", `rel ${rel.toExponential(1)} · ${kinds}`);
  want("the folded Sequential replays in fewer dispatches than its eager eval forward",
    !!recSeq && recSeq.dispatches < eagerDispatches, `${recSeq ? recSeq.dispatches : 0} a replay against ${eagerDispatches} eager`);
  lines.push(`compiled(model): ResNet-18 ${rec ? rec.dispatches : 0} dispatches a replay · Sequential ${kinds}, ${recSeq ? recSeq.dispatches : 0} against ${eagerDispatches} eager`);
  cs.dispose();
}

export async function report(): Promise<Report> {
  const lines: string[] = [];
  await mlp();
  await resnet(lines);
  await streamed(lines);
  await inference(lines);
  await background();
  await checkedTraining();
  const faults = device().faults.count;
  want("no WebGPU faults", faults === 0, `${faults}`);
  const failed = checks.filter((c) => !c.ok);
  const out = [
    `adapter: ${Device.adapterInfo} · ${Device.readbackNote}`,
    ...lines,
    ...checks.map((c) => `  ${c.ok ? "✓" : "✗"} ${c.name}${c.note ? ` — ${c.note}` : ""}`),
    failed.length === 0 ? `all ${checks.length} compiled-step checks passed` : `**${failed.length} failed** / ${checks.length}`,
  ];
  return { text: out.join("\n"), checks };
}
