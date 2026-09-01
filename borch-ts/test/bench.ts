/**
 * A **real training step**, measured on ResNet-18 (the CIFAR build).
 *
 * **The same yardstick** as `tests/browser/bench.py` — same model, same optimizer, same
 * batch sizes, same arithmetic. A different yardstick makes two different numbers rather
 * than a comparison.
 *
 * A figure taken from one layer does not count the bandwidth-bound operations — BN, ReLU,
 * the residual add. "Minutes per epoch" comes out of a real step that includes them, and
 * not out of an estimate divided from FLOPs.
 */

import * as nn from "../src/nn.js";
import { SGD } from "../src/optim.js";
import { Device } from "../src/device.js";
import { device, keepAlive, noGrad, scope, Tensor } from "../src/tensor.js";

/** How many images the epoch time is figured from. It has to be the Python bench's
 * number for the two to compare. */
const CIFAR_TRAIN_IMAGES = 50000;

/** ResNet's basic block. The 1×1 is there only when the shortcut has to change shape. */
export class Block extends nn.Module {
  private readonly conv1: nn.Conv2d;
  private readonly bn1: nn.BatchNormND;
  private readonly conv2: nn.Conv2d;
  private readonly bn2: nn.BatchNormND;
  /**
   * **Pulled out into fields.** These used to be a plain `{ conv, bn }` object written
   * into `children()` alone — and `namedChildren()` sweeps only fields that are
   * `instanceof Module`, so it never saw the two, and **six shortcut layers never learned
   * once.** The loss went down, because the rest compensate.
   *
   * torch does not register a layer put in a Python dict either (which is why
   * `nn.ModuleDict` exists). The library was right and this file was wrong.
   */
  private readonly downConv: nn.Conv2d | null;
  private readonly downBn: nn.BatchNormND | null;

  constructor(cin: number, cout: number, stride: number) {
    super();
    // **`bias` is eighth now**, where torch has it — `dilation` and `groups` took
    // the two seats in front of it. `tsc` named every one of these six call sites
    // the moment the constructor moved, which Python had nothing to do.
    this.conv1 = new nn.Conv2d(cin, cout, 3, stride, 1, 1, 1, false);
    this.bn1 = new nn.BatchNormND(cout);
    this.conv2 = new nn.Conv2d(cout, cout, 3, 1, 1, 1, 1, false);
    this.bn2 = new nn.BatchNormND(cout);
    const shrinks = stride !== 1 || cin !== cout;
    this.downConv = shrinks
      ? new nn.Conv2d(cin, cout, 1, stride, 0, 1, 1, false)
      : null;
    this.downBn = shrinks ? new nn.BatchNormND(cout) : null;
  }

  override forward(x: Tensor): Tensor {
    let out = this.bn1.forward(this.conv1.forward(x)).unary("relu");
    out = this.bn2.forward(this.conv2.forward(out));
    const side = this.downConv && this.downBn
      ? this.downBn.forward(this.downConv.forward(x))
      : x;
    return out.add(side).unary("relu");
  }
}

/**
 * The CIFAR build — a 3×3 stem and no max pool. Take 32×32 through a 7×7 stem and too
 * little is left.
 */
export class ResNet18 extends nn.Module {
  private readonly stem: nn.Conv2d;
  private readonly bn: nn.BatchNormND;
  private readonly body: nn.Sequential;
  private readonly fc: nn.Linear;

  constructor(classes = 10) {
    super();
    this.stem = new nn.Conv2d(3, 64, 3, 1, 1, 1, 1, false);
    this.bn = new nn.BatchNormND(64);
    this.body = new nn.Sequential([
      new Block(64, 64, 1), new Block(64, 64, 1),
      new Block(64, 128, 2), new Block(128, 128, 1),
      new Block(128, 256, 2), new Block(256, 256, 1),
      new Block(256, 512, 2), new Block(512, 512, 1),
    ]);
    this.fc = new nn.Linear(512, classes);
  }

  // `children()` is not overridden — all four are fields, so the default sweep finds
  // them. Overriding it opens a place to disagree with `namedChildren()`, and that
  // disagreement is what kept `Block` above from learning for six layers.

  override forward(x: Tensor): Tensor {
    let h = this.bn.forward(this.stem.forward(x)).unary("relu");
    h = this.body.forward(h).adaptiveAvgPool(1);
    return this.fc.forward(h.reshape([h.shape[0] ?? 1, 512]));
  }
}

export interface StepResult {
  batch: number;
  params: number;
  msPerStep: number;
  epochMin: number;
  /** Buffers per step that were not let go. Anything but 0 is a leak. */
  leakPerStep: number;
  /** Dispatches per step. It separates a slow kernel from too many calls. */
  dispatches: number;
  /** Wall clock per dispatch. Not kernel time — an upper bound on **what a call
   * costs.** */
  usPerDispatch: number;
  /** Dispatches by kind of kernel, most first. What to mend next comes out of here. */
  kinds: [string, number][];
  /** **GPU time** (ms) by kind of kernel, largest first. The point is that the order
   * differs from the count's. */
  hot: [string, number][];
  /** The total measured while profiling (ms). Larger than usual, because a pass is
   * opened per dispatch. */
  profiledMs: number;
  /** Dispatches **not measured**, for want of a query slot. Anything but 0 and the
   * table above is part of a step rather than all of it. */
  profileDropped: number;
  /**
   * Buffers waiting in the pool for the next step (MB). **`gpuMb` does not count this.**
   *
   * That one asks "is it leaking" and this one "is it being held". The batch size changes
   * within one sitting, so the previous batch's buffers pile up here — leave it unsaid and
   * the footprint looks smaller than it is.
   */
  pooledMb: number;
  /** Actual submits per step. How far it parts from the dispatch count is how much was
   * batched. */
  submits: number;
  /**
   * GPU memory held after the step (MB).
   *
   * **The Python bench measures this and this side did not.** Call it the same yardstick
   * while a column is read on one side only, and whatever parts in that column does not
   * appear in the comparison.
   */
  gpuMb: number;
  /** Wall clock with the forward alone. The rest is backward and optimizer. */
  msForward: number;
  /** A step with the weight gradient off. Subtract from the whole and what is left is
   * `gradWeight`'s share. */
  msNoWeightGrad: number;
  /** The dispatch count then. If it did not fall, **the flag did not bite.** */
  noWeightDispatches: number;
  loss: number;
}

/**
 * The wall clock of one step.
 *
 * **The baseline is taken after the warm-up.** Otherwise the first step's shader
 * compilation falls inside the measuring window and passes itself off as per-step cost —
 * and a design that bakes a shader per shape makes that a large figure.
 */
export async function runStep(
  batch = 32, steps = 5, warmup = 2,
): Promise<StepResult> {
  // The input is made **outside the scope.** Made inside, it is let go at the end of the
  // first step.
  const rng = { s: 12345 };
  const next = (): number => {
    let x = rng.s; x ^= x << 13; x >>>= 0; x ^= x >> 17; x ^= x << 5; x >>>= 0;
    rng.s = x;
    return x / 0x100000000;
  };
  const pixels = new Float32Array(batch * 3 * 32 * 32);
  for (let i = 0; i < pixels.length; i++) pixels[i] = next() * 2 - 1;
  const labels = new Float32Array(batch);
  for (let i = 0; i < batch; i++) labels[i] = Math.floor(next() * 10);
  const x = keepAlive(Tensor.from(pixels, [batch, 3, 32, 32]));
  const y = keepAlive(Tensor.from(labels, [batch], { dtype: "int64" }));

  const model = new ResNet18();
  const params = model.parameters();
  const count = params.reduce((n, p) => n + p.size, 0);
  const opt = new SGD(params, 0.05, 0.9);
  const crit = new nn.CrossEntropyLoss();

  // **Typed exactly as a user would type it.** The bench is the only training loop
  // outsiders read, so calling the low-level `beginScope`/`endScope` here would make
  // recommending `scope()` meaningless. The leak is read from `device().lastScope` — which
  // is what used to make this place reach for the low-level pair.
  const one = async (): Promise<number> => scope(async () => {
    opt.zeroGrad();
    const out = crit.call(model.call(x), y);
    out.backward();
    opt.step();
    // **It has to be read inside the scope** — outside it, that buffer is gone.
    return await out.item();
  });

  for (let i = 0; i < warmup; i++) await one();

  const t0 = performance.now();
  const d0 = device().dispatches;
  const s0 = device().submits;
  const k0 = new Map(device().byKind);
  let last = 0;
  for (let i = 0; i < steps; i++) last = await one();
  const perStep = (performance.now() - t0) / steps;
  const perStepDispatches = (device().dispatches - d0) / steps;
  const perStepSubmits = (device().submits - s0) / steps;
  // Split by kind — a total alone does not say what to mend next.
  const kinds: [string, number][] = [];
  for (const [kind, n] of device().byKind) {
    const grew = (n - (k0.get(kind) ?? 0)) / steps;
    if (grew > 0) kinds.push([kind, Math.round(grew)]);
  }
  kinds.sort((a, b) => b[1] - a[1]);

  // A leak is **the number of buffers a scope let out without letting go.**
  //
  // At first this subtracted the depth of the open scopes, which outside a scope is always
  // 0, so it always gave 0 — an instrument named for a leak and measuring nothing. The
  // Python bench had written the same trap down ("an instrument that reports a number
  // other than what its name says cannot be believed when it does meet a real leak") and I
  // read that and built the same thing anyway.
  await one();
  const leak = device().lastScope.survived;

  /**
   * The forward alone.
   *
   * **Measured so that the backward's share is not guessed at.** Subtract this from the
   * whole and what is left is the backward and the optimizer, and which of those two to
   * mend first is decided there. Reading the loss has to happen here too, so that the
   * place where it waits on the GPU is the same.
   */
  const forwardOnly = async (): Promise<void> => {
    const d = device();
    d.beginScope();
    try {
      // `noGrad` takes a **synchronous** body — pass an asynchronous one and the
      // gradient switch flips back before the read. It wraps the graph building alone,
      // and the read is awaited outside it.
      const out = noGrad(() => crit.forward(model.forward(x), y));
      await out.item();
    } finally {
      d.endScope();
    }
  };
  await forwardOnly();
  const f0 = performance.now();
  for (let i = 0; i < steps; i++) await forwardOnly();
  const perForward = (performance.now() - f0) / steps;

  /**
   * One step with the weight gradient off.
   *
   * **Which kernel is heavy comes out of a subtraction rather than a guess.**
   * `gradWeight` is a GEMM with a small output and a large reduction, so depending on the
   * layer it falls to a single workgroup — and whether that is actually expensive is
   * knowable only from the difference between on and off. The learning is wrong while this
   * runs, and what is being measured is time.
   */
  const convWeights = params.filter((p) => p.shape.length >= 3);
  for (const p of convWeights) p.requiresGrad = false;
  await one();
  const w0 = performance.now();
  const wd0 = device().dispatches;
  for (let i = 0; i < steps; i++) await one();
  const perNoWeightGrad = (performance.now() - w0) / steps;
  // **Does the flag really take work away.** By time alone, "gradWeight is cheap" and
  // "the flag does not bite" are the same screen. If the dispatch count did not fall it is
  // the second.
  const noWeightDispatches = (device().dispatches - wd0) / steps;
  for (const p of convWeights) p.requiresGrad = true;

  /**
   * GPU time by kind of kernel. **A count alone does not say where the cost is** — the
   * count stays put as the batch grows while the time grows differently per kind.
   *
   * A pass is opened per dispatch while measuring, so **the total is larger than the
   * ms/step above.** What to read here is the share, not the absolute figure.
   */
  await device().profile(() => one());
  const hot: [string, number][] = [];
  for (const [kind, ns] of device().nsByKind) hot.push([kind, ns / 1e6]);
  hot.sort((a, b) => b[1] - a[1]);
  const profiledMs = hot.reduce((a, [, ms]) => a + ms, 0);
  const profileDropped = device().profileDropped;

  // **One validation fault and no number is given.**
  //
  // An invalid command buffer throws nothing and simply does nothing. The wall clock runs
  // in that state too and produces an ms/step, which is not a measurement but the wall
  // clock of a state in which no learning happens. It happened — average pooling's backward
  // was invalid in its entirety, and "13070 ms/step" came out while the loss sat at
  // 2.27.
  const faults = device().faults;
  if (faults.count > 0) {
    // **Which kind, because the next step differs.** An allocation that could not be
    // made says the model or the batch is too large for this device; a command the
    // device would not run says a kernel is wrong. Reported as one number, the reader
    // starts by looking for the second and there may not be one.
    const split = faults.outOfMemory > 0
      ? `${faults.outOfMemory} out of memory, ${faults.count - faults.outOfMemory} `
        + "validation"
      : "all validation";
    throw new Error(
      `${faults.count} WebGPU fault(s) (${split}) — this is not a measurement.\n` +
        `first: ${faults.first}`,
    );
  }
  // **A lost device is stopped for the same reason.** A lost device silently declines to
  // run the commands, so the screen is the one above — the loss does not move and only the
  // wall clock does. Stop the first and leave this open, and the same false figure comes in
  // through another door.
  const lost = device().lost;
  if (lost) {
    throw new Error(
      `the WebGPU device was lost (${lost.reason}) — this is not a measurement.\n`
        + lost.message,
    );
  }

  const stepsPerEpoch = Math.ceil(CIFAR_TRAIN_IMAGES / batch);
  return {
    batch,
    params: count,
    gpuMb: Math.round(device().memory.bytes / 1e5) / 10,
    pooledMb: Math.round(device().pooled.bytes / 1e5) / 10,
    msPerStep: Math.round(perStep * 10) / 10,
    epochMin: Math.round((perStep * stepsPerEpoch) / 600) / 100,
    leakPerStep: leak,
    dispatches: Math.round(perStepDispatches),
    usPerDispatch: Math.round((perStep * 1000) / Math.max(1, perStepDispatches)),
    kinds,
    hot: hot.map(([k, ms]) => [k, Math.round(ms * 100) / 100]),
    profiledMs: Math.round(profiledMs * 10) / 10,
    profileDropped,
    submits: Math.round(perStepSubmits),
    msForward: Math.round(perForward * 10) / 10,
    msNoWeightGrad: Math.round(perNoWeightGrad * 10) / 10,
    noWeightDispatches: Math.round(noWeightDispatches),
    loss: Math.round(last * 10000) / 10000,
  };
}

/** The Python bench's batch sizes, in its order. */
export async function report(batches: readonly number[] = [16, 32, 64]): Promise<string> {
  const lines: string[] = [];
  for (const b of batches) {
    try {
      const r = await runStep(b);
      lines.push(
        `batch ${String(r.batch).padStart(3)}  ` +
        `${r.msPerStep.toFixed(1).padStart(8)} ms/step  ` +
        `(forward ${r.msForward.toFixed(1).padStart(7)} · no dW ` +
        `${r.msNoWeightGrad.toFixed(1).padStart(7)}/${r.noWeightDispatches}d)  ` +
        `epoch ${r.epochMin.toFixed(2).padStart(6)} min  ` +
        `dispatch ${String(r.dispatches).padStart(5)}  ` +
        `submits ${String(r.submits).padStart(3)}  ` +
        `${String(r.usPerDispatch).padStart(5)}µs/dispatch  ` +
        `leak ${r.leakPerStep.toFixed(1).padStart(5)}  ` +
        `${r.gpuMb.toFixed(1).padStart(6)}MB(+pool ${r.pooledMb.toFixed(1)})  ` +
        `loss ${r.loss.toFixed(4)}`,
      );
      // **The time breakdown is printed per batch.** The counts stay put as the batch
      // grows and the times do not — a superlinear kernel is visible only here.
      const hot = r.hot.slice(0, 8)
        .map(([kind, ms]) => `${kind} ${ms.toFixed(1)}`).join(" · ");
      // If it was cut short, say so **beside the total.** Put in a footnote it is read
      // past, with only the table taken in.
      const cut = r.profileDropped > 0
        ? `, ${r.profileDropped} had no query slot and went unmeasured — this is part of it`
        : "";
      lines.push(`         GPU time (ms, total ${r.profiledMs}${cut}): ${hot}`);
      // The counts by kind are the same for every batch, so they are printed once.
      if (b === batches[0]) {
        const top = r.kinds.slice(0, 8)
          .map(([kind, n]) => `${kind} ${n}`).join(" · ");
        lines.push(`         dispatch breakdown: ${top}`);
      }
    } catch (err) {
      lines.push(`batch ${String(b).padStart(3)}  failed: ` +
        `${err instanceof Error ? `${err.constructor.name}: ${err.message.slice(0, 120)}` : String(err)}`);
      // **Where it happened matters as much as what happened.** Keep the wording alone
      // and, where several places produce the same wording, they cannot be told apart.
      if (err instanceof Error && err.stack) {
        lines.push(`         ${err.stack.split("\n").slice(3, 11).join("\n         ")}`);
      }
    }
  }
  // **Which device it was measured on comes before the numbers.** Where a headless
  // browser hands over a software adapter, slow figures come out with no exception raised,
  // and they are not the library's result.
  return `ResNet-18 (CIFAR) · a real training step, per batch\n`
    + `adapter: ${Device.adapterInfo}\n`
    + lines.join("\n");
}

/** So the scope can be used from outside — the runner wraps the whole in one more. */
export { scope };

/**
 * Take the pieces off one at a time and backpropagate **with a different weight per
 * position.**
 *
 * Weigh the whole model alone and, when it parts, there is no telling where. And folding
 * with a plain `sum()` makes every upstream gradient 1, at which point **BatchNorm's two
 * backward correction terms cancel exactly** — that is why the golden's
 * `grad::BatchNorm2d/x` expects 4.7e-10, and why that case asks nothing at all about the
 * correction terms.
 */
export async function dumpPieces(): Promise<Record<string, {
  params: number[][];
  shapes: number[][];
  input: number[];
  inputShape: number[];
  output: number[];
  inputGrad: number[];
  paramGrads: (number[] | null)[];
}>> {
  const pieces: Record<string, () => { mod: nn.Module; shape: number[] }> = {
    bn: () => ({ mod: new nn.BatchNormND(3), shape: [2, 3, 4, 4] }),
    avgpool: () => ({ mod: new AvgPoolTo1(), shape: [2, 3, 4, 4] }),
    relu: () => ({ mod: new nn.ReLU(), shape: [2, 3, 4, 4] }),
    conv: () => ({
      mod: new nn.Conv2d(3, 4, 3, 1, 1, 1, 1, false), shape: [2, 3, 4, 4],
    }),
    block: () => ({ mod: new Block(3, 3, 1), shape: [2, 3, 4, 4] }),
    blockDown: () => ({ mod: new Block(3, 6, 2), shape: [2, 3, 8, 8] }),
  };
  const out: Record<string, {
    params: number[][]; shapes: number[][]; input: number[]; inputShape: number[];
    output: number[]; inputGrad: number[]; paramGrads: (number[] | null)[];
  }> = {};
  for (const [name, build] of Object.entries(pieces)) {
    const { mod, shape } = build();
    const n = shape.reduce((a, b) => a * b, 1);
    const data = new Float32Array(n);
    for (let i = 0; i < n; i++) data[i] = Math.sin(i * 0.31) * 0.7;
    const x = keepAlive(Tensor.from(data, shape, { requiresGrad: true }));
    const y = mod.forward(x);
    // **A different weight per position.** All ones and the correction terms cancel,
    // and nothing is asked.
    const w = new Float32Array(y.size);
    for (let i = 0; i < w.length; i++) w[i] = ((i % 7) + 1) * 0.3;
    y.mul(Tensor.from(w, y.shape)).sum().backward();
    const grad = x.grad;
    if (!grad) throw new Error(`${name}: no gradient arrived at the input`);
    const params = mod.parameters();
    out[name] = {
      params: await Promise.all(params.map(async (p) => [...await p.toArray()])),
      shapes: params.map((p) => [...p.shape]),
      input: [...data],
      inputShape: shape,
      output: [...await y.toArray()],
      inputGrad: [...await grad.toArray()],
      paramGrads: await Promise.all(params.map(async (p) =>
        p.grad ? [...await p.grad.toArray()] : null)),
    };
  }
  return out;
}

/** A shell for the comparison — the partner of `AdaptiveAvgPool2d(1)` on the Python
 * side. */
class AvgPoolTo1 extends nn.Module {
  override forward(x: Tensor): Tensor {
    return x.adaptiveAvgPool(1);
  }
}

/**
 * The material for asking whether this ResNet-18 is **the same model as the Python
 * side's.**
 *
 * The bench's model was carried across by reading `tests/browser/bench.py` by eye, so it
 * sits outside the golden. Let the block arrangement or a BN's position differ subtly and
 * both the speed and the accuracy become **a comparison of two different models** — and
 * that parting is visible only in the values.
 *
 * The parameters go out **in order.** With the same structure the list of shapes is
 * exactly equal, and with a different one it parts there first — the naming rules differ
 * between the two languages, so they are lined up by position.
 */
export async function dumpForComparison(batch = 2): Promise<{
  shapes: number[][];
  params: number[][];
  input: number[];
  output: number[];
  loss: number;
  inputGrad: number[];
}> {
  const model = new ResNet18();
  const params = model.parameters();
  // The input and the labels are both pinned — a comparison wants both sides looking at
  // the same numbers.
  const pixels = new Float32Array(batch * 3 * 32 * 32);
  for (let i = 0; i < pixels.length; i++) {
    pixels[i] = Math.sin(i * 0.017) * 0.5;
  }
  const labels = new Float32Array(batch);
  for (let i = 0; i < batch; i++) labels[i] = i % 10;

  const x = keepAlive(Tensor.from(pixels, [batch, 3, 32, 32], { requiresGrad: true }));
  const y = keepAlive(Tensor.from(labels, [batch], { dtype: "int64" }));
  const out = model.forward(x);
  const loss = new nn.CrossEntropyLoss().forward(out, y);
  loss.backward();

  const grad = x.grad;
  if (!grad) throw new Error("no gradient arrived at the input — the graph was cut");
  return {
    shapes: params.map((p) => [...p.shape]),
    params: await Promise.all(params.map(async (p) => [...await p.toArray()])),
    input: [...pixels],
    output: [...await out.toArray()],
    loss: await loss.item(),
    inputGrad: [...await grad.toArray()],
  };
}
