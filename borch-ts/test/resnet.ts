/**
 * The same small ResNet as `tests/resnet.py`, written the way a JavaScript user writes it.
 *
 * The Python binding already trains this network through borch.ts — every kernel, the
 * autograd tape and the optimizer are that library's. So what this file adds is not the
 * arithmetic but **the surface**: whether borch.ts can be handed a custom `Module`
 * subclass with a residual join, filled from outside, trained, and read back, using only
 * what a page has. That is the half the binding hides, because the binding writes it.
 *
 * The fixture is `minstd`, and this is the whole reason it exists. `numpy.random` is
 * reachable from three of the four surfaces and not from here, so a scenario resting on
 * it would have to stop one short — which is exactly the shape of gap this repository
 * keeps finding. MINSTD's modulus is small enough that the product stays under 2^53, so
 * this multiply and Python's agree bit for bit.
 *
 * Loaded by `tests/browser/run.py --resnet-ts`, which prints the same key/value lines the
 * Python run prints so the two can be read side by side.
 */

import { Tensor } from "../src/tensor.js";
import * as nn from "../src/nn.js";
import { SGD } from "../src/optim.js";
import { noGrad } from "../src/autograd.js";

const STEPS = 30;
const BATCH = 8, CHANNELS = 3, SIDE = 16, CLASSES = 4;
const WIDTH = 8;

/**
 * `s ← s·48271 mod 2^31−1`, uniform in `[-scale, scale)`.
 *
 * The twin of `minstd` in `tests/resnet.py`. **Two copies of a generator is a rule that
 * diverges** — the guard against that is not a shared file, which no page can import from
 * Python, but that `test_resnet.py` compares the two surfaces' answers and fails the
 * moment a byte parts.
 */
function minstd(count: number, scale: number, seed: number): number[] {
  const out: number[] = [];
  let state = seed % 2147483647 || 1;
  for (let i = 0; i < count; i++) {
    state = (state * 48271) % 2147483647;
    out.push(Math.fround((state / 2147483647.0 - 0.5) * 2.0 * scale));
  }
  return out;
}

class Residual extends nn.Module {
  readonly conv1: nn.Conv2d;
  readonly bn1: nn.BatchNorm2d;
  readonly conv2: nn.Conv2d;
  readonly bn2: nn.BatchNorm2d;
  readonly short: nn.Conv2d | null;
  readonly shortbn: nn.BatchNorm2d | null;

  constructor(cin: number, cout: number, stride = 1) {
    super();
    this.conv1 = new nn.Conv2d(cin, cout, 3, stride, 1, 1, 1, false);
    this.bn1 = new nn.BatchNorm2d(cout);
    this.conv2 = new nn.Conv2d(cout, cout, 3, 1, 1, 1, 1, false);
    this.bn2 = new nn.BatchNorm2d(cout);
    // Identity while the shape survives — giving every block a 1×1 anyway would stop the
    // identity path from ever being asked about.
    this.short = stride !== 1 || cin !== cout
      ? new nn.Conv2d(cin, cout, 1, stride, 0, 1, 1, false) : null;
    this.shortbn = this.short ? new nn.BatchNorm2d(cout) : null;
  }

  override forward(x: Tensor): Tensor {
    let out = this.bn1.call(this.conv1.call(x)).relu();
    out = this.bn2.call(this.conv2.call(out));
    const short = this.short && this.shortbn
      ? this.shortbn.call(this.short.call(x)) : x;
    return out.add(short).relu();
  }
}

class Net extends nn.Module {
  readonly stem = new nn.Conv2d(CHANNELS, WIDTH, 3, 1, 1, 1, 1, false);
  readonly stembn = new nn.BatchNorm2d(WIDTH);
  readonly b1 = new Residual(WIDTH, WIDTH);
  readonly b2 = new Residual(WIDTH, WIDTH * 2, 2);
  readonly head = new nn.Linear(WIDTH * 2, CLASSES);

  override forward(x: Tensor): Tensor {
    let h = this.stembn.call(this.stem.call(x)).relu();
    h = this.b2.call(this.b1.call(h));
    // Averaged over height and width rather than flattened, so the last linear sees a
    // reduction's gradient.
    return this.head.call(h.mean(3).mean(2));
  }
}

/** Trains it and returns the same `key\tvalue` lines `tests/resnet.py` prints. */
export async function report(): Promise<string> {
  const images = minstd(BATCH * CHANNELS * SIDE * SIDE, 1.7, 11);
  const labels = minstd(BATCH, 1.0, 29)
    .map((v) => Math.floor(Math.abs(v) * 1000) % CLASSES);

  const model = new Net();
  // **The same bytes as every other surface.** The seed is the parameter's position
  // rather than a running stream, so a future reordering moves one weight instead of all
  // of them. Names are sorted, which is the order Python's `named_parameters` walks.
  const params = model.namedParameters();
  const order = Object.keys(params);
  order.forEach((name, index) => {
    const p = params[name] as Tensor;
    const shape = [...p.shape];
    const fan = shape.length > 1
      ? shape.slice(1).reduce((a, b) => a * b, 1) : (shape[0] ?? 1);
    const scale = Math.sqrt(6.0 / Math.max(fan, 1));
    noGrad(() => p.copyFrom(Tensor.from(minstd(p.size, scale, 101 + index * 7), shape)));
  });

  const crit = new nn.CrossEntropyLoss();
  const opt = new SGD(model.parameters(), 0.05, 0.9, 0, 1e-4);
  const x = Tensor.from(images, [BATCH, CHANNELS, SIDE, SIDE]);
  const y = Tensor.from(labels, [BATCH]).long();
  const lines: string[] = [];

  model.train();
  for (let step = 0; step < STEPS; step++) {
    opt.zeroGrad();
    const loss = crit.forward(model.call(x), y);
    loss.backward();
    if (step === 0) {
      // Before a single weight has moved, so a difference is the backward pass and
      // nothing else.
      for (const name of order) {
        const g = (params[name] as Tensor).grad;
        const norm = g ? await g.mul(g).sum().sqrt().item() : 0;
        lines.push(`grad·${name}\t${norm}`);
      }
    }
    lines.push(`loss·${String(step).padStart(2, "0")}\t${await loss.item()}`);
    opt.step();
  }

  const buffers = model.namedBuffers();
  for (const name of Object.keys(buffers).sort()) {
    if (name.includes("num_batches")) continue;
    lines.push(`buffer·${name}\t${await (buffers[name] as Tensor).sum().item()}`);
  }

  model.eval();
  const logits = await noGrad(async () => model.call(x));
  lines.push(`eval·logit sum\t${await logits.sum().item()}`);
  const hit = await logits.argmax(1).eq(y).sum().item();
  lines.push(`eval·argmax agreement\t${hit / BATCH}`);
  return lines.join("\n");
}
