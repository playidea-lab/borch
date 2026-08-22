/**
 * **Real accuracy**, measured on CIFAR-10.
 *
 * **The same yardstick** as `train_eval` in `tests/browser/bench.py` — same model, same
 * optimizer (SGD lr 0.05, momentum 0.9, weight decay 5e-4), same batch (128), same
 * normalisation, same augmentation (crop 32 with padding 4, horizontal flip 0.5).
 *
 * ## Both the training accuracy **and** the test accuracy
 *
 * The training accuracy always rises, so on its own it looks like things are going well.
 * The **gap** between the two is the overfitting, and that gap is what augmentation is
 * there to close. The test accuracy alone will not do either — when it stops rising, that
 * alone cannot separate not having learned enough from having memorised.
 */

import { Device } from "../src/device.js";
import * as nn from "../src/nn.js";
import { SGD } from "../src/optim.js";
import { device, keepAlive, noGrad, Tensor } from "../src/tensor.js";
import { augmentBatch, normalizeBatch } from "../src/vision.js";
import { ResNet18 } from "./bench.js";

/** CIFAR-10's usual figures. Without the normalisation the first epoch settles
 * noticeably more slowly. */
const CIFAR_MEAN = [0.4914, 0.4822, 0.4465];
const CIFAR_STD = [0.2470, 0.2435, 0.2616];
/** One picture is 3073 bytes — 1 label and 1024 each of R, G and B. That order is
 * exactly (3,32,32). */
const RECORD = 3073;
const SIDE = 32;
const PIXELS = SIDE * SIDE;

export interface Split {
  readonly x: Float32Array;
  readonly y: Int32Array;
  readonly n: number;
}

/** Unpack one binary blob into `(x, y)`. The values are divided into [0,1]. */
export function decodeCifar(raw: Uint8Array): Split {
  if (raw.length % RECORD !== 0) {
    throw new Error(
      `not a CIFAR-10 binary — ${raw.length} bytes is not a multiple of ${RECORD}`,
    );
  }
  const n = raw.length / RECORD;
  const x = new Float32Array(n * 3 * PIXELS);
  const y = new Int32Array(n);
  for (let i = 0; i < n; i++) {
    const at = i * RECORD;
    y[i] = raw[at] ?? 0;
    for (let k = 0; k < 3 * PIXELS; k++) {
      x[i * 3 * PIXELS + k] = (raw[at + 1 + k] ?? 0) / 255;
    }
  }
  return { x, y, n };
}

/** Gather the chosen pictures into the shape the model takes. Augmentation comes
 * **before** normalisation — pad the border with zeros first and those zeros are then
 * measured by the same ruler as every other pixel. */
function prepare(
  split: Split, picks: Int32Array | number[], augment: boolean,
): Float32Array {
  const n = picks.length;
  const stride = 3 * PIXELS;
  const gathered = new Float32Array(n * stride);
  for (let i = 0; i < n; i++) {
    const from = (picks[i] ?? 0) * stride;
    gathered.set(split.x.subarray(from, from + stride), i * stride);
  }
  const shaped = augment
    ? augmentBatch(gathered, n, 3, SIDE, SIDE,
      { crop: SIDE, padding: 4, hflipP: 0.5 })
    : gathered;
  return normalizeBatch(shaped, n, 3, PIXELS, CIFAR_MEAN, CIFAR_STD);
}

/**
 * The share that was right. Measuring it **on data the training never saw** is the whole
 * of this function.
 *
 * Being in evaluation mode is the condition — with BatchNorm using the training
 * statistics, the figure moves with how the batch happens to be made up, and what is
 * measured is the luck of the batch rather than the accuracy.
 */
async function accuracy(
  model: nn.Module, split: Split, batch = 250,
): Promise<number> {
  model.eval();
  let right = 0;
  for (let i = 0; i < split.n; i += batch) {
    const size = Math.min(batch, split.n - i);
    const picks = Array.from({ length: size }, (_, k) => i + k);
    const d = device();
    d.beginScope();
    try {
      const out = noGrad(() => {
        const x = Tensor.from(prepare(split, picks, false), [size, 3, SIDE, SIDE]);
        return model.forward(x);
      });
      const scores = await out.toArray();
      for (let k = 0; k < size; k++) {
        let best = 0;
        for (let c = 1; c < 10; c++) {
          if ((scores[k * 10 + c] ?? 0) > (scores[k * 10 + best] ?? 0)) best = c;
        }
        if (best === split.y[i + k]) right += 1;
      }
    } finally {
      d.endScope();
    }
  }
  model.train();
  return right / split.n;
}

/** The shuffle. The seed is pinned so both conditions see the same order. */
function shuffled(n: number, seed: number): Int32Array {
  const order = new Int32Array(n);
  for (let i = 0; i < n; i++) order[i] = i;
  let s = seed >>> 0 || 1;
  for (let i = n - 1; i > 0; i--) {
    s ^= s << 13; s >>>= 0; s ^= s >> 17; s ^= s << 5; s >>>= 0;
    const j = s % (i + 1);
    const t = order[i] ?? 0;
    order[i] = order[j] ?? 0;
    order[j] = t;
  }
  return order;
}

export interface EpochRow {
  epoch: number;
  train: number;
  test: number;
  loss: number;
  seconds: number;
}

export async function trainEval(
  train: Split, test: Split,
  opts: { epochs?: number; batch?: number; lr?: number; augment?: boolean } = {},
): Promise<EpochRow[]> {
  const epochs = opts.epochs ?? 10;
  const batch = opts.batch ?? 128;
  const lr = opts.lr ?? 0.05;
  const augment = opts.augment ?? false;

  const model = new ResNet18();
  const opt = new SGD(model.parameters(), lr, 0.9, 5e-4);
  const crit = new nn.CrossEntropyLoss();
  const usable = train.n - (train.n % batch);
  const rows: EpochRow[] = [];

  for (let e = 0; e < epochs; e++) {
    const order = shuffled(train.n, 1234 + e);
    const t0 = performance.now();
    let last = 0;
    for (let i = 0; i < usable; i += batch) {
      const picks = order.subarray(i, i + batch);
      const bx = prepare(train, picks, augment);
      const by = new Float32Array(batch);
      for (let k = 0; k < batch; k++) by[k] = train.y[picks[k] ?? 0] ?? 0;
      const d = device();
      d.beginScope();
      try {
        const x = Tensor.from(bx, [batch, 3, SIDE, SIDE]);
        const y = Tensor.from(by, [batch], { dtype: "int64" });
        opt.zeroGrad();
        const loss = crit.forward(model.forward(x), y);
        loss.backward();
        opt.step();
        last = await loss.item();
      } finally {
        d.endScope();
      }
    }
    const seconds = (performance.now() - t0) / 1000;
    rows.push({
      epoch: e + 1,
      train: await accuracy(model, train),
      test: await accuracy(model, test),
      loss: Math.round(last * 10000) / 10000,
      seconds: Math.round(seconds * 10) / 10,
    });
    // A long measurement that dies partway loses its return value entirely. What was let
    // out along the way is what remains.
    //
    // **The `[accuracy]` prefix is a contract with `accuracy.py`**, which keeps the console
    // lines that begin with it. Change one side alone and the progress goes quietly
    // missing; `test_messages.py` holds the two spellings together.
    console.log(`[accuracy] epoch ${e + 1}/${epochs} augment=${augment} `
      + `train ${(rows[rows.length - 1]?.train ?? 0).toFixed(3)} `
      + `test ${(rows[rows.length - 1]?.test ?? 0).toFixed(3)} `
      + `${seconds.toFixed(1)}s`);
  }
  return rows;
}

export async function report(
  train: Split, test: Split, epochs: number, only?: boolean,
): Promise<string> {
  const conditions = only === undefined ? [false, true] : [only];
  const lines: string[] = [
    `adapter: ${Device.adapterInfo}`,
    `ResNet-18 (CIFAR) · accuracy`,
    `${train.n} training pictures / ${test.n} test, batch 128, ${epochs} epochs`,
  ];
  for (const augment of conditions) {
    const rows = await trainEval(train, test, { epochs, augment });
    lines.push(`\naugmentation ${augment ? "on" : "off"}`);
    for (const r of rows) {
      lines.push(`  epoch ${String(r.epoch).padStart(2)}  `
        + `train ${r.train.toFixed(3)}  test ${r.test.toFixed(3)}  `
        + `loss ${r.loss.toFixed(4)}  ${r.seconds.toFixed(1)}s`);
    }
    const best = rows.reduce((a, b) => (b.test > a.test ? b : a), rows[0]!);
    lines.push(
      `  best test accuracy ${(best.test * 100).toFixed(1)}% (epoch ${best.epoch})`);
  }
  return lines.join("\n");
}

/** The way to keep parameters alive outside a scope — the runner stands the model up
 * from outside. */
export { keepAlive };
