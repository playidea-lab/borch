/**
 * CIFAR-10 으로 **진짜 정확도**를 잰다.
 *
 * `tests/browser/bench.py` 의 `train_eval` 과 **같은 잣대**다 — 같은 모델, 같은
 * 옵티마이저(SGD lr 0.05, 모멘텀 0.9, 가중치 감쇠 5e-4), 같은 배치(128), 같은
 * 정규화, 같은 늘리기(자르기 32·채움 4·좌우뒤집기 0.5).
 *
 * ## 학습 정확도와 시험 정확도를 **둘 다** 잰다
 *
 * 학습 정확도만 보면 늘 오르므로 잘 되는 것처럼 보인다. 둘의 **차이**가 과적합이고,
 * 늘리기가 줄이라고 있는 것이 그 차이다. 시험 정확도만 봐도 안 된다 — 안 오를 때
 * 아직 덜 배운 것인지 이미 외운 것인지가 안 갈린다.
 */

import { Device } from "../src/device.js";
import * as nn from "../src/nn.js";
import { SGD } from "../src/optim.js";
import { device, keepAlive, noGrad, Tensor } from "../src/tensor.js";
import { augmentBatch, normalizeBatch } from "../src/vision.js";
import { ResNet18 } from "./bench.js";

/** CIFAR-10 의 통상값. 정규화를 빼면 첫 에폭이 눈에 띄게 느리게 붙는다. */
const CIFAR_MEAN = [0.4914, 0.4822, 0.4465];
const CIFAR_STD = [0.2470, 0.2435, 0.2616];
/** 한 장이 3073 바이트 — 라벨 1 에 R·G·B 가 1024 씩. 그 순서가 곧 (3,32,32) 다. */
const RECORD = 3073;
const SIDE = 32;
const PIXELS = SIDE * SIDE;

export interface Split {
  readonly x: Float32Array;
  readonly y: Int32Array;
  readonly n: number;
}

/** 바이너리 한 덩이를 `(x, y)` 로 푼다. 값은 [0,1] 로 나눠 둔다. */
export function decodeCifar(raw: Uint8Array): Split {
  if (raw.length % RECORD !== 0) {
    throw new Error(
      `CIFAR-10 바이너리가 아니다 — ${raw.length} 바이트는 ${RECORD} 의 배수가 아니다`,
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

/** 고른 장들을 모아 모델에 넣을 모양으로. 늘리기는 **정규화보다 먼저** 한다 —
 * 가장자리를 0 으로 채운 뒤 정규화해야 그 0 이 다른 화소와 같은 자로 재진다. */
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
 * 맞힌 비율. **학습에 안 쓴 데이터로** 재는 것이 이 함수의 전부다.
 *
 * 평가 모드로 두는 것이 조건이다 — BatchNorm 이 학습 통계를 쓰면 배치 구성에 따라
 * 값이 흔들려서, 재는 것이 정확도가 아니라 배치 운이 된다.
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

/** 섞기. 씨앗을 박아 두 조건이 같은 차례를 보게 한다. */
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
    // 긴 측정은 도중에 죽으면 반환값이 통째로 사라진다. 흘려보낸 것만 남는다.
    console.log(`[accuracy] 에폭 ${e + 1}/${epochs} 늘리기=${augment} `
      + `학습 ${(rows[rows.length - 1]?.train ?? 0).toFixed(3)} `
      + `시험 ${(rows[rows.length - 1]?.test ?? 0).toFixed(3)} `
      + `${seconds.toFixed(1)}초`);
  }
  return rows;
}

export async function report(
  train: Split, test: Split, epochs: number, only?: boolean,
): Promise<string> {
  const conditions = only === undefined ? [false, true] : [only];
  const lines: string[] = [
    `어댑터: ${Device.adapterInfo}`,
    `ResNet-18(CIFAR) · 정확도`,
    `학습 ${train.n}장 / 시험 ${test.n}장, 배치 128, ${epochs} 에폭`,
  ];
  for (const augment of conditions) {
    const rows = await trainEval(train, test, { epochs, augment });
    lines.push(`\n늘리기 ${augment ? "켬" : "끔"}`);
    for (const r of rows) {
      lines.push(`  에폭 ${String(r.epoch).padStart(2)}  `
        + `학습 ${r.train.toFixed(3)}  시험 ${r.test.toFixed(3)}  `
        + `손실 ${r.loss.toFixed(4)}  ${r.seconds.toFixed(1)}초`);
    }
    const best = rows.reduce((a, b) => (b.test > a.test ? b : a), rows[0]!);
    lines.push(`  가장 좋은 시험 정확도 ${(best.test * 100).toFixed(1)}% (에폭 ${best.epoch})`);
  }
  return lines.join("\n");
}

/** 파라미터를 구역 밖에서 살려 두는 통로 — 러너가 모델을 밖에서 세운다. */
export { keepAlive };
