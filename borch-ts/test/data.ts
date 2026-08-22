/**
 * Datasets and the loader — do the batches come out right, does the shuffle follow the
 * seed.
 *
 * **The fast road and the slow road have to give the same answer.** `TensorDataset` takes
 * a batch in one go (`narrow`, `indexSelect`), and a dataset with no `gather` fetches the
 * samples one at a time and stacks them. Let the two part and **only the fast one is
 * wrong and nobody sees it** — the values come out plausible. So both roads are run side
 * by side here and weighed against each other.
 */

import {
  data, device, init, keepAlive, manualSeed, nn, optim, scope, Tensor,
} from "../src/index.js";

interface Check { name: string; ok: boolean; note: string }
/**
 * `checks` is the authority in this report. `text` is the shadow a person reads.
 *
 * **The runner used to judge by scanning a sentence.** That way of judging changes its
 * answer quietly when the wording changes, and in `readme.py` it did — with one of the two
 * examples failing, the word it looked for was still sitting on another line, so it
 * returned 0. Hand the state over as it is and the runner can count, and can say for
 * itself which thing failed.
 */
export interface Report { text: string; checks: Check[] }
const checks: Check[] = [];

function want(name: string, ok: boolean, note = ""): void {
  checks.push({ name, ok, note });
}

function wantThrow(name: string, body: () => unknown): void {
  try {
    body();
    want(name, false, "it did not throw");
  } catch {
    want(name, true);
  }
}

/** A dataset with no `gather`. It forces the loader down the slow road. */
class Plain implements data.Dataset {
  constructor(private readonly inner: data.Dataset) {}
  get length(): number { return this.inner.length; }
  get(index: number): readonly Tensor[] { return this.inner.get(index); }
}

const ROWS = 10;
const WIDTH = 3;

/** Data whose values say where they came from. `x[i][j] === i * 10 + j`, so a sample
 * stays traceable through a shuffle. */
function makeSet(): data.TensorDataset {
  const xs = new Float32Array(ROWS * WIDTH);
  for (let i = 0; i < ROWS; i++) {
    for (let j = 0; j < WIDTH; j++) xs[i * WIDTH + j] = i * 10 + j;
  }
  const ys = Float32Array.from({ length: ROWS }, (_, i) => i);
  return new data.TensorDataset(
    keepAlive(Tensor.from(xs, [ROWS, WIDTH])),
    keepAlive(Tensor.from(ys, [ROWS], { dtype: "int64" })),
  );
}

/** Walk every batch and collect the first column — which sample arrived in which
 * order. */
async function rowsOf(loader: data.DataLoader): Promise<number[]> {
  const seen: number[] = [];
  for (const [x] of loader) {
    const flat = await (x as Tensor).toArray();
    for (let i = 0; i < flat.length; i += WIDTH) seen.push((flat[i] as number) / 10);
  }
  return seen;
}

export async function report(): Promise<Report> {
  await init();
  const set = makeSet();

  // ── The dataset ─────────────────────────────────────────────────────
  want("the length is the number of samples", set.length === ROWS, `${set.length}`);
  const item = set.get(3);
  want("one sample drops the first axis",
    item.length === 2 && item[0]!.shape.join(",") === `${WIDTH}`
      && item[1]!.shape.length === 0,
    `${item[0]!.shape} / ${item[1]!.shape}`);
  want("the sample's value is right", (await item[0]!.toArray())[0] === 30);
  // **This check pulled out a defect.** The shape and index operations were losing the
  // dtype label — `select`, `narrow` and `indexSelect` called `Tensor.make` without a
  // dtype and its default was float32. Take a sample out of int64 labels and it came back
  // as float32: the kind where the value is right and only the label parts, so it is
  // caught by eye alone.
  //
  // It was mended at the root, so the re-labelling that used to sit here (`keepLabel`) is
  // gone. **The check stays** — what holds the contract should be these three lines rather
  // than defensive code. Should the root go back, this goes red.
  want("a sample keeps its dtype label", item[1]!.dtype === "int64", item[1]!.dtype);
  want("a contiguous batch keeps the label too",
    set.gather([0, 1, 2])[1]!.dtype === "int64", set.gather([0, 1, 2])[1]!.dtype);
  want("a scattered batch keeps it too",
    set.gather([3, 1, 2])[1]!.dtype === "int64", set.gather([3, 1, 2])[1]!.dtype);

  // Bind tensors with different sample counts and the labels shift, quietly. It has to
  // stop there.
  wantThrow("differing sample counts are refused", () => new data.TensorDataset(
    Tensor.from([1, 2, 3], [3]), Tensor.from([1, 2], [2])));
  wantThrow("no tensors at all is refused", () => new data.TensorDataset());

  // ── Counting the batches ────────────────────────────────────────────
  want("the batch count rounds up",
    new data.DataLoader(set, { batchSize: 3 }).length === 4);
  want("dropLast rounds down",
    new data.DataLoader(set, { batchSize: 3, dropLast: true }).length === 3);
  want("length counts batches, not samples",
    new data.DataLoader(set, { batchSize: 10 }).length === 1);
  wantThrow("batchSize 0 is refused",
    () => new data.DataLoader(set, { batchSize: 0 }));

  // ── Order and contents ──────────────────────────────────────────────
  const straight = await rowsOf(new data.DataLoader(set, { batchSize: 3 }));
  want("unshuffled, everything comes out in order",
    straight.join(",") === "0,1,2,3,4,5,6,7,8,9", straight.join(","));

  const dropped = await rowsOf(
    new data.DataLoader(set, { batchSize: 3, dropLast: true }));
  want("dropLast throws the remainder away",
    dropped.join(",") === "0,1,2,3,4,5,6,7,8", dropped.join(","));

  // ── The fast road and the slow road ─────────────────────────────────
  // **This is the reason the runner exists.** Where the side with `gather` and the side
  // without part, only the fast one is wrong, and the values are plausible enough not to
  // show.
  manualSeed(5);
  const fast = await rowsOf(new data.DataLoader(set, { batchSize: 4, shuffle: true }));
  manualSeed(5);
  const slow = await rowsOf(
    new data.DataLoader(new Plain(set), { batchSize: 4, shuffle: true }));
  want("the fast road and the slow road give the same answer",
    fast.join(",") === slow.join(","), `${fast.join(",")} / ${slow.join(",")}`);

  // Shuffled or not, every sample has to come exactly once.
  want("a shuffle drops nothing and repeats nothing",
    [...fast].sort((a, b) => a - b).join(",") === "0,1,2,3,4,5,6,7,8,9",
    fast.join(","));

  // ── Does the shuffle follow the seed ────────────────────────────────
  const draw = async (seed: number): Promise<string> => {
    manualSeed(seed);
    return (await rowsOf(new data.DataLoader(set, { batchSize: 4, shuffle: true })))
      .join(",");
  };
  want("the same seed gives the same order", (await draw(9)) === (await draw(9)));
  want("different seeds give different orders", (await draw(1)) !== (await draw(2)));

  // **It has to reshuffle each epoch.** Decided once in the constructor, the second epoch
  // runs in the first one's order and the reason for shuffling is gone.
  manualSeed(3);
  const loader = new data.DataLoader(set, { batchSize: 4, shuffle: true });
  const epoch1 = (await rowsOf(loader)).join(",");
  const epoch2 = (await rowsOf(loader)).join(",");
  want("it reshuffles each epoch", epoch1 !== epoch2, `${epoch1} / ${epoch2}`);

  // ── Splitting ───────────────────────────────────────────────────────
  manualSeed(11);
  const [train, valid] = data.randomSplit(set, [7, 3]);
  want("the split lengths are right", train!.length === 7 && valid!.length === 3);
  const trainRows = await rowsOf(new data.DataLoader(train!, { batchSize: 7 }));
  const validRows = await rowsOf(new data.DataLoader(valid!, { batchSize: 3 }));
  want("the split does not overlap and covers everything",
    [...trainRows, ...validRows].sort((a, b) => a - b).join(",")
      === "0,1,2,3,4,5,6,7,8,9",
    `${trainRows.join(",")} | ${validRows.join(",")}`);
  wantThrow("a total that does not add up is refused",
    () => data.randomSplit(set, [5, 3]));

  // A Subset has to take the fast road too — splitting into train and validation is an
  // ordinary thing to do, so dropping to the slow road here sends most training down it.
  manualSeed(4);
  const subFast = await rowsOf(new data.DataLoader(train!, { batchSize: 3, shuffle: true }));
  manualSeed(4);
  const subSlow = await rowsOf(
    new data.DataLoader(new Plain(train!), { batchSize: 3, shuffle: true }));
  want("both roads agree on a Subset too", subFast.join(",") === subSlow.join(","),
    `${subFast.join(",")} / ${subSlow.join(",")}`);

  // ── Concatenating ───────────────────────────────────────────────────
  const joined = new data.ConcatDataset([set, set]);
  want("the concatenated length is the sum", joined.length === ROWS * 2);
  want("the back half points at the second dataset",
    (await joined.get(ROWS + 2)[0]!.toArray())[0] === 20);

  // ── Does training actually run ──────────────────────────────────────
  //
  // **Asking only "did it go down" passes for free.** This first used the `set` above,
  // whose features run 0 to 92, so the first epoch was large and it settled straight onto
  // chance (ln 10 = 2.303 for ten classes). In that state `last < first` is still true and
  // the check was green.
  //
  // So a learnable problem is built separately and what is asked is whether it gets
  // **clearly below chance.** Let the loader shift the labels and this one never comes
  // down off chance.
  const CLASSES = 3;
  const SAMPLES = 60;
  const feats = new Float32Array(SAMPLES * WIDTH);
  const labels = new Float32Array(SAMPLES);
  for (let i = 0; i < SAMPLES; i++) {
    const cls = i % CLASSES;
    // One axis stands up per class — the problem has to be linearly separable for
    // whether the loader attaches the labels correctly to show in the loss.
    for (let j = 0; j < WIDTH; j++) feats[i * WIDTH + j] = j === cls ? 1 : -1;
    labels[i] = cls;
  }
  const learnable = new data.TensorDataset(
    keepAlive(Tensor.from(feats, [SAMPLES, WIDTH])),
    keepAlive(Tensor.from(labels, [SAMPLES], { dtype: "int64" })),
  );

  manualSeed(7);
  const model = new nn.Sequential(new nn.Linear(WIDTH, 8), new nn.ReLU(),
    new nn.Linear(8, CLASSES));
  for (const p of model.parameters()) keepAlive(p);
  const opt = new optim.Adam(model.parameters(), 0.05);
  const crit = new nn.CrossEntropyLoss();
  const training = new data.DataLoader(learnable, { batchSize: 10, shuffle: true });
  const losses: number[] = [];
  for (let epoch = 0; epoch < 12; epoch++) {
    let total = 0;
    for (const [x, y] of training) {
      await scope(async () => {
        opt.zeroGrad();
        const loss = crit.call(model.call(x as Tensor), y as Tensor);
        loss.backward();
        opt.step();
        total += await loss.item();
      });
    }
    losses.push(total / training.length);
  }
  const chance = Math.log(CLASSES);
  const last = losses[losses.length - 1] ?? NaN;
  want("training through the loader gets below chance", last < chance * 0.3,
    `${losses.map((v) => v.toFixed(3)).join(" → ")} (chance ${chance.toFixed(3)})`);

  // Stops the place where an unchanged value reads as green.
  want("no WebGPU validation fault", device().faults.count === 0,
    device().faults.first);

  const failed = checks.filter((c) => !c.ok);
  const lines = checks.map((c) =>
    `  ${c.ok ? "✓" : "✗"} ${c.name}${c.note ? ` — ${c.note}` : ""}`);
  lines.push(failed.length === 0
    ? `all ${checks.length} data-loading checks passed`
    : `**${failed.length} failed** / ${checks.length}`);
  return { text: lines.join("\n"), checks };
}
