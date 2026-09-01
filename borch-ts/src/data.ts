/**
 * Datasets and loaders — where `torch.utils.data` goes.
 *
 * ## Without this, the story is half untrue
 *
 * This repository says "train in the browser", and with this empty that
 * training ends at **slicing batches by hand.** The accuracy runner inside
 * the repository was in fact writing its own sampling and shuffling — and
 * every user would end up writing the same thing again.
 *
 * ## What was not built
 *
 * **There is no `sampler` option.** The name exists in torch, but there is
 * nothing behind it here yet. Putting the name down alone repeats what
 * happened with `paramGroups` — torch's shape with nothing inside, a place
 * where what the user passes is quietly ignored. What is not there is left
 * absent.
 *
 * There is no `num_workers` either. Spawning a worker does not carry the
 * GPU handle across to it.
 *
 * ## Two places it parts from torch
 *
 * **Batches are GPU tensors, so they have to be received inside
 * `scope()`.** The loader cannot wrap them for you — the whole point is
 * that the tensor leaves the scope. Leave it unwrapped and every batch an
 * epoch makes stays alive.
 *
 * ```ts
 * for (const [x, y] of loader) {
 *   await scope(async () => {
 *     opt.zeroGrad();
 *     const loss = crit.call(model.call(x), y);
 *     loss.backward();
 *     opt.step();
 *   });
 * }
 * ```
 *
 * `x` and `y` were made **outside** the scope, so they need no keep-alive
 * mark. What the scope releases are the intermediates made inside it.
 *
 * **Shuffling follows `manualSeed`.** torch gives the DataLoader its own
 * generator; here one host stream is used (`random.ts`) — one seed rewinds
 * layer initialisation, dropout, the tensor factories and now batch order
 * too. The choice was to not add another door.
 */

import { RuntimeError } from "./errors.js";
import { refuseGenerator, uniform } from "./random.js";
import { Tensor } from "./tensor.js";

/**
 * Something that can be fetched by index. Where `torch.utils.data.Dataset`
 * goes.
 *
 * `gather` **is optional.** With it, the loader pulls a batch in one go;
 * without it, it fetches one at a time and stacks. The difference is large
 * for a dataset holding tensors — at batch 32 with two tensors, GPU
 * operations drop from 66 to 2.
 */
export interface Dataset {
  readonly length: number;
  get(index: number): readonly Tensor[];
  gather?(indices: readonly number[]): readonly Tensor[];
}

/**
 * A few tensors whose first axis is the sample axis. Where
 * `torch.utils.data.TensorDataset` goes.
 */
export class TensorDataset implements Dataset {
  readonly tensors: readonly Tensor[];

  constructor(...tensors: Tensor[]) {
    if (tensors.length === 0) {
      throw new RuntimeError("TensorDataset needs at least one tensor");
    }
    const rows = tensors[0]?.shape[0] ?? 0;
    for (const [i, t] of tensors.entries()) {
      // **Unblocked here, the batches quietly go out of step.** Bundling two tensors
      // with different sample counts overruns the shorter one, and that is a place where
      // training runs while the labels alone are shifted.
      if (t.shape[0] !== rows) {
        throw new RuntimeError(
          `TensorDataset tensors disagree on sample count: index ${i} has ${t.shape[0]}, ` +
            `index 0 has ${rows}`,
        );
      }
    }
    this.tensors = [...tensors];
  }

  get length(): number {
    return this.tensors[0]?.shape[0] ?? 0;
  }

  get(index: number): readonly Tensor[] {
    return this.tensors.map((t) => t.select(0, index));
  }

  gather(indices: readonly number[]): readonly Tensor[] {
    // **A contiguous run builds no index table.** An unshuffled loader is that case, and
    // there `narrow` slices without uploading an index tensor.
    const first = indices[0] ?? 0;
    const contiguous = indices.every((v, i) => v === first + i);
    if (contiguous) {
      return this.tensors.map((t) => t.narrow(0, first, indices.length));
    }
    const picks = Tensor.from(indices, [indices.length], { dtype: "int64" });
    return this.tensors.map((t) => t.indexSelect(0, picks));
  }
}

/**
 * Sees only part of the original. `randomSplit` produces one.
 */
export class Subset implements Dataset {
  constructor(
    readonly dataset: Dataset,
    readonly indices: readonly number[],
  ) {}

  get length(): number {
    return this.indices.length;
  }

  get(index: number): readonly Tensor[] {
    return this.dataset.get(this.at(index));
  }

  /**
   * **Translates the indices and passes them to the original.** Without
   * this, every dataset that has been through `randomSplit` falls onto the
   * slow path — and splitting train from validation is an ordinary thing to
   * do.
   */
  gather(indices: readonly number[]): readonly Tensor[] {
    const mapped = indices.map((i) => this.at(i));
    return this.dataset.gather
      ? this.dataset.gather(mapped)
      : defaultCollate(mapped.map((i) => this.dataset.get(i)));
  }

  private at(index: number): number {
    const mapped = this.indices[index];
    if (mapped === undefined) {
      throw new RuntimeError(`Subset index out of range: ${index}`);
    }
    return mapped;
  }
}

/**
 * Several datasets joined into one. Where `torch.utils.data.ConcatDataset`
 * goes.
 */
export class ConcatDataset implements Dataset {
  private readonly ends: number[] = [];

  constructor(readonly datasets: readonly Dataset[]) {
    let total = 0;
    for (const d of datasets) {
      total += d.length;
      this.ends.push(total);
    }
  }

  get length(): number {
    return this.ends[this.ends.length - 1] ?? 0;
  }

  get(index: number): readonly Tensor[] {
    for (const [i, end] of this.ends.entries()) {
      if (index < end) {
        const start = i === 0 ? 0 : (this.ends[i - 1] as number);
        return (this.datasets[i] as Dataset).get(index - start);
      }
    }
    throw new RuntimeError(`ConcatDataset index out of range: ${index}`);
  }
}

/**
 * Splits without overlap. Where `torch.utils.data.random_split` goes.
 *
 * **The parts must sum to the whole.** torch stops there too — a leftover
 * sample quietly discarded makes the train/validation ratio differ from the
 * one written down, and nobody sees it.
 *
 * Shuffling follows `manualSeed`. The same seed gives the same split.
 */
export function randomSplit(
  dataset: Dataset, lengths: readonly number[], generator?: null,
): Subset[] {
  refuseGenerator("random_split", generator);
  const total = lengths.reduce((a, b) => a + b, 0);
  if (total !== dataset.length) {
    throw new RuntimeError(
      `Sum of input lengths ${total} does not equal the length of the input dataset ${dataset.length}`,
    );
  }
  const order = shuffled(dataset.length);
  const out: Subset[] = [];
  let at = 0;
  for (const n of lengths) {
    out.push(new Subset(dataset, order.slice(at, at + n)));
    at += n;
  }
  return out;
}


// ── Samplers ──────────────────────────────────────────────────────────────
//
// **These were absent, and the note at the top of this file was the reason.**
// It said: putting a `sampler` option down with nothing behind it repeats what
// happened with `paramGroups` — torch's shape, hollow inside, quietly ignoring
// what the caller passes. That argument is right and it is an argument against a
// *hollow* sampler, not against a sampler. So they are written rather than named.
//
// Nine names, and `tests/ts_axis.py` had all nine under `utils.data ✘ gap 12 ·
// without a reason 10` the whole time.

/**
 * An order over a dataset's indices. Where `torch.utils.data.Sampler` goes.
 *
 * torch's is a class other samplers inherit; here it is an interface, because
 * `for (const i of sampler)` is the only thing anything asks of one.
 */
export interface Sampler extends Iterable<number> {
  readonly length: number;
}

/** `0, 1, 2, …` in order. `torch.utils.data.SequentialSampler`. */
export class SequentialSampler implements Sampler {
  constructor(private readonly dataSource: { readonly length: number }) {}

  get length(): number {
    return this.dataSource.length;
  }

  *[Symbol.iterator](): Iterator<number> {
    for (let i = 0; i < this.dataSource.length; i++) yield i;
  }
}

/**
 * A shuffled order. `torch.utils.data.RandomSampler`.
 *
 * **With `replacement` it draws rather than permutes**, so an index can come up
 * twice and another not at all — and only then does `numSamples` mean anything
 * other than the dataset's length. torch refuses `numSamples` without
 * `replacement` for exactly that reason, and so does this.
 */
export class RandomSampler implements Sampler {
  constructor(
    private readonly dataSource: { readonly length: number },
    private readonly replacement = false,
    private readonly numSamples: number | null = null,
    generator?: null,
  ) {
    refuseGenerator("RandomSampler", generator);
    if (numSamples !== null && !replacement) {
      throw new RuntimeError(
        "numSamples should not be specified when replacement is false");
    }
  }

  get length(): number {
    return this.numSamples ?? this.dataSource.length;
  }

  *[Symbol.iterator](): Iterator<number> {
    const n = this.dataSource.length;
    if (!this.replacement) {
      yield* shuffled(n);
      return;
    }
    for (let i = 0; i < this.length; i++) {
      yield Math.floor(uniform() * n);
    }
  }
}

/** A shuffled order over the indices given. `torch.utils.data.SubsetRandomSampler`. */
export class SubsetRandomSampler implements Sampler {
  constructor(private readonly indices: readonly number[], generator?: null) {
    refuseGenerator("SubsetRandomSampler", generator);
  }

  get length(): number {
    return this.indices.length;
  }

  *[Symbol.iterator](): Iterator<number> {
    for (const at of shuffled(this.indices.length)) {
      yield this.indices[at] as number;
    }
  }
}

/**
 * Draws by weight. `torch.utils.data.WeightedRandomSampler` — the one that
 * makes an unbalanced dataset trainable.
 *
 * **The weights need not sum to 1**; torch normalises them, and so does this.
 * `replacement` defaults to **true** here, which is the opposite of
 * `RandomSampler`'s default and is torch's doing.
 */
export class WeightedRandomSampler implements Sampler {
  private readonly cumulative: number[];

  constructor(
    weights: readonly number[],
    readonly numSamples: number,
    replacement = true,
    generator?: null,
  ) {
    refuseGenerator("WeightedRandomSampler", generator);
    if (!replacement) {
      // Drawing without replacement by weight needs the weights renormalised after
      // every pick, and torch's own answer differs from the obvious one. Refused
      // rather than approximated: a sampler that draws a *nearly* right distribution
      // is the hardest kind of wrong to see, because the model still trains.
      throw new RuntimeError(
        "WeightedRandomSampler(replacement=false) is not here yet.");
    }
    let total = 0;
    this.cumulative = weights.map((w) => (total += w));
    if (total <= 0) {
      throw new RuntimeError("weights must sum to a positive number");
    }
  }

  get length(): number {
    return this.numSamples;
  }

  *[Symbol.iterator](): Iterator<number> {
    const total = this.cumulative[this.cumulative.length - 1] as number;
    for (let i = 0; i < this.numSamples; i++) {
      const hit = uniform() * total;
      let lo = 0;
      let hi = this.cumulative.length - 1;
      while (lo < hi) {
        const mid = (lo + hi) >> 1;
        if ((this.cumulative[mid] as number) <= hit) lo = mid + 1;
        else hi = mid;
      }
      yield lo;
    }
  }
}

/**
 * Groups another sampler's indices into batches. `torch.utils.data.BatchSampler`.
 *
 * This is what a `batchSampler` option is *for*: the batches come from here, so
 * `batchSize`, `shuffle` and `dropLast` are already decided and the loader must
 * not decide them again. torch refuses all three alongside it, and so does the
 * loader below.
 */
export class BatchSampler implements Iterable<readonly number[]> {
  constructor(
    private readonly sampler: Sampler,
    private readonly batchSize: number,
    private readonly dropLast = false,
  ) {
    if (batchSize < 1 || !Number.isInteger(batchSize)) {
      throw new RuntimeError(`batchSize must be a positive integer: ${batchSize}`);
    }
  }

  get length(): number {
    return this.dropLast
      ? Math.floor(this.sampler.length / this.batchSize)
      : Math.ceil(this.sampler.length / this.batchSize);
  }

  *[Symbol.iterator](): Iterator<readonly number[]> {
    let batch: number[] = [];
    for (const i of this.sampler) {
      batch.push(i);
      if (batch.length === this.batchSize) {
        yield batch;
        batch = [];
      }
    }
    if (batch.length && !this.dropLast) yield batch;
  }
}

/**
 * A dataset with no length and no index — you iterate it. Where
 * `torch.utils.data.IterableDataset` goes.
 *
 * **It cannot be shuffled and it cannot be sampled**, because both need to know
 * how many there are and where to reach. torch says the same by refusing
 * `shuffle` on one, which `DataLoader` below does too.
 */
export interface IterableDataset extends Iterable<readonly Tensor[]> {
  readonly iterable: true;
}

/** Several `IterableDataset`s end to end. `torch.utils.data.ChainDataset`. */
export class ChainDataset implements IterableDataset {
  readonly iterable = true as const;

  constructor(private readonly datasets: readonly IterableDataset[]) {}

  *[Symbol.iterator](): Iterator<readonly Tensor[]> {
    for (const d of this.datasets) yield* d;
  }
}

/**
 * Several datasets side by side, one sample from each. `torch.utils.data.StackDataset`.
 *
 * **Not `ConcatDataset`, which is end to end.** This one is across: three datasets
 * of 100 give 100 samples of three parts, where `ConcatDataset` gives 300 of one.
 */
export class StackDataset implements Dataset {
  constructor(private readonly datasets: readonly Dataset[]) {
    const first = datasets[0];
    if (!first) throw new RuntimeError("StackDataset needs at least one dataset");
    for (const d of datasets) {
      if (d.length !== first.length) {
        throw new RuntimeError(
          `StackDataset needs equal lengths: ${d.length} and ${first.length}`);
      }
    }
  }

  get length(): number {
    return this.datasets[0]?.length ?? 0;
  }

  get(index: number): readonly Tensor[] {
    return this.datasets.flatMap((d) => d.get(index));
  }
}

export interface LoaderOptions {
  batchSize?: number;
  /**
   * Reshuffles every epoch, as in torch.
   */
  shuffle?: boolean;
  /**
   * Drops the last batch if it is short. Layers that need a fixed batch
   * size use it.
   */
  dropLast?: boolean;
  /**
   * The order to visit the indices in. **Mutually exclusive with `shuffle`** —
   * a sampler already decides the order, and taking both means one of them is
   * quietly ignored. torch refuses the pair and so does this.
   */
  sampler?: Sampler;
  /**
   * Batches, already grouped. **Mutually exclusive with `batchSize`, `shuffle`,
   * `sampler` and `dropLast`** — all four are decisions the batch sampler has
   * already made, and accepting them here would let a caller set a `batchSize`
   * that does nothing.
   */
  batchSampler?: Iterable<readonly number[]> & { readonly length: number };
}

/**
 * Hands out batches. Where `torch.utils.data.DataLoader` goes.
 *
 * **It is a synchronous iterator.** Making a batch is a GPU operation and
 * borch's forward is synchronous, so `for (const [x, y] of loader)` works
 * as written — only reading values is asynchronous.
 *
 * `length` is the number of **batches**, not samples, as in torch.
 *
 * ## Batches have to be received inside `scope()`
 *
 * It is written at the head of the file too, and it is written again here —
 * this is **the place a loader's user hits on their first loop**, and
 * reading one spot of the documentation before writing code is the normal
 * case.
 *
 * ```ts
 * for (const [x, y] of loader) {
 *   await scope(async () => {          // intermediates made in here are released on the way out
 *     opt.zeroGrad();
 *     const loss = crit.call(model.call(x), y);
 *     loss.backward();
 *     opt.step();
 *   });
 * }
 * ```
 *
 * The loader cannot wrap it for you. The point is that the batch leaves the
 * scope — wrap it and `x` and `y` are released before they are ever used.
 * Leave it unwrapped and every intermediate an epoch makes stays, and the
 * device fills within a few epochs.
 */
export class DataLoader implements Iterable<readonly Tensor[]> {
  readonly batchSize: number;
  readonly shuffle: boolean;
  readonly dropLast: boolean;

  readonly sampler: Sampler | null;
  readonly batchSampler: (Iterable<readonly number[]> & { readonly length: number }) | null;

  constructor(readonly dataset: Dataset, options: LoaderOptions = {}) {
    this.batchSize = options.batchSize ?? 1;
    this.shuffle = options.shuffle ?? false;
    this.dropLast = options.dropLast ?? false;
    this.sampler = options.sampler ?? null;
    this.batchSampler = options.batchSampler ?? null;
    if (this.batchSize < 1 || !Number.isInteger(this.batchSize)) {
      throw new RuntimeError(`batchSize must be a positive integer: ${this.batchSize}`);
    }
    // **The refusals are the point of taking these at all.** A sampler beside a
    // `shuffle`, or a batch sampler beside a `batchSize`, means one of the two is
    // being ignored — and a loader that ignores an argument hands back batches that
    // look right. torch stops at the same pairs.
    if (this.sampler && options.shuffle !== undefined) {
      throw new RuntimeError("sampler option is mutually exclusive with shuffle");
    }
    if (this.batchSampler
        && (options.batchSize !== undefined || options.shuffle !== undefined
            || options.sampler !== undefined || options.dropLast !== undefined)) {
      throw new RuntimeError(
        "batchSampler option is mutually exclusive with batchSize, shuffle, "
          + "sampler, and dropLast");
    }
  }

  get length(): number {
    if (this.batchSampler) return this.batchSampler.length;
    const n = this.sampler ? this.sampler.length : this.dataset.length;
    return this.dropLast
      ? Math.floor(n / this.batchSize)
      : Math.ceil(n / this.batchSize);
  }

  [Symbol.iterator](): Iterator<readonly Tensor[]> {
    // **Reshuffled every epoch.** Deciding the order once in the constructor makes the
    // second epoch run in the first's order, and the reason for shuffling disappears.
    // A batch sampler decides the grouping too, so the slicing below is skipped
    // wholesale rather than fed a flattened order — its last batch may be short
    // where the others are not, and re-slicing would silently regularise it.
    const grouped: readonly (readonly number[])[] | null = this.batchSampler
      ? [...this.batchSampler]
      : null;
    const order = grouped
      ? []
      : this.sampler
        ? [...this.sampler]
        : this.shuffle
          ? shuffled(this.dataset.length)
          : Array.from({ length: this.dataset.length }, (_, i) => i);
    const batches = grouped ? grouped.length : this.length;
    let at = 0;

    return {
      next: (): IteratorResult<readonly Tensor[]> => {
        if (at >= batches) return { done: true, value: undefined };
        const from = at * this.batchSize;
        const picks = grouped
          ? (grouped[at] as readonly number[])
          : order.slice(from, from + this.batchSize);
        at += 1;
        const batch = this.dataset.gather
          ? this.dataset.gather(picks)
          : defaultCollate(picks.map((i) => this.dataset.get(i)));
        return { done: false, value: batch };
      },
    };
  }
}

/**
 * Several samples into one batch. `torch.utils.data.default_collate`.
 *
 * It stacks per position — `[[x₀, y₀], [x₁, y₁]]` becomes `[X, Y]`.
 *
 * **It was called `stackItems` and its own comment said "the place
 * `default_collate` occupies".** The function was here the whole time; only
 * torch's name was missing, and the name axis counts names — so it read as an
 * absent feature while the feature was three lines above the loader that used it.
 * A comment naming what something *would* be called is not the name.
 */
export function defaultCollate(
  batch: readonly (readonly Tensor[])[],
): readonly Tensor[] {
  const first = batch[0];
  if (!first) throw new RuntimeError("cannot collate an empty batch");
  const width = first.length;
  for (const item of batch) {
    if (item.length !== width) {
      throw new RuntimeError(
        `samples disagree on tensor count: ${item.length} and ${width}`,
      );
    }
  }
  return Array.from({ length: width }, (_, slot) =>
    Tensor.stack(batch.map((item) => item[slot] as Tensor), 0));
}

/** `0..n-1` shuffled. Fisher–Yates, on the host stream (it follows `manualSeed`). */
function shuffled(n: number): number[] {
  const order = Array.from({ length: n }, (_, i) => i);
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(uniform() * (i + 1));
    const tmp = order[i] as number;
    order[i] = order[j] as number;
    order[j] = tmp;
  }
  return order;
}

/**
 * Every rank takes **an interleaved slice** of the same shuffled order.
 * `torch.utils.data.DistributedSampler`.
 *
 * Its row said *for distributed training — this is inside one tab*, which names
 * what the class is **for** rather than what it needs. Given `numReplicas` and
 * `rank` it touches no network at all: it is arithmetic on indices, and a process
 * group is only where those two numbers come from when nobody passes them.
 *
 * Three things it has to get right, and none of them is a network:
 *
 * - **The slice is `rank`, `rank + numReplicas`, …** — interleaved, not a block.
 *   Cut into blocks instead, each rank sees one region of a sorted dataset and the
 *   batches stop being a sample of it. Nothing throws; the loss curve is the only
 *   witness.
 * - **The order is redrawn per epoch from `seed + epoch`**, and every rank draws
 *   the same permutation because they draw from the same seed — which is what
 *   makes the interleave a partition rather than an overlap.
 * - **Short of a whole multiple it pads by wrapping**, or drops with `dropLast`.
 *   The ranks must receive the same count, and the padding comes from the front of
 *   the same list rather than a repeat of the tail.
 *
 * **`setEpoch` is not decoration.** A loop that never calls it draws the same
 * order every epoch — invisible except as a model that stops improving early.
 *
 * The draw is **its own stream, seeded here**, not the module's: a shared stream
 * would hand each rank a different permutation and the partition would break. The
 * numbers it produces are not numpy's and not torch's, so what is frozen about the
 * shuffled case is a property (the ranks partition, one epoch repeats, two epochs
 * differ) rather than a value.
 */
export class DistributedSampler implements Sampler {
  private epoch = 0;
  readonly numSamples: number;
  readonly totalSize: number;

  constructor(
    private readonly dataset: { readonly length: number },
    private readonly numReplicas: number,
    private readonly rank: number,
    private readonly shuffle = true,
    private readonly seed = 0,
    private readonly dropLast = false,
  ) {
    if (rank >= numReplicas || rank < 0) {
      throw new RuntimeError(
        `Invalid rank ${rank}, rank should be in the interval `
        + `[0, ${numReplicas - 1}]`);
    }
    const count = dataset.length;
    this.numSamples = dropLast && count % numReplicas !== 0
      ? Math.ceil((count - numReplicas) / numReplicas)
      : Math.ceil(count / numReplicas);
    this.totalSize = this.numSamples * numReplicas;
  }

  get length(): number {
    return this.numSamples;
  }

  /** Which epoch's order to draw. See the note above on why it matters. */
  setEpoch(epoch: number): void {
    this.epoch = epoch;
  }

  *[Symbol.iterator](): Iterator<number> {
    const count = this.dataset.length;
    let order = Array.from({ length: count }, (_, i) => i);
    if (this.shuffle) {
      // MINSTD, seeded per epoch. The product stays under 2^53 so a float64
      // multiply is exact and every rank walks the identical sequence.
      let state = ((this.seed + this.epoch) % 2147483646) + 1;
      const next = (): number => {
        state = (state * 48271) % 2147483647;
        return state;
      };
      for (let i = count - 1; i > 0; i--) {
        const j = next() % (i + 1);
        [order[i], order[j]] = [order[j] as number, order[i] as number];
      }
    }
    if (!this.dropLast) {
      const padding = this.totalSize - order.length;
      order = padding <= order.length
        ? order.concat(order.slice(0, padding))
        : order.concat(
            Array.from({ length: padding }, (_, i) => order[i % order.length] as number));
    } else {
      order = order.slice(0, this.totalSize);
    }
    for (let i = this.rank; i < this.totalSize; i += this.numReplicas) {
      yield order[i] as number;
    }
  }
}
