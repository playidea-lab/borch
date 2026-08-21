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
import { uniform } from "./random.js";
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
      // **여기서 안 막으면 배치가 조용히 어긋난다.** 표본 수가 다른 두 텐서를 묶으면
      // 짧은 쪽이 범위를 넘고, 그것은 학습이 도는 채로 라벨만 밀리는 자리다.
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
    // **이어진 자리는 번호표를 안 만든다.** 안 섞은 적재기가 그 경우이고, 그때
    // `narrow` 는 색인 텐서를 올리는 일 없이 잘라 준다.
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
      : stackItems(mapped.map((i) => this.dataset.get(i)));
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
  dataset: Dataset, lengths: readonly number[],
): Subset[] {
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

  constructor(readonly dataset: Dataset, options: LoaderOptions = {}) {
    this.batchSize = options.batchSize ?? 1;
    this.shuffle = options.shuffle ?? false;
    this.dropLast = options.dropLast ?? false;
    if (this.batchSize < 1 || !Number.isInteger(this.batchSize)) {
      throw new RuntimeError(`batchSize must be a positive integer: ${this.batchSize}`);
    }
  }

  get length(): number {
    const n = this.dataset.length;
    return this.dropLast
      ? Math.floor(n / this.batchSize)
      : Math.ceil(n / this.batchSize);
  }

  [Symbol.iterator](): Iterator<readonly Tensor[]> {
    // **에폭마다 새로 섞는다.** 순서를 생성자에서 한 번만 정하면 두 번째 에폭이
    // 첫 번째와 같은 차례로 돌고, 섞는 이유가 사라진다.
    const order = this.shuffle
      ? shuffled(this.dataset.length)
      : Array.from({ length: this.dataset.length }, (_, i) => i);
    const batches = this.length;
    let at = 0;

    return {
      next: (): IteratorResult<readonly Tensor[]> => {
        if (at >= batches) return { done: true, value: undefined };
        const from = at * this.batchSize;
        const picks = order.slice(from, from + this.batchSize);
        at += 1;
        const batch = this.dataset.gather
          ? this.dataset.gather(picks)
          : stackItems(picks.map((i) => this.dataset.get(i)));
        return { done: false, value: batch };
      },
    };
  }
}

/**
 * 표본 여럿을 배치 하나로. `default_collate` 자리다.
 *
 * 자리마다 따로 쌓는다 — `[[x₀, y₀], [x₁, y₁]]` 이 `[X, Y]` 가 된다.
 */
function stackItems(items: readonly (readonly Tensor[])[]): readonly Tensor[] {
  const first = items[0];
  if (!first) throw new RuntimeError("cannot collate an empty batch");
  const width = first.length;
  for (const item of items) {
    if (item.length !== width) {
      throw new RuntimeError(
        `samples disagree on tensor count: ${item.length} and ${width}`,
      );
    }
  }
  return Array.from({ length: width }, (_, slot) =>
    Tensor.stack(items.map((item) => item[slot] as Tensor), 0));
}

/** `0..n-1` 을 섞은 번호. Fisher–Yates, 호스트 줄기(`manualSeed` 를 따른다). */
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
