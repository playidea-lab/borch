/**
 * 데이터셋과 적재기 — `torch.utils.data` 자리.
 *
 * ## 없으면 이야기가 반쯤 거짓이 된다
 *
 * 이 저장소는 "브라우저에서 학습한다" 고 말하는데, 여기가 비어 있으면 그 학습이
 * **배치를 손으로 자르는 것**으로 끝난다. 실제로 저장소 안의 정확도 러너가 뽑기·섞기를
 * 직접 짜고 있었다 — 쓰는 사람은 누구나 같은 것을 다시 짜게 된다.
 *
 * ## 무엇을 안 만들었나
 *
 * **`sampler` 옵션이 없다.** torch 에 있는 이름이지만 지금 받쳐 줄 것이 없다. 이름만
 * 놓으면 `paramGroups` 가 그랬던 꼴이 된다 — 모양은 torch 인데 속이 비어서, 쓰는
 * 사람이 넣은 것이 조용히 무시되는 자리. 없는 것은 없다고 둔다.
 *
 * `num_workers` 도 없다. 워커를 띄우면 GPU 손잡이가 그쪽으로 안 건너간다.
 *
 * ## torch 와 갈리는 두 자리
 *
 * **배치는 GPU 텐서라 `scope()` 안에서 받아야 한다.** 적재기가 대신 감쌀 수가 없다 —
 * 텐서가 구역 밖으로 나가는 것이 목적이기 때문이다. 안 감싸면 한 에폭이 만드는 배치가
 * 전부 남는다.
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
 * `x` 와 `y` 는 구역 **밖에서** 만들어졌으므로 살려 둘 것으로 표시할 필요가 없다.
 * 구역이 놓는 것은 그 안에서 만든 중간 버퍼다.
 *
 * **섞기가 `manualSeed` 를 따른다.** torch 는 DataLoader 에 별도 generator 를 두는데
 * 여기서는 호스트 줄기 하나를 쓴다(`random.ts`) — 씨앗 하나가 층 초기화·dropout·
 * 텐서 팩토리에 이어 배치 순서까지 되돌린다. 문을 늘리지 않는 쪽을 골랐다.
 */

import { RuntimeError } from "./errors.js";
import { uniform } from "./random.js";
import { Tensor } from "./tensor.js";

/**
 * 번호로 꺼낼 수 있는 것. `torch.utils.data.Dataset` 자리다.
 *
 * `gather` 는 **있어도 되고 없어도 된다.** 있으면 적재기가 배치를 한 번에 뽑고,
 * 없으면 하나씩 꺼내 쌓는다. 텐서를 들고 있는 데이터셋에서 이 차이가 크다 — 배치
 * 32 에 텐서 둘이면 GPU 연산이 66 번에서 2 번으로 준다.
 */
export interface Dataset {
  readonly length: number;
  get(index: number): readonly Tensor[];
  gather?(indices: readonly number[]): readonly Tensor[];
}

/** 첫 축이 표본 축인 텐서 몇 개. `torch.utils.data.TensorDataset` 자리다. */
export class TensorDataset implements Dataset {
  readonly tensors: readonly Tensor[];

  constructor(...tensors: Tensor[]) {
    if (tensors.length === 0) {
      throw new RuntimeError("TensorDataset 에 텐서가 하나는 있어야 한다");
    }
    const rows = tensors[0]?.shape[0] ?? 0;
    for (const [i, t] of tensors.entries()) {
      // **여기서 안 막으면 배치가 조용히 어긋난다.** 표본 수가 다른 두 텐서를 묶으면
      // 짧은 쪽이 범위를 넘고, 그것은 학습이 도는 채로 라벨만 밀리는 자리다.
      if (t.shape[0] !== rows) {
        throw new RuntimeError(
          `TensorDataset 의 텐서들이 표본 수가 다르다: ${i} 번은 ${t.shape[0]}, ` +
            `0 번은 ${rows}`,
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

/** 원본의 일부만 본다. `randomSplit` 이 이것을 낸다. */
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
   * **번호를 옮겨 원본에 넘긴다.** 이것이 없으면 `randomSplit` 을 지난 데이터셋이
   * 전부 느린 길로 떨어진다 — 학습·검증을 나누는 것이 예사로운 일인데.
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
      throw new RuntimeError(`Subset 의 번호가 범위를 넘는다: ${index}`);
    }
    return mapped;
  }
}

/** 여러 데이터셋을 이어 하나로. `torch.utils.data.ConcatDataset` 자리다. */
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
    throw new RuntimeError(`ConcatDataset 의 번호가 범위를 넘는다: ${index}`);
  }
}

/**
 * 겹치지 않게 나눈다. `torch.utils.data.random_split` 자리다.
 *
 * **합이 전체와 같아야 한다.** torch 도 거기서 멈춘다 — 남는 표본이 조용히 버려지면
 * 학습·검증 비율이 적어 놓은 것과 달라지고 아무도 못 본다.
 *
 * 섞기는 `manualSeed` 를 따른다. 같은 씨앗이면 같은 나눔이다.
 */
export function randomSplit(
  dataset: Dataset, lengths: readonly number[],
): Subset[] {
  const total = lengths.reduce((a, b) => a + b, 0);
  if (total !== dataset.length) {
    throw new RuntimeError(
      `나눈 길이의 합 ${total} 이 데이터셋 길이 ${dataset.length} 와 다르다`,
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
  /** 에폭마다 다시 섞는다 — torch 와 같다. */
  shuffle?: boolean;
  /** 마지막 배치가 모자라면 버린다. 배치 크기가 고정이어야 하는 층이 쓴다. */
  dropLast?: boolean;
}

/**
 * 배치를 내놓는다. `torch.utils.data.DataLoader` 자리다.
 *
 * **동기 반복자다.** 배치를 만드는 것은 GPU 연산이고 borch 의 순방향은 동기이므로
 * `for (const [x, y] of loader)` 가 그대로 돈다 — 값을 읽을 때만 비동기다.
 *
 * `length` 는 표본 수가 아니라 **배치 수**다. torch 와 같다.
 *
 * ## 배치는 `scope()` 안에서 받아야 한다
 *
 * 파일 머리에도 적었지만 여기 다시 적는다 — **로더를 쓰는 사람이 첫 루프에서 바로
 * 부딪히는 자리**이고, 문서 한 곳만 보고 코드를 쓰는 것이 보통이다.
 *
 * ```ts
 * for (const [x, y] of loader) {
 *   await scope(async () => {          // 이 안에서 만든 중간 버퍼가 나갈 때 놓인다
 *     opt.zeroGrad();
 *     const loss = crit.call(model.call(x), y);
 *     loss.backward();
 *     opt.step();
 *   });
 * }
 * ```
 *
 * 로더가 대신 감쌀 수는 없다. 배치가 구역 밖으로 나가는 것이 목적이기 때문이다 —
 * 감싸면 `x` 와 `y` 가 쓰이기도 전에 놓인다. 안 감싸면 한 에폭이 만드는 중간 버퍼가
 * 전부 남고, 몇 에폭 만에 장치가 찬다.
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
      throw new RuntimeError(`batchSize 는 1 이상의 정수여야 한다: ${this.batchSize}`);
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
  if (!first) throw new RuntimeError("빈 배치는 만들 수 없다");
  const width = first.length;
  for (const item of items) {
    if (item.length !== width) {
      throw new RuntimeError(
        `표본마다 텐서 수가 다르다: ${item.length} 과 ${width}`,
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
