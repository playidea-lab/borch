"""`utils.data` 를 진짜 torch 와 **행동으로** 대조한다.

## 왜 골든이 아니라 여기인가

골든은 세 구현이 **같은 수**를 내는지 보는 자리다. `utils.data` 는 수가 아니라
파이썬 구조다 — `Sampler` 를 상속해서 `__iter__` 를 채우는 것이 계약이고, borch.ts
에는 대응이 없다(TypeScript 라이브러리가 파이썬 DataLoader 를 흉내 낼 이유가 없다).

골든에 넣으면 borch.ts 가 영영 못 채우는 케이스가 남고, 그러면 "안 물은 것 N 건" 이
0 으로 안 돌아온다. 그 수는 **아직 안 한 일**을 세는 자리이지 못 하는 일을 세는
자리가 아니다.

그래서 네이티브 pytest 로 온다. 여기서는 진짜 torch 를 바로 부를 수 있다.

## 무엇을 묻는가

값이 아니라 **순서와 모양과 계약**이다. `RandomSampler` 처럼 난수가 끼는 것은 값이
같을 수 없으므로, 뽑힌 번호의 **집합**과 개수를 본다.

    uv run --with pytest --with numpy --with torch pytest tests/test_data.py
"""

import pathlib
import sys

import numpy as np
import pytest
import torch as real

_root = pathlib.Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import borch as bt                                            # noqa: E402

_X = np.arange(24, dtype=np.float32).reshape(12, 2)
_Y = np.arange(12, dtype=np.int64)


def _pair(lib):
    """양쪽에 같은 데이터로 `TensorDataset` 을 세운다."""
    return lib.utils.data.TensorDataset(lib.tensor(_X), lib.tensor(_Y))


def test_Sampler_is_a_base_class_you_can_subclass_in_both():
    """**이것이 없으면 자기 샘플러를 만들 방법이 아예 없다.**

    층이 없는 것은 다른 층으로 대신할 수 있지만, "번호를 이 순서로 뽑아라" 를 정하는
    길은 이 계약 하나뿐이다.
    """
    def build(lib):
        class Backwards(lib.utils.data.Sampler):
            def __init__(self, source):
                # **`super().__init__()` 에 인자를 안 준다.** torch 가 예전에 받던
                # `data_source` 를 지웠고, 주면 그쪽에서 `TypeError` 가 난다.
                super().__init__()
                self.n = len(source)

            def __iter__(self):
                return iter(range(self.n - 1, -1, -1))

            def __len__(self):
                return self.n

        loader = lib.utils.data.DataLoader(_pair(lib), batch_size=4,
                                           sampler=Backwards(_pair(lib)))
        return [int(y) for _, ys in loader for y in ys.tolist()]

    assert build(bt) == build(real) == list(range(11, -1, -1))


def test_BatchSampler_groups_indices_the_same_way():
    def build(lib):
        base = lib.utils.data.SequentialSampler(_pair(lib))
        return [list(b) for b in lib.utils.data.BatchSampler(base, 5, drop_last=False)]

    assert build(bt) == build(real)
    assert build(bt) == [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11]]


def test_BatchSampler_drop_last_drops_the_short_one():
    def build(lib):
        base = lib.utils.data.SequentialSampler(_pair(lib))
        return [list(b) for b in lib.utils.data.BatchSampler(base, 5, drop_last=True)]

    assert build(bt) == build(real)
    assert len(build(bt)) == 2


def test_SubsetRandomSampler_stays_inside_the_given_indices():
    """난수가 끼므로 **순서가 아니라 집합**을 본다."""
    picked = [3, 5, 7, 9]

    def build(lib):
        return sorted(lib.utils.data.SubsetRandomSampler(picked))

    assert build(bt) == build(real) == picked


def test_IterableDataset_streams_instead_of_indexing():
    def build(lib):
        class Counting(lib.utils.data.IterableDataset):
            def __iter__(self):
                return iter([lib.tensor(np.float32(i)) for i in range(7)])

        loader = lib.utils.data.DataLoader(Counting(), batch_size=3)
        return [b.tolist() for b in loader]

    assert build(bt) == build(real) == [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0]]


def test_IterableDataset_refuses_to_shuffle_because_it_has_no_indices():
    """**조용히 안 섞고 도는 것보다 멈추는 편이 낫다.**

    torch 도 같은 자리에서 던진다. 문구는 다르지만 거절한다는 사실이 계약이다.
    """
    class Empty(bt.utils.data.IterableDataset):
        def __iter__(self):
            return iter([])

    with pytest.raises(ValueError):
        bt.utils.data.DataLoader(Empty(), batch_size=2, shuffle=True)


def test_ChainDataset_joins_streams_end_to_end():
    def build(lib):
        class Fixed(lib.utils.data.IterableDataset):
            def __init__(self, values):
                self.values = values

            def __iter__(self):
                return iter([lib.tensor(np.float32(v)) for v in self.values])

        chained = lib.utils.data.ChainDataset([Fixed([1, 2]), Fixed([3])])
        return [float(v) for v in chained]

    assert build(bt) == build(real) == [1.0, 2.0, 3.0]


def test_StackDataset_puts_two_datasets_side_by_side():
    def build(lib):
        a = lib.utils.data.TensorDataset(lib.tensor(_X))
        b = lib.utils.data.TensorDataset(lib.tensor(_Y))
        stacked = lib.utils.data.StackDataset(a, b)
        first = stacked[0]
        return len(stacked), len(first)

    assert build(bt) == build(real) == (12, 2)


def test_default_collate_folds_a_list_of_samples_into_a_batch():
    """`DataLoader` 가 말없이 하던 일. 이름이 있어야 자기 `collate_fn` 안에서 부른다."""
    def build(lib):
        samples = [(lib.tensor(_X[i]), lib.tensor(_Y[i])) for i in range(3)]
        xs, ys = lib.utils.data.default_collate(samples)
        return list(xs.shape), list(ys.shape)

    assert build(bt) == build(real) == ([3, 2], [3])
