"""Compares `utils.data` against real torch **by behaviour.**

## Why here rather than in the golden cases

The golden cases are where three implementations are checked for **the same numbers.**
`utils.data` is not numbers but Python structure — the contract is subclassing `Sampler` and
filling in `__iter__`, and borch.ts has no counterpart (a TypeScript library has no reason to
imitate a Python DataLoader).

Put into the golden cases, they leave entries borch.ts can never fill, and then "N not asked
about" never returns to zero. That number counts **work not yet done**, not work that cannot
be done.

So it comes to native pytest, where real torch can be called directly.

## What is asked

Not values but **order, shape and contract.** Where randomness is involved, as in
`RandomSampler`, the values cannot match, so the **set** of drawn indices and their count are
what is looked at.

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
    """Stands up a `TensorDataset` on both sides from the same data."""
    return lib.utils.data.TensorDataset(lib.tensor(_X), lib.tensor(_Y))


def test_Sampler_is_a_base_class_you_can_subclass_in_both():
    """**Without this there is no way at all to write your own sampler.**

    A missing layer can be stood in for by another layer, but the only route to saying "draw
    the indices in this order" is this one contract.
    """
    def build(lib):
        class Backwards(lib.utils.data.Sampler):
            def __init__(self, source):
                # **`super().__init__()` takes no argument.** torch removed the
                # `data_source` it used to accept, and passing one raises `TypeError` there.
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
    """Randomness is involved, so it looks at **the set rather than the order.**"""
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
    """**Stopping beats running quietly without shuffling.**

    torch throws in the same place. The wording differs and the fact of refusing is the
    contract.
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
    """What `DataLoader` was doing without saying so. It needs a name to be callable from inside your own `collate_fn`."""
    def build(lib):
        samples = [(lib.tensor(_X[i]), lib.tensor(_Y[i])) for i in range(3)]
        xs, ys = lib.utils.data.default_collate(samples)
        return list(xs.shape), list(ys.shape)

    assert build(bt) == build(real) == ([3, 2], [3])
