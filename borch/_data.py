"""A piece of borch, split out. __init__ gathers the public names."""

import math as _math

import numpy as _np

from ._tensor import (
    Tensor,
)
from ._ops import (
    _Namespace, _rng, _wrap, as_tensor, stack,
)
from ._base import (
    _like_torch,
    _np,
)

# ================================================================ utils.data

class Dataset:
    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, i):
        raise NotImplementedError


class TensorDataset(Dataset):
    def __init__(self, *tensors):
        self.tensors = tensors

    def __len__(self):
        return len(self.tensors[0])

    def __getitem__(self, i):
        return tuple(t[i] for t in self.tensors)


class SequentialSampler:
    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        return iter(range(len(self.data_source)))

    def __len__(self):
        return len(self.data_source)


class WeightedRandomSampler:
    """Draws the rare ones more often. With 10 patients in 1000 it stops a batch
    from containing no patient at all — what chapter 5 teaches."""

    def __init__(self, weights, num_samples, replacement=True, generator=None):
        self.weights = _np.asarray(
            [float(w) for w in (weights.tolist() if isinstance(weights, Tensor) else weights)])
        self.num_samples = num_samples
        self.replacement = replacement
        self.generator = generator

    def __iter__(self):
        rng = self.generator.rng() if self.generator is not None else _rng
        p = self.weights / self.weights.sum()
        return iter(rng.choice(len(p), size=self.num_samples,
                               replace=self.replacement, p=p).tolist())

    def __len__(self):
        return self.num_samples


class RandomSampler:
    def __init__(self, data_source, replacement=False, num_samples=None,
                 generator=None):
        """torch's list. `generator` is the one that matters here: without it a
        shuffled loader cannot be made to repeat, and `WeightedRandomSampler` next
        door has taken one all along."""
        self.data_source = data_source
        self.replacement = replacement
        self._num_samples = num_samples
        self.generator = generator

    def __iter__(self):
        rng = self.generator.rng() if self.generator is not None else _rng
        n = len(self.data_source)
        if self.replacement:
            return iter(rng.integers(0, n, size=len(self)).tolist())
        return iter(rng.permutation(n).tolist()[:len(self)])

    def __len__(self):
        return (len(self.data_source) if self._num_samples is None
                else self._num_samples)


class DistributedSampler:
    """Every rank takes **an interleaved slice** of the same shuffled order.

    Its row read *for distributed training — this is inside one tab*, which names what
    the class is **for** rather than what it needs. Measured: given `num_replicas` and
    `rank` it never touches `torch.distributed` at all —
    `DistributedSampler(range(10), num_replicas=2, rank=0)` answers `[0, 2, 4, 6, 8]`.
    It is arithmetic on indices, and a process group is only where the two numbers come
    from when nobody passes them.

    Three things it has to get right, and none of them is a network:

    - **The slice is `rank::num_replicas`, interleaved and not contiguous.** Cut into
      blocks instead, each rank sees one region of a sorted dataset and the batches stop
      being a sample of it. Nothing raises and the loss curve is the only witness.
    - **The order is redrawn per epoch, from `seed + epoch`.** Every rank draws the same
      permutation because they draw from the same seed, which is what makes the
      interleave a partition. Left at one seed forever, every epoch sees the same order.
    - **Short of a whole multiple it pads by wrapping**, or drops with `drop_last`. The
      ranks have to receive the same count or the one that runs out ends the epoch for
      everybody, and the padding is the front of the same list rather than a repeat of
      the tail.

    **`set_epoch` is not decoration.** A loader that never calls it draws the same
    permutation every epoch, which is the failure this class is most often used wrong
    for — and it is invisible except as a model that stops improving early.

    **Shuffled, the indices are not torch's.** torch draws with `torch.randperm` and a
    `torch.Generator`; this draws with numpy's, and the same seed gives a different
    permutation — measured. Unshuffled the two agree exactly, so that is what the golden
    freezes; what shuffling has to get right is a **property** and not a value, and the
    properties are asked separately: the ranks partition the padded list, one epoch
    repeats, and two epochs differ.
    """

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, seed=0,
                 drop_last=False):
        if num_replicas is None or rank is None:
            # **`ValueError`, not `RuntimeError`, and the difference is not ours.**
            # torch has both here: `RuntimeError` when the distributed package cannot
            # be imported at all, and `ValueError` from `get_world_size` when it can
            # and no process group has been started. On any ordinary install the second
            # is the one a caller meets — measured, including on a CPU-only build where
            # `dist.is_available()` is still True.
            raise ValueError(_like_torch(
                "`num_replicas` and `rank` have to be given. There is one process "
                "here, so there is nothing to read them from — everything after them "
                "is arithmetic and works.",
                "Default process group has not been initialized, please make sure to "
                "call init_process_group"))
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                f"Invalid rank {rank}, rank should be in the interval "
                f"[0, {num_replicas - 1}]")
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        count = len(dataset)
        if drop_last and count % num_replicas != 0:
            self.num_samples = -(-(count - num_replicas) // num_replicas)
        else:
            self.num_samples = -(-count // num_replicas)
        self.total_size = self.num_samples * num_replicas
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        count = len(self.dataset)
        if self.shuffle:
            # **The generator is made here and seeded here**, not taken from the module
            # — every rank has to draw the same permutation for the interleave to be a
            # partition, and a shared stream would give each of them a different one.
            rng = _np.random.default_rng(self.seed + self.epoch)
            indices = rng.permutation(count).tolist()
        else:
            indices = list(range(count))
        if not self.drop_last:
            padding = self.total_size - len(indices)
            if padding <= len(indices):
                indices += indices[:padding]
            else:
                indices += (indices * (-(-padding // len(indices))))[:padding]
        else:
            indices = indices[:self.total_size]
        indices = indices[self.rank:self.total_size:self.num_replicas]
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        """Which epoch's order to draw. See the note above on why it matters."""
        self.epoch = epoch


class DataLoader:
    def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None,
                 batch_sampler=None, num_workers=0, collate_fn=None,
                 pin_memory=False, drop_last=False, timeout=0,
                 worker_init_fn=None, multiprocessing_context=None,
                 generator=None, *, prefetch_factor=None,
                 persistent_workers=False, pin_memory_device="",
                 in_order=True):
        """**torch's list, and it was seven names of seventeen with two of them in
        the wrong seats.** `collate_fn` is torch's seventh and `drop_last` its
        ninth; here they were sixth and seventh, so
        `DataLoader(ds, 4, False, None, 0, True)` set `drop_last` on this side and
        `collate_fn` on torch's — a boolean into a callable's slot, which torch then
        tries to call.

        The rest divide three ways and the division is measured, not assumed.

        **`generator` and `batch_sampler` are real and are implemented.** A shuffled
        loader with no generator cannot be made to repeat, which is the first thing
        anyone wants from a dataset; `batch_sampler` hands out whole batches of
        indices and replaces `batch_size`, `shuffle`, `sampler` and `drop_last` at
        once, as it does in torch.

        **`pin_memory` and `pin_memory_device` are accepted and ignored.** They ask
        for page-locked host memory so a copy to a GPU can be asynchronous. There is
        no device to copy to here, and **no value changes either way** — the same
        standing as `foreach` on the optimizers.

        **The worker settings are refused, in torch's own words where torch has
        them.** `num_workers` is accepted and runs in one process (a browser has no
        fork); `prefetch_factor` and `persistent_workers` torch itself refuses when
        `num_workers` is 0, and that is always true here, so the refusal is torch's
        rather than ours. `timeout`, `worker_init_fn`, `multiprocessing_context` and
        `in_order` mean nothing without workers and say so.
        """
        if sampler is not None and shuffle:
            raise ValueError("sampler and shuffle cannot be used together.")
        if batch_sampler is not None and (shuffle or sampler is not None
                                          or drop_last or batch_size != 1):
            raise ValueError(
                "batch_sampler is mutually exclusive with batch_size, shuffle, "
                "sampler and drop_last.")
        if prefetch_factor is not None:
            raise ValueError(
                "prefetch_factor option could only be specified in multiprocessing."
                "let num_workers > 0 to enable multiprocessing, and it is always 0 "
                "here — a browser has no fork.")
        if persistent_workers:
            raise ValueError("persistent_workers option needs num_workers > 0")
        for name, given in (("timeout", timeout), ("worker_init_fn", worker_init_fn),
                            ("multiprocessing_context", multiprocessing_context)):
            if given:
                raise ValueError(f"{name} needs num_workers > 0")
        if not in_order:
            raise ValueError("in_order=False needs num_workers > 0")
        del pin_memory, pin_memory_device   # no device to pin for; no value changes
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.collate_fn = collate_fn
        self.batch_sampler = batch_sampler
        self.generator = generator
        if isinstance(dataset, IterableDataset):
            # An iterable dataset has neither a length nor indices. Asked to
            # shuffle, it stops here — running on quietly unshuffled is worse.
            if shuffle or sampler is not None:
                raise ValueError("IterableDataset cannot be shuffled — it has no indices.")
            self.sampler = None
        else:
            self.sampler = sampler or (
                RandomSampler(dataset, generator=generator) if shuffle
                else SequentialSampler(dataset))

    def __iter__(self):
        # **An iterable dataset uses no indices.** There are none for a sampler
        # to produce, so it is simply streamed.
        if self.batch_sampler is not None:
            for idx in self.batch_sampler:
                yield self._collate([self.dataset[i] for i in idx])
            return
        source = (self.dataset if isinstance(self.dataset, IterableDataset)
                  else (self.dataset[i] for i in self.sampler))
        batch = []
        for item in source:
            batch.append(item)
            if len(batch) == self.batch_size:
                yield self._collate(batch)
                batch = []
        if batch and not self.drop_last:
            yield self._collate(batch)

    def __len__(self):
        if self.batch_sampler is not None:
            return len(self.batch_sampler)
        n = len(self.sampler)
        return n // self.batch_size if self.drop_last else -(-n // self.batch_size)

    def _collate(self, batch):
        """**Left to `default_collate`.**

        The `zip(*batch)` that used to be here assumed a sample is always a
        tuple. True looking at `TensorDataset` alone, but an `IterableDataset`
        may stream bare tensors, and then `zip` tries to iterate the tensor
        itself and dies with `len() of unsized object` — two places away from
        the cause.
        """
        return self.collate_fn(batch) if self.collate_fn else default_collate(batch)


class ConcatDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = list(datasets)

    def __len__(self):
        return sum(len(d) for d in self.datasets)

    def __getitem__(self, i):
        for d in self.datasets:
            if i < len(d):
                return d[i]
            i -= len(d)
        raise IndexError(i)


def random_split(dataset, lengths, generator=None):
    rng = generator.rng() if generator is not None else _rng
    idx = rng.permutation(len(dataset)).tolist()
    out, start = [], 0
    for n in lengths:
        out.append(Subset(dataset, idx[start:start + n]))
        start += n
    return out


class Subset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        return self.dataset[self.indices[i]]


class Sampler:
    """The base of a sampler. **Without it there is no writing your own.**

    `RandomSampler` and `SequentialSampler` already existed and this base class
    did not. That closes off deciding "draw the indices in this order" entirely —
    unlike adding another layer type, this is the kind with **no way around it.**

    Filling in `__iter__` alone is enough for `DataLoader` to accept it. That is
    torch's contract too.

    **It does not take `data_source`.** torch used to and no longer does —
    `super().__init__(source)` raises `TypeError` over there. Accepting it here
    creates code that runs only against us, and that is the thing this library
    must least do.
    """

    def __iter__(self):
        raise NotImplementedError("Sampler must implement `__iter__`")


class SubsetRandomSampler(Sampler):
    """Shuffles within the given indices only. What each fold of a
    cross-validation uses."""

    def __init__(self, indices, generator=None):
        self.indices = list(indices)
        self.generator = generator

    def __iter__(self):
        rng = self.generator.rng() if self.generator is not None else _rng
        return iter([self.indices[i] for i in rng.permutation(len(self.indices))])

    def __len__(self):
        return len(self.indices)


class BatchSampler(Sampler):
    """Hands out indices **in groups.** For deciding the batch size on the
    sampler side."""

    def __init__(self, sampler, batch_size, drop_last=False):
        self.sampler, self.batch_size, self.drop_last = sampler, batch_size, drop_last

    def __iter__(self):
        batch = []
        for i in self.sampler:
            batch.append(i)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if batch and not self.drop_last:
            yield batch

    def __len__(self):
        n = len(self.sampler)
        return n // self.batch_size if self.drop_last else -(-n // self.batch_size)


class IterableDataset(Dataset):
    """A dataset that is **streamed** rather than indexed.

    Things with no known end (a stream, a large file) take this shape. With
    neither `__len__` nor `__getitem__`, `DataLoader` cannot shuffle it — asked
    to, it stops right there.
    """

    def __iter__(self):
        raise NotImplementedError("IterableDataset must implement `__iter__`")

    def __getitem__(self, i):
        raise TypeError("IterableDataset cannot be indexed — it has to be iterated")


class ChainDataset(IterableDataset):
    """**Concatenates** streamed ones. The iterable counterpart of
    `ConcatDataset`."""

    def __init__(self, datasets):
        self.datasets = list(datasets)

    def __iter__(self):
        for d in self.datasets:
            yield from d


class StackDataset(Dataset):
    """Zips several equal-length datasets **side by side.** Taking one out gives
    a tuple."""

    def __init__(self, *datasets, **named):
        if datasets and named:
            raise ValueError("give them positionally or by name, not both")
        self.datasets = named or datasets

    def __len__(self):
        got = self.datasets.values() if isinstance(self.datasets, dict) else self.datasets
        return min(len(d) for d in got)

    def __getitem__(self, i):
        if isinstance(self.datasets, dict):
            return {k: d[i] for k, d in self.datasets.items()}
        return tuple(d[i] for d in self.datasets)


def default_collate(batch):
    """Folds a list of samples into a batch. **Names what `DataLoader` was doing
    without saying so.**

    There is a reason to call it directly — using your own `collate_fn` and
    wanting part of it folded the default way.
    """
    first = batch[0]
    if isinstance(first, (tuple, list)):
        return type(first)(default_collate([b[i] for b in batch])
                           for i in range(len(first)))
    if isinstance(first, dict):
        return {k: default_collate([b[k] for b in batch]) for k in first}
    if isinstance(first, Tensor):
        return stack(list(batch))
    return stack([_wrap(b) for b in batch])


def default_convert(data):
    """Converts numpy to tensors and **leaves the rest alone.** The partner to
    `default_collate`.

    This is where one element becomes a tensor before folding. Code with its own
    `collate_fn` calls this first and folds the rest itself.

    Two traps. Both were checked by running real torch
    (`tests/probe_data.py`).

    - **A tuple becomes a list.** `(a, b)` goes in and `[tensor, tensor]` comes
      out — backward compatibility torch left itself. Only a namedtuple keeps its
      type.
    - **Python numbers are untouched.** `3` comes out as `3`. `default_collate`
      folds numbers into tensors and this one does not — the similar names
      suggest they match, and they do not.
    """
    if isinstance(data, Tensor):
        return data
    if isinstance(data, (_np.ndarray, _np.generic)):
        return as_tensor(data)
    if isinstance(data, dict):
        made = {k: default_convert(v) for k, v in data.items()}
        try:
            return type(data)(made)
        except TypeError:                  # the ones whose constructor takes no dict
            return made
    if isinstance(data, tuple):
        if hasattr(data, "_fields"):       # a namedtuple has field names
            return type(data)(*(default_convert(d) for d in data))
        return [default_convert(d) for d in data]
    if isinstance(data, list):
        made = [default_convert(d) for d in data]
        try:
            return type(data)(made)
        except TypeError:
            return made
    return data


def get_worker_info():
    """**Always `None`** — there are no worker processes here.

    In torch it is `None` in the main process too, and gives information only
    inside a worker started with `num_workers>0`. This `DataLoader` accepts
    `num_workers` and runs in one process — a browser has no fork, and the core
    has to give the same answer as that side.

    So this answer is a fact rather than an imitation: this is not a worker. Code
    where an `IterableDataset` asks this to split its shards branches at `None`
    into "do it all alone", and that is exactly the right branch here.
    """
    return None


class _UtilsData(_Namespace):
    Dataset = Dataset
    TensorDataset = TensorDataset
    ConcatDataset = ConcatDataset
    Subset = Subset
    DataLoader = DataLoader
    RandomSampler = RandomSampler
    SequentialSampler = SequentialSampler
    WeightedRandomSampler = WeightedRandomSampler
    Sampler = Sampler
    SubsetRandomSampler = SubsetRandomSampler
    DistributedSampler = DistributedSampler
    BatchSampler = BatchSampler
    IterableDataset = IterableDataset
    ChainDataset = ChainDataset
    StackDataset = StackDataset
    default_collate = staticmethod(default_collate)
    default_convert = staticmethod(default_convert)
    get_worker_info = staticmethod(get_worker_info)
    random_split = staticmethod(random_split)


class _Utils(_Namespace):
    data = _UtilsData()


utils = _Utils()


