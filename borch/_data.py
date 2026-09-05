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


class DistributedSampler(Sampler):
    """Hands each of `num_replicas` workers a different slice of one dataset.

    **It needs no distribution at all**, which is why it is here while the
    collectives are not. Given `num_replicas` and `rank` outright it never asks a
    process group anything — it reads that pair *from* one only when both are
    omitted, so here both are required and the omission raises rather than
    pretending there is a group of one. The class of that error is `ValueError`
    and the reason is under `__init__`; this paragraph said `RuntimeError` until
    the two disagreed by a hand's width, which is the distance that hides one.

    Two properties are what make it worth having over slicing by hand.

    **Every rank gets the same count.** Ten rows over three workers is not
    4/3/3 — the tail is padded from the front until the count divides, so the
    total becomes 12 and each rank takes 4, one of them twice. In torch a ragged
    split hangs the training loop at the collective the shortest rank never
    reaches; there is no collective here, so it would only skew the epoch, but
    the arithmetic is torch's either way. `drop_last=True` throws the tail away
    instead, and then ten over three is 3 each with the tenth row unseen.

    **The shuffle is seeded, not random.** Every rank must draw the *same*
    permutation or the ranks would overlap and miss rows between them, so the
    order comes from `seed + epoch` rather than from the global stream that
    `RandomSampler` next door uses. `set_epoch` is what moves it, and a loop
    that forgets to call it trains on one order forever — torch is silent about
    that too, and this reproduces the silence rather than improving on it.

    **The permutation itself is not torch's**, because numpy's stream is not
    torch's. What holds on both sides is the arithmetic above: the counts, the
    padding, the stride, and the fact that the ranks together cover the dataset.
    """

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True,
                 seed=0, drop_last=False):
        if num_replicas is None or rank is None:
            # **`ValueError`, and the type is not ours to choose.** torch has both
            # here — `RuntimeError` when the distributed package cannot be imported
            # at all, and `ValueError` out of `get_world_size` when it can and no
            # process group was started. On any ordinary install the second is the
            # one a caller meets: measured, `DistributedSampler(range(10))` on a
            # CPU-only build raises `ValueError`. Written `RuntimeError`, code that
            # catches torch's actual error walks straight past this one.
            raise ValueError(
                "DistributedSampler needs num_replicas and rank given outright — "
                "torch reads them from the process group when they are omitted, "
                "and there is no process group here")
        num_replicas, rank = int(num_replicas), int(rank)
        if not 0 <= rank < num_replicas:
            raise ValueError(f"Invalid rank {rank}, rank should be in the interval "
                             f"[0, {num_replicas - 1}]")
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        n = len(dataset)
        if drop_last and n % num_replicas:
            self.num_samples = _math.ceil((n - num_replicas) / num_replicas)
        else:
            self.num_samples = _math.ceil(n / num_replicas)
        self.total_size = self.num_samples * num_replicas

    def set_epoch(self, epoch):
        """Moves the shuffle. Call it before each epoch, on every rank."""
        self.epoch = int(epoch)

    def __iter__(self):
        n = len(self.dataset)
        if self.shuffle:
            # Seeded per epoch rather than drawn: the ranks agree only because
            # they compute the same permutation from the same two numbers.
            order = _np.random.default_rng(self.seed + self.epoch).permutation(n)
            indices = order.tolist()
        else:
            indices = list(range(n))
        if self.drop_last:
            indices = indices[:self.total_size]
        else:
            pad = self.total_size - len(indices)
            if pad <= len(indices):
                indices += indices[:pad]
            else:
                # Fewer rows than ranks — one pass over the front is not enough.
                indices += (indices * _math.ceil(pad / len(indices)))[:pad]
        return iter(indices[self.rank:self.total_size:self.num_replicas])

    def __len__(self):
        return self.num_samples


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
    BatchSampler = BatchSampler
    DistributedSampler = DistributedSampler
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




# ------------------------------------------------------------ images in hand, and doubt

def _name_and_bytes(item):
    """One uploaded file as (name, bytes).

    Three shapes arrive here: marimo's `mo.ui.file` value (objects with `.name` and
    `.contents`), plain `(name, bytes)` pairs, and paths on a desktop. Anything else
    is a clear error rather than a guess.
    """
    if hasattr(item, "contents") and hasattr(item, "name"):
        return str(item.name), bytes(item.contents)
    if isinstance(item, (tuple, list)) and len(item) == 2:
        return str(item[0]), bytes(item[1])
    if hasattr(item, "read_bytes"):
        return item.name, item.read_bytes()
    if isinstance(item, str):
        from pathlib import Path
        return Path(item).name, Path(item).read_bytes()
    raise TypeError(
        f"decode_images: expected an uploaded file, a (name, bytes) pair or a path, "
        f"got {type(item).__name__}")


def label_from_name(name):
    """The label an image's file name carries.

    `cats/001.png` → `cats` (the folder, as torchvision's ImageFolder reads it);
    `cat_001.png` → `cat` (the part before the first underscore — a browser upload has
    no folder, so the convention moves into the name).
    """
    head, _, tail = name.replace("\\", "/").rpartition("/")
    if head:
        return head.rsplit("/", 1)[-1]
    return tail.split("_")[0]


def _entries(files):
    """`(name, bytes)` for every image among `files` — a zip is opened and its members
    become entries under their paths, so a zipped folder keeps its folder labels.
    Directories and macOS resource forks (`__MACOSX/`, `._x`) are left out."""
    import io
    import zipfile
    for item in files:
        name, data = _name_and_bytes(item)
        if name.lower().endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                for info in z.infolist():
                    base = info.filename.rsplit("/", 1)[-1]
                    if info.is_dir() or info.filename.startswith("__MACOSX/") or base.startswith("._"):
                        continue
                    if base.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp")):
                        yield info.filename, z.read(info)
        else:
            yield name, data


def _image_module():
    """Pillow's `Image`, or the ImportError that says how to get it."""
    try:
        from PIL import Image
    except ImportError:
        raise ImportError("decoding images needs Pillow (`pip install pillow`) to read "
                          "PNG/JPEG bytes") from None
    return Image


class ImageFiles:
    """Image files already in hand, decoded **on demand** — the field trainer's dataset.

        ds = ImageFiles(upload.value, size=224)      # uploads, (name, bytes) pairs, paths, or a zip
        for x, idx in ds.batches(16):                # x: float32 (n, 3, size, size) in [0, 1]
            feats[idx] = backbone(x)
        x, y = ds[3]                                 # one image, its class index
        ds.classes, ds.labels, ds.names, ds.targets  # sorted label strings; per-image label/name/index

    Nothing is decoded until asked, so five thousand camera images cost the bytes they
    came in as, not 3 GB of float32. `stack()` decodes everything at once — the small
    path, for a few hundred images. A zip is a folder: its members' folder names are
    their labels (`label_from_name`).
    """

    def __init__(self, files, size=64, label=label_from_name):
        self.size = int(size)
        self._items = list(_entries(files))
        if not self._items:
            raise ValueError("ImageFiles: no image files given")
        self.names = [name for name, _ in self._items]
        self.labels = [str(label(name)) for name in self.names]
        self.classes = sorted(set(self.labels))
        self.targets = _np.array([self.classes.index(l) for l in self.labels], dtype=_np.int64)

    def __len__(self):
        return len(self._items)

    def _decode(self, i, size=None):
        import io
        Image = self._image_module()
        img = Image.open(io.BytesIO(self._items[i][1])).convert("RGB").resize((size or self.size,) * 2)
        return _np.asarray(img, dtype=_np.uint8)

    # The binding replaces this: in Pyodide, Pillow is a package the page has to fetch.
    _image_module = staticmethod(_image_module)

    def __getitem__(self, i):
        x = self._decode(i).transpose(2, 0, 1).astype(_np.float32) / 255.0
        return x, int(self.targets[i])

    def thumb(self, i, size=56):
        """One image as `(size, size, 3)` uint8 — a thumbnail for a review table."""
        return self._decode(i, size)

    def batches(self, n=16):
        """`(x, idx)` in order: `x` float32 `(len(idx), 3, size, size)`, `idx` the rows."""
        for start in range(0, len(self), n):
            idx = _np.arange(start, min(start + n, len(self)))
            x = _np.stack([self._decode(int(i)) for i in idx]).transpose(0, 3, 1, 2).astype(_np.float32) / 255.0
            yield x, idx

    def stack(self):
        """Every image at once: `(N, 3, size, size)` float32 — for a few hundred images."""
        return _np.concatenate([x for x, _ in self.batches(64)])


def decode_images(files, size=64, label=label_from_name):
    """Image files already in hand → `(x, y, names, classes)`, torch's NCHW in [0, 1].

    `x` is float32 `(N, 3, size, size)`, `y` int64 class indices into `classes`
    (sorted label strings), `names` the file names in the order given. Every image is
    converted to RGB and resized to `size × size` — the same thing torchvision's
    `ImageFolder` + `Resize` + `ToTensor` do, minus the folder: a browser hands over
    files, not a directory, so the label comes from the name (see `label_from_name`,
    or pass your own `label(name)`). A zip among the files is opened as a folder.

    This decodes everything at once; `ImageFiles` is the same thing decoded on demand.
    """
    ds = ImageFiles(files, size, label)
    return ds.stack(), ds.targets, ds.names, ds.classes


def suspects(features, labels, k=5):
    """How much each label is doubted, in [0, 1]: the share of the sample's `k`
    nearest neighbours (cosine, on `features`) that carry a different label.

    A wrong label sits among samples that disagree with it, so it scores high;
    ordering a review queue by this puts the labels worth a look first. Measured on
    CIFAR-10N: separates noisy from merely hard labels at 0.906 AUROC where the
    training-dynamics score (AUM) manages 0.765.

    `features` and `labels` may be tensors (anything with `.numpy()`) or arrays;
    the result is a float32 numpy array of length N. With fewer than two samples
    every score is 0.
    """
    f = _np.asarray(features.numpy() if hasattr(features, "numpy") else features, dtype=_np.float32)
    y = _np.asarray(labels.numpy() if hasattr(labels, "numpy") else labels)
    if f.ndim != 2 or f.shape[0] != y.shape[0]:
        raise ValueError(f"suspects: features must be (N, D) with one label per row, "
                         f"got {f.shape} and {y.shape}")
    n = f.shape[0]
    k = min(int(k), n - 1)
    if k < 1:
        return _np.zeros(n, dtype=_np.float32)
    unit = f / (_np.linalg.norm(f, axis=1, keepdims=True) + 1e-9)
    sims = unit @ unit.T
    _np.fill_diagonal(sims, -_np.inf)                    # a sample is not its own neighbour
    nearest = _np.argsort(-sims, axis=1)[:, :k]
    return (y[nearest] != y[:, None]).mean(axis=1).astype(_np.float32)
