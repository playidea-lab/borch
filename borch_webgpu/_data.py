"""`utils.data`, the download cache, and save/load.

Carried over from the sister library. **One line** changed on the way:
`backend()` answers with borch.ts's adapter rather than TF.js's. The rest uses
only numpy and browser APIs, so it is the same whatever sits underneath.

Data stays **on the CPU, in numpy.** All of CIFAR-10 on the GPU is 614MB; one
batch is 3MB. Uploading per batch is cheaper and leaves the GPU memory for the
model.
"""

import math as _math

import numpy as _np

import js as _js
from pyodide.ffi import to_js as _to_js

from ._base import Tensor, tensor, wrap
from ._ops import _rng, stack


def _unsupported(what):
    raise RuntimeError(
        f"{what} is not here yet. Use the core `borch` or real PyTorch.")


# ---------------------------------------------------------------- utils.data

class Dataset:
    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, i):
        raise NotImplementedError


class TensorDataset(Dataset):
    def __init__(self, *arrays):
        self.arrays = [_np.asarray(a) for a in arrays]

    def __len__(self):
        return len(self.arrays[0])

    def __getitem__(self, i):
        return tuple(tensor(a[i]) for a in self.arrays)


class Subset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        return self.dataset[self.indices[i]]


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


class SequentialSampler:
    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        return iter(range(len(self.data_source)))

    def __len__(self):
        return len(self.data_source)


class RandomSampler:
    def __init__(self, data_source, replacement=False, num_samples=None,
                 generator=None):
        """`generator` is what makes a shuffled loader repeatable, and it was
        **not here** — so `DataLoader(ds, shuffle=True, generator=g)` gave a
        different order every run on this side and the same order on the core's.
        A seed that does not seed is worse than no seed, because the caller
        stops checking."""
        self.data_source = data_source
        self.replacement = replacement
        self.num_samples = num_samples
        self.generator = generator

    def _rng(self):
        return self.generator.rng() if self.generator is not None else _rng

    def __iter__(self):
        n = len(self.data_source)
        k = self.num_samples if self.num_samples is not None else n
        if self.replacement:
            return iter(self._rng().integers(0, n, size=k).tolist())
        return iter(self._rng().permutation(n).tolist()[:k])

    def __len__(self):
        return (self.num_samples if self.num_samples is not None
                else len(self.data_source))


class BatchSampler:
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


class WeightedRandomSampler:
    """Draw the rare thing more often. Stops a batch from containing no patients
    at all when 10 of every 1000 rows are patients."""

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


class DistributedSampler:
    """Each of `num_replicas` workers takes a different slice of one dataset.

    **No distribution is involved**, which is why it is here: given the pair
    outright it is arithmetic over two integers. torch reads them from a process
    group only when they are omitted, so omitting them raises here — as
    `ValueError`, which is what torch itself raises on an ordinary install:
    `RuntimeError` is its answer when the distributed package cannot be imported
    at all, and that is not the case a caller meets.

    The padding and the seeded shuffle are the core's; see `borch/_data.py` for
    why the shuffle must not come from the global stream.
    """

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True,
                 seed=0, drop_last=False):
        if num_replicas is None or rank is None:
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
            indices = _np.random.default_rng(
                self.seed + self.epoch).permutation(n).tolist()
        else:
            indices = list(range(n))
        if self.drop_last:
            indices = indices[:self.total_size]
        else:
            pad = self.total_size - len(indices)
            if pad <= len(indices):
                indices += indices[:pad]
            else:
                indices += (indices * _math.ceil(pad / len(indices)))[:pad]
        return iter(indices[self.rank:self.total_size:self.num_replicas])

    def __len__(self):
        return self.num_samples


def random_split(dataset, lengths, generator=None):
    rng = generator.rng() if generator is not None else _rng
    idx = rng.permutation(len(dataset)).tolist()
    out, start = [], 0
    for n in lengths:
        out.append(Subset(dataset, idx[start:start + n]))
        start += n
    return out


class DataLoader:
    """Upload one batch at a time. Shuffling moves indices on the CPU only.

    `TensorDataset` is sliced in numpy and uploaded in one go — the fast path.
    Any other Dataset is read a row at a time and gathered with `stack`, the
    same way the core does it: slower, but it takes anything.
    """

    def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None,
                 batch_sampler=None, num_workers=0, collate_fn=None,
                 pin_memory=False, drop_last=False, timeout=0,
                 worker_init_fn=None, multiprocessing_context=None,
                 generator=None, *, prefetch_factor=None,
                 persistent_workers=False, pin_memory_device="",
                 in_order=True):
        """torch's list. **This side held seven of it with two in the wrong
        seats** — `collate_fn` sixth and `drop_last` seventh, where torch has
        them seventh and ninth. `DataLoader(ds, 4, False, None, 0, True)` set
        `drop_last` here and `collate_fn` on torch's side: a boolean handed into
        a callable's slot, which torch then tries to call.

        The core's copy of this docstring explains the three-way division of the
        rest; the division is the same here and is not repeated.
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
        self.sampler = sampler or (
            RandomSampler(dataset, generator=generator) if shuffle
            else SequentialSampler(dataset))

    def __len__(self):
        if self.batch_sampler is not None:
            return len(self.batch_sampler)
        n = len(self.sampler)
        return n // self.batch_size if self.drop_last else -(-n // self.batch_size)

    def _fast(self, idx):
        return tuple(tensor(a[_np.asarray(idx)]) for a in self.dataset.arrays)

    def __iter__(self):
        plain = isinstance(self.dataset, TensorDataset) and self.collate_fn is None
        # **A batch sampler hands out whole batches**, so the batching below does
        # not run at all — it replaces `batch_size`, `shuffle`, `sampler` and
        # `drop_last` at once, which is why the constructor refuses them together.
        if self.batch_sampler is not None:
            for idx in self.batch_sampler:
                yield (self._fast(list(idx)) if plain
                       else self._collate([self.dataset[i] for i in idx]))
            return
        batch = []
        for i in self.sampler:
            batch.append(i if plain else self.dataset[i])
            if len(batch) == self.batch_size:
                yield self._fast(batch) if plain else self._collate(batch)
                batch = []
        if batch and not self.drop_last:
            yield self._fast(batch) if plain else self._collate(batch)

    def _collate(self, batch):
        return self.collate_fn(batch) if self.collate_fn else default_collate(batch)


def default_collate(batch):
    """Folds a list of samples into a batch.

    **The `zip(*batch)` that stood here assumed a sample is always a tuple.** That
    holds for `TensorDataset` and for nothing else: a dataset that is a plain list
    of numbers dies with `'int' object is not iterable`, two frames from the
    cause. The core had already been moved off it; this side had not, and the
    difference was invisible until a case used a bare list as a dataset.
    """
    first = batch[0]
    if isinstance(first, (tuple, list)):
        return type(first)(default_collate([b[i] for b in batch])
                           for i in range(len(first)))
    if isinstance(first, dict):
        return {k: default_collate([b[k] for b in batch]) for k in first}
    if isinstance(first, Tensor):
        return stack(list(batch))
    return stack([wrap(b) for b in batch])


def default_convert(data):
    """Turn numpy into tensors and **leave everything else alone.**

    The same rule the core follows, including both of its traps: **tuples come
    back as lists** (torch's own backwards compatibility) and **Python numbers
    are not converted.** Both were checked against real torch.
    """
    if isinstance(data, Tensor):
        return data
    if isinstance(data, (_np.ndarray, _np.generic)):
        return tensor(data)
    if isinstance(data, dict):
        made = {k: default_convert(v) for k, v in data.items()}
        try:
            return type(data)(made)
        except TypeError:                  # constructors that do not take a dict
            return made
    if isinstance(data, tuple):
        if hasattr(data, "_fields"):       # a namedtuple names its positions
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
    """**Always `None`** — a browser has no worker processes. torch answers
    `None` in the main process too, so this is the fact rather than a pretence."""
    return None


# `torch.utils.data` and `torch.nn.utils` are **different `utils`**. They share
# a name, so letting one shadow the other makes either `nn.utils.rnn.pad_sequence`
# or `utils.data.DataLoader` quietly disappear. This is the top-level one; the
# `nn` one lives in `_nn.py`.
class _UtilsData:
    Dataset = Dataset
    TensorDataset = TensorDataset
    Subset = Subset
    ConcatDataset = ConcatDataset
    DataLoader = DataLoader
    BatchSampler = BatchSampler
    DistributedSampler = DistributedSampler
    RandomSampler = RandomSampler
    SequentialSampler = SequentialSampler
    WeightedRandomSampler = WeightedRandomSampler
    default_collate = staticmethod(default_collate)
    default_convert = staticmethod(default_convert)
    get_worker_info = staticmethod(get_worker_info)
    random_split = staticmethod(random_split)


class _Utils:
    data = _UtilsData()


utils = _Utils()


# ------------------------------------------------------------ download and cache

def _u8_to_np(view):
    out = _np.empty(int(view.length), dtype=_np.uint8)
    view.assign_to(out)
    return out


def _np_to_u8(arr):
    buf = _js.Uint8Array.new(arr.size)
    buf.assign(arr)
    return buf


async def _opfs_read(name):
    root = await _js.navigator.storage.getDirectory()
    handle = await root.getFileHandle(name)
    blob = await handle.getFile()
    return _u8_to_np(_js.Uint8Array.new(await blob.arrayBuffer()))


async def _opfs_write(name, arr):
    root = await _js.navigator.storage.getDirectory()
    opts = _to_js({"create": True}, dict_converter=_js.Object.fromEntries)
    handle = await root.getFileHandle(name, opts)
    writable = await handle.createWritable()
    await writable.write(_np_to_u8(arr))
    await writable.close()


async def fetch_cached(url, name=None):
    """Fetch it, put it in OPFS, and use the stored copy from then on.

    **Asynchronous.** OPFS has no synchronous API outside a worker. This is
    called once during setup rather than inside a training loop, so the promise
    that a step stays synchronous is untouched.

    The caller supplies the URL. A dataset address baked into a library means
    editing the library on the day that address stops existing.
    """
    key = name or url.rsplit("/", 1)[-1]
    try:
        return await _opfs_read(key)
    except Exception:                                                # noqa: BLE001
        pass                       # not there yet — fetch it
    response = await _js.fetch(url)
    if not response.ok:
        raise RuntimeError(f"download failed {response.status}: {url}")
    data = _u8_to_np(_js.Uint8Array.new(await response.arrayBuffer()))
    await _opfs_write(key, data)
    return data


async def cache_put(name, data):
    """Put bytes you already have straight into the cache.

    **The original CIFAR-10 host (`cs.toronto.edu`) sends no CORS header** —
    measured: the browser blocks it. So `fetch_cached` cannot reach it. Bytes
    a user picked from disk, or fetched from a mirror that does send the
    header, go in here and everything after that is the same.
    """
    await _opfs_write(name, _np.asarray(data, dtype=_np.uint8))


async def cache_get(name):
    """The bytes in the cache, or None."""
    try:
        return await _opfs_read(name)
    except Exception:                                                # noqa: BLE001
        return None


_CIFAR_RECORD = 1 + 3 * 32 * 32          # 1 label byte + 3072 pixel bytes


def decode_cifar10(raw):
    """Unpack one CIFAR-10 binary chunk into (x, y).

    One image is 3073 bytes — a label byte followed by 1024 bytes each of R, G
    and B. That order is already (3, 32, 32), which is torch's NCHW.
    """
    arr = _np.asarray(raw, dtype=_np.uint8)
    if arr.size % _CIFAR_RECORD:
        raise ValueError(
            f"not a CIFAR-10 binary — {arr.size} bytes is not a multiple of "
            f"{_CIFAR_RECORD}")
    rows = arr.reshape(-1, _CIFAR_RECORD)
    y = rows[:, 0].astype(_np.int64)
    x = rows[:, 1:].reshape(-1, 3, 32, 32).astype(_np.float32) / 255.0
    return x, y



# Images a visitor already has, and which of their labels to doubt. **The computation
# is the core's** (`borch._data`) — numpy on the CPU, the same grain as `decode_cifar10`:
# bytes in hand become arrays, and the model is what runs on the GPU. What differs here
# is only how Pillow arrives: Pyodide ships it as a package the page has to ask for.
from borch._data import label_from_name, suspects                    # noqa: E402,F401
from borch._data import ImageFiles as _ImageFilesCore                 # noqa: E402


def _pillow():
    """Pillow's `Image` — fetched as a Pyodide package on first use (JSPI, the same
    `run_sync` a readback uses), so a notebook needs no `import PIL` line of its own."""
    try:
        from PIL import Image
    except ImportError:
        import pyodide_js
        from pyodide.ffi import run_sync
        run_sync(pyodide_js.loadPackage("pillow"))
        from PIL import Image
    return Image


class ImageFiles(_ImageFilesCore):
    """See `borch.ImageFiles`. Only how Pillow arrives differs here."""
    _image_module = staticmethod(_pillow)


def decode_images(files, size=64, label=label_from_name):
    """See `borch.decode_images` — everything decoded at once; `ImageFiles` on demand."""
    ds = ImageFiles(files, size, label)
    return ds.stack(), ds.targets, ds.names, ds.classes


# `save` and `load` used to live here as a thin layer over pickle. **They moved
# to `_serialize.py` and the format became safetensors.** They call the core's
# codec now, so a file one side writes is a file the other side reads — which is
# why borch.ts chose that format, and which was not true while both Python
# libraries were using pickle.


class _Cuda:
    """This library uses a GPU but it is **not CUDA.** Pretending otherwise
    takes the lesson with it."""

    @staticmethod
    def is_available():
        return False

    @staticmethod
    def device_count():
        """Zero — torch's own answer on a machine without CUDA (it returns 0 rather
        than raising). `if torch.cuda.device_count() > 1:` opens every multi-GPU
        example and stopped here on a missing attribute while `is_available()`
        beside it answered."""
        return 0

    @staticmethod
    def manual_seed_all(seed):
        return None

    @staticmethod
    def synchronize():
        _unsupported("torch.cuda.synchronize")


cuda = _Cuda()


def get_default_device():
    """The device new tensors land on. **The GPU is behind borch.ts and the tensors
    are still `cpu` to torch's vocabulary** — `x.device` has said so all along, so
    this returns what that says rather than inventing a second answer."""
    from borch._base import device as _device
    return _device("cpu")


def set_default_device(what=None):
    """The other half. `cpu` and `None` pass; anything else stops, because putting
    the tensor where it already was and reporting success is the shape this library
    refuses everywhere else."""
    # **The core's `_unsupported`, not this file's.** The local one says "is not
    # here yet. Use the core `borch` or real PyTorch", which is a second wording for
    # the same kind of refusal; the core's is what every other absent thing in this
    # library says, and the golden compares wording. The local one is left alone —
    # what it already refuses has frozen answers, and moving those is its own
    # question.
    from borch._base import _unsupported as _core_unsupported

    name = "cpu" if what is None else str(getattr(what, "type", what))
    if name != "cpu":
        _core_unsupported(f"set_default_device({name!r})")
    return None


def backend():
    """Whichever adapter is actually attached. **The one line that changed on
    the way over.**

    The sister library asked `tf.getBackend()` here and produced the string
    `'webgpu'`. This one does not go through TF.js, so there is nothing to ask;
    it answers with the adapter borch.ts really attached to, such as
    `apple / metal-3`. Anyone measuring has to record it: a headless browser
    with no GPU hands back a software rasteriser **without saying so.**

    The golden harness recognises a browser implementation by
    `hasattr(lib, "backend")`. So this name is not only surface — it decides
    **which cases the library is given.**
    """
    from ._ops import _ts

    return str(_ts.Device.adapterInfo)
