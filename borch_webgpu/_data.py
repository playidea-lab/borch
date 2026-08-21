"""`utils.data`, the download cache, and save/load.

Carried over from the sister library. **One line** changed on the way:
`backend()` answers with borch.ts's adapter rather than TF.js's. The rest uses
only numpy and browser APIs, so it is the same whatever sits underneath.

Data stays **on the CPU, in numpy.** All of CIFAR-10 on the GPU is 614MB; one
batch is 3MB. Uploading per batch is cheaper and leaves the GPU memory for the
model.
"""

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
    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        return iter(_rng.permutation(len(self.data_source)).tolist())

    def __len__(self):
        return len(self.data_source)


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
                 num_workers=0, drop_last=False, collate_fn=None):
        if sampler is not None and shuffle:
            raise ValueError("sampler and shuffle cannot be used together.")
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.collate_fn = collate_fn
        self.sampler = sampler or (RandomSampler(dataset) if shuffle
                                   else SequentialSampler(dataset))

    def __len__(self):
        n = len(self.sampler)
        return n // self.batch_size if self.drop_last else -(-n // self.batch_size)

    def _fast(self, idx):
        return tuple(tensor(a[_np.asarray(idx)]) for a in self.dataset.arrays)

    def __iter__(self):
        plain = isinstance(self.dataset, TensorDataset) and self.collate_fn is None
        batch = []
        for i in self.sampler:
            batch.append(i if plain else self.dataset[i])
            if len(batch) == self.batch_size:
                yield self._fast(batch) if plain else self._collate(batch)
                batch = []
        if batch and not self.drop_last:
            yield self._fast(batch) if plain else self._collate(batch)

    def _collate(self, batch):
        if self.collate_fn:
            return self.collate_fn(batch)
        return tuple(stack([x if isinstance(x, Tensor) else wrap(x) for x in col])
                     for col in zip(*batch))


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
    RandomSampler = RandomSampler
    SequentialSampler = SequentialSampler
    WeightedRandomSampler = WeightedRandomSampler
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
    def manual_seed_all(seed):
        return None

    @staticmethod
    def synchronize():
        _unsupported("torch.cuda.synchronize")


cuda = _Cuda()


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
