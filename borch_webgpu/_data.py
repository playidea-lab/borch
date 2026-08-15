"""`utils.data`·내려받기 캐시·저장/불러오기.

자매(`borch_webgpu/_data.py`)에서 옮겨왔다. 옮기면서 바뀐 것은 **한 줄**이다 —
`backend()` 가 TF.js 대신 borch.ts 의 어댑터를 답한다. 나머지는 numpy 와 브라우저
API 만 쓰므로 밑바닥이 무엇이든 같다.

데이터는 **CPU(numpy)에 둔다.** CIFAR-10 을 통째로 GPU 에 올리면 614MB 이고 배치
하나는 3MB 다. 매 배치 올리는 쪽이 싸고, GPU 메모리를 모델에 남긴다.
"""

import numpy as _np

import js as _js
from pyodide.ffi import to_js as _to_js

from ._base import Tensor, tensor, wrap
from ._ops import _rng, stack


def _unsupported(what):
    raise RuntimeError(
        f"{what} 은(는) 아직 여기 없다. 코어 `borch` 나 진짜 PyTorch 를 써라.")


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
    """드문 것을 더 자주 뽑는다. 1000명 중 10명이 환자인 데이터에서 배치에 환자가
    한 명도 없는 일을 막는다."""

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
    """배치마다 GPU 로 올린다. 셔플은 CPU 에서 번호만 섞는다.

    `TensorDataset` 은 numpy 째로 잘라 한 번에 올린다(빠른 길). 그 밖의 Dataset 은
    한 칸씩 꺼내 `stack` 으로 모은다 — 코어와 같은 방식이고 느리지만 무엇이든 받는다.
    """

    def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None,
                 num_workers=0, drop_last=False, collate_fn=None):
        if sampler is not None and shuffle:
            raise ValueError("sampler 와 shuffle 은 같이 쓸 수 없다.")
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


# `torch.utils.data` 와 `torch.nn.utils` 는 **다른 `utils`** 다. 이름이 같아서
# 한쪽이 다른 쪽을 덮으면 `nn.utils.rnn.pad_sequence` 나 `utils.data.DataLoader`
# 중 하나가 조용히 사라진다. 여기 것은 최상위 쪽이고, `nn` 쪽은 `_nn.py` 에 있다.
class _UtilsData:
    Dataset = Dataset
    TensorDataset = TensorDataset
    Subset = Subset
    ConcatDataset = ConcatDataset
    DataLoader = DataLoader
    RandomSampler = RandomSampler
    SequentialSampler = SequentialSampler
    WeightedRandomSampler = WeightedRandomSampler
    random_split = staticmethod(random_split)


class _Utils:
    data = _UtilsData()


utils = _Utils()


# ---------------------------------------------------------------- 내려받기·캐시

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
    """받아서 OPFS 에 넣고, 다음부터는 넣어둔 것을 쓴다.

    **비동기다.** OPFS 에 동기 API 가 없다(워커 안에서만 있다). 다만 이것은 학습
    루프가 아니라 **준비 단계에서 한 번** 부르는 것이라, 스텝을 동기로 유지한다는
    약속은 그대로다.

    URL 은 부르는 쪽이 준다. 데이터셋 주소를 라이브러리에 박아두면 그것이 사라졌을 때
    라이브러리를 고쳐야 한다.
    """
    key = name or url.rsplit("/", 1)[-1]
    try:
        return await _opfs_read(key)
    except Exception:                                                # noqa: BLE001
        pass                       # 아직 없다 — 받아온다
    response = await _js.fetch(url)
    if not response.ok:
        raise RuntimeError(f"내려받기 실패 {response.status}: {url}")
    data = _u8_to_np(_js.Uint8Array.new(await response.arrayBuffer()))
    await _opfs_write(key, data)
    return data


async def cache_put(name, data):
    """받아온 바이트를 캐시에 직접 넣는다.

    **CIFAR-10 원본(`cs.toronto.edu`)은 CORS 헤더를 주지 않는다**(실측: 브라우저가
    차단한다). 그래서 `fetch_cached` 로는 못 받는다. 사용자가 파일을 골라 넣거나
    CORS 를 주는 미러에서 받은 바이트를 여기로 넣으면 그다음은 같다.
    """
    await _opfs_write(name, _np.asarray(data, dtype=_np.uint8))


async def cache_get(name):
    """캐시에 있는 바이트. 없으면 None."""
    try:
        return await _opfs_read(name)
    except Exception:                                                # noqa: BLE001
        return None


_CIFAR_RECORD = 1 + 3 * 32 * 32          # 라벨 1바이트 + 픽셀 3072바이트


def decode_cifar10(raw):
    """CIFAR-10 의 바이너리 한 덩이를 (x, y) 로 푼다.

    한 장이 3073 바이트다 — 라벨 1 바이트에 R·G·B 가 각각 1024 바이트씩 이어 붙는다.
    그 순서가 곧 (3, 32, 32) 이라 torch 의 NCHW 와 같다.
    """
    arr = _np.asarray(raw, dtype=_np.uint8)
    if arr.size % _CIFAR_RECORD:
        raise ValueError(
            f"CIFAR-10 바이너리가 아니다 — {arr.size} 바이트는 "
            f"{_CIFAR_RECORD} 의 배수가 아니다")
    rows = arr.reshape(-1, _CIFAR_RECORD)
    y = rows[:, 0].astype(_np.int64)
    x = rows[:, 1:].reshape(-1, 3, 32, 32).astype(_np.float32) / 255.0
    return x, y


def _to_plain(obj):
    """텐서를 numpy 로 바꿔 저장 가능한 형태로. 중첩 dict/list 도 따라간다."""
    if isinstance(obj, Tensor):
        return {"__tensor__": obj.numpy()}
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_plain(v) for v in obj)
    return obj


def _from_plain(obj):
    if isinstance(obj, dict):
        if "__tensor__" in obj:
            return tensor(obj["__tensor__"])
        return {k: _from_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_from_plain(v) for v in obj)
    return obj


def save(obj, path):
    """브라우저에도 가상 파일시스템이 있어 경로가 통한다. pickle 한 겹만 쓴다."""
    import pickle
    with open(path, "wb") as f:
        pickle.dump(_to_plain(obj), f)


def load(path, **kwargs):
    import pickle
    with open(path, "rb") as f:
        return _from_plain(pickle.load(f))


class _Cuda:
    """이 라이브러리는 GPU 를 쓰지만 **CUDA 는 아니다.** 흉내 내면 교훈이 사라진다."""

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
    """지금 붙어 있는 어댑터. **옮겨오면서 바뀐 유일한 줄이다.**

    자매는 여기서 `tf.getBackend()` 를 물어 `'webgpu'` 라는 문자열을 냈다. 이쪽은
    TF.js 를 안 거치므로 물을 것이 없고, 대신 borch.ts 가 실제로 붙은 어댑터를
    답한다(`apple / metal-3` 처럼). 재는 쪽이 반드시 같이 적어야 하는 값이다 —
    헤드리스 브라우저는 GPU 가 없으면 **말없이** 소프트웨어 래스터라이저를 준다.

    골든 하네스가 `hasattr(lib, "backend")` 로 브라우저 구현을 알아본다. 그래서 이
    이름은 표면일 뿐 아니라 **어느 케이스를 받을지**를 정한다.
    """
    from ._ops import _ts

    return str(_ts.Device.adapterInfo)
