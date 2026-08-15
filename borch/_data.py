"""borch 를 쪼갠 조각. 공개 이름은 __init__ 이 모은다."""

import math as _math

import numpy as _np

from ._tensor import (
    Tensor,
)
from ._ops import (
    _Namespace, _rng, _wrap, stack,
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
    """드문 것을 더 자주 뽑는다. 1000명 중 10명이 환자인 데이터에서 배치에 환자가
    한 명도 없는 일을 막는다 — 5장이 가르치는 그것이다."""

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
    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        return iter(_rng.permutation(len(self.data_source)).tolist())

    def __len__(self):
        return len(self.data_source)


class DataLoader:
    def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None,
                 num_workers=0, drop_last=False, collate_fn=None):
        if sampler is not None and shuffle:
            raise ValueError("sampler 와 shuffle 은 같이 쓸 수 없습니다.")
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.collate_fn = collate_fn
        if isinstance(dataset, IterableDataset):
            # 훑는 데이터셋은 길이도 번호도 없다. 섞으라고 하면 여기서 멈춘다 —
            # 조용히 안 섞고 도는 것이 더 나쁘다.
            if shuffle or sampler is not None:
                raise ValueError("IterableDataset 은 섞을 수 없습니다 — 번호가 없습니다.")
            self.sampler = None
        else:
            self.sampler = sampler or (RandomSampler(dataset) if shuffle
                                       else SequentialSampler(dataset))

    def __iter__(self):
        # **훑는 데이터셋은 번호를 안 쓴다.** 샘플러가 낼 번호가 없으므로 그대로 흘린다.
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
        n = len(self.sampler)
        return n // self.batch_size if self.drop_last else -(-n // self.batch_size)

    def _collate(self, batch):
        """**`default_collate` 에 맡긴다.**

        여기 있던 `zip(*batch)` 는 표본이 늘 튜플이라고 가정했다. `TensorDataset` 만
        보면 참이지만 `IterableDataset` 은 맨 텐서를 흘릴 수 있고, 그러면 `zip` 이
        텐서 자체를 훑으려다 `len() of unsized object` 로 죽는다 — 원인에서 두 칸
        떨어진 자리다.
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
    """샘플러의 밑동. **없으면 자기 샘플러를 못 만든다.**

    `RandomSampler`·`SequentialSampler` 는 이미 있었는데 이 기반 클래스가 없었다.
    그러면 "번호를 이 순서로 뽑아라" 를 직접 정하는 길이 아예 막힌다 — 층 수를 늘리는
    것과 달리 **대신할 방법이 없는** 종류다.

    `__iter__` 하나만 채우면 `DataLoader` 가 받는다. torch 도 그 계약이다.

    **`data_source` 를 안 받는다.** torch 는 예전에 받았고 지금은 뺐다 —
    `super().__init__(source)` 를 쓰면 그쪽에서 `TypeError` 가 난다. 여기서 받아
    주면 우리에게서만 도는 코드가 생기고, 그게 이 라이브러리가 가장 하면 안 되는 것이다.
    """

    def __iter__(self):
        raise NotImplementedError("Sampler 는 `__iter__` 를 채워야 한다")


class SubsetRandomSampler(Sampler):
    """준 번호들 안에서만 섞는다. 교차검증의 접기마다 쓰는 것이 이것이다."""

    def __init__(self, indices, generator=None):
        self.indices = list(indices)
        self.generator = generator

    def __iter__(self):
        rng = self.generator.rng() if self.generator is not None else _rng
        return iter([self.indices[i] for i in rng.permutation(len(self.indices))])

    def __len__(self):
        return len(self.indices)


class BatchSampler(Sampler):
    """번호를 **묶어서** 준다. 배치 크기를 샘플러 쪽에서 정하고 싶을 때."""

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
    """번호로 꺼내는 것이 아니라 **흘려보내는** 데이터셋.

    끝을 모르는 것(스트림, 큰 파일)이 이 모양이다. `__len__` 도 `__getitem__` 도
    없으므로 `DataLoader` 가 섞을 수 없다 — 섞으려 하면 그 자리에서 멈춘다.
    """

    def __iter__(self):
        raise NotImplementedError("IterableDataset 은 `__iter__` 를 채워야 한다")

    def __getitem__(self, i):
        raise TypeError("IterableDataset 은 번호로 못 꺼낸다 — 훑어야 한다")


class ChainDataset(IterableDataset):
    """흘려보내는 것들을 **이어 붙인다.** `ConcatDataset` 의 훑는 판이다."""

    def __init__(self, datasets):
        self.datasets = list(datasets)

    def __iter__(self):
        for d in self.datasets:
            yield from d


class StackDataset(Dataset):
    """같은 길이의 데이터셋 여럿을 **나란히** 묶는다. 하나를 꺼내면 튜플이 나온다."""

    def __init__(self, *datasets, **named):
        if datasets and named:
            raise ValueError("자리로 주거나 이름으로 주거나 둘 중 하나다")
        self.datasets = named or datasets

    def __len__(self):
        got = self.datasets.values() if isinstance(self.datasets, dict) else self.datasets
        return min(len(d) for d in got)

    def __getitem__(self, i):
        if isinstance(self.datasets, dict):
            return {k: d[i] for k, d in self.datasets.items()}
        return tuple(d[i] for d in self.datasets)


def default_collate(batch):
    """표본 목록을 배치로 접는다. **`DataLoader` 가 말없이 하던 일에 이름을 준다.**

    직접 부르는 자리가 있다 — 자기 `collate_fn` 을 쓰면서 일부만 기본대로 접고 싶을 때.
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
    IterableDataset = IterableDataset
    ChainDataset = ChainDataset
    StackDataset = StackDataset
    default_collate = staticmethod(default_collate)
    random_split = staticmethod(random_split)


class _Utils(_Namespace):
    data = _UtilsData()


utils = _Utils()


