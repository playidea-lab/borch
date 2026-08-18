"""`torch.save` · `torch.load` — **코덱은 코어의 것을 그대로 쓴다.**

## 왜 여기서 다시 안 쓰는가

형식이 두 벌이면 한쪽이 쓴 파일을 다른 쪽이 못 읽는다. 그런데 이 프로젝트가
safetensors 를 고른 **유일한 이유**가 그것이다 — 브라우저에서 학습해 자기
컴퓨터로 가져가는 길. 그래서 머리·몸·이름표·중첩 나무는 `borch._serialize` 한
군데에 있고, 여기서는 **다른 두 가지만** 준다.

1. 텐서를 알아보는 잣대 — 여기 `Tensor` 는 저쪽 것과 다른 클래스다.
2. 값을 꺼내고 되돌리는 길 — 값이 GPU 에 있다.

## 왜 borch.ts 의 `save` 를 안 부르는가

저쪽 `save` 는 **비동기다**(GPU 에서 값을 되가져오는 왕복이 있다). 파이썬 API 는
통째로 동기여야 하고 — 그것이 `tests/browser/sync_probe.py` 가 지키는 약속이다 —
여기서는 `.numpy()` 가 이미 그 왕복을 동기로 한다. 같은 바이트가 나오는 길이
둘인데 하나는 동기이므로 그것을 쓴다.
"""

import numpy as _np

from borch._serialize import dump as _dump, parse as _parse

from ._base import tensor as _tensor, Tensor as _Tensor


def _array_of(obj):
    """이 결속의 텐서면 그 값을 numpy 로, 아니면 `None`."""
    return obj.numpy() if isinstance(obj, _Tensor) else None


def _make(array):
    """읽은 배열을 GPU 텐서로 되올린다. 잎이 아니므로 기울기는 안 켠다."""
    return _tensor(_np.asarray(array))


def save(obj, where):
    """체크포인트를 쓴다.

    **브라우저에는 진짜 파일시스템이 없다.** 그래도 Pyodide 가 가상 파일시스템을
    주므로 `torch.save(sd, "ckpt.bin")` 이 그대로 돈다 — 파이썬 코드는 한 글자도
    안 바뀐다. 그 바이트를 사람 손에 넘기려면 JS 쪽에서 `FS.readFile` 로 꺼내
    내려받게 하면 되고, 그것은 이 라이브러리가 아니라 페이지가 할 일이다.
    """
    _dump(obj, where, _array_of)


def load(where, **kw):
    """`save` 가 쓴 것을 되읽는다. 코어가 쓴 파일도 읽는다 — 같은 형식이다."""
    return _parse(where, _make, **kw)
