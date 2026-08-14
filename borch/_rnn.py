"""borch 를 쪼갠 조각. 공개 이름은 __init__ 이 모은다."""

import math as _math

import numpy as _np

from ._ops import (
    _Namespace, _wrap,
)
from ._base import (
    _like_torch, _np,
)
from ._nn import (
    nn,
)

# ================================================================ nn.utils.rnn

def pad_sequence(sequences, batch_first=False, padding_value=0.0):
    """길이가 제각각인 텐서들을 가장 긴 것에 맞춰 채워 하나로 쌓는다.

    길이가 다른 것들을 한 배치에 담으려면 어딘가는 채워야 한다. 채운 자리가
    진짜 값처럼 보이면 안 되므로 `padding_value` 가 무엇이었는지 기억해야 하고,
    그래서 진짜 torch 도 이 함수와 마스크를 짝으로 쓴다.
    """
    tensors = [_wrap(s) for s in sequences]
    if not tensors:
        raise ValueError("빈 목록은 쌓을 수 없습니다.")
    rest = tensors[0].data.shape[1:]
    for t in tensors:
        if t.data.shape[1:] != rest:
            raise RuntimeError(_like_torch(
                f"첫 차원 말고는 모양이 같아야 합니다 — {rest} 와 {t.data.shape[1:]} 가 다릅니다.",
                "pad_sequence expects trailing dimensions to match",
            ))
    longest = max(t.data.shape[0] for t in tensors)
    padded = _np.full((len(tensors), longest) + rest, padding_value,
                      dtype=tensors[0].data.dtype)
    for i, t in enumerate(tensors):
        padded[i, :t.data.shape[0]] = t.data
    if not batch_first:
        padded = padded.swapaxes(0, 1)

    # 역방향은 **채운 자리를 도로 떼어내는** 것이다. 채운 값은 입력에서 온 것이 아니므로
    # 그 자리의 기울기는 아무에게도 안 간다.
    def back(g):
        gg = _np.asarray(g)
        if not batch_first:
            gg = gg.swapaxes(0, 1)
        return tuple(gg[i, :t.data.shape[0]] for i, t in enumerate(tensors))

    return tensors[0]._make(padded, tuple(tensors), back, "PadSequenceBackward0")


class _NnUtilsRnn(_Namespace):
    pad_sequence = staticmethod(pad_sequence)


class _NnUtils(_Namespace):
    rnn = _NnUtilsRnn()


nn.utils = _NnUtils()


