"""A piece of borch, split out. __init__ gathers the public names."""

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
    """Pad tensors of differing lengths up to the longest and stack them into
    one.

    Putting different lengths into one batch means padding somewhere. The padded
    positions must not look like real values, so what `padding_value` was has to
    be remembered — which is why real torch uses this function and a mask as a
    pair.
    """
    tensors = [_wrap(s) for s in sequences]
    if not tensors:
        raise ValueError("An empty list cannot be stacked.")
    rest = tensors[0].data.shape[1:]
    for t in tensors:
        if t.data.shape[1:] != rest:
            raise RuntimeError(_like_torch(
                f"Every dimension but the first must match — {rest} and {t.data.shape[1:]} differ.",
                "pad_sequence expects trailing dimensions to match",
            ))
    longest = max(t.data.shape[0] for t in tensors)
    padded = _np.full((len(tensors), longest) + rest, padding_value,
                      dtype=tensors[0].data.dtype)
    for i, t in enumerate(tensors):
        padded[i, :t.data.shape[0]] = t.data
    if not batch_first:
        padded = padded.swapaxes(0, 1)

    # The backward pass is **taking the padding back off.** The padded values
    # did not come from the input, so the gradient at those positions goes to
    # nobody.
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


