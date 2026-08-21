"""`torch.save` and `torch.load` — **the codec is the core's, unchanged.**

## Why it is not written again here

Two copies of a format means a file one side writes is a file the other side
cannot read. And that is the **only reason** this project chose safetensors —
the path from training in a browser to carrying the result to your own
machine. So the header, the body, the dtype labels and the nested tree all
live in `borch._serialize`, and what is here is the **two things that differ**.

1. How to recognise a tensor — the `Tensor` here is a different class.
2. How to get the values out and back — the values live on the GPU.

## Why borch.ts's `save` is not called

That one is **asynchronous** (reading values back off the GPU is a round
trip). The Python API has to be synchronous throughout — that is the promise
`tests/browser/sync_probe.py` keeps — and here `.numpy()` already makes that
round trip synchronously. Two paths produce the same bytes and one of them is
synchronous, so that is the one taken.
"""

import numpy as _np

from borch._serialize import dump as _dump, parse as _parse

from ._base import tensor as _tensor, Tensor as _Tensor


def _array_of(obj):
    """The value as numpy if this is one of our tensors, otherwise `None`."""
    return obj.numpy() if isinstance(obj, _Tensor) else None


def _make(array):
    """Put a loaded array back on the GPU. Not a leaf, so gradients stay off."""
    return _tensor(_np.asarray(array))


def save(obj, where):
    """Write a checkpoint.

    **A browser has no real filesystem.** Pyodide supplies a virtual one, so
    `torch.save(sd, "ckpt.bin")` runs as written — the Python does not change
    by a character. Handing those bytes to a person means reading them back
    out from JavaScript with `FS.readFile` and offering a download, and that
    is the page's job rather than this library's.
    """
    _dump(obj, where, _array_of)


def load(where, **kw):
    """Read back what `save` wrote. Files the core wrote work too — same format."""
    return _parse(where, _make, **kw)
