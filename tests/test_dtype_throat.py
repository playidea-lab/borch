"""Every array that becomes a tensor carries a dtype this library actually has.

`Tensor.__init__` is the throat. `float64` and `complex128` were stopped there long
ago, and the comment above that line says why one place rather than many: *blocked
place by place, every new operation forgets one.* **The narrow widths were forgotten
anyway** — `int32`, `int16`, `int8`, `uint8` and `float16` are all `_AbsentDtype`,
each naming what it is gathered into, and an array carrying one was stored exactly as
it came.

Then `Tensor.dtype` looked the numpy dtype up in a table it was not in, and the
table's default was `float32`. So `Tensor(np.array([1], dtype=np.int32)).dtype`
answered **`torch.float32`** — a confident wrong answer, which is the kind nothing
finds.

Nothing did. On a 64-bit host `argsort` returns `int64` and the table knows it, so
every index-producing operation looked right. In Pyodide `intp` is 32 bits, the same
call returns `int32`, and `topk`, `sort`, `max(dim)`, `min(dim)` and `median(dim)`
handed back index halves labelled `torch.float32`. The golden held those cases and
they passed everywhere except the one runner that opens a browser and loads the core
— `tests/browser/run.py --lib borch`, which runs when a person runs it.

**These are pytest rather than golden cases because they are a deliberate
divergence.** Real torch answers `torch.int32` for an int32 array; this library has no
int32 and says so through `_AbsentDtype`, so the honest answer here is `int64`. A
golden case would compare against torch and be red for being right.
"""

import numpy as np
import pytest

import borch

# Every numpy dtype an array can arrive with, and what this library must call it.
# The right-hand side is `_AbsentDtype`'s own `instead` field, read off `borch/_base.py`
# rather than decided here — `int32 = _AbsentDtype("int32", "int64")` and so on.
ARRIVES = [
    ("int8", borch.int64), ("int16", borch.int64), ("int32", borch.int64),
    ("int64", borch.int64),
    ("uint8", borch.int64), ("uint16", borch.int64), ("uint32", borch.int64),
    ("uint64", borch.int64),
    ("float16", borch.float32), ("float32", borch.float32),
    ("float64", borch.float32),
    ("bool", borch.bool),
]


@pytest.mark.parametrize("numpy_name,want", ARRIVES)
def test_a_tensor_reports_a_dtype_this_library_has(numpy_name, want):
    """The label. **This is the half that was wrong**, and it was wrong quietly."""
    got = borch.Tensor(np.array([1, 0], dtype=numpy_name)).dtype
    assert got is want, (
        f"an array of numpy `{numpy_name}` became a tensor calling itself {got}, "
        f"and this library's own table says {want}.\n"
        "  `borch/_base.py` maps the narrow widths through `_AbsentDtype`; the tensor "
        "has to agree with it.")


@pytest.mark.parametrize("numpy_name,want", ARRIVES)
def test_the_storage_agrees_with_the_label(numpy_name, want):
    """And the storage. **A right label over the wrong array is the same bug moved.**

    Reporting `int64` while holding `int32` would pass the test above and still
    overflow at 2^31, so the array itself has to be the one the label names.
    """
    stored = borch.Tensor(np.array([1, 0], dtype=numpy_name)).data.dtype
    assert stored == want.np, (
        f"an array of numpy `{numpy_name}` is stored as `{stored}` while the tensor "
        f"calls itself {want}. The label is not the storage.")


def test_an_unknown_dtype_says_so_rather_than_guessing():
    """**The default that hid all of the above.**

    `Tensor.dtype` answered `float32` for anything the table did not know. Everything
    is normalised in the throat now, so this cannot be reached the ordinary way — it is
    reached here by writing the array in behind `__init__`, which is what makes the
    guard testable at all. What is asked is that it **stops** rather than naming a
    dtype the tensor does not have.
    """
    t = borch.Tensor(np.array([1, 0], dtype="int64"))
    t._array = np.array([1, 0], dtype="int16")      # behind the throat, on purpose
    with pytest.raises(borch.BorchError, match="not one of this library's dtypes"):
        t.dtype


def test_the_index_half_of_a_pair_is_int64():
    """What the browser run actually failed on, asked where it can be asked.

    These five produce their indices through `np.argsort`, whose width follows the
    platform's pointer size. On this host it is already 64 bits, so the assertion is
    weak here and exact in Pyodide — it is written down because the two places have to
    agree, and only one of them is in CI.
    """
    x = borch.tensor([3.0, 1.0, 2.0, 0.5])
    pairs = {
        "topk": x.topk(2), "sort": x.sort(), "max": x.max(0),
        "min": x.min(0), "median": x.median(0),
    }
    wrong = {name: got.indices.dtype for name, got in pairs.items()
             if got.indices.dtype is not borch.int64}
    assert not wrong, (
        f"these came back with an index half that is not int64: {wrong}\n"
        "  torch's is `torch.int64` on every platform. numpy's `argsort` follows the "
        "pointer size,\n  so this is int32 under Pyodide unless the throat normalises "
        "it.")
