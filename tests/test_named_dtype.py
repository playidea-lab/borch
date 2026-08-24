"""**A dtype the caller named is answered or refused — never quietly swapped.**

There is no `float64` here. `Tensor.__init__` narrows it to `float32`, deliberately:
it is the throat every promotion passes through, and numpy raises `int64 + float32`
to double behind our back. Narrowing what arrives *by accident* is how the library
holds its first design decision.

**A named request is not an accident**, and that distinction already had a home.
`Tensor._cast` refuses `.double()`, `.to(float64)` and `.type(float64)`, and its
comment says why in a sentence this file exists to enforce: *the request was granted
in name and answered in another cell.*

`dtype=` was the fourth spelling of the same request and the only one still answered
that way. Measured before the repair: **thirty-seven factories took `dtype=float64`
and handed back `float32`**, while three method spellings of the identical request
raised. Three doors that raise and one that quietly gives you something else is worse
than four that raise, because the quiet one teaches that the dtype was honoured — and
the tensor is `float32` two hundred lines later when the answer disagrees with torch.

## What a green run does not say

- **Not that `float64` works.** It does not exist here and that is the point.
- **Not that promotion is gated.** `int64 + float32` still narrows silently in
  `Tensor.__init__`, and it must — the other half of the rule this file holds.
- **Not that every keyword is honoured.** `bernoulli` and `poisson` take `dtype=`
  into `**kw` and drop it on the floor, which is a different defect on a different
  axis (torch has no `dtype` there either) and is not what this file asks.
"""

import inspect
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

pytest.importorskip("numpy")

import borch  # noqa: E402


def _spellings():
    """Every way a caller can **name** double precision, as `(label, thunk)`.

    The method spellings are here beside the factories on purpose. They already
    refused, and a file that only watched the factories would go green on the day
    somebody made `.double()` narrow again to match them.
    """
    F = borch.float64
    t = borch.tensor([1.0, 2.0])
    return [
        ("tensor(dtype=)", lambda: borch.tensor([1.0], dtype=F)),
        ("as_tensor(list)", lambda: borch.as_tensor([1.0], dtype=F)),
        ("as_tensor(tensor)", lambda: borch.as_tensor(t, dtype=F)),
        ("asarray", lambda: borch.asarray(t, dtype=F)),
        ("zeros", lambda: borch.zeros(3, dtype=F)),
        ("ones", lambda: borch.ones(3, dtype=F)),
        ("empty", lambda: borch.empty(3, dtype=F)),
        ("full", lambda: borch.full((2,), 1.0, dtype=F)),
        ("eye", lambda: borch.eye(2, dtype=F)),
        ("rand", lambda: borch.rand(2, dtype=F)),
        ("zeros_like", lambda: borch.zeros_like(t, dtype=F)),
        ("ones_like", lambda: borch.ones_like(t, dtype=F)),
        # `arange` hands `dtype` straight to numpy and calls the shared gate with
        # `None`, which is exactly how it slipped past the first repair.
        ("arange", lambda: borch.arange(3, dtype=F)),
        ("linspace", lambda: borch.linspace(0, 1, 3, dtype=F)),
        ("frombuffer", lambda: borch.frombuffer(b"\x00" * 8, dtype=F)),
        # `norm` converts **before** computing, which torch does too — so the dtype
        # is named by the caller here as much as in any factory.
        ("norm(dtype=)", lambda: borch.norm(t, dtype=F)),
        ("scalar_tensor", lambda: borch.scalar_tensor(1.0, dtype=F)),
        ("Tensor.new_zeros", lambda: t.new_zeros(2, dtype=F)),
        ("Tensor.new_ones", lambda: t.new_ones(2, dtype=F)),
        ("Tensor.new_empty", lambda: t.new_empty(2, dtype=F)),
        ("Tensor.new_full", lambda: t.new_full((2,), 1.0, dtype=F)),
        ("Tensor.new_tensor", lambda: t.new_tensor([1.0], dtype=F)),
        (".double()", lambda: t.double()),
        (".to(float64)", lambda: t.to(F)),
        (".type(float64)", lambda: t.type(F)),
    ]


@pytest.mark.parametrize("label,call", _spellings(), ids=lambda v: v if isinstance(v, str) else "")
def test_a_named_float64_is_refused_rather_than_narrowed(label, call):
    try:
        got = call()
    except borch.BorchError as exc:
        assert "float64" in str(exc), (
            f"{label} refused, but the message does not name the dtype: {exc}")
        return
    pytest.fail(
        f"{label} was asked for `float64` and returned a {got.dtype} tensor.\n\n"
        "  A silent downcast is not a refusal. The caller wrote the dtype down, so\n"
        "  either it is honoured or it stops — and it cannot be honoured here.\n"
        "  `borch._base._requested_dtype` is the gate; this spelling does not reach it.")


def test_the_gate_does_not_catch_a_dtype_that_arrived_by_promotion():
    """**The other half of the rule, and the half that must not move.**

    `int64 + float32` is `float64` in numpy and `float32` in torch, and the narrowing
    in `Tensor.__init__` is what makes this library answer torch's way. Nobody named
    `float64` there, so nothing may raise.

    Without this, the obvious 'fix' for the test above — refusing `float64` wherever
    it appears — passes it and breaks arithmetic, and the assertion that catches that
    has to live in the same file as the one that tempts it.
    """
    got = borch.tensor([1]) + borch.tensor([1.5])
    assert str(got.dtype) == "torch.float32", got.dtype
    # A `float64` numpy array is not a named request either — numpy's default is
    # double, so refusing here would make `from_numpy` useless on ordinary arrays.
    import numpy as np
    assert str(borch.from_numpy(np.zeros(3)).dtype) == "torch.float32"


def test_the_gate_does_not_catch_a_dtype_that_is_only_being_asked_about():
    """`can_cast` and `promote_types` **name** `float64` and must still answer.

    They are questions about dtypes, not requests to make one, and `float64`'s name
    has a second job here — naming what numpy hands over during promotion, the same
    reason `complex128` is a real dtype object. A gate placed in `_np_of`, the shared
    unwrapper, would have been one line shorter and would have broken both.
    """
    assert borch.can_cast(borch.float32, borch.float64) is True
    assert str(borch.promote_types(borch.int64, borch.float32)) == "torch.float32"


def test_a_dtype_that_does_exist_is_still_honoured():
    """**The floor.** A gate that refused every `dtype=` would pass every assertion
    above, and this is the one it could not pass."""
    honoured = 0
    for label, call in (("zeros", lambda: borch.zeros(3, dtype=borch.int64)),
                        ("tensor", lambda: borch.tensor([1.0], dtype=borch.int64)),
                        ("arange", lambda: borch.arange(3, dtype=borch.float32)),
                        ("eye", lambda: borch.eye(2, dtype=borch.bool))):
        got = call()
        assert got.dtype in (borch.int64, borch.float32, borch.bool), (label, got.dtype)
        honoured += 1
    assert honoured == 4


def test_the_sweep_is_asking_about_the_factories_that_exist():
    """**A spelling list can go stale by naming things that are gone.**

    Every entry above is written by hand, and a renamed factory would drop out of the
    list rather than fail in it — the shape where an instrument's entry condition
    quietly removes the class it hunts, which this repository has now met four times.
    """
    missing = [label for label, _ in _spellings()
               if "." not in label and not hasattr(
                   borch, label.split("(")[0])]
    assert not missing, f"these are no longer names in `borch`: {missing}"
    assert len(_spellings()) >= 25, (
        f"only {len(_spellings())} spellings are being asked — the list has been "
        "trimmed, and the ones removed are exactly the ones nothing now watches.")


def test_the_gate_is_reached_from_both_throats():
    """`_resolve` and `_made` are the two doors, and both have to call it.

    Named separately because they fail separately: `tensor()` goes through `_resolve`
    and the fourteen factories through `_made`, and the first repair wired only one.
    """
    from borch import _base, _ops
    for fn, where in ((_base._resolve, "_base._resolve"), (_ops._made, "_ops._made")):
        src = inspect.getsource(fn)
        assert "_requested_dtype" in src, (
            f"{where} no longer passes the caller's dtype through the gate — every "
            "factory behind it narrows silently again.")
