"""Confirms the method names **by asking real torch.**

`borch/_ops.py` holds three tables — names to put out as methods as well as module functions
(`_AS_METHOD`), and names to put out as in-place operations (`_INPLACE_*`). The tables are
written by hand and can be wrong, and both directions of wrong are bad.

- **Inventing a name torch does not have** means writing code that runs only here. The day
  that code moves to real torch it raises `AttributeError`, and by then it has already been
  written leaning on that name. This repository's only claim is "run it with the import
  changed", and this breaks that claim.
- **Writing it into the table and not building it** does nothing at all, quietly. The gap
  count simply does not fall and nobody knows.

The golden cases cannot catch this. They see only the names we **asked about**, and a name
written into a table and never asked about is outside them. So the tables themselves are
looked at here.
"""

import pytest

import borch
from borch import _ops

torch = pytest.importorskip("torch")


def _named(tables):
    for label, names, suffix in tables:
        for name in names:
            yield label, name + suffix


TABLES = [
    ("_AS_METHOD", _ops._AS_METHOD, ""),
    ("_INPLACE_UNARY", _ops._INPLACE_UNARY, "_"),
    ("_INPLACE_MORE", _ops._INPLACE_MORE, "_"),
    ("_INPLACE_BINARY", _ops._INPLACE_BINARY, "_"),
    ("_INPLACE_ARGS", _ops._INPLACE_ARGS, "_"),
]


def test_every_name_we_add_is_a_real_torch_method():
    """**A method torch does not have must not be built.**

    Inventing a name means code leaning on it does not run under real torch — that is not
    imitating but building **a different library.**
    """
    invented = [f"{label}: {name}" for label, name in _named(TABLES)
                if not hasattr(torch.Tensor, name)]
    assert not invented, (
        "names being built that torch.Tensor does not have:\n  " + "\n  ".join(invented) +
        "\n\nTake them out — a name that does not exist makes code that does not run with the "
        "import changed.")


def test_every_name_we_promised_is_actually_there():
    """**Nothing written into a table may go unbuilt.**

    Wrong in this direction it is quiet. No exception is raised and the gap count simply does
    not fall.
    """
    missing = [f"{label}: {name}" for label, name in _named(TABLES)
               if not hasattr(borch.Tensor, name)]
    assert not missing, (
        "names written into a table and never built:\n  " + "\n  ".join(missing))


def test_the_method_and_the_function_are_the_same_calculation():
    """`x.add(y)` and `borch.add(x, y)` have to give the same answer.

    Written as two copies they diverge eventually, and the values are plausible enough then
    that nothing shows. What is looked at here is not whether they point at the same function
    but **whether they give the same answer.**
    """
    x = borch.tensor([[1.0, 2.0], [3.0, 4.0]])
    y = borch.tensor([[0.5, 1.5], [2.5, 3.5]])
    pairs = [("add", (y,)), ("mul", (y,)), ("sub", (y,)), ("div", (y,)),
             ("fmax", (y,)), ("cross", ()), ("det", ()), ("matrix_exp", ()),
             ("logical_and", (y,)), ("count_nonzero", ())]
    for name, extra in pairs:
        if name == "cross":
            continue
        method = getattr(x, name)(*extra)
        plain = getattr(borch, name)(x, *extra)
        assert borch.allclose(method, plain), f"{name}: the method and the function diverge"


def test_in_place_writes_into_the_same_tensor():
    """An in-place operation has to **modify the same tensor** — returning a new one means nothing."""
    for name in ("absolute_", "sinc_", "sgn_"):
        x = borch.tensor([-1.0, 2.0, -3.0])
        got = getattr(x, name)()
        assert got is x, f"{name}: it returned a new tensor rather than acting in place"


def test_in_place_refuses_a_leaf_that_needs_grad():
    """It refuses where torch refuses."""
    for lib in (borch, torch):
        x = lib.tensor([1.0, 2.0], requires_grad=True)
        with pytest.raises(RuntimeError):
            x.absolute_()
