"""Every function taking a `dim` or an `index`, asked for one that does not exist.

**The cases are not written down; they are enumerated.** `tests/test_refusal_classes.py`
next door pins 29 refusals and says so honestly: it pins what it asks and nothing about
what it does not ask. Its floor counts rows, and a floor cannot count *variety* — both
that file and the protocol-surface sweep beside it had the right number of rows, all of
one kind, and each missed a defect for that reason.

This file is the other shape. It takes torch's own surface as the list: every public
function in `borch` that torch also has and whose signature carries a `dim`, called with
an axis of 7 on a two-dimensional tensor. Nobody chose the members. **Thirty-eight
functions, and four of them parted on a question no case list contained:**

    diff(x, dim=7)        answered as though dim were the last axis
    gradient(x, dim=7)    the same — `axis % ndim` wraps rather than complains
    nanmedian(x, dim=7)   the same
    unflatten(x, 7, …)    RuntimeError about an element count, where torch names the axis

Three silent wrong answers, all plausible tensors. The root was `_pos_dim`, which
converted a negative axis and checked nothing: callers that reach numpy inherited
`AxisError` — which subclasses `IndexError`, so they agreed with torch by luck — and
callers doing their own slicing inherited nothing at all.

## What enumeration costs

A generated sweep produces findings that look exactly like real ones and are not. Three
of the first seven were **the probe's own error**: `torch.split(t, size=1)` is a wrong
keyword, so torch raised `TypeError` about the call rather than about the axis, and the
row read as a disagreement. The discriminator is whether torch refused *the argument* or
*the call*, and it cannot be read off the exception — it has to be built into how the
call is made. Hence `CALLS` below: each entry is torch's own positional order, so a
`TypeError` from either side means this file is wrong, not the library.

That is the trade. A written list is honest about its coverage and blind outside it; an
enumerated one covers a surface nobody chose and needs a second mechanism to tell its
own mistakes from findings.
"""

import inspect

import numpy as np
import pytest
import torch

import borch

BAD_DIM = 7
DATA = [[1., 2.], [3., 4.]]


def _pieces(lib):
    return {
        "t": lib.tensor(np.asarray(DATA, dtype=np.float32)),
        "idx": lib.tensor(np.array([0], dtype=np.int64)),
        "flat": lib.tensor(np.array([[9., 9.]], dtype=np.float32)),
        "cell": lib.tensor(np.array([[9.]], dtype=np.float32)),
        "badi": lib.tensor(np.array([BAD_INDEX], dtype=np.int64)),
        "badi2": lib.tensor(np.array([[BAD_INDEX]], dtype=np.int64)),
    }


# The extra arguments a function needs before `dim` means anything, **in torch's own
# positional order**. A name absent here is called as `f(t, dim=BAD_DIM)`.
#
# Written positionally on purpose: a keyword torch does not have raises `TypeError`
# from the call rather than from the axis, and reads as a finding. `test_the_sweep_asks
# _about_the_axis_and_not_about_the_call` is what holds that line.
CALLS = {
    "chunk": lambda v: ((v["t"], 2), {"dim": BAD_DIM}),
    "cosine_similarity": lambda v: ((v["t"], v["t"]), {"dim": BAD_DIM}),
    "index_add": lambda v: ((v["t"], BAD_DIM, v["idx"], v["flat"]), {}),
    "index_copy": lambda v: ((v["t"], BAD_DIM, v["idx"], v["flat"]), {}),
    "index_fill": lambda v: ((v["t"], BAD_DIM, v["idx"], 0.0), {}),
    "index_reduce": lambda v: ((v["t"], BAD_DIM, v["idx"], v["flat"], "prod"), {}),
    "narrow": lambda v: ((v["t"], BAD_DIM, 0, 1), {}),
    "repeat_interleave": lambda v: ((v["t"], 2), {"dim": BAD_DIM}),
    "scatter": lambda v: ((v["t"], BAD_DIM, v["idx"].reshape(1, 1), v["cell"]), {}),
    "scatter_add": lambda v: ((v["t"], BAD_DIM, v["idx"].reshape(1, 1), v["cell"]), {}),
    "scatter_reduce":
        lambda v: ((v["t"], BAD_DIM, v["idx"].reshape(1, 1), v["cell"], "sum"), {}),
    "select": lambda v: ((v["t"], BAD_DIM, 0), {}),
    "select_scatter": lambda v: ((v["t"], v["t"][0], BAD_DIM, 0), {}),
    "slice_scatter": lambda v: ((v["t"], v["flat"]), {"dim": BAD_DIM}),
    "split": lambda v: ((v["t"], 1), {"dim": BAD_DIM}),
    "split_with_sizes": lambda v: ((v["t"], [1, 1]), {"dim": BAD_DIM}),
    "tensor_split": lambda v: ((v["t"], 2), {"dim": BAD_DIM}),
    "unflatten": lambda v: ((v["t"], BAD_DIM, (1, 2)), {}),
    "unsqueeze": lambda v: ((v["t"], BAD_DIM), {}),
}

# Names whose `dim` this cannot reach with one tensor, each with the reason. Empty
# today; kept so that a new one is written down rather than dropped.
UNREACHABLE: dict[str, str] = {}

LEAST_FUNCTIONS = 30

# ── the second axis: an index that is not in the tensor ──────────────────────
#
# **Same enumeration, different question, and it parted six ways.** Fourteen public
# functions take an `index` or `indices`; the six below all raised the wrong class.
# None answered silently, which is worth recording as the negative result it is —
# the `dim` axis had three silent answers and this one has none, so "enumerate a
# surface and find silent wrong answers" is not a law.
#
# **torch is not consistent with itself here**, and matching torch means matching
# that: `scatter`, `scatter_add`, `scatter_reduce` and `index_add` refuse with
# `RuntimeError`, while `select` and `put` refuse with `IndexError`. A single rule
# would have been tidier and wrong for half the surface.
BAD_INDEX = 9

INDEX_CALLS = {
    "index_add": lambda v: ((v["t"], 0, v["badi"], v["flat"]), {}),
    "index_copy": lambda v: ((v["t"], 0, v["badi"], v["flat"]), {}),
    "index_fill": lambda v: ((v["t"], 0, v["badi"], 0.0), {}),
    "index_reduce": lambda v: ((v["t"], 0, v["badi"], v["flat"], "prod"), {}),
    "index_put": lambda v: ((v["t"], (v["badi"],), v["flat"]), {}),
    "index_put_": lambda v: ((v["t"], (v["badi"],), v["flat"]), {}),
    "put": lambda v: ((v["t"], v["badi"], v["flat"].reshape(2)), {}),
    "scatter": lambda v: ((v["t"], 0, v["badi2"], v["cell"]), {}),
    "scatter_add": lambda v: ((v["t"], 0, v["badi2"], v["cell"]), {}),
    "scatter_reduce": lambda v: ((v["t"], 0, v["badi2"], v["cell"], "sum"), {}),
    "select": lambda v: ((v["t"], 0, BAD_INDEX), {}),
    "select_scatter": lambda v: ((v["t"], v["t"][0], 0, BAD_INDEX), {}),
    "take": lambda v: ((v["t"], v["badi"]), {}),
}

# `unravel_index` takes an `indices` and means something else by it — the index is
# into a shape the caller supplies, not into the tensor. Named rather than dropped.
NOT_AN_INDEX_INTO_THE_TENSOR = {
    "unravel_index": "`indices` are flat positions into the `shape` argument",
}


def _names():
    """Public `borch` functions torch also has, whose signature carries a `dim`."""
    out = []
    for name in sorted(dir(borch)):
        if name.startswith("_"):
            continue
        mine, theirs = getattr(borch, name, None), getattr(torch, name, None)
        if theirs is None or not callable(mine) or inspect.isclass(mine):
            continue
        try:
            params = inspect.signature(mine).parameters
        except (TypeError, ValueError):
            continue
        if "dim" in params and name not in UNREACHABLE:
            out.append(name)
    return out


def _index_names():
    """Public `borch` functions torch also has, whose signature carries an index."""
    out = []
    for name in sorted(dir(borch)):
        if name.startswith("_") or name in NOT_AN_INDEX_INTO_THE_TENSOR:
            continue
        mine, theirs = getattr(borch, name, None), getattr(torch, name, None)
        if theirs is None or not callable(mine) or inspect.isclass(mine):
            continue
        try:
            params = inspect.signature(mine).parameters
        except (TypeError, ValueError):
            continue
        if "index" in params or "indices" in params:
            out.append(name)
    return out


def _raised(lib, name, table=None):
    table = CALLS if table is None else table
    build = table.get(name, lambda v: ((v["t"],), {"dim": BAD_DIM}))
    args, kw = build(_pieces(lib))
    try:
        getattr(lib, name)(*args, **kw)
    except Exception as e:                                    # noqa: BLE001
        return type(e)
    return None


@pytest.mark.parametrize("name", _names())
def test_an_axis_that_does_not_exist_is_refused_the_way_torch_refuses_it(name):
    theirs, ours = _raised(torch, name), _raised(borch, name)
    assert theirs is not None, (
        f"torch accepts `{name}(…, dim={BAD_DIM})` on a 2-D tensor, which means this "
        "row measures nothing. Fix the call in CALLS or record it in UNREACHABLE.")
    assert ours is not None, (
        f"torch refuses `{name}(…, dim={BAD_DIM})` with {theirs.__name__} and this "
        "answers. An axis that does not exist producing a plausible tensor is the "
        "defect this file was written for — three of them were found this way.")
    assert issubclass(ours, theirs), (
        f"`{name}`: torch raises {theirs.__name__}, we raise {ours.__name__}, and "
        f"`except {theirs.__name__}` does not catch it.")


def test_the_sweep_asks_about_the_axis_and_not_about_the_call():
    """**A `TypeError` from torch means this file made the call wrong.**

    Three of the first seven disagreements were exactly that: `split(t, size=1)`
    against a torch signature spelling it `split_size_or_sections`, so torch refused
    the keyword and the row read as a finding about the axis. Nothing in the
    exception says which it was — the call has to be built so that the question is
    unambiguous, and then checked that it still is.
    """
    wrong = [n for n in _names() if _raised(torch, n) is TypeError]
    assert not wrong, (
        "torch raised TypeError for these, which is a complaint about the call and "
        f"not about the axis: {wrong}\n"
        "  Fix their entry in CALLS to torch's own positional order.")


def test_the_sweep_still_finds_functions_to_ask():
    """The enumeration's own absence. A renamed `dim`, a `dir()` that stops
    answering, an import that half-fails — each leaves this file passing with
    nothing swept, and a per-row parametrisation reports zero rows as success.
    """
    found = _names()
    assert len(found) >= LEAST_FUNCTIONS, (
        f"only {len(found)} functions with a `dim` were found, "
        f"{LEAST_FUNCTIONS} expected — the enumeration stopped working, not the code")


def test_every_extra_call_names_a_function_that_exists():
    """A `CALLS` entry for a name nobody sweeps is a fold that fires on nothing —
    it would sit there looking like coverage."""
    unused = sorted(set(CALLS) - set(_names()))
    assert not unused, (
        f"these have a CALLS entry and are not swept: {unused}\n"
        "  Either the name moved, or its signature no longer carries a `dim`.")


@pytest.mark.parametrize("name", sorted(INDEX_CALLS))
def test_an_index_that_is_not_there_is_refused_the_way_torch_refuses_it(name):
    theirs = _raised(torch, name, INDEX_CALLS)
    ours = _raised(borch, name, INDEX_CALLS)
    assert theirs is not None, (
        f"torch accepts `{name}` with an index of {BAD_INDEX} into a size-2 axis, "
        "so this row measures nothing.")
    assert ours is not None, (
        f"torch refuses it with {theirs.__name__} and this answers — an index "
        "outside the tensor producing a plausible answer.")
    assert issubclass(ours, theirs), (
        f"`{name}`: torch raises {theirs.__name__}, we raise {ours.__name__}. "
        "torch is not uniform here — `scatter` says RuntimeError and `select` says "
        "IndexError — so match the function, not a rule.")


def test_every_index_taking_function_is_asked_or_explained():
    """The index axis's own absence, and its unswept remainder.

    A name that takes an index and is in neither table is a function nobody asks
    about, which reads from outside exactly like a function that agrees.
    """
    missing = sorted(set(_index_names()) - set(INDEX_CALLS))
    assert not missing, (
        f"these take an index and are not swept: {missing}\n"
        "  Add a call to INDEX_CALLS, or record it in "
        "NOT_AN_INDEX_INTO_THE_TENSOR with the reason.")
    stale = sorted(set(INDEX_CALLS) - set(_index_names()))
    assert not stale, f"these have a call and are no longer found: {stale}"


def test_the_index_sweep_asks_about_the_index_and_not_about_the_call():
    """As the `dim` sweep: torch answering `TypeError` means this file is wrong."""
    wrong = [n for n in INDEX_CALLS if _raised(torch, n, INDEX_CALLS) is TypeError]
    assert not wrong, (
        f"torch complained about the call rather than the index for: {wrong}")
