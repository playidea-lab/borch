"""**The methods whose real caller is not our code.**

Every other comparison in this repository asks by calling. The golden calls a case, the
signature axes read a declaration, `test_called_twice` calls twice. So the surface it
all covers is *the surface we thought to call* — and `__len__`, `__iter__`,
`__getitem__` and `__array__` are called by **numpy**, in an order nobody here decided,
with silence taken for an answer.

That is not a hypothetical. `len()` returned `0` for a zero-dimensional tensor where
torch raises, so `np.asarray(t)` walked one level past the last axis and built a
`(3, 5, 4, 0)` array from a `(3, 5, 4)` tensor: empty, correctly typed, wrong rank, no
error. Four separate `vision::` rows reported four different symptoms of it.

Sweeping the rest of the protocol against torch found **26 of 64 pairs parting**, in
three kinds:

- **`__array__` was absent entirely.** Without it numpy falls back to guessing with
  `len()` and `__getitem__`, and the guess **loses axes**: `np.asarray` on a `(0, 3)`
  tensor came back `(0,)`, on `(2, 0, 4)` it came back `(2, 0)`. The sequence walk
  descends by indexing and a zero-length axis has nothing to descend into, so every
  axis past the first zero was never discovered. `t.shape` was right the whole time.
- **`__int__` and `__index__` were missing.** `int(t)` was a `TypeError` where torch
  answers, and `xs[t]` — a tensor used as an index, which `__index__` exists for — was
  an `AttributeError` from inside the interpreter, naming nothing.
- **Two exception classes were crossed.** `bool(t)` on many elements let *numpy's*
  `ValueError` escape where torch raises `RuntimeError`; `int(t)` and `float(t)` raised
  this repository's `RuntimeError` where torch raises `ValueError`. No value differed,
  so nothing here could see it — what differs is which `except` clause catches, and
  `if t:` inside a `try` is ordinary code.

## What this file freezes

The whole grid, both sides, values and exception classes together. A protocol method is
not a function whose answer we choose: **torch decides, numpy asks, and our only job is
to agree.** So the table below is not a budget to work down — every row is expected to
agree, and one that stops agreeing is a defect rather than a to-do.

The shapes are chosen for where protocols are awkward: zero-dimensional (no axes to
walk), empty (a length with nothing behind it), one element (scalar conversions work),
many (they must not), and **empty rows with a real second axis**, which is the shape
that produced the original defect and which an earlier fix at a different shape did not
reach.

## What the pair count is, said exactly

72 pairs is not 72 answers compared. Measured: **46 pairs where both sides produce a
value, and 26 where both refuse** — the refusals agreeing about *whether*, not *what*.

Both are real comparisons and the refusals are not filler: two of the defects this file
found were exception *classes*, which live entirely in that half. But a reader given
only the total would take it for 72 answers, and a sister file spent an afternoon on the
version of that mistake where a fixture silently narrowed its question and left 98
comparisons of a possible 216 with every count still looking plausible.

`test_the_grid_keeps_asking_both_kinds_of_question` pins the split rather than the size.
A change that turns answers into refusals leaves the total untouched, and the total is
the only thing a floor can see.
"""

import pathlib
import sys

import numpy as np
import pytest

pytest.importorskip("torch")
import torch                                                     # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import borch                                                     # noqa: E402

SHAPES = {
    "0-d": np.array(3.0, dtype=np.float32),
    "0-d int": np.array(3, dtype=np.int64),
    "empty": np.zeros((0,), dtype=np.float32),
    "one": np.array([5.0], dtype=np.float32),
    "one int": np.array([5], dtype=np.int64),
    "many": np.array([1.0, 2.0, 3.0], dtype=np.float32),
    # **`many` as floats was not enough**, and the gap had a defect in it. `__index__`
    # refuses a float on dtype alone and never reaches its size check, so the only shape
    # that could exercise that second half is *integer and more than one element*. It
    # was missing, and `index()` on this returned `RuntimeError` where torch says
    # `TypeError`.
    #
    # A grid decides what its instrument can see. Every probe here is crossed with every
    # shape, so a shape absent from this dict is a question nobody asks — the same fault
    # as a parser whose correctness turned out to be a property of its input.
    "many int": np.array([1, 2, 3], dtype=np.int64),
    "2-d": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
    # **The shape the original defect lived at.** A length of 0 and a second axis that
    # still exists — the one an implementation guessing from `len()` cannot recover.
    "2-d empty rows": np.zeros((0, 3), dtype=np.float32),
}

PROBES = {
    "len()": len,
    "bool()": bool,
    "int()": int,
    "float()": float,
    # Materialised, because a generator that raises on the first `next` and one that
    # raises when built are the same to a caller writing `for x in t`.
    "iter()": lambda t: [type(x).__name__ for x in t][:2],
    "index()": lambda t: t.__index__(),
    "np.asarray()": lambda t: np.asarray(t).shape,
    "np.array()": lambda t: np.array(t).shape,
}

# Names torch's `Tensor` carries that this file does not probe, with the reason. **By
# name rather than counted**, so one that becomes probeable does not sit here forever.
NOT_PROBED = {
    "__hash__": "identity on both sides; nothing to compare",
    "__repr__": "the golden freezes the text of these already, case by case",
    "__str__": "the same",
    "__format__": "delegates to __float__, which is probed",
    "__contains__": "`x in t` compares values and is a golden question, not a protocol one",
    "__complex__": "there is no complex dtype on the borch.ts side; refused elsewhere",
    "__dlpack__": "a buffer-sharing protocol with no counterpart here",
    "__dlpack_device__": "the same",
}


def _ask(fn, t):
    """`repr` of the answer, or the exception's class name.

    **The class, not the message.** torch's wording is not something to copy verbatim —
    `_like_torch` exists precisely so ours can say more — but the class decides which
    `except` catches, and that is behaviour.
    """
    try:
        with np.errstate(all="ignore"):
            return repr(fn(t))
    except Exception as e:                                       # noqa: BLE001
        return type(e).__name__


GRID = [(probe, shape) for probe in PROBES for shape in SHAPES]


def test_the_probe_list_names_things_that_exist_on_both_sides():
    """Written by hand, so it can name something gone.

    A probe that raises `AttributeError` on **both** sides looks like agreement — the
    grid below would freeze `AttributeError == AttributeError` and call it a pass. That
    is the absorbing bucket in its cheapest form, so the names are checked directly.
    """
    absent = [name for name in ("__len__", "__bool__", "__int__", "__float__",
                                "__iter__", "__index__", "__array__")
              if not hasattr(borch.Tensor, name) or not hasattr(torch.Tensor, name)]
    assert not absent, (
        f"a protocol method this file probes is missing from one side: {absent}\n"
        "  Two `AttributeError`s compare equal, so the grid would report agreement.")


def test_nothing_probed_is_also_written_down_as_unprobed():
    """`NOT_PROBED` explains what the grid leaves out; it must not explain something the
    grid covers. A row here that is also probed reads as a limit that is not real."""
    both = sorted(n for n in NOT_PROBED
                  if n.strip("_").replace("()", "") in
                  {p.strip("()").replace("np.", "") for p in PROBES})
    assert not both, f"listed as unprobed and also probed: {both}"


def test_the_grid_keeps_asking_both_kinds_of_question():
    """**Pins the split, not the size.**

    A pair where both sides raise agrees about *whether* the call is allowed; a pair
    where both answer agrees about *what*. Those are different questions and the total
    cannot tell them apart — turn every answering pair into a refusing one and the count
    of pairs is unchanged, while the file has stopped comparing values entirely.

    Set as a floor on each half rather than an exact figure, because adding a shape
    should be cheap. The numbers when written were 46 answering and 26 refusing.
    """
    answering = refusing = 0
    for probe, shape in GRID:
        fn, data = PROBES[probe], SHAPES[shape]
        got = _ask(fn, borch.tensor(data.copy()))
        if got.endswith("Error"):
            refusing += 1
        else:
            answering += 1
    assert answering >= 40, (
        f"only {answering} pairs get a value out of the core, and there were 46.\n"
        "  The grid can be the same size and have stopped comparing answers.")
    assert refusing >= 20, (
        f"only {refusing} pairs are refusals, and there were 26.\n"
        "  Two of this file's findings were exception classes, which live only here.")


@pytest.mark.parametrize("probe,shape", GRID, ids=lambda v: v)
def test_the_core_answers_the_protocol_the_way_torch_does(probe, shape):
    """**Every pair in the grid, values and exception classes alike.**

    Not a budget. numpy decides when to ask these and takes the answer without
    checking, so a difference here is not a missing feature — it is a wrong answer
    handed to a caller that has no way to notice.
    """
    fn, data = PROBES[probe], SHAPES[shape]
    ours = _ask(fn, borch.tensor(data.copy()))
    theirs = _ask(fn, torch.tensor(data.copy()))
    assert ours == theirs, (
        f"{probe} on a {shape} tensor: core says {ours}, torch says {theirs}.\n"
        "  These are the methods numpy calls on its own initiative. A wrong answer\n"
        "  here does not surface where it was given — it surfaces as a shape or a\n"
        "  dtype somewhere downstream, blamed on whatever touched it last.")
