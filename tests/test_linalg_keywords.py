"""`linalg`'s parameter names, held by **calling torch** rather than by reading it.

torch implements `linalg` in C, so the signature axis cannot see into it: of its 42
rows, **36 are filed `torch is C` and judged by nothing.** The names in the core came
from a one-off sweep that called torch with each candidate keyword, and borch.ts then
took the same table rather than measuring again — a reasonable call, and it rests on
*"if the core is wrong the core↔torch axis catches it"*, which is true of six of the
forty-two.

So two libraries now carry a table with no standing check behind it. This is that
check. It calls torch, asks which keyword it answers to, and requires the core to
answer to the same one.

## Why the names are not one rule

torch is **not consistent inside its own `linalg`**:

    det(A)      cholesky(input)     multi_dot(tensors)
    inv(A)      eig(input)          lu_solve(LU)
    qr(A)       norm(input)         ldl_solve(LD)
    svd(A)      matrix_rank(input)  vecdot(x)

Roughly nine take `A` and fifteen take `input`. A summary written from five
docstrings said "fifteen use `A`" — **a claim about every row, checked against six**
— and following it would have renamed nine correctly and fifteen wrongly.

## The top level is asked too

Six of these are one function in both namespaces here, and torch spells three of them
differently: `torch.det(input=…)` is taken and `torch.det(A=…)` is not, while
`linalg.det` is the reverse. A check that looked only at `linalg` would pass while the
top-level spelling had been taken away — which happened, to `slogdet`, `qr` and `lu`,
and was caught by asking both.

## What a green run does not say

- **Not that the values agree.** The golden holds that.
- **Not that the rest of the list matches.** Only the first parameter is asked here,
  because that is the one the table decided.
"""

import pathlib
import sys
import warnings

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

import numpy as np   # noqa: E402

import borch         # noqa: E402

SQUARE = np.array([[2.0, 1.0], [1.0, 3.0]], dtype=np.float32)
VECTOR = np.array([1.0, 2.0, 3.0], dtype=np.float32)

# Every spelling torch uses for a first argument anywhere in this namespace. Asked in
# this order, and the first one torch answers to is taken as its name.
CANDIDATES = ("A", "input", "x", "tensors", "LU", "LD")

# **A refusal is one of these three sentences and nothing else.** Every other failure
# means the keyword was accepted and something later went wrong — a wrong shape, a
# missing second argument. Four probes in this repository have been wrong by treating
# any exception as a refusal, or any non-matching exception as success.
REFUSALS = ("missing", "invalid combination", "unexpected keyword")


def _value(lib, name):
    make = borch.tensor if lib is borch else torch.tensor
    if name == "multi_dot":
        return [make(SQUARE.copy()), make(SQUARE.copy())]
    if name in ("vecdot", "cross", "vander"):
        return make(VECTOR.copy())
    return make(SQUARE.copy())


def _answers(fn, keyword, value):
    """Whether `fn` has a parameter of that name — not whether the call succeeds."""
    try:
        fn(**{keyword: value})
        return True
    except TypeError as exc:
        first = str(exc).splitlines()[0]
        if "missing" in first and "required" in first:
            return False
        return not any(mark in first for mark in REFUSALS[1:])
    except Exception:                                            # noqa: BLE001
        return True


def _pairs(ours_ns, theirs_ns, only=None):
    """`(name, torch's keyword, ours, theirs)` for every callable in both."""
    out = []
    for name in sorted(n for n in dir(ours_ns) if not n.startswith("_")):
        if only is not None and name not in only:
            continue
        theirs = getattr(theirs_ns, name, None)
        ours = getattr(ours_ns, name, None)
        if theirs is None or ours is None:
            continue
        if not callable(theirs) or not callable(ours) or isinstance(theirs, type):
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            want = next((k for k in CANDIDATES
                         if _answers(theirs, k, _value(torch, name))), None)
        if want is None:
            continue
        out.append((name, want, ours, theirs))
    return out


def _linalg_pairs():
    return _pairs(borch.linalg, torch.linalg)


def _top_pairs():
    """The same-named ones at the top level. **Six are one function here**, so a
    rename for `linalg` can take the top-level spelling away."""
    shared = {n for n in dir(borch.linalg) if not n.startswith("_")}
    return _pairs(borch, torch, only=shared)


def test_linalg_takes_the_keyword_torch_answers_to():
    wrong = []
    for name, want, ours, _theirs in _linalg_pairs():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if not _answers(ours, want, _value(borch, name)):
                took = [k for k in CANDIDATES
                        if _answers(ours, k, _value(borch, name))]
                wrong.append(f"linalg.{name}: torch answers to {want!r}, "
                             f"the core answers to {took or 'none of them'}")
    assert not wrong, (
        "a `linalg` parameter is not the name torch takes:\n  " + "\n  ".join(wrong)
        + "\n\n  torch is not consistent here — about nine take `A` and fifteen "
          "`input`,\n  plus `tensors`, `LU`, `LD` and `x`. The name comes from "
          "calling torch,\n  never from its prose: `true_divide` documents `value` "
          "and takes `other`.")


def test_the_top_level_keeps_its_own_spelling():
    """`det`, `qr` and `slogdet` are `input` at the top and `A` under `linalg`, and
    six of these are one function here. Fixing one namespace really did break the
    other, three times, before the wrappers went in."""
    wrong = []
    for name, want, ours, _theirs in _top_pairs():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if not _answers(ours, want, _value(borch, name)):
                took = [k for k in CANDIDATES
                        if _answers(ours, k, _value(borch, name))]
                wrong.append(f"borch.{name}: torch answers to {want!r}, "
                             f"the core answers to {took or 'none of them'}")
    assert not wrong, (
        "a top-level name lost its own spelling to `linalg`'s:\n  "
        + "\n  ".join(wrong))


def test_the_two_namespaces_really_do_disagree():
    """**The positive control.** If torch spelled both namespaces the same, the two
    tests above would pass with the wrappers deleted, and this file would be measuring
    a distinction that is not there.

    **Five names, and the first version of this test said three.** It was written
    from the three that needed a wrapper and asserted that those were the whole
    disagreement — a claim about every row taken from the rows that had caused work.
    `diagonal` and `svd` are spelled differently too and needed nothing, because they
    are already two separate functions here (`diagonal_linalg`, `linalg_svd`).

    So the two facts are pinned apart: which names torch spells differently, and
    which of those share one implementation here. A name moving between the two
    lists is what creates or retires a wrapper — the first list alone cannot say.
    """
    split, shared = [], []
    for name in sorted(n for n in dir(borch.linalg) if not n.startswith("_")):
        lin, top = getattr(torch.linalg, name, None), getattr(torch, name, None)
        if lin is None or top is None or isinstance(lin, type):
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            a = next((k for k in CANDIDATES if _answers(lin, k, _value(torch, name))),
                     None)
            b = next((k for k in CANDIDATES if _answers(top, k, _value(torch, name))),
                     None)
        if a and b and a != b:
            split.append(name)
            if getattr(borch, name, None) is getattr(borch.linalg, name, None):
                shared.append(name)

    assert split == ["det", "diagonal", "qr", "slogdet", "svd"], (
        f"torch's two namespaces disagree on {split}, and this file was written "
        "around five.\n  A name arriving or leaving is a thing to read, not a "
        "number to nudge.")
    assert shared == [], (
        f"{shared} are spelled differently by torch in the two namespaces **and are "
        "one\n  function here**, so one of the two spellings is being lost. `det`, "
        "`qr` and\n  `slogdet` were in this state and have wrappers; anything here "
        "needs one too.")


def test_the_sweep_reaches_the_namespace():
    """**A sweep that judges nothing passes every assertion above.** If `_answers`
    starts returning `None` for everything — a classifier change, a torch release
    that raises differently — the three tests go quiet rather than red.

    That is the fault this repository has found in three separate instruments, so
    the floor is pinned: `linalg` has 42 names and most of them have to be reached.
    """
    reached = len(_linalg_pairs())
    assert reached >= 29, (
        f"only {reached} `linalg` names were judged — the sweep has stopped "
        "reaching them,\n  and the checks above would pass on an empty set.")
    # **29 of 42, and the thirteen missing are a limit rather than a gap.** They need
    # a second argument before the first keyword can be tested — `lu_solve` wants
    # pivots and a right-hand side, `solve_triangular` wants `upper` — so no
    # candidate alone gets past the signature. Their names were measured by hand,
    # with the extra arguments supplied, and are carried in the core's own comments.
    #
    # The floor is written `>= 29` rather than `> 30` because the first version was
    # guessed at, went red on its own first run, and would have been "fixed" by
    # lowering it without anyone asking which thirteen were absent.
