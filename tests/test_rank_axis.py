"""Which **input ranks** each function accepts, against real torch.

    uv run --extra dev pytest tests/test_rank_axis.py -q

## The question the other two axes do not ask

`tests/ts_axis.py` asks whether a name is there. `tests/torch_signatures_core.py`
and `tests/ts_signatures.py` ask whether the parameter list matches. A function can
pass all three and still take a different set of shapes, because **rank is not in
the name and not in the signature.**

Ten functions parted here while both axes read green:

    t                 answered at rank 3 by transposing the first two axes
    trace             answered at rank 3 with a batched diagonal sum
    pdist             answered at rank 3 with a (1, 3)
    cartesian_prod    flattened and answered
    combinations      flattened and answered
    chain_matmul      multiplied vectors
    vander            answered from a matrix
    tril triu         answered from a vector
    cholesky_inverse  answered from a vector

torch calls every one of those undefined. Nothing raised here; a tensor came back.

## Which direction is worse

**Answering where torch refuses is the quiet one.** The value flows, the loss goes
down, and the divergence surfaces at the port rather than at the call.

**Refusing where torch accepts is loud** — a peer hit it in borch.ts the same day
(`nn.Linear` at 3-D, which is 2-D only) — and loud is better for the person while
being *exactly as invisible to a checker*. `TORCH_REACHES_FURTHER_BY_POSITION` is 0
and cannot see it: that counter reads `TypeError` from a positional call, and this
is a `RuntimeError` about a shape. A number that answers a narrower question than
its name is the shape this repository has now hit five times.

## What the sweep cost

The probe was built to ask about rank and **the largest thing it found was not a
rank question.** `torch.where(condition)` — the one-argument form — was absent, and
so was `nonzero(as_tuple=)`. Both landed in the report as "ours accepts no rank",
because a function that raises at every rank looks like a rank row to an instrument
that only measures rank. It is an arity row.

Both of those sit in the signature axis's **"torch is C"** bucket: `inspect` says
"no signature found for builtin" for `where` and `nonzero`, so neither was ever
compared. That bucket does not mean the two agree. It means nothing was asked — and
two real absences were living inside it while the axis reported 0 keyword-only
absences.

So the honest summary of this file's cost line, in the form `test_axis_sweep.py`
keeps:

    rank     the probe's largest find was not about rank
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import borch

# One shape per rank. Square, so a function wanting a matrix is not refused for
# being non-square when the question is only about how many axes there are.
SHAPES = {0: (), 1: (4,), 2: (4, 4), 3: (2, 4, 4), 4: (2, 2, 4, 4)}

# Names where the two libraries disagree for a reason already written down
# elsewhere, rather than because a rank rule is missing.
EXCUSED = {
    # Refused wholesale with a message in `borch/_ops.py`; not a rank rule.
    "hash_tensor", "lobpcg",
}


def _configures():
    """The names that change global state, borrowed rather than restated.

    **This file called every one of them before it borrowed the list.** A sweep
    over "every unary name" reaches `manual_seed` and `set_printoptions` along
    with the arithmetic, and the damage lands in a *different file*: this one
    passed alone and `test_why_failing.py` failed in the same run, with a message
    naming neither rank nor this test.

    It is the third time a probe in this repository has done it — the dtype axis
    got there first and `CONFIGURES` exists because of it. Writing a second copy
    of the list would put the next sweep one edit away from the same evening, so
    the import is the point: one list, and a name added for one axis protects all
    of them.
    """
    from test_axis_sweep import CONFIGURES
    return set(CONFIGURES)


def _accepts(lib, name, shape):
    try:
        getattr(lib, name)(lib.tensor(np.ones(shape, dtype=np.float32)))
        return True
    except Exception:
        return False


def _unary_names():
    """Names present on both sides that take one tensor and nothing else."""
    import inspect
    skip = _configures()
    for name in sorted(dir(borch)):
        if name.startswith("_") or name.endswith("_") or name in EXCUSED:
            continue
        if name in skip:
            continue
        ours, theirs = getattr(borch, name, None), getattr(torch, name, None)
        if theirs is None or not callable(ours) or inspect.isclass(ours):
            continue
        yield name, ours, theirs


def test_the_accepted_ranks_match_torch():
    """Neither wider nor narrower than torch, for every unary name on both sides."""
    wider, narrower = [], []
    for name, _ours, _theirs in _unary_names():
        mine = {r for r in SHAPES if _accepts(borch, name, SHAPES[r])}
        theirs = {r for r in SHAPES if _accepts(torch, name, SHAPES[r])}
        if not theirs:                     # torch refuses a bare tensor: not our question
            continue
        if mine - theirs:
            wider.append(f"{name}: torch {sorted(theirs)}, ours {sorted(mine)}")
        if theirs - mine:
            narrower.append(f"{name}: torch {sorted(theirs)}, ours {sorted(mine)}")

    assert not wider, (
        "these answer a question torch calls undefined — the quiet direction, "
        "because the value flows and the parting shows up at the port:\n    "
        + "\n    ".join(wider)
        + "\n  `_ops._rank` is the guard; carry torch's own message word for word.")
    assert not narrower, (
        "these refuse a rank torch accepts — loud for the caller and invisible "
        "to every counter this repository has:\n    " + "\n    ".join(narrower))


def test_where_and_nonzero_carry_the_forms_inspect_cannot_read():
    """The two absences that were living inside the "torch is C" bucket.

    Kept as their own test rather than folded into the sweep above, because the
    sweep would report them as rank rows and they are not.
    """
    x = np.array([[0.0, 2.0], [3.0, 0.0]], dtype=np.float32)
    mine = [t.tolist() for t in borch.where(borch.tensor(x))]
    theirs = [t.tolist() for t in torch.where(torch.tensor(x))]
    assert mine == theirs, "where(condition) is nonzero(as_tuple=True) under another name"

    mine = [t.tolist() for t in borch.nonzero(borch.tensor(x), as_tuple=True)]
    theirs = [t.tolist() for t in torch.nonzero(torch.tensor(x), as_tuple=True)]
    assert mine == theirs

    with pytest.raises(RuntimeError):
        borch.where(borch.tensor(x) > 1, borch.tensor(x))
