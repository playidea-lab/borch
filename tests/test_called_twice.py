"""**Calls every in-place operation twice**, which nothing else here does.

Every other check in this repository has the same shape: make a fresh object, call
once, compare the answer to torch's. The golden runs a case once. Both signature axes
read a declaration. `parity.ts` checks a value. So a defect that only appears on the
*second* call is invisible to all of them at once.

That is not hypothetical. This file was written after four such defects turned up in
one day, and three of them were found by the probe it grew out of:

- **`digamma_` did not answer at all.** The recurrence lifts `x` by adding one until
  it clears six, and `-inf + 1` is `-inf`, so the loop never ended. One call turns
  −2.0 into an infinity and the second hangs on it — no error, no output, the process
  simply stops. Three functions carried the same loop.
- **`erfinv_` answered outside its domain.** `erf` maps the reals onto [−1, 1] and its
  inverse is defined nowhere else, but a `clip` meant to keep the tail formula finite
  kept it producing numbers: 4.7e21 at 1.5, where torch says `nan`. **One call is
  enough to leave the interval**, so only a second call could reach it.
- **`lgamma_` missed its poles by a hair.** `sin(π·−2)` in float64 is 2.4e-16 rather
  than 0, so the reflection formula gave 36.4 where torch gives `inf` — finite, and
  plausible enough to travel through a loss unnoticed.

The fourth was `EmbeddingBag`'s `max_norm`, which renormalises the weight table **in
place**: an implementation that renormalised a copy would agree with torch on the
first call and part on the second, and every instrument here would have called it
correct.

A fifth came from widening it. Giving the **binary** family its second operand — the
thirty-nine names nothing here had ever handed an argument to — found `igamma_` and
`igammac_` **ending the whole call**: `math.lgamma` reached through `np.vectorize`
raises `ValueError` at every non-positive integer, so one entry outside the domain took
down the tensor that contained it, where torch puts a `nan` in that one position and
answers for the rest. A batch with one bad row ended the training step rather than the
row.

## What it does not cover, said out loud

**142 of the 162 in-place names are probed.** Of the remaining 20, 8 draw from a
generator and 12 are asking a different question — and **none is unprobed for want of
an argument.** That is what the widening was for.

The count is the thing this file exists to distrust, so the 20 are listed **by name,
each with its reason**, and a test fails if any in-place name is in neither the probe
nor a table. A name that silently stops being probed shows up as a row that no longer
matches, rather than as a number that stayed the same.

Widening it paid twice. The binary family produced the fifth defect above. The index
and scatter family — the ones that mutate hardest — produced a sixth on its first run:
`as_strided_` twice. That one turned out to be an already-documented divergence, but
the comment recording it said it was visible *only on a write*, which measurement
showed was narrower than the truth. A reason that reads as complete and is not is worth
as much finding as a defect.

The matmul family and the callables produced nothing, which is also a result: it means
those were never a blind spot, only an unasked question, and now the difference between
the two is written down.
"""

import sys

import numpy as np
import pytest

pytest.importorskip("torch")
import torch                                                     # noqa: E402

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

import borch                                                     # noqa: E402

BASE = np.array([[1.5, -2.0, 3.0], [0.5, 4.0, -1.0]], dtype=np.float32)
OTHER = np.array([[2.0, 3.0, -1.5], [4.0, 0.5, 2.0]], dtype=np.float32)
BASE_I = np.array([[6, -4, 12], [3, 8, -2]], dtype=np.int64)
OTHER_I = np.array([[4, 6, 8], [2, 12, 3]], dtype=np.int64)

# For the index and scatter family. `COLS` is a permutation along the row so scatter
# lands somewhere different from where it read; `ROWS` names both rows so nothing is
# left untouched, which would let a wrong write hide in a row nobody wrote to.
SRC = np.array([[9.0, 8.0, 7.0], [5.0, 4.0, 3.0]], dtype=np.float32)
MASK = np.array([[True, False, True], [False, True, False]])
COLS = np.array([[0, 2, 1], [2, 1, 0]], dtype=np.int64)
ROWS = np.array([0, 1], dtype=np.int64)
FLAT = np.array([0, 4], dtype=np.int64)
PAIR = np.array([10.0, 20.0], dtype=np.float32)

# **Both operands hold negatives and a zero-crossing on purpose.** The domain edges are
# where this family goes wrong, and a pair of all-positive tensors makes every one of
# them agree.

# Arguments chosen so the operation changes something on **both** calls. One whose
# second call is a no-op cannot show anything either way, and a case that cannot fail
# is the shape this repository keeps finding.
ARGS = {
    "add_": (2.0,), "sub_": (1.0,), "mul_": (2.0,), "div_": (2.0,),
    "pow_": (2.0,), "clamp_": (-1.0, 2.0), "fill_": (3.0,),
    "clamp_min_": (0.0,), "clamp_max_": (1.0,), "renorm_": (2, 0, 1.0),
    "requires_grad_": (False,),
}

# The binary family, which takes the other operand as a tensor. **Held out of the probe
# until this was written**, and the first run of it found `igamma_` and `igammac_`
# ending the whole call — `math.lgamma` through `np.vectorize` raises at every
# non-positive integer, so one entry outside the domain took down the tensor that
# contained it, where torch puts a `nan` in that position and answers for the rest.
BINARY = (
    "arctan2_", "atan2_", "copysign_", "divide_", "eq_", "floor_divide_", "fmod_",
    "ge_", "greater_", "greater_equal_", "gt_", "heaviside_", "hypot_", "igamma_",
    "igammac_", "ldexp_", "le_", "less_", "less_equal_", "logical_and_", "logical_or_",
    "logical_xor_", "lt_", "multiply_", "ne_", "nextafter_", "not_equal_", "remainder_",
    "subtract_", "true_divide_", "xlogy_",
)

# The same, on integers — both libraries refuse these on a float tensor, and rightly.
BINARY_INT = (
    "bitwise_and_", "bitwise_left_shift_", "bitwise_or_", "bitwise_right_shift_",
    "bitwise_xor_", "gcd_", "lcm_",
)

# **Drawn from a generator, so two runs differ by design.** Not a finding, and not
# skipped silently either: a name that moves from here to the compared set because
# somebody made it deterministic should be noticed.
RANDOM = {
    "bernoulli_", "cauchy_", "exponential_", "geometric_", "log_normal_",
    "normal_", "random_", "uniform_",
}

# The index and scatter family, which mutate hardest. Each row builds its own
# arguments from the library's own constructor, since an index tensor from one side
# cannot be handed to the other.
#
# **The same indices on both calls**, deliberately: the second call writes over what
# the first wrote, which is the only arrangement that separates an implementation that
# accumulates from one that replaces, or one that kept a view of the source from one
# that copied it.
BUILT = {
    "masked_fill_": lambda T: (T(MASK.copy()), 7.0),
    "masked_scatter_": lambda T: (T(MASK.copy()), T(SRC.copy())),
    "index_fill_": lambda T: (0, T(ROWS.copy()), 7.0),
    "index_copy_": lambda T: (0, T(ROWS.copy()), T(SRC.copy())),
    "index_add_": lambda T: (0, T(ROWS.copy()), T(SRC.copy())),
    "index_put_": lambda T: ((T(ROWS.copy()),), T(SRC.copy())),
    "scatter_": lambda T: (1, T(COLS.copy()), T(SRC.copy())),
    "scatter_add_": lambda T: (1, T(COLS.copy()), T(SRC.copy())),
    "put_": lambda T: (T(FLAT.copy()), T(PAIR.copy())),
    "copy_": lambda T: (T(SRC.copy()),),
    "resize_as_": lambda T: (T(SRC.copy()),),
    "lerp_": lambda T: (T(SRC.copy()), 0.25),
    "addcmul_": lambda T: (T(SRC.copy()), T(SRC.copy())),
    "addcdiv_": lambda T: (T(SRC.copy()), T(SRC.copy())),
    "cumsum_": lambda T: (1,),
    "cumprod_": lambda T: (1,),
    "fill_diagonal_": lambda T: (7.0,),
    "swapaxes_": lambda T: (0, 1),
    "swapdims_": lambda T: (0, 1),
    "mvlgamma_": lambda T: (2,),
    "polygamma_": lambda T: (1,),
    "index_reduce_": lambda T: (0, T(ROWS.copy()), T(SRC.copy()), "prod"),
    "scatter_reduce_": lambda T: (1, T(COLS.copy()), T(SRC.copy()), "sum"),
    "set_": lambda T: (T(SRC.copy()),),
    # Callables, which the two libraries hand different things to — a Python float
    # here and a Python float there, but only because both drop to scalars for it.
    "apply_": lambda T: (lambda v: v * 2.0 + 1.0,),
    "map_": lambda T: (T(SRC.copy()), lambda a, b: a + b),
    "map2_": lambda T: (T(SRC.copy()), T(OTHER.copy()), lambda a, b, c: a + b * c),
}

# The matmul family, which needs shapes the tables above do not have — square, and for
# two of them batched. Kept separate rather than forced into `BASE`'s shape: reusing
# one receiver for everything is how a case ends up unable to fail.
SQUARE = np.array([[1.5, -2.0], [0.5, 4.0]], dtype=np.float32)
SQ_A = np.array([[2.0, 3.0], [1.0, -1.0]], dtype=np.float32)
SQ_B = np.array([[0.5, 2.0], [-1.0, 1.5]], dtype=np.float32)
SQ_V = np.array([2.0, -1.0], dtype=np.float32)
BATCH_A = np.stack([SQ_A, SQ_A * 0.5])
BATCH_B = np.stack([SQ_B, SQ_B * 2.0])

MATMUL = {
    "addmm_": (SQUARE, lambda T: (T(SQ_A.copy()), T(SQ_B.copy()))),
    "addmv_": (SQ_V, lambda T: (T(SQ_A.copy()), T(SQ_V.copy()))),
    "addr_": (SQUARE, lambda T: (T(SQ_V.copy()), T(SQ_V.copy()))),
    "addbmm_": (SQUARE, lambda T: (T(BATCH_A.copy()), T(BATCH_B.copy()))),
    "baddbmm_": (np.stack([SQUARE, SQUARE * -1.0]),
                 lambda T: (T(BATCH_A.copy()), T(BATCH_B.copy()))),
}

# Not probed, with what each one needs. **Listed rather than counted** — see the
# module docstring. Every row here is a place the second call has never been asked.
#
# **Empty, and kept.** Every name that needs arguments now gets them. An empty table
# says *nothing is owed*; a deleted one says nothing at all, and the next in-place
# method this file cannot build needs somewhere to be written down.
NEEDS_MORE: dict[str, str] = {}

# Probed by nothing here for a reason that is not "needs arguments".
NOT_A_SECOND_CALL = {
    "resize_": "changes the shape; a different question",
    "transpose_": "the second call undoes the first",
    "t_": "the second call undoes the first",
    "squeeze_": "the second call is a no-op",
    "unsqueeze_": "changes rank each time; a different question",
    "clip_": "torch refuses it with neither bound given",
    "bitwise_not_": "torch refuses it on a float tensor",
    "float_power_": "both refuse it in place on a float tensor, for the same reason",
    "as_strided_": (
        "a known divergence, reached by the second call rather than by a write. "
        "`as_strided` copies here and views in torch (the reason is at the line in "
        "_ops.py), so the first call leaves a 2x2 where torch leaves a 2x2 over the "
        "original eight-element storage — and a second call with a stride that "
        "reaches past the new size answers there and raises here. The first call "
        "agrees exactly; only the second parts."),
    "resize_as_sparse_": "sparse only, and refused here",
    "sparse_resize_": "sparse only, and refused here",
    "sparse_resize_and_clear_": "sparse only, and refused here",
}


def _names():
    """Public in-place methods both libraries have."""
    return sorted(
        n for n in dir(borch.Tensor)
        if n.endswith("_") and not n.startswith("_")
        and callable(getattr(borch.Tensor, n, None))
        and hasattr(torch.Tensor, n))


def _probed():
    return [n for n in _names()
            if n not in RANDOM and n not in NEEDS_MORE
            and n not in NOT_A_SECOND_CALL]


def test_the_binary_lists_name_things_that_exist():
    """`BINARY` and `BINARY_INT` are written by hand, so they can drift.

    A name in them that no longer exists is a row the probe skips **without saying
    so** — `_probed()` filters against `_names()`, so a stale entry simply never
    appears rather than failing.
    """
    stale = [n for n in (*BINARY, *BINARY_INT, *BUILT, *MATMUL)
             if n not in _names()]
    assert not stale, (
        f"listed with arguments but not an in-place method on both sides: {stale}")


def test_every_in_place_name_is_probed_or_written_down():
    """**Nothing falls off the list quietly.**

    A name absent from every table and from the probe is a name nothing asks about,
    and it does not report itself — which is the failure this file exists to catch,
    one level up in the file itself.
    """
    known = set(RANDOM) | set(NEEDS_MORE) | set(NOT_A_SECOND_CALL) | set(_probed())
    loose = [n for n in _names() if n not in known]
    assert not loose, (
        f"in-place names in neither the probe nor a table: {loose}\n"
        "  Give each one arguments, or add it to NEEDS_MORE with what it needs.")


def test_the_probe_still_reaches_most_of_what_it_can():
    """A floor, for the reason every measurement here has one.

    Set under what stands today so ordinary work does not trip it, and well over
    zero — a probe that reaches nothing reports no differences, which reads exactly
    like a library with no defects.
    """
    assert len(_probed()) > 120, (
        f"only {len(_probed())} in-place names are probed, and there were 142.\n"
        "  Check `_names()` first: a change to how in-place methods are detected\n"
        "  empties this file without failing any comparison.")


@pytest.mark.parametrize("name", _probed())
def test_the_second_call_agrees_with_torch(name):
    """Apply it twice on both sides and compare.

    **Twice, not once.** Once is a sample of size one from a sequence, and state is
    exactly what a sample of size one cannot see.
    """
    if name in MATMUL:
        base = MATMUL[name][0]
    elif name in BINARY_INT:
        base = BASE_I
    else:
        base = BASE
    other = OTHER_I if name in BINARY_INT else OTHER

    def twice(make):
        t = make(base.copy())
        # The arguments are built once and reused, so the two calls really are the
        # same call. A fresh operand each time would be a different question.
        if name in MATMUL:
            args = MATMUL[name][1](make)
        elif name in BUILT:
            args = BUILT[name](make)
        elif name in BINARY or name in BINARY_INT:
            args = (make(other.copy()),)
        else:
            args = ARGS.get(name, ())
        getattr(t, name)(*args)
        getattr(t, name)(*args)
        return t

    with np.errstate(all="ignore"):
        ours = np.asarray(twice(borch.tensor).data).astype(np.float64)
    theirs = twice(torch.tensor).detach().numpy().astype(np.float64)

    assert ours.shape == theirs.shape, (
        f"{name} called twice: shape {ours.shape} against torch's {theirs.shape}")
    assert np.allclose(ours, theirs, atol=1e-4, rtol=1e-4, equal_nan=True), (
        f"{name} called twice parted from torch\n"
        f"    ours  {ours.ravel()}\n"
        f"    torch {theirs.ravel()}\n"
        "  The first call may well agree — that is what makes this class invisible\n"
        "  to every other check here.")
