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

## What it does not cover, said out loud

**84 of the 162 in-place names are not probed**, because they need arguments this file
does not supply — `scatter_`, `index_put_`, `masked_fill_` and the rest of the family
that mutates hardest. That is not a coverage figure with a gap in it; it is *the half
that needs no arguments*, and the unreached half is the more dangerous one.

They are listed by name below rather than counted, so a name that silently stops being
probed shows up as a row that no longer matches instead of a number that stayed the
same. Widening this file means giving those arguments, one family at a time.
"""

import sys

import numpy as np
import pytest

pytest.importorskip("torch")
import torch                                                     # noqa: E402

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

import borch                                                     # noqa: E402

BASE = np.array([[1.5, -2.0, 3.0], [0.5, 4.0, -1.0]], dtype=np.float32)

# Arguments chosen so the operation changes something on **both** calls. One whose
# second call is a no-op cannot show anything either way, and a case that cannot fail
# is the shape this repository keeps finding.
ARGS = {
    "add_": (2.0,), "sub_": (1.0,), "mul_": (2.0,), "div_": (2.0,),
    "pow_": (2.0,), "clamp_": (-1.0, 2.0), "fill_": (3.0,),
    "clamp_min_": (0.0,), "clamp_max_": (1.0,), "renorm_": (2, 0, 1.0),
    "requires_grad_": (False,),
}

# **Drawn from a generator, so two runs differ by design.** Not a finding, and not
# skipped silently either: a name that moves from here to the compared set because
# somebody made it deterministic should be noticed.
RANDOM = {
    "bernoulli_", "cauchy_", "exponential_", "geometric_", "log_normal_",
    "normal_", "random_", "uniform_",
}

# Not probed, with what each one needs. **Listed rather than counted** — see the
# module docstring. Every row here is a place the second call has never been asked.
NEEDS_MORE = {
    "addbmm_": "a batch of matrices", "addmm_": "two matrices",
    "addmv_": "a matrix and a vector", "addr_": "two vectors",
    "addcdiv_": "two tensors", "addcmul_": "two tensors",
    "baddbmm_": "two batches", "apply_": "a Python callable",
    "map_": "a Python callable", "map2_": "two tensors and a callable",
    "copy_": "a source tensor", "masked_fill_": "a mask",
    "masked_scatter_": "a mask and a source", "index_fill_": "indices",
    "index_copy_": "indices and a source", "index_add_": "indices and a source",
    "index_put_": "indices and values", "index_reduce_": "indices and a reduction",
    "scatter_": "indices and a source", "scatter_add_": "indices and a source",
    "scatter_reduce_": "indices and a reduction", "put_": "indices and values",
    "set_": "a storage", "resize_as_": "another tensor",
    "as_strided_": "a size and a stride", "fill_diagonal_": "a value",
    "cumprod_": "a dimension", "cumsum_": "a dimension",
    "mvlgamma_": "the dimension p", "polygamma_": "the order n",
    "lerp_": "an end and a weight", "swapaxes_": "two axes",
    "swapdims_": "two axes",
    # Binary and comparison forms — each needs the other operand.
    **{n: "the other operand" for n in (
        "arctan2_", "atan2_", "bitwise_and_", "bitwise_left_shift_", "bitwise_or_",
        "bitwise_right_shift_", "bitwise_xor_", "copysign_", "divide_", "eq_",
        "float_power_", "floor_divide_", "fmod_", "gcd_", "ge_", "greater_",
        "greater_equal_", "gt_", "heaviside_", "hypot_", "igamma_", "igammac_",
        "lcm_", "ldexp_", "le_", "less_", "less_equal_", "logical_and_",
        "logical_or_", "logical_xor_", "lt_", "multiply_", "ne_", "nextafter_",
        "not_equal_", "remainder_", "subtract_", "true_divide_", "xlogy_")},
}

# Probed by nothing here for a reason that is not "needs arguments".
NOT_A_SECOND_CALL = {
    "resize_": "changes the shape; a different question",
    "transpose_": "the second call undoes the first",
    "t_": "the second call undoes the first",
    "squeeze_": "the second call is a no-op",
    "unsqueeze_": "changes rank each time; a different question",
    "clip_": "torch refuses it with neither bound given",
    "bitwise_not_": "torch refuses it on a float tensor",
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
    assert len(_probed()) > 60, (
        f"only {len(_probed())} in-place names are probed, and there were 78.\n"
        "  Check `_names()` first: a change to how in-place methods are detected\n"
        "  empties this file without failing any comparison.")


@pytest.mark.parametrize("name", _probed())
def test_the_second_call_agrees_with_torch(name):
    """Apply it twice on both sides and compare.

    **Twice, not once.** Once is a sample of size one from a sequence, and state is
    exactly what a sample of size one cannot see.
    """
    args = ARGS.get(name, ())

    def twice(make):
        t = make()
        getattr(t, name)(*args)
        getattr(t, name)(*args)
        return t

    with np.errstate(all="ignore"):
        ours = np.asarray(twice(lambda: borch.tensor(BASE.copy())).data)
    theirs = twice(lambda: torch.tensor(BASE.copy())).detach().numpy()

    assert ours.shape == theirs.shape, (
        f"{name} called twice: shape {ours.shape} against torch's {theirs.shape}")
    assert np.allclose(ours, theirs, atol=1e-4, rtol=1e-4, equal_nan=True), (
        f"{name} called twice parted from torch\n"
        f"    ours  {ours.ravel()}\n"
        f"    torch {theirs.ravel()}\n"
        "  The first call may well agree — that is what makes this class invisible\n"
        "  to every other check here.")
