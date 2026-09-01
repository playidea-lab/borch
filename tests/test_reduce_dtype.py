"""How a reduction changes the type — **whether the core answers as torch does, in the same places.**

Only one thing was known, that `bool.sum()` is int64, and from it only that a third rule
exists which is neither the "preserve" used by shape and indexing operations nor the
"promote" used by value operations. Asking torch and pulling out a table showed **four**
rules, and the line dividing them was a familiar one:

    accumulation **makes** a value (a 3 does not fit in a true/false slot),
    selection **hands over** a value that was already there.

It is the same line drawn when the dtype labels on shape and indexing operations were
fixed. Reductions were not an exception; **accumulation and selection are different
things**, and `mean` being a refusal falls out of the same place — division has no answer
inside an integer slot.

## Why here rather than in the golden cases

The golden cases ask all three implementations at once, and borch.ts's reductions all call
`Tensor.make` without a dtype and produce float32 today. Adding the cases alone leaves the
binding's runner red. **The core's half can be settled now**, so that half is pinned here,
and the same table moves into the golden cases when borch.ts is fixed.

## What this check caught

The core diverged from torch in six places. Not on the dtype rules but on **whether it
refuses** — at `mean`, `median(bool)` and `argmax(bool)`, torch stops while the core rode
numpy and produced a value, and `logsumexp` leaked numpy's default float64. "Approximating
where torch stops" is the kind of thing this repository's first line refuses.
"""

import numpy as np
import pytest
import torch

import borch

# Everything worth calling a reduction. It looks at **the type that comes out**, not the value.
OPS = [
    ("sum", lambda t: t.sum()),
    ("prod", lambda t: t.prod()),
    ("cumsum", lambda t: t.cumsum(0)),
    ("cumprod", lambda t: t.cumprod(0)),
    ("amax", lambda t: t.amax()),
    ("amin", lambda t: t.amin()),
    ("max", lambda t: t.max()),
    ("min", lambda t: t.min()),
    ("mean", lambda t: t.mean()),
    ("median", lambda t: t.median()),
    ("argmax", lambda t: t.argmax()),
    ("argmin", lambda t: t.argmin()),
    ("count_nonzero", lambda t: t.count_nonzero()),
    ("any", lambda t: t.any()),
    ("all", lambda t: t.all()),
    ("logsumexp", lambda t: t.logsumexp(0)),
    ("var", lambda t: t.var()),
    ("std", lambda t: t.std()),
    ("norm", lambda t: t.norm()),
    # ── the ones that were outside the table ─────────────────────────────
    #
    # The nineteen frozen first were not enough. **A place not asked about does not match** —
    # measuring the fourteen below found ten diverging, in the same two kinds as above (torch
    # stops and we approximate / we stop and torch answers), plus numpy's float64 leaking.
    ("nansum", lambda t: t.nansum()),
    ("nanmean", lambda t: t.nanmean()),
    ("nanmedian", lambda t: t.nanmedian()),
    ("logcumsumexp", lambda t: t.logcumsumexp(0)),
    ("quantile", lambda t: t.quantile(0.5)),
    ("bincount", lambda t: t.bincount()),
    ("diff", lambda t: t.diff()),
    # The ones giving values and indices together. **Their two types move independently** —
    # `bool.sort()` is `bool + int64`.
    ("cummax", lambda t: t.cummax(0)),
    ("cummin", lambda t: t.cummin(0)),
    ("aminmax", lambda t: t.aminmax()),
    ("mode", lambda t: t.mode()),
    ("sort", lambda t: t.sort()),
    ("topk", lambda t: t.topk(2)),
    ("median(dim=0)", lambda t: t.median(dim=0)),
]

# **Both have to be asked.** Asking int64 alone makes "preserves the type" and "promotes
# bool" look identical, and an implementation that is half right passes.
KINDS = [
    ("int64", [3, 1, 4, 1, 5], True),
    ("bool", [True, False, True], False),
]


def name_of(out):
    """The type's name. Where values and indices come together, **both** are written — looking
    at one hides the divergence.

    torch gives a named tuple such as `return_types.sort` and the core gives its own wrapper.
    Both carry `.values` and `.indices`, but **an ordinary tensor carries `.values` too** (for
    sparsity) — separating on that reads every tensor as a tuple. Measured that way once,
    twenty-five places appeared to diverge.

    **The middle branch used to be a list of our class names**, and it broke the day one of
    them was renamed: `aminmax` moved from `_MinMax` to a class named after the function, and
    two rows here started reporting `aminmax` where a dtype belonged. A check that knows our
    internal class names goes stale every time one moves, and it goes stale *silently* — the
    name simply stops matching and the fallback answers something plausible.

    So the question asked is about the object rather than its name: a tensor is the thing with
    a `dtype`, and what is left over and can be walked is a bundle. That is the same
    discrimination as before — a tensor is iterable and so cannot be told apart *by*
    iterability — but taken in the order that makes it decidable.
    """
    if isinstance(out, tuple) or hasattr(out, "_fields"):
        return " + ".join(name_of(x) for x in tuple(out))
    if hasattr(out, "dtype"):
        return str(out.dtype).replace("torch.", "")
    try:
        parts = list(out)
    except TypeError:
        return type(out).__name__
    return " + ".join(name_of(x) for x in parts)


def answer(lib, values, as_int, fn):
    """The type's name, or `"refused"` where it refused. **A refusal is an answer too.**"""
    try:
        t = lib.tensor(values, dtype=lib.int64) if as_int else lib.tensor(values)
        out = fn(t)
    except Exception:                                           # noqa: BLE001
        return "refused"
    return name_of(out)


@pytest.mark.parametrize("op,fn", OPS, ids=[n for n, _ in OPS])
@pytest.mark.parametrize("kind,values,as_int", KINDS, ids=[k for k, _, _ in KINDS])
def test_reduction_dtype_matches_torch(op, fn, kind, values, as_int):
    expected = answer(torch, values, as_int, fn)
    got = answer(borch, values, as_int, fn)
    assert got == expected, (
        f"{op}({kind}): torch gives {expected} and the core gives {got}.\n"
        "There are four type rules for reductions — accumulation (sum, prod, cumsum,\n"
        "cumprod) promotes bool to int64; selection (amax, amin, max, min) preserves the\n"
        "type; the argmax and count_nonzero family is fixed at int64; and mean, var, std\n"
        "and norm refuse integers and booleans."
    )


def test_dtype_argument_beats_the_rule():
    """Given `dtype=`, it beats everything — it sits above the rules.

    **This place was marked `xfail(strict=True)`.** Filling it in turned that mark red as an
    "unexpectedly passing" test and told us to fix here too — it asked exactly as designed.

    The rule is one line: **convert before folding**, not after. Measurement separates the
    two — `[1.7, −2.3, 0.9].sum(dtype=int64)` is `−1`. Folded first it is `0.3`, and
    truncated still `0`; truncated first it is `[1, −2, 0]`, and the sum is `−1`.
    """
    for values in ([3, 1, 4], [True, False, True]):
        a = torch.tensor(values).sum(dtype=torch.float32).dtype
        b = borch.tensor(values).sum(dtype=borch.float32).dtype
        assert str(a) == str(b) == "torch.float32"
    # Pins **that it truncates first** by value. Asking about the type alone cannot separate the two orders.
    reals = [1.7, -2.3, 0.9]
    assert torch.tensor(reals).sum(dtype=torch.int64).item() == -1
    assert borch.tensor(reals).sum(dtype=borch.int64).item() == -1


def test_dtype_argument_keeps_the_two_refusals_torch_keeps():
    """`dtype=` does not lift **every** refusal — two stay (measured).

    For `mean`, only the refusal about the input lifts. `dtype=float32` on an integer input
    runs, and **a mean whose result is an integer** still has no answer. For `cumsum` and
    `cumprod`, torch simply never built `dtype=bool` — even though `sum(dtype=bool)` works.

    Diverging towards leniency is still diverging. Handing back a value here makes that code
    break under real torch.
    """
    assert borch.tensor([3, 1, 4]).mean(dtype=borch.float32).item() == pytest.approx(
        torch.tensor([3, 1, 4]).mean(dtype=torch.float32).item())
    with pytest.raises(RuntimeError, match="could not infer output dtype"):
        borch.tensor([1.5, 2.5]).mean(dtype=borch.int64)
    with pytest.raises(NotImplementedError):
        borch.tensor([1, 2, 3]).cumsum(0, dtype=borch.bool_)
    with pytest.raises(NotImplementedError):
        borch.tensor([1, 2, 3]).cumprod(0, dtype=borch.bool_)


def test_to_actually_changes_the_dtype():
    """`x.to(torch.float32)` **changes the type.**

    For a long time it did not — `to` looked only at the device string and quietly discarded
    the rest. With no exception and no warning it stayed the original type, and on an integer
    tensor the division after it diverged into **integer division.** It surfaced while adding
    `dtype=` to the reductions — that side calls this function and the type was not
    changing.
    """
    ints = borch.tensor([3, 1, 4], dtype=borch.int64)
    assert str(ints.to(borch.float32).dtype) == "torch.float32"
    assert str(ints.to(borch.int64).dtype) == "torch.int64"
    # The device side is unchanged — 'cpu' passes and any other device stops.
    assert str(ints.to("cpu").dtype) == "torch.int64"


def test_the_line_is_accumulate_versus_select():
    """Pins the rules as sentences — if the table wobbles, this breaks first.

    Accumulation makes a value and leaves the true/false slot; selection hands over a value
    that was there and does not.
    Those two lines explain all the rest.
    """
    flags = borch.tensor([True, False, True])
    assert str(flags.sum().dtype) == "torch.int64", "accumulation promotes bool"
    assert str(flags.amax().dtype) == "torch.bool", "selection preserves the type"
    # The same divergence is invisible on integers — both are int64. It takes bool to separate them.
    ints = borch.tensor([3, 1, 4], dtype=borch.int64)
    assert str(ints.sum().dtype) == str(ints.amax().dtype) == "torch.int64"


def test_mean_refuses_integers_like_torch():
    """A mean is a division and has no answer inside an integer slot. numpy quietly promotes to float64."""
    with pytest.raises(RuntimeError, match="could not infer output dtype"):
        borch.tensor([1, 2, 3], dtype=borch.int64).mean()
    with pytest.raises(RuntimeError, match="could not infer output dtype"):
        borch.tensor([True, False]).mean()
    # Converting to float works — this is not a missing feature but a type that does not fit.
    assert borch.tensor([1, 2, 3], dtype=borch.int64).float().mean().item() == 2.0


def test_logsumexp_answers_float32_not_float64():
    """Where numpy's default leaked. Booleans refused `-` and stopped outright."""
    for t in (borch.tensor([1, 2, 3], dtype=borch.int64), borch.tensor([True, False])):
        got = t.logsumexp(0)
        assert str(got.dtype) == "torch.float32", str(got.dtype)
    ref = torch.tensor([1, 2, 3]).logsumexp(0).item()
    assert np.isclose(borch.tensor([1, 2, 3], dtype=borch.int64).logsumexp(0).item(), ref)
