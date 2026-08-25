"""When both libraries refuse, `except <torch's class>` has to catch ours.

**A difference with no value attached.** Every other axis in this repository
compares values — the golden freezes them, parity compares them, the signature
checks compare argument lists. An exception's *class* is behaviour that none of
them can reach: no number differs, so no number can disagree. What differs is
which `except` clause catches, and `try: … except RuntimeError:` around a shape
that might not fit is ordinary code.

Measured at 30 refusals against real torch, **12 would have escaped a torch user's
`except`** the first time this was run. Ten were the wrong class. Two did not raise
at all:

    narrow(0, 1, 9) on a length-3    torch RuntimeError   ours → tensor([2., 3.])
    repeat(2) on a (1, 2)            torch RuntimeError   ours → tensor([[1., 2., 1., 2.]])

A Python slice stops at the end rather than complaining, and numpy's `tile` pads
the repeat count on the left. Both returned a plausible tensor of the wrong shape,
and there was no golden case for either — writing one needs somebody who already
suspected. **Asking about a missing exception is what found a wrong answer**, which
is the argument for this file over and above the classes it was written for.

The same thing happened once more, independently, in the protocol-surface sweep next
door: chasing `__index__`'s exception class turned up a version that **indexed by the
first element of a three-element tensor and said nothing.** Three wrong answers, all
plausible, none reachable by a value-comparing check, and all three found by someone
looking for the wrong exception rather than for a wrong number.

A class mismatch is at least *visible*: a user reading a traceback sees it. A missing
refusal is not visible anywhere, and this file finds them only because it asks the
question that happens to be next to them.

## Catchability, not equality

The question is not *do the two raise the same class* — it is *would `except
<torch's>` catch ours*. Five rows below differ by name and pass, because
`np.exceptions.AxisError` subclasses `IndexError` and torch raises `IndexError`.
Renaming those into agreement would be work with no reader.

`ZeroDivisionError`, which `split(0)` and `chunk(0)` used to raise, is the other
end: not a wrong choice of class but **no choice at all** — an arithmetic error
from inside a public API, naming nothing the caller did.
"""

import numpy as np
import pytest
import torch

import borch


def _t(lib, data, **kw):
    return lib.tensor(np.asarray(data, dtype=np.float32), **kw)


# Each row is a call **both** libraries are expected to refuse. Keep them small and
# obvious: the subject is which exception comes out, so anything else in the call is
# noise that can fail for its own reasons.
CASES = {
    "mm on mismatched shapes": lambda L: _t(L, [[1., 2.]]) @ _t(L, [[1., 2.]]),
    "add on mismatched shapes": lambda L: _t(L, [1., 2., 3.]) + _t(L, [1., 2.]),
    "reshape to the wrong size": lambda L: _t(L, [1., 2., 3.]).reshape(2, 2),
    "view to the wrong size": lambda L: _t(L, [1., 2., 3.]).view(2, 2),
    "index past the end": lambda L: _t(L, [1., 2.])[5],
    "sum over a dimension that is not there": lambda L: _t(L, [1., 2.]).sum(3),
    "squeeze a dimension that is not there": lambda L: _t(L, [[1., 2.]]).squeeze(9),
    "transpose a dimension that is not there":
        lambda L: _t(L, [[1., 2.]]).transpose(0, 9),
    "gather along a dimension that is not there":
        lambda L: _t(L, [[1., 2.]]).gather(
            5, L.tensor(np.zeros((1, 1), dtype=np.int64))),
    "softmax over a dimension that is not there": lambda L: _t(L, [1., 2.]).softmax(4),
    "unbind a dimension that is not there": lambda L: _t(L, [1., 2.]).unbind(3),
    "item on many": lambda L: _t(L, [1., 2.]).item(),
    "len of a scalar": lambda L: len(_t(L, 1.0)),
    "backward from a non-scalar":
        lambda L: _t(L, [1., 2.]).requires_grad_(True).backward(),
    "cat of nothing": lambda L: L.cat([]),
    "in-place on a leaf that wants gradients":
        lambda L: _t(L, [1., 2.], requires_grad=True).add_(1),
    "expand to a size that does not broadcast": lambda L: _t(L, [1., 2., 3.]).expand(2, 2),
    # The twelve below are the ones that parted. Kept as cases rather than as a
    # changelog: each is the call that would go uncaught if the guard came out.
    "stack of unequal shapes": lambda L: L.stack([_t(L, [1., 2.]), _t(L, [1.])]),
    "split into zero-sized pieces": lambda L: _t(L, [1., 2., 3.]).split(0),
    "chunk into zero pieces": lambda L: _t(L, [1., 2.]).chunk(0),
    "permute with too few dimensions": lambda L: _t(L, [[1., 2.]]).permute(0),
    "masked_select with a mask of another shape":
        lambda L: _t(L, [1., 2., 3.]).masked_select(
            L.tensor(np.array([True, False]))),
    "topk with k past the end": lambda L: _t(L, [1., 2.]).topk(5),
    "kthvalue with k past the end": lambda L: _t(L, [1., 2.]).kthvalue(9),
    "narrow past the end": lambda L: _t(L, [1., 2., 3.]).narrow(0, 1, 9),
    "repeat with fewer counts than dimensions": lambda L: _t(L, [[1., 2.]]).repeat(2),
    # **These three arrived from `OWED` by being fixed**, and they stay here rather
    # than leaving with the exemption. A row that agrees is exactly the row worth
    # keeping: it is what notices if the agreement stops.
    #
    # `bool` let *numpy's* `ValueError` escape where torch raises `RuntimeError`;
    # `int` and `float` raised this repository's `RuntimeError`, correct for the
    # `.item()` they delegated to, where torch raises `ValueError`. Crossed in
    # opposite directions, and no value differed in either — which is why this file
    # is the only thing that could ever have seen them.
    "bool on many": lambda L: bool(_t(L, [1., 2.])),
    "int on many": lambda L: int(_t(L, [1., 2.])),
    "float on many": lambda L: float(_t(L, [1., 2.])),
}

# Rows where the *conversion* protocols part.
#
# **Empty, and it held three for an hour.** `bool`, `int` and `float` were listed here
# while the protocol-surface sweep ran beside this file, and
# `test_no_exempt_row_still_disagrees` failed the moment they were fixed — telling the
# next reader to delete the rows rather than letting three permanent exemptions sit
# where nobody re-reads them. That is the whole design: **an exemption needs a check
# that evicts it, or it becomes a reason**, and a reason that outlives its cause was
# met five times in a single day before this file was written.
#
# The table stays. An empty one says *nothing is owed*; a deleted one says nothing at
# all, and the next protocol that parts needs somewhere to be written down.
OWED: dict[str, str] = {}

OWED_CASES: dict[str, object] = {}


def _raised(fn, lib):
    """The class `fn(lib)` raises, or `None` when it does not raise."""
    try:
        fn(lib)
    except Exception as e:                                    # noqa: BLE001
        return type(e)
    return None


def _catchable(theirs, ours):
    if theirs is None or ours is None:
        return theirs is ours
    return issubclass(ours, theirs)


@pytest.mark.parametrize("name", sorted(CASES))
def test_a_torch_except_clause_catches_our_refusal(name):
    fn = CASES[name]
    theirs, ours = _raised(fn, torch), _raised(fn, borch)
    assert theirs is not None, (
        f"'{name}': torch does not refuse this, so there is nothing to compare. "
        "Fix the case, do not delete it — a call both libraries accept belongs in "
        "the golden, not here.")
    assert ours is not None, (
        f"'{name}': torch raises {theirs.__name__} and this answers. "
        "A missing refusal is a wrong answer with nothing attached to it — the two "
        "found this way both returned a plausible tensor of the wrong shape.")
    assert _catchable(theirs, ours), (
        f"'{name}': torch raises {theirs.__name__}, we raise {ours.__name__}, and "
        f"`except {theirs.__name__}` does not catch it.\n"
        "  Nothing in this repository compares values here, because no value "
        "differs — only which `except` clause runs.")


@pytest.mark.parametrize("name", sorted(OWED))
def test_no_exempt_row_still_disagrees(name):
    """An exemption that has become false has to leave, not sit."""
    fn = OWED_CASES[name]
    theirs, ours = _raised(fn, torch), _raised(fn, borch)
    if _catchable(theirs, ours):
        pytest.fail(
            f"'{name}' agrees with torch now — both {theirs and theirs.__name__}. "
            f"Take it out of OWED.\n  The reason written there was: {OWED[name]}")


def test_an_exemption_cannot_exist_without_being_run():
    """**The exemption is written in two tables, and only one of them is iterated.**

    `test_no_exempt_row_still_disagrees` parametrises over `OWED` and reaches into
    `OWED_CASES` for the call. A name in `OWED` alone raises `KeyError`, which is loud
    and fine. **A name in `OWED_CASES` alone is never run at all** — no parameter is
    generated for it, so nothing reports it, and it does not even show up as the skip an
    empty table produces.

    That is the failure this whole file is built against, one level up: the eviction
    check is what stops an exemption becoming permanent, and an exemption it cannot see
    is exempt from eviction too. Not stale — **unseen**, which reads as nothing at all
    rather than as something out of date.

    Both tables are empty today, so this guards a hole that is latent. It is written now
    because the moment it stops being latent is the moment somebody is adding an
    exemption, and that is the worst moment to be relying on them to notice.
    """
    assert set(OWED) == set(OWED_CASES), (
        "the two exemption tables disagree.\n"
        f"  reason but no call: {sorted(set(OWED) - set(OWED_CASES))}\n"
        f"  call but no reason: {sorted(set(OWED_CASES) - set(OWED))}\n"
        "  A row needs both. With only a reason the eviction check raises KeyError;\n"
        "  with only a call it is never parametrised, so it is exempt from the check\n"
        "  that exists to take exemptions away.")


def test_every_case_asks_torch_something_it_refuses():
    """**Two failures to raise compare equal.** A case that neither library refuses
    would pass this file while measuring nothing at all — the same shape as two
    `AttributeError`s agreeing about a method that does not exist.

    So the count of rows where torch actually raised is pinned. It is the floor that
    a badly-written case falls through.
    """
    refused = [n for n, fn in CASES.items() if _raised(fn, torch) is not None]
    assert len(refused) == len(CASES), (
        "torch accepts these, so they measure nothing: "
        f"{sorted(set(CASES) - set(refused))}")
    assert len(CASES) >= 25, f"only {len(CASES)} refusals are asked about"
