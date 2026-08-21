"""Where the gradient goes when several values **fold into one slot** — the remaining half.

**The table went into the golden cases** (`grad::접힘::*`, twenty-four of them). All three
implementations answer now, so it moved to the place that asks all three at once, and what
stays here is **only what the golden cases cannot ask.**

## The rule (what the golden twenty-four keep)

    an operation that **hands back an index** goes to the one slot it chose
        (max(dim), mode, nanmedian(dim), kthvalue)
    an operation that **hands back no index** spreads evenly over the equal-valued slots
        (max(), min(), median(), nanmedian())
    an operation that folds onto **sorted positions** goes to those positions
        (quantile — two of them when interpolating)

It is a question that only opens up when there are ties. With all values distinct the three
rules give the same answer, so an implementation that is half right passes — the golden
gradient cases were in that state for a long time.

## What it caught (history)

Seven places diverged, in two kinds. Five built a `Tensor(...)` bare-handed and so **had no
graph at all** (the value check passes, because the value is right), and two were connected
but had the rule wrong — `median` even carried the reasoning "shaking the other elements does
not move the answer", which is only true when there are no ties.

On the borch.ts side `i0` was **flowing a zero**, and its comment cited the core's hole as its
grounds. A gradient whose value is zero and a gradient that is absent are different
statements, and in the copying the second turned into the first.
"""

import numpy as np

import borch

EVEN = [1.0, 5.0, 5.0, 5.0]


def test_median_and_quantile_disagree_on_ties():
    """**Two operations that give the same value while standing on different things.** This
    one line explains the table.

    The median of [1,5,5,5] is 5 and so is its 0.5 quantile. But median gives ⅓ to **all three**
    slots holding 5, while quantile gives ½ to **the two positions** it used after sorting.
    Measured by value alone the two look the same, so this fork only shows up going backwards.

    Both cases are in the golden table, but **not side by side.** They are kept together inside
    one function here because the table shows only whether each is right on its own and not
    **that they differ from each other.**
    """
    def grad(fn):
        t = borch.tensor(EVEN, requires_grad=True)
        fn(t).backward()
        return [round(v, 4) for v in t.grad.tolist()]

    assert grad(lambda t: t.median()) == [0.0, 0.3333, 0.3333, 0.3333]
    assert grad(lambda t: t.quantile(0.5)) == [0.0, 0.5, 0.5, 0.0]


def test_a_folding_op_that_does_not_carry_grad_is_a_defect_not_a_choice():
    """A folding operation carries a gradient, **without exception.**

    A cut graph is not caught by a value check — because the value is right. The golden cases
    do not catch it either. What they ask is "is this gradient right"; what is asked here is
    **"is there a gradient at all".** If whoever adds a new reduction builds the result with
    `Tensor(...)`, this turns red naming it — that it catches them without a case being added
    alongside is the point.
    """
    tie = [3.0, 5.0, 5.0, 1.0, 5.0]
    nan = [3.0, float("nan"), 5.0, 1.0, 5.0]
    folding = [
        ("max()", tie, lambda t: t.max()),
        ("min()", tie, lambda t: t.min()),
        ("amax()", tie, lambda t: t.amax()),
        ("amin()", tie, lambda t: t.amin()),
        ("median()", tie, lambda t: t.median()),
        ("median(dim=0)", tie, lambda t: t.median(dim=0).values),
        ("nanmedian()", nan, lambda t: t.nanmedian()),
        ("nanmedian(dim=0)", nan, lambda t: t.nanmedian(0).values),
        ("mode()", [1.0, 1.0, 2.0, 2.0], lambda t: t.mode().values),
        ("kthvalue(2)", tie, lambda t: t.kthvalue(2).values),
        ("quantile(0.5)", tie, lambda t: t.quantile(0.5)),
        ("norm(inf)", tie, lambda t: t.norm(float("inf"))),
        ("norm(3)", tie, lambda t: t.norm(3)),
        ("angle()", [0.5, -1.0, 2.0], lambda t: t.angle()),
        ("i0()", [0.5, -1.0, 2.0], lambda t: t.i0()),
        ("topk(3)", tie, lambda t: t.topk(3).values),
        ("sort()", tie, lambda t: t.sort().values),
        ("cummax(0)", tie, lambda t: t.cummax(0).values),
    ]
    stuck = []
    for name, values, fn in folding:
        out = fn(borch.tensor(values, requires_grad=True))
        if not out.requires_grad:
            stuck.append(name)
    assert not stuck, (
        f"operations carrying no gradient: {stuck}\n"
        "A `Tensor(...)` built bare-handed has no parents attached, and the graph is cut quietly.\n"
        "Use `t._make(value, (t,), backward)`."
    )


def test_the_i1_series_is_convergence_not_approximation():
    """`i0`'s derivative is written as a series. **An approximation could not go into this repository.**

    Every term is positive so they do not cancel, and each is carried on by multiplying the
    previous one so no factorial overflows. The golden cases ask about three points only —
    measuring a wide interval is this file's job.
    """
    torch = __import__("torch")
    xs = np.linspace(-30, 30, 601)
    want = torch.special.i1(torch.tensor(xs, dtype=torch.float64)).numpy()
    got = borch._ops._i1(xs)
    rel = np.abs(want - got) / np.maximum(np.abs(want), 1e-300)
    assert rel.max() < 1e-12, f"maximum relative error {rel.max():.3e}"


def test_retain_grad_actually_retains():
    """**Getting only the refusal right is half of it.** What this name is for is keeping a
    derived tensor's `.grad`, and without that it is a shell that is right only as far as
    stopping at the leaves.

    It cannot go into the golden cases because borch.ts does not hand out a derived tensor's
    `.grad` — this is not a place that asks all three at once.
    """
    x = borch.tensor([1.0, 2.0], requires_grad=True)
    plain = x * 2
    plain.sum().backward()
    assert plain.grad is None, "it was kept without being asked for"

    y = borch.tensor([1.0, 2.0], requires_grad=True)
    kept = y * 3
    kept.retain_grad()
    kept.sum().backward()
    assert kept.grad is not None and kept.grad.tolist() == [1.0, 1.0]
    assert kept.retains_grad is True
    # **A leaf stays false even when asked** — it is not being kept, it accumulates anyway.
    assert y.retains_grad is False
