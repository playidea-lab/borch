"""**A case named after an argument must answer something the default would not.**

A case called `X(cooldown)` beside a case called `X` is claiming that `cooldown` changes
what happens. If the two answers are byte-identical, it is not testing the argument —
it is testing that the code runs, which the case beside it already did.

That sounds like a small thing and it is the mechanical half of the largest hole this
repository has found in its own instruments. Twice in one day a probe measured
everything it claimed to, on data where the answer **could not depend on what was being
tested**:

- `amsgrad` was checked with gradients of `base × (1+i)`. Those only grow, so the
  running maximum of the second moment is always the current one and the flag is a
  no-op **by construction of the input**. The comparison printed `0.000e+00` and could
  not have printed anything else.
- `GTSRB`'s train label is the folder's *position*; an earlier version used the number
  in the folder's *name*. On the real dataset those are the same number — forty-three
  folders, `00000` to `00042`, none missing — so **a complete input cannot tell the two
  rules apart.** The fixture caught it only because two folders were picked to be
  readable and happened not to be contiguous.

Neither had a filter to blame, a stale reason, or a mistake anybody made. The code
looked right, the test looked right, and the number was clean.

## What this can and cannot do

It reads the frozen answers, so it needs no browser, no torch and no run: two cases,
two answers, one comparison. It finds **arguments that provably did nothing**.

It does **not** find the `amsgrad` shape, where the inertness comes from the input
rather than from the argument — there is no pair to compare, because the case is alone.
That half stays a judgement, and the note at the end of `tests/browser/run.py` about
exact-zero agreements is where it is meant to become visible.

## The exemptions are attested, and they evict themselves

Twelve pairs answer identically today and **every one of them is right to**. Each is
named below with why. A thirteenth fails; and a row here that starts differing fails
too, because a reason for something that is no longer true is the defect this
repository has spent a day removing.
"""

import json
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
GOLDEN = ROOT / "tests" / "golden.json"

# Pairs where the argument names something the answer genuinely cannot depend on.
# **Checked one at a time**, not assumed from the name.
ATTESTED = {
    # **`out=` is `inplace` under another name**, and it earns its place here for the
    # same reason: a destination that changed the numbers would be the defect, so the
    # values case equalling the plain `erf` case is the claim. What `out=` *does*
    # change is where the answer landed, and `special::erf(out=)/같은 객체` beside
    # this row is the case a wrapper that computes and discards the destination fails.
    #
    # It matters here more than for `inplace`: `special`'s twenty-two names get their
    # `out=` from a loop over a written list in `borch/__init__.py`, so one wrapper
    # covers all of them and one mistake would cover all of them too.
    "special::erf(out=)":
        "`out=` moves the destination and not the value — torch computes the same "
        "answer and writes it where the caller asked. `special::erf(out=)/같은 객체` "
        "is where the move itself is asked, and a wrapper that computed `erf` and "
        "dropped the destination passes this row and fails that one",
    "act::nn.SELU(inplace)":
        "**`inplace` changes the identity, not the value.** In place the layer writes "
        "into the tensor it was handed and gives back the same object; the numbers are "
        "the ones it would have computed anyway. That is what the flag is for, and it "
        "is why `act::nn.SELU(inplace)/같은 객체` sits beside this row — a version "
        "computing the right values into a *new* tensor passes here and fails there. "
        "The other five activations escape this check only because their neighbour "
        "differs on some other argument, which is luck rather than a better case",
    # **Five more of the same shape, and they arrived together.** These classes
    # declared no constructor on the browser side, so the signature axis called them
    # *unreadable* rather than short — `SELU` above had the seat and they did not, and
    # nothing counted the difference. Each has a `/같은 객체` row beside it, which is
    # the half that can fail.
    **{f"act::nn.{_a}(inplace)":
       "`inplace` changes the identity and not the value — see `act::nn.SELU(inplace)` "
       "above, whose note is this one's. The `/같은 객체` row beside it is what a "
       "version computing into a new tensor fails"
       for _a in ("Hardsigmoid", "Hardswish", "Mish", "ReLU6", "SiLU")},
    "grad::sum(dim)":
        "the gradient of a sum is ones everywhere, whichever axis it was taken over",
    "grad::sort(내림차순)":
        "sorting permutes; summing the result gives every input a gradient of 1, and "
        "reversing a permutation does not change that. Measured: both are six 1.0s",
    "math::grad::trunc(0이어야)":
        "the name says the answer is zero. `trunc` is flat between integers, so its "
        "gradient is zero with an argument or without one — the case exists to say "
        "that out loud",
    "math::grad::fix(0이어야)": "as `trunc`, which `fix` is another name for",
    "ops::box_area(cxcywh)":
        "**a box's area does not depend on how the box is written down**, and the two "
        "answers agreeing is the point rather than a miss — it is what makes the "
        "conversion beside it worth trusting",
    "shape::expand(-1)":
        "`-1` means *leave this axis as it is*, so it asks for the same expansion the "
        "case beside it asks for by size. The two answering alike is the claim",
    "linalg::name2::eigvalsh(아래삼각만)":
        "the input is symmetric, so the lower triangle and the upper hold the same "
        "matrix and the eigenvalues are the same numbers",
    "act::nn.Identity(인자를 삼킨다)":
        "swallowing its arguments is what the name says and what torch does — a "
        "difference here would be the defect",
    "norm::grad::F.conv_transpose2d(편향)":
        "the gradient asked for is the input's, and adding a constant per channel does "
        "not change it. The bias has a gradient of its own; this case does not read it",
    "unpool::층::repr::CTCLoss(인자 있음)":
        "torch prints a loss layer with **no arguments** whatever it was built with — "
        "measured, and the same rule `HuberLoss` follows",
    "fname::제자리::hardtanh_(-1,1)":
        "`(-1, 1)` is `hardtanh`'s default, so the case is the default written out. It "
        "is here because the *positional* form is what a wrong parameter order breaks, "
        "and that is a different question from what the values are",
    "index::searchsorted(side=left)":
        "`left` is the default. The case pins that the argument is accepted and spelled "
        "the way torch spells it; `side=right` beside it is the one that moves",
}


def _cases():
    got = json.loads(GOLDEN.read_text(encoding="utf-8"))
    assert "cases" in got, f"{GOLDEN.name} has no `cases` — its shape changed"
    return got["cases"]


def _pairs(cases):
    """`(name, base, argument)` for every `X(...)` whose `X` is also a case."""
    out = []
    for name in cases:
        m = re.match(r"^(.*?)\(([^()]*)\)$", name)
        if not m:
            continue
        base = m.group(1)
        if base in cases:
            out.append((name, base, m.group(2)))
    return out


CASES = _cases()
PAIRS = _pairs(CASES)
IDENTICAL = [
    (name, base, arg) for name, base, arg in PAIRS
    if json.dumps(CASES[name], sort_keys=True)
    == json.dumps(CASES[base], sort_keys=True)
]


def test_there_are_pairs_to_compare_at_all():
    """**A floor, because an empty comparison passes.**

    The pairing is a regex over case names. Rename the convention — drop the
    parentheses, move to a `::arg` suffix — and this file finds nothing, reports
    nothing, and goes green while checking no cases at all.
    """
    assert len(PAIRS) > 150, (
        f"only {len(PAIRS)} `X(...)`/`X` pairs were found, and there were 224.\n"
        "  Check `_pairs()` before believing the result below: if the naming\n"
        "  convention moved, this file is measuring an empty set.")


def test_no_case_names_an_argument_that_changed_nothing():
    """Every identical pair must be one somebody has looked at and written down."""
    loose = sorted(name for name, _, _ in IDENTICAL if name not in ATTESTED)
    assert not loose, (
        "these cases name an argument and answer exactly what the case beside them "
        "answers:\n  "
        + "\n  ".join(loose)
        + "\n\n  Either the argument does nothing — in which case the case is testing "
          "that the code\n  runs, which its neighbour already did — or the value chosen "
          "for it happens to be\n  the default. Give it a value whose answer differs, or "
          "add it to `ATTESTED` with\n  why the two cannot differ."
    )


def test_no_attested_row_still_names_a_pair_that_differs():
    """**The eviction half.**

    A row here explains why two answers cannot differ. When they start differing the
    explanation is no longer true, and an exemption nobody re-reads is exactly the
    shape this repository keeps finding — six stale reasons in one day, each true when
    written.
    """
    identical = {name for name, _, _ in IDENTICAL}
    mended = sorted(name for name in ATTESTED
                    if name in CASES and name not in identical)
    assert not mended, (
        "these are listed as answering identically and no longer do:\n  "
        + "\n  ".join(mended)
        + "\n\n  Take them out. The row explains something that stopped being true."
    )


def test_no_attested_row_names_a_case_that_is_gone():
    """A name matching nothing exempts nothing, and looks like it is working."""
    missing = sorted(name for name in ATTESTED if name not in CASES)
    assert not missing, (
        f"`ATTESTED` names cases the golden does not have: {missing}\n"
        "  A row that matches nothing removes nothing and cannot fail, which is the\n"
        "  same silence this file exists to break.")


@pytest.mark.parametrize("name", sorted(ATTESTED))
def test_every_reason_says_something(name):
    """A row's value is the whole of its worth: it is what the next reader gets instead
    of re-deriving the argument."""
    reason = ATTESTED[name]
    assert len(reason) > 30, f"{name} is exempt with almost no reason: {reason!r}"
