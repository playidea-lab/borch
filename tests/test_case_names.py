"""Catches **cases that hang an argument on their name and never ask about it.**

There was a case called `grad::sum(dim)`. Named that way, nobody looked at it again — and
the gradients of `sum(dim=1).sum()` and `sum().sum()` are **both all ones**, so ignoring the
axis entirely still passes. Meanwhile `borch_webgpu` was returning a scalar for
`sum(dim=1)` with the axis ignored, and all 792 cases were green.

The same shape appeared again at `unpool::분수::output_ratio` — the name says ratio while
the body wrote the size out by hand, so the ratio-to-size rule was not being asked about
there. **By its name it looks asked.**

## How it asks — by value

Counting whether the word appears in the body does not work. Passed positionally the word is
invisible, and thirty-five places came up as false positives. Instead **the frozen golden
answers are compared against each other**: if the pair with the parentheses stripped off the
name exists in the table and **the answers do not differ by a single byte**, that case does
not ask about what is inside the parentheses.

Nothing has to run. It reads the table only, and leans on none of the three
implementations.

## What it caught

Two, and **both were places where the input made the two arguments equal.**

`nn.Upsample(8)` had a 4×4 input, so its answer matched the default `scale_factor=2` — the
name says "the first position is size" and **an implementation reading the first position as
a scale factor passed too.** Asked with 6 they diverge (an integer scale cannot produce 6).

`FractionalMaxPool2d(output_ratio=0.5)` takes 7×0.5 to 3.5 and so to a size of 3, and its
paired size case happened to be `output_size=(3,3)`. An implementation ignoring the ratio and
using the size default passes. The size side moved to 4.
"""

import json
import pathlib
import re

CASES = json.loads(
    (pathlib.Path(__file__).parent / "golden.json").read_text())["cases"]

# Strips the last parenthesised group — `foo(bar)`'s pair is `foo`.
_TAIL = re.compile(r"\s*[（(][^()（）]*[)）]\s*$")

# **Deliberately equal.** Cases where being equal is the answer, and cases where what is in
# the parentheses is the default. A new one turns this check red, so adding one means writing
# down why, here.
DELIBERATE = {
    # being equal is the answer
    "act::nn.Identity(인자를 삼킨다)":
        "torch's Identity swallows any argument — it is a placeholder, so users change the "
        "layer's name and leave the arguments as they are.",
    "unpool::층::repr::CTCLoss(인자 있음)":
        "torch's repr does not print the arguments. Freezing it as though it did diverges.",
    "math::grad::trunc(0이어야)": "truncation's derivative is zero everywhere.",
    "math::grad::fix(0이어야)": "the same function as `trunc`.",
    # what is in the parentheses is the default — pinning that giving it by hand changes nothing
    "index::searchsorted(side=left)": "`left` is the default.",
    "fname::제자리::hardtanh_(-1,1)": "(-1, 1) is the default.",
    "linalg::name2::eigvalsh(아래삼각만)": "`UPLO='L'` is the default.",
    "shape::expand(-1)": "-1 means leave that axis alone, so the answer is the same.",
    # places a gradient cannot separate — **a value case exists separately**
    "grad::sum(dim)":
        "a sum's gradient is all ones regardless of the axis. `arg::sum(dim)` asks about the axis by value.",
    "grad::sort(내림차순)":
        "a sort's gradient is all ones regardless of order. A value case asks about the order.",
    "norm::grad::F.conv_transpose2d(편향)":
        "the gradient towards the input is independent of the bias. A value case asks about the bias.",
}


def identical_pairs():
    """Cases whose parenthesis-stripped pair exists and **whose answers are equal.**"""
    same = []
    for name, value in CASES.items():
        base = _TAIL.sub("", name).strip()
        if base == name or base not in CASES:
            continue
        if json.dumps(value, sort_keys=True) == json.dumps(CASES[base], sort_keys=True):
            same.append(name)
    return same


def test_a_case_named_after_an_argument_asks_about_it():
    """An argument hung in parentheses has to produce **an answer different from the default's.**

    Where it does not, that case does not ask what its name says — and because of the name,
    nobody looks again. Deliberately equal places are written above with their reasons.
    """
    surprise = [n for n in identical_pairs() if n not in DELIBERATE]
    assert not surprise, (
        "the name hangs an argument and the answer equals the default case's:\n  "
        + "\n  ".join(surprise)
        + "\n\nChange the input so the argument changes the answer, or if being equal is the "
          "answer, write it into `DELIBERATE` with a reason."
    )


# **A gradient that is entirely zero asks nothing** — an implementation that flows no
# gradient at all passes too. Below are the places whose derivative really is zero; anything
# else turning zero is usually **the case's wiring** being wrong.
#
# It was caught exactly that way. `edge::`'s folding helper multiplied by `arange` to give a
# different weight per position, and the first term is zero, so **any case whose output is a
# single slot had an entirely zero gradient.** A device added to avoid uniform folding had
# left that case asking nothing — `max(동점)` was freezing `[0,0,0,0]` instead of
# `[0,1,0,0]`.
ZERO_ON_PURPOSE = {
    "grad::접힘::angle() 은 0 을 흘린다": "a real's argument is a staircase, so its derivative is zero.",
    "math::grad::trunc": "truncation is a staircase.",
    "math::grad::fix": "the same function as `trunc`.",
    "math::grad::copysign/b": "no gradient goes towards the side giving the sign.",
    "blend::grad::addmm(beta=0)": "at `beta=0` the added term drops out.",
    "edge::grad::sign(0포함)": "the sign function's derivative is zero everywhere.",
}


def test_no_gradient_case_is_all_zero_by_accident():
    """A gradient that is zero and **a gradient that does not flow** say different things.

    A name carrying `(0이어야)` has written the answer down itself, so it passes.
    """
    zero = []
    for name, case in CASES.items():
        if "grad::" not in name or "(0이어야)" in name:
            continue
        vals = [v for v in (case.get("values") or []) if isinstance(v, (int, float))]
        if len(vals) >= 2 and set(vals) == {0.0} and name not in ZERO_ON_PURPOSE:
            zero.append(name)
    assert not zero, (
        "cases whose gradient is entirely zero: " + ", ".join(zero) + "\n"
        "If the derivative really is zero, write it into `ZERO_ON_PURPOSE` with a reason; "
        "otherwise look at the case's wiring — a zero mixed into the folding weights does this."
    )


def test_the_deliberate_list_does_not_rot():
    """**Whether what was written down is actually equal, too.**

    Editing a case so the answers diverge makes that row above false. A list left
    is what stops it growing for no reason — `KNOWN_ABSENT` does the same job in this
    repository.
    """
    same = set(identical_pairs())
    stale = sorted(n for n in DELIBERATE if n not in same)
    assert not stale, (
        f"the answers diverge now and it is still in `DELIBERATE`: {stale}\n"
        "Delete that row — a list that grows long stops being read."
    )
