"""**A scheduler's state has to survive a save and a restore.**

Every other check in this repository compares **one call from a fresh object.** The
golden runs a case once, both signature axes read a declaration, `parity.ts` weighs a
value. A difference that only appears the second time round is invisible to all of
them, and three have been found by hand this week:

- `embedding_bag(max_norm=…)` shortens the rows **in the table**, so a version that
  renormalised a copy agrees on the first call and parts on the second.
- `ReduceLROnPlateau`'s cooldown counter runs down *before* the patience is looked at.
  A counter that merely blocked the cut gives the same trajectory shifted by
  `cooldown` steps, which against a loss curve is indistinguishable.
- an optimizer's momentum buffer, weeks ago, which was right on the first step.

They are all one shape: **a fault in a sequence, and every instrument here samples a
sequence once.** Once is not a sample.

This file is the smallest general form of the missing instrument. It does not try to
detect sequence faults in general — it asks the question that causes them, of the
place they keep appearing: *is there any state that stepping changes and saving does
not carry.*

## Two questions, and the second is the one that catches things

`test_every_scheduler_can_be_saved` asks whether the class can be saved at all.
`test_stepping_changes_nothing_that_saving_leaves_behind` steps a scheduler, looks at
which attributes moved, and asks whether each is in `state_dict()`. An attribute that
moves and is not saved is a resume that quietly starts that part again — for a
cooldown, a cut up to `cooldown` steps early: small, plausible, and invisible against
a curve.

The comparison is against the attributes, not against a written-down list of keys. A
list would be one more fact that is correct on the day it is pasted, and the day a
scheduler grows a counter it would say nothing.
"""

import copy
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import borch as B                                               # noqa: E402

S = B.optim.lr_scheduler

# Built with the arguments each one needs, and stepped with a metric where it wants
# one. A scheduler this file cannot build is a scheduler it cannot ask about, so the
# list is checked for completeness below rather than trusted.
def _plateau(opt):
    return S.ReduceLROnPlateau(opt, patience=1, cooldown=2)


CASES = {
    "StepLR": (lambda o: S.StepLR(o, 2, 0.5), None),
    "ExponentialLR": (lambda o: S.ExponentialLR(o, 0.9), None),
    "MultiStepLR": (lambda o: S.MultiStepLR(o, [2, 4], 0.5), None),
    "CosineAnnealingLR": (lambda o: S.CosineAnnealingLR(o, 5), None),
    "LinearLR": (lambda o: S.LinearLR(o), None),
    "ConstantLR": (lambda o: S.ConstantLR(o), None),
    "PolynomialLR": (lambda o: S.PolynomialLR(o), None),
    "CyclicLR": (lambda o: S.CyclicLR(o, 0.01, 0.1), None),
    "LambdaLR": (lambda o: S.LambdaLR(o, lambda e: 0.9 ** e), None),
    "MultiplicativeLR": (lambda o: S.MultiplicativeLR(o, lambda e: 0.95), None),
    "CosineAnnealingWarmRestarts":
        (lambda o: S.CosineAnnealingWarmRestarts(o, 3), None),
    "ReduceLROnPlateau": (_plateau, [1.0] * 8),
    "OneCycleLR": (lambda o: S.OneCycleLR(o, 0.1, total_steps=10), None),
    "ChainedScheduler": (
        lambda o: S.ChainedScheduler([S.StepLR(o, 2, 0.5), S.ExponentialLR(o, 0.9)]),
        None),
    "SequentialLR": (
        lambda o: S.SequentialLR(
            o, [S.ConstantLR(o), S.ExponentialLR(o, 0.9)], milestones=[3]),
        None),
}

# Known to have no `state_dict` today. **Listed by name rather than counted**, so that
# a fourth fails here instead of joining them, and each carries what a resume loses.
NO_STATE_DICT = {
    "ReduceLROnPlateau":
        "seven mutable attributes — best, num_bad_epochs, cooldown_counter among them "
        "— and no way to save one of them. A resume restarts the plateau logic from "
        "nothing: the bad-epoch count goes to zero, so the next cut is up to "
        "`patience` steps late, and a resume inside a cooldown loses the cooldown "
        "entirely. torch's carries fifteen keys.",
    "ChainedScheduler": "holds the schedulers it chains, and none of their state.",
    "SequentialLR": "holds the milestone it has reached and does not save it.",
}


def _fresh():
    import numpy as np
    p = B.tensor(np.ones(2, dtype=np.float32), requires_grad=True)
    return B.optim.SGD([p], lr=0.1)


def _schedulers():
    return sorted(name for name in dir(S)
                  if name[0].isupper() and isinstance(getattr(S, name), type)
                  and name not in ("LRScheduler",))


def test_the_cases_cover_every_scheduler():
    """A scheduler this file cannot build is one it cannot ask about.

    **And it would be invisible.** The two tests below walk `CASES`, so a scheduler
    missing from it is not reported as unchecked — it simply does not appear, which is
    the absorbing bucket this repository keeps finding in other shapes.
    """
    uncovered = [name for name in _schedulers() if name not in CASES]
    assert not uncovered, (
        f"these schedulers are not built here, so nothing below asks about them: "
        f"{uncovered}\n  Add a way to construct each — a list that silently skips is "
        "the failure this file exists to prevent, one level up.")


def test_every_scheduler_can_be_saved():
    """`state_dict` at all. Absent, there is nothing for the next test to check."""
    absent = [name for name in _schedulers()
              if not hasattr(getattr(S, name), "state_dict")]
    surprise = [name for name in absent if name not in NO_STATE_DICT]
    assert not surprise, (
        f"these schedulers cannot be saved and are not written down as such: "
        f"{surprise}\n  Either give them a `state_dict` or add them to "
        "`NO_STATE_DICT` with what a resume loses.")
    mended = [name for name in NO_STATE_DICT if name not in absent]
    assert not mended, (
        f"`NO_STATE_DICT` lists schedulers that can be saved now: {mended}. Take them "
        "out — a row explaining something that is no longer true reads as a limit to "
        "the next person.")


@pytest.mark.parametrize("name", sorted(n for n in CASES if n not in NO_STATE_DICT))
def test_stepping_changes_nothing_that_saving_leaves_behind(name):
    """**Every attribute that stepping moves has to be in `state_dict`.**

    Asked of the attributes rather than of a written list of keys: a list would be one
    more fact correct on the day it was pasted, and the day a scheduler grows a
    counter it would say nothing at all.
    """
    make, metrics = CASES[name]
    opt = _fresh()
    sched = make(opt)
    before = {k: copy.deepcopy(v) for k, v in sched.__dict__.items()}
    for i in range(6):
        opt.step()
        sched.step(metrics[i]) if metrics else sched.step()

    moved = {k for k, v in sched.__dict__.items()
             if k != "optimizer" and repr(v) != repr(before.get(k, object()))}
    saved = set(sched.state_dict())
    lost = sorted(moved - saved)
    assert not lost, (
        f"{name}: stepping moves {lost} and `state_dict` does not carry it.\n"
        "  A resume starts that part again — for a counter, a decision taken early or "
        "late by\n  as many steps as it had counted. Small, plausible, and invisible "
        "against a loss curve,\n  because nothing here compares two runs across a save.")
