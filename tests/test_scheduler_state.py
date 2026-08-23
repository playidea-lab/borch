"""**A scheduler's state has to survive a save and a restore.**

Every other check in this repository compares **one call from a fresh object.** The
golden runs a case once, both signature axes read a declaration, `parity.ts` weighs a
value. A difference that only appears the second time round is invisible to all of
them, and three have been found by hand this week:

- `ReduceLROnPlateau`'s cooldown counter runs down *before* the patience is looked at.
  A counter that merely blocked the cut gives the same trajectory shifted by
  `cooldown` steps, which against a loss curve is indistinguishable.
- an optimizer's momentum buffer, weeks ago, which was right on the first step.
- the binding calling borch.ts with an argument order that had moved on one side —
  repetition in space rather than in time, and no compiler reaches across it.

**A fourth belongs beside them and does not belong in the list**, which is worth the
line it costs. `embedding_bag(max_norm=…)` shortens rows in the table, and a version
that renormalised a copy **never parts on the output at all** — not on the second
call, not on the hundredth, because renormalising an already-short row is a no-op.
It parts on the *state*, immediately. Repeating the call is the wrong prescription
there; looking at `weight` instead of the return value is the right one. Two of us
had it written down the other way for an hour, which is the shape of a reason that
reads correctly and points somewhere useless.

So the shape is not "once is not a sample". It is **the observable we compare is not
the whole state** — sometimes repetition reveals the difference, and sometimes only
looking somewhere else does.

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

# Schedulers that cannot survive a save. **Listed by name rather than counted**, so a
# fourth fails here instead of quietly joining them, and each row carries what a resume
# loses.
#
# **Empty, and kept rather than deleted.** An empty table says *nothing is owed*; a
# deleted one says nothing at all, and the next scheduler without a save needs somewhere
# to be written down.
#
# It held three for an hour: `ReduceLROnPlateau` with seven mutable attributes and
# nowhere to put them, and `ChainedScheduler` and `SequentialLR` holding what they wrap
# without its state. All three were mended in one change, by a `_Saves` base whose
# `state_dict` takes everything but the optimizer — which would make a cycle — and
# recurses into nested schedulers, so the chained pair now carries dictionaries rather
# than the objects themselves and the whole thing is JSON-serialisable. Two golden cases
# hold `ReduceLROnPlateau`'s resume, and the second of the two exists because the first
# passes even when loading does nothing.
NO_STATE_DICT: dict[str, str] = {}


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


def _resumes(name):
    """Save after four steps, restore into a fresh pair, and see whether the two agree.

    **The optimizer is restored with the scheduler**, which is torch's documented order
    and not a detail. The learning rate lives on the optimizer; a scheduler carries the
    counters that decide how it moves. Restore the scheduler alone into a fresh
    optimizer and the rate starts at its initial value, so `StepLR`, `ExponentialLR`,
    `MultiStepLR` and `MultiplicativeLR` all "fail" for a reason that is not theirs.

    Written that way first, this reported **six broken schedulers and every one was
    false** — the naive round trip is itself the wrong question, in the same way
    `hasattr` was. One asks too little of the object and the other asks it of the wrong
    object.
    """
    make, metrics = CASES[name]
    first, second = _fresh(), _fresh()
    live, restored = make(first), make(second)
    for i in range(4):
        first.step()
        live.step(metrics[i]) if metrics else live.step()
    saved_opt = copy.deepcopy(first.state_dict())
    saved = copy.deepcopy(live.state_dict())
    second.load_state_dict(saved_opt)
    restored.load_state_dict(saved)
    for i in range(4, 6):
        first.step()
        live.step(metrics[i]) if metrics else live.step()
        second.step()
        restored.step(metrics[i]) if metrics else restored.step()
    return first.param_groups[0]["lr"], second.param_groups[0]["lr"]


@pytest.mark.parametrize("name", sorted(CASES))
def test_a_saved_scheduler_resumes_where_it_left_off(name):
    """**Ask the object to do the thing rather than asking whether it could.**

    This asked `hasattr(cls, "state_dict")`, which is a proxy for *can this be saved*.
    The day a base class arrived carrying `state_dict` for everything, the proxy went
    true for every child and stopped having anything to do with the ability. It then
    reported, confidently, that three schedulers written down as unsaveable could be
    saved — and two of those rows were still true at the time.

    A proxy keeps answering after it stops being connected to the question, which is
    worse than falling silent: this repository has now paid for that three times —
    `run.py` comparing modification times until it compared the name table instead,
    `lessons.py` waiting on a button's text until it read `disabled`, and this.

    So the ability is exercised: four steps, save both, restore into a fresh pair, and
    the two have to arrive at the same rate.
    """
    if name in NO_STATE_DICT:
        pytest.skip(NO_STATE_DICT[name])
    live, restored = _resumes(name)
    assert live == pytest.approx(restored, abs=1e-12), (
        f"{name}: the run that continued reached {live} and the one restored from a "
        f"checkpoint reached {restored}.\n  Something the stepping moves is not in "
        "`state_dict`, so the resume starts that part again.")


def test_no_unsaveable_row_describes_something_that_now_resumes():
    """A row outliving its defect reads to the next person as a limit that is still
    there. When one is mended the row goes with the fix."""
    mended = []
    for name in NO_STATE_DICT:
        try:
            live, restored = _resumes(name)
        except Exception:                                       # noqa: BLE001
            continue
        if live == pytest.approx(restored, abs=1e-12):
            mended.append(name)
    assert not mended, (
        f"`NO_STATE_DICT` lists schedulers that resume correctly now: {mended}. Take "
        "them out — the defect is fixed and the row now describes nothing.")


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
