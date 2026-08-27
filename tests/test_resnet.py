"""Trains the same small ResNet on both sides and compares the whole run.

`test_scenario.py` next door compares what came out at the end. This compares **when**,
which is a different question: a divergence that starts small and compounds reaches the
last step looking like any other wrong number, and there is nothing in the value to say
whether it began in the backward pass or in the thirtieth optimizer step.

So the loss of every step is compared, and the gradient of every parameter **before a
single weight has moved**, and the running statistics BatchNorm wrote on the way.

## What the measured shape says

Run on this machine the four groups come out ordered, and the order is the point:

    grad     1.8e-08     before any weight moved — the backward pass alone
    loss     1.2e-07     one float32 eps, after thirty steps of accumulation
    buffer   4.9e-07     written once per step, so thirty roundings deep
    eval     1.6e-06     reads those buffers, and inherits their drift

Error appearing **only where accumulation happens** is what agreement looks like. A
structural break does not look like this: a residual join that overwrote instead of
accumulating, a momentum applied to the wrong side of weight decay, or a running
variance taken from the biased estimate all move the numbers by percent, not by eps.

The tolerance is the sibling's and for the sibling's reason — bit equality is an explicit
non-goal, and macOS's BLAS and Linux's have already diverged at the sixth decimal and
turned CI red on their own.
"""

import pathlib
import subprocess
import sys


def _run(lib):
    out = subprocess.run(
        [sys.executable, str(pathlib.Path(__file__).parent / "resnet.py"), lib],
        capture_output=True, text=True, check=True).stdout
    return {k: float(v) for k, v in
            (line.split("\t") for line in out.strip().splitlines() if "\t" in line)}


TOLERANCE = 1e-4


def test_a_resnet_trains_the_same_way_real_torch_does():
    """Thirty steps, and every number along the way.

    It costs about a second for both runs together, which is why it carries no marker
    saying otherwise. This repository has no registered markers, so one would have been a
    warning that did nothing and a claim about the cost that was not true.
    """
    real = _run("real")
    mine = _run("borch")

    # **A run that produced nothing would compare nothing and pass.** The groups are named
    # rather than counted, because a scenario that silently stopped emitting one of them —
    # the gradients, say — would still satisfy a count.
    assert real, "the scenario produced no values at all"
    for group in ("grad·", "loss·", "buffer·", "eval·"):
        assert any(k.startswith(group) for k in real), (
            f"nothing under `{group}` came back — this file is not comparing what it says "
            "it compares, and would pass by having stopped looking")

    worst, wrong = (0.0, None), []
    for key, expected in real.items():
        if key not in mine:
            wrong.append(f"{key} is absent on the borch side")
            continue
        got = mine[key]
        rel = abs(expected - got) / max(1.0, abs(expected))
        if rel > worst[0]:
            worst = (rel, key)
        if rel > TOLERANCE:
            wrong.append(f"{key} — torch {expected!r} · borch {got!r}  (relative {rel:.2e})")

    assert not wrong, (
        "the run parted from torch's:\n  " + "\n  ".join(wrong) +
        f"\n\nthe worst of all {len(real)} was {worst[1]} at {worst[0]:.2e}.\n"
        "  a `grad·` row is the backward pass before any weight moved — the most local\n"
        "  thing here. `loss·NN` says which step it began at. `buffer·` alone means the\n"
        "  running statistics, which training never reads and `eval` does.")
