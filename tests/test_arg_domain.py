"""**An argument whose domain is countable gets every value tried** — the other half.

The question one axis back was whether shaking an argument changes the answer. Something
slips that net — **a place that takes the argument and handles only some of its values.**
Where an `else` carries one value's name and swallows the rest of the domain, as in
`if p == 1: … else: L2`, shaking it does change the answer, so it passes. The changed answer
is simply wrong.

This question needs **only the specification, not the code.** torch's documentation writes
every value down.

**The table moved into the golden cases** (twenty-three `loss::reduction::*`, five
`index::searchsorted(side=…)`). What is left here is only **what the golden cases cannot
ask.**

## What it caught (history)

`reduction` was missing from the five most common losses — `.mean()` was written into the
body, so passing `reduction=` raised `TypeError`. The clue was **that it was inverted**: the
rare losses such as `cosine_embedding_loss`, `multi_margin_loss` and `triplet_margin_loss`
all thirteen took it. What was written later followed torch's signature and what was written
first was never fixed. The golden cases missed it because textbooks use only the default
`mean` — **the most-used place was the least-asked-about one.**

`_reduce`'s `else` swallowed a typo: `reduction="MEAN"` quietly trained on the mean.
`searchsorted(side=…)` went into `**kw` and vanished, while `bucketize(right=True)`, the
other name for the same computation, was right from the start.
"""

import inspect

import numpy as np
import pytest
import torch

import borch

rng = np.random.default_rng(0)
A = rng.standard_normal((4, 3)).astype(np.float32)
B = rng.standard_normal((4, 3)).astype(np.float32)
P = np.clip(np.abs(rng.random((4, 3))).astype(np.float32), 0.05, 0.95)
Y = (P > 0.5).astype(np.float32)
LOGP = np.log(np.abs(rng.random((4, 3))).astype(np.float32) + 0.05)
LABEL = np.array([0, 1, 2, 1])

# **The four that could not go into the golden cases.** borch.ts's `nllLoss` and
# `crossEntropy` return a scalar only, so `none` cannot be made there — putting them where
# all three are asked at once means fixing that side first.
CORE_ONLY = [
    ("F.cross_entropy",
     lambda L, r: L.nn.functional.cross_entropy(L.tensor(A), L.tensor(LABEL), reduction=r)),
    ("F.nll_loss",
     lambda L, r: L.nn.functional.nll_loss(L.tensor(LOGP), L.tensor(LABEL), reduction=r)),
    ("F.binary_cross_entropy",
     lambda L, r: L.nn.functional.binary_cross_entropy(L.tensor(P), L.tensor(Y), reduction=r)),
    ("nn.CrossEntropyLoss",
     lambda L, r: L.nn.CrossEntropyLoss(reduction=r)(L.tensor(A), L.tensor(LABEL))),
    ("nn.NLLLoss",
     lambda L, r: L.nn.NLLLoss(reduction=r)(L.tensor(LOGP), L.tensor(LABEL))),
    ("nn.BCELoss",
     lambda L, r: L.nn.BCELoss(reduction=r)(L.tensor(P), L.tensor(Y))),
]


@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
@pytest.mark.parametrize("name,fn", CORE_ONLY, ids=[n for n, _ in CORE_ONLY])
def test_the_losses_the_golden_cannot_ask_yet(name, fn, reduction):
    want = np.asarray(fn(torch, reduction).tolist(), dtype=np.float64)
    got = np.asarray(fn(borch, reduction).tolist(), dtype=np.float64)
    assert np.allclose(got, want, atol=1e-5, rtol=1e-5), (
        f"{name}(reduction={reduction!r}): torch gives {np.ravel(want)[:4]} and "
        f"the core gives {np.ravel(got)[:4]}."
    )


def test_the_rare_losses_were_the_ones_that_had_it():
    """Pins the inversion as a sentence — **the rare ones had it and the common ones did not.**

    The golden cases ask about values per case; they cannot ask **whether the signature has
    the argument.** If whoever adds a new loss leaves `reduction` out, this turns red naming
    it even with no case added — which is why this check did not move into the table.
    """
    missing = []
    for name in dir(torch.nn.functional):
        if not (name.endswith("_loss") or name in ("cross_entropy", "kl_div",
                                                   "binary_cross_entropy")):
            continue
        ours = getattr(borch.nn.functional, name, None)
        if ours is None:
            continue
        theirs = inspect.signature(getattr(torch.nn.functional, name)).parameters
        if "reduction" in theirs and "reduction" not in inspect.signature(ours).parameters:
            missing.append(name)
    assert not missing, f"losses that do not take reduction: {missing}"


def test_the_loss_layers_take_it_too():
    """The layers are looked at too. **Textbook code uses layers more than functions.**

    Fixing only the functions leaves `nn.MSELoss(reduction="sum")` stopping with a
    `TypeError`, and that state really existed.
    """
    missing = [name for name in ("MSELoss", "L1Loss", "SmoothL1Loss", "HuberLoss",
                                 "CrossEntropyLoss", "NLLLoss", "BCELoss",
                                 "BCEWithLogitsLoss", "KLDivLoss")
               if "reduction" not in inspect.signature(
                   getattr(borch.nn, name).__init__).parameters]
    assert not missing, f"loss layers that do not take reduction: {missing}"


@pytest.mark.parametrize("bad", ["meen", ""])
def test_an_unknown_reduction_stops_instead_of_becoming_mean(bad):
    """Where `else: return out.mean()` was swallowing the rest of the domain.

    The golden cases carry `"MEAN"` and `"batchmean"`. The two here are shapes **whose wording
    does not ride in the value** (an empty string), which the golden cases' fragment
    comparison cannot ask about.
    """
    with pytest.raises((ValueError, RuntimeError)):
        borch.nn.functional.l1_loss(borch.tensor(A), borch.tensor(B), reduction=bad)
    # torch stops too — we are not being stricter.
    with pytest.raises((ValueError, RuntimeError)):
        torch.nn.functional.l1_loss(torch.tensor(A), torch.tensor(B), reduction=bad)


def test_kl_div_is_the_only_one_with_a_fourth_value():
    """**The domain differs per function.** This is the worst edge of this net.

    Keeping the list as "argument name → set of values" alone counts
    `l1_loss(reduction="batchmean")` as fine. Something the net does not catch at all (an
    argument scattered across two functions, as `nonlinearity` is across `rnn_tanh` and
    `rnn_relu`) draws the eye by being uncaught; this one **is caught and is wrong.**
    """
    pred = np.log(np.array([[0.1, 0.6, 0.3], [0.5, 0.2, 0.3]], dtype=np.float32))
    target = np.array([[0.2, 0.5, 0.3], [0.3, 0.4, 0.3]], dtype=np.float32)
    batch = borch.nn.functional.kl_div(borch.tensor(pred), borch.tensor(target),
                                       reduction="batchmean").item()
    mean = borch.nn.functional.kl_div(borch.tensor(pred), borch.tensor(target),
                                      reduction="mean").item()
    assert np.isclose(batch, torch.nn.functional.kl_div(
        torch.tensor(pred), torch.tensor(target), reduction="batchmean").item())
    # Dividing by the batch and dividing by the element count have to **differ** — equal, it asks nothing.
    assert not np.isclose(batch, mean)
    # And **in another loss it is a wrong name.**
    with pytest.raises((ValueError, RuntimeError)):
        borch.nn.functional.l1_loss(borch.tensor(A), borch.tensor(B),
                                    reduction="batchmean")
