"""**Asks torch what it does with an argument, not what it declares.**

Every other check here reads a declaration or compares a value. None of them can see an
argument that torch *accepts and then throws away* — and that class is not rare, it is
where torch keeps its deprecations.

`F.gumbel_softmax(eps=…)` is the worked example. torch stopped using it when its noise
moved to an exponential draw that needs no floor, kept the parameter so old calls still
parse, and warns if you pass one. The core, meanwhile, put the caller's value **inside
the noise**. At `eps=1e-1` one output moved by 0.49 and the mean of 20,000 draws went
from `[0.728, 0.185, 0.087]` to `[0.743, 0.177, 0.080]`, where torch's was bit-identical
at both values.

**Every structural check was green throughout**, and not by accident. They ask whether a
call site drops an argument it accepts. An implementation that uses a dropped argument
enthusiastically satisfies that question *exactly as well as* one that is right. The
more diligently the argument was honoured, the greener the report.

## The instrument that was built first and thrown away

The obvious version reads torch's docstrings for parameters its prose calls deprecated
or ignored. It was built and run: **108 rows** across `F`, `torch` and `nn`. Almost all
were `size_average` and `reduce` — the pair already folded in
`torch_signatures_core.DEPRECATED` — and much of the rest was the scan misreading
itself, `F.cross_entropy.target` matching because the *`ignore_index`* line contains the
word "ignored".

**`gumbel_softmax.eps` was not among the 108.** Its docstring says nothing about it. The
deprecation exists only as a runtime warning.

So the scan missed the single case known in advance to be positive while producing a
report that reads as thorough. Prose about behaviour is not behaviour, and a scan over
prose inherits every silence in the prose. The authority is the call.

## Two things measured before this file had a shape

**torch warns on the value, not on the presence.** `F.gumbel_softmax(x, eps=1e-10)` —
its own default — is silent; `eps=1e-9` warns. So the probe cannot simply pass each
default back; it has to supply a *different* value, which is why `PROBE` below carries
one per argument rather than deriving it.

**A silent probe and a clean library look identical.** `catch_warnings(record=True)`
around a call that never happened returns an empty list, exactly like a call torch had
nothing to say about — so an argument that stopped being reachable would read as a pass.
`test_the_probe_actually_calls_torch` puts a floor under the number of calls that really
ran, and every unprobed pair is listed by name with its reason.
"""

import pathlib
import sys
import warnings

import numpy as np
import pytest

pytest.importorskip("torch")
import torch                                                     # noqa: E402
import torch.nn.functional as TF                                 # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import borch                                                     # noqa: E402

LOGITS = np.array([[1.0, 2.0, 0.5], [0.0, -1.0, 3.0]], dtype=np.float32)
TARGET = np.array([0, 2], dtype=np.int64)
PROBS = np.array([[0.2, 0.7], [0.6, 0.1]], dtype=np.float32)
BINARY = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

# Each row: how to call the function on either library, and the argument to probe with
# the value to probe it with. **The value must differ from torch's default** — see the
# module docstring; passing the default back is silent on both sides and would make
# every row a false pass.
PROBE = [
    ("gumbel_softmax", "eps", 1e-9,
     lambda L, F: (F.gumbel_softmax, (L.tensor(LOGITS.copy()),))),
    ("cross_entropy", "size_average", True,
     lambda L, F: (F.cross_entropy,
                   (L.tensor(LOGITS.copy()), L.tensor(TARGET.copy())))),
    ("cross_entropy", "reduce", True,
     lambda L, F: (F.cross_entropy,
                   (L.tensor(LOGITS.copy()), L.tensor(TARGET.copy())))),
    ("nll_loss", "size_average", True,
     lambda L, F: (F.nll_loss,
                   (F.log_softmax(L.tensor(LOGITS.copy()), dim=1),
                    L.tensor(TARGET.copy())))),
    ("nll_loss", "reduce", True,
     lambda L, F: (F.nll_loss,
                   (F.log_softmax(L.tensor(LOGITS.copy()), dim=1),
                    L.tensor(TARGET.copy())))),
    ("mse_loss", "size_average", True,
     lambda L, F: (F.mse_loss,
                   (L.tensor(PROBS.copy()), L.tensor(BINARY.copy())))),
    ("mse_loss", "reduce", True,
     lambda L, F: (F.mse_loss,
                   (L.tensor(PROBS.copy()), L.tensor(BINARY.copy())))),
    ("l1_loss", "size_average", True,
     lambda L, F: (F.l1_loss,
                   (L.tensor(PROBS.copy()), L.tensor(BINARY.copy())))),
    ("binary_cross_entropy", "size_average", True,
     lambda L, F: (F.binary_cross_entropy,
                   (L.tensor(PROBS.copy()), L.tensor(BINARY.copy())))),
    ("kl_div", "size_average", True,
     lambda L, F: (F.kl_div,
                   (F.log_softmax(L.tensor(LOGITS.copy()), dim=1),
                    F.softmax(L.tensor(LOGITS.copy()), dim=1)))),
    ("smooth_l1_loss", "size_average", True,
     lambda L, F: (F.smooth_l1_loss,
                   (L.tensor(PROBS.copy()), L.tensor(BINARY.copy())))),
    ("soft_margin_loss", "size_average", True,
     lambda L, F: (F.soft_margin_loss,
                   (L.tensor(PROBS.copy()),
                    L.tensor(np.array([[1.0, -1.0], [-1.0, 1.0]],
                                      dtype=np.float32))))),
]

# Where the core refuses the argument outright rather than accepting and ignoring it.
#
# **This set held eleven rows and is empty.** They were the deprecated
# `size_average`/`reduce`, folded away on the ground that torch ignores them once
# `reduction` is given. Measured, torch does the opposite — the pair wins over
# `reduction` — and folding them also moved every later argument one or two seats
# forward, so `F.l1_loss(a, b, 'sum')` returned the sum here and the mean in torch.
#
# So they went in, at torch's own seats, folded by `_ops._legacy_reduction` and warned
# about in torch's own wording. **This file's message is what said how**: *"if it was
# added on purpose, take the row out of REFUSED — and make sure it warns, because torch
# does."* The instruction was written by somebody who expected to be wrong about this,
# and it was followed exactly.
#
# The set is kept rather than deleted, for the reason the docstring above gives about
# empty buckets: at zero it fails the day a refusal comes back, and a set that is not
# there cannot.
REFUSED = set()


def _said(call, arg, value):
    """`(warnings, error)` from one call. Never both, and never neither silently."""
    fn, args = call
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            with np.errstate(all="ignore"):
                fn(*args, **{arg: value})
        except Exception as e:                                   # noqa: BLE001
            return [], f"{type(e).__name__}: {str(e).splitlines()[0][:70]}"
    return [str(w.message) for w in caught], None


def _torch_side(row):
    _, arg, value, build = row
    return _said(build(torch, TF), arg, value)


def _core_side(row):
    _, arg, value, build = row
    return _said(build(borch, borch.nn.functional), arg, value)


def test_the_probe_actually_calls_torch():
    """**A floor under how many calls really ran.**

    `catch_warnings(record=True)` around a call that raised, or that was never made,
    returns an empty list — indistinguishable from a call torch had nothing to say
    about. Without this, a probe that stopped reaching torch would report no
    differences, which reads exactly like a library with none.
    """
    reached = [row for row in PROBE if _torch_side(row)[1] is None]
    assert len(reached) > 8, (
        f"only {len(reached)} of {len(PROBE)} rows reached torch at all.\n"
        "  These rows are the instrument. If torch refuses the call, the warning list\n"
        "  is empty for a reason that has nothing to do with the argument, and every\n"
        "  row below passes for free.")


@pytest.mark.parametrize("row", PROBE, ids=lambda r: f"{r[0]}.{r[1]}")
def test_torch_warns_about_every_argument_this_file_claims_it_ignores(row):
    """The premise of each row, checked rather than assumed.

    If torch stops warning — the parameter is finally removed, or the message moves —
    the row below would compare our behaviour against a claim that is no longer true.
    """
    said, err = _torch_side(row)
    assert err is None, f"torch refused the probe call itself: {err}"
    assert said, (
        f"torch no longer warns about {row[0]}({row[1]}=…).\n"
        "  Either it was removed, or it now does something with the value. Re-measure\n"
        "  before changing our side — this file's whole premise is that call, not a\n"
        "  docstring.")


@pytest.mark.parametrize("row", PROBE, ids=lambda r: f"{r[0]}.{r[1]}")
def test_the_core_does_not_quietly_honour_what_torch_ignores(row):
    """**Accept and warn, or refuse. Never accept and use.**

    The third option is the defect: the answer parts from torch's while every check
    that reads structure stays green, because the argument is visibly in use.
    """
    name, arg, _, _ = row
    said, err = _core_side(row)

    if (name, arg) in REFUSED:
        assert err is not None, (
            f"{name}({arg}=…) is listed in REFUSED but the core accepted it.\n"
            "  If it was added on purpose, take the row out of REFUSED — and make sure\n"
            "  it warns, because torch does.")
        assert "TypeError" in err, (
            f"{name}({arg}=…): expected a refusal, got {err}")
        return

    assert err is None, f"the core refused a call it should answer: {err}"
    assert said, (
        f"{name}({arg}=…): torch warns that this argument has no effect, and the core\n"
        "  said nothing. Silence here means one of two things and both are bad — the\n"
        "  argument is being used (the answers part, invisibly), or it is dropped\n"
        "  without telling the caller, who thinks they set something.")
    assert any("deprecated" in m or "no effect" in m for m in said), (
        f"{name}({arg}=…): the core warned, but not about this.\n  said: {said}")
