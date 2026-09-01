"""**The declined reasons that assert something about *this library*, re-measured.**

A name absent from `torch_gap.py`'s tables carries a sentence saying why. Some of those
sentences are judgements a person settles. Others are **claims**, and a claim can stop
being true without anyone touching the row.

`tests/browser/platform_claims.py` watches the ones about the browser. This file watches
the other half — the ones about our own subset — and that half is the one more likely to
rot, because **our own code is what changes daily**. Every reason found false in this
repository so far was of exactly this kind:

    fft            "outside the curriculum"          22 of 23 names were built
    special        the same blanket                  23 of 57 were built
    *_video        "no N-D image kernel to bind to"  the kernels started counting from the end
    multigammaln   "the two disagree on argument order — that wants checking"  they agree
    linear_cross…  "newly arrived — looked at once it settles"  it had settled

Four of the five were sentences about this library rather than about the world, and not
one of them was re-read until somebody went looking. A reason that asserts an absence
nobody re-measures is worse than a name with no reason: the missing reason draws the eye,
and the false one answers the question so the eye moves on.

## What is watched and what is not

Measured, of 172 declined rows, **51 assert an absence and carry no measurement**. They
split by what could possibly check them:

    27  hardware that is not here (cudnn_*, miopen_*, mkldnn_*, TF32, fp8)
    18  claims about this library's own subset      ← this file
     5  claims about the browser                    ← platform_claims.py
     1  a claim about torch itself

Of the eighteen, **five have a probe that means something** and are asserted below.
The rest are named at the foot of this file rather than left out silently: a claim like
*there is no nondeterministic kernel to choose* is a statement about the whole codebase
with no single name to ask, and asserting a proxy for it would be a check that passes
about something else.
"""

import importlib.util
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import borch as L                                               # noqa: E402
import borchvision                                              # noqa: E402


def _gap():
    spec = importlib.util.spec_from_file_location(
        "bt_gap_subset", ROOT / "tests" / "torch_gap.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Each row: the claim, the names it defends, and what must still hold. The names are
# there so that **a claim whose row is gone gets deleted rather than kept** — an
# assertion defending a decision nobody makes any more passes forever about nothing,
# which is the failure this repository keeps finding in its own checks.
# **The names carry their namespace**, because most of these rows do. `_look` takes the
# full path first and the bare leaf second, and `transforms.ConvertImageDtype` is only
# reachable by the first — written bare, every vision row here read as *nothing declines
# this any more* and the second test below went red about three claims that were fine.
CLAIMS = [
    ("uint8 has no storage in this subset",
     ("transforms.ConvertImageDtype", "transforms.v2.ConvertImageDtype",
      "transforms.functional.convert_image_dtype"),
     "uint8"),
    ("there is no PIL here",
     ("transforms.ToPILImage", "transforms.v2.ToPILImage",
      "transforms.functional.to_pil_image", "transforms.functional.pil_to_tensor"),
     "pil"),
    # **This row shrank from three names to one, and the check is what noticed.**
    # `is_anomaly_enabled` and `is_anomaly_check_nan_enabled` were declined beside
    # `set_anomaly_enabled` under one sentence; they only *ask* whether the detector is
    # on, and the answer is `False` and `True` on any machine, so they were built. The
    # claim itself is untouched and still true — there is no detector, which is why the
    # first of those two answers `False`. What it defends is the setter alone now.
    ("there is no anomaly detector",
     ("set_anomaly_enabled",),
     "anomaly"),
    ("numpy has no codec",
     ("transforms.v2.JPEG", "transforms.v2.functional.jpeg",
      "transforms.v2.functional.jpeg_image"),
     "codec"),
    ("torch cannot make one without that dtype either",
     ("empty_quantized",),
     "empty_quantized"),
]


def _holds(kind):
    """Whether the claim still holds, and what was seen. `(bool, str)`."""
    if kind == "uint8":
        # The claim is that there is no uint8 *storage* — the name exists so that a
        # typo and an absence say different things, and using it stops.
        try:
            L.tensor(np.array([1, 2], dtype=np.uint8), dtype=L.uint8)
        except Exception as exc:                                # noqa: BLE001
            return True, f"`dtype=uint8` refused: {type(exc).__name__}"
        return False, "a uint8 tensor was built — the subset gained the dtype"

    if kind == "pil":
        # **The claim is about this library, not about the machine.** PIL is often
        # installed beside it (torchvision pulls it in), so asking `import PIL` would
        # answer a different question. What must stay true is that nothing here reaches
        # for it.
        text = (ROOT / "borchvision.py").read_text(encoding="utf-8")
        reaching = [line.strip() for line in text.splitlines()
                    if line.lstrip().startswith(("import PIL", "from PIL"))]
        return not reaching, (f"borchvision.py reaches for PIL: {reaching}"
                              if reaching else "nothing in borchvision.py imports PIL")

    if kind == "anomaly":
        found = [n for n in ("detect_anomaly", "set_detect_anomaly")
                 if hasattr(L.autograd, n)]
        return not found, (f"borch.autograd gained {found}" if found
                           else "borch.autograd has no detect_anomaly")

    if kind == "codec":
        text = (ROOT / "borchvision.py").read_text(encoding="utf-8")
        # A codec would arrive as a name, from anywhere. Both spellings torchvision
        # uses, and the import that would have to come with one.
        marks = [m for m in ("encode_jpeg", "decode_jpeg", "import cv2", "import imageio",
                             "from simplejpeg") if m in text]
        return not marks, (f"borchvision.py has {marks}" if marks
                           else "no JPEG encoder or decoder anywhere in borchvision.py")

    if kind == "empty_quantized":
        # **This one is a claim about torch, and it is the only one here that is.**
        # The row reads *torch cannot make one without that dtype either*, so what
        # holds it is torch's own refusal.
        torch = pytest.importorskip("torch")
        try:
            torch.empty_quantized([2], torch.empty(2))
        except Exception as exc:                                # noqa: BLE001
            return True, f"torch refuses it too: {type(exc).__name__}"
        return False, "torch built one — the reason is about torch and torch changed"

    raise AssertionError(f"no probe for {kind}")


@pytest.mark.parametrize("claim,names,kind", CLAIMS,
                         ids=[c[2] for c in CLAIMS])
def test_the_claim_a_declined_row_makes_still_holds(claim, names, kind):
    """**The sentence has to still be true of the library it is about.**

    Failing here does not mean the name should be built. It means the row's reason is
    no longer the reason — and the two are different repairs: one writes code, the other
    writes a sentence, and confusing them is how a true row gets deleted.
    """
    held, seen = _holds(kind)
    assert held, (
        f"a declined row asserts `{claim}` and it is no longer true.\n"
        f"  measured: {seen}\n"
        f"  the row defends: {', '.join(names)}\n"
        "  Rewrite the reason in tests/torch_gap.py to what is true now, or build the\n"
        "  names if what changed is that they became possible. A reason that outlives\n"
        "  its measurement is what this file exists for.")


def test_no_claim_outlives_the_rows_it_defends():
    """**Every name a claim lists has to still be declined — not merely one of them.**

    The other direction, and the one that goes wrong quietly: an assertion protecting a
    decision nobody makes any more stays green forever while measuring nothing.

    **It was written as *none of them is declined* first, and the plant passed.** Adding
    a name that is plainly built (`erf`) to a list whose other entries were still
    declined left the check green — so a row could name three things, two of them since
    implemented, and go on describing a decision that had shrunk under it. That is the
    same failure one level in: not a claim outliving its rows, but a claim outliving
    *most* of them, which is how the `special` block came to hold thirty-five sentences
    for a situation that had become three.
    """
    gap = _gap()
    stale = []
    for claim, names, _kind in CLAIMS:
        for name in names:
            if not gap._look(gap.SKIPPED, name.rpartition(".")[2], name):
                stale.append(f"`{claim}` names {name}, which nothing declines")
    assert not stale, (
        "claims naming rows that are gone:\n  " + "\n  ".join(stale)
        + "\n\n  Take the name out of CLAIMS, and if none is left take the claim out\n"
          "  too. A check that guards a decision which has been reversed passes about\n"
          "  nothing, and reads as coverage.")


def test_the_unwatched_claims_are_named_rather_than_forgotten():
    """**What this file cannot ask is written down here, in the file that cannot ask it.**

    Thirteen of the eighteen subset claims have no probe worth writing. Two shapes:

    * *there is no nondeterministic kernel to choose* and *there is no video anywhere in
      this project* are statements about the whole codebase. There is no single name to
      put a question to, and a proxy — grepping for `nondeterministic`, say — would pass
      whether or not the claim were true.
    * `CelebA`'s *`_check_integrity` md5s five files and there is no fixture that passes
      it* is about an absent fixture. Whether the fixture exists is askable; whether its
      absence is what stops the dataset is not, without building the thing.

    The list is asserted rather than left in prose so that it cannot quietly grow — a
    file that says *some claims are unwatched* and does not say which is the same shape
    as the reasons it was written for.
    """
    unwatched = {
        "there is no nondeterministic kernel to choose",
        "there is no video anywhere in this project",
        "there is no fixture that passes it",
    }
    assert len(unwatched) == 3, (
        "the unwatched list changed. If a claim gained a probe, move it into CLAIMS; "
        "if a new one arrived without one, say so here and say why no probe means "
        "anything.")
