"""**The `(torch: …)` fragments are quotes, and nothing was checking that they were.**

Every refusal in `borchvision.py` ends with the sentence real torch would have used:

    interpolation 'bicubic' does not resample here — bilinear or nearest does.
    (torch: …)

The fragment exists so that a learner who searches the error they saw in a tutorial
lands on ours, and so that the three implementations refuse with **one wording**. Its
whole value is that it is torch's sentence rather than a summary of torch's sentence.

They were written by reading torchvision's source and typing what it said, which is
how one of them ended up being a paraphrase: `max_size should only be passed if size
is int or sequence of length 1` reads exactly like a quote and torchvision has never
said it — the real sentence is `…if size specifies the length of the smaller edge`.
Nothing could tell, because each side was checked against itself.

That is the shape another session found in the refusal-message cases: the Python side
looks for its own fragment, the TS side looks for its own, both agree with themselves,
and the two sentences drift apart with every check green. **A wording claim needs the
other side in the room**, so this file puts it there — it triggers the same misuse in
real torchvision and asks whether the quoted words are in the answer.

## Why a subsequence rather than a substring

torch interpolates values into its messages (`Requested crop size (9, 9) is bigger
than input size (5, 4)`), and ours name the same values in their own places. Demanding
a literal substring would force the fragments to carry torch's exact numbers, which is
a different sentence for every call. So the words have to appear **in order**, and the
values between them are free.
"""

import importlib.util
import pathlib
import re
import sys
import warnings

import numpy as np
import pytest

torch = pytest.importorskip("torch")
R = pytest.importorskip("torchvision.transforms")

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import borch as BT                                              # noqa: E402
import borchvision as V                                         # noqa: E402

V.use(BT)
T = V.transforms

_IMG = np.zeros((5, 4, 3), dtype=np.float32)
_TENSOR = torch.zeros(3, 5, 4)
_QUOTED = re.compile(r"\(torch: (.+?)\)\s*$", re.S)
_WORDS = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

# Each row is one misuse, spelled for both libraries. The point is not that both
# refuse — it is that **the words we quote are the words torch uses.**
CASES = (
    ("Pad(padding_mode)", lambda: T.Pad(1, padding_mode="nope"),
     lambda: R.Pad(1, padding_mode="nope")),
    ("Pad(padding length)", lambda: T.Pad((1, 2, 3)), lambda: R.Pad((1, 2, 3))),
    ("a size of three numbers", lambda: T.FiveCrop((1, 2, 3))(_IMG),
     lambda: R.FiveCrop((1, 2, 3))(_TENSOR)),
    ("Lambda(not callable)", lambda: T.Lambda(3), lambda: R.Lambda(3)),
    ("a transform list that is not a sequence", lambda: T.RandomApply(3),
     lambda: R.RandomChoice(3)),
    ("RandomChoice(p not a sequence)", lambda: T.RandomChoice([T.Pad(1)], p=3),
     lambda: R.RandomChoice([R.Pad(1)], p=3)),
    ("Grayscale(two channels out)", lambda: T.Grayscale(2)(_IMG),
     lambda: R.Grayscale(2)(_TENSOR)),
    ("a crop bigger than the picture", lambda: T.FiveCrop((9, 9))(_IMG),
     lambda: R.FiveCrop((9, 9))(_TENSOR)),
    ("LinearTransformation(not square)",
     lambda: T.LinearTransformation(np.zeros((2, 3), np.float32), np.zeros(2, np.float32)),
     lambda: R.LinearTransformation(torch.zeros(2, 3), torch.zeros(2))),
    ("LinearTransformation(mean too short)",
     lambda: T.LinearTransformation(np.eye(3, dtype=np.float32), np.zeros(2, np.float32)),
     lambda: R.LinearTransformation(torch.eye(3), torch.zeros(2))),
    ("LinearTransformation(shapes do not meet)",
     lambda: T.LinearTransformation(np.eye(3, dtype=np.float32),
                                    np.zeros(3, np.float32))(T.ToTensor()(_IMG)),
     lambda: R.LinearTransformation(torch.eye(3), torch.zeros(3))(_TENSOR)),
    ("a range the wrong way round", lambda: T.RandomResizedCrop(2, scale=(1.0, 0.5)),
     lambda: R.RandomResizedCrop(2, scale=(1.0, 0.5))),
    ("RandomErasing(value as a word)", lambda: T.RandomErasing(value="nope"),
     lambda: R.RandomErasing(value="nope")),
    ("RandomErasing(scale outside 0..1)", lambda: T.RandomErasing(scale=(-1.0, 0.5)),
     lambda: R.RandomErasing(scale=(-1.0, 0.5))),
    ("RandomErasing(p outside 0..1)", lambda: T.RandomErasing(p=2.0),
     lambda: R.RandomErasing(p=2.0)),
    ("RandomErasing(a value per channel, wrong count)",
     lambda: T.RandomErasing(p=1.0, value=[1, 2])(T.ToTensor()(_IMG)),
     lambda: R.RandomErasing(p=1.0, value=[1, 2])(_TENSOR)),
    ("Resize(max_size with a pair)", lambda: T.Resize((4, 3), max_size=9),
     lambda: R.Resize((4, 3), max_size=9)(_TENSOR)),
    ("Resize(max_size no larger than the short side)",
     lambda: T.Resize(4, max_size=4)(_IMG), lambda: R.Resize(4, max_size=4)(_TENSOR)),
)


def _what_it_said(call):
    """`(kind, sentence)` — and **the kind is half the answer.**

    torchvision warns rather than refuses when a range arrives the wrong way round.
    A version of ours that raised there would stop a line that runs over there, and
    since the wording would be torch's either way, comparing only the sentence calls
    that a match. It was a match: this file passed a mutation that turned the warning
    back into a refusal, until the kind was compared too.

    The exception *class* is deliberately not compared. `TypeError` against
    `ValueError` is a difference nobody meets — the line that matters is whether the
    program stops.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        try:
            call()
        except Warning as w:
            return "warning", str(w)
        except Exception as e:                  # noqa: BLE001 - the message is the subject
            return "refusal", str(e)
    return None, None


def _in_order(words, inside):
    """Every word of `words`, in order, somewhere in `inside`."""
    haystack = [w.lower() for w in _WORDS.findall(inside)]
    at = 0
    for want in (w.lower() for w in _WORDS.findall(words)):
        while at < len(haystack) and haystack[at] != want:
            at += 1
        if at == len(haystack):
            return False
        at += 1
    return True


@pytest.mark.parametrize("name,ours,theirs", CASES, ids=[c[0] for c in CASES])
def test_the_quoted_torch_wording_is_torchs_wording(name, ours, theirs):
    kind, said = _what_it_said(ours)
    assert said is not None, f"{name}: ours accepted it and torchvision does not"
    # **Two shapes, one question.** Most of ours are our own sentence with torch's
    # quoted after it; a few *are* torch's sentence outright, and wrapping those in a
    # `(torch: …)` marker would be quoting ourselves. Either way what is asked is
    # whether the words shown are torch's.
    quoted = _QUOTED.search(said)
    claim = quoted.group(1) if quoted else said
    theirs_kind, theirs_said = _what_it_said(theirs)
    assert theirs_said is not None, (
        f"{name}: we refuse and torchvision does not. A refusal it does not make stops "
        "code that runs over there — either it is not a misuse, or it is a warning.")
    assert kind == theirs_kind, (
        f"{name}: ours is a {kind} and torchvision's is a {theirs_kind}.\n"
        "  A refusal where torch warns stops a line that runs over there; a warning "
        "where torch refuses lets a mistake through. The wording being identical is "
        "what makes this one invisible to a comparison of sentences.")
    assert _in_order(claim, theirs_said), (
        f"{name}: the quoted words are not torchvision's, in this order.\n"
        f"  quoted : {claim}\n"
        f"  torch  : {theirs_said}\n\n"
        "A `(torch: …)` fragment is a quote. Written from the source's docstring rather "
        "than from the message, it reads exactly like one and is not — which is how "
        "`max_size should only be passed if size is int or sequence of length 1` got in.")
