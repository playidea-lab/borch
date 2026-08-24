"""**A document may not claim a vendor the code says was never measured.**

`README.md` said, for months, that the golden matched *across two vendors, Apple Metal
and NVIDIA (RTX 4090)*. `tests/browser/launch.py` has said this since the day that run
happened:

> Running the golden cases headless on a Linux GPU server gave 845/845 while the
> adapter was `google / swiftshader` — the pass was real and the claim "confirmed on
> another vendor" was false.

**Same run, same number.** The pass was real; the vendor was not. Headless hands back
Chrome's software rasteriser, which answers every WebGPU call correctly — so the values
proved the logic and nothing about a GPU. The README kept the number and dropped the
adapter, which is the entire distinction.

Two files, one about the other, and nothing compared them. The correction was written
where the browser is opened and the claim went on standing where a reader would meet
it — the same shape as everything else this week, at the largest scale available: a
public page.

## Why the number is what is watched

`845` is not a magic constant; it is the fingerprint. Three facts around the claim were
true and checkable — 845 was the table's size then, it is not today's count, and that
machine really has been unavailable since — and a reader confirming those three walks
past the one that is false. So the check is on the pairing: **that count and a vendor
name in the same sentence**, anywhere a reader reads.

## What a green run does not say

- **Not that NVIDIA works or does not.** It says the documents do not claim it was
  measured when the only run cited is on record as a CPU.
- **Not that other vendor claims are true.** A vendor named with no measurement behind
  it is out of this check's reach — this holds the one that actually happened.
"""

import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent

# The documents a reader meets. `site/` is deployed on every push to `main`, so a
# sentence here is a published sentence.
LIVE = [
    "README.md",
    "site/index.html",
    "site/ko/index.html",
]

LAUNCH = ROOT / "tests" / "browser" / "launch.py"

# The run that was cited as the NVIDIA evidence. Kept as a number because that is what
# a document quotes; the reason it is *this* number is in `launch.py`.
SWIFTSHADER_RUN = "845"

VENDOR = re.compile(r"NVIDIA|RTX|4090|5080", re.I)


def test_the_repository_still_records_what_that_run_was():
    """**The premise, read rather than remembered.**

    Everything below rests on `launch.py` saying the 845 run was SwiftShader. If that
    sentence is ever reworded away, this file is asserting from memory — which is the
    failure it exists to catch, one level up.
    """
    assert LAUNCH.exists(), "no launch.py — this check has lost its source"
    src = LAUNCH.read_text(encoding="utf-8")
    assert SWIFTSHADER_RUN in src, (
        f"`launch.py` no longer mentions the {SWIFTSHADER_RUN} run. That run is the\n"
        "  only NVIDIA evidence any document has ever cited, and what it actually was\n"
        "  is recorded there and nowhere else.")
    assert "swiftshader" in src.lower(), (
        "`launch.py` no longer says what adapter that run had. Without it the number\n"
        "  is just a number again, and the claim it was used for reads as measured.")


# **A line that names the adapter is telling the story, not making the claim.**
#
# The first version of the check below flagged its own correction: the paragraph that
# explains what the 845 was has to say `845` and `NVIDIA` in one line to explain it, and
# so does the sentence *no run on an NVIDIA WebGPU adapter exists*. A net that catches
# the retraction along with the claim makes the retraction unwritable, which leaves
# saying nothing as the only way to pass — and saying nothing is how the claim lasted.
#
# The distinction is the adapter. Crediting the run to a vendor means naming the vendor
# **and not naming what the run actually had**; a line carrying `swiftshader`, or a
# denial, is doing the opposite job.
TELLING = re.compile(r"swiftshader|software|CPU|no run|없다|아니다|false", re.I)


@pytest.mark.parametrize("rel", LIVE)
def test_no_live_document_credits_that_run_to_a_vendor(rel):
    path = ROOT / rel
    if not path.exists():
        pytest.skip(f"no {rel}")
    bad = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if SWIFTSHADER_RUN in line and VENDOR.search(line) and not TELLING.search(line):
            bad.append(f"{rel}:{i}  {line.strip()[:96]}")
    assert not bad, (
        "a document names a GPU vendor in the same sentence as the run that was\n"
        "  measured on a CPU:\n  " + "\n  ".join(bad) + "\n\n"
        "  `tests/browser/launch.py`: *845/845 while the adapter was "
        "`google / swiftshader`\n  — the pass was real and the claim \"confirmed on "
        "another vendor\" was false.*\n\n"
        "  Quoting the count without the adapter is how that claim survived for months.\n"
        "  Say the adapter, or do not cite the run.")


@pytest.mark.parametrize("rel", LIVE)
def test_no_live_document_says_two_vendors(rel):
    """**The claim in its shortest form**, which is how it was actually written.

    The README did not name SwiftShader and then contradict itself; it said *across two
    vendors* and moved on. A count of vendors is the same assertion with the evidence
    left out, and it needs no number beside it to be wrong.
    """
    path = ROOT / rel
    if not path.exists():
        pytest.skip(f"no {rel}")
    claim = re.compile(r"(two|both)\s+vendors?|두 벤더", re.I)
    bad = [f"{rel}:{i}  {line.strip()[:96]}"
           for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
           if claim.search(line) and "one vendor" not in line.lower()]
    assert not bad, (
        "a document claims two vendors:\n  " + "\n  ".join(bad) + "\n\n"
        "  No run on an NVIDIA WebGPU adapter exists in this repository. The 845 was\n"
        "  SwiftShader; one 4090 has a card off the PCI bus and never opened a\n"
        "  browser; the 5080 has no login session, so headless gave SwiftShader again\n"
        "  and headed could not start.\n\n"
        "  When one does exist, this check is what needs deleting — deliberately, by\n"
        "  somebody holding the run that replaces it.")
