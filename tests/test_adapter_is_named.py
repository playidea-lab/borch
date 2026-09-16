"""**A green browser golden has to say what ran the shaders.**

`tests/browser/runner.html` reads the WebGPU adapter into `GOLDEN_RESULT.backend` and
has done so all along. `tests/browser/run.py` never printed it. So every binding run
this repository has ever reported carried a number and no word about what produced it,
and both sessions holding this codebase learned — from the *other* runner's output, on
the same afternoon — that every browser golden they had run was on SwiftShader.

That matters because WGSL goes through a **different compiler per vendor.** Integer
division, boundary handling and rounding are exactly where those compilers part, and
the kernels written this week (an average pool's divisor, `poolOut`'s ceil, nearest at
an arbitrary scale) are all three of those things.

## Why the note goes on the score line and not above it

The comment beside `_adapter_note` in `run.py` records the same mistake made once
already about a different word: the library's name sat on the header, somebody read
`agreeing 3255/3255`, and reported the binding clear. The repair then was to move the
name onto the line carrying the number — and the adapter, sitting in the same place,
was not moved, because that repair was made about `lib` rather than about *the line a
person reads.*

**Printing it somewhere is demonstrably not enough.** borch.ts's runner prints the
adapter at the top of every run. Two sessions read that output more than twenty times
between them and neither saw it. Distance from the number is what makes a true line
invisible.

## What a green run here does not say

- **Not that the golden refuses to run on software.** It must not: a green on
  SwiftShader still proves the values, and refusing to run on the only machine
  available trades a real check for none.
- **Not that a real adapter was reached.** That is what the note exists to say when
  it was not.
"""

import importlib.util
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
RUNNER = ROOT / "tests" / "browser" / "run.py"


def _runner():
    """Import `run.py` as a module. It is a script, not a package member.

    Its own directory has to go on the path first — it does `from launch import
    browser`, which resolves only when `tests/browser` is importable. Reading the
    source with a regex instead would be the easier way in and the wrong one: the
    thing under test is what the function *returns*, and a regex asks what it looks
    like.
    """
    if str(RUNNER.parent) not in sys.path:
        sys.path.insert(0, str(RUNNER.parent))
    spec = importlib.util.spec_from_file_location("_browser_run", RUNNER)
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("_browser_run", mod)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.skipif(not RUNNER.exists(), reason="no tests/browser/run.py")
def test_a_software_adapter_is_said_out_loud():
    note = _runner()._adapter_note
    for backend in ("borch.ts — google / swiftshader",
                    "borch.ts — mesa / llvmpipe (LLVM 15)",
                    "borch.ts — SwiftShader Device (LLVM 10)"):
        got = note({"backend": backend})
        assert "software adapter" in got.lower(), (
            f"{backend!r} produced no warning:\n  {got!r}\n\n"
            "  A run on a software rasteriser reports the same word — `agreeing` — as a\n"
            "  run on a GPU, and the two prove different things.")


@pytest.mark.skipif(not RUNNER.exists(), reason="no tests/browser/run.py")
def test_a_real_adapter_is_named_and_not_warned_about():
    """**The other direction, and it is the one that keeps the warning worth reading.**

    A note that fires on every adapter says nothing about this one. It would also be
    the third instrument in this repository to claim credit for firing on everything —
    `test_scheduler_table` watches that end for the rename folds.
    """
    note = _runner()._adapter_note
    for backend in ("borch.ts — apple / Apple M2 Pro",
                    "borch.ts — nvidia / NVIDIA GeForce RTX 4090"):
        got = note({"backend": backend})
        assert backend in got, f"the adapter was not named at all: {got!r}"
        assert "software" not in got.lower(), (
            f"a real adapter was called software:\n  {got!r}")


@pytest.mark.skipif(not RUNNER.exists(), reason="no tests/browser/run.py")
def test_the_core_run_says_nothing_because_no_shader_ran():
    """`--lib borch` is numpy inside Pyodide. There is no adapter to name, and a line
    about one would be an invented fact rather than a missing one."""
    note = _runner()._adapter_note
    assert note({"backend": "numpy (no browser GPU)"}) == ""
    assert note({}) == "", "an absent backend must not be described"
    assert note({"backend": ""}) == ""


@pytest.mark.skipif(not RUNNER.exists(), reason="no tests/browser/run.py")
def test_the_note_is_actually_on_the_line_that_carries_the_score():
    """**The whole point is placement, and placement is not covered by the three above.**

    Every assertion so far would pass with the note printed on its own line at the top
    of the run — which is precisely the arrangement borch.ts's runner has, and which
    two readers missed twenty times. So the call site is read: `_adapter_note` has to
    be interpolated into the `agreeing` line itself.
    """
    src = RUNNER.read_text(encoding="utf-8")
    score = [ln for ln in src.splitlines() if "agreeing" in ln and "print(" in ln]
    assert score, "the line printing the score was not found — this check is blind"
    assert any("_adapter_note" in ln for ln in score), (
        "`_adapter_note` is no longer part of the score line:\n  "
        + "\n  ".join(score) + "\n\n"
        "  Moving it to its own print passes every other assertion in this file and\n"
        "  undoes the only thing they were written for.")


# ── the same repair, one runner over ────────────────────────────────────────
#
# **This file was written about `tests/browser/run.py` and the defect lives in two
# runners.** `borch-ts/test/run.py` had the adapter nineteen lines above its score,
# with the failing cases in between — the identical arrangement, found only because a
# second session went looking after reading about the first.
#
# That is the shape the commit above names: *a fix aimed at one symptom leaves the
# same class one variable over.* A check aimed at one runner does the same, so this
# one is aimed at the position rather than at the file.

TS_RUNNER = ROOT / "borch-ts" / "test" / "run.py"


@pytest.mark.skipif(not TS_RUNNER.exists(), reason="no borch-ts/test/run.py")
def test_the_ts_runner_names_the_adapter_beside_its_score():
    """borch.ts's runner prints `passed N / failed M`. The adapter belongs on it.

    **It is asked by reading the source rather than by calling a function**, because
    this runner has no `_adapter_note` to call — it interpolates the adapter directly.
    Asking for a particular helper would be asking about how, and what has to be true
    is where.
    """
    src = TS_RUNNER.read_text(encoding="utf-8")
    score = [ln for ln in src.splitlines()
             if "print(" in ln and "passed " in ln and "failed" in ln]
    assert score, "the line printing the score was not found — this check is blind"
    assert any("adapter" in ln for ln in score), (
        "the adapter is not on borch.ts's score line:\n  " + "\n  ".join(score) + "\n\n"
        "  Printing it earlier passes nothing here on purpose: whoever wants the count\n"
        "  reads the last lines, and a whole session of runs went by on\n"
        "  `google / swiftshader` with the warning on screen every time.")


# The four words that mean a CPU pretended to be a GPU. Kept here as data rather than
# imported, so that a change to the one definition has to meet this list too.
_NAMES = ("swiftshader", "llvmpipe", "lavapipe", "software")


def test_only_one_place_in_the_repository_names_the_software_adapters():
    """**One definition, or a green run says the wrong word.**

    `tests/browser/run.py` kept its own tuple of these names beside `launch.py`'s regex —
    the same four words, two spellings — while importing `is_software` from that module
    four lines from the top. Nothing would have reddened if a fifth rasteriser reached one
    and not the other. The run would keep passing and simply stop calling a CPU a CPU on
    the line carrying the score, which is the one line a person reads.

    **This is why the rule is worth a check while a skip list is not.** A page-skip list
    that drifts turns CI red the same day, as `clipped.py` did. This one drifts into a
    green run wearing a wrong word, and nobody looks for a failure that never happened.

    Asked of the tree rather than of a list of files: a second copy put somewhere new is
    exactly the case a list of files would miss.

    **Comment lines are skipped, and that is a decision rather than an oversight.** The
    first version counted `launch.py`'s own comment — *Chrome has SwiftShader; Linux Mesa
    has lavapipe (llvmpipe)* — as a second copy, which would have taught the next person
    to delete the sentence that explains the definition. A comment cannot make two runs
    disagree; only code can. So the check asks about code, and prose is free to name as
    many rasterisers as it takes to be clear.
    """
    offenders = []
    for path in sorted((ROOT / "tests").rglob("*.py")) + sorted((ROOT / "borch-ts" / "test").rglob("*.py")):
        if path.name == pathlib.Path(__file__).name:
            continue
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if line.lstrip().startswith("#"):
                continue
            low = line.lower()
            if sum(name in low for name in _NAMES) >= 3:
                offenders.append(f"{path.relative_to(ROOT)}:{n}  {line.strip()[:100]}")

    assert len(offenders) == 1, (
        "the software-adapter names should be written once, in `tests/browser/launch.py`.\n"
        "Found " + str(len(offenders)) + ":\n  " + "\n  ".join(offenders) + "\n\n"
        "  Import `is_software` from `launch` instead of spelling the names again. Two\n"
        "  copies do not disagree on the day they are written; they disagree on the day\n"
        "  one of them learns a new rasteriser's name.")
    assert "launch.py" in offenders[0], (
        "the one definition is no longer in `tests/browser/launch.py` but in:\n  "
        + offenders[0])
