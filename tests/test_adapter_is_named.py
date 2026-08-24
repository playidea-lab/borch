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
