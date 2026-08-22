"""The six browser runners judge by state now, and this holds that.

`borch-ts/test/verdict.py` is the shared judgement. It cannot be exercised by the browser
runners here — those need a GPU session — so the states are handed to it directly, which
is the whole point of the change: the verdict is a function of the checks and of nothing
else, so it can be asked about a state that has never been produced.

The last test is the defect the conversion was for, written as the shape that produced it
rather than as a description of it.
"""

import io
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "borch-ts" / "test"))

verdict_module = pytest.importorskip("verdict")
failures, verdict = verdict_module.failures, verdict_module.verdict


def _result(checks, text="whatever the page happened to print"):
    return {"text": text, "checks": checks, "adapter": "test"}


def test_all_passing_is_zero():
    ok = [{"name": "a", "ok": True, "note": ""}, {"name": "b", "ok": True, "note": ""}]
    assert verdict(_result(ok), "example") == 0


def test_one_failing_is_one_and_it_is_named():
    checks = [{"name": "a", "ok": True, "note": ""},
              {"name": "the one that broke", "ok": False, "note": "3.0 → 3.0"}]
    out = io.StringIO()
    assert verdict(_result(checks), "example", stream=out) == 1
    said = out.getvalue()
    assert "the one that broke" in said, "a failing check has to be named, not counted"
    assert "3.0 → 3.0" in said, "the note is the evidence — it goes with the name"


def test_missing_checks_stops_rather_than_passing():
    """The absence of state is not the absence of failure.

    A page that stops exposing `checks` would otherwise produce "0 failures", which is the
    same silence the conversion removed, arriving under a new name.
    """
    with pytest.raises(SystemExit) as stopped:
        failures({"text": "everything is fine"}, "example")
    assert "checks" in str(stopped.value)


def test_empty_checks_stops_too():
    """Zero checks passing zero checks is not evidence, and it reads exactly like proof."""
    with pytest.raises(SystemExit):
        failures(_result([]), "example")


def test_the_partial_pass_that_the_text_search_read_as_a_pass():
    """`readme.py`'s defect, as the state that produced it.

    The old judgement was `"그대로 돌고" in result["text"]`, and that phrase sits in the
    success sentence of **both** README examples. So with the first example failing and
    the LBFGS one passing, the phrase was still in the text and the runner exited 0 — a
    documented example whose loss does not go down, reported as fine.

    The text below is the one that shipped. It still contains the phrase, and the verdict
    is now 1, because it is read from the checks.
    """
    text = ("  ✗ README 예시가 적힌 그대로 돌고, 손실이 내려간다 — 2.3026 → 2.3026\n"
            "  ✓ README LBFGS 예시가 한 스텝에 손실을 내린다 — 2.3000 → 0.6931")
    checks = [
        {"name": "README 예시가 적힌 그대로 돌고, 손실이 내려간다", "ok": False,
         "note": "2.3026 → 2.3026"},
        {"name": "README LBFGS 예시가 한 스텝에 손실을 내린다", "ok": True,
         "note": "2.3000 → 0.6931"},
    ]
    assert "그대로 돌고" in text, "the phrase the old rule looked for is still there"
    assert verdict(_result(checks, text), "README 예시", stream=io.StringIO()) == 1
