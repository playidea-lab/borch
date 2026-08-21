"""Looks at whether the golden harness itself does its job.

There is no GPU backend yet. So borch is compared against the golden answers **as though it
were a third library** — that reveals whether the harness runs, and when a backend arrives it
goes into the same place.

But "does it run" is not the only question. A golden harness breaks quietly in two ways: by
comparing against stale golden answers after the table changed, or by comparing while the
inputs have diverged. Both are states where **a pass comes out and nothing was compared.** So
those two are built on purpose to see whether they are caught.

Korean literals below (`낡았다`, `입력이 골든과 다르다`, `GPU 검증 오류`, and the `흐름`
words) are **the wording `golden.py` and `cases.py` print.** They are keys, not prose, and
they move when those files do.
"""

import importlib.util
import pathlib

import pytest

_here = pathlib.Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("bt_golden", _here / "golden.py")
golden = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(golden)


def test_golden_dump_then_check_matches_borch(tmp_path):
    """The core compares **only its own scope.**

    The golden answers are frozen with real torch, so they hold things that exist in the sister
    library alone. The core refuses those on purpose, so they are skipped — the two libraries'
    scopes have begun to diverge, and the divergence is as intended only if the skipped count
    equals the sister-only case count exactly.
    """
    path = tmp_path / "golden.npz"
    count, _ = golden.dump(path)
    assert count > 0, "the golden answers are empty — the case table did not load"

    bad, total = golden.check(golden.load_borch(), path)
    assert total == count - len(golden.cases_mod.webgpu_cases())
    assert not bad, "diverged from the golden answers:\n  " + "\n  ".join(bad)


def test_check_rejects_stale_golden(tmp_path, monkeypatch):
    """After the table changes, stale golden answers **must not produce a pass.**"""
    path = tmp_path / "golden.npz"
    golden.dump(path)
    monkeypatch.setattr(golden.cases_mod, "manifest_hash", lambda cases: "the-hash-after-the-table-changed")
    with pytest.raises(SystemExit, match="낡았다"):
        golden.check(golden.load_borch(), path)


def test_check_rejects_mismatched_inputs(tmp_path, monkeypatch):
    """Diverged inputs have to stop it.

    numpy's `default_rng` promises the same numbers across versions, but with no check on that
    promise, the day it breaks **different inputs get compared and stamped as passing.**
    """
    path = tmp_path / "golden.npz"
    golden.dump(path)
    monkeypatch.setattr(golden.cases_mod, "input_fingerprint", lambda inp: "a-different-input-fingerprint")
    with pytest.raises(SystemExit, match="입력이 골든과 다르다"):
        golden.check(golden.load_borch(), path)


def test_check_names_the_case_that_raised_a_gpu_fault(tmp_path):
    """**Names the case that raised a GPU validation error.**

    WebGPU does not throw one as an exception. An invalid command buffer quietly does nothing,
    so **the culprit passes and a case queued behind it turns red instead** — this happened
    three times (`as_strided_`'s over-copy, a shared buffer in optimizer state, and an
    `index_select` selecting nothing).

    Only the wiring is checked, with no browser. Even with a fake counter doing the counting
    here, the answer has to say **which case the count went up on** — without that, one more
    number is printed in the runner and you are still looking one slot away from the cause.
    """
    path = tmp_path / "golden.npz"
    golden.dump(path)

    # Pretends an error happened exactly once, on the third case.
    state = {"n": 0, "seen": 0}

    def counter():
        state["seen"] += 1
        if state["seen"] == 3:
            state["n"] += 1
        return state["n"]

    bad, _ = golden.check(golden.load_borch(), path, faults=counter)
    hits = [line for line in bad if "GPU 검증 오류" in line]
    assert len(hits) == 1, f"there should be one and there are {len(hits)}:\n  " + "\n  ".join(bad)

    # **Which case it is, is the point.** The first call is outside the loop (the baseline), so
    # the third call lands where the table's second case finished.
    names = [n for n, _ in golden.cases_mod.golden_cases()]
    assert hits[0].startswith(names[1] + ":"), (
        f"it should point at {names[1]}: {hits[0]}")


# ---- whether the gradient-flow table does its job
#
# This table asks not about values but "does a gradient flow". If it **asks and cannot catch**,
# it does nothing while adding one more green. So the two shapes of a quiet break are built on
# purpose to see whether they are caught — both really did occur in the sister library.

class _Shim:
    """Imitates one library with exactly one name swapped out."""

    def __init__(self, lib, **swapped):
        self._lib, self._swapped = lib, swapped

    def __getattr__(self, key):
        return self._swapped.get(key) or getattr(self._lib, key)


def _flow_case(name):
    return dict(golden.cases_mod.flow_cases())["flow::" + name]


def test_flow_table_catches_a_severed_graph():
    """An operation returning a bare tensor — `roll` and `masked_select` really did this."""
    core = golden.load_borch()
    assert _flow_case("roll")(core).startswith("흐름")

    severed = _Shim(core, roll=lambda t, s, dims=None: core.tensor(
        core.roll(t, s, dims).numpy()))
    assert _flow_case("roll")(severed).startswith("안흐름")


def test_flow_table_catches_requires_grad_without_a_gradient():
    """**The worse of the two.** `requires_grad` is True and going backwards leaves `.grad` `None`.

    `.float()` was exactly this, and a check that asks about `requires_grad` alone lets it pass.
    """
    core = golden.load_borch()
    assert _flow_case("sqrt")(core) == "흐름/기울기있음"

    def lying_sqrt(t):
        # Attaches no parents and turns `requires_grad` on — the shape the old `.float()` had.
        return core.Tensor(core.sqrt(t).numpy(), requires_grad=True)

    assert _flow_case("sqrt")(_Shim(core, sqrt=lying_sqrt)) == "흐름/조용히None"
