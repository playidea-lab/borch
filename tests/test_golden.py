"""Checks that the golden harness itself does its job.

There is no GPU backend yet. So borch is compared against the golden **as though
it were a third library** — that shows whether the harness runs, and a backend
goes into the same slot when there is one.

And "does it run" is not enough to ask. There are two ways a golden harness
breaks quietly — comparing against a stale golden after the table changed, or
comparing with the inputs diverged. Both **produce a pass having compared
nothing.** So both are built on purpose and checked for.
"""

import importlib.util
import pathlib

import pytest

_here = pathlib.Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("bt_golden", _here / "golden.py")
golden = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(golden)


def test_golden_dump_then_check_matches_borch(tmp_path):
    """The core compares **its own range** alone.

    The golden is pinned with real torch, so it holds things that exist on the
    sister library alone. The core refuses those on purpose and skips them — the
    two libraries' ranges have begun parting, and the skipped count matching the
    sister-only case count exactly is what says that parting is the intended
    one.
    """
    path = tmp_path / "golden.npz"
    count, _ = golden.dump(path)
    assert count > 0, "the golden is empty — the case table did not load"

    bad, total = golden.check(golden.load_borch(), path)
    assert total == count - len(golden.cases_mod.webgpu_cases())
    assert not bad, "diverged from the golden:\n  " + "\n  ".join(bad)


def test_check_rejects_stale_golden(tmp_path, monkeypatch):
    """A stale golden after the table changed **must not produce a pass.**"""
    path = tmp_path / "golden.npz"
    golden.dump(path)
    monkeypatch.setattr(golden.cases_mod, "manifest_hash",
                        lambda cases: "the-hash-after-the-table-changed")
    with pytest.raises(SystemExit, match="stale"):
        golden.check(golden.load_borch(), path)


def test_check_rejects_mismatched_inputs(tmp_path, monkeypatch):
    """Diverged inputs have to stop it.

    numpy's `default_rng` is meant to give the same numbers across versions, and
    without a check on that promise, a broken one means **comparing different
    inputs and stamping it a pass.**
    """
    path = tmp_path / "golden.npz"
    golden.dump(path)
    monkeypatch.setattr(golden.cases_mod, "input_fingerprint",
                        lambda inp: "a-different-input-fingerprint")
    with pytest.raises(SystemExit, match="the inputs differ"):
        golden.check(golden.load_borch(), path)


def test_check_names_the_case_that_raised_a_gpu_fault(tmp_path):
    """**It names the case that raised a GPU validation error.**

    WebGPU does not throw those as exceptions. An invalid command buffer quietly
    does nothing, so **the culprit passes and a case queued behind it turns red
    instead** — that happened three times (`as_strided_`'s over-copy, the
    optimiser state's shared buffer, and an `index_select` that selects nothing).

    Only the wiring is checked here, without a browser. Even with a fake counter
    doing the counting, the answer has to record **which case it grew at** —
    without that it is one more number printed by the runner, and that still
    leaves you looking one place away from the cause.
    """
    path = tmp_path / "golden.npz"
    golden.dump(path)

    # Pretend one error occurred, at the third case exactly.
    state = {"n": 0, "seen": 0}

    def counter():
        state["seen"] += 1
        if state["seen"] == 3:
            state["n"] += 1
        return state["n"]

    bad, _ = golden.check(golden.load_borch(), path, faults=counter)
    hits = [line for line in bad if "GPU validation errors" in line]
    assert len(hits) == 1, (f"expected one, got {len(hits)}:\n  "
                            + "\n  ".join(bad))

    # **Which case it is, is the point.** The first call is outside the loop (the
    # baseline), so the third call is where the table's second case ended.
    names = [n for n, _ in golden.cases_mod.golden_cases()]
    assert hits[0].startswith(names[1] + ":"), (
        f"it should name {names[1]}: {hits[0]}")


# ---- whether the gradient-flow table does its job
#
# This table asks whether the gradient flows rather than what the value is. And
# **asking without catching** adds one more green while doing nothing. So the two
# shapes that cut quietly are built on purpose and checked for — both really did
# turn up on the sister library.
#
# The strings it compares (`흐름`, `안흐름`, `흐름/기울기있음`,
# `흐름/조용히None`) come from `cases.py:7660` and are **expected values inside
# the committed golden.json.** Translating them would need a re-dump against real
# torch across all 2991 cases, which is the same landmine as the case names. They
# stay Korean until that is done deliberately, in one pass.

class _Shim:
    """Imitate a library with exactly one name swapped out."""

    def __init__(self, lib, **swapped):
        self._lib, self._swapped = lib, swapped

    def __getattr__(self, key):
        return self._swapped.get(key) or getattr(self._lib, key)


def _flow_case(name):
    return dict(golden.cases_mod.flow_cases())["flow::" + name]


def test_flow_table_catches_a_severed_graph():
    """An operation handing back a bare tensor — `roll` and `masked_select`
    really did."""
    core = golden.load_borch()
    assert _flow_case("roll")(core).startswith("흐름")

    severed = _Shim(core, roll=lambda t, s, dims=None: core.tensor(
        core.roll(t, s, dims).numpy()))
    assert _flow_case("roll")(severed).startswith("안흐름")


def test_flow_table_catches_requires_grad_without_a_gradient():
    """**The worse of the two.** `requires_grad` is True and tracing back leaves
    `.grad` as `None`.

    `.float()` was exactly this, and a check asking about `requires_grad` alone
    lets it pass.
    """
    core = golden.load_borch()
    assert _flow_case("sqrt")(core) == "흐름/기울기있음"

    def lying_sqrt(t):
        # No parents attached and requires_grad turned on — the old `.float()`'s
        # shape exactly.
        return core.Tensor(core.sqrt(t).numpy(), requires_grad=True)

    assert _flow_case("sqrt")(_Shim(core, sqrt=lying_sqrt)) == "흐름/조용히None"
