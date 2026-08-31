"""**The comparison has to be able to say what diverged, whatever kind of answer it is.**

`golden.py check` found a divergence in a boolean case and then died reporting it:

    bad.append(f"{name}: max diff {np.abs(want - got).max():.2e}")
    TypeError: numpy boolean subtract, the `-` operator, is not supported

The difference was real, the check had it, and the run ended with a traceback instead of
a case name. That is the one thing a comparison harness must not do, and it went years
unseen **because no boolean case had ever diverged** — the first to try was
`isclose(equal_nan)`, added the same day it was found.

The lesson is not about booleans. It is that the reporting path has a branch per kind of
answer, and **a branch is only ever walked by a failure**. The passing runs that made up
every day of this repository's life exercised none of them.

## What this does

It takes the frozen answers and doctors one, per kind — a float, an integer, a boolean, a
string, and one where the shape moves — then runs `check` against the doctored copy and
asks for two things: that it did not raise, and that the divergence it lists **names the
case that was doctored**. Nothing about the library is being tested; the subject is the
check itself.

Doctoring the *frozen* side rather than the library is what makes this cheap and exact.
A wrong value has to be reported the same way whichever side is wrong, and this side can
be written to.
"""

import importlib.util
import pathlib

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
_here = pathlib.Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("bt_golden_kinds", _here / "golden.py")
golden = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(golden)

PREFIX = golden._PREFIX


def _frozen():
    if not golden.DEFAULT_PATH.exists():
        pytest.skip(f"no golden answers at {golden.DEFAULT_PATH} — run dump first")
    return dict(np.load(golden.DEFAULT_PATH, allow_pickle=False))


def _kind_of(arr):
    if arr.dtype.kind == "U":
        return "string"
    if arr.dtype.kind == "b":
        return "boolean"
    if arr.dtype.kind in "iu":
        return "integer"
    if arr.dtype.kind in "fc":
        return "float"
    return None


def _one_per_kind(z):
    """One case name per kind of answer, and the kinds it could not find.

    Sorted so the pick is the same from run to run — an arbitrary case is fine, an
    *unrepeatable* one is not, because a failure here has to be reproducible.
    """
    picked = {}
    for key in sorted(z):
        if not key.startswith(PREFIX):
            continue
        kind = _kind_of(z[key])
        if not kind or kind in picked:
            continue
        # **A string answer is 0-d and has to be allowed to be.** Requiring more than
        # one element skipped the whole string branch — the very shape of hole this
        # file exists to find, in the file that exists to find it. The numeric kinds
        # keep the rule: a scalar is a poor subject for the shape branch.
        if kind != "string" and z[key].size < 2:
            continue
        picked[kind] = key[len(PREFIX):]
    return picked


def _doctored(z, name, how, tmp_path):
    """The frozen answers with one case's answer changed, written to a new file."""
    changed = dict(z)
    key = PREFIX + name
    was = changed[key]
    if how == "value":
        if was.dtype.kind == "U":
            changed[key] = np.array(str(was) + " (doctored)")
        elif was.dtype.kind == "b":
            changed[key] = ~was
        else:
            changed[key] = was + np.asarray(7, dtype=was.dtype)
    else:                                                   # how == "shape"
        changed[key] = was.reshape(-1)[:1]
    out = tmp_path / "doctored.npz"
    np.savez(out, **changed)
    return out


@pytest.mark.parametrize("kind", ["float", "integer", "boolean", "string"])
def test_a_doctored_answer_is_reported_by_name(kind, tmp_path):
    """**A wrong value of every kind has to come back naming its case.**

    Not "the run fails" — `check` returning a list at all is what the boolean branch
    could not do. The assertion is that the case's own name is in what it says, because
    that is what a person reads at three in the morning.
    """
    z = _frozen()
    picked = _one_per_kind(z)
    if kind not in picked:
        pytest.skip(f"the golden holds no {kind} answer of more than one element")
    name = picked[kind]
    path = _doctored(z, name, "value", tmp_path)

    bad, total = golden.check(golden.load_borch(), path)
    assert total > 0
    assert any(name in line for line in bad), (
        f"a doctored {kind} answer for `{name}` was not reported.\n"
        f"  what came back: {bad[:3] or '(nothing — the check passed)'}\n"
        "  A comparison that misses a wrong value of one kind is not comparing that "
        "kind.")


def test_a_doctored_shape_is_reported_by_name(tmp_path):
    """The shape branch, which is a different line from the value branch and is
    reached before it."""
    z = _frozen()
    picked = _one_per_kind(z)
    name = picked.get("float") or next(iter(picked.values()))
    path = _doctored(z, name, "shape", tmp_path)

    bad, _ = golden.check(golden.load_borch(), path)
    assert any(name in line and "shape" in line for line in bad), (
        f"a doctored shape for `{name}` was not reported as a shape difference.\n"
        f"  what came back: {bad[:3] or '(nothing — the check passed)'}")


def test_an_untouched_copy_still_passes(tmp_path):
    """**The control.** Every assertion above is about a doctored file, and a check
    that reported everything as diverged would satisfy all of them. This is the row
    that says the reporting is answering the question rather than shouting."""
    z = _frozen()
    out = tmp_path / "untouched.npz"
    np.savez(out, **z)
    bad, total = golden.check(golden.load_borch(), out)
    assert not bad, (
        "an untouched copy of the frozen answers reported divergences:\n  "
        + "\n  ".join(bad[:5]))
    assert total > 0
