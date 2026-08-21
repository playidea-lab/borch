"""Pins the wording contract **`why_failing.py` cannot check for itself.**

`why_failing.bucket()` reduces a failure to one phrase by grepping text two other files
write: `golden.py` produces the `max diff` / `shape … vs …` / `expected …, got …` lines, and
`borch_webgpu` raises the `does not have` messages. Neither file knows it is being read.

**This is why that goes wrong quietly.** When a producer's wording moves, `bucket()` does not
fail — it falls through to its last line and returns the first forty characters of the text.
The grouping still prints, still looks like a grouping, and every failure lands in its own
one-member group. A check that catches nothing looks exactly like a check that passes.

It already happened. `borch_webgpu`'s messages were translated to English while the regex
here still matched the Korean, so the bucket that says "not in borch.ts" — the most useful
one while laying a new implementation on — had been dead for as long as nobody ran it.

So the strings are not written out here by hand. **They are produced.** `golden.py`'s
comparison is run against a deliberately wrong library and the lines it actually emits go
through `bucket()`; the binding's messages are pulled out of `borch_webgpu`'s source. Written
by hand, this file would drift the same way and pass while doing so.
"""

import importlib.util
import pathlib
import re
import sys

import numpy as np
import pytest

_here = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_here / "browser"))
sys.path.insert(0, str(_here))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


golden = _load("bt_golden_wf", _here / "golden.py")
why = _load("bt_why", _here / "browser" / "why_failing.py")


def _lines(tmp_path, corrupt):
    """The lines `golden.check` really writes when the **stored answers** are tampered with.

    The tampering goes on the golden file rather than on the library, because every route
    through the library has to survive `to_numpy` — handing back a plain ndarray makes it call
    `.detach()` and every case turns into an AttributeError instead of a comparison. Moving the
    wrongness to the expected side leaves the comparison itself exactly as it runs in earnest,
    which is the only version of it worth pinning.
    """
    path = tmp_path / "golden.npz"
    golden.dump(path)
    stored = dict(np.load(path, allow_pickle=False))
    tampered = tmp_path / "tampered.npz"
    np.savez(tampered, **{k: (corrupt(v) if k.startswith("case::") else v)
                          for k, v in stored.items()})
    bad, _ = golden.check(golden.load_borch(), tampered)
    return bad


def test_the_value_and_shape_lines_land_in_their_buckets(tmp_path):
    """A wrong number has to become "the values diverged", a wrong shape "the shapes diverged".

    Both come out of one run because a scalar case cannot have its shape changed and a
    shaped case can — so the corruption adds an axis where it can and adds a constant where
    it cannot, and both lines appear in the same list.
    """
    def corrupt(arr):
        if arr.dtype.kind in "UO":
            return arr
        if arr.ndim:
            return np.concatenate([arr, arr], axis=0)
        return arr + 1.0

    buckets = {why.bucket(line) for line in _lines(tmp_path, corrupt)}
    assert "the shapes diverged" in buckets, sorted(buckets)[:8]
    assert "the values diverged" in buckets, sorted(buckets)[:8]


def test_the_string_line_lands_in_its_bucket(tmp_path):
    """A dtype case answers with a type name, and a wrong one has to become its own bucket.

    Kept apart from the test above because the string branch of `golden.check` is a separate
    path — folded in, a change to it would hide behind the other two passing.
    """
    def corrupt(arr):
        return np.array("not-a-dtype") if arr.dtype.kind == "U" else arr

    buckets = {why.bucket(line) for line in _lines(tmp_path, corrupt)}
    assert "the answers diverged (string)" in buckets, sorted(buckets)[:8]


# The three shapes `borch_webgpu` raises when a name is not on the other side. Read out of
# the source rather than written down, so that adding a fourth shape turns this red.
_RAISED = re.compile(r'f"(?:the )?borch\.ts[^"]*do(?:es)? not have[^"]*"')


def _binding_messages():
    out = []
    for path in sorted((_here.parent / "borch_webgpu").glob("*.py")):
        for hit in _RAISED.findall(path.read_text(encoding="utf-8")):
            # The f-string's placeholders are filled with a name that has a backtick pair
            # around it, exactly as the real message does.
            out.append((path.name, re.sub(r"\{[^}]+\}", "sigmoid", hit[2:-1])))
    return out


def test_every_message_the_binding_raises_is_recognised():
    """**The bucket that matters most while porting** has to survive the binding's wording.

    "not in borch.ts" is what turns a list of failures into "implement this one name and N
    cases open". It is also the one that died quietly, because nothing connected the regex
    here to the file that writes the text.
    """
    messages = _binding_messages()
    assert messages, "no `does not have` message was found in borch_webgpu — this check is spinning"
    missed = [(where, msg) for where, msg in messages
              if not why.bucket(f"case::x: AttributeError — {msg}").startswith("not in borch.ts")]
    assert not missed, (
        "messages the binding raises that `why_failing` does not recognise:\n  "
        + "\n  ".join(f"{where}: {msg}" for where, msg in missed)
        + "\n\nWiden `ERRORS` in tests/browser/why_failing.py, or bring the wording back.")


def test_an_unrecognised_reason_does_not_masquerade_as_a_bucket():
    """**What the failure mode looks like**, pinned so the next reader can see it.

    When wording moves, `bucket()` returns a slice of the raw text. That is not an error and
    not empty — it is a plausible-looking label, which is why nobody noticed for as long as
    nobody ran it. Written down here so the shape is recognisable when it happens again.
    """
    stale = "case::foo: 최대차 1.2e-03"
    assert why.bucket(stale) == "최대차 1.2e-03"[:40]
    assert why.bucket("case::foo: max diff 1.2e-03") == "the values diverged"


@pytest.mark.parametrize("line, want", [
    ("case::x: AttributeError — 'Tensor' object has no attribute 'half'",
     "not in the Python binding: Tensor.half"),
    ("case::x: JsException — something", "borch.ts threw"),
    ("case::x: TypeError — takes 2 positional arguments", "a Python-side type does not fit"),
])
def test_the_remaining_buckets(line, want):
    """The three that do not come from `golden.py`'s comparison lines.

    Their wording belongs to Python and to Pyodide, not to this repository, so it is written
    here by hand — there is no file of ours to read it out of.
    """
    assert why.bucket(line) == want
