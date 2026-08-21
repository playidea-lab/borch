"""**`why_failing.py` reads wording that other files write.**

That is a contract with nothing holding it together, and it had already come
apart. `ERRORS`'s second alternative matched the binding's Korean AttributeError
(``borch.ts 텐서에 `x` ``); translating those messages to English left it matching
nothing. The tool did not fail — the bucket fell through to a generic tail and it
went on printing counts, just without naming the cause. A parser that reads
another file's output breaks silently by construction.

Two halves, because either alone can pass while the pair is broken:

- the reader still classifies each shape (it is not falling through)
- the writers still emit the fragments the reader looks for

The first alone passes on invented strings nobody writes. The second alone
passes while the reader's own regex is wrong.
"""

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests" / "browser"))

import why_failing  # noqa: E402

# One line per shape the runner can produce, written as the emitters write them.
LINES = {
    "cat::x: max diff 3.20e-04": "the values diverged",
    "cat::x: shape (2, 3) vs (3, 2)": "the shapes diverged",
    "cat::x: expected torch.int64, got torch.float32": "the answers diverged (as strings)",
    "cat::x: 4 GPU validation errors (raised here, whatever the values)": None,
    "cat::x: AttributeError — borch.ts does not have `fooBar` (Python name `foo_bar`)":
        "absent from borch.ts: fooBar",
    "cat::x: AttributeError — `gradMode` is in borch.ts as a **module function**, "
    "not a method on tensors (Python name `grad_mode`).":
        "a module function, not bridged: gradMode",
    "cat::x: AttributeError — 'Tensor' object has no attribute 'baz'":
        "absent from the Python binding: Tensor.baz",
}

# What the reader looks for, and the file that has to still write it.
WRITTEN = (
    ("max diff", "tests/golden.py"),
    ("shape ", "tests/golden.py"),
    ("expected ", "tests/golden.py"),
    ("does not have `", "borch_webgpu/_ops.py"),
    ("is in borch.ts as a **module function**", "borch_webgpu/_ops.py"),
)


def test_the_reader_names_every_shape():
    wrong = []
    for line, want in LINES.items():
        if want is None:                      # no named bucket claimed for this one
            continue
        got = why_failing.bucket(line)
        if got != want:
            wrong.append(f"{line[:60]}\n      wanted {want!r}\n      got    {got!r}")
    assert not wrong, (
        "the failure reader no longer names these shapes. It does not fail when "
        "this happens — it falls through to a generic tail and keeps "
        "printing.\n  " + "\n  ".join(wrong))


def test_the_writers_still_write_what_the_reader_reads():
    missing = [f"{fragment!r} is not in {where}"
               for fragment, where in WRITTEN
               if fragment not in (ROOT / where).read_text()]
    assert not missing, (
        "the failure reader looks for these and nothing writes them any more. "
        "Either the wording moved — in which case move the reader with it — or "
        "this entry is stale.\n  " + "\n  ".join(missing))
