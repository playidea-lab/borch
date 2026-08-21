"""**The Python library carries no Korean.** Not in a message, not in a comment.

## Why the rule is this wide

It started narrower — "the error messages a user sees are in English" — and the
check looked for Korean inside `raise`. That check was evaded four times, each
time by a shape it had no reason to expect:

- a helper that wraps the wording (`_unsupported("a tensor exponent")`), so the
  Korean sits at a call site the regex does not open
- a helper's own default (`_no_complex128(what="이 연산")`), which is not a call
  site at all and interpolated straight into an English sentence
- a table of wording (`_ABSENT`'s values, `_bind_absent(_n, "희소 텐서")`), where
  the string never appears next to the helper's name
- `raise error(...)`, where the class arrived in a lowercase variable and
  `raise \w*Error\(` did not match

Each fix widened the list, and each time the list was the rule the next shape
walked around it. A fifth gap needed no cleverness at all: `borchvision.py` was
not in the list of files.

So the rule stopped being about messages. Every comment and docstring in these
files is English now, which makes "no Korean anywhere" both true and checkable,
and no new helper, table, alias or default can step around it.

## What is allowed

Golden case names. `tests/cases.py` names its cases in Korean and those names
are keys in the committed `tests/golden.json`, so a docstring that cites one
(`opt::StepLR/이어서 학습하기`) is quoting an identifier rather than writing
Korean prose. When the case names are translated this allowance goes with them,
and until then it is listed by name here so that the exception stays small and
visible.
"""

import json
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent
HANGUL = re.compile(r"[가-힣]")

# The Python library, all of it.
SOURCES = (
    ("borch", "*.py"),
    ("borch_webgpu", "*.py"),
    (".", "borchvision.py"),
)

# Golden case names cited from a docstring. Each is a key in `tests/golden.json`,
# and adding one here is a claim that it is. `test_the_quoted_case_names_exist`
# checks that claim against the file rather than against `tests/cases.py` — the
# names there are built with f-strings (`OPT_PREFIX + f"{name}/자취"`) and never
# appear as literals, so the source cannot answer the question and the golden can.
QUOTED_CASE_NAMES = (
    "repr::스칼라",
    "opt::StepLR/이어서 학습하기",
)


def _without_quoted_names(text):
    for name in QUOTED_CASE_NAMES:
        text = text.replace(name, "")
    return text


def _files():
    for folder, glob in SOURCES:
        yield from sorted((ROOT / folder).glob(glob))


def test_the_library_carries_no_korean():
    bad = []
    for path in _files():
        for number, line in enumerate(path.read_text().splitlines(), 1):
            if HANGUL.search(_without_quoted_names(line)):
                bad.append(f"{path.relative_to(ROOT)}:{number}  {line.strip()[:90]}")
    assert not bad, (
        f"{len(bad)} lines carry Korean. The Python library is English "
        "throughout — messages, comments and docstrings alike.\n  "
        + "\n  ".join(bad[:40])
        + (f"\n  … and {len(bad) - 40} more" if len(bad) > 40 else ""))


def test_the_quoted_case_names_exist():
    """An allowance that names something absent stops being an allowance.

    If a case is renamed and this list is not, the entry silently permits Korean
    that no longer quotes anything.

    A docstring may cite a case by its full name or by the tail alone, so a
    listed entry counts as present when it is a key or the end of one.
    """
    names = list(json.loads((ROOT / "tests" / "golden.json").read_text())["cases"])
    missing = [q for q in QUOTED_CASE_NAMES
               if not any(name == q or name.endswith(q) for name in names)]
    assert not missing, (
        "these are listed as quoted golden case names and are not keys in "
        f"tests/golden.json: {missing}. Either the case was renamed — in which "
        "case fix the docstring citing it — or the allowance is stale.")


def test_the_allowance_is_used():
    """Every entry earns its place, or it is dead permission."""
    text = "\n".join(p.read_text() for p in _files())
    unused = [name for name in QUOTED_CASE_NAMES if name not in text]
    assert not unused, (
        f"listed as allowed and cited nowhere: {unused}. Remove them — an "
        "allowance nothing uses is a hole waiting for something else.")
