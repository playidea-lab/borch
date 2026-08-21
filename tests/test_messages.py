"""**The error messages a user sees are in English.**

## Why this check exists

An error message is what an English reader meets **the first time something
breaks** in this library. After the documentation and the site had all become
English, 81% of the messages were still Korean, and that was the largest Korean
surface left (measured: 303 throwing sites across the three).

Fixing it once is not enough. Every new kernel and new layer brings one more
Korean message in with it, and nobody looks at that moment. So the rule is kept
as a check.

## What it does not block

- `repr` and `describe` — strings that print values, not errors

The three libraries are checked together. The playground runs JS and Python side
by side on one page, so English on one side alone shows **the same error in two
languages.**
"""

import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent
HANGUL = re.compile(r"[가-힣]")

# **Throwing is not only `raise`.** There are places that hand the wording to a
# helper which throws inside it (`_unsupported("a tensor exponent")`), and not
# looking at those at first made "all fixed" a false claim — the golden caught it
# instead. A place the check does not look at is a place with no rule.
#
# **And a helper's own defaults are user-facing too.** `_no_complex128` took
# `what="이 연산"` and interpolated it into an English sentence, so calling it
# with no argument produced a half-Korean message. The call sites were English
# and the check read call sites, so it saw nothing. The names live in one list
# now and both patterns are built from it — adding a helper covers both.
HELPERS = ("_unsupported", "_absent_here", "_absent_dtype", "_no_complex128")
_NAMES = "|".join(HELPERS)
_PY_RAISE = rf"(?:raise \w*(?:Error|Exception)|{_NAMES})\("
_PY_HELPER_DEF = rf"def (?:{_NAMES})\("

SURFACES = (
    ("borch-ts/src", "*.ts", re.compile(r"throw new \w*Error\(")),
    ("borch", "*.py", re.compile(_PY_RAISE)),
    ("borch_webgpu", "*.py", re.compile(_PY_RAISE)),
    ("borch", "*.py", re.compile(_PY_HELPER_DEF)),
    ("borch_webgpu", "*.py", re.compile(_PY_HELPER_DEF)),
)


def _sites(text, opener):
    """From the opener until the parentheses balance — a multi-line message
    counts as one site."""
    found = []
    for m in opener.finditer(text):
        start = m.end() - 1
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "(":
                depth += 1
            elif text[i] == ")":
                depth -= 1
                if depth == 0:
                    found.append((text[:start].count("\n") + 1, text[start:i + 1]))
                    break
    return found


def test_error_messages_are_english():
    bad = []
    for folder, glob, opener in SURFACES:
        for path in sorted((ROOT / folder).glob(glob)):
            for line, site in _sites(path.read_text(), opener):
                if HANGUL.search(site):
                    first = HANGUL.search(site)
                    snippet = site[max(0, first.start() - 30):first.start() + 40]
                    bad.append(f"{path.relative_to(ROOT)}:{line}  …{snippet.strip()}…")
    assert not bad, (
        f"{len(bad)} error messages are in Korean. This is the surface a user "
        "meets first, so it has to be English.\n  " + "\n  ".join(bad[:40])
        + (f"\n  … and {len(bad) - 40} more" if len(bad) > 40 else ""))
