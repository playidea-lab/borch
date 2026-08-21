"""**The error messages a user sees are English.**

## Why this check exists

An error message is what an English reader meets **the first time anything breaks** in this
library. Even after the documentation and the site were entirely English, 81% of the
messages were Korean, and that was the largest Korean surface left (measured: 303 throw
sites across the three).

Fixing it once is not enough. Every new kernel and new layer arrives carrying one more
Korean message, and nobody looks at that moment. So the rule is kept as a check.

## What it does not block

- `repr` and `describe` and other strings that print values — they are not errors.

Comments and docstrings are English now too, and that is a separate matter from this check —
this one is about the surface a user meets.

All three libraries are looked at together. The playground runs JS and Python side by side on
one page, so with only one of them in English **the same error appears in two languages.**
"""

import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent
HANGUL = re.compile(r"[가-힣]")

# **A throw site is not only a `raise`.** Some places hand the wording to a helper that
# throws inside it (`_unsupported("텐서 지수")`), and not looking at those at first made "it
# is all fixed" false — the golden cases caught it instead. A place a check does not look at
# is a place with no rule.
_PY_RAISE = r"(?:raise \w*(?:Error|Exception)|_unsupported|_absent_here|_absent_dtype)\("

SURFACES = (
    ("borch-ts/src", "*.ts", re.compile(r"throw new \w*Error\(")),
    ("borch", "*.py", re.compile(_PY_RAISE)),
    ("borch_webgpu", "*.py", re.compile(_PY_RAISE)),
)


def _sites(text, opener):
    """From the opener until the brackets balance — so a message spanning lines counts as one site."""
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
        f"{len(bad)} error messages are Korean. They are the surface a user meets first and have to be English"
        ".\n  " + "\n  ".join(bad[:40])
        + (f"\n  … and {len(bad) - 40} more" if len(bad) > 40 else ""))
