"""**The Korean left in `borch-ts` may shrink and may not grow.**

The rest of the repository is English: the Python library, every root document, the
configuration, the workflows, the site's own pages. What remains is `borch-ts`, and it is
under another session's hand while features land in it daily.

## Why a ceiling rather than a rule

The obvious check — "no Korean in `borch-ts`" — is the right rule and it cannot land yet:
red on arrival, it would be skipped rather than obeyed, which is worse than absent because
it teaches people to switch checks off. So the rule that *can* land today is the one that
stops the loss getting larger.

**The measurement that argued for it.** Over thirty hours `borch-ts/src` went from 40,698
Korean characters to 45,480 and `borch-ts/test` from 44,491 to 52,721, across fifteen
commits of which one was a translation. Nothing was going wrong — features were landing,
and each arrived with Korean comments, because that is what the surrounding file looks
like. Waiting was not holding position; it was losing ground at about 11% a day.

## What a green run here does **not** mean

It does not mean a directory is English. It does not mean a file is. **The ceiling is a
derivative** — it answers "did this grow" and is silent about every absolute fact, and a
green run is compatible with 40,000 Korean characters sitting exactly where they were.

This is written down because it already misled somebody. A session translated the
characters its own commit had added, ran this, saw green, and reported "vision.ts is
translated" — the check answered *did this grow* and the sentence claimed *is this
English*. The file still held 2,883 Korean characters, and the report was believed
downstream until somebody grepped.

Having a green test in front of you is what makes it easy to stop looking. To claim a
file is English, count it:

    grep -c "[가-힣]" path/to/file

## What it costs

Nothing to read and nothing to run. It asks only that **new comments in these directories
be written in English**, which is the direction the repository has already taken
everywhere else. Translating an existing block lowers the number, and lowering it is
always allowed.

## When a number moves

Lower it. The ceilings below are a record of a debt, not a budget to spend: after a
translation pass, set them to what was measured and the ratchet holds the new floor. The
failure message prints the number to write.

If a genuinely new Korean string has to go in — a case name, a fixture, something quoted
from a Korean page — raise the ceiling **in the same commit**, with the reason in the
commit message. That makes it a decision somebody made rather than a number that drifted.
"""

import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent
HANGUL = re.compile(r"[가-힣]")
SUFFIXES = (".ts", ".py", ".html")

# Measured 2026-08-22. Lower these after a translation pass; see the module docstring
# before raising one.
CEILINGS = {
    "borch-ts/src": 18,          # the rest of src; what is left is quoted golden names
    "borch-ts/test": 27201,
}


def _count(folder):
    total, per_file = 0, {}
    for path in sorted((ROOT / folder).rglob("*")):
        if not path.is_file() or path.suffix not in SUFFIXES:
            continue
        if "dist" in path.parts or "node_modules" in path.parts:
            continue
        found = len(HANGUL.findall(path.read_text(errors="ignore")))
        if found:
            per_file[str(path.relative_to(ROOT))] = found
            total += found
    return total, per_file


def test_the_korean_left_in_borch_ts_does_not_grow():
    """The ceiling, per directory, with the worst files named when it is breached."""
    over = []
    for folder, ceiling in CEILINGS.items():
        total, per_file = _count(folder)
        if total > ceiling:
            worst = sorted(per_file.items(), key=lambda kv: -kv[1])[:5]
            over.append(
                f"{folder}: {total} Korean characters against a ceiling of {ceiling} "
                f"(+{total - ceiling})\n    "
                + "\n    ".join(f"{n}  {c}" for n, c in worst))
    assert not over, (
        "Korean grew in a directory that is being translated:\n  " + "\n  ".join(over)
        + "\n\n  New comments in borch-ts go in English — everything else in this "
          "repository already does.\n  Raising a ceiling is allowed when a Korean string "
          "genuinely has to go in (a case\n  name, a quoted fixture); do it in the same "
          "commit and say why.")


def test_the_ceilings_name_directories_that_exist():
    """A ceiling over a directory that moved is a budget nobody is spending.

    It would sit at zero, pass forever, and read as a directory under control.
    """
    missing = [folder for folder in CEILINGS if not (ROOT / folder).is_dir()]
    assert not missing, (
        f"these ceilings name directories that are not there: {missing}. The code moved — "
        "point the ceiling at where it went, or drop the row if the Korean is gone.")


def test_a_ceiling_that_is_far_too_high_is_tightened():
    """**A ratchet nobody tightens is a ratchet that stopped working.**

    After a translation pass the count drops, the ceiling stays where it was, and the
    headroom left behind quietly permits new Korean back up to the old number — the pass
    is undone over the following weeks and every commit doing it is green.

    So it fails, in the same commit that earned the drop, and prints the line to paste.
    Tightening is copying a number, not taking a measurement.
    """
    slack = {}
    for folder, ceiling in CEILINGS.items():
        total, _ = _count(folder)
        if total and ceiling - total > ceiling * 0.1:
            slack[folder] = (total, ceiling)
    assert not slack, (
        "these ceilings are more than 10% above what is actually there — tighten them:\n  "
        + "\n  ".join(f'"{f}": {t},   # was {c}' for f, (t, c) in slack.items()))
