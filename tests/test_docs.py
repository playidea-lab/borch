"""Checks that the **counts** written in the documentation match reality.

Every time the golden cases grow, the numbers in the documentation go stale. This
repository has already caught that three times (`b00e693` the golden count, `b3d7453`
three figures, `e41c043` one of which broke installation) and all three times a person
found and fixed it by eye. **A method that has failed three times is the problem.**

Exactly one thing is asked here: does the case count **the README** states equal what the
table actually holds.

**Design documents are not read.** Reading everything caught ten places at first, and
seven of them were not stale but **a record of that moment** — `BORCH-TS.md`'s "relu
passed all 798 golden cases as it stood" is right at 798. Changing it to 845 is not
fixing a stale number but forging history, which is worse than a stale number.
`WEBGPU-DESIGN.md`'s "golden 141/141" is where stage S3 stood at the time.

So the line is drawn by **kind of document.** The README is a place that speaks of now and
has to be current; design and history documents speak of then and must not be touched. A
first attempt at separating tense with a regular expression could not make that
distinction.

**The explainer pages (`site/`) speak of now too.** They are the first screen anyone else
sees, so they can afford staleness even less than the README — and unread here, nobody
reads them: the site's wording is outside the field of view of whoever adds golden cases.
It is **the same shape of failure** this repository has had three times, so the same net
goes over it. There is an English edition, so the `N golden cases` form is caught too.
"""

import pathlib
import re

import cases as cases_mod

ROOT = pathlib.Path(__file__).resolve().parent.parent
# Places that state a case count. It looks for shapes such as `골든 845건` and `골든 1020/1020`.
#
# **Pinned at three digits, it went quiet the moment the table passed a thousand.** A check
# that catches nothing looks exactly like a check that passes, and that is worse than a
# stale number — a stale number is noticed eventually; a check that does not run is not.
# The digit range is left generous.
COUNT = re.compile(
    r"골든\s*\*{0,2}(\d{3,5})\s*(?:건|/\s*\d{3,5})"
    # The site is English by default. Markup such as `<strong>` is let through on either side.
    r"|(\d{3,5})\s*golden\s*cases")
# **Some places do not put `골든` in front.** "53 건을 빼고 2709 건을 본다" is one, and the
# net above required the prefix so it slipped through entirely — while the golden cases went
# from 2263 to 2953, that one line kept its old number. A site session found it by eye.
# **A derived number is a number.** The rule that it must be one of the three values applies
# to it identically.
DERIVED = re.compile(r"(\d{3,5})\s*건을\s*본다"
                     r"|(?:looks at|examines|covers)\s*\*{0,2}(\d{3,5})\s*cases")

# **Places that speak of now.** Design and history documents are not here — see the docstring above.
LIVE_DOCS = ("README.md", "site/index.html", "site/ko/index.html")


def _hit(found):
    """Pulls the number out of one thing `findall` returned.

    **Several nets mean several shapes.** A regular expression with one group gives a
    string; one with alternatives gives **a tuple with blanks in it.** Using both in one loop
    means reconciling here — unreconciled, the tuple goes into the comparison as it stands
    and **equals no number at all**, which either shouts that every number is stale (if you
    are lucky) or passes quietly (if you are not).
    """
    return found if isinstance(found, str) else next((g for g in found if g), "")


def _counts():
    """(the whole, what the core sees, what the binding sees).

    **There are three because the scope split in both directions.** What is sister-only (1-D
    and 3-D convolution, which the core deliberately refuses) is skipped by the core, and
    what is core-only (complex) is skipped by the binding. Counting only one of them makes
    the other half look like a missing implementation.
    """
    names = [n for n, _ in cases_mod.golden_cases(cases_mod.golden_inputs())]
    core = [n for n in names if not n.startswith(cases_mod.WEBGPU_PREFIX)]
    bind = [n for n in names if not n.startswith(cases_mod.CORE_ONLY_PREFIXES)]
    return len(names), len(core), len(bind)


def test_docs_do_not_name_a_stale_golden_count():
    """The case count the documentation states has to be **the count that exists now.**

    Three numbers are actually in use — the whole table, the core's count with sister-only
    removed, and the binding's with core-only removed. A number in a `골든 N건` position that
    is none of those three is stale.

    **Accepting all three loosens the net.** On the day complex went in, the whole grew from
    2263 to 2287 while the binding's count became exactly 2263, and one stale sentence very
    nearly passed carrying a number that happened to be right. The number was right and the
    sentence said "it passes all of them", which was no longer true — a check that counts
    numbers does not read sentences, and that is written down here.
    """
    total, core, bind = _counts()
    allowed = {str(total), str(core), str(bind)}
    stale = []
    for rel in LIVE_DOCS:
        path = ROOT / rel
        # The site may be absent (depending on the checkout). Absence is not made a failure.
        if not path.exists():
            continue
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            for found in COUNT.findall(line) + DERIVED.findall(line):
                hit = _hit(found)
                if hit not in allowed:
                    stale.append(
                        f"{rel}:{i}  '{hit}' — it is now {total} (whole) / "
                        f"{core} (core) / {bind} (binding)")
    assert not stale, (
        "the golden counts in documents that speak of now are stale:\n  " + "\n  ".join(stale) +
        "\n\nThe README and `site/` speak of now. A sentence that has to speak of then should "
        "be written without the number, or moved into a design document — changing a past "
        "number to the current one is not fixing a stale number but forging history.")


# Places stating a package's size, such as `2,358 줄`. Caught with or without thousands separators.
LINES = re.compile(r"\*{0,2}(\d{1,3}(?:,\d{3})*)\s*줄\*{0,2}"
                   r"|\*{0,2}(\d{1,3}(?:,\d{3})*)[- ]line[s]?\*{0,2}")
# What is counted. Only packages present in the repository now — a deleted one's size belongs to history.
PACKAGES = {"borch", "borch_webgpu"}

# Where a line count is looked for, and where one has to actually be found.
#
# The two are not the same list, and asserting on absence is what made the
# difference visible. `borch/__init__.py` is scanned so that a count added
# there later is checked, and it has never carried one — `git log -S` finds no
# line count in its whole history. Leaving it in the must-yield list would be
# demanding a claim that was never made.
LINE_DOCS = ("README.md", "borch_webgpu/__init__.py", "borch/__init__.py")

# How many claims each document makes, so that losing one is visible. See
# `test_the_documents_still_make_the_claims_this_file_checks` for why a count
# rather than a presence check.
CLAIMS = {
    ("README.md", "golden"): 4,
    ("site/index.html", "golden"): 1,
    ("site/ko/index.html", "golden"): 1,
    ("README.md", "lines"): 4,
    ("borch_webgpu/__init__.py", "lines"): 2,
}


def _package_lines():
    """package name → line count."""
    out = {}
    for name in PACKAGES:
        total = 0
        for path in sorted((ROOT / name).glob("*.py")):
            total += len(path.read_text(encoding="utf-8").splitlines())
        out[name] = total
    return out


# **Numbers from history.** They cannot be verified by counting now, so they are written
# here. Without saying what each one is, they are magic numbers to the next person.
HISTORICAL = {
    "5,307": "the TF.js borch_webgpu, deleted in 45be321",
    "3,300": "borch's size when it was split from one file into a package (8177e1d)",
}

# **How far a drift is tolerated.** What is being caught is a 2.6× error, not three lines.
# Demanding an exact number breaks the documentation every time a line of `_ops.py` changes,
# and then this check teaches people how to make it pass instead of guarding what it guards.
TOLERANCE = 0.05


def test_docs_do_not_name_a_stale_line_count():
    """The **line counts** the documentation states must not be far from reality.

    This machinery went onto the golden count and not onto the line counts, and that is
    exactly where it went wrong twice. Once it counted five of eight files and wrote 5,307 as
    **2,312**; once it carried across an estimate of **900** from before `_data.py` arrived,
    without recounting. Side by side those two made the sentence "2,312 lines became 900
    lines", when in fact 5,307 became 2,361 — the direction was right and the magnitude was
    off by 2.6×.

    Both went into a commit message and the README at the same time. **Counting by eye is not
    a method.**

    The boundary is the golden count's. The README and the package docstrings speak of now
    and have to be current; the design documents (`BORCH-TS.md`, `WEBGPU-DESIGN.md`) record
    then and are not read.
    """
    sizes = _package_lines()
    stale = []
    for rel in LINE_DOCS:
        path = ROOT / rel
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            for found in LINES.findall(line):
                hit = _hit(found)
                if not hit or hit in HISTORICAL:
                    continue
                said = int(hit.replace(",", ""))
                # Catching the small numbers that appear inside sentences turns this to noise.
                if said < 100:
                    continue
                if any(abs(said - real) <= real * TOLERANCE for real in sizes.values()):
                    continue
                stale.append(f"{rel}:{i}  '{hit} 줄' — it is now " +
                             ", ".join(f"{k} {v:,}" for k, v in sorted(sizes.items())))
    assert not stale, (
        "the line counts in the documentation differ from reality:\n  " + "\n  ".join(stale) +
        "\n\nCount and fix. If it is a number about then, write it into `HISTORICAL` along "
        "with what it is — changing history to the current number is not fixing a stale number.")


def test_the_documents_still_make_the_claims_this_file_checks():
    """**Absence has to assert.**

    Everything above reads the documentation's own phrasing, which makes the
    patterns a pair with the prose — and the failure is not symmetric. A number
    that moves is caught. Phrasing that moves leaves the pattern matching
    nothing, and finding no claims reads as "there is nothing to verify" rather
    than as an error. The suite goes green having checked nothing.

    That is not hypothetical. `borch_webgpu/__init__.py` said "7,036 줄" and
    `LINES` could see it; translating that docstring to "7,036 lines" made the
    pattern blind to it, and the number then sat stale through a translation pass
    that fixed the two in the README precisely because those two were still
    visible. Widening the pattern found it. This test is what would have said so
    without the widening.

    **"At least one" is not enough**, and writing it that way first showed why:
    breaking one phrasing in the README left four other counts matching and the
    check passed. Total blindness is not the failure that happens — partial is.
    So the number of claims per document is recorded, and losing one is caught.

    That makes adding a sentence with a count fail until the number here is
    updated. That is the intended cost: it is one line, and it makes somebody
    look at what was added.
    """
    lost = []
    for (rel, kind), expected in sorted(CLAIMS.items()):
        path = ROOT / rel
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        found = ([f for f in COUNT.findall(text) + DERIVED.findall(text)]
                 if kind == "golden" else LINES.findall(text))
        got = len([f for f in found if _hit(f)])
        if got != expected:
            lost.append(f"{rel} — {expected} {kind} claims recorded, {got} parsed")

    assert not lost, (
        "the number of claims parsed out of these documents changed, so the "
        "checks above are no longer looking at what they were.\n  "
        + "\n  ".join(lost) +
        "\n\nFewer means a pattern went blind — the phrasing moved and the "
        "pattern has to move with it. More means a new claim, which is fine: "
        "update the number here, having looked at what was added.")
