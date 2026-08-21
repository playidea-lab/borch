"""Checks that the **counts** written in the documentation match reality.

Every time the golden grows, a number in the documentation goes stale. This
repository has caught that three times already (`b00e693` the golden count,
`b3d7453` three figures, `e41c043` one of which broke the install), and all three
times a person found it by eye. **A way of failing three times is the way's
problem.**

Exactly one thing is asked here: does the case count **the README** names match
what the table actually holds.

**The design documents are not examined.** Pointing this at all of them caught ten
places, and seven of those were not stale — they were **a record of the time.**
`BORCH-TS.md`'s "relu passed 798 golden cases unchanged" is right at 798. Changing
it to 845 is not fixing a stale number but forging history, and that is worse than
a stale number. `WEBGPU-DESIGN.md`'s "golden 141/141" is likewise where stage S3
reached at the time.

So the boundary is drawn by **kind of document.** The README speaks in the present
and has to stay current; the design and history documents speak of a time and must
not be touched. The first attempt, splitting them by tense with a regex, could not
tell the difference.

**The explanatory pages (`site/`) speak in the present too.** They are the first
screen anybody outside sees, so they must go stale even less than the README, and
nothing looks at them if this does not — the site's wording sits outside the view
of whoever grows the golden. It is **the same way of failing** this repository has
been through three times, so the same net goes over it.

## The parsing has to speak when it finds nothing

These patterns read the documentation's own phrasing. That makes them a pair with
the prose, and the failure mode is not symmetric: a **number** that moves is
caught, and **phrasing** that moves leaves the pattern matching nothing at all —
and finding no claims reads as "there is nothing to verify" rather than as an
error. The suite goes green having checked nothing, which is the shape this whole
file exists to stop.

So each pattern carries the English and the Korean phrasing, and
`test_the_documents_still_make_the_claims_this_file_checks` asserts that every
live document still yields at least one parsed count. Absence has to assert.
"""

import pathlib
import re

import cases as cases_mod

ROOT = pathlib.Path(__file__).resolve().parent.parent
# The places naming a case count. It looks for shapes like `골든 845건` and
# `golden 1020/1020`.
#
# **Pinned at three digits, it went quiet the moment the table passed 1000.** A
# check that catches nothing looks like a check that passes, and that is worse
# than a stale number — a stale number is noticed eventually and a check that does
# not run is not. The digit range is left generous.
COUNT = re.compile(
    r"골든\s*\*{0,2}(\d{3,5})\s*(?:건|/\s*\d{3,5})"
    # The site is English by default. Markup like `<strong>` is let through on
    # either side.
    r"|(\d{3,5})\s*golden\s*cases"
    r"|golden\s*\*{0,2}(\d{3,5})\s*/\s*\d{3,5}")
# **Some places do not carry `골든` in front.** "53 건을 빼고 2709 건을 본다" is
# one, and the net above demands the prefix, so it slipped through entirely — that
# one line kept an old number while the golden went from 2263 to 2953. The site
# session found it by eye. **A derived number is a number.** The rule that it has
# to be one of the three values applies to it just the same.
DERIVED = re.compile(r"(\d{3,5})\s*건을\s*본다"
                     r"|(?:looks at|examines|covers)\s*\*{0,2}(\d{3,5})\s*cases")

# **The places that speak in the present.** The design and history documents are
# not here — see the docstring above.
LIVE_DOCS = ("README.md", "site/index.html", "site/ko/index.html")


def _hit(found):
    """Take the number out of one thing `findall` produced.

    **Several nets means several shapes.** A regex with one group gives a string
    and one split into alternatives gives **a tuple with empty slots in it.** Using
    both in one loop means reconciling them here — unreconciled, the tuple goes
    into the comparison as it is and **equals no number at all**, and then it
    either shouts that every number is stale (if you are lucky) or passes quietly
    (if you are not).
    """
    return found if isinstance(found, str) else next((g for g in found if g), "")


def _counts():
    """(total, what the core sees, what the binding sees).

    **There are three because the range parted both ways.** The sister-only ones
    (1-D and 3-D convolutions and the like, which the core refuses on purpose) are
    skipped by the core, and the core-only ones (complex numbers) are skipped by
    the binding. Counting one of the two makes the other half look like a missing
    implementation.
    """
    names = [n for n, _ in cases_mod.golden_cases(cases_mod.golden_inputs())]
    core = [n for n in names if not n.startswith(cases_mod.WEBGPU_PREFIX)]
    bind = [n for n in names if not n.startswith(cases_mod.CORE_ONLY_PREFIXES)]
    return len(names), len(core), len(bind)


def test_docs_do_not_name_a_stale_golden_count():
    """A case count named in the documentation has to be **a count that exists
    now.**

    Three numbers are actually in use — the whole table, the core's minus the
    sister-only ones, and the binding's minus the core-only ones. A number in a
    `golden N cases` slot that is none of the three is stale.

    **Accepting all three loosens the net.** On the day complex numbers went in,
    the total went from 2263 to 2287 while what the binding sees became exactly
    2263, and one stale sentence nearly passed holding an accidentally correct
    number. The number matched and the sentence said "it passes all of them",
    which was no longer true — worth writing down that a check counting numbers
    does not read sentences.
    """
    total, core, bind = _counts()
    allowed = {str(total), str(core), str(bind)}
    stale = []
    for rel in LIVE_DOCS:
        path = ROOT / rel
        # The site may be absent (depending on the checkout). An absent file is
        # not made a failure.
        if not path.exists():
            continue
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            for found in COUNT.findall(line) + DERIVED.findall(line):
                hit = _hit(found)
                if hit not in allowed:
                    stale.append(
                        f"{rel}:{i}  '{hit}' — now it is {total} (total) / "
                        f"{core} (core) / {bind} (binding)")
    assert not stale, (
        "a golden count is stale in a document that speaks in the present:\n  "
        + "\n  ".join(stale) +
        "\n\nThe README and `site/` speak in the present. A sentence that has to "
        "talk about a time should be written without the number or moved to a "
        "design document — changing a past number to the current one is not "
        "fixing a stale number, it is forging history.")


# The places naming a package size, as in `2,358 줄`. It catches them with and
# without thousands separators, in either language.
LINES = re.compile(r"\*{0,2}(\d{1,3}(?:,\d{3})*)\s*줄\*{0,2}"
                   r"|\*{0,2}(\d{1,3}(?:,\d{3})*)[- ]line[s]?\*{0,2}")
# What is counted. Only the packages present in the repository now — the size of
# something deleted belongs to history.
PACKAGES = {"borch", "borch_webgpu"}

# Where a line count is looked for, and where one has to actually be found.
#
# The two are not the same list, and asserting on absence is what made the
# difference visible. `borch/__init__.py` is scanned so that a count added there
# later is checked, and it has never carried one — `git log -S` finds no line
# count in its whole history. Leaving it in the must-yield list would be
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
    """Package name → line count."""
    out = {}
    for name in PACKAGES:
        total = 0
        for path in sorted((ROOT / name).glob("*.py")):
            total += len(path.read_text(encoding="utf-8").splitlines())
        out[name] = total
    return out


# **Numbers from history.** They cannot be confirmed by counting now, so they are
# written down here. Without saying what each one is, they become magic numbers to
# the next person.
HISTORICAL = {
    "5,307": "the TF.js borch_webgpu, deleted in 45be321",
    "3,300": "borch's size when it was split from one file into a package (8177e1d)",
}

# **How much drift is forgiven.** What this catches is a factor of 2.6, not three
# lines. Demanding an exact number breaks the documentation on every one-line edit
# to `_ops.py`, and then this check teaches people how to get past it instead of
# guarding what it is for.
TOLERANCE = 0.05


def test_docs_do_not_name_a_stale_line_count():
    """A **line count** named in the documentation must not be far from reality.

    This machinery went onto the golden count and not onto the line counts, and
    that is exactly where it went wrong twice. Once, counting five of eight files
    wrote 5,307 as **2,312**; once, an estimate of **900** from before `_data.py`
    was carried over without recounting. Side by side those two made the sentence
    "2,312 lines became 900 lines", when in fact 5,307 became 2,361 — the
    direction right and the magnitude wrong by a factor of 2.6.

    Both went into a commit message and the README at once. **Counting by eye is
    not a method.**

    The boundary is the golden count's. The README and the package docstrings
    speak in the present and have to stay current; the design documents
    (`BORCH-TS.md`, `WEBGPU-DESIGN.md`) record a time and are not examined.
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
                # Catching the small numbers mixed into prose makes it noise.
                if said < 100:
                    continue
                if any(abs(said - real) <= real * TOLERANCE for real in sizes.values()):
                    continue
                stale.append(f"{rel}:{i}  '{hit} lines' — now it is " +
                             ", ".join(f"{k} {v:,}" for k, v in sorted(sizes.items())))
    assert not stale, (
        "a line count in the documentation is far from reality:\n  "
        + "\n  ".join(stale) +
        "\n\nCount and fix it. If the number speaks of a time, write it into "
        "`HISTORICAL` along with what it is — changing history to the current "
        "number is not fixing a stale number.")


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
