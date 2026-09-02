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

import importlib.util
import pathlib
import re

import pytest

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
# `README.md` is the front door and `docs/BOOK.md` the long document it opens onto —
# the book was the README until 2026-09-03, and both speak of now.
LIVE_DOCS = ("README.md", "docs/BOOK.md", "site/index.html", "site/ko/index.html")


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
LINE_DOCS = ("docs/BOOK.md", "borch_webgpu/__init__.py", "borch/__init__.py")

# How many claims each document makes, so that losing one is visible. See
# `test_the_documents_still_make_the_claims_this_file_checks` for why a count
# rather than a presence check.
CLAIMS = {
    # Five since the "The core covers 2938 cases" sentence became visible.
    # It read `보는데` rather than `본다` in Korean, so the pattern never
    # matched it and the figure sat stale at 2930 while the two beside it
    # stayed current. Translating it into a phrasing the pattern catches is
    # what put it under watch.
    ("README.md", "golden"): 1,
    ("docs/BOOK.md", "golden"): 5,
    ("site/index.html", "golden"): 1,
    ("site/ko/index.html", "golden"): 1,
    ("docs/BOOK.md", "lines"): 4,
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


# The README says how many pytest cases there are, in two places at once — the file
# `test_diff.py` holds, and the suite's total. Both are **counts of this very run**, and
# nothing was watching either.
SUITE_CLAIM = re.compile(
    r"\*\*[^*]*?\*\*\s*(\d{2,5})\s+cases in that file,\s*over\s+(\d{2,5})\s+in the suite",
    re.S)

# What the README's command installs. The suite total is a count of *this* set, because
# a missing `importorskip` dependency does not skip its file — it takes those cases out
# of the collection. Named here so the check can say which one is absent instead of
# reporting a number as wrong.
DOCUMENTED = ("numpy", "torch", "torchvision", "scipy")


def test_the_readme_does_not_name_a_stale_case_count(request):
    """**The two pytest counts in the README, against what pytest collects now.**

    They read *180 cases in that file, 93% code coverage* for long enough that both had
    drifted: 162 and 92% when someone finally ran it. They sat beside a command that had
    also drifted — missing `torchvision` and `scipy`, so running it as written turned
    whole files into skips and still printed a pass.

    Four wrong things in one paragraph, and the reason is the same for all four:
    **nothing read it.** The golden counts two sections up have had a check since the day
    one of them went stale; these did not, because they are counts of the test suite
    rather than of the library and no instrument was pointed inward.

    Coverage is deliberately not held here. Measuring it needs `pytest-cov` and a full
    instrumented run, which is a real cost on every check of every commit, and a number
    that moves by a point is not the failure this is for. The two case counts move by
    one every time somebody adds a test, which is exactly when a reader should be made
    to look.

    ## It counts this run rather than starting another

    The first version shelled out to `pytest --collect-only`, and that carried an
    assumption it had no business making: **that the child interpreter can import what
    the parent can.** It cannot always — under a `uv run --with torch` the packages live
    in an overlay `sys.executable` alone does not reproduce, and the child then stops on
    `test_diff.py`'s `import torch` with the whole collection interrupted. The check did
    the right thing with that (it refused to compare a number it could not measure) and
    the number was never the problem. A session hit exactly this.

    So the subprocess is gone. pytest has already collected everything before the first
    test runs, and `session.items` is that collection — the same number, from the run
    already happening, with no second environment to get wrong.

    **The cost is that it can only speak for a whole run**, since `-k` and `-m` filter
    what is collected. Asked during a filtered one it skips and says so; CI always runs
    the suite whole, which is where this has to hold.

    ## The suite total was not a property of the repository

    This held both counts to the exact collected number, and it went red on `main` in
    another session's hands: *the README says 1208 in the suite; this run collected
    1189.* Nobody had removed nineteen tests. That session's shell had no `scipy`, and
    `test_svhn.py` opens with an `importorskip` — **which does not turn its file into
    skips, it takes those nineteen out of the collection entirely.** Two environments,
    two totals, one repository.

    So the number this compared was a property of the installed packages, and holding a
    number like that to equality means the check is **wrong wherever the reader's setup
    differs from the author's** — the precise failure it exists to prevent, pointed the
    other way. The README's own warning two lines below already said `scipy` hid
    nineteen checks; the check walked into the trap its own paragraph described.

    Both halves are fixed here, and they are separate fixes:

    **The environment is named.** Absent one of `DOCUMENTED`, the collection is a
    different collection, and this says so and skips rather than reporting a number as
    stale. That skip names the missing package, so it is a thing to install rather than
    a check quietly switched off.

    **The total is a floor.** Held to equality, every test anyone adds anywhere reddens
    one sentence in the README — and lands that on whoever wrote the sentence rather
    than whoever added the test. A reader takes scale from this number, not identity;
    1204 and 1208 tell them the same thing. As a floor it goes red when the suite
    **shrinks**, and tests disappearing is worth stopping for in a way tests arriving
    is not.

    **`test_diff.py` stays exact.** It is what the sentence is about, it moves rarely,
    and it is the count that was actually found wrong. It also needs only what every
    run has, so no dependency moves it.
    """
    text = (ROOT / "docs" / "BOOK.md").read_text(encoding="utf-8")
    claim = SUITE_CLAIM.search(text)
    assert claim, (
        "the README no longer states the pytest counts in the shape this reads.\n"
        "  Fix the pattern, or drop this check if the claim went away — a check that\n"
        "  cannot find its subject must not pass quietly, which is how the sentence it\n"
        "  guards gets to be wrong for weeks.")
    said_file, said_floor = int(claim.group(1)), int(claim.group(2))

    missing = [m for m in DOCUMENTED if importlib.util.find_spec(m) is None]
    if missing:
        pytest.skip(
            f"counts the collection the README's command produces; {', '.join(missing)} "
            "is not installed here, so this run collects fewer files than that one")

    option = request.config.option
    narrowed = getattr(option, "keyword", "") or getattr(option, "markexpr", "")
    args = [pathlib.Path(a).resolve() for a in request.config.args]
    whole = args == [(ROOT / "tests").resolve()]
    if narrowed or not whole:
        pytest.skip("counts a whole run — this one was narrowed by -k/-m or by path")

    items = request.session.items
    got_suite = len(items)
    got_file = sum(1 for i in items if i.path.name == "test_diff.py")
    assert said_file == got_file, (
        f"the README says {said_file} cases in test_diff.py; this run collected "
        f"{got_file}.\n"
        "  That file is what the sentence is about. Update the sentence.")
    assert got_suite > said_floor, (
        f"the README says over {said_floor} cases in the suite; this run collected "
        f"{got_suite}.\n"
        "  The floor is written to catch the suite **shrinking**, so before lowering it,\n"
        "  find out what stopped being collected — an `importorskip` at the top of a\n"
        "  file removes its cases rather than skipping them, and that reads exactly\n"
        "  like tests having been deleted.")
