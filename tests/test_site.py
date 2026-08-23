"""Whether the site has drifted from the repository.

The same place `tests/test_docs.py` occupies for the **numbers** in the documentation.
What this one holds is the **generated files** — `site/assets/api.json` is what
`site/build_api.py` pulled out of the declaration files, and left unregenerated after the
source grows, the site shows an API that does not exist or leaves out one that does.

That drift **does not show on screen.** An index that is slightly short and an index that
was always that length have the same shape, so looking for it by eye works particularly
badly here (which happened while writing the generator — the parser was catching 18 of
the Tensor's 422 methods and the screen looked fine).
"""

import gzip
import hashlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
GENERATOR = ROOT / "site" / "build_api.py"
API = ROOT / "site" / "assets" / "api.json"
INDEX = ROOT / "site" / "assets" / "api-index.json"
DECL = ROOT / "borch-ts" / "dist" / "src"


def _load_module(name, path):
    """Loads a file as a module under a name of our choosing, touching no import path.

    `tests/browser` holds files called `run`, `bench`, `cost` and `serialize`, and so does
    `borch-ts/test`. Putting either directory on `sys.path` decides for the rest of the
    session which one those names mean.
    """
    import importlib.util                                            # noqa: PLC0415

    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _runner():
    """`borch-ts/test/run.py`, loaded as a module.

    It is loaded under its own name, and its directory goes on `sys.path` only for the
    load. Left there, `run`, `bench`, `cost` and `serialize` would all resolve to that tree
    for the rest of the session, and `tests/browser` has files of every one of those names.

    **Two things in this file need it** — the freshness rule and the gap-table ledger —
    and both want the module rather than its text. Matching the text is what
    `test_alias_rows.py` does, and its expression cannot see a row whose reason runs to a
    second line.
    """
    import importlib.util                                            # noqa: PLC0415

    here = str(ROOT / "borch-ts" / "test")
    sys.path.insert(0, here)
    try:
        spec = importlib.util.spec_from_file_location(
            "bt_ts_runner", ROOT / "borch-ts" / "test" / "run.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        if here in sys.path:
            sys.path.remove(here)
    return mod


def _stale_dist():
    """The reason `dist` is out of date, or `None`.

    **The rule is not restated here.** `borch-ts/test/run.py` already decides what "stale"
    means for the golden runner, and a second copy of a freshness rule is a rule that
    diverges — the day the two disagree, one runner stops and the other does not, and the
    difference reads as a defect in whatever was being changed.
    """
    try:
        _runner().require_fresh_dist(ROOT)
    except SystemExit as exc:
        return str(exc)
    return None


def test_api_reference_is_not_stale():
    """`api.json` has to equal what the declaration files give right now.

    The declaration files are gitignored and so appear in no commit — with none of them
    there is nothing to compare against, so it skips. **Making absence a failure** turns
    this red on any checkout that has not built the bundle, and then people learn how to
    switch checks off.

    **A stale `dist` is separated out before the counts are compared.** Without that, one
    message covers two opposite causes: source ahead of `dist` (pulled or rebased without
    building) and `dist` ahead of the committed index (mid-edit of the source). Both print
    the same "entries N → M" line, and the reader chases whichever they thought of first.
    That really happened — a breaking change to borch.ts's save/load landed, a session
    rebased onto it without rebuilding, and 1671 → 1664 read as an index regression when
    the index was correct and the bundle was three commits old.

    The golden runner has stopped on this since it cost two people a day each. The rule is
    borrowed rather than rewritten; see `_stale_dist`.
    """
    if not DECL.exists():
        pytest.skip(f"no declaration files ({DECL.relative_to(ROOT)}) — run npm run build:ts first")
    if not API.exists():
        pytest.fail("site/assets/api.json is missing — python3 site/build_api.py")
    stale = _stale_dist()
    if stale:
        pytest.fail(
            "the bundle is older than the source, so the counts below would be measured\n"
            "against an API that no longer exists:\n\n  " + stale.replace("\n", "\n  "))

    # **The generator writes two files** — the index and the name index. Restoring only
    # one left `api-index.json` modified in the tree the check had run in; a check that
    # touches the working tree and leaves means the next person commits a change they
    # never made as their own.
    made = [API, INDEX]
    before = {p: p.read_text(encoding="utf-8") for p in made if p.exists()}
    proc = subprocess.run([sys.executable, str(GENERATOR)], capture_output=True, text=True)
    assert proc.returncode == 0, f"the generator stopped:\n{proc.stderr}"
    after = API.read_text(encoding="utf-8")
    for path, text in before.items():
        path.write_text(text, encoding="utf-8")

    if before.get(API) != after:
        old, new = json.loads(before[API]), json.loads(after)
        pytest.fail(
            "the API reference differs from the declaration files — entries "
            f"{old['total']} → {new['total']}.\n"
            "  regenerate: python3 site/build_api.py\n"
            "  (a description is fixed in the source comment, not in this file.)\n"
            "\n"
            "  **If you are in the middle of editing borch-ts/src this is expected.** This\n"
            "  check compares against the `dist` on disk right now, so a build made from\n"
            "  uncommitted source counts names that do not exist yet and diverges here.\n"
            "  Regenerating the index alongside that commit settles it — keeping the site\n"
            "  from showing an API that is not there is what this check is for, and that\n"
            "  moment is exactly now.")


def test_site_examples_name_only_real_modules():
    """The module list the site writes down for loading the Python binding has to match reality.

    `site/assets/runner.js` names the `.py` files it lays onto Pyodide's virtual
    filesystem. One left out blows up **loudly as an ImportError**, so that side shows
    itself; the other side — a file gone from the package while the name stays in the
    list — makes fetch return 404 and `runner.js` turn it into an exception. Both are
    known only after the user presses Run. This looks first.
    """
    runner = (ROOT / "site" / "assets" / "runner.js").read_text(encoding="utf-8")
    block = runner[runner.index("const PACKAGES = {"):runner.index("let pyodide")]
    for package in ("borch", "borch_webgpu"):
        listed = set()
        chunk = block[block.index(f"{package}:"):]
        for line in chunk.splitlines():
            if "]" in line:
                listed.update(part.strip().strip('",') for part in line.split('"') if part.startswith("_") or part == "__init__")
                break
            listed.update(part.strip().strip('",') for part in line.split('"') if part.startswith("_") or part == "__init__")
        real = {p.stem for p in (ROOT / package).glob("*.py")}
        missing = listed - real
        assert not missing, (
            f"the site loads modules {package} does not have: {sorted(missing)}\n"
            "  fix PACKAGES in site/assets/runner.js.")
        forgotten = real - listed
        assert not forgotten, (
            f"the site does not load modules {package} has: {sorted(forgotten)}\n"
            "  add them to PACKAGES in site/assets/runner.js — left out, the browser "
            "blows up with an ImportError.")

# ── where sizes are claimed ───────────────────────────────────────────
#
# **`KB` is a number too.** `test_docs.py` holds the golden count and the package line
# count, and size alone was missing, during which "232KB ES module" went **3.3× stale**
# (measured at 770KB). That number had been copied from the README into two pages of the
# site — a stale source makes stale copies.
#
# All three here are **measurable**. A number that cannot be measured is not on this list.
SIZE_CLAIMS = (
    # (document, the marker that line must carry, the name of what measures the truth)
    ("README.md", "ES module", "bundle"),
    ("site/index.html", "ES module", "bundle"),
    ("site/ko/index.html", "ES 모듈", "bundle"),
)

# How far the measurement may drift. The same 5% as the line counts in `test_docs.py` —
# what is being caught is a 3.3× error, not the few kilobytes one commit adds.
SIZE_TOLERANCE = 0.05

KB = re.compile(r"(\d{2,5})\s*KB")


def _bundle_sizes():
    """The ES module the browser loads, as (raw, gzip) sizes in KB."""
    raw = b"".join(p.read_bytes() for p in sorted(DECL.glob("*.js")))
    return len(raw) / 1024, len(gzip.compress(raw, 9)) / 1024


def test_docs_do_not_name_a_stale_bundle_size():
    """The **sizes** the documentation claims must not be far from the truth.

    One line may carry several numbers (raw and gzip). If **none** of them matches
    reality it is stale — passing on one match alone misses a sentence like "232KB (raw)",
    where **the number is right and the label is wrong.**
    """
    if not DECL.exists():
        pytest.skip(f"no declaration files ({DECL.relative_to(ROOT)}) — run npm run build:ts first")

    raw_kb, gzip_kb = _bundle_sizes()
    ok = lambda said: any(abs(said - real) <= real * SIZE_TOLERANCE
                          for real in (raw_kb, gzip_kb))

    stale = []
    for rel, marker, _ in SIZE_CLAIMS:
        path = ROOT / rel
        if not path.exists():
            continue
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if marker not in line:
                continue
            said = [int(hit) for hit in KB.findall(line)]
            if not said:
                stale.append(f"{rel}:{i}  no size written")
            elif not all(ok(v) for v in said):
                stale.append(
                    f"{rel}:{i}  {said} KB — it is now {raw_kb:.0f}KB raw · "
                    f"{gzip_kb:.0f}KB gzipped")
    assert not stale, (
        "the bundle sizes the documentation claims are stale:\n  " + "\n  ".join(stale) +
        "\n\nmeasure and fix: cat borch-ts/dist/src/*.js | wc -c")


# ── where a page quotes another document's heading ────────────────────
#
# A lesson page pointing the reader at a section of the README writes that section's
# title in quotation marks. That is a **wording contract between two files**, and the
# usual way it breaks is the usual way: the heading is rewritten, the quote is not, and
# the reader is sent looking for a section that no longer exists under that name.
#
# It broke exactly that way here — both pages quoted `초록색이 거짓일 수 있는 일곱 자리`
# while the README's heading became `Seven places where green can be a lie`, and nothing
# said so. The Korean page quotes the English heading on purpose: it is a **name of a
# place in another document**, not prose to be read, so translating it would break the
# very link the sentence exists to make.
HEADING_QUOTES = (
    ("site/learn/08-debugging.html", "README.md", "Seven places where green can be a lie"),
    ("site/ko/learn/08-debugging.html", "README.md", "Seven places where green can be a lie"),
)


def test_a_page_quoting_a_heading_quotes_one_that_exists():
    """Both ends of the quote, so neither side can move alone."""
    broken = []
    for rel, target, heading in HEADING_QUOTES:
        page, doc = ROOT / rel, ROOT / target
        if heading not in page.read_text(encoding="utf-8"):
            broken.append(f"{rel} no longer quotes {heading!r} — the page moved")
        elif f"# {heading}" not in doc.read_text(encoding="utf-8"):
            broken.append(f"{target} has no heading {heading!r} — {rel} points at nothing")
    assert not broken, (
        "a page quotes a heading that is not there:\n  " + "\n  ".join(broken) +
        "\n\nfix both ends together, or drop the row if the sentence went away.")


# ── how many cases borch.ts has a body for ────────────────────────────
#
# The README states two numbers about the TypeScript side: how many golden cases have a
# TS body, and how many deliberately do not. They were carried across from a browser run
# and said 2352 and 608 while the truth was 2629 and 362 — the README even recorded that
# they were unverified, which is honest and is not a check.
#
# **They do not need a browser.** The case table registers names without running any of
# them, so loading the compiled module in node and counting the map answers it. A text
# search does not: `out.set(` appears 784 times against 2629 real names, because the
# names are built programmatically.
TS_CASES = ROOT / "borch-ts" / "dist" / "test" / "cases.js"
GOLDEN_JSON = ROOT / "tests" / "golden.json"
WRITTEN = re.compile(r"written TS bodies for (\d[\d,]*) cases")
REMAINING = re.compile(r"The remaining (\d[\d,]*) are")

COUNT_CASES = """
import fs from "node:fs";
const doc = JSON.parse(fs.readFileSync(process.env.GOLDEN, "utf8"));
const { cases, Inputs } = await import(process.env.CASES);
const names = [...cases(new Inputs(doc.inputs)).keys()];
const mine = names.filter((n) => doc.cases[n] !== undefined).length;
console.log(JSON.stringify({ written: names.length, golden: Object.keys(doc.cases).length,
                             unknown: names.length - mine }));
"""


def test_the_readme_counts_the_typescript_bodies_correctly():
    """Both numbers, and that they still add up to the golden.

    Counting only the written half lets the other drift, and the two are a partition:
    written plus deliberately-absent is every golden case. A case registered on the TS
    side under a name the golden does not have is checked too — it is the silent
    disappearance `borch-ts/test/cases.ts` warns about, seen from the other end.
    """
    if not TS_CASES.exists():
        pytest.skip(f"no {TS_CASES.relative_to(ROOT)} — run npm run build:ts first")
    node = shutil.which("node")
    if node is None:
        pytest.skip("no node")
    # **The measurement is a build artefact, so a stale build accuses the document.**
    # This happened on this test's first day: a merge added cases, `dist` was three
    # commits old, and it reported the README as claiming 2635 against a measured 2629 —
    # naming the one file that was right. The rule is `test_api_reference_is_not_stale`'s,
    # for the reason its docstring gives about two people losing a day each.
    stale = _stale_dist()
    if stale:
        pytest.fail(
            "the bundle is older than the source, so the count below would be measured\n"
            "against a case table that no longer exists:\n\n  " + stale.replace("\n", "\n  "))

    out = subprocess.run([node, "--input-type=module", "-e", COUNT_CASES],
                         capture_output=True, text=True, cwd=ROOT,
                         env={**os.environ, "GOLDEN": str(GOLDEN_JSON),
                              "CASES": TS_CASES.as_uri()})
    assert out.returncode == 0, f"could not load the case table:\n{out.stderr[-2000:]}"
    got = json.loads(out.stdout)

    assert got["unknown"] == 0, (
        f"{got['unknown']} TS cases carry a name tests/golden.json does not have — "
        "the answer for those is nowhere, and the runner passes them by in silence.")

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    said_written = WRITTEN.search(readme)
    said_remaining = REMAINING.search(readme)
    assert said_written and said_remaining, (
        "the README stopped stating the TS body counts in the shape this reads — "
        "fix the patterns above, or drop this test if the claim went away.")

    written = int(said_written.group(1).replace(",", ""))
    remaining = int(said_remaining.group(1).replace(",", ""))
    real_remaining = got["golden"] - got["written"]
    assert (written, remaining) == (got["written"], real_remaining), (
        f"the README says {written} written and {remaining} remaining; measured "
        f"{got['written']} and {real_remaining} against {got['golden']} golden cases.\n"
        "  measure: node --input-type=module -e '…' (this test prints the command it ran)")


def test_the_documents_still_carry_the_markers_this_file_looks_for():
    """A marker that matches nothing checks nothing, **and says nothing while doing it.**

    The loop above walks the lines of each document looking for its marker. A document
    that no longer contains its marker contributes no lines, `stale` stays empty and the
    test is green — the same green as a document whose numbers are right.

    This is not hypothetical here. README.md's marker was `ES 모듈`, and translating the
    README to English left it matching nothing. The size claim went unchecked in the one
    document `test_docs.py` calls the source the site copies from, and the suite stayed
    green through the commit that did it. Rewrapping a paragraph is enough to do it
    again, which is why the marker is asserted rather than trusted.
    """
    missing = [f"{rel}: {marker!r}" for rel, marker, _ in SIZE_CLAIMS
               if (ROOT / rel).exists()
               and marker not in (ROOT / rel).read_text(encoding="utf-8")]
    assert not missing, (
        "SIZE_CLAIMS names markers these documents no longer carry:\n  " +
        "\n  ".join(missing) +
        "\n\nthe claim moved or was rewritten — point the marker at it again, or drop "
        "the row if the document stopped claiming a size.")

# ── whether the pages agree with each other ───────────────────────────
#
# The site is twenty pages in two languages. **Sweeping by hand is not a method** — twenty
# pages were opened in a browser and three things were caught that way (only the landing
# carried an extra anchor, the playground had no entry of its own, and the Korean API
# page's label was in English). That method does not repeat itself next time.

SITE = ROOT / "site"
HREF = re.compile(r'(?:href|src)="([^"]+)"')
NAV = re.compile(r'<header class="top">.*?<nav>(.*?)</nav>', re.S)
LINK_TEXT = re.compile(r'<a [^>]*>([^<]*)</a>')


def _pages():
    return sorted(SITE.rglob("*.html"))


def test_site_has_no_broken_relative_links():
    """Every relative path a page points at has to exist.

    A broken link **is invisible until it is pressed.** On a documentation site that is
    not a missing page but missing trust — someone who meets one 404 doubts the rest.
    """
    missing = []
    for page in _pages():
        for raw in HREF.findall(page.read_text(encoding="utf-8")):
            if raw.startswith(("http://", "https://", "#", "data:", "mailto:")):
                continue
            target = (page.parent / raw.split("#")[0].split("?")[0]).resolve()
            if target.is_dir():
                target = target / "index.html"
            if not target.exists():
                missing.append(f"{page.relative_to(ROOT)} → {raw}")
    assert not missing, "the site has broken links:\n  " + "\n  ".join(missing)


def test_every_page_carries_the_same_global_nav():
    """The global nav has to be **one and the same on every page**.

    Entries that change as you move between pages read as arriving at a different site.
    Where you are is marked only by `class="on"`, and there has to be **exactly one** of
    those — two means not knowing where you are, and none (the landing excepted) means
    the nav has no place for that page.

    The labels differ per language, so languages are compared among themselves.
    """
    shapes = {}
    problems = []
    for page in _pages():
        text = page.read_text(encoding="utf-8")
        nav = NAV.search(text)
        assert nav, f"{page.relative_to(ROOT)} has no global nav"
        block = nav.group(1)
        # The language switch points somewhere different per page, so only its label is read.
        labels = tuple(LINK_TEXT.findall(block))
        lang = "ko" if page.relative_to(SITE).as_posix().startswith("ko/") else "en"
        shapes.setdefault(lang, {}).setdefault(labels, []).append(
            page.relative_to(ROOT).as_posix())

        marked = block.count('class="on"')
        is_home = page.name == "index.html" and page.parent in (SITE, SITE / "ko")
        if is_home and marked:
            problems.append(f"{page.relative_to(ROOT)}: the landing, yet the nav marks a place")
        elif not is_home and marked != 1:
            problems.append(f"{page.relative_to(ROOT)}: {marked} marks for the current place")

    for lang, found in shapes.items():
        if len(found) > 1:
            lines = [f"  {labels} ← {len(pages)} pages (e.g. {pages[0]})"
                     for labels, pages in found.items()]
            problems.append(f"the nav diverges across the {lang} pages:\n" + "\n".join(lines))

    assert not problems, "\n".join(problems)


SIDEBAR = re.compile(r'<aside class="sidebar">.*?<nav>(.*?)</nav>', re.S)


def test_a_section_sidebar_lists_the_same_pages_on_every_page_of_it():
    """Within one section the sidebar has to be **one and the same list**.

    The global nav above is checked already; the sidebar was not, and it drifted the way
    an unchecked list always drifts. `site/learn/06-save-load.html` and its Korean twin
    listed six lessons while eight existed — a reader who arrived at lesson 6 was shown a
    course that ended there. **Nothing was broken**: every link on the page worked, the
    page rendered, and the link checker above was satisfied, because a link that is simply
    absent is not a broken link.

    What makes it catchable is that the sidebar is a **claim about its siblings**, so the
    siblings can be asked. Marking the current place is `class="on"`, exactly as in the
    global nav, so it is stripped before the lists are compared.
    """
    sections = {}
    problems = []
    for page in _pages():
        found = SIDEBAR.search(page.read_text(encoding="utf-8"))
        if not found:
            continue
        block = re.sub(r' class="on"| aria-current="page"', "", found.group(1))
        entries = tuple(re.findall(r'<a href="([^"]+)"[^>]*>([^<]*)</a>', block))
        sections.setdefault(page.parent, {}).setdefault(entries, []).append(
            page.relative_to(ROOT).as_posix())

    for folder, found in sections.items():
        if len(found) > 1:
            biggest = max(found, key=lambda e: len(found[e]))
            for entries, pages in found.items():
                if entries == biggest:
                    continue
                missing = [href for href, _ in biggest if href not in dict(entries)]
                problems.append(
                    f"{folder.relative_to(ROOT)}: {', '.join(pages)} list "
                    f"{len(entries)} entries where {len(biggest)} is usual"
                    + (f" (absent: {', '.join(missing)})" if missing else ""))

    assert not problems, "a section's sidebar disagrees with itself:\n  " + "\n  ".join(problems)


def test_site_links_to_this_repository():
    """The GitHub address the site points at has to be **this repository's**.

    When the repository was renamed from `browsertorch` to `borch`, thirty-eight files of
    the site still carried the old address. `tests/rename.py` is a tool for lowercase
    identifiers and had no rule for a name inside a URL, and the links **still opened**
    through a redirect — staleness that does not break, so nobody looked.

    So a hand-written address is not kept by hand. `origin` holds the answer, and this
    pins to it. A checkout with no remote (an archive, some CI modes) has nowhere to ask,
    so it skips.
    """
    remote = subprocess.run(["git", "remote", "get-url", "origin"],
                            cwd=ROOT, capture_output=True, text=True)
    if remote.returncode != 0 or not remote.stdout.strip():
        pytest.skip("no origin — nothing to compare the address against")

    here = remote.stdout.strip().removesuffix(".git").replace("git@github.com:",
                                                             "https://github.com/")
    # **Only our own organisation's addresses.** The site also points at the Pyodide
    # repository (MPL-2.0 asks for a route to the source), which is someone else's and has
    # no reason to match this one. Checking every address caught exactly that notice link.
    owner = here.rsplit("/", 2)[-2]
    linked = re.compile(rf"https://github\.com/{re.escape(owner)}/[\w.-]+")
    wrong = []
    for page in _pages():
        for i, line in enumerate(page.read_text(encoding="utf-8").splitlines(), 1):
            for hit in linked.findall(line):
                if hit.removesuffix(".git") != here:
                    wrong.append(f"{page.relative_to(ROOT)}:{i}  {hit} — this repository is {here}")
    assert not wrong, (
        "the site points at another repository:\n  " + "\n  ".join(wrong[:12]) +
        (f"\n  … and {len(wrong) - 12} more" if len(wrong) > 12 else ""))


def test_share_metadata_is_complete_and_gets_an_address():
    """Every page has share metadata, and something exists that fills the placeholder.

    The site says a single URL is the whole deployment. For that to be true, what appears
    when the link is pasted is part of the claim — and until now nothing appeared.

    Only the deploying side knows the address, so the HTML carries `%OG_BASE%` and the
    workflow fills it — **a shape where the two halves can drift apart**, so this looks
    for both together. With the placeholder present and no step to fill it, a crawler
    takes `%OG_BASE%/…` away as written, and that failure is visible only on someone
    else's timeline after the deploy.
    """
    missing = [str(p.relative_to(ROOT)) for p in _pages()
               if "og:image" not in p.read_text(encoding="utf-8")]
    assert not missing, "pages with no share metadata:\n  " + "\n  ".join(missing)

    image = SITE / "assets" / "og.png"
    assert image.exists(), "site/assets/og.png is missing — uv run --with pillow python site/make_og.py"
    size = image.stat().st_size / 1024
    assert size < 300, f"og.png is {size:.0f}KB — some social crawlers will not fetch it"

    workflow = ROOT / ".github" / "workflows" / "pages.yml"
    if not workflow.exists():
        pytest.skip("no deploy workflow")
    text = workflow.read_text(encoding="utf-8")
    assert "%OG_BASE%" in text, (
        "the pages use %OG_BASE% and the deploy workflow does not fill it — "
        "shipped as is, a crawler reads the placeholder as the address.")


def test_dual_language_blocks_do_not_lose_a_half():
    """When one block holds two languages, this blocks the place a copy disappears quietly.

    `runnable.js` stores the sources **keyed by language**. A second `<script>` without
    `data-lang` counts as the outer `div`'s language, overwrites the first, and no tabs
    appear on screen — Python was written in and the page shows JavaScript only, with
    nothing blowing up. It is the shape this repository has already caught twice, so this
    stops and names it.

    The outer `div`'s `data-lang` is checked too. If it is not among the languages held,
    the side shown first is a language that is not there, and the reader meets a surface
    they never chose.
    """
    inner = re.compile(r'<script type="text/plain"([^>]*)>', re.S)
    wrong = []
    for page in _pages():
        text = page.read_text(encoding="utf-8")
        for m in re.finditer(r'<div class="runnable"([^>]*)>', text):
            head = text[m.end():m.end() + 400]
            attrs = m.group(1)
            outer = "py" if 'data-lang="py"' in attrs else "js"
            # Only the <script> tags belonging to this block — up to the next runnable.
            stop = text.find('<div class="runnable"', m.end())
            body = text[m.end():stop if stop > 0 else len(text)]
            langs = [("py" if 'data-lang="py"' in a else "js" if 'data-lang="js"' in a else outer)
                     for a in inner.findall(body)]
            where = f"{page.relative_to(ROOT)} · {head.strip()[:40]}"
            if len(langs) != len(set(langs)):
                wrong.append(f"{where} — the same language twice: {langs}")
            elif langs and outer not in langs:
                wrong.append(f"{where} — the outer is {outer} while it holds {langs}")
    assert not wrong, "a dual-language block loses one of its halves:\n  " + "\n  ".join(wrong)


def test_no_block_declares_a_name_the_runner_injects():
    """A runnable block that redeclares a name the runner injects does not run.

    The runner spreads names such as `log` and `show` ahead of the user's code and joins
    it all into one module. A block declaring the same name with `const` makes the joined
    module **a syntax error**, and that shows only when someone presses that block.

    It has happened three times here — twice with `probe` and once with `show`. `show`
    was one name added to the injected list while attaching the tutorials, which quietly
    killed lesson 8's block, and it was committed that way. **Whoever adds a name does not
    look at the blocks that already exist** is the shape of the problem, so this blocks it
    rather than human discipline.

    The injected list is read out of `runner.js` — copied here, this check alone would go
    stale the next time a name is added.
    """
    runner = (SITE / "assets" / "runner.js").read_text(encoding="utf-8")
    injected = set()
    for line in runner.splitlines():
        m = re.search(r'^\s*"const \{(.*)$', line)
        if m:
            injected |= {n.strip() for n in m.group(1).split(",") if n.strip()}
        m = re.search(r'^\s*"\s*([a-zA-Z, ]+)\} = borch;",', line)
        if m:
            injected |= {n.strip() for n in m.group(1).split(",") if n.strip()}
        m = re.search(r'^\s*"const ([a-zA-Z_$][\w$]*) = ', line)
        if m:
            injected.add(m.group(1))
    injected -= {"__pg"}
    assert len(injected) > 15, f"could not read the injected list — {sorted(injected)}"

    clash = re.compile(r"^\s*(?:const|let|var|function|class)\s+([a-zA-Z_$][\w$]*)", re.M)
    wrong = []
    for page in _pages():
        text = page.read_text(encoding="utf-8")
        for m in re.finditer(r'<script type="text/plain"([^>]*)>(.*?)</script>', text, re.S):
            if 'data-lang="py"' in m.group(1):
                continue
            for hit in clash.finditer(m.group(2)):
                if hit.group(1) in injected:
                    wrong.append(f"{page.relative_to(ROOT)} — a block redeclares `{hit.group(1)}`")
    assert not wrong, ("blocks overwrite names the runner injects (those blocks die as syntax errors):\n  "
                       + "\n  ".join(sorted(set(wrong))))


def test_korean_api_descriptions_are_not_stale():
    """Whether each Korean description still matches the source it was made from.

    This page read for a long time that it does not translate, because a translation
    drifts from the source the day it is written. The worry is right. Drift cannot be
    prevented; **drift being quiet** can.

    The direction turned over when the source comments became English. It used to hold an
    English translation of a Korean source; it now holds a Korean translation of an
    English one, which is the more useful direction — the source changes in English from
    here on, so the side at risk of going stale is the Korean.

    So every entry in `site/api_ko.json` carries a hash of the English it was made from.
    When the source TSDoc moves, the hash stops matching and this blows up naming it. The
    key is printed as it stands, so whoever fixes it does not have to hunt for what moved.

    Being partly translated is not a failure — the screen shows the source and says it is
    not carried across, so the reader is not deceived. Failing on that too would let
    nothing in short of doing all of them at once.
    """
    api = json.loads((SITE / "assets" / "api.json").read_text(encoding="utf-8"))
    table_path = ROOT / "site" / "api_ko.json"
    assert table_path.exists(), "site/api_ko.json is missing — it is where the Korean lives"
    table = json.loads(table_path.read_text(encoding="utf-8"))

    def fingerprint(text):
        return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()[:12]

    live = {}
    for mod in api["modules"]:
        live[f"{mod['name']}/"] = mod.get("doc") or ""
        for sym in mod["symbols"]:
            live[f"{mod['name']}/{sym['name']}"] = sym.get("doc") or ""
            for mem in sym["members"]:
                live[f"{mod['name']}/{sym['name']}.{mem['name']}"] = mem.get("doc") or ""

    stale, orphan = [], []
    for key, got in table.items():
        if key not in live or not live[key].strip():
            orphan.append(key)
        elif got.get("src") != fingerprint(live[key]):
            stale.append(key)
    assert not stale, ("the Korean did not follow its source — fix site/api_ko.json and "
                       "put the new hash in src:\n  " + "\n  ".join(sorted(stale)[:20]))
    assert not orphan, ("Korean entries whose description is gone — renamed or deleted:\n  "
                        + "\n  ".join(sorted(orphan)[:20]))

    for key, got in table.items():
        assert got.get("ko", "").strip(), f"the Korean for {key} is empty"


def test_vendored_pyodide_matches_its_lock():
    """Whether the six Pyodide files in the repository are the bytes the committed lock names.

    **This check could not exist before the files were committed.** The lock had been
    there a long time, but a fresh runner had no files, and `fetch` then wrote a new lock
    from whatever arrived — what it compared against was itself. With both in the
    repository, it runs here with no network.

    What it catches: a version bump that edited the lock and not the files, the reverse,
    and a half-finished download. All three appear in the browser as the single sentence
    "Python mode does not come up".
    """
    vendor = ROOT / "vendor" / "pyodide"
    lock = ROOT / "tests" / "browser" / "assets.lock"
    assert lock.exists(), "tests/browser/assets.lock is missing"

    # **The comparison is borrowed, not rewritten.** `tests/browser/vendor.py` already
    # decides what "agrees with the lock" means, and the runners stop on its answer. A
    # second copy here was the same rule written twice: the day they disagree, the suite
    # and the runners say different things about the same six files, and the difference
    # reads as a defect in whatever was being changed. Same reason `_stale_dist` loads
    # `run.py` rather than restating freshness.
    checker = _load_module("bt_vendor", ROOT / "tests" / "browser" / "vendor.py")
    want = checker._read_lock()
    assert want, "the lock file is empty"

    wrong = [p.text for p in checker.check(quiet=True)]
    assert not wrong, ("the repository's Pyodide differs from the lock:\n  " + "\n  ".join(wrong))

    # Something not in the lock slipping in ships in the deploy as bytes nobody measured.
    extra = sorted(p.name for p in vendor.iterdir()
                   if f"pyodide/{p.name}" not in want)
    assert not extra, f"vendor/pyodide holds files the lock does not name: {extra}"


def test_tutorial_sprites_agree_with_their_labels():
    """Whether the sprite and its label file agree with each other.

    Tutorials 4 and 5 trust this pair to read pixels — they compute positions from the
    json's `cols` and `tile` and cut them out of the image. Out of step, **it is not an
    exception but the wrong picture**, and on screen it appears only as "the accuracy does
    not go up".

    This place appeared when the data was committed. It used to be regenerated on every
    deploy, so the image and the labels always came from one run; now one of them can be
    updated and committed alone.

    It reads the JPEG header directly — calling Pillow would add a dependency to CI's
    pytest line for the sake of this one check.
    """
    data = SITE / "assets" / "data"
    wrong = []
    for name in ("train", "test"):
        image, meta = data / f"cifar-{name}.jpg", data / f"cifar-{name}.json"
        if not image.exists() or not meta.exists():
            wrong.append(f"cifar-{name}: image or labels missing")
            continue
        spec = json.loads(meta.read_text(encoding="utf-8"))
        count, cols, tile = spec["count"], spec["cols"], spec["tile"]
        if len(spec["labels"]) != count:
            wrong.append(f"cifar-{name}: {len(spec['labels'])} labels while count says {count}")
        got = _jpeg_size(image.read_bytes())
        want = (((count + cols - 1) // cols) * tile, cols * tile)
        if got != want:
            wrong.append(f"cifar-{name}: the image is {got} while the labels say {want}")
    assert not wrong, "the tutorial data disagrees with itself:\n  " + "\n  ".join(wrong)


def _jpeg_size(blob):
    """A JPEG's (height, width), read by finding the SOF marker."""
    i = 2
    while i < len(blob) - 9:
        if blob[i] != 0xFF:
            i += 1
            continue
        marker = blob[i + 1]
        if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
            return (int.from_bytes(blob[i + 5:i + 7], "big"),
                    int.from_bytes(blob[i + 7:i + 9], "big"))
        i += 2 + int.from_bytes(blob[i + 2:i + 4], "big")
    raise AssertionError("could not find the size marker in the JPEG")


def test_the_editors_have_a_way_out_for_the_keyboard():
    """Tab indents in a code editor. **So there has to be a separate way out.**

    Without one, someone who arrives by keyboard cannot leave — Tab only grows the spaces
    and focus does not move. Measured: two presses took 822 characters to 826 with focus
    unmoved. Without a mouse there is nothing to do but close the tab, and the failure has
    a name in accessibility (WCAG 2.1.2, keyboard trap).

    There is a reason this went unseen. The focus **ring** had already been fixed
    (`86528b0` — "unusable without a mouse"). The visible part was repaired and **being
    trapped** stayed — a place nothing separates without pressing the key.

    The door is Escape: it arms once and the next Tab leaves. Here only the three pieces
    of that are checked for in the code — whether it is actually prevented or not was
    measured in a browser.
    """
    wrong = []
    for rel in ("site/assets/runnable.js", "site/assets/playground.js"):
        text = (ROOT / rel).read_text(encoding="utf-8")
        if 'e.key === "Tab"' not in text:
            continue
        if 'e.key === "Escape"' not in text:
            wrong.append(f"{rel}: it intercepts Tab and has no Escape door out")
        elif '"Tab" && leaving' not in text:
            wrong.append(f"{rel}: it takes Escape and does not release the Tab after it")
    assert not wrong, "a code editor traps the keyboard:\n  " + "\n  ".join(wrong)


def test_no_korean_hides_where_only_a_screen_reader_looks():
    """Whether Korean is hardcoded where the eye never reaches — `aria-label`, `title`, `alt`.

    The English playground's editor introduced itself to a screen reader as
    "자바스크립트 코드" — `playground.js` wrote those characters regardless of the page's
    language. It never appears on screen, so a review by eye cannot catch it.

    Only the English side is checked. Korean on a Korean page is the right words.
    """
    attr = re.compile(r'(?:aria-label|title|alt|placeholder)="([^"]*)"')
    hangul = re.compile(r"[가-힣]")
    wrong = []
    for page in _pages():
        if page.relative_to(SITE).as_posix().startswith("ko/"):
            continue
        for value in attr.findall(page.read_text(encoding="utf-8")):
            if hangul.search(value):
                wrong.append(f"{page.relative_to(ROOT)}: \"{value[:40]}\"")
    # What the scripts write is checked too — it never reaches the page, so sweeping pages misses it.
    setter = re.compile(r'setAttribute\(\s*"(?:aria-label|title)"\s*,\s*"([^"]*[가-힣][^"]*)"')
    for js in sorted((SITE / "assets").glob("*.js")):
        for value in setter.findall(js.read_text(encoding="utf-8")):
            wrong.append(f"{js.relative_to(ROOT)}: \"{value[:40]}\" — written regardless of language")
    assert not wrong, ("Korean sits where the eye never reaches, on the English side:\n  " + "\n  ".join(wrong))


def test_the_english_reference_carries_no_korean():
    """Nothing Korean reaches the English API page — descriptions, tags or section names.

    The descriptions were carried across and reported as clean while **the `@param` texts
    stayed Korean**, and the page shipped that way. The count that was reported measured
    `.prose` elements; tags render elsewhere, so the check had quietly chosen an input
    that could not see them. That is the third time in one day a check picked its own
    input, and it is why this one walks the whole file rather than one field of it.

    Section names are here for the same reason. They come from the `// ── … ──` markers in
    the source, and a marker with no entry in `SECTION_EN` falls back to the Korean — so a
    new section added upstream puts Korean on the English page without anything failing.
    """
    api = json.loads((SITE / "assets" / "api.json").read_text(encoding="utf-8"))
    hangul = re.compile(r"[가-힣]")
    wrong = []

    def walk(node, where):
        if isinstance(node, dict):
            name = node.get("name", "")
            here = f"{where}/{name}" if name else where
            if hangul.search(node.get("doc") or ""):
                wrong.append(f"{here}: the description is Korean")
            for tag in node.get("tags") or []:
                if hangul.search(tag.get("text", "")):
                    wrong.append(f"{here}: @{tag.get('tag')} is Korean")
            section = node.get("section")
            if isinstance(section, dict) and hangul.search(section.get("en", "")):
                wrong.append(f"{here}: section \"{section['ko']}\" has no English name "
                             "— add it to SECTION_EN in site/build_api.py")
            for key, value in node.items():
                if key not in ("section", "tags"):
                    walk(value, here)
        elif isinstance(node, list):
            for item in node:
                walk(item, where)

    walk(api["modules"], "")
    seen = sorted(set(wrong))
    assert not seen, ("Korean reaches the English API reference:\n  "
                      + "\n  ".join(seen[:15])
                      + (f"\n  … and {len(seen) - 15} more" if len(seen) > 15 else ""))


DECLINED = re.compile(r"(\d[\d,]*) deliberately not carried across")
OWED = re.compile(r"and (\d[\d,]*) owed")
# The marker that separates the two. It sits at the head of every reason string in
# `borch-ts/test/run.py`'s ledger, and `test_alias_rows.py` reads the same position for
# `별칭` — the markers are keys into that table, not prose, which is what makes them
# readable from here. `아직` is the one that means the work is owed rather than declined.
OWED_MARKER = "아직"


def _ledger_split():
    """(declined, owed) as **the ledger itself has them.**

    Read by importing the runner rather than by matching its text. `test_alias_rows.py`
    does match text, with `\\((\\d+),\\s*"([^"]*)"\\)`, and that expression **cannot see a
    row whose reason is split across two lines** — `unpool::` is one, 20 cases, and it
    has been invisible to that reader the whole time it has existed. Importing has no
    such blind spot and costs one module load this file already pays for.

    The marker is read as **a leading word**, for the reason written beside the same read
    in `test_alias_rows.py`: `dtype::`'s reason contains "형 별칭", where the same word
    means something else, and matching anywhere in the body caught it.
    """
    rows = _runner().NOT_PORTED.values()
    owed = sum(n for n, why in rows if why.lstrip("*").startswith(OWED_MARKER))
    return sum(n for n, _ in rows) - owed, owed


def test_the_readme_splits_the_remainder_the_way_the_ledger_does():
    """**340 declined and 107 owed are not typed into the README — they are read.**

    The two mean opposite things. Declined is a decision that has been made: porting
    those would ask a question borch.ts has no place to be asked. Owed is work that has
    not happened. A single remainder cannot show a debt being paid and taken on at the
    same time, and the runner's ledger learned that years' worth of prefixes ago, which
    is why every row there carries the marker this reads.

    **The check exists because writing the split by hand went wrong twice in one day.**
    The first sentence said the remainder was "all one thing now" and a new block was
    frozen an hour later. The replacement said 376 and 71 and was wrong when it was
    written — `ops::` and `unpool::` are marked owed too and only `v2::` was counted.
    Both times the **total** was right, and the total was all that was checked, so both
    readings were green. This is the same shape as `test_docs.py`'s own lesson that a
    check on a number does not read the sentence beside it; the answer here is not a
    better sentence but not retyping a number the ledger already holds.
    """
    declined, owed = _ledger_split()
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    said_declined, said_owed = DECLINED.search(readme), OWED.search(readme)
    assert said_declined and said_owed, (
        "the README stopped stating the split in the shape this reads — fix the "
        "patterns above, or drop this test if the claim went away.")
    got = (int(said_declined.group(1).replace(",", "")),
           int(said_owed.group(1).replace(",", "")))
    assert got == (declined, owed), (
        f"the README splits the remainder {got[0]} declined / {got[1]} owed; the "
        f"ledger has {declined} / {owed}.\n"
        f"  owed is every row in borch-ts/test/run.py whose reason starts `{OWED_MARKER}`.")


def test_the_ledger_and_the_measured_remainder_are_the_same_number():
    """The two halves have to add back up to what `dist` measures.

    They come from opposite directions — one is `golden − written`, counted by loading
    the compiled case table, and the other is the sum of rows somebody wrote by hand.
    **Nothing has been comparing them.** A prefix could be double-counted in the ledger,
    or a row's frozen number could drift, and the per-prefix check in `run.py` would
    still pass every row it looks at while the total said something else.
    """
    stale = _stale_dist()
    if stale:
        pytest.skip("the bundle is stale; the measured half cannot be trusted")
    node = shutil.which("node")
    if node is None or not TS_CASES.exists():
        pytest.skip("no node, or the bundle has not been built")
    out = subprocess.run([node, "--input-type=module", "-e", COUNT_CASES],
                         capture_output=True, text=True, cwd=ROOT,
                         env={**os.environ, "GOLDEN": str(GOLDEN_JSON),
                              "CASES": TS_CASES.as_uri()})
    assert out.returncode == 0, f"could not load the case table:\n{out.stderr[-2000:]}"
    got = json.loads(out.stdout)
    measured = got["golden"] - got["written"]
    declined, owed = _ledger_split()
    assert declined + owed == measured, (
        f"the ledger's rows add to {declined + owed} and the case tables leave "
        f"{measured} unasked ({got['golden']} golden − {got['written']} written).\n"
        "  a prefix is counted twice, missing, or carrying a stale frozen number.")
