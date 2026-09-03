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


REACHABLE = """
import { readFileSync } from 'node:fs';
const index = await import(process.env.INDEX);
const doc = JSON.parse(readFileSync(process.env.API, 'utf8'));

// Every name any import path arrives at, walking namespaces to the bottom —
// `nn.functional` is a namespace inside a namespace and its 93 names reach through it.
const reachable = new Set();
const walk = (space, depth) => {
  for (const [name, value] of Object.entries(space)) {
    reachable.add(name);
    if (depth > 0 && value && typeof value === 'object') walk(value, depth - 1);
  }
};
walk(index, 3);

// **A type is erased, so the emit cannot be asked about it.** `DType` is exported
// from the index as `export type { DType }` and reaches an `import type` perfectly
// well, while `Object.keys` of the module never sees it — the first draft of this
// check called `dtype` unreachable for that reason alone. So the source of the index
// is read for its type-only exports and they count as arriving.
const src = readFileSync(process.env.INDEX_SRC, 'utf8');
for (const block of src.matchAll(/export\s+type\s*\{([^}]*)\}/g)) {
  for (const part of (block[1] ?? '').split(',')) {
    const name = part.trim().split(/\s+as\s+/).pop()?.trim();
    if (name) reachable.add(name);
  }
}

const missing = {};
for (const mod of doc.modules) {
  const names = mod.symbols.map((s) => s.name);
  if (names.length && !names.some((n) => reachable.has(n))) missing[mod.name] = names;
}
console.log(JSON.stringify(missing));
"""


def test_every_documented_module_is_reachable_from_the_index():
    """**A name the reference lists has to have an import path.**

    `ops.ts` existed, `site/build_api.py` listed 159 names out of it, `site/vision.html`
    described the section — and `index.ts` never exported the module. `import { ops }
    from "borch-ts"` gave `undefined`, and it was not under `vision` either. It was
    found by somebody writing a playground example with `nms`, which is the wrong way
    round: the reference is generated from the declaration files and **never asks
    whether a module is exported**, so every check on it stayed green while the
    documented surface was unreachable.

    That is the second half of one failure. The first was `ops` missing from the
    generator's own module list, which put its golden cases on no name axis at all;
    fixing that made the names visible to the reference and still not to a caller.
    Neither half was being asked this question.

    **It asks about the module, not about each name, and that is deliberate.** The
    per-name form was written first and flagged seventeen modules. Most of what it
    caught was one of two things it cannot tell apart: a `export type` is erased at
    run time and unreachable to `Object.keys` while `import type` reaches it fine, and
    a handful of names (`random`'s `gauss`, `rnn`'s `rnnApply`) are exported from their
    file for their neighbours and are internal on purpose. Forcing those into the index
    to satisfy a check would be worse than the hole it was written for. A module with
    **no** reachable name is unambiguous — 0 of 159 for `ops` — and it is the shape the
    failure actually took.

    Namespaces are walked to the bottom: `nn.functional` is a namespace inside a
    namespace and its 93 names reach through it. Members (`Tensor.mul`) reach through
    their class and are not counted here.
    """
    if not API.exists():
        pytest.fail("site/assets/api.json is missing — python3 site/build_api.py")
    entry = ROOT / "borch-ts" / "dist" / "src" / "index.js"
    if not entry.exists():
        pytest.skip(f"no {entry.relative_to(ROOT)} — run npm run build:ts first")
    node = shutil.which("node")
    if node is None:
        pytest.skip("no node")
    stale = _stale_dist()
    if stale:
        pytest.fail(
            "the bundle is older than the source, so this would judge an index that\n"
            "no longer exists:\n\n  " + stale.replace("\n", "\n  "))

    out = subprocess.run([node, "--input-type=module", "-e", REACHABLE],
                         capture_output=True, text=True, cwd=ROOT,
                         env={**os.environ, "API": str(API), "INDEX": entry.as_uri(),
                              "INDEX_SRC": str(ROOT / "borch-ts" / "src" / "index.ts")})
    assert out.returncode == 0, f"could not load the index:\n{out.stderr[-2000:]}"
    missing = json.loads(out.stdout)

    report = "\n".join(
        f"  {mod} — {len(names)} name(s), first: {' '.join(sorted(names)[:4])}"
        for mod, names in sorted(missing.items()))
    assert not missing, (
        "the API reference documents names no import reaches:\n" + report +
        "\n\nEither export the module from `borch-ts/src/index.ts`, or take it out of\n"
        "the list in `site/build_api.py` — a reference that lists an unreachable\n"
        "surface beside a reachable one is worse than one that omits it.")


def test_site_examples_name_only_real_modules():
    """The module list the site writes down for loading the Python binding has to match reality.

    `site/assets/runner.js` names the `.py` files it lays onto Pyodide's virtual
    filesystem. One left out blows up **loudly as an ImportError**, so that side shows
    itself; the other side — a file gone from the package while the name stays in the
    list — makes fetch return 404 and `runner.js` turn it into an exception. Both are
    known only after the user presses Run. This looks first.

    **Every quoted name counts, not the underscored ones.** The reader used to keep
    only parts starting with `_` (plus `__init__`), which was every module either
    package had at the time — so the first public one, `autograd`, was invisible to
    the left-hand side of the comparison and could never be reconciled: added to the
    list it still read as forgotten, and the only way to green was to delete the
    module. A parser that cannot see a name cannot check it, which is this
    repository's most-repeated shape.
    """
    runner = (ROOT / "site" / "assets" / "runner.js").read_text(encoding="utf-8")
    block = runner[runner.index("const PACKAGES = {"):runner.index("let pyodide")]
    for package in ("borch", "borch_webgpu"):
        listed = set()
        chunk = block[block.index(f"{package}:"):]
        for line in chunk.splitlines():
            listed.update(re.findall(r'"(\w+)"', line))
            if "]" in line:
                break
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


def test_every_page_that_loads_the_packages_lists_the_same_modules():
    """**Three pages carry that list and only one of them was checked.**

    `site/assets/runner.js` is the playground, `tests/browser/runner.html` is the
    golden runner and `tests/browser/scope_escape.html` is the leak probe. The check
    above reads the first against what is on disk; the other two were free to drift,
    and they did — a new module went into the playground's list and the golden runner
    stopped with *cannot import name … from partially initialized module*, a sentence
    about circular imports for a file that was simply never laid down. It cost a
    browser round trip to see, and the pages that run least often are the ones that
    would find out last.

    The lists are compared to each other rather than each to disk: whichever is
    right, they have to be the same, and a mismatch names both sides.
    """
    pages = {
        "site/assets/runner.js": "let pyodide",
        "tests/browser/runner.html": "const FILES",
        "tests/browser/scope_escape.html": "try {",
    }
    found = {}
    for rel, ends in pages.items():
        text = (ROOT / rel).read_text(encoding="utf-8")
        start = text.index("PACKAGES = {")
        block = text[start:text.index(ends, start)]
        per = {}
        for package in ("borch", "borch_webgpu"):
            chunk = block[block.index(f"{package}:"):]
            names = set()
            for line in chunk.splitlines():
                names.update(re.findall(r'"(\w+)"', line))
                if "]" in line:
                    break
            per[package] = names
        found[rel] = per

    first = next(iter(found))
    for rel, per in found.items():
        for package, names in per.items():
            assert names == found[first][package], (
                f"{rel} and {first} disagree about which {package} modules to load:\n"
                f"  only in {rel}: {sorted(names - found[first][package])}\n"
                f"  only in {first}: {sorted(found[first][package] - names)}\n"
                "  all three pages lay the same package onto the same virtual "
                "filesystem; a module in one list and not another is an ImportError "
                "only whoever opens that page will see.")


def test_the_site_deploys_every_root_file_the_runner_fetches():
    """A `.py` at the repository root that `runner.js` fetches has to be **deployed.**

    The check above this one covers `PACKAGES`, and `borchvision.py` is not in it: it is
    not a package, it is one file fetched by name beside the loop. So it fell outside the
    only rule there was, which is this repository's most-repeated shape — what is off the
    list has no rule.

    The consequence is real rather than theoretical. Deployment copies that file in one
    line of `pages.yml`, and **deleting that line breaks nothing anybody can see until a
    reader types `import borchvision` in the playground.**

    It reads both sides out of their files rather than naming `borchvision.py` here. A
    second file fetched the same way tomorrow is then covered on the day it is written,
    which is exactly what did not happen the first time.
    """
    runner = (ROOT / "site" / "assets" / "runner.js").read_text(encoding="utf-8")
    fetched = set(re.findall(r"fetch\(`\$\{repo\}([A-Za-z_]\w*\.py)`\)", runner))
    assert fetched, ("no root-level `.py` fetch found in runner.js — if that loading step "
                     "moved, this check has to follow it rather than quietly pass")

    missing = sorted(name for name in fetched if not (ROOT / name).exists())
    assert not missing, f"runner.js fetches files the repository does not have: {missing}"

    deploy = (ROOT / ".github" / "workflows" / "pages.yml").read_text(encoding="utf-8")
    copied = [line for line in deploy.splitlines() if line.strip().startswith("cp ")]
    undeployed = sorted(name for name in fetched
                        if not any(name in line for line in copied))
    assert not undeployed, (
        f"the runner fetches {undeployed} and the deployed site never receives them.\n"
        "  add them to the `cp` line in .github/workflows/pages.yml — left out, the "
        "playground stops on a 404 the moment somebody imports them.")


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
    ("docs/BOOK.md", "ES module", "bundle"),
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
    ("site/learn/08-debugging.html", "docs/BOOK.md", "Seven places where green can be a lie"),
    ("site/ko/learn/08-debugging.html", "docs/BOOK.md", "Seven places where green can be a lie"),
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

    readme = (ROOT / "docs" / "BOOK.md").read_text(encoding="utf-8")
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
    # `site/lab/` is JupyterLite (built by site/build_lab.py, gitignored) — its pages are
    # not this site's, and `site/lab-src/` is its source. Neither is checked here.
    return sorted(p for p in SITE.rglob("*.html") if "lab" not in p.relative_to(SITE).parts[:1] and "lab-src" not in p.parts)


# **Which adapter names mean "this is the CPU" had three homes and now has two.**
#
# `home.js` and `playground.js` each carried their own copy. Both are JavaScript and
# both already load `borch-ts/dist`, so the rule moved into the library — `probe()`
# carries `software` beside the adapter name and the pages read it. Two copies gone,
# and the two that remain are on opposite sides of a language boundary no import
# crosses.
_SOFTWARE_RULE = {
    "tests/browser/launch.py": re.compile(r'_SOFTWARE\s*=\s*re\.compile\(r"([^"]+)"'),
    "borch-ts/src/device.ts": re.compile(r"const SOFTWARE\s*=\s*/([^/]+)/"),
}


# **The ladder is written down twice** — once in the README and once on the setup page in
# two languages — and both say which rung is the list `tests/browser/launch.py` ships.
# Two readers of the same rule is how the rule drifts, which is the defect this very
# check exists to catch, so the reading happens once here.
_LADDER_FLAG = re.compile(r"--[a-z0-9-]+(?:=[A-Za-z0-9,._-]+)?")


def _marked_rung(rows):
    """The accumulated flag set at the row that claims to be `FLAGS`, or None.

    **The rungs accumulate**: a cell starting with `+` means everything above it too, so
    the claim on the marked row is about the running set and not about that cell alone.
    `rows` is (cell text, is it marked) in table order.
    """
    running = []
    for cell, marked in rows:
        flags = _LADDER_FLAG.findall(cell)
        if not flags:
            running = []
            continue
        running = (running + flags) if cell.lstrip("`| ").startswith("+") else list(flags)
        if marked:
            return list(running)
    return None


def test_the_ladder_row_that_says_it_is_FLAGS_is_FLAGS():
    """**A row labelled `(= FLAGS)` has to be `FLAGS`.**

    The README carries a ladder of Chrome flag sets against what adapter each one reaches,
    and one rung is marked as the list `tests/browser/launch.py` actually ships. That mark
    is the whole load-bearing part: every other rung is a diagnostic step, and this one is
    a claim about the code.

    ## It is here because that exact claim was wrong twice in two commits

    First: a peer measured a rung labelled `--ignore-gpu-blocklist`, the shipped list also
    carries `--enable-features=Vulkan`, and a conclusion about *the shipped list* was drawn
    from a run that never had the flag doing the work. The shipped list sat four lines
    below the table in the file being edited and the two were never put side by side.

    Then, in the commit that fixed it, the rung was labelled `(= FLAGS)` **and left one
    flag short** — `--disable-gpu-driver-bug-workarounds`. The label was added as the
    structural fix for the first mistake and immediately made the second one, which is
    what a label costs when nothing reads it.

    So this reads it. The rungs accumulate — a cell starting with `+` adds to the running
    set — and the marked row's accumulated set is compared to `launch.FLAGS`.

    Naming a thing is not comparing to it, and a comparison a person performs once is a
    comparison that happened once.
    """
    launch = _load_module("bt_launch", ROOT / "tests" / "browser" / "launch.py")
    lines = (ROOT / "docs" / "BOOK.md").read_text(encoding="utf-8").splitlines()

    marked = _marked_rung(
        (line.split("|")[1].strip(), "= `FLAGS`" in line or "(= FLAGS)" in line)
        for line in lines if line.startswith("|"))

    assert marked is not None, (
        "no rung in the README's flag ladder is marked as the shipped list.\n"
        "  Either the table went away or the mark did. The mark is the only thing tying\n"
        "  a measured row to the flags this repository actually sends, so losing it\n"
        "  quietly is how a conclusion gets drawn about a list nobody ran.")
    assert sorted(marked) == sorted(launch.FLAGS), (
        "the README's ladder row marked as the shipped list is not the shipped list:\n"
        f"  README: {sorted(marked)}\n"
        f"  FLAGS : {sorted(launch.FLAGS)}\n"
        f"  only in README: {sorted(set(marked) - set(launch.FLAGS))}\n"
        f"  only in FLAGS : {sorted(set(launch.FLAGS) - set(marked))}\n"
        "  A row that says it is FLAGS is a claim about the code, and every verdict on\n"
        "  that row was measured under whatever it actually lists.")


def test_the_software_adapter_rule_says_the_same_thing_in_every_copy():
    """**Two copies of one judgement, and nothing but this compares them.**

    `refuse_if_software` in `launch.py` decides that a benchmark measured on this
    adapter is void. `isSoftwareAdapter` in `device.ts` decides what `probe()` reports
    and therefore what a published page tells a visitor about their own machine. Same
    list, two files, two languages.

    The list is short and looks stable, which is exactly why it drifts: a name gets
    added where the symptom was met and not in the other, and neither side raises
    anything.

    Merging them is not available — a Python test harness cannot import a TypeScript
    module. Saying so loudly is. **Two of the three copies were merged rather than
    checked**, which is the better fix wherever it is reachable, and it was reachable
    for exactly the pair that shared a language.
    """
    found = {}
    for rel, pattern in _SOFTWARE_RULE.items():
        path = ROOT / rel
        if not path.exists():
            continue
        got = pattern.search(path.read_text(encoding="utf-8"))
        assert got, (
            f"{rel} no longer spells the software-adapter list where this looks for it.\n"
            "  Either it moved or it is gone. Both matter: the judgement is the same one\n"
            "  in three places, and this is the only thing that compares them.")
        found[rel] = frozenset(got.group(1).split("|"))
    assert len(set(found.values())) == 1, (
        "the software-adapter list has diverged:\n  "
        + "\n  ".join(f"{rel}  {' '.join(sorted(names))}"
                      for rel, names in found.items()) + "\n\n"
        "  One decides whether a measurement is void; the other two decide what a\n"
        "  published page tells a visitor about their own machine. A name in one and\n"
        "  not the others means one of them is wrong about a GPU and neither says so.")


def _bench():
    """`borch-ts/test/bench.py`, loaded as a module.

    It needs **both** directories on the path for the load — `import run` is
    `borch-ts/test/run.py` and `from launch import` is `tests/browser/launch.py` — and
    neither may stay there afterwards, for the reason `_runner` gives.

    Nothing here starts a browser: `playwright` is imported inside `main`, so the module
    loads on stdlib alone and this check runs everywhere the suite does.
    """
    import importlib.util                                            # noqa: PLC0415

    added = [str(ROOT / "borch-ts" / "test"), str(ROOT / "tests" / "browser")]
    sys.path[:0] = added
    try:
        spec = importlib.util.spec_from_file_location(
            "bt_ts_bench", ROOT / "borch-ts" / "test" / "bench.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        for here in added:
            if here in sys.path:
                sys.path.remove(here)
    return mod


def test_the_bench_conditions_line_names_the_machine_and_nobody_else():
    """**That line is written to be pasted into a public README.**

    The bench table's three columns were measured *side by side on the same machine* and
    that machine is recorded nowhere — two sessions went looking and neither found it, so
    the milliseconds now stand only as ratios. `bench.py` prints a `measured on:` line to
    stop that recurring: the adapter and the host, under every number it lets stand.

    Which makes it a line that travels **outward**. `platform.node()` is one keystroke
    from `platform.machine()` and returns the hostname; `platform.uname()` reads like the
    obvious upgrade and contains it. Either one publishes the operator's machine name into
    a document that is on the public web, and a README cannot be un-pushed.

    So this asks the function what it actually returns and looks for the two values that
    must never be in it. A source pattern would not do — it would have to guess every
    spelling of the mistake, and it is the **output** that gets pasted.

    It does not check the adapter is present: `refuse_if_software` already decides what
    counts as a measurable device, and a second opinion on that here would drift from it.
    """
    import getpass                                                   # noqa: PLC0415
    import platform                                                  # noqa: PLC0415

    line = _bench().conditions("NVIDIA GeForce RTX 5080")
    assert isinstance(line, str) and line, "conditions() returned nothing to paste"

    # Short values are skipped rather than matched: a two-character login name would hit
    # inside `arm64` and this would fail on a machine that is doing nothing wrong.
    private = {"hostname": platform.node(), "user": _quietly(getpass.getuser)}
    leaked = {what: v for what, v in private.items() if len(v or "") > 3 and v in line}
    assert not leaked, (
        f"the bench's `measured on:` line carries {', '.join(leaked)} — "
        f"{' '.join(leaked.values())}\n"
        f"  got: {line}\n"
        "  That line exists to be pasted into README.md, which is on the public web and\n"
        "  cannot be un-pushed. The machine is the point; the operator is not. Use\n"
        "  platform.system() and platform.machine() — not node(), not uname().")


def _quietly(fn):
    """The value, or empty where the environment has none to give.

    `getpass.getuser()` raises rather than returning a blank when every one of the
    variables it consults is unset and there is no password-database entry — which is a
    container, and containers are where this suite runs. **A check that errors on the
    machine it was meant to protect is not protecting it.**
    """
    try:
        return fn()
    except Exception:
        return ""


# ── where the site counts its own pages ───────────────────────────────
#
# **The count is written in words and the pages are on disk, and nothing was comparing
# them.** Six places said `eight lessons` or `여덟 강` while `site/learn/` held ten, and
# one file disagreed with itself: `learn/index.html` carried `Ten lessons` in its heading
# and `Six lessons` in the meta description a hundred lines above it, so the page told a
# reader one number and a search engine another.
#
# It is the same shape as the stale golden count `tests/test_docs.py` was built for, and
# the same shape as the vendor sentence that outlived its measurement: **a number that no
# longer has to agree with anything.** Adding a lesson is the moment it goes wrong, and
# adding a lesson is exactly when nobody is reading the landing page.
#
# Words rather than digits, because that is how this site writes them.
_EN_NUMBER = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20,
}
_KO_NUMBER = {
    "하나": 1, "둘": 2, "셋": 3, "넷": 4, "다섯": 5, "여섯": 6, "일곱": 7, "여덟": 8,
    "아홉": 9, "열": 10, "열하나": 11, "열둘": 12, "열셋": 13, "열넷": 14, "열다섯": 15,
    "열여섯": 16, "열일곱": 17, "열여덟": 18, "열아홉": 19, "스물": 20,
}

# (regex, which directory the number is claiming, the word table)
_COUNT_CLAIMS = (
    (re.compile(r"([A-Za-z]+)\s+lessons"), "learn", _EN_NUMBER),
    (re.compile(r"([A-Za-z]+)\s+(?:tutorials|projects)"), "tutorials", _EN_NUMBER),
    (re.compile(r"([가-힣]+)\s*강[\s,.·<]"), "learn", _KO_NUMBER),
    (re.compile(r"프로젝트\s+([가-힣]+)"), "tutorials", _KO_NUMBER),
)


# **Where a count lives, and where a similar-looking phrase does not.** A count of the
# site's own pages is a *label* — it sits in a title, a heading, a meta description or a
# table cell. In a paragraph it is prose, and prose says things like *"Two lessons, one
# page"* on `tutorials/03-curve-fitting.html`, which means two things learned and not two
# pages on disk. Reading paragraphs too made this check fail on that sentence, which is
# how a check earns the reputation that gets it switched off.
_LABEL = re.compile(
    r"<title>(.*?)</title>"
    r"|<meta[^>]*content=\"([^\"]*)\""
    r"|<h[123][^>]*>(.*?)</h[123]>"
    r"|<td[^>]*>(.*?)</td>", re.S)


def _labels(text):
    """Every label on the page, with the offset it starts at — for the line number."""
    for m in _LABEL.finditer(text):
        for group in m.groups():
            if group:
                yield m.start(), group


def _pages_in(directory):
    """The lessons or projects themselves — the index is the shelf, not a book on it."""
    return len([p for p in (SITE / directory).glob("*.html") if p.name != "index.html"])


def test_the_site_counts_the_pages_it_links_to():
    """Every `N lessons` and `N projects` on the site has to be the N that is on disk.

    Only pages are read, not the READMEs: what is checked here is what a visitor is told.
    A word that is not a number (`those lessons`, `four ways`) is not a claim and is
    skipped — the tables above decide what counts as one.
    """
    truth = {"learn": _pages_in("learn"), "tutorials": _pages_in("tutorials")}
    assert truth["learn"] and truth["tutorials"], f"no pages found to count: {truth}"

    wrong, seen = [], 0
    for page in _pages():
        text = page.read_text(encoding="utf-8")
        for at, label in _labels(text):
            for pattern, which, table in _COUNT_CLAIMS:
                for hit in pattern.finditer(label):
                    said = table.get(hit.group(1).lower())
                    if said is None:
                        continue                   # not a number — not a claim
                    seen += 1
                    if said != truth[which]:
                        line = text[:at].count("\n") + 1
                        wrong.append(f"{page.relative_to(ROOT)}:{line}  "
                                     f"{hit.group(0).strip()!r} — there are {truth[which]}")

    assert seen > 4, (
        f"only {seen} count claims were recognised — the site writes them differently now "
        "and this check is reading past them, which is the silent half of this failure.")
    assert not wrong, (
        "the site claims a number of pages it does not have:\n  " + "\n  ".join(wrong) +
        f"\n\non disk: {truth['learn']} lessons, {truth['tutorials']} projects")


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
            # `site/lab/` is JupyterLite, built by `site/build_lab.py` and gitignored — the
            # deploy builds it beside the pages. A link into it is checked by
            # `tests/browser/lab_probe.py`, which opens the built notebook and runs it.
            if raw.lstrip("./").startswith("lab/") or "/lab/" in raw:
                continue
            target = (page.parent / raw.split("#")[0].split("?")[0]).resolve()
            if target.is_dir():
                target = target / "index.html"
            if not target.exists():
                missing.append(f"{page.relative_to(ROOT)} → {raw}")
    assert not missing, "the site has broken links:\n  " + "\n  ".join(missing)


def test_no_page_wears_a_class_that_styles_nothing():
    """A class name that no rule defines is **a broken link on the style axis.**

    The check above reads `href`; this one reads `class`. Both ask the same question — does
    this reference point at something that exists — and both catch a defect **no behavioural
    test can see**, because the page does exactly what it was written to do and only looks
    wrong. A peer found one of these by taking a screenshot: a button carrying `class="run"`
    where no `.run` rule existed. It worked. It just did not look like a button.

    Ours was `class="lede"` on the two Python pages, where the other fifty-two say `lead` —
    the same word, two spellings, and the one that is defined is `lead`. The opening
    paragraph rendered as plain body text at full column width on both.

    JavaScript is read as a definition too: `classList.add("x")`, `className = "x"`, and
    `querySelector(".x")` all mean the name is alive even when the stylesheet is not where
    it is spelled out. Without that this check would be **loud in a way that gets it
    switched off**, which is worse than not having it.

    The other direction — a rule nobody wears — is deliberately *not* asserted. Most of this
    site's markup is built at runtime from `api.json`, so the classes in that markup never
    appear in a `class="…"` attribute on disk and the reading would be false against dozens
    of live rules.
    """
    css = "".join(sheet.read_text(encoding="utf-8") for sheet in sorted(SITE.rglob("*.css")))
    scripts = "".join(sheet.read_text(encoding="utf-8") for sheet in sorted(SITE.rglob("*.js")))
    for page in _pages():
        text = page.read_text(encoding="utf-8")
        css += "".join(re.findall(r"<style[^>]*>(.*?)</style>", text, re.S))
        scripts += "".join(re.findall(r"<script[^>]*>(.*?)</script>", text, re.S))

    defined = set(re.findall(r"\.([A-Za-z_][-\w]*)", css))
    for chunk in (re.findall(r"classList\.\w+\(\s*[\"'`]([-\w]+)", scripts)
                  + re.findall(r"className\s*=\s*[\"'`]([-\w ]+)", scripts)
                  + re.findall(r"querySelector(?:All)?\(\s*[\"'`]\.([-\w]+)", scripts)):
        defined.update(chunk.split())

    assert "lead" in defined and "doc-main" in defined, (
        "the stylesheet is not being read — every class would look undefined and this check "
        "would fail on all of them, which reads as a broken site rather than a broken test")

    dangling = {}
    for page in _pages():
        for attr in re.findall(r'class\s*=\s*"([^"]*)"', page.read_text(encoding="utf-8")):
            for name in attr.split():
                if name not in defined:
                    dangling.setdefault(name, set()).add(page.relative_to(ROOT).as_posix())
    assert not dangling, (
        "these class names style nothing — the markup asks for an appearance that has no "
        "rule behind it:\n  " +
        "\n  ".join(f".{name} — {', '.join(sorted(where))}" for name, where in sorted(dangling.items())))


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
NUMBERED = re.compile(r"^\d\d-")


def test_a_section_sidebar_lists_every_page_of_its_section():
    """A section's sidebar has to list **the pages that are actually there**.

    The sidebar drifted the way an unchecked list drifts. `site/learn/06-save-load.html`
    and its Korean twin listed six lessons while eight existed, so a reader who arrived at
    lesson 6 was shown a course that ended there. **Nothing was broken**: every link on the
    page worked, the page rendered, and the link checker above was satisfied, because a
    link that is simply absent is not a broken link.

    The first version of this check compared the sidebars **to each other** and complained
    when one disagreed with the majority. That would have caught this bug, and it would
    have been the wrong instrument: pages that all forget the same lesson agree perfectly,
    and the majority is whichever mistake was copied more times. So it asks the directory
    instead — the sidebar is a claim about which pages exist, and the pages exist on disk.

    Order is compared too. The files are numbered, so the order they sort in is the order
    a reader is meant to walk them.
    """
    problems = []
    for folder in sorted({page.parent for page in _pages()}):
        expected = sorted(f.name for f in folder.glob("*.html") if NUMBERED.match(f.name))
        if not expected:
            continue
        for page in sorted(folder.glob("*.html")):
            found = SIDEBAR.search(page.read_text(encoding="utf-8"))
            if not found:
                problems.append(f"{page.relative_to(ROOT)}: in a section but carries no sidebar")
                continue
            listed = [href for href in re.findall(r'<a href="([^"]+)"', found.group(1))
                      if NUMBERED.match(href)]
            if listed == expected:
                continue
            absent = [name for name in expected if name not in listed]
            extra = [name for name in listed if name not in expected]
            why = []
            if absent:
                why.append("never listed: " + ", ".join(absent))
            if extra:
                why.append("listed but not on disk: " + ", ".join(extra))
            if not why:
                why.append("listed out of order")
            problems.append(f"{page.relative_to(ROOT)}: " + "; ".join(why))

    assert not problems, ("a sidebar disagrees with the pages beside it:\n  "
                          + "\n  ".join(problems))


# ── where a page tells the reader a name is absent ─────────────────────
#
# A lesson that says "`AdaptiveAvgPool2d` is not here" is making a claim about **another
# file's contents**, and the usual way such a claim breaks is that somebody closes the gap.
# Then the page teaches a workaround for a problem that no longer exists, and the reader
# who tries the real name finds it works — worse than a missing feature, because it teaches
# distrust of the page.
#
# It has already broken once in the other direction, which is why the sentences below are
# the ones they are. The first version of the lesson read the gap list by name and told
# readers to use `AvgPool2d` with a fixed size. `AdaptiveAvgPool1d` and `AdaptiveAvgPool3d`
# were there the whole time with **identical bodies**, both calling `adaptivePool`, which
# ignores how many spatial axes it gets — the absent thing was a one-line alias. **A name
# missing from a gap list does not tell you the capability is missing**, and the workaround
# was worse than the real call: `AvgPool2d(8)` on a `[2, 16, 5, 7]` input returns
# `[2, 16, 0, 0]` and raises nothing.
#
# Both ends are checked, the same way the heading quotes above are: the page must still
# carry the sentence, and the name must still be absent. Neither side can move alone.

# The third column is a **positive control**: a name that must be found. A negative
# answer is only worth as much as the surface it was asked of, and that surface has a
# known hole — `site/build_api.py`'s `MODULES` does not list `functional`, so a name
# living only there is absent from the index and reads as absent from borch.ts. Without
# a control, a namespace dropping out of the index would make every claim here *pass
# harder*. The control is the sibling the lesson itself points at.

ABSENCES_A_PAGE_TEACHES = (
    # Empty since `nn.AdaptiveAvgPool2d` was written: the ResNet lesson taught the way
    # around it and now uses it. The table stays, for the next name a page has to
    # teach around — a row is (page, the sentence it must carry, the absent name, and
    # a sibling name that must be found).
)


def test_a_page_teaching_around_a_missing_name_is_still_missing_it():
    """The names lesson pages tell readers to work around have to still be gone.

    `tests/ts_axis.py` reads the generated name index, so this asks it rather than keeping
    a second list that would drift. At the time of writing, `nn`'s gap was being worked
    down from fifteen and `AdaptiveAvgPool2d` was one of the names in it.

    **That index is not the whole surface**, which the first version of this docstring
    said it was. `site/build_api.py` does not sweep `functional`, so a name living only
    there is missing from the index — and a check that reads "absent" off an incomplete
    list cannot tell a real absence from an unswept one. Hence the control below: the
    sibling that must be found makes the surface prove it can see this namespace before
    its silence about a name is believed.
    """
    import ts_axis

    surface = ts_axis.ts_names()
    problems = []
    for page_path, sentence, name, control in ABSENCES_A_PAGE_TEACHES:
        if control not in surface:
            problems.append(
                f"{control} is not in the name index either, so the index cannot see "
                f"this namespace and its silence about {name} means nothing")
        page = ROOT / page_path
        text = page.read_text(encoding="utf-8")
        if sentence not in text:
            problems.append(
                f"{page_path} no longer says {sentence!r} — "
                f"if the lesson was rewritten, this table is what tells you to update it")
        if name in surface:
            problems.append(
                f"{name} is in borch.ts now, and {page_path} still teaches around it")

    assert not problems, "\n  ".join([""] + problems)


# **The stack outgrew one repository.** This check was written when a link to any other
# repository under this owner meant a stale address — the site carried `browsertorch` for
# thirty-eight files after the rename, and the links still opened through a redirect, so
# nobody looked. That reason still holds; what changed is that some of those addresses are
# now correct. So the siblings are named here, with why the site points at each, and an
# address that is neither this repository nor one of these still fails.
SIBLING_REPOSITORIES = {
    "bimm": "the architecture catalog — timm's seat, where a manifest's `arch` resolves",
    "borch-hub": "the client that fetches a manifest, checks the hash and loads the model",
    "borch-hub-registry": "the manifests and the provenance note behind each model",
}


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
    allowed = {here} | {f"https://github.com/{owner}/{r}" for r in SIBLING_REPOSITORIES}
    wrong = []
    for page in _pages():
        for i, line in enumerate(page.read_text(encoding="utf-8").splitlines(), 1):
            for hit in linked.findall(line):
                if hit.removesuffix(".git") not in allowed:
                    wrong.append(f"{page.relative_to(ROOT)}:{i}  {hit} — not this repository "
                                 f"({here}) and not a named sibling")
    assert not wrong, (
        "the site points at a repository it has no name for:\n  " + "\n  ".join(wrong[:12]) +
        (f"\n  … and {len(wrong) - 12} more" if len(wrong) > 12 else "") +
        "\n  add it to SIBLING_REPOSITORIES with the reason, or fix the address.")


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
    readme = (ROOT / "docs" / "BOOK.md").read_text(encoding="utf-8")
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


# The setup page's Linux ladder carries a row that says it is the flag list the test
# runner sends. **Saying so is not checking it.** The row was written twice from memory
# and was wrong both times: once missing `--enable-features=Vulkan`, which made a list
# that opens both cards read as failing on one and put a retraction on the live site for
# an afternoon; then, in the correction, missing
# `--disable-gpu-driver-bug-workarounds` — inside the very label added to stop the first
# mistake. The page and `tests/browser/launch.py` are two lists nobody put side by side.
LADDER_LABEL = re.compile(r"this row is the runner's flags|이 행이 러너의 목록이다")


def _ladder_rows(page):
    """The first cell of each row of the Linux table, in order."""
    body = page[page.index("<tbody>", page.index("RTX 5080")):]
    body = body[:body.index("</tbody>")]
    for row in re.findall(r"<tr>(.*?)</tr>", body, re.S):
        cells = re.findall(r"<td[^>]*>(.*?)</td>", row, re.S)
        if cells:
            yield cells[0]


@pytest.mark.parametrize("name", ["setup.html", "ko/setup.html"])
def test_the_setup_pages_marked_rung_is_the_runners_flags(name):
    """The same claim as the README's, on the page a stranger actually reads.

    **The README's version of this row was wrong twice in two commits**, and the setup
    page carried the same table with the same omission — a copy-and-paste command line
    built from three flags where the runner sends four. The page is the more expensive
    of the two to get wrong: the README is read by whoever works on this repository, and
    the page is read by somebody whose card will not come up.

    A row can fail four ways and each has happened or nearly has: the label falls off,
    the row loses a flag, the row gains one the runner does not send, or the row is
    deleted and the claim slips into prose where nothing reads it.
    """
    launch = _load_module("bt_launch", ROOT / "tests" / "browser" / "launch.py")
    page = (ROOT / "site" / name).read_text(encoding="utf-8")

    marked = _marked_rung(
        (re.sub(r"<[^>]+>", " ", cell).replace("&nbsp;", " "), bool(LADDER_LABEL.search(cell)))
        for cell in _ladder_rows(page))

    assert marked is not None, (
        f"site/{name} has no ladder row marked as the runner's flags.\n"
        "  the claim has to sit on a row a test can read, not in a sentence beside it.")
    assert sorted(marked) == sorted(launch.FLAGS), (
        f"site/{name}'s marked rung and tests/browser/launch.py disagree.\n"
        f"  only on the page : {sorted(set(marked) - set(launch.FLAGS))}\n"
        f"  only in FLAGS    : {sorted(set(launch.FLAGS) - set(marked))}\n"
        "  the page tells a stranger which flags to paste; it has to be this list.")


# **The vision page counts names it does not own.** `site/assets/api.json` is generated
# from the built declarations, so every one of those numbers moves whenever the library
# grows — `ops` went from eleven names to seventeen the day before this page was written,
# and a sentence naming eleven would have survived that with nothing to contradict it.
#
# So the page marks each number where it makes the claim and this reads them. Prose is
# not searched: a page that says "seventeen" in a paragraph is describing, and a page
# that says `data-count="ops"` is asserting.
_COUNT_SPAN = re.compile(r'<span data-count="([a-z_0-9]+)">(\d+)</span>')
_VISION_MODULES = ("vision", "vision_v2", "vision_v2_twins", "ops", "datasets")


# **The guard has to follow the number, not the page.** The vision page kept `653` current
# because a test read its spans; the landing carried `547` for the same fact with nothing
# reading it, and drifted 106 names behind in half a day. Same repository, same day, same
# number — the difference was only whether anything looked.
#
# So this walks every page that marks a count rather than a list of pages, and a page that
# starts marking one is covered the moment it does.
def _pages_marking_counts():
    return [p for p in sorted((ROOT / "site").rglob("*.html")) if p.relative_to(ROOT / "site").parts[0] not in ("lab", "lab-src")
            if _COUNT_SPAN.search(p.read_text(encoding="utf-8"))]


@pytest.mark.parametrize("name", [p.relative_to(ROOT / "site").as_posix()
                                  for p in _pages_marking_counts()])
def test_a_marked_count_agrees_with_the_generated_reference(name):
    """Every marked number against `api.json`, wherever it is claimed.

    **Stale and wrong have to look different on screen.** If `dist` is behind the sources
    then `api.json` was generated from yesterday's library, and a page that disagrees with
    it may be the one telling the truth. Failing here would send somebody to edit a correct
    page until it matched a stale reference — the wrong direction, and it would look like
    progress. `test_api_reference_is_not_stale` owns that case; this one steps aside and
    says which of the two is speaking.
    """
    stale = _stale_dist()
    if stale:
        pytest.skip(f"the bundle is behind the sources, so api.json is not a reference yet "
                    f"— run `npm run build:ts && npm run docs:api` ({stale.splitlines()[0][:80]})")
    api = json.loads(API.read_text(encoding="utf-8"))
    counts = {m["name"]: m["count"] for m in api["modules"]}
    truth = {n: counts[n] for n in _VISION_MODULES}
    truth["_total"] = api["total"]
    truth["_vision_total"] = sum(counts[n] for n in _VISION_MODULES)

    page = (ROOT / "site" / name).read_text(encoding="utf-8")
    claimed = {k: int(v) for k, v in _COUNT_SPAN.findall(page)}

    assert claimed, (
        f"site/{name} marks no counts at all.\n"
        "  the numbers have to stay inside data-count spans, where this can read them.")
    unknown = sorted(set(claimed) - set(truth))
    assert not unknown, (
        f"site/{name} marks counts nothing can check: {unknown}\n"
        "  a `data-count` name has to be a module in api.json, `_total`, or\n"
        "  `_vision_total` — otherwise the span looks guarded and is not.")
    # 페이지가 주장한 것만 비교한다 — 첫 화면은 합계 하나만 주장하고
    # 모듈별 수는 주장하지 않는다. 안 한 주장까지 요구하면 검사가 페이지 모양을 정한다.
    wrong = {k: (claimed[k], truth[k]) for k in claimed if claimed[k] != truth[k]}
    assert not wrong, (
        f"site/{name} and site/assets/api.json disagree (page, reference):\n  " +
        "\n  ".join(f"{k}: {p} vs {t}" for k, (p, t) in sorted(wrong.items())) +
        "\n  regenerate with `npm run docs:api` and update the page, in that order.")


# **The nav is the only way to reach most of the site, and it grows.** It went from six
# entries to eight to ten; the CSS carries a measurement from the first time that broke —
# six items took 404px against a 390px viewport and pushed every page out by 125px. The
# fix was to let the bar scroll instead of overflowing the document, and it still holds:
# measured at 390px the bar scrolls and every entry is reachable, and from 900px up the
# whole bar fits with nothing to scroll.
#
# That measurement lives in a comment, and a comment only works on somebody who reads it.
# Adding the ninth and tenth entries, this session assumed the bar had broken, mis-read
# "extends past the viewport" as "cannot be reached", and came close to rewriting a rule
# that was doing its job. What stopped it was re-measuring — which is luck, not a guard.
#
# So the mechanism is pinned. This cannot see a rendered page; what it can see is whether
# the rule that makes the overflow scrollable is still there, and that is the part whose
# removal produced the documented 125px.
_MOBILE_NAV = re.compile(
    r"@media\s*\(max-width:\s*700px\)\s*\{(.*?)\n\}", re.S)


def test_the_menu_bar_can_still_be_scrolled_on_a_phone():
    """The nav has more entries than a phone fits, so it has to scroll rather than overflow."""
    css = (SITE / "assets" / "style.css").read_text(encoding="utf-8")
    block = _MOBILE_NAV.search(css)
    assert block, (
        "site/assets/style.css has no `@media (max-width: 700px)` block.\n"
        "  the narrow-screen nav rules lived there; without them a bar that does not fit\n"
        "  pushes the whole document sideways (measured once at 125px).")
    rules = block.group(1)
    nav = [line for line in rules.splitlines() if ".top nav" in line and "a" not in
           line.split("{")[0].split(".top nav")[1][:2]]
    assert nav, "no `.top nav` rule inside the narrow-screen block"
    joined = " ".join(nav)
    for prop in ("overflow-x: auto", "width: 100%"):
        assert prop in joined, (
            f"the narrow-screen `.top nav` rule lost `{prop}`:\n    {joined.strip()}\n"
            "  without it the bar overflows the document instead of scrolling inside\n"
            "  itself, and entries past the right edge become unreachable rather than\n"
            "  merely off-screen. Ten entries do not fit 390px and are not meant to.")
    assert "white-space: nowrap" in rules, (
        "`.top nav a { white-space: nowrap }` is gone — the entries wrap mid-word instead\n"
        "  of scrolling as a row.")


# **A run's size goes stale the way a count does, and nothing was reading it.** The pages
# report a GPU run as `agreeing N / N`, and N is the size of the case table on the day
# somebody ran it. Two days and fifteen commits later the table had grown by a hundred and
# twenty-two while the pages still said 3758 — found by sweeping by hand, which is the
# method this repository keeps replacing.
#
# **The live claim is marked rather than matched.** `setup.html` deliberately carries an
# older figure in the clause that says what it used to claim, so a net cast over the prose
# would catch the retraction it is supposed to preserve. `data-measured` says which one is
# speaking for today.
#
# The counting is borrowed, not restated — `tests/test_docs.py` already decides what the
# three legitimate totals are, and a second copy of that rule is a rule that diverges.
_MEASURED = re.compile(r'<code data-measured="golden">agreeing (\d+) / (\d+)')


@pytest.mark.parametrize("name", ["index.html", "setup.html",
                                  "ko/index.html", "ko/setup.html"])
def test_a_reported_run_covers_the_cases_that_exist_now(name):
    """`agreeing N / N` has to be a run over the table as it stands."""
    docs = _load_module("bt_docs", ROOT / "tests" / "test_docs.py")
    total, core, bind = docs._counts()
    page = (ROOT / "site" / name).read_text(encoding="utf-8")

    hit = _MEASURED.search(page)
    assert hit, (
        f"site/{name} has no `data-measured=\"golden\"` run to check.\n"
        "  the figure has to stay marked, or the next reader of this page cannot tell\n"
        "  which of its numbers is a claim about today and which is a retracted one.")
    asked, agreed = int(hit.group(1)), int(hit.group(2))
    assert asked == agreed, (
        f"site/{name} reports {agreed} of {asked} agreeing.\n"
        "  a run with failures is not a thing to advertise on a landing page; say what\n"
        "  diverged and why, as the page did when two cases were about the machine.")
    assert asked in (total, core, bind), (
        f"site/{name} reports a run over {asked} cases; the table now holds "
        f"{total} ({core} for the core, {bind} through the binding).\n"
        "  the measurement is older than the thing it measured. Re-run it on a real GPU\n"
        "  and put the new figure here — editing the number alone would be a claim\n"
        "  about a run nobody made.")
