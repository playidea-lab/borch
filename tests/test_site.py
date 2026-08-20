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
import pathlib
import re
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
GENERATOR = ROOT / "site" / "build_api.py"
API = ROOT / "site" / "assets" / "api.json"
INDEX = ROOT / "site" / "assets" / "api-index.json"
DECL = ROOT / "borch-ts" / "dist" / "src"


def test_api_reference_is_not_stale():
    """`api.json` has to equal what the declaration files give right now.

    The declaration files are gitignored and so appear in no commit — with none of them
    there is nothing to compare against, so it skips. **Making absence a failure** turns
    this red on any checkout that has not built the bundle, and then people learn how to
    switch checks off.
    """
    if not DECL.exists():
        pytest.skip(f"no declaration files ({DECL.relative_to(ROOT)}) — run npm run build:ts first")
    if not API.exists():
        pytest.fail("site/assets/api.json is missing — python3 site/build_api.py")

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
    ("README.md", "ES 모듈", "bundle"),
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


def test_english_api_descriptions_are_not_stale():
    """Whether each English description still matches the source it was made from.

    This page read for a long time that it does not translate, because a translation
    drifts from the source the day it is written. The worry is right. Drift cannot be
    prevented; **drift being quiet** can.

    So every entry in `site/api_en.json` carries a hash of the Korean it was made from.
    When the source TSDoc moves, the hash stops matching and this blows up naming it. The
    key is printed as it stands, so whoever fixes it does not have to hunt for what moved.

    Being partly translated is not a failure — the screen shows the Korean and says it is
    not carried across, so the reader is not deceived. Failing on that too would let
    nothing in short of doing all 614 at once.
    """
    api = json.loads((SITE / "assets" / "api.json").read_text(encoding="utf-8"))
    table_path = ROOT / "site" / "api_en.json"
    assert table_path.exists(), "site/api_en.json is missing — it is where the English lives"
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
    assert not stale, ("the English did not follow its source — fix site/api_en.json and "
                       "put the new hash in src:\n  " + "\n  ".join(sorted(stale)[:20]))
    assert not orphan, ("English entries whose description is gone — renamed or deleted:\n  "
                        + "\n  ".join(sorted(orphan)[:20]))

    for key, got in table.items():
        assert got.get("en", "").strip(), f"the English for {key} is empty"
        assert not re.search(r"[가-힣]", got["en"]), f"Hangul remains in the English for {key}"


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

    want = {}
    for line in lock.read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, path = line.split("  ", 1)
            want[path] = digest
    assert want, "the lock file is empty"

    wrong = []
    for path, digest in sorted(want.items()):
        f = ROOT / "vendor" / path
        if not f.exists():
            wrong.append(f"{path}: missing")
        elif hashlib.sha256(f.read_bytes()).hexdigest() != digest:
            wrong.append(f"{path}: the bytes differ from the lock")
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
