"""Rewrites the case counts the documentation states, from the tables themselves.

    uv run python tests/refresh_counts.py            # say what would change
    uv run python tests/refresh_counts.py --write    # change it

## Why this exists

Adding a case moves five numbers — the whole table, what the core sees, what the
binding sees, the TS bodies written, and the remainder — across the README and two
index pages. Every one of them is checked, so nothing stays wrong; what it costs is a
round of red-fix-red on each, and in one session that ran past twenty times. **A check
catching it and a person not having to write it are different things**, and this is the
second.

It is the same argument `site/build_api.py` makes about the API reference: an index
written by hand is right for a week. These numbers are right until the next case.

## What it will not touch

The `data-measured="golden"` figures — `agreeing N / N [apple / metal-3]` — are a
claim about **a run somebody made on a real GPU**, and `test_site.py` says in as many
words that editing the number alone would be a claim about a run nobody made. So they
are printed for a person to carry over after running the browser suite, and never
rewritten. The Korean ceiling in `test_korean_ceiling.py` is left alone for the
neighbouring reason: raising it is supposed to cost a written sentence.
"""

import argparse
import importlib.util
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def counts():
    """The five numbers, **read from the same places the checks read them.**

    Not derived again here. `test_docs._counts` and `test_site._ledger_split` are the
    readers those checks trust, and a second derivation would be a second thing to keep
    in step — which is the defect this file is about, one level up.
    """
    docs = _load("rc_docs", ROOT / "tests" / "test_docs.py")
    site = _load("rc_site", ROOT / "tests" / "test_site.py")
    total, core, bind = docs._counts()
    declined, owed = site._ledger_split()
    written = total - declined - owed
    return {"total": total, "core": core, "bind": bind,
            "written": written, "remaining": declined + owed,
            "declined": declined, "owed": owed}


# Each rule is (file, pattern, which count). The pattern's one group is the number,
# and everything else in it has to match so a sentence that changed shape is not
# rewritten under its old meaning — the wording and the number move together or the
# rule stops finding it, which is the loud half.
RULES = (
    ("docs/BOOK.md", r"(?<=the binding passes \*\*)\d[\d,]*(?= golden cases)", "bind"),
    ("docs/BOOK.md", r"(?<=It passes \*\*)\d[\d,]*(?= golden cases)", "bind"),
    ("docs/BOOK.md", r"(?<=covers )\d[\d,]*(?= cases)", "core"),
    ("docs/BOOK.md", r"(?<=And \*\*)\d[\d,]*(?= golden cases\*\* compare)", "total"),
    # The front door states the total once, in the book's words minus the "And".
    ("README.md", r"(?<=\*\*)\d[\d,]*(?= golden cases\*\* compare)", "total"),
    # The front door states the total once, in the book's own words minus the "And".
    ("docs/BOOK.md", r"(?<=\*\*)\d[\d,]*(?= golden cases\*\* compare)", "total"),
    ("docs/BOOK.md", r"(?<=written TS bodies for )\d[\d,]*(?= cases)", "written"),
    ("docs/BOOK.md", r"(?<=The remaining )\d[\d,]*(?= are two)", "remaining"),
    ("docs/BOOK.md", r"(?<=things\*\*: )\d[\d,]*(?= deliberately)", "declined"),
    ("docs/BOOK.md", r"(?<=deliberately not carried across, and )\d[\d,]*(?= owed)", "owed"),
    ("site/index.html", r"(?<=<strong>)\d[\d,]*(?= golden cases</strong>)", "total"),
    ("site/ko/index.html", r"(?<=<strong>골든 )\d[\d,]*(?=건</strong>)", "total"),
)

MEASURED = re.compile(r'data-measured="golden">agreeing (\d+) / (\d+)')


def plan(now):
    """`[(file, was, becomes, which)]` for every rule whose number has moved."""
    moves = []
    for rel, pattern, which in RULES:
        text = (ROOT / rel).read_text(encoding="utf-8")
        for found in re.finditer(pattern, text):
            was = int(found.group(0).replace(",", ""))
            if was != now[which]:
                moves.append((rel, was, now[which], which))
    return moves


def apply(now):
    touched = {}
    for rel, pattern, which in RULES:
        path = ROOT / rel
        text = path.read_text(encoding="utf-8")
        fixed = re.sub(pattern, str(now[which]), text)
        if fixed != text:
            path.write_text(fixed, encoding="utf-8")
            touched[rel] = touched.get(rel, 0) + 1
    return touched


def measured_figures():
    """What the pages claim was measured, and where. Printed, never rewritten."""
    found = []
    for rel in ("site/index.html", "site/setup.html",
                "site/ko/index.html", "site/ko/setup.html"):
        path = ROOT / rel
        if not path.exists():
            continue
        for hit in MEASURED.finditer(path.read_text(encoding="utf-8")):
            found.append((rel, int(hit.group(1)), int(hit.group(2))))
    return found


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--write", action="store_true",
                    help="rewrite the numbers (without this, only say what would move)")
    args = ap.parse_args(argv[1:])

    now = counts()
    print(f"golden {now['total']} · core {now['core']} · binding {now['bind']}")
    print(f"TS bodies {now['written']} · remaining {now['remaining']} "
          f"({now['declined']} declined, {now['owed']} owed)")

    moves = plan(now)
    if not moves:
        print("\nthe documents already say so — nothing to move.")
    else:
        print(f"\n{len(moves)} numbers have moved:")
        for rel, was, becomes, which in moves:
            print(f"  {rel}: {was} → {becomes}  ({which})")
        if args.write:
            for rel, count in apply(now).items():
                print(f"  wrote {rel}")
        else:
            print("\n  run again with --write to change them.")

    stale = [(rel, asked) for rel, asked, agreed in measured_figures()
             if asked != now["bind"]]
    if stale:
        print("\n**A measured run is not rewritten here.** These pages report a run "
              f"over a table that has since moved (it is now {now['bind']} through "
              "the binding):")
        for rel, asked in stale:
            print(f"  {rel}: agreeing {asked} / {asked}")
        print("  run `uv run --with playwright python tests/browser/run.py "
              "--lib borch_webgpu`\n  and carry that line over by hand — the figure is "
              "a claim about a run somebody made.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
