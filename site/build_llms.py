"""Writes `llms-full.txt` at the repository root — the documents an agent would otherwise
fetch one by one, in the order `llms.txt` lists them.

Built, not committed: the book changes daily and a committed copy would be the stale
one an agent reads. `pages.yml` runs this after `build_api.py`, so the deployed file is
the deployed book. Run it by hand to see what an agent sees:

    python3 site/build_llms.py
"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
PARTS = ["README.md", "AGENTS.md", "docs/BOOK.md", "BORCH-TS.md", "ROADMAP.md"]
OUT = ROOT / "llms-full.txt"


def main() -> int:
    chunks = []
    for rel in PARTS:
        path = ROOT / rel
        if not path.exists():
            print(f"missing: {rel}", file=sys.stderr)
            return 1
        chunks.append(f"\n\n<!-- ===== {rel} ===== -->\n\n" + path.read_text(encoding="utf-8").rstrip() + "\n")
    OUT.write_text("# borch — every document in one file\n" + "".join(chunks), encoding="utf-8")
    print(f"wrote {OUT.name} — {OUT.stat().st_size // 1024} KB from {len(PARTS)} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
