"""Writes `sitemap.xml` at the repository root — every page and document the deployment
serves, for the crawlers `robots.txt` invites.

Built, not committed, like `llms-full.txt`: the page list is the list on disk at deploy
time. `pages.yml` runs this after the site builds, so it covers what actually goes up.

    python3 site/build_sitemap.py
"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SITE = ROOT / "site"
BASE = "https://playidea-lab.github.io/borch"
OUT = ROOT / "sitemap.xml"
# Built folders carry their tooling's own pages; only their entry is a page of this site.
BUILT = ("lab", "lab-src", "marimo", "marimo-src")
# Documents at the root that agents read (the same set pages.yml copies into _site).
DOCUMENTS = ("llms.txt", "AGENTS.md", "README.md", "docs/BOOK.md", "BORCH-TS.md", "ROADMAP.md")


def pages():
    out = []
    for p in sorted(SITE.rglob("*.html")):
        parts = p.relative_to(SITE).parts
        if parts[0] in BUILT and parts[1:] != ("index.html",):
            continue
        out.append("site/" + "/".join(parts))
    return out


def main() -> int:
    urls = pages() + [d for d in DOCUMENTS if (ROOT / d).exists()]
    body = "\n".join(f"  <url><loc>{BASE}/{u}</loc></url>" for u in urls)
    OUT.write_text('<?xml version="1.0" encoding="UTF-8"?>\n'
                   '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n' + body + "\n</urlset>\n",
                   encoding="utf-8")
    print(f"wrote {OUT.name} — {len(urls)} URLs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
