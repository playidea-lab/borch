"""After `site/build_api.py`: the counts the pages state, made to agree with the index.

`python3 site/build_api.py && python3 site/sync_counts.py` is `npm run sync`. Run it before
a push that adds or renames a public name — the site's tests fail on a stale count, and
a stale count that reaches main turns main red for whoever comes next (2026-09-07, twice).

  · `_total` in the vision pages follows `api.json`'s total.
  · A Korean entry in `site/api_ko.json` whose English moved is **stale**, and that fails
    (the site's test fails on it too). A name with no Korean yet is not a failure — the
    page shows the English and says it is not carried across — so the missing are counted,
    and listed with `--missing`. The Korean is a person's sentence; this only names the gap.
"""
import hashlib
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
API = ROOT / "site" / "assets" / "api.json"
KO = ROOT / "site" / "api_ko.json"
PAGES = (ROOT / "site" / "vision.html", ROOT / "site" / "ko" / "vision.html")


def main():
    api = json.loads(API.read_text(encoding="utf-8"))
    total = api["total"]
    for page in PAGES:
        text = page.read_text(encoding="utf-8")
        new = re.sub(r'(data-count="_total">)\d+(<)', rf"\g<1>{total}\g<2>", text)
        if new != text:
            page.write_text(new, encoding="utf-8")
            print(f"{page.relative_to(ROOT)}: _total → {total}")
    ko = json.loads(KO.read_text(encoding="utf-8"))
    live = {}
    for mod in api["modules"]:
        live[f"{mod['name']}/"] = mod.get("doc") or ""
        for sym in mod["symbols"]:
            live[f"{mod['name']}/{sym['name']}"] = sym.get("doc") or ""
            for mem in sym["members"]:
                live[f"{mod['name']}/{sym['name']}.{mem['name']}"] = mem.get("doc") or ""
    fp = lambda t: hashlib.sha256(t.strip().encode("utf-8")).hexdigest()[:12]  # noqa: E731
    stale = sorted(k for k, v in ko.items() if k in live and v.get("src") != fp(live[k]))
    missing = sorted(k for k in live if live[k].strip() and k not in ko)
    print(f"api total {total} · korean entries {len(ko)} · stale {len(stale)} · missing {len(missing)}")
    for k in stale:
        print(f"  stale   {k}  |  {live[k].strip().splitlines()[0][:90]}")
    if "--missing" in sys.argv:
        for k in missing:
            print(f"  missing {k}  |  {live[k].strip().splitlines()[0][:90]}")
    if stale:
        print("fix the Korean in site/api_ko.json (src = sha256 of the English, first 12 hex)")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
