"""Whether anything on the site is **cut off** rather than merely narrow.

## The difference this rests on

An element wider than its box is not a defect by itself. Three things can happen and only
one of them loses content:

- an ancestor's `overflow-x` is `auto` or `scroll` — it **scrolls**, and the reader can
  reach it;
- `overflow-x` is `visible` — it **spills** past the box and stays on screen;
- an ancestor's `overflow-x` is `hidden` or `clip`, or it leaves the window and the
  document does not scroll sideways — it is **cut**, and that part is unreachable.

**Reading the second case as the third is how this scan was wrong the first time.** It
called four things clipped that were not, two of them `.wide` — the stylesheet doing
exactly what it was asked to do. Only the third case is reported here.

## What it found

`python.html` carries five controls in a runnable's head where a lesson page carries
three, and `.runnable` clips because `overflow: hidden` is what keeps its rounded corner.
The Run button crossed the box by **51px at 360px** and **21px at 390px**, and fitted only
from 414px up: on most phones the primary control of that page was half present.

## Why it is safe to run on a machine with different fonts

The tightest control that passes is the landing's Run button, with 20px of room at 320,
360 and 390 alike — because that room is the card's padding and the button stretches to
fill what is left. Glyph widths do not move it. The next two are 44px and 68px, against a
real defect of 51px over.
"""

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
WIDTHS = (360, 1440)

SCAN = """() => {
  const doc = document.documentElement;
  const docScrolls = doc.scrollWidth > doc.clientWidth + 1;
  const cut = [];
  for (const el of document.querySelectorAll('body *')) {
    if (el.scrollWidth - el.clientWidth <= 1) continue;
    let verdict = null;
    for (let n = el; n && n !== document.body; n = n.parentElement) {
      const ov = getComputedStyle(n).overflowX;
      if (ov === 'auto' || ov === 'scroll') { verdict = 'scrolls'; break; }
      if (ov === 'hidden' || ov === 'clip') {
        verdict = 'cut by ' + n.tagName.toLowerCase() + '.' + (n.className || '');
        break;
      }
    }
    if (verdict === null) {
      const r = el.getBoundingClientRect();
      verdict = (r.right > doc.clientWidth + 1 && !docScrolls) ? 'cut at the window' : null;
    }
    if (verdict && verdict.startsWith('cut'))
      cut.push({tag: el.tagName.toLowerCase(), cls: (el.className || '').toString().slice(0, 40),
                by: (el.scrollWidth - el.clientWidth), why: verdict.slice(0, 60)});
  }
  return {cut: cut.slice(0, 8), painted: document.body.getBoundingClientRect().height};
}"""


def main():
    from playwright.sync_api import sync_playwright                  # noqa: PLC0415
    sys.path.insert(0, str(ROOT / "tests" / "browser"))
    from run import serve                                            # noqa: PLC0415
    from launch import browser                                       # noqa: PLC0415

    # JupyterLite's pages (`site/lab/`, built) are not this site's layout.
    pages = sorted(p for p in (ROOT / "site").rglob("*.html") if p.relative_to(ROOT / "site").parts[0] not in ("lab", "lab-src", "marimo", "marimo-src"))
    port, shutdown = serve(ROOT)
    problems, checked = [], 0
    try:
        with sync_playwright() as pw:
            with browser(pw, headed=False) as b:
                for width in WIDTHS:
                    for page in pages:
                        rel = page.relative_to(ROOT).as_posix()
                        p = b.new_page(viewport={"width": width, "height": 900})
                        p.goto(f"http://127.0.0.1:{port}/{rel}", wait_until="load")
                        p.wait_for_timeout(400)
                        # **Some pages send the browser somewhere else.** The repository
                        # root redirects into `site/`, and a page that navigates while
                        # the scan is running destroys the context it was running in.
                        # Settling and asking once more is enough; a page that keeps
                        # moving is reported rather than skipped.
                        try:
                            got = p.evaluate(SCAN)
                        except Exception:
                            p.wait_for_timeout(800)
                            try:
                                got = p.evaluate(SCAN)
                            except Exception as e:
                                problems.append(f"{rel} @{width}px — could not be measured: "
                                                f"{str(e).splitlines()[0][:80]}")
                                p.close()
                                continue
                        p.close()
                        # **A page that did not render has nothing overflowing and would
                        # pass in silence.** That is the same blindness as a scan whose
                        # stylesheet never loaded, so the height is asserted rather than
                        # assumed.
                        if got["painted"] < 200:
                            problems.append(f"{rel} @{width}px — the page did not render "
                                            f"(body is {round(got['painted'])}px tall)")
                            continue
                        checked += 1
                        for c in got["cut"]:
                            problems.append(f"{rel} @{width}px — {c['tag']}.{c['cls']} "
                                            f"loses {c['by']}px, {c['why']}")
    finally:
        shutdown()

    print(f"{checked} page-widths measured")
    if problems:
        print("\ncontent that cannot be reached:")
        for line in problems:
            print("  " + line)
        print("\n  Each of these is inside something that clips, or past the window edge\n"
              "  with nowhere to scroll. Give it room, let the row wrap, or let the box\n"
              "  scroll — but it cannot stay where it is.")
        return 1
    print("nothing is cut off")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
