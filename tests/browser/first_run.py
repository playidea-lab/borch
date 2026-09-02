"""Time to first run — the landing page, opened cold, Run pressed, until "done".

    uv run --with playwright python tests/browser/first_run.py [--headed]

**This is the product's first number and nothing measured it.** The PRD names
time-to-first-run as the success metric; the landing page has timed its own hero run
since the start (`home.js` prints `done — N ms`), and that number went to the person
who pressed the button and nowhere else. This opens the real page the way a visitor
does — page load, device probe, one click — and writes down three things: the time the
page reports, the wall time from opening the page to the done line, and the adapter.

The adapter is the part that makes the number mean something. A software adapter
answers every WebGPU call correctly and slowly, so the same procedure on a machine with
no GPU prints a true number about the wrong thing — this refuses that number, as the
golden runners do, rather than logging it beside a real one.

**The page is served from this checkout, so the network is not in the clock.** What
this measures is the page and the device — load, probe, compile, run — which is the
part this repository can change. The deployed site adds its transfer on top, and that
is a different measurement with a different owner.

First measurements, 2026-09-03, apple / metal-3: this checkout served locally, 180 ms in
the page and 0.7 s from opening the page to the done line (46 ms / 0.2 s on the quiet
machine at 04:30); the deployed site (`--url=https://playidea-lab.github.io/borch/site/index.html`),
763 ms and 1.8 s — the difference is the transfer, and that row is the visitor's number.

Run nightly (`tests/browser/nightly.py`) so the number has a history, not a moment.
"""
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))

from launch import browser, refuse_if_software                   # noqa: E402
from run import serve                                            # noqa: E402

# A cold page on a slow adapter can take this long before a person gives up; past it the
# number is a failure, not a measurement.
GIVE_UP_S = 120


def main():
    from playwright.sync_api import sync_playwright

    headed = "--headed" in sys.argv
    # `--url <page>` measures a deployed copy instead of this checkout — the network is
    # then inside the clock, which is the number a visitor actually waits for.
    url = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--url=")), None)
    port, shutdown = serve(ROOT)
    try:
        with sync_playwright() as pw, browser(pw, headed=headed) as b:
            page = b.new_page()
            errors = []
            page.on("pageerror", lambda e: errors.append(str(e)))
            opened = time.time()
            page.goto(url or f"http://127.0.0.1:{port}/site/index.html", wait_until="load")
            # The button is disabled until the device probe answers — that wait is part of
            # what a visitor experiences, so it is inside the clock.
            page.wait_for_function("!document.getElementById('hero-run').disabled",
                                   timeout=GIVE_UP_S * 1000)
            page.click("#hero-run")
            page.wait_for_function(
                "[...document.querySelectorAll('#hero-out div')].some(d => d.textContent.startsWith('done'))",
                timeout=GIVE_UP_S * 1000)
            wall = time.time() - opened
            done = page.evaluate(
                "[...document.querySelectorAll('#hero-out div')].map(d => d.textContent).find(t => t.startsWith('done'))")
            badge = page.evaluate("document.getElementById('device-text').textContent")
            lib = "/borch-ts/dist/src/index.js" if not url else url.rsplit("/site/", 1)[0] + "/borch-ts/dist/src/index.js"
            adapter = page.evaluate(
                f"import('{lib}').then(m => m.probe()).then(p => p.ok ? p.adapter : null)")
            in_page = done.split("—", 1)[1].split("ms")[0].strip() if "—" in done else "?"
            where = url or "this checkout, served locally (network not in the clock)"
            print(f"first run: {in_page} ms in the page · {wall:.1f} s from opening the page to done")
            print(f"  page: {where}")
            print(f"  badge: {badge}")
            if errors:
                print("  page errors: " + " | ".join(errors[:3]))
                return 1
            if refuse_if_software(adapter, "time to first run"):
                return 1
            print(f"  measured on: {adapter}")
            return 0
    finally:
        shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
