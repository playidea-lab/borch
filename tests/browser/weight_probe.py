"""What the landing page costs a visitor in bytes, and how much of it comes before the verdict.

    uv run --with playwright python tests/browser/weight_probe.py [--headless] [--url=<page>]

**The badge is the product's first promise and it is the cheap half.** The page says which
device you have before it asks you to wait for anything; that judgement arrived in 0.62 MB
when this was written, against 10.20 MB for the first `done`. Those two numbers are worth
keeping apart, because only one of them can be spent carelessly: a font, a screenshot or a
library added to the landing page lands in the first half, and nothing here would have said
so — the site's other checks read what the page *says*, not what it *weighs*.

So this fails when the bytes before the verdict cross a ceiling, and prints the rest.

**Measuring it is not as simple as asking the browser.** `transferSize` reads 0 for every
resource here, because the site registers a service worker for cross-origin isolation and a
response that passed through one reports nothing; and summing `content-length` over
responses counts the shim's one reload twice. Each URL is counted once instead, which is
what a visitor's line actually carries.

First measurement, 2026-09-10, the deployed site on a fast line: 0.62 MB to the badge in
0.4 s, 10.20 MB to the first `done` in 4.0 s. Of that, Pyodide's wasm (3.19 MB), its
standard library (2.36 MB) and numpy (3.06 MB) are 8.6 MB — the first screen is a Python
training loop, and `borch_webgpu` imports numpy in five modules, so none of it is optional.
"""
import os
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
DEPLOYED = "https://playidea-lab.github.io/borch/site/"
# Generous on purpose: it is not a budget to be tuned but a tripwire for an accident. The
# badge cost 0.62 MB when this was written; something has gone wrong long before 2 MB.
BADGE_CEILING = 2_000_000
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from launch import _headed, probe_lock  # noqa: E402


def main(argv):
    from playwright.sync_api import sync_playwright  # noqa: PLC0415

    url = next((a.split("=", 1)[1] for a in argv if a.startswith("--url=")), DEPLOYED)
    headed = _headed("--headed" in argv)
    probe_lock("the page-weight probe")
    size, kind, order = {}, {}, []

    def on_response(r):
        u = r.url.split("?")[0]
        if u in size:
            return                      # the isolation shim reloads once; a URL is paid for once
        try:
            n = int(r.headers.get("content-length") or 0)
        except ValueError:
            n = 0
        size[u], kind[u] = n, r.request.resource_type
        order.append((time.time(), u))

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=not headed,
                                     channel=os.environ.get("BORCH_CHROME_CHANNEL") or None,
                                     args=list(__import__("launch").FLAGS))
        try:
            page = browser.new_context().new_page()
            page.on("response", on_response)
            t0 = time.time()
            page.goto(url, wait_until="load")
            page.wait_for_function(
                "document.getElementById('device-text') && "
                "!/checking|확인 중/.test(document.getElementById('device-text').textContent)",
                timeout=120_000)
            t_badge = time.time()
            badge = page.inner_text("#device-text")
            page.click("#hero-run")
            page.wait_for_function(
                "[...document.querySelectorAll('#hero-out div')].some(d => d.textContent.startsWith('done'))",
                timeout=300_000)
            t_done = time.time()
            said = page.evaluate("[...document.querySelectorAll('#hero-out div')]"
                                 ".map(d => d.textContent).find(t => t.startsWith('done'))")
        finally:
            browser.close()

    def upto(t):
        return sum(size[u] for ts, u in order if ts <= t), len([1 for ts, _ in order if ts <= t])

    mb = lambda n: f"{n / 1e6:.2f} MB"                                          # noqa: E731
    to_badge, n_badge = upto(t_badge)
    to_done, n_done = upto(t_done)
    print(f"page: {url}")
    print(f"  badge   {t_badge - t0:5.1f} s · {n_badge:3d} URLs · {mb(to_badge)}   {badge}")
    print(f"  done    {t_done - t0:5.1f} s · {n_done:3d} URLs · {mb(to_done)}   {said}")
    print("  largest: " + " · ".join(
        f"{u.split('/')[-1][:28]} {mb(n)}" for u, n in sorted(size.items(), key=lambda x: -x[1])[:3]))
    print("  a first click costs " + " · ".join(
        f"{bw} Mbps {to_done * 8 / 1e6 / bw:.0f} s" for bw in (25, 10, 5, 2)))
    if to_badge > BADGE_CEILING:
        print(f"**{mb(to_badge)} before the page says which device you have** — the ceiling is "
              f"{mb(BADGE_CEILING)}.\n"
              "  The verdict is the half that has to stay cheap; whatever was added to the "
              "landing page\n  is paid by every visitor before they are told anything.")
        return 1
    print("**the verdict is cheap, and the wait after it is narrated**")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
