"""Every page of the site, cross-origin isolated, with nothing blocked — twice.

The cpu device's worker pool needs `SharedArrayBuffer`, which a page gets only under
`Cross-Origin-Opener-Policy: same-origin` and `Cross-Origin-Embedder-Policy: require-corp`.
Those two headers change what else the page may load: a cross-origin response without
CORS or CORP is refused, silently, by the browser. So this walks every HTML page under
`site/` (the built notebook and workbench pages too, where they exist) and asks two things
of each — that `crossOriginIsolated` is true, and that no request was blocked by the
embedder policy — under both ways the headers can arrive:

  headers   the server sends them (`site/serve.py`, the test servers);
  shim      the server sends nothing and `site/coi.js` registers a service worker that
            adds them, reloading the page once — GitHub Pages, `python3 -m http.server`.

A page that only breaks under isolation breaks here first, named. Run:

    npm run coi:site          (or: uv run --with playwright python tests/browser/coi_sweep.py)
"""
import functools
import http.server
import os
import pathlib
import socketserver
import sys
import threading
import time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from launch import _headed  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
SITE = ROOT / "site"
SKIP_TOP = ("lab-src", "marimo-src")
ISOLATION_S = 20
SETTLE_S = 2.0
# **Every call into the page has a clock.** The first nightly to run this (2026-09-07) sat
# on one page for two hours and seventeen minutes with a renderer at full CPU, and the
# runner behind it never reached the fifteen rows after. The timeout was 0 - infinite -
# because a slow page was thought to be the page's business. It is; the sweep's business
# is to say which page, in a line, and move on.
PAGE_MS = 60_000
PlaywrightTimeoutError: type = Exception  # replaced in main() once playwright is imported

# The nightly reads this through a file, and a page's line has to land as it happens -
# block-buffered, the two-hour hang above showed nothing at all until the process was killed.
print = functools.partial(print, flush=True)  # noqa: A001


def pages():
    out = []
    for p in sorted(SITE.rglob("*.html")):
        parts = p.relative_to(SITE).parts
        if parts[0] in SKIP_TOP:
            continue
        # The built folders carry their tooling's own pages; only their entry counts.
        if parts[0] in ("lab", "marimo") and p.name != "index.html":
            continue
        out.append(p)
    return out


def serve(headers):
    from launch import probe_lock  # noqa: PLC0415 — one browser probe at a time on this machine
    probe_lock("the isolation sweep")
    class Handler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def end_headers(self):
            self.send_header("Cache-Control", "no-store")
            if headers:
                self.send_header("Cross-Origin-Opener-Policy", "same-origin")
                self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
            super().end_headers()

    httpd = socketserver.ThreadingTCPServer(("127.0.0.1", 0), functools.partial(Handler, directory=str(ROOT)))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd.server_address[1], httpd.shutdown


def visit(context, url):
    """(isolated, blocked requests, console lines about the policy, seconds)."""
    page = context.new_page()
    page.set_default_timeout(PAGE_MS)
    page.set_default_navigation_timeout(PAGE_MS)
    blocked, console = [], []
    page.on("requestfailed", lambda r: blocked.append(f"{r.url} — {(r.failure or '')}") if "BLOCKED_BY_RESPONSE" in (r.failure or "") else None)
    page.on("console", lambda m: console.append(m.text) if ("Cross-Origin" in m.text or "CORP" in m.text or "COEP" in m.text) else None)
    t0 = time.time()
    isolated = False
    stuck = None
    try:
        page.goto(url, wait_until="load")
        deadline = t0 + ISOLATION_S
        while time.time() < deadline:
            try:
                isolated = bool(page.evaluate("crossOriginIsolated"))
            except PlaywrightTimeoutError:
                # The main thread did not answer for PAGE_MS: a script is spinning. That is
                # the finding - not a slow page, a stuck one - and it is said, not waited on.
                stuck = "the page's main thread did not answer"
                break
            except Exception:  # noqa: BLE001 - the shim reloads the page under us; ask again
                isolated = False
            if isolated:
                break
            time.sleep(0.25)
        if stuck is None:
            time.sleep(SETTLE_S)  # subresources after `load`
    except PlaywrightTimeoutError:
        stuck = f"did not finish loading in {PAGE_MS // 1000}s"
    finally:
        try:
            page.close()
        except Exception:  # noqa: BLE001 - a stuck renderer may refuse; the context goes with it
            pass
    if stuck:
        console.insert(0, f"STUCK: {stuck}")
    return isolated, blocked, console, time.time() - t0


def main(argv):
    from playwright.sync_api import sync_playwright  # noqa: PLC0415
    global PlaywrightTimeoutError
    from playwright.sync_api import TimeoutError as PlaywrightTimeoutError  # noqa: PLC0415

    headed = _headed("--headed" in argv)   # a window unless --headless / BORCH_HEADLESS: headless is SwiftShader here
    only = [a for a in argv if not a.startswith("--")]
    todo = [p for p in pages() if not only or any(o in p.as_posix() for o in only)]
    print(f"{len(todo)} pages · isolation waited up to {ISOLATION_S}s · {SETTLE_S:.0f}s settle")
    failures = 0
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=not headed)
        for mode, headers in (("headers", True), ("shim", False)):
            port, stop = serve(headers)
            context = browser.new_context()
            print(f"\n== {mode} ==")
            for p in todo:
                rel = p.relative_to(ROOT).as_posix()
                if mode == "shim" and p.relative_to(SITE).parts[0] == "lab":
                    # JupyterLite registers a service worker of its own at this scope; a second
                    # one would replace it. The notebook page is isolated by headers only.
                    print(f"  skip {rel:<32} (JupyterLite's own service worker owns this scope)")
                    continue
                isolated, blocked, console, secs = visit(context, f"http://127.0.0.1:{port}/{rel}")
                ok = isolated and not blocked and not any(c.startswith("STUCK") for c in console)
                failures += 0 if ok else 1
                mark = "ok  " if ok else "FAIL"
                print(f"  {mark} {rel:<32} isolated={str(isolated).lower():<5} blocked={len(blocked)} {secs:4.1f}s")
                for b in blocked:
                    print(f"         blocked: {b[:160]}")
                for c in console[:3]:
                    print(f"         console: {c[:160]}")
            context.close()
            stop()
        browser.close()
    print("\n**every page is cross-origin isolated both ways and nothing is blocked**" if not failures else f"\n**{failures} page visit(s) failed** — see above")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
