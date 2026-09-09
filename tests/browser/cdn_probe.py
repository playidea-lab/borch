"""The published package, from a CDN, in a page with no build step.

    uv run --with playwright python tests/browser/cdn_probe.py [--headless] [--from=<url>]

**This is the entry point the documents promise and nothing else checks.** `npm install
borch-ts` needs a bundler, and the reader who wants one runnable example in their own
documentation page does not have one. A `<script type="module">` and one import is the
path they take, and it works only while the published package keeps the shape a browser
can follow: `exports` pointing at real files, relative imports carrying `.js`, no bare
specifier the browser cannot resolve. Any of those can be lost in a refactor with every
test in this repository still green, because every one of them reads the tree rather than
the tarball.

So this reads the tarball, through the CDN, as a stranger would. It is the **published**
package, not this checkout — the version a visitor gets, which lags `main` and should.

**A CDN that is down is not borch being broken.** The repository already learned that a
CDN is a dependency that has to be alive at test time (`vendor.py`), and the answer there
was to stop depending on one. Here the dependency is the subject, so it cannot be removed
— it is told apart instead: the URL is fetched first, and an unreachable CDN prints what
happened and exits 0. Only a module that arrives and then misbehaves is a failure.
"""
import json
import os
import pathlib
import sys
import urllib.error
import urllib.parse
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[2]
PAGE = pathlib.Path(__file__).resolve().parent / "cdn_probe.html"
# Unpinned on purpose: what breaks silently is the *next* release, and a pinned probe
# would keep passing while the package a new reader installs no longer imports. The
# documents show the pinned form, which is what a page of somebody's own should use.
DEFAULT = "https://cdn.jsdelivr.net/npm/borch-ts/+esm"
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from launch import _headed, probe_lock, refuse_if_software  # noqa: E402


def reachable(url):
    """(ok, note). A HEAD the CDN answers means the module is there to be judged."""
    try:
        req = urllib.request.Request(url, method="GET", headers={"User-Agent": "borch-cdn-probe"})
        with urllib.request.urlopen(req, timeout=30) as r:
            body = r.read(4096)
            return True, f"{r.status} · {len(body)} bytes read"
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return False, str(exc)


def main(argv):
    from playwright.sync_api import sync_playwright  # noqa: PLC0415

    url = next((a.split("=", 1)[1] for a in argv if a.startswith("--from=")), DEFAULT)
    headed = _headed("--headed" in argv)
    print(f"module: {url}")

    ok, note = reachable(url)
    if not ok:
        print(f"**the CDN did not answer** — {note}\n"
              "  Nothing is said about borch here: the module never arrived. Not a failure.")
        return 0
    print(f"  reachable: {note}")

    probe_lock("the CDN probe")
    # `+esm` is a jsDelivr path and `+` is a space in a query string — the first run of
    # this probe fetched `/borch-ts/%20esm` and blamed the package.
    page_url = PAGE.as_uri() + "?from=" + urllib.parse.quote(url, safe="")
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=not headed,
                                     channel=os.environ.get("BORCH_CHROME_CHANNEL") or None,
                                     args=list(__import__("launch").FLAGS))
        try:
            page = browser.new_page()
            page.on("pageerror", lambda e: print(f"  [exception] {str(e)[:200]}"))
            page.goto(page_url)
            page.wait_for_function("window.__done !== null", timeout=120_000)
            got = page.evaluate("window.__done")
        finally:
            browser.close()

    if not got.get("ok"):
        print(f"  {got.get('error')}")
        print("**the published package did not run from a CDN** — a page with no build step "
              "is a documented entry point; see the docstring above for what breaks it.")
        return 1
    if refuse_if_software(got["adapter"], "the CDN probe"):
        return 1
    if abs(got["sum"] - 14) > 1e-4 or not (got["loss"] < 0.01):
        print(f"  the module ran and the numbers are wrong: {json.dumps(got)}")
        return 1
    print(f"  {got['names']} names · imported in {got['imported_ms']} ms · adapter {got['adapter']}")
    print(f"  x*x sum {got['sum']} · trained 60 steps to loss {got['loss']:.4f}")
    print("**the published package trains from a CDN, in a file opened from disk**")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
