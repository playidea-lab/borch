"""Whether the embed route runs — a lesson's cell, in a **cross-origin** iframe.

    uv run --with playwright python tests/browser/embed_probe.py

## Why it exists

`site/embed/embed.html` turns a lesson's runnable cell into a one-line iframe widget, and
that route is **excluded from every full-page guard on purpose** — it is chrome-less, so
`nav`, `share-metadata` and `sidebar` would all fire on it wrongly (`_pages()` in
`tests/test_site.py`, and `lessons.py`'s coverage, both skip the `embed` directory). An
exclusion is a decision to *not look here*; this file is *where it is looked at instead*.
Without it the embed is watched by nothing but the coi header sweep, and it sits on the
four most fragile joints there are — a foreign origin, no cross-origin isolation, the
Python twin stripped, and a postMessage resize. Any of the four can die in silence.

## What it asserts (plumbing, not speed)

For a sample of widgets, in a **cross-origin** iframe (an `about:blank` parent, foreign to
the http origin the embed is served from):

- the cell is pulled from its lesson and mounts (a Run button appears);
- `crossOriginIsolated` is **False** — so a pass proves it runs on a non-isolated foreign
  host, the real case, not a page that happens to be isolated;
- no Python tab survives (the py twin needs SharedArrayBuffer a foreign host cannot grant);
- pressing Run leaves no error line;
- the widget posts its height, so a host can auto-resize.

The claim is about the **plumbing**, which does not depend on the adapter — so this runs on
a software adapter as readily as a real one and does **not** call `refuse_if_software`. A
slow SwiftShader run and a fast Metal run answer the same question here.

## Following new lessons

The sample below is booted in a browser (expensive), but every lesson in
`curriculum.json` is checked **statically** to resolve to a page with a runnable cell — so
a lesson added tomorrow is covered against "`?lesson=<id>` finds nothing" without anyone
editing a list here. The disease `lessons.py` documents — a hand-kept list that silently
falls behind the directory — is kept off by that.
"""

import functools
import http.server
import json
import pathlib
import sys
import threading
import urllib.parse

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from launch import FLAGS  # noqa: E402
from run import ROOT, serve  # noqa: E402 — serve() sends COOP/COEP, for the isolated case


def serve_plain(root):
    """A plain static server — **no COOP/COEP headers**, unlike `run.serve`.

    That difference is the point. `run.serve` sends `Cross-Origin-Embedder-Policy:
    require-corp`, which turns a host into a cross-origin-isolated page — and such a page
    *blocks* a borch embed iframe unless the embed carries `Cross-Origin-Resource-Policy`,
    which GitHub Pages does not add. The realistic host — someone's blog — is **not**
    isolated, so a plain server is the case worth testing: it is where the embed has to work.
    """
    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *_a):  # noqa: D401 — the request log is noise in a probe
            pass

    handler = functools.partial(Quiet, directory=str(root))
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd.server_address[1], httpd.shutdown

CURRICULUM = ROOT / "site" / "assets" / "curriculum.json"

# The widgets actually booted — visual cells, so a broken run shows as a missing canvas as
# well as an error line. Their cell indices are the `show`/`plot` blocks of each lesson.
SAMPLE = [
    ("mini-transformer", 4),   # the causal-attention heatmap
    ("cnn", 3),                # the six conv kernels drawn
    ("batchnorm", 3),          # rows before and after
    ("adam", 3),               # the loss curve
]

TIMEOUT_MS = 180_000


def lessons_resolve():
    """Every curriculum lesson resolves to a page carrying a runnable cell. No browser."""
    cur = json.loads(CURRICULUM.read_text(encoding="utf-8"))
    bad = []
    for lesson in cur["lessons"]:
        page = ROOT / "site" / lesson["url"]
        if not page.exists():
            bad.append(f"{lesson['id']}: page {lesson['url']} is missing")
        elif 'class="runnable"' not in page.read_text(encoding="utf-8"):
            bad.append(f"{lesson['id']}: {lesson['url']} has no runnable cell to embed")
    return bad, len(cur["lessons"])


def press(page, host_url, embed_port, lesson_id, cell, problems):
    # The embed is served from `embed_port`; the parent page comes from a *different* port
    # (host_url), so the iframe is genuinely cross-origin. about:blank cannot be used as the
    # parent — WebGPU is absent on it, which is a property of the parent, not of the embed.
    src = f"http://127.0.0.1:{embed_port}/site/embed/embed.html?lesson={lesson_id}&cell={cell}"
    where = f"{lesson_id}/{cell}"
    page.goto(host_url + "?src=" + urllib.parse.quote(src, safe=""), wait_until="load")

    page.wait_for_timeout(1500)
    frame = next((f for f in page.frames if "embed/embed.html" in f.url), None)
    if frame is None:
        problems.append(f"{where}: the iframe never loaded")
        return
    go = frame.locator("div.runnable button.go")
    try:
        go.wait_for(timeout=60_000)
    except Exception:  # noqa: BLE001
        err = frame.locator("#err").inner_text() if frame.locator("#err").count() else "(no #err)"
        problems.append(f"{where}: no cell mounted — {err[:140]}")
        return
    if frame.evaluate("window.crossOriginIsolated"):
        problems.append(f"{where}: the iframe was cross-origin-isolated — the test no "
                        "longer proves the foreign, non-isolated case")
    if frame.locator('button.tab[data-lang="py"]').count():
        problems.append(f"{where}: a Python tab survived — the py twin was not stripped")
    go.first.click()
    frame.wait_for_function(
        "() => { const x = document.querySelector('div.runnable button.go'); return x && !x.disabled; }",
        timeout=TIMEOUT_MS)
    page.wait_for_timeout(350)
    if frame.locator(".err:not(#err)").count() or (
            frame.locator("#err").count() and frame.locator("#err").is_visible()):
        line = frame.locator(".err").first.inner_text().strip().splitlines()[0][:140]
        problems.append(f"{where}: an error line — {line}")
    if not page.evaluate("window.__h"):
        problems.append(f"{where}: no height was posted — a host cannot auto-resize it")


def main():
    from playwright.sync_api import sync_playwright  # noqa: PLC0415

    problems, n_lessons = lessons_resolve()
    # Two origins from the same tree, both plain (non-isolated): one serves the embed, the
    # other is the foreign host — a cross-origin iframe between them, the realistic case.
    embed_port, stop_embed = serve_plain(ROOT)
    host_port, stop_host = serve_plain(ROOT)
    host_url = f"http://127.0.0.1:{host_port}/tests/browser/embed_host.html"
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True, args=list(FLAGS))
            page = browser.new_page()
            page.set_default_timeout(0)
            for lesson_id, cell in SAMPLE:
                press(page, host_url, embed_port, lesson_id, cell, problems)

            # The other half of the rule: where the host IS isolated (this site's own
            # pages, a same-origin embed), the Python twin is **kept** — Pyodide can get
            # SharedArrayBuffer there. `serve` sends COOP/COEP, so a page it serves is
            # isolated; a dual-language cell must still show its Python tab.
            iso_port, stop_iso = serve(ROOT)
            try:
                page.goto(f"http://127.0.0.1:{iso_port}/site/embed/embed.html"
                          "?lesson=mini-transformer&cell=4", wait_until="load")
                page.wait_for_timeout(1200)
                if not page.evaluate("window.crossOriginIsolated"):
                    problems.append("isolated: served COOP/COEP but the frame is not isolated"
                                    " — the Python-kept path cannot be tested here")
                elif not page.locator('button.tab[data-lang="py"]').count():
                    problems.append("isolated: the Python tab was stripped on an isolated host"
                                    " — the twin should be kept where it can run")
            finally:
                stop_iso()
            browser.close()
    finally:
        stop_embed()
        stop_host()

    for line in problems:
        print(f"  ! {line}", file=sys.stderr)
    if problems:
        print("**the embed route is broken** — see above")
        return 1
    print(f"embed route ok — {len(SAMPLE)} widgets ran cross-origin, "
          f"all {n_lessons} lessons resolve to an embeddable cell")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
