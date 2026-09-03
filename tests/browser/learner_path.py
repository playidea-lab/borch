"""The learner's first minutes, on a clock.

    uv run --with playwright python tests/browser/learner_path.py [--headed] [--url=<page>]

The visitor this page is written for has followed a torch tutorial once. The clock
starts when the landing page is opened and stops at the moments that matter to that
visitor: Python ready under the button, the first loss line after the click, the
learned line, the run's own "done". Two visits in one profile — the first pays for
every download, the second is what a returning reader sees.

`first_run.py` times the GPU's first run; this times the person's. They share the
server, the flags and the software-adapter refusal.
"""
import os
import sys
import tempfile
import time

from first_run import FLAGS, GIVE_UP_S, ROOT, refuse_if_software, serve

READY_WORDS = ("Python is ready", "Python 준비됨")
LOSS_WORDS = ("step ", "step")
LEARNED_WORDS = ("learned", "배운 것")


def _has(page, words):
    quoted = ",".join(repr(w) for w in words)
    return page.evaluate(
        f"[...document.querySelectorAll('#hero-out div')].some(d => [{quoted}].some(w => d.textContent.startsWith(w)))")


def _wait(page, words, deadline):
    while time.time() < deadline:
        if _has(page, words):
            return time.time()
        page.wait_for_timeout(50)
    return None


def _visit(context, url, label, wait_ready=True):
    page = context.new_page()
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    opened = time.time()
    page.goto(url, wait_until="load")
    page.wait_for_function(
        "document.getElementById('device-text').textContent !== 'checking device…' && "
        "document.getElementById('device-text').textContent !== '장치 확인 중…'",
        timeout=GIVE_UP_S * 1000)
    probed = time.time()
    ready = None
    if wait_ready:
        quoted = ",".join(repr(w) for w in READY_WORDS)
        page.wait_for_function(
            f"[{quoted}].some(w => document.getElementById('hero-ready').textContent.startsWith(w))",
            timeout=GIVE_UP_S * 1000, polling=100)
        ready = time.time()
    clicked = time.time()
    page.click("#hero-run")
    deadline = clicked + GIVE_UP_S
    first_loss = _wait(page, LOSS_WORDS, deadline)
    learned = _wait(page, LEARNED_WORDS, deadline)
    done = _wait(page, ("done", "끝"), deadline)
    lines = page.evaluate("[...document.querySelectorAll('#hero-out div')].map(d => d.textContent)")
    adapter = page.evaluate(
        "import('/borch-ts/dist/src/index.js').then(m => m.probe()).then(p => p.ok ? p.adapter : null)"
        if not url.startswith("http") or "127.0.0.1" in url else
        f"import('{url.rsplit('/site/', 1)[0]}/borch-ts/dist/src/index.js').then(m => m.probe()).then(p => p.ok ? p.adapter : null)")
    fmt = lambda t: f"{t - opened:5.1f} s" if t else "  —   "
    print(f"{label}:")
    print(f"  probe {fmt(probed)} · Python ready {fmt(ready)} · click {fmt(clicked)} · "
          f"first loss {fmt(first_loss)} · learned {fmt(learned)} · done {fmt(done)}  (from opening the page)")
    if first_loss:
        print(f"  click → first loss line {first_loss - clicked:.2f} s · click → learned {(learned or first_loss) - clicked:.2f} s")
    for line in lines:
        if line.startswith(LOSS_WORDS) or line.startswith(LEARNED_WORDS) or "rror" in line or "Traceback" in line:
            print(f"    {line}")
    if errors:
        print("  page errors: " + " | ".join(errors[:3]))
    ok = bool(learned) and not errors
    page.close()
    return adapter, ok


def main():
    from playwright.sync_api import sync_playwright
    headed = "--headed" in sys.argv
    url = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--url=")), None)
    port, shutdown = serve(ROOT)
    target = url or f"http://127.0.0.1:{port}/site/index.html"
    channel = os.environ.get("BORCH_CHROME_CHANNEL") or None
    profile = tempfile.mkdtemp(prefix="borch-learner-")
    try:
        with sync_playwright() as pw:
            want_headed = headed or not (os.environ.get("BORCH_HEADLESS") or "--headless" in sys.argv)
            context = pw.chromium.launch_persistent_context(
                profile, headless=not want_headed, channel=channel, args=list(FLAGS), timeout=60_000)
            try:
                adapter, ok1 = _visit(context, target, "first visit, click when Python is ready")
                if refuse_if_software(adapter, "the learner's first minutes"):
                    return 1
                _a, ok2 = _visit(context, target, "revisit, click at once", wait_ready=False)
            finally:
                context.close()
            print(f"  page: {url or 'this checkout, served locally (network not in the clock)'}")
            print(f"  measured on: {adapter}")
            return 0 if ok1 and ok2 else 1
    finally:
        shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
