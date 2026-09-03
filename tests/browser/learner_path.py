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

from first_run import FLAGS, GIVE_UP_S, ROOT, refuse_if_screen_off, refuse_if_software, serve

READY_WORDS = ("Python is ready", "Python 준비됨")
LOSS_WORDS = ("step ", "step")
LEARNED_WORDS = ("learned", "배운 것")


def _has(page, words):
    quoted = ",".join(repr(w) for w in words)
    return page.evaluate(
        f"[...document.querySelectorAll('#hero-out div')].some(d => [{quoted}].some(w => d.textContent.startsWith(w)))")


def _wait(page, words, deadline):
    """Waits for a line starting with one of `words` — **inside the page**, with
    `wait_for_function`, not by polling `evaluate` from outside. Polling every 50 ms
    from the driver put a task on the page's main thread between every two of the
    loop's readbacks, and on the 4090 that clock read 7 s where the same code through
    a page nobody was poking took 1 s (measured; the polling version is in git)."""
    quoted = ",".join(repr(w) for w in words)
    remaining = max(1, int((deadline - time.time()) * 1000))
    try:
        page.wait_for_function(
            f"[...document.querySelectorAll('#hero-out div')].some(d => [{quoted}].some(w => d.textContent.startsWith(w)))",
            timeout=remaining, polling="raf")
    except Exception:
        return None
    return time.time()


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
    lib = "/borch-ts/dist/src/index.js" if "127.0.0.1" in url else url.rsplit("/site/", 1)[0] + "/borch-ts/dist/src/index.js"
    pipelines_before = page.evaluate(f"import('{lib}').then(m => (m.currentDevice() && m.currentDevice().pipelineCount) || 0)") or 0
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
    pipelines_after = page.evaluate(f"import('{lib}').then(m => (m.currentDevice() && m.currentDevice().pipelineCount) || 0)") or 0
    if first_loss:
        print(f"  click → first loss line {first_loss - clicked:.2f} s · click → learned {(learned or first_loss) - clicked:.2f} s"
              f" · pipelines compiled during the run: {pipelines_after - pipelines_before}")
    for line in lines:
        if line.startswith(LOSS_WORDS) or line.startswith(LEARNED_WORDS) or "rror" in line or "Traceback" in line:
            print(f"    {line}")
    if errors:
        print("  page errors: " + " | ".join(errors[:3]))
    ok = bool(learned) and not errors
    page.close()
    return adapter, ok


FIX_A = ("lr=0.0", "lr=0.1")
FIX_B = ("    loss.backward(); opt.step()", "    opt.zero_grad(); loss.backward(); opt.step()")


def _run_block(page, index, code=None):
    """Runs the lesson's `index`-th block (with `code` in the editor when given) and
    returns its verdict line and how long the run took."""
    boxes = page.query_selector_all(".runnable")
    box = boxes[index]
    if code is not None:
        box.query_selector("textarea").fill(code)
        box.query_selector("textarea").dispatch_event("input")
    t0 = time.time()
    box.query_selector("button.go").click()
    page.wait_for_function(
        "(i) => [...document.querySelectorAll('.runnable')[i].querySelectorAll('.out div')]"
        ".some(d => d.className.includes('verdict') || d.className === 'err')",
        arg=index, timeout=GIVE_UP_S * 1000)
    took = time.time() - t0
    lines = page.evaluate(
        "(i) => [...document.querySelectorAll('.runnable')[i].querySelectorAll('.out div')].map(d => [d.className, d.textContent])",
        index)
    verdict = next((text for cls, text in lines if "verdict" in cls), None)
    err = next((text for cls, text in lines if cls == "err"), None)
    return verdict, err, took


def _lesson(context, url):
    """Lesson 0: each broken loop must be judged ✗ as written and ✓ once fixed."""
    lesson = url.replace("index.html", "learn/00-fix-it.html")
    page = context.new_page()
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    opened = time.time()
    page.goto(lesson, wait_until="load")
    page.wait_for_selector(".runnable button.go", timeout=GIVE_UP_S * 1000)
    ok = True
    print("lesson 0 — fix the bug:")
    for index, (name, fix) in enumerate((("lr = 0", FIX_A), ("no zero_grad", FIX_B))):
        original = page.evaluate("(i) => document.querySelectorAll('.runnable')[i].querySelector('textarea').value", index)
        v1, e1, t1 = _run_block(page, index)
        v2, e2, t2 = _run_block(page, index, original.replace(fix[0], fix[1]))
        bad = (v1 or "").startswith("✗") and not e1
        good = (v2 or "").startswith("✓") and not e2
        ok = ok and bad and good
        print(f"  {name}: as written {'✗' if bad else 'NOT ✗'} in {t1:.1f} s · fixed {'✓' if good else 'NOT ✓'} in {t2:.1f} s")
        for line in (v1, e1, v2, e2):
            if line: print(f"    {line[:120]}")
    # The third block exports the trained model — no verdict, but it must write the file.
    boxes = page.query_selector_all(".runnable")
    boxes[2].query_selector("button.go").click()
    page.wait_for_function(
        "[...document.querySelectorAll('.runnable')[2].querySelectorAll('.out div')]"
        ".some(d => d.className === 'ok' || d.className === 'err')", timeout=GIVE_UP_S * 1000)
    wrote = page.evaluate(
        "[...document.querySelectorAll('.runnable')[2].querySelectorAll('.out div')].map(d => d.textContent).find(t => t.startsWith('wrote')) || ''")
    export_ok = wrote.startswith("wrote /work/model.onnx")
    ok = ok and export_ok
    print(f"  export block: {'wrote the file' if export_ok else 'did NOT write the file'}" + (f" — {wrote[:80]}" if wrote else ""))
    print(f"  lesson opened → both blocks judged and the export written: {time.time() - opened:.1f} s")
    if errors:
        print("  page errors: " + " | ".join(errors[:3]))
        ok = False
    page.close()
    return ok


def main():
    from playwright.sync_api import sync_playwright
    headed = "--headed" in sys.argv
    url = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--url=")), None)
    if refuse_if_screen_off("the learner's first minutes"):
        return 1
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
                ok3 = _lesson(context, target)
            finally:
                context.close()
            print(f"  page: {url or 'this checkout, served locally (network not in the clock)'}")
            print(f"  measured on: {adapter}")
            return 0 if ok1 and ok2 and ok3 else 1
    finally:
        shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
