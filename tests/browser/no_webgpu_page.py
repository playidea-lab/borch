"""The first screen a visitor without WebGPU reads — in both languages.

    uv run --with playwright python tests/browser/no_webgpu_page.py

**Nothing had ever opened `/ko/`.** On 2026-09-12 a person opened the Korean home page in
Safari 18 and the badge said `WebGPU 없음 (브라우저)` in Korean with an English paragraph
under it, carrying `**` from the source as characters. Two defects on the first screen of
the page, found the most expensive way there is: by a human, on their own laptop.

Every other browser check asks for a GPU and is therefore run on a machine that has one,
so the screen shown to a machine that does not was the one screen nothing looked at. This
one is the opposite: it needs the API to be **absent**, so it runs anywhere, headless, in
a couple of seconds.

Both shapes are pressed, because they are different code paths and a browser gives one or
the other:

- `absent` — `navigator.gpu` is not there at all. Safari 18 to 25 with the flag off.
- `undefined` — the key is there and its value is not. A policy or an extension leaves
  this, and it is the shape that found a third defect the day this was written: the guard
  asked `"gpu" in navigator`, which is true here, and the visitor got
  `Cannot read properties of undefined` where the sentence was meant to go.
- `no-adapter` — the object is there and `requestAdapter()` answers null. What a VM or
  `--disable-features=WebGPU` leaves; measured in `marimo_probe`.

And one shape that is the opposite — the adapter is real and the way back from it is not:

- `no-jspi` — WebGPU is up and `WebAssembly.Suspending` is gone. Safari from 26 is exactly
  this, and there the hero trained and then died inside `.item()` with a Pyodide traceback.
  **This one needs an adapter**, so it is skipped, out loud, where there is none — which is
  every CI runner. The nightly runs it on a machine with a GPU.
"""

import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from first_run import FLAGS                                          # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
PAGES = {"en": "site/index.html", "ko": "site/ko/index.html"}
# **The lessons offer a Run button too, and fifty of their snippets open `import
# borch_webgpu`.** Two are pressed here: one whose Python runs on the core once the import
# is swapped, and one about the device itself, which cannot — it has to say so in a
# sentence rather than answer with a traceback. Seven of the seventeen lesson pages are of
# the second kind (`scope`, `memory`, `backend`, `pooled`), measured 2026-09-14.
LESSONS = {
    "site/learn/02-autograd.html": "runs",
    "site/learn/01-tensors.html": "explains",
    "site/ko/learn/01-tensors.html": "explains",
    # **The lessons whose subject is the mathematics, not the device.** These were written
    # to run on either side, and the names they use — `linalg.solve`, `svd`, `softmax`,
    # `triu` — are the numpy core's as much as the binding's. That is a claim about two
    # packages, and it is worth one press each rather than a reading of both.
    "site/foundations/05-linear-systems.html": "runs",
    "site/foundations/06-eig-svd.html": "runs",
    "site/techniques/05-softmax-ce.html": "runs",
    "site/capstones/01-mini-transformer.html": "runs",
    # **Two more for the figures' sake, and only two.** `show` paints from values the core
    # computed, which is a claim that held until the drawer read a borch.ts-only method
    # off a core tensor and every figure on an adapterless machine stopped. The pages
    # above cover a single heatmap and a curve; these two cover the layouts nothing else
    # does — a bank of six kernels from one `conv.weight`, and two `show` calls in one
    # block for a before and an after.
    "site/learn/05-cnn.html": "runs",
    "site/techniques/03-batchnorm.html": "runs",
    # **A lesson whose subject is a binding-only Python tool.** Block 0 is pure arithmetic
    # (the width of a measurement) and runs on the core; the `torch.workbench` blocks below
    # it are the binding's, so with no adapter they must say so in a sentence — the
    # binding-only mark, not a traceback. Registered so that path is pressed where there is
    # no GPU, which is every CI runner.
    "site/capstones/02-honest-measurement.html": "runs",
    "site/ko/capstones/02-honest-measurement.html": "runs",
}
HANGUL = re.compile(r"[가-힣]")
# One runs before the page's scripts, so the page boots into the shape rather than being
# changed under it.
SHIMS = {
    "absent": "try { delete Navigator.prototype.gpu; } catch (e) {} try { delete navigator.gpu; } catch (e) {}",
    "undefined": "Object.defineProperty(Navigator.prototype, 'gpu', { configurable: true, get: () => undefined });",
    "no-adapter": "",
}


def read(page):
    """The badge and the lines the page wrote under it."""
    badge = page.inner_text("#device-text").strip()
    lines = [t.strip() for t in page.locator("#hero-out div").all_inner_texts() if t.strip()]
    return badge, lines


def press_run(page):
    """Click Run and wait for the last line the snippet prints.

    **The sentence on the screen is a promise and this is the promise.** The page said the
    core would run instead, and the snippet it showed opened with `import borch_webgpu`,
    which is not loaded on this path: Run answered `ModuleNotFoundError` under that
    sentence. Reading the note is not enough — something has to press the button.
    """
    page.click("#hero-run")
    page.wait_for_function(
        "() => document.getElementById('hero-out').innerText.includes('learned')", timeout=180_000)
    return [l for l in page.inner_text("#hero-out").splitlines() if "learned" in l][0].strip()


def press_block(page, box, index, where):
    """Press one block's Python and say what came back: (first line shown, text, problems).

    **Every block, not the first one.** A page's figures and its worked example are
    different code on the same page, and the figures went in after this probe did — so on
    a machine with no adapter they were the one thing nobody had run. Pyodide is already
    up by the second block, so the extra presses cost their own run and not another boot.
    """
    tab = box.locator('button.tab[data-lang="py"]')
    if tab.count():
        tab.first.click()
        page.wait_for_timeout(300)
    area = box.locator("textarea")
    shown = area.input_value().splitlines()[0] if area.count() else ""
    box.locator("button.go").first.click()
    # **It is finished when the button can be pressed again**, which is `runnable.js`'s
    # own `runBtn.disabled`. Waiting for particular words instead means this file has to
    # know their spelling, and it hung for three minutes on a block whose error was
    # phrased in neither language it had been told to expect. `lessons.py` learned the
    # same thing and says so at more length.
    page.wait_for_function(
        "i => !document.querySelectorAll('.runnable')[i].querySelector('button.go').disabled",
        arg=index, timeout=180_000)
    page.wait_for_timeout(150)
    text = box.inner_text()
    # **By class, not by wording.** The finished line and the refusal are both translated,
    # so a check that reads them has to know two languages and goes quiet when a third
    # arrives. `runnable.js` marks the binding-only refusal; everything else with `err` is
    # a failure whatever it says.
    stopped = box.locator(".err:not(.binding-only)")
    explained = box.locator(".binding-only")
    bad = []
    if "Traceback" in text or "ModuleNotFoundError" in text:
        bad.append(f"{where}: a traceback reached the screen — {text[-160:]}")
    if "import borch_webgpu" in shown:
        bad.append(f"{where}: the snippet still opens the binding: {shown!r}")
    if stopped.count():
        bad.append(f"{where}: it stopped — {stopped.first.inner_text()[:140]}")
    return shown, text, bad, explained.count() > 0


def press_lesson(page, kind):
    """Press every block on the page; the first one also has to be of the declared kind."""
    boxes = page.locator(".runnable")
    count = boxes.count()
    if count == 0:
        return "", "", ["the page has no runnable block"]
    first_shown = first_text = ""
    bad = []
    for i in range(count):
        box = boxes.nth(i)
        shown, text, problems, explained = press_block(page, box, i, f"block {i}")
        bad += problems
        if i == 0:
            first_shown, first_text = shown, text
            if kind == "runs" and explained:
                bad.append(f"block 0: it wanted a binding-only name — {text[-160:]}")
            if kind == "explains" and not explained:
                bad.append(f"block 0: nothing said which name it wanted — {text[-160:]}")
    return first_shown, first_text, bad


def judge_jspi(lang, badge, lines):
    """The adapter is named and the Python path says why it is not the binding's."""
    said = " ".join(lines)
    bad = []
    if "Traceback" in said or "PythonError" in said:
        bad.append(f"{lang}/no-jspi: a traceback reached the screen — {said[:120]}")
    if "JSPI" not in said and "jspi" not in said:
        bad.append(f"{lang}/no-jspi: nothing said why the binding is not in use — {said[:120]}")
    if lang == "ko" and said and not HANGUL.search(said):
        bad.append(f"{lang}/no-jspi: the Korean page said it in English — {said[:90]}")
    if not badge or "no " in badge or "없" in badge:
        bad.append(f"{lang}/no-jspi: the badge does not name the adapter — {badge!r}")
    return bad


def judge(lang, shape, badge, lines):
    """What has to be true of that screen, whatever the wording is."""
    said = " ".join(lines)
    bad = []
    if not badge:
        bad.append(f"{lang}/{shape}: the badge is empty")
    if not lines:
        bad.append(f"{lang}/{shape}: nothing was said under the badge")
    # **The language of the page, not of the library.** This is the defect that shipped:
    # the page translated its own words and handed the library's through untouched.
    if lang == "ko" and said and not HANGUL.search(said):
        bad.append(f"{lang}/{shape}: the Korean page said it in English — {said[:90]}")
    if lang == "en" and HANGUL.search(said):
        bad.append(f"{lang}/{shape}: the English page said it in Korean — {said[:90]}")
    if lang == "ko" and not HANGUL.search(badge):
        bad.append(f"{lang}/{shape}: the badge is English on the Korean page — {badge}")
    # Nothing that renders these is a markdown renderer.
    if "**" in said or "**" in badge:
        bad.append(f"{lang}/{shape}: markup reached the screen as characters — {said[:90]}")
    return bad


def main(argv):
    from playwright.sync_api import sync_playwright                  # noqa: PLC0415
    sys.path.insert(0, str(ROOT / "tests" / "browser"))
    from run import serve                                            # noqa: PLC0415

    port, shutdown = serve(ROOT)
    problems = []
    try:
        with sync_playwright() as pw:
            # Without the enabling flags and with the service off, this browser has no
            # adapter; the shim above takes the object away as well for the other shape.
            browser = pw.chromium.launch(headless=True, args=["--disable-features=WebGPU,WebGPUService"])
            try:
                for shape in SHIMS:
                    for lang, rel in PAGES.items():
                        page = browser.new_page()
                        if SHIMS[shape]:
                            page.add_init_script(SHIMS[shape])
                        page.goto(f"http://127.0.0.1:{port}/{rel}", wait_until="load")
                        # The badge is written after `probe()` answers, which is one await.
                        page.wait_for_function(
                            "() => document.getElementById('hero-out').children.length > 0",
                            timeout=30_000)
                        badge, lines = read(page)
                        seen = page.evaluate("() => ('gpu' in navigator) + '/' + (navigator.gpu ? 'value' : 'none')")
                        print(f"  {lang:2s} {shape:10s} navigator.gpu {seen:12s} badge {badge!r}")
                        for line in lines[:2]:
                            print(f"      {line[:120]}")
                        problems += judge(lang, shape, badge, lines)
                        # One press per shape, on the English page: the snippet is the
                        # same code and the claim is the same claim.
                        if lang == "en":
                            try:
                                print(f"      run → {press_run(page)}")
                            except Exception as e:                   # noqa: BLE001
                                problems.append(f"en/{shape}: Run never reached the learned line ({type(e).__name__})")
                        page.close()
            finally:
                browser.close()

            # **The other direction: the adapter is real and the readback is not.**
            with_gpu = pw.chromium.launch(headless=True, args=list(FLAGS))
            try:
                page = with_gpu.new_page()
                page.goto(f"http://127.0.0.1:{port}/{PAGES['en']}", wait_until="load")
                page.wait_for_function(
                    "() => document.getElementById('hero-out').children.length > 0", timeout=30_000)
                has_adapter = "no " not in read(page)[0]
                page.close()
                if not has_adapter:
                    print("  -- no-jspi: skipped, there is no adapter on this machine")
                else:
                    for lang, rel in PAGES.items():
                        page = with_gpu.new_page()
                        page.add_init_script("try { delete WebAssembly.Suspending; } catch (e) {}")
                        page.goto(f"http://127.0.0.1:{port}/{rel}", wait_until="load")
                        page.wait_for_function(
                            "() => document.getElementById('hero-out').children.length > 1",
                            timeout=30_000)
                        page.wait_for_timeout(1500)
                        badge, lines = read(page)
                        print(f"  {lang:2s} {'no-jspi':10s} badge {badge!r}")
                        for line in lines[:3]:
                            print(f"      {line[:120]}")
                        problems += judge_jspi(lang, badge, lines)
                        if lang == "en":
                            try:
                                print(f"      run → {press_run(page)}")
                            except Exception as e:                   # noqa: BLE001
                                problems.append(f"en/no-jspi: Run never reached the learned line ({type(e).__name__})")
                        page.close()
            finally:
                with_gpu.close()

            # The lesson pages, with no adapter: the snippet has to be one they can run,
            # and the ones about the device have to say so.
            lessons = pw.chromium.launch(headless=True, args=["--disable-features=WebGPU,WebGPUService"])
            try:
                for rel, kind in LESSONS.items():
                    page = lessons.new_page()
                    page.goto(f"http://127.0.0.1:{port}/{rel}", wait_until="load")
                    page.wait_for_timeout(1200)
                    blocks = page.locator(".runnable").count()
                    shown, _text, bad = press_lesson(page, kind)
                    print(f"  {kind:8s} {rel.split('site/')[1]:34s} {blocks} blocks · {shown.strip()!r}")
                    problems += [f"{rel}: {b}" for b in bad]
                    page.close()
            finally:
                lessons.close()
    finally:
        shutdown()

    if problems:
        print("\n".join("  " + p for p in problems))
        print("**the no-WebGPU screen is wrong** — see above")
        return 1
    print("**both languages say it themselves, in their own words, with no markup**")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
