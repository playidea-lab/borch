"""How long does one value take to come back from the GPU — bare WebGPU, no borch.

    uv run --with playwright python tests/browser/readback_probe.py [--headed] [--flags=a,b] [--no-vulkan]

Runs `readback_probe.html` under several browser configurations and prints the
medians side by side: the runner's flags as they are; plus `--disable-gpu-vsync`;
plus `--disable-frame-rate-limit`; headless. The question it was written for: on a
Linux NVIDIA machine every readback waited about one second (measured through
`first_run.py` and the learner's clock), and this asks whether that second belongs
to the display's tick, to a timer, or to the driver.
"""
import os
import sys
import tempfile
import time

from first_run import FLAGS, ROOT, serve

PAGE = "/tests/browser/readback_probe.html"
GIVE_UP_MS = 120_000


def one(pw, url, label, flags, headless):
    profile = tempfile.mkdtemp(prefix="borch-readback-")
    channel = os.environ.get("BORCH_CHROME_CHANNEL") or None
    context = pw.chromium.launch_persistent_context(profile, headless=headless, channel=channel, args=list(flags), timeout=60_000)
    try:
        page = context.new_page()
        page.goto(url, wait_until="load")
        page.wait_for_function("window.__readback !== undefined", timeout=GIVE_UP_MS)
        got = page.evaluate("window.__readback")
    finally:
        context.close()
    print(f"== {label}  (flags: {' '.join(flags) or '(none)'} · {'headless' if headless else 'headed'})")
    if "error" in got:
        print("  " + got["error"].splitlines()[0])
        return None
    for line in got["text"].splitlines():
        print("  " + line)
    return got["medians"]


def python_path(argv):
    """`--python`: the same question of the Python path, under the runner's flags only."""
    from playwright.sync_api import sync_playwright
    headed = "--headed" in argv
    port, shutdown = serve(ROOT)
    url = f"http://127.0.0.1:{port}/tests/browser/readback_probe_py.html"
    profile = tempfile.mkdtemp(prefix="borch-readback-py-")
    channel = os.environ.get("BORCH_CHROME_CHANNEL") or None
    try:
        with sync_playwright() as pw:
            context = pw.chromium.launch_persistent_context(profile, headless=not headed, channel=channel, args=list(FLAGS), timeout=60_000)
            try:
                page = context.new_page()
                page.goto(url, wait_until="load")
                page.wait_for_function("window.__readbackPy !== undefined", timeout=GIVE_UP_MS * 3)
                got = page.evaluate("window.__readbackPy")
            finally:
                context.close()
    finally:
        shutdown()
    if "error" in got:
        print(got["error"]); return 1
    print(got["text"])
    m, x = got["medians"], got["maxes"]
    print(f"\n{'path':60} {'median':>8} {'max':>8}")
    for key, label in (("js", "JS await toArray() after idle"), ("py", "Python run_sync after idle"),
                       ("pyTimer", "Python run_sync, JS timer alive"), ("pyLoop", "Python 15 ops then run_sync")):
        print(f"{label:60} {m[key]:8.1f} {x[key]:8.1f}")
    return 0


def main(argv):
    from playwright.sync_api import sync_playwright
    if "--python" in argv:
        return python_path(argv)
    headed = "--headed" in argv
    base = list(FLAGS)
    if "--no-vulkan" in argv:
        base = [f for f in base if "Vulkan" not in f]
    extra = next((a.split("=", 1)[1].split(",") for a in argv if a.startswith("--flags=")), [])
    port, shutdown = serve(ROOT)
    url = f"http://127.0.0.1:{port}{PAGE}"
    rows = []
    try:
        with sync_playwright() as pw:
            configs = [
                ("runner's flags", base + extra, not headed),
                ("+ --disable-gpu-vsync", base + extra + ["--disable-gpu-vsync"], not headed),
                ("+ --disable-frame-rate-limit", base + extra + ["--disable-frame-rate-limit"], not headed),
                ("+ both", base + extra + ["--disable-gpu-vsync", "--disable-frame-rate-limit"], not headed),
                ("headless, runner's flags", base + extra, True),
            ]
            for label, flags, headless in configs:
                m = one(pw, url, label, flags, headless)
                rows.append((label, m))
                time.sleep(1)
    finally:
        shutdown()
    print()
    print(f"{'configuration':32} {'rAF':>7} {'map':>7} {'workDone':>9} {'disp→map':>9} {'rAF+map':>8} {'timer+map':>10} {'idle→map':>9} {'idle,max':>9} {'idle+timer':>11} {'its max':>8} {'pipeline':>9} {'pipe max':>9} {'1st disp':>9} {'its max':>8} {'pipeAsync':>10}")
    for label, m in rows:
        if m is None:
            print(f"{label:32} (blew up)")
            continue
        print(f"{label:32} {m['raf']:7.1f} {m['map']:7.1f} {m['workDone']:9.1f} {m['dispatchMap']:9.1f} {m['rafMap']:8.1f} {m['timerMap']:10.1f} {m['idleMap']:9.1f} {m['idleMax']:9.1f} {m['idleTimerMap']:11.1f} {m['idleTimerMax']:8.1f} {m['pipeline']:9.1f} {m['pipelineMax']:9.1f} {m['firstDispatch']:9.1f} {m['firstDispatchMax']:8.1f} {m['pipelineAsync']:10.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
