"""The kernel bench — **one kernel against another, on the shapes you name.**

    npm run build:ts
    uv run --with playwright python tests/browser/kernel_bench.py [--bench=dw,fwd,bn,pad] [--shapes=16-16@96,32-32@48x8,...]  (cin-cout@side, x batch) [--headless]

Opens `kernel_bench.html`, which imports `borch-ts/dist` directly — no wheel, no bundle,
so a variant is a `build:ts` away. Each bench times its kernels round-robin, five rounds
of twenty dispatches under the timestamp profiler, keeps the minimum, and compares every
kernel's answer with the first's. Exit 1 on a fault, a WRONG, or a software adapter — a
number measured on the CPU is not a number. The page's header says what the numbers
mean against the training step (about half of what the same kernel costs inside it).
"""
import os
import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from first_run import FLAGS, ROOT, refuse_if_screen_off, serve  # noqa: E402
from launch import _headed, refuse_if_software  # noqa: E402

GIVE_UP_MS = 10 * 60 * 1000


def main(argv):
    from playwright.sync_api import sync_playwright
    headed = _headed("--headless" not in argv)
    bench = next((a.split("=", 1)[1] for a in argv if a.startswith("--bench=")), "dw,fwd")
    shapes = next((a.split("=", 1)[1] for a in argv if a.startswith("--shapes=")), "")
    reps = next((a.split("=", 1)[1] for a in argv if a.startswith("--reps=")), "")
    if refuse_if_screen_off("the kernel bench"):
        return 1
    port, shutdown = serve(ROOT)
    url = f"http://127.0.0.1:{port}/tests/browser/kernel_bench.html?bench={bench}" + (f"&shapes={shapes}" if shapes else "") + (f"&reps={reps}" if reps else "")
    profile = tempfile.mkdtemp(prefix="borch-kbench-")
    channel = os.environ.get("BORCH_CHROME_CHANNEL") or None
    try:
        with sync_playwright() as pw:
            context = pw.chromium.launch_persistent_context(profile, headless=not headed, channel=channel, args=list(FLAGS), timeout=60_000)
            try:
                page = context.new_page()
                page.goto(url, wait_until="load")
                page.wait_for_function("window.__bench !== undefined", timeout=GIVE_UP_MS, polling=200)
                got = page.evaluate("window.__bench")
            finally:
                context.close()
    finally:
        shutdown()
    print(got["text"])
    if got.get("error"):
        print("error: " + got["error"][:600])
        return 1
    if refuse_if_software(got.get("adapter"), "the kernel bench"):
        return 1
    print("**the minimum of five rounds, kernel against kernel**" if got["ok"] else "**a fault or a wrong answer** — see above")
    return 0 if got["ok"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
