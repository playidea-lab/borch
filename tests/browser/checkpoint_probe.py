"""Gradient checkpointing: the recomputed backward against the fully-taped one.

    uv run --with playwright python tests/browser/checkpoint_probe.py [--headed] [--headless]

Step 6 of `docs/SCALE.md`. A chain of blocks is run twice on the same parameters — once
fully taped, once with every block wrapped in `checkpoint` — and the page asserts the
gradients agree within the golden tolerance (a wrong recompute is a silently wrong
gradient) and that the buffers held after the forward fall, which is the point of the
feature. Adapter-independent: the values and the buffer counts are the code path's, not
the device's, so this runs on SwiftShader in CI as it does on a GPU.
"""
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "tests", "browser"))

from first_run import FLAGS, serve          # noqa: E402
from launch import _headed                  # noqa: E402

GIVE_UP_MS = 4 * 60 * 1000


def main(argv):
    from playwright.sync_api import sync_playwright
    headed = _headed("--headed" in argv)    # a window unless --headless / BORCH_HEADLESS
    port, shutdown = serve(ROOT)
    url = f"http://127.0.0.1:{port}/tests/browser/checkpoint_probe.html"
    profile = tempfile.mkdtemp(prefix="borch-checkpoint-")
    channel = os.environ.get("BORCH_CHROME_CHANNEL") or None
    try:
        with sync_playwright() as pw:
            context = pw.chromium.launch_persistent_context(profile, headless=not headed, channel=channel, args=list(FLAGS), timeout=60_000)
            try:
                page = context.new_page()
                page.on("pageerror", lambda e: print(f"  [page] {e}"))
                page.goto(url, wait_until="load")
                page.wait_for_function("window.__checkpoint !== undefined", timeout=GIVE_UP_MS, polling=250)
                got = page.evaluate("window.__checkpoint")
            finally:
                context.close()
    finally:
        shutdown()

    print(got.get("text", ""))
    if got.get("error"):
        print(f"\ncheckpoint could not be measured: {got['error'][:400]}", file=sys.stderr)
        return 1
    if not got.get("ok"):
        which = [k for k in ("gradsAgree", "lossAgree", "heldFell", "refusedCapture") if not got.get(k)]
        if got.get("faults"):
            which.append(f"{got['faults']} fault(s)")
        print(f"\ncheckpoint did not hold: {', '.join(which)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
