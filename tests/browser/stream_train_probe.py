"""Training a LoRA adapter on a streamed frozen backbone (Step 7 residency rule).

    uv run --with playwright python tests/browser/stream_train_probe.py [--headed] [--headless]

Trains a chain of LoRA-adapted Linear blocks one step two ways on the same parameters: fully
resident and taped, and with each block's frozen base streamed through a small window (forward
keeps only boundaries; backward refills and recomputes). The adapter gradients must agree within
the golden tolerance, and the window must hold at most a block or two of base weights. Imports
only borch-ts (no CDN), so it is adapter-independent and runs in CI on SwiftShader too.
"""
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "tests", "browser"))

from first_run import FLAGS, serve          # noqa: E402
from launch import _headed                  # noqa: E402

GIVE_UP_MS = 5 * 60 * 1000


def main(argv):
    from playwright.sync_api import sync_playwright
    headed = _headed("--headed" in argv)
    port, shutdown = serve(ROOT)
    url = f"http://127.0.0.1:{port}/tests/browser/stream_train_probe.html"
    profile = tempfile.mkdtemp(prefix="borch-streamtrain-")
    channel = os.environ.get("BORCH_CHROME_CHANNEL") or None
    try:
        with sync_playwright() as pw:
            context = pw.chromium.launch_persistent_context(profile, headless=not headed, channel=channel, args=list(FLAGS), timeout=60_000)
            try:
                page = context.new_page()
                page.on("pageerror", lambda e: print(f"  [page] {e}"))
                page.goto(url, wait_until="load")
                page.wait_for_function("window.__streamTrain !== undefined", timeout=GIVE_UP_MS, polling=250)
                got = page.evaluate("window.__streamTrain")
            finally:
                context.close()
    finally:
        shutdown()

    print(got.get("text", ""))
    if got.get("error"):
        print(f"\nstream-train could not be measured: {got['error'][:400]}", file=sys.stderr)
        return 1
    if not got.get("ok"):
        print(f"\nstream-train did not hold: within={got.get('allWithin')} loss={got.get('lossAgree')} "
              f"bounded={got.get('windowBounded')} faults={got.get('faults')}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
