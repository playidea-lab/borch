"""A real model's blocks streamed through the frozen-weight window (Step 3 ④b).

    uv run --with playwright python tests/browser/stream_model_probe.py [--headed] [--headless]

Builds a bimm ResNet-18, takes layer1 (a Sequential of BasicBlocks — the real conv/BN/
ReLU/residual mix), and streams it through a small window two ways: hand-built StreamBlocks
(conv kernels only), and the `streamSequence` adapter (each real block's frozen params).
Both must match the resident layer output bit for bit. Needs bimm-ts@0.12.0 (the plan-table
export release) from esm.sh, so it runs where the CDN is reachable.
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
    url = f"http://127.0.0.1:{port}/tests/browser/stream_model_probe.html"
    profile = tempfile.mkdtemp(prefix="borch-streammodel-")
    channel = os.environ.get("BORCH_CHROME_CHANNEL") or None
    try:
        with sync_playwright() as pw:
            context = pw.chromium.launch_persistent_context(profile, headless=not headed, channel=channel, args=list(FLAGS), timeout=60_000)
            try:
                page = context.new_page()
                page.on("pageerror", lambda e: print(f"  [page] {e}"))
                page.goto(url, wait_until="load")
                page.wait_for_function("window.__streamModel !== undefined", timeout=GIVE_UP_MS, polling=250)
                got = page.evaluate("window.__streamModel")
            finally:
                context.close()
    finally:
        shutdown()

    print(got.get("text", ""))
    if got.get("error"):
        print(f"\nstream-model could not be measured: {got['error'][:400]}", file=sys.stderr)
        return 1
    if not got.get("ok"):
        print(f"\nstream-model did not hold: eq={got.get('eq')} eqAdapter={got.get('eqAdapter')} "
              f"faults={got.get('faults')}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
