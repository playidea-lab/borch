"""f32 reduction accuracy against a double reference — a measurement, not a check.

    uv run --with playwright python tests/browser/precision_probe.py [--headed]

Opens precision_probe.html: a plain sum, and BatchNorm's mean and one-pass variance, each against
a Kahan/f64 CPU reference, on inputs with a small mean and a large one (the catastrophic-
cancellation trap for mean(x²) − mean(x)²). It prints relative errors; it judges nothing beyond
faults 0. `docs/SCALE.md` reads the numbers to decide whether a stable variance is worth building.
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
    url = f"http://127.0.0.1:{port}/tests/browser/precision_probe.html"
    profile = tempfile.mkdtemp(prefix="borch-precision-")
    channel = os.environ.get("BORCH_CHROME_CHANNEL") or None
    try:
        with sync_playwright() as pw:
            context = pw.chromium.launch_persistent_context(profile, headless=not headed, channel=channel, args=list(FLAGS), timeout=60_000)
            try:
                page = context.new_page()
                page.on("pageerror", lambda e: print(f"  [page] {e}"))
                page.goto(url, wait_until="load")
                page.wait_for_function("window.__precision !== undefined", timeout=GIVE_UP_MS, polling=250)
                got = page.evaluate("window.__precision")
            finally:
                context.close()
    finally:
        shutdown()
    print(got.get("text", ""))
    if got.get("error"):
        print(f"\nprecision could not be measured: {got['error'][:400]}", file=sys.stderr)
        return 1
    return 0 if got.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
