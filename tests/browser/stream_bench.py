"""Streaming vs resident — where a training step's time goes (a measurement).

    uv run --with playwright python tests/browser/stream_bench.py [--headed] [--d=384] [--blocks=12] [--batch=16]

Opens stream_bench.html: the same LoRA stack trained one step four ways (resident/streamed ×
forward-only/full), so the cost of streaming is a number. A measurement, not a check — it prints
a breakdown and always exits 0 unless it could not run at all.
"""
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "tests", "browser"))

from first_run import FLAGS, serve          # noqa: E402
from launch import _headed                  # noqa: E402

GIVE_UP_MS = 5 * 60 * 1000
PASS = ("d", "blocks", "batch", "iters")


def main(argv):
    from playwright.sync_api import sync_playwright
    headed = _headed("--headed" in argv)
    query = "&".join(a[2:] for a in argv if any(a.startswith(f"--{k}=") for k in PASS))
    port, shutdown = serve(ROOT)
    url = f"http://127.0.0.1:{port}/tests/browser/stream_bench.html" + (f"?{query}" if query else "")
    profile = tempfile.mkdtemp(prefix="borch-streambench-")
    channel = os.environ.get("BORCH_CHROME_CHANNEL") or None
    try:
        with sync_playwright() as pw:
            context = pw.chromium.launch_persistent_context(profile, headless=not headed, channel=channel, args=list(FLAGS), timeout=60_000)
            try:
                page = context.new_page()
                page.on("pageerror", lambda e: print(f"  [page] {e}"))
                page.goto(url, wait_until="load")
                page.wait_for_function("window.__streamBench !== undefined", timeout=GIVE_UP_MS, polling=250)
                got = page.evaluate("window.__streamBench")
            finally:
                context.close()
    finally:
        shutdown()
    print(got.get("text", ""))
    if got.get("error"):
        print(f"\nstream-bench could not run: {got['error'][:400]}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
