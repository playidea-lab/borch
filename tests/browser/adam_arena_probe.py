"""the fused Adam arena against the per-parameter step.

    uv run --with playwright python tests/browser/adam_arena_probe.py [--headed]

Opens adam_arena_probe.html: one model trained twice on identical data, once with the Adam arena
(one adamStep over every parameter at once) and once with `suppressArena` forcing the per-parameter
step, compared bit for bit after several steps. Also checks the arena actually fired (far fewer
dispatches), so a silently-skipped arena cannot pass as a match.
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
    url = f"http://127.0.0.1:{port}/tests/browser/adam_arena_probe.html"
    profile = tempfile.mkdtemp(prefix="borch-adamarena-")
    channel = os.environ.get("BORCH_CHROME_CHANNEL") or None
    try:
        with sync_playwright() as pw:
            context = pw.chromium.launch_persistent_context(profile, headless=not headed, channel=channel, args=list(FLAGS), timeout=60_000)
            try:
                page = context.new_page()
                page.on("pageerror", lambda e: print(f"  [page] {e}"))
                page.goto(url, wait_until="load")
                page.wait_for_function("window.__adamArena !== undefined", timeout=GIVE_UP_MS, polling=250)
                got = page.evaluate("window.__adamArena")
            finally:
                context.close()
    finally:
        shutdown()
    print(got.get("text", ""))
    if got.get("error"):
        print(f"\nadam-arena could not be measured: {got['error'][:400]}", file=sys.stderr)
        return 1
    return 0 if got.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
