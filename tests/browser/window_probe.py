"""The frozen-weight window primitive: offset slices of one buffer fill and read back.

    uv run --with playwright python tests/browser/window_probe.py [--headed] [--headless]

Step 3 of `docs/SCALE.md`. A window is one STORAGE buffer holding several weights end to
end, filled through a staging buffer and copyRange, each bound as an offset slice. The page
puts two arrays into two slots, reads each slice at its offset through a kernel, and checks
the values survive — then refills a fresh window. Adapter-independent (copyRange and slice
bindings are core WebGPU), so it runs on Apple, the RTX 5080, and SwiftShader alike.
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
    headed = _headed("--headed" in argv)
    port, shutdown = serve(ROOT)
    url = f"http://127.0.0.1:{port}/tests/browser/window_probe.html"
    profile = tempfile.mkdtemp(prefix="borch-window-")
    channel = os.environ.get("BORCH_CHROME_CHANNEL") or None
    try:
        with sync_playwright() as pw:
            context = pw.chromium.launch_persistent_context(profile, headless=not headed, channel=channel, args=list(FLAGS), timeout=60_000)
            try:
                page = context.new_page()
                page.on("pageerror", lambda e: print(f"  [page] {e}"))
                page.goto(url, wait_until="load")
                page.wait_for_function("window.__window !== undefined", timeout=GIVE_UP_MS, polling=250)
                got = page.evaluate("window.__window")
            finally:
                context.close()
    finally:
        shutdown()

    print(got.get("text", ""))
    if got.get("error"):
        print(f"\nwindow could not be measured: {got['error'][:400]}", file=sys.stderr)
        return 1
    if not got.get("ok"):
        which = [k for k in ("sameBuffer", "offsetsOk", "aOk", "bOk", "cOk", "subEq", "sclEq", "refusedGeneric") if not got.get(k)]
        if got.get("faults"):
            which.append(f"{got['faults']} fault(s)")
        print(f"\nwindow did not hold: {', '.join(which) or 'used=' + str(got.get('used'))}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
