"""The labelling tool's envelope: a 224 px pretrained backbone and a head, on this GPU.

    uv run --with playwright python tests/browser/envelope.py [--headed] [--model=imagenet-efficientnet-b0]

Fetches the backbone from the hub (network), times its forward per image at 224 px,
trains a linear head on 200 cached features, orders 200 rows by cosine neighbours, and
says whether the three together fit in two minutes — the go/no-go of the tool.
"""
import os
import sys
import tempfile

from first_run import FLAGS, ROOT, refuse_if_screen_off, serve
from launch import _headed, refuse_if_software

GIVE_UP_MS = 6 * 60 * 1000


def main(argv):
    from playwright.sync_api import sync_playwright
    headed = _headed("--headed" in argv)   # a window unless --headless / BORCH_HEADLESS: headless is SwiftShader here
    model = next((a.split("=", 1)[1] for a in argv if a.startswith("--model=")), "imagenet-efficientnet-b0")
    if refuse_if_screen_off("the envelope"):
        return 1
    port, shutdown = serve(ROOT)
    url = f"http://127.0.0.1:{port}/tests/browser/envelope.html?model={model}"
    profile = tempfile.mkdtemp(prefix="borch-envelope-")
    channel = os.environ.get("BORCH_CHROME_CHANNEL") or None
    try:
        with sync_playwright() as pw:
            context = pw.chromium.launch_persistent_context(profile, headless=not headed, channel=channel, args=list(FLAGS), timeout=60_000)
            try:
                page = context.new_page()
                page.on("pageerror", lambda e: print(f"  [page] {e}"))
                page.goto(url, wait_until="load")
                page.wait_for_function("window.__envelope !== undefined", timeout=GIVE_UP_MS, polling=500)
                got = page.evaluate("window.__envelope")
            finally:
                context.close()
    finally:
        shutdown()
    if "error" in got:
        print(got["error"]); return 1
    print(got["text"])
    if refuse_if_software(got.get("adapter"), "the envelope"):
        return 1
    return 0 if got["faults"] == 0 and got["total"] <= 120 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
