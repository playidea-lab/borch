"""The envelope, part two: a model of one's own, trained end to end at 128 and 224 px.

    uv run --with playwright python tests/browser/envelope2.py [--headed] [--sizes=128,224]

A small CNN written as torch writes it, three synthetic classes, five epochs at batch
16: seconds, ms/step, held-out accuracy, faults, and the ONNX round trip through ORT
Web — in TypeScript, then the same model and data in Python through the binding.
"""
import os
import sys
import tempfile

from first_run import FLAGS, ROOT, refuse_if_screen_off, serve
from launch import refuse_if_software

GIVE_UP_MS = 12 * 60 * 1000


def main(argv):
    from playwright.sync_api import sync_playwright
    headed = "--headed" in argv
    sizes = next((a.split("=", 1)[1] for a in argv if a.startswith("--sizes=")), "128,224")
    if refuse_if_screen_off("the envelope, part two"):
        return 1
    port, shutdown = serve(ROOT)
    url = f"http://127.0.0.1:{port}/tests/browser/envelope2.html?sizes={sizes}"
    profile = tempfile.mkdtemp(prefix="borch-envelope2-")
    channel = os.environ.get("BORCH_CHROME_CHANNEL") or None
    try:
        with sync_playwright() as pw:
            context = pw.chromium.launch_persistent_context(profile, headless=not headed, channel=channel, args=list(FLAGS), timeout=60_000)
            try:
                page = context.new_page()
                page.on("pageerror", lambda e: print(f"  [page] {e}"))
                page.goto(url, wait_until="load")
                page.wait_for_function("window.__envelope2 !== undefined", timeout=GIVE_UP_MS, polling=500)
                got = page.evaluate("window.__envelope2")
            finally:
                context.close()
    finally:
        shutdown()
    if "error" in got:
        print(got["error"]); return 1
    print(got["text"])
    if refuse_if_software(got.get("adapter"), "the envelope, part two"):
        return 1
    ok = got["faults"] == 0 and all(r["acc"] >= 0.9 and r["gap"] <= 1e-3 for r in got["results"].values()) and "held-out" in got["python"]
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
