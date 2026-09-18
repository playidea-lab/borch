"""fused BatchNorm+ReLU against the unfused composition.

    uv run --with playwright python tests/browser/bn_relu_probe.py [--headed]

Opens bn_relu_probe.html: the fused BatchNorm→ReLU path (`batchNormFused(relu=true)`, whose
backward recomputes the ReLU mask and the standardised value instead of reading stored copies)
against the same BatchNorm with a separate `.relu()`. The golden's op-level cases cover
`batchNormFused` with relu off but not the fused-relu path the U-Net uses, so this is where its
correctness is held — forward and in every gradient.
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
    url = f"http://127.0.0.1:{port}/tests/browser/bn_relu_probe.html"
    profile = tempfile.mkdtemp(prefix="borch-bnrelu-")
    channel = os.environ.get("BORCH_CHROME_CHANNEL") or None
    try:
        with sync_playwright() as pw:
            context = pw.chromium.launch_persistent_context(profile, headless=not headed, channel=channel, args=list(FLAGS), timeout=60_000)
            try:
                page = context.new_page()
                page.on("pageerror", lambda e: print(f"  [page] {e}"))
                page.goto(url, wait_until="load")
                page.wait_for_function("window.__bnRelu !== undefined", timeout=GIVE_UP_MS, polling=250)
                got = page.evaluate("window.__bnRelu")
            finally:
                context.close()
    finally:
        shutdown()
    print(got.get("text", ""))
    if got.get("error"):
        print(f"\nbn-relu could not be measured: {got['error'][:400]}", file=sys.stderr)
        return 1
    return 0 if got.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
