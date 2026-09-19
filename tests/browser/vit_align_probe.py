"""the ViT's aligned tokens against the unaligned ones.

    uv run --with playwright python tests/browser/vit_align_probe.py [--headed]

Opens vit_align_probe.html: bimm's ViT-tiny with its token row padded to a multiple of eight
(so attention's batched matmuls take the subgroup kernel) against the same model unpadded —
features, logits and a gradient, forward and backward, within kernel rounding; the feature
shape stays [B, 197, D]. Skips cleanly where there are no subgroup matrices. Reads bimm from
`node_modules/bimm-ts/dist`, so a local bimm build copied there is what it tests.
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
    url = f"http://127.0.0.1:{port}/tests/browser/vit_align_probe.html"
    profile = tempfile.mkdtemp(prefix="borch-vitalign-")
    channel = os.environ.get("BORCH_CHROME_CHANNEL") or None
    try:
        with sync_playwright() as pw:
            context = pw.chromium.launch_persistent_context(profile, headless=not headed, channel=channel, args=list(FLAGS), timeout=60_000)
            try:
                page = context.new_page()
                page.on("pageerror", lambda e: print(f"  [page] {e}"))
                page.goto(url, wait_until="load")
                page.wait_for_function("window.__vitAlign !== undefined", timeout=GIVE_UP_MS, polling=250)
                got = page.evaluate("window.__vitAlign")
            finally:
                context.close()
    finally:
        shutdown()
    print(got.get("text", ""))
    if got.get("error"):
        print(f"\nvit-align could not be measured: {got['error'][:400]}", file=sys.stderr)
        return 1
    return 0 if got.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
