"""Fine-tune a >=300 MB backbone in a bounded window — the Step 7 gate.

    uv run --with playwright python tests/browser/finetune.py [--headed]
        [--imgPerClass=200] [--epochs=5] [--batch=16] [--r=8]

Loads ViT-Base (346 MB) from the hub, trains a linear-probe frozen-head baseline, then LoRA
fine-tunes with the frozen blocks offloaded and streamed through a bounded window, and checks the
fine-tune is no worse than the baseline while the backbone is never fully resident during
training. Needs the network (hub + esm.sh) and a real GPU; heavy, so the timeout is generous.
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from first_run import FLAGS, ROOT, refuse_if_screen_off, serve  # noqa: E402
from launch import _headed, refuse_if_software  # noqa: E402

GIVE_UP_MS = 20 * 60 * 1000
PASS_ARGS = ("imgPerClass", "epochs", "batch", "heldFrac", "r")


def main(argv):
    from playwright.sync_api import sync_playwright
    headed = _headed("--headed" in argv)
    query = "&".join(a[2:] for a in argv if any(a.startswith(f"--{k}=") for k in PASS_ARGS))
    if refuse_if_screen_off("the fine-tune gate"):
        return 1
    port, shutdown = serve(ROOT)
    url = f"http://127.0.0.1:{port}/tests/browser/finetune.html" + (f"?{query}" if query else "")
    profile = tempfile.mkdtemp(prefix="borch-finetune-")
    channel = os.environ.get("BORCH_CHROME_CHANNEL") or None
    try:
        with sync_playwright() as pw:
            context = pw.chromium.launch_persistent_context(profile, headless=not headed, channel=channel, args=list(FLAGS), timeout=60_000)
            try:
                page = context.new_page()
                page.on("pageerror", lambda e: print(f"  [page] {e}"))
                page.goto(url, wait_until="load")
                page.wait_for_function("window.__finetune !== undefined", timeout=GIVE_UP_MS, polling=1000)
                got = page.evaluate("window.__finetune")
            finally:
                context.close()
    finally:
        shutdown()

    print(got.get("text", ""))
    if got.get("error"):
        print(f"\nfine-tune could not be measured: {got['error'][:600]}", file=sys.stderr)
        return 1
    if refuse_if_software(got.get("adapter"), "the fine-tune gate"):
        return 1
    if not got.get("ok"):
        print(f"\ngate did not hold: beatsBaseline={got.get('beatsBaseline')} int8Close={got.get('int8Close')} "
              f"windowBounded={got.get('windowBounded')} peakUnderBackbone={got.get('peakUnderBackbone')} "
              f"adapterSmall={got.get('adapterSmall')} faults={got.get('faults')}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
