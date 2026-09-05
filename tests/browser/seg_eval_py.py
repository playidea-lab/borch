"""Segmentation in the browser: the U-Net of tests/seg_eval.py trained on Kvasir-SEG
through the wheel, in a worker, on this tab's GPU — twice: on the clean masks, for the
held-out IoU against torch's, and with a fifth of the masks corrupted, for whether its
review queue (one minus the IoU between the model's mask and the given one) puts the
wrong masks first.

    uv run --with playwright python tests/browser/seg_eval_py.py [--headed] [--build] [--train=800] [--epochs=30]

Needs tests/browser/.cache/kvasir_96.npz (`python tests/seg_eval.py prepare`). A run
shorter than the native reference reports its numbers and judges nothing.
"""
import glob
import os
import pathlib
import re
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from first_run import FLAGS, ROOT, refuse_if_screen_off, serve  # noqa: E402
from launch import refuse_if_software  # noqa: E402
from wheel_probe import wheel_is_stale  # noqa: E402

GIVE_UP_MS = 15 * 60 * 1000


def main(argv):
    from playwright.sync_api import sync_playwright
    headed = "--headed" in argv
    if "--build" in argv:
        import subprocess
        for cmd in (["npm", "run", "-s", "bundle:py"], ["uv", "build", "--wheel", "-q"]):
            r = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
            if r.returncode:
                print(f"{' '.join(cmd)} failed:\n{(r.stdout + r.stderr)[-800:]}", file=sys.stderr)
                return 2
    found = sorted(glob.glob(str(ROOT / "dist" / "pyborch-*.whl")))
    if not found:
        print("no wheel under dist/ — run with --build", file=sys.stderr)
        return 2
    wheel = os.path.relpath(found[-1], ROOT)
    if wheel_is_stale(wheel):
        print(f"{wheel} is older than the sources — run with --build", file=sys.stderr)
        return 2
    train = int(next((a.split("=", 1)[1] for a in argv if a.startswith("--train=")), "800"))
    epochs = int(next((a.split("=", 1)[1] for a in argv if a.startswith("--epochs=")), "30"))
    if not (ROOT / "tests" / "browser" / ".cache" / "kvasir_96.npz").exists():
        print("tests/browser/.cache/kvasir_96.npz is missing — `python tests/seg_eval.py prepare` makes it from ~/data/kvasir", file=sys.stderr)
        return 2
    if refuse_if_screen_off("segmentation on Kvasir-SEG"):
        return 1
    port, shutdown = serve(ROOT)
    url = f"http://127.0.0.1:{port}/tests/browser/seg_eval_py.html?wheel=/{wheel}&train={train}&epochs={epochs}"
    profile = tempfile.mkdtemp(prefix="borch-seg-")
    channel = os.environ.get("BORCH_CHROME_CHANNEL") or None
    try:
        with sync_playwright() as pw:
            context = pw.chromium.launch_persistent_context(profile, headless=not headed, channel=channel, args=list(FLAGS), timeout=60_000)
            try:
                page = context.new_page()
                page.goto(url, wait_until="load")
                page.wait_for_function("window.__wheel !== undefined", timeout=GIVE_UP_MS, polling=200)
                got = page.evaluate("window.__wheel")
            finally:
                context.close()
    finally:
        shutdown()
    print(got["text"])
    if got.get("error"):
        print("error: " + got["error"][:600])
        return 1
    adapter = None
    for line in got["text"].splitlines():
        if "adapter " in line:
            adapter = line.split("adapter ", 1)[1].split(" ·")[0]
    if refuse_if_software(adapter, "segmentation on Kvasir-SEG"):
        return 1
    done = got.get("done") or ""
    auroc = re.search(r"AUROC ([0-9.]+)", done)
    clean_iou = re.search(r"clean masks: test IoU ([0-9.]+)", done)
    # Two runs, each judged where its number is stable. The held-out IoU after thirty
    # epochs on corrupted masks swings with the arithmetic: torch over six seeds spans
    # 0.40–0.50, and the browser's own deterministic run moved 0.42 → 0.31 when the conv
    # kernels' summation order changed, with the training loss unchanged. On clean masks
    # torch lands at 0.55 ± 0.01, so that run carries the IoU verdict; the corrupted run
    # carries the queue's (torch 0.88–0.93 across the same seeds).
    full = train >= 800 and epochs >= 30
    ok = bool(auroc and clean_iou) and "faults 0" in done and (
        not full or (float(clean_iou.group(1)) >= 0.45 and float(auroc.group(1)) >= 0.85))
    print(("**the browser learns the masks torch learns, and its queue puts the wrong ones first**" if full else "**a short run — numbers only**") if ok else "**it did not** — see above")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
