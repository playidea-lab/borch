"""The review queue on CIFAR-10N, through the product's own path, on the wheel in a worker.

    uv run --with playwright python tests/browser/review_eval_py.py [--headed] [--build]

Five thousand CIFAR-10N training images as a zipped folder named by the labels people
gave (`aggre_label`, about 9 % wrong), `torch.ImageFiles` → frozen EfficientNet-B0
pre-logits → `torch.suspects` — the workbench's review order — judged against the clean
labels: AUROC, precision at 20/100/500, and how much of the queue catches 90 % of the
mislabels. The native run (tests/review_eval.py) is the reference; this asks whether the
browser path lands on the same numbers. Needs tests/browser/.cache/cifar10n_5k.zip, made
from ~/data/cifar10n (see the native script) — so this is a local check, not a nightly row.
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
    if not (ROOT / "tests" / "browser" / ".cache" / "cifar10n_5k.zip").exists():
        print("tests/browser/.cache/cifar10n_5k.zip is missing — make it from ~/data/cifar10n (tests/review_eval.py's data)", file=sys.stderr)
        return 2
    if refuse_if_screen_off("the review queue on CIFAR-10N"):
        return 1
    port, shutdown = serve(ROOT)
    url = f"http://127.0.0.1:{port}/tests/browser/review_eval_py.html?wheel=/{wheel}"
    profile = tempfile.mkdtemp(prefix="borch-review-")
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
    if refuse_if_software(adapter, "the review queue on CIFAR-10N"):
        return 1
    done = got.get("done") or ""
    auroc = re.search(r"AUROC ([0-9.]+)", done)
    p20 = re.search(r"p@20 ([0-9.]+)", done)
    # The native reference on the same subset: AUROC 0.917, p@20 0.65 (9.3 % noise).
    ok = bool(auroc and p20) and float(auroc.group(1)) >= 0.90 and float(p20.group(1)) >= 0.5 and "faults 0" in done
    print("**the browser's review queue puts the real mislabels first**" if ok else "**it did not** — see above")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
