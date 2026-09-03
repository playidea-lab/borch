"""Runs the comparison bench — borch.ts and TF.js, the same training step, one page.

    npm run build:ts
    uv run --with playwright python borch-ts/test/compare.py [--headed] [--only-infer]

It measures the same thing as `tests/browser/run.py --bench` without going through
Pyodide — borch.ts is JS a browser simply reads.
"""

import platform
import sys

import run as runner
from launch import browser as browser_of, refuse_if_software

PAGE = "/borch-ts/test/compare.html"
TIMEOUT_MS = 1_800_000


def conditions(adapter):
    """The line that has to travel with the number.

    **A time without its machine can be quoted but not contested.** The README's bench
    table says its three columns were measured *side by side on the same machine*, and
    that machine is written down nowhere — two sessions went looking for it separately
    and neither found it. The numbers survive only as ratios now, and no fourth column
    can ever be added to that table honestly.

    Nothing was lost by carelessness. The runner **had** the adapter in hand, used it to
    decide whether to refuse, and then printed the time alone — so what got copied into
    the document was the half that cannot be reproduced from.

    ## The host as well as the adapter

    The adapter is the GPU and the time is not only the GPU's: the training loop crosses
    into JS between steps. Two lines is the whole cost of not having this argument later.

    ## What is deliberately not in it

    Not the hostname, not the user. This line exists to be pasted into a **public**
    README, and `platform.system()` with `platform.machine()` says everything needed to
    re-run it while naming nobody.
    """
    return f"{adapter} / {platform.system()} {platform.machine()}"


def main(argv):
    dist = runner.ROOT / "borch-ts" / "dist" / "test" / "compare.js"
    if not dist.exists():
        print(f"no emit: {dist}\n  first: npm run build:ts", file=sys.stderr)
        return 2

    # The inference half needs the exported weights; torch makes them once.
    out = runner.ROOT / "borch-ts" / "test" / "out"
    if not all((out / f).exists() for f in ("resnet18_cifar.safetensors", "resnet18_cifar.onnx", "resnet18_cifar.probe.json")):
        import subprocess
        r = subprocess.run(["uv", "run", "--project", str(runner.ROOT), "--with", "torch", "--with", "onnx",
                            "python", "-W", "ignore", "tests/browser/export_resnet18.py"], cwd=str(runner.ROOT))
        if r.returncode:
            print("could not export the weights for the inference comparison", file=sys.stderr)
            return 2
    port, stop = runner.serve(runner.ROOT)
    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p, \
                browser_of(p, headed="--headed" in argv) as browser:
            page = browser.new_page()
            page.set_default_timeout(0)
            page.on("console", lambda m: print(f"  [browser] {m.text}")
                    if m.type == "error" else None)
            page.on("pageerror", lambda e: print(f"  [browser exception] {e}"))
            page.goto(f"http://127.0.0.1:{port}{PAGE}" + ("?only=infer" if "--only-infer" in argv else ""))
            page.wait_for_function("window.__borchCompare !== undefined",
                                   timeout=TIMEOUT_MS)
            result = page.evaluate("window.__borchCompare")
    finally:
        stop()

    if "error" in result:
        print(f"could not measure: {result['error']}", file=sys.stderr)
        return 1
    print(result["text"])
    # **This one refuses.** The device changes the time — ms/step measured on a CPU is
    # the software rasteriser's number rather than this library's. This repository recorded
    # that number for a while as "272× slower than the sister library", and that was a CPU
    # against a GPU rather than a comparison of libraries.
    if refuse_if_software(result.get("adapter"), "ms/step and the epoch time"):
        return 1
    # Printed **after** the refusal, so a line that says "measured on" only ever appears
    # under a number that was allowed to stand.
    print(f"\n  measured on: {conditions(result.get('adapter'))}"
          "\n  (carry this with the number — a time whose machine is unrecorded can be"
          "\n   quoted but not contested, and this table already has one such row.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
