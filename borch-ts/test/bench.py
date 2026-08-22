"""Runs borch.ts's training-step bench in the browser.

    npm run build:ts
    uv run --with playwright python borch-ts/test/bench.py [--headed]

It measures the same thing as `tests/browser/run.py --bench` without going through
Pyodide — borch.ts is JS a browser simply reads.
"""

import sys

import run as runner
from launch import browser as browser_of, refuse_if_software

PAGE = "/borch-ts/test/bench.html"
TIMEOUT_MS = 1_800_000


def main(argv):
    dist = runner.ROOT / "borch-ts" / "dist" / "test" / "bench.js"
    if not dist.exists():
        print(f"no emit: {dist}\n  first: npm run build:ts", file=sys.stderr)
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
            page.goto(f"http://127.0.0.1:{port}{PAGE}")
            page.wait_for_function("window.__borchBench !== undefined",
                                   timeout=TIMEOUT_MS)
            result = page.evaluate("window.__borchBench")
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
