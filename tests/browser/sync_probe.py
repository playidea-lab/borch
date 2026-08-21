"""Whether borch.ts can be called **synchronously** from Python on Pyodide.

    npm run build:ts
    uv run --with playwright python tests/browser/sync_probe.py --headed

## Why this is measured

**Whether the Python GPU implementation could move onto borch.ts instead of TF.js was
decided here.** The answer came back yes, and that is why `borch_webgpu` stands on borch.ts
today. This file is the measurement that made that decision and it still runs — if the
condition breaks, this goes red first.

The Python API has to be synchronous throughout. The old TF.js version was, for free,
thanks to `dataSync()` (which that side offered even though WebGPU has no synchronous read
API). borch.ts is `await stage.mapAsync(...)`, and asynchronous.

Laid on as is, it becomes `await loss.item()`, and that breaks **"run the tutorial's code
with only the import changed"**, which is this project's only claim. That claim is the
reason this library exists, so the road that breaks it cannot be taken.

Pyodide has `run_sync`, and JSPI (WebAssembly's promise integration) sits under it.
**Existing and working are different things** — that is what this repository keeps learning,
so it is measured.
"""

import sys

import run as runner
from launch import browser as browser_of

PAGE = "/tests/browser/sync_probe.html"
TIMEOUT_MS = 300_000


def main(argv):
    dist = runner.ROOT / "borch-ts" / "dist" / "src" / "index.js"
    if not dist.exists():
        print(f"no build: {dist}\n  first: npm run build:ts", file=sys.stderr)
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
            page.wait_for_function("window.__syncProbe !== undefined",
                                   timeout=TIMEOUT_MS)
            got = page.evaluate("window.__syncProbe")
    finally:
        stop()

    if "error" in got:
        print(f"could not measure: {got['error']}", file=sys.stderr)
        return 1

    print(got["text"])
    print()
    if got["sync"]:
        print("**it works** — the floor `borch_webgpu` stands on is intact.")
        return 0
    if not got["async"]:
        print("even the async path failed — a different problem, unrelated to synchrony. See the line above.")
        return 1
    print("**it does not work** — the Python API cannot be kept synchronous.\n"
          "  `borch_webgpu` stands on this, so this means that package is broken\n"
          "  right now. Accepting `await loss.item()` here would remove\n"
          "  'run it with only the import changed' entirely.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
