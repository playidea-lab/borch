"""Runs the peers bench — borch.ts, jax-js and Burn, the same training step, one page.

    npm run build:ts
    uv run --with playwright python borch-ts/test/compare_peers.py [--headed] [--skip=jax,burn]

The two other libraries that train in a browser on WebGPU (the 2026-09-19 survey in
`docs/BOOK.md`), on the ResNet-18 (CIFAR) step of `bench.ts`, one after the other on one
page. jax-js arrives from esm.sh at a pinned version; Burn from the wasm bundle in
`tests/browser/burn_resnet18/pkg`, built from a scratch crate (`docs/BOOK.md` says how). A
peer that is absent (no network, no bundle) prints why and the others still stand.
"""

import sys

import run as runner
from compare import conditions
from launch import browser as browser_of, refuse_if_software

PAGE = "/borch-ts/test/compare_peers.html"
TIMEOUT_MS = 1_800_000


def main(argv):
    dist = runner.ROOT / "borch-ts" / "dist" / "test" / "compare_peers.js"
    if not dist.exists():
        print(f"no emit: {dist}\n  first: npm run build:ts", file=sys.stderr)
        return 2
    skip = next((a.split("=", 1)[1] for a in argv if a.startswith("--skip=")), "")
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
            page.goto(f"http://127.0.0.1:{port}{PAGE}" + (f"?skip={skip}" if skip else ""))
            page.wait_for_function("window.__borchPeers !== undefined", timeout=TIMEOUT_MS)
            result = page.evaluate("window.__borchPeers")
    finally:
        stop()

    print(result["text"])
    if "error" in result:
        print(f"\ncould not finish: {result['error'][:600]}", file=sys.stderr)
        return 1
    if refuse_if_software(result.get("adapter"), "ms/step"):
        return 1
    print(f"\n  measured on: {conditions(result.get('adapter'))}"
          "\n  (carry this with the number — a time whose machine is unrecorded can be"
          "\n   quoted but not contested.)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
