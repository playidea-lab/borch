"""`torch.compiled` in JavaScript, held to the eager step — and the ResNet-18 memory number.

    npm run build:ts
    uv run --with playwright python borch-ts/test/capture.py [--headed]

The binding's `capture:py` for borch.ts (`docs/COMPILER.md` Step 6): an MLP over two batch
shapes, bit for bit with `check: true` on both recordings; then the ResNet-18 of `bench.ts`
at batch 16, three replays against three eager steps bit for bit, with the plan's bytes and
the replay's clock printed beside the eager step's. Refuses a software adapter before
measuring: the clock and the bytes are the GPU's.
"""

import sys

import run as runner
from launch import browser as browser_of, refuse_if_software
from verdict import verdict

PAGE = "/borch-ts/test/capture.html"
TIMEOUT_MS = 600_000
ADAPTER_MS = 120_000


def main(argv):
    runner.require_fresh_dist()
    dist = runner.ROOT / "borch-ts" / "dist" / "test" / "capture.js"
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
            page.wait_for_function("window.__borchAdapter !== undefined || window.__borchCaptureTs !== undefined",
                                   timeout=ADAPTER_MS)
            early = page.evaluate("window.__borchAdapter")
            if early is not None and refuse_if_software(early, "the compiled step's clock and bytes"):
                return 1
            page.wait_for_function("window.__borchCaptureTs !== undefined", timeout=TIMEOUT_MS)
            result = page.evaluate("window.__borchCaptureTs")
    finally:
        stop()

    if "error" in result:
        print(f"**the compiled-step check blew up**\n{result['error']}", file=sys.stderr)
        return 1
    print(result["text"])
    return verdict(result, "the compiled step")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
