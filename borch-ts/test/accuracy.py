"""Measures borch.ts's CIFAR-10 accuracy in the browser.

    npm run build:ts
    uv run --with playwright python borch-ts/test/accuracy.py --headed [--epochs 10]

The same yardstick as `tests/browser/run.py --accuracy`, using the same data files:
`cifar-batch1.bin` and `cifar-batch-test.bin` have to be at the repository root.

**The window has to be shown.** Headless gives a software adapter and then this
measurement takes hours — the page prints the adapter's name first, so it stays with the
result.
"""

import sys

import run as runner
from launch import browser as browser_of, refuse_if_software

TIMEOUT_MS = 7_200_000


def main(argv):
    dist = runner.ROOT / "borch-ts" / "dist" / "test" / "accuracy.js"
    if not dist.exists():
        print(f"no emit: {dist}\n  first: npm run build:ts", file=sys.stderr)
        return 2
    for name in ("cifar-batch1.bin", "cifar-batch-test.bin"):
        if not (runner.ROOT / name).exists():
            print(f"no data: {runner.ROOT / name}\n"
                  "  put the CIFAR-10 binaries at the repository root.", file=sys.stderr)
            return 2

    def opt(flag, default):
        return argv[argv.index(flag) + 1] if flag in argv else default

    query = f"?epochs={opt('--epochs', '10')}"
    if "--images" in argv:
        query += f"&images={opt('--images', '0')}"
    if "--augment" in argv:
        query += f"&augment={opt('--augment', 'off')}"

    port, stop = runner.serve(runner.ROOT)
    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p, \
                browser_of(p, headed="--headed" in argv) as browser:
            page = browser.new_page()
            page.set_default_timeout(0)
            # A long measurement that dies partway loses its return value entirely. The
            # progress lines are streamed out.
            page.on("console", lambda m: print(f"  {m.text}")
                    if m.text.startswith("[accuracy]") or m.type == "error" else None)
            page.on("pageerror", lambda e: print(f"  [browser exception] {e}"))
            page.goto(f"http://127.0.0.1:{port}/borch-ts/test/accuracy.html{query}")
            page.wait_for_function("window.__borchAccuracy !== undefined",
                                   timeout=TIMEOUT_MS)
            result = page.evaluate("window.__borchAccuracy")
    finally:
        stop()

    if "error" in result:
        print(f"could not measure: {result['error']}", file=sys.stderr)
        return 1
    print(result["text"])
    # The device does not change the accuracy itself. And **the epoch time is recorded
    # alongside it**, and on a software adapter half of that report becomes false. Rather
    # than mixing a valid number and an invalid one in one report, it asks for a remeasure.
    if refuse_if_software(result.get("adapter"), "the epoch time"):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
