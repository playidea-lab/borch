"""Whether the datasets and loaders run in a real browser.

    npm run build:ts
    uv run --with playwright python borch-ts/test/data.py [--headed]

**The fast path and the slow path have to give the same answer.** `TensorDataset` takes
a batch in one go (`narrow`, `indexSelect`), and a dataset without `gather` pulls them one
at a time and stacks. When the two diverge only the fast one is wrong, the values are
plausible, and nobody sees it — so both are run side by side and compared.

Asked alongside: the batch count (`dropLast` included), whether the shuffle follows
`manualSeed`, whether it reshuffles per epoch, whether `randomSplit` neither overlaps nor
omits, and whether training through the loader actually lowers the loss.

It does not refuse on a software adapter — the batching rules are the same on any.
"""

import sys

import run as runner
from launch import browser as browser_of
from verdict import verdict

PAGE = "/borch-ts/test/data.html"
TIMEOUT_MS = 300_000


def main(argv):
    # **A stale emit is as bad as none** — edit the source, forget the build, and you
    # measure the old code. `require_fresh_dist` watches that place (`run.py`).
    runner.require_fresh_dist()
    dist = runner.ROOT / "borch-ts" / "dist" / "test" / "data.js"
    if not dist.exists():
        print(f"no emit: {dist}\n  first: npm run build:ts", file=sys.stderr)
        return 2

    port, stop = runner.serve(runner.ROOT)
    try:
        from playwright.sync_api import sync_playwright

        # **`with` closes it too** — put on the last line instead, an exception before
        # it leaves it open, and the leftover Chromium ruins another measurement.
        with sync_playwright() as p, \
                browser_of(p, headed="--headed" in argv) as browser:
            page = browser.new_page()
            page.set_default_timeout(0)
            page.on("console", lambda m: print(f"  [browser] {m.text}")
                    if m.type == "error" else None)
            page.on("pageerror", lambda e: print(f"  [browser exception] {e}"))
            page.goto(f"http://127.0.0.1:{port}{PAGE}")
            page.wait_for_function("window.__borchData !== undefined",
                                   timeout=TIMEOUT_MS)
            result = page.evaluate("window.__borchData")
    finally:
        stop()

    if "error" in result:
        print(f"**the data-loading check blew up**\n{result['error']}", file=sys.stderr)
        return 1
    print(f"adapter: {result.get('adapter', '(unknown)')}")
    print(result["text"])
    # The page counts the failures. Counted in two places, there is no way to know which
    # is right when they disagree.
    return verdict(result, "data loading")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
