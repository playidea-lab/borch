"""What one step uses and how much of it — **by counting rather than by timing.**

    npm run build:ts
    uv run --with playwright python borch-ts/test/cost.py [--headed]

## How it differs from the bench

`bench.py` measures a wall clock and therefore **refuses to answer on a software
adapter** — ms measured on a CPU rasteriser is not this library's number. That judgement
is right.

What is counted here (dispatches, submissions, buffers held) is **a number the code path
decides and the device does not change.** So it does not refuse — **running where the
bench cannot** is this check's reason to exist.

## What the golden cannot see

The golden looks only at values. Leak one buffer per step, or issue twice the kernels, and
the values are equally right, so the whole table is green. That is the place `scope()`
exists for, and until now nothing guarded it.
"""

import sys

import run as runner
from launch import browser as browser_of
from verdict import verdict

PAGE = "/borch-ts/test/cost.html"
TIMEOUT_MS = 600_000


def main(argv):
    # Seeing green from a stale emit is worse than not running at all.
    runner.require_fresh_dist()
    dist = runner.ROOT / "borch-ts" / "dist" / "test" / "cost.js"
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
            page.wait_for_function("window.__borchCost !== undefined",
                                   timeout=TIMEOUT_MS)
            result = page.evaluate("window.__borchCost")
    finally:
        stop()

    if "error" in result:
        print(f"**the cost check blew up**\n{result['error']}", file=sys.stderr)
        return 1
    print(result["text"])
    # The verdict comes from the checks' state. Here and the page were scanning one
    # document for different words — `verdict.py` records all three ways that goes wrong.
    return verdict(result, "the cost checks")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
