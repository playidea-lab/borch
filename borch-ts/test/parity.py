"""Whether the wiring is right once torch has been copied across.

    npm run build:ts
    uv run --with playwright python borch-ts/test/parity.py [--headed]

**The golden cannot catch these three.** It plants the weights from outside per case,
so it never asks how the parameters are collected and never looks at the initial values.
What is asked here is the wiring rather than the values:

- whether a layer built outside has its parameters collected by `parameters()` (wrong,
  and **with no exception** training alone quietly stops — the kind that is never visible
  without a runner)
- whether parameter groups actually move with a different learning rate and weight decay
  per group
- whether one seed resets the tensor factories and the layer initialisation together

It does not refuse on a software adapter — the wiring is the same on any.
"""

import sys

import run as runner
from launch import browser as browser_of
from verdict import verdict

PAGE = "/borch-ts/test/parity.html"
TIMEOUT_MS = 300_000


def main(argv):
    # **A stale emit is as bad as none** — edit the source, forget the build, and you
    # measure the old code. `require_fresh_dist` watches that place (`run.py`).
    runner.require_fresh_dist()
    dist = runner.ROOT / "borch-ts" / "dist" / "test" / "parity.js"
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
            page.wait_for_function("window.__borchParity !== undefined",
                                   timeout=TIMEOUT_MS)
            result = page.evaluate("window.__borchParity")
    finally:
        stop()

    if "error" in result:
        print(f"**the wiring check blew up**\n{result['error']}", file=sys.stderr)
        return 1
    print(f"adapter: {result.get('adapter', '(unknown)')}")
    print(result["text"])
    # The page counts the failures. Counted in two places, there is no way to know which
    # is right when they disagree.
    return verdict(result, "torch wiring")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
