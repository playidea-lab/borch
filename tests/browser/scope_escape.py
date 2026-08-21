"""Can Python carry a tensor out of a scope — measured in a browser.

    npm run build:ts
    uv run --with playwright python tests/browser/scope_escape.py --headed

It looks at values and lifetimes alone, so **it holds on a software adapter
too** — the device does not change the rule for releasing a buffer. Which is why
no window has to be opened.
"""

import sys

import run as runner
from launch import browser as browser_of, warn_if_software

PAGE = "/tests/browser/scope_escape.html"
TIMEOUT_MS = 300_000


def main(argv):
    sys.stdout.reconfigure(line_buffering=True)
    dist = runner.ROOT / "borch-ts" / "dist" / "src" / "index.js"
    if not dist.exists():
        print(f"no build output: {dist}\n  first: npm run build:ts",
              file=sys.stderr)
        return 2

    port, stop = runner.serve(runner.ROOT)
    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p, browser_of(p, headed="--headed" in argv) as browser:
            page = browser.new_page()
            page.set_default_timeout(0)
            page.on("pageerror", lambda e: print(f"  [browser exception] {e}"))
            page.on("console", lambda m: print(f"  {m.text}") if m.type == "error" else None)
            page.goto(f"http://127.0.0.1:{port}{PAGE}")
            page.wait_for_function("window.__scopeEscape !== undefined", timeout=TIMEOUT_MS)
            result = page.evaluate("window.__scopeEscape")
    finally:
        stop()

    if "error" in result:
        print(f"could not measure: {result['error']}", file=sys.stderr)
        return 1
    print(result["text"])
    warn_if_software(result.get("adapter"), "the lifetime rules")
    return 0 if result["text"].startswith("scope escape works") else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
