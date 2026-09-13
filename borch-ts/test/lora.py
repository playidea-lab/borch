"""Does nn.LoRALinear behave? Runs the invariant checks in a real browser.

    uv run --with playwright python borch-ts/test/lora.py [--headed]

The page runs `report()` from `borch-ts/test/lora.ts` and the verdict is read from
`checks` (`verdict.py`). torch.nn has no LoRA, so these are invariants, not a golden.
"""
import sys

import run as runner
from launch import browser as browser_of
from verdict import verdict

PAGE = "/borch-ts/test/lora.html"
TIMEOUT_MS = 5 * 60 * 1000


def main(argv):
    runner.require_fresh_dist()
    dist = runner.ROOT / "borch-ts" / "dist" / "test" / "lora.js"
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
            page.wait_for_function("window.__borchLora !== undefined", timeout=TIMEOUT_MS)
            result = page.evaluate("window.__borchLora")
    finally:
        stop()
    if "error" in result:
        print(f"**the LoRA check blew up**\n{result['error']}", file=sys.stderr)
        return 1
    print(f"adapter: {result.get('adapter', '(unknown)')}")
    print(result["text"])
    return 1 if verdict(result, "lora") else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
