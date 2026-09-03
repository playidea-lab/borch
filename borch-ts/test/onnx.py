"""Does the ONNX file borch.ts writes run in ONNX Runtime Web and answer the same?

    uv run --with playwright python borch-ts/test/onnx.py [--headed]

The page exports the bench's ResNet-18, hands the bytes to ORT Web (vendored under
`tests/browser/assets.lock`), and compares logits at two batches, before and after
`fuse()`. The verdict is read from `checks` (`verdict.py`).
"""
import sys

import run as runner
from launch import browser as browser_of
from verdict import verdict

PAGE = "/borch-ts/test/onnx.html"
TIMEOUT_MS = 10 * 60 * 1000


def main(argv):
    runner.require_fresh_dist()
    dist = runner.ROOT / "borch-ts" / "dist" / "test" / "onnx.js"
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
            page.wait_for_function("window.__borchOnnx !== undefined", timeout=TIMEOUT_MS)
            result = page.evaluate("window.__borchOnnx")
    finally:
        stop()
    if "error" in result:
        print(f"**the ONNX check blew up**\n{result['error']}", file=sys.stderr)
        return 1
    print(f"adapter: {result.get('adapter', '(unknown)')}")
    print(result["text"])
    return 1 if verdict(result, "onnx") else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
