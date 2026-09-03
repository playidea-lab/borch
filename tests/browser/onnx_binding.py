"""Does `borch_webgpu.onnx.export` write a file ONNX Runtime Web runs, with the binding's logits?

    uv run --with playwright python tests/browser/onnx_binding.py [--headed]

Python (Pyodide) builds `tests/resnet.py`'s network on the binding, exports it to the
virtual filesystem with no `await`, and the page hands the file to ORT Web. The verdict
is read from `checks` (`borch-ts/test/verdict.py`).
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent / "borch-ts" / "test"))

import run as runner                                     # noqa: E402  (tests/browser/run.py)
from launch import browser as browser_of                 # noqa: E402
from verdict import verdict                              # noqa: E402

PAGE = "/tests/browser/onnx_binding.html"
TIMEOUT_MS = 15 * 60 * 1000


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
            page.wait_for_function("window.__borchOnnxBinding !== undefined", timeout=TIMEOUT_MS)
            result = page.evaluate("window.__borchOnnxBinding")
    finally:
        stop()
    if "error" in result:
        print(f"**the binding ONNX check blew up**\n{result['error']}", file=sys.stderr)
        return 1
    print(result["text"])
    return 1 if verdict(result, "onnx:binding") else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
