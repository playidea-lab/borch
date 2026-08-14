"""borch.ts 학습 스텝 벤치를 브라우저에서 돌린다.

    npm run build:ts
    uv run --with playwright python borch-ts/test/bench.py [--headed]

`tests/browser/run.py --bench` 와 같은 것을 재되 Pyodide 를 안 거친다 —
borch.ts 는 브라우저가 그냥 읽는 JS 다.
"""

import sys

import run as runner
from launch import launch

PAGE = "/borch-ts/test/bench.html"
TIMEOUT_MS = 1_800_000


def main(argv):
    dist = runner.ROOT / "borch-ts" / "dist" / "test" / "bench.js"
    if not dist.exists():
        print(f"방출물이 없다: {dist}\n  먼저: npm run build:ts", file=sys.stderr)
        return 2

    port, stop = runner.serve(runner.ROOT)
    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            browser = launch(p, headed="--headed" in argv)
            page = browser.new_page()
            page.set_default_timeout(0)
            page.on("console", lambda m: print(f"  [브라우저] {m.text}")
                    if m.type == "error" else None)
            page.on("pageerror", lambda e: print(f"  [브라우저 예외] {e}"))
            page.goto(f"http://127.0.0.1:{port}{PAGE}")
            page.wait_for_function("window.__borchBench !== undefined",
                                   timeout=TIMEOUT_MS)
            result = page.evaluate("window.__borchBench")
            browser.close()
    finally:
        stop()

    if "error" in result:
        print(f"재지 못했다: {result['error']}", file=sys.stderr)
        return 1
    print(result["text"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
