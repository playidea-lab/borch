"""borch.ts 학습 스텝 벤치를 브라우저에서 돌린다.

    npm run build:ts
    uv run --with playwright python borch-ts/test/bench.py [--headed]

`tests/browser/run.py --bench` 와 같은 것을 재되 Pyodide 를 안 거친다 —
borch.ts 는 브라우저가 그냥 읽는 JS 다.
"""

import sys

import run as runner
from launch import browser as browser_of, refuse_if_software

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

        with sync_playwright() as p, \
                browser_of(p, headed="--headed" in argv) as browser:
            page = browser.new_page()
            page.set_default_timeout(0)
            page.on("console", lambda m: print(f"  [브라우저] {m.text}")
                    if m.type == "error" else None)
            page.on("pageerror", lambda e: print(f"  [브라우저 예외] {e}"))
            page.goto(f"http://127.0.0.1:{port}{PAGE}")
            page.wait_for_function("window.__borchBench !== undefined",
                                   timeout=TIMEOUT_MS)
            result = page.evaluate("window.__borchBench")
    finally:
        stop()

    if "error" in result:
        print(f"재지 못했다: {result['error']}", file=sys.stderr)
        return 1
    print(result["text"])
    # **여기서는 막는다.** 시간은 장치가 바꾼다 — CPU 에서 잰 ms/step 은 이 라이브러리의
    # 수가 아니라 소프트웨어 래스터라이저의 수다. 이 저장소가 그 수를 한동안 "자매 대비
    # 272배 느림" 으로 적어 두었고, 그것은 라이브러리 비교가 아니라 CPU 대 GPU 비교였다.
    if refuse_if_software(result.get("adapter"), "ms/step 과 에폭 시간"):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
