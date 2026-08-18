"""한 스텝이 무엇을 얼마나 쓰는가 — **시간이 아니라 세는 것**으로.

    npm run build:ts
    uv run --with playwright python borch-ts/test/cost.py [--headed]

## 벤치와 무엇이 다른가

`bench.py` 는 벽시계를 재고, 그래서 소프트웨어 어댑터에서 **답을 거부한다** —
CPU 래스터라이저에서 잰 ms 는 이 라이브러리의 수가 아니다. 그 판단이 맞다.

여기서 세는 것(dispatch 수·제출 수·잡고 있는 버퍼)은 **코드 경로가 정하는 수라
장치가 안 바꾼다.** 그래서 막지 않는다 — **벤치가 못 도는 자리에서 도는 것**이
이 검사의 존재 이유다.

## 골든이 못 보는 것

골든은 값만 본다. 스텝마다 버퍼를 하나씩 흘려도, 커널을 두 배로 걸어도 값은
똑같이 맞으므로 표는 전부 초록이다. `scope()` 가 있는 이유가 그 자리이고,
지금까지 그것을 지키는 검사가 없었다.
"""

import sys

import run as runner
from launch import browser as browser_of

PAGE = "/borch-ts/test/cost.html"
TIMEOUT_MS = 600_000


def main(argv):
    # 낡은 방출물로 초록을 보는 것이 안 도는 것보다 나쁘다.
    runner.require_fresh_dist()
    dist = runner.ROOT / "borch-ts" / "dist" / "test" / "cost.js"
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
            page.wait_for_function("window.__borchCost !== undefined",
                                   timeout=TIMEOUT_MS)
            result = page.evaluate("window.__borchCost")
    finally:
        stop()

    if "error" in result:
        print(f"**비용 점검이 터졌다**\n{result['error']}", file=sys.stderr)
        return 1
    print(result["text"])
    # 세는 것은 페이지 쪽 한 군데다 — 두 곳에서 세면 갈릴 때 어느 쪽이 맞는지 모른다.
    return 0 if "전부 통과" in result["text"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
