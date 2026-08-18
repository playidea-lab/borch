"""torch 를 옮겨 적었을 때 배선이 맞는지 본다.

    npm run build:ts
    uv run --with playwright python borch-ts/test/parity.py [--headed]

**골든이 이 셋을 못 잡는다.** 골든은 케이스마다 가중치를 밖에서 넣어 주므로
파라미터가 어떻게 모이는지를 안 묻고, 초기값도 안 본다. 여기서 묻는 것은 값이
아니라 배선이다:

- 밖에서 만든 층의 파라미터가 `parameters()` 에 모이는가 (틀리면 **예외 없이**
  학습만 조용히 안 된다 — 러너가 없으면 영영 안 보이는 종류다)
- 파라미터 그룹이 그룹마다 다른 학습률·weight decay 로 실제로 움직이는가
- 씨앗 하나가 텐서 팩토리와 층 초기화를 같이 되돌리는가

소프트웨어 어댑터에서도 막지 않는다 — 배선은 어느 어댑터에서나 같다.
"""

import sys

import run as runner
from launch import browser as browser_of

PAGE = "/borch-ts/test/parity.html"
TIMEOUT_MS = 300_000


def main(argv):
    # **낡은 방출물도 없는 것만큼 나쁘다** — 소스를 고치고 빌드를 잊으면 옛 코드를
    # 재게 된다. `require_fresh_dist` 가 그 자리를 본다(`run.py`).
    runner.require_fresh_dist()
    dist = runner.ROOT / "borch-ts" / "dist" / "test" / "parity.js"
    if not dist.exists():
        print(f"방출물이 없다: {dist}\n  먼저: npm run build:ts", file=sys.stderr)
        return 2

    port, stop = runner.serve(runner.ROOT)
    try:
        from playwright.sync_api import sync_playwright

        # **닫는 것도 `with` 가 한다** — 마지막 줄에 두면 그 앞에서 예외가 날 때
        # 안 닫히고, 남은 크로미엄이 다른 측정을 망가뜨린다.
        with sync_playwright() as p, \
                browser_of(p, headed="--headed" in argv) as browser:
            page = browser.new_page()
            page.set_default_timeout(0)
            page.on("console", lambda m: print(f"  [브라우저] {m.text}")
                    if m.type == "error" else None)
            page.on("pageerror", lambda e: print(f"  [브라우저 예외] {e}"))
            page.goto(f"http://127.0.0.1:{port}{PAGE}")
            page.wait_for_function("window.__borchParity !== undefined",
                                   timeout=TIMEOUT_MS)
            result = page.evaluate("window.__borchParity")
    finally:
        stop()

    if "error" in result:
        print(f"**배선 점검이 터졌다**\n{result['error']}", file=sys.stderr)
        return 1
    print(f"어댑터: {result.get('adapter', '(모름)')}")
    print(result["text"])
    # 실패 건수를 세는 것은 페이지 쪽이다. 두 곳에서 세면 갈릴 때 어느 쪽이 맞는지
    # 알 방법이 없다.
    return 0 if "전부 통과" in result["text"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
