"""데이터셋과 적재기가 실제 브라우저에서 도는지 본다.

    npm run build:ts
    uv run --with playwright python borch-ts/test/data.py [--headed]

**빠른 길과 느린 길이 같은 답을 내야 한다.** `TensorDataset` 은 배치를 한 번에
뽑고(`narrow`·`indexSelect`), `gather` 가 없는 데이터셋은 하나씩 꺼내 쌓는다. 둘이
갈리면 빠른 쪽만 틀리고 값이 그럴듯해서 아무도 못 본다 — 그래서 두 길을 나란히
돌려 견준다.

같이 묻는 것: 배치 수 세기(`dropLast` 포함), 섞기가 `manualSeed` 를 따르는가,
에폭마다 다시 섞는가, `randomSplit` 이 겹치지도 빠뜨리지도 않는가, 그리고 적재기로
돌린 학습에서 손실이 실제로 내려가는가.

소프트웨어 어댑터에서도 막지 않는다 — 배치를 뽑는 규칙은 어느 어댑터에서나 같다.
"""

import sys

import run as runner
from launch import browser as browser_of
from verdict import verdict

PAGE = "/borch-ts/test/data.html"
TIMEOUT_MS = 300_000


def main(argv):
    # **낡은 방출물도 없는 것만큼 나쁘다** — 소스를 고치고 빌드를 잊으면 옛 코드를
    # 재게 된다. `require_fresh_dist` 가 그 자리를 본다(`run.py`).
    runner.require_fresh_dist()
    dist = runner.ROOT / "borch-ts" / "dist" / "test" / "data.js"
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
            page.wait_for_function("window.__borchData !== undefined",
                                   timeout=TIMEOUT_MS)
            result = page.evaluate("window.__borchData")
    finally:
        stop()

    if "error" in result:
        print(f"**데이터 적재 점검이 터졌다**\n{result['error']}", file=sys.stderr)
        return 1
    print(f"어댑터: {result.get('adapter', '(모름)')}")
    print(result["text"])
    # 실패 건수를 세는 것은 페이지 쪽이다. 두 곳에서 세면 갈릴 때 어느 쪽이 맞는지
    # 알 방법이 없다.
    return verdict(result, "데이터 적재")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
