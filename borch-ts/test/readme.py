"""문서에 적힌 예시가 실제로 도는지 본다.

    npm run build:ts
    uv run --with playwright python borch-ts/test/readme.py [--headed]

**문서의 코드는 안 돌리면 썩는다.** 이름이 바뀌어도, 인자 순서가 바뀌어도, `await` 가
하나 빠져도 아무도 안 알려주고 첫 사용자가 거기서 막힌다. 이 저장소는 README 의 설치
안내가 실제로 안 듣던 것을 이미 두 번 잡았다.

값은 안 묻는다 — 골든이 그 일을 한다. 여기서 묻는 것은 하나다: **적어 놓은 그대로
쳤을 때 도는가.** 그래서 소프트웨어 어댑터에서도 막지 않는다. 예시가 도는지는
장치와 무관하다.
"""

import sys

import run as runner
from launch import browser as browser_of
from verdict import verdict

PAGE = "/borch-ts/test/readme.html"
TIMEOUT_MS = 300_000


def main(argv):
    dist = runner.ROOT / "borch-ts" / "dist" / "test" / "readme.js"
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
            page.wait_for_function("window.__borchReadme !== undefined",
                                   timeout=TIMEOUT_MS)
            result = page.evaluate("window.__borchReadme")
    finally:
        stop()

    if "error" in result:
        print(f"**예시가 터졌다** — 문서를 그대로 친 사람이 여기서 막힌다.\n"
              f"{result['error']}", file=sys.stderr)
        return 1
    print(f"어댑터: {result.get('adapter', '(모름)')}")
    print(result["text"])
    # **이 줄이 `"그대로 돌고" in text` 였다.** 그 낱말은 두 예시의 성공 문장 양쪽에
    # 들어 있어서, 첫 예시가 실패하고 LBFGS 만 통과해도 0 이 나갔다 — 손실이 안
    # 내려가는 예시를 문서에 그대로 둔 채로.
    return verdict(result, "README 예시")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
