"""장치 관리가 실제 브라우저에서 도는지 본다.

    npm run build:ts
    uv run --with playwright python borch-ts/test/device.py [--headed]

**골든이 이것을 안 잡는다.** 골든은 값이 torch 와 같은지를 묻고, 여기서 묻는 것은
값이 아니라 *어디에 있는가* 와 *없을 때 뭐라고 하는가* 다 — `t.device`, `cpu()`·
`webgpu()` 왕복, 갈린 장치를 섞었을 때 나오는 문구, `synchronize()`.

이 물음들은 노드에서 흉내 낼 수 없다. `navigator.gpu` 가 있어야 하고, 어댑터가
있어야 하고, 진짜 버퍼가 오가야 한다.

소프트웨어 어댑터에서도 막지 않는다 — 배치 규칙은 어느 어댑터에서나 같다.
"""

import sys

import run as runner
from launch import browser as browser_of

PAGE = "/borch-ts/test/device.html"
TIMEOUT_MS = 300_000


def main(argv):
    # **낡은 방출물도 없는 것만큼 나쁘다** — 소스를 고치고 빌드를 잊으면 옛 코드를
    # 재게 된다. `require_fresh_dist` 가 그 자리를 본다(`run.py`).
    runner.require_fresh_dist()
    dist = runner.ROOT / "borch-ts" / "dist" / "test" / "device.js"
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
            page.wait_for_function("window.__borchDevice !== undefined",
                                   timeout=TIMEOUT_MS)
            result = page.evaluate("window.__borchDevice")
    finally:
        stop()

    if "error" in result:
        print(f"**장치 점검이 터졌다**\n{result['error']}", file=sys.stderr)
        return 1
    print(f"어댑터: {result.get('adapter', '(모름)')}")
    # **선택 기능도 같이 적는다.** `timestamp-query` 가 없으면 커널별 시간을 재는
    # 길이 아예 막힌다 — 그때 벽시계만 들고 원인을 찾게 된다.
    print(f"기능:   {result.get('features') or '(없음)'}")
    print(result["text"])
    # 실패 건수를 세는 것은 페이지 쪽이다. 여기서는 그 판정을 그대로 받는다 —
    # 두 곳에서 세면 두 셈이 갈릴 때 어느 쪽이 맞는지 알 방법이 없다.
    return 0 if "전부 통과" in result["text"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
