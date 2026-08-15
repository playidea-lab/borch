"""Pyodide 의 파이썬에서 borch.ts 를 **동기로** 부를 수 있는가.

    npm run build:ts
    uv run --with playwright python tests/browser/sync_probe.py --headed

## 왜 이걸 재는가

**파이썬 쪽 GPU 구현을 TF.js 대신 borch.ts 위로 옮길 수 있는지가 여기서 갈렸다.**
답이 "된다" 로 나왔고, 그래서 지금 `borch_webgpu` 가 borch.ts 위에 서 있다. 이
파일은 그 결정을 낸 계측이고 여전히 돈다 — 그 조건이 깨지면 여기가 먼저 빨개진다.

파이썬 API 는 통째로 동기여야 한다. 예전 TF.js 판은 `dataSync()` 덕에 공짜로 그랬다
(WebGPU 에 동기 읽기 API 가 없는데도 그쪽이 받아줬다). borch.ts 는
`await stage.mapAsync(...)` 라 비동기다.

그대로 얹으면 `await loss.item()` 이 되고, 그러면 **"튜토리얼 코드를 임포트만 바꿔
돌린다"** 는 이 프로젝트의 유일한 주장이 깨진다. 그 주장이 이 라이브러리의 존재
이유이므로, 그걸 깨는 길은 갈 수 없다.

Pyodide 에 `run_sync` 가 있고 그 밑에 JSPI(WebAssembly 의 Promise 통합)가 있다.
**있다는 것과 도는 것은 다르다** — 이 저장소가 반복해서 배운 것이 그것이므로 잰다.
"""

import sys

import run as runner
from launch import launch

PAGE = "/tests/browser/sync_probe.html"
TIMEOUT_MS = 300_000


def main(argv):
    dist = runner.ROOT / "borch-ts" / "dist" / "src" / "index.js"
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
            page.wait_for_function("window.__syncProbe !== undefined",
                                   timeout=TIMEOUT_MS)
            got = page.evaluate("window.__syncProbe")
            browser.close()
    finally:
        stop()

    if "error" in got:
        print(f"재지 못했다: {got['error']}", file=sys.stderr)
        return 1

    print(got["text"])
    print()
    if got["sync"]:
        print("**된다** — `borch_webgpu` 가 서 있는 바닥이 그대로다.")
        return 0
    if not got["async"]:
        print("비동기조차 안 됐다 — 동기 여부와 무관한 다른 문제다. 위 줄을 보라.")
        return 1
    print("**안 된다** — 파이썬 API 를 동기로 유지할 수 없다.\n"
          "  `borch_webgpu` 가 이 위에 서 있으므로 이것은 그 패키지가 지금\n"
          "  깨졌다는 뜻이다. 여기서 `await loss.item()` 을 받아들이면\n"
          "  '임포트만 바꿔 돌린다' 가 통째로 없어진다.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
