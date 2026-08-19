"""파이썬이 구역 밖으로 텐서를 들고 나올 수 있는가 — 브라우저에서 잰다.

    npm run build:ts
    uv run --with playwright python tests/browser/scope_escape.py --headed

값과 수명만 보므로 **소프트웨어 어댑터에서도 유효하다** — 버퍼를 놓는 규칙은
장치가 안 바꾼다. 그래서 창을 안 띄워도 된다.
"""

import sys

import run as runner
from launch import browser as browser_of, warn_if_software

PAGE = "/tests/browser/scope_escape.html"
TIMEOUT_MS = 300_000


def main(argv):
    sys.stdout.reconfigure(line_buffering=True)
    dist = runner.ROOT / "borch-ts" / "dist" / "src" / "index.js"
    if not dist.exists():
        print(f"방출물이 없다: {dist}\n  먼저: npm run build:ts", file=sys.stderr)
        return 2

    port, stop = runner.serve(runner.ROOT)
    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p, browser_of(p, headed="--headed" in argv) as browser:
            page = browser.new_page()
            page.set_default_timeout(0)
            page.on("pageerror", lambda e: print(f"  [브라우저 예외] {e}"))
            page.on("console", lambda m: print(f"  {m.text}") if m.type == "error" else None)
            page.goto(f"http://127.0.0.1:{port}{PAGE}")
            page.wait_for_function("window.__scopeEscape !== undefined", timeout=TIMEOUT_MS)
            result = page.evaluate("window.__scopeEscape")
    finally:
        stop()

    if "error" in result:
        print(f"재지 못했다: {result['error']}", file=sys.stderr)
        return 1
    print(result["text"])
    warn_if_software(result.get("adapter"), "수명 규칙")
    return 0 if result["text"].startswith("구역 탈출이 돈다") else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
