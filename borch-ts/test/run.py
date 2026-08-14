"""borch.ts 골든 러너를 브라우저에서 돌린다.

    npm run build:ts
    uv run --with playwright python borch-ts/test/run.py

`tests/browser/run.py` 와 같은 방식이다 — 저장소 루트를 임시 포트에 얹고 Playwright
로 페이지를 열어 결과를 읽는다. 따로 쓴 이유는 그쪽이 `runner.html?lib=` 로 파이썬
라이브러리를 Pyodide 에 태우는 전용 러너이고, 이쪽은 Pyodide 가 필요 없기 때문이다.
borch.ts 는 브라우저가 그냥 읽는 JS 다.

**돌지 않은 것과 통과한 것을 안 섞는다.** 페이지가 던지면 종료 코드가 0 이 아니고,
"등록했는데 골든에 없는 이름"이 하나라도 있으면 그것도 실패다 — 오타로 0 건을
돌리고 초록색을 보는 것이 이 프로젝트에서 제일 나쁜 결과다.
"""

import functools
import http.server
import pathlib
import socketserver
import sys
import threading

ROOT = pathlib.Path(__file__).resolve().parents[2]
PAGE = "/borch-ts/test/index.html"
TIMEOUT_MS = 120_000


class _Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a):
        pass


def serve(root):
    """저장소 루트를 임시 포트에 얹고 (포트, 종료함수) 를 돌려준다."""
    handler = functools.partial(_ReportMissing, directory=str(root))
    httpd = socketserver.ThreadingTCPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd.server_address[1], httpd.shutdown


class _ReportMissing(_Quiet):
    """**못 찾은 것을 찍는다.** 설명 안 된 404 를 덮어두면 안 된다 — 이 저장소의
    러너가 한 번 404 HTML 을 파이썬 파일로 받아 엉뚱한 자리에서 터진 적이 있다.
    브라우저가 알아서 찾는 것(favicon)도 여기 걸리므로 정체가 드러난다."""

    def send_error(self, code, message=None, explain=None):
        if code == 404:
            print(f"  [404] {self.path}")
        super().send_error(code, message, explain)


def run(headed=False):
    from playwright.sync_api import sync_playwright

    port, stop = serve(ROOT)
    url = f"http://127.0.0.1:{port}{PAGE}"
    try:
        with sync_playwright() as p:
            # 헤드리스 Chromium 은 기본으로 WebGPU 어댑터를 안 준다 — 요청하면
            # 예외가 아니라 null 이 온다. 이 프로젝트의 다른 러너는 TF.js 를 쓰고
            # TF.js 는 못 얻으면 WebGL 로 조용히 내려가서 이 문제가 안 보였다.
            browser = p.chromium.launch(
                headless=not headed,
                args=["--enable-unsafe-webgpu", "--enable-features=Vulkan"],
            )
            page = browser.new_page()
            page.set_default_timeout(0)
            # 셰이더 컴파일 오류는 콘솔로만 나온다. 삼키면 원인을 못 찾는다.
            page.on("console", lambda m: print(f"  [브라우저] {m.text}")
                    if m.type == "error" else None)
            page.on("pageerror", lambda e: print(f"  [브라우저 예외] {e}"))
            # 설명 안 된 404 는 덮어두면 안 된다 — 이 저장소의 러너가 한 번
            # 404 HTML 을 파이썬 파일로 받아 엉뚱한 자리에서 터진 적이 있다.
            page.on("response", lambda r: print(f"  [404] {r.url}")
                    if r.status == 404 else None)
            page.goto(url)
            page.wait_for_function("window.__borchReport !== undefined",
                                   timeout=TIMEOUT_MS)
            report = page.evaluate("window.__borchReport")
            browser.close()
    finally:
        stop()
    return report


def main(argv):
    dist = ROOT / "borch-ts" / "dist" / "test" / "golden.js"
    if not dist.exists():
        # 낡은 dist 로 도는 것보다 안 도는 편이 낫다.
        print(f"방출물이 없다: {dist}\n  먼저: npm run build:ts", file=sys.stderr)
        return 2

    report = run(headed="--headed" in argv)
    if "error" in report:
        print(f"돌지 못했다: {report['error']}", file=sys.stderr)
        return 1

    # **어느 장치에서 돌았는지 먼저 적는다.** 값은 장치가 안 바꾸지만, 안 적어두면
    # 성능을 재는 쪽이 헤드리스의 소프트웨어 어댑터를 진짜 GPU 로 착각한다 —
    # 이 저장소에서 실제로 그렇게 됐다.
    print(f"어댑터: {report.get('adapter', '(모름)')}")
    gap = report["total"] - report["registered"]
    print(f"골든 {report['total']}건 중 {report['registered']}건을 TS 로 썼다 "
          f"— {gap}건은 아직 안 물었다.")
    for name in report["unknown"]:
        print(f"  ? 이름이 골든에 없다: {name}")
    for f in report["failed"]:
        print(f"  ✘ {f['name']} — {f['why']}")
    print(f"통과 {report['passed']} / 실패 {len(report['failed'])}")

    ok = not report["failed"] and not report["unknown"]
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
