"""브라우저에서 골든을 대조한다 — **진짜 GPU 가 달린 기계**에서 돈다.

GitHub 호스팅 러너에는 GPU 가 없어 SwiftShader 로 떨어지고, 그건 아무 사용자도
지나지 않는 경로다. 그래서 이 스크립트는 self-hosted 러너나 개발 기계에서 돈다.

    uv run --with playwright python tests/browser/run.py
    uv run --with playwright python tests/browser/run.py --headed --lib browsertorch_webgpu

먼저 골든이 있어야 한다:

    uv run --with numpy --with torch python tests/golden.py dump

`file://` 로는 안 된다 — 러너가 소스와 골든을 `fetch` 로 가져오므로 서버가 필요하다.
여기서 저장소 루트를 임시 포트에 얹는다.
"""

import argparse
import functools
import http.server
import pathlib
import socketserver
import sys
import threading

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
GOLDEN = ROOT / "tests" / "golden.npz"
TIMEOUT_MS = 180_000          # Pyodide + numpy 내려받기가 첫 실행에서 느리다


class _Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a):
        pass


def serve(root):
    """저장소 루트를 임시 포트에 얹고 (포트, 종료함수) 를 돌려준다."""
    handler = functools.partial(_Quiet, directory=str(root))
    httpd = socketserver.ThreadingTCPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd.server_address[1], httpd.shutdown


def run(lib, headed, probe=None):
    from playwright.sync_api import sync_playwright

    port, stop = serve(ROOT)
    url = f"http://127.0.0.1:{port}/tests/browser/runner.html?lib={lib}"
    probed = None
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=not headed)
            page = browser.new_page()
            page.on("console", lambda m: print(f"  [브라우저] {m.text}") if m.type == "error" else None)
            page.goto(url)
            page.wait_for_function("window.GOLDEN_RESULT !== null", timeout=TIMEOUT_MS)
            result = page.evaluate("window.GOLDEN_RESULT")
            if probe:
                # 브라우저 안에서만 재현되는 것을 눈으로 보는 통로다.
                probed = page.evaluate(
                    "async (code) => String(await window.PY.runPythonAsync(code))", probe)
            browser.close()
    finally:
        stop()
    return result, probed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", default="browsertorch", help="대조할 라이브러리")
    ap.add_argument("--headed", action="store_true",
                    help="창을 띄운다. **WebGPU 는 헤드리스에서 안 뜬다**(실측 — WebGL 로 떨어진다)")
    ap.add_argument("--probe", help="대조 뒤 브라우저 안에서 돌릴 파이썬. 디버깅용")
    ap.add_argument("--bench", action="store_true",
                    help="ResNet-18 로 실제 학습 스텝을 잰다(tests/browser/bench.py)")
    args = ap.parse_args()
    if args.bench and not args.probe:
        args.probe = (f"import bench, importlib\n"
                      f"L = importlib.import_module({args.lib!r})\n"
                      f"bench.report(L)")

    if not GOLDEN.exists():
        print(f"골든이 없다: {GOLDEN}\n"
              "  먼저: uv run --with numpy --with torch python tests/golden.py dump")
        return 1

    result, probed = run(args.lib, args.headed, args.probe)
    if probed is not None:
        print("-- probe --")
        print(probed)
        print()
    total, bad = result["total"], result["bad"]
    if result.get("error"):
        print("러너가 터졌다:\n" + result["error"])
        return 1
    print(f"브라우저 골든 대조 ({result.get('lib', args.lib)}) — 케이스 {total}개")
    print(f"  일치 {total - len(bad)}/{total}")
    if bad:
        print("\n갈린 곳:")
        for why in bad:
            print(f"  ✗ {why}")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
