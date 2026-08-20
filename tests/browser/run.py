"""브라우저에서 골든을 대조한다 — **진짜 GPU 가 달린 기계**에서 돈다.

GitHub 호스팅 러너에는 GPU 가 없어 SwiftShader 로 떨어지고, 그건 아무 사용자도
지나지 않는 경로다. 그래서 이 스크립트는 self-hosted 러너나 개발 기계에서 돈다.

    uv run --with playwright python tests/browser/run.py
    uv run --with playwright python tests/browser/run.py --headed --lib borch_webgpu

먼저 골든이 있어야 한다:

    uv run --with numpy --with torch python tests/golden.py dump

`file://` 로는 안 된다 — 러너가 소스와 골든을 `fetch` 로 가져오므로 서버가 필요하다.
여기서 저장소 루트를 임시 포트에 얹는다.
"""

import argparse
import functools
import http.server
import importlib.util
import pathlib
import socketserver
import sys
import threading

from launch import browser as browser_of

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent

_vspec = importlib.util.spec_from_file_location(
    "bt_vendor", pathlib.Path(__file__).resolve().parent / "vendor.py")
vendor = importlib.util.module_from_spec(_vspec)
_vspec.loader.exec_module(vendor)
GOLDEN = ROOT / "tests" / "golden.npz"
# Pyodide + numpy 내려받기가 첫 실행에서 느리고, 케이스가 늘수록 본 계산도 는다.
# **시간이 모자란 것과 안 끝나는 것은 다른 일이다** — 짧게 잡아 두면 느린 것을
# 멈춘 것으로 읽고 없는 결함을 쫓게 된다. 자매 러너가 같은 이유로 600 초다.
TIMEOUT_MS = 600_000


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

    # **방출물이 소스보다 낡았는지 먼저 본다.** 낡은 `dist` 는 진짜 결손과 같은 문구로
    # 나온다 — 검사를 두 벌 두면 한쪽만 고쳐지므로 TS 러너의 것을 빌려 쓴다.
    sys.path.insert(0, str(ROOT / "borch-ts" / "test"))
    from run import require_fresh_dist, require_fresh_golden   # noqa: PLC0415
    require_fresh_dist(ROOT)
    # 골든도 같은 함정이다 — `cases.py` 만 고치고 안 뽑으면 새 케이스가 "이름이
    # 골든에 없다" 로 나오는데, 오타와 같은 문구다.
    require_fresh_golden(ROOT)

    port, stop = serve(ROOT)
    url = f"http://127.0.0.1:{port}/tests/browser/runner.html?lib={lib}"
    probed = None
    try:
        # **브라우저를 닫는 것도 `with` 가 한다** — 마지막 줄에 두면 그 앞에서
        # 예외가 날 때 안 닫히고, 남은 크로미엄이 다른 측정을 망가뜨린다.
        with sync_playwright() as p, browser_of(p, headed=headed) as browser:
            # **이 러너에는 여태 플래그가 없었다.** macOS 에서는 그래도 WebGPU 가
            # 나와서 안 보였는데, 리눅스 GPU 서버에서는 Vulkan 을 켜 줘야 한다.
            # borch.ts 쪽 러너와 다른 조건으로 재면 그때부터 같은 잣대가 아니다.
            page = browser.new_page()
            # 정확도 측정은 몇 분씩 걸린다. 기본 제한에 걸려 죽으면 잰 것이 없어진다.
            page.set_default_timeout(0)
            # 오류는 늘 내보내고, `[bench]` 로 시작하는 것도 내보낸다 — 긴 측정은
            # 도중에 죽으면 반환값이 통째로 사라져서, 그때까지 잰 것을 잃지 않으려면
            # 진행 중에 흘려보낸 것만 남는다.
            page.on("console", lambda m: print(f"  [브라우저] {m.text}")
                    if m.type == "error" or m.text.startswith("[bench]") else None)
            page.goto(url)
            page.wait_for_function("window.GOLDEN_RESULT !== null", timeout=TIMEOUT_MS)
            result = page.evaluate("window.GOLDEN_RESULT")
            if probe:
                # 브라우저 안에서만 재현되는 것을 눈으로 보는 통로다.
                probed = page.evaluate(
                    "async (code) => String(await window.PY.runPythonAsync(code))", probe)
    finally:
        stop()
    return result, probed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", default="borch", help="대조할 라이브러리")
    ap.add_argument("--headed", action="store_true",
                    help="창을 띄운다. **WebGPU 는 헤드리스에서 안 뜬다**(실측 — WebGL 로 떨어진다)")
    ap.add_argument("--probe", help="대조 뒤 브라우저 안에서 돌릴 파이썬. 디버깅용")
    ap.add_argument("--bench", action="store_true",
                    help="ResNet-18 로 실제 학습 스텝을 잰다(tests/browser/bench.py)")
    ap.add_argument("--cost", action="store_true",
                    help="한 스텝의 **비용**을 센다 — 시간이 아니라 dispatch·버퍼 수라 "
                         "소프트웨어 어댑터에서도 뜻이 있다(tests/browser/cost.py)")
    ap.add_argument("--accuracy", action="store_true",
                    help="**정확도**를 잰다 — 늘리기를 켠 쪽과 끈 쪽을 나란히. "
                         "cifar-batch1.bin 과 cifar-batch-test.bin 이 저장소 루트에 있어야 한다")
    ap.add_argument("--epochs", type=int, default=6, help="--accuracy 의 에폭 수")
    ap.add_argument("--images", type=int, default=0,
                    help="--accuracy 에서 쓸 장수 상한(0 이면 전부). 기계를 확인할 때 쓴다")
    ap.add_argument("--augment", choices=("on", "off"),
                    help="한 조건만 돌린다. **갈라 돌리는 편이 실험으로 낫다** — 한 세션에 "
                         "이어 돌리면 둘째 모델의 초기 가중치가 달라져 늘리기의 효과에 섞인다")
    args = ap.parse_args()
    if args.bench and not args.probe:
        args.probe = (f"import bench, importlib\n"
                      f"L = importlib.import_module({args.lib!r})\n"
                      f"bench.report(L)")
    if args.cost and not args.probe:
        args.probe = (f"import cost, importlib\n"
                      f"L = importlib.import_module({args.lib!r})\n"
                      f"cost.report(L)")
    if args.accuracy and not args.probe:
        # 시험 데이터는 **학습에 안 쓴 것**이어야 한다. 그래서 원본 아카이브의
        # test_batch 를 따로 꺼내 둔다 — 같은 덩이를 나눠 쓰면 재는 것이 정확도가 아니다.
        args.probe = (
            f"import bench, importlib\n"
            f"L = importlib.import_module({args.lib!r})\n"
            # 늘리기의 뽑기도 씨앗을 박는다. 안 그러면 같은 명령이 매번 다른 답을 내고,
            # 두 조건의 차이인지 뽑기의 차이인지 못 가른다.
            f"import borchvision as V; V.use(L); V.manual_seed(0)\n"
            f"tr = await bench.cifar_from(L, '/cifar-batch1.bin', 'cifar-batch1.bin')\n"
            f"te = await bench.cifar_from(L, '/cifar-batch-test.bin', 'cifar-test.bin')\n"
            f"cap = {args.images}\n"
            f"tr = (tr[0][:cap], tr[1][:cap]) if cap else tr\n"
            f"te = (te[0][:cap], te[1][:cap]) if cap else te\n"
            f"only = {None if args.augment is None else args.augment == 'on'!r}\n"
            f"await bench.report_accuracy(L, tr, te, epochs={args.epochs}, only=only)")

    if not GOLDEN.exists():
        print(f"골든이 없다: {GOLDEN}\n"
              "  먼저: uv run --with numpy --with torch python tests/golden.py dump")
        return 1

    # Pyodide 는 로컬에서 온다. 없으면 한 번 받고, 있으면 해시로 대조한다.
    vendor.ensure()

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
    # **검증 오류는 예외로 안 온다.** 하나라도 났으면 위의 초록이 그만큼 못 미덥다 —
    # 무효한 명령 버퍼는 조용히 아무것도 안 하고 벽시계만 돈다.
    if result.get("faults"):
        print(f"  GPU 검증 오류 {result['faults']}건")
    if bad:
        print("\n갈린 곳:")
        for why in bad:
            print(f"  ✗ {why}")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
