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

from launch import browser as browser_of, warn_if_software

ROOT = pathlib.Path(__file__).resolve().parents[2]
PAGE = "/borch-ts/test/index.html"
# **넉넉히 준다.** 전에는 120 초였는데, 표가 커지고 헤드리스가 소프트웨어 어댑터로
# 내려가면 그 안에 못 끝난다. 그때 나오는 화면은 "멈췄다" 와 구별이 안 돼서 실제로
# 한 번 없는 결함을 쫓았다 — 마지막 케이스 이름까지 찍어 놓고도 그랬다.
# 시간이 모자란 것과 안 끝나는 것은 다른 일이므로, 예산은 의심의 여지가 없게 둔다.
TIMEOUT_MS = 600_000


def require_fresh_dist(root=ROOT):
    """**소스가 `dist` 보다 새것이면 여기서 멈춘다.**

    러너가 싣는 것은 `borch-ts/dist` 이고 그것은 `.gitignore` 라 어느 커밋에도 없다.
    그래서 리베이스하거나 브랜치를 옮기면 소스만 바뀌고 방출물은 옛것으로 남는다.

    **그 상태를 러너가 못 알린다.** 낡은 `dist` 와 진짜 결손이 `borch.ts 에 X 가 없다`
    라는 **같은 문구**로 나오기 때문이다. 실제로 두 사람이 각각 밟았다 — 한쪽은 새
    케이스 119 건을 넣고 러너 수가 한 건도 안 움직여 이름 오타를 찾았고, 다른 쪽은
    결속에서 31 건이 빨개진 것을 회귀로 보고했다. 그 31 건은 `tensor.ts` 에 **있는**
    이름들이었고, 소스를 읽으면서 러너는 `dist` 를 읽고 있었다.

    `check:ts` 는 `--noEmit` 이라 이 자리를 안 고친다. 빌드를 잊는 것이 자연스러운
    구조이므로, 잊었을 때 **다른 것을 의심하기 전에** 멈추는 편이 낫다.
    """
    dist = root / "borch-ts" / "dist"
    if not dist.exists():
        raise SystemExit(f"방출물이 없다: {dist}\n  먼저: npm run build:ts")
    newest_src = max(
        (p.stat().st_mtime for p in (root / "borch-ts").rglob("*.ts")
         if "dist" not in p.parts and "node_modules" not in p.parts),
        default=0)
    oldest_out = min((p.stat().st_mtime for p in dist.rglob("*.js")), default=0)
    if newest_src > oldest_out:
        raise SystemExit(
            "방출물이 소스보다 낡았다 — 러너는 `borch-ts/dist` 를 싣는다.\n"
            "  먼저: npm run build:ts\n"
            "  (이대로 돌리면 새 이름이 `borch.ts 에 없다` 로 나오는데, 그것은\n"
            "   진짜 결손일 때와 **같은 문구**라 원인이 안 보인다.)")


def require_fresh_golden(root=ROOT):
    """**`cases.py` 가 `golden.json` 보다 새것이면 여기서 멈춘다.**

    `dist` 와 같은 함정이 골든에도 있다. 러너가 읽는 것은 `tests/golden.json` 이고,
    그것은 `tests/cases.py` → `golden.npz` → `golden.json` 의 두 단계를 거쳐 나온다.
    가운데 `npz` 는 `.gitignore` 라 어느 커밋에도 없다.

    그래서 케이스를 새로 쓰고 뽑기를 잊으면 **그 케이스가 "이름이 골든에 없다" 로
    나온다** — 이름을 오타 낸 것과 **같은 문구**다. 실제로 아홉 건을 넣고 그 화면을
    받아, 없는 오타를 먼저 찾았다.

    두 단계라 잊기 쉽고, 첫 단계는 진짜 torch 를 요구해서 아무 데서나 못 돈다.
    잊었을 때 **다른 것을 의심하기 전에** 멈추는 편이 낫다.
    """
    cases = root / "tests" / "cases.py"
    exported = root / "tests" / "golden.json"
    if not exported.exists():
        raise SystemExit(f"골든이 없다: {exported}")
    if cases.stat().st_mtime > exported.stat().st_mtime:
        raise SystemExit(
            "골든이 케이스 표보다 낡았다 — 러너는 `tests/golden.json` 을 읽는다.\n"
            "  먼저: uv run --with numpy --with torch --with torchvision "
            "python tests/golden.py dump\n"
            "  그다음: uv run --with numpy python tests/export_json.py\n"
            "  (이대로 돌리면 새 케이스가 `이름이 골든에 없다` 로 나오는데, 그것은\n"
            "   이름을 틀렸을 때와 **같은 문구**라 원인이 안 보인다.)")


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


_STARTED = "[골든] "


def run(headed=False, verbose=False):
    """`verbose` 면 콘솔을 전부 찍는다.

    **조용한 멈춤을 붙잡는다.** 러너가 케이스마다 try/catch 를 하므로 예외는 보고서에
    실려 나온다. 안 실리는 것은 **영영 안 끝나는** 케이스이고, 그때는 보고서 자체가
    안 만들어져 화면에 아무것도 안 남은 채 시간만 간다.

    그래서 러너가 케이스를 시작할 때마다 찍는 줄을 여기서 붙잡아 두었다가, 시간이
    다 되면 **마지막으로 시작한 이름**을 말한다 — 그게 범인이다. 터미널 스크롤백에
    기대면 1,199 줄에 묻히고, 실제로 한 번 묻혔다.
    """
    from playwright.sync_api import sync_playwright

    require_fresh_dist()
    require_fresh_golden()
    port, stop = serve(ROOT)
    url = f"http://127.0.0.1:{port}{PAGE}"
    last = []

    def on_console(m):
        if m.text.startswith(_STARTED):
            last.append(m.text[len(_STARTED):])
        if verbose or m.type == "error":
            print(f"  [브라우저] {m.text}")

    try:
        # **브라우저를 닫는 것도 `with` 가 한다** — 마지막 줄에 두면 그 앞에서
        # 예외가 날 때 안 닫히고, 남은 크로미엄이 다른 측정을 망가뜨린다.
        with sync_playwright() as p, browser_of(p, headed=headed) as browser:
            # 헤드리스 Chromium 은 기본으로 WebGPU 어댑터를 안 준다 — 요청하면
            # 예외가 아니라 null 이 온다. 예전 TF.js 판에서는 이 문제가 안 보였다.
            # 못 얻으면 WebGL 로 조용히 내려갔기 때문이다 — 안 보이는 것이 나은
            # 것이 아니라, 그때 잰 수가 GPU 의 것이 아니었다.
            page = browser.new_page()
            page.set_default_timeout(0)
            # 셰이더 컴파일 오류는 콘솔로만 나온다. 삼키면 원인을 못 찾는다.
            page.on("console", on_console)
            page.on("pageerror", lambda e: print(f"  [브라우저 예외] {e}"))
            # 설명 안 된 404 는 덮어두면 안 된다 — 이 저장소의 러너가 한 번
            # 404 HTML 을 파이썬 파일로 받아 엉뚱한 자리에서 터진 적이 있다.
            page.on("response", lambda r: print(f"  [404] {r.url}")
                    if r.status == 404 else None)
            page.goto(url)
            try:
                page.wait_for_function("window.__borchReport !== undefined",
                                       timeout=TIMEOUT_MS)
            except Exception:
                where = last[-1] if last else "(한 건도 시작 못 했다)"
                print(f"보고서가 안 나왔다. 마지막으로 시작한 케이스: {where}\n"
                      f"  {len(last)}건을 시작했다 — 그 케이스가 안 끝난 것이다.",
                      file=sys.stderr)
                raise
            report = page.evaluate("window.__borchReport")
    finally:
        stop()
    return report


# 안 옮긴 까닭. **접두어마다 이유와 개수가 있어야 한다.**
#
# 오래 "N 건은 일부러 안 옮겼다" 한 줄이었다. 그 문장 안에 세 가지가 섞여 있었다 —
# 정말 옮길 값이 없는 것, 아직 안 옮긴 것, 그리고 **borch.ts 에 아예 없는 것.**
# 개수만 찍으니 셋이 화면에서 똑같이 보였고, 실제로 `rnntop::` 35 건이 빠뜨린
# 것인 채로 그 안에 있었다. 여기 적고 나서야 `opt::LBFGS` 와 `index::searchsorted`
# 가 **없는 이름**이라는 것이 드러났다.
#
# 개수는 **정확히 맞아야 한다.** 여유를 두면 새로 안 옮긴 케이스가 그 틈에 숨는다 —
# 바로 그것이 이 검사가 막으려는 일이다. 케이스를 늘렸으면 옮기거나, 못 옮길
# 까닭과 함께 수를 올려라.
#
# 표시:  별칭 = 옮기면 같은 질문을 두 번 한다
#        파이썬 = 파이썬 표면의 것이라 TS 에 대응물이 없다
#        아직 = 옮길 값이 있는데 안 옮겼다 (**밀린 일이다**)
#        없음 = borch.ts 에 그 이름이 없다 (**결손이다**)
NOT_PORTED = {
    # 104 → 103. `dtype::없는이름::` 이 물던 이름을 `narrow_copy`·`unsafe_chunk` 로
    # 옮겼다 — 앞의 것은 torch 가 실제로 답해서 "없는 이름" 이 아니었다.
    # 103 → 111. 형 별칭 다섯(`dtype=torch.float` 꼴)과 공장 함수가 `dtype=`·
    # `requires_grad=` 를 실제로 쓰는지 묻는 셋. **파이썬 쪽 이야기다** — borch.ts 는
    # 형을 문자열로 받으므로 `torch.float` 이라는 이름 자체가 없고, 공장도
    # `Tensor.zeros(shape)` 라 형을 인자로 안 받는다.
    "dtype::": (111, "아직 — 형 보존은 borch.ts 도 지켜야 한다"),
    # 이 수가 82 에서 88 로 뛰는 것을 이 검사가 **붙인 날 잡았다** —
    # `x.real`·`x.device` 를 속성으로 바꾼 묶음이 케이스를 여섯 늘렸다.
    # 88 → 116. 술어 스무 개와 짝 없는 제자리 판 여덟을 넣었다. 둘 다 **파이썬
    # 표면의 이야기**다 — `is_cuda` 는 borch.ts 에 물을 자리가 없고, `apply_`·`map_`
    # 는 칸마다 파이썬 함수를 거는 것이라 GPU 로는 못 한다.
    # 116 → 144. 저장을 들여다보는 것(`stride`·`nbytes`)·전치의 세 이름·`new_*`·
    # `retain_grad` 를 넣었다. **여기 있는 이유가 갈래마다 다르다** — `stride` 는
    # 저쪽이 뷰를 안 만들어 답이 갈리고(그 갈림 자체를 케이스로 뒀다), `new_*` 는
    # 파이썬의 형 물려받기이고, `H`·`mT` 는 저쪽에 이름이 없다.
    # 144 → 158. 희소 접근자 일곱(`values`·`indices`·`crow_indices` …)과 저장·양자화
    # 쪽 없는 기능들, 그리고 `is_set_to`. **전부 거절이 답인 자리**다 — borch.ts 는
    # 조밀 텐서만 다루므로 저쪽에 물을 자리 자체가 없다.
    "inplace::": (158, "별칭·파이썬 — 제자리 이름, 뷰·공유·속성·술어·저장 들여다보기"),
    "method2::": (60, "별칭 — `multiply`=`mul` 처럼 파이썬의 둘째 이름"),
    "make::": (48, "아직 — `real`·`conj` 는 저쪽에도 있다"),
    "top::": (47, "별칭 — 최상위 제자리 함수. TS 는 메서드로만 준다"),
    "spot::": (47, "아직 — `asStrided`·`diagEmbed` 는 저쪽에도 있다"),
    "toplin::": (42, "별칭 — `lu`=`linalg.lu_factor` 처럼 최상위의 둘째 이름"),
    "stat::": (42, "아직 — `histc`·`histogram` 은 저쪽에도 있다(비동기다)"),
    "keep::": (35, "아직 — `dtype=` 는 저쪽 `castFirst` 가 한다"),
    "blend::": (34, "아직 — `addmm` 계열은 저쪽에도 있다"),
    "fname::": (28, "별칭 — `F` 의 제자리 판. 메서드 쪽에서 이미 묻는다"),
    "bit::": (24, "별칭 — 비트 연산의 메서드 이름"),
    "unpool::": (22, "파이썬 — 대부분 `repr` 이고 그것은 파이썬 쪽 글자다"),
    "linalg::": (17, "아직 — `*_ex` 변종. info 를 내는 자리라 값이 있다"),
    "grad::": (12, "별칭 — vjp 는 `backward(seed)` 이고 parity 가 이미 묻는다"),
    "cplx::": (10, "파이썬 — 복소수 `repr` 은 파이썬 formatter 의 것이다"),
    # 버퍼 케이스 열 중 다섯은 옮겼고(등록·저장 제외·목록·값 왕복), 다섯이 남았다.
    # **남은 것의 까닭이 둘로 갈린다.** `InstanceNorm` 셋은 borch.ts 에 그 **층**이
    # 없어서다 — 텐서 메서드 `instanceNorm` 은 있고 층은 파이썬 쪽에서 세운다.
    # 손실 둘은 거절을 묻는 케이스인데, TypeScript 에서는 없는 인자를 주는 것이
    # 실행 중 거절이 아니라 **컴파일 오류**라 물을 자리가 없다.
    "container::": (5, "파이썬 — InstanceNorm 층은 결속이 세우고, 거절은 파이썬 인자다"),
    "index::": (5, "**없음** — borch.ts 에 `searchsorted`·`bucketize` 가 없다"),
    # 4 → 10. "이어서 학습하기" 여섯이 늘었다. **저것들은 borch.ts 를 이미 밟는다** —
    # 결속의 옵티마이저·스케줄러가 저쪽 것을 그대로 부르므로 `--lib borch_webgpu`
    # 로 도는 그 여섯이 곧 borch.ts 의 `StepLR`·은행 왕복을 재는 것이고, TS 쪽
    # `serialize` 가 같은 것을 바이트로 한 번 더 붙잡는다. 넷은 여전히 `LBFGS` 다.
    "opt::": (10, "**없음**(LBFGS) · 이어서 학습하기는 결속이 같은 borch.ts 를 밟는다"),
    "cache::": (4, "별칭 — 전역 상수 오염은 parity 가 같은 것을 묻는다"),
    "dataconv::": (3, "파이썬 — `default_convert`·`get_worker_info` 는 파이썬 쪽이다"),
}


def unasked_report(report):
    """안 물어본 것을 **접두어로 묶어** 보여주고, 까닭 없는 것을 가려낸다.

    개수만 찍으면 그 수가 무엇으로 이루어졌는지 아무도 모른다. `679 건` 안에
    "일부러 안 옮긴 것" 과 "빠뜨린 것" 이 섞여 있어도 화면은 똑같다.
    """
    import json

    doc = json.loads((ROOT / "tests" / "golden.json").read_text(encoding="utf-8"))
    asked = set(report.get("asked", ()))
    rest = [n for n in doc["cases"] if n not in asked]
    groups = {}
    for name in rest:
        head = name.split("::", 1)[0] + "::" if "::" in name else "(접두어 없음)"
        groups.setdefault(head, []).append(name)

    lines = []
    surprise = []
    for head, names in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        entry = NOT_PORTED.get(head)
        if entry is None:
            lines.append(f"    ✘ {head:18s} {len(names):>4}건  "
                         f"**까닭이 안 적혀 있다**  [{names[0]}]")
            surprise.append(f"{head} ({len(names)}건, 까닭 없음)")
            continue
        frozen, why = entry
        mark = " " if len(names) == frozen else "✘"
        lines.append(f"    {mark} {head:18s} {len(names):>4}건  {why}")
        if len(names) != frozen:
            surprise.append(f"{head} (적힌 것 {frozen}, 실제 {len(names)})")
    if lines:
        lines.insert(0, "  안 물어본 것 — 접두어별:")
    # 목록에 있는데 하나도 안 남은 것은 **다 옮긴 것**이다. 그 줄은 지워야 한다 —
    # 안 지우면 다음 사람이 아직 안 옮긴 줄로 읽는다.
    for head, (frozen, _) in NOT_PORTED.items():
        if head not in groups:
            surprise.append(f"{head} (전부 옮겼다 — 목록에서 지워라)")
    if surprise:
        lines.append("  ✘ 맞춰지지 않는 자리: " + " · ".join(surprise))
        lines.append("     옮겼으면 수를 내리고, 못 옮겼으면 까닭과 함께 올려라.")
    return lines


def main(argv):
    dist = ROOT / "borch-ts" / "dist" / "test" / "golden.js"
    if not dist.exists():
        # 낡은 dist 로 도는 것보다 안 도는 편이 낫다.
        print(f"방출물이 없다: {dist}\n  먼저: npm run build:ts", file=sys.stderr)
        return 2

    report = run(headed="--headed" in argv, verbose="--verbose" in argv)
    if "error" in report:
        print(f"돌지 못했다: {report['error']}", file=sys.stderr)
        return 1

    # **어느 장치에서 돌았는지 먼저 적는다.** 값은 장치가 안 바꾸지만, 안 적어두면
    # 성능을 재는 쪽이 헤드리스의 소프트웨어 어댑터를 진짜 GPU 로 착각한다 —
    # 이 저장소에서 실제로 그렇게 됐다.
    adapter = report.get("adapter", "(모름)")
    print(f"어댑터: {adapter}")
    # **여기서는 막지 않는다.** 골든은 값을 묻고 값은 장치가 안 바꾸므로, CPU 에서
    # 통과해도 그것은 진짜 통과다. 다만 그 통과가 증명하는 범위가 좁아지므로 적는다 —
    # 실제로 리눅스 GPU 서버에서 845/845 를 받고 "다른 벤더에서 확인했다"고 읽을
    # 뻔했는데 어댑터가 `google / swiftshader` 였다.
    warn_if_software(adapter, "값")
    gap = report["total"] - report["registered"]
    print(f"골든 {report['total']}건 중 {report['registered']}건을 TS 로 썼다 "
          f"— {gap}건은 아직 안 물었다.")
    gap_lines = unasked_report(report)
    for line in gap_lines:
        print(line)
    gap_ok = not any("✘" in line for line in gap_lines)
    for name in report["unknown"]:
        print(f"  ? 이름이 골든에 없다: {name}")
    for f in report["failed"]:
        print(f"  ✘ {f['name']} — {f['why']}")
    print(f"통과 {report['passed']} / 실패 {len(report['failed'])}")

    # **안 물어본 것도 실패다** — 까닭이 안 적혀 있거나 수가 안 맞으면.
    # 골든이 자라는데 TS 쪽이 안 따라가는 것을 개수 한 줄로는 못 본다.
    ok = not report["failed"] and not report["unknown"] and gap_ok
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
