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

    ## mtime 은 사실이 아니라 대리물이다

    처음에는 시각만 봤다. 그런데 `cases.py` 의 **주석만** 고쳐도 시각은 움직인다 —
    다른 세션이 그 파일을 영어로 옮기자 러너가 멈췄고, 골든은 멀쩡했다. 케이스 이름도
    값도 그대로였는데 시각 하나 때문에 "다시 뽑아라" 가 나온 것이다.

    거짓 경보를 그냥 두면 사람이 그 경고를 지나치는 법을 배우고, 그러면 **진짜일 때도
    지나친다.** 그래서 시각이 어긋나면 거기서 멈추지 않고 **이름 표를 실제로 대조한다**
    (`manifest`). 같으면 시각만 움직인 것이므로 지나간다.

    **값까지는 안 본다.** 그것은 `tests/test_committed_golden.py` 의 일이고 진짜 torch
    없이도 도는데, 여기서 두 벌로 두면 언젠가 갈린다.
    """
    cases = root / "tests" / "cases.py"
    exported = root / "tests" / "golden.json"
    if not exported.exists():
        raise SystemExit(f"골든이 없다: {exported}")
    if cases.stat().st_mtime <= exported.stat().st_mtime:
        return
    if _names_still_match(root, exported):
        return
    raise SystemExit(
        "골든이 케이스 표보다 낡았다 — 러너는 `tests/golden.json` 을 읽는다.\n"
        "  먼저: uv run --with numpy --with torch --with torchvision "
        "python tests/golden.py dump\n"
        "  그다음: uv run --with numpy python tests/export_json.py\n"
        "  (이대로 돌리면 새 케이스가 `이름이 골든에 없다` 로 나오는데, 그것은\n"
        "   이름을 틀렸을 때와 **같은 문구**라 원인이 안 보인다.)")


def _names_still_match(root, exported):
    """굳혀 둔 이름 표가 지금 `cases.py` 가 내는 것과 같은가.

    **못 재면 못 잰다고 답한다.** `cases.py` 를 들여오려면 numpy 가 있어야 하고 이
    러너는 그것 없이도 도는 자리가 있다. 그때 "같다" 로 답하면 이 갈래가 검사를 끄는
    스위치가 된다 — 없는 확신을 만드는 쪽보다 시끄러운 쪽이 낫다.
    """
    import json
    import sys

    try:
        doc = json.loads(exported.read_text(encoding="utf-8"))
        stamped = doc.get("manifest")
        if not stamped:
            return False
        sys.path.insert(0, str(root))
        try:
            from tests import cases as cases_mod
        except ImportError:
            import cases as cases_mod
        return stamped == cases_mod.manifest_hash(cases_mod.golden_cases())
    except Exception:
        return False


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
    # 103 → 111 → 147 → 156 → 157. 형 별칭과 공장들이 `dtype=`·`requires_grad=` 를 실제로
    # 듣는지 묻는 것들. **파이썬 쪽 이야기다** — borch.ts 는 형을 문자열로 받으므로
    # `torch.float` 이라는 이름 자체가 없고, 공장도 `Tensor.zeros(shape)` 라 형을
    # 인자로 안 받는다. 기울기도 저쪽은 `requiresGrad()` 를 따로 부른다.
    #
    # 147 → 156 은 **열넷 밖에 남아 있던 아홉**이다. 창 함수 다섯이 `requires_grad=`
    # 를 삼키고 있었고(조용히 기울기 없는 잎을 준다), 번호 만드는 넷이 `dtype=` 을
    # 아예 안 받았다. 앞의 묶음이 목록으로 고쳐졌기에 그 목록 밖은 그대로였다.
    # 156 → 157 은 `norm(dtype=)` — 축약 중 그것만 안 듣고 있었다.
    # 157 → 160 은 `normal`·`frombuffer` 의 마지막 둘, 그리고 `frombuffer` 가 없는
    # 형을 **조용히 float32 로 떨어뜨리던** 자리다. 후보를 `**kw` grep 으로 넷 뽑았는데
    # torch 서명을 하나씩 보니 진짜는 둘이었다 — 목록을 기계로 만들면 아닌 것이 섞인다.
    # 160 → 164 는 `out=` 을 삼키지 않고 멈추는지 묻는 넷이다. borch.ts 에는
    # 그 인자가 아예 없다.
    # 164 → 172. `out=` 을 **실제로 구현**했다(전에는 거절이 답이었다). 미리 만든
    # 텐서에 쓰고 같은 객체를 돌려주는 것·모양이 다르면 다시 잡는 것·형과 기울기
    # 거절. **파이썬 쪽 이야기다** — borch.ts 에는 `out=` 이라는 인자가 없다.
    # 172 → 158. **이름 있는 형 바꾸기 열넷을 옮겼다.** 그중 여덟은 거절이 답이고,
    # 그 자리에서 맞추는 것은 값이 아니라 **문구**다 — 이름이 없으면
    # `'half' 이 없다` 가 나오는데 오타와 구별이 안 된다. 남은 158 은 옮길 수
    # 없다: `자리만::`·`공장::`·`별칭::`·`out::` 은 전부 파이썬 서명 이야기이고
    # borch.ts 는 형을 문자열로 받아 `torch.float` 이라는 이름 자체가 없다.
    # 기울기 넷만 남았다. **파이썬 쪽 헬퍼가 잎을 만든다** — `_grad_of` 가 잎에
    # 기울기가 도착했는지 확인하고 꺼내는 자리라 그 꼴을 TS 로 옮기면 두 벌이 된다.
    # 값 열일곱은 옮겼고, 커널의 결함을 잡은 것도 그쪽이다.
    "fft::": (4, "파이썬 — 기울기 헬퍼가 잎을 만드는 꼴"),
    # 158 → 86. **까닭 하나가 여덟 묶음을 덮고 있었다.** "아직" 은 밀린 일이라는
    # 뜻인데, 실제로 밀려 있던 것은 `자리만::` 63 과 `묻는것::` 9 뿐이었다 — 그 둘은
    # `t.dtype` 하나만 묻는 순수한 저쪽 성질이라 진작 물었어야 했다. 옮겼다.
    #
    # 나머지는 밀린 것이 아니라 **물을 자리가 없는 것**이다:
    #
    #   `공장::` 40  — 저쪽 공장이 `dtype=`·`requires_grad=` 를 안 받는다. torch 는
    #                  `zeros(3, dtype=int64)` 인데 이쪽은 만들고 나서 바꾸는 두
    #                  단계다. 표 어디에도 안 적혀 있던 API 차이이고, 여기가 그 기록이다.
    #   `별칭::` 16  — `torch.float` 이라는 이름 자체가 없다(형이 문자열이다).
    #   `out::`  12  — `out=` 인자가 없다. 흉내 내면 절약이 안 일어난다.
    #   `없는이름::` 11 · `조밀에도답::` 6 · `없는형::` 1 — 파이썬 표면.
    "dtype::": (86, "파이썬·설계 — 공장의 `dtype=`, 형 별칭, `out=`"),
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
    # 158 → 137. **한 까닭이 열넷을 덮고 있었다.** "별칭·파이썬" 은 큰 묶음
    # (`술어::` 23 은 `is_cuda`·`is_mps` 류, `저장::` 10 은 `stride`·`layout`,
    # 묶음 없는 47 은 사본 의미론)에 대해서는 맞았는데, 그 아래에 `분포::` 22 가
    # 숨어 있었다 — 그리고 그중 **열다섯은 이쪽 코드가 안 하던 것**이었다.
    #
    # 다섯 분포를 넣으면서 거절을 안 넣었다. 정수 칸에서 멈추지 않았고(연속 다섯은
    # 멈춰야 한다), 인자 정의역을 안 봤다(`p` 는 열린 구간, `lambda` 는 양수). 스물하나를
    # 옮기면서 그 셋을 채웠다. 남은 하나(`random_(int64) 의 상한`)는 **못 한다** —
    # torch 는 2⁶² 인데 이쪽 int64 는 f32 칸에 담겨 2²⁴ 위를 못 센다.
    #
    # 나머지 116 은 정말 별칭·파이썬이다. `짝에서::` 40 중 저쪽에 이름이 있는 것은
    # `i0_` 하나뿐인데, 그것들은 대개 torch 의 **둘째 철자**(`divide_`=`div_`)이거나
    # 비트·논리 제자리 판이라 옮기면 같은 질문이 두 번이 된다.
    # 137 → 97. `짝에서::` 40 을 옮겼다. 그 마흔이 "별칭" 으로 적혀 있었는데
    # **별칭인 것은 열뿐이었다**(`divide_`=`div_` 같은 둘째 철자). 열일곱은 계산이
    # 있는데 밑줄 이름만 없었고, 열하나는 커널 표에만 있어서 `binary("gcd", …)` 로만
    # 닿았다 — torch 에서 옮겨 온 코드가 치는 줄은 `x.gcd(y)` 다.
    #
    # 남은 97 은 정말 파이썬이다: `술어::` 23(`is_cuda`·`is_mps`), `저장::` 10
    # (`stride`·`layout`), 묶음 없는 47(사본 의미론·`from_numpy`), `희소::` 5.
    "inplace::": (97, "파이썬 — 뷰·공유·속성·술어·저장 들여다보기"),
    # `method2::` 이 여기 있었다 — 60 건, "별칭 — `multiply`=`mul` 처럼 파이썬의
    # 둘째 이름". 별칭인 것은 그중 일부였고, **아홉은 저쪽에 이름이 아예 없었다**
    # (`fmax`·`vdot`·`moveaxis`·`t`·`broadcast_to`·비교 넷). 넣고 전부 옮겼다.
    #
    # 옮기다가 둘이 더 나왔다. `fmax`·`fmin` 은 결속에만 있던 조립이었고,
    # `remainder` 는 수만 받아서 `x.remainder(y)` 가 안 돌았다 — **있는데 좁은
    # 이름**은 없는 이름보다 찾기 어렵다.
    # 48 → 9. 복소수의 이웃 서른아홉을 옮겼다 — `real`·`conj`·`conjPhysical`·
    # `resolveConj`·`resolveNeg`·`angle` 을 세 형에 값과 형으로 물었고, 판정 셋도.
    # 없던 넷(`resolveConj`·`resolveNeg`·`isConj`·`isNeg`)은 만들었다. **게으른
    # 켤레 비트가 없다는 것은 구현의 사정이지 그 물음이 뜻을 잃는 이유가 아니다.**
    #
    # 남은 아홉은 **저쪽에 그 공장이 없다** — `range`(끝을 포함한다)·`frombuffer`·
    # `asarray`. 앞의 둘은 이름 자체가 없고, `asarray` 는 TS 에서 `Tensor.from` 이
    # 그 자리다. 그리고 `arange` 는 인자를 **하나만** 받는다(torch 는 셋).
    # 9 → 2. `range`·`frombuffer` 를 저쪽에 넣었고 `arange` 도 인자 셋을 받는다.
    # 남은 둘은 `asarray` 로, numpy 배열과 파이썬 목록을 받는 자리라 TS 에 없다.
    "make::": (2, "파이썬 — `asarray` 는 ndarray·목록을 받는다. 저쪽엔 그 둘이 없다"),
    # 47 → 50. `finfo`·`iinfo` 의 **종류**와 인자 없는 기본형. 파이썬 쪽
    # 이야기다 — borch.ts 에는 그 두 이름이 없다.
    # 50 → 39. **까닭이 열하나만 설명하고 있었다.** "최상위 제자리 함수" 는
    # `제자리::` 열(떨구기 넷과 `nan_to_num_`)의 이야기였고, 그 넷은 저쪽에 이름이
    # 없어서 못 옮기던 것이었다 — 넣고 옮겼다.
    #
    # 남은 39 는 **파이썬 표면**이고 성격이 셋으로 갈린다:
    #
    #   `살펴보기::` 16 — `finfo`·`iinfo`·`can_cast`·`promote_types`·`typename`.
    #                     형을 값으로 들여다보는 일이라 형이 문자열인 저쪽에 자리가 없다.
    #   `device::`   9  — `torch.device` 는 `.type`·`.index` 를 가진 **객체**다.
    #                     저쪽 `t.device` 는 문자열이고, 어댑터를 내는 `device()` 는
    #                     아예 다른 함수다.
    #   나머지 14   — `resize_as_`(손잡이를 갈아 끼운다) · `inference_mode`(with 문) ·
    #                 난수 상태 왕복 · 정수 열거형을 받는 최상위 서명들.
    "top::": (39, "파이썬 — 형 들여다보기, `device` 객체, with 문, 정수 열거형 서명"),
    # `spot::` 이 여기 있었다 — 47 건, "아직". 전부 옮겼으므로 줄을 지운다.
    "toplin::": (42, "별칭 — `lu`=`linalg.lu_factor` 처럼 최상위의 둘째 이름"),
    # `stat::` 이 여기 있었다 — 42 건. 31 건은 그냥 안 물어본 것이었고, 나머지 11 은
    # **이름이 저쪽에 없어서** 못 옮기던 것이라 그 다섯을 borch.ts 에 넣었다.
    # `keep::` 이 여기 있었다 — 35 건, "아직". 전부 옮겼으므로 줄을 지운다.
    #
    # 옮겨 보니 **값은 서른네 건이 첫 시도에 맞았다.** 축약의 `dtype=` 은 "넣기 전에
    # 바꾼다" 는 규칙도, `sum(→bool)` 은 되고 `cumsum(→bool)` 은 안 된다는 갈림도
    # 이미 지켜지고 있었다. 안 지켜진 것은 하나 — **`sum` 만 `dtype=` 을 받을 자리가
    # 없었다.** 이웃(`mean`·`prod`·`nansum`·`cumsum`·`sumDim`)이 전부 받는데
    # 축약에서 제일 많이 불리는 이름 하나가 빠져 있었다.
    # `blend::` 가 여기 있었다 — 34 건. 전부 옮겼으므로 줄을 지운다.
    #
    # **서른넷이 첫 시도에 맞았다.** `beta=0` 이 값에서는 빠지고 그래프에는 남는
    # 자리도, `input` 이 `(4,)` 나 스칼라로 퍼지는 자리도, 제자리 판이 자기를
    # 돌려주는 자리도 이미 옳았다. 여기서는 결손이 안 나왔다 — **묻지 않았을 뿐인
    # 곳과 틀린 곳은 다르고, 그 둘은 물어봐야 갈린다.**
    "fname::": (28, "별칭 — `F` 의 제자리 판. 메서드 쪽에서 이미 묻는다"),
    # `bit::` 이 여기 있었다 — 24 건, "별칭 — 비트 연산의 메서드 이름". 그 이름들이
    # **저쪽에 없었다는 것이 요점이었는데** 까닭이 그것을 별칭이라 불렀다. 넣고 전부
    # 옮겼으므로 줄을 지운다.
    #
    # 옮기다가 `bitwise_not(참거짓)` 이 갈렸다: 커널 주석이 "참거짓이면 논리 부정이고
    # 그 갈림은 결속이 한다" 고 적어 두었는데, 그러면 TypeScript 쪽은 `-2` 를 받는다 —
    # **없는 답이 아니라 틀린 답**이다. 갈림을 저쪽으로 옮겼다.
    # **"대부분 `repr`" 이 아니었다.** `--show unpool` 로 펴 보니 스물둘 중 `repr` 은
    # 여섯뿐이고, 나머지 열넷은 값·기울기·모양을 묻는다. 그리고 그 이름들이
    # borch.ts 에 **이미 있다** — `CTCLoss`·`FractionalMaxPool`·
    # `AdaptiveLogSoftmaxWithLoss` 셋 다 클래스가 서 있고 케이스만 안 왔다.
    #
    # 한 줄로 굳은 까닭은 그 안에서 성격이 갈려도 안 보인다. 이 줄이 그것을 여섯 번째로
    # 보여줬고, 그래서 수를 성격별로 적는다.
    "unpool::": (20, "**아직**(값 14 — CTCLoss·FractionalMaxPool·적응형softmax 는 "
                     "이름이 서 있고 케이스만 안 왔다) · repr 6 은 파이썬 쪽 글자다"),
    # `linalg::` 이 여기 있었다 — 17 건. 열여섯은 그냥 안 물어본 것이었고, 하나
    # (`ldl_factor_ex`)는 결속이 자리 셋을 손으로 세우고 있어서 못 물던 것이다.
    "grad::": (12, "별칭 — vjp 는 `backward(seed)` 이고 parity 가 이미 묻는다"),
    "cplx::": (10, "파이썬 — 복소수 `repr` 은 파이썬 formatter 의 것이다"),
    # 버퍼 케이스 열 중 다섯은 옮겼고(등록·저장 제외·목록·값 왕복), 다섯이 남았다.
    # **남은 것의 까닭이 둘로 갈린다.** `InstanceNorm` 셋은 borch.ts 에 그 **층**이
    # 없어서다 — 텐서 메서드 `instanceNorm` 은 있고 층은 파이썬 쪽에서 세운다.
    # 손실 둘은 거절을 묻는 케이스인데, TypeScript 에서는 없는 인자를 주는 것이
    # 실행 중 거절이 아니라 **컴파일 오류**라 물을 자리가 없다.
    "container::": (5, "파이썬 — InstanceNorm 층은 결속이 세우고, 거절은 파이썬 인자다"),
    # `torch.pi`·`inf`·`nan`·`newaxis` 는 **파이썬 최상위의 값**이다. borch.ts 는
    # 모듈이 아니라 클래스 묶음이고, JS 에는 `Math.PI`·`Infinity`·`null` 이 이미
    # 있어서 같은 이름을 다시 낼 자리가 없다 — `x[:, None]` 도 저쪽에서는
    # `unsqueeze(1)` 이라 색인 문법 자체가 파이썬 쪽 이야기다.
    "const::": (6, "파이썬 — 최상위 값과 색인 문법. JS 에는 그 자리가 없다"),
    # 5 → 2. `searchSorted`·`bucketize` 가 생겼다(이진 탐색 커널 하나). 남은 둘은
    # **거절**이고 파이썬 쪽 이야기다 — torch 가 같은 것을 `right`(참거짓)와
    # `side`(글자) 두 이름으로 받고, 그 둘이 어긋나면 멈춘다. borch.ts 는 `right`
    # 하나만 알므로 어긋날 짝이 없다.
    "index::": (2, "파이썬 — `side` 와 `right` 를 맞춰 보는 일. TS 는 하나만 안다"),
    # 4 → 10. "이어서 학습하기" 여섯이 늘었다. **저것들은 borch.ts 를 이미 밟는다** —
    # 결속의 옵티마이저·스케줄러가 저쪽 것을 그대로 부르므로 `--lib borch_webgpu`
    # 로 도는 그 여섯이 곧 borch.ts 의 `StepLR`·은행 왕복을 재는 것이고, TS 쪽
    # `serialize` 가 같은 것을 바이트로 한 번 더 붙잡는다. 넷은 여전히 `LBFGS` 다.
    # 10 → 14. `save`/`load` 왕복 넷이 늘었다. **TS 쪽은 `serialize` 가 이미
    # 바이트로 묻는다** — 같은 코덱을 왕복시키고, 남이(numpy·파이썬 `borch`) 읽는지
    # 까지 확인한다. 골든으로 한 번 더 묻는 것은 파이썬의 `torch.save(경로)` 표면
    # 이고 borch.ts 는 바이트만 다루므로(파일은 페이지의 일이다) 물을 자리가 없다.
    # 14 → 17. LBFGS 케이스 셋이 늘었다 — 기존 셋이 **준뉴턴 부분을 한 번도 안
    # 밟고 있었다.** 닫힘이 기울기를 상수로 넣어 주어서 `y = 0`, `ys = 0` 이고
    # 이력에 아무것도 안 쌓였다. 이름은 LBFGS 인데 재는 것은 첫 반복의 경사하강뿐.
    #
    # **borch.ts 에 없는 이유가 "안 만들었다" 가 아니다.** 이 알고리즘은 제어 흐름이
    # 값에 달렸는데(`ys > 1e-10`, 수렴 판정) 저쪽 읽기는 전부 비동기다 — 동기
    # `step()` 안에서 GPU 위의 수를 볼 수가 없다. 결속은 `run_sync` 가 있어서 된다.
    # 넣으려면 `async step(closure)` 가 되어야 하고, 그러면 저쪽에서 **혼자만
    # 비동기인 옵티마이저**가 된다.
    # `opt::` 는 **한 줄도 안 남았다.** 적혀 있던 까닭 셋이 차례로 틀렸다 —
    # "LBFGS 는 동기 step 으로는 못 쓴다"(사실이지만 결론이 틀렸다: `step` 을
    # 비동기로 두면 된다), "이어서 학습은 결속이 밟는다"(아예 사실이 아니었다),
    # "save/load 는 평평한 텐서 표만 받는다"(그때는 사실이었고, 이제 나무를 든다).
    # `vision::` 이 목록에 없어서 케이스들이 **까닭 없이** 남아 러너가 거절했다
    # (`fda5540` 이후). 한 줄로 "아직" 이라고만 적으면 그 안에서 성격이 갈리는 것이
    # 안 보이므로 나눠 적는다 — 이 파일이 접두어별로 나눠 적는 것과 같은 이유가
    # 한 칸 더 안쪽에 있다.
    #
    # **42 → 38 이 그 나눔이 한 일이다.** 나눠 적었을 때 42 는 "밀린 일 38 + 결손 4"
    # 였다. 결손 넷은 저쪽이 **이미 가진** 변환의 인자가 좁아서 못 맞추던 것이고
    # (`Resize` 의 `max_size`, `describe()` 의 네 자리 중 둘, `RandomCrop` 의
    # `padding` 기본값), `ae60832` 에서 고쳐져 넷 다 답을 냈다. 한 수로 적었다면
    # 그 넷은 백로그로 계산돼 아무도 손대지 않았을 것이다.
    #
    # 남은 43 은 순수한 밀린 일이다. `borchvision` 이 스물하나를 들고 `vision.ts` 는
    # 일곱만 든다. 열넷이 저쪽에 없고, 거기에 `transforms.functional` 의 다섯이
    # 붙는다 — 저쪽에는 그 네임스페이스 자체가 없다.
    "vision::": (43, "아직 — 열다섯이 `vision.ts` 에 없다 (값 29 · repr 14)"),
    "cache::": (4, "별칭 — 전역 상수 오염은 parity 가 같은 것을 묻는다"),
    "dataconv::": (3, "파이썬 — `default_convert`·`get_worker_info` 는 파이썬 쪽이다"),
}


def unasked_report(report, show=None):
    """안 물어본 것을 **접두어로 묶어** 보여주고, 까닭 없는 것을 가려낸다.

    개수만 찍으면 그 수가 무엇으로 이루어졌는지 아무도 모른다. `679 건` 안에
    "일부러 안 옮긴 것" 과 "빠뜨린 것" 이 섞여 있어도 화면은 똑같다.

    `show` 에 접두어를 주면 **그 자리의 이름을 전부 편다.** 개수와 한 줄짜리 까닭만
    보고는 "무엇이 빠졌는지" 를 물을 수가 없었다 — 한 묶음을 옮기려면 먼저 그 목록을
    봐야 하는데, 그 목록을 내주는 자리가 없어서 케이스 표를 손으로 뒤져야 했다.
    까닭이 한 줄로 굳으면 그 안에서 성격이 갈려도 안 보인다는 것이 이 저장소가
    접두어별로 나눠 적기 시작한 이유이고, 같은 이유가 한 칸 더 안쪽에도 있었다.
    """
    import json

    doc = json.loads((ROOT / "tests" / "golden.json").read_text(encoding="utf-8"))
    asked = set(report.get("asked", ()))
    rest = [n for n in doc["cases"] if n not in asked]
    groups = {}
    for name in rest:
        head = name.split("::", 1)[0] + "::" if "::" in name else "(접두어 없음)"
        groups.setdefault(head, []).append(name)

    if show is not None:
        want = show if show.endswith("::") else show + "::"
        names = groups.get(want, [])
        out = [f"  {want} 에서 안 물어본 것 {len(names)}건:"]
        out.extend(f"    · {n}" for n in sorted(names))
        return out

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
    show = argv[argv.index("--show") + 1] if "--show" in argv else None
    gap_lines = unasked_report(report, show)
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
