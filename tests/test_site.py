"""사이트가 저장소와 어긋나지 않았는가.

`tests/test_docs.py` 가 문서의 **수**를 붙잡는 것과 같은 자리다. 이쪽이 붙잡는 것은
**생성물**이다 — `site/assets/api.json` 은 `site/build_api.py` 가 선언 파일에서 뽑은
것이고, 소스가 자란 뒤 다시 안 뽑으면 사이트가 없는 API 를 보여주거나 새 API 를 빠뜨린다.

그 어긋남은 **화면에서 안 보인다.** 목록이 조금 짧은 것과 원래 그만큼인 것이 같은
모양이라, 사람이 눈으로 찾는 방식은 여기서 특히 안 듣는다(생성기를 쓰다가 실제로 겪었다 —
파서가 텐서 메서드 422 개 중 18 개만 물고 있었는데 화면은 멀쩡해 보였다).
"""

import gzip
import json
import pathlib
import re
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
GENERATOR = ROOT / "site" / "build_api.py"
API = ROOT / "site" / "assets" / "api.json"
INDEX = ROOT / "site" / "assets" / "api-index.json"
DECL = ROOT / "borch-ts" / "dist" / "src"


def test_api_reference_is_not_stale():
    """`api.json` 이 지금 선언 파일에서 뽑은 것과 같아야 한다.

    선언 파일은 `.gitignore` 라 어느 커밋에도 없다 — 없으면 대조할 것이 없으므로
    건너뛴다. **없는 것을 실패로 만들면** 방출물을 안 만든 체크아웃에서 이 검사가
    빨개지고, 그러면 사람이 검사를 끄는 법을 배운다.
    """
    if not DECL.exists():
        pytest.skip(f"선언 파일이 없다({DECL.relative_to(ROOT)}) — 먼저 npm run build:ts")
    if not API.exists():
        pytest.fail("site/assets/api.json 이 없다 — python3 site/build_api.py")

    # **생성기가 만드는 것은 둘이다.** 목록과 이름 색인. 하나만 되돌려 놓았다가
    # 검사를 돌린 트리에 `api-index.json` 이 고쳐진 채로 남았다 — 검사가 작업 트리를
    # 건드리고 가면, 다음 사람은 자기가 안 한 변경을 자기 것으로 커밋한다.
    made = [API, INDEX]
    before = {p: p.read_text(encoding="utf-8") for p in made if p.exists()}
    proc = subprocess.run([sys.executable, str(GENERATOR)], capture_output=True, text=True)
    assert proc.returncode == 0, f"생성기가 멈췄다:\n{proc.stderr}"
    after = API.read_text(encoding="utf-8")
    for path, text in before.items():
        path.write_text(text, encoding="utf-8")

    if before.get(API) != after:
        old, new = json.loads(before[API]), json.loads(after)
        pytest.fail(
            "API 레퍼런스가 선언 파일과 다르다 — 항목 "
            f"{old['total']} → {new['total']}.\n"
            "  다시 뽑아라: python3 site/build_api.py\n"
            "  (설명문을 고칠 곳은 이 파일이 아니라 소스의 주석이다.)\n"
            "\n"
            "  **borch-ts/src 를 고치는 중이라면 이것이 정상이다.** 이 검사는 지금\n"
            "  디스크에 있는 `dist` 와 대조하므로, 커밋 안 한 소스로 빌드해 두었으면\n"
            "  아직 없는 이름까지 세어 여기서 갈린다. 그 변경을 커밋할 때 목록도\n"
            "  같이 뽑으면 맞는다 — 사이트가 없는 API 를 보여주지 않게 하려는 것이\n"
            "  이 검사의 목적이고, 그 시점이 바로 지금이다.")


def test_site_examples_name_only_real_modules():
    """사이트가 파이썬 결속을 실을 때 적어 둔 모듈 목록이 실제와 같아야 한다.

    `site/assets/runner.js` 는 Pyodide 가상 파일시스템에 얹을 `.py` 를 이름으로 적어
    둔다. 하나 빠뜨리면 **ImportError 로 시끄럽게** 터지므로 그건 스스로 드러나는데,
    반대쪽(패키지에서 파일이 사라졌는데 목록에 남은 경우)은 fetch 가 404 를 내고
    `runner.js` 가 그것을 예외로 바꾼다 — 둘 다 사용자가 Run 을 누른 뒤에야 안다.
    여기서 미리 본다.
    """
    runner = (ROOT / "site" / "assets" / "runner.js").read_text(encoding="utf-8")
    block = runner[runner.index("const PACKAGES = {"):runner.index("let pyodide")]
    for package in ("borch", "borch_webgpu"):
        listed = set()
        chunk = block[block.index(f"{package}:"):]
        for line in chunk.splitlines():
            if "]" in line:
                listed.update(part.strip().strip('",') for part in line.split('"') if part.startswith("_") or part == "__init__")
                break
            listed.update(part.strip().strip('",') for part in line.split('"') if part.startswith("_") or part == "__init__")
        real = {p.stem for p in (ROOT / package).glob("*.py")}
        missing = listed - real
        assert not missing, (
            f"{package} 에 없는 모듈을 사이트가 싣는다: {sorted(missing)}\n"
            "  site/assets/runner.js 의 PACKAGES 를 고쳐라.")
        forgotten = real - listed
        assert not forgotten, (
            f"{package} 의 모듈을 사이트가 안 싣는다: {sorted(forgotten)}\n"
            "  site/assets/runner.js 의 PACKAGES 에 넣어라 — 빠지면 브라우저에서 "
            "ImportError 로 터진다.")

# ── 크기를 대는 자리 ──────────────────────────────────────────────────
#
# **`KB` 도 수다.** `test_docs.py` 가 골든 개수와 패키지 줄 수를 붙잡는데 크기만
# 빠져 있었고, 그 사이 "ES 모듈 232KB" 가 **3.3 배** 낡았다(실측 770KB). 그 수는
# README 에서 사이트 두 페이지로 그대로 옮겨 적혔다 — 원천이 낡으면 사본도 낡는다.
#
# 여기 있는 셋은 전부 **잴 수 있는 것**이다. 못 재는 수는 애초에 이 목록에 없다.
SIZE_CLAIMS = (
    # (문서, 그 줄에 있어야 하는 표시, 실제를 재는 함수 이름)
    ("README.md", "ES 모듈", "bundle"),
    ("site/index.html", "ES module", "bundle"),
    ("site/ko/index.html", "ES 모듈", "bundle"),
)

# 재는 것이 얼마나 어긋나도 되는가. `test_docs.py` 의 줄 수와 같은 5% 다 — 잡으려는
# 것은 3.3 배 오차이지 커밋 하나가 만드는 몇 킬로바이트가 아니다.
SIZE_TOLERANCE = 0.05

KB = re.compile(r"(\d{2,5})\s*KB")


def _bundle_sizes():
    """브라우저가 싣는 ES 모듈의 (압축 전, gzip) 크기를 KB 로."""
    raw = b"".join(p.read_bytes() for p in sorted(DECL.glob("*.js")))
    return len(raw) / 1024, len(gzip.compress(raw, 9)) / 1024


def test_docs_do_not_name_a_stale_bundle_size():
    """문서가 대는 **크기**가 실제와 크게 어긋나지 않아야 한다.

    한 줄에 여러 수가 있을 수 있다(압축 전과 gzip). 그중 **아무것도** 실제와 안 맞으면
    낡은 것이다 — 하나만 맞아도 통과시키면 "232KB(압축 전)" 처럼 **수는 맞고 이름표가
    틀린** 문장을 놓친다.
    """
    if not DECL.exists():
        pytest.skip(f"선언 파일이 없다({DECL.relative_to(ROOT)}) — 먼저 npm run build:ts")

    raw_kb, gzip_kb = _bundle_sizes()
    ok = lambda said: any(abs(said - real) <= real * SIZE_TOLERANCE
                          for real in (raw_kb, gzip_kb))

    stale = []
    for rel, marker, _ in SIZE_CLAIMS:
        path = ROOT / rel
        if not path.exists():
            continue
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if marker not in line:
                continue
            said = [int(hit) for hit in KB.findall(line)]
            if not said:
                stale.append(f"{rel}:{i}  크기를 안 적었다")
            elif not all(ok(v) for v in said):
                stale.append(
                    f"{rel}:{i}  {said} KB — 지금은 압축 전 {raw_kb:.0f}KB · "
                    f"gzip {gzip_kb:.0f}KB")
    assert not stale, (
        "문서가 대는 방출물 크기가 낡았다:\n  " + "\n  ".join(stale) +
        "\n\n재서 고쳐라: cat borch-ts/dist/src/*.js | wc -c")

# ── 페이지들이 서로 맞는가 ─────────────────────────────────────────────
#
# 사이트는 지금 스무 페이지이고 두 언어다. **손으로 훑는 것은 방식이 아니다** —
# 실제로 브라우저로 스무 장을 열어 세 가지를 잡았다(랜딩만 앵커를 더 갖고 있었고,
# 플레이그라운드에 자기 항목이 없었고, 한국어 API 페이지의 이름표가 영어였다).
# 그 방식은 다음번에 반복되지 않는다.

SITE = ROOT / "site"
HREF = re.compile(r'(?:href|src)="([^"]+)"')
NAV = re.compile(r'<header class="top">.*?<nav>(.*?)</nav>', re.S)
LINK_TEXT = re.compile(r'<a [^>]*>([^<]*)</a>')


def _pages():
    return sorted(SITE.rglob("*.html"))


def test_site_has_no_broken_relative_links():
    """페이지가 가리키는 상대 경로가 실제로 있어야 한다.

    깨진 링크는 **누르기 전까지 안 보인다.** 문서 사이트에서 그것은 없는 페이지가
    아니라 없는 신뢰다 — 한 번 404 를 만난 사람은 나머지도 의심한다.
    """
    missing = []
    for page in _pages():
        for raw in HREF.findall(page.read_text(encoding="utf-8")):
            if raw.startswith(("http://", "https://", "#", "data:", "mailto:")):
                continue
            target = (page.parent / raw.split("#")[0].split("?")[0]).resolve()
            if target.is_dir():
                target = target / "index.html"
            if not target.exists():
                missing.append(f"{page.relative_to(ROOT)} → {raw}")
    assert not missing, "사이트에 깨진 링크가 있다:\n  " + "\n  ".join(missing)


def test_every_page_carries_the_same_global_nav():
    """전역 차림표는 **모든 페이지에서 한 벌**이어야 한다.

    페이지를 옮길 때 항목이 바뀌면 다른 사이트에 온 것처럼 읽힌다. 지금 어디에
    있는지는 표시(`class="on"`)로만 갈리고, 그 표시는 **정확히 하나**여야 한다 —
    둘이면 어디에 있는지 모르는 것이고, 없으면(랜딩만 예외) 차림표에 그 자리가
    없다는 뜻이다.

    언어별로 이름표가 다르므로 언어끼리 비교한다.
    """
    shapes = {}
    problems = []
    for page in _pages():
        text = page.read_text(encoding="utf-8")
        nav = NAV.search(text)
        assert nav, f"{page.relative_to(ROOT)} 에 전역 차림표가 없다"
        block = nav.group(1)
        # 언어 전환 고리는 페이지마다 목적지가 다르므로 이름표만 본다.
        labels = tuple(LINK_TEXT.findall(block))
        lang = "ko" if page.relative_to(SITE).as_posix().startswith("ko/") else "en"
        shapes.setdefault(lang, {}).setdefault(labels, []).append(
            page.relative_to(ROOT).as_posix())

        marked = block.count('class="on"')
        is_home = page.name == "index.html" and page.parent in (SITE, SITE / "ko")
        if is_home and marked:
            problems.append(f"{page.relative_to(ROOT)}: 첫 화면인데 차림표에 표시가 있다")
        elif not is_home and marked != 1:
            problems.append(f"{page.relative_to(ROOT)}: 현재 위치 표시가 {marked} 개")

    for lang, found in shapes.items():
        if len(found) > 1:
            lines = [f"  {labels} ← {len(pages)} 쪽 (예: {pages[0]})"
                     for labels, pages in found.items()]
            problems.append(f"{lang} 페이지들의 차림표가 갈렸다:\n" + "\n".join(lines))

    assert not problems, "\n".join(problems)


def test_site_links_to_this_repository():
    """사이트가 가리키는 GitHub 주소가 **이 저장소의 주소**여야 한다.

    저장소 이름이 `browsertorch` 에서 `borch` 로 바뀌었을 때 사이트 38 개 파일이 옛
    주소를 그대로 들고 있었다. `tests/rename.py` 는 소문자 식별자를 바꾸는 도구라
    URL 안의 이름을 규칙에 넣어 두지 않았고, 링크는 리다이렉트로 **여전히 열렸다** —
    깨지지 않는 낡음이라 아무도 안 봤다.

    그래서 손으로 적은 주소를 손으로 지키지 않는다. `origin` 이 답을 갖고 있으므로
    거기에 못 박는다. 원격이 없는 체크아웃(압축본·CI 의 일부 모드)에서는 물을 곳이
    없으므로 건너뛴다.
    """
    remote = subprocess.run(["git", "remote", "get-url", "origin"],
                            cwd=ROOT, capture_output=True, text=True)
    if remote.returncode != 0 or not remote.stdout.strip():
        pytest.skip("origin 이 없다 — 대조할 주소가 없다")

    here = remote.stdout.strip().removesuffix(".git").replace("git@github.com:",
                                                             "https://github.com/")
    # **우리 조직의 주소만 본다.** 사이트는 Pyodide 저장소도 가리키는데(MPL-2.0 이
    # 소스를 구할 길을 적으라고 한다) 그것은 남의 것이고 여기와 같을 이유가 없다.
    # 처음에 전부 보게 했다가 정확히 그 고지 링크가 걸렸다.
    owner = here.rsplit("/", 2)[-2]
    linked = re.compile(rf"https://github\.com/{re.escape(owner)}/[\w.-]+")
    wrong = []
    for page in _pages():
        for i, line in enumerate(page.read_text(encoding="utf-8").splitlines(), 1):
            for hit in linked.findall(line):
                if hit.removesuffix(".git") != here:
                    wrong.append(f"{page.relative_to(ROOT)}:{i}  {hit} — 이 저장소는 {here}")
    assert not wrong, (
        "사이트가 다른 저장소를 가리킨다:\n  " + "\n  ".join(wrong[:12]) +
        (f"\n  … 그리고 {len(wrong) - 12} 곳 더" if len(wrong) > 12 else ""))


def test_share_metadata_is_complete_and_gets_an_address():
    """모든 페이지에 공유 메타가 있고, 자리표시자를 채우는 쪽이 존재해야 한다.

    사이트는 "URL 하나가 곧 배포다" 라고 말한다. 그 말이 사실이려면 링크를 붙였을 때
    무엇이 뜨는지가 그 주장의 일부인데, 여태 아무것도 안 떴다.

    주소는 배포하는 쪽만 안다. 그래서 HTML 에는 `%OG_BASE%` 를 두고 워크플로가
    채운다 — **두 반쪽이 따로 놀 수 있는 구조**라 여기서 같이 있는지 본다. 자리표시자만
    있고 채우는 단계가 없으면 크롤러가 `%OG_BASE%/…` 를 그대로 받아 가고, 그 실패는
    배포한 뒤 남의 타임라인에서만 보인다.
    """
    missing = [str(p.relative_to(ROOT)) for p in _pages()
               if "og:image" not in p.read_text(encoding="utf-8")]
    assert not missing, "공유 메타가 없는 페이지:\n  " + "\n  ".join(missing)

    image = SITE / "assets" / "og.png"
    assert image.exists(), "site/assets/og.png 이 없다 — uv run --with pillow python site/make_og.py"
    size = image.stat().st_size / 1024
    assert size < 300, f"og.png 이 {size:.0f}KB 다 — 소셜 쪽에서 안 받아 갈 수 있다"

    workflow = ROOT / ".github" / "workflows" / "pages.yml"
    if not workflow.exists():
        pytest.skip("배포 워크플로가 없다")
    text = workflow.read_text(encoding="utf-8")
    assert "%OG_BASE%" in text, (
        "페이지가 %OG_BASE% 를 쓰는데 배포 워크플로가 그것을 안 채운다 — "
        "그대로 나가면 크롤러가 자리표시자를 주소로 읽는다.")


def test_dual_language_blocks_do_not_lose_a_half():
    """한 블록에 두 언어를 담을 때, 한 벌이 조용히 사라지는 자리를 막는다.

    `runnable.js` 는 원본을 **언어를 열쇠로** 담는다. 두 번째 `<script>` 에
    `data-lang` 을 안 적으면 바깥 `div` 의 언어로 쳐서 첫 번째 위에 덮어쓰고, 화면에는
    탭도 안 나온다 — 파이썬을 적어 넣었는데 페이지는 자바스크립트만 보여 주고, 아무도
    안 터진다. 이 저장소가 이미 두 번 잡은 그 꼴이라 여기서 이름을 대며 멈춘다.

    바깥 `div` 의 `data-lang` 도 본다. 그것이 담긴 언어 중에 없으면 처음 보이는 쪽이
    없는 언어라, 읽는 사람은 자기가 고른 적 없는 표면을 먼저 만난다.
    """
    inner = re.compile(r'<script type="text/plain"([^>]*)>', re.S)
    wrong = []
    for page in _pages():
        text = page.read_text(encoding="utf-8")
        for m in re.finditer(r'<div class="runnable"([^>]*)>', text):
            head = text[m.end():m.end() + 400]
            attrs = m.group(1)
            outer = "py" if 'data-lang="py"' in attrs else "js"
            # 이 블록에 속한 <script> 만 — 다음 runnable 전까지.
            stop = text.find('<div class="runnable"', m.end())
            body = text[m.end():stop if stop > 0 else len(text)]
            langs = [("py" if 'data-lang="py"' in a else "js" if 'data-lang="js"' in a else outer)
                     for a in inner.findall(body)]
            where = f"{page.relative_to(ROOT)} · {head.strip()[:40]}"
            if len(langs) != len(set(langs)):
                wrong.append(f"{where} — 같은 언어가 둘: {langs}")
            elif langs and outer not in langs:
                wrong.append(f"{where} — 바깥은 {outer} 인데 담긴 것은 {langs}")
    assert not wrong, "이중 언어 블록이 한 벌을 잃는다:\n  " + "\n  ".join(wrong)


def test_no_block_declares_a_name_the_runner_injects():
    """실행 블록이 러너가 주입하는 이름을 다시 선언하면 그 블록은 안 돈다.

    러너는 사용자 코드 앞에 `log`·`show` 같은 이름을 펼쳐 놓고 한 모듈로 합친다.
    블록이 같은 이름을 `const` 로 선언하면 합쳐진 모듈이 **문법 오류**가 되고, 그것은
    누가 그 블록을 눌러야만 보인다.

    이 저장소에서 세 번 났다. `probe` 로 두 번, `show` 로 한 번 — `show` 는 튜토리얼을
    붙이며 주입 목록에 이름을 하나 더한 순간 8강의 블록이 조용히 죽은 것이고, 그
    상태로 커밋됐다. **이름을 더하는 쪽은 이미 있는 블록을 안 본다**는 게 문제의 모양이라,
    사람 규율이 아니라 여기서 막는다.

    주입 목록은 `runner.js` 에서 읽는다 — 여기 베껴 두면 다음에 이름이 늘 때 이 검사만
    낡는다.
    """
    runner = (SITE / "assets" / "runner.js").read_text(encoding="utf-8")
    injected = set()
    for line in runner.splitlines():
        m = re.search(r'^\s*"const \{(.*)$', line)
        if m:
            injected |= {n.strip() for n in m.group(1).split(",") if n.strip()}
        m = re.search(r'^\s*"\s*([a-zA-Z, ]+)\} = borch;",', line)
        if m:
            injected |= {n.strip() for n in m.group(1).split(",") if n.strip()}
        m = re.search(r'^\s*"const ([a-zA-Z_$][\w$]*) = ', line)
        if m:
            injected.add(m.group(1))
    injected -= {"__pg"}
    assert len(injected) > 15, f"주입 목록을 못 읽었다 — {sorted(injected)}"

    clash = re.compile(r"^\s*(?:const|let|var|function|class)\s+([a-zA-Z_$][\w$]*)", re.M)
    wrong = []
    for page in _pages():
        text = page.read_text(encoding="utf-8")
        for m in re.finditer(r'<script type="text/plain"([^>]*)>(.*?)</script>', text, re.S):
            if 'data-lang="py"' in m.group(1):
                continue
            for hit in clash.finditer(m.group(2)):
                if hit.group(1) in injected:
                    wrong.append(f"{page.relative_to(ROOT)} — 블록이 `{hit.group(1)}` 를 다시 선언한다")
    assert not wrong, ("러너가 주입하는 이름을 블록이 덮어쓴다 (그 블록은 문법 오류로 안 돈다):\n  "
                       + "\n  ".join(sorted(set(wrong))))
