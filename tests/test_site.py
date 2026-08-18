"""사이트가 저장소와 어긋나지 않았는가.

`tests/test_docs.py` 가 문서의 **수**를 붙잡는 것과 같은 자리다. 이쪽이 붙잡는 것은
**생성물**이다 — `site/assets/api.json` 은 `site/build_api.py` 가 선언 파일에서 뽑은
것이고, 소스가 자란 뒤 다시 안 뽑으면 사이트가 없는 API 를 보여주거나 새 API 를 빠뜨린다.

그 어긋남은 **화면에서 안 보인다.** 목록이 조금 짧은 것과 원래 그만큼인 것이 같은
모양이라, 사람이 눈으로 찾는 방식은 여기서 특히 안 듣는다(생성기를 쓰다가 실제로 겪었다 —
파서가 텐서 메서드 422 개 중 18 개만 물고 있었는데 화면은 멀쩡해 보였다).
"""

import json
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
GENERATOR = ROOT / "site" / "build_api.py"
API = ROOT / "site" / "assets" / "api.json"
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

    before = API.read_text(encoding="utf-8")
    proc = subprocess.run([sys.executable, str(GENERATOR)], capture_output=True, text=True)
    assert proc.returncode == 0, f"생성기가 멈췄다:\n{proc.stderr}"
    after = API.read_text(encoding="utf-8")

    if before != after:
        # 되돌려 놓는다 — 검사가 작업 트리를 고쳐 놓고 가면 안 된다.
        API.write_text(before, encoding="utf-8")
        old, new = json.loads(before), json.loads(after)
        pytest.fail(
            "API 레퍼런스가 소스보다 낡았다 — 항목 "
            f"{old['total']} → {new['total']}.\n"
            "  다시 뽑아라: python3 site/build_api.py\n"
            "  (설명문을 고칠 곳은 이 파일이 아니라 소스의 주석이다.)")


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
