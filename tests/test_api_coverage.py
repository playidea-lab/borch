"""레퍼런스 생성기가 **소스를 다 봤는가.**

`tests/test_site.py::test_api_reference_is_not_stale` 은 "뽑아 둔 목록이 지금 선언
파일과 같은가" 를 묻는다. 그 검사가 못 묻는 것이 하나 있다 — **어느 선언 파일을
볼지는 생성기가 스스로 정한다.** 목록에서 빠진 파일은 뽑히지도 않고 대조되지도
않으므로, 그 안의 이름이 통째로 없어도 두 쪽 다 초록이다.

그 자리가 실제로 비어 있었다. `index.ts` 가 `MODULES` 에 없어서 거기서만 선언되는
`isTensor` 와 `setNull` 이 레퍼런스에도 이름 색인에도 없었고, 결속이 쓰는 이름인데도
아무 검사가 안 울었다. 다른 세션이 **밖에서** 봤다 — 그쪽이 borch.ts 표면 목록으로
이 색인을 읽고 있었고, 있어야 할 이름이 없어서 드러났다.

이 파일은 그 바깥 눈을 붙박아 두는 일만 한다. `test_site.py` 에 안 넣은 것은 그
파일을 지금 다른 세션이 고치고 있어서다 — 검사가 부딪혀 한쪽이 지워지는 것보다
파일 하나가 느는 편이 낫다.
"""

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
INDEX_TS = ROOT / "borch-ts" / "src" / "index.ts"


def _module_list():
    """생성기가 훑기로 한 모듈 이름들."""
    sys.path.insert(0, str(ROOT / "site"))
    try:
        import build_api
    finally:
        sys.path.pop(0)
    return {name for name, _, _ in build_api.MODULES}


def test_generator_reads_every_module_index_exports():
    """`index.ts` 가 내보내는 모듈을 생성기가 **다 훑어야** 한다.

    **`index.ts` 를 기준으로 삼는 이유**: 그 파일이 곧 공개 표면의 정의다. 안쪽
    사정(`kernels`·`repr`)은 거기서 안 나가므로 저절로 빠지고, 목록이 그보다 넓은
    것은 괜찮다 — 좁은 것만 문제다.
    """
    # **`index` 자신을 손으로 넣는다.** 그 파일은 자기에게서 재수출하지 않으므로
    # 재수출만 모으면 목록에 절대 안 나온다. 처음 그렇게 짰고, 검사에 이빨이 있는지
    # 재려고 `MODULES` 에서 `index` 를 빼 봤더니 **통과했다** — 이 검사가 생긴 까닭이
    # 바로 그 한 줄인데 그 한 줄만 못 보고 있었다. 재 보지 않으면 이렇게 된다.
    exported = {"index"} | set(
        re.findall(r'from\s+"\./([A-Za-z_]\w*)\.js"',
                   INDEX_TS.read_text(encoding="utf-8")))
    missing = sorted(exported - _module_list())
    assert not missing, (
        f"`index.ts` 가 내보내는데 생성기가 안 훑는 모듈이 있다: {missing}\n"
        "  site/build_api.py 의 MODULES 에 넣어라 — 안 넣으면 그 파일에서만\n"
        "  선언되는 이름이 레퍼런스와 이름 색인에서 조용히 빠지고, 목록이\n"
        "  선언 파일과 같은지 묻는 검사는 그것을 못 본다.")
