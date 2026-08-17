"""문서에 적힌 **개수**가 실제와 같은지 본다.

골든이 늘 때마다 문서의 수가 낡는다. 이 저장소는 그것을 이미 세 번 잡았고
(`b00e693` 골든 수, `b3d7453` 세 수치, `e41c043` 그중 하나는 설치를 깨뜨렸다)
세 번 다 사람이 눈으로 찾아 고쳤다. **세 번 실패한 방식은 방식이 문제다.**

여기서 묻는 것은 정확히 하나다: **README 가** 말하는 케이스 수가 표가 실제로 담는
수와 같은가.

**설계 문서는 안 본다.** 처음에 전부 보게 했더니 열 곳이 걸렸는데, 그중 일곱이
낡은 것이 아니라 **그때의 기록**이었다 — `BORCH-TS.md` 의 "relu 가 골든 798 건을
그대로 통과했다" 는 798 이 맞다. 845 로 고치면 낡은 수를 고치는 것이 아니라 역사를
위조하는 것이고, 그건 낡은 수보다 나쁘다. `WEBGPU-DESIGN.md` 의 "골든 141/141" 도
S3 단계가 그때 도달한 지점이다.

그래서 경계를 **문서의 종류**로 긋는다. README 는 지금을 말하는 자리이므로 늘
현재여야 하고, 설계·이력 문서는 그때를 말하는 자리이므로 손대면 안 된다. 시제를
정규식으로 가르려던 첫 시도는 그 구분을 못 했다.
"""

import pathlib
import re

import cases as cases_mod

ROOT = pathlib.Path(__file__).resolve().parent.parent
# 케이스 수를 말하는 자리들. `골든 845건` · `골든 1020/1020` 같은 모양을 찾는다.
#
# **세 자리로 못 박아 두었다가 표가 1000 을 넘는 순간 조용해졌다.** 아무것도 안
# 잡는 검사는 통과하는 검사처럼 보이고, 그것이 낡은 수보다 나쁘다 — 낡은 수는
# 언젠가 눈에 띄지만 안 도는 검사는 안 띈다. 자릿수를 넉넉히 잡는다.
COUNT = re.compile(r"골든\s*\*{0,2}(\d{3,5})\s*(?:건|/\s*\d{3,5})")


def _counts():
    """(전체, 코어가 보는 수, 결속이 보는 수).

    **셋이 된 것은 범위가 양쪽으로 갈렸기 때문이다.** 자매 전용(1·3 차원 합성곱처럼
    코어가 일부러 거절하는 것)은 코어가 건너뛰고, 코어 전용(복소수)은 결속이
    건너뛴다. 둘 중 하나만 세면 나머지 절반이 "구현이 빠졌다" 로 보인다.
    """
    names = [n for n, _ in cases_mod.golden_cases(cases_mod.golden_inputs())]
    core = [n for n in names if not n.startswith(cases_mod.WEBGPU_PREFIX)]
    bind = [n for n in names if not n.startswith(cases_mod.CORE_ONLY_PREFIXES)]
    return len(names), len(core), len(bind)


def test_docs_do_not_name_a_stale_golden_count():
    """문서가 대는 케이스 수는 **지금 있는 수**여야 한다.

    실제로 쓰이는 수는 셋이다 — 표 전체, 코어가 자매 전용을 뺀 수, 결속이 코어
    전용을 뺀 수. 그 셋 중 아무것도 아닌 수가 `골든 N건` 자리에 있으면 낡은 것이다.

    **셋을 다 받으면 그물이 성겨진다.** 실제로 복소수를 넣던 날 전체가 2263 에서
    2287 로 늘었는데 결속이 보는 수가 정확히 2263 이 되어, 낡은 문장 하나가 우연히
    맞는 수를 들고 통과할 뻔했다. 수가 맞아도 그 문장은 "전부를 지난다" 였고 그건
    더 이상 사실이 아니었다 — 수를 세는 검사는 문장을 안 읽는다는 것을 적어 둔다.
    """
    total, core, bind = _counts()
    allowed = {str(total), str(core), str(bind)}
    stale = []
    path = ROOT / "README.md"
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        for hit in COUNT.findall(line):
            if hit not in allowed:
                stale.append(
                    f"README.md:{i}  '{hit}' — 지금은 {total}(전체) / "
                    f"{core}(코어) / {bind}(결속)")
    assert not stale, (
        "README 의 골든 수가 낡았다:\n  " + "\n  ".join(stale) +
        "\n\nREADME 는 지금을 말하는 자리다. 그때를 이야기해야 하는 문장이면 수를 "
        "빼고 쓰거나 설계 문서로 옮겨라 — 지난 수를 현재 수로 고치는 것은 낡은 수를 "
        "고치는 것이 아니라 역사를 위조하는 것이다.")


# `2,358 줄` 처럼 패키지 크기를 말하는 자리. 천 단위 쉼표를 쓰든 안 쓰든 잡는다.
LINES = re.compile(r"\*{0,2}(\d{1,3}(?:,\d{3})*)\s*줄\*{0,2}")
# 세는 대상. 지금 저장소에 있는 패키지만 — 지워진 것의 크기는 이력의 몫이다.
PACKAGES = {"borch", "borch_webgpu"}


def _package_lines():
    """패키지 이름 → 줄 수."""
    out = {}
    for name in PACKAGES:
        total = 0
        for path in sorted((ROOT / name).glob("*.py")):
            total += len(path.read_text(encoding="utf-8").splitlines())
        out[name] = total
    return out


# **이력의 수.** 지금 세어서 확인할 수 없는 것들이라 여기 적어 둔다. 각각 무엇인지
# 안 적으면 다음 사람에게는 그냥 마법 숫자가 된다.
HISTORICAL = {
    "5,307": "45be321 에서 지운 TF.js 판 borch_webgpu",
    "3,300": "borch 를 파일 하나에서 패키지로 쪼갠 시점의 크기 (8177e1d)",
}

# **어긋남을 얼마까지 봐주는가.** 잡으려는 것은 2.6 배 오차이지 세 줄이 아니다.
# 정확한 수를 요구하면 `_ops.py` 한 줄을 고칠 때마다 문서가 깨지고, 그러면 이 검사가
# 지켜야 할 것을 지키는 대신 통과시키는 법을 배우게 만든다.
TOLERANCE = 0.05


def test_docs_do_not_name_a_stale_line_count():
    """문서가 대는 **줄 수**가 실제와 크게 어긋나지 않아야 한다.

    골든 수에 이 장치를 달면서 줄 수에는 안 달았고, 바로 그 자리에서 두 번 틀렸다.
    한 번은 여덟 파일 중 다섯만 세어 5,307 을 **2,312** 라고 적었고, 한 번은
    `_data.py` 를 들여오기 전의 추정 **900** 을 다시 안 세고 옮겼다. 그 둘이 나란히
    놓여 "2,312 줄이 900 줄이 되었다" 는 문장을 만들었는데, 실제로는 5,307 이
    2,361 이 된 것이다 — 방향은 맞고 크기가 2.6 배 틀렸다.

    둘 다 커밋 메시지와 README 에 동시에 들어갔다. **눈으로 세는 것은 방식이 아니다.**

    경계는 골든 수와 같다. README 와 패키지 docstring 은 지금을 말하는 자리라
    현재여야 하고, 설계 문서(`BORCH-TS.md`·`WEBGPU-DESIGN.md`)는 그때를 적은 자리라
    안 본다.
    """
    sizes = _package_lines()
    stale = []
    for rel in ("README.md", "borch_webgpu/__init__.py", "borch/__init__.py"):
        path = ROOT / rel
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            for hit in LINES.findall(line):
                if hit in HISTORICAL:
                    continue
                said = int(hit.replace(",", ""))
                # 문장에 섞이는 작은 수까지 잡으면 노이즈가 된다.
                if said < 100:
                    continue
                if any(abs(said - real) <= real * TOLERANCE for real in sizes.values()):
                    continue
                stale.append(f"{rel}:{i}  '{hit} 줄' — 지금은 " +
                             ", ".join(f"{k} {v:,}" for k, v in sorted(sizes.items())))
    assert not stale, (
        "문서의 줄 수가 실제와 어긋난다:\n  " + "\n  ".join(stale) +
        "\n\n세어서 고쳐라. 그때를 말하는 수라면 `HISTORICAL` 에 무엇인지와 함께 "
        "적어라 — 이력을 현재 수로 고치는 것은 낡은 수를 고치는 것이 아니다.")
