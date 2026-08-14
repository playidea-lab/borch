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
# 케이스 수를 말하는 자리들. `골든 845건` · `골든 845/845` 같은 모양을 찾는다.
COUNT = re.compile(r"골든\s*\*{0,2}(\d{3})\s*(?:건|/\s*\d{3})")


def _counts():
    """(전체, 코어가 보는 수). 코어는 자매 전용 케이스를 건너뛴다."""
    names = [n for n, _ in cases_mod.golden_cases(cases_mod.golden_inputs())]
    core = [n for n in names if not n.startswith(cases_mod.WEBGPU_PREFIX)]
    return len(names), len(core)


def test_docs_do_not_name_a_stale_golden_count():
    """문서가 대는 케이스 수는 **지금 있는 수**여야 한다.

    실제로 쓰이는 수는 둘이다 — 표 전체(브라우저 구현이 보는 것)와, 코어가 자매
    전용을 뺀 수. 그 둘 중 아무것도 아닌 세 자리 수가 `골든 N건` 자리에 있으면
    그것은 낡은 것이다.
    """
    total, core = _counts()
    allowed = {str(total), str(core)}
    stale = []
    path = ROOT / "README.md"
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        for hit in COUNT.findall(line):
            if hit not in allowed:
                stale.append(f"README.md:{i}  '{hit}' — 지금은 {total} 또는 {core}")
    assert not stale, (
        "README 의 골든 수가 낡았다:\n  " + "\n  ".join(stale) +
        "\n\nREADME 는 지금을 말하는 자리다. 그때를 이야기해야 하는 문장이면 수를 "
        "빼고 쓰거나 설계 문서로 옮겨라 — 지난 수를 현재 수로 고치는 것은 낡은 수를 "
        "고치는 것이 아니라 역사를 위조하는 것이다.")
