"""**이름에 인자를 달아 놓고 그 인자를 안 묻는 케이스**를 잡는다.

`grad::sum(dim)` 이라는 케이스가 있었다. 이름이 그렇게 붙어 있으니 아무도 다시 안
봤는데, `sum(dim=1).sum()` 과 `sum().sum()` 의 기울기는 **둘 다 전부 1** 이라 축을
통째로 무시해도 통과한다. 그 사이 `borch_webgpu` 는 `sum(dim=1)` 에 축을 무시한
스칼라를 내고 있었고, 792 건이 전부 초록이었다.

같은 꼴이 `unpool::분수::output_ratio` 에서 또 나왔다 — 이름은 비율인데 본문이 크기를
손으로 적고 있어서 비율→크기 규칙이 그 자리에 없었다. **이름만 보면 물은 것 같다.**

## 어떻게 묻는가 — 값으로

본문에 그 낱말이 있는지 세는 것으로는 안 된다. 위치 인자로 넘기면 낱말이 안 보이고,
서른다섯 자리가 거짓으로 걸렸다. 대신 **골든에 굳은 답끼리 견준다**: 이름에서 괄호를
뗀 짝이 표에 있고 **답이 한 바이트도 안 다르면**, 그 케이스는 괄호 안의 것을 안 묻는다.

실행이 필요 없다. 표만 읽으면 되고, 세 구현 어디에도 안 기댄다.

## 잡은 것

둘이 걸렸고 **둘 다 입력이 두 인자를 같게 만든 자리**였다.

`nn.Upsample(8)` 은 입력이 4×4 라 기본값 `scale_factor=2` 와 답이 같았다 — 이름은
"첫 자리는 size" 인데 첫 자리를 **배율로 읽는 구현도 통과했다.** 6 으로 물으면
갈린다(정수 배율로는 6 이 안 나온다).

`FractionalMaxPool2d(output_ratio=0.5)` 는 7×0.5 가 3.5 라 크기 3 이 되는데, 짝인
크기 케이스가 마침 `output_size=(3,3)` 이었다. 비율을 무시하고 크기 기본값을 쓰는
구현이 통과한다. 크기 쪽을 4 로 옮겼다.
"""

import json
import pathlib
import re

CASES = json.loads(
    (pathlib.Path(__file__).parent / "golden.json").read_text())["cases"]

# 맨 끝 괄호 한 덩이를 뗀다 — `foo(bar)` 의 짝은 `foo` 다.
_TAIL = re.compile(r"\s*[（(][^()（）]*[)）]\s*$")

# **일부러 같은 자리.** 같다는 것이 답인 케이스와, 괄호 안이 곧 기본값인 케이스다.
# 새로 늘어나면 이 검사가 빨개지므로, 늘릴 때는 왜인지 여기 적게 된다.
DELIBERATE = {
    # 같다는 것이 곧 답이다
    "act::nn.Identity(인자를 삼킨다)":
        "torch 의 Identity 는 아무 인자나 받아 버린다 — 자리 채우개라 쓰는 쪽이 "
        "층 이름만 바꾸고 인자는 그대로 두기 때문이다.",
    "unpool::층::repr::CTCLoss(인자 있음)":
        "torch 의 repr 이 인자를 안 적는다. 적는 줄 알고 굳히면 갈린다.",
    "math::grad::trunc(0이어야)": "버림의 도함수는 어디서나 0 이다.",
    "math::grad::fix(0이어야)": "`trunc` 와 같은 함수다.",
    # 괄호 안이 곧 기본값이다 — 기본값을 손으로 줘도 같다는 것을 못 박는 자리
    "index::searchsorted(side=left)": "`left` 가 기본값이다.",
    "fname::제자리::hardtanh_(-1,1)": "(-1, 1) 이 기본값이다.",
    "linalg::name2::eigvalsh(아래삼각만)": "`UPLO='L'` 이 기본값이다.",
    "shape::expand(-1)": "-1 은 그 축을 그대로 두라는 뜻이라 답이 같다.",
    # 기울기로는 못 갈리는 자리 — **값 케이스가 따로 있다**
    "grad::sum(dim)":
        "합의 기울기는 축과 무관하게 전부 1 이다. 축은 `arg::sum(dim)` 이 값으로 묻는다.",
    "grad::sort(내림차순)":
        "정렬의 기울기는 차례와 무관하게 전부 1 이다. 차례는 값 케이스가 묻는다.",
    "norm::grad::F.conv_transpose2d(편향)":
        "입력에 대한 기울기는 편향과 무관하다. 편향은 값 케이스가 묻는다.",
}


def identical_pairs():
    """이름에서 괄호를 뗀 짝이 있고 **답이 같은** 케이스."""
    same = []
    for name, value in CASES.items():
        base = _TAIL.sub("", name).strip()
        if base == name or base not in CASES:
            continue
        if json.dumps(value, sort_keys=True) == json.dumps(CASES[base], sort_keys=True):
            same.append(name)
    return same


def test_a_case_named_after_an_argument_asks_about_it():
    """괄호에 인자를 달았으면 **기본값과 다른 답**이 나와야 한다.

    다르지 않다면 그 케이스는 이름이 말하는 것을 안 묻는다 — 그리고 이름 때문에
    아무도 다시 안 본다. 일부러 같은 자리는 위에 이유와 함께 적어 둔다.
    """
    surprise = [n for n in identical_pairs() if n not in DELIBERATE]
    assert not surprise, (
        "이름에 인자를 달았는데 기본값 케이스와 답이 같다:\n  "
        + "\n  ".join(surprise)
        + "\n\n인자가 답을 바꾸는 입력으로 바꾸거나, 같은 것이 답이면 "
          "`DELIBERATE` 에 이유와 함께 적어라."
    )


# **기울기가 통째로 0 이면 그 케이스는 아무것도 안 묻는다** — 기울기를 아예 안
# 흘리는 구현도 통과한다. 아래는 도함수가 진짜 0 인 자리들이고, 그 밖의 것이 0 이 되면
# 대개 **케이스 배선**이 잘못된 것이다.
#
# 실제로 그렇게 걸렸다. `edge::` 의 접기 헬퍼가 자리마다 다른 가중치를 주려고
# `arange` 를 곱했는데 첫 몫이 0 이라, **출력이 한 칸인 케이스는 기울기가 통째로
# 0** 이 됐다. 균일 접기를 피하려고 넣은 장치가 그 케이스를 아무것도 안 묻는 상태로
# 만든 것이다 — `max(동점)` 이 `[0,1,0,0]` 대신 `[0,0,0,0]` 을 굳히고 있었다.
ZERO_ON_PURPOSE = {
    "grad::접힘::angle() 은 0 을 흘린다": "실수의 편각은 계단이라 도함수가 0 이다.",
    "math::grad::trunc": "버림은 계단이다.",
    "math::grad::fix": "`trunc` 와 같은 함수다.",
    "math::grad::copysign/b": "부호를 주는 쪽으로는 기울기가 안 간다.",
    "blend::grad::addmm(beta=0)": "`beta=0` 이면 더해지는 항이 빠진다.",
    "edge::grad::sign(0포함)": "부호 함수의 도함수는 어디서나 0 이다.",
}


def test_no_gradient_case_is_all_zero_by_accident():
    """0 인 기울기와 **안 흐르는 기울기**는 다른 말이다.

    이름에 `(0이어야)` 가 붙은 것은 그 자체로 답을 적어 둔 것이라 지나간다.
    """
    zero = []
    for name, case in CASES.items():
        if "grad::" not in name or "(0이어야)" in name:
            continue
        vals = [v for v in (case.get("values") or []) if isinstance(v, (int, float))]
        if len(vals) >= 2 and set(vals) == {0.0} and name not in ZERO_ON_PURPOSE:
            zero.append(name)
    assert not zero, (
        "기울기가 통째로 0 인 케이스: " + ", ".join(zero) + "\n"
        "도함수가 진짜 0 이면 `ZERO_ON_PURPOSE` 에 이유와 함께 적고, 아니면 "
        "케이스 배선을 보라 — 접는 가중치에 0 이 섞이면 이렇게 된다."
    )


def test_the_deliberate_list_does_not_rot():
    """**적어 둔 것이 실제로 같은지도 본다.**

    케이스를 고쳐 답이 갈리게 만들면 위 목록의 그 줄은 거짓이 된다. 남은 목록이
    이유 없이 길어지는 것을 막는 자리다 — 이 저장소에서 `KNOWN_ABSENT` 가 같은 일을
    한다.
    """
    same = set(identical_pairs())
    stale = sorted(n for n in DELIBERATE if n not in same)
    assert not stale, (
        f"이제는 답이 갈리는데 `DELIBERATE` 에 남아 있다: {stale}\n"
        "그 줄을 지워라 — 목록이 길어지면 아무도 안 읽는다."
    )
