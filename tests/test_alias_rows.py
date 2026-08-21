"""**"별칭" 이라고 적힌 줄이 정말 별칭인가.**

`borch-ts/test/run.py` 의 갭 표는 안 옮긴 케이스를 접두어별로 묶고 까닭을 한 줄씩
적는다. 그중 `별칭` 은 *"옮기면 같은 질문을 두 번 한다"* 는 뜻이고, 그 말이 성립하려면
**그 이름이 borch.ts 에 다른 철자로 있어야 한다.** 없으면 두 번이 아니라 영이다.

이 검사가 있는 까닭: 그 주장이 **세 줄 연속으로 틀렸다.**

- `bit::` 24 — "비트 연산의 메서드 이름". 그 메서드 이름들이 저쪽에 없었다.
- `method2::` 60 — "`multiply`=`mul` 처럼 둘째 이름". 아홉은 이름이 아예 없었다.
- `top::` 50 — "최상위 제자리 함수". 떨구기 넷이 없어서 못 옮기던 것이었다.

셋 다 사람이 한 줄씩 까 보다가 나왔고, 세 번 연속이면 그것은 우연이 아니라 구조다.
까닭은 **주장**이고, 검사되지 않는 주장은 시간이 지나면 그냥 글자가 된다.

## 무엇을 재는가

케이스 이름에서 부르는 torch 이름을 뽑아, 선언된 borch.ts 표면과 대조한다. 철자는
`test_binding_fills_in.py` 와 같은 규칙으로 맞춘다 — **끝 밑줄은 남긴다**(제자리 판과
아닌 것은 다른 연산이다).

## 무엇을 안 재는가

`파이썬`·`없음`·`아직` 줄은 안 본다. 그 셋은 "저쪽에 있다" 고 주장하지 않는다.
그리고 이름을 못 뽑는 케이스(설명이 이름이 아닌 것들)는 **센 것에서 빼고 그 수를
말한다** — 조용히 넘기면 이 검사도 남의 주장을 검사 안 한 게 된다.
"""

import json
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent
RUNNER = ROOT / "borch-ts" / "test" / "run.py"
INDEX = ROOT / "site" / "assets" / "api-index.json"
GOLDEN = ROOT / "tests" / "golden.json"

# 이름을 못 뽑는 자리. **까닭과 함께 적는다** — 비면 이 검사가 조용해진다.
NOT_A_NAME = {
    "cache::": "전역 상수가 더럽혀졌는지를 묻는다 — 이름이 아니라 상태다",
    "grad::": "`vjp` 는 `backward(씨앗)` 이라 케이스 이름이 연산 이름이 아니다",
}


def _flat(name):
    """`test_binding_fills_in._flat` 과 같은 규칙. 끝 밑줄만 남긴다."""
    tail = "_" if name.endswith("_") else ""
    return name.replace("_", "").lower() + tail


def _alias_rows():
    """갭 표에서 `별칭` 이라 적힌 접두어들."""
    text = RUNNER.read_text(encoding="utf-8")
    rows = re.findall(r'^\s*"([a-z0-9]+::)":\s*\((\d+),\s*"([^"]*)"\)', text, re.M)
    # **머리글자로 본다.** 처음엔 `"별칭" in why` 였는데, `dtype::` 의 까닭에 든
    # "형 별칭" 이라는 다른 뜻의 낱말이 걸렸다 — 표시는 줄 맨 앞의 한 낱말이고
    # 본문에 같은 글자가 나오는 것과는 다르다.
    return {head for head, _, why in rows if why.startswith("별칭")}


def _declared():
    return {_flat(n.split(".")[-1])
            for n in json.loads(INDEX.read_text(encoding="utf-8"))}


# 케이스 제목에 **인자 이름**이 앞에 오는 자리들. 연산 이름이 아니므로 안 센다.
#
# 이 목록이 필요한 이유: 제목은 사람이 읽으라고 쓴 글이지 문법이 아니다. 뽑기가
# 완벽할 수 없고, **못 뽑는 것을 조용히 넘기면 이 검사도 남의 주장을 검사 안 한
# 게 된다** — 그래서 세지 않는 것은 여기 이름으로 적고 건너뛴 수를 화면에 낸다.
ARGUMENT_NAMES = {
    "bias_k", "is_causal", "key_padding_mask", "need_weights", "offsets",
    "per_sample_weights", "hard",
}


def _called_name(case):
    """케이스 이름에서 부르는 torch 이름. 못 뽑으면 `None`."""
    rest = case.split("::", 1)[1]
    # `제자리::foo_` 처럼 한 칸 더 들어간 자리는 마지막 칸이 이름이다.
    leaf = rest.split("::")[-1]
    hit = re.match(r"([A-Za-z_][A-Za-z_0-9]*)", leaf)
    if hit is None or hit.group(1) in ARGUMENT_NAMES:
        return None
    return hit.group(1)


def test_rows_calling_themselves_aliases_really_are():
    """`별칭` 줄의 이름은 **저쪽에 있어야 한다.** 없으면 별칭이 아니라 결손이다."""
    heads = _alias_rows()
    assert heads, "갭 표에서 `별칭` 줄을 하나도 못 찾았다 — 이 검사가 헛돌고 있다."

    declared = _declared()
    cases = json.loads(GOLDEN.read_text(encoding="utf-8"))["cases"]
    missing, unnamed = {}, 0
    for case in cases:
        head = case.split("::", 1)[0] + "::"
        if head not in heads or head in NOT_A_NAME:
            continue
        name = _called_name(case)
        if name is None:
            unnamed += 1
            continue
        if _flat(name) not in declared:
            missing.setdefault(head, set()).add(name)

    report = "\n".join(
        f"  {head} — {' '.join(sorted(names))}" for head, names in sorted(missing.items()))
    assert not missing, (
        "`별칭` 이라 적힌 줄인데 borch.ts 에 그 이름이 없다:\n" + report +
        "\n\n별칭은 *옮기면 같은 질문을 두 번 한다* 는 뜻이고, 이름이 없으면 두 번이\n"
        "아니라 영이다. borch.ts 에 넣고 케이스를 옮기거나, 까닭을 사실로 고쳐라.\n"
        f"(이름을 못 뽑아 건너뛴 케이스 {unnamed}건 — 그건 이 검사의 사각지대다.)")
