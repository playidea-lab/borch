"""여러 값이 **한 칸으로 접힐 때** 기울기가 어디로 돌아가는가.

`max()` 가 [3,5,5,1,5] 를 5 로 접으면, 그 5 하나를 만든 것은 세 자리다. 되돌릴 때
셋 중 어디로 가는가 — 이 물음은 **동점이 있는 자료로만 열린다.** 값이 전부 다르면
"고른 자리 하나로 준다" 와 "값이 같은 칸에 나눈다" 가 같은 답을 내므로, 절반만
맞는 구현이 통과한다.

## torch 에게 물어 뽑은 규칙

    번호를 **건네는** 연산은 고른 자리 하나로 준다   (max(dim=0)·mode·nanmedian(dim))
    번호를 **안 건네는** 연산은 값이 같은 칸에 나눈다 (max()·min()·median()·nanmedian())
    **정렬 자리**로 접는 연산은 그 자리들로 나눈다    (quantile — 보간이면 둘)

`median()` 과 `quantile(0.5)` 가 갈리는 것이 이 표의 핵심이다. [1,5,5,5] 에서 median 은
세 5 에 ⅓ 씩 주고 quantile 은 **앞의 두 5 에 ½ 씩** 준다. 같은 값을 내는 두 연산이
서로 다른 것을 떠받치고 있다고 보는 셈이다.

## 이 검사가 잡은 것

일곱 자리가 갈렸고, 갈래는 둘이었다.

**넷은 그래프가 아예 끊겨 있었다** — `max()`·`min()`·`mode()`·`quantile()`·`nanmedian()`
이 `Tensor(...)` 를 맨손으로 만들어 부모를 안 달았다. 값 검사는 전부 통과한다.
값은 맞기 때문이다. 드러나는 것은 `backward()` 를 불렀을 때이고, 그때 나오는 말은
"requires_grad 가 아닌 텐서" 라 **사용자를 가리킨다** — 이 연산에 역방향이 없다고는
아무도 안 말해 준다.

**셋은 이어져 있는데 규칙이 틀렸다** — `median()` 은 뽑은 자리 하나로만 줬다. 코드에
근거까지 적혀 있었다("나머지 원소를 흔들어도 답이 안 움직인다"). 동점이 아닐 때만
맞는 말이다. `angle`·`i0` 은 계단·베셀이라 각각 0 과 `i1` 이 답인데 둘 다 없었다.

## 왜 골든이 아니라 여기인가

골든은 셋을 함께 묻는 자리인데 borch.ts 는 `median` 의 동점 규칙이 아직 코어가
틀렸던 것과 같고 `mode`·`quantile`·`nanmedian`·`i0` 의 역방향이 없다. 케이스만 먼저
넣으면 결속 러너가 빨간 채로 남는다. **코어 쪽 절반은 지금 맞출 수 있으므로** 그
절반을 여기서 못 박고, borch.ts 가 따라오면 같은 표를 골든으로 옮긴다
(`tests/test_reduce_dtype.py` 가 같은 이유로 같은 자리에 있다).

셋 다 통과하는 자리(`max`·`min`·`norm(inf)`·`angle`)는 골든에 넣었다 —
`grad::접힘::*`.
"""

import numpy as np
import pytest
import torch

import borch

# 동점이 있는 자료. **없으면 이 파일 전체가 아무것도 안 묻는다.**
TIE = [3.0, 5.0, 5.0, 1.0, 5.0]         # 최대가 셋
EVEN = [1.0, 5.0, 5.0, 5.0]             # 짝수 개 · 중앙값 자리에 동점
DUP = [1.0, 1.0, 2.0, 2.0, 2.0]         # 최빈값이 셋
NAN = [3.0, float("nan"), 5.0, 1.0, 5.0]
STEP = [0.5, -1.0, 2.0]

CASES = [
    # 번호를 안 건네는 축약 — 값이 같은 칸에 고르게 나눈다
    ("max()", TIE, lambda t: t.max()),
    ("min()", TIE, lambda t: t.min()),
    ("amax()", TIE, lambda t: t.amax()),
    ("median()", TIE, lambda t: t.median()),
    ("median() 짝수", EVEN, lambda t: t.median()),
    ("nanmedian()", NAN, lambda t: t.nanmedian()),
    ("nanmedian() 동점", EVEN, lambda t: t.nanmedian()),
    # 번호를 건네는 축약 — 고른 자리 하나로
    ("max(dim=0)", TIE, lambda t: t.max(dim=0)),
    ("median(dim=0)", TIE, lambda t: t.median(dim=0)),
    ("nanmedian(dim=0)", NAN, lambda t: t.nanmedian(0)),
    ("mode()", DUP, lambda t: t.mode()),
    ("kthvalue(2)", TIE, lambda t: t.kthvalue(2)),
    # 정렬 자리로 접는 것 — 보간이면 둘로 나뉜다
    ("quantile(0.5)", TIE, lambda t: t.quantile(0.5)),
    ("quantile(0.3)", TIE, lambda t: t.quantile(0.3)),
    ("quantile(0.5) 짝수", EVEN, lambda t: t.quantile(0.5)),
    ("quantile(0.75)", EVEN, lambda t: t.quantile(0.75)),
    # 노름 — `p` 가 규칙을 통째로 바꾼다
    ("norm(inf)", TIE, lambda t: t.norm(float("inf"))),
    ("norm(-inf)", TIE, lambda t: t.norm(float("-inf"))),
    ("norm(3)", TIE, lambda t: t.norm(3)),
    # 미분 불가능한 점 위 · 특수함수
    ("angle()", STEP, lambda t: t.angle()),
    ("i0()", STEP, lambda t: t.i0()),
    ("i0() 큰 값", [8.0, -12.0, 0.0], lambda t: t.i0()),
    # 접히지 않는 이웃 — 규칙이 다르다는 것을 같은 자료로 보인다
    ("topk(3)", TIE, lambda t: t.topk(3)),
    ("sort()", TIE, lambda t: t.sort()),
    ("cummax(0)", TIE, lambda t: t.cummax(0)),
]


def _first(out):
    """값·번호를 같이 내는 것에서 **값**을 꺼낸다.

    **튜플인지 먼저 본다.** 평범한 텐서도 `.values`·`.indices` 를 갖는다(희소용) —
    그것으로 가리면 텐서의 첫 원소를 집는다. 이 저장소에서 세 번째 밟는 함정이라
    `test_reduce_dtype.py` 의 `name_of` 와 같은 순서로 적어 둔다.
    """
    if isinstance(out, tuple) or hasattr(out, "_fields"):
        return out[0]
    if not hasattr(out, "shape") and hasattr(out, "values"):
        return out.values
    return out


def gradient_of(lib, values, fn):
    """`fn` 을 되짚어 얻은 기울기. **멈추는 것도 답이다** — 문자열로 돌려준다."""
    t = lib.tensor(values, requires_grad=True)
    try:
        out = _first(fn(t))
        (out.sum() if out.numel() > 1 else out).backward()
    except Exception as exc:                                    # noqa: BLE001
        return f"거절({type(exc).__name__})"
    if t.grad is None:
        return "기울기 없음"
    return np.asarray(t.grad.reshape(-1).tolist(), dtype=np.float64)


@pytest.mark.parametrize("name,values,fn", CASES, ids=[c[0] for c in CASES])
def test_folding_gradient_matches_torch(name, values, fn):
    want = gradient_of(torch, values, fn)
    got = gradient_of(borch, values, fn)
    if isinstance(want, str) or isinstance(got, str):
        assert got == want, f"{name}: torch 는 {want} 인데 코어는 {got} 다."
        return
    assert np.allclose(got, want, atol=1e-5, rtol=1e-5), (
        f"{name}: torch 는 {want.tolist()} 인데 코어는 {got.tolist()} 다.\n"
        "접히는 자리의 규칙은 셋이다 — 번호를 건네면 그 자리 하나로, 안 건네면 값이\n"
        "같은 칸에 고르게, 정렬 자리로 접으면 그 자리들로 나눈다."
    )


def test_median_and_quantile_disagree_on_ties():
    """**같은 값을 내면서 다른 것을 떠받치는 두 연산.** 이 한 줄이 표를 설명한다.

    [1,5,5,5] 의 중앙값도 0.5 분위수도 5 다. 그런데 median 은 5 인 칸 **셋 전부**에
    ⅓ 씩 주고, quantile 은 정렬해서 쓴 **두 자리**에 ½ 씩 준다. 값만 재면 둘이 같아
    보이므로, 이 갈림은 되짚어야만 드러난다.
    """
    def grad(fn):
        t = borch.tensor(EVEN, requires_grad=True)
        fn(t).backward()
        return [round(v, 4) for v in t.grad.tolist()]

    assert grad(lambda t: t.median()) == [0.0, 0.3333, 0.3333, 0.3333]
    assert grad(lambda t: t.quantile(0.5)) == [0.0, 0.5, 0.5, 0.0]


def test_a_folding_op_that_does_not_carry_grad_is_a_defect_not_a_choice():
    """접히는 연산은 **하나도 빠짐없이** 기울기를 나른다.

    끊긴 그래프는 값 검사로 안 잡힌다 — 값은 맞기 때문이다. 그래서 여기서 이름을
    하나씩 부른다. 새 축약을 넣는 사람이 `Tensor(...)` 로 결과를 만들면 이 검사가
    **그 이름을 대며** 빨개진다.
    """
    stuck = []
    for name, values, fn in CASES:
        t = borch.tensor(values, requires_grad=True)
        try:
            out = _first(fn(t))
        except Exception:                                       # noqa: BLE001
            continue
        if out.dtype is borch.float32 and not out.requires_grad:
            stuck.append(name)
    assert not stuck, (
        f"기울기를 안 나르는 연산: {stuck}\n"
        "`Tensor(...)` 를 맨손으로 만들면 부모가 안 달려 그래프가 조용히 끊긴다.\n"
        "`t._make(값, (t,), 역방향)` 을 쓴다."
    )
