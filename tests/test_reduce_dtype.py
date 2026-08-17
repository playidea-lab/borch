"""축약이 형을 어떻게 바꾸는가 — **코어가 torch 와 같은 자리에서 같은 답을 내는가.**

`bool.sum()` 이 int64 라는 것 하나만 알려져 있었고, 그래서 모양·색인 연산에 쓴
"보존" 도 값 연산에 쓴 "승격" 도 아닌 셋째 규칙이 있다는 것까지만 확인됐다. torch 에게
물어 표를 뽑아 보니 규칙이 **넷**이었고, 가르는 선은 낯익은 것이었다:

    누적은 값을 **만들고**(참·거짓 칸에 3 이 안 들어간다),
    고르기는 있던 값을 **건넨다.**

모양·색인 연산의 dtype 이름표를 고칠 때 그은 선과 같은 선이다. 축약이 예외였던 것이
아니라 **누적과 고르기가 서로 다른 것**이었고, `mean` 이 거절인 것도 거기서 나온다 —
나눗셈은 정수 칸에 답이 없다.

## 왜 골든이 아니라 여기인가

골든은 세 구현을 함께 묻는 자리인데 borch.ts 는 축약이 전부 `Tensor.make` 를 dtype
없이 불러서 지금 float32 를 낸다. 케이스만 먼저 넣으면 결속 러너가 빨간 채로 남는다.
**코어 쪽 절반은 지금 맞출 수 있으므로** 그 절반을 여기서 못 박고, borch.ts 를 고칠 때
같은 표를 골든으로 옮긴다.

## 이 검사가 잡은 것

여섯 자리에서 코어가 torch 와 갈렸다. dtype 규칙이 아니라 **거절 여부**였다 —
`mean`·`median(bool)`·`argmax(bool)` 에서 torch 는 멈추는데 코어는 numpy 를 타고 값을
냈고, `logsumexp` 는 numpy 의 기본값 float64 를 흘렸다. "torch 가 멈추는 자리에서
근사한다" 는 이 저장소의 첫 줄이 거절하는 종류다.
"""

import numpy as np
import pytest
import torch

import borch

# 축약이라고 부를 만한 것 전부. 값이 아니라 **나오는 형**만 본다.
OPS = [
    ("sum", lambda t: t.sum()),
    ("prod", lambda t: t.prod()),
    ("cumsum", lambda t: t.cumsum(0)),
    ("cumprod", lambda t: t.cumprod(0)),
    ("amax", lambda t: t.amax()),
    ("amin", lambda t: t.amin()),
    ("max", lambda t: t.max()),
    ("min", lambda t: t.min()),
    ("mean", lambda t: t.mean()),
    ("median", lambda t: t.median()),
    ("argmax", lambda t: t.argmax()),
    ("argmin", lambda t: t.argmin()),
    ("count_nonzero", lambda t: t.count_nonzero()),
    ("any", lambda t: t.any()),
    ("all", lambda t: t.all()),
    ("logsumexp", lambda t: t.logsumexp(0)),
    ("var", lambda t: t.var()),
    ("std", lambda t: t.std()),
    ("norm", lambda t: t.norm()),
]

# **둘 다 물어야 한다.** int64 만 물으면 "형을 지킨다" 와 "bool 을 올린다" 가 같아
# 보이고, 절반만 맞는 구현이 통과한다.
KINDS = [
    ("int64", [3, 1, 4, 1, 5], True),
    ("bool", [True, False, True], False),
]


def answer(lib, values, as_int, fn):
    """형 이름, 또는 거절했으면 `"거절"`. **거절도 답이다.**"""
    try:
        t = lib.tensor(values, dtype=lib.int64) if as_int else lib.tensor(values)
        out = fn(t)
    except Exception:                                           # noqa: BLE001
        return "거절"
    return str(out.dtype).replace("torch.", "") if hasattr(out, "dtype") \
        else type(out).__name__


@pytest.mark.parametrize("op,fn", OPS, ids=[n for n, _ in OPS])
@pytest.mark.parametrize("kind,values,as_int", KINDS, ids=[k for k, _, _ in KINDS])
def test_reduction_dtype_matches_torch(op, fn, kind, values, as_int):
    expected = answer(torch, values, as_int, fn)
    got = answer(borch, values, as_int, fn)
    assert got == expected, (
        f"{op}({kind}): torch 는 {expected} 인데 코어는 {got} 다.\n"
        "축약의 형 규칙은 넷이다 — 누적(sum·prod·cumsum·cumprod)은 bool 을 int64 로\n"
        "올리고, 고르기(amax·amin·max·min)는 형을 지키고, argmax·count_nonzero 류는\n"
        "int64 로 고정이고, mean·var·std·norm 은 정수·참거짓을 거절한다."
    )


@pytest.mark.xfail(strict=True, reason="코어의 축약에 dtype= 인자가 아직 없다")
def test_dtype_argument_beats_the_rule():
    """`dtype=` 를 주면 그것이 전부를 이긴다 — 규칙보다 위다.

    **아직 없다.** 표를 뽑을 때 같이 나온 자리인데, 이것은 규칙을 어긴 것이 아니라
    **기능이 없는 것**이라 성격이 다르다 — 위의 여섯 자리는 "torch 가 멈추는데 우리가
    값을 낸다" 였고 이쪽은 "torch 가 받는 인자를 우리가 안 받는다" 다.

    `strict=True` 로 둔다. 누군가 `dtype=` 를 넣으면 **이 검사가 빨개져서** 표를 같이
    고치라고 말한다 — 조용히 통과하면 없던 것이 생긴 줄 아무도 모른다.
    """
    for values in ([3, 1, 4], [True, False, True]):
        a = torch.tensor(values).sum(dtype=torch.float32).dtype
        b = borch.tensor(values).sum(dtype=borch.float32).dtype
        assert str(a) == str(b) == "torch.float32"


def test_the_line_is_accumulate_versus_select():
    """규칙을 문장으로 못 박는다 — 표가 흔들리면 여기가 먼저 깨진다.

    누적은 값을 만들어 참·거짓 칸을 벗어나고, 고르기는 있던 값을 건네므로 안 벗어난다.
    이 두 줄이 나머지를 전부 설명한다.
    """
    flags = borch.tensor([True, False, True])
    assert str(flags.sum().dtype) == "torch.int64", "누적은 bool 을 올린다"
    assert str(flags.amax().dtype) == "torch.bool", "고르기는 형을 지킨다"
    # 같은 갈림이 정수에서는 안 보인다 — 둘 다 int64 다. bool 을 물어야 갈린다.
    ints = borch.tensor([3, 1, 4], dtype=borch.int64)
    assert str(ints.sum().dtype) == str(ints.amax().dtype) == "torch.int64"


def test_mean_refuses_integers_like_torch():
    """평균은 나눗셈이라 정수 칸에 답이 없다. numpy 는 조용히 float64 로 올린다."""
    with pytest.raises(RuntimeError, match="could not infer output dtype"):
        borch.tensor([1, 2, 3], dtype=borch.int64).mean()
    with pytest.raises(RuntimeError, match="could not infer output dtype"):
        borch.tensor([True, False]).mean()
    # 실수로 바꾸면 된다 — 없는 기능이 아니라 형이 안 맞는 것이다.
    assert borch.tensor([1, 2, 3], dtype=borch.int64).float().mean().item() == 2.0


def test_logsumexp_answers_float32_not_float64():
    """numpy 의 기본값이 새던 자리. 참거짓은 `-` 를 거절해 아예 멈추기까지 했다."""
    for t in (borch.tensor([1, 2, 3], dtype=borch.int64), borch.tensor([True, False])):
        got = t.logsumexp(0)
        assert str(got.dtype) == "torch.float32", str(got.dtype)
    ref = torch.tensor([1, 2, 3]).logsumexp(0).item()
    assert np.isclose(borch.tensor([1, 2, 3], dtype=borch.int64).logsumexp(0).item(), ref)
