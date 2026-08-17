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
    # ── 표 밖에 있던 것들 ────────────────────────────────────────────────
    #
    # 처음 굳힌 열아홉으로는 부족했다. **묻지 않은 자리는 안 맞는다** — 아래 열넷을
    # 재보니 열 자리가 갈렸고, 갈래는 위와 같은 둘이었다(torch 가 멈추는데 우리가
    # 근사한다 / 우리가 멈추는데 torch 가 답한다) 더하기 numpy 의 float64 누출.
    ("nansum", lambda t: t.nansum()),
    ("nanmean", lambda t: t.nanmean()),
    ("nanmedian", lambda t: t.nanmedian()),
    ("logcumsumexp", lambda t: t.logcumsumexp(0)),
    ("quantile", lambda t: t.quantile(0.5)),
    ("bincount", lambda t: t.bincount()),
    ("diff", lambda t: t.diff()),
    # 값과 색인을 같이 내는 것들. **둘의 형이 따로 논다** — `bool.sort()` 는
    # `bool + int64` 다.
    ("cummax", lambda t: t.cummax(0)),
    ("cummin", lambda t: t.cummin(0)),
    ("aminmax", lambda t: t.aminmax()),
    ("mode", lambda t: t.mode()),
    ("sort", lambda t: t.sort()),
    ("topk", lambda t: t.topk(2)),
    ("median(dim=0)", lambda t: t.median(dim=0)),
]

# **둘 다 물어야 한다.** int64 만 물으면 "형을 지킨다" 와 "bool 을 올린다" 가 같아
# 보이고, 절반만 맞는 구현이 통과한다.
KINDS = [
    ("int64", [3, 1, 4, 1, 5], True),
    ("bool", [True, False, True], False),
]


def name_of(out):
    """형 이름. 값·색인을 같이 내는 것은 **둘 다** 적는다 — 한쪽만 보면 갈림이 숨는다.

    torch 는 `return_types.sort` 같은 네임드튜플을, 코어는 자기 래퍼를 준다. 둘 다
    `.values`/`.indices` 를 갖지만 **평범한 텐서도 `.values` 를 갖는다**(희소용) —
    그것으로 가리면 모든 텐서가 튜플로 읽힌다. 실제로 한 번 그렇게 재서 스물다섯
    자리가 갈린 것처럼 나왔다.
    """
    if isinstance(out, tuple) or hasattr(out, "_fields"):
        return " + ".join(name_of(x) for x in tuple(out))
    if type(out).__name__ in ("_MinMax", "mode", "_Mode"):
        return f"{name_of(out.values)} + {name_of(out.indices)}"
    return str(out.dtype).replace("torch.", "") if hasattr(out, "dtype") \
        else type(out).__name__


def answer(lib, values, as_int, fn):
    """형 이름, 또는 거절했으면 `"거절"`. **거절도 답이다.**"""
    try:
        t = lib.tensor(values, dtype=lib.int64) if as_int else lib.tensor(values)
        out = fn(t)
    except Exception:                                           # noqa: BLE001
        return "거절"
    return name_of(out)


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


def test_dtype_argument_beats_the_rule():
    """`dtype=` 를 주면 그것이 전부를 이긴다 — 규칙보다 위다.

    **`xfail(strict=True)` 로 걸려 있던 자리다.** 채우니 그 표시가 "예상 못 한
    통과" 로 빨개져서 여기를 같이 고치라고 말했다 — 설계한 그대로 물었다.

    규칙은 한 줄이다: **넣기 전에 바꾼다.** 접고 나서가 아니다. 실측이 그 둘을
    가른다 — `[1.7, −2.3, 0.9].sum(dtype=int64)` 이 `−1` 이다. 먼저 접으면 `0.3`
    이고 깎아도 `0` 인데, 먼저 깎으면 `[1, −2, 0]` 이라 합이 `−1` 이다.
    """
    for values in ([3, 1, 4], [True, False, True]):
        a = torch.tensor(values).sum(dtype=torch.float32).dtype
        b = borch.tensor(values).sum(dtype=borch.float32).dtype
        assert str(a) == str(b) == "torch.float32"
    # **먼저 깎는다는 것**을 값으로 못 박는다. 형만 물으면 두 순서가 구별이 안 된다.
    reals = [1.7, -2.3, 0.9]
    assert torch.tensor(reals).sum(dtype=torch.int64).item() == -1
    assert borch.tensor(reals).sum(dtype=borch.int64).item() == -1


def test_dtype_argument_keeps_the_two_refusals_torch_keeps():
    """`dtype=` 이 **모든** 거절을 푸는 것은 아니다 — 둘은 그대로다(실측).

    `mean` 은 입력 쪽 거절만 풀린다. 정수 입력에 `dtype=float32` 는 돌지만
    **결과가 정수인 평균**은 여전히 답이 없다. `cumsum`·`cumprod` 는 `dtype=bool`
    을 torch 가 아예 안 만들었다 — `sum(dtype=bool)` 은 되는데도 그렇다.

    관대한 쪽으로 갈리는 것도 갈리는 것이다. 여기서 값을 내주면 그 코드가 진짜
    torch 에서 깨진다.
    """
    assert borch.tensor([3, 1, 4]).mean(dtype=borch.float32).item() == pytest.approx(
        torch.tensor([3, 1, 4]).mean(dtype=torch.float32).item())
    with pytest.raises(RuntimeError, match="could not infer output dtype"):
        borch.tensor([1.5, 2.5]).mean(dtype=borch.int64)
    with pytest.raises(NotImplementedError):
        borch.tensor([1, 2, 3]).cumsum(0, dtype=borch.bool_)
    with pytest.raises(NotImplementedError):
        borch.tensor([1, 2, 3]).cumprod(0, dtype=borch.bool_)


def test_to_actually_changes_the_dtype():
    """`x.to(torch.float32)` 가 **형을 바꾼다.**

    오래 안 바꿨다 — `to` 가 장치 문자열만 보고 나머지를 조용히 버렸다. 예외도
    경고도 없이 원래 형 그대로였고, 정수 텐서에서는 그 뒤 나눗셈이 **정수
    나눗셈으로** 갈렸다. 축약에 `dtype=` 를 붙이다가 드러났다 — 그쪽이 이 함수를
    부르는데 형이 안 바뀌어서.
    """
    ints = borch.tensor([3, 1, 4], dtype=borch.int64)
    assert str(ints.to(borch.float32).dtype) == "torch.float32"
    assert str(ints.to(borch.int64).dtype) == "torch.int64"
    # 장치 쪽은 그대로다 — 'cpu' 는 통과하고 다른 장치는 멈춘다.
    assert str(ints.to("cpu").dtype) == "torch.int64"


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
