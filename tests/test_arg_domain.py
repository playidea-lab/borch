"""**정의역이 셀 수 있는 인자는 그 값을 전부 넣어 본다.**

축 하나 앞에서 물은 것은 "인자를 흔들면 답이 변하는가" 였다. 그 그물에 안 걸리는
것이 있다 — **인자를 받아서 일부 값만 처리하는 자리**다. `if p == 1: … else: L2` 처럼
`else` 가 한 값의 이름을 달고 정의역의 나머지를 삼키면, 흔들었을 때 답은 변하므로
통과한다. 변한 답이 틀렸을 뿐이다.

이 물음은 **코드를 안 읽고 명세만 읽으면** 된다. torch 문서가 값을 다 적어 둔다 —
`reduction ∈ {none, mean, sum}` · `side ∈ {left, right}` · `mode ∈ {…}`. 분기를 세는
것과 정의역을 아는 것은 다른 일이고, 여기서 쓰는 것은 뒤쪽이다.

## 이 검사가 잡은 것

**`reduction` 이 제일 흔한 다섯 손실에 없었다** — `mse_loss`·`cross_entropy`·
`nll_loss`·`binary_cross_entropy` 와 `nn.MSELoss` 계열. `.mean()` 이 박혀 있어서
`reduction=` 을 주면 `TypeError` 였다.

뒤집혀 있다는 것이 실마리였다. `cosine_embedding_loss`·`multi_margin_loss`·
`triplet_margin_loss` 처럼 **드문 손실은 열셋 전부** 받는다. 나중에 쓴 것이 torch
서명을 따랐고, 처음 쓴 것이 안 고쳐진 것이다. 골든이 못 본 이유는 튜토리얼이
기본값 `mean` 만 쓰기 때문이다 — **가장 많이 쓰는 자리가 가장 안 물어진 자리였다.**

**`_reduce` 의 `else` 가 오타를 삼켰다.** `reduction="MEAN"` 이 조용히 평균으로
학습됐다. torch 는 멈춘다. 사람은 자기가 고른 것이 쓰이는 줄 안다.

**`searchsorted(side=…)` 가 버려졌다.** 같은 것을 두 이름으로 받는데(`right` 는
참거짓, `side` 는 문자열) `side` 가 `**kw` 로 들어가 사라졌다. 자리가 하나씩만
어긋나서 값이 그럴듯하다. 이쪽은 골든에 넣었다 — 결속도 같이 고쳤다.

## 왜 골든이 아니라 여기인가

borch.ts 에 **같은 뒤집힘**이 있다. `reduceAs` 가 있고 `huberLoss`·`klDiv` 는 쓰는데
`mseLoss`·`l1Loss`·`smoothL1Loss` 는 안 쓴다. 그쪽이 따라오면 표를 골든으로 옮긴다
(`tests/test_reduce_dtype.py`·`tests/test_fold_grad.py` 가 같은 이유로 같은 자리에
있다).
"""

import inspect

import numpy as np
import pytest
import torch

import borch

rng = np.random.default_rng(0)
A = rng.standard_normal((4, 3)).astype(np.float32)
B = rng.standard_normal((4, 3)).astype(np.float32)
P = np.clip(np.abs(rng.random((4, 3))).astype(np.float32), 0.05, 0.95)
Y = (P > 0.5).astype(np.float32)
LOGP = np.log(np.abs(rng.random((4, 3))).astype(np.float32) + 0.05)
LABEL = np.array([0, 1, 2, 1])

REDUCTIONS = ["none", "mean", "sum"]

# **가장 흔한 것부터 적는다.** 드문 손실은 처음부터 다 받고 있었다.
LOSSES = [
    ("F.mse_loss",
     lambda L, r: L.nn.functional.mse_loss(L.tensor(A), L.tensor(B), reduction=r)),
    ("F.l1_loss",
     lambda L, r: L.nn.functional.l1_loss(L.tensor(A), L.tensor(B), reduction=r)),
    ("F.smooth_l1_loss",
     lambda L, r: L.nn.functional.smooth_l1_loss(L.tensor(A), L.tensor(B), reduction=r)),
    ("F.cross_entropy",
     lambda L, r: L.nn.functional.cross_entropy(L.tensor(A), L.tensor(LABEL), reduction=r)),
    ("F.nll_loss",
     lambda L, r: L.nn.functional.nll_loss(L.tensor(LOGP), L.tensor(LABEL), reduction=r)),
    ("F.binary_cross_entropy",
     lambda L, r: L.nn.functional.binary_cross_entropy(L.tensor(P), L.tensor(Y), reduction=r)),
    ("nn.MSELoss",
     lambda L, r: L.nn.MSELoss(reduction=r)(L.tensor(A), L.tensor(B))),
    ("nn.L1Loss",
     lambda L, r: L.nn.L1Loss(reduction=r)(L.tensor(A), L.tensor(B))),
    ("nn.SmoothL1Loss",
     lambda L, r: L.nn.SmoothL1Loss(reduction=r)(L.tensor(A), L.tensor(B))),
    ("nn.CrossEntropyLoss",
     lambda L, r: L.nn.CrossEntropyLoss(reduction=r)(L.tensor(A), L.tensor(LABEL))),
    ("nn.NLLLoss",
     lambda L, r: L.nn.NLLLoss(reduction=r)(L.tensor(LOGP), L.tensor(LABEL))),
    ("nn.BCELoss",
     lambda L, r: L.nn.BCELoss(reduction=r)(L.tensor(P), L.tensor(Y))),
]


def value_of(lib, fn, arg):
    try:
        return np.asarray(fn(lib, arg).tolist(), dtype=np.float64)
    except Exception as exc:                                    # noqa: BLE001
        return f"거절({type(exc).__name__})"


@pytest.mark.parametrize("reduction", REDUCTIONS)
@pytest.mark.parametrize("name,fn", LOSSES, ids=[n for n, _ in LOSSES])
def test_every_loss_takes_every_reduction(name, fn, reduction):
    want = value_of(torch, fn, reduction)
    got = value_of(borch, fn, reduction)
    assert not isinstance(got, str), (
        f"{name}(reduction={reduction!r}): 코어가 {got}.\n"
        "접는 방식은 손실의 일부다 — `_reduce` 를 쓰고 `reduction` 을 서명에 둔다."
    )
    assert np.allclose(got, want, atol=1e-5, rtol=1e-5), (
        f"{name}(reduction={reduction!r}): torch 는 {np.ravel(want)[:4]} 인데 "
        f"코어는 {np.ravel(got)[:4]} 다."
    )


@pytest.mark.parametrize("bad", ["MEAN", "meen", "", "batchmean"])
def test_an_unknown_reduction_stops_instead_of_becoming_mean(bad):
    """`else: return out.mean()` 이 정의역의 나머지를 삼키던 자리.

    `batchmean` 도 여기 있다 — **`kl_div` 에만 있는 값**이라 다른 손실에서는 틀린
    이름이다. 삼키면 배치로 나눌 줄 알았던 사람이 원소 수로 나눈 값을 받는다.
    """
    with pytest.raises((ValueError, RuntimeError)):
        borch.nn.functional.l1_loss(borch.tensor(A), borch.tensor(B), reduction=bad)
    # torch 도 멈춘다 — 우리가 더 엄한 것이 아니다.
    with pytest.raises((ValueError, RuntimeError)):
        torch.nn.functional.l1_loss(torch.tensor(A), torch.tensor(B), reduction=bad)


def test_kl_div_still_takes_batchmean():
    """정의역이 손실마다 다르다. `kl_div` 만 넷째 값을 받는다 — 배치로 나눈다."""
    pred = np.log(np.array([[0.1, 0.6, 0.3], [0.5, 0.2, 0.3]], dtype=np.float32))
    target = np.array([[0.2, 0.5, 0.3], [0.3, 0.4, 0.3]], dtype=np.float32)
    a = torch.nn.functional.kl_div(torch.tensor(pred), torch.tensor(target),
                                   reduction="batchmean").item()
    b = borch.nn.functional.kl_div(borch.tensor(pred), borch.tensor(target),
                                   reduction="batchmean").item()
    assert np.isclose(a, b, atol=1e-6)
    # 원소 수로 나눈 것과 **달라야** 한다 — 같으면 이 케이스가 아무것도 안 묻는다.
    mean = borch.nn.functional.kl_div(borch.tensor(pred), borch.tensor(target),
                                      reduction="mean").item()
    assert not np.isclose(b, mean), "배치 크기와 원소 수가 같은 자료로는 못 묻는다"


def test_the_rare_losses_were_the_ones_that_had_it():
    """뒤집힘을 문장으로 못 박는다 — **드문 것이 갖추고 흔한 것이 빠져 있었다.**

    이것이 실마리였다. 서명을 비교해 보면 나중에 쓴 손실은 torch 를 따랐고 처음
    쓴 것은 안 고쳐졌다는 것이 드러난다. 새 손실을 넣는 사람이 `reduction` 을
    빠뜨리면 여기가 그 이름을 대며 빨개진다.
    """
    missing = []
    for name in dir(torch.nn.functional):
        if not (name.endswith("_loss") or name in ("cross_entropy", "kl_div",
                                                   "binary_cross_entropy")):
            continue
        ours = getattr(borch.nn.functional, name, None)
        if ours is None:
            continue
        theirs = inspect.signature(getattr(torch.nn.functional, name)).parameters
        if "reduction" in theirs and "reduction" not in inspect.signature(ours).parameters:
            missing.append(name)
    assert not missing, f"reduction 을 안 받는 손실: {missing}"
