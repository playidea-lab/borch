"""**정의역이 셀 수 있는 인자는 그 값을 전부 넣어 본다** — 남은 절반.

한 축 앞에서 물은 것은 "인자를 흔들면 답이 변하는가" 였다. 그 그물에 안 걸리는
것이 있다 — **인자를 받아서 일부 값만 처리하는 자리**다. `if p == 1: … else: L2`
처럼 `else` 가 한 값의 이름을 달고 정의역의 나머지를 삼키면, 흔들었을 때 답은
변하므로 통과한다. 변한 답이 틀렸을 뿐이다.

이 물음은 **코드를 안 읽고 명세만 읽으면** 된다. torch 문서가 값을 다 적어 둔다.

**표는 골든으로 갔다**(`loss::reduction::*` 스물셋, `index::searchsorted(side=…)` 다섯).
여기 남은 것은 **골든이 물을 수 없는 것**뿐이다.

## 잡은 것 (이력)

`reduction` 이 제일 흔한 다섯 손실에 없었다 — 본문에 `.mean()` 이 박혀 있어서
`reduction=` 을 주면 `TypeError` 였다. 실마리는 **뒤집혀 있다는 것**이었다:
`cosine_embedding_loss`·`multi_margin_loss`·`triplet_margin_loss` 처럼 드문 손실은
열셋 전부 받았다. 나중에 쓴 것이 torch 서명을 따랐고 처음 쓴 것이 안 고쳐진 것이다.
골든이 못 본 이유는 교재가 기본값 `mean` 만 쓰기 때문이고 — **제일 많이 쓰는 자리가
제일 안 물어진 자리**였다.

`_reduce` 의 `else` 가 오타를 삼켰다. `reduction="MEAN"` 이 조용히 평균으로
학습됐다. `searchsorted(side=…)` 는 `**kw` 로 들어가 사라졌고, 같은 계산의 다른
이름인 `bucketize(right=True)` 는 처음부터 맞았다.
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

# **골든에 못 올린 넷.** borch.ts 의 `nllLoss`·`crossEntropy` 가 스칼라만 내므로
# 거기서 `none` 을 만들 수 없다 — 셋을 함께 묻는 자리에 올리려면 그쪽이 먼저다.
CORE_ONLY = [
    ("F.cross_entropy",
     lambda L, r: L.nn.functional.cross_entropy(L.tensor(A), L.tensor(LABEL), reduction=r)),
    ("F.nll_loss",
     lambda L, r: L.nn.functional.nll_loss(L.tensor(LOGP), L.tensor(LABEL), reduction=r)),
    ("F.binary_cross_entropy",
     lambda L, r: L.nn.functional.binary_cross_entropy(L.tensor(P), L.tensor(Y), reduction=r)),
    ("nn.CrossEntropyLoss",
     lambda L, r: L.nn.CrossEntropyLoss(reduction=r)(L.tensor(A), L.tensor(LABEL))),
    ("nn.NLLLoss",
     lambda L, r: L.nn.NLLLoss(reduction=r)(L.tensor(LOGP), L.tensor(LABEL))),
    ("nn.BCELoss",
     lambda L, r: L.nn.BCELoss(reduction=r)(L.tensor(P), L.tensor(Y))),
]


@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
@pytest.mark.parametrize("name,fn", CORE_ONLY, ids=[n for n, _ in CORE_ONLY])
def test_the_losses_the_golden_cannot_ask_yet(name, fn, reduction):
    want = np.asarray(fn(torch, reduction).tolist(), dtype=np.float64)
    got = np.asarray(fn(borch, reduction).tolist(), dtype=np.float64)
    assert np.allclose(got, want, atol=1e-5, rtol=1e-5), (
        f"{name}(reduction={reduction!r}): torch 는 {np.ravel(want)[:4]} 인데 "
        f"코어는 {np.ravel(got)[:4]} 다."
    )


def test_the_rare_losses_were_the_ones_that_had_it():
    """뒤집힘을 문장으로 못 박는다 — **드문 것이 갖추고 흔한 것이 빠져 있었다.**

    골든은 케이스마다 값을 묻지, **서명에 인자가 있는지**는 못 묻는다. 새 손실을
    넣는 사람이 `reduction` 을 빠뜨리면 케이스를 같이 안 넣어도 여기가 그 이름을
    대며 빨개진다 — 그것이 이 검사가 표로 안 옮겨간 이유다.
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


def test_the_loss_layers_take_it_too():
    """층 쪽도 같이 본다. **교재 코드는 함수보다 층을 더 많이 쓴다.**

    함수만 고치면 `nn.MSELoss(reduction="sum")` 이 `TypeError` 로 멈추는데, 그
    상태가 실제로 있었다.
    """
    missing = [name for name in ("MSELoss", "L1Loss", "SmoothL1Loss", "HuberLoss",
                                 "CrossEntropyLoss", "NLLLoss", "BCELoss",
                                 "BCEWithLogitsLoss", "KLDivLoss")
               if "reduction" not in inspect.signature(
                   getattr(borch.nn, name).__init__).parameters]
    assert not missing, f"reduction 을 안 받는 손실 층: {missing}"


@pytest.mark.parametrize("bad", ["meen", ""])
def test_an_unknown_reduction_stops_instead_of_becoming_mean(bad):
    """`else: return out.mean()` 이 정의역의 나머지를 삼키던 자리.

    골든에는 `"MEAN"` 과 `"batchmean"` 이 있다. 여기 둘은 **문구가 값에 안 실리는**
    꼴이라(빈 문자열) 골든의 조각 대조로는 못 묻는다.
    """
    with pytest.raises((ValueError, RuntimeError)):
        borch.nn.functional.l1_loss(borch.tensor(A), borch.tensor(B), reduction=bad)
    # torch 도 멈춘다 — 우리가 더 엄한 것이 아니다.
    with pytest.raises((ValueError, RuntimeError)):
        torch.nn.functional.l1_loss(torch.tensor(A), torch.tensor(B), reduction=bad)


def test_kl_div_is_the_only_one_with_a_fourth_value():
    """**정의역이 함수마다 다르다.** 이것이 이 그물의 제일 나쁜 경계다.

    목록을 "인자 이름 → 값 집합" 으로만 두면 `l1_loss(reduction="batchmean")` 을
    정상으로 센다. 인자로 안 잡히는 것(`nonlinearity` 가 `rnn_tanh`/`rnn_relu` 두
    함수로 흩어진 꼴)은 그물에 안 걸려서 눈에 띄는데, 이쪽은 **걸리면서 틀린다.**
    """
    pred = np.log(np.array([[0.1, 0.6, 0.3], [0.5, 0.2, 0.3]], dtype=np.float32))
    target = np.array([[0.2, 0.5, 0.3], [0.3, 0.4, 0.3]], dtype=np.float32)
    batch = borch.nn.functional.kl_div(borch.tensor(pred), borch.tensor(target),
                                       reduction="batchmean").item()
    mean = borch.nn.functional.kl_div(borch.tensor(pred), borch.tensor(target),
                                      reduction="mean").item()
    assert np.isclose(batch, torch.nn.functional.kl_div(
        torch.tensor(pred), torch.tensor(target), reduction="batchmean").item())
    # 배치로 나눈 것과 원소 수로 나눈 것이 **달라야** 한다 — 같으면 아무것도 안 묻는다.
    assert not np.isclose(batch, mean)
    # 그리고 **다른 손실에서는 틀린 이름이다.**
    with pytest.raises((ValueError, RuntimeError)):
        borch.nn.functional.l1_loss(borch.tensor(A), borch.tensor(B),
                                    reduction="batchmean")
