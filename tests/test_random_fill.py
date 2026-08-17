"""분포에서 뽑아 제자리에 채우는 일곱 — **골든이 못 묻는 절반.**

골든의 계약은 "진짜 torch 가 답을 낸다" 이고, 셋을 대조하는 힘이 거기서 나온다.
분포의 성질은 그 계약 밖이다 — 답이 torch 가 아니라 **우리가 정한 술어**에서
나오므로, 표에 넣어도 그 줄은 torch 를 안 재고 두 구현만 재는 케이스가 된다.
게다가 표본 수와 허용 오차를 고르는 일이라 그것을 고르는 자리가 필요하다.

**모양·형과 거절은 골든에 있다**(`inplace::분포::*`). 여기 있는 것은 값 쪽이다.

## 범위만 묻는 것은 얕다

`uniform_(a, b)` 가 **전부 `a` 를 채워도** "[a, b] 안" 은 통과한다. 늘 참인 술어가
얕다는 것과 같은 이야기다 — 상수를 내는 구현과 못 가린다. 그래서 범위와 함께
**서로 다른 값이 여럿인가**, 그리고 평균·표준편차가 이론값 근처인가를 본다.

## 씨앗은 여기서 안 묻는다

셋의 난수기가 서로 다르다(`randn` 에서 이미 받아들인 자리다). 같은 씨앗이 같은 수를
준다는 것은 **각자 안에서만** 뜻이 있어서, 코어는 여기서 묻고 borch.ts 는 parity 가
묻는다. 셋 사이로 못 묻는 것을 표에 넣으면 그 줄이 무엇을 재는지 다음 사람이 못 읽는다.
"""

import numpy as np
import pytest

import borch

N = 20000          # 표본 수. 아래 허용 오차가 이 수에 매여 있다.


def drawn(name, *args, dtype=None, size=N):
    x = (borch.zeros(size) if dtype is None
         else borch.tensor(np.zeros(size, dtype=dtype)))
    getattr(x, name)(*args)
    return np.asarray(x.tolist(), dtype=np.float64)


def test_uniform_is_not_a_constant():
    """**범위만 묻는 검사가 잡지 못하는 것.** 전부 `a` 를 채워도 범위는 통과한다."""
    got = drawn("uniform_", -1.0, 3.0)
    assert got.min() >= -1.0 and got.max() < 3.0, "범위를 벗어났다"
    assert len(np.unique(got)) > N // 2, "서로 다른 값이 너무 적다 — 상수가 아닌가"
    assert abs(got.mean() - 1.0) < 0.05, got.mean()
    # 균등분포의 표준편차는 (b-a)/√12 다. 상수면 0 이고 범위 검사는 그것을 놓친다.
    assert abs(got.std() - 4.0 / np.sqrt(12)) < 0.05, got.std()


def test_normal_matches_its_mean_and_spread():
    got = drawn("normal_", 5.0, 2.0)
    assert abs(got.mean() - 5.0) < 0.05, got.mean()
    assert abs(got.std() - 2.0) < 0.05, got.std()


def test_exponential_is_positive_with_mean_one_over_lambda():
    got = drawn("exponential_", 2.0)
    assert (got > 0).all(), "지수분포는 양수다"
    assert abs(got.mean() - 0.5) < 0.02, got.mean()


def test_log_normal_is_positive_and_its_log_is_normal():
    """**로그를 취해야 정규다.** 값 자체의 평균을 보면 꼬리가 무거워 잘 안 맞는다."""
    got = drawn("log_normal_", 0.0, 1.0)
    assert (got > 0).all(), "로그정규는 양수다"
    logged = np.log(got)
    assert abs(logged.mean()) < 0.05, logged.mean()
    assert abs(logged.std() - 1.0) < 0.05, logged.std()


def test_cauchy_has_heavy_tails_and_no_useful_mean():
    """**평균으로는 못 묻는다** — 코시는 평균이 없다. 중앙값과 사분위로 본다."""
    got = drawn("cauchy_", 1.0, 0.5)
    assert abs(np.median(got) - 1.0) < 0.05, np.median(got)
    # 사분위 사이 폭은 2·sigma 다.
    spread = np.percentile(got, 75) - np.percentile(got, 25)
    assert abs(spread - 1.0) < 0.1, spread
    # 꼬리가 무겁다는 것 자체를 못 박는다 — 정규였으면 이만큼 안 나간다.
    assert np.abs(got - 1.0).max() > 20, "꼬리가 너무 얌전하다"


def test_geometric_is_a_count_starting_at_one():
    """torch 의 `geometric_` 은 **첫 성공까지의 시도 수**라 1 부터다(0 이 아니다)."""
    got = drawn("geometric_", 0.3)
    assert got.min() >= 1, got.min()
    assert (got == np.floor(got)).all(), "정수여야 한다"
    assert abs(got.mean() - 1 / 0.3) < 0.1, got.mean()


def test_geometric_fills_an_integer_tensor_too():
    """**이산이라 정수 칸에 답이 있다.** 연속 다섯과 갈리는 하나다."""
    got = drawn("geometric_", 0.5, dtype=np.int64)
    assert got.min() >= 1


def test_random_respects_its_range_and_uses_all_of_it():
    got = drawn("random_", 0, 5, dtype=np.int64, size=2000)
    assert got.min() >= 0 and got.max() < 5
    # **끝값이 나오는지까지 본다.** 범위를 좁게 쓰는 구현은 범위 검사를 통과한다.
    assert set(np.unique(got)) == {0, 1, 2, 3, 4}, np.unique(got)


def test_random_on_bool_is_zero_or_one():
    x = borch.tensor(np.zeros(200, dtype=bool))
    x.random_()
    got = np.asarray(x.tolist())
    assert set(np.unique(got)) <= {False, True}
    assert len(np.unique(got)) == 2, "한 값만 나왔다"


@pytest.mark.parametrize("name,args", [
    ("normal_", ()), ("uniform_", ()), ("exponential_", ()),
    ("cauchy_", ()), ("log_normal_", ()), ("geometric_", (0.5,)),
    ("random_", ()),
])
def test_the_same_seed_gives_the_same_draw(name, args):
    """**각자 안에서만 뜻이 있는 물음.** 셋의 난수기가 다르므로 표에서는 못 묻는다.

    씨앗을 안 지키면 실험이 재현되지 않는데, 그 증상은 "값이 조금씩 다르다" 라
    값 대조로는 안 잡힌다.
    """
    def once():
        borch.manual_seed(7)
        x = borch.zeros(50)
        getattr(x, name)(*args)
        return x.tolist()

    assert once() == once()
