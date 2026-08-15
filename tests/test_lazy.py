"""게으른 층의 **초기화 전 기계**를 진짜 torch 와 직접 견준다.

골든은 세 구현에 같은 질문을 던지는 표라, 브라우저 쪽이 답할 수 없는 질문은 거기
두면 안 된다 — 답할 수 없는 질문이 표에 있으면 그 표가 "무엇이 통과했는가" 를 못
말한다. 굳기 전의 파라미터 기계가 정확히 그런 자리다: 코어는 numpy 위라 모양 없는
파라미터를 들고 있을 수 있지만, 브라우저 쪽 층은 굳기 전에 텐서가 아예 없다.

그래서 그 부분만 여기서 본다. `tests/test_data.py` 가 `utils.data` 를 같은 방식으로
본다 — 골든에 안 맞는 질문을 억지로 넣는 대신, 진짜 torch 를 옆에 두고 묻는다.
"""

import pytest

import borch

torch = pytest.importorskip("torch")


def test_uninitialized_parameters_are_there_before_the_first_forward():
    """**굳기 전에도 파라미터가 있다.**

    torch 는 `parameters()` 에 둘을 내놓고 `state_dict` 에 열쇠 둘을 둔다. 없으면
    "층을 만들고 옵티마이저에 넘긴 뒤 첫 배치를 돌린다" 는 흔한 순서가 깨진다.
    """
    ours = borch.nn.LazyLinear(3)
    theirs = torch.nn.LazyLinear(3)
    assert len(list(ours.parameters())) == len(list(theirs.parameters()))
    assert list(ours.state_dict()) == list(theirs.state_dict())


def test_asking_the_shape_before_it_is_known_refuses():
    """모양을 물으면 던진다 — **0 이나 빈 것을 주면 안 된다.**

    빈 모양을 주면 그것으로 계산이 이어지고, 실패가 한참 뒤에 엉뚱한 자리에서 난다.
    """
    for lib in (borch, torch):
        with pytest.raises(RuntimeError):
            _ = lib.nn.LazyLinear(3).weight.shape


def test_arithmetic_before_it_is_known_refuses():
    """셈을 해도 던진다. torch 는 `ValueError` 를 낸다."""
    for lib in (borch, torch):
        with pytest.raises(ValueError):
            _ = lib.nn.LazyLinear(3).weight + 1


def test_the_repr_says_it_is_not_ready():
    """굳기 전의 글자가 같아야 한다 — `in_features=0` 이 "아직 모른다" 는 뜻이다."""
    assert repr(borch.nn.LazyLinear(3)) == repr(torch.nn.LazyLinear(3))


def test_uninitialized_parameter_prints_like_torch():
    assert repr(borch.nn.UninitializedParameter()) == \
        repr(torch.nn.parameter.UninitializedParameter())
    assert repr(borch.nn.UninitializedBuffer()) == \
        repr(torch.nn.parameter.UninitializedBuffer())


def test_it_is_still_a_parameter():
    """`Parameter` 로 안 보이면 `named_parameters` 가 못 찾는다."""
    assert isinstance(borch.nn.UninitializedParameter(), borch.nn.Parameter)
    assert isinstance(torch.nn.parameter.UninitializedParameter(),
                      torch.nn.Parameter)


def test_an_optimizer_takes_it_before_the_first_forward():
    """**torch 가 이것을 허용한다.** 그 순서로 짜는 코드가 있다."""
    for lib in (borch, torch):
        lib.optim.SGD(lib.nn.LazyLinear(3).parameters(), lr=0.1)


def test_buffer_is_just_the_tensor():
    """`nn.Buffer(t)` 는 표시일 뿐 텐서 자신이다 — torch 도 그렇다."""
    t = borch.zeros(2)
    assert isinstance(borch.nn.Buffer(t), borch.Tensor)
    assert isinstance(torch.nn.Buffer(torch.zeros(2)), torch.Tensor)


def test_the_class_itself_changes():
    """**굳으면 클래스가 바뀐다.**

    깃발 하나로 처리하면 이름이 안 바뀌고, 그러면 `repr` 도 `isinstance` 도 갈린다.
    코어는 파이썬이라 torch 와 같은 자리를 그대로 짚을 수 있다.
    """
    for lib, data in ((borch, borch.zeros(2, 5)), (torch, torch.zeros(2, 5))):
        m = lib.nn.LazyLinear(3)
        assert type(m).__name__ == "LazyLinear"
        m(data)
        assert type(m).__name__ == "Linear"
        assert isinstance(m, lib.nn.Linear)
        assert not isinstance(m, lib.nn.LazyLinear)
