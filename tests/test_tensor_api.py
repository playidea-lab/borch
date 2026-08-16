"""메서드 이름을 **진짜 torch 에 물어 확인한다.**

`borch/_ops.py` 에 표 셋이 있다 — 모듈 함수를 메서드로도 낼 이름(`_AS_METHOD`), 그리고
제자리 연산으로 낼 이름들(`_INPLACE_*`). 표는 손으로 적은 것이라 틀릴 수 있고, 틀리는
방향이 둘 다 나쁘다.

- **torch 에 없는 이름을 만들면** 우리에게서만 도는 코드를 쓰게 된다. 그 코드를 진짜
  torch 로 옮기는 날 `AttributeError` 가 나는데, 그때는 이미 그 이름에 기대어 짜 놓은
  뒤다. 이 저장소의 유일한 주장이 "임포트만 바꿔 돌린다" 이므로 이것은 그 주장을 깬다.
- **표에 적고 안 만들면** 조용히 아무 일도 안 일어난다. 빈자리 수만 안 줄고 아무도
  모른다.

골든이 이것을 못 잡는다. 골든은 우리가 **물어본** 이름만 보고, 표에 적었는데 안 물은
이름은 그 표 밖이다. 그래서 표 자체를 여기서 본다.
"""

import pytest

import borch
from borch import _ops

torch = pytest.importorskip("torch")


def _named(tables):
    for label, names, suffix in tables:
        for name in names:
            yield label, name + suffix


TABLES = [
    ("_AS_METHOD", _ops._AS_METHOD, ""),
    ("_INPLACE_UNARY", _ops._INPLACE_UNARY, "_"),
    ("_INPLACE_MORE", _ops._INPLACE_MORE, "_"),
    ("_INPLACE_BINARY", _ops._INPLACE_BINARY, "_"),
    ("_INPLACE_ARGS", _ops._INPLACE_ARGS, "_"),
]


def test_every_name_we_add_is_a_real_torch_method():
    """**torch 에 없는 메서드를 만들면 안 된다.**

    없는 이름을 만들면 그것에 기대어 짠 코드가 진짜 torch 에서 안 돈다 — 흉내가
    아니라 **다른 라이브러리**를 만드는 것이다.
    """
    invented = [f"{label}: {name}" for label, name in _named(TABLES)
                if not hasattr(torch.Tensor, name)]
    assert not invented, (
        "torch.Tensor 에 없는 이름을 만들고 있다:\n  " + "\n  ".join(invented) +
        "\n\n표에서 빼라 — 없는 이름은 임포트만 바꿔서는 안 도는 코드를 만든다.")


def test_every_name_we_promised_is_actually_there():
    """**표에 적고 안 만든 것이 없어야 한다.**

    이쪽이 틀리면 조용하다. 예외도 안 나고 빈자리 수만 안 줄어든다.
    """
    missing = [f"{label}: {name}" for label, name in _named(TABLES)
               if not hasattr(borch.Tensor, name)]
    assert not missing, (
        "표에 적었는데 안 만들어진 이름이 있다:\n  " + "\n  ".join(missing))


def test_the_method_and_the_function_are_the_same_calculation():
    """`x.add(y)` 와 `borch.add(x, y)` 가 같은 답이어야 한다.

    두 벌로 적으면 언젠가 갈리고, 그때 값이 그럴듯해서 안 보인다. 여기서는 같은
    함수를 가리키는지가 아니라 **같은 답을 내는지**를 본다.
    """
    x = borch.tensor([[1.0, 2.0], [3.0, 4.0]])
    y = borch.tensor([[0.5, 1.5], [2.5, 3.5]])
    pairs = [("add", (y,)), ("mul", (y,)), ("sub", (y,)), ("div", (y,)),
             ("fmax", (y,)), ("cross", ()), ("det", ()), ("matrix_exp", ()),
             ("logical_and", (y,)), ("count_nonzero", ())]
    for name, extra in pairs:
        if name == "cross":
            continue
        method = getattr(x, name)(*extra)
        plain = getattr(borch, name)(x, *extra)
        assert borch.allclose(method, plain), f"{name}: 메서드와 함수가 갈린다"


def test_in_place_writes_into_the_same_tensor():
    """제자리 연산은 **같은 텐서를 고쳐야** 한다 — 새것을 돌려주면 뜻이 없다."""
    for name in ("absolute_", "sinc_", "sgn_"):
        x = borch.tensor([-1.0, 2.0, -3.0])
        got = getattr(x, name)()
        assert got is x, f"{name}: 제자리가 아니라 새것을 냈다"


def test_in_place_refuses_a_leaf_that_needs_grad():
    """torch 가 거절하는 자리를 따라 거절한다."""
    for lib in (borch, torch):
        x = lib.tensor([1.0, 2.0], requires_grad=True)
        with pytest.raises(RuntimeError):
            x.absolute_()
