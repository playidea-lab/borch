"""`a += b` — does it change `a`, or quietly make a new tensor and rebind the name?

**Both spellings pass a value check.** `a += b` without `__iadd__` falls back to
`a = a + b`, and the number that comes out is identical; what parts is whether the
object the caller still holds — a parameter inside an optimizer, a running mean inside
a norm layer — moved. So this file reads identity, not values.

It exists because all three of `+=`, `-=` and `*=` raised
`TypeError: add() takes from 2 to 3 positional arguments but 0 were given`, in a
library whose golden files exercise `add_`, `sub_` and `mul_` several hundred times.
The operators and the underscore names were two doors to one room and only one of them
was ever opened.
"""
import operator

import numpy as np
import pytest

import borch

# (python's name for it, the underscore method torch routes it to)
#
# The right-hand side is a **plain number** and not a tensor. `**=` with a tensor
# exponent is refused here by design — it is one of the browser subset's declared
# absences — and writing the fixture with tensors on the right would have turned that
# documented refusal into a failing row and hidden the four real ones behind it.
OPERATORS = [
    ("iadd", "add_"),
    ("isub", "sub_"),
    ("imul", "mul_"),
    ("itruediv", "div_"),
    ("ipow", "pow_"),
    ("ifloordiv", "floor_divide_"),
    ("imod", "remainder_"),
]
RIGHT = 2.0


def _apply(t, other, name):
    return getattr(operator, name)(t, other)


@pytest.mark.parametrize(("name", "underscore"), OPERATORS)
def test_the_augmented_operators_change_the_tensor_they_are_given(name, underscore):
    if not hasattr(borch.Tensor, underscore):
        pytest.skip(f"no `{underscore}` to route to")
    start = np.array([4.0, 9.0], dtype=np.float32)
    t = borch.tensor(start.copy())
    held = t
    out = _apply(t, RIGHT, name)

    assert out is held, f"`{name}` rebound the name instead of changing the tensor"
    assert not np.allclose(np.asarray(held.data), start), (
        f"`{name}` returned the same object without changing it")


@pytest.mark.parametrize(("name", "underscore"), OPERATORS)
def test_the_augmented_operators_agree_with_the_underscore_name(name, underscore):
    if not hasattr(borch.Tensor, underscore):
        pytest.skip(f"no `{underscore}` to route to")
    start = np.array([4.0, 9.0], dtype=np.float32)

    a = borch.tensor(start.copy())
    _apply(a, RIGHT, name)

    b = borch.tensor(start.copy())
    getattr(b, underscore)(RIGHT)

    assert np.allclose(np.asarray(a.data), np.asarray(b.data))


@pytest.mark.parametrize(("name", "underscore"), OPERATORS)
def test_the_augmented_operators_refuse_a_leaf_that_requires_grad(name, underscore):
    """The refusal has to arrive through the operator too. A door that skips the
    check is a door that lets a gradient graph be corrupted silently."""
    if not hasattr(borch.Tensor, underscore):
        pytest.skip(f"no `{underscore}` to route to")
    t = borch.tensor(np.array([4.0, 9.0], dtype=np.float32), requires_grad=True)
    with pytest.raises(RuntimeError):
        _apply(t, RIGHT, name)


def test_a_plain_number_on_the_right_works_too():
    """The failing signature took `other.data if isinstance(other, Tensor)`, so a
    bare float went down a different branch than a tensor did. Both are written."""
    t = borch.tensor(np.array([4.0, 9.0], dtype=np.float32))
    held = t
    t += 1.0
    assert t is held
    assert np.allclose(np.asarray(t.data), [5.0, 10.0])
