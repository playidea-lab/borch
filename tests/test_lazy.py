"""Compares a lazy layer's **machinery before initialisation** against real torch, directly.

The golden cases are a table putting the same question to three implementations, so a
question the browser side cannot answer must not go there — a table holding an unanswerable
question cannot say what passed. The parameter machinery before solidifying is exactly such
a place: the core sits on numpy and can hold a parameter with no shape, while the browser
side's layer has no tensor at all before it solidifies.

So that part is looked at here. `tests/test_data.py` looks at `utils.data` the same way —
rather than forcing a question the golden cases do not fit, it asks with real torch
alongside.
"""

import pytest

import borch

torch = pytest.importorskip("torch")


def test_uninitialized_parameters_are_there_before_the_first_forward():
    """**The parameters exist before it solidifies.**

    torch puts two into `parameters()` and two keys into `state_dict`. Without them, the
    common order — build the layer, hand it to an optimizer, run the first batch — breaks.
    """
    ours = borch.nn.LazyLinear(3)
    theirs = torch.nn.LazyLinear(3)
    assert len(list(ours.parameters())) == len(list(theirs.parameters()))
    assert list(ours.state_dict()) == list(theirs.state_dict())


def test_asking_the_shape_before_it_is_known_refuses():
    """Asking for the shape throws — **it must not hand back 0 or something empty.**

    Handing back an empty shape lets the computation continue on it, and the failure lands
    much later, somewhere unrelated.
    """
    for lib in (borch, torch):
        with pytest.raises(RuntimeError):
            _ = lib.nn.LazyLinear(3).weight.shape


def test_arithmetic_before_it_is_known_refuses():
    """Computing with it throws too. torch raises `ValueError`."""
    for lib in (borch, torch):
        with pytest.raises(ValueError):
            _ = lib.nn.LazyLinear(3).weight + 1


def test_the_repr_says_it_is_not_ready():
    """The text before solidifying has to match — `in_features=0` means "not known yet"."""
    assert repr(borch.nn.LazyLinear(3)) == repr(torch.nn.LazyLinear(3))


def test_uninitialized_parameter_prints_like_torch():
    assert repr(borch.nn.UninitializedParameter()) == \
        repr(torch.nn.parameter.UninitializedParameter())
    assert repr(borch.nn.UninitializedBuffer()) == \
        repr(torch.nn.parameter.UninitializedBuffer())


def test_it_is_still_a_parameter():
    """Not appearing as a `Parameter` means `named_parameters` cannot find it."""
    assert isinstance(borch.nn.UninitializedParameter(), borch.nn.Parameter)
    assert isinstance(torch.nn.parameter.UninitializedParameter(),
                      torch.nn.Parameter)


def test_an_optimizer_takes_it_before_the_first_forward():
    """**torch allows this.** There is code written in that order.

    **The first version of this called both and asserted nothing**, so it passed
    unless something raised — and every other test in this file compares the two
    libraries. An implementation that handed the optimizer an *empty* parameter list
    also raises nothing, and would have passed while silently training nothing.

    So the count is compared. It is 1 on both sides: a `LazyLinear` has an
    uninitialised weight and no bias tensor yet, and the optimizer has to be holding
    that one thing rather than nothing at all.
    """
    counts = []
    for lib in (borch, torch):
        opt = lib.optim.SGD(lib.nn.LazyLinear(3).parameters(), lr=0.1)
        counts.append(len(opt.param_groups[0]["params"]))
    assert counts[0] == counts[1], (
        f"the optimizer holds {counts[0]} parameters here and {counts[1]} in torch.\n"
        "  Constructing it raises on neither side, so only the count says which.")
    assert counts[0] > 0, (
        "the optimizer was built over an empty parameter list on both sides.\n"
        "  That raises nothing and trains nothing — the shape this assertion exists\n"
        "  to refuse, since the test it replaced could not tell it from working.")


def test_buffer_is_just_the_tensor():
    """`nn.Buffer(t)` is a mark and the tensor itself — as in torch."""
    t = borch.zeros(2)
    assert isinstance(borch.nn.Buffer(t), borch.Tensor)
    assert isinstance(torch.nn.Buffer(torch.zeros(2)), torch.Tensor)


def test_the_class_itself_changes():
    """**Solidifying changes the class.**

    Handled with a flag, the name does not change, and then both `repr` and `isinstance`
    diverge. The core is Python and can point at the same place torch does.
    """
    for lib, data in ((borch, borch.zeros(2, 5)), (torch, torch.zeros(2, 5))):
        m = lib.nn.LazyLinear(3)
        assert type(m).__name__ == "LazyLinear"
        m(data)
        assert type(m).__name__ == "Linear"
        assert isinstance(m, lib.nn.Linear)
        assert not isinstance(m, lib.nn.LazyLinear)
