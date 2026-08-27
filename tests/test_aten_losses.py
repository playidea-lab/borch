"""**The seven losses torch keeps at top level as well as under `F`.**

They were declined as *the raw ATen op — its signature differs from F's*, which is true
and is about why the name looks redundant rather than about what is missing: the
arithmetic is `F`'s and was already here.

The values are frozen in the golden table. What is here is what a value cannot hold —
that the reduction is an **integer enum** and what happens at the edges of it, and one
fact about torch that is worth writing down because it looks like a bug in this file:
**the declared schema and the binding disagree.**
"""

import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import borch                                                     # noqa: E402

_NAMES = ["binary_cross_entropy_with_logits", "cosine_embedding_loss",
          "hinge_embedding_loss", "kl_div", "margin_ranking_loss",
          "poisson_nll_loss", "triplet_margin_loss"]


def _pair():
    x = np.array([[0.5, -1.0, 2.0], [1.5, 0.0, -0.5]], dtype=np.float32)
    return borch.tensor(x), borch.tensor(np.abs(x) + 0.1)


@pytest.mark.parametrize("name", _NAMES)
def test_every_one_of_the_seven_is_at_top_level(name):
    """The whole of what was absent: a name. `F` had all seven the entire time."""
    assert hasattr(borch, name), f"torch.{name} is missing"
    assert hasattr(borch.nn.functional, name)
    assert getattr(borch, name) is not getattr(borch.nn.functional, name), (
        "the top-level one is an alias — it is a different function, with an integer "
        "reduction and a different default")


@pytest.mark.parametrize("bad", [3, -1, "mean", "none", 1.0, None, True])
def test_a_reduction_that_is_not_zero_one_or_two_is_refused(bad):
    """**An enum, so a fourth value is an error rather than the nearest legal one.**

    `True` is in this list on purpose: it is `1` to Python and would silently mean
    *mean*, and a caller who wrote `reduction=True` believes something untrue about the
    argument. `"mean"` is here because it is the word `F` takes, which is the mistake
    this signature invites.
    """
    first, second = _pair()
    with pytest.raises(ValueError, match="reduction"):
        borch.kl_div(first, second, bad)


def test_the_default_is_none_where_every_f_loss_defaults_to_mean():
    """**A table, not a number.** Read as `F.kl_div(a, b)` this is a shape error, which
    is loud; summed afterwards it is a different number, which is not — `mean` divides
    by the element count and a sum does not.
    """
    first, second = _pair()
    assert borch.kl_div(first, second).shape == first.shape
    assert borch.nn.functional.kl_div(first, second).shape == ()


def test_poisson_nll_loss_has_no_defaults_at_all():
    """The odd one of the seven: six required arguments. `F.poisson_nll_loss` gives all
    four trailing ones a value and this gives none, so a caller cannot reach it with two.
    """
    first, second = _pair()
    with pytest.raises(TypeError):
        borch.poisson_nll_loss(first, second)
    assert borch.poisson_nll_loss(first, second, True, False, 1e-8, 1).shape == ()


def test_the_declared_schema_disagrees_with_torchs_own_binding():
    """**Not about this library.** `aten::kl_div(..., int reduction=1)` says mean, and
    `torch.kl_div(a, b)` returns a table — so the schema's default is not the binding's.

    It is asserted here because the behaviour is what this file copies, and a reader who
    checked the schema instead would think this file had the default wrong.
    """
    torch = pytest.importorskip("torch")
    schema = str(torch.ops.aten.kl_div.default._schema)
    assert "int reduction=1" in schema, "the schema changed — re-read it"
    x = np.array([[0.5, -1.0, 2.0]], dtype=np.float32)
    logp = np.log(np.exp(x) / np.exp(x).sum(-1, keepdims=True)).astype(np.float32)
    prob = np.full_like(logp, 1.0 / 3.0)
    assert torch.kl_div(torch.from_numpy(logp),
                        torch.from_numpy(prob)).shape == (1, 3), (
        "torch's binding now honours the schema's default — this file's `0` should "
        "follow it")
