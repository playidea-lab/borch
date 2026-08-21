"""`install()` — where `import torch` is made to pick up this subset.

**Coverage was 0%.** None of the 175 tests passed through this function, and yet its
docstring opens with "writing the paths by hand goes out of step — and it did". Somewhere
that had bitten once was fixed with no check attached.

The way it bites is unusual and a value comparison does not catch it. Everything is present
and **the import path is not**, so `from torch.optim.lr_scheduler import StepLR` stops in the
middle of a textbook. So what is asked here is not values but **whether the paths stand.**
"""

import pathlib
import sys

import pytest

_root = pathlib.Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import borch as bt                                            # noqa: E402


@pytest.fixture
def modules():
    """It does not touch the real `sys.modules` — once tests contaminate each other, passing
    and failing depend on the order they run in."""
    return {}


def test_install_registers_the_nested_paths(modules):
    """Not just `torch.nn` but **`torch.optim.lr_scheduler` too** has to stand.

    An implementation that walks one level does not pass this — that is exactly where it once
    went wrong.
    """
    registered = bt.install("torch", modules)

    assert "torch.nn" in registered
    assert "torch.optim" in registered
    assert "torch.optim.lr_scheduler" in registered, "two levels down did not stand"
    assert "torch.utils.data" in registered, "below utils has to stand too"


def test_registered_paths_hold_the_real_namespaces(modules):
    """A path standing with **the wrong thing inside it** is worse — the import works and the value is wrong."""
    bt.install("torch", modules)

    assert modules["torch.nn"] is bt.nn
    assert modules["torch.optim"] is bt.optim
    assert modules["torch.optim.lr_scheduler"] is bt.optim.lr_scheduler
    assert modules["torch.utils.data"] is bt.utils.data


def test_install_does_not_plant_the_root(modules):
    """The root is **planted by the caller.** Planting it here leaves two places holding the module object."""
    bt.install("torch", modules)
    assert "torch" not in modules


def test_install_takes_a_different_root_name(modules):
    """It has to be plantable under another name — there are places where planting as `torch`
    is dangerous, and the README points at another name for those."""
    registered = bt.install("bt", modules)
    assert "bt.nn" in registered
    assert all(path.startswith("bt.") for path in registered)


def test_install_finds_every_namespace_rather_than_a_written_list():
    """**Keeping no list** is this function's point.

    A new submodule has to follow without anyone touching this. One is built and attached to
    confirm — without that, the next thing like `lr_scheduler` goes missing again.
    """
    modules = {}

    class _Fresh(bt._Namespace):
        pass

    bt.nn.freshly_added = _Fresh()
    try:
        registered = bt.install("torch", modules)
        assert "torch.nn.freshly_added" in registered
    finally:
        del bt.nn.freshly_added
