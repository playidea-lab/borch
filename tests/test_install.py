"""`install()` — the place that makes `import torch` pick up this subset.

**Coverage was 0%.** None of the 175 tests passed through this function, and its
docstring opens with "writing the paths by hand drifts — and it did". Which means
a place that had bitten once was fixed and left without a check.

The way it bites is unusual and a value comparison does not catch it. Everything
exists and **the import path does not**, so
`from torch.optim.lr_scheduler import StepLR` stops in the body of a textbook.
So what is asked here is not a value but **whether the paths stand up.**
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
    """The real `sys.modules` is left alone — once tests contaminate each other,
    passing and failing depend on the running order."""
    return {}


def test_install_registers_the_nested_paths(modules):
    """Not `torch.nn` alone — **`torch.optim.lr_scheduler` too** has to stand up.

    An implementation that walks one level does not pass this check — and that is
    exactly where it drifted before.
    """
    registered = bt.install("torch", modules)

    assert "torch.nn" in registered
    assert "torch.optim" in registered
    assert "torch.optim.lr_scheduler" in registered, "two levels down did not stand up"
    assert "torch.utils.data" in registered, "under utils has to stand up too"


def test_registered_paths_hold_the_real_namespaces(modules):
    """A path that stands up **holding the wrong thing** is worse — the import
    works and the value is wrong."""
    bt.install("torch", modules)

    assert modules["torch.nn"] is bt.nn
    assert modules["torch.optim"] is bt.optim
    assert modules["torch.optim.lr_scheduler"] is bt.optim.lr_scheduler
    assert modules["torch.utils.data"] is bt.utils.data


def test_install_does_not_plant_the_root(modules):
    """The root is planted **by the caller.** Planted here, two places would hold
    the module object."""
    bt.install("torch", modules)
    assert "torch" not in modules


def test_install_takes_a_different_root_name(modules):
    """It has to be plantable under a different name — there are places where
    planting as `torch` is risky, and the README points at another name there."""
    registered = bt.install("bt", modules)
    assert "bt.nn" in registered
    assert all(path.startswith("bt.") for path in registered)


def test_install_finds_every_namespace_rather_than_a_written_list():
    """**Keeping no list** is this function's point.

    A new submodule has to follow along untouched. One is built and attached to
    confirm it — without this, the next thing like `lr_scheduler` goes missing
    again.
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
