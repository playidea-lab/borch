"""How far `import borch as torch` gets **on its own.**

The README writes down two routes — planting it with `sys.modules["torch"] = borch`, and
using an alias alone. And it calls the planting one "powerful and dangerous": after it,
**someone else's library doing `import torch`** receives the subset too, and in code that
mixes libraries that becomes an error with no findable cause.

So whether the safe route is enough is the question. An alias makes one name **inside that
file**, while `from X.Y import Z` looks at the **path** registered in `sys.modules`. Those
two differ, so "the alias does everything" is not something to say without checking.

That boundary is pinned here. Writing down what works and what does not, as values, makes
which route the documentation should recommend a fact rather than a taste.
"""

import importlib
import sys

import pytest


def test_alias_alone_reaches_the_namespaces():
    """`torch.nn.Linear` is reachable with the alias alone — it is attribute access."""
    import borch as torch

    assert torch.nn.Linear is not None
    assert torch.optim.SGD is not None
    assert torch.optim.lr_scheduler.StepLR is not None
    assert torch.nn.functional.relu is not None


def test_submodule_import_needs_the_path_planted():
    """**`from borch.nn import Linear` does not work with the alias alone.**

    The namespace is a `_Namespace` object rather than a real module, so no path for it is
    registered in `sys.modules`. Python looks for `a.b` as a module first in
    `from a.b import c`, and stops there.

    That is why `install()` exists, and it means the alias alone is not enough as long as
    textbooks write `from torch.optim.lr_scheduler import StepLR`.
    """
    for path in [k for k in sys.modules if k.startswith("borch.")]:
        del sys.modules[path]

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("borch.nn")


def test_install_makes_the_submodule_import_work():
    """Planting works. **Which name it is planted under is the caller's decision.**

    Planting as `torch` intercepts someone else's `import torch` too, so where libraries mix,
    planting under its own name is safer — then `from borch.nn import Linear`
    works while leaving other people's code alone.
    """
    import borch

    modules = {}
    borch.install("borch", modules)
    assert "borch.nn" in modules
    assert "borch.optim.lr_scheduler" in modules
    assert modules["borch.optim.lr_scheduler"].StepLR is not None


def test_core_install_defaults_to_torch_and_that_is_the_dangerous_one():
    """**The core's default is `torch`.** That is what intercepts someone else's `import torch`.

    The default is written down here because a danger that lives only in the documentation
    and not in the code gets changed absent-mindedly by whoever edits next. Changing it means
    changing this check too, and at that moment what is being changed comes into view.
    """
    import inspect

    import borch

    assert inspect.signature(borch.install).parameters["name"].default == "torch"
