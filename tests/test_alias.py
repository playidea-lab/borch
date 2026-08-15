"""`import borch as torch` **만으로** 어디까지 되는가.

README 는 두 길을 적어 둔다 — `sys.modules["torch"] = borch` 로 심는 것과, 별칭만
쓰는 것. 그리고 심는 쪽을 "강력하고 위험하다" 고 적는다: 그 뒤로는 **남의 라이브러리가
하는 `import torch` 도** 축소판을 받아서, 다른 코드가 섞인 곳에서는 원인을 못 찾는
오류가 된다.

그러면 안전한 쪽만으로 충분한지가 문제다. 별칭은 **그 파일 안의 이름** 하나를 만들
뿐이고, `from X.Y import Z` 는 `sys.modules` 에 등록된 **경로**를 본다. 그 둘이
다르므로 "별칭이면 다 된다" 는 확인 없이 할 말이 아니다.

여기서 그 경계를 못 박는다. 되는 것과 안 되는 것을 값으로 적어 두면, 문서가 어느
쪽을 권해야 하는지가 취향이 아니라 사실이 된다.
"""

import importlib
import sys

import pytest


def test_alias_alone_reaches_the_namespaces():
    """`torch.nn.Linear` 는 별칭만으로 닿는다 — 속성 접근이기 때문이다."""
    import borch as torch

    assert torch.nn.Linear is not None
    assert torch.optim.SGD is not None
    assert torch.optim.lr_scheduler.StepLR is not None
    assert torch.nn.functional.relu is not None


def test_submodule_import_needs_the_path_planted():
    """**`from borch.nn import Linear` 는 별칭만으로는 안 된다.**

    이름 공간이 진짜 모듈이 아니라 `_Namespace` 객체라서 `sys.modules` 에 경로가
    없다. 파이썬은 `from a.b import c` 에서 `a.b` 를 먼저 모듈로 찾으므로 거기서
    멈춘다.

    이것이 `install()` 이 있는 이유이고, 교재가 `from torch.optim.lr_scheduler
    import StepLR` 을 쓰는 한 별칭만으로는 부족하다는 뜻이다.
    """
    for path in [k for k in sys.modules if k.startswith("borch.")]:
        del sys.modules[path]

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("borch.nn")


def test_install_makes_the_submodule_import_work():
    """심으면 된다. **어느 이름으로 심을지는 부르는 쪽이 정한다.**

    `torch` 로 심으면 남의 `import torch` 까지 가로채므로, 섞이는 자리에서는
    자기 이름으로 심는 것이 안전하다 — 그러면 `from borch.nn import Linear` 가
    통하면서 남의 코드는 안 건드린다.
    """
    import borch

    modules = {}
    borch.install("borch", modules)
    assert "borch.nn" in modules
    assert "borch.optim.lr_scheduler" in modules
    assert modules["borch.optim.lr_scheduler"].StepLR is not None


def test_core_install_defaults_to_torch_and_that_is_the_dangerous_one():
    """**코어의 기본값은 `torch` 다.** 그것이 남의 `import torch` 를 가로챈다.

    기본값을 여기 적어 두는 이유는, 그 위험이 문서에만 있고 코드에는 없으면 다음에
    고치는 사람이 기본값을 무심코 바꾸기 때문이다. 바꿀 거면 이 검사를 같이 고쳐야
    하고, 그때 무엇을 바꾸는지가 눈에 들어온다.
    """
    import inspect

    import borch

    assert inspect.signature(borch.install).parameters["name"].default == "torch"
