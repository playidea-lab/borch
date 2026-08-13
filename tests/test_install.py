"""`install()` — `import torch` 가 이 축소판을 집게 만드는 자리.

**커버리지 0% 였다.** 175개 테스트 어디도 이 함수를 안 지났고, 그런데 이 함수의
docstring 은 "경로를 손으로 적으면 어긋난다 — 실제로 어긋났다"로 시작한다. 한 번 물린
자리를 고쳐놓고 검사를 안 붙였다는 뜻이다.

무는 방식이 특이해서 값 대조로는 안 잡힌다. 물건은 다 있는데 **import 경로가 없어서**
`from torch.optim.lr_scheduler import StepLR` 이 교재 본문에서 멈추는 식이다.
그래서 여기서 묻는 것은 값이 아니라 **경로가 서는가**다.
"""

import pathlib
import sys

import pytest

_root = pathlib.Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import browsertorch as bt                                            # noqa: E402


@pytest.fixture
def modules():
    """진짜 `sys.modules` 를 안 건드린다 — 테스트가 서로를 오염시키면 그때부터
    통과·실패가 실행 순서에 달린다."""
    return {}


def test_install_registers_the_nested_paths(modules):
    """`torch.nn` 만이 아니라 **`torch.optim.lr_scheduler` 까지** 서야 한다.

    한 겹만 도는 구현은 이 검사를 통과하지 못한다 — 옛날에 갈렸던 자리가 정확히 거기다.
    """
    registered = bt.install("torch", modules)

    assert "torch.nn" in registered
    assert "torch.optim" in registered
    assert "torch.optim.lr_scheduler" in registered, "두 겹 아래가 안 섰다"
    assert "torch.utils.data" in registered, "utils 아래도 서야 한다"


def test_registered_paths_hold_the_real_namespaces(modules):
    """경로만 서고 **엉뚱한 것이 들어 있으면** 더 나쁘다 — import 는 되고 값이 틀린다."""
    bt.install("torch", modules)

    assert modules["torch.nn"] is bt.nn
    assert modules["torch.optim"] is bt.optim
    assert modules["torch.optim.lr_scheduler"] is bt.optim.lr_scheduler
    assert modules["torch.utils.data"] is bt.utils.data


def test_install_does_not_plant_the_root(modules):
    """뿌리는 **부르는 쪽이 심는다.** 여기서 심으면 모듈 객체를 두 곳이 쥐게 된다."""
    bt.install("torch", modules)
    assert "torch" not in modules


def test_install_takes_a_different_root_name(modules):
    """이름을 바꿔 심을 수 있어야 한다 — `torch` 로 심는 것이 위험한 자리가 있고,
    README 가 그때 다른 이름을 쓰라고 안내한다."""
    registered = bt.install("bt", modules)
    assert "bt.nn" in registered
    assert all(path.startswith("bt.") for path in registered)


def test_install_finds_every_namespace_rather_than_a_written_list():
    """**목록을 두지 않는다**는 것이 이 함수의 요점이다.

    새 하위 모듈을 만들면 손 안 대고 따라와야 한다. 하나 만들어 붙여보고 확인한다 —
    이것이 안 되면 다음에 `lr_scheduler` 같은 것이 또 빠진다.
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
