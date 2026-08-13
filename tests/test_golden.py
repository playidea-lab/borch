"""골든 하네스 자체가 제 일을 하는지 본다.

GPU 백엔드는 아직 없다. 그래서 browsertorch 를 **제3의 라이브러리인 척** 골든에
대조시킨다 — 하네스가 도는지는 그것으로 드러나고, 백엔드가 생기면 같은 자리에 넣는다.

그런데 "도는가"만 물으면 안 된다. 골든 하네스가 조용히 망가지는 방식은 두 가지다 —
표가 바뀐 뒤 낡은 골든과 비교하거나, 입력이 갈린 채로 비교하는 것. 둘 다 **통과가
나오는데 아무것도 대조하지 않은** 상태다. 그래서 그 둘을 일부러 만들어 걸리는지 본다.
"""

import importlib.util
import pathlib

import pytest

_here = pathlib.Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("bt_golden", _here / "golden.py")
golden = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(golden)


def test_golden_dump_then_check_matches_browsertorch(tmp_path):
    """코어는 **자기 범위만** 대조한다.

    골든은 진짜 torch 로 굳히므로 자매 라이브러리에만 있는 것까지 담는다. 코어는 그것들을
    일부러 거절하므로 건너뛴다 — 두 라이브러리의 범위가 갈리기 시작했고, 건너뛴 수가
    정확히 자매 전용 케이스 수와 같아야 그 갈림이 의도한 대로라는 뜻이다.
    """
    path = tmp_path / "golden.npz"
    count, _ = golden.dump(path)
    assert count > 0, "골든이 비었다 — 케이스 표가 안 실렸다"

    bad, total = golden.check(golden.load_browsertorch(), path)
    assert total == count - len(golden.cases_mod.webgpu_cases())
    assert not bad, "골든과 갈렸다:\n  " + "\n  ".join(bad)


def test_check_rejects_stale_golden(tmp_path, monkeypatch):
    """표가 바뀐 뒤 낡은 골든으로 **통과가 나오면 안 된다.**"""
    path = tmp_path / "golden.npz"
    golden.dump(path)
    monkeypatch.setattr(golden.cases_mod, "manifest_hash", lambda cases: "표가바뀐뒤의해시")
    with pytest.raises(SystemExit, match="낡았다"):
        golden.check(golden.load_browsertorch(), path)


def test_check_rejects_mismatched_inputs(tmp_path, monkeypatch):
    """입력이 갈리면 멈춰야 한다.

    numpy 의 `default_rng` 는 버전이 달라도 같은 수를 주기로 되어 있지만, 그 약속에
    검사를 안 걸면 어긋났을 때 **다른 입력끼리 비교하고 통과 도장을 찍는다.**
    """
    path = tmp_path / "golden.npz"
    golden.dump(path)
    monkeypatch.setattr(golden.cases_mod, "input_fingerprint", lambda inp: "다른입력의지문")
    with pytest.raises(SystemExit, match="입력이 골든과 다르다"):
        golden.check(golden.load_browsertorch(), path)


# ---- 기울기 흐름 표가 제 일을 하는지
#
# 이 표는 값이 아니라 "기울기가 흐르는가"를 묻는다. 그런데 그것을 **묻기만 하고 못
# 잡으면** 아무것도 안 하면서 초록을 하나 더 만드는 것이다. 그래서 조용히 끊기는 두
# 모양을 일부러 만들어 걸리는지 본다 — 실제로 자매에서 둘 다 나왔던 모양이다.

class _Shim:
    """라이브러리 하나를 흉내 내되 이름 하나만 바꿔치기한다."""

    def __init__(self, lib, **swapped):
        self._lib, self._swapped = lib, swapped

    def __getattr__(self, key):
        return self._swapped.get(key) or getattr(self._lib, key)


def _flow_case(name):
    return dict(golden.cases_mod.flow_cases())["flow::" + name]


def test_flow_table_catches_a_severed_graph():
    """맨 텐서를 돌려주는 연산 — `roll` 과 `masked_select` 가 실제로 그랬다."""
    core = golden.load_browsertorch()
    assert _flow_case("roll")(core).startswith("흐름")

    severed = _Shim(core, roll=lambda t, s, dims=None: core.tensor(
        core.roll(t, s, dims).numpy()))
    assert _flow_case("roll")(severed).startswith("안흐름")


def test_flow_table_catches_requires_grad_without_a_gradient():
    """**더 나쁜 쪽.** `requires_grad` 는 True 인데 되짚으면 `.grad` 가 `None` 이다.

    `.float()` 이 정확히 이랬고, `requires_grad` 만 묻는 검사는 이것을 통과시킨다.
    """
    core = golden.load_browsertorch()
    assert _flow_case("sqrt")(core) == "흐름/기울기있음"

    def lying_sqrt(t):
        # 부모를 안 달고 requires_grad 만 켠다 — 옛 `.float()` 과 같은 모양이다.
        return core.Tensor(core.sqrt(t).numpy(), requires_grad=True)

    assert _flow_case("sqrt")(_Shim(core, sqrt=lying_sqrt)) == "흐름/조용히None"
