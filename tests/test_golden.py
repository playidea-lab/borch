"""골든 하네스 자체가 제 일을 하는지 본다.

GPU 백엔드는 아직 없다. 그래서 borch 를 **제3의 라이브러리인 척** 골든에
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


def test_golden_dump_then_check_matches_borch(tmp_path):
    """코어는 **자기 범위만** 대조한다.

    골든은 진짜 torch 로 굳히므로 자매 라이브러리에만 있는 것까지 담는다. 코어는 그것들을
    일부러 거절하므로 건너뛴다 — 두 라이브러리의 범위가 갈리기 시작했고, 건너뛴 수가
    정확히 자매 전용 케이스 수와 같아야 그 갈림이 의도한 대로라는 뜻이다.
    """
    path = tmp_path / "golden.npz"
    count, _ = golden.dump(path)
    assert count > 0, "골든이 비었다 — 케이스 표가 안 실렸다"

    bad, total = golden.check(golden.load_borch(), path)
    assert total == count - len(golden.cases_mod.webgpu_cases())
    assert not bad, "골든과 갈렸다:\n  " + "\n  ".join(bad)


def test_check_rejects_stale_golden(tmp_path, monkeypatch):
    """표가 바뀐 뒤 낡은 골든으로 **통과가 나오면 안 된다.**"""
    path = tmp_path / "golden.npz"
    golden.dump(path)
    monkeypatch.setattr(golden.cases_mod, "manifest_hash", lambda cases: "표가바뀐뒤의해시")
    with pytest.raises(SystemExit, match="낡았다"):
        golden.check(golden.load_borch(), path)


def test_check_rejects_mismatched_inputs(tmp_path, monkeypatch):
    """입력이 갈리면 멈춰야 한다.

    numpy 의 `default_rng` 는 버전이 달라도 같은 수를 주기로 되어 있지만, 그 약속에
    검사를 안 걸면 어긋났을 때 **다른 입력끼리 비교하고 통과 도장을 찍는다.**
    """
    path = tmp_path / "golden.npz"
    golden.dump(path)
    monkeypatch.setattr(golden.cases_mod, "input_fingerprint", lambda inp: "다른입력의지문")
    with pytest.raises(SystemExit, match="입력이 골든과 다르다"):
        golden.check(golden.load_borch(), path)


def test_check_names_the_case_that_raised_a_gpu_fault(tmp_path):
    """**GPU 검증 오류를 낸 케이스를 이름으로 짚는다.**

    WebGPU 는 그것을 예외로 안 던진다. 무효한 명령 버퍼는 조용히 아무것도 안 하므로
    **범인은 통과하고 뒤에 줄 선 케이스가 대신 빨개진다** — 세 번 겪었다(`as_strided_`
    의 초과 복사, 옵티마이저 상태의 버퍼 공유, 아무것도 안 고르는 `index_select`).

    브라우저 없이 그 배선만 확인한다. 여기서 세는 사람이 가짜 계수기를 들고 있어도
    **어느 케이스에서 늘었는지**를 답에 적어야 한다 — 안 적으면 러너에 수 하나가
    더 찍힐 뿐이고, 그것으로는 여전히 원인에서 한 칸 떨어진 자리를 보게 된다.
    """
    path = tmp_path / "golden.npz"
    golden.dump(path)

    # 세 번째 케이스에서 딱 한 번 오류가 난 척한다.
    state = {"n": 0, "seen": 0}

    def counter():
        state["seen"] += 1
        if state["seen"] == 3:
            state["n"] += 1
        return state["n"]

    bad, _ = golden.check(golden.load_borch(), path, faults=counter)
    hits = [line for line in bad if "GPU 검증 오류" in line]
    assert len(hits) == 1, f"한 건이어야 하는데 {len(hits)} 건이다:\n  " + "\n  ".join(bad)

    # **어느 케이스인지가 요점이다.** 첫 호출은 루프 밖(기준선)이므로, 세 번째 호출은
    # 케이스 표의 두 번째 케이스가 끝난 자리다.
    names = [n for n, _ in golden.cases_mod.golden_cases()]
    assert hits[0].startswith(names[1] + ":"), (
        f"{names[1]} 을 짚어야 하는데: {hits[0]}")


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
    core = golden.load_borch()
    assert _flow_case("roll")(core).startswith("흐름")

    severed = _Shim(core, roll=lambda t, s, dims=None: core.tensor(
        core.roll(t, s, dims).numpy()))
    assert _flow_case("roll")(severed).startswith("안흐름")


def test_flow_table_catches_requires_grad_without_a_gradient():
    """**더 나쁜 쪽.** `requires_grad` 는 True 인데 되짚으면 `.grad` 가 `None` 이다.

    `.float()` 이 정확히 이랬고, `requires_grad` 만 묻는 검사는 이것을 통과시킨다.
    """
    core = golden.load_borch()
    assert _flow_case("sqrt")(core) == "흐름/기울기있음"

    def lying_sqrt(t):
        # 부모를 안 달고 requires_grad 만 켠다 — 옛 `.float()` 과 같은 모양이다.
        return core.Tensor(core.sqrt(t).numpy(), requires_grad=True)

    assert _flow_case("sqrt")(_Shim(core, sqrt=lying_sqrt)) == "흐름/조용히None"
