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
    path = tmp_path / "golden.npz"
    count, _ = golden.dump(path)
    assert count > 0, "골든이 비었다 — 케이스 표가 안 실렸다"

    bad, total = golden.check(golden.load_browsertorch(), path)
    assert total == count
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
