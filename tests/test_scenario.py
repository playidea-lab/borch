"""통합 시나리오를 양쪽에서 돌려 숫자를 대조한다."""

import pathlib
import subprocess
import sys


def _run(lib):
    out = subprocess.run(
        [sys.executable, str(pathlib.Path(__file__).parent / "scenario.py"), lib],
        capture_output=True, text=True, check=True).stdout
    return dict(line.split("\t") for line in out.strip().splitlines() if "\t" in line)


def test_scenario_matches_real_torch():
    """같은 코드, 임포트만 다르게. 나온 숫자가 같아야 한다.

    MLP 학습·CNN(BatchNorm 포함)·LSTM·트랜스포머 인코더·저장과 불러오기를 한 번에 지난다.
    """
    real_out, nano_out = _run("real"), _run("nano")
    assert real_out, "시나리오가 아무 값도 안 냈다"
    for key, value in real_out.items():
        assert key in nano_out, f"{key} 가 nanotorch 쪽에 없다"
        assert value == nano_out[key], (
            f"{key} 가 갈렸다 — torch {value} · nanotorch {nano_out[key]}")
