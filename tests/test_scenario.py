"""통합 시나리오를 양쪽에서 돌려 숫자를 대조한다."""

import pathlib
import subprocess
import sys


def _run(lib):
    out = subprocess.run(
        [sys.executable, str(pathlib.Path(__file__).parent / "scenario.py"), lib],
        capture_output=True, text=True, check=True).stdout
    return dict(line.split("\t") for line in out.strip().splitlines() if "\t" in line)


TOLERANCE = 1e-4


def test_scenario_matches_real_torch():
    """같은 코드, 임포트만 다르게. 나온 숫자가 **허용 오차 안에서** 같아야 한다.

    MLP 학습·CNN(BatchNorm 포함)·LSTM·트랜스포머 인코더·저장과 불러오기를 한 번에 지난다.

    문자열로 견주지 않는다. 비트 동등은 명시적 비목표이고(ROADMAP), 실제로 맥과 리눅스의
    BLAS 가 소수점 여섯 자리에서 갈려 CI 만 빨갛게 뜬 적이 있다.
    """
    real_out, nano_out = _run("real"), _run("nano")
    assert real_out, "시나리오가 아무 값도 안 냈다"
    for key, value in real_out.items():
        assert key in nano_out, f"{key} 가 nanotorch 쪽에 없다"
        expected, got = float(value), float(nano_out[key])
        assert abs(expected - got) <= TOLERANCE * max(1.0, abs(expected)), (
            f"{key} 가 갈렸다 — torch {expected} · nanotorch {got}")
