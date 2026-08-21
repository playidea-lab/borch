"""Runs the integration scenario on both sides and compares the numbers."""

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
    """The same code with only the import changed. The numbers that come out have to agree
    **within tolerance.**

    It passes through MLP training, a CNN (with BatchNorm), an LSTM, a transformer encoder,
    and saving and loading, in one run.

    They are not compared as strings. Bit equality is an explicit non-goal (ROADMAP), and
    macOS's and Linux's BLAS did once diverge at the sixth decimal and turn CI red on its
    own.
    """
    real_out, nano_out = _run("real"), _run("nano")
    assert real_out, "the scenario produced no values at all"
    for key, value in real_out.items():
        assert key in nano_out, f"{key} is absent on the borch side"
        expected, got = float(value), float(nano_out[key])
        assert abs(expected - got) <= TOLERANCE * max(1.0, abs(expected)), (
            f"{key} diverged — torch {expected} · borch {got}")
