"""One browser probe at a time on this machine — `tests/browser/launch.py:probe_lock`.

Two sessions ran browser probes at once on 2026-09-06 and neither finished the way it
should have (a headed browser that never launched, a uv cache lock fought over, a nightly
killed by a cleanup). The lock is taken where every probe begins — `serve()` — and this
holds that a second taker waits for the first and says so.
"""
import os
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
LAUNCH = ROOT / "tests" / "browser" / "launch.py"

HOLDER = f"""
import sys, time
sys.path.insert(0, {str(LAUNCH.parent)!r})
import launch
launch.probe_lock("the holder")
print("held", flush=True)
time.sleep(1.5)
"""


def test_a_second_probe_waits_for_the_first_and_says_so(tmp_path, monkeypatch):
    env = {**os.environ, "HOME": str(tmp_path)}          # the lock file lives under ~/.cache
    env.pop("BORCH_NO_PROBE_LOCK", None)
    holder = subprocess.Popen([sys.executable, "-c", HOLDER], env=env, stdout=subprocess.PIPE, text=True)
    assert holder.stdout.readline().strip() == "held"
    t0 = time.time()
    taker = subprocess.run([sys.executable, "-c", HOLDER.replace("the holder", "the taker").replace("time.sleep(1.5)", "")],
                           env=env, capture_output=True, text=True, timeout=30)
    waited = time.time() - t0
    holder.wait(timeout=10)
    assert taker.returncode == 0, taker.stderr
    assert "waiting: the taker" in taker.stdout, taker.stdout
    assert waited >= 1.0, f"the taker did not wait ({waited:.2f}s)"


def test_the_lock_can_be_declined_for_a_machine_nobody_shares(tmp_path):
    env = {**os.environ, "HOME": str(tmp_path), "BORCH_NO_PROBE_LOCK": "1"}
    out = subprocess.run([sys.executable, "-c", HOLDER.replace("time.sleep(1.5)", "")], env=env, capture_output=True, text=True, timeout=30)
    assert out.returncode == 0 and "held" in out.stdout
    assert not (tmp_path / ".cache" / "borch" / "browser-probe.lock").exists()
