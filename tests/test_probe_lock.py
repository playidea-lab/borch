"""`tests/browser/launch.py:probe_lock` — one browser probe per machine, but never a process
waiting on itself. Three runners load `launch.py` by path, so the module-level guard is one
per copy; the environment carries the holder's pid across copies."""
import importlib.util
import os
import pathlib
import subprocess
import sys

import pytest

LAUNCH = pathlib.Path(__file__).resolve().parents[1] / "tests" / "browser" / "launch.py"
CHILD = """
import fcntl, os, sys, time
fd = os.open(sys.argv[1], os.O_RDWR | os.O_CREAT, 0o644)
fcntl.flock(fd, fcntl.LOCK_EX)
os.write(fd, str(os.getpid()).encode())
print("held", flush=True)
time.sleep(float(sys.argv[2]))
"""


def _fresh_copy(lock_path):
    spec = importlib.util.spec_from_file_location(f"bt_launch_{id(lock_path)}_{os.urandom(2).hex()}", LAUNCH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.LOCK_PATH = lock_path
    return mod


@pytest.fixture
def lock_env(tmp_path, monkeypatch):
    monkeypatch.delenv("BORCH_PROBE_LOCK_PID", raising=False)
    monkeypatch.delenv("BORCH_NO_PROBE_LOCK", raising=False)
    return tmp_path / "probe.lock"


def test_probe_lock_taken_twice_by_two_copies_in_one_process_returns_at_once(lock_env):
    first, second = _fresh_copy(lock_env), _fresh_copy(lock_env)
    first.probe_lock()
    assert lock_env.read_text() == str(os.getpid())
    assert os.environ["BORCH_PROBE_LOCK_PID"] == str(os.getpid())
    second.probe_lock()  # 09-08: this blocked forever, waiting on our own pid
    assert second._lock_fd is None, "the second copy must not open a descriptor of its own"


def test_probe_lock_with_a_dead_holder_pid_in_the_environment_still_takes_the_lock(lock_env, monkeypatch):
    dead = subprocess.run([sys.executable, "-c", "import os; print(os.getpid())"], capture_output=True, text=True, check=True).stdout.strip()
    monkeypatch.setenv("BORCH_PROBE_LOCK_PID", dead)
    mod = _fresh_copy(lock_env)
    mod.probe_lock()
    assert mod._lock_fd is not None and lock_env.read_text() == str(os.getpid())


def test_probe_lock_held_by_a_live_ancestor_pid_is_not_taken_again(lock_env, monkeypatch):
    child = subprocess.Popen([sys.executable, "-c", CHILD, str(lock_env), "30"], stdout=subprocess.PIPE, text=True)
    try:
        assert child.stdout.readline().strip() == "held"
        monkeypatch.setenv("BORCH_PROBE_LOCK_PID", str(child.pid))  # as if that process were our parent
        mod = _fresh_copy(lock_env)
        mod.probe_lock()  # would block until the child exits if the pid were not honoured
        assert mod._lock_fd is None
    finally:
        child.kill(); child.wait()
