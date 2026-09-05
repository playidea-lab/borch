"""The trajectory golden on the wheel, in a worker — torch's two training runs, step by
step, through the binding on this tab's GPU.

    uv run --with playwright python tests/browser/trajectory_py.py [--headed] [--build]

The same shape as wheel_probe.py: Pyodide in a worker, the wheel from this checkout. The
page fetches tests/trajectory.py, the frozen curves and the initial weights from this
checkout, runs both recipes on the wheel, and reports the largest relative loss deviation
and the prediction agreement. Judged here: every step within one part in a hundred of
torch's, predictions identical, no validation fault.
"""
import glob
import os
import pathlib
import re
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from first_run import FLAGS, ROOT, refuse_if_screen_off, serve  # noqa: E402
from launch import refuse_if_software  # noqa: E402
from wheel_probe import wheel_is_stale  # noqa: E402

GIVE_UP_MS = 10 * 60 * 1000
TOL = 1e-2


def main(argv):
    from playwright.sync_api import sync_playwright
    headed = "--headed" in argv
    if "--build" in argv:
        import subprocess
        for cmd in (["npm", "run", "-s", "bundle:py"], ["uv", "build", "--wheel", "-q"]):
            r = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
            if r.returncode:
                print(f"{' '.join(cmd)} failed:\n{(r.stdout + r.stderr)[-800:]}", file=sys.stderr)
                return 2
    found = sorted(glob.glob(str(ROOT / "dist" / "pyborch-*.whl")))
    if not found:
        print("no wheel under dist/ — run with --build", file=sys.stderr)
        return 2
    wheel = os.path.relpath(found[-1], ROOT)
    if wheel_is_stale(wheel):
        print(f"{wheel} is older than the sources — run with --build", file=sys.stderr)
        return 2
    if refuse_if_screen_off("the trajectory golden"):
        return 1
    port, shutdown = serve(ROOT)
    url = f"http://127.0.0.1:{port}/tests/browser/trajectory_py.html?wheel=/{wheel}"
    profile = tempfile.mkdtemp(prefix="borch-trajectory-")
    channel = os.environ.get("BORCH_CHROME_CHANNEL") or None
    try:
        with sync_playwright() as pw:
            context = pw.chromium.launch_persistent_context(profile, headless=not headed, channel=channel, args=list(FLAGS), timeout=60_000)
            try:
                page = context.new_page()
                page.goto(url, wait_until="load")
                page.wait_for_function("window.__wheel !== undefined", timeout=GIVE_UP_MS, polling=200)
                got = page.evaluate("window.__wheel")
            finally:
                context.close()
    finally:
        shutdown()
    print(got["text"])
    if got.get("error"):
        print("error: " + got["error"][:600])
        return 1
    adapter = None
    for line in got["text"].splitlines():
        if "adapter " in line:
            adapter = line.split("adapter ", 1)[1].split(" ·")[0]
    if refuse_if_software(adapter, "the trajectory golden"):
        return 1
    done = got.get("done") or ""
    worsts = [float(v) for v in re.findall(r"worst ([0-9.e+-]+) at step", done)]
    agrees = re.findall(r"pred agree (\d+)%", done)
    ok = len(worsts) == 2 and all(w < TOL for w in worsts) and agrees == ["100", "100"] and "faults 0" in done
    print("**the wheel walks torch's training curves**" if ok else "**it did not** — see above")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
