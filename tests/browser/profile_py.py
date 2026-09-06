"""Where a training step's time goes — **GPU timestamps per kind of kernel**, on the wheel,
in a worker, on this tab's GPU.

    uv run --with playwright python tests/browser/profile_py.py [--build] [--batch=16] [--size=96] [--headless]

The U-Net of tests/seg_eval.py on synthetic data (no dataset: this runs in the nightly's
worktree). It prints the wall-clock step, the dispatch count, and the device profiler's
per-kind table — the instrument that took a step from 26 to 18 ms in two days
(2026-09-06/07), each change kept only when this table moved. **A measurement, not a
check**: it judges nothing beyond `faults 0` and a table having come out, and it refuses
a software adapter, where every number is the CPU's.

Single-op microtimings were tried first and abandoned — they swung twofold between runs
on this machine; the wall-clock step and this table are what to trust.
"""
import glob
import os
import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from first_run import FLAGS, ROOT, refuse_if_screen_off, serve  # noqa: E402
from launch import _headed, refuse_if_software  # noqa: E402
from wheel_probe import wheel_is_stale  # noqa: E402

GIVE_UP_MS = 10 * 60 * 1000


def main(argv):
    from playwright.sync_api import sync_playwright
    headed = _headed("--headed" in argv)
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
    batch = next((a.split("=", 1)[1] for a in argv if a.startswith("--batch=")), "16")
    size = next((a.split("=", 1)[1] for a in argv if a.startswith("--size=")), "96")
    if refuse_if_screen_off("the step profile"):
        return 1
    port, shutdown = serve(ROOT)
    url = f"http://127.0.0.1:{port}/tests/browser/profile_py.html?wheel=/{wheel}&batch={batch}&size={size}"
    profile = tempfile.mkdtemp(prefix="borch-profile-")
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
    done = got.get("done") or ""
    adapter = done.rsplit("adapter ", 1)[-1].split(" ·")[0] if "adapter " in done else None
    if refuse_if_software(adapter, "the step profile"):
        return 1
    ok = "faults 0" in done and " ms  " in done
    print("**the table above is where the step's GPU time goes**" if ok else "**no table** — see above")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
