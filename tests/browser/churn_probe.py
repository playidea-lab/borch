"""The pool, poisoned between the runs — does any op read what it did not write.

    npm run build:ts && npm run wheel:py
    uv run --with playwright python tests/browser/churn_probe.py [--build]

Opens churn_probe.html: for each op whose backward scatters a gradient into a
full-size buffer (embedding, index_select, narrow, select, amax) the identical
graph runs once on a pool poisoned with zeros and once on a pool poisoned with a
loud sentinel; the real step then runs on the poisoned pool and its gradient must
be the same both times, bit for bit. Plus a constant-pad border and a fresh
zeros() that must read zero even from a sentinel-filled pool. The standing net
for the "a pooled alloc is not zero-initialised" class.
"""
import glob
import os
import pathlib
import re
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
    if refuse_if_screen_off("the poisoned pool"):
        return 1
    port, shutdown = serve(ROOT)
    url = f"http://127.0.0.1:{port}/tests/browser/churn_probe.html?wheel=/{wheel}"
    profile = tempfile.mkdtemp(prefix="borch-churn-")
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
    if refuse_if_software(adapter, "the poisoned pool"):
        return 1
    worst = re.search(r"worst ([0-9.e+-]+)", done)
    # The control must say PROVEN — otherwise the poison never reached the allocations and a
    # green result would mean nothing.
    ok = ("faults 0" in done and "ERR" not in done and "PROVEN" in done and bool(worst)
          and float(worst.group(1)) <= 1e-6)
    print("**no op reads what it did not write — a poisoned pool changes nothing**" if ok
          else "**an op depends on the pool being zero** (or the poison did not land) — see above")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
