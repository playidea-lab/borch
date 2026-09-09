"""Every op class through the compiler, checked against eager.

    npm run build:ts && npm run wheel:py
    uv run --with playwright python tests/browser/compiled_probe.py [--build]

Opens compiled_probe.html: one representative model per op class (elementwise,
matmul, reductions, softmax, layernorm, conv, shape/view, cat) run through
torch.compiled(step, check=True) under both SGD (the arena path) and Adam, and the
compiled two-step loss curve diffed against a fresh eager one. This is the net for
the class the SGD-arena bug fell through — an op correct eager but wrong under
capture/fuse — lifted from the ~4 hand-built models to the whole op surface.
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
    if refuse_if_screen_off("the captured step"):
        return 1
    port, shutdown = serve(ROOT)
    url = f"http://127.0.0.1:{port}/tests/browser/compiled_probe.html?wheel=/{wheel}"
    profile = tempfile.mkdtemp(prefix="borch-compiled-")
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
    if refuse_if_software(adapter, "the captured step"):
        return 1
    m = re.search(r"compiled: (\d+)/(\d+) ok", done)
    ok = ("faults 0" in done and "FAILS" not in done and "ERR" not in done
          and bool(m) and m.group(1) == m.group(2))
    print("**every op class compiles to its eager self — plain bit-for-bit, fused within tol**" if ok
          else "**an op class miscompiles** — see above")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
