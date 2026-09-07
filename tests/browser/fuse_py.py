"""The fusion pass on a captured step — **a hand-written network, its trees as kernels.**

    uv run --with playwright python tests/browser/fuse_py.py [--build] [--headless]

Ten steps of a network nobody fused by hand — GELU written out from tanh, a LayerNorm from
means and a square root, a weighted squared loss — eagerly; then from the same weights with
the first step recorded and the rest replayed, once as recorded and once after
`capture.fuse()` merged the elementwise trees. Judged: the plain replay is the eager run
exactly; the fused replay within 1e-5 relative on every loss and parameter (a fused
expression lets the shader compiler contract a multiply and an add into one rounding,
which the separate kernels could not — the difference is that rounding); fewer dispatches
fused than plain; no fault. Reported: the three step times. On the M4 Max: eager 5.0 ms,
plain 0.85, fused 0.70 — a small network is Python's, not the GPU's.
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
    batch = next((a.split("=", 1)[1] for a in argv if a.startswith("--batch=")), "16")
    size = next((a.split("=", 1)[1] for a in argv if a.startswith("--size=")), "96")
    if refuse_if_screen_off("the fused step"):
        return 1
    port, shutdown = serve(ROOT)
    url = f"http://127.0.0.1:{port}/tests/browser/fuse_py.html?wheel=/{wheel}&batch={batch}&size={size}"
    profile = tempfile.mkdtemp(prefix="borch-fuse-")
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
    if refuse_if_software(adapter, "the fused step"):
        return 1
    plain = re.search(r"plain replay (\d+) dispatches rel\|Δloss\| ([0-9.e+-]+) rel\|Δparam\| ([0-9.e+-]+)", done)
    fused = re.search(r"fused replay (\d+) → (\d+) dispatches rel\|Δloss\| ([0-9.e+-]+) rel\|Δparam\| ([0-9.e+-]+)", done)
    ok = "faults 0" in done and bool(plain and fused)
    ok = ok and float(plain.group(2)) == 0.0 and float(plain.group(3)) == 0.0
    ok = ok and float(fused.group(3)) <= 1e-5 and float(fused.group(4)) <= 1e-5 and int(fused.group(2)) < int(fused.group(1))
    print("**the fused replay is the eager run to a rounding, in fewer kernels**" if ok else "**it is not** — see above")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
