"""`torch.hub.load` on the wheel, in a worker — a catalogue model in Python.

    uv run --with playwright python tests/browser/hub_py.py [--headed]

The same shape as wheel_probe.py: Pyodide in a worker, the wheel from this checkout, and
the page reports what Python saw. Judged here: the model loaded, its output is (2, 1000),
and no validation fault. Network: the registry and the 21 MB of weights (cached after).

Was:
Does the wheel alone, installed into a Pyodide that runs in a Web Worker, train and export?

    uv run --with playwright python tests/browser/wheel_probe.py [--headed] [--wheel=dist/pyborch-X.whl]

JupyterLite's shape: no page-side borch.ts, Pyodide in a worker, the wheel is the
only thing that arrives. `borch_webgpu` boots its own borch.ts from the bundle it
carries (`_borch.js`, `npm run bundle:py`), so `import borch_webgpu as torch` is the
whole setup. The default wheel is the newest under `dist/` (`uv build --wheel`).
"""
import glob
import os
import sys
import tempfile

from first_run import FLAGS, ROOT, refuse_if_screen_off, serve
from launch import refuse_if_software

GIVE_UP_MS = 5 * 60 * 1000


def main(argv):
    from playwright.sync_api import sync_playwright
    headed = "--headed" in argv
    wheel = next((a.split("=", 1)[1] for a in argv if a.startswith("--wheel=")), None)
    if "--build" in argv:
        import subprocess
        for cmd in (["npm", "run", "-s", "bundle:py"], ["uv", "build", "--wheel", "-q"]):
            r = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
            if r.returncode:
                print(f"{' '.join(cmd)} failed:\n{(r.stdout + r.stderr)[-800:]}", file=sys.stderr)
                return 2
    if wheel is None:
        found = sorted(glob.glob(str(ROOT / "dist" / "pyborch-*.whl")))
        if not found:
            print("no wheel under dist/ — first: npm run bundle:py && uv build --wheel", file=sys.stderr)
            return 2
        wheel = os.path.relpath(found[-1], ROOT)
    if refuse_if_screen_off("the wheel in a worker"):
        return 1
    port, shutdown = serve(ROOT)
    url = f"http://127.0.0.1:{port}/tests/browser/hub_py.html?wheel=/{wheel}"
    profile = tempfile.mkdtemp(prefix="borch-wheel-")
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
    print(f"wheel: {wheel}")
    print(got["text"])
    if got.get("error"):
        print("error: " + got["error"][:600])
    adapter = None
    for line in got["text"].splitlines():
        if "adapter " in line:
            adapter = line.split("adapter ", 1)[1].split(" ·")[0]
    if refuse_if_software(adapter, "the wheel in a worker"):
        return 1
    done = got.get("done") or ""
    # Judged here, not in the page: y = 3x + 1 learned to two decimals, and an ONNX file.
    ok = "hub imagenet-efficientnet-b0 loaded" in done and "out (2, 1000)" in done and "faults 0" in done
    print("**torch.hub.load works on the wheel, in a worker**" if ok else "**it did not** — see above")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
