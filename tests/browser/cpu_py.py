"""`borch_cpu` on the wheel, in a worker, with no device brought up — and against the GPU where there is one.

    uv run --with playwright python tests/browser/cpu_py.py [--headed] [--build] [--wheel=dist/pyborch-X.whl]

The wheel is installed into a Pyodide worker and `import borch_cpu` is the whole setup:
no `init()`, no adapter. It loads EfficientNet-B0 through borch-hub, runs six images
through the frozen backbone on the wasm kernels, fits a linear head, scores the labels.
Where the page also has an adapter, the same six images go through `borch_webgpu`'s
`torch.hub` model and the features are compared — that is the check that the Python
door on the CPU side answers what the WebGPU side answers. Without an adapter the
comparison is reported as not available and the rest still has to hold.

Needs the network for the checkpoint (21 MB, cached by the hub after the first time).
"""
import glob
import json
import os
import subprocess
import sys
import tempfile

from first_run import FLAGS, ROOT, serve
from wheel_probe import wheel_is_stale

GIVE_UP_MS = 10 * 60 * 1000


def main(argv):
    from playwright.sync_api import sync_playwright
    headed = "--headed" in argv
    wheel = next((a.split("=", 1)[1] for a in argv if a.startswith("--wheel=")), None)
    if "--build" in argv:
        for cmd in (["npm", "run", "-s", "bundle:py"], ["uv", "build", "--wheel", "-q"]):
            r = subprocess.run(cmd, cwd=ROOT)
            if r.returncode:
                return r.returncode
    if wheel is None:
        found = sorted(glob.glob(str(ROOT / "dist" / "pyborch-*.whl")), key=os.path.getmtime)
        if not found:
            print("no wheel under dist/ — first: npm run bundle:py && uv build --wheel", file=sys.stderr)
            return 2
        wheel = os.path.relpath(found[-1], ROOT)
        if wheel_is_stale(wheel):
            print(f"{wheel} is older than the sources — run with --build (or: npm run bundle:py && uv build --wheel)", file=sys.stderr)
            return 2
    port, shutdown = serve(ROOT)
    url = f"http://127.0.0.1:{port}/tests/browser/cpu_py.html?wheel=/{wheel}"
    profile = tempfile.mkdtemp(prefix="borch-cpu-py-")
    channel = os.environ.get("BORCH_CHROME_CHANNEL") or None
    try:
        with sync_playwright() as pw:
            context = pw.chromium.launch_persistent_context(profile, headless=not headed, channel=channel, args=list(FLAGS), timeout=60_000)
            try:
                page = context.new_page()
                page.goto(url, wait_until="load")
                page.wait_for_function("window.__cpuPy !== undefined", timeout=GIVE_UP_MS, polling=500)
                got = page.evaluate("window.__cpuPy")
            finally:
                context.close()
    finally:
        shutdown()
    print(f"wheel: {wheel}")
    print(got["text"])
    if got.get("error"):
        print("error: " + got["error"][:800])
        return 1
    r = json.loads(got["done"])
    # Judged here: the module is available, the backbone is B0 with 1280 features, the six
    # rows are finite, the head learned its six rows, the swapped-label score is a number
    # per row, and — where a GPU answered — the two sides agree to 1e-3 relative.
    ok = r["available"] and r["backbone"] == "timm/efficientnet_b0" and r["num_features"] == 1280
    ok = ok and r["feats_shape"] == [6, 1280] and r["finite"] and r["lossN"] < r["loss0"] and r["head_shape"] == [3, 1280]
    ok = ok and len(r["doubt"]) == 6
    gap = r["gpu_gap"]
    if isinstance(gap, float):
        print(f"cpu vs gpu features: relative {gap:.2e}")
        ok = ok and gap < 1e-3
    else:
        print(f"cpu vs gpu features: {gap}")
    print(f"borch_cpu: import {r['boot_ms']} ms · load {r['load_ms']} ms · six images {r['feat_ms']} ms · head loss {r['loss0']} → {r['lossN']} · fits {r['fit_on_train']}/6")
    print("**borch_cpu runs the frozen backbone, the head and the score with no device**" if ok else "**it did not** — see above")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
