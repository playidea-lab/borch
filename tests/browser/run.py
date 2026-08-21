"""Compares the golden answers in a browser — running **on a machine with a real GPU.**

A GitHub-hosted runner has no GPU and falls back to SwiftShader, which is a path no user
walks. So this script runs on a self-hosted runner or a development machine.

    uv run --with playwright python tests/browser/run.py
    uv run --with playwright python tests/browser/run.py --headed --lib borch_webgpu

The golden answers have to exist first:

    uv run --with numpy --with torch python tests/golden.py dump

`file://` will not do — the runner fetches the sources and the golden answers, so a server is
needed. The repository root is put on a temporary port here.
"""

import argparse
import functools
import http.server
import importlib.util
import pathlib
import socketserver
import sys
import threading

from launch import browser as browser_of

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent

_vspec = importlib.util.spec_from_file_location(
    "bt_vendor", pathlib.Path(__file__).resolve().parent / "vendor.py")
vendor = importlib.util.module_from_spec(_vspec)
_vspec.loader.exec_module(vendor)
GOLDEN = ROOT / "tests" / "golden.npz"
# Downloading Pyodide and numpy is slow on the first run, and the work itself grows with the
# case count. **Running short on time and never finishing are different things** — set tight,
# it reads slow as stopped and sends people chasing a defect that is not there. The sister
# runner is at 600 seconds for the same reason.
TIMEOUT_MS = 600_000


class _Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a):
        pass


def serve(root):
    """Puts the repository root on a temporary port and returns (port, shutdown)."""
    handler = functools.partial(_Quiet, directory=str(root))
    httpd = socketserver.ThreadingTCPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd.server_address[1], httpd.shutdown


def run(lib, headed, probe=None):
    from playwright.sync_api import sync_playwright

    # **Whether the emitted files are older than the sources is looked at first.** A stale
    # `dist` comes out with the same wording as a real gap — two copies of a check means only
    # one gets fixed, so the TS runner's is borrowed.
    sys.path.insert(0, str(ROOT / "borch-ts" / "test"))
    from run import require_fresh_dist, require_fresh_golden   # noqa: PLC0415
    require_fresh_dist(ROOT)
    # The golden answers are the same trap — editing `cases.py` without dumping makes a new
    # case come out as "that name is not in the golden answers", the same wording as a typo.
    require_fresh_golden(ROOT)

    port, stop = serve(ROOT)
    url = f"http://127.0.0.1:{port}/tests/browser/runner.html?lib={lib}"
    probed = None
    try:
        # **Closing the browser is the `with`'s job too** — put on the last line, an exception
        # before it leaves it open, and the leftover Chromium damages another measurement.
        with sync_playwright() as p, browser_of(p, headed=headed) as browser:
            # **This runner had no flags at all until now.** On macOS WebGPU came out anyway
            # so it went unseen, but a Linux GPU server needs Vulkan turned on. Measured under
            # conditions different from the borch.ts runner's, it stops being one yardstick.
            page = browser.new_page()
            # An accuracy run takes minutes. Killed by the default limit, what was measured is lost.
            page.set_default_timeout(0)
            # Errors always go out, and so does anything starting with `[bench]` — a long
            # measurement that dies partway loses its return value whole, so what was let out
            # along the way is all that is left of what had been measured.
            page.on("console", lambda m: print(f"  [browser] {m.text}")
                    if m.type == "error" or m.text.startswith("[bench]") else None)
            page.goto(url)
            page.wait_for_function("window.GOLDEN_RESULT !== null", timeout=TIMEOUT_MS)
            result = page.evaluate("window.GOLDEN_RESULT")
            if probe:
                # A channel for seeing what only reproduces inside the browser.
                probed = page.evaluate(
                    "async (code) => String(await window.PY.runPythonAsync(code))", probe)
    finally:
        stop()
    return result, probed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", default="borch", help="the library to compare")
    ap.add_argument("--headed", action="store_true",
                    help="opens a window. **WebGPU does not come up headless** (measured — it falls back to WebGL)")
    ap.add_argument("--probe", help="Python to run inside the browser after the comparison. For debugging")
    ap.add_argument("--bench", action="store_true",
                    help="measures a real training step with ResNet-18 (tests/browser/bench.py)")
    ap.add_argument("--cost", action="store_true",
                    help="counts one step's **cost** — dispatches and buffers rather than time, "
                         "so it means something on a software adapter too (tests/browser/cost.py)")
    ap.add_argument("--accuracy", action="store_true",
                    help="measures **accuracy** — the augmented side and the unaugmented side "
                         "next to each other. cifar-batch1.bin and cifar-batch-test.bin have to "
                         "be at the repository root")
    ap.add_argument("--epochs", type=int, default=6, help="the epoch count for --accuracy")
    ap.add_argument("--images", type=int, default=0,
                    help="a cap on the images --accuracy uses (0 for all). For checking a machine")
    ap.add_argument("--augment", choices=("on", "off"),
                    help="runs one condition only. **Splitting them is the better experiment** — "
                         "run back to back in one session, the second model's initial weights "
                         "differ and mix into augmentation's effect")
    args = ap.parse_args()
    if args.bench and not args.probe:
        args.probe = (f"import bench, importlib\n"
                      f"L = importlib.import_module({args.lib!r})\n"
                      f"bench.report(L)")
    if args.cost and not args.probe:
        args.probe = (f"import cost, importlib\n"
                      f"L = importlib.import_module({args.lib!r})\n"
                      f"cost.report(L)")
    if args.accuracy and not args.probe:
        # The test data has to be **what training did not use.** So the original archive's
        # test_batch is taken out separately — splitting one chunk measures something other
        # than accuracy.
        args.probe = (
            f"import bench, importlib\n"
            f"L = importlib.import_module({args.lib!r})\n"
            # Augmentation's draws get a seed too. Otherwise the same command gives a different
            # answer each time, and the difference between conditions cannot be told from the
            # difference between draws.
            f"import borchvision as V; V.use(L); V.manual_seed(0)\n"
            f"tr = await bench.cifar_from(L, '/cifar-batch1.bin', 'cifar-batch1.bin')\n"
            f"te = await bench.cifar_from(L, '/cifar-batch-test.bin', 'cifar-test.bin')\n"
            f"cap = {args.images}\n"
            f"tr = (tr[0][:cap], tr[1][:cap]) if cap else tr\n"
            f"te = (te[0][:cap], te[1][:cap]) if cap else te\n"
            f"only = {None if args.augment is None else args.augment == 'on'!r}\n"
            f"await bench.report_accuracy(L, tr, te, epochs={args.epochs}, only=only)")

    if not GOLDEN.exists():
        print(f"no golden answers: {GOLDEN}\n"
              "  first: uv run --with numpy --with torch python tests/golden.py dump")
        return 1

    # Pyodide comes from local files. Fetched once if absent, compared by hash if present.
    vendor.ensure()

    result, probed = run(args.lib, args.headed, args.probe)
    if probed is not None:
        print("-- probe --")
        print(probed)
        print()
    total, bad = result["total"], result["bad"]
    if result.get("error"):
        print("the runner blew up:\n" + result["error"])
        return 1
    print(f"browser golden comparison ({result.get('lib', args.lib)}) — {total} cases")
    print(f"  agreeing {total - len(bad)}/{total}")
    # **A validation error does not arrive as an exception.** If even one happened, the green
    # above is that much less trustworthy — an invalid command buffer quietly does nothing while
    # the wall clock keeps running.
    if result.get("faults"):
        print(f"  {result['faults']} GPU validation errors")
    if bad:
        print("\nwhere it diverged:")
        for why in bad:
            print(f"  ✗ {why}")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
