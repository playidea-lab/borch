"""Compares the golden answers in a browser — running **on a machine with a real GPU.**

A GitHub-hosted runner has no GPU and falls back to SwiftShader, which is a path no user
walks. So this script runs on a self-hosted runner or a development machine.

    uv run --with playwright python tests/browser/run.py --lib borch
    uv run --with playwright python tests/browser/run.py --headed --lib borch_webgpu

The golden answers have to exist first:

    uv run --with numpy --with torch python tests/golden.py dump

**`--lib` has no default, on purpose.** It used to default to `borch` — the numpy core,
running inside a browser. That is a real comparison and it is almost always green,
because the core is the layer both goldens already cover. `borch_webgpu` is the binding,
and the binding is the layer *between* the two things everything else checks.

Run it bare while hunting a binding defect and it answered `3255/3255`, printed `(borch)`
in a header nobody reads when they are looking for a number, and was completely correct
about a question nobody asked. That happened: a `_SCHED_ARGS` row was fixed, this was
run without the flag, and the green was reported as proof of the fix. What caught it was
putting the defect back and running again — **the same number returned.**

So the name of the library now comes back in the summary line whether or not anyone
looks, and there is nothing to forget: leaving `--lib` off stops rather than measuring
the layer that is always green.

`file://` will not do — the runner fetches the sources and the golden answers, so a server is
needed. The repository root is put on a temporary port here.
"""

import argparse
import functools
import http.server
import os
import importlib.util
import pathlib
import socketserver
import sys
import threading

from launch import browser as browser_of, is_software

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

    def end_headers(self):
        # The same two headers as site/serve.py: the browser tests run cross-origin isolated,
        # so a resource that would break the deployed page under COEP breaks here first.
        # BORCH_NO_COI=1 leaves them out — the way to tell a page that broke under isolation
        # from a page that broke anyway.
        if not os.environ.get("BORCH_NO_COI"):
            self.send_header("Cross-Origin-Opener-Policy", "same-origin")
            self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        super().end_headers()


def serve(root):
    """Puts the repository root on a temporary port and returns (port, shutdown)."""
    handler = functools.partial(_Quiet, directory=str(root))
    httpd = socketserver.ThreadingTCPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd.server_address[1], httpd.shutdown


def run(lib, headed, probe=None, js=None):
    from playwright.sync_api import sync_playwright

    # **Whether the emitted files are older than the sources is looked at first.** A stale
    # `dist` comes out with the same wording as a real gap — two copies of a check means only
    # one gets fixed, so the TS runner's is borrowed.
    sys.path.insert(0, str(ROOT / "borch-ts" / "test"))
    from run import (require_fresh_dist, require_fresh_golden,   # noqa: PLC0415
                     require_installed_matches_lock)
    # Ahead of `dist`, because a stale install builds a stale emit and then the emit is
    # blamed for what the dependency did.
    require_installed_matches_lock(ROOT)
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
            if js:
                # **The same channel one language over.** The page has already loaded
                # borch.ts and awaited `init()`, so a module imported here shares that
                # instance and the adapter that was reported above — importing a second
                # copy would build a second device and measure something else.
                probed = page.evaluate(
                    "async (src) => String(await (await import(src)).report())", js)
    finally:
        stop()
    return result, probed


# Adapter names that mean **the GPU was never reached.** SwiftShader is Chrome's
# software rasteriser and llvmpipe is Mesa's; both answer WebGPU calls correctly and
# neither proves a shader compiled anywhere real.
_SOFTWARE = ("swiftshader", "llvmpipe", "lavapipe", "software")


def _adapter_note(result):
    """What ran the shaders, **on the line that carries the score.**

    `runner.html` has read the adapter into `GOLDEN_RESULT.backend` all along and this
    file never printed it. So every binding run in this repository has reported a
    number with no word about what produced it, and today both sessions holding this
    codebase found out — from the *other* runner's output — that every browser golden
    they had run was on SwiftShader.

    **The comment twelve lines down already names this mistake.** The library's own
    name used to sit on the header rather than on the score line, and somebody read
    `agreeing 3255/3255` and reported the binding clear. The fix then was to move the
    word onto the line with the number. The adapter was in the identical position and
    was not moved, because that repair was made about `lib` rather than about *the
    line a person reads.*

    Printing it elsewhere is not enough either, which is the borch.ts runner's
    evidence: it prints the adapter at the top of every run, both sessions read that
    output twenty times over, and nobody saw it. Distance from the number is what
    makes a true line invisible.

    **This does not stop the run.** A green on SwiftShader still proves the values,
    and refusing to run at all on the only machine available would trade a real check
    for none. What it must not do is let the word *agreeing* stand unqualified.
    """
    backend = result.get("backend") or ""
    if not backend or "no browser GPU" in backend:
        return ""                                   # the core is numpy; no shaders ran
    low = backend.lower()
    if any(mark in low for mark in _SOFTWARE):
        return (f"  [{backend}]\n"
                "  **A software adapter.** The values are proved; that the same values "
                "come off a real\n  GPU is not — WGSL goes through a different compiler "
                "per vendor.")
    return f"  [{backend}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", required=True, choices=("borch", "borch_webgpu"),
                    help="the library to compare. **No default** — see the module "
                         "docstring: the old one measured the layer that is always "
                         "green, under a name that read as the binding")
    # **A window is the default now**, so this flag is kept only because fifteen
    # runners and a dozen documents spell it. See `launch._headed`.
    ap.add_argument("--headed", action="store_true",
                    help="opens a window — now the default, kept because it is written everywhere")
    # **The door needs an escape or it is not a door.** Making a window the default
    # without this left no way to say *measure on the CPU on purpose*, and the first
    # run of it stopped with `unrecognized arguments: --headless` — argparse refusing
    # the one flag the new failure message tells the reader to use.
    ap.add_argument("--headless", action="store_true",
                    help="no window, and therefore a software adapter: the values are proved "
                         "and the GPU path is not")
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
    ap.add_argument("--resnet", action="store_true",
                    help="trains the small ResNet of tests/resnet.py **in the browser** and "
                         "prints the same key/value lines the native run prints, so the two "
                         "can be compared line for line")
    ap.add_argument("--require-gpu", action="store_true",
                    help="stop with a non-zero exit if the adapter turns out to be a software "
                         "one. The default is to run and say so in a line — use this when the "
                         "point of the run is the GPU path rather than the values")
    ap.add_argument("--resnet-ts", action="store_true",
                    help="the same network written in TypeScript (borch-ts/test/resnet.ts), "
                         "trained through borch.ts's own API rather than through the Python "
                         "binding. Prints the same lines")
    args = ap.parse_args()
    if args.resnet and not args.probe:
        # **The native file, loaded by path.** `/work/tests` is not a package, so the same
        # `spec_from_file_location` the page uses for `golden.py` is used here. Copying the
        # network into a browser-side module instead would put the thing being compared in
        # two places, and the copy that is not run is the one that drifts.
        args.probe = (
            "import importlib, importlib.util\n"
            "_s = importlib.util.spec_from_file_location('bt_resnet', '/work/tests/resnet.py')\n"
            "_m = importlib.util.module_from_spec(_s); _s.loader.exec_module(_m)\n"
            f"L = importlib.import_module({args.lib!r})\n"
            "'\\n'.join(f'{k}\\t{v!r}' for k, v in _m.report(L).items())")
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

    # **A golden older than the case table compares yesterday's answers to today's
    # questions**, and every difference it prints reads as a defect in whatever was
    # last changed. That is not a hypothetical: a commit moved `deform_conv2d`'s
    # fixture off `numpy.random` onto a generator both languages can compute, and a
    # dump from before it turned nine cases red here. They were reported as a defect on
    # main, a second session spent time reasoning about the cause, and CI had been green
    # throughout — because CI dumps a fresh golden every run and so cannot have this.
    #
    # `golden.npz` is not committed, which is exactly why only local runs can go stale.
    # The sister rule is `require_fresh_dist` above; `test_site.py` borrows that one
    # rather than restating it, for the same reason this stops rather than warning.
    #
    # **The time is the whole test here, and that is a trade rather than an oversight.**
    # `require_fresh_golden` does better for `golden.json`: when the times disagree it
    # compares the name table and goes through if it is unchanged, because a comment
    # edit moves the time and a false alarm left standing teaches people to walk past
    # the real one. That escalation cannot be borrowed here. The two stamps the `npz`
    # carries are the case **names** (`__manifest__`) and the **shared** input arrays
    # (`__inputs__`), and the case that made this guard necessary moved neither: the
    # deform fixture is built inside the case from a seeded generator, so its bytes
    # changed while both stamps stayed identical. Escalating would make this guard
    # silent for exactly the failure it was written for.
    #
    # So the cost is real and it is the one that docstring warns about — editing only a
    # comment in `cases.py` stops this runner. The remedy it prints is correct anyway
    # and takes a minute, which is what makes the trade bearable; a fingerprint that
    # reached case-local fixtures would end it, and nothing computes one today.
    cases = ROOT / "tests" / "cases.py"
    if cases.exists() and cases.stat().st_mtime > GOLDEN.stat().st_mtime:
        print(f"the golden is older than the case table — {GOLDEN.name} was frozen before "
              f"{cases.relative_to(ROOT)} last changed.\n"
              "  first: uv run --with numpy --with torch --with torchvision --with scipy "
              "python tests/golden.py dump\n"
              "  (run it as it is and a case whose fixture moved comes back as a value "
              "mismatch, which is\n"
              "   **the same wording** as a real divergence, so the cause is invisible.)")
        return 1

    # Pyodide comes from local files. Fetched once if absent, compared by hash if present.
    vendor.ensure()

    result, probed = run(args.lib, args.headed, args.probe,
                         "/borch-ts/dist/test/resnet.js" if args.resnet_ts else None)
    if probed is not None:
        print("-- probe --")
        print(probed)
        print()
    # **A training run that only prints is a comment.** Both `--resnet` flags return the
    # same key/value lines the native run prints, and until this was here reading them was
    # left to whoever remembered to look. They are judged against the answers real torch
    # wrote down, because a page has no torch to ask.
    if (args.resnet or args.resnet_ts) and probed is not None:
        import importlib.util                                       # noqa: PLC0415

        spec = importlib.util.spec_from_file_location(
            "bt_resnet_cmp", ROOT / "tests" / "resnet.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        got = {k: float(v) for k, v in
               (line.split("\t") for line in probed.splitlines() if "\t" in line)}
        parted = mod.compare(got, ROOT)
        where = "borch.ts, through its own API" if args.resnet_ts else args.lib
        if parted:
            print(f"the training run on {where} parted from torch's:")
            for line in parted:
                print(f"  {line}")
            return 1
        print(f"the training run on {where} agrees with torch — {len(got)} values")
    total, bad = result["total"], result["bad"]
    if result.get("error"):
        print("the runner blew up:\n" + result["error"])
        return 1
    # **The library goes on the line that carries the number, not only on the header.**
    # It was on the header already, and that is not where a person looking for a score
    # reads. Someone hunting a binding defect ran this bare, saw `agreeing 3255/3255`,
    # and reported the binding clear — the word `borch` was three lines up and did its
    # job for nobody.
    lib = result.get("lib", args.lib)
    print(f"browser golden comparison ({lib}) — {total} cases")
    print(f"  {lib}: agreeing {total - len(bad)}/{total}{_adapter_note(result)}")
    # **A validation error does not arrive as an exception.** If even one happened, the green
    # above is that much less trustworthy — an invalid command buffer quietly does nothing while
    # the wall clock keeps running.
    if result.get("faults"):
        print(f"  {result['faults']} GPU validation errors")
    # **A run whose purpose is the GPU has to be able to say so.**
    #
    # `warn_if_software` narrows the claim in a printed line and lets the run pass, and
    # that default is right: a device does not change values, so the golden is real
    # evidence on a software adapter, and CI has no GPU and must stay green.
    #
    # But the same green is what somebody sets out to get when they are trying to prove
    # the GPU path, and then the warning is all that separates the two — a line that has
    # to be read. It very nearly was not: a session brought an RTX 4090 up, ran the whole
    # golden, got `3758/3758`, and was one unread line away from reporting *the golden
    # passes on a 4090*. The card had fallen off the PCIe bus mid-run (`rev ff`) and
    # everything ran on SwiftShader.
    #
    # So the intent goes on the command line instead of in the reader. `warn_if_software`'s
    # own docstring says the difference from `refuse_if_software` is deliberate and that
    # widening it is a person's decision — this is that decision, made per run rather than
    # once for everybody.
    if args.require_gpu and is_software(result.get("backend") or ""):
        print("\n**--require-gpu, and this was not a GPU.**\n"
              f"  The adapter is `{result.get('backend')}` — a CPU behind WebGPU's interface.\n"
              "  Every number above is real and none of it is evidence about a GPU.\n"
              "  On Linux the card can also be gone rather than blocked: `lspci` showing\n"
              "  `rev ff` on the slot means it left the bus and a reboot is the fix.")
        return 1
    if bad:
        print("\nwhere it diverged:")
        for why in bad:
            print(f"  ✗ {why}")
        # **A flat list says how many failures, not how many causes**, and those are
        # different numbers: 94 failures here were about four defects, one of which
        # accounted for 64. Reading that took a person opening `cases.py` and noticing
        # which helper the failing cases shared and which sibling had none.
        #
        # Prints nothing when no helper stands out. A ranking on a run whose failures
        # are unrelated is a plausible wrong lead, which costs more than no lead.
        for line in _grouped(bad):
            print(line)
    return 0 if result["ok"] else 1


def _grouped(bad):
    """The blame lines, or nothing at all if anything about building them fails.

    **Wrapped, because this is a convenience printed after the answer.** The failure
    list above is the result; a defect in the grouping must not take the exit code with
    it, and `cases.py` needs numpy and torchvision that a bare runner may not have.
    """
    try:
        import importlib.util                                    # noqa: PLC0415
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
        import blame                                             # noqa: PLC0415

        spec = importlib.util.spec_from_file_location(
            "bt_cases", ROOT / "tests" / "cases.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return blame.report(bad, mod.golden_cases(mod.golden_inputs()))
    except Exception as e:                                       # noqa: BLE001
        return ["", f"  (the failure grouping could not run: {type(e).__name__})"]


if __name__ == "__main__":
    sys.exit(main())
