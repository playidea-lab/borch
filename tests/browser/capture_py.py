"""A captured training step, replayed — **the same numbers, without Python.**

    uv run --with playwright python tests/browser/capture_py.py [--build] [--batch=16] [--size=96] [--headless]

Twelve steps of the U-Net of tests/seg_eval.py on synthetic data, eagerly; then the same
twelve from the same weights with the first step recorded under `torch.capture()` and the
other eleven replayed, the batch copied into the captured inputs. Judged: every loss the
same, every learned parameter and running statistic the same (the one thing that differs
is BatchNorm's `num_batches_tracked`, a CPU counter the replay does not run), no fault.
Then `torch.compiled(step)` over a sequence whose batches are sixteen and, every
seventh, ten: two recordings, and the eager run of the same sequence bit for bit.
Then a small transformer under AdamW with a StepLR schedule — batched attention,
one-kernel softmax, the weight decay as recorded copies, the learning rate moving between
steps through the optimizer's device scalars — through `torch.compiled` plain (bit for
bit against eager) and fused (within 1e-5 on the loss, fewer dispatches).
Reported: the wall-clock step both ways — on the M4 Max 17.3 eager, 14.7 replayed.
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
    if refuse_if_screen_off("the captured step"):
        return 1
    port, shutdown = serve(ROOT)
    url = f"http://127.0.0.1:{port}/tests/browser/capture_py.html?wheel=/{wheel}&batch={batch}&size={size}"
    profile = tempfile.mkdtemp(prefix="borch-capture-")
    channel = os.environ.get("BORCH_CHROME_CHANNEL") or None
    try:
        with sync_playwright() as pw:
            context = pw.chromium.launch_persistent_context(profile, headless=not headed, channel=channel, args=list(FLAGS), timeout=60_000)
            try:
                page = context.new_page()
                # **A page script that does not parse is a ten-minute silence.** The Python
                # of the probe sits inside a JS template literal, and one backtick in a
                # Python comment ended the literal — the worker was never made, nothing
                # was posted, and the runner waited its whole timeout twice (2026-09-20).
                # An uncaught page error now ends the wait with the error named.
                page.on("pageerror", lambda e: page.evaluate(
                    "(m) => { window.__wheel = window.__wheel || { text: 'page error: ' + m, error: m }; }", str(e)))
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
    d_loss = re.search(r"\|Δloss\| ([0-9.e+-]+)", done)
    d_w = re.search(r"\|Δparam\| ([0-9.e+-]+)", done)
    comp = re.search(r"compiled two shapes: recordings (\d+) max \|Δloss\| ([0-9.e+-]+) max \|Δparam\| ([0-9.e+-]+)", done)
    ok = "faults 0" in done and bool(d_loss and d_w and comp) and float(d_loss.group(1)) == 0.0 and float(d_w.group(1)) == 0.0
    ok = ok and int(comp.group(1)) == 2 and float(comp.group(2)) == 0.0 and float(comp.group(3)) == 0.0
    # The small transformer under AdamW: the plain recording bit for bit (its weight
    # decay is recorded copies), the fused one within 1e-5 on the loss; the fused
    # parameters are judged loosely — Adam magnifies a rounding on a bias near zero.
    gpt = re.search(r"gpt compiled plain rel\|Δloss\| ([0-9.e+-]+) rel\|Δparam\| ([0-9.e+-]+) (\d+) dispatches, fused rel\|Δloss\| ([0-9.e+-]+) rel\|Δparam\| ([0-9.e+-]+) (\d+) dispatches", done)
    ok = ok and bool(gpt) and float(gpt.group(1)) == 0.0 and float(gpt.group(2)) == 0.0
    ok = ok and float(gpt.group(4)) <= 1e-5 and float(gpt.group(5)) <= 1e-2 and int(gpt.group(6)) < int(gpt.group(3))
    # `check=True` ran on both recordings and raised on neither.
    ok = ok and bool(re.search(r"check=True passed on \d+ live-ins", done))
    # Every dispatch of both recordings says what it reads and writes — by recipe or by
    # its WGSL — and none is guessed (docs/COMPILER.md Step 0's gate).
    guessed = re.findall(r"guessed (\d+)", done)
    ok = ok and len(guessed) == 2 and all(int(g) == 0 for g in guessed)
    # The planner laid the intermediates into arenas smaller than the buffers they replace,
    # and what the device holds after the plan is less than before (docs/COMPILER.md Step 1);
    # the bit-for-bit gates above ran on the planned recordings.
    plans = re.findall(r"moved (\d+) released \d+ ([0-9.]+)MB→([0-9.]+)MB", done)
    ok = ok and len(plans) == 2 and all(int(m) > 0 and float(a) < float(b) for m, b, a in plans)
    held = re.search(r"held ([0-9.]+)MB→([0-9.]+)MB", done)
    ok = ok and bool(held) and float(held.group(2)) < float(held.group(1))
    # `torch.compiled(model)` — the inference form: the folded, recorded forward gives the
    # plain eval forward to a rounding, replays identically, in fewer dispatches.
    inf = re.search(r"inference compiled\(model\) rel ([0-9.e+-]+) replay-same ([0-9.e+-]+) dispatches (\d+) vs eager (\d+)", done)
    ok = ok and bool(inf) and float(inf.group(1)) <= 1e-5 and float(inf.group(2)) == 0.0 and int(inf.group(3)) < int(inf.group(4))
    print("**the replayed step is the eager step, bit for bit**" if ok else "**it is not** — see above")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
