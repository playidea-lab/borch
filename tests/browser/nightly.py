"""The forty-one browser checks nothing else runs, run once a night in a worktree.

    uv run --project /Users/changmin/git/borch python tests/browser/nightly.py

`gpu.yml` counts thirteen entry points that need a browser and runs none of them —
attaching a runner is a person's job, and until one is attached this is the person.
Run by hand after a gap they were two red files and one real defect in the core that
only Pyodide could see; run every night they are a log line.

**It works in its own worktree of `origin/main`**, never in the checkout somebody is
editing — `CLAUDE.md` rule 1, and this file is the reason the rule has a second half.
`bench` is left out on purpose: it is a measurement, not a check, and a wall clock at
04:30 measures the machine's other jobs.

Exit code is the number of entry points that failed. The log names each.
"""

import datetime
import os
import pathlib
import shutil
import subprocess
import sys

# **The checkout is wherever this file lives**, not a path written down — the second
# machine (an RTX 5080 on Ubuntu, `/home/pi/borch-nv`) has it somewhere else, and one
# number per adapter is the whole point of a second machine. Override with
# `BORCH_NIGHTLY_REPO` when the file is run from a copy.
REPO = pathlib.Path(os.environ.get("BORCH_NIGHTLY_REPO")
                    or pathlib.Path(__file__).resolve().parents[2])
WORKTREE = REPO.parent / "borch-nightly"
# macOS keeps logs where Console.app looks; Linux where the XDG state dir says.
LOGS = (pathlib.Path.home() / "Library" / "Logs" / "borch-nightly"
        if sys.platform == "darwin"
        else pathlib.Path(os.environ.get("XDG_STATE_HOME")
                          or pathlib.Path.home() / ".local" / "state") / "borch-nightly")

# The twelve, as `gpu.yml` lists them minus `bench`. Each is (label, argv).
CHECKS = [
    ("vendor",     ["uv", "run", "--project", str(REPO), "python", "tests/browser/vendor.py", "check"]),
    ("golden:core",    ["uv", "run", "--project", str(REPO), "--with", "playwright", "python", "tests/browser/run.py", "--lib", "borch"]),
    ("golden:binding", ["uv", "run", "--project", str(REPO), "--with", "playwright", "python", "tests/browser/run.py", "--lib", "borch_webgpu"]),
    ("golden:ts",  ["uv", "run", "--project", str(REPO), "--with", "playwright", "python", "borch-ts/test/run.py"]),
    ("parity",     ["uv", "run", "--project", str(REPO), "--with", "playwright", "python", "borch-ts/test/parity.py"]),
    ("data",       ["uv", "run", "--project", str(REPO), "--with", "playwright", "python", "borch-ts/test/data.py"]),
    ("device",     ["uv", "run", "--project", str(REPO), "--with", "playwright", "python", "borch-ts/test/device.py"]),
    ("serialize",  ["uv", "run", "--project", str(REPO), "--with", "playwright", "python", "borch-ts/test/serialize.py"]),
    ("onnx",       ["uv", "run", "--project", str(REPO), "--with", "playwright", "python", "borch-ts/test/onnx.py"]),
    ("onnx:binding", ["uv", "run", "--project", str(REPO), "--with", "playwright", "python", "tests/browser/onnx_binding.py"]),
    # The CPU device against the WebGPU device — two hub checkpoints, logits compared.
    ("cpu",        ["uv", "run", "--project", str(REPO), "--with", "playwright", "python", "borch-ts/test/cpu.py"]),
    # `borch_cpu` on the wheel, no device — and against the GPU where there is one.
    ("cpu:py",     ["uv", "run", "--project", str(REPO), "--with", "playwright", "python", "tests/browser/cpu_py.py", "--build"]),
    # The workbench with WebGPU's service disabled — the `borch_cpu` door, end to end.
    ("marimo:cpu", ["uv", "run", "--project", str(REPO), "--with", "playwright", "python", "tests/browser/marimo_probe.py", "--no-webgpu"]),
    ("coi:site",   ["uv", "run", "--project", str(REPO), "--with", "playwright", "python", "tests/browser/coi_sweep.py"]),
    ("refine:py",  ["uv", "run", "--project", str(REPO), "--with", "playwright", "python", "tests/browser/refine_py.py", "--build", "--folds=3"]),
    # The wheel alone, in a worker — JupyterLite's shape. Builds the wheel first.
    # No `--headed` here: every probe opens a window by default (launch.py `_headed`), and
    # headless is asked for with `--headless` or BORCH_HEADLESS. Five probes had kept the
    # old rule — a window only on `--headed` — and came off SwiftShader in the first nightly
    # that ran them (2026-09-06); they follow the launcher now.
    ("wheel",      ["uv", "run", "--project", str(REPO), "--with", "playwright", "python", "tests/browser/wheel_probe.py", "--build"]),
    # The published package from a CDN, in a file opened from disk — the entry point a
    # reader with no bundler takes. An unreachable CDN is not a failure; the probe says so.
    ("cdn",        ["uv", "run", "--project", str(REPO), "--with", "playwright", "python", "tests/browser/cdn_probe.py"]),
    # What the deployed page costs in bytes, and how much of it comes before the verdict.
    ("weight",     ["uv", "run", "--project", str(REPO), "--with", "playwright", "python", "tests/browser/weight_probe.py"]),
    # The notebook page — JupyterLite built here, the cell pressed, the learned line read.
    ("lab",        ["uv", "run", "--project", str(REPO), "--with", "playwright", "python", "tests/browser/lab_probe.py", "--build"]),
    # The workbench page — marimo built here, run pressed, the four sections read.
    ("marimo",     ["uv", "run", "--project", str(REPO), "--with", "playwright", "python", "tests/browser/marimo_probe.py", "--build"]),
    # 5,000 images through the tab: decode, backbone, cache, head, neighbours. Files made once under /tmp.
    ("folder",     ["uv", "run", "--project", str(REPO), "--with", "playwright", "python", "tests/browser/folder5k.py", "--make=5000"]),
    # The wheel loads a catalogue model in Python (`torch.hub.load`) and runs it at 224 px.
    ("hub",        ["uv", "run", "--project", str(REPO), "--with", "playwright", "python", "tests/browser/hub_py.py", "--build"]),
    # The offline bundle: built from the site's build, then pressed with every outside request refused.
    ("bundle",     ["uv", "run", "--project", str(REPO), "--with", "playwright", "python", "tests/browser/marimo_probe.py", "--bundle"]),
    # torch's training runs — a head, a CNN, a U-Net — step by step on the wheel: the loop, not the op.
    ("trajectory", ["uv", "run", "--project", str(REPO), "--with", "playwright", "python", "tests/browser/trajectory_py.py", "--build"]),
    # The compiler on this GPU: a step captured and replayed bit for bit (U-Net, GPT with
    # AdamW and StepLR under check=True), and the fused replay within a rounding.
    ("capture:py", ["uv", "run", "--project", str(REPO), "--with", "playwright", "python", "tests/browser/capture_py.py", "--build"]),
    ("fuse:py",    ["uv", "run", "--project", str(REPO), "--with", "playwright", "python", "tests/browser/fuse_py.py", "--build"]),
    # The aliasing patterns a golden of single ops misses — several views of one base
    # summing gradients back, and an optimizer step through the compiler with check=True.
    ("aliasing:py", ["uv", "run", "--project", str(REPO), "--with", "playwright", "python", "tests/browser/aliasing_probe.py", "--build"]),
    # One model per op class through torch.compiled(check=True) under SGD-with-momentum and
    # Adam — the op surface the ~4 hand-built models did not run under the compiler.
    ("compiled:py", ["uv", "run", "--project", str(REPO), "--with", "playwright", "python", "tests/browser/compiled_probe.py", "--build"]),
    # The pool filled with a loud sentinel, then a backward that scatters into a full buffer
    # — its gradient must still match numpy, so nothing reads a byte it did not write.
    ("churn:py",   ["uv", "run", "--project", str(REPO), "--with", "playwright", "python", "tests/browser/churn_probe.py", "--build"]),
    # A non-leaf saved for backward, then changed in place, must raise (as torch does) rather
    # than compute a silently wrong gradient — and an op that saved nothing must still allow it.
    ("vcount:py",  ["uv", "run", "--project", str(REPO), "--with", "playwright", "python", "tests/browser/vcount_probe.py", "--build"]),
    # A BatchNorm CNN and a scalar parameter trained under torch.compiled, then read after
    # dispose — the capture must not retire a kept buffer (the workbench's scratch/segment).
    ("compiled-train:py", ["uv", "run", "--project", str(REPO), "--with", "playwright", "python", "tests/browser/compiled_train_probe.py", "--build"]),
    # The step's clock, three models through the compiler, one timed step each — the
    # numbers land in the log as `@<model>_fused_ms=` lines, a curve when they are kept.
    ("step:unet",  ["uv", "run", "--project", str(REPO), "--with", "playwright", "python", "tests/browser/profile_py.py", "--build", "--model=unet", "--steps=1", "--compiled=fused"]),
    ("step:gpt",   ["uv", "run", "--project", str(REPO), "--with", "playwright", "python", "tests/browser/profile_py.py", "--build", "--model=gpt", "--steps=1", "--compiled=fused"]),
    ("step:vit",   ["uv", "run", "--project", str(REPO), "--with", "playwright", "python", "tests/browser/profile_py.py", "--build", "--model=vit", "--batch=8", "--steps=1", "--compiled=fused"]),
    ("example",    ["uv", "run", "--project", str(REPO), "--with", "playwright", "python", "borch-ts/test/readme.py"]),
    # `--py`: every Python twin on the lesson pages is pressed too (Pyodide from vendor/).
    ("lessons",    ["uv", "run", "--project", str(REPO), "--with", "playwright", "python", "borch-ts/test/lessons.py", "--py"]),
    ("scope",      ["uv", "run", "--project", str(REPO), "--with", "playwright", "python", "tests/browser/scope_escape.py"]),
    ("cost",       ["uv", "run", "--project", str(REPO), "--with", "playwright", "python", "borch-ts/test/cost.py"]),
    ("first-run",  ["uv", "run", "--project", str(REPO), "--with", "playwright", "python", "tests/browser/first_run.py"]),
    # The same clock on the deployed site, so the transfer is inside it — the visitor's number.
    ("first-run:deployed", ["uv", "run", "--project", str(REPO), "--with", "playwright", "python", "tests/browser/first_run.py",
                            "--url=https://playidea-lab.github.io/borch/site/index.html"]),
    # The person's clock, not the GPU's: Python ready, the click, the first loss line.
    ("learner:site", ["uv", "run", "--project", str(REPO), "--with", "playwright", "python", "tests/browser/learner_path.py",
                      "--url=https://playidea-lab.github.io/borch/site/index.html"]),
]


def sh(argv, cwd, log):
    # **The display has to be on.** The job fires at 04:30 and every probe opens a
    # window on a real adapter; with the display asleep, GPU work on this Mac stalls in
    # one-second quanta and a headed run that takes two minutes awake never finishes
    # (measured 2026-09-05: a run stopped dead at 19:12, the minute the display turned
    # off). `caffeinate -u` wakes the display and keeps it so for the command's life.
    if sys.platform == "darwin" and shutil.which("caffeinate"):
        argv = ["caffeinate", "-u", "-d", "-i", *argv]
    log.write(f"\n$ {' '.join(argv)}\n")
    log.flush()
    # **Unbuffered, because stdout is a file here.** Python block-buffers a file, so a
    # probe's lines land only when it exits — and a probe that never exits leaves nothing.
    # The sweep that held the 2026-09-07 run for two hours read as silent the whole time;
    # it had printed, into a buffer. Every child now flushes as it prints.
    done = subprocess.run(argv, cwd=cwd, stdout=log, stderr=subprocess.STDOUT,
                          env={**os.environ, "PYTHONUNBUFFERED": "1",
                               "PATH": "/opt/homebrew/bin:"
                               + str(pathlib.Path.home() / ".local/bin") + ":"
                               + os.environ.get("PATH", "")})
    return done.returncode


def prepare(log):
    """A worktree at `origin/main`, fresh each night — created once, moved after."""
    if sh(["git", "fetch", "origin"], REPO, log):
        return False
    if not WORKTREE.exists():
        if sh(["git", "worktree", "add", "--detach", str(WORKTREE), "origin/main"], REPO, log):
            return False
    else:
        if sh(["git", "checkout", "--detach", "origin/main"], WORKTREE, log):
            return False
    # **Its own `node_modules`, from the lock.** This used to be a symlink to the main
    # tree's, and the main tree's was whatever somebody had last installed there — three
    # nights of `golden:*` red (2026-09-04 → 06) because esbuild, bimm-ts and borch-hub had
    # been added to the lock and never installed at the link's target. `npm ci` is thirty
    # seconds and reads the lock the worktree was checked out with.
    link = WORKTREE / "node_modules"
    if link.is_symlink():
        link.unlink()
    if sh(["npm", "ci", "--ignore-scripts"], WORKTREE, log):
        return False
    if sh(["npx", "tsc", "-p", "borch-ts/tsconfig.json"], WORKTREE, log):
        return False
    # The golden is not committed; every runner reads it, so it is frozen here first.
    # **torch is named here, not assumed.** The project's own environment has numpy and
    # nothing else; the first run on the 5080 stopped at `import torch` because the
    # laptop's environment happened to carry it. This is the workflow's spelling.
    return sh(["uv", "run", "--project", str(REPO), "--with", "torch", "--with", "torchvision",
               "--with", "scipy", "python", "-W", "ignore",
               "tests/golden.py", "dump"], WORKTREE, log) == 0


def main():
    if "--list" in sys.argv:
        print(f"repo      {REPO}\nworktree  {WORKTREE}\nlogs      {LOGS}\nchecks    {len(CHECKS)}")
        for label, argv in CHECKS:
            print(f"  {label:16s} {' '.join(argv[-3:])}")
        return 0
    LOGS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M")
    path = LOGS / f"{stamp}.log"
    with path.open("w", encoding="utf-8") as log:
        log.write(f"borch nightly — {stamp}\n")
        if not prepare(log):
            log.write("\n** could not prepare the worktree — nothing was checked **\n")
            return 99
        failed = []
        for label, argv in CHECKS:
            code = sh(argv, WORKTREE, log)
            log.write(f"\n== {label}: {'ok' if code == 0 else f'FAILED ({code})'}\n")
            if code:
                failed.append(label)
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=WORKTREE,
                              capture_output=True, text=True).stdout.strip()
        log.write(f"\n{len(CHECKS) - len(failed)} of {len(CHECKS)} passed at {head}"
                  + (f" — failed: {', '.join(failed)}" if failed else "") + "\n")
    latest = LOGS / "latest.log"
    if latest.exists() or latest.is_symlink():
        latest.unlink()
    latest.symlink_to(path)
    if failed:
        notify(f"borch nightly: {len(failed)} failed", ", ".join(failed))
    return len(failed)


def notify(title, body):
    """A desktop notification where the desktop has one; the log is the record either way."""
    if sys.platform == "darwin":
        argv = ["osascript", "-e", f'display notification "{body}" with title "{title}"']
    elif shutil.which("notify-send"):
        argv = ["notify-send", title, body]
    else:
        return
    subprocess.run(argv, check=False)


if __name__ == "__main__":
    sys.exit(main())
