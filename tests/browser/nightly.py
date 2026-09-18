"""The forty-two browser checks nothing else runs, run once a night in a worktree.

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
    ("lora",       ["uv", "run", "--project", str(REPO), "--with", "playwright", "python", "borch-ts/test/lora.py"]),
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
    # The reasons in torch_gap.py that claim something about the browser, re-measured.
    ("claims",     ["uv", "run", "--project", str(REPO), "--with", "playwright", "python", "tests/browser/platform_claims.py"]),
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
    # docs/SCALE.md Step 7: the Python peft mirror. `apply_lora` walks a Python-composed model
    # and swaps each matched Linear/Conv2d for its LoRA wrapper (base bridged from the leaf's
    # borch.ts layer) — forward unchanged, params reduced to adapters, indexed leaves refused.
    ("peft:py", ["uv", "run", "--project", str(REPO), "--with", "playwright", "python", "tests/browser/peft_py.py", "--build"]),
    # docs/SCALE.md Step 7: the Python streaming bridge. torch.streaming reaches borch.ts's
    # scale primitives from Python — freeze a stack, apply_lora, offload and stream-train one
    # step, adapters and a head trained through the loss get gradients, a streamed forward runs.
    ("streaming:py", ["uv", "run", "--project", str(REPO), "--with", "playwright", "python", "tests/browser/streaming_py.py", "--build"]),
    # docs/SCALE.md Step 7 / the workbench: torch.workbench.setup(..., finetune=True).fit() adapts
    # a backbone with LoRA (a folder of colour classes), scores it, and exports a model — the
    # fine-tune path the workbench page says tissue needs. Network (hub fetches a ViT-Tiny) + GPU.
    ("workbench-lora:py", ["uv", "run", "--project", str(REPO), "--with", "playwright", "python", "tests/browser/workbench_lora_py.py", "--build"]),
    # docs/SCALE.md conv perf: the conv input-gradient (dX) on subgroup matrices vs the direct
    # reference — bit-identical, on shapes the golden is too small to reach. Skips where there
    # are no subgroup matrices.
    ("conv-dx", ["uv", "run", "--project", str(REPO), "--with", "playwright", "python", "tests/browser/conv_dx_probe.py"]),
    # The buffer pool's invariants watched through a real training run — no kept or
    # capture-owned buffer pooled, none pooled twice, each in its size's bucket.
    ("invariants:py", ["uv", "run", "--project", str(REPO), "--with", "playwright", "python", "tests/browser/invariants_probe.py", "--build"]),
    # ONNX export refuses an op it cannot trace (layer_norm/softmax/attention) rather than
    # freezing it into the file as a constant — the silent-drop a non-CNN model used to hit.
    ("onnx-trap:py", ["uv", "run", "--project", str(REPO), "--with", "playwright", "python", "tests/browser/onnx_trap_probe.py", "--build"]),
    # The step's clock, three models through the compiler, one timed step each — the
    # numbers land in the log as `@<model>_fused_ms=` lines, a curve when they are kept.
    ("step:unet",  ["uv", "run", "--project", str(REPO), "--with", "playwright", "python", "tests/browser/profile_py.py", "--build", "--model=unet", "--steps=1", "--compiled=fused"]),
    ("step:gpt",   ["uv", "run", "--project", str(REPO), "--with", "playwright", "python", "tests/browser/profile_py.py", "--build", "--model=gpt", "--steps=1", "--compiled=fused"]),
    ("step:vit",   ["uv", "run", "--project", str(REPO), "--with", "playwright", "python", "tests/browser/profile_py.py", "--build", "--model=vit", "--batch=8", "--steps=1", "--compiled=fused"]),
    ("example",    ["uv", "run", "--project", str(REPO), "--with", "playwright", "python", "borch-ts/test/readme.py"]),
    # `--py`: every Python twin on the lesson pages is pressed too (Pyodide from vendor/).
    ("lessons",    ["uv", "run", "--project", str(REPO), "--with", "playwright", "python", "borch-ts/test/lessons.py", "--py"]),
    # **The row `lessons` above cannot cover.** `site/embed/` is left out of every
    # full-page guard on purpose — it is chrome-less, so `nav`, `share-metadata` and
    # `sidebar` would fire on it wrongly — and an exclusion is a decision to not look
    # there, not a decision that there is nothing to see. Until this row existed the
    # embed was watched by the coi header sweep and nothing else, while sitting on the
    # four most breakable joints in the site: a foreign origin, no cross-origin
    # isolation, the Python twin stripped before mount, and a postMessage resize.
    ("embed",      ["uv", "run", "--project", str(REPO), "--with", "playwright", "python", "tests/browser/embed_probe.py"]),
    ("scope",      ["uv", "run", "--project", str(REPO), "--with", "playwright", "python", "tests/browser/scope_escape.py"]),
    # The backward of expand, repeat and flip: the folded kernel against the walking one,
    # which is the only thing that says the fold is an optimisation and not a new answer.
    ("fold",       ["uv", "run", "--project", str(REPO), "--with", "playwright", "python", "tests/browser/fold_probe.py"]),
    # The floor `borch_webgpu` stands on: whether Pyodide can call borch.ts synchronously.
    # Nothing else here fails if JSPI goes away — every binding row just gets slower or
    # hangs, and this is the row that would say why.
    ("sync",       ["uv", "run", "--project", str(REPO), "--with", "playwright", "python", "tests/browser/sync_probe.py"]),
    ("cost",       ["uv", "run", "--project", str(REPO), "--with", "playwright", "python", "borch-ts/test/cost.py"]),
    # The binding's own cost, the path a user walks — one more leaky place than the TS side
    # (a Python object can hold a JS handle). Reached through `run.py --cost`; running it by
    # hand on 2026-09-11 is what found the empty_cache device fault.
    ("cost:binding", ["uv", "run", "--project", str(REPO), "--with", "playwright", "python", "tests/browser/run.py", "--lib", "borch_webgpu", "--cost"]),
    ("first-run",  ["uv", "run", "--project", str(REPO), "--with", "playwright", "python", "tests/browser/first_run.py"]),
    # The same clock on the deployed site, so the transfer is inside it — the visitor's number.
    ("first-run:deployed", ["uv", "run", "--project", str(REPO), "--with", "playwright", "python", "tests/browser/first_run.py",
                            "--url=https://playidea-lab.github.io/borch/site/index.html"]),
    # The person's clock, not the GPU's: Python ready, the click, the first loss line.
    ("learner:site", ["uv", "run", "--project", str(REPO), "--with", "playwright", "python", "tests/browser/learner_path.py",
                      "--url=https://playidea-lab.github.io/borch/site/index.html"]),
    # docs/SCALE.md Step 0: how much GPU memory this tab holds (allocated until a marker
    # stops coming back), the limit tiers, shader-f16, and what two hub models cost the
    # host and the GPU. A measurement; it fails only when nothing could be measured.
    ("ceiling",    ["uv", "run", "--project", str(REPO), "--with", "playwright", "python", "tests/browser/ceiling.py"]),
    # docs/SCALE.md Step 6: gradient checkpointing. The recomputed backward against the
    # fully-taped one — the gradients must agree within the golden tolerance and the buffers
    # held after the forward must fall. Adapter-independent (values + counts), so it runs in
    # CI on SwiftShader too.
    ("checkpoint",  ["uv", "run", "--project", str(REPO), "--with", "playwright", "python", "tests/browser/checkpoint_probe.py"]),
    # docs/SCALE.md Step 3: the frozen-weight window primitive — two arrays into two offset
    # slices of one STORAGE buffer, filled through staging + copyRange, read back at each
    # offset. Adapter-independent (copyRange, slice bindings are core), so it runs in CI too.
    ("window",      ["uv", "run", "--project", str(REPO), "--with", "playwright", "python", "tests/browser/window_probe.py"]),
    # docs/SCALE.md Step 3 ④b: a real bimm ResNet-18's layer1 streamed block-by-block through
    # a small window — the resident layer output must come back bit-identical. Needs bimm-ts
    # from esm.sh, so it runs where the CDN is reachable (not offline CI).
    ("stream-model", ["uv", "run", "--project", str(REPO), "--with", "playwright", "python", "tests/browser/stream_model_probe.py"]),
    # docs/SCALE.md Step 7: a LoRA adapter trained one step on a streamed frozen backbone —
    # the adapter gradients must equal the fully-resident run's, with the frozen weights bounded
    # by the window. Imports only borch-ts (no CDN), so adapter-independent and CI-runnable.
    ("stream-train", ["uv", "run", "--project", str(REPO), "--with", "playwright", "python", "tests/browser/stream_train_probe.py"]),
    # docs/SCALE.md Step 7's gate: fine-tune a >=300 MB backbone (ViT-Base, 346 MB) with LoRA,
    # the frozen blocks offloaded and streamed through a bounded window — no worse than a
    # frozen-head baseline, the backbone never fully resident. Needs the network (hub + esm.sh)
    # and a real GPU, so it lives in the nightly rather than offline CI.
    ("finetune", ["uv", "run", "--project", str(REPO), "--with", "playwright", "python", "tests/browser/finetune.py"]),
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


# **The list that runs is the list this file had when it started, and it starts before the
# worktree is updated.** `CHECKS` is read at import; `prepare()` then moves the worktree to
# `origin/main`, rewriting this very file underneath a process that has already read it. So
# every night ran the previous day's rows against the current day's code, and nobody saw it
# because the rows that were missing had no line to be missing from. Measured 2026-09-11:
# the run said "36 of 38" while the file in the worktree it had just checked out held
# forty-two, and `cdn`, `weight` and `claims` — merged after the previous 04:30 — went
# unrun for a day each.
REEXEC = "BORCH_NIGHTLY_ON_CURRENT"


def _become_current():
    """Bring the worktree to `origin/main`, then start again from the file that lands there.

    Once only, guarded by the environment: the second process finds the variable set and
    goes on to the run. `prepare()` still does its own fetch and checkout, which is then a
    few seconds against a worktree already there — cheap, and it keeps the log's account of
    the night complete rather than splitting it across a process that exited.
    """
    here = pathlib.Path(__file__).resolve()
    # **Only the copy the checkout rewrites has anything to reload.** Run from somebody's own
    # checkout — which the launcher never does — this file is not the one that just moved, so
    # starting again would read the same rows twice and say it had refreshed them.
    if os.environ.get(REEXEC) or not here.is_relative_to(WORKTREE):
        return
    for argv in (["git", "fetch", "origin"], ):
        subprocess.run(argv, cwd=REPO, check=False)
    if WORKTREE.exists():
        subprocess.run(["git", "checkout", "--detach", "origin/main"], cwd=WORKTREE, check=False)
    else:
        subprocess.run(["git", "worktree", "add", "--detach", str(WORKTREE), "origin/main"],
                       cwd=REPO, check=False)
    os.environ[REEXEC] = "1"
    print(f"starting again from {here} — the rows that run are the rows on origin/main", flush=True)
    os.execv(sys.executable, [sys.executable, str(here), *sys.argv[1:]])


def browsers_ready(log):
    """Put the browser the *resolved* playwright wants on disk, and say which one. True if it
    is there.

    **2026-09-16 came back `1 of 48`.** Forty-seven rows died on one line —
    `BrowserType.launch: Executable doesn't exist at .../chromium-1243/` — and nothing in
    the repository had changed: `--with playwright` is unpinned, it resolved to a newer
    playwright than the night before, and that one asks for a chromium build the cache did
    not have. Every browser row failed for a reason that was not about borch, and the log
    said forty-seven different things about borch.

    **A pin was the obvious fix and is the wrong one here.** `--with playwright` appears in
    forty-seven rows above; pinning means forty-seven edits, and the next person who wants a
    newer playwright has to change all of them. Miss one and nothing turns red — the night
    quietly keeps running an old playwright, which is the failure this file has been bitten
    by in other shapes: an instrument that is silent about what it did not do.

    So the browser is fetched to match whatever resolved, and **the version is written into
    the log either way**. Drift stops breaking the night, and it stops being invisible: the
    line moves, and a person reading the log can see when it did.
    """
    ver = subprocess.run(["uv", "run", "--project", str(REPO), "--with", "playwright",
                          "playwright", "--version"], capture_output=True, text=True,
                         cwd=WORKTREE).stdout.strip()
    log.write(f"\n$ playwright --version\n{ver or '(could not be asked)'}\n")
    code = sh(["uv", "run", "--project", str(REPO), "--with", "playwright",
               "playwright", "install", "chromium"], WORKTREE, log)
    log.write(f"\n== browsers: {'ok' if code == 0 else f'FAILED ({code})'}\n")
    return code == 0


def main():
    if "--list" in sys.argv:
        print(f"repo      {REPO}\nworktree  {WORKTREE}\nlogs      {LOGS}\nchecks    {len(CHECKS)}")
        for label, argv in CHECKS:
            print(f"  {label:16s} {' '.join(argv[-3:])}")
        return 0
    _become_current()
    LOGS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M")
    path = LOGS / f"{stamp}.log"
    with path.open("w", encoding="utf-8") as log:
        log.write(f"borch nightly — {stamp}\n")
        if not prepare(log):
            log.write("\n** could not prepare the worktree — nothing was checked **\n")
            return 99
        # Before the rows, not instead of them: if the browser could not be fetched the
        # night still runs, because a run that stops here would report nothing at all about
        # the checks that need no browser — and the failure list is the honest damage.
        browsers = browsers_ready(log)
        if not browsers:
            log.write("\n** the browser could not be installed — every row that needs one is\n"
                      "   expected to fail below, and those failures are not about borch **\n")
        failed = []
        for label, argv in CHECKS:
            code = sh(argv, WORKTREE, log)
            log.write(f"\n== {label}: {'ok' if code == 0 else f'FAILED ({code})'}\n")
            if code:
                failed.append(label)
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=WORKTREE,
                              capture_output=True, text=True).stdout.strip()
        log.write(f"\n{len(CHECKS) - len(failed)} of {len(CHECKS)} passed at {head}"
                  + (f" — failed: {', '.join(failed)}" if failed else "")
                  + ("" if browsers else " — and the browser was never installed, so read"
                                         " the failures as that, not as forty-seven defects")
                  + "\n")
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
