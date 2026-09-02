"""The twelve browser checks nothing else runs, run once a night in a worktree.

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
import subprocess
import sys

REPO = pathlib.Path("/Users/changmin/git/borch")
WORKTREE = REPO.parent / "borch-nightly"
LOGS = pathlib.Path.home() / "Library" / "Logs" / "borch-nightly"

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
    ("example",    ["uv", "run", "--project", str(REPO), "--with", "playwright", "python", "borch-ts/test/readme.py"]),
    ("lessons",    ["uv", "run", "--project", str(REPO), "--with", "playwright", "python", "borch-ts/test/lessons.py"]),
    ("scope",      ["uv", "run", "--project", str(REPO), "--with", "playwright", "python", "tests/browser/scope_escape.py"]),
    ("cost",       ["uv", "run", "--project", str(REPO), "--with", "playwright", "python", "borch-ts/test/cost.py"]),
]


def sh(argv, cwd, log):
    log.write(f"\n$ {' '.join(argv)}\n")
    log.flush()
    done = subprocess.run(argv, cwd=cwd, stdout=log, stderr=subprocess.STDOUT,
                          env={**os.environ, "PATH": "/opt/homebrew/bin:"
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
    link = WORKTREE / "node_modules"
    if not link.exists():
        link.symlink_to(REPO / "node_modules")
    if sh(["npx", "tsc", "-p", "borch-ts/tsconfig.json"], WORKTREE, log):
        return False
    # The golden is not committed; every runner reads it, so it is frozen here first.
    return sh(["uv", "run", "--project", str(REPO), "python", "-W", "ignore",
               "tests/golden.py", "dump"], WORKTREE, log) == 0


def main():
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
        subprocess.run(["osascript", "-e",
                        f'display notification "{", ".join(failed)}" '
                        f'with title "borch nightly: {len(failed)} failed"'])
    return len(failed)


if __name__ == "__main__":
    sys.exit(main())
