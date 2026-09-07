"""The oracle paper's data-efficient refinement, trained in a Pyodide worker through `borch_webgpu`.
    uv run --with playwright python tests/browser/refine_py.py [--headed] [--build] [--wheel=dist/pyborch-X.whl]
                                                              [--data=/path/under/repo.npz] [--folds=N] [--k=5]

`refine_py.html` trains the paper's 34K-parameter residual MLP (42→128→128→64→42, LayerNorm,
ReLU, Dropout 0.1, zero-initialised output; L1; AdamW 1e-3 / wd 1e-4; cosine over 10 epochs;
batch 256; clipping at 1.0) on the k nearest training sequences of a leave-one-sequence-out
split and reports PA-MPJPE — baseline, k-nearest, oracle — per held-out sequence.

Without `--data` the page fabricates a set of the same shape (24 sequences, low-rank
per-sequence error). That proves the wiring — every op the recipe needs exists and trains —
and nothing about the paper. With the paper's cached predictions as an .npz (`pred`, `gt`,
`seq`) the same page reproduces its Table (TokenHMR: baseline 49.51, k=5 nearest 40.53,
oracle 39.38 mm over all 24 folds).

Judged here: the run finishes on a real adapter, every number is finite, refinement with the
k nearest sequences beats the baseline on every fold, and the oracle (23 sequences) is at
least as good as k-nearest on average. Reproduction against the paper's numbers is judged
only when `--data` is given, at ±0.5 mm.
"""
import glob
import json
import os
import pathlib
import subprocess
import sys
import tempfile

from first_run import FLAGS, ROOT, serve
from wheel_probe import wheel_is_stale
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from launch import _headed, refuse_if_software  # noqa: E402

GIVE_UP_MS = 20 * 60 * 1000
PAPER = {"baseline": 49.51, "k_nearest": 40.53, "oracle": 39.38}   # r-data-efficient, TokenHMR, 24 folds


def main(argv):
    from playwright.sync_api import sync_playwright
    headed = _headed("--headed" in argv)   # a window unless --headless / BORCH_HEADLESS: headless is SwiftShader here
    wheel = next((a.split("=", 1)[1] for a in argv if a.startswith("--wheel=")), None)
    data = next((a.split("=", 1)[1] for a in argv if a.startswith("--data=")), "")
    folds = next((a.split("=", 1)[1] for a in argv if a.startswith("--folds=")), "3")
    k = next((a.split("=", 1)[1] for a in argv if a.startswith("--k=")), "5")
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
    url = f"http://127.0.0.1:{port}/tests/browser/refine_py.html?wheel=/{wheel}&folds={folds}&k={k}"
    if data:
        url += "&data=/" + os.path.relpath(data, ROOT)
    profile = tempfile.mkdtemp(prefix="borch-refine-py-")
    channel = os.environ.get("BORCH_CHROME_CHANNEL") or None
    try:
        with sync_playwright() as pw:
            context = pw.chromium.launch_persistent_context(profile, headless=not headed, channel=channel, args=list(FLAGS), timeout=60_000)
            try:
                page = context.new_page()
                page.on("console", lambda m: print(f"  [page] {m.text}") if m.type in ("error", "warning") else None)
                page.goto(url, wait_until="load")
                page.wait_for_function("window.__refinePy !== undefined", timeout=GIVE_UP_MS, polling=500)
                got = page.evaluate("window.__refinePy")
            finally:
                context.close()
    finally:
        shutdown()
    print(f"wheel: {wheel} · data: {data or 'synthetic'} · folds {folds} · k {k}")
    print(got["text"])
    if got.get("error"):
        print("error: " + got["error"][:800])
        return 1
    r = json.loads(got["done"])
    if refuse_if_software(r.get("adapter"), "the refinement probe"):
        return 1
    rows = r["rows"]
    finite = all(all(isinstance(v, (int, float)) and v == v for v in (row["baseline"], row["k_nearest"], row["oracle"])) for row in rows)
    beats = all(row["k_nearest"] < row["baseline"] for row in rows)
    oracle_ok = r["oracle"] <= r["k_nearest"] + 0.25
    ok = finite and beats and oracle_ok and len(rows) == int(folds)
    print(f"{r['source']} · {r['N']} frames · {r['sequences']} sequences · {len(rows)} folds · adapter {r['adapter']}")
    print(f"PA-MPJPE mean over folds: baseline {r['baseline']:.2f} → k={r['k']} nearest {r['k_nearest']:.2f} · oracle {r['oracle']:.2f} mm · {r['pct_oracle']:.0f}% of the oracle gap · training {r['train_s']:.0f} s for {2 * len(rows)} fits")
    if r["source"] == "file" and int(folds) == 24:
        gaps = {key: abs(r[key] - PAPER[key]) for key in PAPER}
        repro = all(g <= 0.5 for g in gaps.values())
        print("against the paper (TokenHMR, 24 folds): " + " · ".join(f"{key} {r[key]:.2f} vs {PAPER[key]:.2f}" for key in PAPER) + (" — within 0.5 mm" if repro else " — OUTSIDE 0.5 mm"))
        ok = ok and repro
    print("**the paper's refinement recipe trains and refines in the browser**" if ok else "**it did not** — see above")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
