"""Which pretrained backbones load, and what each costs — measured before anything ranks them.

    uv run --with playwright python tests/browser/backbone_sweep.py --build [--budget=50]

**A measurement, not a check.** It prints a row per model and no verdict.

## Why it comes before a comparison

The registry lists eighteen models. This repository names exactly one —
`imagenet-efficientnet-b0`, in seven places — and the TM++ page asks the registry for the
smallest. **Sixteen of the eighteen have never been loaded by anything here.**

A `compare()` would be their first caller, and this repository spent 2026-09-14 learning
what code with no caller is like: a `workbench` wired onto the CPU door that could not have
run, a transform written against the one probe that converted its images first, four
defects on the landing page found by a person opening it in Safari. A leaderboard built on
sixteen untried backbones would be a leaderboard of whichever ones happen to work.

So this asks the smaller question first: do they load, do they produce features, and what
does each cost. A model that answers with the right shape is still checked for being
finite and for varying between photographs — a backbone that returns the same row for
every image has loaded without running.

It wants the photographs `preprocess_cost.py` wants, under `tests/browser/.cache/`.
"""

import glob
import json
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from first_run import FLAGS, ROOT, refuse_if_screen_off, serve  # noqa: E402
from launch import _headed, probe_lock, refuse_if_software  # noqa: E402
from wheel_probe import wheel_is_stale  # noqa: E402

GIVE_UP_MS = 60 * 60 * 1000
IMAGES = ROOT / "tests" / "browser" / ".cache" / "imagenetv2-1perclass"


def main(argv):
    from playwright.sync_api import sync_playwright                  # noqa: PLC0415

    if not (IMAGES / "index.json").exists():
        print(f"no photographs at {IMAGES.relative_to(ROOT)} — see preprocess_cost.py", file=sys.stderr)
        return 2
    if "--build" in argv:
        import subprocess                                            # noqa: PLC0415
        for cmd in (["npm", "run", "-s", "bundle:py"], ["uv", "build", "--wheel", "-q"]):
            r = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
            if r.returncode:
                print((r.stdout + r.stderr)[-1200:], file=sys.stderr)
                return 2
    found = sorted(glob.glob(str(ROOT / "dist" / "*.whl")))
    if not found:
        print("no wheel under dist/ — run with --build", file=sys.stderr)
        return 2
    wheel = os.path.relpath(found[-1], ROOT)
    if wheel_is_stale(wheel):
        print(f"{wheel} is older than the sources — run with --build", file=sys.stderr)
        return 2
    budget = next((a.split("=", 1)[1] for a in argv if a.startswith("--budget=")), "50")
    limit = next((a.split("=", 1)[1] for a in argv if a.startswith("--limit=")), "48")

    if refuse_if_screen_off("the backbone sweep"):
        return 1
    probe_lock("the backbone sweep")
    port, shutdown = serve(ROOT)
    url = (f"http://127.0.0.1:{port}/tests/browser/backbone_sweep.html"
           f"?wheel=/{wheel}&budget={budget}&limit={limit}")
    try:
        with sync_playwright() as pw:
            context = pw.chromium.launch_persistent_context(
                "", headless=not _headed("--headed" in argv), args=list(FLAGS), timeout=60_000)
            try:
                page = context.new_page()
                page.goto(url, wait_until="load")
                # The adapter first: a software one answers the same and takes hours.
                page.wait_for_function("window.__adapter !== undefined || window.__sweep !== undefined",
                                       timeout=10 * 60 * 1000, polling=200)
                adapter = page.evaluate("window.__adapter")
                if adapter and refuse_if_software(adapter, "the backbone sweep"):
                    return 1
                page.wait_for_function("window.__sweep !== undefined", timeout=GIVE_UP_MS, polling=500)
                got = json.loads(page.evaluate("window.__sweep"))
                print(page.inner_text("#out").strip())
            finally:
                context.close()
    finally:
        shutdown()

    if "error" in got:
        print(f"could not measure: {got['error'][:600]}", file=sys.stderr)
        return 1
    rows = got["rows"]
    worked = [r for r in rows if "error" not in r]
    print()
    print(f"  {'model':32s} {'MB':>6s} {'px':>4s} {'dims':>6s} {'load':>7s} {'features':>9s} {'spread':>8s}")
    for r in rows:
        if "error" in r:
            print(f"  {r['name']:32s} {r['mb']:6.1f}    —      —       —         —         FAILED")
            print(f"      {r['error']}")
            continue
        print(f"  {r['name']:32s} {r['mb']:6.1f} {r['size']:4d} {r['features']:6d} "
              f"{r['load_s']:6.1f}s {r['feature_s']:8.1f}s {r['spread']:8.4f}")
    dead = [r["name"] for r in worked if not r["finite"] or r["spread"] == 0]
    print(f"\n  {len(worked)} of {len(rows)} loaded and produced features over {got['n']} photographs"
          f" · {got['adapter']} · faults {got['faults']}")
    if dead:
        print(f"  loaded but did not run — every photograph got the same answer: {', '.join(dead)}")
    if worked:
        total = sum(r["mb"] for r in worked)
        seconds = sum(r["load_s"] + r["feature_s"] for r in worked)
        print(f"  a comparison over these would fetch {total:.0f} MB and take {seconds:.0f} s"
              f" for {got['n']} photographs")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
