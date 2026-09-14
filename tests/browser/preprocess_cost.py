"""What the wrong preprocessing costs a pretrained backbone — measured, not assumed.

    uv run --with playwright python tests/browser/preprocess_cost.py --build [--limit=1000]

**A measurement, not a check.** It prints five numbers and no verdict, because the thing it
is asking about has no right answer written down anywhere: how much accuracy a backbone
loses when it is fed images prepared differently from the way it was trained.

## Why it exists

Every manifest in the registry carries a `preprocess` block — the input size, the resize,
the centre crop, the mean and standard deviation. `borch-hub` can build that pipeline
(`transformFor`). Nothing uses it. `workbench` decodes to a 64px square and does not
normalise at all; the TM++ page copies the numbers into its own source. So the same
backbone is fed three different things, and **nobody has measured what that costs.**

## Why this data

`imagenetv2-1perclass` — a thousand real photographs, one per ImageNet class, with their
classes and a sha256 each. CIFAR would not do: at 32px native, resizing to 224 is a
sevenfold enlargement, so the comparison would measure the enlargement rather than the
question. These are photographs at their own resolution, so resizing **down** to 224 is
the natural operation and 64px is genuinely throwing detail away.

Under `tests/browser/.cache/` — copy it from a borch-hub checkout (`imagenetv2-1perclass/`,
125 MB), or make it there with `uv run python scripts/fetch_images.py`.

## The gate this rests on

The manifest also records what this model scored on this very set: `top1_imagenetv2`. The
`manifest` condition has to land on that number. **If it does not, this file is wrong and
the other four rows mean nothing** — they would be measuring a mistake in the pipeline
written here rather than the cost of getting preprocessing wrong.
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

GIVE_UP_MS = 30 * 60 * 1000
IMAGES = ROOT / "tests" / "browser" / ".cache" / "imagenetv2-1perclass"
# The manifest's own number is a percentage of a thousand photographs, so a run that
# reproduces the pipeline lands within a point or so of it; further than that is a
# different pipeline, not noise.
AGREE_POINTS = 1.5


def main(argv):
    from playwright.sync_api import sync_playwright                  # noqa: PLC0415

    if not (IMAGES / "index.json").exists():
        print(f"no photographs at {IMAGES.relative_to(ROOT)} — see this file's docstring", file=sys.stderr)
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
    limit = next((a.split("=", 1)[1] for a in argv if a.startswith("--limit=")), "1000")
    batch = next((a.split("=", 1)[1] for a in argv if a.startswith("--batch=")), "16")

    # A software adapter would answer, slowly and with the same numbers — but a thousand
    # forward passes five times over is not a thing to ask of one.
    if refuse_if_screen_off("the preprocessing measurement"):
        return 1
    # **One browser probe at a time.** This launched its context directly and so took no
    # lock — measured against a second probe it would fight for the GPU and the uv cache,
    # which is the fault `probe_lock` was written for.
    probe_lock("the preprocessing measurement")
    port, shutdown = serve(ROOT)
    url = (f"http://127.0.0.1:{port}/tests/browser/preprocess_cost.html"
           f"?wheel=/{wheel}&limit={limit}&batch={batch}")
    try:
        with sync_playwright() as pw:
            context = pw.chromium.launch_persistent_context(
                "", headless=not _headed("--headed" in argv), args=list(FLAGS), timeout=60_000)
            try:
                page = context.new_page()
                page.on("console", lambda m: print(f"  [{m.type}] {m.text[:160]}") if m.type == "error" else None)
                page.goto(url, wait_until="load")
                # The adapter arrives long before the numbers do; a software one is
                # refused here rather than after five conditions have crawled through it.
                page.wait_for_function("window.__adapter !== undefined || window.__cost !== undefined",
                                       timeout=10 * 60 * 1000, polling=200)
                adapter = page.evaluate("window.__adapter")
                if adapter and refuse_if_software(adapter, "the preprocessing measurement"):
                    return 1
                page.wait_for_function("window.__cost !== undefined", timeout=GIVE_UP_MS, polling=500)
                got = json.loads(page.evaluate("window.__cost"))
                print(page.inner_text("#out").strip())
            finally:
                context.close()
    finally:
        shutdown()

    if "error" in got:
        print(f"could not measure: {got['error'][:600]}", file=sys.stderr)
        return 1
    if refuse_if_software(got.get("adapter"), "the preprocessing measurement"):
        return 1

    rows = {r["name"]: r for r in got["rows"]}
    recorded = got.get("recorded")
    print()
    print(f"  {'condition':14s} {'top1':>7s} {'top5':>7s} {'seconds':>8s}   against the manifest")
    best = rows["manifest"]["top1"]
    for r in got["rows"]:
        delta = (r["top1"] - best) * 100
        print(f"  {r['name']:14s} {r['top1']:7.3f} {r['top5']:7.3f} {r['seconds']:8.1f}   "
              + ("—" if r["name"] == "manifest" else f"{delta:+.1f} points"))
    print(f"\n  {got['n']} photographs · {got['adapter']} · faults {got['faults']}")

    if recorded is None:
        print("  the manifest records no top1_imagenetv2 — the gate below cannot be read")
        return 0
    # The library's row is held to the same number the hand-written one is, because the
    # point of the library's existing is that a caller does not write this by hand.
    if "library" in rows:
        gap = abs(rows["library"]["top1"] - recorded) * 100
        print(f"  hub.transform_for lands at {rows['library']['top1']:.3f} ({gap:.1f} points from the record)")
        if gap > AGREE_POINTS:
            print("**hub.transform_for is not building the pipeline the manifest describes**", file=sys.stderr)
            return 1
    off = abs(rows["manifest"]["top1"] - recorded) * 100
    print(f"  the manifest records {recorded:.3f} for this model on this set; "
          f"the `manifest` row is {rows['manifest']['top1']:.3f} ({off:.1f} points away)")
    if off > AGREE_POINTS:
        print("**the pipeline written here is not the one the manifest describes** — "
              "the other rows measure that mistake, not the cost of preprocessing", file=sys.stderr)
        return 1
    print("**the pipeline reproduces the recorded number, so the rows below it can be read**")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
