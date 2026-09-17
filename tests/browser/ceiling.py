"""The ceiling: how much GPU memory this tab can actually hold, and what a hub model costs.

    uv run --with playwright python tests/browser/ceiling.py [--headed] [--cap=8] [--models=a,b] [--record]

Step 0 of `docs/SCALE.md`. WebGPU reports no total memory and never throws on
out-of-memory, so the page allocates until a buffer stops holding its marker: the largest
single buffer, the largest total, the adapter's limit tiers and features, the origin's
storage quota — then loads two hub models through borch and records the host peak and the
GPU bytes each one costs. A measurement, not a check: it fails only when nothing could be
measured. `--record` appends the block to `docs/SCALE-MEASURED.md`, the ledger every gate
in the plan writes into.
"""
import datetime
import os
import subprocess
import sys
import tempfile

from first_run import FLAGS, ROOT, refuse_if_screen_off, serve
from launch import _headed, refuse_if_software

GIVE_UP_MS = 10 * 60 * 1000
LEDGER = ROOT / "docs" / "SCALE-MEASURED.md"


def _arg(argv, key, default):
    return next((a.split("=", 1)[1] for a in argv if a.startswith(f"--{key}=")), default)


def _block(got, head):
    """The markdown the ledger keeps: one block per adapter and day."""
    mb = lambda b: f"{b / (1 << 20):.0f} MiB"  # noqa: E731
    lim = got.get("limits", {})
    rows = [
        f"### {got.get('adapter')} — {datetime.date.today().isoformat()} · {head}",
        "",
        "| what | measured |",
        "|---|---|",
        f"| features | {' '.join(got.get('features', [])) or '(none)'} |",
        f"| shader-f16 · subgroups | {'yes' if 'shader-f16' in got.get('features', []) else 'no'} · "
        f"{'yes' if 'subgroups' in got.get('features', []) else 'no'} |",
        f"| maxBufferSize · maxStorageBufferBindingSize | {mb(lim.get('maxBufferSize', 0))} · {mb(lim.get('maxStorageBufferBindingSize', 0))} |",
        f"| storage buffers/stage · workgroup storage | {lim.get('maxStorageBuffersPerShaderStage')} · "
        f"{(lim.get('maxComputeWorkgroupStorageSize') or 0) // 1024} KiB |",
        f"| largest single buffer holding its marker | {mb(got.get('single') or 0)} |",
        f"| largest total holding every marker | {(got.get('total') or 0) / (1 << 30):.2f} GiB "
        f"(chunks of {mb(got.get('chunk') or 0)}) |",
    ]
    st = got.get("storage")
    if st:
        rows.append(f"| storage quota · used | {st['quota'] / (1 << 30):.1f} GiB · {st['usage'] / (1 << 20):.1f} MB |")
    for m in got.get("models", []):
        if m.get("error"):
            rows.append(f"| hub load {m['name']} ({m['bytes'] / 1e6:.0f} MB) | FAILED: {m['error'][:80]} |")
            continue
        rows.append(
            f"| hub load {m['name']} ({m['bytes'] / 1e6:.0f} MB) | {m['ms'] / 1000:.1f} s · gpu +{m['gpuResident'] / 1e6:.0f} MB "
            f"(pool +{m['pooled'] / 1e6:.0f}) · host peak +{(m['hostPeak'] - m['host0']) / 1e6:.0f} MB = "
            f"{(m['hostPeak'] - m['host0']) / m['bytes']:.2f}× file ({m['hostHow']}) · faults {m['faults']} |")
    rows.append(f"| validation faults | {got.get('faults', 0)} |")
    rows.append("")
    return "\n".join(rows)


def main(argv):
    from playwright.sync_api import sync_playwright
    headed = _headed("--headed" in argv)   # a window unless --headless / BORCH_HEADLESS: headless is SwiftShader here
    cap = _arg(argv, "cap", "8")
    models = _arg(argv, "models", "")
    if refuse_if_screen_off("the ceiling"):
        return 1
    port, shutdown = serve(ROOT)
    url = f"http://127.0.0.1:{port}/tests/browser/ceiling.html?cap={cap}" + (f"&models={models}" if models else "")
    profile = tempfile.mkdtemp(prefix="borch-ceiling-")
    channel = os.environ.get("BORCH_CHROME_CHANNEL") or None
    try:
        with sync_playwright() as pw:
            context = pw.chromium.launch_persistent_context(profile, headless=not headed, channel=channel, args=list(FLAGS), timeout=60_000)
            try:
                page = context.new_page()
                page.on("pageerror", lambda e: print(f"  [page] {e}"))
                page.goto(url, wait_until="load")
                page.wait_for_function("window.__ceiling !== undefined", timeout=GIVE_UP_MS, polling=500)
                got = page.evaluate("window.__ceiling")
            finally:
                context.close()
    finally:
        shutdown()

    print(got.get("text", ""))
    if got.get("error"):
        print(f"\nthe ceiling could not be measured: {got['error'][:400]}", file=sys.stderr)
        return 1
    # A CPU adapter's ceiling is the host's RAM, and its "hub load" is a rasteriser's — a
    # number, but not a GPU's. Reported, then refused, as every measurement here is.
    if refuse_if_software(got.get("adapter"), "the ceiling"):
        return 1
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    block = _block(got, head)
    print("\n" + block)
    if "--record" in argv:
        LEDGER.parent.mkdir(exist_ok=True)
        new = not LEDGER.exists()
        with LEDGER.open("a", encoding="utf-8") as f:
            if new:
                f.write("# SCALE — measured\n\n> The ledger `docs/SCALE.md` §7 names: every number a gate in the plan "
                        "produced, with the adapter, the day and the commit. Written by `tests/browser/ceiling.py "
                        "--record` and by hand from the other gates. A claim about scale that is not here is not made.\n\n")
            f.write(block + "\n")
        print(f"recorded into {LEDGER.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
