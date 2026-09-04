"""Does the workbench page run? marimo in the browser, the wheel beside it, every cell pressed.

    uv run --with playwright python tests/browser/marimo_probe.py [--headed] [--build]

Opens `site/marimo/index.html` (built by `site/build_marimo.py`; `--build` runs that
first), waits for the kernel, presses run, and reads the four sections' outputs: the
adapter line, "Trained", the review queue's row count, and the export size.
"""
import os
import re
import subprocess
import sys
import tempfile
import time

from first_run import FLAGS, ROOT, refuse_if_screen_off, serve
from launch import refuse_if_software

GIVE_UP_MS = 6 * 60 * 1000
# What the notebook rendered, and only that. `document.body.innerText` would also carry
# every cell's source (the editors are text), so "KB of ONNX" matched at 5 s before any
# cell had run; and the table's "90 rows, 5 columns" line lives in a shadow root, which
# innerText never crosses. So: the output blocks, plus the shadow text of each table.
OUTPUTS = """() => {
  const outs = [...document.querySelectorAll('.output-area, marimo-cell-output')].map(e => e.innerText);
  const leaves = (root) => [...root.querySelectorAll('*')].filter(x => x.children.length === 0).map(x => x.textContent);
  const tables = [...document.querySelectorAll('marimo-table')].flatMap(e => e.shadowRoot ? leaves(e.shadowRoot) : []);
  return outs.concat(tables).join('\\n');
}"""
KERNEL_WAIT_MS = 10_000     # the run buttons exist before the kernel does; a click at 0.6 s closed the page


def synthetic_pngs(n=90, side=64, classes=("cat", "dog", "bird")):
    """The notebook's synthetic set as PNG files: one low-frequency template per class."""
    import io
    import numpy as np
    from PIL import Image
    rng = np.random.default_rng(11)
    cells = 6
    templates = [rng.standard_normal((cells, cells, 3)).astype(np.float32) for _ in classes]
    idx = np.arange(side) * cells // side
    files = []
    for i in range(n):
        k = i % len(classes)
        img = 0.5 + 0.3 * templates[k][idx][:, idx] + 0.15 * rng.standard_normal((side, side, 3)).astype(np.float32)
        buf = io.BytesIO()
        Image.fromarray((np.clip(img, 0, 1) * 255).astype("uint8")).save(buf, format="PNG")
        files.append({"name": f"{classes[k]}_{i:03d}.png", "mimeType": "image/png", "buffer": buf.getvalue()})
    return files


def upload_pass(page, t0, deadline):
    """Feed the PNGs to the notebook's file input and wait for the review table to show
    their names, then the retrained accuracy. Returns (seconds, "acc% on N rows") or None."""
    page.set_input_files('input[type="file"]', synthetic_pngs())
    while time.time() < deadline:
        page.wait_for_timeout(500)
        body = page.evaluate(OUTPUTS)
        if re.search(r"(cat|dog|bird)_\d{3}\.png", body):
            page.wait_for_timeout(1500)
            body = page.evaluate(OUTPUTS)
            acc = re.search(r"agrees with the given labels on \*?\*?(\d+)%", body)
            rows = re.search(r"(\d+) rows, 5 columns", body)
            if acc and rows:
                return (time.time() - t0, f"{acc.group(1)}% on {rows.group(1)} rows")
    print("  upload pass: the table never showed the uploaded names")
    return None


def main(argv):
    from playwright.sync_api import sync_playwright
    headed = "--headed" in argv
    page_file = ROOT / "site" / "marimo" / "index.html"
    if "--build" in argv or not page_file.exists():
        r = subprocess.run([sys.executable, str(ROOT / "site" / "build_marimo.py")], cwd=ROOT, text=True, capture_output=True)
        if r.returncode:
            print(f"site/build_marimo.py failed:\n{(r.stdout + r.stderr)[-1200:]}", file=sys.stderr)
            return 2
    if refuse_if_screen_off("the workbench page"):
        return 1
    port, shutdown = serve(ROOT)
    url = f"http://127.0.0.1:{port}/site/marimo/index.html"
    profile = tempfile.mkdtemp(prefix="borch-marimo-")
    channel = os.environ.get("BORCH_CHROME_CHANNEL") or None
    marks = {}
    try:
        with sync_playwright() as pw:
            context = pw.chromium.launch_persistent_context(profile, headless=not headed, channel=channel, args=list(FLAGS), timeout=60_000)
            try:
                page = context.new_page()
                t0 = time.time()
                page.goto(url, wait_until="load")
                page.wait_for_selector('[data-testid="run-button"]', timeout=GIVE_UP_MS)
                page.wait_for_timeout(KERNEL_WAIT_MS)
                page.query_selector_all('[data-testid="run-button"]')[-1].click()
                deadline = time.time() + GIVE_UP_MS / 1000
                want = {"adapter": r"borch on ([a-z]+ / [a-z0-9-]+)", "trained": r"agrees with the given labels on \*?\*?(\d+)%",
                        "queue": r"(\d+) rows, 5 columns", "export": r"(\d+) KB of ONNX"}
                body = ""
                while time.time() < deadline and len(marks) < len(want):
                    page.wait_for_timeout(500)
                    body = page.evaluate(OUTPUTS)
                    for key, pat in want.items():
                        if key not in marks:
                            m = re.search(pat, body)
                            if m:
                                marks[key] = (time.time() - t0, m.group(1))
                    if "Traceback" in body:
                        break
                errors = [l for l in body.splitlines() if "Traceback" in l or "Error:" in l][:3]
                if len(marks) < len(want):
                    print("  rendered so far:", repr(body[:1500]))
                else:
                    # Second pass: real files through `mo.ui.file` → `torch.decode_images`.
                    # Ninety PNGs of the same three-template kind, named the way the
                    # notebook reads labels (`cat_000.png`); marimo re-runs the cells
                    # below the upload on its own, so the table's names change.
                    marks["uploaded"] = upload_pass(page, t0, deadline)
            finally:
                context.close()
    finally:
        shutdown()
    for key in ("adapter", "trained", "queue", "export", "uploaded"):
        if marks.get(key):
            print(f"  {key:8s} {marks[key][0]:5.1f} s  {marks[key][1]}")
        else:
            print(f"  {key:8s}   —    (not reached)")
    for e in errors if "errors" in dir() else []:
        print("  error:", e[:200])
    adapter = marks.get("adapter", (0, None))[1]
    if refuse_if_software(adapter, "the workbench page"):
        return 1
    ok = all(marks.get(k) for k in ("adapter", "trained", "queue", "export", "uploaded"))
    ok = ok and int(marks["trained"][1]) >= 90 and int(marks["queue"][1]) == 90
    ok = ok and int(marks["uploaded"][1].split("%")[0]) >= 90 and marks["uploaded"][1].endswith("on 90 rows")
    print("**the workbench trains, reviews and exports**" if ok else "**it did not** — see above")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
