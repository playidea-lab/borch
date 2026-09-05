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

GIVE_UP_MS = 300_000     # the frozen path exports a 21 MB EfficientNet — tracing and encoding take a while
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


def png_bytes(rgb):
    """An 8-bit RGB (H, W, 3) uint8 array as a PNG — zlib and struct only, so the probe
    has no Pillow dependency (the 4090's project environment has none)."""
    import struct
    import zlib
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[r].tobytes() for r in range(h))     # filter 0 per row
    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xffffffff)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def synthetic_pngs(n=90, side=64, classes=("cat", "dog", "bird")):
    """The notebook's synthetic set as PNG files: one low-frequency template per class."""
    import numpy as np
    rng = np.random.default_rng(11)
    cells = 6
    templates = [rng.standard_normal((cells, cells, 3)).astype(np.float32) for _ in classes]
    idx = np.arange(side) * cells // side
    files = []
    for i in range(n):
        k = i % len(classes)
        img = 0.5 + 0.3 * templates[k][idx][:, idx] + 0.15 * rng.standard_normal((side, side, 3)).astype(np.float32)
        files.append({"name": f"{classes[k]}_{i:03d}.png", "mimeType": "image/png",
                      "buffer": png_bytes((np.clip(img, 0, 1) * 255).astype("uint8"))})
    return files


def zipped_folder(files):
    """The PNGs as one zip of class folders — the way thousands arrive. Labels come from
    the folder names, so the file names inside carry no prefix of their own."""
    import io
    import zipfile
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for f in files:
            cls = f["name"].split("_")[0]
            z.writestr(f"{cls}/{f['name']}", f["buffer"])
    return {"name": "photos.zip", "mimeType": "application/zip", "buffer": buf.getvalue()}


def upload_pass(page, t0, deadline, n=90):
    """Feed a zipped folder of the PNGs to the notebook's file input and wait for the review
    table to show their names, then the retrained accuracy. Returns (seconds, "acc% on N
    rows") or None. A zip, not ninety files: that is how a folder reaches a tab."""
    page.set_input_files('input[type="file"]', [zipped_folder(synthetic_pngs(n))])
    while time.time() < deadline:
        page.wait_for_timeout(500)
        body = page.evaluate(OUTPUTS)
        if re.search(r"(cat|dog|bird)_\d{3}\.png", body):
            page.wait_for_timeout(1500)
            body = page.evaluate(OUTPUTS)
            acc = re.search(r"agrees with the given labels on \*?\*?(\d+)%", body)
            rows = re.search(r"([\d,]+) rows, 5 columns", body)          # "1,000 rows" past 999
            feat = re.search(r"features for ([\d,]+) images in \*?\*?([0-9.]+) s", body)
            if acc and rows:
                extra = f" · features {feat.group(2)} s" if feat else ""
                return (time.time() - t0, f"{acc.group(1)}% on {rows.group(1).replace(',', '')} rows{extra}")
    print("  upload pass: the table never showed the uploaded names")
    return None


def scratch_pass(page, t0, deadline):
    """Pick "small CNN from scratch" and wait for the retrained line and a fresh export.
    Returns (seconds, "acc% · N KB") or None."""
    try:
        page.get_by_label("small CNN from scratch", exact=True).check()
    except Exception as e:                                          # noqa: BLE001
        print(f"  scratch pass: the radio's label did not take a check ({type(e).__name__}); clicking its text")
        page.get_by_text("small CNN from scratch", exact=True).click()
    while time.time() < deadline:
        page.wait_for_timeout(500)
        body = page.evaluate(OUTPUTS)
        m = re.search(r"small CNN · \d+ epochs[^\n]*agrees with the given labels on \*?\*?(\d+)%", body)
        kb = re.search(r"(\d+) KB of ONNX", body)
        if m and kb and int(kb.group(1)) < 1000:          # the CNN's file is ~118 KB, the backbone's 16 MB
            return (time.time() - t0, f"{m.group(1)}% · {kb.group(1)} KB")
    print("  scratch pass: the small-CNN line never rendered; the last outputs:", repr(body[-500:]))
    return None


def main(argv):
    from playwright.sync_api import sync_playwright
    headed = "--headed" in argv
    # `--zip=N`: how many PNGs the upload pass zips (90 by default; 1000 for a timing).
    zip_n = int(next((a.split("=", 1)[1] for a in argv if a.startswith("--zip=")), "90"))
    page_file = ROOT / "site" / "marimo" / "index.html"
    if "--build" in argv or not page_file.exists():
        r = subprocess.run([sys.executable, str(ROOT / "site" / "build_marimo.py")], cwd=ROOT, text=True, capture_output=True)
        if r.returncode:
            print(f"site/build_marimo.py failed:\n{(r.stdout + r.stderr)[-1200:]}", file=sys.stderr)
            return 2
    if refuse_if_screen_off("the workbench page"):
        return 1
    # `--dir=bundle`: press the offline bundle instead of the site's build. With
    # `--offline`, every request for another host is refused and counted — one is a
    # failure, since a folder that phones home is not the bundle it claims to be.
    page_dir = next((a.split("=", 1)[1] for a in argv if a.startswith("--dir=")), "site/marimo")
    offline = "--offline" in argv
    if "--bundle" in argv:
        # The offline bundle: built here (site/build_bundle.py records what the site's
        # build fetches and mirrors it), then pressed with the network refused.
        # In this process, not a child python: under `uv run` on the GPU boxes a child
        # started from `sys.executable` re-entered uv's build environment, and the
        # nesting ran to uv's limit of a hundred (measured).
        sys.path.insert(0, str(ROOT / "site"))
        import build_bundle
        if build_bundle.main():
            return 2
        page_dir, offline = "bundle", True
    port, shutdown = serve(ROOT)
    url = f"http://127.0.0.1:{port}/{page_dir}/index.html"
    phoned = []
    profile = tempfile.mkdtemp(prefix="borch-marimo-")
    channel = os.environ.get("BORCH_CHROME_CHANNEL") or None
    marks = {}
    try:
        with sync_playwright() as pw:
            context = pw.chromium.launch_persistent_context(profile, headless=not headed, channel=channel, args=list(FLAGS), timeout=60_000)
            try:
                page = context.new_page()
                if offline:
                    def gate(route, request):
                        if request.url.startswith(("http://127.0.0.1", "blob:", "data:")):
                            route.continue_()
                        else:
                            phoned.append(request.url)
                            route.abort()
                    page.route("**/*", gate)
                t0 = time.time()
                page.goto(url, wait_until="load")
                page.wait_for_selector('[data-testid="run-button"]', timeout=GIVE_UP_MS)
                page.wait_for_timeout(KERNEL_WAIT_MS)
                page.query_selector_all('[data-testid="run-button"]')[-1].click()
                deadline = time.time() + GIVE_UP_MS / 1000
                want = {"adapter": r"borch on ([a-z]+ / [a-z0-9-]+)", "trained": r"agrees with the given labels on \*?\*?(\d+)%",
                        "queue": r"(\d+) rows, 5 columns", "export": r"(\d+) KB of ONNX",
                        "report": r"faults (\d+) · warnings (\d+)"}
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
                    marks["uploaded"] = upload_pass(page, t0, deadline, zip_n)
                    # Third pass: the other model path. The radio is a marimo element in
                    # a shadow root; Playwright's text locator pierces it. marimo reruns
                    # the cells below on its own.
                    marks["scratch"] = scratch_pass(page, t0, deadline)
            finally:
                context.close()
    finally:
        shutdown()
    if offline:
        print(f"  offline: {len(phoned)} request(s) tried to leave" + (":" if phoned else ""))
        for u in phoned[:8]:
            print(f"      {u[:140]}")
    for key in ("adapter", "trained", "queue", "export", "report", "uploaded", "scratch"):
        if marks.get(key):
            print(f"  {key:8s} {marks[key][0]:5.1f} s  {marks[key][1]}")
        else:
            print(f"  {key:8s}   —    (not reached)")
    for e in errors if "errors" in dir() else []:
        print("  error:", e[:200])
    adapter = marks.get("adapter", (0, None))[1]
    if refuse_if_software(adapter, "the workbench page"):
        return 1
    ok = all(marks.get(k) for k in ("adapter", "trained", "queue", "export", "report", "uploaded", "scratch"))
    ok = ok and marks["report"][1] == "0"                   # faults — the warnings count rides in the text
    ok = ok and int(marks["scratch"][1].split("%")[0]) >= 90
    ok = ok and not phoned
    ok = ok and int(marks["trained"][1]) >= 90 and int(marks["queue"][1]) == 90
    ok = ok and int(marks["uploaded"][1].split("%")[0]) >= 90 and f"on {zip_n} rows" in marks["uploaded"][1]
    print("**the workbench trains, reviews and exports**" if ok else "**it did not** — see above")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
