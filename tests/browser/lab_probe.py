"""Does the notebook page train? JupyterLite, the wheel from its own index, one cell.

    uv run --with playwright python tests/browser/lab_probe.py [--headed] [--build]

Opens `site/lab/lab/index.html?path=borch.ipynb` from this checkout (built by
`site/build_lab.py`; `--build` runs that first), presses Shift+Enter on the cell,
and reads the output: the loss lines and "learned y = 2.99x + 1.03" on the adapter's
name. The kernel is JupyterLite's own Pyodide in a worker; nothing of the site's
page-side loading is involved — this is what a stranger's JupyterLite would do
after `%pip install pyborch`.
"""
import os
import subprocess
import sys
import tempfile
import time

from first_run import FLAGS, ROOT, refuse_if_screen_off, serve
from launch import refuse_if_software

GIVE_UP_MS = 5 * 60 * 1000


def main(argv):
    from playwright.sync_api import sync_playwright
    headed = "--headed" in argv
    lab = ROOT / "site" / "lab" / "lab" / "index.html"
    if "--build" in argv or not lab.exists():
        r = subprocess.run([sys.executable, str(ROOT / "site" / "build_lab.py")], cwd=ROOT, text=True, capture_output=True)
        if r.returncode:
            print(f"site/build_lab.py failed:\n{(r.stdout + r.stderr)[-1200:]}", file=sys.stderr)
            return 2
    if refuse_if_screen_off("the notebook page"):
        return 1
    port, shutdown = serve(ROOT)
    url = f"http://127.0.0.1:{port}/site/lab/lab/index.html?path=borch.ipynb"
    profile = tempfile.mkdtemp(prefix="borch-lab-")
    channel = os.environ.get("BORCH_CHROME_CHANNEL") or None
    try:
        with sync_playwright() as pw:
            context = pw.chromium.launch_persistent_context(profile, headless=not headed, channel=channel, args=list(FLAGS), timeout=60_000)
            try:
                page = context.new_page()
                t0 = time.time()
                page.goto(url, wait_until="load")
                page.wait_for_selector(".jp-Notebook .jp-CodeCell .jp-InputArea-editor", timeout=GIVE_UP_MS)
                opened = time.time() - t0
                page.click(".jp-Notebook .jp-CodeCell .jp-InputArea-editor")
                page.keyboard.press("Shift+Enter")
                page.wait_for_function(
                    "[...document.querySelectorAll('.jp-OutputArea-output')].some(o => /learned|Error|error/.test(o.textContent))",
                    timeout=GIVE_UP_MS, polling=500)
                done = time.time() - t0
                outs = page.evaluate("[...document.querySelectorAll('.jp-OutputArea-output')].map(o => o.textContent.trim())")
            finally:
                context.close()
    finally:
        shutdown()
    text = "\n".join(outs)
    print(f"notebook open {opened:.1f} s · cell → output {done:.1f} s from opening the page")
    for line in text.splitlines()[-8:]:
        print("  " + line[:160])
    adapter = text.split(" on ", 1)[1].strip().splitlines()[0] if " on " in text else None
    if refuse_if_software(adapter, "the notebook page"):
        return 1
    ok = ("learned  y = 2.9" in text or "learned  y = 3.0" in text) and "x + 1.0" in text
    print("**the notebook trains on the GPU**" if ok else "**it did not** — see above")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
