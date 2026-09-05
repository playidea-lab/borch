"""Builds the workbench page — marimo, in the browser, with the pyborch wheel beside it.

    python3 site/build_marimo.py           # → site/marimo/  (gitignored; the deploy builds it)

`marimo export html-wasm` turns `site/marimo-src/review.py` into static files that
run the notebook on Pyodide in the visitor's tab; the wheel is copied next to them
and the notebook's first cell installs it from there. The version is pinned to the
one the spike measured.
"""
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "site" / "marimo-src"
OUT = ROOT / "site" / "marimo"
MARIMO = "marimo==0.24.0"


def sh(argv, cwd=ROOT):
    print("$ " + " ".join(argv), flush=True)
    r = subprocess.run(argv, cwd=str(cwd), text=True)
    if r.returncode:
        sys.exit(r.returncode)


def main():
    sh(["npm", "run", "-s", "bundle:py"])
    shutil.rmtree(ROOT / "dist", ignore_errors=True)
    sh(["uv", "build", "--wheel", "-q"])
    wheels = sorted((ROOT / "dist").glob("pyborch-*.whl"))
    if not wheels:
        print("no wheel came out of `uv build`", file=sys.stderr)
        return 2
    shutil.rmtree(OUT, ignore_errors=True)
    # The notebook names the wheel it installs. The name in the source is the version
    # at the time of writing; the one that was just built wins, so a version bump in
    # pyproject does not leave the page installing a wheel that is not beside it.
    src = (SRC / "review.py").read_text(encoding="utf-8")
    src = re.sub(r"pyborch-[0-9.]+-py3-none-any\.whl", wheels[-1].name, src)
    with tempfile.TemporaryDirectory() as tmp:
        (pathlib.Path(tmp) / "review.py").write_text(src, encoding="utf-8")
        sh(["uv", "run", "--with", MARIMO, "marimo", "export", "html-wasm", "review.py",
            "-o", str(OUT), "--mode", "edit"], cwd=tmp)
    shutil.copy(wheels[-1], OUT / wheels[-1].name)
    # Cross-origin isolation where the host cannot set headers: the worker pool needs it.
    # The file sits beside the page so the worker's scope is this folder — the bundle
    # copies the folder whole and keeps working.
    shutil.copy(ROOT / "site" / "coi.js", OUT / "coi.js")
    page = OUT / "index.html"
    html = page.read_text(encoding="utf-8")
    if "coi.js" not in html:
        page.write_text(html.replace("<head>", '<head><script src="coi.js"></script>', 1), encoding="utf-8")
    for junk in (SRC / "__marimo__",):
        if junk.is_dir():
            shutil.rmtree(junk)
    size = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file()) / 1e6
    print(f"built {OUT} — {size:.0f} MB, wheel {wheels[-1].name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
