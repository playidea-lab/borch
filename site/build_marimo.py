"""Builds the workbench page — marimo, in the browser, with the pyborch wheel beside it.

    python3 site/build_marimo.py           # → site/marimo/  (gitignored; the deploy builds it)

`marimo export html-wasm` turns `site/marimo-src/review.py` into static files that
run the notebook on Pyodide in the visitor's tab; the wheel is copied next to them
and the notebook's first cell installs it from there. The version is pinned to the
one the spike measured.
"""
import pathlib
import shutil
import subprocess
import sys

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
    sh(["uv", "run", "--with", MARIMO, "marimo", "export", "html-wasm", "review.py",
        "-o", str(OUT), "--mode", "edit"], cwd=SRC)
    shutil.copy(wheels[-1], OUT / wheels[-1].name)
    for junk in (SRC / "__marimo__",):
        if junk.is_dir():
            shutil.rmtree(junk)
    size = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file()) / 1e6
    print(f"built {OUT} — {size:.0f} MB, wheel {wheels[-1].name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
