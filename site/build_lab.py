"""Builds the notebook page — JupyterLite with the pyborch wheel already in it.

    python3 site/build_lab.py            # → site/lab/  (gitignored; the deploy builds it)

Three steps, each the same command a person would type: esbuild's single-file
borch.ts for the wheel, the wheel, then `jupyter lite build` with that wheel listed
for piplite and the starter notebook as content. **The kernel is pinned to the
Pyodide the site vendors (0.27)** — 0.28 hands JS `null` over as `jsnull`, and the
golden was checked on 0.27.2; one Pyodide across the site keeps one answer.
"""
import pathlib
import shutil
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "site" / "lab-src"
OUT = ROOT / "site" / "lab"
KERNEL = "jupyterlite-pyodide-kernel<0.7"      # Pyodide 0.27.x


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
    sh(["uv", "run", "--with", "jupyterlite-core", "--with", KERNEL, "--with", "jupyter-server",
        "jupyter", "lite", "build", "--contents", "content",
        "--piplite-wheels", str(wheels[-1]), "--output-dir", str(OUT)], cwd=SRC)
    for junk in (SRC / ".jupyterlite.doit.db", SRC / "_output"):
        if junk.is_dir():
            shutil.rmtree(junk)
        elif junk.exists():
            junk.unlink()
    size = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file()) / 1e6
    print(f"built {OUT} — {size:.0f} MB, wheel {wheels[-1].name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
