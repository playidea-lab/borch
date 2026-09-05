"""The field trainer as one folder — the workbench, Pyodide, marimo's packages, the
backbone — that runs with no network at all.

    python3 site/build_bundle.py            # → bundle/  (gitignored)
    python3 -m http.server -d bundle 8000   # any static server; file:// cannot run workers

What the workbench fetches from the network, and where the bundle keeps it instead:

- Pyodide (`cdn.jsdelivr.net/pyodide/<v>/full/`) and marimo's package lock
  (`wasm.marimo.app/pyodide-lock.json`) — marimo's runtime has both addresses written
  into its JavaScript. The bundle mirrors the runtime files and every wheel the lock
  names that the notebook actually loads into `pyodide/`, rewrites the lock so each
  package is a bare file name there, and patches the two addresses in the runtime to
  point at that folder (a small expression, since the kernel runs in a worker under
  `assets/` and the page one level up).
- The catalogue (`models.pilab.kr`) — `models/index.json`, the manifest with the
  weights spelled as a relative path, the weights and the sample. `torch.hub` asks for
  a `models/index.json` beside the page before it asks the public catalogue.

**Which files** is not a list kept here — it is measured. The builder opens the built
workbench once, presses run, records every request that left for another host, and
mirrors exactly those. A list written by hand went stale with the first marimo bump.
"""
import hashlib
import json
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "site" / "marimo"
OUT = ROOT / "bundle"
sys.path.insert(0, str(ROOT / "tests" / "browser"))


def sh(argv, cwd=ROOT):
    print("$ " + " ".join(str(a) for a in argv), flush=True)
    r = subprocess.run(argv, cwd=str(cwd), text=True)
    if r.returncode:
        sys.exit(r.returncode)


def recorded_requests(site_dir):
    """Every URL the workbench asks another host for while it runs once, in order."""
    from first_run import FLAGS, serve
    from marimo_probe import OUTPUTS
    from playwright.sync_api import sync_playwright
    port, shutdown = serve(ROOT)
    urls = []
    try:
        with sync_playwright() as pw:
            # Headed where there is a display: the GPU boxes carry the full Chromium only,
            # not the headless shell a headless launch reaches for (measured on the 4090).
            import os
            # ... and through the browser channel the probes use (`BORCH_CHROME_CHANNEL`,
            # the system Chrome on the GPU boxes — Playwright's own Chromium is not there).
            ctx = pw.chromium.launch_persistent_context(tempfile.mkdtemp(), headless=not os.environ.get("DISPLAY"),
                                                        channel=os.environ.get("BORCH_CHROME_CHANNEL") or None, args=list(FLAGS), timeout=60_000)
            page = ctx.new_page()
            page.on("request", lambda r: urls.append(r.url))
            page.goto(f"http://127.0.0.1:{port}/{site_dir.relative_to(ROOT).as_posix()}/index.html", wait_until="load")
            page.wait_for_selector('[data-testid="run-button"]', timeout=180_000)
            page.wait_for_timeout(10_000)
            page.query_selector_all('[data-testid="run-button"]')[-1].click()
            t0 = time.time()
            while time.time() - t0 < 300:
                page.wait_for_timeout(2_000)
                if "KB of ONNX" in page.evaluate(OUTPUTS):
                    break
            else:
                sys.exit("the workbench did not reach its export while being recorded")
            ctx.close()
    finally:
        shutdown()
    seen, out = set(), []
    for u in urls:
        if u.startswith(("http://127.0.0.1", "blob:")) or u in seen:
            continue
        seen.add(u)
        out.append(u)
    return out


CACHE = pathlib.Path.home() / ".cache" / "borch-bundle"


def fetch(url, into):
    """`url` into `into`, through a cache keyed by the URL's path — a rebuild does not
    fetch 75 MB again, and the catalogue's weights are the same bytes every time."""
    into.parent.mkdir(parents=True, exist_ok=True)
    if into.exists():
        return
    cached = CACHE / urllib.parse.urlparse(url).netloc / urllib.parse.urlparse(url).path.lstrip("/")
    if not cached.exists():
        cached.parent.mkdir(parents=True, exist_ok=True)
        # The catalogue sits behind Cloudflare, which answers urllib's default agent with 403.
        req = urllib.request.Request(url, headers={"User-Agent": "borch-bundle/1 (+https://github.com/playidea-lab/borch)"})
        with urllib.request.urlopen(req, timeout=120) as r, open(cached, "wb") as f:
            shutil.copyfileobj(r, f)
    shutil.copy(cached, into)


def main():
    if not (SRC / "index.html").exists():
        sh([sys.executable, str(ROOT / "site" / "build_marimo.py")])
    shutil.rmtree(OUT, ignore_errors=True)
    shutil.copytree(SRC, OUT, ignore=shutil.ignore_patterns("CLAUDE.md", "__marimo__"))
    urls = recorded_requests(SRC)
    print(f"recorded {len(urls)} external requests")

    pyodide_dir = OUT / "pyodide"
    models_dir = OUT / "models"
    lock_url = next(u for u in urls if "pyodide-lock.json" in u)
    index_url = next((u for u in urls if u.endswith("/index.json")), None)
    runtime_base = None
    for u in urls:
        path = urllib.parse.urlparse(u).path
        name = path.rsplit("/", 1)[-1]
        if "pyodide-lock.json" in u or u == index_url:
            continue
        if "models." in urllib.parse.urlparse(u).netloc or "/manifest.json" in u or name.endswith(".safetensors"):
            fetch(u, models_dir / path.lstrip("/"))
            continue
        # Runtime files and wheels, whatever host served them, land flat in pyodide/.
        fetch(u, pyodide_dir / name)
        if name.startswith("pyodide.asm"):
            runtime_base = u.rsplit("/", 1)[0] + "/"
    if runtime_base is None:
        sys.exit("no Pyodide runtime among the recorded requests")
    # The loader itself is not in the list — marimo bundles it — but `pyodide.mjs` and
    # the runtime's own manifest are asked for by some paths; keep them when they exist.
    for extra in ("pyodide.mjs", "pyodide-lock.json", "package.json"):
        try:
            fetch(runtime_base + extra, pyodide_dir / ("runtime-" + extra if extra == "pyodide-lock.json" else extra))
        except Exception:                                                # noqa: BLE001
            pass

    # The lock, with every package a bare file name in pyodide/.
    with urllib.request.urlopen(urllib.request.Request(lock_url, headers={"User-Agent": "borch-bundle/1"}), timeout=120) as r:
        lock = json.load(r)
    for pkg in lock["packages"].values():
        pkg["file_name"] = pkg["file_name"].rsplit("/", 1)[-1]
    (pyodide_dir / "pyodide-lock.json").write_text(json.dumps(lock), encoding="utf-8")

    # marimo's runtime: the two addresses become "the pyodide/ folder beside the page".
    # The kernel runs in a worker whose location is under assets/; the page is one up.
    base_expr = '((h)=>h.includes("/assets/")?h.split("/assets/")[0]+"/":h.replace(/[^/]*$/,""))(self.location.href)'
    patched = 0
    for js in (OUT / "assets").glob("*.js"):
        text = js.read_text(encoding="utf-8")
        new = re.sub(r"`https://cdn\.jsdelivr\.net/pyodide/\$\{[^}]+\}/full/`", base_expr + '+"pyodide/"', text)
        new = re.sub(r"`https://wasm\.marimo\.app/pyodide-lock\.json\?v=\$\{[^}]+\}&pyodide=\$\{[^}]+\}`",
                     base_expr + '+"pyodide/pyodide-lock.json"', new)
        if new != text:
            js.write_text(new, encoding="utf-8")
            patched += 1
    if not patched:
        sys.exit("the runtime's Pyodide addresses were not found — marimo changed its spelling")

    # The catalogue: the index with local manifest paths, manifests with relative weights.
    if index_url:
        with urllib.request.urlopen(urllib.request.Request(index_url, headers={"User-Agent": "borch-bundle/1"}), timeout=60) as r:
            index = json.load(r)
        rows = index if isinstance(index, list) else index.get("models") or index.get("entries") or []
        kept = []
        for row in rows:
            murl = row.get("manifestUrl", "")
            path = urllib.parse.urlparse(murl).path.lstrip("/")
            local = models_dir / path / "manifest.json" if not path.endswith("manifest.json") else models_dir / path
            if not local.exists():
                continue
            manifest = json.loads(local.read_text(encoding="utf-8"))
            wurl = manifest["weights"]["url"]
            wpath = urllib.parse.urlparse(wurl).path.lstrip("/")
            if (models_dir / wpath).exists():
                manifest["weights"]["url"] = "/".join([".."] * (len(local.relative_to(models_dir).parts) - 1) + [wpath])
                local.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            row = dict(row)
            # The file itself, not its folder: borch-hub fetches the address as given, and
            # a static server answers a folder with an HTML listing (measured).
            row["manifestUrl"] = "models/" + local.relative_to(models_dir).as_posix()
            kept.append(row)
        (models_dir / "index.json").write_text(json.dumps(kept, indent=2), encoding="utf-8")

    version = subprocess.run(["git", "describe", "--always", "--dirty"], cwd=ROOT, text=True, capture_output=True).stdout.strip()
    wheel = next(OUT.glob("pyborch-*.whl")).name
    (OUT / "VERSION").write_text(f"borch field trainer\nsite {version}\nwheel {wheel}\npyodide {runtime_base}\nbuilt {time.strftime('%Y-%m-%d %H:%M')}\n", encoding="utf-8")
    shutil.copy(ROOT / "LICENSE", OUT / "LICENSE")
    (OUT / "README.txt").write_text(
        "The borch field trainer, as one folder.\n\n"
        "Serve this folder with any static web server and open index.html — for example\n"
        "    python3 -m http.server -d . 8000\n"
        "then http://localhost:8000/ in Chrome or Edge. It must be served: a page opened as a\n"
        "file:// cannot start the worker the kernel runs in. Nothing here reaches the network.\n"
        "\nWhat is inside: the marimo workbench (assets/), Pyodide and marimo's packages\n"
        "(pyodide/), the pyborch wheel, and the pretrained models (models/). See VERSION.\n", encoding="utf-8")
    size = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file()) / 1e6
    digest = hashlib.sha256()
    for p in sorted(OUT.rglob("*")):
        if p.is_file():
            digest.update(p.relative_to(OUT).as_posix().encode()); digest.update(p.read_bytes())
    print(f"built {OUT} — {size:.0f} MB, {sum(1 for p in OUT.rglob('*') if p.is_file())} files, sha256 {digest.hexdigest()[:16]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
