"""A folder of thousands of images through the tab — the field trainer's data path.

    uv run --with playwright python tests/browser/folder5k.py --dir=/path/to/folder [--headed] [--batch=32]
    uv run --with playwright python tests/browser/folder5k.py --make=5000 [--side=224] [--headed]

`--make=N` writes N synthetic images (ten classes, one low-frequency template each plus
noise — the set the other benches use) under the system temp directory once and reuses
them: JPEG through Pillow where it is installed, PNG through the probe's own encoder where
it is not (the 4090 has no Pillow). Measured on Apple: 5,000 JPEGs of 224 px in 17.3 s,
1,000 camera-sized JPEGs (1920×1440, 1.5 MB each) in 5.4 s — decode 2.2 ms/image.

Feeds the folder to `folder5k.html` the way a person would (the directory input, with
the file list as the fallback), then reads the five numbers the page measured: decode,
backbone features, IndexedDB write, head on the cache with a held-out split, neighbours
over everything — and the second visit, where the features come back from the cache.
Refuses a software rasteriser and a blanked screen, like every timing here.
"""
import os
import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from first_run import FLAGS, ROOT, refuse_if_screen_off, serve  # noqa: E402
from launch import _headed, refuse_if_software  # noqa: E402

GIVE_UP_MS = 1_800_000


def make_folder(n, side):
    """N images in ten class folders under the temp directory, written once."""
    import numpy as np
    root = (pathlib.Path(tempfile.gettempdir()) / f"borch-folder-{n}x{side}").resolve()
    have = sum(1 for p in root.rglob("*") if p.suffix in (".jpg", ".png")) if root.exists() else 0
    if have == n:
        return str(root)
    try:
        from PIL import Image
        write = lambda arr, path: Image.fromarray(arr).save(path.with_suffix(".jpg"), quality=85)   # noqa: E731
    except ImportError:
        from marimo_probe import png_bytes
        write = lambda arr, path: path.with_suffix(".png").write_bytes(png_bytes(arr))               # noqa: E731
    rng = np.random.default_rng(5)
    cells = 7
    classes = [f"c{k:02d}" for k in range(10)]
    templates = [rng.standard_normal((cells, cells, 3)).astype(np.float32) for _ in classes]
    idx = np.arange(side) * cells // side
    per = n // len(classes)
    for k, name in enumerate(classes):
        d = root / name
        d.mkdir(parents=True, exist_ok=True)
        for i in range(per):
            img = 0.5 + 0.3 * templates[k][idx][:, idx] + 0.15 * rng.standard_normal((side, side, 3)).astype(np.float32)
            write((np.clip(img, 0, 1) * 255).astype("uint8"), d / f"{name}_{i:04d}")
    print(f"wrote {per * len(classes)} images under {root}")
    return str(root)


def main(argv):
    from playwright.sync_api import sync_playwright
    headed = _headed("--headed" in argv)   # a window unless --headless / BORCH_HEADLESS: headless is SwiftShader here
    folder = next((a.split("=", 1)[1] for a in argv if a.startswith("--dir=")), None)
    batch = next((a.split("=", 1)[1] for a in argv if a.startswith("--batch=")), "32")
    make = next((a.split("=", 1)[1] for a in argv if a.startswith("--make=")), None)
    side = int(next((a.split("=", 1)[1] for a in argv if a.startswith("--side=")), "224"))
    if make:
        folder = make_folder(int(make), side)
    if not folder or not pathlib.Path(folder).is_dir():
        print("--dir=<folder of images> or --make=<count> is required", file=sys.stderr)
        return 2
    files = sorted(str(p) for p in pathlib.Path(folder).rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if refuse_if_screen_off("the folder loader"):
        return 1
    port, shutdown = serve(ROOT)
    url = f"http://127.0.0.1:{port}/tests/browser/folder5k.html?batch={batch}"
    profile = tempfile.mkdtemp(prefix="borch-folder5k-")
    channel = os.environ.get("BORCH_CHROME_CHANNEL") or None
    try:
        with sync_playwright() as pw:
            context = pw.chromium.launch_persistent_context(profile, headless=not headed, channel=channel, args=list(FLAGS), timeout=60_000)
            try:
                page = context.new_page()
                page.on("pageerror", lambda e: print(f"  [page] {e}"))
                page.goto(url, wait_until="load")
                page.wait_for_function("window.__folder5kReady === true", timeout=60_000, polling=100)
                # The directory input: the folder as a person would pick it (labels from folder
                # names). When the browser hands the page nothing — measured with a folder
                # under a symlinked temp path — the file list goes through the plain input.
                how = "the directory input"
                try:
                    page.set_input_files("#pick", str(pathlib.Path(folder).resolve()))
                    arrived = page.evaluate("document.getElementById('pick').files.length")
                except Exception as e:                                  # noqa: BLE001
                    arrived, how = 0, f"({type(e).__name__} on the directory input)"
                if not arrived:
                    page.set_input_files("#pickfiles", files)
                    how = f"the file list {how}".strip()
                print(f"fed {len(files)} files through {how}")
                page.wait_for_function("window.__folder5k !== undefined", timeout=GIVE_UP_MS, polling=1000)
                got = page.evaluate("window.__folder5k")
            finally:
                context.close()
    finally:
        shutdown()
    print(got.get("text", ""))
    if got.get("error"):
        print("error: " + got["error"][:800])
        return 1
    if refuse_if_software(got.get("adapter"), "the folder loader"):
        return 1
    ok = got["faults"] == 0 and not got["lost"] and got["acc"] >= 0.9
    print(f"**{got['n']} images went through the tab: {got['total']:.0f} s, held-out {100 * got['acc']:.1f}%, peak gpu {got['peakGpuMB']:.0f} MB, heap {got['peakHeapMB']:.0f} MB**" if ok else "**it did not** — see above")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
