"""Fetches what comes from outside once and checks the bytes have not moved.

A CDN is **a dependency that has to be alive at test time.** It really did go down once and
stopped the verification (`ERR_QUIC_PROTOCOL_ERROR`). And that `@4.22.0` will keep giving the
same bytes is a policy, not a contract — if that version disappears or changes, today's
golden answers cannot be reproduced.

**What was fetched now lives in the repository** (`vendor/pyodide/`). For a long time it was
"keep the hashes only", on the grounds of size, and measuring showed six files at 8.4MB
packed — this repository spends 23.9MB of history on `tests/golden.json` alone. Next to that,
8.4MB put in once and never changed was not a large number. What this decision costs is the
same amount attached permanently at every version bump, and it was worth paying because there
is no plan to move off 0.27.2.

What changed is **where they are kept** only; the lock file is used as before. If anything it
does its job only now — with the files and the lock **both committed**, the comparison runs in
CI with no network. Before, a fresh runner had no files, so `fetch` wrote a new lock from what
it downloaded, and the comparison after that compared with itself.

    uv run python tests/browser/vendor.py check   # compares what is here against the lock
    uv run python tests/browser/vendor.py fetch   # only when bumping. Stops if the lock differs
    uv run python tests/browser/vendor.py fetch --bump   # writes a new lock

The licences of what is fetched here are in [THIRD-PARTY.md](../../THIRD-PARTY.md).
**Pyodide is MPL-2.0** — it does not spread to our code, but having committed it **this
repository is a redistributor too.** So THIRD-PARTY.md records where to get the source.
"""

import collections
import hashlib
import json
import pathlib
import sys
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
VENDOR = ROOT / "vendor"
LOCK = pathlib.Path(__file__).resolve().parent / "assets.lock"

PYODIDE_VERSION = "0.27.2"
PYODIDE_BASE = f"https://cdn.jsdelivr.net/pyodide/v{PYODIDE_VERSION}/full/"
# What Pyodide actually requests as it comes up. A missing one shows as a 404 in the console.
PYODIDE_FILES = ["pyodide.js", "pyodide.asm.js", "pyodide.asm.wasm",
                 "python_stdlib.zip", "pyodide-lock.json"]
PYODIDE_PACKAGES = ["numpy", "pillow"]  # the file names are looked up in the lock; pillow decodes the images ImageFiles is handed

# **TF.js, for the comparison bench and nothing else.** `borch-ts/test/compare.ts` trains
# the same ResNet-18 step in TF.js beside borch.ts, on the same page, and the only
# honest way to hold a competitor still is to pin its bytes the way Pyodide's are pinned.
# The UMD builds are used (a global `tf`), because that is what the CDN serves and what
# a reader of the number can fetch themselves.
# **ONNX Runtime Web, for the inference half of the comparison.** The WebGPU build is a
# UMD script (global `ort`) plus one wasm and its loader; `ort.env.wasm.wasmPaths` points
# the page at `vendor/`.
ORT_VERSION = "1.29.0"
ORT = {
    "ort.webgpu.min.js":
        f"https://cdn.jsdelivr.net/npm/onnxruntime-web@{ORT_VERSION}/dist/ort.webgpu.min.js",
    # 1.29's WebGPU execution provider loads the **asyncify** build, measured — the page
    # asked for these two and refused with "no available backend" when only the `jsep`
    # pair (which the file list suggested) was here.
    "ort-wasm-simd-threaded.asyncify.mjs":
        f"https://cdn.jsdelivr.net/npm/onnxruntime-web@{ORT_VERSION}/dist/ort-wasm-simd-threaded.asyncify.mjs",
    "ort-wasm-simd-threaded.asyncify.wasm":
        f"https://cdn.jsdelivr.net/npm/onnxruntime-web@{ORT_VERSION}/dist/ort-wasm-simd-threaded.asyncify.wasm",
}
TFJS_VERSION = "4.22.0"
TFJS = {
    "tf.min.js":
        f"https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@{TFJS_VERSION}/dist/tf.min.js",
    "tf-backend-webgpu.min.js":
        f"https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-webgpu@{TFJS_VERSION}"
        "/dist/tf-backend-webgpu.min.js",
}

# **This is where TF.js used to be fetched.** Two `@tensorflow/tfjs@4.22.0` files came from
# the CDN into `vendor/`. Replacing the implementation that stood on it with hand-written WGSL
# made them unnecessary — the only thing fetched from outside now is Pyodide.
#
# This file stays even with one vendor left. A CDN is a dependency that has to be alive at test
# time, and it really did go down once.


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _get(url):
    with urllib.request.urlopen(url, timeout=180) as response:
        return response.read()


def _targets():
    """A list of (path, URL). numpy's name can only be known by reading the lock."""
    return ([(pathlib.Path(name), url) for name, url in {**TFJS, **ORT}.items()]
            + [(pathlib.Path("pyodide") / name, PYODIDE_BASE + name)
               for name in PYODIDE_FILES])


def _package_targets(lock_bytes):
    packages = json.loads(lock_bytes)["packages"]
    found = []
    for want in PYODIDE_PACKAGES:
        entry = packages.get(want)
        if entry is None:
            raise SystemExit(f"{want} is not in pyodide-lock.json")
        name = entry["file_name"]
        found.append((pathlib.Path("pyodide") / name, PYODIDE_BASE + name))
    return found


def fetch(bump=False):
    """Fetches into `vendor/`. **With a lock already present it compares, and stops if they differ.**

    For a long time this **overwrote** the lock. That makes whatever was downloaded the right
    answer, so the lock guarded "a machine that already has the files" and guarded nothing at
    all on a machine that has none. A fresh CI runner is exactly the latter, so six committed
    hashes quietly became a new lock there — the shape where a check chooses its own input.

    A bump is stated with `--bump`. A change to the lock is a thing that has to be visible in a
    commit, so it must not slip in on the back of a fetch.
    """
    VENDOR.mkdir(exist_ok=True)
    lock, total = {}, 0
    targets = _targets()
    lock_bytes = None
    for rel, url in targets:
        data = _get(url)
        if rel.name == "pyodide-lock.json":
            lock_bytes = data
        dst = VENDOR / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(data)
        lock[str(rel)] = _sha(data)
        total += len(data)
        print(f"  {rel}  {len(data) / 1e6:.2f} MB")

    for rel, url in _package_targets(lock_bytes):
        data = _get(url)
        (VENDOR / rel).write_bytes(data)
        lock[str(rel)] = _sha(data)
        total += len(data)
        print(f"  {rel}  {len(data) / 1e6:.2f} MB")

    text = "".join(f"{h}  {p}\n" for p, h in sorted(lock.items()))
    old = _read_lock()
    if old is not None and not bump:
        moved = sorted(p for p, h in lock.items() if old.get(p) != h)
        gone = sorted(p for p in old if p not in lock)
        if moved or gone:
            raise SystemExit(
                "what was fetched differs from the lock — nothing was overwritten.\n  "
                + "\n  ".join([f"{p}: the bytes differ" for p in moved]
                               + [f"{p}: no longer fetched" for p in gone])
                + "\n\nIf this is a bump, write a new lock with `fetch --bump`.")
    LOCK.write_text(text, encoding="utf-8")
    print(f"\nfetched — {len(lock)} files · {total / 1e6:.1f} MB → {VENDOR}")
    print(f"lock file: {LOCK}" + (" (rewritten)" if bump or old is None else " (unchanged)"))
    return 0


def _read_lock():
    if not LOCK.exists():
        return None
    entries = {}
    for line in LOCK.read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, path = line.split("  ", 1)
            entries[path] = digest
    return entries


# **What kind of problem it is, said as a field rather than in the sentence.**
# `ensure()` treats drift and absence completely differently — one stops, the other
# fetches — and it used to tell them apart by searching this text for "the bytes differ".
# A reworded sentence would have turned drift into a silent re-fetch, which overwrites
# exactly the files whose difference should have stopped everything.
Problem = collections.namedtuple("Problem", "kind path text")


def check(quiet=False):
    """Compares against the lock. (a list of `Problem`)"""
    entries = _read_lock()
    if entries is None:
        return [Problem("no-lock", "",
                        "there is no lock file — run `vendor.py fetch` first")]
    bad = []
    for path, digest in entries.items():
        f = VENDOR / path
        if not f.exists():
            bad.append(Problem("missing", path, f"{path}: missing"))
        elif _sha(f.read_bytes()) != digest:
            bad.append(Problem("drift", path, f"{path}: **the bytes differ** — what is "
                                              "here is out of step with the lock"))
    if not bad and not quiet:
        print(f"vendor comparison — all {len(entries)} agree")
    return bad


def ensure():
    """Fetches if absent, compares if present. Called as a runner starts."""
    if _read_lock() is None or check(quiet=True):
        problems = check(quiet=True)
        if problems and _read_lock() is not None:
            drifted = [p for p in problems if p.kind == "drift"]
            if drifted:
                raise SystemExit("vendor files differ from the lock:\n  "
                                 + "\n  ".join(p.text for p in drifted))
        print("fetching the vendor files (once)…")
        fetch()


def main(argv):
    what = argv[1] if len(argv) > 1 else "check"
    if what == "fetch":
        return fetch(bump="--bump" in argv)
    if what == "check":
        bad = check()
        for why in bad:
            print(f"  ✗ {why.text}")
        return 1 if bad else 0
    print("usage: vendor.py [check | fetch [--bump]]")
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
