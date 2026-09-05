"""Builds the CPU kernels and writes them into `borch-ts/src/cpu/kernels.ts` as base64.

    python3 borch-ts/wasm/build.py            # build and rewrite kernels.ts
    python3 borch-ts/wasm/build.py --check    # build and compare against the committed bytes

## Why the bytes live in a TypeScript file

`borch-ts` is one npm package with zero runtime dependencies and no build step on the
user's side, and the site serves `dist/` by relative path. A `.wasm` beside the emit
would need a URL to fetch and a second file to ship; base64 in a module needs neither,
and the module is a few kilobytes, so the price is a few kilobytes more.

## Why the file carries two hashes

`KERNELS_WASM_SHA256` is the hash of the bytes the base64 decodes to; the loader recomputes
it before instantiating, so a corrupted or edited blob refuses to load rather than running
different arithmetic under the same name. `KERNELS_SOURCE_SHA256` is the hash of the Rust
source, this script, and `Cargo.toml` together; `tests/test_cpu_kernels.py` recomputes it
from the tree, so a change to the source without a rebuild fails a check instead of
shipping stale kernels under a fresh comment. The same shape as `tests/golden.json`'s
manifest: the generated thing says what it was generated from.

## Two modules

The strict one is built as before. The `relaxed` one is the same source with the crate's
`relaxed` feature and `+relaxed-simd`: every multiply-add becomes `f32x4.relaxed_madd`, one
instruction whose rounding the hardware decides. Measured 2026-09-05 on an M4 Max: the
GEMM went from 55 to 92–100 GFLOPS on every ResNet-18 and EfficientNet-B0 shape. Both go
into `kernels.ts`; `load.ts` takes the relaxed one where `WebAssembly.validate` accepts it
and the caller has not said otherwise. The golden is measured with the strict one.

## Toolchain

Rust with the `wasm32-unknown-unknown` target (`rustup target add wasm32-unknown-unknown`)
and the `simd128` feature. Nothing else. The generated file records the compiler version
so a byte-for-byte rebuild can say why it differs when it does.
"""

import base64
import hashlib
import os
import pathlib
import re
import shutil
import subprocess
import sys
import textwrap

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent
OUT = ROOT / "borch-ts" / "src" / "cpu" / "kernels.ts"
TARGET = "wasm32-unknown-unknown"
WASM = HERE / "target" / TARGET / "release" / "borch_cpu_kernels.wasm"
# Files whose bytes decide the kernels. Order fixed so the hash is.
SOURCES = ("Cargo.toml", "src/lib.rs", "build.py")
# The base64 is 4.7KB today. 32KB is not a budget, it is an alarm: crossing it means
# something that is not a kernel got linked in (a formatter, an allocator, a panic
# message), and that is the thing this crate exists to keep out.
MAX_WASM_BYTES = 32 * 1024


def source_sha256(here=HERE):
    """One hash over the files that produce the module, path names included."""
    h = hashlib.sha256()
    for rel in SOURCES:
        h.update(rel.encode("utf-8"))
        h.update(b"\0")
        h.update((here / rel).read_bytes())
        h.update(b"\0")
    return h.hexdigest()


def _leb128(data, pos):
    result = shift = 0
    while True:
        byte = data[pos]
        pos += 1
        result |= (byte & 0x7F) << shift
        if byte & 0x80 == 0:
            return result, pos
        shift += 7


def _sections(data):
    if data[:4] != b"\0asm" or data[4:8] != b"\1\0\0\0":
        raise ValueError("not a wasm 1.0 module")
    pos = 8
    while pos < len(data):
        section_id = data[pos]
        size, pos = _leb128(data, pos + 1)
        yield section_id, data[pos:pos + size]
        pos += size


def wasm_exports(data):
    """`[(name, kind)]` from the export section — kind is func | table | memory | global."""
    kinds = {0: "func", 1: "table", 2: "memory", 3: "global"}
    out = []
    for section_id, body in _sections(data):
        if section_id != 7:
            continue
        count, pos = _leb128(body, 0)
        for _ in range(count):
            length, pos = _leb128(body, pos)
            name = body[pos:pos + length].decode("utf-8")
            pos += length
            kind = body[pos]
            _, pos = _leb128(body, pos + 1)
            out.append((name, kinds.get(kind, str(kind))))
    return out


def wasm_import_count(data):
    """How many imports the module needs. The kernels need none — that is the property."""
    for section_id, body in _sections(data):
        if section_id == 2:
            count, _ = _leb128(body, 0)
            return count
    return 0


def parse_kernels_ts(text):
    """The constants back out of the generated file. Shared with the guard test and `--check`."""
    def one(name):
        m = re.search(rf'export const {name} = "([^"]*)"', text)
        if not m:
            raise ValueError(f"{name} not found in kernels.ts")
        return m.group(1)
    def blob(name):
        return base64.b64decode("".join(re.findall(r'"([A-Za-z0-9+/=]+)"', text.split(f"{name} =", 1)[1].split(";", 1)[0])), validate=True)
    exports = re.findall(r'"(\w+)"', text.split("KERNELS_EXPORTS =", 1)[1].split(";", 1)[0])
    return {
        "source_sha256": one("KERNELS_SOURCE_SHA256"),
        "wasm_sha256": one("KERNELS_WASM_SHA256"),
        "relaxed_sha256": one("KERNELS_RELAXED_WASM_SHA256"),
        "rustc": one("KERNELS_RUSTC"),
        "exports": exports,
        "wasm": blob("KERNELS_WASM_BASE64"),
        "relaxed": blob("KERNELS_RELAXED_WASM_BASE64"),
    }


def _cargo(cargo, features, extra_args):
    env = dict(os.environ, RUSTFLAGS="-C target-feature=+simd128" + features)
    subprocess.run([cargo, "build", "--release", "--target", TARGET, "--quiet", *extra_args], cwd=HERE, env=env, check=True)
    data = WASM.read_bytes()
    if len(data) > MAX_WASM_BYTES:
        sys.exit(f"{WASM.name} is {len(data)} bytes, over {MAX_WASM_BYTES} — something that is not a kernel got linked in")
    if wasm_import_count(data):
        sys.exit("the module imports something — the kernels are meant to need nothing")
    return data


def build():
    """Both modules: `(strict, relaxed, rustc)`."""
    cargo = shutil.which("cargo")
    if not cargo:
        sys.exit("cargo is not installed — https://rustup.rs, then `rustup target add wasm32-unknown-unknown`")
    installed = subprocess.run(["rustup", "target", "list", "--installed"], capture_output=True, text=True).stdout
    if TARGET not in installed:
        sys.exit(f"the {TARGET} target is missing — `rustup target add {TARGET}`")
    strict = _cargo(cargo, "", [])
    relaxed = _cargo(cargo, ",+relaxed-simd", ["--features", "relaxed"])
    rustc = subprocess.run(["rustc", "--version"], capture_output=True, text=True, check=True).stdout.strip()
    if exports_of(strict) != exports_of(relaxed):
        sys.exit("the two modules export different names — they are meant to be one interface")
    return strict, relaxed, rustc


def exports_of(data):
    return sorted(name for name, kind in wasm_exports(data) if kind == "func" and not name.startswith("__"))


def _blob(data):
    return " +\n".join(f'  "{line}"' for line in textwrap.wrap(base64.b64encode(data).decode("ascii"), 96))


def render(strict, relaxed, rustc):
    exports = exports_of(strict)
    return f'''/**
 * The CPU kernels, as bytes. **Generated by `borch-ts/wasm/build.py` — do not edit.**
 *
 * The Rust source is `borch-ts/wasm/src/lib.rs`; the reasons for a wasm module with no
 * runtime are at the top of that file. This file exists so that the package ships one
 * module and the site fetches nothing extra: the bytes are here, base64.
 *
 * Two modules from one source. The strict one ({len(strict)} bytes) rounds every multiply and
 * every add, the same on every machine — the golden is measured with it. The relaxed one
 * ({len(relaxed)} bytes) is built with the crate's `relaxed` feature: each multiply-add is
 * `f32x4.relaxed_madd`, one instruction, its last bit the hardware's — 1.7× on the GEMM,
 * measured. `load.ts` takes it where the engine validates relaxed SIMD.
 *
 * Hashes. The `*_SHA256` of each blob is recomputed by the loader before instantiating, so
 * edited or corrupted bytes refuse to run. `KERNELS_SOURCE_SHA256` is recomputed by
 * `tests/test_cpu_kernels.py` from the Rust source, so a source change without a rebuild
 * fails a check. `npm run build:wasm` regenerates everything here.
 */
export const KERNELS_SOURCE_SHA256 = "{source_sha256()}";
export const KERNELS_RUSTC = "{rustc}";
/** Every function both modules export, sorted. The loader asks for exactly these. */
export const KERNELS_EXPORTS = [{", ".join(f'"{e}"' for e in exports)}] as const;
export const KERNELS_WASM_SHA256 = "{hashlib.sha256(strict).hexdigest()}";
export const KERNELS_WASM_BYTES = {len(strict)};
export const KERNELS_WASM_BASE64 =
{_blob(strict)};
export const KERNELS_RELAXED_WASM_SHA256 = "{hashlib.sha256(relaxed).hexdigest()}";
export const KERNELS_RELAXED_WASM_BYTES = {len(relaxed)};
export const KERNELS_RELAXED_WASM_BASE64 =
{_blob(relaxed)};
'''


def main(argv):
    strict, relaxed, rustc = build()
    text = render(strict, relaxed, rustc)
    if "--check" in argv:
        if not OUT.exists():
            sys.exit(f"{OUT.relative_to(ROOT)} does not exist — run without --check")
        committed = parse_kernels_ts(OUT.read_text(encoding="utf-8"))
        if committed["wasm"] != strict or committed["relaxed"] != relaxed:
            sys.exit(f"rebuilt bytes differ from {OUT.relative_to(ROOT)} (committed {committed['rustc']}, now {rustc}) — run npm run build:wasm")
        print(f"{OUT.relative_to(ROOT)}: strict {len(strict)} + relaxed {len(relaxed)} bytes, match the rebuild")
        return 0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text, encoding="utf-8")
    print(f"{OUT.relative_to(ROOT)}: strict {len(strict)} + relaxed {len(relaxed)} bytes, exports {', '.join(exports_of(strict))}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
