"""**The committed CPU kernels are what nothing but the loader was reading.**

`borch-ts/src/cpu/kernels.ts` is generated: `borch-ts/wasm/build.py` compiles the Rust in
`borch-ts/wasm/src/lib.rs` to wasm SIMD and writes the bytes into that file as base64, with
two hashes beside them. Nothing else in the repository opens it, and a generated file that
nothing checks is the shape that goes stale — `tests/golden.json` sat exactly like that until
`test_committed_golden.py` (this file borrows its structure).

## The states this file exists for

**The source moved and the bytes did not.** Someone edits a kernel in `lib.rs`, runs the
TypeScript checks, sees green — the checks run the committed bytes, not the edited source —
and commits. `KERNELS_SOURCE_SHA256` is recomputed here from the tree, so that commit is red.

**The bytes moved and the hash did not.** A merge keeps one side's constant and the other
side's blob. The base64 is decoded and hashed here, and the loader does the same before
instantiating, so it is red twice.

**The module grew a runtime.** The whole point of the crate is that it needs nothing: no
imports, a few kilobytes, only the functions the loader asks for. Each of those is a line
below, because each is a property somebody could lose without noticing while adding a
kernel — an allocator pulled in by a `Vec`, a panic message by an `assert!`.

Nothing here runs the kernels. Values are `borch-ts/test/cpu.ts`'s job, against the WebGPU
device, on a machine with one.
"""

import hashlib
import importlib.util
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
CRATE = ROOT / "borch-ts" / "wasm"
KERNELS_TS = ROOT / "borch-ts" / "src" / "cpu" / "kernels.ts"
LOAD_TS = ROOT / "borch-ts" / "src" / "cpu" / "load.ts"


def _build_module():
    spec = importlib.util.spec_from_file_location("borch_cpu_build", CRATE / "build.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


build = _build_module()


@pytest.fixture(scope="module")
def committed():
    if not KERNELS_TS.exists():
        pytest.skip(f"{KERNELS_TS.relative_to(ROOT)} is missing — npm run build:wasm")
    return build.parse_kernels_ts(KERNELS_TS.read_text(encoding="utf-8"))


def test_kernels_ts_carries_the_hash_of_the_rust_source_it_was_built_from(committed):
    """`build.SOURCES` names the files whose bytes decide the module; the hash over them is
    written into the generated file and recomputed here. Move one byte of `lib.rs` and this is
    red until `npm run build:wasm` — which is the point."""
    now = build.source_sha256(CRATE)
    assert committed["source_sha256"] == now, (
        "borch-ts/wasm changed after kernels.ts was generated — run npm run build:wasm\n"
        f"  kernels.ts says {committed['source_sha256'][:16]}…, the tree hashes to {now[:16]}…")


def test_kernels_ts_base64_decodes_to_bytes_with_the_recorded_hash(committed):
    """The loader does this too, at runtime, before instantiating. Here it runs without a
    browser, so the merge that keeps a constant and loses a line of blob is caught in `pytest`.
    Two blobs since the relaxed module arrived; each carries its own hash."""
    assert hashlib.sha256(committed["wasm"]).hexdigest() == committed["wasm_sha256"], "the strict base64 does not hash to KERNELS_WASM_SHA256"
    assert hashlib.sha256(committed["relaxed"]).hexdigest() == committed["relaxed_sha256"], "the relaxed base64 does not hash to KERNELS_RELAXED_WASM_SHA256"
    assert committed["wasm"] != committed["relaxed"], "the two modules are the same bytes — the relaxed build did not take"


def test_kernels_wasm_is_a_module_that_imports_nothing(committed):
    """No imports is the property that makes it a set of kernels rather than a runtime: nothing
    to link, nothing to version against Pyodide's heap. `Vec`, `format!` or `assert!` in `lib.rs`
    would each bring an import or a panic path — this is where that shows. Both modules."""
    assert build.wasm_import_count(committed["wasm"]) == 0
    assert build.wasm_import_count(committed["relaxed"]) == 0


def test_kernels_wasm_stays_a_few_kilobytes(committed):
    """4.7KB on 2026-09-05 for eight kernels. `MAX_WASM_BYTES` is an alarm rather than a budget;
    `build.py` refuses over it and this repeats the refusal for the committed bytes."""
    assert len(committed["wasm"]) <= build.MAX_WASM_BYTES
    assert len(committed["relaxed"]) <= build.MAX_WASM_BYTES


def test_kernels_exports_are_exactly_what_the_module_has_and_the_loader_asks_for(committed):
    """Three lists that have to agree: the module's export section, `KERNELS_EXPORTS` in the
    generated file, and the names `load.ts` fetches with `fn(ex, "…")`. Two of them are written by
    the generator; the third is by hand, and a kernel added in Rust and not in `load.ts` (or the
    reverse) is what this catches."""
    in_module = sorted(n for n, k in build.wasm_exports(committed["wasm"]) if k == "func" and not n.startswith("__"))
    assert committed["exports"] == in_module, "KERNELS_EXPORTS disagrees with the module's export section"
    in_relaxed = sorted(n for n, k in build.wasm_exports(committed["relaxed"]) if k == "func" and not n.startswith("__"))
    assert in_relaxed == in_module, "the relaxed module exports different names from the strict one"
    asked = sorted(set(re.findall(r'fn\(ex, "(\w+)"\)', LOAD_TS.read_text(encoding="utf-8"))))
    assert asked == in_module, (
        f"load.ts asks for {asked}\n  the module exports {in_module}")
    kinds = dict(build.wasm_exports(committed["wasm"]))
    assert kinds.get("memory") == "memory", "the module has to export its memory — the tensor side writes into it"
