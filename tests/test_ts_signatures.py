"""Pins the core ↔ borch.ts **signature** axis, and pins that it is measuring at all.

`tests/ts_signatures.py` is the measurement and its docstring says what it compares.
This file holds the numbers the way `test_ts_axis.py` holds its own: each namespace
exactly, so carrying a signature across lowers a figure somebody has to edit.

## The second test here matters more than the first

While this axis was being built it reported, in three consecutive runs, numbers that
looked like the whole surface and were drawn from top-level functions alone. Every
method was skipped, because `api.json` gives members no `kind` field and the filter
asked for one. **A method that is never filed produces no row of any kind** — not a
mismatch, not even an `unreadable` — so the miss showed up as the counts being small
and tidy, which is what a healthy library looks like.

It became visible only when `nn` reported agree 0, differ 0, unreadable 0 at once: a
namespace of 144 layers with nothing to say. The same shape caught the 1,137-name
mapping error on the name axis. **Too clean to be a finding is a finding.**

So `test_the_measurement_still_reads_methods` pins a floor on how many pairs are
compared. A refactor that quietly stops reading members drops it to a handful and
this goes red, where the frozen counts below would merely have improved.

## What a green run does not say

- **Not that the arguments mean the same thing.** Names are compared; a `dim` that
  counts from the other end is invisible here.
- **Not that types or defaults agree.** Stated in the measurement's docstring too,
  and repeated because this is the file somebody reads when deciding whether an axis
  is covered.
- **Not that `shorter` is safe.** It means borch.ts takes a prefix of what torch
  takes, so nothing silently shifts — the missing tail is still a missing feature.
"""

import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))

pytest.importorskip("torch")
pytest.importorskip("numpy")

# Signatures that hold a **different argument at some position** — the bucket where a
# tutorial line returns a number instead of raising. Each is work; lower it by fixing
# borch.ts or by folding a convention into `ts_signatures.RENAMES` **with its reason**.
#
# **`linalg` was 14 differ against a namespace nobody could call**, and it is 6 now.
# The first run of this axis paired `torch.linalg.det(t)` against `det(f)` and
# `matmul(a, b)` against `matmul(a, b, n, k, m)` — flat, `Mat`-based CPU internals
# that `site/build_api.py` was publishing as though they were the torch namespace,
# because the file happened to be *named* `linalg`. That was a reachability question
# already open, and this axis is where it turned from an argument into a number.
#
# A peer session split the file (`_linalg.ts` for the numerics, `linalg.ts` for the
# namespace) and the row became readable in the same edit: 7 agree, 17 renamed, 2
# shorter, 6 differ. The 17 are one convention — the core calls its matrix `t` and
# borch.ts calls it `a` — and they sit in the harmless bucket rather than being
# folded, because `t` is too short a name to fold everywhere on the strength of one
# namespace.
DIFFER = {
    # 4 → 3. `std(dim, unbiased, keepdim)` against `std(correction)`: this pair took the
    # correction first, alone among the reductions, so `x.std(0)` — the line anybody
    # transcribing torch writes — compiled, ran and returned a scalar at correction 0
    # where torch returns one value per column. `dim` comes first now, as it already did
    # in `mean`, `sumDim` and `amax`. Nothing else in the repository was watching that
    # place: the golden asks `std()` three times and never with an argument.
    "Tensor": 3,
    "nn": 30,
    "nn.functional": 2,
    "optim": 7,
    "optim.lr_scheduler": 4,
    "linalg": 6,
    "utils.data": 0,
}

# borch.ts takes a **prefix** of torch's arguments. Not a shift and not silent — one
# too many raises. Held separately so that fixing a shift cannot be paid for by
# turning it into a truncation without anyone noticing.
SHORTER = {
    "Tensor": 162,
    "nn": 32,
    "nn.functional": 0,
    "optim": 0,
    "optim.lr_scheduler": 11,
    "linalg": 2,
    "utils.data": 2,
}


def _stale():
    """The emitted API is generated, and a stale one is measured against a surface
    that no longer exists. `test_site.py` and `test_ts_axis.py` refuse for the same
    reason; this borrows the rule rather than restating it."""
    api = ROOT / "site" / "assets" / "api.json"
    decl = ROOT / "borch-ts" / "dist" / "src"
    if not api.exists() or not decl.exists():
        return "no generated API — run npm run build:ts && npm run docs:api"
    newest = max((p.stat().st_mtime for p in decl.rglob("*.d.ts")), default=0)
    if newest > api.stat().st_mtime:
        return "site/assets/api.json is older than the declarations — run npm run docs:api"
    return None


def _rows():
    if _stale():
        pytest.skip(_stale())
    import ts_signatures
    return ts_signatures.compare()


def test_the_signature_axis_has_not_widened():
    """Every namespace's shifted and truncated counts, exactly."""
    rows = _rows()
    assert set(rows) == set(DIFFER) == set(SHORTER), (
        f"the namespaces measured changed: {sorted(set(rows) ^ set(DIFFER))}")

    moved = []
    for space, found in sorted(rows.items()):
        got = {"differ": 0, "shorter": 0}
        for _n, _m, _y, note in found:
            if note in ("differ", "reordered"):
                got["differ"] += 1
            elif note in ("shorter", "longer"):
                got["shorter"] += 1
        if got["differ"] != DIFFER[space]:
            moved.append(f"{space} differ: {got['differ']} now, {DIFFER[space]} written")
        if got["shorter"] != SHORTER[space]:
            moved.append(f"{space} shorter: {got['shorter']} now, {SHORTER[space]} written")
    assert not moved, (
        "the signature counts moved:\n  " + "\n  ".join(moved)
        + "\n\n  Lower means a signature was carried across — edit the table down.\n"
          "  Higher means borch.ts and the core drifted apart, or a fold in\n"
          "  ts_signatures.RENAMES stopped applying. Read the rows before editing:\n"
          "    uv run --with numpy --with torch --with torchvision \\\n"
          "      python tests/ts_signatures.py --show nn")


def test_the_measurement_still_reads_methods():
    """**A floor on how many pairs are compared at all.**

    This is the regression that actually happened, three runs running. `api.json`
    gives members no `kind` field, the filter asked for one, and every method fell
    out — while the printed summary went on looking like a measurement of the whole
    surface. Nothing was red. `nn` said agree 0 / differ 0 / unreadable 0.

    The floor is set well under what stands today (570 pairs) so ordinary work does
    not trip it, and well over what top-level functions alone give (30), which is the
    number the broken version produced.
    """
    rows = _rows()
    pairs = sum(len(found) for found in rows.values())
    assert pairs > 300, (
        f"only {pairs} signature pairs were compared, and there were 570.\n"
        "  The likely cause is the member walk in ts_signatures.ts_signatures():\n"
        "  members carry no `kind`, so a filter written on `kind` files none of them\n"
        "  and every method vanishes from the measurement without leaving a row.")
    assert len(rows["Tensor"]) > 200, (
        f"the Tensor namespace compared {len(rows['Tensor'])} pairs — the methods are "
        "the bulk of this library and they are not being read.")


def test_a_dropped_middle_argument_is_not_read_as_a_short_tail():
    """The distinction the whole file turns on, checked on made-up input.

    `ReduceLROnPlateau` is the real instance: torch takes
    `(optimizer, mode, factor, patience, ...)` and borch.ts takes
    `(opt, factor, patience, threshold)`. `mode` is gone from the middle, so
    `new ReduceLROnPlateau(opt, 'min', 0.1)` puts a string in `factor` and `0.1` in
    `patience`, and **nothing raises**. Read as a short tail it would look like a
    feature that was not carried across, which is the harmless kind.
    """
    import ts_signatures

    assert ts_signatures._verdict(["a", "b", "c"], ["a", "b"]) == "shorter"
    assert ts_signatures._verdict(["a", "b", "c"], ["a", "c"]) == "differ"
    assert ts_signatures._verdict(["a", "b"], ["a", "b"]) == "agree"
    assert ts_signatures._verdict(["a", "b"], ["b", "a"]) == "reordered"
    assert ts_signatures._verdict(["a", "b"], ["a", "z"]) == "renamed"
    # torch's `T_max` against borch.ts's `tMax` — the initial capital carries nothing
    # on a parameter, which is the reverse of the rule the *name* axis applies.
    assert ts_signatures._verdict(["TMax"], ["tMax"]) == "agree"


def test_a_nested_type_is_not_read_as_several_arguments():
    """Splitting a parameter list on every comma invents arguments that do not exist.

    The constructor of `Tensor` is the case that forced this: its options object
    carries `backwardFn?: (grad: Tensor) => readonly (Tensor | null)[]`, whose commas
    and brackets are all inside something. A naive split reports six parameters.
    """
    import ts_signatures

    sig = ("constructor(storage: GPUBuffer | Float32Array, shape: readonly number[], "
           "options?: { parents?: readonly Tensor[]; "
           "backwardFn?: (grad: Tensor) => readonly (Tensor | null)[]; })")
    names, bagged = ts_signatures.ts_params(sig)
    assert names == ["storage", "shape"], names
    assert bagged == 1, "the options bag should end the comparison, not be counted"

    got, _ = ts_signatures.ts_params("f(a: Record<string, number>, b: number)")
    assert got == ["a", "b"], got


def test_the_measurement_still_runs_as_a_script():
    """It is meant to be run by hand, and a script that stopped running is a
    measurement nobody can repeat. `test_ts_axis.py` pins its own the same way."""
    if _stale():
        pytest.skip("the generated API is stale")
    out = subprocess.run(
        [sys.executable, str(ROOT / "tests" / "ts_signatures.py"), "--show", "utils"],
        capture_output=True, text=True, cwd=ROOT)
    assert out.returncode == 0, out.stderr[-2000:]
    assert "utils.data" in out.stdout
