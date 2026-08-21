"""Exports the golden answers into a **language-neutral format.**

    uv run --with numpy python tests/export_json.py

## Why

The golden answers today are `tests/golden.npz` (numpy) and `tests/cases.py` (Python
lambdas). That is enough for comparing two Python libraries, but an implementation that is
not Python **has no way to reach the expected values.** The 798 frozen with real torch are
this repository's most expensive asset, and keeping them inside Python alone leaves the next
implementation growing unverified.

## What crosses over and what does not — not mixed

**Crosses over**: the case names, the expected values (shape and dtype included), the input
arrays the cases use, and the case list's hash. That is, all of **"what the answer is".**

**Does not cross over**: the case bodies. `lambda L: L.amax(L.tensor(tie))` is Python code and
does not mechanically become another language. The other side **has to write cases of the same
names in its own language**, and what this file gives is the answer to hold them against.

That division is this file's point. The expensive half (numbers obtained by running real
torch) crosses over; the cheap half (one line of call) can be rewritten. Writing it the other
way round would be wrong — the moment it says "the golden answers were ported whole", the next
person believes cases were written that were not.
"""

import hashlib
import importlib.util
import json
import pathlib
import sys

import numpy as np

_here = pathlib.Path(__file__).resolve().parent
DEFAULT_OUT = _here / "golden.json"

_spec = importlib.util.spec_from_file_location("bt_cases", _here / "cases.py")
cases_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cases_mod)


def _value(arr):
    """One numpy array, in a shape JSON can hold.

    The values go in **flat** with the shape given separately — as nested lists a rank-8 array
    drowns in brackets, and the reading side takes a flat one more easily anyway.
    """
    if arr.dtype.kind == "U":
        return {"kind": "string", "value": str(arr)}
    flat = np.asarray(arr).reshape(-1)
    if arr.dtype.kind == "b":
        return {"kind": "bool", "shape": list(arr.shape),
                "values": [bool(v) for v in flat]}
    if arr.dtype.kind in "iu":
        return {"kind": "int", "shape": list(arr.shape),
                "values": [int(v) for v in flat]}
    # Floats are written **as float64.** Rounded to float32, a reading side more accurate than
    # us leaves the difference unattributable between our rounding and their implementation.
    # **A non-finite value carries its kind too.** Writing `nan` or `Infinity` into JSON gets
    # refused by a strict parser, so the positions are kept separately — and for a long time
    # only the position numbers were kept and not the kind, so the reading side restored them
    # all as `nan` and `inf` became the same thing as `nan`. Nothing caught it while no answer
    # held an infinity; the `fmax` case was the first to put one in an answer and it surfaced
    # (a max diff of 0 that came out as a failure).
    def kind_of(v):
        if np.isnan(v):
            return "nan"
        return "inf" if v > 0 else "-inf"

    return {"kind": "float", "shape": list(arr.shape),
            "values": [None if not np.isfinite(v) else float(v) for v in flat],
            "nonfinite": [[i, kind_of(v)] for i, v in enumerate(flat)
                          if not np.isfinite(v)]}


def export(npz_path=None, out_path=DEFAULT_OUT):
    npz_path = npz_path or (_here / "golden.npz")
    if not npz_path.exists():
        raise SystemExit(
            f"no golden answers: {npz_path}\n"
            "  first: uv run --with numpy --with torch python tests/golden.py dump")
    z = np.load(npz_path, allow_pickle=False)

    inp = cases_mod.golden_inputs()
    names = [name for name, _ in cases_mod.golden_cases(inp)]
    if str(z["__manifest__"]) != cases_mod.manifest_hash(cases_mod.golden_cases(inp)):
        raise SystemExit("the golden answers are stale — the case table changed. Run dump again.")

    doc = {
        "note": ("Expected values frozen with real PyTorch. The case **bodies** are not here — "
                 "the other side writes cases of the same names in its own language and holds "
                 "them against these answers."),
        "tolerance": {"atol": 1e-4, "rtol": 1e-4,
                      "note": "bit equality is this project's explicit non-goal"},
        "manifest": str(z["__manifest__"]),
        "inputs_fingerprint": str(z["__inputs__"]),
        # The inputs the cases share. The names are `golden_inputs()`'s keys.
        "inputs": {k: _value(v) for k, v in inp.items()},
        "cases": {},
    }
    missing = []
    for name in names:
        key = "case::" + name
        if key not in z.files:
            missing.append(name)
            continue
        doc["cases"][name] = _value(z[key])
    if missing:
        raise SystemExit("some cases are not in the golden answers:\n  " + "\n  ".join(missing))

    text = json.dumps(doc, ensure_ascii=False, separators=(",", ":"))
    out_path.write_text(text, encoding="utf-8")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return len(doc["cases"]), out_path, len(text), digest


def main(argv):
    count, path, size, digest = export()
    numeric = sum(1 for c in json.loads(path.read_text(encoding="utf-8"))["cases"].values()
                  if c["kind"] != "string")
    print(f"exported — {count} cases ({numeric} numeric, {count - numeric} string)")
    print(f"  {path}  {size / 1024:.0f}KB  sha256:{digest}")
    print("  the case **bodies** are not in it — only the answers are.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
