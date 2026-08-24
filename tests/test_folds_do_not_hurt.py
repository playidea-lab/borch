"""A declared fold must not turn a match into a mismatch.

`ts_signatures.RENAMES` rewrites the core's parameter name into borch.ts's — so
`kernel_size` becomes `kernel` and the two libraries can be compared at all. It is a
fold, and a fold can be wrong in two directions.

`test_scheduler_table` already watches one of them: **a fold that fires on nothing**
is indistinguishable from a fold doing work, so it fails and the line has to go. That
is the direction where a fold claims credit it has not earned.

Nothing watched the other. **A fold that fires and makes a row worse**: the six
`LazyConv` classes had already taken torch's `kernelSize` and `outChannels` on both
sides, and `kernel_size → kernel` rewrote the core's name into a word borch.ts does
not use there — turning six agreements into six `unaligned` rows. The fold was right
for the eighteen layers that do spell it `kernel` and wrong for these six, at the same
time, because **a fold is global and correctness is per row.**

This is the mirror of what `test_fold_is_lossless.py` holds for `_camel`. There a fold
made a mismatch read as a match; here one makes a match read as a mismatch. The first
is the failure everybody names. The second costs somebody a search for a defect that
is not there — three sessions today, twice on this very family.

## What a green run does not say

- **Not that the folds are right**, only that none of them lands on a name its own
  counterpart lacks while the unfolded name would have matched.
- **Not that a fold is doing anything.** `test_scheduler_table` holds that end.
"""

import inspect
import json
import pathlib
import re
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

pytest.importorskip("numpy")

import ts_axis          # noqa: E402
import ts_signatures    # noqa: E402

API_PATH = ROOT / "site" / "assets" / "api.json"

_PARAM = re.compile(r"[(,]\s*(\w+)\s*[?:]")


def _params(signature):
    return _PARAM.findall(signature or "")


def _ts_symbols():
    """`{name: {parameter names}}` from the same `api.json` the axis reads.

    A class is represented by its **constructor**, which is the list the axis
    compares against the core's `__init__`; anything else by its own call.
    """
    api = json.loads(API_PATH.read_text(encoding="utf-8"))
    out = {}
    for module in api["modules"]:
        for symbol in module.get("symbols", []):
            ctor = next((m for m in symbol.get("members", [])
                         if m["name"] == "constructor"), None)
            names = _params(ctor["signature"]) if ctor else _params(
                symbol.get("signature", ""))
            if names:
                out.setdefault(symbol["name"], set(names))
            for member in symbol.get("members", []):
                if member["name"] == "constructor":
                    continue
                got = _params(member.get("signature", ""))
                if got:
                    out.setdefault(member["name"], set(got))
    return out


def _spaces():
    import borch
    return {"Tensor": borch.Tensor, "nn": borch.nn,
            "nn.functional": borch.nn.functional, "linalg": borch.linalg,
            "optim": borch.optim, "optim.lr_scheduler": borch.optim.lr_scheduler,
            "utils.data": borch.utils.data}


def _rows():
    """`(fired, pairs)` — the folds that fire, and how many symbols were compared.

    **The two are counted apart on purpose.** The first version pinned a floor on
    firings alone, and it went red the moment three folds were *retired* — the fold
    doing its job by having the difference removed under it. A floor that cannot tell
    *the instrument stopped reading* from *a fold was retired* reports the second as
    the first, and the only way out is to lower it, which is what the `linalg` floor
    note two files over warns about.

    Pairs do not fall when a fold retires. They fall when the sweep stops finding the
    two libraries, which is the thing worth pinning.
    """
    ts = _ts_symbols()
    fired, pairs = [], 0
    for space, module in _spaces().items():
        for name in dir(module):
            if name.startswith("_") or name not in ts:
                continue
            try:
                core = [p for p in inspect.signature(getattr(module, name)).parameters
                        if p != "self"]
            except (TypeError, ValueError):
                continue
            pairs += 1
            for parameter in core:
                folded = ts_signatures.RENAMES.get(parameter)
                if folded is None:
                    continue
                fired.append((f"{space}.{name}", parameter, folded,
                              ts_axis._camel(parameter), ts[name]))
    return fired, pairs


@pytest.mark.skipif(not API_PATH.exists(),
                    reason="no api.json — run python3 site/build_api.py first")
def test_no_fold_turns_a_match_into_a_mismatch():
    hurt = []
    for label, parameter, folded, plain, theirs in _rows()[0]:
        if folded not in theirs and plain in theirs:
            hurt.append(
                f"{label}: {parameter!r} already matches borch.ts as {plain!r}, and "
                f"the fold rewrites it to {folded!r}, which that symbol does not have")
    assert not hurt, (
        f"{len(hurt)} fold(s) turn a match into a mismatch:\n  " + "\n  ".join(hurt[:12])
        + "\n\n  A fold is global and correctness is per row. `kernel_size -> kernel` "
          "is right\n  for the layers borch.ts spells `kernel` and wrong for the ones "
          "that took\n  torch's `kernelSize`, at the same time.\n"
          "\n  Two ways out, and they are not the same. **Removing the difference** — "
          "having\n  borch.ts spell it torch's way everywhere — retires the fold and "
          "is what was\n  done here. **Narrowing the fold** would need a per-symbol "
          "exception, and this\n  table has nowhere to write one.")


@pytest.mark.skipif(not API_PATH.exists(),
                    reason="no api.json — run python3 site/build_api.py first")
def test_the_sweep_is_reading_both_libraries():
    """**A sweep that compares nothing passes the assertion above.**

    If `api.json` moves, or the parameter regex stops matching, or a namespace is
    renamed, this file goes quiet rather than red — the shape three instruments in
    this repository have failed in.

    **The floor counts symbols compared, not folds fired**, and the difference was
    learned the hard way: pinned on firings, it went red an hour later because three
    folds had been *retired*, which is a fold succeeding. A number that reports
    success as failure gets lowered, and the next real stall is lowered past too.

    Zero folds firing is a legitimate end state — it means the two libraries spell
    everything alike — and this file has to keep working on the day it arrives.
    """
    _fired, pairs = _rows()
    assert pairs > 100, (
        f"only {pairs} symbols were found in both libraries — this file is no longer "
        "reading\n  them, and the check above would pass on an empty sweep.")
