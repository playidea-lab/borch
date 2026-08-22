"""Pins the core ↔ borch.ts name axis so it can fall and cannot rise.

`tests/ts_axis.py` is the measurement and its docstring says what it counts. This file
holds the numbers, and holds them the way `borch-ts/test/run.py` holds its gap table:
**each namespace's count must match exactly**, so carrying a name across lowers a
figure that somebody has to edit, and adding one raises a figure that goes red.

## What a green run of this file does not say

It says the counts below are what they were when written. It does **not** say:

- that the names present on both sides mean the same thing. A signature can lie and
  this counts names — five of those were found in one day and none was visible here.
- that a namespace with 0 core-only names is finished. borch.ts may carry names the
  core does not, and this file does not look in that direction yet.
- that the 408 without a reason are deliberate. They are the to-do list, and the
  reason each has none is that nobody has judged it yet.

That paragraph is here because of what happened the day this file was written. A
ceiling test asking *did Korean grow* was read as confirming *is this file English*,
and it was green about both. **A check is silent about every sentence it was not
asked**, and the cheapest place to say which sentences those are is the check itself.
"""

import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))

pytest.importorskip("torch")
pytest.importorskip("numpy")

DECL = ROOT / "borch-ts" / "dist" / "src"

# Core-only names per namespace, measured. **Each row is a to-do list, not a budget.**
# Lower it by carrying the name across; raising one needs a reason in this commit.
FROZEN = {
    # 107 until `maximum` and `minimum` were exposed. The kernels were already
    # there — `binary("maximum", …)` is used a dozen times inside `tensor.ts` — so
    # what was missing was the public method, which is the kind of gap a name count
    # finds and a value comparison never can.
    "Tensor": 105,
    "nn": 14,
    "nn.functional": 30,
    "optim": 0,
    "optim.lr_scheduler": 0,
    "linalg": 3,
    "utils.data": 12,
}

# The core carries these only in order to refuse them, so borch.ts not carrying the
# stub is a worse error message rather than a missing feature. Held apart from the
# gaps and pinned separately: a refusal turning into a gap, or a gap being quietly
# reclassified as a refusal, are both movements worth seeing.
REFUSALS = {
    "Tensor": 40,
    "nn": 0,
    "nn.functional": 0,
    "optim": 0,
    "optim.lr_scheduler": 0,
    "linalg": 0,
    "utils.data": 0,
}


def _stale():
    """The index is generated, and a stale one reports present names as absent.

    That direction matters: it lies **towards a gap**, so a stale run inflates the
    counts and the failure reads exactly like real work appearing. `test_site.py`
    refuses for the same reason and this borrows its rule rather than restating it.
    """
    index = ROOT / "site" / "assets" / "api-index.json"
    if not index.exists() or not DECL.exists():
        return "no generated index — run npm run build:ts && npm run docs:api"
    newest = max((p.stat().st_mtime for p in DECL.rglob("*.d.ts")), default=0)
    if newest > index.stat().st_mtime:
        return ("site/assets/api-index.json is older than the declaration files — "
                "run npm run docs:api")
    return None


def test_the_core_to_borch_ts_axis_has_not_widened():
    """Every namespace's core-only count, exactly.

    **This axis had no check at all until now.** `torch_gap.py` measures the core
    against real torch, `test_torch_signatures.py` measures borchvision against real
    torchvision, and `test_binding_arguments.py` measures the binding against
    borch.ts. The core and borch.ts are two implementations of one surface and the
    golden holds their values — nothing held their names, so a name in one and not
    the other was a tutorial line that runs here and raises there, with everything
    green.
    """
    stale = _stale()
    if stale:
        pytest.skip(stale)

    import ts_axis

    rows = ts_axis.compare()
    assert set(rows) == set(FROZEN), (
        f"the namespaces measured changed: {sorted(set(rows) ^ set(FROZEN))}\n"
        "  Add or remove the row in FROZEN in the same commit as the change.")

    moved = []
    for space, (gaps, refusals) in sorted(rows.items()):
        if len(gaps) != FROZEN[space]:
            moved.append(f"{space} gaps: {len(gaps)} now, {FROZEN[space]} written down")
        if len(refusals) != REFUSALS[space]:
            moved.append(f"{space} refusals: {len(refusals)} now, "
                         f"{REFUSALS[space]} written down")
    assert not moved, (
        "the core-only name counts moved:\n  " + "\n  ".join(moved)
        + "\n\n  A gap count lower means a name was carried across — edit FROZEN down.\n"
          "  Higher means the core gained a name borch.ts does not have, or a\n"
          "  borch.ts name was removed. Either wants saying out loud.\n"
          "  A refusal count moving means the core changed what it refuses, or the\n"
          "  factory names ts_axis.refused() looks for have drifted. The second is\n"
          "  the dangerous one: it reclassifies refusals as gaps, and it happened\n"
          "  once already — reading tables alone found 14 of the 40.\n"
          "  See it: uv run --with numpy --with torch --with torchvision \\\n"
          "            python tests/ts_axis.py --show Tensor")


def test_the_measurement_still_runs_as_a_script():
    """`ts_axis.py` is meant to be run by hand, and a script that stopped running is
    a measurement nobody can repeat. `test_gap.py` pins `torch_gap.py` the same way,
    for the reason that a check importing a module exercises less of it than running
    it does — the argument parsing and the printing are only reached this way."""
    if _stale():
        pytest.skip("generated index is stale")
    out = subprocess.run(
        [sys.executable, str(ROOT / "tests" / "ts_axis.py"), "--show", "linalg"],
        capture_output=True, text=True, cwd=ROOT)
    assert out.returncode == 0, out.stderr[-2000:]
    assert "linalg" in out.stdout
