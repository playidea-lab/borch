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
    # **This number went 107 → 105 → 93 → 112, and only the last move was a
    # correction rather than work.**
    #
    # 107 → 105 exposed `maximum` and `minimum`, whose kernels were already there.
    # 105 → 93 added the six short comparison names (`eq`, `ne`, `lt`, `le`, `gt`,
    # `ge`) and should have moved it by six. Twelve was the tell: `_camel` stripped
    # the trailing underscore torch uses for in-place, so `eq_` matched the `eq`
    # just written and five more like it. Keeping the underscore put nineteen
    # in-place names back where they belong, and the count rose to 112.
    #
    # The count is higher than it started and the surface is better than it was.
    # A rule that misses in our favour is worse than no rule — the number improving
    # is what made this one findable.
    "Tensor": 112,
    # 14 until case-folding stopped erasing the class/function boundary. `Embedding`
    # was a layer the core had and borch.ts did not, `embedding` a function both had;
    # folding the two reported the layer as present because the function was. torch's
    # initial capital IS that boundary, and keeping it took the count to 16.
    #
    # 16 → 15: `Embedding` is now on both sides, so it is no longer a core-only name.
    # **The example above is kept even though the class it names is no longer a gap.**
    # It is the reason the rule exists, not a list of what the rule currently finds —
    # and a comment that describes today's data is a comment that goes stale the next
    # time the data moves, which this file has been bitten by twice today.
    # 15 → 11. The four were `BatchNorm1d` and `InstanceNorm1d/2d/3d`, and the
    # way they were found says more than the names do.
    #
    # **This axis had been printing them the whole time.** The `nn` row has read
    # `✘ gap 15` for as long as the row has existed, and the number was written
    # here and asserted equal — which turns a list of defects into a budget. A
    # peer hit `is not a constructor` in a browser and went looking; the axis
    # already knew, and `--show nn` would have named all four in one command.
    #
    # So the failure was not a missing instrument. It was **a red number that had
    # been red long enough to read as furniture**, plus a test that made staying
    # red the passing condition. The `Tensor: 112` comment above is the same
    # hazard held the other way: it explains why its number moved, so the number
    # stays legible. A pinned count with no reading beside it is an accepted loss.
    #
    # Two things had promised the missing names: `BatchNormND`'s comment said
    # "`BatchNorm1d`, `2d` and `3d` are all this" with two written below it, and
    # three golden cases named `nn.InstanceNorm1d/2d/3d` while constructing
    # `InstanceNormND`. All six lazy variants existed, so `LazyInstanceNorm2d`
    # stood for the lazy form of a class nobody could import.
    # 11 → 10. `BCELoss`, whose logits form (`BCEWithLogitsLoss`) was already here —
    # the same shape as `BatchNorm1d` above, and as `default_collate` below.
    "nn": 10,
    # **30 → 10, and eighteen of the twenty were one delegation each.**
    #
    # `poolND`, `lpPool`, `maxUnpool`, `convTransposeND`, `maxPoolWithIndices` and
    # `fractionalMaxPool` were all doing the work already, under generic ND names
    # borch.ts chose and torch does not have. So this counted eighteen absent
    # *features* while every one was a wrapper away.
    #
    # **Adding them moved this number by two, and that was the real finding.**
    # `functional.ts` was not in `site/build_api.py`'s `MODULES`, so the module that
    # exists *so a line written `F.conv2d(x, w, b)` can be copied across* was in
    # neither the API reference nor the name index — and this axis could not see a
    # single name living only there. The two that did move were the two that also
    # became `Tensor` methods.
    #
    # The comment at the top of that list describes exactly this failure, about
    # `index.ts`, and names `isTensor` as the one it caught. **It was written, and
    # the module next to it was never checked against it.** A reason that is true,
    # in place, and about the case adjacent to the one in front of it.
    #
    # The ten left are the `adaptive_*` family and
    # `triplet_margin_with_distance_loss`. The adaptive family is held deliberately:
    # a peer's lesson page teaches a workaround for `AdaptiveAvgPool2d` being absent
    # and pins the absence at both ends, so filling it silently would leave a page
    # teaching a detour around a road that exists. It moves when that page does.
    "nn.functional": 10,
    "optim": 0,
    "optim.lr_scheduler": 0,
    # 3 → 0. The three were `inv`, `pinv` and `matmul` — torch's spellings of
    # `inverse`, `pinverse` and `mm`, which had no name anywhere in borch.ts. The
    # `linalg` namespace carries them now.
    "linalg": 0,
    # **12 → 2, and the two that remain have reasons.** The ten were the samplers
    # and the two dataset shapes: `Sampler`, `SequentialSampler`, `RandomSampler`,
    # `SubsetRandomSampler`, `WeightedRandomSampler`, `BatchSampler`,
    # `IterableDataset`, `ChainDataset`, `StackDataset` — and `default_collate`.
    #
    # `data.ts`'s own header had argued against them, and the argument was good:
    # *putting a `sampler` option down with nothing behind it repeats what happened
    # with `paramGroups` — torch's shape, hollow inside, quietly ignoring what the
    # caller passes.* That is an argument against a **hollow** sampler. They are
    # written rather than named, and the loader's refusals (a `sampler` beside a
    # `shuffle`, a `batchSampler` beside a `batchSize`) are the part that makes the
    # option worth having: taking both means one is ignored, and a loader that
    # ignores an argument still hands back batches that look right.
    #
    # **`default_collate` was already there under another name.** The function was
    # `stackItems`, and its own comment read "the place `default_collate` occupies".
    # This axis counts names, so the feature read as absent while sitting three
    # lines above the loader that called it — a comment naming what something
    # *would* be called is not the name.
    "utils.data": 2,
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
