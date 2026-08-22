"""Pins the core ↔ **real torch** argument axis.

`tests/torch_signatures_core.py` is the measurement and its docstring says what it
compares and what it cannot see. This file holds the numbers, the way
`test_ts_signatures.py` and `test_ts_axis.py` hold theirs.

## Why this axis had to exist separately from the other two

Every other argument check in this repository has both feet inside the project. When
the core and borch.ts disagree, a row says *they disagree* and never *which one to
move* — and when they agree, the whole apparatus can be converged on one error and
report agreement. `SmoothL1Loss` was the first: two instruments reported the split
and neither could say who was wrong. torch settled it, and borch.ts was right.

## What a green run does not say

- **Not that most of the surface was compared.** 571 rows are `torch is C`, which
  `inspect` cannot read — 496 of them in `Tensor`. This axis sees `nn`, `optim` and
  the schedulers well and sees very little else. The measurement's docstring records
  the two other sources that were tried and why neither is used.
- **Not that a `shorter` row is harmless here.** On the core↔borch.ts axis a prefix
  is safe because passing one argument too many raises. Against torch it means a
  *feature* torch has and the core does not, which is `torch_gap.py`'s business.
- **Not that the values agree.** The golden holds that, and only for the cases
  somebody wrote.
"""

import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))

pytest.importorskip("torch")
pytest.importorskip("numpy")

# **An argument at a different position from torch's.** The bucket where a line
# transcribed out of torch's documentation compiles, runs, and means something else.
#
# All thirteen `nn` rows are one shape: the core keeps the arguments it implements,
# in torch's relative order, and drops the ones it does not from the middle. So
#
#     MaxPool2d(2, 2, 1)              torch padding=1        core return_indices=1
#     Conv2d(3, 16, 3, 1, 1, False)   torch dilation=False   core bias=False
#     MultiheadAttention(64, 8, 0.1)  torch dropout=0.1      core bias=0.1
#     EmbeddingBag(10, 3, "sum")      torch max_norm="sum"   core mode="sum"
#
# Consistent with itself, and each one silently different from torch. **Not fixed
# here**: closing them means either implementing the missing arguments or standing
# refusal stubs in their places, and which of those is right is a decision about what
# the browser subset promises rather than a patch. Recorded so the decision is
# findable — that is this table's job.
SHIFTED = {
    # 13 → 11. MultiheadAttention grew torch's five middle arguments -- dropout,
    # add_bias_kv, add_zero_attn, kdim, vdim -- so MultiheadAttention(64, 8, 0.1),
    # torch's own way of writing a dropout, no longer sets bias=0.1. Four of the five
    # are refused rather than implemented, and the refusal already existed one layer
    # down in multi_head_attention_forward; carrying the argument means the refusal
    # arrives with the right name instead of the value landing elsewhere.
    #
    # RNNBase took mode first, as torch does, and RNN/LSTM/GRU pass their own. The
    # string is not decoration: it is what tells the base which recurrence to build.
    # 11 → 10. The four weight-first losses took torch's lists, which moved
    # `BCEWithLogitsLoss` out of this bucket. Two rows also left `unaligned` (10 → 8)
    # and `nn.functional` lost three (30 → 27): **one edit, seen from three
    # buckets**, which is what a real fix looks like against an outside authority.
    "Tensor": 2,
    # 11 → 8. MaxPool1d/2d/3d grew torch's padding, dilation and ceil_mode in
    # torch's positions, implemented rather than refused: padding pads with -inf so
    # a padded cell never wins, the indices stay flat indices into the *unpadded*
    # plane, and ceil_mode drops the window that would start inside the right
    # padding. Checked against real torch over 80 configurations across 1d, 2d and
    # 3d -- values, shapes, indices and gradients.
    #
    # `MaxPool2d(2, 2, 1)` used to set return_indices=1. It sets padding=1 now, as
    # torch reads it.
    #
    # 8 → 5. Conv1d/2d/3d grew torch's dilation, groups and padding_mode, all
    # implemented: dilation widens the im2col window and thins it, groups is done by
    # slicing the channels and joining the pieces so the gradient follows from `cat`
    # rather than from a second hand-written formula, and the non-zero padding modes
    # pad in the layer exactly as torch's layer does. Checked against real torch over
    # 144 functional configurations and 352 through the layers.
    #
    # **`bias` moved from the sixth position to the eighth**, where torch has it, so
    # `Conv2d(3, 16, 3, 1, 1, False)` sets dilation now. Every call site in this
    # repository already passed it by keyword, which is the only reason the move was
    # quiet — a positional call is a silent bet that the callee's order never moves.
    #
    # 5 → 2. ConvTranspose1d/2d/3d grew output_padding, groups and dilation, all
    # implemented, checked against real torch over 346 configurations.
    #
    # **torch puts `dilation` after `bias` here and before it in Conv2d.** The two are
    # not one list in a different spelling, and following torch means following that
    # too — a tidier order of our own would read as agreement and land a positional
    # call somewhere else.
    #
    # `output_padding` is the row that argued for comparing values and not only
    # shapes. It extends the window at the bottom and the right, and **the extra rows
    # are not zeros**: it reaches back into the part the padding trim was about to
    # throw away, which holds computed values. Measured on twelve of the first
    # fifty-six configurations, every one of them `padding` and `output_padding`
    # together.
    #
    # 2 → 1. EmbeddingBag took torch's whole list: `mode` sits sixth, so
    # `EmbeddingBag(10, 3, "sum")` set `max_norm="sum"` in torch and the mode here —
    # both sides then build a layer and return bags of the right shape, and only the
    # numbers differ. `padding_idx` needed measuring rather than reading: the padded
    # entry **leaves the bag** rather than contributing zero, which is the same thing
    # under `sum` and not under `mean`, where it has to leave the denominator.
    #
    # 1 → 0, from the other half of the split in the same window: the four
    # weight-first losses took torch's lists and `BCEWithLogitsLoss` left this bucket.
    #
    # **Two sessions lowering one number is why it is written with every reason
    # rather than a running total.** This line has conflicted on rebase five times;
    # each time the resolution kept both stories, because a single figure with one
    # story attached makes the other invisible, and the thing a reader will want a
    # year from now is which change moved it.
    #
    # **Zero here does not mean `nn` agrees with torch.** 52 rows are `shorter` and 8
    # `unaligned`; what is gone is the bucket where an argument sits in another's
    # seat. That is the dangerous one and it is empty. The rest is work, not risk.
    "nn": 0,
    "nn.functional": 0,
    # 2 → 1. `SGD` took torch's `dampening`, `nesterov` and `maximize`, and `Adagrad`
    # its `initial_accumulator_value` — implemented, and checked against real torch
    # over six configurations. `SGD(p, 0.1, 0.9, 1e-4)`, the line a torch tutorial
    # writes, used to set the **dampening** to the weight decay and leave the decay
    # at zero: two different things, both plausible small numbers, and the run trains
    # and trains slightly wrong.
    #
    # `Adagrad` stays because `foreach` sits before torch's keyword-only boundary and
    # we have nothing to put there — it changes no value, so it is absent rather than
    # refused, and the row is honest about the position being unmatched.
    "optim": 1,
    "optim.lr_scheduler": 2,
    "linalg": 0,
    "utils.data": 0,
}

# The names cannot be aligned against each other, so the row says so rather than
# guessing. `nn.functional`'s 30 are mostly the core naming its first argument `x`
# where torch says `input`.
UNALIGNED = {
    "Tensor": 0,
    "nn": 8,
    "nn.functional": 27,
    "optim": 0,
    "optim.lr_scheduler": 0,
    "linalg": 0,
    "utils.data": 1,
}

# The core takes a prefix of torch's arguments — a **feature torch has and we do
# not**, which is `torch_gap.py`'s kind of finding rather than a silent shift.
SHORTER = {
    "Tensor": 2,
    # 54 → 60 and the judged share 132 → 144 of 161. **Nothing got worse: twelve rows
    # became visible.** The twelve lazy layers declared `(*args, **kwargs)` and sat in
    # the uncomparable bucket while every other layer was measured; they declare their
    # target's signature minus what they infer now, which is torch's own rule for
    # them, so the axis can judge them. Six land on `agree` — the convolutions, whose
    # eager forms match torch — and six on `shorter`, because their targets are short
    # of torch and a lazy layer is exactly as complete as what it becomes.
    #
    # 60 → 58 from the other half of the split: two of the weight-first losses
    # stopped being short when they took torch's argument lists.
    "nn": 58,
    "nn.functional": 0,
    # 10 → 11. `SGD` left `shifted` and arrived here: it now agrees with torch as far
    # as `maximize` and stops, because `foreach`, `differentiable` and `fused` are
    # torch's execution switches and change no value. **A row moving from `shifted`
    # to `shorter` is the fix**, not a wash — one meant an argument in the wrong
    # seat, this one means a feature torch has and we do not.
    "optim": 11,
    "optim.lr_scheduler": 1,
    "linalg": 0,
    "utils.data": 1,
}

# **torch is implemented in C here and `inspect` cannot read it.** Pinned as a total
# rather than per namespace, because the number is a property of torch's build and
# not of our work: it moves when torch moves. It is pinned at all so that a change in
# how the measurement reads signatures — a fourth parser, say — cannot quietly turn
# unreadable rows into comparisons nobody checked.
UNREADABLE_IN_TORCH = 571

# **How many rows each namespace actually gets judged on, out of how many are filed.**
# The measurement's docstring prints this table as prose, and prose goes stale in
# silence; this is the claim, checkable.
#
# It is also the concrete form of what a floor cannot do. A floor asks whether fewer
# things were compared. The fault this axis shipped with compared the same number and
# **classified all of them into a bucket meaning not-our-problem** — an absorbing
# state, where a row is still counted and no longer judged. Pinning the ratio per
# namespace is what makes that say something: `Tensor` going from 9-of-512 to
# 0-of-512 moves a number here and moves nothing in any total.
JUDGED = {
    "Tensor": (9, 512),
    # 119 → 132. Thirteen loss constructors left the uncomparable bucket when they
    # stopped being `(*args, reduction='mean', **kw)` and grew torch's own parameter
    # list, and all thirteen landed in `agree`. **The ratio moving upward is what a
    # fix looks like here** — the total did not change, and no other number in this
    # file would have recorded that anything happened.
    "nn": (144, 161),
    "nn.functional": (76, 126),   # +1: embedding_bag became readable
    "optim": (14, 14),
    "optim.lr_scheduler": (16, 16),
    "linalg": (0, 42),
    "utils.data": (13, 18),
}


def _rows():
    import torch_signatures_core
    return torch_signatures_core.compare()


def test_the_core_to_torch_argument_axis_has_not_widened():
    rows = _rows()
    assert set(rows) == set(SHIFTED), (
        f"the namespaces measured changed: {sorted(set(rows) ^ set(SHIFTED))}")

    moved, unreadable = [], 0
    for space, found in sorted(rows.items()):
        got = {}
        for _n, _m, _y, note in found:
            got[note] = got.get(note, 0) + 1
        unreadable += got.get("torch is C", 0)
        pairs = (
            ("shifted", SHIFTED,
             got.get("dropped", 0) + got.get("inserted", 0) + got.get("reordered", 0)),
            ("unaligned", UNALIGNED, got.get("unaligned", 0)),
            ("shorter", SHORTER, got.get("shorter", 0) + got.get("longer", 0)),
        )
        for label, table, count in pairs:
            if count != table[space]:
                moved.append(f"{space} {label}: {count} now, {table[space]} written")
    assert not moved, (
        "the core-against-torch counts moved:\n  " + "\n  ".join(moved)
        + "\n\n  Lower means an argument list was brought into line — edit the table.\n"
          "  Higher wants saying out loud. Read the rows first:\n"
          "    uv run --with numpy --with torch --with torchvision \\\n"
          "      python tests/torch_signatures_core.py --show nn")
    assert unreadable == UNREADABLE_IN_TORCH, (
        f"{unreadable} rows are unreadable in torch, and {UNREADABLE_IN_TORCH} were "
        "written down.\n"
        "  Lower means something started reading C signatures — check what authority\n"
        "  it is using before believing the rows it produced. torch's testing\n"
        "  overrides are abbreviated (measured: Tensor.std loses correction and\n"
        "  keepdim), so a source that looks better can be worse.\n"
        "  Higher usually means a torch upgrade moved something into C.")


def test_each_namespace_is_judged_on_as_much_as_it_was():
    """The judged-to-filed ratio per namespace, exactly.

    **`linalg` is 0 of 42 and `Tensor` is 9 of 512.** The tensor surface is the
    largest body of API in the project and it has no outside authority on arguments
    at all — what holds it is the core and borch.ts agreeing with each other, plus
    the golden, which compares values and only for cases somebody wrote.

    Written down because a coverage figure taken over the judged rows alone would
    read as 94%, and because knowing where a finding *could* have come from is worth
    more than that figure. `std` was caught on the axis that cannot say which side is
    wrong; had both libraries taken the correction first, nothing would have noticed.
    """
    rows = _rows()
    moved = []
    for space, found in sorted(rows.items()):
        judged = sum(1 for r in found
                     if r[3] not in ("torch is C", "variadic", "no signature"))
        want_judged, want_filed = JUDGED[space]
        if (judged, len(found)) != (want_judged, want_filed):
            moved.append(f"{space}: {judged} of {len(found)} now, "
                         f"{want_judged} of {want_filed} written")
    assert not moved, (
        "the judged-to-filed ratios moved:\n  " + "\n  ".join(moved)
        + "\n\n  Judged going **down** while the total holds is the dangerous one:\n"
          "  it means rows moved into a bucket that means 'cannot judge', which is\n"
          "  an absorbing state and invisible to any count of the total. That is the\n"
          "  fault this axis shipped with — every C method silently classified as\n"
          "  absent, and Tensor reading as three agreements and finished.")


def test_the_measurement_still_compares_something():
    """A floor, for the reason `test_ts_signatures.py` has one.

    This axis lost a whole namespace on its first run: `_read` returned the same
    value for "torch does not have this name" and "torch has it and `inspect` cannot
    read it", so every C-implemented method was skipped as absent. `Tensor` came back
    with three agreements and read as finished.

    **A row that is never produced cannot be counted as missing**, so the defence has
    to be a floor beside the counts rather than a smaller number inside them.

    **And this floor does not catch that particular regression** — put the fault back
    and it stays green, because the rows that vanish were never in the compared count
    to begin with. `UNREADABLE_IN_TORCH` is what goes red (0 against 571). Worth
    saying out loud: a floor guards against *comparing fewer things*, and this fault
    was *classifying fewer things*. Two different silences, and the same test does not
    cover both. Verified by breaking it in each direction.
    """
    rows = _rows()
    compared = sum(1 for found in rows.values() for r in found
                   if r[3] not in ("torch is C", "variadic", "no signature"))
    assert compared > 150, (
        f"only {compared} argument lists were compared, and there were 246.\n"
        "  Check `_read` first: returning one value for absent and for unreadable\n"
        "  drops every C-implemented method without leaving a row behind.")


def test_the_measurement_still_runs_as_a_script():
    out = subprocess.run(
        [sys.executable, str(ROOT / "tests" / "torch_signatures_core.py"),
         "--show", "utils"],
        capture_output=True, text=True, cwd=ROOT)
    assert out.returncode == 0, out.stderr[-2000:]
    assert "utils.data" in out.stdout
