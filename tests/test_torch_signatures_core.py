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
    # `Tensor` 2 → 0. `norm` took torch's `keepdim` between `dim` and `dtype`,
    # through all six branches of `p`; `unique` took `return_inverse` second and
    # `dim` last. Both were middle-drops: `x.norm(2, 1, True)` set the dtype to
    # `True` and folded the axis away, and `x.unique(True, True)` asked for the
    # inverse in torch and the counts here — the same call, a different tuple,
    # both sides plausible.
    #
    # **Zero here is worth less than the other zeros in this file.** Only 9 of
    # `Tensor`'s 512 rows are judged at all; the rest are `torch is C`. The
    # tensor surface is not clean, it is unwatched — see JUDGED below.
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
    "Tensor": 0,
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
    # **1 → 0, and nothing about `Adagrad` changed.** The row said an argument sat in
    # the wrong seat: the core's `maximize` where torch has `foreach`. Both libraries
    # write it `*, maximize`, so no positional call can reach either one. The hazard
    # could not be written down in Python.
    #
    # It was never a bad row in a working bucket. `shifted` claims *a positional call
    # lands on the wrong parameter*, and the measurement was deciding that from names
    # in order with no way to ask whether a positional call was possible — in every
    # row, including all 109 it calls `agree`. `signature_read.positional()` exists
    # now and `_reachable()` re-asks the same verdict on the reachable prefix.
    #
    # `Adagrad` moved to `shorter`, where it names `foreach`. That direction is the
    # whole point: the tempting fix was to drop keyword-only parameters in
    # `parameters()` itself, which retires two invented hazards and **hides 54 real
    # absences** — thirteen `bias` flags on the normalisation layers, six on `Adam`.
    # See `test_keyword_only_arguments_stay_visible`, which is the wall in front of
    # that fix.
    "optim": 0,
    # 2 → 1. `ReduceLROnPlateau` took torch's `threshold_mode`, `cooldown`
    # and `eps` — three arguments missing from the middle, so a call written
    # from torch's documentation put the cooldown where `threshold_mode` goes
    # and the minimum rate where `cooldown` does. All three implemented and
    # checked against real torch over seven configurations, twelve steps each.
    #
    # **1 → 0.** `OneCycleLR` was the row this comment said was "work rather than a
    # shift, and it is not done". Both halves of that were true and it was still the
    # most dangerous row on the axis: the two lists agree for three arguments and then
    # part for **eleven consecutive positions**. `OneCycleLR(opt, 0.1, None, 10, 100)`
    # — a torch recipe's ten epochs of a hundred steps — set `pct_start` to 10 and
    # `div_factor` to 100. `pct_start` is a fraction of the cycle, so the rate climbs
    # past the end of the run and never comes down, and nothing raises.
    #
    # Done rather than described: `epochs`/`steps_per_epoch`, `anneal_strategy`,
    # `three_phase`, and the momentum cycle, which torch turns on by default and this
    # simply did not have. Checked against real torch over 24 configurations of
    # strategy × phases × momentum × `pct_start`, twenty steps each, learning rate and
    # momentum both — worst difference 1.1e-16, and the `betas` path against `Adam`
    # separately.
    #
    # Two arithmetic edges the first version got wrong and torch caught: the middle
    # boundary under `three_phase` is `2 × pct_start × total_steps − 2`, and momentum
    # in the third phase is **flat at `max_momentum`** rather than following the rate.
    # The obvious rule — momentum runs opposite to the rate — gets both other phases
    # exactly right, which is why it survived being written.
    "optim.lr_scheduler": 0,
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
    #
    # 58 → 57. `Embedding` took `(num_embeddings, embedding_dim)` where torch takes
    # nine more, and it is the one row this axis could see that `torch_gap.py` could
    # not: that table counts names, `Embedding` was present, and `nn` read 95%.
    # `EmbeddingBag` next door had carried torch's full list all along, so the two
    # neighbours disagreed about the same five arguments and nothing said so.
    # **Still 57, and it dropped to 44 for an hour on the way here.** The thirteen
    # normalisation layers took torch's `affine`, `track_running_stats` and `bias`,
    # and briefly took `device` and `dtype` as well — which made them `agree` and
    # this number fall. `LayerNorm` next door does not carry those two, and
    # following torch past the repository's own settled choice would have put five
    # rows on the borch.ts axis into `unaligned` to take thirteen out of here.
    #
    # So the count staying put is the right outcome and not a failure to improve:
    # what these thirteen are short of now is `device` and `dtype` alone, which is
    # what every layer in this bucket is short of.
    "nn": 57,
    "nn.functional": 0,
    # 10 → 11. `SGD` left `shifted` and arrived here: it now agrees with torch as far
    # as `maximize` and stops, because `foreach`, `differentiable` and `fused` are
    # torch's execution switches and change no value. **A row moving from `shifted`
    # to `shorter` is the fix**, not a wash — one meant an argument in the wrong
    # seat, this one means a feature torch has and we do not.
    #
    # 11 → 12: `Adagrad` arrived the same way, except that nothing about `Adagrad` was
    # fixed. The measurement stopped claiming a positional hazard that Python forbids,
    # and what remains is the true difference — torch has a positional `foreach` after
    # `eps` and the core stops there. Landing it here rather than in a bucket of its
    # own is deliberate: this is the only bucket that names a missing argument, and a
    # private one would have taken `foreach` out of the report to make it look tidier.
    # 12 → 10. Two optimizers left for `agree` when they took `maximize`: torch has
    # it keyword-only, so their positional lists were already level and the name was
    # the whole difference.
    "optim": 10,
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


# **torch's keyword-only parameters that the core does not have.** Counted here so
# that hiding them has a price.
#
# The fix that retired the `Adagrad` row could have been written two ways. The one
# taken adds `signature_read.positional()` and uses it only to re-ask a shift verdict
# on the reachable prefix. The one not taken drops keyword-only parameters inside
# `parameters()`, which is three lines shorter, retires the same row, and makes every
# number below vanish at once — 54 arguments torch has and we do not, gone from every
# comparison in the repository, with no bucket going up to say so.
#
# It is the shape this repository keeps finding, in its most comfortable form: **the
# cheaper fix and the silent one are the same edit.** A total would not have moved.
# `SHORTER` would have gone *down*, which reads as progress. The only thing that
# separates them is a number that exists solely to be lowered by work and raised by
# nothing else.
#
# Thirteen of the 54 are one argument: `bias` on `BatchNorm{1,2,3}d`, `GroupNorm`,
# `InstanceNorm{1,2,3}d` and the six Lazy variants — `BatchNorm1d(..., *, bias=True)`.
# Six more are `Adam`'s. None of them is fixed by this commit; they are made
# countable by it, which is the difference between an absence and a silence.
# 54 → 41. **Thirteen of the thirteen `bias` flags are paid** — the three
# `BatchNorm`s, the three `InstanceNorm`s, `GroupNorm` and the six lazy variants all
# take it now, and `affine` came with it because the two are halves of one idea:
# `affine=False` is no learnable scale or shift at all, `bias=False` keeps the scale
# and drops the shift. Checked against real torch over 24 configurations, worst
# difference 7.8e-07, and the `state_dict` keys match in every one.
#
# **The number is why it happened.** The absence had been sitting in this file since
# `positional()` was written, as a count rather than as a paragraph, and a count is
# something somebody eventually pays.
#
# The 41 left are a different kind and it is worth saying so rather than leaving them
# to read as the same backlog: about thirty are torch's execution switches —
# `foreach`, `capturable`, `differentiable`, `fused` — which change no value and are
# absences in name only. The ones that do change values are `maximize` on eight
# optimizers and `Adam`'s `decoupled_weight_decay`, and those are work.
# 41 → 33. **`maximize` on ten optimizers**, which is the other half of what this
# number was holding: it turns the gradient round, so it changes values, where
# `foreach` and `fused` do not. Checked against real torch over twelve optimizers ×
# two directions × five steps — worst difference 2.4e-07, and `maximize=True` moves
# the answer on every one of the twelve, which is the half a value comparison alone
# would not have asked.
#
# The 33 left are almost entirely torch's execution switches — `foreach`,
# `capturable`, `differentiable`, `fused` — which change no value. `Adam`'s
# `decoupled_weight_decay` and `dim_order`'s `ambiguity_check` are the two that are
# still work.
KEYWORD_ONLY_ABSENCES = 33


def _keyword_only_gaps():
    """`[(space, name, [argument, ...]), ...]` — torch's keyword-only, ours absent.

    **Reads `inspect` directly and not `signature_read.parameters`.** A check that
    guards a reader against dropping something cannot ask that reader whether it
    dropped it — the answer would come back clean by construction, and the count
    would fall to zero for the one reason it must never fall to zero for.
    """
    import inspect

    import torch_gap
    import torch_signatures_core

    def read(thing):
        target = thing.__init__ if inspect.isclass(thing) else thing
        try:
            return inspect.signature(target).parameters
        except (TypeError, ValueError):
            return None

    found = []
    for space, ours, theirs in torch_signatures_core._spaces():
        for name in sorted(torch_gap._public(ours)):
            mine, yours = getattr(ours, name, None), getattr(theirs, name, None)
            if mine is None or yours is None:
                continue
            theirs_p, mine_p = read(yours), read(mine)
            if theirs_p is None or mine_p is None:
                continue
            missing = [p.name for p in theirs_p.values()
                       if p.kind is p.KEYWORD_ONLY and p.name not in mine_p]
            if missing:
                found.append((space, name, missing))
    return found


def test_keyword_only_arguments_stay_visible():
    """Reading a signature must not drop what cannot be passed by position.

    This is the direction `_reachable()` must not over-correct in. It fails if
    `parameters()` ever starts filtering by `kind` — every gap below would read as
    zero, and no other check in this repository would notice.
    """
    gaps = _keyword_only_gaps()
    total = sum(len(args) for _s, _n, args in gaps)
    assert total == KEYWORD_ONLY_ABSENCES, (
        f"{total} keyword-only arguments are torch's and not ours, "
        f"{KEYWORD_ONLY_ABSENCES} written.\n"
        "  Lower means some were implemented — edit the number down and say which.\n"
        "  Zero means the reader started hiding them rather than the core gaining\n"
        "  them; check `signature_read.parameters` before believing it.\n  "
        + "\n  ".join(f"{s}.{n}: {', '.join(a)}" for s, n, a in gaps[:8]))

    from signature_read import parameters
    import torch
    assert "maximize" in (parameters(torch.optim.Adagrad.__init__) or []), (
        "`parameters()` has stopped reporting keyword-only arguments. That retires "
        "the two rows `_reachable()` was written for and hides all "
        f"{KEYWORD_ONLY_ABSENCES} absences above with them.")


def test_a_forbidden_shift_is_not_reported_as_one():
    """The `Adagrad` shape, asked directly rather than through the totals.

    Built by hand so the check survives `Adagrad` being fixed: a pair whose names go
    out of order only where neither side can be called positionally.
    """
    import ts_signatures
    from signature_read import parameters, positional

    def theirs(a, b, c=None, *, maximize=False, fused=None):
        pass

    def ours(a, b, *, maximize=False):
        pass

    assert ts_signatures._verdict(parameters(ours), parameters(theirs)) == "inserted", (
        "the by-name reading of this pair is supposed to claim a shift — if it does "
        "not, the example no longer demonstrates the thing it guards")
    # `longer` rather than `shorter` because `_verdict(wanted, yours)` is asked with
    # ours first: torch's list runs past the end of ours. Both land in the same bucket
    # in the tables above, which is why the totals read as "shorter" either way.
    assert ts_signatures._verdict(positional(ours), positional(theirs)) == "longer", (
        "asked of the reachable prefix, the same pair is a short tail and nothing more")


# **Rows where torch reaches further by position than the core does.**
#
# **This was called `TORCH_REACHES_FURTHER_BY_POSITION` for about an hour**, which is
# the same defect as `shifted` in a name written while arguing against it. The
# evidence supports *torch accepts more positions than we do*. Whether any particular
# call raises depends on how many the caller writes, and the extra positions are often
# ones nobody passes positionally — measured:
#
#     Dropout(0.5, True)            torch ok · ours TypeError
#     ELU(1.0, True)                torch ok · ours TypeError
#     Flatten(1, -1)                torch ok · ours TypeError
#     AvgPool2d(2, 2, 0, False)     torch ok · ours TypeError
#     GroupNorm(2, 4, 1e-5, True)   torch ok · ours ok
#     Bilinear(2, 3, 4, True)       torch ok · ours ok
#
# The last two are in the 57 and neither raises: torch's extra positions there are
# `device` and `dtype`, which no recipe passes by position. So 57 is an **upper
# bound on names**, not a count of broken calls, and the name has to say the thing
# that was measured.
#
# 57, and the number is the point. It was first noticed on `Adagrad` —
# `Adagrad(params, 0.01, 0, 0, 0, 1e-10, True)` sets `foreach` in torch and raises
# here — and the tempting move was to give `Adagrad` a bucket of its own for it.
# `Adagrad` is one of 57. `SGD`, `AvgPool2d`, the six `BatchNorm`s and fifty more
# have exactly the same property, and `SHORTER` has been counting them all along
# under a name that says what is missing rather than what happens.
#
# Naming the behaviour after the row it was noticed on would have been the same
# scope mistake one level up: a true observation about `Adagrad` turned into a
# category, in a file whose whole subject is categories claiming more than their
# evidence.
TORCH_REACHES_FURTHER_BY_POSITION = 57

# **`agree` rows with the same problem: none.** Worth pinning precisely because it
# is empty. `agree` means the two name lists match, and the worry — raised while
# `positional()` was being written — is that two lists can match by name while one
# side takes fewer of them positionally, so the row reads as identical and a torch
# call still raises. Measured, that set is empty today.
#
# An empty set that nobody counts is the absorbing bucket this file keeps finding:
# the check would pass the day it stops being empty and say nothing. Pinned at zero,
# it fails.
AGREE_ROWS_THAT_RAISE = 0


def _positional_shortfalls():
    """`[(space, name, verdict, how_many_more), ...]` — torch reaches further."""
    import torch_signatures_core

    rows = torch_signatures_core.compare()
    out = []
    for space, ours, theirs in torch_signatures_core._spaces():
        for name, _mine, _kept, note in rows[space]:
            if note in ("torch is C", "variadic", "no signature"):
                continue
            mine = torch_signatures_core._read(space, ours, name, reach=True)
            yours = torch_signatures_core._read(space, theirs, name, reach=True)
            if not isinstance(mine, list) or not isinstance(yours, list):
                continue
            yours = [p for p in yours if p not in torch_signatures_core.DEPRECATED]
            if len(yours) > len(mine) and yours[:len(mine)] == mine:
                out.append((space, name, note, len(yours) - len(mine)))
    return out


def test_where_torch_reaches_further_by_position_is_counted():
    found = _positional_shortfalls()
    assert len(found) == TORCH_REACHES_FURTHER_BY_POSITION, (
        f"{len(found)} rows take fewer positional arguments than torch, "
        f"{TORCH_REACHES_FURTHER_BY_POSITION} written.\n"
        "  Lower means arguments were added — edit the number down.\n  "
        + "\n  ".join(f"{s}.{n} ({v}) +{k}" for s, n, v, k in found[:8]))

    slipped = [(s, n) for s, n, v, _k in found if v == "agree"]
    assert len(slipped) == AGREE_ROWS_THAT_RAISE, (
        f"{len(slipped)} rows read as `agree` while torch reaches further by "
        "position — the name lists match and a torch call still raises here:\n  "
        + "\n  ".join(f"{s}.{n}" for s, n in slipped[:8]))
