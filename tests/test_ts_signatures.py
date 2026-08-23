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
# **`linalg` was 14 differ against a namespace nobody could call**, and it is 0 now.
# The first run of this axis paired `torch.linalg.det(t)` against `det(f)` and
# `matmul(a, b)` against `matmul(a, b, n, k, m)` — flat, `Mat`-based CPU internals
# that `site/build_api.py` was publishing as though they were the torch namespace,
# because the file happened to be *named* `linalg`. That was a reachability question
# already open, and this axis is where it turned from an argument into a number.
#
# A peer session split the file (`_linalg.ts` for the numerics, `linalg.ts` for the
# namespace) and the row became readable in the same edit. The last six left it when
# `_verdict` learned to tell a shift from a rename that is also short — see below.
SHIFTED = {
    # `Tensor` was 4 here and one of them was the worst row the axis has produced:
    # `std(dim, unbiased, keepdim)` against `std(correction)`. This pair took the
    # correction first, alone among the reductions, so `x.std(0)` — the line anybody
    # transcribing torch writes — compiled, ran, and returned a scalar at correction 0
    # where torch returns one value per column. A wrong **rank**, which flows
    # downstream and breaks somewhere unrelated. A peer fixed it (`dim` first, as in
    # `mean`, `sumDim` and `amax`) and found the same defect's second half while
    # there: `stdMean` took its mean over everything regardless of the axis.
    #
    # Nothing else was watching that place — the golden asks `std()` three times and
    # never with an argument. Worth noting beside `ReduceLROnPlateau`: `tsc` caught
    # the scheduler's call site the moment `mode` was added, because the type was a
    # string union, and could not catch this one because `number` is not narrow.
    # `nn` was 1 and is 0: `HuberLoss` took `(delta, reduction)` where torch takes
    # `(reduction, delta)`, so `new HuberLoss('sum')` set the delta to a string. It is
    # the odd one out — the nine other margin-taking losses all match torch exactly,
    # which is what showed the nine `inserted` rows this table briefly carried were
    # an artefact of dropping the core's `*args` rather than a fault in borch.ts.
    "Tensor": 1,
# **0 → 1 → 0 → 3 in one afternoon, and the shape of that trace is the finding.**
    #
    # `MaxPool1d/2d/3d`: the core took torch's `padding`, `dilation` and `ceil_mode`,
    # which opened the row — borch.ts still had `(kernel, stride, returnIndices)`, so
    # `new nn.MaxPool2d(2, 2, 1)` set `returnIndices` where torch and the core set
    # `padding`. The session that moved the core **raised this number with the reason
    # attached rather than holding it**, which is the only thing that made the gap a
    # row instead of a silence. borch.ts then took all six positions; `padding`,
    # `dilation` and `ceilMode` **refuse**, because the WGSL pooling kernel does not
    # do them yet and an argument that raises with its own name beats one that
    # quietly takes another's seat.
    #
    # `Conv1d/2d/3d`: opened the same way and still open. The core grew `dilation`,
    # `groups` and `padding_mode`, and moved `bias` from sixth to eighth where torch
    # has it; borch.ts still takes `(in, out, kernel, stride, padding, bias)`, so
    # `new nn.Conv2d(3, 16, 3, 1, 1, false)` turns the bias off there and sets a
    # dilation here.
    #
    # **Closing a gap against an outside authority opens one between two of our own
    # libraries first**, and it is not a side effect. With three implementations and
    # one outside authority, the core can only move toward torch by moving away from
    # borch.ts until borch.ts follows. The pair of axes disagreeing is the mechanism
    # working. The danger is only ever a number held still while the sides part.
    #
    # 4 → 7. Three more the same way: the core's ConvTranspose1d/2d/3d took torch's
    # output_padding, groups and dilation, and borch.ts had not followed. That was
    # the fourth family in a row to open a row here while closing one against torch.
    #
    # 7 → 4. The three Conv layers followed the core: torch's order, with `bias`
    # eighth and `dilation` sixth. `dilation` is implemented in the shader (one token
    # in three index expressions, plus the spacing in the cache key, without which two
    # calls differing only in dilation would silently share a shader), `groups` by
    # slicing and joining so the gradient follows from the pieces, and `padding_mode`
    # by padding in the layer as torch's layer does.
    #
    # **`tsc` named all six positional call sites the moment the constructor moved** —
    # `new Conv2d(cin, cout, 3, stride, 1, false)`, with a boolean where a number now
    # goes. The same move in Python was silent, and six tests broke on it instead.
    #
    # 4 → 1. The three ConvTranspose layers followed: torch's order, `bias` eighth
    # and `dilation` **ninth**, which is not the convolution's order and is torch's.
    # `outputPadding` is expressed as a longer output and nothing else — the shader
    # finds, for each output cell, the input cells that reach it, and asked for more
    # cells it answers by the same rule, which is what torch's extra rows are. They
    # are **not zeros**, and a version that wrote zeros would agree on every shape and
    # part on the values.
    #
    # 1 → 0. `MaxPool2d` closed from the other half of the split: borch.ts took all
    # six of torch's positions, with `padding`, `dilation` and `ceilMode` **refusing**
    # rather than working, because the WGSL pooling kernel does not do them yet. An
    # argument that raises with its own name beats one that quietly takes another's
    # seat — the same trade the core made for `add_bias_kv`.
    #
    # **Zero here does not mean borch.ts matches the core.** Nineteen rows are
    # `shorter` and nineteen `unaligned`. What is empty is the bucket where an
    # argument sits in another's seat, and that was the dangerous one.
    "nn": 0,
    # 1 → 2 → 1. `F.embedding_bag` moved `mode` from third to sixth on the core side,
    # and `F.embeddingBag` followed. **`tsc` named all eight call sites the instant it
    # moved** — five golden cases and the layer's own two — because a mode string does
    # not fit `number | null`. The identical move on the Python side was silent.
    "nn.functional": 1,
    "optim": 0,
    "optim.lr_scheduler": 0,
    "linalg": 0,
    "utils.data": 0,
}

# **238 pairs cannot be compared at all, and that is the largest number here.**
# The core writes 238 of its callables as `(*args, reduction='mean', **kw)` or the
# like — it takes whatever is passed and ignores what it does not know. There is no
# argument list to compare, so this axis is blind to a third of the surface, and it
# is blind in the direction that matters: `borch.nn.HuberLoss(delta=0.5)` is accepted
# and does nothing.
#
# Not pinned as a table, because it is not this axis's finding to hold — it is the
# core's own signature surface, and the check that ought to catch it is a core ↔ real
# torch signature axis, which does not exist. `torch_gap.py` measures those two by
# name only. Written here so the number is not mistaken for a measurement problem.
UNREADABLE_TOTAL = 238

# **Neither list can be aligned against the other by name.** Not shown to be a shift
# and not shown to be safe — a row to read, held apart so it is neither claimed as a
# defect nor tidied in with the spelling differences.
#
# `optim`'s 7 are one shape worth naming: torch's `Adam(params, lr, betas, eps,
# weight_decay)` against borch.ts's `(params, lr, beta1, beta2, eps, weightDecay)`.
# The pair became two positions, which is a real arity change and not a rename.
UNALIGNED = {
    # 2 → 6. **Thirty-odd `Tensor` methods stopped being uncomparable** when the
    # method binders started setting `__wrapped__` — until then `inspect` saw
    # `(self, *args, **kw)` and the axis filed them under `variadic`, which means
    # *nothing was compared*. Four landed here: `stft`, `istft`, `isclose` and
    # `random_`, where borch.ts takes the tensor as a first argument or stops
    # short of torch's list.
    #
    # **`shifted` did not move**, which is the number that would have mattered:
    # nothing newly visible has an argument in the wrong seat.
    "Tensor": 6,
    # `SmoothL1Loss` left this table when a peer fixed it: the core took
    # `(beta, reduction)` and borch.ts `(reduction, beta)`. **borch.ts was right** —
    # torch's live arguments are `(reduction, beta)`, with the deprecated
    # `size_average` and `reduce` in front of them. So `SmoothL1Loss("sum")` set
    # `beta="sum"` in Python and `reduction="sum"` in TypeScript, and nothing raised
    # at construction either way.
    #
    # **This axis could not say which side was wrong**, and that is the sentence to
    # carry out of it. Its two sides are the core and borch.ts, and neither of them
    # is torch, so a row here means they disagree and never which one to move.
    # `parity.ts` has the same blind spot in the same words — *the sisters have
    # parted*. Two instruments, one blind spot, because they share a pair of sides.
    # Asking real torch settled it, and no check in this repository does that for
    # `borch.nn`: `test_torch_signatures.py` covers borchvision against torchvision
    # and stops there.
    # 19 -> 25. The twelve lazy layers stopped declaring `(*args, **kwargs)`
    # and can be judged now -- six of the six that moved here are the norms,
    # whose eager forms borch.ts spells differently. The count rose because the
    # measurement reaches further, not because anything parted.
    # 26 → 27. `Hardtanh`, which took torch's deprecated `min_value`/`max_value`
    # alongside `inplace` and so is no longer a prefix of borch.ts's list.
    # 27 → 36. **The core-to-torch `shorter` row fell from 45 to 20 and this is
    # where the difference went.** Carrying `device`/`dtype` in order to refuse
    # them lines the core up with the outside authority and moves it away from
    # borch.ts, which has neither and no seat for them.
    #
    # That trade was refused once already, on the thirteen normalisation
    # layers, and taken here — the difference is size: five rows into
    # `unaligned` for thirteen out of `shorter` was not worth it, twenty-five
    # for nine is. Written down because the two look like one decision made
    # twice and are one decision made on different numbers.
    "nn": 36,
                #     are the same length again — it left for `renamed` below, which
                #     is a spelling difference rather than a shape one
                # +1, Embedding: a layer borch.ts did not have, so nothing could be
                #     compared. It parts the same two ways `EmbeddingBag` next door
                #     already does — `_weight`/`_freeze` against `weightIn`/`freeze`,
                #     and no `device`/`dtype`, which borch.ts has nowhere.
                #     **A count going up because a name became comparable is not the
                #     same as two sides drifting**, and this bucket cannot tell them
                #     apart on its own; that is what the comment is for.
    "nn.functional": 1,
    # 7 → 1 → 7. It went down when `maximize` landed on both sides at the same
    # length, and back up when the core took torch's **whole** optimizer surface —
    # `amsgrad`, `centered`, `momentum`, `decoupled_weight_decay` and the four
    # execution switches — and borch.ts did not follow.
    #
    # **This is the axis paying its stated price, not a regression.** The comment
    # below already says it: closing a gap against torch opens one between our own
    # two libraries first, because the core can only move toward the outside
    # authority by moving away from borch.ts until borch.ts follows. `optim` on the
    # core-to-torch axis is 0 now — the first namespace there to empty — and these
    # seven are what that cost.
    #
    # **`Adam` would stay here even after borch.ts follows**: over there the pair is
    # `beta1, beta2` where torch has one `betas`, so the lists cannot be the same
    # length whatever else is added. That one is a shape difference and not a debt.
    "optim": 7,
    "optim.lr_scheduler": 3,
    "linalg": 6,
    "utils.data": 0,
}

# Same arity, names differ. **Pinned, because this bucket was called harmless and
# is not.** The reasoning was that TypeScript has no keyword arguments; then
# `F.gumbel_softmax(logits, tau, hard, eps, dim)` turned up against `(logits, tau,
# hard, dim, noise)`, where position three is a tolerance in torch and an axis in
# borch.ts. A name difference and a different argument look identical from here.
#
# A difference that really is only spelling belongs in `ts_signatures.RENAMES`, where
# a person attests it and the row becomes `agree`. That is the only way out of this
# table, and it is deliberately a way that requires someone to write a sentence.
RENAMED = {
    # 30 → 28. `flatten` arrived from `shorter`; `type` and `bincount` left for
    # `agree` when they took torch's arguments — `type()` with none names the
    # type, and `bincount` takes `weights` and `minlength` as `_ops.bincount`
    # next door already did.
    # 28 → 55. The rest of the same thirty, almost all of them a spelling —
    # `b` against `other`, `size` against `split_size`. A large jump in this
    # bucket from a change that compared *more* is the expected shape; the same
    # jump from a change that compared the same number would not be.
    "Tensor": 55,
    # 19 → 20. `GroupNorm` arrived from `shorter` when both sides took `affine` and
    # `bias`: same length now, and borch.ts spells the flag `useBias`, as it already
    # does in `LayerNorm`, `Bilinear` and the recurrent layers.
    # 20 → 19. `Hardtanh` left for `unaligned`.
    # 19 → 10. Nine layers left for `unaligned` when the core took torch's
    # `device`/`dtype`: same names as borch.ts up to a point, then two more that
    # borch.ts has no seat for at all.
    "nn": 10,   # Bilinear arrived from `unaligned`: borch.ts spells the flag
                #     `useBias`, as it already does in LayerNorm and the recurrent
                #     layers, where the constructor has a `bias` field to not shadow
    "nn.functional": 1,
    # 7 → 1: the six went back to `unaligned` when the core took torch's whole
    # optimizer surface and borch.ts stayed where it was. See the note there.
    "optim": 1,
    "optim.lr_scheduler": 0,
    "linalg": 17,
    "utils.data": 0,
}

# borch.ts takes a **prefix** of torch's arguments. Not a shift and not silent — one
# too many raises. Held separately so that fixing a shift cannot be paid for by
# turning it into a truncation without anyone noticing.
SHORTER = {
    # 16 → 15. `flatten` took torch's `end_dim`, so its list is no longer a
    # prefix of borch.ts's — it moved to `renamed` below.
    # 15 → 17. Two short tails, `stft`'s `align_to_window` and `backward`'s
    # `create_graph`/`inputs`.
    # 17 → 18. `relu` took torch's `inplace` in the core; borch.ts has no in-place
    # write-back to route it to, so it takes the prefix. This is the paragraph below
    # happening again, and the row is named rather than the count nudged: the whole
    # increase is one row, and if a second one ever rides along on the same reason it
    # has to be written down too.
    "Tensor": 18,
    # 15 → 10 → 13 → 24. The loss constructors followed the core into torch's argument
    # order, so five truncations became agreements; the twelve lazy layers stopped
    # being uncomparable and three landed here; then borch.ts's Conv and
    # ConvTranspose layers took torch's lists and six rows left `shifted` for this
    # bucket.
    #
    # **Six dangerous rows became six harmless ones**, which is what following torch
    # looks like when the far side cannot do everything torch does: a prefix refuses
    # what it does not reach, and one argument too many raises.
    #
    # **Closing a gap against torch opens one between our own two libraries first** —
    # the core can only move toward the outside authority by moving away from
    # borch.ts until borch.ts follows. The pair of axes disagreeing for a while is
    # the mechanism working; a number held still while the two part is not.
    #
    # 24 → 19 on the merge: five loss constructors that had been truncations here
    # became agreements once both sides carried torch's lists. **The two halves of
    # the split moved this number in opposite directions in the same window**, and it
    # is written with both reasons because a running total makes one of them
    # invisible.
    # 19 → 18. `GroupNorm` left for `renamed` above.
    #
    # **The six lazy normalisation layers nearly landed in `unaligned` instead.**
    # borch.ts declared them `(eps, m)` while the eager layers took five, so growing
    # the core's list made them unalignable rather than short — three rows moving
    # into the bucket that means *somebody has to read this*. They were given their
    # target's list, which is the rule the core already derives automatically and
    # borch.ts writes out by hand.
    # 18 → 19. `LazyLinear` arrived when it stopped being uncomparable — see the
    # core-to-torch table.
    # 19 → 24. The thirteen activations plus `Dropout` took torch's `inplace` and
    # borch.ts has not followed; five of them are now longer than their borch.ts
    # counterpart rather than the same length. The safe end of the parting — one
    # argument too many raises rather than landing somewhere.
    # 24 → 27. `AvgPool2d`, `Flatten` and a third grew past borch.ts's list when
    # they took torch's arguments. The safe end of the parting.
    # 27 → 36. **The core-to-torch `shorter` row fell from 45 to 20 and this is
    # where the difference went.** Carrying `device`/`dtype` in order to refuse
    # them lines the core up with the outside authority and moves it away from
    # borch.ts, which has neither and no reason to.
    #
    # That trade was refused once already, on the thirteen normalisation layers,
    # and taken here — the difference is the size: five rows into `unaligned` for
    # thirteen out of `shorter` was not worth it, and nine for twenty-five is.
    # Written down because the two look like the same decision made twice, and
    # they are the same decision made on different numbers.
    "nn": 28,
    "nn.functional": 0,
    # 0 → 1. `Adagrad`: the core grew torch's `maximize` and borch.ts has no place to
    # put it, so borch.ts takes a prefix. **The safe direction** — one argument too
    # many raises there, where the same gap in `shifted` would have meant a value
    # landing on the wrong parameter. `SGD` did get `maximize` on both sides in the
    # same edit and is not here.
    #
    # 1 → 4, and it stayed there. Four optimizers whose borch.ts list is still a
    # prefix of the core's after the core took torch's whole surface — the safe end
    # of the parting, where one argument too many raises rather than landing
    # somewhere. The six that are not prefixes are in `unaligned`.
    #
    # **4 → 0.** borch.ts caught up: `maximize` went onto the `Optimizer` base and
    # reached all eleven at once, and `amsgrad`, `centered`, `momentum` and
    # `decoupled_weight_decay` went into the four algorithms that have them. This is
    # the far side of the mechanism the `nn` note above describes — the core moved
    # toward torch first and this number rose, and it falls again when borch.ts
    # follows. It is the first bucket on this axis to reach zero.
    "optim": 0,
    # 12 → 11. borch.ts's `ReduceLROnPlateau` followed the core into torch's
    # list in the same edit, so the row stopped being a truncation. The
    # cooldown counter joined `stateDict` with it — left out, a resume inside
    # a cooldown starts counting patience again at once, which is a cut up to
    # `cooldown` steps early: small, plausible, invisible against a curve.
    "optim.lr_scheduler": 11,
    "linalg": 2,
    "utils.data": 1,
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
    assert set(rows) == set(SHIFTED) == set(SHORTER) == set(UNALIGNED), (
        f"the namespaces measured changed: {sorted(set(rows) ^ set(SHIFTED))}")

    moved = []
    for space, found in sorted(rows.items()):
        got = {"shifted": 0, "shorter": 0, "unaligned": 0, "renamed": 0}
        for _n, _m, _y, note in found:
            if note in ("dropped", "inserted", "reordered"):
                got["shifted"] += 1
            elif note in ("shorter", "longer"):
                got["shorter"] += 1
            elif note == "unaligned":
                got["unaligned"] += 1
            elif note == "renamed":
                got["renamed"] += 1
        for key, table in (("shifted", SHIFTED), ("shorter", SHORTER),
                           ("unaligned", UNALIGNED), ("renamed", RENAMED)):
            if got[key] != table[space]:
                moved.append(f"{space} {key}: {got[key]} now, {table[space]} written")
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

    The floor is set well under what stands today (614 pairs) so ordinary work does
    not trip it, and well over what top-level functions alone give (30), which is the
    number the broken version produced.
    """
    rows = _rows()
    pairs = sum(len(found) for found in rows.values())
    assert pairs > 300, (
        f"only {pairs} signature pairs were filed, and there were 614.\n"
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
    assert ts_signatures._verdict(["a", "b", "c"], ["a", "c"]) == "dropped"
    assert ts_signatures._verdict(["a", "b"], ["z", "a", "b"]) == "inserted"
    assert ts_signatures._verdict(["a", "b"], ["a", "b"]) == "agree"
    assert ts_signatures._verdict(["a", "b"], ["b", "a"]) == "reordered"
    assert ts_signatures._verdict(["a", "b"], ["a", "z"]) == "renamed"
    # **Renamed and short at once.** The first version tested `shorter` as an exact
    # prefix, so this matched neither and landed among the shifts — six `linalg` rows
    # read as dangerous when the names simply give nothing to align on. A peer
    # reading the rows found it; no count could have.
    assert ts_signatures._verdict(["t", "p", "dim"], ["a"]) == "unaligned"
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
