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
    #
    # **1 → 0, and the bucket is empty on both namespaces.** The last row was `sum`:
    # torch has two overloads under that name — `sum(dtype)` over the whole tensor and
    # `sum(dim, keepdim, dtype)` over one axis — and borch.ts had the first, with the
    # axis form next door as `sumDim`. The comment on `variance` called this *safe by
    # accident*, because a `DType` first parameter meant `x.sum(0)` would not compile.
    # Safe and right are different: the line a reader transcribes from torch has to
    # work. It now tells the two apart the way torch does, a string being a dtype and
    # a number an axis.
    #
    # **What refills this line:** a positional argument landing on a different
    # parameter in the two libraries. Nothing does today.
    "Tensor": 0,
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
    #
    # **0 → 17, and this is the bucket that was empty an hour ago.** The core took
    # torch's deprecated `size_average` and `reduce` — they win over `reduction` in
    # torch (measured), and leaving them out had moved every later argument one or
    # two seats forward, so `F.l1_loss(a, b, 'sum')` gave the sum here and the mean
    # there. Seventeen losses now carry the pair at torch's seats and borch.ts does
    # not, so `new nn.L1Loss('sum')` sets `reduction` over there and `size_average`
    # here.
    #
    # **Raised with the reason rather than held.** The paragraph in `SHORTER` below
    # describes this mechanism and says the danger is only ever a number held still
    # while the sides part — and it is worse here than in `shorter`, because a prefix
    # refuses what it cannot reach and a shift answers. Closing it is borch.ts's
    # side of the same edit; the peer session holding that library has been told
    # which seventeen and where the seats are.
    #
    # **17 → 0, closed the same afternoon.** All seventeen carry `sizeAverage` and
    # `reduce` at torch's seats now, with one `legacyReduction` helper doing the fold
    # so the rule lives in one place on each side rather than seventeen.
    #
    # Two things came out of doing it that the count cannot show:
    #
    #   **`tsc` refused to compile the call sites.** Ten of them across `cases.ts` and
    #   `parity.ts` passed a reduction positionally into what had just become
    #   `sizeAverage`, and the compiler named every one. The core had the identical
    #   mistake in `L1Loss.forward` and Python took it happily — it was found by
    #   comparing values against torch. Same defect, two languages, and only one of
    #   them asked.
    #
    #   **A parity check named `SmoothL1Loss takes reduction first` kept passing after
    #   it stopped being true.** It was written to pin a difference between the two
    #   libraries, the difference is gone, and what it actually asserted was that the
    #   first seat is not `beta`. It now asserts the fold, which nothing but torch's
    #   order could produce.
    "nn": 0,
    # 1 → 2 → 1. `F.embedding_bag` moved `mode` from third to sixth on the core side,
    # and `F.embeddingBag` followed. **`tsc` named all eight call sites the instant it
    # moved** — five golden cases and the layer's own two — because a mode string does
    # not fit `number | null`. The identical move on the Python side was silent.
    # **1 → 0. `shifted` is empty on both axes now.**
    #
    # The row was `scaled_dot_product_attention`: borch.ts had neither
    # `dropoutP` nor `scale`, so `isCausal` sat one seat early and a positional
    # call landed a dropout where a boolean belongs. The binding accepted both
    # missing arguments and dropped them — `scale` **replaces** `1/√dim`, so a
    # caller who set it got the default back and a model whose attention is
    # weighted wrong that trains to somewhere plausible.
    #
    # **Our own case table had made the mistake the row warns about**, and `tsc`
    # caught it: `scaledDotProductAttention(x, x, x, null, true)` put a boolean
    # in the new number slot. The lucky direction — the same two arguments with
    # the same type would have compiled and asked about a dropout of 1.
    "nn.functional": 0,
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
# `optim`'s 7 were one shape worth naming: torch's `Adam(params, lr, betas, eps,
# weight_decay)` against borch.ts's `(params, lr, beta1, beta2, eps, weightDecay)`.
# The pair became two positions, which is a real arity change and not a rename.
#
# **They are 0 now, and reading the sentence above as a decision is what kept them.**
# It describes the divergence and never says anyone chose it; a note further down even
# said `Adam` "would stay here even after borch.ts follows", which is true only while
# the pair stays split. borch.ts takes `betas`, `etas`, `stepSizes` and Adafactor's
# `eps` as pairs now, and the seven land in `agree to the bag` — matching torch
# exactly as far as borch.ts's options object.
#
# **A description left where a reason belongs is read as a reason.** That is the same
# shape as a stale one, one step earlier: nothing here was ever wrong, and nothing
# here ever said the divergence was wanted.
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
    #
    # 6 → 5. `repeat_interleave` left this bucket for `renamed` when borch.ts took
    # `outputSize`, and then left `renamed` too when its first parameter stopped
    # being called `times` — torch and the core both say `repeats`. A rename in the
    # **first** seat is the one that costs most: every keyword call written against
    # torch misses it, and the row reads as a spelling while behaving as an absence.
    # 5 → 4. `isclose` left for `shorter`; see the note there.
    # **4 → 2, and neither of the two that left was a divergence.** `random_` and
    # `uniform_` take a parameter torch spells `from` — measured: `x.uniform_(from=0.,
    # to=1.)` is accepted and `from_=` is a `TypeError`. The core writes `from_`
    # because `from` is a Python keyword and no other spelling is open to it, and
    # borch.ts writes torch's. **A fact about Python's grammar was being counted as
    # two libraries disagreeing**, so it is an attested fold in `RENAMES` rather than
    # a change to either side.
    #
    # The two left are `stft` and `istft`, which really do stop at `nFft`.
    "Tensor": 2,
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
    # 36 → 37. `RNNBase`, and it is a row that **became visible rather than
    # appearing**: borch.ts called the class `Recurrent`, so the name existed on one
    # side only and nothing was compared. Under torch's name the two lists can be
    # put side by side, and they differ — the core takes torch's eleven
    # (`mode, input_size, …, device, dtype`) and borch.ts takes three
    # (`inputSize, hidden, kind`).
    #
    # The same thing the `torch is C` bucket did at scale, one row wide: a name
    # nobody could compare is not a name that agrees.
    #
    # 37 → 35. The three `AdaptiveAvgPool`s stopped taking a `stride` torch does not
    # have, which made their lists a length borch.ts's could be compared against —
    # **a row leaving `unaligned` because the core dropped an argument rather than
    # gaining one**, which is the rarer direction and worth naming.
    #
    # **35 → 23.** The twelve are the pad layers, and they moved in two steps.
    #
    # First the core dropped `value` from the twelve that torch does not give it to
    # (`_PadNd` handed it to every subclass; torch gives it to `ConstantPad*`
    # alone). That took them out of `unaligned` — lists that cannot be lined up —
    # and into `renamed`, because what was left was `padding` here against `p`
    # there. Then borch.ts took torch's spelling and the rows closed entirely.
    #
    # **borch.ts was right about `value` from the start**: its fifteen classes each
    # write their own constructor and only `ConstantPad*` has one. The row was
    # about the core the whole time.
    # **23 → 20.** `Threshold`, `Embedding` and `EmbeddingBag` left for `shorter`
    # once they took torch's spelling.
    #
    # **Two rows in `agree` were wrong and this is where that was found.**
    # `ts_axis._camel` ate a *leading* underscore — `_weight` became `Weight`, which
    # folds to `weight` — while its own docstring explains at length why a *trailing*
    # one is kept. So the core's `_random_samples` folded onto borch.ts's
    # `randomSamples` and `FractionalMaxPool2d/3d` were reported as agreeing when
    # they did not. **The instrument read a mismatch as a match**, which is the one
    # direction a checker must not fail in. Fixed, the two rows appeared as `renamed`
    # and borch.ts now spells it `_randomSamples`.
    #
    # **20 → 11, and four of the nine came out of here holding a defect.** Once the
    # `bias` flag took torch's spelling, `GroupNorm` and the three `LazyBatchNorm`s
    # stopped being unalignable and landed in `shifted` — torch puts `bias` *behind*
    # `device` and `dtype` and makes it keyword-only, so no torch call reaches it by
    # position, while here a fifth positional argument did. Both calls succeed and
    # mean different things.
    #
    # They hold the two seats and refuse now, which is the trade the core made
    # seventeen times for the same two names, and `tsc` named the two case bodies
    # that had been passing the flag into what became `device`. **Clearing a vague
    # classification showed a specific defect beneath it** — the third time today.
    #
    # **What refills this line:** a name on one side with no counterpart on the other
    # at any position.
    #
    # **11 → 10, and the one that left was another shift in hiding.** `RNNCellBase`
    # took `(inputSize, hidden, gates, bias)` where torch takes `(input_size,
    # hidden_size, bias, num_chunks)`, so `new RNNCellBase(4, 8, false)` set the gate
    # count to `false` here and the bias flag in torch — both build a layer and only
    # the shapes differ. Calling the third parameter `gates` is what kept the lists
    # from lining up at all, so the row said nothing; renaming it to `numChunks` made
    # the order visible and the order was the defect. **Twice now a rename has been
    # the thing that exposed a shift**, which is the argument for doing the cheap
    # renames rather than leaving them as cosmetic.
    #
    # **10 → 9, and it was a third one.** `RNNBase` took `(inputSize, hidden, kind)`
    # where torch takes `(mode, input_size, hidden_size, …)` — the same argument, at
    # opposite ends of the list. `MultiheadAttention` left too, on names alone
    # (`embed`/`heads` → `embedDim`/`numHeads`).
    #
    # **7 → 6. `MultiheadAttention` took torch's eleven.** It had two, so seven were
    # missing from the middle and each shifted what followed — `new
    # MultiheadAttention(64, 8, 0.1)`, torch's own way of writing a dropout, reached
    # nothing here. Three work (`dropout`, `bias`, `batchFirst`) and four stop with
    # their own name, which is the trade this file keeps making.
    #
    # **`batchFirst` defaults to `false` and that flips what the class did.** It read
    # `(batch, len, E)` unconditionally — torch's `batch_first=True` — so the default
    # was the option torch does not take, and the TS case body was passing because it
    # leaned on that. It says `true` now, as the Python case always did.
    #
    # **9 → 7. `UpsamplingNearest2d` and `UpsamplingBilinear2d` grew torch's `size`**,
    # which they had no seat for at all — and with it torch's rule that exactly one of
    # `size` and `scale_factor` is given. The old constructor defaulted the scale to 2,
    # so `new UpsamplingNearest2d()` doubled where torch and the core both refuse:
    # **a default that answers where the authority refuses is an argument accepted and
    # dropped, one step earlier.**
    #
    # Everything downstream was written for the old shape and none of it could be
    # caught by a type: the binding's table named `scale_factor` alone, so it would
    # have laid the scale into `size`, and three TS case bodies read `(2)`. Both are
    # numbers — the row `test_binding_arguments` calls *number into a number slot*,
    # where `tsc` is silent by construction.
    # 7 → 6 → 7 → 6 → **0. The bucket is empty.**
    #
    # The last six were `LazyConv*`, and they took three separate fixes that each
    # looked like the whole thing:
    #
    #   the axis read `constructor(...a: ConvArgs)` and took the rest parameter's
    #   name, so the six were one letter wide and unalignable by construction;
    #
    #   with the alias resolved they were still unalignable, because **one shared
    #   five-long tuple stood for two different lists** — torch's plain convolutions
    #   take `(…, padding, dilation, groups, bias, padding_mode)` and the transposed
    #   ones `(…, padding, output_padding, groups, bias, dilation, padding_mode)`,
    #   and `bias` was sitting in the seat torch gives to `dilation`;
    #
    #   and with both lists written out they were *still* unalignable, because
    #   `RENAMES` folded the core's `kernel_size` and `out_channels` into borch.ts's
    #   older `kernel` and `outC` — rewriting the side that was right.
    #
    # **Two wrong verdicts in a row, each hiding the next.** Nobody predicted the
    # second or the third; both sessions guessed this would land in `shorter` as soon
    # as the alias was read.
    "nn": 0,
                #     are the same length again — it left for `renamed` below, which
                #     is a spelling difference rather than a shape one
                # +1, Embedding: a layer borch.ts did not have, so nothing could be
                #     compared. It parts the same two ways `EmbeddingBag` next door
                #     already does — `_weight`/`_freeze` against `weightIn`/`freeze`,
                #     and no `device`/`dtype`, which borch.ts has nowhere.
                #     **A count going up because a name became comparable is not the
                #     same as two sides drifting**, and this bucket cannot tell them
                #     apart on its own; that is what the comment is for.
    # 1 → 2. `F.embedding` took torch's five — `padding_idx`, `max_norm`,
    # `norm_type`, and `scale_grad_by_freq`/`sparse` refused by name — and borch.ts
    # still has `(idx, weight)`. It joins `multi_head_attention_forward`, which has
    # been the lone row here.
    #
    # **The core had this on the layer and not on the function**, so the row is the
    # tail of a fix rather than a new gap: `nn.Embedding` and `F.embedding` were two
    # neighbours disagreeing about the same five arguments, and closing that on the
    # core side opens this until borch.ts follows.
    # 2 → 1. `F.embedding`'s first parameter was `idx`; torch says `input`, and with
    # the name matching the row is a prefix of torch's seven rather than unalignable.
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
# 7 → 0. See the paragraph above `UNALIGNED`.
    "optim": 0,
    # **3 → 0, and the bucket is empty.** `LambdaLR` and `MultiplicativeLR` called
    # torch's `lr_lambda` `fn`, and `CyclicLR` called `step_size_up` `up`; all three
    # were attested folds rather than divergences, and borch.ts spells them torch's
    # way now. The folds went with them — `test_scheduler_table` fails on a fold that
    # fires on nothing, so closing the difference retired the line automatically
    # rather than leaving it to look like work.
    "optim.lr_scheduler": 0,
    # 6 → 7. The core took torch's `linalg` names — about half `A`, about half
    # `input`, plus `tensors`, `LU`, `LD` and `x` — and borch.ts spells them `a`.
    # One row moved here from `renamed` because its two lists stopped being the same
    # length as well as the same words.
    #
    # **This is the debt the core creates by reaching torch**, the mechanism the
    # `nn` note describes, and it falls when borch.ts follows. **What closes this
    # line:** borch.ts taking the same per-function names — they cannot come from a
    # rule, because torch has none here.
    # **7 → 0, and this namespace is empty in every bucket but `shorter`.** Nothing
    # here was a divergence of behaviour: the two libraries answered the same values
    # all along and disagreed about what to call the matrix.
    "linalg": 0,
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
    # **55 → 60, and the five are evidence about the core rather than about borch.ts.**
    # `hypot`, `copysign`, `logaddexp`, `logaddexp2` and `xlogy` became `Tensor`
    # methods over there; borch.ts calls the operand `other`, which is torch's name,
    # and the core calls it `b`.
    #
    # Measured across the whole surface: **42 core functions name it `b` where
    # torch's docstring says `other`**, and their first parameter is `a` where torch
    # says `input` — so `borch.add(x, other=y)` raises where `torch.add(x, other=y)`
    # works. Every one of those 42 is invisible to the core↔torch axis, because
    # torch implements them in C and the docstring rows there are compared by arity
    # alone: same count, different names, no row. **This axis is the only one that
    # can see the class**, and it sees it only where borch.ts has already taken
    # torch's spelling.
    #
    # Left as a rename here rather than fixed in passing: it is 42 signatures and
    # belongs in its own commit, not riding along with a declaration fix.
    # **60 → 26.** The core's parameters are torch's names now.
    #
    # 235 core functions were positionally aligned against torch's docstrings and
    # renamed: `t`/`x` → `input` on the unaries, `a`/`b` → `input`/`other` on the
    # binaries. borch.ts had already taken torch's spelling in most of these places,
    # so the rows were the core being the odd one — and **this axis was the only one
    # that could see them**, because torch implements them in C and the prose rows on
    # the core↔torch axis are compared by arity alone.
    #
    # Only 2 of the 235 named the parameter in their own docstring prose, which is
    # what made a mechanical rename safe; both were read by hand. The rename runs in
    # code spans only — an f-string's `{...}` is code, and treating the whole literal
    # as text left `f"{t.data.shape}"` pointing at the module-level `t` (the
    # transpose), which failed loudly. **A name that had resolved to something
    # plausible would not have.**
    #
    # **26 → 6, and which side to move was measured rather than assumed.** Each row's
    # two spellings were compared against torch's own — trimming the leading `input`
    # and trailing `out` the docstring carries for the function form, which a method
    # does not take. Twenty-one were borch.ts alone, **none were the core alone**, so
    # this bucket was almost entirely one library's to fix and the sentence above,
    # written when the traffic went the other way, no longer describes it.
    #
    # **The docstring is not the keyword.** Four rows would have gone the wrong way on
    # torch's prose: `tensor_split` documents `indices_or_sections` and accepts
    # `sections`, `clamp_max`/`clamp_min` document `value` and accept `max`/`min`, and
    # `true_divide` documents `divisor` and accepts `other`. borch.ts already had the
    # accepted spelling in three of them — renaming to the documented one would have
    # *created* the divergence. Every name here was checked by calling torch with it.
    #
    # **`split` was not a rename at all.** borch.ts's `split(dim, parts)` had torch's
    # name with `chunk`'s meaning and the arguments reversed, while torch's `split`
    # lived next door as `splitSize`. Nothing had ever diverged — the binding routes
    # torch's `split` to `splitSize` with a note saying why, and every golden case
    # goes through the binding — so **a name meaning something else was invisible to
    # the value comparison** and visible only here. It is `splitParts` now.
    #
    # The six left are the core's side: it spells four of them from torch's prose
    # (`value`, `value`, `indices_or_sections`, `divisor`) where torch answers
    # `TypeError`, and `b`/`size` in the other two.
    # **6 → 0, and the other session did the six it had just named.** Each was called
    # on real torch under both spellings before it was touched, which the paragraph
    # above had already shown was the only way to know:
    #
    #     clamp_max     doc: (none)               takes `max`      was `value`
    #     clamp_min     doc: (none)               takes `min`      was `value`
    #     tensor_split  doc: indices_or_sections  takes `sections` was the doc's name
    #     true_divide   doc: value                takes `other`    was `divisor`
    #     remainder     doc: divisor              takes `other`    was `b`
    #     split         see torch.split           takes `split_size` (method)
    #
    # **`true_divide` and `remainder` name each other's wrong argument.** The prose
    # says `value` for the first and `divisor` for the second, and both actually take
    # `other` — so a reader following the docstrings would have crossed them.
    #
    # **This paragraph first said the handover above had been crossed that way, and
    # that was a cause invented to fit.** The handover's `divisor` came from reading
    # a row of axis output — `['divisor'] → ['other']`, the core on the left — and
    # taking the left column for torch's. The docstring hazard is real and measured;
    # attributing somebody else's slip to it was not. **A plausible cause written
    # where the real one belongs** is the shape this repository keeps finding, and it
    # is worse in a note about somebody else's work than in a note about one's own.
    # Corrected on their say-so, which is the only way it could have been.
    #
    # **`split` needed two signatures, because torch has two.** The function takes
    # `split_size_or_sections` and refuses `split_size`; the method takes
    # `split_size` and refuses `split_size_or_sections`. `Tensor.split` is bound
    # separately in `__init__`, joining `softmax` and `log_softmax` — the third pair
    # in a day where a module function bound straight on as a method carried the
    # function's signature into a place torch spells differently.
    #
    # **Zero here does not mean the two libraries name everything alike.** 24 rows
    # were folded into `ts_signatures.RENAMES` long before this, and this bucket
    # counts what is left over that fold.
    "Tensor": 0,
    # 19 → 20. `GroupNorm` arrived from `shorter` when both sides took `affine` and
    # `bias`: same length now, and borch.ts spells the flag `useBias`, as it already
    # does in `LayerNorm`, `Bilinear` and the recurrent layers.
    # 20 → 19. `Hardtanh` left for `unaligned`.
    # 19 → 10. Nine layers left for `unaligned` when the core took torch's
    # `device`/`dtype`: same names as borch.ts up to a point, then two more that
    # borch.ts has no seat for at all.
    # 10 → 9. The `_Rearrange` family: `PixelShuffle`, `PixelUnshuffle` and
    # `ChannelShuffle` shared one base whose parameter was `value`, collapsing three
    # different torch names (`upscale_factor`, `downscale_factor`, `groups`) into
    # one. One of the three now agrees outright and the other two moved.
    #
    # **The tell was a `repr` that would not run.** It printed
    # `PixelShuffle(upscale_factor=2)` while the constructor refused that keyword —
    # so the author knew the three names differ and had applied the knowledge to the
    # printing side alone.
    #
    # **The line above used to say borch.ts already spelled them torch's way, and it
    # did not.** Its constructors took `factor` for both `PixelShuffle` and
    # `PixelUnshuffle`; what carried torch's name was `describe()`, the same
    # printing-side-only knowledge the core had, in the same two classes. Searching
    # the file for `upscale_factor` finds it — in a template string. **A name found
    # in a file is not a name the callable has**, and this is the second time in one
    # day that reading one implementation while writing about the other produced a
    # sentence that was confident and wrong.
    #
    # **9 → 0. The `nn` renames are gone.** `outSize` → `outputSize` on the three
    # adaptive pools, `v` → `value` on the three `ConstantPad`s, `factor` →
    # `upscaleFactor`/`downscaleFactor`, `sizes` → `unflattenedSize`. All of it on
    # borch.ts's side, none of it a behaviour change, and `PixelShuffle`'s `describe`
    # now prints a name its constructor answers to.
    # **0 → 2, and the two were here all along — the instrument was hiding them.**
    # `ts_axis._camel` ate a *leading* underscore, so the core's `_random_samples`
    # folded onto borch.ts's `randomSamples` and `FractionalMaxPool2d` and `3d`
    # reported `agree` against parameter lists they did not match.
    #
    # **A count going up because the instrument stopped lying is not a regression**,
    # and this bucket cannot tell that from two sides drifting apart — the same thing
    # the `nn` note in `unaligned` says about a name becoming comparable, one level
    # worse, because here the rows were being counted as *agreements*.
    #
    # `test_fold_is_lossless.py` now holds the rule the fold has to obey; it caught
    # these two and two more names in the same class (`_weight`, `_freeze`,
    # `_stacklevel`) that happen to have no counterpart to collide with yet.
    #
    # **What closes this line:** borch.ts spelling the parameter `_randomSamples`.
    # The peer session holding that library has it in hand; when it lands this goes
    # back to 0 and the paragraph above stays, because the reason it was ever 0 by
    # accident is the part worth keeping.
    #
    # **2 → 0. It landed.** borch.ts spells it `_randomSamples`, and the retirement
    # condition written above is why this edit took one line instead of a re-reading:
    # the number said what would make it stale, so meeting it was not a judgement
    # call. That is the difference between this line and a parity check earlier today
    # that kept passing after the thing it was named for had gone.
    "nn": 0,    # Bilinear arrived from `unaligned`: borch.ts spells the flag
                #     `useBias`, as it already does in LayerNorm and the recurrent
                #     layers, where the constructor has a `bias` field to not shadow
    # 1 → 2. `scaled_dot_product_attention` arrived from `shifted`: same
    # length now, and `scale` against `scaleOverride` is the whole of what
    # differs.
    #
    # 2 → 1. It took torch's spelling. The function already had a local `scale`
    # holding the tensor, which is why the parameter had the longer name — **the
    # wrong way round**, since the caller reads the parameter and only this function
    # reads the local. The local yielded.
    #
    # The one left is `gumbel_softmax`, and it is not a spelling: borch.ts has no
    # `eps` because torch's does nothing (`deprecated and has no effect`), and a
    # `noise` seat exists so a caller can supply the draw. Both are written down
    # where they are.
    "nn.functional": 1,
    # 7 → 1: the six went back to `unaligned` when the core took torch's whole
    # optimizer surface and borch.ts stayed where it was. See the note there.
    "optim": 1,
    "optim.lr_scheduler": 0,
    # 17 → 18. `linalg.matmul`: the core says `input, other` now and borch.ts still
    # says `a, b`. **The row appeared because the core moved toward torch**, not
    # because borch.ts moved away — the direction a rising count usually means, and
    # the reason it is written here rather than only counted.
    #
    # 18 → 14. The core took torch's `linalg` names outright — thirty-four of them —
    # and four rows stopped being a rename because the two lists parted in length as
    # well, moving to `shorter` and `unaligned` next door. **Four left this bucket by
    # getting worse-looking**, which is the direction that needs saying: `renamed`
    # means *a spelling*, and the buckets it feeds mean *a missing tail* and *cannot
    # be compared*. The rows did not improve; they became legible.
    #
    # **What closes the remaining fourteen:** borch.ts taking the same per-function
    # names. They cannot come from a rule — torch is about half `A` and half `input`
    # inside `linalg`, with `tensors`, `LU`, `LD` and `x` besides — so the table has
    # to be measured there too, by calling torch with each keyword.
    # **14 → 0. torch's names are a table, not a rule.** Nine of its `linalg`
    # functions call the first argument `A` and fifteen call it `input`, with
    # `det(A)` and `cholesky(input)` sitting next to each other — so any sentence
    # short enough to be a rule is wrong about half of them. The core measured each
    # one by calling torch with the keyword; borch.ts took the core's spellings,
    # which are that measurement.
    #
    # **The summary that started this was mine and it was wrong** — "torch writes `A`
    # in fifteen of them", read off the first docstring line of a handful and
    # generalised. The peer's probe found 9 and 15. A summary is a claim about every
    # row, and this one had been checked against six.
    "linalg": 0,
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
    #
    # **18 → 15.** Eight rows came in and eleven went out in one pass, and both
    # halves are the same work: the core↔torch axis learned to read torch's C
    # signatures, so the core grew arguments it had been missing, and each of those
    # is a row here until borch.ts has it too.
    #
    # In: seven `memoryFormat` seats (`float`, `long`, `bool`, `cfloat`, `int`,
    # `double`, `cpu`) and `div_`'s `roundingMode` — that last one because **`div`
    # took it and `div_` did not**, so the two spellings of one operation had
    # different arguments and the in-place one quietly did true division where torch
    # floors. Out: `sort`/`argsort` took `stable`, `topk` took `largest` and
    # `sorted`, `clone` took `memoryFormat`, `gather` took `sparseGrad`, `round`
    # took `decimals`, `quantile` and `nanquantile` took `dim`/`keepdim`/
    # `interpolation`, and `svd` was split from `linalgSvd`.
    # 15 → 16, and `unaligned` went 5 → 4: **the same row moving to a truer
    # bucket.** `isclose` had the core saying `b` where borch.ts said `other`, so
    # the lists could not be aligned at all; with the core on torch's names they
    # align, and what is left is borch.ts being short of `equal_nan`. An absence is
    # a more useful thing to be told than "these cannot be compared".
    #
    # 16 → 18. `softmax` and `log_softmax` took torch's `dtype`, which casts **before**
    # the softmax — an integer input with `dtype=float32` gives the real answer, not a
    # cast of an integer one — and borch.ts has not followed. The safe end of the
    # parting.
    #
    # These two also sit in `TORCH_DOC_IS_WRONG` on the other axis: torch's docstring
    # says `softmax(dim)` while the method takes `dtype` too. **One pair of rows, and
    # the two axes disagree about which side is odd** — against torch's prose the core
    # reads long, against borch.ts it reads right and borch.ts short. Only a call
    # settles it, and the call says torch takes it.
    #
    # 18 → 20. `random_` and `uniform_` arrived from `unaligned` once `from_` was
    # folded onto torch's `from`; what they are short of is `generator`.
    "Tensor": 20,
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
    # 28 → 30. `Upsample` and the LP pools grew past borch.ts's lists when they took
    # torch's last arguments. The safe end of the parting: one argument too many
    # raises rather than landing somewhere.
    # 30 → 33. The three transformer classes, and each differs by exactly
    # `device` and `dtype` — **which is what every borch.ts layer does.** There is
    # one device and one dtype over there, and the core carries the pair only to
    # refuse it, so giving these three seats nobody else has would make them the
    # odd ones rather than the correct ones.
    #
    # Measured before writing this: of the 33, twelve others also lack the pair
    # (alongside other differences), and none of the 33 lacks *only* it. These
    # three are the first to be that clean, which is why the row is worth a
    # sentence rather than a nudge.
    #
    # 33 → 35. `AvgPool1d` and `AvgPool3d`. They already declared torch's arguments
    # and refused them inside; what changed is that the refusals became real, and
    # the two rows arrive here because the *adaptive* pools next door gave up a
    # `stride` and stopped being uncomparable — the same edit moved two rows out of
    # `unaligned` and two into this bucket, in opposite directions, and a running
    # total would have shown neither.
    #
    # **35 → 32.** `AvgPool1d`, `AvgPool3d` and `LPPool1d` took the rest of torch's
    # list, and taking it meant the WGSL kernel first: `PoolNDShape` had extents, a
    # kernel and a stride and nowhere to put a padding, a `countIncludePad` or an
    # overriding divisor, so there was nothing underneath for a constructor argument
    # to be forwarded to. The average's divisor stopped being the kernel volume and
    # became what the window actually covers, which is what all three of those
    # arguments are asking about.
    #
    # **`AvgPool2d` is still here and that is deliberate.** It runs on a separate
    # two-dimensional kernel rather than `poolND`, so the same six arguments are a
    # second implementation and not a second constructor line. Left as it is until
    # the two paths are one.
    #
    # **32 → 35, and every one of the three arrived from `unaligned`** — the bucket
    # where the lists cannot be lined up at all, so nothing beneath is reported.
    # `Threshold` spelled its first argument `t`; `Embedding` and `EmbeddingBag`
    # abbreviated `numEmbeddings`/`embeddingDim` to `num`/`dim`. Taking torch's
    # spelling makes each a prefix of torch's list, which is the safe reading: what
    # is accepted means what torch means, and one argument too many raises.
    #
    # A number going *up* here is the work, not a regression. `unaligned` hides; this
    # bucket says exactly what is missing (`device`/`dtype`, mostly).
    #
    # **35 → 40.** Ten layers spelled torch's `bias` flag `useBias` or `hasBias`, and
    # the comment beside them said the reason was not shadowing a `bias` field.
    # **In TypeScript there is nothing to shadow** — `this.bias` is unambiguous, and
    # renaming all thirty-one occurrences compiles unchanged. A reason that holds in
    # another language is not a reason here.
    #
    # **What retires this line:** borch.ts growing `device` and `dtype` seats, which
    # is most of what these forty rows are short of.
    #
    # 40 → 41. `RNNCellBase` arrived from `unaligned` — see the note there.
    # 41 → 42. `RNNBase` likewise — **and it came back once.** Moving `mode` to the
    # front left it still called `kind`, which is enough to stop the lists lining up,
    # so the row returned to `unaligned` and the note here said it had left. Order and
    # name are two fixes; doing one is not doing the row.
    # 42 → 43. `MultiheadAttention`, short of `device`/`dtype` and nothing else now.
    # `RNNBase` leaving a second time adds nothing: it was already counted here, went
    # back to `unaligned` on the name, and returned.
    #
    # **43 → 49, and `unaligned` is empty.** The six `LazyConv*` closed, and the six
    # rows they had to pass through to get here are the finding: the axis could not
    # read a rest parameter, then read one and found `bias` in torch's `dilation`
    # seat, then still could not line them up because `RENAMES` was rewriting the
    # *core's* `kernel_size` into borch.ts's older `kernel`. Three layers, one row.
    #
    # **What retires this line:** borch.ts growing `device` and `dtype` seats, which
    # is what nearly all forty-nine are short of.
    "nn": 49,
    # 0 → 1. `F.embedding` arrived from `unaligned`, short of torch's five
    # table-side arguments — `padding_idx`, `max_norm` and the rest, which the layer
    # next door does have.
    "nn.functional": 1,
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
    # 11 → 12. `ChainedScheduler` took torch's `optimizer` and borch.ts has not.
    # 12 → 15. The three that left `unaligned`; what they are short of is
    # `last_epoch`, which every scheduler here lacks.
    "optim.lr_scheduler": 15,
    # 2 → 1. One of the two left for `unaligned` when the core took torch's `linalg`
    # names — a prefix stopped being a prefix once the words changed. See the note
    # in `RENAMED`.
    # 1 → 8. The seven that left `unaligned` and `renamed` — what they are short of
    # is `dim`, `keepdim` and `dtype` on the norms, and `pivot`/`left`/`adjoint` on
    # the factorisations.
    "linalg": 8,
    # **1 → 4, and all four are the same `generator`.** `random_split` was the one;
    # `RandomSampler`, `SubsetRandomSampler` and `WeightedRandomSampler` join it now
    # that they exist at all.
    #
    # It is one decision, made once and written at the top of `borch-ts/src/data.ts`:
    # torch gives a DataLoader its own generator, and borch.ts runs **one host
    # stream**, so a single `manualSeed` rewinds layer initialisation, dropout, the
    # tensor factories and batch order together. "The choice was to not add another
    # door." Three more names inheriting a decision is not three more decisions —
    # and the number is here rather than the reason so that the day the door opens,
    # all four move at once and one of them not moving is visible.
    "utils.data": 4,
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
