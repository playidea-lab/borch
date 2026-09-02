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

- **Not that most of the surface was measured the same way.** 59 rows are still
  `torch is C`, down from 571, and 231 more are read from **torch's prose** rather
  than from torch — a `prose` column that is deliberately never added to `agree`.
  Those are compared by how many arguments each side takes, not by which, because a
  docstring's order is not reliable. They are leads that have been checked, not
  measurements.
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
    # **0 → 5, and none of the five is new.** They were behind a reader that could not
    # see them: torch writes a C function's signature in the first line of its
    # docstring, this axis reads it from there, and the reader matched the **bare**
    # name. `linalg`'s docstrings write `linalg.solve(A, B, *, left=True, out=None)`,
    # so every one of its thirty-six C functions fell into *torch has it and nothing
    # could read it* — the absorbing bucket the docstring below this table warns about.
    # `linalg` read `agree 3 · torch is C 36` and reads `agree 14 · torch is C 1` now.
    #
    # The five, and each is an argument torch has in the middle:
    #   · `solve`      — torch `(A, B, left, out)`; `left` is absent, so `out` stands
    #                    in its seat and `linalg.solve(A, B, False)` sets `out=False`
    #                    here and `left=False` there
    #   · `solve_ex`   — the same `left`
    #   · `svd`        — torch `(A, full_matrices, driver, out)`; `driver` is absent
    #   · `svdvals`    — the same `driver`
    #   · `norm`       — torch orders the tail `out, dtype` and this orders it
    #                    `dtype, out`
    #
    # Written out rather than left as a number: the count says *five somethings*, and
    # what the next person needs is which five.
    #
    # **5 → 1, and the one left is `norm`.** `solve` and `solve_ex` take `left` now —
    # it is one transpose, `X A = B` being `Aᵀ Xᵀ = Bᵀ` — and `svd` and `svdvals` take
    # `driver` in order to refuse it, which is what torch does off CUDA. `norm`'s tail
    # stays `dtype, out` against torch's `out, dtype` because `out` is appended by
    # `_accepts_out` rather than declared, and both are keyword-only on torch, so the
    # order is something this axis reads and no caller can trip over.
    "linalg": 1,
    "utils.data": 0,
}

# The names cannot be aligned against each other, so the row says so rather than
# guessing. `nn.functional`'s 30 are mostly the core naming its first argument `x`
# where torch says `input`.
UNALIGNED = {
    # **0 → 5, and 15 of 512 judged → 386 of 512.** Not a regression: the axis
    # started reading a namespace it had been declaring unreadable.
    #
    # `inspect` gives no signature for a C implementation and `torch.Tensor` is
    # almost entirely C, so 571 names — more than every other bucket on this axis
    # together, and more than the 445 it now calls `agree` — sat in `torch is C`. A
    # bucket that size does not read as *nothing was asked about most of the
    # surface*. It reads as a footnote under a green run.
    #
    # **It was not empty, and that was found by accident.** A probe built to ask
    # which input ranks each function accepts turned up `where`'s one-argument form
    # and `nonzero(as_tuple=)`, both absent, both living here — while this axis
    # reported 0 keyword-only absences. Two found by something that was not looking
    # is not evidence of two.
    #
    # torch writes the argument list in the first line of the docstring, so
    # `signature_read.from_docstring` reads it there. That is torch's prose rather
    # than torch's behaviour and the rows are leads rather than measurements — the
    # docstring for that function says so at length. Still much better than 571
    # names nobody asked about.
    #
    # The five were real and all five are carried across, so this is 0 and the
    # whole `Tensor` namespace is empty in every measured bucket. What they were:
    #
    #   copy_   short of `non_blocking`, and its first name was `other` for torch's
    #           `src` — a rename in the first seat, which a keyword call misses.
    #   div_    short of `rounding_mode`. **`div` had it and `div_` did not**, so
    #           the two spellings of one operation took different arguments and the
    #           in-place one quietly did true division where torch floors. `round`
    #           and `round_` were the same pair, found in the same run.
    #   type    took `dt` for torch's `dtype`. Spelling it torch's way shadows this
    #           module's `dtype` class inside the function — `triangular_solve` has
    #           the same collision on `transpose` and writes an alias for it.
    #   svd     **was `torch.linalg.svd` under `torch.svd`'s name.** Three things
    #           part between them and all three were wrong: the default is reduced
    #           rather than full, the third field is `V` rather than `Vh`, and
    #           `some` is the *opposite* of `full_matrices`. `borch.svd(x)` and
    #           `torch.svd(x)` returned **different shapes from the same call**, and
    #           the overlapping block agreed, so anything reading `S` alone or
    #           `U[:, :k]` saw nothing wrong. `linalg_svd` now holds the other one.
    #   true_divide  carried a `rounding_mode` **torch does not have** — the only
    #           row of the five pointing the other way. It was an alias for `div`,
    #           and `div` is the one that takes the argument; torch answers
    #           `received an invalid combination of arguments`, because true
    #           division is exactly the thing a rounding mode undoes. Accepted here,
    #           `true_divide(x, y, rounding_mode="floor")` returned a floored value
    #           that torch will not produce at all.
    #
    # **The last one is also where the arity rule is blind, and it took a second
    # look to see that.** Once the docstring reading landed, `true_divide` moved to
    # `prose agree` — torch documents `(dividend, divisor, out)` and the core takes
    # `(b, rounding_mode)`, which is two against two, so counting arguments called it
    # settled. It was not: the counts match and the names do not, and the row that
    # said so was gone.
    #
    # It survived because the note above this line said all five were carried across
    # while four of them were — **a claim of completeness scoped to what its author
    # had open.** Checking the fifth against a call rather than against the axis is
    # what found it. The rule stays as it is: arity is what prose can support, and
    # the alternative is 74 rows of noise. What is written down instead is that a
    # rename-shaped divergence at equal arity does not appear here at all.
    "Tensor": 0,
    # **8 → 5.** `TransformerEncoderLayer`, `TransformerDecoderLayer` and
    # `Transformer` took torch's parameter order, which is not the order they had.
    #
    # torch is `(…, activation, layer_norm_eps, batch_first, norm_first, bias)` and
    # the sixth seat here was `batch_first`, so
    # `TransformerEncoderLayer(4, 2, 8, 0.1, "relu", True)` put `True` into torch's
    # epsilon and the layer normalised with **eps = 1**. Nothing raises; the shapes
    # are right and the loss goes down. `Transformer` was two arguments out, missing
    # `custom_encoder` and `custom_decoder` from the middle of its list.
    #
    # **They were sitting in this bucket the whole time**, and `unaligned` says
    # *these lists cannot be lined up* and then says nothing else. Third time today
    # that clearing a vague classification showed a specific defect underneath —
    # `F.normalize`'s missing `out=` and `isclose`'s missing `equal_nan` were the
    # others. A row in the vaguest bucket can be hiding a sharp one.
    #
    # Found while writing a golden case for the layer: the case was frozen against
    # real torch, and the core disagreed by 1.6e-01. Every intermediate step then
    # matched to 1e-7 — so it was not the arithmetic, it was **my own call putting
    # `True` where torch keeps an epsilon**, which is exactly what the row warns
    # about and exactly what nobody had written a call to find out.
    #
    # **5 → 0, and the last five were the average-pooling family.** `AvgPool1d` and
    # `AvgPool3d` read `(size, stride)` where torch reads `(kernel_size, stride,
    # padding, ceil_mode, count_include_pad[, divisor_override])`, and the three
    # `AdaptiveAvgPool`s read `(size, stride)` where torch reads `(output_size)` —
    # **a `stride` torch does not have at all**, accepted here and doing nothing.
    #
    # One base class was serving both families, and the shape that fitted both was
    # the intersection of the two, which is neither. Splitting it is what let the
    # fixed poolers grow torch's four and the adaptive ones drop the argument that
    # was never theirs.
    "nn": 0,
    # **27 → 16.** Eleven of these were activations whose only difference from torch
    # was the missing `inplace` at the end — `relu(t)` against `relu(input, inplace)`
    # cannot align, so they sat here rather than in `shorter`. Giving them the
    # argument left the first parameter's name (`t` against `input`) as the whole of
    # what remains, which is a spelling and not an absence.
    #
    # 16 → 14. `dropout` and `threshold`. The four other dropouts were already here
    # for a different reason: they **took `inplace` and discarded it**, so they
    # aligned with torch on paper while doing none of it. Alignment is a fact about
    # names and this file cannot see that — `test_inert_arguments.py` can.
    #
    # 14 → 11. The three `lp_pool*` functions took `ceil_mode`. **Two sessions cut
    # this row in the same window for unrelated reasons** and the rebase put both
    # notes here; the number is measured after both, not either.
    # 11 → 10. `normalize`. It had been filed here because the core called its
    # first parameter `x` and torch calls it `input` — the lists could not be
    # aligned, so nothing else about them was reported. With the names matched they
    # aligned, and **what was actually missing showed through underneath**: torch
    # takes `out=` and this did not. **A row in the vaguest bucket can be hiding a
    # specific one**, and clearing the vague reason is what lets the specific one be
    # seen.
    #
    # It could not go through `_accepts_out`: that wrapper is driven by
    # `_TAKES_OUT`, which lists names on `torch` itself, and `normalize` lives only
    # under `torch.nn.functional` — adding it there asked `test_out_names.py` for
    # `torch.normalize`, which does not exist. Written into the function instead.
    #
    # **10 → 6, and the four that left took two defects out with them.** Eleven
    # losses called their first parameter `pred`, `logits`, `p` or `log_probs` where
    # torch says `input`; renamed, the lists could be lined up for the first time and
    # two rows fell straight out of here into buckets that name something sharp:
    #
    #   `smooth_l1_loss`  reordered — the third and fourth arguments were exchanged.
    #                     `nn.SmoothL1Loss` had this same swap and was corrected long
    #                     ago; the *function* kept it, which is what happens when a
    #                     family is fixed one member at a time.
    #   `nll_loss`        inserted  — `weight` and `ignore_index` were missing, while
    #                     `nn.NLLLoss` two thousand lines up had both. The function
    #                     was a second, poorer implementation; it routes to the layer
    #                     now, the way `cross_entropy` already did.
    #
    # **Renaming a parameter is cosmetic and what it uncovers is not.** That is the
    # third time in this file that clearing a vague classification showed a specific
    # defect underneath — `F.normalize`'s missing `out=` and `isclose`'s `equal_nan`
    # were the others — and the first time the vague bucket was cleared *on purpose*
    # to find out what it was hiding.
    #
    # **6 → 0, and the last six were four functions that had the work next door.**
    #
    #   `embedding`      took two of torch's seven while `nn.Embedding` carried all
    #                    of them — `padding_idx` and `max_norm` implemented,
    #                    `scale_grad_by_freq` and `sparse` refused by name. The
    #                    function is the primitive, so the body moved *into* it and
    #                    the layer calls it; `nll_loss` earlier today went the other
    #                    way, because there the layer was the primitive.
    #   `softmax`,       defaulted `dim` to `-1` while `nn.Softmax` had the real
    #   `log_softmax`,   rule — 0 or 1 by rank, and torch warns. **`-1` is right at
    #   `softmin`        rank 2 and wrong at rank 3**, so it was invisible wherever
    #                    the golden asked. `_default_softmax_dim` was written out in
    #                    `_nn.py` and reachable from one side of the file only.
    #   `instance_norm`  was missing torch's first three seats, so
    #                    `F.instance_norm(x, None, None, w)` put the weight in
    #                    `running_mean` here and in `weight` there.
    #   `interpolate`    was missing `antialias`, which changes the values when
    #                    shrinking. Carried and refused: accepted and ignored, it
    #                    returns the aliased answer under the filtered one's name.
    #
    # **Three of the four were the same finding**: an argument absent from the
    # function and present on the layer beside it, or a rule written for the layers
    # and unreachable from the functions. A peer session put the shape into words
    # while this was being worked — *read a gap list by name and you do the repair
    # the name suggests* — and it was true four times out of four.
    "nn.functional": 0,
    "optim": 0,
    "optim.lr_scheduler": 0,
    # 0 → 4, from the same docstring reading as `Tensor` above. All four are
    # `householder_product`, `lu`, `matrix_power` and `qr` naming their input `A`
    # where the core says `t`, plus torch's `out` — a spelling and a tail.
    #
    # **4 → 2. The spelling half went; the `out` half is what is left.**
    # Thirty-four `linalg` functions took the name torch answers to, and the two
    # remaining rows — `householder_product` and `matrix_power` — differ only by
    # torch's `out=`, which this library does not carry here.
    #
    # **torch is not consistent inside `linalg`.** Roughly half take `A` and half
    # take `input`: `det(A)` and `cholesky(input)` sit side by side, `multi_dot`
    # takes `tensors`, `lu_solve` takes `LU`, `ldl_solve` takes `LD`, `vecdot` takes
    # `x`. Every one was measured by calling torch with the keyword, because a rule
    # anybody can state in a sentence gets about half of them wrong — an earlier
    # summary of this namespace said "fifteen use `A`" and the real count is nine.
    #
    # **And three names are spelled differently in the two namespaces.**
    # `torch.det(input=…)` is taken and `torch.det(A=…)` is not, while `linalg.det`
    # is the reverse; the same for `qr` and `slogdet`. One function cannot answer to
    # both, so the top-level definitions keep `input` and `linalg` has three
    # wrappers. Third time in a day — `Tensor.split`/`torch.split` and
    # `Tensor.softmax`/`F.softmax` were the others.
    #
    # **2 → 0, and the sentence above is what kept it at 2.** *"…differ only by
    # torch's `out=`, which this library does not carry here"* is a picture, not a
    # decision: it says what the difference is and nothing about whether the
    # difference was chosen. A picture cannot go stale, so it never asks to be
    # re-read, and this one was passed over three times.
    #
    # Asked once, it turned out to be **half true.** `_accepts_out` had existed all
    # along and was applied to `globals()`, and `linalg`'s members are bound from
    # `_ops` directly, so the wrapper never reached them. A mechanical gap wearing
    # a reason's clothes. `borch/__init__.py` now carries `_LINALG_TAKES_OUT` and
    # `tests/test_out_names.py` holds it against torch.
    #
    # The second half was that the wrapper declared `__wrapped__` and not
    # `__signature__` — so `out=` worked and no reader could see it. Both closed,
    # and **zero here is the end state**, not a number waiting to be lowered again.
    #
    # **0 → 5, and it is the same opening as the `SHIFTED` row above** — thirty-six
    # rows left the unreadable bucket and five of them land here. The sentence above
    # said zero was the end state; it was the end state *of what could be seen*, which
    # is the shape this whole file is about.
    #
    # The five, each an argument torch has that the core does not:
    #   · `matrix_norm`  — `dtype`, and `out` sits in its seat
    #   · `vector_norm`  — the same `dtype`
    #   · `matrix_rank`  — torch `(A, atol, rtol, hermitian, out)` against a single
    #                      `tol`; torch deprecated `tol` for the pair and this kept it
    #   · `pinv`         — the same three against a single `rcond`
    #   · `lstsq`        — the other direction: the core carries an `out` torch's
    #                      documented list does not have
    #
    # **5 → 3.** `matrix_norm` and `vector_norm` take `dtype` now — the seat
    # `linalg.norm` next door had all along while the two it dispatches to did not.
    # Here it can only be `float32`; asking for `float64` stops with this library's
    # standing sentence rather than accumulating in something else.
    #
    # The three left are real work rather than a seat: `matrix_rank` and `pinv` want
    # torch's `atol`/`rtol`/`hermitian` where the core has one `tol`/`rcond` (torch
    # deprecated the single one), and `lstsq`'s is the other direction — the core has
    # an `out` torch's *documented* list lacks while its runtime accepts one, which is
    # this reader's standing caveat rather than a defect.
    "linalg": 3,
    # 1 → 0. `DataLoader` had seven of torch's seventeen names **and two of them
    # in the wrong seats** — `collate_fn` is torch's seventh and `drop_last` its
    # ninth, where here they were sixth and seventh. `DataLoader(ds, 4, False,
    # None, 0, True)` set `drop_last` on this side and handed `True` to
    # `collate_fn` on torch's, which then tries to call it.
    "utils.data": 0,
}

# The core takes a prefix of torch's arguments — a **feature torch has and we do
# not**, which is `torch_gap.py`'s kind of finding rather than a silent shift.
SHORTER = {
    # 2 → 1. `dim_order` took `ambiguity_check`, which refuses where the answer is
    # not unique: an axis of length 1 has no position of its own, so `(1, 3)` has two
    # orders equally true and the function has to pick one. Without the flag it picks
    # silently — that is the default on both sides, and the flag is the only way to
    # find out that a pick was made.
    # 1 → 2. `stft` and `backward` became comparable when the method binders started
    # setting `__wrapped__`; both are short tails against torch (`align_to_window`,
    # and `create_graph`/`inputs`).
    # 2 → 0. `backward` took `create_graph` and `inputs`, `stft` took
    # `align_to_window`. The `Tensor` half of this bucket is empty.
    "Tensor": 0,
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
    # 57 → 58. `LazyLinear` joined when it stopped being uncomparable: it is written
    # by hand rather than generated, so it alone kept `_Lazy.__init__`'s
    # `(*args, **kw)` and was the one lazy layer this axis could not read. It is
    # short of torch's `device`/`dtype`, like every other layer in this bucket.
    #
    # **58 → 45, and two sessions lowered it, which is why both stories are here.**
    # `inplace` went in — `nn.ReLU(inplace=True)` is a line every torch model writes
    # and it raised, a `TypeError` about an argument count on the most common layer
    # there is. It was the largest single absence left in this namespace and the
    # second-largest overall after `device`/`dtype`, and unlike those two it is not a
    # question the browser answers differently.
    #
    # **The two sessions disagreed about six of them and the disagreement was real.**
    # One routed the layers through the `_` functions that already existed and refused
    # `SELU`, `SiLU`, `Mish`, `ReLU6`, `Hardsigmoid`, `Hardswish` and `Dropout` by
    # name, on the ground that an activation promising to reuse the buffer and quietly
    # making a new one is a promise about *memory* that no value comparison catches.
    # That ground was right; its premise stopped holding in the same window, because
    # the other session gave `_ops` a write-back for every one of them through the
    # same `Tensor._inplace` the underscore names use. Measured against torch: input
    # moved, values equal, the same object returned, on all eleven. So the refusals
    # went and the seven joined the rest — a refusal kept past its cause is the shape
    # this repository keeps finding, and it would have been ours.
    #
    # `Hardtanh` took torch's deprecated `min_value`/`max_value` with it, so a
    # positional call reaching that far lands where torch lands. `nn.Dropout` gave up
    # **its own second copy of the dropout formula** — a copy that divided by zero at
    # `p=1` where the function next door branched — because taking `inplace` meant
    # calling the function. `RReLU` had been taking the flag, storing it, printing it
    # in its `repr` and never passing it on.
    #
    # **The number did not move when the seven refusals became implementations.**
    # This axis reads signatures, and a refusal has the same signature as an
    # implementation — which is exactly why the argument had to be measured against
    # torch's *behaviour* somewhere else. `test_inert_arguments.py` is that somewhere.
    #
    # 45 → 18. **Seventeen layers took torch's `device` and `dtype`**, carried and
    # refused, which is what `_no_device_dtype` was written for and what eight layers
    # already did. `AvgPool2d` took its four real ones at the same time and `Flatten`
    # took `end_dim`.
    #
    # Carrying an argument in order to refuse it is not a fiction: the *seat* is
    # torch's, so a positional call that reaches it lands where torch lands, and the
    # refusal names which argument it was. Left out, the same call lands on nothing
    # and the two lists part at every layer that has them.
    # 18 → 12. `Upsample`, the three LP pools, `TransformerEncoder` and
    # `RNNBase` all took what they were short of.
    #
    # **It did not move when `AvgPool1d` and `AvgPool3d` stopped refusing `padding`,
    # `count_include_pad` and `divisor_override`**, and that is worth a line. Both
    # already *declared* those names and raised inside; this bucket counts declared
    # names, so a refusal and an implementation look identical to it. The two rows
    # were `agree` before the change and `agree` after, while the behaviour went from
    # "stops" to "matches torch over 156 configurations".
    #
    # An axis that cannot see the difference between declaring an argument and
    # honouring it is not broken — it is measuring names, and it says so. It is the
    # reason `test_inert_arguments.py` exists next to it, and the reason a green run
    # here is not evidence that anything works.
    #
    # **12 → 0. This bucket is empty: the core takes no `nn` argument torch does
    # not.** All twelve were one argument, `value`, on one shared base.
    #
    # `_PadNd.__init__(padding, value)` gave it to every subclass, and torch gives
    # it to `ConstantPad*` alone — `ZeroPad2d(1, 9.0)` is a `TypeError` there. Here
    # it was accepted, and what happened next depended on the mode:
    #
    #     ZeroPad2d(1, value=9)         filled with 9    ← a pad named Zero
    #     ReflectionPad2d(1, value=9)   filled with 4    ← accepted, discarded
    #     ReplicationPad2d(1, value=9)  filled with 0    ← accepted, discarded
    #
    # Two kinds of wrong at once: three answer with a number their own name rules
    # out, and nine take an argument and drop it.
    #
    # **borch.ts had it right from the start** — its fifteen classes each write
    # their own constructor and only `ConstantPad*` has a `v`. So this axis was
    # reporting twelve rows about the core while the other implementation was
    # already correct, which is the direction this axis is *for* and the one it is
    # easiest to read past when the column is called `우리가 더 받는다`.
    #
    # The binding had the same defect in its own shape: `make(padding, value=0.0)`
    # for all fifteen, forwarding it only for `ConstantPad*`. Nothing diverged
    # there, because borch.ts has no seat to receive it — an argument accepted and
    # discarded looks exactly like one honoured until somebody checks the answer.
    "nn": 0,
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
    # 10 → 0. **Every optimizer agrees with torch exactly**, which is the first time
    # a namespace in this table has emptied. `amsgrad`, `centered`, `momentum` and
    # `decoupled_weight_decay` were the four that changed values; the rest are
    # torch's execution switches, accepted where they cannot change an answer and
    # refused where torch refuses them.
    "optim": 0,
    # 1 → 0. `ChainedScheduler` took torch's `optimizer` — optional there too, and
    # refused here when it is not the one the schedulers are stepping, since
    # `get_last_lr` reads the rates off it.
    #
    # **This namespace and `optim` are both empty now.** Two of the seven.
    "optim.lr_scheduler": 0,
    # 0 → 2. `lu` and `qr` became readable when they took torch's names, and what
    # showed underneath is that torch declares an `out=` on both and this does not.
    # **A row arriving in `shorter` from `unaligned` is the axis working**: the vague
    # bucket said *these lists cannot be lined up* and this one says *torch has a
    # tail we do not*, which is a thing somebody can act on.
    #
    # **2 → 0.** Somebody acted on it: `linalg` takes `out=` and declares it. See
    # the `unaligned` note above for why the sentence sat unacted-on for three
    # readings — it described the gap and never said the gap was chosen.
    "linalg": 0,
    "utils.data": 0,
}

# **torch is implemented in C here and neither `inspect` nor the docstring can read
# it.** Pinned as a total rather than per namespace, because the number is a property
# of torch's build and not of our work: it moves when torch moves. It is pinned at all
# so that a change in how the measurement reads signatures cannot quietly turn
# unreadable rows into comparisons nobody checked.
#
# **571 → 59, and this pin is what asked the question that decided how.** Its own
# failure message says: *lower means something started reading C signatures — check
# what authority it is using before believing the rows it produced*. The authority is
# torch's docstring, it is weaker than `inspect`'s, and the weakness is carried in the
# result rather than in a footnote — deferred rows are compared by arity alone and
# counted in a separate `prose` column that is never folded into `agree`.
# 59 → 58, and **it moved for a reason on the other axis.** `compare()` skips
# `Tensor` rows that `ts_axis.refused()` names, and that function was missing the
# `_sparse_only` factory — so three sparse stubs were being counted as gaps there
# and filed here. One of them is C in torch, and it left this bucket when the
# classification was fixed. Not a torch upgrade; a correction one file over.
#
# **58 → 23, and the message above asked the right question again.** *Lower means
# something started reading C signatures — check what authority it is using.* The
# authority is the same as last time and no weaker: torch's own docstring, first line,
# which is where torch writes a C function's argument list. What changed is that the
# reader matched the **bare** name and torch writes `linalg.solve(A, B, *, left=True,
# out=None)` with its namespace in front. Thirty-five `linalg` rows were unreadable for
# a prefix.
#
# `signature_read._opens` accepts a dotted path now, anchored — the line has to *be*
# the name behind an optional path, not merely contain it, because a substring test
# would read `norm(` out of a sentence about `matrix_norm` and a wrong name out of
# prose becomes a row that looks like a finding.
#
# Ten divergences came out of those thirty-five, five `shifted` and five `unaligned`,
# each named on its row in the two tables above.
UNREADABLE_IN_TORCH = 23

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
    # 9 → 15 of 512. **Six rows stopped being uncomparable.** `_as_method` and
    # `_bind_from_module` write `(self, *args, **kw)` and forward everything, so
    # `inspect.signature` saw the wrapper and the axis filed them under `variadic`,
    # which means *cannot be compared at all* — for names whose module form is fully
    # spelled out three lines away. Setting `__wrapped__` is the whole fix.
    #
    # What they turned out to be is worth recording as the negative result it is:
    # four renames (`b` against `other`, `size` against `split_size`) and two short
    # tails. **No silent mis-seat.** The last time this repair was made — on
    # `_accepts_out` — it turned up two functions answering for an axis that does not
    # exist, so the outcome was not predictable from the shape of the fix.
    # **15 of 512 → 386 of 512.** The axis was judging 3% of `torch.Tensor` and the
    # run was green, because everything it could not read went into `torch is C` and
    # a bucket is not a failure.
    #
    # `inspect` has no signature for a C implementation, so 571 names sat there —
    # more than every other bucket on this axis together, and more than it called
    # `agree`. torch writes the argument list in the first line of the docstring;
    # `signature_read.from_docstring` reads it, and follows a `See :func:` when a
    # method's line is only a pointer (`bitwise_and() -> Tensor`, with the operand
    # named in `torch.bitwise_and`).
    #
    # **The bucket was not empty and nothing here found that out.** A probe built to
    # ask which input *ranks* each function accepts turned up `where`'s one-argument
    # form and `nonzero(as_tuple=)`, both absent, both living in it — while this axis
    # reported 0 keyword-only absences. Reading the rest turned up five more.
    #
    # What the reading is worth is bounded, and the bound showed up on the first run:
    # ordering from prose is not reliable (`torch.where(condition, input, other)` has
    # its receiver *second*, and one docstring can hold several overloads), so the
    # deferred rows are compared by **arity only** and counted in their own `prose`
    # column. And torch's docstring can simply be wrong — `atanh_` documents an
    # argument it refuses. That one is named in `TORCH_DOC_IS_WRONG` rather than
    # implemented: a check must not drive the code somewhere torch will not go.
    # 512 → 509 filed. Three `_sparse_only` stubs stopped being filed here when
    # `ts_axis.refused()` learned to recognise their factory — they were refusals
    # all along. **The judged count did not move**, which is what says nothing was
    # lost: 386 of a smaller total is the same 386.
    #
    # 386 → 388, and up is the safe direction this row exists to distinguish.
    # `bernoulli` and `poisson` were `(**kw)` — the bag that let them accept any
    # keyword at all and drop it — and a bag reads as *variadic*, which is the bucket
    # meaning **cannot be judged**. They lost it: `bernoulli` declares `out=` (the
    # seat torch has, refused here with the wording that says why) and `poisson`
    # declares nothing more than it takes. Two rows left an absorbing state, which is
    # the movement this check was written to make visible in the other direction.
    #
    # **388 → 414, and this is the largest single move the row has recorded.** The
    # core's in-place methods are generated: `_make_inplace` wraps the partner
    # function in a `def method(self, *args, **kw)`, so every one of them declared a
    # bag and every one of them was `variadic` — *cannot be judged*, twenty-six rows
    # deep, and none of them counted as a gap anywhere. Teaching the generator to
    # copy the partner's signature onto what it builds emptied that pocket, and two
    # real defects were underneath it: `logit_` refused an `eps` torch computes, and
    # `scatter_` had no `reduce` at all. Neither was reachable from a bucket that
    # means the question cannot be asked.
    #
    # **414 → 470, and this one was larger again.** `_ops._make_inplace` was only one of
    # four generators; the other three live in `_tensor.py` (`_bind_inplace`,
    # `_bind_from_module`) and could not read their sources at class-build time, because
    # `_ops` imports `_tensor` and not the other way round. `borch/__init__.py` copies
    # the partner's signature onto each of them now, after both modules exist — the same
    # place and the same reason `_link_wrapped` already sat there.
    #
    # Eleven more came from a `__wrapped__` that **pointed at itself**: `_ops.eq` and
    # `Tensor.eq` are one object, so linking one to the other made a loop, and
    # `inspect.signature` does not shrug at a loop — it raises. Worse than the bag it
    # was meant to remove, and filed under the same wording.
    #
    # What the 56 exposed: fifteen places where torch's own prose is wrong (see
    # `TORCH_DOC_IS_WRONG`), one real core defect (`resize_as_`'s argument is
    # `the_template`, and neither the docstring's `tensor` nor the family's `other` is
    # accepted by torch), and five borch.ts rows on the other axis.
    # **470 → 469, and down is the direction this row exists to stop.** It is
    # allowed here for the one reason that makes it not a loss: `squeeze` became
    # `(self, *dim)` because **torch's takes several axes**, so the row moved into
    # `variadic` by matching torch rather than by hiding from it. The bucket means
    # *no positional list to compare*, and there genuinely is none on either side.
    #
    # Written out because the check cannot tell those apart and a number moving the
    # wrong way must not be edited quietly. What went in with it: `x.squeeze(0, 2)`
    # is torch's form and stopped here, and `x.squeeze(1)` on an axis that is not
    # length 1 is a no-op in torch and raised numpy's `ValueError` here.
    #
    # **469 → 470**, and back up by a different row: `transpose_`. `_make_inplace`
    # reads a module function, and there is no `_ops.transpose` — so it fell back to
    # a closure that calls the method, and `_forwards` had a bag to copy from. It
    # reads `Tensor.transpose(dim0, dim1)` now, one attribute away and fully spelled
    # the whole time.
    "Tensor": (471, 510),
    # 119 → 132. Thirteen loss constructors left the uncomparable bucket when they
    # stopped being `(*args, reduction='mean', **kw)` and grew torch's own parameter
    # list, and all thirteen landed in `agree`. **The ratio moving upward is what a
    # fix looks like here** — the total did not change, and no other number in this
    # file would have recorded that anything happened.
    # 144 → 145. `LazyLinear`, see `SHORTER`.
    # 145 → 147 of 161 → 163. `LinearCrossEntropyLoss` and its `Options`, which were
    # declined under *newly arrived in torch — looked at once it settles* until that
    # sentence was re-read by calling. Both land in `agree`; the filed total rises by
    # the same two, which is what an implementation looks like from here.
    "nn": (147, 163),
    # 76 → 84. **Eight pooling functions stopped being uncomparable.** They ended
    # `(…, **_)`, which makes the whole signature read as `VARIADIC` — and on this
    # axis `variadic` means *cannot be compared at all*, so one `**_` bought silence
    # over eight names. What it swallowed was `return_indices`, which changes nothing
    # in a `*_with_indices` function (measured: torch returns the pair whichever way
    # it is set), so it is named and unused now. The difference between that and
    # `**_` is that a reader can see it.
    # 84 → 109. Twenty-five `nn.functional` names are C too.
    # 109 → 110 of 126 → 127. `linear_cross_entropy`, the functional half of the pair
    # one namespace up.
    # 110 → 118. **Eight rows left the bucket meaning *cannot judge*.** The in-place
    # activations — `relu_`, `celu_`, `elu_`, `selu_`, `hardtanh_`, `leaky_relu_`,
    # `threshold_`, `rrelu_` — were generated with `(x, *args, **kw)`, so the axis
    # could not compare them and a caller could pass anything. The arguments did
    # reach; what a bag adds is accepting what torch refuses, and `F.relu_(x, 0.5)`
    # ran here with the 0.5 landing in `inplace` while torch answered `TypeError`.
    #
    # The list is now derived from the base function minus `inplace`, which is the
    # rule across all eight, and `bind` makes the refusal real rather than documented.
    "nn.functional": (118, 127),
    "optim": (14, 14),
    "optim.lr_scheduler": (16, 16),
    # 0 → 5, same reading. The four `unaligned` are `householder_product`, `lu`,
    # `matrix_power` and `qr` naming their input `A` where the core says `t`.
    #
    # **5 → 40 of 42, and this is the number this whole file exists to move.** The
    # docstring below says a judged share falling while the total holds is the
    # dangerous direction, because *cannot judge* is absorbing. It had 37 rows in it
    # here, and they were not unreadable — the reader matched the bare name and
    # `linalg` writes `linalg.solve(...)`. One anchored prefix in
    # `signature_read._opens` and the bucket empties to 1.
    #
    # **It was not empty.** Ten divergences came out, five `shifted` and five
    # `unaligned`, each named on its row above. Sixteen more are `renamed` — the core
    # calls torch's `A` its `input` — which is the kind already written down two lines
    # up as the row's standing shape.
    "linalg": (40, 42),
    # 13 → 14 of 18 → 19. `DistributedSampler`, declined until its reason was read
    # again: it was grouped with the collectives by its name, and given `num_replicas`
    # and `rank` it never touches a process group. It lands in `agree`, so the filed
    # total and the judged count rise together — an implementation, not a widening.
    "utils.data": (14, 19),
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
# 33 → 5, and **every one of the twelve optimizers now matches torch name-for-name
# and kind-for-kind.** The five left are `dim_order`'s `ambiguity_check` and
# `DataLoader`'s four worker settings.
#
# The last pass split the 33 by measuring rather than by reading:
#
#   `foreach`, `fused`   accepted and ignored. Sixteen settings across every
#                        optimizer torch offers them on, four steps each — **all
#                        sixteen reproduce the default answer exactly.** They pick a
#                        kernel; there is one kernel here.
#   `capturable`,        refused, because **torch refuses them too** on CPU:
#   `differentiable`     `AssertionError` and `RuntimeError`, measured on ten each.
#
# **And four real features were hiding among them.** `amsgrad` on Adam and AdamW,
# `centered` and `momentum` on RMSprop, `decoupled_weight_decay` on NAdam and RAdam —
# all algorithm variants that change values, all absent, and **none of them on this
# list**, because torch takes them positionally. This number was the list of what was
# owed here, and the four things most worth owing were on the other axis.
# **5 → 0. The table is empty, and how to read an empty one is the lesson in it.**
#
# It counts *keyword-only* arguments torch has and the core does not. That filter is
# in its name and not in anybody's reading of it, so while it stood at 33 it looked
# like the list of what the optimizers owed — and `amsgrad`, `centered`, `momentum`
# and `decoupled_weight_decay`, which all change values, sat outside it because torch
# declares them positionally. **A number can be correct, complete and named, and
# still be read as answering a wider question than it asks.**
#
# Stated once, in the form the next such table should copy: **this omits every
# absence torch declares positionally, and that set is not empty** — those live in
# `SHORTER` above.
KEYWORD_ONLY_ABSENCES = 0


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
# 57 → 48. Nine optimizers stopped taking fewer positional arguments than torch when
# they took `foreach`, `capturable` and the rest in torch's own seats.
# 48 → 47. `DataLoader` took torch's whole list.
# 47 → 48. `stft` joined when it became readable.
# 48 → 49. `LazyLinear`, which is short of torch's `device`/`dtype` like the rest of
# the layers, and could not be counted here at all while its signature was variadic.
# 49 → 36. The thirteen activations plus `Dropout`, all short by exactly `inplace`.
# Unlike the `device`/`dtype` rows below, these *were* calls that raise:
# `nn.LeakyReLU(0.2, True)` is a line people actually write, and it did.
# 36 → 9. Seventeen `device`/`dtype` pairs, `AvgPool2d`'s four and `Flatten`'s
# `end_dim`. The nine left are named arguments this library does not have —
# `create_graph`, `align_to_window`, `ceil_mode` on the LP pools, and the rest.
# **9 → 0. There is no longer a call torch takes by position that this refuses.**
#
# The number started at 57 and came down in five steps, each a family rather than a
# name: `maximize` on ten optimizers, `bias` and `affine` on thirteen normalisation
# layers, `inplace` on thirteen activations, `device`/`dtype` on seventeen more, and
# then nine singletons. **Every one of them was a line a torch recipe writes.**
#
# Zero here does not mean the two libraries agree. It means the narrower thing the
# name says: **a positional call that works in torch does not stop here.** The
# arguments can still be refused once they arrive — `device`, `create_graph`,
# `capturable` all are — and `SHORTER` above still counts rows where torch declares
# more names than the core does. What has gone is the failure that happens before any
# of that: a `TypeError` about an argument count, from a line copied out of the
# documentation.
TORCH_REACHES_FURTHER_BY_POSITION = 0

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
