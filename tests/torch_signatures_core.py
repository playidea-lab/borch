"""Counts where the **core** takes different arguments from **real torch**.

    uv run --with numpy --with torch --with torchvision python tests/torch_signatures_core.py
    uv run --with numpy --with torch --with torchvision python tests/torch_signatures_core.py --show nn

## Every other comparison in this repository is between two of our own libraries

    tests/torch_gap.py             core        ↔ real torch          names only
    tests/test_torch_signatures.py borchvision ↔ real torchvision    arguments
    tests/ts_signatures.py         core        ↔ borch.ts            arguments
    tests/test_binding_arguments   binding     ↔ borch.ts            arguments
    borch-ts/test/parity.ts        core        ↔ borch.ts            behaviour

Only the second asks an outside authority, and it asks it about borchvision. So for
`borch` itself **nothing compares an argument list against torch**, and the checks
that do compare arguments have both feet inside the project.

That is not a missing axis so much as a shared blind spot. When the core and borch.ts
disagree, a row says *they disagree* and never *which one to move* — and when they
agree, the whole apparatus can be converged on one error and report agreement. Both
happened this week:

- `SmoothL1Loss` — the core took `(beta, reduction)` and borch.ts `(reduction, beta)`.
  Two instruments reported the split (`ts_signatures.py` as a row, `parity.ts` in the
  words *the sisters have parted*) and **neither could say who was wrong.** torch
  settled it: borch.ts was right.
- The nine loss constructors that `ts_signatures.py` briefly reported as borch.ts
  inserting arguments. borch.ts had them because torch has them.

## What it compares, and what it deliberately folds

Constructor arguments for `nn`, and function arguments for the rest, by name and in
order, through `signature_read.parameters` — the one reader all three of these files
share, after the same defect was found sitting in two of them at once.

**torch's `size_average` and `reduce` are dropped, and that is a claim.** torch
documents both as deprecated, keeps them in every loss signature for compatibility,
and ignores them whenever `reduction` is passed. Keeping them makes every loss
mismatch at position 0 for a reason nobody acts on; dropping them lets the real
orderings show. The fold is written into `DEPRECATED` below rather than into the
comparison, because a true fold still has to be attested somewhere a reader can find
it — which is the lesson `_camel` taught by folding `eq_` onto `eq` and reporting a
name present that was not.

**"ignores them whenever `reduction` is passed" was too strong, and the gap is
exactly where the fold can mislead.** It holds for a keyword call, and a positional
call is the only kind that reaches these seats — measured:

    F.l1_loss(a, b, 'sum')        torch: 2.5  (the *mean*)   here: 10.0 (the sum)
    F.mse_loss(a, b, 'sum')       torch: 7.5  (the mean)     here: 30.0
    F.smooth_l1_loss(a, b, 'sum') torch: 2.0  (the mean)     here: 8.0
    F.huber_loss(a, b, 'sum')     torch: 8.0  (the sum)      here: 8.0

torch's third seat there is `size_average`, and its legacy path reads a truthy value
as `reduction='mean'`. `huber_loss` is the control: it is newer and carries no
deprecated pair, so its third seat really is `reduction` and the two agree.

So on the five that carry the pair, **`agree` in this table means the names line up
once two are removed, and not that the same call means the same thing.** Recorded
rather than fixed, because carrying `size_average` and `reduce` means implementing a
reinterpretation torch itself warns about on every use, and that is a decision about
what this subset promises rather than a patch — the same standing the `SHIFTED` table
above gives its own rows.

## The limit that matters: 571 rows where torch is C

`inspect` cannot read a C implementation, and `torch.Tensor` is almost entirely C —
496 of its methods, plus 41 in `linalg` and 34 in `nn.functional`. They are **counted
as `torch is C` rather than skipped**, because the first version returned the same
value for "torch does not have it" and "torch has it and I cannot read it", and
`Tensor` came back with three agreements and looked finished.

So this axis sees `nn`, `optim` and the schedulers well, and sees very little of the
tensor surface. Two other sources were measured and **neither is used**:

- `torch.overrides.get_testing_overrides()` gives readable Python stubs for 1,426
  entries — and they are abbreviated. `Tensor.std` comes back as `(input, dim=None)`
  where the real signature carries `correction` and `keepdim` as well. Comparing
  against it would report the core as having *extra* arguments it correctly has.
  A lower-fidelity source that looks like a higher-fidelity one is worse than none.
- The docstring's first line is the real thing — `add(other, *, alpha=1) -> Tensor`
  — and reading it means a **third argument parser**. Two of the three that existed
  this week had the same defect in them, which is the reason `signature_read.py`
  exists; adding a fourth reader to close this gap would be undoing that.

**The second door was opened, and the objection above is what decided how.** The
parser went *into* `signature_read.py` as `from_docstring`, beside the two readers
already there — not a fourth reader, the same one growing a second source. That is
the whole of the answer to "adding a fourth reader would be undoing that": there is
still one place where an argument list is read, and a defect in it is still findable
in one place.

571 → 59. What made it worth doing was not the size of the number but that **the
bucket turned out not to be empty, and nothing here found that out.** A probe built
to ask which input *ranks* each function accepts turned up two absences living in it
— `torch.where`'s one-argument form and `nonzero(as_tuple=)` — while this axis
reported 0 keyword-only absences. Reading the rest turned up five more, one of them
`svd` being `torch.linalg.svd` under `torch.svd`'s name, returning a different shape
from the same call.

**The authority is weaker than `inspect`'s and the rows say so.** A method's line is
often only a pointer (`bitwise_and() -> Tensor`, with the operand named over in
`torch.bitwise_and`), so a `See :func:` is followed — and past that follow the
*order* is not usable: `torch.where(condition, input, other)` has its receiver
second, and one docstring can hold several overloads. Those rows are compared by
**arity alone** and counted in a `prose` column that is never added to `agree`. And
torch's prose can be wrong outright: `Tensor.atanh_` documents `atanh_(other)` and
raises `takes no arguments` when called that way. `TORCH_DOC_IS_WRONG` names it
rather than the library growing a parameter torch refuses — **a check must not drive
the code somewhere the thing it checks against will not go.**

The first door is still shut, for the reason above it.

## A third source, measured and rejected — **and it failed the way that is hard to see**

An argument torch *accepts and then ignores* is invisible to every check here.
`F.gumbel_softmax(eps=…)` was found by hand: accepted, and dropped on the floor. The
binding check asks whether a call site drops what it accepts, so threading the argument
dutifully through satisfies it **exactly as well as being right does** — the structural
checks stayed green while the two libraries returned different numbers for the same
call. torch's own answer is `warnings.warn("eps parameter is deprecated and has no
effect")`.

The obvious instrument is to scan torch's docstrings for parameters its prose calls
deprecated or ignored. It was built and run. **108 rows** across `F`, `torch` and `nn`:

- Almost all of them are `size_average` and `reduce` — the pair already folded above,
  so that mass is a restatement of something known.
- Much of the rest is the scan misreading itself. `F.cross_entropy.target` matched
  because the *`ignore_index`* line contains the word "ignored"; `F.lp_pool1d.input`
  matched on a sentence about padded windows.
- **`gumbel_softmax.eps` is not among the 108.** Its docstring says nothing about it.
  The deprecation exists only as a runtime warning.

So the scan missed the single case known in advance to be positive, while producing a
hundred-row report that reads as thorough. That is the failure mode this repository
keeps meeting from a new direction: not an instrument that is quiet, but one that is
**confidently full**. Nobody reading that output would go looking for what is absent.

The general form is worth keeping even if the check never gets built: **prose about
behaviour is not behaviour, and a scan over prose inherits every silence in the prose.**
That applies to torch's documentation and to ours.

The authority is the call. `warnings.catch_warnings(record=True)` around a real
invocation answers it, and it asks a question no axis here asks — *what does torch do
with this argument* — as opposed to *what argument does torch declare*. Left unbuilt on
purpose: it is a new axis, not a fix. If it is built, one thing goes in at the first
line — **it must fail when it probes nothing.** An empty warning list from a call that
silently never happened is indistinguishable from torch warning about nothing, and an
empty result reads as a pass. That is the absorbing bucket again, one level up.

## Which findings this repository is entitled to expect, measured

How many argument lists actually get judged against torch, per namespace:

    optim.lr_scheduler      16 of  16
    optim                   14 of  14
    utils.data              13 of  18
    nn                     119 of 161
    nn.functional           75 of 126
    Tensor                   9 of 512
    linalg                   0 of  42

**The tensor surface is the largest body of API in the project and almost none of it
is checked against torch by argument.** `linalg` is none of it. What holds those is
the core and borch.ts agreeing with each other, plus the golden — which compares
values, and only for cases somebody wrote.

That is not an argument for building the parser. It is the map of where a finding
like `std` could have come from: `std` was caught on `ts_signatures.py`'s axis, and
that axis cannot say which side is wrong. Had both libraries taken the correction
first, nothing here would have noticed. **Knowing which findings we are entitled to
expect is worth more than a coverage figure**, and a coverage figure computed over
the judged rows alone would have read as 94%.

## What it cannot see

Types, defaults, keyword-only-ness, and whether the values agree. The third of those
matters more here than on the other axes: the core writes `(reduction='mean', *,
weight=None)`, and reading that as `[reduction, weight]` loses the fact that `weight`
**cannot be passed positionally at all**. A row can therefore look like an ordering
difference when the argument is not reachable by position on one side. Said here
because the four losses below are exactly that shape.
"""

import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

# torch keeps these in every loss signature and ignores them when `reduction` is
# given. **A fold, and therefore a claim** — see the module docstring.
#
# **Scoped to the namespace the claim is about.** It used to apply everywhere, and
# `reduce` is not only a deprecated loss argument: `Tensor.scatter_reduce` takes a
# live, required `reduce` naming the operation ("sum", "amax", …). Folding it away
# there removed a parameter the caller must pass, and the row came back reading as
# *the core has an argument torch does not* — the fold inventing the opposite of
# what is true.
#
# It could not be seen until this axis could read C signatures at all: the whole
# `Tensor` namespace was one unjudged bucket, so the row did not exist to be wrong.
# **A reason that is true about one thing, standing where a reason about another
# belongs** — the same shape three times in this repository now.
#
# **The fold is gone, and this set is empty.** Not because folding was wrong in
# principle but because there is nothing left to fold: the core carries
# `size_average` and `reduce` now, at torch's own seats, so the two lists agree
# without help. A fold exists to make comparable things that differ only in a way
# nobody acts on, and its cost is that it hides a difference somebody might.
#
# Here it hid one. Removing the pair moved every later argument one or two seats
# forward, so `F.l1_loss(a, b, 'sum')` gave the sum here and the mean in torch while
# this table read `agree`. The fold was not the cause of that, but it was the reason
# nothing said so.
#
# The names stay written down. Empty, this attests that the fold was considered and
# retired; deleted, the next reader meets a table with no history and folds again.
DEPRECATED = frozenset()
DEPRECATED_IN = ("nn", "nn.functional")

# **Where torch's docstring contradicts torch.**
#
# `Tensor.atanh_` documents itself as `atanh_(other) -> Tensor`, and calling it that
# way gives `TensorBase.atanh_() takes no arguments (1 given)`. `arctanh_` is its
# alias and carries the same line. Both are measured, not assumed.
#
# This is the standing caveat on `signature_read.from_docstring` arriving in the
# first run that used it: the docstring is what somebody wrote next to the function,
# not the function. Implementing the argument to satisfy the axis would have put a
# parameter into this library that torch refuses — **a check driving the code
# somewhere the thing it checks against will not go**, which is worse than the row.
#
# Held here rather than by making the reader smarter, because there is nothing to be
# smart about: only a call can tell, and the reader does not make calls.
TORCH_DOC_IS_WRONG = {"Tensor::atanh_", "Tensor::arctanh_"}


def _spaces():
    """The namespaces with a real torch counterpart, as `(name, ours, theirs)`."""
    import torch
    import borch

    return [
        ("Tensor", borch.Tensor, torch.Tensor),
        ("nn", borch.nn, torch.nn),
        ("nn.functional", borch.nn.functional, torch.nn.functional),
        ("optim", borch.optim, torch.optim),
        ("optim.lr_scheduler", borch.optim.lr_scheduler, torch.optim.lr_scheduler),
        ("linalg", borch.linalg, torch.linalg),
        ("utils.data", borch.utils.data, torch.utils.data),
    ]


ABSENT = object()       # the name is not in that namespace at all


def _read(space, holder, name, reach=False):
    """`[names]` · `VARIADIC` · `ABSENT` · `None` (present but unreadable).

    **`ABSENT` and `None` are held apart on purpose.** The first version returned
    `None` for both and the caller skipped it as "not in torch", which quietly
    dropped every method torch implements in C — `inspect` cannot read those, and
    `torch.Tensor` is almost all of them. `Tensor` came back with three agreements
    and looked finished.

    That is this week's recurring shape once more: a name that produces no row of any
    kind cannot be counted as uncounted. Here it costs a whole namespace.
    """
    import inspect
    from signature_read import VARIADIC, parameters, positional, of_class

    read = positional if reach else parameters
    thing = getattr(holder, name, None)
    if thing is None:
        return ABSENT
    if inspect.isclass(thing):
        return of_class(thing, reach=reach)
    if space == "Tensor":
        # Reached through the class, so the first parameter is the receiver — except
        # a `staticmethod`, which has none. Asked of the raw attribute so a
        # descriptor answers as itself.
        held = inspect.getattr_static(holder, name, None)
        return read(thing, receiver=not isinstance(held, staticmethod))
    return read(thing)


# The verdicts that claim *a positional call lands on the wrong parameter*. They are
# the only ones `_reachable` re-asks, because they are the only ones whose meaning
# depends on a positional call being possible.
SHIFTS = ("dropped", "inserted", "reordered")


def _reachable(space, ours, theirs, name, note):
    """`note`, unless nothing a caller can reach by position is out of place.

    **A shift the language forbids is not a shift.** `optim.Adagrad` read as
    `inserted` — the sharpest bucket here — because the core's `maximize` sits in
    torch's `foreach` seat by name order. Both libraries write it `*, maximize`, so
    no positional call can land on either. The row described a call nobody can write.

    Asked by *re-running the same verdict on the positional prefixes*, rather than by
    a rule about tails. A rule would have to guess what a shift confined to
    keyword-only arguments means; re-asking makes the answer the same function of the
    same evidence, one question narrower.

    **The re-ask's answer is the row's answer** — it is not downgraded to a bucket of
    its own, and it does not become `agree`. `Adagrad` comes back `shorter`, which is
    true and useful: torch has a positional `foreach` after `eps` and the core stops
    there. A private bucket would have been the tidier-looking choice and would have
    taken `foreach` out of the only bucket that names missing arguments.

    That is the failure this function has to avoid in its *other* direction, and it is
    the larger of the two. Across these namespaces torch has **54 keyword-only
    parameters the core does not have** — thirteen of them a `bias` flag on the
    normalisation layers, six on `Adam` alone. Making keyword-only parameters
    invisible to the comparison, rather than merely ineligible for `shifted`, would
    retire two invented hazards and hide all 54 real absences. `parameters()` is
    therefore left alone; only this re-ask uses `positional()`, and
    `test_keyword_only_arguments_stay_visible` holds that line.
    """
    mine = _read(space, ours, name, reach=True)
    yours = _read(space, theirs, name, reach=True)
    if not isinstance(mine, list) or not isinstance(yours, list):
        return note                                  # variadic or unreadable — unchanged
    import ts_signatures
    kept = [p for p in yours if p not in DEPRECATED]
    narrower = ts_signatures._verdict(mine, kept)
    return note if narrower in SHIFTS else narrower


def compare():
    """`{space: [(name, ours, theirs, verdict), ...]}` over names present in both.

    A name missing on one side is `torch_gap.py`'s count and is not asked again here.
    Refusal stubs are dropped: the core carries them **in order to refuse**, so their
    argument list is a message rather than a feature.
    """
    import torch_gap
    import ts_axis
    import ts_signatures
    from signature_read import DEFERRED, VARIADIC, from_docstring

    stubs = ts_axis.refused()
    out = {}
    for space, ours, theirs in _spaces():
        rows = []
        for name in sorted(torch_gap._public(ours)):
            if space == "Tensor" and name in stubs:
                continue
            mine = _read(space, ours, name)
            yours = _read(space, theirs, name)
            if yours is ABSENT:
                continue                             # not in torch — the name axis's row
            if yours is None:
                # **torch has it and `inspect` cannot read it** — a C implementation.
                # torch writes the argument list in the docstring's first line, so
                # the list is read from there before the row is given up on. See
                # `signature_read.from_docstring` for what that is worth: it is
                # torch's prose rather than torch's behaviour, and a row from it is
                # a lead to check rather than a measurement.
                #
                # It matters because this bucket was **571** — bigger than every
                # other bucket on this axis together, and bigger than `agree`. Two
                # real absences were found inside it by a probe built to ask about
                # something else, which is not evidence of two.
                # `compare()` reads full lists (`parameters`), so the docstring is
                # read the same way — keyword-only names kept. `_reachable()` below
                # re-asks with the positional prefix where a shift is claimed.
                yours = from_docstring(getattr(theirs, name, None), name)
                if yours is None or yours is VARIADIC:
                    rows.append((name, mine if isinstance(mine, list) else None, None,
                                 "torch is C" if yours is None else "variadic"))
                    continue
                if isinstance(yours, DEFERRED) and isinstance(mine, list):
                    # Read from the module function this method's docstring points
                    # at, so the order cannot be used — see `DEFERRED`.
                    #
                    # **Membership cannot be used either, and that took a second
                    # run to see.** By name, 74 methods were missing something torch
                    # documents; 42 of the 74 were missing `other`, which the core
                    # calls `b`. `mat2`, `vec2`, `exponent`, `L`, `A`, `B` and
                    # `tensor` are the same thing — torch's name for the operand
                    # against ours. A set difference cannot tell a rename from an
                    # absence, so it called every rename an absence and produced a
                    # list where four of every five rows were noise.
                    #
                    # **Arity can be used.** How many arguments torch documents, and
                    # how many the core takes, is a claim prose supports: fewer here
                    # means at least one is genuinely missing, whatever it is called.
                    # Exactly one of torch's names is the receiver, and `out` is
                    # keyword-only and never a method argument — both come off.
                    counted = [q for q in yours if q != "out"]
                    rows.append((name, mine, list(yours),
                                 "prose short" if len(mine) < len(counted) - 1
                                 else "prose agree"))
                    continue
                if isinstance(yours, DEFERRED):
                    # The core side is variadic or unreadable, so there is nothing to
                    # match membership against either.
                    rows.append((name, None, None, "variadic"))
                    continue
                if space == "Tensor" and yours and yours[0] in ("input", "self"):
                    # Reached as a method, so torch's documented receiver is not an
                    # argument the caller passes.
                    yours = yours[1:]
            if mine is VARIADIC or yours is VARIADIC:
                rows.append((name, None, None, "variadic"))
                continue
            if mine is None:
                rows.append((name, None, None, "no signature"))
                continue
            if f"{space}::{name}" in TORCH_DOC_IS_WRONG:
                rows.append((name, mine, list(yours), "torch doc is wrong"))
                continue
            kept = ([p for p in yours if p not in DEPRECATED]
                    if space in DEPRECATED_IN else list(yours))
            note = ts_signatures._verdict(mine, kept)
            if note in SHIFTS:
                note = _reachable(space, ours, theirs, name, note)
            rows.append((name, mine, kept, note))
        out[space] = rows
    return out


# **The verdict words mean the opposite thing here, and the row said so out loud.**
#
# `_verdict(wanted, yours)` is `ts_signatures`'s, written for the axis that asks
# *what has borch.ts not carried across from torch* — so it is handed `(torch,
# borch.ts)` there and answers `shorter` when borch.ts is the short one. This axis
# asks the other question, hands it `(ours, torch)`, and gets `shorter` back when
# **torch** is the short one.
#
# The summary already compensated: its `shorter` column prints `shorter + longer`,
# so the number a reader sees is right. The compensation stopped at the column. Rows
# kept printing the raw word, so one screen carried `— longer` on `AvgPool2d` (where
# ours is the shorter list) directly under `shorter 57`.
#
# Two sessions read that screen, quoted different halves of it, and disagreed for an
# hour about which bucket those names were in. **Both quotes were accurate.** So the
# flip happens once, here, before anything is printed, and the two halves of the
# output say the same word about the same row.
#
# **The column is split too**, because the merged one answered a question nobody
# asks. It printed `shorter + longer` under the name of one of them, and the two
# have opposite consequences: where ours is the shorter list a torch call can raise
# here, and where ours is the longer one nothing written against torch breaks. One
# number cannot say which of those a namespace has.
#
# **Three numbers live near each other now and none of them is the same number.**
# Written out because two of them were 57 on the day this was split, which is the
# kind of coincidence that gets two things equated:
#
#     shorter   61   ours takes fewer names; torch's list is longer and ours a prefix
#     …of which 57   torch also reaches further **by position**
#     longer    12   ours takes more names than torch. Nothing of torch's breaks.
#
# (Those two lines were written the other way round first, an hour after the flip
# above was added, by reading the words as they mean *before* the flip. The flip
# fixes the output and not the habit.)
#
# The four in 61 and not in 57 are `Tensor.dim_order`, `optim.Adafactor`,
# `optim.Rprop` and `optim.SGD`. Measured: in every one, torch's extra arguments are
# keyword-only, so torch's name list is longer and no positional call can reach the
# difference. `TORCH_REACHES_FURTHER_BY_POSITION` in the test file is that 57, and it
# is a **subset** of this 61 rather than a second opinion about it.
_FROM_HERE = {"shorter": "longer", "longer": "shorter"}


def main(argv):
    show = argv[argv.index("--show") + 1] if "--show" in argv else None
    rows = compare()
    tally = {}
    for space, found in sorted(rows.items()):
        counts = {}
        for _n, _m, _y, note in found:
            note = _FROM_HERE.get(note, note)
            counts[note] = counts.get(note, 0) + 1
            tally[note] = tally.get(note, 0) + 1
        shifted = counts.get("dropped", 0) + counts.get("inserted", 0) \
            + counts.get("reordered", 0)
        mark = " " if not shifted else "✘"
        print(f"  {mark} {space:22s} agree {counts.get('agree', 0):>4}   "
              f"shifted {shifted:>3}   unaligned {counts.get('unaligned', 0):>3}   "
              f"shorter {counts.get('shorter', 0):>3}   "
              f"longer {counts.get('longer', 0):>4}   "
              f"renamed {counts.get('renamed', 0):>4}   "
              f"variadic {counts.get('variadic', 0):>4}   "
              f"torch is C {counts.get('torch is C', 0):>4}   "
              # **Printed apart from every measured bucket, and never folded into
              # one.** These come from torch's prose by way of a `See :func:` and
              # are compared by membership only. Adding them to `agree` would let a
              # count of what was *read* stand in for a count of what was *checked*
              # — which is the fault this whole axis was built after.
              f"prose {counts.get('prose agree', 0):>3}/"
              f"{counts.get('prose short', 0):<3}")
        if show is not None and space.startswith(show):
            for name, mine, yours, note in found:
                if note in ("agree", "prose agree"):
                    continue
                print(f"      · {name}")
                print(f"          torch ({', '.join(yours or [])})")
                print(f"          core  ({', '.join(mine or [])})  "
                      f"— {_FROM_HERE.get(note, note)}")
    print("\n코어를 **진짜 torch** 와 인자로 대조한다 — 이 저장소에서 바깥 권위를 묻는 "
          "둘째 검사다.")
    print(f"  맞음 {tally.get('agree', 0)} · 밀림 "
          f"{tally.get('dropped', 0) + tally.get('inserted', 0) + tally.get('reordered', 0)}"
          f" · 못 맞춤 {tally.get('unaligned', 0)} · 우리가 덜 받는다 "
          f"{tally.get('shorter', 0)} · 우리가 더 받는다 "
          f"{tally.get('longer', 0)} · 이름만 다르다 "
          f"{tally.get('renamed', 0)} · 못 잼 {tally.get('variadic', 0)}"
          f" · torch 가 C {tally.get('torch is C', 0)}\n"
          # **Apart from the measured tally, never inside it.** These come from
          # torch's docstring by way of a `See :func:`, and are compared by how many
          # arguments each side takes rather than by which. Folded into `맞음` they
          # would let a count of what was *read* pass for a count of what was
          # *checked*.
          f"  산문에서 읽은 것 {tally.get('prose agree', 0)}"
          f" · 그중 우리가 짧은 것 {tally.get('prose short', 0)}"
          f" · torch 문서가 틀린 것 {tally.get('torch doc is wrong', 0)}")
    print("  `size_average` 와 `reduce` 는 접었다 — torch 가 폐기했다고 적어둔 둘이다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
