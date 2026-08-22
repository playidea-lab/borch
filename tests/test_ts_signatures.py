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
    # 0 -> 1. **This row is a real gap that opened today rather than one that was
    # found.** The core's MaxPool1d/2d/3d grew torch's padding, dilation and
    # ceil_mode; borch.ts still takes `(kernel, stride, returnIndices)`, so
    # `new nn.MaxPool2d(2, 2, 1)` sets returnIndices where torch and the core set
    # padding. Raised by the session that moved the core, recorded rather than
    # tidied away -- `nn.ts` belongs to another session and is blocked on its own
    # decision, and a number quietly held at 0 while the two sides part is the thing
    # this table exists to prevent.
    # 1 -> 4. Three more opened the same way the MaxPool row did: the core's Conv1d,
    # Conv2d and Conv3d took torch's dilation, groups and padding_mode, and borch.ts
    # still takes `(in, out, kernel, stride, padding, bias)`. `new nn.Conv2d(3, 16,
    # 3, 1, 1, false)` turns the bias off over there and sets dilation here.
    #
    # **Closing a gap against an outside authority opens one between two of our own
    # libraries**, and it is not a side effect: with three implementations and one
    # outside authority, the core can only move toward torch by moving away from
    # borch.ts until borch.ts follows. The pair of axes disagreeing is the mechanism
    # working. The danger is only ever a number held still while the sides part.
    # 4 -> 7. Three more, the same way: the core's ConvTranspose1d/2d/3d took
    # torch's output_padding, groups and dilation, and borch.ts has not followed
    # yet. This is the fourth family in a row to open a row here while closing one
    # against torch, which is what a repository with three implementations and one
    # outside authority does -- the core cannot move toward torch except by moving
    # away from borch.ts until borch.ts follows.
    # 7 -> 4. The three Conv layers followed the core: torch's order, with `bias`
    # eighth and `dilation` sixth. `dilation` is implemented in the shader (one
    # token in three index expressions, plus the spacing in the cache key, without
    # which two calls differing only in dilation would silently share a shader),
    # `groups` by slicing and joining so the gradient follows from the pieces, and
    # `padding_mode` by padding in the layer as torch's layer does.
    #
    # **`tsc` named all six positional call sites the moment the constructor moved**
    # -- `new Conv2d(cin, cout, 3, stride, 1, false)` with a boolean where a number
    # now goes. The same move in Python was silent, and another session had six
    # tests break on it this morning.
    # 4 -> 1. The three ConvTranspose layers followed the core: torch's order,
    # `bias` eighth and `dilation` **ninth**, which is not the convolution's order
    # and is torch's. outputPadding, groups and dilation implemented.
    #
    # `outputPadding` is expressed as a longer output and nothing else: the shader
    # finds, for each output cell, the input cells that reach it, and asked for more
    # cells it answers by the same rule -- which is what torch's extra rows are.
    # They are **not zeros**, and a version that wrote zeros would agree on every
    # shape and part on the values.
    "nn": 1,
    # 1 -> 2. `F.embedding_bag` moved `mode` from third to sixth, where torch
    # has it, and borch.ts still takes it third. Same pair as the layer above,
    # one level down.
    # 2 -> 1. `F.embeddingBag` followed the core: `mode` sixth, where torch has it.
    # `tsc` named all eight call sites the instant it moved -- five golden cases and
    # the layer's own two -- because a mode string does not fit `number | null`. The
    # identical move on the Python side was silent.
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
    "nn": 26,   # +1, EmbeddingBag: the core took torch's list, borch.ts has not
    "nn.functional": 1,
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
    "Tensor": 30,
    "nn": 18,   # -1, EmbeddingBag left `renamed` for `unaligned`
    "nn.functional": 1,
    "optim": 1,
    "optim.lr_scheduler": 0,
    "linalg": 17,
    "utils.data": 0,
}

# borch.ts takes a **prefix** of torch's arguments. Not a shift and not silent — one
# too many raises. Held separately so that fixing a shift cannot be paid for by
# turning it into a truncation without anyone noticing.
SHORTER = {
    "Tensor": 16,
    "nn": 24,   # +6: the Conv and ConvTranspose layers left `shifted` for
                #     `shorter`, which is the safe bucket -- a prefix, refusing what
                #     it does not reach. Six dangerous rows became six harmless ones.
    "nn.functional": 0,
    "optim": 0,
    "optim.lr_scheduler": 12,
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
