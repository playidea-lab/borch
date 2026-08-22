"""**Counts the places the binding fills in on borch.ts's behalf.**

This repository's golden cases are written as comparing three implementations against one
expectation. But when a case passes through `borch_webgpu` and that side **assembles
something itself** on top of borch.ts, the case never asks borch.ts anything. The table is
green, and what is missing is missing only for whoever writes TypeScript.

This shape appeared seven times in one session:

- `trapezoid` and `cumulative_trapezoid` — a few lines slicing and adding, in Python. The
  comment read "building one more here would make two copies of the assembly", and there
  **was no first copy; it existed only on the Python side.**
- `bernoulli`, `normal`, `poisson` and `binomial` — built out of numpy.
- `ldl_factor_ex` — a `_Fields` stood up by hand with a scalar slotted into `info`.

And fourteen type conversions such as `half`, `float` and `long` were the same place. All
twenty-one were "the golden cases are green and borch.ts has no such name", and all were
found **by a person spreading one bundle at a time with `--show`.** That is discipline, and
discipline leaks.

## What is used as the signal

**A name torch has, which the binding implements and borch.ts does not have** is the
binding filling in. All twenty-one above are caught by that.

The body is not read to decide "is this an assembly" — `ldl_factor_ex` called borch.ts and
assembled at the same time, and `trapezoid` called several borch.ts methods. The split is
not whether it calls but **whether the name exists over there.**

## Why a count rather than a list

A tool that only prints a list is discipline someone has to run, which repeats exactly the
problem this check exists to stop. So only what is written into the tables below **with a
reason** passes, and anything new blows up naming itself. The same shape as `NOT_PORTED`
and `KNOWN_ABSENT`.

Marks:  python = it belongs to the Python surface, so TS can have no counterpart
        design = deliberately **not** put in borch.ts (a reason is required)
"""

import ast
import json
import pathlib
import re

import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent

# Names it is **right** for borch.ts not to have. Each one needs a reason.
#
# Adding a name here is the decision "this will not be built in borch.ts". If it is simply
# outstanding work, it goes into borch.ts and not here — this table is not a queue.
# **Things belonging to the Python surface, which cannot exist over there.** Serialising the
# random stream, going to and from numpy, looking into storage and sparsity, Python type
# names — none of them has a counterpart in TypeScript.
PYTHON_SIDE = """
as_tensor can_cast dense_dim from_numpy get_default_dtype get_device
get_rng_state initial_seed is_contiguous is_distributed is_grad_enabled
is_inference is_inference_mode_enabled is_storage numpy promote_types
set_rng_state share_memory_ sparse_dim to_dense tolist typename
asarray resize_as_ storage_offset type values
""".split()
# `is_floating_point`, `is_signed` and `is_nonzero` were here. The first two were written
# down as "dtype attributes" and the third as "the Python surface", and **all three are
# torch's names and were absent over there.** It surfaced while carrying across the nine
# `dtype::묻는것::` cases — that carrying cases across is also a way of finding missing
# names repeated several times in this session.

# **Not yet judged.** The place everything landed at once the first time this check ran,
# and each one is one of two things: it exists in borch.ts **under a different spelling** (so
# it is not missing), or it **does not** (so it is work to carry across).
#
# Freezing the count is this list's job. A newly appearing fill is not here and so blows up
# at once, while the old ones stay **by name**, which shows this is not a count left
# half-done. Anything judged comes off this list and moves up, or into borch.ts.
# **It exists over there under a different spelling** — not missing. borch.ts's name is
# written beside it. Unwritten, the next person repeats the same check.
ALIASED = {
    "grid_sampler": "gridSample",
    "is_same_size": "comparing shape",
    "max_pool1d_with_indices": "maxPoolWithIndices",
    "numel": "size",
    "scatter": "scatterSet",
    "swapdims": "swapaxes",
    "take": "indexSelect (after flattening)",
    "take_along_dim": "gather",
    # `broadcast_to`, `moveaxis`, `vdot` and `t` were here, written down as **not missing,
    # since they exist under a different spelling** — but if the spelling torch offers is
    # absent over there, code written with that name simply does not run. All four went in
    # while carrying across the sixty `method2::` cases. "It is an alias" and "the name is
    # not there" can both be true at once.
    "trapz": "trapezoid",
    # **These two arrived when `_flat` stopped folding the first letter**, and they had
    # been hidden by exactly the fault that rule was tightened against: `tensor`
    # matched the class `Tensor` and `flatten` matched the layer `nn.Flatten`. A
    # factory function is not its class and a layer is not a method, so both were
    # reported present because a differently-cased neighbour existed.
    #
    # Verified against the index rather than assumed: it carries `Tensor`, `Flatten`,
    # `ravel`, `reshape` and `from` — and neither `tensor` nor `flatten`.
    "tensor": "Tensor.from",
    "flatten": "ravel (or reshape([-1]))",
}
# `is_tensor` had one line here. The declaration index was not sweeping `index.ts`, so
# `isTensor` went uncaught despite being a public name — not a missing name but **a blind
# spot in the index** — and it vanished when the generator added that file.
#
# That blind spot was caught **from outside.** The check on the generator's side asks
# whether the index matches the declaration files; it does not ask whether every declaration
# file was read. A check that verifies its own input always leaves a blind spot that size,
# and it is visible only from another angle.

# `UNJUDGED` was here — 62 the first time it ran, and it went 62 → 29 → 11 → 2 → 0.
# Forty-three went into borch.ts, fourteen already existed under another spelling, and five
# belonged to the Python surface. **An empty list gets deleted** — left in, the next person
# reads it as work still outstanding.
FILLED_ON_PURPOSE = set(PYTHON_SIDE) | set(ALIASED)

# Things the binding exposes that are not torch's names — never candidates to begin with.
_PRIVATE = re.compile(r"^_")


def _flat(name):
    """Erases the spelling difference — what is being compared is **whether the name exists**,
    not how it is spelled.

    torch runs words together without underscores, as in `searchsorted` and `logsumexp`,
    while borch.ts writes `searchSorted` and `logSumExp`. Converting to camel case by
    looking at underscores alone leaves those two never meeting, and **a name that exists
    comes out as absent** — the first version produced sixty phantom entries that way.

    **The trailing underscore stays.** An in-place form and its counterpart are different
    operations. With all of them stripped, the day `t_` went in the check read it as `t`
    having appeared, and before that `bernoulli_` folded onto `bernoulli` and produced a
    phantom "nineteen present". The same normalisation erased meaning three times in this
    session — erasing and matching are different jobs.

    **And the first letter stays too**, for the same reason one level up. In torch an
    initial capital is the class/function boundary: `nn.Embedding` is a layer and
    `nn.functional.embedding` is a function. Folding both reports the layer as present
    because the function is — measured on the core-to-borch.ts axis, where exactly that
    hid `Embedding` and one other.

    This check passes with the fold and without it, so nothing here exercises the
    difference today. That is the reason to keep the stricter rule rather than the
    reason to skip it: passing both ways means the check cannot tell them apart.
    """
    tail = "_" if name.endswith("_") else ""
    body = name.replace("_", "")
    return (body[:1] + body[1:].lower()) + tail


def _ts_surface():
    """Every name borch.ts **declares.**

    It first swept the source with a regular expression. The comment read "every name
    borch.ts puts out", and the code was counting **any word that is indented and followed
    by an opening parenthesis** — the local variable `const inner = …` was caught as the
    public name `inner`. The regular expression counted 1,323 where the real declarations
    were 845.

    So **a blind spot appeared**: `torch.inner` is absent from borch.ts and this check
    answered that it was there. A check for missing names hiding a missing name, and the
    cause is item seven of the README exactly — what the comment said and what the code
    asked were different.

    It now reads **the same index** the generator behind `site/assets/api.json` reads. That
    comes out of the declaration files `tsc` emits, so a local variable cannot leak into it.
    If that file goes stale this check goes stale with it, and `tests/test_site.py` already
    guards that place — rather than two checks hanging on one file, each watches its own
    share.
    """
    index = ROOT / "site" / "assets" / "api-index.json"
    declared = json.loads(index.read_text(encoding="utf-8"))
    return {_flat(str(name).split(".")[-1]) for name in declared}


def _touches_the_ts_side(node):
    """Whether this function calls a borch.ts handle **even once.**

    **The name does not spell borch.ts in underscore form.** Python cannot use a dot, so the
    temptation is to write it that way — but that spelling is the retired Python package's
    name, and the rename tool turns it into the binding's name. It happened once, and a
    function asking whether the other side is called came to read as "whether the binding is
    called". **The code changes consistently so nothing breaks, and what is wrong is only the
    meaning, so no check cries.** The same story is in `rename.py`'s comments.

    A call is usually a rename or a combination — `scatter` is `scatterSet` over there and
    `take` is `indexSelect`. Those places exist in TypeScript **under a different spelling**
    and are not missing. Never calling means Python is doing the arithmetic alone, and that
    is when the other side most likely has no such capability at all.

    **A classification, not a verdict.** `lerp` is written with `+`, `-` and `*` only so no
    handle is visible, while a `lerpFrom` with a different signature exists over there; and
    `ldl_factor_ex` called a handle and was missing anyway. So this split only sets the order
    a person reads in.
    """
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call) and getattr(sub.func, "id", "") == "handle":
            return True
        if isinstance(sub, ast.Attribute) and sub.attr in ("numpy", "handle"):
            return True
    return False


def _binding_bodies():
    """name → whether it calls a borch.ts handle."""
    out = {}
    for stem in ("_ops", "_base", "_nn", "_data", "_optim"):
        path = ROOT / "borch_webgpu" / f"{stem}.py"
        if not path.exists():
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                out.setdefault(node.name, _touches_the_ts_side(node))
    return out


def _binding_names():
    """The names the binding implements — module-level functions and `Tensor` methods."""
    names = set()
    for stem in ("_ops", "_base", "_nn", "_data", "_optim"):
        path = ROOT / "borch_webgpu" / f"{stem}.py"
        if not path.exists():
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if not _PRIVATE.match(node.name):
                    names.add(node.name)
    return names


def _asked_by_golden():
    """The names the case table calls **by that exact spelling.**

    A name the golden cases never ask about is outside this check's interest — the binding
    filling that one in deceives no table. It is simply absent everywhere, and that is a
    different story. Only **the places that manufacture green** stay here.

    **It reads the calls, not the characters.** This used to grep the raw text for
    `name(`, which also matched prose: a comment reading "moves with the seed (measured…)"
    put `seed` into this set, and `seed` is a torch name the binding has and borch.ts does
    not, so the check reported a filling-in that no case performs. It surfaced while
    `cases.py` was being translated, which is the fourth time in this repository that
    editing a file has changed what a check greps out of it.

    **Attribute references count, not only calls.** Taking calls alone dropped `values`,
    which the cases reach as `x.mode().values` — a name that is asked and never called. That
    turned a live row of `FILLED_ON_PURPOSE` into a stale one, which is the opposite error and
    just as quiet. Both are collected.

    Parsing costs nothing here and the result stays a subset — 1,078 real references against
    1,248 text matches, the extra 170 being words in comments, local helper names and keywords
    like `and` and `None`. Nothing a case actually asks about is lost.
    """
    tree = ast.parse((ROOT / "tests" / "cases.py").read_text(encoding="utf-8"))
    asked = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                asked.add(func.id)
            elif isinstance(func, ast.Attribute):
                asked.add(func.attr)
        if isinstance(node, ast.Attribute):
            asked.add(node.attr)
    return asked


def _candidates():
    ts = _ts_surface()
    asked = _asked_by_golden()
    found = set()
    for name in _binding_names():
        if not hasattr(torch, name) and not hasattr(torch.Tensor, name):
            continue                      # not a torch name, so not a candidate
        if _flat(name) in ts:
            continue
        if name not in asked:
            continue
        found.add(name)
    return found


def test_binding_does_not_quietly_fill_in():
    """**The only places the binding fills in should be the ones written in the table.**

    A new name blowing up here goes one of two ways. If it is worth putting into borch.ts,
    put it there — the golden cases start asking about that name. If it is decided against,
    write it into `FILLED_ON_PURPOSE` **with a reason.** Merely raising the number is the
    same as switching this check off.
    """
    bodies = _binding_bodies()
    surprise = sorted(_candidates() - set(FILLED_ON_PURPOSE))
    alone = [n for n in surprise if not bodies.get(n)]
    via = [n for n in surprise if bodies.get(n)]
    assert not surprise, (
        "the binding fills in on borch.ts's behalf — the golden cases pass through the "
        f"binding and cannot see this:\n\n  Python does the arithmetic alone ({len(alone)}):\n    "
        + "\n    ".join(alone)
        + f"\n\n  assembles while calling handles ({len(via)}):\n    " + "\n    ".join(via))


def test_the_table_has_no_stale_rows():
    """**A row fully carried across has to be deleted.** Left in, the next person reads it as still absent."""
    gone = sorted(set(FILLED_ON_PURPOSE) - _candidates())
    assert not gone, (
        "these names are in borch.ts now — delete them from the table:\n  " + "\n  ".join(gone))
