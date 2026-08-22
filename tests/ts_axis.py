"""Counts the names the **core** has that **borch.ts** does not, and the reverse.

    uv run --with numpy --with torch --with torchvision python tests/ts_axis.py
    uv run --with numpy --with torch --with torchvision python tests/ts_axis.py --show nn

## The axis nothing was looking along

Three files already measure this repository's surface, and none of them measures this
pair:

    tests/torch_gap.py           core Python  ↔ real torch          names
    tests/test_torch_signatures  borchvision  ↔ real torchvision    parameters, in order
    tests/test_binding_arguments borch_webgpu ↔ borch.ts            parameters, in order

The core and borch.ts are **two independent implementations of the same surface** —
one over numpy, one over WGSL — and the golden holds their *values* against each
other. What no check holds is their *names*. A name the core has and borch.ts does
not is not a wrong answer anywhere; it is a line of tutorial code that runs in one
and raises in the other, and nothing goes red.

That state was real. Measured by hand in one session, seventeen `nn` names stood in
the core and not in borch.ts, and every golden case was green throughout — because a
case can only ask about a name somebody wrote a case for.

## What this counts, and the two things it cannot

It counts **names**, from the same enumeration `torch_gap.py` uses on the Python side
and from the generated index on the TypeScript side.

It cannot see a **signature**. `MaxPool2d` present in both, taking `(kernel)` here and
`(kernel, stride, padding, dilation, returnIndices)` there, counts as agreement — the
exact defect `test_torch_signatures.py` was written for, on the axis it does not
cover. Five of those were found in one day and none was visible to a count.

It cannot see a **value**. That is the golden's job, and the golden is why the two
implementations agree at all.

So a green run of this file says *the same names exist on both sides*. It does not say
they mean the same thing. Reading it as coverage is reading it as a sentence it does
not support.

## Why the TypeScript side is read from the index rather than the source

`site/assets/api-index.json` is generated from the `.d.ts` files, which is what a
consumer of the package actually gets. Reading `src/*.ts` instead would count names
that never reach a user, and reading the case table would count only what has a case.

**That makes this measurement a build artefact, and a stale one lies in our favour** —
names added since the last build read as absent from borch.ts, which is the same
sentence as a real gap. `test_ts_axis.py` refuses to run against a stale bundle for
that reason, the same rule `test_site.py` applies to its counts.

## Filling this in is the same dangerous work `torch_gap.py` describes

Every name written into `DELIBERATE` below raises the agreement figure, so the work
slides towards making the number look good. The rule is the one that file uses:
**every row carries a reason, and a row whose reason cannot be written is a gap.**
"""

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

INDEX = ROOT / "site" / "assets" / "api-index.json"

# Which Python namespaces have a borch.ts side to compare against.
#
# **`torch` — the top level — is not on this list, and that is a finding rather than
# an omission.** The first version measured it and reported 202 core-only names.
# Reading them showed the question does not have an answer: `borch/__init__.py`
# mirrors *every* `Tensor` method into module scope (`for _name in dir(Tensor)`), the
# way real torch offers `torch.numel(x)` beside `x.numel()`. borch.ts does not — its
# methods live on the class and its `index` module carries two names. So comparing the
# two top levels counts every method a second time and answers a question about
# module structure while looking like one about features.
#
# The methods themselves are still compared: they are the `Tensor` row.
#
# `transforms` and `transforms.functional` are `borchvision`'s. Their TypeScript side
# is `vision.ts` and the golden's `vision::` cases hold it name by name, so measuring
# it here would ask the same question twice.
SPACES = frozenset({
    "Tensor", "nn", "nn.functional",
    "optim", "optim.lr_scheduler", "linalg", "utils.data",
})


def refused():
    """Names the core carries **only in order to refuse them.**

    `borch/_tensor.py` binds a stub for each: `to_mkldnn` raises "MKL-DNN is not in
    the browser subset", `symeig` says torch removed it and points at `linalg.eigh`.
    They are the design principle working — *an absent feature beats a wrong answer* —
    and they are not features.

    borch.ts not carrying the stub is therefore **not a missing feature**. It is a
    worse error message: `x.to_mkldnn()` says "not a function" there and says what is
    wrong and why here. Worth fixing one day, and not the same list as `maximum`.

    **Asked of the bound method, not of a table.** The first version imported
    `_GONE`, `_NO_MACHINERY` and `_ABSENT_DTYPES` and found fourteen. Reading the
    source showed why that was too few: as many again are bound by **inline loops**
    with no table to import — sparse, quantisation, storage, the four accelerators.
    A list of table names would have gone on missing those and the miss would have
    been silent, because a refusal counted as a gap looks exactly like a gap.

    So the marker is where the function came from. Every stub is built by one of a
    few factories and keeps that factory in its `__qualname__`; nothing else in the
    core does. If that ever stops being true this raises rather than quietly
    reporting zero — a rule that misses in our favour is worse than no rule, which is
    `torch_gap.py`'s sentence and the reason this one is written to fail closed.
    """
    from borch import _tensor

    factories = ("_bind_gone", "_bind_absent", "_needs_sparse", "_bind_absent_dtype")
    out = {name for name in dir(_tensor.Tensor)
           if not name.startswith("_")
           and getattr(getattr(_tensor.Tensor, name, None), "__qualname__", "")
           .startswith(factories)}
    # `_bind_absent_dtype` rewrites its own `__qualname__`, so its names are read
    # from the one table that does exist. Both paths are kept because either alone
    # was measured to be short.
    out |= set(_tensor._ABSENT_DTYPES)
    if len(out) < 20:
        raise SystemExit(
            f"only {len(out)} refusal stubs found in borch/_tensor.py — the factory "
            "names above have probably changed. Fix them rather than letting "
            "refusals be counted as gaps.")
    return out

# **Names the core has and borch.ts is not going to.** Each row is a judgement and
# carries its reason; a name absent from both this table and borch.ts is the to-do
# list. Keyed by `space::name` so that a reason about one name cannot excuse a whole
# namespace — the shape `test_binding_arguments.py` found by keying its own table the
# wrong way first.
#
# **Every row here raises the agreement figure**, so the work slides towards writing
# rows. The rule is `torch_gap.py`'s: a reason has to be checkable, and one that
# cannot be written is a gap. Each row below cites where its reason is already
# established in this repository rather than asserting it fresh.

_ALIAS = "torch's own alias for {}; borch.ts carries the one name"
_DEVICE = ("one device in a browser — borch/__init__.py's `_NOT_OURS` says the same "
           "for `cpu`: there is one device to choose, so we have none")
_STORAGE = ("no storage layer to look into — borch/__init__.py's `_NOT_OURS`, and "
            "the core answers `.storage()` with 'use `.numpy()`'")
_NUMPY = "the numpy bridge; borch.ts has no numpy on the other side of it"

DELIBERATE: dict[str, str] = {
    # torch keeps both spellings of six arcs and four arithmetic names. The core
    # mirrors torch, borch.ts carries the primary — a learner who types the alias
    # gets an error rather than a different answer, which is this project's rule.
    **{f"Tensor::{n}": _ALIAS.format(p)
       for n, p in (("arccos", "acos"), ("arccos_", "acos_"),
                    ("arccosh", "acosh"), ("arccosh_", "acosh_"),
                    ("arcsin", "asin"), ("arcsin_", "asin_"),
                    ("arcsinh", "asinh"), ("arcsinh_", "asinh_"),
                    ("clip", "clamp"), ("divide", "div"), ("subtract", "sub"),
                    ("ndimension", "dim"), ("nelement", "numel"),
                    ("swapdims", "transpose"), ("swapdims_", "transpose_"))},
    # There is one device and no storage objects. Both reasons are the core's own,
    # written into `_NOT_OURS` when the same question was asked of torch.
    **{f"Tensor::{n}": _DEVICE
       for n in ("get_device", "is_pinned", "pin_memory", "share_memory_",
                 "is_shared", "record_stream", "is_distributed")},
    **{f"Tensor::{n}": _STORAGE
       for n in ("data_ptr", "const_data_ptr", "storage_offset", "element_size",
                 "is_set_to", "dim_order", "untyped_storage")},
    # The numpy bridge, which has nothing on the far side in TypeScript.
    **{f"Tensor::{n}": _NUMPY for n in ("numpy", "tolist", "new_tensor")},
    # Worker processes do not exist in a browser, which is `run.py`'s stated reason
    # for the two `dataconv::` cases it leaves unported.
    "utils.data::get_worker_info":
        "no worker processes in a browser — borch-ts/test/run.py says the same",
    "utils.data::default_convert":
        "converts numpy and Python containers into tensors; neither is on the "
        "TypeScript side of the bridge",
}

# **Names borch.ts has and the core does not.** The reverse direction is not
# symmetric: borch.ts is a browser library and carries things a numpy core has no
# reason to (`init`, `device`, `keepAlive`, `scope`). Left empty until measured —
# writing rows here before running it would be inventing the answer.
EXTRA: dict[str, str] = {}


def ts_names():
    """Every name in the generated index, as one set.

    **The index records one home per name and the comparison must not use it.** It is
    `{name: "module.path.name"}`, and `det` is recorded as `tensor.Tensor.det` — the
    method — not as `linalg.det`. Asking "is `det` in the `linalg` module of the
    index" therefore answers no about a name that is present.

    The first version of this file did exactly that and reported **1,137 core-only
    names**, of which nearly all were a name filed under a different home. A number
    that large reads as a finding; it was a mapping error, and it was caught by the
    one namespace whose whole content came back missing at once.

    So membership is asked of the **whole surface**: does borch.ts have this name
    anywhere. What that gives up is "and in the right namespace", which the index
    cannot answer — said here rather than left for the next reader to assume.
    """
    if not INDEX.exists():
        raise SystemExit(f"no {INDEX.relative_to(ROOT)} — run npm run docs:api first")
    raw = json.loads(INDEX.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not raw:
        raise SystemExit(f"{INDEX.relative_to(ROOT)} is not the expected name → path map")
    return set(raw)


def _camel(name):
    """`return_indices` → `returnIndices`. **Only the spelling, not a translation.**

    borch.ts is camelCase and the core is snake_case, so comparing them raw reports
    every multi-word name as missing on both sides at once. That is not a gap; it is
    two spellings of one name. A name with no underscore passes through unchanged,
    which is most class names.

    **A trailing underscore is kept, and the first version dropped it.** torch marks
    in-place with it — `eq_` writes into the receiver where `eq` returns a new
    tensor, and they are two functions rather than two spellings. Splitting on `_`
    naively gives `eq_` an empty last part, so it camelled to `eq` and matched the
    out-of-place method.

    That was caught by the number improving too much: adding six comparison methods
    dropped the count by twelve. `torch_gap.py` says a rule that misses in our favour
    is worse than no rule, and this one missed in our favour by exactly the six
    in-place forms nobody had written.
    """
    trailing = "_" if name.endswith("_") else ""
    head, *rest = name.rstrip("_").split("_")
    return head + "".join(p[:1].upper() + p[1:] for p in rest) + trailing


def _folds_onto(name, lowered):
    """Whether `name` matches something in borch.ts once case is set aside.

    **Needed, because `_camel` cannot bridge every spelling.** The core writes
    `tensorinv` and `tensorsolve` with no underscore at all, and borch.ts writes
    `tensorInv` and `tensorSolve`; there is nothing for a split-on-underscore rule
    to work with. Twelve names sit in that shape and dropping the fold invents
    twelve gaps that are not there — measured.

    **But the first letter is kept, and that is the whole care in this function.**
    In torch, an initial capital is the class/function boundary: `nn.Embedding` is
    a layer and `nn.functional.embedding` is a function, and they are not each
    other. Folding both would report the layer as present because the function is,
    which is a normaliser erasing a distinction the domain makes — the same fault
    as `_camel` stripping the in-place underscore, one line down and pointing the
    other way.

    So: any normaliser is a claim about which differences do not matter. This one
    claims that internal capitalisation does not and that the first letter does.
    """
    if not name:
        return False
    if name[0].isupper():                        # a class; the capital is the name
        return False
    return name.lower() in lowered


def compare():
    """`{space: (gaps, refusals)}`, with the two spellings reconciled.

    Both are core-only. They are returned apart because they are different work: a
    gap wants the feature written, a refusal wants the message carried across.
    """
    import torch_gap

    theirs = ts_names()
    lowered = {n.lower() for n in theirs}
    stubs = refused()
    out = {}
    for space, _real, ours in torch_gap._spaces():
        if space not in SPACES:                  # borchvision's spaces are held elsewhere
            continue
        missing = [n for n in torch_gap._public(ours)
                   if _camel(n) not in theirs and not _folds_onto(n, lowered)]
        out[space] = (sorted(n for n in missing if n not in stubs),
                      sorted(n for n in missing if n in stubs))
    return out


def main(argv):
    show = argv[argv.index("--show") + 1] if "--show" in argv else None
    rows = compare()
    unexplained = 0
    stubs = 0
    for space, (gaps, refusals) in rows.items():
        loose = [n for n in gaps if f"{space}::{n}" not in DELIBERATE]
        unexplained += len(loose)
        stubs += len(refusals)
        mark = " " if not loose else "✘"
        print(f"  {mark} {space:22s} gap {len(gaps):>4}  "
              f"without a reason {len(loose):>4}   refusal stub {len(refusals):>4}")
        if show is not None and space.startswith(show):
            for name in gaps:
                why = DELIBERATE.get(f"{space}::{name}", "**no reason**")
                print(f"      · {name}  {why}")
            for name in refusals:
                print(f"      · {name}  (the core only refuses it too)")
    print(f"\n이름만 센다 — 서명도 값도 안 본다.")
    print(f"까닭 없는 결손 {unexplained}건 · 거절 스텁 {stubs}건.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
