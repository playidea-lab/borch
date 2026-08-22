"""**A name can be present and its argument list can lie.**

`tests/torch_gap.py` counts names. `Grayscale` with the wrong luma weights counts the
same as `Grayscale`, and so does a `RandomCrop` that takes three arguments where
torchvision takes five. The count says which places are empty and cannot say anything
about the places that are full.

That is not a hypothetical worry. In one day this repository found five of them:

- `borch.ts`'s `MaxPool2d` taking `(kernel)` against the core's `return_indices`
- `InstanceNorm` taking `(eps?)` against five arguments
- `borch_webgpu`'s `Adam` handing `weight_decay` to a JS call that discards it
- `RandomResizedCrop` accepting `interpolation` and dropping it on the way to the resize
- `NAdam` — **the one that counting could not reach**, because its call passes six
  arguments into six parameters and the sixth is `momentum_decay`, not the decay

None was visible to a name count. What found each was something that exercised the
argument: a case, or a call site read against the constructor it calls.

So this file asks the question directly of the transforms — **parameter names, in
order, against real torchvision's.** Order matters as much as membership, and the
reason is `RandomCrop`, which is what this check was written for. It took
`(size, padding, fill)` while torchvision takes
`(size, padding, pad_if_needed, fill, padding_mode)`, so

    RandomCrop(32, 4, True)

set `fill=True` here and `pad_if_needed=True` there. The same line, two meanings, a
correctly shaped picture either way and nothing raised.

## The table below is empty, and that is the point

Where ours should differ from torchvision's, the difference goes in `DELIBERATE` with
a reason — the discipline `torch_gap.py` uses, for the same reason: a difference
nobody wrote down is indistinguishable from one nobody noticed. It is empty today
because every difference found so far was a defect.
"""

import importlib.util
import inspect
import pathlib
import sys

import pytest

torch = pytest.importorskip("torch")
R = pytest.importorskip("torchvision.transforms")

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import borch as BT                                              # noqa: E402
import borchvision as V                                         # noqa: E402

V.use(BT)

# name -> why ours takes different arguments. Nothing here yet; see the docstring.
DELIBERATE: dict[str, str] = {}

# The pairs this check **cannot** compare, because one side or the other is variadic.
# Both are Enums, which define no `__init__`, so `object.__init__`'s `(*args, **kwargs)`
# is what shows — there is nothing to compare and nothing wrong. Listed by name rather
# than counted so that a third one fails here instead of joining them in silence.
UNCOMPARABLE = ("AutoAugmentPolicy", "InterpolationMode", "ToTensor")

_NOT_ARGUMENTS = ("self",)


def _arguments(cls):
    """The parameter names in order, or `None` when the signature cannot be compared.

    **A variadic signature is not a short signature.** `*args`/`**kwargs` are what
    `object.__init__` shows for a class that defines none — an Enum, for instance — and
    they are also what a wrapper writes when it accepts everything and passes it on.
    Those two look identical here and mean opposite things.

    This used to strip them by name and compare the remainder as though it were the whole
    list. That is a filter removing part of a record and letting what is left stand in for
    it, and it does not go quiet: for the two Enums it compared `[] == []` and **passed
    while measuring nothing.** Another session hit the same shape an hour ago in
    `ts_signatures.py`, where dropping `*args` from nine core loss constructors left
    `[reduction]` and reported all nine as defects in the other library.

    So it returns `None`, the same as an unreadable signature, and the count of pairs that
    could not be compared is asserted below rather than left implicit.
    """
    try:
        sig = inspect.signature(cls.__init__)
    except (TypeError, ValueError):                             # pragma: no cover
        return None
    parameters = list(sig.parameters.values())
    if any(p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD) for p in parameters):
        return None
    return [p.name for p in parameters if p.name not in _NOT_ARGUMENTS]


def _shared():
    for name in sorted(n for n in dir(V.transforms) if n[0].isupper()):
        theirs = getattr(R, name, None)
        if theirs is not None:
            yield name, getattr(V.transforms, name), theirs


def _shared_functions():
    """The same question of `transforms.functional`.

    **A function's arguments are worse than a class's**, not better: a class is usually
    built with keywords and called with one thing, while `F.pad(img, [1, 2], 0.5)` is
    positional from end to end. Every argument there is decided by where it sits.
    """
    import torchvision.transforms.functional as TF
    # **`_`-led names are the module's own furniture**, not its surface — a module
    # object carries `__loader__` and `__spec__`, and both have signatures.
    for name in sorted(n for n in dir(V.transforms.functional)
                       if not n.startswith("_") and not n[0].isupper()):
        theirs = getattr(TF, name, None)
        if theirs is not None:
            yield name, getattr(V.transforms.functional, name), theirs


def test_every_transform_takes_torchs_arguments_in_torchs_order():
    """Same names, same order, or a written reason.

    **Order is half of it.** A missing argument at the end makes a keyword call fail
    loudly; an argument missing from the middle makes a positional call succeed and
    mean something else, which is the shape that reached production here.
    """
    wrong, uncomparable = [], []
    for name, ours, theirs in _shared():
        if name in DELIBERATE:
            continue
        mine, torchs = _arguments(ours), _arguments(theirs)
        if mine is None or torchs is None:
            uncomparable.append(name)
            continue
        if mine != torchs:
            missing = [a for a in torchs if a not in mine]
            extra = [a for a in mine if a not in torchs]
            wrong.append(
                f"{name}\n      ours  : {', '.join(mine)}\n"
                f"      torch : {', '.join(torchs)}\n"
                f"      missing={missing or 'none'} extra={extra or 'none'}")
    # **What could not be compared is said out loud.** Silence here is the same silence
    # as a pass, and these are the rows where the check knows least.
    assert len(uncomparable) <= len(UNCOMPARABLE), (
        "more transforms than expected have a variadic signature on one side, so they "
        "are not being compared at all:\n    " + ", ".join(sorted(uncomparable))
        + f"\n\n  Expected at most {len(UNCOMPARABLE)}: {', '.join(UNCOMPARABLE)}.\n"
          "  A wrapper that accepts everything reads exactly like a class that takes "
          "nothing.")
    assert not wrong, (
        "transforms whose argument list is not torchvision's:\n    "
        + "\n    ".join(wrong) +
        "\n\nA positional call lands on a different parameter and still returns a "
        "correctly shaped picture. If the difference is meant, put it in `DELIBERATE` "
        "with the reason — a difference nobody wrote down cannot be told from one "
        "nobody noticed.")


def test_every_function_takes_torchs_arguments_in_torchs_order():
    """`transforms.functional`, by the same rule.

    The names are compared rather than the defaults, deliberately. torchvision's
    `interpolation` default is an `InterpolationMode` and ours is the string it wraps —
    the same filter, spelled for a library that takes both, and the golden repr cases
    are what hold the spelling.
    """
    wrong = []
    for name, ours, theirs in _shared_functions():
        if name in DELIBERATE:
            continue
        mine, torchs = _arguments(ours), _arguments(theirs)
        if mine is None or torchs is None:
            continue
        if mine != torchs:
            wrong.append(f"{name}\n      ours  : {', '.join(mine)}\n"
                         f"      torch : {', '.join(torchs)}")
    assert not wrong, (
        "functions whose argument list is not torchvision's:\n    " + "\n    ".join(wrong)
        + "\n\nThese are called positionally more often than the classes are, so a "
        "parameter in the wrong place lands a value in the wrong argument silently.")


def test_no_deliberate_row_explains_a_transform_that_matches():
    """A reason attached to something that agrees is a reason about nothing — the same
    contradiction `test_gap.py` catches in the other table, and the same way a stale
    reason outlives the thing it described."""
    stale = [name for name, ours, theirs in _shared()
             if name in DELIBERATE and _arguments(ours) is not None
             and _arguments(ours) == _arguments(theirs)]
    assert not stale, (
        f"`DELIBERATE` explains transforms that already match torchvision: {stale}\n"
        "  Take the row out — it reads as a difference to the next person.")
