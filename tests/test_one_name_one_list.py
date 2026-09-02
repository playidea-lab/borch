"""One function may stand under two torch names only if torch gives the two one list.

## The shape, four times

    linalg.norm     → torch.norm      ord vs p, and a different formula for ord=2
    linalg.svd      → torch.svd       Vh vs V, full vs reduced
    linalg.pinv     → pinverse        (A, *, atol, rtol, hermitian) vs (input, rcond)
    special.softmax → F.softmax       (input, dim, dtype) vs (input, dim=None, _stacklevel, dtype)

Each was `staticmethod(<the other one>)`. Each time the binding accepted what one of
torch's two names refuses — `special.softmax(x, 1, _stacklevel=4)` ran here — or
refused what one accepts. Four is a pattern, and the four were found by four separate
sweeps. This holds the rule instead.

## What is compared

Every public callable in every namespace is keyed by the function object underneath
(`inspect.unwrap`, so `_accepts_out` and the like do not split one function into two).
For an object bound under two or more torch names, torch's **documented** argument
list is read for each name — the first line of the docstring, which is where torch
writes a C function's signature — and the lists are compared after two normalisations:

- `out` is dropped: keyword-only everywhere and never the difference.
- the **first** name is dropped: torch calls the operand `input` in one namespace and
  `A` in the next, and that rename is filed elsewhere.

Order is kept after that, because a shift is exactly what a positional call trips on.

## What it does not compare

`Tensor` is out. Its methods drop the receiver, so `Tensor.add` documents `(other, *,
alpha)` against `torch.add`'s `(input, other, *, alpha)` and every method would read as
a finding. The receiver convention is a rename the whole axis already knows about.

A name whose docstring cannot be read (prose, `*args`, a deferral) is skipped for that
binding only — a comparison this file cannot make is not reported as agreement.
"""

import inspect
import pathlib
import sys

import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import borch                                                     # noqa: E402
import signature_read                                            # noqa: E402

SPACES = [
    ("torch", borch, torch),
    ("nn.functional", borch.nn.functional, torch.nn.functional),
    ("linalg", borch.linalg, torch.linalg),
    ("fft", borch.fft, torch.fft),
    ("special", borch.special, torch.special),
]

# A pair torch gives two lists and this library deliberately serves with one function.
# Each row carries the reason; an empty table is the end state.
# **Two where torch's prose parts and torch's runtime does not.** Both measured on
# 2026-09-02 against torch itself; both are torch's docstring being shorter than its
# parser, the same shape as `tol`/`rcond` in `test_torch_signatures_core.py`. An entry
# here says the runtimes agree, not that the lists do — and the golden asks each
# name with the other's spelling, so the agreement is measured, not attested.
ATTESTED = {
    # `concatenate` documents `axis=0`; `cat` and `concat` document `dim=0`. All three
    # take both spellings on torch (`cat([x, x], axis=0)` runs), and now here.
    (("torch", "cat"), ("torch", "concat"), ("torch", "concatenate")):
        "torch's parser takes `axis` and `dim` on all three — measured; the golden "
        "asks `axis=` on each",
    # `special.round`'s docstring is `round(input, *, out=None)` plus "Alias for
    # torch.round" — and it takes `decimals=` like the function it aliases.
    (("special", "round"), ("torch", "round")):
        "`special.round(x, decimals=3)` runs on torch and positional `2` stops on "
        "both names — measured; the golden asks both",
}


def _underlying(obj):
    """The function object a namespace attribute stands for, or None."""
    if isinstance(obj, staticmethod):
        obj = obj.__func__
    if inspect.isclass(obj) or not callable(obj):
        return None
    try:
        return inspect.unwrap(obj)
    except ValueError:
        return obj


def _torch_list(theirs, name):
    """torch's documented list for `theirs.name`, normalised, or None if unreadable."""
    fn = getattr(theirs, name, None)
    if fn is None:
        return None
    got = signature_read.from_docstring(fn, name)
    if got is None or got is signature_read.VARIADIC:
        got = signature_read.parameters(fn)
    if not isinstance(got, list):
        return None
    names = [p for p in got if p != "out"]
    return tuple(names[1:])


def _bindings():
    """`function → [(space, name), …]` for every public callable in every namespace."""
    by_function = {}
    for space, ours, theirs in SPACES:
        for name in dir(ours):
            if name.startswith("_") or not hasattr(theirs, name):
                continue
            fn = _underlying(inspect.getattr_static(ours, name, None)
                             or getattr(ours, name, None))
            if fn is None:
                continue
            by_function.setdefault(fn, []).append((space, name))
    return by_function


def test_a_function_under_two_names_has_one_list_in_torch():
    findings = []
    for fn, bound in _bindings().items():
        if len(bound) < 2:
            continue
        lists = {}
        for space, name in bound:
            theirs = next(t for s, _o, t in SPACES if s == space)
            got = _torch_list(theirs, name)
            if got is not None:
                lists[(space, name)] = got
        if len(set(lists.values())) <= 1:
            continue
        key = tuple(sorted(lists))
        if key in ATTESTED:
            continue
        findings.append(
            "  " + "  vs  ".join(f"{s}.{n}({', '.join(l) or '…'})" for (s, n), l in
                                 sorted(lists.items())))
    assert not findings, (
        "one function stands under torch names whose documented lists differ:\n"
        + "\n".join(findings)
        + "\n\n  Split the binding — give the namespace its own function with torch's list\n"
          "  for that name. This is the shape `linalg.norm`, `linalg.svd`, `linalg.pinv`\n"
          "  and `special.softmax` each turned out to be, and each time the shared\n"
          "  function accepted a call one of the two names refuses.")


def test_the_comparison_is_not_empty():
    """**A floor.** Every alias torch keeps — `special.expit` is `sigmoid`,
    `swapaxes` is `transpose` — is a function under two names, so the map has to be
    populated or this file is measuring nothing."""
    shared = [b for b in _bindings().values() if len(b) >= 2]
    assert len(shared) >= 20, (
        f"only {len(shared)} functions are bound under two or more names — the walk\n"
        "  found almost nothing. Check `_bindings()` before trusting the test above.")
