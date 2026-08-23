"""One reader for argument lists, because three copies had the same bug in two of them.

    from signature_read import parameters, VARIADIC

Three checks in this repository compare argument lists, on three different pairs:

    tests/test_torch_signatures.py   borchvision  ↔ real torchvision
    tests/ts_signatures.py           core         ↔ borch.ts
    tests/torch_signatures_core.py   core         ↔ real torch

Each grew its own reader, and **the same defect was sitting in two of them at once,
put there by two people who were not reading each other's file.** Both dropped
`*args`/`**kwargs` and compared the remainder as though it were the whole list.

The two failures were not equally visible, and the difference is the reason this
module exists rather than a shared style guide.

- In `ts_signatures.py` it produced **nine loud rows pointing at the wrong library**.
  Nine core loss constructors are written `(*args, reduction='mean', **kw)`, so the
  remainder was `[reduction]` and each read as borch.ts having inserted arguments in
  front. borch.ts had them because torch has them.
- In `test_torch_signatures.py` it produced **a pass**. `AutoAugmentPolicy` and
  `InterpolationMode` are Enums with no `__init__`, so `object.__init__` showed
  `(*args, **kwds)`; the exclusion list said `args`/`kwargs` and Python writes
  `kwds`, so both sides came back as `['kwds']` and agreed with each other. A green
  check comparing nothing, agreeing on the name its own filter failed to remove.

The second is the worse one and it is worth naming: **the filter's incompleteness
became the thing the two sides agreed on.** Nothing was measured, nothing was
skipped, and the row read as coverage.

## The rule

A signature containing `*args` or `**kwargs` is **not a short signature**. It is a
signature that cannot be compared, and it says so by returning `VARIADIC`. Whether
that is an Enum with no `__init__` or a wrapper that swallows everything, this module
does not know and does not guess — the caller decides what an uncomparable pair
means, and every caller is expected to *count* them rather than let them vanish.
"""

import inspect

# What a variadic signature returns. A distinct object rather than `None`, so that
# "there is no signature at all" and "the signature cannot be compared" stay apart —
# they are different findings and a caller that lumps them says less than it could.
VARIADIC = object()

# The receiver, by name. Kept for the class-method path, where a function reached
# through its class carries the receiver first whatever it is called; `parameters()`
# takes `receiver=True` for that case and does not rely on the spelling.
_RECEIVER_NAMES = ("self", "cls")


def parameters(fn, receiver=False):
    """`[name, ...]` in order · `VARIADIC` · `None` when there is no signature.

    `receiver=True` drops the first parameter **by position rather than by name**.
    A function reached through its class carries the receiver first however it is
    spelled, and the core does not always spell it `self`: a tally of same-position
    name pairs once came back led by `t → dim` twenty-eight times, which was not
    twenty-eight renamed parameters but twenty-eight lists off by one.

    Keyword-only parameters are kept and are **not marked** in this list — see
    `positional()` below, which is where the distinction lives now that a caller has
    needed it.
    """
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return None
    got = list(sig.parameters.values())
    if any(p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD) for p in got):
        return VARIADIC
    names = [p.name for p in got]
    if names and (receiver or names[0] in _RECEIVER_NAMES):
        names = names[1:]
    return names


def positional(fn, receiver=False):
    """`parameters()` cut where the caller can no longer reach by position.

    **A shift the language forbids is not a shift.** The three axes all have a
    bucket for *an argument sits where the other side has a different one, so a
    positional call lands on the wrong parameter* — and every one of them was
    deciding it from names in order, with no way to ask whether a positional call
    was possible at all.

    `optim.Adagrad` is what that cost. torch writes it

        (params, lr, lr_decay, weight_decay, initial_accumulator_value, eps,
         foreach, *, maximize, differentiable, fused)

    and the core stops after `eps` with its own `*, maximize`. As names in order,
    the core's `maximize` sits in torch's `foreach` seat and the row read as an
    inserted argument — the sharpest bucket there is. Neither `maximize` can be
    passed positionally by anybody, in either library. The row described a call
    that cannot be written.

    It is wrong in the other direction too, and that half is worse: two lists that
    disagree only past their positional prefixes are **safe**, and two that agree on
    names while one takes fewer of them positionally are **not** — and a reader
    without `kind` calls the first dangerous and the second identical. The bucket was
    not making mistakes at its edges; it was answering a different question from the
    one its name asks, everywhere, including in all 109 rows it calls `agree`.

    Returns `VARIADIC` and `None` exactly as `parameters()` does, so a caller can use
    the two together without a second set of cases.
    """
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return None
    got = list(sig.parameters.values())
    if any(p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD) for p in got):
        return VARIADIC
    names = [p.name for p in got if p.kind is not p.KEYWORD_ONLY]
    if names and (receiver or names[0] in _RECEIVER_NAMES):
        names = names[1:]
    return names


def of_class(cls, reach=False):
    """The constructor's arguments — `cls.__init__` without its receiver.

    `reach=True` asks `positional()` instead, giving the prefix a caller can reach
    by position. Most of the classes on these axes are optimisers and schedulers,
    where the keyword-only tail is exactly where torch puts the switches nobody
    passes positionally.

    Asked of `__init__` rather than of the class, because a class with no `__init__`
    of its own inherits `object.__init__`, whose `(*args, **kwargs)` is exactly the
    variadic case above and must reach the caller as `VARIADIC` rather than as an
    empty list. `inspect.signature(cls)` hides that behind `()`.
    """
    return (positional if reach else parameters)(cls.__init__)
