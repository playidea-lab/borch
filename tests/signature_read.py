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
import re

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


def _close(text, start):
    """The index of the `)` that closes the `(` at `start`, or -1."""
    depth = 0
    for i in range(start, len(text)):
        if text[i] in "([{":
            depth += 1
        elif text[i] in ")]}":
            depth -= 1
            if depth == 0:
                return i
    return -1


def _split_top(inner):
    """Split on commas that are not inside brackets. `Union[int, str]` stays whole."""
    parts, depth, cur = [], 0, ""
    for ch in inner:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    parts.append(cur)
    return [p.strip() for p in parts if p.strip()]


def from_docstring(fn, name, positional_only=False):
    """The argument list torch writes in its **docstring**, when `inspect` has none.

    ## Why this exists

    `parameters()` returns `None` for anything implemented in C, and the core↔torch
    axis counted those into a bucket labelled *torch is C*. It stood at **571** —
    larger than every other bucket on that axis put together, and larger than the
    177 the axis called `agree`.

    A bucket that size is not a footnote. It says *nothing was asked about most of
    the surface*, and it is easy to read a green axis as though it covered the
    library it names.

    **It is not empty, and that was found by accident.** A probe built to ask which
    input *ranks* each function accepts turned up two absences — `where`'s
    one-argument form and `nonzero(as_tuple=)` — and both were sitting in this
    bucket, unreachable by the axis whose whole job is missing arguments, while that
    axis reported 0 keyword-only absences. Two found by a probe that was not looking
    is not evidence of two.

    ## What it reads

    torch writes the signature as the first line of the docstring:

        nonzero(input, *, out=None, as_tuple=False) -> LongTensor or tuple of ...
        abs(input: Tensor, *, out: Optional[Tensor]) -> Tensor

    so the annotation and the default are both stripped, and the closing paren is
    found by **counting depth rather than by regex**. A greedy `.*)` runs past the
    parameters into a return tuple — `aminmax` documents `-> (Tensor min, Tensor
    max)` — and produced twelve rows naming `Tensor max` as a parameter. Twelve
    plausible-looking findings out of a bracket-matching mistake.

    ## What it is worth, said plainly

    **This is torch's prose, not torch's behaviour.** `inspect` reads the thing that
    runs; this reads what somebody wrote next to it, and the two can part — torch's
    `scalar_tensor` message says its argument "must be Number, not Tensor" and then
    accepts a 0-D tensor. A row from here is a lead worth checking against a call,
    not a measurement. It is still much better than not asking.

    Returns what `parameters()` returns: a list, `VARIADIC`, or `None`.
    """
    doc = (getattr(fn, "__doc__", "") or "").strip()
    if not doc:
        return None
    # **A `Tensor` method's docstring often is not the signature — it is a pointer.**
    #
    #     bitwise_and() -> Tensor
    #
    #     See :func:`torch.bitwise_and`
    #
    # and `torch.bitwise_and` documents `(input, other, *, out=None)`. Read where it
    # stands, the method looks as though it takes nothing, and every core method that
    # takes the operand came back as **the core having an argument torch does not** —
    # nine rows pointing at the wrong library, which is what `ts_signatures`'s reader
    # did once before for a different reason.
    #
    # So the deferral is followed. Only `:func:`, never `:meth:`: `atanh_` defers with
    # `In-place version of :meth:`~Tensor.atanh`` while its own first line reads
    # `atanh_(other)`, and `atanh_` takes no argument at all. **torch's docstring is
    # simply wrong there** — which is the standing caveat on this whole function
    # arriving on the first day it was used, and the reason a row from here is a lead
    # rather than a measurement.
    seen = _defers_to(doc)
    if seen is not None:
        target = getattr(_torch_root(fn), seen, None)
        if target is not None and target is not fn:
            got = from_docstring(target, seen, positional_only=positional_only)
            return DEFERRED(got) if isinstance(got, list) else got
    for line in doc.splitlines()[:3]:
        line = line.strip()
        if not line.startswith(name + "("):
            continue
        end = _close(line, len(name))
        if end < 0:
            continue
        names, seen_star = [], False
        for part in _split_top(line[len(name) + 1:end]):
            if part == "*":
                seen_star = True
                continue
            if part.startswith("**"):
                continue
            if part.startswith("*"):
                return VARIADIC
            if positional_only and seen_star:
                continue
            bare = part.split("=")[0].split(":")[0].strip()
            if not bare.isidentifier():
                # Not a parameter — a stray from a line this reader misread. Said
                # nothing rather than guessed: a wrong name here becomes a row.
                return None
            names.append(bare)
        return names
    return None


class DEFERRED(list):
    """A list read from **the module function a method's docstring points at**.

    It is a `list` and behaves as one, so a caller that does not care is unaffected.
    What the type adds is a warning that **the order is not to be trusted**, and
    that is not a caution — it is measured.

    The receiver is not reliably the first name. `Tensor.where` defers to
    `torch.where(condition, input, other)`, where the receiver is the *second*;
    `Tensor.triangular_solve` defers to `torch.triangular_solve(b, A, …)`, where it
    is the second again and neither is called `input`. Dropping the first name gives
    a list that is wrong by one, everywhere, in a way that looks exactly like the
    library having inserted an argument.

    And one docstring can hold several overloads. `torch.mean` documents
    `mean(input, *, dtype=None)` first and `mean(input, dim, keepdim=False, *,
    dtype=None)` below it; reading the first line alone says `mean` takes no `dim`.

    Between them those two produced **eight `shifted` rows in one run** — the
    sharpest bucket this axis has, every one of them false, on the first run after
    the deferral was followed. A rule that misses in our favour is worse than no
    rule; this one missed in the *alarming* direction, which is not better, only
    louder.

    So a caller with this type in hand is expected to compare **membership and not
    order** — which name is absent, never which seat it sits in. That keeps the half
    the prose can support and gives up the half it cannot.
    """


def _defers_to(doc):
    """The module-level name a ``See :func:`torch.X` `` docstring points at, or None."""
    m = re.search(r"See :func:`torch\.([A-Za-z_]\w*)`", doc)
    return m.group(1) if m else None


def _torch_root(fn):
    """The `torch` module, imported lazily so this file stays importable without it."""
    import torch
    return torch


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
