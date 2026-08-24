"""The shared spelling fold must not read a mismatch as a match.

`ts_axis._camel` bridges `return_indices` and `returnIndices` so the two libraries can
be compared at all. It is a fold, and a fold's whole risk is that it maps two different
things onto one — which in a measuring instrument means **reporting agreement that is
not there.** That is the one direction an instrument must never fail in: a row that
wrongly says *these differ* costs somebody an hour, and a row that wrongly says *these
agree* costs nothing until it costs a user.

It failed that way once. `_camel` ate a **leading** underscore — `_weight` → `Weight` →
`weight` — so the core's `_random_samples` folded onto borch.ts's `randomSamples` and
`FractionalMaxPool2d` and `3d` reported `agree` while their parameter lists differed.
The function's own docstring, three lines above the bug, explains at length why the
*trailing* underscore has to survive (`eq_` is not `eq`). **The same reason was written
down and applied to one end of the string.**

This file is that finding as a rule rather than as a row. Two properties, and neither
can be seen by looking at any single comparison:

1. **The fold keeps the letters and the underscores that carry meaning.** torch marks
   in-place with a trailing `_` and private with a leading one; a spelling change may
   move case and drop the underscores *between* words and nothing else.
2. **The fold does not collide.** Two names folding to one string means whichever is
   compared second silently matches the first's counterpart, and no row shows it —
   the pair looks like two ordinary agreements.

## What a green run here does not say

- **Not that the fold is right**, only that it loses nothing. A fold that mapped every
  name to itself would pass and compare nothing; `test_ts_signatures` pins that the
  comparison still happens.
- **Not that the other folds are safe.** `ts_signatures.RENAMES` and
  `torch_signatures_core.DEPRECATED` are folds too, declared row by row, and a
  declared fold is attested where a reader meets it. This one is a rule applied to
  every name at once, which is why it needs a check instead.
"""

import inspect
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

pytest.importorskip("numpy")

import ts_axis   # noqa: E402


def _spaces():
    import borch
    return {
        "Tensor": borch.Tensor, "nn": borch.nn,
        "nn.functional": borch.nn.functional, "linalg": borch.linalg,
        "optim": borch.optim, "utils.data": borch.utils.data,
    }


def _every_name():
    """Public members and **every parameter name**, per namespace.

    The parameters matter as much as the members and are the half that was wrong:
    a leading-underscore *member* is filtered out of the comparison by convention,
    and a leading-underscore *parameter* is not. `_weight`, `_freeze`,
    `_random_samples` and `_stacklevel` are all real, all torch's, and all compared.
    """
    members, params = {}, set()
    for space, mod in _spaces().items():
        for name in dir(mod):
            if name.startswith("_"):
                continue
            members.setdefault(space, set()).add(name)
            try:
                got = inspect.signature(getattr(mod, name)).parameters
            except (TypeError, ValueError):
                continue
            params |= {p for p in got if p != "self"}
    return members, params


def _bare(text):
    return text.replace("_", "").lower()


def _lead(text):
    return len(text) - len(text.lstrip("_"))


def _trail(text):
    return len(text) - len(text.rstrip("_"))


def test_the_fold_keeps_every_letter():
    """A spelling change moves case and drops the underscores between words. Any
    letter gained or lost means it is translating, not spelling."""
    members, params = _every_name()
    names = params | {n for group in members.values() for n in group}
    lost = [(n, ts_axis._camel(n)) for n in sorted(names)
            if _bare(ts_axis._camel(n)) != _bare(n)]
    assert not lost, (
        "`_camel` changed the letters of a name, so it is not a spelling rule:\n  "
        + "\n  ".join(f"{n!r} -> {c!r}" for n, c in lost[:12]))


def test_the_fold_keeps_the_underscores_that_mean_something():
    """Leading and trailing underscores are torch's markers — private and in-place.
    Eating either makes two different names one, which is how
    `FractionalMaxPool2d` came to report `agree` against a list it did not match."""
    _members, params = _every_name()
    eaten = [(p, ts_axis._camel(p)) for p in sorted(params)
             if _lead(ts_axis._camel(p)) != _lead(p)
             or _trail(ts_axis._camel(p)) != _trail(p)]
    assert not eaten, (
        "`_camel` ate an underscore that marks something:\n  "
        + "\n  ".join(f"{p!r} -> {c!r}" for p, c in eaten)
        + "\n\n  A leading underscore is torch's private marker (`_weight`, "
          "`_random_samples`,\n  `_stacklevel`) and a trailing one marks in-place "
          "(`eq_`). Either eaten, two\n  different names become one and the "
          "comparison reports an agreement that is not there.")


def test_the_fold_does_not_put_two_names_on_one():
    """**The half no single row can show.** Two names folding to one string means the
    second silently matches the first's counterpart, and both rows read as ordinary
    agreements.

    Members are checked per namespace, because `Tensor.det` and `linalg.det` are two
    real names that are never compared with each other — a collision across
    namespaces is not one.
    """
    members, params = _every_name()

    clashes = []
    for space, group in sorted(members.items()):
        folded = {}
        for name in group:
            folded.setdefault(ts_axis._camel(name), []).append(name)
        clashes += [f"{space}: {sorted(v)} all fold to {k!r}"
                    for k, v in sorted(folded.items()) if len(v) > 1]

    folded = {}
    for p in params:
        folded.setdefault(ts_axis._camel(p), []).append(p)
    clashes += [f"parameters: {sorted(v)} all fold to {k!r}"
                for k, v in sorted(folded.items()) if len(v) > 1]

    assert not clashes, (
        "two names fold onto one, so a comparison can match the wrong pair:\n  "
        + "\n  ".join(clashes[:12]))


def test_the_fold_still_folds():
    """**A fold that does nothing passes the three above.** `_camel = lambda n: n`
    keeps every letter, every underscore and collides with nothing — and then the two
    libraries are compared raw and every multi-word name reports as missing on both
    sides at once.

    So the floor: the fold has to still be bridging the two spellings. Pinned low
    enough that only a fold which has stopped working trips it.
    """
    _members, params = _every_name()
    bridged = [p for p in params if ts_axis._camel(p) != p]
    assert len(bridged) > 100, (
        f"only {len(bridged)} names are being folded — `_camel` looks like it has "
        "stopped bridging snake_case to camelCase, and the axes that use it would "
        "then report every multi-word name as missing on both sides.")
