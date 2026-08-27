"""**The third axis: torch's runtime against the core.**

The two axes already here — `ts_axis.py` for names and `ts_signatures.py` for
argument lists — both compare the core against borch.ts. torch is not in either
comparison, so when the two sides agree with each other and neither agrees with
torch, everything is green and the green reads as "matches torch".

That is not hypothetical. It happened twice in one afternoon:

    Tensor.matmul     core `o`      borch.ts `tensor2`    torch `other`
    Tensor.transpose  core `d0/d1`  borch.ts `d0/d1`      torch `dim0/dim1`

Both were found by a person noticing, and in the first case *both* sessions
looking at it independently reached `tensor2` — which is what
`torch.Tensor.matmul.__doc__` says and what the runtime rejects.

## Where the names come from

`inspect.signature` answers for 26 of `torch.Tensor`'s 566 methods; the rest are
C-implemented and it raises. The JIT schema registry answers for nearly all of
them, and it is the better source anyway: it is what the C++ binding *registered*,
which is exactly what the runtime accepts as a keyword.

    torch._C._jit_get_schemas_for_operator("aten::transpose")
    -> [self, dim0, dim1]

Two things had to be right about reading it, and both were wrong first:

- **Overloads.** `aten::mul` also carries `int * int`, whose arguments are `l`
  and `n`. Only schemas whose first argument is `self` are the method.
- **The receiver.** The core writes `def add(input, other, alpha)` and that
  `input` is `self`. Counting it makes every method look shifted by one.

A first pass without those two reported 261 divergences, nearly all of them
invented. With them: 25, of which 12 were real and the rest were the schema
being narrower than the Python wrapper (`nonzero(as_tuple)` and friends), which
is a question for the other axes.

## What this asks and what it does not

Only **same-position name disagreement**. Not length — a schema that stops short
of the Python API is not a finding, and the `shorter`/`longer` columns next door
are where that belongs. This axis answers one question: *can a caller write the
keyword torch takes?*
"""

import inspect

import pytest

torch = pytest.importorskip("torch")


def _schemas(name):
    """Argument names for each method overload of `aten::<name>`, `self` removed."""
    try:
        schemas = torch._C._jit_get_schemas_for_operator(f"aten::{name}")
    except Exception:
        return []
    out = []
    for s in schemas:
        args = [a.name for a in s.arguments]
        if not args or args[0] != "self":
            continue
        # `out=` and `out_dtype=` are places to put a result, not arguments this
        # library takes.
        rest = [a for a in args[1:] if a not in ("out", "out_dtype")]
        if rest not in out:
            out.append(rest)
    return out


def _ours(fn):
    """The core's argument names with the receiver dropped, or None if unreadable."""
    try:
        params = list(inspect.signature(fn).parameters.values())
    except (TypeError, ValueError):
        return None
    if not params:
        return None
    if any(p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD) for p in params):
        return None
    return [p.name for p in params[1:]]


# Names the core cannot or should not spell torch's way. Each is a fact rather
# than a preference, and each says which.
ALLOWED = {
    # `from` is a Python keyword. `uniform_(from=0.)` cannot be written here, and
    # `ts_signatures.RENAMES` carries the same entry for the same reason.
    ("random_", "from_"): "from",
    ("uniform_", "from_"): "from",
    # **The schema names a seat the Python layer does not open.** `aten::resize_as_`
    # registers `the_template`, and torch's own binding takes it positionally only:
    #
    #     a.resize_as(the_template=b)  TypeError: unexpected keyword 'the_template'
    #     a.resize_as(other=b)         TypeError: unexpected keyword 'other'
    #
    # Measured on both spellings. So there is no keyword to match here, and moving
    # the core to `the_template` would trade a name nobody can write for another
    # name nobody can write.
    #
    # This is the axis's one blind spot stated out loud: the schema is the C++
    # registry and the Python wrapper can be narrower. It is a short list — one
    # entry across 378 methods — and each row is a measurement rather than a
    # preference, so the cost of keeping it by hand is small and the cost of
    # calling every method with valid arguments to find it automatically is not.
    ("resize_as", "other"): "the_template",
}


def _rows():
    import borch

    rows = []
    for name in sorted(dir(borch.Tensor)):
        if name.startswith("_"):
            continue
        fn = getattr(borch.Tensor, name, None)
        if not callable(fn):
            continue
        theirs = _schemas(name)
        if not theirs:
            continue
        mine = _ours(fn)
        if mine is None:
            continue
        if any(mine == t[:len(mine)] for t in theirs):
            continue
        # Same position, different name — and only where both sides have that seat.
        best = min(theirs, key=lambda t: sum(a != b for a, b in zip(t, mine)))
        clash = [(a, b) for a, b in zip(mine, best)
                 if a != b and ALLOWED.get((name, a)) != b]
        if clash:
            rows.append((name, clash))
    return rows


def test_the_core_takes_the_keyword_torch_takes():
    """Every argument seat the core and torch both have carries torch's name.

    A failure here is a keyword a caller copying torch code writes and this
    library refuses — the kind that raises rather than answering wrongly, which
    is the good direction, but only for someone reading the traceback.
    """
    rows = _rows()
    assert not rows, (
        "the core names an argument something torch will not accept:\n  "
        + "\n  ".join(
            f"Tensor.{name}: " + ", ".join(f"{a} -> {b}" for a, b in clash)
            for name, clash in rows)
        + "\n\nThe name torch registered is the name a caller can write. Check by\n"
          "calling with every argument supplied and only the one under test\n"
          "renamed — passing a single keyword raises about the *others* being\n"
          "missing, and that reads as a refusal when it is an acceptance.")


def test_the_schema_reader_is_reading_schemas():
    """**The reader has two ways to go quietly wrong, and both did.**

    Take the wrong overload and every elementwise method looks renamed; count the
    receiver and every method looks shifted. Either mistake makes the axis report
    *more*, which is the direction that looks like a better instrument.

    So the two corrections are pinned against operators whose answers are known.
    """
    assert _schemas("transpose") == [["dim0", "dim1"]], _schemas("transpose")

    # `aten::mul` carries scalar overloads named `l`/`n`; those are not the method.
    muls = _schemas("mul")
    assert ["other"] in muls, muls
    assert not any("l" in names for names in muls), muls

    # The receiver is dropped on both sides, so a one-argument method reads as one.
    assert _schemas("matmul") == [["other"]], _schemas("matmul")
