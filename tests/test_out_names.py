"""**Whether the table of names taking `out=` is neither stale nor short.**

Only the names written into `_TAKES_OUT` in `borch/__init__.py` take `out=`. The core does
not lean on torch, so the table is written by hand, and holding that table up against torch
is what happens here.

## Why it cannot be built from the docstrings

torch's C functions do not expose their signatures, so the first version picked them from
`out=None` in the docstrings. That list is **wide** — `rand_like`, `zeros_like`, `ones_like`,
`median`, `nanmedian`, `where`, `std_mean`, `var_mean` and `hamming_window` are all written
there while the actual overload takes no `out=`. The aten schema says the same, that
`rand_like` has an out variant — it is only blocked at the Python layer.

So it **actually calls them.** It builds arguments in a few shapes, and once a shape works it
calls again with `out=` added to the same arguments. A `TypeError` means it is not taken; any
other error came **after** it was taken, so it counts as taken.

## Names whose arguments could not be built

Some work in no shape at all (`from_file`, `hspmm`, `sparse_compressed_tensor` and others we
do not have or that are special). Those names are **not judged** — pretending to know what is
unknown is where the table starts lying. A name we do not have cannot enter the table
anyway.
"""

import inspect
import warnings

import torch

import borch

_V = torch.tensor([1.0, 2.0, 3.0])
_I = torch.tensor([0, 1, 2])
_M = torch.eye(3)
_B = torch.ones(2, 3, 3)
_P = torch.ones(2, 3)

PATTERNS = [
    (_V,), (_V, _V), (_M,), (_M, _M), (_V, 1), (_V, 0), (_I,), (_I, _I),
    ([_V, _V],), (_V, _V, _V), (3,), (0, 3), (_M, 0), (_M, _M, _M), (_B, _B),
    (_M, _V), (_V, 2), (_M, _I), (0.0, 1.0, 3), (_P, _P), (_B,), (_M, 1),
    (2, 3), (_M, 0, True),
]
# The ones no shape catches are given by hand. Absent here and uncaught by a shape, it is **not judged.**
HAND = {
    "addbmm": (_M, _B, _B), "addmv": (_V, _M, _V), "baddbmm": (_B, _B, _B),
    "gather": (_M, 0, torch.zeros(3, 3, dtype=torch.int64)),
    "masked_select": (_V, torch.tensor([True, False, True])),
    "polygamma": (1, _V), "randint": (0, 5, (3,)), "renorm": (_M, 2, 0, 1.0),
    "narrow_copy": (_V, 0, 0, 2),
    "lu_solve": (_M, _M, torch.tensor([1, 2, 3], dtype=torch.int32)),
    "ormqr": (_M, _V, _M),
}


def _classify(name, module=torch, hand=None, patterns=None):
    """('single' | 'several' | 'not taken' | None). None means **could not be judged.**"""
    fn = getattr(module, name)
    hand = HAND if hand is None else hand
    patterns = PATTERNS if patterns is None else patterns
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for args in ([hand[name]] if name in hand else patterns):
            # **Fresh tensors every attempt.** `PATTERNS` holds module-level tensors
            # and the widened intake reaches torch's in-place family — `relu_`,
            # `sigmoid_`, `clamp_` — which write into what they are given. The shared
            # `_V` and `_M` were being edited under every later name in this file and
            # under every later *file* in the suite: `test_golden.py` came back with
            # `resize_as_ 는 제자리다: expected (1, 4), got (2, 2)` and a broken
            # gradient chain, neither of which names this probe.
            #
            # Third cost of enumerating rather than reading, after *the probe's own
            # call looked like a finding* and *the probe changed global state*: *the
            # probe edited its own fixture.* The first two are about what the sweep
            # reports; this one is about what it leaves behind.
            args = tuple(a.clone() if isinstance(a, torch.Tensor) else a
                         for a in args)
            try:
                got = fn(*args)
            except Exception:                               # noqa: BLE001
                continue
            many = (isinstance(got, tuple) and got
                    and all(isinstance(x, torch.Tensor) for x in got))
            if not isinstance(got, torch.Tensor) and not many:
                continue
            dst = (tuple(torch.empty_like(x) for x in got) if many
                   else torch.empty_like(got))
            try:
                fn(*args, out=dst)
            except TypeError:
                return "not taken"
            except Exception:                               # noqa: BLE001
                pass
            return "several" if many else "single"
    return None


def _candidates():
    """**Every public callable, not the ones whose docstring says `out=None`.**

    The paragraph at the top of this file rejects the docstrings for deciding
    *whether* a name takes `out=`, and then the enumeration below used them anyway
    to decide *which names get asked.* Two different jobs, one source, and only one
    of them was examined.

    It cost 24 names: `abs`, `acos`, `asin`, `atan`, `log2`, `log10`, `square`,
    `norm`, `nansum`, `msort`, `diff`, `rad2deg`, `hardshrink` and the rest of the
    inverse-trigonometric family. **torch takes `out=` on every one of them and none
    has `out=None` in its docstring** — they are documented with a bare `out` or with
    none at all. `test_the_out_table_is_not_short` was measuring exactly the set the
    table already covered, so it passed while a quarter of the surface was missing.

    That is the third time in this repository an instrument's entry condition removed
    the class it was hunting, and the first where the file's own docstring named the
    source as unreliable one screen above the line that used it.

    `_classify` returns `None` for a name no shape can call, so widening the intake
    costs nothing but time: an unjudgeable name is still not judged.

    **Except that calling everything is not free.** The widened intake reached
    `torch.set_printoptions` and set a global precision to a tensor, and ten golden
    cases in another file began failing with `Format specifier missing precision` —
    a message naming neither this file nor printing. `CONFIGURES` is
    `tests/test_axis_sweep.py`'s list of names that configure rather than compute,
    shared rather than copied: two lists of the same thing drift, and the second one
    to drift is the one nobody re-reads.
    """
    from test_axis_sweep import CONFIGURES

    for name in sorted(dir(torch)):
        if name.startswith("_") or name in CONFIGURES:
            continue
        fn = getattr(torch, name)
        if not callable(fn) or inspect.isclass(fn):
            continue
        yield name


def test_the_out_table_is_not_stale():
    """A name written into the table has to actually take `out=` in torch."""
    wrong = []
    for name in sorted(borch._TAKES_OUT | borch._TAKES_OUT_TUPLE):
        want = "several" if name in borch._TAKES_OUT_TUPLE else "single"
        got = _classify(name)
        if got is not None and got != want:
            wrong.append(f"{name} — the table says {want}, torch says {got}")
    assert not wrong, (
        "`_TAKES_OUT` is stale:\n  " + "\n  ".join(wrong) + "\n\n"
        "Take that name out of the table — where we are more lenient than torch, code that\n"
        "runs here stops on someone's own machine. Being lenient is diverging too."
    )


def test_the_out_table_is_not_short():
    """A name torch takes `out=` for, which we also have, has to be in the table."""
    missing = []
    for name in _candidates():
        if not hasattr(borch, name):
            continue
        if name in borch._TAKES_OUT or name in borch._TAKES_OUT_TUPLE:
            continue
        if _classify(name) in ("single", "several"):
            missing.append(name)
    assert not missing, (
        "these should take `out=` and are not in the table: " + ", ".join(missing) + "\n\n"
        "Add them. A missing name is refused by `_no_out` while torch accepts it, so\n"
        "**that line of the textbook stops here and nowhere else.**"
    )


def test_this_file_leaves_the_library_as_it_found_it():
    """**Calling every public name is not a read.**

    This has bitten twice now, both times through `set_printoptions`, and both times
    the failure appeared in a different file with a message about format specifiers.
    So the state that can move is checked here rather than trusted to a list somebody
    has to keep right.
    """
    import copy

    before = copy.copy(torch._tensor_str.PRINT_OPTS.__dict__)
    rng = torch.get_rng_state()
    try:
        for name in _candidates():
            _classify(name)
    finally:
        torch.set_rng_state(rng)
    after = copy.copy(torch._tensor_str.PRINT_OPTS.__dict__)
    assert before == after, (
        "classifying the candidates changed torch's print options:\n"
        f"  before {before}\n  after  {after}\n"
        "  Something reached by `dir(torch)` configures rather than computes — name "
        "it in `test_axis_sweep.CONFIGURES`.")


# ── the same two questions for `linalg` ─────────────────────────────────────
#
# `_LINALG_TAKES_OUT` is a second table, and it exists for the same reason: the core
# does not lean on torch, so the list of names taking `out=` is written down. The
# comment above it in `borch/__init__.py` promises this file holds it up against
# torch, and this is where that promise is kept.
#
# **It was drafted from the docstrings** — the source this file's opening paragraph
# rejects, one screen above the line that used it, for the second time in this
# repository. Re-measured by calling, thirty-seven of the thirty-eight were right and
# `lstsq` was missing.
#
# `linalg` needs its own argument shapes: almost everything wants a square matrix, and
# several want a factorisation that has to be produced first.

_SQ = torch.eye(3) * 2 + 0.1
_RHS = torch.randn(3, 2)

_LINALG_PATTERNS = [
    (_SQ,), (_SQ, _SQ), (_SQ, _V), (_V,), (_SQ, _RHS),
]


def _linalg_hand():
    """Shapes no pattern reaches. **Built by calling torch**, because three of them
    want a factorisation as input and one wants a keyword to be legal at all."""
    ld, piv = torch.linalg.ldl_factor(_SQ)
    lu, pivots = torch.linalg.lu_factor(_SQ)
    return {
        "cross": (_V, _V),
        "householder_product": (torch.randn(3, 3), torch.randn(3)),
        "ldl_solve": (ld, piv, _RHS),
        "lu_solve": (lu, pivots, _RHS),
        "matrix_power": (_SQ, 2),
        "multi_dot": ([_SQ, _SQ],),
        "solve_triangular": (torch.tril(_SQ), _RHS),
        "tensorinv": (torch.randn(2, 2, 2, 2),),
        "tensorsolve": (torch.randn(4, 2, 2), torch.randn(4)),
        "vecdot": (_V, _V),
    }


def _linalg_classify(name):
    got = _classify(name, torch.linalg, _linalg_hand(), _LINALG_PATTERNS)
    if got is None and name == "solve_triangular":
        # `upper=` has no default, so no positional shape is a legal call. Asked
        # with it, rather than filed as unjudgeable — an unjudged name is a name
        # the table is not held against.
        try:
            out = torch.empty_like(_RHS)
            torch.linalg.solve_triangular(torch.tril(_SQ), _RHS, upper=False, out=out)
            return "single"
        except TypeError:
            return "not taken"
    return got


def test_the_linalg_out_table_is_not_stale():
    """A name written into `_LINALG_TAKES_OUT` has to actually take `out=` in torch."""
    wrong = [f"{n} — the table says it takes `out=`, torch refuses it"
             for n in sorted(borch._LINALG_TAKES_OUT)
             if _linalg_classify(n) == "not taken"]
    assert not wrong, (
        "`_LINALG_TAKES_OUT` is stale:\n  " + "\n  ".join(wrong) + "\n\n"
        "Take the name out. Being more lenient than torch is diverging too — code\n"
        "written here stops on someone's own machine."
    )


def test_the_linalg_out_table_is_not_short():
    """A `linalg` name torch takes `out=` for, which we also have, has to be in it."""
    missing = []
    for name in sorted(dir(torch.linalg)):
        if name.startswith("_") or name in borch._LINALG_TAKES_OUT:
            continue
        if not hasattr(borch.linalg, name):
            continue
        if _linalg_classify(name) in ("single", "several"):
            missing.append(name)
    assert not missing, (
        "these `linalg` names should take `out=` and are not in the table: "
        + ", ".join(missing) + "\n\n"
        "This is how `lstsq` was found. The draft came off the docstrings and that\n"
        "name's docstring writes the signature as `lstsq(A, B, rcond=None, *,\n"
        "driver=None)` — three of those four are names torch itself refuses."
    )


def test_the_linalg_sweep_actually_judges():
    """**An unjudged name is a name the table is not held against.**

    Both checks above skip anything `_linalg_classify` cannot call, which is the
    right answer for a name no shape reaches and the wrong one for a whole table: a
    sweep that judges nothing passes both assertions in silence. That failure has
    already happened here — a `linalg` wrapper written as `(A, *args, **kw)` read as
    variadic and dropped three rows into *cannot judge* on a different axis.

    So the ratio is pinned. Not the count, which moves when torch adds a name.
    """
    judged = sum(_linalg_classify(n) is not None
                 for n in sorted(borch._LINALG_TAKES_OUT))
    total = len(borch._LINALG_TAKES_OUT)
    assert judged >= total - 2, (
        f"only {judged} of {total} names in `_LINALG_TAKES_OUT` could be called at "
        "all.\n  The shapes above stopped reaching torch, and both checks either "
        "side of this\n  one are passing on an empty sweep."
    )
