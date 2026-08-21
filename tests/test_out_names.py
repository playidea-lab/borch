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


def _classify(name):
    """('single' | 'several' | 'not taken' | None). None means **could not be judged.**"""
    fn = getattr(torch, name)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for args in ([HAND[name]] if name in HAND else PATTERNS):
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
    for name in sorted(dir(torch)):
        if name.startswith("_"):
            continue
        fn = getattr(torch, name)
        if not callable(fn) or inspect.isclass(fn):
            continue
        if "out=None" in (fn.__doc__ or ""):
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
