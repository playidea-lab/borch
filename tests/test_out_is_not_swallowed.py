"""**The same property as `test_no_silent_out.py`, asked by calling instead of reading.**

That file walks the source for functions with a `**kw` bag and requires `_no_out` in
the body. It was right, and it is the kind of check that quietly stops working: its
population is *functions that have a bag*, and the repair for this whole class is to
**remove the bag.**

Thirty-three seats in `borch/_ops.py` lost theirs — seventeen that swallowed any
keyword at all and dropped it, sixteen that only ever touched `kw` to hand it to the
gate. Every one of them left that check's intake on the way out. A check whose
population shrinks as the defect is repaired ends up green by having nothing left to
look at, and this repository has now met that shape five times.

So the property is asked again in the form that survives the representation:

    a name torch takes `out=` for must **write into the destination** or **refuse
    with the wording that says why** — never take it and go on.

Reading the source cannot ask that. Two functions with identical text differ here if
one of them is wrapped by `_accepts_out`, and two with different text agree.

## What a green run does not say

- **Not that `out=` is implemented.** For most of these it is deliberately not, and
  refusing is the right answer; `tests/test_out_names.py` holds which is which.
- **Not that unknown keywords are refused.** That is a wider property and Python
  enforces it for free once the bag is gone — `test_no_bag_swallows_a_keyword`
  below asks it directly.
"""

import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

pytest.importorskip("numpy")

import borch  # noqa: E402


def _shapes():
    """`{name: (args, kwargs)}` — one working call per name.

    Written by hand rather than swept, because a sweep that guesses arguments reports
    *could not call* for a defect and for a bad guess in the same words.
    """
    t = borch.tensor([1.0, 2.0, 3.0])
    m = borch.tensor([[1.0, 2.0], [3.0, 4.0]])
    return {
        "randint": ((0, 5, (3,)), {}),
        "randperm": ((3,), {}),
        "rand_like": ((t,), {}),
        "randn_like": ((t,), {}),
        "empty_like": ((t,), {}),
        "searchsorted": ((t, t), {}),
        "bucketize": ((t, t), {}),
        "hamming_window": ((4,), {}),
        "hann_window": ((4,), {}),
        "bartlett_window": ((4,), {}),
        "blackman_window": ((4,), {}),
        "kaiser_window": ((4,), {}),
        "tril_indices": ((2, 2), {}),
        "triu_indices": ((2, 2), {}),
        "logspace": ((0, 1, 3), {}),
        "bernoulli": ((borch.zeros(3),), {}),
        "poisson": ((borch.ones(3),), {}),
        "normal": ((0.0, 1.0, (3,)), {}),
        "isin": ((t, t), {}),
        "tensordot": ((t, t), {"dims": 1}),
        "trapezoid": ((t,), {}),
        "cumulative_trapezoid": ((t,), {}),
        "flip": ((t, 0), {}),
        "tile": ((t, 2), {}),
        "count_nonzero": ((t,), {}),
        "asarray": (([1.0],), {}),
        "scalar_tensor": ((1.0,), {}),
        "cov": ((m,), {}),
        "broadcast_to": ((t, 3), {}),
        "frombuffer": ((b"\x00" * 12,), {}),
    }


def _called(name, args, kw):
    fn = getattr(borch, name, None)
    if fn is None:
        return None, None
    try:
        return fn, fn(*args, **kw)
    except Exception:                                       # noqa: BLE001
        return fn, None


def test_out_is_implemented_or_refused_but_never_swallowed():
    implements = borch._TAKES_OUT | borch._TAKES_OUT_TUPLE
    swallowed = []
    for name, (args, kw) in _shapes().items():
        fn, got = _called(name, args, kw)
        if got is None or not hasattr(got, "shape"):
            continue
        dst = borch.zeros(tuple(got.shape), dtype=got.dtype)
        try:
            back = fn(*args, out=dst, **kw)
        except TypeError:
            continue                    # Python stopped it — absent, and it says so
        except borch.BorchError as exc:
            if "out=" in str(exc):
                continue                # refused with the wording that says why
            raise
        if name in implements:
            if back is not dst:
                swallowed.append(
                    f"{name} — written into `_TAKES_OUT` and did not hand back the "
                    "destination")
            continue
        swallowed.append(f"{name} — took `out=` and neither wrote into it nor refused")
    assert not swallowed, (
        "places that swallow `out=`:\n  " + "\n  ".join(sorted(swallowed)) + "\n\n"
        "  Taken and dropped, the destination stays as it was and nothing says so, and\n"
        "  the wrong value surfaces later somewhere unrelated. Either implement it —\n"
        "  the name goes in `_TAKES_OUT` — or let it stop: no `**kw`, or `_no_out(out)`.")


def test_no_bag_swallows_a_keyword():
    """**The wider property the bag was hiding**, and the reason it could be removed.

    `out=` was the keyword somebody noticed. The bag took every other one too:
    `bernoulli(x, dtype=…)`, `asarray(v, zzz=1)`, `flip(t, 0, zzz=1)` all ran and said
    nothing where torch raises. Measured against real torch before the repair: 23
    top-level names.

    torch's own wording is not one thing — the C dispatcher says *received an invalid
    combination of arguments* and the Python layer says *got an unexpected keyword
    argument*. Inventing a third to print at eighty-three call sites would have been
    wrong in both directions; removing the bag hands the second one over for free.
    """
    took = []
    for name, (args, kw) in _shapes().items():
        fn, got = _called(name, args, kw)
        if got is None:
            continue
        try:
            fn(*args, **kw, zzz=1)
        except TypeError:
            continue
        except Exception:                                   # noqa: BLE001
            continue        # stopped for some other reason; still not swallowed
        took.append(name)
    assert not took, (
        "these accept a keyword that means nothing and go on: " + ", ".join(sorted(took))
        + "\n\n  torch raises. A keyword accepted and dropped reads to the caller as a\n"
          "  setting that was honoured.")


def test_the_shapes_still_reach_the_library():
    """**Both checks above `continue` past anything they cannot call.**

    Every skip is a shape that did not work, and a table whose names have all been
    renamed skips every one and passes — the failure this file's own docstring
    describes, one level down. So the floor is on how many were actually reached, not
    on how many are written.
    """
    reached = [n for n, (a, k) in _shapes().items()
               if hasattr(_called(n, a, k)[1], "shape")]
    assert len(reached) >= 25, (
        f"only {len(reached)} of {len(_shapes())} shapes could be called:\n  "
        + ", ".join(sorted(set(_shapes()) - set(reached)))
        + "\n\n  The two checks above are asking almost nothing and would pass on an\n"
          "  empty sweep.")
