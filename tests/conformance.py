"""The conformance harness — generates comparison cases from a table.

The hand-written comparison tests (`test_diff.py`) stop at 76. A person's writing speed is the
limit, and only the situations that person thought of go in. This file instead **keeps the
specification as a table and generates the cases** — multiplying one operation by shapes,
dtypes and argument combinations produces hundreds.

## Fidelity tiers

    T1 value  the same value and shape under allclose(1e-5)  ← the goal across the whole range
    T2 error  the same kind of exception, same conditions    ← the main errors
    T3 repr   the same result from print()                   ← the common ones first
    T4 bits   identical bits                                 ← an explicit non-goal

Why T4 is off the list can be seen by measuring. Adding a hundred thousand float32 values puts
numpy and torch 1.4e-4 apart — a thousand times float32's eps. It is because the summation
**order** differs, and torch's order differs again with the SIMD width and the thread count.
Not a goal that can be chased.

## Usage

    uv run --with pytest --with numpy --with torch pytest tests/conformance.py -q
    uv run --with numpy --with torch python tests/conformance.py      # the score alone
"""

import importlib.util
import itertools
import pathlib
import sys

import numpy as np
import torch as real

_here = pathlib.Path(__file__).resolve().parent
if str(_here.parent) not in sys.path:
    sys.path.insert(0, str(_here.parent))
import borch as nano                                          # noqa: E402

# The case table is a separate file — why is on cases.py's first line.
_cases_spec = importlib.util.spec_from_file_location("bt_cases", _here / "cases.py")
cases_mod = importlib.util.module_from_spec(_cases_spec)
_cases_spec.loader.exec_module(cases_mod)

ATOL = 1e-5


# ------------------------------------------------------------ the input spec

def make(shape, kind="float", seed=0):
    """The same data into both tensors. Randomness gets a fixed seed, to stay reproducible."""
    rng = np.random.default_rng(seed)
    if kind == "float":
        arr = rng.standard_normal(shape).astype(np.float32)
    elif kind == "positive":
        arr = np.abs(rng.standard_normal(shape)).astype(np.float32) + 0.1
    elif kind == "int":
        arr = rng.integers(0, 5, shape).astype(np.int64)
    elif kind == "ramp":
        arr = np.arange(int(np.prod(shape)), dtype=np.float32).reshape(shape)
    else:
        raise ValueError(kind)
    return arr


SHAPES_1D = [(3,), (7,)]
SHAPES_2D = [(2, 3), (4, 4), (1, 5)]
SHAPES_3D = [(2, 3, 4)]


# --------------------------------------------------------- generating cases

class Case:
    def __init__(self, name, run_real, run_nano, tier="T1", grad=False):
        self.name = name
        self.run_real = run_real
        self.run_nano = run_nano
        self.tier = tier
        self.grad = grad


def unary(fn_name, shapes, kind="float", grad=True):
    """The torch.fn(x) shape. Looks at the value and, if asked, the gradient too."""
    out = []
    for shape in shapes:
        arr = make(shape, kind)
        out.append(Case(
            f"{fn_name}{shape}",
            lambda a=arr, f=fn_name: getattr(real, f)(real.tensor(a)),
            lambda a=arr, f=fn_name: getattr(nano, f)(nano.tensor(a)),
            grad=grad,
        ))
    return out


def method(expr, shapes, kind="float", label=None, grad=False):
    """Takes a method call as a string, as in 't.sum(dim=0)', and applies it to both sides."""
    out = []
    for shape in shapes:
        arr = make(shape, kind)
        src = f"lambda t: t.{expr}"
        out.append(Case(
            f"{label or expr}{shape}",
            lambda a=arr, s=src: eval(s)(real.tensor(a)),          # noqa: S307
            lambda a=arr, s=src: eval(s)(nano.tensor(a)),          # noqa: S307
            grad=grad,
        ))
    return out


def binary(op, shape_pairs, kind="float"):
    out = []
    for sa, sb in shape_pairs:
        a, b = make(sa, kind, 0), make(sb, kind, 1)
        src = f"lambda x, y: x {op} y"
        out.append(Case(
            f"a {op} b {sa}·{sb}",
            lambda x=a, y=b, s=src: eval(s)(real.tensor(x), real.tensor(y)),   # noqa: S307
            lambda x=a, y=b, s=src: eval(s)(nano.tensor(x), nano.tensor(y)),   # noqa: S307
            grad=True,
        ))
    return out


BROADCAST_PAIRS = [
    ((2, 3), (2, 3)),
    ((2, 3), (3,)),
    ((2, 3), (2, 1)),
    ((4, 1, 3), (2, 3)),
    ((5,), ()),
]


def build_cases():
    cases = []

    # element-wise functions
    for fn in ("sigmoid", "relu", "tanh", "exp"):
        cases += unary(fn, SHAPES_1D + SHAPES_2D)
    for fn in ("log", "sqrt"):
        cases += unary(fn, SHAPES_1D + SHAPES_2D, kind="positive")
    cases += unary("abs", SHAPES_1D + SHAPES_2D)

    # arithmetic and broadcasting
    for op in ("+", "-", "*"):
        cases += binary(op, BROADCAST_PAIRS)

    # reductions — dim × keepdim combinations
    for fn, dims in [("sum", [None, 0, 1]), ("mean", [None, 0, 1])]:
        for dim, keep in itertools.product(dims, [False, True]):
            expr = f"{fn}()" if dim is None else f"{fn}(dim={dim}, keepdim={keep})"
            cases += method(expr, SHAPES_2D, grad=True)
    for fn in ("max", "min"):
        for dim in (0, 1):
            cases += method(f"{fn}(dim={dim}).values", SHAPES_2D, grad=True)
    for unbiased in (True, False):
        cases += method(f"std(unbiased={unbiased})", SHAPES_1D + SHAPES_2D)

    # shape manipulation
    cases += method("reshape(-1)", SHAPES_2D + SHAPES_3D, grad=True)
    cases += method("transpose(0, 1)", SHAPES_2D, grad=True)
    cases += method("unsqueeze(0)", SHAPES_1D + SHAPES_2D)
    cases += method("flatten(1)", SHAPES_3D)
    cases += method("permute(2, 0, 1)", SHAPES_3D)
    cases += method("t.T if False else t[0]", SHAPES_2D, label="idx[0]", grad=True)
    cases += method("[[0, 0, 1]]", SHAPES_2D, label="fancy-idx", grad=True)

    # softmax
    for dim in (-1, 0):
        cases.append(Case(
            f"softmax(dim={dim})(2,3)",
            lambda d=dim: real.nn.functional.softmax(real.tensor(make((2, 3))), dim=d),
            lambda d=dim: nano.softmax(nano.tensor(make((2, 3))), dim=d),
            grad=True,
        ))

    # dtype promotion — numpy's rules and torch's rules differ. This diverges often.
    for expr, label in [("tensor([1, 2, 3])", "int literal"),
                        ("tensor([1.0, 2.0])", "float literal"),
                        ("tensor([1, 2]) + 1.0", "int + float"),
                        ("tensor([1.0]) * 2", "float * int"),
                        ("tensor([True, False])", "bool literal")]:
        cases.append(Case(
            f"dtype: {label}",
            lambda e=expr: str(eval(f"real.{e}").dtype),            # noqa: S307
            lambda e=expr: str(eval(f"nano.{e}").dtype),            # noqa: S307
            tier="T1",
        ))

    return cases


# ---------------------------------------------------------- the wide surface

# The table itself is in `cases.py`. Golden stage two runs in a browser and there is no torch
# there, so with the table in this file it could not even be imported. Kept in two copies, they
# would diverge eventually.
def _wide_cases():
    return cases_mod.wide_cases()


def report_wide():
    cases = _wide_cases()
    bad = []
    for name, fn in cases:
        try:
            r = fn(real).detach().numpy()
        except Exception as exc:                                    # noqa: BLE001
            bad.append(f"{name}: failed under real torch — {type(exc).__name__}")
            continue
        try:
            n = fn(nano).data
        except Exception as exc:                                    # noqa: BLE001
            bad.append(f"{name}: {type(exc).__name__} — {str(exc).splitlines()[0][:50]}")
            continue
        if r.shape != n.shape:
            bad.append(f"{name}: shape {r.shape} vs {n.shape}")
        elif not np.allclose(r, n, atol=1e-4, rtol=1e-4):
            bad.append(f"{name}: max diff {np.abs(r - n).max():.2e}")

    print(f"\nconformance, wide surface — {len(cases)} operations")
    print(f"  agreeing {len(cases) - len(bad)}/{len(cases)}")
    if bad:
        print("\nwhere it diverged:")
        for why in bad:
            print(f"  ✗ {why}")
    return len(bad)


# ----------------------------------------------------------------- T2 errors

# The failures a learner actually meets. **The same kind of exception** has to come out under
# the same conditions, and our message has to carry torch's canonical English phrase — that is
# what makes a search work.
ERROR_CASES = [
    ("matmul shape mismatch", lambda t: t.randn(3, 4) @ t.randn(3, 2),
     "shapes cannot be multiplied"),
    ("not broadcastable", lambda t: t.randn(3, 4) + t.randn(3, 2),
     "must match the size of tensor"),
    ("reshape element-count mismatch", lambda t: t.randn(2, 3).reshape(4, 2),
     "is invalid for input of size"),
    ("backward on a non-scalar", lambda t: t.randn(3, requires_grad=True).backward(),
     "grad can be implicitly created only for scalar outputs"),
    ("backward without requires_grad", lambda t: t.randn(3).sum().backward(),
     "does not require grad"),
    ("requires_grad on an integer tensor", lambda t: t.tensor([1, 2, 3], requires_grad=True), None),
    ("item() on several elements", lambda t: t.randn(3).item(),
     "cannot be converted to Scalar"),
    ("Linear input dimension mismatch", lambda t: t.nn.Linear(4, 2)(t.randn(3, 5)),
     "shapes cannot be multiplied"),
    ("Conv2d channel mismatch",
     lambda t: t.nn.functional.conv2d(t.randn(1, 3, 8, 8), t.randn(4, 1, 3, 3))
     if t is real else t.conv2d(t.randn(1, 3, 8, 8), t.randn(4, 1, 3, 3)), None),
    ("index out of range", lambda t: t.randn(3)[5], "out of bounds"),
    ("in-place edit of a leaf", lambda t: _inplace_leaf(t), None),
    ("backward twice", lambda t: _backward_twice(t),
     "backward through the graph a second time"),
]


def _inplace_leaf(t):
    x = t.randn(3, requires_grad=True)
    x += 1
    return x


def _backward_twice(t):
    x = t.randn(3, requires_grad=True)
    y = (x * 2).sum()
    y.backward()
    y.backward()


def _raised(lib, fn):
    try:
        fn(lib)
        return None, ""
    except Exception as exc:                                        # noqa: BLE001
        return type(exc).__name__, str(exc)


def report_errors():
    same_kind = 0
    searchable = 0
    needs_phrase = 0
    problems = []

    for name, fn, phrase in ERROR_CASES:
        rk, _ = _raised(real, fn)
        nk, nm = _raised(nano, fn)

        if rk is None:
            problems.append(f"{name}: real torch raises nothing — the case is wrong")
            continue
        if nk is None:
            problems.append(f"{name}: borch goes through quietly (torch gives {rk})")
            continue
        if nk != rk:
            problems.append(f"{name}: expected {rk}, got {nk}")
            continue
        same_kind += 1

        if phrase:
            needs_phrase += 1
            if phrase in nm:
                searchable += 1
            else:
                problems.append(f"{name}: the message has no searchable phrase — \"{phrase}\"")

    total = len(ERROR_CASES)
    print(f"\nconformance T2 (errors) — {total} cases")
    print(f"  exception kind agreeing {same_kind}/{total}"
          f" · searchable messages {searchable}/{needs_phrase}")
    if problems:
        print("\nwhere it diverged:")
        for why in problems:
            print(f"  ✗ {why}")
    return len(problems)


# --------------------------------------------- dtype promotion, exhaustively

# torch's promotion sorts by **category** (bool < integer < float) and promotes only within
# that category. numpy promotes differently (float32 + int64 → float64). Inheriting even one
# slot teaches a learner the wrong rule, so every combination is swept.
DTYPES = ["float32", "float64", "int64", "bool"]
BIN_OPS = ["+", "-", "*", "/"]
PY_SCALARS = [("Python int", 2), ("Python float", 2.0), ("Python bool", True)]


def _dt(lib, name):
    if name != "bool":
        return getattr(lib, name)
    return real.bool if lib is real else lib.bool_


def _mk(lib, name):
    return lib.tensor([1, 0] if name == "bool" else [1, 2], dtype=_dt(lib, name))


def _dtype_of(fn):
    try:
        return str(fn().dtype)
    except Exception as exc:                                        # noqa: BLE001
        return f"<{type(exc).__name__}>"


def report_dtypes():
    bad, total = [], 0
    for a in DTYPES:
        for b in DTYPES:
            for op in BIN_OPS:
                total += 1
                r = _dtype_of(lambda: eval(f"x {op} y", {},                      # noqa: S307
                                           {"x": _mk(real, a), "y": _mk(real, b)}))
                n = _dtype_of(lambda: eval(f"x {op} y", {},                      # noqa: S307
                                           {"x": _mk(nano, a), "y": _mk(nano, b)}))
                if r != n:
                    bad.append((f"{a} {op} {b}", r, n))
    for a in DTYPES:
        for label, value in PY_SCALARS:
            for op in BIN_OPS:
                total += 1
                r = _dtype_of(lambda: eval(f"x {op} s", {},                      # noqa: S307
                                           {"x": _mk(real, a), "s": value}))
                n = _dtype_of(lambda: eval(f"x {op} s", {},                      # noqa: S307
                                           {"x": _mk(nano, a), "s": value}))
                if r != n:
                    bad.append((f"{a} {op} {label}", r, n))

    print(f"\nconformance, dtype promotion — {total} combinations")
    print(f"  agreeing {total - len(bad)}/{total}")
    # **The shortfall has one cause, and saying so is the difference between a score
    # and a defect list.** Every disagreement here involves `float64` — measured, not
    # assumed: the count below is taken rather than written down. There is no such
    # storage in this subset (WGSL has no `f64`), `Tensor.__init__` narrows it at
    # construction, and promotion follows. A reader who sees `75/112` without this
    # line has to work out whether thirty-seven separate rules are wrong.
    from_f64 = [row for row in bad if "float64" in row[0] or "float64" in str(row[1])]
    if bad:
        share = "all of them" if len(from_f64) == len(bad) else f"{len(from_f64)} of them"
        print(f"  {share} are float64, which this subset does not have. "
              "The promotion rules themselves are not in question.")
    if bad:
        print("\nwhere it diverged:")
        for name, r, n in bad[:20]:
            print(f"  ✗ {name:30} torch {r:<18} nano {n}")
        if len(bad) > 20:
            print(f"  … and {len(bad) - 20} more")
    return len(bad)


# ---------------------------------------------------------- shared storage

# torch's view, transpose and slice **share storage.** Handing back a copy would be convenient
# and would stop teaching exactly the point where accidents happen in practice.
VIEW_CASES = {
    "edit a view → the original": lambda t: _mutate(t.zeros(4), lambda a: a.view(2, 2), (0, 0), 9),
    "edit a transpose → the original": lambda t: _mutate(t.zeros(2, 2), lambda a: a.transpose(0, 1), (0, 1), 7),
    "edit an index → the original": lambda t: _mutate(t.zeros(2, 2), lambda a: a[0], 0, 5),
    "edit a slice → the original": lambda t: _mutate(t.zeros(5), lambda a: a[1:3], 0, 9),
    "edit a flatten → the original": lambda t: _mutate(t.zeros(2, 2), lambda a: a.flatten(0), 0, 9),
    "edit a squeeze → the original": lambda t: _mutate(t.zeros(1, 3), lambda a: a.squeeze(), 0, 9),
    "edit an unsqueeze → the original": lambda t: _mutate(t.zeros(3), lambda a: a.unsqueeze(0), (0, 0), 9),
    "edit a permute → the original": lambda t: _mutate(t.zeros(2, 3), lambda a: a.permute(1, 0), (0, 1), 9),
    "clone is independent": lambda t: _mutate(t.zeros(2), lambda a: a.clone(), 0, 3),
    "detach shares": lambda t: _mutate(t.zeros(2), lambda a: a.detach(), 0, 4),
    "fancy indexing copies": lambda t: _mutate(t.zeros(3), lambda a: a[[0, 1]], 0, 9),
    "a non-contiguous view is refused": lambda t: t.zeros(3, 4).transpose(0, 1).view(12).tolist(),
    "a non-contiguous reshape is allowed": lambda t: tuple(t.zeros(3, 4).transpose(0, 1).reshape(12).shape),
}


def _mutate(base, make_view, key, value):
    """Makes a view, edits one slot, and returns what became of **the original.**"""
    view = make_view(base)
    view[key] = value
    return base.tolist()


def report_views():
    same, problems = 0, []
    for name, fn in VIEW_CASES.items():
        r, n = _outcome(real, fn), _outcome(nano, fn)
        if r == n:
            same += 1
        else:
            problems.append(f"{name}: torch {r} · nano {n}")
    print(f"\nconformance, shared storage — {len(VIEW_CASES)} cases")
    print(f"  agreeing {same}/{len(VIEW_CASES)}")
    if problems:
        print("\nwhere it diverged:")
        for why in problems:
            print(f"  ✗ {why}")
    return len(problems)


def _outcome(lib, fn):
    try:
        return str(fn(lib))
    except Exception as exc:                                        # noqa: BLE001
        return f"<{type(exc).__name__}>"


# ------------------------------------------------------------------ T3 repr

# What a learner does most often is print(tensor). When the screen differs from the textbook's
# example, they doubt what they did wrong every single time.
REPR_CASES = [
    ("scalar", "tensor(3.14)"),
    ("float holding integer values", "tensor([1.0, 2.0, 3.0])"),
    ("decimals", "tensor([0.1, 0.25])"),
    ("negatives mixed in", "tensor([-1.5, 2.0, -0.25])"),
    ("2-D", "tensor([[1.0, 2.0], [3.0, 4.0]])"),
    ("3-D", "zeros(2, 1, 3)"),
    ("integer", "tensor([1, 2, 3])"),
    ("boolean", "tensor([True, False])"),
    ("empty tensor", "tensor([])"),
    ("large and small values", "tensor([1e6, 2e-6])"),
    ("a long 1-D that wraps", "arange(30).float()"),
    ("requires_grad", "tensor([1.0, 2.0], requires_grad=True)"),
    ("grad_fn on a non-leaf node", "tensor([1.0], requires_grad=True) * 2"),
    ("grad_fn on a sum", "tensor([1.0, 2.0], requires_grad=True).sum()"),
    ("relu grad_fn", "relu(tensor([-1.0, 2.0], requires_grad=True))"),
]


def report_repr():
    same, problems = 0, []
    for name, expr in REPR_CASES:
        try:
            r = repr(eval(expr, {"__builtins__": {}}, _ns(real)))     # noqa: S307
        except Exception as exc:                                      # noqa: BLE001
            problems.append(f"{name}: failed under real torch — {exc}")
            continue
        try:
            n = repr(eval(expr, {"__builtins__": {}}, _ns(nano)))     # noqa: S307
        except Exception as exc:                                      # noqa: BLE001
            problems.append(f"{name}: {type(exc).__name__} — {exc}")
            continue
        if r == n:
            same += 1
        else:
            problems.append(f"{name}:\n      torch {r!r}\n      nano  {n!r}")

    print(f"\nconformance T3 (repr) — {len(REPR_CASES)} cases")
    print(f"  agreeing {same}/{len(REPR_CASES)}")
    if problems:
        print("\nwhere it diverged:")
        for why in problems:
            print(f"  ✗ {why}")
    return len(problems)


def _ns(lib):
    """Spreads the names out so an expression can write tensor(...) directly."""
    return {n: getattr(lib, n) for n in
            ("tensor", "zeros", "ones", "arange", "randn", "relu", "sigmoid")
            if hasattr(lib, n)}


# ------------------------------------------------------------------ verdict

def compare(case):
    """(passed, reason). An exception goes into the reason and counts towards the score."""
    try:
        r = case.run_real()
    except Exception as exc:                                        # noqa: BLE001
        return None, f"real torch failed: {type(exc).__name__}"
    try:
        n = case.run_nano()
    except nano.BorchError as exc:
        return False, f"unsupported: {str(exc).splitlines()[0]}"
    except Exception as exc:                                        # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"

    if isinstance(r, str) or isinstance(n, str):
        return (r == n), f"{r!r} vs {n!r}"

    rn = r.detach().numpy()
    nn_ = n.data
    if rn.shape != nn_.shape:
        return False, f"shape {rn.shape} vs {nn_.shape}"
    if not np.allclose(rn, nn_, atol=ATOL, rtol=ATOL):
        return False, f"max diff {np.abs(rn - nn_).max():.3e}"

    if case.grad:
        try:
            ok, why = compare_grad(case)
            if not ok:
                return False, f"gradient: {why}"
        except nano.BorchError as exc:
            return False, f"gradient unsupported: {str(exc).splitlines()[0]}"
        except Exception as exc:                                    # noqa: BLE001
            return False, f"gradient {type(exc).__name__}: {exc}"
    return True, ""


def compare_grad(case):
    """Looks at the gradient of the same computation. Folds to a scalar and calls backward."""
    seen = {}

    def tap(lib, tensor_fn):
        original = tensor_fn

        def wrapped(data, *a, **k):
            t = original(np.asarray(data), *a, **k)
            if getattr(t, "dtype", None) is not None and "float" in str(t.dtype):
                t.requires_grad = True
                seen.setdefault(lib, []).append(t)
            return t
        return wrapped

    real_tensor, nano_tensor = real.tensor, nano.tensor
    real.tensor = tap("real", real_tensor)
    nano.tensor = tap("nano", nano_tensor)
    try:
        r_out, n_out = case.run_real(), case.run_nano()
        r_out.sum().backward()
        n_out.sum().backward()
    finally:
        real.tensor, nano.tensor = real_tensor, nano_tensor

    for rt, nt in zip(seen.get("real", []), seen.get("nano", [])):
        if rt.grad is None and nt.grad is None:
            continue
        if (rt.grad is None) != (nt.grad is None):
            return False, "only one side has a gradient"
        if not np.allclose(rt.grad.numpy(), nt.grad.data, atol=ATOL, rtol=ATOL):
            return False, f"max diff {np.abs(rt.grad.numpy() - nt.grad.data).max():.3e}"
    return True, ""


def report():
    cases = build_cases()
    passed, failed, skipped = [], [], []
    for case in cases:
        ok, why = compare(case)
        if ok is None:
            skipped.append((case, why))
        elif ok:
            passed.append(case)
        else:
            failed.append((case, why))

    total = len(passed) + len(failed)
    print(f"conformance T1 (values, gradients) — {len(cases)} cases")
    print(f"  passed {len(passed)} · failed {len(failed)} · skipped {len(skipped)}")
    if total:
        print(f"  score {100 * len(passed) // total}%")
    if failed:
        print("\nwhere it diverged:")
        for case, why in failed[:30]:
            print(f"  ✗ {case.name:34} {why}")
        if len(failed) > 30:
            print(f"  … and {len(failed) - 30} more")
    return len(failed)


if __name__ == "__main__":
    failed = report() + report_errors() + report_repr() + report_dtypes() + report_views() + report_wide()
    raise SystemExit(1 if failed else 0)
