"""A piece of borch, split out. __init__ gathers the public names."""

import math as _math

import numpy as _np

__all__ = ["Tensor", "tensor", "nn", "optim", "no_grad"]

_DEFAULT_DTYPE = _np.float32


class BorchError(NotImplementedError):
    """Something the subset does not support. It stops here rather than
    approximating."""


def _like_torch(said: str, torch_phrase: str) -> str:
    """The shape of an error message.

    Our sentence says **what is blocked and why**, and torch's wording is
    attached **so that it can be searched for.** Ours alone never reaches the
    page that holds the answer, and copying theirs alone leaves nowhere to say
    the things torch never meets (that WGSL has no f64, for instance).

    Both are English. It is the surface a user meets the first time something
    breaks.
    """
    return f"{said}\n(torch: {torch_phrase})"


def _unsupported(what: str):
    raise BorchError(
        f"{what} is not in the browser subset.\n"
        "Use real PyTorch on your own machine (`uv add torch`) — this subset is for "
        "practising the syntax, and imitating what is missing teaches the wrong thing."
    )


# The dtype names torch uses in its error messages. **Complex was missing** —
# `out=`'s dtype refusal announced it by blowing up with a `KeyError` there.
_TYPE_NAMES = {"b": "Bool", "i": "Long", "u": "Long", "f": "Float",
               "c": "ComplexFloat"}


def _only_cpu(what, requested):
    """One rule for every `device=` seat: **`cpu` is this library's device and
    everything else stops.**

    Two opposite mistakes lived under this one argument. The layers refused it
    outright, so `nn.Linear(3, 2, device="cpu")` — a line naming the device the
    tensor was going to be on anyway — stopped. And the factories read it not at
    all, so `zeros(2, device="cuda")` handed back a CPU tensor with no exception:
    the values right and the claim about where they are false. Neither habit could
    be corrected without the other, because the argument had no rule.

    `None` passes: it is the default and means "wherever things go".
    """
    if requested is None:
        return
    name = str(getattr(requested, "type", requested))
    if name != "cpu":
        _unsupported(f"{what}(device={name!r})")


def _float_in(data):
    """The array a float-only function should compute on.

    torch promotes an integral input to the default dtype and answers there;
    numpy either promotes to `float64` — wider than torch's `float32` — or, where
    the expression is written in terms that stay integral, **does not promote at
    all and truncates the answer into the input's cells.** One line, applied at
    the door, removes both.
    """
    return data.astype(_DEFAULT_DTYPE) if data.dtype.kind not in "fc" else data


def _arith_in(data):
    """The array an arithmetic function should compute on.

    torch promotes a boolean to `int64` before arithmetic — `square(tensor([True,
    False]))` is `tensor([1, 0])` and not `tensor([True, False])`. Booleans are the
    one place where "the values are already right" and "the type is already right"
    come apart, because `True * True` is `True` and also `1`.
    """
    return data.astype(_np.int64) if data.dtype.kind == "b" else data


def _needs_float(data, said: str, torch_phrase: str):
    """**We stop where torch stops.**

    Mean, variance and norm are division and square roots, so the answer does not
    fit in an integer cell. numpy quietly promotes to float64 and hands back a
    value, and whoever receives that value finds out later that the same line
    stops with a `RuntimeError` in torch — the kind this repository's first line
    refuses.

    What leaks is numpy, so each place has to be blocked separately. Gathered
    here, which functions live under this rule reads as a list.
    """
    if data.dtype.kind not in "fc":
        raise RuntimeError(_like_torch(said, torch_phrase))


def _refuses_bool(data, said: str, torch_phrase: str, kind=RuntimeError):
    """The places that refuse booleans only. `argmax` and `median` are like that
    (measured)."""
    if data.dtype.kind == "b":
        raise kind(_like_torch(said, torch_phrase))


def _refuses_nonfloat_kernel(data, name: str, kernel: str):
    """The places that imitate a **hole in torch's kernels** exactly.

    Not a rule — `logsumexp` takes integers and `logcumsumexp` does not
    (measured). torch simply did not build the CPU kernel for that dtype, which
    is why it comes out as `NotImplementedError` rather than `RuntimeError`.

    **It is imitated anyway.** Handing back a value here means that code breaks
    against real torch, and being more permissive is still diverging. The kernel
    name goes into the wording because torch does that, and that wording is the
    one a search finds.
    """
    if data.dtype.kind not in "fc":
        raise NotImplementedError(_like_torch(
            f"{name} is for floating point only. Call `.float()` first.",
            f'"{kernel}" not implemented for '
            f"'{_TYPE_NAMES.get(data.dtype.kind, data.dtype.name)}'"))


# ---------------------------------------------------------------- dtype

class dtype:
    def __init__(self, name, np_type):
        self.name = name
        self.np = np_type

    def __repr__(self):
        return f"torch.{self.name}"

    def __eq__(self, other):
        return isinstance(other, dtype) and self.name == other.name

    def __hash__(self):
        return hash(self.name)


float32 = dtype("float32", _np.float32)
float64 = dtype("float64", _np.float64)
int64 = dtype("int64", _np.int64)
long = int64
bool_ = dtype("bool", _np.bool_)
# **A complex number is two float32s.** A layout convention rather than a
# hardware type (measured: 8 bytes per element, `view_as_real` giving `(re, im)`
# on the last axis), which is why it is representable on the GPU side too.
complex64 = dtype("complex64", _np.complex64)
cfloat = complex64
# **There will never be a `complex128`.** WGSL has no `f64`, so there is no
# `float64`, and then there is no double-precision complex either. The name is
# kept **because promotion produces it** — `complex64 + float64` is `complex128`
# in torch (measured), and stopping there means being able to say what it was
# about to build.
complex128 = dtype("complex128", _np.complex128)
cdouble = complex128

# ── the five numeric constants torch keeps at top level ─────────────────────
#
# **A place the coverage table structurally could not see.**
# `tests/torch_gap.py` counts names that are `callable`, and these five are
# **values** rather than things that can be called. So they entered neither the
# numerator nor the denominator, and the number "torch 79% · 0 to review" came
# out with them missing. A place the measure cannot see stays invisible however
# much is counted.
#
# All of them are names textbooks actually use — `torch.clamp(x,
# min=-torch.inf)`, `x[:, torch.newaxis]`, `torch.pi`. They point straight at
# numpy's names of the same spelling, so there is nowhere for the values to
# diverge.
e = _math.e
pi = _math.pi
inf = _math.inf
nan = _math.nan
# In torch this is plain `None` too — a sign that it means the same as
# `x[:, None]`.
newaxis = None


class _AbsentDtype(dtype):
    """A dtype **whose name torch shares and which has no storage in this
    subset.**

    Leaving the name out entirely makes `dtype=torch.int` stop with an
    `AttributeError`, and that wording is **indistinguishable from a typo.** The
    name is kept and says what is missing when it is used — `complex128` exists
    as a name for the same reason.

    The parent's `__init__` is not called. The parent plants `self.np` as a
    value, and here that place has to be **a gate that stops on read.**
    """

    def __init__(self, name, instead):
        self.name = name
        self._instead = instead

    @property
    def np(self):
        _unsupported(f"`torch.{self.name}` (use `{self._instead}` instead)")


# **`torch.int` is int32** (measured — `torch.long` is int64). The integer
# storage is gathered into int64 alone, so there is no int32. The name is kept
# anyway: textbooks write `dtype=torch.int`, and at that moment "absent" and
# "a typo" have to be different sentences.
int32 = _AbsentDtype("int32", "int64")
# The rest kept as names for the same reason. **Half precision does not exist in
# WGSL** (the f16 extension varies by device) and the narrow integers are gathered
# into int64. Without the name, `dtype=torch.half` stops with
# `'function' object has no attribute 'np'`, which is a typo's wording.
float16 = _AbsentDtype("float16", "float32")
bfloat16 = _AbsentDtype("bfloat16", "float32")
int16 = _AbsentDtype("int16", "int64")
complex32 = _AbsentDtype("complex32", "complex64")
# **`uint8` was the one narrow integer with no name at all**, and it is the one a
# textbook writes most: an image is `uint8` before `ToTensor` divides it by 255,
# and `read_image` hands one back. Without the name, `dtype=torch.uint8` stopped
# with `module 'borch' has no attribute 'uint8'` while its four siblings said what
# was missing. `int8` is its pair.
uint8 = _AbsentDtype("uint8", "int64")
int8 = _AbsentDtype("int8", "int64")
half = float16
short = int16
chalf = complex32

_NP_TO_DTYPE = {_np.dtype("float32"): float32, _np.dtype("float64"): float64,
                _np.dtype("int64"): int64, _np.dtype("bool"): bool_,
                _np.dtype("complex64"): complex64,
                _np.dtype("complex128"): complex128}


def _resolve(data, dt):
    """Follows real torch's rule — integers alone give int64, and one float
    anywhere gives float32.

    **A Python `complex` in the mix gives complex64** (measured:
    `torch.tensor([1+1j])` is `complex64`). Left to numpy it becomes
    `complex128`, and there is no such thing here.
    """
    if dt is not None:
        # A dtype named by the caller — see `_requested_dtype`.
        return _requested_dtype(dt).np
    arr = _np.asarray(data)
    if arr.dtype.kind == "c":
        return _np.complex64
    if arr.dtype.kind == "b":
        return _np.bool_
    if arr.dtype.kind in "iu":
        return _np.int64
    return _np.float32


def _requested_dtype(dt, where="dtype=float64"):
    """A dtype the caller **named**. Double precision stops here.

    `Tensor.__init__` narrows `float64` to `float32` and that is deliberate — it is
    the throat every promotion passes through, and numpy raises `int64 + float32` to
    double behind our back. Narrowing what arrives by accident is how the library
    holds its first design decision.

    **A named request is not an accident.** `.double()`, `.to(float64)` and
    `.type(float64)` all refuse, in `Tensor._cast`, whose comment draws the line
    this function is the other half of: *the request was granted in name and
    answered in another cell.* `tensor(x, dtype=float64)` was the fourth spelling
    of the same request and the only one still answered that way — measured across
    thirty-seven factories, every one of which took the argument and handed back
    `float32`.

    Three doors that raise and one that quietly gives you something else is worse
    than four that raise, because the quiet one teaches that the dtype was honoured.

    **The binding never had this hole, and the reason names the core's.** Over there
    `float64` is an `_AbsentDtype` — a name that exists and stops the moment it is
    used — so every spelling refuses without anything being written. Here the name
    cannot be that, because it has a **second job**: naming what numpy hands over
    during promotion, which is also why `complex128` is a real dtype object. The gate
    could not go on the name, so it goes on the request, and one library needing a
    function where the other needs nothing is the whole difference.
    """
    if dt is not None and _np.dtype(dt.np if isinstance(dt, dtype) else dt) == _np.float64:
        _unsupported(f"float64 (`{where}`)")
    return dt


def _no_complex128(what="This operation"):
    """**A double-precision complex cannot be made.** The same place as there
    being no `float64`."""
    raise BorchError(
        f"{what} would make complex128 — the browser subset has no `float64` "
        "(WGSL has no `f64`), so it has no double-precision complex either. "
        "Use `complex64`.")



# ---------------------------------------------------------------------- repr
#
# The thing a learner does most is print(tensor). Printed differently from the
# real thing, the screen stops matching the textbook's example, and every time
# that happens they suspect they did something wrong.
# Follows the rules in torch/_tensor_str.py.

_PRINT_PRECISION = 4
_LINE_WIDTH = 80


def set_printoptions(precision=None, linewidth=None):
    global _PRINT_PRECISION, _LINE_WIDTH
    if precision is not None:
        _PRINT_PRECISION = precision
    if linewidth is not None:
        _LINE_WIDTH = linewidth


def _nonfinite_str(v):
    """`nan`, `inf` and `-inf`. **No trailing dot** — torch is the same
    (measured)."""
    return "nan" if _np.isnan(v) else ("inf" if v > 0 else "-inf")


def _integral_str(v):
    """The integer form. Non-finite values pass through without the dot.

    **This printed `nan.` for a while.** `f"{v:.0f}."` attached the dot to `nan`
    as well, so `tensor([nan, 1.])` came out as `tensor([nan., 1.])`. The decimal
    form was unaffected because `f"{nan:.4f}"` is already `nan`, so only this
    place diverged, and it surfaced only when printing an **integer** tensor with
    a nan in it — caught while adding complex numbers and trying a nan in the
    real part.
    """
    return _nonfinite_str(v) if not _np.isfinite(v) else f"{v:.0f}."


def _float_formatter(arr):
    """torch's rule: all-integer values give `1.`, otherwise four decimal
    places, and a wide range gives exponents."""
    finite = arr[_np.isfinite(arr)]
    nonzero = finite[finite != 0]
    if nonzero.size == 0:
        return _integral_str
    amax, amin = _np.abs(nonzero).max(), _np.abs(nonzero).min()
    integral = bool(_np.all(finite == _np.floor(finite)))

    if integral and amax < 1e8:
        return _integral_str
    if amax / amin > 1000 or amax > 1e8 or amin < 1e-4:
        return lambda v, p=_PRINT_PRECISION: f"{v:.{p}e}"
    return lambda v, p=_PRINT_PRECISION: f"{v:.{p}f}"


def _field_width(arr, fmt):
    """The right-aligned width.

    **Only finite values are counted** — torch is the same (measured). Counting
    `nan` towards the width makes it 3 in the integer form and pushes `1.` out to
    ` 1.`, while torch gives `tensor([nan, 1.])`. A non-finite value longer than
    the width simply overflows it.
    """
    return max((len(fmt(v)) for v in _np.asarray(arr).reshape(-1)
                if _np.isfinite(v)), default=0)


def _tensor_str(data):
    if data.size == 0:
        return "[]" if data.ndim else "[]"
    if data.dtype.kind == "f":
        fmt = _float_formatter(data)
        # torch right-aligns the elements to one width — with negatives in the
        # mix, room appears in front of the positives.
        width = _field_width(data, fmt)
        padded = lambda v, f=fmt, w=width: f(v).rjust(w)
        body = _np.array2string(
            data, formatter={"float_kind": padded}, separator=", ",
            max_line_width=_LINE_WIDTH - 8, threshold=1000)
    elif data.dtype.kind == "c":
        # **The real and imaginary parts are measured separately** (measured).
        # In `[1+2j, -0.5-1j]` the real part demands four decimal places and the
        # imaginary part is integral, so torch prints `1.0000+2.j` — measured
        # under one format it becomes `1.0000+2.0000j` and the characters
        # diverge.
        #
        # **The padding applies to the real part only** (measured). The
        # imaginary part is not pushed and the sign is the value's own sign — so
        # a **negative zero** keeps its sign, as in `1.-0.j`.
        re_fmt = _float_formatter(data.real)
        im_fmt = _float_formatter(data.imag)
        width = _field_width(data.real, re_fmt)

        def one(v, rf=re_fmt, mf=im_fmt, w=width):
            im = mf(v.imag)
            return f"{rf(v.real).rjust(w)}{im if im.startswith('-') else '+' + im}j"

        # **Where the line breaks is part of the specification too.** torch
        # counts how many fit on a line by **width** rather than by characters —
        # `floor((linewidth − 7) / (real width + imag width + 3))`. numpy breaks
        # by actual character length, so the budget is recomputed and handed over
        # to make it break in the same place. Using the real path's
        # `_LINE_WIDTH - 8` as-is folds twelve elements as 5+5+2 rather than 6+6,
        # and then every value is right and the characters diverge.
        im_width = _field_width(data.imag, im_fmt)
        per_line = max(1, (_LINE_WIDTH - 7) // (width + im_width + 3))
        budget = per_line * (width + im_width + 4)
        body = _np.array2string(
            data, formatter={"complex_kind": one}, separator=", ",
            max_line_width=budget, threshold=1000)
    else:
        body = _np.array2string(data, separator=", ",
                                max_line_width=_LINE_WIDTH - 8, threshold=1000)
    # numpy indents continuation lines by one. torch indents by the width of
    # "tensor(" — eight.
    return body.replace("\n ", "\n" + " " * 8)


def _tensor_repr(t):
    parts = [_tensor_str(t.data)]
    dt = t.data.dtype
    plain = dt in (_np.dtype("float32"), _np.dtype("int64"), _np.dtype("bool"))
    # **complex64 does not print its dtype when there are values** (measured).
    # The trailing `j` already says it is complex, so torch omits it. **An empty
    # tensor has no such clue, so it is printed** —
    # `tensor([], dtype=torch.complex64)`. The rule hangs on **whether the clue
    # is there** rather than on the dtype, so put into a list of dtypes alone it
    # diverges on the empty case.
    if dt == _np.dtype("complex64") and t.data.size > 0:
        plain = True
    if not plain:
        parts.append(f"dtype={t.dtype}")
    if t._op:
        parts.append(f"grad_fn=<{t._op}>")
    elif t.requires_grad:
        parts.append("requires_grad=True")
    return f"tensor({', '.join(parts)})"


# ---------------------------------------------------------------- Size

class device:                                                   # noqa: N801
    """`torch.device` — **a label naming a device.**

    This name was absent for a long time, and it was the largest gap on the
    list:

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

    It is **the first line of half the tutorials**, and without the name it
    stops there with an `AttributeError` — even though `cuda.is_available()` is
    false and what actually gets built is `cpu`.

    **Making one and using one are separated.** `torch.device("cuda")` **is built
    even with no such hardware** (measured — torch is the same). Stopping there
    means the ternary above cannot run at all, and then the learner reads it as
    their own code being wrong. The place to stop is **moving a tensor** to that
    device, and the wording there names the cause.
    """

    __slots__ = ("type", "index")

    def __init__(self, kind, index=None):
        if isinstance(kind, device):
            self.type, self.index = kind.type, kind.index
            return
        text = str(kind)
        if ":" in text:
            text, _, tail = text.partition(":")
            index = int(tail)
        self.type = text
        self.index = None if index is None else int(index)

    def __repr__(self):
        tail = "" if self.index is None else f", index={self.index}"
        return f"device(type='{self.type}'{tail})"

    def __str__(self):
        return self.type if self.index is None else f"{self.type}:{self.index}"

    def __eq__(self, other):
        # **Not equal to a string** (measured: `torch.device("cpu") == "cpu"` is
        # false). Answering true out of leniency makes `if d == "cpu":` run here
        # and not against real torch — being more permissive is still diverging,
        # and this kind changes **which way a conditional goes.**
        return (isinstance(other, device) and self.type == other.type
                and self.index == other.index)

    def __hash__(self):
        return hash((self.type, self.index))


class Size(tuple):
    def __repr__(self):
        return f"torch.Size([{', '.join(str(x) for x in self)}])"


