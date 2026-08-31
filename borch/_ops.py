"""A piece of borch, split out. __init__ gathers the public names."""

import builtins as _builtins
import inspect as _inspect
import itertools as _itertools
import math as _math

import warnings as _warnings

import numpy as _np

from ._tensor import (
    # **Imported under a different name on purpose.** Defining the public
    # `result_type` below shadowed this one at module scope, and the seven
    # `floor_divide` rows on the dtype axis started raising `TypeError:
    # Cannot interpret 'torch.float32' as a data type` — the third name in
    # this package shadowed by a torch-shaped one today, after `any` and
    # `type`. The two functions ask different questions and now say so.
    Tensor, _MinMax, _grad_mode, _unbroadcast,
    result_type as _dtype_result_type,
)
from ._base import (
    _DEFAULT_DTYPE, _NP_TO_DTYPE, _TYPE_NAMES, _arith_in, _float_in, _like_torch,
    _math,
    _needs_float,
    _np, _refuses_bool,
    _refuses_nonfloat_kernel, _requested_dtype, _resolve, _unsupported, Size,
    device as _device,
    dtype,
)
# **Imported under different names.** A name like `float32` at this file's module
# scope collides with a function below that uses the name — shadowing `bool` that
# way happened once already.
# **`from . import _fft` must not be used.** That form looks at the parent
# package's **attribute**, and while this file runs `borch/__init__` is still
# half initialised. It passed natively and stopped **inside Pyodide alone** with
# `cannot import name '_fft' from partially initialized module` — one golden case,
# `repr::스칼라`, caught it. Importing the submodule directly does not look at
# that attribute.
from ._fft import fft as _fft_fft, fftfreq as _fft_fftfreq
from ._fft import fftshift as _fft_fftshift, ifft as _fft_ifft
from ._fft import ifftshift as _fft_ifftshift, irfft as _fft_irfft
from ._fft import istft as _fft_istft, rfft as _fft_rfft
# Multi-axis and Hermitian — all assembled from the four above, so their bodies
# are in the same file.
from ._fft import fft2 as _fft_fft2, fftn as _fft_fftn
from ._fft import hfft as _fft_hfft, hfft2 as _fft_hfft2
from ._fft import hfftn as _fft_hfftn, ifft2 as _fft_ifft2
from ._fft import ifftn as _fft_ifftn, ihfft as _fft_ihfft
from ._fft import ihfft2 as _fft_ihfft2, ihfftn as _fft_ihfftn
from ._fft import irfft2 as _fft_irfft2, irfftn as _fft_irfftn
from ._fft import rfft2 as _fft_rfft2, rfftn as _fft_rfftn
from ._fft import rfftfreq as _fft_rfftfreq, stft as _fft_stft
from ._base import bool_ as _bool_dtype, float32 as _float32
from ._base import float64 as _float64, int64 as _int64

# ------------------------------------------------------------------- factories

def tensor(data, dtype=None, requires_grad=False):
    """**It always takes a copy.** torch is documented that way, and `from_numpy`
    is what to use for sharing.

    With `_np.asarray` alone, an ndarray that is already the right dtype **passes
    straight through and is shared.** Then `t = torch.tensor(arr); t.add_(1)`
    changes the user's `arr` as well — with no exception and no warning, and
    since real torch does not do it, that code runs differently on their own
    machine.

    This repository's case file really was bitten by it. One case that shared its
    input raised `plain` by 1 in place, and **torch took a copy and did not leak
    while the core did.** So the cases that came after were wrong in the core
    alone, and with the cause absent from their own case it took sixteen places
    of wandering.
    """
    if isinstance(data, Tensor):
        data = data.data
    return Tensor(_np.array(data, dtype=_resolve(data, dtype), copy=True),
                  requires_grad)


def as_tensor(data, dtype=None):
    if isinstance(data, Tensor) and dtype is None:
        return data
    if isinstance(data, Tensor):
        return Tensor(data.data.astype(_requested_dtype(dtype).np))
    return tensor(data, dtype)


def from_numpy(arr):
    return Tensor(arr)


def _np_of(dt):
    """Take the numpy dtype out of a dtype. **A dtype that is only a name stops
    here with its own wording.**

    It must not be asked with `hasattr(dt, "np")` — that place is a gate left
    open in order to stop, and `hasattr` tries to swallow it. What it is gets
    asked first.
    """
    return dt.np if isinstance(dt, dtype) else _np.dtype(dt)


def _made(arr, dt=None, requires_grad=False):
    """Apply **`dtype=` and `requires_grad=`** to the array a factory made.

    **Without this gate the factories answered differently.** `zeros` heard both,
    `zeros_like` took `dtype=` and did not use it (the values are right and only
    the dtype is wrong, so a value comparison does not catch it), and `rand` did
    not take `requires_grad=` at all, so `rand(3, requires_grad=True)` stopped
    with a `TypeError`. **A mix of what works and what does not leaves a learner
    unable to form the rule** — so fourteen of them are gathered into one gate.

    The `requires_grad` side is worse. A wrong dtype eventually catches the eye,
    and a leaf with no gradient attached means **that one parameter quietly does
    not move while the loss goes down.**
    """
    arr = _np.asarray(arr)
    if dt is not None:
        # The fourteen factories gathered here are the other place a caller
        # **names** a dtype, so double precision refuses here as it does in
        # `tensor()`. See `_base._requested_dtype`.
        arr = arr.astype(_np_of(_requested_dtype(dt)))
    return Tensor(arr, requires_grad)


def zeros(*shape, dtype=None, requires_grad=False, device=None):
    shape = shape[0] if len(shape) == 1 and isinstance(shape[0], (tuple, list)) else shape
    return _made(_np.zeros(shape, dtype=_DEFAULT_DTYPE), dtype, requires_grad)


def ones(*shape, dtype=None, requires_grad=False, device=None):
    shape = shape[0] if len(shape) == 1 and isinstance(shape[0], (tuple, list)) else shape
    return _made(_np.ones(shape, dtype=_DEFAULT_DTYPE), dtype, requires_grad)


def full(shape, value, dtype=None, requires_grad=False):
    return _made(_np.full(shape, value, dtype=_DEFAULT_DTYPE), dtype, requires_grad)


def zeros_like(t, dtype=None, requires_grad=False):
    return _made(_np.zeros_like(t.data if isinstance(t, Tensor) else t),
                 dtype, requires_grad)


def ones_like(t, dtype=None, requires_grad=False):
    return _made(_np.ones_like(t.data if isinstance(t, Tensor) else t),
                 dtype, requires_grad)


def full_like(t, value, dtype=None, requires_grad=False):
    return _made(_np.full_like(t.data, value), dtype, requires_grad)


def _needs_step(step, who):
    """A step of 0 **stops here.** A place where all three have to say the same
    thing.

    Unblocked, numpy raises `ZeroDivisionError: division by zero`, and that reads
    as a different story from torch's `step must be nonzero` — it looks like a
    division went wrong, and in fact a step of 0 means the value never moves and
    there is no answer.
    """
    if step == 0:
        raise RuntimeError(
            f"{who}: step must be nonzero — with a step of 0 the value never moves "
            "and the range never ends.")


def arange(*args, dtype=None, requires_grad=False):
    if len(args) == 3:
        _needs_step(args[2], "arange")
    # **This is the one place that does not fix a default dtype** — numpy
    # choosing from the arguments is what torch does (integers alone give int64,
    # one float anywhere gives a float). Left to `_made` that rule breaks.
    return _made(_np.arange(*args, dtype=(_requested_dtype(dtype).np if dtype else None)),
                 requires_grad=requires_grad)


def linspace(start, end, steps, dtype=None, requires_grad=False):
    return _made(_np.linspace(start, end, steps, dtype=_DEFAULT_DTYPE),
                 dtype, requires_grad)


def eye(n, m=None, dtype=None, requires_grad=False):
    return _made(_np.eye(n, n if m is None else m, dtype=_DEFAULT_DTYPE),
                 dtype, requires_grad)


_rng = _np.random.default_rng(0)


class Generator:
    """A container carrying a seed. `random_split(generator=...)` takes this —
    without the split fixed there is no telling whether changing the model helped
    or the split got lucky."""

    def __init__(self):
        self.seed = 0

    def manual_seed(self, seed):
        self.seed = seed
        return self

    def rng(self):
        return _np.random.default_rng(self.seed)


def manual_seed(seed):
    """Plant a seed. **It swaps the state of the existing generator — it does not
    build a new one.**

    The earlier version rebound the name with `global _rng`. And `_nn.py` grabs
    **the object as it stood at that moment** with `from ._ops import _rng` at
    import time, so rebinding leaves that side going on with the old generator.
    The result was this (measured):

        borch.manual_seed(0); Linear(4,3).weight   ← a different value every time
        borch.manual_seed(0); borch.randn(3)       ← reproduces

    **Layer initialisation and dropout did not follow the seed.** Those two are
    the first things somebody trying to reproduce a run expects, and with `randn`
    alone reproducing they read it as "the seed works" and move on. The golden
    did not see it for a long time because every case supplies its weights from
    outside — the place was first asked about when the lazy layers began
    initialising themselves.

    Swapping the state fixes it **wherever anybody grabbed it from.** Rebinding
    the name means finding and fixing every place that grabbed it, and that list
    grows.
    """
    # torch takes a 0-D tensor here as well as an int, and numpy's generator does
    # not — `SeedSequence expects int or sequence of ints`. Coerced rather than
    # excused: the fix is one line and an excuse would be a sentence to keep true.
    seed = int(seed)
    _rng.bit_generator.state = _np.random.default_rng(seed).bit_generator.state
    _LAST_SEED[0] = seed
    return seed


# The seed planted most recently. `initial_seed` answers with this.
_LAST_SEED = [0]


def initial_seed():
    """The seed planted most recently. Without a `manual_seed` call it is 0 —
    a fact, since our generator starts from that seed."""
    return _LAST_SEED[0]


def seed():
    """Plant **an arbitrary** new seed and answer with it. torch does the
    same."""
    got = int(_np.random.SeedSequence().entropy % (2 ** 63))
    manual_seed(got)
    return got


def get_rng_state():
    """Hand over the generator's whole state. **Our state as it is, rather than
    a tensor.**

    torch gives a `uint8` tensor holding bytes. That byte layout is the inside of
    torch's Mersenne Twister, so there is nothing to imitate, and imitating it
    creates a situation where **somebody else's state gets read as ours.** Here
    the generator's state is carried as it is and `set_rng_state` takes only
    that — the pair matches on the way out and back, so resuming training does
    what it says.
    """
    return dict(_rng.bit_generator.state)


def set_rng_state(state):
    """Restore what `get_rng_state` gave. It takes nothing but its partner."""
    if not isinstance(state, dict):
        _unsupported("set_rng_state — it only takes what `get_rng_state` returned")
    _rng.bit_generator.state = state
    return None


def randn(*shape, requires_grad=False):
    shape = shape[0] if len(shape) == 1 and isinstance(shape[0], (tuple, list)) else shape
    return Tensor(_rng.standard_normal(shape).astype(_DEFAULT_DTYPE), requires_grad)


def rand(*shape, dtype=None, requires_grad=False, device=None):
    shape = shape[0] if len(shape) == 1 and isinstance(shape[0], (tuple, list)) else shape
    return _made(_rng.random(shape).astype(_DEFAULT_DTYPE), dtype, requires_grad)


def _out(result, out, name="op"):
    """torch's `out=` convention. **Written down from real torch rather than
    guessed.**

    Five things are observed:

    - The result is written into `out` and **`out` itself is returned**
      (`result is out` is true).
    - **A different shape re-allocates `out`** — a warning rather than an error.
      Larger or smaller, the same.
    - **The dtype rule is `can_cast`** (measured). float32→int64 stops and
      float64→float32 works. Precision is free and only **narrowing the
      category** is blocked.
    - **Requiring gradients stops it**, on `out` or on the input. `out=` does not
      take differentiation.
    - A reduction cannot take `out=` without `dim` — that is torch's signature
      and not something built here.

    **The saving does not happen here.** The result is built and then moved.
    `out=` exists in order to avoid the allocation and that cannot be given —
    and the destination changing and the same object coming back are **facts
    rather than performance**, so those two are kept.
    """
    if out is None:
        return result
    # The ones producing several things (`sort`, `topk`, `svd`, …) take
    # `out=(a, b)`.
    if isinstance(out, (tuple, list)):
        parts = [_out(r, o, name) for r, o in zip(tuple(result), out)]
        # **`hasattr(result, "_fields")` is a namedtuple's question, and the
        # results carrying names here are not namedtuples.** `_named` builds a
        # class with `__slots__`, so the test came back false for every one of
        # them and `qr(A, out=(Q, R))` handed back a plain tuple — right values,
        # no `.Q`. That is the exact failure `_named` was written to end, let
        # back in through the one path that rebuilds the container.
        #
        # Asking about the container instead: anything that is not literally a
        # `tuple`/`list` gets rebuilt as itself.
        return (tuple(parts) if type(result) in (tuple, list)
                else type(result)(*parts))
    if result.requires_grad or out.requires_grad:
        raise RuntimeError(
            f"{name}(): functions with out=... arguments don't support automatic "
            "differentiation, but one of the arguments requires grad.")
    if not can_cast(_NP_TO_DTYPE[_np.dtype(result.data.dtype)],
                    _NP_TO_DTYPE[_np.dtype(out.data.dtype)]):
        raise RuntimeError(
            f"result type {_TYPE_NAMES[result.data.dtype.kind]} can't be cast to "
            f"the desired output type {_TYPE_NAMES[out.data.dtype.kind]}")
    if out.data.shape != result.data.shape:
        _warnings.warn(
            f"An output with one or more elements was resized since it had shape "
            f"{list(out.data.shape)}, which does not match the required output shape "
            f"{list(result.data.shape)}.", UserWarning, stacklevel=3)
        out._array = _np.empty(result.data.shape, dtype=out.data.dtype)
    out._array[...] = result.data.astype(out.data.dtype, copy=False)
    return out


def _no_out(kw):
    """`out=` is **not swallowed quietly.**

    torch takes `out=` at 198 names and writes into that tensor. This does not,
    and in most places there is no `**kw`, so Python stops with a `TypeError`.
    The trouble is the places that take `**kw` — six of them were taking `out=`
    and **discarding it.** With no error, code writing
    `randint(0, 5, (4,), out=buf)` moves to the next line with `buf` still zero.

    **Being absent and swallowing are different things.** Stopping says so at
    that point; swallowing surfaces much later as a stray value.

    **It takes the value now, not the bag.** The bag was the problem: `**kw` had to
    exist for this gate to read `out` out of it, and while it existed it also
    accepted every other keyword and dropped it. Seventeen functions in this module
    swallowed anything at all — `bernoulli(x, dtype=…)`, `asarray(v, zzz=1)` — where
    torch raises, and seventeen more only ever touched `kw` to hand it here.

    So `out` is written as a keyword-only parameter at each of those seats and the
    bag is gone. `out=` keeps this wording, which says *why* it is absent; everything
    else gets Python's own `unexpected keyword argument`, which is one of the two
    forms torch itself uses and is exact for free.

    Called with `None` this must do nothing — `out=None` is the default and is not a
    request. The first version checked `"out" in kw` and a literal translation of it
    would refuse every call.

    `tests/test_no_silent_out.py` no longer reads the source for `**kw`, because
    removing the bag emptied that population and the check would have gone green by
    having nothing to look at. It calls instead.
    """
    if isinstance(kw, dict):
        if "out" in kw:
            _unsupported("`out=` (writing into a tensor you made beforehand)")
        return
    if kw is not None:
        _unsupported("`out=` (writing into a tensor you made beforehand)")


def randint(low, high, shape, dtype=None, requires_grad=False, *, out=None):
    _no_out(out)
    return _made(_rng.integers(low, high, shape).astype(_np.int64),
                 dtype, requires_grad)


def randperm(n, dtype=None, requires_grad=False, *, out=None):
    _no_out(out)
    return _made(_rng.permutation(n).astype(_np.int64), dtype, requires_grad)


def multinomial(probs, num_samples, replacement=True, *, generator=None):
    if generator is not None:
        _unsupported("multinomial(generator=…)")
    p = probs.data / probs.data.sum(axis=-1, keepdims=True)
    if p.ndim == 1:
        return Tensor(_rng.choice(len(p), size=num_samples, p=p).astype(_np.int64))
    out = [_rng.choice(p.shape[-1], size=num_samples, p=row) for row in p]
    return Tensor(_np.asarray(out, dtype=_np.int64))


# ------------------------------------------------------------------- functions

def _wrap(t):
    return t if isinstance(t, Tensor) else Tensor(_np.asarray(t))


def stack(items, dim=0):
    items = [_wrap(t) for t in items]
    # **Mismatched shapes came out as numpy's `ValueError`.** torch raises
    # `RuntimeError` and names which two entries disagree; numpy's says only that
    # the arrays differ, and a torch user's `except RuntimeError` does not catch it.
    for i, one in enumerate(items[1:], 1):
        if one.data.shape != items[0].data.shape:
            raise RuntimeError(
                f"stack expects each tensor to be equal size, but got "
                f"{list(items[0].data.shape)} at entry 0 and "
                f"{list(one.data.shape)} at entry {i}")
    out = _np.stack([t.data for t in items], axis=dim)
    if not items:
        return Tensor(out)
    return items[0]._make(
        out, tuple(items),
        lambda g: tuple(_np.take(_np.asarray(g), i, axis=dim) for i in range(len(items))),
        "StackBackward0")


def cat(items, dim=0):
    items = [_wrap(t) for t in items]
    out = _np.concatenate([t.data for t in items], axis=dim)
    sizes = [t.data.shape[dim] for t in items]

    def back(g):
        g = _np.asarray(g)
        cuts = _np.cumsum(sizes)[:-1]
        return tuple(_np.split(g, cuts, axis=dim))

    return items[0]._make(out, tuple(items), back, "CatBackward0")


_MISSING = object()


def where(cond, a=_MISSING, b=_MISSING):
    """**Given only the condition it answers with positions**, not with values —
    torch's one-argument form is `nonzero(as_tuple=True)` under another name, and
    it gives a tuple with one tensor per axis.

    It was found by a probe asking about **rank**, which is not what it is. The
    probe fed one tensor to every unary-looking name and read which ranks came
    back; here torch answered at every rank and this library at none, so it landed
    in the report as a rank row. It is an arity row. A probe names what it finds
    after the axis it was built for, and the name can be wrong.
    """
    if a is _MISSING and b is _MISSING:
        return nonzero(_wrap(cond), as_tuple=True)
    if a is _MISSING or b is _MISSING:
        raise RuntimeError("where() expects either one argument or three, but got two")
    c = cond.data if isinstance(cond, Tensor) else cond
    ta, tb = _wrap(a), _wrap(b)
    out = _np.where(c, ta.data, tb.data)
    return ta._make(out, (ta, tb), lambda g: (_np.where(c, g, 0), _np.where(c, 0, g)))


def sigmoid(input):
    out = 1.0 / (1.0 + _np.exp(-_np.clip(_float_in(input.data), -60, 60)))
    return input._make(out, (input,), lambda g: (g * out * (1 - out),), "SigmoidBackward0")


def relu(input, inplace=False):
    """torch takes `inplace` here; it is the underscore name by another
    spelling, routed through the same write-back rather than a second
    formula."""
    return _inplace_arg(input, inplace, "relu",
                        lambda: _relu_body(input))


def _relu_body(t):
    return t._make(_np.maximum(t.data, 0), (t,), lambda g: (g * (t.data > 0),), "ReluBackward0")


def tanh(input):
    out = _np.tanh(_float_in(input.data))
    return input._make(out, (input,), lambda g: (g * (1 - out * out),), "TanhBackward0")


def exp(input): return input.exp()
def log(input): return input.log()
def sqrt(input): return input.sqrt()
def abs(input): return input.abs()


def _default_softmax_dim(ndim):
    """The axis torch chooses when `dim` is not given.

    **It is not `-1`.** It is 0 or 1 depending on the rank, and torch even warns
    at that point ("Implicit dimension choice for softmax has been deprecated").
    The rule was measured — rank 1 → 0, 2 → 1, 3 → **0**, 4 → 1.

    **Asked at rank 2 only, this defect is invisible.** There `dim=1` and
    `dim=-1` are the same axis, so a default of `-1` gives the same answer. It
    really was left that way, quietly folding the wrong axis at rank 3.

    It was written in `_nn.py` for the layers alone, and the three functions here
    defaulted to `-1` — **the same rule known on one side of the file and not the
    other**, which is the shape `nll_loss` and `embedding` were found in today.
    """
    return 0 if ndim in (0, 1, 3) else 1


def _softmax_args(t, dim, dtype, stacklevel, name):
    """The three arguments torch's softmax family shares, resolved in one place.

    `dtype` casts **before** the softmax rather than after (measured: an integer
    input with `dtype=float32` gives the real answer, not a cast of an integer one).

    `_stacklevel` is torch's own private plumbing and its only observable effect is
    where the deprecation warning points. Carrying it is not decoration: it sits
    third, so `F.softmax(x, 1, 3, float32)` — a call torch takes — needs the seat, and
    honouring it means the warning lands on the caller rather than in here.
    """
    t = _wrap(t)
    if dtype is not None:
        t = t.to(dtype)
    if dim is None:
        _warnings.warn(
            f"Implicit dimension choice for {name} has been deprecated. "
            "Change the call to include dim=X as an argument.",
            UserWarning, stacklevel=stacklevel)
        dim = _default_softmax_dim(t.data.ndim)
    return t, dim


def softmax(input, dim=None, _stacklevel=3, dtype=None):   # noqa: A002
    t, dim = _softmax_args(input, dim, dtype, _stacklevel, "softmax")
    shifted = t.data - t.data.max(axis=dim, keepdims=True)
    e = _np.exp(shifted)
    out = e / e.sum(axis=dim, keepdims=True)

    def back(g):
        s = (g * out).sum(axis=dim, keepdims=True)
        return ((out * (g - s)),)

    return t._make(out, (t,), back, "SoftmaxBackward0")



def _pair(v):
    """It takes `3` and `(3, 3)` alike — torch does that."""
    return (v, v) if isinstance(v, int) else tuple(v)


def _pad2d(x, padding):
    ph, pw = _pair(padding)
    if ph == 0 and pw == 0:
        return x
    return _np.pad(x, ((0, 0), (0, 0), (ph, ph), (pw, pw)))


def _im2col(xd, KH, KW, stride, dilation=1):
    """Spread (N,C,H,W) into (N*OH*OW, C*KH*KW). So that one GEMM finishes the
    convolution.

    **Dilation widens the window and then thins it.** A dilated filter covers
    `(K-1)·d + 1` cells and uses every `d`-th one, so the sliding view is taken at
    the covered size and sliced with a step — which keeps the column layout
    identical, and `_col2im` only has to know where each filter position landed.
    """
    sh, sw = _pair(stride)
    dh, dw = _pair(dilation)
    N, C, H, W = xd.shape
    span_h, span_w = (KH - 1) * dh + 1, (KW - 1) * dw + 1
    OH = (H - span_h) // sh + 1
    OW = (W - span_w) // sw + 1
    win = _np.lib.stride_tricks.sliding_window_view(xd, (span_h, span_w), axis=(2, 3))
    win = win[:, :, ::sh, ::sw, ::dh, ::dw]            # (N, C, OH, OW, KH, KW)
    cols = win.transpose(0, 2, 3, 1, 4, 5)             # (N, OH, OW, C, KH, KW)
    return _np.ascontiguousarray(cols).reshape(N * OH * OW, C * KH * KW), OH, OW


def _col2im(gcols, shape, KH, KW, stride, OH, OW, dilation=1):
    """The inverse of im2col. It loops over **the filter positions (KH×KW)**
    rather than the output positions (OH×OW) — on a 28×28 image that is 9 rounds
    instead of 784.

    Filter position `(i, j)` lands at `i·dh` under dilation, which is the only
    thing dilation changes here.
    """
    sh, sw = _pair(stride)
    dh, dw = _pair(dilation)
    N, C, H, W = shape
    gx = _np.zeros(shape, dtype=gcols.dtype)
    g = gcols.reshape(N, OH, OW, C, KH, KW).transpose(0, 3, 4, 5, 1, 2)   # (N,C,KH,KW,OH,OW)
    for i in range(KH):
        for j in range(KW):
            top, left = i * dh, j * dw
            gx[:, :, top:top + OH * sh:sh, left:left + OW * sw:sw] += g[:, :, i, j]
    return gx


def _grouped(call, x, weight, bias, groups, channel_axis=1):
    """`groups` by composition: slice the channels, convolve each group, join.

    **Written as slicing and joining rather than inside the GEMM** because the
    gradient then follows from the pieces — `cat` and the slice already carry
    theirs, and a hand-written backward for the grouped case would be a second
    formula that has to agree with the first. The convolutions in this file each
    carry one such formula already; two per operation is where they part.
    """
    x, weight = _wrap(x), _wrap(weight)
    in_ch = x.data.shape[channel_axis]
    out_ch = weight.data.shape[0]
    if in_ch % groups or out_ch % groups:
        raise RuntimeError(
            f"groups={groups} divides neither the input channels ({in_ch}) nor "
            f"the filters ({out_ch})")
    cin, cout = in_ch // groups, out_ch // groups
    parts = []
    for g in range(groups):
        xs = x[_slice_at(channel_axis, g * cin, (g + 1) * cin)]
        ws = weight[_slice_at(0, g * cout, (g + 1) * cout)]
        bs = None if bias is None else _wrap(bias)[_slice_at(0, g * cout, (g + 1) * cout)]
        parts.append(call(xs, ws, bs))
    return cat(parts, channel_axis)


def conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    """A convolution for small inputs. The same computation as the double loop
    written by hand in chapter 26.

    Spread through im2col and finished with one matmul — numpy calls BLAS, so it
    is over 20× faster (measured) than looping the windows with einsum. Fast as
    it is, real training happens on real torch.

    **stride and padding are taken per axis.** Taking squares only makes it
    impossible to build `conv1d` on top — 1-D works by inserting a height of 1,
    so the padding on the height axis has to be 0. The sister library already
    takes them per axis, so this matches it.
    """
    if groups != 1:
        return _grouped(
            lambda xs, ws, bs: conv2d(xs, ws, bs, stride, padding, dilation),
            input, weight, bias, groups)
    xd = _pad2d(input.data, padding)
    wd = weight.data
    ph, pw = _pair(padding)
    dh, dw = _pair(dilation)
    N, C, H, W = xd.shape
    F, C2, KH, KW = wd.shape
    if C != C2:
        raise RuntimeError(f"channels do not match: input {C}, filter {C2}")
    if H < (KH - 1) * dh + 1 or W < (KW - 1) * dw + 1:
        raise RuntimeError("the filter is larger than the input.")

    cols, OH, OW = _im2col(xd, KH, KW, stride, dilation)
    w2 = wd.reshape(F, -1)
    out = (cols @ w2.T).reshape(N, OH, OW, F).transpose(0, 3, 1, 2)

    def back(g):
        g = _np.asarray(g)
        g2 = g.transpose(0, 2, 3, 1).reshape(-1, F)
        gw = (g2.T @ cols).reshape(wd.shape)
        gx = _col2im(g2 @ w2, xd.shape, KH, KW, stride, OH, OW, dilation)
        if ph:
            gx = gx[:, :, ph:-ph, :]
        if pw:
            gx = gx[:, :, :, pw:-pw]
        return (gx, gw) if bias is None else (gx, gw, g.sum(axis=(0, 2, 3)))

    parents = (input, weight) if bias is None else (input, weight, bias)
    shifted = out if bias is None else out + bias.data.reshape(1, -1, 1, 1)
    return input._make(shifted, parents, back)


def _grouped_transpose(call, x, weight, bias, groups):
    """`groups` for a transposed convolution.

    **The weight axes are the other way round here** — `(in, out/groups, …)` — so
    the slice that walks the groups runs down axis 0 for the input side and the
    bias is cut by axis 1. Written apart from `_grouped` for that reason alone: one
    helper taking an axis argument reads as though the two were the same shape, and
    they are the trap this file already warns about two functions up.
    """
    x, weight = _wrap(x), _wrap(weight)
    in_ch = x.data.shape[1]
    per_group_out = weight.data.shape[1]
    if in_ch % groups:
        raise RuntimeError(
            f"groups={groups} does not divide the input channels ({in_ch})")
    cin = in_ch // groups
    parts = []
    for g in range(groups):
        xs = x[_slice_at(1, g * cin, (g + 1) * cin)]
        ws = weight[_slice_at(0, g * cin, (g + 1) * cin)]
        bs = (None if bias is None else
              _wrap(bias)[_slice_at(0, g * per_group_out, (g + 1) * per_group_out)])
        parts.append(call(xs, ws, bs))
    return cat(parts, 1)


def conv_transpose2d(input, weight, bias=None, stride=1, padding=0, output_padding=0,
                     groups=1, dilation=1):
    """A transposed convolution — **the same computation `conv2d` flows towards
    its input.**

    No new kernel is written. `conv2d`'s backward already does that job through
    `col2im`, and two copies of the same computation means the day comes when one
    is fixed and they diverge.

    **The weight axes are reversed from `conv2d`'s** — `(in, out, kh, kw)`. With a
    square kernel the shape fits even reversed, so it diverges only in the values.
    The commonest mistake in this layer.
    """
    if groups != 1:
        return _grouped_transpose(
            lambda xs, ws, bs: conv_transpose2d(xs, ws, bs, stride, padding,
                                                output_padding, 1, dilation),
            input, weight, bias, groups)
    input, weight = _wrap(input), _wrap(weight)
    N, C, H, W = input.data.shape
    C2, F, KH, KW = weight.data.shape
    if C != C2:
        raise RuntimeError(f"channels do not match: input {C}, filter {C2}")
    sh, sw = _pair(stride)
    ph, pw = _pair(padding)
    dh, dw = _pair(dilation)
    oph, opw = _pair(output_padding)
    if oph >= sh and oph >= dh or opw >= sw and opw >= dw:
        raise RuntimeError(
            "output padding must be smaller than either stride or dilation, but "
            f"got output_padding={output_padding}, stride={stride}, "
            f"dilation={dilation}")
    OH = (H - 1) * sh + (KH - 1) * dh + 1
    OW = (W - 1) * sw + (KW - 1) * dw + 1

    # Each input cell is scattered across the kernel into the output positions.
    # `col2im` is exactly that shape — the input is spread to `(N·H·W, C)`,
    # multiplied by the weights into `(N·H·W, F·KH·KW)` and then folded.
    cols = input.data.transpose(0, 2, 3, 1).reshape(N * H * W, C)
    w2 = weight.data.reshape(C, F * KH * KW)
    spread = cols @ w2
    out = _col2im(spread, (N, F, OH, OW), KH, KW, (sh, sw), H, W, (dh, dw))

    def back(g):
        g = _np.asarray(g)
        gcols, _, _ = _im2col(g, KH, KW, (sh, sw), (dh, dw))   # (N·H·W, F·KH·KW)
        gx = (gcols @ w2.T).reshape(N, H, W, C).transpose(0, 3, 1, 2)
        gw = (cols.T @ gcols).reshape(weight.data.shape)
        got = (gx, gw)
        return got if bias is None else got + (g.sum(axis=(0, 2, 3)),)

    if ph or pw or oph or opw:
        # The padding is **trimmed off the output** — the opposite direction from an
        # ordinary convolution — and `output_padding` extends the window at the
        # bottom and the right.
        #
        # **The extra rows are not zeros.** `output_padding` reaches back into the
        # part the trim was about to throw away, which holds computed values; only
        # where it runs past the untrimmed output is there nothing to take. Filling
        # them with zeros instead matches the shape exactly and differs in the
        # values — measured, on twelve of fifty-six configurations, all of them
        # `padding` and `output_padding` together.
        bottom, right = OH - ph + oph, OW - pw + opw
        kept = out[:, :, ph:min(bottom, OH), pw:min(right, OW)]
        tail_h, tail_w = max(0, bottom - OH), max(0, right - OW)
        if tail_h or tail_w:
            kept = _np.pad(kept, ((0, 0), (0, 0), (0, tail_h), (0, tail_w)))
        out = kept

        def back(g, _inner=back, _ph=ph, _pw=pw, _oh=OH, _ow=OW,
                 _bottom=bottom, _right=right):                   # noqa: F811
            g = _np.asarray(g)
            keep_h, keep_w = min(_bottom, _oh) - _ph, min(_right, _ow) - _pw
            full = _np.zeros((N, F, _oh, _ow), dtype=g.dtype)
            full[:, :, _ph:_ph + keep_h, _pw:_pw + keep_w] = g[:, :, :keep_h, :keep_w]
            return _inner(full)

    if bias is not None:
        out = out + bias.data.reshape(1, -1, 1, 1)
    parents = (input, weight) if bias is None else (input, weight, bias)
    return input._make(out, parents, back, "ConvTranspose2DBackward0")


def conv_transpose1d(input, weight, bias=None, stride=1, padding=0, output_padding=0,
                     groups=1, dilation=1):
    """Insert a height of 1 into `conv_transpose2d` — the same way as
    `conv1d`."""
    input, weight = _wrap(input), _wrap(weight)
    n, c, length = input.data.shape
    c2, f, k = weight.data.shape
    out = conv_transpose2d(input.reshape(n, c, 1, length), weight.reshape(c2, f, 1, k),
                           bias, (1, stride), (0, padding), (0, output_padding),
                           groups, (1, dilation))
    shape = out.data.shape
    return out.reshape(shape[0], shape[1], shape[3])


def conv_transpose3d(input, weight, bias=None, stride=1, padding=0, output_padding=0,
                     groups=1, dilation=1):
    """Run a 2-D transposed convolution per depth and **add where they overlap.**

    The same way as `conv3d` — no separate 3-D kernel is written, so there is no
    new derivative to write.
    """
    if groups != 1:
        return _grouped_transpose(
            lambda xs, ws, bs: conv_transpose3d(xs, ws, bs, stride, padding,
                                                output_padding, 1, dilation),
            input, weight, bias, groups)
    input, weight = _wrap(input), _wrap(weight)
    n, c, d, h, w = input.data.shape
    c2, f, kd, kh, kw = weight.data.shape
    if c != c2:
        raise RuntimeError(f"channels do not match: input {c}, filter {c2}")
    sd, sh, sw = (stride,) * 3 if isinstance(stride, int) else tuple(stride)
    pd, ph, pw = (padding,) * 3 if isinstance(padding, int) else tuple(padding)
    opd, oph, opw = ((output_padding,) * 3 if isinstance(output_padding, int)
                     else tuple(output_padding))
    dd, dh, dw = (dilation,) * 3 if isinstance(dilation, int) else tuple(dilation)
    out_d = (d - 1) * sd + (kd - 1) * dd + 1

    # The per-depth results are gathered into a list and stacked at once. Several
    # input depths overlap at each position, so they **have to be added** —
    # overwriting leaves only the last one.
    slabs = [None] * out_d
    for od in range(d):
        for i in range(kd):
            plane = input[_slice_at(2, od, od + 1)].reshape(n, c, h, w)
            slab = weight[_slice_at(2, i, i + 1)].reshape(c2, f, kh, kw)
            part = conv_transpose2d(plane, slab, None, (sh, sw), (ph, pw),
                                    (oph, opw), 1, (dh, dw))
            at = od * sd + i * dd
            slabs[at] = part if slabs[at] is None else slabs[at] + part
    shape = next(s for s in slabs if s is not None).data.shape
    empty = None
    for at, slab in enumerate(slabs):
        if slab is None:
            if empty is None:
                empty = _wrap(_np.zeros(shape, dtype=_DEFAULT_DTYPE))
            slabs[at] = empty
    out = cat([s.reshape(shape[0], shape[1], 1, shape[2], shape[3]) for s in slabs], 2)
    # The depth axis is trimmed and then extended by `output_padding`, by the same
    # rule the height and the width follow one function up: what the extension
    # reaches is computed values where they exist, and nothing only past the end.
    if pd or opd:
        bottom = out_d - pd + opd
        kept = out[_slice_at(2, pd, min(bottom, out_d))]
        if bottom > out_d:
            tail = _wrap(_np.zeros(
                (n, f, bottom - out_d) + kept.data.shape[3:], dtype=_DEFAULT_DTYPE))
            kept = cat([kept, tail], 2)
        out = kept
    if bias is not None:
        out = out + bias.reshape(1, -1, 1, 1, 1)
    return out


def _norm_flat(x, groups, eps, center=True):
    """The body the three normalisations share. **The grouped extent arrives
    flattened onto one axis.**

    `mean(dim=…)` takes one axis, so the caller flattens to
    `(group count, elements within it)` and only the last axis is folded here.
    Making it take a list of axes widens the reduction surface, and nothing but
    the normalisations would use that surface.

    With `center=False` the mean is not subtracted — the only difference between
    `RMSNorm` and `LayerNorm`. Written separately there are two copies, and two
    copies means the day comes when one is fixed.
    """
    centered = x - x.mean(dim=-1, keepdim=True) if center else x
    var = (centered * centered).mean(dim=-1, keepdim=True)
    return centered / (var + eps).sqrt()


def _channel_shape(x, size):
    """Spread to line up with the channel axis (1). It has to be `(1, C, 1, …)`
    for the broadcasting to fit."""
    shape = [1] * len(x.data.shape)
    shape[1] = size
    return tuple(shape)


def group_norm(input, num_groups, weight=None, bias=None, eps=1e-5):
    """Normalise with the channels bundled into groups. **The group count sets
    the boundaries.**

    At `num_groups=1` all the channels are one bundle, which is `LayerNorm`, and
    at the channel count each channel stands alone, which is `InstanceNorm`. The
    three are special cases of one another, and a wrong bundling rule makes two
    of the three identical — which is why the golden asks about all three side by
    side.
    """
    input = _wrap(input)
    shape = input.data.shape
    n, c = shape[0], shape[1]
    if c % num_groups:
        raise RuntimeError(f"cannot split {c} channels into {num_groups} groups")
    inner = (c // num_groups) * int(_np.prod(shape[2:], dtype=int))
    out = _norm_flat(input.reshape(n, num_groups, inner), num_groups, eps).reshape(*shape)
    if weight is not None:
        out = out * _wrap(weight).reshape(*_channel_shape(input, c))
    if bias is not None:
        out = out + _wrap(bias).reshape(*_channel_shape(input, c))
    return out


def instance_norm(input, running_mean=None, running_var=None, weight=None,   # noqa: A002
                  bias=None, use_input_stats=True, momentum=0.1, eps=1e-5):
    """Per sample and per channel. `group_norm` with the group count set to the
    channel count.

    **torch's first three seats were missing**, so `F.instance_norm(x, None, None, w)`
    put the weight in `running_mean`'s place here and in `weight`'s there — the same
    call, a different layer, no exception.

    **The running statistics were refused on the ground that this keeps none.** It
    keeps none of its own, which is true and was not the question: torch's buffers
    are the *caller's*, handed in and written back, exactly as `batch_norm` next door
    already does. What the update is was measured rather than derived —

        running_mean ← (1−m)·running_mean + m·mean over (N, H, W) per channel
        running_var  ← (1−m)·running_var  + m·mean over N of the **unbiased**
                       per-(sample, channel) variance

    — and the second line is the one worth writing down: it is the average of the
    per-plane variances, not the variance of the whole channel. On a 2×3×2×2 of
    consecutive integers those are 1.667 and 47.7, so nothing subtle separates them.

    **Handing statistics in does not change the output** under `use_input_stats=True`;
    the normalisation stays per plane and only the buffers move. Measured, because the
    opposite is the natural guess.

    `use_input_stats=False` normalises by the stored statistics instead, per channel,
    and leaves them untouched. With none given torch stops, and the wording is torch's.
    """
    input = _wrap(input)  # noqa: A001
    if not use_input_stats and (running_mean is None or running_var is None):
        raise RuntimeError("Expected running_mean and running_var to be defined when "
                           "use_input_stats is false")
    rank = input.data.ndim
    shape = (1, -1) + (1,) * (rank - 2)
    planes = tuple(range(2, rank))

    if not use_input_stats:
        rm = _np.asarray(_instance_raw(running_mean)).reshape(shape)
        rv = _np.sqrt(_np.asarray(_instance_raw(running_var)) + eps).reshape(shape)
        normed = (input - Tensor(rm)) / Tensor(rv)
        if weight is not None:
            normed = normed * _wrap(weight).reshape(shape)
        if bias is not None:
            normed = normed + _wrap(bias).reshape(shape)
        return normed

    if running_mean is not None:
        with no_grad():
            seen = input.data.mean(axis=(0, *planes))
            spread = input.data.var(axis=planes, ddof=1).mean(axis=0)
            _instance_raw(running_mean)[...] = (
                (1 - momentum) * _instance_raw(running_mean) + momentum * seen)
            _instance_raw(running_var)[...] = (
                (1 - momentum) * _instance_raw(running_var) + momentum * spread)
    return group_norm(input, input.data.shape[1], weight, bias, eps)


def _instance_raw(v):
    """The array behind a buffer, whether it arrived wrapped or bare. `batch_norm`
    keeps its own copy of this; the two are the same three lines."""
    return v.data if isinstance(v, Tensor) else v


def rms_norm(input, normalized_shape, weight=None, eps=None):
    """**It does not subtract the mean.** That is the only difference from
    `LayerNorm`.

    **The default eps is not `1e-5`.** Given nothing, torch uses that dtype's
    machine epsilon (1.19e-07 for float32), and with every other normalisation
    layer at `1e-5` it was carelessly written to match. The forward pass came
    within tolerance and passed, and it diverged **in the gradient alone, by a
    max diff of 2.26e-02** — because it is amplified where the variance is
    small.
    """
    input = _wrap(input)
    if eps is None:
        eps = float(_np.finfo(_np.float32).eps)
    shape = input.data.shape
    k = len(normalized_shape) if isinstance(normalized_shape, (list, tuple)) else 1
    lead = int(_np.prod(shape[:len(shape) - k], dtype=int))
    inner = int(_np.prod(shape[len(shape) - k:], dtype=int))
    out = _norm_flat(input.reshape(lead, inner), lead, eps, center=False).reshape(*shape)
    return out if weight is None else out * _wrap(weight)


def conv1d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    """A 1-D convolution. **Built by inserting a height of 1 into `conv2d`.**

    The sister library (webgpu) already works this way. A new im2col would make
    two copies of the same computation, and then the day comes when one is fixed
    and they diverge.
    """
    input, weight = _wrap(input), _wrap(weight)
    n, c, length = input.data.shape
    f, c2, k = weight.data.shape
    lifted = input.reshape(n, c, 1, length)
    kernel = weight.reshape(f, c2, 1, k)
    # The height axis is left alone — stride 1, padding 0, dilation 1.
    out = conv2d(lifted, kernel, bias, (1, stride), (0, padding), (1, dilation),
                 groups)
    shape = out.data.shape
    return out.reshape(shape[0], shape[1], shape[3])


def conv3d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    """A 3-D convolution. **A 2-D convolution per depth, summed.**

    im2col is not rewritten in 3-D — built from multiplications and additions the
    backward follows on its own, and there is no new derivative to write. Slow
    and not wrong.
    """
    if groups != 1:
        return _grouped(
            lambda xs, ws, bs: conv3d(xs, ws, bs, stride, padding, dilation),
            input, weight, bias, groups)
    input, weight = _wrap(input), _wrap(weight)
    n, c, d, h, w = input.data.shape
    f, c2, kd, kh, kw = weight.data.shape
    if c != c2:
        raise RuntimeError(f"channels do not match: input {c}, filter {c2}")
    sd, sh, sw = (stride, stride, stride) if isinstance(stride, int) else tuple(stride)
    pd, ph, pw = (padding, padding, padding) if isinstance(padding, int) else tuple(padding)
    dd, dh, dw = (dilation,) * 3 if isinstance(dilation, int) else tuple(dilation)
    if pd:
        pads = [(0, 0)] * 5
        pads[2] = (pd, pd)
        input = input._make(_np.pad(input.data, pads), (input,),
                    lambda g: (_np.asarray(g)[:, :, pd:-pd],), "Pad3dBackward0")
        d = input.data.shape[2]

    out_d = (d - ((kd - 1) * dd + 1)) // sd + 1
    slabs = []
    for od in range(out_d):
        acc = None
        for i in range(kd):
            depth = od * sd + i * dd
            plane = input[_slice_at(2, depth, depth + 1)].reshape(n, c, h, w)
            slab = weight[_slice_at(2, i, i + 1)].reshape(f, c2, kh, kw)
            part = conv2d(plane, slab, None, (sh, sw), (ph, pw), (dh, dw))
            acc = part if acc is None else acc + part
        shape = acc.data.shape
        slabs.append(acc.reshape(shape[0], shape[1], 1, shape[2], shape[3]))
    out = cat(slabs, 2)
    if bias is not None:
        bt = _wrap(bias)
        out = out + bt.reshape(1, -1, 1, 1, 1)
    return out


def _pool_geometry(shape, spatial, kernel_size, stride, padding, dilation, ceil_mode):
    """The per-axis window lists, and the pad widths that go with them.

    **torch refuses padding larger than half the window** and so does this — with
    a wider pad a window can be entirely padding, and its maximum would be the
    `-inf` the padding is made of. A silent `-inf` in a feature map is the kind of
    thing that surfaces an hour later as a NaN loss.
    """
    ks = _spread(kernel_size, spatial)
    st = _spread(stride if stride is not None else kernel_size, spatial)
    pd = _spread(padding, spatial)
    dl = _spread(dilation, spatial)
    for k in range(spatial):
        if pd[k] * 2 > ks[k]:
            raise RuntimeError(
                "pad should be at most half of effective kernel size, but got "
                f"padding={pd[k]} and kernel_size={ks[k]}")
    axes = [_pool_windows(shape[2 + k], ks[k], st[k], pd[k], dl[k], ceil_mode)
            for k in range(spatial)]
    return axes, pd


def _padded_for_max(x, pads):
    """Pad the spatial axes with `-inf`, so a padded cell never wins a window."""
    if not any(pads):
        return x
    widths = []
    for width in reversed(pads):
        widths += [width, width]
    return pad(x, widths, value=float("-inf"))


def _unpadded_positions(shape, pads):
    """The flat index **within the unpadded plane**, laid out on the padded one.

    The padded cells hold whatever falls there; they carry `-inf` as a value and
    so never win, and `_pool_geometry` refuses the padding width that would let a
    window be all padding.
    """
    spatial = shape[2:]
    plane = int(_np.prod(spatial)) if spatial else 1
    base = _np.arange(plane).reshape(spatial)
    for axis, width in enumerate(pads):
        if width:
            base = _np.pad(base, [(width, width) if a == axis else (0, 0)
                                  for a in range(len(spatial))])
    padded = shape[:2] + base.shape
    return _np.broadcast_to(base, padded)


def _max_pool_nd(x, spatial, kernel_size, stride, padding, dilation, ceil_mode,
                 return_indices):
    """The general pool. **One window list feeds both the value and the index.**

    The fast paths below keep the default case quick; everything the new
    arguments touch comes through here, so `padding`, `dilation` and `ceil_mode`
    are decided once. Splitting them would put the same arithmetic in two places,
    which is the rule `_fixed_windows` is already written under — and a value and
    an index looking at different windows agree on the value.
    """
    x = _wrap(x)
    axes, pads = _pool_geometry(x.data.shape, spatial, kernel_size, stride,
                                padding, dilation, ceil_mode)
    padded = _padded_for_max(x, pads)
    if return_indices:
        positions = _unpadded_positions(x.data.shape, pads)
        out, pos = _max_with_index(padded, axes, positions)
        return out, Tensor(pos)
    out = padded
    for k in range(spatial):
        out = _fold_axis(out, 2 + k, axes[k], "max")
    return out


def _pool_is_plain(padding, dilation, ceil_mode):
    """Whether the fast path answers this call."""
    return not any(_spread(padding, 3)) and all(
        d == 1 for d in _spread(dilation, 3)) and not ceil_mode


def max_pool1d(x, kernel_size, stride=None, padding=0, dilation=1,
               return_indices=False, ceil_mode=False):
    """Insert a height of 1 into `max_pool2d`. The height and its window are
    both 1, so that axis does not move."""
    if not _pool_is_plain(padding, dilation, ceil_mode):
        return _max_pool_nd(x, 1, kernel_size, stride, padding, dilation,
                            ceil_mode, return_indices)
    if return_indices:
        return max_pool1d_with_indices(x, kernel_size, stride)
    x = _wrap(x)
    n, c, length = x.data.shape
    stride = stride or kernel_size
    out_len = (length - kernel_size) // stride + 1
    lifted = x.reshape(n, c, 1, length)
    pooled = _pool_1d_over_last(lifted, kernel_size, stride)
    return pooled.reshape(n, c, out_len)


def _pool_1d_over_last(x, kernel_size, stride):
    """Only the last axis is reduced by the window — the height axis stays put
    at window 1, stride 1."""
    parts = []
    length = x.data.shape[3]
    for start in range(0, length - kernel_size + 1, stride):
        window = [x[_slice_at(3, start + i, start + i + 1)] for i in range(kernel_size)]
        acc = window[0]
        for piece in window[1:]:
            acc = _maximum_first(acc, piece)
        parts.append(acc)
    return cat(parts, 3)


def max_pool3d(x, kernel_size, stride=None, padding=0, dilation=1,
               return_indices=False, ceil_mode=False):
    """The depth direction is sliced and the maxima taken across the slices, and
    `max_pool2d` does the rest."""
    if not _pool_is_plain(padding, dilation, ceil_mode):
        return _max_pool_nd(x, 3, kernel_size, stride, padding, dilation,
                            ceil_mode, return_indices)
    if return_indices:
        return max_pool3d_with_indices(x, kernel_size, stride)
    x = _wrap(x)
    stride = stride or kernel_size
    n, c, d, h, w = x.data.shape
    out_d = (d - kernel_size) // stride + 1
    slabs = []
    for od in range(out_d):
        acc = None
        for i in range(kernel_size):
            plane = x[_slice_at(2, od * stride + i, od * stride + i + 1)].reshape(n, c, h, w)
            pooled = max_pool2d(plane, kernel_size, stride)
            acc = pooled if acc is None else _maximum_first(acc, pooled)
        shape = acc.data.shape
        slabs.append(acc.reshape(shape[0], shape[1], 1, shape[2], shape[3]))
    return cat(slabs, 2)


# ── the versions that also give the winning positions ───────────────────────
#
# torch offers one computation under two names:
# `max_pool2d(..., return_indices=True)` and `max_pool2d_with_indices(...)`. One
# of them produces the value and the other lays a name on top.

# **`return_indices` is named rather than swallowed.**
#
# These nine took `**_` at the end, which made their whole signature read as
# `VARIADIC` to `inspect` — and `variadic` on the signature axis means *cannot be
# compared at all*, so nothing had ever checked their argument lists against torch's.
# One `**_` bought silence over seven names.
#
# What it was swallowing is `return_indices`, and it changes nothing: measured, a
# `*_with_indices` function in torch hands back the pair whichever way the flag is
# set — the name has already decided. So it is accepted and unused, like `foreach` on
# the optimizers, and **the difference between that and `**_` is that a reader can
# see it.**
def max_pool1d_with_indices(input, kernel_size, stride=None, padding=0, dilation=1,
                            ceil_mode=False, return_indices=True):
    """**This name exists at torch's top level too**, and takes everything
    positionally. The three slots that used to be refused are implemented now and
    go through `_max_pool_nd`, which is where the window arithmetic lives."""
    if not _pool_is_plain(padding, dilation, ceil_mode):
        return _max_pool_nd(input, 1, kernel_size, stride, padding, dilation,
                            ceil_mode, True)
    input = _wrap(input)
    windows = _fixed_windows(input.data.shape[2], kernel_size, stride or kernel_size)
    out, pos = _max_with_index(input, [windows])
    return out, Tensor(pos)


def max_pool2d_with_indices(input, kernel_size, stride=None, padding=0, dilation=1,
                            ceil_mode=False, return_indices=True):
    if not _pool_is_plain(padding, dilation, ceil_mode):
        return _max_pool_nd(input, 2, kernel_size, stride, padding, dilation,
                            ceil_mode, True)
    input = _wrap(input)
    out, pos = _max_with_index(
        input, _fixed_window_axes(input.data.shape, kernel_size, stride))
    return out, Tensor(pos)


def max_pool3d_with_indices(input, kernel_size, stride=None, padding=0, dilation=1,
                            ceil_mode=False, return_indices=True):
    if not _pool_is_plain(padding, dilation, ceil_mode):
        return _max_pool_nd(input, 3, kernel_size, stride, padding, dilation,
                            ceil_mode, True)
    input = _wrap(input)
    out, pos = _max_with_index(
        input, _fixed_window_axes(input.data.shape, kernel_size, stride))
    return out, Tensor(pos)


def adaptive_max_pool1d_with_indices(input, output_size, return_indices=True):
    return _adaptive_with_indices(input, _spread(output_size, 1))


def adaptive_max_pool2d_with_indices(input, output_size, return_indices=True):
    return _adaptive_with_indices(input, _pair(output_size))


def adaptive_max_pool3d_with_indices(input, output_size, return_indices=True):
    return _adaptive_with_indices(input, _spread(output_size, 3))


def _adaptive_with_indices(x, sizes):
    x = _wrap(x)
    shape = x.data.shape
    axes = [_adaptive_windows(shape[2 + k], sizes[k]) for k in range(len(sizes))]
    out, pos = _max_with_index(x, axes)
    return out, Tensor(pos)


def _unpool(x, indices, kernel_size, stride, padding, output_size, spatial):
    """**Put the values back** at the positions `max_pool` chose. The rest are 0.

    The positions are the indices the pooling produced, flat indices within the
    plane. So this function does not recompute the positions — recomputing leaves
    room for that computation to diverge from the pooling's, and a picture whose
    non-zero positions have shifted slightly is invisible to the eye.

    The default output size is `(n-1)·stride - 2·padding + kernel`. What the
    pooling threw away cannot be revived, so torch also opens a path to give
    `output_size` directly.
    """
    x = _wrap(x)
    shape = x.data.shape
    ks = _spread(kernel_size, spatial)
    st = _spread(stride if stride is not None else kernel_size, spatial)
    pd = _spread(padding, spatial)
    if output_size is None:
        out_spatial = tuple((shape[2 + k] - 1) * st[k] - 2 * pd[k] + ks[k]
                            for k in range(spatial))
    else:
        got = tuple(output_size)
        out_spatial = got[-spatial:]        # torch takes the whole shape too
    plane = int(_np.prod(out_spatial))

    pos = _np.asarray(indices.data if isinstance(indices, Tensor) else indices)
    base = (_np.arange(shape[0] * shape[1]) * plane).reshape(shape[0], shape[1],
                                                             *([1] * spatial))
    flat = (base + pos).reshape(-1)
    out_shape = (shape[0], shape[1]) + tuple(out_spatial)

    filled = _np.zeros(shape[0] * shape[1] * plane, dtype=x.data.dtype)
    filled[flat] = x.data.reshape(-1)

    def back(g):
        # Collected straight from where the values went — the inverse of the
        # scatter.
        return (_np.asarray(g).reshape(-1)[flat].reshape(shape),)

    return x._make(filled.reshape(out_shape), (x,), back, "MaxUnpoolBackward0")


# ── CTC ────────────────────────────────────────────────────────────────────
#
# A loss that connects audio to characters **without aligning them.** In speech
# recognition it removes a person having to write down "which character these
# five frames correspond to" — it sums over every possible alignment to get the
# probability.
#
# The number of alignments to sum is exponential, so it is folded with the
# forward algorithm. An **extended target with blanks inserted** is built between
# the targets (`[_ , l1, _, l2, …, _]`), and one state is held per time step,
# reachable from the three before it only.

_CTC_NEG = -1e30       # "absent" as a log probability. `-inf` becomes NaN in logsumexp


def _ctc_extended(labels, blank):
    """`[l1, l2]` → `[_, l1, _, l2, _]`. Blanks are inserted between them.

    **Repeated characters must have a blank between them** — without one, the two
    characters fold into one. That rule is `_ctc_skip` below.
    """
    ext = [blank]
    for lab in labels:
        ext.append(int(lab))
        ext.append(blank)
    return ext


def _ctc_skip(ext, blank):
    """Can this position be reached by skipping from `s-2`. It must not be a
    blank and must differ from the one two back."""
    return [0.0 if (u >= 2 and ext[u] != blank and ext[u] != ext[u - 2])
            else _CTC_NEG for u in range(len(ext))]


def _ctc_needs(labels):
    """**How many time steps at minimum** are needed to produce this target.

    The character count plus one blank for every adjacent repeated pair. Shorter
    than this there is no alignment at all, so the probability is 0 and the loss
    is `inf` — torch produces `inf` at that point. This condition is checked
    directly rather than approximated with a threshold.
    """
    return len(labels) + sum(1 for a, b in zip(labels, labels[1:]) if a == b)


def _ctc_targets(targets, target_lengths):
    """It may arrive as `(N, S)` or as a concatenated 1-D row. torch takes
    both."""
    data = _np.asarray(targets.data if isinstance(targets, Tensor) else targets)
    lengths = [int(n) for n in _np.asarray(
        target_lengths.data if isinstance(target_lengths, Tensor) else target_lengths
    ).reshape(-1)]
    if data.ndim == 1:
        out, at = [], 0
        for n in lengths:
            out.append([int(v) for v in data[at:at + n]])
            at += n
        return out
    return [[int(v) for v in data[i, :n]] for i, n in enumerate(lengths)]


def _ctc_one(lp, labels, n_time, blank):
    """One sample's `-log P(target | audio)`.

    **Folded with our own operations** — no gradient is written by hand. CTC's
    backward is the well-known formula that runs the backward sweep once more,
    and writing it separately adds one more place to diverge from the forward.

    The `u` axis is shifted all at once. Only time loops in Python, so the graph
    is proportional to `T` — slow at real speech-recognition lengths (hundreds of
    frames). The accurate side was chosen.
    """
    ext = _ctc_extended(labels, blank)
    u = len(ext)
    idx = Tensor(_np.array(ext, dtype=_np.int64))
    skip = Tensor(_np.array(_ctc_skip(ext, blank), dtype=_np.float32))
    gap = lambda n: Tensor(_np.full(n, _CTC_NEG, dtype=_np.float32))   # noqa: E731

    emit = index_select(lp, 1, idx)                 # (T, U) — the emission log probability per position
    head = min(2, u)
    alpha = emit[0][:head]
    if u > head:
        alpha = cat([alpha, gap(u - head)], 0)

    for t in range(1, n_time):
        same = alpha
        one = cat([gap(1), alpha[:u - 1]], 0) if u > 1 else gap(1)
        two = (cat([gap(2), alpha[:u - 2]], 0) if u > 2 else gap(u)) + skip
        alpha = logsumexp(stack([same, one, two], 0), dim=0) + emit[t]

    tail = alpha[u - 2:] if u >= 2 else alpha
    return -logsumexp(tail, dim=0)


def _ctc_torch_bias(lp, i, n_time):
    """**A term whose value is 0 and whose gradient alone is `exp(log_probs)`.**
    A place that follows torch.

    The gradient torch's `ctc_loss` flows into `log_probs` is not the true
    derivative. It was measured (`tests/probe_ctc3.py`): finite differences give
    `-γ` and torch gives `exp(log_probs) - γ`. The difference is exactly
    `exp(log_probs)`, and it attaches only within `t < input_length`.

    **And at the point of use the two give the same answer.** There is always a
    `log_softmax` in front of CTC, and its backward is `g - softmax·Σg`. With
    `g = -γ`, `Σg = -1`, giving `softmax - γ`; with `g = softmax - γ`, `Σg = 0`,
    giving `softmax - γ` again. The form torch chose is that transformation's
    fixed point.

    It is matched here anyway. This repository's claim is "change the import and
    run", and code that puts `log_probs` straight at a leaf without a
    `log_softmax` in front is where the numbers diverge. It is matched, and **why
    it is matched** is written down — this term is not the loss's derivative.
    """
    window = lp[:n_time, i, :]
    bias = exp(window).sum()
    return bias - bias.detach()


def ctc_loss(log_probs, targets, input_lengths, target_lengths, blank=0,
             reduction="mean", zero_infinity=False):
    """`log_probs` is `(T, N, C)` — **time first.** That is what torch does.

    `reduction="mean"` is not the ordinary one: each sample is **divided by its
    own target length** before the mean. It is not a plain mean, so a case whose
    target lengths are all equal does not show the difference.
    """
    lp = _wrap(log_probs)
    labels = _ctc_targets(targets, target_lengths)
    times = [int(n) for n in _np.asarray(
        input_lengths.data if isinstance(input_lengths, Tensor) else input_lengths
    ).reshape(-1)]

    losses = []
    for i, labs in enumerate(labels):
        if times[i] < _ctc_needs(labs):
            # There is no alignment at all — probability 0, loss `inf`.
            # `zero_infinity` turns that into 0 (and no gradient flows).
            losses.append(_wrap(0.0 if zero_infinity else _np.float32("inf")))
            continue
        one = _ctc_one(lp[:, i, :], labs, times[i], blank)
        losses.append(one + _ctc_torch_bias(lp, i, times[i]))

    per = stack(losses, 0)
    if reduction == "none":
        return per
    if reduction == "sum":
        return per.sum()
    lens = Tensor(_np.array([max(len(l), 1) for l in labels], dtype=_np.float32))
    return (per / lens).mean()


def _fractional_intervals(n_in, k, n_out, u):
    """The window start positions. **ATen's `generate_intervals` exactly**
    (measured).

    It takes `α = (input - window) / (output - 1)` and uses
    `floor((i+u)·α) - floor(u·α)`. Only the last window is pinned to the right
    edge — so that the input's last cell is certain to be covered.

    `u` is the value that jitters the window positions, and that is what makes
    this pooling "fractional". When it divides evenly, `α` is an integer and any
    `u` gives the same answer — **asked as 6→3 the random part is invisible
    entirely.** The golden asks 7→3.
    """
    if n_out <= 1:
        return [0]
    alpha = (n_in - k) / (n_out - 1)
    seq = [int((i + u) * alpha) - int(u * alpha) for i in range(n_out - 1)]
    return seq + [n_in - k]


def _fractional_pool(x, kernel_size, output_size, output_ratio, samples, spatial):
    """Fractional max pooling. The window positions can differ per plane, so
    each plane is folded separately.

    **Expensive** — it loops once per plane, because torch's samples are
    `(N, C, axis)` and the windows can differ per plane. Not a frequently used
    layer, so it is left as it is. Bundling the samples into one to save the cost
    makes it a different layer from torch at that moment.
    """
    x = _wrap(x)
    shape = x.data.shape
    if (output_size is None) == (output_ratio is None):
        raise ValueError(
            "fractional_max_pool takes either output_size or output_ratio, not both.")
    ks = _spread(kernel_size, spatial)
    if output_size is not None:
        sizes = _spread(output_size, spatial)
    else:
        ratios = _spread(output_ratio, spatial) if not isinstance(output_ratio, float) \
            else [output_ratio] * spatial
        sizes = [int(shape[2 + k] * ratios[k]) for k in range(spatial)]

    n, c = shape[0], shape[1]
    if samples is None:
        samples = _rng.random((n, c, spatial))
    else:
        samples = _np.asarray(samples.data if isinstance(samples, Tensor) else samples)

    values, positions = [], []
    for i in range(n):
        for j in range(c):
            plane = x[i:i + 1, j:j + 1]
            # **The 2-D version reads the samples reversed.** ATen's 2-D version
            # reads `[0]` as the width and `[1]` as the height, and the 3-D
            # version reads `[0]`, `[1]` and `[2]` as depth, height and width —
            # the two functions disagree with each other. What is imitated here
            # is that disagreement itself. It shows only when the samples differ
            # per axis, and asked at a size that divides evenly (α being an
            # integer) any sample gives the same answer, so again it does not
            # show.
            order = list(range(spatial)) if spatial == 3 else list(reversed(range(spatial)))
            axes = [_fractional_intervals(shape[2 + k], ks[k], sizes[k],
                                          float(samples[i, j, order[k]]))
                    for k in range(spatial)]
            windows = [[(s, s + ks[k]) for s in axes[k]] for k in range(spatial)]
            got, pos = _max_with_index(plane, windows)
            values.append(got)
            positions.append(pos)
    out = cat(values, 0).reshape(n, c, *sizes)
    idx = _np.concatenate(positions, 0).reshape(n, c, *sizes)
    return out, Tensor(idx)


def fractional_max_pool2d(input, kernel_size, output_size=None, output_ratio=None,
                          return_indices=False, _random_samples=None):
    out, idx = _fractional_pool(input, kernel_size, output_size, output_ratio,
                                _random_samples, 2)
    return (out, idx) if return_indices else out


def fractional_max_pool3d(input, kernel_size, output_size=None, output_ratio=None,
                          return_indices=False, _random_samples=None):
    out, idx = _fractional_pool(input, kernel_size, output_size, output_ratio,
                                _random_samples, 3)
    return (out, idx) if return_indices else out


def fractional_max_pool2d_with_indices(input, kernel_size, output_size=None,
                                       output_ratio=None, return_indices=True,
                                       _random_samples=None):
    return _fractional_pool(input, kernel_size, output_size, output_ratio,
                            _random_samples, 2)


def fractional_max_pool3d_with_indices(input, kernel_size, output_size=None,
                                       output_ratio=None, return_indices=True,
                                       _random_samples=None):
    return _fractional_pool(input, kernel_size, output_size, output_ratio,
                            _random_samples, 3)


def max_unpool1d(input, indices, kernel_size, stride=None, padding=0, output_size=None):
    return _unpool(input, indices, kernel_size, stride, padding, output_size, 1)


def max_unpool2d(input, indices, kernel_size, stride=None, padding=0, output_size=None):
    return _unpool(input, indices, kernel_size, stride, padding, output_size, 2)


def max_unpool3d(input, indices, kernel_size, stride=None, padding=0, output_size=None):
    return _unpool(input, indices, kernel_size, stride, padding, output_size, 3)


def _out_size(in_size, scale_factor):
    """torch's output length: **floor**, per axis, from a scale that may be a float."""
    scales = (scale_factor if isinstance(scale_factor, (tuple, list))
              else (scale_factor,) * len(in_size))
    return tuple(int(_math.floor(n * float(sc))) for n, sc in zip(in_size, scales))


def interpolate(input, size=None, scale_factor=2, mode="nearest",   # noqa: A002
                align_corners=None, recompute_scale_factor=None, antialias=False):
    """Enlargement. Nearest and bilinear.

    **`antialias` is torch's last seat and is refused rather than left out.** It
    changes the values when *shrinking* with `bilinear` or `bicubic`: torch widens
    the filter so every input cell reaches the output, where plain sampling skips
    cells and aliases. Enlarging, it does nothing at all.

    Left out, the seat did not exist and a positional call reaching it landed on
    nothing; accepted and ignored, `interpolate(x, size=8, mode='bilinear',
    antialias=True)` on a shrink would return the aliased answer under the name of
    the filtered one — the same numbers a caller asked to stop getting.

    **`align_corners` changes the values.** True pins both ends and divides
    evenly between them (`src = i·(in−1)/(out−1)`); false measures from **the
    centre** of each cell. `UpsamplingBilinear2d` is true and
    `Upsample(mode='bilinear')` defaults to false, so making one an alias of the
    other by name alone puts the edges off — the interior is similar enough that
    the eye does not part them.
    """
    x = _wrap(input)
    if antialias:
        # torch's own restriction, and its wording.
        if mode not in ("bilinear", "bicubic"):
            raise RuntimeError(
                "Anti-alias option is restricted to bilinear, bicubic, and lanczos "
                "modes and requires a 4-D tensor as input")
        return _interpolate_antialias(x, size, scale_factor, mode,
                                      bool(align_corners), recompute_scale_factor)
    if mode == "bilinear":
        return _interpolate_bilinear(x, size, scale_factor, bool(align_corners),
                                     recompute_scale_factor)
    if mode == "bicubic":
        return _interpolate_bicubic(x, size, scale_factor, bool(align_corners),
                                    recompute_scale_factor)
    if mode not in ("nearest", "nearest-exact", "area"):
        _unsupported(f"interpolate(mode={mode!r}) — nearest, nearest-exact, area, "
                     "bilinear and bicubic are here; linear and trilinear want a "
                     "rank this function does not take")
    xd = x.data
    n, c, h, w = xd.shape
    if size is not None:
        oh, ow = _pair(size)
    else:
        oh, ow = _out_size((h, w), scale_factor)
    # **`area` is `adaptive_avg_pool2d` and nothing else.** Measured against torch on
    # three output sizes — a shrink, an enlargement and one that divides evenly into
    # neither — `F.interpolate(x, size, mode="area")` and `F.adaptive_avg_pool2d(x,
    # size)` agree bit for bit. So this names what is already here rather than writing
    # a second averaging, and the gradient comes with it.
    if mode == "area":
        return adaptive_avg_pool2d(x, (oh, ow))
    # **A source index per output cell, not a whole-number repeat.**
    #
    # This was `np.repeat` twice and it refused anything that was not an integer
    # multiple — `interpolate(x, scale_factor=1.5)` stopped with `'float' object is
    # not iterable`, from `_pair`, which names neither the scale nor the mode. torch
    # takes any positive scale and takes `size=` that is not a multiple, and nearest
    # neighbour has no difficulty with either: the source is `floor(i / scale)`.
    #
    # **`recompute_scale_factor` decides which scale that is.** True rebuilds it as
    # `out / in` after the output size has been floored; False uses the number the
    # caller gave. They differ exactly when the flooring loses something — `5 → 1.7`
    # gives 8, and `8 / 5 = 1.6` is not `1.7`, so the two pick different source cells
    # in the middle of the row.
    #
    # **`None` is torch's default and it behaves as `False`, not as `True`.** That is
    # the opposite of what the name suggests — "recompute" reads like the thing a
    # default would do — and it was written the other way here first. Measured: with
    # `None`, torch agrees with `False` at every scale tried and with `True` at none
    # of them.
    use_given = recompute_scale_factor is not True and size is None
    given = (scale_factor if isinstance(scale_factor, (tuple, list))
             else (scale_factor, scale_factor))
    sh = given[0] if use_given else (oh / h)
    sw = given[1] if use_given else (ow / w)
    # **`nearest-exact` measures from the centre of the output cell and `nearest` does
    # not.** `floor((i + 0.5) / s)` against `floor(i / s)` — half a cell, which is
    # nothing when enlarging by a whole number and is a different row entirely when
    # shrinking. torch keeps both because the plain one is the one everybody else
    # already shipped and is off by half; measured on a 4×5 to 2×3, `nearest` takes
    # rows 0 and 2 and `nearest-exact` takes 1 and 3.
    half = 0.5 if mode == "nearest-exact" else 0.0
    rows = _np.minimum(((_np.arange(oh) + half) / sh).astype(int), h - 1)
    cols = _np.minimum(((_np.arange(ow) + half) / sw).astype(int), w - 1)
    out = xd[:, :, rows][:, :, :, cols]

    def back(g):
        g = _np.asarray(g)
        gx = _np.zeros_like(xd)
        _np.add.at(gx, (slice(None), slice(None), rows[:, None], cols[None, :]), g)
        return (gx,)

    return x._make(out, (x,), back, "UpsampleBackward0")


def _bilinear_axis(size_in, size_out, align_corners, scale=None):
    """For each output position, **which two input positions and in what
    proportion.**

    `scale` is the one the caller asked for, used only where `align_corners` is
    false and `recompute_scale_factor` is false — see `interpolate`. Left as `None`
    the scale is `size_in / size_out`, which is what torch uses by default.
    """
    if align_corners:
        # Both ends are pinned and the space between is divided evenly.
        src = (_np.arange(size_out, dtype=_np.float64)
               * ((size_in - 1) / max(1, size_out - 1)))
    else:
        # Measured from the centre of each cell. Positions that fall outside are
        # clamped to the edge.
        step = (size_in / size_out) if scale is None else (1.0 / float(scale))
        src = (_np.arange(size_out, dtype=_np.float64) + 0.5) * step - 0.5
        src = _np.clip(src, 0, None)
    lo = _np.floor(src).astype(_np.intp)
    hi = _np.minimum(lo + 1, size_in - 1)
    return lo, hi, (src - lo)


# torch's cubic convolution constant. **`a` is a choice, not a derivation** — the
# family of cubic kernels is parameterised by it, OpenCV uses −0.75 and Photoshop
# −0.5, and they give visibly different edges. torch uses −0.75, and reproducing
# torch's numbers to float64 noise is what fixed it here rather than any argument
# about which is better.
_BICUBIC_A = -0.75


def _cubic_weight(t):
    """Keys' cubic convolution kernel at `t`, zero beyond two cells."""
    t = _np.abs(t)
    a = _BICUBIC_A
    near = ((a + 2) * t - (a + 3)) * t * t + 1
    far = ((t - 5) * t + 8) * t * a - 4 * a
    return _np.where(t <= 1, near, _np.where(t < 2, far, 0.0))


def _bicubic_axis(size_in, size_out, align_corners, scale=None):
    """The continuous source coordinate for each output position.

    The same two rules `_bilinear_axis` uses, which is the point: `align_corners`
    pins both ends, and otherwise the coordinate is measured from the **centre** of
    the output cell — `(i + 0.5)·s − 0.5`.
    """
    if align_corners:
        if size_out == 1:
            return _np.zeros(1)
        return _np.arange(size_out) * ((size_in - 1) / (size_out - 1))
    step = (1.0 / scale) if scale else (size_in / size_out)
    return (_np.arange(size_out) + 0.5) * step - 0.5


def _antialias_axis(size_in, size_out, radius, filt, align_corners, given=None):
    """One axis of torch's anti-aliased resampling, as a `(size_out, size_in)` matrix.

    The window is **widened by the shrink factor and the weights renormalised** —
    that is the whole of what `antialias` means. Enlarging, the scale is below one,
    the support stays at the kernel's own radius and the weights are the plain ones,
    which is why torch says the flag does nothing when going up.

    **Two things in here are torch disagreeing with itself, and both are measured
    rather than reasoned.**

    *The cubic constant is not the same one.* Plain `bicubic` uses `a = −0.75` and
    this path uses `a = −0.5`. Fitted against torch: at `−0.75` the two part by 0.13
    to 0.39 on a 4×5, and at `−0.5` they agree to float64 noise on every size tried.

    *`align_corners` is half applied.* The scale becomes `(in−1)/(out−1)`, which is
    the align-corners rule, while the centre stays `scale·(i + 0.5)`, which is the
    other one. Taking the align-corners centre `scale·i` instead parts by 1.3 to 4.5.
    Twenty combinations — two modes, both flags, five output sizes — agree with the
    mixture and with nothing tidier.
    """
    if align_corners:
        # **The caller's scale is not read here**, and that is measured: with
        # `align_corners=True` the `scale_factor` cases agree with `(in−1)/(out−1)`
        # alone. The align-corners rule pins both ends and has no room for one.
        scale = ((size_in - 1) / (size_out - 1)) if size_out > 1 else 0.0
    else:
        scale = (1.0 / given) if given else (size_in / size_out)
    wide = scale >= 1.0
    support = radius * scale if wide else radius
    inv = (1.0 / scale) if wide else 1.0
    rows = _np.zeros((size_out, size_in))
    for i in range(size_out):
        centre = scale * (i + 0.5)
        lo = max(int(centre - support + 0.5), 0)
        span = min(int(centre + support + 0.5), size_in) - lo
        taps = _np.array([filt((j + lo - centre + 0.5) * inv) for j in range(span)])
        total = taps.sum()
        rows[i, lo:lo + span] = taps / total if total else taps
    return rows


def _interpolate_antialias(x, size, scale_factor, mode, align_corners,
                           recompute_scale_factor=None):
    """`antialias=True` for `bilinear` and `bicubic`.

    Both axes are a plain matrix multiply against the weight matrices above, so the
    gradient is the multiply's own and nothing was written for it.
    """
    _, _, h, w = x.data.shape
    keep = None
    if size is not None:
        oh, ow = _pair(size)
    else:
        oh, ow = _out_size((h, w), scale_factor)
        if recompute_scale_factor is not True:
            keep = (scale_factor if isinstance(scale_factor, (tuple, list))
                    else (scale_factor, scale_factor))
    # **`abs` is this module's own `abs`**, which takes a tensor — the builtin is
    # shadowed here exactly as `any` is, and calling it on a float raised
    # `'float' object has no attribute 'abs'`. Both kernels get the magnitude by hand.
    radius, filt = ((1, _triangle_weight_aa) if mode == "bilinear"
                    else (2, _cubic_weight_aa))
    wy = _antialias_axis(h, oh, radius, filt, align_corners,
                         keep and keep[0]).astype(x.data.dtype)
    wx = _antialias_axis(w, ow, radius, filt, align_corners,
                         keep and keep[1]).astype(x.data.dtype)
    # `(o, h) · (n, c, h, w) · (w, p)` — the rows fold first, then the columns.
    spread = _wrap(wy) @ x
    return spread @ _wrap(wx.T)


def _grid_pad_index(padding_mode, align_corners):
    """How one tap index is brought back inside, for `grid_sample`'s bicubic window.

    `None` means *leave it outside* — `zeros` wants the tap masked to zero, which is
    what `pick` already does. The other two move it, and `pick`'s mask then never
    fires because the moved index is in range.
    """
    if padding_mode == "border":
        return lambda i, n: _np.clip(i, 0, n - 1)
    if padding_mode == "reflection":
        lo = 0.0 if align_corners else -0.5

        def reflect(i, n):
            hi = (n - 1.0) if align_corners else (n - 0.5)
            if hi <= lo:
                return _np.zeros_like(i)
            span = 2.0 * (hi - lo)
            t = _np.remainder(i - lo, span)
            back = _np.minimum(t, span - t) + lo
            return _np.clip(_np.rint(back), 0, n - 1).astype(int)

        return reflect
    return None


def _pick_padded(pick, edge, iy, ix, h, w):
    """One tap, with the padding applied to the index rather than to the centre."""
    if edge is None:
        return pick(iy, ix)
    return pick(edge(iy, h), edge(ix, w))


def _cubic_of(v):
    """Keys' kernel at `a = −0.75` **as a tensor expression**, so the gradient flows
    through it. The branch points are read off the values, which is where torch reads
    them too — they are a measure-zero set and the polynomial is continuous across
    them, so nothing turns on which side a boundary falls."""
    t = v.abs()
    a = _BICUBIC_A
    near = ((a + 2) * t - (a + 3)) * t * t + 1
    far = ((t - 5) * t + 8) * t * a - 4 * a
    return where(Tensor(t.data <= 1), near,
                 where(Tensor(t.data < 2), far, t * 0.0))


def _triangle_weight_aa(t):
    """The bilinear kernel: one at the centre, zero a cell away."""
    t = -t if t < 0 else t
    return 1.0 - t if t < 1.0 else 0.0


def _cubic_weight_aa(t):
    """Keys' kernel at **`a = −0.5`**, which is the constant torch's anti-aliased path
    uses where its plain `bicubic` uses `−0.75`. Measured, not chosen."""
    t = -t if t < 0 else t
    a = -0.5
    if t <= 1:
        return ((a + 2) * t - (a + 3)) * t * t + 1
    if t < 2:
        return ((t - 5) * t + 8) * t * a - 4 * a
    return 0.0


def _interpolate_bicubic(x, size, scale_factor, align_corners,
                         recompute_scale_factor=None):
    """Cubic convolution over a 4×4 neighbourhood.

    **It was refused, and the refusal said only *not here*.** What it needed was one
    constant and one kernel: with `a = −0.75` and coordinates taken the way the
    bilinear path already takes them, this reproduces torch to float64 noise on six
    combinations — three output sizes against both `align_corners`.

    **The edges clamp rather than reflect or wrap**, which is torch's rule and the
    reason a 4×4 window is safe one cell outside the image on either side.

    Nothing is clamped on the way out: cubic convolution overshoots, so upsampling an
    image in `[0, 1]` gives values slightly outside it. torch does not clamp either,
    and a caller who wants pixels is expected to clamp.
    """
    n, c, h, w = x.data.shape
    keep = None
    if size is not None:
        oh, ow = _pair(size)
    else:
        oh, ow = _out_size((h, w), scale_factor)
        if recompute_scale_factor is not True:
            keep = (scale_factor if isinstance(scale_factor, (tuple, list))
                    else (scale_factor, scale_factor))
    ys = _bicubic_axis(h, oh, align_corners, keep and keep[0])
    xs = _bicubic_axis(w, ow, align_corners, keep and keep[1])
    y0 = _np.floor(ys).astype(int)
    x0 = _np.floor(xs).astype(int)
    ty, tx = ys - y0, xs - x0

    taps = []
    for ky in range(-1, 3):
        wy = _cubic_weight(ky - ty).astype(x.data.dtype)[:, None]
        ry = _np.clip(y0 + ky, 0, h - 1)
        for kx in range(-1, 3):
            wx = _cubic_weight(kx - tx).astype(x.data.dtype)[None, :]
            rx = _np.clip(x0 + kx, 0, w - 1)
            taps.append((ry, rx, wy * wx))

    out = _np.zeros((n, c, oh, ow), dtype=x.data.dtype)
    for ry, rx, weight in taps:
        out = out + x.data[:, :, ry][:, :, :, rx] * weight

    def back(g):
        gg = _np.asarray(g)
        got = _np.zeros_like(x.data)
        for ry, rx, weight in taps:
            share = gg * weight
            # A clamped edge reads the same row several times, so the shares
            # **accumulate** rather than overwrite.
            for i, yi in enumerate(ry):
                _np.add.at(got[:, :, yi], (slice(None), slice(None), rx),
                           share[:, :, i])
        return (got,)

    return x._make(out, (x,), back, "UpsampleBicubic2DBackward0")


def _interpolate_bilinear(x, size, scale_factor, align_corners,
                          recompute_scale_factor=None):
    n, c, h, w = x.data.shape
    keep = None
    if size is not None:
        oh, ow = _pair(size)
    else:
        oh, ow = _out_size((h, w), scale_factor)
        if recompute_scale_factor is not True:
            # **The caller's scale, not the one the output size implies**, and that
            # is torch's default — `None` behaves as `False` here. They part once the
            # floor has lost something: 5 at 1.7 gives 8, and 8 / 5 is 1.6.
            keep = (scale_factor if isinstance(scale_factor, (tuple, list))
                    else (scale_factor, scale_factor))
    y0, y1, wy = _bilinear_axis(h, oh, align_corners, keep and keep[0])
    x0, x1, wx = _bilinear_axis(w, ow, align_corners, keep and keep[1])
    wy = wy.astype(x.data.dtype)[:, None]
    wx = wx.astype(x.data.dtype)[None, :]

    def blend(t):
        top = t[:, :, y0][:, :, :, x0] * (1 - wx) + t[:, :, y0][:, :, :, x1] * wx
        bot = t[:, :, y1][:, :, :, x0] * (1 - wx) + t[:, :, y1][:, :, :, x1] * wx
        return top * (1 - wy) + bot * wy

    def back(g):
        gg = _np.asarray(g)
        out = _np.zeros_like(x.data)
        for ys, wgt_y in ((y0, 1 - wy), (y1, wy)):
            for xs, wgt_x in ((x0, 1 - wx), (x1, wx)):
                share = gg * wgt_y * wgt_x
                # The same position is read several times, so the values are
                # **accumulated.**
                for i, yi in enumerate(ys):
                    _np.add.at(out[:, :, yi], (slice(None), slice(None), xs),
                               share[:, :, i])
        return (out,)

    return x._make(blend(x.data), (x,), back, "UpsampleBilinear2DBackward0")


def _spread(v, n):
    """One number means the same value on every axis; a list is used as it
    is."""
    return [v] * n if isinstance(v, int) else list(v)


def _window_range(window):
    """The positions one window covers. **A third element is the dilation step.**

    A window used to be `(start, end)` and dilation makes it not contiguous:
    `MaxPool2d(2, dilation=2)` looks at columns 0 and 2, not 0 and 1. Written as a
    step on the window rather than as a separate argument, every consumer keeps
    reading one list and there is no second place for the two to disagree — which
    is the rule `_fixed_windows` already states for itself.
    """
    return range(window[0], window[1], window[2] if len(window) > 2 else 1)


def _fold_axis(x, axis, windows, kind):
    """Fold one axis according to a list of windows. **The windows may have
    differing lengths.**

    Adaptive pooling is that place — reducing 8 to 3 gives windows of 3, 3 and 2.
    A fixed window size cannot handle the non-dividing cases at all, and that is
    really why `adaptive_avg_pool2d` was refusing anything that was not a
    multiple.

    The slices are taken out one at a time and folded, so **the derivative
    follows on its own** — there is no backward formula to write. The maximum is
    folded with `_maximum_first`, and giving the earlier position on a tie is
    torch's rule.
    """
    parts = []
    for window in windows:
        pieces = [x[_slice_at(axis, j, j + 1)] for j in _window_range(window)]
        acc = pieces[0]
        for piece in pieces[1:]:
            acc = acc + piece if kind == "avg" else _maximum_first(acc, piece)
        parts.append(acc * (1.0 / len(pieces)) if kind == "avg" else acc)
    return cat(parts, axis)


def _adaptive_windows(n_in, n_out):
    """torch's adaptive rule. The start floors and the end ceils.

    Dividing evenly it is uniform, and otherwise **the window size differs per
    position.** That rule cannot be written in one line, which is where the
    values diverge, and the golden asks both the dividing and the non-dividing
    case.
    """
    return [((i * n_in) // n_out, -((-(i + 1) * n_in) // n_out))
            for i in range(n_out)]


def _adaptive(x, output_size, kind):
    """Fold one axis at a time. **The windows are rectangular, so splitting by
    axis gives the same value** — every row has the same length, so the mean of
    means is the overall mean, and the maximum is that way by nature."""
    x = _wrap(x)
    spatial = len(x.data.shape) - 2
    sizes = _spread(output_size, spatial)
    out = x
    for k in range(spatial):
        axis = 2 + k
        out = _fold_axis(out, axis,
                         _adaptive_windows(out.data.shape[axis], sizes[k]), kind)
    return out


def _fixed_windows(n_in, size, step, ceil_mode=False):
    """The list of fixed windows. **Written in one place only** — values and
    positions looking at different windows would diverge.

    `ceil_mode` adds the trailing window that rounding up allows, **clipped to the
    input** rather than padded: with no padding there is nothing to pad with, and
    torch folds over the cells that are really there. It is dropped if it would
    start past the end, which is the same rule `_pool_windows` states at length.
    """
    stops = list(range(0, n_in - size + 1, step))
    if ceil_mode:
        nxt = (stops[-1] + step) if stops else 0
        if nxt < n_in and (not stops or stops[-1] + size < n_in):
            stops.append(nxt)
    return [(s, min(s + size, n_in), 1) for s in stops]


def _pool_windows(n_in, size, stride, padding, dilation, ceil_mode):
    """Pooling windows over the **padded** axis, in torch's arithmetic.

    `n_in` is the unpadded length; the windows are positions in the padded one,
    because padding is done by padding and the window list is what carries every
    other difference.

    **`ceil_mode` is not simply a ceiling.** Rounding up can put the last window
    entirely inside the right padding, and torch drops that one — the
    documentation says a window is allowed off the end only if it *starts* within
    the input or the left padding. Take the ceiling and stop there and a 5-long
    axis with kernel 2, stride 3 and padding 1 gives one window too many, all of
    it padding, whose maximum is `-inf`.

    Measured against real torch rather than read off the formula, which is how
    the `ceil_mode` row in `tests/torch_signatures_core.py` was closed.
    """
    span = (size - 1) * dilation + 1
    total = n_in + 2 * padding
    if ceil_mode:
        count = -(-(total - span) // stride) + 1
        while count > 1 and (count - 1) * stride >= n_in + padding:
            count -= 1
    else:
        count = (total - span) // stride + 1
    if count < 1:
        raise RuntimeError(
            f"pool: the window ({span}) does not fit the padded input ({total})")
    # **The last window is clipped to the padded extent.** Under `ceil_mode` the
    # ceiling lets a window run off the end, and torch simply ignores the cells
    # that are not there — for a maximum that is the same as clipping, because
    # what is not there is the `-inf` the padding is made of. Left unclipped the
    # slice runs past the axis and the fold builds an empty piece.
    return [(s * stride, min(s * stride + span, total), dilation)
            for s in range(count)]


def _fixed_window_axes(shape, kernel_size, stride):
    """The window list per axis. It uses the same position arithmetic as
    `_fixed`."""
    spatial = len(shape) - 2
    kernels = _spread(kernel_size, spatial)
    strides = _spread(stride if stride is not None else kernel_size, spatial)
    return [_fixed_windows(shape[2 + k], kernels[k], strides[k])
            for k in range(spatial)]


def _fixed(x, kernel_size, stride, kind, ceil_mode=False):
    """A fixed window. The same machine as the adaptive one with a different
    window list."""
    x = _wrap(x)
    spatial = len(x.data.shape) - 2
    kernels = _spread(kernel_size, spatial)
    strides = _spread(stride if stride is not None else kernel_size, spatial)
    out = x
    for k in range(spatial):
        axis = 2 + k
        windows = _fixed_windows(out.data.shape[axis], kernels[k], strides[k],
                                 ceil_mode)
        out = _fold_axis(out, axis, windows, kind)
    return out


def _max_with_index(x, window_axes, positions=None):
    """Produce the maximum together with **the winning position.**

    The position follows torch's convention as **a flat index within the plane** —
    `h*W + w` in 2-D and `(d*H + h)*W + w` in 3-D. It restarts from 0 per batch
    and per channel (measured: `tests/probe_pool.py`). `MaxUnpool` puts values
    back at exactly this index.

    **The axes are folded from the back.** On a tie torch chooses the smaller
    flat index, that is, the position that comes first in row-major order.
    Folding from the front chooses "the first row within each column" and then
    "the first column among those rows", giving **the column-major first**, and
    since the values match nothing catches it while the position alone differs.
    Folding from the back settles the first column within a row and then the
    first row, giving the row-major first.

    The value is produced here as well. Obtaining the value by another path
    leaves room for "position A with value B", and both being plausible makes it
    invisible.
    """
    x = _wrap(x)
    data = x.data
    shape = data.shape
    spatial = shape[2:]
    plane = int(_np.prod(spatial)) if spatial else 1

    val = data
    # **`positions` is given when the input was padded.** torch's indices are into
    # the *unpadded* plane (measured), so counting them off the padded one would be
    # right in shape and wrong in value — and `MaxUnpool` would then put every
    # value back in the wrong cell, quietly, since the values themselves match.
    pos = (positions if positions is not None
           else _np.broadcast_to(_np.arange(plane).reshape(spatial), shape))
    for k in reversed(range(len(window_axes))):
        axis = 2 + k
        vparts, pparts = [], []
        for window in window_axes[k]:
            span = _window_range(window)
            cut = (slice(None),) * axis + (slice(span.start, span.stop, span.step),)
            vs, ps = val[cut], pos[cut]
            j = vs.argmax(axis=axis)[(slice(None),) * axis + (None,)]
            vparts.append(_np.take_along_axis(vs, j, axis))
            pparts.append(_np.take_along_axis(ps, j, axis))
        val = _np.concatenate(vparts, axis)
        pos = _np.concatenate(pparts, axis)

    # The gradient goes only to the winning positions. The indices are lifted to
    # whole-tensor flat indices and scattered in one go.
    base = (_np.arange(shape[0] * shape[1]) * plane).reshape(shape[0], shape[1],
                                                             *([1] * len(spatial)))
    flat = (base + pos).reshape(-1)

    def back(g):
        gx = _np.zeros(data.size, dtype=_np.asarray(g).dtype)
        _np.add.at(gx, flat, _np.asarray(g).reshape(-1))
        return (gx.reshape(shape),)

    return x._make(val, (x,), back, "MaxPoolWithIndicesBackward0"), pos


def adaptive_avg_pool2d(input, output_size):
    """Average pooling to a chosen output size.

    **It does not have to be a multiple.** It used to refuse, and torch handles it
    by taking a different window size per position — the refusal was a different
    rule rather than an imitation.
    """
    return _adaptive(input, _pair(output_size), "avg")


def adaptive_avg_pool1d(input, output_size):
    return _adaptive(input, _spread(output_size, 1), "avg")


def adaptive_avg_pool3d(input, output_size):
    return _adaptive(input, _spread(output_size, 3), "avg")


def adaptive_max_pool1d(input, output_size, return_indices=False):
    if return_indices:
        return adaptive_max_pool1d_with_indices(input, output_size)
    return _adaptive(input, _spread(output_size, 1), "max")


def adaptive_max_pool2d(input, output_size, return_indices=False):
    if return_indices:
        return adaptive_max_pool2d_with_indices(input, output_size)
    return _adaptive(input, _pair(output_size), "max")


def adaptive_max_pool3d(input, output_size, return_indices=False):
    if return_indices:
        return adaptive_max_pool3d_with_indices(input, output_size)
    return _adaptive(input, _spread(output_size, 3), "max")


def avg_pool1d(input, kernel_size, stride=None, padding=0, ceil_mode=False,   # noqa: A002
               count_include_pad=True):
    """**`padding` and `count_include_pad` were refused here**, on the ground that
    `avg_pool2d` next door has them and this *shares none of its code*. The ground
    was exact, and it named its own remedy: the 2-D body was lifted into a
    rank-agnostic `_avg_pool_nd` and all three ranks now call it, so the refusal has
    nothing left to stand on.

    torch gives 1-D no `divisor_override` — 2-D and 3-D have one and this does not.
    Following the authority means following its inconsistencies; offering one here
    would be an argument torch declines.
    """
    return _avg_pool_nd(input, 1, kernel_size, stride, padding, ceil_mode,
                        count_include_pad, None, "AvgPool1DBackward0")


def avg_pool3d(input, kernel_size, stride=None, padding=0, ceil_mode=False,   # noqa: A002
               count_include_pad=True, divisor_override=None):
    """See `avg_pool1d` on the refusals that used to be here."""
    return _avg_pool_nd(input, 3, kernel_size, stride, padding, ceil_mode,
                        count_include_pad, divisor_override, "AvgPool3DBackward0")


def lp_pool2d(input, norm_type, kernel_size, stride=None, ceil_mode=False):   # noqa: A002
    """The `p`-th root of the sum of `p`-th powers. At p=1 it is the sum, and at
    large p it approaches the maximum.

    **It follows torch's assembly exactly** — average pooling, multiplied back by
    the window size into a sum, then the root. The sign and the `relu` in the
    middle are that implementation's too.

    `ceil_mode` therefore costs nothing beyond passing it on, and that is the whole
    of what was missing: torch's is the fourth argument here and this stopped at
    three, so `lp_pool2d(x, 2, 3, 2, True)` raised where torch rounded up.
    """
    input = _wrap(input)
    kh, kw = _pair(kernel_size)
    out = avg_pool2d(input ** norm_type, kernel_size, stride, ceil_mode=ceil_mode)
    return ((out.sign() * relu(out.abs())) * (kh * kw)) ** (1.0 / norm_type)


def lp_pool1d(input, norm_type, kernel_size, stride=None, ceil_mode=False):  # noqa: A002
    input = _wrap(input)                                                     # noqa: A001
    k = kernel_size if isinstance(kernel_size, int) else kernel_size[0]
    out = avg_pool1d(input ** norm_type, k, stride, ceil_mode=ceil_mode)
    return ((out.sign() * relu(out.abs())) * k) ** (1.0 / norm_type)


def lp_pool3d(input, norm_type, kernel_size, stride=None, ceil_mode=False):  # noqa: A002
    """The same assembly as 1-D and 2-D. Only the window cell count becomes the
    product of three axes."""
    input = _wrap(input)
    kd, kh, kw = _spread(kernel_size, 3)
    out = avg_pool3d(input ** norm_type, kernel_size, stride, ceil_mode=ceil_mode)
    return ((out.sign() * relu(out.abs())) * (kd * kh * kw)) ** (1.0 / norm_type)


def max_pool2d(x, kernel_size, stride=None, padding=0, dilation=1,
               return_indices=False, ceil_mode=False):
    if not _pool_is_plain(padding, dilation, ceil_mode):
        return _max_pool_nd(x, 2, kernel_size, stride, padding, dilation,
                            ceil_mode, return_indices)
    if return_indices:
        return max_pool2d_with_indices(x, kernel_size, stride)
    stride = stride or kernel_size
    xd = x.data
    N, C, H, W = xd.shape
    OH = (H - kernel_size) // stride + 1
    OW = (W - kernel_size) // stride + 1
    win = _np.lib.stride_tricks.sliding_window_view(xd, (kernel_size, kernel_size), axis=(2, 3))
    win = win[:, :, ::stride, ::stride, :, :].reshape(N, C, OH, OW, -1)
    idx = win.argmax(axis=-1)
    out = _np.take_along_axis(win, idx[..., None], axis=-1).squeeze(-1)

    def back(g):
        # The gradient goes only to where the maximum was. The positions are
        # turned into flat indices and scattered in one go — far faster than
        # looping N·C·OH·OW in Python, with the same result.
        g = _np.asarray(g)
        di, dj = _np.divmod(idx, kernel_size)
        n_i, c_i, oh_i, ow_i = _np.ogrid[:N, :C, :OH, :OW]
        h = oh_i * stride + di
        w = ow_i * stride + dj
        flat = ((n_i * C + c_i) * H + h) * W + w
        gx = _np.zeros(xd.size, dtype=g.dtype)
        _np.add.at(gx, flat.reshape(-1), g.reshape(-1))
        return (gx.reshape(xd.shape),)

    return x._make(out, (x,), back)


def sin(input): return input._make(_np.sin(_float_in(input.data)), (input,),
                           lambda g: (g * _np.cos(_float_in(input.data)),), "SinBackward0")
def cos(input): return input._make(_np.cos(_float_in(input.data)), (input,),
                           lambda g: (-g * _np.sin(_float_in(input.data)),), "CosBackward0")


def clamp(input, min=None, max=None):
    out = _np.clip(input.data, min, max)
    inside = _np.ones_like(input.data, dtype=bool)
    if min is not None:
        inside &= input.data >= min
    if max is not None:
        inside &= input.data <= max
    return input._make(out, (input,), lambda g: (g * inside,), "ClampBackward0")



# --------------------------------------------------------- elementwise functions
#
# Mostly one numpy line and one derivative line. The ones with no derivative
# (floor, sign and the like) get a gradient of 0 — torch does that too, because a
# step function's derivative is 0 almost everywhere.

# **Which unary functions promote an integer input to a float**, and it is torch's
# list rather than a judgement: `torch.sin(tensor([1, 2, 3]))` is `float32` and
# `torch.abs` of the same is `int64`. Derived by asking torch once and written down,
# because a library cannot ask torch at run time.
#
# Two families were wrong and they were wrong differently. Twelve — `sin`, `exp`,
# `sqrt` and the rest — handed numpy an integer array, got `float64` back, and
# returned that: right values, **twice the memory and a dtype that spreads**, since
# everything downstream promotes to meet it. Eight others — `erf`, `erfc`, `erfinv`,
# `digamma`, `lgamma`, `i0`, `reciprocal` — are written as numpy expressions that stay
# integral, so `erf(tensor([1, 2, 3]))` was `tensor([0, 0, 0])`: **the answer
# truncated into the input's type**, with no warning and no error.
#
# `erfinv` was the loudest of the silent ones: its answer runs to infinity, and cast
# into an integer cell that is `9223372036854775807`.
#
# One line fixes both, because both come from the same place — the input crosses into
# numpy as an integer. Cast it first and numpy produces the default dtype throughout.
_PROMOTES_INTEGERS = frozenset({
    "Acos", "Acosh", "Asin", "Asinh", "Atan", "Atanh", "Cos", "Cosh", "Digamma",
    "Erf", "Erfc", "Erfinv", "Exp", "Exp2", "Expm1", "I0", "Lgamma", "Log", "Log10",
    "Log1p", "Log2", "Logit", "Reciprocal", "Rsqrt", "Sigmoid", "Sin", "Sinc",
    "Sinh", "Sqrt", "Tan", "Tanh", "Deg2rad", "Rad2deg",
})


# **`None` is a legitimate value here**, so absence cannot be spelled as `None`.
# `flip(t, dims=None)` means *every axis* and is not the same call as `flip(t, 0)`.
#
# **Not called `_MISSING`, and the first version was.** That name is already a
# module-level sentinel fifteen hundred lines up, holding `where`'s two optional
# arguments apart from a real `None`. Rebinding it left `where`'s defaults pointing
# at the old object while its own comparisons read the new one, so both branches went
# false and the sentinel came back **as a value** — `where(cond)` returned a list of
# them. A name collision inside one module, and the symptom was a tensor full of
# `<object object>`.
_NOT_GIVEN = object()


def _loose(args, kw, key):
    """The axes given as loose numbers, as a sequence, or by keyword.

    **torch takes all three and each of these functions took one.** `x.flip(0, 1)`,
    `x.flip([0, 1])` and `x.flip(dims=[0, 1])` are the same call there, and here the
    first and third stopped with a `TypeError` about the argument count. Widening to
    `*dims` alone fixes the first and breaks the third — `keep::count_nonzero(dim=1)`
    in the golden went red on exactly that, which is what a frozen case is for.

    Returns `()` when nothing was given, which every caller reads as *all axes*.

    **What it pops is the only keyword these four take, and the rest was going
    nowhere.** `kw` arrives as the callers' `**kw`, this reads one name out of it and
    nothing ever looked at what stayed — so `flip(t, 0, zzz=1)` ran and said nothing
    where torch raises. The other thirty-three seats in this module lost their bag
    outright; these four keep one because `dims=` genuinely arrives that way, and the
    refusal has to live here, at the one place that knows which key is legitimate.

    The wording is Python's own, because that is what the caller would have met had
    the parameter simply not existed — and it is one of the two forms torch uses.
    """
    got = kw.pop(key, _NOT_GIVEN)
    if kw:
        stray = sorted(kw)[0]
        raise TypeError(f"got an unexpected keyword argument {stray!r}")
    if got is not _NOT_GIVEN:
        if got is None:
            return ()
        return tuple(got) if isinstance(got, (tuple, list)) else (got,)
    if not args:
        return ()
    if len(args) == 1:
        first = args[0]
        if first is None:
            return ()
        return tuple(first) if isinstance(first, (tuple, list)) else (first,)
    return tuple(args)


def _unary(name, forward, derivative=None, op=None):
    # **The parameter is torch's name, and it is one line for forty functions.**
    # Every table unary is this closure, so `abs`, `sin`, `sqrt` and the rest all
    # took `t` where torch takes `input` — one rename here reaches all of them.
    def fn(input):
        input = _wrap(input)
        data = input.data
        if name in _PROMOTES_INTEGERS:
            data = _float_in(data)
        out = forward(data)
        if derivative is None:
            return Tensor(out)
        return input._make(out, (input,), lambda g: (g * derivative(data, out),),
                           op or f"{name}Backward0")
    fn.__name__ = name
    return fn


# numpy has no erf. `np.vectorize(math.erf)` **calls Python per element** — a
# loop rather than a vectorisation, and especially bad on wasm where a Python
# call is expensive. Abramowitz & Stegun 7.1.26 is written with numpy elementwise
# operations (absolute error 1.5e-7 — around float32's eps of 1.19e-7, so below
# the last digit for a library that answers in float32).
_ERF_P = 0.3275911
_ERF_A = (0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429)


def _erfc_pos(y):
    """erfc for y >= 0. A polynomial × exp(-y²), so **there is no subtraction** —
    this is the primitive and erf is derived from it. The other way round (erf as
    the primitive) loses digits in the tail."""
    t = 1.0 / (1.0 + _ERF_P * y)
    poly = t * (_ERF_A[0] + t * (_ERF_A[1] + t * (_ERF_A[2] + t * (_ERF_A[3] + t * _ERF_A[4]))))
    return poly * _np.exp(-y * y)


def _erf64(x):
    """Computed in float64 and handed back.

    In float32 it becomes `1 - (something near 1)` around the origin and the
    significant digits go (measured: computed in float32, 5,124 of 46,000 grid
    points break allclose(1e-5)).
    """
    d = _np.asarray(x, dtype=_np.float64)
    return _np.sign(d) * (1.0 - _erfc_pos(_np.abs(d)))


def _one_plus_erf64(z):
    """1 + erf(z). At large negative z the 1 and the erf cancel, so that side is
    obtained directly through erfc — gelu's left tail is exactly that place."""
    d = _np.asarray(z, dtype=_np.float64)
    tail = _erfc_pos(_np.abs(d))
    return _np.where(d >= 0, 2.0 - tail, tail)


log2 = _unary("Log2", _np.log2, lambda x, o: 1.0 / (x * _np.log(2)))
log10 = _unary("Log10", _np.log10, lambda x, o: 1.0 / (x * _np.log(10)))
rsqrt = _unary("Rsqrt", lambda x: 1.0 / _np.sqrt(x), lambda x, o: -0.5 * o / x)
square = _unary("Square", lambda x: _np.square(_arith_in(x)), lambda x, o: 2 * x)
reciprocal = _unary("Reciprocal", _np.reciprocal, lambda x, o: -o * o)
tan = _unary("Tan", _np.tan, lambda x, o: 1 + o * o)
sinh = _unary("Sinh", _np.sinh, lambda x, o: _np.cosh(x))
cosh = _unary("Cosh", _np.cosh, lambda x, o: _np.sinh(x))
erf = _unary("Erf", lambda x: _erf64(x).astype(x.dtype),
             lambda x, o: 2 / _np.sqrt(_np.pi) * _np.exp(-x * x))
# Step-shaped — the derivative is 0 almost everywhere.
#
# **They flow a 0. They do not cut the graph.** Asked of torch, all four run
# `backward()` and fill `.grad` with zeros. This used to hand back a bare tensor
# and `backward()` refused, and that was "an absent feature" rather than matching
# torch. A value of 0 and being uncallable are different things, and a loss with a
# step function in the middle really does run in torch.
_zero_grad = lambda x, o: _np.zeros_like(x)                          # noqa: E731

sign = _unary("Sign", _np.sign, _zero_grad)
floor = _unary("Floor", _np.floor, _zero_grad)
ceil = _unary("Ceil", _np.ceil, _zero_grad)
_round_unary = _unary("Round", lambda x: _np.round(x), _zero_grad)


def round(input, decimals=0):
    """**torch takes `decimals` and this did not**, so `round(x, 2)` was a
    `TypeError` where torch rounds to two places. The name is one of the most
    ordinary in the library and the argument is the reason anybody reaches for it.

    torch rounds **half to even** at every scale, which is numpy's rule too — 0.5
    and 1.5 both give 0 and 2. Scaling by a power of ten before rounding keeps
    that, which is what numpy does internally.
    """
    if decimals:
        input = _wrap(input)
        _needs_float(input.data,
                     "Rounding to a number of places has no meaning in an "
                     "integer cell.",
                     "round_cpu not implemented for integer decimals")
        scale = 10.0 ** int(decimals)
        return _round_unary(input * scale) / scale
    return _round_unary(input)


def neg(input): return -_wrap(input)
def pow(input, exponent): return _wrap(input) ** exponent


# ---- the rest of the trigonometric, exponential and logarithmic ones
#
# All elementwise with closed-form derivatives, so `_unary` finishes them. There
# is no reason to write them out one at a time. **torch's aliases are attached
# alongside** — `arccos` is the same function as `acos` and tutorials use both.
# Only the names differ and the implementation is one, so there is nowhere to
# diverge.

acos = _unary("Acos", _np.arccos, lambda x, o: -1.0 / _np.sqrt(1 - x * x))
asin = _unary("Asin", _np.arcsin, lambda x, o: 1.0 / _np.sqrt(1 - x * x))
atan = _unary("Atan", _np.arctan, lambda x, o: 1.0 / (1 + x * x))
acosh = _unary("Acosh", _np.arccosh, lambda x, o: 1.0 / _np.sqrt(x * x - 1))
asinh = _unary("Asinh", _np.arcsinh, lambda x, o: 1.0 / _np.sqrt(x * x + 1))
atanh = _unary("Atanh", _np.arctanh, lambda x, o: 1.0 / (1 - x * x))
expm1 = _unary("Expm1", _np.expm1, lambda x, o: o + 1.0)
log1p = _unary("Log1p", _np.log1p, lambda x, o: 1.0 / (1 + x))
exp2 = _unary("Exp2", _np.exp2, lambda x, o: o * _np.log(2))
deg2rad = _unary("Deg2rad", _np.deg2rad, lambda x, o: _np.float32(_np.pi / 180))
rad2deg = _unary("Rad2deg", _np.rad2deg, lambda x, o: _np.float32(180 / _np.pi))
# The truncating ones — being steps they flow a 0 (`floor`'s reason above).
trunc = _unary("Trunc", _np.trunc, _zero_grad)
frac = _unary("Frac", lambda x: x - _np.trunc(x), lambda x, o: _np.ones_like(x))
# `sgn` equals `sign` over the reals — it flows a 0.
#
# It was first written down that "torch refuses to backpropagate sgn", and that
# was **wrong.** The exception came from the `print` showing the result rather
# than from `backward()`, and it was read as a refusal. torch's sgn gradient is a
# ZeroTensor (a lazy zero tensor), so `.numpy()` refuses it and the value is 0.
sgn = _unary("Sgn", _np.sign, _zero_grad)
positive = _unary("Positive", lambda x: x, lambda x, o: _np.ones_like(x))
# Written as `erfc = 1 - erf` the digits go in the tail. **`_erfc_pos` is the
# primitive**, so it is derived from there directly — which is exactly why erf was
# built that way.
erfc = _unary("Erfc",
              lambda x: _np.where(x >= 0, _erfc_pos(_np.abs(_np.asarray(x, _np.float64))),
                                  2.0 - _erfc_pos(_np.abs(_np.asarray(x, _np.float64)))
                                  ).astype(x.dtype),
              lambda x, o: -2 / _np.sqrt(_np.pi) * _np.exp(-x * x))
sinc = _unary("Sinc", _np.sinc,
              # d/dx sinc(x) = (cos(πx) - sinc(x)) / x, and 0 at x=0.
              lambda x, o: _np.where(x == 0, 0.0,
                                     (_np.cos(_np.pi * _np.where(x == 0, 1.0, x)) - o)
                                     / _np.where(x == 0, 1.0, x)))
def logit(input, eps=None):
    """The inverse of the sigmoid. **`eps` clamps the input away from 0 and 1**,
    where the answer runs to infinity — torch takes it and this did not, so
    `x.logit(1e-6)`, the form used on probabilities that may be exactly 0, stopped
    with a `TypeError`.

    Without `eps` the infinities are torch's answer too, so the default is unchanged.
    """
    input = _wrap(input)
    if eps is None:
        return _logit_raw(input)
    lo = float(eps)
    return _logit_raw(_wrap(_np.clip(_np.asarray(input.data), lo, 1.0 - lo))
                      if not isinstance(input, Tensor) else input.clamp(lo, 1.0 - lo))


_logit_raw = _unary("Logit", lambda x: _np.log(x / (1 - x)),
                    lambda x, o: 1.0 / (x * (1 - x)))

# torch's aliases. They point at the same function.
arccos, arcsin, arctan = acos, asin, atan
arccosh, arcsinh, arctanh = acosh, asinh, atanh
fix = trunc
absolute = abs
negative = neg
clip = clamp


def _binary_math(name, forward, d_a, d_b, op=None):
    """An elementwise function taking two tensors. Broadcasting and the backward
    are left to `_binary`.

    The derivative takes `(x, y)` and hands back **what to multiply the gradient
    by.** The signature `_binary` passes is `(g, x, y)`, so it is wrapped here.
    """
    # As `_unary` above: one rename for every table binary.
    def fn(input, other):
        input = _wrap(input)
        return input._binary(other, forward,
                             lambda g, x, y: g * d_a(x, y),
                         lambda g, x, y: g * d_b(x, y),
                         op or f"{name}Backward0")
    fn.__name__ = name
    return fn


atan2 = _binary_math("Atan2", _np.arctan2,
                     lambda x, y: y / (x * x + y * y),
                     lambda x, y: -x / (x * x + y * y))
hypot = _binary_math("Hypot", _np.hypot,
                     lambda x, y: x / _np.hypot(x, y),
                     lambda x, y: y / _np.hypot(x, y))
# It is |x|·sign(y), so towards x it is sign(x)·sign(y) and towards y it is 0 (a
# step).
copysign = _binary_math("Copysign", _np.copysign,
                        lambda x, y: _np.sign(x) * _np.sign(y),
                        lambda x, y: _np.zeros_like(_np.copysign(x, y)))
logaddexp = _binary_math("Logaddexp", _np.logaddexp,
                         lambda x, y: _np.exp(x - _np.logaddexp(x, y)),
                         lambda x, y: _np.exp(y - _np.logaddexp(x, y)))
logaddexp2 = _binary_math("Logaddexp2", _np.logaddexp2,
                          lambda x, y: _np.exp2(x - _np.logaddexp2(x, y)),
                          lambda x, y: _np.exp2(y - _np.logaddexp2(x, y)))


def xlogy(input, other):
    """`x · log(y)`, and **0 where x is 0** — `0 · log(0)` is not left as nan."""
    input = _wrap(input)
    with _np.errstate(divide="ignore", invalid="ignore"):
        return input._binary(
            other,
            lambda x, y: _np.where(x == 0, 0.0, x * _np.log(y)),
            lambda g, x, y: g * _np.where(x == 0, 0.0, _np.log(y)),
            lambda g, x, y: g * _np.where(x == 0, 0.0, x / y),
            "XlogyBackward0")


def signbit(input):
    return Tensor(_np.signbit(_wrap(input).data))


def heaviside(input, values):
    input, v = _wrap(input), _wrap(values)
    return Tensor(_np.heaviside(input.data, v.data))


def ldexp(input, other):
    input, o = _wrap(input), _wrap(other)
    return input * Tensor(_np.exp2(o.data.astype(input.data.dtype)))


# ----------------------------------------------------------------- comparison

def _compare(name, fn):
    def cmp(input, other):
        input = _wrap(input)
        rhs = other.data if isinstance(other, Tensor) else other
        return Tensor(fn(input.data, rhs))
    cmp.__name__ = name
    return cmp


eq = _compare("eq", _np.equal)
ne = _compare("ne", _np.not_equal)
lt = _compare("lt", _np.less)
le = _compare("le", _np.less_equal)
gt = _compare("gt", _np.greater)
ge = _compare("ge", _np.greater_equal)
logical_and = _compare("logical_and", _np.logical_and)
logical_or = _compare("logical_or", _np.logical_or)
isnan = _unary("IsNan", _np.isnan)
isinf = _unary("IsInf", _np.isinf)


def logical_not(input): return Tensor(_np.logical_not(_wrap(input).data))


def _split_at_ties(a, b):
    """**A tie splits in half.** That is what torch does — the gradient of
    `maximum(2, 2)` is 0.5 on both sides.

    Split with `a >= b` alone, a takes everything on a tie and b receives 0. The
    forward pass is equally right either way, so a value comparison never catches
    it, and neither does input with no ties — two sets of random numbers are
    never exactly equal.
    """
    tie = a.data == b.data
    left = _np.where(tie, 0.5, (a.data > b.data).astype(a.data.dtype))
    return left, 1.0 - left


def _maximum_first(a, b):
    """On a tie **the earlier one** takes everything — max pooling is this side.

    There is a reason it is kept apart from `maximum`. torch's `maximum` splits a
    tie in half and `max_pool` chooses **one** winning position and flows only
    there (it uses argmax inside). Built on top of `maximum`, the pooling
    diverges quietly whenever a window holds two equal values.
    """
    a, b = _wrap(a), _wrap(b)
    pick = a.data >= b.data
    return a._make(_np.maximum(a.data, b.data), (a, b),
                   lambda g: (g * pick, g * ~pick), "MaximumBackward0")


def maximum(input, other):
    input, other = _wrap(input), _wrap(other)
    la, lb = _split_at_ties(input, other)
    return input._make(_np.maximum(input.data, other.data), (input, other),
                   lambda g: (g * la, g * lb), "MaximumBackward0")


def minimum(input, other):
    input, other = _wrap(input), _wrap(other)
    lb, la = _split_at_ties(input, other)
    return input._make(_np.minimum(input.data, other.data), (input, other),
                   lambda g: (g * la, g * lb), "MinimumBackward0")


# ------------------------------------------------------- shape and selection

def split(tensor, split_size_or_sections, dim=0):
    """**The function and the method take different keywords, and torch refuses
    each other's.** Measured:

        torch.split(t, split_size_or_sections=2)   accepted
        torch.split(t, split_size=2)               TypeError
        t.split(split_size=2)                      accepted
        t.split(split_size_or_sections=2)          TypeError

    So one signature cannot serve both, and this is the function's. `Tensor.split`
    is bound separately in `__init__` — the same split `Tensor.softmax` needed an
    hour earlier, and the same reason: sharing an implementation is not sharing a
    signature.

    The first parameter is `tensor`, not `input`. torch spells this one differently
    from every other function in the namespace (`torch.split(input=…)` is a
    `TypeError`), and matching an authority means matching where it disagrees with
    itself.
    """
    t = _wrap(tensor)
    size = split_size_or_sections
    dim = _pos_dim(t, dim)
    n = t.data.shape[dim]
    if not isinstance(size, (list, tuple)) and size <= 0 and not (size == 0 and n == 0):
        # **`n // 0` was reaching the caller as `ZeroDivisionError`.** That is not a
        # wrong choice of exception, it is no choice at all: an arithmetic error from
        # inside, naming nothing the caller did, and not caught by the `except
        # RuntimeError` a torch user writes. torch allows `0` only where the
        # dimension is empty too.
        raise RuntimeError(
            f"split_size can only be 0 if dimension size is 0, but got dimension "
            f"size of {n}")
    sizes = size if isinstance(size, (list, tuple)) else \
        [size] * (n // size) + ([n % size] if n % size else []) if size else [0]
    cuts, out, start = [], [], 0
    for sz in sizes[:-1]:
        start += sz
        cuts.append(start)
    return tuple(t[_slice_at(dim, s, e)] for s, e in zip([0] + cuts, cuts + [n]))


def chunk(input, chunks, dim=0):
    input = _wrap(input)
    if chunks <= 0:
        raise RuntimeError(
            f"chunk expects `chunks` to be greater than 0, got: {chunks}")
    n = input.data.shape[dim]
    size = -(-n // chunks)
    return split(input, size, dim)


def _slice_at(dim, start, end):
    """An index tuple that slices axis `dim` alone. **`dim` has to be
    non-negative** — given a negative, `range(dim)` is empty and it **slices axis
    0** instead. With no exception."""
    return tuple(slice(None) for _ in range(dim)) + (slice(start, end),)


def _pos_dim(t, dim, extra=0):
    """A negative axis to a non-negative one.

    **`_slice_at` cannot take a negative.** `narrow(x, -1, …)` was **slicing axis
    0** at rank 2 and above, and at rank 1 axis −1 and axis 0 are the same, so it
    went unseen for a long time. It first surfaced on a batched signal (1, 16)
    while assembling `stft` — the shape came out (0, 24) and `stack` stopped. It
    was luck that the shape collapsed loudly instead of the values being wrong.

    **It did not check the range**, and every caller that reaches numpy inherited
    numpy's `AxisError` — which subclasses `IndexError`, so those agree with torch by
    luck. The callers that do their own slicing inherited **nothing**: `diff`,
    `gradient` and `nanmedian` answered for `dim=7` on a two-dimensional tensor as
    though the axis were the last one. Three plausible tensors, no error.

    Found by sweeping torch's `dim` surface mechanically rather than by writing cases:
    38 functions take a `dim`, and asking every one of them for an axis that does not
    exist is a question no case list contained. The wording is torch's own.
    """
    n = t.data.ndim
    limit = n + extra
    if not -limit <= dim < limit:
        raise IndexError(
            f"Dimension out of range (expected to be in range of "
            f"[{-limit}, {limit - 1}], but got {dim})")
    return dim + n if dim < 0 else dim


def unbind(input, dim=0):
    input = _wrap(input)
    dim = _pos_dim(input, dim)
    return tuple(input[_slice_at(dim, i, i + 1)].squeeze(dim)
                 for i in range(input.data.shape[dim]))


def narrow(input, dim, start, length):
    """**A run that would leave the end refuses.** A Python slice does not — it
    stops at the end and hands back what it has — so `narrow(0, 1, 9)` on a length
    of three answered `[2., 3.]`, a plausible tensor of the wrong length, and
    nothing said so. torch raises.

    Found by asking which refusals differ from torch's in the *class* they raise:
    two of the thirty asked turned out not to raise at all. Looking for a missing
    exception is what found a wrong answer — there was no golden case, because
    writing one needs somebody who already suspected.
    """
    input = _wrap(input)
    axis = _pos_dim(input, dim)
    size = input.data.shape[axis]
    start = start + size if start < 0 else start
    if length < 0 or start < 0 or start + length > size:
        raise RuntimeError(
            f"start ({start}) + length ({length}) exceeds dimension size ({size}).")
    return input[_slice_at(axis, start, start + length)]


def flip(t, *dims, **kw):
    """**Loose axis numbers, a list, or `dims=`** — torch takes all three."""
    t = _wrap(t)
    dims = _loose(dims, kw, "dims")
    return t._make(_np.flip(t.data, dims).copy(), (t,),
                   lambda g: (_np.flip(_np.asarray(g), dims).copy(),), "FlipBackward0")


def roll(input, shifts, dims=None):
    input = _wrap(input)
    return input._make(_np.roll(input.data, shifts, dims), (input,),
                   lambda g: (_np.roll(_np.asarray(g), _negate(shifts), dims),), "RollBackward0")


# ---- elementwise in-place operations
#
# `Tensor`'s `_inplace` is used as it is. An existing function does the
# arithmetic and this does **only the writing back into its own buffer** — two
# copies of the same formula eventually diverge.

_INPLACE_UNARY = ("abs", "sqrt", "exp", "log", "sin", "cos", "tan", "tanh", "sigmoid",
                  "relu", "erf", "floor", "ceil", "sign", "reciprocal",
                  "square", "trunc", "frac", "neg", "rsqrt", "log2", "log10",
                  "expm1", "log1p", "acos", "asin", "atan", "sinh", "cosh")

# **The ones that already have a partner without the underscore.** That side
# does the computation and this only writes back.
#
# Written out forty-seven times by hand there are forty-seven places that can
# drift, and the only real difference is the argument count. Whether the name
# actually exists on `torch.Tensor` is confirmed by `tests/test_tensor_api.py`
# asking real torch — building a name that does not exist creates code that runs
# only against us.
_INPLACE_MORE = (
    "absolute", "acosh", "arccos", "arccosh", "arcsin", "arcsinh", "arctan",
    "arctanh", "asinh", "atanh", "deg2rad", "erfc", "exp2", "fix",
    "negative", "rad2deg", "sgn", "sinc",
)
# The ones taking one more argument. Only the argument count differs.
_INPLACE_BINARY = (
    "atan2", "copysign", "eq", "ge", "gt", "heaviside", "hypot", "le", "lt",
    "ne", "xlogy",
)
# The ones taking an axis or an index.
_INPLACE_ARGS = (
    # **`round` moved here from the nullary list**, because `round_` takes
    # `decimals` exactly as `round` does. Left nullary, the two spellings of one
    # operation took different arguments — which is what `div_` had against `div`
    # with `rounding_mode`, found in the same run.
    "round",
    # **`logit` moved here for the same reason, and it took a new instrument to
    # see.** `logit_` was nullary while `logit` takes `eps`, so `x.logit_(eps=1e-6)`
    # — which torch computes — stopped with *takes 1 positional argument but 2 were
    # given*. It stayed because the method's declared signature was `(*args, **kw)`
    # and every check that reads one went blind: the signature axis filed it as *no
    # python signature*, one of ninety-seven `Tensor` rows in that bucket.
    #
    # `relu_` is the opposite case and stays nullary: its partner's extra argument
    # is `inplace`, which **torch's method does not take either** (measured —
    # `TensorBase.relu_() takes no keyword arguments`). Two names in one list with
    # opposite answers, and only asking torch tells them apart.
    "logit",
    "cumprod", "cumsum", "index_add", "index_copy", "index_fill", "ldexp",
    "masked_fill", "scatter", "scatter_add", "squeeze", "swapaxes", "swapdims",
    "transpose", "tril", "triu", "unsqueeze",
)


def _make_inplace(name, arity="nullary"):
    # **It may be a module function or exist as a method only.** Some, like
    # `cumsum` and `squeeze`, exist on the tensor alone, so when the module does
    # not have it the method is called. Either way that does the computation and
    # this only writes back — two copies eventually diverge.
    fn = globals().get(name)
    # **When there is no module form, the method is the source and also the thing to
    # read.** The closure below is a bag, so `_forwards` had nothing to copy and
    # `transpose_` stayed `(self, *args, **kw)` while `Tensor.transpose(dim0, dim1)`
    # sat one attribute away, fully spelled. `reads` carries the readable one past
    # the wrapper.
    reads = fn
    if fn is None:
        reads = getattr(Tensor, name, None)

        def fn(t, *a, **k):
            return getattr(t, name)(*a, **k)

    if arity == "nullary":
        def method(self):
            return self._inplace(lambda: fn(self), name + "_")
        # **It takes nothing, and it was declaring otherwise.** `inspect.signature`
        # follows `__wrapped__`, and some of the partners carry one — `relu_` read
        # back as `(x, *args, **kw)`, promising to take anything while the body
        # takes `self` alone. Over-promising is the same failure as the
        # under-declaring below, pointing the other way, and both end in the axis's
        # *no python signature* bucket.
        method.__signature__ = _inspect.Signature(
            [_inspect.Parameter("self", _inspect.Parameter.POSITIONAL_OR_KEYWORD)])
    else:
        def method(self, *args, **kw):
            return self._inplace(lambda: fn(self, *args, **kw), name + "_")
        # **Only the forwarding branch may claim the forwarded list.** A nullary
        # method takes `(self)` and already reads as that; attaching its partner's
        # signature would declare arguments the method cannot accept, which is a
        # worse lie than the `(*args, **kw)` it replaces — `relu_` is exactly that
        # shape, and this line is why it is not.
        #
        # **The receiver comes off when the source is a method.** A module function
        # takes the tensor first and so does the wrapper, so the lists line up; a
        # method's first parameter is already `self` and copying it whole would
        # declare it twice.
        _forwards(method, reads, drop=reads is not fn)

    method.__name__ = name + "_"
    method.__doc__ = (f"`{name}` in place. `{name}` does the arithmetic and this "
                      "only writes the result back.")
    return method


def _forwards(method, fn, drop=False):
    """Say what the forwarder forwards, so `inspect.signature` can read it.

    **A `(*args, **kw)` wrapper is a signature nothing can compare.** `add_` takes
    exactly what `add` takes — that is what "in place" means here — and the wrapper
    was declaring neither. `tests/ts_signatures.py` files such a row as *no python
    signature*, which is the wording for **could not be measured**: ninety-seven
    `Tensor` rows sat in that bucket, and a bucket that large reads as a limit of
    Python rather than as a thing to fix.

    It is not Python's limit. torch's C methods genuinely have no signature and this
    one has one — it is sitting on the function next door.

    The receiver is renamed to `self`, because that is what the axis strips. Where
    the wrapped function is itself variadic (a handful reach a method by name and
    have nothing to copy) nothing is attached and the row stays honestly unread.
    """
    try:
        got = _inspect.signature(fn)
    except (TypeError, ValueError):
        return
    params = list(got.parameters.values())
    if not params or any(p.kind is p.VAR_POSITIONAL for p in params):
        return
    if drop:
        params = params[1:]
    first = params[0].replace(name="self", kind=_inspect.Parameter.POSITIONAL_OR_KEYWORD) \
        if not drop else _inspect.Parameter(
            "self", _inspect.Parameter.POSITIONAL_OR_KEYWORD)
    method.__signature__ = got.replace(
        parameters=[first] + (params[1:] if not drop else params))


# **They are attached at the end of this file.** Attached here, the functions
# defined below are not visible yet — `add` alone raised a `KeyError`.


# ---- the rest of the reshaping
#
# **In this file `abs`, `round` and `pow` are not Python's.** Tensor functions of
# the same names are defined above. Used on an integer they stop with
# `'int' object has no attribute 'abs'` — `diagflat` really did stop that way. So
# separate aliases for integers are kept.
_abs = _builtins.abs
#
# `expand` and `repeat` have similar names and do different jobs — confused, a
# quietly different shape comes out. `expand` stretches **only axes of size 1**
# and does not replicate the values (it is a view in torch). `repeat` concatenates
# the whole thing. The gradients differ correspondingly: expand folds the
# stretched axis back, and repeat sums the repeated pieces on top of each other.

def expand(t, *sizes):
    """Stretch axes of size 1. `-1` means "leave it alone".

    In torch it is a view sharing the storage and this replicates — shared views
    are an explicit limit of the core, and imitating it here alone would make
    that limit differ from place to place.
    """
    t = _wrap(t)
    want = sizes[0] if len(sizes) == 1 and isinstance(sizes[0], (list, tuple)) else sizes
    src = t.data.shape
    lead = len(want) - len(src)
    target = []
    for i, size in enumerate(want):
        if i < lead:
            target.append(int(size))
            continue
        have = src[i - lead]
        if size == -1:
            target.append(have)
        elif have != 1 and int(size) != have:
            raise RuntimeError(_like_torch(
                f"a dimension of size {have} cannot expand to {size} — expand only grows "
                "dimensions of size 1.",
                f"The expanded size of the tensor ({size}) must match the existing size "
                f"({have}) at non-singleton dimension {i - lead}"))
        else:
            target.append(int(size))
    target = tuple(target)
    lifted = t.data.reshape((1,) * lead + src)
    out = _np.broadcast_to(lifted, target)

    def back(g):
        return (_unbroadcast(_np.asarray(g), src),)

    return t._make(_np.ascontiguousarray(out), (t,), back, "ExpandBackward0")


def expand_as(t, other):
    return expand(t, *_wrap(other).data.shape)


# ── the ones torch offers under a **second name** ───────────────────────────
#
# `a + b` worked and `torch.add(a, b)` did not exist. Textbooks use both, and
# code that meets the missing side stops there. The operator already does the
# work, so there is nothing new to compute and **only the name** is needed — and
# without the name that code does not run.
#
# Some carry arguments the operator does not have, such as `alpha` and
# `rounding_mode`, so they cannot simply be aliases and get one layer of
# wrapping.

def add(input, other, alpha=1):
    """`a + alpha·b`. **The operator has no `alpha`** — so it is a function
    rather than an alias."""
    return _wrap(input) + (other if alpha == 1 else _wrap(other) * alpha)


def sub(input, other, alpha=1):
    return _wrap(input) - (other if alpha == 1 else _wrap(other) * alpha)


def mul(input, other):
    return _wrap(input) * other


def div(input, other, rounding_mode=None):
    """`rounding_mode` has three settings — none gives true division, and
    `'floor'` and `'trunc'` go to the integer side.

    **This is where the dtype diverges.** True division is always a float, and
    truncating or flooring **comes back as the input's dtype** (measured:
    `int64 / int64` with `trunc` gives int64). Matching the values and leaving
    the dtype a float diverges later, where indexing or `bincount` demands an
    integer — invisible to a value comparison.
    """
    left, right = _wrap(input), _wrap(other)
    out = left / right
    if rounding_mode is None:
        return out
    if rounding_mode == "floor":
        out = out.floor()
    elif rounding_mode == "trunc":
        out = out.trunc()
    else:
        raise RuntimeError(
            f"rounding_mode is one of None, 'floor', 'trunc': {rounding_mode!r}")
    kind = _dtype_result_type(left.data.dtype, _np.asarray(
        right.data if isinstance(right, Tensor) else right).dtype)
    return out if _np.dtype(kind).kind == "f" else out.type(kind)


def floor_divide(input, other):
    return div(input, other, rounding_mode="floor")


def remainder(input, other):                                 # noqa: A002
    """**The sign follows the divisor.** That is where it parts from `fmod`.

    torch's docstring says `remainder(divisor)` and torch takes `other` — measured,
    by calling it both ways. `fmod` next door was already on `(input, other)`, so the
    two neighbours spelled the same pair of arguments differently.
    """
    return _wrap(input) % other


def fmod(input, other):
    """**The sign follows the dividend.** C's `fmod` rule, and the opposite of
    `remainder`.

    **Written as `a - trunc(a / b) * b` it was right in value and wrong in type.**
    The division promotes, so two integer tensors came back `float32` where torch
    answers `int64`, and a boolean dividend stopped outright — `bool / int` is
    refused here, so the shape of the expression decided which inputs the function
    accepted. numpy's `fmod` is the same rule with none of that attached.

    The derivative is the one the expression implied: 1 for the dividend, and
    `-trunc(a / b)` for the divisor.
    """
    input = _wrap(input)
    return input._binary(other, _np.fmod,
                     lambda g, x, y: g,
                     lambda g, x, y: -g * _np.trunc(_np.divide(x, y)),
                     "FmodBackward0")


def rsub(a, b, alpha=1):
    return sub(b, a, alpha)


multiply = mul
divide = div
subtract = sub


def true_divide(input, other):                               # noqa: A002
    """**It cannot be `div`, because it is the one that has no `rounding_mode`.**

    `divide` is an alias and takes the argument; torch refuses it here —
    `true_divide() received an invalid combination of arguments` (measured) — and
    the name is why: true division is the thing a rounding mode would undo.

    Aliased, this library accepted `true_divide(x, y, rounding_mode="floor")` and
    floored. The answer was an integer where torch will not produce one at all, so
    code written here ran and the same code against torch raised — the divergence
    surfacing at the port rather than at the call.

    **The parameters were `(dividend, divisor)` and torch takes `(input, other)`.**
    Its docstring says `true_divide(value)`, which is a third name again and one
    torch also refuses; only calling it settles which of the three is real.
    """
    return div(input, other)

greater = gt
greater_equal = ge
less = lt
less_equal = le
not_equal = ne


def t(input):
    """A 2-D transpose. **1-D and below are left alone** — torch does that."""
    input = _wrap(input)
    _rank(input.data, (0, 1, 2),
          "t() expects a tensor with <= 2 dimensions, but self is {n}D")
    return input if len(input.data.shape) < 2 else input.transpose(0, 1)


def adjoint(input):
    """Swap the last two axes. Everything is real, so the conjugate is the
    identity."""
    input = _wrap(input)
    if input.data.ndim == 0:
        return input.reshape(())
    return input.transpose(-2, -1)


def moveaxis(input, source, destination):
    """The other name for `movedim`. **It cannot be an alias** — `movedim` is
    below this."""
    return movedim(input, source, destination)


concat = cat
concatenate = cat


def broadcast_to(x, *shape, **kw):
    """**Loose sizes, a tuple, or `size=`** — torch takes all three."""
    return expand(_wrap(x), *_loose(shape, kw, "size"))


def broadcast_tensors(*tensors):
    """Stretch them all to a common shape. The shape arithmetic is left to
    numpy."""
    ts = [_wrap(v) for v in tensors]
    shape = _np.broadcast_shapes(*[v.data.shape for v in ts])
    return tuple(broadcast_to(v, shape) for v in ts)


def broadcast_shapes(*shapes):
    return Size(_np.broadcast_shapes(*shapes))


def _stack_along(items, dim, lift):
    """The body the four stacking functions share. Only **which axis they join
    on and what rank they lift to** differs."""
    ts = [lift(_wrap(v)) for v in items]
    return cat(ts, dim)


def hstack(tensors):
    """1-D concatenates and above that it joins **along the columns** — that is
    how torch splits them."""
    ts = [_wrap(v) for v in tensors]
    dim = 0 if len(ts[0].data.shape) == 1 else 1
    return cat(ts, dim)


def vstack(tensors):
    return _stack_along(tensors, 0, atleast_2d)


def dstack(tensors):
    return _stack_along(tensors, 2, atleast_3d)


def column_stack(tensors):
    """1-D is **stood up as a single column** and joined. Where it parts from
    `hstack`."""
    ts = []
    for v in tensors:
        v = _wrap(v)
        ts.append(v.reshape(v.data.shape[0], 1) if len(v.data.shape) == 1 else v)
    return cat(ts, 1)


row_stack = vstack


# ── the ones where **the computation was missing**, not only the name ───────

def empty_like(t, dtype=None, requires_grad=False, *, out=None):
    """It borrows the shape alone. The values are undefined — torch is the
    same."""
    _no_out(out)
    # **The dtype is borrowed as well as the shape** — `zeros_like` and `ones_like`
    # next door already do, and this one did not, so `empty_like(int_tensor)` came
    # back `float32`. The values being undefined is what made it invisible: nothing
    # compares them, so only the type could have said anything and it was wrong.
    return _made(_np.zeros_like(_wrap(t).data), dtype, requires_grad)


def rand_like(t, dtype=None, requires_grad=False, *, out=None):
    _no_out(out)
    return _made(rand(*_wrap(t).data.shape).data, dtype, requires_grad)


def randn_like(t, dtype=None, requires_grad=False, *, out=None):
    _no_out(out)
    return _made(randn(*_wrap(t).data.shape).data, dtype, requires_grad)


def randint_like(t, low, high=None, dtype=None, requires_grad=False, *, out=None):
    _no_out(out)
    if high is None:
        low, high = 0, low
    return _made(randint(low, high, _wrap(t).data.shape).data, dtype, requires_grad)


def scalar_tensor(value, dtype=None, requires_grad=False):
    """A 0-D tensor from **a number**. torch refuses a tensor here outright, and
    this passed one straight through — `scalar_tensor(ones(2, 3))` returned a
    (2, 3), which is the one thing the name rules out."""
    if _np.ndim(value.data if isinstance(value, Tensor) else value) != 0:
        # **torch's message here is wider than torch's behaviour.** It says the
        # argument "must be Number, not Tensor" and then accepts a 0-D tensor
        # without complaint. Matching the behaviour rather than the sentence.
        raise TypeError("scalar_tensor(): argument 's' (position 1) must be "
                        f"Number, not {type(value).__name__}")
    if isinstance(value, Tensor):
        value = value.data
    return _made(_np.asarray(value, dtype=_DEFAULT_DTYPE), dtype, requires_grad)


def logspace(start, end, steps, base=10.0, dtype=None, requires_grad=False, *, out=None):
    """Evenly spaced as powers of `base`. `linspace` supplies the exponents."""
    _no_out(out)
    return _made((base ** _np.linspace(start, end, steps)).astype(_DEFAULT_DTYPE),
                 dtype, requires_grad)


def meshgrid(*tensors, indexing="ij"):
    """Build a grid. **Without `indexing`, torch warns and goes with `ij`.**

    `xy` has the first two axes swapped, so lumping the rule into one gets it
    right by accident in 2-D alone.
    """
    # A 0-D input is one point on that axis, which is what torch makes of it —
    # the grid comes back with a 1 in that position rather than an `IndexError`.
    ts = [_wrap(v) for v in tensors]
    ts = [v.reshape(1) if v.data.ndim == 0 else v for v in ts]
    if indexing not in ("ij", "xy"):
        raise RuntimeError(f"indexing must be 'ij' or 'xy': {indexing!r}")
    order = list(range(len(ts)))
    if indexing == "xy" and len(ts) >= 2:
        order[0], order[1] = order[1], order[0]
    sizes = [ts[i].data.shape[0] for i in order]
    out = []
    for place, i in enumerate(order):
        shape = [1] * len(ts)
        shape[place] = sizes[place]
        out.append(expand(ts[i].reshape(*shape), *sizes))
    if indexing == "xy" and len(ts) >= 2:
        out[0], out[1] = out[1], out[0]
    return tuple(out)


def lerp(start, end, weight):
    """`start + weight·(end − start)`. It connects two points evenly."""
    start, end = _wrap(start), _wrap(end)
    return start + (end - start) * weight


def nan_to_num(input, nan=0.0, posinf=None, neginf=None):
    """Turn NaN and infinities into finite numbers. **Given nothing, that
    dtype's extremes.**"""
    input = _wrap(input)
    d = input.data
    hi = _np.finfo(d.dtype).max if posinf is None else posinf
    lo = _np.finfo(d.dtype).min if neginf is None else neginf
    fixed = _np.nan_to_num(d, nan=nan, posinf=hi, neginf=lo)
    keep = _np.isfinite(d)
    return input._make(fixed.astype(d.dtype), (input,), lambda g: (g * keep,),
                   "NanToNumBackward0")


def isclose(input, other, rtol=1e-5, atol=1e-8, equal_nan=False):
    input, other = _wrap(input), _wrap(other)
    return Tensor(_np.isclose(input.data, other.data, rtol=rtol, atol=atol,
                              equal_nan=equal_nan))


def isreal(input):
    """Everything is real, so all of it is true. **A fact rather than a lie** —
    there is no complex here."""
    return Tensor(_np.ones(_wrap(input).data.shape, dtype=bool))


def isposinf(input):
    return Tensor(_np.isposinf(_wrap(input).data))


def isneginf(input):
    return Tensor(_np.isneginf(_wrap(input).data))


def isin(elements, test_elements):
    return Tensor(_np.isin(_wrap(elements).data, _wrap(test_elements).data))


def _nan_extreme(name, pick):
    """`fmax` and `fmin`. **They skip NaN** — `maximum` carries NaN out with
    it."""
    def call(input, other):
        input, other = _wrap(input), _wrap(other)
        out = pick(input.data, other.data)
        take_first = out == input.data

        def back(g):
            g = _np.asarray(g)
            return (g * take_first, g * ~take_first)

        return input._make(out, (input, other), back,
                           f"{name.capitalize()}Backward0")
    call.__name__ = name
    return call


fmax = _nan_extreme("fmax", _np.fmax)
fmin = _nan_extreme("fmin", _np.fmin)


def float_power(input, exponent):
    """A floating-point exponent. torch computes in double precision and there
    is only float32 here."""
    return _wrap(input) ** exponent


def logical_xor(input, other):
    return Tensor(_np.logical_xor(_wrap(input).data != 0, _wrap(other).data != 0))


def var_mean(t, dim=None, keepdim=False, *, out=None):
    """**Both at once.** Asking for one leaves the other free to be wrong
    uncaught.

    **Passed by keyword.** `var`'s positional order is
    `(dim, correction, keepdim)`, so passed positionally `keepdim` lands in
    `correction`'s slot — `True` becomes a correction of 1 and `False` a correction
    of 0, and the value is off by a factor of 12/11. It really was caught that way.

    (That middle seat was called `unbiased` when this was written and is torch's
    `correction` now. The hazard is the same one and the sentence had to move with
    the name, or it would describe a slot that is no longer there.)
    """
    _no_out(out)
    t = _wrap(t)
    return (t.var(dim=dim, keepdim=keepdim), t.mean(dim=dim, keepdim=keepdim))


def std_mean(t, dim=None, keepdim=False, *, out=None):
    _no_out(out)
    t = _wrap(t)
    return (t.std(dim=dim, keepdim=keepdim), t.mean(dim=dim, keepdim=keepdim))


def inner(input, other):
    """The inner product over the last axes. In 1-D it is the dot product."""
    input, other = _wrap(input), _wrap(other)
    return (input @ other.transpose(-2, -1) if len(input.data.shape) > 1
            else (input * other).sum())


def vdot(input, other):
    return (_wrap(input) * _wrap(other)).sum()


def kron(input, other):
    """The Kronecker product. One side is stretched, multiplied and folded back
    — no new kernel needed.

    **Any rank, and the 1-D line was already the general one.** It read
    `a.reshape(n, 1) * b.reshape(1, m)` and then refused everything above one axis;
    what that line does is interleave the two shapes and let broadcasting do the
    product, which is `kron`'s definition at every rank:

        out[(i, k), (j, l), …] = a[i, j, …] · b[k, l, …]

    So `a` becomes `(a0, 1, a1, 1, …)`, `b` becomes `(1, b0, 1, b1, …)`, and the
    result folds each pair back into one axis. **The shorter operand is padded at
    the front**, which is numpy's rule and torch's — measured against both on
    1×1, 2×2, 2×1, 1×2, 3×3, 2×3 and 0×2.

    Written this way it stays inside operations that already carry a gradient, so
    the backward is right by construction rather than by a second derivation. That
    is what the original line's *no new kernel needed* was for, and the refusal
    below it was hiding how far it already reached.
    """
    input, other = _wrap(input), _wrap(other)
    ash, bsh = input.data.shape, other.data.shape
    rank = max(len(ash), len(bsh))
    # numpy pads the shorter shape with leading 1s — `kron([[1, 2]], b3d)` is
    # `(1, 1, 2)` against `(2, 2, 2)` and comes out `(2, 2, 4)`.
    ash = (1,) * (rank - len(ash)) + tuple(ash)
    bsh = (1,) * (rank - len(bsh)) + tuple(bsh)
    a_spread, b_spread, folded = [], [], []
    for a_dim, b_dim in zip(ash, bsh):
        a_spread += [a_dim, 1]
        b_spread += [1, b_dim]
        folded.append(a_dim * b_dim)
    out = input.reshape(*a_spread) * other.reshape(*b_spread)
    return out.reshape(*folded) if folded else out


def cross(input, other, dim=-1):
    """The cross product. The axis has to have length 3."""
    input, other = _wrap(input), _wrap(other)
    rank = len(input.data.shape)
    axis = dim + rank if dim < 0 else dim
    if input.data.shape[axis] != 3:
        raise RuntimeError(f"cross needs dimension {dim} to have length 3")

    def part(t, i):
        return narrow(t, axis, i, 1)

    return cat([part(input, 1) * part(other, 2) - part(input, 2) * part(other, 1),
                part(input, 2) * part(other, 0) - part(input, 0) * part(other, 2),
                part(input, 0) * part(other, 1) - part(input, 1) * part(other, 0)], axis)


def block_diag(*tensors):
    """Blocks laid along the diagonal with zeros elsewhere. **The zeros are
    filler, so no gradient goes there.**"""
    ts = [atleast_2d(_wrap(v)) for v in tensors]
    rows = sum(v.data.shape[0] for v in ts)
    cols = sum(v.data.shape[1] for v in ts)
    lines, at = [], 0
    for v in ts:
        h, w = v.data.shape
        pieces = []
        if at:
            pieces.append(zeros(h, at))
        pieces.append(v)
        if cols - at - w:
            pieces.append(zeros(h, cols - at - w))
        lines.append(cat(pieces, 1) if len(pieces) > 1 else v)
        at += w
    return cat(lines, 0) if len(lines) > 1 else lines[0]


def repeat(t, *reps):
    """Repeat the whole thing and join. **The same job as `tile`** — torch simply
    keeps two names, so neither side is rewritten."""
    want = reps[0] if len(reps) == 1 and isinstance(reps[0], (list, tuple)) else reps
    want = tuple(int(r) for r in want)
    # **Fewer repeats than dimensions is refused, and this is where `repeat` and
    # `tile` part.** numpy's `tile` pads the count on the left, so `repeat(2)` on a
    # `(1, 2)` answered `[[1., 2., 1., 2.]]` — the tile of a shape the caller did not
    # give. torch refuses: a repeat count shorter than the tensor's rank is a call
    # that cannot mean one thing.
    #
    # So "the same job as `tile`" is true of the arithmetic and not of the contract,
    # which is the kind of sentence that stays true while what it is used to justify
    # stops being.
    if len(want) < _wrap(t).data.ndim:
        raise RuntimeError(
            "Number of dimensions of repeat dims can not be smaller than number of "
            f"dimensions of tensor: got {len(want)} for a tensor of "
            f"{_wrap(t).data.ndim}.")
    return tile(t, want)


def ravel(input):
    return _wrap(input).reshape(-1)


def swapaxes(input, axis0, axis1):
    """Swap two axes. The same as `transpose`; torch keeps one more name
    following numpy."""
    input = _wrap(input)
    order = list(range(input.data.ndim))
    rank = input.data.ndim
    order[axis0], order[axis1] = order[axis1 % rank], order[axis0 % rank]
    return input.permute(*order)


def swapdims(input, dim0, dim1):                             # noqa: A002
    """The same operation as `swapaxes`, **and not the same signature.**

    It was `swapdims = swapaxes`, which is the obvious way to write an alias and
    makes `x.swapdims(dim0=0, dim1=1)` raise: the seats were called `axis0` and
    `axis1`. torch names them apart on purpose — `swapaxes` follows numpy and
    `swapdims` follows `transpose`, and it accepts only its own spelling on each:

        x.swapaxes(axis0=0, axis1=1)   works        x.swapaxes(dim0=0, …)   raises
        x.swapdims(dim0=0, dim1=1)     works        x.swapdims(axis0=0, …)  raises

    So an alias by assignment is right about the values and wrong about the door.
    A peer found the same shape in `transpose`, whose seats were `d0`/`d1` here.
    """
    return swapaxes(input, dim0, dim1)


def select(input, dim, index):
    """Take one slice from an axis and **remove that axis.** Unlike a slice, the
    rank drops by one."""
    input = _wrap(input)
    axis = _pos_dim(input, dim)
    size = input.data.shape[axis]
    if not -size <= index < size:
        # torch says `IndexError` here and `RuntimeError` for `scatter`. numpy said
        # `ValueError` about a squeeze, which is a complaint about the *next* step.
        raise IndexError(
            f"select(): index {index} out of range for tensor of size "
            f"{list(input.data.shape)} at dimension {axis}")
    at = index + size if index < 0 else index
    return input[_slice_at(axis, at, at + 1)].squeeze(dim)


def diagonal(input, offset=0, dim1=0, dim2=1):
    """Take the diagonal. `offset` says how many cells up or down the diagonal
    sits.

    The backward puts values back at the positions taken — numpy's `diagonal`
    gives a read-only view, so rather than writing into it an empty array is
    built and filled.
    """
    input = _wrap(input)
    out = _np.diagonal(input.data, offset=offset, axis1=dim1, axis2=dim2)

    def back(g):
        # numpy's `diagonal` is a **read-only view** and cannot be written into.
        # An empty array is built and the coordinates computed directly.
        z = _np.zeros_like(input.data)
        n = out.shape[-1]
        rows = _np.arange(n) + max(0, -offset)
        cols = _np.arange(n) + max(0, offset)
        idx = [slice(None)] * z.ndim
        idx[dim1], idx[dim2] = rows, cols
        z[tuple(idx)] = _np.moveaxis(_np.asarray(g), -1, 0)
        return (z,)

    return input._make(_np.ascontiguousarray(out), (input,), back, "DiagonalBackward0")


def diagflat(input, offset=0):
    """Flatten and then build a diagonal matrix."""
    input = _wrap(input)
    flat = input.reshape(-1)
    n = flat.data.shape[0] + _abs(offset)
    out = _np.zeros((n, n), dtype=input.data.dtype)
    rows = _np.arange(flat.data.shape[0]) + max(0, -offset)
    cols = _np.arange(flat.data.shape[0]) + max(0, offset)
    out[rows, cols] = flat.data

    def back(g):
        return (_np.asarray(g)[rows, cols].reshape(flat.data.shape),)

    return flat._make(out, (flat,), back, "DiagflatBackward0")


def rot90(input, k=1, dims=(0, 1)):
    input = _wrap(input)
    dims = tuple(dims)
    return input._make(_np.ascontiguousarray(_np.rot90(input.data, k, dims)), (input,),
                   lambda g: (_np.ascontiguousarray(_np.rot90(_np.asarray(g), -k, dims)),),
                   "Rot90Backward0")


def unfold(input, dimension, size, step):        # noqa: A002
    """Turn a sliding window into a new axis. Where the windows overlap **the
    gradient accumulates in the backward** (measured: length 5 unfolded at size 3,
    stride 1 gives [1,2,3,2,1])."""
    input = _wrap(input)
    axis = dimension % input.data.ndim
    count = (input.data.shape[axis] - size) // step + 1
    starts = _np.arange(count) * step
    pieces = [_np.take(input.data, _np.arange(s, s + size), axis=axis) for s in starts]
    out = _np.stack([_np.moveaxis(p, axis, -1) for p in pieces], axis=axis)

    def back(g):
        z = _np.zeros_like(input.data)
        gg = _np.asarray(g)
        for i, s in enumerate(starts):
            piece = _np.moveaxis(_np.take(gg, i, axis=axis), -1, axis)
            idx = [slice(None)] * z.ndim
            idx[axis] = slice(s, s + size)
            z[tuple(idx)] += piece
        return (z,)

    return input._make(out, (input,), back, "UnfoldBackward0")


def hsplit(t, sections):
    """Split horizontally — axis 0 in 1-D and axis 1 otherwise.

    **`sections`, not `parts`.** torch takes that keyword on both the free
    function and the method, and a name the caller cannot write is a name that
    is not there."""
    t = _wrap(t)
    return chunk(t, sections, dim=0 if t.data.ndim == 1 else 1)


def vsplit(t, sections):
    return chunk(_wrap(t), sections, dim=0)


def dsplit(t, sections):
    return chunk(_wrap(t), sections, dim=2)


def fliplr(input):
    return flip(_wrap(input), (1,))


def flipud(input):
    return flip(_wrap(input), (0,))


def unflatten(input, dim, sizes):
    input = _wrap(input)
    # Without this, an out-of-range `dim` slid past the end of the shape list and the
    # refusal that followed was about the element count — a `RuntimeError` describing
    # a shape nobody asked for, where torch names the axis.
    _pos_dim(input, dim)
    shape = list(input.data.shape)
    shape[dim:dim + 1] = list(sizes)
    return input.reshape(*shape)


def atleast_1d(t):
    t = _wrap(t)
    return t if t.data.ndim >= 1 else t.reshape(1)


def atleast_2d(t):
    t = atleast_1d(_wrap(t))
    return t if t.data.ndim >= 2 else t.reshape(1, t.data.shape[0])


def atleast_3d(t):
    t = atleast_2d(_wrap(t))
    return t if t.data.ndim >= 3 else t.reshape(*(t.data.shape + (1,)))


def _negate(shifts):
    return -shifts if isinstance(shifts, int) else tuple(-s for s in shifts)


def index_select(input, dim, index):   # noqa: A002
    """**A negative `dim` counts from the end**, as everywhere else in torch.

    It did not here: `_index_at` built `dim` leading slices, and `range(-2)` is empty —
    so `index_select(x, -2, i)` selected along **axis 0**. On a tensor whose axes happen
    to be the same length that is not an error but a different answer, and on one whose
    are not it is an `IndexError` from a place that says nothing about the argument.

    Found by calling it, not by reading it: `uniform_temporal_subsample` picks frames
    along `-4` because the batch axes in front of them may or may not be there, and that
    is what a negative dimension is for.
    """
    input = _wrap(input)
    idx = index.data.astype(int) if isinstance(index, Tensor) else _np.asarray(index, dtype=int)
    return input[_index_at(dim, idx, len(input.data.shape))]


def _index_at(dim, idx, rank):
    axis = dim + rank if dim < 0 else dim
    if not 0 <= axis < rank:
        raise IndexError(
            f"Dimension out of range (expected to be in range of [{-rank}, "
            f"{rank - 1}], but got {dim})")
    return tuple(slice(None) for _ in range(axis)) + (idx,)


# **The function form** of the ones that existed as methods only. torch offers
# both `torch.matmul(a, b)` and `a @ b` and only the latter existed here — it
# surfaced while comparing against the sister library, and it was a gap against
# torch rather than only against the sister.

def matmul(input, other):
    return _wrap(input) @ _wrap(other)


def reshape(t, *shape):
    return _wrap(t).reshape(*shape)


def unsqueeze(input, dim):
    return _wrap(input).unsqueeze(dim)


def masked_fill(t, mask, value):
    t, m = _wrap(t), _wrap(mask)
    return where(m, Tensor(_np.asarray(value, dtype=t.data.dtype)), t)


def masked_select(input, mask):
    input = _wrap(input)
    m = mask.data.astype(bool) if isinstance(mask, Tensor) else _np.asarray(mask, dtype=bool)
    # A mask that does not broadcast came out as numpy's `IndexError` about a
    # boolean index. torch treats it as the shape mismatch it is.
    if m.shape != input.data.shape:
        try:
            _np.broadcast_shapes(m.shape, input.data.shape)
        except ValueError:
            raise RuntimeError(
                f"The size of tensor a {list(m.shape)} must match the size of "
                f"tensor b {list(input.data.shape)}") from None
    return input[m]


def gather(input, dim, index, sparse_grad=False):
    """Take the positions the index points at. Used in classification to pull
    out the probability of the correct class.

    `sparse_grad` asks torch for a sparse gradient. There is no sparse layout here,
    so it is **carried and refused** rather than left out — left out, torch's fourth
    position lands on nothing."""
    if sparse_grad:
        _unsupported("gather(sparse_grad=True)")
    input = _wrap(input)
    idx = index.data.astype(int) if isinstance(index, Tensor) else _np.asarray(index, dtype=int)
    # numpy's own complaint is an `IndexError`; torch says `RuntimeError` here and
    # `IndexError` for `select` two files over. Matching torch means matching that.
    _in_bounds(idx, input.data.shape[_pos_dim(input, dim)], dim)
    out = _np.take_along_axis(input.data, idx, axis=dim)
    shape = input.data.shape

    def back(g):
        z = _np.zeros(shape, dtype=_np.asarray(g).dtype)
        _np.put_along_axis(z, idx, _np.asarray(g), axis=dim)
        return (z,)

    return input._make(out, (input,), back, "GatherBackward0")


# ── the numeric family ──────────────────────────────────────────────────────

def cdist(a, b, p=2.0):
    """The distance between every pair. **One broadcast solves it** — no new
    kernel needed."""
    a, b = _wrap(a), _wrap(b)
    n, k = a.data.shape
    m = b.data.shape[0]
    diff = a.reshape(n, 1, k) - b.reshape(1, m, k)
    if p == 2.0:
        return (diff * diff).sum(dim=2).sqrt()
    return ((diff.abs() ** p).sum(dim=2)) ** (1.0 / p)


def cov(t, correction=1):
    """Covariance. Rows are variables and columns are observations — the axes
    are the reverse of numpy's, which makes it a confusing place.

    **Below rank 2 the answer is a scalar, not a 1×1 matrix.** One variable has
    one number for its spread and torch returns it with no axes at all; this
    reshaped to (1, -1), computed, and handed back the (1, 1) that fell out. The
    value was right the whole time, which is why it lasted: `.item()` reads the
    same from either, and so does anything that prints. It parts where the shape
    is used — a `stack` of per-variable variances came out (n, 1, 1), and
    broadcasting against it silently widens rather than raising.

    Found by a probe asking which **ranks** each function accepts. Rank was the
    question; the shape of the answer at that rank was the finding.
    """
    t = _wrap(t)
    _rank(t.data, (0, 1, 2),
          "cov(): expected input to have two or fewer dimensions but got an "
          "input with {n} dimensions")
    d = t.data
    if d.ndim <= 1:
        n = int(d.size)
        flat = t.reshape(1, n) if d.ndim else t.reshape(1, 1)
        centered = flat - flat.mean(dim=1, keepdim=True)
        pair = (centered @ centered.transpose(0, 1)) * (1.0 / max(1, n - correction))
        return pair.reshape(())
    n = d.shape[1]
    centered = t - t.mean(dim=1, keepdim=True)
    return (centered @ centered.transpose(0, 1)) * (1.0 / max(1, n - correction))


def corrcoef(input):
    """Covariance divided by the standard deviations. **The diagonal becomes 1** —
    that is the check."""
    c = cov(input)
    if c.data.ndim == 0:
        # One variable correlates with itself perfectly, and torch says so with a
        # scalar 1. Dividing by its own deviation would be 0/0 at zero variance.
        return c * 0 + 1
    d = c.data
    scale = _np.sqrt(_np.outer(_np.diag(d), _np.diag(d)))
    return c / Tensor(scale.astype(d.dtype))


def tensordot(a, b, dims=2):
    """Fold the named axes together and multiply. Only the form taking lists of
    axes is handled — that is torch's basic form."""
    a, b = _wrap(a), _wrap(b)
    if isinstance(dims, int):
        rank = len(a.data.shape)
        left = list(range(rank - dims, rank))
        right = list(range(dims))
    else:
        left, right = [list(v) for v in dims]
    # The folded axes are herded to the back and the front and one matmul
    # finishes it.
    a_keep = [i for i in range(len(a.data.shape)) if i not in left]
    b_keep = [i for i in range(len(b.data.shape)) if i not in right]
    a_shape = [a.data.shape[i] for i in a_keep]
    b_shape = [b.data.shape[i] for i in b_keep]
    inner = int(_np.prod([a.data.shape[i] for i in left], dtype=int))
    am = a.permute(*(a_keep + left)).reshape(int(_np.prod(a_shape, dtype=int)), inner)
    bm = b.permute(*(right + b_keep)).reshape(inner, int(_np.prod(b_shape, dtype=int)))
    return (am @ bm).reshape(*(a_shape + b_shape))


def trapezoid(y, x=None, dx=1.0, dim=-1):
    """Trapezoidal integration. The mean of each neighbouring pair times the
    spacing, summed."""
    y = _wrap(y)
    rank = len(y.data.shape)
    axis = dim + rank if dim < 0 else dim
    n = y.data.shape[axis]
    left = narrow(y, axis, 0, n - 1)
    right = narrow(y, axis, 1, n - 1)
    if x is None:
        return ((left + right) * (dx / 2.0)).sum(dim=axis)
    x = _wrap(x)
    step = narrow(x, axis, 1, n - 1) - narrow(x, axis, 0, n - 1)
    return ((left + right) * step * 0.5).sum(dim=axis)


def cumulative_trapezoid(y, x=None, dx=1.0, dim=-1):
    """The cumulative version of `trapezoid`. Its last value has to equal
    `trapezoid`."""
    y = _wrap(y)
    rank = len(y.data.shape)
    axis = dim + rank if dim < 0 else dim
    n = y.data.shape[axis]
    left = narrow(y, axis, 0, n - 1)
    right = narrow(y, axis, 1, n - 1)
    pieces = (left + right) * (dx / 2.0)
    if x is not None:
        x = _wrap(x)
        step = narrow(x, axis, 1, n - 1) - narrow(x, axis, 0, n - 1)
        pieces = (left + right) * step * 0.5
    return cumsum(pieces, axis)


# The Lanczos coefficients (g=7, n=9). **Not chosen by hand** — a well-known
# table, and dropping digits here makes the answer wrong by that much.
_LANCZOS = (0.99999999999980993, 676.5203681218851, -1259.1392167224028,
            771.32342877765313, -176.61502916214059, 12.507343278686905,
            -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7)


def _lgamma_np(d):
    """The log of the gamma function. **numpy has none** — the Lanczos
    approximation is written elementwise.

    `np.vectorize` is not used. This repository lost a factor of 20 to it in
    `gelu` and got the same values 20× faster by switching to elementwise numpy —
    the same mistake is not made twice.

    The negative side is folded through the reflection formula:
    `Γ(x)Γ(1−x) = π/sin(πx)`.
    """
    x = _np.asarray(d, dtype=_np.float64)
    neg = x < 0.5
    z = _np.where(neg, 1.0 - x, x) - 1.0
    acc = _np.full(z.shape, _LANCZOS[0])
    for i in range(1, len(_LANCZOS)):
        acc = acc + _LANCZOS[i] / (z + i)
    t = z + len(_LANCZOS) - 1.5
    out = (0.5 * _math.log(2 * _math.pi) + (z + 0.5) * _np.log(t) - t
           + _np.log(_np.abs(acc)))
    # Reflection: lgamma(x) = log(π/|sin(πx)|) − lgamma(1−x)
    flipped = _np.log(_np.pi / _np.abs(_np.sin(_np.pi * _np.where(neg, x, 0.5)))) - out
    got = _np.where(neg, flipped, out)
    # **The poles, which the reflection formula misses by a hair.** Γ has one at
    # every non-positive integer, so `lgamma` is `inf` there — but `sin(π·−2)` in
    # float64 is 2.4e-16 rather than 0, and `log(π/2.4e-16)` is 36.4. A finite
    # number where torch says infinity, at three of the first four integers below
    # zero, and plausible enough to pass through a loss unnoticed.
    #
    # Found by applying `lgamma_` twice: the first call sends 4.0 to 1.79 and −2.0
    # to 36.4, and it is the second that puts a value where torch has already
    # overflowed. **Nothing here calls anything twice.**
    pole = (x <= 0) & (x == _np.floor(x))
    return _np.where(_np.isnan(x), _np.nan,
                     _np.where(pole | ~_np.isfinite(x), _np.inf, got))


def _lifted(d, limit):
    """`(x, running)` for the recurrences below — **with the values that never
    terminate held out.**

    All three polygamma functions lift `x` by adding one until it clears a limit.
    `-inf + 1` is `-inf`, so `while any(x < limit)` **never ends** on a negative
    infinity: the process stops answering, with no error and nothing on screen.

    That is what it did. `x.digamma_()` twice was enough — the first call turns
    -2.0 into an infinity and the second hangs on it. **Found by calling every
    in-place operation twice**, which nothing in this repository did before: every
    check makes a fresh tensor, calls once, and compares.

    `nan` leaves on its own (`nan < limit` is false) and so does `+inf`. Only the
    negative side loops, and only where a value reached it.
    """
    x = _np.asarray(d, dtype=_np.float64)
    running = _np.isfinite(x)
    # A placeholder above the limit for the rest, so the loop sees nothing to lift.
    return _np.where(running, x, float(limit) + 1.0), running


def _polygamma0(d):
    """`digamma` — the logarithmic derivative of gamma. **The recurrence pushes
    it upwards and then the asymptotic expansion is used.**

    The asymptotic expansion does not hold at small x, so `ψ(x) = ψ(x+1) − 1/x`
    lifts it to 6 and above before computing. One of the few places numpy does
    not have, so it is written by hand.

    **The poles are torch's**, measured rather than derived: `ψ` has one at every
    non-positive integer, and torch answers `nan` at the negative ones and `-inf`
    at zero. The recurrence walks straight through them and reaches `-inf`
    everywhere, which is right at 0 and wrong at −1, −2, …
    """
    raw = _np.asarray(d, dtype=_np.float64)
    x, running = _lifted(raw, 6)
    out = _np.zeros_like(x)
    while _np.any(x < 6):
        small = x < 6
        out = _np.where(small, out - 1.0 / _np.where(small, x, 1.0), out)
        x = _np.where(small, x + 1.0, x)
    inv = 1.0 / x
    inv2 = inv * inv
    # The Stirling-family asymptotic expansion. Six terms exceed float32
    # precision for x ≥ 6.
    series = (_np.log(x) - 0.5 * inv
              - inv2 * (1.0 / 12 - inv2 * (1.0 / 120 - inv2 / 252)))
    got = out + series
    # The poles and the infinities, in torch's values. `-inf` is `nan` there and
    # `+inf` is `inf`; a negative integer is `nan` and zero alone is `-inf`.
    pole = running & (raw <= 0) & (raw == _np.floor(raw))
    got = _np.where(running, got, _np.where(raw > 0, _np.inf, _np.nan))
    return _np.where(pole, _np.where(raw == 0, -_np.inf, _np.nan), got)


def _polygamma1(d):
    """`trigamma` — the derivative of `digamma`. Pushed and expanded the same
    way, and **held out of the same non-terminating loop** — see `_lifted`.

    torch answers `nan` at `-inf`, `0` at `+inf`, and `inf` at zero. At a negative
    integer it gives a very large finite number (1.6e32 at −2) rather than an
    infinity, which is its own float64 recurrence running into the reciprocal of
    an epsilon rather than a value anybody derived. **Not reproduced**: matching it
    would mean copying an artefact, and the recurrence here reaches `inf`, which is
    the limit the function actually has. Written down rather than hidden.
    """
    raw = _np.asarray(d, dtype=_np.float64)
    x, running = _lifted(raw, 6)
    out = _np.zeros_like(x)
    while _np.any(x < 6):
        small = x < 6
        out = _np.where(small, out + 1.0 / _np.where(small, x, 1.0) ** 2, out)
        x = _np.where(small, x + 1.0, x)
    inv = 1.0 / x
    inv2 = inv * inv
    got = out + inv * (1.0 + 0.5 * inv
                       + inv2 * (1.0 / 6 - inv2 * (1.0 / 30 - inv2 / 42)))
    # Zero at `+inf` and no limit at `-inf` — torch's own answers for `n=1`.
    return _np.where(running, got, _np.where(raw > 0, 0.0, _np.nan))


def _polygamma_np(n, d):
    """`ψ^(n)` — the `n+1`-th derivative of log gamma. `n=0` is `digamma`.

    **The same method as `digamma`** — the recurrence pushes it upwards and the
    asymptotic expansion follows. Those two (`n=0` and `n=1`) were already written
    by hand, and this only generalises the rule to `n`. Three copies of the
    formula means the day comes when one is fixed, so sending small `n` through
    here too would be better, and the existing two are already held by the golden,
    so they stay.

    Recurrence: `ψ^(n)(x) = ψ^(n)(x+1) + (−1)^(n+1) n! / x^(n+1)`
    Asymptotic: `ψ^(n)(x) ≈ (−1)^(n+1) [ (n−1)!/xⁿ + n!/(2x^(n+1)) + Σ B_2k … ]`
    """
    raw = _np.asarray(d, dtype=_np.float64)
    x, running = _lifted(raw, 20)
    fact = float(_math.factorial(n))
    sign = 1.0 if (n + 1) % 2 == 0 else -1.0
    out = _np.zeros_like(x)
    while _np.any(x < 20):
        small = x < 20
        safe = _np.where(small, x, 1.0)
        out = _np.where(small, out + sign * fact / safe ** (n + 1), out)
        x = _np.where(small, x + 1.0, x)
    # The Bernoulli numbers B₂, B₄, B₆ and B₈. Four terms exceed float32
    # precision for x ≥ 20.
    series = _math.factorial(n - 1) / x ** n + fact / (2.0 * x ** (n + 1))
    for k, bern in enumerate((1 / 6, -1 / 30, 1 / 42, -1 / 30), start=1):
        series = series + (bern * _math.factorial(2 * k + n - 1)
                           / _math.factorial(2 * k) / x ** (2 * k + n))
    got = out + sign * series
    # **At the infinities this follows the function rather than torch, because
    # torch does not follow itself.** Measured: at `+inf` torch gives `inf` for
    # `n=0`, `0` for `n=1`, and `nan` for `n=2`; at `-inf` it gives `nan`, `nan`,
    # and `-inf`. The first two are the limits — ψ grows without bound, every
    # derivative of it decays to zero, and going the other way there is no limit at
    # all because the poles never stop. The `n=2` pair is its own recurrence
    # reaching a reciprocal, not a value anybody derived.
    #
    # So `n=2` differs from torch at exactly two inputs, and that is written here
    # rather than matched: copying an artefact makes the next reader believe it.
    away = _np.where(raw > 0, _np.inf if n == 0 else 0.0, _np.nan)
    return _np.where(running, got, _np.where(_np.isnan(raw), _np.nan, away))


def _igamma_np(a, x):
    """The regularised lower incomplete gamma `P(a, x) = γ(a,x)/Γ(a)`.

    **One formula cannot cover it.** Below `x < a+1` the series converges quickly
    and beyond it the continued fraction is faster — written the other way round
    the terms cancel and digits are lost. Splitting at the boundary is this
    function's standard form, and measured at small values alone that split is
    invisible.
    """
    av = _np.asarray(a, dtype=_np.float64)
    xv = _np.asarray(x, dtype=_np.float64)
    av, xv = _np.broadcast_arrays(av, xv)
    out = _np.zeros(av.shape, dtype=_np.float64)

    # **Everything below runs on stand-in values wherever the domain says nothing.**
    # `lnΓ(a)` used to be taken with `np.vectorize(math.lgamma)`, and `math.lgamma`
    # raises `ValueError: math domain error` at every non-positive integer. So a single
    # entry outside the domain **took the whole tensor down** — where torch puts a `nan`
    # in that one position and answers for the rest. A batch with one bad row ended the
    # training step rather than the row.
    #
    # Found by calling `igamma_` twice with a second operand: nothing here had ever
    # given the binary in-place family its other argument, so nothing had ever asked.
    #
    # Two things changed and **either one alone is enough** (measured by reverting each
    # separately — the check only goes red with both reverted): this mask, and the swap
    # to the vectorised `_lgamma_np`, which returns `inf` at the poles rather than
    # raising. Both are kept, because they answer different questions. The mask says
    # which entries have an answer; `_lgamma_np` says what happens to the ones that do
    # not. A later edit to either would otherwise silently restore the crash.
    ok = (av > 0) & _np.isfinite(av) & (xv >= 0) & _np.isfinite(xv)
    a0, x0 = av, xv                                    # kept for the edges, below
    av = _np.where(ok, av, 1.0)
    xv = _np.where(ok, xv, 1.0)
    lg = _lgamma_np(av)

    # ── the series (x < a+1): P = e^(−x + a·ln x − lnΓ(a)) · Σ xⁿ / (a(a+1)…(a+n))
    low = xv < av + 1.0
    if _np.any(low):
        ap = _np.where(low, av, 1.0)
        term = 1.0 / ap
        total = term.copy()
        for _ in _builtin_range(300):
            ap = ap + 1.0
            term = term * _np.where(low, xv, 0.0) / ap
            total = total + term
            if _np.all(_np.abs(term) <= _np.abs(total) * 1e-16):
                break
        out = _np.where(low, total * _np.exp(-xv + av * _np.log(
            _np.where(xv > 0, xv, 1.0)) - lg), out)

    # ── the continued fraction (x ≥ a+1): Q is computed and P = 1 − Q
    high = ~low
    if _np.any(high):
        tiny = 1e-300
        b = _np.where(high, xv + 1.0 - av, 1.0)
        c = 1.0 / tiny
        dd = 1.0 / b
        h = dd.copy()
        for i in _builtin_range(1, 300):
            an = -i * (i - av)
            b = b + 2.0
            dd = an * dd + b
            dd = _np.where(_np.abs(dd) < tiny, tiny, dd)
            c = b + an / c
            c = _np.where(_np.abs(c) < tiny, tiny, c)
            dd = 1.0 / dd
            delta = dd * c
            h = h * delta
            if _np.all(_np.abs(delta - 1.0) <= 1e-16):
                break
        q = _np.exp(-xv + av * _np.log(_np.where(xv > 0, xv, 1.0)) - lg) * h
        out = _np.where(high, 1.0 - q, out)

    # ── the edges, **read off torch rather than reasoned out**
    #
    # Thirty-six pairs from {−1, 0, ½, 2, ∞, nan}² were measured, and three of the rows
    # are hard to defend on their own: `P(nan, 0)` is 0, `P(nan, ∞)` is 1, `P(∞, nan)`
    # is 0. They look like an ordering artefact — a `x == 0 → 0` short-circuit standing
    # ahead of the nan check, with `nan < 0` quietly false on the way past.
    #
    # They are copied anyway, and that is the opposite of the call made for `polygamma`
    # a few lines up. The difference is that there torch answers the *same input* three
    # different ways depending on the order, so there was no single answer to copy;
    # here there is one, it is stable, and a port that improves on its subject silently
    # is a port whose differences you can no longer look up.
    got = _np.where(a0 == 0, 1.0, out)                     # P(0, x>0) = 1
    got = _np.where(_np.isnan(a0) | _np.isnan(x0), _np.nan, got)
    got = _np.where(_np.isposinf(a0), 0.0, got)            # all the mass is beyond x
    got = _np.where(_np.isposinf(x0), 1.0, got)            # the whole integral
    got = _np.where(x0 == 0, 0.0, got)                     # ∫ from 0 to 0
    got = _np.where(_np.isposinf(a0) & _np.isposinf(x0), _np.nan, got)
    got = _np.where((a0 == 0) & (x0 == 0), _np.nan, got)
    got = _np.where(a0 < 0, _np.nan, got)                  # Γ(a) has no branch here
    return _np.where(x0 < 0, _np.nan, got)


def _erfinv_np(d):
    """The inverse of `erf`. **There is no closed form** — a well-known rational
    approximation is used.

    The range is split in two. The middle (|x| ≤ 0.7) and the tails converge
    differently and one formula cannot cover both; forcing one formula pushes one
    side past the tolerance.
    """
    x = _np.asarray(d, dtype=_np.float64)
    a = (0.886226899, -1.645349621, 0.914624893, -0.140543331)
    b = (1.0, -2.118377725, 1.442710462, -0.329097515, 0.012229801)
    c = (-1.970840454, -1.624906493, 3.429567803, 1.641345311)
    e = (1.0, 3.543889200, 1.637067800)

    z = x * x
    mid = (x * (((a[3] * z + a[2]) * z + a[1]) * z + a[0])
           / ((((b[4] * z + b[3]) * z + b[2]) * z + b[1]) * z + b[0]))
    # The tails — computed after moving to `sqrt(-log((1−|x|)/2))`.
    safe = _np.clip(_np.abs(x), 0.0, 1 - 1e-12)
    w = _np.sqrt(-_np.log((1.0 - safe) / 2.0))
    tail = (_np.sign(x) * (((c[3] * w + c[2]) * w + c[1]) * w + c[0])
            / ((e[2] * w + e[1]) * w + e[0]))
    out = _np.where(_np.abs(x) <= 0.7, mid, tail)
    # One Newton step. The approximation alone sits around float32's tolerance,
    # so it is tightened once more.
    err = _erf64(out) - x
    out = out - err / (2.0 / _math.sqrt(_math.pi) * _np.exp(-out * out))
    # **Outside [−1, 1] there is no answer, and this returned one.** `erf` maps the
    # reals onto that interval, so its inverse is defined nowhere else; the `clip`
    # above keeps the tail formula finite and it goes on producing numbers — 4.7e21
    # at x = 1.5, where torch says `nan`.
    #
    # Nothing saw it because **no case applies `erfinv` twice**, and once is enough
    # to leave the interval: `erfinv(0.5)` is 0.48 and stays inside, but
    # `erfinv(1.5)` is where a second call lands. Every check here makes a fresh
    # tensor and calls once.
    return _np.where(
        _np.abs(x) <= 1.0,
        _np.where(_np.abs(x) == 1.0, _np.sign(x) * _np.inf, out),
        _np.nan)


def lgamma(input):
    """The log of the gamma function. **Its derivative is `digamma`**, so having
    one of the two is half of it."""
    input = _wrap(input)
    d = _float_in(input.data)
    out = _lgamma_np(d).astype(d.dtype)
    return input._make(out, (input,), lambda g: (g * _polygamma0(d).astype(d.dtype),),
                   "LgammaBackward0")


def digamma(input):
    """The logarithmic derivative of gamma. Its derivative is `trigamma`."""
    input = _wrap(input)
    d = _float_in(input.data)
    out = _polygamma0(d).astype(d.dtype)
    return input._make(out, (input,), lambda g: (g * _polygamma1(d).astype(d.dtype),),
                   "DigammaBackward0")


def erfinv(input):
    """The inverse of `erf`. Its derivative is `√π/2 · exp(erfinv(x)²)`."""
    input = _wrap(input)
    d = _float_in(input.data)
    out = _erfinv_np(d)
    grad = (_math.sqrt(_math.pi) / 2.0) * _np.exp(out * out)
    return input._make(out.astype(d.dtype), (input,),
                   lambda g: (g * grad.astype(d.dtype),), "ErfinvBackward0")


# ── the names that were left at top level ───────────────────────────────────
#
# What was left after measuring and splitting `tests/torch_gap.py`'s "to review"
# list. **Counting by name gets it wrong** — `fake_quantize_*` is named for
# quantisation and takes reals and produces reals, and `dequantize` is the
# identity over the reals. Only measuring showed they were not refusals.

def igamma(input, other):                                       # noqa: A002
    """The regularised lower incomplete gamma `P(a, x)`.

    **The gradient exists on the `x` side only** (measured). Differentiating with
    respect to `a` makes torch raise `NotImplementedError` — there is no closed
    form. It is followed: flowing one side only makes the other quietly 0, and
    then it surfaces only as training that does not happen.

        dP/dx = x^(a−1)·e^(−x) / Γ(a)
    """
    a, x = _wrap(input), _wrap(other)
    av = _np.asarray(a.data, dtype=_np.float64)
    xv = _np.asarray(x.data, dtype=_np.float64)
    out = _igamma_np(av, xv)
    # **The slope has to survive the same domain the value does.** `math.lgamma`
    # through `np.vectorize` raises at every non-positive integer, so building the
    # gradient used to end the call even for the entries that had an answer — and
    # `back` is built eagerly, so it happened whether or not anything asked for a
    # gradient. The vectorised `_lgamma_np` returns `inf` there instead, which drives
    # the slope to 0 at exactly the places the value is `nan`.
    lg = _lgamma_np(_np.where((av > 0) & _np.isfinite(av), av, 1.0))
    lg = _np.where((av > 0) & _np.isfinite(av), lg, _np.inf)
    with _np.errstate(over="ignore", invalid="ignore"):
        slope = _np.exp((av - 1.0) * _np.log(_np.where(xv > 0, xv, 1.0)) - xv - lg)
    slope = _np.where((xv > 0) & _np.isfinite(slope), slope, 0.0)

    def back(g):
        if a.requires_grad:
            raise NotImplementedError(_like_torch(
                "igamma is not differentiable in its first argument — there is no closed form.",
                "the derivative for 'igamma: input' is not implemented."))
        return (None, _unbroadcast(_np.asarray(g) * slope, x.data.shape)
                .astype(x.data.dtype))

    return a._make(out.astype(_DEFAULT_DTYPE), (a, x), back, "IgammaBackward0")


def igammac(input, other):                                      # noqa: A002
    """The upper side, `Q(a, x) = 1 − P(a, x)`. **The two sum to exactly 1**
    (measured)."""
    a, x = _wrap(input), _wrap(other)
    return ones(a.data.shape if a.data.ndim >= x.data.ndim else x.data.shape) \
        - igamma(a, x)


def polygamma(n, input):                                        # noqa: A002
    """`ψ^(n)` — the `n+1`-th derivative of log gamma. `n=0` is `digamma`.

    **`n` comes first** — the tensor is second. `torch.polygamma(1, x)` is the
    form, and reversing them puts an integer in the tensor's slot and stops
    loudly.

    Its derivative is `ψ^(n+1)` (measured: the gradient of `polygamma(1, x)` is
    `polygamma(2, x)`).
    """
    t = _wrap(input)
    k = int(n)
    if k < 0:
        raise RuntimeError(_like_torch(
            "polygamma needs n >= 0.",
            "polygamma(n, x) does not support negative n."))
    out = (_polygamma0(t.data) if k == 0 else _polygamma_np(k, t.data))
    nxt = _polygamma_np(k + 1, t.data)
    return t._make(out.astype(t.data.dtype), (t,),
                   lambda g: (_np.asarray(g) * nxt.astype(t.data.dtype),),
                   "PolygammaBackward0")


def constant_pad_nd(input, pad, value=0.0):                     # noqa: A002
    """The raw name for `F.pad(mode='constant')`. **From the last axis, in
    (before, after) order.**

    A place with one computation under two names — one side can be right alone, so
    **nothing is computed here** and it is forwarded to that side. This repository
    has been bitten by that shape three times (item 1 in the README's list).
    """
    return globals()["pad"](_wrap(input), list(pad), mode="constant",
                            value=value)


def _quantize_round(x, scale, zero_point, quant_min, quant_max):
    """The value after `clamp(round(x/scale) + zp, [qmin, qmax])` and back again.

    **No quantised dtype is needed** — it takes reals and produces reals. The name
    kept it counted as a refusal for a long time, and measured, torch takes a real
    tensor too.
    """
    q = _np.clip(_np.round(x / scale) + zero_point, quant_min, quant_max)
    return (q - zero_point) * scale


def fake_quantize_per_tensor_affine(input, scale, zero_point,   # noqa: A002
                                    quant_min, quant_max):
    """**Imitates quantisation over the reals.** Used to observe the quantisation
    error during training.

    **The gradient is 1 inside the range only** (measured: 0 where it is clipped).
    Rounding is a step and its derivative is 0 almost everywhere, and torch keeps
    that place as a straight-through estimator — otherwise no training reaches
    below this layer at all.
    """
    t = _wrap(input)
    s, z = float(scale), float(zero_point)
    x = _np.asarray(t.data, dtype=_np.float64)
    out = _quantize_round(x, s, z, quant_min, quant_max)
    inside = ((_np.round(x / s) + z >= quant_min)
              & (_np.round(x / s) + z <= quant_max)).astype(t.data.dtype)
    return t._make(out.astype(t.data.dtype), (t,),
                   lambda g: (_np.asarray(g) * inside,),
                   "FakeQuantizePerTensorAffineBackward0")


def fake_quantize_per_channel_affine(input, scale, zero_point,  # noqa: A002
                                     axis, quant_min, quant_max):
    """A different scale per cell. **The scale varies along one axis.**"""
    t = _wrap(input)
    s = _np.asarray(_wrap(scale).data, dtype=_np.float64)
    z = _np.asarray(_wrap(zero_point).data, dtype=_np.float64)
    shape = [1] * t.data.ndim
    shape[int(axis)] = -1
    s, z = s.reshape(shape), z.reshape(shape)
    x = _np.asarray(t.data, dtype=_np.float64)
    out = _quantize_round(x, s, z, quant_min, quant_max)
    inside = ((_np.round(x / s) + z >= quant_min)
              & (_np.round(x / s) + z <= quant_max)).astype(t.data.dtype)
    return t._make(out.astype(t.data.dtype), (t,),
                   lambda g: (_np.asarray(g) * inside,),
                   "FakeQuantizePerChannelAffineBackward0")


def dequantize(input):                                          # noqa: A002
    """A quantised tensor to reals. **Always the identity here.**

    This identity differs from "an identity that passes today" because there will
    **never** be a quantised dtype — that is already decided, so the only input
    this function can receive is real, and over the reals torch is the identity
    too (measured). Were that dtype to appear later this line would be wrong, and
    its not appearing is the decision.

    **It is not differentiable** — torch stops at `backward` (measured).

    **The identity is over the values, not over the type.** torch's `dequantize`
    always answers in a real dtype, so an integer tensor comes back `float32` there
    and came back `int64` here. "Always the identity" was true of the arithmetic and
    read as true of the whole function.
    """
    return Tensor(_float_in(_np.asarray(_wrap(input).data).copy()))


def resize_as_(input, the_template, memory_format=None):        # noqa: A002
    """Change **in place** to `the_template`'s shape.

    **The values in the added cells are undefined** — torch does not initialise
    them either (measured). So the golden asks about **the shape only.** Pinning
    the values would mount that implementation's accident as the
    specification.

    **The argument is `the_template`, and finding that out took a call.** This was
    `other`, which is what the rest of the family takes and what torch's own
    docstring for *this* one says — `resize_as_(tensor)`. Both are wrong:
    `x.resize_as_(the_template=y)` is the call torch accepts, and it refuses both
    `tensor=` and `other=` (measured, all three). A docstring disagreeing with the
    registration is not unusual here; a docstring disagreeing with it while the
    plausible guess is also wrong is why `test_torch_names.py` calls rather than
    reads.

    Its non-in-place twin `resize_as` really is `other` — the two do not share the
    name, which is the part nobody would guess.

    `memory_format` is torch's second seat, **carried and refused** the way
    `clone`'s is.
    """
    if memory_format is not None:
        _unsupported("Tensor.resize_as_(memory_format=…)")
    t, o = _wrap(input), _wrap(the_template)
    flat = _np.asarray(t.data).reshape(-1)
    want = int(_np.prod(o.data.shape)) if o.data.shape else 1
    grown = _np.zeros(want, dtype=t.data.dtype)
    keep = min(flat.size, want)
    grown[:keep] = flat[:keep]
    t.data = Tensor(grown.reshape(o.data.shape))
    return t


# ── the **writing** side of indexing. The opposite of the reading side
# (`gather`). ───────────────────────────────────────────────────────────────

def _as_index(index):
    return (index.data.astype(int) if isinstance(index, Tensor)
            else _np.asarray(index, dtype=int))


def _at_rank_0(t, call):
    """torch's answer for a 0-D input: **one element, and no axes on the way out.**

    Fourteen functions refused a 0-D tensor here, and not one of them had decided
    to. The refusal was numpy's — `AxisError: axis -1 is out of bounds for array of
    dimension 0`, `ValueError: Calling nonzero on 0d arrays is not allowed` — arriving
    through a call that never mentioned rank. **An unmade decision still behaves like
    a decision**, and this one behaved differently from torch while reading, in the
    traceback, as though the library had chosen something.

    torch is uniform about it: `sort`, `argsort`, `msort`, `mode`, `adjoint`,
    `poisson` and `normal` all answer for a scalar as though it were the one element
    it is, and give back a tensor with no axes. So the reshape goes out and comes
    back, rather than each site growing its own `if ndim == 0`.

    `nonzero` and `argwhere` are the exception and keep their own path — their answer
    is (count, rank), and at rank 0 that is a (1, 0) or a (0, 0). Reshaping to `()`
    would be wrong there, which is why this helper is called rather than applied.
    """
    out = call(t.reshape(1))
    if hasattr(out, "reshape"):
        return out.reshape(())
    # (values, indices): `_MinMax`, a namedtuple, or a plain pair. Rebuilt through
    # its own type so the caller keeps the field names it came with.
    parts = tuple(out) if isinstance(out, tuple) else (out.values, out.indices)
    return type(out)(*(o.reshape(()) for o in parts))


def _rank(data, ok, message):
    """The ranks torch accepts, or torch's own refusal.

    The two implementations can part on **which input ranks a function takes**,
    and neither the name axis nor the signature axis asks that: one asks whether
    the name is there, the other whether the parameter list matches. A function
    can pass both and still take a different set of shapes.

    Parting in this direction is the quiet one. Ten functions here **answered a
    question torch calls undefined** — `t` on a 3-D input transposed the first two
    axes and returned a tensor, `pdist` on a 3-D input returned a (1, 3), `trace`
    returned a batched diagonal sum that torch has no name for. Nothing raised.
    Code written against this library got a number, and the same code against
    torch got a `RuntimeError` — so the divergence surfaced not here but at the
    port, with the value already flowing.

    The mirror direction — refusing a rank torch accepts — is loud, and a peer hit
    it in borch.ts the same day (`nn.Linear` at 3-D). Loud is better for the person
    and **exactly as invisible to a checker**: `TORCH_REACHES_FURTHER_BY_POSITION`
    is 0 and cannot see it, because that number counts `TypeError` from a positional
    call and this is a `RuntimeError` about a shape.

    The message is torch's, word for word, because a caller who searches the text
    should land on torch's documentation rather than on ours.
    """
    n = len(data.shape)
    if n not in ok:
        raise RuntimeError(message.format(n=n, shape=list(data.shape)))


def _in_bounds(idx, size, dim, kind=RuntimeError):
    """Every index within `[-size, size)`, or torch's refusal.

    **numpy's own complaint about an index is an `IndexError` and torch's is not
    always one**, so a caller who wrote `except RuntimeError:` around a scatter —
    which is what torch's own message asks for — caught nothing. `select` and `put`
    are the two where torch says `IndexError` instead, so the class is an argument
    rather than a constant: torch is not consistent with itself here, and matching
    torch means matching that too.

    Enumerated rather than written down. Fourteen functions take an index; six
    parted, and no case list contained them because a case list is written by
    somebody who already suspects.
    """
    bad = _np.asarray(idx).reshape(-1)
    over = bad[(bad >= size) | (bad < -size)]
    if over.size:
        raise kind(
            f"index {int(over[0])} is out of bounds for dimension {dim} "
            f"with size {size}")


def scatter(input, dim, index, src, reduce=None):
    """**Overwrites** at the positions the indices point at. On a collision the
    last write survives.

    It parts from `scatter_add` at colliding indices only — measured with
    non-colliding indices the two functions look identical. So the golden asks
    with indices where 0 appears twice.

    **`reduce` is torch's deprecated overload and it is carried anyway.** torch
    warns that `scatter_reduce` replaces it and still answers `'add'` and
    `'multiply'`, refusing every other word — so a call written against torch
    runs here. Given, it accumulates *onto what is already there* rather than
    overwriting: `[[1,2]].scatter(0, [[0,0]], [[3,5]], reduce='add')` is
    `1+3+5`, not `3+5` (measured).

    **And it refuses to differentiate, because torch does** — `derivative for
    aten::scatter is not implemented` the moment `reduce` is given, for both
    operands. Computing a slope torch will not compute is the failure this
    repository keeps finding from the other side: the number comes out, nothing
    says it is ours alone, and it is wrong only in training.
    """
    input = _wrap(input)
    idx = _as_index(index)
    _in_bounds(idx, input.data.shape[dim], dim)
    out = input.data.copy()
    scalar = not isinstance(src, Tensor)
    values = (_np.full(idx.shape, src, dtype=input.data.dtype) if scalar
              else _wrap(src).data)
    if reduce is not None:
        if reduce not in ("add", "multiply"):
            raise RuntimeError(_like_torch(
                f"scatter's `reduce` takes 'add' or 'multiply'; got {reduce!r}. "
                "The wider set ('sum', 'prod', 'mean', 'amax', 'amin') belongs to "
                "`scatter_reduce`, which is the replacement torch points at.",
                "reduce argument must be either add or multiply."))
        # Colliding indices accumulate here, so `put_along_axis` is not it — the
        # same reason `scatter_add` reaches for `add.at`.
        grid = _np.indices(idx.shape)
        where = list(grid)
        where[dim] = idx
        (_np.add if reduce == "add" else _np.multiply).at(out, tuple(where), values)

        def refuse(g):
            # `RuntimeError`, not `NotImplementedError`, because that is what torch
            # raises — and `NotImplementedError` *is* a `RuntimeError`, so an
            # `except` clause would not have told them apart. Only the name does.
            raise RuntimeError(_like_torch(
                "scatter with `reduce` has no gradient — torch does not define one "
                "either, and a slope invented here would be wrong only in training.",
                "derivative for aten::scatter is not implemented"))

        parents = (input,) if scalar else (input, _wrap(src))
        return input._make(out, parents, refuse, "ScatterBackward0")
    _np.put_along_axis(out, idx, values, axis=dim)

    def back(g):
        g = _np.asarray(g)
        # An overwritten position is cut off from the original — a 0 goes
        # there.
        keep = _np.ones(input.data.shape, dtype=g.dtype)
        _np.put_along_axis(keep, idx, 0.0, axis=dim)
        got = (g * keep,)
        return got if scalar else got + (_np.take_along_axis(g, idx, axis=dim),)

    parents = (input,) if scalar else (input, _wrap(src))
    return input._make(out, parents, back, "ScatterBackward0")


def scatter_add(input, dim, index, src):
    """**Adds** at the positions the indices point at. Collisions accumulate —
    where it parts from `scatter`."""
    input, src = _wrap(input), _wrap(src)
    idx = _as_index(index)
    _in_bounds(idx, input.data.shape[dim], dim)
    out = input.data.copy()
    # `put_along_axis` overwrites and cannot be used. Accumulating colliding
    # indices properly needs `add.at` — that is the whole difference between this
    # function and `scatter`.
    grid = _np.indices(idx.shape)
    where = list(grid)
    where[dim] = idx
    _np.add.at(out, tuple(where), src.data)

    def back(g):
        g = _np.asarray(g)
        return (g, _np.take_along_axis(g, idx, axis=dim))

    return input._make(out, (input, src), back, "ScatterAddBackward0")


def index_add(input, dim, index, source, alpha=1):
    """Add to the **rows** the indices point at. Colliding indices
    accumulate."""
    input, source = _wrap(input), _wrap(source)
    idx = _as_index(index)
    _in_bounds(idx, input.data.shape[dim], dim)
    out = input.data.copy()
    _np.add.at(out, (slice(None),) * dim + (idx,), source.data * alpha)

    def back(g):
        g = _np.asarray(g)
        return (g, _np.take(g, idx, axis=dim) * alpha)

    return input._make(out, (input, source), back, "IndexAddBackward0")


def index_copy(input, dim, index, source):
    """**Replace** the rows the indices point at. No gradient goes to those
    rows."""
    input, source = _wrap(input), _wrap(source)
    idx = _as_index(index)
    out = input.data.copy()
    picker = (slice(None),) * dim + (idx,)
    out[picker] = source.data

    def back(g):
        g = _np.asarray(g)
        keep = _np.ones(input.data.shape, dtype=g.dtype)
        keep[picker] = 0.0
        return (g * keep, _np.take(g, idx, axis=dim))

    return input._make(out, (input, source), back, "IndexCopyBackward0")


def index_fill(t, dim, index, value):
    """Fill the rows the indices point at with one value."""
    t = _wrap(t)
    idx = _as_index(index)
    out = t.data.copy()
    picker = (slice(None),) * dim + (idx,)
    out[picker] = value

    def back(g):
        g = _np.asarray(g)
        keep = _np.ones(t.data.shape, dtype=g.dtype)
        keep[picker] = 0.0
        return (g * keep,)

    return t._make(out, (t,), back, "IndexFillBackward0")


def take(input, index):
    """Takes from **the flattened tensor** — it has no notion of an axis."""
    input = _wrap(input)
    idx = _as_index(index)
    shape = input.data.shape

    def back(g):
        z = _np.zeros(int(_np.prod(shape)), dtype=_np.asarray(g).dtype)
        _np.add.at(z, idx.reshape(-1), _np.asarray(g).reshape(-1))
        return (z.reshape(shape),)

    return input._make(_np.take(input.data, idx), (input,), back, "TakeBackward0")


def take_along_dim(input, indices, dim=None):
    """The same as `gather`. torch offers both names."""
    if dim is None:
        return take(input, indices)
    return gather(input, dim, indices)


def searchsorted(sorted_sequence, values, side=None, right=False, *, out=None):
    """Where a value would be inserted into something sorted. **Which side of a
    tie is decided by two arguments together.**

    torch takes the same thing under two names — the boolean `right` and the
    string `side`. Only `right` existed here and `side` went into `**kw` and was
    **quietly discarded.** `searchsorted(seq, v, side="right")` gave the left
    answer, and being off by one at a time it looks plausible.
    `bucketize(right=True)` was right from the start — this is the third place in
    this repository where **one computation had two names and only one of them
    was right.**

    Disagreeing, torch stops (measured). Give one, or give both meaning the same
    thing.
    """
    _no_out(out)
    if side is not None:
        if side not in ("left", "right"):
            raise RuntimeError(_like_torch(
                f"side must be 'left' or 'right' (got {side!r}).",
                f"torch.searchsorted(): side can only be 'left' or 'right' but "
                f"got {side}"))
        if right and side == "left":
            raise RuntimeError(_like_torch(
                "side and right contradict each other — give only one.",
                "torch.searchsorted(): side and right can't be set to opposites, "
                "got side of left while right was True"))
        right = side == "right"
    seq = _wrap(sorted_sequence).data
    want = _wrap(values).data
    return Tensor(_np.searchsorted(seq, want, side="right" if right else "left")
                  .astype(_np.int64))


def bucketize(values, boundaries, right=False, *, out=None):
    """**The argument order is reversed from `searchsorted`.** That is the whole
    difference between the two names."""
    _no_out(out)
    return searchsorted(boundaries, values, right=right)


def repeat_interleave(input, repeats, dim=None, *, output_size=None):
    """Stretch in place. The backward is **folding the stretched ones back per
    group.**

    `output_size` is torch telling the kernel the answer's length in advance so it
    need not read the repeats back off the GPU. It changes no value, and a wrong one
    is a caller error rather than a hint, so it is checked against what came out."""
    input = _wrap(input)
    out = _np.repeat(input.data, repeats, axis=dim)
    length = input.data.size if dim is None else input.data.shape[dim]
    counts = (_np.full(length, repeats, dtype=_np.int64) if isinstance(repeats, int)
              else _np.asarray(repeats, dtype=_np.int64))
    # **Given as `intp`.** numpy's default integer is C's `long`, which is int64
    # on 64-bit macOS and Linux and 32-bit on wasm32 (Pyodide), and `reduceat`
    # demands `intp` for its index array — unmatched it is a TypeError **in the
    # browser alone.** The third time this repository has been caught in the same
    # place, and a native check never produces it.
    starts = _np.concatenate(([0], _np.cumsum(counts)[:-1])).astype(_np.intp)
    axis = 0 if dim is None else dim

    def back(g):
        gg = _np.asarray(g)
        if dim is None:
            gg = gg.reshape(-1)
        return (_np.add.reduceat(gg, starts, axis=axis).reshape(input.data.shape),)

    return input._make(out, (input,), back, "RepeatInterleaveBackward0")


def tile(t, *reps, **kw):
    """Repeat the whole thing and join. The backward is **summing the repeated
    pieces on top of each other.**

    Each axis's output is (repeat count × original length), so splitting that axis
    in two and summing the repeat side is enough.
    """
    t = _wrap(t)
    # **Loose counts, a list, or `dims=`** — torch takes all three.
    reps_t = _loose(reps, kw, "dims")
    out = _np.tile(t.data, reps_t)
    src = t.data.shape
    nd = max(len(src), len(reps_t))
    src_p = (1,) * (nd - len(src)) + src
    reps_p = (1,) * (nd - len(reps_t)) + reps_t

    def back(g):
        split = []
        for r, s in zip(reps_p, src_p):
            split += [r, s]
        gg = _np.asarray(g).reshape(split).sum(axis=tuple(range(0, 2 * nd, 2)))
        return (gg.reshape(src),)

    return t._make(out, (t,), back, "TileBackward0")


def movedim(input, source, destination):
    input = _wrap(input)
    return input._make(_np.moveaxis(input.data, source, destination), (input,),
                   lambda g: (_np.moveaxis(_np.asarray(g), destination, source),),
                   "MovedimBackward0")


# ------------------------------------------------------- reductions (further)

def prod(t, dim=None, keepdim=False, dtype=None):
    t = _wrap(t)
    if dtype is not None:
        return prod(t.to(dtype), dim, keepdim).to(dtype)
    out = _np.prod(t.data, axis=dim, keepdims=bool(keepdim) and dim is not None)
    # The backward spreads back to the pre-fold shape — with `keepdim` the axis
    # is already alive and it stays as it is.
    wide = out if keepdim or dim is None else _np.expand_dims(out, dim)
    return t._make(out, (t,),
                   lambda g: (_np.asarray(g if keepdim or dim is None
                                          else _np.expand_dims(g, dim))
                              * wide / t.data,),
                   "ProdBackward0")


def median(t, dim=None, keepdim=False):
    """With an even element count torch gives **the smaller of the middle two.**
    numpy takes their mean — used as it is, a quietly different value comes out.

    **One NaN anywhere makes it NaN** (measured). `argsort` pushes NaN to the very
    end, so sorting and picking **skips the NaN** and produces a sound value —
    that is `nanmedian` and this is not. It was caught while adding a case that
    asks about the two side by side.
    """
    t = _wrap(t)
    # **Booleans are refused.** torch stops with
    # `"median_cpu" not implemented for 'Bool'` (measured). A hole in torch rather
    # than a rule, and handing back a value here means that code breaks against
    # real torch — being more permissive is still diverging.
    _refuses_bool(t.data, "median does not take booleans.",
                  '"median_cpu" not implemented for \'Bool\'',
                  kind=NotImplementedError)
    if dim is None:
        flat = t.data.reshape(-1)
        if _np.isnan(flat).any():
            return Tensor(_np.asarray(_np.nan, dtype=t.data.dtype))
        pick = int(_np.argsort(flat)[(flat.size - 1) // 2])

        # The gradient goes **evenly to every cell holding the same value** —
        # `max()`'s rule (measured: the median gradient of [1,5,5,5] is
        # [0, ⅓, ⅓, ⅓]).
        #
        # This used to say "it goes to the one chosen position", with the
        # reasoning written down — that shaking the other elements does not move
        # the answer. **True only when there is no tie.** On a tie those elements
        # hold the answer up together, so shaking one of them moves the answer
        # along with it. Measured on data with no ties the two rules give the same
        # answer and never part.
        share = (flat == flat[pick]).astype(_np.float64)
        share = (share / share.sum()).reshape(t.data.shape)

        def back(g):
            return (_np.asarray(g) * share,)

        return t._make(flat[pick], (t,), back, "MedianBackward0")

    order = _np.argsort(t.data, axis=dim)
    idx = (t.data.shape[dim] - 1) // 2
    take = _np.take(order, idx, axis=dim)
    at = _np.expand_dims(take, dim)
    picked = _np.take_along_axis(t.data, at, axis=dim).squeeze(dim)
    # A row holding a NaN is NaN throughout — the place the docstring above
    # names.
    sick = _np.isnan(t.data).any(axis=dim)
    if sick.any():
        picked = _np.where(sick, _np.asarray(_np.nan, dtype=picked.dtype), picked)

    def back_dim(g):
        z = _np.zeros_like(t.data)
        # **With `keepdim` the axis is already alive.** Spreading once more here
        # adds a rank and `put_along_axis` stops — a place caught on the shape
        # rather than on a value.
        wide = _np.asarray(g) if keepdim else _np.expand_dims(_np.asarray(g), dim)
        _np.put_along_axis(z, at, wide, axis=dim)
        return (z,)

    if keepdim:
        picked = _np.expand_dims(picked, dim)
        take = _np.expand_dims(take, dim)

    return _MinMax(t._make(picked, (t,), back_dim, "MedianBackward0"), Tensor(take))


def norm(input, p=2, dim=None, keepdim=False, dtype=None):  # noqa: A002
    """torch's order — `keepdim` between `dim` and `dtype`.

    **It was missing from the middle**, so `x.norm(2, 1, True)` set the dtype to
    `True` and folded the axis away; every later argument was one seat out. It
    reaches all six branches of `p` rather than the two common ones, because a
    reduction that keeps its rank for `p=2` and drops it for `p=inf` is a shape that
    changes with a hyperparameter.
    """
    input = _wrap(input)
    # **The dtype is converted before computing** — torch does that (measured:
    # a float32 asked with `dtype=float64` answers in float64). Converted after
    # computing the precision has already been cut and the value differs.
    if dtype is not None:
        input = _wrap(input.data.astype(_np_of(_requested_dtype(dtype))))
    _needs_float(
        input.data,
        "A norm exists over the reals only — a square root does not fit in an "
        "integer cell. Call `.float()` first.",
        "linalg.vector_norm: Expected a floating point or complex tensor as input")
    if p == 1:
        return input.abs().sum(dim=dim, keepdim=keepdim)
    if p == 2:
        return (input * input).sum(dim=dim, keepdim=keepdim) ** 0.5
    # **`p` arrives as more than 1 and 2.** Everything else was counted as 2 for
    # a long time — `dist(a, b, 3)` handed back the L2 and the value was plausible
    # (the same order of magnitude), so it went unseen. `inf` is the largest
    # absolute value, `-inf` the smallest, and `0` the count of non-zeros
    # (measured).
    if p == float("inf"):
        return (input.abs().max(dim=dim) if dim is None
                else input.abs().amax(dim=dim, keepdim=keepdim))
    if p == -float("inf"):
        return (input.abs().min(dim=dim) if dim is None
                else input.abs().amin(dim=dim, keepdim=keepdim))
    if p == 0:
        # **`input * 0` is added to carry the graph through.** Counting is a step, so
        # the derivative is 0, and 0 is the right answer rather than "absent".
        # Without it, `norm(0).backward()` stops and torch does not — it keeps a
        # `grad_fn`, never reaches a leaf, and `grad` stays None (measured). The
        # only difference is that 0 accumulates here and None stays there, and
        # added into a loss the effect on training is the same. **Stopping is the
        # furthest of the three from right.**
        return ((input != 0).float().sum(dim=dim, keepdim=keepdim)
                + (input * 0).sum(dim=dim, keepdim=keepdim))
    return (input.abs() ** float(p)).sum(dim=dim, keepdim=keepdim) ** (1.0 / float(p))


# ---- the rest of the reductions
#
# `amax` and `amin` give the same values as `max` and `min` and **no indices.**
# That is not the only difference — on a tie they **split the gradient evenly**
# (measured: the amax gradient of [1,3,3,2] is [0,.5,.5,0]). Piled onto one
# position the value checks pass and only the training diverges subtly.

def _spread_max(t, dim, keepdim, take, name):
    """Reduce by maximum (or minimum), **splitting the gradient evenly across a
    tie.**"""
    t = _wrap(t)
    out = take(t.data, axis=dim, keepdims=True)
    hit = (t.data == out).astype(t.data.dtype)
    share = hit / hit.sum(axis=dim, keepdims=True)
    final = out if keepdim else (out if dim is None else _np.squeeze(out, axis=dim))
    if dim is None:
        final = out.reshape(())

    def back(g):
        gg = _np.asarray(g)
        if dim is not None and not keepdim:
            gg = _np.expand_dims(gg, dim)
        return (gg * share,)

    return t._make(final, (t,), back, name)


def amax(input, dim=None, keepdim=False):
    return _spread_max(input, dim, keepdim, _np.max, "AmaxBackward0")


def amin(input, dim=None, keepdim=False):
    return _spread_max(input, dim, keepdim, _np.min, "AminBackward0")


def aminmax(input, dim=None, keepdim=False):
    return _MinMax(amin(input, dim, keepdim), amax(input, dim, keepdim))


def _nan_mask(t):
    """The tensor with the nan positions replaced by 0, and where the nans
    were."""
    bad = _np.isnan(t.data)
    return _np.where(bad, 0.0, t.data), bad


def nansum(t, dim=None, keepdim=False, dtype=None):
    """A sum that **counts nan as 0.** No gradient goes to those positions
    either."""
    t = _wrap(t)
    if dtype is not None:
        return nansum(t.to(dtype), dim, keepdim).to(dtype)
    # **The dtype is kept.** `sum`'s rule, and `_nan_mask` was promoting to a
    # float in order to handle nan, so integers and booleans came out float64.
    # Integers hold no nan, so they are computed as they are.
    if t.data.dtype.kind not in "fc":
        return t.sum(dim=dim, keepdim=keepdim)
    clean, bad = _nan_mask(t)
    return t._make(clean.sum(axis=dim, keepdims=keepdim), (t,),
                   lambda g: (_np.where(bad, 0.0, _expand_reduced(g, t.data.shape, dim, keepdim)),),
                   "NansumBackward0")


def nanmean(input, dim=None, keepdim=False, dtype=None):
    """A mean taken **excluding** nan — the count excludes them too.

    **`dtype=` does not lift the integer refusal.** `mean` lifts it and this does
    not (measured: `torch.tensor([3,1,4]).nanmean(dtype=torch.float32)` stops). An
    asymmetry in torch rather than a rule, and diverging towards the permissive
    side is still diverging, so it is followed.
    """
    input = _wrap(input)
    _needs_float(
        input.data,
        "nanmean exists over the reals only. Call `.float()` first.",
        "nanmean(): expected input to have floating point or complex dtype")
    clean, bad = _nan_mask(input)
    count = (~bad).sum(axis=dim, keepdims=keepdim)
    total = clean.sum(axis=dim, keepdims=keepdim)
    # **numpy's promotion rule must not be left alone.** Dividing a float32 by an
    # int64, numpy promotes to float64 and torch gives float32 — the values match
    # and the dtype diverges.
    out = total / count.astype(total.dtype)

    def back(g):
        gg = _expand_reduced(g, input.data.shape, dim, keepdim)
        n = (_expand_reduced(count, input.data.shape, dim, keepdim)
             if dim is not None else count)
        return (_np.where(bad, 0.0, gg / n),)

    got = input._make(out, (input,), back, "NanmeanBackward0")
    return got if dtype is None else got.to(dtype)


def _expand_reduced(g, shape, dim, keepdim):
    """Revive the axis a reduction folded so that it can spread back over the
    original shape."""
    gg = _np.asarray(g)
    if dim is None:
        return _np.broadcast_to(gg, shape)
    if not keepdim:
        gg = _np.expand_dims(gg, dim)
    return _np.broadcast_to(gg, shape)


def logsumexp(input, dim, keepdim=False):                    # noqa: A002
    """`log(sum(exp(x)))` computed **without overflow** — the maximum is
    subtracted before summing.

    **`dim` is required, and it had a default of `None`.** torch refuses both
    `t.logsumexp()` and `t.logsumexp(dim=None)` — the first for a missing argument
    and the second because `dim` must be a tuple of ints (measured). This answered
    the whole-tensor value for both.

    Accepting what the authority refuses misleads exactly as much as accepting an
    argument and ignoring it: code written here runs and the same line against torch
    stops, with the divergence surfacing at the port rather than at the call. It is
    the rule `true_divide` records for `rounding_mode`, the other way round.

    Found by a sweep asking whether each reduction carries **both** of torch's
    overloads, after a peer found borch.ts's `sum` carrying one of two. This is the
    only row of twenty-one that parted, and it parted in our favour.
    """
    input = _wrap(input)                                     # noqa: A001
    # **It takes integers and booleans too and produces float32** (measured).
    # Left alone two places are wrong — numpy promotes integers to float64, and
    # booleans refuse `-` and stop at the subtraction below.
    if input.data.dtype.kind not in "fc":
        input = _wrap(input.data.astype(_DEFAULT_DTYPE))
    big = _np.max(input.data, axis=dim, keepdims=True)
    shifted = _np.exp(input.data - big)
    total = shifted.sum(axis=dim, keepdims=True)
    out = _np.log(total) + big
    soft = shifted / total
    if not keepdim:
        out = out.reshape(()) if dim is None else _np.squeeze(out, axis=dim)
    return input._make(out, (input,),
                   lambda g: (_expand_reduced(g, input.data.shape, dim, keepdim) * soft,),
                   "LogsumexpBackward0")


def _cum_extreme(t, dim, pick, name):
    """A running maximum or minimum. It gives the values and **the indices** —
    torch's shape."""
    t = _wrap(t)
    idx = pick(t.data, axis=dim)
    out = _np.take_along_axis(t.data, idx, axis=dim)

    def back(g):
        # The gradient goes **only to the chosen positions.** A position chosen
        # several times accumulates that many times.
        z = _np.zeros_like(t.data)
        _np.add.at(z, _index_for(idx, dim, t.data.ndim), _np.asarray(g))
        return (z,)

    return _MinMax(t._make(out, (t,), back, name), Tensor(idx.astype(_np.int64)))


def _index_for(idx, dim, ndim):
    """The index to give `np.add.at` — coordinates per axis, handed over as a
    tuple."""
    grid = _np.indices(idx.shape)
    return tuple(idx if a == dim % ndim else grid[a] for a in range(ndim))


def _running_idx(better):
    """Produce **the indices** of the running maximum or minimum. It walks the
    axis one cell at a time.

    A vectorised version was attempted and abandoned. On a tie torch gives **the
    later position** (measured: the cummax indices of [1,3,3,2] are [0,1,2,2] — at
    i=2 it is 2, not 1). That rule fits in the one character `>=`, and forcing a
    vectorisation hides that character somewhere it cannot be seen.
    """
    def make(d, axis):
        moved = _np.moveaxis(d, axis, 0)
        idx = _np.zeros(moved.shape, dtype=_np.intp)
        best = moved[0].copy()
        for i in range(1, moved.shape[0]):
            take = better(moved[i], best)
            idx[i] = _np.where(take, i, idx[i - 1])
            best = _np.where(take, moved[i], best)
        return _np.moveaxis(idx, 0, axis)
    return make


def cummax(input, dim):
    return _cum_extreme(input, dim, _running_idx(lambda cur, best: cur >= best),
                        "CummaxBackward0")


def cummin(input, dim):
    return _cum_extreme(input, dim, _running_idx(lambda cur, best: cur <= best),
                        "CumminBackward0")


def kthvalue(input, k, dim=-1, keepdim=False):
    """The **k-th smallest** value. torch counts from 1."""
    input = _wrap(input)
    size = input.data.shape[dim]
    if not 1 <= k <= size:
        raise RuntimeError(
            f"kthvalue(): selected number k out of range for dimension {dim}")
    order = _np.argsort(input.data, axis=dim, kind="stable")
    at = _np.take(order, k - 1, axis=dim)
    at_e = _np.expand_dims(at, dim)
    out = _np.take_along_axis(input.data, at_e, axis=dim)
    if not keepdim:
        out = _np.squeeze(out, axis=dim)

    def back(g):
        z = _np.zeros_like(input.data)
        gg = _np.asarray(g)
        _np.put_along_axis(z, at_e, gg if keepdim else _np.expand_dims(gg, dim), axis=dim)
        return (z,)

    return _MinMax(input._make(out, (input,), back, "KthvalueBackward0"),
                   Tensor(at.astype(_np.int64)))


def msort(input):
    """Sort **along the first axis.** The same as the values side of
    `sort(dim=0)`."""
    input = _wrap(input)
    if input.data.ndim == 0:
        return input.reshape(())
    return sort(input, dim=0).values


def diff(input, n=1, dim=-1, prepend=None, append=None):
    """The difference between neighbours. `x[1:] - x[:-1]`, n times.

    **Built from slicing** — slicing already carries the graph, so there is no
    backward to write.

    **For booleans it is XOR rather than subtraction.** torch gives `[T, T]` from
    `[T, F, T]` (measured), which asks whether the neighbours differ. Here `-`
    refused booleans and it stopped outright — being stingier is still
    diverging.

    **Appending at either end keeps the length.** `prepend` and `append`
    concatenate **before** the difference is taken, so adding one makes the
    result the same length as the input — used in time series so the first cell
    is not lost.
    """
    out = _wrap(input)
    # **The axis is checked here rather than borrowed from a slice.** Slicing an
    # axis that does not exist is not an error in Python — the answer came back as
    # though `dim` were the last one, so `diff(x, dim=7)` on a 2-D tensor gave a
    # plausible tensor and no complaint.
    _pos_dim(out, dim)
    if prepend is not None or append is not None:
        parts = ([_wrap(prepend)] if prepend is not None else []) + [out] \
            + ([_wrap(append)] if append is not None else [])
        out = cat(parts, dim=dim)
    if out.data.dtype.kind == "b":
        data = out.data
        for _ in range(n):
            data = _np.logical_xor(data.take(_np.arange(1, data.shape[dim]), axis=dim),
                                   data.take(_np.arange(0, data.shape[dim] - 1), axis=dim))
        return Tensor(data)
    axis = dim % out.data.ndim
    for _ in range(n):
        length = out.data.shape[axis]
        out = out[_slice_at(axis, 1, length)] - out[_slice_at(axis, 0, length - 1)]
    return out


def dist(input, other, p=2):
    """The distance between two tensors — `norm(a - b, p)`."""
    return norm(_wrap(input) - _wrap(other), p=p)


_INTERPOLATIONS = ("linear", "lower", "higher", "midpoint", "nearest")


def _interpolation(how):
    """torch's `interpolation`, checked. **It changes the answer** — on `[1,2,3,4]`
    the 0.3 quantile is 1.9 under `linear`, 1.0 under `lower` and 1.5 under
    `midpoint`, all of them plausible numbers with no way to tell from the value
    which rule produced it."""
    if how not in _INTERPOLATIONS:
        raise RuntimeError(
            f"quantile() interpolation must be one of {_INTERPOLATIONS}, got {how!r}")
    return how


def quantile(input, q, dim=None, keepdim=False, *, interpolation="linear"):
    """Quantiles. torch's default is **linear interpolation**, the same as
    numpy's — and the other four rules were not reachable at all until this
    argument existed."""
    input = _wrap(input)
    _needs_float(
        input.data,
        "Quantiles exist over the reals only — the interpolation does not fit "
        "in an integer cell.",
        "quantile() input tensor must be either float or double dtype")
    qq = q.data if isinstance(q, Tensor) else _np.asarray(q, dtype=input.data.dtype)
    out = _np.quantile(input.data, qq, axis=dim, keepdims=keepdim,
                       method=_interpolation(interpolation))

    # **The gradient splits across the two positions used in the interpolation** —
    # landing exactly gives one position (measured: the quantile(0.3) gradient of
    # [3,5,5,1,5] is [0.8, 0.2, 0, 0, 0]).
    #
    # A different rule from `median`'s. `median` splits across **every cell
    # holding the same value** and `quantile` splits across **the sorted
    # positions** — on [1,5,5,5], median gives ⅓ to each of the three 5s and
    # quantile(0.5) gives ½ to the first two. Measured on data with no ties the
    # two give the same answer and this split is invisible.
    #
    # Here too there was only a bare `Tensor(...)` and the graph was quietly
    # cut.
    data = _np.asarray(input.data, dtype=_np.float64)
    lines = data.reshape(1, -1) if dim is None else \
        _np.moveaxis(data, dim, -1).reshape(-1, data.shape[dim])
    order = _np.argsort(lines, axis=-1, kind="stable")
    n = lines.shape[-1]
    rows = _np.arange(lines.shape[0])
    # One weight array per q. A scalar q gives one.
    qs = _np.atleast_1d(_np.asarray(qq, dtype=_np.float64))
    sheets = _np.zeros((qs.size,) + lines.shape, dtype=_np.float64)
    for k, one in enumerate(qs):
        pos = float(one) * (n - 1)
        lo, hi = int(_np.floor(pos)), int(_np.ceil(pos))
        # **The split follows the rule that produced the value.** Written for
        # `linear` alone, the gradient kept splitting by the fraction while the
        # forward had stopped interpolating — the two would have described
        # different functions, with only the backward wrong and nothing to show it.
        frac = {"linear": pos - lo, "lower": 0.0, "higher": 1.0, "midpoint": 0.5,
                "nearest": float(round(pos) - lo)}[interpolation]
        _np.add.at(sheets[k], (rows, order[:, lo]), 1.0 - frac)
        _np.add.at(sheets[k], (rows, order[:, hi]), frac)

    def back(g):
        gg = _np.asarray(g, dtype=_np.float64)
        # With a vector q the result's leading axis is q. Each array carries its
        # share and they are summed.
        parts = gg.reshape(qs.size, -1) if _np.ndim(qq) else gg.reshape(1, -1)
        total = (sheets * parts[:, :, None]).sum(axis=0)
        if dim is None:
            return (total.reshape(input.data.shape),)
        moved = data.shape[:dim] + data.shape[dim + 1:] + (n,)
        return (_np.moveaxis(total.reshape(moved), -1, dim),)

    return input._make(_np.asarray(out, dtype=input.data.dtype), (input,), back,
                   "QuantileBackward0")


def nanquantile(input, q, dim=None, keepdim=False, *, interpolation="linear"):
    input = _wrap(input)
    qq = q.data if isinstance(q, Tensor) else _np.asarray(q, dtype=input.data.dtype)
    out = _np.nanquantile(input.data, qq, axis=dim, keepdims=keepdim,
                          method=_interpolation(interpolation))
    return Tensor(_np.asarray(out, dtype=input.data.dtype))


def nonzero(input, as_tuple=False):
    """The coordinates of the non-zero positions. **The shape depends on the
    values** — which is why there is no gradient.

    `as_tuple` turns the (count, rank) table into **one 1-D tensor per axis**,
    which is what indexing wants: `x[nonzero(m, as_tuple=True)]` works and
    `x[nonzero(m)]` does not.

    Both this and `where` are read by `inspect` as **"no signature found for
    builtin"**, so the signature axis never compared either one. That bucket does
    not mean *these agree*; it means *nothing was asked*. Two absences sat inside
    it — this argument, and `where`'s one-argument form — while the axis reported
    0 keyword-only absences, and both were found by a probe built for a different
    question entirely.
    """
    data = _wrap(input).data
    if data.ndim == 0:
        # **torch answers the two forms with different ranks here.** The table
        # form gives (count, 0) — no columns, because there are no axes to name a
        # position along — and the tuple form gives a 1-tuple, as though the
        # scalar were 1-D. Neither is derivable from the other, so both are
        # written out rather than one being folded into the other.
        hit = _np.asarray([0] if data != 0 else [], dtype=_np.int64)
        if as_tuple:
            return (Tensor(hit),)
        return Tensor(_np.zeros((hit.size, 0), dtype=_np.int64))
    idx = _np.nonzero(data)
    if as_tuple:
        return tuple(Tensor(i.astype(_np.int64)) for i in idx)
    return Tensor(_np.stack(idx, axis=-1).astype(_np.int64))


def argwhere(input):
    return nonzero(input)


def _no_bool_accumulate(name, dt):
    """torch refuses `dtype=bool` on a cumulative operation (measured — a
    `NotImplementedError`).

    **`sum(dtype=bool)` works and `cumsum(dtype=bool)` does not.** torch simply
    did not build that kernel rather than there being a rule, and diverging
    towards the permissive side is still diverging, so it is followed — handing
    back a value here means that code breaks against real torch.
    """
    plain = getattr(dt, "np", dt)
    if _np.dtype(plain) == _np.bool_:
        raise NotImplementedError(_like_torch(
            f"{name} cannot have a bool result dtype.",
            f'"{name}_out_cpu" not implemented for \'Bool\''))


def cumsum(input, dim, dtype=None):
    input = _wrap(input)
    if dtype is not None:
        # **Converted before going in.** Measured: the float `[1.7, −2.3, 0.9]`
        # with `dtype=int64` gives `[1, −1, −1]` — the running sum of the
        # truncated `[1, −2, 0]`. Truncated after folding it comes out
        # `[1, 0, 0]`.
        _no_bool_accumulate("cumsum", dtype)
        return cumsum(input.to(dtype), dim).to(dtype)
    return input._make(_np.cumsum(input.data, axis=dim), (input,),
                   lambda g: (_np.flip(_np.cumsum(_np.flip(_np.asarray(g), dim), axis=dim), dim),),
                   "CumsumBackward0")


def cumprod(input, dim, dtype=None):
    """A running product. Its backward is written **without division.**

    **Writing `dtype` in the body without declaring it as a parameter caused
    infinite recursion.** This file imports `_base`'s `dtype` into module scope,
    so the missing argument resolved to **a truthy global** — it surfaces as a
    `RecursionError` rather than a `NameError`. The eleventh shadowed name in
    this repository.

    The common derivation is `dL/dx_k = (1/x_k) * sum_{j>=k} g_j y_j`, and with a
    0 in the input that division blows up there and a `nan` flows quietly. No
    exception either. So the product with `x_k` left out is built directly for
    each k — it costs the square of the length, and `cumprod` is not on the inner
    training path, and **the side that is right when a 0 is in the mix** is this
    repository's standard.
    """
    input = _wrap(input)
    if dtype is not None:
        _no_bool_accumulate("cumprod", dtype)
        return cumprod(input.to(dtype), dim).to(dtype)
    out = _np.cumprod(input.data, axis=dim)

    def back(g):
        x = _np.moveaxis(input.data, dim, 0)
        gg = _np.moveaxis(_np.asarray(g), dim, 0)
        grad = _np.zeros_like(x, dtype=_np.result_type(x.dtype, _np.float32))
        prefix = _np.ones_like(x[0])                 # x_0 … x_{k-1}
        for k in range(x.shape[0]):
            run = prefix.copy()                      # the product at j=k (with x_k left out)
            acc = gg[k] * run
            for j in range(k + 1, x.shape[0]):
                run = run * x[j]
                acc = acc + gg[j] * run
            grad[k] = acc
            prefix = prefix * x[k]
        return (_np.moveaxis(grad, 0, dim),)

    return input._make(out, (input,), back, "CumprodBackward0")


def count_nonzero(t, *dim, **kw):
    """**Several axes as loose numbers, a tuple, or `dim=`** — torch takes all
    three."""
    axes = _loose(dim, kw, "dim")
    return Tensor(_np.count_nonzero(_wrap(t).data, axis=axes or None))


def _pick(t, idx, dim, op):
    """**Leave a gradient path** on the values taken. Taking them and cutting the
    path makes training quietly stop — that happens in top-k sampling and in a
    loss with a sort in it."""
    values = _np.take_along_axis(t.data, idx, axis=dim)
    shape = t.data.shape

    def back(g):
        z = _np.zeros(shape, dtype=_np.asarray(g).dtype)
        _np.put_along_axis(z, idx, _np.asarray(g), axis=dim)
        return (z,)

    return t._make(values, (t,), back, op)


def _order(data, dim, descending):
    """The sort indices. **The order among ties is part of the answer.**

    Sorting ascending and then reversing reverses the order among equal values as
    well, so the one torch puts first (the smaller index) goes last. Negating and
    sorting stably keeps it. numpy's default sort is quicksort and is not stable,
    so the ascending case is specified too — it being right today is an accident,
    and it diverges as the input grows.
    """
    return _np.argsort(-data if descending else data, axis=dim, kind="stable")


def topk(input, k, dim=-1, largest=True, sorted=True):
    """The top k as (values, indices). Chapter 32's top-k sampling is this.

    **`sorted` is torch's fifth seat and was missing**, so `topk(k, dim, largest,
    False)` was a `TypeError` rather than an unsorted answer. It is kept here
    because the position is torch's; the values come back sorted either way, which
    torch allows — `sorted=False` promises nothing about the order, not a different
    order."""
    input = _wrap(input)
    if not 0 <= k <= input.data.shape[dim]:
        raise RuntimeError("selected index k out of range")
    order = _order(input.data, dim, largest)
    idx = _np.take(order, _np.arange(k), axis=dim)
    return _MinMax(_pick(input, idx, dim, "TopkBackward0"), Tensor(idx))


def sort(input, dim=-1, descending=False, stable=False):
    """**`stable` is torch's fourth seat.** The sort underneath is numpy's
    `argsort`, which is stable by default, so equal values already keep their
    original order and the flag asks for what happens anyway — carried because the
    position is torch's and a positional call has to reach the same parameter."""
    input = _wrap(input)
    if input.data.ndim == 0:
        return _at_rank_0(input, lambda x: sort(x, 0, descending, stable))
    idx = _order(input.data, dim, descending)
    return _MinMax(_pick(input, idx, dim, "SortBackward0"), Tensor(idx))


def argsort(input, dim=-1, descending=False, stable=False):
    return sort(input, dim, descending, stable).indices


def unique(input, sorted=True, return_inverse=False, return_counts=False, dim=None):
    """torch's order — `return_inverse` **second**, and `dim` last.

    **Two arguments were missing from the middle**, so `x.unique(True, True)` asked
    for the inverse in torch and for the counts here: the same call, a different
    tuple, and both sides return something plausible.

    ## What `sorted=False` is, and is not

    torch documents it as *may return the elements in a different order*, and its CPU
    path sorts regardless. numpy always sorts. So the argument changes nothing here,
    changes nothing in torch on the CPU, and **is not refused** — it is honoured to
    exactly the degree torch honours it. Said out loud because a reader who sees it
    accepted would otherwise assume it does something.

    ## The inverse indexes the sorted values

    `values[inverse]` rebuilds the input, and that holds only because the values are
    sorted. torch's inverse means the same thing.
    """
    data = _wrap(input).data
    if dim is None:
        values, inverse, counts = _np.unique(
            data, return_inverse=True, return_counts=True)
        # **numpy shapes the inverse like the input and torch flattens first.**
        # Measured: a (2, 3) input gives torch a (6,) inverse and numpy a (2, 3) one.
        # A shape difference nothing else in this file would have looked at.
        inverse = inverse.reshape(-1)
    else:
        values, inverse, counts = _np.unique(
            data, axis=dim, return_inverse=True, return_counts=True)
        inverse = inverse.reshape(-1)
    out = (Tensor(values),)
    if return_inverse:
        out += (Tensor(inverse),)
    if return_counts:
        out += (Tensor(counts),)
    return out[0] if len(out) == 1 else out


# -------------------------------------------------------------- linear algebra

def mm(input, mat2): return _wrap(input) @ _wrap(mat2)
def bmm(input, mat2): return _wrap(input) @ _wrap(mat2)


def dot(input, tensor): return (_wrap(input) * _wrap(tensor)).sum()


def outer(input, vec2):
    input, vec2 = _wrap(input), _wrap(vec2)
    return input.reshape(-1, 1) @ vec2.reshape(1, -1)


def _diagonal_scatter(shape, g):
    """A zero matrix with `g` laid on the diagonal. The backwards of `diag` and
    `trace` have the same shape."""
    z = _np.zeros(shape, dtype=_np.asarray(g).dtype)
    n = min(shape)
    z[_np.arange(n), _np.arange(n)] = g
    return z


def diag(input, diagonal=0):
    """1-D builds a diagonal matrix and 2-D takes the diagonal — opposite
    directions, so opposite backwards.

    **`diagonal` says which diagonal.** Positive is above and negative below.
    Without it, `x.diag(1)` stops with a `TypeError` — the loud kind, so no value
    diverged.
    """
    input = _wrap(input)
    k = int(diagonal)
    out = _np.diag(input.data, k)
    if input.data.ndim == 1:
        def back(g):
            # Only that diagonal is taken back out of the matrix built.
            return (_np.diagonal(_np.asarray(g), k).copy(),)
    else:
        def back(g):
            z = _np.zeros_like(input.data)
            _np.fill_diagonal(z[max(0, -k):, max(0, k):], 1.0)
            spread = _np.zeros_like(input.data)
            rows, cols = _np.nonzero(z)
            spread[rows, cols] = _np.asarray(g)
            return (spread,)
    return input._make(out, (input,), back, "DiagBackward0")


def trace(input):
    input = _wrap(input)
    _rank(input.data, (2,), "trace: expected a matrix, but got tensor with dim {n}")
    return input._make(_np.trace(input.data), (input,),
                   lambda g: (_diagonal_scatter(input.data.shape, _np.asarray(g)),),
                   "TraceBackward0")


def einsum(equation, *operands):
    """The backward is an einsum too — swapping the output subscripts into that
    term's slot produces that term's gradient.

    One place snags. When a subscript appears **in that term alone** and in
    neither the output nor any other term (the `j` of `ij->i`), einsum cannot
    create an axis that was not there. In that case one more term filled with 1s
    of that axis's length is inserted — the value does not change and the axis
    appears.

    `...` and a subscript repeated within one term (`ii->i`) do not follow this
    rule directly. So **rather than give a wrong gradient it gives none** — in
    that case `backward()` refuses.
    """
    ops = [_wrap(o) for o in operands]
    out = _np.einsum(equation, *[o.data for o in ops])
    eq = equation.replace(" ", "")
    if "->" in eq:
        lhs, rhs = eq.split("->")
    else:
        lhs = eq
        counts = {}
        for c in lhs.replace(",", ""):
            counts[c] = counts.get(c, 0) + 1
        rhs = "".join(sorted(c for c, k in counts.items() if k == 1))
    subs = lhs.split(",")
    if "..." in eq or any(len(set(s)) != len(s) for s in subs):
        return Tensor(out)

    def back(g):
        g = _np.asarray(g)
        grads = []
        for i, mine in enumerate(subs):
            rest = [(subs[j], ops[j].data) for j in range(len(subs)) if j != i]
            known = set(rhs) | {c for s, _ in rest for c in s}
            missing = [c for c in mine if c not in known]
            spec = [rhs] + [s for s, _ in rest]
            terms = [g] + [d for _, d in rest]
            if missing:
                sizes = [ops[i].data.shape[mine.index(c)] for c in missing]
                spec.append("".join(missing))
                terms.append(_np.ones(sizes, dtype=ops[i].data.dtype))
            grads.append(_np.einsum(",".join(spec) + "->" + mine, *terms))
        return tuple(grads)

    return ops[0]._make(out, tuple(ops), back, "EinsumBackward0")


def empty(*shape, dtype=None, requires_grad=False, device=None):
    return zeros(*shape, dtype=dtype, requires_grad=requires_grad)




def leaky_relu(input, negative_slope=0.01, inplace=False):
    """torch takes `inplace` here; it is the underscore name by another
    spelling, routed through the same write-back rather than a second
    formula."""
    return _inplace_arg(input, inplace, "leaky_relu",
                        lambda: _leaky_relu_body(input, negative_slope))


def _leaky_relu_body(t, negative_slope=0.01):
    t = _wrap(t)
    pick = t.data > 0
    return t._make(_np.where(pick, t.data, negative_slope * t.data), (t,),
                   lambda g: (g * _np.where(pick, 1.0, negative_slope),), "LeakyReluBackward0")


def elu(input, alpha=1.0, inplace=False):
    """torch takes `inplace` here; it is the underscore name by another
    spelling, routed through the same write-back rather than a second
    formula."""
    return _inplace_arg(input, inplace, "elu",
                        lambda: _elu_body(input, alpha))


def _elu_body(t, alpha=1.0):
    t = _wrap(t)
    pick = t.data > 0
    out = _np.where(pick, t.data, alpha * (_np.exp(_np.minimum(t.data, 0)) - 1))
    return t._make(out, (t,), lambda g: (g * _np.where(pick, 1.0, out + alpha),),
                   "EluBackward0")


def silu(input, inplace=False):
    """torch takes `inplace` here; it is the underscore name by another
    spelling, routed through the same write-back rather than a second
    formula."""
    return _inplace_arg(input, inplace, "silu",
                        lambda: _silu_body(input))


def _silu_body(t):
    """x·σ(x). Also called Swish."""
    t = _wrap(t)
    sig = 1.0 / (1.0 + _np.exp(-_np.clip(t.data, -60, 60)))
    return t._make(t.data * sig, (t,),
                   lambda g: (g * (sig * (1 + t.data * (1 - sig))),), "SiluBackward0")


def _gelu_tanh(t):
    """The `approximate="tanh"` side —
    0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³))).

    **Its values differ from the exact form's.** The max diff is around 1e-4,
    which sits near this project's tolerance, so "near enough, keep one" nearly
    worked here. torch keeps the two apart because the tanh side is faster, not
    because they are equal — the golden asks about them separately.
    """
    d = _np.asarray(t.data, dtype=_np.float64)
    root = _math.sqrt(2.0 / _math.pi)
    inner = root * (d + 0.044715 * d ** 3)
    th = _np.tanh(inner)
    out = (0.5 * d * (1.0 + th)).astype(t.data.dtype)

    def back(g):
        dinner = root * (1.0 + 3 * 0.044715 * d * d)
        grad = 0.5 * (1.0 + th) + 0.5 * d * (1.0 - th * th) * dinner
        return (g * grad.astype(t.data.dtype),)

    return t._make(out, (t,), back, "GeluBackward0")


def _gelu(t):
    """The same formula as torch's default gelu (the exact form) —
    0.5·x·(1 + erf(x/√2)).

    Both the forward and the backward were `np.vectorize`. Calling Python per
    element, one 8×32×2048 pass took 197ms, and switching to elementwise numpy
    takes 9.9ms (measured, 20×). The max diff against real torch is 4.77e-07, **the
    same** as before the change (46,000 points over x ∈ [-8, 8] plus the tails,
    all passing allclose(1e-5)).

    **Moved here from `nn`.** The transformer layers use it and so does `gelu`,
    and `gelu` is above this. As one file the ordering was invisible, and
    splitting revealed the shape of something below calling something above — the
    inverted layering is corrected.
    """
    d = _np.asarray(t.data, dtype=_np.float64)
    ope = _one_plus_erf64(d / _math.sqrt(2.0))
    out = (0.5 * d * ope).astype(t.data.dtype)

    def back(g):
        grad = 0.5 * ope + d * _np.exp(-d * d / 2) / _math.sqrt(2 * _math.pi)
        return (g * grad.astype(t.data.dtype),)

    return t._make(out, (t,), back, "GeluBackward0")


def gelu(input, approximate="none"):
    if approximate == "tanh":
        return _gelu_tanh(_wrap(input))
    if approximate != "none":
        raise ValueError(
            f"gelu(): approximate is 'none' or 'tanh' (got {approximate!r})")
    return _gelu(_wrap(input))


# ── the seventeen activations ───────────────────────────────────────────────
#
# **Which side is chosen at a kink is the whole of it.** The formulas are in the
# documentation, and what torch gives exactly on a boundary — `x == 0`,
# `x == ±3`, `x == 6` — has to be measured; random input never produces those
# points. The golden's `kinks` are those points.
#
# The constants are torch's, written out. An approximation diverges around the
# fifth digit, and that is a "nearly right" state which goes uncaught for a long
# time.
_SELU_ALPHA = 1.6732632423543772848170429916717
_SELU_SCALE = 1.0507009873554804934193349852946


def _sigmoid_of(d):
    return 1.0 / (1.0 + _np.exp(-_np.clip(d, -60, 60)))


def celu(input, alpha=1.0, inplace=False):
    """torch takes `inplace` here; it is the underscore name by another
    spelling, routed through the same write-back rather than a second
    formula."""
    return _inplace_arg(input, inplace, "celu",
                        lambda: _celu_body(input, alpha))


def _celu_body(t, alpha=1.0):
    """CELU. Unlike `ELU` it **divides** the negative side by α before taking the
    exponential.

    At α=1 it gives ELU's values — so measured without giving α, the two cannot be
    told apart.
    """
    t = _wrap(t)
    pick = t.data > 0
    inner = _np.exp(_np.minimum(t.data, 0) / alpha)
    out = _np.where(pick, t.data, alpha * (inner - 1))
    return t._make(out, (t,), lambda g: (g * _np.where(pick, 1.0, inner),),
                   "CeluBackward0")


def hardshrink(input, lambd=0.5):
    """The value where |x| > λ and 0 otherwise. **On the boundary it is 0**
    (`>`, not `>=`)."""
    input = _wrap(input)
    keep = _np.abs(input.data) > lambd
    return input._make(_np.where(keep, input.data, 0.0), (input,),
                   lambda g: (g * keep,), "HardshrinkBackward0")


def hardsigmoid(input, inplace=False):
    """torch takes `inplace` here; it is the underscore name by another
    spelling, routed through the same write-back rather than a second
    formula."""
    return _inplace_arg(input, inplace, "hardsigmoid",
                        lambda: _hardsigmoid_body(input))


def _hardsigmoid_body(t):
    """Imitates a sigmoid with piecewise straight lines. The kinks are at ±3."""
    t = _wrap(t)
    d = t.data
    out = _np.clip(d / 6.0 + 0.5, 0.0, 1.0)
    inside = (d > -3.0) & (d < 3.0)
    return t._make(out, (t,), lambda g: (g * _np.where(inside, 1.0 / 6.0, 0.0),),
                   "HardsigmoidBackward0")


def hardswish(input, inplace=False):
    """torch takes `inplace` here; it is the underscore name by another
    spelling, routed through the same write-back rather than a second
    formula."""
    return _inplace_arg(input, inplace, "hardswish",
                        lambda: _hardswish_body(input))


def _hardswish_body(t):
    """x·hardsigmoid(x). Used instead of swish on mobile."""
    t = _wrap(t)
    d = t.data
    out = _np.where(d <= -3.0, 0.0, _np.where(d >= 3.0, d, d * (d + 3.0) / 6.0))
    grad = _np.where(d <= -3.0, 0.0, _np.where(d >= 3.0, 1.0, (2.0 * d + 3.0) / 6.0))
    return t._make(out, (t,), lambda g: (g * grad,), "HardswishBackward0")


def hardtanh(input, min_val=-1.0, max_val=1.0, inplace=False):
    """torch takes `inplace` here; it is the underscore name by another
    spelling, routed through the same write-back rather than a second
    formula."""
    return _inplace_arg(input, inplace, "hardtanh",
                        lambda: _hardtanh_body(input, min_val, max_val))


def _hardtanh_body(t, min_val=-1.0, max_val=1.0):
    t = _wrap(t)
    d = t.data
    inside = (d > min_val) & (d < max_val)
    return t._make(_np.clip(d, min_val, max_val), (t,),
                   lambda g: (g * inside,), "HardtanhBackward0")


def logsigmoid(input):
    """log σ(x). **Computed directly at large negatives it becomes log(0)** — the
    stable form is used."""
    input = _wrap(input)
    d = input.data
    out = -(_np.logaddexp(0.0, -d))
    sig = _sigmoid_of(d)
    return input._make(out.astype(d.dtype), (input,), lambda g: (g * (1.0 - sig),),
                   "LogSigmoidBackward0")


def softplus(input, beta=1.0, threshold=20.0):
    """(1/β)·log(1+e^{βx}). **Past the threshold, βx is simply x** — so it does
    not overflow.

    Without that branch, large input produces `inf` and every gradient after it
    becomes NaN.
    """
    input = _wrap(input)
    d = input.data
    big = beta * d > threshold
    out = _np.where(big, d, _np.logaddexp(0.0, beta * d) / beta)
    sig = _sigmoid_of(beta * d)
    return input._make(out.astype(d.dtype), (input,),
                   lambda g: (g * _np.where(big, 1.0, sig),), "SoftplusBackward0")


def mish(input, inplace=False):
    """torch takes `inplace` here; it is the underscore name by another
    spelling, routed through the same write-back rather than a second
    formula."""
    return _inplace_arg(input, inplace, "mish",
                        lambda: _mish_body(input))


def _mish_body(t):
    """x·tanh(softplus(x))."""
    t = _wrap(t)
    d = t.data
    sp = _np.logaddexp(0.0, d)
    th = _np.tanh(sp)
    sig = _sigmoid_of(d)
    out = (d * th).astype(d.dtype)
    grad = th + d * (1.0 - th * th) * sig
    return t._make(out, (t,), lambda g: (g * grad.astype(d.dtype),), "MishBackward0")


def relu6(input, inplace=False):
    """torch takes `inplace` here; it is the underscore name by another
    spelling, routed through the same write-back rather than a second
    formula."""
    return _inplace_arg(input, inplace, "relu6",
                        lambda: _relu6_body(input))


def _relu6_body(t):
    """clamp(x, 0, 6). **The gradient is 0 on the boundaries** — both of
    them."""
    t = _wrap(t)
    d = t.data
    inside = (d > 0.0) & (d < 6.0)
    return t._make(_np.clip(d, 0.0, 6.0), (t,), lambda g: (g * inside,),
                   "Relu6Backward0")


def selu(input, inplace=False):
    """torch takes `inplace` here; it is the underscore name by another
    spelling, routed through the same write-back rather than a second
    formula."""
    return _inplace_arg(input, inplace, "selu",
                        lambda: _selu_body(input))


def _selu_body(t):
    t = _wrap(t)
    d = t.data
    pick = d > 0
    inner = _np.exp(_np.minimum(d, 0))
    out = _SELU_SCALE * _np.where(pick, d, _SELU_ALPHA * (inner - 1))
    grad = _SELU_SCALE * _np.where(pick, 1.0, _SELU_ALPHA * inner)
    return t._make(out.astype(d.dtype), (t,), lambda g: (g * grad.astype(d.dtype),),
                   "SeluBackward0")


def softshrink(input, lambd=0.5):
    """**Pulls towards the origin** by λ. Unlike `hardshrink` the values stay
    continuous."""
    input = _wrap(input)
    d = input.data
    out = _np.where(d > lambd, d - lambd, _np.where(d < -lambd, d + lambd, 0.0))
    keep = _np.abs(d) > lambd
    return input._make(out.astype(d.dtype), (input,), lambda g: (g * keep,),
                   "SoftshrinkBackward0")


def softsign(input):
    """x/(1+|x|)."""
    input = _wrap(input)
    d = input.data
    denom = 1.0 + _np.abs(d)
    return input._make((d / denom).astype(d.dtype), (input,),
                   lambda g: (g / (denom * denom),), "SoftsignBackward0")


def tanhshrink(input):
    """x − tanh(x)."""
    input = _wrap(input)
    d = input.data
    th = _np.tanh(d)
    return input._make((d - th).astype(d.dtype), (input,), lambda g: (g * (th * th),),
                   "TanhshrinkBackward0")


def threshold(input, threshold, value, inplace=False):       # noqa: A002
    """torch takes `inplace` here; it is the underscore name by another
    spelling, routed through the same write-back rather than a second
    formula."""
    return _inplace_arg(input, inplace, "threshold",
                        lambda: _threshold_body(input, threshold, value))


def _threshold_body(t, threshold, value):                # noqa: A002
    """The value where x > threshold and `value` otherwise. **The boundary goes
    to value.**"""
    t = _wrap(t)
    keep = t.data > threshold
    return t._make(_np.where(keep, t.data, value), (t,), lambda g: (g * keep,),
                   "ThresholdBackward0")


def softmin(input, dim=None, _stacklevel=3, dtype=None):   # noqa: A002
    """softmax(−x). **Without the negation it becomes softmax** — it diverges
    only in the values.

    The negation happens here and the rest is `softmax`'s, `_stacklevel` included:
    passed straight through, the warning names this caller and not this line.
    """
    t, dim = _softmax_args(input, dim, dtype, _stacklevel + 1, "softmin")
    return softmax(-t, dim=dim)


def glu(input, dim=-1):
    """Split the axis in half and take `a · σ(b)`. The only activation that is
    not elementwise."""
    input = _wrap(input)
    n = input.data.shape[dim]
    if n % 2:
        raise RuntimeError(f"glu needs an even length along dimension {dim} (got {n})")
    half = n // 2
    a = narrow(input, dim, 0, half)
    b = narrow(input, dim, half, half)
    return a * sigmoid(b)


def prelu(input, weight):
    """The slope on the negative side is **learned.** With one weight, every
    channel shares it.

    **Exactly 0 belongs to the negative side.** The forward pass is 0 either way
    so it is invisible there, and the gradient parts — torch gives 1 only where
    `x > 0` and gives w at `x == 0`. Split with `x < 0` at first, that single
    point gave a max diff of 3.75, and the golden's `kinks` input contains a 0, so
    it was caught. With random input it would never have been.
    """
    input, weight = _wrap(input), _wrap(weight)
    d = input.data
    w = weight.data
    if w.size != 1:
        # A different slope per channel — spread to line up with the channel
        # axis (1).
        shape = [1] * d.ndim
        shape[1 if d.ndim > 1 else 0] = w.size
        w = w.reshape(shape)
    pos = d > 0
    out = _np.where(pos, d, w * d)

    def back(g):
        g = _np.asarray(g)
        dx = g * _np.where(pos, 1.0, w)
        dw = _unbroadcast(g * _np.where(pos, 0.0, d), weight.data.shape)
        return (dx, dw)

    return input._make(out.astype(d.dtype), (input, weight), back, "PreluBackward0")


def log_softmax(input, dim=None, _stacklevel=3, dtype=None):   # noqa: A002
    t, dim = _softmax_args(input, dim, dtype, _stacklevel, "log_softmax")
    shifted = t.data - t.data.max(axis=dim, keepdims=True)
    out = shifted - _np.log(_np.exp(shifted).sum(axis=dim, keepdims=True))
    soft = _np.exp(out)

    def back(g):
        g = _np.asarray(g)
        return (g - soft * g.sum(axis=dim, keepdims=True),)

    return t._make(out, (t,), back, "LogSoftmaxBackward0")


def dropout(input, p=0.5, training=True, inplace=False):
    """torch takes `inplace` here; it is the underscore name by another
    spelling, routed through the same write-back rather than a second
    formula."""
    return _inplace_arg(input, inplace, "dropout",
                        lambda: _dropout_body(input, p, training))


def _dropout_body(t, p=0.5, training=True):
    """The surviving values are scaled by `1/(1-p)` — **so that the magnitudes
    match between training and inference.**

    `p=1` is branched separately. Without that, `1/(1-p)` becomes a division by
    zero and produces NaN, and NaN differs even from itself, so nothing compared
    after it can pass. torch gives 0 at that point.
    """
    if not training or p == 0:
        return _wrap(t)
    t = _wrap(t)
    if p == 1:
        return t * Tensor(_np.zeros_like(t.data))
    mask = (_rng.random(t.data.shape) > p).astype(t.data.dtype) / (1 - p)
    return t * Tensor(mask)


def _avg_pool_nd(input, spatial, kernel_size, stride, padding, ceil_mode,     # noqa: A002
                 count_include_pad, divisor_override, name):
    """Average pooling over `spatial` trailing axes, in torch's arithmetic.

    **Written once for all three ranks.** It began as `avg_pool2d` alone, and
    `avg_pool1d` and `avg_pool3d` were `_fixed(...)` calls that took a kernel and a
    stride and nothing else — so `padding`, `ceil_mode`, `count_include_pad` and
    `divisor_override` existed at rank 2 and not at ranks 1 and 3. Copying the body
    twice would have made three places for the divisor rule to drift apart, and the
    divisor rule is the whole difficulty here.

    The pair that decides the *divisor* is why this is more than plumbing: an average
    over a padded window can count the padding or not, and `count_include_pad=False`
    is what makes an edge window an average of the values that are really there.
    torch's default is `True`, so the padded edges are pulled toward zero — a choice,
    not an accident, and one a caller has to be able to reverse. `divisor_override`
    replaces the count outright, which is how a fixed-scale pooling layer is built.

    The zeros are laid down explicitly rather than through `_pool_windows`, because
    with `count_include_pad=False` the divisor is **per window** — every edge window
    has a different count of real cells — and a shared helper that returns windows
    has nowhere to carry that.
    """
    x = _wrap(input)
    ks = _spread(kernel_size, spatial)
    st = _spread(stride if stride is not None else kernel_size, spatial)
    pd = _spread(padding, spatial)
    xd = x.data
    real_len = xd.shape[2:]
    if any(pd):
        xd = _np.pad(xd, ((0, 0), (0, 0)) + tuple((p, p) for p in pd))
    padded = list(xd.shape[2:])

    up = (lambda a, b: -(-a // b)) if ceil_mode else (lambda a, b: a // b)
    outs = []
    for k in range(spatial):
        n = up(padded[k] - ks[k], st[k]) + 1
        # torch drops a ceil-mode window that begins inside the padding on the far side.
        if ceil_mode:
            while n > 1 and (n - 1) * st[k] >= padded[k] - pd[k]:
                n -= 1
        outs.append(n)

    # A ceil-mode window may run off the end; the cells that are not there are zeros
    # and never counted (see the mask below), so the array is simply extended.
    grow = [max(0, (outs[k] - 1) * st[k] + ks[k] - padded[k]) for k in range(spatial)]
    if any(grow):
        xd = _np.pad(xd, ((0, 0), (0, 0)) + tuple((0, g) for g in grow))

    axes = tuple(range(2, 2 + spatial))
    win = _np.lib.stride_tricks.sliding_window_view(xd, tuple(ks), axis=axes)
    picker = (slice(None), slice(None)) + tuple(
        slice(None, outs[k] * st[k], st[k]) for k in range(spatial))
    win = win[picker]
    totals = win.sum(axis=tuple(range(2 + spatial, 2 + 2 * spatial)))

    if divisor_override is not None:
        counts = _np.full(tuple(outs), float(divisor_override))
    else:
        # **One mask decides every divisor, and the two kinds of padding differ.**
        # Explicit `padding` counts or does not, by `count_include_pad`. The zeros a
        # ceil-mode window runs into past the end of the input **never** count —
        # torch divides that clipped window by its real cells whichever way the flag
        # is set (measured: a 5×5 with `kernel=2, stride=2, ceil_mode=True` gives
        # `24.0` in the corner, which is the single cell and not a quarter of it).
        #
        # Written as a mask rather than as two cases because the count is **per
        # window** once anything is clipped, and a formula has nowhere to put that.
        real = _np.zeros(xd.shape[2:], dtype=_np.float64)
        real[tuple(slice(pd[k], pd[k] + real_len[k]) for k in range(spatial))] = 1.0
        if count_include_pad:
            real[tuple(slice(None, padded[k]) for k in range(spatial))] = 1.0
        rw = _np.lib.stride_tricks.sliding_window_view(
            real, tuple(ks), axis=tuple(range(spatial)))
        rw = rw[tuple(slice(None, outs[k] * st[k], st[k]) for k in range(spatial))]
        counts = rw.sum(axis=tuple(range(spatial, 2 * spatial)))
    out = totals / counts

    def back(g):
        g = _np.asarray(g) / counts
        gx = _np.zeros(xd.shape, dtype=xd.dtype)
        for offset in _itertools.product(*[range(ks[k]) for k in range(spatial)]):
            where = (slice(None), slice(None)) + tuple(
                slice(offset[k], offset[k] + outs[k] * st[k], st[k])
                for k in range(spatial))
            gx[where] += g
        keep = (slice(None), slice(None)) + tuple(
            slice(pd[k], pd[k] + real_len[k]) for k in range(spatial))
        return (gx[keep],)

    return x._make(out, (x,), back, name)


def avg_pool2d(input, kernel_size, stride=None, padding=0, ceil_mode=False,   # noqa: A002
               count_include_pad=True, divisor_override=None):
    """**It takes a different window per axis.** Because
    `adaptive_avg_pool2d` has to be able to reduce the height and the width
    differently — taking squares only leaves nothing to build it on.
    """
    return _avg_pool_nd(input, 2, kernel_size, stride, padding, ceil_mode,
                        count_include_pad, divisor_override, "AvgPool2DBackward0")


def _pool_all(x):
    """AdaptiveAvgPool2d(1) alone is supported — that is the only common one and
    the rest are refused."""
    return x.mean(dim=2).mean(dim=2).reshape(x.data.shape[0], x.data.shape[1], 1, 1)


def layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-5):
    mean = input.mean(dim=-1, keepdim=True)
    centered = input - mean
    var = (centered * centered).mean(dim=-1, keepdim=True)
    out = centered / (var + eps) ** 0.5
    if weight is not None:
        out = out * weight
    return out + bias if bias is not None else out


def embedding(input, weight, padding_idx=None, max_norm=None,   # noqa: A002
              norm_type=2.0, scale_grad_by_freq=False, sparse=False):
    """torch's list. **This took two of seven, while `nn.Embedding` had all of
    them** — the layer carried `padding_idx` and `max_norm` and refused
    `scale_grad_by_freq` and `sparse` by name, and the function beside it silently
    did none of that.

    So the direction of the fix is the interesting part. `F.nll_loss` was the same
    shape earlier today and was pointed at its layer; here it goes the other way,
    because a *function* is the primitive — it takes the caller's own weight tensor,
    and building a layer around one would copy the table and cut the gradient. The
    layer now calls this.

    `max_norm` shortens rows **in the table itself** — a side effect on a parameter,
    which `_renorm_rows` explains at length and which no output comparison can see.
    """
    if sparse:
        _unsupported("embedding(sparse=True) — there is no sparse gradient here")
    ids = _wrap(input).data.astype(int)
    dim = weight.data.shape[1]
    if max_norm is not None:
        _renorm_rows(weight, ids, max_norm, norm_type)
    out = weight.data[ids]

    def back(g):
        gw = _np.zeros_like(weight.data)
        flat = ids.reshape(-1)
        _np.add.at(gw, flat, _np.asarray(g).reshape(-1, dim))
        # **`scale_grad_by_freq` divides each row by how often that index appears in
        # this batch**, so a token seen three times does not pull three times as hard
        # as one seen once. Measured against torch: with ids `[[0,1,1],[2,1,0]]` row 0
        # is halved and row 1 is divided by three.
        #
        # A row nobody indexed has a count of zero and a gradient of zero; the divisor
        # is forced to one there rather than dividing zero by zero into a `nan` that
        # would then poison every step the optimiser takes on that row.
        if scale_grad_by_freq:
            seen = _np.bincount(flat, minlength=gw.shape[0]).astype(gw.dtype)
            gw = gw / _np.where(seen == 0, 1.0, seen)[:, None]
        # **The padding row learns nothing.** Left in, a pad token drifts toward
        # whatever the loss wants and the mask stops meaning "ignore this".
        if padding_idx is not None:
            gw[padding_idx] = 0.0
        return (gw,)

    return weight._make(out, (weight,), back, "EmbeddingBackward0")


def nll_loss(input, target, reduction="mean"):  # noqa: A002
    n = input.data.shape[0]
    picked = input[_np.arange(n), target.data.astype(int)]
    return _reduce(-picked, reduction)


def _weighted_reduce(out, weight, reduction, where_, mean_over_weights):
    """torch's `weight` on the elementwise losses.

    **The seat was taken and the value refused**, on the ground that under `mean`
    torch divides by the sum of the weights rather than by the sample count, so a
    `weight` accepted and unused changes the number quietly. That reason was right,
    and it was also the specification — measured on three weight vectors:

        none  w · ℓ                      all three
        sum   Σ w · ℓ                    all three
        mean  Σ w·ℓ / Σ w                `l1_loss` and `mse_loss`
        mean  Σ w·ℓ / n                  `huber_loss`

    **`huber_loss` divides by the count and the other two do not**, which is why
    `mean_over_weights` is a parameter rather than a rule. Assuming the family
    agreed would have made huber's `mean` wrong by a factor of `Σw / n` — 2.5 on a
    weight vector of `[1, 2, 3, 4]`, and no exception anywhere.

    The shapes must match exactly. torch does not broadcast here: a `(6,)` weight
    against a `(2, 3)` input raises *Weights and input must have the same size*,
    and so does a `(3,)` one, so the wording is taken from torch rather than
    invented.
    """
    if weight is None:
        return _reduce(out, reduction)
    weight = _wrap(weight)
    if tuple(weight.data.shape) != tuple(out.data.shape):
        raise ValueError("Weights and input must have the same size.")
    scaled = out * weight
    if reduction != "mean":
        return _reduce(scaled, reduction)
    # `_reduce` is still asked for the divisor-free part so that an unknown
    # `reduction` stops in exactly one place.
    if not mean_over_weights:
        return _reduce(scaled, "mean")
    return scaled.sum() / weight.sum()


def l1_loss(input, target, size_average=None, reduce=None, reduction="mean",
            weight=None):  # noqa: A002  # noqa: A002
    reduction = _legacy_reduction(size_average, reduce, reduction)
    return _weighted_reduce((_wrap(input) - _wrap(target)).abs(), weight,
                            reduction, "l1_loss", True)


def smooth_l1_loss(input, target, size_average=None, reduce=None, reduction="mean",
                   beta=1.0):  # noqa: A002  # noqa: A002
    """Squared for small errors and absolute for large ones. Less shaken by
    outliers.

    **The third and fourth arguments were the other way round**, so
    `F.smooth_l1_loss(a, b, 'sum')` set `beta` to a string here and `reduction` in
    torch. The layer `nn.SmoothL1Loss` had this same swap and was corrected long ago
    — it is the row the core↔borch.ts axis was built on — and the *function* kept it,
    which is what happens when a family is fixed one member at a time.

    **It was invisible until the first parameter was renamed.** While this was
    `pred` and torch said `input`, the lists could not be lined up at all, so the row
    sat in `unaligned` and reported nothing further. Matching the name was cosmetic;
    what it uncovered was not. That is the third time in this repository that
    clearing a vague classification showed a specific defect beneath it.
    """
    reduction = _legacy_reduction(size_average, reduce, reduction)
    diff = _wrap(input) - _wrap(target)
    small = _np.abs(diff.data) < beta
    return _reduce(where(Tensor(small), 0.5 * diff * diff / beta,
                         diff.abs() - 0.5 * beta), reduction)


# ---------------------------------------------------------------------- losses
#
# **How it folds is part of the loss.** Every torch loss takes `reduction`, and
# by its value becomes elementwise, a mean or a sum. Gathered in one place,
# thirteen of them use the same rule — written per loss there are thirteen places
# that can drift, and what actually differs is the three lines here.

def _legacy_reduction(size_average, reduce, reduction):     # noqa: A002
    """torch's deprecated `size_average`/`reduce`, folded into a `reduction`.

    **These were left out on the ground that torch ignores them whenever `reduction`
    is given. Measured, torch does the opposite** — the pair wins:

        F.l1_loss(a, b, size_average=True, reduce=True, reduction='sum')
            → the *mean*, not the sum

    and the whole truth table is
    `reduce=False → none`, else `size_average=False → sum`, else `mean`,
    with `None` reading as `True` on both.

    Leaving them out also moved every later argument one or two seats forward, so a
    positional call meant two different things:

        F.l1_loss(a, b, 'sum')   torch 2.5 (the mean, from the legacy path)
                                 here 10.0 (the sum) — before this

    `huber_loss` was the control that showed it: newer, no deprecated pair, third
    seat really is `reduction`, and it agreed all along.

    The warning is torch's own wording, so a caller who hits it can search for the
    same sentence in torch's issues. `stacklevel` points at the caller rather than
    at this helper — a deprecation notice naming a private function inside the
    library is a notice nobody can act on.
    """
    if size_average is None and reduce is None:
        return reduction
    got = ("none" if reduce is False
           else "sum" if size_average is False
           else "mean")
    _warnings.warn(
        f"size_average and reduce args will be deprecated, "
        f"please use reduction='{got}' instead.",
        UserWarning, stacklevel=3)
    return got


def _reduce(out, reduction):
    if reduction == "none":
        return out
    if reduction == "sum":
        return out.sum()
    # **An unknown value is not swallowed as the mean.** It used to be
    # `else: return out.mean()`, and then a typo like `reduction="MEAN"` passes
    # quietly and training goes on with it — somebody writes it in capitals and
    # believes what they chose is being used. torch stops (measured).
    #
    # It is the shape of an `else` wearing one value's name and swallowing the
    # rest of the domain, and the same shape turned up in `norm(p)` and
    # `dist(p)`.
    if reduction != "mean":
        raise ValueError(_like_torch(
            f"reduction must be one of 'none', 'mean', 'sum' "
            f"(got {reduction!r}).",
            f"{reduction} is not a valid value for reduction"))
    return out.mean()


def huber_loss(input, target, reduction="mean", delta=1.0,   # noqa: A002
               weight=None):
    """**It equals `SmoothL1Loss` at δ=1 alone.**

    The actual relation is `huber(δ) = δ · smooth_l1(β=δ)`. Measured at the
    defaults only, treating the two as one function still passes, so the golden
    asks with δ changed.

    `weight` — see `_weighted_reduce`. **This one's `mean` divides by the count**
    where `l1_loss` and `mse_loss` divide by the sum of the weights; measured, not
    inferred from the family.
    """
    diff = _wrap(input) - _wrap(target)
    small = _np.abs(diff.data) < delta
    return _weighted_reduce(where(Tensor(small), 0.5 * diff * diff,
                                  delta * (diff.abs() - 0.5 * delta)),
                            weight, reduction, "huber_loss", False)


def kl_div(input, target, size_average=None, reduce=None, reduction="mean",
           log_target=False):  # noqa: A002  # noqa: A002
    """`target · (log target − input)`. `input` has to be **already logged.**

    **`reduction` has four settings here.** `mean` divides by the element count
    and `batchmean` by the batch size — the latter is what matches the
    mathematical definition, and torch itself warns that it will change in a
    coming release. The present values have to match, so the present rule is
    followed.
    """
    reduction = _legacy_reduction(size_average, reduce, reduction)
    p, t = _wrap(input), _wrap(target)
    out = (t.exp() * (t - p)) if log_target else (t * (t.log() - p))
    if reduction == "batchmean":
        return out.sum() / out.data.shape[0]
    return _reduce(out, reduction)


def poisson_nll_loss(input, target, log_input=True, full=False, size_average=None,
                     eps=1e-8, reduce=None, reduction="mean"):  # noqa: A002
    """The Poisson negative log likelihood.

    **The Stirling correction is added only where `target > 1`.** Added
    unconditionally it is wrong only where the target is small — confirmed by
    measurement (at targets of 0, 0.5 and 1 the difference is 0).
    """
    reduction = _legacy_reduction(size_average, reduce, reduction)
    p, t = _wrap(input), _wrap(target)
    out = (p.exp() - t * p) if log_input else (p - t * (p + eps).log())
    if full:
        big = t.data > 1
        stirling = (t * t.log() - t + 0.5 * (2 * _math.pi * t).log())
        out = out + where(Tensor(big.astype(t.data.dtype)), stirling,
                          Tensor(_np.zeros_like(t.data)))
    return _reduce(out, reduction)


def gaussian_nll_loss(input, target, var, full=False, eps=1e-6, reduction="mean"):
    """The Gaussian negative log likelihood.

    **The variance is clamped by `eps`.** Unclamped it divides by zero and becomes
    infinite — at `var=1e-9` with the default `eps=1e-6` the clamped value gives
    124993 (measured).
    """
    p, t, v = _wrap(input), _wrap(target), _wrap(var)
    safe = clamp(v, min=eps)
    diff = p - t
    out = 0.5 * (safe.log() + diff * diff / safe)
    if full:
        out = out + 0.5 * _math.log(2 * _math.pi)
    return _reduce(out, reduction)


def margin_ranking_loss(input1, input2, target, margin=0.0, size_average=None,
                        reduce=None, reduction="mean"):
    """`max(0, −y·(x₁ − x₂) + margin)`."""
    reduction = _legacy_reduction(size_average, reduce, reduction)
    a, b, t = _wrap(input1), _wrap(input2), _wrap(target)
    return _reduce(relu(-t * (a - b) + margin), reduction)


def cosine_embedding_loss(input1, input2, target, margin=0.0, size_average=None,
                          reduce=None, reduction="mean"):
    """`1 − cos` at `y=1` and `max(0, cos − margin)` at `y=−1`."""
    reduction = _legacy_reduction(size_average, reduce, reduction)
    a, b, t = _wrap(input1), _wrap(input2), _wrap(target)
    cos = cosine_similarity(a, b, dim=1)
    same = Tensor((t.data > 0).astype(cos.data.dtype))
    return _reduce(same * (1 - cos) + (1 - same) * relu(cos - margin), reduction)


def hinge_embedding_loss(input, target, margin=1.0, size_average=None, reduce=None,
                         reduction="mean"):  # noqa: A002  # noqa: A002
    """`x` itself at `y=1` and `max(0, margin − x)` at `y=−1`.

    **The two are added rather than branched between.** torch puts the margin term
    where `y ≠ 1` and `x` where `y ≠ −1` and **sums them** — at ±1 only one is on
    and it matches the usual formula, and at `y=0` **both** are on (measured: 1.0
    at `x=−1` and 2.0 at `x=2`).

    Branched on `y > 0` it diverged quietly here. The loss being documented as
    taking ±1 alone is no guarantee that the values arriving are only those, and
    `sign()` produces 0.
    """
    reduction = _legacy_reduction(size_average, reduce, reduction)
    p, t = _wrap(input), _wrap(target)
    dt = p.data.dtype
    not_one = Tensor((t.data != 1).astype(dt))
    not_neg = Tensor((t.data != -1).astype(dt))
    return _reduce(not_one * relu(margin - p) + not_neg * p, reduction)


def soft_margin_loss(input, target, size_average=None, reduce=None, reduction="mean"):  # noqa: A002  # noqa: A002
    """`log(1 + e^{−y·x})`. The log and the exponential used directly overflow at
    large values, so it goes through `softplus`."""
    reduction = _legacy_reduction(size_average, reduce, reduction)
    p, t = _wrap(input), _wrap(target)
    return _reduce(softplus(-t * p), reduction)


def pairwise_distance(x1, x2, p=2.0, eps=1e-6, keepdim=False):
    """The distance between two paired rows.

    **`eps` is added to the difference, not to the result.** Asked at `p=1` where
    the difference is exactly 1.0, it gives 1.0000020 (= 1 + 2·1e-6) — read as
    added to the result it would be 1.000001 and a digit diverges. Confirmed by
    measurement.
    """
    a, b = _wrap(x1), _wrap(x2)
    diff = (a - b) + eps
    return vector_norm(diff, ord=p, dim=-1, keepdim=keepdim)


def pdist(input, p=2.0):
    """The distance between **every pair** within one batch. It gives the upper
    triangle only."""
    t = _wrap(input)
    _rank(t.data, (2,), "pdist only supports 2D tensors, got: {n}D")
    n = t.data.shape[0]
    rows = [i for i in range(n) for _ in range(i + 1, n)]
    cols = [j for i in range(n) for j in range(i + 1, n)]
    diff = t[rows] - t[cols]
    return vector_norm(diff, ord=p, dim=-1)


def triplet_margin_loss(anchor, positive, negative, margin=1.0, p=2.0, eps=1e-6,
                        swap=False, size_average=None, reduce=None, reduction="mean"):
    """`max(0, d(a,p) − d(a,n) + margin)`.

    `swap` uses `min(d(a,n), d(p,n))` instead of `d(a,n)` — when the negative is
    closer to the positive, that is the harder pair.
    """
    reduction = _legacy_reduction(size_average, reduce, reduction)
    a, pos, neg = _wrap(anchor), _wrap(positive), _wrap(negative)
    dp = pairwise_distance(a, pos, p=p, eps=eps)
    dn = pairwise_distance(a, neg, p=p, eps=eps)
    if swap:
        dn = minimum(dn, pairwise_distance(pos, neg, p=p, eps=eps))
    return _reduce(relu(dp - dn + margin), reduction)


def triplet_margin_with_distance_loss(anchor, positive, negative,
                                      distance_function=None, margin=1.0,
                                      swap=False, reduction="mean"):
    """A triplet loss taking a distance function. The default is the pairwise
    distance, so it gives the same answer as the one above."""
    a, pos, neg = _wrap(anchor), _wrap(positive), _wrap(negative)
    dist = distance_function or (lambda u, v: pairwise_distance(u, v))
    dp, dn = dist(a, pos), dist(a, neg)
    if swap:
        dn = minimum(dn, dist(pos, neg))
    return _reduce(relu(dp - dn + margin), reduction)


def multilabel_soft_margin_loss(input, target, weight=None, size_average=None,
                                reduce=None, reduction="mean"):  # noqa: A002  # noqa: A002
    """An independent binary classification per position, **averaged over the
    whole class set.**"""
    reduction = _legacy_reduction(size_average, reduce, reduction)
    p, t = _wrap(input), _wrap(target)
    each = t * logsigmoid(p) + (1 - t) * logsigmoid(-p)
    if weight is not None:
        each = each * _wrap(weight)
    return _reduce(-each.mean(dim=-1), reduction)


def multi_margin_loss(input, target, p=1, margin=1.0, weight=None, size_average=None,
                      reduce=None, reduction="mean"):  # noqa: A002  # noqa: A002
    """The margin between the correct position and the rest.

    **Divided by the class count** — not by the number of pairs compared. That
    means the correct position enters the denominator too, and dividing by the
    pair count gives 3/2 times the value at three classes.
    """
    reduction = _legacy_reduction(size_average, reduce, reduction)
    x, t = _wrap(input), _wrap(target)
    n, classes = x.data.shape
    idx = _np.arange(n)
    correct = x[idx, t.data.astype(_np.intp)].unsqueeze(1)
    each = relu(margin - correct + x) ** p
    if weight is not None:
        each = each * _wrap(weight)[t.data.astype(_np.intp)].unsqueeze(1)
    # At the correct position `margin` survives intact, so it is subtracted.
    keep = _np.ones((n, classes), dtype=x.data.dtype)
    keep[idx, t.data.astype(_np.intp)] = 0.0
    return _reduce((each * Tensor(keep)).sum(dim=1) / classes, reduction)


def multilabel_margin_loss(input, target, size_average=None, reduce=None,
                           reduction="mean"):  # noqa: A002  # noqa: A002
    """**The target is a list of positions and −1 marks the end.**

    `[3, 0, -1, 1]` means "3 and 0 are correct" and the trailing 1 is not read.
    Without that convention, −1 gets counted as one of the classes or reading
    continues past the end.
    """
    reduction = _legacy_reduction(size_average, reduce, reduction)
    x, t = _wrap(input), _wrap(target)
    rows, classes = x.data.shape
    each = []
    for r in range(rows):
        labels = []
        for v in t.data[r]:
            if v < 0:
                break
            labels.append(int(v))
        others = [c for c in range(classes) if c not in labels]
        row = None
        for i in labels:
            for j in others:
                term = relu(1 - (x[r, i] - x[r, j]))
                row = term if row is None else row + term
        each.append(Tensor(_np.zeros((), dtype=x.data.dtype)) if row is None
                    else row)
    # **One number per row, not one number.** This ran the two loops into a single
    # accumulator and reshaped it to `(1,)`, so `reduction="none"` handed back the
    # batch's *sum* under the name of the per-row losses and `mean` divided it by
    # one. Only `sum` agreed with torch, and it agreed by accident.
    #
    # The case that stood here had **a single row**, where a collapsed batch and a
    # kept one are the same array. It was written for the −1 convention and it
    # measured that; nothing above it asked for a second row until a positional
    # `reduction` case needed one.
    out = stack(each) / classes
    return _reduce(out, reduction)


# ------------------------------------------------------------ unfolding windows
#
# **`unfold` and `fold` are not each other's inverse.** `unfold` spreads the
# windows into columns and `fold` folds them back **adding where they overlap** —
# spreading a 4×4 with a 2×2 window and folding it straight back counts the middle
# four times and does not give the original. Summing is the convention.
#
# So both are built from one index. With where each cell came from written down,
# `unfold` is gathering from those positions and `fold` is **adding into** them,
# which makes one's backward the other. The same machine as the one used in the
# padding.

def _window_index(shape, kernel, dilation, padding, stride):
    """A `(C·kh·kw, L)` index table. The values are flat positions in the
    **padded** input."""
    c, h, w = shape
    kh, kw = kernel
    dh, dw = dilation
    ph, pw = padding
    sh, sw = stride
    hp, wp = h + 2 * ph, w + 2 * pw
    out_h = (hp - dh * (kh - 1) - 1) // sh + 1
    out_w = (wp - dw * (kw - 1) - 1) // sw + 1
    idx = _np.empty((c * kh * kw, out_h * out_w), dtype=_np.intp)
    row = 0
    for ch in range(c):
        for i in range(kh):
            for j in range(kw):
                col = 0
                for oh in range(out_h):
                    for ow in range(out_w):
                        y = oh * sh + i * dh
                        x = ow * sw + j * dw
                        idx[row, col] = (ch * hp + y) * wp + x
                        col += 1
                row += 1
    return idx, (out_h, out_w)


def unfold_im2col(input, kernel_size, dilation=1, padding=0, stride=1):  # noqa: A002
    """Spread the windows into columns. `(N, C, H, W)` → `(N, C·kh·kw, L)`.

    **The name collides with an existing one.** `Tensor.unfold(dim, size, step)`
    is a **view** sliding a window along one axis and this is im2col — torch has
    the same collision (`torch.Tensor.unfold` versus
    `torch.nn.functional.unfold`). Put into the module slot under the same name it
    covered the former and three `shape::unfold` cases collapsed at once. Here the
    names are kept apart and this is attached to `F.unfold` alone.
    """
    t = _mat(input, "unfold", square=False)
    # **torch takes 3-D as one unbatched sample**, and the refusal here said *anything
    # but 4-D* — half right. Measured: `(2, 3, 4)` with a 2×2 kernel comes back as
    # `(8, 6)`, which is `(C·kh·kw, L)` with no batch axis, and 5-D is refused with the
    # message repeated below.
    if t.data.ndim == 3:
        got = unfold_im2col(t.reshape(1, *t.data.shape), kernel_size, dilation,
                            padding, stride)
        return got.reshape(*got.data.shape[1:])
    if t.data.ndim != 4:
        raise RuntimeError(
            "Expected 3D or 4D (batch mode) tensor with possibly 0 batch size and "
            f"other non-zero dimensions for input, but got: {list(t.data.shape)}")
    kernel, dil = _pair(kernel_size), _pair(dilation)
    pad_, strd = _pair(padding), _pair(stride)
    padded = pad(t, (pad_[1], pad_[1], pad_[0], pad_[0]))
    n, c = t.data.shape[0], t.data.shape[1]
    idx, _ = _window_index(t.data.shape[1:], kernel, dil, pad_, strd)
    flat = padded.reshape(n, -1)
    return flat[:, idx.reshape(-1)].reshape(n, idx.shape[0], idx.shape[1])


def fold(input, output_size, kernel_size, dilation=1, padding=0, stride=1):
    """Fold what was spread back. **Overlaps are added** — that is what this
    function means."""
    t = _wrap(input)
    # **The unbatched form is 2-D here, one rank below `unfold`'s**, because this side
    # has already folded the channel and the kernel into one axis. Measured: `(8, 6)`
    # comes back as `(2, 3, 4)` and 4-D is refused.
    if t.data.ndim == 2:
        got = fold(t.reshape(1, *t.data.shape), output_size, kernel_size,
                   dilation, padding, stride)
        return got.reshape(*got.data.shape[1:])
    if t.data.ndim != 3:
        raise RuntimeError(
            "Expected 2D or 3D (batch mode) tensor for input with possibly 0 batch "
            f"size and non-zero dimensions for input, but got: {list(t.data.shape)}")
    kernel, dil = _pair(kernel_size), _pair(dilation)
    pad_, strd = _pair(padding), _pair(stride)
    out_h, out_w = _pair(output_size)
    n = t.data.shape[0]
    c = t.data.shape[1] // (kernel[0] * kernel[1])
    idx, _ = _window_index((c, out_h, out_w), kernel, dil, pad_, strd)
    hp, wp = out_h + 2 * pad_[0], out_w + 2 * pad_[1]
    flat_idx = idx.reshape(-1)

    def back(g):
        gg = _np.asarray(g).reshape(n, -1)
        return (gg[:, flat_idx].reshape(t.data.shape),)

    out = _np.zeros((n, c * hp * wp), dtype=t.data.dtype)
    _np.add.at(out, (slice(None), flat_idx), t.data.reshape(n, -1))
    made = out.reshape(n, c, hp, wp)
    if pad_[0] or pad_[1]:
        made = made[:, :, pad_[0]:hp - pad_[0], pad_[1]:wp - pad_[1]]
    return t._make(made, (t,), back, "FoldBackward0")


# --------------------------------------------------------- the rest of the layers

def bilinear(input1, input2, weight, bias=None):
    """`y[o] = x₁ᵀ·W[o]·x₂ + b[o]`. The weights have **three axes.**"""
    a, b_, w = _wrap(input1), _wrap(input2), _wrap(weight)
    out = einsum("bi,oij,bj->bo", a, w, b_)
    return out + _wrap(bias) if bias is not None else out


def local_response_norm(input, size, alpha=1e-4, beta=0.75, k=1.0):
    """Divide by the neighbouring channels.

    **The window is lopsided.** Channel `c`'s window is
    `[c − n//2, c + n − 1 − n//2]`, and at `size=2` that is `{c−1, c}` rather than
    `{c, c+1}` — confirmed by measurement. Centred, the values shift by one cell,
    and with the same size it is invisible in the shape.
    """
    t = _wrap(input)
    left = size // 2
    sq = t * t
    total = None
    for offset in range(size):
        shift = offset - left
        piece = _np.zeros_like(sq.data)
        c = sq.data.shape[1]
        src = slice(max(0, shift), min(c, c + shift))
        dst = slice(max(0, -shift), min(c, c - shift))
        piece[:, dst] = 1.0
        moved = _roll_channels(sq, shift) * Tensor(piece)
        total = moved if total is None else total + moved
    return t / (total * (alpha / size) + k) ** beta


def _roll_channels(t, shift):
    """Shift the channel axis by `shift`. What wrapped in from outside is erased
    to 0 afterwards."""
    if shift == 0:
        return t
    rolled = _np.roll(t.data, -shift, axis=1)
    return t._make(rolled, (t,),
                   lambda g: (_np.roll(_np.asarray(g), shift, axis=1),),
                   "RollBackward0")


def rrelu(input, lower=1.0 / 8, upper=1.0 / 3, training=False, inplace=False):
    """The slope on the negative side is drawn at random.

    **In evaluation mode it is fixed to the midpoint** — at the defaults,
    `(1/8 + 1/3)/2 = 0.2292`. It is drawn from `[lower, upper]` during training
    alone, so that is the only place randomness enters.
    """
    t = _wrap(input)
    if not training:
        return leaky_relu(t, (lower + upper) / 2)
    slope = _rng.uniform(lower, upper, t.data.shape).astype(t.data.dtype)
    return where(Tensor((t.data > 0).astype(t.data.dtype)), t, t * Tensor(slope))


# ------------------------------------------------------------------ rearrangement
#
# All three **move the positions without changing the values.** So the forward is
# a `reshape` plus an axis swap and the backward is the reverse — our `transpose`
# and `reshape` already do that job, so this is an assembly. With `arange` as the
# input, where each position went reads straight off the answer.

def pixel_shuffle(input, upscale_factor):
    """`(N, C·r², H, W)` → `(N, C, H·r, W·r)`. The channels are cut up and planted
    into space.

    **The interleaving order is the whole of the value.** The channels are split
    into `(C, r, r)` and the two `r`s inserted behind `H` and `W` respectively —
    standing it up as `(N, C, H, r, W, r)` and then joining is what that means.
    Reordered, the shape matches and only the picture is scrambled.
    """
    t = _wrap(input)
    r = upscale_factor
    n, c, h, w = t.data.shape
    out = t.reshape(n, c // (r * r), r, r, h, w).permute(0, 1, 4, 2, 5, 3)
    return out.reshape(n, c // (r * r), h * r, w * r)


def pixel_unshuffle(input, downscale_factor):
    """The inverse of `pixel_shuffle`. Space is cut up and stacked into the
    channels."""
    t = _wrap(input)
    r = downscale_factor
    n, c, h, w = t.data.shape
    out = t.reshape(n, c, h // r, r, w // r, r).permute(0, 1, 3, 5, 2, 4)
    return out.reshape(n, c * r * r, h // r, w // r)


def channel_shuffle(input, groups):
    """Split the channels into groups and **lay them back interleaved.**

    `[0,1,2,3]` shuffled into two groups is `[0,2,1,3]` — this is where the
    information being trapped inside its group after a grouped convolution gets
    released, so the direction of the interleaving is the whole of the value.
    """
    t = _wrap(input)
    n, c = t.data.shape[0], t.data.shape[1]
    rest = t.data.shape[2:]
    out = t.reshape(n, groups, c // groups, *rest)
    out = out.transpose(1, 2)
    return out.reshape(n, c, *rest)


# ------------------------------------------------------- channel-wise dropout
#
# **It drops channels rather than elements.** The name sitting next to `Dropout`
# makes it easy to read as "the 2-D one" and it does a different job — a channel
# is zeroed as a whole or kept as a whole.
#
# `AlphaDropout` additionally **does not insert 0.** Built to be used with SELU,
# it puts a negative constant at the dropped positions and applies an affine
# transformation over the whole thing to preserve the mean and the variance.
# Inserting 0 breaks SELU's self-normalisation, and the values are plausible, so
# it is invisible while training runs.

def _channel_mask(t, p):
    """One 0/1 drawn per channel. The spatial axes are left at 1 and
    broadcast."""
    shape = t.data.shape[:2] + (1,) * (t.data.ndim - 2)
    return (_rng.random(shape) > p).astype(t.data.dtype)


def _feature_dropout(x, p, training, name):
    t = _wrap(x)
    if not training or p == 0:
        return t
    if p >= 1:
        return t * Tensor(_np.zeros((), dtype=t.data.dtype))
    return t * Tensor(_channel_mask(t, p) / (1 - p))


def dropout1d(input, p=0.5, training=True, inplace=False):
    """These five took `inplace` from the day they were written and **threw it
    away** — the caller's tensor was never touched and nothing said so. An
    argument that is accepted and ignored is worse than one that is missing,
    because the missing one raises."""
    return _inplace_arg(input, inplace, "dropout",
                        lambda: _dropout1d_body(input, p, training))


def _dropout1d_body(x, p=0.5, training=True):
    t = _wrap(x)
    if t.data.ndim not in (2, 3):
        raise RuntimeError(
            f"dropout1d: Expected 2D or 3D input, but received a {t.data.ndim}D "
            "input. Note that dropout1d exists to provide channel-wise dropout on "
            "inputs with 1 spatial dimension, a channel dimension, and an optional "
            "batch dimension (i.e. 2D or 3D inputs).")
    return _feature_dropout(t, p, training, "dropout1d")


def dropout2d(input, p=0.5, training=True, inplace=False):
    return _inplace_arg(input, inplace, "dropout",
                        lambda: _feature_dropout(input, p, training, "dropout2d"))


def dropout3d(input, p=0.5, training=True, inplace=False):
    return _inplace_arg(input, inplace, "dropout",
                        lambda: _feature_dropout(input, p, training, "dropout3d"))


# SELU's fixed point. The value `alpha_dropout` inserts at a dropped position
# comes from here.
_ALPHA_PRIME = -1.7580993408473766


def _alpha_affine(p):
    """The affine coefficients `(a, b)` that restore the mean and the variance
    after the dropped positions are filled with a constant."""
    a = ((1 - p) * (1 + p * _ALPHA_PRIME ** 2)) ** -0.5
    return a, -a * p * _ALPHA_PRIME


def alpha_dropout(input, p=0.5, training=False, inplace=False):
    return _inplace_arg(input, inplace, "alpha_dropout",
                        lambda: _alpha_dropout_body(input, p, training))


def _alpha_dropout_body(x, p=0.5, training=False):
    t = _wrap(x)
    if not training or p == 0:
        return t
    keep = Tensor((_rng.random(t.data.shape) > p).astype(t.data.dtype))
    a, b = _alpha_affine(p)
    return (t * keep + (1 - keep) * _ALPHA_PRIME) * a + b


def feature_alpha_dropout(input, p=0.5, training=False, inplace=False):
    return _inplace_arg(input, inplace, "feature_alpha_dropout",
                        lambda: _feature_alpha_dropout_body(input, p, training))


def _feature_alpha_dropout_body(x, p=0.5, training=False):
    """`alpha_dropout` dropping whole channels."""
    t = _wrap(x)
    if not training or p == 0:
        return t
    keep = Tensor(_channel_mask(t, p))
    a, b = _alpha_affine(p)
    return (t * keep + (1 - keep) * _ALPHA_PRIME) * a + b


def _pad_index(mode, size, before, after):
    """For each output position, **which input position it reads.** `-1` marks a
    padded position.

    The four modes part here and nowhere else. The three lines below are the whole
    convention, matched position by position against real torch (`[0,1,2]`
    extended by 2 in front and 1 behind):

        reflect    2 1 [0 1 2] 1   ← mirrored at the edge, **without repeating the edge itself**
        replicate  0 0 [0 1 2] 2   ← the edge is stretched
        circular   1 2 [0 1 2] 0   ← taken from the opposite side

    Written down as one index, the forward is a `take` and the backward is
    **gathering and adding through the same index.** Writing a backward per mode
    creates four places to get it wrong.
    """
    idx = []
    for i in range(-before, size + after):
        if 0 <= i < size:
            idx.append(i)
        elif mode == "constant":
            idx.append(-1)
        elif mode == "replicate":
            idx.append(0 if i < 0 else size - 1)
        elif mode == "circular":
            idx.append(i % size)
        else:
            j = i
            while not (0 <= j < size):
                j = -j if j < 0 else 2 * (size - 1) - j
            idx.append(j)
    return _np.asarray(idx, dtype=_np.intp)


def pad(input, pad, mode="constant", value=0.0):   # noqa: A002
    """Taken from the last dimension in (before, after) order — torch's rule.

    **The pair count and the rank are interlocked.** One pair needs rank 2 or 3,
    two pairs rank 3 or 4, and three pairs rank 4 or 5 — torch refuses anything
    else with a `NotImplementedError`. Accepting any rank lets a wrongly chosen
    axis pass, so it is blocked here alongside.
    """
    input = _wrap(input)
    rank = input.data.ndim
    pairs = len(pad) // 2
    if mode != "constant" and rank not in (pairs + 1, pairs + 2):
        raise NotImplementedError(
            f"Padding size {len(pad)} is not supported for {rank}D input tensor")

    data = input.data
    steps = []
    for i in range(pairs):
        axis = rank - 1 - i
        before, after = pad[2 * i], pad[2 * i + 1]
        if before == 0 and after == 0:
            continue
        size = data.shape[axis]
        if mode == "reflect" and (before >= size or after >= size):
            raise RuntimeError(
                "Argument #4: Padding size should be less than the corresponding "
                f"input dimension, but got: pad ({before}, {after}) at dimension "
                f"{axis} of input {rank}")
        idx = _pad_index(mode, size, before, after)
        steps.append((axis, idx, size))
        taken = _np.take(data, _np.maximum(idx, 0), axis=axis)
        if mode == "constant":
            hole = idx < 0
            if hole.any():
                cut = [slice(None)] * rank
                cut[axis] = hole
                taken[tuple(cut)] = value
        data = taken

    def back(g):
        gg = _np.asarray(g)
        # Traced back from the end. Each position read from is **gathered and
        # added** — mirroring and wrapping read one input several times, so
        # overwriting loses that much.
        for axis, idx, size in reversed(steps):
            shape = list(gg.shape)
            shape[axis] = size
            out = _np.zeros(shape, dtype=gg.dtype)
            keep = idx >= 0
            head = (slice(None),) * axis
            _np.add.at(out, head + (idx[keep],), gg[head + (keep,)])
            gg = out
        return (gg,)

    return input._make(data, (input,), back, "PadBackward0")


def normalize(input, p=2, dim=1, eps=1e-12, out=None):
    """**`out` is written here rather than through `_accepts_out`.** That wrapper is
    driven by `_TAKES_OUT`, which lists names on `torch` itself, and `normalize`
    lives only under `torch.nn.functional` — putting it in that table asked
    `test_out_names.py` for `torch.normalize`, which does not exist.

    It surfaced when the core's first parameter became `input`: the row had been
    filed as *cannot be aligned* on the name, and once the names matched, what was
    actually missing showed through underneath. **A row in the vaguest bucket can be
    hiding a specific one.**
    """
    input = _wrap(input)
    denom = norm(input, p=p, dim=dim)
    got = input / maximum(denom.unsqueeze(dim),
                          Tensor(_np.array(eps, dtype=_DEFAULT_DTYPE)))
    return _out(got, out, "normalize")


def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    x1, x2 = _wrap(x1), _wrap(x2)
    return (x1 * x2).sum(dim=dim) / maximum(
        norm(x1, dim=dim) * norm(x2, dim=dim), Tensor(_np.array(eps, dtype=_DEFAULT_DTYPE)))



def tril(input, diagonal=0):
    """Keep the lower triangle alone. The backward **passes the same positions
    through** — the erased positions never appeared in the output, so their
    gradient is 0 too."""
    input = _wrap(input)
    _rank(input.data, range(2, 65), "tril: input tensor must have at least 2 dimensions")
    return input._make(_np.tril(input.data, k=diagonal), (input,),
                   lambda g: (_np.tril(_np.asarray(g), k=diagonal),), "TrilBackward0")


def triu(input, diagonal=0):
    input = _wrap(input)
    _rank(input.data, range(2, 65), "triu: input tensor must have at least 2 dimensions")
    return input._make(_np.triu(input.data, k=diagonal), (input,),
                   lambda g: (_np.triu(_np.asarray(g), k=diagonal),), "TriuBackward0")


def allclose(input, other, rtol=1e-5, atol=1e-8, equal_nan=False):
    """**It takes `equal_nan`.** The default is false, so NaN does not even equal
    NaN (measured).

    The golden harness **does not turn this on** — turned on, a NaN passes
    somewhere it must not. That and this are different places: here torch offers
    an argument and so do we, and whether to turn it on is the caller's
    decision.
    """
    return bool(_np.allclose(_wrap(input).data, _wrap(other).data, rtol=rtol, atol=atol,
                             equal_nan=bool(equal_nan)))


def equal(input, other):
    return bool(_np.array_equal(_wrap(input).data, _wrap(other).data))


def isfinite(input):
    return Tensor(_np.isfinite(_wrap(input).data))


def bincount(input, weights=None, minlength=0):
    """How many times each cell occurred. **Given weights it sums the weights
    rather than the counts.**

    The dtype parts (measured): `int64` without weights and the weights' dtype
    with them — because counting and summing values are different jobs.
    """
    input = _wrap(input)
    _refuses_bool(input.data, "bincount does not take booleans.",
                  '"bincount_cpu" not implemented for \'Bool\'',
                  kind=NotImplementedError)
    # `intp` — on wasm32, handing it int64 is refused. See `repeat_interleave`
    # above.
    w = None if weights is None else _np.asarray(_wrap(weights).data)
    out = _np.bincount(input.data.astype(_np.intp), weights=w,
                       minlength=int(minlength))
    # numpy always gives float64 when there are weights. It is restored to the
    # weights' dtype.
    if w is not None:
        return Tensor(out.astype(w.dtype))
    return Tensor(out)


# `save` and `load` used to be here — one layer over pickle. **They moved to
# `_serialize.py` and the format became safetensors.**
#
# The move happened because the three had diverged. borch.ts was safetensors from
# the start, and that file's opening paragraph gives as its reason for the format
# that "**Python `borch`, numpy and the HF tools read the same file**". And the
# Python side was using pickle, so that sentence was not true — the path from
# training in a browser to carrying it to your own machine was blocked, and that
# path is this project's only reason for choosing the format.
#
# Something is lost too. pickle carried any Python object and safetensors carries
# tensors, numbers and strings. Anything else is now **refused** — better than
# quietly failing to carry it, and being readable by somebody else is this
# format's condition to begin with.


class no_grad:
    def __enter__(self):
        self._prev = _grad_mode.enabled
        _grad_mode.enabled = False
        return self

    def __exit__(self, *exc):
        _grad_mode.enabled = self._prev
        return False

    def __call__(self, fn):
        def wrapper(*a, **k):
            with no_grad():
                return fn(*a, **k)
        return wrapper


class enable_grad:                                       # noqa: N801
    """**Turn it back on** inside `no_grad`. It has to nest, so the previous
    value is restored."""

    def __enter__(self):
        self._prev = _grad_mode.enabled
        _grad_mode.enabled = True
        return self

    def __exit__(self, *exc):
        _grad_mode.enabled = self._prev
        return False

    def __call__(self, fn):
        def wrapper(*a, **k):
            with enable_grad():
                return fn(*a, **k)
        return wrapper


class set_grad_enabled:                                  # noqa: N801
    """Takes on or off **as a value.** It works as a `with` and as a plain
    call."""

    def __init__(self, mode):
        self._prev = _grad_mode.enabled
        _grad_mode.enabled = bool(mode)
        self._mode = bool(mode)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        _grad_mode.enabled = self._prev
        return False

    def __call__(self, fn):
        def wrapper(*a, **k):
            with set_grad_enabled(self._mode):
                return fn(*a, **k)
        return wrapper


def is_grad_enabled():
    return bool(_grad_mode.enabled)


class inference_mode:                                    # noqa: N801
    """**The same as `no_grad` here.**

    In real torch this is stronger — it marks the tensors made inside so that they
    cannot enter autograd later. Imitating that mark would mean manufacturing our
    own "why can I not use this tensor" errors, so only the gradient is turned off
    here. That is why `is_inference` is always false — the mark is never attached,
    so saying it is absent is the fact.
    """

    def __init__(self, mode=True):
        self._mode = bool(mode)
        self._prev = None

    def __enter__(self):
        self._prev = _grad_mode.enabled
        if self._mode:
            _grad_mode.enabled = False
        return self

    def __exit__(self, *exc):
        _grad_mode.enabled = self._prev
        return False

    def __call__(self, fn):
        def wrapper(*a, **k):
            with inference_mode(self._mode):
                return fn(*a, **k)
        return wrapper


def is_inference(input):
    """**Always false.** As written above, the mark is never attached, so it says
    so."""
    return False


def is_inference_mode_enabled():
    return False


# ── introspection ───────────────────────────────────────────────────────────
#
# The ones that **only ask** and change no value. Textbook code branches on them,
# so without them the arithmetic is all right and it stops at that line.

def is_tensor(x):
    return isinstance(x, Tensor)


def is_storage(x):
    """**Always false.** There is no Storage layer here — the numpy array is that
    place, and calling it a Storage would be claiming an API that does not
    exist."""
    return False


def is_floating_point(input):
    return _wrap(input).data.dtype.kind == "f"


def is_signed(x):
    return _wrap(x).data.dtype.kind in "fi"


def is_nonzero(input):
    """**There has to be exactly one element.** torch throws otherwise — with
    several there is no defined answer to what is true."""
    data = _wrap(input).data
    if data.size != 1:
        raise RuntimeError(
            f"Boolean value of Tensor with {data.size} elements is ambiguous")
    return bool(data.reshape(-1)[0] != 0)


def is_same_size(a, b):
    return tuple(_wrap(a).data.shape) == tuple(_wrap(b).data.shape)


def is_distributed(x):
    """**Always false.** Inside one tab there is nothing to distribute across."""
    return False


def typename(x):
    """Gives the old-style name, as in `torch.FloatTensor`. For a non-tensor it
    is the Python type's name."""
    if not isinstance(x, Tensor):
        return type(x).__name__
    kinds = {"float32": "FloatTensor", "float64": "DoubleTensor",
             "int64": "LongTensor", "bool": "BoolTensor"}
    return "torch." + kinds.get(str(x.data.dtype), "FloatTensor")


_PROMOTE_ORDER = ("bool", "int64", "float32", "float64")


def result_type(tensor, other):
    """The dtype two **operands** would produce. torch takes tensors here, not dtypes.

    **The name that was public was the internal helper.** `_tensor.result_type`
    takes two numpy dtypes and hands back a numpy dtype, and `borch/__init__.py`
    re-exported it under torch's name — so every call shaped the way torch
    documents raised, and the one shape that worked (two dtypes) is the shape
    torch refuses. The contract was inverted end to end, and **no golden case
    mentioned the name**, which is why it sat.

    `promote_types` one function down is the dtype-to-dtype question and has
    torch's contract already; the two are easy to mistake for each other and
    torch keeps both for that reason.

    **The scalar rule is `_promote`'s, not a second table.** A Python number is
    weaker than a tensor: at or below the tensor's category it takes the tensor's
    dtype, and only above it does it rise to that category's default — which is
    why an int tensor with a Python float is `float32` and not `float64`. That is
    the rule the arithmetic already uses, and the comment beside `promote_types`
    records what happened the last time this question got a table of its own:
    `float32 + complex64` came out `float32` and no value comparison saw it.
    """
    from ._tensor import _DEFAULT_BY_CATEGORY, _promote, _scalar_category  # noqa: PLC0415, E501

    # **A dtype pair is `promote_types`'s question and torch refuses it here.**
    # Answering it anyway is the lenient direction, and lenient is diverging too:
    # `result_type(int64, float32)` would work in the browser and stop on a real
    # machine, which is the one failure a learner cannot debug from the error.
    if isinstance(tensor, dtype) or isinstance(other, dtype):
        raise TypeError(
            "result_type() received an invalid combination of arguments — it takes "
            "tensors or numbers, not dtypes. `promote_types(a, b)` is the one that "
            "takes two dtypes.")
    a, b = isinstance(tensor, Tensor), isinstance(other, Tensor)
    if a and b:
        return _NP_TO_DTYPE[_dtype_result_type(tensor.data.dtype, other.data.dtype)]
    if a:
        return _NP_TO_DTYPE[_np.dtype(_promote(tensor.data, other))]
    if b:
        return _NP_TO_DTYPE[_np.dtype(_promote(other.data, tensor))]
    # Two Python numbers. torch answers this — `result_type(1, 2.5)` is float32 —
    # and the weak-scalar rule reduces to the wider of the two categories.
    return _NP_TO_DTYPE[_np.dtype(_DEFAULT_BY_CATEGORY[
        max(_scalar_category(tensor), _scalar_category(other))])]


def promote_types(type1, type2):
    """The dtype that can hold both.

    **It uses the rule the arithmetic uses.** A separate ordering table here did
    not know about complex, so `float32 + complex64` came out float32 — the
    arithmetic had that place right and there were **two copies** of the function
    answering the same question, so only one of them was. A value comparison does
    not catch it, because whoever calls this function asks about the dtype
    alone.
    """
    return _NP_TO_DTYPE[_dtype_result_type(_np_of(type1), _np_of(type2))]


def can_cast(from_type, to_type):
    """**Category only** — bool < integer < float < complex.

    Precision is free: `float64 → float32` is true (measured). It is surprising
    that it is true when values are cut, and torch is like that, confirmed across
    all eight pairs. What is false is **narrowing the category** alone — a float
    into an integer cell, an integer into a boolean cell.

    It used to look at a table ordered by name, and that table had no complex, so
    **even complex to complex was false.** It surfaced while building `out=`.
    """
    from ._tensor import _category                         # noqa: PLC0415

    return _category(_np_of(from_type)) <= _category(_np_of(to_type))


def get_default_dtype():
    return _float32


def set_default_dtype(dt):
    """**Accepted and not acted on.** The storage here is float32 throughout, and
    pretending to change it makes the next tensor built come out in a dtype other
    than the one claimed. float32 passes quietly and anything else is refused
    loudly — ignoring it without a word is the worst of the three."""
    if getattr(dt, "name", str(dt)) != "float32":
        _unsupported(f"set_default_dtype({dt}) — the storage is float32 only")
    return None


class finfo:
    """What `torch.finfo` gives. Numbers numpy already knows, under different
    names.

    **It has to be a class.** torch's is a type and can be asked with
    `isinstance`. A wrapper function gives the same values with a different type,
    and a check that only looks at whether the name exists cannot see the
    difference — the same place as the dtype aliases sitting as functions.
    """

    def __init__(self, dt=None):
        dt = _float32 if dt is None else dt
        info = _np.finfo(getattr(dt, "np", _np.float32))
        self.eps = float(info.eps)
        self.max = float(info.max)
        self.min = float(info.min)
        self.tiny = float(info.tiny)
        self.smallest_normal = float(info.tiny)
        self.resolution = float(info.resolution)
        self.bits = int(info.bits)
        self.dtype = getattr(dt, "name", "float32")

    def __repr__(self):
        return (f"finfo(resolution={self.resolution}, min={self.min}, "
                f"max={self.max}, eps={self.eps}, "
                f"smallest_normal={self.smallest_normal}, tiny={self.tiny}, "
                f"dtype={self.dtype})")


class iinfo:
    """A class for `finfo`'s reason."""

    def __init__(self, dt):
        info = _np.iinfo(getattr(dt, "np", _np.int64))
        self.max = int(info.max)
        self.min = int(info.min)
        self.bits = int(info.bits)
        self.dtype = getattr(dt, "name", "int64")

    def __repr__(self):
        return (f"iinfo(min={self.min}, max={self.max}, dtype={self.dtype})")





class _Namespace:
    """A submodule slot of torch (`torch.nn`, `torch.optim.lr_scheduler`, …).

    An object rather than a Python module, and once `install()` walks it and
    plants it into `sys.modules`, an import such as
    `from torch.optim.lr_scheduler import StepLR` goes straight through.
    Inheritance is the only mark — a place that did not come in here gets no
    import path.
    """


# ------------------------------------------------- linear algebra (factorisations)
#
# numpy does all of it. **Gradients go in only where there is a closed form** —
# `det`, `logdet`, `inverse`, `solve` and `cholesky` are those, and the five were
# derived and compared against torch (Cholesky at a max diff of 2.8e-17).
#
# `qr`, `svd`, `pinverse` and `lstsq` give values alone. torch differentiates
# these too and this does not — the derivation is delicate (especially with
# repeated singular values) and getting it wrong is wrong quietly. `backward()`
# refuses, so the absence surfaces loudly.

class LinAlgError(RuntimeError):
    """torch's `linalg.LinAlgError`.

    **The name does a job.** Code that meets a singular matrix wraps it in
    `except linalg.LinAlgError`, and throwing something else means passing that
    wrapper by and the program dies. numpy puts it under `ValueError` and torch
    under `RuntimeError`, and this follows torch.
    """


def _named(kind, *fields):
    """Build a named result.

    **torch's linalg can be asked positionally or by name** — `slogdet(A)[1]` and
    `slogdet(A).logabsdet` are the same thing. Matching the positions alone, the
    values are right and textbook code stops at the attribute access. `lstsq` had
    already been through that with `.solution`, and a class was written by hand at
    the time — rather than write it eight more times, they are stamped out here.
    """
    class _R:
        __slots__ = fields

        def __init__(self, *vals):
            for f, v in zip(fields, vals):
                setattr(self, f, v)

        def __iter__(self):
            for f in fields:
                yield getattr(self, f)

        def __len__(self):
            return len(fields)

        def __getitem__(self, i):
            return getattr(self, fields[i])

        def __repr__(self):
            inner = ", ".join(f"{f}={getattr(self, f)!r}" for f in fields)
            return f"torch.return_types.{kind}(\n{inner})"

    _R.__name__ = _R.__qualname__ = kind
    return _R


_Slogdet = _named("slogdet", "sign", "logabsdet")
_QR = _named("linalg_qr", "Q", "R")
_SVD = _named("linalg_svd", "U", "S", "Vh")
# **`torch.svd` and `torch.linalg.svd` are two functions, and the third field is
# not the same matrix.** `linalg.svd` gives `Vh`; `torch.svd` gives `V`, its
# transpose. The fields are spelled out so an attribute access says which one it
# reached rather than quietly handing over the other.
_TorchSVD = _named("svd", "U", "S", "V")
_Eigh = _named("linalg_eigh", "eigenvalues", "eigenvectors")
_Eig = _named("linalg_eig", "eigenvalues", "eigenvectors")
_Lstsq = _named("linalg_lstsq", "solution", "residuals", "rank", "singular_values")
_LuFactor = _named("linalg_lu_factor", "LU", "pivots")
_Lu = _named("linalg_lu", "P", "L", "U")
_InvEx = _named("linalg_inv_ex", "inverse", "info")
_CholeskyEx = _named("linalg_cholesky_ex", "L", "info")
_SolveEx = _named("linalg_solve_ex", "result", "info")
_LuFactorEx = _named("linalg_lu_factor_ex", "LU", "pivots", "info")
_LdlFactor = _named("linalg_ldl_factor", "LD", "pivots")
_LdlFactorEx = _named("linalg_ldl_factor_ex", "LD", "pivots", "info")
_Geqrf = _named("geqrf", "a", "tau")
_Frexp = _named("frexp", "mantissa", "exponent")
# The statistics results. `histogram` and `histogramdd` have a parameter named
# `range`, which shadows the Python builtin inside them — the ninth time in this
# file, so an alias is put down in advance.
_Histogram = _named("histogram", "hist", "bin_edges")
_HistogramDD = _named("histogramdd", "hist", "bin_edges")
_Mode = _named("mode", "values", "indices")
_NanMedian = _named("nanmedian", "values", "indices")
_builtin_range = range
# The top-level linear algebra results. **`triangular_solve` gives two things and
# the second is a copy of the coefficient matrix** (measured) — it looks useless,
# and torch gives it that way, so the positions are matched.
_TriangularSolve = _named("triangular_solve", "solution", "cloned_coefficient")
_LuInfos = _named("lu", "LU", "pivots", "info")
_LuUnpack = _named("lu_unpack", "P", "L", "U")
_Lobpcg = _named("lobpcg", "eigenvalues", "eigenvectors")
_SvdLowrank = _named("svd_lowrank", "U", "S", "V")


# ---- batching
#
# **Every torch `linalg` function is batched.** `det((3,2,2))` gives `(3,)`, and
# so do `inv`, `solve`, `cholesky`, `slogdet` and `matrix_rank`. The earlier
# `_mat` refused anything that was not 2-D, and that was an absence rather than an
# imitation — batching is the shape real code always uses.
#
# numpy's `linalg` also treats the last two axes as the matrix and loops the rest
# as a batch. So the forward opens up almost as it is and **the work is in the
# backward.** A transpose written as `.T` is right at 2-D alone and reverses every
# axis in a batch, quietly wrong. Everything below uses `_T`.

def _T(a):
    """Swap the last two axes alone. The batch axes stay where they are."""
    return _np.swapaxes(a, -1, -2)


def _mat(t, what, square=True):
    t = _wrap(t)
    if t.data.ndim < 2:
        _unsupported(f"{what} (fewer than 2 dimensions)")
    if square and t.data.shape[-1] != t.data.shape[-2]:
        _unsupported(f"{what} (a non-square matrix)")
    return t


def _guard(what, fn, *args, **kw):
    """Re-label what numpy throws on a singular matrix under our name."""
    try:
        return fn(*args, **kw)
    except _np.linalg.LinAlgError as exc:
        raise LinAlgError(f"linalg.{what}: {exc}") from None


def _is_singular(data):
    """Is it singular — **the decision is ours.**

    Left to numpy, the answer depends on **what numpy was built against.** On the
    same input `[[1,2],[2,4]]`, native numpy threw a `LinAlgError` and numpy
    inside Pyodide passed quietly — for `inv` and for `cholesky` alike (measured).
    Then "what happens on a singular matrix" **depends on the user's browser.**
    That is the library's to decide and not the underlying LAPACK's.

    An exact 0 on the diagonal of the partially pivoted LU means singular — the
    same criterion LAPACK uses, and the arithmetic being ours it gives the same
    answer wherever it runs.
    """
    packed, _ = _lu_pack(data)
    k = min(packed.shape[-2], packed.shape[-1])
    idx = _np.arange(k)
    return bool(_np.any(packed[..., idx, idx] == 0))


def _reject_singular(data, what):
    if _is_singular(data):
        raise LinAlgError(
            f"linalg.{what}: the matrix is singular — it has no inverse (The diagonal "
            "element of the factorization is zero)")


def _cholesky_checked(data, what):
    """Cholesky. **The positive-definiteness check is ours too** — the reason
    above."""
    try:
        low = _np.linalg.cholesky(data)
    except _np.linalg.LinAlgError:
        low = None
    if low is None or not _np.all(low[..., _np.arange(low.shape[-1]),
                                      _np.arange(low.shape[-1])] > 0):
        raise LinAlgError(
            f"linalg.{what}: the input is not symmetric positive definite (matrix is not "
            "positive definite)")
    return low


# **The inverse is computed in the backward.** The earlier version computed it
# ahead of time in the forward, and then `det(a singular matrix)` cannot give 0 and
# throws — torch gives 0 quite happily. What cannot be differentiated can say so
# at the point of differentiation, and there is no reason to block somebody asking
# for a value as well.

def det(input):
    input = _mat(input, "det")
    out = _np.linalg.det(input.data)

    def back(g):
        inv_t = _T(_guard("det", _np.linalg.inv, input.data))
        # The determinant is a scalar per batch — multiplying it into a matrix
        # needs two axes stood up.
        return ((_np.asarray(g) * out)[..., None, None] * inv_t,)

    return input._make(_np.asarray(out, dtype=input.data.dtype), (input,), back,
                       "DetBackward0")


def logdet(input):
    input = _mat(input, "logdet")
    sign, logabs = _np.linalg.slogdet(input.data)
    out = _np.where(sign > 0, logabs, _np.nan)
    return input._make(_np.asarray(out, dtype=input.data.dtype), (input,),
                   lambda g: (_np.asarray(g)[..., None, None]
                              * _T(_guard("logdet", _np.linalg.inv, input.data)),),
                   "LogdetBackward0")


def slogdet(input):                                          # noqa: A002
    """`input` at the top level and `A` under `linalg` — see `_linalg_A` below."""
    A = _mat(input, "slogdet")                               # noqa: N806
    sign, logabs = _np.linalg.slogdet(A.data)
    return _Slogdet(Tensor(_np.asarray(sign, dtype=A.data.dtype)),
                    A._make(_np.asarray(logabs, dtype=A.data.dtype), (A,),
                            lambda g: (_np.asarray(g)[..., None, None]
                                       * _T(_guard("slogdet", _np.linalg.inv, A.data)),),
                            "SlogdetBackward0"))


def inverse(A):  # noqa: N803
    """The inverse. Its gradient is `-A⁻ᵀ G A⁻ᵀ`."""
    A = _mat(A, "inverse")
    _reject_singular(A.data, "inv")
    out = _guard("inv", _np.linalg.inv, A.data)
    out_t = _T(out)
    return A._make(out, (A,),
                   lambda g: (-(out_t @ _np.asarray(g) @ out_t),), "InverseBackward0")


def inv_ex(A, check_errors=False):  # noqa: N803
    """The same as `inv` and **it does not throw** — it puts a non-zero number in
    `info` instead.

    The distinction earns its keep in a batch. With one of twenty matrices
    singular, the throwing side kills all of them and this says which one is bad
    through `info`.
    """
    A = _mat(A, "inv_ex")
    try:
        _reject_singular(A.data, "inv_ex")
        out = _np.linalg.inv(A.data)
        info = _np.zeros(A.data.shape[:-2], dtype=_np.int32)
    except LinAlgError:
        if check_errors:
            raise
        out = _np.full_like(A.data, _np.inf)
        # LAPACK records which leading pivot is zero. This says only "bad".
        info = _np.full(A.data.shape[:-2], _SINGULAR_INFO, dtype=_np.int32)
        return _InvEx(Tensor(out), Tensor(info))
    return _InvEx(A._make(out, (A,),
                          lambda g: (-(_T(out) @ _np.asarray(g) @ _T(out)),),
                          "InverseBackward0"), Tensor(info))


# Matched to the magnitude LAPACK records on a singular matrix. Measured: 2 for a
# 2×2 singular matrix.
_SINGULAR_INFO = 2


def solve(A, b):  # noqa: N803
    """Solve `A x = b`. More accurate and faster than building the inverse and
    multiplying.

    When `b` has one axis fewer than `A` it is read as **a batch of vectors.** The
    outer product in the backward hangs on that distinction.

    **A place that must not be left to numpy.** numpy 2.0 changed the rule and now
    reads a batch of vectors only when `b` is 1-D — given `A(3,2,2)` and `b(3,2)`
    it reads a matrix and throws about mismatched dimensions. torch keeps the old
    rule. So the axis is stood up here.
    """
    A = _mat(A, "solve")
    _reject_singular(A.data, "solve")
    bt = _wrap(b)
    vector = bt.data.ndim == A.data.ndim - 1
    rhs = bt.data[..., None] if vector else bt.data
    x = _guard("solve", _np.linalg.solve, A.data, rhs)
    if vector:
        x = x[..., 0]
    inv_t = _T(_guard("solve", _np.linalg.inv, A.data))

    def back(g):
        gg = _np.asarray(g)
        if vector:
            gb = (inv_t @ gg[..., None])[..., 0]
            ga = -(gb[..., :, None] * x[..., None, :])
        else:
            gb = inv_t @ gg
            ga = -(gb @ _T(x))
        return (ga, gb)

    return A._make(x, (A, bt), back, "SolveBackward0")


def solve_ex(A, b, check_errors=False):  # noqa: N803
    """`solve`'s non-throwing side."""
    A = _mat(A, "solve_ex")
    try:
        out = solve(A, b)
    except LinAlgError:
        if check_errors:
            raise
        bt = _wrap(b)
        return _SolveEx(Tensor(_np.full_like(bt.data, _np.inf)),
                        Tensor(_np.full(A.data.shape[:-2], _SINGULAR_INFO,
                                        dtype=_np.int32)))
    return _SolveEx(out, Tensor(_np.zeros(A.data.shape[:-2], dtype=_np.int32)))


def _cholesky_raw(data, upper):
    low = _np.linalg.cholesky(data.astype(_np.float64))
    return _T(low) if upper else low


def cholesky(input, upper=False):
    """The lower triangle `L` of `A = L Lᵀ`. **It has a gradient** — Murray's
    algorithm was derived and compared against torch at a max diff of
    2.8e-17."""
    input = _mat(input, "cholesky")
    low = _cholesky_checked(input.data.astype(_np.float64), "cholesky")
    idx = _np.arange(low.shape[-1])

    def back(g):
        gg = _np.asarray(g, dtype=_np.float64)
        if upper:
            gg = _T(gg)
        bar = _T(low) @ gg
        half = _np.tril(bar).copy()
        # The diagonal alone is halved. In a batch `diag_indices_from` picks the
        # wrong axes.
        half[..., idx, idx] *= 0.5
        low_inv = _np.linalg.inv(low)
        sym = _T(low_inv) @ half @ low_inv
        return (((sym + _T(sym)) * 0.5).astype(input.data.dtype),)

    out = (_T(low) if upper else low).astype(input.data.dtype)
    return input._make(out, (input,), back, "CholeskyBackward0")


def cholesky_ex(input, upper=False, check_errors=False):  # noqa: A002
    """`cholesky`'s non-throwing side. `info` is non-zero when it is not positive
    definite."""
    input = _mat(input, "cholesky_ex")
    try:
        out = cholesky(input, upper=upper)
    except LinAlgError:
        if check_errors:
            raise
        return _CholeskyEx(Tensor(_np.full_like(input.data, _np.nan)),
                           Tensor(_np.full(input.data.shape[:-2], _SINGULAR_INFO,
                                           dtype=_np.int32)))
    return _CholeskyEx(out, Tensor(_np.zeros(input.data.shape[:-2], dtype=_np.int32)))


def matrix_power(input, n):
    """**Built by chaining multiplications** — the backward then follows on its
    own. Built from a factorisation there would be a new derivative to write, and
    that is one more place to get it wrong."""
    input = _mat(input, "matrix_power")
    if n < 0:
        return matrix_power(inverse(input), -n)
    if n == 0:
        eye = _np.eye(input.data.shape[-1], dtype=input.data.dtype)
        return Tensor(_np.broadcast_to(eye, input.data.shape).copy())
    out = input
    for _ in range(n - 1):
        out = out @ input
    return out


# ---- the gradients of the factorisations
#
# They went in late, for a reason — the derivations are delicate and getting one
# wrong is wrong **quietly.** The kind where the values are right and only the
# training diverges subtly, so leaving the absence loud was better.
#
# They go in now. What changed is not that the derivations got easier but that
# **there is something to compare against.** The golden holds real torch's numbers
# position by position, so getting one wrong is wrong loudly rather than
# quietly.

def qr(input, some=True):                                    # noqa: A002
    """`torch.qr` — **which is not `torch.linalg.qr`**, in the same way
    `svd` below is not `linalg.svd`.

    The factorisation is one thing asked for in two vocabularies, and both are
    torch's. This door takes `input` and a boolean; the other takes `A` and a
    string. It was one function with `mode=` on both, so `x.qr(some=False)` —
    a line torch code contains — raised on the keyword.

        some=True   ->  mode="reduced"
        some=False  ->  mode="complete"
    """
    return _qr_impl(input, "reduced" if some else "complete")


def _qr_impl(input, mode="reduced"):                         # noqa: A002
    """The arithmetic both doors reach.

        N = Qᵀ·Q̄ − R̄·Rᵀ
        Ā = [Q̄ + Q·(tril(N − Nᵀ, −1) − N)]·R⁻ᵀ

    Keeping the lower triangle alone is the whole of this derivation.
    Differentiating `QᵀQ = I` makes `C = Qᵀ·dQ` **antisymmetric**, and overlaid
    with `dR·R⁻¹` being upper triangular, `C`'s freedom is left in the lower
    triangle alone. The upper side is its mirror and there is nothing separate to
    compute.

    **Resolving `R⁻ᵀ` as `R⁻¹` by mistake is quietly wrong here.** It really was
    wrong that way, and with all eight candidates failing to match the derivation
    came under suspicion when the trouble was the transpose rather than the
    derivation. `X·R⁻ᵀ` is `solve(R, Xᵀ)ᵀ`, not `solve(Rᵀ, Xᵀ)ᵀ`.
    """
    A = _mat(input, "qr", square=False)                      # noqa: N806
    q, r = _np.linalg.qr(A.data, mode=mode)
    if mode != "reduced" or A.data.shape[-2] < A.data.shape[-1]:
        # The complete form leaves extra columns in `Q` and no information flows
        # towards them — a different derivation.
        return _QR(Tensor(_np.ascontiguousarray(q)), Tensor(_np.ascontiguousarray(r)))

    def back_from(gq, gr):
        n = _T(q) @ gq - gr @ _T(r)
        inner = _np.tril(n - _T(n), -1) - n
        return _T(_np.linalg.solve(r, _T(gq + q @ inner)))

    qt = A._make(_np.ascontiguousarray(q), (A,),
                 lambda g: (back_from(_np.asarray(g), _np.zeros_like(r)),),
                 "QrBackward0")
    rt = A._make(_np.ascontiguousarray(r), (A,),
                 lambda g: (back_from(_np.zeros_like(q), _np.asarray(g)),),
                 "QrBackward0")
    return _QR(qt, rt)


def _svd_raw(data, full_matrices):
    u, s, vh = _np.linalg.svd(data, full_matrices=full_matrices)
    return _np.ascontiguousarray(u), s, _np.ascontiguousarray(vh)


def svd(input, some=True, compute_uv=True):
    """`torch.svd` — **which is not `torch.linalg.svd`,** and this was the latter
    under the former's name.

    Three things part between them, and all three were wrong here:

    - **The default is reduced, not full.** `torch.svd` defaults `some=True`, so a
      3×2 input gives a 3×2 `U`; `linalg.svd` defaults `full_matrices=True` and
      gives 3×3. `borch.svd(x)` and `torch.svd(x)` came back **different shapes
      from the same call**, and the values in the overlapping block agreed, so
      anything that read `U[:, :k]` or only `S` saw nothing wrong.
    - **The third field is `V`, not `Vh`.** They are transposes. The docstring here
      claimed "(U, S, Vh) order as in torch", which was true about `linalg.svd` and
      written under `svd` — a reason true about one thing standing where a reason
      about another belongs.
    - **The argument is `some`, and it means the opposite of `full_matrices`.**
      Named `full_matrices`, a caller porting from torch passes `False` for the
      reduced form and gets the full one.

    `borch.linalg.svd` is untouched and still takes `full_matrices`. Two functions,
    two signatures, as torch has them.

    **The singular values have a gradient and `U` and `V` do not.**
    `dS = diag(Uᵀ dA V)`, so the singular value side is one line with no
    repeated-value trouble. The vector side carries a `1/(sᵢ²−sⱼ²)` and blows up
    where singular values repeat, and that place is left out — the absence being
    loud is better.
    """
    input = _mat(input, "svd", square=False)
    u, s, vh = _svd_raw(input.data, not some)
    k = s.shape[-1]
    u_thin, vh_thin = u[..., :, :k], vh[..., :k, :]

    def back(g):
        gg = _np.asarray(g)
        idx = _np.arange(k)
        mid = _np.zeros(gg.shape + (k,), dtype=u.dtype)
        mid[..., idx, idx] = gg
        return (u_thin @ mid @ vh_thin,)

    values = input._make(s, (input,), back, "SvdBackward0")
    if not compute_uv:
        # torch fills both with zeros of the shape they would have had rather than
        # returning a shorter tuple — the caller's unpacking keeps working.
        return _TorchSVD(Tensor(_np.zeros_like(u)), values,
                         Tensor(_np.zeros_like(_np.swapaxes(vh, -1, -2))))
    return _TorchSVD(Tensor(u), values, Tensor(_np.ascontiguousarray(
        _np.swapaxes(vh, -1, -2))))


def linalg_svd(A, full_matrices=True):  # noqa: N803
    """`torch.linalg.svd` — **the other one.** It gives `Vh` and defaults to the
    full form, where `torch.svd` gives `V` and defaults to the reduced one.

    Both names pointed at a single function for a long time and `_Linalg` said so:
    "It points at the same implementation, so there is nowhere to diverge." That
    sentence had already stopped being true three names earlier — `lu`, `lu_solve`
    and `vander` all diverge and the note about them sits in `__init__.py`, in a
    different file from the claim it contradicts.
    """
    A = _mat(A, "linalg.svd", square=False)
    u, s, vh = _svd_raw(A.data, full_matrices)
    k = s.shape[-1]
    u_thin, vh_thin = u[..., :, :k], vh[..., :k, :]

    def back(g):
        gg = _np.asarray(g)
        idx = _np.arange(k)
        mid = _np.zeros(gg.shape + (k,), dtype=u.dtype)
        mid[..., idx, idx] = gg
        return (u_thin @ mid @ vh_thin,)

    return _SVD(Tensor(u), A._make(s, (A,), back, "SvdBackward0"), Tensor(vh))


def pinverse(input, rcond=1e-15):
    """The pseudo-inverse. **It has a gradient** — in three terms.

        Ā = −Pᵀ·Ḡ·Pᵀ + (I − A·P)·Ḡᵀ·P·Pᵀ + Pᵀ·P·Ḡᵀ·(I − P·A)

    **The last two terms vanish for a square non-singular matrix** — there
    `I − AP` and `I − PA` are both 0 and only the first term survives, and that
    first term is the inverse's gradient formula. So leaving the two out is
    **right on squares and wrong on rectangles alone.** It really was wrong that
    way, and the square cases had been passing all along — which is why the golden
    asks with rectangles too.
    """
    input = _mat(input, "pinverse", square=False)
    p = _np.linalg.pinv(input.data, rcond=rcond)
    m, n = input.data.shape[-2], input.data.shape[-1]
    eye_m = _np.eye(m, dtype=p.dtype)
    eye_n = _np.eye(n, dtype=p.dtype)

    def back(g):
        gg = _np.asarray(g)
        pt = _T(p)
        left = -(pt @ gg @ pt)
        mid = (eye_m - input.data @ p) @ _T(gg) @ p @ pt
        right = pt @ p @ _T(gg) @ (eye_n - p @ input.data)
        return (left + mid + right,)

    return input._make(p, (input,), back, "PinverseBackward0")


def matrix_rank(input, tol=None):  # noqa: A002
    input = _mat(input, "matrix_rank", square=False)
    return Tensor(_np.asarray(_np.linalg.matrix_rank(input.data, tol=tol), dtype=_np.int64))


def eigh(input, UPLO="L"):  # noqa: A002
    """The eigenvalues and eigenvectors of a symmetric matrix. **Both have
    gradients.**

    **It reads one triangle only.** The default is the lower one, so
    `[[4,99],[1,3]]` and `[[4,1],[1,3]]` give the same answer (confirmed by asking
    real torch). A convention that never surfaces as long as symmetric input is
    given, so an implementation looking at the whole matrix diverges from this
    quietly.

    The eigenvalue side is one line, `Ā = V·diag(ḡ)·Vᵀ`. The eigenvector side is
    `Ā = V·(F ∘ (Vᵀ·Ḡ))·Vᵀ` with `F_ij = 1/(λⱼ − λᵢ)` — **it blows up where
    eigenvalues repeat.** torch blows up alongside, so this is the same limit
    rather than an imitation.
    """
    input = _mat(input, "eigh")
    w, v = _np.linalg.eigh(input.data, UPLO=UPLO)
    v = _np.ascontiguousarray(v)
    vt = _T(v)

    def back_values(g):
        gg = _np.asarray(g)
        idx = _np.arange(w.shape[-1])
        mid = _np.zeros(gg.shape + (w.shape[-1],), dtype=v.dtype)
        mid[..., idx, idx] = gg
        return (v @ mid @ vt,)

    def back_vectors(g):
        gg = _np.asarray(g)
        gap = w[..., None, :] - w[..., :, None]
        idx = _np.arange(w.shape[-1])
        # The diagonal is left at 0 — it is a difference with itself, so nothing
        # flows there by definition rather than by division.
        with _np.errstate(divide="ignore", invalid="ignore"):
            f = _np.where(gap == 0, 0.0, 1.0 / _np.where(gap == 0, 1.0, gap))
        f[..., idx, idx] = 0.0
        raw = v @ (f * (vt @ gg)) @ vt
        # **The symmetrisation must not be left out.** `A` being symmetric, the
        # upper and lower triangles share the same freedom, and the raw formula
        # piles it onto one side. The diagonal is right and only the off-diagonal
        # parts, so it is invisible without a value comparison — chosen by
        # measurement.
        return ((raw + _T(raw)) * 0.5,)

    return _Eigh(input._make(w, (input,), back_values, "EighBackward0"),
                 input._make(v, (input,), back_vectors, "EighBackward0"))


def eig(input):  # noqa: A002
    """The eigenvalues and eigenvectors of **a non-symmetric matrix.** The answer
    is complex.

    It sits next to `eigh` and is a different function — that one takes symmetric
    matrices only, reads one triangle, and its answer is real. This takes any
    square matrix, and since some of them **have no real eigenvalues at all, a
    rotation matrix for instance**, its answer is always complex (measured:
    `[[0,-1],[1,0]]` gives ±i).

    **The reason this function was absent was written in the table as "there is no
    complex dtype".** That stopped being true once complex64 went in, and all the
    while `x.eig()`'s refusal message said "use `linalg.eig`" and **pointed at
    something that did not exist.**

    ## The gradient

    The eigenvalue side is `Ā = V⁻ᴴ·diag(λ̄)·Vᴴ`, and the eigenvector side adds
    `F ∘ (Vᴴ·V̄)` inside it with `F_ij = 1/(λⱼ − λᵢ)`. **It blows up where
    eigenvalues repeat** — `eigh`'s limit, and torch blows up alongside.

    Real input has to give a real answer, so only the real part is taken at the
    end. The complex gradient convention is `∂L/∂re + i·∂L/∂im`, so the imaginary
    part means nothing at that point.
    """
    input = _mat(input, "eig")
    # **The backward is computed in double precision.** The answer comes out as
    # complex64, and the `1/(λⱼ − λᵢ)` inside it grows as the eigenvalues come
    # closer, so computed in single precision it diverges from torch at 3×3
    # already (measured). The arithmetic ahead of it stays double and it is cut
    # **only on the way out.**
    w64, v64 = _np.linalg.eig(input.data.astype(_np.float64))
    v64 = _np.ascontiguousarray(v64)
    w, v = w64.astype(_np.complex64), v64.astype(_np.complex64)
    vh = _np.conjugate(_T(v64))
    vinv_h = _np.conjugate(_T(_np.linalg.inv(v64)))

    def _pull(mid):
        return (_np.real(vinv_h @ mid @ vh).astype(input.data.dtype),)

    def back_values(g):
        gg = _np.asarray(g, dtype=_np.complex128)
        mid = _np.zeros_like(v64)
        idx = _np.arange(w.shape[-1])
        mid[..., idx, idx] = gg
        return _pull(mid)

    def back_vectors(g):
        gg = _np.asarray(g, dtype=_np.complex128)
        # **It is determined only up to a phase.** A complex eigenvector is still
        # an eigenvector multiplied by `e^{iφ}`, so a loss that leans on that
        # phase **has no defined value.** torch stops here too, so this is the
        # same limit rather than an imitation — not stopping would make us the
        # permissive one, and being more permissive is still diverging (code that
        # ran here stops against torch).
        #
        # The test is whether the diagonal of `Vᴴ·V̄` is real. A surviving
        # imaginary part means that loss looks at the phase.
        probe = _np.sum(_np.conjugate(v64) * gg, axis=-2)
        if _np.abs(_np.imag(probe)).max() > 1e-5:
            raise RuntimeError(
                "linalg_eig_backward: The eigenvectors in the complex case are "
                "specified up to multiplication by e^{i phi}. The specified loss "
                "function depends on this quantity, so it is ill-defined.")
        # **The length is pinned at 1.** LAPACK gives the eigenvectors at unit
        # length, so they cannot move along their own direction — without
        # subtracting that component the gradient flows into a freedom that does
        # not exist. Before and after subtracting were both compared against
        # torch, and **on a symmetric matrix the two answers agree**, so measured
        # on symmetric input alone the missing term is invisible (measured).
        gg = gg - v64 * _np.real(_np.sum(_np.conjugate(v64) * gg, axis=-2))
        gap = w64[..., None, :] - w64[..., :, None]
        idx = _np.arange(w.shape[-1])
        with _np.errstate(divide="ignore", invalid="ignore"):
            f = _np.where(gap == 0, 0.0, 1.0 / _np.where(gap == 0, 1.0, gap))
        f[..., idx, idx] = 0.0
        return _pull(f * (vh @ gg))

    return _Eig(input._make(w, (input,), back_values, "LinalgEigBackward0"),
                input._make(v, (input,), back_vectors, "LinalgEigBackward0"))


def eigvals(input):  # noqa: A002
    """`eig`'s eigenvalues alone. Where the eigenvectors are not used, this is
    torch's name."""
    return eig(input).eigenvalues


# The four LAPACK least-squares routines torch will name. Measured from the
# message torch raises on anything else.
_LSTSQ_DRIVERS = ("gels", "gelsy", "gelsd", "gelss")


def lstsq(input, b, rcond=None, *, driver=None):  # noqa: A002
    """The least-squares solution. **Values alone.**

    It has to be asked through `.solution` — torch gives the residuals, the rank
    and the singular values alongside the solution. Handing back a bare tensor
    makes torch code stop at `.solution`, and that becomes a place where "the
    value is right and it does not work".

    **`input`/`b`, not `A`/`B`.** torch's own docstring writes the signature as
    `lstsq(A, B, rcond=None, *, driver=None)` and the overload underneath refuses
    both names — `A=`/`B=` raise *missing 2 required positional argument: "input",
    "b"*. Measured, not read. `tests/torch_signatures_core.py` keeps the four this
    axis has caught in `TORCH_DOC_IS_WRONG`, and this one is not among them: that
    axis reads `linalg` off the docstring on both sides, so a disagreement between
    torch's prose and torch's own overload is the one thing it cannot see.

    The measurement nearly cost a rename in the wrong direction: the sweep that
    compares parameter names against torch reported this row as the one divergence
    in `linalg`, and the divergence was **in the reference.** Renaming to match it
    would have turned the only agreement into a mismatch.

    **`driver` decides what comes back, so it could not be swallowed.** The first
    version of this took the argument and ignored it, on the reasoning that numpy
    exposes one path and every driver gives the same solution. The solution, yes —
    but not the other three fields, and torch was measured rather than assumed:

        driver          residuals   rank     singular_values
        gels            numpy's     empty    empty
        gelsy (default) empty       scalar   empty
        gelsd, gelss    numpy's     scalar   numpy's

    Accepting a knob and dropping it is the shape this repository keeps naming —
    *the value is right and it does not work* — except one turn worse, because
    nothing raises. `lstsq(A, B, driver="gelsd").singular_values` would have come
    back the right numbers by accident and `driver="gelsy"` the wrong ones.

    **The default was already wrong here.** Passing numpy's four fields straight
    through matches `gelsd`, and torch's CPU default is `gelsy` — so `residuals`
    and `singular_values` were arriving filled where torch hands back empty
    tensors. Reading the table fixed a divergence that predates the argument.

    `rcond` sets the cutoff below which a singular value counts as zero, which is
    numpy's own `rcond` and is what moves `rank`. Under `gels` there is no rank
    estimate to move.

    **The table above is about the other three fields. `solution` has its own
    story, and it was wrong in two places.** numpy exposes one path, the SVD one,
    which is `gelsd`. At full rank all four of torch's drivers agree to float
    noise and nothing shows. Once `rcond` actually cuts a singular value they
    separate, measured on `[[1,1],[1,2],[1,3],[1,4]]` against `[6,5,7,10]` with
    `rcond=0.9`:

        gels             3.500, 1.400   — the cutoff is not applied at all
        gelsy (default)  0.770, 2.310   — a rank-revealing QR
        gelsd, gelss     0.790, 2.322   — the SVD, which is numpy's

    So `gels` now passes `rcond=None` down: torch's `gels` assumes full rank and
    has no cutoff to apply, and applying one was **a wrong number under an
    argument that reads as a tuning knob** — 3.5 against 0.79, which is not a
    tolerance away from anything.

    And `gelsy` — *the default*, so this is what an unadorned call gets — is
    refused when the cutoff bites. There is no pivoted QR in numpy to produce it
    with; the alternative is handing back the SVD's answer under the name of a
    different algorithm, and the two differ by 2.5% here. At full rank the
    cutoff changes nothing and the call goes through, which is every ordinary
    use of it.
    """
    if driver is not None and driver not in _LSTSQ_DRIVERS:
        raise RuntimeError(
            "torch.linalg.lstsq: parameter `driver` should be one of "
            "(gels, gelsy, gelsd, gelss)")
    driver = driver or "gelsy"
    input, bt = _mat(input, "lstsq", square=False), _wrap(b)
    lead, vector = _lstsq_batch(input.data.shape, bt.data.shape)
    if lead is None:
        return _lstsq_one(input.data, bt.data, driver, rcond)
    a_all = _np.broadcast_to(input.data, (*lead, *input.data.shape[-2:]))
    b_tail = bt.data.shape[-1:] if vector else bt.data.shape[-2:]
    b_all = _np.broadcast_to(bt.data, (*lead, *b_tail))
    parts = [_lstsq_one(a_all[at], b_all[at], driver, rcond)
             for at in _np.ndindex(*lead)]
    # **An empty field stays empty rather than becoming a stack of empties.** torch
    # gives `rank` as `(0,)` under `gels` whether or not there is a batch, and
    # stacking would turn the absence into a shape.
    def stacked(get):
        rows = [_np.asarray(get(p).data) for p in parts]
        if rows[0].size == 0 and rows[0].ndim == 1:
            return rows[0]
        return _np.stack(rows).reshape(*lead, *rows[0].shape)
    return _Lstsq(Tensor(stacked(lambda p: p.solution)),
                  Tensor(stacked(lambda p: p.residuals)),
                  Tensor(stacked(lambda p: p.rank)),
                  Tensor(stacked(lambda p: p.singular_values)))


def _lstsq_batch(a, b):
    """Which of the two readings of the right-hand side torch takes, and what the
    batch shape is. `(None, False)` means there is no batch and numpy's own
    two-dimensional path answers it.

    **The two readings do not broadcast the same way, and that is measured rather
    than assumed.** Against a `(2, 4, 2)` matrix torch takes `(2, 4)` as a batch of
    vectors and refuses `(1, 4)`, which broadcasts perfectly well — so the vector
    reading wants the leading dimensions *equal*. The matrix reading does
    broadcast: `(1, 4, 1)` is accepted and stretched to two.

    A shape neither reading accepts is refused here rather than answered, because
    numpy would happily answer several of them and torch does not.

    **Which refusal is a difference, and it is deliberate.** Ten right-hand-side
    shapes were put to torch against `(2, 4, 2)`; both sides accept the same three
    and refuse the same seven. On three of the seven — `(4, 1)`, `(4, 2)` and
    `(3, 4, 1)` — torch names a broadcast failure and this names the size. Trying to
    reproduce the choice: torch reports the broadcast for those three and the size
    for `(1, 4)`, whose leading dimensions *do* broadcast, and for `(3, 4)`, whose do
    not. No rule separates the five, because the two messages come from different
    places inside torch rather than from one decision. **Refusing is what matters
    and that agrees**; the wording of a refusal on a malformed shape does not.
    """
    if len(b) == len(a) and b[-2] == a[-2]:
        return (_np.broadcast_shapes(a[:-2], b[:-2]) or None), False
    if len(b) == len(a) - 1 and tuple(b[:-1]) == tuple(a[:-2]):
        return (tuple(a[:-2]) or None), True
    if len(b) < len(a) - 1:
        raise RuntimeError(
            "torch.linalg.lstsq: input.dim() must be greater or equal to "
            "other.dim() and (input.dim() - other.dim()) <= 1")
    raise RuntimeError(
        "torch.linalg.lstsq: input.size(-2) should match other.size(-2)")


def _lstsq_one(a, b, driver, rcond):
    """One matrix's least squares. The batched path above calls this per matrix —
    numpy's `lstsq` is two-dimensional and has no batch of its own."""
    # `gels` has no rank estimate, so it has no cutoff — torch's does not read
    # `rcond` at all and neither does this.
    cut = None if driver == "gels" else rcond
    sol, res, rank, sv = _np.linalg.lstsq(a, b, rcond=cut)
    if driver == "gelsy" and rank < min(a.shape):
        # The sentence has to stay a noun phrase — `_unsupported` finishes it with
        # *is not in the browser subset*, which is the wording the golden matches on.
        # The why is in the docstring above; `driver="gelsd"` is the way through.
        _unsupported(f'lstsq(rcond={rcond}, driver="gelsy") on a cut that leaves '
                     "the matrix rank-deficient, where a pivoted QR and the SVD "
                     "part and only the SVD is here")
    empty_f = _np.zeros(0, dtype=sol.dtype)
    if driver == "gelsy":
        res = empty_f
    if driver == "gels":
        rank = _np.zeros(0, dtype=_np.int64)
    if driver in ("gels", "gelsy"):
        sv = empty_f
    return _Lstsq(Tensor(_np.ascontiguousarray(sol)), Tensor(_np.asarray(res)),
                  Tensor(_np.asarray(rank, dtype=_np.int64)), Tensor(_np.asarray(sv)))


# ---- LU
#
# The LU factorisation was already running underneath `det`, `inv` and `solve`.
# It simply was not exposed.
#
# **The pivots count from 1.** LAPACK's convention, inherited by torch — on a 2×2
# with no swaps `pivots` is `[1, 2]` rather than `[0, 1]`. Counting from 0 makes
# `lu_solve` give a different answer without a sound. Matched by measurement.

def _lu_pack(data):
    """Partially pivoted LU. One packed `LU` matrix and a swap table **counting
    from 1.**"""
    a = data.astype(_np.float64).copy()
    n, m = a.shape[-2], a.shape[-1]
    k = min(n, m)
    flat = a.reshape(-1, n, m)
    piv = _np.zeros((flat.shape[0], k), dtype=_np.int32)
    for b in range(flat.shape[0]):
        mat = flat[b]
        for col in range(k):
            best = col + int(_np.argmax(_np.abs(mat[col:, col])))
            piv[b, col] = best + 1                      # ← LAPACK counts from 1
            if best != col:
                mat[[col, best]] = mat[[best, col]]
            pivot = mat[col, col]
            if pivot == 0:
                continue
            mat[col + 1:, col] /= pivot
            mat[col + 1:, col + 1:] -= _np.outer(mat[col + 1:, col], mat[col, col + 1:])
    return flat.reshape(a.shape), piv.reshape(data.shape[:-2] + (k,))


def lu_factor(A):  # noqa: N803
    """The factorisation packed into one `LU` matrix, plus the swap table. Values
    alone."""
    A = _mat(A, "lu_factor", square=False)
    lu_data, piv = _lu_pack(A.data)
    return _LuFactor(Tensor(lu_data.astype(A.data.dtype)), Tensor(piv))


def lu_factor_ex(A, pivot=True, check_errors=False):  # noqa: N803
    """`lu_factor` with **one more field, `info`.** It reports through a number
    rather than throwing.

    0 means it went well, and `k` means the `k`-th pivot is 0 and the matrix is
    singular (counting from 1). Measured: `[[1,2],[2,4]]` gives 2. It is kept apart
    from the throwing version (`lu_factor`) so that solving a batch can carry on
    with the rest when one matrix is bad.
    """
    A = _mat(A, "lu_factor_ex", square=False)
    if not pivot:
        _unsupported("lu_factor_ex(pivot=False)")
    lu_data, piv = _lu_pack(A.data)
    n, m = lu_data.shape[-2], lu_data.shape[-1]
    k = min(n, m)
    flat = lu_data.reshape(-1, n, m)
    info = _np.zeros(flat.shape[0], dtype=_np.int32)
    for b in range(flat.shape[0]):
        zero = _np.flatnonzero(_np.diagonal(flat[b])[:k] == 0)
        info[b] = 0 if zero.size == 0 else int(zero[0]) + 1
    shape = A.data.shape[:-2]
    return _LuFactorEx(Tensor(lu_data.astype(A.data.dtype)), Tensor(piv),
                       Tensor(info.reshape(shape) if shape else info[0]))


# Bunch-Kaufman's threshold. **It is a constant of the method, not a tolerance** —
# the value that bounds the growth factor while keeping the 1×1 pivot whenever it is
# safe. LAPACK, torch and this all use it.
_BK_ALPHA = (1.0 + 17.0 ** 0.5) / 8.0


def _ldl_one(mat):
    """LAPACK's `dsytf2` on the lower triangle: `L D Lᵀ` **with pivoting.**

    **This refused an indefinite matrix and the reason was accurate** — torch uses
    Bunch-Kaufman, which swaps where it needs to, and a factorisation without the
    swaps is a different one. So the way through was to write Bunch-Kaufman rather
    than to loosen anything: there are many valid `L D Lᵀ` decompositions and only
    LAPACK's packing and swap table compare against torch's.

    The pivot table is LAPACK's own: a positive `k+1` is a 1×1 pivot with row `k`
    swapped for row `ipiv[k]−1`, and **a repeated negative pair is a 2×2 block**.
    torch hands that straight through, so `[-3, -3, 3]` on a 3×3 is a block over the
    first two rows.

    **The swap is over columns, not rows**, and that one line is where this first
    went wrong: written as a row swap, ten of thirteen matrices still agreed — the
    three that did not were the ones whose later pivot search read a corrupted entry,
    so the first symptom was a *pivot table* diverging two steps further on.

    Checked against torch on 470 symmetric matrices, ranks 1 to 8, definite,
    indefinite and hollow: every packed entry and every pivot code.
    """
    a = _np.tril(mat.astype(_np.float64))
    n = a.shape[0]
    ipiv = _np.zeros(n, dtype=_np.int32)
    # **`info` is the first zero pivot, counting from 1**, which is what LAPACK
    # reports and `ldl_factor_ex` hands back. It used to be hardcoded to 0 with the
    # note *the bad cases are refused* — true while they were.
    info = 0
    k = 0
    while k < n:
        step = 1
        # **`abs` is a tensor function in this file** — the module scope shadows the
        # builtin, so `_np.abs` is this file's rule and it has been stepped on here
        # before.
        here = _np.abs(a[k, k])
        if k < n - 1:
            imax = k + 1 + int(_np.argmax(_np.abs(a[k + 1:, k])))
            colmax = _np.abs(a[imax, k])
        else:
            imax, colmax = -1, 0.0
        if max(here, colmax) == 0.0 or here >= _BK_ALPHA * colmax:
            kp = k
        else:
            rowmax = max((_np.abs(a[imax, j]) for j in range(k, imax)), default=0.0)
            if imax < n - 1:
                jmax = imax + 1 + int(_np.argmax(_np.abs(a[imax + 1:, imax])))
                rowmax = max(rowmax, _np.abs(a[jmax, imax]))
            if here >= _BK_ALPHA * colmax * (colmax / rowmax):
                kp = k
            elif _np.abs(a[imax, imax]) >= _BK_ALPHA * rowmax:
                kp = imax
            else:
                kp, step = imax, 2
        kk = k + step - 1
        if kp != kk:
            if kp + 1 < n:
                keep = a[kp + 1:, kk].copy()
                a[kp + 1:, kk] = a[kp + 1:, kp]
                a[kp + 1:, kp] = keep
            for j in range(kk + 1, kp):
                a[j, kk], a[kp, j] = a[kp, j], a[j, kk]
            a[kk, kk], a[kp, kp] = a[kp, kp], a[kk, kk]
            if step == 2:
                a[kk, k], a[kp, k] = a[kp, k], a[kk, k]
        if step == 1:
            d11 = a[k, k]
            if d11 == 0.0 and info == 0:
                info = k + 1
            if d11 != 0.0 and k < n - 1:
                a[k + 1:, k] /= d11
                a[k + 1:, k + 1:] = _np.tril(
                    a[k + 1:, k + 1:] - d11 * _np.outer(a[k + 1:, k], a[k + 1:, k]))
            ipiv[k] = kp + 1
        else:
            if k < n - 2:
                d21 = a[k + 1, k]
                d11 = a[k + 1, k + 1] / d21
                d22 = a[k, k] / d21
                d21 = (1.0 / (d11 * d22 - 1.0)) / d21
                for j in range(k + 2, n):
                    wk = d21 * (d11 * a[j, k] - a[j, k + 1])
                    wkp1 = d21 * (d22 * a[j, k + 1] - a[j, k])
                    a[j:, j] -= a[j:, k] * wk + a[j:, k + 1] * wkp1
                    a[j, k], a[j, k + 1] = wk, wkp1
            ipiv[k] = ipiv[k + 1] = -(kp + 1)
        k += step
    return a, ipiv, info


def _ldl_pack(data):
    """`_ldl_one` over a batch. The answer is **packed into one matrix** in torch's
    shape — `D` on the diagonal and `L` below it."""
    a = data.astype(_np.float64)
    n = a.shape[-1]
    flat = a.reshape(-1, n, n)
    outs, pivs, infos = [], [], []
    for b in range(flat.shape[0]):
        ld, piv, info = _ldl_one(flat[b])
        outs.append(ld)
        pivs.append(piv)
        infos.append(info)
    return (_np.stack(outs).reshape(a.shape),
            _np.stack(pivs).reshape(data.shape[:-2] + (n,)),
            _np.array(infos, dtype=_np.int32))


def ldl_factor(input, hermitian=False):  # noqa: A002
    """A symmetric matrix as `L D Lᵀ`. It gives the packed `LD` and the swap table.

    **A zero pivot stops here**, which is the one place this and torch part on
    purpose: torch's `ldl_factor` meets a singular matrix and raises
    *INTERNAL ASSERT FAILED … please report a bug to PyTorch* rather than saying
    anything about the matrix. Both stop, and the golden asks for that rather than
    for a wording nobody would want to copy.
    """
    input = _mat(input, "ldl_factor")
    ld, piv, info = _ldl_pack(input.data)
    if int(info.max()) != 0:
        raise RuntimeError(
            f"linalg.ldl_factor: the leading minor of order {int(info.max())} is "
            "singular — `ldl_factor_ex` reports it in `info` instead of stopping")
    return _LdlFactor(Tensor(ld.astype(input.data.dtype)), Tensor(piv))


def ldl_factor_ex(input, hermitian=False, check_errors=False):  # noqa: A002
    """`ldl_factor` with `info` attached — **the first zero pivot, counting from 1.**

    It used to be hardcoded to 0, with the note *the bad cases are refused*, which
    was true while they were. Measured against torch: `[[1,1],[1,1]]` gives 2 and a
    zero matrix gives 1.
    """
    input = _mat(input, "ldl_factor_ex")
    ld, piv, info = _ldl_pack(input.data)
    shape = input.data.shape[:-2]
    got = info.reshape(shape) if shape else _np.int32(info[0])
    return _LdlFactorEx(Tensor(ld.astype(input.data.dtype)), Tensor(piv), Tensor(got))


def _ldl_solve_one(packed, piv, rhs):
    """`P L D Lᵀ Pᵀ x = b` with LAPACK's pivot table, which is `dsytrs`.

    **The old body read the packed matrix and ignored the table**, which was right
    only while nothing was ever swapped — the factorisation refused everything else.
    With Bunch-Kaufman in place it was wrong on 47 of 80 random symmetric matrices
    and returned a plausible number every time.

    Three parts, and the pivots enter in two of them. The swaps are applied to `b`
    going in and undone coming out; `D` is block diagonal, so a 2×2 block is a 2×2
    solve rather than a division.
    """
    n = packed.shape[0]
    x = rhs.copy()

    # Forward: swap, eliminate, divide — **in that order, one step at a time.**
    # Written as *permute, then solve `L`, then solve `D`* it is wrong, because `L`
    # was built in the swapped order and the two do not commute. That version was
    # wrong on 97 of 279 random matrices and plausible on every one of them.
    k = 0
    while k < n:
        if piv[k] > 0:
            kp = int(piv[k]) - 1
            if kp != k:
                x[[k, kp]] = x[[kp, k]]
            if k < n - 1:
                x[k + 1:] -= _np.outer(packed[k + 1:, k], x[k])
            x[k] = x[k] / packed[k, k]
            k += 1
        else:
            kp = -int(piv[k]) - 1
            if kp != k + 1:
                x[[k + 1, kp]] = x[[kp, k + 1]]
            if k < n - 2:
                x[k + 2:] -= _np.outer(packed[k + 2:, k], x[k])
                x[k + 2:] -= _np.outer(packed[k + 2:, k + 1], x[k + 1])
            # **The block's off-diagonal is part of `D`**, not of the unit triangle
            # around it. Divided through by it, as LAPACK does.
            off = packed[k + 1, k]
            top = packed[k, k] / off
            bot = packed[k + 1, k + 1] / off
            denom = top * bot - 1.0
            b0, b1 = x[k] / off, x[k + 1] / off
            x[k] = (bot * b0 - b1) / denom
            x[k + 1] = (top * b1 - b0) / denom
            k += 2

    # Back: `Lᵀ`, then the swap undone, walking up. A 2×2 block is met at its **second**
    # row and both of its columns are applied before the pair steps past.
    k = n - 1
    while k >= 0:
        if piv[k] > 0:
            if k < n - 1:
                x[k] -= packed[k + 1:, k] @ x[k + 1:]
            kp = int(piv[k]) - 1
            if kp != k:
                x[[k, kp]] = x[[kp, k]]
            k -= 1
        else:
            if k < n - 1:
                x[k] -= packed[k + 1:, k] @ x[k + 1:]
                x[k - 1] -= packed[k + 1:, k - 1] @ x[k + 1:]
            kp = -int(piv[k]) - 1
            if kp != k:
                x[[k, kp]] = x[[kp, k]]
            k -= 2
    return x


def ldl_solve(LD, pivots, b, hermitian=False):  # noqa: N803
    """Solve using the factorisation `ldl_factor` produced — `dsytrs`, pivots and
    all."""
    LD = _mat(LD, "ldl_solve")
    packed = _np.asarray(LD.data, dtype=_np.float64)
    piv = _np.asarray(_wrap(pivots).data).astype(int)
    rhs = _np.asarray(_wrap(b).data, dtype=_np.float64)
    n = packed.shape[-1]
    flat_ld = packed.reshape(-1, n, n)
    flat_piv = piv.reshape(-1, n)
    flat_b = rhs.reshape(-1, n, rhs.shape[-1])
    outs = [_ldl_solve_one(flat_ld[i], flat_piv[i], flat_b[i])
            for i in range(flat_ld.shape[0])]
    got = _np.stack(outs).reshape(rhs.shape)
    return Tensor(got.astype(_wrap(b).data.dtype))


def geqrf(input):
    """QR **in reflector form.** `householder_product` builds `Q` out of it.

    It imitates LAPACK's two stages exactly — `geqrf` holds the reflectors and
    spreading them into `Q` is separate. They are kept apart because some code
    multiplies by `Q` without ever building it.
    """
    input = _mat(input, "geqrf", square=False)
    a = _np.asarray(input.data, dtype=_np.float64)
    m, n = a.shape[-2], a.shape[-1]
    flat = a.reshape(-1, m, n).copy()
    taus = _np.zeros((flat.shape[0], min(m, n)))
    for b in range(flat.shape[0]):
        mat = flat[b]
        for j in range(min(m, n)):
            x = mat[j:, j].copy()
            # **With everything below the diagonal at 0 there is no reflection** —
            # LAPACK's `dlarfg` puts `tau = 0` there and leaves the values alone.
            # The last column of a square matrix is always that place, and
            # flipping the sign there flipped `Q`'s last column wholesale. Asked
            # with rectangles alone that column never appears and it is
            # invisible.
            if _np.linalg.norm(x[1:]) == 0:
                continue
            norm = _np.linalg.norm(x)
            alpha = x[0]
            beta = -_np.sign(alpha) * norm if alpha != 0 else -norm
            tau = (beta - alpha) / beta
            v = x / (alpha - beta)
            v[0] = 1.0
            mat[j:, j:] -= tau * _np.outer(v, v @ mat[j:, j:])
            mat[j, j] = beta
            mat[j + 1:, j] = v[1:]
            taus[b, j] = tau
    return _Geqrf(Tensor(flat.reshape(a.shape).astype(input.data.dtype)),
                  Tensor(taus.reshape(a.shape[:-2] + (min(m, n),)).astype(input.data.dtype)))


def householder_product(input, tau):  # noqa: A002
    """Multiply the reflectors together to build `Q`. `geqrf`'s partner.

    `H_i = I − τ_i v_i v_iᵀ` are multiplied in turn. `v_i` has 1 on the diagonal
    and `A[i+1:, i]` below it — the 1 on the diagonal is **an agreement not to
    store it**, so reading that cell and using it mistakes the `R` the
    factorisation packed there for a reflector.
    """
    input = _mat(input, "householder_product", square=False)
    a = _np.asarray(input.data, dtype=_np.float64)
    taus = _np.asarray(_wrap(tau).data, dtype=_np.float64)
    m, n = a.shape[-2], a.shape[-1]
    flat = a.reshape(-1, m, n)
    flat_tau = taus.reshape(-1, taus.shape[-1])
    outs = []
    for b in range(flat.shape[0]):
        q = _np.eye(m)
        for j in range(flat_tau.shape[1] - 1, -1, -1):
            v = _np.zeros(m)
            v[j] = 1.0
            v[j + 1:] = flat[b][j + 1:, j]
            q -= flat_tau[b, j] * _np.outer(v, v @ q)
        outs.append(q[:, :n])
    got = _np.stack(outs).reshape(a.shape[:-2] + (m, n))
    return Tensor(got.astype(input.data.dtype))


def lu(A, pivot=True):  # noqa: N803
    """Spread into `P`, `L` and `U`. Easier to read than the packed form."""
    A = _mat(A, "lu", square=False)
    if not pivot:
        _unsupported("lu(pivot=False)")
    lu_data, piv = _lu_pack(A.data)
    n, m = lu_data.shape[-2], lu_data.shape[-1]
    k = min(n, m)
    flat_lu = lu_data.reshape(-1, n, m)
    flat_piv = piv.reshape(-1, k)
    diag = _np.arange(k)
    ps, ls, us = [], [], []
    for b in range(flat_lu.shape[0]):
        low = _np.tril(flat_lu[b][:, :k], -1).copy()
        low[diag, diag] = 1.0
        ls.append(low)
        us.append(_np.triu(flat_lu[b])[:k, :])
        # The swaps are traced back to build the permutation matrix. `piv` counts
        # from 1, so one is subtracted.
        order = _np.arange(n)
        for col in range(k):
            src = int(flat_piv[b, col]) - 1
            if src != col:
                order[[col, src]] = order[[src, col]]
        perm = _np.zeros((n, n), dtype=_np.float64)
        perm[order, _np.arange(n)] = 1.0
        ps.append(perm)

    def pack(arrs, shape):
        return Tensor(_np.asarray(arrs).reshape(shape).astype(A.data.dtype))

    lead = A.data.shape[:-2]
    return _Lu(pack(ps, lead + (n, n)), pack(ls, lead + (n, k)),
               pack(us, lead + (k, m)))


def lu_solve(LU, pivots, b, left=True, adjoint=False):  # noqa: N803
    """Solve `A x = b` with what `lu_factor` produced.

    **The permutation is per matrix and it goes on the rows.** It used to be built
    once from `pivots.reshape(-1)` — the whole batch flattened, so every matrix got
    the first one's pivots and only the first `n` of them — and applied as
    `rhs[order]`, which indexes **axis 0**. On a batch that is the batch axis, so the
    right-hand sides were permuted between matrices.

    Both faults were invisible at 2×2, because two pivots that do not swap give the
    identity and permuting anything by the identity is harmless. At 3×3 the pivot
    value 3 ran off the end of a batch of 2 and it raised
    `IndexError: index 2 is out of bounds for axis 0 with size 2` — numpy's words
    about the wrong axis entirely. **And at 2×2 with a real swap it did not raise:**
    `[[1,2],[3,4]]` needs one, and the second matrix came back `[0.222, -0.111]`
    where the answer is `[-0.111, 0.556]`. A plausible number and no exception,
    which is what the existing case could not see by asking one matrix.
    """
    lu_t, piv_t, bt = _wrap(LU), _wrap(pivots), _wrap(b)
    n = lu_t.data.shape[-1]
    low = _np.tril(lu_t.data.astype(_np.float64), -1) + _np.eye(n)
    up = _np.triu(lu_t.data.astype(_np.float64))
    rhs = bt.data.astype(_np.float64).copy()
    piv = _np.asarray(piv_t.data).astype(int)
    lead = piv.shape[:-1]
    order = _np.broadcast_to(_np.arange(n), (*lead, n)).copy()
    for at in (_np.ndindex(*lead) if lead else [()]):
        row = order[at]
        for col in range(piv.shape[-1]):
            src = int(piv[at][col]) - 1
            if src != col:
                row[[col, src]] = row[[src, col]]
    if left:
        got = _lu_apply(low, up, order, rhs, adjoint)
    else:
        # `X A = B` is `Aᵀ Xᵀ = Bᵀ`, so the right-hand solve is the left one on the
        # transposed sides with the adjoint flipped. Real storage only, so the
        # conjugate half of torch's `Aᴴ` is a transpose.
        got = _np.swapaxes(
            _lu_apply(low, up, order, _np.swapaxes(rhs, -1, -2), not adjoint), -1, -2)
    return Tensor(got.astype(bt.data.dtype))


def _lu_apply(low, up, order, rhs, adjoint):
    """One of the two left-hand solves against a factorisation already unpacked.

    `A = P L U`, so `A x = b` permutes `b` onto `L U x = Pᵀ b` and runs forward then
    back. **The adjoint runs the whole thing in reverse**: `Aᵀ = Uᵀ Lᵀ Pᵀ`, so `Uᵀ`
    goes first, `Lᵀ` second, and the permutation lands on the *answer* rather than on
    the right-hand side — `Pᵀ x = z` means `x = P z`, which is a scatter where the
    forward direction is a gather. The inverse permutation is `argsort`.
    """
    # **Broadcast to the right-hand side's own leading shape.** torch takes a batch
    # of matrices against a single `b` as well as against a matching batch.
    take = _np.broadcast_to(order, (*rhs.shape[:-1],))
    if not adjoint:
        moved = _np.take_along_axis(rhs, take[..., None], axis=-2)
        return _np.linalg.solve(up, _np.linalg.solve(low, moved))
    w = _np.linalg.solve(_np.swapaxes(up, -1, -2), rhs)
    z = _np.linalg.solve(_np.swapaxes(low, -1, -2), w)
    return _np.take_along_axis(z, _np.argsort(take, axis=-1)[..., None], axis=-2)


# ---- the composite layers
#
# Mostly places attaching a name to something that already exists. The one that
# needs new computation is `matrix_exp`, and that one has no closed form.

def vecdot(x, b, dim=-1):
    return (_wrap(x) * _wrap(b)).sum(dim=dim)


def diagonal_linalg(A, offset=0, dim1=-2, dim2=-1):  # noqa: N803
    """**The default axes differ from `torch.diagonal`'s.**

    This looks at the last two axes (`-2, -1`) and that one at the first two
    (`0, 1`). Given 3-D input, `(2,3,4)` comes out as `(2,3)` on one side and
    `(4,2)` on the other — the similar names make them easy to read as the same
    thing and they differ starting at the shape. So the defaults are written out
    by hand.
    """
    return diagonal(A, offset=offset, dim1=dim1, dim2=dim2)


def svdvals(A):  # noqa: N803
    """The singular values alone. The middle of `svd`."""
    return linalg_svd(A, full_matrices=False).S


def eigvalsh(input, UPLO="L"):  # noqa: A002
    """The eigenvalues alone, for a symmetric matrix."""
    return eigh(input, UPLO=UPLO).eigenvalues


def vector_norm(input, ord=2, dim=None, keepdim=False):  # noqa: A002
    """A norm measured over the elements as a vector. **Given a matrix it
    flattens the whole thing** — where it parts from `matrix_norm`.

    `ord=0` is the count of non-zeros and `±inf` are the largest and smallest
    absolute values — branches that must not go into the power formula, so they
    are written out separately.
    """
    x = _wrap(input).abs()
    rank = x.data.ndim
    if dim is None and rank > 1:
        # **The rank is remembered before the flatten.** With no `dim` torch reduces
        # everything and, asked to keep, hands back **all ones** rather than a scalar
        # (measured). Flattened first, the reduction below keeps one axis instead of
        # `rank` of them, and the answer comes back a rank short — right value, wrong
        # shape, which broadcasts differently and surfaces somewhere else.
        x = x.reshape(-1)
        if keepdim:
            return vector_norm(x, ord, None, False).reshape((1,) * rank)
    if ord == _np.inf:
        return amax(x, dim=dim, keepdim=keepdim)
    if ord == -_np.inf:
        return amin(x, dim=dim, keepdim=keepdim)
    if ord == 0:
        return (x != 0).float().sum(dim=dim, keepdim=keepdim)
    if ord == 1:
        return x.sum(dim=dim, keepdim=keepdim)
    if ord == 2:
        return (x * x).sum(dim=dim, keepdim=keepdim) ** 0.5
    return (x ** ord).sum(dim=dim, keepdim=keepdim) ** (1.0 / ord)


# The branches that need singular values. The rest finish with a sum of absolute
# values over the rows or the columns.
_SPECTRAL = ("nuc", 2, -2)


def _linalg_norm(input, ord=None, dim=None, keepdim=False, dtype=None):  # noqa: A002
    """`torch.linalg.norm` — **which is not `torch.norm`, and was bound to it.**

    The two have the same seats and different meanings. `linalg.norm` dispatches: with
    an `ord` and no `dim` on a matrix it is `matrix_norm`, which for `ord=2` is the
    largest singular value; everything else is `vector_norm`. `torch.norm` is the
    elementwise p-norm throughout.

    On `[[1..9]]` that is 16.848 against 16.882 — near enough to read as rounding and
    produced by a different formula. Bound to the wrong one, `linalg.norm(A, 2)`
    answered a question nobody asked and nothing said so; the golden case written the
    same afternoon caught it here and in borch.ts at once.

    torch spells the order `ord` here and `p` at the top level, which is the other half
    of why one was mistaken for the other.
    """
    x = _wrap(input)
    if dtype is not None:
        x = _wrap(x.data.astype(_np_of(_requested_dtype(dtype))))
    if dim is None and ord is not None and x.data.ndim == 2:
        return matrix_norm(x, ord=ord, keepdim=keepdim)
    if isinstance(dim, (tuple, list)) and len(dim) == 2:
        return matrix_norm(x, ord=("fro" if ord is None else ord), dim=tuple(dim),
                           keepdim=keepdim)
    return vector_norm(x, ord=(2 if ord is None else ord), dim=dim, keepdim=keepdim)


def matrix_norm(input, ord="fro", dim=(-2, -1), keepdim=False):  # noqa: A002
    """A norm measured over the matrix. **A different number per branch.**

    The default is Frobenius; `2` is the largest singular value, `nuc` the sum of
    the singular values, `1` the largest column-wise sum of absolute values, and
    `inf` the row-wise one. Given a rank-1 matrix the first three happen to
    coincide and cannot be told apart, so the golden asks at rank 2.

    **The two axes move together.** They used to move one at a time —
    `movedim(dim[0], -2).movedim(dim[1], -1)` — and the first move shifts where the
    second one's axis sits, so `dim[1]` named whatever had slid into that index. On a
    `(3, 2, 2)` with `dim=(0, 1)` it reduced axes 1 and 0 of the *moved* tensor, which
    are the original 2 and 0: **5.568 where torch says 5.916.** Not an exception and
    not a wrong shape — a plausible norm of a different pair of axes.

    Nothing saw it because every case asked the default, where the branch does not run.

    `keepdim` then puts the ones back **where the caller's axes were** and not where
    they were moved to: torch gives `(1, 1, 2)` for `dim=(0, 1)` and `(3, 1, 1)` for
    the default.
    """
    x = _wrap(input)
    if tuple(dim) != (-2, -1):
        rank = len(x.shape)
        at = tuple(d % rank for d in dim)
        x = x.movedim(tuple(dim), (-2, -1))
        if keepdim:
            flat = matrix_norm(x, ord, (-2, -1), False)
            return flat.reshape(tuple(1 if i in at else n
                                      for i, n in enumerate(_wrap(input).shape)))
    if ord in _SPECTRAL:
        s = svdvals(x)
        if ord == "nuc":
            return s.sum(dim=-1, keepdim=keepdim)
        return (amax if ord == 2 else amin)(s, dim=-1, keepdim=keepdim)
    if ord == "fro":
        return (x * x).sum(dim=(-2, -1), keepdim=keepdim) ** 0.5
    # 1 works column-wise (summing the rows) and inf row-wise (summing the
    # columns). The sign selects the maximum or the minimum.
    #
    # **It must not be written `abs(ord)`.** This module has an `abs` shadowing
    # the Python builtin, and it takes the integer for a tensor and stops with
    # `'int' object has no attribute 'abs'`. This repository has stepped on the
    # same trap three times already with `bool`, `max` and `min` — this is the
    # fourth.
    axis = -2 if ord in (1, -1) else -1
    sums = x.abs().sum(dim=axis, keepdim=True)
    pick = amax if ord > 0 else amin
    out = pick(sums, dim=-1 if axis == -2 else -2, keepdim=True)
    return out if keepdim else out.reshape(out.shape[:-2])


def cond(input, p=None):  # noqa: A002
    """The condition number. The default is `‖A‖₂·‖A⁻¹‖₂`, which is the ratio of
    the singular values."""
    x = _mat(input, "cond")
    if p is None or p == 2:
        s = svdvals(x)
        return amax(s, dim=-1) / amin(s, dim=-1)
    if p == -2:
        s = svdvals(x)
        return amin(s, dim=-1) / amax(s, dim=-1)
    return matrix_norm(x, ord=p) * matrix_norm(inverse(x), ord=p)


def multi_dot(tensors):
    """Multiply several matrices together. **The grouping does not change the
    value** — multiplication is associative. Only the operation count changes, so
    they are multiplied in order here."""
    out = _wrap(tensors[0])
    for m in tensors[1:]:
        out = out @ _wrap(m)
    return out


def _vander_increasing(x, N=None):
    """`linalg.vander`'s version. **The powers increase across the columns** and
    the last column is the highest.

    torch keeps the two apart as well — `torch.vander` defaults to decreasing and
    `torch.linalg.vander` to increasing. **Two different functions** whose values
    are reversed.

    **Why the names were split.** Both used to be `vander`, and which one the class
    body below caught turned on *which was defined first.* Wiring that leans on
    definition order, with the dependence written down nowhere, so to a reader the
    later definition looks like it covers the earlier one.
    """
    v = _wrap(x)
    n = v.data.shape[-1] if N is None else N
    cols = [v ** k for k in range(n)]
    return stack(cols, dim=-1)


def solve_triangular(input, b, upper, left=True, unitriangular=False):  # noqa: A002
    """Solve **knowing** the matrix is triangular. One forward and one backward
    sweep finish it.

    `unitriangular` **ignores the diagonal and treats it as 1** — a branch whose
    values change quietly when it is not honoured. `left=False` solves `X A = B`,
    so both sides are transposed and sent down the same path.
    """
    at, bt = _mat(input, "solve_triangular"), _wrap(b)
    tri = _np.triu(at.data) if upper else _np.tril(at.data)
    if unitriangular:
        idx = _np.arange(tri.shape[-1])
        tri = tri.copy()
        tri[..., idx, idx] = 1.0
    if not left:
        x = _np.linalg.solve(_T(tri), _T(bt.data))
        return Tensor(_T(x))
    return Tensor(_np.linalg.solve(tri, bt.data))


def tensorsolve(input, b, dims=None):  # noqa: A002
    """Fold the tensor into input matrix, solve, and spread it back.

    **`dims` moves those axes of `A` to the end before the fold**, so it changes
    which axes become the matrix and therefore the shape of the answer as well as
    its values. It was refused; what the refusal did not say is that numpy's
    `tensorsolve` already takes it, under the name `axes`. Measured across six
    settings on a `(2, 3, 2, 3)` — `None`, `(0, 1)`, `(1, 0)`, `(0,)`, `(2, 3)`,
    `(3,)` — torch and numpy agree on every one, shape included.
    """
    at, bt = _wrap(input), _wrap(b)
    out = _np.linalg.tensorsolve(at.data, bt.data,
                                 axes=None if dims is None else tuple(dims))
    return Tensor(out)


def tensorinv(input, ind=2):  # noqa: A002
    at = _wrap(input)
    lead = at.data.shape[:ind]
    n = int(_np.prod(lead))
    out = _np.linalg.inv(at.data.reshape(n, -1))
    return Tensor(out.reshape(at.data.shape[ind:] + lead))


# What counts as "small" for scaling and squaring. Below this 1-norm the Taylor
# series converges quickly.
_EXP_SMALL = 0.5
# How many terms that condition needs. 0.5^18/18! is far below double
# precision's floor.
_EXP_TERMS = 18


def _expm_raw(a):
    """Scaling and squaring plus Taylor. Computed in double precision."""
    n = a.shape[-1]
    norm = _np.abs(a).sum(axis=-2).max() if a.size else 0.0
    squarings = max(0, int(_np.ceil(_np.log2(norm / _EXP_SMALL)))) if norm > _EXP_SMALL \
        else 0
    scaled = a / (2.0 ** squarings)
    eye = _np.broadcast_to(_np.eye(n), a.shape).copy()
    term, out = eye.copy(), eye.copy()
    for k in range(1, _EXP_TERMS + 1):
        term = term @ scaled / k
        out = out + term
    for _ in range(squarings):
        out = out @ out
    return out


def matrix_exp(input):  # noqa: A002
    """The matrix exponential `e^A`. **It goes by scaling and squaring.**

    Taylor alone does not converge on a large matrix — the answer for `A*5` is
    4.8e+10, and there the growing terms overflow first. Lowering the 1-norm of
    `A/2^s` below 0.5, running the series and then squaring `s` times gives the
    same answer safely (`e^A = (e^{A/2^s})^{2^s}`).

    **The gradient is obtained from the function itself.** The Fréchet derivative
    of `e^A` carries this identity:

        the upper-right block of expm([[Aᵀ, Ḡ], [0, Aᵀ]]) = Ā

    An identity rather than an approximation. So **calling the same series the
    forward used** brings the gradient with it — there is nowhere to derive and
    write a new derivative. That is why this method was chosen. A derived formula
    can be wrong, and wrong quietly.
    """
    x = _mat(input, "matrix_exp")
    a = x.data.astype(_np.float64)
    n = a.shape[-1]

    def back(g):
        gg = _np.asarray(g, dtype=_np.float64)
        at = _T(a)
        block = _np.zeros(a.shape[:-2] + (2 * n, 2 * n), dtype=_np.float64)
        block[..., :n, :n] = at
        block[..., :n, n:] = gg
        block[..., n:, n:] = at
        return (_expm_raw(block)[..., :n, n:].astype(x.data.dtype),)

    return input._make(_expm_raw(a).astype(x.data.dtype), (input,), back, "MatrixExpBackward0")


# **torch spells the two namespaces differently for three names.**
# `torch.det(input=…)` is taken and `torch.det(A=…)` is not; under `linalg` it is the
# other way round, and the same for `qr` and `slogdet`. One function cannot answer to
# both, so the top-level definitions keep `input` and `linalg` gets these three.
#
# It is the third shape of this in a day — `Tensor.split` against `torch.split`,
# `Tensor.softmax` against `F.softmax`, and now the top level against `linalg`. Each
# time a single implementation was serving two names torch spells apart, and each time
# the answer is a wrapper rather than a choice between them.
#
# **Written out rather than made by a factory**, and the first version was a factory
# taking `(A, *args, **kw)`. That reads as *variadic* to the signature axis, so three
# rows left `agree` for a bucket meaning **cannot judge** — an absorbing state, and
# the axis has a check watching for exactly that drop because it shipped with every C
# method silently classified as absent. A wrapper that hides the signature it exists
# to correct is worse than no wrapper.
#
# **torch is not consistent inside `linalg` either.** Roughly half take `A` and half
# take `input`, with `multi_dot` on `tensors`, `lu_solve` on `LU`, `ldl_solve` on `LD`
# and `vecdot` on `x`. Every one was measured by calling it; no rule short of the
# table gets more than half of them right.

def _linalg_det(A):                                          # noqa: N803
    """`linalg.det`. `torch.det` takes `input`; this takes `A`."""
    return det(A)


def _linalg_slogdet(A):                                      # noqa: N803
    """`linalg.slogdet`. `torch.slogdet` takes `input`; this takes `A`."""
    return slogdet(A)


def _linalg_qr(A, mode="reduced"):                           # noqa: N803
    """`linalg.qr`. **Two names and two vocabularies, both torch's.**

    `torch.qr` takes `input` and a boolean `some`; `torch.linalg.qr` takes `A`
    and a string `mode`. They are the same factorisation asked for in the two
    ways torch offers, so the translation happens here rather than the older
    spelling being dropped — `x.qr(some=False)` is a line torch code contains.

        some=True   <->  mode="reduced"
        some=False  <->  mode="complete"
    """
    return _qr_impl(A, mode)


class _Linalg(_Namespace):
    """The `torch.linalg` slot.

    **It mostly points at the same implementation — and where it does not, that is
    torch's doing rather than ours.** This used to say "so there is nowhere to
    diverge", which was true when written and had stopped being true three names
    later: `lu` spreads `P`, `L`, `U` here and packs them at the top level,
    `lu_solve` takes its arguments the other way round, and `vander` counts its
    powers the other way. Four now, with `svd`: `torch.svd` gives `V` and the
    reduced form, `torch.linalg.svd` gives `Vh` and the full one.

    Five, counting the argument *names*: `det`, `qr` and `slogdet` are `input` at
    the top level and `A` here, so those three go through `_linalg_A`.

    The note listing the first three sits in `__init__.py`, in a different file from
    the sentence it contradicted — which is why the sentence went on being read.
    """

    LinAlgError = LinAlgError
    det = staticmethod(_linalg_det)
    slogdet = staticmethod(_linalg_slogdet)
    inv = staticmethod(inverse)
    inv_ex = staticmethod(inv_ex)
    solve = staticmethod(solve)
    solve_ex = staticmethod(solve_ex)
    cholesky = staticmethod(cholesky)
    cholesky_ex = staticmethod(cholesky_ex)
    matrix_power = staticmethod(matrix_power)
    qr = staticmethod(_linalg_qr)
    svd = staticmethod(linalg_svd)
    pinv = staticmethod(pinverse)
    matrix_rank = staticmethod(matrix_rank)
    eig = staticmethod(eig)
    eigh = staticmethod(eigh)
    lstsq = staticmethod(lstsq)
    lu = staticmethod(lu)
    lu_factor = staticmethod(lu_factor)
    lu_factor_ex = staticmethod(lu_factor_ex)
    lu_solve = staticmethod(lu_solve)
    ldl_factor = staticmethod(ldl_factor)
    ldl_factor_ex = staticmethod(ldl_factor_ex)
    ldl_solve = staticmethod(ldl_solve)
    householder_product = staticmethod(householder_product)
    norm = staticmethod(_linalg_norm)
    # The composite layers.
    matmul = staticmethod(matmul)
    vecdot = staticmethod(vecdot)
    cross = staticmethod(cross)
    diagonal = staticmethod(diagonal_linalg)
    svdvals = staticmethod(svdvals)
    eigvals = staticmethod(eigvals)
    eigvalsh = staticmethod(eigvalsh)
    vector_norm = staticmethod(vector_norm)
    matrix_norm = staticmethod(matrix_norm)
    cond = staticmethod(cond)
    multi_dot = staticmethod(multi_dot)
    vander = staticmethod(_vander_increasing)
    solve_triangular = staticmethod(solve_triangular)
    tensorsolve = staticmethod(tensorsolve)
    tensorinv = staticmethod(tensorinv)
    matrix_exp = staticmethod(matrix_exp)


linalg = _Linalg()


class _Fft(_Namespace):
    """`torch.fft`. The body lives in `borch/_fft.py` — this file is already large
    and that one imports nothing but `_tensor`, so it can stand on its own."""

    fft = staticmethod(_fft_fft)
    ifft = staticmethod(_fft_ifft)
    rfft = staticmethod(_fft_rfft)
    irfft = staticmethod(_fft_irfft)
    fftfreq = staticmethod(_fft_fftfreq)
    rfftfreq = staticmethod(_fft_rfftfreq)
    fftshift = staticmethod(_fft_fftshift)
    ifftshift = staticmethod(_fft_ifftshift)
    fft2 = staticmethod(_fft_fft2)
    ifft2 = staticmethod(_fft_ifft2)
    fftn = staticmethod(_fft_fftn)
    ifftn = staticmethod(_fft_ifftn)
    rfft2 = staticmethod(_fft_rfft2)
    irfft2 = staticmethod(_fft_irfft2)
    rfftn = staticmethod(_fft_rfftn)
    irfftn = staticmethod(_fft_irfftn)
    hfft = staticmethod(_fft_hfft)
    ihfft = staticmethod(_fft_ihfft)
    hfft2 = staticmethod(_fft_hfft2)
    ihfft2 = staticmethod(_fft_ihfft2)
    hfftn = staticmethod(_fft_hfftn)
    ihfftn = staticmethod(_fft_ihfftn)


fft = _Fft()


class _Cuda(_Namespace):
    @staticmethod
    def is_available():
        return False

    @staticmethod
    def manual_seed_all(seed):
        return None

    @staticmethod
    def synchronize():
        _unsupported("torch.cuda.synchronize")


cuda = _Cuda()


# ── the in-place activations and the `upsample` aliases ─────────────────────
#
# torch's `F.relu_(x)` changes `x` **in its own buffer.** Used in a training loop
# to avoid building an intermediate tensor, and in textbook code it pairs with
# `nn.ReLU(inplace=True)`.
#
# **The version without the underscore does the computation.** This only writes
# the result back into its own buffer — two copies of the same formula eventually
# diverge, and the values are plausible enough to be invisible.

_FUNCTIONAL_INPLACE = ("relu", "celu", "elu", "selu", "hardtanh", "leaky_relu",
                       "threshold", "rrelu")


def _make_functional_inplace(name):
    fn = globals()[name]

    def call(x, *args, **kw):
        x = _wrap(x)
        return x._inplace(lambda: fn(x, *args, **kw), name + "_")

    call.__name__ = name + "_"
    call.__doc__ = (f"`F.{name}` in place. That side does the computation and "
                    "this writes it back into its own buffer. A leaf with "
                    "gradients on is refused as torch does.")
    return call


for _nm in _FUNCTIONAL_INPLACE:
    globals()[_nm + "_"] = _make_functional_inplace(_nm)


# ── `inplace=` on the activations that take it ──────────────────────────────
#
# **twenty-six names took no `inplace` at all**, which `torch_signatures_core` counted
# as the single largest real absence in `nn` — larger than any feature. torch's
# textbook line is `nn.ReLU(inplace=True)`, and a caller writing it here met
# `TypeError: __init__() got an unexpected keyword argument`.
#
# **`inplace=True` is exactly the underscore name**, routed through the same
# `Tensor._inplace` the block above uses. Written a second way it would be a second
# formula, and the two would drift while both looked right — the fault this file's
# own comment fifteen lines up warns about.
#
# What it is *not* is a no-op. Three functions here already accepted `inplace` and
# discarded it (`dropout1d`, `dropout2d`, `dropout3d`), which is the shape this
# repository spent a day removing: an argument that is taken and dropped reads as one
# that works, and a caller relying on the input being changed gets an unchanged input
# and no complaint.

def _inplace_arg(x, inplace, name, fn):
    """`fn()`, or `fn()` written back into `x` — the same path the `_` name takes.

    **Returns the input object itself when `inplace`**, which is what makes the
    argument worth having: the caller's tensor is the one that changed, and code that
    keeps a reference sees it.
    """
    if not inplace:
        return fn()
    return _wrap(x)._inplace(fn, name + "_")


# ── the spatial transformer ─────────────────────────────────────────────────
#
# `affine_grid` builds a grid recording "where in the input does this output cell
# look", and `grid_sample` fetches the values from those positions. The two are a
# pair, and the `theta` between them is learned — the structure by which a model
# learns to crop, rotate and zoom on its own.
#
# **The grid coordinates are kept as differentiable tensors.** Then the gradient
# towards the input and the gradient towards the grid (and so towards `theta`)
# both come out on their own. Only the position indices (the floored integers) are
# constants — torch does not flow there either.

def _grid_base(n, align_corners):
    """The sample positions over `[-1, 1]`. **`align_corners` changes this.**

    True pins both ends (`-1`, `1`) and divides evenly between them. False takes
    **the centre** of each cell (`(2i+1)/n − 1`) — half of the end cell falls
    outside. The same split as `interpolate`'s, and the values are similar enough
    in the interior that the eye does not part them.
    """
    if align_corners:
        return (_np.linspace(-1.0, 1.0, n, dtype=_np.float32) if n > 1
                else _np.zeros(1, dtype=_np.float32))
    return ((2 * _np.arange(n, dtype=_np.float32) + 1) / n - 1).astype(_np.float32)


def affine_grid(theta, size, align_corners=False):
    """The sampling grid `theta` draws. It takes `(N, 2, 3)` and gives
    `(N, H, W, 2)`.

    The last axis is in **`(x, y)` order** — reversed from the shape's `(H, W)`.
    Written reversed, a square input gives the same answer and it is invisible; it
    surfaces on a rectangle.
    """
    theta = _wrap(theta)
    n, _, h, w = tuple(int(v) for v in size)
    xs = _grid_base(w, align_corners)
    ys = _grid_base(h, align_corners)
    # Homogeneous coordinates `(x, y, 1)` — the translation finishes in the same
    # multiplication.
    base = _np.stack([_np.broadcast_to(xs[None, :], (h, w)),
                      _np.broadcast_to(ys[:, None], (h, w)),
                      _np.ones((h, w), dtype=_np.float32)], axis=-1)
    flat = Tensor(base.reshape(h * w, 3).astype(_np.float32))
    out = matmul(flat, theta.transpose(-2, -1))      # (N, H·W, 2)
    return out.reshape(n, h, w, 2)


def _grid_denorm(g, n, align_corners):
    """Turn `[-1, 1]` back into the input's cell indices. The inverse of
    `_grid_base`."""
    if align_corners:
        return (g + 1.0) * ((n - 1) / 2.0)
    return ((g + 1.0) * n - 1.0) * 0.5


def _grid_reflect(v, n, align_corners):
    """**Reflect** what falls outside the range. The interval reflected across
    differs by `align_corners`.

    True gives `[0, n−1]` and false `[−0.5, n−0.5]` (measured). It is clamped once
    more after reflecting — the false interval reaches past the actual cells.
    """
    lo, hi = (0.0, n - 1.0) if align_corners else (-0.5, n - 0.5)
    if hi <= lo:
        return v * 0.0 + lo
    span = 2.0 * (hi - lo)
    t = remainder(v - lo, span)
    return clamp(minimum(t, span - t) + lo, 0.0, n - 1.0)


def grid_sample(input, grid, mode="bilinear", padding_mode="zeros",
                align_corners=False):
    """Fetch the values at the positions the grid points at. `affine_grid`'s
    partner.

    **The position indices are constants and the weights are tensors.** The
    floored integers have no derivative and their remainders become the weights,
    so keeping the weights alone in the graph flows gradient towards both the
    input and the grid — that is the path by which a spatial transformer learns
    `theta`.
    """
    input, grid = _wrap(input), _wrap(grid)
    n, c, h, w = input.data.shape
    oh, ow = grid.data.shape[1], grid.data.shape[2]
    gx = grid[:, :, :, 0]
    gy = grid[:, :, :, 1]
    sx = _grid_denorm(gx, w, align_corners)
    sy = _grid_denorm(gy, h, align_corners)
    # **The centre is padded for the two-tap kernels and left alone for bicubic.**
    # Clamping the centre puts both bilinear corners inside, which is the whole job
    # there; a 4×4 window steps one cell further and needs the rule at the tap
    # instead, so moving the centre as well would move it twice.
    if padding_mode not in ("zeros", "border", "reflection"):
        _unsupported(f"grid_sample(padding_mode={padding_mode!r})")
    if mode != "bicubic":
        if padding_mode == "border":
            sx, sy = clamp(sx, 0.0, w - 1.0), clamp(sy, 0.0, h - 1.0)
        elif padding_mode == "reflection":
            sx = _grid_reflect(sx, w, align_corners)
            sy = _grid_reflect(sy, h, align_corners)

    flat = input.reshape(-1)
    batch = _np.arange(n).reshape(n, 1, 1, 1)
    chan = _np.arange(c).reshape(1, c, 1, 1)

    def pick(iy, ix):
        """Fetch one corner. **What falls outside is left at 0 and the index is
        clamped before being passed on** — unclamped it reads a stray
        position."""
        inside = ((ix >= 0) & (ix < w) & (iy >= 0) & (iy < h))
        cy = _np.clip(iy, 0, h - 1)
        cx = _np.clip(ix, 0, w - 1)
        idx = (((batch * c + chan) * h + cy[:, None]) * w + cx[:, None])
        got = take(flat, Tensor(idx.reshape(-1).astype(_np.int64)))
        got = got.reshape(n, c, oh, ow)
        return got * Tensor(inside[:, None].astype(input.data.dtype))

    if mode == "nearest":
        # torch rounds. Only a value comes out and there is no weight, so nothing
        # flows towards the grid.
        return pick(_np.rint(sy.data).astype(int), _np.rint(sx.data).astype(int))
    if mode not in ("bilinear", "bicubic"):
        _unsupported(f"grid_sample(mode={mode!r}) — bilinear, nearest and bicubic "
                     "are here")

    x0 = _np.floor(sx.data).astype(int)
    y0 = _np.floor(sy.data).astype(int)
    wx = (sx - Tensor(x0.astype(input.data.dtype))).reshape(n, 1, oh, ow)
    wy = (sy - Tensor(y0.astype(input.data.dtype))).reshape(n, 1, oh, ow)
    one = 1.0
    if mode == "bicubic":
        # **The same Keys kernel as `interpolate`'s `bicubic`, at `a = −0.75`** — this
        # is the plain path's constant, not the anti-aliased one's `−0.5`.
        #
        # The weights are written as tensor expressions in the fractional offset
        # rather than computed in numpy, because **the gradient has to reach the
        # grid**: that is the path by which a spatial transformer learns `theta`, and
        # constants would silently cut it while every value case still passed.
        #
        # **The padding is applied per tap here, not to the centre.** Bilinear can
        # clamp the continuous coordinate once and be done, because both corners of a
        # clamped centre are inside; a 4×4 window steps one cell further and its outer
        # taps land outside even so. Clamping the centre and then masking gave `border`
        # the same numbers as `zeros` — 6.04 where torch says 5.48, with the gradient
        # to the grid zeroed at the edges as well.
        edge = _grid_pad_index(padding_mode, align_corners)
        out = None
        for ky in (-1, 0, 1, 2):
            ty = _cubic_of(Tensor(float(ky)) - wy)
            for kx in (-1, 0, 1, 2):
                tap = _pick_padded(pick, edge, y0 + ky, x0 + kx, h, w)
                out_ = tap * ty * _cubic_of(Tensor(float(kx)) - wx)
                out = out_ if out is None else out + out_
        return out
    return (pick(y0, x0) * (one - wy) * (one - wx)
            + pick(y0, x0 + 1) * (one - wy) * wx
            + pick(y0 + 1, x0) * wy * (one - wx)
            + pick(y0 + 1, x0 + 1) * wy * wx)


def batch_norm(input, running_mean=None, running_var=None, weight=None, bias=None,
               training=False, momentum=0.1, eps=1e-5):
    """The function form of `BatchNorm*d`. **The layer calls this** — one copy of
    the formula.

    **With `training` on, the running statistics are changed in place.** torch does
    that — the tensors handed in come back updated. Handing back new ones leaves
    the caller's buffers unmoved, and training runs while only the evaluation-mode
    values are wrong.

    **Two different variances are used.** The normalisation uses the biased one
    (ddof=0) and the `running_var` update the unbiased one (ddof=1). Biased in both
    places, the values are off by 2.6% — a place this repository lived with for a
    long time, which is why it is written down here.
    """
    input = _wrap(input)
    rank = input.data.ndim
    shape = (1, -1) + (1,) * (rank - 2)
    reduced = tuple(i for i in range(rank) if i != 1)

    def _raw(v):
        return v.data if isinstance(v, Tensor) else v

    if training:
        # The mean and the variance are computed **inside the graph.** Taken out
        # through numpy and used as constants, the path x → mean → y is cut, the
        # gradient is wrong, and nothing reaches weight at all.
        mean = input.mean(dim=0)
        for _ in range(rank - 2):
            mean = mean.mean(dim=1)
        centered = input - mean.reshape(shape)
        var = (centered * centered).mean(dim=0)
        for _ in range(rank - 2):
            var = var.mean(dim=1)
        if running_mean is not None:
            with no_grad():
                unbiased = input.data.var(axis=reduced, ddof=1)
                _raw(running_mean)[...] = ((1 - momentum) * _raw(running_mean)
                                           + momentum * mean.data)
                _raw(running_var)[...] = ((1 - momentum) * _raw(running_var)
                                          + momentum * unbiased)
        normed = centered / (var.reshape(shape) + eps) ** 0.5
    else:
        rm = _np.asarray(_raw(running_mean)).reshape(shape)
        rv = _np.sqrt(_np.asarray(_raw(running_var)) + eps).reshape(shape)
        normed = (input - Tensor(rm)) / Tensor(rv)
    if weight is not None:
        normed = normed * _wrap(weight).reshape(shape)
    if bias is not None:
        normed = normed + _wrap(bias).reshape(shape)
    return normed


def _renorm_rows(weight, ids, max_norm, norm_type):
    """torch's `max_norm`: the rows that were looked up and are too long are
    shortened, **in the table itself.**

    It is a side effect on a parameter, which is unusual enough to be worth
    saying: `embedding_bag(idx, w, max_norm=1.0)` leaves `w` changed. torch does
    exactly this (measured).

    **A version that renormalised a copy would never part on the output.** Not on
    the second call, not on the hundredth — renormalising an already-short row is a
    no-op, so both implementations return the same numbers forever. Measured against
    real torch three calls deep: identical to seven figures, while the tables read

        torch    [0.0, 0.4472, 0.8944]   shortened, and it stays shortened
        a copy   [0.0, 1.0,    2.0   ]   untouched

    The two part on the **state**, at once, and on the output only once training
    steps from the shortened weights.

    This paragraph said "agrees on the first call and parts on the second" for an
    hour, which is a reason that reads correctly and points at the wrong check: run
    it twice, see agreement, and conclude it was confirmed. What the difference wants
    is **looking at `weight` after the call**, and no number of calls substitutes for
    that.
    """
    rows = _np.unique(_np.asarray(ids).astype(int).reshape(-1))
    data = weight.data
    lengths = _np.linalg.norm(data[rows], ord=norm_type, axis=1)
    over = rows[lengths > max_norm]
    if over.size:
        scale = max_norm / (lengths[lengths > max_norm] + 1e-7)
        data[over] = data[over] * scale.reshape(-1, 1)


def embedding_bag(input, weight, offsets=None, max_norm=None, norm_type=2.0,
                  scale_grad_by_freq=False, mode="mean", sparse=False,
                  per_sample_weights=None, include_last_offset=False,
                  padding_idx=None):
    """One row per bag. Selecting from the table and **combining** is all one
    function.

    Given `offsets`, a 1-D row of indices is cut into bags — the case where the
    bags have differing lengths. `per_sample_weights` is used in torch under
    `mode='sum'` alone.

    **`mode` sits sixth, where torch has it.** It used to be third, so
    `embedding_bag(idx, w, offsets, "sum")` set `max_norm="sum"` in torch and the
    mode here — the same call, two meanings, and both sides return a bag of the
    right shape.
    """
    input, weight = _wrap(input), _wrap(weight)
    # **`scale_grad_by_freq` is answered on `embedding` and refused here, because
    # torch disagrees with itself.** `embedding_bag(mode="sum")` is by definition
    # `embedding(...).sum(1)`, and on torch 2.13.0 the two give different gradients
    # under this flag — with the table `arange(12).reshape(4, 3)` and ids
    # `[[0,1,1],[2,1,0]]`:
    #
    #     embedding(scale=True).sum(1)   row1 ÷3, row2 ÷1   — the documented rule
    #     embedding_bag(scale=True)      row1 ÷2, row2 ÷3
    #
    # Eight further probes found no rule reproducing the second: the divisor is the
    # true frequency for some rows and something else for others, and running-max,
    # previous-present-row and sorted-run-length all fit some probes and break
    # others. `embedding`'s scaling is self-consistent and matches the documentation,
    # so that one is implemented; copying an answer nobody can state a rule for would
    # be a wrong number under an argument that reads as a tuning knob.
    if scale_grad_by_freq:
        _unsupported("embedding_bag(scale_grad_by_freq=True) — torch's own bag "
                     "disagrees with `embedding(...).sum(1)` under this flag")
    if sparse:
        _unsupported("embedding_bag(sparse=True) — there is no sparse gradient here")
    if max_norm is not None:
        _renorm_rows(weight, input.data, max_norm, norm_type)
    picked = embedding(input, weight)
    if per_sample_weights is not None:
        picked = picked * _wrap(per_sample_weights).reshape(
            *_wrap(per_sample_weights).data.shape, 1)

    # **`padding_idx` leaves the bag rather than contributing zero to it.** Under
    # `sum` those are the same thing; under `mean` they are not, because the
    # padded entry has to leave the denominator too (measured against torch).
    keep = None
    if padding_idx is not None:
        keep = (_np.asarray(input.data).astype(int) != padding_idx).astype(_DEFAULT_DTYPE)
        picked = picked * _wrap(keep).reshape(*keep.shape, 1)

    def squash(part, dim, mask=None):
        if mode == "sum":
            return part.sum(dim=dim)
        if mode == "max":
            return amax(part, dim=dim)
        if mask is None:
            return part.mean(dim=dim)
        counted = _np.maximum(mask.sum(axis=dim, keepdims=True), 1.0)
        return part.sum(dim=dim) / _wrap(counted.reshape(-1, 1) if dim == 1
                                         else counted.reshape(1, -1)[0])

    if offsets is None:
        return squash(picked, dim=1, mask=keep)
    bounds = [int(v) for v in _np.asarray(_wrap(offsets).data).reshape(-1)]
    # **`include_last_offset` means the last entry closes the final bag** rather
    # than opening a new one — so the count of bags is one fewer than the offsets,
    # not one more than the gaps between them.
    if not include_last_offset:
        bounds = bounds + [int(input.data.size)]
    parts = []
    for i in range(len(bounds) - 1):
        lo, hi = bounds[i], bounds[i + 1]
        piece = picked[lo:hi]
        piece_mask = None if keep is None else keep[lo:hi].reshape(-1, 1)
        if mode == "mean" and piece_mask is not None:
            total = max(float(piece_mask.sum()), 1.0)
            parts.append(piece.sum(dim=0) / _wrap(_np.array(total, dtype=_DEFAULT_DTYPE)))
        else:
            parts.append(squash(piece, dim=0))
    return stack(parts, dim=0)


def gumbel_softmax(logits, tau=1.0, hard=False, eps=1e-10, dim=-1):
    """Choose one at random **in a way the gradient can flow through.**

    Drawing one category is not differentiable, so Gumbel noise is added and
    `softmax` smooths it. The smaller `tau` is, the more it concentrates on one
    side.

    With `hard=True` the answer is 0/1 and **the gradient is the smooth one's** —
    the familiar `hard - soft.detach() + soft` trick, whose value is hard and whose
    derivative is soft. Without keeping those two apart this function means
    nothing.

    **`eps` is accepted and not used, which is what torch does with it.** torch
    deprecated it when its noise moved to an exponential draw that needs no floor;
    the parameter survives only so that old calls keep parsing.

    This used to put the caller's `eps` inside the noise, and it moved the answer:
    at `eps=1e-1` a single output shifted by 0.49 and the mean of 20,000 draws went
    from `[0.728, 0.185, 0.087]` to `[0.743, 0.177, 0.080]`, while torch's was
    **bit-identical** at both values. So this was one of the few places where the
    two libraries returned different numbers for the same call — and no check here
    could see it, because every structural check asks whether an accepted argument
    is *used*, and this one was being used enthusiastically.

    The binding and borch.ts were mended first; the core kept the old formula for
    another day because the fix went in one layer at a time and nothing compares
    this layer against torch on the question *what does torch do with this
    argument.* Found by the axis built to ask exactly that, on its first probe.
    """
    logits = _wrap(logits)
    if float(eps) != 1e-10:
        _warnings.warn("`eps` parameter is deprecated and has no effect.",
                       stacklevel=2)
    # The constant stays inside, where borch.ts also pins it, so all three draw the
    # same noise. What changed is that the **caller** can no longer reach it.
    floor = 1e-10
    u = _rng.random(logits.data.shape).astype(logits.data.dtype)
    gumbel = -_np.log(-_np.log(u + floor) + floor)
    soft = softmax((logits + Tensor(gumbel)) / tau, dim=dim)
    if not hard:
        return soft
    at = _np.argmax(soft.data, axis=dim)
    onehot = _np.zeros_like(soft.data)
    _np.put_along_axis(onehot, _np.expand_dims(at, dim), 1.0, axis=dim)
    return Tensor(onehot) - soft.detach() + soft


def upsample(input, size=None, scale_factor=None, mode="nearest", align_corners=None):
    """`interpolate`'s old name. torch warns that it is deprecated and goes on
    accepting it."""
    return interpolate(input, size, scale_factor, mode, align_corners)


def upsample_nearest(input, size=None, scale_factor=None):
    return interpolate(input, size, scale_factor, mode="nearest")


def upsample_bilinear(input, size=None, scale_factor=None):
    """**`align_corners=True`.** `interpolate(mode='bilinear')` defaults to false,
    so making one an alias of the other by name alone puts the edges off — the
    interior is similar enough that the eye does not part them."""
    return interpolate(input, size, scale_factor, mode="bilinear", align_corners=True)


# ── bitwise operations and integer maths ────────────────────────────────────
#
# **On `bool` they become logical operations.** torch branches on the dtype —
# `bitwise_and` on booleans is `logical_and` and `bitwise_not` on booleans is
# `logical_not`. Asked with integers alone that branch never runs at all.
#
# There is no gradient. Bits are steps and there is nothing to flow, and torch
# does not build gradients for integer dtypes either.

def _bitwise(name, op, bool_op=None):
    def call(input, other):
        input, other = _wrap(input), _wrap(other)
        # **Both have to be boolean, not just the first.** torch promotes to
        # `int64` when either side is not, so `bitwise_and(bool, int64)` is `int64`
        # there and came back `bool` here — the integer operand narrowed to a bit.
        if (input.data.dtype.kind == "b" and other.data.dtype.kind == "b"
                and bool_op is not None):
            return Tensor(bool_op(input.data, other.data))
        return Tensor(op(input.data.astype(_np.int64),
                         other.data.astype(_np.int64)))
    call.__name__ = name
    return call


bitwise_and = _bitwise("bitwise_and", _np.bitwise_and, _np.logical_and)
bitwise_or = _bitwise("bitwise_or", _np.bitwise_or, _np.logical_or)
bitwise_xor = _bitwise("bitwise_xor", _np.bitwise_xor, _np.logical_xor)
bitwise_left_shift = _bitwise("bitwise_left_shift", _np.left_shift)
bitwise_right_shift = _bitwise("bitwise_right_shift", _np.right_shift)


def bitwise_not(input):
    input = _wrap(input)
    if input.data.dtype.kind == "b":
        return Tensor(_np.logical_not(input.data))
    return Tensor(_np.bitwise_not(input.data.astype(_np.int64)))


def gcd(input, other):
    input, other = _wrap(input), _wrap(other)
    return Tensor(_np.gcd(input.data.astype(_np.int64), other.data.astype(_np.int64)))


def lcm(input, other):
    input, other = _wrap(input), _wrap(other)
    return Tensor(_np.lcm(input.data.astype(_np.int64), other.data.astype(_np.int64)))


def gcd_(a, b):
    a = _wrap(a)
    return a._inplace(lambda: gcd(a, b), "gcd_")


def lcm_(a, b):
    a = _wrap(a)
    return a._inplace(lambda: lcm(a, b), "lcm_")


def nextafter(input, other):
    """**The next representable number** from `input` towards `other`. It moves by
    one ulp and no more."""
    input, other = _wrap(input), _wrap(other)
    return input._make(_np.nextafter(input.data, other.data), (input,), lambda g: (g,),
                   "NextafterBackward0")


def frexp(input):
    """`x = mantissa × 2^exponent`. **The exponent is int32** — torch does that
    (measured)."""
    input = _wrap(input)
    mantissa, exponent = _np.frexp(input.data)
    return _Frexp(Tensor(mantissa.astype(input.data.dtype)),
                  Tensor(exponent.astype(_np.int32)))


def logcumsumexp(input, dim):
    """A running `logsumexp`. Computed **without overflow** — the maximum is
    subtracted, summed and restored."""
    input = _wrap(input)
    # **`logsumexp` takes integers and this does not** (measured). A hole in
    # torch's kernels rather than a rule, and handing back a value here means that
    # code breaks against real torch.
    _refuses_nonfloat_kernel(input.data, "logcumsumexp", "logcumsumexp_out_cpu")
    data = input.data
    big = _np.max(data, axis=dim, keepdims=True)
    shifted = _np.exp(data - big)
    total = _np.cumsum(shifted, axis=dim)
    out = _np.log(total) + big
    soft = shifted / total          # each position's share of the running sum

    def back(g):
        # It accumulates from the back — position `i` enters every running term
        # from `i` onwards.
        gg = _np.asarray(g)
        flipped = _np.flip(_np.cumsum(_np.flip(gg / total, axis=dim), axis=dim),
                           axis=dim)
        return (flipped * shifted,)

    del soft
    return input._make(out, (input,), back, "LogcumsumexpBackward0")


# **These four have no docstring in torch at all**, so the prose reader had nothing
# to read and the axis compared them by arity — same count, different name, no row.
# Only the core↔borch.ts axis could see them, and only because borch.ts had already
# taken torch's spelling. The names are `max` and `min`, confirmed by calling torch
# with each: `x.clamp_max(max=2.0)` is taken and `x.clamp_max(value=2.0)` is not.
def clamp_max(input, max):                                   # noqa: A002
    return clamp(input, None, max)


def clamp_min(input, min):                                   # noqa: A002
    return clamp(input, min, None)


def clamp_max_(input, max):                                  # noqa: A002
    input = _wrap(input)                                     # noqa: A001
    return input._inplace(lambda: clamp(input, None, max), "clamp_max_")


def clamp_min_(input, min):                                  # noqa: A002
    input = _wrap(input)                                     # noqa: A001
    return input._inplace(lambda: clamp(input, min, None), "clamp_min_")


def arctan2(input, other):
    return atan2(input, other)


def fill(x, value):
    """**Not in place.** One character apart from `fill_` and a different job —
    this one produces a new tensor and leaves the original alone (measured)."""
    x = _wrap(x)
    return Tensor(_np.full_like(x.data, value))


def detach_(x):
    """Cut the graph on **the same tensor.** `detach()` produces a new one and
    this is in place."""
    x = _wrap(x)
    x.requires_grad = False
    x._parents = ()
    x._backward = None
    return x


def _i1(x):
    """The order-1 modified Bessel function — the derivative of `i0`. numpy gives
    `i0` alone, so it is built as a series.

    The series is `i1(x) = Σ (x/2)^(2k+1) / (k! (k+1)!)`, and carrying each term
    forward by multiplying the previous one keeps the factorials from overflowing.
    **Every term is positive and they do not cancel**, so there is nowhere to lose
    digits — the sign is attached at the end, the function being odd.

    Convergence rather than an approximation. Compared densely against torch over
    [-30, 30] and confirmed to stay under a relative error of 1e-6 within float32
    (`tests/test_bessel.py`).
    """
    a = _np.abs(_np.asarray(x, dtype=_np.float64))
    half = a / 2.0
    term = half.copy()                     # the k=0 term
    total = term.copy()
    for k in _builtin_range(1, 400):
        term = term * (half * half) / (k * (k + 1.0))
        total = total + term
        if _np.all(term <= _np.abs(total) * 1e-17):
            break
    return _np.sign(_np.asarray(x, dtype=_np.float64)) * total


def i0(input):
    """The order-0 modified Bessel function. `kaiser_window` stands on it."""
    input = _wrap(input)
    data = _float_in(_np.asarray(input.data))
    # Its derivative is `i1`. Here there was only a bare `Tensor(...)` and the
    # graph was quietly cut, and until `backward()` was called no value check
    # showed it.
    return input._make(_np.i0(data).astype(data.dtype), (input,),
                   lambda g: (_np.asarray(g) * _i1(data),), "I0Backward0")


def i0_(x):
    x = _wrap(x)
    return x._inplace(lambda: i0(x), "i0_")


def mvlgamma(input, p):
    """The multivariate log gamma.
    `log Γ_p(x) = p(p−1)/4 · log π + Σ log Γ(x + (1−i)/2)`."""
    input = _wrap(input)
    out = _np.full_like(input.data, p * (p - 1) / 4.0 * _math.log(_math.pi))
    for i in range(1, p + 1):
        out = out + _np.asarray(lgamma(input + (1 - i) / 2.0).data)
    return Tensor(out.astype(input.data.dtype))


# ── window functions ────────────────────────────────────────────────────────
#
# **`periodic` is the default and it adds one to the length.** When true, torch
# builds a symmetric window of `N+1` and discards the last (measured:
# `hann_window(5)` equals the first five of the symmetric 6 exactly). Asked with
# false alone that rule never surfaces.

def _window(n, periodic, shape, dt=None, requires_grad=False):
    """The gate the five share. **`dtype=` and `requires_grad=` are taken here
    too.**

    All five were **swallowing** the two into `**kw`. The gradient side is worse
    than the dtype — `hann_window(8, requires_grad=True)` quietly giving a leaf
    with no gradient means code training that window runs without an error and
    **that one thing does not move.** The reason fourteen factories were gathered
    under `_made`, and these five were outside at the time.
    """
    if n <= 0:
        return _made(_np.zeros(0, dtype=_DEFAULT_DTYPE), dt, requires_grad)
    if n == 1:
        return _made(_np.ones(1, dtype=_DEFAULT_DTYPE), dt, requires_grad)
    total = n + 1 if periodic else n
    k = _np.arange(total, dtype=_np.float64)
    return _made(shape(k, total)[:n].astype(_DEFAULT_DTYPE), dt, requires_grad)


def bartlett_window(window_length, periodic=True, dtype=None,
                    requires_grad=False):
    return _window(window_length, periodic,
                   lambda k, n: 1.0 - _np.abs(2.0 * k / (n - 1) - 1.0),
                   dtype, requires_grad)


def hann_window(window_length, periodic=True, dtype=None,
                requires_grad=False):
    return _window(window_length, periodic,
                   lambda k, n: 0.5 - 0.5 * _np.cos(2 * _np.pi * k / (n - 1)),
                   dtype, requires_grad)


def hamming_window(window_length, periodic=True, alpha=0.54, beta=0.46,
                   dtype=None, requires_grad=False, *, out=None):
    _no_out(out)
    return _window(window_length, periodic,
                   lambda k, n: alpha - beta * _np.cos(2 * _np.pi * k / (n - 1)),
                   dtype, requires_grad)


def blackman_window(window_length, periodic=True, dtype=None,
                    requires_grad=False):
    def shape(k, n):
        t = 2 * _np.pi * k / (n - 1)
        return 0.42 - 0.5 * _np.cos(t) + 0.08 * _np.cos(2 * t)
    return _window(window_length, periodic, shape, dtype, requires_grad)


def kaiser_window(window_length, periodic=True, beta=12.0, dtype=None,
                  requires_grad=False):
    def shape(k, n):
        half = (n - 1) / 2.0
        return _np.i0(beta * _np.sqrt(1.0 - ((k - half) / half) ** 2)) / _np.i0(beta)
    return _window(window_length, periodic, shape, dtype, requires_grad)


# ── the names that exist only at torch's top level ──────────────────────────
#
# torch keeps some of `nn.functional`'s at the top level too, and **the signatures
# do not match.** The top-level ones are raw ATen operations, so the argument order
# differs and the enums are integers. The same computation called a different way,
# so the computation is kept as one copy and only the positions move here.

def nan_to_num_(x, nan=0.0, posinf=None, neginf=None):
    x = _wrap(x)
    return x._inplace(lambda: nan_to_num(x, nan, posinf, neginf), "nan_to_num_")


def dropout_(x, p=0.5, train=True):
    x = _wrap(x)
    return x._inplace(lambda: dropout(x, p, train), "dropout_")


def feature_dropout(x, p=0.5, train=True):
    """**Drops whole channels** — the same computation as `F.dropout2d`
    (measured). A name that exists at the top level alone."""
    return dropout2d(x, p, train)


def feature_dropout_(x, p=0.5, train=True):
    x = _wrap(x)
    return x._inplace(lambda: dropout2d(x, p, train), "feature_dropout_")


def alpha_dropout_(x, p=0.5, train=True):
    x = _wrap(x)
    return x._inplace(lambda: alpha_dropout(x, p, train), "alpha_dropout_")


def feature_alpha_dropout_(x, p=0.5, train=True):
    x = _wrap(x)
    return x._inplace(lambda: feature_alpha_dropout(x, p, train),
                      "feature_alpha_dropout_")


def batch_norm_aten(x, weight, bias, running_mean, running_var, training,
                    momentum, eps, cudnn_enabled=False):
    """**A different argument order from `F.batch_norm`.** Here the weights come
    **before** the statistics.

    The same computation with the positions swapped, so forwarded as-is the
    weights get used as the mean — not an exception but a plausibly different
    value.
    """
    return batch_norm(x, running_mean, running_var, weight, bias, training,
                      momentum, eps)


def grid_sampler(x, grid, interpolation_mode=0, padding_mode=0,
                 align_corners=False):
    """**The enums are integers.** 0 and 1 are `bilinear` and `nearest`, and for
    the padding 0, 1 and 2 are `zeros`, `border` and `reflection`. The side that
    takes them by name is `F.grid_sample`."""
    modes = ("bilinear", "nearest", "bicubic")
    pads = ("zeros", "border", "reflection")
    return grid_sample(x, grid, modes[int(interpolation_mode)],
                       pads[int(padding_mode)], align_corners)


def ctc_loss_aten(log_probs, targets, input_lengths, target_lengths, blank=0,
                  reduction=1, zero_infinity=False):
    """**`reduction` is an integer** — 0, 1 and 2 are `none`, `mean` and `sum`.

    The side that takes it by name is `F.ctc_loss`, whose default is `"mean"`. The
    default `1` here points at the same place.
    """
    kinds = ("none", "mean", "sum")
    return ctc_loss(log_probs, targets, input_lengths, target_lengths, blank,
                    kinds[int(reduction)], zero_infinity)


# ── shape and indexing ──────────────────────────────────────────────────────
#
# **`as_strided` is a view in torch.** This produces a copy.
#
# torch is built around viewing one storage through several frames, so writing
# into an `as_strided` result changes the original. A borch.ts tensor **holds its
# own GPU buffer**, so that view is not representable, and producing a real view
# here alone makes the three implementations diverge — a divergence invisible in
# the values and visible **only on a write**, which is the worst kind. All three
# are matched as copies.
#
# The read-only uses (making windows, walking a diagonal) run unchanged, and code
# that writes into the view is rare even in torch. `as_strided_scatter` does
# "write into those positions" properly instead.
#
# **"Only on a write" was not the whole of it**, and the line above said so for
# months. `as_strided_` twice shows it without writing anywhere: the first call
# leaves a 2×2 where torch leaves a 2×2 *over an eight-element storage*, so a
# second call with stride 4 reads position 4 — which torch still has and this
# does not. torch answers; here it is an `IndexError`.
#
# Measured, not reasoned: the first call agrees exactly, and only the second parts.
# Left as it stands, because the alternative is a storage model, and the shape of
# the divergence is the one already chosen. Written down so the next reader does
# not have to rediscover that the sentence above is narrower than it sounds.

def _strided_flat(size, stride, offset):
    """Build the **flat index table** the strides point at. Its shape is
    `size`."""
    size = tuple(int(s) for s in size)
    stride = tuple(int(s) for s in stride)
    flat = _np.full(size, int(offset), dtype=_np.int64)
    for axis, step in enumerate(stride):
        shape = [1] * len(size)
        shape[axis] = size[axis]
        flat = flat + _np.arange(size[axis], dtype=_np.int64).reshape(shape) * step
    return flat


def as_strided(input, size, stride, storage_offset=0):
    """Read a flat storage **with different strides.** They may overlap and they
    may skip."""
    input = _wrap(input)
    flat = _strided_flat(size, stride, storage_offset)
    out = input.data.reshape(-1)[flat]

    def back(g):
        # Overlapping positions **accumulate** — a cell read twice receives
        # gradient twice.
        acc = _np.zeros(input.data.size, dtype=_np.asarray(g).dtype)
        _np.add.at(acc, flat.reshape(-1), _np.asarray(g).reshape(-1))
        return (acc.reshape(input.data.shape),)

    return input._make(out, (input,), back, "AsStridedBackward0")


def as_strided_(t, size, stride, storage_offset=0):
    t = _wrap(t)
    return t._inplace(lambda: as_strided(t, size, stride, storage_offset),
                      "as_strided_")


def _marks(shape):
    """The **flat index table** of an array of shape `shape`. It records where in
    the storage each cell sits."""
    return _np.arange(int(_np.prod(shape)) if shape else 1,
                      dtype=_np.int64).reshape(shape)


def _scatter_into(t, src, spots, name):
    """Put `src` into the **flat positions** `spots` points at, in a copy of `t`.

    `select_scatter`, `slice_scatter`, `diagonal_scatter` and `as_strided_scatter`
    all take this shape, and the only difference is **which positions.**

    **The write and the read-back use the same index table.** Written as two
    copies there is room for the forward to be right and the gradient alone to
    drift, and that drift is invisible because the values are plausible —
    especially where the axis order is reversed, as it is on a diagonal.
    """
    t, src = _wrap(t), _wrap(src)
    flat = _np.asarray(spots).reshape(-1)
    out = t.data.copy().reshape(-1)
    out[flat] = _np.broadcast_to(_np.asarray(src.data),
                                 _np.asarray(spots).shape).reshape(-1)
    out = out.reshape(t.data.shape)

    def back(g):
        g = _np.asarray(g)
        keep = _np.ones(t.data.size, dtype=g.dtype)
        keep[flat] = 0.0
        into = g.reshape(-1)[flat].reshape(_np.asarray(spots).shape)
        return (g * keep.reshape(t.data.shape),
                into.reshape(src.data.shape))

    return t._make(out, (t, src), back, name)


def as_strided_scatter(input, src, size, stride, storage_offset=0):
    """Produce a copy **written into** at the positions `as_strided` was
    viewing."""
    return _scatter_into(input, src, _strided_flat(size, stride, storage_offset),
                         "AsStridedScatterBackward0")


def select_scatter(input, src, dim, index):
    """A copy with the slice `select` was taking **swapped out.**"""
    spots = _marks(_wrap(input).data.shape)[(slice(None),) * dim + (int(index),)]
    return _scatter_into(input, src, spots, "SelectScatterBackward0")


def slice_scatter(input, src, dim=0, start=None, end=None, step=1):
    """A copy with the `x[..., start:end:step]` positions **swapped out.** `step`
    is the point — measured at 1 alone, nobody looks at the skipped
    positions."""
    spots = _marks(_wrap(input).data.shape)[
        (slice(None),) * dim + (slice(start, end, step),)]
    return _scatter_into(input, src, spots, "SliceScatterBackward0")


def diagonal_scatter(input, src, offset=0, dim1=0, dim2=1):
    """A copy with the diagonal positions **swapped out.** A non-zero `offset`
    shifts them.

    The positions are taken with `_np.diagonal` — that side has the convention of
    **sending the diagonal axis to the back** and torch does the same. Written by
    hand, the order diverges once there is a batch axis.
    """
    spots = _np.diagonal(_marks(_wrap(input).data.shape), offset=offset,
                         axis1=dim1, axis2=dim2)
    return _scatter_into(input, src, spots, "DiagonalScatterBackward0")


def diag_embed(input, offset=0, dim1=-2, dim2=-1):
    """**Spread the last axis onto a diagonal**, adding an axis. The inverse of
    `diagonal`."""
    input = _wrap(input)
    # `abs` is **a tensor function** in this file — the Python builtin is
    # shadowed. `_np.abs` is this file's rule, and forgetting it arrives as
    # `'int' object has no attribute 'abs'`.
    n = input.data.shape[-1] + int(_np.abs(offset))
    rank = input.data.ndim + 1
    d1, d2 = dim1 % rank, dim2 % rank
    shape = list(input.data.shape[:-1])
    for at in sorted((d1, d2)):
        shape.insert(at, n)
    out = _np.zeros(tuple(shape), dtype=input.data.dtype)
    spots = _np.diagonal(_marks(out.shape), offset=offset, axis1=d1, axis2=d2)
    out.reshape(-1)[_np.asarray(spots).reshape(-1)] = input.data.reshape(-1)

    def back(g):
        g = _np.asarray(g)
        return (g.reshape(-1)[_np.asarray(spots).reshape(-1)]
                .reshape(input.data.shape),)

    return input._make(out, (input,), back, "DiagEmbedBackward0")


def tensor_split(input, sections, dim=0):                    # noqa: A002
    """**The remainder is shared out from the front.** Splitting 10 into 4 gives
    3, 3, 2, 2 (measured).

    Different from `chunk` — that one fills the front large and the last piece
    takes what is left. Measured at sizes that divide evenly the two functions look
    the same.

    **The parameter is `sections`, and torch's docstring says
    `indices_or_sections`.** Called with the documented name torch answers
    `received an invalid combination of arguments`; called with `sections` it works,
    on the function and on the method alike. Unlike `split` next door, this pair
    agrees with itself.
    """
    input = _wrap(input)                                     # noqa: A001
    if isinstance(sections, (list, tuple)):
        return tuple(_wrap(p) for p in
                     _split_at(input, list(sections), dim))
    k = int(sections)
    n = input.data.shape[dim]
    base, extra = divmod(n, k)
    cuts, at = [], 0
    for i in range(k - 1):
        at += base + (1 if i < extra else 0)
        cuts.append(at)
    return tuple(_wrap(p) for p in _split_at(input, cuts, dim))


def _split_at(t, cuts, dim):
    """Split by a **list of cut positions.** Each piece returns its gradient to
    its own place."""
    out, prev = [], 0
    for stop in list(cuts) + [t.data.shape[dim]]:
        out.append(narrow(t, dim, prev, max(0, stop - prev)))
        prev = stop
    return out


def split_with_sizes(t, split_sizes, dim=0):
    """Split by a **list of piece sizes.** The same as `split` taking a list."""
    t = _wrap(t)
    cuts, at = [], 0
    for s in list(split_sizes)[:-1]:
        at += int(s)
        cuts.append(at)
    return tuple(_split_at(t, cuts, dim))


def unravel_index(indices, shape):
    """Unpack a flat index into per-axis indices. **One tensor per axis, returned
    as a tuple** (measured)."""
    idx = _as_index(indices)
    return tuple(Tensor(part.astype(_np.int64))
                 for part in _np.unravel_index(idx, tuple(int(s) for s in shape)))


def unique_consecutive(t, return_inverse=False, return_counts=False, dim=None):
    """Collapses **consecutive** duplicates only. Unlike `unique` it does not
    sort — `[1,1,2,2,1]` becomes `[1,2,1]` (measured). Measured on sorted input
    alone the two look the same."""
    t = _wrap(t)
    data = t.data
    if dim is None:
        flat = data.reshape(-1)
        keep = _np.ones(flat.shape[0], dtype=bool)
        keep[1:] = flat[1:] != flat[:-1]
    else:
        moved = _np.moveaxis(data, dim, 0)
        rows = moved.reshape(moved.shape[0], -1)
        keep = _np.ones(rows.shape[0], dtype=bool)
        keep[1:] = _np.any(rows[1:] != rows[:-1], axis=1)
    starts = _np.flatnonzero(keep)
    groups = _np.cumsum(keep) - 1
    values = (Tensor(flat[keep]) if dim is None
              else Tensor(_np.moveaxis(_np.moveaxis(data, dim, 0)[keep], 0, dim)))
    if not return_inverse and not return_counts:
        return values
    out = [values]
    if return_inverse:
        shape = data.shape if dim is None else (data.shape[dim],)
        out.append(Tensor(groups.reshape(shape).astype(_np.int64)))
    if return_counts:
        edges = _np.append(starts, keep.shape[0])
        out.append(Tensor(_np.diff(edges).astype(_np.int64)))
    return tuple(out)


def masked_scatter(t, mask, source):
    """Fill the positions where the mask is true from `source`, **in flat
    order.**

    Which value lands where is the point — as many taken from the front as there
    are true positions. Measured with a source of matching shape alone, that order
    never surfaces.
    """
    t, source = _wrap(t), _wrap(source)
    m = _np.broadcast_to(_np.asarray(mask.data if isinstance(mask, Tensor)
                                     else mask).astype(bool), t.data.shape)
    out = t.data.copy()
    out[m] = _np.asarray(source.data).reshape(-1)[:int(m.sum())]

    def back(g):
        g = _np.asarray(g)
        into = _np.zeros(source.data.size, dtype=g.dtype)
        into[:int(m.sum())] = g[m]
        return (g * (~m), into.reshape(source.data.shape))

    return t._make(out, (t, source), back, "MaskedScatterBackward0")


def masked_scatter_(t, mask, source):
    t = _wrap(t)
    return t._inplace(lambda: masked_scatter(t, mask, source), "masked_scatter_")


def index_put(t, indices, values, accumulate=False):
    """Take one index tensor per axis and write into those positions.

    **They part where the indices collide** — with `accumulate` they add and
    otherwise the last write survives. Measured with non-colliding indices the two
    branches give the same answer.
    """
    t, values = _wrap(t), _wrap(values)
    where = tuple(_as_index(i) for i in indices)
    out = t.data.copy()
    if accumulate:
        _np.add.at(out, where, _np.asarray(values.data))
    else:
        out[where] = _np.asarray(values.data)

    def back(g):
        g = _np.asarray(g)
        into = g[where]
        if accumulate:
            return (g, into)
        keep = _np.ones(t.data.shape, dtype=g.dtype)
        keep[where] = 0.0
        return (g * keep, into)

    return t._make(out, (t, values), back, "IndexPutBackward0")


def index_put_(t, indices, values, accumulate=False):
    t = _wrap(t)
    return t._inplace(lambda: index_put(t, indices, values, accumulate),
                      "index_put_")


def put(t, index, source, accumulate=False):
    """Write by index into **the flattened tensor** — it has no notion of an
    axis. The inverse of `take`."""
    t, source = _wrap(t), _wrap(source)
    idx = _as_index(index).reshape(-1)
    over = idx[(idx >= t.data.size) | (idx < -t.data.size)]
    if over.size:
        raise IndexError(
            f"out of range: tried to access index {int(over[0])} on a tensor of "
            f"{t.data.size} elements.")
    out = t.data.copy().reshape(-1)
    if accumulate:
        _np.add.at(out, idx, _np.asarray(source.data).reshape(-1))
    else:
        out[idx] = _np.asarray(source.data).reshape(-1)
    out = out.reshape(t.data.shape)

    def back(g):
        g = _np.asarray(g)
        into = g.reshape(-1)[idx].reshape(source.data.shape)
        if accumulate:
            return (g, into)
        keep = _np.ones(t.data.size, dtype=g.dtype)
        keep[idx] = 0.0
        return (g * keep.reshape(t.data.shape), into)

    return t._make(out, (t, source), back, "PutBackward0")


# The arithmetic of the reducing writes. `include_self` decides **whether the
# original value enters as the first term.**
_REDUCE_OPS = {
    "sum": (0.0, _np.add),
    "prod": (1.0, _np.multiply),
    "amax": (-_np.inf, _np.maximum),
    "amin": (_np.inf, _np.minimum),
}


def _reduce_into(out, where, values, reduce, include_self):
    """Combine `values` into `out[where]` through `reduce`. Colliding positions
    accumulate."""
    if reduce == "mean":
        total = _np.zeros(out.shape, dtype=out.dtype)
        count = _np.zeros(out.shape, dtype=out.dtype)
        _np.add.at(total, where, values)
        _np.add.at(count, where, _np.ones_like(values))
        touched = count > 0
        if include_self:
            total = total + out
            count = count + 1.0
        # **Untouched positions stay as they are.** They are branched out so that
        # nothing divides by zero.
        merged = _np.where(touched, total / _np.where(count == 0, 1.0, count), out)
        out[...] = merged
        return
    start, op = _REDUCE_OPS[reduce]
    acc = _np.full(out.shape, start, dtype=out.dtype)
    op.at(acc, where, values)
    touched = _np.zeros(out.shape, dtype=bool)
    _np.logical_or.at(touched, where, True)
    if include_self:
        acc = op(acc, out)
    out[...] = _np.where(touched, acc, out)


def index_reduce(input, dim, index, source, reduce, include_self=True):
    """Combine the **rows** the indices point at. `include_self` puts the original
    value in as the first term.

    **Measured with add and multiply alone that flag is invisible** — multiplying
    into an array filled with 1s gives the same answer on or off (measured). Mean
    and minimum show the split.
    """
    input, source = _wrap(input), _wrap(source)
    out = input.data.copy()
    where = (slice(None),) * dim + (_as_index(index),)
    _reduce_into(out, where, _np.asarray(source.data), reduce, include_self)
    return Tensor(out)


def scatter_reduce(input, dim, index, src, reduce, include_self=True):
    """`scatter`'s place, **combining instead of overwriting.**

    The reductions are `sum`, `prod`, `amax`, `amin` and `mean`, and with
    `include_self` the `mean` counts the original value as one term in the divisor
    too (measured).
    """
    input, src = _wrap(input), _wrap(src)
    idx = _as_index(index)
    _in_bounds(idx, input.data.shape[dim], dim)
    out = input.data.copy()
    grid = _np.indices(idx.shape)
    where = list(grid)
    where[dim] = idx
    _reduce_into(out, tuple(where), _np.asarray(src.data), reduce, include_self)
    return Tensor(out)


def renorm(input, p, dim, maxnorm):
    """Pull **each slice's norm below `maxnorm`**, slicing along `dim`.

    A slice that is already small is **left alone** — made all large, that
    condition never surfaces.

    **The scale is not a constant.** `x` appears inside the scale as well, so
    writing the gradient as `g·s` alone leaves the forward right and the backward
    wrong — a place invisible because the values are plausible. It diverges on the
    clipped slices alone, so measured with everything small that too never
    surfaces.
    """
    input = _wrap(input)
    x = input.data
    # `dim % x.ndim` wraps rather than complains, so an axis that does not exist
    # became the last one and the answer came back plausible. torch raises.
    axes = tuple(a for a in range(x.ndim) if a != _pos_dim(input, dim))
    norms = _np.sum(_np.abs(x) ** p, axis=axes, keepdims=True) ** (1.0 / p)
    # torch adds a very small number so that nothing divides by zero.
    cut = norms > maxnorm
    scale = _np.where(cut, maxnorm / (norms + 1e-7), 1.0)

    def back(g):
        g = _np.asarray(g)
        # `∂n/∂x = n^(1-p)·|x|^(p-1)·sign(x)`, `∂s/∂x = -M/(n+ε)²·∂n/∂x`.
        dn = norms ** (1.0 - p) * _np.abs(x) ** (p - 1) * _np.sign(x)
        ds = _np.where(cut, -maxnorm / (norms + 1e-7) ** 2 * dn, 0.0)
        dot = _np.sum(g * x, axis=axes, keepdims=True)
        return (g * scale + dot * ds,)

    return input._make(x * scale, (input,), back, "RenormBackward0")


def cartesian_prod(*tensors):
    """Every pair. **Given one it is simply that one** (measured) — it stays
    1-D."""
    for a in tensors:
        _rank(_wrap(a).data, (1,), "Expect a 1D vector, but got shape {shape}")
    arrays = [_wrap(a).data.reshape(-1) for a in tensors]
    if len(arrays) == 1:
        return Tensor(arrays[0].copy())
    mesh = _np.meshgrid(*arrays, indexing="ij")
    return Tensor(_np.stack([m.reshape(-1) for m in mesh], axis=1))


def combinations(input, r=2, with_replacement=False):
    """Combinations of `r`. **Order does not count**, and repetition is a
    separate option."""
    from itertools import combinations as _comb, combinations_with_replacement

    _rank(_wrap(input).data, (1,), "Expect a 1D vector, but got shape {shape}")
    flat = _wrap(input).data.reshape(-1)
    pick = combinations_with_replacement if with_replacement else _comb
    rows = [list(c) for c in pick(range(flat.shape[0]), r)]
    if not rows:
        return Tensor(_np.zeros((0, r), dtype=flat.dtype))
    return Tensor(flat[_np.asarray(rows, dtype=_np.int64)])


def tril_indices(row, col, offset=0, dtype=None, requires_grad=False):
    """The positions of the lower triangle. **A `(2, count)` int64 table**
    (measured) — not pairs of positions but a row of rows and a row of
    columns."""
    r, c = _np.tril_indices(int(row), int(offset), int(col))
    return _made(_np.stack([r, c]).astype(_np.int64), dtype, requires_grad)


def triu_indices(row, col, offset=0, dtype=None, requires_grad=False):
    r, c = _np.triu_indices(int(row), int(offset), int(col))
    return _made(_np.stack([r, c]).astype(_np.int64), dtype, requires_grad)


def vander(x, N=None, increasing=False):
    """The Vandermonde matrix. **The default has the powers decreasing** — the
    last column is 1 (measured)."""
    x = _wrap(x)
    _rank(x.data, (1,), "x must be a one-dimensional tensor.")
    n = x.data.shape[0] if N is None else int(N)
    powers = _np.arange(n, dtype=_np.float64)
    if not increasing:
        powers = powers[::-1]
    # **The result's type is the promoted one, not the input's.** Casting back to
    # `x.data.dtype` gave a Vandermonde matrix of booleans, where torch answers in
    # `int64` — every power of `True` is `True`, so the values looked right.
    kind = _arith_in(x.data)
    out = x.data.reshape(-1, 1).astype(_np.float64) ** powers.reshape(1, -1)
    return Tensor(out.astype(kind.dtype))


def chain_matmul(*matrices):
    """Multiply several matrices in succession. `linalg.multi_dot` takes the same
    thing as a list."""
    mats = list(matrices[0]) if len(matrices) == 1 and \
        isinstance(matrices[0], (list, tuple)) else list(matrices)
    for m in mats:
        _rank(_wrap(m).data, (2,), "Tensor dimension is {n}, expected 2 instead.")
    out = _wrap(mats[0])
    for m in mats[1:]:
        out = matmul(out, _wrap(m))
    return out


def ger(input, vec2):
    """The old name for the outer product. The same as `outer`."""
    return outer(input, vec2)


def mv(mat, vec):
    """Matrix times vector. `matmul` does the job and torch gives it its own
    name."""
    return matmul(_wrap(mat), _wrap(vec))


# ── the addmm family ────────────────────────────────────────────────────────
#
# All eight take one shape — `β·input + α·(some product)`. The only difference is
# **which product**, so that alone is passed in.
#
# **At `beta` of 0, `input` is not looked at at all.** It is not `0 · input` —
# torch does not even read `input` there, so a NaN in it leaves the result sound
# and the gradient 0 (measured). Written as `0 * input` the NaN spreads, and that
# difference is never visible with ordinary input.

def _blend(base, product, beta, alpha):
    """`β·base + α·product`.

    **`β == 0` skips the value and stays in the graph.** It has to be both —

    - written as `base * 0`, a NaN in it makes the result NaN. torch stays sound.
    - and taken out of the graph, `base.grad` is **absent** rather than 0. torch
      gives 0 (measured). Taken out, `backward()` stops with "does not require
      grad".

    The two requirements pull in opposite directions, so it is easy to satisfy
    one, and with ordinary input **neither** is visible — a NaN surfaces the first
    and asking for the gradient surfaces the second.
    """
    scaled = product if alpha == 1 else product * alpha
    if beta != 0:
        return (base if beta == 1 else base * beta) + scaled
    return scaled._make(scaled.data, (scaled, base),
                        lambda g: (g, _np.zeros_like(base.data)),
                        "AddmmBackward0")


def addmm(input, mat1, mat2, beta=1, alpha=1):
    """`β·input + α·(mat1 @ mat2)`. `input` broadcasts to the result's shape
    (measured)."""
    return _blend(_wrap(input), matmul(_wrap(mat1), _wrap(mat2)), beta, alpha)


def addbmm(input, batch1, batch2, beta=1, alpha=1):
    """**It folds the batch** — it multiplies and then sums the batch axis into a
    2-D result.

    One character apart from `baddbmm` and a different result rank. At a batch of 1
    the two look the same, so the cases keep the batch above one.
    """
    product = matmul(_wrap(batch1), _wrap(batch2)).sum(0)
    return _blend(_wrap(input), product, beta, alpha)


def baddbmm(input, batch1, batch2, beta=1, alpha=1):
    """**It keeps the batch.** Where it parts from `addbmm`."""
    return _blend(_wrap(input), matmul(_wrap(batch1), _wrap(batch2)),
                  beta, alpha)


def addmv(input, mat, vec, beta=1, alpha=1):
    """`β·input + α·(mat @ vec)`. The result is 1-D."""
    return _blend(_wrap(input), mv(_wrap(mat), _wrap(vec)), beta, alpha)


def addr(input, vec1, vec2, beta=1, alpha=1):
    """`β·input + α·(vec1 ⊗ vec2)`. An outer product, so the result is 2-D."""
    return _blend(_wrap(input), outer(_wrap(vec1), _wrap(vec2)), beta, alpha)


def addcmul(input, tensor1, tensor2, value=1):
    """`input + value·(t1 · t2)`. **There is no `beta`** — `input`'s coefficient
    is always 1."""
    return _blend(_wrap(input), _wrap(tensor1) * _wrap(tensor2), 1, value)


def addcdiv(input, tensor1, tensor2, value=1):
    """`input + value·(t1 / t2)`. The form an optimiser writes its update in."""
    return _blend(_wrap(input), _wrap(tensor1) / _wrap(tensor2), 1, value)


def sspaddmm(input, mat1, mat2, beta=1, alpha=1):
    """**Sparse-only, so it is absent.**

    torch's version takes a sparse COO tensor and produces a sparse one (measured:
    without going through `to_sparse()` it refuses). There is no sparse layout
    here, and imitating it with a dense tensor hands back something **whose shape
    matches and whose storage differs** — and whoever learns from that learns the
    wrong thing about what sparse is.
    """
    _unsupported("torch.sspaddmm — there is no sparse tensor layout here")


def addmm_(input, mat1, mat2, beta=1, alpha=1):
    input = _wrap(input)
    return input._inplace(lambda: addmm(input, mat1, mat2, beta, alpha),
                          "addmm_")


def addbmm_(input, batch1, batch2, beta=1, alpha=1):
    input = _wrap(input)
    return input._inplace(lambda: addbmm(input, batch1, batch2, beta, alpha),
                          "addbmm_")


def baddbmm_(input, batch1, batch2, beta=1, alpha=1):
    input = _wrap(input)
    return input._inplace(lambda: baddbmm(input, batch1, batch2, beta, alpha),
                          "baddbmm_")


def addmv_(input, mat, vec, beta=1, alpha=1):
    input = _wrap(input)
    return input._inplace(lambda: addmv(input, mat, vec, beta, alpha), "addmv_")


def addr_(input, vec1, vec2, beta=1, alpha=1):
    input = _wrap(input)
    return input._inplace(lambda: addr(input, vec1, vec2, beta, alpha), "addr_")


def addcmul_(input, tensor1, tensor2, value=1):
    input = _wrap(input)
    return input._inplace(lambda: addcmul(input, tensor1, tensor2, value),
                          "addcmul_")


def addcdiv_(input, tensor1, tensor2, value=1):
    input = _wrap(input)
    return input._inplace(lambda: addcdiv(input, tensor1, tensor2, value),
                          "addcdiv_")


# ── top-level linear algebra ────────────────────────────────────────────────
#
# **The argument order differs from `linalg`'s.** torch left the old names at the
# top level, and they mostly take **the right-hand side first** —
# `lu_solve(b, LU, piv)` versus `linalg.lu_solve(LU, piv, b)`. The same computation
# called a different way, so the computation is kept as one copy and only the
# positions move. Whether that move is right is confirmed by values alone.

def _mT(t):
    """Swap the last two axes.

    **In this file `transpose` is a method rather than a module function.** And
    `triangular_solve`'s third parameter is named `transpose`, so inside it the
    name is shadowed by the parameter as well. One short name avoids both
    places.
    """
    return _wrap(t).transpose(-2, -1)


def _as_lower(factor, upper):
    """Stand the factor up **as a lower triangle**, so that `A = L Lᵀ`.

    It is kept as an assembly — passing through `tril` and `transpose` flows
    **gradient towards the factor as well.** Trimmed directly with numpy the
    values are right and the backward reaches `b` alone, and torch flows into the
    factor too (measured). That divergence is invisible unless the factor is made
    a differentiation target.
    """
    return _mT(triu(factor)) if upper else tril(factor)


def cholesky_solve(b, input2, upper=False):
    """Solve `A x = b` **through a Cholesky input2.** `A = L Lᵀ` (or `Uᵀ U`).

    `A` is rebuilt and handed to `solve`. Two triangular substitutions would be
    cheaper, and written that way the backward has to be written by hand and **the
    gradient towards the input2 quietly goes missing** — a bigger risk than what
    the saving is worth at this size.
    """
    low = _as_lower(_wrap(input2), upper)
    return solve(matmul(low, _mT(low)), _wrap(b))


def cholesky_inverse(factor, upper=False):
    """From a Cholesky factor, produce **the original matrix's inverse.** Not the
    factor's inverse."""
    _rank(_wrap(factor).data, range(2, 65),
          "cholesky_inverse: The input tensor A must have at least 2 dimensions.")
    low = _as_lower(_wrap(factor), upper)
    return inverse(matmul(low, _mT(low)))


def triangular_solve(b, A, upper=True, transpose=False, unitriangular=False):
    """**It gives two things** — the solution and **a copy** of the coefficient
    matrix handed in (measured).

    The same computation as `linalg.solve_triangular` with the argument order
    reversed and **`upper` defaulting to true.** Missing those two, it solves the
    other triangle and the value still comes out plausible.

    The third parameter is named `transpose`, so inside this function the module's
    `transpose` is shadowed. The `_mT` alias fills that place.
    """
    b, A = _wrap(b), _wrap(A)
    tri = triu(A) if upper else tril(A)
    if unitriangular:
        # **The diagonal is ignored and treated as 1.** Left as it is, a quietly
        # different answer comes out.
        n = tri.data.shape[-1]
        off = triu(tri, 1) if upper else tril(tri, -1)
        tri = off + Tensor(_np.eye(n, dtype=tri.data.dtype))
    if transpose:
        tri = _mT(tri)
    return _TriangularSolve(solve(tri, b), Tensor(_np.array(A.data, copy=True)))


def lu_solve_top(b, LU_data, LU_pivots):                     # noqa: N803
    """**The argument order is reversed from `linalg.lu_solve`** — `b` comes
    first here, and the two that follow carry torch's capitals.

    `LU_data`/`LU_pivots` on the method, `LU`/`pivots` on the free function one
    screen up. Both are torch's: it spells the same two things differently
    depending on which door you come through."""
    return lu_solve(LU_data, LU_pivots, b)


def lu_top(A, pivot=True, get_infos=False):                  # noqa: N803
    """`(LU, pivots)`. **A different thing from `linalg.lu`** — that one spreads
    it into `P`, `L` and `U` and this gives **one packed matrix plus the swap
    list** (measured).

    With `get_infos=True` an info code is attached as a third field. It is always
    0 here — meeting a singular matrix it throws at that point rather than
    reporting quietly through a code.

    `A` is torch's name at the top level too. `torch.lu` is one of the few that took
    the `linalg` spelling with it — most did not, and `det`, `qr` and `slogdet` below
    take `input` at the top and `A` under `linalg`.
    """
    if not pivot:
        _unsupported("lu(pivot=False)")
    data, piv = lu_factor(A)
    if get_infos:
        return _LuInfos(data, piv, Tensor(_np.zeros((), dtype=_np.int32)))
    return _LuFactor(data, piv)


def lu_unpack(lu_data, lu_pivots, unpack_data=True, unpack_pivots=True):
    """Spread one packed matrix into `P`, `L` and `U`.

    **Turned off it gives an empty tensor rather than `None`** (measured: the shape
    is `(0,)`). Left as `None` the receiving side branches on `if p is None`, and
    that is not torch code.
    """
    lu_data, lu_pivots = _wrap(lu_data), _wrap(lu_pivots)
    empty = Tensor(_np.zeros(0, dtype=lu_data.data.dtype))
    if not unpack_data and not unpack_pivots:
        return _LuUnpack(empty, empty, empty)
    n, m = lu_data.data.shape[-2], lu_data.data.shape[-1]
    k = min(n, m)
    data = _np.asarray(lu_data.data, dtype=_np.float64)
    low = _np.tril(data[:, :k], -1).copy()
    low[_np.arange(k), _np.arange(k)] = 1.0
    up = _np.triu(data)[:k, :]
    order = _np.arange(n)
    flat = _np.asarray(lu_pivots.data).reshape(-1)
    for col in range(min(k, flat.shape[0])):
        src = int(flat[col]) - 1
        if src != col:
            order[[col, src]] = order[[src, col]]
    perm = _np.zeros((n, n))
    perm[order, _np.arange(n)] = 1.0
    kind = lu_data.data.dtype
    return _LuUnpack(
        Tensor(perm.astype(kind)) if unpack_pivots else empty,
        Tensor(low.astype(kind)) if unpack_data else empty,
        Tensor(up.astype(kind)) if unpack_data else empty)


def orgqr(input, input2):                                    # noqa: A002
    """Multiply the reflectors `geqrf` packed away and **build Q.** The same as
    `linalg.householder_product`; torch gives it two names."""
    return householder_product(input, input2)


def ormqr(input, input2, input3, left=True, transpose=False):  # noqa: A002
    """Multiply into `C` **without building Q.** That is the point on a large
    matrix, and here it is built and multiplied — the value is the same and there
    is nothing to save at this size.

    **A different Q from `orgqr`'s.** That one gives the **trimmed** `m×k` Q (the
    same Q as `linalg.qr`'s) and this uses the untrimmed `m×m` — the reflectors are
    a map on `Rᵐ`, and trimming multiplies by only part of that map. Caught by
    measurement: on a tall matrix the answer was entirely different. Measured on
    squares alone the two coincide and it is invisible.

    `left` says which side to multiply from and `transpose` whether to use `Qᵀ`.
    """
    q = _full_q(input, input2)
    if transpose:
        q = q.T
    c = _np.asarray(_wrap(input3).data, dtype=_np.float64)
    out = (q @ c) if left else (c @ q)
    return Tensor(out.astype(_wrap(input3).data.dtype))


def _full_q(a, tau):
    """Multiply the reflectors together and build the **untrimmed `m×m`** Q.

    The same loop as `householder_product` without the final column trim. That one
    line is the whole difference between `orgqr` and `ormqr`.
    """
    mat = _np.asarray(_wrap(a).data, dtype=_np.float64)
    taus = _np.asarray(_wrap(tau).data, dtype=_np.float64).reshape(-1)
    m = mat.shape[-2]
    q = _np.eye(m)
    for j in range(taus.shape[0] - 1, -1, -1):
        v = _np.zeros(m)
        v[j] = 1.0
        v[j + 1:] = mat[j + 1:, j]
        q = q - taus[j] * _np.outer(v, v @ q)
    return q


def lobpcg(a, k=None, B=None, X=None, n=None, iK=None, niter=None, tol=None,
           largest=True, method=None, tracker=None, ortho_iparams=None,
           ortho_fparams=None, ortho_bparams=None):
    """The `k` **extreme eigenpairs** of a symmetric matrix.

    **torch is iterative and this is exact.** That side iterates so that a few can
    be obtained cheaply from a large sparse matrix, and here there is no sparse
    layout and the sizes are small. Measured, torch's answer converges to within
    **7e-6** of the exact one and shakes by no more than that with the seed
    (measured) — far below this repository's tolerance. So the values match and
    only the cost differs.

    **`largest` sets the order too** — true gives largest first and false smallest
    first (measured).

    **`B` is the generalised problem `A x = λ B x`, and it was refused.** With `B`
    symmetric positive definite it reduces to a standard one in four lines: `B = L Lᵀ`,
    then the eigenvalues of `L⁻¹ A L⁻ᵀ` are the generalised ones and `x = L⁻ᵀ y` are
    the generalised vectors. Nothing iterative and no new dependency.

    **Those vectors come out `B`-orthonormal**, not unit length — `xᵀBx = 1` and
    `xᵀx = 0.996` on the fixture. That falls out of the reduction (`xᵀBx = yᵀy`) and
    it is also what torch returns, measured. Normalising them would look tidier and
    disagree.

    **`X` is a starting basis and this has nothing to start.** What it does change is
    the count: given `X` and no `k`, torch takes `k` from `X`'s columns, and that is
    read here. The converged eigenvalues do not depend on it — measured, torch with
    and without `X` agrees to 5e-6, which is the same distance its own answer moves
    with the seed.
    """
    # **`k` defaulted to 1 here and to `None` in torch**, which was invisible while
    # nothing read `X`: with neither given both mean one pair. Given `X` and no `k`,
    # torch takes the count from `X`'s columns, and a default of 1 makes that
    # unreachable — the argument would be received and the answer would be one pair
    # wide whatever was handed in.
    if k is None:
        k = 1 if X is None else int(_np.asarray(_wrap(X).data).shape[-1])
    if B is None:
        vals, vecs = eigh(_wrap(a))
        vals, vecs = _np.asarray(vals.data), _np.asarray(vecs.data)
    else:
        mat = _np.asarray(_wrap(a).data, dtype=_np.float64)
        low = _np.linalg.cholesky(_np.asarray(_wrap(B).data, dtype=_np.float64))
        inner = _np.linalg.solve(low, _np.linalg.solve(low, mat).T).T
        vals, y = _np.linalg.eigh((inner + inner.T) / 2.0)
        vecs = _np.linalg.solve(low.T, y)
        vals = vals.astype(mat.dtype)
        vecs = vecs.astype(mat.dtype)
    order = slice(None, None, -1) if largest else slice(None)
    idx = _np.arange(vals.shape[-1])[order][:k]
    return _Lobpcg(Tensor(vals[idx]), Tensor(vecs[:, idx]))


def svd_lowrank(a, q=6, niter=2, M=None):
    """A **low-rank SVD** obtained by random projection. `(U, S, V)`, and **V is
    not transposed.**

    **The answer stops shaking only on exactly low-rank input.** torch projects
    with a random matrix, and once the rank exceeds `q` the singular values move
    by around 0.5 with the seed (measured). At rank `q` or below, changing the seed
    stays within 7e-7 — that is the only place the golden can ask about.

    This does not project. The full SVD is computed and the first `q` taken — the
    same answer where the input is exactly low rank, and **a more accurate answer
    than torch's** where it is not.
    """
    a = _wrap(a)
    data = _np.asarray(a.data, dtype=_np.float64)
    if M is not None:
        data = data - _np.asarray(_wrap(M).data, dtype=_np.float64)
    u, s, vh = _np.linalg.svd(data, full_matrices=False)
    kind = a.data.dtype
    return _SvdLowrank(Tensor(u[:, :q].astype(kind)), Tensor(s[:q].astype(kind)),
                       Tensor(vh[:q].T.astype(kind)))


def pca_lowrank(a, q=None, center=True, niter=2):
    """A low-rank PCA. **With `center=False` it is the same as `svd_lowrank`**
    (measured).

    The centring is the whole difference between this function and that one.
    Measured with true alone that branch is invisible.
    """
    a = _wrap(a)
    data = _np.asarray(a.data, dtype=_np.float64)
    if q is None:
        q = min(6, *data.shape)
    if center:
        data = data - data.mean(axis=0, keepdims=True)
    return svd_lowrank(Tensor(data.astype(a.data.dtype)), q, niter)


# ── statistics ──────────────────────────────────────────────────────────────
#
# **The four random ones have a corner that can be pinned.**
#
# The golden cannot pin the values of `normal`, `bernoulli`, `poisson` and
# `binomial` — torch's random stream and ours differ, and there is no way to make
# them agree. And **the extremes are deterministic**: `std=0` gives the mean
# itself, `p=0` gives all zeros, `p=1` gives all ones, and `poisson(0)` gives 0
# (measured). The golden asks about those places and looks at the shape and the
# dtype for the rest.
#
# That is the difference between "random, so it cannot be asked" and "it is not
# asked".

def _edges(data, bins, low, high):
    """Build the edges. **The last bin is closed on the right** (measured)."""
    if low == high:
        low, high = float(_np.min(data)), float(_np.max(data))
        if low == high:
            low, high = low - 0.5, high + 0.5
    return _np.linspace(low, high, int(bins) + 1)


def _count_into(data, edges, weights=None):
    """Count into the bins `edges` divides. **What falls outside is discarded** —
    torch does that (measured)."""
    flat = _np.asarray(data, dtype=_np.float64).reshape(-1)
    w = (_np.ones_like(flat) if weights is None
         else _np.asarray(weights, dtype=_np.float64).reshape(-1))
    out = _np.zeros(len(edges) - 1, dtype=_np.float64)
    for value, weight in zip(flat, w):
        if value < edges[0] or value > edges[-1]:
            continue
        # The right edge goes into the last bin.
        slot = int(_np.searchsorted(edges, value, side="right")) - 1
        out[min(max(slot, 0), len(out) - 1)] += weight
    return out


def histc(input, bins=100, min=0, max=0):
    """How many fall in each bin. **With `min == max` the data's own range is
    used** (measured).

    Given a range, **what falls outside is discarded** — it is not piled into the
    end bins. Measured on data that is entirely inside the range that rule never
    surfaces.
    """
    input = _wrap(input)
    edges = _edges(input.data, bins, float(min), float(max))
    return Tensor(_count_into(input.data, edges).astype(input.data.dtype))


def histogram(input, bins=100, range=None, weight=None, density=False):
    """`histc`'s arithmetic **with the edges given as well.**

    A tensor for `bins` is the edges themselves — the bin widths may differ, and
    then `density` divides by a different value per bin.
    """
    input = _wrap(input)
    if isinstance(bins, (Tensor, list, tuple, _np.ndarray)):
        edges = _np.asarray(_wrap(bins).data if isinstance(bins, Tensor) else bins,
                            dtype=_np.float64)
    else:
        low, high = (0.0, 0.0) if range is None else (float(range[0]), float(range[1]))
        edges = _edges(input.data, bins, low, high)
    counts = _count_into(input.data, edges,
                         None if weight is None else _wrap(weight).data)
    if density:
        widths = _np.diff(edges)
        total = counts.sum()
        counts = counts / (widths * (total if total else 1.0))
    kind = input.data.dtype
    return _Histogram(Tensor(counts.astype(kind)), Tensor(edges.astype(kind)))


def histogramdd(input, bins=10, range=None, weight=None, density=False):
    """A histogram over several axes. `input` is `(sample count, dimensions)`."""
    input = _wrap(input)
    data = _np.asarray(input.data, dtype=_np.float64)
    dims = data.shape[-1]
    counts = [bins] * dims if isinstance(bins, int) else list(bins)
    edges = []
    for d in _builtin_range(dims):
        low, high = (0.0, 0.0)
        if range is not None:
            low, high = float(range[2 * d]), float(range[2 * d + 1])
        edges.append(_edges(data[:, d], counts[d], low, high))
    hist, _ = _np.histogramdd(data, bins=edges, density=density,
                              weights=None if weight is None
                              else _np.asarray(_wrap(weight).data).reshape(-1))
    kind = input.data.dtype
    return _HistogramDD(Tensor(hist.astype(kind)),
                        [Tensor(e.astype(kind)) for e in edges])


def mode(input, dim=-1, keepdim=False):
    """The most frequent value. **On an equal count the smaller value wins, and
    the index is that value's last occurrence** (measured: `[4,4,5,5]` gives value
    4 and index 1).

    Measured on data with no ties that rule never surfaces.

    **The axis is checked.** `mode(x, dim=7)` on a 2-D tensor answered as though the
    axis were the last one — the fourth silent wrong answer the `dim` sweep has
    found, and it only became visible when `_accepts_out` started preserving the
    signature it wraps: until then this name read as `(*args, **kwargs)` and the
    sweep skipped it for want of a `dim` to ask about.
    """
    input = _wrap(input)
    if input.data.ndim == 0:
        return _at_rank_0(input, lambda x: mode(x, 0, keepdim))
    _pos_dim(input, dim)
    data = _np.asarray(input.data)
    axis = dim % data.ndim
    moved = _np.moveaxis(data, axis, -1)
    flat = moved.reshape(-1, moved.shape[-1])
    vals = _np.empty(flat.shape[0], dtype=data.dtype)
    idx = _np.empty(flat.shape[0], dtype=_np.int64)
    for row in _builtin_range(flat.shape[0]):
        line = flat[row]
        best, best_count = None, -1
        for value in _np.unique(line):
            count = int((line == value).sum())
            if count > best_count:
                best, best_count = value, count
        vals[row] = best
        idx[row] = int(_np.flatnonzero(line == best)[-1])
    # The gradient goes **to the one position named.** Being the most frequent
    # value there are several cells holding it, and `mode` hands back the last of
    # them as the index, so that position represents the answer (measured: the
    # gradient of [1,1,2,2,2] goes to the last 2 alone). Here too there was only a
    # bare `Tensor(...)` and the graph was cut.
    weight = _np.zeros(flat.shape, dtype=_np.float64)
    weight[_np.arange(flat.shape[0]), idx] = 1.0
    weight = _np.moveaxis(weight.reshape(moved.shape), -1, axis)
    shape = moved.shape[:-1]
    vals = vals.reshape(shape)
    idx = idx.reshape(shape)
    if keepdim:
        vals = _np.expand_dims(vals, axis)
        idx = _np.expand_dims(idx, axis)

    def back(g):
        gg = _np.asarray(g)
        if keepdim:
            gg = _np.squeeze(gg, axis)
        return (_np.expand_dims(gg, axis) * weight,)

    return _Mode(input._make(vals, (input,), back, "ModeBackward0"), Tensor(idx))


def nanmedian(t, dim=None, keepdim=False):
    """The median computed **excluding** NaN. `median` gives NaN when even one is
    present (measured).

    With an even count it **takes the lower one** — it does not average.
    """
    t = _wrap(t)
    _refuses_bool(t.data, "nanmedian does not take booleans.",
                  '"median_cpu" not implemented for \'Bool\'',
                  kind=NotImplementedError)
    data = _np.asarray(t.data, dtype=_np.float64)
    if dim is not None:
        _pos_dim(t, dim)              # the sort below takes any axis without minding
    if dim is None:
        flat = data.reshape(-1)
        clean = flat[~_np.isnan(flat)]
        pick = _np.sort(clean)[(clean.shape[0] - 1) // 2]
        # **Split evenly across every cell holding the same value** —
        # `median()`'s rule. Here there was only a bare `Tensor(...)` and the
        # graph was quietly cut.
        share = (flat == pick).astype(_np.float64)
        share = (share / share.sum()).reshape(t.data.shape)
        return t._make(_np.asarray(pick, dtype=t.data.dtype), (t,),
                       lambda g: (_np.asarray(g) * share,), "NanmedianBackward0")
    axis = dim % data.ndim
    moved = _np.moveaxis(data, axis, -1)
    flat = moved.reshape(-1, moved.shape[-1])
    vals = _np.empty(flat.shape[0], dtype=_np.float64)
    idx = _np.empty(flat.shape[0], dtype=_np.int64)
    for row in _builtin_range(flat.shape[0]):
        line = flat[row]
        keep = _np.flatnonzero(~_np.isnan(line))
        order = keep[_np.argsort(line[keep], kind="stable")]
        at = order[(order.shape[0] - 1) // 2]
        vals[row] = line[at]
        idx[row] = int(at)
    # **Given an axis, indices come out, and once indices come out the gradient
    # goes to that one position.** The opposite of splitting evenly with no axis,
    # and the split is the same one — an operation that hands back indices names
    # the position it chose, and one that does not draws no distinction between
    # cells holding the same value.
    weight = _np.zeros(flat.shape, dtype=_np.float64)
    weight[_np.arange(flat.shape[0]), idx] = 1.0
    weight = _np.moveaxis(weight.reshape(moved.shape), -1, axis)
    shape = moved.shape[:-1]
    vals = vals.reshape(shape).astype(t.data.dtype)
    idx = idx.reshape(shape)
    if keepdim:
        vals = _np.expand_dims(vals, axis)
        idx = _np.expand_dims(idx, axis)

    def back(g):
        gg = _np.asarray(g)
        if keepdim:
            gg = _np.squeeze(gg, axis)
        return (_np.expand_dims(gg, axis) * weight,)

    return _NanMedian(t._make(vals, (t,), back, "NanmedianBackward0"), Tensor(idx))


def gradient(input, spacing=1, dim=None, edge_order=1):
    """Central differences. **One per axis, returned as a tuple** — with no axis
    given, all of them.

    An `edge_order` of 1 fits the ends with a one-sided difference and 2 with a
    quadratic (measured: on `x²`, 2 gives the exact derivative and 1 is off at
    both ends).
    """
    input = _wrap(input)
    data = _np.asarray(input.data, dtype=_np.float64)
    axes = (tuple(_builtin_range(data.ndim)) if dim is None
            else (dim,) if isinstance(dim, int) else tuple(dim))
    # `axis % data.ndim` below wraps rather than complains, so 7 became 1 on a 2-D
    # tensor and the answer was the last axis's gradient under another name.
    for axis in axes:
        _pos_dim(input, axis)
    step = spacing if isinstance(spacing, (list, tuple)) else [spacing] * len(axes)
    outs = []
    for axis, gap in zip(axes, step):
        if isinstance(gap, Tensor):
            gap = _np.asarray(gap.data, dtype=_np.float64)
        got = _np.gradient(data, gap, axis=axis % data.ndim,
                           edge_order=int(edge_order))
        outs.append(Tensor(got.astype(input.data.dtype)))
    return tuple(outs)


def trapz(y, x=None, dx=1.0, dim=-1):
    """The old name for `trapezoid`. The same thing (measured)."""
    return trapezoid(y, x, dx, dim)


def nonzero_static(input, size, fill_value=-1):
    """Give **a fixed number of** non-zero positions. Short, it pads; over, it
    trims.

    `nonzero`'s result size depends on the values and needs one read back from the
    GPU, and this is given the size in advance so there is no such round trip —
    the name exists for that place.
    """
    input = _wrap(input)
    found = _np.argwhere(_np.asarray(input.data) != 0)
    rank = max(1, _np.asarray(input.data).ndim)
    out = _np.full((int(size), rank), int(fill_value), dtype=_np.int64)
    take = min(int(size), found.shape[0])
    out[:take] = found[:take]
    return Tensor(out)


def normal(mean=0.0, std=1.0, size=None, dtype=None, requires_grad=False, *, out=None):
    """A normal sample. **With `std` at 0 it is the mean itself** — the golden asks
    about that place.

    Given `mean` and `std` as tensors it is a different distribution per position.
    In that case it does not take `size`.

    `dtype=` and `requires_grad=` were **being swallowed** by `**kw`. It was
    outside the list both when fourteen factories were gathered under `_made` and
    when the other nine were — walking the functions carrying `**kw` produced four
    candidates, and against torch's signatures the real ones were this and
    `frombuffer` (torch itself does not take the argument on `bernoulli`, and
    `empty_strided` already refuses).
    """
    _no_out(out)
    if isinstance(mean, Tensor) or isinstance(std, Tensor):
        m = _np.asarray(_wrap(mean).data, dtype=_np.float64)
        s = _np.asarray(_wrap(std).data, dtype=_np.float64)
        m, s = _np.broadcast_arrays(m, s)
        return _made(_np.asarray(_rng.normal(m, s)).astype(_DEFAULT_DTYPE),
                     dtype, requires_grad)
    shape = () if size is None else tuple(size)
    return _made(_rng.normal(float(mean), float(std), shape).astype(_DEFAULT_DTYPE),
                 dtype, requires_grad)


def bernoulli(t, p=None, *, generator=None, out=None):
    """A 1 at each position with that probability. **0 gives all zeros and 1 all
    ones** — those two extremes are deterministic.

    **`p` as one number is torch's other form** and was not taken: `x.bernoulli(0.5)`
    draws at that probability everywhere and ignores the tensor's values, where
    `x.bernoulli()` uses them. It stopped with a `TypeError` about the argument
    count — the tensor holding the probabilities is the form a tutorial uses second.
    """
    _no_out(out)
    t = _wrap(t)
    rng = generator.rng() if generator is not None else _rng
    probs = (_np.full(t.data.shape, float(p)) if p is not None
             else _np.asarray(t.data, dtype=_np.float64))
    return Tensor((rng.random(probs.shape) < probs).astype(t.data.dtype))


def poisson(t):
    """A Poisson sample at that rate per position. **0 gives all zeros**
    (measured)."""
    t = _wrap(t)
    lam = _np.asarray(t.data, dtype=_np.float64)
    # `_np.asarray` around the draw: at rank 0 numpy hands back a **python int**,
    # which has no `.astype`. The same leak sat in `normal`, and both are reachable
    # from ordinary code — `normal(tensor(0.), tensor(1.))` is a scalar draw.
    return Tensor(_np.asarray(_rng.poisson(lam)).astype(t.data.dtype))


def binomial(count, prob):
    """The number of successes in `count` trials. **`p=0` gives 0 and `p=1` gives
    `count`.**"""
    n = _np.asarray(_wrap(count).data, dtype=_np.float64)
    p = _np.asarray(_wrap(prob).data, dtype=_np.float64)
    n, p = _np.broadcast_arrays(n, p)
    return Tensor(_rng.binomial(n.astype(_np.int64), p).astype(_DEFAULT_DTYPE))


# **A refusal for a long time.** The refusal said "the complex convention has not
# been settled", and the reason was right — what was missing was not storage but
# **the Wirtinger convention never having been measured.** Measured and pinned
# (`z.grad = ∂L/∂re + i·∂L/∂im`), these two names came out as assemblies.
#
# Writing down **precisely** why something could not be done paid off here.
# Written as "there is no storage", nobody would have asked again on the day the
# storage arrived.
stft = _fft_stft
istft = _fft_istft


def hash_tensor(*args, **kw):
    """**No uint64 and no specification.**

    What torch produces is `uint64` (measured), and which hash it is is not in the
    documentation either. Putting down a name for something whose values cannot be
    matched creates code that trusts those values.
    """
    _unsupported("torch.hash_tensor — there is no uint64 and no settled hash spec")


# ── the neighbourhood of complex numbers, and a few factories ───────────────
#
# **The names that have an answer even without complex numbers.**
#
# `real`, `conj` and `resolve_conj` are **the identity** on a real tensor
# (measured: they share the buffer too), `is_complex`, `is_conj` and `is_neg` are
# all false, and `angle` is π for negatives. Leaving these names out as well
# because the complex convention was unsettled makes textbook code that branches on
# them stop with an `AttributeError` — what can be answered and what cannot are
# different things.
#
# `imag` alone differs. **torch itself refuses on the reals** (measured) — so
# refusing here is **torch carried over exactly** rather than a limit of ours.

def _is_complex(t):
    return _np.asarray(t.data).dtype.kind == "c"


def _alias(t, name):
    """The identity, giving the same value back. **It keeps the dtype and the
    graph.**

    It must not be sent to `positive`'s unary kernel — that side falls to float32,
    so a `bool` going in does not come out `bool` (measured: torch gives `bool`
    back unchanged).
    """
    t = _wrap(t)
    return t._make(t.data, (t,), lambda g: (g,), name)


# ── the complex gradient convention ─────────────────────────────────────────
#
# **Wirtinger collapses because the loss is always real.** torch refuses
# `backward()` on a complex loss (measured: "grad can be implicitly created only
# for real scalar outputs"). That settles the convention as this —
#
#   z.grad = ∂L/∂re + i·∂L/∂im        (differentiated as two reals and bundled)
#
# Measurement pins it (z = 1+2j):
#
#   L = z.real  → 1+0j        L = z.imag  → **0+1j** (not −1j)
#   L = |z|²    → 2+4j        L = (z·z̄).real → 2+4j
#
# Under this convention **a holomorphic f's backward is `conj(f'(z))·g`** — one
# conjugate away from the `f'(x)·g` the real-valued code uses. Leaving that one
# out produces a gradient with the sign flipped, and the values are plausible
# enough that real input never shows it.

def _cgrad(local, g):
    """One term of a holomorphic function's backward. **Where the conjugate
    attaches.**"""
    return _np.conj(local) * g


def real(input):
    """The real part.

    On a real tensor it is **the tensor itself**, dtype included (`bool` stays
    `bool`). On a complex one it takes the real part, and **the gradient flows to
    the real slot alone** — which is what the gradient of `z.real` being `1+0j`
    means (measured).
    """
    input = _wrap(input)
    if not _is_complex(input):
        return _alias(input, "RealBackward0")
    return input._make(_np.real(input.data).copy(), (input,),
                   lambda g: (_np.asarray(g).astype(input.data.dtype),),
                   "RealBackward0")


def imag(input):
    """The imaginary part.

    **torch refuses on a real tensor too** (measured) — not a limit of ours. On a
    complex one it takes the imaginary part, and **the gradient goes back carrying
    an `i`** — the gradient of `z.imag` is `0+1j` (measured). Written as `−1j` it
    runs plausibly with the sign flipped.
    """
    input = _wrap(input)
    if not _is_complex(input):
        raise RuntimeError(_like_torch(
            "A real tensor has no imaginary part.",
            "imag is not implemented for tensors with non-complex dtypes."))
    return input._make(_np.imag(input.data).copy(), (input,),
                   lambda g: (_np.asarray(g).astype(input.data.dtype) * 1j,),
                   "ImagBackward0")


def conj(input):
    """The conjugate. Over the reals it is the identity and **a view** — torch
    shares the buffer too (measured).

    **It is not holomorphic.** So its backward is **`conj(g)`** rather than the
    `conj(f')·g` form — being the conjugate of a conjugate.
    """
    input = _wrap(input)
    if not _is_complex(input):
        return _alias(input, "ConjBackward0")
    return input._make(_np.conj(input.data), (input,), lambda g: (_np.conj(_np.asarray(g)),),
                   "ConjBackward0")


def conj_physical(input):
    """The same value as `conj`. torch splits the names to say this is **the
    version that actually copies.**

    **Before complex numbers arrived this function was the identity** — while
    everything was real that was the right value and the golden passed. The moment
    complex was attached the same code became a wrong answer. "An identity that
    passes today" is the first thing to break when the domain widens.
    """
    input = _wrap(input)
    if not _is_complex(input):
        return _alias(input, "ConjPhysicalBackward0")
    return input._make(_np.conj(input.data), (input,), lambda g: (_np.conj(_np.asarray(g)),),
                   "ConjPhysicalBackward0")


def conj_physical_(t):
    t = _wrap(t)
    return t._inplace(lambda: conj_physical(t), "conj_physical_")


def resolve_conj(input):
    """Materialise the conjugate flag into actual values. The reals carry no such
    flag, so it is the identity."""
    return _alias(input, "ResolveConjBackward0")


def resolve_neg(input):
    """Materialise the negation flag. `resolve_conj`'s place."""
    return _alias(input, "ResolveNegBackward0")


def angle(input):
    """The angle. Over the reals it is **π for negatives and 0 for the rest** — the
    complex case specialised.

    **The dtype is always float32** — an integer going in does not come out an
    integer (measured). An angle does not fit in an integer cell, so that is right,
    and feeding it floats alone never surfaces the rule.
    """
    input = _wrap(input)
    data = _np.asarray(input.data)
    if data.dtype.kind == "c":
        return Tensor(_np.angle(data).astype(_DEFAULT_DTYPE))
    out = _np.where(data < 0, _math.pi, 0.0).astype(_DEFAULT_DTYPE)
    # **It flows a 0 — the right answer rather than "absent".** The angle of a
    # real is a step and its derivative is 0 everywhere, and torch fills 0 too
    # (measured). Without carrying the graph through, `backward()` stops, and what
    # comes out then points at the user rather than at this operation.
    return input._make(out, (input,), lambda g: (_np.zeros_like(data, dtype=_np.float64),),
                   "AngleBackward0")


def _complex_abs(t):
    """The magnitude of a complex number. **No conjugate attaches to the
    backward** — it produces a real, so it is not holomorphic.

    Bundling `∂|z|/∂re = re/|z|` and `∂|z|/∂im = im/|z|` gives `z/|z|`. Measurement
    backs it: on `L = |z|²` the gradient is `2z` (2+4j at z=1+2j).
    """
    data = _np.asarray(t.data)
    mag = _np.abs(data)
    out = mag.astype(_DEFAULT_DTYPE)
    safe = _np.where(mag == 0, 1.0, mag)

    def back(g):
        return ((_np.asarray(g) * data / safe).astype(data.dtype),)

    return t._make(out, (t,), back, "AbsBackward0")


def complex(real, imag):
    """Bundle a real part and an imaginary part. **This name shadows the Python
    builtin** — which is why `_is_complex` is used for the complex test inside this
    file."""
    real, imag = _wrap(real), _wrap(imag)
    out = (_np.asarray(real.data, dtype=_np.float32)
           + 1j * _np.asarray(imag.data, dtype=_np.float32)).astype(_np.complex64)
    # **The gradient flows into the real leaves.** The real part takes the real
    # share and the imaginary part the imaginary one — the reverse direction of the
    # `∂L/∂re + i·∂L/∂im` convention.
    return real._make(out, (real, imag),
                    lambda g: (_np.real(_np.asarray(g)).astype(_np.float32),
                               _np.imag(_np.asarray(g)).astype(_np.float32)),
                    "ComplexBackward0")


def polar(abs_, angle_):
    """Build from a magnitude and an angle. `abs·(cos θ + i sin θ)`."""
    abs_, angle_ = _wrap(abs_), _wrap(angle_)
    mag = _np.asarray(abs_.data, dtype=_np.float64)
    ang = _np.asarray(angle_.data, dtype=_np.float64)
    return Tensor((mag * _np.exp(1j * ang)).astype(_np.complex64))


def view_as_real(input):
    """View a complex tensor as `(…, 2)` reals. The last axis is `(re, im)`
    (measured).

    **It does not work on a real tensor** — torch refuses too.
    """
    input = _wrap(input)
    if not _is_complex(input):
        raise RuntimeError(_like_torch(
            "Not available on a real tensor — complex only.",
            "view_as_real is only supported for complex tensors"))
    out = _np.stack([_np.real(input.data), _np.imag(input.data)], axis=-1)
    return input._make(out.astype(_np.float32), (input,),
                   lambda g: ((_np.asarray(g)[..., 0]
                               + 1j * _np.asarray(g)[..., 1]).astype(input.data.dtype),),
                   "ViewAsRealBackward0")


def view_as_complex(input):
    """View `(…, 2)` reals as complex. The inverse of `view_as_real`."""
    input = _wrap(input)
    data = _np.asarray(input.data)
    if data.shape[-1] != 2:
        raise RuntimeError(_like_torch(
            "The last dimension must be 2.",
            "Tensor must have a last dimension of size 2"))
    out = (data[..., 0] + 1j * data[..., 1]).astype(_np.complex64)
    return input._make(out, (input,),
                   lambda g: (_np.stack([_np.real(_np.asarray(g)),
                                         _np.imag(_np.asarray(g))],
                                        axis=-1).astype(data.dtype),),
                   "ViewAsComplexBackward0")


def is_complex(input):
    """Is it complex. **Before complex numbers went in it was always false** — it
    now actually looks."""
    return _is_complex(_wrap(input))


def is_conj(input):
    """Does it carry a conjugate flag. There is no way to create that flag, so it
    is always false."""
    return False


def is_neg(t):
    """Does it carry a negation flag. `is_conj`'s place."""
    return False


def asarray(obj, dtype=None, copy=None):
    """**Given a tensor it is not a copy** (measured). `copy=True` makes it one.

    Almost the same as `as_tensor`, differing in taking `copy` explicitly — without
    that argument the caller cannot override the rule that not copying is the
    default.
    """
    if isinstance(obj, Tensor) and dtype is None and not copy:
        return obj
    if isinstance(obj, Tensor):
        data = (obj.data.astype(_requested_dtype(dtype).np)
                if dtype is not None else obj.data)
        return Tensor(_np.array(data, copy=True) if copy else data)
    got = tensor(obj, dtype)
    return Tensor(_np.array(got.data, copy=True)) if copy else got


def frombuffer(buffer, dtype=_float32, count=-1, offset=0,
               requires_grad=False):
    """Read the bytes as they are. **`offset` is a byte count** — not an element
    count (measured).

    **`dtype=` means something different here.** In the other factories it converts
    after building, and here it decides **what the bytes are read as** — converting
    afterwards is after they have already been read as something else. So the dtype
    is not left to `_made` and is read here, with only the gradient left to it.

    `requires_grad=` was being swallowed by `**kw`.
    """
    kind = _np_of(_requested_dtype(dtype))
    return _made(_np.frombuffer(buffer, dtype=kind, count=count,
                                offset=offset).copy(), None, requires_grad)


def range_top(start, end=None, step=1, dtype=None, requires_grad=False, *, out=None):
    """**The end is included** — `arange` excludes it (measured: `range(0, 4)`
    gives five elements).

    torch has it marked for removal and it survives in older textbooks, and being
    one element apart from `arange` is precisely the reason for the removal.
    Forwarded quietly to `arange`, one element goes missing.

    **Why the name is not `range`**: this file uses the Python builtin `range` in
    91 places. Putting that name on the module would send all of them to this
    function — the same trick as `lu` and `lu_solve`, so here it is `range_top` and
    `borch/__init__.py` exports it as `range`. The ninth "a module name shadows a
    builtin" in this file.
    """
    _no_out(out)
    _needs_step(step, "range")
    if end is None:
        start, end = 0, start
    return _made(_np.arange(start, end + step / 2.0, step, dtype=_DEFAULT_DTYPE),
                 dtype, requires_grad)


def empty_strided(size, stride, *, out=None):
    """**Absent because strides cannot be expressed.**

    A different place from `as_strided`. There **the values** are the answer, so a
    copy gives the same answer; here the values are garbage and **the strides
    themselves are the only answer.** Our tensors have no such thing as strides, so
    handing back something whose shape merely matches creates code that believes
    the strides are what it asked for.
    """
    _no_out(out)
    _unsupported("torch.empty_strided — there is no such thing as a stride here")


def empty_permuted(size, physical_layout, *, out=None):
    """Absent for `empty_strided`'s reason."""
    _no_out(out)
    _unsupported("torch.empty_permuted — there is no such thing as a stride here")


# ============================================================ wiring the names
#
# **This has to be the end of the file.** The two loops below look this file's
# functions up by name, so run higher up they cannot see what is not defined yet —
# `add` alone stopped with a `KeyError`.

for _nm in _INPLACE_UNARY + _INPLACE_MORE:
    setattr(Tensor, _nm + "_", _make_inplace(_nm))
for _nm in _INPLACE_BINARY + _INPLACE_ARGS:
    setattr(Tensor, _nm + "_", _make_inplace(_nm, "args"))


# ---- module functions exposed **as methods too**
#
# torch gives both — `torch.add(x, y)` and `x.add(y)`. `borch/__init__.py` has the
# loop going the other way (method → module function), and **this direction was
# missing.** So the computation was all there and the name reached from one side
# alone — `borch.matrix_exp(x)` worked and `x.matrix_exp()` did not. `x.add(y)` is
# a very common shape in torch code.
#
# **Not every name may be attached.** Making a method out of everything on the
# module creates methods torch does not have, and then people write code that runs
# only against us. So a list is written, and whether that list really consists of
# torch methods is confirmed by `tests/test_tensor_api.py`.
_AS_METHOD = (
    "add", "sub", "mul", "div", "subtract", "multiply", "divide", "true_divide",
    "floor_divide", "remainder", "fmod", "float_power", "lerp",
    "greater", "greater_equal", "less", "less_equal", "not_equal",
    "logical_and", "logical_or", "logical_not", "logical_xor",
    "isclose", "isneginf", "isposinf", "isreal", "nan_to_num",
    "fmax", "fmin", "cross", "inner", "kron", "vdot", "count_nonzero",
    "corrcoef", "cov", "adjoint", "broadcast_to", "moveaxis", "t",
    "det", "logdet", "slogdet", "inverse", "pinverse", "matrix_exp",
    "matrix_power", "cholesky", "qr", "svd",
    "digamma", "erfinv", "lgamma", "hardshrink", "prelu", "log_softmax",
    "bitwise_and", "bitwise_or", "bitwise_xor", "bitwise_not",
    "bitwise_left_shift", "bitwise_right_shift", "gcd", "lcm",
    "nextafter", "frexp", "logcumsumexp", "mvlgamma", "i0",
    # `fill` is not here — torch keeps it at the top level alone and offers only
    # `fill_` as a method.
    "clamp_max", "clamp_min", "detach_",
    # Shape and indexing. `unravel_index`, `cartesian_prod`, `combinations`,
    # `tril_indices`, `triu_indices`, `vander` and `chain_matmul` are not here —
    # torch keeps those at the top level alone.
    "as_strided", "as_strided_", "as_strided_scatter", "diag_embed",
    "diagonal_scatter", "select_scatter", "slice_scatter", "split_with_sizes",
    "tensor_split", "unique_consecutive",
    "index_put", "index_put_", "index_reduce", "masked_scatter",
    "masked_scatter_", "put", "renorm", "scatter_reduce", "ger", "mv",
    # The addmm family. **The in-place versions exist as methods only in torch** —
    # there is no top-level name `torch.addmm_` (measured). So they are here and
    # not in `borch/__init__.py`. The one exception is `addmv_`, which torch alone
    # also keeps at the top level.
    "addmm", "addmm_", "addbmm", "addbmm_", "baddbmm", "baddbmm_",
    "addmv", "addmv_", "addr", "addr_", "addcmul", "addcmul_",
    "addcdiv", "addcdiv_", "sspaddmm",
    # Top-level linear algebra. `lu_unpack`, `lobpcg`, `pca_lowrank` and
    # `svd_lowrank` are not here — torch keeps those four at the top level alone
    # (measured).
    "cholesky_solve", "cholesky_inverse", "triangular_solve", "orgqr", "ormqr",
    # Statistics. `histogramdd`, `gradient`, `trapz`, `normal`, `poisson` and
    # `binomial` are not here — torch keeps those at the top level alone
    # (measured).
    "histc", "histogram", "mode", "nanmedian", "bernoulli", "nonzero_static",
    "stft", "istft", "hash_tensor",
    # The neighbourhood of complex numbers. `asarray`, `frombuffer`, `range` and
    # `empty_strided` are not here — torch keeps those at the top level alone
    # (measured).
    "angle", "conj", "conj_physical", "conj_physical_", "imag", "is_complex",
    "is_conj", "is_neg", "resolve_conj", "resolve_neg",
)


def _as_method(name):
    fn = globals()[name]

    def method(self, *args, **kw):
        return fn(self, *args, **kw)

    method.__name__ = name
    method.__doc__ = f"The same as `torch.{name}(x, ...)`. torch offers both."
    # **The forwarded signature has to survive.** This binder writes
    # `(self, *args, **kw)` and hands everything to `fn`, so the two lists are the
    # same modulo the receiver — but `inspect.signature` sees the wrapper, and every
    # check that reads one goes blind here. Fifteen rows on the core-to-torch axis
    # read as `variadic`, which means *cannot be compared at all*, for names whose
    # module form is fully spelled out three lines above.
    method.__wrapped__ = fn
    return method


for _nm in _AS_METHOD:
    if not hasattr(Tensor, _nm):
        setattr(Tensor, _nm, _as_method(_nm))

# **`lu` and `lu_solve` each have two names.** The ones in this file are
# `linalg`'s (`lu` spreads `P`, `L` and `U`, and `lu_solve` takes the factor
# first), and the methods have to be **the top-level ones.** `_AS_METHOD` works
# only where there is one name, so these two are attached by hand — put into the
# list they would give methods that call the wrong function.
# **`Tensor.relu` was `F.relu` itself, so `F`'s `inplace` seat appeared on the
# method.** torch does not put it there — `x.relu(inplace=True)` raises
# `TensorBase.relu() takes no keyword arguments`, while `F.relu(x, inplace=True)`
# works. Measured; and the core offered a call torch rejects, so code written against
# it fails on the real thing.
#
# One object, two doors, two signatures — the same shape as `matmul` (whose free
# function is `other` and whose method is too, once you call it rather than read the
# docstring) and `qr` (a boolean at the top level, a string under `linalg`). The
# functional keeps its argument; the method gets its own two lines.
#
# The signature axis reported this as **borch.ts being short of `inplace`**, because
# it compares the core against borch.ts and torch is in neither column. It is the
# core that was long. `tests/test_torch_names.py` did not catch it either: that axis
# compares names at the same position and says so — it does not look at length.
def _relu_method(self):
    """`x.relu()`. **No `inplace` here** — torch keeps that on `F.relu` alone."""
    return relu(self)


Tensor.relu = _relu_method

Tensor.lu = _as_method("lu_top")
Tensor.lu.__name__ = "lu"
Tensor.lu_solve = _as_method("lu_solve_top")
Tensor.lu_solve.__name__ = "lu_solve"


# ── the seven that draw from a distribution and fill in place ───────────────
#
# **They live in `_ops.py` — `_rng` lives here.** They were first put in
# `_tensor.py` and grabbed `from ._ops import _rng` on every call, and when the
# check that clears `borch.*` out of `sys.modules` (`test_alias`) runs first they
# grab **a different `_ops`'s generator.** Then planting a seed has no effect, and
# the symptom is "it works alone and not when everything runs together", which is
# a long way from the cause.
#
# There is no place with a deterministic extreme as `bernoulli_` has, so **the
# values cannot be pinned.** What the table asks about is three things rather than
# the values — whether the shape and dtype stay put, whether an unusable dtype is
# refused, and whether the arguments' domain is honoured. The last two drift
# especially easily: **torch's rule differs per distribution, down to the exception
# type** (measured).
#
#   The continuous distributions **refuse** integers and booleans — `normal_`,
#   `uniform_` and `log_normal_` with a `NotImplementedError`, and `exponential_`
#   and `cauchy_` with a `RuntimeError` stating the reason. `geometric_` is
#   **discrete and runs on integers.** Grouped by name as "random means floats
#   only", that one comes out wrong.
#
#   `random_` runs on any dtype and **its range depends on the dtype** — int64
#   goes to that dtype's maximum and bool is {0,1}.
_CONTINUOUS_REFUSAL = {
    "normal_": ("NotImplementedError", '"normal_kernel_cpu" not implemented for'),
    "uniform_": ("NotImplementedError", '"check_uniform_bounds" not implemented for'),
    "log_normal_": ("NotImplementedError", '"log_normal_cpu" not implemented for'),
    "exponential_": ("RuntimeError",
                     "Exponential distribution is a continuous probability "
                     "distribution. dtype must be a floating point but you "
                     "specified"),
    "cauchy_": ("RuntimeError",
                "Cauchy distribution is a continuous probability distribution. "
                "dtype must be a floating point but you specified"),
}


def _refuse_leaf(self, name):
    if self.requires_grad and _grad_mode.enabled:
        raise RuntimeError(_like_torch(
            f"`{name}` cannot be used on a leaf tensor that requires grad. "
            "Do it inside `with torch.no_grad():`.",
            "a leaf Variable that requires grad is being used in an in-place operation"))


def _needs_continuous(self, name):
    """A continuous distribution fills floating point cells only — **the
    exception type differs per distribution.**"""
    if self.data.dtype.kind == "f":
        return
    kind, phrase = _CONTINUOUS_REFUSAL[name]
    shown = _TYPE_NAMES.get(self.data.dtype.kind, "Long")
    error = NotImplementedError if kind == "NotImplementedError" else RuntimeError
    raise error(_like_torch(
        f"`{name}` fills a floating point tensor only (it received "
        f"{self.dtype}).",
        f"{phrase} '{shown}'"))


def _fill_from(self, name, draw, generator=None):
    """**`generator` picks the stream and the nine callers used to drop it.**

    Every one of them opened with `del generator` — accepted, discarded, and no
    warning, so `x.normal_(generator=g)` drew from the global stream and the caller's
    reproducibility was quietly somebody else's. `bernoulli` two hundred lines up has
    done the right thing in one line the whole time, and `Generator` exists here and
    is honoured by `DataLoader`, so this was wiring rather than a missing mechanism.
    """
    _refuse_leaf(self, name)
    if name in _CONTINUOUS_REFUSAL:
        _needs_continuous(self, name)
    rng = generator.rng() if generator is not None else _rng
    self.data[...] = _np.asarray(draw(rng, self.data.shape),
                                 dtype=self.data.dtype)
    return self


def _normal_(self, mean=0.0, std=1.0, generator=None):
    if std < 0:
        raise RuntimeError(_like_torch(
            f"the standard deviation for normal_ must be >= 0 (got {std}).",
            f"normal expects std >= 0.0, but found std {std}"))
    return _fill_from(self, "normal_", lambda r, s: r.normal(mean, std, s), generator)


def _uniform_(self, from_=0.0, to=1.0, generator=None):
    if from_ > to:
        raise RuntimeError(_like_torch(
            f"uniform_ takes [from, to) (got {from_}, {to}).",
            f"uniform_ expects to return a [from, to) range, but found from={from_} > to={to}"))
    return _fill_from(self, "uniform_", lambda r, s: r.uniform(from_, to, s), generator)


def _exponential_(self, lambd=1.0, generator=None):
    if lambd <= 0:
        raise RuntimeError(_like_torch(
            f"the lambda for exponential_ must be > 0 (got {lambd}).",
            f"exponential_ expects lambda > 0.0, but found lambda={lambd}"))
    return _fill_from(self, "exponential_",
                      lambda r, s: r.exponential(1.0 / lambd, s), generator)


def _cauchy_(self, median=0.0, sigma=1.0, generator=None):
    return _fill_from(self, "cauchy_",
                      lambda r, s: median + sigma * r.standard_cauchy(s), generator)


def _log_normal_(self, mean=1.0, std=2.0, generator=None):
    return _fill_from(self, "log_normal_", lambda r, s: r.lognormal(mean, std, s), generator)


def _geometric_(self, p, generator=None):
    """**Discrete, so it runs on an integer tensor too.** The one that parts from
    the five continuous ones."""
    if not 0 < p < 1:
        raise RuntimeError(_like_torch(
            f"the p for geometric_ must be in (0, 1) (got {p}).",
            f"geometric_ expects p to be in (0, 1), but got p={p}"))
    return _fill_from(self, "geometric_", lambda r, s: r.geometric(p, s), generator)


def _random_(self, from_=0, to=None, generator=None):
    """**The range depends on the dtype** — given nothing it reaches as far as
    that dtype can hold."""
    # **The upper bound is as far as that dtype counts exactly.** Past 2^24
    # float32 cannot tell neighbouring integers apart, so drawing above that
    # clumps the values — torch cuts it there too (measured: the maximum is
    # 1.677e7). It was first set at 2^53, and that is float64's place while our
    # storage is float32.
    kind, bits = self.data.dtype.kind, self.data.dtype.itemsize * 8
    if to is None:
        to = {"b": 2, "f": 1 << 24, "i": 1 << (bits - 1), "u": 1 << bits}[kind]
    if from_ >= to:
        raise RuntimeError(_like_torch(
            f"the from for random_ must be less than to (got {from_}, {to}).",
            f"random_ expects 'from' to be less than 'to', but got from={from_} >= to={to}"))
    return _fill_from(self, "random_",
                      lambda r, s: r.integers(from_, to, s), generator)


for _rname, _rfn in (("normal_", _normal_), ("uniform_", _uniform_),
                     ("exponential_", _exponential_), ("cauchy_", _cauchy_),
                     ("log_normal_", _log_normal_), ("geometric_", _geometric_),
                     ("random_", _random_)):
    setattr(Tensor, _rname, _rfn)
del _rname, _rfn


def _bernoulli_(self, p=0.5, generator=None):
    """**It fills from `p` — it does not read its own values as probabilities.**
    Different from `bernoulli()`.

    **It lives in `_ops` — `_rng` lives here.** Put in `_tensor.py` at first, it
    used `_np.random` in place of the `_rng` that file does not have, and that is
    numpy's **global** generator, which `manual_seed` does not reach. Drawing twice
    from the same seed gave different values and nobody asked — the seed check ran
    over the seven added later and did not count this one.
    """
    _refuse_leaf(self, "bernoulli_")
    # As `bernoulli` above, and as `_fill_from` now: the stream the caller asked for.
    rng = generator.rng() if generator is not None else _rng
    self.data[...] = (rng.random(self.data.shape) < p).astype(self.data.dtype)
    return self

Tensor.bernoulli_ = _bernoulli_


# ── the three torch offers as **properties** ────────────────────────────────
#
# `x.device`, `x.real` and `x.imag` take no parentheses in torch. All three
# existed here as functions alone, and `_as_method` had attached two of them **as
# methods** — and then `x.real` hands back **a bound method object** rather than a
# tensor. No exception either, and a line like `if x.imag:` passes as true. torch
# stops on `imag` for a real tensor.
#
# `x.device` was absent entirely. The name `torch.device(...)` existed and the
# tensor-side property did not, so **`print(x.device)` and
# `if x.device.type == "cuda"` stop there** — the line a textbook types when
# checking the device.
Tensor.device = property(
    lambda self: _device("cpu"),
    doc="Always `cpu` — it sits on numpy and there is no other device.")
Tensor.real = property(
    real, doc="The real part. On a real tensor it is the tensor itself.")
Tensor.imag = property(
    imag,
    doc="The imaginary part. **It stops on a real tensor** — torch does that.")
