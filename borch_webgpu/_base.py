"""The handle Python holds on a JavaScript tensor, and the path that reads
values back **synchronously.**

This file is fair to call the whole binding. The rest is transcribing names.
"""

import js as _js
import numpy as _np
from pyodide.ffi import run_sync as _run_sync, to_js as _to_js

_ts = _js.borch

def _js_list(seq):
    """A Python list to a JS array. Passed as a proxy, the other side does not
    see an array."""
    return _to_js(list(int(n) for n in seq))


def _js_options(**kw):
    """A Python dict to **a plain JS object.**

    `to_js` turns a dict into a `Map` by default, and borch.ts reads its option
    arguments as properties — `options.requiresGrad` — so a `Map` arriving there
    makes **all of them `undefined`.** Nothing raises and the defaults are used
    quietly. This binding broke in exactly that way when a positional argument
    became an options object, and fifteen `edge::grad::*` cases in the browser
    golden caught it as "gradients were never turned on". The TypeScript runner
    does not go through this path.
    """
    return _to_js(kw, dict_converter=_js.Object.fromEntries)


def camel_name(name):
    """`requires_grad` to `requiresGrad`. The same rule as `_ops.camel`, but
    using that one here would mean importing it, and that is a cycle — so this
    single line stands on its own."""
    head, *rest = name.split("_")
    return head + "".join(p[:1].upper() + p[1:] for p in rest)


def _js_floats(seq):
    """The float version of `_js_list`. **Some places must not be truncated** —
    fractional pooling's samples lie between 0 and 1, so passing them through
    `int()` makes them all 0, and that 0 is not an error but **a window position
    with an answer**, which quietly becomes a different layer."""
    return _to_js(list(float(v) for v in seq))


def _read(handle):
    """Read values off the GPU — **without `await`.**

    WebGPU has no synchronous read. borch.ts's `toArray()` stands on `mapAsync`
    and returns a promise. `run_sync` fills that gap through JSPI (WebAssembly's
    promise integration): it suspends the Python stack and resumes it when the
    value arrives.

    **Only under an asynchronous entry point.** The page has to come in through
    `runPythonAsync` for there to be anywhere on the stack to suspend; entering
    through `runPython` stops with `RuntimeError: No suspender`. That is the
    stack's situation rather than a limit of the library. It was measured
    (`tests/browser/sync_probe.py`), and this binding rests on that one fact —
    without it every read becomes `await loss.item()` and the project's claim
    breaks.

    ## Why it goes through `to_py()` — **negative zero**

    This used to be `_np.asarray(js_array, dtype=float32)`, and that path
    **turns `-0.0` into `0.0`** (measured). On the JavaScript side
    `Object.is(a[0], -0)` was still true; the sign disappeared only after the
    move into numpy, because that path copies element by element. `to_py()`
    hands back **a memoryview**, so `frombuffer` reads the bytes as they are.

    Comparing values can never catch this — `-0.0 == 0.0`. **Only the printed
    form catches it:** `tensor([1.-0.j])` came out as `tensor([1.+0.j])`. It
    surfaced while freezing the complex repr, and it had been there for real
    tensors the whole time.
    """
    raw = _run_sync(handle.toArray())
    # **Take a copy.** `frombuffer` gives a read-only view into the WASM heap,
    # and holding on to it means the next read overwrites that memory.
    return _np.frombuffer(raw.to_py(), dtype=_np.float32).copy()


def int64_name():
    """The dtype of an index tensor. The name is written in one place."""
    return _DType("int64")


def _core_repr(shim):
    """Borrow the core's `_tensor_repr`. In a browser it lives under `/work`."""
    global _REPR
    if _REPR is None:
        from borch._base import _tensor_repr as fn
        _REPR = fn
    return _REPR(shim)


_REPR = None


class _Shim:
    """Imitates only what `_tensor_repr` looks at — `.data`, `.dtype`, `._op`,
    `.requires_grad`."""

    __slots__ = ("data", "dtype", "_op", "requires_grad")

    def __init__(self, t):
        self.data = t.numpy()
        self.dtype = t.dtype
        self._op = t._h.gradName or None
        self.requires_grad = bool(t._h.requiresGrad)


class _DType(str):
    """A dtype name. The value is borch.ts's name and **what is shown is
    torch's.**

    The golden froze `str(x.dtype)` as an answer, and that answer is
    `torch.float32`. Internally it has to travel as `"float32"` to be handed
    straight to borch.ts — so it subclasses `str` and carries both names in one
    object.
    """

    __slots__ = ()

    def __repr__(self):
        return f"torch.{self}" if self != "bool" else "torch.bool"

    def __str__(self):
        return f"torch.{str.__str__(self)}"

    @property
    def plain(self):
        return str.__str__(self)


class _Size(tuple):
    """A shape. It has to print as `torch.Size([2, 2])` — the golden froze that
    string."""

    __slots__ = ()

    def __repr__(self):
        return f"torch.Size([{', '.join(str(n) for n in self)}])"

    __str__ = __repr__


# The names `__getattr__` must not forward, each with the reason it cannot be.
#
# **Lifted out of `__getattr__`.** In there it made a 165-line function, of which
# 93 lines were comment and 45 were this list; the dispatch itself is about twenty.
# The rule that stops a function at a hundred lines exists to catch several
# responsibilities in one place, and this was one responsibility and a table. Out
# here the function reads at its real size, and the table can be checked directly —
# which nothing could do while it was an expression inside a method.
_NOT_FORWARDED = ("clamp", "clip", "split", "chunk", "aminmax", "flip",
            "pow", "squeeze", "repeat_interleave", "flatten",
            "sum", "norm", "transpose", "swapdims", "remainder",
            # `max` and `min` split three ways by argument. Over there
            # only the middle one exists and its default dimension is 0,
            # so forwarding makes `x.max()` produce a pair reduced along
            # dimension 0 instead of the overall maximum.
            "max", "min",
            # The indexing side — the names and arguments differ from
            # borch.ts's.
            "scatter", "scatter_add", "index_add", "index_copy",
            "index_fill", "take", "take_along_dim",
            # **Names on the module that borch.ts does not have.**
            # Everything here is a combination of computations that
            # already exist, so no name was added over there — which left
            # a one-sided link where `borch.t(x)` works and `x.t()` does
            # not. torch offers both, and textbook code uses the method
            # form more.
            #
            # This list checks itself: every name has a golden case, and
            # the golden was frozen **by running real torch**, so a name
            # torch does not have stops at the freezing step.
            "multiply", "true_divide", "floor_divide", "lerp",
            "greater", "less_equal", "isclose", "nan_to_num", "fmax",
            "inner", "adjoint", "moveaxis", "t", "corrcoef", "cov",
            "vdot", "kron", "broadcast_to",
            # Six were missing. **The pair was above and only the alias
            # was absent**, so `x.multiply_(3)` worked and
            # `x.divide_(2)` did not — two names standing side by side
            # with one of them running is the least visible kind. It
            # turned up while filling in forty-one in-place names in the
            # core and measuring all three against each other.
            "divide", "subtract", "greater_equal", "less", "not_equal",
            "logical_xor",
            # On bool it has to branch to logical negation, and that
            # branch lives in `_ops`.
            "bitwise_not",
            # The shape and indexing ones **hand-written on the module.**
            # They take a list of tensors (`index_put`) or answer with a
            # tuple (`tensor_split`), so forwarding the name gets caught
            # on the JavaScript side at integer conversion or list form.
            "index_put", "index_put_", "tensor_split",
            "split_with_sizes", "unique_consecutive",
            # Sparse-only, so they only refuse — borch.ts has no such
            # names.
            "sspaddmm",
            # **The two whose names collide with the `linalg` namespace.**
            # Forwarded, `luSolve` gets picked up and answers differently
            # with the argument order reversed.
            "lu", "lu_solve",
            # The statistics ones hand-written on the module — random
            # draws, refusals, compositions, and `histogramdd`, which
            # takes its edges as a list.
            "bernoulli", "float_power", "stft", "istft", "hash_tensor", "trapz",
            "histogramdd",
            # Complex's neighbours — identities, so borch.ts has no name
            # for them.
            "real", "conj", "conj_physical", "conj_physical_",
            "resolve_conj", "resolve_neg", "imag", "angle",
            "is_complex", "is_conj", "is_neg")


class Tensor:
    """Wraps one borch.ts tensor.

    **The values are not held in Python.** Only the handle is; the values are
    read off the GPU when they are needed. Keeping them on both sides means a
    day arrives when the two disagree about which is real.
    """

    __slots__ = ("_h",)

    def __init__(self, handle):
        self._h = handle

    def __setattr__(self, name, value):
        """**Attributes are written on the tensor over there.** `p.grad = g` is
        an ordinary line of torch code.

        `__slots__` held only `_h`, so any other name was simply an
        `AttributeError`. Reads (`__getattr__`) were forwarded and writes were
        not — code feeding an optimiser by hand stops on its first line.
        """
        if name == "_h":
            object.__setattr__(self, name, value)
            return
        # **`None` has to arrive as JavaScript's `null`.** Pyodide passes it as
        # `undefined`, and borch.ts asks **strictly** — `node.grad === null`
        # (`autograd.ts`). So a backward pass after `p.grad = None` fails to
        # recognise "empty", tries to accumulate, and dies with
        # `Cannot read properties of undefined (reading 'add')`, a long way from
        # the Python line that caused it.
        #
        # **Python cannot produce a `null`.** Pyodide hands `null` back as
        # `None`, so there is no way to go the other way. The place that makes
        # one lives over there instead (`borch.setNull`).
        if value is None:
            _ts.setNull(self._h, camel_name(name))
            return
        setattr(self._h, camel_name(name),
                handle(value) if isinstance(value, Tensor) else value)

    # ── the two the harness requires ──────────────────────────────────────
    #
    # `to_numpy` in `tests/cases.py` calls only `t.detach().numpy()`. Supply
    # those two and the golden harness compares this implementation without a
    # single line changing.

    def detach(self):
        return Tensor(self._h.detach())

    def numpy(self):
        flat = _read(self._h)
        shape = self.shape
        kind = str(self._h.dtype)
        # **Only complex has a different element count from its buffer length.**
        # borch.ts stores it interleaved, so 2n values arrive as
        # `[re, im, re, im, …]`; reshaping straight to `shape` stops there
        # because there are twice as many slots. Other dtypes are labels only
        # and have no such case.
        if kind == "complex64":
            pair = flat.reshape(-1, 2)
            # **Written into the slots — not `re + 1j*im`.** That expression
            # loses **negative zero**: the imaginary part of `1j * (-0.0)` is
            # `-0.0`, and adding the real part's `+0.0` makes it `+0.0`.
            # `tensor([1.-0.j])` printed as `tensor([1.+0.j])`, which comparing
            # values cannot catch (they are `==`) — **only the printed form
            # catches it.**
            out = _np.empty(pair.shape[0], dtype=_np.complex64)
            out.real, out.imag = pair[:, 0], pair[:, 1]
            return out.reshape(shape) if shape else out.reshape(())
        out = flat.reshape(shape) if shape else flat.reshape(())
        # In borch.ts a dtype is **a label** over float32 storage. Coming back,
        # that label is followed — otherwise an int64 case comes out as floats
        # and the values still look right.
        if kind == "int64":
            return out.astype(_np.int64)
        if kind == "bool":
            return out.astype(bool)
        return out

    # ── being Pythonic ────────────────────────────────────────────────────

    @property
    def shape(self):
        return _Size(int(n) for n in self._h.shape)

    @property
    def dtype(self):
        """**It has to show as `torch.float32`.** borch.ts says `"float32"`.

        The golden's dtype cases froze **the dtype name as a string** rather
        than a value. A different name fails all of them even when every
        promotion rule is right — which is what happened.
        """
        return _DType(str(self._h.dtype))

    @property
    def ndim(self):
        return len(self.shape)

    def dim(self):
        return len(self.shape)

    def numel(self):
        return int(self._h.size)

    def size(self, dim=None):
        return self.shape if dim is None else self.shape[dim]

    def item(self):
        """**Exactly one element.** torch throws at that point and the golden
        froze it."""
        if self._h.size != 1:
            raise RuntimeError(
                f"a Tensor with {self._h.size} elements cannot be converted to Scalar")
        flat = _read(self._h)
        # **Complex comes back as a Python `complex`** — torch does that too.
        # borch.ts's `item()` refuses because JavaScript has no complex value;
        # Python has one.
        if str(self._h.dtype) == "complex64":
            return complex(float(flat[0]), float(flat[1]))
        return float(flat[0])

    def backward(self, *args):
        return guarded(self._h.backward, *[handle(a) for a in args])

    # The dtype-changing names. borch.ts takes them all as `to("float32")`.
    def to(self, dtype):
        name = dtype.plain if isinstance(dtype, _DType) else str(dtype)
        name = name.replace("torch.", "")
        # **Double precision stops here, and this is the one place it can** —
        # `.double()`, `.to(float64)` and `.type(float64)` are one request spelled
        # three ways and they all arrive at this line. It used to be guarded on
        # `.double()` alone, so the other two spellings went through to borch.ts
        # and **came back a float32 tensor claiming to be double.**
        #
        # The core places the same gate on its own `_cast`, in the same words. A
        # refusal that reads differently in two of our libraries teaches that the
        # limit is each library's habit rather than the platform's.
        if name == "float64":
            _absent_dtype("double", "float64")
        return guarded(self._h.to, name)

    def float(self):
        return self.to("float32")

    def double(self):
        """**Absent.** WebGPU shaders have no double precision, and TF.js has none
        for its own reasons — different cause, same conclusion. Handing back
        float32 quietly produces code that believes it computed in double."""
        return self.to("float64")

    def long(self):
        return self.to("int64")

    def int(self):
        """**There is no int32 — so this refuses.** It handed back int64 for a
        long time.

        The values look plausible, but code checking
        `x.int().dtype == torch.int32` diverges under real torch, and the cause
        surfaces far from this line. The core was fixed alongside it: **the
        refusal has to read the same in all three** or a learner reads it as
        "something each implementation does differently".
        """
        _absent_dtype("int", "int32")

    def type_as(self, other):
        """Match `other`'s dtype. The core did not have this name either, so it
        went into both."""
        return self.to(other.dtype if isinstance(other, Tensor) else other)

    # ── the four questions ────────────────────────────────────────────────
    #
    # Three fall straight out of the dtype and the values. The fourth,
    # `is_contiguous`, is a different kind of thing — **there are no views here,
    # so it is always true.** That is a fact about views rather than about this
    # predicate, and it shares a root with the places that refuse to propagate
    # one. In the core numpy hands back a transpose as a view, so it goes false
    # there, and that divergence is frozen separately in the golden.

    def is_floating_point(self):
        return self.dtype.plain in ("float32", "float64")

    # ── five that a dense tensor still answers ────────────────────────────
    #
    # By name they read as sparse or device things and it feels right to call
    # them absent, but torch simply answers them on a dense tensor — "this
    # tensor is dense", "it is on the CPU" are answers. They went into the core
    # as well.

    def dense_dim(self):
        return self.ndim

    # ── names that existed only on the module now exist as methods ────────
    #
    # torch offers nearly every operation both ways. Thirteen were module-only
    # in the core and the same here — and the side textbooks use is the method.

    def igamma(self, other):
        from . import _ops
        return _ops.igamma(self, other)

    def igammac(self, other):
        from . import _ops
        return _ops.igammac(self, other)

    def polygamma(self, n):
        """**The arguments are reversed** — the module has `polygamma(n, x)` and
        the method has `x.polygamma(n)`, which is how torch arranges it. Attached
        through the table it produces values with the order and the input
        swapped."""
        from . import _ops
        return _ops.polygamma(n, self)

    def polygamma_(self, n):
        return self._write_back(self.polygamma(n))

    # **The two torch removed in 1.9.** The names survive and calling one stops
    # — answering them here means that code breaks under real torch. The core
    # refuses them as well.
    def lstsq(self, *args, **kw):
        del args, kw
        _deprecated_by_torch("lstsq")

    def solve(self, *args, **kw):
        del args, kw
        _deprecated_by_torch("solve")

    # ── the three torch gives as **properties** ────────────────────────────
    #
    # No parentheses. Left alone, this binding's `__getattr__` hands back **a
    # function object** and nothing raises — `x.real` becomes something that is
    # not a tensor and `if x.imag:` passes as true. The core had the same defect
    # in the same place.
    #
    # `device` has to be **an object, not a string.** `x.device.type` is the line
    # a textbook writes to check the device, and a string stops there.
    @property
    def device(self):
        from . import _ops
        return _ops.device(str(self._h.device))

    @property
    def grad(self):
        """**An absent gradient and an absent name have to be told apart.**

        The general path (`__getattr__`) asks `getattr(self._h, name, None)` and
        stops with "no such name" on `None`. But borch.ts's `grad` is
        `Tensor | null`, and Pyodide hands JavaScript's `null` back as Python's
        `None` — **the two cases become the same value.**

        So `p.grad` stopped with an `AttributeError` exactly when there was no
        gradient yet, which is precisely the case `if p.grad is not None:` was
        asking about. Code writing an optimiser by hand, code clipping
        gradients — all of it opens with that line.

        Writes were already forwarded by `__setattr__`. Only reads were missing,
        for **the same reason** `device`, `real` and `imag` are here — and this
        one was not on the list when those three were written.
        """
        got = self._h.grad
        return None if got is None else wrap(got)

    @property
    def real(self):
        from . import _ops
        return _ops.real(self)

    @property
    def imag(self):
        from . import _ops
        return _ops.imag(self)

    def resize_as_(self, other):
        """An underscore name with no pair, so the derived table does not build it.

        **It is in place** — handing back a new tensor would make the name a
        lie. `_write_back` follows the shape change too, which is the same path
        the other reshaping in-place names take.

        (The Korean this replaces had two backticked identifiers missing, and
        had since it was first committed. What it says here comes from reading
        the line below it rather than from guessing at the gaps.)
        """
        return self._write_back(self.reshape(*[int(v) for v in other.shape]))

    def is_same_size(self, other):
        return tuple(self.shape) == tuple(other.shape)

    def is_inference(self):
        return False

    def is_distributed(self):
        return False

    def share_memory_(self):
        """No sharing between processes. torch hands itself back on the CPU too."""
        return self

    def requires_grad_(self, requires_grad=True):
        """**Dropping the underscore lands on a bool property.** On the general
        path it looks up `requires_grad`, tries to call it, and stops with
        `'bool' object is not callable` — the rule that derives an in-place name
        from its pair **colliding with a property.**
        """
        self.requires_grad = bool(requires_grad)
        return self

    # ── the seven that draw from a distribution and fill in place ─────────
    #
    # **The values cannot be frozen** — all three random generators differ, which
    # was already accepted at `randn`. So what is matched is shape, dtype, and
    # **the refusals.** torch's rules differ per distribution, down to the
    # exception type; the core holds that table, so it is borrowed here.
    #
    # The values are made in Python and written back. Drawing them in a shader
    # would make two sets of seeding rules, and two sets diverge eventually.
    def _draw_(self, name, *args, **kw):
        from borch._tensor import Tensor as _Core

        core = _Core(self.numpy().copy())
        getattr(core, name)(*args, **kw)
        from ._base import tensor as _t
        return self._write_back(_t(core.data))

    def normal_(self, mean=0.0, std=1.0, generator=None):
        del generator
        return self._draw_("normal_", mean, std)

    def uniform_(self, from_=0.0, to=1.0, generator=None):
        del generator
        return self._draw_("uniform_", from_, to)

    def exponential_(self, lambd=1.0, generator=None):
        del generator
        return self._draw_("exponential_", lambd)

    def cauchy_(self, median=0.0, sigma=1.0, generator=None):
        del generator
        return self._draw_("cauchy_", median, sigma)

    def log_normal_(self, mean=1.0, std=2.0, generator=None):
        del generator
        return self._draw_("log_normal_", mean, std)

    def geometric_(self, p, generator=None):
        """**Discrete, so it runs on an integer tensor too** — the one that
        parts ways with the five continuous ones."""
        del generator
        return self._draw_("geometric_", p)

    def random_(self, from_=0, to=None, generator=None):
        del generator
        return self._draw_("random_", from_, to)

    def fill_diagonal_(self, value, wrap=False):
        """**The composition moved over there.** While it lived here the name did
        not exist in borch.ts, and the golden goes through this method, so the
        table was green — it was missing on the TypeScript side only."""
        guarded(handle(self).fillDiagonal_, float(value), bool(wrap))
        return self

    def sparse_dim(self):
        return 0

    def to_dense(self):
        return self

    def storage_offset(self):
        return 0

    def get_device(self):
        """-1 means there is no device index — not an error (measured)."""
        return -1

    def is_signed(self):
        return self.dtype.plain not in ("bool", "uint8")

    def is_nonzero(self):
        if self.numel() != 1:
            raise RuntimeError(
                f"The truth value of a tensor with {self.numel()} values is ambiguous. "
                "(torch: Boolean value of Tensor with "
                f"{'no values' if self.numel() == 0 else 'more than one value'}"
                " is ambiguous)")
        return bool(self.item() != 0)

    def is_contiguous(self):
        """**Always true** — GPU buffers are not shared out as views, so there is
        nowhere for non-contiguous to arise. In the core numpy's views make it
        false after a transpose."""
        return True

    def contiguous(self):
        """Already contiguous, so it hands itself back. In the core a
        non-contiguous tensor gets copied."""
        return self

    def cfloat(self):
        """complex64. **Not a relabel** — borch.ts stores complex interleaved as
        `[re, im]`, so there are twice as many slots. A zero imaginary part is
        attached to make a real one."""
        from . import _ops
        return _ops.complex(self, _ops.zeros_like(self))

    def bool(self):
        return self.to("bool")

    def type(self, dtype=None):
        return self.dtype if dtype is None else self.to(dtype)

    def tolist(self):
        return self.numpy().tolist()

    def __len__(self):
        """**A scalar has no length, and saying `0` is worse than refusing.**

        This returned `0` for a 0-dimensional tensor. torch raises
        `TypeError: len() of a 0-d tensor` and the numpy core raises
        `len() of unsized object`, and the difference is not cosmetic: `len()` is
        how numpy decides whether something is a sequence, so `np.asarray(t)`
        walked one level past the last axis, found `0` there, and built a
        **(3, 5, 4, 0)** array from a (3, 5, 4) tensor. Empty, no error, wrong
        shape.

        It surfaced four golden rows away from here and in four different
        wordings — `RandomErasing` disagreeing on shape, `F.erase` failing an
        item assignment, `elastic` reshaping from size 0, and
        `LinearTransformation` reporting an image that "flattens to 0". None of
        them mentions `len`, and none of them is about vision.

        A scalar reading as empty is wrong on its own terms too: `if len(t):` is
        false for `tensor(5.)`, and `list(t)` is `[]`.
        """
        if not self.shape:
            raise TypeError("len() of a 0-d tensor")
        return self.shape[0]

    def __array__(self, dtype=None, copy=None):
        """What `np.asarray(t)` takes.

        Without it numpy falls back to `__len__` and `__getitem__` and walks the
        tensor element by element — one GPU read per scalar, and the shape decided
        by whatever `len()` says at the bottom. With it, one read of the whole
        buffer, which is what `numpy()` already does.

        `copy=False` is numpy 2's *"do not copy"* request. `numpy()` builds a fresh
        array from a buffer read out of the GPU, so there is nothing to share and
        the honest answer is to refuse rather than to hand back a copy and stay
        quiet — that is what numpy asks implementations to do.
        """
        if copy is False:
            raise ValueError(
                "a GPU tensor cannot be viewed without copying — the values are "
                "read out of a device buffer.")
        out = self.numpy()
        return out if dtype is None else out.astype(dtype)

    def __repr__(self):
        """**Borrows the core's rules.** Written again here, a day comes when the
        second copy differs.

        `_tensor_repr` in `borch/_base.py` already carries torch's printing rules
        — alignment, digits, line breaks, the eight-space indent — and the golden
        froze those strings as answers. Hand it the values and a few flags and it
        produces the answer.
        """
        return _core_repr(_Shim(self))

    __str__ = __repr__

    # ── everything else is forwarded ──────────────────────────────────────

    def __getattr__(self, name):
        """Names this class does not have are forwarded **to the borch.ts
        tensor.**

        Many cases call through a method — `x.exp()`, `x.masked_select(m)`.
        Transcribing them by hand means a day arrives when one of them calls a
        different operation, so they are forwarded instead of written. A missing
        name stops with `AttributeError` — nothing is approximated.

        **Whether it is a method or a property is asked of the JavaScript side.**
        Wrapping everything as a method made `x.T` and `x.grad` return functions,
        which was 96 failures reported as
        `'function' object has no attribute 'detach'`. A function returned where
        a value belongs blows up on the next line, and the cause moves one step
        away from the symptom.
        """
        from ._ops import (
            _BINARY_ONLY, _EXTREME, camel, positional, refuse_if_nullary,
        )

        # What was hand-written on the module has to be the same as a method —
        # places where the argument order is reversed (`split`) or only one side
        # may arrive (`clamp`).
        #
        # **The in-place versions come here too.** Forwarding `transpose_`
        # straight to borch.ts stops, because its `transpose_()` takes no
        # dimensions and `transpose_(0, 1)` does — resolving dimensions happens
        # in this file, so the underscore side has to pass through it as well.
        inplace = name.endswith("_") and not name.endswith("__")
        bare = name[:-1] if inplace else name
        if bare in _NOT_FORWARDED:
            from . import _ops
            # **`max` and `min` are not in the module globals.** Putting those
            # names in `_ops` shadows the Python builtins inside that file, and
            # a place sizing something with `max(a, b)` calls a tensor function
            # instead — the symptom is a failed GPU buffer allocation, a very
            # long way from the cause.
            # **A separate underscore name is used first when one exists.**
            # Usually calling the pair and writing the value back is enough, but
            # an underscore **does not guarantee the same operation** —
            # `bernoulli_(p)` ignores its own values and fills with `p`, unlike
            # `bernoulli()`, which reads its own values as probabilities. Routed
            # through the pair, even the argument count fails to match.
            #
            # **Looked up through `__dict__`.** `getattr` wakes this module's
            # `__getattr__`, which hands a missing name back as itself — infinite
            # recursion, reported as `maximum recursion depth exceeded`, a long
            # way from the cause.
            exact = _ops.__dict__.get(name) if inplace else None
            if exact is not None:
                # **An in-place name, so it writes back.** Returning only the
                # value leaves `x.bernoulli_(0)` without changing `x`, and that
                # is not an in-place operation.
                return self._when_run(name,
                    lambda *a, **k: self._write_back(exact(self, *a, **k)))
            fn = _EXTREME[bare] if bare in _EXTREME else getattr(_ops, bare)
            if not inplace:
                return lambda *a, **k: fn(self, *a, **k)
            return self._when_run(name,
                lambda *a, **k: self._write_back(fn(self, *a, **k)))

        # **A leaf with gradients on cannot be edited in place.** torch throws
        # there and the golden froze it — let it through and a backward pass
        # reads a value that has already moved.
        # **Inside `no_grad` it is allowed.** torch does that too: while no
        # gradients are being built, editing a leaf leaves nothing for a backward
        # pass to see. Leaving that condition out blocked the ordinary path where
        # an optimiser updates its parameters.
        # **`detach_` is the exception.** It leaves the values alone and only
        # cuts the graph, so no backward pass can read a moved value — torch
        # allows it on a leaf as well.
        #
        # **It is refused when it runs, not when the name is looked up.** Raised
        # here, `hasattr(p, "copy_")` on a parameter *raises* — `hasattr` catches
        # `AttributeError` and nothing else, so a `RuntimeError` about an in-place
        # operation comes out of a question about whether a name exists, before any
        # `with no_grad()` the caller was about to enter. That is what `_fill` in
        # the case table does, and **ten golden cases stopped there** while the very
        # next line would have made the write legal.
        guarded_inplace = inplace and bare != "detach"

        js_name = camel(name)

        # **Names borch.ts has no in-place version of.** Over there the `abs_`
        # form is derived automatically from the unary table only; the binary
        # ones (`eq_`) are not in it. In those places **the computation is done
        # by the version without the underscore** and the result is copied into
        # this buffer — the same place and the same reason as the core's
        # `_inplace`. Two copies of the expression diverge eventually, and the
        # values stay plausible enough that nobody sees it.
        if inplace and getattr(self._h, js_name, None) is None:
            return self._when_run(name, lambda *a, **k: self._write_back(
                getattr(self, bare)(*a, **k)), guarded_inplace)

        if name in _BINARY_ONLY:
            # borch.ts derives methods from the table for unary only. Binary
            # goes through `binary(name, other)`.
            return self._when_run(
                name,
                lambda other, *_: guarded(self._h.binary, js_name, handle(other)),
                guarded_inplace)
        got = getattr(self._h, js_name, None)
        if got is None:
            # **The opening clause matches torch's.** It used to say only that
            # borch.ts has no such name on its tensor, while asking the core the
            # same name produced Python's standard wording — and a learner
            # reads those two as **something each implementation does
            # differently.** Real torch says
            # `'Tensor' object has no attribute 'x'`.
            #
            # The hint after it is for us, so it stays. Checks that match on the
            # opening clause still pass, and fixing the binding needs `js_name`.
            raise AttributeError(
                f"'Tensor' object has no attribute '{name}'"
                f" — borch.ts does not have `{js_name}`")
        if not callable(got):
            return settle(got)

        def call(*args, **kw):
            laid = positional(name, args, kw)
            refuse_if_nullary(js_name, got, len(laid))
            out = guarded(got, *laid)
            # **An in-place operation has to hand itself back.** borch.ts's
            # `abs_` returns the same handle, but `guarded` **wraps it in a new
            # Python tensor**, so `x.absolute_() is x` becomes false. Then code
            # chaining calls — `x.mul_(2).add_(1)` — starts editing a copy
            # rather than the original.
            return self if inplace else out

        call.__name__ = name
        return self._when_run(name, call, guarded_inplace)

    def _when_run(self, name, call, guard=True):
        """The refusal moved from the lookup to the call.

        `x.copy_` is a question about a name and `x.copy_(...)` is the operation.
        Refusing at the first means `hasattr` — and anything that asks whether a
        tensor can do something before deciding how — stops with a `RuntimeError`
        that no caller expected there.
        """
        if not guard:
            return call

        def go(*args, **kw):
            self._refuse_inplace_on_leaf(name)
            return call(*args, **kw)

        go.__name__ = name
        return go

    def _refuse_inplace_on_leaf(self, _name):
        """**A leaf with gradients on cannot be edited in place.** torch throws
        there and the golden froze it — let it through and a backward pass reads
        a value that has already moved.

        **Inside `no_grad` it is allowed**, as it is in torch: while no
        gradients are being built, editing a leaf leaves nothing for a backward
        pass to see. Leaving that condition out blocked the ordinary path where
        an optimiser updates its parameters.
        """
        if (_ts.gradMode.enabled and bool(self._h.requiresGrad)
                and not self._h.parents.length):
            raise RuntimeError(
                "a leaf Variable that requires grad is being used in an "
                "in-place operation.")

    def _write_back(self, out):
        """Copy a computed value into this buffer and hand back **the same
        tensor.**

        borch.ts's `copyFrom` follows a change of shape too — `transpose_` keeps
        the element count and changes the frame they are read through.
        """
        self._h.copyFrom(handle(out))
        return self

    # Operators. `x + y` is `x.add(y)`, and the other side may be a number.
    def _op(js_name):                                        # noqa: N805
        def go(self, other):
            return guarded(self._h.binary, js_name, handle(other))
        return go

    def _rop(js_name):                                       # noqa: N805
        def go(self, other):
            return guarded(handle(other).binary, js_name, self._h)
        return go

    __add__, __radd__ = _op("add"), _rop("add")
    __sub__, __rsub__ = _op("sub"), _rop("sub")
    __mul__, __rmul__ = _op("mul"), _rop("mul")
    __truediv__, __rtruediv__ = _op("div"), _rop("div")
    def __pow__(self, other):
        """**An integer exponent is unrolled into multiplication.** WGSL's `pow`
        is `exp2(y·log2(x))`, which has no answer for a negative base: the
        forward pass happens to be right for even exponents and the gradient
        comes out nan. borch.ts's `powScalar` exists for that case."""
        from ._ops import pow as _pow
        return _pow(self, other)
    __eq__, __ne__ = _op("eq"), _op("ne")
    __lt__, __le__ = _op("lt"), _op("le")
    __gt__, __ge__ = _op("gt"), _op("ge")

    def __mod__(self, other):
        return guarded(self._h.remainder, float(other))

    def __matmul__(self, other):
        """`a @ b` is `matmul`, **not `mm`.** torch's operator batches and broadcasts;
        `mm` is the two-dimensional kernel underneath it and refuses everything else."""
        return guarded(self._h.matmul, handle(other))

    def _inplace(js_name):                                   # noqa: N805
        """`x += 1` is an in-place operation too — torch refuses it on a leaf
        with gradients on."""
        def go(self, other):
            return getattr(self, f"{js_name}_")(other)
        return go

    __iadd__ = _inplace("add")
    __isub__ = _inplace("sub")
    __imul__ = _inplace("mul")
    __itruediv__ = _inplace("div")

    del _inplace

    def __neg__(self):
        return wrap(self._h.neg())

    def _spans(self, key):
        """`key` as one `(start, stop)` per axis, in order, covering every axis.

        Only basic indexing — integers, slices with step 1, and one `Ellipsis`.
        An integer becomes a span of length one, which is what makes assignment
        and slicing the same walk. Anything else is refused by name rather than
        approximated, because a *nearly* right region is a wrong picture with no
        error attached to it.
        """
        keys = list(key) if isinstance(key, tuple) else [key]
        if keys.count(Ellipsis) > 1:
            raise IndexError("an index can only have a single ellipsis ('...')")
        if Ellipsis in keys:
            at = keys.index(Ellipsis)
            named = len(keys) - 1
            keys[at:at + 1] = [slice(None)] * (len(self.shape) - named)
        keys += [slice(None)] * (len(self.shape) - len(keys))
        if len(keys) > len(self.shape):
            raise IndexError(
                f"too many indices: this tensor has {len(self.shape)} dimensions "
                f"and {len(keys)} were given")
        spans = []
        for axis, k in enumerate(keys):
            n = self.shape[axis]
            if isinstance(k, slice):
                if k.step not in (None, 1):
                    raise NotImplementedError(
                        "assigning into a strided slice is not here yet — "
                        f"step {k.step} on dimension {axis}.")
                start, stop, _ = k.indices(n)
                spans.append((start, max(start, stop)))
            elif isinstance(k, int):
                at = k + n if k < 0 else k
                if not 0 <= at < n:
                    raise IndexError(
                        f"index {k} is out of bounds for dimension {axis} "
                        f"with size {n}")
                spans.append((at, at + 1))
            else:
                raise NotImplementedError(
                    f"assigning with {type(k).__name__} is not here yet — "
                    "integers, slices and `...` are.")
        return spans

    def __setitem__(self, key, value):
        """`x[0] = 1`, `img[..., y:y + h, x:x + w] = 0`.

        **There was no `__setitem__` at all**, so any assignment stopped with
        `'Tensor' object does not support item assignment` — including
        `borchvision`'s `erase`, which is one line of exactly this and is how
        `RandomErasing` blanks its rectangle. The numpy core has had it all
        along, so the two libraries disagreed about whether a tutorial line works.

        Built out of `sliceScatter` from the inside out: narrow to the region on
        each axis in turn, put the value in at the innermost, then scatter each
        result back into its parent. Doing it as one scatter per axis
        independently is the tempting shape and it is wrong — two axes scattered
        separately write two whole bands rather than their intersection, and on
        a square region the two agree.
        """
        self._refuse_inplace_on_leaf("__setitem__")
        spans = self._spans(key)
        region = self
        for axis, (start, stop) in enumerate(spans):
            region = wrap(region._h.narrow(axis, start, stop - start))
        # **`value` is not always a number.** `borchvision.erase` is called both
        # ways — a scalar fill and a `(C, h, w)` patch — and `float(value)` on the
        # second turns into `only length-1 arrays can be converted to Python
        # scalars`, an error about conversion in code about assignment.
        if not isinstance(value, Tensor) and not isinstance(value, (int, float)):
            import numpy as _numpy
            value = tensor(_numpy.asarray(value, dtype=_numpy.float32))
        if isinstance(value, Tensor):
            # Broadcast to the region: multiplying by zero and adding is the
            # cheapest thing that applies borch.ts's own broadcasting rules rather
            # than keeping a second copy of them here.
            src = wrap(region._h.binary("mul", handle(0.0))).__add__(value)
        else:
            from ._ops import full_like
            src = full_like(region, float(value))

        def put(dst, axis):
            if axis == len(spans):
                return src
            start, stop = spans[axis]
            inner = wrap(dst._h.narrow(axis, start, stop - start))
            return wrap(dst._h.sliceScatter(
                handle(put(inner, axis + 1)), axis, start, stop, 1))

        return self._write_back(put(self, 0))

    def __getitem__(self, key):
        """`x[0]`, `x[1:3]`, `x[:, 1]`. The most common thing torch code does."""
        keys = key if isinstance(key, tuple) else (key,)
        kinds = [isinstance(k, (Tensor, list, tuple)) for k in keys]
        if sum(kinds) > 1 and all(kinds):
            return self._gather_at(keys)
        out, axis = self, 0
        for k in keys:
            if isinstance(k, slice):
                start = 0 if k.start is None else k.start
                stop = out.shape[axis] if k.stop is None else k.stop
                out = wrap(out._h.narrow(axis, start, stop - start))
                axis += 1
            elif k is None:
                # **`x[:, None]` inserts a dimension.** `torch.newaxis` is this
                # `None`. Without it the integer branch below catches it and
                # produces `'<' not supported between instances of 'NoneType'
                # and 'int'`, which has nothing to do with what the indexing was
                # trying to do, so the cause stays hidden.
                out = wrap(out._h.unsqueeze(axis))
                axis += 1
            elif isinstance(k, (Tensor, list, tuple)):
                # `x[[2, 0]]` — selecting by a list of indices. Common in torch
                # code.
                idx = k if isinstance(k, Tensor) else tensor(list(k), int64_name())
                out = wrap(out._h.indexSelect(axis, idx._h))
                axis += 1
            else:
                n = out.shape[axis]
                at = k + n if k < 0 else k
                if not 0 <= at < n:
                    # torch raises `IndexError` here — the type is API too, so
                    # it is matched.
                    raise IndexError(
                        f"index {k} is out of bounds for dimension {axis} "
                        f"with size {n}")
                out = wrap(out._h.select(axis, at))
        return out

    @staticmethod
    def _flat_count(sizes):
        total = 1
        for one in sizes:
            total *= one
        return total

    def _gather_at(self, keys):
        """**Several index tensors at once**, which is a different operation from
        several one at a time.

        torch broadcasts the index tensors against each other and reads **one element
        per position of the broadcast shape**. Applied one axis after another with
        `indexSelect`, each axis instead grows to that index's element count and the
        result is their product: `picture[b, c, y, x]` with the six-dimensional
        indices `roi_align` builds came back `[1, 2, 16, 16]` where torch gives
        `[1, 2, 2, 2, 8, 8]`. **Twenty-one golden cases**, all of them the samplers
        under `roi_align`, `ps_roi_align` and `deform_conv2d`.

        The positions are turned into one offset — the row-major sum `((i₀·n₁ + i₁)·n₂
        + i₂)…` — and read from a flattened view, so the broadcasting is the ordinary
        arithmetic kind and there is no second rule to keep in step. Whatever axes the
        keys do not cover stay behind the offset and come back as they were.
        """
        shape = [int(one) for one in self.shape]
        lead, rest = shape[:len(keys)], shape[len(keys):]
        offset = None
        for size, k in zip(lead, keys):
            idx = k if isinstance(k, Tensor) else tensor(list(k), int64_name())
            offset = idx if offset is None else offset * size + idx
        flat = wrap(self._h.reshape(_js_list(
            [self._flat_count(lead), *rest] if rest else [self._flat_count(lead)])))
        picked = wrap(flat._h.indexSelect(0, offset.reshape(-1).long()._h))
        return wrap(picked._h.reshape(_js_list(
            [int(one) for one in offset.shape] + rest)))

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __bool__(self):
        """**It read element zero and answered**, whatever the size. `if t:` on a
        three-element tensor returned the truth of the first value instead of
        raising, which is torch's loudest refusal turned into a quiet wrong
        answer. The core carries the same three methods; this side had one."""
        if self._h.size != 1:
            raise RuntimeError(
                "Boolean value of Tensor with more than one value is ambiguous")
        return bool(_read(self._h)[0])

    def _scalar(self):
        """The single value, raising the way `int()` and `float()` raise in torch —
        `ValueError`, where `.item()` next door raises `RuntimeError`. The two
        classes are torch's and they really do differ."""
        if self._h.size != 1:
            raise ValueError(
                "only one element tensors can be converted to Python scalars")
        return _read(self._h)[0]

    def __float__(self):
        return float(self._scalar())

    def __int__(self):
        """`int(t)`. **Absent**, so `int(t)` on a one-element tensor raised
        `TypeError` here and answered on the core — a parting no value comparison
        can see, because the call never produces a value."""
        return int(self._scalar())

    def __index__(self):
        """What lets a tensor be an index — `xs[t]`, `range(t)`. torch demands an
        integer dtype **and** a single element, and raises `TypeError` for both."""
        if str(self._h.dtype) not in ("int32", "int64", "uint8", "bool") \
                or self._h.size != 1:
            raise TypeError(
                "only integer tensors of a single element can be converted to an index")
        return int(_read(self._h)[0])

    def __hash__(self):
        return id(self)

    del _op, _rop


class RuntimeError_(RuntimeError):
    """The name is Python's and the wording is torch's."""


class IndexError_(IndexError):
    pass


class LinAlgError(RuntimeError):
    """torch's `linalg.LinAlgError`.

    **The name does work.** Code that may meet a singular matrix usually wraps
    it in `except linalg.LinAlgError`, and raising a plain `RuntimeError` walks
    past that wrapper and kills the program. borch.ts has a class of the same
    name and `translate` joins the two — an exception's type is API as well.
    """


def translate(exc):
    """Move a JavaScript exception to **the type torch would raise.**

    The golden froze **the exception's type name** as part of the answer
    (`RuntimeError|wording=True`). Left alone, a `JsException` reaches Python and
    code catching `except RuntimeError` does not catch it — an exception's type
    is API.

    The wording is not changed. borch.ts already carries torch's own phrasing,
    and it was written that way so a search finds it.
    """
    text = str(exc)
    # The ones that must not be left to the guess below (an `IndexError` when
    # the wording contains "index"). **Refusals have types too** — "not here
    # yet" (`NotImplementedError`) and "the caller is wrong" (`RuntimeError`)
    # say different things, and the golden froze those names as answers.
    for head, cls in (("LinAlgError: ", LinAlgError),
                      ("NotImplementedError: ", NotImplementedError)):
        if text.startswith(head):
            return cls(text[len(head):])
    # **Only the prefix is stripped.** Removing the first `Error: ` with
    # `replace` turned `RuntimeError: shape …` into `Runtimeshape …` and broke
    # the wording — us destroying the very phrasing that was kept so a search
    # would find it.
    for head in ("RuntimeError: ", "IndexError: ", "Error: "):
        if text.startswith(head):
            text = text[len(head):]
            break
    # The Korean spelling used to be checked here as well, back when our own
    # messages were Korean. They are English now, so that branch could never
    # match — a condition that cannot fire reads as one that is guarding
    # something.
    kind = IndexError if "index" in text.lower() else RuntimeError
    return kind(text)


class _Pair:
    """The ones handing back `values` and `indices` together — `sort`, `topk`,
    `median`, `max(dim)`.

    borch.ts returns a plain JavaScript object. Passed through, `.values` is a
    JS proxy with no `.numpy()`, and the failure arrives one step late as
    `AttributeError: numpy`.
    """

    __slots__ = ("values", "indices")

    def __init__(self, obj):
        self.values = wrap(obj.values)
        self.indices = wrap(obj.indices)

    def __iter__(self):
        yield self.values
        yield self.indices

    def __getitem__(self, i):
        return (self.values, self.indices)[i]

    def __getattr__(self, name):
        """**Forwarded to the values side.** torch's `median()` called without a
        dimension hands back a single value while borch.ts always hands back a
        pair — so asking `.numpy()` or `.shape` there is asking about the
        values."""
        return getattr(self.values, name)


def settle(out):
    """Shape what came back into something Python can use.

    **A promise is awaited here.** A few things in borch.ts are asynchronous —
    `unique`, `bincount`, `masked_select` and the rest, whose **result size
    depends on the values**, so the shape is only settled after one read off the
    GPU. Passed through, a `PyodideFuture` ends up loose in Python and the
    failure arrives on the next line as
    `'PyodideFuture' object has no attribute 'detach'`, one step from the cause.

    `run_sync` fills that gap here too — the same machinery as reading a value.
    """
    from pyodide.ffi import JsException

    try:
        if hasattr(out, "then"):
            out = _run_sync(out)
    except JsException as exc:
        raise translate(exc) from None
    if _js.borch.isTensor(out):
        return wrap(out)
    # A `{values, indices}` pair or an array of tensors leaves a proxy loose in
    # Python if it is passed straight through.
    if hasattr(out, "values") and hasattr(out, "indices"):
        return _Pair(out)
    if _js.Array.isArray(out):
        return [wrap(x) if _js.borch.isTensor(x) else x for x in out]
    # **The ones handing back several named slots** — `slogdet`'s
    # `{sign, logabs}`, `qr`'s `{q, r}`, `svd`'s `{u, s, vt}`. Passed through,
    # what is left in Python is a proxy that answers neither indexing nor
    # attribute access.
    if hasattr(out, "constructor") and str(getattr(out, "constructor", "")) and \
            not callable(out) and hasattr(out, "toString"):
        keys = [str(k) for k in _js.Object.keys(out)]
        if keys and all(not k.isdigit() for k in keys):
            return _Fields({k: getattr(out, k) for k in keys},
                           _TORCH_FIELDS.get(tuple(keys)))
    return out


# borch.ts's slot names to **torch's names.**
#
# torch lets these be asked for by name as well — `slogdet(A).logabsdet`,
# `qr(A).Q`, `eigh(A).eigenvalues`. Matching only the positions leaves the
# values right and textbook code stopping at an attribute access. `lstsq`
# already went through that with `.solution`.
#
# **Keyed on the whole set of slots rather than on single names.** `values` is
# the eigenvalues in `eigh` and just the values in `sort` and `topk`, so
# renaming on one name alone renames the wrong things too.
_TORCH_FIELDS = {
    ("sign", "logabs"): {"logabs": "logabsdet"},
    ("q", "r"): {"q": "Q", "r": "R"},
    ("u", "s", "vt"): {"u": "U", "s": "S", "vt": "Vh"},
    # **`torch.svd` is a different function from `torch.linalg.svd`** and its third
    # slot is `V`, the transpose of `Vh`. borch.ts returns `{u, s, v}` from one and
    # `{u, s, vt}` from the other, so the two slot sets are what tells them apart —
    # which is the reason this table is keyed on the whole set and not on names.
    ("u", "s", "v"): {"u": "U", "s": "S", "v": "V"},
    ("values", "vectors"): {"values": "eigenvalues", "vectors": "eigenvectors"},
}


class _Fields:
    """A result with several named slots, reachable by index and by name — the
    way torch does it."""

    __slots__ = ("_d", "_order")

    def __init__(self, d, alias=None):
        # **The positional order stays keyed on the JavaScript names.** Putting
        # the aliases into the order shifts `[0]` and `[1]` — adding names would
        # end up moving the positions.
        self._order = list(d)
        vals = {k: (wrap(v) if _js.borch.isTensor(v) else v) for k, v in d.items()}
        for js_name, torch_name in (alias or {}).items():
            if js_name in vals:
                vals[torch_name] = vals[js_name]
        object.__setattr__(self, "_d", vals)

    def __getattr__(self, name):
        try:
            return self._d[name]
        except KeyError:
            raise AttributeError(name) from None

    def __getitem__(self, i):
        return self._d[self._order[i]] if isinstance(i, int) else self._d[i]

    def __iter__(self):
        for k in self._order:
            yield self._d[k]


def guarded(fn, *args):
    """Call it, and re-raise a JavaScript exception as torch's type."""
    from pyodide.ffi import JsException

    try:
        return settle(fn(*args))
    except JsException as exc:
        raise translate(exc) from None


def wrap(x):
    """A JS tensor or a Python number, either way into our `Tensor`."""
    if isinstance(x, Tensor):
        return x
    # **A Python number's type takes part in the promotion rules.** `int64 + 2`
    # is int64 and `int64 + 2.0` is float32. Making them all float32 scalars
    # collapsed every promotion to float32 — the values stay right and only the
    # dtype name differs, which comparing values cannot see.
    if isinstance(x, bool):
        return Tensor(_ts.Tensor.from_(
            _js.Float32Array.new(_to_js([1.0 if x else 0.0])),
            _js_list([]), _js_options(dtype="bool")))
    if isinstance(x, int):
        return Tensor(_ts.Tensor.from_(
            _js.Float32Array.new(_to_js([float(x)])),
            _js_list([]), _js_options(dtype="int64")))
    if isinstance(x, float):
        return Tensor(_ts.Tensor.full(_js_list([]), x))
    return Tensor(x)


def handle(x):
    """The handle if the other side is one of our tensors; otherwise a scalar
    tensor's handle."""
    return wrap(x)._h


def tensor(data, dtype=None, requires_grad=False):
    """Where `torch.tensor` sits. Takes numpy arrays, nested lists and numbers."""
    from pyodide.ffi import JsException

    arr = _np.asarray(data)
    if dtype is not None:
        # Something that shows as `torch.float32` still crosses to borch.ts as
        # `float32`.
        name = dtype.plain if isinstance(dtype, _DType) else str(dtype)
    elif arr.dtype.kind == "c":
        name = "complex64"
    elif arr.dtype == bool:
        name = "bool"
    elif arr.dtype.kind in "iu":
        name = "int64"
    else:
        name = "float32"
    # **Complex comes in through a different door.** borch.ts's `Tensor.from`
    # refuses a dtype merely labelled `complex64` — the storage is two f32 per
    # slot, so relabelling alone makes the back half somebody else's memory. It
    # is split into real and imaginary parts and joined.
    if name == "complex64":
        if requires_grad:
            # **Refused because it would not be a leaf.** Joined together it
            # becomes **an interior node** carrying `ComplexBackward0`, while a
            # torch tensor with `requires_grad=True` is a leaf. The difference
            # shows only as `.grad` never accumulating — with every value right.
            raise RuntimeError(
                "requires_grad=True cannot be given to a complex64 tensor here — make two "
                "real leaves and join them with `complex(re, im)`.")
        parts = _np.asarray(arr, dtype=_np.complex64)
        pair = [_np.ascontiguousarray(half.ravel(), dtype=_np.float32)
                for half in (parts.real, parts.imag)]
        made = [_ts.Tensor.from_(_js.Float32Array.new(_to_js(half)),
                                 _js_list(parts.shape), _js_options())
                for half in pair]
        return Tensor(_ts.Tensor.complex(made[0], made[1]))
    flat = _js.Float32Array.new(_to_js(arr.ravel().astype(_np.float32)))
    try:
        return Tensor(_ts.Tensor.from_(
            flat, _js_list(arr.shape),
            _js_options(requiresGrad=bool(requires_grad), dtype=name)))
    except JsException as exc:
        # torch also refuses gradients on integers and bools. The type is
        # carried across — code catching `except RuntimeError` must not stop
        # catching it.
        raise translate(exc) from None


# ── absent dtypes are refused by name — **in the core's wording** ────────────
#
# Left alone this produced an `AttributeError` saying borch.ts's tensor has no
# `half`, while the core said `.half()` (float16) is not in the browser subset.
# A learner reads those two as **something each implementation does
# differently.** This is a place where the wording is matched rather than a
# value, and places like it survive cross-checking — because nobody asked.
_ABSENT_DTYPES = {
    "half": "float16", "bfloat16": "bfloat16", "chalf": "complex32",
    "cdouble": "complex128", "byte": "uint8", "char": "int8", "short": "int16",
}


def _absent_dtype(name, shown):
    # The exception **type** has to match too, so the core's `BorchError` is
    # borrowed. In a browser `borch` also lives under `/work`, so importing it
    # late is enough — the same way `_core_repr` does it.
    from borch._base import BorchError
    raise BorchError(
        f"`.{name}()`({shown}) is not in the browser subset.\n"
        "Use real PyTorch on your own machine (`uv add torch`) — this subset is for "
        "practising the syntax, and imitating what is missing teaches the wrong thing.")


def _bind_absent(name, shown):
    def method(self):
        del self
        _absent_dtype(name, shown)

    method.__name__ = name
    return method


for _dname, _shown in _ABSENT_DTYPES.items():
    setattr(Tensor, _dname, _bind_absent(_dname, _shown))
del _dname, _shown


def _deprecated_by_torch(name):
    raise RuntimeError(
        f"`{name}` was removed in torch 1.9 — use `torch.linalg.{name}`. "
        f"(torch: This function was deprecated since version 1.9 and is now removed. "
        f"Please use the `torch.linalg.{name}` function instead.)")


# ── the predicates torch gives as properties ────────────────────────────────
#
# The same answers as the core's except **`is_cpu`, which differs** — the values
# live in a GPU buffer, so it is false. That is this binding's fact, and saying
# true sends code branching on `x.is_cpu` down the wrong path.
_ALWAYS_FALSE = (
    "is_cuda", "is_ipu", "is_maia", "is_meta", "is_mkldnn", "is_mps", "is_mtia",
    "is_nested", "is_quantized", "is_sparse", "is_sparse_csr", "is_vulkan",
    "is_xla", "is_xpu", "is_cpu", "retains_grad",
)

for _pname in _ALWAYS_FALSE:
    setattr(Tensor, _pname, property(lambda self: False))
del _pname

Tensor.is_leaf = property(lambda self: not bool(self._h.gradName))
# `is_neg` and `is_pinned` are the methods — they take parentheses.
Tensor.is_neg = lambda self: False
Tensor.is_pinned = lambda self: False


def _is_coalesced(self):
    del self
    raise RuntimeError(
        "A dense tensor has no coalesce state. "
        "(torch: is_coalesced expected sparse coordinate tensor layout "
        "but got Strided)")


def _borrow_core(name):
    """**Borrows the core's rules.** These hang a Python function on every slot,
    which the GPU cannot do, and two copies of the rules diverge — the same
    arrangement as the seven distributions."""
    def method(self, *args, **kw):
        from borch._tensor import Tensor as _Core
        from ._base import tensor as _t

        core = _Core(self.numpy().copy())
        got = getattr(core, name)(*[
            _Core(a.numpy().copy()) if isinstance(a, Tensor) else a for a in args], **kw)
        return self._write_back(_t(got.data))

    method.__name__ = name
    return method


def _sparse_only(name):
    def method(self, *args, **kw):
        del self, args, kw
        raise NotImplementedError(
            f"`{name}` is for sparse tensors only — not for a dense one. "
            f"(torch: Could not run 'aten::{name}' with arguments from the 'CPU' backend)")

    method.__name__ = name
    return method


Tensor.is_coalesced = _is_coalesced
def _set_(self, source=None):
    """**Swaps the storage** rather than writing values back. The element count
    changes, so `copyFrom` cannot do it and the handle itself is replaced.
    Written as a write-back first, it raised a JavaScript exception wherever the
    counts differed."""
    from ._base import tensor as _t
    import numpy as _n2

    got = _t(_n2.empty(0, dtype=_n2.float32)) if source is None else source
    self._h = handle(got)
    return self


def _resize_(self, *sizes):
    """**The element count can change.** `_write_back` only takes the same size,
    so the handle is replaced here as well — the same reason as `set_`. Written
    as a write-back first, it stopped with a size mismatch wherever it shrank."""
    from borch._tensor import Tensor as _Core
    from ._base import tensor as _t

    core = _Core(self.numpy().copy())
    core.resize_(*sizes)
    self._h = handle(_t(core.data))
    return self


Tensor.set_ = _set_
Tensor.resize_ = _resize_
for _n in ("apply_", "map_", "map2_"):
    setattr(Tensor, _n, _borrow_core(_n))
for _n in ("resize_as_sparse_", "sparse_resize_", "sparse_resize_and_clear_"):
    setattr(Tensor, _n, _sparse_only(_n))
del _n


# ── the ones that look into storage, and transpose's three names ────────────
#
# **`stride`, `dim_order` and `data_ptr` diverge here.** The values are in a GPU
# buffer and no views are made, so it is always contiguous — a transpose does
# not change the strides. In the core numpy's views change them. Rather than
# lie, this hands back **the contiguous strides.**
def _row_major_stride(shape):
    out, step = [], 1
    for n in reversed(shape):
        out.append(step)
        step *= int(n)
    return tuple(reversed(out))


def _stride(self, dim=None):
    got = _row_major_stride(tuple(self.shape))
    return got if dim is None else got[dim]


Tensor.stride = _stride
Tensor.dim_order = lambda self: tuple(range(self.ndim))
Tensor.element_size = lambda self: 4          # the storage is float32 throughout
Tensor.nelement = lambda self: self.numel()
Tensor.ndimension = lambda self: self.ndim
Tensor.itemsize = property(lambda self: 4)
Tensor.nbytes = property(lambda self: 4 * self.numel())
Tensor.output_nr = property(lambda self: 0)
Tensor.volatile = property(lambda self: False)
Tensor.name = property(lambda self: None)
Tensor.grad_dtype = property(lambda self: self.dtype)
Tensor.layout = property(lambda self: _CoreLayout())


def _CoreLayout():                                          # noqa: N802
    from borch._tensor import _Layout
    return _Layout()


def _matrix_transpose(self):
    if self.ndim < 2:
        raise RuntimeError(
            "`.mT` is only on 2-D and above. "
            "(torch: tensor.mT is only supported on matrices or batches of matrices)")
    return self.transpose(-2, -1)


def _hermitian(self):
    if self.ndim != 2:
        raise RuntimeError(
            f"`.H` is only on matrices (2-D) — got {self.ndim}-D. "
            "(torch: tensor.H is only supported on matrices (2-D tensors))")
    return self.transpose(0, 1).conj()


Tensor.mT = property(_matrix_transpose)
Tensor.mH = property(lambda self: _matrix_transpose(self).conj())
Tensor.H = property(_hermitian)
Tensor.grad_fn = property(lambda self: _grad_fn_of(self))


def _grad_fn_of(t):
    name = t._h.gradName
    if not name:
        return None
    from borch._tensor import _GradFn
    got = _GradFn(str(name))
    got.__class__ = type(str(name), (_GradFn,), {"__slots__": ()})
    return got


# ── `new_*` and the names that take their shape from another tensor ─────────
def _new_like(name, fill):
    def method(self, *size, dtype=None, requires_grad=False):
        from ._base import tensor as _t
        shape = tuple(size[0]) if len(size) == 1 and isinstance(size[0], (tuple, list)) \
            else tuple(int(s) for s in size)
        want = str(dtype).replace("torch.", "") if dtype is not None else self.dtype.plain
        got = _t(fill(shape))
        return (got if want == "float32" else got.to(want)) if not requires_grad \
            else _t(fill(shape), requires_grad=True)

    method.__name__ = name
    return method


Tensor.new_zeros = _new_like("new_zeros", lambda s: _np.zeros(s, dtype=_np.float32))
Tensor.new_ones = _new_like("new_ones", lambda s: _np.ones(s, dtype=_np.float32))
Tensor.new_empty = _new_like("new_empty", lambda s: _np.zeros(s, dtype=_np.float32))
Tensor.reshape_as = lambda self, other: self.reshape(*[int(v) for v in other.shape])
Tensor.view_as = lambda self, other: self.reshape(*[int(v) for v in other.shape])
Tensor.resize_as = lambda self, other: self.reshape(*[int(v) for v in other.shape])


def _new_full(self, size, fill_value, dtype=None, requires_grad=False):
    from ._base import tensor as _t
    del dtype, requires_grad
    return _t(_np.full(tuple(size), fill_value, dtype=_np.float32))


def _new_tensor(self, data, dtype=None, requires_grad=False):
    from ._base import tensor as _t
    del dtype, requires_grad
    return _t(_np.array(data, dtype=_np.float32))


def _sum_to_size(self, *size):
    from borch._tensor import Tensor as _Core, _unbroadcast
    from ._base import tensor as _t
    shape = tuple(size[0]) if len(size) == 1 and isinstance(size[0], (tuple, list)) \
        else tuple(int(s) for s in size)
    return _t(_unbroadcast(self.numpy(), shape))


def _retain_grad(self):
    """**Stops on a leaf**, as the core does. borch.ts already keeps gradients
    for a derived tensor, so this only raises the flag."""
    if not self._h.requiresGrad:
        raise RuntimeError(
            "`retain_grad()` cannot be used on a tensor that does not take gradients. "
            "(torch: can't retain_grad on Tensor that has requires_grad=False)")
    return None


Tensor.new_full = _new_full
Tensor.new_tensor = _new_tensor
Tensor.new = lambda self, *size: (self.new_empty(*size) if size
                                  else _new_like("new", lambda s: _np.zeros(0, dtype=_np.float32))(self))
Tensor.sum_to_size = _sum_to_size
Tensor.retain_grad = _retain_grad


def _gone(name, instead):
    def method(self, *a, **k):
        del self, a, k
        raise RuntimeError(
            f"`{name}` was removed from torch — use `torch.{instead}`. "
            "(torch: This function was deprecated since version 1.9 and is now removed.)")
    method.__name__ = name
    return method


for _n, _i in (("eig", "linalg.eig"), ("symeig", "linalg.eigh")):
    setattr(Tensor, _n, _gone(_n, _i))
del _n, _i


# ── sparse, storage and quantisation — **the name exists and does not fit
# this tensor** ─────────────────────────────────────────────────────────────
#
# Fixed here for the core's reason. Left alone it produces
# `'Tensor' object has no attribute 'coalesce'`, which is **indistinguishable
# from a typo.** torch says it expected a sparse layout and got Strided —
# meaning the name exists and does not fit this tensor.
_SPARSE_ACCESSOR = {
    "coalesce": "coordinate tensor layout",
    "indices": "coordinate tensor layout",
    "values": "tensor layout",
    "crow_indices": "row compressed tensor layout",
    "col_indices": "row compressed tensor layout",
    "ccol_indices": "column compressed tensor layout",
    "row_indices": "column compressed tensor layout",
}


def _needs_sparse(name, layout):
    def method(self, *a, **k):
        del self, a, k
        raise RuntimeError(
            f"`{name}` is for sparse tensors only — a dense tensor does not have it. "
            f"(torch: {name} expected sparse {layout} but got Strided)")

    method.__name__ = name
    return method


def _absent_here(name, what):
    """**One copy of the sentence.** Written out separately here, the same
    sentence existed twice in this file — fix one and the other stays, and since
    what the golden looks for is **a fragment of the wording** rather than a
    value, cross-checking does not catch the divergence."""
    def method(self, *a, **k):
        del self, a, k
        _absent_dtype(name, what)

    method.__name__ = name
    return method


for _n, _layout in _SPARSE_ACCESSOR.items():
    setattr(Tensor, _n, _needs_sparse(_n, _layout))
for _n in ("to_sparse", "to_sparse_coo", "to_sparse_csr", "to_sparse_csc",
           "to_sparse_bsr", "to_sparse_bsc", "sparse_mask"):
    setattr(Tensor, _n, _absent_here(_n, "sparse tensors"))
for _n in ("storage", "storage_type", "untyped_storage"):
    setattr(Tensor, _n, _absent_here(_n, "the storage object — use `.numpy()`"))
for _n in ("int_repr", "q_scale", "q_zero_point", "qscheme"):
    setattr(Tensor, _n, _absent_here(_n, "quantisation"))
for _n in ("cuda", "ipu", "mtia", "xpu"):
    setattr(Tensor, _n, _absent_here(_n, "that device"))
for _n in ("pin_memory", "record_stream"):
    setattr(Tensor, _n, _absent_here(_n, "pinned memory and streams"))
del _n, _layout


def _is_set_to(self, other):
    """**Whether two tensors point at the same storage.** Here that means the
    same handle: no views are made, so a transpose or a reshape is always a new
    buffer. In the core numpy's views make it true in places, and that
    divergence is a fact about views rather than about this predicate."""
    return isinstance(other, Tensor) and self._h is other._h


Tensor.is_set_to = _is_set_to
Tensor.is_shared = lambda self: False
