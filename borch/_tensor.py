"""A piece of borch, split out. __init__ gathers the public names."""

import math as _math

import numpy as _np

from ._base import (
    Size, _DEFAULT_DTYPE, _NP_TO_DTYPE, _TYPE_NAMES, _float_in, _like_torch,
    _needs_float,
    _no_complex128, _np, _refuses_bool, _tensor_repr, _unsupported,
    device as _device, dtype, float32,
)

# ---------------------------------------------------------------- Tensor

def _conj(x):
    """**The conjugate that attaches to a holomorphic function's backward.**

    The complex gradient convention is `z.grad = ∂L/∂re + i·∂L/∂im` (measured),
    so a holomorphic `f`'s backward is `conj(f'(z))·g`. Over the reals the
    conjugate is the identity, so **feeding it reals alone cannot tell whether
    this place is there** — which makes it safe to leave in real code, and
    leaving it out flips the sign for complex numbers only.
    """
    return _np.conj(x) if _np.asarray(x).dtype.kind == "c" else x


def _keep(out, source, dim, keepdim):
    """Revive a folded axis at size 1. Used by the functions numpy has no
    `keepdims` for.

    **`argmax` and `argmin` are such places** — numpy does not take the argument,
    so here is the only place to revive the axis. Without it an axis is gone and
    the broadcasting **happens to fit**, and it runs all the way through with
    only the values wrong.
    """
    if not keepdim or dim is None:
        return out
    shape = list(_np.shape(source.data))
    shape[dim if dim >= 0 else dim + len(shape)] = 1
    return _np.reshape(out, shape)


def _unbroadcast(grad, shape):
    """Undo the axes broadcasting stretched. A required step of
    backpropagation."""
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for i, n in enumerate(shape):
        if n == 1 and grad.shape[i] != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad.reshape(shape)


# torch's dtype promotion differs from numpy's — it splits by **category** first
# and promotes only within that category.
#
#   categories:  bool(0) < integer(1) < float(2) < complex(3)
#   the rule:    take the highest category present, and within it the largest.
#                A lower category **does not pull a higher one up.**
#
# So float32 + int64 is float32 in torch (numpy promotes it to float64). Left to
# numpy here, the learner learns the wrong rule.
#
# **Complex sits one step higher** (measured: `complex64 + int64` is complex64).
# And the precision on the real side **carries across** to the complex one —
# `complex64 + float64` is **complex128** (measured). The category rule alone does
# not produce that, so it is written separately.

_CATEGORY = {"b": 0, "i": 1, "u": 1, "f": 2, "c": 3}
_RANK = {_np.dtype("bool"): 0, _np.dtype("int64"): 10,
         _np.dtype("float32"): 20, _np.dtype("float64"): 21,
         _np.dtype("complex64"): 30, _np.dtype("complex128"): 31}
_DEFAULT_BY_CATEGORY = {0: _np.dtype("bool"), 1: _np.dtype("int64"),
                        2: _np.dtype("float32"), 3: _np.dtype("complex64")}
# The table for a real's precision carrying into complex. One double-precision
# real makes a double-precision complex.
_WIDENS_COMPLEX = {_np.dtype("float64"): _np.dtype("complex128")}


def _category(dt):
    return _CATEGORY.get(_np.dtype(dt).kind, 2)


def result_type(a, b):
    """The result type of two tensor dtypes. torch.result_type's rule."""
    da, db = _np.dtype(a), _np.dtype(b)
    cat = max(_category(da), _category(db))
    same = [d for d in (da, db) if _category(d) == cat]
    out = max(same, key=lambda d: _RANK.get(d, 0))
    if cat == 3:
        # **The real side's precision carries across.** `complex64 + float64` is
        # complex128.
        for d in (da, db):
            wide = _WIDENS_COMPLEX.get(d)
            if wide is not None and _RANK[wide] > _RANK[out]:
                out = wide
    return out


def _scalar_category(value):
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return 1
    if isinstance(value, complex):
        return 3
    return 2


def _no_bool_subtract(dtype, other):
    """torch does not allow `-` on booleans. It points at `^` or `~`."""
    other_dtype = other.data.dtype if isinstance(other, Tensor) else _np.asarray(other).dtype
    if _np.dtype(dtype).kind == "b" or other_dtype.kind == "b":
        raise RuntimeError(_like_torch(
            "Subtraction (`-`) is not available on a bool tensor. "
            "Use `^` (exclusive or) or `~` (not).",
            "Subtraction, the `-` operator, with a bool tensor is not supported. "
            "If you are trying to invert a mask use the `~` or `logical_not()` operator instead."))


def _promote(data, scalar):
    """The dtype when mixing with a Python scalar.

    A scalar is weaker than a tensor — at or below the tensor's category it takes
    the tensor's dtype, and only above it does it rise to that category's default
    type. Which is why an int tensor plus a Python float is **float32** rather
    than float64.
    """
    tcat = _category(data.dtype)
    scat = _scalar_category(scalar)
    return data.dtype if scat <= tcat else _DEFAULT_BY_CATEGORY[scat]


class _GradMode:
    """`no_grad`'s switch **held in one object.** It must not simply be a module
    global.

    Splitting the file puts `no_grad` and `_make` in different modules, and a
    global name is created separately per module. `no_grad` sets its own module's
    name to False and `_make` goes on reading the old value — **`no_grad` stops
    working, with no exception and no warning.** It actually happened while
    splitting, and `test_diff` caught it rather than the golden.

    Importing one object means every module sees the same thing.
    """

    enabled = True


_grad_mode = _GradMode()


class _DataDescriptor:
    """`t.data` reads as numpy and accepts **tensors only** on write.

    Because torch refuses `p.data = ndarray`. Accepting it here means code that
    ran in the browser breaks on the user's own machine — being more permissive
    is still diverging.
    """

    def __get__(self, obj, owner=None):
        return obj._array if obj is not None else self

    def __set__(self, obj, value):
        if isinstance(value, Tensor):
            obj._array = value._array
            return
        if isinstance(value, _np.ndarray):
            raise TypeError(_like_torch(
                "`.data` takes a tensor. Wrap it with `torch.tensor(...)`.",
                "Variable data has to be a tensor, but got numpy.ndarray"))
        raise TypeError(_like_torch(
            f"`.data` takes a tensor (got {type(value).__name__}).",
            f"Variable data has to be a tensor, but got {type(value).__name__}"))


class Tensor:
    data = _DataDescriptor()

    def __init__(self, data, requires_grad=False, _parents=(), _backward=None):
        self._array = data if isinstance(data, _np.ndarray) else _np.asarray(data)
        # **Double precision is blocked here.** No `float64` means no
        # `complex128` either.
        #
        # This one line is the throat — the only path by which promotion produces
        # it is `complex64 + float64`, and that result passes through here to
        # become a tensor. Blocked place by place, every new operation forgets
        # one.
        if self._array.dtype == _np.complex128:
            _no_complex128("This tensor's dtype")
        # no_grad only stops **the result of an operation** from carrying a
        # graph; it does not turn off requires_grad on a leaf made directly.
        # torch is the same — turning it off here would quietly drop a parameter
        # made inside a no_grad block from training.
        self.requires_grad = bool(requires_grad)
        self.grad = None
        self._parents = _parents
        self._backward = _backward
        self._freed = False        # one backward releases the graph (as torch does)
        self._op = None            # for showing grad_fn — which operation produced it

        if self.requires_grad and self.data.dtype.kind not in "fc":
            raise RuntimeError(
                "Gradients do not flow through integer tensors. Differentiation is defined "
                "on floating point only — convert with `.float()`."
            )

    # ---- the basics

    @property
    def shape(self):
        return Size(self.data.shape)

    @property
    def dtype(self):
        return _NP_TO_DTYPE.get(self.data.dtype, float32)

    @property
    def ndim(self):
        return self.data.ndim

    def size(self, dim=None):
        return self.shape if dim is None else self.data.shape[dim]

    def dim(self):
        return self.data.ndim

    def numel(self):
        return int(self.data.size)

    def item(self):
        if self.data.size != 1:
            raise RuntimeError(_like_torch(
                f"A tensor with {self.data.size} values cannot become a single number. "
                "Use `.tolist()` or index into it.",
                f"a Tensor with {self.data.size} elements cannot be converted to Scalar"))
        return self.data.reshape(-1)[0].item()

    def tolist(self):
        return self.data.tolist()

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return _tensor_repr(self)

    __str__ = __repr__

    def __iter__(self):
        for i in range(len(self.data)):
            yield self[i]

    def __bool__(self):
        """**The class of the exception was numpy's, not torch's.**

        `bool(t)` on more than one element left `numpy`'s `ValueError` to escape —
        *the truth value of an array … is ambiguous* — where torch raises
        `RuntimeError`. Nothing about the values differed, so no comparison here could
        see it; what differs is which `except` clause catches, and `if t:` inside a
        `try` is ordinary code.
        """
        if self.data.size != 1:
            raise RuntimeError(_like_torch(
                f"A tensor with {self.data.size} values has no single truth value. "
                "Use `.any()` or `.all()`.",
                "Boolean value of Tensor with more than one value is ambiguous"))
        return bool(self.data.reshape(-1)[0])

    def __float__(self):
        return float(self._scalar())

    def __int__(self):
        """`int(t)`. **It was missing entirely**, so `int(t)` raised `TypeError` on a
        tensor holding a single number, where torch answers.

        Truncates toward zero, as `int()` does everywhere in Python and as torch does.
        """
        return int(self._scalar())

    def _scalar(self):
        """The single value, raising **the way `int()` and `float()` raise in torch.**

        `.item()` raises `RuntimeError` and that is torch's class for `.item()`. But
        torch's `int()` and `float()` raise `ValueError` — *only one element tensors
        can be converted to Python scalars* — so passing `.item()`'s exception straight
        through was right about the message and wrong about the class, in the opposite
        direction from `__bool__` on the line above. The two were crossed.
        """
        if self.data.size != 1:
            raise ValueError(_like_torch(
                f"A tensor with {self.data.size} values cannot become one number. "
                "Use `.tolist()` or index into it first.",
                "only one element tensors can be converted to Python scalars"))
        return self.data.reshape(-1)[0]

    def __index__(self):
        """What lets a tensor be used **as an index** — `xs[t]`, `range(t)`, `"ab" * t`.

        Python demands an exact integer here and offers no coercion: a float tensor has
        to raise, and torch raises `TypeError` for one. Without this method the failure
        was an `AttributeError` from somewhere inside the interpreter, which names
        nothing a caller can act on.

        **Both halves of torch's condition raise `TypeError` — integer *and* one
        element.** The first version checked only the dtype and let `.item()` raise for
        the size, so a three-element integer tensor came back `RuntimeError` where torch
        says `TypeError`. It was missed because the grid probing this file had `many`
        only as floats, and a float never reaches the second check: **the instrument's
        list of shapes decided what the instrument could see**, which is the same fault
        as a parser whose correctness was a property of its input.

        Adding that shape then showed the size check is worth more than a class name.
        Take it off this line while the body reads `reshape(-1)[0]` and `xs[t]` with a
        three-element tensor **silently indexes by the first element** — measured, not
        supposed. The `.item()` this replaced had been covering for the missing
        condition, so removing one and not adding the other turns a loud wrong class
        into a quiet wrong answer.
        """
        if self.data.dtype.kind not in "iub" or self.data.size != 1:
            raise TypeError(_like_torch(
                "only an integer tensor holding one value can be used as an index.",
                "only integer tensors of a single element can be converted to an index"))
        return int(self.data.reshape(-1)[0])

    def __array__(self, dtype=None, copy=None):
        """**How numpy is supposed to ask.** Without it, numpy falls back to guessing
        with `len()` and `__getitem__`, and the guess loses axes.

        `np.asarray(t)` on a `(0, 3)` tensor came back `(0,)`; on `(2, 0, 4)` it came
        back `(2, 0)`. The sequence walk descends by indexing, **a zero-length axis has
        nothing to descend into**, and every axis past the first zero is simply not
        discovered. `t.shape` was right the whole time; only the conversion was wrong,
        and it was wrong silently, producing an array of the correct dtype and the
        wrong rank.

        That is not a hypothetical shape. An augmentation whose random draw selects no
        rows produces exactly it, and the value then travels as a correctly-typed empty
        array into arithmetic that reports something else as the fault.

        `copy` and `dtype` are numpy 2's keywords. Accepting them is not optional —
        numpy calls with `copy=False` and warns loudly at anything that cannot take it.
        """
        out = self.data if dtype is None else self.data.astype(dtype)
        if copy is False and out is not self.data:
            raise ValueError("a copy is needed for this dtype and copy=False was asked")
        return _np.array(out, copy=True) if copy else out

    def __hash__(self):
        return id(self)

    # ---- the graph

    def _make(self, data, parents, backward, op=None):
        needs = _grad_mode.enabled and any(p.requires_grad for p in parents)
        out = Tensor(data, requires_grad=False, _parents=parents if needs else (),
                     _backward=backward if needs else None)
        out.requires_grad = needs
        out._op = op if needs else None
        return out

    def backward(self, gradient=None, retain_graph=False):
        if not self.requires_grad:
            raise RuntimeError(_like_torch(
                "backward() cannot be called on a tensor that does not require grad.",
                "element 0 of tensors does not require grad and does not have a grad_fn"))
        if self._freed:
            raise RuntimeError(_like_torch(
                "This graph has already been walked by backward(). Going back once releases "
                "it — recompute, or use `backward(retain_graph=True)`.",
                "Trying to backward through the graph a second time"))
        if gradient is None:
            if self.data.size != 1:
                raise RuntimeError(_like_torch(
                    "A tensor with more than one value needs a gradient. "
                    "Usually the loss is reduced to a scalar first.",
                    "grad can be implicitly created only for scalar outputs"))
            # **A loss has to be real.** torch stops right there (measured).
            #
            # And this one line holds up the whole complex gradient convention —
            # `z.grad = ∂L/∂re + i·∂L/∂im` is well defined only because the loss
            # is always real. Accepting a complex loss would mean settling the
            # other half of Wirtinger, and that is a place left unsettled.
            if self.data.dtype.kind == "c":
                raise RuntimeError(_like_torch(
                    "backward() cannot be called on a complex loss — make it real with "
                    "`.real` or `.abs()` first.",
                    "grad can be implicitly created only for real scalar outputs "
                    "but got torch.complex64"))
            gradient = _np.ones_like(self.data)

        seed = _np.asarray(gradient, dtype=self.data.dtype)
        if seed.shape != self.data.shape:
            # **The shape is checked here.** Unchecked, numpy attempts to
            # broadcast later, and if it fits a quietly wrong gradient comes out
            # and if it does not a `ValueError` appears a long way from the
            # cause. torch stops here with a `RuntimeError` — measured.
            raise RuntimeError(_like_torch(
                f"The gradient shape {tuple(seed.shape)} differs from the value shape "
                f"{tuple(self.data.shape)}.",
                f"Mismatch in shape: grad_output[0] has a shape of "
                f"torch.Size({list(seed.shape)}) and output[0] has a shape of "
                f"torch.Size({list(self.data.shape)})."))

        # A topological sort — back to front, each node once
        order, seen = [], set()

        def visit(t):
            if id(t) in seen:
                return
            seen.add(id(t))
            for p in t._parents:
                visit(p)
            order.append(t)

        visit(self)

        grads = {id(self): seed}
        for t in reversed(order):
            g = grads.get(id(t))
            if g is None:
                continue
            if t._backward is None:                 # a leaf — accumulate here
                if t.requires_grad:
                    t.grad = Tensor(g) if t.grad is None else Tensor(t.grad.data + g)
                continue
            # **Accumulate on a derived tensor that called `retain_grad()`
            # too.** Otherwise that name only imitates the refusal and does
            # nothing — it gets as far as stopping at a leaf and never does the
            # thing it was for.
            if getattr(t, "_retain", False):
                t.grad = Tensor(g) if t.grad is None else Tensor(t.grad.data + g)
            for parent, pg in zip(t._parents, t._backward(g)):
                if pg is None:
                    continue
                # A leaf's .grad is filled by the branch above only. Filling it
                # here as well accumulates twice.
                pg = _unbroadcast(_np.asarray(pg), parent.data.shape)
                grads[id(parent)] = pg if id(parent) not in grads else grads[id(parent)] + pg

        if not retain_graph:
            for t in order:
                if t._backward is not None:
                    t._freed = True

    def detach(self):
        return Tensor(self.data)

    def clone(self):
        return self._make(self.data.copy(), (self,), lambda g: (g,))

    def numpy(self):
        return self.data

    # ---- dtype conversion

    def _cast(self, target):
        """A conversion between floating point dtypes **carries the graph
        through.** That is what torch does.

        It used to be `Tensor(..., self.requires_grad)`, and then the result says
        `requires_grad=True` while having no parents. `backward()` runs without
        an exception and only the original tensor's `.grad` stays `None` — **no
        exception and no warning.** The shape this repository dislikes most, and
        `x.float()` is common in tutorial code, so it becomes a place where
        training quietly does not happen.

        A conversion to an integer or boolean dtype does not come here. torch
        cuts the gradient there too.
        """
        out = self.data.astype(target)
        return self._make(out, (self,), lambda g: (g.astype(self.data.dtype),), "ToCopyBackward0")

    def float(self):
        return self._cast(_np.float32)

    def long(self):
        return Tensor(self.data.astype(_np.int64))

    def int(self):
        """**There is no int32 — so it refuses.**

        It handed back int64 for a long time. The values are plausible and code
        looking at `x.int().dtype == torch.int32` diverges on the user's own
        machine, and the cause surfaces far past this line rather than at it.
        torch's `.int()` is int32 (measured) — with no such storage here,
        stopping beats handing over a different cell instead.
        """
        _unsupported("`.int()`(int32)")

    def bool(self):
        return Tensor(self.data.astype(_np.bool_))

    def double(self):
        return self._cast(_np.float64)

    def type_as(self, other):
        """Match `other`'s dtype. **Not an absent feature but an unwritten one** —
        `type()` existed and this did not, so it stopped with an
        `AttributeError`."""
        return self.type(other.dtype if isinstance(other, Tensor)
                         else _np.asarray(other).dtype)

    def cfloat(self):
        """complex64. This storage exists — unlike `cdouble` and `chalf`."""
        return Tensor(self.data.astype(_np.complex64))


    def type(self, dt):
        target = dt.np if isinstance(dt, dtype) else dt
        if _np.dtype(target).kind != "f":
            return Tensor(self.data.astype(target))
        return self._cast(target)

    def to(self, *args, **kwargs):
        """Takes both a device and **a dtype.** torch's `to` holds the two under
        one name.

        **It quietly discarded the dtype for a long time.** It looked only at the
        device string, ignored the rest and handed back `self`, so
        `x.to(torch.float32)` did nothing — the original dtype, with no exception
        and no warning. That form is common in textbook code (it sits next to
        `x.to(device)`), and on an integer tensor the division that follows
        **quietly becomes integer division.** It surfaced while attaching
        `dtype=` to the reductions — that side calls this function and the dtype
        did not change.
        """
        target = None
        for a in list(args) + list(kwargs.values()):
            # **A `device` object is accepted too.** `x.to(device)` is the
            # tutorial's form, and that `device` is a `torch.device(...)` rather
            # than a string. Looking at strings alone, that line quietly does
            # nothing — and read as a dtype argument into `target` it makes
            # `numpy` stop with "not a dtype" instead.
            if isinstance(a, _device):
                if a.type != "cpu":
                    _unsupported(f"device '{a}'")
                continue
            if isinstance(a, str):
                if a != "cpu":
                    _unsupported(f"device '{a}'")
                continue
            if isinstance(a, Tensor):
                target = a.data.dtype
            elif a is not None and not isinstance(a, bool):
                target = a
        return self if target is None else self.type(target)

    def cpu(self):
        return self

    # ---- arithmetic

    def _binary(self, other, forward, back_self, back_other, op=None):
        if isinstance(other, Tensor):
            target = result_type(self.data.dtype, other.data.dtype)
            o = other if other.data.dtype == target else Tensor(other.data.astype(target))
            mine = self.data if self.data.dtype == target else self.data.astype(target)
        else:
            # The Python scalar is pulled to the tensor's dtype before
            # computing. Left to numpy, int64 + float32 rises to float64, and
            # torch gives float32.
            target = _promote(self.data, other)
            o = Tensor(_np.asarray(other, dtype=target))
            mine = self.data.astype(target) if self.data.dtype != target else self.data
        try:
            out = forward(mine, o.data)
        except ValueError:
            a, b = mine.shape, o.data.shape
            bad = next((i for i in range(1, min(len(a), len(b)) + 1)
                        if a[-i] != b[-i] and a[-i] != 1 and b[-i] != 1), 1)
            raise RuntimeError(_like_torch(
                f"Shapes {tuple(a)} and {tuple(b)} do not broadcast — lined up from the "
                "right, each pair must match or one of them must be 1.",
                f"The size of tensor a ({a[-bad]}) must match the size of tensor b "
                f"({b[-bad]}) at non-singleton dimension {len(a) - bad}")) from None
        return self._make(out, (self, o), lambda g: (back_self(g, mine, o.data),
                                                     back_other(g, mine, o.data)), op)

    def __add__(self, o):
        return self._binary(o, _np.add, lambda g, a, b: g, lambda g, a, b: g, "AddBackward0")

    __radd__ = __add__

    def __sub__(self, o):
        _no_bool_subtract(self.data.dtype, o)
        return self._binary(o, _np.subtract, lambda g, a, b: g, lambda g, a, b: -g, "SubBackward0")

    def __rsub__(self, o):
        _no_bool_subtract(self.data.dtype, o)
        return Tensor(_np.asarray(o, dtype=self.data.dtype)).__sub__(self)

    def __mul__(self, o):
        """Multiplication. **On complex numbers the local derivative takes a
        conjugate.**

        The convention is `z.grad = ∂L/∂re + i·∂L/∂im`, so a holomorphic `f`'s
        backward is `conj(f'(z))·g`. `d(ab)/da = b`, hence `conj(b)·g` — over the
        reals the conjugate is the identity and the formula is the same, and
        **they diverge on complex numbers only.** Real input never shows this
        place.
        """
        return self._binary(o, _np.multiply,
                            lambda g, a, b: g * _conj(b),
                            lambda g, a, b: g * _conj(a), "MulBackward0")

    __rmul__ = __mul__

    def __truediv__(self, o):
        # torch's division gives the default floating point dtype (float32) even
        # between integers or booleans. Left to numpy, int64/int64 becomes
        # float64.
        def div(a, b):
            out = _np.divide(a, b)
            return out.astype(_DEFAULT_DTYPE) if a.dtype.kind not in "fc" else out
        # The same place as multiplication — the conjugate attaches to
        # `d(a/b)/da = 1/b` and `d(a/b)/db = −a/b²`.
        return self._binary(o, div, lambda g, a, b: g / _conj(b),
                            lambda g, a, b: -g * _conj(a) / _conj(b * b),
                            "DivBackward0")

    def __rtruediv__(self, o):
        return Tensor(_np.asarray(o, dtype=self.data.dtype)).__truediv__(self)

    def __pow__(self, p):
        if isinstance(p, Tensor):
            _unsupported("a tensor exponent")
        return self._make(self.data ** p, (self,), lambda g: (g * p * self.data ** (p - 1),),
                          "PowBackward0")

    def __neg__(self):
        return self._make(-self.data, (self,), lambda g: (-g,), "NegBackward0")

    def __mod__(self, o):
        """The remainder. **The gradient flows straight through to the
        dividend** — `a % b` has gradient 1 with respect to `a` (except where the
        step jumps). Towards the divisor it is `-floor(a/b)`."""
        od = o.data if isinstance(o, Tensor) else o
        parents = (self, o) if isinstance(o, Tensor) else (self,)

        def back(g):
            g = _np.asarray(g)
            if isinstance(o, Tensor):
                return (g, -g * _np.floor_divide(self.data, od))
            return (g,)

        return self._make(_np.mod(self.data, od), parents, back, "RemainderBackward0")

    def __floordiv__(self, o):
        return Tensor(_np.floor_divide(self.data, o.data if isinstance(o, Tensor) else o))

    def __matmul__(self, o):
        o = o if isinstance(o, Tensor) else Tensor(o)
        if self.data.ndim >= 2 and o.data.ndim >= 2 and self.data.shape[-1] != o.data.shape[-2]:
            a = "x".join(str(n) for n in self.data.shape[-2:])
            b = "x".join(str(n) for n in o.data.shape[-2:])
            raise RuntimeError(_like_torch(
                f"The matmul shapes do not line up ({a} @ {b}) — the columns on the left "
                f"({self.data.shape[-1]}) must match the rows on the right "
                f"({o.data.shape[-2]}).",
                f"mat1 and mat2 shapes cannot be multiplied ({a} and {b})"))
        def back(g):
            """**1-D borrows an axis.**

            numpy stretches a leading 1-D to `(1, n)` and a trailing 1-D to
            `(n, 1)`, multiplies, and **removes the borrowed axis from the
            result.** The backward has to put that axis back for the transpose to
            hold — without it `swapaxes(v, -1, -2)` stops outright on 1-D
            (`axis -2 is out of bounds`). Measured between 2-D operands alone the
            place never shows, and adding `mv` stepped on it first.
            """
            g = _np.asarray(g)
            a, b = self.data, o.data
            aa = a.reshape((1,) + a.shape) if a.ndim == 1 else a
            bb = b.reshape(b.shape + (1,)) if b.ndim == 1 else b
            lead = _np.broadcast_shapes(aa.shape[:-2], bb.shape[:-2])
            gg = g.reshape(lead + (aa.shape[-2], bb.shape[-1]))
            da = gg @ _np.swapaxes(bb, -1, -2)
            db = _np.swapaxes(aa, -1, -2) @ gg
            # Where the batch was broadcast on one side only the sizes do not
            # match — there it is left alone.
            return (da.reshape(a.shape) if da.size == a.size else da,
                    db.reshape(b.shape) if db.size == b.size else db)

        return self._make(
            self.data @ o.data, (self, o), back,
            "MmBackward0" if self.data.ndim == 2 else "BmmBackward0",
        )

    def matmul(self, o):
        return self.__matmul__(o)

    # In-place updates — allowed inside no_grad only (real torch's rule too)
    def _inplace(self, fn, other):
        if self.requires_grad and _grad_mode.enabled:
            raise RuntimeError(
                "A tensor that requires grad cannot be changed in place. "
                "Do it inside `with torch.no_grad():`."
            )
        o = other.data if isinstance(other, Tensor) else other
        self._array = fn(self._array, o).astype(self._array.dtype)
        return self

    def __iadd__(self, o):
        return self._inplace(_np.add, o)

    def __isub__(self, o):
        return self._inplace(_np.subtract, o)

    def __imul__(self, o):
        return self._inplace(_np.multiply, o)

    # ---- comparison (no gradient)

    def _cmp(self, o, fn):
        return Tensor(fn(self.data, o.data if isinstance(o, Tensor) else o))

    def __gt__(self, o): return self._cmp(o, _np.greater)
    def __ge__(self, o): return self._cmp(o, _np.greater_equal)
    def __lt__(self, o): return self._cmp(o, _np.less)
    def __le__(self, o): return self._cmp(o, _np.less_equal)
    def __eq__(self, o): return self._cmp(o, _np.equal)
    def __ne__(self, o): return self._cmp(o, _np.not_equal)

    def __and__(self, o): return self._cmp(o, _np.logical_and)
    def __or__(self, o): return self._cmp(o, _np.logical_or)

    def all(self, dim=None, keepdim=False):
        """Are they all true. **It takes an axis and `keepdim`** — without them
        it is quietly wrong.

        With no axis, `x.all(dim=1)` falls through to a whole-tensor reduction
        and gives a scalar, and the broadcasting afterwards **happens to fit**,
        so it runs all the way through with only the values wrong. `keepdim` is
        the same branch — a shape missing one axis fits the broadcasting by
        accident.
        """
        return Tensor(_np.all(self.data, axis=dim, keepdims=bool(keepdim)))

    def any(self, dim=None, keepdim=False):
        return Tensor(_np.any(self.data, axis=dim, keepdims=bool(keepdim)))

    # ---- shape

    def reshape(self, *shape):
        # numpy's reshape gives a view where it can — held as-is the storage is
        # shared, and that is torch's behaviour. `b = a.view(2,2); b[0,0]=9`
        # changes a. Copying would be convenient and would stop teaching the
        # exact point where accidents happen in practice.
        shape = shape[0] if len(shape) == 1 and isinstance(shape[0], (tuple, list)) else shape
        old = self.data.shape
        try:
            out = self.data.reshape(shape)
        except ValueError:
            want = list(shape)
            raise RuntimeError(_like_torch(
                f"Shape {want} does not fit a tensor of {self.data.size} elements.",
                f"shape '{want}' is invalid for input of size {self.data.size}")) from None
        return self._make(out, (self,), lambda g: (g.reshape(old),), "ViewBackward0")

    def view(self, *shape):
        """Unlike `reshape`, this works **only when the storage can be used as
        it is.**

        On something whose memory order is off, such as a transposed tensor,
        torch refuses and points at `reshape`. Learning the difference between
        the two here is right.
        """
        if not self.data.flags["C_CONTIGUOUS"]:
            raise RuntimeError(_like_torch(
                "view() cannot be used on a tensor whose memory order is broken — use "
                "`.contiguous().view(...)` or `.reshape(...)`.",
                "view size is not compatible with input tensor's size and stride "
                "(at least one dimension spans across two contiguous subspaces). "
                "Use .reshape(...) instead."))
        return self.reshape(*shape)

    def unsqueeze(self, dim):
        old = self.data.shape
        return self._make(_np.expand_dims(self.data, dim), (self,), lambda g: (g.reshape(old),),
                          "UnsqueezeBackward0")

    def squeeze(self, dim=None):
        old = self.data.shape
        out = _np.squeeze(self.data) if dim is None else _np.squeeze(self.data, axis=dim)
        return self._make(out, (self,), lambda g: (g.reshape(old),))

    def transpose(self, d0, d1):
        return self._make(_np.swapaxes(self.data, d0, d1), (self,),
                          lambda g: (_np.swapaxes(g, d0, d1),), "TransposeBackward0")

    @property
    def T(self):
        return self.transpose(-2, -1)

    def permute(self, *dims):
        dims = dims[0] if len(dims) == 1 and isinstance(dims[0], (tuple, list)) else dims
        if len(dims) != self.data.ndim:
            # numpy's own words here are "axes don't match array", as a
            # `ValueError`. torch says which two numbers disagree and raises
            # `RuntimeError`, which is what a caller's `except` clause names.
            raise RuntimeError(
                "permute: number of dimensions in the tensor input does not match "
                f"the length of the desired ordering: got {self.data.ndim} and "
                f"{len(dims)}")
        inv = _np.argsort(dims)
        return self._make(_np.transpose(self.data, dims), (self,),
                          lambda g: (_np.transpose(g, inv),))

    def contiguous(self):
        """**It simply handed back `self`.** Right when it is already contiguous
        and wrong otherwise — called after a transpose it stayed non-contiguous,
        so `is_contiguous()` gave the opposite answer from torch. The gradient is
        the identity.
        """
        if self.data.flags["C_CONTIGUOUS"]:
            return self
        return self._make(_np.ascontiguousarray(self.data), (self,),
                          lambda g: (_np.asarray(g),), "CloneBackward0")

    # ── the four that ask ─────────────────────────────────────────────────
    #
    # All four **actually come back false in torch** — that was measured before
    # they went in. A predicate that is always true only pins our own
    # implementation, so asking it as a case is not asking anything.
    #
    # `is_contiguous` means something here too. numpy gives a transpose, a
    # permute and a strided slice as **views**, so it becomes false there, and
    # that is torch's answer. The browser side makes no views and is always true,
    # and that is a story about **views** rather than about this predicate.

    def is_floating_point(self):
        return bool(self.data.dtype.kind == "f")

    # ── the six that have an answer on a dense tensor too ─────────────────
    #
    # Going by the names they get counted as **rightly absent, being sparse,
    # device or quantisation names.** Measured, torch simply does all six on a
    # dense tensor — what is needed is not sparse or quantisation machinery but
    # the answers "this tensor is dense" and "it is on the CPU". Counted by name,
    # a defect that does not exist gets pinned, and when somebody implements it
    # later **a case that was green turns red.**
    #
    # `to_dense` and `dequantize` **carry the gradient** (measured) — being the
    # identity, it passes straight through.

    def dense_dim(self):
        """On a dense tensor every axis is dense."""
        return self.data.ndim

    def sparse_dim(self):
        """A dense tensor has no sparse axes."""
        return 0

    def to_dense(self):
        """It is already dense — torch hands back the same object too
        (measured)."""
        return self

    def dequantize(self):
        """The identity over the reals. Not a place that needs a quantised
        dtype."""
        return self

    def storage_offset(self):
        """Our arrays always start at the beginning of their own buffer — the
        storage is not shared."""
        return 0

    def get_device(self):
        """A CPU tensor gives -1 (measured). It means there is no device index,
        not that something went wrong."""
        return -1

    def is_signed(self):
        """False for booleans and unsigned integers only — floats, signed
        integers and complex are true."""
        return bool(self.data.dtype.kind in "fci")

    def is_contiguous(self):
        return bool(self.data.flags["C_CONTIGUOUS"])

    def is_nonzero(self):
        """**There has to be exactly one element.** With more, torch stops with
        "ambiguous" — the place that stops `if tensor:` from quietly looking at
        the first element."""
        if self.data.size != 1:
            raise RuntimeError(_like_torch(
                f"The truth value of a tensor with {self.data.size} values is ambiguous.",
                "Boolean value of Tensor with "
                f"{'no values' if self.data.size == 0 else 'more than one value'}"
                " is ambiguous"))
        return bool(self.data.reshape(-1)[0] != 0)

    def flatten(self, start_dim=0):
        shape = self.data.shape[:start_dim] + (-1,)
        return self.reshape(shape)

    def __getitem__(self, idx):
        key = tuple(i.data if isinstance(i, Tensor) else i for i in idx) \
            if isinstance(idx, tuple) else (idx.data if isinstance(idx, Tensor) else idx)
        old = self.data.shape

        def back(g):
            z = _np.zeros(old, dtype=_np.asarray(g).dtype)
            _np.add.at(z, key, g)
            return (z,)

        return self._make(self.data[key], (self,), back)

    def __setitem__(self, idx, value):
        if self.requires_grad and _grad_mode.enabled:
            raise RuntimeError("A tensor that requires grad cannot be assigned into in place.")
        key = tuple(i.data if isinstance(i, Tensor) else i for i in idx) \
            if isinstance(idx, tuple) else (idx.data if isinstance(idx, Tensor) else idx)
        self.data[key] = value.data if isinstance(value, Tensor) else value

    # ---- in-place operations
    #
    # **The buffer really is changed.** The core's views share the numpy array
    # (measured: `np.shares_memory(a.data, a.view(2,2).data)` is true), so
    # changing a view changes the original — as in torch. The sister library
    # cannot propagate that because a TF.js tensor is immutable, and that is where
    # the two part.
    #
    # The places torch refuses are refused here too. A leaf with gradients on
    # cannot be changed, and neither can a value the backward pass needs — the
    # latter goes uncaught here, and the leaf case is caught.

    def _inplace(self, fn, what):
        if self.requires_grad and _grad_mode.enabled:
            raise RuntimeError(_like_torch(
                f"`{what}` cannot be used on a leaf tensor that requires grad. Do it inside "
                "`with torch.no_grad():`, or use an out-of-place operation.",
                "a leaf Variable that requires grad is being used in an in-place operation"))
        out = fn()
        got = out.data if isinstance(out, Tensor) else _np.asarray(out)
        # **Some in-place operations change the shape.** `transpose_`,
        # `squeeze_` and `unsqueeze_` change the frame rather than the values.
        # Writing back the values alone amounts to putting a 3×2 into a 2×3 slot
        # and blows up, and asked with squares only it **passes with the shape
        # unchanged** — a 2×2 case really did fail to see this. There the array
        # is swapped out instead. numpy's transpose and axis insertion are views,
        # so the buffer stays shared and only the frame changes.
        if got.shape != self.data.shape:
            self._array = got                  # `.data` refuses an ndarray on purpose
            return self
        # **A result that does not fit the buffer's type is refused, not cast.**
        # `self.data[...] = got` let numpy narrow silently, so on an integer tensor
        # `acos_()` wrote `[0, 0, 0]` and `sqrt_()` wrote `[1, 1, 1]` — the float
        # answer truncated into the slots it was assigned to, with a numpy
        # `RuntimeWarning` nobody reads and no error at all. About twenty in-place
        # functions had it, all of them silently wrong numbers rather than wrong
        # types.
        #
        # torch's rule is the one used here, in torch's words: an in-place operation
        # writes into the tensor it was called on, so the result has to be
        # representable there. `mul_(2)` on integers is fine and `div_(2)` is not,
        # which is exactly `np.can_cast` under the "same_kind" rule numpy assignment
        # already applies — it just applies it without complaining.
        #
        # Found by enumerating the dtype axis: every one-argument function called
        # with an integer tensor. No case list had these, because writing one means
        # already suspecting that `sqrt_` on integers is a thing anybody does.
        if not _np.can_cast(got.dtype, self.data.dtype, casting="same_kind"):
            raise RuntimeError(_like_torch(
                f"`{what}` produced {_TYPE_NAMES.get(got.dtype.kind, got.dtype.name)} "
                f"and this tensor holds "
                f"{_TYPE_NAMES.get(self.data.dtype.kind, self.data.dtype.name)}. An "
                "in-place operation writes into its own buffer, so the result has to "
                "fit there — use the out-of-place form.",
                f"result type {_TYPE_NAMES.get(got.dtype.kind, got.dtype.name)} can't "
                "be cast to the desired output type "
                f"{_TYPE_NAMES.get(self.data.dtype.kind, self.data.dtype.name)}"))
        self.data[...] = got
        return self

    def add_(self, other, alpha=1):
        return self._inplace(lambda: self + (other * alpha if alpha != 1 else other), "add_")

    def sub_(self, other, alpha=1):
        return self._inplace(lambda: self - (other * alpha if alpha != 1 else other), "sub_")

    def mul_(self, other):
        return self._inplace(lambda: self * other, "mul_")

    def div_(self, other):
        return self._inplace(lambda: self / other, "div_")

    def pow_(self, exponent):
        return self._inplace(lambda: self ** exponent, "pow_")

    def neg_(self):
        return self._inplace(lambda: -self, "neg_")

    def zero_(self):
        return self._inplace(lambda: _np.zeros_like(self.data), "zero_")

    def fill_(self, value):
        return self._inplace(lambda: _np.full_like(self.data, value), "fill_")

    def copy_(self, other):
        return self._inplace(
            lambda: (other.data if isinstance(other, Tensor) else _np.asarray(other)),
            "copy_")

    def clamp_(self, min=None, max=None):
        return self._inplace(lambda: _np.clip(self.data, min, max), "clamp_")

    clip_ = clamp_

    # ---- reductions

    def _cast_first(self, dtype):
        """A reduction that received `dtype=` calls this first. Given nothing it
        is the identity.

        **One line of rule: convert before going in.** Not convert after folding —
        measurement pins that:

            torch.tensor([1.7, -2.3, 0.9]).sum(dtype=torch.int64)  →  -1

        Folded first it is `0.3`, and truncated to an integer that is `0`.
        Truncated first it is `[1, -2, 0]`, so the sum is `-1`. This is where the
        answers part, which makes this one line the definition of `dtype=`.
        `mean(dtype=float32)` running on integer input is the same reason — what
        was refused was **not being able to decide the output dtype**, and once
        that is given it runs.
        """
        return self if dtype is None else self.to(dtype)

    def _reduce(self, fn, dim, keepdim, grad_fn, op=None):
        axis = dim
        out = fn(self.data, axis=axis, keepdims=keepdim)
        return self._make(out, (self,), lambda g: (grad_fn(g, axis, keepdim),), op)

    def sum(self, dim=None, keepdim=False, dtype=None):
        if dtype is not None:
            # **The output dtype is pinned as well.** With the cast alone the
            # accumulation rule promotes it again — `sum(dtype=bool)` comes out
            # int64 (torch gives `True`, measured).
            return self._cast_first(dtype).sum(dim, keepdim).to(dtype)
        shape = self.data.shape

        def back(g, axis, kd):
            g = _np.asarray(g)
            if axis is not None and not kd:
                g = _np.expand_dims(g, axis)
            return _np.broadcast_to(g, shape).copy()

        return self._reduce(_np.sum, dim, keepdim, back,
                            "SumBackward0" if dim is None else "SumBackward1")

    def mean(self, dim=None, keepdim=False, dtype=None):
        if dtype is not None:
            # **Asking it down to an integer is refused** (measured). `dtype=`
            # lifts the refusal on the input side (integer input plus
            # `dtype=float32` runs), and a mean whose result is an integer still
            # has no answer — what is lifted is **only** the inability to decide
            # the output dtype.
            _needs_float(
                _np.empty(0, dtype=_np.dtype(getattr(dtype, "np", dtype))),
                "The output dtype of a mean has to be a floating point one.",
                "mean(): could not infer output dtype. Input dtype must be either "
                "a floating point or complex dtype")
            return self._cast_first(dtype).mean(dim, keepdim).to(dtype)
        _needs_float(
            self.data,
            "A mean exists over the reals only — the answer to a division does "
            "not fit in an integer or boolean cell. Call `.float()` first.",
            "mean(): could not infer output dtype. Input dtype must be either "
            "a floating point or complex dtype")
        shape = self.data.shape
        n = self.data.size if dim is None else shape[dim]

        def back(g, axis, kd):
            g = _np.asarray(g)
            if axis is not None and not kd:
                g = _np.expand_dims(g, axis)
            return _np.broadcast_to(g, shape).copy() / n

        return self._reduce(_np.mean, dim, keepdim, back,
                            "MeanBackward0" if dim is None else "MeanBackward1")

    def _argreduce(self, np_fn, np_arg, dim, keepdim):
        if dim is None:
            # **With no axis there are no indices, and so the rule reverses.**
            # `max(dim=0)`, which gives indices, hands the whole gradient to the
            # one chosen position, and `max()`, which gives none, **splits it
            # evenly across a tie** — `amax()`'s rule (measured:
            # [3,5,5,1,5] → [0, ⅓, ⅓, 0, ⅓]).
            #
            # This used to build a bare `Tensor(...)` here and **the graph was
            # quietly cut.** Every value check passed — because the values were
            # right. It surfaced on calling `backward()`, and what comes out
            # there says "a tensor that does not require grad", which points at
            # the user. Nobody says the operation is missing.
            value = _np.asarray(np_fn(self.data))
            hit = (self.data == value).astype(self.data.dtype)
            share = hit / hit.sum()
            return self._make(value, (self,),
                              lambda g: (_np.asarray(g) * share,))
        idx = np_arg(self.data, axis=dim)
        values = _np.take_along_axis(self.data, _np.expand_dims(idx, dim), axis=dim)
        if not keepdim:
            values = _np.squeeze(values, axis=dim)
        shape, d = self.data.shape, dim

        def back(g):
            z = _np.zeros(shape, dtype=_np.asarray(g).dtype)
            gg = _np.asarray(g)
            if not keepdim:
                gg = _np.expand_dims(gg, d)
            _np.put_along_axis(z, _np.expand_dims(idx, d), gg, axis=d)
            return (z,)

        out = self._make(values, (self,), back)
        # **The indices have to keep the axis too.** Keeping it on the values
        # alone makes `x.gather(1, m.indices)` stop on a rank mismatch or — worse
        # — pass by broadcasting. torch gives `(2, 1)` for both (measured).
        return _MinMax(out, Tensor(_np.expand_dims(idx, d) if keepdim else idx))

    def _elementwise_extreme(self, other, pick, name):
        """**A tie splits in half.** That is what torch does — the gradient of
        `maximum(2, 2)` is 0.5 on both sides. `_ops.maximum`'s rule, written again
        here because `_tensor.py` cannot import `_ops` (a cycle)."""
        other = other if isinstance(other, Tensor) else Tensor(_np.asarray(other))
        tie = self.data == other.data
        left = _np.where(tie, 0.5, (self.data > other.data).astype(self.data.dtype))
        if name == "MinimumBackward0":
            left = 1.0 - left
        return self._make(pick(self.data, other.data), (self, other),
                          lambda g: (g * left, g * (1.0 - left)), name)

    # **Three things under one name.** torch's `max` produces different things
    # depending on the arguments: with none, one overall maximum; with an axis, a
    # `(values, indices)` pair; **with a tensor, the elementwise maximum.** The
    # last branch was missing, so `torch.max(a, b)` took it for an axis and
    # stopped with `'Tensor' object cannot be interpreted as an integer`.

    def max(self, dim=None, keepdim=False):
        if isinstance(dim, Tensor):
            return self._elementwise_extreme(dim, _np.maximum, "MaximumBackward0")
        return self._argreduce(_np.max, _np.argmax, dim, keepdim)

    def min(self, dim=None, keepdim=False):
        if isinstance(dim, Tensor):
            return self._elementwise_extreme(dim, _np.minimum, "MinimumBackward0")
        return self._argreduce(_np.min, _np.argmin, dim, keepdim)

    def argmax(self, dim=None, keepdim=False):
        _refuses_bool(self.data, "argmax does not take booleans.",
                      "argmax(): does not support bool input")
        return Tensor(_keep(_np.argmax(self.data, axis=dim), self, dim, keepdim))

    def argmin(self, dim=None, keepdim=False):
        _refuses_bool(self.data, "argmin does not take booleans.",
                      "argmin(): does not support bool input")
        return Tensor(_keep(_np.argmin(self.data, axis=dim), self, dim, keepdim))

    def var(self, dim=None, unbiased=True, keepdim=False):
        """Computed **inside the graph.**

        It used to take the value out through `np.var` and hand it back. The
        value is right and no gradient flows — put a variance into the loss and
        training quietly stops. The same kind of thing ROADMAP item 11 caught in
        topk and sort, and it survived here because there was no check.
        """
        _needs_float(
            self.data,
            "Variance and standard deviation exist over the reals only. Call "
            "`.float()` first.",
            "std and var only support floating point and complex dtypes")
        n = self.data.size if dim is None else self.data.shape[dim]
        mean = self.mean(dim=dim, keepdim=True) if dim is not None else self.mean()
        centered = self - mean
        total = (centered * centered).sum(dim=dim, keepdim=keepdim)
        return total / float(n - 1 if unbiased else n)

    def std(self, dim=None, unbiased=True, keepdim=False):
        return self.var(dim=dim, unbiased=unbiased, keepdim=keepdim) ** 0.5

    def abs(self):
        """The magnitude. **On complex numbers the result is real and the
        gradient is `z/|z|`.**

        `sign` must not be used as-is — numpy's complex `sign` differs from
        torch's, and torch refuses `sign` on complex numbers in the first place
        (measured). What is needed here is `z/|z|`, which bundles
        `∂|z|/∂re = re/|z|` and `∂|z|/∂im = im/|z|`.
        """
        if self.data.dtype.kind == "c":
            mag = _np.abs(self.data)
            safe = _np.where(mag == 0, 1.0, mag)
            return self._make(
                mag.astype(_DEFAULT_DTYPE), (self,),
                lambda g: ((_np.asarray(g) * self.data / safe).astype(self.data.dtype),))
        return self._make(_np.abs(self.data), (self,), lambda g: (g * _np.sign(self.data),))

    # **These three promote an integer input, as torch does.** Handed an integer
    # array numpy answers in `float64`, which is right in value and wider than
    # torch's `float32` — and a wider dtype spreads, because everything downstream
    # promotes to meet it. `_float_in` is the same door the module-level functions
    # use.
    def exp(self):
        data = _float_in(self.data)
        out = _np.exp(data)
        return self._make(out, (self,), lambda g: (g * out,))

    def log(self):
        data = _float_in(self.data)
        return self._make(_np.log(data), (self,), lambda g: (g / data,))

    def sqrt(self):
        data = _float_in(self.data)
        out = _np.sqrt(data)
        return self._make(out, (self,), lambda g: (g * 0.5 / out,))

    def masked_fill(self, mask, value):
        m = mask.data.astype(bool) if isinstance(mask, Tensor) else _np.asarray(mask, dtype=bool)
        out = _np.where(m, _np.asarray(value, dtype=self.data.dtype), self.data)
        return self._make(out, (self,), lambda g: (_np.where(m, 0, g),))

    def bincount(self):
        # `intp` — on wasm32, handing it int64 is refused. See
        # `_ops.repeat_interleave`. **Booleans are refused** — torch stops with
        # `"bincount_cpu" not implemented for 'Bool'` (measured). `_ops.bincount`
        # has the same guard and the method was calling numpy directly without
        # passing that gate — this is how two copies diverge.
        _refuses_bool(self.data, "bincount does not take booleans.",
                      '"bincount_cpu" not implemented for \'Bool\'',
                      kind=NotImplementedError)
        return Tensor(_np.bincount(self.data.astype(_np.intp)))


class _MinMax:
    """The (values, indices) `x.max(dim=0)` hands back. Real torch's shape."""

    def __init__(self, values, indices):
        self.values = values
        self.indices = indices

    def __iter__(self):
        yield self.values
        yield self.indices

    def __getitem__(self, i):
        return (self.values, self.indices)[i]




# ── the in-place versions built from their partners ─────────────────────────
#
# torch's in-place operations carry a trailing underscore, as in `x.add_(1)`,
# and **they are the idiom in a training loop** — a textbook that does not write
# `p.data.add_(-lr * g)` is rare. And forty-one of them were missing. The
# partners (`x.add`) all existed and so did the `_inplace` idiom, so what was
# missing was **one connecting line** each.
#
# Forty-one copies are not written by hand. When the only difference is the
# name, one of those forty-one eventually gets fixed differently from the rest,
# and nobody looks at that one. A table is kept and they are attached from it.
#
# **`i0_`, `clamp_min_` and `clamp_max_` all three already existed in `_ops`** —
# as module functions only and not as methods. `borch.i0_(x)` worked and
# `x.i0_()` did not, and the side textbooks use is the latter.
#
# **Two cannot go in this table.** A trailing underscore does not make it the
# same operation as its partner — all forty-one were checked against torch, and
# that is where these two parted.
#
#   `bernoulli_` is **a different operation from its partner.** `x.bernoulli()`
#   reads `x` as probabilities, and `x.bernoulli_(p=0.5)` ignores `x` and fills
#   from `p` (measured: `[0,1,0,1]` going in still comes out different every
#   time). Built from the partner, the positions at probability 0 or 1 are
#   certain and their values match, and it would be **quietly wrong only at the
#   probabilities in between.**
#
#   `float_power_` is **refused by torch.** `float_power`'s result is always
#   float64 and cannot be written back into a float32 destination. There is no
#   float64 here at all, so this operation will never work in place.
_INPLACE_FROM_PAIR = (
    "bitwise_and_", "bitwise_left_shift_", "bitwise_not_",
    "bitwise_or_", "bitwise_right_shift_", "bitwise_xor_", "clamp_max_",
    "clamp_min_", "digamma_", "divide_", "erfinv_",
    "floor_divide_", "fmod_", "gcd_", "greater_", "greater_equal_", "i0_",
    # **Three were missing.** Forty-one were measured and then the list was
    # copied out by hand, dropping `index_reduce_` and `scatter_reduce_`, and
    # `resize_as_` fell out alongside while tidying the names that existed on the
    # module only — a place where **what was measured and what was written
    # diverge.**
    "index_reduce_", "scatter_reduce_",
    "lcm_", "lerp_", "less_", "less_equal_", "lgamma_", "logical_and_",
    "logical_not_", "logical_or_", "logical_xor_", "multiply_", "mvlgamma_",
    "nan_to_num_", "nextafter_", "not_equal_", "put_", "remainder_", "renorm_",
    "subtract_", "t_", "true_divide_",
)


def _bind_inplace(name):
    pair = name[:-1]

    def method(self, *args, **kw):
        return self._inplace(lambda: getattr(self, pair)(*args, **kw), name)

    method.__name__ = name
    method.__qualname__ = f"Tensor.{name}"
    method.__doc__ = (f"`{pair}` in place. It writes the values back and hands "
                      "back itself.")
    return method


for _name in _INPLACE_FROM_PAIR:
    setattr(Tensor, _name, _bind_inplace(_name))
del _name


def _float_power_(self, exponent):
    """**Always refuses.** torch refuses on a float32 destination too —
    `float_power`'s result is float64 and cannot be written back into float32.
    There is no float64 here, so it works at no dtype at all. Handing back a
    value means that code breaks against real torch."""
    del exponent
    raise RuntimeError(_like_torch(
        f"`float_power_` cannot be used on a {self.dtype} slot — the result is double "
        "precision and there is nowhere to put it back. Use `x.float_power(k)` for a "
        "new tensor.",
        f"the base given to float_power_ has dtype {str(self.dtype).split('.')[-1].capitalize()} "
        "but the operation's result requires dtype Double"))


Tensor.float_power_ = _float_power_


# ── an absent dtype is refused by name ──────────────────────────────────────
#
# This used to raise `AttributeError: 'Tensor' object has no attribute 'half'`.
# **That wording is indistinguishable from a typo** — a learner reads it as
# having mistyped the name rather than as that dtype being absent here.
#
# `half` and `bfloat16` are **lines actually typed** in a tutorial's
# mixed-precision section, so what comes out there has to be the same across all
# three. This is a place where **the refusal wording** is matched rather than a
# value, and such places are not caught by comparing the three — the same branch
# `i0` was bitten by.
_ABSENT_DTYPES = {
    "half": "float16", "bfloat16": "bfloat16", "chalf": "complex32",
    "cdouble": "complex128", "byte": "uint8", "char": "int8", "short": "int16",
}


def _bind_absent_dtype(name, shown):
    def method(self):
        del self
        _unsupported(f"`.{name}()`({shown})")

    method.__name__ = name
    method.__qualname__ = f"Tensor.{name}"
    method.__doc__ = (f"{shown} does not exist in this subset — it does not "
                      "hand over a different storage instead.")
    return method


for _dname, _shown in _ABSENT_DTYPES.items():
    setattr(Tensor, _dname, _bind_absent_dtype(_dname, _shown))
del _dname, _shown


# ── names that existed on the module only, exposed as methods too ───────────
#
# torch offers nearly every operation **both ways** — `torch.igamma(x, y)` and
# `x.igamma(y)`. Thirteen places here had the module side and no method.
# `borch.arctan2(x, y)` worked and `x.arctan2(y)` was an `AttributeError`, and
# **the side textbooks use is the method.**
#
# The binding's `_base.py` already wrote the same story down ("only the one-way
# loop was left, where `borch.t(x)` works and `x.t()` does not") — that side
# filled it in at the time and the core did not.
#
# **`lstsq` and `solve` are not here.** torch deprecated them in 1.9 and **now
# refuses them** — the names survive and calling one stops. They were counted in
# at first as "names that exist in torch", and measuring the argument order
# showed that side refusing. Handing back an answer means that code breaks
# against real torch — **being more permissive is still diverging.**
_METHOD_FROM_MODULE = (
    "arctan2", "igamma", "igammac", "geqrf",
    # **An underscore name with no partner.** `resize_as_` exists on the module
    # only and there is no `resize_as` partner at all, so put into the derived
    # table it looks for a name that is not there and stops with an
    # `AttributeError` — writing a name into a table and the table being able to
    # build that name are different things.
    "resize_as_",
)


def _deprecated_by_torch(name, instead):
    def method(self, *args, **kw):
        del self, args, kw
        raise RuntimeError(_like_torch(
            f"`{name}` was removed in torch 1.9 — use `{instead}`.",
            f"This function was deprecated since version 1.9 and is now removed. "
            f"Please use the `torch.linalg.{instead}` function instead."))

    method.__name__ = name
    return method


def _polygamma(self, n):
    """**The arguments are reversed.** The module is `polygamma(n, x)` and the
    method is `x.polygamma(n)` — that is how torch keeps them (measured).

    Attached blindly from the table it was caught by a `TypeError`. Uncaught, a
    value would have come out with the order and the input swapped — `lu_solve`
    had the same shape.
    """
    from . import _ops
    return _ops.polygamma(n, self)


def _bind_from_module(name):
    def method(self, *args, **kw):
        from . import _ops
        return getattr(_ops, name)(self, *args, **kw)

    method.__name__ = name
    method.__qualname__ = f"Tensor.{name}"
    method.__doc__ = f"`borch.{name}` as a method. torch offers both."
    return method


for _mname in _METHOD_FROM_MODULE:
    setattr(Tensor, _mname, _bind_from_module(_mname))
del _mname


def _is_same_size(self, other):
    """Do the shapes match. **The shape only, not the values.**"""
    return tuple(self.data.shape) == tuple(_np.asarray(other.data).shape)


def _fill_diagonal_(self, value, wrap=False):
    """Fill the diagonal. `wrap` **wraps the diagonal around** on a tall
    matrix."""
    if self.requires_grad and _grad_mode.enabled:
        raise RuntimeError(_like_torch(
            "`fill_diagonal_` cannot be used on a leaf tensor that requires grad.",
            "a leaf Variable that requires grad is being used in an in-place operation"))
    _np.fill_diagonal(self._array, value, wrap=wrap)
    return self


def _requires_grad_(self, requires_grad=True):
    """**A textbook idiom** — `x.requires_grad_()` turns a leaf on. It hands
    back itself."""
    if requires_grad and self.data.dtype.kind not in "fc":
        raise RuntimeError(
            "Gradients do not flow through integer tensors. Differentiation is defined "
            "on floating point only — convert with `.float()`.")
    self.requires_grad = bool(requires_grad)
    return self


def _share_memory_(self):
    """There is no sharing between processes here. torch hands back itself on
    the CPU too."""
    return self


Tensor.polygamma = _polygamma
Tensor.lstsq = _deprecated_by_torch("lstsq", "lstsq")
Tensor.solve = _deprecated_by_torch("solve", "solve")
Tensor.is_same_size = _is_same_size
Tensor.is_distributed = lambda self: False
Tensor.is_inference = lambda self: False
Tensor.fill_diagonal_ = _fill_diagonal_
Tensor.requires_grad_ = _requires_grad_
Tensor.share_memory_ = _share_memory_

# With the partners in place, the underscore versions come from the same table.
for _iname in ("arctan2_", "igamma_", "igammac_", "polygamma_"):
    setattr(Tensor, _iname, _bind_inplace(_iname))
del _iname


# ── the predicates torch offers as **properties** ───────────────────────────
#
# Most of them ask where this tensor lives or what storage it uses, and our
# answer is fixed. **The name still has to exist** — without it `if x.is_cuda:`
# stops with an `AttributeError`, and in torch that line simply passes as false.
#
# **`is_leaf` alone is a real computation.** A leaf is a tensor that did not come
# out of an operation, and when it is false `.grad` does not accumulate — a
# different nature from the rest, whose values are fixed.
_ALWAYS_FALSE = (
    "is_cuda", "is_ipu", "is_maia", "is_meta", "is_mkldnn", "is_mps", "is_mtia",
    "is_nested", "is_quantized", "is_sparse", "is_sparse_csr", "is_vulkan",
    "is_xla", "is_xpu",
)

for _pname in _ALWAYS_FALSE:
    setattr(Tensor, _pname,
            property(lambda self: False, doc="Absent from this subset."))
del _pname

# It lives on the CPU. **The binding parts here** — its values are in a GPU
# buffer, so it is false there.
Tensor.is_cpu = property(lambda self: True)
Tensor.is_leaf = property(
    lambda self: not self._parents,
    doc="A tensor that did not come out of an operation. **False means `.grad` "
        "does not accumulate.**")
Tensor.retains_grad = property(
    lambda self: bool(getattr(self, "_retain", False)),
    doc="True only on a derived tensor that called `retain_grad()` — on a leaf "
        "torch gives false too.")


def _is_pinned(self):
    """**A method** — unlike the ones above it takes parentheses (measured).
    There is no pinned memory."""
    del self
    return False


def _is_coalesced(self):
    """Sparse-only, so **it stops on a dense tensor** — torch is the same
    (measured)."""
    raise RuntimeError(_like_torch(
        "A dense tensor has no coalesce state.",
        "is_coalesced expected sparse coordinate tensor layout but got Strided"))


# **`is_neg` and `is_pinned` alone are methods** — they take parentheses
# (measured). Kept as properties, `x.is_neg` hands back a boolean rather than a
# bound method, while torch's is a bound method — this time **making it a
# property was the divergence.**
Tensor.is_neg = lambda self: False
Tensor.is_pinned = _is_pinned
Tensor.is_coalesced = _is_coalesced


# ── the eight in-place versions with no partner ─────────────────────────────
#
# The derived table cannot build them — there is no partner at all. torch does
# five of them and three are sparse-only, so **torch stops on a dense tensor
# too.** Grouped by name as "in-place, so build them all", we become more
# permissive on the last three.

def _apply_(self, fn):
    """Apply a Python function per element. **torch does this on the CPU only** —
    a slow path, so the name gives convenience rather than a value."""
    _refuse_leaf_inplace(self, "apply_")
    flat = self.data.reshape(-1)
    self.data[...] = _np.array([fn(v.item()) for v in flat],
                               dtype=self.data.dtype).reshape(self.data.shape)
    return self


def _map_(self, other, fn):
    _refuse_leaf_inplace(self, "map_")
    a, b = self.data.reshape(-1), _np.broadcast_to(other.data, self.data.shape).reshape(-1)
    self.data[...] = _np.array([fn(x.item(), y.item()) for x, y in zip(a, b)],
                               dtype=self.data.dtype).reshape(self.data.shape)
    return self


def _map2_(self, other, third, fn):
    _refuse_leaf_inplace(self, "map2_")
    a = self.data.reshape(-1)
    b = _np.broadcast_to(other.data, self.data.shape).reshape(-1)
    c = _np.broadcast_to(third.data, self.data.shape).reshape(-1)
    self.data[...] = _np.array(
        [fn(x.item(), y.item(), z.item()) for x, y, z in zip(a, b, c)],
        dtype=self.data.dtype).reshape(self.data.shape)
    return self


def _resize_(self, *sizes):
    """**Growing leaves the new cells undefined** — torch gives garbage
    (measured: sometimes it happens to be zero). This fills with zeros. The
    values cannot be pinned, so the golden asks about **shrinking and the shape
    only.**"""
    _refuse_leaf_inplace(self, "resize_")
    shape = tuple(sizes[0]) if len(sizes) == 1 and isinstance(sizes[0], (tuple, list)) \
        else tuple(int(s) for s in sizes)
    want = int(_np.prod(shape)) if shape else 1
    flat = self.data.reshape(-1)
    if want <= flat.size:
        self._array = flat[:want].reshape(shape).copy()
    else:
        grown = _np.zeros(want, dtype=self.data.dtype)
        grown[:flat.size] = flat
        self._array = grown.reshape(shape)
    return self


def _set_(self, source=None):
    """**Swaps the storage wholesale.** With no arguments it becomes an empty
    tensor."""
    _refuse_leaf_inplace(self, "set_")
    self._array = (_np.empty(0, dtype=self.data.dtype) if source is None
                   else _np.asarray(source.data))
    return self


def _refuse_leaf_inplace(self, name):
    if self.requires_grad and _grad_mode.enabled:
        raise RuntimeError(_like_torch(
            f"`{name}` cannot be used on a leaf tensor that requires grad.",
            "a leaf Variable that requires grad is being used in an in-place operation"))


def _sparse_only(name):
    def method(self, *args, **kw):
        del self, args, kw
        raise NotImplementedError(_like_torch(
            f"`{name}` is for sparse tensors only — not for a dense one.",
            f"Could not run 'aten::{name}' with arguments from the 'CPU' backend"))

    method.__name__ = name
    return method


Tensor.apply_ = _apply_
Tensor.map_ = _map_
Tensor.map2_ = _map2_
Tensor.resize_ = _resize_
Tensor.set_ = _set_
for _sname in ("resize_as_sparse_", "sparse_resize_", "sparse_resize_and_clear_"):
    setattr(Tensor, _sname, _sparse_only(_sname))
del _sname


# ── the ones that look into the storage ─────────────────────────────────────
#
# They ask **how it is laid out** rather than what the values are. numpy answers
# directly, so real numbers can come out here — `stride()` goes from `(3,1)` to
# `(1,3)` on a transpose, and that is the fact of it being a view.

def _stride(self, dim=None):
    """Strides in elements. **Not bytes** — numpy's `strides` divided by the
    element size."""
    got = tuple(s // self.data.itemsize for s in self.data.strides)
    return got if dim is None else got[dim]


def _dim_order(self):
    """The axes ordered from the fastest in memory. Contiguous it is
    `(0, 1, …)`, and a transpose reverses it."""
    return tuple(int(i) for i in _np.argsort([-s for s in self.data.strides]))


Tensor.stride = _stride
Tensor.dim_order = _dim_order
Tensor.element_size = lambda self: int(self.data.itemsize)
Tensor.nelement = lambda self: int(self.data.size)
Tensor.ndimension = lambda self: int(self.data.ndim)
Tensor.itemsize = property(lambda self: int(self.data.itemsize))
Tensor.nbytes = property(lambda self: int(self.data.nbytes))
Tensor.data_ptr = lambda self: int(self.data.__array_interface__["data"][0])
Tensor.const_data_ptr = Tensor.data_ptr
Tensor.layout = property(lambda self: _Layout())
Tensor.output_nr = property(lambda self: 0)
Tensor.volatile = property(lambda self: False)
Tensor.name = property(lambda self: None)
Tensor.grad_dtype = property(lambda self: self.dtype)


class _Layout:
    """`torch.strided`. There is no other layout here — no sparse and no
    mkldnn."""

    __slots__ = ()

    def __repr__(self):
        return "torch.strided"

    __str__ = __repr__

    def __eq__(self, other):
        return repr(other) == "torch.strided"

    def __hash__(self):
        return hash("torch.strided")


# ── the three names for transposition ───────────────────────────────────────
#
# `H` is **2-D only** (torch stops there), and `mT` and `mH` swap **the last two
# axes only**, so they work on batches. The three part on whether they conjugate —
# `mT` alone does not. Over the reals all three give the same answer, so **asking
# with complex numbers** is what shows the difference.

def _hermitian(self):
    if self.data.ndim != 2:
        raise RuntimeError(_like_torch(
            f"`.H` is only on matrices (2-D) — got {self.data.ndim}-D.",
            f"tensor.H is only supported on matrices (2-D tensors). "
            f"Got {self.data.ndim}-D tensor."))
    return self.transpose(0, 1).conj()


def _matrix_transpose(self):
    if self.data.ndim < 2:
        raise RuntimeError(_like_torch(
            "`.mT` is only on 2-D and above.",
            "tensor.mT is only supported on matrices or batches of matrices. "
            f"Got {self.data.ndim}-D tensor."))
    return self.transpose(-2, -1)


Tensor.H = property(_hermitian)
Tensor.mT = property(_matrix_transpose)
Tensor.mH = property(lambda self: _matrix_transpose(self).conj())


# ── `new_*` — build a new tensor **inheriting the dtype** ───────────────────
#
# That is the only difference from `torch.zeros(...)`. It is also why textbooks
# use it — a dtype written by hand does not change when the original does.

def _new_like(name, fill):
    def method(self, *size, dtype=None, requires_grad=False):
        shape = tuple(size[0]) if len(size) == 1 and isinstance(size[0], (tuple, list)) \
            else tuple(int(s) for s in size)
        want = (dtype.np if isinstance(dtype, globals()["dtype"]) else
                (dtype if dtype is not None else self.data.dtype))
        return Tensor(fill(shape, want), requires_grad)

    method.__name__ = name
    method.__doc__ = f"`{name}` — inherits this tensor's dtype."
    return method


Tensor.new_zeros = _new_like("new_zeros", lambda s, d: _np.zeros(s, dtype=d))
Tensor.new_ones = _new_like("new_ones", lambda s, d: _np.ones(s, dtype=d))
# **`new_empty` leaves the values undefined.** torch gives garbage too — this
# fills with zeros and the golden asks about **the shape and the dtype only.**
# Pinning the values would turn them into the specification.
Tensor.new_empty = _new_like("new_empty", lambda s, d: _np.zeros(s, dtype=d))


def _new_full(self, size, fill_value, dtype=None, requires_grad=False):
    want = (dtype.np if isinstance(dtype, globals()["dtype"]) else
            (dtype if dtype is not None else self.data.dtype))
    return Tensor(_np.full(tuple(size), fill_value, dtype=want), requires_grad)


def _new_tensor(self, data, dtype=None, requires_grad=False):
    want = (dtype.np if isinstance(dtype, globals()["dtype"]) else
            (dtype if dtype is not None else self.data.dtype))
    return Tensor(_np.array(data, dtype=want, copy=True), requires_grad)


Tensor.new_full = _new_full
Tensor.new_tensor = _new_tensor
Tensor.new = lambda self, *size: (self.new_empty(*size) if size
                                  else Tensor(_np.empty(0, dtype=self.data.dtype)))

# ── the names that **take their shape from somebody else** ──────────────────
Tensor.reshape_as = lambda self, other: self.reshape(*other.shape)
Tensor.view_as = lambda self, other: self.view(*other.shape)
Tensor.resize_as = lambda self, other: self.reshape(*other.shape)
# **`narrow_copy`, `unsafe_*` and `slice_inverse` are not built.** They went in
# and `tests/test_gap.py` caught them — `NOT_API` already wrote them down as
# **internal variants for the functionalisation pass**, and the real names are
# `narrow`, `chunk` and `split`. A name existing in torch does not make it public
# API, and that judgement had already been made.


def _sum_to_size(self, *size):
    """**Undoes broadcasting.** The stretched axes are folded back into that
    shape — the same thing backpropagation does internally, and torch offers it
    under a name as well."""
    shape = tuple(size[0]) if len(size) == 1 and isinstance(size[0], (tuple, list)) \
        else tuple(int(s) for s in size)
    return Tensor(_unbroadcast(self.data, shape))


Tensor.sum_to_size = _sum_to_size


def _retain_grad(self):
    """**What decides is `requires_grad`, not whether it is a leaf** (measured).

    It was first written as "stop on a leaf", and torch **simply passes** on a
    leaf with `requires_grad=True` — a leaf already accumulates `.grad`, so there
    is nothing to ask for. What stops is only a tensor that receives no gradient
    at all. The earlier measurement was one tensor with `requires_grad=False`, and
    the rule was set from that one.

    Called on a leaf, `retains_grad` **stays false** — it is not retaining
    anything; it accumulates by nature.
    """
    if not self.requires_grad:
        raise RuntimeError(_like_torch(
            "`retain_grad()` cannot be used on a tensor that does not take gradients.",
            "can't retain_grad on Tensor that has requires_grad=False"))
    if self._parents:
        self._retain = True
    return None


Tensor.retain_grad = _retain_grad


# ── the ones **torch refuses itself** ───────────────────────────────────────
#
# The names survive and calling one stops. Handing back an answer means that
# code breaks against real torch — a place already stepped on with `lstsq` and
# `solve`.
_GONE = {
    "eig": "linalg.eig", "symeig": "linalg.eigh",
}
# The ones whose machinery does not exist here. **Saying so is the answer.**
_NO_MACHINERY = {
    "to_mkldnn": "MKL-DNN",
    "q_per_channel_axis": "per-channel quantisation",
    "q_per_channel_scales": "per-channel quantisation",
    "q_per_channel_zero_points": "per-channel quantisation",
    "smm": "sparse matrix multiplication",
    "to_padded_tensor": "nested tensors",
    "as_subclass": "Tensor subclasses",
    "module_load": "`load_state_dict`'s internal hook",
    "reinforce": "the old stochastic graph",
    "new_empty_strided": "storage with hand-given strides",
    "register_hook": "backward hooks",
    "register_post_accumulate_grad_hook": "backward hooks",
    "index": "advanced indexing's internal entry point — use `x[...]`",
}


def _bind_gone(name, instead):
    def method(self, *args, **kw):
        del self, args, kw
        raise RuntimeError(_like_torch(
            f"`{name}` was removed from torch — use `{instead}`.",
            "This function was deprecated since version 1.9 and is now removed. "
            f"Please use the `torch.{instead}` function instead."))

    method.__name__ = name
    return method


def _bind_absent(name, what):
    def method(self, *args, **kw):
        del self, args, kw
        _unsupported(f"`.{name}()`({what})")

    method.__name__ = name
    return method


for _gname, _instead in _GONE.items():
    setattr(Tensor, _gname, _bind_gone(_gname, _instead))
for _aname, _what in _NO_MACHINERY.items():
    setattr(Tensor, _aname, _bind_absent(_aname, _what))
del _gname, _instead, _aname, _what

# `resize` is the old name without the underscore, and torch runs it only when
# the conditions fit. It is kept as a thin shell calling the in-place version —
# matching torch in not handing back a copy.
Tensor.resize = lambda self, *size: self.resize_(*size)


class _GradFn:
    """The `grad_fn` slot. **The address cannot be matched and the name can.**

    torch prints `<MulBackward0 object at 0x…>`. The address is in there, so the
    characters cannot be pinned, and **the type name** (`MulBackward0`) can —
    which operation it came out of is in that name.
    """

    __slots__ = ("_name",)

    def __init__(self, name):
        self._name = name

    def __repr__(self):
        return f"<{self._name} object at 0x{id(self):012x}>"


def _grad_fn(self):
    if self._op is None:
        return None
    got = _GradFn(self._op)
    got.__class__ = type(self._op, (_GradFn,), {"__slots__": ()})
    return got


Tensor.grad_fn = property(_grad_fn)


# ── sparse, storage and quantisation — **the names exist and do not fit a
# dense tensor** ────────────────────────────────────────────────────────────
#
# These three were grouped as "rightly absent" and had no names built at all, and
# two things about that were wrong.
#
# **First, torch answers twelve of them.** `to_sparse()` takes a dense tensor and
# makes a sparse one, and `storage()` has an answer too. Those are **absent
# features** rather than "rightly absent" — the thing called a sparse tensor does
# not exist here.
#
# **Second, the wording diverged on the other twenty-two.** This gave
# `'Tensor' object has no attribute 'coalesce'`, which is **indistinguishable
# from a typo.** torch says `coalesce expected sparse coordinate tensor layout
# but got Strided` — the name exists and does not fit this tensor. The same
# branch fixed across the nine dtype conversions, met again here.

def _needs_sparse(name, layout):
    def method(self, *args, **kw):
        del self, args, kw
        raise RuntimeError(_like_torch(
            f"`{name}` is for sparse tensors only — a dense tensor does not have it.",
            f"{name} expected sparse {layout} but got Strided"))

    method.__name__ = name
    return method


for _n, _layout in (
        ("coalesce", "coordinate tensor layout"),
        ("indices", "coordinate tensor layout"),
        ("values", "tensor layout"),
        ("crow_indices", "row compressed tensor layout"),
        ("col_indices", "row compressed tensor layout"),
        ("ccol_indices", "column compressed tensor layout"),
        ("row_indices", "column compressed tensor layout")):
    setattr(Tensor, _n, _needs_sparse(_n, _layout))
del _n, _layout

# The side that **makes** a sparse tensor. torch does it and the thing called a
# sparse tensor does not exist here. This is **"that feature is absent"** rather
# than "it does not fit this tensor", so the wording is split.
for _n in ("to_sparse", "to_sparse_coo", "to_sparse_csr", "to_sparse_csc",
           "to_sparse_bsr", "to_sparse_bsc", "sparse_mask"):
    setattr(Tensor, _n, _bind_absent(_n, "sparse tensors"))
for _n in ("storage", "storage_type", "untyped_storage"):
    setattr(Tensor, _n, _bind_absent(_n, "Storage objects — use `.numpy()`"))
for _n in ("int_repr", "q_scale", "q_zero_point", "qscheme"):
    setattr(Tensor, _n, _bind_absent(_n, "quantisation"))
for _n in ("cuda", "ipu", "mtia", "xpu"):
    setattr(Tensor, _n, _bind_absent(_n, "that device"))
for _n in ("pin_memory", "record_stream"):
    setattr(Tensor, _n, _bind_absent(_n, "pinned memory and streams"))
del _n


def _is_set_to(self, other):
    """**Do two tensors point at the same storage.** The shape, the strides and
    the starting offset all have to match.

    It is not always false — a view is true and a copy is false. So this
    predicate is also **the name that asks** whether `tensor()` takes a copy and
    whether `from_numpy` shares.
    """
    if not isinstance(other, Tensor):
        return False
    mine, theirs = self.data, other.data
    return bool(
        mine.__array_interface__["data"][0] == theirs.__array_interface__["data"][0]
        and mine.shape == theirs.shape and mine.strides == theirs.strides)


Tensor.is_set_to = _is_set_to
Tensor.is_shared = lambda self: False
