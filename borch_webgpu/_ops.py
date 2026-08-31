"""Forward Python names to borch.ts methods — **without a list.**

Writing 192 names by hand means a day arrives when one of them calls a
different operation, and only comparing values shows it. Reading them off a
table is possible, but that would require `index.ts` to export the kernel
table, which puts an internal matter on the public surface.

So **the module's `__getattr__`** does it (PEP 562). `L.exp(x)` arrives and is
forwarded as `x.exp()`; absent from borch.ts, it stops with `AttributeError`.
Nothing missing is approximated — the golden counts that place as a failure, and
that count is this binding's progress.

One naming rule differs. Python writes `masked_select`, JavaScript writes
`maskedSelect`.
"""

import builtins                    # to reach `max` and `sum` unshadowed
import numpy as _np

import js as _js
from pyodide.ffi import to_js as _to_js

from ._base import (
    LinAlgError as _LinAlgError, Tensor, _DType, _js_list, _js_options, _Size,
    guarded, handle, settle, wrap,
)

_ts = _js.borch
# The prototype of a borch.ts tensor. **The only place a name can be asked about
# without an instance**, so it is taken once here — used to stop `__getattr__`
# from answering any name at all.
_PROTO = _ts.Tensor.prototype

# The ones spelled differently in the two languages. Only what the rule cannot
# reach is written out.
_RENAME = {
    # **`linalg.lu_solve` is received by the factorisation.** borch.ts's
    # `luSolve` is torch's `Tensor.lu_solve`, received by the right-hand side, so
    # forwarding it through plain camel case swaps the receiver — the name and
    # the argument count both match and only the values are wrong. The version
    # the factorisation receives is separate, as `luSolveFactored`.
    "lu_solve": "luSolveFactored",
    "adaptive_avg_pool2d": "adaptiveAvgPool",
    "adaptive_avg_pool1d": "adaptiveAvgPool",
    "absolute": "abs",
    "arccos": "acos",
    "arccosh": "acosh",
    "arcsin": "asin",
    "arcsinh": "asinh",
    "arctan": "atan",
    "arctanh": "atanh",
    "clip": "clamp",
    "fix": "trunc",
    "negative": "neg",
    "swapdims": "transpose",
    "interpolate": "upsample",
    # Ones whose name in the operation table already matches Python's. Putting
    # `camel` over them produces a name that does not exist.
    "logical_not": "logical_not",
    "logical_and": "logical_and",
    "logical_or": "logical_or",
    "logical_xor": "logical_xor",
    # The bitwise ones match the table's names too. Camel case would produce
    # `bitwiseAnd`, which does not exist.
    "bitwise_and": "bitwise_and",
    "bitwise_or": "bitwise_or",
    "bitwise_xor": "bitwise_xor",
    "bitwise_not": "bitwise_not",
    "bitwise_left_shift": "bitwise_left_shift",
    "bitwise_right_shift": "bitwise_right_shift",
    # **`matmul` was `mm` and is not any more.** borch.ts had only the 2-D by 2-D
    # kernel when this rename was written, so pointing torch's general name at it was
    # the closest thing there was; `matmul` over there now batches, broadcasts the
    # leading axes and lifts a 1-D side the way torch does, and the rename went on
    # sending every one of those to the two-dimensional one. Six golden cases came
    # back `mm is 2-D by 2-D`.
    "var": "variance",
    # **`fill` is not here.** Aliases are looked up after the underscore is
    # stripped, so listing it would carry `fill_` along into `fillWith_`, which
    # does not exist — `fill_` is in place and goes through a different door.
    "arctan2": "atan2",
}

# **Keyword arguments become positions.**
#
# torch code calls by name in many places — `clip(x, min=-0.5, max=0.5)` — and
# JavaScript has no such thing. Discarding `**kw` at first sent
# `clip(x, undefined, undefined)` down into a shader and WGSL stopped at
# parsing: that was 72 failures.
#
# So the slot names are written down. The **argument order** is borch.ts's, with
# torch's names placed into those slots. A function absent from here is one that
# takes no keyword arguments.
#
# **Twenty-three rows were removed the day anything read this table.** Nothing
# did: no test named `_SIGNATURE`, so a row could be written, be reached by no
# call, and read for years as the reason a call works. `positional()` is entered
# from four places only — the tensor method (`_base`), the module function
# (`_ops.__getattr__`), `F.` (`_nn`) and `linalg` — and a name in
# `_NOT_FORWARDED` with a hand-written `_ops` function of its own is behind all
# four. `clamp` was one: the paragraph above names it as the example, and its
# row had stopped mattering the day `def clamp` was written, because a real
# Python signature takes its own keywords. `squeeze` was in here **twice**, which
# is the fault the note further down records `quantile` having had.
#
# `tests/test_binding_arguments.py` holds it now. Adding a row that nothing can
# reach is the cheap way to look like a gap was closed, and it was available
# until then.
_SIGNATURE = {
    "clip": ("min", "max"),
    # **`dtype` is the third slot**, exactly as in torch's signature — it means
    # convert before reducing, and that order changes the values when floats are
    # folded into integers.
    "mean": ("dim", "keepdim", "dtype"),
    "prod": ("dim", "keepdim", "dtype"),
    "nansum": ("dim", "keepdim", "dtype"),
    "nanmean": ("dim", "keepdim", "dtype"),
    "amax": ("dim", "keepdim"),
    "amin": ("dim", "keepdim"),
    # **`correction` sits between them**, as it does in torch and in borch.ts. The
    # slot was missing while `std` took `(correction)` alone, and the day `dim` moved
    # to the front `keepdim=True` began arriving as a correction of 1 — a real number
    # in a real slot, so nothing raised and `std_mean` came back 4.13e-02 out.
    "var": ("dim", "correction", "keepdim"),
    "std": ("dim", "correction", "keepdim"),
    "logsumexp": ("dim", "keepdim"),
    "argmax": ("dim", "keepdim"),
    "argmin": ("dim", "keepdim"),
    "softmax": ("dim",),
    "log_softmax": ("dim",),
    "cumsum": ("dim", "dtype"),
    "cumprod": ("dim", "dtype"),
    # **The in-place twins need their own rows, and three arrived at once.** The core's
    # generated `x_ = (*args, **kw)` forwarders were taught to declare what they
    # forward, which handed `round_`, `logit_` and `heaviside_` the argument their
    # partners always had. Here that is a keyword crossing into JavaScript, and a
    # keyword with no row does not go through — `x.round_(decimals=1)` stopped at
    # *does not take keyword arguments* while `x.round(decimals=1)` computed. The
    # pair being one line apart in this table is what makes the absence visible.
    "round_": ("decimals",),
    "logit_": ("eps",),
    "heaviside_": ("values",),
    "logcumsumexp": ("dim",),
    "mvlgamma": ("p",),
    "clamp_max": ("max",),
    "clamp_min": ("min",),
    "fill": ("value",),
    "sort": ("dim", "descending", "stable"),
    "topk": ("k", "dim", "largest", "sorted"),
    "unsqueeze": ("dim",),
    # Activation arguments. Many places call by name — `F.celu(x, alpha=0.5)`.
    "celu": ("alpha",),
    "hardshrink": ("lambd",),
    "softshrink": ("lambd",),
    "hardtanh": ("min_val", "max_val"),
    "softplus": ("beta", "threshold"),
    "softmin": ("dim",),
    "glu": ("dim",),
    # The branches of a factorisation. **Discarded, the result is not an
    # exception but quietly a different answer** — `qr(mode="complete")` produces
    # the reduced form and `svd(full_matrices=False)` produces the full one.
    "qr": ("mode",),
    # **`svd` is `torch.svd` here, not `torch.linalg.svd`.** The row said
    # `full_matrices` and borch.ts's `svd` takes `some`, which is its *opposite* —
    # so `svd(full_matrices=False)` would have been forwarded into `some` and
    # produced the full form for a caller asking for the reduced one. Silently: the
    # values in the overlapping block agree and only the shape parts.
    # `linalg.svd` keeps `full_matrices` and is a different function.
    "svd": ("some", "compute_uv"),
    "linalgSvd": ("full_matrices",),
    # Keyword arguments of the composite layers.
    # **`keepdim` was missing and the seat behind it existed.** borch.ts's `vectorNorm`
    # takes it third; left out of this row it was dropped on the way and the answer came
    # back a rank short — a shape, not an exception.
    "vector_norm": ("ord", "dim", "keepdim"),
    # **`dim` and `keepdim` were missing for the same reason `vector_norm`'s was**, and
    # `matrix_norm(A, "fro", (-2, -1), True)` came back a scalar where torch keeps the
    # two axes as ones.
    "matrix_norm": ("ord", "dim", "keepdim"),
    "matrix_rank": ("tol",),
    # `linalg.norm` is routed to borch.ts's **namespace** function rather than the `norm`
    # method, so its row is torch's `linalg.norm` and not `torch.norm`'s. The two differ
    # in the first name — `ord` against `p` — and in what they compute.
    "linalg.norm": ("ord", "dim", "keepdim", "dtype"),
    # The other four that reach borch.ts's `linalg` namespace rather than a method.
    # Their rows are torch's `linalg` argument names — the tensor methods next to them
    # take neither the same names nor the same count.
    "linalg.lu": ("pivot",),
    "linalg.lu_solve": ("pivots", "b", "left", "adjoint"),
    "linalg.tensorsolve": ("b", "dims"),
    "linalg.lstsq": ("b", "rcond", "driver"),
    "vander": ("N",),
    "vecdot": ("other", "dim"),
    "eigvalsh": ("UPLO",),
    "solve_triangular": ("b", "upper", "left", "unitriangular"),
    # Normalisation and transposed convolution. borch.ts's argument order.
    "group_norm": ("num_groups", "eps"),
    "instance_norm": ("eps",),
    "dropout": ("p", "training"),
    "rms_norm": ("normalized_shape", "eps"),
    "conv_transpose1d": ("weight", "bias", "stride", "padding"),
    "conv_transpose2d": ("weight", "bias", "stride", "padding"),
    "conv_transpose3d": ("weight", "bias", "stride", "padding"),
    "roll": ("shifts", "dims"),
    "diff": ("n", "dim", "prepend", "append"),
    # Where the names diverge — Python has `rounding_mode` and JavaScript has
    # `roundingMode`. `_SIGNATURE` writes **torch's name** and follows borch.ts's
    # positions.
    "div": ("other", "rounding_mode"),
    "dist": ("other", "p"),
    "bincount": ("weights", "minlength"),
    "cholesky": ("upper",),
    "diag": ("diagonal",),
    "diagflat": ("offset",),
    "allclose": ("other", "rtol", "atol", "equal_nan"),
    "median": ("dim", "keepdim"),
    "gather": ("dim", "index", "sparse_grad"),
    "index_select": ("dim", "index"),
    "narrow": ("dim", "start", "length"),
    "movedim": ("source", "destination"),
    "cat": ("dim",),
    "stack": ("dim",),
    "unbind": ("dim",),
    "conv1d": ("weight", "bias", "stride", "padding"),
    "conv2d": ("weight", "bias", "stride", "padding"),
    "conv3d": ("weight", "bias", "stride", "padding"),
    "max_pool1d": ("kernel_size", "stride"),
    "max_pool2d": ("kernel_size", "stride"),
    "max_pool3d": ("kernel_size", "stride"),
    "avg_pool2d": ("kernel_size", "stride"),
    "adaptive_avg_pool2d": ("output_size",),
    "normalize": ("dim", "eps"),
    "cosine_similarity": ("other", "dim", "eps"),
    "layer_norm": ("dim", "eps"),
    "leaky_relu": ("negative_slope",),
    "one_hot": ("num_classes",),
    # **Every loss takes `reduction`.** The four common ones did not for a long
    # time, while the thirteen rare ones did from the start — the tutorials use
    # the default, so nobody asked.
    "smooth_l1_loss": ("target", "beta", "reduction"),
    # **`weight` is at the end on these three**, which is where borch.ts puts it —
    # torch's functional keeps it last too, after `reduction`, so the positions line
    # up and only the two legacy flags in front of `reduction` are the binding's
    # to absorb.
    "l1_loss": ("target", "reduction", "weight"),
    "mse_loss": ("target", "reduction", "weight"),
    # The same tail on the binary pair, now that borch.ts answers them: `weight`
    # scales the whole element and `pos_weight` the positive term alone. Both were a
    # refusal here and had no seat, so a caller who gave one by keyword met *does not
    # take keyword arguments* rather than the refusal that was meant for them.
    "bce_with_logits": ("target", "reduction", "weight", "pos_weight"),
    "binary_cross_entropy_with_logits": ("target", "reduction", "weight",
                                         "pos_weight"),
    # **These two had `ignore_index` missing from the middle** and it did not raise.
    # borch.ts is `nllLoss(target, ignoreIndex = -100, reduction = "mean")`, so
    # `nll_loss(x, t, reduction="none")` handed the string `"mean"`… no: it handed
    # `"none"` to `ignoreIndex`, which becomes `Tensor.full([], "none")` — NaN. Every
    # row then compares unequal to NaN, `keep` is false everywhere, and `mean` divides
    # by zero. **Twelve golden cases came back `nan` with no error anywhere.**
    #
    # A slot whose type is wide enough to swallow the wrong value is the quietest of
    # the positional failures: a string into a string slot is invisible, a string into
    # a *number* slot is invisible too when the number is only ever compared.
    "nll_loss": ("target", "ignore_index", "reduction", "weight"),
    "cross_entropy": ("target", "ignore_index", "reduction", "label_smoothing",
                      "weight"),
    "huber_loss": ("target", "delta", "reduction", "weight"),
    "interpolate": ("scale_factor",),
    # Boolean reductions and counting. **The dimension itself was missing for a
    # long time** — passing one had it quietly discarded and the whole tensor
    # reduced.
    "all": ("dim", "keepdim"),
    "any": ("dim", "keepdim"),
    "count_nonzero": ("dim",),
    # `nonzero(as_tuple=True)` — borch.ts takes it positionally, so the name has to
    # be here or the call is refused for having a keyword.
    "nonzero": ("as_tuple",),
    "kthvalue": ("k", "dim", "keepdim"),
    "quantile": ("q", "dim", "keepdim", "interpolation"),
    "cumulative_trapezoid": ("dim",),
    "diagonal": ("offset",),
    "expand": ("shape",),
    "unflatten": ("dim", "sizes"),
    # **`quantile` was in this table twice**, thirteen lines apart and with the same
    # value, so the second silently won and the first was dead. Identical, nothing
    # diverged; the next edit to either one would have. One row now, up with
    # `kthvalue`.
    #
    # **`squeeze` was the same, and it was still here when this was written** — the
    # note above it did not stop the next one, because a note is not a check. Both
    # copies went out with the twenty-three, and the reachability test is what a
    # third one meets.
    "add_": ("other", "alpha"),
    "sub_": ("other", "alpha"),
    "add": ("other", "alpha"),
    "sub": ("other", "alpha"),
    # Shape and indexing. borch.ts's argument order, with torch's names placed
    # into those slots.
    "as_strided": ("size", "stride", "storage_offset"),
    "as_strided_": ("size", "stride", "storage_offset"),
    "as_strided_scatter": ("src", "size", "stride", "storage_offset"),
    "select_scatter": ("src", "dim", "index"),
    "slice_scatter": ("src", "dim", "start", "end", "step"),
    "diagonal_scatter": ("src", "offset", "dim1", "dim2"),
    "diag_embed": ("offset", "dim1", "dim2"),
    "masked_scatter": ("mask", "source"),
    "masked_scatter_": ("mask", "source"),
    "index_reduce": ("dim", "index", "source", "reduce", "include_self"),
    "scatter_reduce": ("dim", "index", "src", "reduce", "include_self"),
    "put": ("index", "source", "accumulate"),
    "renorm": ("p", "dim", "maxnorm"),
    # The addmm family. In torch `beta`, `alpha` and `value` are **keyword-only**
    # (they sit after the `*`), so the cases always call them by name.
    "addmm": ("mat1", "mat2", "beta", "alpha"),
    "addmm_": ("mat1", "mat2", "beta", "alpha"),
    "addbmm": ("batch1", "batch2", "beta", "alpha"),
    "addbmm_": ("batch1", "batch2", "beta", "alpha"),
    "baddbmm": ("batch1", "batch2", "beta", "alpha"),
    "baddbmm_": ("batch1", "batch2", "beta", "alpha"),
    "addmv": ("mat", "vec", "beta", "alpha"),
    "addmv_": ("mat", "vec", "beta", "alpha"),
    "addr": ("vec1", "vec2", "beta", "alpha"),
    "addr_": ("vec1", "vec2", "beta", "alpha"),
    "addcmul": ("tensor1", "tensor2", "value"),
    "addcmul_": ("tensor1", "tensor2", "value"),
    "addcdiv": ("tensor1", "tensor2", "value"),
    "addcdiv_": ("tensor1", "tensor2", "value"),
    # Top-level linear algebra. borch.ts's argument order.
    "cholesky_solve": ("input2", "upper"),
    "cholesky_inverse": ("upper",),
    "triangular_solve": ("A", "upper", "transpose", "unitriangular"),
    "orgqr": ("input2",),
    "ormqr": ("tau", "other", "left", "transpose"),
    # **`B` and `X` sit behind `largest` over there and in front of it in torch.**
    # torch's order is `(A, k, B, X, …, largest, …)`; borch.ts keeps the two it had
    # first and adds the pair at the end, so this table is what puts each word where
    # it means the same thing.
    "lobpcg": ("k", "largest", "B", "X"),
    "svd_lowrank": ("q", "niter", "M"),
    "pca_lowrank": ("q", "center", "niter"),
    # Statistics. borch.ts's argument order.
    "histc": ("bins", "min", "max"),
    "histogram": ("bins", "range", "weight", "density"),
    "mode": ("dim", "keepdim"),
    "nanmedian": ("dim", "keepdim"),
    "gradient": ("spacing", "dim", "edge_order"),
    "nonzero_static": ("size", "fill_value"),
}

# **Places that take a whole list.** `permute([0,2,1])` reaches JavaScript as a
# single array, while Python also calls it as `permute(0, 2, 1)`. The scattered
# arguments have to be gathered — ungathered, it raises
# `order.map is not a function`.
_GATHERS = frozenset(("permute", "reshape", "view", "broadcast_to"))

# **Ones that take variadic arguments.** borch.ts writes `expand(...sizes)` and
# wants scattered numbers rather than an array — exactly the opposite of
# `_GATHERS`. Python calls it both ways, so it is unrolled here.
_SPREADS = frozenset(("expand", "tile", "repeat"))


def camel(name):
    """`masked_select` becomes `maskedSelect`. The letter after an underscore is
    raised.

    **A trailing underscore survives.** `zero_` means an in-place operation and
    borch.ts uses the same name — split naively it becomes `zero`, which does not
    exist.

    **Aliases are looked up after the underscore is stripped.** `absolute` is in
    the table and `absolute_` is not, so the alias did not fire and `absolute_`
    went across unchanged — borch.ts has only `abs_`, so that name does not
    exist. An in-place version has to follow its alias too.
    """
    tail = "_" if name.endswith("_") and not name.endswith("__") else ""
    bare = name[:-1] if tail else name
    if bare in _RENAME:
        return _RENAME[bare] + tail
    head, *rest = bare.split("_")
    return head + "".join(p[:1].upper() + p[1:] for p in rest) + tail


# Names whose borch.ts side **does not even declare** an argument. Asked once
# and remembered.
_NULLARY = {}


def refuse_if_nullary(js_name, fn, count):
    """**Stop when the other side does not accept an argument being passed.**

    This was a structural hole in the binding. JavaScript discards surplus
    arguments silently, so handing `sum(dim=1)` to borch.ts's `sum()` produces
    **a full sum that ignores the dimension, as a value.** No exception and no
    warning.

    The golden could not see it either. There was a `grad::sum(dim)` case, but it
    folded the result to a scalar and looked only at the gradient — and the
    gradients of `sum(dim=1).sum()` and `sum().sum()` are **both all ones**, so
    the wrong dimension gave the same answer. It was a case in name only.

    So rather than fix one name, the class is blocked. Only what **the source
    itself says** has an empty argument list is caught — anything with a default
    (`softmax(dim = -1)`) keeps the name in the list and does not appear here.
    Measured, four of this table were caught.
    """
    if not count:
        return
    known = _NULLARY.get(js_name)
    if known is None:
        src = fn.toString()
        head = src[:src.find(")") + 1]
        known = head.endswith("()")
        _NULLARY[js_name] = known
    if known:
        raise TypeError(
            f"borch.ts's `{js_name}` takes no arguments, but {count} were passed.\n"
            f"  Left alone they are silently ignored and **a different value** comes out.\n"
            f"  This name has to be spelled out by hand in `_ops.py`.")


def _arg(a):
    """A tensor becomes its handle, a list becomes a JS array, the rest goes
    through unchanged."""
    if isinstance(a, Tensor):
        return a._h
    if isinstance(a, (list, tuple)):
        return _js_list(a)
    # **A dtype name is unwrapped before crossing.** `_DType` subclasses `str`
    # and that `str()` is `torch.float32`, so passing it through hands borch.ts a
    # name it does not know — over there only `"float32"` exists. This path was
    # first taken when reductions gained `dtype=`.
    if isinstance(a, _DType):
        return a.plain
    return a


def positional(name, args, kw):
    """**Unroll keyword arguments into positions.**

    JavaScript has no keyword arguments. Discarded, an `undefined` travels down
    into a shader and WGSL refuses it at parsing — not quietly wrong, but the
    cause surfaces a long way away.

    Trailing `undefined`s are trimmed. borch.ts's defaults (`stride = kernel`)
    have to survive, and passing `undefined` explicitly leaves that slot
    unfilled.
    """
    if not kw:
        out = list(args)
    else:
        order = _SIGNATURE.get(name)
        if order is None:
            raise TypeError(
                f"`{name}` does not take keyword arguments (got: {sorted(kw)})\n"
                f"  If it should, write the positional order into `_SIGNATURE`.")
        out = list(args)
        # **A slot given twice is a refusal, not a preference.** Filling `out[i]`
        # unconditionally let `x.quantile(0.5, q=0.9)` through and used the keyword,
        # where torch raises `TypeError: got multiple values for argument 'q'`. The
        # core is a real Python signature so Python refuses it there; only this path
        # answered — which is the shape this repository spends its checks on, a call
        # torch declines and we oblige.
        clash = [key for i, key in enumerate(order) if i < len(args) and key in kw]
        if clash:
            raise TypeError(
                f"{name}() got multiple values for argument '{clash[0]}'")
        # **A keyword with no seat in the row was dropped without a word**, which is
        # the same silence as a surplus positional in JavaScript and the thing this
        # whole table exists to prevent. `avg_pool2d(x, 2, dilation=2)` — an argument
        # torch does not have and answers with a `TypeError` — came back as an
        # ordinary pooling here. torch's wording, so a caller meets the same sentence.
        stray = [key for key in kw if key not in order]
        if stray:
            raise TypeError(
                f"{name}() got an unexpected keyword argument '{sorted(stray)[0]}'")
        for i, key in enumerate(order):
            if key in kw:
                while len(out) <= i:
                    out.append(None)
                out[i] = kw[key]
    while out and out[-1] is None:
        out.pop()
    # Gather scattered dimension numbers into one array:
    # `permute(0, 2, 1)` becomes `permute([0,2,1])`.
    if name in _GATHERS and all(isinstance(a, int) for a in out):
        out = [list(out)]
    # And the reverse — scatter what arrived as one array:
    # `expand([2,3])` becomes `expand(2, 3)`.
    elif name in _SPREADS and len(out) == 1 and isinstance(out[0], (list, tuple)):
        out = list(out[0])
    return [_arg(a) for a in out]


# Ones that live **only in borch.ts's binary table.** They are not attached as
# methods and are called as `x.binary("maximum", y)` — only the unary table
# becomes methods automatically.
_BINARY_ONLY = frozenset((
    "maximum", "minimum", "atan2", "hypot", "copysign", "logaddexp",
    "logaddexp2", "xlogy", "heaviside", "ldexp", "pow",
    "eq", "ne", "lt", "le", "gt", "ge",
    "logical_and", "logical_or", "logical_xor",
    "bitwise_and", "bitwise_or", "bitwise_xor",
    "bitwise_left_shift", "bitwise_right_shift", "gcd", "lcm", "nextafter",
    "arctan2",
))


# dtypes that keep a name and **stop when used.** The core's table and the
# core's reason — without the name, `dtype=torch.half` stops with the wording a
# typo produces.
_ABSENT_DTYPE_NAMES = {
    "double": ("float64", "float32"), "float64": ("float64", "float32"),
    "int": ("int32", "int64"), "int32": ("int32", "int64"),
    "half": ("float16", "float32"), "float16": ("float16", "float32"),
    "bfloat16": ("bfloat16", "float32"),
    "short": ("int16", "int64"), "int16": ("int16", "int64"),
    "chalf": ("complex32", "complex64"), "complex32": ("complex32", "complex64"),
}


def __getattr__(name):
    """A name the module does not have is forwarded to **the first argument's
    method.**

    torch's own rule that `torch.exp(x)` and `x.exp()` are the same thing.

    **`out=` is attached here too.** Wrapping only the hand-written names left
    out everything coming through this door — `exp`, `matmul` and the rest. A
    name produced in two places gets fixed in one.
    """
    got = _resolve_name(name)
    if callable(got) and not name.startswith("_"):
        from borch import _TAKES_OUT, _TAKES_OUT_TUPLE
        if name in _TAKES_OUT or name in _TAKES_OUT_TUPLE:
            def with_out(*args, _fn=got, _n=name, **kwargs):
                out = kwargs.pop("out", None)
                return _out(_fn(*args, **kwargs), out, _n)
            with_out.__name__ = name
            return with_out
    return got


def _resolve_name(name):
    if name.startswith("_"):
        raise AttributeError(name)
    # The dtype names. Putting `bool` in the module globals shadows the Python
    # builtin, so it is served from here.
    if name in ("bool", "float32", "int64"):
        from ._base import _DType
        return _DType(name)
    # **A dtype alias is a dtype, not a function.** These three are also Tensor
    # methods, so letting them fall through produced a function forwarding to
    # `x.float()`, and `dtype=torch.float` received that function and stopped
    # somewhere unrelated. The core met the same place for the same reason.
    #
    # Only one of the three is a real dtype here. `torch.double` is float64 and
    # WebGPU shaders have no double precision. `torch.int` is **int32** (long is
    # int64) and the integer slots were gathered into int64 alone. The names stay
    # and stop when used — "absent" and "misspelled" have to say different
    # things.
    if name == "float":
        from ._base import _DType
        return _DType("float32")
    if name in _ABSENT_DTYPE_NAMES:
        from borch._base import _AbsentDtype
        return _AbsentDtype(*_ABSENT_DTYPE_NAMES[name])
    # `max` and `min` are served from here for the same reason, exactly as
    # written above.
    if name in _EXTREME:
        return _EXTREME[name]
    # The other spellings of the comparisons — forwarded under the table's
    # name.
    if name in _COMPARE_ALIAS:
        return __getattr__(_COMPARE_ALIAS[name])
    # **An in-place version has to go through the Python tensor's door.** Going
    # straight to the JS handle from here breaks two things: it stops on names
    # borch.ts has no in-place version of (`gcd_`, `clampMax_`), and even for the
    # ones it has it returns **a new Python tensor**, making
    # `torch.detach_(y) is y` false. `Tensor.__getattr__` is where both of those
    # are handled, so it is forwarded there.
    if name.endswith("_") and not name.endswith("__"):
        def call(x, *args, **kw):
            return getattr(wrap(x), name)(*args, **kw)
        call.__name__ = name
        return call

    js_name = camel(name)

    if name in _BINARY_ONLY:
        def call(a, b, *rest):
            return guarded(handle(a).binary, js_name, handle(b))
        call.__name__ = name
        return call

    # **Asked here, not when it is called.** Unasked, `__getattr__` hands out a
    # function for any name at all, and then `hasattr(torch, "compile")` is
    # **always true.** Code branching on a feature's existence
    # (`if hasattr(torch, "compile"): …`) takes the wrong branch and the error
    # arrives much later at the call. The prototype can be asked because borch.ts
    # attaches everything, down to the unary names from the table, through
    # `Object.defineProperty(Tensor.prototype, …)`.
    if getattr(_PROTO, js_name, None) is None:
        # **What is asked and what is said have to match.** What was looked at
        # here is `Tensor.prototype`, not all of borch.ts. Saying "borch.ts does
        # not have it" about a name that exists as a module function is false,
        # and it sends whoever is looking off after something that is not there.
        # `keep_alive` was caught exactly that way — present in borch.ts and
        # reported absent.
        if getattr(_ts, js_name, None) is not None:
            raise AttributeError(
                f"`{js_name}` is in borch.ts as a **module function**, not a method on "
                f"tensors (Python name `{name}`). This binding has not bridged it yet.")
        raise AttributeError(
            f"borch.ts does not have `{js_name}` (Python name `{name}`)")

    def call(x, *args, **kw):
        h = handle(x)
        fn = getattr(h, js_name, None)
        if fn is None:
            raise AttributeError(
                f"borch.ts does not have `{js_name}` (Python name `{name}`)")
        laid = positional(name, args, kw)
        refuse_if_nullary(js_name, fn, len(laid))
        return guarded(fn, *laid)

    call.__name__ = name
    return call


# ── ones whose first argument is not a tensor. Only these are written out. ──

def arange(*args, **kw):
    """`arange(n)` · `arange(a, b)` · `arange(a, b, step)`.

    For a while **borch.ts's `arange` took a count only.** Passing three had the
    first argument, 0, read as the count and produced an empty tensor, and 90
    cases collapsed where that empty tensor met a `reshape` — the failure read
    `shape '[3,3]' is invalid for input of size 0`, two steps from the cause.
    During that time the other two forms were built here **by multiplying and
    adding**, which is a different computation whose rounding accumulates
    differently. The other side takes all three now.
    """
    _no_out(kw.get("out"))
    if len(args) == 1:
        start, stop, step = 0, args[0], 1
    elif len(args) == 2:
        (start, stop), step = args, 1
    else:
        start, stop, step = args
    return _made(_ts.Tensor.arange(start, stop, step), kw)


def _shape_of(shape):
    """Takes both `zeros(2, 3)` and `zeros([2, 3])` — as torch does."""
    if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
        return _js_list(shape[0])
    return _js_list(shape)


def _dtype_to_make(dt):
    """Take **the name borch.ts knows** out of the dtype a factory received.

    `_DType` subclasses a string, but its `str()` prefixes `torch.`, so it cannot
    be used as-is — the inner name is `plain`. A dtype that has a name and no
    storage (the core's `_AbsentDtype`) **stops here with its own wording**, and
    reading `.np` is that door.

    **It must not be named `_dtype_name`.** That name already exists in this
    file (the promotion table uses it), and that one only extracts a name and
    must not stop. Written under the same name at first, Python took the later
    definition and **a different function was quietly called** — with no error
    and no warning, `dtype=torch.int` went through.
    """
    from ._base import _DType

    if isinstance(dt, _DType):
        return dt.plain
    if isinstance(dt, str):
        return str(dt)
    _ = dt.np
    return dt.name


def _kept(t, kw):
    """Apply `dtype=` and `requires_grad=` to **a tensor of ours that already
    exists.**

    The same rules as `_made` and a different input — that one takes a JS handle
    and this one takes our tensor. The decision is gathered into one line here so
    the rules are not written twice.
    """
    dt = kw.get("dtype")
    if dt is not None:
        t = t.to(_dtype_to_make(dt))
    if kw.get("requires_grad"):
        t.requires_grad_(True)
    return t


def _made(out, kw):
    """**Actually apply** the `dtype=` and `requires_grad=` a factory received.

    **Until this existed, both were quietly discarded into `**kw`.**
    `zeros(2, dtype=torch.int64)` produced float32 — the values are zero and
    therefore right, only the dtype is wrong, so comparing values does not catch
    it. The golden had no `zeros(..., dtype=)` case at all, so nobody asked. It
    surfaced while pinning the dtype aliases as cases.

    **Gathering it into one place is the point.** `zeros`, `ones`, `full`, `eye`
    and `linspace` each carried the same defect, and left as five copies the next
    fix reaches one of them.

    **`device=` was the same defect one argument over**, and it is handled here for
    the same reason. `zeros(2, device="cuda")` came back a tensor on the only device
    there is, with no exception — the values right and the claim about where they
    are false. The core's `_only_cpu` is the rule, so the two sides cannot part on
    which devices exist.
    """
    # Imported here rather than at the top: `_ops` is loaded while `borch` is still
    # half initialised in Pyodide, and the note beside `_fft` records what a
    # top-level import of the parent costs there.
    from borch._base import _only_cpu

    _only_cpu("factory", kw.get("device"))
    t = wrap(out)
    dt = kw.get("dtype")
    if dt is not None:
        t = t.to(_dtype_to_make(dt))
    if kw.get("requires_grad"):
        t.requires_grad_(True)
    return t


def zeros(*shape, **kw):
    _no_out(kw.get("out"))
    return _made(_ts.Tensor.zeros(_shape_of(shape)), kw)


def ones(*shape, **kw):
    _no_out(kw.get("out"))
    return _made(_ts.Tensor.ones(_shape_of(shape)), kw)


def full(shape, value, **kw):
    _no_out(kw.get("out"))
    return _made(_ts.Tensor.full(_js_list(shape), float(value)), kw)


def eye(n, m=None, **kw):
    _no_out(kw.get("out"))
    return _made(_ts.Tensor.eye(n, n if m is None else m), kw)


def cat(parts, dim=0):
    return wrap(_ts.Tensor.cat(_js.Array.from_([p._h for p in parts]), dim))


def stack(parts, dim=0):
    return wrap(_ts.Tensor.stack(_js.Array.from_([p._h for p in parts]), dim))


class scope:                                             # noqa: N801
    """`with L.scope():` — GPU buffers made inside are released on the way out.

    **A training loop does not run without it.** One step makes thousands of
    intermediate buffers, and neither Python's nor JavaScript's garbage
    collection releases GPU memory in time. The sister library exposes the same
    name for the same reason.

    ## To carry a result out, use `keep()`

        with torch.scope() as s:
            loss = s.keep(criterion(model(x), y))
        print(loss.item())          # alive outside the scope

    **A tensor made without `keep()` dies the moment the block ends.** Using it
    stops, and that is right — otherwise it quietly reads whatever the next
    allocation wrote over it.

    Without this, Python **could carry nothing out of a scope** (measured). The
    sister library had both `scope(body, () => [t])` and `keepAlive(t)`; this
    binding had neither, so using `with` made even a training loop's loss
    unreadable.
    """

    def __enter__(self):
        # **A fresh one per scope.** Reusing a single list puts a nested inner
        # scope's entries on the outer one's, and closing the outer scope tries
        # to hand over buffers that are already dead.
        self._kept = []
        _ts.device().beginScope()
        return self

    def keep(self, t):
        """Leave this scope and **pass into the enclosing one.** Hands back what
        it was given.

        Different from `keep_alive()`. This one is released when the enclosing
        scope closes; that one is released by no scope at all. Using
        `keep_alive()` on an intermediate value accumulates it every step.
        """
        if not isinstance(t, Tensor):
            raise TypeError(
                f"scope.keep takes a tensor — got {type(t).__name__}")
        # Nothing on the host needs keeping. A scope releases GPU buffers only,
        # and Python collects those values on its own — `raw` simply throws on a
        # CPU tensor.
        if str(handle(t).device) != "cpu":
            self._kept.append(t)
        return t

    def __exit__(self, *exc):
        # `to_js` makes a real JS array. Passed as a proxy, `new Set(keep)` over
        # there quietly builds an empty set without seeing what was given.
        buffers = _to_js([handle(t).raw for t in self._kept])
        self._kept = []
        _ts.device().endScope(buffers)
        return False


def keep_alive(t):
    """Keep it alive **forever**, whatever scope closes. Parameters and optimiser
    state use this.

    **A different thing** from `scope().keep()`. That one leaves the current
    scope for the enclosing one and is released when the enclosing one closes.
    This one is released by no scope, so using it on an intermediate value
    accumulates every step until training collapses on memory.

    A tensor on the host is handed straight back — a scope releases GPU buffers
    only.
    """
    if not isinstance(t, Tensor):
        raise TypeError(
            f"keep_alive takes a tensor — got {type(t).__name__}")
    _ts.keepAlive(handle(t))
    return t


def memory():
    """What is held right now. **Where the benchmark measures leaks.**

    The sister library called `js.tf.memory()` directly, which ties the
    measurement to TF.js and makes the same benchmark unrunnable against another
    implementation. Asked of the library, it answers whatever is underneath.
    """
    got = _ts.device().memory
    return {"tensors": int(got.tensors), "bytes": int(got.bytes)}


def pooled():
    """Buffers released and waiting in the pool for their next use. **Exactly
    what `memory()` leaves out on purpose.**

    The two functions **ask different questions.** That one asks "is it leaking";
    this one asks "how much is held". A pooled buffer is held and is not leaking,
    so counting it as a leak reads something that is not a leak as one.

    Leaving the pool out over there is right, and **without this nobody could ask
    the real footprint.** Measured, with the benchmark running three batch sizes
    in one session, `memory()` answered 269.7MB while the pool held 1,699.6MB.
    """
    got = _ts.device().pooled
    return {"count": int(got.count), "bytes": int(got.bytes)}


def empty_cache():
    """Empty the pool and answer with what came back as `{"count", "bytes"}`.

    **This is not `torch.cuda.empty_cache()`.** Two reasons for not using that
    name.

    One is principle — this library uses a GPU and is not CUDA, and the `cuda`
    name is kept as the place where `is_available()` answers false.

    The other is more practical. Textbook code uses that function like this.

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    `is_available()` is false, so the inside is **never called.** A name chosen
    for compatibility becomes a dead line inside compatibility code — the name
    matches and has no effect. It is kept in the same grain as `backend()`,
    `cache_get` and `fetch_cached`, the names that exist only in a browser.

    ## When to call it — **usually never**

    In training that repeats the same shapes the pool settles at the working set
    (measured: 49 to 49 over ten steps). Making them fresh each time is the cost,
    so the pool is doing the right thing there.

    It grows **when the shapes change.** The pool is split by size, so a batch-16
    buffer cannot serve batch 32.

        re-running with a different batch size · moving from training (large
        batches) to evaluation (small ones) · uploading a dataset once, large,
        and training on small shapes

    In a browser the tabs share GPU memory, so that matters more than it does on
    a desktop.
    """
    freed = _ts.device().emptyCache()
    return {"count": int(freed.count), "bytes": int(freed.bytes)}


# ── the three that measure cost. Here for `memory()`'s reason ───────────────
#
# **The golden looks only at values.** Leaking a buffer every step, or
# dispatching twice as many kernels, leaves the values right and the whole table
# green. The check that asks about that (`tests/browser/cost.py`) has to be able
# to read these numbers from outside, and letting it dig into the handle over
# there ties the measurement to borch.ts's internal shape — the lesson from
# building `memory()`.

def dispatches():
    """How many kernel dispatches so far. **Only differences mean anything** —
    the absolute number depends on the session."""
    return int(_ts.device().dispatches)


def submits():
    """How many submissions to the queue. More than one per step means something
    in the middle is waiting on the GPU."""
    return int(_ts.device().submits)


def last_scope():
    """The count from the most recently closed scope. **A non-zero `survived` is
    the leak.**"""
    got = _ts.device().lastScope
    return {"freed": int(got.freed), "survived": int(got.survived)}


class no_grad:                                           # noqa: N801
    """`with L.no_grad():`.

    borch.ts's `noGrad` **takes a function** — `noGrad(() => …)`. Python uses
    `with`, so the shape is changed here by opening and closing the underlying
    switch directly.
    """

    def __enter__(self):
        _ts.gradMode.enabled = False
        return self

    def __exit__(self, *exc):
        _ts.gradMode.enabled = True
        return False


class enable_grad:                                       # noqa: N801
    """**Turn it back on** inside `no_grad`. It has to nest, so the previous
    value is restored."""

    def __enter__(self):
        self._prev = bool(_ts.gradMode.enabled)
        _ts.gradMode.enabled = True
        return self

    def __exit__(self, *exc):
        _ts.gradMode.enabled = self._prev
        return False


class set_grad_enabled:                                  # noqa: N801
    """Takes on or off as a value. It changes at the call and is restored on the
    way out of the `with`."""

    def __init__(self, mode):
        self._prev = bool(_ts.gradMode.enabled)
        _ts.gradMode.enabled = bool(mode)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        _ts.gradMode.enabled = self._prev
        return False


def is_grad_enabled():
    return bool(_ts.gradMode.enabled)


class inference_mode:                                    # noqa: N801
    """**The same as `no_grad` here.** Real torch marks the tensors made inside,
    and imitating that mark would mean manufacturing our own "why can I not use
    this tensor" errors."""

    def __init__(self, mode=True):
        self._mode = bool(mode)
        self._prev = None

    def __enter__(self):
        self._prev = bool(_ts.gradMode.enabled)
        if self._mode:
            _ts.gradMode.enabled = False
        return self

    def __exit__(self, *exc):
        _ts.gradMode.enabled = self._prev
        return False


def is_inference(t):
    """**Always false** — the mark is never attached, so saying it is absent is
    the fact."""
    return False


def is_inference_mode_enabled():
    return False


# ── random state ────────────────────────────────────────────────────────────

_LAST_SEED = [0]


def initial_seed():
    return _LAST_SEED[0]


def seed():
    got = int(_np.random.SeedSequence().entropy % (2 ** 63))
    manual_seed(got)
    return got


def get_rng_state():
    """**Carries the state of both generators.** `randn` and `randperm` use the
    numpy one, while layer initialisation and dropout use the seed over in
    borch.ts — carrying one of them means restoring leaves the other where it
    was."""
    return {"numpy": dict(_rng.bit_generator.state),
            "ts": int(_ts.Tensor.dropoutSeed)}


def set_rng_state(state):
    if not isinstance(state, dict) or "numpy" not in state:
        raise RuntimeError("set_rng_state — only takes what `get_rng_state` returned")
    _rng.bit_generator.state = state["numpy"]
    _ts.Tensor.dropoutSeed = state["ts"]
    return None


# ── introspection ───────────────────────────────────────────────────────────

def is_tensor(x):
    return isinstance(x, Tensor)


def is_storage(x):
    """**Always false.** There is no Storage layer here."""
    return False


def is_floating_point(x):
    return str(handle(x).dtype) == "float32"


def is_signed(x):
    return str(handle(x).dtype) in ("float32", "int64")


def is_nonzero(x):
    h = handle(x)
    if int(h.size) != 1:
        raise RuntimeError(
            f"Boolean value of Tensor with {int(h.size)} elements is ambiguous")
    return bool(float(_np.asarray(x.numpy()).reshape(-1)[0]) != 0)


def is_same_size(a, b):
    return ([int(v) for v in handle(a).shape] == [int(v) for v in handle(b).shape])


def is_distributed(x):
    return False


def typename(x):
    if not isinstance(x, Tensor):
        return type(x).__name__
    kinds = {"float32": "FloatTensor", "int64": "LongTensor", "bool": "BoolTensor"}
    return "torch." + kinds.get(str(handle(x).dtype), "FloatTensor")


_PROMOTE_ORDER = ("bool", "int64", "float32")


def _dtype_name(t):
    return getattr(t, "name", str(t)).replace("torch.", "")


def promote_types(a, b):
    from . import _base
    names = [_dtype_name(t) for t in (a, b)]
    best = max(names, key=lambda n: _PROMOTE_ORDER.index(n)
               if n in _PROMOTE_ORDER else 0)
    return _base._DType(best)


# A dtype's **category.** `can_cast` looks at nothing else — precision is free
# and only narrowing the category is blocked (measured: `float64 → float32` is
# true). Written as an ordering table, complex was left out and **even complex to
# complex was false.** The core fixed the same place alongside.
_CATEGORY_OF = {"bool": 0, "int64": 1, "float32": 2, "float64": 2, "complex64": 3}


def can_cast(from_type, to_type):
    """**Category only** — bool < integer < float < complex."""
    a, b = (_CATEGORY_OF.get(_dtype_name(t), 2) for t in (from_type, to_type))
    return a <= b


# The dtype a category settles on. `bool + bool` stays bool; a Python `int` above
# a bool tensor rises to `int64`, not to some wider integer, because there is one
# width per category here.
_DEFAULT_OF_CATEGORY = {0: "bool", 1: "int64", 2: "float32", 3: "complex64"}


def _scalar_category(value):
    # **`complex` is a function in this module**, a thousand lines below — torch's
    # `complex(real, imag)`, which builds a tensor. So `isinstance(value, complex)`
    # asked whether `1.5` is an instance of a function and stopped with
    # `isinstance() arg 2 must be a type`, a message naming neither this function nor
    # the shadowing. Two golden cases caught it (`result_type(t, 1.5)` and
    # `result_type(1, 2.5)`); the other four never reached the line.
    #
    # Fourth shadowed builtin in a day, after `type` and `any` in the core's
    # `__init__` and `max` here. `bool` was already written `builtins.bool` when this
    # went in and the three beside it were not — the hazard was known at one line and
    # not at the next. All four are spelled out now: only `complex` is actually taken
    # today, and which ones are taken is not a fact this function should depend on.
    if isinstance(value, builtins.bool):
        return 0
    if isinstance(value, builtins.int):
        return 1
    if isinstance(value, builtins.complex):
        return 3
    return 2


def result_type(tensor, other):
    """The dtype two **operands** would produce. torch takes tensors here, not dtypes.

    `promote_types` above is the dtype-to-dtype question and torch keeps both; this
    one had no implementation on this side at all, because the core's public
    `result_type` was the internal numpy-dtype helper and nothing ever called it
    through the bridge. Six golden cases arrived with the core's repair and every one
    of them landed here as *borch.ts does not have `resultType`* — the bridge
    forwarding a name neither side had.

    **A Python number is weaker than a tensor.** At or below the tensor's category it
    takes the tensor's dtype, and only above it does it rise to that category's
    default — which is why an int tensor with a Python float is `float32`. That is
    the rule the arithmetic already follows; written as a second table it would drift
    from it, which is what the note beside `can_cast` records happening once already.
    """
    if isinstance(tensor, _DType) or isinstance(other, _DType):
        raise TypeError(
            "result_type() received an invalid combination of arguments — it takes "
            "tensors or numbers, not dtypes. `promote_types(a, b)` is the one that "
            "takes two dtypes.")

    def dtype_of(x):
        return _dtype_name(x.dtype)

    a, b = isinstance(tensor, Tensor), isinstance(other, Tensor)
    if a and b:
        return promote_types(_DType(dtype_of(tensor)), _DType(dtype_of(other)))
    if a or b:
        t, scalar = (tensor, other) if a else (other, tensor)
        tcat = _CATEGORY_OF.get(dtype_of(t), 2)
        scat = _scalar_category(scalar)
        return _DType(dtype_of(t) if scat <= tcat
                      else _DEFAULT_OF_CATEGORY[scat])
    return _DType(_DEFAULT_OF_CATEGORY[builtins.max(_scalar_category(tensor),
                                                    _scalar_category(other))])


def _out(result, out, name="op"):
    """torch's `out=` convention. **The core's rules and the core's wording.**

    That side writes back into a numpy array and this one into a borch.ts buffer.
    A different shape means a different element count, so `copyFrom` cannot do it
    and **the handle is swapped** — `_set_` does the same for the same reason.
    """
    if out is None:
        return result
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
    if not can_cast(result.dtype, out.dtype):
        names = {"bool": "Bool", "int64": "Long", "float32": "Float",
                 "float64": "Double", "complex64": "ComplexFloat"}
        raise RuntimeError(
            f"result type {names.get(_dtype_name(result.dtype), 'Float')} can't be "
            f"cast to the desired output type "
            f"{names.get(_dtype_name(out.dtype), 'Float')}")
    want = tuple(result.shape)
    if tuple(out.shape) != want:
        import warnings as _w
        _w.warn(
            f"An output with one or more elements was resized since it had shape "
            f"{list(out.shape)}, which does not match the required output shape "
            f"{list(want)}.", UserWarning, stacklevel=3)
        out._h = handle(result.to(_dtype_name(out.dtype)))
        return out
    return out._write_back(result.to(_dtype_name(out.dtype)))


def get_default_dtype():
    from . import _base
    return _base._DType("float32")


def set_default_dtype(dt):
    """Accepted and not acted on — the storage is float32 throughout. Anything
    else is refused loudly."""
    if _dtype_name(dt) != "float32":
        raise RuntimeError(f"set_default_dtype({dt}) — the storage is float32 only")
    return None


class finfo:
    """`torch.finfo`. **It has to be a class** — a wrapper function gives the
    same values with a different type, and a check that looks only at names
    cannot see the difference. The same place as the core's."""

    def __init__(self, dt=None):
        info = _np.finfo(_np.float32)
        self.eps = float(info.eps)
        self.max = float(info.max)
        self.min = float(info.min)
        self.tiny = float(info.tiny)
        self.smallest_normal = float(info.tiny)
        self.resolution = float(info.resolution)
        self.bits = int(info.bits)
        self.dtype = "float32" if dt is None else _dtype_name(dt)


class iinfo:
    """A class for `finfo`'s reason."""

    def __init__(self, dt=None):
        info = _np.iinfo(_np.int64)
        self.max = int(info.max)
        self.min = int(info.min)
        self.bits = int(info.bits)
        self.dtype = "float32" if dt is None else _dtype_name(dt)





def linspace(start, end, count, **kw):
    _no_out(kw.get("out"))
    return _made(_ts.Tensor.linspace(start, end, count), kw)


# ── window functions ───────────────────────────────────────────────────────
#
# They take **a count**, not a tensor — forwarding to the first argument's
# method does not reach them, so they are written out here. borch.ts builds them
# on the CPU, and the `periodic` convention lives over there too.

# All five were **swallowing** `dtype=` and `requires_grad=` into `**kw`. When
# fourteen factories were gathered under `_made`, these five were outside the
# list — fix a list rather than a branch and the same defect survives next door.
# The core was swallowing them in the same place.
def bartlett_window(n, periodic=True, **kw):
    return _made(_ts.Tensor.bartlettWindow(n, periodic), kw)


def hann_window(n, periodic=True, **kw):
    return _made(_ts.Tensor.hannWindow(n, periodic), kw)


def hamming_window(n, periodic=True, alpha=0.54, beta=0.46, **kw):
    _no_out(kw.get("out"))
    return _made(_ts.Tensor.hammingWindow(n, periodic, alpha, beta), kw)


def blackman_window(n, periodic=True, **kw):
    return _made(_ts.Tensor.blackmanWindow(n, periodic), kw)


def kaiser_window(n, periodic=True, beta=12.0, **kw):
    """**`beta` is positional** — torch takes
    `kaiser_window(n, periodic, beta)`."""
    return _made(_ts.Tensor.kaiserWindow(n, periodic, beta), kw)


# **The random numbers come from one stream.** At first every call built a fresh
# `default_rng(0)`. The golden uses random numbers only in error cases — it looks
# at whether they throw — so identical values were never caught, and in that
# state a shuffling `DataLoader` produces **the same order every epoch.** From
# the caller's side shuffling is on and nothing shuffles, and nothing raises.
_rng = _np.random.default_rng(0)


def manual_seed(seed):
    """Re-seed. torch's name and torch's meaning.

    **The other side has to be seeded too.** The numpy generator here is used by
    `randn` and `randperm`, while layer initialisation and dropout use a
    different generator inside borch.ts — seeding only this one makes `randn`
    reproducible and the weights different every run. Those two are the first
    things somebody checking "same seed, same result" looks at, and with the
    first one reproducing they conclude it works and move on.

    All three implementations carried one defect of this kind, and the lazy-layer
    cases caught all three.
    """
    global _rng
    _rng = _np.random.default_rng(seed)
    _ts.nn.manualSeed(int(seed))
    # **The core's generator is seeded as well.** The seven that draw from a
    # distribution (`uniform_` and the rest) borrow the core's rules so there are
    # not two copies of them, which means they draw from the core's `_rng` too.
    # Seeding only here makes `randn` reproducible and `x.uniform_()` different
    # every time — the defect the comment above describes comes back **every time
    # a generator is added.**
    from borch import manual_seed as _core_seed
    _core_seed(int(seed))
    _LAST_SEED[0] = int(seed)
    return _rng


class Generator:
    """A stream of random numbers of its own, as in `random_split(..., generator=g)`.

    **The core's class carries the same correction** — it built a fresh
    `default_rng(seed)` on every `rng()` call, so two draws from one generator
    returned the same numbers where torch's advance. The two are kept in step by
    hand rather than shared because this side's `manual_seed` also reaches into
    borch.ts, and a subclass that inherited would hide that.
    """

    def __init__(self, device=None):
        self.seed = 0
        self._rng = _np.random.default_rng(0)

    def manual_seed(self, seed):
        self.seed = seed
        self._rng = _np.random.default_rng(seed)
        return self

    def initial_seed(self):
        return self.seed

    def rng(self):
        return self._rng


def _stream(generator):
    """The stream to draw from: the generator's own, or the global one. A generator
    must not disturb the global stream, which is what keeping them apart means."""
    return _rng if generator is None else generator.rng()


def _shaped(shape):
    return shape[0] if len(shape) == 1 and isinstance(shape[0], (list, tuple)) else shape


def randn(*shape, out=None, requires_grad=False, dtype=None, device=None,
          generator=None):
    """Normal random numbers. **borch.ts has none, so they are made here.**

    The golden uses this only in error cases, where it looks at **whether it
    throws** rather than at values (`L.randn(3, 4) @ L.randn(3, 2)`). Once a case
    asks about values, this belongs in borch.ts properly — what is here goes
    through the CPU once.

    **The `**kw` bag is gone, and `generator=` is why.** A bag is the only thing
    that can swallow a keyword: `randn(3, generator=g)` was accepted and the
    generator discarded, so two generators carrying the same seed produced
    different numbers while the call looked exactly like torch's. `_no_out`'s own
    docstring says the bags were removed for this reason; these two still had them.
    """
    _no_out(out)
    from borch._base import _only_cpu
    from ._base import tensor as _t

    _only_cpu("randn", device)
    return _t(_stream(generator).standard_normal(tuple(_shaped(shape))).astype("float32"),
              requires_grad=requires_grad)


def rand(*shape, out=None, requires_grad=False, dtype=None, device=None,
         generator=None):
    """**These two do not go through `_made`**, so the `device=` rule it carries did
    not reach them — `rand(2, device="cuda")` answered where `zeros` had stopped.
    Two factories under one rule and one of them outside it is the shape this file
    keeps finding."""
    _no_out(out)
    from borch._base import _only_cpu
    from ._base import tensor as _t

    _only_cpu("rand", device)
    return _t(_stream(generator).random(tuple(_shaped(shape))).astype("float32"),
              requires_grad=requires_grad)


def _no_out(out):
    """`out=` is **not swallowed quietly.** The core's gate for the core's reason.

    **It used to take the `**kw` bag, and that was the whole hazard.** A bag is the
    only thing that can swallow a keyword, so the gate and the hazard were the same
    parameter: forty seats here declared `**kw`, called this once, and thereby accepted
    every other unknown keyword in silence — `bernoulli(x, zzz=1)` went through where
    torch refuses.

    The bags are gone and `out` is a seat of its own, so this now takes the value.
    Python answers `unexpected keyword argument` for everything else, for free, in the
    wording torch uses for the same mistake — which is the half that cannot be
    imitated by hand, since torch has two wordings and picking one is wrong twice.
    """
    if out is not None:
        from borch._base import _unsupported
        _unsupported("`out=` (writing into a tensor you made beforehand)")


def randint(low, high=None, size=(), *, out=None, dtype=None,
            requires_grad=False, generator=None):
    """**`dtype` and `requires_grad` were falling into `**kw`.** torch declares both,
    keyword-only, and the label that came out here was whatever the body produced —
    `int64`, which is torch's default, so the values and the label agreed and nothing
    ever parted. Removing the bag is what asked the question; `_made` is the same seam
    the other factories already go through."""
    _no_out(out)
    from ._base import tensor as _t

    if high is None:
        low, high = 0, low
    made = _t(_stream(generator).integers(low, high, tuple(size)).astype("int64"))
    return _made(made, {"dtype": dtype, "requires_grad": requires_grad})


def randperm(n, *, out=None, dtype=None, requires_grad=False, generator=None):
    _no_out(out)
    from ._base import tensor as _t

    return _made(_t(_stream(generator).permutation(n).astype("int64")),
                 {"dtype": dtype, "requires_grad": requires_grad})


def einsum(spec, *operands):
    """borch.ts's `einsum` is a free function and takes the operands
    **spread**."""
    return guarded(_ts.einsum, spec, *[handle(t) for t in operands])


def as_tensor(data, dtype=None):
    from ._base import tensor as _t
    return data if isinstance(data, Tensor) else _t(data, dtype)


def from_numpy(arr):
    """**The values carry across; the memory cannot be shared.** torch shares
    storage with the numpy array, so editing one changes the other, and here the
    values live in a GPU buffer with nowhere for that to happen — the reason view
    propagation is refused.

    So it comes out the same as `tensor()`. It is not refused because textbooks
    use this name **to make a tensor**, not to make an alias; a divergence is
    still a divergence, so the golden has a place for it.
    """
    from ._base import tensor as _t
    return _t(arr)


def matrix_power(x, n):
    """**A negative exponent is a power of the inverse.** borch.ts does 1 and
    up.

    `A^-2 = (A^-1)^2`, so invert and then call with a positive one. Zero is the
    identity.
    """
    h = handle(x)
    if n == 0:
        return wrap(_ts.Tensor.eye(int(h.shape[0]), int(h.shape[0])))
    if n < 0:
        h = settle(h.inverse())._h
        n = -n
    return guarded(h.matrixPower, n)


def quantile(x, q, dim=None, out=None):
    """`q` may be one number or a list — borch.ts always takes a list."""
    _no_out(out)
    one = isinstance(q, (int, float))
    qs = [float(v) for v in ([q] if one else q)]
    out = guarded(handle(x).quantile, _to_js(qs))
    # **One number gives a scalar.** That is what torch does — an axis appears
    # only when asked with a list. borch.ts is always a list, so it is folded
    # here.
    return wrap(out._h.reshape(_js_list([]))) if one else out


def numel(x):
    """The element count. In borch.ts it is a **property** called `size` — a
    different name and a different shape."""
    return int(handle(x).size)


def _reduce_all(name):
    """A reduction with **no axis given.** torch flattens and produces one
    value.

    Over in borch.ts the axis defaults to 0, so forwarded as-is it produces **one
    per column.** The shape differs, so a value comparison does catch it, but
    until it does the state is "works, slightly odd".
    """
    def call(x, dim=None, keepdim=False, **kw):
        dim = kw.get("dim", dim)
        h = handle(x)
        if dim is None:
            h = h.reshape(_js_list([int(h.size)]))
            return guarded(getattr(h, camel(name)), 0)
        return guarded(getattr(h, camel(name)), dim, bool(kw.get("keepdim", keepdim)))
    call.__name__ = name
    return call


argmax = _reduce_all("argmax")
argmin = _reduce_all("argmin")


def _extreme(name):
    """The **three faces** of `max` and `min`. torch produces different things
    depending on the arguments.

    - `max(x)` → **one** overall maximum (not a pair)
    - `max(x, dim)` → a `(values, indices)` pair
    - `max(x, other)` → the **elementwise** maximum

    Three things under one name is a confusing place, and it is torch's contract,
    so it must not be tidied here. Tidying it stops textbook code from running.

    The third branch was missing and the first returned a pair. It surfaced only
    once the golden asked about the three branches separately — it is loud only
    when `x.max()` is converted to a scalar, and used in a comparison it becomes
    an elementwise comparison and quietly a different answer.
    """
    pair = {"max": "amax", "min": "amin"}[name]
    elementwise = {"max": "maximum", "min": "minimum"}[name]

    def call(x, dim=None, keepdim=False, other=None, **kw):
        dim = kw.get("dim", dim)
        other = kw.get("other", other)
        if isinstance(dim, Tensor):                # the `max(x, other)` form
            dim, other = None, dim
        h = handle(x)
        if other is not None:
            return wrap(guarded(h.binary, elementwise, handle(other)))
        if dim is None:
            return wrap(guarded(getattr(h, pair)))
        # One axis means a pair. `guarded` already names the fields, so it is
        # not wrapped again.
        return guarded(getattr(h, name), dim, bool(kw.get("keepdim", keepdim)))
    call.__name__ = name
    return call


# **`max` and `min` must not live at module scope.** They would shadow the
# Python builtins inside this file, and then a place sizing something with
# `max(a, b)` calls a tensor function — the symptom was GPU buffer allocation
# dying wholesale (128 `createBuffer` failures), a very long way from the cause.
# The same thing happened with `bool` and was written down there, and it was
# stepped on again anyway.
#
# `__getattr__` below hands these names out. From outside it looks like
# `L.max(x)`, and inside the builtins stay alive.
_EXTREME = {"max": _extreme("max"), "min": _extreme("min")}


def flatten(x, start_dim=0, end_dim=-1, **kw):
    """Fold axes. **borch.ts has none, so it is built from `reshape`** — the
    definition, directly.

    **The default is `start_dim=0`.** The `nn.Flatten` layer folds from 1 to
    leave the batch, but the `torch.flatten` function starts at 0. Copying the
    layer's default onto the function made `flatten(x)` leave the batch, and the
    shape differed.
    """
    h = handle(x)
    shape = [int(n) for n in h.shape]
    rank = len(shape)
    a = kw.get("start_dim", start_dim)
    b = kw.get("end_dim", end_dim)
    a = a + rank if a < 0 else a
    b = b + rank if b < 0 else b
    merged = 1
    for n in shape[a:b + 1]:
        merged *= n
    return guarded(h.reshape, _js_list(shape[:a] + [merged] + shape[b + 1:]))


def squeeze(x, *dim, **kw):
    """With no `dim`, torch removes **every axis of length 1.**

    **Several axes at once is torch's form too** — `x.squeeze(0, 2)` — and this
    took one. borch.ts's `squeeze` is variadic now and the tuple spelling
    (`squeeze((0, 2))`, which torch also takes) is unrolled here.
    """
    h = handle(x)
    if "dim" in kw:
        dim = (kw["dim"],)
    if len(dim) == 1 and isinstance(dim[0], (tuple, list)):
        dim = tuple(dim[0])
    if dim and dim[0] is None:
        dim = ()
    if dim:
        return guarded(h.squeeze, *[int(d) for d in dim])
    keep = [int(n) for n in h.shape if int(n) != 1]
    return guarded(h.reshape, _js_list(keep))


def sum(x, dim=None, keepdim=False, dtype=None, **kw):   # noqa: A001
    """borch.ts keeps the total sum (`sum()`) and the axis sum (`sumDim`) under
    **different names.**

    This place was quietly wrong. `sum(dim=1)` went to `sum()` and produced a
    scalar with the axis ignored, and with no exception nobody knew — until one
    rank-6 case caught it on shape.
    """
    dim = kw.get("dim", dim)
    keepdim = kw.get("keepdim", keepdim)
    dtype = kw.get("dtype", dtype)
    h = handle(x)
    # **`dtype=` applies to the total sum too.** borch.ts's `sum()` does not
    # take the argument, so the dtype is changed on both sides here — the same
    # rule: convert before going in, and pin the result as well.
    if dtype is not None:
        name = dtype.plain if isinstance(dtype, _DType) else str(dtype)
        cast = wrap(guarded(h.to, name.replace("torch.", "")))
        return wrap(guarded(handle(sum(cast, dim, keepdim)).to,
                            name.replace("torch.", "")))
    if dim is None:
        return guarded(h.sum)
    return guarded(h.sumDim, dim, bool(keepdim))


def norm(x, p=2, dim=None, keepdim=False, **kw):
    """borch.ts's `norm()` is one **whole-tensor** L2 and nothing else. The axis
    and the order are built here.

    Passing them only had them quietly discarded — `norm(dim=1)` produced the
    whole-tensor norm.
    """
    p = kw.get("p", p)
    dim = kw.get("dim", dim)
    keepdim = kw.get("keepdim", keepdim)
    # **`dtype=` is not discarded quietly.** torch converts and then computes,
    # and float32 is the only thing that can be asked for here, so stopping on
    # any other dtype is the answer — handing back float32 creates code that
    # believes it measured in double precision.
    if kw.get("dtype") is not None:
        x = wrap(x).to(_dtype_to_make(kw["dtype"]))
    h = handle(x)
    rank = len(h.shape)

    def _fold(t, how):
        """One reduction over `dim`, **which may be several axes or none.**

        borch.ts's `sumDim`/`amax`/`amin` take one axis, and this handed the tuple
        straight through — `norm(x, dim=(1, 2))` reduced axis `(1, 2)`, which
        JavaScript reads as neither and which came back the wrong rank. Folded one at
        a time from the back, so the earlier indices stay valid, and squeezed at the
        end when `keepdim` is off.

        **With no `dim` and `keepdim` on, torch keeps every axis as a 1** — a 2×2 asked
        for its Frobenius norm gives `(1, 1)` and not a scalar. That was the other
        half of the same hole: the whole-tensor branch ignored the flag.
        """
        axes = (list(range(rank)) if dim is None
                else [int(d) % rank for d in ((dim,) if isinstance(dim, int) else dim)])
        out = t
        for axis in sorted(axes, reverse=True):
            out = how(out, axis)
        if keepdim:
            return out
        return out

    def _shape_after(t):
        """`keepdim` off drops the folded axes; `_fold` already kept them."""
        if keepdim:
            return t
        axes = {int(d) % rank for d in
                (range(rank) if dim is None
                 else ((dim,) if isinstance(dim, int) else dim))}
        keep = [n for i, n in enumerate(handle(t).shape) if i not in axes]
        return wrap(guarded(handle(t).reshape, _js_list([int(n) for n in keep])))

    def _reduce(t, how):
        return _shape_after(wrap(_fold(t, how)))

    if p == 2:
        return _reduce(guarded(h.square), lambda o, a: handle(o).sumDim(a, True)).sqrt()
    if p == 1:
        return _reduce(guarded(h.abs), lambda o, a: handle(o).sumDim(a, True))
    if p == float("inf"):
        return _reduce(guarded(h.abs), lambda o, a: handle(o).amax(a, True))
    # The three below were opened alongside fixing the place where the core
    # quietly produced the 2-norm. Fixing the core alone makes **the three
    # diverge** — the golden asks all three implementations the same thing, so a
    # case only one of them can answer cannot go in at all.
    if p == float("-inf"):
        return _reduce(guarded(h.abs), lambda o, a: handle(o).amin(a, True))
    if p == 0:
        got = handle(x)
        return wrap(got.countNonzero() if dim is None
                    else got.countNonzero(dim)).float()
    if p in (None, "fro"):
        return norm(x, 2, dim, keepdim)
    if p == "nuc":
        # **The refusal said it needs an SVD, and borch.ts has had one.** `svdvals`
        # and `matrixNorm` are both over there and both asynchronous, which is what
        # `settle` exists for — `guarded` awaits through `run_sync`. The reason named
        # a computation rather than a gap, and the computation was one call away.
        #
        # torch's two checks, in torch's order and with torch's wording: the rank
        # first, then the count of axes. On a 1-D with no `dim` the axis list is one
        # long and torch still says *at least 2 dimensions*.
        axes = (tuple(range(rank)) if dim is None
                else tuple(int(d) for d in
                           ((dim,) if isinstance(dim, int) else dim)))
        if rank < 2:
            raise RuntimeError("linalg.matrix_norm: The input tensor A must have "
                               "at least 2 dimensions.")
        if len(axes) != 2:
            raise RuntimeError("linalg.matrix_norm: dim must be a 2-tuple. Got "
                               + " ".join(str(a) for a in axes))
        return wrap(guarded(h.matrixNorm, "nuc", _js_list(list(axes)),
                            bool(keepdim)))
    powed = wrap(guarded(handle(guarded(h.abs)).powScalar, float(p)))
    total = _reduce(handle(powed), lambda o, a: handle(o).sumDim(a, True))
    return wrap(handle(total).powScalar(1.0 / float(p)))


def transpose(x, dim0=None, dim1=None, **kw):
    """torch swaps two axes at any rank, and this builds the permutation.

    **The note here said borch.ts's `transpose()` is 2-D only and takes no axes.**
    That was true and stopped being true: that side takes `(dim0?, dim1?)` now.
    The permutation is still built on this side rather than handed over, because
    `swapdims` below shares it and the two spellings must not part.
    """
    dim0 = kw.get("dim0", dim0)
    dim1 = kw.get("dim1", dim1)
    h = handle(x)
    if dim0 is None:
        return guarded(h.transpose)
    rank = len(h.shape)
    a = dim0 + rank if dim0 < 0 else dim0
    b = dim1 + rank if dim1 < 0 else dim1
    order = list(range(rank))
    order[a], order[b] = order[b], order[a]
    return guarded(h.permute, _js_list(order))


def swapdims(x, dim0=None, dim1=None, **kw):
    return transpose(x, dim0, dim1, **kw)


# ── the ones torch offers under a **second name** ───────────────────────────
#
# All of them are combinations of what already exists. No names are added over
# in borch.ts — this is not more computation, it is more spellings that Python
# code calls, so it is Python's job.

def add(a, b, alpha=1, **kw):
    """`a + alpha·b`. A function rather than an alias **because the operator has
    no `alpha`.**"""
    _no_out(kw.get("out"))
    alpha = kw.get("alpha", alpha)
    return wrap(a) + (b if alpha == 1 else wrap(b) * alpha)


def sub(a, b, alpha=1, **kw):
    _no_out(kw.get("out"))
    alpha = kw.get("alpha", alpha)
    return wrap(a) - (b if alpha == 1 else wrap(b) * alpha)


def mul(a, b, out=None):
    _no_out(out)
    return wrap(a) * b


def div(a, b, rounding_mode=None, **kw):
    _no_out(kw.get("out"))
    mode = kw.get("rounding_mode", rounding_mode)
    out = wrap(a) / b
    if mode is None:
        return out
    if mode == "floor":
        return wrap(guarded(handle(out).unary, "floor"))
    if mode == "trunc":
        return wrap(guarded(handle(out).unary, "trunc"))
    raise RuntimeError(f"rounding_mode is one of None, 'floor', 'trunc': {mode!r}")


def floor_divide(a, b, out=None):
    _no_out(out)
    return wrap(guarded(handle(wrap(a)).floorDivide, handle(wrap(b))))


def remainder(a, b, out=None):
    """**The sign follows the divisor.** That is where it parts from `fmod`."""
    _no_out(out)
    a, b = wrap(a), wrap(b)
    return a - wrap(guarded(handle(a / b).unary, "floor")) * b


def fmod(a, b, out=None):
    """**The sign follows the dividend.** C's rule, and the opposite of
    `remainder`."""
    _no_out(out)
    a, b = wrap(a), wrap(b)
    return a - wrap(guarded(handle(a / b).unary, "trunc")) * b


def rsub(a, b, alpha=1):
    return wrap(guarded(handle(wrap(a)).rsub, handle(wrap(b)), alpha))


def t(x):
    """A 2-D transpose. **1-D and below are left alone** — that is what torch
    does."""
    h = handle(x)
    return wrap(h) if len(h.shape) < 2 else transpose(x, 0, 1)


def adjoint(x):
    return wrap(guarded(handle(wrap(x)).adjoint))


def moveaxis(x, source, destination):
    return wrap(guarded(handle(x).movedim, source, destination))


def broadcast_to(x, shape):
    """The name over in borch.ts is `expand` — it takes the axes **spread**."""
    return wrap(guarded(handle(x).expand, *[int(n) for n in shape]))


def _broadcast_shape(shapes):
    """Right-aligned, taking the larger of each axis — numpy's rule."""
    rank = builtins.max(len(s) for s in shapes)
    out = []
    for i in range(rank):
        size = 1
        for s in shapes:
            got = s[i - rank + len(s)] if i - rank + len(s) >= 0 else 1
            if got != 1:
                size = got
        out.append(size)
    return tuple(out)


def broadcast_shapes(*shapes):
    return _Size(_broadcast_shape([tuple(s) for s in shapes]))


def broadcast_tensors(*tensors):
    return tuple(wrap(v) for v in
                 guarded(_ts.Tensor.broadcastTensors, _handles(tensors)))


def _handles(tensors):
    """A list of tensors to a list of handles. The static methods over there
    take **one list.**

    **`_js_list` cannot be used** — it puts `int()` on every element and handles
    lists of numbers only. Handles put through it blow up with
    `int() argument must be a string…`, and that wording is the same as passing a
    bad shape argument, so it does not show where the problem is (measured:
    eleven of them).
    """
    return _to_js([handle(wrap(v)) for v in tensors])


def hstack(tensors, out=None):
    """1-D concatenates; above that it joins **along the columns.**"""
    _no_out(out)
    return wrap(guarded(_ts.Tensor.hstack, _handles(tensors)))


def _lift(x, rank):
    """Pad the missing leading axes with 1. What `atleast_2d` and `atleast_3d`
    do."""
    h = handle(x)
    shape = [int(n) for n in h.shape]
    if len(shape) >= rank:
        return wrap(h)
    return wrap(guarded(h.reshape, _js_list([1] * (rank - len(shape)) + shape)))


def vstack(tensors, out=None):
    _no_out(out)
    return wrap(guarded(_ts.Tensor.vstack, _handles(tensors)))


def _atleast3(x):
    """torch's `atleast_3d`. **The axis goes on the end** — not the front.

    1-D `(n,)` becomes `(1, n, 1)` and 2-D `(m, n)` becomes `(m, n, 1)`. Padding
    the front only gives `(1, 3, 4)`, and then `dstack` joins along the last axis
    instead of the third — the shape came out `(1, 3, 8)` and that caught it.
    """
    h = handle(x)
    shape = [int(n) for n in h.shape]
    if len(shape) >= 3:
        return wrap(h)
    if len(shape) == 2:
        shape = shape + [1]
    elif len(shape) == 1:
        shape = [1] + shape + [1]
    else:
        shape = [1, 1, 1]
    return wrap(guarded(h.reshape, _js_list(shape)))


def dstack(tensors, out=None):
    _no_out(out)
    return wrap(guarded(_ts.Tensor.dstack, _handles(tensors)))


def column_stack(tensors, out=None):
    """1-D is **stood up as a single column** and joined. Where it parts from
    `hstack`."""
    _no_out(out)
    return wrap(guarded(_ts.Tensor.columnStack, _handles(tensors)))


def block_diag(*tensors):
    """Blocks laid along the diagonal and zeros everywhere else."""
    return wrap(guarded(_ts.Tensor.blockDiag, _handles(tensors)))


# ── the ones with no computation of their own. All combinations of existing
# operations. ───────────────────────────────────────────────────────────────

def _shape_list(x):
    return [int(n) for n in handle(x).shape]


# **A factory that borrows a shape has to hear `dtype=` and `requires_grad=`
# too.** The defect from the six above survived here — fixing `zeros` and leaving
# `zeros_like` is half of it. The core fixed the same place alongside, and the
# golden catches the three drifting apart.

# **Four were absent entirely.** The module `__getattr__` was forwarding to
# borch.ts, which has no `fullLike`, and the ones it does have take no named
# arguments. A place that looks like it has the name and stops when called, so
# the core's surface is written out here.

def empty(*shape, **kw):
    _no_out(kw.get("out"))
    return _made(_ts.Tensor.zeros(_shape_of(shape)), kw)


def zeros_like(t, **kw):
    _no_out(kw.get("out"))
    return _kept(zeros(*_shape_list(t)), kw)


def ones_like(t, **kw):
    _no_out(kw.get("out"))
    return _kept(ones(*_shape_list(t)), kw)


def full_like(t, value, **kw):
    return _kept(full(_shape_list(t), value), kw)


def empty_like(t, **kw):
    _no_out(kw.get("out"))
    return _kept(zeros(*_shape_list(t)), kw)


def rand_like(t, generator=None, **kw):
    _no_out(kw.get("out"))
    return _kept(rand(*_shape_list(t), generator=generator), kw)


def randn_like(t, generator=None, **kw):
    _no_out(kw.get("out"))
    return _kept(randn(*_shape_list(t), generator=generator), kw)


def randint_like(t, low, high=None, generator=None, **kw):
    _no_out(kw.get("out"))
    if high is None:
        low, high = 0, low
    return _kept(randint(low, high, tuple(_shape_list(t)), generator=generator), kw)


def scalar_tensor(value, **kw):
    return _kept(wrap(guarded(_ts.Tensor.scalarTensor, float(value))), kw)


def logspace(start, end, steps, base=10.0, **kw):
    """Evenly spaced as powers of `base`. `linspace` supplies the exponents."""
    _no_out(kw.get("out"))
    return _kept(wrap(guarded(_ts.Tensor.logspace, start, end, steps, base)), kw)


def meshgrid(*tensors, indexing="ij"):
    """A grid. **`xy` has the first two axes swapped**, so one rule cannot cover
    both."""
    return tuple(wrap(v) for v in
                 guarded(_ts.Tensor.meshgrid, _handles(tensors), indexing))


def lerp(start, end, weight, out=None):
    """**The weight may be a tensor** — different at every position. Taking a
    number only loses that branch."""
    _no_out(out)
    w = handle(weight) if isinstance(weight, Tensor) else weight
    return wrap(guarded(handle(wrap(start)).lerp, handle(wrap(end)), w))


def _unary(x, name):
    return wrap(guarded(handle(x).unary, name))


def nan_to_num(t, nan=0.0, posinf=None, neginf=None, out=None):
    """NaN and infinities to finite numbers. **Given nothing, f32's extremes.**

    **The assembly moved over there.** While it lived here the name did not exist
    in borch.ts, and since the golden goes through this function the table was
    green — a name missing only for the people writing TypeScript.
    `tests/test_binding_fills_in.py` counts that place.
    """
    _no_out(out)
    return wrap(guarded(handle(wrap(t)).nanToNum, nan, posinf, neginf))


def isposinf(t, out=None):
    _no_out(out)
    return wrap(guarded(handle(wrap(t)).isposinf))


def isneginf(t, out=None):
    _no_out(out)
    return wrap(guarded(handle(wrap(t)).isneginf))


def isreal(t):
    """Everything is real, so all of it is true. **Not a lie — a fact.**"""
    return wrap(guarded(handle(wrap(t)).isreal))


def isclose(a, b, rtol=1e-5, atol=1e-8, equal_nan=False):
    return wrap(guarded(handle(wrap(a)).isclose, handle(wrap(b)), rtol, atol))


def isin(elements, test_elements):
    """Is the element in that list. One broadcast solves it."""
    return wrap(guarded(handle(wrap(elements)).isin, handle(wrap(test_elements))))


def _nan_extreme(name):
    """`fmax` and `fmin` **skip NaN** — `maximum` carries NaN out with it.

    **The assembly moved over there.** While it lived here the name did not exist
    in borch.ts, and since the golden goes through this function the table was
    green — a name missing only on the TypeScript side.
    """
    def call(a, b, **kw):
        del kw
        return wrap(guarded(getattr(handle(wrap(a)), name), handle(wrap(b))))
    call.__name__ = name
    return call


fmax = _nan_extreme("fmax")
fmin = _nan_extreme("fmin")


def float_power(a, b, out=None):
    _no_out(out)
    e = handle(b) if isinstance(b, Tensor) else b
    return wrap(guarded(handle(wrap(a)).floatPower, e))


def logical_xor(a, b):
    """Not in borch.ts's **binary table** — it is built by asking that side for
    inequality twice."""
    return wrap(guarded(handle(wrap(a)).logicalXor, handle(wrap(b))))


# ── shape and indexing ──────────────────────────────────────────────────────
#
# Three kinds are written by hand — **the ones taking no tensor** (triangular
# index tables), **the ones taking a list of tensors** (`index_put`'s indices,
# `cartesian_prod`), and **the ones answering with a tuple** (the splits,
# `unravel_index`). `__getattr__` simply forwards the rest.
#
# Why a list of tensors is separate: `_arg` sends lists through `_js_list`,
# which puts `int()` over them. A list holding tensors stops at the integer
# conversion.

def _js_tensors(seq):
    return _js.Array.from_([handle(t) for t in seq])


def index_put(t, indices, values, accumulate=False):
    return wrap(guarded(handle(t).indexPut, _js_tensors(indices),
                        handle(values), accumulate))


def index_put_(t, indices, values, accumulate=False):
    t = wrap(t)
    guarded(handle(t).indexPut_, _js_tensors(indices), handle(values),
            accumulate)
    return t


def unravel_index(indices, shape):
    """**One tensor per axis, returned as a tuple** (measured)."""
    got = guarded(handle(indices).unravelIndex, _js_list(shape))
    return tuple(wrap(p) for p in got)


def unique_consecutive(t, return_inverse=False, return_counts=False, dim=None,
                       ):
    """Collapses **consecutive** duplicates only. The length depends on the
    values, so borch.ts is asynchronous here — `settle` awaits that promise."""
    got = guarded(handle(t).uniqueConsecutive, return_inverse, return_counts,
                  dim)
    return tuple(got) if isinstance(got, list) else got


def tensor_split(t, indices_or_sections, dim=0):
    return tuple(guarded(handle(t).tensorSplit, _arg(indices_or_sections), dim))


def split_with_sizes(t, split_sizes, dim=0):
    return tuple(guarded(handle(t).splitWithSizes, _js_list(split_sizes), dim))


def tril_indices(row, col, offset=0, *, dtype=None):
    """**A `(2, count)` table** — not pairs of positions but a row of rows and a
    row of columns.

    **`dtype` was falling into `**kw`.** torch declares it keyword-only with a default
    of `torch.long`, so the values agreed and the label was whatever came out — and
    the golden's `tril_indices(dtype=int64)` case asked for the label torch already
    gives, which is why nothing ever parted. `_made` is the same seam, gathered.
    """
    return _made(wrap(_ts.Tensor.trilIndices(row, col, offset)), {"dtype": dtype})


def triu_indices(row, col, offset=0, *, dtype=None):
    return _made(wrap(_ts.Tensor.triuIndices(row, col, offset)), {"dtype": dtype})


def vander(x, N=None, increasing=False):
    return wrap(_ts.Tensor.vander(handle(x), N, increasing))


def cartesian_prod(*tensors):
    return wrap(_ts.Tensor.cartesianProd(*[handle(t) for t in tensors]))


def combinations(t, r=2, with_replacement=False):
    return wrap(_ts.Tensor.combinations(handle(t), r, with_replacement))


def chain_matmul(*matrices):
    mats = (list(matrices[0]) if len(matrices) == 1
            and isinstance(matrices[0], (list, tuple)) else list(matrices))
    return wrap(_ts.Tensor.chainMatmul(*[handle(m) for m in mats]))


# ── top-level linear algebra ────────────────────────────────────────────────
#
# The two written by hand are **the ones whose names collide with `linalg`'s.**
# `camel` forwards `lu` as `lu` and `lu_solve` as `luSolve`, and those names in
# borch.ts belong to `linalg` and give different answers — the top-level ones are
# `luTop` and `luSolveTop`.

def lu(a, pivot=True, get_infos=False):
    """`(LU, pivots)`. **A different thing from `linalg.lu`** — that one spreads
    it into `P`, `L` and `U`, and this one is a single packed matrix plus the
    list of swaps (measured)."""
    got = guarded(handle(a).luTop, pivot, get_infos)
    return tuple(got) if get_infos else (got.LU, got.pivots)


def lu_solve(b, lu_data, lu_pivots, out=None):
    """**The argument order is reversed from `linalg.lu_solve`** — `b` comes
    first here."""
    _no_out(out)
    return wrap(guarded(handle(b).luSolveTop, handle(lu_data),
                        handle(lu_pivots)))


def lu_unpack(lu_data, lu_pivots, unpack_data=True, unpack_pivots=True, out=None):
    """**Turned off it gives an empty tensor, not `None`** (measured: the shape
    is `(0,)`)."""
    _no_out(out)
    got = guarded(handle(lu_data).luUnpack, handle(lu_pivots), unpack_data,
                  unpack_pivots)
    return (got.P, got.L, got.U)


# ── statistics ──────────────────────────────────────────────────────────────
#
# Written by hand: **four random ones**, **three refusals**, and `trapz`, which
# is assembled. `__getattr__` forwards the rest to the first argument's method.
#
# **The golden cannot pin random values** — borch.ts's random stream and torch's
# are different. The extremes are deterministic though (`std=0`, `p=0`, `p=1`),
# and that is what the golden asks about.

def trapz(y, x=None, dx=1.0, dim=-1, **kw):
    """The old name for `trapezoid`. The same thing (measured)."""
    return trapezoid(y, x, dx, dim, **kw)


# ── complex numbers, and a few factories ────────────────────────────────────
#
# For a while this was **"the names that have an answer even without complex
# numbers"** — over the reals the `conj` family is the identity, so one `_alias`
# covered them all, and `is_complex` was pinned to `return False`. Complex
# numbers arriving in borch.ts turned those answers into **wrong answers.**
#
# **An identity that passes today is the first thing to break when the domain
# widens** — `conj_physical` broke exactly that way in the core, and here there
# were six such names.

def _alias(t):
    """The identity. **`to(its own dtype)` is the shortest way to keep both the
    dtype and the graph** — borch.ts's `to` hands back the same tensor when the
    dtype matches, so no kernel runs."""
    t = wrap(t)
    return t


def _is_cplx(t):
    """Is this handle complex. **It goes through `str()`** — `dtype` is a JS
    string."""
    return str(handle(t).dtype) == "complex64"


def complex(re, im, out=None):
    """Weave a real part and an imaginary part together.

    **This name shadows the Python builtin `complex`.** That is why `_is_cplx` is
    used for the complex test inside this file — the core (`borch/_ops.py`) made
    the same choice in the same place, and this is the third builtin whose name
    gets shadowed (`abs`, `bool`, `max`, `range`).
    """
    _no_out(out)
    return wrap(_ts.Tensor.complex(handle(re), handle(im)))


def polar(abs, angle, out=None):                                    # noqa: A002
    """From a magnitude and an angle. The parameter name is torch's, so it
    shadows the builtin `abs`."""
    _no_out(out)
    return wrap(_ts.Tensor.polar(handle(abs), handle(angle)))


def view_as_real(t):
    """Complex to pairs of reals. **A view** — borch.ts stores them
    interleaved."""
    return wrap(guarded(handle(t).viewAsReal))


def view_as_complex(t):
    return wrap(guarded(handle(t).viewAsComplex))


def real(t):
    """The real part. **On a real tensor it is the tensor itself**, dtype
    included (measured)."""
    return wrap(guarded(handle(t).real)) if _is_cplx(t) else _alias(t)


def imag(t):
    """The imaginary part. **torch itself refuses on a real tensor** (measured)
    — not a limit of ours."""
    if not _is_cplx(t):
        raise RuntimeError(
            "imag is not implemented for tensors with non-complex dtypes.")
    return wrap(guarded(handle(t).imag))


def conj(t):
    """The conjugate. Over the reals, the identity.

    **A divergence from torch.** torch's `conj` is lazy and only raises the
    conjugate bit, while this flips the values immediately — which is why
    `is_conj` below is always false. The values are the same.
    """
    return wrap(guarded(handle(t).conjPhysical)) if _is_cplx(t) else _alias(t)


def conj_physical(t, out=None):
    _no_out(out)
    return conj(t)


def conj_physical_(t):
    x = wrap(t)
    return x._write_back(conj(x)) if _is_cplx(x) else _alias(x)


def resolve_conj(t):
    """Materialise the conjugate flag. **There is no such flag here at all**, so
    it is always the identity."""
    return _alias(t)


def resolve_neg(t):
    return _alias(t)


def angle(t):
    """The angle. Complex is `atan2(imag, real)`; **real gives π for negatives**,
    and the dtype is always real."""
    if _is_cplx(t):
        return wrap(guarded(handle(t).angle))
    import math

    x = wrap(t)
    below = x.binary("lt", full([], 0.0))
    return wrap(guarded(handle(below).to, "float32")).mul(full([], math.pi))


def is_complex(t):
    return _is_cplx(t)


def is_conj(t):
    return False


def is_neg(t):
    return False


def asarray(obj, dtype=None, copy=None):
    """**Given a tensor it is not a copy** (measured). `copy=True` makes it
    one."""
    from ._base import tensor as _t

    if isinstance(obj, Tensor) and dtype is None and not copy:
        return obj
    if isinstance(obj, Tensor):
        got = obj.to(dtype) if dtype is not None else obj
        return _t(got.numpy().copy()) if copy else got
    # **`copy=False` must not be handed to numpy as-is.** In numpy 2 it is an
    # order not to copy, so it throws on anything that can only be copied, such
    # as a list — different from the default `None` ("avoid copying if
    # possible"). It is mentioned here only when true.
    arr = _np.array(obj, copy=True) if copy else _np.asarray(obj)
    return _t(arr, dtype)


def frombuffer(buffer, dtype=None, count=-1, offset=0, requires_grad=False):
    """Read the bytes as they are. **`offset` is a byte count** — not an element
    count (measured).

    Choosing the dtype is `_dtype_to_make`'s job. **This was the one place
    writing out its own branch**, and that branch fell quietly to float32 on
    anything that was not a `_DType` — `frombuffer(buf, dtype=torch.half)`
    **produced float32 without a word.** A dtype that has a name and no storage
    has to stop, and that gate already lives inside that function.

    `dtype=` decides **what the bytes are read as** here, so it is not left to
    `_made` — converting the dtype afterwards is after they have already been
    read as something else. Only the gradient is left to it.
    """
    from ._base import tensor as _t

    name = "float32" if dtype is None else _dtype_to_make(dtype)
    kind = _np.dtype(name.replace("torch.", ""))
    return _made(_t(_np.frombuffer(buffer, dtype=kind, count=count,
                                   offset=offset).copy()),
                 {"requires_grad": requires_grad})


def range_top(start, end=None, step=1, **kw):
    """**The end is included** — `arange` excludes it (measured). Forwarded
    quietly to `arange`, one element goes missing."""
    _no_out(kw.get("out"))
    if end is None:
        start, end = 0, start
    return _made(_ts.Tensor.range(start, end, step), kw)


def empty_strided(size, stride, out=None):
    """**Absent because strides cannot be expressed.** A different place from
    `as_strided` — there the values are the answer, so a copy gives the same
    answer, and here **the strides themselves are the only answer.**"""
    _no_out(out)
    raise RuntimeError(
        "torch.empty_strided — there is no such thing as a stride here.")


def empty_permuted(size, physical_layout, out=None):
    _no_out(out)
    raise RuntimeError(
        "torch.empty_permuted — there is no such thing as a stride here.")


def histogramdd(t, bins=10, out=None):
    """A histogram over several axes.

    **The edges arrive as a list of tensors.** borch.ts hands over a JS array and
    `settle` does not reach inside it, so left alone JS handles stay on the
    Python side — the receiver can use neither `.shape` nor `._h`. They are
    wrapped one at a time here.
    """
    _no_out(out)
    from ._base import _Fields

    got = guarded(handle(t).histogramdd, _arg(bins))
    out = _Fields.__new__(_Fields)
    object.__setattr__(out, "_order", ["hist", "bin_edges"])
    object.__setattr__(out, "_d", {
        "hist": got.hist,
        "bin_edges": [wrap(e) for e in got.bin_edges],
    })
    return out


def normal(mean=0.0, std=1.0, size=None, generator=None, **kw):
    """A normal sample. **With `std` at 0 it is the mean itself.**

    `dtype=` and `requires_grad=` were being swallowed by `**kw` — both times the
    factories were gathered under `_made`, this one was outside the list. `out=`
    was being swallowed by the same `**kw`.
    """
    _no_out(kw.get("out"))
    from ._base import tensor as _t

    stream = _stream(generator)
    if isinstance(mean, Tensor) or isinstance(std, Tensor):
        m = _np.asarray(wrap(mean).numpy(), dtype=_np.float64)
        s = _np.asarray(wrap(std).numpy(), dtype=_np.float64)
        m, s = _np.broadcast_arrays(m, s)
        return _made(_t(stream.normal(m, s).astype(_np.float32)), kw)
    shape = () if size is None else tuple(size)
    return _made(_t(stream.normal(float(mean), float(std), shape).astype(_np.float32)), kw)


def bernoulli(t, p=None, *, generator=None, out=None):
    """A 1 at each position with that probability. **0 gives all zeros, 1 gives
    all ones.**

    **`p` is torch's other form** — a number is the probability everywhere and the
    tensor's own values are ignored; given nothing, the values are the probabilities.
    This took neither, so `x.bernoulli(0.5)` stopped on the argument count.

    `generator` picks the stream, and it was not here at all. The core holds
    `Generator` and honours it; this side hands it along.
    """
    from ._base import tensor as _t

    if out is not None:
        raise NotImplementedError("`bernoulli(out=…)` is not carried across")
    rng = _stream(generator)
    probs = (_np.full(tuple(int(v) for v in handle(wrap(t)).shape), float(p))
             if p is not None
             else _np.asarray(wrap(t).numpy(), dtype=_np.float64))
    return _t((rng.random(probs.shape) < probs).astype(_np.float32))


def multinomial(probs, num_samples, replacement=True, *, generator=None, out=None):
    """Draw indices in proportion to the weights.

    **The name was not on this side at all**, so `multinomial` fell through to
    borch.ts, which has no such name either, and came back *borch.ts does not have
    `multinomial`* — a sentence about the far side for something the core has had
    all along. Nothing asked, because no case did.

    Drawn here rather than over there for `bernoulli`'s reason: the sampling is one
    CPU pass over weights already coming down, and a WGSL kernel for it would be a
    second copy of a distribution to keep in step.
    """
    _no_out(out)
    from ._base import tensor as _t

    from borch._ops import _multinomial_checks

    stream = _stream(generator)
    p = _np.asarray(wrap(probs).numpy(), dtype=_np.float64)
    # **The core's four checks, borrowed rather than retyped.** Written again here
    # they would be a second copy of torch's wording to keep in step, and the core's
    # own copy had just been found accepting `replacement=False` and ignoring it.
    _multinomial_checks(p, num_samples, replacement)
    p = p / p.sum(axis=-1, keepdims=True)
    if p.ndim == 1:
        drawn = stream.choice(len(p), size=num_samples, replace=bool(replacement), p=p)
        return _t(drawn.astype("int64"))
    rows = [stream.choice(p.shape[-1], size=num_samples, replace=bool(replacement), p=row)
            for row in p]
    return _t(_np.asarray(rows, dtype=_np.int64))


def bernoulli_(t, p=0.5, generator=None, **kw):
    """**A different operation from its partner.** `bernoulli()` reads its own
    values as probabilities, while this one **ignores** its values and fills from
    `p` (measured: `[0,1,0,1]` going in still comes out different every time).

    Built from the partner on the strength of the trailing underscore, positions
    with probability 0 or 1 are certain and their values match, so it is
    **quietly wrong only at the probabilities in between.** The core kept it out
    of the automatic table for the same reason.
    """
    del kw
    from ._base import tensor as _t

    # As `bernoulli` above: the stream the caller asked for, not always the global one.
    rng = _stream(generator)
    got = wrap(t)
    shape = tuple(int(v) for v in handle(got).shape)
    return _t((rng.random(shape) < p).astype(_np.float32))


def float_power_(t, exponent, **kw):
    """**Always refuses.** `float_power` produces double precision and there is
    nowhere to write it back. torch stops for the same reason on a float32
    destination (measured)."""
    del t, exponent, kw
    raise RuntimeError(
        "`float_power_` cannot be used in place — the result is double precision and "
        "there is nowhere to put it back. Use `x.float_power(k)` for a new tensor. "
        "(torch: the base given to float_power_ has dtype Float but the "
        "operation's result requires dtype Double)")


def poisson(t):
    from ._base import tensor as _t

    lam = _np.asarray(wrap(t).numpy(), dtype=_np.float64)
    return _t(_rng.poisson(lam).astype(_np.float32))


def binomial(count, prob):
    from ._base import tensor as _t

    n = _np.asarray(wrap(count).numpy(), dtype=_np.float64)
    p = _np.asarray(wrap(prob).numpy(), dtype=_np.float64)
    n, p = _np.broadcast_arrays(n, p)
    return _t(_rng.binomial(n.astype(_np.int64), p).astype(_np.float32))


# ── short-time transforms ───────────────────────────────────────────────────
#
# **A refusal for a long time.** The refusal said "the complex convention
# (Wirtinger) has not been measured", and the reason was right — what blocked it
# was the convention, not the storage. Measuring and pinning it opened the door.

def _stft_options(hop_length, win_length, window, center, pad_mode,
                  normalized, onesided, return_complex, length=None,
                  align_to_window=None):
    """Into borch.ts's `StftOptions`. **What is absent is left out** — over
    there `undefined` and `null` mean different things (the default versus
    "explicitly none"), so forwarding Python's `None` as-is means the
    `return_complex` requirement never fires."""
    kw = {}
    if hop_length is not None:
        kw["hopLength"] = int(hop_length)
    if win_length is not None:
        kw["winLength"] = int(win_length)
    if window is not None:
        kw["window"] = handle(window)
    kw["center"] = bool(center)
    if pad_mode is not None:
        kw["padMode"] = str(pad_mode)
    kw["normalized"] = bool(normalized)
    # **Left out when absent, like the rest.** torch's whole behaviour for it is the
    # refusal unless `center` is false, so what has to cross is *whether it was given*
    # — forwarding a `None` would make borch.ts's guard fire on every call.
    if align_to_window is not None:
        kw["alignToWindow"] = bool(align_to_window)
    if onesided is not None:
        kw["onesided"] = bool(onesided)
    if return_complex is not None:
        kw["returnComplex"] = bool(return_complex)
    if length is not None:
        kw["length"] = int(length)
    return _js_options(**kw)


def stft(input, n_fft, hop_length=None, win_length=None, window=None,
         center=True, pad_mode="reflect", normalized=False, onesided=None,
         return_complex=None, align_to_window=None):
    """The short-time Fourier transform. **It refuses without `return_complex`**
    (measured).

    **`align_to_window` was falling into `**kw`**, which is accepting an argument and
    dropping it — torch rejects it unless `center` is false and this said nothing at
    all. Named here so it crosses, and borch.ts refuses in torch's own words."""
    return wrap(guarded(
        _ts.stft, handle(input), int(n_fft),
        _stft_options(hop_length, win_length, window, center, pad_mode,
                      normalized, onesided, return_complex,
                      align_to_window=align_to_window)))


def istft(input, n_fft, hop_length=None, win_length=None, window=None,
          center=True, normalized=False, onesided=None, length=None,
          return_complex=False):
    """**`return_complex` stopped at this line** — the seat was here, the options
    object carried the name, and the word was never put in it. borch.ts reads it now
    and the two-sided branch there was running a forward transform besides."""
    return wrap(guarded(
        _ts.istft, handle(input), int(n_fft),
        _stft_options(hop_length, win_length, window, center, None,
                      normalized, onesided, return_complex, length)))


# **`torch.fft` is a namespace.** It is a module over in borch.ts too, so it can
# be forwarded as-is; what remains is unpacking Python's `None` defaults and
# named arguments into positions.
class _Fft:
    @staticmethod
    def fft(input, n=None, dim=-1, norm=None):
        return wrap(guarded(_ts.fft.fft, handle(input), n, int(dim), norm))

    @staticmethod
    def ifft(input, n=None, dim=-1, norm=None):
        return wrap(guarded(_ts.fft.ifft, handle(input), n, int(dim), norm))

    @staticmethod
    def rfft(input, n=None, dim=-1, norm=None):
        return wrap(guarded(_ts.fft.rfft, handle(input), n, int(dim), norm))

    # ── multi-axis and Hermitian — **borch.ts assembles them** ─────────────
    #
    # Assembled in Python here the golden goes green and the name does not exist
    # for anyone using borch.ts. This repository has hit that seven times, so it
    # lives over there and this side only forwards.

    @staticmethod
    def _many(name, input, s, dim, norm):
        js = getattr(_ts.fft, name)
        return wrap(guarded(js, handle(input),
                            None if s is None else _js_list(list(s)),
                            None if dim is None else (
                                _js_list([int(d) for d in dim])
                                if not isinstance(dim, int) else int(dim)),
                            norm))

    @staticmethod
    def irfft(input, n=None, dim=-1, norm=None):
        return wrap(guarded(_ts.fft.irfft, handle(input), n, int(dim), norm))

    @staticmethod
    def fftfreq(n, d=1.0):
        return wrap(guarded(_ts.fft.fftfreq, int(n), float(d)))

    @staticmethod
    def rfftfreq(n, d=1.0):
        return wrap(guarded(_ts.fft.rfftfreq, int(n), float(d)))

    @staticmethod
    def fftshift(input, dim=None):
        return wrap(guarded(_ts.fft.fftshift, handle(input),
                            None if dim is None else _dim_arg(dim)))

    @staticmethod
    def ifftshift(input, dim=None):
        return wrap(guarded(_ts.fft.ifftshift, handle(input),
                            None if dim is None else _dim_arg(dim)))


def _dim_arg(dim):
    """One axis as a number, several as a JS array. A Python list forwarded
    as-is is not seen as an array over there — `_js_list` exists for that."""
    return _js_list(dim) if isinstance(dim, (list, tuple)) else int(dim)


for _n in ("fft2", "ifft2", "fftn", "ifftn", "rfft2", "irfft2", "rfftn", "irfftn",
           "hfft2", "ihfft2", "hfftn", "ihfftn"):
    def _make_many(name):
        def call(input, s=None, dim=None, norm=None, out=None):
            _no_out(out)
            if dim is None and name.endswith("2"):
                dim = (-2, -1)
            return _Fft._many(name, input, s, dim, norm)
        call.__name__ = name
        return staticmethod(call)
    setattr(_Fft, _n, _make_many(_n))
del _n


def _fft_one(name):
    def call(input, n=None, dim=-1, norm=None, out=None):
        _no_out(out)
        return wrap(guarded(getattr(_ts.fft, name), handle(input), n, int(dim), norm))
    call.__name__ = name
    return staticmethod(call)


_Fft.hfft = _fft_one("hfft")
_Fft.ihfft = _fft_one("ihfft")

fft = _Fft()

# **`torch.device` is borrowed from the core.** It is a pure value object (a
# kind and an index) with nothing to rewrite here — kept as two copies, the day
# comes when the `repr`s diverge, and the golden has pinned those characters.
from borch._base import device                            # noqa: E402,F401


# ── the eight top-level recurrent ones ──────────────────────────────────────
#
# **`__getattr__` cannot forward them.** It sends things to the first argument's
# method, and these eight are free functions in borch.ts too (they take the
# weights as a list). Written by hand.

def _rnn_options(has_biases, num_layers, dropout, train, bidirectional,
                 batch_first):
    return _js_options(hasBiases=bool(has_biases), numLayers=int(num_layers),
                       dropout=float(dropout), train=bool(train),
                       bidirectional=bool(bidirectional),
                       batchFirst=bool(batch_first))


def _rnn_params(params):
    return _js.Array.from_([handle(p) for p in params])


def lstm(input, hx, params, has_biases, num_layers, dropout, train,     # noqa: A002
         bidirectional, batch_first=False):
    """`(output, h_n, c_n)` — **all three spread.** The layer side groups them as
    `(output, (h, c))`."""
    got = guarded(_ts.lstm, handle(input),
                  _js.Array.from_([handle(hx[0]), handle(hx[1])]),
                  _rnn_params(params),
                  _rnn_options(has_biases, num_layers, dropout, train,
                               bidirectional, batch_first))
    return tuple(wrap(t) for t in got)


def _rnn_two(name):
    def call(input, hx, params, has_biases, num_layers, dropout, train,  # noqa: A002
              bidirectional, batch_first=False):
        got = guarded(getattr(_ts, name), handle(input), handle(hx),
                      _rnn_params(params),
                      _rnn_options(has_biases, num_layers, dropout, train,
                                   bidirectional, batch_first))
        return tuple(wrap(t) for t in got)

    return call


gru = _rnn_two("gru")
rnn_tanh = _rnn_two("rnnTanh")
rnn_relu = _rnn_two("rnnRelu")


def lstm_cell(input, hx, w_ih, w_hh, b_ih=None, b_hh=None):       # noqa: A002
    got = guarded(_ts.lstmCell, handle(input),
                  _js.Array.from_([handle(hx[0]), handle(hx[1])]),
                  handle(w_ih), handle(w_hh),
                  None if b_ih is None else handle(b_ih),
                  None if b_hh is None else handle(b_hh))
    return tuple(wrap(t) for t in got)


def _cell_one(name):
    def call(input, hx, w_ih, w_hh, b_ih=None, b_hh=None):        # noqa: A002
        return wrap(guarded(getattr(_ts, name), handle(input), handle(hx),
                            handle(w_ih), handle(w_hh),
                            None if b_ih is None else handle(b_ih),
                            None if b_hh is None else handle(b_hh)))

    return call


gru_cell = _cell_one("gruCell")
rnn_tanh_cell = _cell_one("rnnTanhCell")
rnn_relu_cell = _cell_one("rnnReluCell")


# ── the names that were left at top level ───────────────────────────────────
#
# **Counting by name gets it wrong.** `fake_quantize_*` is named for
# quantisation and takes reals and produces reals, and `dequantize` is the
# identity over the reals — only measuring showed they were not refusals.

def igamma(input, other, out=None):                                 # noqa: A002
    """The regularised lower incomplete gamma. **The gradient exists on the `x`
    side only** (measured)."""
    _no_out(out)
    return wrap(guarded(_ts.igamma, handle(input), handle(other)))


def igammac(input, other, out=None):                                # noqa: A002
    _no_out(out)
    return wrap(guarded(_ts.igammac, handle(input), handle(other)))


def polygamma(n, input, out=None):                                  # noqa: A002
    """**`n` comes first** — the tensor is second. That is torch's
    signature."""
    _no_out(out)
    return wrap(guarded(_ts.polygamma, int(n), handle(input)))


def constant_pad_nd(input, pad, value=0.0):               # noqa: A002
    return wrap(guarded(handle(input).constantPadNd, _js_list(pad),
                        float(value)))


def fake_quantize_per_tensor_affine(input, scale, zero_point,   # noqa: A002
                                    quant_min, quant_max):
    return wrap(guarded(handle(input).fakeQuantizePerTensorAffine,
                        float(scale), float(zero_point), int(quant_min),
                        int(quant_max)))


def fake_quantize_per_channel_affine(input, scale, zero_point,  # noqa: A002
                                     axis, quant_min, quant_max):
    return wrap(guarded(handle(input).fakeQuantizePerChannelAffine,
                        handle(scale), handle(zero_point), int(axis),
                        int(quant_min), int(quant_max)))


def dequantize(input):                                    # noqa: A002
    """The identity over the reals. There will **never** be a quantised dtype, so
    that is the complete answer."""
    return wrap(guarded(handle(input).dequantize))


def resize_as_(input, the_template, memory_format=None):  # noqa: A002
    """In place, to `the_template`'s shape. **The values in the added cells are
    undefined** (measured).

    The argument is `the_template` because that is the name torch registers — see
    the method of the same name in `_base.py`.

    **`copyFrom` cannot do it** — that needs the same element count, and changing
    the element count is the whole of this operation. In-placeness is kept by
    swapping the handle on the Python side: the object the caller holds stays,
    and only the buffer underneath changes.
    """
    if memory_format is not None:
        from borch._base import _unsupported
        _unsupported("Tensor.resize_as_(memory_format=…)")
    x = wrap(input)
    want = wrap(the_template).shape
    flat = x.numpy().reshape(-1)
    need = 1
    for d in want:
        need *= int(d)
    grown = _np.zeros(need, dtype=flat.dtype)
    keep = min(flat.size, need)
    grown[:keep] = flat[:keep]
    from ._base import tensor as _mk
    x._h = handle(_mk(grown.reshape(tuple(int(d) for d in want))))
    return x


def hash_tensor(*args):
    """**No uint64 and no specification.** A name is not put down for something
    whose values cannot be matched."""
    raise RuntimeError(
        "torch.hash_tensor — there is no uint64 and no settled hash spec.")


def sspaddmm(input, mat1, mat2, beta=1, alpha=1, out=None):
    """**Sparse-only, so it is absent.** The core's place and the core's reason
    for refusing — imitated with a dense tensor, the shape matches and what comes
    back has a different storage layout."""
    _no_out(out)
    raise RuntimeError(
        "torch.sspaddmm — there is no sparse tensor layout here. "
        "Use real PyTorch on your own machine.")


def fill(x, value):
    """**Not in place.** One character apart from `fill_` and a different job —
    this one produces a new tensor and leaves the original alone (measured).

    Sent through the alias table, `fill_` gets dragged under the same name too,
    so it is written by hand here.
    """
    return wrap(guarded(handle(x).fillWith, float(value)))


def bitwise_not(x, out=None):
    """**On booleans it is logical negation.** On integers it is `~x`, so
    `~1 == -2`, and applied to true torch gives false (measured) — the two
    branches differ outright in value.

    The binary ones (`and`, `or`, `xor`) need no branch. Over 0/1 the bitwise and
    logical computations give the same answer, and `bool` with `bool` stays
    `bool`. Only negation differs.

    **The branch moved over there.** While it lived here, borch.ts's kernel
    comment said "this only looks at integers", which means somebody calling from
    TypeScript gets **a wrong answer** rather than no answer. That side knows its
    own dtype, so the branch belongs there.
    """
    _no_out(out)
    return wrap(guarded(handle(wrap(x)).bitwise_not))


def var_mean(t, dim=None, keepdim=False, out=None):
    """**Both at once.** Asking for one leaves the other free to be wrong
    uncaught."""
    _no_out(out)
    t = wrap(t)
    return (t.var(dim=dim, keepdim=keepdim), t.mean(dim=dim, keepdim=keepdim))


def std_mean(t, dim=None, keepdim=False, out=None):
    _no_out(out)
    t = wrap(t)
    return (t.std(dim=dim, keepdim=keepdim), t.mean(dim=dim, keepdim=keepdim))


def inner(a, b, out=None):
    _no_out(out)
    return wrap(guarded(handle(wrap(a)).inner, handle(wrap(b))))


def vdot(a, b, out=None):
    _no_out(out)
    return (wrap(a) * wrap(b)).sum()


def kron(a, b, out=None):
    """Any rank. The version that lived here looked at one axis, so 2-D input gave
    a quietly wrong answer — moving it over there turned that place into a refusal,
    and the refusal is gone now that borch.ts interleaves the two shapes."""
    _no_out(out)
    return wrap(guarded(handle(wrap(a)).kron, handle(wrap(b))))


def cross(a, b, dim=-1, out=None):
    _no_out(out)
    a, b = wrap(a), wrap(b)
    rank = len(_shape_list(a))
    axis = dim + rank if dim < 0 else dim

    def part(t, i):
        return wrap(guarded(handle(t).narrow, axis, i, 1))

    return cat([part(a, 1) * part(b, 2) - part(a, 2) * part(b, 1),
                part(a, 2) * part(b, 0) - part(a, 0) * part(b, 2),
                part(a, 0) * part(b, 1) - part(a, 1) * part(b, 0)], axis)


# ── the numeric family. **The three computed as series live in WGSL and the
# rest are combinations.** ──────────────────────────────────────────────────

def cdist(a, b, p=2.0):
    """The distance between every pair. One broadcast solves it."""
    return wrap(guarded(handle(wrap(a)).cdist, handle(wrap(b)), p))


def cov(t, correction=1):
    """Covariance. **Rows are variables and columns are observations** — the
    axes are the reverse of numpy's, which is confusing."""
    return wrap(guarded(handle(wrap(t)).cov, correction))


# ── the names that exist only at torch's top level ──────────────────────────
#
# The top level is raw ATen, so **the argument order differs and the enums are
# integers.** The same computation called a different way, so the computation is
# kept as one copy in `nn.functional` and only the positions move here.

def _inplace_from(name, fn_name=None):
    def call(x, *args, **kw):
        from . import _nn
        x._refuse_inplace_on_leaf(name)
        got = getattr(_nn.functional, fn_name or name.rstrip("_"))(x, *args, **kw)
        return x._write_back(got)
    call.__name__ = name
    return call


def nan_to_num_(x, *args, **kw):
    """**`nan_to_num` lives on the module, not on `F`.** Going by the name and
    looking in `F` stops with "absent" — not every top-level name exists on `F`
    as well."""
    x._refuse_inplace_on_leaf("nan_to_num_")
    return x._write_back(nan_to_num(x, *args, **kw))


dropout_ = _inplace_from("dropout_")
alpha_dropout_ = _inplace_from("alpha_dropout_")
feature_alpha_dropout_ = _inplace_from("feature_alpha_dropout_")
feature_dropout_ = _inplace_from("feature_dropout_", "dropout2d")


def feature_dropout(x, p=0.5, train=True):
    """**Drops whole channels** — the same computation as `F.dropout2d`
    (measured)."""
    from . import _nn
    return _nn.functional.dropout2d(x, p, train)


def batch_norm(x, weight, bias, running_mean, running_var, training=False,
               momentum=0.1, eps=1e-5, cudnn_enabled=False):
    """**A different argument order from `F.batch_norm`** — the weights come
    before the statistics here."""
    from . import _nn
    return _nn.functional.batch_norm(x, running_mean, running_var, weight, bias,
                                     training, momentum, eps)


def grid_sampler(x, grid, interpolation_mode=0, padding_mode=0,
                 align_corners=False):
    """**The enums are integers.** 0 and 1 are `bilinear` and `nearest`, and the
    padding is 0, 1 and 2."""
    from . import _nn
    modes = ("bilinear", "nearest", "bicubic")
    pads = ("zeros", "border", "reflection")
    return _nn.functional.grid_sample(x, grid, modes[int(interpolation_mode)],
                                      pads[int(padding_mode)], align_corners)


def max_pool1d_with_indices(x, kernel_size, stride=None, padding=0, dilation=1,
                            ceil_mode=False):
    """**The three window arguments go through and are refused one level down.**

    They used to be checked here with a message of this file's own, which put two
    spellings of one refusal in two places — and the one a caller met depended on
    which name they had reached for. `_pool_with_indices` says why: the positions come
    from the window-list machinery, where a window is `[start, end)` with no step.
    """
    from . import _nn
    return _nn.functional.max_pool1d_with_indices(
        x, kernel_size, stride, padding, dilation, ceil_mode)


def ctc_loss(log_probs, targets, input_lengths, target_lengths, blank=0,
             reduction=1, zero_infinity=False):
    """**`reduction` is an integer** — 0, 1 and 2 are `none`, `mean` and
    `sum`."""
    from . import _nn
    kinds = ("none", "mean", "sum")
    return _nn.functional.ctc_loss(log_probs, targets, input_lengths,
                                   target_lengths, blank, kinds[int(reduction)],
                                   zero_infinity)


def geqrf(t, out=None):
    """QR in reflector form. The partner to `linalg.householder_product`, so it
    exists at top level too."""
    _no_out(out)
    return guarded(handle(t).geqrf)


def corrcoef(t):
    """Covariance divided by the standard deviations. **The diagonal becomes
    1** — that is the check."""
    return wrap(guarded(handle(wrap(t)).corrcoef))


def tensordot(a, b, dims=2):
    """Fold the named axes together and multiply. The folded axes are herded
    together and one matmul finishes it.

    **The axis lists are forwarded as they are** — wrapped in `_js_list` the
    inner lists meet `int()` and blow up. Both the single-number form and the
    two-lists form are in the signature over there.
    """
    dd = dims if isinstance(dims, int) else _to_js(
        [_js_list([int(v) for v in side]) for side in dims])
    return wrap(guarded(handle(wrap(a)).tensordot, handle(wrap(b)), dd))




def _trapezoid_x(x):
    """Positional tensors to handles. **`None` is forwarded as-is** — Pyodide
    turns it into `undefined`, which is exactly the default in the signature over
    there."""
    return None if x is None else handle(wrap(x))


def trapezoid(y, x=None, dx=1.0, dim=-1, **kw):
    """Trapezoidal integration. The mean of each neighbouring pair times the
    spacing, summed.

    **The assembly used to be here.** A few lines slicing and adding, and the
    comment over in borch.ts said "building another one here makes two copies of
    the assembly". What that missed is that for anyone using borch.ts from
    TypeScript this name **did not exist at all** — it was not one copy, it was
    Python-only. The name goes over there and this side forwards.
    """
    return wrap(guarded(handle(wrap(y)).trapezoid,
                        _trapezoid_x(x), kw.get("dx", dx), dim))


def cumulative_trapezoid(y, x=None, dx=1.0, dim=-1, **kw):
    """The cumulative version. **The last value has to equal `trapezoid`** —
    that is the check."""
    return wrap(guarded(handle(wrap(y)).cumulativeTrapezoid,
                        _trapezoid_x(x), kw.get("dx", dx), dim))


# ── the **writing** side of indexing. The opposite of the reading side
# (`gather`). ───────────────────────────────────────────────────────────────

def _spread_index(index, dim, shape):
    """Spread 1-D indices into `shape`.

    The `index_add` family's indices point at **rows**, and the kernel takes an
    index per element. Placed on one axis and broadcast across the rest, the two
    become the same thing — no new kernel needed.
    """
    lifted = [1] * len(shape)
    lifted[dim] = int(handle(index).size)
    return broadcast_to(wrap(guarded(handle(index).reshape, _js_list(lifted))), shape)


def scatter(t, dim, index, src, reduce=None):
    """**Overwrites** at the positions the indices point at. On a collision the
    last write survives.

    `reduce` is torch's deprecated overload — `'add'` or `'multiply'`, combining
    onto what is already there. It goes to borch.ts's own `scatter`, which refuses
    to differentiate exactly where torch does.
    """
    t = wrap(t)
    if not isinstance(src, Tensor):
        src = zeros(*[int(n) for n in handle(index).shape]) + float(src)
    if reduce is not None:
        return wrap(guarded(handle(t).scatter, dim, handle(index), handle(src),
                            reduce))
    return wrap(guarded(handle(t).scatterSet, dim, handle(index), handle(src)))


def scatter_add(t, dim, index, src):
    """**Adds** at the positions the indices point at. Collisions accumulate —
    where it parts from `scatter`."""
    return wrap(guarded(handle(t).scatterAdd, dim, handle(index), handle(src)))


def index_add(t, dim, index, source, alpha=1):
    return wrap(guarded(handle(wrap(t)).indexAdd, dim, handle(wrap(index)),
                        handle(wrap(source)), alpha))


def index_copy(t, dim, index, source):
    return wrap(guarded(handle(wrap(t)).indexCopy, dim, handle(wrap(index)),
                        handle(wrap(source))))


def index_fill(t, dim, index, value):
    return wrap(guarded(handle(wrap(t)).indexFill, dim, handle(wrap(index)),
                        float(value)))


def take(t, index):
    """Takes from **the flattened tensor** — it has no notion of an axis."""
    h = handle(t)
    flat = wrap(guarded(h.reshape, _js_list([int(h.size)])))
    picked = wrap(guarded(handle(flat).indexSelect, 0,
                          handle(wrap(index).reshape(int(handle(index).size)))))
    return wrap(guarded(handle(picked).reshape,
                        _js_list([int(n) for n in handle(index).shape])))


def take_along_dim(t, indices, dim=None, out=None):
    _no_out(out)
    if dim is None:
        return take(t, indices)
    return wrap(guarded(handle(t).gather, dim, handle(indices)))


def searchsorted(sorted_sequence, values, side=None, right=False, **kw):
    """Where a value would be inserted into something sorted. **Which side of a
    tie is decided by two arguments together.**

    No kernel is needed — count "how many are smaller than me" and that is the
    position.

    torch takes the same thing under two names — the boolean `right` and the
    string `side`. Only `right` existed here and `side` went into `**kw` and was
    **quietly discarded.** The core was the same, and `bucketize(right=True)` was
    right on both sides from the start. Only one argument each is off, so the
    values look plausible.
    """
    _no_out(kw.get("out"))
    side = kw.get("side", side)
    right = kw.get("right", right)
    if side is not None:
        if side not in ("left", "right"):
            raise RuntimeError(
                f"side must be 'left' or 'right' (got {side!r}). "
                f"(torch: torch.searchsorted(): side can only be 'left' or 'right' "
                f"but got {side})")
        if right and side == "left":
            raise RuntimeError(
                "side and right contradict each other — give only one. "
                "(torch: torch.searchsorted(): side and right can't be set to "
                "opposites, got side of left while right was True)")
        right = side == "right"
    seq, want = wrap(sorted_sequence), wrap(values)
    n = int(handle(seq).size)
    m = int(handle(want).size)
    row = wrap(guarded(handle(seq).reshape, _js_list([1, n])))
    col = wrap(guarded(handle(want).reshape, _js_list([m, 1])))
    hit = (row <= col) if right else (row < col)
    counted = wrap(guarded(handle(hit).to, "float32")).sum(dim=1)
    return wrap(guarded(handle(counted).to, "int64")).reshape(
        *[int(v) for v in handle(want).shape])


def bucketize(values, boundaries, right=False, **kw):
    """**The argument order is reversed from `searchsorted`.** That is the whole
    difference between the two names."""
    _no_out(kw.get("out"))
    return searchsorted(boundaries, values, right=kw.get("right", right))


row_stack = vstack
multiply = mul
divide = div
subtract = sub
true_divide = div
concat = cat
concatenate = cat

# The comparisons' other names. Forwarded under the name in the table.
_COMPARE_ALIAS = {"greater": "gt", "greater_equal": "ge",
                  "less": "lt", "less_equal": "le", "not_equal": "ne"}


def where(cond, a, b):
    """torch is `where(condition, true, false)` and borch.ts is
    `true.where(condition, false)`. Without the reorder the true and false
    branches come out swapped — visible only by comparing values."""
    return guarded(handle(a).where, handle(cond), handle(b))


def layer_norm(x, normalized_shape=None, weight=None, bias=None, eps=1e-5):
    """torch takes **the shape to normalise over** and borch.ts takes a count.

    **Two things were wrong and one of them had no seat at all.**

    `weight` and `bias` were in the signature and the body never read them, so
    `F.layer_norm(x, shape, w, b)` came back unscaled and unshifted — real numbers,
    the right shape, no exception, and a transformer block silently missing its
    learned affine.

    And the fold was `layerNorm(-len(shape))`, which folds **one axis, that far from
    the end** — for `(3, 4)` it took the mean over axis −2 alone. torch folds the
    last two *together*, which is `layerNormOver(2)` over there; the two agree at one
    axis, which is every case that asked.

    The refusals come from the core so there are not two copies of four wordings.
    """
    from borch._ops import _layer_norm_checks

    h = handle(x)
    full = [int(n) for n in h.shape]
    shape = _layer_norm_checks(
        full, normalized_shape,
        *[None if v is None else [int(n) for n in handle(v).shape]
          for v in (weight, bias)])
    out = wrap(guarded(h.layerNormOver, len(shape), eps))
    if weight is not None:
        out = out * weight
    return out + bias if bias is not None else out


def repeat_interleave(x, repeats, dim=None):
    """With no `dim`, torch repeats **after flattening.**"""
    h = handle(x)
    if dim is None:
        h = h.reshape(_js_list([int(h.size)]))
        dim = 0
    return guarded(h.repeatInterleave, repeats, dim)


def flip(x, dims=None, **kw):
    """torch takes **a list** of axes and borch.ts takes one at a time. Flipped
    in turn."""
    dims = kw.get("dims", dims)
    if isinstance(dims, int):
        dims = [dims]
    out = handle(x)
    for d in (dims or []):
        out = out.flip(d)
    return wrap(out)


def pow(x, exponent):                                    # noqa: A001
    """A numeric exponent means `powScalar` — an integer exponent is expanded
    into multiplications, which keeps the sign."""
    if isinstance(exponent, Tensor):
        return guarded(handle(x).binary, "pow", exponent._h)
    return guarded(handle(x).powScalar, exponent)


def pad(x, pairs, mode="constant", value=0.0, **kw):
    """torch's `F.pad` takes pairs **from the last axis** —
    `(left, right, top, bottom, …)`.

    **A place that accepted `mode` and never used it.** It was in the parameters
    and never went any further down, so asking for `reflect` produced constant
    padding — not an exception but **quietly a different value**, and it surfaced
    only once the golden had a case asking for that mode. The same kind of thing
    as JS discarding surplus arguments, discarded on the Python side this time.
    """
    value = kw.get("value", value)
    return wrap(guarded(handle(x).padND, _js_list(list(pairs)), mode, float(value)))


def split(x, size, dim=0):
    """**The argument order is reversed.** torch is `split(size, axis)` and
    borch.ts is `splitSize(axis, size)`. Forwarded as-is the size lands in the
    axis slot and it blows up somewhere unrelated — that was
    `axis 2 of size 0 is not divisible by undefined`."""
    return [wrap(t) for t in handle(x).splitSize(dim, size)]


def chunk(x, chunks, dim=0):
    """**Not `split`.** That one has to divide evenly and `chunk` does not —
    torch splitting 3 into 2 gives 2 and 1, and splitting 2 into 5 gives **two
    pieces** (measured).

    borch.ts has `chunk` properly and this was forwarding to `split`. Measured
    only at sizes that divide evenly, the two functions look the same.
    """
    return [wrap(t) for t in handle(x).chunk(chunks, dim)]


def clamp(x, min=None, max=None):                        # noqa: A002
    """**Giving only one side is common.** borch.ts takes both as
    `clamp(low, high)`, and an `undefined` on one side travels all the way into
    the shader and WGSL refuses it. So it is split into `clampMin` and
    `clampMax` here."""
    h = handle(x)
    if min is not None and max is not None:
        return guarded(h.clamp, min, max)
    if min is not None:
        return guarded(h.clampMin, min)
    if max is not None:
        return guarded(h.clampMax, max)
    return wrap(h)


clip = clamp


def aminmax(x, out=None):
    """The minimum and the maximum together. borch.ts keeps them separately, so
    they are paired here."""
    _no_out(out)
    h = handle(x)
    return _MinMax(wrap(h.amin()), wrap(h.amax()))


class _MinMax:
    """`aminmax`'s answer. torch calls them `.min` and `.max`."""

    __slots__ = ("min", "max")

    def __init__(self, lo, hi):
        self.min, self.max = lo, hi

    def __iter__(self):
        yield self.min
        yield self.max

    def __getitem__(self, i):
        return (self.min, self.max)[i]


# **`torch.linalg` is a namespace.** Most of it exists as tensor methods, and the
# ones whose sizes depend on the values (`cholesky`, `svd`, `eigh`) are
# asynchronous, so `settle` waits on them.
# **Answered by borch.ts's `linalg` namespace rather than by a tensor method.**
#
# Each of these five has an argument the method next to it does not take, and every one
# of those arguments changes the answer:
#
#     norm         `ord` with no `dim` on a matrix is the largest singular value there
#                  and the elementwise p-norm on the method — 16.848 against 16.882
#     lu           `pivot=False` is a different factorisation, refused
#     lu_solve     `left`/`adjoint` are different systems, and each is answered now
#     tensorsolve  `dims` reorders the axes before the fold, and it is answered now
#     lstsq        `rcond` is the cutoff and `driver` picks among four algorithms
#
# Reached through the method the extra words are handed to JavaScript and dropped, so
# the default answer comes back under the name of a computation nobody ran. The values
# are named here rather than derived because `lu_solve` is `luSolveFactored` over there.
_VIA_NAMESPACE = {
    "norm": "norm",
    "lu": "lu",
    "lu_solve": "luSolve",
    "tensorsolve": "tensorsolve",
    "lstsq": "lstsq",
}


class _Linalg:
    # Code that meets a singular matrix wraps it in
    # `except linalg.LinAlgError`. Without this here that wrapper cannot find the
    # name and the program dies — needed before any value is.
    LinAlgError = _LinAlgError

    def lstsq(self, a, b, rcond=None, *, driver=None):
        """torch gives an object holding `.solution` — borch.ts gives the answer
        directly.

        **This one cannot go through `__getattr__` even though it is in
        `_VIA_NAMESPACE`**, because the result has to be dressed as torch's named
        tuple before it is handed back. It reaches the same namespace function the
        table names; only the wrapping is here.

        `rcond` and `driver` were absent, so `linalg.lstsq(A, B, 0.9)` was **a third
        argument JavaScript discarded** and the uncut answer came back — 3.5 where
        torch says 0.77, under an argument whose whole purpose is to move it.
        """
        from ._base import _Fields
        ns = getattr(_js.borch, "linalg", None)
        got = settle(ns.lstsq(handle(a), handle(b), rcond, driver or "gelsy"))
        out = _Fields.__new__(_Fields)
        object.__setattr__(out, "_order", ["solution"])
        object.__setattr__(out, "_d", {"solution": got})
        return out

    def matrix_power(self, a, n):
        return matrix_power(a, n)

    def multi_dot(self, mats):
        """**The first argument is a list, not a tensor** — it cannot take the
        general path below.

        The grouping does not change the value (multiplication is associative).
        Only the operation count changes, so they are multiplied in order here.
        """
        out = wrap(mats[0])
        for m in mats[1:]:
            out = wrap(guarded(handle(out).mm, handle(m)))
        return out

    def diagonal(self, a, offset=0, dim1=-2, dim2=-1):
        """**The default axes differ from `torch.diagonal`'s.**

        This one is the last two axes and that one is the first two. Sent down
        the general path, borch.ts's defaults (`0, 1`) are used and 3-D quietly
        gives a different shape.
        """
        return wrap(guarded(handle(a).diagonal, offset, dim1, dim2))

    def tensorsolve(self, a, b, dims=None):
        """**The refusal moved to borch.ts and this forwards to it.**

        It used to raise here, which was right while borch.ts had no seat for `dims`
        at all — but it meant two libraries deciding one question, and a caller
        reaching borch.ts directly got the unmoved answer in silence. One rule, one
        place; this only carries the word across.
        """
        ns = getattr(_js.borch, "linalg", None)
        return wrap(guarded(ns.tensorsolve, handle(a), handle(b), dims))

    def tensorinv(self, a, ind=2):
        return wrap(guarded(handle(a).tensorInv, ind))

    # ── `_ex`, LDL and reflectors ──────────────────────────────────────────
    #
    # They cannot take the general path — the arguments are several tensors
    # (`ldl_solve`) or they return several named fields. `guarded` supplies the
    # field names.

    def lu_factor_ex(self, a, pivot=True, check_errors=False):
        return guarded(handle(a).luFactorEx)

    def ldl_factor(self, a, hermitian=False):
        return guarded(handle(a).ldlFactor)

    def ldl_factor_ex(self, a, hermitian=False, check_errors=False):
        """`ldl_factor` with `info` attached. It is always 0 here — the bad cases
        are refused.

        **The three fields were being stood up by hand.** A `_Fields` was built
        directly, with borch.ts's values for `LD` and `pivots` and a numpy scalar
        wedged into `info`, and then the golden cannot catch that this name is
        missing from borch.ts — every case goes through here. The same place as
        `trapezoid`, fixed the same way.
        """
        return guarded(handle(a).ldlFactorEx)

    def ldl_solve(self, ld, pivots, b, hermitian=False):
        # **The pivots were not handed over**, which was safe only while the
        # factorisation refused every matrix that needed a swap. They are the
        # difference between a right answer and a plausible one now.
        return wrap(guarded(handle(ld).ldlSolve, handle(pivots), handle(b)))

    def householder_product(self, a, tau):
        return wrap(guarded(handle(a).householderProduct, handle(tau)))

    def qr(self, a, mode="reduced"):
        """torch takes **a word** and borch.ts takes a boolean.

        `mode` had a row in `_SIGNATURE` so that it would not be discarded, and the
        note above that row names this very call — *`qr(mode="complete")` produces the
        reduced form*. It was written as the hazard and the row underneath was the
        hazard: forwarded as it arrives, `"complete"` is a non-empty string, which is
        true, so it asked for `reduced` and got the opposite of what was written.

        Not an exception. A `Q` one column narrower than it should be, and the values
        in the block the two share agree — so the only thing that says so is a shape,
        and the only check that looks is this binding's golden.
        """
        return guarded(handle(a).qr, mode != "complete")

    def __getattr__(self, name):
        # torch's abbreviations. `pinv` was empty for a long time and never
        # surfaced because the golden always asked under the long name
        # (`L.pinverse`) — it appeared as soon as one more spelling called it.
        # **`svd` here is `torch.linalg.svd` and borch.ts calls that `linalgSvd`.**
        # `svd` over there is `torch.svd`, which is a different function: reduced by
        # default, and its third field is `V` rather than `Vh`. Routed to the wrong
        # one, `linalg.svd(A)` came back (3, 2) where it must be (3, 3) and
        # `linalg.svd(A, full_matrices=False)` came back (3, 3) — **each getting the
        # other's answer**, because `some` is `full_matrices` negated.
        #
        # The values in the overlapping block agree, so the only thing that says so
        # is a shape, and the only check that looks is this binding's golden.
        # **`matmul` is not `mm` any more.** It was, while borch.ts had only the
        # two-dimensional kernel; that side's `matmul` batches and broadcasts now, and
        # the same stale rename at the top level sent six golden cases into
        # `mm is 2-D by 2-D`.
        js_name = camel({"inv": "inverse", "pinv": "pinverse", "svd": "linalgSvd",
                         "matrix_rank": "matrixRank"}.get(name, name))

        def call(x, *args, **kw):
            # **`linalg.norm` is not the `norm` method and must not be routed to it.**
            # With an `ord` and no `dim` on a matrix torch takes the largest singular
            # value where the method takes the elementwise p-norm — 16.848 against
            # 16.882 on `[[1..9]]`, near enough to read as rounding. The dispatch lives
            # in borch.ts's `linalg` namespace, which `index.ts` exports, so this reaches
            # the free function rather than reimplementing the rule a third time.
            #
            # All three implementations had it wired to the elementwise one. The golden
            # case written for it caught borch.ts, then the core, then this.
            #
            # **The list grew and became a table.** `lu`, `luSolve` and `tensorsolve`
            # each carry an argument in order to refuse it — `pivot`, `left`/`adjoint`,
            # `dims` — and every one of those refusals lives in borch.ts's `linalg`
            # namespace, not on the tensor. Reached by method the words are received by
            # JavaScript and dropped, and the default answer comes back under the name
            # of a computation nobody performed. That is the same failure `norm` was,
            # one argument further in.
            if name in _VIA_NAMESPACE:
                ns = getattr(_js.borch, "linalg", None)
                if ns is not None:
                    # The free function takes the tensor first; the method took it as
                    # the receiver.
                    return guarded(getattr(ns, _VIA_NAMESPACE[name]), handle(x),
                                   *positional(f"linalg.{name}", args, kw))
            fn = getattr(handle(x), js_name, None)
            if fn is None:
                raise AttributeError(f"borch.ts does not have `{js_name}` (linalg.{name})")
            # **Named arguments must not be discarded.**
            # `qr(mode="complete")` and `svd(full_matrices=False)` are the
            # places, and discarded they give not an exception but **quietly a
            # different answer under the defaults** — the kind only a value
            # comparison catches.
            # **The signature row has to follow the routing.** `positional` keys
            # off the name it is given, and `linalg.svd` reaches borch.ts's
            # `linalgSvd`, whose argument is `full_matrices` — while `svd`'s row is
            # `torch.svd`'s `(some, compute_uv)`. Keyed by the python name it would
            # read the other function's row and put `full_matrices` into `some`,
            # which is that argument negated.
            key = js_name if js_name in _SIGNATURE else name
            return guarded(fn, *positional(key, args, kw))

        call.__name__ = name
        return call


linalg = _Linalg()
