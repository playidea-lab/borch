"""borch — a thin PyTorch-shaped layer over numpy.

For practising PyTorch **syntax** in a browser (Pyodide) with nothing installed.
torch is not ported to wasm — hundreds of MB of native code, hand-tuned AVX and
NEON kernels that do not carry over to wasm SIMD, and OpenMP threads that want
headers Pyodide does not ship. And **none of that is needed to learn the
syntax.** numpy is enough.

## The design principle — an absent feature beats a wrong answer

A subset that behaves even slightly differently from the real thing teaches the
student something false. So **what is absent throws rather than approximating.**
It stops loudly rather than quietly producing a different value.

Outside the supported range a `BorchError` is raised, and the message says to do
it on your own machine.

## How that is guaranteed

Two layers.

1. `borch-check` — runs the same **lab tests** against real torch and against the
   subset.
2. `borch-diff` — compares **the numbers of the same operation directly**,
   independently of the labs (`tests/test_borch_diff.py`). The first alone sees
   only the paths the labs walk, which was 73% of the subset, and backpropagation
   sat in that blind spot.

The two checks now cover 86%. What is left is places where no value is at stake,
such as `__repr__`.

Something `borch-diff` actually caught: BatchNorm uses the biased variance for
the normalisation and the unbiased one for updating running_var. Biased in both
places is off by 2.6%.
"""


import math as _math

import numpy as _np

from ._base import (
    BorchError, Size, _DEFAULT_DTYPE, _LINE_WIDTH, _NP_TO_DTYPE,
    _PRINT_PRECISION, __all__, _float_formatter, _like_torch, _resolve, _tensor_repr,
    _tensor_str, _unsupported, bool_, device, dtype, float32, float64, int64,
    bfloat16, chalf, complex32, float16, half, int16, int32, long,
    set_printoptions, short,
    # The complex dtype names. `complex128` and `cdouble` exist **as names
    # only** — trying to make one stops at the gate in `Tensor.__init__`.
    cdouble, cfloat, complex128, complex64,
    # The top-level numeric constants. **The five a coverage table counting only
    # `callable` could not see.**
    e, inf, nan, newaxis, pi,
)
from ._tensor import (
    Tensor, _CATEGORY, _DEFAULT_BY_CATEGORY, _DataDescriptor, _GradMode, _MinMax, _RANK,
    _category, _grad_mode, _no_bool_subtract, _promote, _scalar_category, _unbroadcast,
    result_type,
)
from ._ops import _out                                   # noqa: E402
from ._ops import (
    Generator, _Cuda, _ERF_A, _ERF_P, _INPLACE_UNARY, _Linalg, _Lstsq, _Namespace, _SVD,
    _abs, _binary_math, _col2im, _compare, _cum_extreme, _diagonal_scatter, _erf64,
    _erfc_pos, _expand_reduced, _gelu, _im2col, _index_at, _index_for,
    _make_inplace, _mat, _nan_mask, _negate, _nm, _one_plus_erf64, _pad2d, _pair, _pick,
    _pool_1d_over_last, _pool_all, _rng, _running_idx, _slice_at, _spread_max,
    _unary, _wrap, _zero_grad, abs, absolute, acos, acosh,
    adaptive_avg_pool2d, allclose, amax, amin, aminmax, arange, arccos, arccosh, arcsin,
    arcsinh, arctan, arctanh, argsort, argwhere, as_tensor, asin, asinh, atan, atan2,
    atanh, atleast_1d, atleast_2d, atleast_3d, avg_pool2d, bincount, bmm, cat, ceil,
    cholesky, chunk, clamp, clip, conv1d, conv2d, conv3d, copysign, cos, cosh,
    cosine_similarity, count_nonzero, cuda, cummax, cummin, cumprod, cumsum, deg2rad,
    det, diag, diagflat, diagonal, diff, dist, dot, dropout, dsplit, eigh, einsum, elu,
    embedding, empty, eq, equal, erf, erfc, exp, exp2, expand, expand_as, expm1, eye,
    constant_pad_nd, dequantize, fake_quantize_per_channel_affine,
    fake_quantize_per_tensor_affine, igamma, igammac, polygamma, resize_as_,
    fft, fix, flip, fliplr, flipud, floor, frac, from_numpy, full, full_like, gather, ge,
    gelu, gt, heaviside, hsplit, hypot, index_select, interpolate, inverse, isfinite,
    isinf, isnan, kthvalue, l1_loss, layer_norm, ldexp, le, leaky_relu, linalg,
    linspace, log, log10, log1p, log2, log_softmax, logaddexp, logaddexp2, logdet,
    logical_and, logical_not, logical_or, logit, logsumexp, lstsq, lt, manual_seed,
    masked_fill, masked_select, matmul, matrix_exp, matrix_power, matrix_rank,
    max_pool1d,
    max_pool2d, max_pool3d, maximum, median, minimum, mm, movedim, msort, multinomial,
    nanmean, nanquantile, nansum, narrow, ne, neg, negative, nll_loss, no_grad, nonzero,
    norm, normalize, ones, ones_like, outer, pad, pinverse, positive, pow, prod, qr,
    quantile, rad2deg, rand, randint, randn, randperm, ravel, reciprocal, relu, repeat,
    repeat_interleave, reshape, roll, rot90, round, rsqrt, select, sgn, sigmoid,
    sign, signbit, silu, sin, sinc, sinh, slogdet, smooth_l1_loss, softmax, solve, sort,
    split, sqrt, square, stack, svd, swapaxes, swapdims, tan, tanh, tensor, tile, topk,
    trace, tril, triu, trunc, unbind, unflatten, unfold, unique, unsqueeze, vsplit,
    where, xlogy, zeros, zeros_like,
    # The ones torch offers under a second name — a name attached to what an
    # operator already does.
    add, adjoint, block_diag, broadcast_shapes, broadcast_tensors, broadcast_to,
    column_stack, concat, concatenate, div, divide, dstack, floor_divide, fmod,
    greater, greater_equal, hstack, less, less_equal, moveaxis, mul, multiply,
    not_equal, remainder, row_stack, rsub, sub, subtract, t, true_divide, vstack,
    # The ones with no computation of their own.
    cross, empty_like, float_power, fmax, fmin, inner, isclose, isin, isneginf,
    isposinf, isreal, kron, lerp, logical_xor, logspace, meshgrid, nan_to_num,
    rand_like, randint_like, randn_like, scalar_tensor, std_mean, var_mean, vdot,
    # The writing side of indexing.
    bucketize, index_add, index_copy, index_fill, scatter, scatter_add,
    searchsorted, take, take_along_dim,
    # The numeric family. The last three are computed as series.
    cdist, corrcoef, cov, cumulative_trapezoid, digamma, erfinv, lgamma,
    tensordot, trapezoid,
    # QR in reflector form. The partner to `linalg.householder_product`, so it
    # exists at top level too.
    geqrf,
    # **The names that exist only at top level.** Some have a different
    # signature from `F`'s, so the positions are moved.
    alpha_dropout_, dropout_, feature_alpha_dropout_, feature_dropout,
    feature_dropout_, grid_sampler, nan_to_num_,
    # The top-level names that are **the same computation** as `F`'s (confirmed
    # by measurement).
    alpha_dropout, bilinear, celu_, channel_shuffle, embedding_bag,
    feature_alpha_dropout, max_pool1d_with_indices, pixel_shuffle,
    pixel_unshuffle, rrelu, rrelu_, selu_, threshold_,
    # **The two with a different signature.** The top level is raw ATen, so the
    # argument order and the enums differ — the `_aten` versions take that place
    # and `F`'s keep their own names.
    batch_norm_aten as batch_norm, ctc_loss_aten as ctc_loss,
    # Gradient modes.
    enable_grad, inference_mode, is_grad_enabled, is_inference,
    is_inference_mode_enabled, set_grad_enabled,
    # Random state.
    get_rng_state, initial_seed, seed, set_rng_state,
    # Introspection.
    can_cast, finfo, get_default_dtype, iinfo, is_distributed, is_floating_point,
    is_nonzero, is_same_size, is_signed, is_storage, is_tensor, promote_types,
    set_default_dtype, typename,
    # Bitwise operations and integer maths. On `bool` they become logical
    # operations — torch looks at the dtype.
    bitwise_and, bitwise_left_shift, bitwise_not, bitwise_or,
    bitwise_right_shift, bitwise_xor, gcd, gcd_, lcm, lcm_,
    arctan2, clamp_max, clamp_max_, clamp_min, clamp_min_, detach_, fill,
    frexp, i0, i0_, logcumsumexp, mvlgamma, nextafter,
    # Window functions. `periodic` is the default and it adds one to the
    # length.
    bartlett_window, blackman_window, hamming_window, hann_window,
    kaiser_window,
    # Shape and indexing. **`as_strided` is a view in torch and a copy here** —
    # the details are written at that place in `_ops.py`.
    as_strided, as_strided_, as_strided_scatter, diag_embed, diagonal_scatter,
    select_scatter, slice_scatter, split_with_sizes, tensor_split,
    unique_consecutive, unravel_index,
    index_put, index_put_, index_reduce, masked_scatter, masked_scatter_,
    put, renorm, scatter_reduce,
    cartesian_prod, chain_matmul, combinations, ger, mv, tril_indices,
    triu_indices, vander,
    # The addmm family. **The in-place versions are not exposed** — torch keeps
    # those as methods only. `addmv_` is the single exception and it alone is
    # here (measured).
    addbmm, addcdiv, addcmul, addmm, addmv, addmv_, addr, baddbmm, sspaddmm,
    # Top-level linear algebra. **The two whose names collide with `linalg`'s
    # have their positions moved** — that side's `lu` spreads `P`, `L` and `U`
    # while this one gives a single packed matrix, and this `lu_solve` takes the
    # right-hand side first.
    cholesky_inverse, cholesky_solve, lobpcg, lu_top as lu,
    lu_solve_top as lu_solve, lu_unpack, orgqr, ormqr, pca_lowrank,
    svd_lowrank, triangular_solve,
    # Statistics. **The four random ones cannot have their values pinned and
    # their extremes are deterministic** — that is what the golden asks about.
    # `stft`, `istft` and `hash_tensor` are names that refuse (no complex, no
    # uint64).
    bernoulli, binomial, gradient, hash_tensor, histc, histogram, histogramdd,
    istft, mode, nanmedian, nonzero_static, normal, poisson, stft, trapz,
    # **The names that have an answer even without complex numbers.** Over the
    # reals the `conj` family is the identity and `is_complex` is false. `imag`
    # alone refuses, and **torch itself does that** (measured).
    angle, asarray, conj, conj_physical, conj_physical_, empty_permuted,
    empty_strided, frombuffer, imag, is_complex, is_conj, is_neg, real,
    resolve_conj, resolve_neg,
    # **Complex numbers.** `complex128` is a name and trying to make one stops —
    # because there is no `float64`. The gradient convention is
    # `∂L/∂re + i·∂L/∂im`, pinned by measurement.
    complex, polar, view_as_complex, view_as_real,
    # **This file uses the builtin `range` in 91 places** — inside `_ops` it has
    # a different name and it is exposed as `range` only on the way out. The same
    # place as `lu` and `lu_solve`.
    range_top as range,
    # **The two distances that exist at top level as well.** In torch these two
    # are **literally the same function** as `F`'s (`torch.pdist is F.pdist` is
    # true).
    #
    # The seven losses that surfaced alongside (`kl_div`, `poisson_nll_loss`, …)
    # are not exposed — the top-level ones are raw ATen operations, so **the
    # default reduction is `none` and `reduction` is an integer.**
    # `torch.kl_div(a, b)` gives `[2,2]` and `F.kl_div(a, b)` gives a scalar.
    # Put down as a friendly alias, they diverge starting at the shape.
    pairwise_distance, pdist,
)
from ._nn import (
    AdaptiveAvgPool2d, AvgPool2d, BCELoss, BCEWithLogitsLoss, BatchNorm1d, BatchNorm2d,
    BatchNorm3d, Conv1d, Conv2d, Conv3d, CrossEntropyLoss, Dropout, ELU, Embedding,
    Flatten, GELU, GRU, Identity, L1Loss, LSTM, LayerNorm, LeakyReLU, Linear,
    LogSoftmax, MSELoss, MaxPool1d, MaxPool2d, MaxPool3d, Module, ModuleDict,
    ModuleList, ParameterDict, ParameterList,
    MultiheadAttention, NLLLoss, Parameter, RNN, ReLU, Sequential, SiLU, Sigmoid,
    SmoothL1Loss, Softmax, Tanh, Transformer, TransformerDecoder,
    TransformerDecoderLayer, TransformerEncoder, TransformerEncoderLayer, Unflatten,
    Upsample, _Activation, _Functional, _NN, _RNNBase, _apply_mask, _cls,
    _nn_unsupported, _split_heads, nn, one_hot,
    # **The eight top-level recurrent ones.** torch offers both the layer
    # (`nn.LSTM`) and the function (`torch.lstm`), and what the layer calls
    # inside is the function. The difference is that they take the weights as a
    # list.
    gru, gru_cell, lstm, lstm_cell, rnn_relu, rnn_relu_cell, rnn_tanh,
    rnn_tanh_cell,
)
from ._optim import (
    Adadelta, Adagrad, Adam, AdamW, Adamax, ChainedScheduler, ConstantLR,
    CosineAnnealingLR, CosineAnnealingWarmRestarts, ExponentialLR, LambdaLR, LinearLR,
    MultiStepLR, MultiplicativeLR, NAdam, OneCycleLR, Optimizer, PolynomialLR, RAdam,
    RMSprop, ReduceLROnPlateau, SGD, SequentialLR, StepLR, _LRScheduler, _Optim,
    _Scheduler, optim,
    ASGD, Adafactor, LBFGS, Rprop, CyclicLR,
)
from ._data import (
    BatchSampler, ChainDataset, ConcatDataset, DataLoader, Dataset, IterableDataset,
    RandomSampler, Sampler, SequentialSampler, StackDataset, Subset,
    SubsetRandomSampler, TensorDataset, WeightedRandomSampler, _Utils, _UtilsData,
    default_collate, random_split, utils,
)
from ._rnn import (
    _NnUtils, _NnUtilsRnn, pad_sequence,
)
from ._serialize import (
    load, save,
)

# ==================================================== exposing them as methods
#
# torch code mixes `torch.sin(x)` and `x.sin()`. Having module functions only,
# a tutorial using dot notation stopped with an `AttributeError` — **a feature
# that exists and is one calling convention short.**
#
# This list was not chosen by hand. torch was asked whether `x.f(...)` and
# `torch.f(x, ...)` give the same value, and only the ones it said yes to are
# here. 62 came back equal and one differed — `where` (see below). Attaching one
# of those blindly gives a quietly wrong answer.

_AS_METHOD = (
    "allclose", "argsort", "bmm", "ceil", "chunk", "clamp", "cos", "cosh", "cumprod",
    "cumsum", "diag", "dot", "eq", "equal", "erf", "flip", "floor", "gather", "ge",
    "gt", "isfinite", "isinf", "isnan", "le", "log10", "log2", "lt", "maximum",
    "median", "minimum", "mm", "movedim", "multinomial", "narrow", "ne", "neg",
    "norm", "outer", "pow", "prod", "reciprocal", "relu", "roll", "round", "rsqrt",
    "sigmoid", "sign", "sin", "sinh", "softmax", "sort", "split", "square", "tan",
    "tanh", "tile", "topk", "trace", "tril", "triu", "unbind", "unique",
    # The maths group. Confirmed the same way — asked of torch, and only what
    # it said was equal.
    "acos", "acosh", "arccos", "arccosh", "arcsin", "arcsinh", "arctan", "arctanh",
    "asin", "asinh", "atan", "atan2", "atanh", "absolute", "clip", "copysign",
    "deg2rad", "erfc", "exp2", "expm1", "fix", "frac", "heaviside", "hypot", "ldexp",
    "log1p", "logaddexp", "logaddexp2", "logit", "negative", "positive", "rad2deg",
    "sgn", "signbit", "sinc", "trunc", "xlogy",
    # The reduction group. torch exposes these sixteen as methods too.
    "amax", "amin", "aminmax", "argwhere", "cummax", "cummin", "diff", "dist",
    "kthvalue", "logsumexp", "msort", "nanmean", "nanquantile", "nansum", "nonzero",
    "quantile",
    # The shape group. Of these, `expand`, `repeat`, `ravel`, `select`, `unfold`
    # and `expand_as` **have no module function in torch and exist as methods
    # only** — places with a single calling convention.
    "diagflat", "diagonal", "dsplit", "expand", "expand_as", "fliplr", "flipud",
    "hsplit", "ravel", "repeat", "rot90", "select", "swapaxes", "swapdims",
    "unflatten", "unfold", "vsplit",
    # The three the sister library had as methods and this had as functions
    # only. torch offers them as methods too.
    "index_select", "masked_select", "repeat_interleave", "masked_fill",
    # The writing side of indexing. torch offers all of them as methods as
    # well — `x.scatter_(…)` is the form.
    "scatter", "scatter_add", "index_add", "index_copy", "index_fill", "take",
    "take_along_dim",
)

for _method in _AS_METHOD:
    if not hasattr(Tensor, _method):
        setattr(Tensor, _method, globals()[_method])


def _where_method(self, condition, other):
    """**The argument order differs from the function's.**
    `x.where(condition, y)` is `torch.where(condition, x, y)`.

    Attached blindly like the rest, `x` lands in the condition slot and the
    answer is quietly wrong. It was found by asking torch; reading down the list
    by eye would not have found it.
    """
    return where(condition, self, other)


Tensor.where = _where_method


# ── the ones in `nn.functional` that **torch also keeps at top level** ──────
#
# torch mostly keeps the layer functions under `F.` alone, and puts some of them
# on `torch.` as well. Which is which is that side's history rather than a rule,
# so **it has to be asked** — `tests/torch_gap.py` produces the list.
#
# The things all existed already and only the names were missing. Without the
# name that code does not run.
for _name in ("conv_transpose1d", "conv_transpose2d", "conv_transpose3d",
              "group_norm", "instance_norm", "rms_norm",
              "celu", "selu", "prelu", "hardshrink", "threshold",
              "avg_pool1d", "adaptive_avg_pool1d", "adaptive_max_pool1d"):
    globals()[_name] = getattr(nn.functional, _name)


# ── methods exposed **as module functions too.** Exactly the opposite
# direction from `_AS_METHOD`. ──────────────────────────────────────────────
#
# torch offers nearly everything under two names — `x.sum()` and `torch.sum(x)`.
# Only one side existed here, so `torch.sum(x, dim=1)` stopped with an
# `AttributeError`. The golden did not catch it either — it turned up while
# writing cases, because the table held no case of that form at all.
#
# The list is not written by hand. The intersection of **what torch also offers
# as a module function** and what we have as a method is the answer, and the
# machine produces it. Written by hand, the next method added forgets this
# side.
# ── torch's **dtype aliases** go down first. They have to sit above the loop
# below. ────────────────────────────────────────────────────────────────────
#
# `torch.float`, `torch.double`, `torch.int` and `torch.bool` are dtypes rather
# than functions. And `float`, `double`, `int` and `bool` are also Tensor
# methods, so the loop below was filling these names with **functions built from
# the methods.** That made the textbook-common `zeros(2, dtype=torch.float)` stop
# with `'function' object has no attribute 'np'` — the dtype it points at was
# perfectly present and only the name was covered over.
#
# The loop skips on `_name in globals()`, so **putting them down here is the
# fix.** The method side (`x.float()`) is untouched — these names go into the
# module slot only.
#
# `int` alone is not an alias. **`torch.int` is int32 and there is no such
# storage here** — the name is kept and using it stops (`int32` is an
# `_AbsentDtype`).
float = float32
double = float64
bool = bool_
# The four below have no dtype to point at — names only, and using one stops.
int = int32
half = float16
short = int16
chalf = complex32


def _as_function(name):
    """Wrap a method as a function taking it as the first argument."""
    def call(t, *args, **kwargs):
        return getattr(_wrap_tensor(t), name)(*args, **kwargs)
    call.__name__ = name
    call.__doc__ = f"The same as `x.{name}(...)`. torch offers both."
    return call


def _wrap_tensor(t):
    return t if isinstance(t, Tensor) else tensor(t)


# **A name torch exposes as a different kind of thing is not taken.**
#
# This loop puts method names straight into the module slot, and there are
# places where that name is not a function in torch. Then **a function sits**
# on our side, and somebody using the name for its real purpose sees an error one
# step displaced — `dtype=torch.float` stopping with
# `'function' object has no attribute 'np'` was that shape.
#
# The eight dtypes (`float`, `bool`, `half`, …) are blocked **by being put down
# above**, and the five below are what is left. Removed by hand three times, so
# this time it is written as a rule.
#
# **This table is written without looking at torch** — the core does not lean on
# torch. Instead `tests/test_module_names.py` holds torch and checks that this
# table is neither stale nor short.
_NOT_OURS = {
    "cpu": "a namespace in torch — there is one device to choose, so we have none",
    "storage": "a namespace in torch — there is nowhere here to look into a storage layer",
    "mtia": "a namespace in torch — a different accelerator",
    "xpu": "a namespace in torch — a different accelerator",
    "qscheme": "a class in torch — a quantisation scheme, and that dtype is absent",
}

for _name in dir(Tensor):
    if _name.startswith("_") or _name in globals() or _name in _NOT_OURS:
        continue
    if callable(getattr(Tensor, _name, None)):
        globals()[_name] = _as_function(_name)

# **This has to be near the end of the file.** The loop above puts `sum`, `min`,
# `max`, `all` and `any` into module scope, and those are Python builtins too.
# Code below this calling the builtins quietly calls something else — the binding
# went through this once with `bool`.


# ── `out=` — writing into a tensor made in advance ──────────────────────────
#
# For these names torch writes into the tensor it was handed rather than making
# a new result. **The saving does not happen here** — it computes and then moves,
# so the allocation occurs anyway. The two observable things are real, though:
# the destination changes, and what comes back is **the same object.** Code leans
# on those two, so this keeps a fact rather than an imitation.
#
# **The list was built by measuring.** Going by the `out=None` in the docstrings
# it is wider — `rand_like`, `zeros_like`, `median` and `where` are written there
# and the actual overload does not accept it. So the split came from **actually
# calling torch**, and `tests/test_out_names.py` measures that split again. The
# core does not lean on torch, so the table lives here and the comparison lives
# there.
_TAKES_OUT = frozenset("""
    add addbmm addcdiv addcmul addmm addmv addr all amax amin any arange
    baddbmm bitwise_and bitwise_left_shift bitwise_not bitwise_right_shift
    bitwise_xor bmm bucketize cat ceil cholesky cholesky_inverse
    cholesky_solve clamp clip column_stack complex concat concatenate
    conj_physical copysign cos cosh cross cumprod cumsum deg2rad diag
    digamma div divide dot dstack empty eq erf erfc erfinv exp exp2 expm1
    eye fix float_power floor floor_divide fmax fmin fmod frac full gather
    gcd ge ger greater greater_equal gt heaviside histc hstack hypot i0
    igamma igammac index_select inner inverse isneginf isposinf kron lcm
    ldexp le lerp less less_equal lgamma linspace log log1p logaddexp
    logaddexp2 logcumsumexp logical_and logical_not logical_or logit
    logspace logsumexp lt lu_solve masked_select matmul matrix_power max
    maximum mean min minimum mm mul multinomial multiply mv mvlgamma
    nan_to_num nanmean nanquantile ne neg negative nextafter nonzero normal
    not_equal ones ormqr outer polar polygamma pow quantile rand randint
    randn randperm range reciprocal remainder renorm round row_stack rsqrt
    searchsorted sgn sigmoid sign signbit sin sinc sinh sqrt stack std sub
    subtract take_along_dim tan tanh tril triu trunc var vdot vstack xlogy
    zeros
""".split())

# The ones taking **several**, as in `out=(values, indices)`. The same rule with
# more than one slot.
_TAKES_OUT_TUPLE = frozenset("""
    aminmax cummax cummin frexp geqrf histogram kthvalue mode sort svd topk
    triangular_solve
""".split())


def _accepts_out(fn, name):
    """Take `out=` and hand it to `_out`. **Not written per function** — fixing
    a hundred and seventy-two by hand leaves one of them out, and the one left
    out swallows quietly."""
    def call(*args, **kwargs):
        out = kwargs.pop("out", None)
        return _out(fn(*args, **kwargs), out, name)
    call.__name__ = getattr(fn, "__name__", name)
    call.__doc__ = getattr(fn, "__doc__", None)
    return call


for _name in _TAKES_OUT | _TAKES_OUT_TUPLE:
    _fn = globals().get(_name)
    if _fn is not None:
        globals()[_name] = _accepts_out(_fn, _name)
del _name


# ================================================================ install

def install(name="torch", modules=None):
    """Plant the submodule paths so that `import torch` picks up this subset.

    Writing the paths by hand drifts — and it did. The runner, the checker and
    the tests each held their own list and all three left out
    `torch.optim.lr_scheduler`, so the thing existed and
    `from torch.optim.lr_scheduler import StepLR` stopped in the body of a
    textbook. So there is no list; it is built by walking `_Namespace`.

    The root (`sys.modules["torch"]`) is planted by the caller — that side is
    what holds the module object.
    """
    import sys

    modules = sys.modules if modules is None else modules
    registered = []

    def walk(namespace, prefix):
        for key in sorted(dir(namespace)):
            if key.startswith("_"):
                continue
            value = getattr(namespace, key)
            if isinstance(value, _Namespace):
                path = prefix + "." + key
                modules[path] = value
                registered.append(path)
                walk(value, path)

    walk_root = [(key, value) for key, value in sorted(globals().items())
                 if not key.startswith("_") and isinstance(value, _Namespace)]
    for key, value in walk_root:
        path = name + "." + key
        modules[path] = value
        registered.append(path)
        walk(value, path)
    return registered
