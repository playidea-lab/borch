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


import builtins as _builtins
import inspect as _inspect
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
    result_type,
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


def _softmax_method(self, dim=None, dtype=None):
    """**The method is not the function.** `F.softmax` carries torch's private
    `_stacklevel` third, and binding the function straight on as a method put that
    into `Tensor.softmax` too — where torch's method has `(dim, dtype)` and nothing
    between them. `x.softmax(1, torch.float32)` would then have set a stack level.

    The seat exists on the function because a positional call reaches it there; it
    does not exist on the method because torch's method has no such seat. **Sharing
    an implementation is not the same as sharing a signature**, and the loop above
    cannot tell the two apart — it binds whatever the module has.
    """
    return softmax(self, dim=dim, dtype=dtype)


def _log_softmax_method(self, dim=None, dtype=None):
    """See `_softmax_method`."""
    return log_softmax(self, dim=dim, dtype=dtype)


def _split_method(self, split_size, dim=0):
    """**torch's function and torch's method disagree, and refuse each other's
    keyword.** `torch.split(t, split_size_or_sections=2)` is taken and
    `torch.split(t, split_size=2)` is not; `t.split(split_size=2)` is taken and
    `t.split(split_size_or_sections=2)` is not. Both measured.

    So the method gets its own name for the same argument. This is the third pair in
    a day where binding a module function straight on as a method carried the
    function's signature into a place torch spells differently — `softmax` and
    `log_softmax` above are the others.
    """
    return split(self, split_size, dim)


Tensor.softmax = _softmax_method
Tensor.log_softmax = _log_softmax_method
Tensor.split = _split_method


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


# ── the seven losses that are **not** the same function at top level ────────
#
# These were declined, and the reason read *the top-level one is the raw ATen op — its
# signature differs from F's.* Every word true, and none of it about what is missing:
# `F.kl_div` is here, so what was absent is a name and a set of defaults.
#
# **They really are different functions**, which is why they are not in the loop above:
#
# - the reduction is an **integer** — `0` none, `1` mean, `2` sum — where `F` takes the
#   word. Passing `"mean"` to the ATen op is a `TypeError` in torch, so accepting the
#   word here would be a wider door than the one being copied.
# - it **defaults to none**, where every `F` loss defaults to mean. A caller who reads
#   `torch.kl_div(a, b)` as `F.kl_div(a, b)` gets a table where they expected a number,
#   which is loud — and a caller who sums it afterwards gets a different number
#   quietly, since `mean` divides.
# - `poisson_nll_loss` has **no defaults at all**: all six arguments are required.
#   The others have theirs, and the two facts are measured rather than assumed.
#
# **The declared schema disagrees with the binding**, which is the part worth writing
# down: `aten::kl_div(..., int reduction=1)` says mean, and `torch.kl_div(a, b)` returns
# a table. The behaviour is the authority here, so `0` is what these take.
_REDUCTIONS = ("none", "mean", "sum")


def _aten_reduction(value):
    """The integer the ATen ops take, as the word `F` takes.

    **Not a lookup with a default** — an out-of-range integer is an error rather than
    the nearest legal one, because the three values are an enum and a fourth means the
    caller believes something untrue about it.
    """
    # **`int` in this module is torch's dtype**, not the builtin — the alias loop above
    # binds `int`, `float` and `bool` as dtypes because torch has them under those
    # names. So `isinstance(value, int)` here asks whether a number is a dtype and
    # raises `TypeError: isinstance() arg 2 must be a type`. Measured, not guessed: it
    # is the first thing this function did.
    if (not isinstance(value, _builtins.int) or isinstance(value, _builtins.bool)
            or not 0 <= value <= 2):
        raise ValueError(_like_torch(
            f"reduction has to be 0, 1 or 2, but got {value!r}.",
            "reduction is expected to be an int in [0, 2]"))
    return _REDUCTIONS[value]


def binary_cross_entropy_with_logits(self, target, weight=None, pos_weight=None,
                                     reduction=0):
    """`F.binary_cross_entropy_with_logits` with ATen's argument order and defaults."""
    return nn.functional.binary_cross_entropy_with_logits(
        self, target, weight=weight, pos_weight=pos_weight,
        reduction=_aten_reduction(reduction))


def cosine_embedding_loss(input1, input2, target, margin=0.0, reduction=0):
    """As above. **`margin` defaults to 0 here and in `F`** — the two agree on that one
    and disagree on the reduction, which is why neither can be assumed from the other."""
    return nn.functional.cosine_embedding_loss(
        input1, input2, target, margin=margin, reduction=_aten_reduction(reduction))


def hinge_embedding_loss(self, target, margin=1.0, reduction=0):
    return nn.functional.hinge_embedding_loss(
        self, target, margin=margin, reduction=_aten_reduction(reduction))


def kl_div(self, target, reduction=0, *, log_target=False):
    """**`log_target` is keyword-only**, as in the schema. `F.kl_div` takes it
    positionally after two deprecated arguments, so a caller moving between the two has
    to name it either way."""
    return nn.functional.kl_div(self, target, reduction=_aten_reduction(reduction),
                                log_target=log_target)


def margin_ranking_loss(input1, input2, target, margin=0.0, reduction=0):
    return nn.functional.margin_ranking_loss(
        input1, input2, target, margin=margin, reduction=_aten_reduction(reduction))


def poisson_nll_loss(input, target, log_input, full, eps, reduction):
    """**Six required arguments and no defaults.** `F.poisson_nll_loss` gives all four
    of the trailing ones a value; this gives none, which is the schema and is what makes
    it the odd one of the seven — a caller cannot reach it with two arguments at all."""
    return nn.functional.poisson_nll_loss(
        input, target, log_input=log_input, full=full, eps=eps,
        reduction=_aten_reduction(reduction))


def triplet_margin_loss(anchor, positive, negative, margin=1.0, p=2.0, eps=1e-6,
                        swap=False, reduction=0):
    return nn.functional.triplet_margin_loss(
        anchor, positive, negative, margin=margin, p=p, eps=eps, swap=swap,
        reduction=_aten_reduction(reduction))


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


# ── the methods that forward to a module function ──────────────────────────
#
# `_tensor.py` binds a few dozen names as `def method(self, *args, **kw)` that call
# `_ops.<name>`, because writing each list out twice is how the two drift. The cost is
# that **`inspect.signature` sees the wrapper**, so every check that reads a signature
# goes blind at those names: fifteen of the `variadic` rows on the core-to-torch axis
# were this, and `variadic` means *cannot be compared at all.*
#
# `__wrapped__` is what `inspect` follows, and it is set here rather than in
# `_tensor.py` because the module function does not exist yet when the binding runs —
# `_ops` imports `_tensor`, not the other way round.
#
# The same omission in `_accepts_out` below cost two names on the `dim` sweep, and
# widening it there turned up two silent wrong answers. This is the same repair on a
# larger set.
def _link_wrapped():
    from . import _ops as _o
    from ._tensor import Tensor as _T

    for _n in dir(_T):
        _m = getattr(_T, _n, None)
        if getattr(_m, "__wrapped__", None) is not None:
            continue
        if getattr(_m, "_forwards_nothing", False):
            # A tombstone — the name exists because torch keeps it and raises.
            # Linking it to the live `_ops` function of the same name lends it an
            # argument list it does not have; see `_deprecated_by_torch`.
            continue
        _target = getattr(_o, _n, None)
        # **A name can be the same object on both sides, and then this pointed it at
        # itself.** `_ops.eq` and `Tensor.eq` are one `_compare.<locals>.cmp`, so
        # `_m.__wrapped__ = _target` made `__wrapped__` a self-loop — and
        # `inspect.signature` does not shrug at that, it **raises**
        # `ValueError: wrapper loop when unwrapping`. Eleven `Tensor` names were in
        # that state: the eight comparisons and the three aliases pointing at them.
        #
        # Worse than the bag it was written to remove. A bag reads as
        # `(self, *args, **kw)` and says little; a loop makes `inspect.signature`,
        # `help()` and every editor's hover **fail** on `x.eq`, which torch answers.
        # The axis filed all eleven as *no python signature*, one wording for two very
        # different conditions.
        if _target is _m:
            continue
        if callable(_m) and callable(_target) and "<locals>" in getattr(
                _m, "__qualname__", "") and getattr(_m, "__name__", None) == _n:
            try:
                _m.__wrapped__ = _target
            except (AttributeError, TypeError):
                pass


_link_wrapped()
del _link_wrapped


# **The two generators in `_tensor.py` build `(self, *args, **kw)`, and a bag says
# nothing.** `_bind_inplace` wraps the partner method and `_bind_from_module` wraps
# an `_ops` function; both hand the arguments straight on, so what each will accept
# is already written down one call away. Forty-seven `Tensor` rows were filed as
# *no python signature* — the bucket meaning **cannot be judged**, which is an
# absorbing state: no count holds it and nothing downstream of it gets asked.
#
# The same repair `_ops._make_inplace` took, on the two generators that live on the
# other side of the import. **It runs here for the ordering reason above** —
# `_bind_from_module` reads `_ops`, which does not exist while `_tensor` is being
# built, and `_bind_inplace`'s partners are only fully readable once `_link_wrapped`
# has finished.
#
# **A variadic source is left alone.** `expand(*sizes)` and `reshape(*shape)` really
# do take any number, and copying a promise the method cannot keep is worse than the
# bag — that is the mistake `relu_` caught in `_ops.py`, where a declared signature
# outran what the method would accept.
def _declare_forwarders():
    import inspect as _inspect

    from . import _ops as _o
    from ._tensor import _INPLACE_FROM_PAIR, _INPLACE_LATE, _METHOD_FROM_MODULE
    from ._tensor import Tensor as _T

    def _copy(method, source, drop_receiver):
        if source is None or getattr(method, "__signature__", None) is not None:
            return
        try:
            got = _inspect.signature(source)
        except (TypeError, ValueError):
            return
        params = list(got.parameters.values())
        # **Not `any(...)`.** This module exports a tensor reduction called `any`, and
        # inside a function defined here that global wins over the builtin — the
        # generator expression went to `borch.any`, which tried to make a tensor out of
        # it. A plain loop has no such name to collide with.
        for one in params:
            if one.kind in (one.VAR_POSITIONAL, one.VAR_KEYWORD):
                return
        if drop_receiver:
            if not params:
                return
            params = params[1:]
        first = _inspect.Parameter("self", _inspect.Parameter.POSITIONAL_OR_KEYWORD)
        method.__signature__ = got.replace(parameters=[first] + params)

    # **The module-borrowed methods go first.** An in-place name reads its partner off
    # the class, and three of them — `arctan2_`, `igamma_`, `igammac_` — have a partner
    # that is itself one of these forwarders. Repaired in the other order they copied a
    # bag and stayed bags, with nothing to say which pass had been the wrong way round.
    for _n in _METHOD_FROM_MODULE:
        _copy(getattr(_T, _n, None), getattr(_o, _n, None), drop_receiver=True)
    for _n in _INPLACE_FROM_PAIR + _INPLACE_LATE:
        _copy(getattr(_T, _n, None), getattr(_T, _n[:-1], None), drop_receiver=True)


_declare_forwarders()
del _declare_forwarders


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
#
# **47 names arrived at once, and their absence was the check's own doing.** That
# file rejected the docstrings for deciding *whether* a name takes `out=` — and then
# enumerated its candidates from `"out=None" in fn.__doc__`, which is the same source
# doing the other half of the job. `abs`, `acos`, `asin`, `atan`, `log2`, `log10`,
# `square`, `norm`, `nansum`, `msort`, `diff` and the rest of the inverse-trigonometric
# family take `out=` in torch and are documented with a bare `out`, so they were never
# asked about. The check passed on exactly the set the table already held.
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
    abs absolute acos acosh angle arccos arccosh arcsin arcsinh
    arctan arctan2 arctanh argmax argmin asin asinh atan atan2
    atanh bernoulli bitwise_or chain_matmul clamp_max clamp_min diff hardshrink hash_tensor
    isin log10 log2 log_softmax logical_xor msort nansum norm orgqr
    rad2deg slice_scatter softmax square take tensordot threshold true_divide
    
""".split())

# The ones taking **several**, as in `out=(values, indices)`. The same rule with
# more than one slot.
_TAKES_OUT_TUPLE = frozenset("""
    aminmax cummax cummin frexp geqrf histogram kthvalue mode sort svd topk
    triangular_solve
    lu qr slogdet
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
    # **The wrapped signature has to survive**, or every check that reads one goes
    # blind at these names. `inspect.signature` follows `__wrapped__`, so setting it
    # is what makes `narrow(t, dim, start, length)` still read as that rather than as
    # `(*args, **kwargs)`.
    #
    # It was missing, and adding 50 names to the table above is what surfaced it:
    # `slice_scatter` and `take` dropped out of `tests/test_axis_sweep.py` the moment
    # they were wrapped, and the two guards there — *a call entry naming a function
    # nobody sweeps* and *a function taking an index that is in neither table* — both
    # fired. Without those two the axes would have quietly stopped asking about
    # whatever the table grew to cover, which is the one direction a coverage check
    # cannot report on itself.
    call.__wrapped__ = fn
    # **And `__wrapped__` alone under-reports: it says the list without `out`.**
    # That is what the wrapped function takes, not what this one takes, and the
    # difference is exactly the argument this wrapper exists to add. Four rows on
    # the core-against-torch axis sat frozen under *"torch declares an `out=` and
    # this does not"* — read three times and true each time, because the sentence
    # is about the **declaration** and nobody re-asked once `out=` began working.
    #
    # `linalg` is the only namespace that axis compares where the gap shows; the
    # same understatement covers all 229 top-level names in the table above and is
    # simply not looked at there. So it is repaired where it lives.
    #
    # (The docstring says a hundred and seventy-two, which was the count the day it
    # was written. Left alone — a past number changed to the current one stops being
    # a record of anything.)
    #
    # `__signature__` wins over `__wrapped__` in `inspect`, and it is **built from**
    # the wrapped list rather than replacing it — `narrow(input, dim, start, length,
    # *, out=None)`, the whole thing, not `(*args, **kwargs)`.
    try:
        _sig = _inspect.signature(fn)
    except (TypeError, ValueError):
        return call
    # **No builtins here.** This module binds `any`, `all`, `type`, `range`, `sum`,
    # `min`, `max` and `abs` as torch functions, so inside it those names are the
    # tensor operations. `any(p.name == "out" for p in _params)` reached
    # `borch.any`, which tried to make a tensor out of a generator — a `TypeError`
    # about `float()` that names neither `any` nor this function. `type` had cost
    # the same half hour one screen down, and the loop below is what both fixes
    # look like.
    _params = list(_sig.parameters.values())
    _at = len(_params)
    for _i, _param in enumerate(_params):
        if _param.name == "out":
            return call
        if _param.kind is _inspect.Parameter.VAR_KEYWORD and _at == len(_params):
            _at = _i
    _params.insert(_at, _inspect.Parameter(
        "out", _inspect.Parameter.KEYWORD_ONLY, default=None))
    call.__signature__ = _sig.replace(parameters=_params)
    return call


for _name in _TAKES_OUT | _TAKES_OUT_TUPLE:
    _fn = globals().get(_name)
    if _fn is not None:
        globals()[_name] = _accepts_out(_fn, _name)
del _name


# **The same wrapping, reaching `linalg`, which it did not.**
#
# The loop above walks this module's `globals()`. `linalg` is a namespace object whose
# members are bound from `_ops` directly, so none of them passed through it —
# `borch.qr(x, out=…)` was taken and `borch.linalg.qr(x, out=…)` was a `TypeError`,
# for the same function.
#
# **It was described and never decided.** Four frozen rows on the core↔torch axis
# ended with *"torch declares an `out=` and this does not"*, which is a true sentence
# that says nothing about whether the absence was chosen. Read three times and passed
# over three times, because a description has no retirement condition — it cannot go
# stale, so it never asks to be re-read. The peer session holding borch.ts named that
# shape today after `optim`'s seven sat under one for a day.
#
# Followed through, the sentence was **half true**: the machinery exists and simply
# did not reach here. What it buys is torch's *observable* `out=` — the destination
# changes and the same object comes back — and not the allocation it exists to save,
# which `_out`'s docstring has said all along.
#
# **The list is written down and not read off torch.** The first version walked
# `torch.linalg.__doc__` at import time, which makes the library's own surface depend
# on whether real torch happens to be installed: `out=` accepted on a development
# machine and refused in the browser, where this module *is* torch. Two libraries
# under one name, decided by the environment — the exact silent divergence this
# repository exists to hunt, and it was one edit from being shipped.
#
# So it is a table, and `tests/test_out_names.py` holds it against torch. That is the
# same shape as `_TAKES_OUT` above and the reason it is a table too.
#
# **The first draft of this list came from the docstrings**, and the file that now
# holds it opens by rejecting the docstrings for deciding exactly this — with the
# receipt: twenty-four names lost that way once already. Re-measured by calling, the
# thirty-seven were right and `lstsq` was missing, which is the direction that costs
# a reader: torch takes `out=` there and this refused it.
_LINALG_TAKES_OUT = frozenset("""
    cholesky cholesky_ex cond cross det eig eigh eigvals eigvalsh
    householder_product inv inv_ex ldl_factor ldl_factor_ex ldl_solve lstsq lu
    lu_factor lu_factor_ex lu_solve matmul matrix_norm matrix_power matrix_rank
    multi_dot norm pinv qr slogdet solve solve_ex solve_triangular svd svdvals
    tensorinv tensorsolve vecdot vector_norm
""".split())

for _name in _LINALG_TAKES_OUT:
    # **No `isinstance(_fn, type)` guard here, and the first version had one.**
    # `type` is not the builtin in this module — `torch.type` is a real name and this
    # package exports it — so the guard raised `arg 2 must be a type`. Every entry
    # above is a function and none is a class, so the guard was doing nothing but
    # being wrong. A name meaning something else, in the file that spent the day on
    # exactly that.
    _fn = getattr(linalg, _name, None)
    if _fn is not None and callable(_fn):
        setattr(linalg, _name, _accepts_out(_fn, f"linalg.{_name}"))
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


# ======================================== closing the doors torch does not have
#
# Everything above needs these names **in scope**: the layers to build `nn`, the
# functions to be bound onto `Tensor`, the optimisers to build `optim`. Importing
# them by name is how they get there, and binding them at the top level is what
# that leaves behind. Nobody chose it.
#
# The cost is not a wrong value, it is a **wrong lesson.** This library exists so
# that torch's syntax can be practised with nothing installed, and
#
#     model = torch.Linear(3, 4)
#
# runs here and stops on real torch with `AttributeError`. The stop arrives after
# the student has left, which is the one place we cannot say anything from.
#
# The standard was already written down two hundred lines above: seven losses are
# deliberately not re-exported, because torch's top-level versions of them take an
# integer reduction and default to `none`, so the friendly alias would diverge
# starting at the shape. This applies the same reasoning to the rest of them.
#
# None of these was invented — every one has a home in torch, just not this one.
# `tests/test_gap.py` holds the list closed from now on.
for _name in (
    # ── torch keeps these 169 on `Tensor`. The module-level function is what gets
    #    bound on as the method, so the name has to exist here first.
    "absolute_", "add_", "addbmm_", "addcdiv_", "addcmul_", "addmm_", "addr_", "apply_",
    "arctan2_", "as_subclass", "atan2_", "backward", "baddbmm_", "bernoulli_",
    "bitwise_and_", "bitwise_left_shift_", "bitwise_not_", "bitwise_or_",
    "bitwise_right_shift_", "bitwise_xor_", "byte", "cauchy_", "ccol_indices", "char",
    "coalesce", "col_indices", "const_data_ptr", "contiguous", "copy_", "copysign_",
    "crow_indices", "cumprod_", "cumsum_", "data_ptr", "dense_dim", "digamma_", "dim",
    "dim_order", "div_", "divide_", "element_size", "eq_", "erfinv_", "expand",
    "expand_as", "exponential_", "fill_diagonal_", "float_power_", "floor_divide_",
    "fmod_", "ge_", "geometric_", "greater_", "greater_equal_", "gt_", "heaviside_",
    "hypot_", "igamma_", "igammac_", "index", "index_add_", "index_copy_",
    "index_fill_", "index_reduce_", "indices", "ipu", "is_coalesced", "is_contiguous",
    "is_pinned", "is_set_to", "is_shared", "item", "le_", "lerp_", "less_",
    "less_equal_", "lgamma_", "log_normal_", "logical_and_", "logical_not_",
    "logical_or_", "logical_xor_", "lt_", "map2_", "map_", "masked_fill_",
    "masked_scatter_", "module_load", "mul_", "multiply_", "mvlgamma_", "ndimension",
    "ne_", "nelement", "new", "new_empty", "new_empty_strided", "new_full", "new_ones",
    "new_tensor", "new_zeros", "nextafter_", "normal_", "not_equal_", "numpy",
    "pin_memory", "polygamma_", "pow_", "put_", "random_", "record_stream",
    "register_hook", "register_post_accumulate_grad_hook", "reinforce", "remainder_",
    "renorm_", "repeat", "requires_grad_", "reshape_as", "resize", "resize_",
    "resize_as", "retain_grad", "row_indices", "scatter_", "scatter_add_",
    "scatter_reduce_", "set_", "sgn_", "share_memory_", "sign_", "size", "sparse_dim",
    "sparse_mask", "sparse_resize_", "sparse_resize_and_clear_", "squeeze_",
    "storage_offset", "storage_type", "stride", "sub_", "subtract_", "sum_to_size",
    "swapaxes_", "swapdims_", "t_", "to", "to_dense", "to_mkldnn", "to_padded_tensor",
    "to_sparse", "to_sparse_bsc", "to_sparse_bsr", "to_sparse_coo", "to_sparse_csc",
    "to_sparse_csr", "tolist", "transpose_", "tril_", "triu_", "true_divide_", "type",
    "type_as", "uniform_", "unsqueeze_", "untyped_storage", "values", "view", "view_as",
    # ── torch keeps these 49 in `nn`.
    "AdaptiveAvgPool2d", "AvgPool2d", "BCELoss", "BCEWithLogitsLoss", "BatchNorm1d",
    "BatchNorm2d", "BatchNorm3d", "Conv1d", "Conv2d", "Conv3d", "CrossEntropyLoss",
    "Dropout", "ELU", "Embedding", "Flatten", "GELU", "GRU", "Identity", "L1Loss",
    "LSTM", "LayerNorm", "LeakyReLU", "Linear", "LogSoftmax", "MSELoss", "MaxPool1d",
    "MaxPool2d", "MaxPool3d", "Module", "ModuleList", "MultiheadAttention", "NLLLoss",
    "Parameter", "ParameterList", "RNN", "ReLU", "Sequential", "SiLU", "Sigmoid",
    "SmoothL1Loss", "Softmax", "Tanh", "Transformer", "TransformerDecoder",
    "TransformerDecoderLayer", "TransformerEncoder", "TransformerEncoderLayer",
    "Unflatten", "Upsample",
    # ── torch keeps these 14 in `optim`.
    "ASGD", "Adadelta", "Adafactor", "Adagrad", "Adam", "AdamW", "Adamax", "LBFGS",
    "NAdam", "Optimizer", "RAdam", "RMSprop", "Rprop", "SGD",
    # ── and these 15 in `optim.lr_scheduler`.
    "ChainedScheduler", "ConstantLR", "CosineAnnealingLR",
    "CosineAnnealingWarmRestarts", "CyclicLR", "ExponentialLR", "LambdaLR", "LinearLR",
    "MultiStepLR", "MultiplicativeLR", "OneCycleLR", "PolynomialLR",
    "ReduceLROnPlateau", "SequentialLR", "StepLR",
    # ── these 16 in `utils.data`.
    "BatchSampler", "ChainDataset", "ConcatDataset", "DataLoader", "Dataset",
    "IterableDataset", "RandomSampler", "Sampler", "SequentialSampler", "StackDataset",
    "Subset", "SubsetRandomSampler", "TensorDataset", "WeightedRandomSampler",
    "default_collate", "random_split",
    # ── these 14 in `nn.functional`. `F.pad` is the door; `torch.pad` is not one.
    "adaptive_avg_pool2d", "avg_pool2d", "elu", "gelu", "interpolate", "l1_loss",
    "leaky_relu", "nll_loss", "normalize", "one_hot", "pad", "silu", "smooth_l1_loss",
    "unfold",
    # ── and one apiece in `linalg` and `nn.utils.rnn`.
    "eigh", "pad_sequence",
):
    # **Written out rather than measured.** Working the list out at import time would
    # mean importing torch to ask, and this package exists for machines that have not
    # got it. The list being static is what makes the test above necessary.
    del globals()[_name]
del _name
