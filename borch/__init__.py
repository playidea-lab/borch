"""borch — numpy 위에 얹은 PyTorch 모양의 얇은 층.

설치 없이 브라우저(Pyodide)에서 PyTorch **문법**을 연습하기 위한 것이다.
torch 는 wasm 으로 포팅되지 않는다 — 수백 MB의 네이티브 코드에, 손튜닝된 AVX·NEON
커널은 wasm SIMD 로 옮겨지지 않고, OpenMP 스레드는 Pyodide 가 싣지 않는 헤더를 요구한다.
그런데 **문법을 익히는 데는 그중 아무것도 필요하지 않다.** numpy 면 된다.

## 설계 원칙 — 틀린 답보다 없는 기능이 낫다

축소판이 진짜와 조금이라도 다르게 동작하면 학생은 거짓을 배운다. 그래서
**없는 것은 근사하지 않고 예외를 던진다.** 조용히 다른 값을 내느니 시끄럽게 멈춘다.

지원 범위 밖을 만나면 `BrowserTorchError` 가 나고, 메시지가 "자기 컴퓨터에서 하라"고 말한다.

## 어떻게 보장하는가

두 겹이다.

1. `borch-check` — 같은 **랩 테스트**를 진짜 torch 와 축소판 양쪽에서 돌린다.
2. `borch-diff` — 랩과 무관하게 **같은 연산의 숫자를 직접 비교**한다
   (`tests/test_borch_diff.py`). 1번만으로는 랩이 지나가는 길만 봐서
   축소판의 73% 였고, 그 사각지대에 역전파가 들어 있었다.

지금은 두 검사가 86% 를 덮는다. 남은 곳은 `__repr__` 처럼 값이 걸리지 않는 자리다.

`borch-diff` 가 실제로 잡은 것: BatchNorm 은 정규화에 편향 분산을,
running_var 갱신에는 비편향 분산을 쓴다. 둘 다 편향으로 두면 2.6% 어긋난다.
"""


import math as _math

import numpy as _np

from ._base import (
    BrowserTorchError, Size, _DEFAULT_DTYPE, _LINE_WIDTH, _NP_TO_DTYPE,
    _PRINT_PRECISION, __all__, _float_formatter, _like_torch, _resolve, _tensor_repr,
    _tensor_str, _unsupported, bool_, dtype, float32, float64, int64, long,
    set_printoptions,
)
from ._tensor import (
    Tensor, _CATEGORY, _DEFAULT_BY_CATEGORY, _DataDescriptor, _GradMode, _MinMax, _RANK,
    _category, _grad_mode, _no_bool_subtract, _promote, _scalar_category, _unbroadcast,
    result_type,
)
from ._ops import (
    Generator, _Cuda, _ERF_A, _ERF_P, _INPLACE_UNARY, _Linalg, _Lstsq, _Namespace, _SVD,
    _abs, _binary_math, _col2im, _compare, _cum_extreme, _diagonal_scatter, _erf64,
    _erfc_pos, _expand_reduced, _from_plain, _gelu, _im2col, _index_at, _index_for,
    _make_inplace, _mat, _nan_mask, _negate, _nm, _one_plus_erf64, _pad2d, _pair, _pick,
    _pool_1d_over_last, _pool_all, _rng, _running_idx, _slice_at, _spread_max,
    _to_plain, _unary, _wrap, _zero_grad, abs, absolute, acos, acosh,
    adaptive_avg_pool2d, allclose, amax, amin, aminmax, arange, arccos, arccosh, arcsin,
    arcsinh, arctan, arctanh, argsort, argwhere, as_tensor, asin, asinh, atan, atan2,
    atanh, atleast_1d, atleast_2d, atleast_3d, avg_pool2d, bincount, bmm, cat, ceil,
    cholesky, chunk, clamp, clip, conv1d, conv2d, conv3d, copysign, cos, cosh,
    cosine_similarity, count_nonzero, cuda, cummax, cummin, cumprod, cumsum, deg2rad,
    det, diag, diagflat, diagonal, diff, dist, dot, dropout, dsplit, eigh, einsum, elu,
    embedding, empty, eq, equal, erf, erfc, exp, exp2, expand, expand_as, expm1, eye,
    fix, flip, fliplr, flipud, floor, frac, from_numpy, full, full_like, gather, ge,
    gelu, gt, heaviside, hsplit, hypot, index_select, interpolate, inverse, isfinite,
    isinf, isnan, kthvalue, l1_loss, layer_norm, ldexp, le, leaky_relu, linalg,
    linspace, load, log, log10, log1p, log2, log_softmax, logaddexp, logaddexp2, logdet,
    logical_and, logical_not, logical_or, logit, logsumexp, lstsq, lt, manual_seed,
    masked_fill, masked_select, matmul, matrix_power, matrix_rank, max_pool1d,
    max_pool2d, max_pool3d, maximum, median, minimum, mm, movedim, msort, multinomial,
    nanmean, nanquantile, nansum, narrow, ne, neg, negative, nll_loss, no_grad, nonzero,
    norm, normalize, ones, ones_like, outer, pad, pinverse, positive, pow, prod, qr,
    quantile, rad2deg, rand, randint, randn, randperm, ravel, reciprocal, relu, repeat,
    repeat_interleave, reshape, roll, rot90, round, rsqrt, save, select, sgn, sigmoid,
    sign, signbit, silu, sin, sinc, sinh, slogdet, smooth_l1_loss, softmax, solve, sort,
    split, sqrt, square, stack, svd, swapaxes, swapdims, tan, tanh, tensor, tile, topk,
    trace, tril, triu, trunc, unbind, unflatten, unfold, unique, unsqueeze, vsplit,
    where, xlogy, zeros, zeros_like,
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
)
from ._optim import (
    Adadelta, Adagrad, Adam, AdamW, Adamax, ChainedScheduler, ConstantLR,
    CosineAnnealingLR, CosineAnnealingWarmRestarts, ExponentialLR, LambdaLR, LinearLR,
    MultiStepLR, MultiplicativeLR, NAdam, OneCycleLR, Optimizer, PolynomialLR, RAdam,
    RMSprop, ReduceLROnPlateau, SGD, SequentialLR, StepLR, _LRScheduler, _Optim,
    _Scheduler, optim,
)
from ._data import (
    ConcatDataset, DataLoader, Dataset, RandomSampler, SequentialSampler, Subset,
    TensorDataset, WeightedRandomSampler, _Utils, _UtilsData, random_split, utils,
)
from ._rnn import (
    _NnUtils, _NnUtilsRnn, pad_sequence,
)

# ================================================================ 메서드 노출
#
# torch 코드는 `torch.sin(x)` 와 `x.sin()` 을 섞어 쓴다. 우리는 모듈 함수만 갖고 있어서
# 점 표기를 쓴 튜토리얼이 `AttributeError` 로 멈췄다 — **있는 기능인데 부르는 법이 하나
# 모자란** 자리였다.
#
# 이 목록은 손으로 고르지 않았다. torch 에게 `x.f(...)` 와 `torch.f(x, ...)` 가 같은
# 값을 내는지 물어보고, 같다고 답한 것만 담았다. 62개가 같다고 나왔고 하나가 달랐다 —
# `where` 다(아래 참고). 그런 것을 그냥 붙이면 조용히 틀린 답이 나온다.

_AS_METHOD = (
    "allclose", "argsort", "bmm", "ceil", "chunk", "clamp", "cos", "cosh", "cumprod",
    "cumsum", "diag", "dot", "eq", "equal", "erf", "flip", "floor", "gather", "ge",
    "gt", "isfinite", "isinf", "isnan", "le", "log10", "log2", "lt", "maximum",
    "median", "minimum", "mm", "movedim", "multinomial", "narrow", "ne", "neg",
    "norm", "outer", "pow", "prod", "reciprocal", "relu", "roll", "round", "rsqrt",
    "sigmoid", "sign", "sin", "sinh", "softmax", "sort", "split", "square", "tan",
    "tanh", "tile", "topk", "trace", "tril", "triu", "unbind", "unique",
    # 수학 함수 묶음. 같은 방법으로 확인했다 — torch 에게 물어보고 같다고 한 것만.
    "acos", "acosh", "arccos", "arccosh", "arcsin", "arcsinh", "arctan", "arctanh",
    "asin", "asinh", "atan", "atan2", "atanh", "absolute", "clip", "copysign",
    "deg2rad", "erfc", "exp2", "expm1", "fix", "frac", "heaviside", "hypot", "ldexp",
    "log1p", "logaddexp", "logaddexp2", "logit", "negative", "positive", "rad2deg",
    "sgn", "signbit", "sinc", "trunc", "xlogy",
    # 축약 묶음. torch 도 이 열여섯을 메서드로 노출한다.
    "amax", "amin", "aminmax", "argwhere", "cummax", "cummin", "diff", "dist",
    "kthvalue", "logsumexp", "msort", "nanmean", "nanquantile", "nansum", "nonzero",
    "quantile",
    # 모양 묶음. 이 중 `expand`·`repeat`·`ravel`·`select`·`unfold`·`expand_as` 는
    # **torch 에 모듈 함수가 없고 메서드로만 있다** — 부르는 법이 하나뿐인 자리다.
    "diagflat", "diagonal", "dsplit", "expand", "expand_as", "fliplr", "flipud",
    "hsplit", "ravel", "repeat", "rot90", "select", "swapaxes", "swapdims",
    "unflatten", "unfold", "vsplit",
    # 자매에는 메서드로 있고 여기에는 함수로만 있던 셋. torch 도 메서드로 준다.
    "index_select", "masked_select", "repeat_interleave", "masked_fill",
)

for _method in _AS_METHOD:
    if not hasattr(Tensor, _method):
        setattr(Tensor, _method, globals()[_method])


def _where_method(self, condition, other):
    """**인자 순서가 함수와 다르다.** `x.where(조건, y)` 는 `torch.where(조건, x, y)` 다.

    이것만 그냥 붙이면 `x` 가 조건 자리로 들어가 조용히 틀린 답이 나온다.
    torch 에 물어봐서 알았지, 목록을 눈으로 훑어서는 안 나왔을 자리다.
    """
    return where(condition, self, other)


Tensor.where = _where_method


# ================================================================ install

def install(name="torch", modules=None):
    """`import torch` 가 이 축소판을 집도록 하위 모듈 경로를 심는다.

    경로를 손으로 적으면 어긋난다 — 실제로 어긋났다. 러너·검사기·테스트가 각자
    목록을 들고 있었고 셋 다 `torch.optim.lr_scheduler` 를 빠뜨려서, 물건은 있는데
    `from torch.optim.lr_scheduler import StepLR` 이 교재 본문에서 멈췄다.
    그래서 목록을 두지 않고 `_Namespace` 를 훑어 만든다.

    뿌리(`sys.modules["torch"]`)는 부르는 쪽이 심는다 — 모듈 객체를 쥔 것은 그쪽이다.
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
