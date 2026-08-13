"""browsertorch-webgpu — TF.js WebGPU 위에 얹은 browsertorch 모양의 층.

코어 `browsertorch` 를 **대체하지 않는다.** 코어는 numpy 위에서 MNIST 급까지 가고,
이쪽은 GPU 위에서 그 위를 간다. 왜 별도인지는 ROADMAP.md 의 ADR-001 에 있다.

## 브라우저 안에서만 돈다

`js.tf` 를 부른다. 네이티브 CPython 에서 임포트하면 바로 멈춘다 — 조용히 다른 것으로
폴백하면 "GPU 로 돌렸다"고 착각하게 되고, 그건 이 프로젝트가 가장 싫어하는 종류다.

## 왜 자체 autograd 인가

TF.js 의 `tf.grad` 를 쓰지 않는다. 재봤더니 conv 역방향 커널이 순방향의 1/26 이었고
(WEBGPU-DESIGN.md 1.5절), 역방향을 **순방향 conv 로 다시 써야** 목표에 닿는다.
그러려면 역방향을 우리가 들고 있어야 한다. 테이프 구조는 코어와 같게 둔다 —
같은 모양이면 코어에서 고친 것을 여기로 옮기기 쉽다.

## 한계 — 코어와 다른 점

- **dtype 은 `float32` · `int64` · `bool` 세 가지다.** 승격 규칙은 torch 와 같게
  맞췄지만(72/72), **`float64` 는 없다** — TF.js 에 배정도가 없어서 거절한다.
  그리고 정수는 float32 에 담으므로 **2^24 까지 정확**하고, 넘으면 조용히 자르지 않고
  던진다. 코어는 float64 까지 112/112 다
- **뷰가 저장소를 공유하지 않는다.** torch 는 `b = a.view(2,2); b[0,0] = 9` 로 `a` 가
  바뀌는데, TF.js 텐서는 불변이라 그럴 수 없다. 코어는 이것을 13/13 로 맞춘다.
  조용히 다르게 두지 않고, **뷰에 대입하면 거절한다**
- **랭크 6 까지다.** 랭크 7 부터 TF.js 가 커널을 안 갖고 있고, 그것도 **일부만**이다 —
  원소별·`permute`·`reshape`·0 으로 두르는 `pad` 는 랭크 8 에서도 돌지만, **축을 따라
  줄이는 것**(`sum(dim=)`)과 `fill`, 그리고 대부분의 기울기가 없다. 없는 자리는
  `GPU for rank 7 is not yet supported` 를 던진다. 재본 범위에서 **틀린 값을 낸 적은
  없다** — 되거나 던지거나 둘 중 하나였고, 그 경계를 골든이 붙잡고 있다
- **`scope()` 를 노출한다.** 역전파 클로저가 든 중간 버퍼는 파이썬 GC 가 못 놓는다.
  코어와 다른 한 줄이고, 이유는 WEBGPU-DESIGN.md 7절에 있다

없는 것은 근사하지 않고 예외를 던진다. `topk(largest=False)` 처럼 TF.js 가 못 하는
인자도 조용히 무시하지 않고 거절한다 — 조용히 다른 값을 내는 것이 가장 나쁘다.

## 이 파일을 고칠 때의 규칙 둘

**랭크 5 이상에서 `_tf.pad` 를 직접 부르지 마라. `_pad_const` 를 거쳐라.**
거기서 `tf.pad` 는 모양을 맞게 돌려주고 값을 깨뜨리며 **예외를 안 던진다.**

이 하나를 잡는 데 세 번 걸렸고, 세 번 다 "고쳤다"고 생각한 뒤에 다음이 나왔다.
conv3d 에서 처음 보고 거기만 고쳤더니, 케이스를 세워 물어보니 자르기의 역방향이
같은 함수를 불러 `narrow`·`unbind`·`split` 셋이 랭크 5 에서 틀린 기울기를 내고
있었다. 그것도 고치고 랭크 6 을 물어보니 이번에는 `F.pad` 자신 — 사용자가 직접
부르는 문 — 이 랭크 5·6 양쪽에서 틀리고 있었다. **눈으로 훑어서는 세 번 다 못 봤고,
케이스를 세워 물어봐서 세 번 다 나왔다.**

랭크 6 자체는 멀쩡하다. 원소별·축 합·permute·reshape·기울기 전부 맞는다. 고장난
것은 랭크가 아니라 `pad` 이므로, 고랭크를 만났다고 거절할 이유는 없다.

**`_wrap(x)._h` 를 인라인으로 쓰지 마라.** `x` 가 텐서가 아니면 `_wrap` 이 임시를
만드는데, `._h` 를 꺼내는 순간 그 임시의 참조가 0 이 되어 `__del__` 이 버퍼를 놓는다.
그 다음 줄이 이미 죽은 손잡이를 넘긴다.

    나쁨:  _tf.add(a, _wrap(b)._h)
    좋음:  bt = _wrap(b); _tf.add(a, bt._h)

이 함정에 **세 번** 걸렸다(`_cmp`, `__setitem__`, 그리고 디버깅용 프로브). 증상은
`TypeError: Cannot read properties of undefined (reading 'backend')` 다.
"""


import numpy as _np

try:
    import js as _js
    from pyodide.ffi import create_proxy as _create_proxy
    from pyodide.ffi import to_js as _to_js
except ImportError as _exc:                                          # pragma: no cover
    raise ImportError(
        "browsertorch_webgpu 는 브라우저(Pyodide) 안에서만 돕니다. "
        "네이티브에서는 `browsertorch` 를 쓰세요 — 이쪽을 CPU 로 흉내 내면 "
        "GPU 로 돌렸다고 착각하게 됩니다."
    ) from _exc

_tf = getattr(_js, "tf", None)
if _tf is None:                                                      # pragma: no cover
    raise ImportError("TF.js 가 페이지에 없습니다. tf.min.js 를 먼저 실으세요.")

from ._base import (
    BrowserTorchError, Size, _BY_CATEGORY, _INT_EXACT, _LINE_WIDTH, _PAD_SAFE_RANK,
    _PRINT_PRECISION, _ValuesIndices, _broadcast_error, _dtype_of, _float_formatter,
    _keep, _last_axis_only, _like_torch, _pad_const, _pick_last, _reject_float64,
    _shape_of, _slice_along, _slice_tensor, _tensor_repr, _tensor_str, _tf, _to_np,
    _to_tf, _unsupported, bool_, dtype, float32, int64, long, set_printoptions,
)
from ._tensor import (
    Tensor, _GradMode, _NCHW_TO_NHWC, _NHWC_TO_NCHW, _align, _both_bool, _canonical,
    _grad_mode, _no_bool_subtract, _relayout, _reshape_for_broadcast, _result_dtype,
    _scalar_dtype, _storage_bool, _storage_for, _tf, _unbroadcast, _wrap, result_type,
)
from ._ops import (
    Generator, _DEG, _LN10, _LN2, _binary_math, _compare, _logaddexp_h, _masked, _rng,
    _tf, _to_int32, _trunc, _unary, _zeros_like, abs, absolute, acos, acosh, allclose,
    arange, arccos, arccosh, arcsin, arcsinh, arctan, arctanh, argsort, as_tensor, asin,
    asinh, atan, atan2, atanh, bincount, bmm, cat, ceil, chunk, clamp, clip, copysign,
    cos, cosh, count_nonzero, cumprod, cumsum, deg2rad, diag, dot, einsum, empty, eq,
    equal, erf, erfc, exp, exp2, expm1, eye, fix, flip, floor, frac, from_numpy, full,
    full_like, gather, ge, gt, heaviside, hypot, index_select, isfinite, isinf, isnan,
    ldexp, le, linspace, log, log10, log1p, log2, logaddexp, logaddexp2, logical_and,
    logical_not, logical_or, logit, lt, manual_seed, masked_fill, masked_select, matmul,
    maximum, median, minimum, mm, movedim, multinomial, narrow, ne, neg, negative, norm,
    ones, ones_like, outer, positive, pow, prod, rad2deg, rand, randint, randn,
    randperm, reciprocal, relu, repeat_interleave, reshape, roll, round, rsqrt, sgn,
    sigmoid, sign, signbit, sin, sinc, sinh, sort, split, sqrt, square, stack, tan,
    tanh, tensor, tile, topk, trace, tril, triu, trunc, unbind, unique, where, xlogy,
    zeros, zeros_like,
)
from ._functional import (
    _Functional, _SQRT2, _SQRT2PI, _dilate, _pair, _tf, _to_nchw, _to_nhwc, _warn_once,
    _warned, adaptive_avg_pool2d, avg_pool2d, batch_norm, binary_cross_entropy,
    binary_cross_entropy_with_logits, conv1d, conv2d, conv3d, cosine_similarity,
    cross_entropy, dropout, elu, embedding, gelu, interpolate, l1_loss, layer_norm,
    leaky_relu, linear, log_softmax, max_pool1d, max_pool2d, max_pool3d, mse_loss,
    nll_loss, normalize, one_hot, pad, silu, smooth_l1_loss, softmax, unsqueeze,
)
from ._nn import (
    AdaptiveAvgPool2d, AvgPool2d, BCELoss, BCEWithLogitsLoss, BatchNorm1d, BatchNorm2d,
    BatchNorm3d, Conv1d, Conv2d, Conv3d, CrossEntropyLoss, Dropout, ELU, Embedding,
    Flatten, GELU, GRU, Identity, L1Loss, LSTM, LayerNorm, LeakyReLU, Linear,
    LogSoftmax, MSELoss, MaxPool1d, MaxPool2d, MaxPool3d, Module, ModuleList,
    MultiheadAttention, NLLLoss, Parameter, RNN, ReLU, Sequential, SiLU, Sigmoid,
    SmoothL1Loss, Softmax, Tanh, Transformer, TransformerDecoder,
    TransformerDecoderLayer, TransformerEncoder, TransformerEncoderLayer, Unflatten,
    Upsample, _Activation, _NN, _NnUtils, _NnUtilsRnn, _RNNBase, _apply_mask,
    _split_heads, _tf, nn, pad_sequence,
)
from ._optim import (
    Adam, AdamW, CosineAnnealingLR, ExponentialLR, LambdaLR, MultiStepLR, Optimizer,
    RMSprop, ReduceLROnPlateau, SGD, StepLR, _LRScheduler, _Optim, _Scheduler, _replace,
    _tf, no_grad, optim, scope,
)
from ._data import (
    ConcatDataset, DataLoader, Dataset, RandomSampler, SequentialSampler, Subset,
    TensorDataset, WeightedRandomSampler, _CIFAR_RECORD, _Cuda, _Utils, _UtilsData,
    _from_plain, _np_to_u8, _opfs_read, _opfs_write, _tf, _to_plain, _u8_to_np, backend,
    cache_get, cache_put, cuda, decode_cifar10, fetch_cached, load, random_split, save,
    utils,
)

# 모듈 함수를 메서드로도 노출한다 — torch 코드는 `x.exp()` 와 `torch.exp(x)` 를
# 섞어 쓴다.
#
# **여기 있어야 한다.** 예전에는 파일 맨 아래, 즉 쪼갠 뒤의 `_data` 에 있었는데
# 거기서는 `globals()` 에 이 이름들이 없다(각자 다른 모듈에 있다). 전부 모이는
# 자리는 여기뿐이다.
#
# 목록은 **코어와 같다.** 손으로 고르지 않고 torch 에게 `x.f(...)` 와 `torch.f(x, ...)`
# 가 같은 값을 내는지 물어 같다고 답한 것만 담았다. 없는 이름은 건너뛴다 — 자매에만
# 없는 것이 몇 개 있고, 없는 것을 붙이려다 임포트가 통째로 멈추면 안 된다.
_AS_METHOD = (
    "abs", "allclose", "argsort", "bmm", "ceil", "chunk", "clamp", "cos", "cosh",
    "cumprod", "cumsum", "diag", "dot", "eq", "equal", "erf", "exp", "flip", "floor",
    "gather", "ge", "gt", "index_select", "isfinite", "isinf", "isnan", "le", "log",
    "log10", "log2", "lt", "masked_fill", "masked_select", "maximum", "median",
    "minimum", "mm", "movedim", "multinomial", "narrow", "ne", "neg", "norm", "outer",
    "pow", "prod", "reciprocal", "relu", "repeat_interleave", "roll", "round", "rsqrt",
    "sigmoid", "sign", "sin", "sinh", "softmax", "sort", "split", "sqrt", "square",
    "tan", "tanh", "tile", "topk", "trace", "tril", "triu", "unbind", "unique",
    "unsqueeze",
    # 수학 함수 묶음. 코어와 같은 목록이다.
    "acos", "acosh", "arccos", "arccosh", "arcsin", "arcsinh", "arctan", "arctanh",
    "asin", "asinh", "atan", "atan2", "atanh", "absolute", "clip", "copysign",
    "deg2rad", "erfc", "exp2", "expm1", "fix", "frac", "heaviside", "hypot", "ldexp",
    "log1p", "logaddexp", "logaddexp2", "logit", "negative", "positive", "rad2deg",
    "sgn", "signbit", "sinc", "trunc", "xlogy",
)

for _method in _AS_METHOD:
    if _method in globals() and not hasattr(Tensor, _method):
        setattr(Tensor, _method, globals()[_method])


def _where_method(self, condition, other):
    """**인자 순서가 함수와 다르다.** `x.where(조건, y)` 는 `torch.where(조건, x, y)` 다.

    이것만 그냥 붙이면 `x` 가 조건 자리로 들어가 조용히 틀린 답이 나온다.
    """
    return where(condition, self, other)


Tensor.where = _where_method
