"""파이썬 이름을 borch.ts 메서드로 넘긴다 — **목록 없이.**

손으로 이름 192 개를 적으면 그중 하나가 다른 연산을 부르는 날이 오고, 그것은 값
대조로만 보인다. 표를 읽어 다는 방법도 있지만 그러려면 `index.ts` 가 커널 표를
내보내야 하고, 그건 안쪽 사정을 공개 표면에 올리는 것이다.

그래서 **모듈의 `__getattr__`** 을 쓴다(PEP 562). `L.exp(x)` 가 오면 그때
`x.exp()` 로 넘기고, borch.ts 에 없으면 `AttributeError` 로 멈춘다. 없는 것을
근사하지 않는다 — 골든이 그 자리를 실패로 세고, 그 수가 곧 이 결속의 진도다.

이름 규칙 하나만 다르다. 파이썬은 `masked_select`, JS 는 `maskedSelect` 다.
"""

import builtins                    # `max`·`sum` 을 이름 가림 없이 부르려고
import numpy as _np

import js as _js
from pyodide.ffi import to_js as _to_js

from ._base import (
    LinAlgError as _LinAlgError, Tensor, _DType, _js_list, _js_options, _Size,
    guarded, handle, settle, wrap,
)

_ts = _js.borch

# 두 언어에서 철자가 아예 다른 것들. 규칙으로 안 되는 것만 적는다.
_RENAME = {
    # **`linalg.lu_solve` 는 인수가 받는다.** borch.ts 의 `luSolve` 는 torch 의
    # `Tensor.lu_solve` 라 오른쪽 변이 받으므로, 그냥 camel 로 넘기면 수신자가
    # 뒤바뀐다 — 이름도 인자 개수도 맞아서 값만 틀린다. 인수가 받는 쪽은
    # `luSolveFactored` 로 따로 있다.
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
    # 연산 표의 이름이 파이썬과 같은 것들. `camel` 을 씌우면 오히려 없는 이름이 된다.
    "logical_not": "logical_not",
    "logical_and": "logical_and",
    "logical_or": "logical_or",
    "logical_xor": "logical_xor",
    # 비트 연산도 표의 이름이 파이썬과 같다. `camel` 을 씌우면 `bitwiseAnd` 라는
    # 없는 이름이 된다.
    "bitwise_and": "bitwise_and",
    "bitwise_or": "bitwise_or",
    "bitwise_xor": "bitwise_xor",
    "bitwise_not": "bitwise_not",
    "bitwise_left_shift": "bitwise_left_shift",
    "bitwise_right_shift": "bitwise_right_shift",
    "matmul": "mm",
    "var": "variance",
    # **`fill` 은 여기 없다.** 별칭은 밑줄을 뗀 뒤에 찾으므로 여기 적으면 `fill_` 까지
    # 따라와 `fillWith_` 라는 없는 이름이 된다 — `fill_` 은 제자리라 다른 문이다.
    "arctan2": "atan2",
}

# **이름 붙은 인자를 자리로 바꾼다.**
#
# torch 코드는 `clamp(x, min=-0.5, max=0.5)` 처럼 이름으로 부르는 자리가 많은데,
# JS 에는 그런 것이 없다. 처음에 `**kw` 를 그냥 버렸더니 `clamp(x, undefined,
# undefined)` 가 셰이더로 내려가 WGSL 이 파싱에서 멈췄다 — 실패 72 건이 그것이었다.
#
# 그래서 자리 이름을 적어 둔다. borch.ts 쪽 **인자 순서**이고, torch 의 이름을 그
# 자리에 놓는다. 여기 없는 함수는 이름 붙은 인자를 안 받는다는 뜻이다.
_SIGNATURE = {
    "clamp": ("min", "max"),
    "clip": ("min", "max"),
    # **`dtype` 이 셋째 자리다.** torch 의 서명 그대로다 — 넣기 전에 형을 바꾸라는
    # 뜻이고, 그 순서가 값을 바꾼다(실수를 정수로 접을 때).
    "sum": ("dim", "keepdim", "dtype"),
    "mean": ("dim", "keepdim", "dtype"),
    "prod": ("dim", "keepdim", "dtype"),
    "nansum": ("dim", "keepdim", "dtype"),
    "nanmean": ("dim", "keepdim", "dtype"),
    "amax": ("dim", "keepdim"),
    "amin": ("dim", "keepdim"),
    "var": ("dim", "keepdim"),
    "std": ("dim", "keepdim"),
    "logsumexp": ("dim", "keepdim"),
    "argmax": ("dim", "keepdim"),
    "argmin": ("dim", "keepdim"),
    "softmax": ("dim",),
    "log_softmax": ("dim",),
    "cumsum": ("dim", "dtype"),
    "cumprod": ("dim", "dtype"),
    "logcumsumexp": ("dim",),
    "mvlgamma": ("p",),
    "clamp_max": ("max",),
    "clamp_min": ("min",),
    "fill": ("value",),
    "sort": ("dim", "descending"),
    "topk": ("k", "dim", "largest"),
    "squeeze": ("dim",),
    "unsqueeze": ("dim",),
    "flatten": ("start_dim", "end_dim"),
    # 활성함수의 인자. `F.celu(x, alpha=0.5)` 처럼 이름으로 부르는 자리가 많다.
    "celu": ("alpha",),
    "hardshrink": ("lambd",),
    "softshrink": ("lambd",),
    "hardtanh": ("min_val", "max_val"),
    "softplus": ("beta", "threshold"),
    "softmin": ("dim",),
    "glu": ("dim",),
    # 분해의 갈래. **버리면 예외가 아니라 조용히 다른 답이 나온다** —
    # `qr(mode="complete")` 가 축소본을, `svd(full_matrices=False)` 가 완전본을 낸다.
    "qr": ("mode",),
    "svd": ("full_matrices",),
    # 조합층의 이름 붙은 인자들.
    "vector_norm": ("ord", "dim"),
    "matrix_norm": ("ord",),
    "vander": ("N",),
    "vecdot": ("other", "dim"),
    "eigvalsh": ("UPLO",),
    "solve_triangular": ("b", "upper", "left", "unitriangular"),
    # 정규화·전치 합성곱. borch.ts 쪽 인자 순서다.
    "group_norm": ("num_groups", "eps"),
    "instance_norm": ("eps",),
    "dropout": ("p", "training"),
    "rms_norm": ("normalized_shape", "eps"),
    "conv_transpose1d": ("weight", "bias", "stride", "padding"),
    "conv_transpose2d": ("weight", "bias", "stride", "padding"),
    "conv_transpose3d": ("weight", "bias", "stride", "padding"),
    "flip": ("dims",),
    "roll": ("shifts", "dims"),
    "norm": ("p", "dim", "keepdim"),
    "diff": ("n", "dim", "prepend", "append"),
    # 이름이 갈리는 자리 — 파이썬은 `rounding_mode`, JS 는 `roundingMode` 다.
    # `_SIGNATURE` 는 **torch 의 이름**을 적고 자리는 borch.ts 의 것을 따른다.
    "div": ("other", "rounding_mode"),
    "dist": ("other", "p"),
    "bincount": ("weights", "minlength"),
    "cholesky": ("upper",),
    "diag": ("diagonal",),
    "diagflat": ("offset",),
    "allclose": ("other", "rtol", "atol", "equal_nan"),
    "median": ("dim", "keepdim"),
    "gather": ("dim", "index"),
    "index_select": ("dim", "index"),
    "narrow": ("dim", "start", "length"),
    "transpose": ("dim0", "dim1"),
    "swapdims": ("dim0", "dim1"),
    "movedim": ("source", "destination"),
    "repeat_interleave": ("repeats", "dim"),
    "cat": ("dim",),
    "stack": ("dim",),
    "split": ("size", "dim"),
    "chunk": ("chunks", "dim"),
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
    "smooth_l1_loss": ("target", "beta"),
    "interpolate": ("scale_factor",),
    "max": ("dim", "keepdim"),
    "min": ("dim", "keepdim"),
    "aminmax": ("dim",),
    # 참거짓 축약과 개수 세기. **오래 축 자체가 없었다** — 인자를 주면 조용히
    # 버려지고 전체 축약이 나왔다.
    "all": ("dim", "keepdim"),
    "any": ("dim", "keepdim"),
    "count_nonzero": ("dim",),
    "kthvalue": ("k", "dim", "keepdim"),
    "quantile": ("q", "dim"),
    "index_fill": ("dim", "index", "value"),
    "scatter": ("dim", "index", "src"),
    "cumulative_trapezoid": ("dim",),
    "diagonal": ("offset",),
    "squeeze": ("dim",),
    "expand": ("shape",),
    "unflatten": ("dim", "sizes"),
    "quantile": ("q", "dim"),
    "add_": ("other", "alpha"),
    "sub_": ("other", "alpha"),
    "add": ("other", "alpha"),
    "sub": ("other", "alpha"),
    # 모양·색인. borch.ts 쪽 인자 순서이고, torch 의 이름을 그 자리에 놓는다.
    "as_strided": ("size", "stride", "storage_offset"),
    "as_strided_": ("size", "stride", "storage_offset"),
    "as_strided_scatter": ("src", "size", "stride", "storage_offset"),
    "select_scatter": ("src", "dim", "index"),
    "slice_scatter": ("src", "dim", "start", "end", "step"),
    "diagonal_scatter": ("src", "offset", "dim1", "dim2"),
    "diag_embed": ("offset", "dim1", "dim2"),
    "tensor_split": ("indices_or_sections", "dim"),
    "split_with_sizes": ("split_sizes", "dim"),
    "unique_consecutive": ("return_inverse", "return_counts", "dim"),
    "masked_scatter": ("mask", "source"),
    "masked_scatter_": ("mask", "source"),
    "index_put": ("indices", "values", "accumulate"),
    "index_put_": ("indices", "values", "accumulate"),
    "index_reduce": ("dim", "index", "source", "reduce", "include_self"),
    "scatter_reduce": ("dim", "index", "src", "reduce", "include_self"),
    "put": ("index", "source", "accumulate"),
    "renorm": ("p", "dim", "maxnorm"),
    # addmm 계열. torch 에서 `beta`·`alpha`·`value` 는 **이름으로만** 받는 자리라
    # (`*` 뒤에 있다) 케이스가 늘 이름으로 부른다.
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
    # 최상위 선형대수. borch.ts 쪽 인자 순서다.
    "cholesky_solve": ("input2", "upper"),
    "cholesky_inverse": ("upper",),
    "triangular_solve": ("A", "upper", "transpose", "unitriangular"),
    "orgqr": ("input2",),
    "ormqr": ("tau", "other", "left", "transpose"),
    "lobpcg": ("k", "largest"),
    "svd_lowrank": ("q", "niter", "M"),
    "pca_lowrank": ("q", "center", "niter"),
    # 통계. borch.ts 쪽 인자 순서다.
    "histc": ("bins", "min", "max"),
    "histogram": ("bins", "range", "weight", "density"),
    "histogramdd": ("bins",),
    "mode": ("dim", "keepdim"),
    "nanmedian": ("dim", "keepdim"),
    "gradient": ("spacing", "dim", "edge_order"),
    "nonzero_static": ("size", "fill_value"),
}

# **목록을 통째로 받는 자리들.** `permute([0,2,1])` 은 JS 쪽이 배열 하나를 받는데,
# 파이썬은 `permute(0, 2, 1)` 로도 부른다. 흩어진 인자를 하나로 모아야 한다 —
# 안 모으면 `order.map is not a function` 이 난다.
_GATHERS = frozenset(("permute", "reshape", "view", "broadcast_to"))

# **가변 인자로 받는 것들.** borch.ts 가 `expand(...sizes)` 라 배열이 아니라 흩어진
# 수를 원한다 — `_GATHERS` 와 정확히 반대다. 파이썬은 둘 다로 부르므로 여기서 편다.
_SPREADS = frozenset(("expand", "tile", "repeat"))


def camel(name):
    """`masked_select` → `maskedSelect`. 밑줄 뒤 첫 글자를 올린다.

    **끝의 밑줄은 살린다.** `zero_` 는 제자리 연산이라는 뜻이고 borch.ts 도 같은
    이름을 쓴다 — 그냥 나누면 `zero` 가 되어 없는 이름이 된다.

    **별칭은 밑줄을 뗀 뒤에 찾는다.** `absolute` 는 표에 있는데 `absolute_` 는 없어서
    별칭이 안 걸리고 `absolute_` 가 그대로 나갔다 — borch.ts 에는 `abs_` 만 있으므로
    없는 이름이 된다. 제자리 판은 별칭도 같이 따라가야 한다.
    """
    tail = "_" if name.endswith("_") and not name.endswith("__") else ""
    bare = name[:-1] if tail else name
    if bare in _RENAME:
        return _RENAME[bare] + tail
    head, *rest = bare.split("_")
    return head + "".join(p[:1].upper() + p[1:] for p in rest) + tail


# borch.ts 쪽이 인자를 **선언조차 안 한** 이름들. 한 번 물어보고 기억한다.
_NULLARY = {}


def refuse_if_nullary(js_name, fn, count):
    """넘기는 인자를 저쪽이 **안 받으면 멈춘다.**

    이 결속의 구조적 구멍이었다. JS 는 남는 인자를 조용히 버리므로, borch.ts 의
    `sum()` 에 `sum(dim=1)` 을 넘기면 **축을 무시한 전체 합이 값으로 나온다.**
    예외도 경고도 없다.

    골든도 못 봤다. `grad::sum(dim)` 케이스가 있었지만 그것은 결과를 스칼라로 접고
    기울기만 봤는데, `sum(dim=1).sum()` 과 `sum().sum()` 의 기울기가 **둘 다 전부
    1** 이라 축을 틀려도 답이 같았다. 이름만 케이스였다.

    그래서 이름 하나를 고치는 대신 부류를 막는다. 인자 목록이 비었다고 **소스에
    적혀 있는** 것만 걸린다 — 기본값이 있는 것은(`softmax(dim = -1)`) 목록에
    이름이 남으므로 여기 안 걸린다. 재보니 이 표에서 걸리는 것은 넷이었다.
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
            f"borch.ts 의 `{js_name}` 은 인자를 안 받는데 {count} 개를 넘겼다.\n"
            f"  그대로 두면 조용히 무시되고 **다른 값**이 나온다.\n"
            f"  이 이름은 `_ops.py` 에 손으로 적어 맞춰야 한다.")


def _arg(a):
    """텐서면 손잡이로, 목록이면 JS 배열로, 나머지는 그대로."""
    if isinstance(a, Tensor):
        return a._h
    if isinstance(a, (list, tuple)):
        return _js_list(a)
    # **형 이름은 벗겨서 넘긴다.** `_DType` 은 `str` 을 물려받는데 그 `str()` 이
    # `torch.float32` 라, 그대로 넘기면 borch.ts 가 모르는 이름을 받는다 —
    # 저쪽은 `"float32"` 만 안다. 축약에 `dtype=` 이 붙으면서 처음 지나는 길이다.
    if isinstance(a, _DType):
        return a.plain
    return a


def positional(name, args, kw):
    """이름 붙은 인자를 **자리로 편다.**

    JS 에는 이름 붙은 인자가 없다. 버리면 `undefined` 가 셰이더까지 내려가고, WGSL 은
    그것을 파싱에서 거절한다 — 조용히 틀리지는 않지만 원인이 한참 멀리서 나온다.

    뒤에 붙는 `undefined` 는 잘라낸다. borch.ts 의 기본값(`stride = kernel`)이
    살아나야 하는데, `undefined` 를 명시로 넘기면 그 자리가 안 채워진다.
    """
    if not kw:
        out = list(args)
    else:
        order = _SIGNATURE.get(name)
        if order is None:
            raise TypeError(
                f"`{name}` 은 이름 붙은 인자를 안 받는다 (받은 것: {sorted(kw)})\n"
                f"  받아야 한다면 `_SIGNATURE` 에 자리 순서를 적어라.")
        out = list(args)
        for i, key in enumerate(order):
            if key in kw:
                while len(out) <= i:
                    out.append(None)
                out[i] = kw[key]
    while out and out[-1] is None:
        out.pop()
    # 흩어진 축 번호를 배열 하나로 모은다. `permute(0, 2, 1)` → `permute([0,2,1])`.
    if name in _GATHERS and all(isinstance(a, int) for a in out):
        out = [list(out)]
    # 반대로, 배열 하나로 온 것을 흩뿌린다. `expand([2,3])` → `expand(2, 3)`.
    elif name in _SPREADS and len(out) == 1 and isinstance(out[0], (list, tuple)):
        out = list(out[0])
    return [_arg(a) for a in out]


# borch.ts 에서 **이항 표에만 있는 것들.** 메서드로는 안 달려 있고
# `x.binary("maximum", y)` 로 부른다 — 표에서 자동으로 메서드가 되는 것은 단항뿐이다.
_BINARY_ONLY = frozenset((
    "maximum", "minimum", "atan2", "hypot", "copysign", "logaddexp",
    "logaddexp2", "xlogy", "heaviside", "ldexp", "pow",
    "eq", "ne", "lt", "le", "gt", "ge",
    "logical_and", "logical_or", "logical_xor",
    "bitwise_and", "bitwise_or", "bitwise_xor",
    "bitwise_left_shift", "bitwise_right_shift", "gcd", "lcm", "nextafter",
    "arctan2",
))


def __getattr__(name):
    """모듈에 없는 이름은 **첫 인자의 메서드**로 넘긴다.

    `torch.exp(x)` 와 `x.exp()` 가 같은 것이라는 torch 의 규칙을 그대로 쓴다.
    """
    if name.startswith("_"):
        raise AttributeError(name)
    # dtype 이름들. `bool` 을 모듈 전역에 두면 파이썬 내장을 가리므로 여기서 준다.
    if name in ("bool", "float32", "int64"):
        from ._base import _DType
        return _DType(name)
    # `max`·`min` 도 같은 이유로 여기서 준다 — 위에 적은 그대로다.
    if name in _EXTREME:
        return _EXTREME[name]
    # 비교의 다른 이름들 — 표에 있는 이름으로 넘긴다.
    if name in _COMPARE_ALIAS:
        return __getattr__(_COMPARE_ALIAS[name])
    # **제자리 판은 파이썬 텐서의 문을 지나야 한다.** 여기서 곧장 JS 손잡이로 가면
    # 두 가지가 어긋난다 — borch.ts 에 제자리 판이 없는 이름(`gcd_`·`clampMax_`)에서
    # 멈추고, 있는 이름도 **새 파이썬 텐서**를 돌려줘서 `torch.detach_(y) is y` 가
    # 거짓이 된다. 그 두 일을 하는 곳이 `Tensor.__getattr__` 이므로 그리로 넘긴다.
    if name.endswith("_") and not name.endswith("__"):
        def call(x, *args, **kw):
            return getattr(wrap(x), name)(*args, **kw)
        call.__name__ = name
        return call

    js_name = camel(name)

    if name in _BINARY_ONLY:
        def call(a, b, *rest, **kw):
            return guarded(handle(a).binary, js_name, handle(b))
        call.__name__ = name
        return call

    def call(x, *args, **kw):
        h = handle(x)
        fn = getattr(h, js_name, None)
        if fn is None:
            raise AttributeError(
                f"borch.ts 에 `{js_name}` 이 없다 (파이썬 이름 `{name}`)")
        laid = positional(name, args, kw)
        refuse_if_nullary(js_name, fn, len(laid))
        return guarded(fn, *laid)

    call.__name__ = name
    return call


# ── 첫 인자가 텐서가 아닌 것들. 여기만 손으로 적는다. ────────────────────

def arange(*args, **kw):
    """`arange(n)` · `arange(a, b)` · `arange(a, b, step)`.

    **borch.ts 의 `arange` 는 개수 하나만 받는다.** 셋을 넘겼더니 첫 인자 0 이 개수로
    읽혀 빈 텐서가 나왔고, 그 빈 텐서를 `reshape` 하는 자리에서 90 건이 무너졌다 —
    실패 문구는 `shape '[3,3]' is invalid for input of size 0` 이었고, 원인에서
    두 칸 떨어진 자리다. 나머지 두 꼴은 여기서 만든다.
    """
    if len(args) == 1:
        start, stop, step = 0, args[0], 1
    elif len(args) == 2:
        (start, stop), step = args, 1
    else:
        start, stop, step = args
    n = max(0, -(-int(stop - start) // int(step)) if step else 0)
    out = _ts.Tensor.arange(n)
    if start or step != 1:
        out = out.binary("mul", _ts.Tensor.full(_js_list([]), float(step)))
        out = out.binary("add", _ts.Tensor.full(_js_list([]), float(start)))
    return wrap(out)


def _shape_of(shape):
    """`zeros(2, 3)` 와 `zeros([2, 3])` 을 둘 다 받는다 — torch 가 그렇다."""
    if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
        return _js_list(shape[0])
    return _js_list(shape)


def zeros(*shape, **kw):
    return wrap(_ts.Tensor.zeros(_shape_of(shape)))


def ones(*shape, **kw):
    return wrap(_ts.Tensor.ones(_shape_of(shape)))


def full(shape, value, **kw):
    return wrap(_ts.Tensor.full(_js_list(shape), float(value)))


def eye(n, m=None, **kw):
    return wrap(_ts.Tensor.eye(n, n if m is None else m))


def cat(parts, dim=0):
    return wrap(_ts.Tensor.cat(_js.Array.from_([p._h for p in parts]), dim))


def stack(parts, dim=0):
    return wrap(_ts.Tensor.stack(_js.Array.from_([p._h for p in parts]), dim))


class scope:                                             # noqa: N801
    """`with L.scope():` — 이 안에서 만든 GPU 버퍼를 나갈 때 놓는다.

    **학습 루프에 이것이 없으면 안 돈다.** 한 스텝이 중간 버퍼를 수천 개 만들고,
    파이썬·자바스크립트 어느 쪽 쓰레기 수집도 GPU 메모리를 제때 안 놓아준다.
    자매도 같은 이유로 같은 이름을 노출한다.
    """

    def __enter__(self):
        _ts.device().beginScope()
        return self

    def __exit__(self, *exc):
        _ts.device().endScope()
        return False


def memory():
    """지금 잡고 있는 것. **벤치가 누수를 재는 자리다.**

    자매는 `js.tf.memory()` 를 직접 불렀는데, 그러면 계측이 TF.js 에 묶여서 다른
    구현으로는 같은 벤치를 못 돌린다. 라이브러리에 물으면 누가 밑에 있든 답한다.
    """
    got = _ts.device().memory
    return {"tensors": int(got.tensors), "bytes": int(got.bytes)}


class no_grad:                                           # noqa: N801
    """`with L.no_grad():`.

    borch.ts 의 `noGrad` 는 **함수를 받는다**(`noGrad(() => …)`). 파이썬은 `with` 를
    쓰므로 여기서 모양을 바꾼다 — 안쪽 스위치를 직접 여닫는다.
    """

    def __enter__(self):
        _ts.gradMode.enabled = False
        return self

    def __exit__(self, *exc):
        _ts.gradMode.enabled = True
        return False


class enable_grad:                                       # noqa: N801
    """`no_grad` 안에서 **다시 켠다.** 중첩이 되어야 하므로 이전 값을 되돌린다."""

    def __enter__(self):
        self._prev = bool(_ts.gradMode.enabled)
        _ts.gradMode.enabled = True
        return self

    def __exit__(self, *exc):
        _ts.gradMode.enabled = self._prev
        return False


class set_grad_enabled:                                  # noqa: N801
    """켤지 끌지를 값으로 받는다. 부르는 순간 바뀌고, `with` 를 나가면 되돌아온다."""

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
    """**여기서는 `no_grad` 와 같다.** 진짜 torch 는 안에서 만든 텐서에 표를 붙이는데,
    그 표를 흉내내면 "왜 이 텐서를 못 쓰나" 하는 오류를 우리가 만들어 내는 셈이다."""

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
    """**늘 거짓이다** — 그 표를 안 붙이므로 없다고 말하는 것이 사실이다."""
    return False


def is_inference_mode_enabled():
    return False


# ── 난수 상태 ───────────────────────────────────────────────────────────────

_LAST_SEED = [0]


def initial_seed():
    return _LAST_SEED[0]


def seed():
    got = int(_np.random.SeedSequence().entropy % (2 ** 63))
    manual_seed(got)
    return got


def get_rng_state():
    """**두 생성기의 상태를 함께 담는다.** numpy 쪽은 `randn`·`randperm` 이 쓰고
    borch.ts 쪽 씨앗은 층 초기화와 dropout 이 쓴다 — 하나만 담으면 되돌려도
    나머지가 안 돌아간다."""
    return {"numpy": dict(_rng.bit_generator.state),
            "ts": int(_ts.Tensor.dropoutSeed)}


def set_rng_state(state):
    if not isinstance(state, dict) or "numpy" not in state:
        raise RuntimeError("set_rng_state — `get_rng_state` 가 준 것만 받습니다")
    _rng.bit_generator.state = state["numpy"]
    _ts.Tensor.dropoutSeed = state["ts"]
    return None


# ── 살펴보기 ────────────────────────────────────────────────────────────────

def is_tensor(x):
    return isinstance(x, Tensor)


def is_storage(x):
    """**늘 거짓이다.** 저장(Storage) 이라는 층을 우리는 안 둔다."""
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


def can_cast(from_type, to_type):
    """**한 방향만 참이다.** 좁아지는 쪽(실수 → 정수)은 거짓이다."""
    names = [_dtype_name(t) for t in (from_type, to_type)]
    if any(n not in _PROMOTE_ORDER for n in names):
        return False
    return _PROMOTE_ORDER.index(names[0]) <= _PROMOTE_ORDER.index(names[1])


def get_default_dtype():
    from . import _base
    return _base._DType("float32")


def set_default_dtype(dt):
    """받되 바꾸지 않는다 — 저장이 float32 하나다. 그 밖은 시끄럽게 거절한다."""
    if _dtype_name(dt) != "float32":
        raise RuntimeError(f"set_default_dtype({dt}) — 저장이 float32 하나입니다")
    return None


class _FInfo:
    def __init__(self, dt):
        info = _np.finfo(_np.float32)
        self.eps = float(info.eps)
        self.max = float(info.max)
        self.min = float(info.min)
        self.tiny = float(info.tiny)
        self.smallest_normal = float(info.tiny)
        self.resolution = float(info.resolution)
        self.bits = int(info.bits)
        self.dtype = _dtype_name(dt)


class _IInfo:
    def __init__(self, dt):
        info = _np.iinfo(_np.int64)
        self.max = int(info.max)
        self.min = int(info.min)
        self.bits = int(info.bits)
        self.dtype = _dtype_name(dt)


def finfo(dt=None):
    return _FInfo(dt if dt is not None else "float32")


def iinfo(dt):
    return _IInfo(dt)


def linspace(start, end, count, **kw):
    return wrap(_ts.Tensor.linspace(start, end, count))


# ── 창 함수 ─────────────────────────────────────────────────────────────────
#
# 텐서를 받지 않고 **개수를 받는다** — 첫 인자의 메서드로 넘기는 길이 안 통하므로
# 여기 손으로 적는다. borch.ts 쪽이 CPU 에서 만들고, `periodic` 의 규약도 저쪽에 있다.

def bartlett_window(n, periodic=True, **kw):
    return wrap(_ts.Tensor.bartlettWindow(n, periodic))


def hann_window(n, periodic=True, **kw):
    return wrap(_ts.Tensor.hannWindow(n, periodic))


def hamming_window(n, periodic=True, alpha=0.54, beta=0.46, **kw):
    return wrap(_ts.Tensor.hammingWindow(n, periodic, alpha, beta))


def blackman_window(n, periodic=True, **kw):
    return wrap(_ts.Tensor.blackmanWindow(n, periodic))


def kaiser_window(n, periodic=True, beta=12.0, **kw):
    """**`beta` 는 자리 인자다** — torch 가 `kaiser_window(n, periodic, beta)` 로 받는다."""
    return wrap(_ts.Tensor.kaiserWindow(n, periodic, beta))


# **난수는 한 흐름에서 나온다.** 처음에는 부를 때마다 `default_rng(0)` 을 새로
# 만들었다. 골든이 난수를 오류 케이스에서만 써서(던지는지만 본다) 값이 늘 같아도
# 안 걸렸는데, 그 상태로는 셔플하는 `DataLoader` 가 **매 에폭 같은 순서**를 낸다.
# 부르는 쪽에서 보면 셔플을 켰는데 안 섞이는 것이고, 아무 예외도 안 난다.
_rng = _np.random.default_rng(0)


def manual_seed(seed):
    """씨앗을 다시 심는다. torch 와 같은 이름·같은 뜻이다.

    **저쪽에도 심어야 한다.** 여기 numpy 생성기는 `randn`·`randperm` 이 쓰고, 층
    초기화와 dropout 은 borch.ts 안의 다른 생성기가 쓴다 — 이쪽만 심으면 `randn` 은
    재현되고 가중치는 매번 달라진다. "같은 씨앗에 같은 결과" 를 확인하는 사람이 가장
    먼저 보는 것이 그 둘인데, 앞의 것만 재현되니 먹는 줄 알고 넘어간다.

    세 구현이 같은 갈래의 결함을 하나씩 갖고 있었고 게으른 층 케이스가 셋 다 잡았다.
    """
    global _rng
    _rng = _np.random.default_rng(seed)
    _ts.nn.manualSeed(int(seed))
    _LAST_SEED[0] = int(seed)
    return _rng


class Generator:
    """`random_split(..., generator=g)` 처럼 흐름을 따로 두고 싶을 때."""

    def __init__(self, device=None):
        self.seed = 0

    def manual_seed(self, seed):
        self.seed = seed
        return self

    def rng(self):
        return _np.random.default_rng(self.seed)


def _shaped(shape):
    return shape[0] if len(shape) == 1 and isinstance(shape[0], (list, tuple)) else shape


def randn(*shape, **kw):
    """정규분포 난수. **borch.ts 에는 없어서 여기서 만든다.**

    골든에서 이것을 쓰는 자리는 오류 케이스뿐이고, 거기서는 값을 안 보고 **던지는지**만
    본다(`L.randn(3, 4) @ L.randn(3, 2)`). 값을 묻는 케이스가 생기면 그때는 borch.ts
    쪽에 제대로 넣어야 한다 — 여기 있는 것은 CPU 를 한 번 거친다.
    """
    from ._base import tensor as _t

    return _t(_rng.standard_normal(tuple(_shaped(shape))).astype("float32"),
              requires_grad=kw.get("requires_grad", False))


def rand(*shape, **kw):
    from ._base import tensor as _t

    return _t(_rng.random(tuple(_shaped(shape))).astype("float32"),
              requires_grad=kw.get("requires_grad", False))


def randint(low, high=None, size=(), **kw):
    from ._base import tensor as _t

    if high is None:
        low, high = 0, low
    return _t(_rng.integers(low, high, tuple(size)).astype("int64"))


def randperm(n, **kw):
    from ._base import tensor as _t

    return _t(_rng.permutation(n).astype("int64"))


def einsum(spec, *operands):
    """borch.ts 의 `einsum` 은 자유 함수이고 **피연산자를 흩어서** 받는다."""
    return guarded(_ts.einsum, spec, *[handle(t) for t in operands])


def as_tensor(data, dtype=None):
    from ._base import tensor as _t
    return data if isinstance(data, Tensor) else _t(data, dtype)


def matrix_power(x, n):
    """**음수 지수는 역행렬의 거듭제곱이다.** borch.ts 는 1 이상만 한다.

    `A^-2 = (A^-1)^2` 라, 뒤집고 나서 양수로 부르면 된다. 0 은 단위행렬이다.
    """
    h = handle(x)
    if n == 0:
        return wrap(_ts.Tensor.eye(int(h.shape[0]), int(h.shape[0])))
    if n < 0:
        h = settle(h.inverse())._h
        n = -n
    return guarded(h.matrixPower, n)


def quantile(x, q, dim=None, **kw):
    """`q` 는 수 하나일 수도 목록일 수도 있다 — borch.ts 는 늘 목록을 받는다."""
    one = isinstance(q, (int, float))
    qs = [float(v) for v in ([q] if one else q)]
    out = guarded(handle(x).quantile, _to_js(qs))
    # **수 하나를 주면 스칼라가 나온다.** torch 가 그렇다 — 목록으로 물었을 때만
    # 축이 생긴다. borch.ts 는 늘 목록이라 여기서 접는다.
    return wrap(out._h.reshape(_js_list([]))) if one else out


def numel(x, **kw):
    """원소 수. borch.ts 에서는 `size` 라는 **속성**이라 이름도 꼴도 다르다."""
    return int(handle(x).size)


def _reduce_all(name):
    """축을 **안 준** 축약. torch 는 평평하게 편 뒤 하나를 낸다.

    borch.ts 쪽은 축이 기본값 0 이라 그대로 넘기면 **열마다 하나씩** 나온다. 모양이
    달라서 값 대조에서 걸리기는 하지만, 걸리기 전까지는 "되는데 좀 이상한" 상태다.
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
    """`max`·`min` 의 **세 얼굴.** torch 는 인자에 따라 다른 것을 낸다.

    - `max(x)` → 전부의 최댓값 **하나** (짝이 아니다)
    - `max(x, dim)` → `(값, 번호)` 짝
    - `max(x, other)` → **칸마다**의 최댓값

    셋이 한 이름에 붙어 있는 것이 헷갈리는 자리인데, 그것이 torch 의 계약이므로
    여기서 정리하면 안 된다. 정리하면 교재 코드가 안 돈다.

    세 번째 갈래가 없었고, 첫 번째는 짝을 냈다. 골든이 세 갈래를 따로 묻고서야
    드러났다 — `x.max()` 를 스칼라로 바꿀 때만 시끄럽고, 비교에 쓰면 칸마다 비교가
    되어 조용히 다른 답이다.
    """
    pair = {"max": "amax", "min": "amin"}[name]
    elementwise = {"max": "maximum", "min": "minimum"}[name]

    def call(x, dim=None, keepdim=False, other=None, **kw):
        dim = kw.get("dim", dim)
        other = kw.get("other", other)
        if isinstance(dim, Tensor):                # `max(x, other)` 꼴
            dim, other = None, dim
        h = handle(x)
        if other is not None:
            return wrap(guarded(h.binary, elementwise, handle(other)))
        if dim is None:
            return wrap(guarded(getattr(h, pair)))
        # 축 하나면 짝이다. `guarded` 가 이미 자리에 이름을 붙여 주므로 다시 안 묶는다.
        return guarded(getattr(h, name), dim, bool(kw.get("keepdim", keepdim)))
    call.__name__ = name
    return call


# **모듈 전역에 `max`·`min` 을 두면 안 된다.** 이 파일 안에서 파이썬 내장을 가리고,
# 그러면 `max(a, b)` 로 크기를 재던 자리가 텐서 함수를 부른다 — 증상은 GPU 버퍼
# 할당이 통째로 죽는 것이었고(`createBuffer` 실패 128 건), 원인에서 아주 멀다.
# `bool` 에서 같은 것을 겪고 그 자리에 적어 두었는데도 다시 밟았다.
#
# 아래 `__getattr__` 이 이 이름들을 내준다. 밖에서는 `L.max(x)` 로 보이고 안에서는
# 내장이 그대로 산다.
_EXTREME = {"max": _extreme("max"), "min": _extreme("min")}


def flatten(x, start_dim=0, end_dim=-1, **kw):
    """축을 접는다. **borch.ts 에는 없어서 `reshape` 로 만든다** — 정의 그대로다.

    **기본이 `start_dim=0` 이다.** `nn.Flatten` 층은 배치를 남기느라 1 부터 접지만
    `torch.flatten` 함수는 0 부터다. 층의 기본값을 함수에 옮겨 적었더니 `flatten(x)`
    가 배치를 남겨서 모양이 달랐다.
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


def squeeze(x, dim=None, **kw):
    """`dim` 이 없으면 torch 는 **길이 1 인 축을 전부** 없앤다. borch.ts 는 하나씩이다."""
    h = handle(x)
    dim = kw.get("dim", dim)
    if dim is not None:
        return guarded(h.squeeze, dim)
    keep = [int(n) for n in h.shape if int(n) != 1]
    return guarded(h.reshape, _js_list(keep))


def sum(x, dim=None, keepdim=False, dtype=None, **kw):   # noqa: A001
    """borch.ts 는 전체 합(`sum()`)과 축 합(`sumDim`)을 **다른 이름**으로 둔다.

    이 자리가 조용히 틀렸다. `sum(dim=1)` 이 `sum()` 으로 가서 축을 무시한 스칼라를
    냈고, 예외가 없으니 아무도 몰랐다 — 랭크 6 케이스 하나가 모양으로 걸릴 때까지.
    """
    dim = kw.get("dim", dim)
    keepdim = kw.get("keepdim", keepdim)
    dtype = kw.get("dtype", dtype)
    h = handle(x)
    # **`dtype=` 는 전체 합에도 붙는다.** borch.ts 쪽 `sum()` 은 그 인자를 안 받으므로
    # 여기서 앞뒤로 형을 바꿔 준다 — 규칙은 같다: 넣기 전에 바꾸고 결과도 못 박는다.
    if dtype is not None:
        name = dtype.plain if isinstance(dtype, _DType) else str(dtype)
        cast = wrap(guarded(h.to, name.replace("torch.", "")))
        return wrap(guarded(handle(sum(cast, dim, keepdim)).to,
                            name.replace("torch.", "")))
    if dim is None:
        return guarded(h.sum)
    return guarded(h.sumDim, dim, bool(keepdim))


def norm(x, p=2, dim=None, keepdim=False, **kw):
    """borch.ts 의 `norm()` 은 **전체** L2 하나뿐이다. 축과 차수는 여기서 만든다.

    넘겨봐야 조용히 버려지던 자리다 — `norm(dim=1)` 이 전체 노름을 냈다.
    """
    p = kw.get("p", p)
    dim = kw.get("dim", dim)
    keepdim = kw.get("keepdim", keepdim)
    h = handle(x)
    if dim is None and p == 2:
        return guarded(h.norm)
    if p == 2:
        return guarded(handle(guarded(h.square).sumDim(dim, bool(keepdim))).sqrt)
    if p == 1:
        got = guarded(h.abs)
        return wrap(got.sum() if dim is None else got.sumDim(dim, bool(keepdim)))
    if p == float("inf"):
        got = guarded(h.abs)
        return wrap(got.max() if dim is None else got.amax(dim, bool(keepdim)))
    # 아래 셋은 코어가 조용히 2-노름을 내던 자리를 고치면서 같이 열었다. 코어만
    # 고치면 **셋이 서로 갈린다** — 골든은 세 구현에 같은 것을 묻는 자리라, 한쪽만
    # 답할 수 있으면 그 케이스를 아예 못 넣는다.
    if p == float("-inf"):
        got = guarded(h.abs)
        return wrap(got.min() if dim is None else got.amin(dim, bool(keepdim)))
    if p == 0:
        got = handle(x)
        return wrap(got.countNonzero() if dim is None
                    else got.countNonzero(dim)).float()
    if p in (None, "fro"):
        return norm(x, 2, dim, keepdim)
    if p == "nuc":
        raise NotImplementedError(
            "norm('nuc') 는 아직 없다 — 특이값의 합이라 SVD 가 필요하다. 근사하지 않는다")
    powed = handle(guarded(handle(guarded(h.abs)).powScalar, float(p)))
    total = powed.sum() if dim is None else powed.sumDim(dim, bool(keepdim))
    return wrap(handle(wrap(total)).powScalar(1.0 / float(p)))


def transpose(x, dim0=None, dim1=None, **kw):
    """borch.ts 의 `transpose()` 는 **2차원 전용**이고 축을 안 받는다.

    torch 는 어느 랭크에서든 두 축을 바꾼다. 축을 넘기면 버려지던 자리인데, 랭크 2
    에서는 답이 우연히 같고 랭크 3 이상에서는 borch.ts 가 던져서 조용히 틀리지는
    않았다. 그래도 `x.transpose(1, 2)` 는 torch 코드가 늘 하는 일이라 맞춰 준다.
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


# ── torch 가 **두 번째 이름**으로 주는 것들 ─────────────────────────────────
#
# 전부 이미 있는 것의 조합이다. borch.ts 쪽에 이름을 늘리지 않는다 — 계산이 늘어나는
# 것이 아니라 파이썬 코드가 부르는 철자가 늘어나는 것이라, 파이썬 쪽 일이다.

def add(a, b, alpha=1, **kw):
    """`a + alpha·b`. **`alpha` 가 연산자에 없어서** 별칭이 아니라 함수다."""
    alpha = kw.get("alpha", alpha)
    return wrap(a) + (b if alpha == 1 else wrap(b) * alpha)


def sub(a, b, alpha=1, **kw):
    alpha = kw.get("alpha", alpha)
    return wrap(a) - (b if alpha == 1 else wrap(b) * alpha)


def mul(a, b, **kw):
    return wrap(a) * b


def div(a, b, rounding_mode=None, **kw):
    mode = kw.get("rounding_mode", rounding_mode)
    out = wrap(a) / b
    if mode is None:
        return out
    if mode == "floor":
        return wrap(guarded(handle(out).unary, "floor"))
    if mode == "trunc":
        return wrap(guarded(handle(out).unary, "trunc"))
    raise RuntimeError(f"rounding_mode 는 None·'floor'·'trunc' 뿐이다: {mode!r}")


def floor_divide(a, b, **kw):
    return div(a, b, rounding_mode="floor")


def remainder(a, b, **kw):
    """**부호가 나누는 쪽을 따른다.** `fmod` 와 갈리는 자리가 그것이다."""
    a, b = wrap(a), wrap(b)
    return a - wrap(guarded(handle(a / b).unary, "floor")) * b


def fmod(a, b, **kw):
    """**부호가 나뉘는 쪽을 따른다.** C 의 규칙이고 `remainder` 와 반대다."""
    a, b = wrap(a), wrap(b)
    return a - wrap(guarded(handle(a / b).unary, "trunc")) * b


def rsub(a, b, alpha=1, **kw):
    return sub(b, a, alpha)


def t(x, **kw):
    """2 차원 전치. **1 차원 이하는 그대로 둔다** — torch 가 그렇다."""
    h = handle(x)
    return wrap(h) if len(h.shape) < 2 else transpose(x, 0, 1)


def adjoint(x, **kw):
    return transpose(x, -2, -1)


def moveaxis(x, source, destination, **kw):
    return wrap(guarded(handle(x).movedim, source, destination))


def broadcast_to(x, shape, **kw):
    """borch.ts 쪽 이름은 `expand` 다 — 축을 **흩어서** 받는다."""
    return wrap(guarded(handle(x).expand, *[int(n) for n in shape]))


def _broadcast_shape(shapes):
    """오른쪽 맞춤으로 축마다 큰 쪽을 고른다 — numpy 와 같은 규칙이다."""
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
    shape = _broadcast_shape([tuple(int(n) for n in handle(v).shape) for v in tensors])
    return tuple(broadcast_to(v, shape) for v in tensors)


def hstack(tensors, **kw):
    """1 차원은 이어 붙이고 그 위는 **열 방향**으로 붙인다."""
    ts = list(tensors)
    dim = 0 if len(handle(ts[0]).shape) == 1 else 1
    return cat(ts, dim)


def _lift(x, rank):
    """모자란 앞축을 1 로 채운다. `atleast_2d`·`atleast_3d` 가 하는 일이다."""
    h = handle(x)
    shape = [int(n) for n in h.shape]
    if len(shape) >= rank:
        return wrap(h)
    return wrap(guarded(h.reshape, _js_list([1] * (rank - len(shape)) + shape)))


def vstack(tensors, **kw):
    return cat([_lift(v, 2) for v in tensors], 0)


def _atleast3(x):
    """torch 의 `atleast_3d`. **뒤에 축을 붙인다** — 앞이 아니다.

    1 차원 `(n,)` 은 `(1, n, 1)` 이 되고 2 차원 `(m, n)` 은 `(m, n, 1)` 이다. 앞에만
    채우면 `(1, 3, 4)` 가 되어 `dstack` 이 세 번째 축이 아니라 마지막 축으로 붙는다 —
    모양이 `(1, 3, 8)` 로 나와서 걸렸다.
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


def dstack(tensors, **kw):
    return cat([_atleast3(v) for v in tensors], 2)


def column_stack(tensors, **kw):
    """1 차원을 **열 하나로 세워** 붙인다. `hstack` 과 여기서 갈린다."""
    ts = []
    for v in tensors:
        h = handle(v)
        shape = [int(n) for n in h.shape]
        ts.append(wrap(guarded(h.reshape, _js_list([shape[0], 1])))
                  if len(shape) == 1 else wrap(h))
    return cat(ts, 1)


def block_diag(*tensors):
    """대각선에 블록을 늘어놓고 나머지는 0."""
    ts = [_lift(v, 2) for v in tensors]
    widths = [int(handle(v).shape[1]) for v in ts]
    total = builtins.sum(widths)
    lines, at = [], 0
    for v, w in zip(ts, widths):
        h = int(handle(v).shape[0])
        pieces = []
        if at:
            pieces.append(zeros(h, at))
        pieces.append(v)
        if total - at - w:
            pieces.append(zeros(h, total - at - w))
        lines.append(cat(pieces, 1) if len(pieces) > 1 else v)
        at += w
    return cat(lines, 0) if len(lines) > 1 else lines[0]


# ── 계산 자체가 없던 것들. 전부 있는 연산의 조합이다. ───────────────────────

def _shape_list(x):
    return [int(n) for n in handle(x).shape]


def empty_like(t, **kw):
    return zeros(*_shape_list(t))


def rand_like(t, **kw):
    return rand(*_shape_list(t))


def randn_like(t, **kw):
    return randn(*_shape_list(t))


def randint_like(t, low, high=None, **kw):
    if high is None:
        low, high = 0, low
    return randint(low, high, tuple(_shape_list(t)))


def scalar_tensor(value, **kw):
    return full([], float(value))


def logspace(start, end, steps, base=10.0, **kw):
    """`base` 의 거듭제곱으로 고르게. `linspace` 를 지수로 쓴다."""
    return pow(full([], float(base)), linspace(start, end, steps))


def meshgrid(*tensors, indexing="ij"):
    """격자. **`xy` 는 앞의 두 축이 뒤바뀐 것**이라 한 규칙으로 못 덮는다."""
    ts = list(tensors)
    if indexing not in ("ij", "xy"):
        raise RuntimeError(f"indexing 은 'ij' 나 'xy' 다: {indexing!r}")
    order = list(range(len(ts)))
    if indexing == "xy" and len(ts) >= 2:
        order[0], order[1] = order[1], order[0]
    sizes = [_shape_list(ts[i])[0] for i in order]
    out = []
    for place, i in enumerate(order):
        shape = [1] * len(ts)
        shape[place] = sizes[place]
        lifted = wrap(guarded(handle(ts[i]).reshape, _js_list(shape)))
        out.append(broadcast_to(lifted, sizes))
    if indexing == "xy" and len(ts) >= 2:
        out[0], out[1] = out[1], out[0]
    return tuple(out)


def lerp(start, end, weight, **kw):
    start, end = wrap(start), wrap(end)
    return start + (end - start) * weight


def _unary(x, name):
    return wrap(guarded(handle(x).unary, name))


def nan_to_num(t, nan=0.0, posinf=None, neginf=None, **kw):
    """NaN 과 무한대를 유한한 수로. **안 주면 f32 의 끝값이다.**"""
    t = wrap(t)
    hi = 3.4028234663852886e38 if posinf is None else posinf
    lo = -3.4028234663852886e38 if neginf is None else neginf
    # **`where` 는 첫 인자에서 모양을 가져온다.** 스칼라를 거기 넣으면 결과가
    # 스칼라가 된다 — 채울 값도 같은 모양으로 펴 놓고 넘긴다.
    shape = _shape_list(t)

    def spread(v):
        return zeros(*shape) + float(v)

    out = where(_unary(t, "isnan"), spread(nan), t)
    out = where(isposinf(t), spread(hi), out)
    return where(isneginf(t), spread(lo), out)


def isposinf(t, **kw):
    t = wrap(t)
    return _unary(t, "isinf") * (t > full([], 0.0))


def isneginf(t, **kw):
    t = wrap(t)
    return _unary(t, "isinf") * (t < full([], 0.0))


def isreal(t, **kw):
    """실수만 있으므로 전부 참이다. **거짓말이 아니라 사실이다.**"""
    return ones(*_shape_list(t)) > full([], 0.0)


def isclose(a, b, rtol=1e-5, atol=1e-8, equal_nan=False, **kw):
    a, b = wrap(a), wrap(b)
    return _unary(a - b, "abs") <= (wrap(_unary(b, "abs")) * rtol + atol)


def isin(elements, test_elements, **kw):
    """원소가 그 목록에 있는가. 브로드캐스팅 하나로 풀린다."""
    e, t = wrap(elements), wrap(test_elements)
    n = int(handle(e).size)
    m = int(handle(t).size)
    grid = (wrap(guarded(handle(e).reshape, _js_list([n, 1])))
            == wrap(guarded(handle(t).reshape, _js_list([1, m]))))
    hit = wrap(guarded(handle(grid).to, "float32")).sum(dim=1)
    return (hit > full([], 0.0)).reshape(*_shape_list(e))


def _nan_extreme(name, better):
    """`fmax`·`fmin` 은 **NaN 을 건너뛴다** — `maximum` 은 NaN 을 물고 나온다."""
    def call(a, b, **kw):
        a, b = wrap(a), wrap(b)
        picked = wrap(guarded(handle(a).binary, better, handle(b)))
        out = where(_unary(a, "isnan"), b, picked)
        return where(_unary(b, "isnan"), a, out)
    call.__name__ = name
    return call


fmax = _nan_extreme("fmax", "maximum")
fmin = _nan_extreme("fmin", "minimum")


def float_power(a, b, **kw):
    return wrap(a) ** b


def logical_xor(a, b, **kw):
    """borch.ts 의 이항 표에 없다 — **다름**으로 만든다."""
    a, b = wrap(a), wrap(b)
    return (a != full([], 0.0)) != (b != full([], 0.0))


# ── 모양·색인 ───────────────────────────────────────────────────────────────
#
# 손으로 적는 것은 세 종류다 — **텐서를 안 받는 것**(삼각 자리표), **텐서 목록을
# 받는 것**(`index_put` 의 색인들, `cartesian_prod`), 그리고 **묶음으로 답하는
# 것**(쪼개기·`unravel_index`). 나머지는 `__getattr__` 이 그냥 넘긴다.
#
# 텐서 목록이 왜 따로인가: `_arg` 는 목록을 `_js_list` 로 보내는데 그쪽이 `int()` 를
# 씌운다. 텐서가 담긴 목록이 거기 들어가면 정수 변환에서 멈춘다.

def _js_tensors(seq):
    return _js.Array.from_([handle(t) for t in seq])


def index_put(t, indices, values, accumulate=False, **kw):
    return wrap(guarded(handle(t).indexPut, _js_tensors(indices),
                        handle(values), accumulate))


def index_put_(t, indices, values, accumulate=False, **kw):
    t = wrap(t)
    guarded(handle(t).indexPut_, _js_tensors(indices), handle(values),
            accumulate)
    return t


def unravel_index(indices, shape, **kw):
    """**축마다 텐서 하나씩, 묶음으로 낸다**(실측)."""
    got = guarded(handle(indices).unravelIndex, _js_list(shape))
    return tuple(wrap(p) for p in got)


def unique_consecutive(t, return_inverse=False, return_counts=False, dim=None,
                       **kw):
    """**이어진** 중복만 줄인다. 길이가 값에 달려 borch.ts 쪽이 비동기다 —
    `settle` 이 그 약속을 여기서 기다린다."""
    got = guarded(handle(t).uniqueConsecutive, return_inverse, return_counts,
                  dim)
    return tuple(got) if isinstance(got, list) else got


def tensor_split(t, indices_or_sections, dim=0, **kw):
    return tuple(guarded(handle(t).tensorSplit, _arg(indices_or_sections), dim))


def split_with_sizes(t, split_sizes, dim=0, **kw):
    return tuple(guarded(handle(t).splitWithSizes, _js_list(split_sizes), dim))


def tril_indices(row, col, offset=0, **kw):
    """**`(2, 개수)` 짜리 표다** — 자리 쌍이 아니라 행 줄과 열 줄로 나뉘어 온다."""
    return wrap(_ts.Tensor.trilIndices(row, col, offset))


def triu_indices(row, col, offset=0, **kw):
    return wrap(_ts.Tensor.triuIndices(row, col, offset))


def vander(x, N=None, increasing=False, **kw):
    return wrap(_ts.Tensor.vander(handle(x), N, increasing))


def cartesian_prod(*tensors, **kw):
    return wrap(_ts.Tensor.cartesianProd(*[handle(t) for t in tensors]))


def combinations(t, r=2, with_replacement=False, **kw):
    return wrap(_ts.Tensor.combinations(handle(t), r, with_replacement))


def chain_matmul(*matrices, **kw):
    mats = (list(matrices[0]) if len(matrices) == 1
            and isinstance(matrices[0], (list, tuple)) else list(matrices))
    return wrap(_ts.Tensor.chainMatmul(*[handle(m) for m in mats]))


# ── 최상위 선형대수 ─────────────────────────────────────────────────────────
#
# 손으로 적는 것은 **`linalg` 쪽과 이름이 겹치는 둘**이다. `camel` 이 `lu` 를
# `lu` 로, `lu_solve` 를 `luSolve` 로 넘기는데 borch.ts 의 그 이름들은 `linalg` 쪽
# 것이라 다른 답을 낸다 — 최상위는 `luTop`·`luSolveTop` 이다.

def lu(a, pivot=True, get_infos=False, **kw):
    """`(LU, pivots)`. **`linalg.lu` 와 다른 것을 낸다** — 그쪽은 `P·L·U` 셋으로
    펴 주고 이쪽은 겹쳐 담은 한 판과 교환 목록이다(실측)."""
    got = guarded(handle(a).luTop, pivot, get_infos)
    return tuple(got) if get_infos else (got.LU, got.pivots)


def lu_solve(b, lu_data, lu_pivots, **kw):
    """**`linalg.lu_solve` 와 인자 순서가 뒤집혀 있다** — 이쪽은 `b` 가 먼저다."""
    return wrap(guarded(handle(b).luSolveTop, handle(lu_data),
                        handle(lu_pivots)))


def lu_unpack(lu_data, lu_pivots, unpack_data=True, unpack_pivots=True, **kw):
    """**끄면 `None` 이 아니라 빈 텐서가 온다**(실측: 모양이 `(0,)` 이다)."""
    got = guarded(handle(lu_data).luUnpack, handle(lu_pivots), unpack_data,
                  unpack_pivots)
    return (got.P, got.L, got.U)


# ── 통계 ────────────────────────────────────────────────────────────────────
#
# 손으로 적는 것은 **난수 넷**과 **거절 셋**, 그리고 조립인 `trapz` 다. 나머지는
# `__getattr__` 이 첫 인자의 메서드로 넘긴다.
#
# **난수의 값은 골든이 못 굳힌다** — borch.ts 의 난수 줄기와 torch 의 것이 다르다.
# 그런데 끝값은 결정적이다(`std=0`·`p=0`·`p=1`), 그 자리를 골든이 묻는다.

def trapz(y, x=None, dx=1.0, dim=-1, **kw):
    """`trapezoid` 의 옛 이름. 같은 것이다(실측)."""
    return trapezoid(y, x, dx, dim, **kw)


# ── 복소수, 그리고 생성 몇 ─────────────────────────────────────────────────
#
# 한동안 이 자리는 **"복소수가 없어도 답이 있는 이름들"** 이었다 — 실수에서 `conj`
# 계열이 항등이라 `_alias` 하나로 다 됐고, `is_complex` 는 `return False` 로 못
# 박혀 있었다. borch.ts 에 복소수가 생기면서 그 답들이 **틀린 답**이 됐다.
#
# **"지금 통과하는 항등" 이 범위가 넓어질 때 제일 먼저 무너진다** — 코어에서
# `conj_physical` 이 정확히 그렇게 무너졌고, 여기는 그 이름이 여섯 개였다.

def _alias(t, **kw):
    """항등. **형과 그래프를 지키려면 `to(자기 형)` 이 가장 짧다** — borch.ts 의
    `to` 는 형이 같으면 자기 자신을 돌려주므로 커널도 안 돈다."""
    t = wrap(t)
    return t


def _is_cplx(t):
    """이 손잡이가 복소수인가. **`str()` 을 거친다** — `dtype` 은 JS 문자열이다."""
    return str(handle(t).dtype) == "complex64"


def complex(re, im, **kw):
    """실수부와 허수부를 엮는다.

    **이 이름이 파이썬 내장 `complex` 를 가린다.** 이 파일 안에서 복소수 판정에
    `_is_cplx` 를 쓰는 이유가 그것이다 — 코어(`borch/_ops.py`)가 같은 자리에서
    같은 선택을 했고, 셋째로 이름을 가리는 내장이다(`abs`·`bool`·`max`·`range`).
    """
    return wrap(_ts.Tensor.complex(handle(re), handle(im)))


def polar(abs, angle, **kw):                                    # noqa: A002
    """크기와 편각으로. 인자 이름이 torch 의 것이라 내장 `abs` 를 가린다."""
    return wrap(_ts.Tensor.polar(handle(abs), handle(angle)))


def view_as_real(t, **kw):
    """복소수를 실수 짝으로. **뷰다** — borch.ts 의 저장이 인터리브라서 그렇다."""
    return wrap(guarded(handle(t).viewAsReal))


def view_as_complex(t, **kw):
    return wrap(guarded(handle(t).viewAsComplex))


def real(t, **kw):
    """실수부. **실수 텐서에서는 자기 자신**이고 형도 그대로다(실측)."""
    return wrap(guarded(handle(t).real)) if _is_cplx(t) else _alias(t)


def imag(t, **kw):
    """허수부. **실수 텐서에서는 torch 자신이 거절한다**(실측) — 우리 한계가 아니다."""
    if not _is_cplx(t):
        raise RuntimeError(
            "imag is not implemented for tensors with non-complex dtypes.")
    return wrap(guarded(handle(t).imag))


def conj(t, **kw):
    """켤레. 실수에서는 항등이다.

    **torch 와 갈린다.** torch 의 `conj` 는 게을러서 켤레 비트만 세우는데 우리는
    값을 바로 뒤집는다 — 그래서 아래 `is_conj` 가 언제나 거짓이다. 값은 같다.
    """
    return wrap(guarded(handle(t).conjPhysical)) if _is_cplx(t) else _alias(t)


def conj_physical(t, **kw):
    return conj(t)


def conj_physical_(t, **kw):
    x = wrap(t)
    return x._write_back(conj(x)) if _is_cplx(x) else _alias(x)


def resolve_conj(t, **kw):
    """켤레 표시를 굳힌다. **우리에게는 그 표시가 아예 없어서** 언제나 항등이다."""
    return _alias(t)


def resolve_neg(t, **kw):
    return _alias(t)


def angle(t, **kw):
    """편각. 복소수는 `atan2(허수, 실수)`, **실수는 음수에 π** 이고 형은 언제나 실수다."""
    if _is_cplx(t):
        return wrap(guarded(handle(t).angle))
    import math

    x = wrap(t)
    below = x.binary("lt", full([], 0.0))
    return wrap(guarded(handle(below).to, "float32")).mul(full([], math.pi))


def is_complex(t, **kw):
    return _is_cplx(t)


def is_conj(t, **kw):
    return False


def is_neg(t, **kw):
    return False


def asarray(obj, dtype=None, copy=None, **kw):
    """**텐서를 주면 사본이 아니다**(실측). `copy=True` 여야 사본이다."""
    from ._base import tensor as _t

    if isinstance(obj, Tensor) and dtype is None and not copy:
        return obj
    if isinstance(obj, Tensor):
        got = obj.to(dtype) if dtype is not None else obj
        return _t(got.numpy().copy()) if copy else got
    # **`copy=False` 를 numpy 에 그대로 넘기면 안 된다.** numpy 2 에서 그것은
    # "베끼지 마라" 는 명령이라 목록처럼 베낄 수밖에 없는 것에서 던진다 — 기본값
    # `None`("되면 안 베낀다")과 다르다. 여기서는 참일 때만 말한다.
    arr = _np.array(obj, copy=True) if copy else _np.asarray(obj)
    return _t(arr, dtype)


def frombuffer(buffer, dtype=None, count=-1, offset=0, **kw):
    """바이트를 그대로 읽는다. **`offset` 은 바이트 수다** — 원소 수가 아니다(실측)."""
    from ._base import _DType, tensor as _t

    name = dtype.plain if isinstance(dtype, _DType) else "float32"
    kind = _np.dtype(name.replace("torch.", ""))
    return _t(_np.frombuffer(buffer, dtype=kind, count=count,
                             offset=offset).copy())


def range_top(start, end=None, step=1, **kw):
    """**끝을 포함한다** — `arange` 는 뺀다(실측). 조용히 `arange` 로 넘기면
    원소가 하나 모자란다."""
    from ._base import tensor as _t

    if end is None:
        start, end = 0, start
    return _t(_np.arange(start, end + step / 2.0, step, dtype=_np.float32))


def empty_strided(size, stride, **kw):
    """**걸음을 표현할 수 없어서 없다.** `as_strided` 와 다른 자리다 — 그쪽은 값이
    답이라 사본으로도 같은 답을 내는데, 이쪽은 **걸음 자체가 유일한 답**이다."""
    raise RuntimeError(
        "torch.empty_strided — 걸음(stride)이라는 것이 없습니다.")


def empty_permuted(size, physical_layout, **kw):
    raise RuntimeError(
        "torch.empty_permuted — 걸음(stride)이라는 것이 없습니다.")


def histogramdd(t, bins=10, **kw):
    """축이 여럿인 히스토그램.

    **경계가 텐서 목록으로 온다.** borch.ts 가 JS 배열로 주는데 `settle` 은 그 안까지
    안 들어가서, 그대로 두면 파이썬 쪽에 JS 손잡이가 남는다 — 받는 쪽이 `.shape` 도
    `._h` 도 못 쓴다. 여기서 하나씩 감싼다.
    """
    from ._base import _Fields

    got = guarded(handle(t).histogramdd, _arg(bins))
    out = _Fields.__new__(_Fields)
    object.__setattr__(out, "_order", ["hist", "bin_edges"])
    object.__setattr__(out, "_d", {
        "hist": got.hist,
        "bin_edges": [wrap(e) for e in got.bin_edges],
    })
    return out


def normal(mean=0.0, std=1.0, size=None, **kw):
    """정규분포 표본. **`std` 가 0 이면 평균 그대로다.**"""
    from ._base import tensor as _t

    if isinstance(mean, Tensor) or isinstance(std, Tensor):
        m = _np.asarray(wrap(mean).numpy(), dtype=_np.float64)
        s = _np.asarray(wrap(std).numpy(), dtype=_np.float64)
        m, s = _np.broadcast_arrays(m, s)
        return _t(_rng.normal(m, s).astype(_np.float32))
    shape = () if size is None else tuple(size)
    return _t(_rng.normal(float(mean), float(std), shape).astype(_np.float32))


def bernoulli(t, **kw):
    """자리마다 그 확률로 1. **0 이면 전부 0, 1 이면 전부 1.**"""
    from ._base import tensor as _t

    p = _np.asarray(wrap(t).numpy(), dtype=_np.float64)
    return _t((_rng.random(p.shape) < p).astype(_np.float32))


def poisson(t, **kw):
    from ._base import tensor as _t

    lam = _np.asarray(wrap(t).numpy(), dtype=_np.float64)
    return _t(_rng.poisson(lam).astype(_np.float32))


def binomial(count, prob, **kw):
    from ._base import tensor as _t

    n = _np.asarray(wrap(count).numpy(), dtype=_np.float64)
    p = _np.asarray(wrap(prob).numpy(), dtype=_np.float64)
    n, p = _np.broadcast_arrays(n, p)
    return _t(_rng.binomial(n.astype(_np.int64), p).astype(_np.float32))


# ── 짧은 시간 변환 ────────────────────────────────────────────────────────
#
# **오래 거절이었다.** 거절문에 "복소수 규약(Wirtinger)을 안 재서" 라고 적혀 있었고
# 그 이유가 맞았다 — 저장이 아니라 규약이 막고 있었다. 재서 못 박고 나니 열렸다.

def _stft_options(hop_length, win_length, window, center, pad_mode,
                  normalized, onesided, return_complex, length=None):
    """borch.ts 의 `StftOptions` 로. **없는 것은 안 넣는다** — `undefined` 와
    `null` 이 저쪽에서 다른 뜻이라(기본값 대 "명시적으로 없음"), 파이썬의 `None` 을
    그대로 넘기면 `return_complex` 강제 검사가 안 걸린다."""
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
    if onesided is not None:
        kw["onesided"] = bool(onesided)
    if return_complex is not None:
        kw["returnComplex"] = bool(return_complex)
    if length is not None:
        kw["length"] = int(length)
    return _js_options(**kw)


def stft(input, n_fft, hop_length=None, win_length=None, window=None,
         center=True, pad_mode="reflect", normalized=False, onesided=None,
         return_complex=None, **kw):
    """짧은 시간 푸리에 변환. **`return_complex` 를 안 주면 거절한다**(실측)."""
    return wrap(guarded(
        _ts.stft, handle(input), int(n_fft),
        _stft_options(hop_length, win_length, window, center, pad_mode,
                      normalized, onesided, return_complex)))


def istft(input, n_fft, hop_length=None, win_length=None, window=None,
          center=True, normalized=False, onesided=None, length=None,
          return_complex=False, **kw):
    return wrap(guarded(
        _ts.istft, handle(input), int(n_fft),
        _stft_options(hop_length, win_length, window, center, None,
                      normalized, onesided, None, length)))


# **`torch.fft` 는 이름 공간이다.** borch.ts 쪽도 모듈이라 그대로 넘기면 되는데,
# 파이썬의 `None` 기본값과 이름 붙은 인자를 자리로 푸는 일이 남는다.
class _Fft:
    @staticmethod
    def fft(input, n=None, dim=-1, norm=None, **kw):
        return wrap(guarded(_ts.fft.fft, handle(input), n, int(dim), norm))

    @staticmethod
    def ifft(input, n=None, dim=-1, norm=None, **kw):
        return wrap(guarded(_ts.fft.ifft, handle(input), n, int(dim), norm))

    @staticmethod
    def rfft(input, n=None, dim=-1, norm=None, **kw):
        return wrap(guarded(_ts.fft.rfft, handle(input), n, int(dim), norm))

    @staticmethod
    def irfft(input, n=None, dim=-1, norm=None, **kw):
        return wrap(guarded(_ts.fft.irfft, handle(input), n, int(dim), norm))

    @staticmethod
    def fftfreq(n, d=1.0, **kw):
        return wrap(guarded(_ts.fft.fftfreq, int(n), float(d)))

    @staticmethod
    def rfftfreq(n, d=1.0, **kw):
        return wrap(guarded(_ts.fft.rfftfreq, int(n), float(d)))

    @staticmethod
    def fftshift(input, dim=None, **kw):
        return wrap(guarded(_ts.fft.fftshift, handle(input),
                            None if dim is None else _dim_arg(dim)))

    @staticmethod
    def ifftshift(input, dim=None, **kw):
        return wrap(guarded(_ts.fft.ifftshift, handle(input),
                            None if dim is None else _dim_arg(dim)))


def _dim_arg(dim):
    """축 하나면 수로, 여럿이면 JS 배열로. 파이썬 목록을 그냥 넘기면 저쪽이 배열로
    안 본다 — `_js_list` 가 그 자리를 위해 있다."""
    return _js_list(dim) if isinstance(dim, (list, tuple)) else int(dim)


fft = _Fft()


# ── 최상위 순환 여덟 ───────────────────────────────────────────────────────
#
# **`__getattr__` 이 못 넘긴다.** 그쪽은 첫 인자의 메서드로 보내는데, 이 여덟은
# borch.ts 에서도 자유 함수다(가중치를 목록으로 받는다). 손으로 적는다.

def _rnn_options(has_biases, num_layers, dropout, train, bidirectional,
                 batch_first):
    return _js_options(hasBiases=bool(has_biases), numLayers=int(num_layers),
                       dropout=float(dropout), train=bool(train),
                       bidirectional=bool(bidirectional),
                       batchFirst=bool(batch_first))


def _rnn_params(params):
    return _js.Array.from_([handle(p) for p in params])


def lstm(input, hx, params, has_biases, num_layers, dropout, train,     # noqa: A002
         bidirectional, batch_first=False, **kw):
    """`(출력, h_n, c_n)` — **셋을 편다.** 층 쪽은 `(출력, (h, c))` 로 묶는다."""
    got = guarded(_ts.lstm, handle(input),
                  _js.Array.from_([handle(hx[0]), handle(hx[1])]),
                  _rnn_params(params),
                  _rnn_options(has_biases, num_layers, dropout, train,
                               bidirectional, batch_first))
    return tuple(wrap(t) for t in got)


def _rnn_two(name):
    def call(input, hx, params, has_biases, num_layers, dropout, train,  # noqa: A002
              bidirectional, batch_first=False, **kw):
        got = guarded(getattr(_ts, name), handle(input), handle(hx),
                      _rnn_params(params),
                      _rnn_options(has_biases, num_layers, dropout, train,
                                   bidirectional, batch_first))
        return tuple(wrap(t) for t in got)

    return call


gru = _rnn_two("gru")
rnn_tanh = _rnn_two("rnnTanh")
rnn_relu = _rnn_two("rnnRelu")


def lstm_cell(input, hx, w_ih, w_hh, b_ih=None, b_hh=None, **kw):       # noqa: A002
    got = guarded(_ts.lstmCell, handle(input),
                  _js.Array.from_([handle(hx[0]), handle(hx[1])]),
                  handle(w_ih), handle(w_hh),
                  None if b_ih is None else handle(b_ih),
                  None if b_hh is None else handle(b_hh))
    return tuple(wrap(t) for t in got)


def _cell_one(name):
    def call(input, hx, w_ih, w_hh, b_ih=None, b_hh=None, **kw):        # noqa: A002
        return wrap(guarded(getattr(_ts, name), handle(input), handle(hx),
                            handle(w_ih), handle(w_hh),
                            None if b_ih is None else handle(b_ih),
                            None if b_hh is None else handle(b_hh)))

    return call


gru_cell = _cell_one("gruCell")
rnn_tanh_cell = _cell_one("rnnTanhCell")
rnn_relu_cell = _cell_one("rnnReluCell")


def hash_tensor(*args, **kw):
    """**uint64 도 없고 규격도 없다.** 값을 맞출 수 없는 것에 이름만 놓지 않는다."""
    raise RuntimeError(
        "torch.hash_tensor — uint64 도, 정해진 해시 규격도 없습니다.")


def sspaddmm(input, mat1, mat2, beta=1, alpha=1, **kw):
    """**희소 텐서 전용이라 없다.** 코어와 같은 자리, 같은 이유로 거절한다 —
    조밀 텐서로 흉내 내면 모양은 맞고 저장 방식이 다른 것을 주게 된다."""
    raise RuntimeError(
        "torch.sspaddmm — 희소(sparse) 텐서 배치가 없습니다. "
        "자기 컴퓨터에서 진짜 PyTorch 를 쓰세요.")


def fill(x, value, **kw):
    """**제자리가 아니다.** `fill_` 과 이름이 한 글자 다르고 하는 일이 다르다 —
    이쪽은 새 텐서를 내고 원본은 그대로다(실측).

    별칭 표로 넘기면 `fill_` 까지 같은 이름으로 끌려가므로 여기 손으로 적는다.
    """
    return wrap(guarded(handle(x).fillWith, float(value)))


def bitwise_not(x, **kw):
    """**참거짓이면 논리 부정이다.** 정수면 `~x` 라 `~1 == -2` 인데, 참에 그것을
    적용하면 torch 는 거짓을 준다(실측) — 두 갈래가 값에서 아예 다르다.

    이항 쪽(`and`·`or`·`xor`)은 갈래가 필요 없다. 0/1 에서 비트 셈과 논리 셈이 같은
    답을 내고, 형도 `bool` 끼리면 `bool` 로 남는다. 부정만 다르다.
    """
    x = wrap(x)
    return _unary(x, "logical_not" if str(handle(x).dtype) == "bool"
                  else "bitwise_not")


def var_mean(t, dim=None, keepdim=False, **kw):
    """**둘을 한 번에 준다.** 하나만 물으면 다른 하나가 틀려도 안 걸린다."""
    t = wrap(t)
    return (t.var(dim=dim, keepdim=keepdim), t.mean(dim=dim, keepdim=keepdim))


def std_mean(t, dim=None, keepdim=False, **kw):
    t = wrap(t)
    return (t.std(dim=dim, keepdim=keepdim), t.mean(dim=dim, keepdim=keepdim))


def inner(a, b, **kw):
    a, b = wrap(a), wrap(b)
    if len(_shape_list(a)) > 1:
        return a @ transpose(b, -2, -1)
    return (a * b).sum()


def vdot(a, b, **kw):
    return (wrap(a) * wrap(b)).sum()


def kron(a, b, **kw):
    a, b = wrap(a), wrap(b)
    n, m = _shape_list(a)[0], _shape_list(b)[0]
    out = (wrap(guarded(handle(a).reshape, _js_list([n, 1])))
           * wrap(guarded(handle(b).reshape, _js_list([1, m]))))
    return wrap(guarded(handle(out).reshape, _js_list([n * m])))


def cross(a, b, dim=-1, **kw):
    a, b = wrap(a), wrap(b)
    rank = len(_shape_list(a))
    axis = dim + rank if dim < 0 else dim

    def part(t, i):
        return wrap(guarded(handle(t).narrow, axis, i, 1))

    return cat([part(a, 1) * part(b, 2) - part(a, 2) * part(b, 1),
                part(a, 2) * part(b, 0) - part(a, 0) * part(b, 2),
                part(a, 0) * part(b, 1) - part(a, 1) * part(b, 0)], axis)


# ── 수치 계열. **급수로 세는 셋은 WGSL 에 있고 나머지는 조합이다.** ─────────

def cdist(a, b, p=2.0, **kw):
    """모든 짝 사이의 거리. 브로드캐스팅 하나로 풀린다."""
    a, b = wrap(a), wrap(b)
    n, k = _shape_list(a)
    m = _shape_list(b)[0]
    diff = (wrap(guarded(handle(a).reshape, _js_list([n, 1, k])))
            - wrap(guarded(handle(b).reshape, _js_list([1, m, k]))))
    if p == 2.0:
        return (diff * diff).sum(dim=2).sqrt()
    return ((_unary(diff, "abs") ** p).sum(dim=2)) ** (1.0 / p)


def cov(t, correction=1, **kw):
    """공분산. **줄이 변수이고 칸이 관측이다** — numpy 와 축이 반대라 헷갈린다."""
    t = wrap(t)
    shape = _shape_list(t)
    if len(shape) == 1:
        t = wrap(guarded(handle(t).reshape, _js_list([1, shape[0]])))
        shape = [1, shape[0]]
    n = shape[1]
    centered = t - t.mean(dim=1, keepdim=True)
    return (centered @ transpose(centered, 0, 1)) * (1.0 / builtins.max(1, n - correction))


# ── torch 최상위에만 있는 이름들 ────────────────────────────────────────────
#
# 최상위 쪽은 날 ATen 이라 **인자 순서가 다르고 열거형이 정수다.** 같은 계산인데
# 부르는 법이 다른 것이라, 계산은 `nn.functional` 한 벌만 두고 여기서 자리만 옮긴다.

def _inplace_from(name, fn_name=None):
    def call(x, *args, **kw):
        from . import _nn
        x._refuse_inplace_on_leaf(name)
        got = getattr(_nn.functional, fn_name or name.rstrip("_"))(x, *args, **kw)
        return x._write_back(got)
    call.__name__ = name
    return call


def nan_to_num_(x, *args, **kw):
    """**`nan_to_num` 은 `F` 가 아니라 모듈 쪽에 있다.** 이름만 보고 `F` 에서 찾으면
    없다고 멈춘다 — 최상위 이름이 전부 `F` 에도 있는 것은 아니다."""
    x._refuse_inplace_on_leaf("nan_to_num_")
    return x._write_back(nan_to_num(x, *args, **kw))


dropout_ = _inplace_from("dropout_")
alpha_dropout_ = _inplace_from("alpha_dropout_")
feature_alpha_dropout_ = _inplace_from("feature_alpha_dropout_")
feature_dropout_ = _inplace_from("feature_dropout_", "dropout2d")


def feature_dropout(x, p=0.5, train=True):
    """**채널째 떨군다** — `F.dropout2d` 와 같은 계산이다(실측)."""
    from . import _nn
    return _nn.functional.dropout2d(x, p, train)


def batch_norm(x, weight, bias, running_mean, running_var, training=False,
               momentum=0.1, eps=1e-5, cudnn_enabled=False):
    """**`F.batch_norm` 과 인자 순서가 다르다** — 여기서는 가중치가 통계보다 앞이다."""
    from . import _nn
    return _nn.functional.batch_norm(x, running_mean, running_var, weight, bias,
                                     training, momentum, eps)


def grid_sampler(x, grid, interpolation_mode=0, padding_mode=0,
                 align_corners=False):
    """**열거형이 정수다.** 0·1 이 `bilinear`·`nearest`, 채우기는 0·1·2 다."""
    from . import _nn
    modes = ("bilinear", "nearest", "bicubic")
    pads = ("zeros", "border", "reflection")
    return _nn.functional.grid_sample(x, grid, modes[int(interpolation_mode)],
                                      pads[int(padding_mode)], align_corners)


def max_pool1d_with_indices(x, kernel_size, stride=None, padding=0, dilation=1,
                            ceil_mode=False, **kw):
    from . import _nn
    if padding or dilation != 1 or ceil_mode:
        raise RuntimeError(
            "max_pool1d_with_indices(padding·dilation·ceil_mode) 은 아직 없다.")
    return _nn.functional.max_pool1d_with_indices(x, kernel_size, stride)


def ctc_loss(log_probs, targets, input_lengths, target_lengths, blank=0,
             reduction=1, zero_infinity=False):
    """**`reduction` 이 정수다** — 0·1·2 가 `none`·`mean`·`sum` 이다."""
    from . import _nn
    kinds = ("none", "mean", "sum")
    return _nn.functional.ctc_loss(log_probs, targets, input_lengths,
                                   target_lengths, blank, kinds[int(reduction)],
                                   zero_infinity)


def geqrf(t, **kw):
    """반사자 꼴 QR. `linalg.householder_product` 의 짝이라 최상위에도 있다."""
    return guarded(handle(t).geqrf)


def corrcoef(t, **kw):
    """공분산을 표준편차로 나눈 것. **대각선이 1 이 된다** — 그것이 검산이다."""
    c = cov(t)
    n = _shape_list(c)[0]
    diag = wrap(guarded(handle(c).diagonal))
    scale = (wrap(guarded(handle(diag).reshape, _js_list([n, 1])))
             * wrap(guarded(handle(diag).reshape, _js_list([1, n]))))
    return c / _unary(scale, "sqrt")


def tensordot(a, b, dims=2, **kw):
    """지정한 축끼리 접어 곱한다. 접을 축을 몰고 행렬곱 한 번으로 끝낸다."""
    a, b = wrap(a), wrap(b)
    ash, bsh = _shape_list(a), _shape_list(b)
    if isinstance(dims, int):
        left = list(range(len(ash) - dims, len(ash)))
        right = list(range(dims))
    else:
        left, right = [list(v) for v in dims]
    a_keep = [i for i in range(len(ash)) if i not in left]
    b_keep = [i for i in range(len(bsh)) if i not in right]
    a_shape = [ash[i] for i in a_keep]
    b_shape = [bsh[i] for i in b_keep]
    inner = 1
    for i in left:
        inner *= ash[i]
    rows = 1
    for v in a_shape:
        rows *= v
    cols = 1
    for v in b_shape:
        cols *= v
    am = wrap(guarded(handle(wrap(guarded(handle(a).permute,
                                          _js_list(a_keep + left)))).reshape,
                      _js_list([rows, inner])))
    bm = wrap(guarded(handle(wrap(guarded(handle(b).permute,
                                          _js_list(right + b_keep)))).reshape,
                      _js_list([inner, cols])))
    return wrap(guarded(handle(am @ bm).reshape, _js_list(a_shape + b_shape)))


def _trapezoid_pieces(y, x, dx, dim):
    y = wrap(y)
    rank = len(_shape_list(y))
    axis = dim + rank if dim < 0 else dim
    n = _shape_list(y)[axis]
    left = wrap(guarded(handle(y).narrow, axis, 0, n - 1))
    right = wrap(guarded(handle(y).narrow, axis, 1, n - 1))
    if x is None:
        return (left + right) * (dx / 2.0), axis
    x = wrap(x)
    step = (wrap(guarded(handle(x).narrow, axis, 1, n - 1))
            - wrap(guarded(handle(x).narrow, axis, 0, n - 1)))
    return (left + right) * step * 0.5, axis


def trapezoid(y, x=None, dx=1.0, dim=-1, **kw):
    """사다리꼴 적분. 이웃한 두 점의 평균에 간격을 곱해 더한다."""
    pieces, axis = _trapezoid_pieces(y, x, kw.get("dx", dx), dim)
    return pieces.sum(dim=axis)


def cumulative_trapezoid(y, x=None, dx=1.0, dim=-1, **kw):
    """누적판. **마지막 값이 `trapezoid` 와 같아야 한다** — 그것이 검산이다."""
    pieces, axis = _trapezoid_pieces(y, x, kw.get("dx", dx), dim)
    return wrap(guarded(handle(pieces).cumsum, axis))


# ── 색인으로 **쓰는** 쪽. 읽는 쪽(`gather`)의 반대다. ───────────────────────

def _spread_index(index, dim, shape):
    """1 차원 번호를 `shape` 모양으로 편다.

    `index_add` 류는 번호가 **줄**을 가리키는데 커널은 칸마다의 번호를 받는다.
    축 하나에 놓고 나머지로 늘리면 그 둘이 같은 것이 된다 — 새 커널이 필요 없다.
    """
    lifted = [1] * len(shape)
    lifted[dim] = int(handle(index).size)
    return broadcast_to(wrap(guarded(handle(index).reshape, _js_list(lifted))), shape)


def scatter(t, dim, index, src, **kw):
    """번호가 가리키는 자리에 **덮어쓴다.** 겹치면 마지막에 쓴 것이 남는다."""
    t = wrap(t)
    if not isinstance(src, Tensor):
        src = zeros(*[int(n) for n in handle(index).shape]) + float(src)
    return wrap(guarded(handle(t).scatterSet, dim, handle(index), handle(src)))


def scatter_add(t, dim, index, src, **kw):
    """번호가 가리키는 자리에 **더한다.** 겹치면 쌓인다 — `scatter` 와 여기서 갈린다."""
    return wrap(guarded(handle(t).scatterAdd, dim, handle(index), handle(src)))


def index_add(t, dim, index, source, alpha=1, **kw):
    t, source = wrap(t), wrap(source)
    shape = [int(n) for n in handle(source).shape]
    spread = _spread_index(index, dim, shape)
    return scatter_add(t, dim, spread, source if alpha == 1 else source * alpha)


def index_copy(t, dim, index, source, **kw):
    t, source = wrap(t), wrap(source)
    shape = [int(n) for n in handle(source).shape]
    return scatter(t, dim, _spread_index(index, dim, shape), source)


def index_fill(t, dim, index, value, **kw):
    t = wrap(t)
    shape = [int(n) for n in handle(t).shape]
    shape[dim] = int(handle(index).size)
    return scatter(t, dim, _spread_index(index, dim, shape),
                   zeros(*shape) + float(value))


def take(t, index, **kw):
    """**평평하게 펴서** 뽑는다 — 축이라는 개념이 없다."""
    h = handle(t)
    flat = wrap(guarded(h.reshape, _js_list([int(h.size)])))
    picked = wrap(guarded(handle(flat).indexSelect, 0,
                          handle(wrap(index).reshape(int(handle(index).size)))))
    return wrap(guarded(handle(picked).reshape,
                        _js_list([int(n) for n in handle(index).shape])))


def take_along_dim(t, indices, dim=None, **kw):
    if dim is None:
        return take(t, indices)
    return wrap(guarded(handle(t).gather, dim, handle(indices)))


def searchsorted(sorted_sequence, values, right=False, **kw):
    """정렬된 것 안에서 들어갈 자리. **`right` 가 동점의 어느 쪽인지 정한다.**

    커널이 필요 없다 — "나보다 작은 것이 몇 개인가" 를 세면 그것이 자리다.
    """
    right = kw.get("right", right)
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
    """`searchsorted` 와 **인자 순서가 뒤집혀 있다.** 그것이 두 이름의 차이 전부다."""
    return searchsorted(boundaries, values, right=kw.get("right", right))


row_stack = vstack
multiply = mul
divide = div
subtract = sub
true_divide = div
concat = cat
concatenate = cat

# 비교의 다른 이름들. 표에 있는 이름으로 넘긴다.
_COMPARE_ALIAS = {"greater": "gt", "greater_equal": "ge",
                  "less": "lt", "less_equal": "le", "not_equal": "ne"}


def where(cond, a, b):
    """torch 는 `where(조건, 참, 거짓)`, borch.ts 는 `참.where(조건, 거짓)` 이다.
    자리를 바꾸지 않으면 참·거짓이 뒤집힌 값이 나온다 — 값 대조로만 보인다."""
    return guarded(handle(a).where, handle(cond), handle(b))


def layer_norm(x, shape=None, weight=None, bias=None, eps=1e-5, **kw):
    """torch 는 **정규화할 모양**을 받고 borch.ts 는 축을 받는다.

    `(마지막 축의 길이,)` 처럼 뒤에서부터 세는 것이 torch 의 규칙이므로, 받은 모양의
    길이만큼 뒤에서 센 축이 시작점이다. 그대로 넘기면 축 4 를 랭크 2 에 물어보게 된다.
    """
    dim = -len(shape) if isinstance(shape, (list, tuple)) and shape else -1
    return guarded(handle(x).layerNorm, dim, kw.get("eps", eps))


def repeat_interleave(x, repeats, dim=None, **kw):
    """`dim` 이 없으면 torch 는 **평평하게 편 뒤** 되풀이한다."""
    h = handle(x)
    if dim is None:
        h = h.reshape(_js_list([int(h.size)]))
        dim = 0
    return guarded(h.repeatInterleave, repeats, dim)


def flip(x, dims=None, **kw):
    """torch 는 축 **목록**을 받고 borch.ts 는 하나씩 받는다. 차례로 뒤집는다."""
    dims = kw.get("dims", dims)
    if isinstance(dims, int):
        dims = [dims]
    out = handle(x)
    for d in (dims or []):
        out = out.flip(d)
    return wrap(out)


def pow(x, exponent):                                    # noqa: A001
    """지수가 수면 `powScalar` 다 — 정수 지수를 곱셈으로 풀어 부호를 지킨다."""
    if isinstance(exponent, Tensor):
        return guarded(handle(x).binary, "pow", exponent._h)
    return guarded(handle(x).powScalar, exponent)


def pad(x, pairs, mode="constant", value=0.0, **kw):
    """torch 의 `F.pad` 는 **마지막 축부터** 짝을 받는다 — `(왼, 오, 위, 아래, …)`.

    **`mode` 를 받아만 놓고 안 쓰던 자리다.** 인자에는 있는데 아래로 안 내려가서
    `reflect` 를 달라고 해도 상수 패딩이 나왔다 — 예외가 아니라 **조용히 다른 값**이라,
    골든에 그 모드를 묻는 케이스가 생기고 나서야 드러났다. JS 가 남는 인자를 버리는
    것과 같은 종류인데 이번에는 파이썬 쪽에서 버렸다.
    """
    value = kw.get("value", value)
    return wrap(guarded(handle(x).padND, _js_list(list(pairs)), mode, float(value)))


def split(x, size, dim=0):
    """**인자 순서가 뒤집혀 있다.** torch 는 `split(조각크기, 축)`, borch.ts 는
    `splitSize(축, 조각크기)` 다. 그대로 넘기면 축 자리에 크기가 들어가 엉뚱한 데서
    터진다 — `축 2 의 크기 0 는 undefined 로 안 나뉜다` 가 그것이었다."""
    return [wrap(t) for t in handle(x).splitSize(dim, size)]


def chunk(x, chunks, dim=0):
    """**`split` 이 아니다.** 그쪽은 나눠떨어져야 하고 `chunk` 는 아니다 —
    torch 는 3 을 2 로 쪼개면 2·1 을 주고, 2 를 5 로 쪼개면 **조각이 둘**이다(실측).

    borch.ts 에 `chunk` 가 제대로 있는데 여기서 `split` 으로 넘기고 있었다. 나눠
    떨어지는 크기로만 재면 두 함수가 같아 보인다.
    """
    return [wrap(t) for t in handle(x).chunk(chunks, dim)]


def clamp(x, min=None, max=None):                        # noqa: A002
    """**한쪽만 주는 것이 흔하다.** borch.ts 는 `clamp(low, high)` 로 둘 다 받고,
    한쪽에 `undefined` 를 넘기면 그것이 셰이더 안까지 내려가 WGSL 이 거절한다.
    그래서 여기서 `clampMin`·`clampMax` 로 갈라 준다."""
    h = handle(x)
    if min is not None and max is not None:
        return guarded(h.clamp, min, max)
    if min is not None:
        return guarded(h.clampMin, min)
    if max is not None:
        return guarded(h.clampMax, max)
    return wrap(h)


clip = clamp


def aminmax(x, **kw):
    """최소와 최대를 함께. borch.ts 는 둘을 따로 갖고 있어서 여기서 묶는다."""
    h = handle(x)
    return _MinMax(wrap(h.amin()), wrap(h.amax()))


class _MinMax:
    """`aminmax` 의 답. torch 는 `.min` 과 `.max` 라고 부른다."""

    __slots__ = ("min", "max")

    def __init__(self, lo, hi):
        self.min, self.max = lo, hi

    def __iter__(self):
        yield self.min
        yield self.max

    def __getitem__(self, i):
        return (self.min, self.max)[i]


# **`torch.linalg` 는 이름 공간이다.** 대부분 텐서 메서드로 있고, 값에 따라 크기가
# 정해지는 것들(`cholesky`·`svd`·`eigh`)은 비동기라 `settle` 이 기다린다.
class _Linalg:
    # 특이행렬을 만나는 코드가 `except linalg.LinAlgError` 로 감싼다. 여기 없으면
    # 그 감싸기가 이름을 못 찾고 프로그램이 죽는다 — 값보다 먼저 필요한 것이다.
    LinAlgError = _LinAlgError

    def lstsq(self, a, b):
        """torch 는 `.solution` 이 든 물건을 준다 — borch.ts 는 답을 바로 준다."""
        from ._base import _Fields
        got = settle(handle(a).lstsq(handle(b)))
        out = _Fields.__new__(_Fields)
        object.__setattr__(out, "_order", ["solution"])
        object.__setattr__(out, "_d", {"solution": got})
        return out

    def matrix_power(self, a, n):
        return matrix_power(a, n)

    def multi_dot(self, mats):
        """**첫 인자가 텐서가 아니라 목록이다** — 아래의 일반 길로 못 간다.

        묶는 순서는 값을 안 바꾼다(곱셈이 결합적이라). 바뀌는 것은 셈의 개수뿐이라
        여기서는 순서대로 곱한다.
        """
        out = wrap(mats[0])
        for m in mats[1:]:
            out = wrap(guarded(handle(out).mm, handle(m)))
        return out

    def diagonal(self, a, offset=0, dim1=-2, dim2=-1):
        """**`torch.diagonal` 과 기본 축이 다르다.**

        이쪽은 마지막 두 축이고 저쪽은 앞의 두 축이다. 일반 길로 보내면 borch.ts 의
        기본값(`0, 1`)이 쓰여서 3 차원에서 조용히 다른 모양이 나온다.
        """
        return wrap(guarded(handle(a).diagonal, offset, dim1, dim2))

    def tensorsolve(self, a, b, dims=None):
        if dims is not None:
            raise RuntimeError("tensorsolve(dims) 는 아직 없다")
        return wrap(guarded(handle(a).tensorSolve, handle(b)))

    def tensorinv(self, a, ind=2):
        return wrap(guarded(handle(a).tensorInv, ind))

    # ── `_ex`·LDL·반사자 ────────────────────────────────────────────────
    #
    # 일반 길로 못 간다 — 인자가 텐서 여럿이거나(`ldl_solve`) 이름 붙은 자리를 여럿
    # 돌려준다. 자리 이름은 `guarded` 가 붙여 준다.

    def lu_factor_ex(self, a, pivot=True, check_errors=False):
        return guarded(handle(a).luFactorEx)

    def ldl_factor(self, a, hermitian=False):
        return guarded(handle(a).ldlFactor)

    def ldl_factor_ex(self, a, hermitian=False, check_errors=False):
        """`ldl_factor` 에 `info` 를 붙인 것. 여기서는 늘 0 이다 — 나쁜 자리는 거절한다."""
        from ._base import _Fields, tensor as _tensor
        got = guarded(handle(a).ldlFactor)
        zero = _tensor(_np.zeros((), dtype=_np.float32))
        out = _Fields.__new__(_Fields)
        object.__setattr__(out, "_order", ["LD", "pivots", "info"])
        object.__setattr__(out, "_d", {"LD": got.LD, "pivots": got.pivots,
                                       "info": zero})
        return out

    def ldl_solve(self, ld, pivots, b, hermitian=False):
        return wrap(guarded(handle(ld).ldlSolve, handle(b)))

    def householder_product(self, a, tau):
        return wrap(guarded(handle(a).householderProduct, handle(tau)))

    def __getattr__(self, name):
        # torch 가 줄여 부르는 것들. `pinv` 는 오래 비어 있었는데 골든이 늘 긴 이름
        # (`L.pinverse`)으로만 물어서 안 드러났다 — 부르는 철자가 하나 늘자 나왔다.
        js_name = camel({"inv": "inverse", "pinv": "pinverse",
                         "matmul": "mm", "matrix_rank": "matrixRank"}.get(name, name))

        def call(x, *args, **kw):
            fn = getattr(handle(x), js_name, None)
            if fn is None:
                raise AttributeError(f"borch.ts 에 `{js_name}` 이 없다 (linalg.{name})")
            # **이름 붙은 인자를 버리면 안 된다.** `qr(mode="complete")` 와
            # `svd(full_matrices=False)` 가 그 자리인데, 버리면 예외가 아니라
            # **기본값으로 조용히 다른 답**이 나온다 — 값 대조만이 잡을 수 있는 종류다.
            return guarded(fn, *positional(name, args, kw))

        call.__name__ = name
        return call


linalg = _Linalg()
