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

from ._base import Tensor, _js_list, _Size, guarded, handle, settle, wrap

_ts = _js.borch

# 두 언어에서 철자가 아예 다른 것들. 규칙으로 안 되는 것만 적는다.
_RENAME = {
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
    "swapdims": "transpose",
    "interpolate": "upsample",
    # 연산 표의 이름이 파이썬과 같은 것들. `camel` 을 씌우면 오히려 없는 이름이 된다.
    "logical_not": "logical_not",
    "logical_and": "logical_and",
    "logical_or": "logical_or",
    "logical_xor": "logical_xor",
    "matmul": "mm",
    "var": "variance",
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
    "sum": ("dim", "keepdim"),
    "mean": ("dim", "keepdim"),
    "prod": ("dim", "keepdim"),
    "amax": ("dim", "keepdim"),
    "amin": ("dim", "keepdim"),
    "var": ("dim", "keepdim"),
    "std": ("dim", "keepdim"),
    "logsumexp": ("dim", "keepdim"),
    "argmax": ("dim",),
    "argmin": ("dim",),
    "softmax": ("dim",),
    "log_softmax": ("dim",),
    "cumsum": ("dim",),
    "cumprod": ("dim",),
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
    "diff": ("n", "dim"),
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
    """
    if name in _RENAME:
        return _RENAME[name]
    tail = "_" if name.endswith("_") and not name.endswith("__") else ""
    head, *rest = name.rstrip("_").split("_")
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


def linspace(start, end, count, **kw):
    return wrap(_ts.Tensor.linspace(start, end, count))


# **난수는 한 흐름에서 나온다.** 처음에는 부를 때마다 `default_rng(0)` 을 새로
# 만들었다. 골든이 난수를 오류 케이스에서만 써서(던지는지만 본다) 값이 늘 같아도
# 안 걸렸는데, 그 상태로는 셔플하는 `DataLoader` 가 **매 에폭 같은 순서**를 낸다.
# 부르는 쪽에서 보면 셔플을 켰는데 안 섞이는 것이고, 아무 예외도 안 난다.
_rng = _np.random.default_rng(0)


def manual_seed(seed):
    """씨앗을 다시 심는다. torch 와 같은 이름·같은 뜻이다."""
    global _rng
    _rng = _np.random.default_rng(seed)
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
    """`max`·`min`. **축을 주면 짝을, 안 주면 스칼라 하나를 낸다** — torch 가 그렇다.

    두 꼴이 한 이름에 붙어 있는 것이 헷갈리는 자리인데, 그것이 torch 의 계약이므로
    여기서 정리하면 안 된다. 정리하면 교재 코드가 안 돈다.
    """
    def call(x, dim=None, keepdim=False, **kw):
        dim = kw.get("dim", dim)
        h = handle(x)
        if dim is None:
            h = h.reshape(_js_list([int(h.size)]))
            return guarded(getattr(h, name), 0)
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


def sum(x, dim=None, keepdim=False, **kw):               # noqa: A001
    """borch.ts 는 전체 합(`sum()`)과 축 합(`sumDim`)을 **다른 이름**으로 둔다.

    이 자리가 조용히 틀렸다. `sum(dim=1)` 이 `sum()` 으로 가서 축을 무시한 스칼라를
    냈고, 예외가 없으니 아무도 몰랐다 — 랭크 6 케이스 하나가 모양으로 걸릴 때까지.
    """
    dim = kw.get("dim", dim)
    keepdim = kw.get("keepdim", keepdim)
    h = handle(x)
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
    raise NotImplementedError(f"norm 의 p={p} 는 아직 없다 — 근사하지 않는다")


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

    borch.ts 의 `pad(축, 앞, 뒤, 값)` 은 축 하나씩이다. 짝을 뒤에서부터 풀어 차례로
    두른다 — 순서를 뒤집지 않으면 엉뚱한 축이 늘어난다.
    """
    value = kw.get("value", value)
    out = handle(x)
    rank = len(out.shape)
    for i in range(0, len(pairs), 2):
        axis = rank - 1 - (i // 2)
        out = out.pad(axis, pairs[i], pairs[i + 1], float(value))
    return wrap(out)


def split(x, size, dim=0):
    """**인자 순서가 뒤집혀 있다.** torch 는 `split(조각크기, 축)`, borch.ts 는
    `splitSize(축, 조각크기)` 다. 그대로 넘기면 축 자리에 크기가 들어가 엉뚱한 데서
    터진다 — `축 2 의 크기 0 는 undefined 로 안 나뉜다` 가 그것이었다."""
    return [wrap(t) for t in handle(x).splitSize(dim, size)]


def chunk(x, chunks, dim=0):
    return [wrap(t) for t in handle(x).split(dim, chunks)]


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

    def __getattr__(self, name):
        js_name = camel({"inv": "inverse", "matrix_rank": "matrixRank"}.get(name, name))

        def call(x, *args, **kw):
            fn = getattr(handle(x), js_name, None)
            if fn is None:
                raise AttributeError(f"borch.ts 에 `{js_name}` 이 없다 (linalg.{name})")
            return guarded(fn, *[_arg(a) for a in args])

        call.__name__ = name
        return call


linalg = _Linalg()
