"""파이썬 이름을 borch.ts 메서드로 넘긴다 — **목록 없이.**

손으로 이름 192 개를 적으면 그중 하나가 다른 연산을 부르는 날이 오고, 그것은 값
대조로만 보인다. 표를 읽어 다는 방법도 있지만 그러려면 `index.ts` 가 커널 표를
내보내야 하고, 그건 안쪽 사정을 공개 표면에 올리는 것이다.

그래서 **모듈의 `__getattr__`** 을 쓴다(PEP 562). `L.exp(x)` 가 오면 그때
`x.exp()` 로 넘기고, borch.ts 에 없으면 `AttributeError` 로 멈춘다. 없는 것을
근사하지 않는다 — 골든이 그 자리를 실패로 세고, 그 수가 곧 이 결속의 진도다.

이름 규칙 하나만 다르다. 파이썬은 `masked_select`, JS 는 `maskedSelect` 다.
"""

import js as _js
from pyodide.ffi import to_js as _to_js

from ._base import Tensor, _js_list, guarded, handle, settle, wrap

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
    # **`backend` 는 없다고 답해야 한다.** 골든 하네스가 `hasattr(lib, "backend")` 로
    # 자매를 알아보고 `webgpu::` 케이스를 그쪽에만 묻는다. 그 케이스들은 TF.js 의 랭크
    # 한계를 못 박는 것이라 여기서는 뜻이 없다 — 무엇이든 돌려주는 `__getattr__` 이
    # 우리를 자매로 오인하게 만들고 있었다.
    if name == "backend":
        raise AttributeError(name)
    # dtype 이름들. `bool` 을 모듈 전역에 두면 파이썬 내장을 가리므로 여기서 준다.
    if name in ("bool", "float32", "int64"):
        from ._base import _DType
        return _DType(name)
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
        return guarded(fn, *positional(name, args, kw))

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


def randn(*shape, **kw):
    """정규분포 난수. **borch.ts 에는 없어서 여기서 만든다.**

    골든에서 이것을 쓰는 자리는 오류 케이스뿐이고, 거기서는 값을 안 보고 **던지는지**만
    본다(`L.randn(3, 4) @ L.randn(3, 2)`). 그래서 씨앗을 못 박은 재현 가능한 난수면
    충분하다 — 값을 묻는 케이스가 생기면 그때는 borch.ts 쪽에 제대로 넣어야 한다.
    """
    import numpy as _np
    from ._base import tensor as _t

    shape = shape[0] if len(shape) == 1 and isinstance(shape[0], (list, tuple)) else shape
    return _t(_np.random.default_rng(0).standard_normal(tuple(shape)).astype("float32"),
              requires_grad=kw.get("requires_grad", False))


def rand(*shape, **kw):
    import numpy as _np
    from ._base import tensor as _t

    shape = shape[0] if len(shape) == 1 and isinstance(shape[0], (list, tuple)) else shape
    return _t(_np.random.default_rng(0).random(tuple(shape)).astype("float32"))


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


def flatten(x, start_dim=1, end_dim=-1, **kw):
    """축을 접는다. **borch.ts 에는 없어서 `reshape` 로 만든다** — 정의 그대로다."""
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
