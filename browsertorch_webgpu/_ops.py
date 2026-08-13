"""browsertorch_webgpu 를 쪼갠 조각. 공개 이름은 __init__ 이 모은다."""

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

from ._tensor import (
    Tensor, _align, _canonical, _storage_for, _wrap,
)
from ._base import (
    _ValuesIndices, _dtype_of, _last_axis_only, _pick_last, _reject_float64, _shape_of,
    _slice_along, _slice_tensor, _to_np, _to_tf, _unsupported, bool_, dtype, float32,
    int64,
)

# ---------------------------------------------------------------- 만들기

def tensor(data, dtype=None, requires_grad=False):
    _reject_float64(dtype)
    if isinstance(data, Tensor):
        out = Tensor(_tf.clone(data._h), requires_grad,      # 손잡이를 나눠 갖지 않는다
                     dt=dtype or data._dtype)
        out._nhwc = data._nhwc
        return out
    arr = _np.asarray(data)
    dt = dtype or _dtype_of(arr)
    return Tensor(_to_tf(arr, dt), requires_grad, dt=dt)


def from_numpy(arr):
    arr = _np.asarray(arr)
    dt = _dtype_of(arr)
    return Tensor(_to_tf(arr, dt), dt=dt)


def zeros(*shape, dtype=None, requires_grad=False):
    _reject_float64(dtype)
    shape = shape[0] if len(shape) == 1 and isinstance(shape[0], (tuple, list)) else shape
    return Tensor(_tf.zeros(_to_js(list(shape))), requires_grad, dt=dtype or float32)


def ones(*shape, dtype=None, requires_grad=False):
    _reject_float64(dtype)
    shape = shape[0] if len(shape) == 1 and isinstance(shape[0], (tuple, list)) else shape
    return Tensor(_tf.ones(_to_js(list(shape))), requires_grad, dt=dtype or float32)


# 난수는 **numpy 로 뽑아 올린다.** TF.js 의 난수를 쓰면 `manual_seed` 가 코어와 다른
# 흐름을 타서 같은 씨앗에 다른 값이 나온다. 초기화는 한 번뿐이라 올리는 비용도 없다.
_rng = _np.random.default_rng(0)


def manual_seed(seed):
    global _rng
    _rng = _np.random.default_rng(seed)
    return seed


class Generator:
    """씨앗을 담아 다니는 그릇. `random_split(generator=...)` 이 이것을 받는다 —
    나누기를 고정하지 않으면 모델을 바꿔 좋아진 건지 나누기가 운이 좋았던 건지 알 수 없다."""

    def __init__(self):
        self.seed = 0

    def manual_seed(self, seed):
        self.seed = seed
        return self

    def rng(self):
        return _np.random.default_rng(self.seed)


def rand(*shape):
    shape = shape[0] if len(shape) == 1 and isinstance(shape[0], (tuple, list)) else shape
    return Tensor(_to_tf(_rng.random(shape).astype(_np.float32)), dt=float32)


def randint(low, high, shape):
    return Tensor(_to_tf(_rng.integers(low, high, shape).astype(_np.int64), int64), dt=int64)


def randperm(n):
    return Tensor(_to_tf(_rng.permutation(n).astype(_np.int64), int64), dt=int64)


def arange(*args, dtype=None):
    _reject_float64(dtype)
    arr = _np.arange(*args)
    dt = dtype or _dtype_of(arr)
    return Tensor(_to_tf(arr, dt), dt=dt)


def randn(*shape, requires_grad=False):
    shape = shape[0] if len(shape) == 1 and isinstance(shape[0], (tuple, list)) else shape
    return Tensor(_to_tf(_rng.standard_normal(shape).astype(_np.float32)),
                  requires_grad, dt=float32)


# ---------------------------------------------------------------- 원소별
#
# TF.js 이름과 torch 이름이 갈리는 자리가 있어서 표로 둔다(matMul·notEqual 등).
# 미분이 정의되지 않는 것(sign·floor·ceil·round)은 기울기를 0 으로 둔다 — torch 도 그렇다.

def _unary(name, forward, derivative=None, keeps_dtype=False):
    """`keeps_dtype` 은 정수를 정수로 돌려주는 것들이다 — torch 에서 `abs`·`sign`·
    `floor`·`relu` 가 그렇고, `exp`·`log` 같은 것은 정수를 넣어도 실수를 준다."""
    def fn(t):
        t = _wrap(t)          # 원소별이라 레이아웃과 무관하다 — 되돌리면 안 된다
        dt = t._dtype if keeps_dtype else float32
        h = _storage_for(t, dt)
        out = forward(h)
        if derivative is None:
            return Tensor(out, dt=dt)
        return t._make(out, (t,), lambda g: (_tf.mul(g, derivative(h, out)),),
                       f"{name}Backward0", dt=dt)
    fn.__name__ = name
    return fn


_LN2 = float(_np.log(2.0))
_LN10 = float(_np.log(10.0))

exp = _unary("Exp", lambda x: _tf.exp(x), lambda x, o: o)
log = _unary("Log", lambda x: _tf.log(x), lambda x, o: _tf.div(1.0, x))
log2 = _unary("Log2", lambda x: _tf.div(_tf.log(x), _LN2),
              lambda x, o: _tf.div(1.0, _tf.mul(x, _LN2)))
log10 = _unary("Log10", lambda x: _tf.div(_tf.log(x), _LN10),
               lambda x, o: _tf.div(1.0, _tf.mul(x, _LN10)))
sqrt = _unary("Sqrt", lambda x: _tf.sqrt(x), lambda x, o: _tf.div(0.5, o))
rsqrt = _unary("Rsqrt", lambda x: _tf.rsqrt(x), lambda x, o: _tf.div(_tf.mul(-0.5, o), x))
square = _unary("Square", lambda x: _tf.square(x), lambda x, o: _tf.mul(2.0, x))
reciprocal = _unary("Reciprocal", lambda x: _tf.reciprocal(x), lambda x, o: _tf.neg(_tf.mul(o, o)))
abs = _unary("Abs", lambda x: _tf.abs(x), lambda x, o: _tf.sign(x), keeps_dtype=True)
sin = _unary("Sin", lambda x: _tf.sin(x), lambda x, o: _tf.cos(x))
cos = _unary("Cos", lambda x: _tf.cos(x), lambda x, o: _tf.neg(_tf.sin(x)))
tan = _unary("Tan", lambda x: _tf.tan(x), lambda x, o: _tf.add(1.0, _tf.mul(o, o)))
sinh = _unary("Sinh", lambda x: _tf.sinh(x), lambda x, o: _tf.cosh(x))
cosh = _unary("Cosh", lambda x: _tf.cosh(x), lambda x, o: _tf.sinh(x))
tanh = _unary("Tanh", lambda x: _tf.tanh(x), lambda x, o: _tf.sub(1.0, _tf.mul(o, o)))
erf = _unary("Erf", lambda x: _tf.erf(x),
             lambda x, o: _tf.mul(2.0 / float(_np.sqrt(_np.pi)), _tf.exp(_tf.neg(_tf.square(x)))))
relu = _unary("Relu", lambda x: _tf.relu(x), lambda x, o: _tf.step(x), keeps_dtype=True)
sigmoid = _unary("Sigmoid", lambda x: _tf.sigmoid(x),
                 lambda x, o: _tf.mul(o, _tf.sub(1.0, o)))
# 계단 모양 — 미분이 거의 모든 곳에서 0 이다.
#
# **0 을 흘린다. 그래프를 끊지 않는다.** torch 에 물어보니 넷 다 `backward()` 가 돌고
# `.grad` 가 0 으로 채워진다. 전에는 맨 텐서를 돌려줘 `backward()` 가 거절했는데,
# 그건 "없는 기능"이지 torch 와 같은 것이 아니었다. 계단을 중간에 낀 손실은 torch 에서
# 실제로 돈다.
_zeros_like = lambda x, o: _tf.zerosLike(x)                          # noqa: E731

sign = _unary("Sign", lambda x: _tf.sign(x), _zeros_like, keeps_dtype=True)
floor = _unary("Floor", lambda x: _tf.floor(x), _zeros_like, keeps_dtype=True)
ceil = _unary("Ceil", lambda x: _tf.ceil(x), _zeros_like, keeps_dtype=True)
round = _unary("Round", lambda x: _tf.round(x), _zeros_like, keeps_dtype=True)


# ---- 삼각·지수·로그의 나머지
#
# 코어와 같은 목록이다. TF.js 에 원시 함수가 다 있는지 브라우저에 **물어보고** 넣었다 —
# 없는 것을 근사로 채우면 그게 조용히 틀리는 자리가 된다. 34개 중 없는 것은 없었다.

acos = _unary("Acos", lambda x: _tf.acos(x),
              lambda x, o: _tf.neg(_tf.rsqrt(_tf.sub(1.0, _tf.square(x)))))
asin = _unary("Asin", lambda x: _tf.asin(x),
              lambda x, o: _tf.rsqrt(_tf.sub(1.0, _tf.square(x))))
atan = _unary("Atan", lambda x: _tf.atan(x),
              lambda x, o: _tf.div(1.0, _tf.add(1.0, _tf.square(x))))
acosh = _unary("Acosh", lambda x: _tf.acosh(x),
               lambda x, o: _tf.rsqrt(_tf.sub(_tf.square(x), 1.0)))
asinh = _unary("Asinh", lambda x: _tf.asinh(x),
               lambda x, o: _tf.rsqrt(_tf.add(_tf.square(x), 1.0)))
atanh = _unary("Atanh", lambda x: _tf.atanh(x),
               lambda x, o: _tf.div(1.0, _tf.sub(1.0, _tf.square(x))))
expm1 = _unary("Expm1", lambda x: _tf.expm1(x), lambda x, o: _tf.add(o, 1.0))
log1p = _unary("Log1p", lambda x: _tf.log1p(x),
               lambda x, o: _tf.div(1.0, _tf.add(1.0, x)))
exp2 = _unary("Exp2", lambda x: _tf.exp(_tf.mul(x, _LN2)),
              lambda x, o: _tf.mul(o, _LN2))
_DEG = float(180.0 / _np.pi)
deg2rad = _unary("Deg2rad", lambda x: _tf.div(x, _DEG), lambda x, o: _tf.fill(
    _to_js(list(_shape_of(x))), 1.0 / _DEG))
rad2deg = _unary("Rad2deg", lambda x: _tf.mul(x, _DEG), lambda x, o: _tf.fill(
    _to_js(list(_shape_of(x))), _DEG))
# 0 쪽으로 자른다 — `floor` 는 아래로 자르므로 음수에서 갈린다.
_trunc = lambda x: _tf.mul(_tf.sign(x), _tf.floor(_tf.abs(x)))       # noqa: E731
trunc = _unary("Trunc", _trunc, _zeros_like, keeps_dtype=True)
frac = _unary("Frac", lambda x: _tf.sub(x, _trunc(x)),
              lambda x, o: _tf.onesLike(x))
# `sgn` 은 실수에서 `sign` 과 같다 — 0 을 흘린다. (한 번 "torch 가 거절한다"고 적었는데
# 그 예외는 `backward()` 가 아니라 결과를 찍던 쪽에서 났다. 코어 `_ops` 에 자세히 적었다.)
sgn = _unary("Sgn", lambda x: _tf.sign(x), _zeros_like, keeps_dtype=True)
positive = _unary("Positive", lambda x: _tf.clone(x), lambda x, o: _tf.onesLike(x))
erfc = _unary("Erfc", lambda x: _tf.sub(1.0, _tf.erf(x)),
              lambda x, o: _tf.mul(-2.0 / float(_np.sqrt(_np.pi)),
                                   _tf.exp(_tf.neg(_tf.square(x)))))
logit = _unary("Logit", lambda x: _tf.log(_tf.div(x, _tf.sub(1.0, x))),
               lambda x, o: _tf.div(1.0, _tf.mul(x, _tf.sub(1.0, x))))


def sinc(t):
    """`sin(πx)/(πx)`, x=0 에서는 1. **0 에서 나누지 않도록** 자리를 미리 바꾼다."""
    t = _wrap(t)
    h = _storage_for(t, float32)
    zero = _tf.equal(h, 0.0)
    safe = _tf.where(zero, _tf.onesLike(h), h)
    pix = _tf.mul(safe, float(_np.pi))
    out = _tf.where(zero, _tf.onesLike(h), _tf.div(_tf.sin(pix), pix))

    def back(g):
        # d/dx sinc(x) = (cos(πx) - sinc(x)) / x, x=0 에서는 0.
        d = _tf.div(_tf.sub(_tf.cos(pix), out), safe)
        return (_tf.mul(g, _tf.where(zero, _tf.zerosLike(h), d)),)

    return t._make(out, (t,), back, "SincBackward0")


def _binary_math(name, forward, d_a, d_b):
    """두 텐서를 받는 원소별 함수. 브로드캐스팅과 되돌리기를 `_binary` 에 맡긴다.

    `_binary` 는 역방향을 **하나만** 받고 그것이 짝을 돌려준다(코어는 둘을 받는다).
    두 라이브러리에서 같은 실수를 하지 않으려고 여기서 감싼다.
    """
    def fn(a, b):
        a = _wrap(a)
        return a._binary(b, forward,
                         lambda g, x, y: (d_a(g, x, y), d_b(g, x, y)),
                         f"{name}Backward0")
    fn.__name__ = name
    return fn


atan2 = _binary_math(
    "Atan2", lambda x, y: _tf.atan2(x, y),
    lambda g, x, y: _tf.mul(g, _tf.div(y, _tf.add(_tf.square(x), _tf.square(y)))),
    lambda g, x, y: _tf.mul(g, _tf.neg(_tf.div(x, _tf.add(_tf.square(x), _tf.square(y))))))
hypot = _binary_math(
    "Hypot", lambda x, y: _tf.sqrt(_tf.add(_tf.square(x), _tf.square(y))),
    lambda g, x, y: _tf.mul(g, _tf.div(x, _tf.sqrt(_tf.add(_tf.square(x), _tf.square(y))))),
    lambda g, x, y: _tf.mul(g, _tf.div(y, _tf.sqrt(_tf.add(_tf.square(x), _tf.square(y))))))
# |x|·sign(y) 이므로 x 로는 sign(x)·sign(y), y 로는 0 이다(계단).
copysign = _binary_math(
    "Copysign", lambda x, y: _tf.mul(_tf.abs(x), _tf.sign(y)),
    lambda g, x, y: _tf.mul(g, _tf.mul(_tf.sign(x), _tf.sign(y))),
    lambda g, x, y: _tf.zerosLike(_tf.mul(g, y)))


def _logaddexp_h(x, y, base=None):
    """`log(exp(x) + exp(y))` 를 **넘치지 않게** 센다 — 큰 쪽을 빼고 더한다."""
    big = _tf.maximum(x, y)
    small = _tf.minimum(x, y)
    diff = _tf.sub(small, big)
    if base is None:
        return _tf.add(big, _tf.log1p(_tf.exp(diff)))
    return _tf.add(big, _tf.div(_tf.log1p(_tf.exp(_tf.mul(diff, _LN2))), _LN2))


logaddexp = _binary_math(
    "Logaddexp", lambda x, y: _logaddexp_h(x, y),
    lambda g, x, y: _tf.mul(g, _tf.exp(_tf.sub(x, _logaddexp_h(x, y)))),
    lambda g, x, y: _tf.mul(g, _tf.exp(_tf.sub(y, _logaddexp_h(x, y)))))
logaddexp2 = _binary_math(
    "Logaddexp2", lambda x, y: _logaddexp_h(x, y, base=2),
    lambda g, x, y: _tf.mul(g, _tf.exp(_tf.mul(_tf.sub(x, _logaddexp_h(x, y, 2)), _LN2))),
    lambda g, x, y: _tf.mul(g, _tf.exp(_tf.mul(_tf.sub(y, _logaddexp_h(x, y, 2)), _LN2))))


def xlogy(a, b):
    """`x · log(y)` 인데 **x 가 0 이면 0 이다** — `0 · log(0)` 을 nan 으로 두지 않는다."""
    a = _wrap(a)
    zero = lambda x: _tf.equal(x, 0.0)                               # noqa: E731
    return a._binary(
        b,
        lambda x, y: _tf.where(zero(x), _tf.zerosLike(x), _tf.mul(x, _tf.log(y))),
        lambda g, x, y: (
            _tf.mul(g, _tf.where(zero(x), _tf.zerosLike(x), _tf.log(y))),
            _tf.mul(g, _tf.where(zero(x), _tf.zerosLike(x), _tf.div(x, y)))),
        "XlogyBackward0")


def signbit(t):
    t = _wrap(t)
    return Tensor(_tf.less(_storage_for(t, float32), 0.0), dt=bool_)


def heaviside(t, values):
    """x<0 이면 0, x>0 이면 1, x=0 이면 `values`."""
    t, v = _wrap(t), _wrap(values)
    h = _storage_for(t, float32)
    return Tensor(_tf.where(_tf.equal(h, 0.0), _storage_for(v, float32), _tf.step(h)))


def ldexp(t, other):
    t, o = _wrap(t), _wrap(other)
    return t * Tensor(_tf.exp(_tf.mul(_storage_for(o, float32), _LN2)))


# torch 의 별칭들. 같은 함수를 가리킨다. `clip` 은 `clamp` 정의 뒤에 있어야 해서
# 이 묶음에 없다 — 아래 `clamp` 바로 밑에 있다.
arccos, arcsin, arctan = acos, asin, atan
arccosh, arcsinh, arctanh = acosh, asinh, atanh
fix = trunc
absolute = abs


def neg(t):
    return -_wrap(t)


negative = neg


def prod(t, dim=None):
    t = _canonical(t)
    out = _tf.prod(t._h) if dim is None else _tf.prod(t._h, dim)
    return t._make(out, (t,), lambda g: (_tf.div(_tf.mul(g, out), t._h),), "ProdBackward0")


def count_nonzero(t, dim=None):
    t = _canonical(t)
    nz = _tf.cast(_tf.notEqual(t._h, 0.0), "float32")
    return Tensor(_tf.sum(nz) if dim is None else _tf.sum(nz, dim))


def matmul(a, b):
    return _wrap(a) @ _wrap(b)


def mm(a, b):
    return _wrap(a) @ _wrap(b)


# ---------------------------------------------------------------- 비교·클램프

def maximum(a, b):
    a, b = _align(_wrap(a), _wrap(b))
    pick = _tf.cast(_tf.greaterEqual(a._h, b._h), "float32")
    return a._make(_tf.maximum(a._h, b._h), (a, b),
                   lambda g: (_tf.mul(g, pick), _tf.mul(g, _tf.sub(1.0, pick))),
                   "MaximumBackward0")


def minimum(a, b):
    a, b = _align(_wrap(a), _wrap(b))
    pick = _tf.cast(_tf.lessEqual(a._h, b._h), "float32")
    return a._make(_tf.minimum(a._h, b._h), (a, b),
                   lambda g: (_tf.mul(g, pick), _tf.mul(g, _tf.sub(1.0, pick))),
                   "MinimumBackward0")


def clamp(t, min=None, max=None):
    t = _canonical(t)
    lo = -1e30 if min is None else float(min)
    hi = 1e30 if max is None else float(max)
    inside = _tf.cast(_tf.logicalAnd(_tf.greaterEqual(t._h, lo), _tf.lessEqual(t._h, hi)), "float32")
    return t._make(_tf.clipByValue(t._h, lo, hi), (t,),
                   lambda g: (_tf.mul(g, inside),), "ClampBackward0")


clip = clamp                                    # torch 의 별칭


# ---------------------------------------------------------------- 선형대수

def dot(a, b):
    return (_wrap(a) * _wrap(b)).sum()


def bmm(a, b):
    return _wrap(a) @ _wrap(b)


def eye(n):
    return Tensor(_tf.eye(n), dt=float32)


def full(shape, value, dtype=None):
    _reject_float64(dtype)
    dt = dtype or (int64 if isinstance(value, int) and not isinstance(value, bool) else float32)
    return Tensor(_tf.fill(_to_js(list(shape)), float(value)), dt=dt)


def full_like(t, value):
    return full(_wrap(t).shape, value)


def zeros_like(t, dtype=None):
    t = _wrap(t)
    return Tensor(_tf.zerosLike(t._h), dt=dtype or t._dtype)


def ones_like(t, dtype=None):
    t = _wrap(t)
    return Tensor(_tf.onesLike(t._h), dt=dtype or t._dtype)


def empty(*shape, dtype=None):
    return zeros(*shape, dtype=dtype)


def linspace(start, end, steps):
    return Tensor(_to_tf(_np.linspace(start, end, steps).astype(_np.float32)), dt=float32)


def as_tensor(data, dtype=None):
    if isinstance(data, Tensor) and dtype is None:
        return data
    return tensor(data, dtype)


def where(cond, a, b):
    """조건이 참인 자리는 a, 아니면 b. 기울기도 그 자리로만 간다."""
    c = _wrap(cond)
    mask = c._h if c._dtype is bool_ else _tf.notEqual(c._h, 0.0)
    ta, tb = _align(_wrap(a), _wrap(b))
    keep = _tf.cast(mask, "float32")
    return ta._make(_tf.where(mask, ta._h, tb._h), (ta, tb),
                    lambda g: (_tf.mul(g, keep), _tf.mul(g, _tf.sub(1.0, keep))),
                    "WhereBackward0")


def _masked(t, mask, op):
    """마스크를 곱한다. 역방향은 **같은 마스크를 다시 곱하는** 것이다 — 지운 자리는
    출력에 안 나타났으니 기울기도 0 이다.

    마스크 손잡이를 닫아두지 않고 역방향에서 새로 만든다. 붙잡아 두면 `scope()` 가
    닫힐 때 이미 놓인 버퍼를 가리키게 되고, 그건 이 파일에서 세 번 겪은 함정이다.
    """
    return t._make(_tf.mul(t._h, _to_tf(mask)), (t,),
                   lambda g: (_tf.mul(g, _to_tf(mask)),), op)


def tril(t, diagonal=0):
    t = _canonical(t)
    n, m = t.shape[-2], t.shape[-1]
    return _masked(t, _np.tril(_np.ones((n, m), dtype=_np.float32), k=diagonal),
                   "TrilBackward0")


def triu(t, diagonal=0):
    t = _canonical(t)
    n, m = t.shape[-2], t.shape[-1]
    return _masked(t, _np.triu(_np.ones((n, m), dtype=_np.float32), k=diagonal),
                   "TriuBackward0")


def masked_fill(t, mask, value):
    t = _canonical(t)
    m = _wrap(mask)
    keep = _tf.cast(_tf.logicalNot(m._h if m._dtype is bool_
                                   else _tf.notEqual(m._h, 0.0)), "float32")
    filled = _tf.where(m._h if m._dtype is bool_ else _tf.notEqual(m._h, 0.0),
                       _tf.fill(_to_js(list(t.shape)), float(value)), t._h)
    return t._make(filled, (t,), lambda g: (_tf.mul(g, keep),), "MaskedFillBackward0")


def repeat_interleave(t, repeats, dim=None):
    """제자리에서 늘린다. **`stack` 과 `reshape` 로 짠다** — 둘 다 이미 그래프를 잇는다.

    셋 다 전에는 numpy 로 내려받아 다시 올렸다. 값은 맞았지만 GPU 를 왕복하면서
    기울기가 끊겼다 — 되찾는 가장 안전한 길은 역방향을 새로 쓰는 것이 아니라 **이미
    검증된 연산으로 다시 짜는 것**이다.
    """
    t = _canonical(t)
    if not isinstance(repeats, int):
        _unsupported("repeat_interleave(repeats 가 정수가 아닌 경우)")
    if dim is None:
        n = t.numel()
        return stack([t.reshape(n)] * repeats, 1).reshape(n * repeats)
    shape = tuple(t.shape)
    merged = shape[:dim] + (shape[dim] * repeats,) + shape[dim + 1:]
    return stack([t] * repeats, dim + 1).reshape(*merged)


def tile(t, reps):
    """통째로 반복해 붙인다. `cat` 으로 짠다."""
    t = _canonical(t)
    reps_t = (reps,) if isinstance(reps, int) else tuple(reps)
    nd = max(t.ndim, len(reps_t))
    if t.ndim < nd:                       # reps 가 더 길면 앞에 1 차원을 세운다
        t = t.reshape(*((1,) * (nd - t.ndim) + tuple(t.shape)))
    out = t
    for axis, r in enumerate((1,) * (nd - len(reps_t)) + reps_t):
        if r > 1:
            out = cat([out] * r, axis)
    return out


def movedim(t, source, destination):
    """축 하나를 다른 자리로 옮긴다.

    **음수 자리를 먼저 편다.** `list.insert(-1, x)` 는 맨 뒤가 아니라 **마지막 앞**에
    넣는다 — `movedim(0, -1)` 이 아무 일도 안 하고 원래 모양을 돌려줬다. 예외도 없이
    조용히 항등이 되는 종류다.

    골든이 이것을 못 봤던 이유도 적어둔다. 내가 세운 케이스가 `movedim(0, 0)` 이었다.
    프로브에서 `IndexError` 를 피하려고 고른 인자였고, 그래서 **음수 자리는 아무도
    물어본 적이 없었다.** 통과하는 검사가 있다는 것과 그것이 무엇을 물었는지는 다르다.
    """
    t = _canonical(t)
    n = t.ndim
    src, dst = source % n, destination % n
    order = list(range(n))
    order.insert(dst, order.pop(src))
    return t.permute(*order)


def argsort(t, dim=-1, descending=False):
    return sort(t, dim, descending).indices


def bincount(t):
    # int32 로 낮춰서 센다 — wasm32 에서는 numpy 의 intp 가 32비트라 int64 를 거부한다.
    counts = _np.bincount(_canonical(t).numpy().astype(_np.int32))
    return Tensor(_to_tf(counts.astype(_np.int64), int64), dt=int64)


def multinomial(probs, num_samples, replacement=True):
    p = _canonical(probs).numpy().astype(_np.float64)
    p = p / p.sum(axis=-1, keepdims=True)
    if p.ndim == 1:
        out = _rng.choice(len(p), size=num_samples, p=p)
    else:
        out = _np.asarray([_rng.choice(p.shape[-1], size=num_samples, p=row) for row in p])
    return Tensor(_to_tf(out.astype(_np.int64), int64), dt=int64)


def einsum(equation, *operands):
    """TF.js 에 einsum 이 없어 numpy 로 계산한다. **역방향도 numpy 로 한다.**

    이미 내려받아 계산하고 있었으므로 역방향을 붙인다고 더 느려지지 않는다. 규칙은
    하나다 — 출력 첨자를 항의 자리에 바꿔 넣으면 그 항의 기울기가 나온다. 어떤 첨자가
    그 항에만 있으면(`ij->i` 의 `j`) einsum 이 없던 축을 못 만들므로 1 로 채운 항을
    끼워 넣는다. `...` 과 한 항 안의 반복 첨자는 이 규칙이 안 통해서 **틀린 기울기를
    주는 대신 기울기를 안 준다** — 그때는 `backward()` 가 거절한다.
    """
    ops = [_canonical(o) for o in operands]
    arrays = [o.numpy() for o in ops]
    out = _np.einsum(equation, *arrays)
    eq = equation.replace(" ", "")
    if "->" in eq:
        lhs, rhs = eq.split("->")
    else:
        lhs = eq
        counts = {}
        for c in lhs.replace(",", ""):
            counts[c] = counts.get(c, 0) + 1
        rhs = "".join(sorted(c for c, k in counts.items() if k == 1))
    subs = lhs.split(",")
    if "..." in eq or any(len(set(s)) != len(s) for s in subs):
        return Tensor(_to_tf(out), dt=float32)

    def back(g):
        gg = _np.asarray(_to_np(g), dtype=_np.float32).reshape(out.shape)
        grads = []
        for i, mine in enumerate(subs):
            rest = [(subs[j], arrays[j]) for j in range(len(subs)) if j != i]
            known = set(rhs) | {c for s, _ in rest for c in s}
            missing = [c for c in mine if c not in known]
            spec = [rhs] + [s for s, _ in rest]
            terms = [gg] + [d for _, d in rest]
            if missing:
                spec.append("".join(missing))
                terms.append(_np.ones([arrays[i].shape[mine.index(c)] for c in missing],
                                      dtype=_np.float32))
            grads.append(_to_tf(_np.einsum(",".join(spec) + "->" + mine, *terms)))
        return tuple(grads)

    return ops[0]._make(_to_tf(out), tuple(ops), back, "EinsumBackward0")


def allclose(a, b, rtol=1e-5, atol=1e-8):
    return bool(_np.allclose(_wrap(a).numpy(), _wrap(b).numpy(), rtol=rtol, atol=atol))


def equal(a, b):
    return bool(_np.array_equal(_wrap(a).numpy(), _wrap(b).numpy()))


def _compare(name, fn):
    def cmp(a, b):
        ta, tb = _align(_wrap(a), _wrap(b))
        return Tensor(fn(ta._h, tb._h), dt=bool_)
    cmp.__name__ = name
    return cmp


eq = _compare("eq", lambda a, b: _tf.equal(a, b))
ne = _compare("ne", lambda a, b: _tf.notEqual(a, b))
lt = _compare("lt", lambda a, b: _tf.less(a, b))
le = _compare("le", lambda a, b: _tf.lessEqual(a, b))
gt = _compare("gt", lambda a, b: _tf.greater(a, b))
ge = _compare("ge", lambda a, b: _tf.greaterEqual(a, b))
logical_and = _compare("logical_and", lambda a, b: _tf.logicalAnd(a, b))
logical_or = _compare("logical_or", lambda a, b: _tf.logicalOr(a, b))


def logical_not(t):
    t = _wrap(t)
    h = t._h if t._dtype is bool_ else _tf.notEqual(t._h, 0.0)
    return Tensor(_tf.logicalNot(h), dt=bool_)


def isnan(t):
    return Tensor(_tf.isNaN(_wrap(t)._h), dt=bool_)


def isinf(t):
    return Tensor(_tf.isInf(_wrap(t)._h), dt=bool_)


def isfinite(t):
    return Tensor(_tf.isFinite(_wrap(t)._h), dt=bool_)


def pow(t, exponent):
    return _wrap(t) ** exponent


def outer(a, b):
    a, b = _wrap(a), _wrap(b)
    return reshape(a, (-1, 1)) @ reshape(b, (1, -1))


def reshape(t, shape):
    t = _canonical(t)
    old = t.shape
    return t._make(_tf.reshape(t._h, _to_js(list(shape))), (t,),
                   lambda g: (_tf.reshape(g, _to_js(list(old))),), "ViewBackward0")


def diag(t):
    """torch 의 `diag` 는 **행렬에서 대각을 뽑는다.** TF.js 의 `diag` 는 반대로
    벡터에서 행렬을 만든다 — 이름이 같고 뜻이 반대라, 그대로 부르면 조용히 다른 값이 나온다."""
    t = _canonical(t)
    if t.ndim == 1:                       # 벡터 → 대각행렬
        n = t.shape[0]
        return t.reshape(n, 1) * eye(n)
    # 행렬 → 대각. 단위행렬을 곱해 한 축을 접으면 되고, **곱셈과 합은 이미 그래프를
    # 잇는다** — 손으로 역방향을 쓸 이유가 없다.
    return (t * eye(t.shape[0])).sum(dim=1)


def trace(t):
    t = _canonical(t)
    return (t * eye(t.shape[0])).sum()


def norm(t, p=2, dim=None):
    t = _canonical(t)
    if p == 1:
        return abs(t).sum(dim=dim)
    return (t * t).sum(dim=dim) ** 0.5


# ---- 축약의 나머지
#
# 코어와 같은 의미를 지킨다. 특히 `amax`·`amin` 은 **동점일 때 기울기를 고르게 나눈다**
# (실측: [1,3,3,2] 의 amax 기울기가 [0,.5,.5,0]). 한 자리에 몰아주면 값 검사는 통과하고
# 학습만 미묘하게 갈린다.

def _spread_extreme(t, dim, keepdim, take, name):
    t = _canonical(t)
    raw = _tf.max(t._h, dim) if take == "max" else _tf.min(t._h, dim)
    kept = raw if dim is None else (
        _tf.max(t._h, dim, True) if take == "max" else _tf.min(t._h, dim, True))
    hit = _tf.cast(_tf.equal(t._h, kept if dim is not None else raw), "float32")
    share = _tf.div(hit, _tf.sum(hit, dim, True) if dim is not None else _tf.sum(hit))
    out = kept if (keepdim and dim is not None) else raw

    def back(g):
        gg = g if (dim is None or keepdim) else _tf.expandDims(g, dim)
        return (_tf.mul(gg, share),)

    return t._make(out, (t,), back, name)


def amax(t, dim=None, keepdim=False):
    return _spread_extreme(t, dim, keepdim, "max", "AmaxBackward0")


def amin(t, dim=None, keepdim=False):
    return _spread_extreme(t, dim, keepdim, "min", "AminBackward0")


def aminmax(t, dim=None, keepdim=False):
    return _ValuesIndices(amin(t, dim, keepdim), amax(t, dim, keepdim))


def _nan_split(t):
    """nan 자리를 0 으로 바꾼 손잡이와, 어디가 nan 이었는지."""
    bad = _tf.logicalNot(_tf.equal(t._h, t._h))          # nan 은 자기 자신과 다르다
    return _tf.where(bad, _tf.zerosLike(t._h), t._h), bad


def nansum(t, dim=None, keepdim=False):
    """nan 을 **0 으로 세는** 합. 기울기도 그 자리로는 안 간다."""
    t = _canonical(t)
    clean, bad = _nan_split(t)
    out = _tf.sum(clean) if dim is None else _tf.sum(clean, dim, keepdim)

    def back(g):
        gg = g if (dim is None or keepdim) else _tf.expandDims(g, dim)
        spread = _tf.mul(_tf.onesLike(t._h), gg)
        return (_tf.where(bad, _tf.zerosLike(spread), spread),)

    return t._make(out, (t,), back, "NansumBackward0")


def nanmean(t, dim=None, keepdim=False):
    """nan 을 **빼고** 낸 평균 — 세는 개수도 nan 이 아닌 것만이다."""
    t = _canonical(t)
    clean, bad = _nan_split(t)
    good = _tf.cast(_tf.logicalNot(bad), "float32")
    count = _tf.sum(good) if dim is None else _tf.sum(good, dim, True)
    total = _tf.sum(clean) if dim is None else _tf.sum(clean, dim, True)
    full = _tf.div(total, count)
    out = full if (keepdim or dim is None) else _tf.squeeze(full, _to_js([dim]))

    def back(g):
        gg = g if (dim is None or keepdim) else _tf.expandDims(g, dim)
        spread = _tf.div(_tf.mul(_tf.onesLike(t._h), gg), count)
        return (_tf.where(bad, _tf.zerosLike(spread), spread),)

    return t._make(out, (t,), back, "NanmeanBackward0")


def logsumexp(t, dim=None, keepdim=False):
    """`log(sum(exp(x)))` 를 **넘치지 않게** 센다 — 큰 값을 빼고 더한다."""
    t = _canonical(t)
    big = _tf.max(t._h) if dim is None else _tf.max(t._h, dim, True)
    shifted = _tf.exp(_tf.sub(t._h, big))
    total = _tf.sum(shifted) if dim is None else _tf.sum(shifted, dim, True)
    full = _tf.add(_tf.log(total), big)
    soft = _tf.div(shifted, total)
    out = full if (keepdim or dim is None) else _tf.squeeze(full, _to_js([dim]))

    def back(g):
        gg = g if (dim is None or keepdim) else _tf.expandDims(g, dim)
        return (_tf.mul(gg, soft),)

    return t._make(out, (t,), back, "LogsumexpBackward0")


def _cum_extreme(t, dim, better, name):
    """누적 최대·최소. **번호는 CPU 에서 낸다.**

    TF.js 에 `cummax` 가 없고, 자리마다 앞을 돌아보는 계산이라 원시 연산 조합으로는
    안 나온다. 값은 뽑은 번호로 GPU 에서 모으므로 **기울기는 그래프 안에 남는다** —
    번호가 CPU 에서 나왔다고 그래프가 끊기지는 않는다.
    """
    t = _canonical(t)
    d = t.numpy()
    axis = dim % d.ndim

    # 번호는 앞을 돌아보며 한 칸씩 정한다. torch 는 동점에서 **나중 자리**를 준다.
    moved = _np.moveaxis(d, axis, 0)
    idx = _np.zeros(moved.shape, dtype=_np.intp)
    best = moved[0].copy()
    for i in range(1, moved.shape[0]):
        take = better(moved[i], best)
        idx[i] = _np.where(take, i, idx[i - 1])
        best = _np.where(take, moved[i], best)
    idx = _np.moveaxis(idx, 0, axis)

    # 값은 **원-핫 곱으로 GPU 에서** 모은다 — 그래야 역전파가 저절로 따라온다.
    # 축을 마지막으로 옮겨서 곱하고 되돌린다.
    n = d.shape[axis]
    order = [i for i in range(d.ndim) if i != axis] + [axis]
    back_order = [order.index(i) for i in range(d.ndim)]
    xm = t.permute(*order) if order != list(range(d.ndim)) else t
    im = _np.moveaxis(idx, axis, -1)
    onehot = Tensor(_to_tf((im[..., None] == _np.arange(n)).astype(_np.float32)))
    lifted = xm.reshape(*(tuple(xm.shape[:-1]) + (1, n)))
    got = (onehot * lifted).sum(dim=-1)
    out = got.permute(*back_order) if back_order != list(range(d.ndim)) else got
    return _ValuesIndices(out, Tensor(_to_tf(idx.astype(_np.float32)), dt=int64))


def cummax(t, dim):
    return _cum_extreme(t, dim, lambda cur, best: cur >= best, "CummaxBackward0")


def cummin(t, dim):
    return _cum_extreme(t, dim, lambda cur, best: cur <= best, "CumminBackward0")


def kthvalue(t, k, dim=-1, keepdim=False):
    """**k 번째로 작은** 값. torch 는 1 부터 센다."""
    t = _canonical(t)
    _last_axis_only(t, dim, "kthvalue")
    d = t.numpy()
    order = _np.argsort(d, axis=-1, kind="stable")
    at = _np.take(order, k - 1, axis=-1)
    # **int32 로 올린다.** `tf.oneHot` 은 다른 것을 받지 않는다 — float32 에 int64
    # 라벨을 붙여 넘겼더니 거기서 멈췄다. 시끄럽게 멈춘 것이 다행이다.
    picked = _pick_last(t, _tf.cast(_to_tf(at[..., None].astype(_np.float32)), "int32"))
    values = picked if keepdim else picked.reshape(*d.shape[:-1])
    return _ValuesIndices(values, Tensor(_to_tf(at.astype(_np.float32)), dt=int64))


def msort(t):
    """**첫 축을 따라** 정렬한다."""
    t = _canonical(t)
    if t.ndim == 1:
        return sort(t).values
    return sort(t.movedim(0, -1)).values.movedim(-1, 0)


def diff(t, n=1, dim=-1):
    """이웃한 것의 차. **자르기로 짠다** — 자르기가 이미 그래프를 잇는다."""
    out = _canonical(t)
    axis = dim % out.ndim
    for _ in range(n):
        length = out.shape[axis]
        out = narrow(out, axis, 1, length - 1) - narrow(out, axis, 0, length - 1)
    return out


def dist(a, b, p=2):
    return norm(_wrap(a) - _wrap(b), p=p)


def quantile(t, q, dim=None, keepdim=False):
    """분위수. **CPU 에서 센다** — TF.js 에 보간이 없고, 정렬 뒤 두 값을 섞는 계산이
    자리마다 달라서 원시 연산으로 안 나온다. 기울기는 없다(torch 도 여기서는 잘 안 쓴다)."""
    t = _canonical(t)
    qq = q.numpy() if isinstance(q, Tensor) else _np.asarray(q, dtype=_np.float32)
    out = _np.quantile(t.numpy().astype(_np.float64), qq, axis=dim, keepdims=keepdim)
    return Tensor(_to_tf(_np.asarray(out, dtype=_np.float32)))


def nanquantile(t, q, dim=None, keepdim=False):
    t = _canonical(t)
    qq = q.numpy() if isinstance(q, Tensor) else _np.asarray(q, dtype=_np.float32)
    out = _np.nanquantile(t.numpy().astype(_np.float64), qq, axis=dim, keepdims=keepdim)
    return Tensor(_to_tf(_np.asarray(out, dtype=_np.float32)))


def nonzero(t):
    """0 이 아닌 자리의 좌표. **모양이 값에 달렸다** — 그래서 기울기가 없고, GPU 위에서
    모양을 미리 알 수 없어 읽어와서 만든다."""
    t = _canonical(t)
    out = _np.stack(_np.nonzero(t.numpy()), axis=-1)
    return Tensor(_to_tf(out.astype(_np.float32)), dt=int64)


def argwhere(t):
    return nonzero(t)


def cumsum(t, dim):
    t = _canonical(t)
    return t._make(_tf.cumsum(t._h, dim), (t,),
                   lambda g: (_tf.reverse(_tf.cumsum(_tf.reverse(g, dim), dim), dim),),
                   "CumsumBackward0")


def cumprod(t, dim):
    """누적 곱. **역방향은 CPU 에서 정확히 센다.**

    GPU 로 짜려면 `dL/dx_k = (1/x_k) * sum_{j>=k} g_j y_j` 를 쓰게 되는데, 입력에 0 이
    있으면 거기서 나눗셈이 터져 조용히 `nan` 이 흐른다. 예외도 안 난다 — 이 저장소가
    가장 싫어하는 모양이다. 그래서 `x_k` 를 뺀 곱을 직접 쌓는다. 내려받았다 올리므로
    느리지만 `cumprod` 는 학습 경로의 안쪽이 아니고, **0 이 섞였을 때 답이 맞는 쪽**이
    기준이다.
    """
    t = _canonical(t)

    def back(g):
        x = _np.moveaxis(t.numpy(), dim, 0)
        gg = _np.moveaxis(_np.asarray(_to_np(g), dtype=_np.float32), dim, 0)
        grad = _np.zeros_like(x, dtype=_np.float32)
        prefix = _np.ones_like(x[0])
        for k in range(x.shape[0]):
            run = prefix.copy()
            acc = gg[k] * run
            for j in range(k + 1, x.shape[0]):
                run = run * x[j]
                acc = acc + gg[j] * run
            grad[k] = acc
            prefix = prefix * x[k]
        return (_to_tf(_np.moveaxis(grad, 0, dim)),)

    return t._make(_tf.cumprod(t._h, dim), (t,), back, "CumprodBackward0")


# ---------------------------------------------------------------- 뽑기·모양


def topk(t, k, dim=-1, largest=True):
    t = _canonical(t)
    _last_axis_only(t, dim, "topk")
    if not largest:
        _unsupported("topk(largest=False)")
    idx = _tf.topk(t._h, k).indices
    return _ValuesIndices(_pick_last(t, idx), Tensor(idx))


def sort(t, dim=-1, descending=False):
    """TF.js 에는 정렬이 없다. `topk` 로 전부 뽑으면 내림차순이므로, 오름차순은 뒤집는다."""
    t = _canonical(t)
    _last_axis_only(t, dim, "sort")
    idx = _tf.topk(t._h, t.shape[-1]).indices
    if not descending:
        idx = _tf.reverse(idx, -1)
    return _ValuesIndices(_pick_last(t, idx), Tensor(idx))


def unique(t, sorted=True, return_counts=False):
    """**TF.js 의 WebGPU 백엔드에 `Unique` 커널이 없다**(실측: Kernel not registered).

    그리고 결과의 크기가 값에 따라 정해지므로 GPU 에 두기도 어렵다. 읽어와서 numpy 로
    하고 다시 올린다 — 동기화 한 번을 무는 대신 값이 맞는다. 학습 경로에는 안 쓰인다.
    """
    values = _np.unique(_wrap(t).numpy())
    return Tensor(_to_tf(values))


def masked_select(t, mask):
    """골라낸 개수가 값에 따라 달라진다. TF.js 의 `booleanMask` 는 **비동기**라
    동기 API 를 지키려면 쓸 수 없다. unique 와 같은 이유로 읽어와서 처리한다."""
    t = _canonical(t)
    m = mask.numpy() if isinstance(mask, Tensor) else _np.asarray(mask)
    return Tensor(_to_tf(t.numpy()[m.astype(bool)]))


def median(t, dim=None):
    """torch 는 원소가 짝수일 때 **가운데 둘 중 작은 쪽**을 준다."""
    t = _canonical(t)
    if dim is None:
        # 고른 자리를 **원-핫으로 곱해** 꺼낸다. 값만 슬라이스해 오면 그래프가 끊기는데,
        # 곱하고 더하는 길로 가면 기울기가 그 한 자리로만 흐른다 — torch 도 그렇다.
        n = t.numel()
        order = _np.argsort(t.numpy().reshape(-1), kind="stable")
        hot = _np.zeros(n, dtype=_np.float32)
        hot[order[(n - 1) // 2]] = 1.0
        return (t.reshape(n) * Tensor(_to_tf(hot))).sum()   # torch 는 0차원을 준다
    _last_axis_only(t, dim, "median")
    if t.ndim != 2:
        _unsupported("median(2차원이 아닌 것에 dim 을 준 경우)")
    rows, n = t.shape
    order = _tf.reverse(_tf.topk(t._h, n).indices, -1)          # 오름차순 자리 번호
    idx = _tf.slice(order, _to_js([0, (n - 1) // 2]), _to_js([rows, 1]))
    # **진짜 번호를 준다.** 예전에는 0 으로 채워 돌려줬는데, 값만 맞고 번호는 거짓이었다.
    #
    # 번호는 int32 그대로 둔다. `tf.cast(int32 → float32)` 는 WebGPU 에서 dtype 라벨만
    # 바꾸고 **비트를 안 바꾼다**(실측: 2 가 2.8e-45 로 읽힌다). torch 도 번호는 정수다.
    return _ValuesIndices(_pick_last(t, idx).reshape(rows),
                          Tensor(_tf.reshape(idx, _to_js([rows]))))


def flip(t, dims):
    t = _canonical(t)
    dims = [dims] if isinstance(dims, int) else list(dims)
    return t._make(_tf.reverse(t._h, _to_js(dims)), (t,),
                   lambda g: (_tf.reverse(g, _to_js(dims)),), "FlipBackward0")


def roll(t, shifts, dims=None):
    """TF.js 에 `roll` 이 없다. 잘라서 순서를 바꿔 붙인다."""
    t = _canonical(t)
    axis = 0 if dims is None else (dims if isinstance(dims, int) else dims[0])
    n = t.shape[axis]
    s = int(shifts) % n
    if s == 0:
        return Tensor(_tf.clone(t._h))
    head = _slice_along(t._h, axis, n - s, s)
    tail = _slice_along(t._h, axis, 0, n - s)
    return Tensor(_tf.concat(_to_js([head, tail]), axis))


def cat(items, dim=0):
    """이어 붙인다. 역방향은 붙인 자리대로 도로 자르는 것이다."""
    items = [_canonical(_wrap(t)) for t in items]
    sizes = [t.shape[dim] for t in items]
    out = _tf.concat(_to_js([t._h for t in items]), dim)

    def back(g):
        pieces, start = [], 0
        for size in sizes:
            pieces.append(_slice_along(g, dim, start, size))
            start += size
        return tuple(pieces)

    return items[0]._make(out, tuple(items), back, "CatBackward0")


def stack(items, dim=0):
    """새 축을 만들어 쌓는다. 역방향은 그 축에서 한 장씩 떼는 것이다."""
    items = [_canonical(_wrap(t)) for t in items]
    out = _tf.stack(_to_js([t._h for t in items]), dim)

    def back(g):
        return tuple(_tf.squeeze(_slice_along(g, dim, i, 1), _to_js([dim]))
                     for i in range(len(items)))

    return items[0]._make(out, tuple(items), back, "StackBackward0")


def narrow(t, dim, start, length):
    return _slice_tensor(_canonical(t), dim, start, length)


def split(t, size, dim=0):
    t = _canonical(t)
    n = t.shape[dim]
    sizes = size if isinstance(size, (list, tuple)) else \
        [size] * (n // size) + ([n % size] if n % size else [])
    out, start = [], 0
    for sz in sizes:
        out.append(_slice_tensor(t, dim, start, sz))
        start += sz
    return tuple(out)


def chunk(t, chunks, dim=0):
    t = _canonical(t)
    n = t.shape[dim]
    return split(t, -(-n // chunks), dim)


def unbind(t, dim=0):
    t = _canonical(t)
    shape = t.shape
    rest = tuple(s for i, s in enumerate(shape) if i != dim)
    return tuple(_slice_tensor(t, dim, i, 1).reshape(rest) for i in range(shape[dim]))


def gather(t, dim, index):
    """torch 의 `gather` — 원소마다 자리를 고른다. TF.js 의 `gather` 는 축을 통째로
    뽑는 다른 연산이라 그대로 못 쓴다.

    자리를 원-핫으로 만들어 곱하고 접는다. 그러면 **역전파가 그냥 따라온다** —
    뽑기만 하고 그래프를 끊으면 뽑은 자리로 기울기가 안 가고, 분류 손실이 통째로
    미분 불가가 된다(실제로 그랬다).
    """
    t = _canonical(t)
    if t.ndim != 2 or dim != 1:
        _unsupported("gather(2차원 · dim=1 이 아닌 것)")
    rows, cols = t.shape
    idx32 = _to_int32(index)
    k = _shape_of(idx32)[1]
    onehot = _tf.cast(_tf.oneHot(_tf.reshape(idx32, _to_js([rows * k])), cols), "float32")
    onehot = _tf.reshape(onehot, _to_js([rows, k, cols]))

    picked = _tf.sum(_tf.mul(onehot, _tf.reshape(t._h, _to_js([rows, 1, cols]))), 2)

    def back(g):
        return (_tf.sum(_tf.mul(onehot, _tf.reshape(g, _to_js([rows, k, 1]))), 1),)

    return t._make(picked, (t,), back, "GatherBackward0")


def _to_int32(index):
    handle = index._h if isinstance(index, Tensor) else _to_tf(_np.asarray(index))
    return _tf.cast(handle, "int32")


def index_select(t, dim, index):
    """원-핫 행렬을 곱해서 뽑는다 — 그래야 역전파가 따라온다."""
    t = _canonical(t)
    if dim != 0:
        _unsupported("index_select(dim=0 이 아닌 것)")
    shape = t.shape
    n = shape[0]
    idx32 = _tf.reshape(_to_int32(index), _to_js([-1]))
    k = _shape_of(idx32)[0]
    onehot = _tf.cast(_tf.oneHot(idx32, n), "float32")           # (k, n)
    rest = int(_np.prod(shape[1:])) if len(shape) > 1 else 1
    flat = t.reshape(n, rest)
    picked = flat._make(_tf.matMul(onehot, flat._h), (flat,),
                        lambda g: (_tf.matMul(onehot, g, True, False),),
                        "IndexSelectBackward0")
    return picked.reshape((k,) + tuple(shape[1:]))


