"""browsertorch 를 쪼갠 조각. 공개 이름은 __init__ 이 모은다."""

import math as _math

import numpy as _np

from ._tensor import (
    Tensor, _MinMax, _grad_mode,
)
from ._base import (
    _DEFAULT_DTYPE, _math, _np, _resolve, _unsupported, dtype,
)

# ---------------------------------------------------------------- 만들기

def tensor(data, dtype=None, requires_grad=False):
    if isinstance(data, Tensor):
        data = data.data
    return Tensor(_np.asarray(data, dtype=_resolve(data, dtype)), requires_grad)


def as_tensor(data, dtype=None):
    if isinstance(data, Tensor) and dtype is None:
        return data
    if isinstance(data, Tensor):
        return Tensor(data.data.astype(dtype.np))
    return tensor(data, dtype)


def from_numpy(arr):
    return Tensor(arr)


def zeros(*shape, dtype=None, requires_grad=False):
    shape = shape[0] if len(shape) == 1 and isinstance(shape[0], (tuple, list)) else shape
    return Tensor(_np.zeros(shape, dtype=(dtype.np if dtype else _DEFAULT_DTYPE)), requires_grad)


def ones(*shape, dtype=None, requires_grad=False, device=None):
    shape = shape[0] if len(shape) == 1 and isinstance(shape[0], (tuple, list)) else shape
    return Tensor(_np.ones(shape, dtype=(dtype.np if dtype else _DEFAULT_DTYPE)), requires_grad)


def full(shape, value, dtype=None):
    return Tensor(_np.full(shape, value, dtype=(dtype.np if dtype else _DEFAULT_DTYPE)))


def zeros_like(t, dtype=None):
    return Tensor(_np.zeros_like(t.data if isinstance(t, Tensor) else t))


def ones_like(t, dtype=None):
    return Tensor(_np.ones_like(t.data if isinstance(t, Tensor) else t))


def full_like(t, value):
    return Tensor(_np.full_like(t.data, value))


def arange(*args, dtype=None):
    return Tensor(_np.arange(*args, dtype=(dtype.np if dtype else None)))


def linspace(start, end, steps):
    return Tensor(_np.linspace(start, end, steps, dtype=_DEFAULT_DTYPE))


def eye(n):
    return Tensor(_np.eye(n, dtype=_DEFAULT_DTYPE))


_rng = _np.random.default_rng(0)


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


def manual_seed(seed):
    global _rng
    _rng = _np.random.default_rng(seed)
    return seed


def randn(*shape, requires_grad=False):
    shape = shape[0] if len(shape) == 1 and isinstance(shape[0], (tuple, list)) else shape
    return Tensor(_rng.standard_normal(shape).astype(_DEFAULT_DTYPE), requires_grad)


def rand(*shape):
    shape = shape[0] if len(shape) == 1 and isinstance(shape[0], (tuple, list)) else shape
    return Tensor(_rng.random(shape).astype(_DEFAULT_DTYPE))


def randint(low, high, shape):
    return Tensor(_rng.integers(low, high, shape).astype(_np.int64))


def randperm(n):
    return Tensor(_rng.permutation(n).astype(_np.int64))


def multinomial(probs, num_samples, replacement=True):
    p = probs.data / probs.data.sum(axis=-1, keepdims=True)
    if p.ndim == 1:
        return Tensor(_rng.choice(len(p), size=num_samples, p=p).astype(_np.int64))
    out = [_rng.choice(p.shape[-1], size=num_samples, p=row) for row in p]
    return Tensor(_np.asarray(out, dtype=_np.int64))


# ---------------------------------------------------------------- 함수

def _wrap(t):
    return t if isinstance(t, Tensor) else Tensor(_np.asarray(t))


def stack(items, dim=0):
    items = [_wrap(t) for t in items]
    out = _np.stack([t.data for t in items], axis=dim)
    if not items:
        return Tensor(out)
    return items[0]._make(
        out, tuple(items),
        lambda g: tuple(_np.take(_np.asarray(g), i, axis=dim) for i in range(len(items))),
        "StackBackward0")


def cat(items, dim=0):
    items = [_wrap(t) for t in items]
    out = _np.concatenate([t.data for t in items], axis=dim)
    sizes = [t.data.shape[dim] for t in items]

    def back(g):
        g = _np.asarray(g)
        cuts = _np.cumsum(sizes)[:-1]
        return tuple(_np.split(g, cuts, axis=dim))

    return items[0]._make(out, tuple(items), back, "CatBackward0")


def where(cond, a, b):
    c = cond.data if isinstance(cond, Tensor) else cond
    ta, tb = _wrap(a), _wrap(b)
    out = _np.where(c, ta.data, tb.data)
    return ta._make(out, (ta, tb), lambda g: (_np.where(c, g, 0), _np.where(c, 0, g)))


def sigmoid(t):
    out = 1.0 / (1.0 + _np.exp(-_np.clip(t.data, -60, 60)))
    return t._make(out, (t,), lambda g: (g * out * (1 - out),), "SigmoidBackward0")


def relu(t):
    return t._make(_np.maximum(t.data, 0), (t,), lambda g: (g * (t.data > 0),), "ReluBackward0")


def tanh(t):
    out = _np.tanh(t.data)
    return t._make(out, (t,), lambda g: (g * (1 - out * out),), "TanhBackward0")


def exp(t): return t.exp()
def log(t): return t.log()
def sqrt(t): return t.sqrt()
def abs(t): return t.abs()


def softmax(t, dim=-1):
    shifted = t.data - t.data.max(axis=dim, keepdims=True)
    e = _np.exp(shifted)
    out = e / e.sum(axis=dim, keepdims=True)

    def back(g):
        s = (g * out).sum(axis=dim, keepdims=True)
        return ((out * (g - s)),)

    return t._make(out, (t,), back, "SoftmaxBackward0")



def _pad2d(x, padding):
    if padding == 0:
        return x
    return _np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))


def _im2col(xd, KH, KW, stride):
    """(N,C,H,W) 를 (N*OH*OW, C*KH*KW) 로 편다. GEMM 한 번으로 합성곱을 끝내기 위한 것."""
    N, C, H, W = xd.shape
    OH = (H - KH) // stride + 1
    OW = (W - KW) // stride + 1
    win = _np.lib.stride_tricks.sliding_window_view(xd, (KH, KW), axis=(2, 3))
    win = win[:, :, ::stride, ::stride, :, :]          # (N, C, OH, OW, KH, KW)
    cols = win.transpose(0, 2, 3, 1, 4, 5)             # (N, OH, OW, C, KH, KW)
    return _np.ascontiguousarray(cols).reshape(N * OH * OW, C * KH * KW), OH, OW


def _col2im(gcols, shape, KH, KW, stride, OH, OW):
    """im2col 의 역. 출력 자리(OH×OW)가 아니라 **필터 자리(KH×KW)** 를 돈다 —
    28×28 이미지에서 784번 대신 9번이면 끝난다."""
    N, C, H, W = shape
    gx = _np.zeros(shape, dtype=gcols.dtype)
    g = gcols.reshape(N, OH, OW, C, KH, KW).transpose(0, 3, 4, 5, 1, 2)   # (N,C,KH,KW,OH,OW)
    for i in range(KH):
        for j in range(KW):
            gx[:, :, i:i + OH * stride:stride, j:j + OW * stride:stride] += g[:, :, i, j]
    return gx


def conv2d(x, weight, bias=None, stride=1, padding=0):
    """작은 입력용 합성곱. 26장에서 손으로 짠 이중 반복문과 같은 계산이다.

    im2col 로 펴서 행렬곱 한 번으로 끝낸다 — numpy 가 BLAS 를 부르므로,
    창을 돌며 einsum 하는 것보다 (실측) 20배 이상 빠르다.
    빠르다고 해도 실제 학습은 진짜 torch 로 한다.
    """
    xd = _pad2d(x.data, padding)
    wd = weight.data
    N, C, H, W = xd.shape
    F, C2, KH, KW = wd.shape
    if C != C2:
        raise RuntimeError(f"채널이 안 맞습니다: 입력 {C}, 필터 {C2}")
    if H < KH or W < KW:
        raise RuntimeError("필터가 입력보다 큽니다.")

    cols, OH, OW = _im2col(xd, KH, KW, stride)
    w2 = wd.reshape(F, -1)
    out = (cols @ w2.T).reshape(N, OH, OW, F).transpose(0, 3, 1, 2)

    def back(g):
        g = _np.asarray(g)
        g2 = g.transpose(0, 2, 3, 1).reshape(-1, F)
        gw = (g2.T @ cols).reshape(wd.shape)
        gx = _col2im(g2 @ w2, xd.shape, KH, KW, stride, OH, OW)
        if padding:
            gx = gx[:, :, padding:-padding, padding:-padding]
        return (gx, gw) if bias is None else (gx, gw, g.sum(axis=(0, 2, 3)))

    parents = (x, weight) if bias is None else (x, weight, bias)
    return x._make(out if bias is None else out + bias.data.reshape(1, -1, 1, 1), parents, back)


def max_pool2d(x, kernel_size, stride=None):
    stride = stride or kernel_size
    xd = x.data
    N, C, H, W = xd.shape
    OH = (H - kernel_size) // stride + 1
    OW = (W - kernel_size) // stride + 1
    win = _np.lib.stride_tricks.sliding_window_view(xd, (kernel_size, kernel_size), axis=(2, 3))
    win = win[:, :, ::stride, ::stride, :, :].reshape(N, C, OH, OW, -1)
    idx = win.argmax(axis=-1)
    out = _np.take_along_axis(win, idx[..., None], axis=-1).squeeze(-1)

    def back(g):
        # 최댓값이 있던 자리로만 기울기를 보낸다. 자리를 평평한 번호로 바꿔 한 번에 흩뿌린다 —
        # N·C·OH·OW 를 파이썬으로 도는 것보다 훨씬 빠르고, 결과는 같다.
        g = _np.asarray(g)
        di, dj = _np.divmod(idx, kernel_size)
        n_i, c_i, oh_i, ow_i = _np.ogrid[:N, :C, :OH, :OW]
        h = oh_i * stride + di
        w = ow_i * stride + dj
        flat = ((n_i * C + c_i) * H + h) * W + w
        gx = _np.zeros(xd.size, dtype=g.dtype)
        _np.add.at(gx, flat.reshape(-1), g.reshape(-1))
        return (gx.reshape(xd.shape),)

    return x._make(out, (x,), back)


def sin(t): return t._make(_np.sin(t.data), (t,), lambda g: (g * _np.cos(t.data),), "SinBackward0")
def cos(t): return t._make(_np.cos(t.data), (t,), lambda g: (-g * _np.sin(t.data),), "CosBackward0")


def clamp(t, min=None, max=None):
    out = _np.clip(t.data, min, max)
    inside = _np.ones_like(t.data, dtype=bool)
    if min is not None:
        inside &= t.data >= min
    if max is not None:
        inside &= t.data <= max
    return t._make(out, (t,), lambda g: (g * inside,), "ClampBackward0")



# ---------------------------------------------------------------- 원소별 함수
#
# 대부분 numpy 한 줄에 미분 한 줄이다. 미분이 없는 것(floor·sign 등)은 기울기를 0 으로 둔다 —
# torch 도 그렇게 한다. 계단 함수의 미분은 거의 모든 곳에서 0 이기 때문이다.

def _unary(name, forward, derivative=None, op=None):
    def fn(t):
        t = _wrap(t)
        out = forward(t.data)
        if derivative is None:
            return Tensor(out)
        return t._make(out, (t,), lambda g: (g * derivative(t.data, out),), op or f"{name}Backward0")
    fn.__name__ = name
    return fn


# erf 는 numpy 에 없다. `np.vectorize(math.erf)` 로 두면 **원소마다 파이썬을 부른다** —
# 벡터화가 아니라 반복문이고, 파이썬 호출이 비싼 wasm 에서 특히 나쁘다.
# Abramowitz & Stegun 7.1.26 을 numpy 원소별 연산으로 쓴다(절대오차 1.5e-7 — float32
# eps 1.19e-7 언저리라, float32 로 답하는 이 라이브러리에서는 자릿수 아래다).
_ERF_P = 0.3275911
_ERF_A = (0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429)


def _erfc_pos(y):
    """y >= 0 에서의 erfc. 다항식 × exp(-y²) 라 **뺄셈이 없다** — 이것이 원형이고,
    erf 는 여기서 유도한다. 반대로 하면(erf 를 원형으로 두면) 꼬리에서 자릿수가 날아간다."""
    t = 1.0 / (1.0 + _ERF_P * y)
    poly = t * (_ERF_A[0] + t * (_ERF_A[1] + t * (_ERF_A[2] + t * (_ERF_A[3] + t * _ERF_A[4]))))
    return poly * _np.exp(-y * y)


def _erf64(x):
    """float64 로 계산해서 돌려준다.

    float32 로 하면 원점 근처에서 `1 - (1 에 가까운 값)` 이 되어 유효숫자가 날아간다
    (실측: float32 로 계산하면 격자 4.6만 점 중 5,124 점이 allclose(1e-5) 를 깬다).
    """
    d = _np.asarray(x, dtype=_np.float64)
    return _np.sign(d) * (1.0 - _erfc_pos(_np.abs(d)))


def _one_plus_erf64(z):
    """1 + erf(z). z 가 크게 음수면 1 과 erf 가 상쇄되므로 그쪽은 erfc 로 바로 구한다 —
    gelu 의 왼쪽 꼬리가 정확히 그 자리다."""
    d = _np.asarray(z, dtype=_np.float64)
    tail = _erfc_pos(_np.abs(d))
    return _np.where(d >= 0, 2.0 - tail, tail)


log2 = _unary("Log2", _np.log2, lambda x, o: 1.0 / (x * _np.log(2)))
log10 = _unary("Log10", _np.log10, lambda x, o: 1.0 / (x * _np.log(10)))
rsqrt = _unary("Rsqrt", lambda x: 1.0 / _np.sqrt(x), lambda x, o: -0.5 * o / x)
square = _unary("Square", _np.square, lambda x, o: 2 * x)
reciprocal = _unary("Reciprocal", _np.reciprocal, lambda x, o: -o * o)
tan = _unary("Tan", _np.tan, lambda x, o: 1 + o * o)
sinh = _unary("Sinh", _np.sinh, lambda x, o: _np.cosh(x))
cosh = _unary("Cosh", _np.cosh, lambda x, o: _np.sinh(x))
erf = _unary("Erf", lambda x: _erf64(x).astype(x.dtype),
             lambda x, o: 2 / _np.sqrt(_np.pi) * _np.exp(-x * x))
# 계단 모양 — 미분이 거의 모든 곳에서 0 이다.
#
# **0 을 흘린다. 그래프를 끊지 않는다.** torch 에 물어보니 넷 다 `backward()` 가 돌고
# `.grad` 가 0 으로 채워진다. 전에는 맨 텐서를 돌려줘서 `backward()` 가 거절했는데,
# 그건 "없는 기능"이지 torch 와 같은 것이 아니었다. 값이 0 인 것과 부를 수 없는 것은
# 다르고, 계단 함수를 중간에 낀 손실은 실제로 torch 에서 돈다.
_zero_grad = lambda x, o: _np.zeros_like(x)                          # noqa: E731

sign = _unary("Sign", _np.sign, _zero_grad)
floor = _unary("Floor", _np.floor, _zero_grad)
ceil = _unary("Ceil", _np.ceil, _zero_grad)
round = _unary("Round", lambda x: _np.round(x), _zero_grad)


def neg(t): return -_wrap(t)
def pow(t, exponent): return _wrap(t) ** exponent


# ---- 삼각·지수·로그의 나머지
#
# 전부 원소별이고 도함수가 닫힌 꼴이라 `_unary` 로 끝난다. 하나씩 손으로 쓸 이유가 없다.
# **torch 의 별칭도 같이 단다** — `arccos` 는 `acos` 와 같은 함수이고, 튜토리얼이 둘 다
# 쓴다. 이름만 다르고 구현이 하나이므로 갈릴 자리가 없다.

acos = _unary("Acos", _np.arccos, lambda x, o: -1.0 / _np.sqrt(1 - x * x))
asin = _unary("Asin", _np.arcsin, lambda x, o: 1.0 / _np.sqrt(1 - x * x))
atan = _unary("Atan", _np.arctan, lambda x, o: 1.0 / (1 + x * x))
acosh = _unary("Acosh", _np.arccosh, lambda x, o: 1.0 / _np.sqrt(x * x - 1))
asinh = _unary("Asinh", _np.arcsinh, lambda x, o: 1.0 / _np.sqrt(x * x + 1))
atanh = _unary("Atanh", _np.arctanh, lambda x, o: 1.0 / (1 - x * x))
expm1 = _unary("Expm1", _np.expm1, lambda x, o: o + 1.0)
log1p = _unary("Log1p", _np.log1p, lambda x, o: 1.0 / (1 + x))
exp2 = _unary("Exp2", _np.exp2, lambda x, o: o * _np.log(2))
deg2rad = _unary("Deg2rad", _np.deg2rad, lambda x, o: _np.float32(_np.pi / 180))
rad2deg = _unary("Rad2deg", _np.rad2deg, lambda x, o: _np.float32(180 / _np.pi))
# 잘라내는 것 — 계단이라 0 을 흘린다(위 `floor` 와 같은 이유).
trunc = _unary("Trunc", _np.trunc, _zero_grad)
frac = _unary("Frac", lambda x: x - _np.trunc(x), lambda x, o: _np.ones_like(x))
# `sgn` 은 실수에서 `sign` 과 같다 — 0 을 흘린다.
#
# 처음에 "torch 는 sgn 역전파를 거절한다"고 적었는데 **틀렸다.** 예외가 `backward()`
# 가 아니라 결과를 찍던 내 `print` 에서 났고, 그것을 거절로 읽었다. torch 의 sgn 기울기는
# ZeroTensor(게으른 0 텐서)라 `.numpy()` 가 거절할 뿐 값은 0 이다.
sgn = _unary("Sgn", _np.sign, _zero_grad)
positive = _unary("Positive", lambda x: x, lambda x, o: _np.ones_like(x))
# `erfc = 1 - erf` 로 쓰면 꼬리에서 자릿수가 날아간다. **`_erfc_pos` 가 원형이므로**
# 거기서 직접 유도한다 — erf 를 그렇게 세운 이유가 바로 이것이다.
erfc = _unary("Erfc",
              lambda x: _np.where(x >= 0, _erfc_pos(_np.abs(_np.asarray(x, _np.float64))),
                                  2.0 - _erfc_pos(_np.abs(_np.asarray(x, _np.float64)))
                                  ).astype(x.dtype),
              lambda x, o: -2 / _np.sqrt(_np.pi) * _np.exp(-x * x))
sinc = _unary("Sinc", _np.sinc,
              # d/dx sinc(x) = (cos(πx) - sinc(x)) / x, x=0 에서는 0.
              lambda x, o: _np.where(x == 0, 0.0,
                                     (_np.cos(_np.pi * _np.where(x == 0, 1.0, x)) - o)
                                     / _np.where(x == 0, 1.0, x)))
logit = _unary("Logit", lambda x: _np.log(x / (1 - x)), lambda x, o: 1.0 / (x * (1 - x)))

# torch 의 별칭들. 같은 함수를 가리킨다.
arccos, arcsin, arctan = acos, asin, atan
arccosh, arcsinh, arctanh = acosh, asinh, atanh
fix = trunc
absolute = abs
negative = neg
clip = clamp


def _binary_math(name, forward, d_a, d_b, op=None):
    """두 텐서를 받는 원소별 함수. 브로드캐스팅과 역방향을 `_binary` 에 맡긴다.

    도함수는 `(x, y)` 를 받아 **기울기에 곱할 것**을 돌려준다. `_binary` 가 넘겨주는
    서명은 `(g, x, y)` 이므로 여기서 감싼다.
    """
    def fn(a, b):
        a = _wrap(a)
        return a._binary(b, forward,
                         lambda g, x, y: g * d_a(x, y),
                         lambda g, x, y: g * d_b(x, y),
                         op or f"{name}Backward0")
    fn.__name__ = name
    return fn


atan2 = _binary_math("Atan2", _np.arctan2,
                     lambda x, y: y / (x * x + y * y),
                     lambda x, y: -x / (x * x + y * y))
hypot = _binary_math("Hypot", _np.hypot,
                     lambda x, y: x / _np.hypot(x, y),
                     lambda x, y: y / _np.hypot(x, y))
# |x|·sign(y) 이므로 x 로는 sign(x)·sign(y), y 로는 0 이다(계단).
copysign = _binary_math("Copysign", _np.copysign,
                        lambda x, y: _np.sign(x) * _np.sign(y),
                        lambda x, y: _np.zeros_like(_np.copysign(x, y)))
logaddexp = _binary_math("Logaddexp", _np.logaddexp,
                         lambda x, y: _np.exp(x - _np.logaddexp(x, y)),
                         lambda x, y: _np.exp(y - _np.logaddexp(x, y)))
logaddexp2 = _binary_math("Logaddexp2", _np.logaddexp2,
                          lambda x, y: _np.exp2(x - _np.logaddexp2(x, y)),
                          lambda x, y: _np.exp2(y - _np.logaddexp2(x, y)))


def xlogy(a, b):
    """`x · log(y)` 인데 **x 가 0 이면 0 이다** — `0 · log(0)` 을 nan 으로 두지 않는다."""
    a = _wrap(a)
    with _np.errstate(divide="ignore", invalid="ignore"):
        return a._binary(
            b,
            lambda x, y: _np.where(x == 0, 0.0, x * _np.log(y)),
            lambda g, x, y: g * _np.where(x == 0, 0.0, _np.log(y)),
            lambda g, x, y: g * _np.where(x == 0, 0.0, x / y),
            "XlogyBackward0")


def signbit(t):
    return Tensor(_np.signbit(_wrap(t).data))


def heaviside(t, values):
    t, v = _wrap(t), _wrap(values)
    return Tensor(_np.heaviside(t.data, v.data))


def ldexp(t, other):
    t, o = _wrap(t), _wrap(other)
    return t * Tensor(_np.exp2(o.data.astype(t.data.dtype)))


# ---------------------------------------------------------------- 비교

def _compare(name, fn):
    def cmp(a, b):
        a = _wrap(a)
        bd = b.data if isinstance(b, Tensor) else b
        return Tensor(fn(a.data, bd))
    cmp.__name__ = name
    return cmp


eq = _compare("eq", _np.equal)
ne = _compare("ne", _np.not_equal)
lt = _compare("lt", _np.less)
le = _compare("le", _np.less_equal)
gt = _compare("gt", _np.greater)
ge = _compare("ge", _np.greater_equal)
logical_and = _compare("logical_and", _np.logical_and)
logical_or = _compare("logical_or", _np.logical_or)
isnan = _unary("IsNan", _np.isnan)
isinf = _unary("IsInf", _np.isinf)


def logical_not(t): return Tensor(_np.logical_not(_wrap(t).data))


def maximum(a, b):
    a, b = _wrap(a), _wrap(b)
    pick = a.data >= b.data
    return a._make(_np.maximum(a.data, b.data), (a, b),
                   lambda g: (g * pick, g * ~pick), "MaximumBackward0")


def minimum(a, b):
    a, b = _wrap(a), _wrap(b)
    pick = a.data <= b.data
    return a._make(_np.minimum(a.data, b.data), (a, b),
                   lambda g: (g * pick, g * ~pick), "MinimumBackward0")


# ---------------------------------------------------------------- 모양·선택

def split(t, size, dim=0):
    t = _wrap(t)
    n = t.data.shape[dim]
    sizes = size if isinstance(size, (list, tuple)) else \
        [size] * (n // size) + ([n % size] if n % size else [])
    cuts, out, start = [], [], 0
    for sz in sizes[:-1]:
        start += sz
        cuts.append(start)
    return tuple(t[_slice_at(dim, s, e)] for s, e in zip([0] + cuts, cuts + [n]))


def chunk(t, chunks, dim=0):
    t = _wrap(t)
    n = t.data.shape[dim]
    size = -(-n // chunks)
    return split(t, size, dim)


def _slice_at(dim, start, end):
    return tuple(slice(None) for _ in range(dim)) + (slice(start, end),)


def unbind(t, dim=0):
    t = _wrap(t)
    return tuple(t[_slice_at(dim, i, i + 1)].squeeze(dim) for i in range(t.data.shape[dim]))


def narrow(t, dim, start, length):
    return _wrap(t)[_slice_at(dim, start, start + length)]


def flip(t, dims):
    t = _wrap(t)
    dims = (dims,) if isinstance(dims, int) else tuple(dims)
    return t._make(_np.flip(t.data, dims).copy(), (t,),
                   lambda g: (_np.flip(_np.asarray(g), dims).copy(),), "FlipBackward0")


def roll(t, shifts, dims=None):
    t = _wrap(t)
    return t._make(_np.roll(t.data, shifts, dims), (t,),
                   lambda g: (_np.roll(_np.asarray(g), _negate(shifts), dims),), "RollBackward0")


def _negate(shifts):
    return -shifts if isinstance(shifts, int) else tuple(-s for s in shifts)


def index_select(t, dim, index):
    t = _wrap(t)
    idx = index.data.astype(int) if isinstance(index, Tensor) else _np.asarray(index, dtype=int)
    return t[_index_at(dim, idx)]


def _index_at(dim, idx):
    return tuple(slice(None) for _ in range(dim)) + (idx,)


def masked_select(t, mask):
    t = _wrap(t)
    m = mask.data.astype(bool) if isinstance(mask, Tensor) else _np.asarray(mask, dtype=bool)
    return t[m]


def gather(t, dim, index):
    """index 가 가리키는 자리를 뽑는다. 분류에서 정답 클래스의 확률을 꺼낼 때 쓴다."""
    t = _wrap(t)
    idx = index.data.astype(int) if isinstance(index, Tensor) else _np.asarray(index, dtype=int)
    out = _np.take_along_axis(t.data, idx, axis=dim)
    shape = t.data.shape

    def back(g):
        z = _np.zeros(shape, dtype=_np.asarray(g).dtype)
        _np.put_along_axis(z, idx, _np.asarray(g), axis=dim)
        return (z,)

    return t._make(out, (t,), back, "GatherBackward0")


def repeat_interleave(t, repeats, dim=None):
    """제자리에서 늘린다. 역방향은 늘어난 것들을 **묶음마다 도로 합치는** 것이다."""
    t = _wrap(t)
    out = _np.repeat(t.data, repeats, axis=dim)
    length = t.data.size if dim is None else t.data.shape[dim]
    counts = (_np.full(length, repeats, dtype=_np.int64) if isinstance(repeats, int)
              else _np.asarray(repeats, dtype=_np.int64))
    # **`intp` 로 준다.** numpy 의 기본 정수는 C 의 `long` 이라 64비트 맥·리눅스에서는
    # int64 지만 wasm32(Pyodide)에서는 32비트이고, `reduceat` 은 색인 배열을 `intp` 로
    # 요구한다 — 안 맞추면 **브라우저에서만** TypeError 다. 이 저장소에서 세 번째로
    # 같은 자리에 걸렸고, 네이티브 검사로는 절대 안 나온다.
    starts = _np.concatenate(([0], _np.cumsum(counts)[:-1])).astype(_np.intp)
    axis = 0 if dim is None else dim

    def back(g):
        gg = _np.asarray(g)
        if dim is None:
            gg = gg.reshape(-1)
        return (_np.add.reduceat(gg, starts, axis=axis).reshape(t.data.shape),)

    return t._make(out, (t,), back, "RepeatInterleaveBackward0")


def tile(t, reps):
    """통째로 반복해 붙인다. 역방향은 **반복된 조각들을 겹쳐 더하는** 것이다.

    축마다 출력이 (반복수 × 원래길이) 이므로, 그 축을 둘로 쪼개 반복 쪽만 더하면 된다.
    """
    t = _wrap(t)
    reps_t = (reps,) if isinstance(reps, int) else tuple(reps)
    out = _np.tile(t.data, reps_t)
    src = t.data.shape
    nd = max(len(src), len(reps_t))
    src_p = (1,) * (nd - len(src)) + src
    reps_p = (1,) * (nd - len(reps_t)) + reps_t

    def back(g):
        split = []
        for r, s in zip(reps_p, src_p):
            split += [r, s]
        gg = _np.asarray(g).reshape(split).sum(axis=tuple(range(0, 2 * nd, 2)))
        return (gg.reshape(src),)

    return t._make(out, (t,), back, "TileBackward0")


def movedim(t, source, destination):
    t = _wrap(t)
    return t._make(_np.moveaxis(t.data, source, destination), (t,),
                   lambda g: (_np.moveaxis(_np.asarray(g), destination, source),),
                   "MovedimBackward0")


# ---------------------------------------------------------------- 축약(추가)

def prod(t, dim=None):
    t = _wrap(t)
    out = _np.prod(t.data, axis=dim)
    return t._make(out, (t,), lambda g: (_np.asarray(g) * out / t.data,), "ProdBackward0")


def median(t, dim=None):
    """torch 는 원소가 짝수일 때 **가운데 둘 중 작은 쪽**을 준다. numpy 는 평균을 낸다 —
    그대로 쓰면 조용히 다른 값이 나온다."""
    t = _wrap(t)
    if dim is None:
        flat = t.data.reshape(-1)
        pick = int(_np.argsort(flat)[(flat.size - 1) // 2])

        # 기울기는 **뽑힌 그 자리 하나로만** 간다. 중앙값은 고른 원소를 그대로 내놓는
        # 연산이라, 나머지 원소를 조금 흔들어도 답이 안 움직인다.
        def back(g):
            z = _np.zeros_like(flat)
            z[pick] = _np.asarray(g)
            return (z.reshape(t.data.shape),)

        return t._make(flat[pick], (t,), back, "MedianBackward0")

    order = _np.argsort(t.data, axis=dim)
    idx = (t.data.shape[dim] - 1) // 2
    take = _np.take(order, idx, axis=dim)
    at = _np.expand_dims(take, dim)
    picked = _np.take_along_axis(t.data, at, axis=dim).squeeze(dim)

    def back_dim(g):
        z = _np.zeros_like(t.data)
        _np.put_along_axis(z, at, _np.expand_dims(_np.asarray(g), dim), axis=dim)
        return (z,)

    return _MinMax(t._make(picked, (t,), back_dim, "MedianBackward0"), Tensor(take))


def norm(t, p=2, dim=None):
    t = _wrap(t)
    if p == 1:
        return t.abs().sum(dim=dim)
    return (t * t).sum(dim=dim) ** 0.5


def cumsum(t, dim):
    t = _wrap(t)
    return t._make(_np.cumsum(t.data, axis=dim), (t,),
                   lambda g: (_np.flip(_np.cumsum(_np.flip(_np.asarray(g), dim), axis=dim), dim),),
                   "CumsumBackward0")


def cumprod(t, dim):
    """누적 곱. 역방향을 **나눗셈 없이** 쓴다.

    흔한 유도는 `dL/dx_k = (1/x_k) * sum_{j>=k} g_j y_j` 인데, 입력에 0 이 있으면
    거기서 나눗셈이 터져 조용히 `nan` 이 흐른다. 예외도 안 난다. 그래서 각 k 마다
    `x_k` 를 뺀 곱을 직접 쌓는다 — 길이의 제곱만큼 걸리지만 `cumprod` 는 학습 경로의
    안쪽이 아니고, **0 이 섞였을 때 답이 맞는 쪽**이 이 저장소의 기준이다.
    """
    t = _wrap(t)
    out = _np.cumprod(t.data, axis=dim)

    def back(g):
        x = _np.moveaxis(t.data, dim, 0)
        gg = _np.moveaxis(_np.asarray(g), dim, 0)
        grad = _np.zeros_like(x, dtype=_np.result_type(x.dtype, _np.float32))
        prefix = _np.ones_like(x[0])                 # x_0 … x_{k-1}
        for k in range(x.shape[0]):
            run = prefix.copy()                      # j=k 일 때의 곱 (x_k 를 뺀 것)
            acc = gg[k] * run
            for j in range(k + 1, x.shape[0]):
                run = run * x[j]
                acc = acc + gg[j] * run
            grad[k] = acc
            prefix = prefix * x[k]
        return (_np.moveaxis(grad, 0, dim),)

    return t._make(out, (t,), back, "CumprodBackward0")


def count_nonzero(t, dim=None):
    return Tensor(_np.count_nonzero(_wrap(t).data, axis=dim))


def _pick(t, idx, dim, op):
    """뽑은 값에 **기울기 길을 남긴다.** 뽑기만 하고 끊으면 학습이 조용히 멈춘다 —
    top-k 샘플링이나 정렬을 끼운 손실에서 그 일이 난다."""
    values = _np.take_along_axis(t.data, idx, axis=dim)
    shape = t.data.shape

    def back(g):
        z = _np.zeros(shape, dtype=_np.asarray(g).dtype)
        _np.put_along_axis(z, idx, _np.asarray(g), axis=dim)
        return (z,)

    return t._make(values, (t,), back, op)


def topk(t, k, dim=-1, largest=True):
    """상위 k개의 (값, 번호). 32장의 top-k 샘플링이 이것이다."""
    t = _wrap(t)
    order = _np.argsort(t.data, axis=dim)
    if largest:
        order = _np.flip(order, axis=dim)
    idx = _np.take(order, _np.arange(k), axis=dim)
    return _MinMax(_pick(t, idx, dim, "TopkBackward0"), Tensor(idx))


def sort(t, dim=-1, descending=False):
    t = _wrap(t)
    idx = _np.argsort(t.data, axis=dim)
    if descending:
        idx = _np.flip(idx, axis=dim)
    return _MinMax(_pick(t, idx, dim, "SortBackward0"), Tensor(idx))


def argsort(t, dim=-1, descending=False):
    return sort(t, dim, descending).indices


def unique(t, sorted=True, return_counts=False):
    values, counts = _np.unique(_wrap(t).data, return_counts=True)
    return (Tensor(values), Tensor(counts)) if return_counts else Tensor(values)


# ---------------------------------------------------------------- 선형대수

def mm(a, b): return _wrap(a) @ _wrap(b)
def bmm(a, b): return _wrap(a) @ _wrap(b)


def dot(a, b): return (_wrap(a) * _wrap(b)).sum()


def outer(a, b):
    a, b = _wrap(a), _wrap(b)
    return a.reshape(-1, 1) @ b.reshape(1, -1)


def _diagonal_scatter(shape, g):
    """대각선 위에 `g` 를 얹은 영행렬. `diag`·`trace` 의 역방향이 같은 모양이다."""
    z = _np.zeros(shape, dtype=_np.asarray(g).dtype)
    n = min(shape)
    z[_np.arange(n), _np.arange(n)] = g
    return z


def diag(t):
    """1차원이면 대각행렬을 만들고, 2차원이면 대각선을 뽑는다 — 방향이 반대라
    역방향도 반대다."""
    t = _wrap(t)
    out = _np.diag(t.data)
    if t.data.ndim == 1:
        def back(g):
            return (_np.diag(_np.asarray(g)),)
    else:
        def back(g):
            return (_diagonal_scatter(t.data.shape, _np.asarray(g)),)
    return t._make(out, (t,), back, "DiagBackward0")


def trace(t):
    t = _wrap(t)
    return t._make(_np.trace(t.data), (t,),
                   lambda g: (_diagonal_scatter(t.data.shape, _np.asarray(g)),),
                   "TraceBackward0")


def einsum(equation, *operands):
    """역방향도 einsum 이다 — 출력 첨자를 항의 자리에 바꿔 넣으면 그 항의 기울기가 나온다.

    한 가지 걸리는 자리가 있다. 어떤 첨자가 **그 항에만** 있고 출력에도 다른 항에도
    없으면(`ij->i` 의 `j`), einsum 은 없던 축을 만들지 못한다. 그럴 때는 그 축만큼의
    1 로 채운 항을 하나 더 끼워 넣는다 — 값은 안 바뀌고 축만 생긴다.

    `...` 과 한 항 안의 반복 첨자(`ii->i`)는 이 규칙이 그대로 안 통한다. 그래서 **틀린
    기울기를 주는 대신 기울기를 안 준다** — 그 경우 `backward()` 가 거절한다.
    """
    ops = [_wrap(o) for o in operands]
    out = _np.einsum(equation, *[o.data for o in ops])
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
        return Tensor(out)

    def back(g):
        g = _np.asarray(g)
        grads = []
        for i, mine in enumerate(subs):
            rest = [(subs[j], ops[j].data) for j in range(len(subs)) if j != i]
            known = set(rhs) | {c for s, _ in rest for c in s}
            missing = [c for c in mine if c not in known]
            spec = [rhs] + [s for s, _ in rest]
            terms = [g] + [d for _, d in rest]
            if missing:
                sizes = [ops[i].data.shape[mine.index(c)] for c in missing]
                spec.append("".join(missing))
                terms.append(_np.ones(sizes, dtype=ops[i].data.dtype))
            grads.append(_np.einsum(",".join(spec) + "->" + mine, *terms))
        return tuple(grads)

    return ops[0]._make(out, tuple(ops), back, "EinsumBackward0")


def empty(*shape, dtype=None):
    return zeros(*shape, dtype=dtype)




def leaky_relu(t, negative_slope=0.01):
    t = _wrap(t)
    pick = t.data > 0
    return t._make(_np.where(pick, t.data, negative_slope * t.data), (t,),
                   lambda g: (g * _np.where(pick, 1.0, negative_slope),), "LeakyReluBackward0")


def elu(t, alpha=1.0):
    t = _wrap(t)
    pick = t.data > 0
    out = _np.where(pick, t.data, alpha * (_np.exp(_np.minimum(t.data, 0)) - 1))
    return t._make(out, (t,), lambda g: (g * _np.where(pick, 1.0, out + alpha),),
                   "EluBackward0")


def silu(t):
    """x·σ(x). Swish 라고도 한다."""
    t = _wrap(t)
    sig = 1.0 / (1.0 + _np.exp(-_np.clip(t.data, -60, 60)))
    return t._make(t.data * sig, (t,),
                   lambda g: (g * (sig * (1 + t.data * (1 - sig))),), "SiluBackward0")


def _gelu(t):
    """torch 의 기본 gelu(정확형)와 같은 식 — 0.5·x·(1 + erf(x/√2)).

    순·역방향 모두 `np.vectorize` 였다. 원소마다 파이썬을 부르는 것이라
    8×32×2048 한 번에 197ms 가 걸렸고, numpy 원소별로 바꾸니 9.9ms 다(실측, 20배).
    진짜 torch 와의 최대차는 4.77e-07 로 바꾸기 전과 **같다**
    (x ∈ [-8, 8] 에 꼬리를 더한 4.6만 점, allclose(1e-5) 전부 통과).

    **`nn` 쪽에 있던 것을 여기로 옮겼다.** 트랜스포머 층이 이것을 쓰지만 `gelu` 도
    쓰고, `gelu` 는 이 위쪽에 있다. 파일 하나일 때는 순서가 안 보였는데 쪼개려니
    아래에서 위를 부르는 모양이 드러났다 — 층위가 뒤집혀 있던 것을 바로잡는다.
    """
    d = _np.asarray(t.data, dtype=_np.float64)
    ope = _one_plus_erf64(d / _math.sqrt(2.0))
    out = (0.5 * d * ope).astype(t.data.dtype)

    def back(g):
        grad = 0.5 * ope + d * _np.exp(-d * d / 2) / _math.sqrt(2 * _math.pi)
        return (g * grad.astype(t.data.dtype),)

    return t._make(out, (t,), back, "GeluBackward0")


def gelu(t):
    return _gelu(_wrap(t))


def log_softmax(t, dim=-1):
    t = _wrap(t)
    shifted = t.data - t.data.max(axis=dim, keepdims=True)
    out = shifted - _np.log(_np.exp(shifted).sum(axis=dim, keepdims=True))
    soft = _np.exp(out)

    def back(g):
        g = _np.asarray(g)
        return (g - soft * g.sum(axis=dim, keepdims=True),)

    return t._make(out, (t,), back, "LogSoftmaxBackward0")


def dropout(t, p=0.5, training=True):
    if not training or p == 0:
        return _wrap(t)
    t = _wrap(t)
    mask = (_rng.random(t.data.shape) > p).astype(t.data.dtype) / (1 - p)
    return t * Tensor(mask)


def avg_pool2d(x, kernel_size, stride=None):
    stride = stride or kernel_size
    xd = x.data
    N, C, H, W = xd.shape
    OH = (H - kernel_size) // stride + 1
    OW = (W - kernel_size) // stride + 1
    win = _np.lib.stride_tricks.sliding_window_view(xd, (kernel_size, kernel_size), axis=(2, 3))
    win = win[:, :, ::stride, ::stride, :, :]
    out = win.mean(axis=(4, 5))
    area = kernel_size * kernel_size

    def back(g):
        g = _np.asarray(g) / area
        gx = _np.zeros_like(xd)
        for i in range(kernel_size):
            for j in range(kernel_size):
                gx[:, :, i:i + OH * stride:stride, j:j + OW * stride:stride] += g
        return (gx,)

    return x._make(out, (x,), back, "AvgPool2DBackward0")


def _pool_all(x):
    """AdaptiveAvgPool2d(1) 만 지원한다 — 흔한 것은 그것뿐이고, 나머지는 거절한다."""
    return x.mean(dim=2).mean(dim=2).reshape(x.data.shape[0], x.data.shape[1], 1, 1)


def layer_norm(x, normalized_shape, weight=None, bias=None, eps=1e-5):
    mean = x.mean(dim=-1, keepdim=True)
    centered = x - mean
    var = (centered * centered).mean(dim=-1, keepdim=True)
    out = centered / (var + eps) ** 0.5
    if weight is not None:
        out = out * weight
    return out + bias if bias is not None else out


def embedding(idx, weight):
    ids = idx.data.astype(int)
    dim = weight.data.shape[1]
    out = weight.data[ids]

    def back(g):
        gw = _np.zeros_like(weight.data)
        _np.add.at(gw, ids.reshape(-1), _np.asarray(g).reshape(-1, dim))
        return (gw,)

    return weight._make(out, (weight,), back, "EmbeddingBackward0")


def nll_loss(log_probs, target):
    n = log_probs.data.shape[0]
    picked = log_probs[_np.arange(n), target.data.astype(int)]
    return -picked.mean()


def l1_loss(pred, target):
    return (pred - target).abs().mean()


def smooth_l1_loss(pred, target, beta=1.0):
    """작은 오차는 제곱, 큰 오차는 절댓값. 이상치에 덜 흔들린다."""
    diff = pred - target
    small = _np.abs(diff.data) < beta
    return (where(Tensor(small), 0.5 * diff * diff / beta, diff.abs() - 0.5 * beta)).mean()


def pad(x, padding, value=0.0):
    """마지막 차원부터 (앞, 뒤) 순으로 받는다 — torch 의 규칙이다."""
    x = _wrap(x)
    pairs = [(0, 0)] * x.data.ndim
    for i in range(0, len(padding), 2):
        pairs[-(i // 2 + 1)] = (padding[i], padding[i + 1])
    out = _np.pad(x.data, pairs, constant_values=value)
    cuts = tuple(slice(a, s - b if b else None) for (a, b), s in zip(pairs, out.shape))
    return x._make(out, (x,), lambda g: (_np.asarray(g)[cuts],), "PadBackward0")


def normalize(x, p=2, dim=1, eps=1e-12):
    x = _wrap(x)
    denom = norm(x, p=p, dim=dim)
    return x / maximum(denom.unsqueeze(dim), Tensor(_np.array(eps, dtype=_DEFAULT_DTYPE)))


def cosine_similarity(a, b, dim=1, eps=1e-8):
    a, b = _wrap(a), _wrap(b)
    return (a * b).sum(dim=dim) / maximum(
        norm(a, dim=dim) * norm(b, dim=dim), Tensor(_np.array(eps, dtype=_DEFAULT_DTYPE)))



def tril(t, diagonal=0):
    """아래 삼각만 남긴다. 역방향은 **같은 자리만 통과시키는** 것이다 — 지운 자리는
    출력에 안 나타났으니 기울기도 0 이다."""
    t = _wrap(t)
    return t._make(_np.tril(t.data, k=diagonal), (t,),
                   lambda g: (_np.tril(_np.asarray(g), k=diagonal),), "TrilBackward0")


def triu(t, diagonal=0):
    t = _wrap(t)
    return t._make(_np.triu(t.data, k=diagonal), (t,),
                   lambda g: (_np.triu(_np.asarray(g), k=diagonal),), "TriuBackward0")


def allclose(a, b, rtol=1e-5, atol=1e-8):
    return bool(_np.allclose(_wrap(a).data, _wrap(b).data, rtol=rtol, atol=atol))


def equal(a, b):
    return bool(_np.array_equal(_wrap(a).data, _wrap(b).data))


def isfinite(t):
    return Tensor(_np.isfinite(_wrap(t).data))


def bincount(t):
    # `intp` 다 — wasm32 에서 int64 를 주면 거절한다. 위 `repeat_interleave` 참고.
    return Tensor(_np.bincount(_wrap(t).data.astype(_np.intp)))


def _to_plain(obj):
    """텐서를 numpy 로 바꿔 저장 가능한 형태로. 중첩 dict/list 도 따라간다."""
    if isinstance(obj, Tensor):
        return {"__tensor__": obj.data}
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_plain(v) for v in obj)
    return obj


def _from_plain(obj):
    if isinstance(obj, dict):
        if "__tensor__" in obj:
            return Tensor(obj["__tensor__"])
        return {k: _from_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_from_plain(v) for v in obj)
    return obj


def save(obj, path):
    """진짜 torch 와 달리 pickle 한 겹만 쓴다. 브라우저에도 가상 파일시스템이 있어 경로가 통한다."""
    import pickle
    with open(path, "wb") as f:
        pickle.dump(_to_plain(obj), f)


def load(path, **kwargs):
    import pickle
    with open(path, "rb") as f:
        return _from_plain(pickle.load(f))


class no_grad:
    def __enter__(self):
        self._prev = _grad_mode.enabled
        _grad_mode.enabled = False
        return self

    def __exit__(self, *exc):
        _grad_mode.enabled = self._prev
        return False

    def __call__(self, fn):
        def wrapper(*a, **k):
            with no_grad():
                return fn(*a, **k)
        return wrapper


class _Namespace:
    """torch 의 하위 모듈 자리(`torch.nn`, `torch.optim.lr_scheduler` …).

    파이썬 모듈이 아니라 객체지만, `install()` 이 이것을 훑어 `sys.modules` 에 심어주면
    `from torch.optim.lr_scheduler import StepLR` 같은 import 가 그대로 통한다.
    상속만이 표시다 — 여기 들어오지 않은 자리는 import 경로가 안 생긴다.
    """


class _Cuda(_Namespace):
    @staticmethod
    def is_available():
        return False

    @staticmethod
    def manual_seed_all(seed):
        return None

    @staticmethod
    def synchronize():
        _unsupported("torch.cuda.synchronize")


cuda = _Cuda()


