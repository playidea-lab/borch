"""borch 를 쪼갠 조각. 공개 이름은 __init__ 이 모은다."""

import builtins as _builtins
import math as _math

import numpy as _np

from ._tensor import (
    Tensor, _MinMax, _grad_mode, _unbroadcast,
)
from ._base import (
    _DEFAULT_DTYPE, _math, _np, _resolve, _unsupported, Size, dtype,
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



def _pair(v):
    """`3` 도 `(3, 3)` 도 받는다 — torch 가 그렇다."""
    return (v, v) if isinstance(v, int) else tuple(v)


def _pad2d(x, padding):
    ph, pw = _pair(padding)
    if ph == 0 and pw == 0:
        return x
    return _np.pad(x, ((0, 0), (0, 0), (ph, ph), (pw, pw)))


def _im2col(xd, KH, KW, stride):
    """(N,C,H,W) 를 (N*OH*OW, C*KH*KW) 로 편다. GEMM 한 번으로 합성곱을 끝내기 위한 것."""
    sh, sw = _pair(stride)
    N, C, H, W = xd.shape
    OH = (H - KH) // sh + 1
    OW = (W - KW) // sw + 1
    win = _np.lib.stride_tricks.sliding_window_view(xd, (KH, KW), axis=(2, 3))
    win = win[:, :, ::sh, ::sw, :, :]                  # (N, C, OH, OW, KH, KW)
    cols = win.transpose(0, 2, 3, 1, 4, 5)             # (N, OH, OW, C, KH, KW)
    return _np.ascontiguousarray(cols).reshape(N * OH * OW, C * KH * KW), OH, OW


def _col2im(gcols, shape, KH, KW, stride, OH, OW):
    """im2col 의 역. 출력 자리(OH×OW)가 아니라 **필터 자리(KH×KW)** 를 돈다 —
    28×28 이미지에서 784번 대신 9번이면 끝난다."""
    sh, sw = _pair(stride)
    N, C, H, W = shape
    gx = _np.zeros(shape, dtype=gcols.dtype)
    g = gcols.reshape(N, OH, OW, C, KH, KW).transpose(0, 3, 4, 5, 1, 2)   # (N,C,KH,KW,OH,OW)
    for i in range(KH):
        for j in range(KW):
            gx[:, :, i:i + OH * sh:sh, j:j + OW * sw:sw] += g[:, :, i, j]
    return gx


def conv2d(x, weight, bias=None, stride=1, padding=0):
    """작은 입력용 합성곱. 26장에서 손으로 짠 이중 반복문과 같은 계산이다.

    im2col 로 펴서 행렬곱 한 번으로 끝낸다 — numpy 가 BLAS 를 부르므로,
    창을 돌며 einsum 하는 것보다 (실측) 20배 이상 빠르다.
    빠르다고 해도 실제 학습은 진짜 torch 로 한다.

    **stride·padding 을 축마다 다르게 받는다.** 정사각만 받으면 `conv1d` 를 이 위에
    얹을 수 없다 — 1차원은 높이를 1 로 끼워 넣는 것이라 높이 쪽 padding 이 0 이어야 한다.
    자매가 이미 축별로 받으므로 거기에 맞춘다.
    """
    xd = _pad2d(x.data, padding)
    wd = weight.data
    ph, pw = _pair(padding)
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
        if ph:
            gx = gx[:, :, ph:-ph, :]
        if pw:
            gx = gx[:, :, :, pw:-pw]
        return (gx, gw) if bias is None else (gx, gw, g.sum(axis=(0, 2, 3)))

    parents = (x, weight) if bias is None else (x, weight, bias)
    return x._make(out if bias is None else out + bias.data.reshape(1, -1, 1, 1), parents, back)


def conv_transpose2d(x, weight, bias=None, stride=1, padding=0):
    """전치 합성곱 — **`conv2d` 가 입력 쪽으로 흘리는 것과 같은 계산이다.**

    새 커널을 안 쓴다. `conv2d` 의 역방향이 이미 `col2im` 으로 그 일을 하고 있고,
    같은 계산을 두 벌 두면 한쪽만 고쳐진 채 갈리는 날이 온다.

    **가중치 축이 `conv2d` 와 뒤집혀 있다** — `(입력, 출력, kh, kw)` 다. 정사각
    커널이면 뒤집어 놓아도 모양이 맞으므로 값으로만 갈린다. 이 층에서 가장 흔한 실수다.
    """
    x, weight = _wrap(x), _wrap(weight)
    N, C, H, W = x.data.shape
    C2, F, KH, KW = weight.data.shape
    if C != C2:
        raise RuntimeError(f"채널이 안 맞습니다: 입력 {C}, 필터 {C2}")
    sh, sw = _pair(stride)
    ph, pw = _pair(padding)
    OH = (H - 1) * sh + KH
    OW = (W - 1) * sw + KW

    # 출력 자리마다 입력 한 칸을 커널만큼 흩뿌린다. `col2im` 이 정확히 그 모양이다 —
    # 입력을 `(N·H·W, C)` 로 펴고 가중치와 곱해 `(N·H·W, F·KH·KW)` 로 만든 뒤 접는다.
    cols = x.data.transpose(0, 2, 3, 1).reshape(N * H * W, C)
    w2 = weight.data.reshape(C, F * KH * KW)
    spread = cols @ w2
    out = _col2im(spread, (N, F, OH, OW), KH, KW, (sh, sw), H, W)

    def back(g):
        g = _np.asarray(g)
        gcols, _, _ = _im2col(g, KH, KW, (sh, sw))          # (N·H·W, F·KH·KW)
        gx = (gcols @ w2.T).reshape(N, H, W, C).transpose(0, 3, 1, 2)
        gw = (cols.T @ gcols).reshape(weight.data.shape)
        got = (gx, gw)
        return got if bias is None else got + (g.sum(axis=(0, 2, 3)),)

    if ph or pw:
        # 채움은 **출력에서 잘라낸다** — 보통 합성곱과 반대 방향이다.
        out = out[:, :, ph:OH - ph, pw:OW - pw]

        def back(g, _inner=back, _ph=ph, _pw=pw, _oh=OH, _ow=OW):   # noqa: F811
            full = _np.zeros((N, F, _oh, _ow), dtype=_np.asarray(g).dtype)
            full[:, :, _ph:_oh - _ph, _pw:_ow - _pw] = g
            return _inner(full)

    if bias is not None:
        out = out + bias.data.reshape(1, -1, 1, 1)
    parents = (x, weight) if bias is None else (x, weight, bias)
    return x._make(out, parents, back, "ConvTranspose2DBackward0")


def conv_transpose1d(x, weight, bias=None, stride=1, padding=0):
    """`conv_transpose2d` 에 높이 1 을 끼워 넣는다 — `conv1d` 와 같은 방식이다."""
    x, weight = _wrap(x), _wrap(weight)
    n, c, length = x.data.shape
    c2, f, k = weight.data.shape
    out = conv_transpose2d(x.reshape(n, c, 1, length), weight.reshape(c2, f, 1, k),
                           bias, (1, stride), (0, padding))
    shape = out.data.shape
    return out.reshape(shape[0], shape[1], shape[3])


def conv_transpose3d(x, weight, bias=None, stride=1, padding=0):
    """깊이마다 2차원 전치 합성곱을 돌려 **겹치는 자리에 더한다.**

    `conv3d` 와 같은 방식이다 — 3차원 커널을 따로 쓰지 않으므로 새로 쓸 미분식이 없다.
    """
    x, weight = _wrap(x), _wrap(weight)
    n, c, d, h, w = x.data.shape
    c2, f, kd, kh, kw = weight.data.shape
    if c != c2:
        raise RuntimeError(f"채널이 안 맞습니다: 입력 {c}, 필터 {c2}")
    sd, sh, sw = (stride,) * 3 if isinstance(stride, int) else tuple(stride)
    pd, ph, pw = (padding,) * 3 if isinstance(padding, int) else tuple(padding)
    out_d = (d - 1) * sd + kd

    # 깊이별 결과를 목록에 모았다가 한 번에 쌓는다. 자리마다 여러 입력 깊이가
    # 겹치므로 **더해야 한다** — 덮어쓰면 마지막 것만 남는다.
    slabs = [None] * out_d
    for od in range(d):
        for i in range(kd):
            plane = x[_slice_at(2, od, od + 1)].reshape(n, c, h, w)
            slab = weight[_slice_at(2, i, i + 1)].reshape(c2, f, kh, kw)
            part = conv_transpose2d(plane, slab, None, (sh, sw), (ph, pw))
            at = od * sd + i
            slabs[at] = part if slabs[at] is None else slabs[at] + part
    shape = slabs[0].data.shape
    out = cat([s.reshape(shape[0], shape[1], 1, shape[2], shape[3]) for s in slabs], 2)
    if pd:
        out = out[_slice_at(2, pd, out_d - pd)]
    if bias is not None:
        out = out + bias.reshape(1, -1, 1, 1, 1)
    return out


def _norm_flat(x, groups, eps, center=True):
    """정규화 셋이 나눠 쓰는 몸통. **묶는 구간을 한 축으로 눕혀서 받는다.**

    `mean(dim=…)` 이 축을 하나만 받으므로, 부르는 쪽이 `(그룹 수, 그 안의 원소 수)`
    모양으로 눕혀 주고 여기서는 마지막 축만 접는다. 축 목록을 받게 만들면 축약 쪽
    표면이 늘고, 그 표면은 정규화 말고는 아무도 안 쓴다.

    `center=False` 면 평균을 안 뺀다 — 그것이 `RMSNorm` 과 `LayerNorm` 의 유일한
    차이다. 따로 쓰면 두 벌이 되고, 두 벌이면 한쪽만 고쳐지는 날이 온다.
    """
    centered = x - x.mean(dim=-1, keepdim=True) if center else x
    var = (centered * centered).mean(dim=-1, keepdim=True)
    return centered / (var + eps).sqrt()


def _channel_shape(x, size):
    """채널 축(1번)에 맞춰 편다. `(1, C, 1, …)` 이라야 브로드캐스팅이 맞는다."""
    shape = [1] * len(x.data.shape)
    shape[1] = size
    return tuple(shape)


def group_norm(x, num_groups, weight=None, bias=None, eps=1e-5):
    """채널을 그룹으로 묶어 정규화. **그룹 수가 경계를 정한다.**

    `num_groups=1` 이면 채널 전체가 한 묶음이라 `LayerNorm` 과 같고, 채널 수와 같으면
    채널마다 따로라 `InstanceNorm` 과 같다. 셋은 서로의 특수한 경우이고, 묶는 규칙이
    틀리면 셋 중 둘이 같아진다 — 그래서 골든이 셋을 나란히 묻는다.
    """
    x = _wrap(x)
    shape = x.data.shape
    n, c = shape[0], shape[1]
    if c % num_groups:
        raise RuntimeError(f"채널 {c} 를 {num_groups} 그룹으로 못 나눕니다")
    inner = (c // num_groups) * int(_np.prod(shape[2:], dtype=int))
    out = _norm_flat(x.reshape(n, num_groups, inner), num_groups, eps).reshape(*shape)
    if weight is not None:
        out = out * _wrap(weight).reshape(*_channel_shape(x, c))
    if bias is not None:
        out = out + _wrap(bias).reshape(*_channel_shape(x, c))
    return out


def instance_norm(x, weight=None, bias=None, eps=1e-5):
    """표본마다·채널마다 따로. `group_norm` 에 그룹 수를 채널 수로 준 것이다."""
    x = _wrap(x)
    return group_norm(x, x.data.shape[1], weight, bias, eps)


def rms_norm(x, normalized_shape, weight=None, eps=None):
    """**평균을 안 뺀다.** 그것이 `LayerNorm` 과의 유일한 차이다.

    **기본 eps 가 `1e-5` 가 아니다.** torch 는 안 주면 그 dtype 의 기계 엡실론을
    쓰는데(float32 에서 1.19e-07), 다른 정규화 층들이 전부 `1e-5` 라 무심코 맞춰
    적었다. 순방향은 허용 오차 안에 들어와 통과했고 **기울기에서만 최대차 2.26e-02**
    로 갈렸다 — 분산이 작은 자리에서 증폭되기 때문이다.
    """
    x = _wrap(x)
    if eps is None:
        eps = float(_np.finfo(_np.float32).eps)
    shape = x.data.shape
    k = len(normalized_shape) if isinstance(normalized_shape, (list, tuple)) else 1
    lead = int(_np.prod(shape[:len(shape) - k], dtype=int))
    inner = int(_np.prod(shape[len(shape) - k:], dtype=int))
    out = _norm_flat(x.reshape(lead, inner), lead, eps, center=False).reshape(*shape)
    return out if weight is None else out * _wrap(weight)


def conv1d(x, weight, bias=None, stride=1, padding=0):
    """1차원 합성곱. **`conv2d` 에 높이 1 을 끼워 넣어 짠다.**

    자매(webgpu)가 이미 이 방식이다. 새 im2col 을 쓰면 같은 계산을 두 벌로 두게 되고,
    그러면 한쪽만 고쳐진 채로 갈리는 날이 온다.
    """
    x, weight = _wrap(x), _wrap(weight)
    n, c, length = x.data.shape
    f, c2, k = weight.data.shape
    lifted = x.reshape(n, c, 1, length)
    kernel = weight.reshape(f, c2, 1, k)
    # 높이 축은 건드리지 않는다 — 걸음 1, 채움 0.
    out = conv2d(lifted, kernel, bias, (1, stride), (0, padding))
    shape = out.data.shape
    return out.reshape(shape[0], shape[1], shape[3])


def conv3d(x, weight, bias=None, stride=1, padding=0):
    """3차원 합성곱. **깊이마다 2차원 합성곱을 돌려 더한다.**

    im2col 을 3차원으로 다시 쓰지 않는다 — 곱셈과 덧셈으로 짜면 역방향이 저절로
    따라오고, 새로 쓸 미분식이 없다. 느리지만 틀리지 않는다.
    """
    x, weight = _wrap(x), _wrap(weight)
    n, c, d, h, w = x.data.shape
    f, c2, kd, kh, kw = weight.data.shape
    if c != c2:
        raise RuntimeError(f"채널이 안 맞습니다: 입력 {c}, 필터 {c2}")
    sd, sh, sw = (stride, stride, stride) if isinstance(stride, int) else tuple(stride)
    pd, ph, pw = (padding, padding, padding) if isinstance(padding, int) else tuple(padding)
    if pd:
        pads = [(0, 0)] * 5
        pads[2] = (pd, pd)
        x = x._make(_np.pad(x.data, pads), (x,),
                    lambda g: (_np.asarray(g)[:, :, pd:-pd],), "Pad3dBackward0")
        d = x.data.shape[2]

    out_d = (d - kd) // sd + 1
    slabs = []
    for od in range(out_d):
        acc = None
        for i in range(kd):
            plane = x[_slice_at(2, od * sd + i, od * sd + i + 1)].reshape(n, c, h, w)
            slab = weight[_slice_at(2, i, i + 1)].reshape(f, c2, kh, kw)
            part = conv2d(plane, slab, None, (sh, sw), (ph, pw))
            acc = part if acc is None else acc + part
        shape = acc.data.shape
        slabs.append(acc.reshape(shape[0], shape[1], 1, shape[2], shape[3]))
    out = cat(slabs, 2)
    if bias is not None:
        bt = _wrap(bias)
        out = out + bt.reshape(1, -1, 1, 1, 1)
    return out


def max_pool1d(x, kernel_size, stride=None):
    """`max_pool2d` 에 높이 1 을 끼워 넣는다. 높이가 1 이고 창도 1 이라 그 축은 안 움직인다."""
    x = _wrap(x)
    n, c, length = x.data.shape
    stride = stride or kernel_size
    out_len = (length - kernel_size) // stride + 1
    lifted = x.reshape(n, c, 1, length)
    pooled = _pool_1d_over_last(lifted, kernel_size, stride)
    return pooled.reshape(n, c, out_len)


def _pool_1d_over_last(x, kernel_size, stride):
    """마지막 축만 창으로 줄인다 — 높이 축은 창 1·걸음 1 로 그대로 둔다."""
    parts = []
    length = x.data.shape[3]
    for start in range(0, length - kernel_size + 1, stride):
        window = [x[_slice_at(3, start + i, start + i + 1)] for i in range(kernel_size)]
        acc = window[0]
        for piece in window[1:]:
            acc = _maximum_first(acc, piece)
        parts.append(acc)
    return cat(parts, 3)


def max_pool3d(x, kernel_size, stride=None):
    """깊이 방향은 잘라서 최댓값을 겹쳐 취하고, 나머지는 `max_pool2d` 가 한다."""
    x = _wrap(x)
    stride = stride or kernel_size
    n, c, d, h, w = x.data.shape
    out_d = (d - kernel_size) // stride + 1
    slabs = []
    for od in range(out_d):
        acc = None
        for i in range(kernel_size):
            plane = x[_slice_at(2, od * stride + i, od * stride + i + 1)].reshape(n, c, h, w)
            pooled = max_pool2d(plane, kernel_size, stride)
            acc = pooled if acc is None else _maximum_first(acc, pooled)
        shape = acc.data.shape
        slabs.append(acc.reshape(shape[0], shape[1], 1, shape[2], shape[3]))
    return cat(slabs, 2)


def interpolate(x, scale_factor=2, mode="nearest"):
    """최근접 확대. 한 칸이 s×s 로 복제되므로 **역방향은 그 블록을 합하는 것**이다."""
    if mode != "nearest":
        _unsupported(f"interpolate(mode={mode!r}) — 최근접만 있습니다")
    x = _wrap(x)
    sh, sw = _pair(scale_factor)
    xd = x.data
    n, c, h, w = xd.shape
    out = _np.repeat(_np.repeat(xd, sh, axis=2), sw, axis=3)

    def back(g):
        gg = _np.asarray(g).reshape(n, c, h, sh, w, sw)
        return (gg.sum(axis=(3, 5)),)

    return x._make(out, (x,), back, "UpsampleBackward0")


def _spread(v, n):
    """수 하나면 축마다 같은 값으로, 목록이면 그대로."""
    return [v] * n if isinstance(v, int) else list(v)


def _fold_axis(x, axis, windows, kind):
    """축 하나를 창 목록대로 접는다. **창 목록이 길이가 달라도 된다.**

    적응형 풀링이 그 자리다 — 8 을 3 으로 줄이면 창이 3·3·2 다. 창 크기를 고정으로
    두면 안 떨어지는 경우를 통째로 못 하고, 실제로 그래서 `adaptive_avg_pool2d` 가
    배수가 아니면 거절하고 있었다.

    조각을 하나씩 꺼내 접으므로 **미분이 저절로 따라온다** — 새로 쓸 역전파식이 없다.
    최댓값은 `_maximum_first` 로 접는데, 동점일 때 앞자리를 주는 것이 torch 의 규칙이다.
    """
    parts = []
    for start, end in windows:
        pieces = [x[_slice_at(axis, j, j + 1)] for j in range(start, end)]
        acc = pieces[0]
        for piece in pieces[1:]:
            acc = acc + piece if kind == "avg" else _maximum_first(acc, piece)
        parts.append(acc * (1.0 / len(pieces)) if kind == "avg" else acc)
    return cat(parts, axis)


def _adaptive_windows(n_in, n_out):
    """torch 의 적응형 규칙. 시작은 내림, 끝은 올림이다.

    나누어떨어지면 균등하고, 안 떨어지면 **창 크기가 자리마다 다르다.** 그 규칙을
    한 줄로 못 적어서 값이 갈리는 자리이고, 골든이 떨어지는 경우와 아닌 경우를
    둘 다 묻는다.
    """
    return [((i * n_in) // n_out, -((-(i + 1) * n_in) // n_out))
            for i in range(n_out)]


def _adaptive(x, output_size, kind):
    """축을 하나씩 접는다. **창이 직사각형이라 축별로 나눠 해도 같은 값이다** —
    평균은 각 줄의 길이가 같아서 평균의 평균이 전체 평균이고, 최댓값은 원래 그렇다."""
    x = _wrap(x)
    spatial = len(x.data.shape) - 2
    sizes = _spread(output_size, spatial)
    out = x
    for k in range(spatial):
        axis = 2 + k
        out = _fold_axis(out, axis,
                         _adaptive_windows(out.data.shape[axis], sizes[k]), kind)
    return out


def _fixed(x, kernel_size, stride, kind):
    """고정 창. 적응형과 같은 기계에 창 목록만 다르게 준다."""
    x = _wrap(x)
    spatial = len(x.data.shape) - 2
    kernels = _spread(kernel_size, spatial)
    strides = _spread(stride if stride is not None else kernel_size, spatial)
    out = x
    for k in range(spatial):
        axis = 2 + k
        n_in = out.data.shape[axis]
        step, size = strides[k], kernels[k]
        windows = [(s, s + size) for s in range(0, n_in - size + 1, step)]
        out = _fold_axis(out, axis, windows, kind)
    return out


def adaptive_avg_pool2d(x, output_size):
    """출력 크기를 정해 평균 풀링.

    **배수가 아니어도 된다.** 예전에는 거절했는데, torch 는 창 크기를 자리마다 달리
    잡아 처리한다 — 거절하는 것이 흉내가 아니라 다른 규칙이었다.
    """
    return _adaptive(x, _pair(output_size), "avg")


def adaptive_avg_pool1d(x, output_size):
    return _adaptive(x, _spread(output_size, 1), "avg")


def adaptive_avg_pool3d(x, output_size):
    return _adaptive(x, _spread(output_size, 3), "avg")


def adaptive_max_pool1d(x, output_size):
    return _adaptive(x, _spread(output_size, 1), "max")


def adaptive_max_pool2d(x, output_size):
    return _adaptive(x, _pair(output_size), "max")


def adaptive_max_pool3d(x, output_size):
    return _adaptive(x, _spread(output_size, 3), "max")


def avg_pool1d(x, kernel_size, stride=None):
    return _fixed(x, _spread(kernel_size, 1), stride, "avg")


def avg_pool3d(x, kernel_size, stride=None):
    return _fixed(x, _spread(kernel_size, 3), stride, "avg")


def lp_pool2d(x, norm_type, kernel_size, stride=None):
    """`p` 승의 합을 `p` 제곱근한 것. p=1 이면 합, p 가 크면 최댓값에 가까워진다.

    **torch 의 조립을 그대로 따른다** — 평균 풀링을 쓰고 창 크기를 곱해 합으로
    되돌린 뒤 제곱근을 취한다. 부호와 `relu` 가 끼는 것도 그쪽 구현 그대로다.
    """
    x = _wrap(x)
    kh, kw = _pair(kernel_size)
    out = avg_pool2d(x ** norm_type, kernel_size, stride)
    return ((out.sign() * relu(out.abs())) * (kh * kw)) ** (1.0 / norm_type)


def lp_pool1d(x, norm_type, kernel_size, stride=None):
    x = _wrap(x)
    k = kernel_size if isinstance(kernel_size, int) else kernel_size[0]
    out = avg_pool1d(x ** norm_type, k, stride)
    return ((out.sign() * relu(out.abs())) * k) ** (1.0 / norm_type)


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


def _split_at_ties(a, b):
    """**동점이면 반씩 나눈다.** torch 가 그렇다 — `maximum(2, 2)` 의 기울기는 양쪽 다 0.5 다.

    `a >= b` 하나로 가르면 동점에서 a 가 전부 가져가고 b 는 0 을 받는다. 순방향은
    어느 쪽이든 똑같이 맞으므로 값 대조로는 절대 안 잡히고, 동점이 없는 입력으로도
    안 잡힌다 — 난수 두 벌이 정확히 같을 일이 없기 때문이다.
    """
    tie = a.data == b.data
    left = _np.where(tie, 0.5, (a.data > b.data).astype(a.data.dtype))
    return left, 1.0 - left


def _maximum_first(a, b):
    """동점이면 **먼저 온 쪽**이 다 가져간다 — 최댓값 풀링이 이쪽이다.

    `maximum` 과 갈라 둔 이유가 있다. torch 의 `maximum` 은 동점에서 반씩 나누지만
    `max_pool` 은 이긴 자리 **하나**를 골라 거기로만 흘린다(안에서 argmax 를 쓴다).
    풀링을 `maximum` 위에 얹어 두면 창 안에 같은 값이 둘 있을 때만 조용히 갈린다.
    """
    a, b = _wrap(a), _wrap(b)
    pick = a.data >= b.data
    return a._make(_np.maximum(a.data, b.data), (a, b),
                   lambda g: (g * pick, g * ~pick), "MaximumBackward0")


def maximum(a, b):
    a, b = _wrap(a), _wrap(b)
    la, lb = _split_at_ties(a, b)
    return a._make(_np.maximum(a.data, b.data), (a, b),
                   lambda g: (g * la, g * lb), "MaximumBackward0")


def minimum(a, b):
    a, b = _wrap(a), _wrap(b)
    lb, la = _split_at_ties(a, b)
    return a._make(_np.minimum(a.data, b.data), (a, b),
                   lambda g: (g * la, g * lb), "MinimumBackward0")


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


# ---- 원소별 제자리 연산
#
# `Tensor` 안의 `_inplace` 를 그대로 쓴다. 산수는 이미 있는 함수가 하고, 여기서는
# **그 결과를 제 버퍼에 되쓰는 것**만 한다 — 같은 식을 두 벌로 두면 언젠가 갈린다.

_INPLACE_UNARY = ("abs", "sqrt", "exp", "log", "sin", "cos", "tan", "tanh", "sigmoid",
                  "relu", "erf", "floor", "ceil", "round", "sign", "reciprocal",
                  "square", "trunc", "frac", "neg", "rsqrt", "log2", "log10",
                  "expm1", "log1p", "acos", "asin", "atan", "sinh", "cosh")


def _make_inplace(name):
    fn = globals()[name]

    def method(self):
        return self._inplace(lambda: fn(self), name + "_")

    method.__name__ = name + "_"
    method.__doc__ = f"`{name}` 을 제자리에서. 산수는 `{name}` 이 하고 여기서는 되쓰기만 한다."
    return method


for _nm in _INPLACE_UNARY:
    setattr(Tensor, _nm + "_", _make_inplace(_nm))


# ---- 모양 바꾸기의 나머지
#
# **이 파일에서는 `abs`·`round`·`pow` 가 파이썬 것이 아니다.** 위에서 같은 이름의 텐서
# 함수를 정의했기 때문이다. 정수에 쓰면 `'int' object has no attribute 'abs'` 로
# 멈춘다 — 실제로 `diagflat` 에서 그렇게 멈췄다. 그래서 정수용 별칭을 따로 둔다.
_abs = _builtins.abs
#
# `expand` 와 `repeat` 은 이름이 비슷한데 하는 일이 다르다 — 헷갈리면 조용히 다른 모양이
# 나온다. `expand` 는 **크기 1 인 축만** 늘리고 값을 복제하지 않는다(torch 에서는 뷰다).
# `repeat` 은 통째로 이어 붙인다. 기울기도 그만큼 다르다: expand 는 늘린 축을 도로 합치고,
# repeat 은 반복된 조각을 겹쳐 더한다.

def expand(t, *sizes):
    """크기 1 인 축을 늘린다. `-1` 은 "그대로 두라"는 뜻이다.

    torch 에서는 저장소를 공유하는 뷰지만 우리는 복제한다 — 뷰 공유는 코어의 명시적
    한계이고, 여기서만 흉내 내면 그 한계가 자리마다 달라진다.
    """
    t = _wrap(t)
    want = sizes[0] if len(sizes) == 1 and isinstance(sizes[0], (list, tuple)) else sizes
    src = t.data.shape
    lead = len(want) - len(src)
    target = []
    for i, size in enumerate(want):
        if i < lead:
            target.append(int(size))
            continue
        have = src[i - lead]
        if size == -1:
            target.append(have)
        elif have != 1 and int(size) != have:
            raise RuntimeError(_like_torch(
                f"크기 {have} 인 축은 {size} 로 늘릴 수 없습니다 — expand 는 크기 1 인 "
                "축만 늘립니다.",
                f"The expanded size of the tensor ({size}) must match the existing size "
                f"({have}) at non-singleton dimension {i - lead}"))
        else:
            target.append(int(size))
    target = tuple(target)
    lifted = t.data.reshape((1,) * lead + src)
    out = _np.broadcast_to(lifted, target)

    def back(g):
        return (_unbroadcast(_np.asarray(g), src),)

    return t._make(_np.ascontiguousarray(out), (t,), back, "ExpandBackward0")


def expand_as(t, other):
    return expand(t, *_wrap(other).data.shape)


# ── torch 가 **두 번째 이름**으로 주는 것들 ─────────────────────────────────
#
# `a + b` 는 되는데 `torch.add(a, b)` 가 없었다. 교재는 둘 다 쓰고, 없는 쪽을 만난
# 코드는 거기서 멈춘다. 연산자가 이미 하는 일이라 새로 계산할 것은 없고 **이름만**
# 필요하다 — 그런데 이름이 없으면 그 코드는 안 돈다.
#
# `alpha`·`rounding_mode` 처럼 연산자에 없는 인자가 붙는 자리가 있어서 그냥 별칭으로
# 못 두고 한 겹 감싼다.

def add(a, b, alpha=1):
    """`a + alpha·b`. **`alpha` 가 연산자에는 없다** — 그래서 별칭이 아니라 함수다."""
    return _wrap(a) + (b if alpha == 1 else _wrap(b) * alpha)


def sub(a, b, alpha=1):
    return _wrap(a) - (b if alpha == 1 else _wrap(b) * alpha)


def mul(a, b):
    return _wrap(a) * b


def div(a, b, rounding_mode=None):
    """`rounding_mode` 는 셋이다 — 없으면 참나눗셈, `'floor'`·`'trunc'` 는 정수 쪽."""
    out = _wrap(a) / b
    if rounding_mode is None:
        return out
    if rounding_mode == "floor":
        return out.floor()
    if rounding_mode == "trunc":
        return out.trunc()
    raise RuntimeError(f"rounding_mode 는 None·'floor'·'trunc' 뿐입니다: {rounding_mode!r}")


def floor_divide(a, b):
    return div(a, b, rounding_mode="floor")


def remainder(a, b):
    """**부호가 나누는 쪽을 따른다.** `fmod` 와 갈리는 자리가 그것이다."""
    return _wrap(a) % b


def fmod(a, b):
    """**부호가 나뉘는 쪽을 따른다.** C 의 `fmod` 규칙이고 `remainder` 와 반대다."""
    a, b = _wrap(a), _wrap(b)
    return a - (a / b).trunc() * b


def rsub(a, b, alpha=1):
    return sub(b, a, alpha)


multiply = mul
divide = div
subtract = sub
true_divide = div

greater = gt
greater_equal = ge
less = lt
less_equal = le
not_equal = ne


def t(x):
    """2 차원 전치. **1 차원 이하는 그대로 둔다** — torch 가 그렇다."""
    x = _wrap(x)
    return x if len(x.data.shape) < 2 else x.transpose(0, 1)


def adjoint(x):
    """마지막 두 축을 바꾼다. 실수만 있으므로 켤레는 항등이다."""
    return _wrap(x).transpose(-2, -1)


def moveaxis(t, source, destination):
    """`movedim` 의 다른 이름. **별칭으로 못 둔다** — `movedim` 이 이 아래에 있다."""
    return movedim(t, source, destination)


concat = cat
concatenate = cat


def broadcast_to(x, shape):
    return expand(_wrap(x), *shape)


def broadcast_tensors(*tensors):
    """전부를 공통 모양으로 늘린다. 모양 계산은 numpy 에 맡긴다."""
    ts = [_wrap(v) for v in tensors]
    shape = _np.broadcast_shapes(*[v.data.shape for v in ts])
    return tuple(broadcast_to(v, shape) for v in ts)


def broadcast_shapes(*shapes):
    return Size(_np.broadcast_shapes(*shapes))


def _stack_along(items, dim, lift):
    """쌓기 넷이 나눠 쓰는 몸통. **어느 축으로 붙이고 몇 차원으로 올리느냐**만 다르다."""
    ts = [lift(_wrap(v)) for v in items]
    return cat(ts, dim)


def hstack(tensors):
    """1 차원은 이어 붙이고 그 위는 **열 방향**으로 붙인다 — torch 가 그렇게 가른다."""
    ts = [_wrap(v) for v in tensors]
    dim = 0 if len(ts[0].data.shape) == 1 else 1
    return cat(ts, dim)


def vstack(tensors):
    return _stack_along(tensors, 0, atleast_2d)


def dstack(tensors):
    return _stack_along(tensors, 2, atleast_3d)


def column_stack(tensors):
    """1 차원을 **열 하나로 세워** 붙인다. `hstack` 과 여기서 갈린다."""
    ts = []
    for v in tensors:
        v = _wrap(v)
        ts.append(v.reshape(v.data.shape[0], 1) if len(v.data.shape) == 1 else v)
    return cat(ts, 1)


row_stack = vstack


# ── 이름만이 아니라 **계산이 없던** 것들 ───────────────────────────────────

def empty_like(t, **kw):
    """모양만 빌린다. 값은 정하지 않는다 — torch 도 그렇다."""
    return zeros(*_wrap(t).data.shape)


def rand_like(t, **kw):
    return rand(*_wrap(t).data.shape)


def randn_like(t, **kw):
    return randn(*_wrap(t).data.shape)


def randint_like(t, low, high=None, **kw):
    if high is None:
        low, high = 0, low
    return randint(low, high, _wrap(t).data.shape)


def scalar_tensor(value, **kw):
    return tensor(_np.asarray(value, dtype=_DEFAULT_DTYPE))


def logspace(start, end, steps, base=10.0, **kw):
    """`base` 의 거듭제곱으로 고르게. `linspace` 를 지수로 쓴다."""
    return tensor((base ** _np.linspace(start, end, steps)).astype(_DEFAULT_DTYPE))


def meshgrid(*tensors, indexing="ij"):
    """격자를 만든다. **`indexing` 을 안 주면 torch 가 경고하고 `ij` 로 간다.**

    `xy` 는 앞의 두 축이 뒤바뀐 것이라, 규칙을 하나로 뭉뚱그리면 2 차원에서만
    우연히 맞는다.
    """
    ts = [_wrap(v) for v in tensors]
    if indexing not in ("ij", "xy"):
        raise RuntimeError(f"indexing 은 'ij' 나 'xy' 다: {indexing!r}")
    order = list(range(len(ts)))
    if indexing == "xy" and len(ts) >= 2:
        order[0], order[1] = order[1], order[0]
    sizes = [ts[i].data.shape[0] for i in order]
    out = []
    for place, i in enumerate(order):
        shape = [1] * len(ts)
        shape[place] = sizes[place]
        out.append(expand(ts[i].reshape(*shape), *sizes))
    if indexing == "xy" and len(ts) >= 2:
        out[0], out[1] = out[1], out[0]
    return tuple(out)


def lerp(start, end, weight):
    """`start + weight·(end − start)`. 두 점 사이를 고르게 잇는다."""
    start, end = _wrap(start), _wrap(end)
    return start + (end - start) * weight


def nan_to_num(t, nan=0.0, posinf=None, neginf=None):
    """NaN 과 무한대를 유한한 수로 바꾼다. **안 주면 그 dtype 의 끝값이다.**"""
    t = _wrap(t)
    d = t.data
    hi = _np.finfo(d.dtype).max if posinf is None else posinf
    lo = _np.finfo(d.dtype).min if neginf is None else neginf
    fixed = _np.nan_to_num(d, nan=nan, posinf=hi, neginf=lo)
    keep = _np.isfinite(d)
    return t._make(fixed.astype(d.dtype), (t,), lambda g: (g * keep,),
                   "NanToNumBackward0")


def isclose(a, b, rtol=1e-5, atol=1e-8, equal_nan=False):
    a, b = _wrap(a), _wrap(b)
    return Tensor(_np.isclose(a.data, b.data, rtol=rtol, atol=atol,
                              equal_nan=equal_nan))


def isreal(t):
    """실수만 있으므로 전부 참이다. **거짓말이 아니라 사실이다** — 복소수가 없다."""
    return Tensor(_np.ones(_wrap(t).data.shape, dtype=bool))


def isposinf(t):
    return Tensor(_np.isposinf(_wrap(t).data))


def isneginf(t):
    return Tensor(_np.isneginf(_wrap(t).data))


def isin(elements, test_elements, **kw):
    return Tensor(_np.isin(_wrap(elements).data, _wrap(test_elements).data))


def _nan_extreme(name, pick):
    """`fmax`·`fmin`. **NaN 을 건너뛴다** — `maximum` 은 NaN 을 물고 나온다."""
    def call(a, b):
        a, b = _wrap(a), _wrap(b)
        out = pick(a.data, b.data)
        take_a = out == a.data

        def back(g):
            g = _np.asarray(g)
            return (g * take_a, g * ~take_a)

        return a._make(out, (a, b), back, f"{name.capitalize()}Backward0")
    call.__name__ = name
    return call


fmax = _nan_extreme("fmax", _np.fmax)
fmin = _nan_extreme("fmin", _np.fmin)


def float_power(a, b):
    """실수 지수. torch 는 배정도로 세지만 여기는 float32 뿐이다."""
    return _wrap(a) ** b


def logical_xor(a, b):
    return Tensor(_np.logical_xor(_wrap(a).data != 0, _wrap(b).data != 0))


def var_mean(t, dim=None, keepdim=False, **kw):
    """**둘을 한 번에 준다.** 하나만 물으면 다른 하나가 틀려도 안 걸린다.

    **이름으로 넘긴다.** `var` 의 자리 순서가 `(dim, unbiased, keepdim)` 이라 자리로
    주면 `keepdim` 이 `unbiased` 자리에 들어간다 — 그러면 ddof 가 1 에서 0 으로
    바뀌어 값이 12/11 배 어긋난다. 실제로 그렇게 걸렸다.
    """
    t = _wrap(t)
    return (t.var(dim=dim, keepdim=keepdim), t.mean(dim=dim, keepdim=keepdim))


def std_mean(t, dim=None, keepdim=False, **kw):
    t = _wrap(t)
    return (t.std(dim=dim, keepdim=keepdim), t.mean(dim=dim, keepdim=keepdim))


def inner(a, b):
    """마지막 축끼리의 안쪽 곱. 1 차원이면 점곱이다."""
    a, b = _wrap(a), _wrap(b)
    return a @ b.transpose(-2, -1) if len(a.data.shape) > 1 else (a * b).sum()


def vdot(a, b):
    return (_wrap(a) * _wrap(b)).sum()


def kron(a, b):
    """크로네커 곱. 한쪽을 늘려 곱하고 다시 접는다 — 새 커널이 필요 없다."""
    a, b = _wrap(a), _wrap(b)
    ash, bsh = a.data.shape, b.data.shape
    if len(ash) != 1 or len(bsh) != 1:
        _unsupported("kron(1 차원이 아닌 것)")
    out = a.reshape(ash[0], 1) * b.reshape(1, bsh[0])
    return out.reshape(ash[0] * bsh[0])


def cross(a, b, dim=-1):
    """외적. 축의 길이가 3 이어야 한다."""
    a, b = _wrap(a), _wrap(b)
    rank = len(a.data.shape)
    axis = dim + rank if dim < 0 else dim
    if a.data.shape[axis] != 3:
        raise RuntimeError(f"cross 는 축 {dim} 의 길이가 3 이어야 합니다")

    def part(t, i):
        return narrow(t, axis, i, 1)

    return cat([part(a, 1) * part(b, 2) - part(a, 2) * part(b, 1),
                part(a, 2) * part(b, 0) - part(a, 0) * part(b, 2),
                part(a, 0) * part(b, 1) - part(a, 1) * part(b, 0)], axis)


def block_diag(*tensors):
    """대각선에 블록을 늘어놓고 나머지는 0. **0 으로 메우므로 그쪽에는 기울기가 없다.**"""
    ts = [atleast_2d(_wrap(v)) for v in tensors]
    rows = sum(v.data.shape[0] for v in ts)
    cols = sum(v.data.shape[1] for v in ts)
    lines, at = [], 0
    for v in ts:
        h, w = v.data.shape
        pieces = []
        if at:
            pieces.append(zeros(h, at))
        pieces.append(v)
        if cols - at - w:
            pieces.append(zeros(h, cols - at - w))
        lines.append(cat(pieces, 1) if len(pieces) > 1 else v)
        at += w
    return cat(lines, 0) if len(lines) > 1 else lines[0]


def repeat(t, *reps):
    """통째로 반복해 붙인다. **`tile` 과 같은 일이다** — torch 가 이름을 둘 둔 것뿐이라
    한쪽을 다시 쓰지 않는다."""
    want = reps[0] if len(reps) == 1 and isinstance(reps[0], (list, tuple)) else reps
    return tile(t, tuple(int(r) for r in want))


def ravel(t):
    return _wrap(t).reshape(-1)


def swapaxes(t, a, b):
    """두 축을 맞바꾼다. `transpose` 와 같고, torch 가 numpy 를 따라 이름을 하나 더 둔 것이다."""
    t = _wrap(t)
    order = list(range(t.data.ndim))
    order[a], order[b] = order[b % t.data.ndim], order[a % t.data.ndim]
    return t.permute(*order)


swapdims = swapaxes


def select(t, dim, index):
    """축 하나에서 한 장을 뽑되 **그 축을 없앤다.** 자르기와 달리 랭크가 하나 준다."""
    t = _wrap(t)
    return t[_slice_at(dim % t.data.ndim, index, index + 1)].squeeze(dim)


def diagonal(t, offset=0, dim1=0, dim2=1):
    """대각선을 뽑는다. `offset` 은 위·아래로 몇 칸 옮긴 대각선인지다.

    역방향은 뽑은 자리에만 돌려놓는 것이다 — numpy 의 `diagonal` 은 읽기 전용 뷰를
    주므로 거기에 쓰지 않고 빈 판을 만들어 채운다.
    """
    t = _wrap(t)
    out = _np.diagonal(t.data, offset=offset, axis1=dim1, axis2=dim2)

    def back(g):
        # numpy 의 `diagonal` 은 **읽기 전용 뷰**라 거기에 쓸 수 없다. 빈 판을 만들고
        # 좌표를 직접 계산해 넣는다.
        z = _np.zeros_like(t.data)
        n = out.shape[-1]
        rows = _np.arange(n) + max(0, -offset)
        cols = _np.arange(n) + max(0, offset)
        idx = [slice(None)] * z.ndim
        idx[dim1], idx[dim2] = rows, cols
        z[tuple(idx)] = _np.moveaxis(_np.asarray(g), -1, 0)
        return (z,)

    return t._make(_np.ascontiguousarray(out), (t,), back, "DiagonalBackward0")


def diagflat(t, offset=0):
    """평평하게 편 뒤 대각행렬로 만든다."""
    t = _wrap(t)
    flat = t.reshape(-1)
    n = flat.data.shape[0] + _abs(offset)
    out = _np.zeros((n, n), dtype=t.data.dtype)
    rows = _np.arange(flat.data.shape[0]) + max(0, -offset)
    cols = _np.arange(flat.data.shape[0]) + max(0, offset)
    out[rows, cols] = flat.data

    def back(g):
        return (_np.asarray(g)[rows, cols].reshape(flat.data.shape),)

    return flat._make(out, (flat,), back, "DiagflatBackward0")


def rot90(t, k=1, dims=(0, 1)):
    t = _wrap(t)
    dims = tuple(dims)
    return t._make(_np.ascontiguousarray(_np.rot90(t.data, k, dims)), (t,),
                   lambda g: (_np.ascontiguousarray(_np.rot90(_np.asarray(g), -k, dims)),),
                   "Rot90Backward0")


def unfold(t, dim, size, step):
    """미끄러지는 창을 새 축으로 만든다. 창이 겹치면 **역방향에서 기울기가 쌓인다**
    (실측: 길이 5 를 크기 3·걸음 1 로 펴면 [1,2,3,2,1] 이 나온다)."""
    t = _wrap(t)
    axis = dim % t.data.ndim
    count = (t.data.shape[axis] - size) // step + 1
    starts = _np.arange(count) * step
    pieces = [_np.take(t.data, _np.arange(s, s + size), axis=axis) for s in starts]
    out = _np.stack([_np.moveaxis(p, axis, -1) for p in pieces], axis=axis)

    def back(g):
        z = _np.zeros_like(t.data)
        gg = _np.asarray(g)
        for i, s in enumerate(starts):
            piece = _np.moveaxis(_np.take(gg, i, axis=axis), -1, axis)
            idx = [slice(None)] * z.ndim
            idx[axis] = slice(s, s + size)
            z[tuple(idx)] += piece
        return (z,)

    return t._make(out, (t,), back, "UnfoldBackward0")


def hsplit(t, parts):
    """가로로 나눈다 — 1차원이면 축 0, 아니면 축 1 이다."""
    t = _wrap(t)
    return chunk(t, parts, dim=0 if t.data.ndim == 1 else 1)


def vsplit(t, parts):
    return chunk(_wrap(t), parts, dim=0)


def dsplit(t, parts):
    return chunk(_wrap(t), parts, dim=2)


def fliplr(t):
    return flip(_wrap(t), (1,))


def flipud(t):
    return flip(_wrap(t), (0,))


def unflatten(t, dim, sizes):
    t = _wrap(t)
    shape = list(t.data.shape)
    shape[dim:dim + 1] = list(sizes)
    return t.reshape(*shape)


def atleast_1d(t):
    t = _wrap(t)
    return t if t.data.ndim >= 1 else t.reshape(1)


def atleast_2d(t):
    t = atleast_1d(_wrap(t))
    return t if t.data.ndim >= 2 else t.reshape(1, t.data.shape[0])


def atleast_3d(t):
    t = atleast_2d(_wrap(t))
    return t if t.data.ndim >= 3 else t.reshape(*(t.data.shape + (1,)))


def _negate(shifts):
    return -shifts if isinstance(shifts, int) else tuple(-s for s in shifts)


def index_select(t, dim, index):
    t = _wrap(t)
    idx = index.data.astype(int) if isinstance(index, Tensor) else _np.asarray(index, dtype=int)
    return t[_index_at(dim, idx)]


def _index_at(dim, idx):
    return tuple(slice(None) for _ in range(dim)) + (idx,)


# 메서드로만 있던 것들의 **함수 형태.** torch 는 `torch.matmul(a, b)` 도 `a @ b` 도
# 주는데 우리는 뒤쪽만 있었다 — 자매와 맞춰보다 드러났고, 자매 대비만이 아니라
# torch 대비 구멍이었다.

def matmul(a, b):
    return _wrap(a) @ _wrap(b)


def reshape(t, *shape):
    return _wrap(t).reshape(*shape)


def unsqueeze(t, dim):
    return _wrap(t).unsqueeze(dim)


def masked_fill(t, mask, value):
    t, m = _wrap(t), _wrap(mask)
    return where(m, Tensor(_np.asarray(value, dtype=t.data.dtype)), t)


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


# ── 수치 계열 ──────────────────────────────────────────────────────────────

def cdist(a, b, p=2.0):
    """모든 짝 사이의 거리. **브로드캐스팅 하나로 풀린다** — 새 커널이 필요 없다."""
    a, b = _wrap(a), _wrap(b)
    n, k = a.data.shape
    m = b.data.shape[0]
    diff = a.reshape(n, 1, k) - b.reshape(1, m, k)
    if p == 2.0:
        return (diff * diff).sum(dim=2).sqrt()
    return ((diff.abs() ** p).sum(dim=2)) ** (1.0 / p)


def cov(t, correction=1, **kw):
    """공분산. 줄이 변수이고 칸이 관측이다 — numpy 와 축이 반대라 헷갈리는 자리다."""
    t = _wrap(t)
    d = t.data
    if d.ndim == 1:
        d = d.reshape(1, -1)
        t = t.reshape(1, d.shape[1])
    n = d.shape[1]
    centered = t - t.mean(dim=1, keepdim=True)
    return (centered @ centered.transpose(0, 1)) * (1.0 / max(1, n - correction))


def corrcoef(t):
    """공분산을 표준편차로 나눈 것. **대각선이 1 이 된다** — 그것이 검산이다."""
    c = cov(t)
    d = c.data
    scale = _np.sqrt(_np.outer(_np.diag(d), _np.diag(d)))
    return c / Tensor(scale.astype(d.dtype))


def tensordot(a, b, dims=2, **kw):
    """지정한 축끼리 접어 곱한다. 축 목록을 받는 꼴만 다룬다 — 그것이 torch 의 기본형이다."""
    a, b = _wrap(a), _wrap(b)
    if isinstance(dims, int):
        rank = len(a.data.shape)
        left = list(range(rank - dims, rank))
        right = list(range(dims))
    else:
        left, right = [list(v) for v in dims]
    # 접을 축을 뒤로·앞으로 몰고 행렬곱 한 번으로 끝낸다.
    a_keep = [i for i in range(len(a.data.shape)) if i not in left]
    b_keep = [i for i in range(len(b.data.shape)) if i not in right]
    a_shape = [a.data.shape[i] for i in a_keep]
    b_shape = [b.data.shape[i] for i in b_keep]
    inner = int(_np.prod([a.data.shape[i] for i in left], dtype=int))
    am = a.permute(*(a_keep + left)).reshape(int(_np.prod(a_shape, dtype=int)), inner)
    bm = b.permute(*(right + b_keep)).reshape(inner, int(_np.prod(b_shape, dtype=int)))
    return (am @ bm).reshape(*(a_shape + b_shape))


def trapezoid(y, x=None, dx=1.0, dim=-1, **kw):
    """사다리꼴 적분. 이웃한 두 점의 평균에 간격을 곱해 더한다."""
    y = _wrap(y)
    rank = len(y.data.shape)
    axis = dim + rank if dim < 0 else dim
    n = y.data.shape[axis]
    left = narrow(y, axis, 0, n - 1)
    right = narrow(y, axis, 1, n - 1)
    if x is None:
        return ((left + right) * (dx / 2.0)).sum(dim=axis)
    x = _wrap(x)
    step = narrow(x, axis, 1, n - 1) - narrow(x, axis, 0, n - 1)
    return ((left + right) * step * 0.5).sum(dim=axis)


def cumulative_trapezoid(y, x=None, dx=1.0, dim=-1, **kw):
    """`trapezoid` 의 누적판. 마지막 값이 `trapezoid` 와 같아야 한다."""
    y = _wrap(y)
    rank = len(y.data.shape)
    axis = dim + rank if dim < 0 else dim
    n = y.data.shape[axis]
    left = narrow(y, axis, 0, n - 1)
    right = narrow(y, axis, 1, n - 1)
    pieces = (left + right) * (dx / 2.0)
    if x is not None:
        x = _wrap(x)
        step = narrow(x, axis, 1, n - 1) - narrow(x, axis, 0, n - 1)
        pieces = (left + right) * step * 0.5
    return cumsum(pieces, axis)


# 란초시 계수(g=7, n=9). **손으로 안 고른다** — 잘 알려진 표이고, 여기서 자릿수를
# 줄이면 그만큼 답이 틀린다.
_LANCZOS = (0.99999999999980993, 676.5203681218851, -1259.1392167224028,
            771.32342877765313, -176.61502916214059, 12.507343278686905,
            -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7)


def _lgamma_np(d):
    """감마의 로그. **numpy 에 없다** — 란초시 근사를 원소별로 적는다.

    `np.vectorize` 를 안 쓴다. 이 저장소가 `gelu` 에서 그것으로 20 배를 잃었고,
    원소별 numpy 로 바꿔 같은 값을 20 배 빠르게 냈다 — 같은 실수를 다시 하지 않는다.

    반사 공식으로 음수 쪽을 접는다: `Γ(x)Γ(1−x) = π/sin(πx)`.
    """
    x = _np.asarray(d, dtype=_np.float64)
    neg = x < 0.5
    z = _np.where(neg, 1.0 - x, x) - 1.0
    acc = _np.full(z.shape, _LANCZOS[0])
    for i in range(1, len(_LANCZOS)):
        acc = acc + _LANCZOS[i] / (z + i)
    t = z + len(_LANCZOS) - 1.5
    out = (0.5 * _math.log(2 * _math.pi) + (z + 0.5) * _np.log(t) - t
           + _np.log(_np.abs(acc)))
    # 반사: lgamma(x) = log(π/|sin(πx)|) − lgamma(1−x)
    flipped = _np.log(_np.pi / _np.abs(_np.sin(_np.pi * _np.where(neg, x, 0.5)))) - out
    return _np.where(neg, flipped, out)


def _polygamma0(d):
    """`digamma` — 감마의 로그미분. **되풀이 식으로 큰 쪽으로 밀고 점근식을 쓴다.**

    작은 x 에서 점근식이 안 맞으므로 `ψ(x) = ψ(x+1) − 1/x` 로 6 이상까지 올린 뒤
    센다. numpy 에 없는 몇 안 되는 자리라 손으로 적는다.
    """
    x = _np.asarray(d, dtype=_np.float64)
    out = _np.zeros_like(x)
    while _np.any(x < 6):
        small = x < 6
        out = _np.where(small, out - 1.0 / _np.where(small, x, 1.0), out)
        x = _np.where(small, x + 1.0, x)
    inv = 1.0 / x
    inv2 = inv * inv
    # 스털링 계열의 점근 전개. 여섯 항이면 x ≥ 6 에서 float32 정밀도를 넘는다.
    series = (_np.log(x) - 0.5 * inv
              - inv2 * (1.0 / 12 - inv2 * (1.0 / 120 - inv2 / 252)))
    return out + series


def _polygamma1(d):
    """`trigamma` — `digamma` 의 미분. 같은 방식으로 밀고 점근식을 쓴다."""
    x = _np.asarray(d, dtype=_np.float64)
    out = _np.zeros_like(x)
    while _np.any(x < 6):
        small = x < 6
        out = _np.where(small, out + 1.0 / _np.where(small, x, 1.0) ** 2, out)
        x = _np.where(small, x + 1.0, x)
    inv = 1.0 / x
    inv2 = inv * inv
    return out + inv * (1.0 + 0.5 * inv
                        + inv2 * (1.0 / 6 - inv2 * (1.0 / 30 - inv2 / 42)))


def _erfinv_np(d):
    """`erf` 의 역함수. **닫힌 식이 없다** — 잘 알려진 유리식 근사를 쓴다.

    구간을 둘로 가른다. 가운데(|x| ≤ 0.7)와 꼬리는 수렴이 달라서 한 식으로 못 덮고,
    한 식으로 덮으려 들면 한쪽이 허용 오차를 넘는다.
    """
    x = _np.asarray(d, dtype=_np.float64)
    a = (0.886226899, -1.645349621, 0.914624893, -0.140543331)
    b = (1.0, -2.118377725, 1.442710462, -0.329097515, 0.012229801)
    c = (-1.970840454, -1.624906493, 3.429567803, 1.641345311)
    e = (1.0, 3.543889200, 1.637067800)

    z = x * x
    mid = (x * (((a[3] * z + a[2]) * z + a[1]) * z + a[0])
           / ((((b[4] * z + b[3]) * z + b[2]) * z + b[1]) * z + b[0]))
    # 꼬리 — `sqrt(-log((1−|x|)/2))` 로 옮겨 센다.
    safe = _np.clip(_np.abs(x), 0.0, 1 - 1e-12)
    w = _np.sqrt(-_np.log((1.0 - safe) / 2.0))
    tail = (_np.sign(x) * (((c[3] * w + c[2]) * w + c[1]) * w + c[0])
            / ((e[2] * w + e[1]) * w + e[0]))
    out = _np.where(_np.abs(x) <= 0.7, mid, tail)
    # 뉴턴 한 번. 근사식만으로는 float32 허용 오차 언저리라 한 번 더 조인다.
    err = _erf64(out) - x
    return out - err / (2.0 / _math.sqrt(_math.pi) * _np.exp(-out * out))


def lgamma(t):
    """감마 함수의 로그. **미분이 `digamma`** 라 둘 중 하나만 있으면 반쪽이다."""
    t = _wrap(t)
    d = t.data
    out = _lgamma_np(d).astype(d.dtype)
    return t._make(out, (t,), lambda g: (g * _polygamma0(d).astype(d.dtype),),
                   "LgammaBackward0")


def digamma(t):
    """감마의 로그미분. 미분은 `trigamma` 다."""
    t = _wrap(t)
    d = t.data
    out = _polygamma0(d).astype(d.dtype)
    return t._make(out, (t,), lambda g: (g * _polygamma1(d).astype(d.dtype),),
                   "DigammaBackward0")


def erfinv(t):
    """`erf` 의 역함수. 미분은 `√π/2 · exp(erfinv(x)²)` 다."""
    t = _wrap(t)
    d = t.data
    out = _erfinv_np(d)
    grad = (_math.sqrt(_math.pi) / 2.0) * _np.exp(out * out)
    return t._make(out.astype(d.dtype), (t,),
                   lambda g: (g * grad.astype(d.dtype),), "ErfinvBackward0")


# ── 색인으로 **쓰는** 쪽. 읽는 쪽(`gather`)의 반대다. ───────────────────────

def _as_index(index):
    return (index.data.astype(int) if isinstance(index, Tensor)
            else _np.asarray(index, dtype=int))


def scatter(t, dim, index, src):
    """번호가 가리키는 자리에 **덮어쓴다.** 겹치면 마지막에 쓴 것이 남는다.

    `scatter_add` 와 겹치는 번호에서만 갈린다 — 안 겹치는 번호로 재면 두 함수가
    같아 보인다. 그래서 골든이 0 이 두 번 나오는 번호로 묻는다.
    """
    t = _wrap(t)
    idx = _as_index(index)
    out = t.data.copy()
    scalar = not isinstance(src, Tensor)
    values = (_np.full(idx.shape, src, dtype=t.data.dtype) if scalar
              else _wrap(src).data)
    _np.put_along_axis(out, idx, values, axis=dim)

    def back(g):
        g = _np.asarray(g)
        # 덮어쓴 자리는 원본과 끊긴다 — 그 자리에는 0 이 간다.
        keep = _np.ones(t.data.shape, dtype=g.dtype)
        _np.put_along_axis(keep, idx, 0.0, axis=dim)
        got = (g * keep,)
        return got if scalar else got + (_np.take_along_axis(g, idx, axis=dim),)

    parents = (t,) if scalar else (t, _wrap(src))
    return t._make(out, parents, back, "ScatterBackward0")


def scatter_add(t, dim, index, src):
    """번호가 가리키는 자리에 **더한다.** 겹치면 쌓인다 — `scatter` 와 여기서 갈린다."""
    t, src = _wrap(t), _wrap(src)
    idx = _as_index(index)
    out = t.data.copy()
    # `put_along_axis` 는 덮어쓰므로 못 쓴다. 겹치는 번호를 제대로 쌓으려면
    # `add.at` 이어야 한다 — 그것이 이 함수와 `scatter` 의 차이 전부다.
    grid = _np.indices(idx.shape)
    where = list(grid)
    where[dim] = idx
    _np.add.at(out, tuple(where), src.data)

    def back(g):
        g = _np.asarray(g)
        return (g, _np.take_along_axis(g, idx, axis=dim))

    return t._make(out, (t, src), back, "ScatterAddBackward0")


def index_add(t, dim, index, source, alpha=1):
    """번호가 가리키는 **줄**에 더한다. 번호가 겹치면 쌓인다."""
    t, source = _wrap(t), _wrap(source)
    idx = _as_index(index)
    out = t.data.copy()
    _np.add.at(out, (slice(None),) * dim + (idx,), source.data * alpha)

    def back(g):
        g = _np.asarray(g)
        return (g, _np.take(g, idx, axis=dim) * alpha)

    return t._make(out, (t, source), back, "IndexAddBackward0")


def index_copy(t, dim, index, source):
    """번호가 가리키는 줄을 **갈아 끼운다.** 그 줄로는 기울기가 안 간다."""
    t, source = _wrap(t), _wrap(source)
    idx = _as_index(index)
    out = t.data.copy()
    picker = (slice(None),) * dim + (idx,)
    out[picker] = source.data

    def back(g):
        g = _np.asarray(g)
        keep = _np.ones(t.data.shape, dtype=g.dtype)
        keep[picker] = 0.0
        return (g * keep, _np.take(g, idx, axis=dim))

    return t._make(out, (t, source), back, "IndexCopyBackward0")


def index_fill(t, dim, index, value):
    """번호가 가리키는 줄을 한 값으로 채운다."""
    t = _wrap(t)
    idx = _as_index(index)
    out = t.data.copy()
    picker = (slice(None),) * dim + (idx,)
    out[picker] = value

    def back(g):
        g = _np.asarray(g)
        keep = _np.ones(t.data.shape, dtype=g.dtype)
        keep[picker] = 0.0
        return (g * keep,)

    return t._make(out, (t,), back, "IndexFillBackward0")


def take(t, index):
    """**평평하게 펴서** 뽑는다 — 축이라는 개념이 없다."""
    t = _wrap(t)
    idx = _as_index(index)
    shape = t.data.shape

    def back(g):
        z = _np.zeros(int(_np.prod(shape)), dtype=_np.asarray(g).dtype)
        _np.add.at(z, idx.reshape(-1), _np.asarray(g).reshape(-1))
        return (z.reshape(shape),)

    return t._make(_np.take(t.data, idx), (t,), back, "TakeBackward0")


def take_along_dim(t, indices, dim=None):
    """`gather` 와 같은 것. torch 가 두 이름을 다 준다."""
    if dim is None:
        return take(t, indices)
    return gather(t, dim, indices)


def searchsorted(sorted_sequence, values, right=False, **kw):
    """정렬된 것 안에서 들어갈 자리. **`right` 가 동점의 어느 쪽인지 정한다.**"""
    seq = _wrap(sorted_sequence).data
    want = _wrap(values).data
    return Tensor(_np.searchsorted(seq, want, side="right" if right else "left")
                  .astype(_np.int64))


def bucketize(values, boundaries, right=False, **kw):
    """`searchsorted` 와 **인자 순서가 뒤집혀 있다.** 그것이 두 이름의 차이 전부다."""
    return searchsorted(boundaries, values, right=right)


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


# ---- 축약의 나머지
#
# `amax`·`amin` 은 `max`·`min` 과 값이 같고 **번호를 안 준다.** 다른 것은 그것뿐이 아니다 —
# 동점일 때 기울기를 **똑같이 나눈다**(실측: [1,3,3,2] 의 amax 기울기가 [0,.5,.5,0]).
# 한 자리에만 몰아주면 값 검사는 통과하고 학습만 미묘하게 갈린다.

def _spread_max(t, dim, keepdim, take, name):
    """최댓값(또는 최솟값)으로 축약하되 **동점에 기울기를 고르게 나눈다.**"""
    t = _wrap(t)
    out = take(t.data, axis=dim, keepdims=True)
    hit = (t.data == out).astype(t.data.dtype)
    share = hit / hit.sum(axis=dim, keepdims=True)
    final = out if keepdim else (out if dim is None else _np.squeeze(out, axis=dim))
    if dim is None:
        final = out.reshape(())

    def back(g):
        gg = _np.asarray(g)
        if dim is not None and not keepdim:
            gg = _np.expand_dims(gg, dim)
        return (gg * share,)

    return t._make(final, (t,), back, name)


def amax(t, dim=None, keepdim=False):
    return _spread_max(t, dim, keepdim, _np.max, "AmaxBackward0")


def amin(t, dim=None, keepdim=False):
    return _spread_max(t, dim, keepdim, _np.min, "AminBackward0")


def aminmax(t, dim=None, keepdim=False):
    return _MinMax(amin(t, dim, keepdim), amax(t, dim, keepdim))


def _nan_mask(t):
    """nan 자리를 0 으로 바꾼 것과, 어디가 nan 이었는지."""
    bad = _np.isnan(t.data)
    return _np.where(bad, 0.0, t.data), bad


def nansum(t, dim=None, keepdim=False):
    """nan 을 **0 으로 세는** 합. 기울기도 그 자리로는 안 간다."""
    t = _wrap(t)
    clean, bad = _nan_mask(t)
    return t._make(clean.sum(axis=dim, keepdims=keepdim), (t,),
                   lambda g: (_np.where(bad, 0.0, _expand_reduced(g, t.data.shape, dim, keepdim)),),
                   "NansumBackward0")


def nanmean(t, dim=None, keepdim=False):
    """nan 을 **빼고** 낸 평균 — 세는 개수도 nan 이 아닌 것만이다."""
    t = _wrap(t)
    clean, bad = _nan_mask(t)
    count = (~bad).sum(axis=dim, keepdims=keepdim)
    total = clean.sum(axis=dim, keepdims=keepdim)
    out = total / count

    def back(g):
        gg = _expand_reduced(g, t.data.shape, dim, keepdim)
        n = _expand_reduced(count, t.data.shape, dim, keepdim) if dim is not None else count
        return (_np.where(bad, 0.0, gg / n),)

    return t._make(out, (t,), back, "NanmeanBackward0")


def _expand_reduced(g, shape, dim, keepdim):
    """축약으로 접힌 축을 되살려 원래 모양에 퍼질 수 있게 한다."""
    gg = _np.asarray(g)
    if dim is None:
        return _np.broadcast_to(gg, shape)
    if not keepdim:
        gg = _np.expand_dims(gg, dim)
    return _np.broadcast_to(gg, shape)


def logsumexp(t, dim=None, keepdim=False):
    """`log(sum(exp(x)))` 를 **넘치지 않게** 센다 — 큰 값을 빼고 더한다."""
    t = _wrap(t)
    big = _np.max(t.data, axis=dim, keepdims=True)
    shifted = _np.exp(t.data - big)
    total = shifted.sum(axis=dim, keepdims=True)
    out = _np.log(total) + big
    soft = shifted / total
    if not keepdim:
        out = out.reshape(()) if dim is None else _np.squeeze(out, axis=dim)
    return t._make(out, (t,),
                   lambda g: (_expand_reduced(g, t.data.shape, dim, keepdim) * soft,),
                   "LogsumexpBackward0")


def _cum_extreme(t, dim, pick, name):
    """누적 최대·최소. 값과 **번호**를 준다 — torch 와 같은 모양이다."""
    t = _wrap(t)
    idx = pick(t.data, axis=dim)
    out = _np.take_along_axis(t.data, idx, axis=dim)

    def back(g):
        # 기울기는 **뽑힌 자리로만** 간다. 같은 자리가 여러 번 뽑혔으면 그만큼 쌓인다.
        z = _np.zeros_like(t.data)
        _np.add.at(z, _index_for(idx, dim, t.data.ndim), _np.asarray(g))
        return (z,)

    return _MinMax(t._make(out, (t,), back, name), Tensor(idx.astype(_np.int64)))


def _index_for(idx, dim, ndim):
    """`np.add.at` 에 줄 색인 — 축마다 좌표를 만들어 튜플로 준다."""
    grid = _np.indices(idx.shape)
    return tuple(idx if a == dim % ndim else grid[a] for a in range(ndim))


def _running_idx(better):
    """누적 최대·최소의 **번호**를 낸다. 축을 따라 한 칸씩 간다.

    벡터화로 짜보다 접었다. torch 는 동점일 때 **나중 자리**를 준다(실측:
    [1,3,3,2] 의 cummax 번호가 [0,1,2,2] — i=2 에서 1 이 아니라 2 다). 그 규칙은
    `>=` 한 글자에 담기는데, 억지로 벡터화하면 그 한 글자가 안 보이는 곳으로 숨는다.
    """
    def make(d, axis):
        moved = _np.moveaxis(d, axis, 0)
        idx = _np.zeros(moved.shape, dtype=_np.intp)
        best = moved[0].copy()
        for i in range(1, moved.shape[0]):
            take = better(moved[i], best)
            idx[i] = _np.where(take, i, idx[i - 1])
            best = _np.where(take, moved[i], best)
        return _np.moveaxis(idx, 0, axis)
    return make


def cummax(t, dim):
    return _cum_extreme(t, dim, _running_idx(lambda cur, best: cur >= best),
                        "CummaxBackward0")


def cummin(t, dim):
    return _cum_extreme(t, dim, _running_idx(lambda cur, best: cur <= best),
                        "CumminBackward0")


def kthvalue(t, k, dim=-1, keepdim=False):
    """**k 번째로 작은** 값. torch 는 1 부터 센다."""
    t = _wrap(t)
    order = _np.argsort(t.data, axis=dim, kind="stable")
    at = _np.take(order, k - 1, axis=dim)
    at_e = _np.expand_dims(at, dim)
    out = _np.take_along_axis(t.data, at_e, axis=dim)
    if not keepdim:
        out = _np.squeeze(out, axis=dim)

    def back(g):
        z = _np.zeros_like(t.data)
        gg = _np.asarray(g)
        _np.put_along_axis(z, at_e, gg if keepdim else _np.expand_dims(gg, dim), axis=dim)
        return (z,)

    return _MinMax(t._make(out, (t,), back, "KthvalueBackward0"),
                   Tensor(at.astype(_np.int64)))


def msort(t):
    """**첫 축을 따라** 정렬한다. `sort(dim=0)` 의 값 쪽과 같다."""
    return sort(_wrap(t), dim=0).values


def diff(t, n=1, dim=-1):
    """이웃한 것의 차. `x[1:] - x[:-1]` 을 n 번 한다.

    **자르기로 짠다** — 자르기가 이미 그래프를 이으므로 역방향을 새로 쓸 것이 없다.
    """
    out = _wrap(t)
    axis = dim % out.data.ndim
    for _ in range(n):
        length = out.data.shape[axis]
        out = out[_slice_at(axis, 1, length)] - out[_slice_at(axis, 0, length - 1)]
    return out


def dist(a, b, p=2):
    """두 텐서 사이의 거리 — `norm(a - b, p)` 다."""
    return norm(_wrap(a) - _wrap(b), p=p)


def quantile(t, q, dim=None, keepdim=False):
    """분위수. torch 의 기본은 **선형 보간**이고 numpy 와 같다."""
    t = _wrap(t)
    qq = q.data if isinstance(q, Tensor) else _np.asarray(q, dtype=t.data.dtype)
    out = _np.quantile(t.data, qq, axis=dim, keepdims=keepdim)
    return Tensor(_np.asarray(out, dtype=t.data.dtype))


def nanquantile(t, q, dim=None, keepdim=False):
    t = _wrap(t)
    qq = q.data if isinstance(q, Tensor) else _np.asarray(q, dtype=t.data.dtype)
    out = _np.nanquantile(t.data, qq, axis=dim, keepdims=keepdim)
    return Tensor(_np.asarray(out, dtype=t.data.dtype))


def nonzero(t):
    """0 이 아닌 자리의 좌표. **모양이 값에 달렸다** — 그래서 기울기가 없다."""
    return Tensor(_np.stack(_np.nonzero(_wrap(t).data), axis=-1).astype(_np.int64))


def argwhere(t):
    return nonzero(t)


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


def _order(data, dim, descending):
    """정렬 번호. **동점끼리의 순서가 답의 일부다.**

    오름차순으로 정렬한 뒤 뒤집으면 같은 값끼리의 순서까지 뒤집혀서, torch 가 앞에
    두는 쪽(번호가 작은 쪽)이 뒤로 간다. 부호를 뒤집어 안정 정렬하면 유지된다.
    numpy 의 기본 정렬은 quicksort 라 안정적이지 않으므로 오름차순도 명시한다 —
    지금 맞는 것은 우연이고, 입력이 길어지면 갈린다.
    """
    return _np.argsort(-data if descending else data, axis=dim, kind="stable")


def topk(t, k, dim=-1, largest=True):
    """상위 k개의 (값, 번호). 32장의 top-k 샘플링이 이것이다."""
    t = _wrap(t)
    order = _order(t.data, dim, largest)
    idx = _np.take(order, _np.arange(k), axis=dim)
    return _MinMax(_pick(t, idx, dim, "TopkBackward0"), Tensor(idx))


def sort(t, dim=-1, descending=False):
    t = _wrap(t)
    idx = _order(t.data, dim, descending)
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


# ── 활성함수 열일곱 ─────────────────────────────────────────────────────────
#
# **꺾이는 점에서 어느 쪽을 고르는지가 전부다.** 식은 문서에 있지만 `x == 0`,
# `x == ±3`, `x == 6` 처럼 정확히 경계인 자리에서 torch 가 무엇을 주는지는 재봐야
# 안다 — 난수 입력은 그 점을 절대 안 준다. 골든의 `kinks` 가 그 점들이다.
#
# 상수는 torch 의 것을 그대로 적는다. 근사치를 쓰면 값이 5 자리쯤에서 갈리고,
# 그것은 "거의 맞는" 상태라 오래 안 걸린다.
_SELU_ALPHA = 1.6732632423543772848170429916717
_SELU_SCALE = 1.0507009873554804934193349852946


def _sigmoid_of(d):
    return 1.0 / (1.0 + _np.exp(-_np.clip(d, -60, 60)))


def celu(t, alpha=1.0):
    """CELU. `ELU` 와 달리 음수 쪽을 α 로 **나눈 뒤** 지수를 취한다.

    α=1 이면 ELU 와 같은 값이 나온다 — 그래서 α 를 안 주고 재면 둘을 못 가른다.
    """
    t = _wrap(t)
    pick = t.data > 0
    inner = _np.exp(_np.minimum(t.data, 0) / alpha)
    out = _np.where(pick, t.data, alpha * (inner - 1))
    return t._make(out, (t,), lambda g: (g * _np.where(pick, 1.0, inner),),
                   "CeluBackward0")


def hardshrink(t, lambd=0.5):
    """|x| > λ 면 그대로, 아니면 0. **경계에서는 0 이다**(`>` 이지 `>=` 가 아니다)."""
    t = _wrap(t)
    keep = _np.abs(t.data) > lambd
    return t._make(_np.where(keep, t.data, 0.0), (t,),
                   lambda g: (g * keep,), "HardshrinkBackward0")


def hardsigmoid(t):
    """구간별 직선으로 시그모이드를 흉내 낸다. 꺾이는 점이 ±3 이다."""
    t = _wrap(t)
    d = t.data
    out = _np.clip(d / 6.0 + 0.5, 0.0, 1.0)
    inside = (d > -3.0) & (d < 3.0)
    return t._make(out, (t,), lambda g: (g * _np.where(inside, 1.0 / 6.0, 0.0),),
                   "HardsigmoidBackward0")


def hardswish(t):
    """x·hardsigmoid(x). 모바일 쪽에서 swish 대신 쓰는 것."""
    t = _wrap(t)
    d = t.data
    out = _np.where(d <= -3.0, 0.0, _np.where(d >= 3.0, d, d * (d + 3.0) / 6.0))
    grad = _np.where(d <= -3.0, 0.0, _np.where(d >= 3.0, 1.0, (2.0 * d + 3.0) / 6.0))
    return t._make(out, (t,), lambda g: (g * grad,), "HardswishBackward0")


def hardtanh(t, min_val=-1.0, max_val=1.0):
    t = _wrap(t)
    d = t.data
    inside = (d > min_val) & (d < max_val)
    return t._make(_np.clip(d, min_val, max_val), (t,),
                   lambda g: (g * inside,), "HardtanhBackward0")


def logsigmoid(t):
    """log σ(x). **큰 음수에서 곧장 계산하면 log(0) 이 된다** — 안정형으로 쓴다."""
    t = _wrap(t)
    d = t.data
    out = -(_np.logaddexp(0.0, -d))
    sig = _sigmoid_of(d)
    return t._make(out.astype(d.dtype), (t,), lambda g: (g * (1.0 - sig),),
                   "LogSigmoidBackward0")


def softplus(t, beta=1.0, threshold=20.0):
    """(1/β)·log(1+e^{βx}). **βx 가 threshold 를 넘으면 그냥 x 다** — 넘치지 않게.

    그 갈래를 빠뜨리면 큰 입력에서 `inf` 가 나오고, 그 뒤 기울기가 전부 NaN 이 된다.
    """
    t = _wrap(t)
    d = t.data
    big = beta * d > threshold
    out = _np.where(big, d, _np.logaddexp(0.0, beta * d) / beta)
    sig = _sigmoid_of(beta * d)
    return t._make(out.astype(d.dtype), (t,),
                   lambda g: (g * _np.where(big, 1.0, sig),), "SoftplusBackward0")


def mish(t):
    """x·tanh(softplus(x))."""
    t = _wrap(t)
    d = t.data
    sp = _np.logaddexp(0.0, d)
    th = _np.tanh(sp)
    sig = _sigmoid_of(d)
    out = (d * th).astype(d.dtype)
    grad = th + d * (1.0 - th * th) * sig
    return t._make(out, (t,), lambda g: (g * grad.astype(d.dtype),), "MishBackward0")


def relu6(t):
    """clamp(x, 0, 6). **경계에서 기울기가 0 이다** — 양쪽 다."""
    t = _wrap(t)
    d = t.data
    inside = (d > 0.0) & (d < 6.0)
    return t._make(_np.clip(d, 0.0, 6.0), (t,), lambda g: (g * inside,),
                   "Relu6Backward0")


def selu(t):
    t = _wrap(t)
    d = t.data
    pick = d > 0
    inner = _np.exp(_np.minimum(d, 0))
    out = _SELU_SCALE * _np.where(pick, d, _SELU_ALPHA * (inner - 1))
    grad = _SELU_SCALE * _np.where(pick, 1.0, _SELU_ALPHA * inner)
    return t._make(out.astype(d.dtype), (t,), lambda g: (g * grad.astype(d.dtype),),
                   "SeluBackward0")


def softshrink(t, lambd=0.5):
    """λ 만큼 **원점 쪽으로 당긴다.** `hardshrink` 와 달리 값이 이어진다."""
    t = _wrap(t)
    d = t.data
    out = _np.where(d > lambd, d - lambd, _np.where(d < -lambd, d + lambd, 0.0))
    keep = _np.abs(d) > lambd
    return t._make(out.astype(d.dtype), (t,), lambda g: (g * keep,),
                   "SoftshrinkBackward0")


def softsign(t):
    """x/(1+|x|)."""
    t = _wrap(t)
    d = t.data
    denom = 1.0 + _np.abs(d)
    return t._make((d / denom).astype(d.dtype), (t,),
                   lambda g: (g / (denom * denom),), "SoftsignBackward0")


def tanhshrink(t):
    """x − tanh(x)."""
    t = _wrap(t)
    d = t.data
    th = _np.tanh(d)
    return t._make((d - th).astype(d.dtype), (t,), lambda g: (g * (th * th),),
                   "TanhshrinkBackward0")


def threshold(t, threshold, value):                      # noqa: A002
    """x > threshold 면 그대로, 아니면 `value`. **경계는 value 쪽이다.**"""
    t = _wrap(t)
    keep = t.data > threshold
    return t._make(_np.where(keep, t.data, value), (t,), lambda g: (g * keep,),
                   "ThresholdBackward0")


def softmin(t, dim=-1):
    """softmax(−x). **부호를 빠뜨리면 softmax 와 같아진다** — 값으로만 갈린다."""
    return softmax(-_wrap(t), dim=dim)


def glu(t, dim=-1):
    """축을 반으로 갈라 `a · σ(b)`. 활성함수 중 유일하게 원소별이 아니다."""
    t = _wrap(t)
    n = t.data.shape[dim]
    if n % 2:
        raise RuntimeError(f"glu 는 축 {dim} 의 길이가 짝수여야 합니다 (지금 {n})")
    half = n // 2
    a = narrow(t, dim, 0, half)
    b = narrow(t, dim, half, half)
    return a * sigmoid(b)


def prelu(t, weight):
    """음수 쪽 기울기가 **학습된다.** 가중치가 하나면 전 채널이 나눠 쓴다.

    **정확히 0 은 음수 쪽이다.** 순방향은 어느 쪽으로 놓아도 0 이라 안 보이는데,
    기울기는 갈린다 — torch 는 `x > 0` 일 때만 1 을 주고 `x == 0` 에는 w 를 준다.
    처음에 `x < 0` 으로 갈랐더니 그 한 점에서 최대차 3.75 가 났고, 골든의 `kinks`
    입력에 0 이 들어 있어서 잡혔다. 난수 입력이었으면 영원히 안 걸렸다.
    """
    t, weight = _wrap(t), _wrap(weight)
    d = t.data
    w = weight.data
    if w.size != 1:
        # 채널마다 다른 기울기 — 채널 축(1번)에 맞춰 편다.
        shape = [1] * d.ndim
        shape[1 if d.ndim > 1 else 0] = w.size
        w = w.reshape(shape)
    pos = d > 0
    out = _np.where(pos, d, w * d)

    def back(g):
        g = _np.asarray(g)
        dx = g * _np.where(pos, 1.0, w)
        dw = _unbroadcast(g * _np.where(pos, 0.0, d), weight.data.shape)
        return (dx, dw)

    return t._make(out.astype(d.dtype), (t, weight), back, "PreluBackward0")


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
    """살아남은 값을 `1/(1-p)` 로 키운다 — **그래야 학습과 추론의 크기가 맞는다.**

    `p=1` 을 따로 가른다. 안 가르면 `1/(1-p)` 가 0 으로 나누기가 되어 NaN 이 나오고,
    NaN 은 자기 자신과도 달라서 그 뒤로는 무엇을 비교해도 통과할 수 없다. torch 는
    그 자리에서 0 을 준다.
    """
    if not training or p == 0:
        return _wrap(t)
    t = _wrap(t)
    if p == 1:
        return t * Tensor(_np.zeros_like(t.data))
    mask = (_rng.random(t.data.shape) > p).astype(t.data.dtype) / (1 - p)
    return t * Tensor(mask)


def avg_pool2d(x, kernel_size, stride=None):
    """**축마다 다른 창을 받는다.** `adaptive_avg_pool2d` 가 세로·가로를 다르게 줄일 수
    있어야 해서다 — 정사각만 받으면 그 위에 못 얹는다."""
    kh, kw = _pair(kernel_size)
    sh, sw = _pair(stride if stride is not None else kernel_size)
    xd = x.data
    N, C, H, W = xd.shape
    OH = (H - kh) // sh + 1
    OW = (W - kw) // sw + 1
    win = _np.lib.stride_tricks.sliding_window_view(xd, (kh, kw), axis=(2, 3))
    win = win[:, :, ::sh, ::sw, :, :]
    out = win.mean(axis=(4, 5))
    area = kh * kw

    def back(g):
        g = _np.asarray(g) / area
        gx = _np.zeros_like(xd)
        for i in range(kh):
            for j in range(kw):
                gx[:, :, i:i + OH * sh:sh, j:j + OW * sw:sw] += g
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


# ---------------------------------------------------------------- 선형대수(분해)
#
# numpy 가 다 해준다. **기울기는 닫힌 꼴이 있는 것만 넣는다** — `det`·`logdet`·
# `inverse`·`solve`·`cholesky` 가 그렇고, 이 다섯은 유도해서 torch 와 대조했다
# (콜레스키는 최대차 2.8e-17).
#
# `qr`·`svd`·`pinverse`·`lstsq` 는 값만 준다. torch 는 이것들도 미분하는데 우리는 안 한다 —
# 유도가 까다롭고(특히 특잇값이 겹칠 때) 틀리면 조용히 틀린다. `backward()` 가 거절하므로
# 없다는 것이 시끄럽게 드러난다.

class LinAlgError(RuntimeError):
    """torch 의 `linalg.LinAlgError`.

    **이름이 하는 일이 있다.** 특이행렬을 만난 코드가 `except linalg.LinAlgError` 로
    감싸는데, 우리가 다른 것을 던지면 그 감싸기를 지나쳐 프로그램이 죽는다.
    numpy 는 `ValueError` 밑에 두지만 torch 는 `RuntimeError` 밑이라 이쪽을 따른다.
    """


def _named(kind, *fields):
    """이름 붙은 결과를 만든다.

    **torch 의 linalg 는 자리로도 이름으로도 물을 수 있다** — `slogdet(A)[1]` 과
    `slogdet(A).logabsdet` 이 같은 것이다. 자리만 맞춰 두면 값이 맞는데도 교재
    코드가 속성 접근에서 멈춘다. `lstsq` 가 `.solution` 으로 그 자리를 이미 겪었고,
    그때 클래스를 손으로 하나 적었다 — 그것을 여덟 번 더 적는 대신 여기서 찍는다.
    """
    class _R:
        __slots__ = fields

        def __init__(self, *vals):
            for f, v in zip(fields, vals):
                setattr(self, f, v)

        def __iter__(self):
            for f in fields:
                yield getattr(self, f)

        def __len__(self):
            return len(fields)

        def __getitem__(self, i):
            return getattr(self, fields[i])

        def __repr__(self):
            inner = ", ".join(f"{f}={getattr(self, f)!r}" for f in fields)
            return f"torch.return_types.{kind}(\n{inner})"

    _R.__name__ = _R.__qualname__ = kind
    return _R


_Slogdet = _named("slogdet", "sign", "logabsdet")
_QR = _named("linalg_qr", "Q", "R")
_SVD = _named("linalg_svd", "U", "S", "Vh")
_Eigh = _named("linalg_eigh", "eigenvalues", "eigenvectors")
_Lstsq = _named("linalg_lstsq", "solution", "residuals", "rank", "singular_values")
_LuFactor = _named("linalg_lu_factor", "LU", "pivots")
_Lu = _named("linalg_lu", "P", "L", "U")
_InvEx = _named("linalg_inv_ex", "inverse", "info")
_CholeskyEx = _named("linalg_cholesky_ex", "L", "info")
_SolveEx = _named("linalg_solve_ex", "result", "info")


# ---- 배치
#
# **torch 의 `linalg` 는 전부 배치다.** `det((3,2,2))` 이 `(3,)` 을 내고 `inv`·`solve`·
# `cholesky`·`slogdet`·`matrix_rank` 가 다 그렇다. 앞의 `_mat` 은 2 차원이 아니면
# 거절했는데, 그건 흉내가 아니라 없는 것이었다 — 배치는 실제 코드가 늘 쓰는 모양이다.
#
# numpy 의 `linalg` 도 마지막 두 축을 행렬로 보고 앞을 배치로 돈다. 그래서 순방향은
# 거의 그대로 열리고, **손이 가는 곳은 역방향이다.** 전치를 `.T` 로 적으면 2 차원에서만
# 맞고 배치에서는 축을 통째로 뒤집어 조용히 틀린다. 아래는 전부 `_T` 를 쓴다.

def _T(a):
    """마지막 두 축만 바꾼다. 배치 축은 그대로 둔다."""
    return _np.swapaxes(a, -1, -2)


def _mat(t, what, square=True):
    t = _wrap(t)
    if t.data.ndim < 2:
        _unsupported(f"{what}(2차원 미만)")
    if square and t.data.shape[-1] != t.data.shape[-2]:
        _unsupported(f"{what}(정사각이 아닌 것)")
    return t


def _guard(what, fn, *args, **kw):
    """numpy 가 특이행렬에서 던지는 것을 우리 이름으로 바꿔 단다."""
    try:
        return fn(*args, **kw)
    except _np.linalg.LinAlgError as exc:
        raise LinAlgError(f"linalg.{what}: {exc}") from None


def _is_singular(data):
    """특이한가 — **판정을 우리가 한다.**

    numpy 에 맡기면 답이 numpy 가 **무엇으로 빌드됐는지**에 달린다. 같은 입력
    `[[1,2],[2,4]]` 에 네이티브 numpy 는 `LinAlgError` 를 던지고 Pyodide 안의
    numpy 는 조용히 지나갔다 — `inv` 도 `cholesky` 도 그랬다(실측). 그러면 "특이
    행렬에서 무엇이 나는가" 가 **사용자의 브라우저에 달린다.** 그건 라이브러리가
    정할 일이지 밑에 깔린 LAPACK 이 정할 일이 아니다.

    부분 피벗 LU 의 대각에 정확히 0 이 있으면 특이다 — LAPACK 이 쓰는 것과 같은
    기준이고, 셈이 우리 것이라 어디서 돌든 같은 답이 나온다.
    """
    packed, _ = _lu_pack(data)
    k = min(packed.shape[-2], packed.shape[-1])
    idx = _np.arange(k)
    return bool(_np.any(packed[..., idx, idx] == 0))


def _reject_singular(data, what):
    if _is_singular(data):
        raise LinAlgError(
            f"linalg.{what}: 특이행렬이다 — 역행렬이 없다 (The diagonal element of "
            "the factorization is zero)")


def _cholesky_checked(data, what):
    """콜레스키. **양정부호 확인도 우리가 한다** — 위와 같은 이유다."""
    try:
        low = _np.linalg.cholesky(data)
    except _np.linalg.LinAlgError:
        low = None
    if low is None or not _np.all(low[..., _np.arange(low.shape[-1]),
                                      _np.arange(low.shape[-1])] > 0):
        raise LinAlgError(
            f"linalg.{what}: 대칭 양정부호가 아니다 (matrix is not positive definite)")
    return low


# **역행렬은 역방향에서 구한다.** 앞의 판에서는 순방향에서 미리 구했는데, 그러면
# `det(특이행렬)` 이 0 을 못 내고 던진다 — torch 는 멀쩡히 0 을 낸다. 미분할 수 없는
# 것은 미분할 때 말하면 되고, 값을 묻는 사람까지 막을 이유가 없다.

def det(t):
    t = _mat(t, "det")
    out = _np.linalg.det(t.data)

    def back(g):
        inv_t = _T(_guard("det", _np.linalg.inv, t.data))
        # 행렬식은 배치마다 스칼라다 — 행렬에 곱하려면 축 둘을 세워 줘야 한다.
        return ((_np.asarray(g) * out)[..., None, None] * inv_t,)

    return t._make(_np.asarray(out, dtype=t.data.dtype), (t,), back, "DetBackward0")


def logdet(t):
    t = _mat(t, "logdet")
    sign, logabs = _np.linalg.slogdet(t.data)
    out = _np.where(sign > 0, logabs, _np.nan)
    return t._make(_np.asarray(out, dtype=t.data.dtype), (t,),
                   lambda g: (_np.asarray(g)[..., None, None]
                              * _T(_guard("logdet", _np.linalg.inv, t.data)),),
                   "LogdetBackward0")


def slogdet(t):
    t = _mat(t, "slogdet")
    sign, logabs = _np.linalg.slogdet(t.data)
    return _Slogdet(Tensor(_np.asarray(sign, dtype=t.data.dtype)),
                    t._make(_np.asarray(logabs, dtype=t.data.dtype), (t,),
                            lambda g: (_np.asarray(g)[..., None, None]
                                       * _T(_guard("slogdet", _np.linalg.inv, t.data)),),
                            "SlogdetBackward0"))


def inverse(t):
    """역행렬. 기울기는 `-A⁻ᵀ G A⁻ᵀ` 다."""
    t = _mat(t, "inverse")
    _reject_singular(t.data, "inv")
    out = _guard("inv", _np.linalg.inv, t.data)
    out_t = _T(out)
    return t._make(out, (t,),
                   lambda g: (-(out_t @ _np.asarray(g) @ out_t),), "InverseBackward0")


def inv_ex(t, check_errors=False):
    """`inv` 와 같은데 **안 던진다** — 대신 `info` 에 0 이 아닌 수를 담는다.

    배치에서 이 구분이 살아난다. 스무 장 중 한 장이 특이일 때 던지는 쪽은 전부를
    죽이고, 이쪽은 어느 장이 상했는지를 `info` 로 알려 준다.
    """
    t = _mat(t, "inv_ex")
    try:
        _reject_singular(t.data, "inv_ex")
        out = _np.linalg.inv(t.data)
        info = _np.zeros(t.data.shape[:-2], dtype=_np.int32)
    except LinAlgError:
        if check_errors:
            raise
        out = _np.full_like(t.data, _np.inf)
        # LAPACK 은 몇 번째 주피벗이 0 인지를 담는다. 여기서는 "상했다" 만 말한다.
        info = _np.full(t.data.shape[:-2], _SINGULAR_INFO, dtype=_np.int32)
        return _InvEx(Tensor(out), Tensor(info))
    return _InvEx(t._make(out, (t,),
                          lambda g: (-(_T(out) @ _np.asarray(g) @ _T(out)),),
                          "InverseBackward0"), Tensor(info))


# LAPACK 이 특이행렬에서 담는 값과 자릿수를 맞춘다. 실측: 2×2 특이행렬에 2 였다.
_SINGULAR_INFO = 2


def solve(a, b):
    """`A x = b` 를 푼다. 역행렬을 만들어 곱하는 것보다 정확하고 빠르다.

    `b` 가 `A` 보다 축이 하나 적으면 **벡터 묶음**으로 본다. 역방향의 바깥곱이 그
    구분에 걸린다.

    **numpy 에 맡기면 안 되는 자리다.** numpy 2.0 이 규칙을 바꿔서, 이제 `b` 가
    1 차원일 때만 벡터 묶음으로 본다 — `A(3,2,2)` 에 `b(3,2)` 를 주면 행렬로 읽고
    차원이 안 맞다고 던진다. torch 는 옛 규칙 그대로다. 그래서 축을 여기서 세운다.
    """
    a = _mat(a, "solve")
    _reject_singular(a.data, "solve")
    bt = _wrap(b)
    vector = bt.data.ndim == a.data.ndim - 1
    rhs = bt.data[..., None] if vector else bt.data
    x = _guard("solve", _np.linalg.solve, a.data, rhs)
    if vector:
        x = x[..., 0]
    inv_t = _T(_guard("solve", _np.linalg.inv, a.data))

    def back(g):
        gg = _np.asarray(g)
        if vector:
            gb = (inv_t @ gg[..., None])[..., 0]
            ga = -(gb[..., :, None] * x[..., None, :])
        else:
            gb = inv_t @ gg
            ga = -(gb @ _T(x))
        return (ga, gb)

    return a._make(x, (a, bt), back, "SolveBackward0")


def solve_ex(a, b, check_errors=False):
    """`solve` 의 안 던지는 쪽."""
    a = _mat(a, "solve_ex")
    try:
        out = solve(a, b)
    except LinAlgError:
        if check_errors:
            raise
        bt = _wrap(b)
        return _SolveEx(Tensor(_np.full_like(bt.data, _np.inf)),
                        Tensor(_np.full(a.data.shape[:-2], _SINGULAR_INFO,
                                        dtype=_np.int32)))
    return _SolveEx(out, Tensor(_np.zeros(a.data.shape[:-2], dtype=_np.int32)))


def _cholesky_raw(data, upper):
    low = _np.linalg.cholesky(data.astype(_np.float64))
    return _T(low) if upper else low


def cholesky(t, upper=False):
    """`A = L Lᵀ` 의 아래삼각 `L`. **기울기가 있다** — Murray 알고리즘을 유도해
    torch 와 대조했고 최대차가 2.8e-17 이었다."""
    t = _mat(t, "cholesky")
    low = _cholesky_checked(t.data.astype(_np.float64), "cholesky")
    idx = _np.arange(low.shape[-1])

    def back(g):
        gg = _np.asarray(g, dtype=_np.float64)
        if upper:
            gg = _T(gg)
        bar = _T(low) @ gg
        half = _np.tril(bar).copy()
        # 대각만 반으로. 배치에서는 `diag_indices_from` 이 축을 잘못 잡는다.
        half[..., idx, idx] *= 0.5
        low_inv = _np.linalg.inv(low)
        sym = _T(low_inv) @ half @ low_inv
        return (((sym + _T(sym)) * 0.5).astype(t.data.dtype),)

    out = (_T(low) if upper else low).astype(t.data.dtype)
    return t._make(out, (t,), back, "CholeskyBackward0")


def cholesky_ex(t, upper=False, check_errors=False):
    """`cholesky` 의 안 던지는 쪽. 양정부호가 아니면 `info` 가 0 이 아니다."""
    t = _mat(t, "cholesky_ex")
    try:
        out = cholesky(t, upper=upper)
    except LinAlgError:
        if check_errors:
            raise
        return _CholeskyEx(Tensor(_np.full_like(t.data, _np.nan)),
                           Tensor(_np.full(t.data.shape[:-2], _SINGULAR_INFO,
                                           dtype=_np.int32)))
    return _CholeskyEx(out, Tensor(_np.zeros(t.data.shape[:-2], dtype=_np.int32)))


def matrix_power(t, n):
    """**곱셈을 이어 붙여 만든다** — 그러면 역방향이 저절로 따라온다.
    분해로 짜면 미분식을 새로 써야 하고, 그건 틀릴 자리를 하나 더 만드는 것이다."""
    t = _mat(t, "matrix_power")
    if n < 0:
        return matrix_power(inverse(t), -n)
    if n == 0:
        eye = _np.eye(t.data.shape[-1], dtype=t.data.dtype)
        return Tensor(_np.broadcast_to(eye, t.data.shape).copy())
    out = t
    for _ in range(n - 1):
        out = out @ t
    return out


def qr(t, mode="reduced"):
    """QR 분해. **값만 준다** — 기울기는 안 넣었다(위 주석 참고)."""
    t = _mat(t, "qr", square=False)
    q, r = _np.linalg.qr(t.data, mode=mode)
    return _QR(Tensor(_np.ascontiguousarray(q)), Tensor(_np.ascontiguousarray(r)))


def svd(t, full_matrices=True):
    """특잇값 분해. **값만 준다.** torch 와 같이 (U, S, Vh) 순서로 돌려준다."""
    t = _mat(t, "svd", square=False)
    u, s, vh = _np.linalg.svd(t.data, full_matrices=full_matrices)
    return _SVD(Tensor(_np.ascontiguousarray(u)), Tensor(s),
                Tensor(_np.ascontiguousarray(vh)))


def pinverse(t, rcond=1e-15):
    """유사역행렬. **값만 준다.**"""
    t = _mat(t, "pinverse", square=False)
    return Tensor(_np.linalg.pinv(t.data, rcond=rcond))


def matrix_rank(t, tol=None):
    t = _mat(t, "matrix_rank", square=False)
    return Tensor(_np.asarray(_np.linalg.matrix_rank(t.data, tol=tol), dtype=_np.int64))


def eigh(t, UPLO="L"):
    """대칭 행렬의 고윳값·고유벡터. **값만 준다.**

    **한쪽 삼각만 읽는다.** 기본은 아래쪽이라 `[[4,99],[1,3]]` 과 `[[4,1],[1,3]]` 의
    답이 같다(진짜 torch 에 물어서 확인했다). 대칭을 주는 한 안 드러나는 규약이라,
    행렬 전체를 보는 구현과 여기서 조용히 갈린다.
    """
    t = _mat(t, "eigh")
    w, v = _np.linalg.eigh(t.data, UPLO=UPLO)
    return _Eigh(Tensor(w), Tensor(_np.ascontiguousarray(v)))


def lstsq(a, b):
    """최소제곱 해. **값만 준다.**

    `.solution` 으로 물어야 한다 — torch 는 해 말고 잔차·랭크·특잇값도 같이 준다.
    그냥 텐서를 돌려주면 torch 코드가 `.solution` 에서 멈추고, 그건 "값이 맞는데
    안 통하는" 자리가 된다.
    """
    a, bt = _mat(a, "lstsq", square=False), _wrap(b)
    if a.data.ndim != 2:
        _unsupported("lstsq(배치)")
    sol, res, rank, sv = _np.linalg.lstsq(a.data, bt.data, rcond=None)
    return _Lstsq(Tensor(_np.ascontiguousarray(sol)), Tensor(_np.asarray(res)),
                  Tensor(_np.asarray(rank, dtype=_np.int64)), Tensor(_np.asarray(sv)))


# ---- LU
#
# LU 분해는 이미 `det`·`inv`·`solve` 밑에서 돌고 있었다. 밖으로 안 낸 것뿐이다.
#
# **피벗은 1 부터 센다.** LAPACK 규약이고 torch 가 그대로 물려받았다 — 교환이 없는
# 2×2 에서 `pivots` 가 `[1, 2]` 이지 `[0, 1]` 이 아니다. 0 부터 세면 `lu_solve` 가
# 아무 소리 없이 다른 답을 낸다. 실측해서 맞췄다.

def _lu_pack(data):
    """부분 피벗 LU. `LU` 한 장과 **1 부터 세는** 교환표를 준다."""
    a = data.astype(_np.float64).copy()
    n, m = a.shape[-2], a.shape[-1]
    k = min(n, m)
    flat = a.reshape(-1, n, m)
    piv = _np.zeros((flat.shape[0], k), dtype=_np.int32)
    for b in range(flat.shape[0]):
        mat = flat[b]
        for col in range(k):
            best = col + int(_np.argmax(_np.abs(mat[col:, col])))
            piv[b, col] = best + 1                      # ← LAPACK 은 1 부터 센다
            if best != col:
                mat[[col, best]] = mat[[best, col]]
            pivot = mat[col, col]
            if pivot == 0:
                continue
            mat[col + 1:, col] /= pivot
            mat[col + 1:, col + 1:] -= _np.outer(mat[col + 1:, col], mat[col, col + 1:])
    return flat.reshape(a.shape), piv.reshape(data.shape[:-2] + (k,))


def lu_factor(t):
    """`LU` 한 장에 겹쳐 담은 분해와 교환표. 값만 준다."""
    t = _mat(t, "lu_factor", square=False)
    lu_data, piv = _lu_pack(t.data)
    return _LuFactor(Tensor(lu_data.astype(t.data.dtype)), Tensor(piv))


def lu(t, pivot=True):
    """`P L U` 셋으로 펴서 준다. 겹쳐 담은 것보다 읽기 쉽다."""
    t = _mat(t, "lu", square=False)
    if not pivot:
        _unsupported("lu(pivot=False)")
    lu_data, piv = _lu_pack(t.data)
    n, m = lu_data.shape[-2], lu_data.shape[-1]
    k = min(n, m)
    flat_lu = lu_data.reshape(-1, n, m)
    flat_piv = piv.reshape(-1, k)
    diag = _np.arange(k)
    ps, ls, us = [], [], []
    for b in range(flat_lu.shape[0]):
        low = _np.tril(flat_lu[b][:, :k], -1).copy()
        low[diag, diag] = 1.0
        ls.append(low)
        us.append(_np.triu(flat_lu[b])[:k, :])
        # 교환을 되짚어 순열 행렬을 세운다. `piv` 는 1 부터 세므로 하나 뺀다.
        order = _np.arange(n)
        for col in range(k):
            src = int(flat_piv[b, col]) - 1
            if src != col:
                order[[col, src]] = order[[src, col]]
        perm = _np.zeros((n, n), dtype=_np.float64)
        perm[order, _np.arange(n)] = 1.0
        ps.append(perm)

    def pack(arrs, shape):
        return Tensor(_np.asarray(arrs).reshape(shape).astype(t.data.dtype))

    lead = t.data.shape[:-2]
    return _Lu(pack(ps, lead + (n, n)), pack(ls, lead + (n, k)),
               pack(us, lead + (k, m)))


def lu_solve(lu_data, pivots, b, left=True, adjoint=False):
    """`lu_factor` 가 낸 것으로 `A x = b` 를 푼다."""
    lu_t, piv_t, bt = _wrap(lu_data), _wrap(pivots), _wrap(b)
    if not left or adjoint:
        _unsupported("lu_solve(left=False 또는 adjoint=True)")
    n = lu_t.data.shape[-1]
    low = _np.tril(lu_t.data.astype(_np.float64), -1) + _np.eye(n)
    up = _np.triu(lu_t.data.astype(_np.float64))
    rhs = bt.data.astype(_np.float64).copy()
    order = _np.arange(n)
    for col in range(piv_t.data.shape[-1]):
        src = int(_np.asarray(piv_t.data).reshape(-1)[col]) - 1
        if src != col:
            order[[col, src]] = order[[src, col]]
    rhs = rhs[order]
    y = _np.linalg.solve(low, rhs)
    return Tensor(_np.linalg.solve(up, y).astype(bt.data.dtype))


# ---- 조합층
#
# 대부분 이미 있는 것에 이름을 붙이는 자리다. 계산이 새로 필요한 것은 `matrix_exp`
# 하나뿐인데, 그 하나가 닫힌 식이 없다.

def matmul(a, b):
    return _wrap(a) @ _wrap(b)


def vecdot(a, b, dim=-1):
    return (_wrap(a) * _wrap(b)).sum(dim=dim)


def diagonal_linalg(t, offset=0, dim1=-2, dim2=-1):
    """**`torch.diagonal` 과 기본 축이 다르다.**

    이쪽은 마지막 두 축(`-2, -1`)을 보고 저쪽은 앞의 두 축(`0, 1`)을 본다. 3 차원을
    주면 `(2,3,4)` 가 각각 `(2,3)` 과 `(4,2)` 로 갈린다 — 이름이 비슷해 같은 것으로
    읽기 쉬운데 모양부터 다르다. 그래서 기본값을 손으로 적어 둔다.
    """
    return diagonal(t, offset=offset, dim1=dim1, dim2=dim2)


def svdvals(t):
    """특잇값만. `svd` 의 가운데다."""
    return svd(t, full_matrices=False).S


def eigvalsh(t, UPLO="L"):
    """대칭 행렬의 고윳값만."""
    return eigh(t, UPLO=UPLO).eigenvalues


def vector_norm(t, ord=2, dim=None, keepdim=False):
    """벡터로 보고 재는 노름. **행렬을 줘도 통째로 편다** — 그것이 `matrix_norm` 과
    갈리는 자리다.

    `ord=0` 은 0 이 아닌 것의 개수이고, `±inf` 는 절댓값의 최대·최소다 — 거듭제곱
    식에 넣으면 안 되는 갈래라 따로 적는다.
    """
    x = _wrap(t).abs()
    if dim is None and x.data.ndim > 1:
        x = x.reshape(-1)
    if ord == _np.inf:
        return amax(x, dim=dim, keepdim=keepdim)
    if ord == -_np.inf:
        return amin(x, dim=dim, keepdim=keepdim)
    if ord == 0:
        return (x != 0).float().sum(dim=dim, keepdim=keepdim)
    if ord == 1:
        return x.sum(dim=dim, keepdim=keepdim)
    if ord == 2:
        return (x * x).sum(dim=dim, keepdim=keepdim) ** 0.5
    return (x ** ord).sum(dim=dim, keepdim=keepdim) ** (1.0 / ord)


# 특잇값이 필요한 갈래. 나머지는 행·열의 절댓값 합으로 끝난다.
_SPECTRAL = ("nuc", 2, -2)


def matrix_norm(t, ord="fro", dim=(-2, -1), keepdim=False):
    """행렬로 보고 재는 노름. **갈래마다 다른 수다.**

    기본은 프로베니우스이고, `2` 는 최대 특잇값, `nuc` 는 특잇값의 합, `1` 은 열
    절댓값 합의 최대, `inf` 는 행 쪽이다. rank 1 행렬을 주면 앞의 셋이 우연히 같아져
    구분이 안 되므로 골든은 rank 2 로 묻는다.
    """
    x = _wrap(t)
    if dim != (-2, -1):
        x = x.movedim(dim[0], -2).movedim(dim[1], -1)
    if ord in _SPECTRAL:
        s = svdvals(x)
        if ord == "nuc":
            return s.sum(dim=-1, keepdim=keepdim)
        return (amax if ord == 2 else amin)(s, dim=-1, keepdim=keepdim)
    if ord == "fro":
        return (x * x).sum(dim=(-2, -1), keepdim=keepdim) ** 0.5
    # 1 은 열 방향(행을 더한다), inf 는 행 방향(열을 더한다). 부호는 최대·최소를 가른다.
    #
    # **`abs(ord)` 라고 쓰면 안 된다.** 이 모듈에 `abs` 가 있어서 파이썬 내장을 가리고,
    # 정수를 텐서로 알고 `'int' object has no attribute 'abs'` 로 멈춘다. 이 저장소가
    # 같은 함정을 `bool`·`max`·`min` 에서 이미 세 번 밟았다 — 네 번째다.
    axis = -2 if ord in (1, -1) else -1
    sums = x.abs().sum(dim=axis, keepdim=True)
    pick = amax if ord > 0 else amin
    out = pick(sums, dim=-1 if axis == -2 else -2, keepdim=True)
    return out if keepdim else out.reshape(out.shape[:-2])


def cond(t, p=None):
    """조건수. 기본은 `‖A‖₂·‖A⁻¹‖₂` 이고 그것은 특잇값의 비다."""
    x = _mat(t, "cond")
    if p is None or p == 2:
        s = svdvals(x)
        return amax(s, dim=-1) / amin(s, dim=-1)
    if p == -2:
        s = svdvals(x)
        return amin(s, dim=-1) / amax(s, dim=-1)
    return matrix_norm(x, ord=p) * matrix_norm(inverse(x), ord=p)


def multi_dot(mats):
    """행렬 여럿을 이어 곱한다. **묶는 순서가 값을 안 바꾼다** — 곱셈이 결합적이라
    그렇다. 바뀌는 것은 셈의 개수뿐이라, 여기서는 순서대로 곱한다."""
    out = _wrap(mats[0])
    for m in mats[1:]:
        out = out @ _wrap(m)
    return out


def vander(x, N=None):
    """반데르몽드 행렬. 열이 **커지는 차수**다 — numpy 의 기본과 반대라 뒤집는다."""
    v = _wrap(x)
    n = v.data.shape[-1] if N is None else N
    cols = [v ** k for k in range(n)]
    return stack(cols, dim=-1)


def solve_triangular(a, b, upper, left=True, unitriangular=False):
    """삼각행렬이라는 것을 **알고 푼다.** 앞으로·뒤로 한 번씩이면 끝난다.

    `unitriangular` 는 **대각을 안 보고 1 로 친다** — 안 지키면 값이 조용히 달라지는
    갈래다. `left=False` 는 `X A = B` 를 푸는 것이라 양쪽을 전치해 같은 길로 보낸다.
    """
    at, bt = _mat(a, "solve_triangular"), _wrap(b)
    tri = _np.triu(at.data) if upper else _np.tril(at.data)
    if unitriangular:
        idx = _np.arange(tri.shape[-1])
        tri = tri.copy()
        tri[..., idx, idx] = 1.0
    if not left:
        x = _np.linalg.solve(_T(tri), _T(bt.data))
        return Tensor(_T(x))
    return Tensor(_np.linalg.solve(tri, bt.data))


def tensorsolve(a, b, dims=None):
    """텐서를 행렬로 접어 풀고 다시 편다."""
    at, bt = _wrap(a), _wrap(b)
    if dims is not None:
        _unsupported("tensorsolve(dims)")
    n = bt.data.size
    out = _np.linalg.solve(at.data.reshape(n, -1), bt.data.reshape(n))
    return Tensor(out.reshape(at.data.shape[bt.data.ndim:]))


def tensorinv(a, ind=2):
    at = _wrap(a)
    lead = at.data.shape[:ind]
    n = int(_np.prod(lead))
    out = _np.linalg.inv(at.data.reshape(n, -1))
    return Tensor(out.reshape(at.data.shape[ind:] + lead))


# 스케일링·제곱에서 무엇을 "작다" 로 볼지. 1-노름이 이 아래면 테일러가 빨리 모인다.
_EXP_SMALL = 0.5
# 그 조건에서 필요한 항의 개수. 0.5^18/18! 은 배정도의 바닥보다 한참 아래다.
_EXP_TERMS = 18


def matrix_exp(t):
    """행렬 지수 `e^A`. **스케일링과 제곱으로 간다.**

    테일러만으로는 큰 행렬에서 안 모인다 — `A*5` 의 답이 4.8e+10 인데, 그 자리에서는
    항이 커지는 쪽이 먼저 넘친다. `A/2^s` 의 1-노름을 0.5 아래로 낮춰 급수를 태운 뒤
    `s` 번 제곱하면 같은 답이 안전하게 나온다(`e^A = (e^{A/2^s})^{2^s}`).

    **값만 준다.** torch 는 미분하는데 우리는 안 한다.
    """
    x = _mat(t, "matrix_exp")
    a = x.data.astype(_np.float64)
    n = a.shape[-1]
    norm = _np.abs(a).sum(axis=-2).max() if a.size else 0.0
    squarings = max(0, int(_np.ceil(_np.log2(norm / _EXP_SMALL)))) if norm > _EXP_SMALL \
        else 0
    scaled = a / (2.0 ** squarings)
    eye = _np.broadcast_to(_np.eye(n), a.shape).copy()
    term, out = eye.copy(), eye.copy()
    for k in range(1, _EXP_TERMS + 1):
        term = term @ scaled / k
        out = out + term
    for _ in range(squarings):
        out = out @ out
    return Tensor(out.astype(x.data.dtype))


class _Linalg(_Namespace):
    """`torch.linalg` 자리. 같은 구현을 가리키므로 갈릴 자리가 없다."""

    LinAlgError = LinAlgError
    det = staticmethod(det)
    slogdet = staticmethod(slogdet)
    inv = staticmethod(inverse)
    inv_ex = staticmethod(inv_ex)
    solve = staticmethod(solve)
    solve_ex = staticmethod(solve_ex)
    cholesky = staticmethod(cholesky)
    cholesky_ex = staticmethod(cholesky_ex)
    matrix_power = staticmethod(matrix_power)
    qr = staticmethod(qr)
    svd = staticmethod(svd)
    pinv = staticmethod(pinverse)
    matrix_rank = staticmethod(matrix_rank)
    eigh = staticmethod(eigh)
    lstsq = staticmethod(lstsq)
    lu = staticmethod(lu)
    lu_factor = staticmethod(lu_factor)
    lu_solve = staticmethod(lu_solve)
    norm = staticmethod(norm)
    # 조합층.
    matmul = staticmethod(matmul)
    vecdot = staticmethod(vecdot)
    cross = staticmethod(cross)
    diagonal = staticmethod(diagonal_linalg)
    svdvals = staticmethod(svdvals)
    eigvalsh = staticmethod(eigvalsh)
    vector_norm = staticmethod(vector_norm)
    matrix_norm = staticmethod(matrix_norm)
    cond = staticmethod(cond)
    multi_dot = staticmethod(multi_dot)
    vander = staticmethod(vander)
    solve_triangular = staticmethod(solve_triangular)
    tensorsolve = staticmethod(tensorsolve)
    tensorinv = staticmethod(tensorinv)
    matrix_exp = staticmethod(matrix_exp)


linalg = _Linalg()


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


