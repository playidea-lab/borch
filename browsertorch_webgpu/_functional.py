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
    Tensor, _canonical, _relayout, _wrap,
)
from ._base import (
    _pad_const, _shape_of, _to_tf, _unsupported, _warn_once,
)
from ._ops import (
    _rng, _to_int32, abs, gather, gt, maximum, norm, relu, sigmoid, tanh,
)

# ---------------------------------------------------------------- nn.functional

def softmax(t, dim=-1):
    t = _canonical(t)
    out = _tf.softmax(t._h, dim)

    def back(g):
        s = _tf.sum(_tf.mul(g, out), dim, True)
        return (_tf.mul(out, _tf.sub(g, s)),)

    return t._make(out, (t,), back, "SoftmaxBackward0")


def log_softmax(t, dim=-1):
    t = _canonical(t)
    out = _tf.logSoftmax(t._h, dim)
    soft = _tf.exp(out)

    def back(g):
        return (_tf.sub(g, _tf.mul(soft, _tf.sum(g, dim, True))),)

    return t._make(out, (t,), back, "LogSoftmaxBackward0")


def leaky_relu(t, negative_slope=0.01):
    t = _canonical(t)
    pick = _tf.cast(_tf.greater(t._h, 0.0), "float32")
    return t._make(
        _tf.leakyRelu(t._h, float(negative_slope)), (t,),
        lambda g: (_tf.mul(g, _tf.add(pick, _tf.mul(_tf.sub(1.0, pick), float(negative_slope)))),),
        "LeakyReluBackward0")


def elu(t, alpha=1.0):
    t = _canonical(t)
    out = _tf.elu(t._h)
    pick = _tf.cast(_tf.greater(t._h, 0.0), "float32")
    return t._make(
        out, (t,),
        lambda g: (_tf.mul(g, _tf.add(pick, _tf.mul(_tf.sub(1.0, pick),
                                                    _tf.add(out, float(alpha))))),),
        "EluBackward0")


def silu(t):
    """x·σ(x). Swish 라고도 한다."""
    t = _canonical(t)
    sig = _tf.sigmoid(t._h)
    return t._make(
        _tf.mul(t._h, sig), (t,),
        lambda g: (_tf.mul(g, _tf.mul(sig, _tf.add(1.0, _tf.mul(t._h, _tf.sub(1.0, sig))))),),
        "SiluBackward0")


_SQRT2 = float(_np.sqrt(2.0))
_SQRT2PI = float(_np.sqrt(2.0 * _np.pi))


def gelu(t):
    """torch 의 기본 gelu(정확형) — 0.5·x·(1 + erf(x/√2)). TF.js 에 erf 가 있다."""
    t = _canonical(t)
    ope = _tf.add(1.0, _tf.erf(_tf.div(t._h, _SQRT2)))

    def back(g):
        bell = _tf.div(_tf.exp(_tf.neg(_tf.div(_tf.square(t._h), 2.0))), _SQRT2PI)
        return (_tf.mul(g, _tf.add(_tf.mul(0.5, ope), _tf.mul(t._h, bell))),)

    return t._make(_tf.mul(0.5, _tf.mul(t._h, ope)), (t,), back, "GeluBackward0")


def dropout(t, p=0.5, training=True):
    if not training or p == 0:
        return _wrap(t)
    t = _wrap(t)
    mask = (_rng.random(t.shape) > p).astype(_np.float32) / (1 - p)
    return t * Tensor(_to_tf(mask))


def layer_norm(x, normalized_shape=None, weight=None, bias=None, eps=1e-5):
    """마지막 축에서 평균·분산을 낸다. 그래프 안에서 계산해야 기울기가 흐른다."""
    x = _canonical(x)
    mean = x.mean(dim=-1, keepdim=True)
    centered = x - mean
    var = (centered * centered).mean(dim=-1, keepdim=True)
    out = centered / (var + eps) ** 0.5
    if weight is not None:
        out = out * weight
    return out + bias if bias is not None else out


def embedding(idx, weight):
    """번호를 벡터로. 원-핫 곱이라 역전파가 따라온다 — 같은 번호가 여러 번 나오면
    그 행에 기울기가 **쌓여야** 하고, 곱셈이 그것을 알아서 해준다."""
    weight = _canonical(weight)
    rows, dim = weight.shape
    flat = _tf.reshape(_to_int32(idx), _to_js([-1]))
    n = _shape_of(flat)[0]
    onehot = _tf.cast(_tf.oneHot(flat, rows), "float32")          # (n, rows)
    out = weight._make(_tf.matMul(onehot, weight._h), (weight,),
                       lambda g: (_tf.matMul(onehot, g, True, False),),
                       "EmbeddingBackward0")
    shape = tuple(idx.shape) if isinstance(idx, Tensor) else _np.asarray(idx).shape
    return out.reshape(tuple(shape) + (dim,))


def linear(x, weight, bias=None):
    out = _wrap(x) @ _canonical(weight).transpose(0, 1)
    return out + bias if bias is not None else out


def binary_cross_entropy_with_logits(logits, target):
    # log(1+e^-|x|) + max(x,0) - x*t — 큰 값에서도 안전한 형태
    x, t = _wrap(logits), _wrap(target)
    return (relu(x) - x * t + (1 + (-(x.abs())).exp()).log()).mean()


def binary_cross_entropy(p, target):
    p, t = _wrap(p), _wrap(target)
    eps = 1e-12
    return -(t * (p + eps).log() + (1 - t) * (1 - p + eps).log()).mean()


def one_hot(t, num_classes=-1):
    t = _canonical(t)
    depth = int(t.numpy().max()) + 1 if num_classes == -1 else int(num_classes)
    return Tensor(_tf.oneHot(_to_int32(t), depth))


def pad(x, padding, value=0.0):
    """마지막 차원부터 (앞, 뒤) 순으로 받는다 — torch 의 규칙이다."""
    x = _canonical(x)
    old = x.shape
    pairs = [[0, 0] for _ in range(x.ndim)]
    for i in range(0, len(padding), 2):
        pairs[-(i // 2 + 1)] = [int(padding[i]), int(padding[i + 1])]

    def back(g):
        return (_tf.slice(g, _to_js([p[0] for p in pairs]), _to_js(list(old))),)

    pads = [(i, p[0], p[1]) for i, p in enumerate(pairs) if p != [0, 0]]
    return x._make(_pad_const(x._h, list(old), pads, value), (x,), back, "PadBackward0")


def normalize(x, p=2, dim=1, eps=1e-12):
    x = _canonical(x)
    denom = norm(x, p=p, dim=dim)
    return x / maximum(unsqueeze(denom, dim), _wrap(eps))


def unsqueeze(t, dim):
    t = _canonical(t)
    old = t.shape
    return t._make(_tf.expandDims(t._h, dim), (t,),
                   lambda g: (_tf.reshape(g, _to_js(list(old))),), "UnsqueezeBackward0")


def cosine_similarity(a, b, dim=1, eps=1e-8):
    a, b = _wrap(a), _wrap(b)
    return (a * b).sum(dim=dim) / maximum(norm(a, dim=dim) * norm(b, dim=dim), _wrap(eps))


def l1_loss(pred, target):
    return abs(_wrap(pred) - _wrap(target)).mean()


def mse_loss(pred, target):
    diff = _wrap(pred) - _wrap(target)
    return (diff * diff).mean()


def smooth_l1_loss(pred, target, beta=1.0):
    """작은 오차는 제곱, 큰 오차는 절댓값. 이상치에 덜 흔들린다."""
    diff = _wrap(pred) - _wrap(target)
    small = _tf.cast(_tf.less(_tf.abs(diff._h), float(beta)), "float32")
    quad = diff * diff * (0.5 / float(beta))
    lin = abs(diff) - 0.5 * float(beta)
    return (Tensor(small) * quad + Tensor(_tf.sub(1.0, small)) * lin).mean()


def nll_loss(log_probs, target):
    log_probs = _wrap(log_probs)
    rows = log_probs.shape[0]
    idx = _tf.reshape(_to_int32(target), _to_js([rows, 1]))
    picked = gather(log_probs, 1, Tensor(_tf.cast(idx, "float32")))
    return -picked.mean()


def cross_entropy(logits, target):
    return nll_loss(log_softmax(logits, dim=-1), target)


# ---------------------------------------------------------------- 합성곱
#
# torch 는 NCHW, TF.js 는 NHWC 다. 축을 바꿔 넣고 되돌린다.
#
# **역방향을 직접 짠다.** TF.js 의 conv 역방향 커널은 순방향의 1/26 이고(S0 실측),
# 같은 계산을 **순방향 conv 로 다시 쓰면 33배 빠르다**. 그래서 `tf.grad` 에 안 맡긴다 —
# 이 라이브러리가 자체 테이프를 드는 이유가 이것이다.

def _to_nhwc(h):
    return _tf.transpose(h, _to_js([0, 2, 3, 1]))


def _to_nchw(h):
    return _tf.transpose(h, _to_js([0, 3, 1, 2]))


def _pair(v):
    return (v, v) if isinstance(v, int) else (int(v[0]), int(v[1]))


def _dilate(handle, stride, extra):
    """기울기 사이에 0 을 끼운다.

    스트라이드가 1보다 크면 순방향이 칸을 건너뛴 것이므로, 역방향에서는 그 자리를
    0 으로 되살려야 스트라이드 1 짜리 conv 로 계산할 수 있다. `extra` 는 입력 크기가
    스트라이드로 나누어떨어지지 않을 때 남는 칸이다.

    처음에는 6차원으로 펴서 `pad` 로 끼웠는데, **모양은 맞고 값이 전부 0 으로 나왔다.**
    던지지도 않았다 — 6차원 경로가 조용히 0 을 준다. 그래서 5차원을 넘지 않게
    `stack` 으로 다시 짰다.
    """
    sh, sw = _pair(stride)
    eh, ew = _pair(extra)
    n, oh, ow, f = _shape_of(handle)

    x = handle
    if sh > 1:
        zeros_h = _tf.zerosLike(x)
        x = _tf.reshape(_tf.stack(_to_js([x] + [zeros_h] * (sh - 1)), 2),
                        _to_js([n, oh * sh, ow, f]))
    if sw > 1:
        zeros_w = _tf.zerosLike(x)
        x = _tf.reshape(_tf.stack(_to_js([x] + [zeros_w] * (sw - 1)), 3),
                        _to_js([n, _shape_of(x)[1], ow * sw, f]))

    keep_h = (oh - 1) * sh + 1 + eh
    keep_w = (ow - 1) * sw + 1 + ew
    return _tf.slice(x, _to_js([0, 0, 0, 0]), _to_js([n, keep_h, keep_w, f]))


def conv2d(x, weight, bias=None, stride=1, padding=0):
    x, weight = _wrap(x), _wrap(weight)
    N, C, H, W = x.shape
    F, C2, KH, KW = weight.shape
    if C != C2:
        raise RuntimeError(f"채널이 안 맞습니다: 입력 {C}, 필터 {C2}")

    # 축별로 받는다 — 1차원 합성곱이 (1, K) 커널로 이 경로를 그대로 쓰기 때문이다.
    sh, sw = _pair(stride)
    ph, pw = _pair(padding)

    # **들어올 때 한 번만 바꾼다.** 이미 NHWC 면 그냥 지나가고, 결과도 NHWC 로 나가서
    # 다음 conv·BN·활성까지 전치 없이 이어진다.
    xin = _relayout(x, True)
    xh = xin._h
    wh = _tf.transpose(weight._h, _to_js([2, 3, 1, 0]))          # (F,C,KH,KW) → (KH,KW,C,F)
    xpad_fwd = xh if (ph, pw) == (0, 0) else _tf.pad(
        xh, _to_js([[0, 0], [ph, ph], [pw, pw], [0, 0]]))
    out = _tf.conv2d(xpad_fwd, wh, _to_js([sh, sw]), "valid")
    bias_t = _wrap(bias) if bias is not None else None
    if bias_t is not None:
        # NHWC 는 마지막 축이 채널이라 1차원 편향이 그대로 붙는다. 모양을 만들 필요가 없다.
        out = _tf.add(out, bias_t._h)
    extra = ((H + 2 * ph - KH) % sh, (W + 2 * pw - KW) % sw)

    def back(gd):                                                # gd 는 NHWC 로 온다
        if (sh, sw) != (1, 1) or extra != (0, 0):
            gd = _dilate(gd, (sh, sw), extra)

        # dx — 커널을 공간 반전하고 입출력 채널을 바꾸면 **순방향 conv 가 된다**
        wflip = _tf.transpose(_tf.reverse(wh, _to_js([0, 1])), _to_js([0, 1, 3, 2]))
        gpad = _tf.pad(gd, _to_js([[0, 0], [KH - 1 - ph, KH - 1 - ph],
                                   [KW - 1 - pw, KW - 1 - pw], [0, 0]]))
        dx = _tf.conv2d(gpad, wflip, 1, "valid")

        # dw — 배치와 채널을 맞바꾸고 기울기를 필터로 쓰면 역시 순방향 conv 가 된다
        xt = _tf.transpose(xpad_fwd, _to_js([3, 1, 2, 0]))       # (C, H', W', N)
        gt = _tf.transpose(gd, _to_js([1, 2, 0, 3]))             # (OH, OW, N, F)
        dw = _tf.transpose(_tf.conv2d(xt, gt, 1, "valid"), _to_js([3, 0, 1, 2]))

        if bias is None:
            return (dx, dw)
        return (dx, dw, _tf.sum(gd, _to_js([0, 1, 2])))

    parents = (xin, weight) if bias_t is None else (xin, weight, bias_t)
    return xin._make(out, parents, back, "ConvolutionBackward0")


_warned = set()


# `_warn_once` 는 `_base` 로 옮겼다 — `_ops` 도 써야 하는데 이 모듈은 그보다 늦게
# 실린다. 여기서는 들여온 것을 그대로 쓴다.


def conv3d(x, weight, bias=None, stride=1, padding=0):
    """3차원 합성곱. **역방향은 TF.js 에 맡긴다 — 느리다.**

    2차원은 역방향을 손으로 써서 `tf.grad` 대비 약 10배를 얻었는데, 그 유도가 3차원으로
    그대로 넘어가지 않는다. 그래서 여기서는 `tf.grad` 를 쓴다. 값은 맞고 속도는 그만큼
    못 나온다 — 처음 부를 때 한 번 경고한다.
    """
    x, weight = _canonical(x), _canonical(weight)
    n, c, d, h, w = x.shape
    f, c2, kd, kh, kw = weight.shape
    if c != c2:
        raise RuntimeError(f"채널이 안 맞습니다: 입력 {c}, 필터 {c2}")
    _warn_once("conv3d",
               "conv3d 의 역방향은 TF.js 커널을 씁니다 — 2차원처럼 손으로 쓰지 않아서 "
               "느립니다. 부피 데이터가 주 용도라면 그 유도부터 하는 것이 순서입니다.")

    sd, sh, sw = (stride, stride, stride) if isinstance(stride, int) else tuple(stride)
    pd, ph, pw = (padding, padding, padding) if isinstance(padding, int) else tuple(padding)
    ncdhw_to_ndhwc, back_perm = [0, 2, 3, 4, 1], [0, 4, 1, 2, 3]

    xh = _tf.transpose(x._h, _to_js(ncdhw_to_ndhwc))
    xh = _pad_const(xh, [n, d, h, w, c], [(1, pd, pd), (2, ph, ph), (3, pw, pw)])
    wh = _tf.transpose(weight._h, _to_js([2, 3, 4, 1, 0]))    # (F,C,D,H,W) → (D,H,W,C,F)
    strides = _to_js([sd, sh, sw])
    out = _tf.conv3d(xh, wh, strides, "valid")

    bias_t = _wrap(bias) if bias is not None else None
    result = out if bias_t is None else _tf.add(out, bias_t._h)

    def back(g):
        gh = _tf.transpose(g, _to_js(ncdhw_to_ndhwc))
        fx = _create_proxy(lambda t: _tf.conv3d(t, wh, strides, "valid"))
        fw = _create_proxy(lambda t: _tf.conv3d(xh, t, strides, "valid"))
        try:
            dx_pad = _tf.grad(fx)(xh, gh)
            dw = _tf.grad(fw)(wh, gh)
        finally:
            fx.destroy()
            fw.destroy()
        if (pd, ph, pw) != (0, 0, 0):
            dx_pad = _tf.slice(dx_pad, _to_js([0, pd, ph, pw, 0]),
                               _to_js([n, d, h, w, c]))
        grads = [_tf.transpose(dx_pad, _to_js(back_perm)),
                 _tf.transpose(dw, _to_js([4, 3, 0, 1, 2]))]
        if bias_t is not None:
            grads.append(_tf.sum(gh, _to_js([0, 1, 2, 3])))
        return tuple(grads)

    parents = (x, weight) if bias_t is None else (x, weight, bias_t)
    return x._make(_tf.transpose(result, _to_js(back_perm)), parents, back,
                   "Conv3DBackward0")


def max_pool3d(x, kernel_size, stride=None):
    """`MaxPool2d` 와 같은 방식이다 — 역방향을 `tf.grad` 에 맡긴다.
    최댓값 자리를 우리가 다시 만들면 동점에서 torch 와 갈리기 때문이고, 2차원에서
    이미 그렇게 하고 있으므로 여기서 새로 치르는 대가는 없다."""
    x = _canonical(x)
    stride = stride or kernel_size
    ncdhw_to_ndhwc, back_perm = [0, 2, 3, 4, 1], [0, 4, 1, 2, 3]
    xh = _tf.transpose(x._h, _to_js(ncdhw_to_ndhwc))
    k, s = _to_js([kernel_size] * 3), _to_js([stride] * 3)
    out = _tf.maxPool3d(xh, k, s, "valid")

    def back(g):
        fn = _create_proxy(lambda t: _tf.maxPool3d(t, k, s, "valid"))
        try:
            dx = _tf.grad(fn)(xh, _tf.transpose(g, _to_js(ncdhw_to_ndhwc)))
        finally:
            fn.destroy()
        return (_tf.transpose(dx, _to_js(back_perm)),)

    return x._make(_tf.transpose(out, _to_js(back_perm)), (x,), back, "MaxPool3DBackward0")


def conv1d(x, weight, bias=None, stride=1, padding=0):
    """`(N,C,L)` 을 `(N,C,1,L)` 로 세워 **검증된 2차원 경로**를 그대로 쓴다.

    TF.js 에 `conv1d` 가 있지만 부르지 않는다 — 그쪽 역방향은 우리가 손으로 쓴 것보다
    느린 커널을 탄다(2차원에서 26배를 쟀다). 세워서 쓰면 그 이득이 1차원에도 그대로 온다.
    """
    x, weight = _wrap(x), _canonical(weight)
    n, c, length = x.shape
    f, c2, k = weight.shape
    out = conv2d(x.reshape(n, c, 1, length), weight.reshape(f, c2, 1, k),
                 bias, (1, stride), (0, padding))
    on, of, _, ol = out.shape
    return out.reshape(on, of, ol)


def max_pool1d(x, kernel_size, stride=None):
    x = _wrap(x)
    n, c, length = x.shape
    out = max_pool2d(x.reshape(n, c, 1, length), (1, kernel_size),
                     (1, stride or kernel_size))
    on, oc, _, ol = out.shape
    return out.reshape(on, oc, ol)


def interpolate(x, scale_factor=2, mode="nearest"):
    """최근접 확대. 한 칸이 s×s 로 복제되므로 **역방향은 그 블록을 합하는 것**이다."""
    if mode != "nearest":
        _unsupported(f"interpolate(mode={mode!r}) — 최근접만 있습니다")
    xin = _relayout(_wrap(x), True)
    _, h, w, _ = _shape_of(xin._h)
    sh, sw = _pair(scale_factor)
    out = _tf.image.resizeNearestNeighbor(xin._h, _to_js([h * sh, w * sw]))

    def back(g):
        pooled = _tf.avgPool(g, _to_js([sh, sw]), _to_js([sh, sw]), "valid")
        return (_tf.mul(pooled, float(sh * sw)),)

    return xin._make(out, (xin,), back, "UpsampleBackward0")


def max_pool2d(x, kernel_size, stride=None):
    """역방향은 **번호를 받아서 그 자리에만** 흘린다.

    원래 `tf.grad(maxPool)` 에 맡기고 "동점일 때 정확한 쪽"이라고 적어 두었는데,
    **그 가정이 틀렸다.** 창 안에 최댓값이 둘 있으면 TF.js 는 기울기를 양쪽에 나눠
    주고 torch 는 먼저 나온 자리 하나에만 준다(안에서 argmax 를 쓴다). 값 대조로는
    순방향이 같아서 안 잡히고, `edge::grad::max_pool2d(동점)` 이 최대차 3 으로 잡았다.

    흔한 자리다. ReLU 뒤에는 정확히 0 이 널려 있어서, 창이 통째로 0 이면 매번 동점이다.

    `maxPoolWithArgmax` 가 이긴 자리의 **평평한 번호**를 준다. 그 번호로 흩뿌리면
    torch 와 같아진다 — 나누는 일이 없다.
    """
    xin = _relayout(_wrap(x), True)
    stride = stride or kernel_size
    xh = xin._h
    ksize, strides = _to_js(list(_pair(kernel_size))), _to_js(list(_pair(stride)))
    # **번호에 배치를 포함시킨다(마지막 인자).** 기본값은 배치를 빼고 세는 것이라
    # 배치가 여럿이면 같은 번호가 겹쳐서, 흩뿌릴 때 첫 장으로 전부 몰린다.
    picked = _tf.maxPoolWithArgmax(xh, ksize, strides, "valid", True)
    out, where = picked.result, picked.indexes
    shape = list(_shape_of(xh))
    kh, kw = _pair(kernel_size)
    sh, sw = _pair(stride)
    # 창과 걸음이 같고 나누어떨어지면 창끼리 안 겹친다 — 그때만 아래의 정확한 길이 선다.
    tidy = (kh, kw) == (sh, sw) and shape[1] % kh == 0 and shape[2] % kw == 0

    total = shape[0] * shape[1] * shape[2] * shape[3]
    full = _to_js([shape[1], shape[2]])

    def back(g):
        if tidy:
            # **동점이면 번호가 가장 작은 자리로 간다** — torch 가 그렇게 한다.
            # `maxPoolWithArgmax` 도 자리 하나를 고르기는 하는데 동점에서 고르는 자리가
            # torch 와 다르다. 창이 안 겹치면 창의 최댓값을 도로 펼쳐서 같은 자리를 전부
            # 찾고, 그중 최소 번호를 다시 창으로 접어 정확히 하나만 남길 수 있다.
            idx = _tf.reshape(_tf.range(0, float(total), 1, "float32"), _to_js(shape))
            big = _tf.mul(_tf.onesLike(idx), float(total + 1))
            spread = _tf.image.resizeNearestNeighbor(out, full)
            masked = _tf.where(_tf.equal(xh, spread), idx, big)
            # 최댓값 풀링으로 최소를 내려면 부호를 뒤집는다.
            low = _tf.neg(_tf.maxPool(_tf.neg(masked), ksize, strides, "valid"))
            keep = _tf.cast(_tf.equal(masked, _tf.image.resizeNearestNeighbor(low, full)),
                            "float32")
            return (_tf.mul(keep, _tf.image.resizeNearestNeighbor(g, full)),)
        # 겹치는 창은 자리를 나눠 갖지 않으므로 위의 펼치기가 안 선다. 번호를 그대로
        # 쓴다 — 동점이 아니면 정확하고, 동점이면 torch 와 다른 자리를 고른다.
        flat = _tf.reshape(_tf.cast(where, "int32"), _to_js([-1, 1]))
        spread = _tf.scatterND(flat, _tf.reshape(g, _to_js([-1])), _to_js([total]))
        return (_tf.reshape(spread, _to_js(shape)),)

    return xin._make(out, (xin,), back, "MaxPool2DBackward0")


def batch_norm(x, weight, bias, eps=1e-5):
    """학습 모드 배치 정규화 — **역전파를 손으로 썼다.**

    우리 연산으로 조립하면 레이아웃마다 축이 달라져 다루기 까다롭고 커널도 는다.
    식은 알려진 그대로다.

        x̂ = (x-μ)/√(σ²+ε),   y = γ·x̂ + β
        dx = γ/(m√(σ²+ε)) · (m·dy − Σdy − x̂·Σ(dy·x̂))

    (μ, σ², 결과)를 돌려준다 — running 통계는 부르는 쪽이 갱신한다. torch 가 정규화에는
    편향 분산을, running_var 에는 비편향을 쓰는 것도 거기서 처리한다.
    """
    x, weight, bias = _wrap(x), _wrap(weight), _wrap(bias)
    raw = _shape_of(x._h)
    rank = len(raw)
    # 채널 축만 남기고 나머지를 접는다. 랭크를 안 따지므로 1·2·3차원 정규화가
    # 같은 함수를 쓴다 — (N,C) 든 (N,C,H,W) 든 (N,C,D,H,W) 든.
    caxis = rank - 1 if x._nhwc else 1
    reduced = [i for i in range(rank) if i != caxis]
    axes = _to_js(reduced)
    m = float(_np.prod([raw[i] for i in reduced]))
    broadcast = [1] * rank
    broadcast[caxis] = raw[caxis]
    bshape = _to_js(broadcast)

    mu = _tf.mean(x._h, axes, True)
    centered = _tf.sub(x._h, mu)
    var = _tf.mean(_tf.square(centered), axes, True)
    inv = _tf.rsqrt(_tf.add(var, float(eps)))
    xhat = _tf.mul(centered, inv)
    gamma = _tf.reshape(weight._h, bshape)
    out = _tf.add(_tf.mul(xhat, gamma), _tf.reshape(bias._h, bshape))

    def back(g):
        dxhat = _tf.mul(g, gamma)
        s1 = _tf.sum(dxhat, axes, True)
        s2 = _tf.sum(_tf.mul(dxhat, xhat), axes, True)
        dx = _tf.mul(_tf.div(inv, m),
                     _tf.sub(_tf.sub(_tf.mul(m, dxhat), s1), _tf.mul(xhat, s2)))
        return (dx, _tf.sum(_tf.mul(g, xhat), axes), _tf.sum(g, axes))

    return x._make(out, (x, weight, bias), back, "NativeBatchNormBackward0"), mu, var


def adaptive_avg_pool2d(x, output_size=1):
    """출력 크기를 정해 평균 풀링.

    **1 만 받던 것을 넓혔다.** 코어에 같은 것을 넣고 같은 케이스를 자매에도 물었더니
    거기서 걸렸다 — 코어에서 되고 자매에서 안 되는, 방향만 반대인 같은 비대칭이었다.
    (ResNet 이 쓰는 것은 1 이라 그것만으로 오래 버텼다.)

    1 은 평균 한 번으로 끝내고, 그 밖은 `avg_pool2d` 에 넘긴다 — 우리 연산으로
    조립하므로 역전파가 그냥 따라온다.
    """
    xin = _relayout(_wrap(x), True)
    _, h, w, _ = _shape_of(xin._h)
    oh, ow = _pair(output_size)
    if (oh, ow) != (1, 1):
        if h % oh or w % ow:
            _unsupported("adaptive_avg_pool2d(입력이 출력의 배수가 아닌 경우)")
        return avg_pool2d(x, (h // oh, w // ow))
    out = _tf.mean(xin._h, _to_js([1, 2]), True)                 # (N,1,1,C)

    def back(g):
        return (_tf.div(_tf.mul(_tf.onesLike(xin._h), g), float(h * w)),)

    return xin._make(out, (xin,), back, "MeanBackward1")


def avg_pool2d(x, kernel_size, stride=None):
    xin = _relayout(_wrap(x), True)
    stride = stride or kernel_size
    ksize, strides = _to_js(list(_pair(kernel_size))), _to_js(list(_pair(stride)))
    out = _tf.avgPool(xin._h, ksize, strides, "valid")

    def back(g):
        fn = _create_proxy(lambda t: _tf.avgPool(t, ksize, strides, "valid"))
        try:
            return (_tf.grad(fn)(xin._h, g),)
        finally:
            fn.destroy()

    return xin._make(out, (xin,), back, "AvgPool2DBackward0")


class _Functional:
    softmax = staticmethod(softmax)
    log_softmax = staticmethod(log_softmax)
    relu = staticmethod(relu)
    sigmoid = staticmethod(sigmoid)
    tanh = staticmethod(tanh)
    leaky_relu = staticmethod(leaky_relu)
    elu = staticmethod(elu)
    silu = staticmethod(silu)
    gelu = staticmethod(gelu)
    one_hot = staticmethod(one_hot)
    pad = staticmethod(pad)
    normalize = staticmethod(normalize)
    cosine_similarity = staticmethod(cosine_similarity)
    l1_loss = staticmethod(l1_loss)
    mse_loss = staticmethod(mse_loss)
    smooth_l1_loss = staticmethod(smooth_l1_loss)
    nll_loss = staticmethod(nll_loss)
    cross_entropy = staticmethod(cross_entropy)
    avg_pool2d = staticmethod(avg_pool2d)
    conv1d = staticmethod(conv1d)
    conv2d = staticmethod(conv2d)
    max_pool1d = staticmethod(max_pool1d)
    max_pool2d = staticmethod(max_pool2d)
    interpolate = staticmethod(interpolate)
    conv3d = staticmethod(conv3d)
    max_pool3d = staticmethod(max_pool3d)
    adaptive_avg_pool2d = staticmethod(adaptive_avg_pool2d)
    dropout = staticmethod(dropout)
    layer_norm = staticmethod(layer_norm)
    embedding = staticmethod(embedding)
    linear = staticmethod(linear)
    binary_cross_entropy = staticmethod(binary_cross_entropy)
    binary_cross_entropy_with_logits = staticmethod(binary_cross_entropy_with_logits)


