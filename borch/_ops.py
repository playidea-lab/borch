"""borch 를 쪼갠 조각. 공개 이름은 __init__ 이 모은다."""

import builtins as _builtins
import math as _math

import numpy as _np

from ._tensor import (
    Tensor, _MinMax, _grad_mode, _unbroadcast, result_type,
)
from ._base import (
    _DEFAULT_DTYPE, _TYPE_NAMES, _like_torch, _math, _needs_float, _np, _refuses_bool,
    _refuses_nonfloat_kernel, _resolve, _unsupported, Size, dtype,
)
# **이름을 바꿔 들여온다.** `float32` 같은 이름을 이 파일 전역에 두면 아래에서
# 그 이름을 쓰는 함수와 부딪힌다 — `bool` 을 그렇게 가려서 한 번 겪었다.
# **`from . import _fft` 로 쓰면 안 된다.** 그 꼴은 부모 패키지의 **속성**을 보는데,
# 이 파일이 도는 동안 `borch/__init__` 은 아직 반쯤 초기화된 상태다. 네이티브에서는
# 지나갔고 **Pyodide 안에서만** `cannot import name '_fft' from partially initialized
# module` 로 멈췄다 — 골든의 `repr::스칼라` 하나가 그것을 잡았다. 하위 모듈을 직접
# 들여오는 꼴은 그 속성을 안 본다.
from ._fft import fft as _fft_fft, fftfreq as _fft_fftfreq
from ._fft import fftshift as _fft_fftshift, ifft as _fft_ifft
from ._fft import ifftshift as _fft_ifftshift, irfft as _fft_irfft
from ._fft import istft as _fft_istft, rfft as _fft_rfft
from ._fft import rfftfreq as _fft_rfftfreq, stft as _fft_stft
from ._base import bool_ as _bool_dtype, float32 as _float32
from ._base import float64 as _float64, int64 as _int64

# ---------------------------------------------------------------- 만들기

def tensor(data, dtype=None, requires_grad=False):
    """**언제나 사본을 뜬다.** torch 가 그렇게 문서화되어 있고, 공유하고 싶으면
    `from_numpy` 를 쓴다.

    `_np.asarray` 만 쓰면 이미 맞는 형인 ndarray 는 **그대로 통과해서 공유된다.**
    그러면 `t = torch.tensor(arr); t.add_(1)` 이 사용자의 `arr` 까지 바꾼다 — 예외도
    경고도 없고, 진짜 torch 에서는 안 그러므로 그 코드가 자기 컴퓨터에서 다르게 돈다.

    이 저장소의 케이스 파일이 실제로 그것에 걸렸다. 입력을 공유하던 케이스 하나가
    `plain` 을 제자리에서 1 만큼 올렸는데, **torch 는 사본을 떠서 안 샜고 코어만
    샜다.** 그래서 그 뒤에 오는 케이스들이 코어에서만 틀렸고, 원인이 자기 케이스에
    없어서 열여섯 자리를 헤맸다.
    """
    if isinstance(data, Tensor):
        data = data.data
    return Tensor(_np.array(data, dtype=_resolve(data, dtype), copy=True),
                  requires_grad)


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
    """씨앗을 심는다. **있는 생성기의 상태를 갈아 끼운다 — 새로 만들지 않는다.**

    앞의 판은 `global _rng` 로 이름을 다시 묶었다. 그런데 `_nn.py` 가 임포트 때
    `from ._ops import _rng` 로 **그 시점의 물건**을 붙잡아 두므로, 다시 묶어도 저쪽은
    옛 생성기를 계속 쓴다. 그 결과가 이랬다(실측):

        borch.manual_seed(0); Linear(4,3).weight   ← 매번 다른 값
        borch.manual_seed(0); borch.randn(3)       ← 재현된다

    **층 초기화와 dropout 이 씨앗을 안 따랐다.** 학습을 재현하려는 사람이 가장 먼저
    기대하는 것이 그 둘인데, `randn` 만 재현되니 "씨앗이 먹는다" 고 읽고 넘어간다.
    골든이 오래 못 본 것은 케이스마다 가중치를 밖에서 넣어 주기 때문이다 — 게으른
    층이 초기화를 스스로 하면서 처음으로 그 자리가 물어졌다.

    상태를 갈아 끼우면 **누가 어디서 붙잡아 갔든** 같이 고쳐진다. 이름을 다시 묶는
    쪽은 붙잡아 간 자리를 전부 찾아 고쳐야 하고, 그 목록은 늘어난다.
    """
    _rng.bit_generator.state = _np.random.default_rng(seed).bit_generator.state
    _LAST_SEED[0] = int(seed)
    return seed


# 마지막으로 심은 씨앗. `initial_seed` 가 이것을 답한다.
_LAST_SEED = [0]


def initial_seed():
    """마지막으로 심은 씨앗. `manual_seed` 를 안 불렀으면 0 이다 — 우리 생성기가
    그 씨앗으로 시작하므로 사실이다."""
    return _LAST_SEED[0]


def seed():
    """씨앗을 **아무거나** 새로 심고 그것을 답한다. torch 도 그렇게 한다."""
    got = int(_np.random.SeedSequence().entropy % (2 ** 63))
    manual_seed(got)
    return got


def get_rng_state():
    """생성기의 상태를 통째로 담아 준다. **텐서가 아니라 우리 상태 그대로다.**

    torch 는 바이트를 담은 `uint8` 텐서를 준다. 그 바이트 배치는 torch 의 Mersenne
    Twister 내부라 흉내낼 것이 없고, 흉내내면 **남의 상태를 우리 것으로 읽는** 일이
    생긴다. 여기서는 우리 생성기의 상태를 그대로 담고, `set_rng_state` 가 그것만
    받는다 — 오가는 짝이 맞으므로 이어서 학습하기가 뜻대로 된다.
    """
    return dict(_rng.bit_generator.state)


def set_rng_state(state):
    """`get_rng_state` 가 준 것을 되돌린다. 그 짝 말고는 안 받는다."""
    if not isinstance(state, dict):
        _unsupported("set_rng_state — `get_rng_state` 가 준 것만 받습니다")
    _rng.bit_generator.state = state
    return None


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


def max_pool1d(x, kernel_size, stride=None, return_indices=False):
    """`max_pool2d` 에 높이 1 을 끼워 넣는다. 높이가 1 이고 창도 1 이라 그 축은 안 움직인다."""
    if return_indices:
        return max_pool1d_with_indices(x, kernel_size, stride)
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


def max_pool3d(x, kernel_size, stride=None, return_indices=False):
    """깊이 방향은 잘라서 최댓값을 겹쳐 취하고, 나머지는 `max_pool2d` 가 한다."""
    if return_indices:
        return max_pool3d_with_indices(x, kernel_size, stride)
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


# ── 이긴 자리를 함께 내는 판 ─────────────────────────────────────────────────
#
# torch 는 같은 계산에 이름을 둘 준다: `max_pool2d(..., return_indices=True)` 와
# `max_pool2d_with_indices(...)`. 값은 하나가 내고 나머지는 이름만 얹는다.

def max_pool1d_with_indices(x, kernel_size, stride=None, padding=0, dilation=1,
                            ceil_mode=False, **_):
    """**torch 최상위에도 이 이름이 있다.** 그쪽은 자리로만 받으므로 나머지 셋까지
    자리를 열어 두고, 기본값이 아니면 시끄럽게 거절한다."""
    if padding or dilation != 1 or ceil_mode:
        _unsupported("max_pool1d_with_indices(padding·dilation·ceil_mode)")
    x = _wrap(x)
    windows = _fixed_windows(x.data.shape[2], kernel_size, stride or kernel_size)
    out, pos = _max_with_index(x, [windows])
    return out, Tensor(pos)


def max_pool2d_with_indices(x, kernel_size, stride=None, **_):
    x = _wrap(x)
    out, pos = _max_with_index(
        x, _fixed_window_axes(x.data.shape, kernel_size, stride))
    return out, Tensor(pos)


def max_pool3d_with_indices(x, kernel_size, stride=None, **_):
    x = _wrap(x)
    out, pos = _max_with_index(
        x, _fixed_window_axes(x.data.shape, kernel_size, stride))
    return out, Tensor(pos)


def adaptive_max_pool1d_with_indices(x, output_size, **_):
    return _adaptive_with_indices(x, _spread(output_size, 1))


def adaptive_max_pool2d_with_indices(x, output_size, **_):
    return _adaptive_with_indices(x, _pair(output_size))


def adaptive_max_pool3d_with_indices(x, output_size, **_):
    return _adaptive_with_indices(x, _spread(output_size, 3))


def _adaptive_with_indices(x, sizes):
    x = _wrap(x)
    shape = x.data.shape
    axes = [_adaptive_windows(shape[2 + k], sizes[k]) for k in range(len(sizes))]
    out, pos = _max_with_index(x, axes)
    return out, Tensor(pos)


def _unpool(x, indices, kernel_size, stride, padding, output_size, spatial):
    """`max_pool` 이 고른 자리로 값을 **되돌려 놓는다.** 나머지는 0 이다.

    자리표는 풀링이 낸 그 번호이고, 평면 안의 평평한 번호다. 그래서 이 함수는
    자리를 새로 계산하지 않는다 — 계산하면 그 계산이 풀링과 갈릴 수 있고, 값이
    0 이 아닌 자리가 조금 옮겨 앉은 그림은 눈으로 안 보인다.

    기본 출력 크기는 `(n-1)·stride - 2·padding + kernel`. 풀링이 버린 자투리는
    되살릴 수 없으므로 torch 는 `output_size` 로 직접 주는 길도 연다.
    """
    x = _wrap(x)
    shape = x.data.shape
    ks = _spread(kernel_size, spatial)
    st = _spread(stride if stride is not None else kernel_size, spatial)
    pd = _spread(padding, spatial)
    if output_size is None:
        out_spatial = tuple((shape[2 + k] - 1) * st[k] - 2 * pd[k] + ks[k]
                            for k in range(spatial))
    else:
        got = tuple(output_size)
        out_spatial = got[-spatial:]        # torch 는 전체 모양도 받는다
    plane = int(_np.prod(out_spatial))

    pos = _np.asarray(indices.data if isinstance(indices, Tensor) else indices)
    base = (_np.arange(shape[0] * shape[1]) * plane).reshape(shape[0], shape[1],
                                                             *([1] * spatial))
    flat = (base + pos).reshape(-1)
    out_shape = (shape[0], shape[1]) + tuple(out_spatial)

    filled = _np.zeros(shape[0] * shape[1] * plane, dtype=x.data.dtype)
    filled[flat] = x.data.reshape(-1)

    def back(g):
        # 값이 간 자리에서 그대로 받아 온다 — 채우기의 반대다.
        return (_np.asarray(g).reshape(-1)[flat].reshape(shape),)

    return x._make(filled.reshape(out_shape), (x,), back, "MaxUnpoolBackward0")


# ── CTC ────────────────────────────────────────────────────────────────────
#
# 소리와 글자를 **자리를 맞추지 않고** 잇는 손실이다. 음성 인식에서 "이 5 프레임이
# 어느 글자에 해당하는가" 를 사람이 안 적어도 되게 해 준다 — 가능한 정렬을 전부
# 더해서 확률을 낸다.
#
# 더할 정렬의 수가 지수라 앞으로 훑기(forward algorithm)로 접는다. 표적 사이에
# **공백을 끼운 확장 표적**을 만들고(`[_ , l1, _, l2, …, _]`), 한 시각에 상태 하나를
# 잡아 앞 셋에서만 올 수 있게 한다.

_CTC_NEG = -1e30       # 로그 확률의 "없음". `-inf` 는 logsumexp 에서 NaN 이 된다


def _ctc_extended(labels, blank):
    """`[l1, l2]` → `[_, l1, _, l2, _]`. 공백을 사이사이에 끼운다.

    **같은 글자가 이어지면 사이에 공백이 반드시 있어야 한다** — 없으면 두 글자가
    한 글자로 접힌다. 그 규칙이 아래 `_ctc_skip` 이다.
    """
    ext = [blank]
    for lab in labels:
        ext.append(int(lab))
        ext.append(blank)
    return ext


def _ctc_skip(ext, blank):
    """`s-2` 에서 건너뛰어 올 수 있는 자리인가. 공백이 아니고 두 칸 앞과 달라야 한다."""
    return [0.0 if (u >= 2 and ext[u] != blank and ext[u] != ext[u - 2])
            else _CTC_NEG for u in range(len(ext))]


def _ctc_needs(labels):
    """이 표적을 내려면 시간이 **최소 몇 칸** 있어야 하는가.

    글자 수에, 붙어 있는 같은 글자 쌍마다 공백 한 칸을 더한다. 이보다 짧으면 어떤
    정렬도 없어서 확률이 0 이고 손실이 `inf` 다 — torch 가 그 자리에서 `inf` 를 낸다.
    문턱값으로 어림잡지 않고 이 조건을 그대로 본다.
    """
    return len(labels) + sum(1 for a, b in zip(labels, labels[1:]) if a == b)


def _ctc_targets(targets, target_lengths):
    """`(N, S)` 로 와도 되고 이어붙인 1차원으로 와도 된다. torch 가 둘 다 받는다."""
    data = _np.asarray(targets.data if isinstance(targets, Tensor) else targets)
    lengths = [int(n) for n in _np.asarray(
        target_lengths.data if isinstance(target_lengths, Tensor) else target_lengths
    ).reshape(-1)]
    if data.ndim == 1:
        out, at = [], 0
        for n in lengths:
            out.append([int(v) for v in data[at:at + n]])
            at += n
        return out
    return [[int(v) for v in data[i, :n]] for i, n in enumerate(lengths)]


def _ctc_one(lp, labels, n_time, blank):
    """표본 하나의 `-log P(표적 | 소리)`.

    **우리 연산으로 접는다** — 기울기를 손으로 안 적는다. CTC 의 역방향은 뒤로 훑기를
    한 번 더 도는 유명한 식인데, 그것을 따로 적으면 순방향과 갈릴 자리가 하나 는다.

    `u` 축은 한 번에 민다. 시간만 파이썬으로 돌므로 그래프가 `T` 에 비례한다 — 진짜
    음성 인식 길이(수백 프레임)에서는 느리다. 정확한 쪽을 골랐다.
    """
    ext = _ctc_extended(labels, blank)
    u = len(ext)
    idx = Tensor(_np.array(ext, dtype=_np.int64))
    skip = Tensor(_np.array(_ctc_skip(ext, blank), dtype=_np.float32))
    gap = lambda n: Tensor(_np.full(n, _CTC_NEG, dtype=_np.float32))   # noqa: E731

    emit = index_select(lp, 1, idx)                 # (T, U) — 자리마다의 방출 로그확률
    head = min(2, u)
    alpha = emit[0][:head]
    if u > head:
        alpha = cat([alpha, gap(u - head)], 0)

    for t in range(1, n_time):
        same = alpha
        one = cat([gap(1), alpha[:u - 1]], 0) if u > 1 else gap(1)
        two = (cat([gap(2), alpha[:u - 2]], 0) if u > 2 else gap(u)) + skip
        alpha = logsumexp(stack([same, one, two], 0), dim=0) + emit[t]

    tail = alpha[u - 2:] if u >= 2 else alpha
    return -logsumexp(tail, dim=0)


def _ctc_torch_bias(lp, i, n_time):
    """**값이 0 이고 기울기만 `exp(log_probs)` 인 항.** torch 를 따라가는 자리다.

    torch 의 `ctc_loss` 가 `log_probs` 로 흘리는 기울기는 참도함수가 아니다. 재봤다
    (`tests/probe_ctc3.py`): 유한차분은 `-γ` 인데 torch 는 `exp(log_probs) - γ` 를
    낸다. 차이가 정확히 `exp(log_probs)` 이고, `t < input_length` 안에서만 붙는다.

    **그런데 쓰는 자리에서는 둘이 같은 답이다.** CTC 앞에는 언제나 `log_softmax` 가
    있고, 그 역방향이 `g - softmax·Σg` 이기 때문이다. `g = -γ` 면 `Σg = -1` 이라
    `softmax - γ` 가 되고, `g = softmax - γ` 면 `Σg = 0` 이라 역시 `softmax - γ` 다.
    torch 가 고른 꼴은 그 변환의 고정점이다.

    그래도 여기서 맞춘다. 이 저장소의 주장은 "임포트만 바꿔 돌린다" 이고, `log_softmax`
    를 안 끼고 `log_probs` 를 바로 잎으로 두는 코드가 있으면 거기서 수가 갈린다.
    맞추되 **왜 맞추는지**를 적어 둔다 — 이 항은 손실의 도함수가 아니다.
    """
    window = lp[:n_time, i, :]
    bias = exp(window).sum()
    return bias - bias.detach()


def ctc_loss(log_probs, targets, input_lengths, target_lengths, blank=0,
             reduction="mean", zero_infinity=False):
    """`log_probs` 는 `(T, N, C)` 다 — **시간이 앞이다.** torch 가 그렇다.

    `reduction="mean"` 이 예사롭지 않다: 표본마다 **제 표적 길이로 나눈 뒤** 평균한다.
    그냥 평균이 아니라서, 표적 길이가 다 같은 케이스로 물으면 그 차이가 안 보인다.
    """
    lp = _wrap(log_probs)
    labels = _ctc_targets(targets, target_lengths)
    times = [int(n) for n in _np.asarray(
        input_lengths.data if isinstance(input_lengths, Tensor) else input_lengths
    ).reshape(-1)]

    losses = []
    for i, labs in enumerate(labels):
        if times[i] < _ctc_needs(labs):
            # 정렬이 하나도 없다 — 확률 0, 손실 `inf`. `zero_infinity` 는 그것을
            # 0 으로 바꾼다(기울기도 안 흐른다).
            losses.append(_wrap(0.0 if zero_infinity else _np.float32("inf")))
            continue
        one = _ctc_one(lp[:, i, :], labs, times[i], blank)
        losses.append(one + _ctc_torch_bias(lp, i, times[i]))

    per = stack(losses, 0)
    if reduction == "none":
        return per
    if reduction == "sum":
        return per.sum()
    lens = Tensor(_np.array([max(len(l), 1) for l in labels], dtype=_np.float32))
    return (per / lens).mean()


def _fractional_intervals(n_in, k, n_out, u):
    """창의 시작 자리들. **ATen 의 `generate_intervals` 그대로다** (재봤다).

    `α = (입력 - 창) / (출력 - 1)` 로 잡고 `floor((i+u)·α) - floor(u·α)` 를 쓴다.
    마지막 창만 오른쪽 끝에 붙인다 — 그래야 입력의 마지막 칸이 반드시 덮인다.

    `u` 가 창 자리를 흔드는 값이고, 그래서 이 풀링이 "분수" 다. 나누어떨어지면
    `α` 가 정수라 `u` 가 무엇이든 같은 답이 나온다 — **6→3 으로 물으면 무작위
    부분이 통째로 안 보인다.** 골든은 7→3 으로 묻는다.
    """
    if n_out <= 1:
        return [0]
    alpha = (n_in - k) / (n_out - 1)
    seq = [int((i + u) * alpha) - int(u * alpha) for i in range(n_out - 1)]
    return seq + [n_in - k]


def _fractional_pool(x, kernel_size, output_size, output_ratio, samples, spatial):
    """분수 최대 풀링. 창 자리가 평면마다 다를 수 있어서 평면마다 따로 접는다.

    **비싸다** — 평면 수만큼 도는데, torch 의 표본이 `(N, C, 축)` 이라 평면마다
    창이 갈릴 수 있기 때문이다. 자주 쓰는 층이 아니라 그대로 둔다. 값을 아끼려고
    표본을 하나로 뭉치면 그 순간 torch 와 다른 층이 된다.
    """
    x = _wrap(x)
    shape = x.data.shape
    if (output_size is None) == (output_ratio is None):
        raise ValueError(
            "fractional_max_pool 은 output_size 나 output_ratio 중 하나만 받습니다.")
    ks = _spread(kernel_size, spatial)
    if output_size is not None:
        sizes = _spread(output_size, spatial)
    else:
        ratios = _spread(output_ratio, spatial) if not isinstance(output_ratio, float) \
            else [output_ratio] * spatial
        sizes = [int(shape[2 + k] * ratios[k]) for k in range(spatial)]

    n, c = shape[0], shape[1]
    if samples is None:
        samples = _rng.random((n, c, spatial))
    else:
        samples = _np.asarray(samples.data if isinstance(samples, Tensor) else samples)

    values, positions = [], []
    for i in range(n):
        for j in range(c):
            plane = x[i:i + 1, j:j + 1]
            # **2차원은 표본을 뒤집어 읽는다.** ATen 의 2차원판은 `[0]` 을 너비,
            # `[1]` 을 높이로 읽고, 3차원판은 `[0]`·`[1]`·`[2]` 를 깊이·높이·너비로
            # 읽는다 — 두 함수가 서로 어긋나 있다. 여기서 흉내내는 것은 그 어긋남
            # 자체다. 축마다 다른 표본을 줘야만 드러나고, 나누어떨어지는 크기로
            # 물으면 (α 가 정수라) 표본이 무엇이든 답이 같아 또 안 보인다.
            order = list(range(spatial)) if spatial == 3 else list(reversed(range(spatial)))
            axes = [_fractional_intervals(shape[2 + k], ks[k], sizes[k],
                                          float(samples[i, j, order[k]]))
                    for k in range(spatial)]
            windows = [[(s, s + ks[k]) for s in axes[k]] for k in range(spatial)]
            got, pos = _max_with_index(plane, windows)
            values.append(got)
            positions.append(pos)
    out = cat(values, 0).reshape(n, c, *sizes)
    idx = _np.concatenate(positions, 0).reshape(n, c, *sizes)
    return out, Tensor(idx)


def fractional_max_pool2d(x, kernel_size, output_size=None, output_ratio=None,
                          return_indices=False, _random_samples=None):
    out, idx = _fractional_pool(x, kernel_size, output_size, output_ratio,
                                _random_samples, 2)
    return (out, idx) if return_indices else out


def fractional_max_pool3d(x, kernel_size, output_size=None, output_ratio=None,
                          return_indices=False, _random_samples=None):
    out, idx = _fractional_pool(x, kernel_size, output_size, output_ratio,
                                _random_samples, 3)
    return (out, idx) if return_indices else out


def fractional_max_pool2d_with_indices(x, kernel_size, output_size=None,
                                       output_ratio=None, _random_samples=None, **_):
    return _fractional_pool(x, kernel_size, output_size, output_ratio,
                            _random_samples, 2)


def fractional_max_pool3d_with_indices(x, kernel_size, output_size=None,
                                       output_ratio=None, _random_samples=None, **_):
    return _fractional_pool(x, kernel_size, output_size, output_ratio,
                            _random_samples, 3)


def max_unpool1d(x, indices, kernel_size, stride=None, padding=0, output_size=None):
    return _unpool(x, indices, kernel_size, stride, padding, output_size, 1)


def max_unpool2d(x, indices, kernel_size, stride=None, padding=0, output_size=None):
    return _unpool(x, indices, kernel_size, stride, padding, output_size, 2)


def max_unpool3d(x, indices, kernel_size, stride=None, padding=0, output_size=None):
    return _unpool(x, indices, kernel_size, stride, padding, output_size, 3)


def interpolate(x, size=None, scale_factor=2, mode="nearest", align_corners=None):
    """확대. 최근접과 겹선형 둘.

    **`align_corners` 가 값을 바꾼다.** 참이면 양 끝을 못 박고 그 사이를 고르게
    나누고(`src = i·(in−1)/(out−1)`), 거짓이면 칸의 **가운데**를 기준으로 잰다.
    `UpsamplingBilinear2d` 는 참이고 `Upsample(mode='bilinear')` 의 기본값은
    거짓이라, 이름만 보고 별명으로 두면 가장자리가 어긋난다 — 안쪽은 비슷해서
    눈으로는 안 갈린다.
    """
    x = _wrap(x)
    if mode == "bilinear":
        return _interpolate_bilinear(x, size, scale_factor, bool(align_corners))
    if mode != "nearest":
        _unsupported(f"interpolate(mode={mode!r}) — 최근접과 겹선형만 있습니다")
    if size is not None:
        n, c, h, w = x.data.shape
        oh, ow = _pair(size)
        if oh % h or ow % w:
            _unsupported("interpolate(size=) — 배수가 아닌 확대")
        scale_factor = (oh // h, ow // w)
    sh, sw = _pair(scale_factor)
    xd = x.data
    n, c, h, w = xd.shape
    out = _np.repeat(_np.repeat(xd, sh, axis=2), sw, axis=3)

    def back(g):
        gg = _np.asarray(g).reshape(n, c, h, sh, w, sw)
        return (gg.sum(axis=(3, 5)),)

    return x._make(out, (x,), back, "UpsampleBackward0")


def _bilinear_axis(size_in, size_out, align_corners):
    """출력 자리마다 **어느 두 입력 자리를 얼마씩** 섞을지."""
    if align_corners:
        # 양 끝을 못 박고 그 사이를 고르게 나눈다.
        src = (_np.arange(size_out, dtype=_np.float64)
               * ((size_in - 1) / max(1, size_out - 1)))
    else:
        # 칸의 가운데를 기준으로 잰다. 밖으로 나가는 자리는 가장자리에 붙인다.
        scale = size_in / size_out
        src = (_np.arange(size_out, dtype=_np.float64) + 0.5) * scale - 0.5
        src = _np.clip(src, 0, None)
    lo = _np.floor(src).astype(_np.intp)
    hi = _np.minimum(lo + 1, size_in - 1)
    return lo, hi, (src - lo)


def _interpolate_bilinear(x, size, scale_factor, align_corners):
    n, c, h, w = x.data.shape
    if size is not None:
        oh, ow = _pair(size)
    else:
        sh, sw = _pair(scale_factor)
        oh, ow = int(h * sh), int(w * sw)
    y0, y1, wy = _bilinear_axis(h, oh, align_corners)
    x0, x1, wx = _bilinear_axis(w, ow, align_corners)
    wy = wy.astype(x.data.dtype)[:, None]
    wx = wx.astype(x.data.dtype)[None, :]

    def blend(t):
        top = t[:, :, y0][:, :, :, x0] * (1 - wx) + t[:, :, y0][:, :, :, x1] * wx
        bot = t[:, :, y1][:, :, :, x0] * (1 - wx) + t[:, :, y1][:, :, :, x1] * wx
        return top * (1 - wy) + bot * wy

    def back(g):
        gg = _np.asarray(g)
        out = _np.zeros_like(x.data)
        for ys, wgt_y in ((y0, 1 - wy), (y1, wy)):
            for xs, wgt_x in ((x0, 1 - wx), (x1, wx)):
                share = gg * wgt_y * wgt_x
                # 같은 자리를 여러 번 읽으므로 **모아 더한다.**
                for i, yi in enumerate(ys):
                    _np.add.at(out[:, :, yi], (slice(None), slice(None), xs),
                               share[:, :, i])
        return (out,)

    return x._make(blend(x.data), (x,), back, "UpsampleBilinear2DBackward0")


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


def _fixed_windows(n_in, size, step):
    """고정 창의 목록. **한 자리에만 적는다** — 값과 자리가 다른 창을 보면 갈린다."""
    return [(s, s + size) for s in range(0, n_in - size + 1, step)]


def _fixed_window_axes(shape, kernel_size, stride):
    """축마다의 창 목록. `_fixed` 와 자리 계산이 같은 것을 쓴다."""
    spatial = len(shape) - 2
    kernels = _spread(kernel_size, spatial)
    strides = _spread(stride if stride is not None else kernel_size, spatial)
    return [_fixed_windows(shape[2 + k], kernels[k], strides[k])
            for k in range(spatial)]


def _fixed(x, kernel_size, stride, kind):
    """고정 창. 적응형과 같은 기계에 창 목록만 다르게 준다."""
    x = _wrap(x)
    spatial = len(x.data.shape) - 2
    kernels = _spread(kernel_size, spatial)
    strides = _spread(stride if stride is not None else kernel_size, spatial)
    out = x
    for k in range(spatial):
        axis = 2 + k
        windows = _fixed_windows(out.data.shape[axis], kernels[k], strides[k])
        out = _fold_axis(out, axis, windows, kind)
    return out


def _max_with_index(x, window_axes):
    """최댓값과 **이긴 자리**를 함께 낸다.

    자리는 torch 의 규약대로 **평면 안의 평평한 번호**다 — 2차원이면 `h*W + w`,
    3차원이면 `(d*H + h)*W + w`. 배치와 채널마다 0 부터 다시 센다(재봤다:
    `tests/probe_pool.py`). `MaxUnpool` 이 이 번호를 그대로 되돌린다.

    **축을 뒤에서부터 접는다.** 같은 값이 둘이면 torch 는 평평한 번호가 작은 쪽,
    즉 행 우선으로 먼저 나오는 자리를 고른다. 앞 축부터 접으면 "열마다 먼저인 행"
    을 고른 뒤 "행 중 먼저인 열" 을 골라서 **열 우선 첫째**가 되고, 값이 같으므로
    아무 검사에도 안 걸린 채 자리만 달라진다. 뒤에서부터 접으면 행 안의 첫 열을
    먼저 정하고 그다음 첫 행을 정해서 행 우선 첫째가 된다.

    값도 여기서 같이 낸다. 값을 다른 경로로 구하면 "자리는 A 인데 값은 B" 가 될 수
    있고, 그것은 둘 다 그럴듯해서 안 보인다.
    """
    x = _wrap(x)
    data = x.data
    shape = data.shape
    spatial = shape[2:]
    plane = int(_np.prod(spatial)) if spatial else 1

    val = data
    pos = _np.broadcast_to(_np.arange(plane).reshape(spatial), shape)
    for k in reversed(range(len(window_axes))):
        axis = 2 + k
        vparts, pparts = [], []
        for start, end in window_axes[k]:
            cut = (slice(None),) * axis + (slice(start, end),)
            vs, ps = val[cut], pos[cut]
            j = vs.argmax(axis=axis)[(slice(None),) * axis + (None,)]
            vparts.append(_np.take_along_axis(vs, j, axis))
            pparts.append(_np.take_along_axis(ps, j, axis))
        val = _np.concatenate(vparts, axis)
        pos = _np.concatenate(pparts, axis)

    # 기울기는 이긴 자리로만 간다. 자리표를 전체 평평한 번호로 올려 한 번에 흩뿌린다.
    base = (_np.arange(shape[0] * shape[1]) * plane).reshape(shape[0], shape[1],
                                                             *([1] * len(spatial)))
    flat = (base + pos).reshape(-1)

    def back(g):
        gx = _np.zeros(data.size, dtype=_np.asarray(g).dtype)
        _np.add.at(gx, flat, _np.asarray(g).reshape(-1))
        return (gx.reshape(shape),)

    return x._make(val, (x,), back, "MaxPoolWithIndicesBackward0"), pos


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


def adaptive_max_pool1d(x, output_size, return_indices=False):
    if return_indices:
        return adaptive_max_pool1d_with_indices(x, output_size)
    return _adaptive(x, _spread(output_size, 1), "max")


def adaptive_max_pool2d(x, output_size, return_indices=False):
    if return_indices:
        return adaptive_max_pool2d_with_indices(x, output_size)
    return _adaptive(x, _pair(output_size), "max")


def adaptive_max_pool3d(x, output_size, return_indices=False):
    if return_indices:
        return adaptive_max_pool3d_with_indices(x, output_size)
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


def lp_pool3d(x, norm_type, kernel_size, stride=None):
    """1·2 차원과 같은 조립이다. 창 칸 수만 세 축의 곱이 된다."""
    x = _wrap(x)
    kd, kh, kw = _spread(kernel_size, 3)
    out = avg_pool3d(x ** norm_type, kernel_size, stride)
    return ((out.sign() * relu(out.abs())) * (kd * kh * kw)) ** (1.0 / norm_type)


def max_pool2d(x, kernel_size, stride=None, return_indices=False):
    if return_indices:
        return max_pool2d_with_indices(x, kernel_size, stride)
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
    dim = _pos_dim(t, dim)
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
    """축 `dim` 만 자르는 색인 묶음. **`dim` 은 양수여야 한다** — 음수를 주면
    `range(dim)` 이 비어서 **축 0 을 자른다.** 예외 없이."""
    return tuple(slice(None) for _ in range(dim)) + (slice(start, end),)


def _pos_dim(t, dim):
    """음수 축을 양수로.

    **`_slice_at` 이 음수를 못 받는다.** `narrow(x, -1, …)` 이 랭크 2 이상에서
    **축 0 을 잘랐고**, 랭크 1 에서는 축 −1 과 축 0 이 같아서 오래 안 보였다.
    `stft` 를 조립하다가 배치 신호(1, 16)에서 처음 드러났다 — 모양이 (0, 24) 가
    되어 `stack` 이 멈췄다. 값이 틀리는 대신 모양이 무너져서 시끄러웠던 것이 운이다.
    """
    return dim + t.data.ndim if dim < 0 else dim


def unbind(t, dim=0):
    t = _wrap(t)
    dim = _pos_dim(t, dim)
    return tuple(t[_slice_at(dim, i, i + 1)].squeeze(dim) for i in range(t.data.shape[dim]))


def narrow(t, dim, start, length):
    t = _wrap(t)
    return t[_slice_at(_pos_dim(t, dim), start, start + length)]


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

# **밑줄 없는 짝을 이미 가진 것들.** 계산은 그쪽이 하고 여기서는 되쓰기만 한다.
#
# 손으로 마흔일곱 벌을 적으면 마흔일곱 자리가 어긋날 수 있는데, 실제로 다른 것은
# 인자 개수뿐이다. `torch.Tensor` 에 그 이름이 정말 있는지는 `tests/test_tensor_api.py`
# 가 진짜 torch 에 물어 확인한다 — 없는 이름을 만들면 우리에게만 도는 코드가 된다.
_INPLACE_MORE = (
    "absolute", "acosh", "arccos", "arccosh", "arcsin", "arcsinh", "arctan",
    "arctanh", "asinh", "atanh", "deg2rad", "erfc", "exp2", "fix", "logit",
    "negative", "rad2deg", "sgn", "sinc",
)
# 인자를 하나 더 받는 것들. 자리 수만 다르고 나머지는 같다.
_INPLACE_BINARY = (
    "atan2", "copysign", "eq", "ge", "gt", "heaviside", "hypot", "le", "lt",
    "ne", "xlogy",
)
# 축이나 번호를 받는 것들.
_INPLACE_ARGS = (
    "cumprod", "cumsum", "index_add", "index_copy", "index_fill", "ldexp",
    "masked_fill", "scatter", "scatter_add", "squeeze", "swapaxes", "swapdims",
    "transpose", "tril", "triu", "unsqueeze",
)


def _make_inplace(name, arity="nullary"):
    # **모듈 함수일 수도, 메서드로만 있을 수도 있다.** `cumsum`·`squeeze` 처럼 텐서
    # 쪽에만 있는 것이 있어서, 모듈에서 못 찾으면 메서드를 부른다. 어느 쪽이든
    # 계산은 그것이 하고 여기서는 되쓰기만 한다 — 두 벌로 적으면 언젠가 갈린다.
    fn = globals().get(name)
    if fn is None:
        def fn(t, *a, **k):
            return getattr(t, name)(*a, **k)

    if arity == "nullary":
        def method(self):
            return self._inplace(lambda: fn(self), name + "_")
    else:
        def method(self, *args, **kw):
            return self._inplace(lambda: fn(self, *args, **kw), name + "_")

    method.__name__ = name + "_"
    method.__doc__ = f"`{name}` 을 제자리에서. 산수는 `{name}` 이 하고 여기서는 되쓰기만 한다."
    return method


# **거는 자리는 이 파일 끝이다.** 여기서 걸면 아래에 정의되는 함수들을 아직 못 본다 —
# `add` 하나로 `KeyError` 가 났다.


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
    """`rounding_mode` 는 셋이다 — 없으면 참나눗셈, `'floor'`·`'trunc'` 는 정수 쪽.

    **형이 갈리는 자리다.** 참나눗셈은 언제나 실수인데, 자르거나 내림하면 **입력의
    형으로 돌아온다**(실측: `int64 / int64` 에 `trunc` 를 주면 int64 다). 값만 맞추고
    형을 실수로 두면 그 뒤 색인이나 `bincount` 가 정수를 요구하는 자리에서 갈린다 —
    값 대조로는 안 보인다.
    """
    left, right = _wrap(a), _wrap(b)
    out = left / right
    if rounding_mode is None:
        return out
    if rounding_mode == "floor":
        out = out.floor()
    elif rounding_mode == "trunc":
        out = out.trunc()
    else:
        raise RuntimeError(
            f"rounding_mode 는 None·'floor'·'trunc' 뿐입니다: {rounding_mode!r}")
    kind = result_type(left.data.dtype, _np.asarray(
        right.data if isinstance(right, Tensor) else right).dtype)
    return out if _np.dtype(kind).kind == "f" else out.type(kind)


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


def _polygamma_np(n, d):
    """`ψ^(n)` — 감마 로그의 `n+1` 번째 미분. `n=0` 이 `digamma` 다.

    **`digamma` 와 같은 방식이다** — 되풀이 식으로 큰 쪽으로 민 뒤 점근식을 쓴다.
    그 둘(`n=0`·`n=1`)이 이미 손으로 적혀 있었고, 여기는 그 규칙을 `n` 으로 일반화한
    것뿐이다. 식을 세 벌로 두면 한쪽만 고쳐지는 날이 오므로 `n` 이 작을 때도 이쪽을
    지나게 두는 편이 낫지만, 있던 둘은 이미 골든이 물고 있어 그대로 둔다.

    되풀이: `ψ^(n)(x) = ψ^(n)(x+1) + (−1)^(n+1) n! / x^(n+1)`
    점근  : `ψ^(n)(x) ≈ (−1)^(n+1) [ (n−1)!/xⁿ + n!/(2x^(n+1)) + Σ B_2k … ]`
    """
    x = _np.asarray(d, dtype=_np.float64)
    fact = float(_math.factorial(n))
    sign = 1.0 if (n + 1) % 2 == 0 else -1.0
    out = _np.zeros_like(x)
    while _np.any(x < 20):
        small = x < 20
        safe = _np.where(small, x, 1.0)
        out = _np.where(small, out + sign * fact / safe ** (n + 1), out)
        x = _np.where(small, x + 1.0, x)
    # 베르누이 수 B₂·B₄·B₆·B₈. x ≥ 20 에서 네 항이면 float32 정밀도를 넘는다.
    series = _math.factorial(n - 1) / x ** n + fact / (2.0 * x ** (n + 1))
    for k, bern in enumerate((1 / 6, -1 / 30, 1 / 42, -1 / 30), start=1):
        series = series + (bern * _math.factorial(2 * k + n - 1)
                           / _math.factorial(2 * k) / x ** (2 * k + n))
    return out + sign * series


def _igamma_np(a, x):
    """정규화된 하부 불완전 감마 `P(a, x) = γ(a,x)/Γ(a)`.

    **한 식으로 못 덮는다.** `x < a+1` 에서는 급수가 빨리 수렴하고 그 밖에서는 연분수가
    빠르다 — 반대로 쓰면 항이 서로 지워 자릿수를 잃는다. 경계에서 갈라 쓰는 것이
    이 함수의 표준 꼴이고, 작은 값으로만 재면 그 갈림이 안 보인다.
    """
    av = _np.asarray(a, dtype=_np.float64)
    xv = _np.asarray(x, dtype=_np.float64)
    av, xv = _np.broadcast_arrays(av, xv)
    out = _np.zeros(av.shape, dtype=_np.float64)
    lg = _np.vectorize(_math.lgamma)(av)

    # ── 급수 (x < a+1): P = e^(−x + a·ln x − lnΓ(a)) · Σ xⁿ / (a(a+1)…(a+n))
    low = xv < av + 1.0
    if _np.any(low):
        ap = _np.where(low, av, 1.0)
        term = 1.0 / ap
        total = term.copy()
        for _ in _builtin_range(300):
            ap = ap + 1.0
            term = term * _np.where(low, xv, 0.0) / ap
            total = total + term
            if _np.all(_np.abs(term) <= _np.abs(total) * 1e-16):
                break
        out = _np.where(low, total * _np.exp(-xv + av * _np.log(
            _np.where(xv > 0, xv, 1.0)) - lg), out)

    # ── 연분수 (x ≥ a+1): Q 를 구하고 P = 1 − Q
    high = ~low
    if _np.any(high):
        tiny = 1e-300
        b = _np.where(high, xv + 1.0 - av, 1.0)
        c = 1.0 / tiny
        dd = 1.0 / b
        h = dd.copy()
        for i in _builtin_range(1, 300):
            an = -i * (i - av)
            b = b + 2.0
            dd = an * dd + b
            dd = _np.where(_np.abs(dd) < tiny, tiny, dd)
            c = b + an / c
            c = _np.where(_np.abs(c) < tiny, tiny, c)
            dd = 1.0 / dd
            delta = dd * c
            h = h * delta
            if _np.all(_np.abs(delta - 1.0) <= 1e-16):
                break
        q = _np.exp(-xv + av * _np.log(_np.where(xv > 0, xv, 1.0)) - lg) * h
        out = _np.where(high, 1.0 - q, out)
    # `x = 0` 이면 P = 0 이다. 위 로그가 그 자리를 못 지난다.
    return _np.where(xv <= 0, 0.0, out)


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


# ── 최상위에 남아 있던 이름들 ─────────────────────────────────────────────
#
# `tests/torch_gap.py` 의 "검토 대상" 을 재서 가른 나머지다. **이름으로 세면
# 틀린다** — `fake_quantize_*` 는 이름이 양자화인데 실수를 받아 실수를 내고,
# `dequantize` 는 실수에서 항등이다. 재고 나서야 거절이 아닌 줄 알았다.

def igamma(input, other):                                       # noqa: A002
    """정규화된 하부 불완전 감마 `P(a, x)`.

    **기울기가 `x` 쪽에만 있다**(실측). `a` 로 미분하려 들면 torch 가
    `NotImplementedError` 를 낸다 — 닫힌 꼴이 없어서다. 따라간다: 한쪽만 흘리면
    나머지 한쪽이 조용히 0 이 되고, 그러면 학습이 안 되는 것으로만 드러난다.

        dP/dx = x^(a−1)·e^(−x) / Γ(a)
    """
    a, x = _wrap(input), _wrap(other)
    av = _np.asarray(a.data, dtype=_np.float64)
    xv = _np.asarray(x.data, dtype=_np.float64)
    out = _igamma_np(av, xv)
    lg = _np.vectorize(_math.lgamma)(av)
    slope = _np.exp((av - 1.0) * _np.log(_np.where(xv > 0, xv, 1.0)) - xv - lg)
    slope = _np.where(xv > 0, slope, 0.0)

    def back(g):
        if a.requires_grad:
            raise NotImplementedError(_like_torch(
                "igamma 는 첫 인자로 미분되지 않습니다 — 닫힌 꼴이 없습니다.",
                "the derivative for 'igamma: input' is not implemented."))
        return (None, _unbroadcast(_np.asarray(g) * slope, x.data.shape)
                .astype(x.data.dtype))

    return a._make(out.astype(_DEFAULT_DTYPE), (a, x), back, "IgammaBackward0")


def igammac(input, other):                                      # noqa: A002
    """상부 쪽 `Q(a, x) = 1 − P(a, x)`. **둘을 더하면 정확히 1 이다**(실측)."""
    a, x = _wrap(input), _wrap(other)
    return ones(a.data.shape if a.data.ndim >= x.data.ndim else x.data.shape) \
        - igamma(a, x)


def polygamma(n, input):                                        # noqa: A002
    """`ψ^(n)` — 감마 로그의 `n+1` 번째 미분. `n=0` 이 `digamma` 다.

    **`n` 이 첫 자리다** — 텐서가 둘째다. `torch.polygamma(1, x)` 가 그 꼴이고,
    자리를 뒤집으면 정수를 텐서 자리에 넣게 되어 시끄럽게 멈춘다.

    미분은 `ψ^(n+1)` 이다(실측: `polygamma(1, x)` 의 기울기가 `polygamma(2, x)`).
    """
    t = _wrap(input)
    k = int(n)
    if k < 0:
        raise RuntimeError(_like_torch(
            "polygamma 의 n 은 0 이상이어야 합니다.",
            "polygamma(n, x) does not support negative n."))
    out = (_polygamma0(t.data) if k == 0 else _polygamma_np(k, t.data))
    nxt = _polygamma_np(k + 1, t.data)
    return t._make(out.astype(t.data.dtype), (t,),
                   lambda g: (_np.asarray(g) * nxt.astype(t.data.dtype),),
                   "PolygammaBackward0")


def constant_pad_nd(input, pad, value=0.0):                     # noqa: A002
    """`F.pad(mode='constant')` 의 날 이름. **마지막 축부터 (앞, 뒤) 순이다.**

    같은 계산에 이름이 둘인 자리다 — 한쪽만 맞을 수 있으므로 **여기서 계산하지 않고**
    그쪽으로 넘긴다. 이 저장소가 그 모양으로 세 번 물렸다(README 의 목록 1번).
    """
    return globals()["pad"](_wrap(input), list(pad), mode="constant",
                            value=value)


def _quantize_round(x, scale, zero_point, quant_min, quant_max):
    """`clamp(round(x/scale) + zp, [qmin, qmax])` 뒤 되돌린 값.

    **양자화 dtype 이 필요 없다** — 실수를 받아 실수를 낸다. 이름 때문에 오래 거절
    쪽으로 세어 두었는데, 재보니 torch 도 실수 텐서를 받는다.
    """
    q = _np.clip(_np.round(x / scale) + zero_point, quant_min, quant_max)
    return (q - zero_point) * scale


def fake_quantize_per_tensor_affine(input, scale, zero_point,   # noqa: A002
                                    quant_min, quant_max):
    """양자화를 **실수 위에서 흉내 낸다.** 학습 중 양자화 오차를 보려고 쓰는 자리다.

    **기울기는 범위 안에서만 1 이다**(실측: 잘린 자리는 0). 반올림은 계단이라
    도함수가 거의 어디서나 0 인데, torch 는 그 자리를 "곧바로 통과(STE)" 로 둔다 —
    안 그러면 이 층 아래로 학습이 아예 안 간다.
    """
    t = _wrap(input)
    s, z = float(scale), float(zero_point)
    x = _np.asarray(t.data, dtype=_np.float64)
    out = _quantize_round(x, s, z, quant_min, quant_max)
    inside = ((_np.round(x / s) + z >= quant_min)
              & (_np.round(x / s) + z <= quant_max)).astype(t.data.dtype)
    return t._make(out.astype(t.data.dtype), (t,),
                   lambda g: (_np.asarray(g) * inside,),
                   "FakeQuantizePerTensorAffineBackward0")


def fake_quantize_per_channel_affine(input, scale, zero_point,  # noqa: A002
                                     axis, quant_min, quant_max):
    """칸마다 다른 눈금으로. **축 하나를 따라 눈금이 바뀐다.**"""
    t = _wrap(input)
    s = _np.asarray(_wrap(scale).data, dtype=_np.float64)
    z = _np.asarray(_wrap(zero_point).data, dtype=_np.float64)
    shape = [1] * t.data.ndim
    shape[int(axis)] = -1
    s, z = s.reshape(shape), z.reshape(shape)
    x = _np.asarray(t.data, dtype=_np.float64)
    out = _quantize_round(x, s, z, quant_min, quant_max)
    inside = ((_np.round(x / s) + z >= quant_min)
              & (_np.round(x / s) + z <= quant_max)).astype(t.data.dtype)
    return t._make(out.astype(t.data.dtype), (t,),
                   lambda g: (_np.asarray(g) * inside,),
                   "FakeQuantizePerChannelAffineBackward0")


def dequantize(input):                                          # noqa: A002
    """양자화된 텐서를 실수로. **우리에게는 언제나 항등이다.**

    항등인 것이 "지금 통과하는 항등" 과 다른 이유는, 양자화 dtype 이 **영원히
    없다**는 것이 이미 정해져 있어서다 — 이 함수가 받을 수 있는 입력이 실수뿐이고
    실수에서 torch 도 항등이다(실측). 나중에 그 dtype 이 생기면 이 줄이 틀리는데,
    생기지 않는 것이 결정이다.

    **미분되지 않는다** — torch 가 `backward` 에서 멈춘다(실측).
    """
    return Tensor(_np.asarray(_wrap(input).data).copy())


def resize_as_(input, other):                                   # noqa: A002
    """`other` 의 모양으로 **제자리에서** 바꾼다.

    **늘어난 칸의 값은 정해지지 않는다** — torch 도 초기화하지 않는다(실측). 그래서
    골든은 **모양만** 묻는다. 값을 굳히면 그 구현의 우연을 명세로 박제하게 된다.
    """
    t, o = _wrap(input), _wrap(other)
    flat = _np.asarray(t.data).reshape(-1)
    want = int(_np.prod(o.data.shape)) if o.data.shape else 1
    grown = _np.zeros(want, dtype=t.data.dtype)
    keep = min(flat.size, want)
    grown[:keep] = flat[:keep]
    t.data = Tensor(grown.reshape(o.data.shape))
    return t


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


def searchsorted(sorted_sequence, values, side=None, right=False, **kw):
    """정렬된 것 안에서 들어갈 자리. **동점의 어느 쪽인지를 두 인자가 함께 정한다.**

    torch 는 같은 것을 두 이름으로 받는다 — 참거짓 `right` 와 문자열 `side` 다.
    여기에는 `right` 만 있었고 `side` 는 `**kw` 로 들어가 **조용히 버려졌다.**
    `searchsorted(seq, v, side="right")` 가 왼쪽 답을 냈고, 값이 하나씩만 어긋나서
    그럴듯해 보인다. `bucketize(right=True)` 는 처음부터 맞았다 — **같은 계산에
    이름이 둘인데 한쪽만 맞은** 자리가 이 저장소에서 세 번째다.

    둘이 어긋나면 torch 는 멈춘다(실측). 하나만 주거나, 같은 뜻으로 둘 다 줘야 한다.
    """
    if side is not None:
        if side not in ("left", "right"):
            raise RuntimeError(_like_torch(
                f"side 는 'left' 나 'right' 여야 합니다 ({side!r} 을 받았습니다).",
                f"torch.searchsorted(): side can only be 'left' or 'right' but "
                f"got {side}"))
        if right and side == "left":
            raise RuntimeError(_like_torch(
                "side 와 right 가 서로 반대입니다 — 둘 중 하나만 주세요.",
                "torch.searchsorted(): side and right can't be set to opposites, "
                "got side of left while right was True"))
        right = side == "right"
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

def prod(t, dim=None, keepdim=False, dtype=None):
    t = _wrap(t)
    if dtype is not None:
        return prod(t.to(dtype), dim, keepdim).to(dtype)
    out = _np.prod(t.data, axis=dim, keepdims=bool(keepdim) and dim is not None)
    # 역방향은 접기 전 모양으로 편다 — `keepdim` 이면 축이 이미 살아 있어 그대로다.
    wide = out if keepdim or dim is None else _np.expand_dims(out, dim)
    return t._make(out, (t,),
                   lambda g: (_np.asarray(g if keepdim or dim is None
                                          else _np.expand_dims(g, dim))
                              * wide / t.data,),
                   "ProdBackward0")


def median(t, dim=None, keepdim=False):
    """torch 는 원소가 짝수일 때 **가운데 둘 중 작은 쪽**을 준다. numpy 는 평균을 낸다 —
    그대로 쓰면 조용히 다른 값이 나온다.

    **NaN 이 하나라도 있으면 NaN 이다**(실측). `argsort` 는 NaN 을 맨 뒤로 밀어내므로
    그냥 정렬해 고르면 **NaN 을 건너뛰고** 멀쩡한 값이 나온다 — 그것이 `nanmedian` 이고
    이쪽은 아니다. 둘을 나란히 묻는 케이스를 넣으면서 걸렸다.
    """
    t = _wrap(t)
    # **참·거짓은 거절한다.** torch 가 `"median_cpu" not implemented for 'Bool'` 로
    # 멈춘다(실측). 규칙이 아니라 torch 의 구멍이지만, 여기서 값을 내주면 그 코드가
    # 진짜 torch 에서 깨진다 — 관대한 것도 갈리는 것이다.
    _refuses_bool(t.data, "median 은 참거짓을 받지 않습니다.",
                  '"median_cpu" not implemented for \'Bool\'',
                  kind=NotImplementedError)
    if dim is None:
        flat = t.data.reshape(-1)
        if _np.isnan(flat).any():
            return Tensor(_np.asarray(_np.nan, dtype=t.data.dtype))
        pick = int(_np.argsort(flat)[(flat.size - 1) // 2])

        # 기울기는 **값이 같은 칸 전부에 고르게** 간다 — `max()` 와 같은 규칙이다
        # (실측: [1,5,5,5] 의 median 기울기가 [0, ⅓, ⅓, ⅓]).
        #
        # 여기에는 "뽑힌 자리 하나로만 간다" 가 있었고, 근거도 적혀 있었다 — 나머지
        # 원소를 흔들어도 답이 안 움직인다는 것. **동점이 아닐 때만 맞는 말이다.**
        # 동점이면 그 원소들도 답을 같이 떠받치고 있어서, 하나만 흔들어도 답이
        # 따라 움직인다. 동점 없는 자료로만 재면 두 규칙이 같은 답을 내므로 안 갈린다.
        share = (flat == flat[pick]).astype(_np.float64)
        share = (share / share.sum()).reshape(t.data.shape)

        def back(g):
            return (_np.asarray(g) * share,)

        return t._make(flat[pick], (t,), back, "MedianBackward0")

    order = _np.argsort(t.data, axis=dim)
    idx = (t.data.shape[dim] - 1) // 2
    take = _np.take(order, idx, axis=dim)
    at = _np.expand_dims(take, dim)
    picked = _np.take_along_axis(t.data, at, axis=dim).squeeze(dim)
    # NaN 이 든 줄은 통째로 NaN 이다 — 위 docstring 의 그 자리다.
    sick = _np.isnan(t.data).any(axis=dim)
    if sick.any():
        picked = _np.where(sick, _np.asarray(_np.nan, dtype=picked.dtype), picked)

    def back_dim(g):
        z = _np.zeros_like(t.data)
        # **`keepdim` 이면 축이 이미 살아 있다.** 여기서 한 번 더 펴면 랭크가 하나
        # 늘어 `put_along_axis` 가 멈춘다 — 값이 아니라 모양에서 걸리는 자리다.
        wide = _np.asarray(g) if keepdim else _np.expand_dims(_np.asarray(g), dim)
        _np.put_along_axis(z, at, wide, axis=dim)
        return (z,)

    if keepdim:
        picked = _np.expand_dims(picked, dim)
        take = _np.expand_dims(take, dim)

    return _MinMax(t._make(picked, (t,), back_dim, "MedianBackward0"), Tensor(take))


def norm(t, p=2, dim=None):
    t = _wrap(t)
    _needs_float(
        t.data,
        "노름은 실수에만 있습니다 — 제곱근이 정수 칸에 안 들어갑니다. "
        "`.float()` 을 먼저 부르세요.",
        "linalg.vector_norm: Expected a floating point or complex tensor as input")
    if p == 1:
        return t.abs().sum(dim=dim)
    if p == 2:
        return (t * t).sum(dim=dim) ** 0.5
    # **`p` 가 1·2 말고도 온다.** 오래 나머지를 전부 2 로 셌다 — `dist(a, b, 3)` 이
    # L2 를 돌려줬고 값이 그럴듯해서(같은 크기 대) 안 보였다. `inf` 는 최대 절댓값이고
    # `-inf` 는 최소, `0` 은 0 이 아닌 것의 개수다(실측).
    if p == float("inf"):
        return t.abs().max(dim=dim) if dim is None else t.abs().amax(dim=dim)
    if p == -float("inf"):
        return t.abs().min(dim=dim) if dim is None else t.abs().amin(dim=dim)
    if p == 0:
        # **`t * 0` 을 더해 그래프를 잇는다.** 세는 것은 계단이라 도함수가 0 이고,
        # 0 은 "없다" 가 아니라 맞는 답이다. 안 이으면 `norm(0).backward()` 가 멈추는데
        # torch 는 안 멈춘다 — `grad_fn` 은 두고 잎에는 안 닿아서 `grad` 가 None 으로
        # 남는다(실측). 우리는 0 이 쌓이고 torch 는 None 이 남는 차이만 있고, 손실에
        # 더했을 때 학습에 미치는 영향은 같다. **멈추는 쪽이 제일 멀다.**
        return (t != 0).float().sum(dim=dim) + (t * 0).sum(dim=dim)
    return (t.abs() ** float(p)).sum(dim=dim) ** (1.0 / float(p))


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


def nansum(t, dim=None, keepdim=False, dtype=None):
    """nan 을 **0 으로 세는** 합. 기울기도 그 자리로는 안 간다."""
    t = _wrap(t)
    if dtype is not None:
        return nansum(t.to(dtype), dim, keepdim).to(dtype)
    # **형을 지킨다.** `sum` 과 같은 규칙인데 `_nan_mask` 가 nan 을 다루려고 실수로
    # 올려 버려서 정수·참거짓이 float64 로 나왔다. 정수엔 nan 이 없으므로 그대로 센다.
    if t.data.dtype.kind not in "fc":
        return t.sum(dim=dim, keepdim=keepdim)
    clean, bad = _nan_mask(t)
    return t._make(clean.sum(axis=dim, keepdims=keepdim), (t,),
                   lambda g: (_np.where(bad, 0.0, _expand_reduced(g, t.data.shape, dim, keepdim)),),
                   "NansumBackward0")


def nanmean(t, dim=None, keepdim=False, dtype=None):
    """nan 을 **빼고** 낸 평균 — 세는 개수도 nan 이 아닌 것만이다.

    **`dtype=` 이 정수 거절을 안 풀어 준다.** `mean` 은 풀어 주는데 이쪽은 안 풀린다
    (실측: `torch.tensor([3,1,4]).nanmean(dtype=torch.float32)` 이 멈춘다). 규칙이
    아니라 torch 의 비대칭이고, 관대한 쪽으로 갈리는 것도 갈리는 것이라 따라간다.
    """
    t = _wrap(t)
    _needs_float(
        t.data,
        "nanmean 은 실수에만 있습니다. `.float()` 을 먼저 부르세요.",
        "nanmean(): expected input to have floating point or complex dtype")
    clean, bad = _nan_mask(t)
    count = (~bad).sum(axis=dim, keepdims=keepdim)
    total = clean.sum(axis=dim, keepdims=keepdim)
    out = total / count

    def back(g):
        gg = _expand_reduced(g, t.data.shape, dim, keepdim)
        n = _expand_reduced(count, t.data.shape, dim, keepdim) if dim is not None else count
        return (_np.where(bad, 0.0, gg / n),)

    got = t._make(out, (t,), back, "NanmeanBackward0")
    return got if dtype is None else got.to(dtype)


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
    # **정수·참거짓도 받고 float32 를 낸다**(실측). 그냥 두면 두 자리가 틀린다 —
    # numpy 가 정수를 float64 로 올리고, 참거짓은 `-` 를 거절해 아래 뺄셈에서 멈춘다.
    if t.data.dtype.kind not in "fc":
        t = _wrap(t.data.astype(_DEFAULT_DTYPE))
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


def diff(t, n=1, dim=-1, prepend=None, append=None):
    """이웃한 것의 차. `x[1:] - x[:-1]` 을 n 번 한다.

    **자르기로 짠다** — 자르기가 이미 그래프를 이으므로 역방향을 새로 쓸 것이 없다.

    **참·거짓은 뺄셈이 아니라 XOR 이다.** torch 가 `[T, F, T]` 에서 `[T, T]` 를
    주는데(실측) 그것은 이웃이 다른가를 묻는 것이다. 여기서는 `-` 가 불리언을 거절해
    아예 멈추고 있었다 — 인색한 것도 갈리는 것이다.

    **앞뒤로 붙이면 길이가 안 줄어든다.** `prepend`·`append` 는 차를 구하기 **전에**
    이어 붙이는 것이라, 하나를 붙이면 결과가 입력과 같은 길이가 된다 — 시계열에서
    첫 칸을 잃지 않으려고 쓰는 자리다.
    """
    out = _wrap(t)
    if prepend is not None or append is not None:
        parts = ([_wrap(prepend)] if prepend is not None else []) + [out] \
            + ([_wrap(append)] if append is not None else [])
        out = cat(parts, dim=dim)
    if out.data.dtype.kind == "b":
        data = out.data
        for _ in range(n):
            data = _np.logical_xor(data.take(_np.arange(1, data.shape[dim]), axis=dim),
                                   data.take(_np.arange(0, data.shape[dim] - 1), axis=dim))
        return Tensor(data)
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
    _needs_float(
        t.data,
        "분위수는 실수에만 있습니다 — 보간이 정수 칸에 안 들어갑니다.",
        "quantile() input tensor must be either float or double dtype")
    qq = q.data if isinstance(q, Tensor) else _np.asarray(q, dtype=t.data.dtype)
    out = _np.quantile(t.data, qq, axis=dim, keepdims=keepdim)

    # **기울기는 보간에 쓰인 두 자리로 나뉜다** — 정확히 맞아떨어지면 한 자리다
    # (실측: [3,5,5,1,5] 의 quantile(0.3) 기울기가 [0.8, 0.2, 0, 0, 0]).
    #
    # `median` 과 규칙이 다르다. `median` 은 **값이 같은 칸 전부**에 나누는데
    # `quantile` 은 **정렬한 자리**로 나눈다 — [1,5,5,5] 에서 median 은 세 5 에
    # ⅓ 씩 주고 quantile(0.5) 는 앞의 두 5 에 ½ 씩 준다. 동점이 없는 자료로 재면
    # 둘이 같은 답을 내므로 이 갈림이 안 보인다.
    #
    # 여기에도 `Tensor(...)` 만 있어서 그래프가 조용히 끊겨 있었다.
    data = _np.asarray(t.data, dtype=_np.float64)
    lines = data.reshape(1, -1) if dim is None else \
        _np.moveaxis(data, dim, -1).reshape(-1, data.shape[dim])
    order = _np.argsort(lines, axis=-1, kind="stable")
    n = lines.shape[-1]
    rows = _np.arange(lines.shape[0])
    # q 하나마다 무게판을 한 장 만든다. 스칼라 q 면 한 장이다.
    qs = _np.atleast_1d(_np.asarray(qq, dtype=_np.float64))
    sheets = _np.zeros((qs.size,) + lines.shape, dtype=_np.float64)
    for k, one in enumerate(qs):
        pos = float(one) * (n - 1)
        lo, hi = int(_np.floor(pos)), int(_np.ceil(pos))
        frac = pos - lo
        _np.add.at(sheets[k], (rows, order[:, lo]), 1.0 - frac)
        _np.add.at(sheets[k], (rows, order[:, hi]), frac)

    def back(g):
        gg = _np.asarray(g, dtype=_np.float64)
        # q 가 벡터면 결과의 맨 앞 축이 q 다. 판마다 그 몫을 실어 더한다.
        parts = gg.reshape(qs.size, -1) if _np.ndim(qq) else gg.reshape(1, -1)
        total = (sheets * parts[:, :, None]).sum(axis=0)
        if dim is None:
            return (total.reshape(t.data.shape),)
        moved = data.shape[:dim] + data.shape[dim + 1:] + (n,)
        return (_np.moveaxis(total.reshape(moved), -1, dim),)

    return t._make(_np.asarray(out, dtype=t.data.dtype), (t,), back,
                   "QuantileBackward0")


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


def _no_bool_accumulate(name, dt):
    """누적에 `dtype=bool` 은 torch 가 거절한다(실측 — `NotImplementedError` 다).

    **`sum(dtype=bool)` 은 되는데 `cumsum(dtype=bool)` 은 안 된다.** 규칙이 아니라
    torch 가 그 커널을 안 만든 것이고, 관대한 쪽으로 갈리는 것도 갈리는 것이라
    따라간다 — 여기서 값을 내주면 그 코드가 진짜 torch 에서 깨진다.
    """
    plain = getattr(dt, "np", dt)
    if _np.dtype(plain) == _np.bool_:
        raise NotImplementedError(_like_torch(
            f"{name} 은 결과 형이 참거짓일 수 없습니다.",
            f'"{name}_out_cpu" not implemented for \'Bool\''))


def cumsum(t, dim, dtype=None):
    t = _wrap(t)
    if dtype is not None:
        # **넣기 전에 바꾼다.** 실측: 실수 `[1.7, −2.3, 0.9]` 에 `dtype=int64` 를 주면
        # `[1, −1, −1]` 이다 — 먼저 깎은 `[1, −2, 0]` 의 누적합이다. 접고 나서
        # 깎으면 `[1, 0, 0]` 이 나온다.
        _no_bool_accumulate("cumsum", dtype)
        return cumsum(t.to(dtype), dim).to(dtype)
    return t._make(_np.cumsum(t.data, axis=dim), (t,),
                   lambda g: (_np.flip(_np.cumsum(_np.flip(_np.asarray(g), dim), axis=dim), dim),),
                   "CumsumBackward0")


def cumprod(t, dim, dtype=None):
    """누적 곱. 역방향을 **나눗셈 없이** 쓴다.

    **`dtype` 을 인자로 안 적고 몸통에만 썼다가 무한 재귀가 났다.** 이 파일이
    `_base` 의 `dtype` 을 전역으로 들여오고 있어서, 없는 인자가 **참인 전역**으로
    잡혔다 — `NameError` 가 아니라 `RecursionError` 로 나온다. 이름을 가리는 것이
    이 저장소에서 열한 번째 자리다.

    흔한 유도는 `dL/dx_k = (1/x_k) * sum_{j>=k} g_j y_j` 인데, 입력에 0 이 있으면
    거기서 나눗셈이 터져 조용히 `nan` 이 흐른다. 예외도 안 난다. 그래서 각 k 마다
    `x_k` 를 뺀 곱을 직접 쌓는다 — 길이의 제곱만큼 걸리지만 `cumprod` 는 학습 경로의
    안쪽이 아니고, **0 이 섞였을 때 답이 맞는 쪽**이 이 저장소의 기준이다.
    """
    t = _wrap(t)
    if dtype is not None:
        _no_bool_accumulate("cumprod", dtype)
        return cumprod(t.to(dtype), dim).to(dtype)
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


def diag(t, diagonal=0):
    """1차원이면 대각행렬을 만들고, 2차원이면 대각선을 뽑는다 — 방향이 반대라
    역방향도 반대다.

    **`diagonal` 은 어느 대각선인가다.** 양수는 위쪽, 음수는 아래쪽. 안 받으면
    `x.diag(1)` 이 `TypeError` 로 멈춘다 — 시끄럽게 멈추는 쪽이라 값은 안 갈렸다.
    """
    t = _wrap(t)
    k = int(diagonal)
    out = _np.diag(t.data, k)
    if t.data.ndim == 1:
        def back(g):
            # 만든 행렬에서 그 대각선만 도로 뽑는다.
            return (_np.diagonal(_np.asarray(g), k).copy(),)
    else:
        def back(g):
            z = _np.zeros_like(t.data)
            _np.fill_diagonal(z[max(0, -k):, max(0, k):], 1.0)
            spread = _np.zeros_like(t.data)
            rows, cols = _np.nonzero(z)
            spread[rows, cols] = _np.asarray(g)
            return (spread,)
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


def _gelu_tanh(t):
    """`approximate="tanh"` 쪽 — 0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³))).

    **정확형과 값이 다르다.** 최대차가 1e-4 쯤이라 이 프로젝트의 허용 오차 언저리이고,
    그래서 "거의 같으니 하나로 둔다" 가 통할 뻔한 자리다. torch 가 둘을 나눠 둔 이유는
    tanh 쪽이 빠르기 때문이지 같아서가 아니다 — 골든이 둘을 따로 묻는다.
    """
    d = _np.asarray(t.data, dtype=_np.float64)
    root = _math.sqrt(2.0 / _math.pi)
    inner = root * (d + 0.044715 * d ** 3)
    th = _np.tanh(inner)
    out = (0.5 * d * (1.0 + th)).astype(t.data.dtype)

    def back(g):
        dinner = root * (1.0 + 3 * 0.044715 * d * d)
        grad = 0.5 * (1.0 + th) + 0.5 * d * (1.0 - th * th) * dinner
        return (g * grad.astype(t.data.dtype),)

    return t._make(out, (t,), back, "GeluBackward0")


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


def gelu(t, approximate="none"):
    if approximate == "tanh":
        return _gelu_tanh(_wrap(t))
    if approximate != "none":
        raise ValueError(
            f"gelu(): approximate 는 'none' 또는 'tanh' 입니다 (받은 것: {approximate!r})")
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


def nll_loss(log_probs, target, reduction="mean"):
    n = log_probs.data.shape[0]
    picked = log_probs[_np.arange(n), target.data.astype(int)]
    return _reduce(-picked, reduction)


def l1_loss(pred, target, reduction="mean"):
    return _reduce((_wrap(pred) - _wrap(target)).abs(), reduction)


def smooth_l1_loss(pred, target, beta=1.0, reduction="mean"):
    """작은 오차는 제곱, 큰 오차는 절댓값. 이상치에 덜 흔들린다."""
    diff = _wrap(pred) - _wrap(target)
    small = _np.abs(diff.data) < beta
    return _reduce(where(Tensor(small), 0.5 * diff * diff / beta,
                         diff.abs() - 0.5 * beta), reduction)


# ---------------------------------------------------------------- 손실
#
# **접는 방식이 손실의 일부다.** torch 의 손실은 전부 `reduction` 을 받고, 그 값에
# 따라 원소별·평균·합이 된다. 한 자리에 모아 두면 열셋이 같은 규칙을 쓴다 — 손실마다
# 적으면 열세 자리에서 어긋날 수 있는데 실제로 갈리는 것은 여기 세 줄뿐이다.

def _reduce(out, reduction):
    if reduction == "none":
        return out
    if reduction == "sum":
        return out.sum()
    # **모르는 값을 평균으로 삼키지 않는다.** `else: return out.mean()` 이었는데,
    # 그러면 `reduction="MEAN"` 같은 오타가 조용히 통과해 그대로 학습된다 — 사람이
    # 대문자로 적어 놓고 자기가 고른 것이 쓰이는 줄 안다. torch 는 멈춘다(실측).
    #
    # `else` 가 한 값의 이름을 달고 정의역의 나머지를 전부 삼키는 꼴이고, 같은
    # 모양을 `norm(p)`·`dist(p)` 에서도 봤다.
    if reduction != "mean":
        raise ValueError(_like_torch(
            f"reduction 은 'none'·'mean'·'sum' 중 하나여야 합니다 "
            f"({reduction!r} 을 받았습니다).",
            f"{reduction} is not a valid value for reduction"))
    return out.mean()


def huber_loss(pred, target, reduction="mean", delta=1.0):
    """**`SmoothL1Loss` 와 δ=1 에서만 같다.**

    실제 관계는 `huber(δ) = δ · smooth_l1(β=δ)` 다. 기본값으로만 재면 둘을 같은
    함수로 두고도 통과하므로, 골든이 δ 를 바꿔 묻는다.
    """
    diff = _wrap(pred) - _wrap(target)
    small = _np.abs(diff.data) < delta
    return _reduce(where(Tensor(small), 0.5 * diff * diff,
                         delta * (diff.abs() - 0.5 * delta)), reduction)


def kl_div(pred, target, reduction="mean", log_target=False):
    """`target · (log target − pred)`. `pred` 는 **이미 로그**여야 한다.

    **`reduction` 이 넷이다.** `mean` 은 원소 수로 나누고 `batchmean` 은 배치 크기로
    나눈다 — 수학적 정의에 맞는 것은 뒤쪽이고, torch 자신도 "다음 주 버전에서 바꾸겠다"
    고 경고를 낸다. 지금 값을 맞춰야 하므로 지금 규칙을 따른다.
    """
    p, t = _wrap(pred), _wrap(target)
    out = (t.exp() * (t - p)) if log_target else (t * (t.log() - p))
    if reduction == "batchmean":
        return out.sum() / out.data.shape[0]
    return _reduce(out, reduction)


def poisson_nll_loss(pred, target, log_input=True, full=False, eps=1e-8,
                     reduction="mean"):
    """포아송 음의 로그가능도.

    **스털링 보정은 `target > 1` 일 때만 더한다.** 조건 없이 늘 더하면 target 이 작은
    자리에서만 틀린다 — 실측으로 확인했다(target 이 0·0.5·1 이면 차이가 0 이다).
    """
    p, t = _wrap(pred), _wrap(target)
    out = (p.exp() - t * p) if log_input else (p - t * (p + eps).log())
    if full:
        big = t.data > 1
        stirling = (t * t.log() - t + 0.5 * (2 * _math.pi * t).log())
        out = out + where(Tensor(big.astype(t.data.dtype)), stirling,
                          Tensor(_np.zeros_like(t.data)))
    return _reduce(out, reduction)


def gaussian_nll_loss(pred, target, var, full=False, eps=1e-6, reduction="mean"):
    """가우스 음의 로그가능도.

    **분산을 `eps` 로 자른다.** 안 자르면 0 으로 나눠 무한대가 된다 — `var=1e-9` 에
    기본 `eps=1e-6` 이면 잘린 값으로 124993 이 나온다(실측).
    """
    p, t, v = _wrap(pred), _wrap(target), _wrap(var)
    safe = clamp(v, min=eps)
    diff = p - t
    out = 0.5 * (safe.log() + diff * diff / safe)
    if full:
        out = out + 0.5 * _math.log(2 * _math.pi)
    return _reduce(out, reduction)


def margin_ranking_loss(input1, input2, target, margin=0.0, reduction="mean"):
    """`max(0, −y·(x₁ − x₂) + margin)`."""
    a, b, t = _wrap(input1), _wrap(input2), _wrap(target)
    return _reduce(relu(-t * (a - b) + margin), reduction)


def cosine_embedding_loss(input1, input2, target, margin=0.0, reduction="mean"):
    """`y=1` 이면 `1 − cos`, `y=−1` 이면 `max(0, cos − margin)`."""
    a, b, t = _wrap(input1), _wrap(input2), _wrap(target)
    cos = cosine_similarity(a, b, dim=1)
    same = Tensor((t.data > 0).astype(cos.data.dtype))
    return _reduce(same * (1 - cos) + (1 - same) * relu(cos - margin), reduction)


def hinge_embedding_loss(pred, target, margin=1.0, reduction="mean"):
    """`y=1` 이면 `x` 그대로, `y=−1` 이면 `max(0, margin − x)`.

    **둘로 가르는 것이 아니라 둘을 더한다.** torch 는 `y ≠ 1` 인 자리에 여백 항을,
    `y ≠ −1` 인 자리에 `x` 를 놓고 **합한다** — ±1 에서는 한쪽만 켜져 평소 식과 같지만
    `y=0` 에서는 **둘 다** 켜진다(실측: `x=−1` 에 1.0, `x=2` 에 2.0).

    `y > 0` 으로 갈라 놓았더니 여기서 조용히 갈렸다. 손실이 ±1 만 받는다고 적혀 있어도
    실제로 오는 값이 그것뿐이라는 보장은 없고, `sign()` 은 0 을 만든다.
    """
    p, t = _wrap(pred), _wrap(target)
    dt = p.data.dtype
    not_one = Tensor((t.data != 1).astype(dt))
    not_neg = Tensor((t.data != -1).astype(dt))
    return _reduce(not_one * relu(margin - p) + not_neg * p, reduction)


def soft_margin_loss(pred, target, reduction="mean"):
    """`log(1 + e^{−y·x})`. 로그·지수를 그대로 쓰면 큰 값에서 넘치므로 `softplus` 로 간다."""
    p, t = _wrap(pred), _wrap(target)
    return _reduce(softplus(-t * p), reduction)


def pairwise_distance(x1, x2, p=2.0, eps=1e-6, keepdim=False):
    """짝지어진 두 줄 사이의 거리.

    **`eps` 는 결과가 아니라 차에 더한다.** `p=1` 로 차가 정확히 1.0 인 자리를
    물으면 1.0000020 이 나온다(= 1 + 2·1e-6) — 결과에 더한다고 읽으면 1.000001 이
    되어 자릿수 하나가 갈린다. 실측으로 확인했다.
    """
    a, b = _wrap(x1), _wrap(x2)
    diff = (a - b) + eps
    return vector_norm(diff, ord=p, dim=-1, keepdim=keepdim)


def pdist(x, p=2.0):
    """한 묶음 안의 **모든 짝** 사이 거리. 위 삼각만 준다."""
    t = _wrap(x)
    n = t.data.shape[0]
    rows = [i for i in range(n) for _ in range(i + 1, n)]
    cols = [j for i in range(n) for j in range(i + 1, n)]
    diff = t[rows] - t[cols]
    return vector_norm(diff, ord=p, dim=-1)


def triplet_margin_loss(anchor, positive, negative, margin=1.0, p=2.0, eps=1e-6,
                        swap=False, reduction="mean"):
    """`max(0, d(a,p) − d(a,n) + margin)`.

    `swap` 은 `d(a,n)` 대신 `min(d(a,n), d(p,n))` 을 쓴다 — 음성이 양성에 더 가까우면
    그쪽이 더 어려운 짝이기 때문이다.
    """
    a, pos, neg = _wrap(anchor), _wrap(positive), _wrap(negative)
    dp = pairwise_distance(a, pos, p=p, eps=eps)
    dn = pairwise_distance(a, neg, p=p, eps=eps)
    if swap:
        dn = minimum(dn, pairwise_distance(pos, neg, p=p, eps=eps))
    return _reduce(relu(dp - dn + margin), reduction)


def triplet_margin_with_distance_loss(anchor, positive, negative,
                                      distance_function=None, margin=1.0,
                                      swap=False, reduction="mean"):
    """거리 함수를 받는 삼중항. 기본값은 쌍별 거리라 위와 같은 답이 나온다."""
    a, pos, neg = _wrap(anchor), _wrap(positive), _wrap(negative)
    dist = distance_function or (lambda u, v: pairwise_distance(u, v))
    dp, dn = dist(a, pos), dist(a, neg)
    if swap:
        dn = minimum(dn, dist(pos, neg))
    return _reduce(relu(dp - dn + margin), reduction)


def multilabel_soft_margin_loss(pred, target, weight=None, reduction="mean"):
    """자리마다 독립인 이진 분류를 **반 전체로 평균**한다."""
    p, t = _wrap(pred), _wrap(target)
    each = t * logsigmoid(p) + (1 - t) * logsigmoid(-p)
    if weight is not None:
        each = each * _wrap(weight)
    return _reduce(-each.mean(dim=-1), reduction)


def multi_margin_loss(pred, target, p=1, margin=1.0, weight=None, reduction="mean"):
    """정답 자리와 나머지 사이의 여백.

    **반의 개수로 나눈다** — 견준 짝의 수가 아니다. 정답 자리도 분모에 들어간다는
    뜻이고, 짝의 수로 나누면 클래스가 셋일 때 3/2 배가 나온다.
    """
    x, t = _wrap(pred), _wrap(target)
    n, classes = x.data.shape
    idx = _np.arange(n)
    correct = x[idx, t.data.astype(_np.intp)].unsqueeze(1)
    each = relu(margin - correct + x) ** p
    if weight is not None:
        each = each * _wrap(weight)[t.data.astype(_np.intp)].unsqueeze(1)
    # 정답 자리는 `margin` 이 그대로 남으므로 빼 준다.
    keep = _np.ones((n, classes), dtype=x.data.dtype)
    keep[idx, t.data.astype(_np.intp)] = 0.0
    return _reduce((each * Tensor(keep)).sum(dim=1) / classes, reduction)


def multilabel_margin_loss(pred, target, reduction="mean"):
    """**표적이 자리 목록이고 −1 이 끝을 뜻한다.**

    `[3, 0, -1, 1]` 은 "3 번과 0 번이 정답" 이라는 뜻이고 뒤의 1 은 안 읽는다. 그
    규약을 안 지키면 −1 을 반의 하나로 세거나 끝난 뒤를 계속 읽는다.
    """
    x, t = _wrap(pred), _wrap(target)
    rows, classes = x.data.shape
    total = None
    for r in range(rows):
        labels = []
        for v in t.data[r]:
            if v < 0:
                break
            labels.append(int(v))
        others = [c for c in range(classes) if c not in labels]
        for i in labels:
            for j in others:
                term = relu(1 - (x[r, i] - x[r, j]))
                total = term if total is None else total + term
    out = (total if total is not None else Tensor(_np.zeros((), dtype=x.data.dtype)))
    return _reduce(out.reshape(1) / classes, reduction)


# ---------------------------------------------------------------- 창 펴기
#
# **`unfold` 와 `fold` 는 서로의 역이 아니다.** `unfold` 는 창을 열로 펴고 `fold` 는
# 그것을 되접는데 **겹친 자리를 더한다** — 4×4 를 2×2 창으로 펴서 그대로 되접으면
# 가운데가 네 번 세어져 원본이 안 나온다. 합치는 것이 규약이다.
#
# 그래서 색인 하나로 둘을 만든다. 어느 자리에서 왔는지를 정리해 두면 `unfold` 는
# 그 자리를 모으는 것이고 `fold` 는 그 자리로 **더해 넣는 것**이라, 한쪽의 역방향이
# 곧 다른 쪽이 된다. 패딩에서 쓴 것과 같은 기계다.

def _window_index(shape, kernel, dilation, padding, stride):
    """`(C·kh·kw, L)` 자리표. 값은 **패딩된** 입력의 평평한 자리다."""
    c, h, w = shape
    kh, kw = kernel
    dh, dw = dilation
    ph, pw = padding
    sh, sw = stride
    hp, wp = h + 2 * ph, w + 2 * pw
    out_h = (hp - dh * (kh - 1) - 1) // sh + 1
    out_w = (wp - dw * (kw - 1) - 1) // sw + 1
    idx = _np.empty((c * kh * kw, out_h * out_w), dtype=_np.intp)
    row = 0
    for ch in range(c):
        for i in range(kh):
            for j in range(kw):
                col = 0
                for oh in range(out_h):
                    for ow in range(out_w):
                        y = oh * sh + i * dh
                        x = ow * sw + j * dw
                        idx[row, col] = (ch * hp + y) * wp + x
                        col += 1
                row += 1
    return idx, (out_h, out_w)


def _pair(v):
    return (v, v) if isinstance(v, int) else tuple(v)


def unfold_im2col(x, kernel_size, dilation=1, padding=0, stride=1):
    """창을 열로 편다. `(N, C, H, W)` → `(N, C·kh·kw, L)`.

    **이름이 이미 있는 것과 부딪힌다.** `Tensor.unfold(dim, size, step)` 은 한 축을
    창으로 미는 **뷰**이고 이것은 im2col 이다 — torch 도 이름이 같고 하는 일이 다르다
    (`torch.Tensor.unfold` 대 `torch.nn.functional.unfold`). 모듈 자리에 같은 이름으로
    두었더니 앞의 것을 덮어서 `shape::unfold` 케이스 셋이 한꺼번에 무너졌다.
    여기서는 이름을 갈라 두고 `F.unfold` 쪽에만 이것을 건다.
    """
    t = _mat(x, "unfold", square=False)
    if t.data.ndim != 4:
        _unsupported("unfold(4차원이 아닌 것)")
    kernel, dil = _pair(kernel_size), _pair(dilation)
    pad_, strd = _pair(padding), _pair(stride)
    padded = pad(t, (pad_[1], pad_[1], pad_[0], pad_[0]))
    n, c = t.data.shape[0], t.data.shape[1]
    idx, _ = _window_index(t.data.shape[1:], kernel, dil, pad_, strd)
    flat = padded.reshape(n, -1)
    return flat[:, idx.reshape(-1)].reshape(n, idx.shape[0], idx.shape[1])


def fold(x, output_size, kernel_size, dilation=1, padding=0, stride=1):
    """편 것을 되접는다. **겹친 자리는 더한다** — 그것이 이 함수의 뜻이다."""
    t = _wrap(x)
    kernel, dil = _pair(kernel_size), _pair(dilation)
    pad_, strd = _pair(padding), _pair(stride)
    out_h, out_w = _pair(output_size)
    n = t.data.shape[0]
    c = t.data.shape[1] // (kernel[0] * kernel[1])
    idx, _ = _window_index((c, out_h, out_w), kernel, dil, pad_, strd)
    hp, wp = out_h + 2 * pad_[0], out_w + 2 * pad_[1]
    flat_idx = idx.reshape(-1)

    def back(g):
        gg = _np.asarray(g).reshape(n, -1)
        return (gg[:, flat_idx].reshape(t.data.shape),)

    out = _np.zeros((n, c * hp * wp), dtype=t.data.dtype)
    _np.add.at(out, (slice(None), flat_idx), t.data.reshape(n, -1))
    made = out.reshape(n, c, hp, wp)
    if pad_[0] or pad_[1]:
        made = made[:, :, pad_[0]:hp - pad_[0], pad_[1]:wp - pad_[1]]
    return t._make(made, (t,), back, "FoldBackward0")


# ---------------------------------------------------------------- 나머지 층

def bilinear(input1, input2, weight, bias=None):
    """`y[o] = x₁ᵀ·W[o]·x₂ + b[o]`. 가중치가 **세 축**이다."""
    a, b_, w = _wrap(input1), _wrap(input2), _wrap(weight)
    out = einsum("bi,oij,bj->bo", a, w, b_)
    return out + _wrap(bias) if bias is not None else out


def local_response_norm(x, size, alpha=1e-4, beta=0.75, k=1.0):
    """이웃 채널로 나눈다.

    **창이 한쪽으로 치우쳐 있다.** 채널 `c` 의 창은 `[c − n//2, c + n − 1 − n//2]`
    이고, `size=2` 면 `{c−1, c}` 이지 `{c, c+1}` 이 아니다 — 재서 확인했다.
    가운데를 잡으면 값이 한 칸씩 밀리는데 크기가 같아서 모양으로는 안 보인다.
    """
    t = _wrap(x)
    left = size // 2
    sq = t * t
    total = None
    for offset in range(size):
        shift = offset - left
        piece = _np.zeros_like(sq.data)
        c = sq.data.shape[1]
        src = slice(max(0, shift), min(c, c + shift))
        dst = slice(max(0, -shift), min(c, c - shift))
        piece[:, dst] = 1.0
        moved = _roll_channels(sq, shift) * Tensor(piece)
        total = moved if total is None else total + moved
    return t / (total * (alpha / size) + k) ** beta


def _roll_channels(t, shift):
    """채널 축을 `shift` 만큼 민다. 밖에서 들어온 자리는 뒤에서 0 으로 지운다."""
    if shift == 0:
        return t
    rolled = _np.roll(t.data, -shift, axis=1)
    return t._make(rolled, (t,),
                   lambda g: (_np.roll(_np.asarray(g), shift, axis=1),),
                   "RollBackward0")


def rrelu(x, lower=1.0 / 8, upper=1.0 / 3, training=False, inplace=False):
    """음수 쪽 기울기를 뽑아 쓴다.

    **평가 모드에서는 가운데로 정해진다** — 기본값이면 `(1/8 + 1/3)/2 = 0.2292` 다.
    학습 때만 `[lower, upper]` 에서 뽑으므로, 난수가 끼는 자리는 그쪽뿐이다.
    """
    t = _wrap(x)
    if not training:
        return leaky_relu(t, (lower + upper) / 2)
    slope = _rng.uniform(lower, upper, t.data.shape).astype(t.data.dtype)
    return where(Tensor((t.data > 0).astype(t.data.dtype)), t, t * Tensor(slope))


# ---------------------------------------------------------------- 자리 옮기기
#
# 셋 다 **값을 안 바꾸고 자리만 바꾼다.** 그래서 순방향이 `reshape` + 축 바꾸기이고
# 역방향은 그 반대다 — 우리 `transpose`·`reshape` 가 이미 그 일을 하므로 여기는
# 조합이다. 입력을 `arange` 로 두면 어느 자리가 어디로 갔는지가 답에 그대로 나온다.

def pixel_shuffle(x, upscale_factor):
    """`(N, C·r², H, W)` → `(N, C, H·r, W·r)`. 채널을 잘라 공간에 심는다.

    **엇갈리는 순서가 값의 전부다.** 채널을 `(C, r, r)` 로 갈라 두 `r` 을 각각 `H` 와
    `W` 뒤에 끼워 넣는다 — `(N, C, H, r, W, r)` 로 세운 뒤 붙이는 것이 그 뜻이다.
    순서를 바꾸면 모양은 같고 그림만 뒤섞인다.
    """
    t = _wrap(x)
    r = upscale_factor
    n, c, h, w = t.data.shape
    out = t.reshape(n, c // (r * r), r, r, h, w).permute(0, 1, 4, 2, 5, 3)
    return out.reshape(n, c // (r * r), h * r, w * r)


def pixel_unshuffle(x, downscale_factor):
    """`pixel_shuffle` 의 역. 공간을 잘라 채널에 쌓는다."""
    t = _wrap(x)
    r = downscale_factor
    n, c, h, w = t.data.shape
    out = t.reshape(n, c, h // r, r, w // r, r).permute(0, 1, 3, 5, 2, 4)
    return out.reshape(n, c * r * r, h // r, w // r)


def channel_shuffle(x, groups):
    """채널을 묶음으로 갈라 **엇갈려 다시 놓는다.**

    `[0,1,2,3]` 을 두 묶음으로 섞으면 `[0,2,1,3]` 이다 — 묶음별 합성곱 뒤에 정보가
    묶음 안에만 갇히는 것을 푸는 자리라, 엇갈리는 방향이 값의 전부다.
    """
    t = _wrap(x)
    n, c = t.data.shape[0], t.data.shape[1]
    rest = t.data.shape[2:]
    out = t.reshape(n, groups, c // groups, *rest)
    out = out.transpose(1, 2)
    return out.reshape(n, c, *rest)


# ---------------------------------------------------------------- 채널째 dropout
#
# **원소가 아니라 채널을 떨군다.** 이름이 `Dropout` 옆에 있어서 "2 차원용" 으로 읽기
# 쉬운데 하는 일이 다르다 — 한 채널을 통째로 0 으로 만들거나 통째로 남긴다.
#
# `AlphaDropout` 은 거기에 더해 **0 을 안 넣는다.** SELU 와 함께 쓰라고 만든 것이라
# 떨군 자리에 음의 상수를 넣고 전체에 아핀 변환을 걸어 평균과 분산을 지킨다. 0 을
# 넣으면 SELU 의 자기정규화가 깨지는데, 값이 그럴듯해서 학습이 도는 동안은 안 보인다.

def _channel_mask(t, p):
    """채널마다 하나씩 뽑은 0/1. 공간 축은 1 로 두어 브로드캐스트한다."""
    shape = t.data.shape[:2] + (1,) * (t.data.ndim - 2)
    return (_rng.random(shape) > p).astype(t.data.dtype)


def _feature_dropout(x, p, training, name):
    t = _wrap(x)
    if not training or p == 0:
        return t
    if p >= 1:
        return t * Tensor(_np.zeros((), dtype=t.data.dtype))
    return t * Tensor(_channel_mask(t, p) / (1 - p))


def dropout1d(x, p=0.5, training=True, inplace=False):
    t = _wrap(x)
    if t.data.ndim not in (2, 3):
        raise RuntimeError(
            f"dropout1d: Expected 2D or 3D input, but received a {t.data.ndim}D "
            "input. Note that dropout1d exists to provide channel-wise dropout on "
            "inputs with 1 spatial dimension, a channel dimension, and an optional "
            "batch dimension (i.e. 2D or 3D inputs).")
    return _feature_dropout(t, p, training, "dropout1d")


def dropout2d(x, p=0.5, training=True, inplace=False):
    return _feature_dropout(x, p, training, "dropout2d")


def dropout3d(x, p=0.5, training=True, inplace=False):
    return _feature_dropout(x, p, training, "dropout3d")


# SELU 의 고정점. `alpha_dropout` 이 떨군 자리에 넣는 값이 여기서 나온다.
_ALPHA_PRIME = -1.7580993408473766


def _alpha_affine(p):
    """떨군 자리를 상수로 채운 뒤 평균과 분산을 되돌리는 아핀 계수 `(a, b)`."""
    a = ((1 - p) * (1 + p * _ALPHA_PRIME ** 2)) ** -0.5
    return a, -a * p * _ALPHA_PRIME


def alpha_dropout(x, p=0.5, training=False, inplace=False):
    t = _wrap(x)
    if not training or p == 0:
        return t
    keep = Tensor((_rng.random(t.data.shape) > p).astype(t.data.dtype))
    a, b = _alpha_affine(p)
    return (t * keep + (1 - keep) * _ALPHA_PRIME) * a + b


def feature_alpha_dropout(x, p=0.5, training=False, inplace=False):
    """채널째 떨구는 `alpha_dropout`."""
    t = _wrap(x)
    if not training or p == 0:
        return t
    keep = Tensor(_channel_mask(t, p))
    a, b = _alpha_affine(p)
    return (t * keep + (1 - keep) * _ALPHA_PRIME) * a + b


def _pad_index(mode, size, before, after):
    """출력 자리마다 **입력의 어느 자리를 읽는지.** `-1` 은 채운 자리다.

    네 모드가 여기서만 갈린다. 아래 세 줄이 규약의 전부이고, 진짜 torch 에 물어
    자리마다 맞췄다(`[0,1,2]` 를 앞 2·뒤 1 로 늘린 것):

        reflect    2 1 [0 1 2] 1   ← 가장자리를 거울로 **하되 가장자리는 안 겹친다**
        replicate  0 0 [0 1 2] 2   ← 가장자리를 늘인다
        circular   1 2 [0 1 2] 0   ← 반대편에서 가져온다

    색인 하나로 정리해 두면 순방향은 `take` 이고 역방향은 **같은 색인으로 모아
    더하기**다. 모드마다 역방향을 따로 적으면 네 번 틀릴 자리가 생긴다.
    """
    idx = []
    for i in range(-before, size + after):
        if 0 <= i < size:
            idx.append(i)
        elif mode == "constant":
            idx.append(-1)
        elif mode == "replicate":
            idx.append(0 if i < 0 else size - 1)
        elif mode == "circular":
            idx.append(i % size)
        else:
            j = i
            while not (0 <= j < size):
                j = -j if j < 0 else 2 * (size - 1) - j
            idx.append(j)
    return _np.asarray(idx, dtype=_np.intp)


def pad(x, padding, mode="constant", value=0.0):
    """마지막 차원부터 (앞, 뒤) 순으로 받는다 — torch 의 규칙이다.

    **짝의 개수와 랭크가 맞물린다.** 짝이 하나면 2·3 차원, 둘이면 3·4 차원, 셋이면
    4·5 차원이라야 한다 — torch 가 그 밖을 `NotImplementedError` 로 거절한다. 아무
    랭크나 받으면 축을 잘못 잡고도 통과하므로 여기서 같이 막는다.
    """
    x = _wrap(x)
    rank = x.data.ndim
    pairs = len(padding) // 2
    if mode != "constant" and rank not in (pairs + 1, pairs + 2):
        raise NotImplementedError(
            f"Padding size {len(padding)} is not supported for {rank}D input tensor")

    data = x.data
    steps = []
    for i in range(pairs):
        axis = rank - 1 - i
        before, after = padding[2 * i], padding[2 * i + 1]
        if before == 0 and after == 0:
            continue
        size = data.shape[axis]
        if mode == "reflect" and (before >= size or after >= size):
            raise RuntimeError(
                "Argument #4: Padding size should be less than the corresponding "
                f"input dimension, but got: padding ({before}, {after}) at dimension "
                f"{axis} of input {rank}")
        idx = _pad_index(mode, size, before, after)
        steps.append((axis, idx, size))
        taken = _np.take(data, _np.maximum(idx, 0), axis=axis)
        if mode == "constant":
            hole = idx < 0
            if hole.any():
                cut = [slice(None)] * rank
                cut[axis] = hole
                taken[tuple(cut)] = value
        data = taken

    def back(g):
        gg = _np.asarray(g)
        # 뒤에서부터 되짚는다. 읽어 온 자리마다 **모아 더한다** — 거울과 감기는
        # 한 입력을 여러 번 읽으므로 덮어쓰면 그만큼이 사라진다.
        for axis, idx, size in reversed(steps):
            shape = list(gg.shape)
            shape[axis] = size
            out = _np.zeros(shape, dtype=gg.dtype)
            keep = idx >= 0
            head = (slice(None),) * axis
            _np.add.at(out, head + (idx[keep],), gg[head + (keep,)])
            gg = out
        return (gg,)

    return x._make(data, (x,), back, "PadBackward0")


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


def allclose(a, b, rtol=1e-5, atol=1e-8, equal_nan=False):
    """**`equal_nan` 을 받는다.** 기본은 거짓이라 NaN 끼리도 안 같다(실측).

    골든 하네스는 이 인자를 **안 켠다** — 켜면 NaN 이 통과하면 안 되는 자리에서
    통과한다. 그것과 이것은 다른 자리다: 여기는 torch 가 주는 인자를 우리도 주는
    것이고, 켤지 말지는 부르는 쪽이 정한다.
    """
    return bool(_np.allclose(_wrap(a).data, _wrap(b).data, rtol=rtol, atol=atol,
                             equal_nan=bool(equal_nan)))


def equal(a, b):
    return bool(_np.array_equal(_wrap(a).data, _wrap(b).data))


def isfinite(t):
    return Tensor(_np.isfinite(_wrap(t).data))


def bincount(t, weights=None, minlength=0):
    """칸마다 몇 번 나왔는가. **무게를 주면 개수 대신 무게를 더한다.**

    형이 갈린다(실측): 무게 없이는 `int64`, 무게가 있으면 그 무게의 형이다 —
    개수를 세는 것과 값을 더하는 것이 다른 일이기 때문이다.
    """
    t = _wrap(t)
    _refuses_bool(t.data, "bincount 는 참거짓을 받지 않습니다.",
                  '"bincount_cpu" not implemented for \'Bool\'',
                  kind=NotImplementedError)
    # `intp` 다 — wasm32 에서 int64 를 주면 거절한다. 위 `repeat_interleave` 참고.
    w = None if weights is None else _np.asarray(_wrap(weights).data)
    out = _np.bincount(t.data.astype(_np.intp), weights=w,
                       minlength=int(minlength))
    # numpy 는 무게가 있으면 언제나 float64 를 준다. 무게의 형으로 되돌린다.
    if w is not None:
        return Tensor(out.astype(w.dtype))
    return Tensor(out)


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


class enable_grad:                                       # noqa: N801
    """`no_grad` 안에서 **다시 켠다.** 중첩이 되어야 하므로 이전 값을 되돌린다."""

    def __enter__(self):
        self._prev = _grad_mode.enabled
        _grad_mode.enabled = True
        return self

    def __exit__(self, *exc):
        _grad_mode.enabled = self._prev
        return False

    def __call__(self, fn):
        def wrapper(*a, **k):
            with enable_grad():
                return fn(*a, **k)
        return wrapper


class set_grad_enabled:                                  # noqa: N801
    """켤지 끌지를 **값으로** 받는다. `with` 로도 쓰고 그냥 불러도 된다."""

    def __init__(self, mode):
        self._prev = _grad_mode.enabled
        _grad_mode.enabled = bool(mode)
        self._mode = bool(mode)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        _grad_mode.enabled = self._prev
        return False

    def __call__(self, fn):
        def wrapper(*a, **k):
            with set_grad_enabled(self._mode):
                return fn(*a, **k)
        return wrapper


def is_grad_enabled():
    return bool(_grad_mode.enabled)


class inference_mode:                                    # noqa: N801
    """**여기서는 `no_grad` 와 같다.**

    진짜 torch 에서 이것은 더 세다 — 안에서 만든 텐서에 표를 붙여 나중에 autograd
    에 못 들어가게 막는다. 그 표를 흉내내면 "왜 이 텐서를 못 쓰나" 하는 오류를
    우리가 만들어 내는 셈이라, 여기서는 기울기만 끈다. `is_inference` 가 늘 거짓인
    이유가 그것이다 — 그 표를 안 붙이므로 없다고 말하는 것이 사실이다.
    """

    def __init__(self, mode=True):
        self._mode = bool(mode)
        self._prev = None

    def __enter__(self):
        self._prev = _grad_mode.enabled
        if self._mode:
            _grad_mode.enabled = False
        return self

    def __exit__(self, *exc):
        _grad_mode.enabled = self._prev
        return False

    def __call__(self, fn):
        def wrapper(*a, **k):
            with inference_mode(self._mode):
                return fn(*a, **k)
        return wrapper


def is_inference(t):
    """**늘 거짓이다.** 위에 적은 대로 그 표를 안 붙이므로 없다고 말한다."""
    return False


def is_inference_mode_enabled():
    return False


# ── 살펴보기 ────────────────────────────────────────────────────────────────
#
# 값을 안 바꾸고 **묻기만 하는** 것들. 교재 코드가 분기에 쓰는 자리라, 없으면
# 계산이 다 맞아도 그 줄에서 멈춘다.

def is_tensor(x):
    return isinstance(x, Tensor)


def is_storage(x):
    """**늘 거짓이다.** 저장(Storage) 이라는 층을 우리는 안 둔다 — numpy 배열이
    그 자리이고, 그것을 Storage 라고 부르면 없는 API 를 있다고 말하는 것이 된다."""
    return False


def is_floating_point(x):
    return _wrap(x).data.dtype.kind == "f"


def is_signed(x):
    return _wrap(x).data.dtype.kind in "fi"


def is_nonzero(x):
    """**원소가 하나여야 한다.** torch 는 아니면 던진다 — 여러 개일 때 무엇이 참인지가
    정해져 있지 않기 때문이다."""
    data = _wrap(x).data
    if data.size != 1:
        raise RuntimeError(
            f"Boolean value of Tensor with {data.size} elements is ambiguous")
    return bool(data.reshape(-1)[0] != 0)


def is_same_size(a, b):
    return tuple(_wrap(a).data.shape) == tuple(_wrap(b).data.shape)


def is_distributed(x):
    """**늘 거짓이다.** 탭 하나 안이라 나눌 자리가 없다."""
    return False


def typename(x):
    """`torch.FloatTensor` 처럼 옛 이름을 낸다. 텐서가 아니면 파이썬 형 이름이다."""
    if not isinstance(x, Tensor):
        return type(x).__name__
    kinds = {"float32": "FloatTensor", "float64": "DoubleTensor",
             "int64": "LongTensor", "bool": "BoolTensor"}
    return "torch." + kinds.get(str(x.data.dtype), "FloatTensor")


_PROMOTE_ORDER = ("bool", "int64", "float32", "float64")


def promote_types(a, b):
    """둘을 담을 수 있는 형. **순서가 정해져 있다** — bool < int64 < float32 < float64."""
    names = [getattr(t, "name", str(t)) for t in (a, b)]
    best = max(names, key=lambda n: _PROMOTE_ORDER.index(n)
               if n in _PROMOTE_ORDER else 0)
    return {"bool": _bool_dtype, "int64": _int64, "float32": _float32,
            "float64": _float64}[best]


def can_cast(from_type, to_type):
    """**한 방향만 참이다.** 좁아지는 쪽(실수 → 정수)은 거짓이다 — 값이 깎이므로."""
    names = [getattr(t, "name", str(t)) for t in (from_type, to_type)]
    if any(n not in _PROMOTE_ORDER for n in names):
        return False
    return _PROMOTE_ORDER.index(names[0]) <= _PROMOTE_ORDER.index(names[1])


def get_default_dtype():
    return _float32


def set_default_dtype(dt):
    """**받되 바꾸지 않는다.** 우리 저장은 float32 하나이고, 바꾼 척하면 그다음에
    만드는 텐서가 말과 다른 형으로 나온다. float32 를 주면 조용히 넘어가고 그 밖은
    시끄럽게 거절한다 — 아무 말 없이 무시하는 것이 제일 나쁘다."""
    if getattr(dt, "name", str(dt)) != "float32":
        _unsupported(f"set_default_dtype({dt}) — 저장이 float32 하나입니다")
    return None


class _FInfo:
    """`torch.finfo` 가 주는 것. numpy 가 이미 아는 수를 이름만 바꿔 낸다."""

    def __init__(self, dt):
        info = _np.finfo(getattr(dt, "np", _np.float32))
        self.eps = float(info.eps)
        self.max = float(info.max)
        self.min = float(info.min)
        self.tiny = float(info.tiny)
        self.smallest_normal = float(info.tiny)
        self.resolution = float(info.resolution)
        self.bits = int(info.bits)
        self.dtype = getattr(dt, "name", "float32")

    def __repr__(self):
        return (f"finfo(resolution={self.resolution}, min={self.min}, "
                f"max={self.max}, eps={self.eps}, "
                f"smallest_normal={self.smallest_normal}, tiny={self.tiny}, "
                f"dtype={self.dtype})")


class _IInfo:
    def __init__(self, dt):
        info = _np.iinfo(getattr(dt, "np", _np.int64))
        self.max = int(info.max)
        self.min = int(info.min)
        self.bits = int(info.bits)
        self.dtype = getattr(dt, "name", "int64")

    def __repr__(self):
        return (f"iinfo(min={self.min}, max={self.max}, dtype={self.dtype})")


def finfo(dt=None):
    return _FInfo(_float32 if dt is None else dt)


def iinfo(dt):
    return _IInfo(dt)


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
_LuFactorEx = _named("linalg_lu_factor_ex", "LU", "pivots", "info")
_LdlFactor = _named("linalg_ldl_factor", "LD", "pivots")
_LdlFactorEx = _named("linalg_ldl_factor_ex", "LD", "pivots", "info")
_Geqrf = _named("geqrf", "a", "tau")
_Frexp = _named("frexp", "mantissa", "exponent")
# 통계의 답들. `histogram` 과 `histogramdd` 의 인자 이름이 `range` 라 그 안에서는
# 파이썬 내장이 가려진다 — 이 파일에서 아홉 번째 겪는 일이라 별칭을 미리 둔다.
_Histogram = _named("histogram", "hist", "bin_edges")
_HistogramDD = _named("histogramdd", "hist", "bin_edges")
_Mode = _named("mode", "values", "indices")
_NanMedian = _named("nanmedian", "values", "indices")
_builtin_range = range
# 최상위 선형대수의 답들. **`triangular_solve` 는 둘을 주는데 둘째가 계수 행렬의
# 사본이다**(실측) — 쓸모가 없어 보이지만 torch 가 그렇게 주므로 자리를 맞춘다.
_TriangularSolve = _named("triangular_solve", "solution", "cloned_coefficient")
_LuInfos = _named("lu", "LU", "pivots", "info")
_LuUnpack = _named("lu_unpack", "P", "L", "U")
_Lobpcg = _named("lobpcg", "eigenvalues", "eigenvectors")
_SvdLowrank = _named("svd_lowrank", "U", "S", "V")


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


# ---- 분해의 기울기
#
# 오래 안 넣었다. 이유가 있었다 — 유도가 까다롭고 틀리면 **조용히** 틀린다. 값은
# 맞고 학습만 미묘하게 갈리는 종류라, 없는 것을 시끄럽게 두는 편이 나았다.
#
# 이제 넣는다. 바뀐 것은 유도가 쉬워진 것이 아니라 **대조할 것이 생긴 것**이다.
# 골든이 진짜 torch 의 수를 자리마다 들고 있어서, 틀리면 조용히가 아니라 크게 틀린다.

def qr(t, mode="reduced"):
    """QR 분해. **기울기가 있다.**

        N = Qᵀ·Q̄ − R̄·Rᵀ
        Ā = [Q̄ + Q·(tril(N − Nᵀ, −1) − N)]·R⁻ᵀ

    아래 삼각만 남기는 자리가 이 유도의 전부다. `QᵀQ = I` 를 미분하면 `C = Qᵀ·dQ` 가
    **반대칭**이 되고, `dR·R⁻¹` 이 위삼각이라는 것과 겹치면 `C` 의 자유도가 아래
    삼각에만 남는다. 위쪽은 그 거울이라 따로 셀 것이 없다.

    **`R⁻ᵀ` 를 `R⁻¹` 로 잘못 풀면 여기서 조용히 틀린다.** 실제로 그렇게 틀렸고, 후보
    여덟 개가 전부 안 맞아서 유도를 의심했는데 유도가 아니라 전치가 문제였다.
    `X·R⁻ᵀ` 는 `solve(R, Xᵀ)ᵀ` 이지 `solve(Rᵀ, Xᵀ)ᵀ` 가 아니다.
    """
    t = _mat(t, "qr", square=False)
    q, r = _np.linalg.qr(t.data, mode=mode)
    if mode != "reduced" or t.data.shape[-2] < t.data.shape[-1]:
        # 완전본은 `Q` 에 남는 열이 있고 그쪽으로는 정보가 안 흐른다 — 유도가 다르다.
        return _QR(Tensor(_np.ascontiguousarray(q)), Tensor(_np.ascontiguousarray(r)))

    def back_from(gq, gr):
        n = _T(q) @ gq - gr @ _T(r)
        inner = _np.tril(n - _T(n), -1) - n
        return _T(_np.linalg.solve(r, _T(gq + q @ inner)))

    qt = t._make(_np.ascontiguousarray(q), (t,),
                 lambda g: (back_from(_np.asarray(g), _np.zeros_like(r)),),
                 "QrBackward0")
    rt = t._make(_np.ascontiguousarray(r), (t,),
                 lambda g: (back_from(_np.zeros_like(q), _np.asarray(g)),),
                 "QrBackward0")
    return _QR(qt, rt)


def _svd_raw(data, full_matrices):
    u, s, vh = _np.linalg.svd(data, full_matrices=full_matrices)
    return _np.ascontiguousarray(u), s, _np.ascontiguousarray(vh)


def svd(t, full_matrices=True):
    """특잇값 분해. torch 와 같이 (U, S, Vh) 순서로 돌려준다.

    **특잇값에는 기울기가 있고 `U`·`Vh` 에는 없다.** `dS = diag(Uᵀ dA V)` 라 특잇값
    쪽은 한 줄이고 겹침 문제도 없다. 벡터 쪽은 `1/(sᵢ²−sⱼ²)` 가 들어가서 특잇값이
    겹치면 터지는데, 그 자리는 안 넣었다 — 없는 것이 시끄러운 편이 낫다.
    """
    t = _mat(t, "svd", square=False)
    u, s, vh = _svd_raw(t.data, full_matrices)
    k = s.shape[-1]
    u_thin, vh_thin = u[..., :, :k], vh[..., :k, :]

    def back(g):
        gg = _np.asarray(g)
        idx = _np.arange(k)
        mid = _np.zeros(gg.shape + (k,), dtype=u.dtype)
        mid[..., idx, idx] = gg
        return (u_thin @ mid @ vh_thin,)

    return _SVD(Tensor(u), t._make(s, (t,), back, "SvdBackward0"), Tensor(vh))


def pinverse(t, rcond=1e-15):
    """유사역행렬. **기울기가 있다** — 항이 셋이다.

        Ā = −Pᵀ·Ḡ·Pᵀ + (I − A·P)·Ḡᵀ·P·Pᵀ + Pᵀ·P·Ḡᵀ·(I − P·A)

    **뒤의 두 항은 정사각 정칙에서 0 이 된다** — 그때는 `I − AP` 와 `I − PA` 가 둘 다
    0 이라 첫 항만 남고, 그 첫 항은 역행렬의 기울기와 같은 식이다. 그래서 둘을
    빠뜨려도 **정사각에서는 맞고 직사각에서만 틀린다.** 실제로 그렇게 틀렸고, 정사각
    케이스는 그동안 통과하고 있었다 — 골든이 직사각으로도 묻는 이유가 그것이다.
    """
    t = _mat(t, "pinverse", square=False)
    p = _np.linalg.pinv(t.data, rcond=rcond)
    m, n = t.data.shape[-2], t.data.shape[-1]
    eye_m = _np.eye(m, dtype=p.dtype)
    eye_n = _np.eye(n, dtype=p.dtype)

    def back(g):
        gg = _np.asarray(g)
        pt = _T(p)
        left = -(pt @ gg @ pt)
        mid = (eye_m - t.data @ p) @ _T(gg) @ p @ pt
        right = pt @ p @ _T(gg) @ (eye_n - p @ t.data)
        return (left + mid + right,)

    return t._make(p, (t,), back, "PinverseBackward0")


def matrix_rank(t, tol=None):
    t = _mat(t, "matrix_rank", square=False)
    return Tensor(_np.asarray(_np.linalg.matrix_rank(t.data, tol=tol), dtype=_np.int64))


def eigh(t, UPLO="L"):
    """대칭 행렬의 고윳값·고유벡터. **둘 다 기울기가 있다.**

    **한쪽 삼각만 읽는다.** 기본은 아래쪽이라 `[[4,99],[1,3]]` 과 `[[4,1],[1,3]]` 의
    답이 같다(진짜 torch 에 물어서 확인했다). 대칭을 주는 한 안 드러나는 규약이라,
    행렬 전체를 보는 구현과 여기서 조용히 갈린다.

    고윳값 쪽은 `Ā = V·diag(ḡ)·Vᵀ` 로 한 줄이다. 고유벡터 쪽은
    `Ā = V·(F ∘ (Vᵀ·Ḡ))·Vᵀ` 이고 `F_ij = 1/(λⱼ − λᵢ)` 다 — **고윳값이 겹치면 터진다.**
    torch 도 같이 터지므로 흉내가 아니라 같은 한계다.
    """
    t = _mat(t, "eigh")
    w, v = _np.linalg.eigh(t.data, UPLO=UPLO)
    v = _np.ascontiguousarray(v)
    vt = _T(v)

    def back_values(g):
        gg = _np.asarray(g)
        idx = _np.arange(w.shape[-1])
        mid = _np.zeros(gg.shape + (w.shape[-1],), dtype=v.dtype)
        mid[..., idx, idx] = gg
        return (v @ mid @ vt,)

    def back_vectors(g):
        gg = _np.asarray(g)
        gap = w[..., None, :] - w[..., :, None]
        idx = _np.arange(w.shape[-1])
        # 대각은 0 으로 둔다 — 자기 자신과의 차라 나눗셈이 아니라 정의상 안 흐른다.
        with _np.errstate(divide="ignore", invalid="ignore"):
            f = _np.where(gap == 0, 0.0, 1.0 / _np.where(gap == 0, 1.0, gap))
        f[..., idx, idx] = 0.0
        raw = v @ (f * (vt @ gg)) @ vt
        # **대칭화가 빠지면 안 된다.** `A` 가 대칭이라 위·아래 삼각이 같은 자유도를
        # 나눠 갖는데, 날 식은 그것을 한쪽에 몰아준다. 대각은 맞고 비대각만 갈려서
        # 값 대조 없이는 안 보인다 — 실측으로 골랐다.
        return ((raw + _T(raw)) * 0.5,)

    return _Eigh(t._make(w, (t,), back_values, "EighBackward0"),
                 t._make(v, (t,), back_vectors, "EighBackward0"))


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


def lu_factor_ex(t, pivot=True, check_errors=False):
    """`lu_factor` 에 **`info` 를 하나 더 붙인 것.** 던지는 대신 번호로 알린다.

    0 이면 잘 됐고, `k` 면 `k` 번째 피벗이 0 이라 특이행렬이다(1 부터 센다). 실측:
    `[[1,2],[2,4]]` 이 2 를 낸다. 던지는 판(`lu_factor`)과 갈라 둔 이유는 배치로 풀
    때 한 장이 나빠도 나머지를 이어 가려는 것이다.
    """
    t = _mat(t, "lu_factor_ex", square=False)
    if not pivot:
        _unsupported("lu_factor_ex(pivot=False)")
    lu_data, piv = _lu_pack(t.data)
    n, m = lu_data.shape[-2], lu_data.shape[-1]
    k = min(n, m)
    flat = lu_data.reshape(-1, n, m)
    info = _np.zeros(flat.shape[0], dtype=_np.int32)
    for b in range(flat.shape[0]):
        zero = _np.flatnonzero(_np.diagonal(flat[b])[:k] == 0)
        info[b] = 0 if zero.size == 0 else int(zero[0]) + 1
    shape = t.data.shape[:-2]
    return _LuFactorEx(Tensor(lu_data.astype(t.data.dtype)), Tensor(piv),
                       Tensor(info.reshape(shape) if shape else info[0]))


def _ldl_pack(data):
    """대칭 행렬의 `L D Lᵀ`. **피벗을 안 한다.**

    torch 는 LAPACK 의 Bunch-Kaufman 을 쓰고 그것은 필요하면 자리를 바꾼다. 여기서는
    양의 정부호처럼 바꿀 일이 없는 자리만 다루고, 대각이 0 에 가까우면 **시끄럽게
    거절한다** — 조용히 이어 가면 자리를 안 바꾼 것과 바꾼 것이 다른 답을 내는데
    둘 다 그럴듯하다.

    답은 torch 와 같은 모양으로 **한 장에 겹쳐 담는다** — 대각이 `D`, 그 아래가 `L`.
    """
    a = data.astype(_np.float64)
    n = a.shape[-1]
    flat = a.reshape(-1, n, n).copy()
    out = _np.zeros_like(flat)
    for b in range(flat.shape[0]):
        mat, ld = flat[b], out[b]
        for j in range(n):
            d = mat[j, j] - sum(ld[j, k] ** 2 * ld[k, k] for k in range(j))
            # **`abs` 는 이 파일에서 텐서 함수다** — 모듈 전역이 내장을 가린다.
            # 그 자리를 여기서 또 밟았고, `_np.abs` 로 부르는 것이 이 파일의 규칙이다.
            if _np.abs(d) < 1e-12:
                _unsupported("ldl_factor — 피벗이 필요한 대칭 행렬 (부정부호)")
            ld[j, j] = d
            for i in range(j + 1, n):
                s = sum(ld[i, k] * ld[k, k] * ld[j, k] for k in range(j))
                ld[i, j] = (mat[i, j] - s) / d
    # 교환표는 1 부터 센 항등이다 — 자리를 안 바꿨으므로.
    piv = _np.tile(_np.arange(1, n + 1, dtype=_np.int32), flat.shape[0], )
    return out.reshape(a.shape), piv.reshape(data.shape[:-2] + (n,))


def ldl_factor(t, hermitian=False):
    """대칭 행렬을 `L D Lᵀ` 로. 한 장에 겹쳐 담은 `LD` 와 교환표를 준다."""
    t = _mat(t, "ldl_factor")
    ld, piv = _ldl_pack(t.data)
    return _LdlFactor(Tensor(ld.astype(t.data.dtype)), Tensor(piv))


def ldl_factor_ex(t, hermitian=False, check_errors=False):
    """`ldl_factor` 에 `info` 를 붙인 것. 여기서는 늘 0 이다 — 나쁜 자리는 거절한다."""
    t = _mat(t, "ldl_factor_ex")
    ld, piv = _ldl_pack(t.data)
    shape = t.data.shape[:-2]
    zero = _np.zeros(shape, dtype=_np.int32) if shape else _np.int32(0)
    return _LdlFactorEx(Tensor(ld.astype(t.data.dtype)), Tensor(piv), Tensor(zero))


def ldl_solve(ld, pivots, b, hermitian=False):
    """`ldl_factor` 가 낸 분해로 푼다. `L y = b`, `D z = y`, `Lᵀ x = z` 세 번이다."""
    ld = _mat(ld, "ldl_solve")
    packed = _np.asarray(ld.data, dtype=_np.float64)
    rhs = _np.asarray(_wrap(b).data, dtype=_np.float64)
    n = packed.shape[-1]
    flat_ld = packed.reshape(-1, n, n)
    single = rhs.ndim == 2
    flat_b = rhs.reshape(-1, n, rhs.shape[-1]) if single else rhs.reshape(-1, n, rhs.shape[-1])
    outs = []
    for i in range(flat_ld.shape[0]):
        low = _np.tril(flat_ld[i], -1) + _np.eye(n)
        diag = _np.diagonal(flat_ld[i]).copy()
        y = _np.linalg.solve(low, flat_b[i])
        outs.append(_np.linalg.solve(low.T, y / diag[:, None]))
    got = _np.stack(outs).reshape(rhs.shape)
    return Tensor(got.astype(_wrap(b).data.dtype))


def geqrf(t):
    """QR 을 **반사자 꼴로** 낸다. `householder_product` 가 그것으로 `Q` 를 세운다.

    LAPACK 의 두 단계를 그대로 흉내낸다 — `geqrf` 가 반사자를 담고, 그것을 `Q` 로
    펴는 것은 따로다. `Q` 를 안 만들고 곱하기만 하는 코드가 있어서 갈라 둔 것이다.
    """
    t = _mat(t, "geqrf", square=False)
    a = _np.asarray(t.data, dtype=_np.float64)
    m, n = a.shape[-2], a.shape[-1]
    flat = a.reshape(-1, m, n).copy()
    taus = _np.zeros((flat.shape[0], min(m, n)))
    for b in range(flat.shape[0]):
        mat = flat[b]
        for j in range(min(m, n)):
            x = mat[j:, j].copy()
            # **대각 아래가 전부 0 이면 반사를 안 한다** — LAPACK 의 `dlarfg` 가
            # 그 자리에서 `tau = 0` 을 놓고 값을 그대로 둔다. 정사각의 마지막 열이
            # 늘 그 자리인데, 거기서 부호를 뒤집었더니 `Q` 의 마지막 열이 통째로
            # 반대가 됐다. 직사각으로만 물으면 그 열이 안 나와서 안 보인다.
            if _np.linalg.norm(x[1:]) == 0:
                continue
            norm = _np.linalg.norm(x)
            alpha = x[0]
            beta = -_np.sign(alpha) * norm if alpha != 0 else -norm
            tau = (beta - alpha) / beta
            v = x / (alpha - beta)
            v[0] = 1.0
            mat[j:, j:] -= tau * _np.outer(v, v @ mat[j:, j:])
            mat[j, j] = beta
            mat[j + 1:, j] = v[1:]
            taus[b, j] = tau
    return _Geqrf(Tensor(flat.reshape(a.shape).astype(t.data.dtype)),
                  Tensor(taus.reshape(a.shape[:-2] + (min(m, n),)).astype(t.data.dtype)))


def householder_product(t, tau):
    """반사자들을 곱해 `Q` 를 세운다. `geqrf` 의 짝이다.

    `H_i = I − τ_i v_i v_iᵀ` 를 차례로 곱한다. `v_i` 는 대각이 1 이고 그 아래가
    `A[i+1:, i]` 다 — 대각의 1 은 **저장 안 하는 약속**이라, 그 자리를 읽어 쓰면
    분해가 담아 둔 `R` 을 반사자로 착각한다.
    """
    t = _mat(t, "householder_product", square=False)
    a = _np.asarray(t.data, dtype=_np.float64)
    taus = _np.asarray(_wrap(tau).data, dtype=_np.float64)
    m, n = a.shape[-2], a.shape[-1]
    flat = a.reshape(-1, m, n)
    flat_tau = taus.reshape(-1, taus.shape[-1])
    outs = []
    for b in range(flat.shape[0]):
        q = _np.eye(m)
        for j in range(flat_tau.shape[1] - 1, -1, -1):
            v = _np.zeros(m)
            v[j] = 1.0
            v[j + 1:] = flat[b][j + 1:, j]
            q -= flat_tau[b, j] * _np.outer(v, v @ q)
        outs.append(q[:, :n])
    got = _np.stack(outs).reshape(a.shape[:-2] + (m, n))
    return Tensor(got.astype(t.data.dtype))


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


def _expm_raw(a):
    """스케일링·제곱 + 테일러. 배정도로 센다."""
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
    return out


def matrix_exp(t):
    """행렬 지수 `e^A`. **스케일링과 제곱으로 간다.**

    테일러만으로는 큰 행렬에서 안 모인다 — `A*5` 의 답이 4.8e+10 인데, 그 자리에서는
    항이 커지는 쪽이 먼저 넘친다. `A/2^s` 의 1-노름을 0.5 아래로 낮춰 급수를 태운 뒤
    `s` 번 제곱하면 같은 답이 안전하게 나온다(`e^A = (e^{A/2^s})^{2^s}`).

    **기울기는 자기 자신으로 구한다.** `e^A` 의 프레셰 도함수에는 이런 항등식이 있다:

        expm([[Aᵀ, Ḡ], [0, Aᵀ]]) 의 오른쪽 위 블록 = Ā

    근사가 아니라 항등식이다. 그래서 순방향에 쓴 급수를 **그대로 다시 부르면**
    기울기가 따라온다 — 미분식을 새로 유도해 적을 자리가 없다. 그것이 이 방법을
    고른 이유다. 유도한 식은 틀릴 수 있고 틀리면 조용하다.
    """
    x = _mat(t, "matrix_exp")
    a = x.data.astype(_np.float64)
    n = a.shape[-1]

    def back(g):
        gg = _np.asarray(g, dtype=_np.float64)
        at = _T(a)
        block = _np.zeros(a.shape[:-2] + (2 * n, 2 * n), dtype=_np.float64)
        block[..., :n, :n] = at
        block[..., :n, n:] = gg
        block[..., n:, n:] = at
        return (_expm_raw(block)[..., :n, n:].astype(x.data.dtype),)

    return t._make(_expm_raw(a).astype(x.data.dtype), (t,), back, "MatrixExpBackward0")


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
    lu_factor_ex = staticmethod(lu_factor_ex)
    lu_solve = staticmethod(lu_solve)
    ldl_factor = staticmethod(ldl_factor)
    ldl_factor_ex = staticmethod(ldl_factor_ex)
    ldl_solve = staticmethod(ldl_solve)
    householder_product = staticmethod(householder_product)
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


class _Fft(_Namespace):
    """`torch.fft`. 몸통은 `borch/_fft.py` 에 있다 — 이 파일이 이미 크고, 그쪽은
    `_tensor` 말고는 아무것도 안 들여와서 따로 설 수 있다."""

    fft = staticmethod(_fft_fft)
    ifft = staticmethod(_fft_ifft)
    rfft = staticmethod(_fft_rfft)
    irfft = staticmethod(_fft_irfft)
    fftfreq = staticmethod(_fft_fftfreq)
    rfftfreq = staticmethod(_fft_rfftfreq)
    fftshift = staticmethod(_fft_fftshift)
    ifftshift = staticmethod(_fft_ifftshift)


fft = _Fft()


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


# ── 제자리 활성과 `upsample` 별칭 ────────────────────────────────────────────
#
# torch 의 `F.relu_(x)` 는 `x` 를 **제 버퍼에서** 고친다. 학습 루프에서 중간 텐서를
# 안 만들려고 쓰는 자리이고, 교재 코드에서는 `nn.ReLU(inplace=True)` 와 짝이다.
#
# **계산은 밑줄 없는 쪽이 한다.** 여기서는 그 결과를 제 버퍼에 되쓰기만 한다 —
# 같은 식을 두 벌로 두면 언젠가 갈리고, 값이 그럴듯해서 안 보인다.

_FUNCTIONAL_INPLACE = ("relu", "celu", "elu", "selu", "hardtanh", "leaky_relu",
                       "threshold", "rrelu")


def _make_functional_inplace(name):
    fn = globals()[name]

    def call(x, *args, **kw):
        x = _wrap(x)
        return x._inplace(lambda: fn(x, *args, **kw), name + "_")

    call.__name__ = name + "_"
    call.__doc__ = (f"`F.{name}` 을 제자리로. 계산은 그쪽이 하고 여기서는 제 버퍼에 "
                    "되쓴다. 기울기가 켜진 잎은 torch 처럼 거절한다.")
    return call


for _nm in _FUNCTIONAL_INPLACE:
    globals()[_nm + "_"] = _make_functional_inplace(_nm)


# ── 공간 변환기 ────────────────────────────────────────────────────────────
#
# `affine_grid` 가 "출력의 이 칸은 입력의 어디를 보는가" 를 적은 격자를 만들고,
# `grid_sample` 이 그 자리에서 값을 떠 온다. 둘이 짝이고, 사이에 놓인 `theta` 가
# 학습된다 — 모델이 스스로 자르고 돌리고 확대하는 법을 배우는 구조다.
#
# **격자 좌표를 미분 가능한 텐서로 둔다.** 그러면 입력 쪽 기울기와 격자(따라서
# `theta`) 쪽 기울기가 둘 다 저절로 나온다. 자리 번호(내림한 정수)만 상수다 —
# torch 도 거기서는 안 흘린다.

def _grid_base(n, align_corners):
    """`[-1, 1]` 위의 표본 자리. **`align_corners` 가 이것을 바꾼다.**

    참이면 양 끝을 못 박고(`-1`, `1`) 그 사이를 고르게 나눈다. 거짓이면 칸의
    **가운데**를 잡는다(`(2i+1)/n − 1`) — 끝 칸의 절반이 밖으로 나간다.
    `interpolate` 와 같은 갈림이고, 값이 안쪽에서는 비슷해 눈으로는 안 갈린다.
    """
    if align_corners:
        return (_np.linspace(-1.0, 1.0, n, dtype=_np.float32) if n > 1
                else _np.zeros(1, dtype=_np.float32))
    return ((2 * _np.arange(n, dtype=_np.float32) + 1) / n - 1).astype(_np.float32)


def affine_grid(theta, size, align_corners=False):
    """`theta` 가 그리는 표본 격자. `(N, 2, 3)` 을 받아 `(N, H, W, 2)` 를 낸다.

    마지막 축은 **`(x, y)` 순서다** — 모양의 `(H, W)` 와 뒤집혀 있다. 뒤집어 적으면
    가로세로가 같은 정사각 입력에서는 답이 같아서 안 보이고, 직사각에서 드러난다.
    """
    theta = _wrap(theta)
    n, _, h, w = tuple(int(v) for v in size)
    xs = _grid_base(w, align_corners)
    ys = _grid_base(h, align_corners)
    # 균질좌표 `(x, y, 1)` — 이동까지 한 번의 곱으로 끝낸다.
    base = _np.stack([_np.broadcast_to(xs[None, :], (h, w)),
                      _np.broadcast_to(ys[:, None], (h, w)),
                      _np.ones((h, w), dtype=_np.float32)], axis=-1)
    flat = Tensor(base.reshape(h * w, 3).astype(_np.float32))
    out = matmul(flat, theta.transpose(-2, -1))      # (N, H·W, 2)
    return out.reshape(n, h, w, 2)


def _grid_denorm(g, n, align_corners):
    """`[-1, 1]` 을 입력의 칸 번호로 되돌린다. `_grid_base` 의 반대다."""
    if align_corners:
        return (g + 1.0) * ((n - 1) / 2.0)
    return ((g + 1.0) * n - 1.0) * 0.5


def _grid_reflect(v, n, align_corners):
    """범위 밖을 **되접는다.** 되접는 구간이 `align_corners` 로 갈린다.

    참이면 `[0, n−1]`, 거짓이면 `[−0.5, n−0.5]` 다(실측). 되접은 뒤 한 번 더 자른다 —
    거짓 쪽 구간이 실제 칸 밖까지 걸쳐 있기 때문이다.
    """
    lo, hi = (0.0, n - 1.0) if align_corners else (-0.5, n - 0.5)
    if hi <= lo:
        return v * 0.0 + lo
    span = 2.0 * (hi - lo)
    t = remainder(v - lo, span)
    return clamp(minimum(t, span - t) + lo, 0.0, n - 1.0)


def grid_sample(x, grid, mode="bilinear", padding_mode="zeros",
                align_corners=False):
    """격자가 가리키는 자리에서 값을 떠 온다. `affine_grid` 의 짝이다.

    **자리 번호는 상수, 무게는 텐서다.** 내림한 정수는 미분이 없고 그 나머지가
    무게가 되므로, 무게만 그래프에 두면 입력과 격자 양쪽으로 기울기가 흐른다 —
    공간 변환기가 `theta` 를 배우는 길이 그것이다.
    """
    x, grid = _wrap(x), _wrap(grid)
    n, c, h, w = x.data.shape
    oh, ow = grid.data.shape[1], grid.data.shape[2]
    gx = grid[:, :, :, 0]
    gy = grid[:, :, :, 1]
    sx = _grid_denorm(gx, w, align_corners)
    sy = _grid_denorm(gy, h, align_corners)
    if padding_mode == "border":
        sx, sy = clamp(sx, 0.0, w - 1.0), clamp(sy, 0.0, h - 1.0)
    elif padding_mode == "reflection":
        sx = _grid_reflect(sx, w, align_corners)
        sy = _grid_reflect(sy, h, align_corners)
    elif padding_mode != "zeros":
        _unsupported(f"grid_sample(padding_mode={padding_mode!r})")

    flat = x.reshape(-1)
    batch = _np.arange(n).reshape(n, 1, 1, 1)
    chan = _np.arange(c).reshape(1, c, 1, 1)

    def pick(iy, ix):
        """한 모서리를 떠 온다. **범위 밖은 0 으로 두되 번호는 잘라서 넘긴다** —
        안 자르면 엉뚱한 자리를 읽는다."""
        inside = ((ix >= 0) & (ix < w) & (iy >= 0) & (iy < h))
        cy = _np.clip(iy, 0, h - 1)
        cx = _np.clip(ix, 0, w - 1)
        idx = (((batch * c + chan) * h + cy[:, None]) * w + cx[:, None])
        got = take(flat, Tensor(idx.reshape(-1).astype(_np.int64)))
        got = got.reshape(n, c, oh, ow)
        return got * Tensor(inside[:, None].astype(x.data.dtype))

    if mode == "nearest":
        # torch 는 반올림한다. 값만 나오고 무게가 없으므로 격자로는 안 흐른다.
        return pick(_np.rint(sy.data).astype(int), _np.rint(sx.data).astype(int))
    if mode != "bilinear":
        _unsupported(f"grid_sample(mode={mode!r}) — 겹선형과 최근접만 있습니다")

    x0 = _np.floor(sx.data).astype(int)
    y0 = _np.floor(sy.data).astype(int)
    wx = (sx - Tensor(x0.astype(x.data.dtype))).reshape(n, 1, oh, ow)
    wy = (sy - Tensor(y0.astype(x.data.dtype))).reshape(n, 1, oh, ow)
    one = 1.0
    return (pick(y0, x0) * (one - wy) * (one - wx)
            + pick(y0, x0 + 1) * (one - wy) * wx
            + pick(y0 + 1, x0) * wy * (one - wx)
            + pick(y0 + 1, x0 + 1) * wy * wx)


def batch_norm(x, running_mean=None, running_var=None, weight=None, bias=None,
               training=False, momentum=0.1, eps=1e-5):
    """`BatchNorm*d` 의 함수 꼴. **층이 이것을 부른다** — 식을 한 벌만 둔다.

    **`training` 이면 running 통계를 제자리에서 고친다.** torch 가 그렇다 — 넘긴
    텐서가 갱신되어 돌아온다. 새것을 돌려주면 부르는 쪽의 버퍼가 안 움직이고,
    학습은 도는데 평가 모드의 값만 틀린다.

    **분산을 두 가지로 쓴다.** 정규화는 편향(ddof=0), `running_var` 갱신은
    비편향(ddof=1) 이다. 둘 다 편향으로 두면 값이 2.6% 어긋난다 — 이 저장소가
    오래 겪은 자리이고 그래서 여기 적어 둔다.
    """
    x = _wrap(x)
    rank = x.data.ndim
    shape = (1, -1) + (1,) * (rank - 2)
    reduced = tuple(i for i in range(rank) if i != 1)

    def _raw(v):
        return v.data if isinstance(v, Tensor) else v

    if training:
        # 평균·분산을 **그래프 안에서** 낸다. numpy 로 빼서 상수처럼 쓰면 x → mean → y
        # 로 흐르는 길이 끊겨 기울기가 틀리고, weight 에는 아예 안 간다.
        mean = x.mean(dim=0)
        for _ in range(rank - 2):
            mean = mean.mean(dim=1)
        centered = x - mean.reshape(shape)
        var = (centered * centered).mean(dim=0)
        for _ in range(rank - 2):
            var = var.mean(dim=1)
        if running_mean is not None:
            with no_grad():
                unbiased = x.data.var(axis=reduced, ddof=1)
                _raw(running_mean)[...] = ((1 - momentum) * _raw(running_mean)
                                           + momentum * mean.data)
                _raw(running_var)[...] = ((1 - momentum) * _raw(running_var)
                                          + momentum * unbiased)
        normed = centered / (var.reshape(shape) + eps) ** 0.5
    else:
        rm = _np.asarray(_raw(running_mean)).reshape(shape)
        rv = _np.sqrt(_np.asarray(_raw(running_var)) + eps).reshape(shape)
        normed = (x - Tensor(rm)) / Tensor(rv)
    if weight is not None:
        normed = normed * _wrap(weight).reshape(shape)
    if bias is not None:
        normed = normed + _wrap(bias).reshape(shape)
    return normed


def embedding_bag(idx, weight, offsets=None, mode="mean", per_sample_weights=None,
                  **_):
    """가방마다 한 줄. 표에서 골라 **합치는 것**까지가 한 함수다.

    `offsets` 를 주면 1 차원 번호 줄을 가방으로 자른다 — 가방 길이가 제각각인 자리다.
    `per_sample_weights` 는 torch 에서 `mode='sum'` 일 때만 쓴다.
    """
    picked = embedding(idx, weight)
    if per_sample_weights is not None:
        picked = picked * _wrap(per_sample_weights).reshape(
            *_wrap(per_sample_weights).data.shape, 1)

    def squash(part, dim):
        if mode == "sum":
            return part.sum(dim=dim)
        if mode == "max":
            return amax(part, dim=dim)
        return part.mean(dim=dim)

    if offsets is None:
        return squash(picked, dim=1)
    bounds = [int(v) for v in _wrap(offsets).data] + [int(_wrap(idx).data.size)]
    return stack([squash(picked[bounds[i]:bounds[i + 1]], dim=0)
                  for i in range(len(bounds) - 1)], dim=0)


def gumbel_softmax(logits, tau=1.0, hard=False, eps=1e-10, dim=-1):
    """무작위로 하나를 고르되 **미분이 흐르게** 고른다.

    범주 하나를 뽑는 것은 미분이 안 되는 일이라, Gumbel 잡음을 더해 `softmax` 로
    부드럽게 만든다. `tau` 가 작을수록 한쪽으로 몰린다.

    `hard=True` 면 답은 0/1 이지만 **기울기는 부드러운 쪽 것을 쓴다** —
    `hard - soft.detach() + soft` 라는 흔한 수법이고, 값은 hard 이고 미분은 soft 다.
    그 둘을 갈라 두지 않으면 이 함수의 뜻이 없어진다.
    """
    logits = _wrap(logits)
    u = _rng.random(logits.data.shape).astype(logits.data.dtype)
    gumbel = -_np.log(-_np.log(u + eps) + eps)
    soft = softmax((logits + Tensor(gumbel)) / tau, dim=dim)
    if not hard:
        return soft
    at = _np.argmax(soft.data, axis=dim)
    onehot = _np.zeros_like(soft.data)
    _np.put_along_axis(onehot, _np.expand_dims(at, dim), 1.0, axis=dim)
    return Tensor(onehot) - soft.detach() + soft


def upsample(x, size=None, scale_factor=None, mode="nearest", align_corners=None):
    """`interpolate` 의 옛 이름. torch 가 폐기 경고를 내면서도 계속 받는다."""
    return interpolate(x, size, scale_factor, mode, align_corners)


def upsample_nearest(x, size=None, scale_factor=None):
    return interpolate(x, size, scale_factor, mode="nearest")


def upsample_bilinear(x, size=None, scale_factor=None):
    """**`align_corners=True` 다.** `interpolate(mode='bilinear')` 의 기본값은 거짓이라,
    이름만 보고 별명으로 두면 가장자리가 어긋난다 — 안쪽은 비슷해서 눈으로는 안 갈린다."""
    return interpolate(x, size, scale_factor, mode="bilinear", align_corners=True)


# ── 비트 연산과 정수 수학 ───────────────────────────────────────────────────
#
# **`bool` 에서는 논리 연산이 된다.** torch 가 dtype 을 보고 가른다 —
# `bitwise_and(참거짓)` 은 `logical_and` 이고 `bitwise_not(참거짓)` 은 `logical_not` 이다.
# 정수로만 물으면 그 갈래가 통째로 안 돌아간다.
#
# 기울기는 없다. 비트는 계단이라 흘릴 것이 없고, torch 도 정수 dtype 에는 기울기를
# 안 만든다.

def _bitwise(name, op, bool_op=None):
    def call(a, b):
        a, b = _wrap(a), _wrap(b)
        if a.data.dtype.kind == "b" and bool_op is not None:
            return Tensor(bool_op(a.data, b.data))
        return Tensor(op(a.data.astype(_np.int64), b.data.astype(_np.int64)))
    call.__name__ = name
    return call


bitwise_and = _bitwise("bitwise_and", _np.bitwise_and, _np.logical_and)
bitwise_or = _bitwise("bitwise_or", _np.bitwise_or, _np.logical_or)
bitwise_xor = _bitwise("bitwise_xor", _np.bitwise_xor, _np.logical_xor)
bitwise_left_shift = _bitwise("bitwise_left_shift", _np.left_shift)
bitwise_right_shift = _bitwise("bitwise_right_shift", _np.right_shift)


def bitwise_not(x):
    x = _wrap(x)
    if x.data.dtype.kind == "b":
        return Tensor(_np.logical_not(x.data))
    return Tensor(_np.bitwise_not(x.data.astype(_np.int64)))


def gcd(a, b):
    a, b = _wrap(a), _wrap(b)
    return Tensor(_np.gcd(a.data.astype(_np.int64), b.data.astype(_np.int64)))


def lcm(a, b):
    a, b = _wrap(a), _wrap(b)
    return Tensor(_np.lcm(a.data.astype(_np.int64), b.data.astype(_np.int64)))


def gcd_(a, b):
    a = _wrap(a)
    return a._inplace(lambda: gcd(a, b), "gcd_")


def lcm_(a, b):
    a = _wrap(a)
    return a._inplace(lambda: lcm(a, b), "lcm_")


def nextafter(a, b):
    """`a` 에서 `b` 쪽으로 **표현 가능한 다음 수.** 한 ulp 만큼만 움직인다."""
    a, b = _wrap(a), _wrap(b)
    return a._make(_np.nextafter(a.data, b.data), (a,), lambda g: (g,),
                   "NextafterBackward0")


def frexp(x):
    """`x = 가수 × 2^지수`. **지수는 int32 다** — torch 가 그렇다(실측)."""
    x = _wrap(x)
    mantissa, exponent = _np.frexp(x.data)
    return _Frexp(Tensor(mantissa.astype(x.data.dtype)),
                  Tensor(exponent.astype(_np.int32)))


def logcumsumexp(x, dim):
    """누적 `logsumexp`. **넘치지 않게** 센다 — 큰 값을 빼고 더한 뒤 되돌린다."""
    x = _wrap(x)
    # **`logsumexp` 는 정수를 받는데 이쪽은 안 받는다**(실측). 규칙이 아니라 torch 의
    # 커널 구멍이지만, 여기서 값을 내주면 그 코드가 진짜 torch 에서 깨진다.
    _refuses_nonfloat_kernel(x.data, "logcumsumexp", "logcumsumexp_out_cpu")
    data = x.data
    big = _np.max(data, axis=dim, keepdims=True)
    shifted = _np.exp(data - big)
    total = _np.cumsum(shifted, axis=dim)
    out = _np.log(total) + big
    soft = shifted / total          # 각 자리가 누적합에서 차지하는 몫

    def back(g):
        # 뒤에서부터 쌓인다 — 자리 `i` 는 `i` 이후의 모든 누적항에 들어간다.
        gg = _np.asarray(g)
        flipped = _np.flip(_np.cumsum(_np.flip(gg / total, axis=dim), axis=dim),
                           axis=dim)
        return (flipped * shifted,)

    del soft
    return x._make(out, (x,), back, "LogcumsumexpBackward0")


def clamp_max(x, value):
    return clamp(x, None, value)


def clamp_min(x, value):
    return clamp(x, value, None)


def clamp_max_(x, value):
    x = _wrap(x)
    return x._inplace(lambda: clamp(x, None, value), "clamp_max_")


def clamp_min_(x, value):
    x = _wrap(x)
    return x._inplace(lambda: clamp(x, value, None), "clamp_min_")


def arctan2(a, b):
    return atan2(a, b)


def fill(x, value):
    """**제자리가 아니다.** `fill_` 과 이름이 한 글자 다르고 하는 일이 다르다 —
    이쪽은 새 텐서를 내고 원본은 그대로다(실측)."""
    x = _wrap(x)
    return Tensor(_np.full_like(x.data, value))


def detach_(x):
    """**같은 텐서**에서 그래프를 끊는다. `detach()` 는 새것을 내고 이쪽은 제자리다."""
    x = _wrap(x)
    x.requires_grad = False
    x._parents = ()
    x._backward = None
    return x


def _i1(x):
    """1 차 변형 베셀 함수 — `i0` 의 도함수다. numpy 가 `i0` 만 주므로 급수로 짠다.

    급수는 `i1(x) = Σ (x/2)^(2k+1) / (k! (k+1)!)` 이고, 항을 앞 항에 곱해 이어 가면
    계승이 넘치지 않는다. **항이 전부 양수라 서로 지우지 않으므로** 자릿수를 잃는
    자리가 없다 — 부호는 홀함수라 마지막에 붙인다.

    근사가 아니라 수렴이다. torch 와 [-30, 30] 을 촘촘히 대조해 float32 안에서
    상대오차 1e-6 아래임을 확인했다(`tests/test_bessel.py`).
    """
    a = _np.abs(_np.asarray(x, dtype=_np.float64))
    half = a / 2.0
    term = half.copy()                     # k=0 항
    total = term.copy()
    for k in _builtin_range(1, 400):
        term = term * (half * half) / (k * (k + 1.0))
        total = total + term
        if _np.all(term <= _np.abs(total) * 1e-17):
            break
    return _np.sign(_np.asarray(x, dtype=_np.float64)) * total


def i0(x):
    """0 차 변형 베셀 함수. `kaiser_window` 가 이것 위에 선다."""
    x = _wrap(x)
    data = _np.asarray(x.data)
    # 도함수는 `i1` 이다. 여기에는 `Tensor(...)` 만 있어서 그래프가 조용히 끊겨
    # 있었고, `backward()` 를 부르기 전까지는 값 검사로 안 드러났다.
    return x._make(_np.i0(data).astype(x.data.dtype), (x,),
                   lambda g: (_np.asarray(g) * _i1(data),), "I0Backward0")


def i0_(x):
    x = _wrap(x)
    return x._inplace(lambda: i0(x), "i0_")


def mvlgamma(x, p):
    """다변량 로그감마. `log Γ_p(x) = p(p−1)/4 · log π + Σ log Γ(x + (1−i)/2)`."""
    x = _wrap(x)
    out = _np.full_like(x.data, p * (p - 1) / 4.0 * _math.log(_math.pi))
    for i in range(1, p + 1):
        out = out + _np.asarray(lgamma(x + (1 - i) / 2.0).data)
    return Tensor(out.astype(x.data.dtype))


# ── 창 함수 ─────────────────────────────────────────────────────────────────
#
# **`periodic` 이 기본이고 그것이 길이를 하나 늘린다.** torch 는 참이면 `N+1` 짜리
# 대칭 창을 만들어 마지막을 버린다(실측: `hann_window(5)` 가 대칭 6 의 앞 다섯과
# 정확히 같다). 거짓으로만 물으면 그 규칙이 안 드러난다.

def _window(n, periodic, shape):
    if n <= 0:
        return Tensor(_np.zeros(0, dtype=_DEFAULT_DTYPE))
    if n == 1:
        return Tensor(_np.ones(1, dtype=_DEFAULT_DTYPE))
    total = n + 1 if periodic else n
    k = _np.arange(total, dtype=_np.float64)
    return Tensor(shape(k, total)[:n].astype(_DEFAULT_DTYPE))


def bartlett_window(window_length, periodic=True, **kw):
    return _window(window_length, periodic,
                   lambda k, n: 1.0 - _np.abs(2.0 * k / (n - 1) - 1.0))


def hann_window(window_length, periodic=True, **kw):
    return _window(window_length, periodic,
                   lambda k, n: 0.5 - 0.5 * _np.cos(2 * _np.pi * k / (n - 1)))


def hamming_window(window_length, periodic=True, alpha=0.54, beta=0.46, **kw):
    return _window(window_length, periodic,
                   lambda k, n: alpha - beta * _np.cos(2 * _np.pi * k / (n - 1)))


def blackman_window(window_length, periodic=True, **kw):
    def shape(k, n):
        t = 2 * _np.pi * k / (n - 1)
        return 0.42 - 0.5 * _np.cos(t) + 0.08 * _np.cos(2 * t)
    return _window(window_length, periodic, shape)


def kaiser_window(window_length, periodic=True, beta=12.0, **kw):
    def shape(k, n):
        half = (n - 1) / 2.0
        return _np.i0(beta * _np.sqrt(1.0 - ((k - half) / half) ** 2)) / _np.i0(beta)
    return _window(window_length, periodic, shape)


# ── torch 최상위에만 있는 이름들 ────────────────────────────────────────────
#
# torch 는 `nn.functional` 의 것을 최상위에도 두는데, **서명이 같지 않다.** 최상위
# 쪽은 날 ATen 연산이라 인자 순서가 다르고 열거형이 정수다. 같은 계산인데 부르는
# 법이 다른 것이라, 계산은 한 벌만 두고 여기서 자리만 옮긴다.

def nan_to_num_(x, nan=0.0, posinf=None, neginf=None):
    x = _wrap(x)
    return x._inplace(lambda: nan_to_num(x, nan, posinf, neginf), "nan_to_num_")


def dropout_(x, p=0.5, train=True):
    x = _wrap(x)
    return x._inplace(lambda: dropout(x, p, train), "dropout_")


def feature_dropout(x, p=0.5, train=True):
    """**채널째 떨군다** — `F.dropout2d` 와 같은 계산이다(실측). 최상위에만 있는 이름이다."""
    return dropout2d(x, p, train)


def feature_dropout_(x, p=0.5, train=True):
    x = _wrap(x)
    return x._inplace(lambda: dropout2d(x, p, train), "feature_dropout_")


def alpha_dropout_(x, p=0.5, train=True):
    x = _wrap(x)
    return x._inplace(lambda: alpha_dropout(x, p, train), "alpha_dropout_")


def feature_alpha_dropout_(x, p=0.5, train=True):
    x = _wrap(x)
    return x._inplace(lambda: feature_alpha_dropout(x, p, train),
                      "feature_alpha_dropout_")


def batch_norm_aten(x, weight, bias, running_mean, running_var, training,
                    momentum, eps, cudnn_enabled=False):
    """**`F.batch_norm` 과 인자 순서가 다르다.** 여기서는 가중치가 통계보다 **앞**이다.

    같은 계산인데 자리가 뒤바뀐 것이라, 그대로 넘기면 가중치를 평균으로 쓴다 —
    예외가 아니라 그럴듯하게 다른 값이다.
    """
    return batch_norm(x, running_mean, running_var, weight, bias, training,
                      momentum, eps)


def grid_sampler(x, grid, interpolation_mode=0, padding_mode=0,
                 align_corners=False):
    """**열거형이 정수다.** 0·1 이 `bilinear`·`nearest` 이고, 채우기는 0·1·2 가
    `zeros`·`border`·`reflection` 이다. 이름으로 받는 쪽은 `F.grid_sample` 이다."""
    modes = ("bilinear", "nearest", "bicubic")
    pads = ("zeros", "border", "reflection")
    return grid_sample(x, grid, modes[int(interpolation_mode)],
                       pads[int(padding_mode)], align_corners)


def ctc_loss_aten(log_probs, targets, input_lengths, target_lengths, blank=0,
                  reduction=1, zero_infinity=False):
    """**`reduction` 이 정수다** — 0·1·2 가 `none`·`mean`·`sum` 이다.

    이름으로 받는 쪽은 `F.ctc_loss` 이고, 그쪽 기본값은 `"mean"` 이다. 여기 기본값
    `1` 이 그것과 같은 자리를 가리킨다.
    """
    kinds = ("none", "mean", "sum")
    return ctc_loss(log_probs, targets, input_lengths, target_lengths, blank,
                    kinds[int(reduction)], zero_infinity)


# ── 모양·색인 ───────────────────────────────────────────────────────────────
#
# **`as_strided` 는 torch 에서 뷰다.** 우리는 사본을 낸다.
#
# torch 는 저장소 하나를 여러 틀로 보는 구조라 `as_strided` 결과에 쓰면 원본이 바뀐다.
# borch.ts 의 텐서는 GPU 버퍼를 **하나씩 가지므로** 그 뷰가 표현이 안 되고, 여기서만
# 진짜 뷰를 내면 세 구현이 갈린다 — 값으로는 안 보이고 **쓸 때만** 보이는 갈림이라
# 제일 나쁜 종류다. 셋 다 사본으로 맞춘다.
#
# 읽기만 하는 쓰임(창 내기, 대각선 훑기)은 그대로 돌고, 뷰에 쓰는 코드는 torch 에서도
# 드물다. 대신 `as_strided_scatter` 가 "그 자리에 써 넣기" 를 제대로 한다.

def _strided_flat(size, stride, offset):
    """걸음이 가리키는 **평평한 번호표**를 만든다. 모양은 `size` 다."""
    size = tuple(int(s) for s in size)
    stride = tuple(int(s) for s in stride)
    flat = _np.full(size, int(offset), dtype=_np.int64)
    for axis, step in enumerate(stride):
        shape = [1] * len(size)
        shape[axis] = size[axis]
        flat = flat + _np.arange(size[axis], dtype=_np.int64).reshape(shape) * step
    return flat


def as_strided(t, size, stride, storage_offset=0):
    """평평한 저장소를 **다른 걸음으로** 읽는다. 겹쳐도 되고 건너뛰어도 된다."""
    t = _wrap(t)
    flat = _strided_flat(size, stride, storage_offset)
    out = t.data.reshape(-1)[flat]

    def back(g):
        # 겹치는 자리로는 **쌓인다** — 한 칸을 두 번 읽었으면 기울기도 두 번 온다.
        acc = _np.zeros(t.data.size, dtype=_np.asarray(g).dtype)
        _np.add.at(acc, flat.reshape(-1), _np.asarray(g).reshape(-1))
        return (acc.reshape(t.data.shape),)

    return t._make(out, (t,), back, "AsStridedBackward0")


def as_strided_(t, size, stride, storage_offset=0):
    t = _wrap(t)
    return t._inplace(lambda: as_strided(t, size, stride, storage_offset),
                      "as_strided_")


def _marks(shape):
    """`shape` 짜리 판의 **평평한 번호표.** 어느 칸이 저장소 어디인가를 담는다."""
    return _np.arange(int(_np.prod(shape)) if shape else 1,
                      dtype=_np.int64).reshape(shape)


def _scatter_into(t, src, spots, name):
    """`t` 의 사본에서 `spots` 가 가리키는 **평평한 자리**에 `src` 를 넣는다.

    `select_scatter`·`slice_scatter`·`diagonal_scatter`·`as_strided_scatter` 가
    전부 이 꼴이고, 다른 것은 **어느 자리인가**뿐이다.

    **쓰기와 되읽기가 같은 번호표를 쓴다.** 자리를 두 벌로 적으면 순방향은 맞는데
    기울기만 어긋나는 자리가 생기고, 그 어긋남은 값이 그럴듯해서 안 보인다 —
    대각선처럼 축 순서가 뒤집히는 자리가 특히 그렇다.
    """
    t, src = _wrap(t), _wrap(src)
    flat = _np.asarray(spots).reshape(-1)
    out = t.data.copy().reshape(-1)
    out[flat] = _np.broadcast_to(_np.asarray(src.data),
                                 _np.asarray(spots).shape).reshape(-1)
    out = out.reshape(t.data.shape)

    def back(g):
        g = _np.asarray(g)
        keep = _np.ones(t.data.size, dtype=g.dtype)
        keep[flat] = 0.0
        into = g.reshape(-1)[flat].reshape(_np.asarray(spots).shape)
        return (g * keep.reshape(t.data.shape),
                into.reshape(src.data.shape))

    return t._make(out, (t, src), back, name)


def as_strided_scatter(t, src, size, stride, storage_offset=0):
    """`as_strided` 가 보던 자리에 **써 넣은 사본**을 낸다."""
    return _scatter_into(t, src, _strided_flat(size, stride, storage_offset),
                         "AsStridedScatterBackward0")


def select_scatter(t, src, dim, index):
    """`select` 가 꺼내던 한 장을 **갈아끼운 사본.**"""
    spots = _marks(_wrap(t).data.shape)[(slice(None),) * dim + (int(index),)]
    return _scatter_into(t, src, spots, "SelectScatterBackward0")


def slice_scatter(t, src, dim=0, start=None, end=None, step=1):
    """`x[..., start:end:step]` 자리를 **갈아끼운 사본.** `step` 이 요점이다 —
    1 로만 재면 건너뛰는 자리를 아무도 안 본다."""
    spots = _marks(_wrap(t).data.shape)[
        (slice(None),) * dim + (slice(start, end, step),)]
    return _scatter_into(t, src, spots, "SliceScatterBackward0")


def diagonal_scatter(t, src, offset=0, dim1=0, dim2=1):
    """대각선 자리를 **갈아끼운 사본.** `offset` 이 0 이 아니면 자리가 밀린다.

    자리를 `_np.diagonal` 로 뽑는다 — 그쪽이 **대각선 축을 맨 뒤로 보내는** 규약을
    갖고 있고 torch 도 같다. 손으로 색인을 짜면 배치 축이 있을 때 순서가 갈린다.
    """
    spots = _np.diagonal(_marks(_wrap(t).data.shape), offset=offset,
                         axis1=dim1, axis2=dim2)
    return _scatter_into(t, src, spots, "DiagonalScatterBackward0")


def diag_embed(t, offset=0, dim1=-2, dim2=-1):
    """마지막 축을 **대각선으로 펴서** 축을 하나 늘린다. `diagonal` 의 반대다."""
    t = _wrap(t)
    # `abs` 는 이 파일에서 **텐서 함수**다 — 파이썬 내장이 가려져 있다. `_np.abs` 가
    # 이 파일의 규칙이고, 그것을 잊으면 `'int' object has no attribute 'abs'` 로 온다.
    n = t.data.shape[-1] + int(_np.abs(offset))
    rank = t.data.ndim + 1
    d1, d2 = dim1 % rank, dim2 % rank
    shape = list(t.data.shape[:-1])
    for at in sorted((d1, d2)):
        shape.insert(at, n)
    out = _np.zeros(tuple(shape), dtype=t.data.dtype)
    spots = _np.diagonal(_marks(out.shape), offset=offset, axis1=d1, axis2=d2)
    out.reshape(-1)[_np.asarray(spots).reshape(-1)] = t.data.reshape(-1)

    def back(g):
        g = _np.asarray(g)
        return (g.reshape(-1)[_np.asarray(spots).reshape(-1)]
                .reshape(t.data.shape),)

    return t._make(out, (t,), back, "DiagEmbedBackward0")


def tensor_split(t, indices_or_sections, dim=0):
    """**나머지를 앞에서부터 나눠 갖는다.** 10 을 4 로 쪼개면 3·3·2·2 다(실측).

    `chunk` 와 다르다 — 그쪽은 앞을 크게 채우고 마지막이 남는 것을 받는다. 나눠
    떨어지는 크기로만 재면 두 함수가 같아 보인다.
    """
    t = _wrap(t)
    if isinstance(indices_or_sections, (list, tuple)):
        return tuple(_wrap(p) for p in
                     _split_at(t, list(indices_or_sections), dim))
    k = int(indices_or_sections)
    n = t.data.shape[dim]
    base, extra = divmod(n, k)
    cuts, at = [], 0
    for i in range(k - 1):
        at += base + (1 if i < extra else 0)
        cuts.append(at)
    return tuple(_wrap(p) for p in _split_at(t, cuts, dim))


def _split_at(t, cuts, dim):
    """자르는 **자리 목록**으로 쪼갠다. 각 조각이 기울기를 제 자리로 되돌린다."""
    out, prev = [], 0
    for stop in list(cuts) + [t.data.shape[dim]]:
        out.append(narrow(t, dim, prev, max(0, stop - prev)))
        prev = stop
    return out


def split_with_sizes(t, split_sizes, dim=0):
    """조각 **크기 목록**으로 쪼갠다. `split` 이 목록을 받는 꼴과 같은 것이다."""
    t = _wrap(t)
    cuts, at = [], 0
    for s in list(split_sizes)[:-1]:
        at += int(s)
        cuts.append(at)
    return tuple(_split_at(t, cuts, dim))


def unravel_index(indices, shape):
    """평평한 번호를 축별 번호로 푼다. **축마다 텐서 하나씩, 묶음으로 낸다**(실측)."""
    idx = _as_index(indices)
    return tuple(Tensor(part.astype(_np.int64))
                 for part in _np.unravel_index(idx, tuple(int(s) for s in shape)))


def unique_consecutive(t, return_inverse=False, return_counts=False, dim=None):
    """**이어진** 중복만 줄인다. `unique` 와 달리 정렬하지 않는다 — `[1,1,2,2,1]`
    이 `[1,2,1]` 이 된다(실측). 정렬된 입력으로만 재면 둘이 같아 보인다."""
    t = _wrap(t)
    data = t.data
    if dim is None:
        flat = data.reshape(-1)
        keep = _np.ones(flat.shape[0], dtype=bool)
        keep[1:] = flat[1:] != flat[:-1]
    else:
        moved = _np.moveaxis(data, dim, 0)
        rows = moved.reshape(moved.shape[0], -1)
        keep = _np.ones(rows.shape[0], dtype=bool)
        keep[1:] = _np.any(rows[1:] != rows[:-1], axis=1)
    starts = _np.flatnonzero(keep)
    groups = _np.cumsum(keep) - 1
    values = (Tensor(flat[keep]) if dim is None
              else Tensor(_np.moveaxis(_np.moveaxis(data, dim, 0)[keep], 0, dim)))
    if not return_inverse and not return_counts:
        return values
    out = [values]
    if return_inverse:
        shape = data.shape if dim is None else (data.shape[dim],)
        out.append(Tensor(groups.reshape(shape).astype(_np.int64)))
    if return_counts:
        edges = _np.append(starts, keep.shape[0])
        out.append(Tensor(_np.diff(edges).astype(_np.int64)))
    return tuple(out)


def masked_scatter(t, mask, source):
    """가면이 참인 자리에 `source` 를 **평평한 차례대로** 채운다.

    자리마다 어느 값이 오는지가 요점이다 — 참인 자리 수만큼 앞에서부터 가져온다.
    모양이 맞는 원천으로만 재면 그 차례가 안 드러난다.
    """
    t, source = _wrap(t), _wrap(source)
    m = _np.broadcast_to(_np.asarray(mask.data if isinstance(mask, Tensor)
                                     else mask).astype(bool), t.data.shape)
    out = t.data.copy()
    out[m] = _np.asarray(source.data).reshape(-1)[:int(m.sum())]

    def back(g):
        g = _np.asarray(g)
        into = _np.zeros(source.data.size, dtype=g.dtype)
        into[:int(m.sum())] = g[m]
        return (g * (~m), into.reshape(source.data.shape))

    return t._make(out, (t, source), back, "MaskedScatterBackward0")


def masked_scatter_(t, mask, source):
    t = _wrap(t)
    return t._inplace(lambda: masked_scatter(t, mask, source), "masked_scatter_")


def index_put(t, indices, values, accumulate=False):
    """축마다 번호 텐서를 하나씩 받아 그 자리에 넣는다.

    **번호가 겹칠 때 갈린다** — `accumulate` 면 쌓이고, 아니면 마지막에 쓴 것이
    남는다. 안 겹치는 번호로 재면 두 갈래가 같은 답을 낸다.
    """
    t, values = _wrap(t), _wrap(values)
    where = tuple(_as_index(i) for i in indices)
    out = t.data.copy()
    if accumulate:
        _np.add.at(out, where, _np.asarray(values.data))
    else:
        out[where] = _np.asarray(values.data)

    def back(g):
        g = _np.asarray(g)
        into = g[where]
        if accumulate:
            return (g, into)
        keep = _np.ones(t.data.shape, dtype=g.dtype)
        keep[where] = 0.0
        return (g * keep, into)

    return t._make(out, (t, values), back, "IndexPutBackward0")


def index_put_(t, indices, values, accumulate=False):
    t = _wrap(t)
    return t._inplace(lambda: index_put(t, indices, values, accumulate),
                      "index_put_")


def put(t, index, source, accumulate=False):
    """**평평하게 펴서** 번호대로 넣는다 — 축이라는 개념이 없다. `take` 의 반대다."""
    t, source = _wrap(t), _wrap(source)
    idx = _as_index(index).reshape(-1)
    out = t.data.copy().reshape(-1)
    if accumulate:
        _np.add.at(out, idx, _np.asarray(source.data).reshape(-1))
    else:
        out[idx] = _np.asarray(source.data).reshape(-1)
    out = out.reshape(t.data.shape)

    def back(g):
        g = _np.asarray(g)
        into = g.reshape(-1)[idx].reshape(source.data.shape)
        if accumulate:
            return (g, into)
        keep = _np.ones(t.data.size, dtype=g.dtype)
        keep[idx] = 0.0
        return (g * keep.reshape(t.data.shape), into)

    return t._make(out, (t, source), back, "PutBackward0")


# 줄이며 넣는 것들의 셈법. `include_self` 가 **원래 값을 첫 항으로 넣는가**다.
_REDUCE_OPS = {
    "sum": (0.0, _np.add),
    "prod": (1.0, _np.multiply),
    "amax": (-_np.inf, _np.maximum),
    "amin": (_np.inf, _np.minimum),
}


def _reduce_into(out, where, values, reduce, include_self):
    """`out[where]` 에 `values` 를 `reduce` 로 합친다. 자리가 겹치면 쌓인다."""
    if reduce == "mean":
        total = _np.zeros(out.shape, dtype=out.dtype)
        count = _np.zeros(out.shape, dtype=out.dtype)
        _np.add.at(total, where, values)
        _np.add.at(count, where, _np.ones_like(values))
        touched = count > 0
        if include_self:
            total = total + out
            count = count + 1.0
        # **안 닿은 자리는 그대로다.** 0 으로 나누지 않도록 그 자리를 갈라 둔다.
        merged = _np.where(touched, total / _np.where(count == 0, 1.0, count), out)
        out[...] = merged
        return
    start, op = _REDUCE_OPS[reduce]
    acc = _np.full(out.shape, start, dtype=out.dtype)
    op.at(acc, where, values)
    touched = _np.zeros(out.shape, dtype=bool)
    _np.logical_or.at(touched, where, True)
    if include_self:
        acc = op(acc, out)
    out[...] = _np.where(touched, acc, out)


def index_reduce(t, dim, index, source, reduce, include_self=True):
    """번호가 가리키는 **줄**을 합친다. `include_self` 가 원래 값을 첫 항으로 넣는다.

    **더하기·곱하기로만 재면 그 깃발이 안 보인다** — 1 로 채운 판에 곱하기를 하면
    켜나 끄나 같은 답이다(실측). 평균과 최소가 그 갈림을 보여 준다.
    """
    t, source = _wrap(t), _wrap(source)
    out = t.data.copy()
    where = (slice(None),) * dim + (_as_index(index),)
    _reduce_into(out, where, _np.asarray(source.data), reduce, include_self)
    return Tensor(out)


def scatter_reduce(t, dim, index, src, reduce, include_self=True):
    """`scatter` 와 같은 자리지만 **덮어쓰는 대신 합친다.**

    `sum`·`prod`·`amax`·`amin`·`mean` 이고, `mean` 은 `include_self` 면 원래 값도
    한 항으로 세어 나눈다(실측).
    """
    t, src = _wrap(t), _wrap(src)
    idx = _as_index(index)
    out = t.data.copy()
    grid = _np.indices(idx.shape)
    where = list(grid)
    where[dim] = idx
    _reduce_into(out, tuple(where), _np.asarray(src.data), reduce, include_self)
    return Tensor(out)


def renorm(t, p, dim, maxnorm):
    """`dim` 을 따라 잘라 본 **각 조각의 노름을 `maxnorm` 아래로** 끌어내린다.

    이미 작은 조각은 **안 건드린다** — 전부 크게 만들면 그 조건이 안 드러난다.

    **배율이 상수가 아니다.** `x` 가 배율 안에도 들어 있어서, 기울기를 `g·s` 로만
    적으면 순방향은 맞고 역방향만 틀린다 — 값이 그럴듯해서 안 보이는 자리다.
    깎인 조각에서만 갈리므로, 전부 작은 입력으로 재면 그것도 안 드러난다.
    """
    t = _wrap(t)
    x = t.data
    axes = tuple(a for a in range(x.ndim) if a != (dim % x.ndim))
    norms = _np.sum(_np.abs(x) ** p, axis=axes, keepdims=True) ** (1.0 / p)
    # torch 는 0 으로 나누지 않으려고 아주 작은 수를 더한다.
    cut = norms > maxnorm
    scale = _np.where(cut, maxnorm / (norms + 1e-7), 1.0)

    def back(g):
        g = _np.asarray(g)
        # `∂n/∂x = n^(1-p)·|x|^(p-1)·sign(x)`, `∂s/∂x = -M/(n+ε)²·∂n/∂x`.
        dn = norms ** (1.0 - p) * _np.abs(x) ** (p - 1) * _np.sign(x)
        ds = _np.where(cut, -maxnorm / (norms + 1e-7) ** 2 * dn, 0.0)
        dot = _np.sum(g * x, axis=axes, keepdims=True)
        return (g * scale + dot * ds,)

    return t._make(x * scale, (t,), back, "RenormBackward0")


def cartesian_prod(*tensors):
    """모든 짝. **하나만 주면 그냥 그것이다**(실측) — 1차원으로 남는다."""
    arrays = [_wrap(a).data.reshape(-1) for a in tensors]
    if len(arrays) == 1:
        return Tensor(arrays[0].copy())
    mesh = _np.meshgrid(*arrays, indexing="ij")
    return Tensor(_np.stack([m.reshape(-1) for m in mesh], axis=1))


def combinations(t, r=2, with_replacement=False):
    """`r` 개씩 고른 조합. **순서는 없고**, 중복 허용이 따로 있다."""
    from itertools import combinations as _comb, combinations_with_replacement

    flat = _wrap(t).data.reshape(-1)
    pick = combinations_with_replacement if with_replacement else _comb
    rows = [list(c) for c in pick(range(flat.shape[0]), r)]
    if not rows:
        return Tensor(_np.zeros((0, r), dtype=flat.dtype))
    return Tensor(flat[_np.asarray(rows, dtype=_np.int64)])


def tril_indices(row, col, offset=0):
    """아래 삼각의 자리들. **`(2, 개수)` 짜리 int64 표다**(실측) — 자리 쌍이 아니라
    행 줄과 열 줄로 나뉘어 온다."""
    r, c = _np.tril_indices(int(row), int(offset), int(col))
    return Tensor(_np.stack([r, c]).astype(_np.int64))


def triu_indices(row, col, offset=0):
    r, c = _np.triu_indices(int(row), int(offset), int(col))
    return Tensor(_np.stack([r, c]).astype(_np.int64))


def vander(x, N=None, increasing=False):
    """판데르몬드 행렬. **기본은 차수가 줄어드는 쪽이다** — 마지막 열이 1 이다(실측)."""
    x = _wrap(x)
    n = x.data.shape[0] if N is None else int(N)
    powers = _np.arange(n, dtype=_np.float64)
    if not increasing:
        powers = powers[::-1]
    out = x.data.reshape(-1, 1).astype(_np.float64) ** powers.reshape(1, -1)
    return Tensor(out.astype(x.data.dtype))


def chain_matmul(*matrices):
    """여러 행렬을 잇달아 곱한다. `linalg.multi_dot` 이 같은 것을 목록으로 받는다."""
    mats = list(matrices[0]) if len(matrices) == 1 and \
        isinstance(matrices[0], (list, tuple)) else list(matrices)
    out = _wrap(mats[0])
    for m in mats[1:]:
        out = matmul(out, _wrap(m))
    return out


def ger(a, b):
    """바깥곱의 옛 이름. `outer` 와 같은 것이다."""
    return outer(a, b)


def mv(mat, vec):
    """행렬 × 벡터. `matmul` 이 하는 일이지만 torch 는 이름을 따로 준다."""
    return matmul(_wrap(mat), _wrap(vec))


# ── addmm 계열 ──────────────────────────────────────────────────────────────
#
# 여덟이 전부 한 꼴이다 — `β·input + α·(무슨 곱)`. 다른 것은 **곱이 무엇인가**뿐이라
# 그 하나만 넘긴다.
#
# **`beta` 가 0 이면 `input` 을 아예 안 본다.** `0 · input` 이 아니다 — torch 는 그
# 자리에서 `input` 을 읽지도 않아서, NaN 을 넣어도 결과가 멀쩡하고 기울기도 0 이다
# (실측). `0 * input` 으로 적으면 NaN 이 번지고, 그 차이는 평범한 입력으로는 절대
# 안 보인다.

def _blend(base, product, beta, alpha):
    """`β·base + α·product`.

    **`β == 0` 은 값만 안 보고 그래프에는 남는다.** 둘 다여야 한다 —

    - `base * 0` 으로 적으면 NaN 을 넣었을 때 결과가 NaN 이 된다. torch 는 멀쩡하다.
    - 그렇다고 그래프에서 빼면 `base.grad` 가 0 이 아니라 **없다.** torch 는 0 을 준다
      (실측). 빼 두면 `backward()` 가 "requires_grad 가 아니다" 로 멈춘다.

    두 요구가 반대 방향이라 한쪽만 맞추기 쉽고, 평범한 입력으로는 **어느 쪽도** 안
    보인다 — NaN 을 넣어야 첫째가, 기울기를 물어야 둘째가 드러난다.
    """
    scaled = product if alpha == 1 else product * alpha
    if beta != 0:
        return (base if beta == 1 else base * beta) + scaled
    return scaled._make(scaled.data, (scaled, base),
                        lambda g: (g, _np.zeros_like(base.data)),
                        "AddmmBackward0")


def addmm(input, mat1, mat2, beta=1, alpha=1):
    """`β·input + α·(mat1 @ mat2)`. `input` 은 결과 모양으로 퍼진다(실측)."""
    return _blend(_wrap(input), matmul(_wrap(mat1), _wrap(mat2)), beta, alpha)


def addbmm(input, batch1, batch2, beta=1, alpha=1):
    """**배치를 합친다** — 곱한 뒤 배치 축을 더해 2차원을 낸다.

    `baddbmm` 과 이름이 한 글자 다르고 결과 차수가 다르다. 배치가 1 이면 둘이 같아
    보이므로 케이스는 배치를 둘 이상으로 둔다.
    """
    product = matmul(_wrap(batch1), _wrap(batch2)).sum(0)
    return _blend(_wrap(input), product, beta, alpha)


def baddbmm(input, batch1, batch2, beta=1, alpha=1):
    """**배치를 지킨다.** `addbmm` 과 여기서 갈린다."""
    return _blend(_wrap(input), matmul(_wrap(batch1), _wrap(batch2)),
                  beta, alpha)


def addmv(input, mat, vec, beta=1, alpha=1):
    """`β·input + α·(mat @ vec)`. 결과가 1차원이다."""
    return _blend(_wrap(input), mv(_wrap(mat), _wrap(vec)), beta, alpha)


def addr(input, vec1, vec2, beta=1, alpha=1):
    """`β·input + α·(vec1 ⊗ vec2)`. 바깥곱이라 결과가 2차원이다."""
    return _blend(_wrap(input), outer(_wrap(vec1), _wrap(vec2)), beta, alpha)


def addcmul(input, tensor1, tensor2, value=1):
    """`input + value·(t1 · t2)`. **`beta` 가 없다** — `input` 의 계수는 늘 1 이다."""
    return _blend(_wrap(input), _wrap(tensor1) * _wrap(tensor2), 1, value)


def addcdiv(input, tensor1, tensor2, value=1):
    """`input + value·(t1 / t2)`. 옵티마이저가 갱신을 적을 때 쓰는 꼴이다."""
    return _blend(_wrap(input), _wrap(tensor1) / _wrap(tensor2), 1, value)


def sspaddmm(input, mat1, mat2, beta=1, alpha=1):
    """**희소 텐서 전용이라 없다.**

    torch 의 이것은 희소 COO 를 받아 희소를 낸다(실측: `to_sparse()` 를 안 거치면
    거절한다). 여기에는 희소 배치가 없고, 조밀 텐서로 흉내 내면 **모양은 맞고 저장
    방식이 다른** 것을 주게 된다 — 그것을 배운 사람은 희소가 무엇인지 잘못 안다.
    """
    _unsupported("torch.sspaddmm — 희소(sparse) 텐서 배치가 없습니다")


def addmm_(input, mat1, mat2, beta=1, alpha=1):
    input = _wrap(input)
    return input._inplace(lambda: addmm(input, mat1, mat2, beta, alpha),
                          "addmm_")


def addbmm_(input, batch1, batch2, beta=1, alpha=1):
    input = _wrap(input)
    return input._inplace(lambda: addbmm(input, batch1, batch2, beta, alpha),
                          "addbmm_")


def baddbmm_(input, batch1, batch2, beta=1, alpha=1):
    input = _wrap(input)
    return input._inplace(lambda: baddbmm(input, batch1, batch2, beta, alpha),
                          "baddbmm_")


def addmv_(input, mat, vec, beta=1, alpha=1):
    input = _wrap(input)
    return input._inplace(lambda: addmv(input, mat, vec, beta, alpha), "addmv_")


def addr_(input, vec1, vec2, beta=1, alpha=1):
    input = _wrap(input)
    return input._inplace(lambda: addr(input, vec1, vec2, beta, alpha), "addr_")


def addcmul_(input, tensor1, tensor2, value=1):
    input = _wrap(input)
    return input._inplace(lambda: addcmul(input, tensor1, tensor2, value),
                          "addcmul_")


def addcdiv_(input, tensor1, tensor2, value=1):
    input = _wrap(input)
    return input._inplace(lambda: addcdiv(input, tensor1, tensor2, value),
                          "addcdiv_")


# ── 최상위 선형대수 ─────────────────────────────────────────────────────────
#
# **`linalg` 쪽과 인자 순서가 다르다.** torch 는 옛 이름들을 최상위에 남겨 뒀는데,
# 그것들은 대개 **오른쪽 변을 먼저** 받는다 — `lu_solve(b, LU, piv)` 대 `linalg.
# lu_solve(LU, piv, b)`. 같은 계산인데 부르는 법만 다른 것이라, 계산은 한 벌만 두고
# 자리만 옮긴다. 그 옮김이 맞는지는 값으로만 확인된다.

def _mT(t):
    """마지막 두 축을 맞바꾼다.

    **`transpose` 는 이 파일에서 모듈 함수가 아니라 메서드다.** 게다가
    `triangular_solve` 의 셋째 인자 이름이 `transpose` 라 그 안에서는 그 이름이
    인자에 가려지기도 한다. 짧은 이름 하나로 두 자리를 다 피한다.
    """
    return _wrap(t).transpose(-2, -1)


def _as_lower(factor, upper):
    """인수를 **아래 삼각으로** 세운다. `A = L Lᵀ` 가 되도록.

    조립으로 둔다 — `tril`·`transpose` 를 지나면 **인수 쪽으로도 기울기가 흐른다.**
    numpy 로 곧장 잘라 쓰면 값은 맞고 역방향이 `b` 로만 가는데, torch 는 인수로도
    흘린다(실측). 그 갈림은 인수를 미분 대상으로 두지 않으면 안 보인다.
    """
    return _mT(triu(factor)) if upper else tril(factor)


def cholesky_solve(b, factor, upper=False):
    """`A x = b` 를 **촐레스키 인수로** 푼다. `A = L Lᵀ` (또는 `Uᵀ U`).

    `A` 를 다시 세워 `solve` 로 보낸다. 삼각 대입 두 번이 더 싸지만, 그 길로 적으면
    역방향을 손으로 써야 하고 **인수 쪽 기울기가 조용히 빠진다** — 이 크기에서 아끼는
    것보다 그 위험이 크다.
    """
    low = _as_lower(_wrap(factor), upper)
    return solve(matmul(low, _mT(low)), _wrap(b))


def cholesky_inverse(factor, upper=False):
    """촐레스키 인수에서 **원래 행렬의 역행렬**을 낸다. 인수의 역이 아니다."""
    low = _as_lower(_wrap(factor), upper)
    return inverse(matmul(low, _mT(low)))


def triangular_solve(b, a, upper=True, transpose=False, unitriangular=False):
    """**둘을 준다** — 해와, 넘긴 계수 행렬의 **사본**(실측).

    `linalg.solve_triangular` 와 같은 계산인데 인자 순서가 뒤집혀 있고 **기본
    `upper` 가 참이다.** 그 둘을 놓치면 다른 삼각을 풀고도 값이 그럴듯하게 나온다.

    셋째 인자 이름이 `transpose` 라 이 함수 안에서는 모듈의 `transpose` 가 가려진다.
    `_mT` 별칭이 그 자리를 메운다.
    """
    b, a = _wrap(b), _wrap(a)
    tri = triu(a) if upper else tril(a)
    if unitriangular:
        # **대각을 안 보고 1 로 친다.** 대각을 그대로 두면 조용히 다른 답이 나온다.
        n = tri.data.shape[-1]
        off = triu(tri, 1) if upper else tril(tri, -1)
        tri = off + Tensor(_np.eye(n, dtype=tri.data.dtype))
    if transpose:
        tri = _mT(tri)
    return _TriangularSolve(solve(tri, b), Tensor(_np.array(a.data, copy=True)))


def lu_solve_top(b, lu_data, pivots):
    """**`linalg.lu_solve` 와 인자 순서가 뒤집혀 있다** — 이쪽은 `b` 가 먼저다."""
    return lu_solve(lu_data, pivots, b)


def lu_top(a, pivot=True, get_infos=False):
    """`(LU, pivots)`. **`linalg.lu` 와 다른 것을 낸다** — 그쪽은 `P·L·U` 셋으로
    펴 주고 이쪽은 **겹쳐 담은 한 판과 교환 목록**이다(실측).

    `get_infos=True` 면 셋째로 정보 코드가 붙는다. 우리는 늘 0 이다 — 특이 행렬을
    만나면 그 자리에서 던지지 조용히 코드로 알리지 않는다.
    """
    if not pivot:
        _unsupported("lu(pivot=False)")
    data, piv = lu_factor(a)
    if get_infos:
        return _LuInfos(data, piv, Tensor(_np.zeros((), dtype=_np.int32)))
    return _LuFactor(data, piv)


def lu_unpack(lu_data, lu_pivots, unpack_data=True, unpack_pivots=True):
    """겹쳐 담은 한 판을 `P·L·U` 로 편다.

    **끄면 `None` 이 아니라 빈 텐서가 온다**(실측: 모양이 `(0,)` 이다). `None` 으로
    두면 받는 쪽이 `if p is None` 으로 갈라 쓰게 되고, 그것은 torch 코드가 아니다.
    """
    lu_data, lu_pivots = _wrap(lu_data), _wrap(lu_pivots)
    empty = Tensor(_np.zeros(0, dtype=lu_data.data.dtype))
    if not unpack_data and not unpack_pivots:
        return _LuUnpack(empty, empty, empty)
    n, m = lu_data.data.shape[-2], lu_data.data.shape[-1]
    k = min(n, m)
    data = _np.asarray(lu_data.data, dtype=_np.float64)
    low = _np.tril(data[:, :k], -1).copy()
    low[_np.arange(k), _np.arange(k)] = 1.0
    up = _np.triu(data)[:k, :]
    order = _np.arange(n)
    flat = _np.asarray(lu_pivots.data).reshape(-1)
    for col in range(min(k, flat.shape[0])):
        src = int(flat[col]) - 1
        if src != col:
            order[[col, src]] = order[[src, col]]
    perm = _np.zeros((n, n))
    perm[order, _np.arange(n)] = 1.0
    kind = lu_data.data.dtype
    return _LuUnpack(
        Tensor(perm.astype(kind)) if unpack_pivots else empty,
        Tensor(low.astype(kind)) if unpack_data else empty,
        Tensor(up.astype(kind)) if unpack_data else empty)


def orgqr(a, tau):
    """`geqrf` 가 담아 둔 반사자들을 곱해 **Q 를 세운다.**
    `linalg.householder_product` 와 같은 것이고 torch 가 이름을 둘 준다."""
    return householder_product(a, tau)


def ormqr(a, tau, other, left=True, transpose=False):
    """**Q 를 안 세우고** `C` 에 곱한다. 큰 행렬에서 그것이 요점인데, 여기서는 세워서
    곱한다 — 값이 같고, 이 크기에서 아끼는 것이 없다.

    **`orgqr` 과 다른 Q 다.** 그쪽은 `m×k` 로 **자른** Q 를 주는데(`linalg.qr` 의 Q
    와 같다), 이쪽은 자르지 않은 `m×m` 을 쓴다 — 반사자들은 `Rᵐ` 위의 사상이고,
    자르면 그 사상의 일부만 곱하게 된다. 실측으로 걸렸다: 세로로 긴 행렬에서 답이
    통째로 달랐다. 정사각으로만 재면 둘이 같아서 안 보인다.

    `left` 는 어느 쪽에서 곱하는가이고 `transpose` 는 `Qᵀ` 를 쓰는가다.
    """
    q = _full_q(a, tau)
    if transpose:
        q = q.T
    c = _np.asarray(_wrap(other).data, dtype=_np.float64)
    out = (q @ c) if left else (c @ q)
    return Tensor(out.astype(_wrap(other).data.dtype))


def _full_q(a, tau):
    """반사자들을 곱해 **자르지 않은 `m×m`** Q 를 세운다.

    `householder_product` 와 같은 고리인데 마지막에 열을 안 자른다. 그 한 줄이
    `orgqr` 과 `ormqr` 의 차이 전부다.
    """
    mat = _np.asarray(_wrap(a).data, dtype=_np.float64)
    taus = _np.asarray(_wrap(tau).data, dtype=_np.float64).reshape(-1)
    m = mat.shape[-2]
    q = _np.eye(m)
    for j in range(taus.shape[0] - 1, -1, -1):
        v = _np.zeros(m)
        v[j] = 1.0
        v[j + 1:] = mat[j + 1:, j]
        q = q - taus[j] * _np.outer(v, v @ q)
    return q


def lobpcg(a, k=1, B=None, X=None, n=None, iK=None, niter=None, tol=None,
           largest=True, method=None, tracker=None, ortho_iparams=None,
           ortho_fparams=None, ortho_bparams=None):
    """대칭 행렬의 **끝쪽 고유쌍 `k` 개.**

    **torch 는 반복법이고 우리는 정확해다.** 그쪽은 큰 희소 행렬에서 몇 개만 싸게
    얻으려고 반복하는데, 우리에게는 희소가 없고 크기도 작다. 재보니 torch 의 답이
    정확해로 **7e-6 안까지** 수렴하고 씨앗에도 그만큼만 흔들린다(실측) — 이 저장소의
    허용 오차 한참 아래다. 그래서 값은 같고 비용만 다르다.

    **`largest` 가 순서까지 정한다** — 참이면 큰 것부터, 거짓이면 작은 것부터다(실측).
    """
    if B is not None or X is not None:
        _unsupported("lobpcg(B= 또는 X=) — 일반화 고유값 문제")
    vals, vecs = eigh(_wrap(a))
    order = slice(None, None, -1) if largest else slice(None)
    idx = _np.arange(vals.data.shape[-1])[order][:k]
    return _Lobpcg(Tensor(_np.asarray(vals.data)[idx]),
                   Tensor(_np.asarray(vecs.data)[:, idx]))


def svd_lowrank(a, q=6, niter=2, M=None):
    """무작위 사영으로 얻는 **저계수 SVD.** `(U, S, V)` 이고 **V 는 전치가 아니다.**

    **정확히 저계수인 입력에서만 답이 안 흔들린다.** torch 는 무작위 행렬로 사영하는데,
    계수가 `q` 를 넘으면 씨앗에 따라 특이값이 0.5 씩 움직인다(실측). 계수가 `q` 이하면
    씨앗을 바꿔도 7e-7 안이다 — 골든이 물을 수 있는 자리는 그쪽뿐이다.

    우리는 사영을 안 한다. 전체 SVD 를 구해 앞의 `q` 개를 자른다 — 정확히 저계수인
    자리에서는 같은 답이고, 넘치는 자리에서는 **torch 보다 정확한** 답이다.
    """
    a = _wrap(a)
    data = _np.asarray(a.data, dtype=_np.float64)
    if M is not None:
        data = data - _np.asarray(_wrap(M).data, dtype=_np.float64)
    u, s, vh = _np.linalg.svd(data, full_matrices=False)
    kind = a.data.dtype
    return _SvdLowrank(Tensor(u[:, :q].astype(kind)), Tensor(s[:q].astype(kind)),
                       Tensor(vh[:q].T.astype(kind)))


def pca_lowrank(a, q=None, center=True, niter=2):
    """저계수 PCA. **`center=False` 면 `svd_lowrank` 와 같은 것이다**(실측).

    가운데 맞추기가 이 함수와 저쪽의 차이 전부다. 참으로만 재면 그 갈래가 안 보인다.
    """
    a = _wrap(a)
    data = _np.asarray(a.data, dtype=_np.float64)
    if q is None:
        q = min(6, *data.shape)
    if center:
        data = data - data.mean(axis=0, keepdims=True)
    return svd_lowrank(Tensor(data.astype(a.data.dtype)), q, niter)


# ── 통계 ────────────────────────────────────────────────────────────────────
#
# **난수 넷에는 굳힐 수 있는 구석이 있다.**
#
# `normal`·`bernoulli`·`poisson`·`binomial` 의 값은 골든이 못 굳힌다 — torch 의 난수
# 줄기와 우리 것이 다르고, 같게 만들 방법도 없다. 그런데 **끝값은 결정적이다**:
# `std=0` 이면 평균 그대로, `p=0` 이면 전부 0, `p=1` 이면 전부 1, `poisson(0)` 은 0 이다
# (실측). 골든은 그 자리를 묻고, 나머지는 모양과 형만 본다.
#
# 그것이 "난수라 못 묻는다" 와 "안 묻는다" 의 차이다.

def _edges(data, bins, low, high):
    """경계를 세운다. **마지막 칸은 오른쪽이 닫혀 있다**(실측)."""
    if low == high:
        low, high = float(_np.min(data)), float(_np.max(data))
        if low == high:
            low, high = low - 0.5, high + 0.5
    return _np.linspace(low, high, int(bins) + 1)


def _count_into(data, edges, weights=None):
    """`edges` 가 나눈 칸에 센다. **범위 밖은 버린다** — torch 가 그렇다(실측)."""
    flat = _np.asarray(data, dtype=_np.float64).reshape(-1)
    w = (_np.ones_like(flat) if weights is None
         else _np.asarray(weights, dtype=_np.float64).reshape(-1))
    out = _np.zeros(len(edges) - 1, dtype=_np.float64)
    for value, weight in zip(flat, w):
        if value < edges[0] or value > edges[-1]:
            continue
        # 오른쪽 끝은 마지막 칸에 넣는다.
        slot = int(_np.searchsorted(edges, value, side="right")) - 1
        out[min(max(slot, 0), len(out) - 1)] += weight
    return out


def histc(t, bins=100, min=0, max=0):
    """칸마다 몇 개인가. **`min == max` 면 자료의 범위를 쓴다**(실측).

    범위를 주면 **밖은 버린다** — 양끝 칸으로 몰아넣지 않는다. 전부 범위 안인 자료로
    재면 그 규칙이 안 드러난다.
    """
    t = _wrap(t)
    edges = _edges(t.data, bins, float(min), float(max))
    return Tensor(_count_into(t.data, edges).astype(t.data.dtype))


def histogram(t, bins=100, range=None, weight=None, density=False):
    """`histc` 와 같은 셈에 **경계까지 준다.**

    `bins` 에 텐서를 주면 그것이 곧 경계다 — 칸 너비가 다를 수 있고, 그러면
    `density` 가 칸마다 다른 값으로 나눈다.
    """
    t = _wrap(t)
    if isinstance(bins, (Tensor, list, tuple, _np.ndarray)):
        edges = _np.asarray(_wrap(bins).data if isinstance(bins, Tensor) else bins,
                            dtype=_np.float64)
    else:
        low, high = (0.0, 0.0) if range is None else (float(range[0]), float(range[1]))
        edges = _edges(t.data, bins, low, high)
    counts = _count_into(t.data, edges,
                         None if weight is None else _wrap(weight).data)
    if density:
        widths = _np.diff(edges)
        total = counts.sum()
        counts = counts / (widths * (total if total else 1.0))
    kind = t.data.dtype
    return _Histogram(Tensor(counts.astype(kind)), Tensor(edges.astype(kind)))


def histogramdd(t, bins=10, range=None, weight=None, density=False):
    """축이 여럿인 히스토그램. `t` 는 `(표본 수, 차원)` 이다."""
    t = _wrap(t)
    data = _np.asarray(t.data, dtype=_np.float64)
    dims = data.shape[-1]
    counts = [bins] * dims if isinstance(bins, int) else list(bins)
    edges = []
    for d in _builtin_range(dims):
        low, high = (0.0, 0.0)
        if range is not None:
            low, high = float(range[2 * d]), float(range[2 * d + 1])
        edges.append(_edges(data[:, d], counts[d], low, high))
    hist, _ = _np.histogramdd(data, bins=edges, density=density,
                              weights=None if weight is None
                              else _np.asarray(_wrap(weight).data).reshape(-1))
    kind = t.data.dtype
    return _HistogramDD(Tensor(hist.astype(kind)),
                        [Tensor(e.astype(kind)) for e in edges])


def mode(t, dim=-1, keepdim=False):
    """가장 자주 나온 값. **같은 횟수면 작은 값이 이기고, 자리는 그 값의 마지막이다**
    (실측: `[4,4,5,5]` 가 값 4 · 자리 1 을 준다).

    비긴 자리가 없는 자료로 재면 그 규칙이 안 드러난다.
    """
    t = _wrap(t)
    data = _np.asarray(t.data)
    axis = dim % data.ndim
    moved = _np.moveaxis(data, axis, -1)
    flat = moved.reshape(-1, moved.shape[-1])
    vals = _np.empty(flat.shape[0], dtype=data.dtype)
    idx = _np.empty(flat.shape[0], dtype=_np.int64)
    for row in _builtin_range(flat.shape[0]):
        line = flat[row]
        best, best_count = None, -1
        for value in _np.unique(line):
            count = int((line == value).sum())
            if count > best_count:
                best, best_count = value, count
        vals[row] = best
        idx[row] = int(_np.flatnonzero(line == best)[-1])
    # 기울기는 **밝힌 그 자리 하나로** 간다. 가장 자주 나온 값이라 같은 값이 여럿
    # 있지만, `mode` 는 그중 마지막을 번호로 건네므로 그 자리가 답을 대표한다
    # (실측: [1,1,2,2,2] 의 기울기가 마지막 2 에만 간다). 여기에도 `Tensor(...)` 만
    # 있어서 그래프가 끊겨 있었다.
    weight = _np.zeros(flat.shape, dtype=_np.float64)
    weight[_np.arange(flat.shape[0]), idx] = 1.0
    weight = _np.moveaxis(weight.reshape(moved.shape), -1, axis)
    shape = moved.shape[:-1]
    vals = vals.reshape(shape)
    idx = idx.reshape(shape)
    if keepdim:
        vals = _np.expand_dims(vals, axis)
        idx = _np.expand_dims(idx, axis)

    def back(g):
        gg = _np.asarray(g)
        if keepdim:
            gg = _np.squeeze(gg, axis)
        return (_np.expand_dims(gg, axis) * weight,)

    return _Mode(t._make(vals, (t,), back, "ModeBackward0"), Tensor(idx))


def nanmedian(t, dim=None, keepdim=False):
    """NaN 을 **빼고** 센 중앙값. `median` 은 NaN 이 하나만 있어도 NaN 을 낸다(실측).

    짝수 개면 **아래를 고른다** — 평균을 내지 않는다.
    """
    t = _wrap(t)
    _refuses_bool(t.data, "nanmedian 은 참거짓을 받지 않습니다.",
                  '"median_cpu" not implemented for \'Bool\'',
                  kind=NotImplementedError)
    data = _np.asarray(t.data, dtype=_np.float64)
    if dim is None:
        flat = data.reshape(-1)
        clean = flat[~_np.isnan(flat)]
        pick = _np.sort(clean)[(clean.shape[0] - 1) // 2]
        # **값이 같은 칸 전부에 고르게 나눈다** — `median()` 과 같은 규칙이다.
        # 여기에는 `Tensor(...)` 만 있어서 그래프가 조용히 끊겨 있었다.
        share = (flat == pick).astype(_np.float64)
        share = (share / share.sum()).reshape(t.data.shape)
        return t._make(_np.asarray(pick, dtype=t.data.dtype), (t,),
                       lambda g: (_np.asarray(g) * share,), "NanmedianBackward0")
    axis = dim % data.ndim
    moved = _np.moveaxis(data, axis, -1)
    flat = moved.reshape(-1, moved.shape[-1])
    vals = _np.empty(flat.shape[0], dtype=_np.float64)
    idx = _np.empty(flat.shape[0], dtype=_np.int64)
    for row in _builtin_range(flat.shape[0]):
        line = flat[row]
        keep = _np.flatnonzero(~_np.isnan(line))
        order = keep[_np.argsort(line[keep], kind="stable")]
        at = order[(order.shape[0] - 1) // 2]
        vals[row] = line[at]
        idx[row] = int(at)
    # **축을 주면 번호가 나오고, 번호가 나오면 기울기는 그 자리 하나로 간다.**
    # 축이 없을 때 고르게 나누는 것과 반대인데, 갈림은 같다 — 번호를 건네는 연산은
    # 고른 자리를 밝히고, 안 건네는 연산은 값이 같은 칸을 구별하지 않는다.
    weight = _np.zeros(flat.shape, dtype=_np.float64)
    weight[_np.arange(flat.shape[0]), idx] = 1.0
    weight = _np.moveaxis(weight.reshape(moved.shape), -1, axis)
    shape = moved.shape[:-1]
    vals = vals.reshape(shape).astype(t.data.dtype)
    idx = idx.reshape(shape)
    if keepdim:
        vals = _np.expand_dims(vals, axis)
        idx = _np.expand_dims(idx, axis)

    def back(g):
        gg = _np.asarray(g)
        if keepdim:
            gg = _np.squeeze(gg, axis)
        return (_np.expand_dims(gg, axis) * weight,)

    return _NanMedian(t._make(vals, (t,), back, "NanmedianBackward0"), Tensor(idx))


def gradient(t, spacing=1, dim=None, edge_order=1):
    """중심 차분. **축마다 하나씩, 묶음으로 낸다** — 축을 안 주면 전부다.

    `edge_order` 가 1 이면 양끝을 한쪽 차분으로, 2 면 이차식으로 맞춘다(실측:
    `x²` 에서 2 면 정확한 도함수가 나오고 1 이면 양끝이 어긋난다).
    """
    t = _wrap(t)
    data = _np.asarray(t.data, dtype=_np.float64)
    axes = (tuple(_builtin_range(data.ndim)) if dim is None
            else (dim,) if isinstance(dim, int) else tuple(dim))
    step = spacing if isinstance(spacing, (list, tuple)) else [spacing] * len(axes)
    outs = []
    for axis, gap in zip(axes, step):
        if isinstance(gap, Tensor):
            gap = _np.asarray(gap.data, dtype=_np.float64)
        got = _np.gradient(data, gap, axis=axis % data.ndim,
                           edge_order=int(edge_order))
        outs.append(Tensor(got.astype(t.data.dtype)))
    return tuple(outs)


def trapz(y, x=None, dx=1.0, dim=-1):
    """`trapezoid` 의 옛 이름. 같은 것이다(실측)."""
    return trapezoid(y, x, dx, dim)


def nonzero_static(t, size, fill_value=-1):
    """0 이 아닌 자리를 **정해진 개수만큼** 낸다. 모자라면 채우고 넘치면 자른다.

    `nonzero` 는 결과 크기가 값에 달려 GPU 에서 한 번 읽어야 하는데, 이쪽은 크기를
    미리 주므로 그 왕복이 없다 — 그 자리를 위해 있는 이름이다.
    """
    t = _wrap(t)
    found = _np.argwhere(_np.asarray(t.data) != 0)
    rank = max(1, _np.asarray(t.data).ndim)
    out = _np.full((int(size), rank), int(fill_value), dtype=_np.int64)
    take = min(int(size), found.shape[0])
    out[:take] = found[:take]
    return Tensor(out)


def normal(mean=0.0, std=1.0, size=None, **kw):
    """정규분포 표본. **`std` 가 0 이면 평균 그대로다** — 골든이 그 자리를 묻는다.

    `mean`·`std` 를 텐서로 주면 자리마다 다른 분포다. 그때는 `size` 를 안 받는다.
    """
    if isinstance(mean, Tensor) or isinstance(std, Tensor):
        m = _np.asarray(_wrap(mean).data, dtype=_np.float64)
        s = _np.asarray(_wrap(std).data, dtype=_np.float64)
        m, s = _np.broadcast_arrays(m, s)
        return Tensor(_rng.normal(m, s).astype(_DEFAULT_DTYPE))
    shape = () if size is None else tuple(size)
    return Tensor(_rng.normal(float(mean), float(std), shape).astype(_DEFAULT_DTYPE))


def bernoulli(t, **kw):
    """자리마다 그 확률로 1. **0 이면 전부 0, 1 이면 전부 1** — 그 두 끝이 결정적이다."""
    t = _wrap(t)
    p = _np.asarray(t.data, dtype=_np.float64)
    return Tensor((_rng.random(p.shape) < p).astype(t.data.dtype))


def poisson(t, **kw):
    """자리마다 그 세기의 포아송 표본. **0 이면 전부 0 이다**(실측)."""
    t = _wrap(t)
    lam = _np.asarray(t.data, dtype=_np.float64)
    return Tensor(_rng.poisson(lam).astype(t.data.dtype))


def binomial(count, prob, **kw):
    """`count` 번 중 성공 횟수. **`p=0` 이면 0, `p=1` 이면 `count` 다.**"""
    n = _np.asarray(_wrap(count).data, dtype=_np.float64)
    p = _np.asarray(_wrap(prob).data, dtype=_np.float64)
    n, p = _np.broadcast_arrays(n, p)
    return Tensor(_rng.binomial(n.astype(_np.int64), p).astype(_DEFAULT_DTYPE))


# **오래 거절이었다.** 거절문에는 "복소수 규약을 안 정했다" 고 적혀 있었고, 그
# 이유가 맞았다 — 저장이 모자란 것이 아니라 **Wirtinger 규약을 안 재본 것**이었다.
# 재서 못 박고 나니(`z.grad = ∂L/∂re + i·∂L/∂im`) 이 두 이름이 조립으로 나왔다.
#
# 못 하는 이유를 **정확히** 적어 둔 값어치가 여기서 나왔다. "저장이 없다" 로 적어
# 두었으면 저장이 생긴 날에도 아무도 다시 안 물었을 것이다.
stft = _fft_stft
istft = _fft_istft


def hash_tensor(*args, **kw):
    """**uint64 도 없고 규격도 없다.**

    torch 가 내는 것은 `uint64` 이고(실측), 어떤 해시인지는 문서에도 없다. 값을
    맞출 수 없는 것을 이름만 놓으면 그 값을 믿고 쓰는 코드가 생긴다.
    """
    _unsupported("torch.hash_tensor — uint64 도, 정해진 해시 규격도 없습니다")


# ── 복소수의 이웃, 그리고 생성 몇 ──────────────────────────────────────────
#
# **복소수가 없어도 답이 있는 이름들이다.**
#
# `real`·`conj`·`resolve_conj` 는 실수 텐서에서 **항등**이고(실측: 버퍼까지 공유한다),
# `is_complex`·`is_conj`·`is_neg` 는 전부 거짓이며, `angle` 은 음수에서 π 다. 복소수
# 규약을 안 정했다고 이 이름들까지 없으면, 그것을 분기에 쓰는 교재 코드가 `AttributeError`
# 로 멈춘다 — 답할 수 있는 것과 없는 것은 다르다.
#
# `imag` 만 다르다. **torch 자신이 실수에서 거절한다**(실측) — 그래서 여기서 거절하는
# 것은 우리 한계가 아니라 **torch 를 그대로 옮긴 것**이다.

def _is_complex(t):
    return _np.asarray(t.data).dtype.kind == "c"


def _alias(t, name):
    """같은 값을 그대로 내는 항등. **형과 그래프를 지킨다.**

    `positive` 의 단항 커널로 보내면 안 된다 — 그쪽은 형이 float32 로 떨어져서
    `bool` 을 넣으면 `bool` 이 안 나온다(실측: torch 는 `bool` 을 그대로 준다).
    """
    t = _wrap(t)
    return t._make(t.data, (t,), lambda g: (g,), name)


# ── 복소수의 기울기 규약 ────────────────────────────────────────────────────
#
# **손실이 늘 실수라서 Wirtinger 가 무너진다.** torch 는 복소 손실에 `backward()` 를
# 거절한다(실측: "grad can be implicitly created only for real scalar outputs").
# 그러면 규약이 이것으로 정리된다 —
#
#   z.grad = ∂L/∂re + i·∂L/∂im        (실수 둘로 따로 미분해서 묶는다)
#
# 실측이 그것을 못 박는다(z = 1+2j):
#
#   L = z.real  → 1+0j        L = z.imag  → **0+1j** (−1j 가 아니다)
#   L = |z|²    → 2+4j        L = (z·z̄).real → 2+4j
#
# 이 규약에서 **정칙 함수 f 의 역방향은 `conj(f'(z))·g` 다** — 실수 쪽 코드가 쓰는
# `f'(x)·g` 와 **켤레 하나**가 다르다. 그 하나를 빼먹으면 부호만 뒤집힌 기울기가
# 나오고, 값이 그럴듯해서 실수 입력으로는 절대 안 보인다.

def _cgrad(local, g):
    """정칙 함수의 역방향 한 항. **켤레가 붙는 자리**다."""
    return _np.conj(local) * g


def real(t):
    """실수부.

    실수 텐서에서는 **자기 자신**이고 형도 그대로다(`bool` 도 `bool`). 복소수에서는
    실수부를 꺼내고, **기울기는 실수 자리로만 흐른다** — `z.real` 의 기울기가
    `1+0j` 인 것이 그 뜻이다(실측).
    """
    t = _wrap(t)
    if not _is_complex(t):
        return _alias(t, "RealBackward0")
    return t._make(_np.real(t.data).copy(), (t,),
                   lambda g: (_np.asarray(g).astype(t.data.dtype),),
                   "RealBackward0")


def imag(t):
    """허수부.

    **실수 텐서에서는 torch 도 거절한다**(실측) — 우리 한계가 아니다. 복소수에서는
    허수부를 꺼내고, **기울기가 `i` 를 달고 돌아간다** — `z.imag` 의 기울기가
    `0+1j` 다(실측). `−1j` 로 적으면 부호가 뒤집힌 채 그럴듯하게 돈다.
    """
    t = _wrap(t)
    if not _is_complex(t):
        raise RuntimeError(_like_torch(
            "실수 텐서에는 허수부가 없습니다.",
            "imag is not implemented for tensors with non-complex dtypes."))
    return t._make(_np.imag(t.data).copy(), (t,),
                   lambda g: (_np.asarray(g).astype(t.data.dtype) * 1j,),
                   "ImagBackward0")


def conj(t):
    """켤레. 실수에서는 항등이고 **뷰다** — torch 도 버퍼를 공유한다(실측).

    **정칙이 아니다.** 그래서 역방향이 `conj(f')·g` 꼴이 아니라 **`conj(g)`** 다 —
    켤레의 켤레라서 그렇다.
    """
    t = _wrap(t)
    if not _is_complex(t):
        return _alias(t, "ConjBackward0")
    return t._make(_np.conj(t.data), (t,), lambda g: (_np.conj(_np.asarray(g)),),
                   "ConjBackward0")


def conj_physical(t):
    """`conj` 와 같은 값. torch 는 이쪽이 **실제로 복사하는 판**이라고 이름을 나눈다.

    **복소수가 들어오기 전에는 이 함수가 항등이었다** — 실수만 있던 시절에는 그것이
    맞는 값이었고, 골든도 통과했다. 복소수를 붙이는 순간 같은 코드가 틀린 답이 됐다.
    "지금 통과하는 항등" 은 범위가 넓어질 때 제일 먼저 무너지는 자리다.
    """
    t = _wrap(t)
    if not _is_complex(t):
        return _alias(t, "ConjPhysicalBackward0")
    return t._make(_np.conj(t.data), (t,), lambda g: (_np.conj(_np.asarray(g)),),
                   "ConjPhysicalBackward0")


def conj_physical_(t):
    t = _wrap(t)
    return t._inplace(lambda: conj_physical(t), "conj_physical_")


def resolve_conj(t):
    """켤레 표시를 실제 값으로 굳힌다. 실수에는 그 표시가 없어 항등이다."""
    return _alias(t, "ResolveConjBackward0")


def resolve_neg(t):
    """부호 표시를 굳힌다. `resolve_conj` 와 같은 자리다."""
    return _alias(t, "ResolveNegBackward0")


def angle(t):
    """편각. 실수에서는 **음수가 π, 나머지가 0** 이다 — 복소수의 특수한 경우다.

    **형이 언제나 float32 다** — 정수를 넣어도 정수가 안 나온다(실측). 각도는 정수 칸에
    안 들어가므로 그것이 맞고, 실수만 넣어 보면 그 규칙이 안 드러난다.
    """
    t = _wrap(t)
    data = _np.asarray(t.data)
    if data.dtype.kind == "c":
        return Tensor(_np.angle(data).astype(_DEFAULT_DTYPE))
    out = _np.where(data < 0, _math.pi, 0.0).astype(_DEFAULT_DTYPE)
    # **0 을 흘린다 — "없다" 가 아니라 맞는 답이다.** 실수의 편각은 계단이라 어디서든
    # 도함수가 0 이고, torch 도 0 을 채운다(실측). 그래프를 안 이으면 `backward()` 가
    # 멈추는데, 그때 나오는 말은 사용자를 가리키지 이 연산을 가리키지 않는다.
    return t._make(out, (t,), lambda g: (_np.zeros_like(data, dtype=_np.float64),),
                   "AngleBackward0")


def _complex_abs(t):
    """복소수의 크기. **역방향에 켤레가 안 붙는다** — 실수를 내는 함수라 정칙이 아니다.

    `∂|z|/∂re = re/|z|`, `∂|z|/∂im = im/|z|` 를 묶으면 `z/|z|` 다. 실측이 그것을
    받친다: `L = |z|²` 에서 기울기가 `2z` 다(z=1+2j 에서 2+4j).
    """
    data = _np.asarray(t.data)
    mag = _np.abs(data)
    out = mag.astype(_DEFAULT_DTYPE)
    safe = _np.where(mag == 0, 1.0, mag)

    def back(g):
        return ((_np.asarray(g) * data / safe).astype(data.dtype),)

    return t._make(out, (t,), back, "AbsBackward0")


def complex(re, im):
    """실수부와 허수부를 묶는다. **이 이름이 파이썬 내장을 가린다** — 이 파일 안에서
    복소수 판정에 `_is_complex` 를 쓰는 이유가 그것이다."""
    re, im = _wrap(re), _wrap(im)
    out = (_np.asarray(re.data, dtype=_np.float32)
           + 1j * _np.asarray(im.data, dtype=_np.float32)).astype(_np.complex64)
    # **실수 잎으로 기울기가 흐른다.** 실수부는 실수 몫을, 허수부는 허수 몫을 받는다 —
    # 그것이 `∂L/∂re + i·∂L/∂im` 규약의 반대 방향이다.
    return re._make(out, (re, im),
                    lambda g: (_np.real(_np.asarray(g)).astype(_np.float32),
                               _np.imag(_np.asarray(g)).astype(_np.float32)),
                    "ComplexBackward0")


def polar(abs_, angle_):
    """크기와 편각으로 만든다. `abs·(cos θ + i sin θ)`."""
    abs_, angle_ = _wrap(abs_), _wrap(angle_)
    mag = _np.asarray(abs_.data, dtype=_np.float64)
    ang = _np.asarray(angle_.data, dtype=_np.float64)
    return Tensor((mag * _np.exp(1j * ang)).astype(_np.complex64))


def view_as_real(t):
    """복소수를 `(…, 2)` 실수로 본다. 마지막 축이 `(re, im)` 이다(실측).

    **실수 텐서에는 안 된다** — torch 도 거절한다.
    """
    t = _wrap(t)
    if not _is_complex(t):
        raise RuntimeError(_like_torch(
            "실수 텐서에는 쓸 수 없습니다 — 복소수만 됩니다.",
            "view_as_real is only supported for complex tensors"))
    out = _np.stack([_np.real(t.data), _np.imag(t.data)], axis=-1)
    return t._make(out.astype(_np.float32), (t,),
                   lambda g: ((_np.asarray(g)[..., 0]
                               + 1j * _np.asarray(g)[..., 1]).astype(t.data.dtype),),
                   "ViewAsRealBackward0")


def view_as_complex(t):
    """`(…, 2)` 실수를 복소수로 본다. `view_as_real` 의 반대다."""
    t = _wrap(t)
    data = _np.asarray(t.data)
    if data.shape[-1] != 2:
        raise RuntimeError(_like_torch(
            "마지막 축이 2 여야 합니다.",
            "Tensor must have a last dimension of size 2"))
    out = (data[..., 0] + 1j * data[..., 1]).astype(_np.complex64)
    return t._make(out, (t,),
                   lambda g: (_np.stack([_np.real(_np.asarray(g)),
                                         _np.imag(_np.asarray(g))],
                                        axis=-1).astype(data.dtype),),
                   "ViewAsComplexBackward0")


def is_complex(t):
    """복소수인가. **복소수를 넣기 전에는 늘 거짓이었다** — 이제 진짜로 본다."""
    return _is_complex(_wrap(t))


def is_conj(t):
    """켤레 표시가 붙어 있는가. 그 표시를 만들 길이 없으니 늘 거짓이다."""
    return False


def is_neg(t):
    """부호 표시가 붙어 있는가. `is_conj` 와 같은 자리다."""
    return False


def asarray(obj, dtype=None, copy=None, **kw):
    """**텐서를 주면 사본이 아니다**(실측). `copy=True` 여야 사본이다.

    `as_tensor` 와 거의 같은데 `copy` 를 명시로 받는 자리가 다르다 — 그 인자가
    없으면 "안 베끼는 것이 기본" 이라는 규칙을 부르는 쪽이 못 고른다.
    """
    if isinstance(obj, Tensor) and dtype is None and not copy:
        return obj
    if isinstance(obj, Tensor):
        data = obj.data.astype(dtype.np) if dtype is not None else obj.data
        return Tensor(_np.array(data, copy=True) if copy else data)
    got = tensor(obj, dtype)
    return Tensor(_np.array(got.data, copy=True)) if copy else got


def frombuffer(buffer, dtype=_float32, count=-1, offset=0, **kw):
    """바이트를 그대로 읽는다. **`offset` 은 바이트 수다** — 원소 수가 아니다(실측)."""
    kind = dtype.np if hasattr(dtype, "np") else _np.dtype(dtype)
    return Tensor(_np.frombuffer(buffer, dtype=kind, count=count,
                                 offset=offset).copy())


def range_top(start, end=None, step=1, **kw):
    """**끝을 포함한다** — `arange` 는 뺀다(실측: `range(0, 4)` 가 다섯 개다).

    torch 가 폐기 예정으로 두었지만 옛 교재에 남아 있고, `arange` 와 한 칸 다른 것이
    바로 그 폐기 사유다. 조용히 `arange` 로 넘기면 원소가 하나 모자란다.

    **이름이 `range` 가 아닌 이유**: 이 파일이 파이썬 내장 `range` 를 91 곳에서 쓴다.
    모듈에 그 이름을 두면 전부가 이 함수를 부르게 된다 — `lu`·`lu_solve` 와 같은
    수법으로 여기서는 `range_top` 이고 `borch/__init__.py` 가 `range` 로 내보낸다.
    이 파일에서 아홉 번째 겪는 "모듈 이름이 내장을 가린다" 다.
    """
    if end is None:
        start, end = 0, start
    return Tensor(_np.arange(start, end + step / 2.0, step,
                             dtype=_DEFAULT_DTYPE))


def empty_strided(size, stride, **kw):
    """**걸음을 표현할 수 없어서 없다.**

    `as_strided` 와 다른 자리다. 그쪽은 **값**이 답이라 사본으로도 같은 답을 내는데,
    이쪽은 값이 쓰레기이고 **걸음 자체가 유일한 답**이다. 우리 텐서에는 걸음이라는
    것이 없으므로 모양만 맞춘 것을 주면 "걸음이 그렇다" 고 믿는 코드가 생긴다.
    """
    _unsupported("torch.empty_strided — 걸음(stride)이라는 것이 없습니다")


def empty_permuted(size, physical_layout, **kw):
    """`empty_strided` 와 같은 이유로 없다."""
    _unsupported("torch.empty_permuted — 걸음(stride)이라는 것이 없습니다")


# ================================================================ 이름 잇기
#
# **파일 끝이어야 한다.** 아래 두 고리가 이 파일의 함수들을 이름으로 찾으므로, 위에서
# 돌면 아직 정의 안 된 것을 못 본다 — `add` 하나에서 `KeyError` 로 멈췄다.

for _nm in _INPLACE_UNARY + _INPLACE_MORE:
    setattr(Tensor, _nm + "_", _make_inplace(_nm))
for _nm in _INPLACE_BINARY + _INPLACE_ARGS:
    setattr(Tensor, _nm + "_", _make_inplace(_nm, "args"))


# ---- 모듈 함수를 **메서드로도** 낸다
#
# torch 는 같은 것을 둘 다 준다 — `torch.add(x, y)` 와 `x.add(y)`. `borch/__init__.py`
# 에 그 반대 방향 고리가 있는데(메서드 → 모듈 함수), **이쪽 방향이 없었다.** 그래서
# 계산은 다 해 놓고 이름이 한쪽에서만 닿았다 — `borch.matrix_exp(x)` 는 되고
# `x.matrix_exp()` 는 안 됐다. `x.add(y)` 는 torch 코드에서 아주 흔한 꼴이다.
#
# **아무 이름이나 걸면 안 된다.** 모듈에 있는 것을 전부 메서드로 만들면 torch 에 없는
# 메서드가 생기고, 그러면 우리에게서만 도는 코드를 쓰게 된다. 그래서 목록을 적고,
# 그 목록이 진짜 torch 의 메서드인지는 `tests/test_tensor_api.py` 가 확인한다.
_AS_METHOD = (
    "add", "sub", "mul", "div", "subtract", "multiply", "divide", "true_divide",
    "floor_divide", "remainder", "fmod", "float_power", "lerp",
    "greater", "greater_equal", "less", "less_equal", "not_equal",
    "logical_and", "logical_or", "logical_not", "logical_xor",
    "isclose", "isneginf", "isposinf", "isreal", "nan_to_num",
    "fmax", "fmin", "cross", "inner", "kron", "vdot", "count_nonzero",
    "corrcoef", "cov", "adjoint", "broadcast_to", "moveaxis", "t",
    "det", "logdet", "slogdet", "inverse", "pinverse", "matrix_exp",
    "matrix_power", "cholesky", "qr", "svd",
    "digamma", "erfinv", "lgamma", "hardshrink", "prelu", "log_softmax",
    "bitwise_and", "bitwise_or", "bitwise_xor", "bitwise_not",
    "bitwise_left_shift", "bitwise_right_shift", "gcd", "lcm",
    "nextafter", "frexp", "logcumsumexp", "mvlgamma", "i0",
    # `fill` 은 여기 없다 — torch 는 최상위에만 두고 메서드로는 `fill_` 만 준다.
    "clamp_max", "clamp_min", "detach_",
    # 모양·색인. `unravel_index`·`cartesian_prod`·`combinations`·`tril_indices`
    # ·`triu_indices`·`vander`·`chain_matmul` 은 여기 없다 — torch 가 그것들은
    # 최상위에만 두기 때문이다.
    "as_strided", "as_strided_", "as_strided_scatter", "diag_embed",
    "diagonal_scatter", "select_scatter", "slice_scatter", "split_with_sizes",
    "tensor_split", "unique_consecutive",
    "index_put", "index_put_", "index_reduce", "masked_scatter",
    "masked_scatter_", "put", "renorm", "scatter_reduce", "ger", "mv",
    # addmm 계열. **제자리 판은 torch 에서 메서드로만 있다** — `torch.addmm_` 이라는
    # 최상위 이름은 없다(실측). 그래서 여기에는 있고 `borch/__init__.py` 에는 없다.
    # 예외가 `addmv_` 하나인데 그것만 torch 가 최상위에도 둔다.
    "addmm", "addmm_", "addbmm", "addbmm_", "baddbmm", "baddbmm_",
    "addmv", "addmv_", "addr", "addr_", "addcmul", "addcmul_",
    "addcdiv", "addcdiv_", "sspaddmm",
    # 최상위 선형대수. `lu_unpack`·`lobpcg`·`pca_lowrank`·`svd_lowrank` 는 여기
    # 없다 — torch 가 그 넷은 최상위에만 둔다(실측).
    "cholesky_solve", "cholesky_inverse", "triangular_solve", "orgqr", "ormqr",
    # 통계. `histogramdd`·`gradient`·`trapz`·`normal`·`poisson`·`binomial` 은
    # 여기 없다 — torch 가 그것들을 최상위에만 둔다(실측).
    "histc", "histogram", "mode", "nanmedian", "bernoulli", "nonzero_static",
    "stft", "istft", "hash_tensor",
    # 복소수의 이웃. `asarray`·`frombuffer`·`range`·`empty_strided` 는 여기 없다 —
    # torch 가 그것들을 최상위에만 둔다(실측).
    "angle", "conj", "conj_physical", "conj_physical_", "imag", "is_complex",
    "is_conj", "is_neg", "resolve_conj", "resolve_neg",
)


def _as_method(name):
    fn = globals()[name]

    def method(self, *args, **kw):
        return fn(self, *args, **kw)

    method.__name__ = name
    method.__doc__ = f"`torch.{name}(x, ...)` 과 같다. torch 는 둘 다 준다."
    return method


for _nm in _AS_METHOD:
    if not hasattr(Tensor, _nm):
        setattr(Tensor, _nm, _as_method(_nm))

# **`lu`·`lu_solve` 는 이름이 둘씩이다.** 이 파일의 그 이름들은 `linalg` 쪽 것이고
# (`lu` 는 `P·L·U` 셋을 펴 주고, `lu_solve` 는 인수를 먼저 받는다), 메서드는
# **최상위 쪽**이어야 한다. `_AS_METHOD` 는 이름이 하나일 때만 쓸 수 있으므로
# 이 둘만 손으로 건다 — 그냥 목록에 넣으면 메서드가 다른 함수를 부른다.
Tensor.lu = _as_method("lu_top")
Tensor.lu.__name__ = "lu"
Tensor.lu_solve = _as_method("lu_solve_top")
Tensor.lu_solve.__name__ = "lu_solve"


# ── 분포에서 뽑아 제자리에 채우는 일곱 ────────────────────────────────────────
#
# **`_ops.py` 에 둔다 — `_rng` 가 여기 산다.** 처음에 `_tensor.py` 에 두고 부를 때마다
# `from ._ops import _rng` 로 집었는데, `sys.modules` 에서 `borch.*` 를 지우는 검사
# (`test_alias`)가 먼저 돌면 **다른 `_ops` 의 생성기**를 집는다. 그러면 씨앗을 심어도
# 안 먹고, 그 증상은 "혼자 돌리면 되는데 다 같이 돌리면 안 된다" 라 원인에서 멀다.
#
# `bernoulli_` 처럼 끝값이 확정인 자리가 없어서 **값은 못 굳힌다.** 그래서 표에
# 물을 것은 값이 아니라 셋이다 — 모양·형이 안 바뀌는가, 못 쓰는 형을 거절하는가,
# 인자의 정의역을 지키는가. 뒤의 둘이 특히 갈리기 쉽다: **torch 의 규칙이 분포마다
# 다르고 예외 종류까지 다르다**(실측).
#
#   연속 분포는 정수·참거짓을 **거절**한다 — `normal_`·`uniform_`·`log_normal_` 은
#   `NotImplementedError`, `exponential_`·`cauchy_` 는 이유를 적은 `RuntimeError` 다.
#   `geometric_` 은 **이산이라 정수에서 돈다.** 이름만 보고 "난수는 실수만" 으로
#   묶으면 그 하나에서 틀린다.
#
#   `random_` 은 어느 형에서든 돌고 **범위가 형에 달렸다** — int64 는 그 형의
#   최대까지, bool 은 {0,1} 이다.
_CONTINUOUS_REFUSAL = {
    "normal_": ("NotImplementedError", '"normal_kernel_cpu" not implemented for'),
    "uniform_": ("NotImplementedError", '"check_uniform_bounds" not implemented for'),
    "log_normal_": ("NotImplementedError", '"log_normal_cpu" not implemented for'),
    "exponential_": ("RuntimeError",
                     "Exponential distribution is a continuous probability "
                     "distribution. dtype must be a floating point but you "
                     "specified"),
    "cauchy_": ("RuntimeError",
                "Cauchy distribution is a continuous probability distribution. "
                "dtype must be a floating point but you specified"),
}


def _refuse_leaf(self, name):
    if self.requires_grad and _grad_mode.enabled:
        raise RuntimeError(_like_torch(
            f"기울기가 필요한 잎 텐서에는 `{name}` 을(를) 쓸 수 없습니다. "
            "`with torch.no_grad():` 안에서 하세요.",
            "a leaf Variable that requires grad is being used in an in-place operation"))


def _needs_continuous(self, name):
    """연속 분포는 실수 칸에만 채운다 — **예외 종류가 분포마다 다르다.**"""
    if self.data.dtype.kind == "f":
        return
    kind, phrase = _CONTINUOUS_REFUSAL[name]
    shown = _TYPE_NAMES.get(self.data.dtype.kind, "Long")
    error = NotImplementedError if kind == "NotImplementedError" else RuntimeError
    raise error(_like_torch(
        f"`{name}` 은 실수 텐서에만 채웁니다 ({self.dtype} 을 받았습니다).",
        f"{phrase} '{shown}'"))


def _fill_from(self, name, draw):
    _refuse_leaf(self, name)
    if name in _CONTINUOUS_REFUSAL:
        _needs_continuous(self, name)
    self.data[...] = _np.asarray(draw(_rng, self.data.shape),
                                 dtype=self.data.dtype)
    return self


def _normal_(self, mean=0.0, std=1.0, generator=None):
    del generator
    if std < 0:
        raise RuntimeError(_like_torch(
            f"normal_ 의 표준편차는 0 이상이어야 합니다 ({std} 을 받았습니다).",
            f"normal expects std >= 0.0, but found std {std}"))
    return _fill_from(self, "normal_", lambda r, s: r.normal(mean, std, s))


def _uniform_(self, from_=0.0, to=1.0, generator=None):
    del generator
    if from_ > to:
        raise RuntimeError(_like_torch(
            f"uniform_ 은 [from, to) 를 받습니다 ({from_}, {to} 을 받았습니다).",
            f"uniform_ expects to return a [from, to) range, but found from={from_} > to={to}"))
    return _fill_from(self, "uniform_", lambda r, s: r.uniform(from_, to, s))


def _exponential_(self, lambd=1.0, generator=None):
    del generator
    if lambd <= 0:
        raise RuntimeError(_like_torch(
            f"exponential_ 의 lambda 는 0 보다 커야 합니다 ({lambd} 을 받았습니다).",
            f"exponential_ expects lambda > 0.0, but found lambda={lambd}"))
    return _fill_from(self, "exponential_",
                      lambda r, s: r.exponential(1.0 / lambd, s))


def _cauchy_(self, median=0.0, sigma=1.0, generator=None):
    del generator
    return _fill_from(self, "cauchy_",
                      lambda r, s: median + sigma * r.standard_cauchy(s))


def _log_normal_(self, mean=1.0, std=2.0, generator=None):
    del generator
    return _fill_from(self, "log_normal_", lambda r, s: r.lognormal(mean, std, s))


def _geometric_(self, p, generator=None):
    """**이산이라 정수 텐서에서도 돈다.** 연속 다섯과 갈리는 하나다."""
    del generator
    if not 0 < p < 1:
        raise RuntimeError(_like_torch(
            f"geometric_ 의 p 는 (0, 1) 안이어야 합니다 ({p} 을 받았습니다).",
            f"geometric_ expects p to be in (0, 1), but got p={p}"))
    return _fill_from(self, "geometric_", lambda r, s: r.geometric(p, s))


def _random_(self, from_=0, to=None, generator=None):
    """**범위가 형에 달렸다** — 안 주면 그 형이 담을 수 있는 데까지다."""
    del generator
    kind = self.data.dtype.kind
    if to is None:
        to = 2 if kind == "b" else (1 << 53 if kind == "f" else 1 << 62)
    if from_ >= to:
        raise RuntimeError(_like_torch(
            f"random_ 의 from 은 to 보다 작아야 합니다 ({from_}, {to} 을 받았습니다).",
            f"random_ expects 'from' to be less than 'to', but got from={from_} >= to={to}"))
    return _fill_from(self, "random_",
                      lambda r, s: r.integers(from_, to, s))


for _rname, _rfn in (("normal_", _normal_), ("uniform_", _uniform_),
                     ("exponential_", _exponential_), ("cauchy_", _cauchy_),
                     ("log_normal_", _log_normal_), ("geometric_", _geometric_),
                     ("random_", _random_)):
    setattr(Tensor, _rname, _rfn)
del _rname, _rfn
