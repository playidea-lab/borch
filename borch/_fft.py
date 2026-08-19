"""`torch.fft` — 이산 푸리에 변환.

**복소수 위에 선다.** 이 이름들이 오래 없었던 이유는 저장이 아니라 규약이었고
(`stft` 의 거절문이 그렇게 적혀 있었다), 규약이 정해진 지금 그 문이 열린다.

## 기울기 — 유도해서 실측과 맞췄다

푸리에 변환은 **선형**이다. `y = F·x` 이고 `F[k,j] = e^{-2πijk/n}` 이므로, 복소수
기울기 규약(`z.grad = ∂L/∂re + i·∂L/∂im`)에서 정칙 함수의 역방향이 `conj(f')·g`
라는 규칙이 그대로 걸린다 —

    grad_x[j] = Σ_k conj(F[k,j]) · g[k] = Σ_k e^{+2πijk/n} · g[k]

즉 **정규화 없는 역변환**이다. 그래서 순방향과 역방향이 같은 기계를 쓴다. 실측이
그것을 받친다: `fft(x).real.sum()` 의 기울기가 `[n, 0, 0, …]` 이다(x 가 실수일 때
`Σ_k Re(X_k) = n·x_0` 이므로).

`rfft` 는 저장된 반쪽에만 기울기가 오므로 **켤레 짝을 더하지 않는다** — 실측:
n=6 에서 `rfft(x).real.sum()` 의 기울기가 `[4, 0, 1, 0, 1, 0]` 이고, 이것은
`Re(Σ_{k=0}^{3} e^{2πijk/6})` 이다. 반쪽을 두 배로 세면 `[8, …]` 이 나온다.

`irfft` 는 반대로 **가장자리만 한 번, 가운데는 두 번** 세어야 한다 — 허미시안
대칭으로 되살린 켤레 짝이 같은 저장 칸에서 왔기 때문이다. `k=0` 과 (짝수 n 의)
`k=n/2` 만 무게가 1 이고 나머지가 2 다.

## `norm`

`None` 은 `"backward"` 와 같다(실측). 정변환에 1, 역변환에 1/n. `"forward"` 는
반대, `"ortho"` 는 양쪽에 1/√n.
"""

import numpy as _np

from ._base import _like_torch
from ._tensor import Tensor

_TENSOR_ARGS = (Tensor,)


def _wrap(t):
    """텐서가 아니면 텐서로. `_ops` 에 같은 이름이 있지만 **거기서 들여오면 순환**이라
    (그쪽이 이 파일을 들여온다) 한 줄을 여기 다시 둔다."""
    return t if isinstance(t, _TENSOR_ARGS) else Tensor(_np.asarray(t))

_TWO_PI = 2.0 * _np.pi


def _norm_scale(norm, n, inverse):
    """`norm` 이름 → 곱할 값. 이름이 틀리면 멈춘다 — 조용히 1 을 쓰면 값이 갈린다."""
    if norm is None or norm == "backward":
        return (1.0 / n) if inverse else 1.0
    if norm == "forward":
        return 1.0 if inverse else (1.0 / n)
    if norm == "ortho":
        return 1.0 / _np.sqrt(n)
    raise RuntimeError(_like_torch(
        f"unknown normalization name: {norm!r} — one of backward, forward, ortho.",
        f'Invalid normalization mode: "{norm}"'))


def _axis(dim, rank):
    return dim + rank if dim < 0 else dim


def _resize(arr, n, axis):
    """축 하나를 길이 `n` 으로. 짧으면 **0 으로 채우고** 길면 자른다(실측)."""
    have = arr.shape[axis]
    if have == n:
        return arr
    if have > n:
        cut = [slice(None)] * arr.ndim
        cut[axis] = slice(0, n)
        return arr[tuple(cut)]
    pad = [(0, 0)] * arr.ndim
    pad[axis] = (0, n - have)
    return _np.pad(arr, pad)


def _dft(arr, n, axis, sign, scale, out_bins=None, hermitian=False):
    """정규화 없는 DFT 한 판. **모든 이름이 이것 하나를 쓴다.**

    `sign` 이 −1 이면 정변환, +1 이면 역변환이다. `hermitian` 이면 저장된 반쪽을
    켤레로 되살려 길이 `n` 을 만든다(`irfft` 와 그 역방향).

    numpy 의 `fft` 를 부르지 않고 손으로 도는 이유는 **자리마다 다른 조합이 필요해서**
    다 — 반쪽 입력, 반쪽 출력, 부호, 배율이 이름마다 다르게 붙는다. numpy 로는 그
    조합마다 앞뒤로 자르고 붙이는 코드가 붙고, 그쪽이 오히려 틀리기 쉽다.
    """
    data = _np.asarray(arr)
    moved = _np.moveaxis(data, axis, -1)
    have = moved.shape[-1]
    flat = moved.reshape(-1, have)
    if hermitian:
        # 저장된 반쪽 → 길이 n. `X[n-k] = conj(X[k])` 이고 0 과 (짝수 n 의) n/2 는
        # 자기 자신이라 안 접힌다.
        full = _np.zeros((flat.shape[0], n), dtype=_np.complex128)
        take = min(have, n // 2 + 1)
        full[:, :take] = flat[:, :take]
        for k in range(1, (n + 1) // 2):
            if k < take:
                full[:, n - k] = _np.conj(flat[:, k])
        flat = full
        have = n
    else:
        flat = _resize(flat, n, -1)
        have = n
    bins = n if out_bins is None else out_bins
    j = _np.arange(have)[None, :]
    k = _np.arange(bins)[:, None]
    kernel = _np.exp(sign * 1j * _TWO_PI * (k * j) / n)
    out = flat.astype(_np.complex128) @ kernel.T
    out = out * scale
    shaped = out.reshape(moved.shape[:-1] + (bins,))
    return _np.moveaxis(shaped, -1, axis)


def _leaf(t):
    """복소수든 실수든 numpy 배열로. 형은 여기서 안 본다."""
    return _np.asarray(_wrap(t).data)


def fft(input, n=None, dim=-1, norm=None):                      # noqa: A002
    """정변환. 실수를 넣어도 **복소수가 나온다**(실측)."""
    t = _wrap(input)
    axis = _axis(dim, t.data.ndim)
    length = t.data.shape[axis] if n is None else int(n)
    scale = _norm_scale(norm, length, inverse=False)
    out = _dft(t.data, length, axis, -1.0, scale).astype(_np.complex64)

    def back(g, ax=axis, ln=length, sc=scale):
        # **정규화 없는 역변환.** 배율은 순방향의 것이 그대로 곱해진다 — 선형이라.
        got = _dft(_np.asarray(g), ln, ax, +1.0, sc)
        got = _resize(got, t.data.shape[ax], ax)
        return (got.astype(_np.complex64)
                if t.data.dtype.kind == "c" else _np.real(got).astype(_np.float32),)

    return t._make(out, (t,), back, "FftC2CBackward0")


def ifft(input, n=None, dim=-1, norm=None):                     # noqa: A002
    """역변환. `norm=None` 이면 1/n 이 붙는다(실측)."""
    t = _wrap(input)
    axis = _axis(dim, t.data.ndim)
    length = t.data.shape[axis] if n is None else int(n)
    scale = _norm_scale(norm, length, inverse=True)
    out = _dft(t.data, length, axis, +1.0, scale).astype(_np.complex64)

    def back(g, ax=axis, ln=length, sc=scale):
        got = _dft(_np.asarray(g), ln, ax, -1.0, sc)
        got = _resize(got, t.data.shape[ax], ax)
        return (got.astype(_np.complex64)
                if t.data.dtype.kind == "c" else _np.real(got).astype(_np.float32),)

    return t._make(out, (t,), back, "FftC2CBackward0")


def rfft(input, n=None, dim=-1, norm=None):                     # noqa: A002
    """실수 정변환. **저장은 `n//2+1` 칸뿐**이다 — 나머지는 켤레라 안 든다."""
    t = _wrap(input)
    if t.data.dtype.kind == "c":
        raise RuntimeError(_like_torch(
            "rfft takes a real input — use `fft` for a complex one.",
            "rfft expects a real input tensor, but got complex"))
    axis = _axis(dim, t.data.ndim)
    length = t.data.shape[axis] if n is None else int(n)
    scale = _norm_scale(norm, length, inverse=False)
    bins = length // 2 + 1
    out = _dft(t.data, length, axis, -1.0, scale, out_bins=bins).astype(_np.complex64)

    def back(g, ax=axis, ln=length, sc=scale, m=bins):
        # **켤레 짝을 안 더한다.** 저장된 반쪽에만 기울기가 오고, 안 저장된 반쪽은
        # 애초에 손실에 안 들어갔다. 더하면 실측(`[4,0,1,0,1,0]`)의 두 배가 나온다.
        got = _dft(_np.asarray(g), ln, ax, +1.0, sc, out_bins=ln)
        got = _resize(_np.real(got), t.data.shape[ax], ax)
        return (got.astype(_np.float32),)

    return t._make(out, (t,), back, "FftR2CBackward0")


def irfft(input, n=None, dim=-1, norm=None):                    # noqa: A002
    """반쪽 스펙트럼 → 실수. `n` 을 안 주면 `2*(m-1)` 이다(실측)."""
    t = _wrap(input)
    axis = _axis(dim, t.data.ndim)
    have = t.data.shape[axis]
    length = 2 * (have - 1) if n is None else int(n)
    scale = _norm_scale(norm, length, inverse=True)
    out = _np.real(_dft(t.data, length, axis, +1.0, scale,
                        out_bins=length, hermitian=True)).astype(_np.float32)

    def back(g, ax=axis, ln=length, sc=scale, m=have):
        # **가장자리는 한 번, 가운데는 두 번.** 되살린 켤레 짝이 같은 저장 칸에서
        # 왔으므로 그 칸에 기울기가 두 번 도착한다. `k=0` 과 짝수 n 의 `k=n/2` 만
        # 자기 켤레라 한 번이다 — 그 둘을 두 번 세면 가장자리만 두 배로 틀린다.
        got = _dft(_np.asarray(g), ln, ax, -1.0, sc, out_bins=ln)
        got = _resize(got, m, ax)
        weight = _np.full(m, 2.0)
        weight[0] = 1.0
        if ln % 2 == 0 and m > ln // 2:
            weight[ln // 2] = 1.0
        shape = [1] * got.ndim
        shape[ax] = m
        return ((got * weight.reshape(shape)).astype(_np.complex64),)

    return t._make(out, (t,), back, "FftC2RBackward0")


# ── 여러 축 · 에르미트 — **전부 위 넷의 조립이다** ─────────────────────────
#
# 새 커널이 하나도 없다. `fft2` 는 축을 하나씩 도는 것이고(실측: `fft2(x)` 와
# `fft(fft(x, dim=-1), dim=-2)` 가 정확히 같다), 에르미트 갈래도 켤레와 배율로
# 풀린다(`hfft(c, n) = irfft(conj(c), n)·n`, `ihfft(r) = conj(rfft(r))/n` — 둘 다 실측).
#
# **그래서 기울기를 따로 안 쓴다.** 조립이므로 테이프가 그대로 이어진다 — 여기에
# 역방향을 손으로 적으면 위 넷과 두 벌이 되고, 그 둘이 갈리는 날 값 대조로는 안 걸린다.
#
# `norm` 도 축마다 곱해지면 맞는다. `ortho` 는 축마다 `1/√nᵢ` 라 곱이 `1/√Πnᵢ` 이고,
# `forward` 는 `1/nᵢ` 라 곱이 `1/Πnᵢ` 다 — 셋 다 곱셈이라 축을 나눠 돌아도 같다.

def _axes_and_sizes(t, s, dim, default_last):
    """`s`·`dim` 을 축 목록과 크기 목록으로 편다. torch 의 기본을 따른다."""
    rank = t.data.ndim
    if dim is None:
        dim = tuple(range(rank)) if s is None else tuple(
            range(rank - len(s), rank))
    elif isinstance(dim, int):
        dim = (dim,)
    axes = [_axis(d, rank) for d in dim]
    if s is None:
        sizes = [t.data.shape[a] for a in axes]
    else:
        sizes = [t.data.shape[a] if v is None else int(v)
                 for a, v in zip(axes, s)]
    del default_last
    return axes, sizes


def fftn(input, s=None, dim=None, norm=None):                    # noqa: A002
    """모든(또는 고른) 축의 정변환. **축마다 `fft` 를 한 번씩 돈다.**"""
    t = _wrap(input)
    axes, sizes = _axes_and_sizes(t, s, dim, False)
    for a, n in zip(axes, sizes):
        t = fft(t, n=n, dim=a, norm=norm)
    return t


def ifftn(input, s=None, dim=None, norm=None):                   # noqa: A002
    t = _wrap(input)
    axes, sizes = _axes_and_sizes(t, s, dim, False)
    for a, n in zip(axes, sizes):
        t = ifft(t, n=n, dim=a, norm=norm)
    return t


def fft2(input, s=None, dim=(-2, -1), norm=None):                # noqa: A002
    return fftn(input, s=s, dim=dim, norm=norm)


def ifft2(input, s=None, dim=(-2, -1), norm=None):               # noqa: A002
    return ifftn(input, s=s, dim=dim, norm=norm)


def rfftn(input, s=None, dim=None, norm=None):                   # noqa: A002
    """실수 입력. **마지막 축만 `rfft` 이고 나머지는 `fft`** 다 — 차례가 답을 정한다."""
    t = _wrap(input)
    axes, sizes = _axes_and_sizes(t, s, dim, True)
    t = rfft(t, n=sizes[-1], dim=axes[-1], norm=norm)
    for a, n in zip(axes[:-1], sizes[:-1]):
        t = fft(t, n=n, dim=a, norm=norm)
    return t


def irfftn(input, s=None, dim=None, norm=None):                  # noqa: A002
    """`rfftn` 의 역. **`ifft` 를 먼저 돌고 마지막에 `irfft`** 다."""
    t = _wrap(input)
    axes, sizes = _axes_and_sizes(t, s, dim, True)
    if s is None:
        # 마지막 축만 반쪽이라 크기를 되돌려 준다 — torch 의 기본과 같다.
        sizes[-1] = 2 * (t.data.shape[axes[-1]] - 1)
    for a, n in zip(axes[:-1], sizes[:-1]):
        t = ifft(t, n=n, dim=a, norm=norm)
    return irfft(t, n=sizes[-1], dim=axes[-1], norm=norm)


def rfft2(input, s=None, dim=(-2, -1), norm=None):               # noqa: A002
    return rfftn(input, s=s, dim=dim, norm=norm)


def irfft2(input, s=None, dim=(-2, -1), norm=None):              # noqa: A002
    return irfftn(input, s=s, dim=dim, norm=norm)


def _flip_norm(norm):
    """에르미트 갈래는 정·역이 뒤바뀐다 — 정규화 이름도 같이 뒤집는다."""
    return {"forward": "backward", "backward": "forward"}.get(norm or "backward",
                                                              norm)


def hfft(input, n=None, dim=-1, norm=None):                      # noqa: A002
    """에르미트 대칭인 복소수 입력 → **실수 출력.**

    `irfft` 의 켤레 관계다(실측: `hfft(c, n) == irfft(conj(c), n)·n`). 배율은
    `irfft` 가 붙이는 `1/n` 을 되돌리는 것이고, `norm` 은 정·역이 뒤바뀌므로 같이
    뒤집는다.
    """
    from . import _ops                                       # noqa: PLC0415

    t = _wrap(input)
    axis = _axis(dim, t.data.ndim)
    length = 2 * (t.data.shape[axis] - 1) if n is None else int(n)
    return irfft(_ops.conj(t), n=length, dim=axis, norm=_flip_norm(norm))


def ihfft(input, n=None, dim=-1, norm=None):                     # noqa: A002
    """실수 입력 → **에르미트 대칭인 복소수.** `rfft` 의 켤레다."""
    from . import _ops                                       # noqa: PLC0415

    t = _wrap(input)
    axis = _axis(dim, t.data.ndim)
    return _ops.conj(rfft(t, n=n, dim=axis, norm=_flip_norm(norm)))


def hfftn(input, s=None, dim=None, norm=None):                   # noqa: A002
    """마지막 축이 `hfft` 이고 **앞 축은 `fft`** 다.

    **앞뒤를 반대로 짐작했다가 걸렸다.** `rfftn` 의 거울이니 `ifft` 일 것 같은데
    torch 는 `fft` 다(실측 — 후보를 둘 다 만들어 대 봤다). 모양은 양쪽 다 맞아서
    값을 안 재면 안 드러난다.
    """
    t = _wrap(input)
    axes, sizes = _axes_and_sizes(t, s, dim, True)
    if s is None:
        sizes[-1] = 2 * (t.data.shape[axes[-1]] - 1)
    for a, n in zip(axes[:-1], sizes[:-1]):
        t = fft(t, n=n, dim=a, norm=norm)
    return hfft(t, n=sizes[-1], dim=axes[-1], norm=norm)


def ihfftn(input, s=None, dim=None, norm=None):                  # noqa: A002
    """`ihfft` 를 마지막 축에, **앞 축에는 `ifft`** 를(실측 — `hfftn` 과 짝이다)."""
    t = _wrap(input)
    axes, sizes = _axes_and_sizes(t, s, dim, True)
    t = ihfft(t, n=sizes[-1], dim=axes[-1], norm=norm)
    for a, n in zip(axes[:-1], sizes[:-1]):
        t = ifft(t, n=n, dim=a, norm=norm)
    return t


def hfft2(input, s=None, dim=(-2, -1), norm=None):               # noqa: A002
    return hfftn(input, s=s, dim=dim, norm=norm)


def ihfft2(input, s=None, dim=(-2, -1), norm=None):              # noqa: A002
    return ihfftn(input, s=s, dim=dim, norm=norm)


def fftfreq(n, d=1.0, **kw):
    """표본 주파수. `[0, 1, …, n/2-1, -n/2, …, -1] / (n·d)` 다(실측)."""
    half = (n - 1) // 2 + 1
    out = _np.empty(n, dtype=_np.float32)
    out[:half] = _np.arange(half)
    out[half:] = _np.arange(-(n // 2), 0)
    return Tensor(out / (n * d))


def rfftfreq(n, d=1.0, **kw):
    """`rfft` 가 내는 칸의 주파수. 음수가 없고 길이가 `n//2+1` 이다."""
    return Tensor(_np.arange(n // 2 + 1, dtype=_np.float32) / (n * d))


def _roll(t, dim, by):
    x = _wrap(t)
    axes = range(x.data.ndim) if dim is None else (
        [_axis(d, x.data.ndim) for d in dim] if isinstance(dim, (list, tuple))
        else [_axis(dim, x.data.ndim)])
    shifts = [by(x.data.shape[a]) for a in axes]
    out = _np.roll(x.data, shifts, axis=tuple(axes))
    return x._make(out, (x,),
                   lambda g: (_np.roll(_np.asarray(g), [-s for s in shifts],
                                       axis=tuple(axes)),),
                   "RollBackward0")


def fftshift(input, dim=None):                                  # noqa: A002
    """0 주파수를 가운데로. **`n//2` 만큼 민다**(실측 — 홀수에서도 그렇다)."""
    return _roll(input, dim, lambda n: n // 2)


def ifftshift(input, dim=None):                                 # noqa: A002
    """`fftshift` 의 반대. **홀수에서 `n//2` 로 되돌리면 안 맞는다** — `(n+1)//2` 다."""
    return _roll(input, dim, lambda n: -(n // 2))


# ---------------------------------------------------------------- 짧은 시간 변환
#
# **`stft` 는 새 커널이 아니라 조립이다.** 자르고(틀 나누기) · 창을 곱하고 · `rfft`.
# 셋 다 이미 미분되는 이름이라, 이렇게 쌓으면 **기울기가 저절로 맞는다** — 실측으로
# 확인했다(`stft(x).abs().sum()` 의 기울기가 torch 와 같다).
#
# 손으로 커널을 쓰면 순방향은 금방 맞는데 역방향이 창과 겹침을 다 통과해야 해서,
# 틀리면 값은 그럴듯하고 학습만 안 된다. 조립이 그 자리를 아예 없앤다.

def _window_of(window, n_fft, win_length, like):
    """창을 `n_fft` 길이로. **짧으면 가운데에 놓고 양쪽을 0 으로 채운다**(실측)."""
    from . import _ops

    if window is None:
        # torch 는 여기서 경고를 낸다(사각창은 스펙트럼이 샌다). 값은 같다.
        return _ops.ones([n_fft])
    win = _wrap(window)
    have = win.data.shape[-1]
    if have == n_fft:
        return win
    if have > n_fft:
        raise RuntimeError(_like_torch(
            f"the window is longer than n_fft: {have} > {n_fft}",
            "window length should be less than or equal to n_fft"))
    left = (n_fft - have) // 2
    return _ops.pad(win, [left, n_fft - have - left])


def stft(input, n_fft, hop_length=None, win_length=None, window=None,     # noqa: A002
         center=True, pad_mode="reflect", normalized=False, onesided=None,
         return_complex=None):
    """짧은 시간 푸리에 변환. 결과는 `(…, 칸, 틀)` 이다 — **틀이 마지막**이다.

    **`return_complex` 를 안 주면 거절한다.** torch 가 실수 입력에서 그렇게 한다
    (실측) — 실수 `(…, 2)` 로 내는 옛 길이 폐기 예정이라, 기본값을 정해 주면 곧
    사라질 모양을 가르치게 된다.
    """
    from . import _ops

    t = _wrap(input)
    if return_complex is None and t.data.dtype.kind != "c":
        raise RuntimeError(_like_torch(
            "stft needs return_complex to be given — the old path that returns a real "
            "(…, 2) tensor is deprecated in torch.",
            "stft requires the return_complex parameter be given for real inputs"))
    if return_complex is False:
        raise RuntimeError(_like_torch(
            "return_complex=False is not here — it is deprecated in torch too.",
            "stft with return_complex=False is deprecated"))
    hop = n_fft // 4 if hop_length is None else int(hop_length)
    win_length = n_fft if win_length is None else int(win_length)
    if onesided is None:
        onesided = t.data.dtype.kind != "c"

    x = t
    if center:
        # **`n_fft//2` 씩 양쪽**, 기본은 반사다(실측).
        pad = n_fft // 2
        flat = x.data.ndim == 1
        if flat:
            x = _ops.reshape(x, [1, -1])
        x = _ops.pad(x, [pad, pad], mode=pad_mode)
        if flat:
            x = _ops.reshape(x, [-1])
    length = x.data.shape[-1]
    if length < n_fft:
        raise RuntimeError(_like_torch(
            f"the signal is shorter than n_fft: {length} < {n_fft}",
            "Expected size of signal to be at least n_fft"))
    count = 1 + (length - n_fft) // hop
    frames = _ops.stack([_ops.narrow(x, -1, k * hop, n_fft)
                         for k in range(count)], dim=-2)
    frames = frames * _window_of(window, n_fft, win_length, t)
    spec = rfft(frames, dim=-1) if onesided else fft(frames, dim=-1)
    if normalized:
        spec = spec * (1.0 / _np.sqrt(n_fft))
    # `(…, 틀, 칸)` → `(…, 칸, 틀)`. torch 가 칸을 앞에 둔다.
    return _ops.swapaxes(spec, -1, -2)


def istft(input, n_fft, hop_length=None, win_length=None, window=None,   # noqa: A002
          center=True, normalized=False, onesided=None, length=None,
          return_complex=False):
    """`stft` 의 되돌리기. **창의 제곱 겹침으로 나눈다** — 그 나눗셈이 없으면
    겹친 자리가 창 무게만큼 부풀어 오른다.
    """
    from . import _ops

    t = _wrap(input)
    hop = n_fft // 4 if hop_length is None else int(hop_length)
    win_length = n_fft if win_length is None else int(win_length)
    if onesided is None:
        onesided = t.data.shape[-2] == n_fft // 2 + 1
    count = t.data.shape[-1]
    spec = _ops.swapaxes(t, -1, -2)                        # (…, 틀, 칸)
    frames = irfft(spec, n=n_fft, dim=-1) if onesided else fft(spec, dim=-1)
    if normalized:
        frames = frames * _np.sqrt(n_fft)
    win = _window_of(window, n_fft, win_length, t)
    frames = frames * win

    total = n_fft + hop * (count - 1)
    # **겹쳐 더하기.** 틀마다 자리를 맞춰 0 으로 두르고 전부 더한다 — 조각을
    # 흩뿌리는 커널 없이 되고, 역방향이 그대로 따라온다.
    pieces, envelope = [], _np.zeros(total, dtype=_np.float32)
    wsq = _np.asarray(win.data) ** 2
    for k in range(count):
        left, right = k * hop, total - n_fft - k * hop
        pieces.append(_ops.pad(_ops.select(frames, -2, k), [left, right]))
        envelope[left:left + n_fft] += wsq
    out = pieces[0]
    for piece in pieces[1:]:
        out = out + piece
    # 0 으로 나누지 않는다. torch 도 그 자리를 비워 두지 않는다.
    safe = _np.where(_np.abs(envelope) < 1e-11, 1.0, envelope)
    out = out / Tensor(safe.astype(_np.float32))
    if center:
        out = _ops.narrow(out, -1, n_fft // 2, total - 2 * (n_fft // 2))
    if length is not None:
        out = _ops.narrow(out, -1, 0, int(length))
    return out
