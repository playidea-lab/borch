"""`torch.fft` — the discrete Fourier transform.

**It stands on complex numbers.** These names were absent for a long time because
of the convention rather than the storage (`stft`'s refusal said as much), and
with the convention settled that door opens.

## Gradients — derived, then matched against measurement

The Fourier transform is **linear**. `y = F·x` with `F[k,j] = e^{-2πijk/n}`, so
under the complex gradient convention (`z.grad = ∂L/∂re + i·∂L/∂im`) the rule
that a holomorphic function's backward is `conj(f')·g` applies directly —

    grad_x[j] = Σ_k conj(F[k,j]) · g[k] = Σ_k e^{+2πijk/n} · g[k]

which is **the unnormalised inverse.** So forward and backward use the same
machinery. Measurement backs it: the gradient of `fft(x).real.sum()` is
`[n, 0, 0, …]` (for real x, since `Σ_k Re(X_k) = n·x_0`).

`rfft` receives gradient on the stored half only, so it **does not add the
conjugate partner** — measured: at n=6 the gradient of `rfft(x).real.sum()` is
`[4, 0, 1, 0, 1, 0]`, which is `Re(Σ_{k=0}^{3} e^{2πijk/6})`. Counting the half
twice gives `[8, …]`.

`irfft` is the reverse and has to count **the edges once and the middle twice** —
because the conjugate partners revived by Hermitian symmetry came from the same
stored cells. Only `k=0` and (for even n) `k=n/2` carry weight 1; the rest carry
2.

## `norm`

`None` is the same as `"backward"` (measured). 1 on the forward transform and
1/n on the inverse. `"forward"` is the reverse, and `"ortho"` is 1/√n on both.
"""

import numpy as _np

from ._base import _like_torch
from ._tensor import Tensor

_TENSOR_ARGS = (Tensor,)


def _wrap(t):
    """To a tensor if it is not one. `_ops` has the same name, and **importing
    from there is a cycle** (that side imports this file), so one line is
    repeated here."""
    return t if isinstance(t, _TENSOR_ARGS) else Tensor(_np.asarray(t))

_TWO_PI = 2.0 * _np.pi


def _norm_scale(norm, n, inverse):
    """A `norm` name to the factor. A wrong name stops — quietly using 1 makes
    the values diverge."""
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
    """One axis to length `n`. Shorter is **zero-padded** and longer is trimmed
    (measured)."""
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
    """One unnormalised DFT. **Every name here uses this one.**

    `sign` of −1 is the forward transform and +1 the inverse. With `hermitian`,
    the stored half is revived by conjugation into length `n` (`irfft` and its
    backward).

    It loops by hand rather than calling numpy's `fft` because **each place needs
    a different combination** — half input, half output, sign and scale attach
    differently for every name. Through numpy each combination gains code that
    trims and joins on both ends, and that is the easier way to be wrong.
    """
    data = _np.asarray(arr)
    moved = _np.moveaxis(data, axis, -1)
    have = moved.shape[-1]
    flat = moved.reshape(-1, have)
    if hermitian:
        # The stored half to length n. `X[n-k] = conj(X[k])`, and 0 and (for
        # even n) n/2 are their own partners and do not fold.
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
    """To a numpy array, complex or real. The dtype is not examined here."""
    return _np.asarray(_wrap(t).data)


def fft(input, n=None, dim=-1, norm=None):                      # noqa: A002
    """The forward transform. Real input still gives **complex output**
    (measured)."""
    t = _wrap(input)
    axis = _axis(dim, t.data.ndim)
    length = t.data.shape[axis] if n is None else int(n)
    scale = _norm_scale(norm, length, inverse=False)
    out = _dft(t.data, length, axis, -1.0, scale).astype(_np.complex64)

    def back(g, ax=axis, ln=length, sc=scale):
        # **The unnormalised inverse.** The forward scale multiplies through
        # unchanged — it is linear.
        got = _dft(_np.asarray(g), ln, ax, +1.0, sc)
        got = _resize(got, t.data.shape[ax], ax)
        return (got.astype(_np.complex64)
                if t.data.dtype.kind == "c" else _np.real(got).astype(_np.float32),)

    return t._make(out, (t,), back, "FftC2CBackward0")


def ifft(input, n=None, dim=-1, norm=None):                     # noqa: A002
    """The inverse. With `norm=None` a 1/n is attached (measured)."""
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
    """The real forward transform. **Only `n//2+1` cells are stored** — the rest
    are conjugates and are not carried."""
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
        # **The conjugate partner is not added.** Gradient arrives on the stored
        # half only, and the unstored half never entered the loss to begin with.
        # Adding it gives twice the measured `[4,0,1,0,1,0]`.
        got = _dft(_np.asarray(g), ln, ax, +1.0, sc, out_bins=ln)
        got = _resize(_np.real(got), t.data.shape[ax], ax)
        return (got.astype(_np.float32),)

    return t._make(out, (t,), back, "FftR2CBackward0")


def irfft(input, n=None, dim=-1, norm=None):                    # noqa: A002
    """A half spectrum to reals. Without `n` it is `2*(m-1)` (measured)."""
    t = _wrap(input)
    axis = _axis(dim, t.data.ndim)
    have = t.data.shape[axis]
    length = 2 * (have - 1) if n is None else int(n)
    scale = _norm_scale(norm, length, inverse=True)
    out = _np.real(_dft(t.data, length, axis, +1.0, scale,
                        out_bins=length, hermitian=True)).astype(_np.float32)

    def back(g, ax=axis, ln=length, sc=scale, m=have):
        # **The edges once and the middle twice.** The revived conjugate
        # partners came from the same stored cells, so gradient arrives at those
        # cells twice. Only `k=0` and, for even n, `k=n/2` are their own
        # conjugates and arrive once — counting those two twice is wrong by a
        # factor of two at the edges alone.
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


# ── multi-axis and Hermitian — **all assembled from the four above** ────────
#
# Not one new kernel. `fft2` is looping the axes one at a time (measured:
# `fft2(x)` and `fft(fft(x, dim=-1), dim=-2)` are exactly equal), and the
# Hermitian branch resolves into a conjugate and a scale
# (`hfft(c, n) = irfft(conj(c), n)·n`, `ihfft(r) = conj(rfft(r))/n` — both
# measured).
#
# **So no gradients are written here.** Being assembled, the tape carries
# through — writing a backward by hand here makes two copies alongside the four
# above, and the day they diverge a value comparison does not catch it.
#
# `norm` comes out right multiplied per axis too. `ortho` is `1/√nᵢ` per axis, so
# the product is `1/√Πnᵢ`, and `forward` is `1/nᵢ`, so the product is `1/Πnᵢ` —
# all three are multiplications, so splitting the axes gives the same answer.

def _axes_and_sizes(t, s, dim, default_last):
    """Unpack `s` and `dim` into a list of axes and a list of sizes. Follows
    torch's defaults."""
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
    """The forward transform over every (or the chosen) axis. **One `fft` per
    axis.**"""
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
    """Real input. **`rfft` on the last axis and `fft` on the rest** — the order
    decides the answer."""
    t = _wrap(input)
    axes, sizes = _axes_and_sizes(t, s, dim, True)
    t = rfft(t, n=sizes[-1], dim=axes[-1], norm=norm)
    for a, n in zip(axes[:-1], sizes[:-1]):
        t = fft(t, n=n, dim=a, norm=norm)
    return t


def irfftn(input, s=None, dim=None, norm=None):                  # noqa: A002
    """The inverse of `rfftn`. **`ifft` first and `irfft` last.**"""
    t = _wrap(input)
    axes, sizes = _axes_and_sizes(t, s, dim, True)
    if s is None:
        # Only the last axis is a half, so its size is restored — the same as
        # torch's default.
        sizes[-1] = 2 * (t.data.shape[axes[-1]] - 1)
    for a, n in zip(axes[:-1], sizes[:-1]):
        t = ifft(t, n=n, dim=a, norm=norm)
    return irfft(t, n=sizes[-1], dim=axes[-1], norm=norm)


def rfft2(input, s=None, dim=(-2, -1), norm=None):               # noqa: A002
    return rfftn(input, s=s, dim=dim, norm=norm)


def irfft2(input, s=None, dim=(-2, -1), norm=None):              # noqa: A002
    return irfftn(input, s=s, dim=dim, norm=norm)


def _flip_norm(norm):
    """The Hermitian branch swaps forward and inverse — the normalisation name
    flips with it."""
    return {"forward": "backward", "backward": "forward"}.get(norm or "backward",
                                                              norm)


def hfft(input, n=None, dim=-1, norm=None):                      # noqa: A002
    """Hermitian-symmetric complex input to **real output.**

    The conjugate relation of `irfft` (measured:
    `hfft(c, n) == irfft(conj(c), n)·n`). The scale undoes the `1/n` `irfft`
    attaches, and `norm` flips along with the swapped forward and inverse.
    """
    from . import _ops                                       # noqa: PLC0415

    t = _wrap(input)
    axis = _axis(dim, t.data.ndim)
    length = 2 * (t.data.shape[axis] - 1) if n is None else int(n)
    return irfft(_ops.conj(t), n=length, dim=axis, norm=_flip_norm(norm))


def ihfft(input, n=None, dim=-1, norm=None):                     # noqa: A002
    """Real input to **Hermitian-symmetric complex.** The conjugate of
    `rfft`."""
    from . import _ops                                       # noqa: PLC0415

    t = _wrap(input)
    axis = _axis(dim, t.data.ndim)
    return _ops.conj(rfft(t, n=n, dim=axis, norm=_flip_norm(norm)))


def hfftn(input, s=None, dim=None, norm=None):                   # noqa: A002
    """`hfft` on the last axis and **`fft` on the ones before it.**

    **Guessed backwards, and caught.** As the mirror of `rfftn` it looks like it
    should be `ifft`, and torch uses `fft` (measured — both candidates were built
    and compared). The shape matches either way, so without measuring the values
    it does not surface.
    """
    t = _wrap(input)
    axes, sizes = _axes_and_sizes(t, s, dim, True)
    if s is None:
        sizes[-1] = 2 * (t.data.shape[axes[-1]] - 1)
    for a, n in zip(axes[:-1], sizes[:-1]):
        t = fft(t, n=n, dim=a, norm=norm)
    return hfft(t, n=sizes[-1], dim=axes[-1], norm=norm)


def ihfftn(input, s=None, dim=None, norm=None):                  # noqa: A002
    """`ihfft` on the last axis and **`ifft` on the ones before it** (measured —
    the partner of `hfftn`)."""
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
    """The sample frequencies. `[0, 1, …, n/2-1, -n/2, …, -1] / (n·d)`
    (measured)."""
    half = (n - 1) // 2 + 1
    out = _np.empty(n, dtype=_np.float32)
    out[:half] = _np.arange(half)
    out[half:] = _np.arange(-(n // 2), 0)
    return Tensor(out / (n * d))


def rfftfreq(n, d=1.0, **kw):
    """The frequencies of the cells `rfft` produces. No negatives, and length
    `n//2+1`."""
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
    """Zero frequency to the middle. **Shifted by `n//2`** (measured — odd
    lengths included)."""
    return _roll(input, dim, lambda n: n // 2)


def ifftshift(input, dim=None):                                 # noqa: A002
    """The reverse of `fftshift`. **Undoing it with `n//2` is wrong at odd
    lengths** — it is `(n+1)//2`."""
    return _roll(input, dim, lambda n: -(n // 2))


# ------------------------------------------------------- short-time transforms
#
# **`stft` is an assembly, not a new kernel.** Slice (into frames), multiply by
# the window, `rfft`. All three are already differentiable names, so stacked this
# way **the gradient comes out right on its own** — confirmed by measurement (the
# gradient of `stft(x).abs().sum()` matches torch).
#
# Writing the kernel by hand, the forward pass comes out right quickly and the
# backward has to travel through the window and the overlap, so getting it wrong
# leaves plausible values and training that does not train. Assembly removes that
# place entirely.

def _window_of(window, n_fft, win_length, like):
    """The window to length `n_fft`. **Shorter is centred and zero-padded on
    both sides** (measured)."""
    from . import _ops

    if window is None:
        # torch warns here (a rectangular window leaks across the spectrum). The
        # values are the same.
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
         return_complex=None, align_to_window=None):
    """The short-time Fourier transform. The result is `(…, bins, frames)` —
    **frames last.**

    **`align_to_window` is accepted, and torch's own refusal is the whole of what
    it does here.** torch rejects it outright unless `center=False` — *"stft
    align_to_window should only be set when center = false"* — and with
    `center=False` it produced the identical answer at every setting tried, `True`
    and `False` alike. So there is nothing to imitate: the observable behaviour is
    the refusal, and that is mirrored exactly.
    
    Recorded rather than left out, because *the seat is torch's*. The alternative —
    leaving the argument off — would put a positional call that reaches it on
    nothing, which is the difference this whole axis exists to remove.

    **It refuses without `return_complex`.** torch does the same on real input
    (measured) — the old path producing a real `(…, 2)` is slated for removal, so
    choosing a default here would teach a shape that is about to disappear.
    """
    if align_to_window is not None and center:
        raise RuntimeError(
            "stft align_to_window should only be set when center = false")
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
        # **`n_fft//2` on each side**, reflected by default (measured).
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
    # `(…, frames, bins)` → `(…, bins, frames)`. torch puts the bins first.
    return _ops.swapaxes(spec, -1, -2)


def istft(input, n_fft, hop_length=None, win_length=None, window=None,   # noqa: A002
          center=True, normalized=False, onesided=None, length=None,
          return_complex=False):
    """The inverse of `stft`. **Divided by the overlapped window squared** —
    without that division the overlapped regions swell by the window weight.
    """
    from . import _ops

    t = _wrap(input)
    hop = n_fft // 4 if hop_length is None else int(hop_length)
    win_length = n_fft if win_length is None else int(win_length)
    if onesided is None:
        onesided = t.data.shape[-2] == n_fft // 2 + 1
    count = t.data.shape[-1]
    spec = _ops.swapaxes(t, -1, -2)                        # (…, frames, bins)
    frames = irfft(spec, n=n_fft, dim=-1) if onesided else fft(spec, dim=-1)
    if normalized:
        frames = frames * _np.sqrt(n_fft)
    win = _window_of(window, n_fft, win_length, t)
    frames = frames * win

    total = n_fft + hop * (count - 1)
    # **Overlap-add.** Each frame is zero-padded into position and all of them
    # are summed — this works without a kernel that scatters pieces, and the
    # backward follows along.
    pieces, envelope = [], _np.zeros(total, dtype=_np.float32)
    wsq = _np.asarray(win.data) ** 2
    for k in range(count):
        left, right = k * hop, total - n_fft - k * hop
        pieces.append(_ops.pad(_ops.select(frames, -2, k), [left, right]))
        envelope[left:left + n_fft] += wsq
    out = pieces[0]
    for piece in pieces[1:]:
        out = out + piece
    # No division by zero. torch does not leave that place empty either.
    safe = _np.where(_np.abs(envelope) < 1e-11, 1.0, envelope)
    out = out / Tensor(safe.astype(_np.float32))
    if center:
        out = _ops.narrow(out, -1, n_fft // 2, total - 2 * (n_fft // 2))
    if length is not None:
        out = _ops.narrow(out, -1, 0, int(length))
    return out
