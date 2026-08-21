"""borch-vision — a thin torchvision-shaped layer. **`transforms` and nothing
else.**

## Why a separate file

It is `torchvision.transforms` rather than `torch.transforms`. Put inside the core
it becomes `borch.transforms`, and that is **a place real torch does not have.**
Imitating the structure exactly is this project's point, and inventing a place
that does not exist breaks that point first.

    import borchvision as torchvision
    from borchvision import transforms

## Whose tensors it builds

`ToTensor` is the only thing that builds a tensor. The default is the core
`borch`, and attaching it to the sister library (WebGPU) is done by calling
`use(L)`.

## What is absent, and why

- **`datasets`** — the fetching side is blocked. Measured: `cs.toronto.edu` does
  not send `Access-Control-Allow-Origin`, so a browser cannot fetch the CIFAR
  originals. Not the kind of thing diligence fixes. And torch's `download=True`
  keeps it under `root` and reuses it on the next run, while Pyodide's filesystem
  is gone on a refresh — a place where the same code behaves differently with no
  exception, which must not be imitated. **Once the bytes are in hand it already
  works**
  (`fetch_cached`·`cache_put`·`TensorDataset`)
- **`ops`** — `nms` is short in numpy, so "it is large" would be a false reason.
  The real reason is that nobody stands in front of it. Detection needs a
  pre-trained backbone and COCO-scale data to reach the end, and until then `nms`
  stays a correct function with nowhere to go. A surface with no users grows no
  cases, and **surfaces with no cases were every place this repository was quietly
  wrong**
- **pre-trained weights** — a `.pth` is a pickle inside a zip and `torch.load`
  revives the classes through their module paths. Reading it means imitating
  torch's internal structure, and getting it subtly wrong brings in **the wrong
  numbers in correctly shaped weights.** ResNet-18 alone is 45MB on top of that,
  and above all, once `pretrained=True` runs people compare against the published
  top-1 — bit equivalence is this project's explicit non-goal, so that is a promise
  it cannot keep

## The random numbers differ from torch's — written down because they cannot be
imitated

`RandomCrop` and `RandomHorizontalFlip` cannot use torch's generator. So **the
same seed does not produce the same picture as torchvision's.** It is the draw
that diverges rather than the values. The places where the probability is pinned
at 0 or 1 are deterministic, so that is where the golden compares.
"""

import numpy as _np

_rng = _np.random.default_rng()
_lib = None


def use(L):
    """Choose which library `ToTensor` builds its tensors in. Called when
    attaching to the sister library."""
    global _lib
    _lib = L
    return L


def _backend():
    global _lib
    if _lib is None:
        import borch as _core
        _lib = _core
    return _lib


def manual_seed(seed):
    """**It does not give torch's picture.** It reproduces within this module
    alone."""
    global _rng
    _rng = _np.random.default_rng(seed)


# --- the skeleton: the axes arrive as arguments -----------------------------
# The per-image transforms handle (H,W,C) and the batch ones (N,C,H,W). Writing
# the same job twice because the positions differ eventually diverges, so they take
# **which axis** the height and the width are and share one copy.

def _hflip(arr, w_axis):
    return _np.flip(arr, axis=w_axis)


def _crop(arr, h_axis, top, height, w_axis, left, width):
    idx = [slice(None)] * arr.ndim
    idx[h_axis] = slice(top, top + height)
    idx[w_axis] = slice(left, left + width)
    return arr[tuple(idx)]


def _pad_hw(arr, h_axis, w_axis, padding, fill):
    if not padding:
        return arr
    pads = [(0, 0)] * arr.ndim
    pads[h_axis] = pads[w_axis] = (padding, padding)
    return _np.pad(arr, pads, constant_values=fill)


class Compose:
    def __init__(self, transforms):
        self.transforms = list(transforms)

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x

    def __repr__(self):
        inner = "".join(f"\n    {t}" for t in self.transforms)
        return f"{type(self).__name__}({inner}\n)"


class ToTensor:
    """(H,W,C) or (H,W) → a (C,H,W) tensor.

    **It divides by 255 for uint8 alone** — that is torchvision's rule. A float
    array goes through undivided. Missing that distinction divides data that is
    already in [0,1] once more and it comes out 255× darker, with no exception and
    training that does not train.
    """

    def __call__(self, pic):
        arr = _np.asarray(pic)
        if arr.ndim == 2:
            arr = arr[:, :, None]
        if arr.ndim != 3:
            raise ValueError(
                f"ToTensor takes (H,W) or (H,W,C) — it received {arr.shape}.\n"
                "(torch: pic should be 2/3 dimensional)")
        out = arr.transpose(2, 0, 1)
        out = (out.astype(_np.float32) / 255.0 if arr.dtype == _np.uint8
               else out.astype(_np.float32))
        return _backend().tensor(_np.ascontiguousarray(out))

    def __repr__(self):
        return f"{type(self).__name__}()"


class Normalize:
    """`(x - mean) / std` per channel. It takes tensors and numpy arrays alike.

    It takes numpy as well because there is a place that normalises a whole batch
    at once (see `augment_batch`). To avoid writing the same arithmetic twice, the
    axes are lined up and one formula is kept.
    """

    def __init__(self, mean, std, inplace=False):
        # It holds what it was given **as it was given.** torchvision does the
        # same, and that is where the repr printing a tuple as a tuple and a list
        # as a list comes from.
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, x):
        m = _np.atleast_1d(_np.asarray(self.mean, dtype=_np.float32))
        s = _np.atleast_1d(_np.asarray(self.std, dtype=_np.float32))
        shape = (m.size, 1, 1)
        m, s = m.reshape(shape), s.reshape(shape)
        if isinstance(x, _np.ndarray):
            return (x - m) / s
        L = _backend()
        return (x - L.tensor(m)) / L.tensor(s)

    def __repr__(self):
        return f"{type(self).__name__}(mean={self.mean}, std={self.std})"


class RandomHorizontalFlip:
    """Flip left to right. **It takes a (H,W,C) numpy array** — what arrives at
    this position in torchvision is a PIL image, and with no PIL here an array
    stands in.

    A tensor is refused. Making a tensor per image creates one GPU buffer per image
    on the sister library, and that is the kind that looks like it works and then
    collapses on memory. Telling the caller to reorder is better.
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        img = _require_hwc(img, type(self).__name__)
        if _rng.random() >= self.p:
            return img
        return _np.ascontiguousarray(_hflip(img, 1))

    def __repr__(self):
        return f"{type(self).__name__}(p={self.p})"


class RandomCrop:
    """Pad the edges and then crop at random. `RandomHorizontalFlip`'s place."""

    def __init__(self, size, padding=0, fill=0):
        self.size = (int(size), int(size)) if isinstance(size, int) else tuple(size)
        self.padding = int(padding)
        self.fill = fill

    def __call__(self, img):
        img = _require_hwc(img, type(self).__name__)
        img = _pad_hw(img, 0, 1, self.padding, self.fill)
        th, tw = self.size
        h, w = img.shape[0], img.shape[1]
        if h < th or w < tw:
            raise ValueError(
                f"The crop size {self.size} is larger than the image {(h, w)}.\n"
                "(torch: Required crop size is larger than input image size)")
        top = int(_rng.integers(0, h - th + 1))
        left = int(_rng.integers(0, w - tw + 1))
        return _np.ascontiguousarray(_crop(img, 0, top, th, 1, left, tw))

    def __repr__(self):
        return f"{type(self).__name__}(size={self.size}, padding={self.padding})"


def _require_hwc(img, who):
    if isinstance(img, _np.ndarray):
        return img
    raise TypeError(
        f"{who} takes a (H,W,C) numpy array — a tensor arrived.\n"
        "  What arrives at this position in torchvision is a PIL image. Put\n"
        "  `ToTensor()` after it: "
        "Compose([RandomCrop(...), RandomHorizontalFlip(), ToTensor()])")


def augment_batch(x, crop=None, padding=0, hflip_p=0.0, fill=0.0):
    """**Ours, absent from torchvision.** It augments an (N,C,H,W) batch at once.

    Why it is needed. A per-image `ToTensor` creates **one GPU buffer per image**
    on the sister library. Ten thousand images in an epoch is ten thousand of
    them. Augmenting the whole batch in numpy and then building **one** tensor is
    the only order that holds up, so that order is given a name and exposed.

    The draws happen **per image.** Applying one crop and one flip across the whole
    batch leaves nothing varied within the batch and the augmentation's effect
    disappears. torchvision's classes draw once per call (given a batch, the whole
    batch receives the same draw), which is where this parts from them — and why it
    carries our name rather than torchvision's.
    """
    if x.ndim != 4:
        raise ValueError(
            f"augment_batch takes (N,C,H,W) — it received {x.shape}")
    out = _pad_hw(x, 2, 3, padding, fill)
    n, _, h, w = out.shape
    th, tw = (h, w) if crop is None else (
        (int(crop), int(crop)) if isinstance(crop, int) else tuple(crop))
    if h < th or w < tw:
        raise ValueError(
            f"The crop size {(th, tw)} is larger than the image {(h, w)}")

    tops = _rng.integers(0, h - th + 1, size=n)
    lefts = _rng.integers(0, w - tw + 1, size=n)
    flips = _rng.random(n) < hflip_p
    pieces = _np.empty((n, x.shape[1], th, tw), dtype=x.dtype)
    for i in range(n):
        one = _crop(out[i], 1, int(tops[i]), th, 2, int(lefts[i]), tw)
        pieces[i] = _hflip(one, 2) if flips[i] else one
    return pieces


# --- resizing ---------------------------------------------------------------
#
# **The antialiased side was chosen.** torchvision's `Resize` defaults to
# `antialias=True`, and the difference between off and on is up to 0.0301 at
# 8×8→4×4 (measured). It is not zero, so writing only "bilinear" leaves it
# undecided, and feeding the off version to a model trained on the on version is a
# different input.
#
# The rule is the one PIL and torch use, exactly. For each output pixel the range
# looked at in the input widens by `support = max(1, downscale factor)`, the
# weights come from a triangle filter, and they are divided so that they sum to 1.
# Upscaling, the support is 1 and it becomes ordinary bilinear — which is why one
# branch is enough.
#
# Two boundary rules were measured (`floor(center - support)` and the same with
# `+0.5` added). **The answers are identical in every case** — because the triangle
# filter is 0 at the widened positions. The one matching torch's C implementation
# (`+0.5`) is used.


def _aa_weights(src, dst):
    """For each output position, (where to start reading, the weights). For one
    axis — the filter is separable."""
    scale = src / dst
    support = max(1.0, scale)
    rows = []
    for i in range(dst):
        center = (i + 0.5) * scale
        lo = int(max(0, _np.floor(center - support + 0.5)))
        hi = int(min(src, _np.ceil(center + support + 0.5)))
        w = _np.array([max(0.0, 1.0 - abs((j + 0.5 - center) / support))
                       for j in range(lo, hi)], dtype=_np.float64)
        total = w.sum()
        rows.append((lo, w / total if total else w))
    return rows


def _resize_axis(arr, axis, dst, mode):
    """Resize one axis. Passing the rows and the columns separately is what this
    filter being separable means."""
    src = arr.shape[axis]
    if src == dst:
        return arr
    if mode == "nearest":
        pick = (_np.arange(dst) * (src / dst)).astype(int)
        return _np.take(arr, pick, axis=axis)
    out = _np.empty(arr.shape[:axis] + (dst,) + arr.shape[axis + 1:], dtype=_np.float64)
    moved = _np.moveaxis(arr, axis, 0)
    dest = _np.moveaxis(out, axis, 0)
    for i, (lo, w) in enumerate(_aa_weights(src, dst)):
        chunk = moved[lo:lo + len(w)]
        dest[i] = (chunk * w.reshape((-1,) + (1,) * (chunk.ndim - 1))).sum(axis=0)
    return out


def _short_side(h, w, size):
    """The short side to `size`. The long side keeps the ratio — torchvision's
    `Resize(int)`."""
    short, long = min(h, w), max(h, w)
    if short == size:
        return h, w
    new_long = int(size * long / short)
    return (size, new_long) if h < w else (new_long, size)


class Resize:
    """Resize a `(H,W,C)` array. **It takes an array rather than a tensor** —
    what arrives at this position in torchvision is a PIL image, and with no PIL
    here an array stands in (`RandomCrop`'s rule).

    An integer `size` sets **the short side** to that value and keeps the ratio.
    Given two, they are used as they are.

    `interpolation` is `"bilinear"` (the default, antialiasing included) or
    `"nearest"`. It produces the same values as torchvision's default — matched by
    measurement.
    """

    def __init__(self, size, interpolation="bilinear"):
        self.size = int(size) if isinstance(size, int) else tuple(size)
        if interpolation not in ("bilinear", "nearest"):
            raise ValueError(
                f"interpolation is 'bilinear' or 'nearest' — got {interpolation!r}")
        self.interpolation = interpolation

    def __call__(self, img):
        img = _require_hwc(img, type(self).__name__)
        h, w = img.shape[0], img.shape[1]
        th, tw = (_short_side(h, w, self.size) if isinstance(self.size, int)
                  else self.size)
        out = _resize_axis(_np.asarray(img, dtype=_np.float64), 0, th, self.interpolation)
        out = _resize_axis(out, 1, tw, self.interpolation)
        return _np.ascontiguousarray(out)

    def __repr__(self):
        return f"{type(self).__name__}(size={self.size}, interpolation={self.interpolation})"


class CenterCrop:
    """Crop the centre. **A crop larger than the original is zero-padded and then
    cropped** — torchvision does that, and refusing makes the same code
    diverge."""

    def __init__(self, size):
        self.size = (int(size), int(size)) if isinstance(size, int) else tuple(size)

    def __call__(self, img):
        img = _require_hwc(img, type(self).__name__)
        th, tw = self.size
        h, w = img.shape[0], img.shape[1]
        pad_h, pad_w = max(0, th - h), max(0, tw - w)
        if pad_h or pad_w:
            pads = [(pad_h // 2, pad_h - pad_h // 2),
                    (pad_w // 2, pad_w - pad_w // 2)] + [(0, 0)] * (img.ndim - 2)
            img = _np.pad(img, pads)
            h, w = img.shape[0], img.shape[1]
        top = int(round((h - th) / 2.0))
        left = int(round((w - tw) / 2.0))
        return _np.ascontiguousarray(_crop(img, 0, top, th, 1, left, tw))

    def __repr__(self):
        return f"{type(self).__name__}(size={self.size})"


class _Transforms:
    """The slot for `from borchvision import transforms`. The name alone is
    borrowed so that torchvision's module structure stays intact."""


transforms = _Transforms()
for _name in ("Compose", "ToTensor", "Normalize", "RandomHorizontalFlip",
              "RandomCrop", "Resize", "CenterCrop"):
    setattr(transforms, _name, globals()[_name])
