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

import enum as _enum

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


class InterpolationMode(_enum.Enum):
    """torchvision's names for the filters. **Two of the seven resample here.**

    The names of the other five are kept rather than left out, for the reason
    `torch.int32` keeps its name in the core: leaving it out makes
    `InterpolationMode.BICUBIC` stop with an `AttributeError`, and that wording is
    **indistinguishable from a typo.** Kept, it can say what is absent.
    """

    NEAREST = "nearest"
    NEAREST_EXACT = "nearest-exact"
    BILINEAR = "bilinear"
    BICUBIC = "bicubic"
    BOX = "box"
    HAMMING = "hamming"
    LANCZOS = "lanczos"


_RESAMPLES = ("bilinear", "nearest")


def _interpolation(value):
    """A string or an `InterpolationMode`, into the one string we resample with.

    Both spellings are taken because both are written. torchvision's own
    documentation passes `InterpolationMode.BILINEAR`, and its tutorials pass
    `"bilinear"` — a copied line has to run either way.
    """
    name = value.value if isinstance(value, InterpolationMode) else value
    if name not in _RESAMPLES:
        raise ValueError(
            f"interpolation {name!r} does not resample here — "
            f"{' or '.join(_RESAMPLES)} does.\n"
            "  The other filters are PIL's, and each is a different kernel to write;\n"
            "  none of them is what an introductory tutorial passes.")
    return name

def _pair(size, who):
    """One number into `(h, w)`, a pair through as it is.

    torchvision takes a one-element sequence as well and spreads it over both —
    `FiveCrop([3])` is `(3, 3)`. It is written out because a copied line uses it.
    """
    if isinstance(size, int):
        return (int(size), int(size))
    seq = tuple(int(v) for v in size)
    if len(seq) == 1:
        return (seq[0], seq[0])
    if len(seq) != 2:
        raise ValueError(
            f"{who} takes a size of one or two numbers — it received {len(seq)}.\n"
            "(torch: Please provide only two dimensions (h, w) for size)")
    return seq


def _pad_sides(padding):
    """`(left, top, right, bottom)` — **torchvision's order, which is not numpy's.**

    One number is all four sides, two are (left/right, top/bottom), four are the
    sides one by one. The two-element form is the one that misreads: it is not
    (left, top).
    """
    if isinstance(padding, int):
        return (padding,) * 4
    seq = tuple(int(v) for v in padding)
    if len(seq) == 1:
        return seq * 4
    if len(seq) == 2:
        return (seq[0], seq[1], seq[0], seq[1])
    if len(seq) != 4:
        raise ValueError(
            f"padding is one, two or four numbers — it received {len(seq)}.\n"
            "(torch: Padding must be an int or a 1, 2, or 4 element tuple)")
    return seq


def _to_numpy(x):
    """A tensor of either library, or an array, into numpy — for the transforms
    that are **given** tensors as arguments rather than as the image."""
    if isinstance(x, _np.ndarray):
        return x
    take = getattr(x, "numpy", None)
    return take() if callable(take) else _np.asarray(x)


def _antialias(value):
    """`antialias=False` is refused rather than accepted and ignored.

    The two are not the same picture — up to 0.0301 apart at 8×8→4×4 (measured,
    and written out in the resizing section below). Accepting the argument and
    resampling the other way anyway would be the quiet kind of wrong: the code
    says one thing, the pixels are another, and nothing raises.
    """
    if value is True:
        return True
    raise ValueError(
        f"antialias={value!r} is not resampled here — only the antialiased filter is.\n"
        "  torchvision's own default is True, and off differs by up to 0.0301 "
        "at 8x8 to 4x4 (measured).")

# --- the skeleton: the axes arrive as arguments -----------------------------
# The per-image transforms handle (H,W,C) and the batch ones (N,C,H,W). Writing
# the same job twice because the positions differ eventually diverges, so they take
# **which axis** the height and the width are and share one copy.
#
# **The two flips are one function for that same reason.** Horizontal and vertical
# differ only in which axis is handed over, and a `_vflip` written next to this one
# would be the same line under a second name — two places to fix on the day it moves.

def _flip(arr, axis):
    return _np.flip(arr, axis=axis)


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


class Lambda:
    """Wraps a function so it can stand inside a `Compose`.

    It looks like nothing, and it is the only place a learner's own function can
    enter the pipeline — without it a one-line `x * 2` has to become a class.
    """

    def __init__(self, lambd):
        if not callable(lambd):
            raise TypeError(
                f"Lambda takes a callable — it received "
                f"{type(lambd).__name__!r}.\n"
                "(torch: Argument lambd should be callable)")
        self.lambd = lambd

    def __call__(self, x):
        return self.lambd(x)

    def __repr__(self):
        # **The function is not printed.** torchvision prints an empty pair of
        # brackets here, and a lambda's repr carries a memory address — printing it
        # would make the same pipeline print differently on every run.
        return f"{type(self).__name__}()"


class _RandomTransforms:
    """What `RandomApply`, `RandomChoice` and `RandomOrder` share — the list, and
    how the list prints."""

    def __init__(self, transforms):
        if isinstance(transforms, str) or not hasattr(transforms, "__len__"):
            raise TypeError(
                f"{type(self).__name__} takes a sequence of transforms — it "
                f"received {type(transforms).__name__!r}.\n"
                "(torch: Argument transforms should be a sequence)")
        self.transforms = transforms

    def __repr__(self):
        inner = "".join(f"\n    {t}" for t in self.transforms)
        return f"{type(self).__name__}({inner}\n)"


class RandomApply(_RandomTransforms):
    """Applies the whole list, or none of it, with probability `p`.

    **All of them or none** — not each with its own draw. One draw decides the
    lot, which is what makes it different from putting a `p` on each transform.
    """

    def __init__(self, transforms, p=0.5):
        super().__init__(transforms)
        self.p = p

    def __call__(self, x):
        if _rng.random() >= self.p:
            return x
        for t in self.transforms:
            x = t(x)
        return x

    def __repr__(self):
        inner = "".join(f"\n    {t}" for t in self.transforms)
        return f"{type(self).__name__}(\n    p={self.p}{inner}\n)"


class RandomChoice(_RandomTransforms):
    """Draws **one** of the list and applies it. `p` weights the draw."""

    def __init__(self, transforms, p=None):
        super().__init__(transforms)
        if p is not None and (isinstance(p, str) or not hasattr(p, "__len__")):
            raise TypeError(
                f"p is a sequence of weights, one per transform — it received "
                f"{type(p).__name__!r}.\n"
                "(torch: Argument p should be a sequence)")
        self.p = p

    def __call__(self, x):
        # **The weights are normalised here and are not in torch.** torch's
        # `random.choices` takes relative weights, numpy's `choice` takes a
        # distribution that sums to 1. Handing the weights straight through makes
        # `p=[1, 1]` stop rather than mean "evenly".
        weights = None
        if self.p is not None:
            weights = _np.asarray(self.p, dtype=_np.float64)
            weights = weights / weights.sum()
        i = int(_rng.choice(len(self.transforms), p=weights))
        return self.transforms[i](x)

    def __repr__(self):
        return f"{super().__repr__()}(p={self.p})"


class RandomOrder(_RandomTransforms):
    """Applies every one of them, in a shuffled order."""

    def __call__(self, x):
        for i in _rng.permutation(len(self.transforms)):
            x = self.transforms[int(i)](x)
        return x


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


class LinearTransformation:
    """Flattens, subtracts the mean, multiplies by a matrix, puts the shape back.

    This is where whitening (ZCA/PCA) is applied: the matrix and the mean are
    worked out **beforehand** from the training set, and this only applies them.
    It takes tensors and numpy arrays alike, for `Normalize`'s reason.
    """

    def __init__(self, transformation_matrix, mean_vector):
        m = _np.asarray(_to_numpy(transformation_matrix), dtype=_np.float32)
        v = _np.asarray(_to_numpy(mean_vector), dtype=_np.float32)
        if m.ndim != 2 or m.shape[0] != m.shape[1]:
            raise ValueError(
                f"transformation_matrix should be square — it received {m.shape}.\n"
                "(torch: transformation_matrix should be square)")
        if v.ndim != 1 or v.shape[0] != m.shape[0]:
            raise ValueError(
                f"mean_vector should be as long as one side of the matrix "
                f"{m.shape} — it received {v.shape}.\n"
                "(torch: mean_vector should have the same length)")
        self.transformation_matrix = m
        self.mean_vector = v

    def __call__(self, x):
        shape = tuple(x.shape)
        if len(shape) < 3:
            raise ValueError(
                f"LinearTransformation takes (...,C,H,W) — it received {shape}")
        n = int(shape[-3] * shape[-2] * shape[-1])
        if n != self.transformation_matrix.shape[0]:
            raise ValueError(
                f"The image flattens to {n} and the matrix is "
                f"{self.transformation_matrix.shape[0]} wide — they do not meet.\n"
                "(torch: Input tensor and transformation matrix have incompatible shape)")
        if isinstance(x, _np.ndarray):
            flat = x.reshape(-1, n) - self.mean_vector
            return (flat @ self.transformation_matrix).reshape(shape)
        L = _backend()
        flat = x.reshape(-1, n) - L.tensor(self.mean_vector)
        return (flat @ L.tensor(self.transformation_matrix)).reshape(shape)

    def __repr__(self):
        return (f"{type(self).__name__}(transformation_matrix="
                f"{self.transformation_matrix.tolist()}"
                f", mean_vector={self.mean_vector.tolist()})")


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
        return _np.ascontiguousarray(_flip(img, 1))

    def __repr__(self):
        return f"{type(self).__name__}(p={self.p})"


class RandomVerticalFlip:
    """Flip top to bottom. `RandomHorizontalFlip`'s place, on the other axis.

    **The default is 0.5 here as well, and that is worth saying out loud.** A
    vertical flip is wrong for most photographs — an upside-down cat is not a cat
    the model will meet — so this is the one transform where the default is
    usually not what is wanted. torchvision keeps 0.5 anyway, and so does this.
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        img = _require_hwc(img, type(self).__name__)
        if _rng.random() >= self.p:
            return img
        return _np.ascontiguousarray(_flip(img, 0))

    def __repr__(self):
        return f"{type(self).__name__}(p={self.p})"


class RandomCrop:
    """Pad the edges and then crop at random. `RandomHorizontalFlip`'s place."""

    def __init__(self, size, padding=0, fill=0):
        self.size = _pair(size, "RandomCrop")
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


class Pad:
    """Pad the four sides. **The order is left, top, right, bottom** — which is
    not numpy's order, and not the order the two-element form reads as.

    One number pads all four sides; two are (left/right, top/bottom); four are
    the sides one by one. `padding_mode` is `constant` (the default, filled with
    `fill`), `edge`, `reflect` or `symmetric` — the same four numpy has, with the
    same meanings, so the arithmetic is numpy's and not ours.
    """

    def __init__(self, padding, fill=0, padding_mode="constant"):
        if padding_mode not in ("constant", "edge", "reflect", "symmetric"):
            raise ValueError(
                f"padding_mode is constant, edge, reflect or symmetric — "
                f"got {padding_mode!r}.\n"
                "(torch: Padding mode should be either constant, edge, reflect or symmetric)")
        _pad_sides(padding)                     # stops here rather than at the first call
        self.padding = padding
        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, img):
        img = _require_hwc(img, type(self).__name__)
        left, top, right, bottom = _pad_sides(self.padding)
        pads = [(top, bottom), (left, right)] + [(0, 0)] * (img.ndim - 2)
        if self.padding_mode != "constant":
            return _np.pad(img, pads, mode=self.padding_mode)
        # **A per-channel fill cannot go through `constant_values`.** That argument
        # is read per axis, so a three-colour fill given there paints the channel
        # axis instead of the colours. Each channel is padded with its own number.
        if isinstance(self.fill, (tuple, list)):
            if img.ndim != 3 or len(self.fill) != img.shape[2]:
                raise ValueError(
                    f"fill has {len(self.fill)} numbers and the image has "
                    f"{img.shape[2] if img.ndim == 3 else 1} channels")
            planes = [_np.pad(img[:, :, c], pads[:2], constant_values=self.fill[c])
                      for c in range(img.shape[2])]
            return _np.ascontiguousarray(_np.stack(planes, axis=2))
        return _np.pad(img, pads, constant_values=self.fill)

    def __repr__(self):
        return (f"{type(self).__name__}(padding={self.padding}, fill={self.fill}, "
                f"padding_mode={self.padding_mode})")


# **The weights are TensorFlow's, and torchvision says so in a comment.** Not
# ITU-R 601-2's 299/587/114 exactly — 0.2989 is the rounded one, and torchvision
# went with it. Rewriting them "properly" moves every grayscale pixel.
_LUMA = (0.2989, 0.587, 0.114)


def _to_gray(img, num_output_channels, who):
    if num_output_channels not in (1, 3):
        raise ValueError(
            f"num_output_channels is 1 or 3 — got {num_output_channels!r}.\n"
            "(torch: num_output_channels should be either 1 or 3)")
    arr = img if img.ndim == 3 else img[:, :, None]
    if arr.shape[2] not in (1, 3):
        raise TypeError(
            f"{who} takes a 1- or 3-channel image — it received "
            f"{arr.shape[2]} channels.")
    if arr.shape[2] == 3:
        lum = (arr[:, :, 0] * _LUMA[0] + arr[:, :, 1] * _LUMA[1]
               + arr[:, :, 2] * _LUMA[2])
        # **`astype` truncates, and that is the point.** torch's `.to(dtype)`
        # truncates too, so a uint8 image comes out the same on both sides. PIL's
        # `convert("L")` rounds instead, which is where our uint8 answer and a PIL
        # answer part by one — measured, and written down rather than smoothed over.
        one = lum.astype(arr.dtype)[:, :, None]
    else:
        one = arr.copy()
    if num_output_channels == 3:
        return _np.ascontiguousarray(_np.repeat(one, 3, axis=2))
    return _np.ascontiguousarray(one)


class Grayscale:
    """Three channels to one. `num_output_channels=3` gives it back as three
    equal ones — which is what a pre-trained three-channel model needs."""

    def __init__(self, num_output_channels=1):
        self.num_output_channels = num_output_channels

    def __call__(self, img):
        img = _require_hwc(img, type(self).__name__)
        return _to_gray(img, self.num_output_channels, type(self).__name__)

    def __repr__(self):
        return f"{type(self).__name__}(num_output_channels={self.num_output_channels})"


class RandomGrayscale:
    """Grayscale with probability `p`. **The channel count does not change** —
    a three-channel image comes back as three equal channels, so the batch that
    follows still stacks."""

    def __init__(self, p=0.1):
        self.p = p

    def __call__(self, img):
        img = _require_hwc(img, type(self).__name__)
        if _rng.random() >= self.p:
            return img
        channels = img.shape[2] if img.ndim == 3 else 1
        return _to_gray(img, channels, type(self).__name__)

    def __repr__(self):
        return f"{type(self).__name__}(p={self.p})"


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
        pieces[i] = _flip(one, 2) if flips[i] else one
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


def _short_side(h, w, size, max_size=None):
    """The short side to `size`. The long side keeps the ratio — torchvision's
    `Resize(int)`.

    `max_size` caps the long side afterwards. It exists because the ratio-keeping
    form has no upper bound: one panorama gives one enormous tensor, and the cap is
    how torchvision's detection recipes stop that.
    """
    short, long = min(h, w), max(h, w)
    new_short, new_long = size, int(size * long / short)
    if max_size is not None:
        if max_size <= size:
            raise ValueError(
                f"max_size={max_size} has to be larger than the short side {size}.\n"
                "(torch: max_size must be strictly greater than the requested size "
                "for the smaller edge size)")
        if new_long > max_size:
            new_short, new_long = int(max_size * new_short / new_long), max_size
    if (new_short, new_long) == (short, long):
        return h, w
    return (new_short, new_long) if h < w else (new_long, new_short)


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

    def __init__(self, size, interpolation="bilinear", max_size=None,
                 antialias=True):
        self.size = int(size) if isinstance(size, int) else tuple(size)
        self.interpolation = _interpolation(interpolation)
        if max_size is not None and not isinstance(self.size, int):
            raise ValueError(
                "max_size means something only when the size is the short side "
                "(a single number).\n"
                "(torch: max_size should only be passed if size is int or sequence "
                "of length 1)")
        self.max_size = max_size
        self.antialias = _antialias(antialias)

    def __call__(self, img):
        img = _require_hwc(img, type(self).__name__)
        h, w = img.shape[0], img.shape[1]
        th, tw = (_short_side(h, w, self.size, self.max_size)
                  if isinstance(self.size, int) else self.size)
        out = _resize_axis(_np.asarray(img, dtype=_np.float64), 0, th, self.interpolation)
        out = _resize_axis(out, 1, tw, self.interpolation)
        return _np.ascontiguousarray(out)

    def __repr__(self):
        # **Four fields, not two.** torchvision prints `max_size` and `antialias`
        # here as well, and while this printed two the golden table simply left
        # `repr::Resize` out — the one place a repr was treated as unspecifiable
        # because it did not match.
        return (f"{type(self).__name__}(size={self.size}, "
                f"interpolation={self.interpolation}, max_size={self.max_size}, "
                f"antialias={self.antialias})")


class CenterCrop:
    """Crop the centre. **A crop larger than the original is zero-padded and then
    cropped** — torchvision does that, and refusing makes the same code
    diverge."""

    def __init__(self, size):
        self.size = _pair(size, "CenterCrop")

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


class FiveCrop:
    """The four corners and the centre — **five images out of one.**

    What comes back is a tuple, not an image, so `ToTensor` cannot simply follow
    it. torchvision's own documentation says the same and hands the tuple on with
    a `Lambda`. That is why `Lambda` exists in this file at all.
    """

    def __init__(self, size):
        self.size = _pair(size, "FiveCrop")

    def __call__(self, img):
        img = _require_hwc(img, type(self).__name__)
        th, tw = self.size
        h, w = img.shape[0], img.shape[1]
        if th > h or tw > w:
            raise ValueError(
                f"The crop size {self.size} is larger than the image {(h, w)}.\n"
                "(torch: Requested crop size is bigger than input size)")
        corners = ((0, 0), (0, w - tw), (h - th, 0), (h - th, w - tw))
        out = [_np.ascontiguousarray(_crop(img, 0, top, th, 1, left, tw))
               for top, left in corners]
        # **The centre is `CenterCrop`'s, not another rounding written here.** The
        # halves land differently at odd sizes, and two roundings that agree today
        # are two places to fix on the day they stop.
        out.append(CenterCrop(self.size)(img))
        return tuple(out)

    def __repr__(self):
        return f"{type(self).__name__}(size={self.size})"


class TenCrop:
    """`FiveCrop`, and then five more from the flipped image. Ten out of one.

    `vertical_flip` flips top to bottom instead of left to right.
    """

    def __init__(self, size, vertical_flip=False):
        self.size = _pair(size, "TenCrop")
        self.vertical_flip = vertical_flip

    def __call__(self, img):
        img = _require_hwc(img, type(self).__name__)
        first = FiveCrop(self.size)(img)
        turned = _flip(img, 0 if self.vertical_flip else 1)
        return first + FiveCrop(self.size)(_np.ascontiguousarray(turned))

    def __repr__(self):
        return (f"{type(self).__name__}(size={self.size}, "
                f"vertical_flip={self.vertical_flip})")


class RandomResizedCrop:
    """Crop a random area of a random shape, then resize it to `size`. **The
    ImageNet recipe**, and the reason a tutorial's accuracy moves when it is left
    out.

    `scale` is the fraction of the area to keep and `ratio` the width-to-height
    range. Ten draws are made, and if none of them fits inside the image it falls
    back to a centre crop — torchvision does exactly that, fallback included,
    because without it the draw can fail on a thin image and there is nothing to
    return.
    """

    def __init__(self, size, scale=(0.08, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0),
                 interpolation="bilinear", antialias=True):
        self.size = _pair(size, "RandomResizedCrop")
        if scale[0] > scale[1] or ratio[0] > ratio[1]:
            raise ValueError(
                f"scale and ratio read as (min, max) — got {tuple(scale)} and "
                f"{tuple(ratio)}.\n"
                "(torch: Scale and ratio should be of kind (min, max))")
        self.scale = scale
        self.ratio = ratio
        self.interpolation = _interpolation(interpolation)
        self.antialias = _antialias(antialias)

    def get_params(self, img):
        """Where and how big. **Ten draws, then a centre crop.** Kept as a method
        of its own because it is the only part that draws — the pytest side calls
        it directly to see the distribution."""
        h, w = img.shape[0], img.shape[1]
        area = h * w
        log_ratio = (_np.log(self.ratio[0]), _np.log(self.ratio[1]))
        for _ in range(10):
            target = area * _rng.uniform(self.scale[0], self.scale[1])
            aspect = float(_np.exp(_rng.uniform(log_ratio[0], log_ratio[1])))
            cw = int(round(float(_np.sqrt(target * aspect))))
            ch = int(round(float(_np.sqrt(target / aspect))))
            if 0 < cw <= w and 0 < ch <= h:
                top = int(_rng.integers(0, h - ch + 1))
                left = int(_rng.integers(0, w - cw + 1))
                return top, left, ch, cw
        in_ratio = float(w) / float(h)
        if in_ratio < min(self.ratio):
            cw, ch = w, int(round(w / min(self.ratio)))
        elif in_ratio > max(self.ratio):
            ch, cw = h, int(round(h * max(self.ratio)))
        else:
            cw, ch = w, h
        return (h - ch) // 2, (w - cw) // 2, ch, cw

    def __call__(self, img):
        img = _require_hwc(img, type(self).__name__)
        top, left, ch, cw = self.get_params(img)
        piece = _np.ascontiguousarray(_crop(img, 0, top, ch, 1, left, cw))
        return Resize(self.size, self.interpolation)(piece)

    def __repr__(self):
        return (f"{type(self).__name__}(size={self.size}, "
                f"scale={tuple(round(s, 4) for s in self.scale)}, "
                f"ratio={tuple(round(r, 4) for r in self.ratio)}, "
                f"interpolation={self.interpolation}, antialias={self.antialias})")


class RandomErasing:
    """Blank out a random rectangle of a **tensor** — this one runs after
    `ToTensor`, unlike every other transform in this file.

    That is torchvision's position for it and not a choice made here: the erased
    value is `0` on a normalised image, which means the channel mean, and that
    only has a meaning once the image is numbers rather than pixels.

    `value` is a number, one number per channel, or `"random"` for normal noise.
    Unless `inplace`, the tensor is cloned first — a backend without `clone` and
    slice assignment cannot run this, and it says so rather than half-erasing.
    """

    def __init__(self, p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0,
                 inplace=False):
        if isinstance(value, str) and value != "random":
            raise ValueError(
                f"value as a string is only 'random' — got {value!r}.\n"
                "(torch: If value is str, it should be 'random')")
        if scale[0] < 0 or scale[1] > 1:
            raise ValueError(
                f"scale is a fraction of the area, between 0 and 1 — got {tuple(scale)}.\n"
                "(torch: Scale should be between 0 and 1)")
        if p < 0 or p > 1:
            raise ValueError(
                f"p is a probability, between 0 and 1 — got {p!r}.\n"
                "(torch: Random erasing probability should be between 0 and 1)")
        self.p = p
        self.scale = scale
        self.ratio = ratio
        self.value = value
        self.inplace = inplace

    def get_params(self, shape):
        """`(top, left, height, width)`, or `None` when ten draws all missed.

        **The rectangle has to be strictly smaller than the image** on both sides.
        torchvision's condition is `<` rather than `<=`, so an erase covering the
        whole image never happens, and on a small image the ten draws can all miss
        — that is the `None`.
        """
        _, h, w = shape[-3], shape[-2], shape[-1]
        area = h * w
        log_ratio = (_np.log(self.ratio[0]), _np.log(self.ratio[1]))
        for _ in range(10):
            erase = area * _rng.uniform(self.scale[0], self.scale[1])
            aspect = float(_np.exp(_rng.uniform(log_ratio[0], log_ratio[1])))
            eh = int(round(float(_np.sqrt(erase * aspect))))
            ew = int(round(float(_np.sqrt(erase / aspect))))
            if not (eh < h and ew < w):
                continue
            return (int(_rng.integers(0, h - eh + 1)),
                    int(_rng.integers(0, w - ew + 1)), eh, ew)
        return None

    def __call__(self, x):
        if _rng.random() >= self.p:
            return x
        shape = tuple(x.shape)
        if len(shape) < 3:
            raise ValueError(
                f"RandomErasing takes (...,C,H,W) — it received {shape}.\n"
                "  It runs after `ToTensor`, not before it.")
        found = self.get_params(shape)
        if found is None:
            return x
        top, left, eh, ew = found
        channels = shape[-3]
        if self.value == "random":
            fill = _rng.standard_normal((channels, eh, ew)).astype(_np.float32)
        elif isinstance(self.value, (tuple, list)):
            if len(self.value) not in (1, channels):
                raise ValueError(
                    f"value has {len(self.value)} numbers and the image has "
                    f"{channels} channels.\n"
                    "(torch: If value is a sequence, it should have either a single "
                    "value or the number of input channels)")
            fill = _np.asarray(self.value, dtype=_np.float32).reshape(-1, 1, 1)
        else:
            fill = float(self.value)
        out = x if self.inplace else (
            x.copy() if isinstance(x, _np.ndarray) else x.clone())
        out[..., top:top + eh, left:left + ew] = fill
        return out

    def __repr__(self):
        return (f"{type(self).__name__}(p={self.p}, scale={self.scale}, "
                f"ratio={self.ratio}, value={self.value}, inplace={self.inplace})")


class _Transforms:
    """The slot for `from borchvision import transforms`. The name alone is
    borrowed so that torchvision's module structure stays intact."""


transforms = _Transforms()
for _name in ("CenterCrop", "Compose", "FiveCrop", "Grayscale",
              "InterpolationMode", "Lambda", "LinearTransformation", "Normalize",
              "Pad", "RandomApply", "RandomChoice", "RandomCrop", "RandomErasing",
              "RandomGrayscale", "RandomHorizontalFlip", "RandomOrder",
              "RandomResizedCrop", "RandomVerticalFlip", "Resize", "TenCrop",
              "ToTensor"):
    setattr(transforms, _name, globals()[_name])
