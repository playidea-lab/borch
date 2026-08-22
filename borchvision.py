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
import math as _math
import sys as _sys
import types as _types
import warnings as _warnings

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


def _warn_min_max(scale, ratio):
    """torchvision **warns** when a range arrives the wrong way round and goes on.

    Both `RandomResizedCrop` and `RandomErasing` do it, and both keep running. The
    line between warning and refusing is theirs to draw, not ours — a refusal here
    stops code that runs over there, and this file's whole claim is that the same
    code does the same thing.
    """
    if scale[0] > scale[1] or ratio[0] > ratio[1]:
        _warnings.warn("Scale and ratio should be of kind (min, max)")


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
        # **A tuple is what `FiveCrop` and `TenCrop` hand back**, and it survives
        # `asarray` as a stacked 4-D array — so without this the refusal below reads
        # `it received (5, 2, 2, 3)` and leaves the caller to work out that the five is
        # five pictures. torchvision's own documentation answers it with `Lambda`.
        #
        # Prompted by the sister library, where the same composition type-checked and
        # died inside `ToTensor` instead: a shared interface widened to admit several
        # pictures authorises them everywhere it is accepted.
        if isinstance(pic, (tuple, list)):
            raise TypeError(
                f"ToTensor takes one picture — {len(pic)} arrived together.\n"
                "  `FiveCrop` and `TenCrop` hand back several. Put a `Lambda` after\n"
                "  them to turn the group into one tensor:\n"
                "  Compose([FiveCrop(size), Lambda(lambda crops: "
                "stack([ToTensor()(c) for c in crops]))])")
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

    **`inplace` is accepted and does nothing, and that is written here because it
    was true before it was written.** An audit for constructor arguments that never
    reach `__call__` found exactly one in this file, and it was this — the same
    shape as `borch_webgpu`'s optimizers taking `weight_decay` and handing it to a
    JS call that discards surplus arguments.

    It is kept rather than refused, and the distinction from `antialias=False` next
    door is the whole reason. Ignoring `antialias` silently would give **different
    pixels**; ignoring `inplace` gives **the same values and one more allocation** —
    torchvision documents it as an optimisation, and `x = Normalize(..., inplace=
    True)(x)` returns exactly what the out-of-place form returns. Refusing it would
    stop a copied tutorial line for nothing.

    It cannot be honoured either. The core's tensors could be written through, but
    the sister library's cannot — a TF.js tensor is immutable, which `borch/_tensor.py`
    already records as where the two part. Doing it on one and not the other would
    make the same line mean two things. `tests/test_vision.py` pins the no-op so this
    paragraph cannot quietly stop being true.
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
    """Pad the edges and then crop at random. `RandomHorizontalFlip`'s place.

    **The argument list is torchvision's, and it was not.** This took
    `(size, padding, fill)` while torchvision takes
    `(size, padding, pad_if_needed, fill, padding_mode)`, so `RandomCrop(32, 4, True)`
    set `fill=True` here and `pad_if_needed=True` there — the same line, quietly
    meaning two things, with the right shape coming out either way. Found by
    comparing every constructor against torchvision's rather than by anything going
    wrong; `tests/test_torch_signatures.py` now asks that question on every run.

    `padding` defaults to `None` rather than `0` for the same reason: the default
    repr read `padding=0` against torchvision's `padding=None`, and the golden case
    passed a padding so it never looked at the default.

    `pad_if_needed` pads a picture smaller than the crop instead of refusing — **on
    both sides**, so a shortfall of two makes the picture four wider, which is
    torchvision's arithmetic and not a rounding of it.
    """

    def __init__(self, size, padding=None, pad_if_needed=False, fill=0,
                 padding_mode="constant"):
        self.size = _pair(size, "RandomCrop")
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, img):
        img = _require_hwc(img, type(self).__name__)
        if self.padding is not None:
            img = Pad(self.padding, self.fill, self.padding_mode)(img)
        th, tw = self.size
        h, w = img.shape[0], img.shape[1]
        # Width first and then height, each on its own — torchvision pads them in two
        # separate steps and the second reads the width the first produced.
        if self.pad_if_needed and w < tw:
            img = Pad([tw - w, 0], self.fill, self.padding_mode)(img)
        if self.pad_if_needed and img.shape[0] < th:
            img = Pad([0, th - img.shape[0]], self.fill, self.padding_mode)(img)
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
                "(torch: max_size should only be passed if size specifies the length "
                "of the smaller edge)")
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
        # **A warning and not a refusal**, because torchvision warns here and carries
        # on. Refusing stops a line that runs over there, and "imitate the structure"
        # includes imitating where it lets you through.
        _warn_min_max(scale, ratio)
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
        _warn_min_max(scale, ratio)
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
                    "value or (number of input channels))")
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


# --- photometric: the arithmetic torchvision does in float ------------------
#
# **These are the tensor path's numbers, not PIL's.** torchvision has two
# implementations of every one of them — `ImageEnhance` for a PIL image, and this
# arithmetic for a tensor — and they do not agree to the last bit. `Grayscale`
# already parts from PIL by one for the same reason (measured), and the golden
# compares against the tensor path because that is the one with a formula to copy.


def _bound(dtype):
    """The top of the range. **255 for uint8 and 1 for a float image** — every
    blend clamps to it, and using 1 on a uint8 picture blanks it to black."""
    return 255.0 if dtype == _np.uint8 else 1.0


def _working_dtype(dtype):
    """**float32 for an integer picture, and the picture's own float otherwise.**

    torch promotes `uint8 * python float` to float32, and the blend below then
    truncates back to uint8. Doing the arithmetic in float64 instead moves values
    across that truncation boundary — **about 1% of pixels on uint8 input**, measured
    over 300 random pictures at four factors.

    That sentence used to name `adjust_saturation(1.7)` on the golden's byte picture as
    its evidence, and **on that picture at that factor nothing differs at all.** The
    sister library read the docstring, wrote a float32 chain on the strength of it, and
    then found that deleting every narrowing in their port left all ten of their cases
    green. A comment that says "measured" is read as evidence by everyone downstream of
    it; this one was pointing at a case that did not contain any. The factor was the
    problem and not the picture — the same picture parts at 0.1.

    **The deciding question is whether a narrowing cast comes after the arithmetic.**
    If it does, the working dtype chooses the output and no tolerance covers picking
    wrong — this function is that decision, which is why it is a function rather than
    an expression. If it does not, matching numpy's promotion is free and should be
    done anyway, so that nobody later "simplifies" it: the sister library measured
    that same weighted sum and found 0 of 20 pixels exact in float64 against 20 of 20
    with every product and partial sum narrowed, while both passed the tolerance the
    whole time.
    """
    return _np.float32 if _np.dtype(dtype).kind != "f" else dtype


def _blend(a, b, ratio):
    work = _working_dtype(a.dtype)
    out = _np.clip(ratio * a.astype(work) + (1.0 - ratio) * _np.asarray(b, dtype=work),
                   0.0, _bound(a.dtype))
    return out.astype(a.dtype)


def _to_float01(img):
    """uint8 to float in [0,1], **torch's conversion and not a division of choice.**"""
    return img.astype(_np.float32) / 255.0 if img.dtype == _np.uint8 else img


def _from_float01(arr, dtype):
    """Back again. torch multiplies by `256 - 1e-3` and truncates rather than
    rounding — copied because the two differ on about half the values."""
    if dtype != _np.uint8:
        return arr.astype(dtype)
    return (arr * (255.0 + 1.0 - 1e-3)).astype(_np.uint8)


def _rgb2hsv(arr):
    """Pillow's algorithm, which is the one torchvision copied. The equal-channel
    case is **kept out of the division rather than fixed afterwards** — a grey pixel
    has `maxc == minc`, and dividing by the difference would make a NaN that then
    has to be found again."""
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    maxc, minc = arr.max(axis=-1), arr.min(axis=-1)
    eqc = maxc == minc
    cr = maxc - minc
    ones = _np.ones_like(maxc)
    s = cr / _np.where(eqc, ones, maxc)
    divisor = _np.where(eqc, ones, cr)
    rc, gc, bc = (maxc - r) / divisor, (maxc - g) / divisor, (maxc - b) / divisor
    h = ((maxc == r) * (bc - gc)
         + ((maxc == g) & (maxc != r)) * (2.0 + rc - bc)
         + ((maxc != g) & (maxc != r)) * (4.0 + gc - rc))
    return _np.stack((_np.fmod(h / 6.0 + 1.0, 1.0), s, maxc), axis=-1)


def _hsv2rgb(hsv):
    """The way back. torch selects the sextant with a one-hot mask and an einsum;
    the same choice is a `take_along_axis` here, and the numbers are identical —
    the einsum is how you write a gather when it also has to differentiate."""
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    sextant = _np.floor(h * 6.0)
    f = h * 6.0 - sextant
    idx = (sextant.astype(_np.int32) % 6)[..., None]
    p = _np.clip(v * (1.0 - s), 0.0, 1.0)
    q = _np.clip(v * (1.0 - s * f), 0.0, 1.0)
    t = _np.clip(v * (1.0 - s * (1.0 - f)), 0.0, 1.0)
    pick = lambda six: _np.take_along_axis(_np.stack(six, axis=-1), idx, axis=-1)[..., 0]
    return _np.stack((pick((v, q, p, p, t, v)),
                      pick((t, v, v, q, p, p)),
                      pick((p, p, t, v, v, q))), axis=-1)


def adjust_brightness(img, brightness_factor):
    """Toward black at 0, unchanged at 1, brighter above."""
    if brightness_factor < 0:
        raise ValueError(
            f"brightness_factor is not non-negative — got {brightness_factor}.\n"
            f"(torch: brightness_factor ({brightness_factor}) is not non-negative.)")
    img = _require_hwc(img, "adjust_brightness")
    return _blend(img, _np.zeros_like(img), brightness_factor)


def adjust_contrast(img, contrast_factor):
    """Toward **the picture's own mean grey**, which is one number for the whole
    image rather than one per channel or per pixel."""
    if contrast_factor < 0:
        raise ValueError(
            f"contrast_factor is not non-negative — got {contrast_factor}.\n"
            f"(torch: contrast_factor ({contrast_factor}) is not non-negative.)")
    img = _require_hwc(img, "adjust_contrast")
    work = _working_dtype(img.dtype)
    grey = _to_gray(img, 1, "adjust_contrast")
    return _blend(img, _np.full(img.shape, grey.astype(work).mean(), dtype=work),
                  contrast_factor)


def adjust_saturation(img, saturation_factor):
    """Toward grey, per pixel. **A one-channel picture comes back untouched** —
    torchvision does that to match PIL, and it is a branch rather than an
    accident."""
    if saturation_factor < 0:
        raise ValueError(
            f"saturation_factor is not non-negative — got {saturation_factor}.\n"
            f"(torch: saturation_factor ({saturation_factor}) is not non-negative.)")
    img = _require_hwc(img, "adjust_saturation")
    if (img.shape[2] if img.ndim == 3 else 1) == 1:
        return img
    return _blend(img, _to_gray(img, 1, "adjust_saturation"), saturation_factor)


def adjust_hue(img, hue_factor):
    """Rotate the hue. **The only one that leaves RGB** — through HSV, add to the
    angle, and back.

    `hue_factor` is a turn rather than degrees: 0.5 is half the wheel. A
    one-channel picture comes back untouched, matching PIL.
    """
    if not -0.5 <= hue_factor <= 0.5:
        raise ValueError(
            f"hue_factor is not in [-0.5, 0.5] — got {hue_factor}.\n"
            f"(torch: hue_factor ({hue_factor}) is not in [-0.5, 0.5].)")
    img = _require_hwc(img, "adjust_hue")
    if (img.shape[2] if img.ndim == 3 else 1) == 1:
        return img
    dtype = img.dtype
    hsv = _rgb2hsv(_to_float01(img).astype(_np.float32))
    hsv[..., 0] = _np.fmod(hsv[..., 0] + hue_factor + 1.0, 1.0)
    return _from_float01(_hsv2rgb(hsv), dtype)


def adjust_gamma(img, gamma, gain=1):
    """`gain * x ** gamma`, clamped. Below 1 it lifts the shadows and above 1 it
    deepens them — **the correction a display does**, which is why it is the one
    here whose name is not a direction."""
    if gamma < 0:
        raise ValueError(
            f"gamma is not non-negative — got {gamma}.\n"
            "(torch: Gamma should be a non-negative real number)")
    img = _require_hwc(img, "adjust_gamma")
    dtype = img.dtype
    out = _np.clip(gain * _np.power(_to_float01(img).astype(_np.float32), gamma), 0.0, 1.0)
    return _from_float01(out, dtype)


class ColorJitter:
    """Brightness, contrast, saturation and hue, each drawn from a range — **and
    the four applied in a drawn order.**

    The order is part of the draw and not a detail: brightness then contrast is not
    contrast then brightness, because contrast measures the picture's mean and
    brightness has already moved it.

    A single number `b` means the range `[1-b, 1+b]` (hue is centred on 0 instead).
    A range that comes out as exactly the identity is **turned off rather than
    applied** — torchvision stores `None` there, and that is why the repr shows
    `None` for anything left at its default.
    """

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = self._check_input(brightness, "brightness")
        self.contrast = self._check_input(contrast, "contrast")
        self.saturation = self._check_input(saturation, "saturation")
        self.hue = self._check_input(hue, "hue", center=0, bound=(-0.5, 0.5),
                                     clip_first_on_zero=False)

    def _check_input(self, value, name, center=1, bound=(0, float("inf")),
                     clip_first_on_zero=True):
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if value < 0:
                raise ValueError(
                    f"{name} as a single number must be non-negative — got {value}.\n"
                    f"(torch: If {name} is a single number, it must be non negative.)")
            value = [center - float(value), center + float(value)]
            if clip_first_on_zero:
                value[0] = max(value[0], 0.0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            value = [float(value[0]), float(value[1])]
        else:
            raise TypeError(
                f"{name} is a single number or a pair — got {value!r}.\n"
                f"(torch: {name} should be a single number or a list/tuple with "
                "length 2.)")
        if not bound[0] <= value[0] <= value[1] <= bound[1]:
            raise ValueError(
                f"{name} values should be between {bound}, but got {value}.\n"
                f"(torch: {name} values should be between {bound}, but got {value}.)")
        # **The identity is stored as `None`, not as a range that does nothing.**
        # Applied anyway it would still cost a blend and, on a uint8 picture, a
        # rounding — so "no jitter" and "a jitter of exactly 1" are different.
        return None if value[0] == value[1] == center else tuple(value)

    def get_params(self):
        """The four factors and **the order to apply them in.** Kept as a method of
        its own because it is the only part that draws."""
        draw = lambda span: None if span is None else float(
            _rng.uniform(span[0], span[1]))
        return (_rng.permutation(4), draw(self.brightness), draw(self.contrast),
                draw(self.saturation), draw(self.hue))

    def __call__(self, img):
        img = _require_hwc(img, type(self).__name__)
        order, brightness, contrast, saturation, hue = self.get_params()
        for which in order:
            if which == 0 and brightness is not None:
                img = adjust_brightness(img, brightness)
            elif which == 1 and contrast is not None:
                img = adjust_contrast(img, contrast)
            elif which == 2 and saturation is not None:
                img = adjust_saturation(img, saturation)
            elif which == 3 and hue is not None:
                img = adjust_hue(img, hue)
        return img

    def __repr__(self):
        return (f"{type(self).__name__}(brightness={self.brightness}"
                f", contrast={self.contrast}"
                f", saturation={self.saturation}"
                f", hue={self.hue})")


# --- the six that rewrite pixels, and the six that draw them ----------------
#
# **This is AutoAugment's op set**, which is why the six arrive together. A sweep
# that lands five and leaves one makes the sixth read as declined rather than
# missed — the shape this repository has now found in a gap-table row, a README
# sentence and a ledger row on the same day.
#
# Two of them are **uint8 only**, and that is torchvision's rule rather than a
# shortcut here: `posterize` throws bits away by masking them, and `equalize`
# counts a 256-bin histogram. Neither means anything on a float image, and
# torchvision raises rather than inventing a meaning.


def invert(img):
    """`bound - x`. White for black, and the bound is 255 or 1 depending on the
    dtype — which is the whole of it, and the whole of what goes wrong."""
    img = _require_hwc(img, "invert")
    return (_bound(img.dtype) - img.astype(_working_dtype(img.dtype))).astype(img.dtype)


def posterize(img, bits):
    """Keep the top `bits` of each byte and zero the rest — **fewer colours, by
    masking rather than by rounding.**"""
    img = _require_hwc(img, "posterize")
    if img.dtype != _np.uint8:
        raise TypeError(
            f"posterize takes a uint8 image — it received {img.dtype}.\n"
            "  It throws away the low bits of a byte, and a float image has no bits\n"
            "  to throw away.\n"
            f"(torch: Only torch.uint8 image tensors are supported, but found {img.dtype})")
    return img & _np.uint8(-int(2 ** (8 - bits)) & 0xFF)


def solarize(img, threshold):
    """Invert **only the pixels at or above** the threshold. Below it, nothing
    happens — so the picture comes back part positive and part negative."""
    img = _require_hwc(img, "solarize")
    if threshold > _bound(img.dtype):
        raise TypeError(
            f"threshold {threshold} is above this image's bound "
            f"{_bound(img.dtype)}.\n"
            "(torch: Threshold should be less than bound of img.)")
    return _np.where(img >= threshold, invert(img), img)


def autocontrast(img):
    """Stretch each channel to fill the range. **Per channel and not per picture** —
    a channel that is already flat is left alone rather than divided by zero."""
    img = _require_hwc(img, "autocontrast")
    work = _working_dtype(img.dtype)
    bound = _bound(img.dtype)
    lo = img.min(axis=(0, 1), keepdims=True).astype(work)
    hi = img.max(axis=(0, 1), keepdims=True).astype(work)
    with _np.errstate(divide="ignore", invalid="ignore"):
        scale = bound / (hi - lo)
    flat = ~_np.isfinite(scale)
    lo = _np.where(flat, 0, lo)
    scale = _np.where(flat, 1, scale)
    return _np.clip((img.astype(work) - lo) * scale, 0, bound).astype(img.dtype)


def _equalize_channel(plane):
    """One channel's histogram equalisation, **in torch's integer arithmetic.**

    Every division here floors, and the shifted lookup table (`[0] + lut[:-1]`) is
    what makes the darkest value map to 0 rather than to the first step. Written
    with floating point it is off by one over most of the range.
    """
    hist = _np.bincount(plane.reshape(-1), minlength=256)
    nonzero = hist[hist != 0]
    step = int(nonzero[:-1].sum()) // 255
    if step == 0:
        return plane
    lut = (_np.cumsum(hist) + step // 2) // step
    lut = _np.clip(_np.concatenate(([0], lut[:-1])), 0, 255).astype(_np.uint8)
    return lut[plane]


def equalize(img):
    """Flatten the histogram, per channel. **uint8 only**, for torchvision's
    reason: it counts 256 bins."""
    img = _require_hwc(img, "equalize")
    if img.dtype != _np.uint8:
        raise TypeError(
            f"equalize takes a uint8 image — it received {img.dtype}.\n"
            "  It counts a 256-bin histogram, and a float image has no bins.\n"
            f"(torch: Only torch.uint8 image tensors are supported, but found {img.dtype})")
    arr = img if img.ndim == 3 else img[:, :, None]
    return _np.stack([_equalize_channel(arr[:, :, c]) for c in range(arr.shape[2])],
                     axis=-1)


def _blurred(img):
    """The 3x3 smoothing `adjust_sharpness` blends toward — **ones with a 5 in the
    middle, over 13.**

    The border is left as it was. torchvision convolves without padding and writes
    the result back into the middle, so the outermost ring of pixels is the
    original — copied rather than tidied, because a padded convolution gives
    different numbers there and the difference is invisible in the middle.
    """
    work = _working_dtype(img.dtype)
    kernel = _np.ones((3, 3), dtype=work)
    kernel[1, 1] = 5.0
    kernel /= kernel.sum()
    src = img.astype(work)
    h, w = img.shape[0], img.shape[1]
    middle = _np.zeros((h - 2, w - 2) + img.shape[2:], dtype=work)
    for di in range(3):
        for dj in range(3):
            middle += kernel[di, dj] * src[di:di + h - 2, dj:dj + w - 2]
    out = src.copy()
    out[1:-1, 1:-1] = middle
    if _np.dtype(img.dtype).kind != "f":
        # **Rounded, not truncated.** torch casts the convolution back to an integer
        # dtype through `round`, and truncating instead is one step low on about half
        # the pixels — measured against `adjust_sharpness` on the byte picture.
        out = _np.clip(_np.round(out), 0, _bound(img.dtype))
    return out.astype(img.dtype)


def adjust_sharpness(img, sharpness_factor):
    """Blur at 0, unchanged at 1, sharper above — the blend that `ImageEnhance`
    calls sharpness. **A picture two pixels wide or shorter comes back untouched**,
    because there is no middle to convolve."""
    if sharpness_factor < 0:
        raise ValueError(
            f"sharpness_factor is not non-negative — got {sharpness_factor}.\n"
            f"(torch: sharpness_factor ({sharpness_factor}) is not non-negative.)")
    img = _require_hwc(img, "adjust_sharpness")
    if img.shape[0] <= 2 or img.shape[1] <= 2:
        return img
    return _blend(img, _blurred(img), sharpness_factor)


class _RandomPixelOp:
    """What the six `Random…` wrappers share: a probability, and one call."""

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        img = _require_hwc(img, type(self).__name__)
        if _rng.random() >= self.p:
            return img
        return self._apply(img)

    def __repr__(self):
        return f"{type(self).__name__}(p={self.p})"


class RandomInvert(_RandomPixelOp):
    """`invert` with probability `p`."""

    def _apply(self, img):
        return invert(img)


class RandomAutocontrast(_RandomPixelOp):
    """`autocontrast` with probability `p`."""

    def _apply(self, img):
        return autocontrast(img)


class RandomEqualize(_RandomPixelOp):
    """`equalize` with probability `p`. uint8 only, as the function is."""

    def _apply(self, img):
        return equalize(img)


class RandomPosterize(_RandomPixelOp):
    """`posterize` with probability `p`."""

    def __init__(self, bits, p=0.5):
        super().__init__(p)
        self.bits = bits

    def _apply(self, img):
        return posterize(img, self.bits)

    def __repr__(self):
        # **No space after the comma**, which is torchvision's own spelling here and
        # not a slip in the copy — three of these six print that way and the other
        # three have one field.
        return f"{type(self).__name__}(bits={self.bits},p={self.p})"


class RandomSolarize(_RandomPixelOp):
    """`solarize` with probability `p`."""

    def __init__(self, threshold, p=0.5):
        super().__init__(p)
        self.threshold = threshold

    def _apply(self, img):
        return solarize(img, self.threshold)

    def __repr__(self):
        return f"{type(self).__name__}(threshold={self.threshold},p={self.p})"


class RandomAdjustSharpness(_RandomPixelOp):
    """`adjust_sharpness` with probability `p`."""

    def __init__(self, sharpness_factor, p=0.5):
        super().__init__(p)
        self.sharpness_factor = sharpness_factor

    def _apply(self, img):
        return adjust_sharpness(img, self.sharpness_factor)

    def __repr__(self):
        return (f"{type(self).__name__}(sharpness_factor={self.sharpness_factor}"
                f",p={self.p})")


# --- resampling on a grid ---------------------------------------------------
#
# **Rotation, shear, scale and translation are one operation**, and this is it. A
# grid of positions is built in the output's coordinates, mapped back through the
# inverse of the transform, and the input is read at wherever that lands — which is
# almost never a pixel centre, so it is interpolated.
#
# Written once because torchvision writes it once. `rotate` is `affine` with only an
# angle, and giving each its own sampler would give two answers to the same question
# on the day one of them was edited.


def _grid_sample(img, grid, mode):
    """torch's `grid_sample` with `align_corners=False` and zero padding, on `(H,W,C)`.

    **`align_corners=False` is the whole of the coordinate convention** and it is not
    a detail: it puts -1 and 1 at the *outer edges* of the border pixels rather than
    at their centres, so the un-normalising is `((g + 1) * size - 1) / 2`. With the
    other convention every resampled pixel is half a pixel out, which looks like a
    slightly soft image rather than like a bug.
    """
    h, w = img.shape[0], img.shape[1]
    work = _working_dtype(img.dtype)
    src = img.astype(work)
    x = ((grid[..., 0] + 1.0) * w - 1.0) / 2.0
    y = ((grid[..., 1] + 1.0) * h - 1.0) / 2.0

    def read(yy, xx):
        """The input at integer positions, **zero outside** — torch's `padding_mode`."""
        inside = (xx >= 0) & (xx < w) & (yy >= 0) & (yy < h)
        out = _np.zeros(xx.shape + (src.shape[2],), dtype=work)
        cy = _np.clip(yy, 0, h - 1).astype(_np.intp)
        cx = _np.clip(xx, 0, w - 1).astype(_np.intp)
        out[inside] = src[cy[inside], cx[inside]]
        return out

    if mode == "nearest":
        # **Half goes to even**, which is `rint` here and `nearbyint` in torch. Python's
        # `round` agrees; numpy's `floor(x + 0.5)` does not, and the disagreement shows
        # only on positions landing exactly halfway — which a 90-degree rotation
        # produces on every pixel.
        return read(_np.rint(y).astype(_np.intp), _np.rint(x).astype(_np.intp))

    x0, y0 = _np.floor(x), _np.floor(y)
    fx, fy = x - x0, y - y0
    x0i, y0i = x0.astype(_np.intp), y0.astype(_np.intp)
    corners = ((y0i, x0i, (1 - fy) * (1 - fx)), (y0i, x0i + 1, (1 - fy) * fx),
               (y0i + 1, x0i, fy * (1 - fx)), (y0i + 1, x0i + 1, fy * fx))
    out = _np.zeros(x.shape + (src.shape[2],), dtype=work)
    for yy, xx, weight in corners:
        out += read(yy, xx) * weight[..., None]
    return out


def _grid_transform(img, grid, mode, fill):
    """Sample, and paint the outside with `fill`.

    **The mask is sampled alongside the picture** rather than the outside being
    computed. torchvision appends a channel of ones, resamples it with everything
    else, and reads the result as "how much of this pixel came from inside" — so a
    bilinear edge pixel is a blend of picture and fill in the same proportion the
    interpolation used. Deciding inside-ness from the coordinates instead gives a
    hard edge that is wrong by up to one whole pixel.
    """
    sampled = _grid_sample(img, grid, mode)
    if fill is None:
        painted = sampled
    else:
        ones = _np.ones(img.shape[:2] + (1,), dtype=img.dtype)
        mask = _grid_sample(ones, grid, mode)
        values = _np.asarray(fill if isinstance(fill, (tuple, list)) else [float(fill)],
                             dtype=_working_dtype(img.dtype))
        if values.size == 1:
            values = _np.repeat(values, sampled.shape[2])
        if mode == "nearest":
            painted = _np.where(mask < 0.5, values, sampled)
        else:
            painted = sampled * mask + (1.0 - mask) * values
    if _np.dtype(img.dtype).kind != "f":
        painted = _np.clip(_np.round(painted), 0, _bound(img.dtype))
    return painted.astype(img.dtype)


def _affine_grid(matrix, w, h, ow, oh, work):
    """The output's pixel centres, mapped back through `matrix`, in [-1,1].

    The half-pixel offset (`d = 0.5`) is torch's and it is what makes the grid line up
    with pixel centres rather than corners.
    """
    theta = _np.asarray(matrix, dtype=work).reshape(2, 3)
    xs = _np.linspace(-ow * 0.5 + 0.5, ow * 0.5 + 0.5 - 1, ow, dtype=work)
    ys = _np.linspace(-oh * 0.5 + 0.5, oh * 0.5 + 0.5 - 1, oh, dtype=work)
    base = _np.empty((oh, ow, 3), dtype=work)
    base[..., 0] = xs[None, :]
    base[..., 1] = ys[:, None]
    base[..., 2] = 1.0
    rescaled = theta.T / _np.asarray([0.5 * w, 0.5 * h], dtype=work)
    return base.reshape(-1, 3) @ rescaled


def _inverse_affine_matrix(center, angle, translate, scale, shear):
    """The six numbers, **inverted** — the grid maps output positions back to input
    ones, so what goes in is the inverse of the transform being described."""
    rot = _math.radians(angle)
    sx, sy = _math.radians(shear[0]), _math.radians(shear[1])
    cx, cy = center
    tx, ty = translate
    a = _math.cos(rot - sy) / _math.cos(sy)
    b = -_math.cos(rot - sy) * _math.tan(sx) / _math.cos(sy) - _math.sin(rot)
    c = _math.sin(rot - sy) / _math.cos(sy)
    d = -_math.sin(rot - sy) * _math.tan(sx) / _math.cos(sy) + _math.cos(rot)
    matrix = [x / scale for x in (d, -b, 0.0, -c, a, 0.0)]
    matrix[2] += matrix[0] * (-cx - tx) + matrix[1] * (-cy - ty)
    matrix[5] += matrix[3] * (-cx - tx) + matrix[4] * (-cy - ty)
    matrix[2] += cx
    matrix[5] += cy
    return matrix


def _affine_output_size(matrix, w, h):
    """How big the picture has to be to hold the whole rotated one — `expand=True`.

    **The truncation to 1e-4 is carried rather than justified.** torchvision's comment
    says it avoids ceiling a corner at 1e-15 up to a whole pixel, and I could not
    reproduce that: sweeping 36 picture sizes by 360 whole degrees, the answer is the
    same with the truncation and without it, every time. So it is here because
    removing it would be a change to a ported formula on the strength of one sweep,
    not because a case was found — and that sentence is the honest one.

    What the sizes are is measured, though, and it is not the obvious thing: a quarter
    turn of a 4x5 picture comes out **6 wide**, not 5. torchvision does that too, and
    the golden holds it.
    """
    pts = _np.array([[-0.5 * w, -0.5 * h, 1.0], [-0.5 * w, 0.5 * h, 1.0],
                     [0.5 * w, 0.5 * h, 1.0], [0.5 * w, -0.5 * h, 1.0]],
                    dtype=_np.float32)
    moved = pts @ _np.asarray(matrix, dtype=_np.float32).reshape(2, 3).T
    lo = moved.min(axis=0) + _np.asarray([w * 0.5, h * 0.5], dtype=_np.float32)
    hi = moved.max(axis=0) + _np.asarray([w * 0.5, h * 0.5], dtype=_np.float32)
    tol = 1e-4
    cmax = _np.ceil(_np.trunc(hi / tol) * tol)
    cmin = _np.floor(_np.trunc(lo / tol) * tol)
    size = cmax - cmin
    return int(size[0]), int(size[1])


def _center_offset(center, w, h):
    """torch's centre convention: **(0, 0) is the middle of the picture**, so an
    explicit centre arrives as an offset from it rather than as a pixel position. The
    default is `[0, 0]` and not `[w/2, h/2]` — the grid is already centred, and passing
    the middle as a centre shifts the picture by half its own size."""
    if center is None:
        return [0.0, 0.0]
    return [1.0 * (c - s * 0.5) for c, s in zip(center, (w, h))]


def _shear_pair(shear):
    if isinstance(shear, (int, float)):
        return [float(shear), 0.0]
    values = [float(s) for s in shear]
    if len(values) == 1:
        return [values[0], values[0]]
    if len(values) != 2:
        raise ValueError(
            f"shear is one or two numbers — it received {len(values)}.\n"
            f"(torch: Shear should be a sequence containing two values. Got {shear})")
    return values


def rotate(img, angle, interpolation="nearest", expand=False, center=None, fill=None):
    """Turn the picture about its centre. **Counter-clockwise for a positive angle**,
    which is PIL's direction and the opposite of what a screen's y-axis suggests.

    `expand` grows the output to hold the whole rotated picture; without it the
    corners go outside and are lost.

    **The angle is negated on the way in** and torchvision's own comment says why —
    `rotate` and `affine` disagree about which way is positive, and the negation here
    is what makes them agree from outside.
    """
    img = _require_hwc(img, "rotate")
    mode = _interpolation(interpolation)
    h, w = img.shape[0], img.shape[1]
    matrix = _inverse_affine_matrix(_center_offset(center, w, h), -angle,
                                    [0.0, 0.0], 1.0, [0.0, 0.0])
    ow, oh = _affine_output_size(matrix, w, h) if expand else (w, h)
    grid = _affine_grid(matrix, w, h, ow, oh, _working_dtype(img.dtype))
    return _grid_transform(img, grid.reshape(oh, ow, 2), mode, fill)


def affine(img, angle, translate, scale, shear, interpolation="nearest", fill=None,
           center=None):
    """Rotate, shear, scale and shift in one resampling. **One pass and not four** —
    four would interpolate four times and blur what a single grid keeps sharp."""
    img = _require_hwc(img, "affine")
    mode = _interpolation(interpolation)
    if scale <= 0:
        raise ValueError(
            f"scale is a positive number — got {scale}.\n"
            "(torch: Argument scale should be positive)")
    h, w = img.shape[0], img.shape[1]
    matrix = _inverse_affine_matrix(_center_offset(center, w, h), angle,
                                    [1.0 * t for t in translate], scale,
                                    _shear_pair(shear))
    grid = _affine_grid(matrix, w, h, w, h, _working_dtype(img.dtype))
    return _grid_transform(img, grid.reshape(h, w, 2), mode, fill)


def _setup_angle(x, name):
    """A number `d` means `[-d, d]`; a pair is taken as it is."""
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        if x < 0:
            raise ValueError(
                f"{name} as a single number must be positive — got {x}.\n"
                f"(torch: If {name} is a single number, it must be positive.)")
        x = [-x, x]
    elif not (isinstance(x, (tuple, list)) and len(x) == 2):
        raise TypeError(
            f"{name} is a single number or a pair — got {x!r}.\n"
            f"(torch: {name} should be a sequence of length 2.)")
    return [float(d) for d in x]


class RandomRotation:
    """A turn drawn from `degrees`. **The fill is spelled per channel before the
    call** because torchvision does that in its `forward` and not in `rotate` — a
    single number there becomes one per channel, and passing it through undone gives
    a different picture on a three-channel image."""

    def __init__(self, degrees, interpolation="nearest", expand=False, center=None,
                 fill=0):
        self.degrees = _setup_angle(degrees, "degrees")
        self.interpolation = _interpolation(interpolation)
        self.expand = expand
        self.center = center
        self.fill = fill

    def get_params(self):
        return float(_rng.uniform(self.degrees[0], self.degrees[1]))

    def __call__(self, img):
        img = _require_hwc(img, type(self).__name__)
        channels = img.shape[2] if img.ndim == 3 else 1
        fill = ([float(self.fill)] * channels if isinstance(self.fill, (int, float))
                else [float(f) for f in self.fill])
        return rotate(img, self.get_params(), self.interpolation, self.expand,
                      self.center, fill)

    def __repr__(self):
        # **`center` and `fill` are printed only when they are set**, which is
        # torchvision's own shape here and not the same rule as `RandomAffine`'s two
        # classes down — that one drops a field when it equals its default, and this
        # one drops it when it is `None`.
        out = (f"{type(self).__name__}(degrees={self.degrees}"
               f", interpolation={self.interpolation}, expand={self.expand}")
        if self.center is not None:
            out += f", center={self.center}"
        if self.fill is not None:
            out += f", fill={self.fill}"
        return out + ")"


class RandomAffine:
    """A rotation, a shift, a scaling and a shear, each drawn from its own range and
    **applied in one resampling.**

    `translate` is a *fraction* of the picture's width and height rather than a number
    of pixels, so the same transform means the same thing on any size.
    """

    def __init__(self, degrees, translate=None, scale=None, shear=None,
                 interpolation="nearest", fill=0, center=None):
        self.degrees = _setup_angle(degrees, "degrees")
        if translate is not None:
            for t in translate:
                if not 0.0 <= t <= 1.0:
                    raise ValueError(
                        f"translate is a fraction of the picture, between 0 and 1 — "
                        f"got {tuple(translate)}.\n"
                        "(torch: translation values should be between 0 and 1)")
        self.translate = translate
        if scale is not None:
            for s in scale:
                if s < 0:
                    raise ValueError(
                        f"scale values should be positive — got {tuple(scale)}.\n"
                        "(torch: scale values should be positive)")
        self.scale = scale
        self.shear = _setup_angle(shear, "shear") if shear is not None else None
        self.interpolation = _interpolation(interpolation)
        self.fill = fill
        self.center = center

    def get_params(self, size):
        """`(angle, (tx, ty), scale, (shear_x, shear_y))`. **The shift is drawn in
        pixels and rounded**, so a fraction that works out to less than half a pixel
        draws zero rather than a fraction of one."""
        angle = float(_rng.uniform(self.degrees[0], self.degrees[1]))
        if self.translate is not None:
            max_dx = float(self.translate[0] * size[0])
            max_dy = float(self.translate[1] * size[1])
            shift = (int(round(float(_rng.uniform(-max_dx, max_dx)))),
                     int(round(float(_rng.uniform(-max_dy, max_dy)))))
        else:
            shift = (0, 0)
        scale = 1.0 if self.scale is None else float(
            _rng.uniform(self.scale[0], self.scale[1]))
        shear_x = shear_y = 0.0
        if self.shear is not None:
            shear_x = float(_rng.uniform(self.shear[0], self.shear[1]))
            if len(self.shear) == 4:
                shear_y = float(_rng.uniform(self.shear[2], self.shear[3]))
        return angle, shift, scale, (shear_x, shear_y)

    def __call__(self, img):
        img = _require_hwc(img, type(self).__name__)
        channels = img.shape[2] if img.ndim == 3 else 1
        fill = ([float(self.fill)] * channels if isinstance(self.fill, (int, float))
                else [float(f) for f in self.fill])
        angle, shift, scale, shear = self.get_params(
            (img.shape[1], img.shape[0]))          # torch asks for (w, h) here
        return affine(img, angle, shift, scale, shear, self.interpolation, fill,
                      self.center)

    def __repr__(self):
        out = f"{type(self).__name__}(degrees={self.degrees}"
        if self.translate is not None:
            out += f", translate={self.translate}"
        if self.scale is not None:
            out += f", scale={self.scale}"
        if self.shear is not None:
            out += f", shear={self.shear}"
        if self.interpolation != "nearest":
            out += f", interpolation={self.interpolation}"
        if self.fill != 0:
            out += f", fill={self.fill}"
        if self.center is not None:
            out += f", center={self.center}"
        return out + ")"


# --- the other three that resample -----------------------------------------
#
# `perspective` and `elastic_transform` are the same sampler as `rotate` with a
# different grid; `gaussian_blur` is not a resampling at all but arrives with them
# because `ElasticTransform` is built out of it.


def _gaussian_kernel1d(size, sigma, work):
    half = (size - 1) * 0.5
    x = _np.linspace(-half, half, size, dtype=work)
    pdf = _np.exp(-0.5 * (x / sigma) ** 2)
    return pdf / pdf.sum()


def gaussian_blur(img, kernel_size, sigma=None):
    """Blur with a Gaussian. **The border is reflected**, not zeroed — a zero border
    darkens the edge of every blurred picture, which looks like a vignette.

    `kernel_size` is one number or two, and both have to be **odd**: an even kernel
    has no centre pixel to sit on, so the picture would shift by half a pixel.
    """
    img = _require_hwc(img, "gaussian_blur")
    sizes = _pair(kernel_size, "gaussian_blur")
    for s in sizes:
        if s <= 0 or s % 2 == 0:
            raise ValueError(
                f"kernel_size is odd and positive — got {sizes}.\n"
                f"(torch: Kernel size value should be an odd and positive number.)")
    if sigma is None:
        sigmas = [0.3 * ((s - 1) * 0.5 - 1) + 0.8 for s in sizes]
    elif isinstance(sigma, (int, float)):
        sigmas = [float(sigma), float(sigma)]
    else:
        sigmas = [float(s) for s in sigma]
        if len(sigmas) == 1:
            sigmas = [sigmas[0], sigmas[0]]
    for s in sigmas:
        if s <= 0:
            raise ValueError(
                f"sigma is a positive number — got {sigmas}.\n"
                "(torch: sigma should have positive values.)")

    work = _working_dtype(img.dtype)
    # **`kernel_size` and `sigma` are (x, y)** — width first, like `get_image_size` and
    # unlike everything shaped. torchvision builds the 2-D kernel as an outer product
    # of the y kernel with the x one, and swapping them is invisible on a square kernel.
    kx = _gaussian_kernel1d(sizes[0], sigmas[0], work)
    ky = _gaussian_kernel1d(sizes[1], sigmas[1], work)
    kernel = _np.outer(ky, kx)
    ph, pw = sizes[1] // 2, sizes[0] // 2
    padded = _np.pad(img.astype(work), [(ph, ph), (pw, pw)] + [(0, 0)] * (img.ndim - 2),
                     mode="reflect")
    h, w = img.shape[0], img.shape[1]
    out = _np.zeros(img.shape, dtype=work)
    for di in range(kernel.shape[0]):
        for dj in range(kernel.shape[1]):
            out += kernel[di, dj] * padded[di:di + h, dj:dj + w]
    if _np.dtype(img.dtype).kind != "f":
        out = _np.clip(_np.round(out), 0, _bound(img.dtype))
    return out.astype(img.dtype)


def _perspective_coefficients(startpoints, endpoints):
    """The eight numbers, solved by least squares. **In float64 and returned as
    float32**, which is torchvision's own precision split — the solve is
    ill-conditioned enough that doing it in float32 moves the corners visibly."""
    if len(startpoints) != 4 or len(endpoints) != 4:
        raise ValueError(
            f"Please provide exactly four corners, got {len(startpoints)} startpoints "
            f"and {len(endpoints)} endpoints.\n"
            "(torch: Please provide exactly four corners)")
    a = _np.zeros((8, 8), dtype=_np.float64)
    for i, (p1, p2) in enumerate(zip(endpoints, startpoints)):
        a[2 * i] = [p1[0], p1[1], 1, 0, 0, 0, -p2[0] * p1[0], -p2[0] * p1[1]]
        a[2 * i + 1] = [0, 0, 0, p1[0], p1[1], 1, -p2[1] * p1[0], -p2[1] * p1[1]]
    b = _np.asarray(startpoints, dtype=_np.float64).reshape(8)
    solved = _np.linalg.lstsq(a, b, rcond=None)[0]
    return [float(v) for v in solved.astype(_np.float32)]


def _perspective_grid(coeffs, ow, oh, work):
    """The projective map, **with the division that makes it projective.** An affine
    grid is this one with the last two coefficients zero."""
    xs = _np.linspace(0.5, ow + 0.5 - 1.0, ow, dtype=work)
    ys = _np.linspace(0.5, oh + 0.5 - 1.0, oh, dtype=work)
    base = _np.empty((oh, ow, 3), dtype=work)
    base[..., 0] = xs[None, :]
    base[..., 1] = ys[:, None]
    base[..., 2] = 1.0
    theta1 = _np.asarray([coeffs[0:3], coeffs[3:6]], dtype=work)
    theta2 = _np.asarray([[coeffs[6], coeffs[7], 1.0]] * 2, dtype=work)
    flat = base.reshape(-1, 3)
    top = flat @ (theta1.T / _np.asarray([0.5 * ow, 0.5 * oh], dtype=work))
    bottom = flat @ theta2.T
    return (top / bottom - 1.0).reshape(oh, ow, 2)


def perspective(img, startpoints, endpoints, interpolation="bilinear", fill=None):
    """Move the four corners somewhere else — **a photograph of a photograph held at
    an angle.** Unlike `affine`, straight lines stay straight but parallel ones stop
    being parallel."""
    img = _require_hwc(img, "perspective")
    mode = _interpolation(interpolation)
    coeffs = _perspective_coefficients(startpoints, endpoints)
    grid = _perspective_grid(coeffs, img.shape[1], img.shape[0],
                             _working_dtype(img.dtype))
    return _grid_transform(img, grid, mode, fill)


def _identity_grid(h, w, work):
    """The grid that reads each output pixel from its own position — what a
    displacement is added to."""
    ys = _np.linspace((-h + 1) / h, (h - 1) / h, h, dtype=work)
    xs = _np.linspace((-w + 1) / w, (w - 1) / w, w, dtype=work)
    grid = _np.empty((h, w, 2), dtype=work)
    grid[..., 0] = xs[None, :]
    grid[..., 1] = ys[:, None]
    return grid


def elastic_transform(img, displacement, interpolation="bilinear", fill=None):
    """Push every pixel a little way, smoothly. **The displacement is given rather
    than drawn** — `ElasticTransform` draws it and this applies it, so a whole batch
    can share one warp."""
    img = _require_hwc(img, "elastic_transform")
    mode = _interpolation(interpolation)
    work = _working_dtype(img.dtype)
    shift = _np.asarray(displacement, dtype=work).reshape(img.shape[0], img.shape[1], 2)
    grid = _identity_grid(img.shape[0], img.shape[1], work) + shift
    return _grid_transform(img, grid, mode, fill)


class GaussianBlur:
    """Blur by a Gaussian whose width is **drawn from a range each call.**

    The kernel size is fixed and the sigma is the draw, which is the opposite way
    round from most of these — a bigger kernel costs time, so torchvision fixes the
    cost and varies the effect inside it.
    """

    def __init__(self, kernel_size, sigma=(0.1, 2.0)):
        self.kernel_size = _pair(kernel_size, "GaussianBlur")
        for s in self.kernel_size:
            if s <= 0 or s % 2 == 0:
                raise ValueError(
                    f"kernel_size is odd and positive — got {self.kernel_size}.\n"
                    "(torch: Kernel size value should be an odd and positive number.)")
        if isinstance(sigma, (int, float)):
            if sigma <= 0:
                raise ValueError(
                    f"sigma is a positive number — got {sigma}.\n"
                    "(torch: If sigma is a single number, it must be positive.)")
            sigma = (float(sigma), float(sigma))
        else:
            if not 0.0 < sigma[0] <= sigma[1]:
                raise ValueError(
                    f"sigma is (min, max) with min above zero — got {tuple(sigma)}.\n"
                    "(torch: sigma values should be positive and of the form "
                    "(min, max).)")
            sigma = (float(sigma[0]), float(sigma[1]))
        self.sigma = sigma

    def get_params(self):
        return float(_rng.uniform(self.sigma[0], self.sigma[1]))

    def __call__(self, img):
        img = _require_hwc(img, type(self).__name__)
        drawn = self.get_params()
        return gaussian_blur(img, self.kernel_size, [drawn, drawn])

    def __repr__(self):
        return (f"{type(self).__name__}(kernel_size={tuple(self.kernel_size)}, "
                f"sigma={self.sigma})")


class RandomPerspective:
    """Tilt the picture, with probability `p`. `distortion_scale` is how far the
    corners may move, as a fraction of half the picture."""

    def __init__(self, distortion_scale=0.5, p=0.5, interpolation="bilinear", fill=0):
        self.distortion_scale = distortion_scale
        self.p = p
        self.interpolation = _interpolation(interpolation)
        self.fill = fill

    def get_params(self, width, height):
        """The four corners before and after. **Drawn in whole pixels** — torchvision
        uses integer draws here, so a small picture has a small number of distinct
        distortions rather than a continuum."""
        half_h, half_w = height // 2, width // 2
        dx, dy = int(self.distortion_scale * half_w), int(self.distortion_scale * half_h)
        topleft = [int(_rng.integers(0, dx + 1)), int(_rng.integers(0, dy + 1))]
        topright = [int(_rng.integers(width - dx - 1, width)),
                    int(_rng.integers(0, dy + 1))]
        botright = [int(_rng.integers(width - dx - 1, width)),
                    int(_rng.integers(height - dy - 1, height))]
        botleft = [int(_rng.integers(0, dx + 1)),
                   int(_rng.integers(height - dy - 1, height))]
        start = [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]
        return start, [topleft, topright, botright, botleft]

    def __call__(self, img):
        img = _require_hwc(img, type(self).__name__)
        if _rng.random() >= self.p:
            return img
        channels = img.shape[2] if img.ndim == 3 else 1
        fill = ([float(self.fill)] * channels if isinstance(self.fill, (int, float))
                else [float(f) for f in self.fill])
        start, end = self.get_params(img.shape[1], img.shape[0])
        return perspective(img, start, end, self.interpolation, fill)

    def __repr__(self):
        # **Only `p`.** torchvision prints nothing else here — not the distortion, not
        # the fill — and that is its spelling rather than an oversight to improve on.
        return f"{type(self).__name__}(p={self.p})"


class ElasticTransform:
    """Push every pixel a little way along a **smooth random field** — the warp that
    makes handwriting look handwritten differently.

    `alpha` is how far pixels move and `sigma` how smoothly: a small sigma with a
    large alpha is noise rather than a warp, which is why both are drawn from the same
    blur.
    """

    def __init__(self, alpha=50.0, sigma=5.0, interpolation="bilinear", fill=0):
        self.alpha = ([float(alpha)] * 2 if isinstance(alpha, (int, float))
                      else [float(a) for a in alpha])
        self.sigma = ([float(sigma)] * 2 if isinstance(sigma, (int, float))
                      else [float(s) for s in sigma])
        self.interpolation = _interpolation(interpolation)
        self.fill = ([float(fill)] if isinstance(fill, (int, float))
                     else [float(f) for f in fill])

    def get_params(self, height, width):
        """A displacement field: noise, blurred, scaled by `alpha` — and **divided by
        the picture's size**, because the grid is in [-1,1] and not in pixels."""
        def one(sigma, alpha, extent):
            noise = (_rng.random((height, width, 1)).astype(_np.float32) * 2 - 1)
            if sigma > 0.0:
                size = int(8 * sigma + 1)
                if size % 2 == 0:
                    size += 1
                noise = gaussian_blur(noise, [size, size], self.sigma)
            return noise * alpha / extent
        return _np.concatenate([one(self.sigma[0], self.alpha[0], width),
                                one(self.sigma[1], self.alpha[1], height)], axis=2)

    def __call__(self, img):
        img = _require_hwc(img, type(self).__name__)
        displacement = self.get_params(img.shape[0], img.shape[1])
        return elastic_transform(img, displacement, self.interpolation, self.fill)

    def __repr__(self):
        # **The enum's name, not its value** — this is the one class that prints
        # `InterpolationMode.BILINEAR` where every other prints `bilinear`. Rendered
        # through our own enum so it is our name rather than a string spelled to look
        # like one.
        return (f"{type(self).__name__}(alpha={self.alpha}, sigma={self.sigma}, "
                f"interpolation={InterpolationMode(self.interpolation)}, "
                f"fill={self.fill})")


# --- transforms.functional -------------------------------------------------
#
# **The same arithmetic, called without building an object.** Every function here
# hands the work to the class above or to the helper the class uses — none of them
# reimplements anything. That is the point: two copies of `Resize`'s filter would
# agree on the day they were written and not on some later one.
#
# torchvision's own division is the reverse — the classes call the functions. Ours
# goes the other way because the classes were here first, and turning them inside out
# to match would be a rewrite of working code for the shape of the call graph, which
# nobody can see from outside.
#
# **What it is for**: a tutorial writes `F.hflip(x)` as often as it writes
# `RandomHorizontalFlip()`, and until now that line stopped with an `AttributeError`
# on a namespace that did not exist.


def hflip(img):
    """Left to right, with no draw in it."""
    return _np.ascontiguousarray(_flip(_require_hwc(img, "hflip"), 1))


def vflip(img):
    return _np.ascontiguousarray(_flip(_require_hwc(img, "vflip"), 0))


def crop(img, top, left, height, width):
    """**Not `RandomCrop` without the draw** — the position is given, and it may
    fall outside the picture, which numpy would answer with a smaller array rather
    than an error."""
    img = _require_hwc(img, "crop")
    h, w = img.shape[0], img.shape[1]
    if top < 0 or left < 0 or top + height > h or left + width > w:
        raise ValueError(
            f"The crop ({top}, {left}, {height}, {width}) leaves the image {(h, w)}.\n"
            "  numpy would answer a shorter array here rather than stopping, and the "
            "batch that follows would refuse to stack for a reason pointing elsewhere.")
    return _np.ascontiguousarray(_crop(img, 0, top, height, 1, left, width))


def center_crop(img, output_size):
    return CenterCrop(output_size)(img)


def pad(img, padding, fill=0, padding_mode="constant"):
    return Pad(padding, fill, padding_mode)(img)


def resize(img, size, interpolation="bilinear", max_size=None, antialias=True):
    return Resize(size, interpolation, max_size, antialias)(img)


def resized_crop(img, top, left, height, width, size, interpolation="bilinear",
                 antialias=True):
    """Crop, then resize — `RandomResizedCrop` with the draw already made."""
    return resize(crop(img, top, left, height, width), size, interpolation,
                  antialias=antialias)


def normalize(tensor, mean, std, inplace=False):
    return Normalize(mean, std, inplace)(tensor)


def to_tensor(pic):
    return ToTensor()(pic)


def rgb_to_grayscale(img, num_output_channels=1):
    return _to_gray(_require_hwc(img, "rgb_to_grayscale"), num_output_channels,
                    "rgb_to_grayscale")


def to_grayscale(img, num_output_channels=1):
    """**torchvision keeps both names** — this one is the PIL path's and
    `rgb_to_grayscale` is the tensor path's. They compute the same thing here, and
    the second name is kept because tutorials written against PIL call it."""
    return rgb_to_grayscale(img, num_output_channels)


def five_crop(img, size):
    return FiveCrop(size)(img)


def ten_crop(img, size, vertical_flip=False):
    return TenCrop(size, vertical_flip)(img)


def erase(img, i, j, h, w, v, inplace=False):
    """Blank one rectangle of a **tensor**. `RandomErasing` with the draw made.

    `i` and `j` are torchvision's names and they are top and left in that order —
    kept rather than renamed, because a tutorial passing them positionally is the
    only way anybody calls this.
    """
    out = img if inplace else (
        img.copy() if isinstance(img, _np.ndarray) else img.clone())
    out[..., i:i + h, j:j + w] = v
    return out


def get_dimensions(img):
    """`[channels, height, width]` — **and `get_image_size` is not this reversed.**"""
    img = _require_hwc(img, "get_dimensions")
    return [img.shape[2] if img.ndim == 3 else 1, img.shape[0], img.shape[1]]


def get_image_size(img):
    """`[width, height]`. **Width first**, which is the opposite order to every other
    size in this file — torchvision inherits it from PIL, where a size is `(w, h)`.
    Copying the shape here instead would be right in shape and swapped in meaning on
    any picture that is not square."""
    channels_h_w = get_dimensions(img)
    return [channels_h_w[2], channels_h_w[1]]


def get_image_num_channels(img):
    return get_dimensions(img)[0]


# --- the namespaces, and why they are modules ------------------------------
#
# `transforms` used to be an instance of a class with the name borrowed. That was
# enough for `from borchvision import transforms`, and it is **not** enough for the
# line tutorials actually write:
#
#     import torchvision.transforms.functional as F
#
# `import a.b.c` walks `sys.modules` for each dotted name, and an attribute holding an
# object is not a module however much it looks like one. So both are real modules and
# both are registered — after which all four spellings work, including
# `from borchvision.transforms import functional as F`.
#
# `borchvision` is a single file rather than a package, so nothing else would create
# those entries. Registering them by hand is the only way a file can carry a
# namespace path, and the alternative — telling people to spell the import
# differently here — is the kind of difference that makes the imitation worthless.

transforms = _types.ModuleType("borchvision.transforms")
functional = _types.ModuleType("borchvision.transforms.functional")
transforms.functional = functional
_sys.modules["borchvision.transforms"] = transforms
_sys.modules["borchvision.transforms.functional"] = functional

for _name in ("CenterCrop", "Compose", "FiveCrop", "Grayscale",
              "InterpolationMode", "Lambda", "LinearTransformation", "Normalize",
              "Pad", "RandomApply", "RandomChoice", "RandomCrop", "RandomErasing",
              "RandomGrayscale", "RandomHorizontalFlip", "RandomOrder",
              "ColorJitter", "ElasticTransform", "GaussianBlur",
              "RandomAdjustSharpness", "RandomAffine",
              "RandomAutocontrast",
              "RandomEqualize", "RandomInvert", "RandomPosterize",
              "RandomPerspective", "RandomResizedCrop", "RandomRotation",
              "RandomSolarize",
              "RandomVerticalFlip", "Resize",
              "TenCrop", "ToTensor"):
    setattr(transforms, _name, globals()[_name])

# `InterpolationMode` is in both, because torchvision has it in both — it is defined
# in `functional` and re-exported by `transforms`, and a name counted in one namespace
# and missing from the other is a gap that is not one.
for _name in ("InterpolationMode", "adjust_brightness", "adjust_contrast",
              "adjust_gamma", "adjust_hue", "adjust_saturation",
              "adjust_sharpness", "autocontrast", "equalize", "invert",
              "affine", "elastic_transform", "gaussian_blur", "perspective",
              "posterize", "rotate", "solarize",
              "center_crop", "crop", "erase", "five_crop",
              "get_dimensions", "get_image_num_channels", "get_image_size", "hflip",
              "normalize", "pad", "resize", "resized_crop", "rgb_to_grayscale",
              "ten_crop", "to_grayscale", "to_tensor", "vflip"):
    setattr(functional, _name, globals()[_name])
