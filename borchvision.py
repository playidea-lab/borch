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

- **`datasets`** — **torchvision's own addresses are unreachable from a browser**,
  which is narrower than "the fetching side is blocked" and is what the measurement
  actually says. Re-measured: `cs.toronto.edu` redirects to `cave.cs.toronto.edu`
  and neither sends `Access-Control-Allow-Origin`, and neither does
  `ossci-datasets.s3.amazonaws.com`, which is torchvision's **first** MNIST mirror.
  Hosts that do send it exist — `raw.githubusercontent.com` and `huggingface.co`
  both answer with the header — and torchvision itself keeps a `mirrors` list for
  exactly this kind of reason.

  The second half of the old reason was **stale rather than narrow**. It said
  Pyodide's filesystem is gone on a refresh, so caching could not work; but
  `borch_webgpu.fetch_cached` puts the bytes in **OPFS**, which survives one. The
  machinery this paragraph said was missing was built next door and this file did
  not hear about it — two files in one repository, each right on its own, with
  nothing comparing them.

  So what is left is a decision rather than an impossibility: a `datasets.CIFAR10`
  whose `download=True` reaches a mirror is the same code doing the same thing from
  a different address, and a `download=True` that raises is a trap. **Once the bytes
  are in hand it already works** — `TensorDataset` is in `borch`, and `fetch_cached`
  and `cache_put` are in `borch_webgpu` rather than here, which the old wording did
  not say and a reader following it would not find
- **`ops`** — `nms` is short in numpy, so "it is large" would be a false reason. The
  real one was "nobody stands in front of it", and counting the namespace shows that
  reason is **wider than what it justifies**. Of the 39 public callables, 16 are
  `nn.Module` layers and 12 more need a model's feature maps or predictions; those 28
  do need a detector, and there is no detector in the catalogue. The remaining **11
  are box geometry with no weights anywhere in them** — `nms`, `batched_nms`,
  `box_iou`, `box_area`, `box_convert`, `clip_boxes_to_image`, `masks_to_boxes`,
  `remove_small_boxes` and the three generalised IoUs. They are deterministic, so
  they would have golden cases from the first day, and the person in front of them is
  not somebody running a detector but somebody learning what IoU and NMS compute —
  which is who this project is for. **A surface with no users grows no cases**, and
  that is still true of the 28
- **`models` and `pretrained=True`** — this used to read "pre-trained weights", and
  that is no longer what is refused. **Weights load in this project**: `bimm` holds
  the architecture catalogue and `borch-hub` fetches a manifest, checks its hash and
  builds the model. What is refused is narrower and still stands. A `.pth` is a
  pickle inside a zip and `torch.load` revives the classes through their module
  paths; reading it means imitating torch's internal structure, and getting it subtly
  wrong brings **the wrong numbers in correctly shaped weights** — which is why the
  hub carries its own manifest and hash instead. And once `pretrained=True` runs,
  people compare against the published top-1, which bit equivalence being an explicit
  non-goal makes a promise this cannot keep

## The random numbers differ from torch's — written down because they cannot be
imitated

`RandomCrop` and `RandomHorizontalFlip` cannot use torch's generator. So **the
same seed does not produce the same picture as torchvision's.** It is the draw
that diverges rather than the values. The places where the probability is pinned
at 0 or 1 are deterministic, so that is where the golden compares.
"""

import bz2 as _bz2
import csv as _csv
import enum as _enum
import gzip as _gzip
import hashlib as _hashlib
import io as _io
import math as _math
import os as _os
import pickle as _pickle
import string as _string
import struct as _struct
import sys as _sys
import tarfile as _tarfile
import types as _types
import zipfile as _zipfile
import zlib as _zlib
import urllib.request as _urlreq
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


# --- the policies -----------------------------------------------------------
#
# **Nothing new is computed here.** Every one of these four picks from the operations
# above and applies them; the work was the operations, and this is the layer that
# finally uses all of them at once. `_apply_op` below is the whole vocabulary, and
# every name in it resolves to something in this file.
#
# They differ only in **what decides the choice**: a learned table (`AutoAugment`), a
# uniform draw with a fixed strength (`RandAugment`), a uniform draw with a drawn
# strength (`TrivialAugmentWide`), or several chains blended together (`AugMix`).
#
# **They want a uint8 picture.** `posterize` masks bits and `equalize` counts a
# histogram, so a float image raises the moment either is drawn — which is
# torchvision's behaviour too, and it surfaces at a random moment rather than on the
# first call. Written down here because "sometimes it works" is the worst way to meet
# it.


def _apply_op(img, op_name, magnitude, interpolation, fill):
    """One named operation at one strength. **The names are the policies' vocabulary**
    and they are not our function names — `Color` is saturation, `Identity` does
    nothing, and the shears take a tangent."""
    if op_name == "ShearX":
        # **The magnitude is a slope and `affine` takes an angle**, which is why the
        # arctangent is here. The original AutoAugment passed the slope straight into a
        # matrix; torchvision's `affine` takes degrees, so the two agree only through
        # this conversion — dropping it is a shear that is wrong by more the larger it
        # gets, and right at zero.
        return affine(img, 0.0, [0, 0], 1.0,
                      [_math.degrees(_math.atan(magnitude)), 0.0],
                      interpolation=interpolation, fill=fill, center=[0, 0])
    if op_name == "ShearY":
        return affine(img, 0.0, [0, 0], 1.0,
                      [0.0, _math.degrees(_math.atan(magnitude))],
                      interpolation=interpolation, fill=fill, center=[0, 0])
    if op_name == "TranslateX":
        return affine(img, 0.0, [int(magnitude), 0], 1.0, [0.0, 0.0],
                      interpolation=interpolation, fill=fill)
    if op_name == "TranslateY":
        return affine(img, 0.0, [0, int(magnitude)], 1.0, [0.0, 0.0],
                      interpolation=interpolation, fill=fill)
    if op_name == "Rotate":
        return rotate(img, magnitude, interpolation=interpolation, fill=fill)
    if op_name == "Brightness":
        return adjust_brightness(img, 1.0 + magnitude)
    if op_name == "Color":
        return adjust_saturation(img, 1.0 + magnitude)
    if op_name == "Contrast":
        return adjust_contrast(img, 1.0 + magnitude)
    if op_name == "Sharpness":
        return adjust_sharpness(img, 1.0 + magnitude)
    if op_name == "Posterize":
        return posterize(img, int(magnitude))
    if op_name == "Solarize":
        return solarize(img, magnitude)
    if op_name == "AutoContrast":
        return autocontrast(img)
    if op_name == "Equalize":
        return equalize(img)
    if op_name == "Invert":
        return invert(img)
    if op_name == "Identity":
        return img
    raise ValueError(
        f"The provided operator {op_name} is not recognized.\n"
        "(torch: The provided operator is not recognized.)")


class AutoAugmentPolicy(_enum.Enum):
    """Which learned table `AutoAugment` uses. The three are the datasets the search
    was run on, and they are **not interchangeable** — the SVHN policy inverts and
    shears hard because house numbers survive it, and a photograph does not."""

    IMAGENET = "imagenet"
    CIFAR10 = "cifar10"
    SVHN = "svhn"


def _linspace_bins(lo, hi, num_bins):
    return _np.linspace(lo, hi, num_bins, dtype=_np.float64)


def _posterize_bins(top, num_bins, divisor):
    """`top - round(arange / ((num_bins - 1) / divisor))`, in **integers**. A float
    magnitude here would be truncated by `posterize`'s `int()` rather than rounded, and
    the two disagree on half the bins.

    `top` is 8 everywhere except `AugMix`, where it is 4 — the same formula counting
    down from half as high, which is a stronger posterize at every bin.
    """
    steps = _np.arange(num_bins) / ((num_bins - 1) / divisor)
    return (top - _np.rint(steps)).astype(_np.int64)


# **Four tables and not one parameterised one.** They look like variations — shear,
# translate, rotate, the photometric four, posterize, solarize — and they are not: the
# ranges differ, `AugMix` translates by a third of the picture where the others use
# 150/331 of it, its posterize counts down from 4 rather than 8, `AutoAugment` alone
# has `Invert`, and three of the four put `Identity` first while one has none.
#
# The first draft of this file was one table with a `kind` flag, written from the two
# I had read. It was wrong about `AugMix` in four places at once. **A shape that looks
# like a family is not evidence that it is one**, and each of these is checkable
# against its source only while it is written out.
#
# The order matters as much as the numbers: the operation is chosen by drawing an
# index into the dictionary, so moving a name changes which strength every later draw
# lands on.


def _space_auto(num_bins, size):
    """`AutoAugment` — the only one with `Invert`, and the only one without
    `Identity`."""
    return {
        "ShearX": (_linspace_bins(0.0, 0.3, num_bins), True),
        "ShearY": (_linspace_bins(0.0, 0.3, num_bins), True),
        "TranslateX": (_linspace_bins(0.0, 150.0 / 331.0 * size[1], num_bins), True),
        "TranslateY": (_linspace_bins(0.0, 150.0 / 331.0 * size[0], num_bins), True),
        "Rotate": (_linspace_bins(0.0, 30.0, num_bins), True),
        "Brightness": (_linspace_bins(0.0, 0.9, num_bins), True),
        "Color": (_linspace_bins(0.0, 0.9, num_bins), True),
        "Contrast": (_linspace_bins(0.0, 0.9, num_bins), True),
        "Sharpness": (_linspace_bins(0.0, 0.9, num_bins), True),
        "Posterize": (_posterize_bins(8, num_bins, 4), False),
        "Solarize": (_linspace_bins(255.0, 0.0, num_bins), False),
        "AutoContrast": (None, False),
        "Equalize": (None, False),
        "Invert": (None, False),
    }


def _space_rand(num_bins, size):
    """`RandAugment` — `AutoAugment`'s ranges with `Identity` in front and `Invert`
    gone."""
    space = {"Identity": (None, False)}
    space.update(_space_auto(num_bins, size))
    del space["Invert"]
    return space


def _space_trivial(num_bins):
    """`TrivialAugmentWide` — **much wider, and it does not know the picture's size.**
    Its translate is a flat 32 pixels rather than a fraction, which is the paper's
    point: one strength ladder, no tuning."""
    return {
        "Identity": (None, False),
        "ShearX": (_linspace_bins(0.0, 0.99, num_bins), True),
        "ShearY": (_linspace_bins(0.0, 0.99, num_bins), True),
        "TranslateX": (_linspace_bins(0.0, 32.0, num_bins), True),
        "TranslateY": (_linspace_bins(0.0, 32.0, num_bins), True),
        "Rotate": (_linspace_bins(0.0, 135.0, num_bins), True),
        "Brightness": (_linspace_bins(0.0, 0.99, num_bins), True),
        "Color": (_linspace_bins(0.0, 0.99, num_bins), True),
        "Contrast": (_linspace_bins(0.0, 0.99, num_bins), True),
        "Sharpness": (_linspace_bins(0.0, 0.99, num_bins), True),
        "Posterize": (_posterize_bins(8, num_bins, 6), False),
        "Solarize": (_linspace_bins(255.0, 0.0, num_bins), False),
        "AutoContrast": (None, False),
        "Equalize": (None, False),
    }


def _space_augmix(num_bins, size, all_ops):
    """`AugMix` — **a third of the picture, and posterize counting down from four.**
    The photometric four are absent unless asked for, and they arrive at the *end*,
    which changes what every index above them draws."""
    space = {
        "ShearX": (_linspace_bins(0.0, 0.3, num_bins), True),
        "ShearY": (_linspace_bins(0.0, 0.3, num_bins), True),
        "TranslateX": (_linspace_bins(0.0, size[1] / 3.0, num_bins), True),
        "TranslateY": (_linspace_bins(0.0, size[0] / 3.0, num_bins), True),
        "Rotate": (_linspace_bins(0.0, 30.0, num_bins), True),
        "Posterize": (_posterize_bins(4, num_bins, 4), False),
        "Solarize": (_linspace_bins(255.0, 0.0, num_bins), False),
        "AutoContrast": (None, False),
        "Equalize": (None, False),
    }
    if all_ops:
        space.update({
            "Brightness": (_linspace_bins(0.0, 0.9, num_bins), True),
            "Color": (_linspace_bins(0.0, 0.9, num_bins), True),
            "Contrast": (_linspace_bins(0.0, 0.9, num_bins), True),
            "Sharpness": (_linspace_bins(0.0, 0.9, num_bins), True),
        })
    return space


def _fill_per_channel(fill, img):
    if fill is None:
        return None
    channels = img.shape[2] if img.ndim == 3 else 1
    if isinstance(fill, (int, float)):
        return [float(fill)] * channels
    return [float(f) for f in fill]


# **Lists, not tuples**, because `AutoAugment(...).policies` is a public attribute and
# torchvision hands back a list. The golden caught the difference on the first run:
# identical data, different brackets. What holds a value is part of the surface.
_POLICIES = {
    "imagenet": [
        (("Posterize", 0.4, 8), ("Rotate", 0.6, 9)),
        (("Solarize", 0.6, 5), ("AutoContrast", 0.6, None)),
        (("Equalize", 0.8, None), ("Equalize", 0.6, None)),
        (("Posterize", 0.6, 7), ("Posterize", 0.6, 6)),
        (("Equalize", 0.4, None), ("Solarize", 0.2, 4)),
        (("Equalize", 0.4, None), ("Rotate", 0.8, 8)),
        (("Solarize", 0.6, 3), ("Equalize", 0.6, None)),
        (("Posterize", 0.8, 5), ("Equalize", 1.0, None)),
        (("Rotate", 0.2, 3), ("Solarize", 0.6, 8)),
        (("Equalize", 0.6, None), ("Posterize", 0.4, 6)),
        (("Rotate", 0.8, 8), ("Color", 0.4, 0)),
        (("Rotate", 0.4, 9), ("Equalize", 0.6, None)),
        (("Equalize", 0.0, None), ("Equalize", 0.8, None)),
        (("Invert", 0.6, None), ("Equalize", 1.0, None)),
        (("Color", 0.6, 4), ("Contrast", 1.0, 8)),
        (("Rotate", 0.8, 8), ("Color", 1.0, 2)),
        (("Color", 0.8, 8), ("Solarize", 0.8, 7)),
        (("Sharpness", 0.4, 7), ("Invert", 0.6, None)),
        (("ShearX", 0.6, 5), ("Equalize", 1.0, None)),
        (("Color", 0.4, 0), ("Equalize", 0.6, None)),
        (("Equalize", 0.4, None), ("Solarize", 0.2, 4)),
        (("Solarize", 0.6, 5), ("AutoContrast", 0.6, None)),
        (("Invert", 0.6, None), ("Equalize", 1.0, None)),
        (("Color", 0.6, 4), ("Contrast", 1.0, 8)),
        (("Equalize", 0.8, None), ("Equalize", 0.6, None)),
    ],
    "cifar10": [
        (("Invert", 0.1, None), ("Contrast", 0.2, 6)),
        (("Rotate", 0.7, 2), ("TranslateX", 0.3, 9)),
        (("Sharpness", 0.8, 1), ("Sharpness", 0.9, 3)),
        (("ShearY", 0.5, 8), ("TranslateY", 0.7, 9)),
        (("AutoContrast", 0.5, None), ("Equalize", 0.9, None)),
        (("ShearY", 0.2, 7), ("Posterize", 0.3, 7)),
        (("Color", 0.4, 3), ("Brightness", 0.6, 7)),
        (("Sharpness", 0.3, 9), ("Brightness", 0.7, 9)),
        (("Equalize", 0.6, None), ("Equalize", 0.5, None)),
        (("Contrast", 0.6, 7), ("Sharpness", 0.6, 5)),
        (("Color", 0.7, 7), ("TranslateX", 0.5, 8)),
        (("Equalize", 0.3, None), ("AutoContrast", 0.4, None)),
        (("TranslateY", 0.4, 3), ("Sharpness", 0.2, 6)),
        (("Brightness", 0.9, 6), ("Color", 0.2, 8)),
        (("Solarize", 0.5, 2), ("Invert", 0.0, None)),
        (("Equalize", 0.2, None), ("AutoContrast", 0.6, None)),
        (("Equalize", 0.2, None), ("Equalize", 0.6, None)),
        (("Color", 0.9, 9), ("Equalize", 0.6, None)),
        (("AutoContrast", 0.8, None), ("Solarize", 0.2, 8)),
        (("Brightness", 0.1, 3), ("Color", 0.7, 0)),
        (("Solarize", 0.4, 5), ("AutoContrast", 0.9, None)),
        (("TranslateY", 0.9, 9), ("TranslateY", 0.7, 9)),
        (("AutoContrast", 0.9, None), ("Solarize", 0.8, 3)),
        (("Equalize", 0.8, None), ("Invert", 0.1, None)),
        (("TranslateY", 0.7, 9), ("AutoContrast", 0.9, None)),
    ],
    "svhn": [
        (("ShearX", 0.9, 4), ("Invert", 0.2, None)),
        (("ShearY", 0.9, 8), ("Invert", 0.7, None)),
        (("Equalize", 0.6, None), ("Solarize", 0.6, 6)),
        (("Invert", 0.9, None), ("Equalize", 0.6, None)),
        (("Equalize", 0.6, None), ("Rotate", 0.9, 3)),
        (("ShearX", 0.9, 4), ("AutoContrast", 0.8, None)),
        (("ShearY", 0.9, 8), ("Invert", 0.4, None)),
        (("ShearY", 0.9, 5), ("Solarize", 0.2, 6)),
        (("Invert", 0.9, None), ("AutoContrast", 0.8, None)),
        (("Equalize", 0.6, None), ("Rotate", 0.9, 3)),
        (("ShearX", 0.9, 4), ("Solarize", 0.3, 3)),
        (("ShearY", 0.8, 8), ("Invert", 0.7, None)),
        (("Equalize", 0.9, None), ("TranslateY", 0.6, 6)),
        (("Invert", 0.9, None), ("Equalize", 0.6, None)),
        (("Contrast", 0.3, 3), ("Rotate", 0.8, 4)),
        (("Invert", 0.8, None), ("TranslateY", 0.0, 2)),
        (("ShearY", 0.7, 6), ("Solarize", 0.4, 8)),
        (("Invert", 0.6, None), ("Rotate", 0.8, 4)),
        (("ShearY", 0.3, 7), ("TranslateX", 0.9, 3)),
        (("ShearX", 0.1, 6), ("Invert", 0.6, None)),
        (("Solarize", 0.7, 2), ("TranslateY", 0.6, 7)),
        (("ShearY", 0.8, 4), ("Invert", 0.8, None)),
        (("ShearX", 0.7, 9), ("TranslateY", 0.8, 3)),
        (("ShearY", 0.8, 5), ("AutoContrast", 0.7, None)),
        (("ShearX", 0.7, 2), ("Invert", 0.1, None)),
    ],
}


class AutoAugment:
    """The **learned** one: twenty-five pairs of operations found by a search, each
    with its own probability and strength, one pair drawn per call.

    Nothing about the table is derivable — it is the output of the search, which is why
    it is written out rather than computed, and why the three datasets are three
    tables.
    """

    def __init__(self, policy=AutoAugmentPolicy.IMAGENET, interpolation="nearest",
                 fill=None):
        self.policy = policy if isinstance(policy, AutoAugmentPolicy) else \
            AutoAugmentPolicy(policy)
        self.interpolation = _interpolation(interpolation)
        self.fill = fill
        self.policies = _POLICIES[self.policy.value]

    def get_params(self, transform_num):
        """Which pair, and **two probabilities and two signs** — drawn before anything
        is applied, because both halves of a pair are decided at once."""
        return (int(_rng.integers(0, transform_num)), _rng.random(2),
                _rng.integers(0, 2, 2))

    def __call__(self, img):
        img = _require_hwc(img, type(self).__name__)
        fill = _fill_per_channel(self.fill, img)
        which, probs, signs = self.get_params(len(self.policies))
        space = _space_auto(10, (img.shape[0], img.shape[1]))
        for i, (op_name, p, magnitude_id) in enumerate(self.policies[which]):
            if probs[i] <= p:
                magnitudes, signed = space[op_name]
                magnitude = (float(magnitudes[magnitude_id])
                             if magnitude_id is not None else 0.0)
                if signed and signs[i] == 0:
                    magnitude *= -1.0
                img = _apply_op(img, op_name, magnitude, self.interpolation, fill)
        return img

    def __repr__(self):
        return f"{type(self).__name__}(policy={self.policy}, fill={self.fill})"


class RandAugment:
    """The **uniform** one: `num_ops` operations drawn evenly, all at the same fixed
    strength.

    Its point is that the search was unnecessary — one strength dial and a count do as
    well as the learned table, which is why `magnitude` is a number you tune rather
    than a distribution.
    """

    def __init__(self, num_ops=2, magnitude=9, num_magnitude_bins=31,
                 interpolation="nearest", fill=None):
        self.num_ops = num_ops
        self.magnitude = magnitude
        self.num_magnitude_bins = num_magnitude_bins
        self.interpolation = _interpolation(interpolation)
        self.fill = fill

    def __call__(self, img):
        img = _require_hwc(img, type(self).__name__)
        fill = _fill_per_channel(self.fill, img)
        space = _space_rand(self.num_magnitude_bins, (img.shape[0], img.shape[1]))
        names = list(space)
        for _ in range(self.num_ops):
            op_name = names[int(_rng.integers(0, len(names)))]
            magnitudes, signed = space[op_name]
            magnitude = 0.0 if magnitudes is None else float(magnitudes[self.magnitude])
            if signed and _rng.integers(0, 2):
                magnitude *= -1.0
            img = _apply_op(img, op_name, magnitude, self.interpolation, fill)
        return img

    def __repr__(self):
        return (f"{type(self).__name__}(num_ops={self.num_ops}"
                f", magnitude={self.magnitude}"
                f", num_magnitude_bins={self.num_magnitude_bins}"
                f", interpolation={InterpolationMode(self.interpolation)}"
                f", fill={self.fill})")


class TrivialAugmentWide:
    """The one with **no dials at all**: one operation, drawn evenly, at a strength
    also drawn evenly from a wide ladder.

    That is the paper's claim — tuning the strength was never worth it — so there is no
    magnitude argument to pass. The ladder is much wider than `RandAugment`'s to make
    up for drawing it.
    """

    def __init__(self, num_magnitude_bins=31, interpolation="nearest", fill=None):
        self.num_magnitude_bins = num_magnitude_bins
        self.interpolation = _interpolation(interpolation)
        self.fill = fill

    def __call__(self, img):
        img = _require_hwc(img, type(self).__name__)
        fill = _fill_per_channel(self.fill, img)
        space = _space_trivial(self.num_magnitude_bins)
        names = list(space)
        op_name = names[int(_rng.integers(0, len(names)))]
        magnitudes, signed = space[op_name]
        magnitude = (0.0 if magnitudes is None
                     else float(magnitudes[int(_rng.integers(0, len(magnitudes)))]))
        if signed and _rng.integers(0, 2):
            magnitude *= -1.0
        return _apply_op(img, op_name, magnitude, self.interpolation, fill)

    def __repr__(self):
        return (f"{type(self).__name__}("
                f"num_magnitude_bins={self.num_magnitude_bins}"
                f", interpolation={InterpolationMode(self.interpolation)}"
                f", fill={self.fill})")


class AugMix:
    """The **blended** one: several independent chains of operations, mixed back into
    the original by weights drawn from a Dirichlet.

    That is what makes it different in kind from the other three — they replace the
    picture and this one **averages several versions of it with the original**, so the
    result stays close to the input however hard the chains hit.
    """

    _PARAMETER_MAX = 10

    def __init__(self, severity=3, mixture_width=3, chain_depth=-1, alpha=1.0,
                 all_ops=True, interpolation="bilinear", fill=None):
        if not 1 <= severity <= self._PARAMETER_MAX:
            raise ValueError(
                f"The severity must be between [1, {self._PARAMETER_MAX}]. "
                f"Got {severity} instead.\n"
                f"(torch: The severity must be between [1, {self._PARAMETER_MAX}].)")
        self.severity = severity
        self.mixture_width = mixture_width
        self.chain_depth = chain_depth
        self.alpha = alpha
        self.all_ops = all_ops
        self.interpolation = _interpolation(interpolation)
        self.fill = fill

    def _dirichlet(self, params):
        return _rng.dirichlet(params)

    def __call__(self, img):
        img = _require_hwc(img, type(self).__name__)
        fill = _fill_per_channel(self.fill, img)
        space = _space_augmix(self._PARAMETER_MAX, (img.shape[0], img.shape[1]),
                              self.all_ops)
        names = list(space)
        work = _working_dtype(img.dtype)

        # **Two Dirichlet draws and not one.** The first splits the picture from the
        # chains; the second splits the chains among themselves. Drawing one over
        # `mixture_width + 1` would look equivalent and give a different distribution.
        m = self._dirichlet([self.alpha, self.alpha])
        weights = self._dirichlet([self.alpha] * self.mixture_width) * m[1]

        mixed = img.astype(work) * m[0]
        for i in range(self.mixture_width):
            chain = img
            depth = (self.chain_depth if self.chain_depth > 0
                     else int(_rng.integers(1, 4)))
            for _ in range(depth):
                op_name = names[int(_rng.integers(0, len(names)))]
                magnitudes, signed = space[op_name]
                magnitude = (0.0 if magnitudes is None else
                             float(magnitudes[int(_rng.integers(0, self.severity))]))
                if signed and _rng.integers(0, 2):
                    magnitude *= -1.0
                chain = _apply_op(chain, op_name, magnitude, self.interpolation, fill)
            mixed = mixed + weights[i] * chain.astype(work)
        # **Truncated, not rounded**, because torch casts with `.to(dtype)` here where
        # the convolutions round. Two casts, two rules, one file.
        return mixed.astype(img.dtype)

    def __repr__(self):
        return (f"{type(self).__name__}(severity={self.severity}"
                f", mixture_width={self.mixture_width}"
                f", chain_depth={self.chain_depth}"
                f", alpha={self.alpha}"
                f", all_ops={self.all_ops}"
                f", interpolation={InterpolationMode(self.interpolation)}"
                f", fill={self.fill})")


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


# --- ops: the box geometry, and only that -----------------------------------
#
# **Eleven of torchvision's thirty-nine.** The other twenty-eight are `nn.Module`
# layers and the functions that need a model's feature maps, and those need a
# detector nobody here has. These eleven need nothing but four numbers a box: they
# are deterministic, they compare against real torchvision exactly, and the person in
# front of them is somebody working out what IoU and NMS actually compute rather than
# somebody running a detector.
#
# **Boxes are `(N, 4)` and the format is a named argument, not a guess.** `xyxy` is
# two corners, `xywh` is a corner and a size, `cxcywh` is a centre and a size. The
# three are indistinguishable by inspection — four numbers either way — so a wrong
# `fmt` is a wrong answer that raises nothing.


_BOX_FORMATS = ("xyxy", "xywh", "cxcywh")


def _boxes_in(boxes):
    """Numpy, and **remember whether a tensor came in.** These take and return the
    kind they were given, as `Normalize` does — a caller who has tensors should not
    have to unwrap them to ask a question about geometry."""
    if isinstance(boxes, _np.ndarray):
        return boxes.astype(_np.float64), False
    return _to_numpy(boxes).astype(_np.float64), True


def _boxes_out(values, was_tensor, dtype=None):
    out = values.astype(dtype) if dtype is not None else values
    return _backend().tensor(out) if was_tensor else out


def _to_xyxy(boxes, fmt):
    if fmt not in _BOX_FORMATS:
        raise ValueError(
            f"Unsupported Bounding Box format {fmt} — it is one of "
            f"{', '.join(_BOX_FORMATS)}.\n"
            f"(torch: Unsupported Bounding Box area for given format {fmt})")
    if fmt == "xyxy":
        return boxes
    if fmt == "xywh":
        x, y, w, h = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
        return _np.stack((x, y, x + w, y + h), axis=-1)
    cx, cy, w, h = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
    return _np.stack((cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h), axis=-1)


def box_convert(boxes, in_fmt, out_fmt):
    """Between the three spellings of a box. **The identity is a copy and not the
    same array** — torchvision returns a new tensor even when the formats match, and
    a caller who mutates the result should not reach the caller's boxes."""
    for name, fmt in (("in_fmt", in_fmt), ("out_fmt", out_fmt)):
        if fmt not in _BOX_FORMATS:
            raise ValueError(
                f"Unsupported Bounding Box Conversions for given {name} {fmt}.\n"
                "(torch: Unsupported Bounding Box Conversions for given in_fmt and "
                "out_fmt)")
    arr, was_tensor = _boxes_in(boxes)
    xyxy = _to_xyxy(arr, in_fmt)
    if out_fmt == "xyxy":
        out = xyxy.copy()
    else:
        x1, y1, x2, y2 = xyxy[..., 0], xyxy[..., 1], xyxy[..., 2], xyxy[..., 3]
        w, h = x2 - x1, y2 - y1
        out = (_np.stack((x1, y1, w, h), axis=-1) if out_fmt == "xywh"
               else _np.stack((x1 + 0.5 * w, y1 + 0.5 * h, w, h), axis=-1))
    return _boxes_out(out, was_tensor, _np.float32)


def box_area(boxes, fmt="xyxy"):
    """Width times height. **A box with `x2 < x1` gets a negative area** rather than
    zero — torchvision does not clamp here, and clamping would hide a box built the
    wrong way round."""
    arr, was_tensor = _boxes_in(boxes)
    xyxy = _to_xyxy(arr, fmt)
    return _boxes_out((xyxy[..., 2] - xyxy[..., 0]) * (xyxy[..., 3] - xyxy[..., 1]),
                      was_tensor, _np.float32)


def _inter_union(a, b):
    """Intersection and union of every box in `a` against every box in `b`."""
    lt = _np.maximum(a[..., None, :2], b[..., None, :, :2])
    rb = _np.minimum(a[..., None, 2:], b[..., None, :, 2:])
    wh = _np.clip(rb - lt, 0.0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    return inter, area_a[..., None] + area_b[..., None, :] - inter


def box_iou(boxes1, boxes2, fmt="xyxy"):
    """**An `N x M` matrix, not a paired list.** Every box against every box, which is
    what a detector needs and what surprises everyone the first time."""
    a, was_tensor = _boxes_in(boxes1)
    b, _ = _boxes_in(boxes2)
    inter, union = _inter_union(_to_xyxy(a, fmt), _to_xyxy(b, fmt))
    return _boxes_out(inter / union, was_tensor, _np.float32)


def generalized_box_iou(boxes1, boxes2):
    """IoU, **minus what the smallest enclosing box wastes.** Two boxes that do not
    touch have an IoU of 0 whatever the distance between them; this one keeps falling
    to -1, which is why a loss can be built on it and not on IoU."""
    a, was_tensor = _boxes_in(boxes1)
    b, _ = _boxes_in(boxes2)
    inter, union = _inter_union(a, b)
    iou = inter / union
    lt = _np.minimum(a[..., None, :2], b[..., None, :, :2])
    rb = _np.maximum(a[..., None, 2:], b[..., None, :, 2:])
    wh = _np.clip(rb - lt, 0.0, None)
    area = wh[..., 0] * wh[..., 1]
    return _boxes_out(iou - (area - union) / area, was_tensor, _np.float32)


def _centre_distance(a, b):
    """Squared distance between centres, and the squared diagonal of the enclosing
    box — the two halves both distance-based IoUs need."""
    lt = _np.minimum(a[..., None, :2], b[..., None, :, :2])
    rb = _np.maximum(a[..., None, 2:], b[..., None, :, 2:])
    wh = _np.clip(rb - lt, 0.0, None)
    diagonal = wh[..., 0] ** 2 + wh[..., 1] ** 2
    cx_a = (a[..., 0] + a[..., 2]) / 2
    cy_a = (a[..., 1] + a[..., 3]) / 2
    cx_b = (b[..., 0] + b[..., 2]) / 2
    cy_b = (b[..., 1] + b[..., 3]) / 2
    centres = ((cx_a[..., None] - cx_b[..., None, :]) ** 2
               + (cy_a[..., None] - cy_b[..., None, :]) ** 2)
    return centres, diagonal


def distance_box_iou(boxes1, boxes2, eps=1e-7):
    """IoU penalised by **how far apart the centres are**, as a fraction of the
    enclosing box's diagonal."""
    a, was_tensor = _boxes_in(boxes1)
    b, _ = _boxes_in(boxes2)
    inter, union = _inter_union(a, b)
    centres, diagonal = _centre_distance(a, b)
    return _boxes_out(inter / union - centres / (diagonal + eps), was_tensor,
                      _np.float32)


def complete_box_iou(boxes1, boxes2, eps=1e-7):
    """`distance_box_iou` and **one more term for the aspect ratio** — two boxes with
    the same centre and area but different shapes score lower here and identically
    under the distance one."""
    a, was_tensor = _boxes_in(boxes1)
    b, _ = _boxes_in(boxes2)
    inter, union = _inter_union(a, b)
    iou = inter / union
    centres, diagonal = _centre_distance(a, b)
    diou = iou - centres / (diagonal + eps)
    w_a, h_a = a[..., 2] - a[..., 0], a[..., 3] - a[..., 1]
    w_b, h_b = b[..., 2] - b[..., 0], b[..., 3] - b[..., 1]
    v = (4 / (_np.pi ** 2)) * (_np.arctan(w_b / h_b)[..., None, :]
                               - _np.arctan(w_a / h_a)[..., None]) ** 2
    with _np.errstate(invalid="ignore"):
        alpha = v / (1 - iou + v + eps)
    return _boxes_out(diou - alpha * v, was_tensor, _np.float32)


def clip_boxes_to_image(boxes, size):
    """Push every corner back inside a picture of `size`, which is **(height, width)**
    — the opposite order to a box's own `(x, y)`, and torchvision's own convention."""
    arr, was_tensor = _boxes_in(boxes)
    height, width = size
    out = arr.copy()
    out[..., 0::2] = _np.clip(out[..., 0::2], 0, width)
    out[..., 1::2] = _np.clip(out[..., 1::2], 0, height)
    return _boxes_out(out, was_tensor, _np.float32)


def remove_small_boxes(boxes, min_size):
    """**Indices, not boxes.** Every one of these that filters returns the positions
    rather than the survivors, because the caller almost always has scores and labels
    to filter by the same positions."""
    arr, was_tensor = _boxes_in(boxes)
    keep = ((arr[..., 2] - arr[..., 0]) >= min_size) & \
           ((arr[..., 3] - arr[..., 1]) >= min_size)
    return _boxes_out(_np.nonzero(keep)[0], was_tensor, _np.int64)


def masks_to_boxes(masks):
    """The tightest box around each mask. **An empty mask gives all zeros** rather
    than an error — torchvision's behaviour, and the one that lets a batch with a
    blank mask in it still stack."""
    arr = masks if isinstance(masks, _np.ndarray) else _to_numpy(masks)
    was_tensor = not isinstance(masks, _np.ndarray)
    out = _np.zeros((arr.shape[0], 4), dtype=_np.float64)
    for i in range(arr.shape[0]):
        ys, xs = _np.nonzero(arr[i])
        if xs.size:
            out[i] = (xs.min(), ys.min(), xs.max(), ys.max())
    return _boxes_out(out, was_tensor, _np.int64 if arr.dtype != _np.float32 else _np.float32)


def nms(boxes, scores, iou_threshold):
    """Non-maximum suppression: keep the best-scoring box, throw away everything that
    overlaps it too much, repeat.

    **`> iou_threshold` and not `>=`.** At a threshold of 0 two boxes that merely
    touch — zero overlap — both survive, and that is the boundary anybody testing this
    reaches for first.

    Ties in the score are **not decided here and torchvision does not decide them
    either**; its own documentation says the choice is not guaranteed to match between
    CPU and GPU. So a case built on tied scores is a case with no answer.
    """
    arr, was_tensor = _boxes_in(boxes)
    values = _to_numpy(scores) if not isinstance(scores, _np.ndarray) else scores
    order = _np.argsort(-_np.asarray(values, dtype=_np.float64), kind="stable")
    kept = []
    while order.size:
        best = order[0]
        kept.append(best)
        if order.size == 1:
            break
        rest = order[1:]
        inter, union = _inter_union(arr[best][None, :], arr[rest])
        overlap = (inter / union)[0]
        order = rest[overlap <= iou_threshold]
    return _boxes_out(_np.asarray(kept, dtype=_np.int64), was_tensor, _np.int64)


def batched_nms(boxes, scores, idxs, iou_threshold):
    """NMS **per class**, done by moving each class's boxes somewhere the others
    cannot reach.

    The offset trick is torchvision's and it is worth reading twice: every box is
    shifted by its class index times more than the largest coordinate, so boxes of
    different classes can no longer overlap and a single pass of `nms` does the lot.
    """
    arr, was_tensor = _boxes_in(boxes)
    if arr.size == 0:
        return _boxes_out(_np.zeros((0,), dtype=_np.int64), was_tensor, _np.int64)
    labels = _np.asarray(_to_numpy(idxs) if not isinstance(idxs, _np.ndarray) else idxs,
                         dtype=_np.float64)
    offsets = labels * (arr.max() + 1)
    return nms(arr + offsets[:, None], scores, iou_threshold)


# --- transforms.v2 ----------------------------------------------------------
#
# **torchvision's current recommended API, and it is v1 with a different surface.**
# Measured before any of this was written: on a plain image v2's transforms give the
# same values as v1's (`Resize` v1 against v2, max difference 0.0). So the arithmetic
# is not written twice — every class here inherits its behaviour from the one above.
#
# **What is written twice is the `repr`, because that genuinely differs.** Measured:
# of the 33 transforms comparable in both, **21 print differently**.
#
#     v1: Resize(size=(4, 3), interpolation=bilinear, max_size=None, antialias=True)
#     v2: Resize(size=[4, 3], interpolation=bilinear, antialias=True)
#
#     v1: ColorJitter(brightness=(0.5, 1.5), contrast=None, saturation=None, hue=None)
#     v2: ColorJitter(brightness=(0.5, 1.5))
#
# The plan for this namespace was "re-export the 38, the values are the same" — and
# that was a decision made from a measurement of **values** which a measurement of the
# **surface** then refuted. This project treats `repr` as specification, because
# tutorials print transforms and a learner reads what is printed. Twenty-one wrong
# ones is not a rounding.
#
# ## The rule is one rule, and torchvision's own
#
# v2 does not hand-write those reprs. `Transform.extra_repr` walks the instance's
# attributes, skips the private ones, keeps only what is a bool, number, string,
# tuple, list or enum, and joins them. **That is why `ColorJitter` drops three
# fields** — they are `None`, and `None` is not in the list. So the difference between
# the two namespaces is not the printing but **what each class stores, under what name
# and in what order**, and that is what the table below records.
#
# Every one of the 38 declares its printed fields, including the twelve that happen to
# agree with v1 today. Leaving those to inherit would make this file depend on a
# coincidence, and a coincidence is what stops holding without telling anyone.


class _V2Repr:
    """v2's printing rule, once.

    The fields are declared rather than read off `__dict__` because the behaviour
    comes from the v1 class above, whose attributes are its own — same values, other
    names, other order. Declaring keeps one implementation of the arithmetic and one
    statement of the surface, which is the whole shape of this namespace.
    """

    _shown: dict = {}

    def _v2(self, **fields):
        self.__dict__["_shown"] = fields

    def __repr__(self):
        # **The type filter is torchvision's and it is load-bearing**: `None` is not
        # among the kinds it keeps, so a field left unset disappears from the line
        # rather than printing as `None`. That single rule is most of the difference
        # between the two namespaces.
        parts = [f"{name}={value}" for name, value in self._shown.items()
                 if isinstance(value, (bool, int, float, str, tuple, list, _enum.Enum))]
        return f"{type(self).__name__}({', '.join(parts)})"


def _v2_twin(base, fields):
    """A v2 class: the v1 one's behaviour, the v2 one's printed surface.

    **Built from a table rather than written out thirty-eight times.** The repository
    prefers three similar lines to an early abstraction, and thirty-eight is not
    three — every one of these would be the same four lines with two names changed,
    and the thing that actually differs between them is the field list, which is what
    the table holds.

    The name is set on the class because `repr` reads `type(self).__name__`, and a
    twin called `_V2Resize` would print itself under a name torchvision does not have.
    """
    class Twin(_V2Repr, base):
        def __init__(self, *args, **kwargs):
            base.__init__(self, *args, **kwargs)
            self._v2(**{name: read(self) for name, read in fields})

    Twin.__name__ = base.__name__
    Twin.__qualname__ = base.__name__
    Twin.__doc__ = (f"`transforms.v2.{base.__name__}` — {base.__name__}'s behaviour "
                    "with v2's printed surface.")
    return Twin


def _listed(value):
    """v2 stores several sizes as **lists** where v1 keeps tuples, and the repr shows
    the difference. One number stays one number."""
    return value if isinstance(value, (int, float)) else list(value)

# The table. Each row is `(class, ((printed name, how to read it), ...))` in **v2's
# order**, which is the order its constructor assigns and therefore the order its repr
# prints. Read out of real torchvision rather than inferred — the same discipline the
# policy tables got, and for the same reason: every plausible ordering is plausible.
_V2_TABLE = (
    # **One number is a list too** — v2 prints `Resize(5)` as `size=[5]`. `_listed`
    # leaves a number a number, so the size is read differently here and only here.
    (Resize, (("size", lambda s: [s.size] if isinstance(s.size, int) else list(s.size)),
              ("interpolation", lambda s: s.interpolation),
              ("antialias", lambda s: s.antialias))),
    (CenterCrop, (("size", lambda s: s.size),)),
    (RandomCrop, (("size", lambda s: s.size),
                  ("pad_if_needed", lambda s: s.pad_if_needed),
                  ("fill", lambda s: s.fill),
                  ("padding_mode", lambda s: s.padding_mode))),
    (RandomResizedCrop, (("size", lambda s: s.size), ("scale", lambda s: tuple(s.scale)),
                         ("ratio", lambda s: tuple(s.ratio)),
                         ("interpolation", lambda s: s.interpolation),
                         ("antialias", lambda s: s.antialias))),
    (FiveCrop, (("size", lambda s: s.size),)),
    (TenCrop, (("size", lambda s: s.size), ("vertical_flip", lambda s: s.vertical_flip))),
    (Pad, (("padding", lambda s: s.padding), ("fill", lambda s: s.fill),
           ("padding_mode", lambda s: s.padding_mode))),
    (RandomHorizontalFlip, (("p", lambda s: s.p),)),
    (RandomVerticalFlip, (("p", lambda s: s.p),)),
    (Grayscale, (("num_output_channels", lambda s: s.num_output_channels),)),
    (RandomGrayscale, (("p", lambda s: s.p),)),
    (Normalize, (("mean", lambda s: _listed(s.mean)), ("std", lambda s: _listed(s.std)),
                 ("inplace", lambda s: s.inplace))),
    (RandomErasing, (("p", lambda s: s.p), ("scale", lambda s: tuple(s.scale)),
                     ("ratio", lambda s: tuple(s.ratio)),
                     ("value", lambda s: (s.value if isinstance(s.value, str)
                                          else [float(v) for v in _np.atleast_1d(s.value)])),
                     ("inplace", lambda s: s.inplace))),
    # **`ColorJitter` is the clearest case of the type filter doing the work.** It
    # stores `None` for a factor nobody asked for, and `None` is not a kind v2 prints,
    # so the default one prints its own name and nothing else.
    (ColorJitter, (("brightness", lambda s: s.brightness),
                   ("contrast", lambda s: s.contrast),
                   ("saturation", lambda s: s.saturation),
                   ("hue", lambda s: s.hue))),
    (RandomInvert, (("p", lambda s: s.p),)),
    (RandomPosterize, (("p", lambda s: s.p), ("bits", lambda s: s.bits))),
    (RandomSolarize, (("p", lambda s: s.p), ("threshold", lambda s: s.threshold))),
    (RandomAutocontrast, (("p", lambda s: s.p),)),
    (RandomEqualize, (("p", lambda s: s.p),)),
    (RandomAdjustSharpness, (("p", lambda s: s.p),
                             ("sharpness_factor", lambda s: s.sharpness_factor))),
    (RandomRotation, (("degrees", lambda s: _listed(s.degrees)),
                      ("interpolation", lambda s: s.interpolation),
                      ("expand", lambda s: s.expand), ("fill", lambda s: s.fill))),
    (RandomAffine, (("degrees", lambda s: _listed(s.degrees)),
                    ("interpolation", lambda s: s.interpolation),
                    ("fill", lambda s: s.fill))),
    (RandomPerspective, (("p", lambda s: s.p),
                         ("distortion_scale", lambda s: s.distortion_scale),
                         ("interpolation", lambda s: s.interpolation),
                         ("fill", lambda s: s.fill))),
    (GaussianBlur, (("kernel_size", lambda s: tuple(s.kernel_size)),
                    ("sigma", lambda s: _listed(s.sigma)))),
    # **The policies put `interpolation` first**, where every other class here has it
    # after the arguments that decide the picture. That is v2's assignment order and
    # not a tidier one.
    (AutoAugment, (("interpolation", lambda s: s.interpolation),
                   ("policy", lambda s: s.policy))),
    (RandAugment, (("interpolation", lambda s: s.interpolation),
                   ("num_ops", lambda s: s.num_ops),
                   ("magnitude", lambda s: s.magnitude),
                   ("num_magnitude_bins", lambda s: s.num_magnitude_bins))),
    (TrivialAugmentWide, (("interpolation", lambda s: s.interpolation),
                          ("num_magnitude_bins", lambda s: s.num_magnitude_bins))),
    (AugMix, (("interpolation", lambda s: s.interpolation),
              ("severity", lambda s: s.severity),
              ("mixture_width", lambda s: s.mixture_width),
              ("chain_depth", lambda s: s.chain_depth),
              ("alpha", lambda s: s.alpha), ("all_ops", lambda s: s.all_ops))),
    # **These two print nothing at all.** Their state is arrays and functions, and
    # neither is a kind v2's rule keeps — so the name and empty brackets is the whole
    # of it, which is easy to mistake for an unfinished repr.
    (LinearTransformation, ()),
    (ToTensor, ()),
    (RandomOrder, (("transforms", lambda s: list(s.transforms)),)),
    # **v2 fills `p` in.** v1 leaves it `None`; v2 builds the uniform distribution and
    # stores it, so two transforms given no probabilities print `p=[0.5, 0.5]`.
    (RandomChoice, (("transforms", lambda s: list(s.transforms)),
                    ("p", lambda s: (list(s.p) if s.p is not None
                                     else [1 / len(s.transforms)] * len(s.transforms))))),
)

def _module_repr(name, lines):
    """torch's `nn.Module` printing, which is what v2's containers inherit.

    **One transform prints inline and two print over several lines**, and the indent
    is not the same in the two cases — four spaces inline, six once it breaks, because
    torch indents each line of `extra_repr` by two more when it wraps. Measured rather
    than derived; it is the kind of thing nobody would guess and everybody would get
    almost right.
    """
    body = "\n".join(f"    {line}" for line in lines)
    if "\n" not in body:
        return f"{name}({body})"
    inner = "\n".join(f"  {line}" for line in body.split("\n"))
    return f"{name}(\n{inner}\n)"


class _V2ElasticTransform(_V2Repr, ElasticTransform):
    """**The one whose printed `fill` cannot be read back off the object.** v1
    normalises it to a list in the constructor and v2 prints the number as it was
    given, so the twin keeps the argument rather than recovering it — recovering it
    would turn `0` into `0.0`, and the two print differently."""

    def __init__(self, alpha=50.0, sigma=5.0, interpolation="bilinear", fill=0):
        ElasticTransform.__init__(self, alpha, sigma, interpolation, fill)
        self._v2(alpha=_listed(self.alpha), sigma=_listed(self.sigma),
                 interpolation=self.interpolation, fill=fill)


class _V2Compose(Compose):
    """v2's `Compose`. Same behaviour, torch's module printing."""

    def __repr__(self):
        return _module_repr(type(self).__name__, [str(t) for t in self.transforms])


class _V2RandomApply(RandomApply):
    """v2's `RandomApply`. **`p` is not printed**, unlike v1's — it is stored and left
    out, which is torch's module repr showing only what `extra_repr` returns."""

    def __repr__(self):
        return _module_repr(type(self).__name__, [str(t) for t in self.transforms])


class _V2Lambda(_V2Repr, Lambda):
    """v2's `Lambda` takes **the types it applies to** as well as the function — the
    one place in this namespace where the constructor differs, not just the printing.
    Given tv_tensors it would run only on the kinds named; here there is one kind, so
    the argument is kept and recorded rather than acted on."""

    def __init__(self, lambd, *types):
        Lambda.__init__(self, lambd)
        self.types = types or (object,)
        self._v2()

    def __repr__(self):
        names = [t.__name__ for t in self.types]
        return f"{type(self).__name__}({self.lambd.__name__}, types={names})"


class _V2Identity(_V2Repr):
    """Does nothing, and **that is a transform** — it is what a policy draws when it
    draws no operation, and what a `Compose` holds when a branch is switched off."""

    def __init__(self):
        self._v2()

    def __call__(self, x):
        return x


class _V2ToPureTensor(_V2Repr):
    """Strips the tv_tensor wrappers off a sample. **Here there are none**, so it is
    the identity — kept because a pipeline copied from torchvision ends with it and
    should not stop, and named rather than aliased to `Identity` because the two mean
    different things the day tv_tensors arrive."""

    def __init__(self):
        self._v2()

    def __call__(self, x):
        return x


class _V2RGB(_V2Repr):
    """One channel to three. A three-channel picture passes through, which is what
    makes it safe to put in front of a model that needs three."""

    def __init__(self):
        self._v2()

    def __call__(self, img):
        img = _require_hwc(img, "RGB")
        arr = img if img.ndim == 3 else img[:, :, None]
        if arr.shape[2] == 3:
            return arr
        if arr.shape[2] != 1:
            raise ValueError(
                f"RGB takes a 1- or 3-channel picture — it received {arr.shape[2]}.")
        return _np.ascontiguousarray(_np.repeat(arr, 3, axis=2))


class _V2ToImage(_V2Repr):
    """`(H,W,C)` to a `(C,H,W)` tensor — **and it does not divide by 255.**

    That is the whole reason v2 split `ToTensor` in two. `ToTensor` both moved the
    axes and scaled, so a float image was scaled a second time by anyone who did not
    know; here the moving is one transform and the scaling is `ToDtype(scale=True)`,
    and each says which it does.

    One difference from torchvision, and it is the core's, not this file's: given a
    `uint8` picture torchvision hands back a `uint8` tensor and this hands back an
    `int64` one, because the core has no `uint8` storage to hand back. The numbers are
    the same 0..255; only the box they sit in is wider. It shows in memory and in
    `.dtype`, and it stops mattering the moment `ToDtype(float32, scale=True)` runs —
    that pair, which is what v2 tells you to write instead of `ToTensor`, agrees with
    torchvision to 6e-08 and with v1 `ToTensor` exactly.
    """

    def __init__(self):
        self._v2()

    def __call__(self, pic):
        arr = _np.asarray(pic)
        if arr.ndim == 2:
            arr = arr[:, :, None]
        return _backend().tensor(_np.ascontiguousarray(arr.transpose(2, 0, 1)))


class _V2ToDtype(_V2Repr):
    """Cast, and **optionally scale on the way.** `scale=True` is the half of the old
    `ToTensor` that divided; without it this only changes the dtype, which is why the
    flag is not a default."""

    def __init__(self, dtype, scale=False):
        self.dtype = dtype
        self.scale = scale
        self._v2(scale=scale)

    def __call__(self, x):
        arr = x if isinstance(x, _np.ndarray) else _to_numpy(x)
        target = _np.dtype(self.dtype.np if hasattr(self.dtype, "np") else self.dtype)
        if self.scale and _np.dtype(arr.dtype).kind != "f" and target.kind == "f":
            out = arr.astype(target) / 255.0
        elif self.scale and _np.dtype(arr.dtype).kind == "f" and target.kind != "f":
            out = (arr * 255.0).astype(target)
        else:
            out = arr.astype(target)
        return out if isinstance(x, _np.ndarray) else _backend().tensor(out)


class _V2GaussianNoise(_V2Repr):
    """Add normal noise. **Float pictures only** — torchvision has an integer path
    that works in int16 and clamps, and a uint8 picture with sigma in units of [0,1]
    is a different question, so it is refused here rather than answered differently."""

    def __init__(self, mean=0.0, sigma=0.1, clip=True):
        if sigma < 0:
            raise ValueError(
                f"sigma shouldn't be negative. Got {sigma}\n"
                "(torch: sigma shouldn't be negative)")
        self.mean, self.sigma, self.clip = mean, sigma, clip
        self._v2(mean=mean, sigma=sigma, clip=clip)

    def __call__(self, img):
        img = _require_hwc(img, "GaussianNoise")
        if _np.dtype(img.dtype).kind != "f":
            raise TypeError(
                f"GaussianNoise takes a float picture — it received {img.dtype}.\n"
                "  Its sigma is in the units of a normalised image; on bytes the same "
                "number means something else.")
        out = img + (self.mean + _rng.standard_normal(img.shape).astype(img.dtype)
                     * self.sigma)
        return _np.clip(out, 0.0, 1.0) if self.clip else out


class _V2RandomChannelPermutation(_V2Repr):
    """Shuffle the channels. **Every ordering including the original** — it is a draw
    over permutations, not a guarantee of change."""

    def __init__(self):
        self._v2()

    def __call__(self, img):
        img = _require_hwc(img, "RandomChannelPermutation")
        arr = img if img.ndim == 3 else img[:, :, None]
        order = _rng.permutation(arr.shape[2])
        return _np.ascontiguousarray(arr[:, :, order])


class _V2RandomPhotometricDistort(_V2Repr):
    """The SSD recipe: each of four adjustments applied with probability `p`, **the
    contrast either before or after the other two**, and then maybe a channel shuffle.

    The contrast's position is itself a coin flip, which is the part that reads as a
    detail and is not: contrast measures the picture's mean, so doing it first and
    doing it last are different pictures.
    """

    def __init__(self, brightness=(0.875, 1.125), contrast=(0.5, 1.5),
                 saturation=(0.5, 1.5), hue=(-0.05, 0.05), p=0.5):
        self.brightness, self.contrast = brightness, contrast
        self.saturation, self.hue, self.p = saturation, hue, p
        self._v2(brightness=brightness, contrast=contrast, hue=hue,
                 saturation=saturation, p=p)

    def __call__(self, img):
        img = _require_hwc(img, "RandomPhotometricDistort")
        draw = lambda span: (float(_rng.uniform(span[0], span[1]))
                             if _rng.random() < self.p else None)
        brightness, contrast = draw(self.brightness), draw(self.contrast)
        saturation, hue = draw(self.saturation), draw(self.hue)
        contrast_first = bool(_rng.random() < 0.5)
        shuffle = _rng.random() < self.p
        if brightness is not None:
            img = adjust_brightness(img, brightness)
        if contrast is not None and contrast_first:
            img = adjust_contrast(img, contrast)
        if saturation is not None:
            img = adjust_saturation(img, saturation)
        if hue is not None:
            img = adjust_hue(img, hue)
        if contrast is not None and not contrast_first:
            img = adjust_contrast(img, contrast)
        if shuffle:
            arr = img if img.ndim == 3 else img[:, :, None]
            img = _np.ascontiguousarray(arr[:, :, _rng.permutation(arr.shape[2])])
        return img


class _V2RandomResize(_V2Repr):
    """Resize the short side to a number drawn from `[min_size, max_size)`. **A range
    of sizes rather than one**, which is what multi-scale training wants."""

    def __init__(self, min_size, max_size, interpolation="bilinear", antialias=True):
        self.min_size, self.max_size = min_size, max_size
        self.interpolation = _interpolation(interpolation)
        self.antialias = _antialias(antialias)
        self._v2(min_size=min_size, max_size=max_size,
                 interpolation=self.interpolation, antialias=self.antialias)

    def __call__(self, img):
        img = _require_hwc(img, "RandomResize")
        size = int(_rng.integers(self.min_size, self.max_size))
        return Resize(size, self.interpolation)(img)


class _V2RandomShortestSize(_V2Repr):
    """The short side to one of `min_size`, **with the long side capped** by
    `max_size` — so a very wide picture is scaled by whichever of the two constraints
    binds first, rather than by the short side alone."""

    def __init__(self, min_size, max_size=None, interpolation="bilinear",
                 antialias=True):
        self.min_size = ([int(min_size)] if isinstance(min_size, int)
                         else [int(s) for s in min_size])
        self.max_size = max_size
        self.interpolation = _interpolation(interpolation)
        self.antialias = _antialias(antialias)
        self._v2(min_size=self.min_size, max_size=max_size,
                 interpolation=self.interpolation, antialias=self.antialias)

    def __call__(self, img):
        img = _require_hwc(img, "RandomShortestSize")
        h, w = img.shape[0], img.shape[1]
        drawn = self.min_size[int(_rng.integers(0, len(self.min_size)))]
        ratio = drawn / min(h, w)
        if self.max_size is not None:
            ratio = min(ratio, self.max_size / max(h, w))
        return Resize((int(h * ratio), int(w * ratio)), self.interpolation)(img)


class _V2RandomZoomOut(_V2Repr):
    """Put the picture on a **larger canvas**, somewhere random on it, with the rest
    filled. The picture shrinks relative to the frame without being resampled — which
    is why detection recipes reach for it rather than for a scale-down."""

    def __init__(self, fill=0, side_range=(1.0, 4.0), p=0.5):
        if side_range[0] < 1.0 or side_range[0] > side_range[1]:
            raise ValueError(
                f"Invalid side range provided {tuple(side_range)}.\n"
                "(torch: Invalid canvas side range provided)")
        self.fill, self.side_range, self.p = fill, side_range, p
        self._v2(p=p, fill=fill, side_range=tuple(float(s) for s in side_range))

    def __call__(self, img):
        img = _require_hwc(img, "RandomZoomOut")
        if _rng.random() >= self.p:
            return img
        h, w = img.shape[0], img.shape[1]
        ratio = self.side_range[0] + _rng.random() * (self.side_range[1]
                                                      - self.side_range[0])
        canvas_w, canvas_h = int(w * ratio), int(h * ratio)
        draw = _rng.random(2)
        left = int((canvas_w - w) * draw[0])
        top = int((canvas_h - h) * draw[1])
        return Pad([left, top, canvas_w - (left + w), canvas_h - (top + h)],
                   self.fill)(img)


class _V2ScaleJitter(_V2Repr):
    """Resize toward `target_size` by a **drawn factor** — the large-scale jitter of
    the detection recipes, where the same picture is seen at a tenth and at twice its
    size across an epoch."""

    def __init__(self, target_size, scale_range=(0.1, 2.0), interpolation="bilinear",
                 antialias=True):
        self.target_size, self.scale_range = tuple(target_size), tuple(scale_range)
        self.interpolation = _interpolation(interpolation)
        self.antialias = _antialias(antialias)
        self._v2(target_size=self.target_size, scale_range=self.scale_range,
                 interpolation=self.interpolation, antialias=self.antialias)

    def __call__(self, img):
        img = _require_hwc(img, "ScaleJitter")
        h, w = img.shape[0], img.shape[1]
        scale = self.scale_range[0] + _rng.random() * (self.scale_range[1]
                                                       - self.scale_range[0])
        ratio = min(self.target_size[1] / h, self.target_size[0] / w) * scale
        return Resize((int(h * ratio), int(w * ratio)), self.interpolation)(img)


class _V2MixBase(_V2Repr):
    """What `MixUp` and `CutMix` share — **a batch, and labels that move with it.**

    These two are the only transforms in v2 whose input is a training pair rather
    than a picture. Everything else here takes `(H,W,C)` and gives `(H,W,C)` back;
    these take `(N,H,W,C)` with a label per row and give both back changed, because
    mixing two pictures without mixing their labels teaches the wrong thing.

    The pairing is **row `i` with row `i-1`** — a roll by one, not a random partner.
    torchvision says outright that this is an implementation detail and that the
    batch is expected to be shuffled already; kept the same here, because a recipe
    that shuffles for torchvision's sake would otherwise be silently unnecessary.

    `labels_getter` is **not taken.** In torchvision it exists to find the labels
    inside a nested sample — a dict, a tuple of tv_tensors — and there are no
    tv_tensors here, so the labels arrive as the second argument and nothing has to
    go looking. Taking the argument and ignoring it would read as support.
    """

    def __init__(self, *, alpha=1.0, num_classes=None):
        self.alpha = float(alpha)
        self.num_classes = num_classes
        self._v2(alpha=self.alpha, num_classes=num_classes)

    def _read(self, images, labels):
        """The two checks torchvision makes, in its words — a wrong batch here is a
        silent mis-train otherwise, since every shape involved is plausible."""
        images = _np.asarray(images)
        labels = _np.asarray(labels)
        if labels.ndim not in (1, 2):
            raise ValueError(
                f"labels should be index based with shape (batch_size,) "
                f"or probability based with shape (batch_size, num_classes), "
                f"but got a tensor of shape {labels.shape} instead.")
        if labels.ndim == 1 and self.num_classes is None:
            raise ValueError(
                "num_classes must be passed if the labels are index-based (1D)")
        if images.ndim != 4:
            raise ValueError(
                f"Expected a batched input with 4 dims, but got {images.ndim} "
                "dimensions instead.")
        if images.shape[0] != labels.shape[0]:
            raise ValueError(
                "The batch size of the image or video does not match the batch size "
                f"of the labels: {images.shape[0]} != {labels.shape[0]}.")
        return images, labels

    def _mix_label(self, labels, lam):
        """One-hot first if they came in as indices, then the same blend the pictures
        get. **`lam` weights the row itself and `1-lam` its partner** — the way round
        that matters, and the way round torchvision has it."""
        if labels.ndim == 1:
            hot = _np.zeros((labels.shape[0], self.num_classes), dtype=_np.float32)
            hot[_np.arange(labels.shape[0]), labels.astype(int)] = 1.0
            labels = hot
        labels = labels.astype(_np.float32, copy=False)
        return _np.roll(labels, 1, axis=0) * (1.0 - lam) + labels * lam


class _V2MixUp(_V2MixBase):
    """Blend each picture with the one before it, and their labels by the same weight.

    <https://arxiv.org/abs/1710.09412>. The whole transform is one weighted average,
    which is what makes it worth having: no crop, no resample, nothing to get subtly
    wrong, and it still moves a classifier's calibration.
    """

    def __call__(self, images, labels):
        images, labels = self._read(images, labels)
        lam = float(_rng.beta(self.alpha, self.alpha))
        mixed = (_np.roll(images, 1, axis=0) * (1.0 - lam)
                 + images * lam).astype(images.dtype, copy=False)
        return mixed, self._mix_label(labels, lam)


class _V2CutMix(_V2MixBase):
    """Paste a rectangle of the previous picture into this one, and mix the labels by
    **the area actually pasted** rather than by the weight that was drawn.

    <https://arxiv.org/abs/1905.04899>. That adjustment is the part worth pointing at:
    the box is centred on a random point and clipped at the edges, so a box near a
    corner loses half its area, and a label mixed by the drawn weight would then
    claim more of the other class than the picture contains.
    """

    def __call__(self, images, labels):
        images, labels = self._read(images, labels)
        lam = float(_rng.beta(self.alpha, self.alpha))
        h, w = images.shape[1], images.shape[2]
        centre_x, centre_y = int(_rng.integers(w)), int(_rng.integers(h))
        half = 0.5 * _math.sqrt(1.0 - lam)
        half_w, half_h = int(half * w), int(half * h)
        x1, y1 = max(centre_x - half_w, 0), max(centre_y - half_h, 0)
        x2, y2 = min(centre_x + half_w, w), min(centre_y + half_h, h)
        out = images.copy()
        out[:, y1:y2, x1:x2] = _np.roll(images, 1, axis=0)[:, y1:y2, x1:x2]
        by_area = float(1.0 - (x2 - x1) * (y2 - y1) / (w * h))
        return out, self._mix_label(labels, by_area)


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

# --------------------------------------------------------------------------- datasets
#
# **The refusal that used to stand here was about addresses, not about datasets.**
# What a browser cannot reach is torchvision's own hosts — `cs.toronto.edu` and
# `ossci-datasets.s3.amazonaws.com` send no CORS header, measured — and that is a fact
# about two servers. Reading the bytes once they are in hand needs no network at all,
# and this is that half: the decoders, and a download that works where a socket does.
#
# The formats are the whole content. MNIST's is IDX, four files of it; CIFAR's is a
# Python pickle inside a tar. Both are read here **from bytes**, so the same code
# serves a file on disk, a response body, and a golden case built in memory.

_IDX_DTYPES = {8: "u1", 9: "i1", 11: ">i2", 12: ">i4", 13: ">f4", 14: ">f8"}


def _read_idx(data):
    """IDX ("Pascal Vincent") bytes to an array. **Big endian, always.**

    The header is four bytes: two zero, then the type, then the number of axes; then
    one big-endian 32-bit length per axis. torchvision reads the same header by hand
    and then flips the bytes afterwards when the machine is little-endian — numpy does
    that in the dtype string, which is why `>i2` appears above and no flip appears
    below. The result is the same array; only who does the swapping differs.

    The lengths are **not** trusted to match the payload. torchvision passes
    `strict=False` from both its readers, so a file whose header promises more than it
    carries is reshaped and raises there rather than here; a file that carries more is
    truncated. Both are kept, because a divergence in what an malformed file does is
    the kind that only shows up on the day one arrives.
    """
    kind, axes = data[2], data[3]
    if kind not in _IDX_DTYPES:
        raise ValueError(f"unknown IDX type code {kind} — expected one of "
                         f"{sorted(_IDX_DTYPES)}")
    if not 1 <= axes <= 3:
        raise ValueError(f"IDX header says {axes} axes; 1 to 3 is the format")
    shape = [int.from_bytes(data[4 * (i + 1):4 * (i + 2)], "big") for i in range(axes)]
    flat = _np.frombuffer(data, dtype=_np.dtype(_IDX_DTYPES[kind]),
                          offset=4 * (axes + 1))
    want = int(_np.prod(shape))
    if flat.size < want:
        # **torch's sentence, because that is the one people search for.** Its own
        # readers pass `strict=False`, which relaxes an assert and not the reshape
        # underneath it, so a short file refuses there too — with these words. numpy's
        # own message names the same two numbers the other way round, and a phrase that
        # differs between the two implementations is a phrase nobody can grep for.
        raise ValueError(
            f"shape '{list(shape)}' is invalid for input of size {flat.size}\n"
            f"  the IDX header promises {want} values and the file carries "
            f"{flat.size} — it is truncated.")
    return flat[:want].reshape(shape)


def _read_idx_images(data):
    """The three-axis half, checked. The message is torchvision's."""
    out = _read_idx(data)
    if out.dtype != _np.uint8:
        raise TypeError(f"x should be of dtype uint8 instead of {out.dtype}")
    if out.ndim != 3:
        raise ValueError(f"x should have 3 dimensions instead of {out.ndim}")
    return out


def _read_idx_labels(data):
    """The one-axis half. **Widened to int64**, as torchvision's `.long()` does — the
    labels are bytes on disk and an index everywhere else."""
    out = _read_idx(data)
    if out.dtype != _np.uint8:
        raise TypeError(f"x should be of dtype uint8 instead of {out.dtype}")
    if out.ndim != 1:
        raise ValueError(f"x should have 1 dimension instead of {out.ndim}")
    return out.astype(_np.int64)


class _CifarUnpickler(_pickle.Unpickler):
    """A CIFAR batch is **a Python pickle that was downloaded**, and a pickle names the
    classes it will build before it builds them.

    torchvision calls `pickle.load` on it. That is safe as long as the file is the one
    the checksum names, and it is arbitrary code execution the moment it is not — a
    proxy, a mirror, a cache, a redirect. The checksum is verified here too and this
    is not a claim that it fails; it is that **a second lock costs ten lines** and the
    failure it prevents is the worst one in this file.

    So the classes are named instead of trusted. What a CIFAR batch legitimately needs
    is numpy's array reconstruction and its scalar type objects, nothing else.
    """

    _ALLOWED = {
        ("numpy", "dtype"), ("numpy", "ndarray"), ("numpy", "uint8"),
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy._core.multiarray", "_reconstruct"),
    }

    def find_class(self, module, name):
        if (module, name) not in self._ALLOWED:
            raise _pickle.UnpicklingError(
                f"a CIFAR batch asked for `{module}.{name}`, which is not one of the "
                "names this format needs.\n"
                "  A batch file is a pickle, so a name it asks for is code that will "
                "run. This refuses rather than runs it.")
        return super().find_class(module, name)


def _read_cifar_batch(data):
    """One CIFAR batch's `(images, labels)`, images as **(N,H,W,C) uint8**.

    On disk a batch is `(N, 3072)`, one row a picture, and the row is planar: 1024 red,
    then 1024 green, then 1024 blue. So it reshapes to `(N,3,32,32)` and then moves the
    channel last — torchvision does exactly this, and the transpose is the part that is
    silently survivable if it is got wrong, since a channel-swapped CIFAR still trains.

    The label key is `labels` in CIFAR-10 and `fine_labels` in CIFAR-100. Both are
    looked for, because the file does not say which of the two it is.
    """
    entry = _CifarUnpickler(_io.BytesIO(data), encoding="latin1").load()
    labels = entry.get("labels", entry.get("fine_labels"))
    if labels is None:
        raise ValueError("a CIFAR batch with neither `labels` nor `fine_labels` — "
                         f"it carries {sorted(entry)}")
    images = _np.asarray(entry["data"]).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    return _np.ascontiguousarray(images), [int(v) for v in labels]


# MATLAB level 5, the format `.mat` files are written in and `scipy.io.loadmat` reads.
#
# **This is here so that `scipy` does not have to be.** The gap table said of `SVHN`
# that "the refusal is the dependency, and it is the same answer PIL and a JPEG decoder
# get" — and that lumped two different things together. A JPEG decoder is a *codec*:
# thousands of lines, decades of edge cases, and no reasonable way to write one here.
# A `.mat` file is a **documented container** — a header, then tagged elements, with
# `zlib` around them — and the two arrays SVHN keeps in one are plain `uint8`. The
# whole reader below is under a hundred lines and uses nothing outside the standard
# library.
#
# So the line is not "no dependencies" but "**no dependency we could not have written
# in an afternoon**", and those two had been the same sentence.

_MAT_TYPES = {                                  # tag code → numpy dtype
    1: "i1", 2: "u1", 3: "i2", 4: "u2", 5: "i4", 6: "u4",
    7: "f4", 9: "f8", 12: "i8", 13: "u8", 16: "u1", 18: "i4",
}
_MAT_CLASS = {                                  # array class → numpy dtype
    6: "f8", 7: "f4", 8: "i1", 9: "u1", 10: "i2", 11: "u2",
    12: "i4", 13: "u4", 14: "i8", 15: "u8",
}


def _mat_element(buf, at, pad=False):
    """One tagged element: `(type, payload bytes, position after it)`.

    `pad` rounds the end up to eight bytes, which is right **inside** a matrix body
    and wrong between top-level elements — see the note below.

    **The small-element form is the trap.** When a value fits in four bytes MATLAB
    packs the size into the *high* half of the first word and the data into the next
    four bytes, with no second word at all — so a reader that always takes eight bytes
    of tag walks off by four and every field after it is garbage that still parses.
    """
    (word,) = _struct.unpack_from("<I", buf, at)
    if word >> 16:                              # small form: size in the high half
        kind, size, start = word & 0xFFFF, word >> 16, at + 4
        return kind, buf[start:start + size], at + 8
    kind, size = _struct.unpack_from("<II", buf, at)
    start = at + 8
    # **Padding to eight bytes applies inside a matrix and not between top-level
    # elements**, and applying it everywhere walks four bytes past the end of the
    # first compressed element. `scipy` writes each variable as its own `miCOMPRESSED`
    # block, so the second one begins in the middle of nothing: the reader found `X`,
    # then read `kind=0`, `kind=176`, `kind=1614` out of the tail and finally ran off
    # the buffer. `y` came back absent — a file that parses, pictures with no labels,
    # and nothing saying so.
    #
    # `pad` is passed by the caller, who knows which of the two it is in.
    end = start + ((size + 7) // 8 * 8 if pad else size)
    return kind, buf[start:start + size], end


def _mat_read(data):
    """`{name: ndarray}` from a MATLAB level-5 file. Numeric arrays only.

    Column-major on disk, so the shape is reversed and the axes transposed back —
    SVHN's `X` is `(32, 32, 3, N)` in MATLAB's order and has to stay that way, which
    is what torchvision then transposes.
    """
    if not data.startswith(b"MATLAB 5.0"):
        raise ValueError("not a MATLAB level-5 file — it does not begin `MATLAB 5.0`")
    out, at = {}, 128
    # **`at + 8 <= len` rather than `at < len`**, so a truncated file says so.
    # Reading to the last byte and letting `struct` fall off the end gives
    # `unpack_from requires a buffer of at least 1564 bytes` — a sentence about a
    # buffer that names no file and offers no move. A download cut halfway produces
    # exactly that, and so did a padding mistake in this reader.
    while at + 8 <= len(data):
        kind, payload, at = _mat_element(data, at)
        # **An element whose payload runs past the end is where a cut file lands.**
        # Carrying on gives `zlib.error: incomplete or truncated stream` from two
        # frames down — again a message about a stream rather than about a file. The
        # element is simply not there, so it is not read; the ones before it are, and
        # `SVHN` then refuses by name because `y` is missing.
        if at > len(data):
            break
        if kind == 15:                          # miCOMPRESSED
            # **One compressed element can hold several matrices**, and reading only
            # the first is a reader that finds `X` and loses `y` — a file that parses,
            # a dataset with pictures and no labels, and nothing saying so. Measured
            # against `scipy.io.loadmat`, which is the only reason it was noticed.
            inner, seen = _zlib.decompress(payload), 0
            while seen < len(inner):
                k2, p2, seen = _mat_element(inner, seen)
                if k2 != 14:
                    continue
                name, array = _mat_matrix(p2)
                if array is not None:
                    out[name] = array
        elif kind == 14:                        # miMATRIX, stored uncompressed
            name, array = _mat_matrix(payload)
            if array is not None:
                out[name] = array
    return out


def _mat_matrix(body):
    """`(name, ndarray)` for one matrix element, or `(name, None)` when it is not
    numeric — a struct or a cell array, which SVHN does not use and this does not
    invent an answer for."""
    _, flags, at = _mat_element(body, 0, pad=True)
    cls = flags[0] & 0xFF
    _, dims_raw, at = _mat_element(body, at, pad=True)
    dims = list(_np.frombuffer(dims_raw, dtype="<i4"))
    _, name_raw, at = _mat_element(body, at, pad=True)
    name = name_raw.decode("ascii", "replace")
    if cls not in _MAT_CLASS:
        return name, None
    kind, values, _ = _mat_element(body, at, pad=True)
    dtype = _MAT_TYPES.get(kind)
    if dtype is None:
        return name, None
    flat = _np.frombuffer(values, dtype="<" + dtype)
    # **MATLAB writes column-major.** Reading the dimensions forward and reshaping
    # row-major would give an array of the right size with every element in the wrong
    # place — a shape that agrees and values that do not.
    return name, flat.reshape(dims[::-1]).transpose().astype(_MAT_CLASS[cls])


def _md5(data):
    return _hashlib.md5(data).hexdigest()


def _md5_of_file(path, chunk=1 << 20):
    """The same digest **without holding the file**. The archives this checks are
    measured in hundreds of megabytes, and the point of streaming them in was not to
    read them whole a moment later."""
    running = _hashlib.md5()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            running.update(block)
    return running.hexdigest()


def _fetch(url, digest, timeout=60.0):
    """One file's bytes, **checked against the digest before they are returned.**

    Handing back unverified bytes and checking later is the version that goes wrong:
    the caller has already written them somewhere by then, and the next run finds a
    file of the right name holding the wrong thing. It is one function so that there
    is one place where bytes become trusted.

    For the small files only — every MNIST piece is under ten megabytes. `_fetch_to`
    is the one to reach for when the file is not.
    """
    with _urlreq.urlopen(url, timeout=timeout) as response:      # noqa: S310
        data = response.read()
    if digest is not None and _md5(data) != digest:
        raise RuntimeError(
            f"{url} came back with md5 {_md5(data)}, expected {digest}.\n"
            "  A mirror, a proxy or a captive portal serving something else is the "
            "usual cause; the file is not written.")
    return data


def _fetch_to(url, path, digest, timeout=60.0, chunk=1 << 20):
    """The same, **streamed to a file** and hashed as it goes.

    CIFAR-10 is 170MB. Read into memory it is 170MB of memory, and this is a library
    that also runs where that is most of what there is — so the bytes go past a
    megabyte at a time and only the digest is carried.

    The invariant from `_fetch` is kept by writing to `path + ".part"` and renaming
    **after** the digest agrees. Nothing sits at the real path unverified, which is
    what makes a failed download safe to simply run again: there is no half file to
    recognise, because a half file has the wrong name.
    """
    partial = path + ".part"
    running = _hashlib.md5()
    with _urlreq.urlopen(url, timeout=timeout) as response:      # noqa: S310
        with open(partial, "wb") as out:
            while True:
                block = response.read(chunk)
                if not block:
                    break
                running.update(block)
                out.write(block)
    if digest is not None and running.hexdigest() != digest:
        _os.unlink(partial)
        raise RuntimeError(
            f"{url} came back with md5 {running.hexdigest()}, expected {digest}.\n"
            "  A mirror, a proxy or a captive portal serving something else is the "
            "usual cause; the partial file is removed.")
    _os.replace(partial, path)
    return path


class VisionDataset:
    """The base every dataset here subclasses. **It holds the two transform slots.**

    `transform` acts on the picture and `target_transform` on the label; `transforms`
    takes both together, for the datasets where they cannot be decided apart — a crop
    that moves the boxes with it. Giving `transforms` alongside either of the others is
    an error rather than a precedence rule, which is torchvision's choice and worth
    keeping: a silent precedence is how you find out months later which one lost.
    """

    def __init__(self, root=None, transforms=None, transform=None,
                 target_transform=None):
        if transforms is not None and (transform is not None
                                       or target_transform is not None):
            raise ValueError("Only transforms or transform/target_transform can be "
                             "passed as argument")
        self.root = root
        self.transform, self.target_transform = transform, target_transform
        if transforms is None and (transform is not None
                                   or target_transform is not None):
            transforms = _StandardTransform(transform, target_transform)
        self.transforms = transforms

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError

    def extra_repr(self):
        return ""

    def __repr__(self):
        """torchvision's, **including the part that looks like a mistake.**

        The transforms' repr is added to the body as a single element and the indent is
        applied per element, so only its first line is indented and `Transform:` comes
        out at column zero under an indented `StandardTransform`. It reads as ragged and
        it is what torchvision prints; matching it is the point, since what a reader
        compares against is the output in front of them, not the tidier one.
        """
        body = [f"Number of datapoints: {len(self)}"]
        if self.root is not None:
            body.append(f"Root location: {self.root}")
        body += self.extra_repr().splitlines()
        if self.transforms is not None:
            body.append(repr(self.transforms))
        return "\n".join([f"Dataset {type(self).__name__}"]
                          + [" " * 4 + line for line in body])


class _StandardTransform:
    """The pair, applied. torchvision names this `StandardTransform` and does not
    export it; it is here because the repr above prints it and a reader who sees the
    name should be able to find what makes it."""

    def __init__(self, transform=None, target_transform=None):
        self.transform, self.target_transform = transform, target_transform

    def __call__(self, picture, target):
        if self.transform is not None:
            picture = self.transform(picture)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return picture, target

    def _lines(self, head, held):
        """**The continuation lines are padded to the length of the label**, so a
        `Compose` printed after `Transform: ` lines up under its own opening bracket
        rather than under the label."""
        body = repr(held).splitlines()
        return [f"{head}{body[0]}"] + [" " * len(head) + line for line in body[1:]]

    def __repr__(self):
        lines = ["StandardTransform"]
        if self.transform is not None:
            lines += self._lines("Transform: ", self.transform)
        if self.target_transform is not None:
            lines += self._lines("Target transform: ", self.target_transform)
        return "\n".join(lines)


class MNIST(VisionDataset):
    """Handwritten digits. **`__getitem__` gives an `(H,W)` array, not a PIL image.**

    That is the one place this diverges, and it is the same divergence `ToTensor` has
    everywhere else in this file: torchvision hands the transform a PIL image because
    every one of its transforms accepts one, and there is no PIL here. A recipe reads
    the same either way — `transform=Compose([ToTensor(), Normalize(...)])` — because
    `ToTensor` is where the two conventions meet.

    `.data` is `(N,28,28)` uint8 and `.targets` is `(N,)` int64, both arrays. In
    torchvision they are tensors of the same shape and dtype; arrays are what the rest
    of this file speaks, and `borch.tensor(ds.data)` is the bridge.
    """

    mirrors = ["https://ossci-datasets.s3.amazonaws.com/mnist/",
               "http://yann.lecun.com/exdb/mnist/"]
    resources = [("train-images-idx3-ubyte.gz", "f68b3c2dcbeaaa9fbdd348bbdeb94873"),
                 ("train-labels-idx1-ubyte.gz", "d53e105ee54ea40749a09fcbcd1e9432"),
                 ("t10k-images-idx3-ubyte.gz", "9fb629c4189551a2d022fa330f9573f3"),
                 ("t10k-labels-idx1-ubyte.gz", "ec29112dd5afa0611ce80d1b7f02629c")]
    classes = ["0 - zero", "1 - one", "2 - two", "3 - three", "4 - four",
               "5 - five", "6 - six", "7 - seven", "8 - eight", "9 - nine"]

    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.train = train
        if download:
            self.download()
        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to "
                               "download it")
        self.data, self.targets = self._load_data()

    @property
    def raw_folder(self):
        return _os.path.join(self.root, type(self).__name__, "raw")

    @property
    def class_to_idx(self):
        return {name: i for i, name in enumerate(self.classes)}

    def _files(self):
        return [_os.path.splitext(name)[0] for name, _ in self.resources]

    def _check_exists(self):
        return all(_os.path.isfile(_os.path.join(self.raw_folder, name))
                   for name in self._files())

    def download(self):
        """**Every mirror is tried before the file is given up on**, and the digest
        decides, not the response code. A mirror that answers 200 with an error page
        is the case that a status check passes and a checksum does not."""
        if self._check_exists():
            return
        _os.makedirs(self.raw_folder, exist_ok=True)
        for name, digest in self.resources:
            out = _os.path.join(self.raw_folder, _os.path.splitext(name)[0])
            if _os.path.isfile(out):
                continue
            errors = []
            for mirror in self.mirrors:
                try:
                    # **The digest names the compressed file**, so it is checked
                    # against what came off the wire and not against the gzip of
                    # what was decompressed — gzip is not reproducible, the header
                    # carries a timestamp, and recompressing gives another digest.
                    raw = _gzip.decompress(_fetch(mirror + name, digest))
                except Exception as exc:                          # noqa: BLE001
                    errors.append(f"{mirror}{name}: {type(exc).__name__}: {exc}")
                    continue
                with open(out, "wb") as handle:
                    handle.write(raw)
                break
            else:
                raise RuntimeError(
                    f"could not fetch {name} from any mirror:\n  "
                    + "\n  ".join(errors))

    def _load_data(self):
        which = "train" if self.train else "t10k"
        folder = self.raw_folder
        with open(_os.path.join(folder, f"{which}-images-idx3-ubyte"), "rb") as handle:
            data = _read_idx_images(handle.read())
        with open(_os.path.join(folder, f"{which}-labels-idx1-ubyte"), "rb") as handle:
            targets = _read_idx_labels(handle.read())
        return data, targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        picture, target = self.data[index], int(self.targets[index])
        if self.transforms is not None:
            picture, target = self.transforms(picture, target)
        return picture, target

    def extra_repr(self):
        return f"Split: {'Train' if self.train else 'Test'}"


class FashionMNIST(MNIST):
    """Clothes, in MNIST's shape and MNIST's format. **Only the addresses and the
    class names differ**, which is why it is four lines and not another decoder."""

    mirrors = ["http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"]
    resources = [("train-images-idx3-ubyte.gz", "8d4fb7e6c68d591d4c3dfef9ec88bf0d"),
                 ("train-labels-idx1-ubyte.gz", "25c81989df183df01b3e8a0aad5dffbe"),
                 ("t10k-images-idx3-ubyte.gz", "bef4ecab320f06d8554ea6380940ec79"),
                 ("t10k-labels-idx1-ubyte.gz", "bb300cfdad3c16e7a12a480ee83cd310")]
    classes = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


class KMNIST(MNIST):
    """Kuzushiji — ten classical Japanese characters, again in MNIST's format."""

    mirrors = ["http://codh.rois.ac.jp/kmnist/dataset/kmnist/"]
    resources = [("train-images-idx3-ubyte.gz", "bdb82020997e1d708af4cf47b453dcf7"),
                 ("train-labels-idx1-ubyte.gz", "e144d726b3acfaa3e44228e80efcd344"),
                 ("t10k-images-idx3-ubyte.gz", "5c965bf0a639b31b8f53240b1b52f4d7"),
                 ("t10k-labels-idx1-ubyte.gz", "7320c461ea6c1c855c0b718fb2a4b134")]
    classes = ["o", "ki", "su", "tsu", "na", "ha", "ma", "ya", "re", "wo"]


class CIFAR10(VisionDataset):
    """Sixty thousand 32×32 colour pictures in ten classes.

    Two things differ from MNIST beyond the format. The whole set arrives as **one
    tar**, so a partial download is all-or-nothing rather than four files deep; and
    `.targets` is a plain Python list rather than an array, which is torchvision's
    choice and is kept — `ds.targets[i]` is an `int` on both sides, and code that
    indexes it with an array is code that would have broken there too.
    """

    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = "c58f30108f718f92721af3b95e74349a"
    base_folder = "cifar-10-batches-py"
    train_list = [["data_batch_1", "c99cafc152244af753f735de768cd75f"],
                  ["data_batch_2", "d4bba439e000b95fd0a9bffe97cbabec"],
                  ["data_batch_3", "54ebc095f3ab1f0389bbae665268c751"],
                  ["data_batch_4", "634d18415352ddfa80567beed471001a"],
                  ["data_batch_5", "482c414d41f54cd18b22e5b47cb7c3cb"]]
    test_list = [["test_batch", "40351d587109b95175f43aff81a1287e"]]
    meta = {"filename": "batches.meta", "key": "label_names",
            "md5": "5ff9c542aee3614f3951f8cda6e48888"}

    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.train = train
        if download:
            self.download()
        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. You can use "
                               "download=True to download it")
        pieces = self.train_list if train else self.test_list
        blocks, targets = [], []
        for name, _digest in pieces:
            images, labels = _read_cifar_batch(self._read(name))
            blocks.append(images)
            targets += labels
        self.data = _np.concatenate(blocks) if len(blocks) > 1 else blocks[0]
        self.targets = targets
        self._load_meta()

    def _folder(self):
        return _os.path.join(self.root, self.base_folder)

    def _read(self, name):
        with open(_os.path.join(self._folder(), name), "rb") as handle:
            return handle.read()

    def _load_meta(self):
        """The class names live in their own pickle beside the batches. **A dataset
        whose labels are integers with no names is one where a confusion matrix cannot
        be read**, so this is loaded rather than hard-coded here."""
        entry = _CifarUnpickler(_io.BytesIO(self._read(self.meta["filename"])),
                                encoding="latin1").load()
        self.classes = [name.decode() if isinstance(name, bytes) else name
                        for name in entry[self.meta["key"]]]

    @property
    def class_to_idx(self):
        return {name: i for i, name in enumerate(self.classes)}

    def _pieces(self):
        return list(self.train_list) + list(self.test_list) + [
            [self.meta["filename"], self.meta["md5"]]]

    def _check_integrity(self):
        """**Existence is not integrity, and the difference decides whether a bad
        batch can be healed.**

        Checking only that the files are there was the first version, with the md5
        done later at read time. Every corrupt batch was still caught — but caught in
        the wrong place: `download=True` saw files, skipped the download, and the read
        raised. The one thing that fixes a truncated batch could not fix it, and the
        message asked the reader to delete files by hand.

        So the digests are checked here, which is torchvision's arrangement. It costs
        one pass over 180MB when the dataset is opened; the alternative costs somebody
        an afternoon once.
        """
        for name, digest in self._pieces():
            path = _os.path.join(self._folder(), name)
            if not _os.path.isfile(path):
                return False
            if _md5_of_file(path) != digest:
                return False
        return True

    def download(self):
        """Fetch the tar and unpack it. **Only the members this dataset names are
        written** — a tar can carry a path that climbs out of the directory it is
        unpacked into, and `extractall` follows it."""
        if self._check_integrity():
            return
        wanted = {name for name, _ in self._pieces()}
        _os.makedirs(self._folder(), exist_ok=True)
        archive = _os.path.join(self.root, self.filename)
        # **The kept archive is verified, not trusted by name.** Measured, not
        # imagined: a download of this file was interrupted, the truncated tar stayed
        # at the right name, and the next run trusted it because it existed. What came
        # out was `EOFError: Compressed file ended before the end-of-stream marker`
        # from inside gzip — a sentence about a stream, naming no file, offering no
        # move. Hashing 170MB costs a fraction of a second and the alternative costs
        # somebody the afternoon of working out which file to delete.
        if _os.path.isfile(archive) and _md5_of_file(archive) != self.tgz_md5:
            _os.unlink(archive)
        if not _os.path.isfile(archive):
            _fetch_to(self.url, archive, self.tgz_md5)
        with _tarfile.open(archive, mode="r:gz") as tar:
            for member in tar:
                base = _os.path.basename(member.name)
                if not member.isfile() or base not in wanted:
                    continue
                extracted = tar.extractfile(member)
                if extracted is None:
                    continue
                with open(_os.path.join(self._folder(), base), "wb") as handle:
                    while True:
                        block = extracted.read(1 << 20)
                        if not block:
                            break
                        handle.write(block)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        picture, target = self.data[index], self.targets[index]
        if self.transforms is not None:
            picture, target = self.transforms(picture, target)
        return picture, target

    def extra_repr(self):
        return f"Split: {'Train' if self.train else 'Test'}"


class CIFAR100(CIFAR10):
    """A hundred classes over the same pictures, in the same format. The batches are
    two rather than six and the label key is `fine_labels` — the coarse twenty are in
    the same file under another key, and torchvision does not read them either."""

    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = "eb9058c3a382ffc7106e4002c42a8d85"
    base_folder = "cifar-100-python"
    train_list = [["train", "16019d7e3df5f24257cddd939b257f8d"]]
    test_list = [["test", "f0ef6b0ae62326f3e7ffdfab6717acfc"]]
    meta = {"filename": "meta", "key": "fine_label_names",
            "md5": "7973b15100ade9c7d40fb424638fde48"}


class FakeData(VisionDataset):
    """Random pictures with random labels. **No download and no format** — it exists
    so that a pipeline can be run before the data arrives, which is the one moment
    when a dataset that needs nothing is worth more than a real one.

    Same `image_size` convention as torchvision: `(C,H,W)`. What comes out is `(H,W,C)`
    like every other picture here, so a `(3,32,32)` request gives a `(32,32,3)` array —
    the argument names the picture torchvision would build and the result is in this
    file's layout.
    """

    def __init__(self, size=1000, image_size=(3, 224, 224), num_classes=10,
                 transform=None, target_transform=None, random_offset=0):
        super().__init__(None, transform=transform, target_transform=target_transform)
        self.size, self.num_classes = size, num_classes
        self.image_size = tuple(image_size)
        self.random_offset = random_offset

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        """**Seeded per index**, so the same index gives the same picture within a run
        and across runs. torchvision does this too, and it is what makes a fake dataset
        usable for a reproducibility check rather than only for a smoke test."""
        if index >= len(self):
            raise IndexError(f"{index} is out of range for {len(self)} items")
        rng = _np.random.default_rng(index + self.random_offset)
        channels, height, width = self.image_size
        picture = rng.random((height, width, channels), dtype=_np.float32)
        target = int(rng.integers(self.num_classes))
        if self.transforms is not None:
            picture, target = self.transforms(picture, target)
        return picture, target


# --------------------------------------------------- transforms.v2.functional
#
# **165 public names, and 114 of them are one operation counted five times.**
# `affine_image`, `affine_mask`, `affine_bounding_boxes`, `affine_keypoints` and
# `affine_video` are v2's dispatch kernels: the type decides which runs, and the
# type system that decides is the half of v2 declined here. What is left is 51
# names, 34 of which v1 already has under `transforms.functional`.
#
# So most of this file is a re-export, and the nine below are what v2 adds that can
# be built without tv_tensors. They are thin on purpose — a second body for `hflip`
# under the name `horizontal_flip` is two implementations of one thing, and the one
# that is not being looked at is the one that drifts.


def horizontal_flip(img):
    """v2's spelling of `hflip`. **The same function, not a copy of it.**"""
    return hflip(img)


def vertical_flip(img):
    """v2's spelling of `vflip`."""
    return vflip(img)


def elastic(img, displacement, interpolation="bilinear", fill=None):
    """v2's spelling of `elastic_transform`. **Measured, the two agree exactly** — max
    difference 0.0 on a random picture — so this forwards rather than repeating it.

    `fill=None` and not `0`. Writing the zero looked harmless and is not: `None` leaves
    the outside of the warp untouched and `0` paints it black, and on a picture whose
    edges barely move the difference is a thin dark rim that reads as the warp working.
    Caught by comparing, which is the only way that one gets caught.
    """
    return elastic_transform(img, displacement, interpolation, fill)


def get_size(img):
    """`[height, width]` — **a list, and in that order.**

    `get_image_size` in v1 answers `[width, height]`. Reversing the pair was one of
    v2's deliberate corrections, and the two names sit one namespace apart giving
    opposite answers, so this is the place to say which is which rather than the place
    to be brief.
    """
    height, width = get_dimensions(img)[1:]
    return [height, width]


def get_num_channels(img):
    """v2's spelling of `get_image_num_channels`."""
    return get_image_num_channels(img)


def grayscale_to_rgb(img):
    """One channel to three. **Three channels pass through untouched** — measured
    against torchvision, which returns the input rather than raising, so a pipeline
    can carry mixed pictures without a branch."""
    arr = _np.asarray(img)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    if arr.shape[2] == 1:
        return _np.ascontiguousarray(_np.repeat(arr, 3, axis=2))
    return arr


def permute_channels(img, permutation):
    """Reorder the channels. The list is **positions to take from**, so `[2, 0, 1]`
    puts the old third channel first — the same direction as indexing, and the
    opposite of "where each channel goes"."""
    arr = _np.asarray(img)
    arr = arr if arr.ndim == 3 else arr[:, :, None]
    if sorted(permutation) != list(range(arr.shape[2])):
        raise ValueError(
            f"Invalid permutation {list(permutation)} for {arr.shape[2]} channels\n"
            "  (torch: Invalid permutation)")
    return _np.ascontiguousarray(arr[:, :, list(permutation)])


def to_dtype(img, dtype=None, scale=False):
    """Cast, and **optionally scale on the way** — the half of `ToTensor` that v2
    split out, as a function. `scale=False` is the default here as it is there, which
    is the trap: `to_dtype(bytes, float32)` gives 0..255 floats and looks like it
    worked."""
    return v2.ToDtype(dtype, scale)(img)


def gaussian_noise(img, mean=0.0, sigma=0.1, clip=True):
    """Add normal noise. Float pictures only, for the reason `GaussianNoise` gives."""
    return v2.GaussianNoise(mean, sigma, clip)(img)


class QMNIST(MNIST):
    """MNIST's pictures **with the rest of what NIST recorded about them.**

    Four things differ from `MNIST` and only one is the format. The labels are
    `idx2-int` rather than `idx1-ubyte` — a table of **eight int32 columns per
    picture**, of which the first is the digit and the others are who wrote it, which
    NIST partition it came from, and where in that partition. `compat=True`, the
    default, hands back the digit alone so that a recipe written for MNIST runs
    unchanged; `compat=False` hands back the row.

    `what` picks the subset rather than `train` doing it. **The test set is 60,000
    pictures**, not 10,000: MNIST's familiar test set is the first 10,000 of it, which
    is what `test10k` means, and `test50k` is the remainder — 50,000 pictures that were
    never in MNIST at all and are measurably harder. A count of 60,000 where 10,000 was
    expected is the first sign somebody has swapped one for the other.
    """

    subsets = {"train": "train", "test": "test", "test10k": "test",
               "test50k": "test", "nist": "nist"}
    resources = {
        "train": [
            ("https://raw.githubusercontent.com/facebookresearch/qmnist/master/"
             "qmnist-train-images-idx3-ubyte.gz", "ed72d4157d28c017586c42bc6afe6370"),
            ("https://raw.githubusercontent.com/facebookresearch/qmnist/master/"
             "qmnist-train-labels-idx2-int.gz", "0058f8dd561b90ffdd0f734c6a30e5e4")],
        "test": [
            ("https://raw.githubusercontent.com/facebookresearch/qmnist/master/"
             "qmnist-test-images-idx3-ubyte.gz", "1394631089c404de565df7b7aeaf9412"),
            ("https://raw.githubusercontent.com/facebookresearch/qmnist/master/"
             "qmnist-test-labels-idx2-int.gz", "5b5b05890a5e13444e108efe57b788aa")],
        "nist": [
            ("https://raw.githubusercontent.com/facebookresearch/qmnist/master/"
             "xnist-images-idx3-ubyte.xz", "7f124b3b8ab81486c9d8c2749c17f834"),
            ("https://raw.githubusercontent.com/facebookresearch/qmnist/master/"
             "xnist-labels-idx2-int.xz", "5ed0e788978e45d4a8bd4b7caec3d79d")],
    }

    def __init__(self, root, what=None, compat=True, train=True, **kwargs):
        if what is None:
            what = "train" if train else "test"
        if what not in self.subsets:
            raise ValueError(f"what should be one of {sorted(self.subsets)}, got {what}")
        self.what, self.compat = what, compat
        super().__init__(root, train=(what == "train"), **kwargs)

    def _files(self):
        return [_os.path.splitext(name.rsplit("/", 1)[-1])[0]
                for name, _ in self.resources[self.subsets[self.what]]]

    def download(self):
        """**The `.xz` one is declined rather than half-supported.** `nist` is the raw
        402,953-picture partition and it ships LZMA-compressed; nothing else here needs
        that decompressor and a dataset that downloads but cannot be opened is worse
        than one that says so."""
        if self._check_exists():
            return
        if self.subsets[self.what] == "nist":
            raise RuntimeError(
                "QMNIST's `nist` subset ships as .xz and is not read here.\n"
                "  `train`, `test`, `test10k` and `test50k` are gzip and do work.")
        _os.makedirs(self.raw_folder, exist_ok=True)
        for url, digest in self.resources[self.subsets[self.what]]:
            name = _os.path.splitext(url.rsplit("/", 1)[-1])[0]
            out = _os.path.join(self.raw_folder, name)
            if _os.path.isfile(out):
                continue
            with open(out, "wb") as handle:
                handle.write(_gzip.decompress(_fetch(url, digest)))

    def _load_data(self):
        images, labels = self._files()
        with open(_os.path.join(self.raw_folder, images), "rb") as handle:
            data = _read_idx_images(handle.read())
        with open(_os.path.join(self.raw_folder, labels), "rb") as handle:
            # **Not `_read_idx_labels`** — that one insists on one byte and one axis,
            # and this is a two-axis table of int32. The reader underneath takes both.
            targets = _read_idx(handle.read()).astype(_np.int64)
        if self.what == "test10k":
            data, targets = data[:10000], targets[:10000]
        elif self.what == "test50k":
            data, targets = data[10000:], targets[10000:]
        return data, targets

    def __getitem__(self, index):
        picture, target = self.data[index], self.targets[index]
        if self.compat:
            target = int(target[0])
        if self.transforms is not None:
            picture, target = self.transforms(picture, target)
        return picture, target

    def extra_repr(self):
        return f"Split: {self.what}"


class SEMEION(VisionDataset):
    """1,593 handwritten digits at 16×16, **as a text file of ones and zeros.**

    No compression, no container, one line a picture: 256 pixel values then ten
    one-hot columns. The pixels are already 0 or 1, so the only conversion is a scale
    to 0..255 — which is why this dataset is here at all while forty others are not.

    `.labels` rather than `.targets`, which is torchvision's name for it and is worth
    keeping even though it is the odd one out. A recipe that reads `.targets` should
    fail loudly here rather than find an attribute that happens to exist.
    """

    url = "http://archive.ics.uci.edu/ml/machine-learning-databases/semeion/semeion.data"
    filename = "semeion.data"
    md5_checksum = "cb545d371d2ce14ec121470795a77432"

    def __init__(self, root, transform=None, target_transform=None, download=False):
        super().__init__(root, transform=transform, target_transform=target_transform)
        path = _os.path.join(self.root, self.filename)
        if download and not _os.path.isfile(path):
            _os.makedirs(self.root, exist_ok=True)
            with open(path, "wb") as handle:
                handle.write(_fetch(self.url, self.md5_checksum))
        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. You can use "
                               "download=True to download it")
        rows = _np.loadtxt(path)
        self.data = (rows[:, :256] * 255).astype(_np.uint8).reshape(-1, 16, 16)
        # **The label is which one-hot column is set**, not the value in it. Reading
        # the value gives 1.0 for every picture and a dataset of a single class.
        self.labels = _np.nonzero(rows[:, 256:])[1]

    def _check_integrity(self):
        path = _os.path.join(self.root, self.filename)
        return _os.path.isfile(path) and _md5_of_file(path) == self.md5_checksum

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        picture, target = self.data[index], int(self.labels[index])
        if self.transforms is not None:
            picture, target = self.transforms(picture, target)
        return picture, target


class USPS(VisionDataset):
    """7,291 training and 2,007 test digits at 16×16, **in LIBSVM text inside bzip2.**

    Two conversions, and both are the kind that pass unnoticed when reversed. The
    pixel values run −1 to 1 and become bytes as `(v + 1) / 2 * 255`, so a version
    that forgot the shift gives a picture that is still a picture. And **the labels on
    disk are 1 to 10**, so every one has 1 subtracted; forgotten, the tenth class
    becomes an index nothing has a name for and the zeros disappear.
    """

    split_list = {
        "train": ("https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/"
                  "usps.bz2", "usps.bz2", "ec16c51db3855ca6c91edd34d0e9b197"),
        "test": ("https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/"
                 "usps.t.bz2", "usps.t.bz2", "8ea070ee2aca1ac39742fdd1ef5ed118"),
    }

    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.train = train
        url, filename, digest = self.split_list["train" if train else "test"]
        path = _os.path.join(self.root, filename)
        if download and not _os.path.isfile(path):
            _os.makedirs(self.root, exist_ok=True)
            with open(path, "wb") as handle:
                handle.write(_fetch(url, digest))
        if not _os.path.isfile(path):
            raise RuntimeError("Dataset not found. You can use download=True to "
                               "download it")
        with _bz2.open(path) as handle:
            rows = [line.decode().split() for line in handle.readlines()]
        pixels = [[cell.split(":")[-1] for cell in row[1:]] for row in rows]
        picture = _np.asarray(pixels, dtype=_np.float32).reshape(-1, 16, 16)
        self.data = ((picture + 1) / 2 * 255).astype(_np.uint8)
        self.targets = [int(row[0]) - 1 for row in rows]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        picture, target = self.data[index], self.targets[index]
        if self.transforms is not None:
            picture, target = self.transforms(picture, target)
        return picture, target

    def extra_repr(self):
        return f"Split: {'Train' if self.train else 'Test'}"


class SVHN(VisionDataset):
    """Street View House Numbers — 32×32 colour digits, **in a MATLAB file.**

    **This was declined, and the reason confused two different walls.** It read: *the
    refusal is the dependency, and it is the same answer PIL and a JPEG decoder get.*
    A JPEG decoder is a codec — thousands of lines and no reasonable way to write one
    here. A `.mat` is a **documented container**, and the reader for it (`_mat_read`
    above) is under a hundred lines of `struct` and `zlib`, both already in the
    standard library. Grouping them under one sentence made the cheap one look as
    expensive as the dear one.

    Two conversions that pass unnoticed when reversed:

    - **The digit zero is labelled 10**, so every 10 becomes 0. Left alone, a loss
      expecting classes `[0, C-1]` gets an index one past the end — and `CrossEntropy`
      with `ignore_index` unset simply reads out of range rather than complaining.
    - **The array is `(32, 32, 3, N)` on disk**, MATLAB's own order, and becomes
      `(N, 3, 32, 32)`. Transposed the wrong way it is still a stack of colour
      pictures of the right size, of something else.

    `split` is `train`, `test` or `extra`; torchvision's three, with torchvision's
    digests.
    """

    split_list = {
        "train": ("http://ufldl.stanford.edu/housenumbers/train_32x32.mat",
                  "train_32x32.mat", "e26dedcc434d2e4c54c9b2d4a06d8373"),
        "test": ("http://ufldl.stanford.edu/housenumbers/test_32x32.mat",
                 "test_32x32.mat", "eb5a983be6a315427106f1b164d9cef3"),
        "extra": ("http://ufldl.stanford.edu/housenumbers/extra_32x32.mat",
                  "extra_32x32.mat", "a93ce644f1a588dc4d68dda5feec44a7"),
    }

    def __init__(self, root, split="train", transform=None, target_transform=None,
                 download=False):
        super().__init__(root, transform=transform, target_transform=target_transform)
        if split not in self.split_list:
            raise ValueError(
                f"Unknown value '{split}' for argument split. "
                f"Valid values are {tuple(self.split_list)}.")
        self.split = split
        url, filename, digest = self.split_list[split]
        path = _os.path.join(self.root, filename)
        if download and not (_os.path.isfile(path) and _md5_of_file(path) == digest):
            _os.makedirs(self.root, exist_ok=True)
            _fetch_to(url, path, digest)
        if not _os.path.isfile(path):
            raise RuntimeError("Dataset not found or corrupted. You can use "
                               "download=True to download it")
        with open(path, "rb") as handle:
            got = _mat_read(handle.read())
        # **The `y` half is the one a reader can lose.** `scipy` writes each variable
        # as its own compressed element, and a version of `_mat_read` that stopped
        # after the first found `X` alone — pictures, no labels, and a file that
        # parsed. Named here rather than assumed.
        for key in ("X", "y"):
            if key not in got:
                raise RuntimeError(
                    f"{filename} has no `{key}` — it holds {sorted(got)}")
        labels = _np.asarray(got["y"]).astype(_np.int64).reshape(-1)
        labels[labels == 10] = 0
        self.labels = labels
        self.data = _np.ascontiguousarray(
            _np.transpose(_np.asarray(got["X"]), (3, 2, 0, 1)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        picture = _np.transpose(self.data[index], (1, 2, 0))
        target = int(self.labels[index])
        if self.transforms is not None:
            picture, target = self.transforms(picture, target)
        return picture, target

    def extra_repr(self):
        return f"Split: {self.split}"


class DatasetFolder(VisionDataset):
    """A folder per class, and **a `loader` you hand it.**

    This is the one dataset here that reads no format at all: it walks directories,
    sorts what it finds, and calls your function on each path. `ImageFolder` is this
    with the loader defaulted to PIL, which is why that one is absent and this one is
    not — the codec was never in this class, only in that default.

    The classes are **the sorted names of the subdirectories**, and the sort is the
    part worth stating: the index a class gets depends on its name and on nothing
    else, so renaming a folder renumbers the labels of a trained model.
    """

    def __init__(self, root, loader, extensions=None, transform=None,
                 target_transform=None, is_valid_file=None):
        super().__init__(root, transform=transform, target_transform=target_transform)
        if (extensions is None) == (is_valid_file is None):
            raise ValueError("Both extensions and is_valid_file cannot be None or "
                             "not None at the same time")
        self.loader = loader
        self.extensions = extensions
        keep = (is_valid_file if is_valid_file is not None
                else (lambda path: path.lower().endswith(tuple(extensions))))
        self.classes, self.class_to_idx = self.find_classes(root)
        self.samples = []
        for name in self.classes:
            folder = _os.path.join(root, name)
            for base, _dirs, files in sorted(_os.walk(folder, followlinks=True)):
                for leaf in sorted(files):
                    path = _os.path.join(base, leaf)
                    if keep(path):
                        self.samples.append((path, self.class_to_idx[name]))
        if not self.samples:
            raise FileNotFoundError(
                f"Found no valid file for the classes {', '.join(self.classes)}. "
                f"Supported extensions are: "
                f"{','.join(extensions) if extensions else 'the given is_valid_file'}")
        self.targets = [index for _path, index in self.samples]

    def find_classes(self, directory):
        classes = sorted(entry.name for entry in _os.scandir(directory)
                         if entry.is_dir())
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")
        return classes, {name: i for i, name in enumerate(classes)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transforms is not None:
            sample, target = self.transforms(sample, target)
        return sample, target


class EMNIST(MNIST):
    """The whole NIST Special Database 19 in MNIST's format — **letters as well as
    digits, and six ways of carving it up.**

    `split` is required and picks both the pictures and the class list. `byclass` is
    all 62 characters as written; `bymerge` and `balanced` fold the fifteen letters
    whose upper and lower cases are indistinguishable by hand — `c`, `i`, `j`, `k`,
    `l`, `m`, `o`, `p`, `s`, `u`, `v`, `w`, `x`, `y`, `z` — into 47; `letters` is the
    26 lowercase with **a placeholder at index 0**, because its labels are 1-based on
    disk and nothing sits at zero; `digits` and `mnist` are the ten digits at different
    sizes.

    **The pictures arrive transposed** and torchvision leaves them that way, so this
    does too. A picture that reads as a mirrored, rotated character is not a bug here
    and not one there; it is what the files contain, and `transforms` has `hflip` and
    `rotate` for anyone who wants them upright. Correcting it silently would make this
    library and torchvision disagree on every EMNIST pixel.

    One archive holds all six splits — 536MB of zip containing gzipped IDX — so the
    first `download=True` fetches everything regardless of which split was asked for,
    and the ones after that fetch nothing.
    """

    url = "https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip"
    md5 = "58c8d27c78d21e728a6bc7b3cc06412e"
    splits = ("byclass", "bymerge", "balanced", "letters", "digits", "mnist")
    # The fifteen whose cases are not told apart by hand. Written out rather than
    # derived: there is no rule that produces this set, it is a judgement NIST made.
    _merged_classes = {"c", "i", "j", "k", "l", "m", "o", "p", "s", "u", "v", "w",
                       "x", "y", "z"}
    _all_classes = set(_string.digits + _string.ascii_letters)
    classes_split_dict = {
        "byclass": sorted(_all_classes),
        "bymerge": sorted(_all_classes - _merged_classes),
        "balanced": sorted(_all_classes - _merged_classes),
        # **Index 0 is a placeholder and not a class.** The labels run 1..26, so a
        # reader that drops this entry has every letter off by one and a confusion
        # matrix that looks like a systematically confused model.
        "letters": ["N/A"] + list(_string.ascii_lowercase),
        "digits": list(_string.digits),
        "mnist": list(_string.digits),
    }

    def __init__(self, root, split, **kwargs):
        if split not in self.splits:
            raise ValueError(f"Unknown value '{split}' for argument split. "
                             f"Valid values are {{{', '.join(self.splits)}}}.")
        self.split = split
        super().__init__(root, **kwargs)
        self.classes = self.classes_split_dict[split]

    def _files(self):
        prefix = f"emnist-{self.split}-{'train' if self.train else 'test'}"
        return [f"{prefix}-images-idx3-ubyte", f"{prefix}-labels-idx1-ubyte"]

    def _load_data(self):
        images, labels = self._files()
        with open(_os.path.join(self.raw_folder, images), "rb") as handle:
            data = _read_idx_images(handle.read())
        with open(_os.path.join(self.raw_folder, labels), "rb") as handle:
            targets = _read_idx_labels(handle.read())
        return data, targets

    def download(self):
        """**The zip is unpacked one member at a time and only where it belongs.**

        `zipfile.extractall` follows a member's own path, and a member's path can climb
        out of the directory it is unpacked into. Nothing in this archive does; the
        guard costs three lines and does not depend on that staying true.
        """
        if self._check_exists():
            return
        _os.makedirs(self.raw_folder, exist_ok=True)
        archive = _os.path.join(self.raw_folder, "gzip.zip")
        if _os.path.isfile(archive) and _md5_of_file(archive) != self.md5:
            _os.unlink(archive)
        if not _os.path.isfile(archive):
            _fetch_to(self.url, archive, self.md5)
        with _zipfile.ZipFile(archive) as zipped:
            for member in zipped.namelist():
                base = _os.path.basename(member)
                if not base.endswith(".gz"):
                    continue
                out = _os.path.join(self.raw_folder, base[:-len(".gz")])
                if _os.path.isfile(out):
                    continue
                with zipped.open(member) as source:
                    with open(out, "wb") as handle:
                        handle.write(_gzip.decompress(source.read()))

    def extra_repr(self):
        return f"Split: {self.split}"


class FER2013(VisionDataset):
    """35,887 faces at 48×48, **as integers in a CSV cell.**

    Each row's `pixels` column is 2,304 numbers separated by spaces. No codec, no
    archive, no network — which is the whole reason this one is here while forty
    others are not, and the reason it stayed off the list for a day longer than it
    should have.

    **torchvision cannot download it**, and that was written here as "there is
    nothing to compare an implementation against". That was wrong, and wrong in a
    way this project keeps catching: *cannot fetch the data* was carried over into
    *cannot check the code*. torchvision's reader takes a directory. Point both at
    the same file and the comparison is exactly as real as every other one here.

    Two layouts, and `Usage` is what tells them apart. The Kaggle file — `fer2013.csv`
    or `icml_face_data.csv` — holds every row with a `Usage` column saying which
    split it belongs to; the per-split `train.csv` and `test.csv` do not, and
    `test.csv` has **no `emotion` column at all**, so its labels are `None`.
    """

    _RESOURCES = {
        "train": ("train.csv", "3f0dfb3d3fd99c811a1299cb947e3131"),
        "test": ("test.csv", "b02c2298636a634e8c2faabbf3ea9a23"),
        "fer": ("fer2013.csv", "f8428a1edbd21e88f42c73edd2a14f95"),
        "icml": ("icml_face_data.csv", "b114b9e04e6949e5fe8b6a98b3892b1d"),
    }
    classes = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

    def __init__(self, root, split="train", transform=None, target_transform=None):
        super().__init__(root, transform=transform, target_transform=target_transform)
        if split not in ("train", "test"):
            raise ValueError(f"Unknown value '{split}' for argument split. "
                             "Valid values are {train, test}.")
        self._split = split
        folder = _os.path.join(self.root, "fer2013")
        combined = next((key for key in ("fer", "icml")
                         if _os.path.isfile(_os.path.join(folder,
                                                          self._RESOURCES[key][0]))), None)
        name, digest = self._RESOURCES[combined or split]
        path = _os.path.join(folder, name)
        if not (_os.path.isfile(path) and _md5_of_file(path) == digest):
            raise RuntimeError(
                f"{name} not found in {folder} or corrupted. You can download it from "
                "https://www.kaggle.com/c/challenges-in-representation-learning-"
                "facial-expression-recognition-challenge")
        # **The ICML file's headers carry a leading space** and the others' do not.
        # It is the kind of difference that a reader written against one file finds
        # by raising `KeyError: 'pixels'` on the other, which reads as a corrupt file.
        pixels_key = " pixels" if combined == "icml" else "pixels"
        usage_key = " Usage" if combined == "icml" else "Usage"
        keep = (("Training",) if split == "train"
                else ("PublicTest", "PrivateTest"))
        rows = []
        with open(path, newline="") as handle:
            for row in _csv.DictReader(handle):
                if combined and row[usage_key] not in keep:
                    continue
                picture = _np.asarray([int(v) for v in row[pixels_key].split()],
                                      dtype=_np.uint8).reshape(48, 48)
                label = (int(row["emotion"])
                         if combined or split == "train" else None)
                rows.append((picture, label))
        self._samples = rows

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, index):
        picture, target = self._samples[index]
        if self.transforms is not None:
            picture, target = self.transforms(picture, target)
        return picture, target

    def extra_repr(self):
        return f"split={self._split}"


class MovingMNIST(VisionDataset):
    """Ten thousand clips of two digits drifting across a 64×64 frame.

    **`__getitem__` gives one video and no label** — the only dataset here that
    returns a bare array rather than a pair. It is a next-frame-prediction set: the
    thing to predict is the clip's own later frames, so there is nothing else to
    hand back.

    `split` cuts **the frames, not the clips.** `split_ratio=10` means the first ten
    frames of every clip are "train" and the last ten are "test", and both halves
    still have ten thousand clips in them. Read as a clip split — which is what
    every other dataset here means by `split` — a training run silently learns and
    scores on the same ten thousand videos. `split=None`, the default, gives all
    twenty frames.
    """

    _URL = ("https://www.cs.toronto.edu/~nitish/unsupervised_video/"
            "mnist_test_seq.npy")

    def __init__(self, root, split=None, split_ratio=10, download=False,
                 transform=None):
        super().__init__(root, transform=transform)
        if split is not None and split not in ("train", "test"):
            raise ValueError(f"Unknown value '{split}' for argument split. "
                             "Valid values are {train, test}.")
        if not isinstance(split_ratio, int):
            raise TypeError("`split_ratio` should be an integer, but got "
                            f"{type(split_ratio)}")
        if not 1 <= split_ratio <= 19:
            raise ValueError("`split_ratio` should be `1 <= split_ratio <= 19`, but "
                             f"got {split_ratio} instead.")
        self.split, self.split_ratio = split, split_ratio
        self._folder = _os.path.join(self.root, type(self).__name__)
        self._filename = self._URL.rsplit("/", 1)[-1]
        path = _os.path.join(self._folder, self._filename)
        if download and not _os.path.isfile(path):
            _os.makedirs(self._folder, exist_ok=True)
            _fetch_to(self._URL, path, None)
        if not _os.path.isfile(path):
            raise RuntimeError("Dataset not found. You can use download=True to "
                               "download it.")
        data = _np.load(path)                       # (frames, clips, 64, 64)
        if split == "train":
            data = data[:split_ratio]
        elif split == "test":
            data = data[split_ratio:]
        # To (clips, frames, 1, 64, 64) — the channel axis is added rather than
        # found, since the file is greyscale and says so by having no such axis.
        self.data = _np.ascontiguousarray(data.transpose(1, 0, 2, 3)[:, :, None])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        video = self.data[index]
        if self.transform is not None:
            video = self.transform(video)
        return video


class STL10(VisionDataset):
    """96×96 colour, ten classes, and **a hundred thousand unlabelled pictures**
    alongside the five thousand labelled ones — which is what the set is for.

    Three things are worth stating because each is a place to be quietly wrong.

    **The pictures are stored column-major.** The file reshapes to `(N,3,96,96)` and
    then axes 2 and 3 are swapped. Skip the swap and every picture is transposed —
    still a picture, still trains, and a model that learns transposed features scores
    plausibly.

    **The labels on disk are 1 to 10** and one is subtracted. Left alone, class 10
    is an index the ten class names have no entry for.

    **`unlabeled` carries `-1` rather than nothing.** A sentinel keeps the pair shape
    that every other dataset here has, so the same loop reads both — and `-1` is
    loud in a loss, where `0` would silently mean *aeroplane*.

    `folds` picks one of ten predefined 1,000-picture subsets from
    `fold_indices.txt`, for the low-data comparisons the set was published for.
    """

    base_folder = "stl10_binary"
    url = "http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz"
    filename = "stl10_binary.tar.gz"
    tgz_md5 = "91f7769df0f17e558f3565bffb0c7dfb"
    class_names_file = "class_names.txt"
    folds_list_file = "fold_indices.txt"
    train_list = [["train_X.bin", "918c2871b30a85fa023e0c44e0bee87f"],
                  ["train_y.bin", "5a34089d4802c674881badbb80307741"],
                  ["unlabeled_X.bin", "5242ba1fed5e4be9e1e742405eb56ca4"]]
    test_list = [["test_X.bin", "7f263ba9f9e0b06b93213547f721ac82"],
                 ["test_y.bin", "36f9794fa4beb8a2c72628de14fa638e"]]
    splits = ("train", "train+unlabeled", "unlabeled", "test")

    def __init__(self, root, split="train", folds=None, transform=None,
                 target_transform=None, download=False):
        super().__init__(root, transform=transform, target_transform=target_transform)
        if split not in self.splits:
            raise ValueError(f"Unknown value '{split}' for argument split. Valid "
                             f"values are {{{', '.join(self.splits)}}}.")
        self.split = split
        self.folds = self._verify_folds(folds)
        if download:
            self.download()
        elif not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. You can use "
                               "download=True to download it")
        if split == "train":
            self.data, self.labels = self._load("train_X.bin", "train_y.bin")
            self._take_fold(folds)
        elif split == "train+unlabeled":
            self.data, self.labels = self._load("train_X.bin", "train_y.bin")
            self._take_fold(folds)
            extra, _ = self._load("unlabeled_X.bin")
            self.data = _np.concatenate((self.data, extra))
            self.labels = _np.concatenate(
                (self.labels, _np.asarray([-1] * extra.shape[0])))
        elif split == "unlabeled":
            self.data, _ = self._load("unlabeled_X.bin")
            self.labels = _np.asarray([-1] * self.data.shape[0])
        else:
            self.data, self.labels = self._load("test_X.bin", "test_y.bin")
        names = _os.path.join(self.root, self.base_folder, self.class_names_file)
        if _os.path.isfile(names):
            with open(names) as handle:
                self.classes = handle.read().splitlines()

    def _verify_folds(self, folds):
        if folds is None or (isinstance(folds, int) and folds in range(10)):
            return folds
        if isinstance(folds, int):
            raise ValueError("Value for argument folds should be in the range "
                             f"[0, 10), but got {folds}.")
        raise ValueError("Expected type None or int for argument folds, but got "
                         f"type {type(folds)}.")

    def _load(self, data_file, labels_file=None):
        labels = None
        if labels_file:
            with open(_os.path.join(self.root, self.base_folder, labels_file),
                      "rb") as handle:
                labels = _np.frombuffer(handle.read(), dtype=_np.uint8) - 1
        with open(_os.path.join(self.root, self.base_folder, data_file),
                  "rb") as handle:
            flat = _np.frombuffer(handle.read(), dtype=_np.uint8)
        images = flat.reshape(-1, 3, 96, 96).transpose(0, 1, 3, 2)
        return _np.ascontiguousarray(images), labels

    def _take_fold(self, folds):
        if folds is None:
            return
        with open(_os.path.join(self.root, self.base_folder,
                                self.folds_list_file)) as handle:
            picked = _np.asarray(handle.read().splitlines()[folds].split(),
                                 dtype=_np.int64)
        self.data = self.data[picked]
        if self.labels is not None:
            self.labels = self.labels[picked]

    def _check_integrity(self):
        for name, digest in self.train_list + self.test_list:
            path = _os.path.join(self.root, self.base_folder, name)
            if not _os.path.isfile(path) or _md5_of_file(path) != digest:
                return False
        return True

    def download(self):
        """As `CIFAR10.download`, and for the same reasons — streamed, hashed on the
        way past, and **only the members this dataset names are written.**"""
        if self._check_integrity():
            return
        folder = _os.path.join(self.root, self.base_folder)
        _os.makedirs(folder, exist_ok=True)
        wanted = {name for name, _ in self.train_list + self.test_list}
        wanted |= {self.class_names_file, self.folds_list_file}
        archive = _os.path.join(self.root, self.filename)
        if _os.path.isfile(archive) and _md5_of_file(archive) != self.tgz_md5:
            _os.unlink(archive)
        if not _os.path.isfile(archive):
            _fetch_to(self.url, archive, self.tgz_md5)
        with _tarfile.open(archive, mode="r:gz") as tar:
            for member in tar:
                base = _os.path.basename(member.name)
                if not member.isfile() or base not in wanted:
                    continue
                extracted = tar.extractfile(member)
                if extracted is None:
                    continue
                with open(_os.path.join(folder, base), "wb") as handle:
                    while True:
                        block = extracted.read(1 << 20)
                        if not block:
                            break
                        handle.write(block)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # **`.data` is `(N,C,H,W)` and what comes out here is `(H,W,C)`.** torchvision
        # transposes at exactly this line too, with the comment "so that it is
        # consistent with all other datasets" — its other datasets hold HWC and this
        # one does not, so the attribute and the item disagree on purpose.
        #
        # Caught by comparing: `data`, `labels`, `len` and `classes` matched on all
        # four splits while `__getitem__` did not, which is the shape that says the
        # store is right and the door is wrong.
        picture = _np.ascontiguousarray(self.data[index].transpose(1, 2, 0))
        target = None if self.labels is None else int(self.labels[index])
        if self.transforms is not None:
            picture, target = self.transforms(picture, target)
        return picture, target

    def extra_repr(self):
        return f"Split: {self.split}"


transforms = _types.ModuleType("borchvision.transforms")
functional = _types.ModuleType("borchvision.transforms.functional")
transforms.functional = functional
_sys.modules["borchvision.transforms"] = transforms
_sys.modules["borchvision.transforms.functional"] = functional

# `ops` is torchvision's **top-level** namespace, not one under `transforms` — so it is
# registered beside it rather than inside it. Getting that wrong would make
# `import borchvision.ops` work and `import torchvision.ops` mean something else.
datasets = _types.ModuleType("borchvision.datasets")
_sys.modules["borchvision.datasets"] = datasets
for _name in ("VisionDataset", "MNIST", "FashionMNIST", "KMNIST", "QMNIST", "EMNIST",
              "CIFAR10", "CIFAR100", "FakeData", "SEMEION", "USPS", "DatasetFolder",
              "FER2013", "MovingMNIST", "STL10", "SVHN"):
    setattr(datasets, _name, globals()[_name])

ops = _types.ModuleType("borchvision.ops")
_sys.modules["borchvision.ops"] = ops

# `transforms.v2` is a namespace inside `transforms`, and registered the same way the
# other two are. **The classes in it are not the classes in `transforms`** — they are
# twins that inherit the behaviour and print v2's surface, which is the whole of what
# separates the two namespaces on a plain image.
v2 = _types.ModuleType("borchvision.transforms.v2")
transforms.v2 = v2
_sys.modules["borchvision.transforms.v2"] = v2

for _name in ("CenterCrop", "Compose", "FiveCrop", "Grayscale",
              "InterpolationMode", "Lambda", "LinearTransformation", "Normalize",
              "Pad", "RandomApply", "RandomChoice", "RandomCrop", "RandomErasing",
              "RandomGrayscale", "RandomHorizontalFlip", "RandomOrder",
              "AugMix", "AutoAugment", "AutoAugmentPolicy", "ColorJitter",
              "ElasticTransform", "GaussianBlur", "RandAugment",
              "RandomAdjustSharpness", "RandomAffine",
              "RandomAutocontrast",
              "RandomEqualize", "RandomInvert", "RandomPosterize",
              "RandomPerspective", "RandomResizedCrop", "RandomRotation",
              "RandomSolarize", "TrivialAugmentWide",
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

for _name in ("batched_nms", "box_area", "box_convert", "box_iou",
              "clip_boxes_to_image", "complete_box_iou", "distance_box_iou",
              "generalized_box_iou", "masks_to_boxes", "nms", "remove_small_boxes"):
    setattr(ops, _name, globals()[_name])

for _base, _fields in _V2_TABLE:
    setattr(v2, _base.__name__, _v2_twin(_base, _fields))

# The three whose printing is torch's module repr rather than v2's field rule, and the
# fourteen v2 adds. Assigned under the names torchvision gives them — the leading `_V2`
# on the class is so that the name in this module still means the v1 one.
for _name, _cls in (("Compose", _V2Compose), ("RandomApply", _V2RandomApply),
                    ("Lambda", _V2Lambda), ("Identity", _V2Identity),
                    ("ElasticTransform", _V2ElasticTransform),
                    ("RGB", _V2RGB), ("ToImage", _V2ToImage), ("ToDtype", _V2ToDtype),
                    ("ToPureTensor", _V2ToPureTensor),
                    ("GaussianNoise", _V2GaussianNoise),
                    ("RandomChannelPermutation", _V2RandomChannelPermutation),
                    ("RandomPhotometricDistort", _V2RandomPhotometricDistort),
                    ("RandomResize", _V2RandomResize),
                    ("RandomShortestSize", _V2RandomShortestSize),
                    ("RandomZoomOut", _V2RandomZoomOut),
                    ("ScaleJitter", _V2ScaleJitter),
                    ("MixUp", _V2MixUp), ("CutMix", _V2CutMix)):
    _cls.__name__ = _name
    _cls.__qualname__ = _name
    setattr(v2, _name, _cls)

# **The enum is shared rather than twinned.** torchvision's v2 re-exports the same
# `InterpolationMode`, and a second one would make `v2.InterpolationMode.BILINEAR` a
# different object from `transforms.InterpolationMode.BILINEAR` — equal by value and
# not by identity, which is the kind of difference that bites once and takes an hour.
v2.InterpolationMode = InterpolationMode
v2.AutoAugmentPolicy = AutoAugmentPolicy

# `transforms.v2.functional`. **34 names are v1's, re-exported rather than rewritten**
# — v2 changed what its transforms print, not what its functions compute, and a second
# body under a second name is the one that drifts because nobody is looking at it.
v2_functional = _types.ModuleType("borchvision.transforms.v2.functional")
v2.functional = v2_functional
_sys.modules["borchvision.transforms.v2.functional"] = v2_functional
for _name in ("adjust_brightness", "adjust_contrast", "adjust_gamma", "adjust_hue",
              "adjust_saturation", "adjust_sharpness", "affine", "autocontrast",
              "center_crop", "crop", "elastic_transform", "equalize", "erase",
              "five_crop", "gaussian_blur", "get_dimensions", "get_image_num_channels",
              "get_image_size", "hflip", "invert", "normalize", "pad", "perspective",
              "posterize", "resize", "resized_crop", "rgb_to_grayscale", "rotate",
              "solarize", "ten_crop", "to_grayscale", "to_tensor", "vflip",
              # And the nine v2 adds that need no tv_tensors.
              "horizontal_flip", "vertical_flip", "elastic", "get_size",
              "get_num_channels", "grayscale_to_rgb", "permute_channels", "to_dtype",
              "gaussian_noise"):
    setattr(v2_functional, _name, globals()[_name])
v2_functional.InterpolationMode = InterpolationMode
