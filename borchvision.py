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
import collections as _collections
import csv as _csv
import enum as _enum
import glob as _glob
import gzip as _gzip
import hashlib as _hashlib
import html.parser as _html
import inspect as _inspect
import io as _io
import json as _json
import math as _math
import os as _os
import pickle as _pickle
import random as _random
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
    attaching to the sister library.

    **The tv_tensor types are rebound here too.** They subclass the backend's
    `Tensor`, so a module imported before this call holds five classes built on
    whichever library `_backend()` reached for first.
    """
    global _lib
    _lib = L
    globals().update(_tv_types(L))
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


# ── v2's bounding-box kernels ────────────────────────────────────────────────
#
# **These were declined as tv_tensor kernels and they take a plain tensor.**
#
# The row read *a tv_tensor type, and the type system is declined in `v2`*. Measured:
# torchvision's own error on a bare tensor is *"For pure tensor inputs, `format`,
# `canvas_size` and `clamping_mode` have to be passed"* — so the plain-tensor path is a
# **supported, documented** one, and what the tv_tensor carries is handed over as
# ordinary arguments instead. Sixth time in this repository that a sentence which was
# true about torchvision's design was read as a reason we could not build something.
#
# ## What is deliberately different here
#
# `format` is a **string**, as everywhere else in this file, where torchvision wants a
# `BoundingBoxFormat`. That is not only for consistency: passing `"XYXY"` to torchvision
# raises `IndexError: index 4 is out of bounds` — the string is iterated as if it were a
# format — and a name that reads as accepted and fails four frames later is the thing
# this library refuses on purpose.
#
# `clamping_mode` is not taken. torchvision defaults it to `"soft"` on the crop family,
# `"auto"` on `clamp` and does not have it on the flips, and only the `"soft"` behaviour
# is reproduced here. A parameter accepted and ignored is worse than one that is absent,
# so it is absent, and `clamp_bounding_boxes` stays declined for the same reason: its
# whole subject is that taxonomy.


def _canvas(canvas_size, where):
    """`(height, width)`, and it is checked rather than unpacked.

    **A box's own numbers run `(x, y)` and a canvas runs `(height, width)`**, so the two
    are transposed with respect to each other and a caller who swaps them gets boxes that
    are wrong without being obviously wrong. Two of torchvision's own names disagree about
    this (`get_size` against `get_image_size`), which is why it is spelled out at every
    entrance rather than once at the top.
    """
    try:
        height, width = canvas_size
    except (TypeError, ValueError):
        raise ValueError(
            f"{where} wants canvas_size as (height, width) — got {canvas_size!r}.\n"
            "(torch: canvas_size must be a two-element sequence)") from None
    return float(height), float(width)


def _boxes_as_xyxy(boxes, fmt, where):
    if fmt not in _BOX_FORMATS:
        raise ValueError(
            f"Unsupported Bounding Box format {fmt} for {where} — it is one of "
            f"{', '.join(_BOX_FORMATS)}.\n"
            f"(torch: Unsupported Bounding Box format {fmt})")
    arr, was_tensor = _boxes_in(boxes)
    return _to_xyxy(arr, fmt), was_tensor


def _xyxy_back(xyxy, fmt, was_tensor):
    if fmt == "xyxy":
        out = xyxy
    else:
        x1, y1, x2, y2 = xyxy[..., 0], xyxy[..., 1], xyxy[..., 2], xyxy[..., 3]
        w, h = x2 - x1, y2 - y1
        out = (_np.stack((x1, y1, w, h), axis=-1) if fmt == "xywh"
               else _np.stack((x1 + 0.5 * w, y1 + 0.5 * h, w, h), axis=-1))
    return _boxes_out(out, was_tensor, _np.float32)


def _clamp_to(xyxy, height, width):
    """Corners back inside a canvas of `(height, width)`.

    **This is what torchvision's `clamping_mode="soft"` does** on the crop family, and it
    is why `center_crop_bounding_boxes` can return a box of zero width: a box that left
    the picture entirely collapses onto the edge rather than being dropped. Dropping is
    `sanitize_bounding_boxes`, one function over, and keeping the two apart is the point —
    the caller usually has labels indexed alongside these rows.
    """
    out = xyxy.copy()
    out[..., 0::2] = _np.clip(out[..., 0::2], 0.0, width)
    out[..., 1::2] = _np.clip(out[..., 1::2], 0.0, height)
    return out


def horizontal_flip_bounding_boxes(bounding_boxes, format, canvas_size):  # noqa: A002
    """Boxes mirrored left-to-right across a canvas of `canvas_size`."""
    height, width = _canvas(canvas_size, "horizontal_flip_bounding_boxes")
    xyxy, was_tensor = _boxes_as_xyxy(bounding_boxes, format,
                                      "horizontal_flip_bounding_boxes")
    out = xyxy.copy()
    out[..., 0], out[..., 2] = width - xyxy[..., 2], width - xyxy[..., 0]
    del height
    return _xyxy_back(out, format, was_tensor)


def vertical_flip_bounding_boxes(bounding_boxes, format, canvas_size):  # noqa: A002
    """Boxes mirrored top-to-bottom across a canvas of `canvas_size`."""
    height, width = _canvas(canvas_size, "vertical_flip_bounding_boxes")
    xyxy, was_tensor = _boxes_as_xyxy(bounding_boxes, format,
                                      "vertical_flip_bounding_boxes")
    out = xyxy.copy()
    out[..., 1], out[..., 3] = height - xyxy[..., 3], height - xyxy[..., 1]
    del width
    return _xyxy_back(out, format, was_tensor)


def crop_bounding_boxes(bounding_boxes, format, top, left, height, width):  # noqa: A002
    """Boxes moved into a crop's frame and clipped to it, as **`(boxes, canvas_size)`.**

    **Every one of these that changes the canvas returns the pair**, even where the
    caller just passed the new size in. An earlier draft of this file returned bare boxes
    from here, from `center_crop` and from `resized_crop`, and explained at length why
    torchvision was asymmetric about it. torchvision is not: the flips return a bare
    tensor because the canvas does not move, and the other five return the pair.

    The wrong version came out of a probe that read
    `out[0] if isinstance(out, tuple) else out` — **a helper that normalised away the
    exact property being measured**, and then the prose explained the normalised picture
    as though it were the measurement.
    """
    xyxy, was_tensor = _boxes_as_xyxy(bounding_boxes, format, "crop_bounding_boxes")
    out = xyxy.copy()
    out[..., 0::2] -= float(left)
    out[..., 1::2] -= float(top)
    return (_xyxy_back(_clamp_to(out, float(height), float(width)), format, was_tensor),
            (int(height), int(width)))


def center_crop_bounding_boxes(bounding_boxes, format, canvas_size,  # noqa: A002
                               output_size):
    """`crop` about the middle, as **`(boxes, canvas_size)`.**

    **The offset is `round`, and `round` in Python breaks ties to even.** torchvision
    writes `int(round((image_width - crop_width) / 2.0))`, so a margin of 19 gives 10 and
    a margin of 13 gives 6 — up in one and down in the other, from the same rule.

    This said *floor-divided* and was wrong, and every even output size agrees with floor
    division, so the only cases that can tell the two apart are the odd ones. The golden
    asks an odd size for that reason: measured against torchvision, `[10, 12]` matched
    while `[11, 13]` was off by a whole pixel in x.
    """
    height, width = _canvas(canvas_size, "center_crop_bounding_boxes")
    out_h, out_w = _canvas(output_size, "center_crop_bounding_boxes")
    top = int(round((height - out_h) / 2.0))
    left = int(round((width - out_w) / 2.0))
    return crop_bounding_boxes(bounding_boxes, format, top, left, out_h, out_w)


def pad_bounding_boxes(bounding_boxes, format, canvas_size, padding,  # noqa: A002
                       padding_mode="constant"):
    """Boxes shifted by a pad, as **`(boxes, canvas_size)`** — the canvas grew and the
    caller cannot tell by how much without repeating this arithmetic.

    `padding` is torch's four spellings: one number for all sides, two as
    `(left/right, top/bottom)`, four as `(left, top, right, bottom)`.
    """
    height, width = _canvas(canvas_size, "pad_bounding_boxes")
    pad = [padding] * 4 if isinstance(padding, (int, float)) else list(padding)
    if len(pad) == 2:
        pad = [pad[0], pad[1], pad[0], pad[1]]
    if len(pad) != 4:
        raise ValueError(
            f"padding wants 1, 2 or 4 numbers — got {len(pad)}.\n"
            "(torch: Padding must be an int or a 1, 2, or 4 element tuple)")
    # **`padding_mode` changes what fills the new pixels, not where a box goes.** It is
    # taken so that a caller writing torchvision's call reaches the same behaviour, and
    # it is genuinely unused here rather than silently unsupported.
    del padding_mode
    left, top, right, bottom = (float(v) for v in pad)
    xyxy, was_tensor = _boxes_as_xyxy(bounding_boxes, format, "pad_bounding_boxes")
    out = xyxy.copy()
    out[..., 0::2] += left
    out[..., 1::2] += top
    return (_xyxy_back(out, format, was_tensor),
            (int(height + top + bottom), int(width + left + right)))


def resize_bounding_boxes(bounding_boxes, canvas_size, size, max_size=None,
                          format="xyxy"):  # noqa: A002
    """Boxes scaled to a new canvas, as **`(boxes, canvas_size)`**.

    **`size` as a single number keeps the aspect ratio**, matching the shorter edge, and
    as a pair it does not. `max_size` then caps the longer edge, which only bites in the
    single-number case — torchvision raises if both are given with a pair, and so does
    this, because a cap that can never apply is a caller who meant something else.
    """
    height, width = _canvas(canvas_size, "resize_bounding_boxes")
    # **One rule, read from one place.** The mask and keypoint kernels ask the same
    # question about `size` and `max_size`, and this rule was written out inline here
    # first — three copies of *a single number keeps the aspect ratio* is exactly the
    # shape this repository keeps catching in its tables.
    new_h, new_w = _resize_target(int(height), int(width), size, max_size,
                                  "resize_bounding_boxes")
    xyxy, was_tensor = _boxes_as_xyxy(bounding_boxes, format, "resize_bounding_boxes")
    out = xyxy.copy()
    out[..., 0::2] *= new_w / width
    out[..., 1::2] *= new_h / height
    return _xyxy_back(out, format, was_tensor), (new_h, new_w)


def resized_crop_bounding_boxes(bounding_boxes, format, top, left,  # noqa: A002
                                height, width, size):
    """`crop` and then `resize`, as **`(boxes, canvas_size)`.**"""
    cropped, _ = crop_bounding_boxes(bounding_boxes, format, top, left, height, width)
    return resize_bounding_boxes(cropped, (height, width), size, format=format)


def sanitize_bounding_boxes(bounding_boxes, format="xyxy", canvas_size=None,  # noqa: A002
                            min_size=1.0, min_area=1.0):
    """The boxes worth keeping, as **`(boxes, mask)`** — a boolean per row, not indices.

    A mask is what a caller needs here: the labels and scores sit in parallel arrays and
    get filtered by the same one. `remove_small_boxes` in `ops` answers with indices for
    the same reason from the other direction — both spellings exist in torchvision and
    each is the shape its own callers already have.
    """
    xyxy, was_tensor = _boxes_as_xyxy(bounding_boxes, format,
                                      "sanitize_bounding_boxes")
    w = xyxy[..., 2] - xyxy[..., 0]
    h = xyxy[..., 3] - xyxy[..., 1]
    keep = (w >= min_size) & (h >= min_size) & (w * h >= min_area)
    if canvas_size is not None:
        height, width = _canvas(canvas_size, "sanitize_bounding_boxes")
        inside = ((xyxy[..., 0] >= 0) & (xyxy[..., 1] >= 0)
                  & (xyxy[..., 2] <= width) & (xyxy[..., 3] <= height))
        keep = keep & inside
    kept = _xyxy_back(xyxy[keep], format, was_tensor)
    return kept, (_backend().tensor(keep) if was_tensor else keep)


# ── v2's mask and keypoint kernels ───────────────────────────────────────────
#
# **Seventh time this shape has been met.** These carried *a tv_tensor type, and the
# type system is declined in `v2`*, and both take a plain tensor: a mask is an array
# whose last two axes are the picture, a keypoint set is `(..., K, 2)` of coordinates.
# Called on bare tensors torchvision answers normally, exactly as the box kernels do.
#
# **A mask is not an image with a different name.** Every one of these is the image
# operation with **nearest** sampling and nothing else — measured, `resize_mask` is
# `resize_image(interpolation=NEAREST)` to the last value, and `NEAREST_EXACT` is a
# different answer. Sampling a label map any other way averages class 3 and class 5 into
# class 4, which is a picture that looks right and means nothing.


def _mask_in(mask):
    arr = mask if isinstance(mask, _np.ndarray) else _to_numpy(mask)
    return arr, not isinstance(mask, _np.ndarray)


def _mask_out(arr, was_tensor):
    """**The dtype comes back as it went in.** A mask is labels, and widening `uint8`
    to float here would make `==` comparisons against a class index start failing on
    values that are exactly equal on torchvision's side."""
    return _backend().tensor(arr) if was_tensor else arr


def _nearest_axis(src_len, dst_len):
    """Which source row each destination row reads from.

    `floor(i * src / dst)`, which is torch's `nearest` and **not** its `nearest-exact`.
    The two differ on every non-integer ratio and torchvision keeps both; masks get this
    one, so this one is what is written.
    """
    return _np.minimum((_np.arange(dst_len) * src_len // dst_len).astype(_np.int64),
                       src_len - 1)


def horizontal_flip_mask(mask):
    """A mask mirrored left-to-right."""
    arr, was_tensor = _mask_in(mask)
    return _mask_out(_np.ascontiguousarray(arr[..., ::-1]), was_tensor)


def vertical_flip_mask(mask):
    """A mask mirrored top-to-bottom."""
    arr, was_tensor = _mask_in(mask)
    return _mask_out(_np.ascontiguousarray(arr[..., ::-1, :]), was_tensor)


def crop_mask(mask, top, left, height, width):
    """A window out of a mask. **Outside the picture is zero rather than an error** —
    torchvision pads there, and a crop that half-leaves the frame is ordinary in a
    detection pipeline."""
    arr, was_tensor = _mask_in(mask)
    src_h, src_w = arr.shape[-2], arr.shape[-1]
    out = _np.zeros(arr.shape[:-2] + (int(height), int(width)), dtype=arr.dtype)
    y0, x0 = max(int(top), 0), max(int(left), 0)
    y1 = min(int(top) + int(height), src_h)
    x1 = min(int(left) + int(width), src_w)
    if y1 > y0 and x1 > x0:
        out[..., y0 - int(top):y1 - int(top), x0 - int(left):x1 - int(left)] = \
            arr[..., y0:y1, x0:x1]
    return _mask_out(out, was_tensor)


def center_crop_mask(mask, output_size):
    """`crop_mask` about the middle, **with the same `round`-to-even offset** the box
    kernels use. Measured against torchvision on an odd output size, where a floor
    division is off by a whole row."""
    arr, _ = _mask_in(mask)
    out_h, out_w = _canvas(output_size, "center_crop_mask")
    top = int(round((arr.shape[-2] - out_h) / 2.0))
    left = int(round((arr.shape[-1] - out_w) / 2.0))
    return crop_mask(mask, top, left, out_h, out_w)


def pad_mask(mask, padding, fill=0, padding_mode="constant"):
    """A mask with a border. **`fill` is 0 by default and 0 is a class**, so padding a
    label map writes background around it — which is what torchvision does and what a
    segmentation loss then has to be told to ignore."""
    arr, was_tensor = _mask_in(mask)
    pad = [padding] * 4 if isinstance(padding, (int, float)) else list(padding)
    if len(pad) == 2:
        pad = [pad[0], pad[1], pad[0], pad[1]]
    if len(pad) != 4:
        raise ValueError(
            f"padding wants 1, 2 or 4 numbers — got {len(pad)}.\n"
            "(torch: Padding must be an int or a 1, 2, or 4 element tuple)")
    if padding_mode != "constant":
        raise ValueError(
            f"pad_mask takes padding_mode='constant' here — got {padding_mode!r}.\n"
            "  The other modes reflect or repeat the edge, which invents labels that "
            "were never annotated.")
    left, top, right, bottom = (int(v) for v in pad)
    widths = [(0, 0)] * (arr.ndim - 2) + [(top, bottom), (left, right)]
    return _mask_out(_np.pad(arr, widths, mode="constant", constant_values=fill),
                     was_tensor)


def resize_mask(mask, size, max_size=None):
    """A mask resampled, **nearest and only nearest.**

    Any other sampling averages neighbouring class indices — 3 and 5 becoming 4 — and
    produces a label map that is smooth, plausible and wrong everywhere two regions meet.
    """
    arr, was_tensor = _mask_in(mask)
    src_h, src_w = arr.shape[-2], arr.shape[-1]
    new_h, new_w = _resize_target(src_h, src_w, size, max_size, "resize_mask")
    rows = _nearest_axis(src_h, new_h)
    cols = _nearest_axis(src_w, new_w)
    return _mask_out(_np.ascontiguousarray(arr[..., rows, :][..., :, cols]), was_tensor)


def resized_crop_mask(mask, top, left, height, width, size):
    """`crop_mask` then `resize_mask`."""
    return resize_mask(crop_mask(mask, top, left, height, width), size)


def _resize_target(src_h, src_w, size, max_size, where):
    """The `(height, width)` a `size` argument asks for.

    **One number keeps the aspect ratio and a pair does not**, which is two functions
    wearing one name, and `max_size` only bites in the first case — so passing it with a
    pair raises rather than being quietly unused.
    """
    want = [size] if isinstance(size, (int, float)) else list(size)
    if len(want) == 1:
        short, long_ = min(src_h, src_w), max(src_h, src_w)
        new_short = float(want[0])
        new_long = long_ * new_short / short
        if max_size is not None and new_long > max_size:
            new_short = new_short * max_size / new_long
            new_long = float(max_size)
        new_h, new_w = ((new_short, new_long) if src_h <= src_w
                        else (new_long, new_short))
        return int(new_h), int(new_w)
    if len(want) == 2:
        if max_size is not None:
            raise ValueError(
                "max_size is only used when size is a single number.\n"
                "(torch: max_size should only be passed if size specifies the length "
                "of the smaller edge)")
        return int(want[0]), int(want[1])
    raise ValueError(
        f"{where}: size wants one or two numbers — got {len(want)}.\n"
        "(torch: size should be an int or a 1 or 2 element tuple/list)")


def _points_in(keypoints):
    """**Not `_boxes_in`.** That one is written for `(..., 4)` rows and these are
    `(..., 2)`; the numbers survive either way today, and reusing it would tie a
    keypoint's shape to a box's the first time one of them changes."""
    if isinstance(keypoints, _np.ndarray):
        return keypoints.astype(_np.float64), False
    return _to_numpy(keypoints).astype(_np.float64), True


def _points_out(arr, was_tensor):
    return _boxes_out(arr, was_tensor, _np.float32)


def horizontal_flip_keypoints(keypoints, canvas_size):
    """Points mirrored left-to-right — **`(width - 1) - x`, not `width - x`.**

    A box's `x2` is an exclusive edge living on `[0, width]`; a keypoint is a pixel
    index living on `[0, width - 1]`, so the same canvas mirrors them differently. The
    box kernels above really do use `width - x` and this really does use one less, and
    the two sit ten lines apart.

    Written the box way it is off by exactly one pixel everywhere — a skeleton that is
    still a skeleton, drawn one column from where it belongs. Measured against
    torchvision: on a 32-wide canvas, 0 goes to 31 and 31 goes to 0.

    **And nothing is clamped afterwards.** A point at `x = 32` on that canvas comes back
    at `-1`, because a keypoint that was outside stays outside; `clamp_keypoints` is
    where that decision lives.
    """
    _, width = _canvas(canvas_size, "horizontal_flip_keypoints")
    arr, was_tensor = _points_in(keypoints)
    out = arr.copy()
    out[..., 0] = (width - 1) - arr[..., 0]
    return _points_out(out, was_tensor)


def vertical_flip_keypoints(keypoints, canvas_size):
    """Points mirrored top-to-bottom — `(height - 1) - y`, for the reason above."""
    height, _ = _canvas(canvas_size, "vertical_flip_keypoints")
    arr, was_tensor = _points_in(keypoints)
    out = arr.copy()
    out[..., 1] = (height - 1) - arr[..., 1]
    return _points_out(out, was_tensor)


def crop_keypoints(keypoints, top, left, height, width):
    """Points moved into a crop's frame, as **`(keypoints, canvas_size)`.**

    **Nothing is clipped and nothing is dropped.** A point outside the crop keeps its
    negative coordinate, because a keypoint that left the frame is still that keypoint —
    the caller decides whether it counts, and `sanitize_keypoints` is where that
    decision lives.
    """
    arr, was_tensor = _points_in(keypoints)
    out = arr.copy()
    out[..., 0] -= float(left)
    out[..., 1] -= float(top)
    return _points_out(out, was_tensor), (int(height), int(width))


def resize_keypoints(keypoints, size, canvas_size, max_size=None):
    """Points scaled to a new canvas, as **`(keypoints, canvas_size)`.**"""
    height, width = _canvas(canvas_size, "resize_keypoints")
    new_h, new_w = _resize_target(int(height), int(width), size, max_size,
                                  "resize_keypoints")
    arr, was_tensor = _points_in(keypoints)
    out = arr.copy()
    out[..., 0] *= new_w / width
    out[..., 1] *= new_h / height
    return _points_out(out, was_tensor), (new_h, new_w)


def center_crop_keypoints(inpt, canvas_size, output_size):
    """`crop_keypoints` about the middle, as **`(keypoints, canvas_size)`.**

    **The first parameter is `inpt` and its neighbours take `keypoints`.** torchvision
    spells this one differently from the rest of the family; `tests/ts_signatures.py`
    compares parameter names, so the odd one stays odd rather than being tidied.
    """
    height, width = _canvas(canvas_size, "center_crop_keypoints")
    out_h, out_w = _canvas(output_size, "center_crop_keypoints")
    top = int(round((height - out_h) / 2.0))
    left = int(round((width - out_w) / 2.0))
    return crop_keypoints(inpt, top, left, out_h, out_w)


def pad_keypoints(keypoints, canvas_size, padding, padding_mode="constant"):
    """Points shifted by a pad, as **`(keypoints, canvas_size)`.**"""
    height, width = _canvas(canvas_size, "pad_keypoints")
    pad = [padding] * 4 if isinstance(padding, (int, float)) else list(padding)
    if len(pad) == 2:
        pad = [pad[0], pad[1], pad[0], pad[1]]
    if len(pad) != 4:
        raise ValueError(
            f"padding wants 1, 2 or 4 numbers — got {len(pad)}.\n"
            "(torch: Padding must be an int or a 1, 2, or 4 element tuple)")
    del padding_mode          # what fills new pixels, not where a point goes
    left, top, right, bottom = (float(v) for v in pad)
    arr, was_tensor = _points_in(keypoints)
    out = arr.copy()
    out[..., 0] += left
    out[..., 1] += top
    return (_points_out(out, was_tensor),
            (int(height + top + bottom), int(width + left + right)))


def resized_crop_keypoints(keypoints, top, left, height, width, size):
    """`crop_keypoints` then `resize_keypoints`, as **`(keypoints, canvas_size)`.**"""
    cropped, canvas = crop_keypoints(keypoints, top, left, height, width)
    return resize_keypoints(cropped, size, canvas)


def sanitize_keypoints(key_points, canvas_size=None):
    """The point groups worth keeping, as **`(keypoints, mask)`.**

    **The unit is the group, not the point.** These arrive as `(N, K, 2)` — `N`
    skeletons of `K` points each — and a group survives only if **every** one of its
    points is inside the canvas. The mask is one boolean per group, so `N` and not
    `N * K`.

    Filtering point by point would be the obvious reading and would return a ragged
    thing that is no longer a skeleton: half a pose, with the joints renumbered.

    `canvas_size` has a default of `None` and **is refused when it is None**, which is
    torchvision's own behaviour on a plain tensor — the canvas normally rides on the
    tv_tensor, and here there is nothing for it to ride on.
    """
    if canvas_size is None:
        raise ValueError(
            "canvas_size cannot be None if key_points is a pure tensor. Set it to an "
            "appropriate value.\n"
            "(torch: canvas_size cannot be None if key_points is a pure tensor)")
    height, width = _canvas(canvas_size, "sanitize_keypoints")
    arr, was_tensor = _points_in(key_points)
    inside = ((arr[..., 0] >= 0) & (arr[..., 0] <= width)
              & (arr[..., 1] >= 0) & (arr[..., 1] <= height))
    keep = inside.reshape(arr.shape[0], -1).all(axis=1)
    return (_points_out(arr[keep], was_tensor),
            _backend().tensor(keep) if was_tensor else keep)


def clamp_keypoints(keypoints, canvas_size):
    """Points pushed back inside the canvas — **the last pixel is `width - 1`.**

    `_clamp_to`, which the box kernels use, clips to `width`. Same canvas, different
    last legal value, for the same reason the flips differ: an edge may sit on the
    boundary and an index may not.
    """
    height, width = _canvas(canvas_size, "clamp_keypoints")
    arr, was_tensor = _points_in(keypoints)
    out = arr.copy()
    out[..., 0] = _np.clip(out[..., 0], 0.0, width - 1)
    out[..., 1] = _np.clip(out[..., 1], 0.0, height - 1)
    return _points_out(out, was_tensor)


# ── v2's corner warps ────────────────────────────────────────────────────────
#
# `affine`, `rotate`, `perspective` and `elastic`, for boxes, masks and keypoints —
# twelve names, and three quite different jobs behind them.
#
# **A mask is the image kernel with `nearest`**, measured for all four: `affine_mask`
# equals `affine_image(interpolation=NEAREST)` to the last value, and so do the other
# three. Nothing new is computed for them; only the layout moves, since a mask is
# `(..., H, W)` and this file's image functions take `(H, W, C)`.
#
# **A keypoint is the transform applied to the point.** **A box is the transform applied
# to its four corners, and then the axis-aligned box around them** — which is why a
# rotated box grows: the hull of a tilted rectangle is bigger than the rectangle.
#
# ## The return shapes are not uniform and were measured one at a time
#
#     affine        boxes: bare      keypoints: (points, canvas)
#     rotate        boxes: (boxes, canvas)      keypoints: (points, canvas)
#     perspective   boxes: bare      keypoints: bare
#     elastic       boxes: bare      keypoints: bare
#
# `affine` returning a pair for keypoints and a bare tensor for boxes is torchvision's,
# and it is reproduced rather than tidied. A probe that normalised tuples away hid this
# once already in this file's history.
#
# ## Boxes clamp and keypoints do not
#
# Measured: `affine_bounding_boxes` with `translate=[3, -2]` gives `y1 = 0` where the
# arithmetic gives `-2`; `affine_keypoints` at the same angle answers `-6.392` and keeps
# it. A box that half-leaves the picture is still a region of the picture; a keypoint
# that leaves it is outside, and saying so is the only way `sanitize_keypoints` can mean
# anything.


def _forward_affine(center, angle, translate, scale, shear, w, h):
    """The 2x3 that moves a **point**, from the one that moves a grid.

    `_inverse_affine_matrix` answers what image resampling needs: output positions mapped
    back to input ones. A corner travels the other way, so its 2x2 is inverted here —
    arithmetic rather than a second formula, because two formulas for one transform is
    how they come to disagree.

    ## The centre is absolute here and an offset there

    `_center_offset` exists because **an image grid is already centred**: torch's grid
    runs `[-1, 1]` from the middle, so a centre arrives as a displacement from it and the
    default is `[0, 0]`. A **point** lives in pixel space, where the origin is the corner
    and the default centre is `[w/2, h/2]`.

    Reusing the grid convention for points was the first version and it is off by exactly
    one centring: measured, its answer for the default centre equalled torchvision's
    answer for `center=[0, 0]`, all the way through — a rotation about the corner where
    a rotation about the middle was asked for. Both conventions are correct and they are
    correct about different spaces.

    `translate` is applied **after** the linear part rather than inside it, which is what
    torchvision does and what makes `translate=[3, -2]` at angle 0 move a box by exactly
    three and two.
    """
    m = _inverse_affine_matrix([0.0, 0.0], angle, [0.0, 0.0], scale, _shear_pair(shear))
    a, b, _, d, e, _ = m
    det = a * e - b * d
    if abs(det) < 1e-12:
        raise ValueError(
            "this affine collapses the picture to a line, so a point cannot be carried "
            "back through it.\n(torch: singular affine matrix)")
    fwd = (e / det, -b / det, -d / det, a / det)
    cx, cy = ([w * 0.5, h * 0.5] if center is None else [float(v) for v in center])
    tx, ty = (float(t) for t in translate)
    return fwd, (cx, cy), (tx, ty)


def _apply_affine(points, prepared):
    (a, b, d, e), (cx, cy), (tx, ty) = prepared
    x = points[..., 0] - cx
    y = points[..., 1] - cy
    return _np.stack((a * x + b * y + cx + tx, d * x + e * y + cy + ty), axis=-1)


def _apply_perspective(points, coeffs):
    a, b, c, d, e, f, g, h = coeffs
    x, y = points[..., 0], points[..., 1]
    denom = g * x + h * y + 1.0
    return _np.stack(((a * x + b * y + c) / denom, (d * x + e * y + f) / denom), axis=-1)


def _corners(xyxy):
    """A box's four corners as `(..., 4, 2)`, in the order a hull does not care about."""
    x1, y1, x2, y2 = xyxy[..., 0], xyxy[..., 1], xyxy[..., 2], xyxy[..., 3]
    return _np.stack((_np.stack((x1, y1), axis=-1), _np.stack((x2, y1), axis=-1),
                      _np.stack((x2, y2), axis=-1), _np.stack((x1, y2), axis=-1)),
                     axis=-2)


def _hull(points):
    """The axis-aligned box around each group of corners."""
    return _np.stack((points[..., 0].min(axis=-1), points[..., 1].min(axis=-1),
                      points[..., 0].max(axis=-1), points[..., 1].max(axis=-1)), axis=-1)


def _warp_boxes(bounding_boxes, format, canvas, move, where):  # noqa: A002
    """Corners through `move`, then the hull, then clipped to `canvas`."""
    xyxy, was_tensor = _boxes_as_xyxy(bounding_boxes, format, where)
    moved = _hull(move(_corners(xyxy)))
    return _xyxy_back(_clamp_to(moved, canvas[0], canvas[1]), format, was_tensor)


def _mask_through(mask, call):
    """A mask through an image warp. **The layout moves and nothing else does.**

    A mask is `(..., H, W)` and this file's image functions are `(H, W, C)`, so the
    leading axes are folded into channels for the call and unfolded after. The dtype
    survives the round trip because a mask is labels.
    """
    arr, was_tensor = _mask_in(mask)
    lead, h, w = arr.shape[:-2], arr.shape[-2], arr.shape[-1]
    planes = int(_np.prod(lead)) if lead else 1
    hwc = arr.reshape(planes, h, w).transpose(1, 2, 0)
    out = _np.asarray(call(hwc))
    back = out.transpose(2, 0, 1).reshape(lead + out.shape[:2])
    return _mask_out(back.astype(arr.dtype), was_tensor)


def affine_mask(mask, angle, translate, scale, shear, fill=None, center=None):
    """`affine` on a label map — **nearest, and nearest is not a default here.**"""
    return _mask_through(mask, lambda img: affine(
        img, angle, translate, scale, shear, interpolation="nearest", fill=fill,
        center=center))


def rotate_mask(mask, angle, expand=False, center=None, fill=None):
    """`rotate` on a label map."""
    return _mask_through(mask, lambda img: rotate(
        img, angle, interpolation="nearest", expand=expand, center=center, fill=fill))


def perspective_mask(mask, startpoints, endpoints, fill=None, coefficients=None):
    """`perspective` on a label map."""
    del coefficients          # solved from the corners here, as `perspective` does
    return _mask_through(mask, lambda img: perspective(
        img, startpoints, endpoints, interpolation="nearest", fill=fill))


def elastic_mask(mask, displacement, fill=None):
    """`elastic` on a label map."""
    return _mask_through(mask, lambda img: elastic_transform(
        img, displacement, interpolation="nearest", fill=fill))


def affine_keypoints(keypoints, canvas_size, angle, translate, scale, shear,
                     center=None):
    """Points through an affine, as **`(keypoints, canvas_size)`.** Not clamped."""
    height, width = _canvas(canvas_size, "affine_keypoints")
    m = _forward_affine(center, angle, translate, scale, shear, width, height)
    arr, was_tensor = _points_in(keypoints)
    return _points_out(_apply_affine(arr, m), was_tensor), (int(height), int(width))


def affine_bounding_boxes(bounding_boxes, format, canvas_size, angle,  # noqa: A002
                          translate, scale, shear, center=None):
    """Box corners through an affine, then the axis-aligned box around them.

    **A rotated box grows**, and that is the transform being honest rather than a defect:
    the smallest upright rectangle containing a tilted one is larger than it. Repeated
    rotation therefore inflates a box, which is why detection pipelines rotate once.
    """
    height, width = _canvas(canvas_size, "affine_bounding_boxes")
    m = _forward_affine(center, angle, translate, scale, shear, width, height)
    return _warp_boxes(bounding_boxes, format, (height, width),
                       lambda pts: _apply_affine(pts, m), "affine_bounding_boxes")


def _expanded(angle, center, width, height):
    """`(new_h, new_w)` for `expand=True`, **and this is not the image's answer.**

    torchvision's coordinate path and its image path disagree about the expanded size,
    and they disagree because they hand `_compute_affine_output_size` different matrices:
    the image path passes a centre **offset** (`[0, 0]` by default, since a grid is
    already centred) and this path passes the **absolute** centre `[w/2, h/2]`. Same
    function, different translation column, different answer.

    Measured across four canvases and six angles: they agree on 0, 90 and a few others
    and part on most, in both directions — a 24x32 at 30 degrees is 38 rows to the image
    and 37 to the boxes; at -47 it is 40 to the image and 41 to the boxes. Boxes and
    keypoints agree with each other throughout.

    So this is not the same call `rotate` makes for a picture, and it cannot be: a mask
    rotated with `expand` comes out the image's size and a box in that same call comes
    out clipped to this one. **Both are reproduced rather than reconciled** — the point
    of this library is to answer what torch answers.
    """
    center_abs = ([width * 0.5, height * 0.5] if center is None
                  else [float(v) for v in center])
    matrix = _inverse_affine_matrix(center_abs, -angle, [0.0, 0.0], 1.0, [0.0, 0.0])
    ow, oh = _affine_output_size(matrix, int(width), int(height))
    return int(oh), int(ow)


def _recentred(move, width, height):
    """`move`, then shifted so the turned picture's own corner sits at the origin.

    **The shift is the minimum of the transformed canvas corners**, not half the growth
    in each direction. Half the growth was the first version and it is wrong by a
    fraction of a pixel whenever the picture is not square about its centre — measured,
    0.14 in x and 0.61 in y on a 24x32 at 30 degrees, which is small enough to read as a
    rounding difference and is not one.
    """
    frame = _np.asarray([[0.0, 0.0], [0.0, height], [width, height], [width, 0.0]])
    shift = move(frame).min(axis=0)
    return lambda pts: move(pts) - shift


def _rotation_move(angle, center, width, height, expand):
    """**`rotate` and `affine` disagree about which way is positive**, and the negation
    is what makes them agree from outside — the same negation `rotate` does for images,
    said again here rather than inherited, because these two do not share a body."""
    prepared = _forward_affine(center, -angle, [0.0, 0.0], 1.0, [0.0, 0.0], width, height)

    def move(pts):
        return _apply_affine(pts, prepared)
    return _recentred(move, width, height) if expand else move


def rotate_keypoints(keypoints, canvas_size, angle, expand=False, center=None):
    """Points turned about the centre, as **`(keypoints, canvas_size)`.**"""
    height, width = _canvas(canvas_size, "rotate_keypoints")
    new_h, new_w = (_expanded(angle, center, width, height) if expand
                    else (int(height), int(width)))
    move = _rotation_move(angle, center, width, height, expand)
    arr, was_tensor = _points_in(keypoints)
    return _points_out(move(arr), was_tensor), (new_h, new_w)


def rotate_bounding_boxes(bounding_boxes, format, canvas_size, angle,  # noqa: A002
                          expand=False, center=None):
    """Box corners turned about the centre, as **`(boxes, canvas_size)`.**

    `expand` grows the canvas to hold the whole turned picture, and the boxes are then
    clipped to **that** canvas rather than the original one.
    """
    height, width = _canvas(canvas_size, "rotate_bounding_boxes")
    new_h, new_w = (_expanded(angle, center, width, height) if expand
                    else (int(height), int(width)))
    move = _rotation_move(angle, center, width, height, expand)
    out = _warp_boxes(bounding_boxes, format, (new_h, new_w), move,
                      "rotate_bounding_boxes")
    return out, (new_h, new_w)


def _forward_perspective(startpoints, endpoints, coefficients):
    """**The arguments go in the other order than for an image.**

    `_perspective_coefficients` solves the map an image needs — output positions back to
    input ones, so `endpoints` to `startpoints`. A corner travels the other way, and
    swapping the two lists is the whole difference.
    """
    if coefficients is not None:
        return list(coefficients)
    return _perspective_coefficients(endpoints, startpoints)


def perspective_keypoints(keypoints, canvas_size, startpoints, endpoints,
                          coefficients=None):
    """Points through a projective map. Not clamped, and not a pair."""
    _canvas(canvas_size, "perspective_keypoints")
    coeffs = _forward_perspective(startpoints, endpoints, coefficients)
    arr, was_tensor = _points_in(keypoints)
    return _points_out(_apply_perspective(arr, coeffs), was_tensor)


def perspective_bounding_boxes(bounding_boxes, format, canvas_size,  # noqa: A002
                               startpoints, endpoints, coefficients=None):
    """Box corners through a projective map, then the hull."""
    height, width = _canvas(canvas_size, "perspective_bounding_boxes")
    coeffs = _forward_perspective(startpoints, endpoints, coefficients)
    return _warp_boxes(bounding_boxes, format, (height, width),
                       lambda pts: _apply_perspective(pts, coeffs),
                       "perspective_bounding_boxes")


def _elastic_move(displacement, width, height):
    """The field, read at each point's own pixel — **and the point snaps to that pixel.**

    Three things here, and two of them were measured rather than guessed.

    **The field is in normalised coordinates**, `[-1, 1]` across the picture, so `0.1` is
    `0.1 * width / 2` pixels. And it is **subtracted**: the field says where a destination
    pixel reads *from*, so a feature at `p` ends up at `p` minus what the field holds.

    **The point is floored to a whole pixel and clamped to the canvas, and the
    displacement is then applied to that pixel rather than to the original.** The
    fractional part is discarded. Measured with a field whose every entry names its own
    coordinate: `(10.4, 5.6)` and `(10.6, 5.4)` come back **identical**, which they
    cannot if the fraction survives; and `(33, 26)` on a 24x32 canvas comes back
    identical to `(31, 23)`, which it cannot if only the lookup index is clamped.

    The obvious implementation — round the index, keep the point — agrees on every point
    that is inside the picture and on a whole pixel. The golden's box table has corners
    at `(32, 24)` and its keypoints one at `(33, 26)` for that reason; without them this
    reads as correct.
    """
    field = _np.asarray(displacement if isinstance(displacement, _np.ndarray)
                        else _to_numpy(displacement), dtype=_np.float64)
    field = field.reshape(field.shape[-3], field.shape[-2], 2)

    def move(points):
        cols = _np.clip(_np.floor(points[..., 0]).astype(_np.int64), 0,
                        field.shape[1] - 1)
        rows = _np.clip(_np.floor(points[..., 1]).astype(_np.int64), 0,
                        field.shape[0] - 1)
        picked = field[rows, cols]
        snapped = _np.stack((cols.astype(_np.float64), rows.astype(_np.float64)),
                            axis=-1)
        return snapped - _np.stack((picked[..., 0] * width / 2.0,
                                    picked[..., 1] * height / 2.0), axis=-1)
    return move


def elastic_keypoints(keypoints, canvas_size, displacement):
    """Points moved by a displacement field. Not clamped, and not a pair."""
    height, width = _canvas(canvas_size, "elastic_keypoints")
    arr, was_tensor = _points_in(keypoints)
    return _points_out(_elastic_move(displacement, width, height)(arr), was_tensor)


def elastic_bounding_boxes(bounding_boxes, format, canvas_size,  # noqa: A002
                           displacement):
    """Box corners moved by a displacement field, then the hull."""
    height, width = _canvas(canvas_size, "elastic_bounding_boxes")
    return _warp_boxes(bounding_boxes, format, (height, width),
                       _elastic_move(displacement, width, height),
                       "elastic_bounding_boxes")


def _reduce(values, reduction):
    """`none`, `mean` or `sum`, as every loss in torch takes.

    **An unknown name is refused rather than treated as `none`.** A typo silently
    meaning "no reduction" gives back a vector where a scalar was wanted, and the
    shape error then surfaces somewhere else entirely.
    """
    if reduction == "none":
        return values
    if reduction == "mean":
        return values.mean() if values.size else values.sum() * 0.0
    if reduction == "sum":
        return values.sum()
    raise ValueError(
        f"{reduction} is not a valid value for reduction — it is one of none, mean, sum.\n"
        f"(torch: Invalid Value for arg 'reduction': '{reduction}')")


def _pairwise(boxes1, boxes2):
    """The two box sets as numpy, **matched one to one.**

    The IoU functions above answer *every box against every box* because that is what a
    detector's assignment step needs. **A loss is the other question**: these arrive
    already paired, one prediction against its own target, and the answer is one number
    per pair rather than a matrix.

    Taking the diagonal of the matrix would give the same numbers and compute `N²` of
    them to keep `N`, which on a real batch is the whole cost of the loss.
    """
    a, was_tensor = _boxes_in(boxes1)
    b, _ = _boxes_in(boxes2)
    if a.shape != b.shape:
        raise ValueError(
            f"the two box sets must be the same shape to be paired — got {a.shape} "
            f"and {b.shape}.\n(torch: The size of tensor a must match the size of "
            "tensor b)")
    return a, b, was_tensor


def _pair_inter_union(a, b):
    lt = _np.maximum(a[..., :2], b[..., :2])
    rb = _np.minimum(a[..., 2:], b[..., 2:])
    wh = _np.clip(rb - lt, 0.0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    return inter, area_a + area_b - inter


def _enclosing(a, b):
    """Width and height of the smallest box containing both — the term every one of
    these losses adds on top of plain IoU."""
    lt = _np.minimum(a[..., :2], b[..., :2])
    rb = _np.maximum(a[..., 2:], b[..., 2:])
    wh = _np.clip(rb - lt, 0.0, None)
    return wh[..., 0], wh[..., 1]


def generalized_box_iou_loss(boxes1, boxes2, reduction="none", eps=1e-7):
    """`1 - giou`, **pair by pair.**

    Plain IoU is 0 for any two boxes that do not touch, however far apart they are, so
    it has no gradient to follow back. Subtracting what the enclosing box wastes keeps
    the value falling all the way to -1, and that is the whole reason a loss is built on
    this one.
    """
    a, b, was_tensor = _pairwise(boxes1, boxes2)
    inter, union = _pair_inter_union(a, b)
    iou = inter / (union + eps)
    w, h = _enclosing(a, b)
    area = w * h
    loss = 1 - (iou - (area - union) / (area + eps))
    return _boxes_out(_reduce(loss, reduction), was_tensor, _np.float32)


def distance_box_iou_loss(boxes1, boxes2, reduction="none", eps=1e-7):
    """`1 - diou` — IoU penalised by how far apart the centres are, as a fraction of
    the enclosing box's diagonal."""
    a, b, was_tensor = _pairwise(boxes1, boxes2)
    inter, union = _pair_inter_union(a, b)
    iou = inter / (union + eps)
    w, h = _enclosing(a, b)
    diagonal = w ** 2 + h ** 2
    centres = (((a[..., 0] + a[..., 2]) - (b[..., 0] + b[..., 2])) / 2) ** 2 + \
              (((a[..., 1] + a[..., 3]) - (b[..., 1] + b[..., 3])) / 2) ** 2
    return _boxes_out(_reduce(1 - (iou - centres / (diagonal + eps)), reduction),
                      was_tensor, _np.float32)


def complete_box_iou_loss(boxes1, boxes2, reduction="none", eps=1e-7):
    """`1 - ciou` — the distance term and **one more for the aspect ratio.**

    Two boxes sharing a centre and an area but not a shape score the same under the
    distance loss and differently here. The golden case for this one was chosen so that
    the two disagree: with matched aspect ratios the extra term is exactly zero and a
    case like that would pass while measuring nothing this function does.
    """
    a, b, was_tensor = _pairwise(boxes1, boxes2)
    inter, union = _pair_inter_union(a, b)
    iou = inter / (union + eps)
    w, h = _enclosing(a, b)
    diagonal = w ** 2 + h ** 2
    centres = (((a[..., 0] + a[..., 2]) - (b[..., 0] + b[..., 2])) / 2) ** 2 + \
              (((a[..., 1] + a[..., 3]) - (b[..., 1] + b[..., 3])) / 2) ** 2
    w_a, h_a = a[..., 2] - a[..., 0], a[..., 3] - a[..., 1]
    w_b, h_b = b[..., 2] - b[..., 0], b[..., 3] - b[..., 1]
    v = (4 / (_np.pi ** 2)) * (_np.arctan(w_b / (h_b + eps))
                               - _np.arctan(w_a / (h_a + eps))) ** 2
    alpha = v / (1 - iou + v + eps)
    loss = 1 - (iou - (centres / (diagonal + eps) + alpha * v))
    return _boxes_out(_reduce(loss, reduction), was_tensor, _np.float32)


def sigmoid_focal_loss(inputs, targets, alpha=0.25, gamma=2, reduction="none"):
    """Cross-entropy with **the easy examples turned down.**

    A detector looks at tens of thousands of boxes and almost all of them are plainly
    background. Summed with equal weight, that majority drowns out the few that are
    hard, so each term is scaled by `(1 - p_t) ** gamma` — near zero once the model is
    already confident and right.

    `alpha` is the separate, older fix for the same imbalance: one weight for the
    positive class and `1 - alpha` for the negative. **`alpha=-1` turns it off**, which
    is torchvision's own switch rather than a value.

    The inputs are logits, not probabilities. Passing something already through a
    sigmoid gives a number rather than an error, which is why it is said here.
    """
    x = _to_numpy(inputs) if not isinstance(inputs, _np.ndarray) else inputs
    y = _to_numpy(targets) if not isinstance(targets, _np.ndarray) else targets
    was_tensor = not isinstance(inputs, _np.ndarray)
    x = x.astype(_np.float64)
    y = y.astype(_np.float64)
    # **Both of these are written around `exp` overflowing, and it overflows at ±710.**
    #
    # The plain sigmoid `1 / (1 + exp(-x))` gives the right answer for a logit of -800 —
    # `exp(800)` is `inf` and the quotient is 0 — but it gets there through a numpy
    # overflow warning, and a library people are reading to learn from should not print
    # one while being correct. Taking `exp` of the negative half only keeps every
    # intermediate inside the range.
    #
    # `log1p(exp(-|x|))` is the same care for the cross-entropy: `log(1 + exp(-x))`
    # returns `inf` at -800 where the loss is finite. torch's own binary cross-entropy is
    # written this way, and measured against it here ±800 gives 200 and 600 on both sides.
    z = _np.exp(-_np.abs(x))
    p = _np.where(x >= 0, 1.0 / (1.0 + z), z / (1.0 + z))
    ce = _np.clip(x, 0, None) - x * y + _np.log1p(z)
    p_t = p * y + (1 - p) * (1 - y)
    loss = ce * ((1 - p_t) ** gamma)
    if alpha >= 0:
        loss = (alpha * y + (1 - alpha) * (1 - y)) * loss
    return _boxes_out(_reduce(loss, reduction), was_tensor, _np.float32)


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



# ── tv_tensors: the types that travel beside a picture ───────────────────────────
#
# **This is the half of v2 that was declined, and it was declined as one decision.**
# Twenty names in `transforms.v2` had reasons reading *boxes travelling with the
# picture — the point of v2's type system*, and thirteen of them were the type system
# rather than users of it. So it is built here, and the users follow.
#
# **A tv_tensor is a tensor with a label on it and almost no behaviour.** Reading
# torchvision's, the surprising part is how little survives: `img * 2` is a plain
# `Tensor` there, not an `Image`. Measured, the whole list of operations that keep the
# subclass is **`clone`, `detach`, `to` and `requires_grad_`** — everything else,
# including `cpu()`, indexing and every arithmetic operator, decays. That is deliberate
# on their side: a transform wraps the result back on purpose, so a half-transformed
# tensor cannot go on claiming to be a box.
#
# It means the type system is a **dispatch key**, not a container. Which is why the
# thirteen names on top of it are mostly one-line questions about a flattened sample.


class BoundingBoxFormat(_enum.Enum):
    """How four numbers describe a box. **`XYXY` is corners, `XYWH` is a corner and a
    size, `CXCYWH` is a centre and a size** — and the three are indistinguishable by
    looking at the numbers, which is the whole reason the label rides along."""

    XYXY = "XYXY"
    XYWH = "XYWH"
    CXCYWH = "CXCYWH"


_TV_NAMES = ("TVTensor", "Image", "Video", "Mask", "BoundingBoxes", "KeyPoints")
_TV_TYPES = {}


def _tv_types(L):
    """The five tv_tensor types and their base, **built on the backend in use.**

    They subclass `L.Tensor`, and which `L` that is has to be decided when the module
    is bound rather than when it is imported. Written as plain classes, the base was
    whichever library happened to be imported first — always the numpy core, because
    `_backend()` falls back to it — and in the browser the boxes then carried the
    core's `.data` descriptor while holding the binding's handle. **Thirteen golden
    cases said `'BoundingBoxes' object has no attribute '_array'`**, four frames away
    from anything that mentioned a backend.

    Cached on the backend, as `_ops_layers` is and for the same reason: rebuilt per
    call, `isinstance` would fail against a class the caller already holds.
    """
    made = _TV_TYPES.get(id(L))
    if made is not None:
        return made

    class TVTensor(L.Tensor):
        """The base every one of these subclasses.

        **Four operations keep the subclass and the rest do not.** `clone`, `detach`, `to`
        and `requires_grad_` come back as the same type; `+`, indexing, `reshape` and
        `cpu()` come back as plain tensors. That is torchvision's rule, measured rather
        than assumed, and the reason `wrap` exists: a transform puts the label back on
        deliberately, so nothing half-transformed keeps claiming to be a box.
        """

        _METADATA = ()

        def __new__(cls, data, **kwargs):
            """A tensor of `cls`, **built through the backend's own factory.**

            Not by calling `Tensor.__init__` with the arguments the core's takes: the three
            implementations do not share that signature, and the binding's raised
            *unexpected keyword argument `requires_grad`* on thirteen golden cases while
            numpy and the browser passed. Two of three answering is not this table's
            standard.

            So `L.tensor(...)` builds a real one and its instance state is moved onto an
            object of this class. What that assumes is only that the state lives in
            `__dict__`, which is true of all three.
            """
            L = _backend()
            built = data if isinstance(data, L.Tensor) else L.tensor(
                _np.ascontiguousarray(_np.asarray(data, dtype=_np.float32)))
            out = L.Tensor.__new__(cls)
            _copy_tensor_state(built, out)
            kwargs.pop("requires_grad", None)
            return out

        def __init__(self, *args, **kwargs):
            """**Everything happens in `__new__`, and this exists to swallow the arguments.**

            Python calls `__init__` with whatever `__new__` was called with, so
            `BoundingBoxes(data, canvas_size=...)` would reach `Tensor.__init__` and be
            refused for a keyword it does not take. The tensor is already built by the time
            this runs.
            """

        def _metadata(self):
            return {name: getattr(self, name) for name in self._METADATA}

        @classmethod
        def wrap(cls, wrappee, like, **kwargs):
            """The label from `like` put onto `wrappee`, with any of it overridden.

            **Subclasses do not each write this** — the metadata names are declared once in
            `_METADATA` and copied by name. A subclass that grew a field and forgot to copy
            it would produce boxes whose format silently came from nowhere.
            """
            out = cls.__new__(cls, wrappee)
            for name in cls._METADATA:
                setattr(out, name, kwargs.get(name, getattr(like, name)))
            return out

        def _same(self, values):
            return type(self).wrap(values, self)

        def _base(self, name, *args, **kwargs):
            """`super().<name>(...)`, **including the names the backend answers
            dynamically.**

            `super()` searches types. The binding forwards a name it does not have to
            the JavaScript tensor through `__getattr__` **on the instance**, which a
            type search never reaches — so `super().clone()` stopped with *'super'
            object has no attribute 'clone'* there while the core, where `clone` is a
            written method, answered. One golden case said so, and the other three
            below happened to be written methods on both sides: the difference was in
            which backend, not in which operation.
            """
            fn = getattr(L.Tensor, name, None)
            if fn is not None:
                return fn(self, *args, **kwargs)
            plain = L.Tensor.__new__(L.Tensor)
            _copy_tensor_state(self, plain)
            return getattr(plain, name)(*args, **kwargs)

        def clone(self):
            return self._same(self._base("clone"))

        def detach(self):
            return self._same(self._base("detach"))

        def to(self, *args, **kwargs):
            return self._same(self._base("to", *args, **kwargs))

        def requires_grad_(self, requires_grad=True):
            self._base("requires_grad_", requires_grad)
            return self

        def __repr__(self):
            inner = super().__repr__()
            extra = ", ".join(f"{name}={getattr(self, name)}" for name in self._METADATA)
            return f"{type(self).__name__}({inner}{', ' + extra if extra else ''})"


    class Image(TVTensor):
        """A picture. **No metadata at all** — the type is the whole of what it carries, and
        that is enough: a transform asks *is this the image* and the answer is the class."""


    class Video(TVTensor):
        """A clip. As `Image`, and separate from it because a transform that resizes a
        picture and a transform that resizes a video ask different questions about the
        leading axes."""


    class Mask(TVTensor):
        """A segmentation map. **Resized with nearest-neighbour and never interpolated**,
        which is the one thing knowing it is a mask buys — a bilinear resize invents labels
        that are the average of two classes and belong to neither."""


    class BoundingBoxes(TVTensor):
        """Boxes, with **the format and the canvas they are drawn on.**

        Both are needed and neither can be read off the numbers. The format because four
        numbers are four numbers; the canvas because clamping a box needs to know what it is
        being clamped to, and the picture may have been resized since.
        """

        _METADATA = ("format", "canvas_size", "clamping_mode")

        def __new__(cls, data, *, format=BoundingBoxFormat.XYXY, canvas_size=None,
                    clamping_mode="soft", **kwargs):
            out = super().__new__(cls, data, **kwargs)
            out.format = (format if isinstance(format, BoundingBoxFormat)
                          else BoundingBoxFormat[str(format).upper()])
            out.canvas_size = tuple(canvas_size) if canvas_size is not None else None
            out.clamping_mode = _check_clamping_mode(clamping_mode)
            return out


    class KeyPoints(TVTensor):
        """Points, with the canvas. **No format** — a keypoint is two numbers and there is
        only one way to write them, which is why this carries one field where the boxes
        carry three."""

        _METADATA = ("canvas_size",)

        def __new__(cls, data, *, canvas_size=None, **kwargs):
            out = super().__new__(cls, data, **kwargs)
            out.canvas_size = tuple(canvas_size) if canvas_size is not None else None
            return out

    made = {"TVTensor": TVTensor, "Image": Image, "Video": Video, "Mask": Mask,
            "BoundingBoxes": BoundingBoxes, "KeyPoints": KeyPoints}
    _TV_TYPES[id(L)] = made
    return made


# The names exist from import, so `from borchvision import BoundingBoxes` works
# before anything has chosen a backend. `use(L)` rebinds them.
globals().update(_tv_types(_backend()))


def _copy_tensor_state(src, dst):
    """One tensor's state onto another object, **whichever way the backend keeps it.**

    The core keeps it in `__dict__`; **the binding keeps it in `__slots__`**, because a
    tensor there is a handle to a GPU buffer and a per-instance dict is a real cost on
    something meant to be lean. So there is nothing to copy *from* a base-class
    instance's `__dict__` there — measured, thirteen golden cases said
    *'Tensor' object has no attribute '__dict__'*.

    The subclass declares no `__slots__` of its own, so it gets a dict and can hold the
    metadata; the tensor's own fields go across by name.

    **Both are copied, not one or the other.** Written as *dict if there is one, else
    slots*, it read the source's shape as the backend's — true while the source was
    always a freshly built base tensor, and false the moment a `BoundingBoxes` was the
    source: a subclass has a dict *and* inherits the binding's `_h`, so the handle was
    left behind. The plain tensor built from it then had no `_h`, its `__getattr__`
    asked for `_h`, and that asks `__getattr__` again — a `RecursionError` reported
    from a box that was only being clamped.
    """
    for kind in type(src).__mro__:
        for name in getattr(kind, "__slots__", ()):
            try:
                setattr(dst, name, getattr(src, name))
            except AttributeError:
                pass
    if hasattr(src, "__dict__") and hasattr(dst, "__dict__"):
        dst.__dict__.update(src.__dict__)
_CLAMPING_MODES = ("soft", "hard", None)


def _check_clamping_mode(value):
    """**`soft`, `hard` or `None`, and nothing else.**

    `soft` clamps the corners and leaves a box that has left the picture as a sliver on
    the edge; `hard` clamps and then drops what has no area; `None` does not clamp at
    all. An unknown word here would be accepted and ignored, and the boxes would be
    whichever the caller did not ask for.
    """
    if value not in _CLAMPING_MODES:
        raise ValueError(
            f"clamping_mode must be one of {_CLAMPING_MODES}, got {value!r}.")
    return value


def wrap(wrappee, *, like, **kwargs):
    """`wrappee` as the same kind as `like`, carrying its label.

    **This is how a transform puts the type back on.** Everything inside a transform
    works on plain tensors — the subclass decayed at the first arithmetic — so the last
    step is always this, and any field that changed is passed here: a resize hands over
    a new `canvas_size`, a format conversion a new `format`.
    """
    kind = type(like)
    if hasattr(kind, "wrap"):
        return kind.wrap(wrappee, like, **kwargs)
    return wrappee


_RETURN_TVTENSOR = [False]


class _ReturnType:
    def __init__(self, restore):
        self._restore = restore

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        _RETURN_TVTENSOR[0] = self._restore
        return False


def set_return_type(return_type="Tensor"):
    """Whether arithmetic on a tv_tensor comes back as one.

    **The default is `Tensor`, and that is the surprising half of the design.**
    `img + 2` is a plain tensor; inside `with set_return_type("TVTensor")` it is an
    `Image`. torchvision made the plain tensor the default because a subclass that
    survived every operation would keep a stale `canvas_size` through a resize, and a
    box that lies about its canvas is worse than one that has forgotten it is a box.

    Here the flag is **stored and read but changes nothing yet**, because arithmetic on
    this side never keeps the subclass — there is no `__torch_function__` to intercept.
    Saying so is the point: the name exists so that code written against torchvision
    runs, and the difference is written down rather than left to be discovered.
    """
    wanted = {"tensor": False, "tvtensor": True}.get(str(return_type).lower())
    if wanted is None:
        raise ValueError(
            f"return_type must be 'TVTensor' or 'Tensor', got {return_type}")
    previous = _RETURN_TVTENSOR[0]
    _RETURN_TVTENSOR[0] = wanted
    return _ReturnType(previous)


def is_rotated_bounding_format(format):
    """Whether a format describes a **rotated** box.

    None of the three formats here is: rotated boxes carry five numbers or eight, and
    this subset takes four. The name exists because code asks it before branching, and
    the answer is `False` — which is a real answer rather than a refusal.
    """
    if isinstance(format, str):
        format = BoundingBoxFormat[format.upper()]
    return format not in (BoundingBoxFormat.XYXY, BoundingBoxFormat.XYWH,
                          BoundingBoxFormat.CXCYWH)

# ── transforms.v2: the dispatch, and the thirteen names that are it ──────────────
#
# **Thirteen of `v2`'s twenty absences were the type system rather than users of it.**
# The rows read *boxes travelling with the picture — the point of v2's type system*, and
# the point of a thing is not a reason it cannot exist. What it needed was the types
# above, and they are eighty lines.
#
# What is here is the machinery every v2 transform stands on:
#
# - **a flatten and an unflatten**, so a transform can be handed a tensor, a tuple, a
#   dict or a nest of them and answer in the same shape;
# - **the question each transform asks of a flattened sample** — is this a box, is there
#   an image here, what size is everything;
# - **`Transform` itself**, whose whole body is that walk.
#
# The individual kernels — `affine_bounding_boxes` and its neighbours — are not here.


def _tree_flatten(value):
    """A nest into a flat list and a recipe for putting it back.

    **Dicts keep their key order and tuples keep their type**, because a transform hands
    back what it was given: a caller who passed `(image, target)` gets a tuple, and one
    who passed `{"img": ..., "boxes": ...}` gets those keys. Flattening to a list and
    rebuilding as a list would work for every test written with one tensor.
    """
    if isinstance(value, dict):
        flat, specs = [], []
        for key, item in value.items():
            inner, spec = _tree_flatten(item)
            flat.extend(inner)
            specs.append((key, spec))
        return flat, ("dict", specs)
    if isinstance(value, (list, tuple)):
        flat, specs = [], []
        for item in value:
            inner, spec = _tree_flatten(item)
            flat.extend(inner)
            specs.append(spec)
        return flat, ("tuple" if isinstance(value, tuple) else "list", specs)
    return [value], ("leaf", None)


def _tree_unflatten(flat, spec):
    """The other half. Consumes `flat` left to right, in the order `_tree_flatten` laid
    it out — the two walk the nest the same way or the values land in other slots."""
    kind, inner = spec
    if kind == "leaf":
        return flat.pop(0)
    if kind == "dict":
        return {key: _tree_unflatten(flat, sub) for key, sub in inner}
    rebuilt = [_tree_unflatten(flat, sub) for sub in inner]
    return tuple(rebuilt) if kind == "tuple" else rebuilt


def check_type(obj, types_or_checks):
    """Whether `obj` matches any of a mixed list of **types and predicates**.

    The mixture is what makes it worth a function: `is_pure_tensor` is a predicate
    because *a tensor that is not a tv_tensor* cannot be written as a type, and every
    caller here wants to ask about both in one list.
    """
    for one in types_or_checks:
        if isinstance(one, type):
            if isinstance(obj, one):
                return True
        elif one(obj):
            return True
    return False


def has_any(flat_inputs, *types_or_checks):
    """Whether **any** item matches any of them."""
    return any(check_type(one, types_or_checks) for one in flat_inputs)


def has_all(flat_inputs, *types_or_checks):
    """Whether **every** one of them is matched by some item.

    Not the same question as `has_any` with the arguments reversed, and the difference
    is the loop nesting: this asks each *check* whether anything satisfies it, and
    `has_any` asks each *item* whether it satisfies anything.
    """
    for one in types_or_checks:
        if not any(check_type(item, (one,)) for item in flat_inputs):
            return False
    return True


def is_pure_tensor(obj):
    """A tensor that is **not** one of the labelled kinds.

    The distinction the whole dispatch turns on: a transform treats the first plain
    tensor it meets as the image, and leaves the rest alone.
    """
    L = _backend()
    return isinstance(obj, L.Tensor) and not isinstance(obj, TVTensor)


def _size_of(item):
    shape = tuple(int(one) for one in item.shape)
    if isinstance(item, (BoundingBoxes, KeyPoints)):
        return item.canvas_size
    return shape[-2], shape[-1]


def query_size(flat_inputs):
    """The one `(H, W)` the whole sample agrees on.

    **Boxes answer with their canvas, not with their own shape.** A `(4, 4)` box tensor
    is four boxes, not a four-by-four picture — which is why this cannot be a `shape[-2:]`
    on everything, and why `canvas_size` is carried at all.

    Disagreement is an error rather than a first-one-wins, because a sample whose boxes
    think the picture is one size and whose mask thinks it is another has already gone
    wrong somewhere earlier.
    """
    L = _backend()
    sizes = {_size_of(one) for one in flat_inputs
             if check_type(one, (is_pure_tensor, Image, Video, Mask, BoundingBoxes,
                                 KeyPoints))}
    sizes.discard(None)
    if not sizes:
        raise TypeError("No image, video, mask, bounding box of keypoint was found in "
                        "the sample")
    if len(sizes) > 1:
        raise ValueError("Found multiple HxW dimensions in the sample: "
                         + ", ".join(str(one) for one in sorted(sizes)))
    return sizes.pop()


def query_chw(flat_inputs):
    """The one `(C, H, W)` the pictures agree on.

    **Only images and videos are asked** — a mask has no channels to speak of and a box
    has none at all, so including them would make the set disagree with itself on every
    real sample.
    """
    found = set()
    for one in flat_inputs:
        if check_type(one, (is_pure_tensor, Image, Video)):
            shape = tuple(int(part) for part in one.shape)
            if len(shape) < 3:
                raise TypeError("No image or video was found in the sample")
            found.add(shape[-3:])
    if not found:
        raise TypeError("No image or video was found in the sample")
    if len(found) > 1:
        raise ValueError("Found multiple CxHxW dimensions in the sample: "
                         + ", ".join(str(one) for one in sorted(found)))
    return found.pop()


def get_bounding_boxes(flat_inputs):
    """The sample's boxes, and **there must be exactly one lot of them.**

    torchvision's own note says so: one `BoundingBoxes` per sample, holding as many
    boxes as it likes. Two would leave every transform choosing between them silently.
    """
    found = [one for one in flat_inputs if isinstance(one, BoundingBoxes)]
    if not found:
        raise ValueError("No bounding boxes were found in the sample")
    if len(found) > 1:
        raise ValueError("Found multiple bounding boxes instances in the sample")
    return found[0]


def get_keypoints(flat_inputs):
    """As above, for keypoints."""
    found = [one for one in flat_inputs if isinstance(one, KeyPoints)]
    if not found:
        raise ValueError("No keypoints were found in the sample")
    if len(found) > 1:
        raise ValueError("Found multiple keypoints instances in the sample")
    return found[0]


def _v2_transform_base():
    """`Transform`, built against the attached backend for the reason the `ops` layers
    are — it subclasses `nn.Module` and `use(L)` can change which one that is."""
    nn = _backend().nn

    class Transform(nn.Module):
        """**The base class every v2 transform inherits, and its body is the dispatch.**

        A transform is handed a nest — a tensor, or `(image, target)`, or a dict — and
        has to answer in the same shape with only the parts it owns changed. So:
        flatten, decide per item whether it is transformed, draw the random parameters
        **once** for the whole sample, transform, unflatten.

        Two rules in that walk are what separate v2 from v1, and both are invisible on a
        sample holding one tensor:

        - **The parameters are drawn once.** A random crop has to take the same rectangle
          out of the picture and out of the mask; drawn per item they would drift apart
          and the pair would stop meaning anything.
        - **Only the first plain tensor is treated as the image**, and only when no
          `Image` or `Video` is present. A sample of `(picture, boxes_as_plain_tensor)`
          would otherwise have its boxes brightened.
        """

        _transformed_types = (_backend().Tensor,)

        def check_inputs(self, flat_inputs):
            """Refuse a sample this transform cannot serve. Empty by default."""

        def make_params(self, flat_inputs):
            """The random draw, once per sample. Empty by default."""
            return {}

        def transform(self, inpt, params):
            raise NotImplementedError

        def forward(self, *inputs):
            flat_inputs, spec = _tree_flatten(
                inputs if len(inputs) > 1 else inputs[0])
            self.check_inputs(flat_inputs)
            wanted = self._needs_transform_list(flat_inputs)
            params = self.make_params(
                [one for one, needs in zip(flat_inputs, wanted) if needs])
            out = [self.transform(one, params) if needs else one
                   for one, needs in zip(flat_inputs, wanted)]
            return _tree_unflatten(out, spec)

        def _needs_transform_list(self, flat_inputs):
            """Which items this transform touches.

            **The plain-tensor rule lives here**, and it is a running flag rather than a
            per-item test: the *first* plain tensor is the image, and only if the sample
            contains no `Image` or `Video` that has already claimed the job.
            """
            wanted = []
            take_plain = not has_any(flat_inputs, Image, Video)
            for one in flat_inputs:
                needs = True
                if not check_type(one, self._transformed_types):
                    needs = False
                elif is_pure_tensor(one):
                    if take_plain:
                        take_plain = False
                    else:
                        needs = False
                wanted.append(needs)
            return wanted

        def extra_repr(self):
            """**The plain settings, and nothing that is a tensor.** torchvision's rule,
            and the reason is that a transform holding a whitening matrix would print
            the matrix."""
            out = []
            for name, value in vars(self).items():
                if name.startswith("_") or name == "training":
                    continue
                if not isinstance(value, (bool, int, float, str, tuple, list,
                                          _enum.Enum)):
                    continue
                out.append(f"{name}={value}")
            return ", ".join(out)

    class ConvertBoundingBoxFormat(Transform):
        """Boxes from one format to another, **and the label goes with them.**

        The conversion itself is `ops.box_convert`, which was here already. What this
        adds is that the result is still a `BoundingBoxes` and its `format` now says the
        new one — without which the next transform reads the numbers under the old rule.
        """

        _transformed_types = (BoundingBoxes,)

        def __init__(self, format):
            super().__init__()
            self.format = format

        def transform(self, inpt, params):
            return convert_bounding_box_format(inpt, new_format=self.format)

    class ClampBoundingBoxes(Transform):
        """Corners pushed back inside the canvas.

        `clamping_mode="auto"` means **take the mode off the boxes themselves** — which
        is why `BoundingBoxes` carries one. A transform that hard-coded a mode would
        override a choice the caller made when they built the boxes.
        """

        _transformed_types = (BoundingBoxes,)

        def __init__(self, clamping_mode="auto"):
            super().__init__()
            self.clamping_mode = clamping_mode

        def transform(self, inpt, params):
            return clamp_bounding_boxes(inpt, clamping_mode=self.clamping_mode)

    class ClampKeyPoints(Transform):
        """As above, for points. **There is no mode** — a point is either inside or it is
        not, and there is no sliver to keep or drop."""

        _transformed_types = (KeyPoints,)

        def transform(self, inpt, params):
            return clamp_keypoints(inpt)

    class SetClampingMode(Transform):
        """Change which mode the boxes carry, **without clamping them.**

        Two things in one place is why this exists separately from `ClampBoundingBoxes`:
        the mode is a property of the boxes and applies to every later transform, and
        setting it is not the same as applying it now.
        """

        _transformed_types = (BoundingBoxes,)

        def __init__(self, clamping_mode):
            super().__init__()
            self.clamping_mode = _check_clamping_mode(clamping_mode)

        def transform(self, inpt, params):
            return BoundingBoxes.wrap(inpt, inpt,
                                      clamping_mode=self.clamping_mode)

    return {"Transform": Transform,
            "ConvertBoundingBoxFormat": ConvertBoundingBoxFormat,
            "ClampBoundingBoxes": ClampBoundingBoxes,
            "ClampKeyPoints": ClampKeyPoints,
            "SetClampingMode": SetClampingMode}


_V2_BASES = {}


def _v2_dispatch(L):
    made = _V2_BASES.get(id(L))
    if made is None:
        made = _v2_transform_base()
        _V2_BASES[id(L)] = made
    return made


def convert_bounding_box_format(inpt, old_format=None, new_format=None,
                                inplace=False):
    """Boxes between the three formats, **carrying the label across.**

    On a `BoundingBoxes` the old format is read off the boxes and passing it as well is
    refused — two sources for one fact is a place they can disagree. On a plain tensor
    there is nothing to read, so it must be given.
    """
    L = _backend()
    if new_format is None:
        raise ValueError("new_format must be given")
    new_format = (new_format if isinstance(new_format, BoundingBoxFormat)
                  else BoundingBoxFormat[str(new_format).upper()])
    if isinstance(inpt, BoundingBoxes):
        if old_format is not None:
            raise ValueError("For bounding box tv_tensor inputs, `old_format` must not "
                             "be passed.")
        old_format = inpt.format
    elif old_format is None:
        raise ValueError("For pure tensor inputs, `old_format` has to be passed.")
    else:
        old_format = (old_format if isinstance(old_format, BoundingBoxFormat)
                      else BoundingBoxFormat[str(old_format).upper()])
    if old_format == new_format:
        values = inpt.clone() if hasattr(inpt, "clone") else inpt
    else:
        values = L.tensor(_np.ascontiguousarray(_np.asarray(
            box_convert(_np.asarray(_to_numpy(inpt), dtype=_np.float32),
                        old_format.value.lower(), new_format.value.lower()),
            dtype=_np.float32)))
    if isinstance(inpt, BoundingBoxes):
        return BoundingBoxes.wrap(values, inpt, format=new_format)
    return values


def clamp_bounding_boxes(inpt, format=None, canvas_size=None, clamping_mode="auto"):
    """Corners pushed back inside the canvas.

    **`soft` and `hard` are the same thing for a box that is not rotated**, which is
    every box here — torchvision says so in a comment and it is copied rather than
    guessed at, because the two names suggest a difference the arithmetic does not have.
    `None` returns the boxes unchanged.

    `auto` reads the mode off the boxes, and is refused on a plain tensor: there is
    nothing to read it from, and defaulting would pick a mode the caller did not.
    """
    L = _backend()
    if clamping_mode is not None and clamping_mode not in ("soft", "hard", "auto"):
        raise ValueError("clamping_mode must be soft, hard, auto or None, got "
                         f"{clamping_mode}")
    if isinstance(inpt, BoundingBoxes):
        if format is not None or canvas_size is not None:
            raise ValueError("For bounding box tv_tensor inputs, `format` and "
                             "`canvas_size` must not be passed.")
        format, canvas_size = inpt.format, inpt.canvas_size
        if clamping_mode == "auto":
            clamping_mode = inpt.clamping_mode
    elif format is None or canvas_size is None or clamping_mode == "auto":
        raise ValueError("For pure tensor inputs, `format`, `canvas_size` and "
                         "`clamping_mode` have to be passed.")
    if clamping_mode is None:
        return inpt.clone() if hasattr(inpt, "clone") else inpt
    corners = _np.asarray(_to_numpy(convert_bounding_box_format(
        L.tensor(_np.asarray(_to_numpy(inpt), dtype=_np.float32)),
        old_format=format, new_format=BoundingBoxFormat.XYXY)), dtype=_np.float32)
    corners = corners.copy()
    corners[..., 0::2] = _np.clip(corners[..., 0::2], 0, canvas_size[1])
    corners[..., 1::2] = _np.clip(corners[..., 1::2], 0, canvas_size[0])
    back = convert_bounding_box_format(
        L.tensor(_np.ascontiguousarray(corners)),
        old_format=BoundingBoxFormat.XYXY, new_format=format)
    if isinstance(inpt, BoundingBoxes):
        return BoundingBoxes.wrap(back, inpt)
    return back


def clamp_keypoints(inpt, canvas_size=None):
    """Points pushed back inside the canvas. **No mode** — a point has no area to lose."""
    L = _backend()
    if isinstance(inpt, KeyPoints):
        if canvas_size is not None:
            raise ValueError("For keypoints tv_tensor inputs, `canvas_size` must not "
                             "be passed.")
        canvas_size = inpt.canvas_size
    elif canvas_size is None:
        raise ValueError("For pure tensor inputs, `canvas_size` has to be passed.")
    points = _np.asarray(_to_numpy(inpt), dtype=_np.float32).copy()
    points[..., 0] = _np.clip(points[..., 0], 0, canvas_size[1] - 1)
    points[..., 1] = _np.clip(points[..., 1], 0, canvas_size[0] - 1)
    out = L.tensor(_np.ascontiguousarray(points))
    return KeyPoints.wrap(out, inpt) if isinstance(inpt, KeyPoints) else out

# ── ops: sampling a feature map at coordinates that are not integers ─────────────
#
# **Eleven names waited on one piece of arithmetic.** Their rows read *it crops from a
# feature map. A feature map comes from a model* — which is true of where the tensor
# came from and not of what the function needs, the sixth time this table has been
# caught saying that. What they actually need is bilinear sampling at fractional
# coordinates, and that is written here once.
#
# **It is written in the backend's own operations rather than in numpy**, unlike the box
# geometry above it. Those take four numbers a box and hand back four numbers; these sit
# inside a network, so a gradient has to flow back through them to the feature map. The
# indexing below is `L`'s, which means autograd sees it.


def _both(first, second):
    """`first and second`, elementwise, **without `&`.**

    The binding's tensor has no `__and__` — measured, not assumed: sixteen golden cases
    went red under `borch_webgpu` with *unsupported operand type(s) for `&`* while the
    same cases passed on numpy and in the browser. Two of the three implementations
    answering is not this table's standard.

    `logical_and` is on all three, so it is what these use.
    """
    return _backend().logical_and(first, second)

def _roi_boxes(boxes, L):
    """`(k, 5)` — a batch index and four coordinates — from either form torchvision
    takes.

    **A list of `(l, 4)` tensors is one per image**, and the position in the list is the
    batch index. Concatenating them without writing that index down loses which picture
    each box belongs to, and every box then samples image zero — which is an answer of
    the right shape for a batch of one.
    """
    if isinstance(boxes, (list, tuple)):
        rows = []
        for index, one in enumerate(boxes):
            one = one if _is_tensor(one) else L.tensor(_np.asarray(one, _np.float32))
            if one.shape[1] != 4:
                raise ValueError("The shape of the tensor in the boxes list is not "
                                 "correct as List[Tensor[L, 4]]")
            rows.append(L.cat([L.full_like(one[:, :1], float(index)), one], dim=1))
        return L.cat(rows, dim=0) if rows else L.zeros((0, 5))
    boxes = boxes if _is_tensor(boxes) else L.tensor(_np.asarray(boxes, _np.float32))
    if boxes.shape[1] != 5:
        raise ValueError("The boxes tensor shape is not correct as Tensor[K, 5]")
    return boxes


def _is_tensor(value):
    return not isinstance(value, (_np.ndarray, list, tuple))


def _bilinear_sample(picture, batch_index, y, x):
    """The four neighbours of each `(y, x)`, weighted by how near they are.

    `picture` is `(n, c, h, w)`, `y` is `(k, ph, iy)` and `x` is `(k, pw, ix)`; what
    comes back is `(k, c, ph, pw, iy, ix)` — every sample of every bin of every box.

    Three things here each return a picture rather than an error:

    - **A sample outside the map contributes zero, and is still counted.** Not the
      clamped edge value — the kernel's guard is `y < -1 or y > height`, and the
      divisor does not shrink. torchvision's own Python reference *does* clamp instead,
      which is the trap: written from that reference, a box hanging off the left edge
      comes back with the edge column smeared across it, and the numbers are the right
      size. Measured against the compiled op, a half-outside box was exactly twice
      torchvision's answer — one real sample and one zero, averaged over two.
    - **A coordinate below zero is then clamped, not wrapped.** A negative index in
      Python reads from the far edge, so the sample would come from the opposite side
      of the image and be a number.
    - **The last row and column have no neighbour above them**, so `y_high` is held at
      `h - 1` and `y` is snapped there with it. Left alone, `y_low + 1` reads past the
      end; snapped without also moving `y`, the weights are computed against a
      neighbour that is not the one being read.
    - **The weights are an outer product**, `hy * hx` and the rest — the two axes are
      independent, and multiplying the wrong pair transposes the interpolation.
    """
    L = _backend()
    _, channels, height, width = picture.shape
    inside_y = _both(y >= -1.0, y <= height)
    inside_x = _both(x >= -1.0, x <= width)
    y = y.clamp(min=0)
    x = x.clamp(min=0)
    y_low = y.floor()
    x_low = x.floor()
    y_high = L.where(y_low >= height - 1, L.full_like(y_low, height - 1.0), y_low + 1)
    y_low = L.where(y_low >= height - 1, L.full_like(y_low, height - 1.0), y_low)
    y = L.where(y_low >= height - 1, y_low, y)
    x_high = L.where(x_low >= width - 1, L.full_like(x_low, width - 1.0), x_low + 1)
    x_low = L.where(x_low >= width - 1, L.full_like(x_low, width - 1.0), x_low)
    x = L.where(x_low >= width - 1, x_low, x)

    low_y, high_y = y - y_low, 1.0 - (y - y_low)
    low_x, high_x = x - x_low, 1.0 - (x - x_low)

    rows = L.arange(channels).long()

    def at(row, column):
        return picture[
            batch_index[:, None, None, None, None, None],
            rows[None, :, None, None, None, None],
            row.long()[:, None, :, None, :, None],
            column.long()[:, None, None, :, None, :]]

    def outer(down, across):
        return (down[:, None, :, None, :, None]
                * across[:, None, None, :, None, :])

    value = (outer(high_y, high_x) * at(y_low, x_low)
             + outer(high_y, low_x) * at(y_low, x_high)
             + outer(low_y, high_x) * at(y_high, x_low)
             + outer(low_y, low_x) * at(y_high, x_high))
    keep = _both(inside_y[:, None, :, None, :, None],
                 inside_x[:, None, None, :, None, :])
    return L.where(keep, value, L.zeros_like(value))


def roi_align(input, boxes, output_size, spatial_scale=1.0, sampling_ratio=-1,
              aligned=False):
    """Crop each box out of the feature map at a fixed size, **without rounding it.**

    <https://arxiv.org/abs/1703.06870>

    That is the whole difference from `roi_pool` next door, and it is what the paper is
    named for: a box's edges land between pixels, and rounding them to the grid moves
    the crop by up to half a cell — which is nothing on a classification map and is the
    difference between a mask that lines up and one that does not.

    Three arguments each change the answer quietly:

    - **`spatial_scale` is how much smaller the map is than the image.** The boxes are
      in image coordinates and the map is not; a stride-16 backbone wants `1/16`.
      Leaving it at 1 samples the top-left sixteenth of everything.
    - **`sampling_ratio=-1` means "as many samples as the bin is wide"**, computed per
      box, rather than a fixed number. A fixed one over-samples small boxes and
      under-samples large ones, and every value is still a plausible average.
    - **`aligned=True` shifts by half a pixel** and stops clamping the box to a minimum
      size of one. It is the correction to the original implementation and it moves
      every output; the default is `False` because that is what the released weights
      were trained with.
    """
    L = _backend()
    height, width = int(input.shape[2]), int(input.shape[3])
    rois = _roi_boxes(boxes, L)
    pooled_h, pooled_w = _pair(output_size, "output_size")
    batch_index = rois[:, 0].long()

    offset = 0.5 if aligned else 0.0
    start_h = rois[:, 2] * spatial_scale - offset
    start_w = rois[:, 1] * spatial_scale - offset
    roi_h = rois[:, 4] * spatial_scale - offset - start_h
    roi_w = rois[:, 3] * spatial_scale - offset - start_w
    if not aligned:
        roi_h = roi_h.clamp(min=1.0)
        roi_w = roi_w.clamp(min=1.0)
    bin_h, bin_w = roi_h / pooled_h, roi_w / pooled_w

    if sampling_ratio > 0:
        grid_h = grid_w = float(sampling_ratio)
        count = max(sampling_ratio * sampling_ratio, 1)
        steps_y = L.arange(sampling_ratio).float()
        steps_x = L.arange(sampling_ratio).float()
        mask_y = mask_x = None
    else:
        # **The number of samples depends on the box**, so every box is given room for
        # the largest possible grid and the surplus is masked out. That is torchvision's
        # own way round it: the alternative is a loop over boxes, and the mask is what
        # keeps this one expression.
        grid_h, grid_w = (roi_h / pooled_h).ceil(), (roi_w / pooled_w).ceil()
        count = (grid_h * grid_w).clamp(min=1)
        # **Room for the largest grid any box asks for**, not for the map's size.
        # torchvision's Python reference uses `arange(height)` and gets away with it
        # because a box is rarely taller than the map it is read from — but
        # `MultiScaleRoIAlign` sends a 200-pixel box to a 4×4 level, where the adaptive
        # grid wants more samples than there are rows and the surplus is silently
        # dropped. The compiled kernel loops to `roi_bin_grid_h` and has no such cap.
        rows = max(height, int(_as_number(grid_h.max())) if int(grid_h.shape[0]) else 0)
        columns = max(width, int(_as_number(grid_w.max())) if int(grid_w.shape[0]) else 0)
        steps_y = L.arange(rows).float()
        steps_x = L.arange(columns).float()
        mask_y = steps_y[None, :] < grid_h[:, None]
        mask_x = steps_x[None, :] < grid_w[:, None]

    def spread(one):
        return one[:, None, None]

    y = (spread(start_h) + L.arange(pooled_h).float()[None, :, None] * spread(bin_h)
         + (steps_y[None, None, :] + 0.5) * spread(bin_h / grid_h))
    x = (spread(start_w) + L.arange(pooled_w).float()[None, :, None] * spread(bin_w)
         + (steps_x[None, None, :] + 0.5) * spread(bin_w / grid_w))

    value = _bilinear_sample(input, batch_index, y, x)
    if mask_y is not None:
        zero = L.zeros_like(value)
        value = L.where(mask_y[:, None, None, None, :, None], value, zero)
        value = L.where(mask_x[:, None, None, None, None, :], value, zero)
    out = value.sum(-1).sum(-1)
    if sampling_ratio > 0:
        return out / count
    return out / count[:, None, None, None]


def roi_pool(input, boxes, output_size, spatial_scale=1.0):
    """The **maximum** in each bin, with the box rounded to the grid.

    <https://arxiv.org/abs/1504.08083>

    Faster R-CNN's pooling, and the thing `roi_align` was written to replace: the box's
    edges are rounded and so are the bin boundaries, which moves the crop by up to a
    cell twice over.

    **An empty bin answers zero rather than negative infinity.** A box small enough that
    a bin covers no cell at all is real — that is what happens to a distant object on a
    stride-32 map — and a maximum over nothing has no value to give. Zero is
    torchvision's answer and it is what lets the batch stack.
    """
    L = _backend()
    rois = _roi_boxes(boxes, L)
    pooled_h, pooled_w = _pair(output_size, "output_size")
    height, width = int(input.shape[2]), int(input.shape[3])
    out = []
    for index in range(int(rois.shape[0])):
        row = rois[index]
        which = int(_as_number(row[0]))
        # **Rounded, and rounded again per bin** — `round` and not `floor`, which is
        # what the kernel does and what makes a box at `x.5` land one cell to the right.
        #
        # And **not Python's `round`**, which goes to the nearest even: `round(0.5)` is
        # 0 there and 1 in C. A `spatial_scale` of 0.5 puts every odd coordinate on a
        # half, so half the boxes of a stride-2 map land one cell left of torchvision's
        # — measured, and invisible at `spatial_scale=1` where nothing is ever on a half.
        left = _c_round(float(_as_number(row[1])) * spatial_scale)
        top = _c_round(float(_as_number(row[2])) * spatial_scale)
        right = _c_round(float(_as_number(row[3])) * spatial_scale)
        bottom = _c_round(float(_as_number(row[4])) * spatial_scale)
        box_h = max(bottom - top + 1, 1)
        box_w = max(right - left + 1, 1)
        cells = []
        for ph in range(pooled_h):
            row_cells = []
            for pw in range(pooled_w):
                y0 = min(max(top + int(_math.floor(ph * box_h / pooled_h)), 0), height)
                y1 = min(max(top + int(_math.ceil((ph + 1) * box_h / pooled_h)), 0),
                         height)
                x0 = min(max(left + int(_math.floor(pw * box_w / pooled_w)), 0), width)
                x1 = min(max(left + int(_math.ceil((pw + 1) * box_w / pooled_w)), 0),
                         width)
                if y1 <= y0 or x1 <= x0:
                    row_cells.append(L.zeros_like(input[which, :, 0, 0]))
                else:
                    row_cells.append(
                        input[which, :, y0:y1, x0:x1].reshape(
                            int(input.shape[1]), -1).max(dim=1).values)
            cells.append(L.stack(row_cells, dim=1))
        out.append(L.stack(cells, dim=1))
    if not out:
        return L.zeros((0, int(input.shape[1]), pooled_h, pooled_w))
    return L.stack(out, dim=0)


def _ps_channels(input, pooled_h, pooled_w):
    """How many channels come out, and the refusal when it does not divide.

    **The channel axis is the position.** A position-sensitive map carries one bank of
    channels per bin — `pooled_h * pooled_w` banks — and bin `(i, j)` reads only from
    bank `i * pooled_w + j`. So the input has `out * ph * pw` channels and the output
    has `out`, and a map whose channels do not divide is not this kind of map at all.
    """
    channels = int(input.shape[1])
    banks = pooled_h * pooled_w
    if channels % banks != 0:
        raise ValueError("input channels must be a multiple of pooling height * "
                         "pooling width")
    return channels // banks


def ps_roi_align(input, boxes, output_size, spatial_scale=1.0, sampling_ratio=-1):
    """`roi_align`, **but each bin reads its own bank of channels.**

    <https://arxiv.org/abs/1605.06409>

    R-FCN's pooling. The idea is that the convolutions before it produce a map that
    already knows *where* in the object each channel is looking, so the pooling does not
    have to be followed by a fully connected layer to recover position — bin `(i, j)`
    takes bank `i * pooled_w + j` and the position is in the channel index.

    **It has no `aligned` flag, and the correction is on.** `roi_align` grew a flag and
    defaults it to `False`; this one always subtracts the half pixel and never clamps
    the box to a minimum size of one. So the absent argument does not mean the absent
    behaviour — written the other way round, from *no flag, so no correction*, every
    value is out by half a cell and looks like a plausible pooling. That sentence was
    written here before it was measured, and measuring is what changed it.
    """
    L = _backend()
    rois = _roi_boxes(boxes, L)
    pooled_h, pooled_w = _pair(output_size, "output_size")
    out_channels = _ps_channels(input, pooled_h, pooled_w)
    height, width = int(input.shape[2]), int(input.shape[3])
    batch_index = rois[:, 0].long()

    start_h = rois[:, 2] * spatial_scale - 0.5
    start_w = rois[:, 1] * spatial_scale - 0.5
    roi_h = rois[:, 4] * spatial_scale - 0.5 - start_h
    roi_w = rois[:, 3] * spatial_scale - 0.5 - start_w
    bin_h, bin_w = roi_h / pooled_h, roi_w / pooled_w

    if sampling_ratio > 0:
        grid_h = grid_w = float(sampling_ratio)
        steps_y = steps_x = L.arange(sampling_ratio).float()
        mask_y = mask_x = None
        count = float(max(sampling_ratio * sampling_ratio, 1))
    else:
        grid_h, grid_w = bin_h.ceil(), bin_w.ceil()
        count = (grid_h * grid_w).clamp(min=1)
        steps_y, steps_x = L.arange(height).float(), L.arange(width).float()
        mask_y = steps_y[None, :] < grid_h[:, None]
        mask_x = steps_x[None, :] < grid_w[:, None]

    def spread(one):
        return one[:, None, None]

    y = (spread(start_h) + L.arange(pooled_h).float()[None, :, None] * spread(bin_h)
         + (steps_y[None, None, :] + 0.5) * spread(bin_h / grid_h))
    x = (spread(start_w) + L.arange(pooled_w).float()[None, :, None] * spread(bin_w)
         + (steps_x[None, None, :] + 0.5) * spread(bin_w / grid_w))

    value = _bilinear_sample(input, batch_index, y, x)   # (k, c, ph, pw, iy, ix)
    if mask_y is not None:
        zero = L.zeros_like(value)
        value = L.where(mask_y[:, None, None, None, :, None], value, zero)
        value = L.where(mask_x[:, None, None, None, None, :], value, zero)
    pooled = value.sum(-1).sum(-1)
    pooled = pooled / (count if sampling_ratio > 0 else count[:, None, None, None])
    return _ps_pick(pooled, out_channels, pooled_h, pooled_w, L)


def _ps_pick(pooled, out_channels, pooled_h, pooled_w, L):
    """`(k, c, ph, pw)` down to `(k, out, ph, pw)` by taking each bin's own bank.

    **The output channel is the slow axis and the bin is the fast one**:
    `c * pooled_h * pooled_w + (i * pooled_w + j)`. The other order — banks of
    `out_channels` — gives an output of exactly the right shape built from the wrong
    channels, which trains and is not R-FCN, and it is what this function did until it
    was compared. The comment describing the trap was written above the code that fell
    into it.
    """
    banks = pooled_h * pooled_w
    rows = []
    for i in range(pooled_h):
        columns = []
        for j in range(pooled_w):
            picked = [pooled[:, c * banks + i * pooled_w + j, i, j]
                      for c in range(out_channels)]
            columns.append(L.stack(picked, dim=1))
        rows.append(L.stack(columns, dim=2))
    return L.stack(rows, dim=2)


def ps_roi_pool(input, boxes, output_size, spatial_scale=1.0):
    """`roi_pool`'s grid with `ps_roi_align`'s channel banks — and **an average, not a
    maximum.**

    <https://arxiv.org/abs/1605.06409>

    That is the one thing here that reads like a typo and is not: `roi_pool` next door
    takes the largest value in each bin and this takes the mean. A maximum over a
    position-sensitive bank would pick the single most confident position and throw away
    the vote, which is the opposite of what the bank is for.
    """
    L = _backend()
    rois = _roi_boxes(boxes, L)
    pooled_h, pooled_w = _pair(output_size, "output_size")
    out_channels = _ps_channels(input, pooled_h, pooled_w)
    height, width = int(input.shape[2]), int(input.shape[3])
    out = []
    for index in range(int(rois.shape[0])):
        row = rois[index]
        which = int(_as_number(row[0]))
        left = _c_round(float(_as_number(row[1])) * spatial_scale)
        top = _c_round(float(_as_number(row[2])) * spatial_scale)
        right = _c_round(float(_as_number(row[3])) * spatial_scale)
        bottom = _c_round(float(_as_number(row[4])) * spatial_scale)
        box_h = max(bottom - top, 0.1)
        box_w = max(right - left, 0.1)
        cells = []
        for ph in range(pooled_h):
            row_cells = []
            for pw in range(pooled_w):
                # **Two things here differ from `roi_pool` next door**, and both only
                # show on a box that runs off the map:
                #
                # - the rounding takes the **start inside it** — `floor(ph * bin +
                #   start)` rather than `start + floor(ph * bin)`. On a fractional start
                #   the two disagree by a cell.
                # - the clip is to **`height - 1`**, not `height`. `roi_pool` clips to
                #   the size and this clips to the last index, so a bin reaching the
                #   bottom edge covers one row fewer.
                #
                # Written from the neighbour, a box hanging off the bottom-right comes
                # back with numbers that are averages of the right kind over the wrong
                # window — measured against torchvision, 2.31 where it says 1.90.
                y0 = min(max(int(_math.floor(ph * box_h / pooled_h + top)), 0),
                         height - 1)
                y1 = min(max(int(_math.ceil((ph + 1) * box_h / pooled_h + top)), 0),
                         height - 1)
                x0 = min(max(int(_math.floor(pw * box_w / pooled_w + left)), 0),
                         width - 1)
                x1 = min(max(int(_math.ceil((pw + 1) * box_w / pooled_w + left)), 0),
                         width - 1)
                # As `_ps_pick` — **the output channel is the slow axis**, so the
                # channels for one bin are `pooled_h * pooled_w` apart rather than
                # adjacent.
                banks = pooled_h * pooled_w
                which_c = [c * banks + ph * pooled_w + pw for c in range(out_channels)]
                if y1 <= y0 or x1 <= x0:
                    row_cells.append(L.zeros_like(input[which, :out_channels, 0, 0]))
                else:
                    row_cells.append(L.stack(
                        [input[which, c, y0:y1, x0:x1].reshape(-1).mean()
                         for c in which_c], dim=0))
            cells.append(L.stack(row_cells, dim=1))
        out.append(L.stack(cells, dim=1))
    if not out:
        return L.zeros((0, out_channels, pooled_h, pooled_w))
    return L.stack(out, dim=0)

def _deform_sample(picture, y, x):
    """`picture[b, :, y, x]` at fractional `(y, x)`, **zero outside the map.**

    `y` and `x` are `(b, oh, ow)`; what comes back is `(b, c, oh, ow)`.

    **It is not `_bilinear_sample`'s rule with different bounds — it is a different
    rule.** That one clamps the coordinate to the edge and reads four real neighbours;
    this one keeps the coordinate and **drops each of the four corners separately** when
    that corner is off the map. At `y = -0.5` the two disagree completely: clamped, the
    sample is read at rows 0 and 1; here, the row above does not exist, so it
    contributes nothing and the row below carries half its weight.

    Written with a clamp, the whole thing is right in the middle of the map and wrong
    along every border — which is exactly where a deformable convolution spends its
    offsets, and the reason it was measured against the compiled op rather than shared
    with the sampler next door.

    The outer guard is the kernel's own: a coordinate at or past the edge by a whole
    pixel contributes nothing at all, whichever corners would have been valid.
    """
    L = _backend()
    _, channels, height, width = picture.shape
    inside = _both(_both(y > -1.0, y < height), _both(x > -1.0, x < width))
    y_low, x_low = y.floor(), x.floor()
    y_high, x_high = y_low + 1, x_low + 1
    ly, lx = y - y_low, x - x_low
    hy, hx = 1.0 - ly, 1.0 - lx

    batch = L.arange(int(picture.shape[0])).long()[:, None, None, None]
    rows = L.arange(channels).long()[None, :, None, None]

    def at(down, across, low_ok, high_ok):
        """One corner, read from a clamped index and then **thrown away if that index
        was not the one asked for.** The clamp is only so the gather has somewhere to
        land; the mask is what decides."""
        safe_down = down.clamp(min=0, max=height - 1)
        safe_across = across.clamp(min=0, max=width - 1)
        value = picture[batch, rows,
                        safe_down.long()[:, None, :, :],
                        safe_across.long()[:, None, :, :]]
        keep = _both(low_ok, high_ok)[:, None, :, :]
        return L.where(keep, value, L.zeros_like(value))

    def weigh(down, across):
        return (down * across)[:, None, :, :]

    low_y_ok, high_y_ok = y_low >= 0, y_high <= height - 1
    low_x_ok, high_x_ok = x_low >= 0, x_high <= width - 1

    value = (weigh(hy, hx) * at(y_low, x_low, low_y_ok, low_x_ok)
             + weigh(hy, lx) * at(y_low, x_high, low_y_ok, high_x_ok)
             + weigh(ly, hx) * at(y_high, x_low, high_y_ok, low_x_ok)
             + weigh(ly, lx) * at(y_high, x_high, high_y_ok, high_x_ok))
    return L.where(inside[:, None, :, :], value, L.zeros_like(value))


def deform_conv2d(input, offset, weight, bias=None, stride=(1, 1), padding=(0, 0),
                  dilation=(1, 1), mask=None):
    """A convolution whose sampling positions are **learned**.

    <https://arxiv.org/abs/1703.06211> — and v2, with `mask`,
    <https://arxiv.org/abs/1811.11168>

    An ordinary convolution reads a fixed grid: nine positions in a 3×3, always the same
    nine relative to the output pixel. This adds a learned displacement to each of them,
    produced by another convolution over the same input, so the receptive field bends
    with the content. `mask` is v2's addition — a learned weight per position as well as
    a learned place, which lets the layer turn a tap off rather than only move it.

    **The gradient runs back through the positions, not only the values.** That is the
    whole difficulty and the reason this could not be built on the RoI sampler: there
    the coordinates come from boxes and are given, here they come from a tensor the
    network produced, so `∂out/∂offset` has to exist. It does because the bilinear
    weights are arithmetic on the coordinate — `1 - frac` and `frac` — and the backend
    differentiates them like anything else.

    Three shapes have to line up and each mismatch is silent in a different way:

    - **The offset has `2 * groups * kh * kw` channels**, `y` before `x` for each
      kernel position. Swapping the pair transposes every displacement, which is a
      layer that trains and reads the wrong way round.
    - **`groups` is inferred, not passed.** The weight's second dimension against the
      input's channels gives the weight groups; the offset's channel count against
      `2 * kh * kw` gives the offset groups. The two are allowed to differ — one offset
      field can steer several groups of filters.
    - **The kernel's own dilation is applied before the offset**, so a displacement of
      zero leaves an ordinary dilated convolution. Adding it after makes the offsets
      mean something different at every dilation.
    """
    L = _backend()
    stride_h, stride_w = _pair(stride, "stride")
    pad_h, pad_w = _pair(padding, "padding")
    dil_h, dil_w = _pair(dilation, "dilation")
    batch, in_channels, height, width = [int(one) for one in input.shape]
    out_channels = int(weight.shape[0])
    group_channels = int(weight.shape[1])
    kernel_h, kernel_w = int(weight.shape[2]), int(weight.shape[3])

    offset_groups = int(offset.shape[1]) // (2 * kernel_h * kernel_w)
    if offset_groups == 0:
        raise RuntimeError(
            "the shape of the offset tensor at dimension 1 is not valid. It should "
            "be a multiple of 2 * weight.size[2] * weight.size[3].\n"
            f"Got offset.shape[1]={int(offset.shape[1])}, while 2 * weight.size[2] * "
            f"weight.size[3]={2 * kernel_h * kernel_w}")
    weight_groups = in_channels // group_channels

    out_h = (height + 2 * pad_h - (dil_h * (kernel_h - 1) + 1)) // stride_h + 1
    out_w = (width + 2 * pad_w - (dil_w * (kernel_w - 1) + 1)) // stride_w + 1

    base_y = (L.arange(out_h).float() * stride_h - pad_h)[None, :, None]
    base_x = (L.arange(out_w).float() * stride_w - pad_w)[None, None, :]

    total = None
    for p in range(kernel_h):
        for q in range(kernel_w):
            which = p * kernel_w + q
            columns = []
            for group in range(offset_groups):
                start = (group * kernel_h * kernel_w + which) * 2
                dy = offset[:, start]
                dx = offset[:, start + 1]
                y = base_y + p * dil_h + dy
                x = base_x + q * dil_w + dx
                per_group = in_channels // offset_groups
                taken = _deform_sample(
                    input[:, group * per_group:(group + 1) * per_group], y, x)
                if mask is not None:
                    # **The mask is per offset group as well as per kernel position.**
                    # Its channels run `group * kh * kw + which`, the same layout as the
                    # offsets without the pair — and read as `which` alone, every group
                    # gets the first group's weights. That is right whenever there is one
                    # group, which is what a fixture written with `groups=1` shows, and
                    # it survived `offset_groups=2` on its own because the value it got
                    # wrong was multiplied by an offset it also got wrong.
                    taken = taken * mask[:, group * kernel_h * kernel_w + which][:, None]
                columns.append(taken)
            sampled = L.cat(columns, dim=1) if len(columns) > 1 else columns[0]
            # **The weight groups are folded here**, not in the sampling: filter `co`
            # sees only the channels of its own group, and `co // (out / groups)` is
            # which one. Letting every filter see every channel gives an answer of the
            # right shape from `groups` times as many terms.
            pieces = []
            per_out = out_channels // weight_groups
            for group in range(weight_groups):
                block = sampled[:, group * group_channels:(group + 1) * group_channels]
                taps = weight[group * per_out:(group + 1) * per_out, :, p, q]
                pieces.append(
                    (block[:, None] * taps[None, :, :, None, None]).sum(dim=2))
            step = L.cat(pieces, dim=1) if len(pieces) > 1 else pieces[0]
            total = step if total is None else total + step
    if bias is not None:
        total = total + bias[None, :, None, None]
    return total

def _ordered(pairs):
    """An `OrderedDict`, which is what torchvision hands back and what a caller iterates
    in order. A plain dict keeps insertion order too, and is not the same type — a
    detector that checks is checking for the type."""
    return _collections.OrderedDict(pairs)


def _kaiming_uniform(weight, gain):
    """`nn.init.kaiming_uniform_` for the one case this file needs.

    **`nn.init` is not in the core**, so the arithmetic is here: the bound is
    `√3 · gain / √fan_in`, and `fan_in` for a convolution is the input channels times
    the kernel's area — not the number of weights. Using the whole count divides by the
    output channels as well and gives a bound that is too small by their square root.
    """
    shape = tuple(int(one) for one in weight.shape)
    fan_in = shape[1]
    for one in shape[2:]:
        fan_in *= one
    bound = _math.sqrt(3.0) * gain / _math.sqrt(fan_in)
    L = _backend()
    with L.no_grad():
        weight.copy_(L.empty(shape).uniform_(-bound, bound))


def _box_area_tensor(boxes):
    """`box_area` without leaving the backend, because the level choice is arithmetic on
    what comes back and `box_area` above hands over numpy."""
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def _infer_scale(feature, original_size):
    """A map's scale as a **power of two**, read off its size against the image's.

    Rounded in log space rather than taken as the ratio: a 800×800 image gives a 50×50
    map at stride 16, and 50/800 is exactly 1/16 — but 801 gives 51, and 51/801 is not a
    power of two at all. Rounding the exponent is what makes both answer 1/16.
    """
    size = [int(one) for one in feature.shape[-2:]]
    scales = []
    for one, whole in zip(size, original_size):
        scales.append(2.0 ** round(_math.log2(float(one) / float(whole))))
    return scales[0]

def _c_round(value):
    """C's `round` — to the nearest, halves away from zero.

    Python's rounds halves to the nearest **even**, so `round(0.5)` is 0 here and 1 in
    the kernel this is matching. It only shows where a coordinate lands exactly on a
    half, which `spatial_scale=1` never does and `0.5` does for every odd number.
    """
    return int(_math.copysign(_math.floor(abs(value) + 0.5), value))


def _as_number(value):
    """One element of a tensor as a Python number, whichever backend it came from."""
    return float(_np.asarray(value.numpy() if hasattr(value, "numpy") else value))

# ── ops: the structured dropouts ─────────────────────────────────────────────────
#
# **These six were declined for being a backbone's.** *Structured dropout for
# convolutional backbones*, *it drops whole residual blocks — it needs blocks*: true of
# what they are for, and neither about whether they can exist. Each takes a feature map
# and a probability.
#
# It is the third pass over the same table. The one-line reason went first, then the
# per-kind reasons that replaced it, and then the sentence that replaced *those* —
# *what is left below needs something that is not here* — which was written in the
# commit that took the layers and is wrong here too. **An over-wide sentence removed by
# writing a narrower over-wide sentence** is the failure this file exists to catch,
# and it has now happened three times in a row while catching it.

def stochastic_depth(input, p, mode, training=True):
    """Whole residual branches dropped at random.

    <https://arxiv.org/abs/1603.09382>

    **`mode` decides what a coin is tossed for**: `batch` drops the branch for the whole
    batch at once, `row` drops it per example. The two differ only in the shape of the
    noise, and getting it wrong gives a network that trains at a different effective
    depth — nothing raises and the curve moves.

    **The survivors are divided by the survival rate**, so the expected value is
    unchanged and evaluation needs no rescaling. Dropping without it makes every layer
    quieter than the next one expects.
    """
    if p < 0.0 or p > 1.0:
        raise ValueError(f"drop probability has to be between 0 and 1, but got {p}")
    if mode not in ["batch", "row"]:
        raise ValueError(f"mode has to be either 'batch' or 'row', but got {mode}")
    if not training or p == 0.0:
        return input
    L = _backend()
    survival = 1.0 - p
    shape = ([input.shape[0]] + [1] * (input.ndim - 1) if mode == "row"
             else [1] * input.ndim)
    noise = L.empty(shape, dtype=input.dtype).bernoulli_(survival)
    if survival > 0.0:
        noise = noise / survival
    return input * noise


def _drop_block(input, p, block_size, inplace, eps, training, spatial):
    """DropBlock in `spatial` dimensions — **contiguous regions, not single pixels.**

    <https://arxiv.org/abs/1810.12890>

    Dropping pixels one at a time does little to a convolutional map, because the
    neighbours carry the same information. So a seed is drawn per position and then
    **grown to a block by a max pool**, which is the whole trick: the pool spreads each
    surviving one over its window, and one minus that is the mask.

    Two numbers here each give a mask that looks plausible:

    - **The seeds are drawn on a smaller grid**, `H - block + 1` on each axis, because a
      seed near the edge would grow a block that hangs off it. Drawing on the full grid
      drops more than `p` asks for, and only at the edges.
    - **`gamma` is not `p`.** It is `p` scaled by how many positions a block covers and
      how many seeds there are to draw, so that the fraction actually dropped comes out
      at `p`. Using `p` directly drops roughly `block ** spatial` times too much.

    What is kept is then divided by the fraction kept, as in dropout.
    """
    if p < 0.0 or p > 1.0:
        raise ValueError(f"drop probability has to be between 0 and 1, but got {p}.")
    if input.ndim != spatial + 2:
        raise ValueError(f"input should be {spatial + 2} dimensional. Got "
                         f"{input.ndim} dimensions.")
    if not training or p == 0.0:
        return input
    L = _backend()
    sizes = list(input.shape)[2:]
    block_size = min(block_size, *sizes)
    if block_size % 2 == 0:
        raise ValueError(f"block size should be odd. Got {block_size} which is even.")
    seeds = [one - block_size + 1 for one in sizes]
    total = 1
    for one in sizes:
        total *= one
    positions = 1
    for one in seeds:
        positions *= one
    gamma = (p * total) / ((block_size ** spatial) * positions)
    noise = L.empty([input.shape[0], input.shape[1]] + seeds,
                    dtype=input.dtype).bernoulli_(gamma)
    noise = L.nn.functional.pad(noise, [block_size // 2] * (spatial * 2), value=0)
    pool = (L.nn.functional.max_pool2d if spatial == 2
            else L.nn.functional.max_pool3d)
    noise = pool(noise, stride=(1,) * spatial,
                 kernel_size=(block_size,) * spatial, padding=block_size // 2)
    noise = 1 - noise
    scale = noise.numel() / (eps + noise.sum())
    if inplace:
        return input.mul_(noise).mul_(scale)
    return input * noise * scale


def drop_block2d(input, p, block_size, inplace=False, eps=1e-06, training=True):
    """`_drop_block` over height and width."""
    return _drop_block(input, p, block_size, inplace, eps, training, 2)


def drop_block3d(input, p, block_size, inplace=False, eps=1e-06, training=True):
    """`_drop_block` over depth, height and width."""
    return _drop_block(input, p, block_size, inplace, eps, training, 3)


# ── ops: the layers, and a sentence that was about use rather than need ──────────
#
# **Twenty-eight names here were declined for what they are *for*.** The reasons read
# *it needs a model to be a block of*, *a feature map comes from a model*, *it takes a
# model's predictions* — every one true, and none of them about whether the thing can
# exist. `Conv2dNormActivation` is a convolution, a norm and an activation in a
# `Sequential`, and all three are in the core; `MLP` is linear layers and dropout;
# `FrozenBatchNorm2d` is an affine transform over four buffers.
#
# It is the same shape as `as above — a codec`, which was wrong four times before
# anyone opened the files: **a true sentence about one name, used as a reason for
# every name under it.** What was measured this time is the ingredient list.
#
# **These are the first classes here that must subclass the backend's `nn.Module`**,
# and `use(L)` can change which backend that is *after* import. Defined at module level
# they would freeze onto whichever library happened to be attached first, so they are
# built on first access and cached per backend — which is why `ops` grows a
# `__getattr__` below rather than a list of names.

_OPS_LAYERS = {}


def _ops_layers(L):
    """The layer classes for one backend, built once.

    Rebuilding them per call would make `isinstance` fail against a class the caller
    already holds, and two `Sequential`s that print the same would not be the same
    type — so the cache is keyed on the backend rather than on nothing.
    """
    made = _OPS_LAYERS.get(id(L))
    if made is not None:
        return made
    nn = L.nn

    class ConvNormActivation(nn.Sequential):
        """A convolution, a normalisation and an activation, in that order.

        **`padding=None` means "keep the size"**, computed as `(k - 1) // 2 * dilation`
        rather than left at zero — a block built with zero padding shrinks its input by
        two per layer, which is a network that trains and a resolution that quietly
        runs out.

        **`bias=None` means "only when there is no norm"**, because a normalisation
        immediately after a convolution subtracts the mean and the bias with it.
        Leaving it on costs a parameter per channel that provably does nothing, and a
        checkpoint written against torchvision's would carry a tensor this one has no
        slot for.
        """

        def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                     padding=None, groups=1, norm_layer=nn.BatchNorm2d,
                     activation_layer=nn.ReLU, dilation=1, inplace=True, bias=None,
                     conv_layer=nn.Conv2d):
            if padding is None:
                if isinstance(kernel_size, int) and isinstance(dilation, int):
                    padding = (kernel_size - 1) // 2 * dilation
                else:
                    width = (len(kernel_size)
                             if isinstance(kernel_size, (tuple, list))
                             else len(dilation))
                    sizes = _ops_ntuple(kernel_size, width)
                    spread = _ops_ntuple(dilation, width)
                    padding = tuple((sizes[i] - 1) // 2 * spread[i]
                                    for i in range(width))
            if bias is None:
                bias = norm_layer is None
            layers = [conv_layer(in_channels, out_channels, kernel_size, stride,
                                 padding, dilation=dilation, groups=groups,
                                 bias=bias)]
            if norm_layer is not None:
                layers.append(norm_layer(out_channels))
            if activation_layer is not None:
                layers.append(activation_layer(
                    **({} if inplace is None else {"inplace": inplace})))
            super().__init__(*layers)
            self.out_channels = out_channels

    class Conv2dNormActivation(ConvNormActivation):
        """`ConvNormActivation` with the convolution fixed to two dimensions.

        Its 3-D twin stays declined, and **that row is the one real refusal in this
        group**: 3-D convolution is absent from the core, which is a missing
        ingredient rather than a missing use.
        """

        def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                     padding=None, groups=1, norm_layer=nn.BatchNorm2d,
                     activation_layer=nn.ReLU, dilation=1, inplace=True, bias=None):
            super().__init__(in_channels, out_channels, kernel_size, stride, padding,
                             groups, norm_layer, activation_layer, dilation, inplace,
                             bias, nn.Conv2d)

    class SqueezeExcitation(nn.Module):
        """Squeeze-and-Excitation — **the channels weight themselves.**

        <https://arxiv.org/abs/1709.01507>

        Pool each channel to one number, run two 1×1 convolutions over that, multiply
        the input by the result. **The two activations are not the same one**: the
        inner is a rectifier and the outer squashes to `(0, 1)`, because the outer's
        output is a gain. Using the rectifier for both gives gains unbounded above —
        the block still trains and the scale drifts.
        """

        def __init__(self, input_channels, squeeze_channels, activation=nn.ReLU,
                     scale_activation=nn.Sigmoid):
            super().__init__()
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            self.fc1 = nn.Conv2d(input_channels, squeeze_channels, 1)
            self.fc2 = nn.Conv2d(squeeze_channels, input_channels, 1)
            self.activation = activation()
            self.scale_activation = scale_activation()

        def _scale(self, input):
            scale = self.avgpool(input)
            scale = self.fc1(scale)
            scale = self.activation(scale)
            scale = self.fc2(scale)
            return self.scale_activation(scale)

        def forward(self, input):
            return self._scale(input) * input

    class MLP(nn.Sequential):
        """Linear layers with an activation and dropout between them.

        **The last layer gets neither norm nor activation, and still gets dropout.**
        That asymmetry is the whole of the class: written with the loop covering every
        hidden size, the output passes through a rectifier and can never be negative —
        a head that trains, on half the range.
        """

        def __init__(self, in_channels, hidden_channels, norm_layer=None,
                     activation_layer=nn.ReLU, inplace=None, bias=True, dropout=0.0):
            params = {} if inplace is None else {"inplace": inplace}
            layers = []
            width = in_channels
            for size in hidden_channels[:-1]:
                layers.append(nn.Linear(width, size, bias=bias))
                if norm_layer is not None:
                    layers.append(norm_layer(size))
                layers.append(activation_layer(**params))
                layers.append(nn.Dropout(dropout, **params))
                width = size
            layers.append(nn.Linear(width, hidden_channels[-1], bias=bias))
            layers.append(nn.Dropout(dropout, **params))
            super().__init__(*layers)

    class Permute(nn.Module):
        """`permute` as a module, so it can sit in a `Sequential`.

        Its old row read *an `nn.Module` wrapper around `permute`; the core has the
        function* — which is a reason it is easy, not a reason it is absent.
        """

        def __init__(self, dims):
            super().__init__()
            self.dims = dims

        def forward(self, x):
            return x.permute(*self.dims)

    class FrozenBatchNorm2d(nn.Module):
        """Batch norm with the statistics fixed — **an affine transform wearing four
        buffers.**

        All four are buffers rather than parameters, and that is the point: this exists
        so a batch norm inside a pre-trained network stops moving when the network is
        fine-tuned on batches too small to estimate a mean from. `weight` and `bias`
        are buffers too, so nothing here trains.

        **Epsilon is added before the reciprocal square root**, not after. The other
        order divides by something arbitrarily small on variances near zero, and what
        comes back is finite, enormous, and shaped exactly like an activation.
        """

        def __init__(self, num_features, eps=1e-5):
            super().__init__()
            self.eps = eps
            self.register_buffer("weight", L.ones(num_features))
            self.register_buffer("bias", L.zeros(num_features))
            self.register_buffer("running_mean", L.zeros(num_features))
            self.register_buffer("running_var", L.ones(num_features))

        def forward(self, x):
            weight = self.weight.reshape(1, -1, 1, 1)
            bias = self.bias.reshape(1, -1, 1, 1)
            variance = self.running_var.reshape(1, -1, 1, 1)
            mean = self.running_mean.reshape(1, -1, 1, 1)
            scale = weight * (variance + self.eps).rsqrt()
            return x * scale + (bias - mean * scale)

        def __repr__(self):
            return f"{type(self).__name__}({self.weight.shape[0]}, eps={self.eps})"

    class DropBlock2d(nn.Module):
        """`drop_block2d` as a module. **The repr does not print `eps`**, which
        torchvision's does not either — it is a guard against dividing by zero rather
        than a setting, and printing it would invite somebody to tune it."""

        def __init__(self, p, block_size, inplace=False, eps=1e-06):
            super().__init__()
            self.p = p
            self.block_size = block_size
            self.inplace = inplace
            self.eps = eps

        def forward(self, input):
            return drop_block2d(input, self.p, self.block_size, self.inplace,
                                self.eps, self.training)

        def __repr__(self):
            return (f"{type(self).__name__}(p={self.p}, "
                    f"block_size={self.block_size}, inplace={self.inplace})")

    class DropBlock3d(DropBlock2d):
        """As `DropBlock2d`, over three spatial axes. **It subclasses the 2-D one and
        the repr comes with it** — printing the class's own name, which is why that
        line reads `type(self).__name__` rather than the literal."""

        def forward(self, input):
            return drop_block3d(input, self.p, self.block_size, self.inplace,
                                self.eps, self.training)

    class StochasticDepth(nn.Module):
        """`stochastic_depth` as a module.

        **The mode prints unquoted** — `mode=row`, not `mode='row'` — which is
        torchvision's own repr and not this file's usual rule. It is copied because the
        string is the answer, and a tidier one would differ from the library being
        matched.
        """

        def __init__(self, p, mode):
            super().__init__()
            self.p = p
            self.mode = mode

        def forward(self, input):
            return stochastic_depth(input, self.p, self.mode, self.training)

        def __repr__(self):
            return f"{type(self).__name__}(p={self.p}, mode={self.mode})"

    class RoIAlign(nn.Module):
        """`roi_align` as a module. See it for what the four arguments do."""

        def __init__(self, output_size, spatial_scale, sampling_ratio, aligned=False):
            super().__init__()
            self.output_size = output_size
            self.spatial_scale = spatial_scale
            self.sampling_ratio = sampling_ratio
            self.aligned = aligned

        def forward(self, input, rois):
            return roi_align(input, rois, self.output_size, self.spatial_scale,
                             self.sampling_ratio, self.aligned)

        def __repr__(self):
            return (f"{type(self).__name__}(output_size={self.output_size}"
                    f", spatial_scale={self.spatial_scale}"
                    f", sampling_ratio={self.sampling_ratio}"
                    f", aligned={self.aligned})")

    class RoIPool(nn.Module):
        """`roi_pool` as a module."""

        def __init__(self, output_size, spatial_scale):
            super().__init__()
            self.output_size = output_size
            self.spatial_scale = spatial_scale

        def forward(self, input, rois):
            return roi_pool(input, rois, self.output_size, self.spatial_scale)

        def __repr__(self):
            return (f"{type(self).__name__}(output_size={self.output_size}, "
                    f"spatial_scale={self.spatial_scale})")

    class PSRoIAlign(nn.Module):
        """`ps_roi_align` as a module. **No `aligned` argument**, and the correction is
        on regardless — see the function."""

        def __init__(self, output_size, spatial_scale, sampling_ratio):
            super().__init__()
            self.output_size = output_size
            self.spatial_scale = spatial_scale
            self.sampling_ratio = sampling_ratio

        def forward(self, input, rois):
            return ps_roi_align(input, rois, self.output_size, self.spatial_scale,
                                self.sampling_ratio)

        def __repr__(self):
            return (f"{type(self).__name__}(output_size={self.output_size}"
                    f", spatial_scale={self.spatial_scale}"
                    f", sampling_ratio={self.sampling_ratio})")

    class PSRoIPool(nn.Module):
        """`ps_roi_pool` as a module."""

        def __init__(self, output_size, spatial_scale):
            super().__init__()
            self.output_size = output_size
            self.spatial_scale = spatial_scale

        def forward(self, input, rois):
            return ps_roi_pool(input, rois, self.output_size, self.spatial_scale)

        def __repr__(self):
            return (f"{type(self).__name__}(output_size={self.output_size}, "
                    f"spatial_scale={self.spatial_scale})")

    class FeaturePyramidNetwork(nn.Module):
        """A pyramid of feature maps that all carry the same number of channels.

        <https://arxiv.org/abs/1612.03144>

        A backbone produces maps that get smaller and deeper together — 256 channels at
        one resolution, 2048 at a sixteenth of it. A detector wants to look at all of
        them with one head, so each is put through a 1×1 convolution to a common width
        and then **the coarse map is added into the fine one**, top down, so that the
        fine map gains what only the coarse one could see.

        Three things about the shape of it:

        - **The maps arrive in increasing depth order**, so the last one is the
          coarsest. Reversing them adds the fine into the coarse, which runs and learns
          the opposite of the point.
        - **The top-down step is nearest-neighbour interpolation to the lateral's own
          size**, not a fixed factor of two. Feature maps are not always exactly half
          their neighbour — an odd input makes them off by one — and a hard-coded factor
          gives a shape error on some inputs and a silent crop on others.
        - **There are two convolutions per level**: a 1×1 that fixes the width, and a
          3×3 afterwards that smooths the seam the addition leaves. Dropping the second
          leaves an aliased map that trains.

        **The weights are re-initialised wider than a convolution's default.**
        `kaiming_uniform_(a=1)` gives a bound of `√3 / √fan_in` where `Conv2d`'s own
        default gives `1 / √fan_in` — √3 times wider. `nn.init` is not in the core, so
        it is done here from the backend's uniform; the arithmetic is written out
        because the two bounds differ by a constant nobody would notice in a histogram.
        """

        _KAIMING_GAIN = 1.0                      # gain at a=1, for leaky_relu

        def __init__(self, in_channels_list, out_channels, extra_blocks=None,
                     norm_layer=None):
            super().__init__()
            self.inner_blocks = nn.ModuleList()
            self.layer_blocks = nn.ModuleList()
            for in_channels in in_channels_list:
                if in_channels == 0:
                    raise ValueError("in_channels=0 is currently not supported")
                self.inner_blocks.append(Conv2dNormActivation(
                    in_channels, out_channels, kernel_size=1, padding=0,
                    norm_layer=norm_layer, activation_layer=None))
                self.layer_blocks.append(Conv2dNormActivation(
                    out_channels, out_channels, kernel_size=3,
                    norm_layer=norm_layer, activation_layer=None))
            for module in self.modules():
                if isinstance(module, nn.Conv2d):
                    _kaiming_uniform(module.weight, self._KAIMING_GAIN)
                    if module.bias is not None:
                        with L.no_grad():
                            module.bias.copy_(L.zeros_like(module.bias))
            self.extra_blocks = extra_blocks

        def get_result_from_inner_blocks(self, x, idx):
            return self.inner_blocks[idx](x)

        def get_result_from_layer_blocks(self, x, idx):
            return self.layer_blocks[idx](x)

        def forward(self, x):
            names = list(x.keys())
            maps = list(x.values())
            last = self.get_result_from_inner_blocks(maps[-1], -1)
            results = [self.get_result_from_layer_blocks(last, -1)]
            for idx in range(len(maps) - 2, -1, -1):
                lateral = self.get_result_from_inner_blocks(maps[idx], idx)
                size = tuple(int(one) for one in lateral.shape[-2:])
                last = lateral + L.nn.functional.interpolate(
                    last, size=size, mode="nearest")
                results.insert(0, self.get_result_from_layer_blocks(last, idx))
            if self.extra_blocks is not None:
                results, names = self.extra_blocks(results, maps, names)
            return _ordered(zip(names, results))

    class MultiScaleRoIAlign(nn.Module):
        """`roi_align` over a pyramid, **choosing a level per box by its size.**

        <https://arxiv.org/abs/1612.03144> — equation 1.

        A small box wants the fine map and a large one wants the coarse: a 224-pixel box
        goes to level 4 and the level moves by one for every doubling of the box's
        side. `canonical_scale` and `canonical_level` are those two numbers.

        **The scales are inferred once, from the first call.** Each map's scale is read
        off as a power of two from its size against the image's, and then cached — so a
        second call with differently sized images reuses the first call's answer. That
        is torchvision's behaviour, cache and all, and it is copied rather than fixed:
        a detector calls this with one image size.

        **The boxes are in image coordinates, not map coordinates.** That is what
        `spatial_scale` is for and what makes the level choice possible at all — the box
        has one size across the whole pyramid, and only the map changes.
        """

        _CANONICAL_SCALE = 224
        _CANONICAL_LEVEL = 4
        _LEVEL_EPS = 1e-6

        def __init__(self, featmap_names, output_size, sampling_ratio, *,
                     canonical_scale=224, canonical_level=4):
            super().__init__()
            if isinstance(output_size, int):
                output_size = (output_size, output_size)
            self.featmap_names = featmap_names
            self.sampling_ratio = sampling_ratio
            self.output_size = tuple(output_size)
            self.scales = None
            self.map_levels = None
            self.canonical_scale = canonical_scale
            self.canonical_level = canonical_level

        def _levels(self, boxes, k_min, k_max):
            """Equation 1: `floor(lvl0 + log2(√area / s0))`, clamped.

            **The epsilon is added inside the floor**, not outside. A box of exactly the
            canonical size lands on an integer, and floating point puts it a hair below
            as often as above — so without it the same box goes to two different levels
            on two runs of the same detector.
            """
            areas = L.cat([_box_area_tensor(one) for one in boxes], dim=0)
            side = areas.sqrt()
            levels = (self.canonical_level
                      + (side / self.canonical_scale).log2() + self._LEVEL_EPS).floor()
            return levels.clamp(min=k_min, max=k_max).long() - k_min

        def forward(self, x, boxes, image_shapes):
            maps = [value for name, value in x.items() if name in self.featmap_names]
            if self.scales is None:
                if not image_shapes:
                    raise ValueError("images list should not be empty")
                widest = max(int(shape[0]) for shape in image_shapes)
                tallest = max(int(shape[1]) for shape in image_shapes)
                self.scales = [_infer_scale(one, (widest, tallest)) for one in maps]
                self.k_min = int(-_math.log2(self.scales[0]))
                self.k_max = int(-_math.log2(self.scales[-1]))
                self.map_levels = True
            rois = _roi_boxes(list(boxes), L)
            if len(maps) == 1:
                return roi_align(maps[0], rois, self.output_size,
                                 spatial_scale=self.scales[0],
                                 sampling_ratio=self.sampling_ratio)
            levels = self._levels(boxes, self.k_min, self.k_max)
            pieces = []
            for index in range(int(rois.shape[0])):
                level = int(_as_number(levels[index]))
                pieces.append(roi_align(
                    maps[level], rois[index:index + 1], self.output_size,
                    spatial_scale=self.scales[level],
                    sampling_ratio=self.sampling_ratio)[0])
            if not pieces:
                return L.zeros((0, int(maps[0].shape[1])) + self.output_size)
            return L.stack(pieces, dim=0)

        def __repr__(self):
            return (f"{type(self).__name__}(featmap_names={self.featmap_names}, "
                    f"output_size={self.output_size}, "
                    f"sampling_ratio={self.sampling_ratio})")

    class DeformConv2d(nn.Module):
        """`deform_conv2d` as a module, holding the weight.

        **It does not produce the offsets.** They arrive as a second argument, from
        another convolution the caller writes — which is why `forward` takes two tensors
        and three when the mask is used. A layer that made its own offsets would be a
        different and more convenient thing, and would not load a torchvision
        checkpoint.

        **`kaiming_uniform_(a=√5)` here, not `a=1`.** That is `Conv2d`'s own default and
        not the pyramid's wider one — the two initialisations differ by √3, and this one
        takes the narrower because a deformable convolution is a convolution.
        """

        _KAIMING_A = 5.0

        def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                     dilation=1, groups=1, bias=True):
            super().__init__()
            if in_channels % groups != 0:
                raise ValueError("in_channels must be divisible by groups")
            if out_channels % groups != 0:
                raise ValueError("out_channels must be divisible by groups")
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.kernel_size = _pair(kernel_size, "kernel_size")
            self.stride = _pair(stride, "stride")
            self.padding = _pair(padding, "padding")
            self.dilation = _pair(dilation, "dilation")
            self.groups = groups
            shape = (out_channels, in_channels // groups) + self.kernel_size
            self.weight = nn.Parameter(L.empty(shape))
            self.bias = nn.Parameter(L.empty((out_channels,))) if bias else None
            self.reset_parameters()

        def reset_parameters(self):
            gain = _math.sqrt(2.0 / (1.0 + self._KAIMING_A))
            _kaiming_uniform(self.weight, gain)
            if self.bias is not None:
                shape = tuple(int(one) for one in self.weight.shape)
                fan_in = shape[1]
                for one in shape[2:]:
                    fan_in *= one
                bound = 1.0 / _math.sqrt(fan_in)
                with L.no_grad():
                    self.bias.copy_(L.empty(tuple(self.bias.shape))
                                    .uniform_(-bound, bound))

        def forward(self, input, offset, mask=None):
            return deform_conv2d(input, offset, self.weight, self.bias,
                                 stride=self.stride, padding=self.padding,
                                 dilation=self.dilation, mask=mask)

        def __repr__(self):
            out = (f"{type(self).__name__}({self.in_channels}, {self.out_channels}"
                   f", kernel_size={self.kernel_size}, stride={self.stride}")
            if self.padding != (0, 0):
                out += f", padding={self.padding}"
            if self.dilation != (1, 1):
                out += f", dilation={self.dilation}"
            if self.groups != 1:
                out += f", groups={self.groups}"
            if self.bias is None:
                out += ", bias=False"
            return out + ")"

    made = {"ConvNormActivation": ConvNormActivation,
            "DeformConv2d": DeformConv2d,
            "FeaturePyramidNetwork": FeaturePyramidNetwork,
            "MultiScaleRoIAlign": MultiScaleRoIAlign,
            "RoIAlign": RoIAlign, "RoIPool": RoIPool,
            "PSRoIAlign": PSRoIAlign, "PSRoIPool": PSRoIPool,
            "DropBlock2d": DropBlock2d, "DropBlock3d": DropBlock3d,
            "StochasticDepth": StochasticDepth,
            "Conv2dNormActivation": Conv2dNormActivation,
            "SqueezeExcitation": SqueezeExcitation, "MLP": MLP,
            "Permute": Permute, "FrozenBatchNorm2d": FrozenBatchNorm2d}
    _OPS_LAYERS[id(L)] = made
    return made


def _ops_ntuple(value, width):
    """A number or a sequence, as a tuple of `width` — torchvision's `_make_ntuple`."""
    if isinstance(value, (tuple, list)):
        return tuple(value)
    return (value,) * width


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

    **And the signature is set for the same kind of reason.** `__init__` here takes
    `*args, **kwargs` because it forwards, so `inspect.signature(v2.Resize)` answered
    `(*args, **kwargs)` for all thirty-eight — that is what `help()` prints, what an
    editor completes from, and what `tests/ts_signatures.py` reads to compare against
    borch.ts. The arguments *are* v1's; nothing had said so anywhere a tool could see.

    A forwarding `__init__` is exactly the shape `__signature__` exists for. Copying
    v1's is honest rather than a claim: the call `Twin.__init__` makes is
    `base.__init__(*args, **kwargs)`, so a call v1 accepts is a call this accepts, and
    one it refuses this refuses in the same place with the same message.
    """
    class Twin(_V2Repr, base):
        def __init__(self, *args, **kwargs):
            base.__init__(self, *args, **kwargs)
            self._v2(**{name: read(self) for name, read in fields})

    Twin.__name__ = base.__name__
    Twin.__qualname__ = base.__name__
    Twin.__doc__ = (f"`transforms.v2.{base.__name__}` — {base.__name__}'s behaviour "
                    "with v2's printed surface.")
    try:
        Twin.__init__.__signature__ = _inspect.signature(base.__init__)
    except (TypeError, ValueError):
        # A base whose own `__init__` cannot be read leaves the twin as it was. Better
        # an unreadable signature than a wrong one — this is the direction an
        # instrument must not fail in.
        pass
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


_MAT_CELL, _MAT_STRUCT, _MAT_CHAR = 1, 2, 4


def _mat_matrix(body):
    """`(name, value)` for one matrix element.

    A numeric array comes back as an `ndarray`; the three other classes this reads
    come back as the Python shape that matches what they are:

    - **char** is a `str`. MATLAB stores text as an array of code units, so a reader
      that treated it as numeric returns a list of small integers that is the right
      length and means nothing.
    - **a cell array** is a `list`, because that is what a cell array is — a container
      of unrelated matrices rather than a rectangle of one dtype.
    - **a struct array** is a `list of dicts`, one per element, and a **single dict**
      when it holds one element. `StanfordCars` reads the first shape and
      `StanfordCars`'s class list reads a cell of chars.

    `None` for anything else, which is what the caller drops.
    """
    _, flags, at = _mat_element(body, 0, pad=True)
    cls = flags[0] & 0xFF
    _, dims_raw, at = _mat_element(body, at, pad=True)
    dims = list(_np.frombuffer(dims_raw, dtype="<i4"))
    _, name_raw, at = _mat_element(body, at, pad=True)
    name = name_raw.decode("ascii", "replace")
    count = int(_np.prod(dims)) if dims else 0

    if cls == _MAT_CHAR:
        kind, values, _ = _mat_element(body, at, pad=True)
        return name, _mat_text(kind, values)
    if cls == _MAT_CELL:
        cells, _ = _mat_children(body, at, count)
        return name, cells
    if cls == _MAT_STRUCT:
        return name, _mat_struct(body, at, count)
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


def _mat_text(kind, values):
    """MATLAB text as a `str`.

    **The code unit width is in the tag**, not fixed, and which one you meet depends on
    who wrote the file. Measured: `scipy.io.savemat` writes `miUTF8`, so every fixture
    in this repository takes that branch — MATLAB itself writes `miUINT16`, which is
    the branch no `.mat` written here would ever exercise. It is tested directly
    instead, because decoding sixteen-bit units as bytes gives a string with a NUL
    between every letter: printable, wrong, and the same length as the answer once
    something strips them.
    """
    if kind == 16:                              # miUTF8
        return values.decode("utf-8", "replace").rstrip("\x00")
    if kind == 18:                              # miUTF32
        return _np.frombuffer(values, dtype="<u4").tobytes().decode(
            "utf-32-le", "replace").rstrip("\x00")
    return "".join(chr(one) for one in _np.frombuffer(values, dtype="<u2")).rstrip(
        "\x00")


def _mat_children(body, at, count):
    """`count` matrix elements laid end to end, and where they stop.

    Cells and struct fields are both written this way — a bare run of `miMATRIX`
    elements with nothing between them, so the count has to come from the dimensions
    rather than from the bytes.
    """
    out = []
    for _ in range(count):
        if at + 8 > len(body):
            break
        kind, payload, at = _mat_element(body, at, pad=True)
        if kind != 14:                          # miMATRIX
            continue
        out.append(_mat_matrix(payload)[1])
    return out, at


def _mat_struct(body, at, count):
    """A struct array as a list of dicts, or one dict when it holds one element.

    **The field names are a fixed-width block**, not a list of strings: one length
    (usually 32) and then that many bytes per name, NUL-padded. Splitting the block on
    NUL instead would give the same names on a file whose names are all shorter than
    the width and lose every name that fills it exactly.

    **The values run element by element, all fields of one before the next.** Reading
    them field by field instead transposes the whole array: on a struct of `n` elements
    with `f` fields every value lands on the wrong element, and on `n == 1` — which is
    most files anyone tests with — the two orders agree.
    """
    _, width_raw, at = _mat_element(body, at, pad=True)
    width = int(_np.frombuffer(width_raw, dtype="<i4")[0])
    _, names_raw, at = _mat_element(body, at, pad=True)
    fields = [names_raw[i:i + width].split(b"\x00")[0].decode("ascii", "replace")
              for i in range(0, len(names_raw), width)]
    values, _ = _mat_children(body, at, count * len(fields))
    out = []
    for index in range(count):
        row = values[index * len(fields):(index + 1) * len(fields)]
        out.append(dict(zip(fields, row)))
    return out[0] if count == 1 else out


# ── pictures, for the formats that are not a codec ──────────────────────────
#
# **"A codec" was one sentence covering two costs**, the same way the `.mat` refusal
# was. JPEG is a discrete cosine transform, Huffman tables, chroma subsampling and a
# progressive mode: that is a codec and this library does not have one. PNG is
# **`zlib` plus arithmetic** — walk the chunks, inflate one stream, undo a filter that
# is chosen per row from five that each subtract a neighbour. PPM is a text header and
# then the bytes.
#
# Measured before writing either: of the forty-five datasets declined for "a codec",
# **two open PNG and no JPEG** and one opens PPM. So the sentence was carrying fifteen
# rows that are genuinely blocked and three that were not, and nobody had asked which.

def _png_read(data, keep_depth=False):
    """One PNG as `uint8` — `(h, w)` for grey, `(h, w, 3)` or `(h, w, 4)` for colour.

    Interlaced files are refused by name: Adam7 reorders the whole image into seven
    passes, and a reader that ignored the flag would return a picture built from the
    first pass alone — recognisable, wrong, and silent.

    ## Sixteen bits

    **PIL keeps them for grey and drops them for colour**, and this follows PIL because
    `_folder_loader` stands in its place. A sixteen-bit grey PNG comes back `uint16`;
    a sixteen-bit colour one comes back `uint8`, with the low byte gone, which is what
    PIL hands over and what a viewer shows.

    `keep_depth=True` asks for the samples at their own depth whatever the channel
    count — what `torchvision.io.decode_png` does. **The two are not a preference.**
    KITTI stores its flow as sixteen-bit colour and reads it as `(x - 2**15) / 64`;
    on the eight-bit answer that arithmetic still runs and still returns a flow field
    of the right shape, pointing somewhere else.
    """
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("not a PNG — the eight-byte signature does not match")
    at, idat, palette, alpha = 8, [], None, None
    width = height = depth = colour = interlace = 0
    while at + 8 <= len(data):
        (size,) = _struct.unpack_from(">I", data, at)
        kind = data[at + 4:at + 8]
        body = data[at + 8:at + 8 + size]
        at += 12 + size                          # 4 length, 4 type, body, 4 CRC
        if kind == b"IHDR":
            width, height, depth, colour, _, _, interlace = _struct.unpack(
                ">IIBBBBB", body)
        elif kind == b"PLTE":
            palette = _np.frombuffer(body, dtype=_np.uint8).reshape(-1, 3)
        elif kind == b"tRNS" and colour == 3:
            alpha = _np.frombuffer(body, dtype=_np.uint8)
        elif kind == b"IDAT":
            idat.append(body)
        elif kind == b"IEND":
            break
    if interlace:
        raise NotImplementedError(
            "interlaced PNG (Adam7) is not read here — the rows arrive in seven "
            "passes and reading only the first gives a recognisable picture of the "
            "wrong thing")
    if depth not in (1, 2, 4, 8, 16):
        raise ValueError(f"PNG bit depth {depth} is not one this reads")

    channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}[colour]
    raw = _zlib.decompress(b"".join(idat))
    return _png_rows(raw, width, height, depth, channels, colour, palette, alpha,
                     keep_depth)


def _png_rows(raw, width, height, depth, channels, colour, palette, alpha,
              keep_depth=False):
    """Undo the per-row filters and lay the samples out.

    **Every filter refers to the row above**, so this cannot be vectorised over rows —
    row `i` needs row `i-1` already reconstructed. It can be vectorised *within* a row
    for three of the five; `Sub`, `Average` and `Paeth` refer to the pixel to the left
    and are written as loops, which is what they are.
    """
    stride = (width * channels * depth + 7) // 8
    out = _np.zeros((height, stride), dtype=_np.uint8)
    step = max(1, channels * depth // 8)         # bytes between a sample and its left
    prior = _np.zeros(stride, dtype=_np.uint8)
    at = 0
    for row in range(height):
        kind = raw[at]
        line = _np.frombuffer(raw, dtype=_np.uint8, count=stride, offset=at + 1).copy()
        at += 1 + stride
        if kind == 0:
            pass
        elif kind == 2:                          # Up — the whole row at once
            line += prior
        else:
            for i in range(stride):
                left = int(line[i - step]) if i >= step else 0
                up = int(prior[i])
                upleft = int(prior[i - step]) if i >= step else 0
                if kind == 1:
                    line[i] = (int(line[i]) + left) & 0xFF
                elif kind == 3:
                    line[i] = (int(line[i]) + (left + up) // 2) & 0xFF
                elif kind == 4:
                    # Paeth: whichever of the three neighbours the gradient is nearest.
                    p = left + up - upleft
                    pa, pb, pc = abs(p - left), abs(p - up), abs(p - upleft)
                    near = left if (pa <= pb and pa <= pc) else (up if pb <= pc else upleft)
                    line[i] = (int(line[i]) + near) & 0xFF
                else:
                    raise ValueError(f"PNG row filter {kind} is not one of the five")
        out[row] = line
        prior = line

    if depth == 8:
        samples = out.reshape(height, width, channels)
    elif depth == 16:
        pairs = out.reshape(height, -1, 2).astype(_np.uint16)
        wide = ((pairs[..., 0] << 8) | pairs[..., 1]).reshape(height, width, channels)
        # **Grey keeps its sixteen bits and colour does not** — PIL's split, not one
        # chosen here, and the disparity maps depend on the grey half: KITTI's are
        # sixteen-bit grey divided by 256, so a reader that scaled them down first
        # would be off by that factor with every number still looking like a distance.
        samples = wide if (keep_depth or channels == 1) else (wide >> 8).astype(_np.uint8)
    else:
        # **Sub-byte depths are packed high bit first**, and a row is padded to a whole
        # byte. Omniglot's strokes are 1-bit, so this is the path its pictures take.
        bits = _np.unpackbits(out, axis=1)
        per = depth
        grouped = bits[:, :width * channels * per].reshape(height, -1, per)
        values = _np.zeros(grouped.shape[:2], dtype=_np.uint16)
        for b in range(per):
            values = (values << 1) | grouped[:, :, b]
        # Scale so that the largest value the depth can hold becomes 255 — that is what
        # a viewer shows and what PIL hands back.
        scale = 255 // ((1 << per) - 1) if colour != 3 else 1
        samples = (values * scale).astype(_np.uint8).reshape(height, width, channels)

    if colour == 3:                              # palette: the sample is an index
        if palette is None:
            raise ValueError("a palette PNG with no PLTE chunk")
        picture = palette[samples[:, :, 0]]
        if alpha is not None:
            wide = _np.full(samples.shape[:2], 255, dtype=_np.uint8)
            wide[:] = alpha[_np.clip(samples[:, :, 0], 0, len(alpha) - 1)]
            picture = _np.dstack([picture, wide])
        return _np.ascontiguousarray(picture)
    if channels == 1:
        return _np.ascontiguousarray(samples[:, :, 0])
    return _np.ascontiguousarray(samples)


def _bmp_read(data):
    """One uncompressed BMP — `(h, w)` for a grey palette, `(h, w, 3)` otherwise.

    **The format with no compression to speak of**: a fourteen-byte file header, a
    forty-byte information header, an optional palette, and the rows. `PhotoTour`'s
    patch sheets are stored this way, which is what makes that dataset a walk rather
    than a codec.

    Three things here each produce a picture rather than an error, so each is done
    where it can be seen:

    - **The rows run bottom to top** unless the height is negative, which is how the
      format says top-down. Ignoring it returns the picture upside down.
    - **Every row is padded to four bytes.** On a width that is not a multiple of four
      the padding shifts each row a little further than the last, and the result is a
      sheared picture that still looks like the original.
    - **The samples are BGR, not RGB.** Red and blue swap, which on a photograph is
      obvious and on a greyscale patch is invisible.

    Compression and the depths below eight are refused by name. `BI_RLE8` exists and
    is not written here, because the only thing that reads BMPs in this library is a
    dataset that does not use it.
    """
    if data[:2] != b"BM":
        raise ValueError("not a BMP — the two-byte signature does not match")
    (offset,) = _struct.unpack_from("<I", data, 10)
    header, width, height, _planes, depth, compression = _struct.unpack_from(
        "<IiiHHI", data, 14)
    if compression != 0:
        raise ValueError(
            f"this BMP is compressed (BI_ type {compression}) and only the "
            "uncompressed form is read here")
    if depth not in (8, 24, 32):
        raise ValueError(f"BMP bit depth {depth} is not one this reads")

    top_down = height < 0
    height = abs(height)
    stride = (width * depth // 8 + 3) // 4 * 4          # rows padded to four bytes
    rows = _np.frombuffer(data, dtype=_np.uint8, count=stride * height,
                          offset=offset).reshape(height, stride)
    if not top_down:
        rows = rows[::-1]

    if depth == 8:
        samples = rows[:, :width]
        palette = _np.frombuffer(data, dtype=_np.uint8, count=(offset - 14 - header),
                                 offset=14 + header).reshape(-1, 4)
        # **A grey palette is a grey picture.** PIL calls that mode `L` and hands back
        # one channel; expanding it to three would make `PhotoTour`'s own reshape an
        # error against code that works.
        if _np.array_equal(palette[:, 0], palette[:, 1]) and \
                _np.array_equal(palette[:, 1], palette[:, 2]):
            if _np.array_equal(palette[:, 0], _np.arange(len(palette), dtype=_np.uint8)):
                return _np.ascontiguousarray(samples)
            return _np.ascontiguousarray(palette[samples, 0])
        return _np.ascontiguousarray(palette[samples][:, :, 2::-1])

    channels = depth // 8
    samples = rows[:, :width * channels].reshape(height, width, channels)
    return _np.ascontiguousarray(samples[:, :, 2::-1])   # BGR to RGB, alpha dropped


def _ppm_read(data):
    """One binary PPM/PGM (`P5` or `P6`) as `uint8`.

    **The cheapest format there is**: a magic number, three numbers, then the samples.
    Comments run from `#` to the end of a line and may sit between any two fields,
    which is the only part that is not a `split()`.
    """
    if data[:2] not in (b"P5", b"P6"):
        raise ValueError("not a binary PGM or PPM — it does not begin `P5` or `P6`")
    channels = 1 if data[:2] == b"P5" else 3
    fields, at = [], 2
    while len(fields) < 3:
        while at < len(data) and data[at:at + 1].isspace():
            at += 1
        if data[at:at + 1] == b"#":
            while at < len(data) and data[at] != 0x0A:
                at += 1
            continue
        start = at
        while at < len(data) and not data[at:at + 1].isspace():
            at += 1
        fields.append(int(data[start:at]))
    width, height, largest = fields
    at += 1                                      # exactly one whitespace byte follows
    if largest > 255:
        raise NotImplementedError(
            "a 16-bit PPM is not read here — its samples are big-endian pairs")
    got = _np.frombuffer(data, dtype=_np.uint8, count=width * height * channels,
                         offset=at)
    shape = (height, width) if channels == 1 else (height, width, channels)
    return _np.ascontiguousarray(got.reshape(shape))


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


class Omniglot(VisionDataset):
    """1,623 hand-drawn characters from 50 alphabets, 20 drawings each, **as PNG.**

    **Declined for "a codec", and PNG is not the codec that word was standing for.**
    JPEG is a discrete cosine transform with Huffman tables and a progressive mode.
    PNG is `zlib` — which the standard library has — plus a chunk walk and a row
    filter chosen from five that each subtract a neighbour. `_png_read` above is the
    whole of it, and Omniglot's pictures take its narrowest path: 105×105, one
    channel, **one bit deep.**

    Two things a reader gets wrong quietly:

    - **The classes are alphabet *and* character**, not alphabet. `Latin/character07`
      is one class and `Latin/character08` another, so folding by alphabet gives 50
      classes where torchvision gives 964, and every accuracy computed on it is a
      different number for a different problem.
    - **A stroke is 0 and the paper is 1** in the file, and PIL's `convert("L")`
      leaves it that way. Inverting to make the ink bright is the natural thing and
      makes every pixel disagree with torchvision.

    `background=True` is the 964-class training half; `False` is the 659-class
    evaluation half. torchvision's word, kept.
    """

    folder = "omniglot-py"
    url_prefix = ("https://raw.githubusercontent.com/brendenlake/omniglot/"
                  "master/python")
    zips_md5 = {
        "images_background": "68d2efa1b9178cc56df9314c21c6e718",
        "images_evaluation": "6b91aef0f799c5bb55b94e3f2daec811",
    }

    def __init__(self, root, background=True, transform=None, target_transform=None,
                 download=False, loader=None):
        super().__init__(_os.path.join(root, self.folder), transform=transform,
                         target_transform=target_transform)
        self.background = background
        self.loader = loader
        name = self._target_folder()
        folder = _os.path.join(self.root, name)
        archive = _os.path.join(self.root, name + ".zip")
        if download and not _os.path.isdir(folder):
            _os.makedirs(self.root, exist_ok=True)
            if not (_os.path.isfile(archive)
                    and _md5_of_file(archive) == self.zips_md5[name]):
                _fetch_to(f"{self.url_prefix}/{name}.zip", archive,
                          self.zips_md5[name])
            with _zipfile.ZipFile(archive) as zipped:
                zipped.extractall(self.root)
        if not _os.path.isdir(folder):
            raise RuntimeError("Dataset not found or corrupted. You can use "
                               "download=True to download it")
        self.target_folder = folder
        # **Not sorted, and that is deliberate.** torchvision's `list_dir` returns
        # `os.listdir` unchanged, so the class index of a character is the position the
        # filesystem happened to hand it back in. Sorting is the better rule — the same
        # folder would then give the same labels everywhere — and it is a *different*
        # rule: on a directory `os.listdir` returns unsorted, every label disagrees
        # with torchvision's while both are internally consistent, and an accuracy
        # compared against a published number is comparing two class orders.
        #
        # Measured on a fixture: sorted gave `Greek/character01` class 0 and
        # torchvision gave it class 3. Nothing raises, both train, and the confusion
        # matrices are permutations of each other.
        #
        # The same call this repository makes for EMNIST's transposed pictures:
        # correcting silently is what makes two libraries disagree everywhere.
        self._alphabets = [d for d in _os.listdir(folder)
                           if _os.path.isdir(_os.path.join(folder, d))]
        self._characters = [
            _os.path.join(a, c) for a in self._alphabets
            for c in _os.listdir(_os.path.join(folder, a))
            if _os.path.isdir(_os.path.join(folder, a, c))]
        self._character_images = [
            [(f, idx) for f in _os.listdir(_os.path.join(folder, character))
             if f.endswith(".png")]
            for idx, character in enumerate(self._characters)]
        self._flat_character_images = [
            pair for images in self._character_images for pair in images]

    def _target_folder(self):
        return "images_background" if self.background else "images_evaluation"

    def __len__(self):
        return len(self._flat_character_images)

    def __getitem__(self, index):
        name, character_class = self._flat_character_images[index]
        path = _os.path.join(self.target_folder,
                             self._characters[character_class], name)
        if self.loader is not None:
            picture = self.loader(path)
        else:
            with open(path, "rb") as handle:
                picture = _png_read(handle.read())
        if self.transforms is not None:
            picture, character_class = self.transforms(picture, character_class)
        return picture, character_class


class GTSRB(VisionDataset):
    """German traffic signs — 43 classes, **as binary PPM.**

    The cheapest picture format there is: a magic number, three numbers, then the
    samples. It sat behind the same "a codec" sentence as the JPEG sets for as long as
    nobody asked what the archive holds.

    **The two splits are laid out differently and that is the whole of the work.**
    Training is a folder per class and the class is the folder's name; test is one flat
    folder with the answers in a semicolon-separated CSV, and its rows are not in the
    order the files sort in. Reading the folder for the test split gives 12,630
    pictures with the labels of whichever order the filesystem returned.
    """

    base = ("https://sid.erda.dk/public/archives/"
            "daaeac0d7ce1152aea9b61d9f1e19370/")
    resources = {
        "train": ("GTSRB-Training_fixed.zip", "513f3c79a4c5141765e10e952eaa2478"),
        "test": ("GTSRB_Final_Test_Images.zip", "c7e4e6327067d32654124b0fe9e82185"),
        "test_gt": ("GTSRB_Final_Test_GT.zip", "fe31e9c9270bbcd7b84b7f21a9d9d9e5"),
    }

    def __init__(self, root, split="train", transform=None, target_transform=None,
                 download=False):
        super().__init__(root, transform=transform, target_transform=target_transform)
        if split not in ("train", "test"):
            raise ValueError(
                f"Unknown value '{split}' for argument split. "
                "Valid values are ('train', 'test').")
        self.split = split
        self._base_folder = _os.path.join(self.root, "gtsrb")
        self._target_folder = _os.path.join(
            self._base_folder, "GTSRB",
            "Training" if split == "train" else _os.path.join("Final_Test", "Images"))
        if download:
            self.download()
        if not _os.path.isdir(self._target_folder):
            raise RuntimeError("Dataset not found. You can use download=True to "
                               "download it")
        if split == "train":
            # **The label is the folder's position, not the number in its name.**
            # torchvision walks the sorted folders and hands out 0, 1, 2 …, which is
            # `DatasetFolder`'s rule everywhere. Reading `int("00007")` as 7 agrees on
            # the real dataset — its forty-three folders are `00000` to `00042` with
            # none missing, so position and name are the same number — and parts on
            # any subset. Found on a fixture with two folders, `00000` and `00007`:
            # ours said `[0, 0, 7, 7]` and torchvision `[0, 0, 1, 1]`.
            #
            # That is the shape where **a complete input cannot tell two rules apart**,
            # and the fixture only caught it because the folders picked to be readable
            # happened not to be contiguous.
            folders = sorted(
                d for d in _os.listdir(self._target_folder)
                if _os.path.isdir(_os.path.join(self._target_folder, d)))
            self._samples = []
            for index, folder in enumerate(folders):
                here = _os.path.join(self._target_folder, folder)
                for name in sorted(_os.listdir(here)):
                    if name.endswith(".ppm"):
                        self._samples.append(
                            (_os.path.join(here, name), index))
        else:
            answers = _os.path.join(self._base_folder, "GT-final_test.csv")
            with open(answers, newline="") as handle:
                self._samples = [
                    (_os.path.join(self._target_folder, row["Filename"]),
                     int(row["ClassId"]))
                    for row in _csv.DictReader(handle, delimiter=";",
                                               skipinitialspace=True)]

    def download(self):
        _os.makedirs(self._base_folder, exist_ok=True)
        wanted = ["train"] if self.split == "train" else ["test", "test_gt"]
        for key in wanted:
            name, digest = self.resources[key]
            archive = _os.path.join(self._base_folder, name)
            if not (_os.path.isfile(archive) and _md5_of_file(archive) == digest):
                _fetch_to(self.base + name, archive, digest)
            with _zipfile.ZipFile(archive) as zipped:
                zipped.extractall(self._base_folder)

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, index):
        path, target = self._samples[index]
        with open(path, "rb") as handle:
            picture = _ppm_read(handle.read())
        if self.transforms is not None:
            picture, target = self.transforms(picture, target)
        return picture, target

    def extra_repr(self):
        return f"Split: {self.split}"


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


#: The extensions `ImageFolder` walks. **torchvision's nine minus the six with a
#: codec behind them** — `.jpg`, `.jpeg`, `.bmp`, `.tif`, `.tiff` and `.webp` are
#: absent because opening one needs a decoder this library does not carry, and a
#: folder holding them is a folder this cannot read. Naming them here rather than
#: taking torch's list whole is what makes that visible at construction instead of
#: on the picture that happens to be JPEG.
IMG_EXTENSIONS = (".bmp", ".png", ".ppm", ".pgm")


def _folder_loader(path):
    """One picture from a path, **by its bytes and not by its name.**

    A file called `.png` that is not one is a real thing — a download that fetched
    an error page, an archive that renamed on extraction — and the magic number
    settles it in two bytes. torchvision's loader dispatches on the suffix and hands
    the rest to PIL, which sniffs the same way one layer down.

    The refusal names the codec it met rather than saying "unsupported", because
    JPEG is the answer somebody will actually be holding.
    """
    with open(path, "rb") as handle:
        data = handle.read()
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return _png_read(data)
    if data[:2] in (b"P5", b"P6"):
        return _ppm_read(data)
    if data[:2] == b"BM":
        return _bmp_read(data)
    if data[:2] == b"\xff\xd8":
        raise ValueError(
            f"{path} is JPEG, and there is no JPEG decoder here.\n"
            "  PNG and PPM are read directly; a JPEG needs a discrete cosine "
            "transform, Huffman tables and a progressive mode, which is a codec "
            "rather than a container.")
    raise ValueError(
        f"{path} is not a PNG or a PPM — it begins {data[:4]!r}.\n"
        f"  ImageFolder here reads {', '.join(IMG_EXTENSIONS)}.")


class ImageFolder(DatasetFolder):
    """`DatasetFolder` with the loader chosen. **That choice was the whole of what
    was missing**, and it used to be the whole of what was refused.

    The gap table said of this name *its pictures need a codec — the same answer PIL
    gets*, which was one true sentence about a class that reads no format itself.
    `DatasetFolder` walks directories and calls a function; the codec lives in the
    function, and this library has two of them. So what is here opens **PNG and PPM**
    and refuses the rest by name.

    That is narrower than torchvision's, which opens nine extensions through PIL, and
    the narrowness is stated rather than hidden: `IMG_EXTENSIONS` above is the list,
    a file outside it is not walked, and a file inside it that turns out to be JPEG
    stops with a sentence naming JPEG. **A folder of PNGs gives the same answer as
    torchvision's** — compared class for class, index for index, and pixel for pixel.

    **And it is narrower than "PNG" too.** `_png_read` refuses Adam7 interlacing,
    where the rows arrive in seven passes; a reader that ignored the flag would hand
    back a picture built from the first pass — recognisable, wrong and silent. So a
    folder holding one stops on that file rather than on the format, which is the
    honest place for it to stop and is measured rather than assumed: PIL will not
    write an interlaced PNG, so the fixture that proves it sets the IHDR flag by hand.
    """

    def __init__(self, root, transform=None, target_transform=None,
                 loader=_folder_loader, is_valid_file=None):
        super().__init__(root, loader,
                         None if is_valid_file is not None else IMG_EXTENSIONS,
                         transform=transform, target_transform=target_transform,
                         is_valid_file=is_valid_file)
        #: torchvision keeps this alias and code in the wild reads it.
        self.imgs = self.samples




class RenderedSST2(DatasetFolder):
    """Sentences from SST-2 **rendered as pictures**, for reading text off an image.

    <https://github.com/openai/CLIP/blob/main/data/rendered-sst2.md>

    ## Why this one was refused and is not

    The gap table put it under *a codec*. It is `png` — the extension is written into
    torchvision's own constructor as `make_dataset(..., extensions=("png",))`, so the
    answer sat in the source the whole time and nobody read it. That is the fourth
    refusal here to have covered something nobody had looked at.

    ## What it is

    A folder per class under a folder per split, which is `DatasetFolder` with the
    walk already described — so this subclasses it rather than repeating the walk.
    Two things are fixed rather than discovered, and both are torchvision's:

    - **The classes are declared and not sorted out of the directory.** `negative` is
      0 and `positive` is 1, which is the sorted order anyway; declaring it means a
      stray folder appearing on disk cannot renumber a trained model.
    - **`val` lives in a folder called `valid`.** A reader that used the split name
      finds nothing and reports an empty dataset rather than a missing directory.
    """

    _URL = "https://openaipublic.azureedge.net/clip/data/rendered-sst2.tgz"
    _MD5 = "2384d08e9dcfa4bd55b324e610496ee5"
    _SPLIT_TO_FOLDER = {"train": "train", "val": "valid", "test": "test"}

    def __init__(self, root, split="train", transform=None, target_transform=None,
                 download=False, loader=None):
        if split not in self._SPLIT_TO_FOLDER:
            raise ValueError(f"Unknown value '{split}' for argument split. Valid "
                             "values are {train, val, test}.")
        self._split = split
        self._given_root = root
        self._base_folder = _os.path.join(root, "rendered-sst2")
        self._folder = _os.path.join(self._base_folder,
                                     self._SPLIT_TO_FOLDER[split])
        if download:
            self.download()
        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to "
                               "download it")
        super().__init__(self._folder, _folder_loader if loader is None else loader,
                         (".png",), transform=transform,
                         target_transform=target_transform)
        # **The root printed is the one that was passed.** `DatasetFolder` was handed
        # the split's folder so its walk starts in the right place, and `repr` shows
        # the root — printing the deeper path would show a reader a directory they
        # never named.
        self.root = root
        self.classes = ["negative", "positive"]
        self.class_to_idx = {"negative": 0, "positive": 1}

    def _check_exists(self):
        return all(_os.path.isdir(_os.path.join(self._folder, name))
                   for name in ("negative", "positive"))

    def download(self):
        """The archive is a `.tgz` of some tens of megabytes, fetched and hashed the
        way the others here are."""
        if self._check_exists():
            return
        _os.makedirs(self._given_root, exist_ok=True)
        archive = _os.path.join(self._given_root, "rendered-sst2.tgz")
        if _os.path.isfile(archive) and _md5_of_file(archive) != self._MD5:
            _os.unlink(archive)
        if not _os.path.isfile(archive):
            _fetch_to(self._URL, archive, self._MD5)
        root = _os.path.abspath(self._given_root)
        with _tarfile.open(archive, mode="r:gz") as tar:
            for member in tar:
                if not member.isfile():
                    continue
                target = _os.path.join(root, *member.name.split("/"))
                # **A tar can carry `../` in a member's name**, and joining it blind
                # writes outside the directory the caller named. Members that leave
                # the root are skipped rather than trusted.
                if not _os.path.abspath(target).startswith(root + _os.sep):
                    continue
                _os.makedirs(_os.path.dirname(target), exist_ok=True)
                extracted = tar.extractfile(member)
                if extracted is None:
                    continue
                with open(target, "wb") as handle:
                    while True:
                        block = extracted.read(1 << 20)
                        if not block:
                            break
                        handle.write(block)

    def extra_repr(self):
        return f"split={self._split}"



# --- optical flow: two containers that were counted as codecs -----------------
#
# The gap table said of the stereo and flow datasets *a codec and then another
# format*. The first half is about the pictures and was measured — most of them are
# PNG, which is read here. **The second half was never opened.**
#
# `.flo` is a magic number, two little-endian integers and a block of float32.
# `.pfm` is a text header, a size, a scale whose sign carries the endianness, and a
# block of float — a PPM with floats in it, which is what the name says. Neither is a
# codec; both are the kind of container `_mat_read` already stands as precedent for,
# and each is under twenty lines of `struct` and `numpy`.


def _read_flo(data):
    """Middlebury `.flo` — **little endian, always**, and the magic says so.

    The four bytes are `PIEH`, which is `1e10` read as a float and is how the format
    announces the byte order it was written in. A reader that took the machine's order
    gets a flow field that is the right shape and points the wrong way.
    """
    if data[:4] != b"PIEH":
        raise ValueError("Magic number incorrect. Invalid .flo file")
    width = int.from_bytes(data[4:8], "little")
    height = int.from_bytes(data[8:12], "little")
    flat = _np.frombuffer(data, dtype="<f4", offset=12, count=2 * width * height)
    return flat.reshape(height, width, 2).transpose(2, 0, 1)


def _read_pfm(data, slice_channels=2):
    """`.pfm` — a PPM with floats. **The scale's sign is the endianness** and the rows
    arrive bottom-up.

    Three things in that sentence each produce a plausible wrong answer on their own:
    reading big-endian where the file said little gives numbers that are enormous
    rather than absurd, keeping the sign of the scale gives every value negated, and
    skipping the flip gives an image that is upside down and still an image.

    `PF` is three channels and `Pf` is one; `slice_channels` takes the leading ones,
    because a disparity map is a flow field's first channel and torchvision reads both
    through here.
    """
    at = 0

    def line():
        nonlocal at
        end = data.index(b"\n", at)
        got = data[at:end].rstrip()
        at = end + 1
        return got

    header = line()
    if header not in (b"PF", b"Pf"):
        raise ValueError("Invalid PFM file")
    fields = line().split()
    width, height = int(fields[0]), int(fields[1])
    scale = float(line())
    endian = "<" if scale < 0 else ">"
    channels = 3 if header == b"PF" else 1
    flat = _np.frombuffer(data, dtype=endian + "f4", offset=at,
                          count=width * height * channels)
    out = flat.reshape(height, width, channels).transpose(2, 0, 1)
    out = _np.flip(out, axis=1)
    return _np.ascontiguousarray(out[:slice_channels])



class Sintel(VisionDataset):
    """Optical flow on the Sintel film. **A pair of frames in, the motion between
    them out.**

    <http://sintel.is.tue.mpg.de/>

    ## Why it was refused and is not

    Its row read *a codec and then another format*. The pictures are PNG — read here —
    and the other format is `.flo`, which turned out to be a magic number, two
    integers and a block of floats. One sentence, two walls, and only one of them
    was ever there.

    ## What the item is

    `(frame, next frame, flow)` on `train` and `(frame, next frame, None)` on `test`,
    because the test split ships no flow. Two things about the pairing are worth
    stating, and each gives a dataset that trains:

    - **The pairs are consecutive within a scene**, so a scene of `n` frames yields
      `n - 1` items and the last frame is a second frame only. Pairing across the
      scene boundary would put the end of one shot against the start of another and
      ask the model to explain a cut.
    - **The scenes are walked in sorted order.** torchvision walks them in
      `os.listdir` order, which is the disk's and not the dataset's — sorted on one
      filesystem, hashed on another — so the order of its items depends on where the
      tree sits. Sorting here costs nothing and makes the answer the same twice.
    - **`pass_name` is which rendering**, not which split. `clean` and `final` are
      the same motion through different shading, and `both` is the two concatenated —
      so the flow list is walked once per pass, and a reader that walked it once for
      `both` runs out of flows halfway.
    """

    def __init__(self, root, split="train", pass_name="clean", transforms=None,
                 loader=None):
        if split not in ("train", "test"):
            raise ValueError(f"Unknown value '{split}' for argument split. Valid "
                             "values are {train, test}.")
        if pass_name not in ("clean", "final", "both"):
            raise ValueError(f"Unknown value '{pass_name}' for argument pass_name. "
                             "Valid values are {clean, final, both}.")
        super().__init__(root)
        self.transforms = transforms
        self.loader = _folder_loader if loader is None else loader
        self._split = split
        self._image_list = []
        self._flow_list = []

        base = _os.path.join(self.root, "Sintel")
        flow_root = _os.path.join(base, "training", "flow")
        split_dir = "training" if split == "train" else split
        for name in (("clean", "final") if pass_name == "both" else (pass_name,)):
            image_root = _os.path.join(base, split_dir, name)
            for scene in sorted(_os.listdir(image_root)):
                frames = sorted(
                    _os.path.join(image_root, scene, f)
                    for f in _os.listdir(_os.path.join(image_root, scene))
                    if f.endswith(".png"))
                self._image_list += [[frames[i], frames[i + 1]]
                                     for i in range(len(frames) - 1)]
                if split == "train":
                    folder = _os.path.join(flow_root, scene)
                    self._flow_list += sorted(
                        _os.path.join(folder, f) for f in _os.listdir(folder)
                        if f.endswith(".flo"))

    def __len__(self):
        return len(self._image_list)

    def __getitem__(self, index):
        first = self.loader(self._image_list[index][0])
        second = self.loader(self._image_list[index][1])
        flow = None
        if self._flow_list:
            with open(self._flow_list[index], "rb") as handle:
                flow = _read_flo(handle.read())
        if self.transforms is not None:
            first, second, flow, _ = self.transforms(first, second, flow, None)
        return first, second, flow


def _read_16bit_flow_png(path):
    """KITTI's and HD1K's flow: **a sixteen-bit colour PNG carrying three things.**

    Two channels are the motion, stored as `x * 64 + 2**15` so that a signed
    displacement fits in an unsigned sample, and the third is a flag saying whether
    the pixel was measured at all. Undoing it is the two lines below.

    The reason this needs `keep_depth` is that PIL cannot hold the file: it narrows
    sixteen-bit colour to eight, and `(x - 2**15) / 64` on the narrowed samples still
    runs, still returns a field of the right shape, and points somewhere else.
    """
    with open(path, "rb") as handle:
        both = _png_read(handle.read(), keep_depth=True).astype(_np.float32)
    flow = (both[:, :, :2] - 2 ** 15) / 64
    return flow.transpose(2, 0, 1), both[:, :, 2].astype(bool)


def _as_rgb(picture):
    """What torchvision's stereo `_read_img` does — a grey picture is stacked three
    deep rather than left as one channel, so the two sides of a pair have the same
    shape whichever way each was stored."""
    picture = _np.asarray(picture)
    if picture.ndim == 2:
        return _np.repeat(picture[:, :, None], 3, axis=2)
    return picture[:, :, :3]


class _StereoMatchingDataset(VisionDataset):
    """The half that four stereo datasets share: two lists of pairs, and an item that
    is `(left, right, disparity)` — or the same with a mask when the dataset ships
    one.

    **`_scan_pairs` sorts both sides separately and zips them**, which only pairs the
    right pictures with the right ones because the two directories use the same names.
    A dataset that broke that convention would pair silently and wrongly, so the
    counts are compared and a mismatch is refused rather than truncated by `zip`.
    """

    _has_built_in_disparity_mask = False

    def __init__(self, root, transforms=None):
        super().__init__(root)
        self.transforms = transforms
        self._images = []
        self._disparities = []

    def _scan_pairs(self, left_pattern, right_pattern=None):
        left = sorted(_glob.glob(left_pattern))
        if not left:
            raise FileNotFoundError(
                f"Could not find any files matching the patterns: {left_pattern}")
        if right_pattern is None:
            return [(one, None) for one in left]
        right = sorted(_glob.glob(right_pattern))
        if len(left) != len(right):
            raise ValueError(
                f"Found {len(left)} left files but {len(right)} right files using:\n "
                f"left pattern: {left_pattern}\n"
                f"right pattern: {right_pattern}\n")
        return list(zip(left, right))

    def _read_img(self, path):
        return _as_rgb(_folder_loader(path))

    def _read_disparity(self, path):
        raise NotImplementedError

    def __len__(self):
        return len(self._images)

    def __getitem__(self, index):
        left = self._read_img(self._images[index][0])
        right = self._read_img(self._images[index][1])
        maps, masks = [], []
        for path in self._disparities[index]:
            one, mask = self._read_disparity(path)
            maps.append(one)
            masks.append(mask)
        images = (left, right)
        if self.transforms is not None:
            images, maps, masks = self.transforms(images, tuple(maps), tuple(masks))
        if self._has_built_in_disparity_mask or masks[0] is not None:
            return images[0], images[1], maps[0], masks[0]
        return images[0], images[1], maps[0]


class KittiFlow(VisionDataset):
    """KITTI 2015, **as optical flow**: two frames and the motion between them.

    <http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=flow>

    ## Why it was refused and is not

    Its row read *a codec and then another format*. Both halves were wrong here. The
    pictures are PNG, read already; the flow is **also** PNG — sixteen-bit, with the
    motion in two channels and a validity flag in the third. There was never a second
    format, only a second use of the first.

    The frames pair by **suffix, not by order**: `*_10.png` is every first frame and
    `*_11.png` is every second, in two sorted lists zipped together. A reader that
    took the directory in order would pair each scene's second frame with the next
    scene's first — half the items would be a cut, and all of them would load.
    """

    _has_builtin_flow_mask = True

    def __init__(self, root, split="train", transforms=None, loader=None):
        if split not in ("train", "test"):
            raise ValueError(f"Unknown value '{split}' for argument split. Valid "
                             "values are {train, test}.")
        super().__init__(root)
        self.transforms = transforms
        self.loader = _folder_loader if loader is None else loader
        base = _os.path.join(self.root, "KittiFlow", split + "ing")
        first = sorted(_glob.glob(_os.path.join(base, "image_2", "*_10.png")))
        second = sorted(_glob.glob(_os.path.join(base, "image_2", "*_11.png")))
        if not first or not second:
            raise FileNotFoundError(
                "Could not find the Kitti flow images. Please make sure the "
                "directory structure is correct.")
        self._image_list = [[one, two] for one, two in zip(first, second)]
        self._flow_list = (sorted(_glob.glob(_os.path.join(base, "flow_occ", "*_10.png")))
                           if split == "train" else [])

    def __len__(self):
        return len(self._image_list)

    def __getitem__(self, index):
        first = self.loader(self._image_list[index][0])
        second = self.loader(self._image_list[index][1])
        flow = mask = None
        if self._flow_list:
            flow, mask = _read_16bit_flow_png(self._flow_list[index])
        if self.transforms is not None:
            first, second, flow, mask = self.transforms(first, second, flow, mask)
        return first, second, flow, mask


class HD1K(VisionDataset):
    """HD1K, **thirty-six sequences** of frames and the flow between them.

    <http://hci-benchmark.iwr.uni-heidelberg.de/>

    ## Why it was refused and is not

    As `KittiFlow` — the row said *a codec and then another format*, and the second
    format is the same sixteen-bit PNG.

    **The sequences are walked one at a time and each stops one frame short**, which
    is the whole of what this class has to get right. Flattening the thirty-six into
    one sorted list would pair the last frame of a sequence with the first of the
    next, and the flow file it took would belong to neither.
    """

    _has_builtin_flow_mask = True
    _SEQUENCES = 36

    def __init__(self, root, split="train", transforms=None, loader=None):
        if split not in ("train", "test"):
            raise ValueError(f"Unknown value '{split}' for argument split. Valid "
                             "values are {train, test}.")
        super().__init__(root)
        self.transforms = transforms
        self.loader = _folder_loader if loader is None else loader
        self._image_list = []
        self._flow_list = []
        base = _os.path.join(self.root, "hd1k")
        if split == "train":
            for sequence in range(self._SEQUENCES):
                flows = sorted(_glob.glob(_os.path.join(
                    base, "hd1k_flow_gt", "flow_occ", f"{sequence:06d}_*.png")))
                frames = sorted(_glob.glob(_os.path.join(
                    base, "hd1k_input", "image_2", f"{sequence:06d}_*.png")))
                for i in range(len(flows) - 1):
                    self._flow_list += [flows[i]]
                    self._image_list += [[frames[i], frames[i + 1]]]
        else:
            first = sorted(_glob.glob(_os.path.join(
                base, "hd1k_challenge", "image_2", "*10.png")))
            second = sorted(_glob.glob(_os.path.join(
                base, "hd1k_challenge", "image_2", "*11.png")))
            self._image_list = [[one, two] for one, two in zip(first, second)]
        if not self._image_list:
            raise FileNotFoundError(
                "Could not find the HD1K images. Please make sure the directory "
                "structure is correct.")

    def __len__(self):
        return len(self._image_list)

    def __getitem__(self, index):
        first = self.loader(self._image_list[index][0])
        second = self.loader(self._image_list[index][1])
        flow = mask = None
        if self._flow_list:
            flow, mask = _read_16bit_flow_png(self._flow_list[index])
        if self.transforms is not None:
            first, second, flow, mask = self.transforms(first, second, flow, mask)
        return first, second, flow, mask


class Kitti2012Stereo(_StereoMatchingDataset):
    """KITTI 2012, **as stereo**: a left and a right picture and the disparity.

    <http://www.cvlibs.net/datasets/kitti/eval_stereo_flow.php>

    ## Why it was refused and is not

    Its row named a codec. The pictures are PNG and so is the disparity — **sixteen-bit
    grey, divided by 256**, which is the half that needed the reader fixed rather than
    extended. On a reader that narrowed sixteen bits to eight the division still runs
    and every distance is off by that factor while still looking like a distance.

    The `colored_0` and `colored_1` directories are the RGB pair; the dataset also
    ships a greyscale pair, and torchvision takes the colour one for consistency with
    2015.
    """

    _has_built_in_disparity_mask = True

    def __init__(self, root, split="train", transforms=None):
        if split not in ("train", "test"):
            raise ValueError(f"Unknown value '{split}' for argument split. Valid "
                             "values are {train, test}.")
        super().__init__(root, transforms)
        base = _os.path.join(self.root, "Kitti2012", split + "ing")
        self._images = self._scan_pairs(
            _os.path.join(base, "colored_0", "*_10.png"),
            _os.path.join(base, "colored_1", "*_10.png"))
        if split == "train":
            self._disparities = self._scan_pairs(
                _os.path.join(base, "disp_noc", "*.png"), None)
        else:
            self._disparities = [(None, None) for _ in self._images]

    def _read_disparity(self, path):
        if path is None:
            return None, None
        return _np.asarray(_folder_loader(path))[None, :, :] / 256.0, None


class Kitti2015Stereo(_StereoMatchingDataset):
    """KITTI 2015, **as stereo**. As `Kitti2012Stereo`, with both disparities present.

    <http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php>

    The pair is `image_2` against `image_3` and the disparities are `disp_occ_0` and
    `disp_occ_1` — **the right disparity is read and then dropped**, because the item
    is the left one. It is read anyway so that a missing right file is an error here
    rather than a surprise to whoever writes the transform that wants it.
    """

    _has_built_in_disparity_mask = True

    def __init__(self, root, split="train", transforms=None):
        if split not in ("train", "test"):
            raise ValueError(f"Unknown value '{split}' for argument split. Valid "
                             "values are {train, test}.")
        super().__init__(root, transforms)
        base = _os.path.join(self.root, "Kitti2015", split + "ing")
        self._images = self._scan_pairs(_os.path.join(base, "image_2", "*.png"),
                                        _os.path.join(base, "image_3", "*.png"))
        if split == "train":
            self._disparities = self._scan_pairs(
                _os.path.join(base, "disp_occ_0", "*.png"),
                _os.path.join(base, "disp_occ_1", "*.png"))
        else:
            self._disparities = [(None, None) for _ in self._images]

    def _read_disparity(self, path):
        if path is None:
            return None, None
        return _np.asarray(_folder_loader(path))[None, :, :] / 256.0, None


class InStereo2k(_StereoMatchingDataset):
    """InStereo2k — **one scene per directory**, each holding its four files.

    <https://github.com/YuhuaXu/StereoDataset>

    ## Why it was refused and is not

    A codec, said the row, and it is PNG throughout. The disparity is **divided by
    1024** rather than 256; the number is the dataset's and not a convention, which is
    why it is written here beside the one it is not.
    """

    _DISPARITY_SCALE = 1024.0

    def __init__(self, root, split="train", transforms=None):
        if split not in ("train", "test"):
            raise ValueError(f"Unknown value '{split}' for argument split. Valid "
                             "values are {train, test}.")
        super().__init__(root, transforms)
        base = _os.path.join(self.root, "InStereo2k", split)
        self._images = self._scan_pairs(_os.path.join(base, "*", "left.png"),
                                        _os.path.join(base, "*", "right.png"))
        self._disparities = self._scan_pairs(
            _os.path.join(base, "*", "left_disp.png"),
            _os.path.join(base, "*", "right_disp.png"))

    def _read_disparity(self, path):
        one = _np.asarray(_folder_loader(path), dtype=_np.float32)
        return one[None, :, :] / self._DISPARITY_SCALE, None


class SintelStereo(_StereoMatchingDataset):
    """Sintel again, **as stereo** — the same film, a left and a right eye.

    <http://sintel.is.tue.mpg.de/stereo>

    ## Why it was refused and is not

    A codec, said the row. Everything here is PNG, including two things that are not
    pictures:

    - **The disparity is packed into a colour**, `r * 4 + g / 2**6 + b / 2**14`, which
      is the dataset's README and not a convention. Reading the red channel alone
      gives a disparity map that is coarse and plausible.
    - **The validity mask is two files**, occlusion and out-of-frame, and a pixel is
      valid only where both are zero. Taking either alone leaves invalid pixels marked
      valid, and they are exactly the ones a stereo model would otherwise be scored on.

    Both mask files are required. A missing one is refused by path rather than
    defaulted to all-valid, because all-valid is a mask that trains.
    """

    _has_built_in_disparity_mask = True

    def __init__(self, root, pass_name="final", transforms=None):
        if pass_name not in ("final", "clean", "both"):
            raise ValueError(f"Unknown value '{pass_name}' for argument pass_name. "
                             "Valid values are {final, clean, both}.")
        super().__init__(root, transforms)
        base = _os.path.join(self.root, "Sintel")
        for name in (("final", "clean") if pass_name == "both" else (pass_name,)):
            self._images += self._scan_pairs(
                _os.path.join(base, "training", f"{name}_left", "*", "*.png"),
                _os.path.join(base, "training", f"{name}_right", "*", "*.png"))
            self._disparities += self._scan_pairs(
                _os.path.join(base, "training", "disparities", "*", "*.png"), None)

    def _get_occlussion_mask_paths(self, path):
        scene = _os.path.dirname(path)
        sample = _os.path.dirname(_os.path.dirname(scene))
        name = _os.path.basename(path)
        occlusion = _os.path.join(sample, "occlusions", _os.path.basename(scene), name)
        out_of_frame = _os.path.join(sample, "outofframe", _os.path.basename(scene), name)
        if not _os.path.exists(occlusion):
            raise FileNotFoundError(f"Occlusion mask {occlusion} does not exist")
        if not _os.path.exists(out_of_frame):
            raise FileNotFoundError(
                f"Out of frame mask {out_of_frame} does not exist")
        return occlusion, out_of_frame

    def _read_disparity(self, path):
        if path is None:
            return None, None
        packed = _np.asarray(_folder_loader(path), dtype=_np.float32)
        red, green, blue = _np.split(packed, 3, axis=-1)
        one = _np.transpose(red * 4 + green / (2 ** 6) + blue / (2 ** 14), (2, 0, 1))
        occlusion, out_of_frame = self._get_occlussion_mask_paths(path)
        mask = _np.logical_and(
            _np.asarray(_folder_loader(out_of_frame)) == 0,
            _np.asarray(_folder_loader(occlusion)) == 0)
        return one, mask



def _read_pfm_file(path, slice_channels=1):
    """`_read_pfm` from a path. **The default is one channel, not two.**

    A `.pfm` carries one channel or three, and how many are sliced out is what
    distinguishes a disparity from a flow — torchvision binds this default with a
    `functools.partial` in the stereo module and leaves the flow module on two.

    **The wrong default is only visible on a three-channel file**, measured: `[:2]` of
    one channel is one channel, so a one-channel disparity read with the flow's
    default is unchanged. On a three-channel one it returns a `(2, h, w)` map whose
    second channel is somebody else's, with every number in the first still right.
    """
    with open(path, "rb") as handle:
        return _read_pfm(handle.read(), slice_channels)


class CarlaStereo(_StereoMatchingDataset):
    """Carla's high-resolution stereo, **one scene per directory.**

    <https://github.com/megvii-research/CREStereo>

    ## Why it was refused and is not

    Its row read *a codec and then another format*. The codec is PNG, read already,
    and the other format is `.pfm` — a text header, a size, a scale whose sign carries
    the endianness, and floats. Twenty lines.

    **The disparity is stored signed and used positive.** The sign is the direction the
    pixel moved, which the file keeps and the dataset does not want; `abs` here is
    torchvision's, and without it half the map is negative and still finite.
    """

    def __init__(self, root, transforms=None):
        super().__init__(root, transforms)
        base = _os.path.join(self.root, "carla-highres", "trainingF")
        self._images = self._scan_pairs(_os.path.join(base, "*", "im0.png"),
                                        _os.path.join(base, "*", "im1.png"))
        self._disparities = self._scan_pairs(
            _os.path.join(base, "*", "disp0GT.pfm"),
            _os.path.join(base, "*", "disp1GT.pfm"))

    def _read_disparity(self, path):
        return _np.abs(_read_pfm_file(path)), None


class ETH3DStereo(_StereoMatchingDataset):
    """ETH3D's low-resolution two-view set.

    <https://www.eth3d.net/datasets>

    ## Why it was refused and is not

    As `CarlaStereo` — PNG and `.pfm`, both read here.

    **The pictures and the ground truth live in different directories**, scene by
    scene: `two_view_training` holds the pair and `two_view_training_gt` holds the
    disparity and its mask. Two sorted lists zipped by position, which is only right
    because the scene directories carry the same names. The mask is a PNG sitting
    beside the `.pfm`, and it is read as a boolean rather than compared to zero — a
    non-zero pixel is valid here, which is the opposite of `SintelStereo`'s
    convention and is why neither is written as "the usual way".
    """

    _has_built_in_disparity_mask = True

    def __init__(self, root, split="train", transforms=None):
        if split not in ("train", "test"):
            raise ValueError(f"Unknown value '{split}' for argument split. Valid "
                             "values are {train, test}.")
        super().__init__(root, transforms)
        base = _os.path.join(self.root, "ETH3D")
        pictures = "two_view_training" if split == "train" else "two_view_test"
        self._images = self._scan_pairs(
            _os.path.join(base, pictures, "*", "im0.png"),
            _os.path.join(base, pictures, "*", "im1.png"))
        if split == "test":
            self._disparities = [(None, None) for _ in self._images]
        else:
            self._disparities = self._scan_pairs(
                _os.path.join(base, "two_view_training_gt", "*", "disp0GT.pfm"), None)

    def _read_disparity(self, path):
        if path is None:
            return None, None
        mask = _os.path.join(_os.path.dirname(path), "mask0nocc.png")
        return (_np.abs(_read_pfm_file(path)),
                _np.asarray(_folder_loader(mask)).astype(bool))


class SceneFlowStereo(_StereoMatchingDataset):
    """The Scene Flow family — `FlyingThings3D`, `Monkaa` and `Driving` behind one
    class.

    <https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html>

    ## Why it was refused and is not

    PNG and `.pfm`, both read here.

    **The three variants differ only in how deep the scene directories nest** —
    `Monkaa` is one level and the other two are three — and that is the whole of what
    `variant` selects. A glob written for one variant finds nothing under another,
    which is a `FileNotFoundError` rather than a wrong answer, so this one is
    load-bearing in the quiet direction.
    """

    _PREFIX = {"Monkaa": ("*",),
               "FlyingThings3D": ("*", "*", "*"),
               "Driving": ("*", "*", "*")}

    def __init__(self, root, variant="FlyingThings3D", pass_name="clean",
                 transforms=None):
        if variant not in self._PREFIX:
            raise ValueError(f"Unknown value '{variant}' for argument variant. Valid "
                             "values are {FlyingThings3D, Driving, Monkaa}.")
        if pass_name not in ("clean", "final", "both"):
            raise ValueError(f"Unknown value '{pass_name}' for argument pass_name. "
                             "Valid values are {clean, final, both}.")
        super().__init__(root, transforms)
        base = _os.path.join(self.root, "SceneFlow", variant)
        middle = self._PREFIX[variant]
        passes = {"clean": ("frames_cleanpass",), "final": ("frames_finalpass",),
                  "both": ("frames_cleanpass", "frames_finalpass")}[pass_name]
        for name in passes:
            self._images += self._scan_pairs(
                _os.path.join(base, name, *middle, "left", "*.png"),
                _os.path.join(base, name, *middle, "right", "*.png"))
            self._disparities += self._scan_pairs(
                _os.path.join(base, "disparity", *middle, "left", "*.pfm"),
                _os.path.join(base, "disparity", *middle, "right", "*.pfm"))

    def _read_disparity(self, path):
        return _np.abs(_read_pfm_file(path)), None


class FlyingThings3D(VisionDataset):
    """FlyingThings3D, **as optical flow** — and the one in this family with a
    genuinely awkward walk.

    <https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html>

    ## Why it was refused and is not

    PNG frames and `.pfm` flows, both read here.

    ## The three loops, and what each gets wrong on its own

    `pass_name` × `camera` × direction, and **direction is the one that is not a
    filter**. `into_future` pairs frame `i` with `i + 1` and takes flow `i`;
    `into_past` pairs `i + 1` with `i` — the frames swapped, not the list reversed —
    and takes flow `i + 1`. A reader that treated `into_past` as the same pairing
    would hand back a flow pointing backwards through a pair that runs forwards, and
    every shape would agree.

    Both directions are always walked; neither is an argument. `camera="both"` and
    `pass_name="both"` multiply the list rather than choosing within it.
    """

    def __init__(self, root, split="train", pass_name="clean", camera="left",
                 transforms=None, loader=None):
        if split not in ("train", "test"):
            raise ValueError(f"Unknown value '{split}' for argument split. Valid "
                             "values are {train, test}.")
        if pass_name not in ("clean", "final", "both"):
            raise ValueError(f"Unknown value '{pass_name}' for argument pass_name. "
                             "Valid values are {clean, final, both}.")
        if camera not in ("left", "right", "both"):
            raise ValueError(f"Unknown value '{camera}' for argument camera. Valid "
                             "values are {left, right, both}.")
        super().__init__(root)
        self.transforms = transforms
        self.loader = _folder_loader if loader is None else loader
        self._image_list = []
        self._flow_list = []

        upper = split.upper()
        passes = {"clean": ("frames_cleanpass",), "final": ("frames_finalpass",),
                  "both": ("frames_cleanpass", "frames_finalpass")}[pass_name]
        cameras = ("left", "right") if camera == "both" else (camera,)
        base = _os.path.join(self.root, "FlyingThings3D")
        for name in passes:
            for eye in cameras:
                for direction in ("into_future", "into_past"):
                    picture_dirs = sorted(
                        _os.path.join(one, eye) for one in
                        sorted(_glob.glob(_os.path.join(base, name, upper, "*", "*"))))
                    flow_dirs = sorted(
                        _os.path.join(one, direction, eye) for one in
                        sorted(_glob.glob(_os.path.join(
                            base, "optical_flow", upper, "*", "*"))))
                    if not picture_dirs or not flow_dirs:
                        raise FileNotFoundError(
                            "Could not find the FlyingThings3D flow images. "
                            "Please make sure the directory structure is correct.")
                    for picture_dir, flow_dir in zip(picture_dirs, flow_dirs):
                        frames = sorted(_glob.glob(_os.path.join(picture_dir, "*.png")))
                        flows = sorted(_glob.glob(_os.path.join(flow_dir, "*.pfm")))
                        for i in range(len(flows) - 1):
                            if direction == "into_future":
                                self._image_list += [[frames[i], frames[i + 1]]]
                                self._flow_list += [flows[i]]
                            else:
                                self._image_list += [[frames[i + 1], frames[i]]]
                                self._flow_list += [flows[i + 1]]

    def __len__(self):
        return len(self._image_list)

    def __getitem__(self, index):
        first = self.loader(self._image_list[index][0])
        second = self.loader(self._image_list[index][1])
        flow = _read_pfm_file(self._flow_list[index], 2) if self._flow_list else None
        if self.transforms is not None:
            first, second, flow, _ = self.transforms(first, second, flow, None)
        return first, second, flow


class FlyingChairs(VisionDataset):
    """FlyingChairs — **one flat directory, and a text file that cuts it in two.**

    <https://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs.en.html>

    ## Why it was refused and is not

    Its row read *a codec and then another format*, and the codec here is **PPM** —
    the cheapest format there is, read since before the row was written. The other
    format is `.flo`, fifteen lines.

    ## Two positions, not two names

    The pictures pair by **position in one sorted list**: item `i` is `images[2i]` and
    `images[2i + 1]`. Nothing in the filenames is matched, so a directory with one
    file missing pairs every later item with the wrong partner and loads.

    `FlyingChairs_train_val.txt` is one integer per pair, `1` for train and `2` for
    val, and it is **required** — the split is not derivable from the directory, and
    without the file torchvision refuses rather than defaulting to all of it. So does
    this.
    """

    _SPLIT_ID = {"train": 1, "val": 2}
    _SPLIT_FILE = "FlyingChairs_train_val.txt"

    def __init__(self, root, split="train", transforms=None, loader=None):
        if split not in self._SPLIT_ID:
            raise ValueError(f"Unknown value '{split}' for argument split. Valid "
                             "values are {train, val}.")
        super().__init__(root)
        self.transforms = transforms
        self.loader = _folder_loader if loader is None else loader
        base = _os.path.join(self.root, "FlyingChairs")
        pictures = sorted(_glob.glob(_os.path.join(base, "data", "*.ppm")))
        flows = sorted(_glob.glob(_os.path.join(base, "data", "*.flo")))
        listing = _os.path.join(base, self._SPLIT_FILE)
        if not _os.path.exists(listing):
            raise FileNotFoundError(
                "The FlyingChairs_train_val.txt file was not found - please download "
                "it from the dataset page (see docstring).")
        with open(listing) as handle:
            wanted = [int(float(line)) for line in handle.read().split()]
        self._image_list = []
        self._flow_list = []
        for i in range(len(flows)):
            if wanted[i] == self._SPLIT_ID[split]:
                self._flow_list += [flows[i]]
                self._image_list += [[pictures[2 * i], pictures[2 * i + 1]]]

    def __len__(self):
        return len(self._image_list)

    def __getitem__(self, index):
        first = self.loader(self._image_list[index][0])
        second = self.loader(self._image_list[index][1])
        with open(self._flow_list[index], "rb") as handle:
            flow = _read_flo(handle.read())
        if self.transforms is not None:
            first, second, flow, _ = self.transforms(first, second, flow, None)
        return first, second, flow



class Middlebury2014Stereo(_StereoMatchingDataset):
    """Middlebury's 2014 set — **thirty-eight named scenes, and a suffix.**

    <https://vision.middlebury.edu/stereo/data/scenes2014/>

    ## Why it was refused and is not

    Its row was the last of the stereo family, and it read *a calibration file per
    scene to parse*. **That was written without opening one.** `calib.txt` sits in
    every scene directory and torchvision never reads it: `calibration` is a
    **directory suffix** — `-perfect` or `-imperfect` — chosen before the glob and
    never opened after. The dataset is PNG and `.pfm`, both read here, and what was
    called a parser is a string.

    ## What is actually here

    - **The split is a list of names**, not a directory. `train`, `additional` and
      `test` are thirty-eight scene names written into the class, and a root that
      holds the directory but none of the names is an error rather than an empty
      dataset — which is the difference between a wrong path and a wrong download.
    - **`test` has no calibration and the others require one.** Passing either the
      wrong way is refused, because the suffix decides which directories exist and a
      silent default would find nothing and say nothing.
    - **Infinite disparities are zeroed and the mask is what remains.** The `.pfm`
      stores unmatched pixels as `inf`; leaving them turns any average into `inf` and
      any subtraction into `nan`, and zeroing them without the mask marks them as
      *touching the camera* instead of *unknown*.
    - **`use_ambient_views` picks at random** among `im1.png`, `im1E.png` and
      `im1L.png` — different exposures of the same right frame. It is off by default,
      and the golden case leaves it off: a case that turned it on would be comparing
      two draws from two libraries' random modules.
    """

    _has_built_in_disparity_mask = True

    splits = {
        "train": ["Adirondack", "Jadeplant", "Motorcycle", "Piano", "Pipes",
                  "Playroom", "Playtable", "Recycle", "Shelves", "Vintage"],
        "additional": ["Backpack", "Bicycle1", "Cable", "Classroom1", "Couch",
                       "Flowers", "Mask", "Shopvac", "Sticks", "Storage", "Sword1",
                       "Sword2", "Umbrella"],
        "test": ["Plants", "Classroom2E", "Classroom2", "Australia", "DjembeL",
                 "CrusadeP", "Crusade", "Hoops", "Bicycle2", "Staircase", "Newkuba",
                 "AustraliaP", "Djembe", "Livingroom", "Computer"],
    }

    _SUFFIXES = {None: [""], "perfect": ["-perfect"], "imperfect": ["-imperfect"],
                 "both": ["-perfect", "-imperfect"]}
    _AMBIENT_VIEWS = ["im1E.png", "im1L.png"]

    def __init__(self, root, split="train", calibration="perfect",
                 use_ambient_views=False, transforms=None, download=False):
        if split not in self.splits:
            raise ValueError(f"Unknown value '{split}' for argument split. Valid "
                             "values are {train, test, additional}.")
        super().__init__(root, transforms)
        self.split = split
        if calibration:
            if calibration not in self._SUFFIXES:
                raise ValueError(f"Unknown value '{calibration}' for argument "
                                 "calibration. Valid values are {perfect, imperfect, "
                                 "both, None}.")
            if split == "test":
                raise ValueError("Split 'test' has only no calibration settings, "
                                 "please set `calibration=None`.")
        elif split != "test":
            raise ValueError(
                f"Split '{split}' has calibration settings, however None was provided "
                f"as an argument.\nSetting calibration to 'perfect' for split "
                f"'{split}'. Available calibration settings are: 'perfect', "
                "'imperfect', 'both'.")
        if download:
            raise ValueError(
                "Middlebury2014Stereo(download=True) is not implemented here.\n"
                "  It is one zip per scene per calibration — seventy-six requests to "
                "vision.middlebury.edu for the two splits that have them — and the "
                "reader above takes the extracted tree.")

        base = _os.path.join(self.root, "Middlebury2014")
        if not _os.path.exists(_os.path.join(base, split)):
            raise FileNotFoundError(
                f"The {split} directory was not found in the provided root directory")
        wanted = self.splits[split]
        if not any(scene.startswith(name)
                   for scene in _os.listdir(_os.path.join(base, split))
                   for name in wanted):
            raise FileNotFoundError(
                f"Provided root folder does not contain any scenes from the {split} "
                "split.")

        for suffix in self._SUFFIXES[calibration]:
            pattern = "*" + suffix
            self._images += self._scan_pairs(
                _os.path.join(base, split, pattern, "im0.png"),
                _os.path.join(base, split, pattern, "im1.png"))
            if split == "test":
                self._disparities = [(None, None) for _ in self._images]
            else:
                self._disparities += self._scan_pairs(
                    _os.path.join(base, split, pattern, "disp0.pfm"),
                    _os.path.join(base, split, pattern, "disp1.pfm"))
        self.use_ambient_views = use_ambient_views

    def _read_img(self, path):
        """The right frame, or one of its other exposures when asked for.

        **The choice is random**, which is why it is off by default: two calls give two
        right pictures against the same left one, and that is the point — it is an
        augmentation written into the reader.
        """
        if _os.path.basename(path) == "im1.png" and self.use_ambient_views:
            folder = _os.path.dirname(path)
            choices = [_os.path.join(folder, name) for name in self._AMBIENT_VIEWS]
            choices = [one for one in choices if _os.path.exists(one)] + [path]
            path = _random.choice(choices)
        return super()._read_img(path)

    def _read_disparity(self, path):
        if path is None:
            return None, None
        one = _np.abs(_read_pfm_file(path))
        one[one == _np.inf] = 0
        return one, (one > 0).squeeze(0)



class Kitti(VisionDataset):
    """KITTI's 2D object set — **a picture and a list of boxes.**

    <http://www.cvlibs.net/datasets/kitti/eval_object.php>

    ## Why it was refused and is not

    Its row read *as above — a codec*, pointing at a sentence about JPEG. **The
    pictures are PNG.** They sit in `image_2`, the same directory name this library
    already reads for `KittiFlow` and `Kitti2015Stereo`, and the labels are one line
    of fifteen space-separated fields per object. Nothing here is a codec; the row was
    a family resemblance to forty-four others that do have one.

    ## The fifteen fields

    They are parsed by position, and the positions are not evenly spaced: one string,
    a float, an **int**, a float, four for the box, three for the size, three for the
    place, one for the angle. `occluded` is a level and not a fraction, which is why
    it alone is an `int`; reading it as a float gives a target that trains and a
    number that means the same thing, until something groups by it.

    **The directory is walked in `os.listdir` order by torchvision and sorted here.**
    That order is the disk's — sorted on one filesystem, hashed on another — so
    torchvision's item order changes with where the tree sits, and sorting costs
    nothing.
    """

    image_dir_name = "image_2"
    labels_dir_name = "label_2"

    def __init__(self, root, train=True, transform=None, target_transform=None,
                 transforms=None, download=False):
        super().__init__(root, transform=transform,
                         target_transform=target_transform, transforms=transforms)
        if download:
            raise ValueError(
                "Kitti(download=True) is not implemented here.\n"
                "  It is two zips from an S3 bucket, and the reader above takes the "
                "extracted tree.")
        self.images = []
        self.targets = []
        self.train = train
        self._location = "training" if self.train else "testing"
        base = _os.path.join(self.root, "Kitti", "raw", self._location)
        if not _os.path.isdir(_os.path.join(base, self.image_dir_name)):
            raise RuntimeError(
                "Dataset not found. You may use download=True to download it.")
        pictures = _os.path.join(base, self.image_dir_name)
        for name in sorted(_os.listdir(pictures)):
            self.images.append(_os.path.join(pictures, name))
            if self.train:
                self.targets.append(_os.path.join(
                    base, self.labels_dir_name, f"{name.split('.')[0]}.txt"))

    def __len__(self):
        return len(self.images)

    def _parse_target(self, index):
        target = []
        with open(self.targets[index]) as handle:
            for line in _csv.reader(handle, delimiter=" "):
                target.append({
                    "type": line[0],
                    "truncated": float(line[1]),
                    "occluded": int(line[2]),
                    "alpha": float(line[3]),
                    "bbox": [float(one) for one in line[4:8]],
                    "dimensions": [float(one) for one in line[8:11]],
                    "location": [float(one) for one in line[11:14]],
                    "rotation_y": float(line[14]),
                })
        return target

    def __getitem__(self, index):
        image = _folder_loader(self.images[index])
        target = self._parse_target(index) if self.train else None
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        return image, target


class PhotoTour(VisionDataset):
    """Brown's patch set — **64×64 patches cut out of BMP sheets.**

    <http://phototour.cs.washington.edu/patches/default.htm>

    ## Why it was refused and is not

    Its row read *as above — a codec*. The pictures are **BMP**: a file header, an
    information header, a palette and the rows, with no compression involved. It is
    the cheapest format this library reads after PPM, and it was refused for
    forty-four other datasets' reason.

    ## Two shapes, one dataset

    `train=True` gives **one patch**; `train=False` gives **two patches and whether
    they match**. That is not a split of the same items — it is a different task off
    the same bytes, and a reader that returned patches for both would leave the
    matching half untestable.

    ## What is not here

    torchvision caches the decoded patches to a `.pt` beside the data and reads that
    afterwards. This builds from the sheets every time. A `.pt` is `torch.save`'s
    pickle, and writing one would mean writing a format whose reader this library has
    deliberately never had.
    """

    image_ext = "bmp"
    info_file = "info.txt"
    matches_files = "m50_100000_100000_0.txt"
    _PATCH = 64

    lens = {"notredame": 468159, "yosemite": 633587, "liberty": 450092,
            "liberty_harris": 379587, "yosemite_harris": 450912,
            "notredame_harris": 325295}
    means = {"notredame": 0.4854, "yosemite": 0.4844, "liberty": 0.4437,
             "notredame_harris": 0.4854, "yosemite_harris": 0.4844,
             "liberty_harris": 0.4437}
    stds = {"notredame": 0.1864, "yosemite": 0.1818, "liberty": 0.2019,
            "notredame_harris": 0.1864, "yosemite_harris": 0.1818,
            "liberty_harris": 0.2019}

    def __init__(self, root, name, train=True, transform=None, download=False):
        super().__init__(root, transform=transform)
        self.name = name
        self.data_dir = _os.path.join(self.root, name)
        self.train = train
        self.mean = self.means[name]
        self.std = self.stds[name]
        if download:
            raise ValueError(
                "PhotoTour(download=True) is not implemented here.\n"
                "  It is one zip per subset from a university host, and the reader "
                "above takes the extracted directory.")
        self.data = self._read_patches()
        self.labels = self._read_info()
        self.matches = self._read_matches()

    def _read_patches(self):
        """**Each sheet is a grid of patches**, read left to right and then down.

        Row-major within a sheet and sheets in sorted order: the patch index is a
        position in that traversal and nothing in a filename gives it, so a reader that
        took the sheets in directory order would number every patch differently and
        still return patches.
        """
        patches = []
        sheets = sorted(name for name in _os.listdir(self.data_dir)
                        if name.endswith(self.image_ext))
        for sheet in sheets:
            picture = _np.asarray(_folder_loader(_os.path.join(self.data_dir, sheet)))
            height, width = picture.shape[:2]
            for top in range(0, height, self._PATCH):
                for left in range(0, width, self._PATCH):
                    patches.append(picture[top:top + self._PATCH,
                                           left:left + self._PATCH])
        return _np.asarray(patches[:self.lens[self.name]], dtype=_np.uint8)

    def _read_info(self):
        """The 3D point each patch belongs to — **the first field only**, and the
        second is the camera, which is not a label."""
        with open(_os.path.join(self.data_dir, self.info_file)) as handle:
            return _np.asarray([int(line.split()[0]) for line in handle],
                               dtype=_np.int64)

    def _read_matches(self):
        """`(first patch, second patch, whether they are the same point)`.

        **The answer is a comparison, not a column.** Each line names two patches and
        the point each belongs to; the label is whether those two point ids agree, and
        no field in the file holds it.
        """
        matches = []
        with open(_os.path.join(self.data_dir, self.matches_files)) as handle:
            for line in handle:
                parts = line.split()
                matches.append([int(parts[0]), int(parts[3]),
                                int(parts[1] == parts[4])])
        return _np.asarray(matches, dtype=_np.int64)

    def __len__(self):
        return len(self.data if self.train else self.matches)

    def __getitem__(self, index):
        if self.train:
            data = self.data[index]
            if self.transform is not None:
                data = self.transform(data)
            return data
        first, second, same = self.matches[index]
        one, two = self.data[first], self.data[second]
        if self.transform is not None:
            one = self.transform(one)
            two = self.transform(two)
        return one, two, same



class Country211(ImageFolder):
    """CLIP's country set — **the label is which country the photograph was taken in.**

    <https://github.com/openai/CLIP/blob/main/data/country211.md>

    ## Why it was refused and is not

    Its row read *as above — a codec*, inherited from a sentence about JPEG. The
    pictures **are** JPEG — fetched and checked, the first bytes of `country211.tgz`
    are `ff d8` — and that is still not what this class does. It is `ImageFolder`
    pointed at one of three directories, which is the same shape `ImageFolder` itself
    was let in for: the walk is here, the codec is in the loader, and `loader=` is
    torchvision's own parameter rather than an escape hatch invented here.

    So this loads on a tree of PNG. **On the real one it stops before the codec** and
    says so differently from the rest of this file: `ImageFolder` filters by
    `IMG_EXTENSIONS` before any loader runs, so a directory of `.jpg` is *no valid
    file for the classes …, supported extensions are …* rather than *that is JPEG*.
    Both name what is read here; the first names it earlier.
    """

    _SPLITS = ("train", "valid", "test")

    def __init__(self, root, split="train", transform=None, target_transform=None,
                 download=False, loader=None):
        if split not in self._SPLITS:
            raise ValueError(f"Unknown value '{split}' for argument split. Valid "
                             "values are {train, valid, test}.")
        if download:
            raise ValueError(
                "Country211(download=True) is not implemented here.\n"
                "  It is one 2.6GB archive; the reader above takes the extracted tree.")
        self._split = split
        base = _os.path.join(_os.path.expanduser(root), "country211")
        if not _os.path.isdir(base):
            raise RuntimeError(
                "Dataset not found. You can use download=True to download it")
        super().__init__(_os.path.join(base, split), transform=transform,
                         target_transform=target_transform,
                         loader=_folder_loader if loader is None else loader)
        self.root = _os.path.expanduser(root)


class EuroSAT(ImageFolder):
    """Sentinel-2 land-cover patches, **RGB version.**

    <https://github.com/phelber/eurosat>

    As `Country211`: `ImageFolder` pointed at `eurosat/2750`, with ten class
    directories. The multispectral version is a different dataset and not this one.
    """

    def __init__(self, root, transform=None, target_transform=None, download=False,
                 loader=None):
        if download:
            raise ValueError(
                "EuroSAT(download=True) is not implemented here.\n"
                "  The reader above takes the extracted tree.")
        base = _os.path.expanduser(root)
        self._base_folder = _os.path.join(base, "eurosat")
        self._data_folder = _os.path.join(self._base_folder, "2750")
        if not _os.path.exists(self._data_folder):
            raise RuntimeError(
                "Dataset not found. You can use download=True to download it")
        super().__init__(self._data_folder, transform=transform,
                         target_transform=target_transform,
                         loader=_folder_loader if loader is None else loader)
        self.root = base


class DTD(VisionDataset):
    """Describable Textures — **forty-seven words for what a surface looks like.**

    <https://www.robots.ox.ac.uk/~vgg/data/dtd/>

    ## The partition is not a split

    `split` chooses train, val or test; `partition` chooses **which of ten shufflings**
    of the same images that split refers to. Combining all three splits gives every
    image whichever partition is asked for, which is what makes the argument safe to
    get wrong: a reader that ignored it returns a dataset of the right size, drawn
    from the right pictures, and cross-validating on it measures nothing.

    The classes come from the **split file's own paths**, so they are the classes
    present in that file rather than the forty-seven on disk.
    """

    _SPLITS = ("train", "val", "test")
    _PARTITIONS = 10

    def __init__(self, root, split="train", partition=1, transform=None,
                 target_transform=None, download=False, loader=None):
        if split not in self._SPLITS:
            raise ValueError(f"Unknown value '{split}' for argument split. Valid "
                             "values are {train, val, test}.")
        if not isinstance(partition, int) or not 1 <= partition <= self._PARTITIONS:
            raise ValueError(
                "Parameter 'partition' should be an integer with `1 <= partition <= "
                f"10`, but got {partition} instead")
        if download:
            raise ValueError(
                "DTD(download=True) is not implemented here.\n"
                "  The reader above takes the extracted tree.")
        self._split = split
        self._partition = partition
        super().__init__(root, transform=transform, target_transform=target_transform)
        self._base_folder = _os.path.join(self.root, type(self).__name__.lower())
        self._data_folder = _os.path.join(self._base_folder, "dtd")
        self._meta_folder = _os.path.join(self._data_folder, "labels")
        self._images_folder = _os.path.join(self._data_folder, "images")
        if not _os.path.isdir(self._data_folder):
            raise RuntimeError(
                "Dataset not found. You can use download=True to download it")
        self.loader = _folder_loader if loader is None else loader
        self._image_files = []
        classes = []
        with open(_os.path.join(self._meta_folder,
                                f"{split}{partition}.txt")) as handle:
            for line in handle:
                folder, name = line.strip().split("/")
                self._image_files.append(
                    _os.path.join(self._images_folder, folder, name))
                classes.append(folder)
        self.classes = sorted(set(classes))
        self.class_to_idx = dict(zip(self.classes, range(len(self.classes))))
        self._labels = [self.class_to_idx[one] for one in classes]

    def __len__(self):
        return len(self._image_files)

    def __getitem__(self, index):
        image = self.loader(self._image_files[index])
        label = self._labels[index]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

    def extra_repr(self):
        return f"split={self._split}, partition={self._partition}"


class Food101(VisionDataset):
    """A hundred and one dishes, **and the training half was never cleaned.**

    <https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/>

    The split is a JSON of class name to a list of relative paths, and **the order of
    the items is the order of that file** rather than sorted. The classes are sorted
    and the labels index into them, so a reader that took the JSON's key order for the
    class list would give every label a different number and nothing would look wrong.
    """

    _SPLITS = ("train", "test")

    def __init__(self, root, split="train", transform=None, target_transform=None,
                 download=False, loader=None):
        if split not in self._SPLITS:
            raise ValueError(f"Unknown value '{split}' for argument split. Valid "
                             "values are {train, test}.")
        if download:
            raise ValueError(
                "Food101(download=True) is not implemented here.\n"
                "  It is a 5GB archive; the reader above takes the extracted tree.")
        super().__init__(root, transform=transform, target_transform=target_transform)
        self._split = split
        self._base_folder = _os.path.join(self.root, "food-101")
        self._meta_folder = _os.path.join(self._base_folder, "meta")
        self._images_folder = _os.path.join(self._base_folder, "images")
        if not all(_os.path.isdir(one) for one in
                   (self._meta_folder, self._images_folder)):
            raise RuntimeError(
                "Dataset not found. You can use download=True to download it")
        self.loader = _folder_loader if loader is None else loader
        with open(_os.path.join(self._meta_folder, f"{split}.json")) as handle:
            metadata = _json.loads(handle.read())
        self.classes = sorted(metadata.keys())
        self.class_to_idx = dict(zip(self.classes, range(len(self.classes))))
        self._labels = []
        self._image_files = []
        for name, relative in metadata.items():
            self._labels += [self.class_to_idx[name]] * len(relative)
            self._image_files += [
                _os.path.join(self._images_folder, *f"{one}.jpg".split("/"))
                for one in relative]

    def __len__(self):
        return len(self._image_files)

    def __getitem__(self, index):
        image = self.loader(self._image_files[index])
        label = self._labels[index]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

    def extra_repr(self):
        return f"split={self._split}"


class SUN397(VisionDataset):
    """Scene UNderstanding — **three hundred and ninety-seven kinds of place.**

    <https://vision.princeton.edu/projects/2010/SUN/>

    Two things here are easy to write differently and hard to see:

    - **The class name has its first three characters cut off.** `ClassName.txt` reads
      `/a/abbey`, and the class is `abbey`; keeping the prefix gives 397 classes with
      the right count and the wrong names.
    - **The label comes from the path, minus its first part.** A scene at
      `a/abbey/sun_x.jpg` is `abbey`, and one at `a/apartment_building/outdoor/...`
      is `apartment_building/outdoor` — the letter directory is dropped and everything
      after it is kept, so a reader taking the parent directory alone maps every
      nested scene to its last component.
    """

    def __init__(self, root, transform=None, target_transform=None, download=False,
                 loader=None):
        if download:
            raise ValueError(
                "SUN397(download=True) is not implemented here.\n"
                "  It is a 37GB archive; the reader above takes the extracted tree.")
        super().__init__(root, transform=transform, target_transform=target_transform)
        self._data_dir = _os.path.join(self.root, "SUN397")
        if not _os.path.exists(_os.path.join(self._data_dir, "ClassName.txt")):
            raise RuntimeError(
                "Dataset not found. You can use download=True to download it")
        self.loader = _folder_loader if loader is None else loader
        with open(_os.path.join(self._data_dir, "ClassName.txt")) as handle:
            self.classes = [line[3:].strip() for line in handle]
        self.class_to_idx = dict(zip(self.classes, range(len(self.classes))))
        self._image_files = []
        for folder, _folders, names in _os.walk(self._data_dir):
            for name in names:
                if name.startswith("sun_") and name.endswith(".jpg"):
                    self._image_files.append(_os.path.join(folder, name))
        self._labels = [
            self.class_to_idx["/".join(
                _os.path.relpath(one, self._data_dir).split(_os.sep)[1:-1])]
            for one in self._image_files]

    def __len__(self):
        return len(self._image_files)

    def __getitem__(self, index):
        image = self.loader(self._image_files[index])
        label = self._labels[index]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


class FGVCAircraft(VisionDataset):
    """Aircraft, **labelled at three heights at once.**

    <https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/>

    `annotation_level` picks `variant`, `family` or `manufacturer`, and it chooses
    **two files together** — the class list and the label file — so the labels always
    index into the list they were written against. Mixing one level's classes with
    another's labels is a `KeyError` here rather than a silently shifted target, which
    is why the two lookups are built from the same argument.

    Each label line is `name` then the rest of the line: **the class name contains
    spaces**, so it splits once and not on every space. `Boeing 737-700` is one class.
    """

    _SPLITS = ("train", "val", "trainval", "test")
    _LEVELS = {"variant": "variants.txt", "family": "families.txt",
               "manufacturer": "manufacturers.txt"}

    def __init__(self, root, split="trainval", annotation_level="variant",
                 transform=None, target_transform=None, download=False, loader=None):
        if split not in self._SPLITS:
            raise ValueError(f"Unknown value '{split}' for argument split. Valid "
                             "values are {train, val, trainval, test}.")
        if annotation_level not in self._LEVELS:
            raise ValueError(
                f"Unknown value '{annotation_level}' for argument annotation_level. "
                "Valid values are {variant, family, manufacturer}.")
        if download:
            raise ValueError(
                "FGVCAircraft(download=True) is not implemented here.\n"
                "  The reader above takes the extracted tree.")
        super().__init__(root, transform=transform, target_transform=target_transform)
        self._split = split
        self._annotation_level = annotation_level
        self._data_path = _os.path.join(self.root, "fgvc-aircraft-2013b")
        if not _os.path.isdir(self._data_path):
            raise RuntimeError(
                "Dataset not found. You can use download=True to download it")
        self.loader = _folder_loader if loader is None else loader
        data = _os.path.join(self._data_path, "data")
        with open(_os.path.join(data, self._LEVELS[annotation_level])) as handle:
            self.classes = [line.strip() for line in handle]
        self.class_to_idx = dict(zip(self.classes, range(len(self.classes))))
        self._image_files = []
        self._labels = []
        with open(_os.path.join(
                data, f"images_{annotation_level}_{split}.txt")) as handle:
            for line in handle:
                name, label = line.strip().split(" ", 1)
                self._image_files.append(
                    _os.path.join(data, "images", f"{name}.jpg"))
                self._labels.append(self.class_to_idx[label])

    def __len__(self):
        return len(self._image_files)

    def __getitem__(self, index):
        image = self.loader(self._image_files[index])
        label = self._labels[index]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label



class Imagenette(VisionDataset):
    """Ten ImageNet classes that are **easy to tell apart on purpose.**

    <https://github.com/fastai/imagenette>

    ## Why it was refused and is not

    Its row read *as above — a codec*, and its pictures are JPEG — fetched and checked
    before this was written. That is still not what the class does: it walks
    directories named by WordNet id, maps each to a class name, and calls
    `self.loader`. `loader` is torchvision's own parameter here.

    ## The class name is a tuple, and `class_to_idx` is not its inverse

    Each WordNet id carries **every** name for the thing — `n03425413` is a gas pump, a
    gasoline pump, a petrol pump and an island dispenser — so `classes` is a list of
    tuples and `class_to_idx` has more keys than there are classes, four of them
    pointing at that one index. A reader that took the first name of each would build
    a dictionary that looks right and cannot look up three names in four.

    The label comes from the **sorted WordNet ids**, not from the class names, which is
    a different order: `n01440764` is tench and sorts first, while `cassette player`
    would.
    """

    _SPLITS = ("train", "val")
    _SIZES = {"full": "imagenette2", "320px": "imagenette2-320",
              "160px": "imagenette2-160"}
    _WNID_TO_CLASS = {
        "n01440764": ("tench", "Tinca tinca"),
        "n02102040": ("English springer", "English springer spaniel"),
        "n02979186": ("cassette player",),
        "n03000684": ("chain saw", "chainsaw"),
        "n03028079": ("church", "church building"),
        "n03394916": ("French horn", "horn"),
        "n03417042": ("garbage truck", "dustcart"),
        "n03425413": ("gas pump", "gasoline pump", "petrol pump", "island dispenser"),
        "n03445777": ("golf ball",),
        "n03888257": ("parachute", "chute"),
    }
    _EXTENSION = ".jpeg"

    def __init__(self, root, split="train", size="full", download=False,
                 transform=None, target_transform=None, loader=None):
        super().__init__(root, transform=transform, target_transform=target_transform)
        if split not in self._SPLITS:
            raise ValueError(f"Unknown value '{split}' for argument split. Valid "
                             "values are {train, val}.")
        if size not in self._SIZES:
            raise ValueError(f"Unknown value '{size}' for argument size. Valid "
                             "values are {full, 320px, 160px}.")
        if download:
            raise ValueError(
                "Imagenette(download=True) is not implemented here.\n"
                "  The reader above takes the extracted tree.")
        self._split = split
        self._size = size
        self._size_root = _os.path.join(self.root, self._SIZES[size])
        self._image_root = _os.path.join(self._size_root, split)
        if not _os.path.exists(self._size_root):
            raise RuntimeError(
                "Dataset not found. You can use download=True to download it.")
        self.loader = _folder_loader if loader is None else loader

        self.wnids = sorted(name for name in _os.listdir(self._image_root)
                            if _os.path.isdir(_os.path.join(self._image_root, name)))
        self.wnid_to_idx = {wnid: i for i, wnid in enumerate(self.wnids)}
        self.classes = [self._WNID_TO_CLASS[wnid] for wnid in self.wnids]
        self.class_to_idx = {name: index
                             for wnid, index in self.wnid_to_idx.items()
                             for name in self._WNID_TO_CLASS[wnid]}
        self._samples = []
        for wnid in self.wnids:
            folder = _os.path.join(self._image_root, wnid)
            for name in sorted(_os.listdir(folder)):
                if name.lower().endswith(self._EXTENSION):
                    self._samples.append((_os.path.join(folder, name),
                                          self.wnid_to_idx[wnid]))

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, index):
        path, label = self._samples[index]
        image = self.loader(path)
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return image, label


class _Flickr8kParser(_html.HTMLParser):
    """The captions arrive as **an HTML page**, and this walks it.

    Not a document format anybody chose — the Flickr8k release ships its annotations as
    a table of links and list items, and torchvision reads it with the standard
    library's parser. So does this.

    **The link text names a directory and the file is found by glob**, because the page
    refers to an image whose extension it does not give. `Image Not Found` is a real
    row in that table and clears the current image rather than raising: the captions
    that follow it belong to nothing and are dropped.
    """

    def __init__(self, root):
        super().__init__()
        self.root = root
        self.annotations = {}
        self.in_table = False
        self.current_tag = None
        self.current_img = None

    def handle_starttag(self, tag, attrs):
        self.current_tag = tag
        if tag == "table":
            self.in_table = True

    def handle_endtag(self, tag):
        self.current_tag = None
        if tag == "table":
            self.in_table = False

    def handle_data(self, data):
        if not self.in_table:
            return
        if data == "Image Not Found":
            self.current_img = None
        elif self.current_tag == "a":
            found = _glob.glob(_os.path.join(self.root, data.split("/")[-2] + "_*.jpg"))
            self.current_img = found[0]
            self.annotations[self.current_img] = []
        elif self.current_tag == "li" and self.current_img:
            self.annotations[self.current_img].append(data.strip())


class Flickr8k(VisionDataset):
    """Eight thousand photographs, **five captions each.**

    <http://hockenmaier.cs.illinois.edu/8k-pictures.html>

    Its row read *as above — a codec*. What the class reads is an **HTML page** of
    captions, walked with the standard library's parser; the pictures go to
    `self.loader`, which is torchvision's own parameter.

    **The item's id is a path, not a name.** `ids` is the sorted list of the file paths
    the parser resolved, so `__getitem__` opens the key itself rather than joining
    anything to `root`. That is unlike `Flickr30k` next door, which stores names and
    joins — two classes for one dataset shape, differing in the one place it matters.
    """

    def __init__(self, root, ann_file, transform=None, target_transform=None,
                 loader=None):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.ann_file = _os.path.expanduser(ann_file)
        parser = _Flickr8kParser(self.root)
        with open(self.ann_file) as handle:
            parser.feed(handle.read())
        self.annotations = parser.annotations
        self.ids = sorted(self.annotations.keys())
        self.loader = _folder_loader if loader is None else loader

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        name = self.ids[index]
        image = self.loader(name)
        if self.transform is not None:
            image = self.transform(image)
        target = self.annotations[name]
        if self.target_transform is not None:
            target = self.target_transform(target)
        return image, target


class Flickr30k(VisionDataset):
    """Thirty thousand photographs and their captions.

    <http://web.engr.illinois.edu/~bplumme2/Flickr30kEntities/>

    As `Flickr8k`, with a plainer annotation file: one line of `id#n<TAB>caption`.

    **The last two characters of the id are cut off**, because `1000092795.jpg#4` names
    the fifth caption of one photograph and the photograph is `1000092795.jpg`. A
    reader that split on `#` instead would agree on this dataset and disagree on any
    file whose caption index reached ten — the slice is what torchvision does, and it
    is written here because it is the kind of thing that looks like a mistake.
    """

    _INDEX_SUFFIX = 2

    def __init__(self, root, ann_file, transform=None, target_transform=None,
                 loader=None):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.ann_file = _os.path.expanduser(ann_file)
        self.annotations = {}
        with open(self.ann_file) as handle:
            for line in handle:
                name, caption = line.strip().split("\t")
                self.annotations.setdefault(
                    name[:-self._INDEX_SUFFIX], []).append(caption)
        self.ids = sorted(self.annotations.keys())
        self.loader = _folder_loader if loader is None else loader

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        name = self.ids[index]
        image = self.loader(_os.path.join(self.root, name))
        if self.transform is not None:
            image = self.transform(image)
        target = self.annotations[name]
        if self.target_transform is not None:
            target = self.target_transform(target)
        return image, target



class StanfordCars(VisionDataset):
    """Sixteen thousand photographs of cars, **labelled by make, model and year.**

    <https://www.kaggle.com/datasets/jessicali9530/stanford-cars-dataset>

    ## Why it was refused and is not

    Its row said *a codec*, then said **not yet** with something genuinely missing:
    the annotations are a `.mat` holding a **struct array**, and `_mat_read` read
    numeric arrays only — measured, it returned no keys at all for one. That was a
    reader to extend rather than a codec to write, and it is extended.

    ## Two things off by one in different directions

    - **The class in the file starts at 1** and the label here starts at 0, so one is
      subtracted. Leaving it gives a dataset whose labels run 1 to 196 against a class
      list of 196 names — every prediction shifted by one make, and the accuracy
      identical.
    - **The two splits keep their annotations in different places.** `train`'s is
      inside `devkit`; `test`'s is beside it, in a file whose name says
      `withlabels`, because the devkit's own test file has none.

    `download` is refused the way torchvision refuses it — the original URL is gone —
    rather than by this library's usual reason.
    """

    _SPLITS = ("train", "test")

    def __init__(self, root, split="train", transform=None, target_transform=None,
                 download=False, loader=None):
        super().__init__(root, transform=transform, target_transform=target_transform)
        if split not in self._SPLITS:
            raise ValueError(f"Unknown value '{split}' for argument split. Valid "
                             "values are {train, test}.")
        self._split = split
        self._base_folder = _os.path.join(str(root), "stanford_cars")
        devkit = _os.path.join(self._base_folder, "devkit")
        if split == "train":
            self._annotations_mat_path = _os.path.join(
                devkit, "cars_train_annos.mat")
            self._images_base_path = _os.path.join(self._base_folder, "cars_train")
        else:
            self._annotations_mat_path = _os.path.join(
                self._base_folder, "cars_test_annos_withlabels.mat")
            self._images_base_path = _os.path.join(self._base_folder, "cars_test")
        if download:
            self.download()
        if not self._check_exists():
            raise RuntimeError("Dataset not found.")
        self.loader = _folder_loader if loader is None else loader

        with open(self._annotations_mat_path, "rb") as handle:
            annotations = _mat_read(handle.read())["annotations"]
        if isinstance(annotations, dict):        # a file holding one car
            annotations = [annotations]
        self._samples = [
            (_os.path.join(self._images_base_path, str(one["fname"])),
             int(_np.asarray(one["class"]).reshape(-1)[0]) - 1)
            for one in annotations]
        with open(_os.path.join(devkit, "cars_meta.mat"), "rb") as handle:
            self.classes = list(_mat_read(handle.read())["class_names"])
        self.class_to_idx = {name: i for i, name in enumerate(self.classes)}

    def _check_exists(self):
        return (_os.path.isdir(_os.path.join(self._base_folder, "devkit"))
                and _os.path.exists(self._annotations_mat_path)
                and _os.path.isdir(self._images_base_path))

    def download(self):
        raise ValueError("The original URL is broken so the StanfordCars dataset "
                         "cannot be downloaded anymore.")

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, index):
        path, target = self._samples[index]
        image = self.loader(path)
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return image, target



_CATEGORIES_2021 = ("kingdom", "phylum", "class", "order", "family", "genus")


class INaturalist(VisionDataset):
    """iNaturalist, **and three releases that do not agree on anything.**

    <https://github.com/visipedia/inat_comp>

    ## Why it was refused and is not

    Its row read *as above — a codec*, and there is no annotation file here at all:
    **the taxonomy is in the directory names**, and the class walks them. `loader` is
    torchvision's own parameter, and it is the only place a format is read.

    ## What `version` actually chooses

    Not a split — **a different scheme for what a directory means**, and all three
    build the same `index` at the end, so getting it wrong gives a dataset that loads:

    - **2021** names each directory `00000_Animalia_Chordata_…_Trichodes_apiarius`:
      eight underscore-separated pieces, of which six are the ranks. The leading number
      must equal the directory's own position, checked rather than trusted — a missing
      category shifts every later label by one and nothing else changes.
    - **2018 and 2019** nest supercategory over a **numeric** subdirectory, and that
      number is the category id. The directories are sparse, so the list is grown to
      fit and a gap in it is an error: a missing id read as absent rather than as
      "renumber the rest" is the difference between a refusal and a silent shift.
    - **2017** has the same nesting with **non-numeric** subdirectories, so the id is
      the position instead. One version reading the other's rule either raises on a
      name that is not a number, or renumbers everything.

    `target_type` may be a list, and then the target is a tuple in that order.
    `full` is the category id itself; the named ranks index into per-rank tables built
    while walking, so two species in one genus share a genus id and not a category one.
    """

    _VERSIONS = ("2017", "2018", "2019", "2021_train", "2021_train_mini", "2021_valid")
    _PIECES_2021 = 8
    _RANKS_2021 = 6

    def __init__(self, root, version="2021_train", target_type="full", transform=None,
                 target_transform=None, download=False, loader=None):
        if version not in self._VERSIONS:
            raise ValueError(f"Unknown value '{version}' for argument version. Valid "
                             "values are {" + ", ".join(self._VERSIONS) + "}.")
        self.version = version
        super().__init__(_os.path.join(root, version), transform=transform,
                         target_transform=target_transform)
        if download:
            raise ValueError(
                "INaturalist(download=True) is not implemented here.\n"
                "  The archives run to hundreds of gigabytes; the reader above takes "
                "the extracted tree.")
        if not (_os.path.exists(self.root) and _os.listdir(self.root)):
            raise RuntimeError(
                "Dataset not found or corrupted. You can use download=True to "
                "download it")

        self.all_categories = []
        self.categories_index = {}
        self.categories_map = []
        if not isinstance(target_type, list):
            target_type = [target_type]
        if self.version[:4] == "2021":
            allowed = ("full",) + _CATEGORIES_2021
            self._init_2021()
        else:
            allowed = ("full", "super")
            self._init_pre2021()
        for one in target_type:
            if one not in allowed:
                raise ValueError(
                    f"Unknown value '{one}' for argument target_type. Valid values "
                    "are {" + ", ".join(allowed) + "}.")
        self.target_type = list(target_type)

        self.index = []
        for position, name in enumerate(self.all_categories):
            for filename in sorted(_os.listdir(_os.path.join(self.root, name))):
                self.index.append((position, filename))
        self.loader = _folder_loader if loader is None else loader

    def _init_2021(self):
        self.all_categories = sorted(_os.listdir(self.root))
        self.categories_index = {rank: {} for rank in _CATEGORIES_2021}
        for position, name in enumerate(self.all_categories):
            pieces = name.split("_")
            if len(pieces) != self._PIECES_2021:
                raise RuntimeError(
                    f"Unexpected category name {name}, wrong number of pieces")
            if pieces[0] != f"{position:05d}":
                raise RuntimeError(
                    f"Unexpected category id {pieces[0]}, expecting {position:05d}")
            row = {}
            for rank, value in zip(_CATEGORIES_2021,
                                   pieces[1:1 + self._RANKS_2021]):
                table = self.categories_index[rank]
                row[rank] = table.setdefault(value, len(table))
            self.categories_map.append(row)

    def _init_pre2021(self):
        self.categories_index = {"super": {}}
        running = 0
        for above, supercategory in enumerate(sorted(_os.listdir(self.root))):
            self.categories_index["super"][supercategory] = above
            folder = _os.path.join(self.root, supercategory)
            for name in sorted(_os.listdir(folder)):
                if self.version == "2017":
                    where = running
                    running += 1
                else:
                    try:
                        where = int(name)
                    except ValueError:
                        raise RuntimeError(
                            f"Unexpected non-numeric dir name: {name}")
                if where >= len(self.categories_map):
                    grow = where - len(self.categories_map) + 1
                    self.categories_map.extend([{}] * grow)
                    self.all_categories.extend([""] * grow)
                if self.categories_map[where]:
                    raise RuntimeError(f"Duplicate category {name}")
                self.categories_map[where] = {"super": above}
                self.all_categories[where] = _os.path.join(supercategory, name)
        for where, row in enumerate(self.categories_map):
            if not row:
                raise RuntimeError(f"Missing category {where}")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        category, filename = self.index[index]
        image = self.loader(_os.path.join(
            self.root, self.all_categories[category], filename))
        target = [category if one == "full" else self.categories_map[category][one]
                  for one in self.target_type]
        target = tuple(target) if len(target) > 1 else target[0]
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return image, target

    def category_name(self, category_type, category_id):
        """The name behind an id. **`full` indexes a list and the ranks search a
        table**, because the rank tables are built name-to-id while walking and nothing
        reverses them — which is fine at the sizes involved and worth saying, since a
        caller reading this as a lookup would expect it to be cheap."""
        if category_type == "full":
            return self.all_categories[category_id]
        if category_type not in self.categories_index:
            raise ValueError(f"Invalid category type '{category_type}'")
        for name, where in self.categories_index[category_type].items():
            if where == category_id:
                return name
        raise ValueError(
            f"Invalid category id {category_id} for {category_type}")


class CLEVRClassification(VisionDataset):
    """CLEVR, **counted**: the label is how many objects are in the scene.

    <https://cs.stanford.edu/people/jcjohns/clevr/>

    ## Why this one is here and its neighbours are not

    The gap table said of this name *its pictures are JPEG or PNG and numpy decodes
    neither*. Half of that was true when it was written and the other half stopped
    being true when `_png_read` landed — and which half applied was never checked,
    because checking meant a 19GB download.

    **It did not.** A zip keeps its file list at the end, and the host serves ranges:
    70KB out of 19,021,600,724 reads the central directory, and every one of the
    100,015 entries is a `.png`, a `.json` or a `.txt`. So there was no codec wall
    here at all — the sentence that stood in this slot for months was about a file
    format nobody had looked at.

    ## What the label is

    `scenes/CLEVR_{split}_scenes.json` holds one entry per picture with its objects
    listed; the label is `len(objects)`. **`test` has no scenes file**, so its labels
    are `None` — torchvision does the same, and a reader that returned 0 there would
    train on a class that does not exist.

    The pictures are matched to scenes **by filename and not by position**. The
    directory listing is sorted and the JSON is not, so pairing them by index gives
    every picture the wrong count while every shape stays right.
    """

    _URL = "https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip"
    _MD5 = "b11922020e72d0cd9154779b2d3d07d2"

    def __init__(self, root, split="train", transform=None, target_transform=None,
                 download=False, loader=None):
        if split not in ("train", "val", "test"):
            raise ValueError(f"Unknown value '{split}' for argument split. Valid "
                             "values are {train, val, test}.")
        super().__init__(root, transform=transform, target_transform=target_transform)
        self._split = split
        self.loader = _folder_loader if loader is None else loader
        self._base_folder = _os.path.join(self.root, "clevr")
        self._data_folder = _os.path.join(self._base_folder, "CLEVR_v1.0")
        if download:
            self.download()
        if not self._check_exists():
            raise RuntimeError("Dataset not found or corrupted. You can use "
                               "download=True to download it")

        folder = _os.path.join(self._data_folder, "images", self._split)
        self._image_files = sorted(
            _os.path.join(folder, name) for name in _os.listdir(folder)
            if not name.startswith("."))
        if self._split != "test":
            scenes = _os.path.join(self._data_folder, "scenes",
                                   f"CLEVR_{self._split}_scenes.json")
            with open(scenes) as handle:
                content = _json.load(handle)
            counts = {scene["image_filename"]: len(scene["objects"])
                      for scene in content["scenes"]}
            self._labels = [counts[_os.path.basename(p)] for p in self._image_files]
        else:
            self._labels = [None] * len(self._image_files)

    def _check_exists(self):
        return _os.path.isdir(self._data_folder)

    def download(self):
        """**The archive is 19GB and this refuses to fetch it silently.**

        Every other `download()` here streams tens or hundreds of megabytes; this one
        is two orders of magnitude larger, and a `download=True` typed once by
        somebody following a tutorial should not start it without saying so. The URL
        and the checksum are named so the fetch can be made deliberately.
        """
        if self._check_exists():
            return
        raise RuntimeError(
            f"CLEVR is {19021600724 / 1e9:.0f}GB and is not fetched automatically.\n"
            f"  {self._URL}\n"
            f"  md5 {self._MD5}\n"
            f"  Unpack it so that {self._data_folder} exists, then construct without "
            "download=True.")

    def __len__(self):
        return len(self._image_files)

    def __getitem__(self, index):
        picture = self.loader(self._image_files[index])
        target = self._labels[index]
        if self.transforms is not None:
            picture, target = self.transforms(picture, target)
        return picture, target

    def extra_repr(self):
        # **`split=train`, not `Split: train`.** Every other dataset here prints the
        # capitalised form and this one does not — torchvision writes this line per
        # class and CLEVR's was written by a different hand. Copied rather than
        # tidied: the printed line is what a reader compares.
        return f"split={self._split}"


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
              "CIFAR10", "CIFAR100", "FakeData", "SEMEION", "USPS", "DatasetFolder", "ImageFolder", "CLEVRClassification", "Sintel", "RenderedSST2", "KittiFlow", "HD1K",
              "Kitti2012Stereo", "Kitti2015Stereo", "InStereo2k", "SintelStereo",
              "CarlaStereo", "ETH3DStereo", "SceneFlowStereo", "FlyingThings3D",
              "Middlebury2014Stereo", "Kitti", "PhotoTour",
              "Country211", "EuroSAT", "DTD", "Food101", "SUN397", "FGVCAircraft",
              "Imagenette", "Flickr8k", "Flickr30k", "StanfordCars", "INaturalist",
              "FlyingChairs",
              "FER2013", "MovingMNIST", "STL10", "SVHN", "Omniglot", "GTSRB"):
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
              "clip_boxes_to_image", "complete_box_iou", "complete_box_iou_loss",
              "distance_box_iou", "distance_box_iou_loss", "generalized_box_iou",
              "generalized_box_iou_loss", "masks_to_boxes", "nms",
              "remove_small_boxes", "sigmoid_focal_loss",
              "stochastic_depth", "drop_block2d", "drop_block3d",
              "roi_align", "roi_pool", "ps_roi_align", "ps_roi_pool",
              "deform_conv2d"):
    setattr(ops, _name, globals()[_name])

# The layer classes are built against whichever backend is attached **when they are
# first asked for**, because `use(L)` can change that after this module is imported.


def _ops_getattr(name):
    """`ops.<Layer>`, built on demand.

    **It has to raise `AttributeError` and not `KeyError`.** A module's `__getattr__`
    is asked for `__file__`, `__path__` and `__all__` by ordinary machinery —
    `inspect.getmodule` asks every loaded module for `__file__` — and a `KeyError`
    escaping from here is not caught by `hasattr`, so it comes out of whatever
    unrelated call was walking `sys.modules`. Measured: it surfaced from inside
    `torchvision`'s own import, in a traceback naming `torch.library`, four frames from
    anything to do with this file.
    """
    if name.startswith("__"):
        raise AttributeError(f"module 'borchvision.ops' has no attribute {name!r}")
    built = _ops_layers(_backend())
    if name not in built:
        raise AttributeError(f"module 'borchvision.ops' has no attribute {name!r}")
    return built[name]


ops.__getattr__ = _ops_getattr
ops.__dir__ = lambda: sorted(set(ops.__dict__) | set(_ops_layers(_backend())))

# ── the tv_tensor namespace, and v2's dispatch on top of it ─────────────────
#
# `tv_tensors` is a namespace of its own in torchvision, so it is one here. The five
# types plus `wrap`, `set_return_type` and the format enum; `v2` then gets the thirteen
# names that are the dispatch, and the four transform classes are built lazily for the
# reason the `ops` layers are — they subclass the backend's `nn.Module`.
tv_tensors = _types.ModuleType("borchvision.tv_tensors")
_sys.modules["borchvision.tv_tensors"] = tv_tensors
for _name in ("BoundingBoxFormat", "wrap", "set_return_type",
              "is_rotated_bounding_format"):
    setattr(tv_tensors, _name, globals()[_name])


def _tv_tensors_getattr(name):
    """The five types and their base, **read off the module rather than copied.**

    Copied in with `setattr` at import, `tv_tensors.Image` would be the class built
    for whichever backend was bound first and `borchvision.Image` the one built for
    the backend in use — two classes with one name, and `isinstance` false between
    them. `AttributeError` rather than `KeyError`, for the reason written on
    `ops.__getattr__`.
    """
    if name not in _TV_NAMES:
        raise AttributeError(
            f"module 'borchvision.tv_tensors' has no attribute {name!r}")
    return globals()[name]


tv_tensors.__getattr__ = _tv_tensors_getattr
tv_tensors.__dir__ = lambda: sorted(set(tv_tensors.__dict__) | set(_TV_NAMES))

for _name in ("query_size", "query_chw", "check_type", "has_any", "has_all",
              "get_bounding_boxes", "get_keypoints"):
    setattr(v2, _name, globals()[_name])

def _v2_getattr(_name):
    """`Transform` and the four that inherit it, built on first access.

    As `ops.__getattr__` above and for the same reason — `use(L)` can change which
    `nn.Module` they subclass after this module is imported. **It must raise
    `AttributeError` and not `KeyError`**: a module's `__getattr__` is asked for
    `__file__` by ordinary machinery, and a `KeyError` escaping it comes out of whatever
    unrelated call was walking `sys.modules`.
    """
    if _name.startswith("__"):
        raise AttributeError(
            f"module 'borchvision.transforms.v2' has no attribute {_name!r}")
    built = _v2_dispatch(_backend())
    if _name not in built:
        raise AttributeError(
            f"module 'borchvision.transforms.v2' has no attribute {_name!r}")
    return built[_name]


v2.__getattr__ = _v2_getattr
v2.__dir__ = lambda: sorted(set(v2.__dict__) | set(_v2_dispatch(_backend())))

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

# **The three that the dispatch needs and the kernels do not own.** The corner warps
# next door belong to `v2.functional` proper; these three are the clamping taxonomy and
# the format conversion the type system is built on, so they go up with it.
for _name in ("convert_bounding_box_format", "clamp_bounding_boxes",
              "clamp_keypoints"):
    setattr(v2_functional, _name, globals()[_name])
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

# **`<op>_image` is the same function, and here it is the same object.**
#
# v2 routes by the type of what arrives: `resize(inpt, …)` looks at `inpt` and calls
# `resize_image`, `resize_mask`, `resize_bounding_boxes`. This was declined whole as *a
# dispatch kernel — the image body is the public name one namespace over*, and that is a
# true description of why the pair exists in torchvision. It is not a reason to be
# missing here. **Every tensor in this library is a plain one**, so the dispatcher's
# only branch is the image branch, and the two names name one function.
#
# Measured before it was written, against real torchvision: 36 of the 38 `*_image`
# kernels take **the same parameter names in the same order** as their dispatcher, and
# the values are bit-identical on a plain tensor. So this binds the object rather than
# wrapping it — a wrapper is a second body under a second name, which is the thing the
# comment above already refuses for v1.
#
# **The three that are not here are not oversights.** `jpeg_image` needs a codec;
# `to_image` and `to_pil_image` have no dispatcher of that name to be an alias of.
#
# `test_gap.py` compares each pair with `is`, so these cannot drift into two bodies.
_V2_IMAGE_KERNELS = (
    "adjust_brightness", "adjust_contrast", "adjust_gamma", "adjust_hue",
    "adjust_saturation", "adjust_sharpness", "affine", "autocontrast", "center_crop",
    "crop", "elastic", "equalize", "erase", "five_crop", "gaussian_blur",
    "gaussian_noise", "get_dimensions", "get_num_channels", "get_size",
    "grayscale_to_rgb", "horizontal_flip", "invert", "normalize", "pad",
    "permute_channels", "perspective", "posterize", "resize", "resized_crop",
    "rgb_to_grayscale", "rotate", "solarize", "ten_crop", "to_dtype", "vertical_flip",
)
for _name in _V2_IMAGE_KERNELS:
    setattr(v2_functional, _name + "_image", getattr(v2_functional, _name))


for _name in ("horizontal_flip_bounding_boxes", "vertical_flip_bounding_boxes",
              "crop_bounding_boxes", "center_crop_bounding_boxes",
              "pad_bounding_boxes", "resize_bounding_boxes",
              "resized_crop_bounding_boxes", "sanitize_bounding_boxes",
              "horizontal_flip_mask", "vertical_flip_mask", "crop_mask",
              "center_crop_mask", "pad_mask", "resize_mask", "resized_crop_mask",
              "horizontal_flip_keypoints", "vertical_flip_keypoints",
              "crop_keypoints", "resize_keypoints", "clamp_keypoints",
              "center_crop_keypoints", "pad_keypoints", "resized_crop_keypoints",
              "sanitize_keypoints",
              "affine_mask", "rotate_mask", "perspective_mask", "elastic_mask",
              "affine_keypoints", "rotate_keypoints", "perspective_keypoints",
              "elastic_keypoints",
              "affine_bounding_boxes", "rotate_bounding_boxes",
              "perspective_bounding_boxes", "elastic_bounding_boxes"):
    setattr(v2_functional, _name, globals()[_name])


def to_image(inpt):
    """`(H,W,C)` to a `(C,H,W)` tensor — **and it does not divide by 255.**

    The function form of `ToImage`, and it *is* that class's body rather than a second
    copy: the class was here and the function was declined, which put the library in the
    position of shipping the transform and refusing the one-line call that does the same
    thing.

    **torchvision returns a `tv_tensors.Image` and this returns a plain tensor**, which
    is the same compromise `ToImage` already makes here and is said in both places. That
    type is the half of v2 declined in this library — every tensor here is a plain one,
    which is exactly why the `*_image` kernels above can be aliases at all. The values
    and the shape are torchvision's; only the class around them differs.
    """
    return _V2ToImage()(inpt)


v2_functional.to_image = to_image
