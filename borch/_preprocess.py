"""The preparation a set of weights was trained under, built from its manifest.

    steps = transform_for(model.manifest["preprocess"])
    x = steps(image)                      # (3, H, W) float32, ready for the backbone

**Why this is not a new pipeline.** `borch-hub` already builds this on the TypeScript
side out of borchvision's transforms, one image at a time. This is the same composition
out of the same transforms' Python twins — which are held to torchvision's values by the
golden — rather than a second description of what the manifest means. What differs is the
end: the TS one returns a `[1, C, H, W]` tensor, and this returns `(C, H, W)` as numpy,
because what asks for it here is a batch loop that stacks.

**Why anyone should care.** Measured on a thousand ImageNetV2 photographs through
`imagenet-efficientnet-b0` (`tests/browser/preprocess_cost.py`): the manifest's own
preparation scores **0.658** top-1, a plain 224 square without normalising **0.482**, and
a 64px square without normalising — which is what `workbench` did before this —
**0.025**. The size is worth about fifty points of that, the normalisation seven to
fifteen, the crop three.
"""

import numpy as _np

# **Every manifest in the registry asks for bicubic, and this runtime resamples bilinear.**
# `borchvision`'s Resize carries bilinear and nearest — the others are PIL's, each a
# separate kernel — and PIL's bicubic is not torch's either (a = −0.5 against −0.75), so
# porting one of them would have been porting the wrong one as easily as the right one.
#
# Measured before deciding, on the thousand ImageNetV2 photographs through
# imagenet-efficientnet-b0 (`tests/browser/preprocess_cost.py`): the manifest's own
# bicubic scores 0.658 top-1 and the same pipeline with bilinear scores **0.655** — three
# tenths of a point, which is less than the four tenths this run sits from the number the
# manifest itself records. The filter is not where the accuracy is; the size is.
#
# So the manifest says what the weights were trained under, and this says what they were
# prepared with. The two are written down separately because they are different facts.
_INTERPOLATION = {"bicubic": "bilinear", "bilinear": "bilinear", "nearest": "nearest"}


def transform_for(pre, vision=None):
    """`preprocess` from a manifest → a callable that prepares one image.

    `vision` is the module holding the transforms; `borchvision` by default. The image is
    anything those accept — a PIL image, or an array shaped (H, W, C).
    """
    if vision is None:
        import borchvision as vision                       # noqa: PLC0415

    if not pre:
        raise ValueError(
            "this manifest does not say how to prepare an image (schema version 1 had no "
            "such field) — the weights load and nothing can be fed to them.")
    channels, height, width = (int(v) for v in pre["inputSize"])
    mean, std = list(pre["mean"]), list(pre["std"])
    if len(mean) != channels or len(std) != channels:
        raise ValueError(
            f"the manifest asks for {channels} channels and gives {len(mean)} means and "
            f"{len(std)} standard deviations — one of the two is not describing this model.")
    if pre.get("valueRange", "unit") != "unit":
        raise ValueError(f"this runtime only knows the 'unit' value range, not {pre['valueRange']!r}")

    steps = []
    resize = pre.get("resize")
    if resize:
        how = str(resize.get("interpolation", "bilinear"))
        if how not in _INTERPOLATION:
            raise ValueError(f"unknown interpolation {how!r} — one of {sorted(_INTERPOLATION)}")
        steps.append(vision.transforms.Resize(int(resize["shortSide"]),
                                              interpolation=_INTERPOLATION[how]))
    crop = pre.get("centerCrop")
    if crop:
        steps.append(vision.transforms.CenterCrop([int(crop[0]), int(crop[1])]))
    geometry = vision.transforms.Compose(steps) if steps else None
    to_tensor = vision.transforms.ToTensor()
    normalize = vision.transforms.Normalize(mean, std)
    sized = bool(resize or crop)

    def prepare(image):
        # **The picture goes back to eight bits between the two halves.** `ToTensor`
        # divides by 255 for uint8 alone — torchvision's rule, and the right one, since a
        # float array is usually already in [0, 1]. But `Resize` here returns float64, so
        # composed the ordinary way the division never happens and every value arrives 255
        # times too large. The manifest's pipeline is defined on an image, and the number
        # it records was measured through PIL, whose resize also hands back eight bits.
        out = geometry(image) if geometry is not None else image
        arr = _np.asarray(out)
        if arr.dtype != _np.uint8:
            arr = _np.clip(_np.rint(arr), 0, 255).astype(_np.uint8)
        out = normalize(to_tensor(arr))
        x = out.numpy() if hasattr(out, "numpy") else _np.asarray(out)
        x = _np.asarray(x, dtype=_np.float32)
        if x.shape != (channels, height, width):
            # **Said with both shapes.** Without a resize in the manifest the caller has to
            # hand over an image that is already the right size, and the failure otherwise
            # is a matmul complaining about a dimension four layers in.
            got = "x".join(str(n) for n in x.shape)
            want = f"{channels}x{height}x{width}"
            raise ValueError(
                f"this manifest prepares a {want} image and the result is {got}"
                + ("" if sized else " — it asks for no resize or crop, so the image has to arrive at that size"))
        return x

    return prepare
