"""Looks at the part of `borchvision` **the golden cases cannot see.**

The golden cases match values against real torchvision, and a random transform cannot be
matched that way — we cannot use torch's generator. So the golden cases only ask where the
probability is pinned at 0 or 1, or where there is exactly one place to crop.

That leaves **nobody looking at whether the draw actually happens.** A broken draw (always
cropping the same place, say, or using one draw across a whole batch) leaves the golden cases
green. This is that place.
"""

import pathlib
import sys

import numpy as np
import pytest

_root = pathlib.Path(__file__).resolve().parent.parent


if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import borch as BT                                            # noqa: E402
import borchvision as V                                      # noqa: E402

V.use(BT)

# An image marked at one end only. Flipped, the mark moves to the other side — this shape is
# used because whether it flipped can be decided from a single value.
_MARKED = np.zeros((4, 4, 3), dtype=np.uint8)
_MARKED[:, 0, :] = 255


def _flipped(img):
    return bool(img[0, -1, 0] == 255)


def test_flip_with_half_probability_produces_both_outcomes():
    """At p=0.5, one side only means **a constant rather than a draw.** The golden cases cannot see it."""
    V.manual_seed(0)
    flip = V.RandomHorizontalFlip(p=0.5)
    seen = {_flipped(flip(_MARKED)) for _ in range(60)}
    assert seen == {True, False}, f"sixty draws and only one side came out: {seen}"


def test_crop_with_padding_visits_more_than_one_offset():
    """With several places to crop, always cropping the same one is not augmentation."""
    V.manual_seed(0)
    crop = V.RandomCrop(4, padding=2)
    seen = {crop(_MARKED).tobytes() for _ in range(60)}
    assert len(seen) > 1, "sixty crops and one distinct result — the draw is dead"


def test_manual_seed_makes_the_same_draws_again():
    """It cannot give torch's scene, and **within ourselves** it has to reproduce."""
    def draw():
        V.manual_seed(7)
        crop = V.RandomCrop(4, padding=2)
        return [crop(_MARKED).tobytes() for _ in range(10)]

    assert draw() == draw()


def test_augment_batch_draws_per_image_not_once_per_batch():
    """Exactly what `augment_batch`'s docstring claims.

    One draw across a whole batch means nothing inside the batch was augmented relative to
    anything else. torchvision's classes draw once per call and diverge here, which is why
    this has a name of its own. A claim written down and never measured is a claim the next
    person believes.
    """
    V.manual_seed(0)
    x = np.zeros((64, 1, 4, 4), dtype=np.float32)
    x[:, :, :, 0] = 1.0                                  # marked at the left edge only
    out = V.augment_batch(x, crop=4, padding=0, hflip_p=0.5)
    flipped = out[:, 0, 0, -1] == 1.0
    assert flipped.any() and not flipped.all(), (
        f"{int(flipped.sum())} of 64 flipped — the whole batch received one draw")


def test_augment_batch_keeps_shape_and_dtype():
    x = np.zeros((5, 3, 8, 8), dtype=np.float32)
    out = V.augment_batch(x, crop=8, padding=4, hflip_p=0.5)
    assert out.shape == (5, 3, 8, 8)
    assert out.dtype == np.float32


def test_augment_batch_rejects_wrong_rank():
    with pytest.raises(ValueError, match="N,C,H,W"):
        V.augment_batch(np.zeros((3, 8, 8), dtype=np.float32))


def test_crop_given_a_tensor_says_where_to_put_totensor():
    """A tensor is refused — one tensor per image makes one GPU buffer per image on the sister side.

    What is looked at is not the refusal but **whether it says what to do.** That is this
    project's specification for an error message.
    """
    with pytest.raises(TypeError, match="ToTensor"):
        V.RandomCrop(4)(BT.tensor(np.zeros((3, 4, 4), dtype=np.float32)))


def test_totensor_does_not_divide_a_float_image():
    """Only uint8 is divided by 255. Dividing a float again makes it 255× darker **with no
    exception**, and
    only the training quietly fails — this pins it by value."""
    img = np.full((2, 2, 3), 0.5, dtype=np.float32)
    assert np.allclose(V.ToTensor()(img).numpy(), 0.5)


def test_normalize_accepts_numpy_and_tensor_alike():
    """Normalising a batch through numpy and through tensors has to give **the same answer.**
    Two routes diverging means the training pipeline and the tutorial teach different things."""
    arr = np.random.default_rng(0).random((3, 4, 4)).astype(np.float32)
    norm = V.Normalize((0.5, 0.4, 0.3), (0.2, 0.3, 0.4))
    assert np.allclose(norm(arr), norm(BT.tensor(arr)).numpy(), atol=1e-6)
