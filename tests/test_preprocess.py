"""The manifest's preparation, built from the manifest.

**There is no golden for this either.** It is a composition of transforms that already
have one, so what is checked here is that the composition is the manifest's: the size it
lands on, the numbers it divides by, and that a manifest it cannot honour is refused
before an image reaches it rather than four layers into a matmul.
"""

import numpy as np
import pytest

from borch._preprocess import transform_for

MANIFEST = {
    "inputSize": [3, 224, 224],
    "valueRange": "unit",
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
    "resize": {"shortSide": 256, "interpolation": "bicubic"},
    "centerCrop": [224, 224],
}


def _picture(h=480, w=640, value=128):
    return np.full((h, w, 3), value, dtype=np.uint8)


def test_a_photograph_comes_out_the_size_the_weights_were_trained_on():
    x = transform_for(MANIFEST)(_picture())
    assert x.shape == (3, 224, 224) and x.dtype == np.float32


def test_a_tall_photograph_and_a_wide_one_land_on_the_same_shape():
    """The short side is scaled and the middle is cropped, so the aspect does not leak in."""
    steps = transform_for(MANIFEST)
    assert steps(_picture(1000, 300)).shape == steps(_picture(300, 1000)).shape == (3, 224, 224)


def test_the_numbers_are_the_manifests_own():
    """A flat grey at 128/255, normalised, is (128/255 - mean) / std per channel."""
    x = transform_for(MANIFEST)(_picture(value=128))
    for c, (m, s) in enumerate(zip(MANIFEST["mean"], MANIFEST["std"])):
        want = (128 / 255 - m) / s
        assert np.allclose(x[c], want, atol=2e-2), f"channel {c}: {x[c].mean()} against {want}"


def test_without_a_resize_an_image_of_the_wrong_size_is_refused_by_name():
    bare = {**MANIFEST, "resize": None, "centerCrop": None}
    with pytest.raises(ValueError, match="arrive at that size"):
        transform_for(bare)(_picture(480, 640))
    assert transform_for(bare)(_picture(224, 224)).shape == (3, 224, 224)


def test_a_manifest_with_no_preprocess_says_so_rather_than_guessing():
    with pytest.raises(ValueError, match="schema version 1"):
        transform_for(None)


def test_means_that_do_not_match_the_channels_are_refused():
    with pytest.raises(ValueError, match="channels"):
        transform_for({**MANIFEST, "mean": [0.5]})


def test_an_interpolation_this_runtime_does_not_have_is_named():
    with pytest.raises(ValueError, match="interpolation"):
        transform_for({**MANIFEST, "resize": {"shortSide": 256, "interpolation": "lanczos"}})


def test_a_value_range_other_than_unit_is_refused():
    with pytest.raises(ValueError, match="value range"):
        transform_for({**MANIFEST, "valueRange": "signed"})


def test_a_picture_is_taken_as_well_as_an_array():
    """What holds a photograph in a browser is PIL; what these transforms take is numpy."""
    from borch._data import _image_module

    Image = _image_module()
    picture = Image.fromarray(_picture(300, 500))
    assert transform_for(MANIFEST)(picture).shape == (3, 224, 224)
