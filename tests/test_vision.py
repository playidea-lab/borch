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


# --- the fourteen that arrived later ----------------------------------------
#
# The same division as above. The golden cases hold the values against real torchvision
# wherever the draw can be pinned, and **what is left over is the draw itself** — whether it
# happens, whether it happens once or per call, and whether the branch nobody pins is right.


def test_vertical_flip_with_half_probability_produces_both_outcomes():
    """The horizontal flip's check, on the other axis. **It is not a copy for its own sake:**
    a vertical flip written on the width axis passes every value case on a square picture,
    and this marked image is 4x4 — so the mark's position is what separates them."""
    V.manual_seed(0)
    flip = V.RandomVerticalFlip(p=0.5)
    top_marked = np.zeros((4, 4, 3), dtype=np.uint8)
    top_marked[0, :, :] = 255
    seen = {bool(flip(top_marked)[-1, 0, 0] == 255) for _ in range(60)}
    assert seen == {True, False}, f"sixty draws and only one side came out: {seen}"


def test_random_apply_takes_all_of_them_or_none():
    """**All or none, never one.** That is the whole difference between `RandomApply` and
    putting a `p` on each transform, and one draw per transform would pass a value case
    pinned at 0 or 1 while failing here."""
    V.manual_seed(0)
    both = V.RandomApply([V.Pad(1), V.Pad(2)], p=0.5)
    sizes = {both(np.zeros((4, 4, 3), dtype=np.float32)).shape[0] for _ in range(60)}
    assert sizes == {4, 10}, (
        f"sixty draws gave sizes {sorted(sizes)} — 4 is none of them applied and 10 is "
        "both. Anything between means the list was drawn for one at a time.")


def test_random_choice_honours_its_weights():
    """A weight of zero has to mean never. torchvision's weights are relative and numpy's
    have to sum to 1, so this is where the normalisation in between is checked."""
    V.manual_seed(0)
    pick = V.RandomChoice([V.Pad(1), V.Pad(3)], p=[1, 0])
    sizes = {pick(np.zeros((4, 4, 3), dtype=np.float32)).shape[0] for _ in range(40)}
    assert sizes == {6}, f"the zero-weighted transform was chosen: {sorted(sizes)}"


def test_random_order_applies_every_one_exactly_once():
    """Shuffling must not drop or repeat. Two pads of different sizes are used because
    **their sum is the same whatever the order** — so the shape says every one ran, and
    nothing else in this test can say it."""
    V.manual_seed(0)
    shuffled = V.RandomOrder([V.Pad(1), V.Pad(2)])
    sizes = {shuffled(np.zeros((4, 4, 3), dtype=np.float32)).shape[0] for _ in range(40)}
    assert sizes == {10}, f"a transform was skipped or repeated: {sorted(sizes)}"


def test_padding_of_two_numbers_is_left_right_then_top_bottom():
    """**The two-element form is the one that misreads.** It is (left/right, top/bottom),
    not (left, top) — and read the wrong way the picture still comes back, one axis too
    tall. The golden cases pin the four-element form, where both readings agree."""
    out = V.Pad((1, 2))(np.zeros((4, 4, 3), dtype=np.float32))
    assert out.shape == (8, 6, 3), (
        f"(1, 2) gave {out.shape[:2]} — expected 4+2+2 tall and 4+1+1 wide")


def test_grayscale_of_three_channels_gives_three_equal_ones():
    """`num_output_channels=3` is what a three-channel model needs, and the failure it hides
    is a broadcast that leaves the three channels **different** — which still trains, worse."""
    img = np.stack([np.full((4, 4), v, dtype=np.float32) for v in (0.1, 0.5, 0.9)], axis=2)
    out = V.Grayscale(3)(img)
    assert out.shape == (4, 4, 3)
    assert np.allclose(out[:, :, 0], out[:, :, 1]) and np.allclose(out[:, :, 1], out[:, :, 2])


def test_random_resized_crop_visits_more_than_one_place():
    """With room to move, always cropping the same place is not augmentation — `RandomCrop`'s
    check, on the transform that draws **both** the size and the position."""
    V.manual_seed(0)
    crop = V.RandomResizedCrop(4)
    seen = {crop(_MARKED.astype(np.float32)).tobytes() for _ in range(60)}
    assert len(seen) > 1, "sixty crops and one distinct result — the draw is dead"


def test_random_resized_crop_falls_back_to_the_centre():
    """**Ten draws can all miss**, and then torchvision centre-crops rather than failing. A
    ratio no draw can satisfy is how that branch is reached on purpose; without it the
    fallback is only ever exercised by accident."""
    V.manual_seed(0)
    crop = V.RandomResizedCrop((2, 2), scale=(1.0, 1.0), ratio=(100.0, 100.0))
    out = crop(np.zeros((8, 4, 3), dtype=np.float32))
    assert out.shape == (2, 2, 3)


def test_random_erasing_blanks_one_rectangle_and_leaves_the_rest():
    """What is erased is **a rectangle**, and it holds the value asked for. A version that
    erases the whole image, or fills with something else, passes both golden cases — one is
    pinned at p=0 and the other at the branch where nothing is erased."""
    V.manual_seed(0)
    x = np.ones((3, 8, 8), dtype=np.float32)
    out = V.RandomErasing(p=1.0, scale=(0.2, 0.2), ratio=(1.0, 1.0), value=0.0)(x)
    blanked = out == 0.0
    assert blanked.any(), "p=1 erased nothing"
    assert not blanked.all(), "the whole picture was erased"
    # The same rows and columns in every channel — that is what makes it a rectangle.
    rows, cols = np.where(blanked[0])
    assert blanked.sum() == blanked[0].sum() * 3
    assert blanked[0].sum() == (rows.max() - rows.min() + 1) * (cols.max() - cols.min() + 1)


def test_random_erasing_leaves_the_original_alone():
    """`inplace=False` is the default and it has to be true. Erasing into the caller's tensor
    changes **the dataset**, not the batch — the same picture arrives already blanked next
    epoch, and nothing ever raises."""
    V.manual_seed(0)
    x = np.ones((3, 8, 8), dtype=np.float32)
    V.RandomErasing(p=1.0, scale=(0.2, 0.2), ratio=(1.0, 1.0), value=0.0)(x)
    assert (x == 1.0).all(), "the input was erased in place while inplace was False"


def test_random_erasing_moves_the_rectangle_around():
    V.manual_seed(0)
    x = np.ones((3, 8, 8), dtype=np.float32)
    erase = V.RandomErasing(p=1.0, scale=(0.2, 0.2), ratio=(1.0, 1.0), value=0.0)
    seen = {erase(x).tobytes() for _ in range(40)}
    assert len(seen) > 1, "forty erasures in the same place — the draw is dead"


def test_manual_seed_reaches_the_transforms_that_arrived_later():
    """The seed is one generator for the whole module, and a transform that reaches for its
    own would reproduce **within itself** and diverge here."""
    def draw():
        V.manual_seed(11)
        crop = V.RandomResizedCrop(4)
        erase = V.RandomErasing(p=0.5, value=0.0)
        img = _MARKED.astype(np.float32)
        return [(crop(img).tobytes(),
                 erase(np.ones((3, 6, 6), dtype=np.float32)).tobytes())
                for _ in range(10)]

    assert draw() == draw()


def test_resize_refuses_a_filter_it_does_not_have():
    """The five filters it cannot resample with **keep their names** and say so. Left out
    entirely, `InterpolationMode.BICUBIC` stops with an `AttributeError`, and that wording
    is the one a typo produces."""
    assert V.InterpolationMode.BICUBIC.value == "bicubic"
    with pytest.raises(ValueError, match="does not resample here"):
        V.Resize(4, interpolation=V.InterpolationMode.BICUBIC)


def test_resize_takes_the_enum_and_the_string_alike():
    """Both spellings are written in the wild — torchvision's documentation passes the enum
    and its tutorials pass the string."""
    img = np.zeros((8, 6, 3), dtype=np.float32)
    a = V.Resize((4, 3), interpolation="bilinear")(img)
    b = V.Resize((4, 3), interpolation=V.InterpolationMode.BILINEAR)(img)
    assert np.array_equal(a, b)


# --- arguments the cases above pass over ------------------------------------
#
# Prompted by a measurement from another session: on the borch.ts side `MaxPool2d` was
# present taking `(kernel)` alone against the core's `return_indices`, `InstanceNorm` took
# `(eps?)` against five, and `Adam` was missing `weight_decay` — three names present and
# narrower than they look, none of them visible to a count of names. **What finds that is a
# case whose arguments exercise the parameter**, so these are the arguments the golden cases
# do not reach.


def test_erasing_takes_a_value_for_each_channel():
    """A sequence fills channel by channel. Handed straight through it would broadcast into
    one number for all three, which on a normalised image is **a colour rather than the
    channel means** — and no shape says so."""
    V.manual_seed(0)
    out = V.RandomErasing(p=1.0, scale=(0.2, 0.2), ratio=(1.0, 1.0),
                          value=[0.1, 0.2, 0.3])(np.ones((3, 8, 8), dtype=np.float32))
    for channel, want in enumerate((0.1, 0.2, 0.3)):
        blanked = out[channel][out[channel] != 1.0]
        assert blanked.size and np.allclose(blanked, want), (
            f"channel {channel} was filled with {set(blanked.tolist()) or 'nothing'}, not {want}")


def test_erasing_with_random_fills_the_rectangle_with_noise():
    """`value="random"` is a different branch, not a different number — one draw per pixel."""
    V.manual_seed(0)
    out = V.RandomErasing(p=1.0, scale=(0.2, 0.2), ratio=(1.0, 1.0),
                          value="random")(np.ones((3, 8, 8), dtype=np.float32))
    blanked = out[out != 1.0]
    assert blanked.size > 1 and len(set(blanked.tolist())) > 1, (
        "the rectangle came out one repeated value — that is a constant, not noise")


def test_erasing_refuses_a_value_that_is_not_one_per_channel():
    with pytest.raises(ValueError, match="channels"):
        V.RandomErasing(p=1.0, scale=(0.2, 0.2), ratio=(1.0, 1.0),
                        value=[0.1, 0.2])(np.ones((3, 8, 8), dtype=np.float32))


def test_a_size_of_one_number_or_a_one_element_list_means_both_sides():
    """torchvision spreads `[3]` over both sides, and a copied line uses that form. Read as
    a pair it would be a size of one dimension and stop somewhere else entirely."""
    img = np.zeros((5, 4, 3), dtype=np.float32)
    assert V.FiveCrop(2)(img)[0].shape == V.FiveCrop([2])(img)[0].shape == (2, 2, 3)
    assert V.Pad([1])(img).shape == (7, 6, 3)


def test_grayscale_of_a_single_channel_image_passes_it_through():
    """A one-channel picture is already grey. The luma sum reaches for three channels, so
    without the branch this is an `IndexError` on the very input that needs no work."""
    img = np.full((5, 4, 1), 0.25, dtype=np.float32)
    out = V.Grayscale()(img)
    assert out.shape == (5, 4, 1) and np.allclose(out, 0.25)


def test_normalize_inplace_is_a_no_op_and_the_input_survives():
    """**The one argument in this file that is accepted and does nothing.** Found by
    auditing constructor arguments that never reach `__call__`, after another session
    found the same shape in `borch_webgpu`'s optimizers — `weight_decay` handed to a JS
    call that discards surplus arguments, so five optimizers trained without it and
    nothing raised.

    Here the values are identical either way, so the argument stays and the docstring
    says what it does. This is what stops that sentence from rotting: if somebody makes
    `inplace` real on the core alone, the sister library cannot follow and the two
    libraries part on the same line — so it fails here first.
    """
    x = np.ones((3, 2, 2), dtype=np.float32)
    out = V.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)(x)
    assert np.allclose(out, 1.0), "the returned value must be normalised either way"
    assert np.allclose(x, 1.0), (
        "the input was written through. `inplace` is documented as a no-op because the "
        "sister library's tensors are immutable — making it real here parts the two.")


def test_random_crop_takes_its_arguments_in_torchvisions_order():
    """**`RandomCrop(32, 4, True)` set `fill=True` here and `pad_if_needed=True` there.**

    The list was `(size, padding, fill)` against torchvision's
    `(size, padding, pad_if_needed, fill, padding_mode)`, so a positional call landed on
    a different parameter and returned a correctly shaped picture either way. Nothing
    raised, and no count of names could see it — `tests/test_torch_signatures.py` asks
    the general question, and this pins the one that was wrong.
    """
    assert V.RandomCrop(2, 1, True).pad_if_needed is True


def test_random_crop_pads_a_picture_smaller_than_the_crop_when_asked():
    """`pad_if_needed` pads **both sides**, so a shortfall of one makes the picture two
    larger — torchvision's arithmetic, not a rounding of it. Without the flag the same
    call refuses."""
    V.manual_seed(0)
    img = np.arange(20, dtype=np.float32).reshape(5, 4, 1)
    assert V.RandomCrop((6, 5), pad_if_needed=True)(img).shape == (6, 5, 1)
    with pytest.raises(ValueError, match="larger than the image"):
        V.RandomCrop((6, 5))(img)


def test_random_crops_padding_mode_reaches_the_padding_it_adds():
    """The mode is not decoration: `edge` and `constant` put different numbers at the
    border, and a `padding_mode` accepted and dropped leaves zeros that look deliberate."""
    img = np.arange(20, dtype=np.float32).reshape(5, 4, 1)
    V.manual_seed(0)
    edged = V.RandomCrop((7, 6), padding=1, padding_mode="edge")(img)
    assert np.array_equal(edged, V.Pad(1, padding_mode="edge")(img))


# --- transforms.functional --------------------------------------------------


def test_the_functional_namespace_answers_every_spelling_of_the_import():
    """**This is why it is a module and not an attribute.**

    `import torchvision.transforms.functional as F` is the line tutorials write, and
    `import a.b.c` walks `sys.modules` for each dotted name — an attribute holding an
    object satisfies none of it, however much it looks like a namespace from inside.
    All four spellings are asked because a package gives all four for free and a
    hand-registered pair only gives what it was given.
    """
    import borchvision.transforms.functional as one
    from borchvision.transforms import functional as two
    from borchvision import transforms as three
    import borchvision as four

    assert one is two is three.functional is four.transforms.functional


def test_get_image_size_is_width_first_and_get_dimensions_is_not():
    """**The one size in this file that is not height first.** torchvision inherits it
    from PIL, where a size is `(w, h)`, and returning the shape here instead would be
    right in length and swapped in meaning on any picture that is not square — which is
    why the picture in this test is 5 by 4."""
    img = np.zeros((5, 4, 3), dtype=np.float32)
    assert V.transforms.functional.get_image_size(img) == [4, 5]
    assert V.transforms.functional.get_dimensions(img) == [3, 5, 4]
    assert V.transforms.functional.get_image_num_channels(img) == 3


def test_functional_crop_refuses_to_leave_the_picture():
    """numpy answers an out-of-range slice with **a shorter array**, not an error. Left
    to it, the batch that follows fails to stack and the message points at the stacking
    rather than at the crop."""
    img = np.zeros((5, 4, 3), dtype=np.float32)
    assert V.transforms.functional.crop(img, 1, 1, 3, 2).shape == (3, 2, 3)
    with pytest.raises(ValueError, match="leaves the image"):
        V.transforms.functional.crop(img, 3, 0, 4, 4)


def test_functional_and_the_class_are_one_implementation():
    """The functions delegate rather than reimplement. Two copies of the resize filter
    would agree the day they were written and not on some later one, so what is checked
    is that they are **the same answer**, not that both are close to torch."""
    img = np.arange(60, dtype=np.float32).reshape(5, 4, 3)
    F = V.transforms.functional
    assert np.array_equal(F.resize(img, [3, 2]), V.Resize((3, 2))(img))
    assert np.array_equal(F.center_crop(img, [3, 2]), V.CenterCrop((3, 2))(img))
    assert np.array_equal(F.pad(img, 1, 0.5), V.Pad(1, 0.5)(img))
    assert np.array_equal(F.rgb_to_grayscale(img, 3), V.Grayscale(3)(img))


# --- the photometric five ---------------------------------------------------


def test_colour_jitter_draws_the_order_as_well_as_the_factors():
    """**The order is part of the draw**, and it has to be, because these do not
    commute: `adjust_contrast` measures the picture's mean grey, and a brightness
    applied first has already moved it.

    Two factors pinned to single values leave the factors deterministic and the order
    not — so two distinct results over many calls is the order varying, and one is a
    fixed order wearing a draw's name.
    """
    V.manual_seed(0)
    jitter = V.ColorJitter(brightness=(0.6, 0.6), contrast=(1.8, 1.8))
    img = np.linspace(0, 1, 60, dtype=np.float32).reshape(5, 4, 3)
    seen = {jitter(img).tobytes() for _ in range(60)}
    assert len(seen) == 2, (
        f"sixty draws gave {len(seen)} distinct results — with both factors pinned "
        "there are exactly two orders that differ, so one means the order is fixed.")


def test_a_factor_left_alone_is_stored_as_nothing_rather_than_as_the_identity():
    """`ColorJitter()` keeps `None`, not `(1, 1)`. It is the difference between not
    blending and blending by a ratio that happens to cancel — the second still costs
    a cast, and on a uint8 picture a cast is a rounding."""
    assert V.ColorJitter().brightness is None
    assert V.ColorJitter(0.5).contrast is None
    assert V.ColorJitter(brightness=(1.0, 1.0)).brightness is None
    assert V.ColorJitter(brightness=(0.5, 1.5)).brightness == (0.5, 1.5)


def test_hue_and_saturation_leave_a_one_channel_picture_alone():
    """torchvision returns it untouched to match PIL, and it is **a branch rather
    than arithmetic that happens to cancel** — without it, `_rgb2hsv` reads three
    channels off a picture that has one."""
    grey = np.linspace(0, 1, 20, dtype=np.float32).reshape(5, 4, 1)
    assert np.array_equal(V.transforms.functional.adjust_hue(grey, 0.3), grey)
    assert np.array_equal(V.transforms.functional.adjust_saturation(grey, 0.3), grey)


def test_the_photometric_five_refuse_what_torch_refuses():
    F = V.transforms.functional
    img = np.zeros((4, 4, 3), dtype=np.float32)
    for call, match in ((lambda: F.adjust_brightness(img, -1), "non-negative"),
                        (lambda: F.adjust_contrast(img, -1), "non-negative"),
                        (lambda: F.adjust_saturation(img, -1), "non-negative"),
                        (lambda: F.adjust_gamma(img, -1), "non-negative"),
                        (lambda: F.adjust_hue(img, 0.7), r"\[-0.5, 0.5\]")):
        with pytest.raises(ValueError, match=match):
            call()


def test_the_blend_is_done_in_the_precision_torch_promotes_to():
    """**One pixel decides this, and here it is.**

    A uint8 blend ends in a truncating cast. torch promotes `uint8 * float` to
    float32; doing the same arithmetic in float64 moves values across that boundary,
    and `adjust_saturation(1.7)` on the golden's byte picture was one step out until
    the working precision matched. `(102, 168, 82)` is a pixel where the two
    precisions genuinely disagree — float64 gives 188 in the green channel and
    float32 gives 189 — so this fails rather than passing by luck if the precision
    widens again.
    """
    px = np.array([[[102, 168, 82]]], dtype=np.uint8)
    grey = np.asarray(V.transforms.functional.rgb_to_grayscale(px), dtype=np.float64)
    in_float64 = np.clip(1.7 * px.astype(np.float64) + (1 - 1.7) * grey,
                         0, 255).astype(np.uint8)
    ours = V.transforms.functional.adjust_saturation(px, 1.7)
    assert ours.dtype == np.uint8
    assert list(in_float64.ravel()) == [76, 188, 42], (
        "the float64 answer moved — this test pins a disagreement between two "
        "precisions, so it is worth nothing if one of them changed")
    assert list(ours.ravel()) == [76, 189, 42], (
        f"ours gave {list(ours.ravel())}; torch promotes to float32 and answers "
        "[76, 189, 42]. Widening the working precision is the change that breaks this.")


# --- the six that rewrite pixels --------------------------------------------


def test_the_two_that_need_bytes_say_so_rather_than_inventing_a_meaning():
    """`posterize` throws away the low bits of a byte and `equalize` counts 256 bins.
    Neither means anything on a float image, and torchvision raises — so a float
    picture has to stop here too, or the same line gives a number over there and an
    answer here."""
    f = np.zeros((4, 4, 3), dtype=np.float32)
    with pytest.raises(TypeError, match="uint8"):
        V.transforms.functional.posterize(f, 4)
    with pytest.raises(TypeError, match="uint8"):
        V.transforms.functional.equalize(f)


def test_posterize_at_eight_bits_changes_nothing():
    """The mask is `-(2 ** (8 - bits))`, so eight bits is `-1` and keeps everything.
    Written as `2 ** (8 - bits)` — the obvious spelling — eight bits masks with 1 and
    the picture becomes its own low bit."""
    img = np.arange(48, dtype=np.uint8).reshape(4, 4, 3)
    assert np.array_equal(V.transforms.functional.posterize(img, 8), img)
    assert not np.array_equal(V.transforms.functional.posterize(img, 4), img)


def test_autocontrast_leaves_a_flat_channel_alone():
    """A channel with one value has `max == min`, and the scale is `bound / 0`.
    torchvision replaces the non-finite scale with 1 and the minimum with 0 rather
    than clamping afterwards — a picture that is all one colour comes back unchanged
    instead of all black or all NaN."""
    flat = np.full((4, 4, 3), 7, dtype=np.uint8)
    assert np.array_equal(V.transforms.functional.autocontrast(flat), flat)


def test_sharpness_leaves_the_border_as_it_found_it():
    """The 3×3 blur is convolved **without padding** and written back into the middle,
    so the outermost ring is the original. A padded convolution gives different
    numbers exactly there, and the middle — which is what anyone looks at — agrees
    either way."""
    img = (np.arange(75, dtype=np.float32).reshape(5, 5, 3) / 75).astype(np.float32)
    out = V.transforms.functional.adjust_sharpness(img, 0.0)
    assert np.array_equal(out[0], img[0]) and np.array_equal(out[-1], img[-1])
    assert np.array_equal(out[:, 0], img[:, 0]) and np.array_equal(out[:, -1], img[:, -1])
    assert not np.array_equal(out[1:-1, 1:-1], img[1:-1, 1:-1])


def test_a_picture_too_small_to_blur_comes_back_untouched():
    """Two pixels wide has no middle to convolve. torchvision returns the input, and
    the alternative is an empty result written into an empty slice — which raises
    nothing and returns the right shape."""
    tiny = np.arange(12, dtype=np.float32).reshape(2, 2, 3)
    assert np.array_equal(V.transforms.functional.adjust_sharpness(tiny, 2.0), tiny)


def test_every_random_pixel_wrapper_actually_draws():
    """Six wrappers share one implementation, so **one broken draw is six**. Each is
    asked at p=0.5 for both outcomes — the golden pins p=0 and p=1, and neither can
    see a draw that never happens."""
    V.manual_seed(0)
    # **Big enough, and varied enough, that every one of the six actually changes it.**
    # A 4x4 ramp was the first choice and `equalize` returned it untouched — its
    # `step` floors to zero below 255 pixels a channel, so the two outcomes were the
    # same picture and the test read that as a dead draw. The picture has to be able
    # to show the difference before absence of difference means anything.
    img = (np.arange(1200) ** 2 % 256).astype(np.uint8).reshape(20, 20, 3)
    for build in (lambda: V.RandomInvert(p=0.5),
                  lambda: V.RandomAutocontrast(p=0.5),
                  lambda: V.RandomEqualize(p=0.5),
                  lambda: V.RandomPosterize(3, p=0.5),
                  lambda: V.RandomSolarize(100, p=0.5),
                  lambda: V.RandomAdjustSharpness(2.0, p=0.5)):
        transform = build()
        seen = {transform(img).tobytes() for _ in range(60)}
        assert len(seen) == 2, (
            f"{transform} gave {len(seen)} distinct results over sixty draws — one "
            "means the draw is dead, and these six share one implementation")


def test_edge_and_symmetric_are_not_the_same_padding():
    """**At a padding of one they are the same picture**, and the golden held them as
    two entries with identical numbers for it.

    `symmetric` mirrors the border row, and the mirror of a single row is that row —
    which is what `edge` repeats. So the pair proved nothing, and another session
    swapped the two modes in their port with all sixteen value cases still green. Two
    is the smallest padding where the words diverge, and this asks it of the library
    directly so the distinction does not depend on which padding a case happened to
    pick.
    """
    img = np.arange(60, dtype=np.float32).reshape(5, 4, 3)
    assert np.array_equal(V.Pad(1, padding_mode="edge")(img),
                          V.Pad(1, padding_mode="symmetric")(img)), (
        "at one they agree — if that changed, this test is measuring something else now")
    assert not np.array_equal(V.Pad(2, padding_mode="edge")(img),
                              V.Pad(2, padding_mode="symmetric")(img)), (
        "edge and symmetric give the same answer at a padding of two, so nothing here "
        "can tell them apart")


def test_totensor_says_what_to_do_when_a_group_of_crops_arrives():
    """`Compose([FiveCrop(3), ToTensor()])` is a natural thing to write and a wrong
    one, and **the refusal has to say which part is wrong.**

    Without the guard the tuple survives `asarray` as a stacked 4-D array and the
    message reads `it received (5, 2, 2, 3)` — true, and it leaves the reader to work
    out that the five is five pictures. This project's standard for an error is that it
    says what to do, and the answer is torchvision's own: a `Lambda`.

    Found from the sister library, where the same composition type-checked and died
    inside `ToTensor` with a shape complaint instead.
    """
    img = np.arange(60, dtype=np.float32).reshape(5, 4, 3)
    with pytest.raises(TypeError, match="Lambda"):
        V.Compose([V.FiveCrop(2), V.ToTensor()])(img)
    # And the way through still works, so the message is advice rather than a wall.
    out = V.Compose([V.FiveCrop(2),
                     V.Lambda(lambda crops: [V.ToTensor()(c) for c in crops])])(img)
    assert len(out) == 5 and out[0].shape == (3, 2, 2)


# --- resampling on a grid ---------------------------------------------------


def test_rotation_and_affine_draw_a_new_angle_each_call():
    """The golden pins the range to one value, so **nothing there can see the draw.**"""
    V.manual_seed(0)
    img = np.arange(300, dtype=np.float32).reshape(10, 10, 3)
    for transform in (V.RandomRotation(45), V.RandomAffine(45)):
        seen = {transform(img).tobytes() for _ in range(40)}
        assert len(seen) > 1, f"{transform} gave one result over forty draws"


def test_expand_grows_the_picture_to_torchvisions_sizes():
    """**The obvious expectation is wrong and the golden is right.**

    A quarter turn of a 5x4 picture gives 5x6, not 4x5 — the corner arithmetic and its
    ceil/floor pair land there, and torchvision agrees. I wrote 4x5 from the geometry
    and this test failed, which is the only reason the number in the docstring above
    is measured rather than assumed.
    """
    img = np.zeros((5, 4, 3), dtype=np.float32)
    F = V.transforms.functional
    assert F.rotate(img, 90, expand=True).shape == (5, 6, 3)
    assert F.rotate(img, 180, expand=True).shape == (5, 4, 3)
    assert F.rotate(img, 30, expand=True).shape[:2] > (5, 4)
    assert F.rotate(img, 30).shape == (5, 4, 3)


def test_a_centre_is_an_offset_from_the_middle_not_a_pixel():
    """torch's convention: `(0, 0)` is the middle of the picture. Passing the middle
    itself as a centre would shift the picture by half its own size, and **the shift is
    exactly the kind that looks like a different crop rather than a wrong argument.**"""
    img = np.arange(300, dtype=np.float32).reshape(10, 10, 3)
    F = V.transforms.functional
    assert np.array_equal(F.rotate(img, 30, "bilinear"),
                          F.rotate(img, 30, "bilinear", center=[5, 5]))
    assert not np.array_equal(F.rotate(img, 30, "bilinear"),
                              F.rotate(img, 30, "bilinear", center=[0, 0]))


def test_the_fill_is_sampled_alongside_the_picture():
    """A mask of ones goes through the same grid, so a **bilinear** edge pixel is part
    picture and part fill. Deciding inside-ness from the coordinates gives a hard edge,
    and the tell is that no pixel is ever strictly between the fill and the picture."""
    img = np.ones((9, 9, 3), dtype=np.float32)
    out = V.transforms.functional.rotate(img, 30, "bilinear", fill=[0.0, 0.0, 0.0])
    between = (out > 1e-6) & (out < 1.0 - 1e-6)
    assert between.any(), (
        "every pixel is either the picture or the fill — the mask is being decided "
        "rather than sampled, and the edge is up to a pixel out")


def test_nearest_rounds_halves_to_even():
    """A quarter turn lands **every** sampled position exactly halfway between two
    pixels. `floor(x + 0.5)` and half-to-even disagree on all of them at once, so this
    is the angle where the rounding rule is the whole answer rather than a corner
    case."""
    img = np.arange(25, dtype=np.float32).reshape(5, 5, 1)
    turned = V.transforms.functional.rotate(img, 90, "nearest")
    assert np.array_equal(turned, np.rot90(img, 1))


def test_affine_refuses_a_scale_that_is_not_positive():
    with pytest.raises(ValueError, match="positive"):
        V.transforms.functional.affine(np.zeros((4, 4, 3), dtype=np.float32),
                                       0, [0, 0], 0.0, [0, 0])


# --- the other three that resample ------------------------------------------


def test_gaussian_blur_reflects_the_border_rather_than_zeroing_it():
    """A zero border darkens the edge of every blurred picture, which reads as a
    vignette rather than as a mistake. Reflected, a picture of one constant value comes
    back **exactly that value everywhere** — which is the cheapest thing that can tell
    the two apart."""
    flat = np.full((7, 7, 3), 0.6, dtype=np.float32)
    out = V.transforms.functional.gaussian_blur(flat, [3, 3], [1.0, 1.0])
    assert np.allclose(out, 0.6), "the border is darker than the middle"


def test_gaussian_blur_refuses_an_even_kernel():
    """An even kernel has no centre pixel to sit on, so the picture would move by half
    a pixel — a blur that also shifts."""
    img = np.zeros((5, 5, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="odd"):
        V.transforms.functional.gaussian_blur(img, [2, 3])
    with pytest.raises(ValueError, match="positive"):
        V.transforms.functional.gaussian_blur(img, [3, 3], [0.0, 1.0])


def test_gaussian_blur_takes_its_sizes_width_first():
    """`kernel_size` and `sigma` are **(x, y)**, like `get_image_size` and unlike every
    shape in this file. A picture blurred hard across and barely down has to come out
    smeared horizontally, and swapping the pair is invisible on a square kernel."""
    img = np.zeros((9, 9, 1), dtype=np.float32)
    img[4, 4] = 1.0
    out = V.transforms.functional.gaussian_blur(img, [9, 9], [3.0, 0.3])
    across = out[4, :, 0].std()
    down = out[:, 4, 0].std()
    assert across < down, (
        "a large sigma_x should spread the point sideways, flattening the row — "
        f"row spread {across:.4f}, column spread {down:.4f}")


def test_perspective_with_the_corners_unmoved_changes_nothing_much():
    """**The identity is the case that catches a sign or a transpose.** A projective
    map wrong in either still looks like a plausible tilt when the corners have moved,
    and only stops looking plausible when they have not."""
    img = np.arange(60, dtype=np.float32).reshape(5, 4, 3)
    corners = [[0, 0], [3, 0], [3, 4], [0, 4]]
    out = V.transforms.functional.perspective(img, corners, corners, "bilinear")
    assert np.allclose(out, img, atol=1e-4)


def test_elastic_transform_is_given_its_field_rather_than_drawing_one():
    """`ElasticTransform` draws the displacement and `elastic_transform` applies it, so
    a batch can share one warp — and, here, so the golden can compare a warp at all."""
    img = np.arange(60, dtype=np.float32).reshape(5, 4, 3)
    nothing = np.zeros((5, 4, 2), dtype=np.float32)
    assert np.allclose(V.transforms.functional.elastic_transform(img, nothing), img,
                       atol=1e-4)


def test_the_three_drawn_wrappers_draw():
    """`GaussianBlur` draws its sigma, `RandomPerspective` its corners, and
    `ElasticTransform` a whole field. **Nothing in the golden can see any of the
    three** — the frozen cases give the field, the corners and the sigma."""
    V.manual_seed(0)
    img = (np.arange(1200) ** 2 % 256).astype(np.float32).reshape(20, 20, 3) / 255
    for transform in (V.GaussianBlur(3), V.RandomPerspective(p=1.0),
                      V.ElasticTransform(alpha=20.0)):
        seen = {transform(img).tobytes() for _ in range(30)}
        assert len(seen) > 1, f"{transform} gave one result over thirty draws"


# --- the policies -----------------------------------------------------------


def test_every_policy_transform_draws():
    """All four draw on every call, which is why **the golden holds almost nothing
    about them** — a frozen picture would be a frozen dice roll. This is the check that
    the dice are thrown at all."""
    V.manual_seed(0)
    img = (np.arange(1200) ** 2 % 256).astype(np.uint8).reshape(20, 20, 3)
    for transform in (V.AutoAugment(), V.RandAugment(), V.TrivialAugmentWide(),
                      V.AugMix()):
        seen = {transform(img).tobytes() for _ in range(30)}
        assert len(seen) > 1, f"{transform} gave one result over thirty draws"


def test_the_operation_names_are_the_vocabulary_and_all_of_them_resolve():
    """**Every name a policy can draw has to exist**, and the failure is that it does
    not until the draw happens to land there — a rare op with a typo'd name is a crash
    on an unlucky epoch rather than on the first call."""
    img = np.arange(48, dtype=np.uint8).reshape(4, 4, 3)
    every = set()
    for space in (V._space_auto(10, (4, 4)), V._space_rand(31, (4, 4)),
                  V._space_trivial(31), V._space_augmix(10, (4, 4), True)):
        every |= set(space)
    for name in sorted(every):
        V._apply_op(img, name, 1.0, "nearest", [0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="not recognized"):
        V._apply_op(img, "Twirl", 1.0, "nearest", None)


def test_augmix_keeps_the_original_in_the_mixture():
    """**This is the one that is different in kind.** The other three replace the
    picture; `AugMix` averages several augmented versions back into the original, so
    the result stays near the input however hard the chains hit. With `alpha` small the
    original's weight is large, and the output has to be closer to it than a single
    chain would be."""
    V.manual_seed(0)
    img = (np.arange(1200) ** 2 % 256).astype(np.uint8).reshape(20, 20, 3)
    far = np.mean([np.abs(V.TrivialAugmentWide()(img).astype(float) - img).mean()
                   for _ in range(20)])
    near = np.mean([np.abs(V.AugMix(alpha=0.05)(img).astype(float) - img).mean()
                    for _ in range(20)])
    assert near < far, (
        f"AugMix moved the picture {near:.1f} on average and a single chain {far:.1f} "
        "— the mixture is not keeping the original")


def test_the_policies_are_a_list_because_torchvision_hands_back_one():
    """`policies` is a public attribute. **What holds a value is part of the surface**,
    and the golden caught this on its first run: identical data, different brackets."""
    assert isinstance(V.AutoAugment().policies, list)
    assert len(V.AutoAugment().policies) == 25


def test_augmix_refuses_a_severity_outside_its_range():
    with pytest.raises(ValueError, match="severity"):
        V.AugMix(severity=0)
    with pytest.raises(ValueError, match="severity"):
        V.AugMix(severity=11)


def test_randaugment_with_no_operations_is_the_identity():
    """The only configuration of any of the four that does not draw."""
    img = np.arange(48, dtype=np.uint8).reshape(4, 4, 3)
    assert np.array_equal(V.RandAugment(num_ops=0)(img), img)


# ----------------------------------------------------------------- transforms.v2
#
# The golden cases carry the v2 reprs and the deterministic values. What is left here is
# the same thing this file is for everywhere else — that the draws draw — plus one
# difference from torchvision that a value comparison cannot see, because the golden
# check compares numbers and not the type they are stored in.

V2 = V.transforms.v2


def test_ToImage_gives_a_wider_integer_than_torchvision_does():
    """**int64 where torchvision gives uint8**, and the numbers the same.

    The core has no uint8 storage, so this is not a choice this file made and not one
    it can undo. It is asserted rather than left to be noticed because the golden check
    compares values with `allclose`, which is blind to the dtype — the one place this
    difference lives is the one place the comparison cannot look.
    """
    picture = np.arange(24, dtype=np.uint8).reshape(4, 3, 2)
    out = V2.ToImage()(picture)
    assert str(out.dtype).endswith("int64")
    assert out.shape == (2, 4, 3)
    assert np.array_equal(np.asarray(out.numpy()), picture.transpose(2, 0, 1))
    # And the pair v2 tells you to write instead of `ToTensor` lands where v1 lands.
    scaled = V2.Compose([V2.ToImage(), V2.ToDtype(BT.float32, scale=True)])(picture)
    assert np.allclose(np.asarray(scaled.numpy()),
                       picture.transpose(2, 0, 1) / 255.0, atol=1e-6)


def test_GaussianNoise_draws_a_value_per_pixel_not_one_per_picture():
    """A single draw broadcast over the picture is the plausible wrong version: it
    passes a mean test, passes a standard-deviation test on a batch, and adds no noise
    at all in the sense anyone wants."""
    flat = np.full((16, 16, 3), 0.5, dtype=np.float32)
    out = np.asarray(V2.GaussianNoise(0.0, 0.05, clip=False)(flat))
    assert out.std() > 0.02
    assert len(np.unique(out.round(6))) > 100


def test_RandomChannelPermutation_reaches_every_ordering():
    """Three channels have six orderings, and a shuffle that draws an index rather than
    a permutation reaches only three of them."""
    picture = np.zeros((2, 2, 3), dtype=np.float32)
    picture[:, :, 0], picture[:, :, 1], picture[:, :, 2] = 1.0, 2.0, 3.0
    seen = {tuple(np.asarray(V2.RandomChannelPermutation()(picture))[0, 0].tolist())
            for _ in range(300)}
    assert len(seen) == 6


def test_RandomZoomOut_puts_the_picture_somewhere_other_than_the_corner():
    """The offsets are two draws, and a version that pads evenly — or that uses one
    draw for both axes — leaves the golden case at `p=0` green."""
    picture = np.ones((4, 4, 3), dtype=np.float32)
    corners = set()
    for _ in range(200):
        out = np.asarray(V2.RandomZoomOut(fill=0, side_range=(2.0, 2.0), p=1.0)(picture))
        assert out.shape == (8, 8, 3)
        rows, cols = np.nonzero(out[:, :, 0])
        corners.add((int(rows.min()), int(cols.min())))
    assert len({r for r, _ in corners}) > 2 and len({c for _, c in corners}) > 2
    assert any(r != c for r, c in corners)              # not one draw used twice


def test_RandomResize_covers_the_range_and_stops_before_the_top():
    """`[min_size, max_size)` — the top is excluded, which is the off-by-one this
    transform exists to get wrong."""
    picture = np.ones((10, 10, 3), dtype=np.float32)
    sizes = {np.asarray(V2.RandomResize(4, 7)(picture)).shape[0] for _ in range(400)}
    assert sizes == {4, 5, 6}


def test_ScaleJitter_moves_both_axes_by_the_same_factor():
    """It is one factor applied to the target, not a factor per axis — a per-axis draw
    would change the aspect ratio, which is the thing scale jitter must not do.

    Asserted in **pixels rather than as a ratio**: the factor can take the 10×20 target
    down to 2×5, where whole-pixel rounding alone moves the ratio from 2.0 to 2.5. A
    ratio tolerance loose enough for that case is loose enough to miss a real per-axis
    draw at the large end; one pixel is the true bound at both ends.
    """
    picture = np.ones((10, 20, 3), dtype=np.float32)
    for _ in range(200):
        h, w = np.asarray(V2.ScaleJitter((10, 20), (0.5, 2.0))(picture)).shape[:2]
        assert abs(w - 2 * h) <= 1


def test_RandomPhotometricDistort_applies_each_adjustment_sometimes():
    """`p` is per adjustment, not one coin for all four. With one coin the picture is
    either untouched or fully distorted and never partly, which shows up as far fewer
    distinct outcomes than four independent coins give."""
    picture = np.linspace(0.1, 0.9, 4 * 4 * 3, dtype=np.float32).reshape(4, 4, 3)
    outcomes = {np.asarray(V2.RandomPhotometricDistort(p=0.5)(picture)).round(4).tobytes()
                for _ in range(200)}
    assert len(outcomes) > 32


def test_MixUp_pairs_each_row_with_the_one_before_it():
    """The pairing is a roll by one and **not** a random partner — torchvision says so
    outright, and a recipe that shuffles its batch for that reason is relying on it."""
    batch = np.zeros((3, 2, 2, 1), dtype=np.float32)
    batch[0], batch[1], batch[2] = 0.0, 1.0, 2.0
    labels = np.array([0, 1, 2])
    mixed, _ = V2.MixUp(num_classes=3)(batch, labels)
    mixed = np.asarray(mixed)
    # row 0 lies between row 0 and row 2 (its partner), never near row 1.
    assert mixed[0, 0, 0, 0] <= 2.0 and mixed[1, 0, 0, 0] <= 1.0
    lam = mixed[1, 0, 0, 0]                             # row 1 = 1*lam + 0*(1-lam)
    assert abs(mixed[2, 0, 0, 0] - (2.0 * lam + 1.0 * (1 - lam))) < 1e-5


def test_MixUp_labels_stay_a_distribution():
    """Two classes' worth of weight summing to one is the only property a loss
    downstream depends on, and a swapped `lam`/`1-lam` keeps the sum while inverting
    the picture — so the row's own class getting the larger share is asserted too."""
    batch = np.zeros((4, 2, 2, 1), dtype=np.float32)
    labels = np.array([0, 1, 2, 3])
    for _ in range(50):
        _, out = V2.MixUp(alpha=5.0, num_classes=4)(batch, labels)
        out = np.asarray(out)
        assert np.allclose(out.sum(axis=1), 1.0, atol=1e-5)
        assert (out >= 0).all()
        # alpha=5 puts lam near 0.5 but the row's own class is index i, its partner i-1.
        assert out[2, 2] > 0.1 and out[2, 1] > 0.1


def test_CutMix_mixes_the_label_by_the_area_it_actually_pasted():
    """The box is clipped at the picture's edge, so the drawn weight and the pasted
    area come apart — and the label has to follow **the area**. Measured directly:
    count the pixels that changed and compare with the weight the label was given."""
    batch = np.zeros((2, 8, 8, 1), dtype=np.float32)
    batch[1] = 1.0
    labels = np.array([0, 1])
    for _ in range(100):
        out, mixed = V2.CutMix(num_classes=2)(batch, labels)
        out, mixed = np.asarray(out), np.asarray(mixed)
        pasted = float((out[0, :, :, 0] == 1.0).sum()) / 64.0
        # row 0's own class keeps `lam`, its partner takes what was pasted.
        assert abs(mixed[0, 1] - pasted) < 1e-5
        assert abs(mixed[0, 0] - (1.0 - pasted)) < 1e-5


def test_CutMix_and_MixUp_refuse_a_batch_that_does_not_match_its_labels():
    """Every shape involved here is plausible, so a mismatch is a silent mis-train
    rather than a crash — which is why the checks are worth having at all."""
    batch = np.zeros((4, 2, 2, 1), dtype=np.float32)
    with pytest.raises(ValueError, match="num_classes must be passed"):
        V2.MixUp()(batch, np.array([0, 1, 2, 3]))
    with pytest.raises(ValueError, match="does not match the batch size"):
        V2.CutMix(num_classes=2)(batch, np.array([0, 1]))
    with pytest.raises(ValueError, match="4 dims"):
        V2.MixUp(num_classes=2)(np.zeros((2, 2, 1), dtype=np.float32), np.array([0, 1]))
    with pytest.raises(ValueError, match="index based"):
        V2.CutMix(num_classes=2)(batch, np.zeros((4, 2, 2)))


def test_v2_names_are_not_a_second_copy_of_the_arithmetic():
    """The v2 twins subclass v1's transforms; only the repr is theirs. If one ever
    grows a body the two spellings drift apart silently, and every value case in the
    golden file goes on passing because it asks one spelling or the other."""
    picture = np.random.default_rng(3).random((6, 5, 3)).astype(np.float32)
    for build in (lambda m: m.Resize((3, 4)), lambda m: m.CenterCrop(3),
                  lambda m: m.Pad(2, padding_mode="reflect"),
                  lambda m: m.Grayscale(3), lambda m: m.Normalize([0.4], [0.2]),
                  lambda m: m.GaussianBlur(3, (1.0, 1.0))):
        assert np.allclose(np.asarray(build(V2)(picture)),
                           np.asarray(build(V.transforms)(picture)), atol=1e-6)
