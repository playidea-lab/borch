"""**What a golden case cannot hold about the pyramid and the level-picker.**

Both are compared value for value in the golden table. What is here is the half that
needs a *differently shaped* input to show:

- the top-down step interpolates to the lateral's own size, which only matters when the
  two are not exactly a factor of two apart — an odd input is what makes that happen
- `MultiScaleRoIAlign` chooses a level from the box's size, which only shows when the
  boxes differ in size
- the weights are re-initialised wider than a convolution's default, which is a
  distribution rather than a value
"""

import collections
import math
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

pytest.importorskip("torchvision")

import torch                                                     # noqa: E402
import torchvision.ops as T                                      # noqa: E402

import borch                                                     # noqa: E402
import borchvision as V                                          # noqa: E402


def _fill(ours, theirs, seed=0):
    """The same numbers on both sides — two libraries draw from two generators."""
    rng = np.random.default_rng(seed)
    mine = dict(ours.named_parameters())
    mine.update(dict(ours.named_buffers()))
    yours = dict(theirs.named_parameters())
    yours.update(dict(theirs.named_buffers()))
    assert sorted(mine) == sorted(yours)
    for name in sorted(yours):
        if "num_batches" in name:
            continue
        block = rng.standard_normal(tuple(yours[name].shape)).astype(np.float32) * 0.3
        with torch.no_grad():
            yours[name].copy_(torch.from_numpy(block))
        target = mine[name]
        if hasattr(target, "copy_"):
            with borch.no_grad():
                target.copy_(borch.tensor(block))
        else:
            np.asarray(target)[...] = block


def _maps(sizes, widths):
    out = collections.OrderedDict()
    for i, ((h, w), c) in enumerate(zip(sizes, widths)):
        shape = (1, c, h, w)
        out[f"feat{i}"] = ((np.arange(int(np.prod(shape))).reshape(shape) % 11)
                           * 0.2).astype(np.float32)
    return out


@pytest.mark.parametrize("sizes", [
    [(16, 16), (8, 8), (4, 4)],       # exactly halving
    [(15, 17), (8, 9), (4, 5)],       # **odd** — the halves are not halves
    [(13, 7), (7, 4), (4, 2)],        # odd twice over
])
def test_the_pyramid_interpolates_to_the_lateral_and_not_by_a_factor(sizes):
    """**The top-down step goes to the lateral's own size**, not up by two.

    A backbone on an odd input gives maps that are off by one — 15 becomes 8, and twice
    8 is 16. A reader that doubled would try to add a 16-row map into a 15-row one,
    which is an error on some inputs and, once somebody "fixes" it with a crop, a
    silent misalignment on the rest. The evenly halving case cannot tell the two apart,
    which is why it is only the first of three here.
    """
    ours = V.ops.FeaturePyramidNetwork([4, 6, 8], 5)
    theirs = T.FeaturePyramidNetwork([4, 6, 8], 5)
    _fill(ours, theirs)
    ours.eval()
    theirs.eval()
    given = _maps(sizes, [4, 6, 8])

    with borch.no_grad():
        mine = ours({k: borch.tensor(v) for k, v in given.items()})
    with torch.no_grad():
        yours = theirs(collections.OrderedDict(
            (k, torch.from_numpy(v)) for k, v in given.items()))

    assert list(mine.keys()) == list(yours.keys())
    for key in mine:
        got = np.asarray(mine[key].numpy())
        want = yours[key].numpy()
        assert got.shape == want.shape, f"{key}: {got.shape} against {want.shape}"
        assert np.allclose(got, want, atol=2e-5)


def test_the_pyramid_keeps_the_names_and_their_order():
    """An `OrderedDict` in and an `OrderedDict` out, with the same keys. A detector
    reads them by name; a list would be the same values and unusable."""
    ours = V.ops.FeaturePyramidNetwork([4, 6, 8], 5)
    given = _maps([(16, 16), (8, 8), (4, 4)], [4, 6, 8])
    with borch.no_grad():
        out = ours({k: borch.tensor(v) for k, v in given.items()})
    assert isinstance(out, collections.OrderedDict)
    assert list(out.keys()) == ["feat0", "feat1", "feat2"]


def test_a_pyramid_level_of_no_channels_is_refused():
    with pytest.raises(ValueError, match="in_channels=0"):
        V.ops.FeaturePyramidNetwork([4, 0, 8], 5)


def test_the_pyramids_weights_are_initialised_wider_than_a_convolutions_default():
    """`kaiming_uniform_(a=1)` gives `√3 / √fan_in`; `Conv2d`'s own default gives
    `1 / √fan_in`. **√3 apart** — a factor no histogram would be looked at closely
    enough to catch, and the reason the arithmetic is written out rather than borrowed.
    """
    model = V.ops.FeaturePyramidNetwork([4, 6, 8], 5)
    weight = np.asarray(model.inner_blocks[0][0].weight.numpy())
    fan_in = int(np.prod(weight.shape[1:]))
    bound = math.sqrt(3.0) / math.sqrt(fan_in)
    largest = float(np.abs(weight).max())
    assert largest <= bound + 1e-6, f"{largest} is outside ±{bound}"
    assert largest > bound / 2, (
        f"{largest} against a bound of {bound} — this looks like the narrower default, "
        "which is what a convolution initialises itself with before this overwrites it")
    assert np.allclose(np.asarray(model.inner_blocks[0][0].bias.numpy()), 0.0)


def test_multiscale_sends_each_box_to_a_level_by_its_size():
    """**Equation 1**: a 224-pixel box goes to level 4 and the level moves one per
    doubling. Three boxes of very different sizes therefore read three different maps —
    and against a reader that always took the first, the answer for the small box is the
    same and the other two are not.
    """
    pooler = V.ops.MultiScaleRoIAlign(["feat0", "feat1", "feat2"], 3, 2)
    reference = T.MultiScaleRoIAlign(["feat0", "feat1", "feat2"], 3, 2)
    given = _maps([(16, 16), (8, 8), (4, 4)], [5, 5, 5])
    boxes = np.array([[0.0, 0.0, 10.0, 10.0],
                      [0.0, 0.0, 200.0, 200.0],
                      [4.0, 4.0, 60.0, 60.0]], dtype=np.float32)

    with borch.no_grad():
        mine = np.asarray(pooler({k: borch.tensor(v) for k, v in given.items()},
                                 [borch.tensor(boxes)], [(64, 64)]).numpy())
    with torch.no_grad():
        yours = reference(
            collections.OrderedDict((k, torch.from_numpy(v)) for k, v in given.items()),
            [torch.from_numpy(boxes)], [(64, 64)]).numpy()

    assert mine.shape == yours.shape
    assert np.allclose(mine, yours, atol=2e-5)
    # The three do not all come from one level — if they did, this test would pass
    # against a reader that ignores the box size.
    assert not np.allclose(mine[0], mine[1], atol=1e-3)


def test_multiscale_with_one_level_takes_that_level():
    """One map is the path that skips the level arithmetic entirely."""
    pooler = V.ops.MultiScaleRoIAlign(["feat0"], 3, 2)
    reference = T.MultiScaleRoIAlign(["feat0"], 3, 2)
    given = _maps([(16, 16)], [5])
    boxes = np.array([[0.0, 0.0, 10.0, 10.0], [4.0, 4.0, 60.0, 60.0]], dtype=np.float32)
    with borch.no_grad():
        mine = np.asarray(pooler({k: borch.tensor(v) for k, v in given.items()},
                                 [borch.tensor(boxes)], [(64, 64)]).numpy())
    with torch.no_grad():
        yours = reference(
            collections.OrderedDict((k, torch.from_numpy(v)) for k, v in given.items()),
            [torch.from_numpy(boxes)], [(64, 64)]).numpy()
    assert mine.shape == yours.shape and np.allclose(mine, yours, atol=2e-5)


def test_multiscale_without_images_says_so():
    pooler = V.ops.MultiScaleRoIAlign(["feat0", "feat1"], 3, 2)
    given = _maps([(16, 16), (8, 8)], [5, 5])
    with pytest.raises(ValueError, match="images list should not be empty"):
        pooler({k: borch.tensor(v) for k, v in given.items()},
               [borch.tensor(np.zeros((0, 4), np.float32))], [])
