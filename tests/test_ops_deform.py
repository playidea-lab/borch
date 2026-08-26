"""**The gradient through the sampling positions**, which is the point of the layer and
which no golden case holds.

A golden case freezes an output. What makes a deformable convolution deformable is that
`∂out/∂offset` exists at all — the offsets are produced by another convolution, and if
nothing flows back into them the layer is an expensive ordinary convolution reading a
fixed grid. That is a wrong answer no value comparison can see, because the forward pass
is identical either way.

Also here: the two edge rules, each of which a small-offset fixture hides.
"""

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


def _given(seed=0, batch=1, in_c=2, out_c=3, size=6, spread=0.7):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((batch, in_c, size, size)).astype(np.float32),
            rng.standard_normal((out_c, in_c, 3, 3)).astype(np.float32) * 0.3,
            (rng.standard_normal((batch, 18, size - 2, size - 2)).astype(np.float32)
             * spread))


def test_the_gradient_reaches_the_offsets():
    """**Not only the values.** The bilinear weights are `1 - frac` and `frac` on the
    coordinate, so the derivative with respect to a displacement is the difference
    between the neighbours it sits between — and a reader that rounded the coordinate
    to an integer would produce the same forward pass with no gradient here at all.
    """
    picture, weight, offsets = _given()

    mine_in = borch.tensor(picture.copy())
    mine_off = borch.tensor(offsets.copy())
    mine_in.requires_grad_(True)
    mine_off.requires_grad_(True)
    V.ops.deform_conv2d(mine_in, mine_off, borch.tensor(weight)).sum().backward()

    theirs_in = torch.from_numpy(picture.copy())
    theirs_off = torch.from_numpy(offsets.copy())
    theirs_in.requires_grad_(True)
    theirs_off.requires_grad_(True)
    T.deform_conv2d(theirs_in, theirs_off, torch.from_numpy(weight)).sum().backward()

    got = np.asarray(mine_off.grad.numpy() if hasattr(mine_off.grad, "numpy")
                     else mine_off.grad)
    assert np.abs(got).max() > 1e-3, (
        "the offsets got no gradient — the layer computes the right numbers and cannot "
        "learn where to look, which is the whole of what it is for")
    assert np.allclose(got, theirs_off.grad.numpy(), atol=3e-5)

    into_input = np.asarray(mine_in.grad.numpy() if hasattr(mine_in.grad, "numpy")
                            else mine_in.grad)
    assert np.allclose(into_input, theirs_in.grad.numpy(), atol=3e-5)


def test_the_gradient_reaches_the_mask_too():
    """v2's mask is a multiplier, so its gradient is the sampled value — the simplest of
    the three and the one most easily left out, because dropping it still trains the
    offsets and the weights."""
    picture, weight, offsets = _given(seed=5)
    mask = np.random.default_rng(6).random((1, 9, 4, 4)).astype(np.float32)

    mine = borch.tensor(mask.copy())
    mine.requires_grad_(True)
    V.ops.deform_conv2d(borch.tensor(picture), borch.tensor(offsets),
                        borch.tensor(weight), mask=mine).sum().backward()
    theirs = torch.from_numpy(mask.copy())
    theirs.requires_grad_(True)
    T.deform_conv2d(torch.from_numpy(picture), torch.from_numpy(offsets),
                    torch.from_numpy(weight), mask=theirs).sum().backward()

    got = np.asarray(mine.grad.numpy() if hasattr(mine.grad, "numpy") else mine.grad)
    assert np.abs(got).max() > 1e-3
    assert np.allclose(got, theirs.grad.numpy(), atol=3e-5)


@pytest.mark.parametrize("spread", [0.05, 0.7, 3.0])
def test_the_edges_agree_at_every_offset_size(spread):
    """**A small displacement never crosses a border**, and both edge rules are about
    borders.

    At a twentieth of a pixel this agrees with an implementation that clamps the
    coordinate to the edge, and with one that keeps a sample the kernel would drop. At
    three pixels most of the kernel is outside the map for the border outputs. The
    middle value is what an ordinary fixture would use, and it only sometimes crosses.
    """
    picture, weight, offsets = _given(seed=7, size=5, spread=spread)
    got = np.asarray(V.ops.deform_conv2d(
        borch.tensor(picture), borch.tensor(offsets), borch.tensor(weight)).numpy())
    want = T.deform_conv2d(torch.from_numpy(picture), torch.from_numpy(offsets),
                           torch.from_numpy(weight)).detach().numpy()
    assert np.allclose(got, want, atol=3e-5), (
        f"max {np.abs(got - want).max()} at spread {spread} — the four corners are "
        "dropped one at a time here, not clamped together")


def test_zero_offsets_are_an_ordinary_convolution():
    """The sanity check that says the grid, the dilation and the padding line up before
    any displacement is involved."""
    picture, weight, _ = _given(seed=9)
    zeros = np.zeros((1, 18, 4, 4), dtype=np.float32)
    got = np.asarray(V.ops.deform_conv2d(
        borch.tensor(picture), borch.tensor(zeros), borch.tensor(weight)).numpy())
    plain = torch.nn.functional.conv2d(torch.from_numpy(picture),
                                       torch.from_numpy(weight)).numpy()
    assert np.allclose(got, plain, atol=1e-5)


def test_an_offset_channel_count_that_does_not_divide_is_refused():
    picture, weight, offsets = _given()
    with pytest.raises(RuntimeError, match="not valid"):
        V.ops.deform_conv2d(borch.tensor(picture), borch.tensor(offsets[:, :5]),
                            borch.tensor(weight))


@pytest.mark.parametrize("groups,which", [(3, "in_channels"), (4, "in_channels")])
def test_a_module_whose_channels_do_not_divide_is_refused(groups, which):
    with pytest.raises(ValueError, match=which):
        V.ops.DeformConv2d(2, 6, 3, groups=groups)


def test_the_modules_weights_are_the_narrower_default():
    """`kaiming_uniform_(a=√5)` — a convolution's own — and **not** the pyramid's
    `a=1`, which is √3 wider. Two initialisations in one file, and the difference is a
    constant nobody would see in a histogram."""
    import math
    layer = V.ops.DeformConv2d(4, 6, 3)
    weight = np.asarray(layer.weight.numpy())
    fan_in = int(np.prod(weight.shape[1:]))
    bound = 1.0 / math.sqrt(fan_in)
    largest = float(np.abs(weight).max())
    assert largest <= bound + 1e-6, f"{largest} outside ±{bound}"
    assert largest > bound / 2, (
        f"{largest} against {bound} — this looks like the wider `a=1` bound the pyramid "
        "uses, which is √3 further out")
