"""**The half a golden case cannot hold.**

`stochastic_depth` and `drop_block` draw from a generator, and the two libraries draw
from different ones — so the numbers cannot be frozen against torchvision's. What can
be checked is what they *distribute*, and each of these asserts something that a
plausible wrong implementation gets wrong:

- dropping roughly `p`, which is what `gamma` exists to arrange and what using `p`
  directly gets wrong by a factor of `block_size ** spatial`
- scaling the survivors by `1 / (1 - p)`, without which every layer is quieter than
  the next one expects
- dropping **blocks** rather than pixels, which is the whole point of DropBlock and is
  invisible in any average
- `row` and `batch` differing, which is invisible in a single draw

The deterministic settings — `training=False`, `p=0` — are frozen in the golden table
instead, where they are compared against torchvision's own answer.
"""

import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

pytest.importorskip("torchvision")

import borch                                                     # noqa: E402
import borchvision as V                                          # noqa: E402

# Big enough that a proportion means something: 64 * 8 * 21 * 21 is a quarter of a
# million numbers, so a rate that is out by a tenth cannot be a run of luck.
_SHAPE = (64, 8, 21, 21)
_TOLERANCE = 0.05


def _ones():
    return borch.tensor(np.ones(_SHAPE, dtype=np.float32))


@pytest.mark.parametrize("p", [0.1, 0.3, 0.5, 0.7])
def test_drop_block2d_drops_what_torchvisions_own_drops(p):
    """**`gamma` is not `p`.** It is `p` scaled by how many positions a block covers
    and how many seeds there are to draw. Using `p` directly drops roughly
    `block_size ** 2` times too much — nine times here, which no eye on a feature map
    would catch and no shape check would either.

    **And the realised rate is not `p` either**, which the first version of this test
    asserted and torchvision failed: at `p=0.5` it drops 0.39 and at `p=0.7` it drops
    0.51. `gamma` is derived as though the blocks never overlapped, and the more of them
    there are the more they land on each other — so the rate approaches `p` from below
    and only meets it when dropping is rare. Written as *about `p`* this passed at 0.1
    and 0.3 and would have been read as a check on the arithmetic.

    So it is compared against torchvision's own realised rate, which is the standard
    the rest of this repository holds.
    """
    from torchvision import ops as real
    import torch

    ours = np.asarray(V.ops.drop_block2d(_ones(), p=p, block_size=3, training=True))
    theirs = real.drop_block2d(torch.ones(_SHAPE), p=p, block_size=3,
                               training=True).numpy()
    mine, yours = float((ours == 0).mean()), float((theirs == 0).mean())
    assert abs(mine - yours) < _TOLERANCE, (
        f"dropped {mine:.3f} where torchvision drops {yours:.3f}")
    assert mine <= p + _TOLERANCE, (
        f"dropped {mine:.3f}, more than the {p} asked for — overlapping blocks can "
        "only take the rate down")


def test_drop_block2d_drops_blocks_and_not_pixels(tmp_path):
    """**The point of DropBlock, and it does not show in any average.**

    A dropped position's neighbours are dropped with it, so the zeros come in square
    runs. Ordinary dropout at the same rate gives the same mean and isolated zeros;
    this counts a zero whose four neighbours are all non-zero, which the block version
    can only produce at an edge.
    """
    out = np.asarray(V.ops.drop_block2d(_ones(), p=0.3, block_size=5, training=True))
    zero = out == 0
    inner = zero[:, :, 1:-1, 1:-1]
    alone = (inner & ~zero[:, :, :-2, 1:-1] & ~zero[:, :, 2:, 1:-1]
             & ~zero[:, :, 1:-1, :-2] & ~zero[:, :, 1:-1, 2:])
    assert zero.any(), "nothing was dropped at all"
    assert alone.sum() / max(zero.sum(), 1) < 0.01, (
        "the zeros are scattered rather than in blocks — this is dropout wearing "
        "DropBlock's name, and the drop rate would not show it")


def test_drop_block_rescales_what_it_keeps():
    """What survives is divided by the fraction kept, so the mean is unchanged."""
    out = np.asarray(V.ops.drop_block2d(_ones(), p=0.2, block_size=3, training=True))
    assert abs(float(out.mean()) - 1.0) < _TOLERANCE, (
        f"mean {out.mean():.3f} — the surviving values are not scaled back up")


@pytest.mark.parametrize("mode", ["row", "batch"])
def test_stochastic_depth_scales_the_survivors_by_the_survival_rate(mode):
    """`1 / (1 - p)` and nothing else: every value is either zero or exactly that."""
    out = np.asarray(V.ops.stochastic_depth(_ones(), p=0.5, mode=mode, training=True))
    assert set(np.unique(np.round(out, 5))) <= {0.0, 2.0}, (
        "a survivor is not 1 / (1 - p) — without the division every layer is quieter "
        "than the next one expects")


def test_stochastic_depths_two_modes_are_not_the_same_thing():
    """**`batch` tosses one coin and `row` tosses one per example.** A single draw
    cannot tell them apart — `batch` looks like `row` that happened to agree — so this
    asks over many draws whether the rows within a batch ever disagree.
    """
    def rows_disagree(mode):
        for _ in range(20):
            out = np.asarray(V.ops.stochastic_depth(
                _ones(), p=0.5, mode=mode, training=True))
            per_row = {float(one.mean()) for one in out}
            if len(per_row) > 1:
                return True
        return False

    assert rows_disagree("row"), "every row was dropped together — this is `batch`"
    assert not rows_disagree("batch"), (
        "the rows of one batch differ — this is `row` wearing `batch`'s name")


def test_a_dropout_asked_at_p_zero_hands_back_the_tensor_it_was_given():
    """Not an equal one — **the same object.** torchvision returns `input` unchanged on
    that path, and a copy would pass every value comparison while costing an allocation
    per layer per step."""
    x = _ones()
    assert V.ops.stochastic_depth(x, p=0.0, mode="row", training=True) is x
    assert V.ops.drop_block2d(x, p=0.0, block_size=3, training=True) is x
    assert V.ops.stochastic_depth(x, p=0.5, mode="row", training=False) is x
