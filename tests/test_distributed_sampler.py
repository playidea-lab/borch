"""`DistributedSampler`'s shuffled half, **which the golden cannot ask.**

Its unshuffled answers are torch's exactly and are frozen there. Shuffled they are not:
torch draws with `torch.randperm` and a `torch.Generator`, this draws with numpy's, and
the same seed gives a different permutation. Measured — `n=7, replicas=2, seed=0` gives
torch `[4, 5, 2, 1]` for rank 0 and this `[2, 3, 5, 1]`.

**So what is asked here is the property and not the value**, and the properties are what
the class is for: the ranks partition the padded list, every rank gets the same count,
the order is redrawn per epoch, and the same epoch repeats.

A value that cannot be shared is not a reason to leave the behaviour unasked — it is a
reason to ask the part that does not depend on which generator drew it.
"""

import borch


def _ranks(count, replicas, epoch=0, shuffle=True, drop_last=False, seed=0):
    out = []
    for rank in range(replicas):
        sampler = borch.utils.data.DistributedSampler(
            list(range(count)), num_replicas=replicas, rank=rank, shuffle=shuffle,
            seed=seed, drop_last=drop_last)
        sampler.set_epoch(epoch)
        out.append(list(sampler))
    return out


def test_the_ranks_partition_the_padded_list_when_shuffled():
    """Every index appears, and the total is the padded length.

    **The union is the check, not the lengths.** Two ranks that both took the same half
    have the right count each and cover half the data — which is the failure a
    contiguous split makes look like a partition.
    """
    for count, replicas in ((10, 2), (10, 3), (7, 4), (5, 5)):
        parts = _ranks(count, replicas)
        flat = [one for part in parts for one in part]
        assert len(set(flat)) == count, (count, replicas, parts)
        assert len({len(part) for part in parts}) == 1, (count, replicas, parts)
        assert len(flat) == len(parts[0]) * replicas


def test_drop_last_shortens_rather_than_pads():
    """With `drop_last` nothing is repeated, so the union is smaller than the data."""
    parts = _ranks(10, 3, drop_last=True)
    flat = [one for part in parts for one in part]
    assert len(flat) == 9
    assert len(set(flat)) == 9


def test_two_epochs_draw_different_orders_and_one_epoch_repeats():
    """**`set_epoch` is the whole of the shuffling.** A loader that never calls it draws
    the same permutation forever, and nothing anywhere says so — the only witness is a
    model that stops improving early."""
    first = _ranks(12, 3, epoch=0)
    again = _ranks(12, 3, epoch=0)
    later = _ranks(12, 3, epoch=1)
    assert first == again
    assert first != later


def test_every_rank_draws_the_same_permutation():
    """The interleave is a partition **only because the ranks agree on the order.**
    Drawn from a shared stream instead, each rank would advance it and take a slice of a
    different permutation — and the union would still have the right length."""
    parts = _ranks(12, 3, epoch=2)
    flat = sorted(one for part in parts for one in part)
    assert flat == list(range(12))
