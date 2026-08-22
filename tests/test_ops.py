"""**Box geometry is the one part of `borchvision` with no distribution half.**

Everything else in this library draws — a flip, a crop, an angle — so its cases split
in two: frozen values where the draw can be pinned, and pytest for whether the draw
happens at all. `ops` does not draw. Every one of the eleven is deterministic, so the
golden holds all of them and this file is not about randomness.

What is left for pytest is what a frozen number cannot say: the **shape** of the
answer, the **boundary** of a comparison, and the two places the library returns
indices where a reader expects boxes.
"""

import pathlib
import sys

import numpy as np
import pytest

_root = pathlib.Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import borch as BT                                              # noqa: E402
import borchvision as V                                         # noqa: E402
import borchvision.ops as O                                     # noqa: E402

V.use(BT)

_A = np.array([[0.0, 0.0, 10.0, 10.0],
               [1.0, 1.0, 11.0, 11.0],
               [30.0, 30.0, 40.0, 40.0]], dtype=np.float32)


def test_the_ops_namespace_is_beside_transforms_and_not_inside_it():
    """`torchvision.ops` is top level. Registering it under `transforms` would make
    `import borchvision.ops` work while meaning something torchvision does not."""
    import borchvision.ops as direct
    from borchvision import ops as attribute

    assert direct is attribute
    assert not hasattr(V.transforms, "ops")


def test_box_iou_is_every_box_against_every_box():
    """**An N by M matrix, not a paired list.** Three boxes against two is six numbers,
    and an implementation that zips them returns two — which is the shape everyone
    expects the first time and never the shape that is wanted."""
    other = np.array([[2.0, 2.0, 8.0, 8.0], [12.0, 0.0, 22.0, 10.0]], dtype=np.float32)
    assert O.box_iou(_A, other).shape == (3, 2)
    assert O.box_iou(_A, _A).shape == (3, 3)
    assert np.allclose(np.diag(O.box_iou(_A, _A)), 1.0)


def test_nms_discards_above_the_threshold_and_not_at_it():
    """**`> iou_threshold`, not `>=`.** At zero, boxes that merely touch survive
    together and only an actual overlap is discarded — which is the boundary anybody
    testing this reaches for first, and the one place the two spellings differ
    visibly."""
    touching = np.array([[0.0, 0.0, 10.0, 10.0], [10.0, 0.0, 20.0, 10.0]],
                        dtype=np.float32)
    scores = np.array([0.9, 0.8], dtype=np.float32)
    assert len(O.nms(touching, scores, 0.0)) == 2, (
        "two boxes sharing an edge overlap by nothing; at a threshold of 0 both stay")
    overlapping = np.array([[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 10.0, 10.0]],
                           dtype=np.float32)
    assert len(O.nms(overlapping, scores, 0.0)) == 1


def test_nms_returns_the_kept_indices_in_score_order():
    """Not the boxes, and not in the input's order. The caller almost always has
    labels and scores to filter by the same positions, which is why every filter here
    answers with positions."""
    scores = np.array([0.1, 0.9, 0.5], dtype=np.float32)
    kept = O.nms(_A, scores, 0.9)
    assert list(kept) == [1, 2, 0]


def test_batched_nms_keeps_classes_apart():
    """Two identical boxes in **different classes** both survive; in the same class one
    goes. The offset trick is what makes a single pass do that, and this is the only
    thing that can tell it worked."""
    same = np.array([[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 10.0, 10.0]], dtype=np.float32)
    scores = np.array([0.9, 0.8], dtype=np.float32)
    assert len(O.batched_nms(same, scores, np.array([0, 1]), 0.5)) == 2
    assert len(O.batched_nms(same, scores, np.array([0, 0]), 0.5)) == 1


def test_masks_to_boxes_answers_zeros_for_a_blank_mask():
    """torchvision does not raise here, and the reason is stacking: one blank plane in
    a batch would otherwise take the whole batch down."""
    masks = np.zeros((2, 5, 5), dtype=np.uint8)
    masks[0, 1:3, 2:4] = 1
    boxes = O.masks_to_boxes(masks)
    assert list(boxes[0]) == [2, 1, 3, 2]
    assert list(boxes[1]) == [0, 0, 0, 0]


def test_a_wrong_format_is_refused_rather_than_computed():
    """Four numbers are four numbers, so `xywh` read as `xyxy` is a wrong answer with
    nothing raised. The name is the only thing that can catch it, so an unknown one
    stops."""
    with pytest.raises(ValueError, match="format"):
        O.box_area(_A, "yxyx")
    with pytest.raises(ValueError, match="Conversions"):
        O.box_convert(_A, "xyxy", "yxyx")


def test_the_ops_take_and_return_the_kind_they_were_given():
    """A caller holding tensors should not have to unwrap them to ask a question about
    geometry — `Normalize`'s rule, applied to the namespace that is all arithmetic."""
    as_tensor = BT.tensor(_A)
    assert isinstance(O.box_area(_A), np.ndarray)
    assert not isinstance(O.box_area(as_tensor), np.ndarray)
    assert np.allclose(O.box_area(as_tensor).numpy(), O.box_area(_A))
