"""The workbench: one call carrying every setting, held to what it claims.

**There is no golden here and there cannot be.** torch has no `setup()`, so nothing can be
frozen from it — the same position `peft` is in. What holds this is invariants: that the
settings reach the run, that a number says what it was measured on, that a refusal arrives
before the work rather than in the middle of it.

It runs on the numpy core, which is the point of the core holding the implementation: the
browser surfaces add a hub and an exporter and change nothing else, so the loop is verified
here, in a second, without a browser.
"""

import io
import zipfile

import numpy as np
import pytest

import borch
from borch._workbench import Session


def _png(rgb):
    """A tiny PNG without Pillow — zlib and struct, as `tests/browser` does it."""
    import struct
    import zlib
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[r].tobytes() for r in range(h))
    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xffffffff)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def _zip_of(n=24, size=32, classes=("cat", "dog", "bird")):
    """A folder of learnable images: one flat colour per class, plus noise."""
    rng = np.random.default_rng(0)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for i in range(n):
            which = i % len(classes)
            base = np.zeros((size, size, 3), dtype=np.float32)
            base[:, :, which] = 1.0
            noisy = np.clip(base + rng.normal(0, 0.05, base.shape), 0, 1)
            z.writestr(f"{classes[which]}/{classes[which]}_{i:03d}.png",
                       _png((noisy * 255).astype(np.uint8)))
    return [("set.zip", buf.getvalue())]


@pytest.fixture(scope="module")
def files():
    return _zip_of()


def test_setup_refuses_an_optimizer_it_does_not_have_before_any_work(files):
    with pytest.raises(ValueError, match="optimizer"):
        Session(borch, files, optimizer="lbfgs")


def test_setup_refuses_a_backbone_on_a_surface_with_no_hub(files):
    """**The numpy core can never fetch one, so the refusal belongs at setup.**"""
    with pytest.raises(ValueError, match="hub"):
        Session(borch, files, backbone="imagenet-efficientnet-b0")


def test_setup_refuses_a_validation_share_that_is_not_a_share(files):
    with pytest.raises(ValueError, match="val"):
        Session(borch, files, val=1.0)


def test_the_settings_given_are_the_settings_recorded(files):
    s = Session(borch, files, size=32, batch=8, epochs=3, lr=0.05, val=0.25, seed=7)
    assert s.config["size"] == 32 and s.config["batch"] == 8 and s.config["epochs"] == 3
    assert s.config["lr"] == 0.05 and s.config["val"] == 0.25 and s.config["seed"] == 7
    assert s.config["images"] == 24 and s.config["classes"] == ["bird", "cat", "dog"]


def test_fit_learns_the_three_colours_and_says_what_it_measured_on(files):
    s = Session(borch, files, size=32, batch=8, epochs=8, lr=0.05).fit()
    assert s.losses[-1] < s.losses[0], f"the loss did not fall: {s.losses}"
    assert s.accuracy >= 0.9, f"three flat colours should be learnable: {s.accuracy}"
    assert s.measured_on == "the training images"


def test_a_held_out_share_is_named_as_held_out_and_is_not_the_training_rows(files):
    s = Session(borch, files, size=32, batch=8, epochs=8, lr=0.05, val=0.25, seed=1).fit()
    assert s.measured_on == "held-out"
    train, scored = s._split()
    assert not set(train.tolist()) & set(scored.tolist())
    assert len(scored) == 6


def test_the_same_seed_gives_the_same_run(files):
    a = Session(borch, files, size=32, batch=8, epochs=3, lr=0.05, seed=3).fit()
    b = Session(borch, files, size=32, batch=8, epochs=3, lr=0.05, seed=3).fit()
    assert a.losses == b.losses


def test_a_model_you_pass_is_the_model_that_trains(files):
    nn = borch.nn
    mine = nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, 3))
    before = [p.numpy().copy() for p in mine.parameters()]
    s = Session(borch, files, size=32, batch=8, epochs=4, lr=0.01).fit(mine)
    assert s.model is mine
    assert any(not np.allclose(b, p.numpy()) for b, p in zip(before, mine.parameters()))


def test_suspects_before_fit_says_so_rather_than_returning_zeros(files):
    s = Session(borch, files, size=32, batch=8, epochs=2)
    with pytest.raises(RuntimeError, match="fit"):
        _ = s.suspects


def test_the_review_queue_puts_a_wrong_label_near_the_front():
    """One image is filed under the wrong colour; the queue is meant to raise it."""
    rng = np.random.default_rng(1)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for i in range(30):
            which = i % 3
            base = np.zeros((32, 32, 3), dtype=np.float32)
            base[:, :, which] = 1.0
            noisy = np.clip(base + rng.normal(0, 0.05, base.shape), 0, 1)
            name = ["cat", "dog", "bird"][which]
            if i == 7:                                   # a blue image filed as a cat
                name = "cat"
                noisy = np.clip(np.dstack([np.zeros((32, 32, 2), dtype=np.float32),
                                           np.ones((32, 32, 1), dtype=np.float32)])
                                + rng.normal(0, 0.05, base.shape), 0, 1)
            z.writestr(f"{name}/{name}_{i:03d}.png", _png((noisy * 255).astype(np.uint8)))
    s = Session(borch, [("set.zip", buf.getvalue())], size=32, batch=8, epochs=8, lr=0.05).fit()
    assert list(s.order).index(7) < 10, f"the planted wrong label sat at {list(s.order).index(7)}"


def test_facts_carry_the_settings_and_the_result_flat(files):
    s = Session(borch, files, size=32, batch=8, epochs=3, lr=0.05, val=0.25)
    early = s.facts()
    assert early["cfg_epochs"] == 3 and "accuracy" not in early
    got = s.fit().facts()
    assert got["accuracy"] == round(s.accuracy, 4) and got["measured_on"] == "held-out"
    assert got["cfg_classes"] == "bird,cat,dog"
    assert all(isinstance(v, (int, float, str, type(None))) for v in got.values()), \
        "report takes flat facts; a nested value would land as a repr"


def test_onnx_on_a_surface_without_an_exporter_names_the_one_that_has_it(files):
    s = Session(borch, files, size=32, batch=8, epochs=2).fit()
    with pytest.raises(RuntimeError, match="borch_webgpu"):
        s.onnx()


def test_a_set_already_decoded_is_taken_as_it_is(files):
    """A page that decoded a folder once should not decode it twice to use this."""
    from borch._data import ImageFiles

    ds = ImageFiles(files, size=32)
    s = Session(borch, ds, epochs=2, batch=8)
    assert s.data is ds and s.config["size"] == 32


def test_a_size_that_contradicts_the_set_is_refused(files):
    from borch._data import ImageFiles

    ds = ImageFiles(files, size=32)
    with pytest.raises(ValueError, match="decoded at 32"):
        Session(borch, ds, size=96)


def test_how_names_the_model_that_actually_trained(files):
    """**The page's line quotes this.** It said "your model" for the small CNN as well,
    because the check was on a field `fit()` had just filled in either way."""
    nn = borch.nn
    mine = nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, 3))
    assert Session(borch, files, size=32, batch=8, epochs=2).fit().how.startswith("small CNN")
    assert Session(borch, files, size=32, batch=8, epochs=2).fit(mine).how.startswith("your model")


def test_a_backbone_and_a_size_together_are_refused_as_the_contradiction_they_are(files):
    """**The manifest says what its weights were trained on.** Asking for another size is
    not a preference, it is a disagreement — and measured, losing that argument costs up
    to sixty-three points of top-1."""
    with pytest.raises(ValueError, match="prepares its own images"):
        Session(borch, files, backbone="imagenet-efficientnet-b0", size=64)


def test_without_a_backbone_the_size_is_the_callers_and_defaults_to_64(files):
    assert Session(borch, files).config["size"] == 64
    assert Session(borch, files, size=96).config["size"] == 96


def test_a_set_already_decoded_still_settles_the_size(files):
    from borch._data import ImageFiles

    ds = ImageFiles(files, size=32)
    assert Session(borch, ds).config["size"] == 32


def test_the_original_photograph_is_reachable_for_a_transform_that_resizes_itself(files):
    """`ImageFiles` hands back squares; a manifest's pipeline needs what arrived."""
    from borch._data import ImageFiles

    ds = ImageFiles(files, size=16)
    assert ds[0][0].shape == (3, 16, 16)
    assert ds.raw(0).size == (32, 32)          # the synthetic set is 32 px square


# -- standardising the cached features ----------------------------------------------

def test_the_scaling_is_learned_from_the_training_rows_alone(files):
    """**Statistics taken from everything let the held-out rows set their own bar.**

    A small leak, and the kind that makes a number look better than it is. It is checked
    by construction: the mean of the *training* rows lands on zero, and the mean of all
    the rows does not have to.
    """
    s = Session(borch, files, size=32, batch=8, epochs=2, val=0.25, seed=5)
    s.features = np.vstack([np.full((18, 4), 1.0, dtype=np.float32),
                            np.full((6, 4), 9.0, dtype=np.float32)])
    train, scored = s._split()
    s.features[scored] = 9.0                       # the held-out rows are far away
    s.features[train] = 1.0
    s._standardise(train)
    assert np.allclose(s.features[train].mean(axis=0), 0.0, atol=1e-5)
    assert not np.allclose(s.features[scored].mean(axis=0), 0.0, atol=1e-5), \
        "the held-out rows moved to zero too — the statistics saw them"


def test_a_dimension_that_never_varies_does_not_become_an_infinity(files):
    s = Session(borch, files, size=32, batch=8, epochs=2)
    s.features = np.hstack([np.random.default_rng(0).normal(size=(24, 3)).astype(np.float32),
                            np.full((24, 1), 7.0, dtype=np.float32)])
    s._standardise(np.arange(24))
    assert np.isfinite(s.features).all()


# -- comparing and picking ----------------------------------------------------------

class _Row(dict):
    pass


class _FakeHub:
    """A registry with three models: two that could take a photograph and one that could not."""

    @staticmethod
    def list():
        return [
            {"name": "small", "bytes": 10_000_000, "task": "image-classification"},
            {"name": "large", "bytes": 40_000_000, "task": "image-classification"},
            {"name": "huge", "bytes": 400_000_000, "task": "image-classification"},
            {"name": "speech", "bytes": 5_000_000, "task": "audio-classification"},
        ]


def test_candidates_are_smallest_first_of_the_task_asked_for_and_under_the_budget():
    """**The budget belongs here, not in the caller.**

    It was a parameter this function accepted and never read — `compare` filtered
    afterwards — so `candidates(budget_mb=10)` handed back everything. Caught by
    `test_unread_arguments.py`, which exists for exactly that.
    """
    import types

    from borch._workbench import candidates

    fake = types.SimpleNamespace(hub=_FakeHub)
    assert [r["name"] for r in candidates(fake, budget_mb=60)] == ["small", "large"]
    assert [r["name"] for r in candidates(fake, budget_mb=500)] == ["small", "large", "huge"]
    assert [r["name"] for r in candidates(fake, budget_mb=12)] == ["small"]
    # `speech` is 5 MB and under every budget above; it is another task.
    assert all("speech" not in r["name"] for r in candidates(fake, budget_mb=500))


def test_a_comparison_without_held_out_rows_is_refused(files):
    import types

    from borch._workbench import compare

    fake = types.SimpleNamespace(hub=_FakeHub)
    with pytest.raises(ValueError, match="held-out"):
        compare(fake, files, val=0.0)


class _FakeSession:
    """Stands in for a fit: says what it was asked for and how well it did."""

    tried = []
    scores = {}

    def __init__(self, torch, files, **config):
        self.config = dict(config)
        self.config.setdefault("size", 224)
        self.accuracy = self.scores.get(config["backbone"], 0.5)
        self.measured_on = "held-out"
        self.held_out = 100
        self.seconds = 1.0
        self.features = np.zeros((4, 8), dtype=np.float32)

    def fit(self):
        _FakeSession.tried.append(self.config["backbone"])
        return self


def _with_fake_sessions(monkeypatch, scores):
    import borch._workbench as wb

    _FakeSession.tried, _FakeSession.scores = [], scores
    monkeypatch.setattr(wb, "Session", _FakeSession)
    return _FakeSession


def test_compare_tries_every_candidate_under_the_budget_and_no_more(monkeypatch, files):
    import types

    from borch._workbench import compare

    fake = _with_fake_sessions(monkeypatch, {"small": 0.7, "large": 0.8})
    rows = compare(types.SimpleNamespace(hub=_FakeHub), files, budget_mb=60)
    assert fake.tried == ["small", "large"], fake.tried       # huge is 400 MB
    assert [r["name"] for r in rows] == ["small", "large"]
    assert all(r["measured_on"] == "held-out" for r in rows)


def test_every_candidate_sees_the_same_split(monkeypatch, files):
    """**A leaderboard where the models saw different held-out rows ranks the splits.**"""
    import types

    from borch._workbench import compare

    seen = []

    class _Recording(_FakeSession):
        def __init__(self, torch, files, **config):
            super().__init__(torch, files, **config)
            seen.append((config["val"], config["seed"]))

    import borch._workbench as wb

    _Recording.tried, _Recording.scores = [], {}
    monkeypatch.setattr(wb, "Session", _Recording)
    compare(types.SimpleNamespace(hub=_FakeHub), files, budget_mb=60, val=0.25, seed=11)
    assert seen == [(0.25, 11), (0.25, 11)], seen


def test_pick_stops_at_the_first_that_clears_and_never_fetches_the_rest(monkeypatch, files):
    """The reason `pick` exists: the larger weights are never downloaded."""
    import types

    from borch._workbench import pick

    fake = _with_fake_sessions(monkeypatch, {"small": 0.93, "large": 0.99})
    name, rows = pick(types.SimpleNamespace(hub=_FakeHub), files, at_least=0.9, budget_mb=60)
    assert name == "small"
    assert fake.tried == ["small"], "it went on after the bar was cleared"
    assert len(rows) == 1


def test_pick_says_nothing_cleared_rather_than_handing_back_the_best_of_a_bad_set(monkeypatch, files):
    import types

    from borch._workbench import pick

    _with_fake_sessions(monkeypatch, {"small": 0.4, "large": 0.5})
    name, rows = pick(types.SimpleNamespace(hub=_FakeHub), files, at_least=0.9, budget_mb=60)
    assert name is None and len(rows) == 2


def test_a_candidate_that_refuses_is_kept_in_the_rows_rather_than_dropped(monkeypatch, files):
    """`cifar10-resnet18` refuses every photograph, and knowing that is part of the answer."""
    import types

    import borch._workbench as wb
    from borch._workbench import compare

    class _Refusing(_FakeSession):
        def fit(self):
            if self.config["backbone"] == "small":
                raise ValueError("this manifest prepares a 3x32x32 image")
            return super().fit()

    _Refusing.tried, _Refusing.scores = [], {"large": 0.8}
    monkeypatch.setattr(wb, "Session", _Refusing)
    rows = compare(types.SimpleNamespace(hub=_FakeHub), files, budget_mb=60)
    assert [r["name"] for r in rows] == ["small", "large"]
    assert "error" in rows[0] and "3x32x32" in rows[0]["error"]
    assert rows[1]["accuracy"] == 0.8


# -- what a short board can and cannot say ------------------------------------------

def test_the_interval_is_wider_than_the_gaps_this_workbench_measured():
    """**The board it produced over CIFAR-10 does not rank anything**, and this says so.

    0.720, 0.670 and 0.760 on a hundred held-out rows: nine points apart, against an
    interval of about eight and a half on each one and a resolution of twelve between two
    of them. Ordering those three by accuracy is ordering the split.
    """
    from borch._workbench import interval, resolution

    assert 0.08 < interval(0.75, 100) < 0.09
    assert 0.11 < resolution(100) < 0.13
    assert resolution(100) > (0.760 - 0.670), "the measured spread is inside the noise"
    # More rows, a finer instrument.
    assert resolution(1000) < resolution(100) < resolution(50)
    assert 0.03 < resolution(1000) < 0.04


def test_a_board_within_the_resolution_is_marked_as_one_answer(monkeypatch, files):
    import types

    from borch._workbench import compare

    _with_fake_sessions(monkeypatch, {"small": 0.72, "large": 0.76})
    rows = compare(types.SimpleNamespace(hub=_FakeHub), files, budget_mb=60, val=0.25)
    assert all(r["ties_with_best"] for r in rows), \
        "four points apart on a handful of rows is not a ranking"


def test_a_board_that_is_far_apart_is_not_marked_as_tied(monkeypatch, files):
    import types

    from borch._workbench import compare

    _with_fake_sessions(monkeypatch, {"small": 0.20, "large": 0.95})
    rows = compare(types.SimpleNamespace(hub=_FakeHub), files, budget_mb=60, val=0.25)
    by = {r["name"]: r for r in rows}
    assert by["large"]["ties_with_best"] and not by["small"]["ties_with_best"]


def test_the_board_says_out_loud_what_it_cannot_separate(monkeypatch, files):
    import types

    from borch._workbench import compare, say

    _with_fake_sessions(monkeypatch, {"small": 0.72, "large": 0.76})
    text = say(compare(types.SimpleNamespace(hub=_FakeHub), files, budget_mb=60, val=0.25))
    assert "points apart and no closer" in text
    assert "does not rank" in text and "take the smallest" in text


# -- masks: the data says which task this is ----------------------------------------

def _discs_and_masks(n=16, side=32):
    """Bright discs on noise, and the mask that says where each one is."""
    rng = np.random.default_rng(2)
    pics, masks = [], []
    for i in range(n):
        cx, cy, r = rng.integers(10, side - 10, 2).tolist() + [6]
        ys, xs = np.mgrid[0:side, 0:side]
        inside = ((xs - cx) ** 2 + (ys - cy) ** 2) <= r * r
        pic = np.clip(rng.normal(0.25, 0.05, (side, side, 3)), 0, 1)
        pic[inside] = np.clip(pic[inside] + 0.6, 0, 1)
        pics.append((f"a/a_{i:03d}.png", _png((pic * 255).astype(np.uint8))))
        m = np.repeat((inside * 255).astype(np.uint8)[:, :, None], 3, axis=2)
        masks.append((f"a/a_{i:03d}.png", _png(m)))
    return _zip_from(pics), _zip_from(masks)


def _zip_from(members):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, data in members:
            z.writestr(name, data)
    return [("set.zip", buf.getvalue())]


@pytest.fixture(scope="module")
def discs():
    from borch._data import ImageFiles

    pics, masks = _discs_and_masks()
    return ImageFiles(pics, size=32), ImageFiles(masks, size=32)


def test_masks_beside_the_images_are_what_makes_it_segmentation(discs):
    """**No `task=` flag.** A second name for the same fact is a second thing to disagree."""
    pics, masks = discs
    s = Session(borch, pics, masks=masks, epochs=2, batch=8)
    assert s.masks is masks and s.accuracy is None


def test_a_u_net_learns_the_discs_and_the_score_is_an_overlap(discs):
    pics, masks = discs
    s = Session(borch, pics, masks=masks, epochs=12, batch=8, lr=1e-2).fit()
    assert s.score_name == "mean IoU" and s.score == s.iou
    assert s.accuracy is None, "an overlap is not an accuracy, and calling it one is the quiet kind of wrong"
    assert s.iou > 0.5, f"the discs were not learned: IoU {s.iou}"
    assert s.predicted.shape == (len(pics), 1, 32, 32)
    assert s.losses[-1] < s.losses[0]


def test_the_review_queue_for_masks_is_one_minus_the_overlap(discs):
    pics, masks = discs
    s = Session(borch, pics, masks=masks, epochs=8, batch=8, lr=1e-2).fit()
    doubt = s.suspects
    assert doubt.shape == (len(pics),) and (0.0 <= doubt).all() and (doubt <= 1.0).all()
    # The worst-drawn mask is the one at the front of the queue.
    assert s.order[0] == int(np.argmax(doubt))


def test_a_backbone_with_masks_is_refused_because_a_vector_is_not_a_mask(discs):
    pics, masks = discs
    with pytest.raises(ValueError, match="one answer per pixel"):
        Session(borch, pics, masks=masks, backbone="imagenet-efficientnet-b0")


def test_masks_that_do_not_match_the_images_are_counted_and_refused(discs):
    from borch._data import ImageFiles

    pics, masks = discs
    fewer_pics, fewer_masks = _discs_and_masks(n=4)
    with pytest.raises(ValueError, match="a mask for each image"):
        Session(borch, pics, masks=ImageFiles(fewer_masks, size=32), epochs=1).fit()
