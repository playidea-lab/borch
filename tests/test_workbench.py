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
