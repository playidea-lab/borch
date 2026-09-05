"""`decode_images` and `suspects` — the two calls the workbench notebook shrank into.

Both live in the core (`borch._data`) so they run here without a browser; the binding
re-exports them and only adds how Pillow arrives in Pyodide.
"""
import io

import numpy as np
import pytest

from borch._data import ImageFiles, decode_images, label_from_name, suspects

PIL = pytest.importorskip("PIL.Image")


def _png(rgb, side=8):
    img = PIL.new("RGB", (side, side), rgb)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_label_from_name_uses_the_folder_when_there_is_one_else_the_prefix():
    assert label_from_name("cats/001.png") == "cats"
    assert label_from_name("data/train/dogs/x_1.png") == "dogs"
    assert label_from_name("cat_001.png") == "cat"
    assert label_from_name("plain.png") == "plain.png"


def test_decode_images_from_name_bytes_pairs_returns_nchw_in_unit_range():
    files = [("b_1.png", _png((0, 0, 255))), ("a_1.png", _png((255, 0, 0))), ("a_2.png", _png((255, 0, 0)))]
    x, y, names, classes = decode_images(files, size=4)
    assert x.shape == (3, 3, 4, 4) and x.dtype == np.float32
    assert classes == ["a", "b"]
    assert y.tolist() == [1, 0, 0] and y.dtype == np.int64
    assert names == ["b_1.png", "a_1.png", "a_2.png"]
    assert x[0, 2].min() == pytest.approx(1.0) and x[0, 0].max() == pytest.approx(0.0)   # blue image, B channel


def test_decode_images_accepts_upload_shaped_objects_and_a_custom_label():
    class Upload:
        def __init__(self, name, contents):
            self.name, self.contents = name, contents
    x, y, names, classes = decode_images([Upload("x.png", _png((1, 2, 3)))], size=2, label=lambda n: "only")
    assert classes == ["only"] and y.tolist() == [0] and x.shape == (1, 3, 2, 2)


def test_decode_images_with_nothing_or_a_strange_item_raises():
    with pytest.raises(ValueError):
        decode_images([])
    with pytest.raises(TypeError):
        decode_images([42])


def test_suspects_flags_the_one_label_that_disagrees_with_its_neighbours():
    rng = np.random.default_rng(0)
    centres = np.eye(3, dtype=np.float32) * 10
    feats = np.concatenate([centres[c] + rng.normal(scale=0.1, size=(10, 3)) for c in range(3)]).astype(np.float32)
    labels = np.repeat([0, 1, 2], 10)
    labels[4] = 2                                  # one wrong label inside cluster 0
    score = suspects(feats, labels, k=5)
    assert score.shape == (30,) and score.dtype == np.float32
    assert score[4] == pytest.approx(1.0)          # all five neighbours say 0
    others = np.delete(score, 4)
    assert others.max() <= 0.2                     # at most the one wrong neighbour


def test_suspects_takes_tensors_and_handles_tiny_inputs():
    import borch
    f = borch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])
    y = borch.tensor([0, 0, 1])
    assert suspects(f, y, k=1).tolist() == [0.0, 0.0, 1.0]
    assert suspects(np.zeros((1, 2), np.float32), np.zeros(1, np.int64)).tolist() == [0.0]
    with pytest.raises(ValueError):
        suspects(np.zeros((3, 2), np.float32), np.zeros(2, np.int64))


def _zip_of(entries):
    import zipfile
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, data in entries:
            z.writestr(name, data)
    return buf.getvalue()


def test_image_files_reads_a_zipped_folder_with_folder_labels_and_decodes_on_demand():
    z = _zip_of([("cats/a.png", _png((255, 0, 0))), ("dogs/b.png", _png((0, 255, 0))),
                 ("dogs/c.png", _png((0, 0, 255))), ("__MACOSX/cats/._a.png", b"junk"), ("notes.txt", b"x")])
    ds = ImageFiles([("photos.zip", z)], size=4)
    assert len(ds) == 3 and ds.classes == ["cats", "dogs"] and ds.targets.tolist() == [0, 1, 1]
    assert ds.names == ["cats/a.png", "dogs/b.png", "dogs/c.png"]
    x, y = ds[1]
    assert x.shape == (3, 4, 4) and y == 1 and x[1].min() == pytest.approx(1.0)
    assert ds.thumb(0, 2).shape == (2, 2, 3) and ds.thumb(0, 2).dtype == np.uint8
    got = [(x.shape, idx.tolist()) for x, idx in ds.batches(2)]
    assert got == [((2, 3, 4, 4), [0, 1]), ((1, 3, 4, 4), [2])]
    assert ds.stack().shape == (3, 3, 4, 4)


def test_decode_images_takes_a_zip_beside_plain_files():
    z = _zip_of([("k/one.png", _png((1, 2, 3)))])
    x, y, names, classes = decode_images([("a_0.png", _png((9, 9, 9))), ("folder.zip", z)], size=2)
    assert x.shape == (2, 3, 2, 2) and classes == ["a", "k"] and y.tolist() == [0, 1]
