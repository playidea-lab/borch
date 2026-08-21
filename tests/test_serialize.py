"""**Where the golden cases do not ask** about `save`/`load`.

The golden cases ask about values — read back what was written, is it the same. That runs in
all three implementations and really does run. What is asked here is outside that.

- **Refusal.** Does it quietly store what this format cannot hold? Refusal code is easy to
  write and, never once exercised, stays as wording with no behaviour behind it.
- **Byte equality.** Saving the same thing twice, is it the same file? That is why the name
  order is fixed, and unasked it breaks quietly the day dictionary order changes.
- **Somebody else's file.** Does a safetensors file with no `borch.tree` read as a flat dict?
  What the browser writes has that shape, so this branch is a real path.
- **The binding's branch.** The codec is one, but the two functions that take a tensor apart
  and put it back are different. numpy is fitted into that place, without a browser, to see
  whether the same bytes come out.
"""

import pathlib
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import borch
from borch._base import BorchError
from borch._serialize import dump, encode, parse


def _tmp(name):
    return pathlib.Path(tempfile.mkdtemp()) / name


def test_save_refuses_integers_float32_cannot_hold_exactly():
    # An integer past 2^24 is indistinguishable from its neighbour in f32. The body being f32
    # is a choice made so borch.ts can read it, so that price must not be paid quietly.
    big = borch.tensor(np.array([2 ** 24 + 1], dtype=np.int64))
    with pytest.raises(BorchError, match="integer too large"):
        borch.save({"labels": big}, _tmp("big.bin"))


def test_save_accepts_integers_at_the_exact_boundary():
    ok = borch.tensor(np.array([2 ** 24], dtype=np.int64))
    path = _tmp("edge.bin")
    borch.save({"labels": ok}, path)
    assert int(borch.load(path)["labels"].data[0]) == 2 ** 24


def test_save_refuses_an_object_the_format_cannot_hold():
    # pickle held anything. This format cannot, and what it cannot hold it has to say.
    with pytest.raises(BorchError, match="cannot be stored"):
        borch.save({"fn": len}, _tmp("obj.bin"))


def test_save_refuses_complex():
    z = borch.tensor(np.array([1 + 2j], dtype=np.complex64))
    with pytest.raises(BorchError, match="is complex"):
        borch.save({"z": z}, _tmp("z.bin"))


def test_save_refuses_two_paths_that_flatten_to_one_name():
    # `{"a": {"b": t}}` and `{"a.b": t}` flatten to the same name. If one overwrites the other,
    # both places hold the same value on the way back, which is worse than an exception.
    t = borch.tensor(np.zeros(2, dtype=np.float32))
    with pytest.raises(BorchError, match="appears twice"):
        borch.save({"a": {"b": t}, "a.b": t}, _tmp("dup.bin"))


def test_saving_twice_gives_the_same_bytes():
    obj = {"w": borch.tensor(np.arange(6, dtype=np.float32).reshape(2, 3)),
           "b": borch.tensor(np.zeros(3, dtype=np.float32)),
           "epoch": 7}
    first, second = _tmp("a.bin"), _tmp("b.bin")
    borch.save(obj, first)
    borch.save(obj, second)
    assert first.read_bytes() == second.read_bytes()


def test_load_reads_a_file_object_and_save_writes_one():
    obj = {"w": borch.tensor(np.array([1.5, -2.5], dtype=np.float32))}
    path = _tmp("handle.bin")
    with open(path, "wb") as f:
        borch.save(obj, f)
    with open(path, "rb") as f:
        got = borch.load(f)
    assert np.array_equal(got["w"].data, obj["w"].data)


def test_load_ignores_weights_only_and_map_location():
    # torch code passes these two along. This format executes no code, so it is always on the
    # `weights_only` side, and merely accepting them keeps the import-only-change path open.
    path = _tmp("kw.bin")
    borch.save({"w": borch.tensor(np.ones(2, dtype=np.float32))}, path)
    got = borch.load(path, weights_only=True, map_location="cpu")
    assert np.array_equal(got["w"].data, np.ones(2, dtype=np.float32))


def test_load_refuses_an_argument_it_does_not_know():
    path = _tmp("kw2.bin")
    borch.save({"w": borch.tensor(np.ones(2, dtype=np.float32))}, path)
    with pytest.raises(BorchError, match="unexpected argument"):
        borch.load(path, pickle_module=None)


def test_a_foreign_safetensors_reads_as_a_flat_dict():
    # A file with no `borch.tree` — what borch.ts writes has this shape, and so does what
    # somebody else's tool writes. With no tree, do not invent nesting; hand it back flat.
    blob = encode({"fc.weight": np.array([[1.0, 2.0]], dtype=np.float32)})
    path = _tmp("foreign.bin")
    path.write_bytes(blob)
    got = borch.load(path)
    assert sorted(got) == ["fc.weight"]
    assert np.array_equal(got["fc.weight"].data, np.array([[1.0, 2.0]], dtype=np.float32))


def test_a_truncated_file_is_refused_not_guessed():
    path = _tmp("short.bin")
    path.write_bytes(b"\x00\x03")
    with pytest.raises(BorchError, match="too short"):
        borch.load(path)


def test_a_header_longer_than_the_file_is_refused():
    path = _tmp("liar.bin")
    path.write_bytes((10 ** 6).to_bytes(8, "little") + b"{}")
    with pytest.raises(BorchError, match="runs past the file"):
        borch.load(path)


def test_the_binding_shaped_hooks_write_the_same_bytes_as_the_core():
    """**Does the core read a file the binding wrote** — how to measure that place with no browser.

    The binding uses the codec as it is and changes two branches only (the test for recognising
    a tensor, and the path for taking a value out and putting it back). Imitating those two
    branches with numpy produces the bytes the other side would write.

    Without this, "the codec is one, so they are the same" is a claim and not a measurement.
    This project believed that one claim and went a long time without seeing that the Python
    side was using pickle.
    """
    class Fake:                       # standing in for the binding's `Tensor`
        def __init__(self, array):
            self.array = array

    def array_of(obj):
        return obj.array if isinstance(obj, Fake) else None

    payload = {"model": {"fc.weight": Fake(np.array([[1.5, -2.25]], dtype=np.float32))},
               "epoch": 3}
    theirs = _tmp("binding.bin")
    dump(payload, theirs, array_of)

    # The core opens that file — structure and values have to come through unchanged.
    got = borch.load(theirs)
    assert sorted(got) == ["epoch", "model"]
    assert got["epoch"] == 3
    assert np.array_equal(got["model"]["fc.weight"].data,
                          np.array([[1.5, -2.25]], dtype=np.float32))

    # And when the core writes the same thing, it has to be **the same bytes.**
    mine = _tmp("core.bin")
    borch.save({"model": {"fc.weight": borch.tensor(
        np.array([[1.5, -2.25]], dtype=np.float32))}, "epoch": 3}, mine)
    assert theirs.read_bytes() == mine.read_bytes()


def test_parse_gives_back_what_dump_took_for_lists_and_tuples():
    # A mix of lists and tuples has to keep its shape too — optimizer state has that shape.
    payload = {"xs": [borch.tensor(np.ones(2, dtype=np.float32)), 1, "two"],
               "pair": (3, 4.5)}
    path = _tmp("mixed.bin")
    borch.save(payload, path)
    got = borch.load(path)
    assert isinstance(got["xs"], list) and isinstance(got["pair"], tuple)
    assert got["xs"][1] == 1 and got["xs"][2] == "two"
    assert got["pair"] == (3, 4.5)


def test_dtype_labels_survive_the_round_trip():
    payload = {"labels": borch.tensor(np.array([3, 1, 4], dtype=np.int64)),
               "flags": borch.tensor(np.array([True, False], dtype=np.bool_))}
    path = _tmp("labels.bin")
    borch.save(payload, path)
    got = borch.load(path)
    assert got["labels"].data.dtype == np.int64
    assert got["flags"].data.dtype == np.bool_
    assert list(got["labels"].data) == [3, 1, 4]


def test_parse_is_the_same_function_the_binding_calls():
    # `parse` is called directly — this is what the binding calls, and unasked here that path
    # runs only in the browser, where a failure shows up late.
    blob = encode({"w": np.array([2.0], dtype=np.float32)})
    path = _tmp("parse.bin")
    path.write_bytes(blob)
    got = parse(path, lambda a: np.asarray(a) * 10)
    assert got["w"][0] == 20.0
