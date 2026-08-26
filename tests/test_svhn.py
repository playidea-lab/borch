"""**`SVHN`, and the MATLAB reader it needed.**

This dataset was declined, and the reason grouped two different walls under one
sentence: *the refusal is the dependency, and it is the same answer PIL and a JPEG
decoder get.* A JPEG decoder is a **codec** — thousands of lines, decades of edge
cases, and no reasonable way to write one here. A `.mat` is a **documented container**:
a header, tagged elements, `zlib` around them. The reader is under a hundred lines of
`struct` and `zlib`, both already imported.

Grouping them made the cheap one look as expensive as the dear one, and it stood for
as long as nobody asked which files the dataset actually reads. **`SVHN` reads no
picture at all** — the pixels arrive as a `uint8` array inside the `.mat`.

## Why the fixture is written rather than downloaded

The real file is 180MB behind a host that sends no CORS header, so no test here can
fetch it. `scipy.io.savemat` writes the same format, and **`scipy` is the authority
this reader is checked against** — it is a test dependency and never a library one,
which is the whole point of having written the reader.

The synthetic file carries both of the traps deliberately:

- **a label of 10**, which is SVHN's name for the digit zero. Left alone it is an
  index one past the end of a ten-class problem, and `CrossEntropyLoss` reads out of
  range rather than complaining.
- **an array in `(32, 32, 3, N)` order**, MATLAB's own. Transposed the wrong way it is
  still a stack of colour pictures of the right size, of something else — so the shape
  agreeing proves nothing and the values have to be compared.

## What the reader tests separately

`test_the_mat_reader_agrees_with_scipy` covers the container rather than the dataset:
six shapes, both compressed and not. Two of its rows exist because they failed —
**a top-level element is not padded to eight bytes** (applying the padding walked four
bytes past the first variable and lost the second), and **one file holds several
matrices** (a reader that stops after the first finds `X` and loses `y`, giving
pictures with no labels and a file that parses).
"""

import io
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

scipy_io = pytest.importorskip("scipy.io")
pytest.importorskip("torchvision")

import torchvision.datasets as T                                 # noqa: E402

import borchvision as V                                          # noqa: E402

# Shapes chosen so a column-major mistake cannot hide: every axis a different length.
SHAPES = {
    "svhn shaped": {"X": np.arange(32 * 32 * 3 * 5, dtype=np.uint8)
                    .reshape(32, 32, 3, 5),
                    "y": np.array([[1], [10], [3], [10], [7]], dtype=np.uint8)},
    "non-square 2-D": {"a": np.arange(21, dtype=np.int32).reshape(3, 7)},
    "vector": {"v": np.arange(9, dtype=np.uint8).reshape(9, 1)},
    "3-D, all axes differ": {"c": np.arange(24, dtype=np.int16).reshape(2, 3, 4)},
    "float64": {"f": np.linspace(-1, 1, 24).reshape(4, 6)},
    "float32": {"g": np.linspace(0, 1, 10, dtype=np.float32).reshape(5, 2)},
}


@pytest.mark.parametrize("label", sorted(SHAPES))
@pytest.mark.parametrize("compress", [True, False])
def test_the_mat_reader_agrees_with_scipy(label, compress):
    """The container, against the library this exists so as not to depend on."""
    payload = SHAPES[label]
    buf = io.BytesIO()
    scipy_io.savemat(buf, payload, do_compression=compress)
    raw = buf.getvalue()

    want = scipy_io.loadmat(io.BytesIO(raw))
    got = V._mat_read(raw)
    for name, _ in payload.items():
        assert name in got, (
            f"`{name}` is missing from what the reader returned: {sorted(got)}.\n"
            "  One compressed element can hold several matrices and each variable can\n"
            "  be its own element — a reader that stops early loses the later ones and\n"
            "  still parses.")
        assert got[name].shape == want[name].shape, (
            f"{name}: shape {got[name].shape}, scipy says {want[name].shape}")
        assert got[name].dtype == want[name].dtype, (
            f"{name}: dtype {got[name].dtype}, scipy says {want[name].dtype}")
        assert np.array_equal(got[name], want[name]), (
            f"{name}: same shape and dtype, different values.\n"
            "  MATLAB writes column-major; reading the dimensions forward gives an\n"
            "  array of the right size with every element in the wrong place.")


def test_a_file_that_is_not_one_says_so():
    """The first bytes are the format's own name, so a wrong file is cheap to refuse."""
    with pytest.raises(ValueError, match="MATLAB 5.0"):
        V._mat_read(b"\x00" * 200)


def test_a_truncated_file_does_not_fall_off_the_end():
    """**A download cut halfway lands here**, and so did a padding mistake in the
    reader while it was being written: the loop walked four bytes past the first
    variable and `struct` reported `unpack_from requires a buffer of at least 1564
    bytes` — a sentence about a buffer, naming no file and offering no move.

    Truncating after the header alone is the cheapest version of that, and it must end
    quietly rather than raise from inside `struct`.
    """
    buf = io.BytesIO()
    scipy_io.savemat(buf, {"a": np.arange(6, dtype=np.uint8).reshape(3, 2)},
                     do_compression=True)
    whole = buf.getvalue()
    assert V._mat_read(whole), "the whole file must read, or this proves nothing"
    # Cut inside the single element, leaving its tag intact and its payload short.
    assert V._mat_read(whole[:140]) == {}, (
        "a file cut inside an element should come back empty rather than raising "
        "from inside `struct`")


@pytest.fixture
def written(tmp_path):
    """A `.mat` in SVHN's own shape, written where the dataset expects it."""
    rng = np.random.default_rng(7)
    payload = {
        "X": rng.integers(0, 256, (32, 32, 3, 6)).astype(np.uint8),
        "y": np.array([[1], [10], [3], [10], [7], [2]], dtype=np.uint8),
    }
    scipy_io.savemat(str(tmp_path / "train_32x32.mat"), payload, do_compression=True)
    return tmp_path


class _Loose(T.SVHN):
    """torchvision's, with the digest check stood down — the file is written here and
    has no published md5. Nothing else about it is changed."""

    def _check_integrity(self):
        return True


def test_svhn_matches_torchvision_value_for_value(written):
    """Data, labels, length and one item, against torchvision on the same bytes."""
    ours = V.datasets.SVHN(str(written), split="train")
    theirs = _Loose(str(written), split="train")

    assert len(ours) == len(theirs)
    assert ours.data.shape == theirs.data.shape, (
        f"{ours.data.shape} against torchvision's {theirs.data.shape} — the array is "
        "(32, 32, 3, N)\n  on disk and (N, 3, 32, 32) after; a wrong transpose is "
        "still a stack of\n  pictures of the right size.")
    assert ours.data.dtype == theirs.data.dtype
    assert np.array_equal(ours.data, theirs.data)
    assert np.array_equal(np.asarray(ours.labels), np.asarray(theirs.labels))


def test_the_digit_zero_is_labelled_ten_on_disk(written):
    """**The remap has to happen and the fixture has to contain a 10.**

    Without it this test passes on a dataset that never does the conversion, which is
    the shape a case is written to avoid: the default answer would be right.
    """
    ours = V.datasets.SVHN(str(written), split="train")
    raw = scipy_io.loadmat(str(written / "train_32x32.mat"))["y"].reshape(-1)
    assert 10 in raw, "the fixture must carry a 10, or this checks nothing"
    assert 10 not in np.asarray(ours.labels), (
        "a label of 10 survived. SVHN calls the digit zero 10, and a ten-class loss\n"
        "  reads one past the end of its own weights rather than refusing.")
    assert np.asarray(ours.labels)[raw == 10].tolist() == [0, 0]


def test_one_item_matches_including_the_picture(written):
    """`__getitem__` hands back an array where torchvision hands back a PIL image —
    the one divergence every dataset here has, and `ToTensor` is where the two meet.
    The **pixels** must agree."""
    ours = V.datasets.SVHN(str(written), split="train")
    theirs = _Loose(str(written), split="train")
    picture, target = ours[1]
    their_picture, their_target = theirs[1]
    assert target == their_target == 0
    assert np.array_equal(picture, np.asarray(their_picture))


def test_an_unknown_split_is_refused(written):
    """torchvision names the three it takes; so does this."""
    with pytest.raises(ValueError, match="split"):
        V.datasets.SVHN(str(written), split="validation")


def test_a_missing_file_says_what_to_do(written):
    """The message a caller meets first when `download` was left off."""
    with pytest.raises(RuntimeError, match="download=True"):
        V.datasets.SVHN(str(written / "nothing"), split="test")


@pytest.mark.parametrize("compress", [True, False])
def test_the_mat_reader_reads_a_struct_array(compress):
    """**A struct array is a list of dicts**, and the values run element by element —
    all fields of one before the next.

    Reading them field by field instead transposes the whole array: with `n` elements
    and `f` fields every value lands on the wrong element, and at `n == 1` — which is
    most files anyone tests with — the two orders agree. So this has two elements and
    two fields of different kinds, and each element's number and name have to arrive
    together.
    """
    annotations = np.array([(np.uint16(7), "00001.jpg"), (np.uint16(2), "00002.jpg")],
                           dtype=[("class", "O"), ("fname", "O")])
    buf = io.BytesIO()
    scipy_io.savemat(buf, {"annotations": annotations}, do_compression=compress)

    got = V._mat_read(buf.getvalue())["annotations"]
    assert isinstance(got, list) and len(got) == 2
    assert [str(one["fname"]) for one in got] == ["00001.jpg", "00002.jpg"]
    assert [int(np.asarray(one["class"]).reshape(-1)[0]) for one in got] == [7, 2]


def test_a_struct_holding_one_element_is_a_dict():
    """One element is a dict rather than a list of one, which is what `scipy` does with
    `squeeze_me` and what a caller writes against."""
    buf = io.BytesIO()
    scipy_io.savemat(buf, {"s": np.array([(np.int32(5), "only")],
                                         dtype=[("n", "O"), ("t", "O")])})
    got = V._mat_read(buf.getvalue())["s"]
    assert isinstance(got, dict)
    assert str(got["t"]) == "only" and int(np.asarray(got["n"]).reshape(-1)[0]) == 5


@pytest.mark.parametrize("compress", [True, False])
def test_the_mat_reader_reads_a_cell_of_text(compress):
    """**MATLAB stores text as code units, not bytes.** Read as bytes, a sixteen-bit
    string comes back with a NUL between every letter — printable, wrong, and the same
    length as the answer once something strips them. The names here are long enough
    that a fixed-width field block cannot be split on NUL instead.
    """
    names = np.array(["AM General Hummer SUV 2000", "Acura RL Sedan 2012"],
                     dtype=object)
    buf = io.BytesIO()
    scipy_io.savemat(buf, {"class_names": names}, do_compression=compress)

    got = V._mat_read(buf.getvalue())["class_names"]
    assert got == ["AM General Hummer SUV 2000", "Acura RL Sedan 2012"]


@pytest.fixture
def cars_root(tmp_path):
    from PIL import Image
    base = tmp_path / "stanford_cars"
    devkit = base / "devkit"
    devkit.mkdir(parents=True)
    for folder in ("cars_train", "cars_test"):
        (base / folder).mkdir()
        for i in (1, 2):
            Image.fromarray(np.full((3, 4, 3), i * 40, np.uint8)).save(
                base / folder / f"{i:05d}.jpg", format="PNG")
    annotations = np.array([(np.uint16(7), "00001.jpg"), (np.uint16(2), "00002.jpg")],
                           dtype=[("class", "O"), ("fname", "O")])
    scipy_io.savemat(str(devkit / "cars_train_annos.mat"),
                     {"annotations": annotations})
    scipy_io.savemat(str(base / "cars_test_annos_withlabels.mat"),
                     {"annotations": annotations})
    scipy_io.savemat(str(devkit / "cars_meta.mat"),
                     {"class_names": np.array(["AM General Hummer", "Acura RL"] * 4,
                                              dtype=object)})
    return tmp_path


@pytest.mark.parametrize("split", ["train", "test"])
def test_stanford_cars_matches_torchvision(split, cars_root):
    """**The class in the file starts at 1 and the label starts at 0.** Leaving the
    subtraction gives labels running 1 to 196 against 196 names — every prediction
    shifted by one make, and the accuracy identical.

    The two splits also keep their annotations in different places: `train`'s inside
    `devkit`, `test`'s beside it in a file whose name says `withlabels`, because the
    devkit's own test file has none.
    """
    ours = V.datasets.StanfordCars(str(cars_root), split=split)
    theirs = T.StanfordCars(str(cars_root), split=split)
    assert ours.classes == theirs.classes
    assert ours.class_to_idx == theirs.class_to_idx
    assert len(ours) == len(theirs)
    assert [one[1] for one in ours._samples] == [one[1] for one in theirs._samples]
    assert [one[1] for one in ours._samples] == [6, 1], "the label starts at 0"
    for i in range(len(ours)):
        assert ours[i][1] == theirs[i][1]
        assert np.array_equal(np.asarray(ours[i][0]), np.asarray(theirs[i][0]))


def test_stanford_cars_download_says_the_url_is_gone(tmp_path):
    """Refused the way torchvision refuses it, rather than by this library's usual
    reason — the original URL is broken and no reader here changes that."""
    with pytest.raises(ValueError, match="original URL is broken"):
        V.datasets.StanfordCars(str(tmp_path), download=True)


@pytest.mark.parametrize("kind,payload,want", [
    (16, b"Acura RL", "Acura RL"),                                    # miUTF8
    (4, "Acura RL".encode("utf-16-le"), "Acura RL"),                  # miUINT16
    (18, "Acura RL".encode("utf-32-le"), "Acura RL"),                 # miUTF32
])
def test_mat_text_reads_every_code_unit_width(kind, payload, want):
    """**Only one of these three is ever reachable through a fixture.**

    Measured: `scipy.io.savemat` writes `miUTF8`, so every `.mat` written in this
    repository takes that branch — and MATLAB itself writes `miUINT16`, the branch a
    fixture cannot produce. Testing the reader through files alone would leave it
    claimed and unchecked, which is how the comment above it was first written, so the
    three widths are given to the function directly.
    """
    assert V._mat_text(kind, payload) == want


def test_sixteen_bit_text_read_as_bytes_would_not_be_this():
    """The failure the branch above prevents, stated as a value rather than a warning:
    the same bytes taken one at a time are the answer with a NUL after every letter.
    """
    raw = "Acura".encode("utf-16-le")
    assert V._mat_text(4, raw) == "Acura"
    assert raw.decode("latin-1") == "A\x00c\x00u\x00r\x00a\x00"
