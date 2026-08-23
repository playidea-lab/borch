"""**The picture formats that are not a codec, and the two datasets behind them.**

Forty-five datasets were declined with the words *a codec*, as one wall. It is two.

- **JPEG** is a discrete cosine transform, Huffman tables, chroma subsampling and a
  progressive mode. That is a codec, and this library does not have one.
- **PNG** is `zlib` — already imported — plus a chunk walk and a per-row filter chosen
  from five that each subtract a neighbour.
- **PPM** is a magic number, three numbers, and the samples.

Measured before either was written: of the forty-five, **two open PNG and no JPEG**
and one opens PPM. The sentence was carrying fifteen rows that are genuinely blocked
and three that were not, and it stood for as long as nobody asked which files the
datasets actually read — the same shape `SVHN` was behind.

## What the fixtures are built to catch

Both archives are hundreds of megabytes behind hosts a test cannot reach, so the
layouts are written here. Each carries the trap its dataset has:

- **Omniglot's class is alphabet *and* character.** Folding by alphabet gives 50
  classes where torchvision gives 964, and every accuracy computed on it answers a
  different question.
- **Omniglot's order is the filesystem's**, not sorted — see the note in the library.
  Sorting is the better rule and a different one, so the fixture builds two alphabets
  and checks the labels agree with torchvision's rather than with alphabetical order.
- **GTSRB's train label is the folder's position, not the number in its name.** On the
  real dataset those are the same number: forty-three folders, `00000` to `00042`,
  none missing. **A complete input cannot tell the two rules apart**, so the fixture
  uses `00000` and `00007` — and that is the only reason the difference was found.
- **GTSRB's test labels come from a CSV** whose rows are not in the order the files
  sort in, so the fixture writes them out of order.

The PNG reader is checked separately against PIL, which is the authority it exists so
as not to depend on: every colour type, both sub-byte depths, three compression levels.
The pictures have gradients in them rather than flat fields — **a flat image
reconstructs correctly under all five row filters**, so a fixture without structure
tests one filter five times.
"""

import csv
import io
import os
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

Image = pytest.importorskip("PIL.Image")
pytest.importorskip("torchvision")

import torchvision.datasets as T                                 # noqa: E402

import borchvision as V                                          # noqa: E402


def _gradient(h, w, c=None):
    """Structure in both directions, so every row filter has something to undo."""
    y, x = np.mgrid[0:h, 0:w]
    base = ((y * 7 + x * 13) % 256).astype(np.uint8)
    if c is None:
        return base
    wide = base.astype(np.int32)
    return np.dstack([base, (wide * 2 % 256).astype(np.uint8),
                      (255 - wide).astype(np.uint8)][:c])


PNG_CASES = {
    "L 8-bit": (Image.fromarray(_gradient(23, 31)), "L"),
    "L 8-bit noisy": (Image.fromarray(
        np.random.default_rng(3).integers(0, 256, (17, 19)).astype(np.uint8)), "L"),
    "RGB 8-bit": (Image.fromarray(_gradient(21, 29, 3)), "RGB"),
    "RGBA 8-bit": (Image.fromarray(
        np.dstack([_gradient(15, 13, 3), np.full((15, 13), 128, np.uint8)])), "RGBA"),
    "1-bit": (Image.fromarray(_gradient(25, 27) > 127).convert("1"), "1"),
    "palette": (Image.fromarray(_gradient(19, 23)).convert("P"), "P"),
}


@pytest.mark.parametrize("label", sorted(PNG_CASES))
@pytest.mark.parametrize("level", [0, 6, 9])
def test_the_png_reader_agrees_with_pil(label, level):
    """**Three compression levels**, because the level decides which row filters the
    encoder picks and a reader can undo four of five and pass at one setting."""
    img, mode = PNG_CASES[label]
    buf = io.BytesIO()
    img.save(buf, format="PNG", compress_level=level)
    raw = buf.getvalue()

    want = np.asarray(Image.open(io.BytesIO(raw)).convert(
        "L" if mode in ("L", "1") else ("RGBA" if mode == "RGBA" else "RGB")))
    got = V._png_read(raw)
    if got.ndim == 3 and got.shape[2] == 3 and want.ndim == 2:
        got = got[:, :, 0]
    assert got.shape == want.shape, f"{got.shape} against PIL's {want.shape}"
    assert np.array_equal(got, want), (
        "same shape, different pixels — a row filter was undone wrongly, and every "
        "filter\n  refers to the row above, so one mistake carries downward through "
        "the picture.")


@pytest.mark.parametrize("mode,magic", [("L", "P5"), ("RGB", "P6")])
def test_the_ppm_reader_agrees_with_pil(mode, magic):
    img = Image.fromarray(_gradient(11, 17, 3) if mode == "RGB" else _gradient(11, 17))
    buf = io.BytesIO()
    img.save(buf, format="PPM")
    raw = buf.getvalue()
    assert raw[:2].decode() == magic
    got, want = V._ppm_read(raw), np.asarray(Image.open(io.BytesIO(raw)))
    assert got.shape == want.shape
    assert np.array_equal(got, want)


def test_a_file_that_is_not_a_png_says_so():
    with pytest.raises(ValueError, match="signature"):
        V._png_read(b"\x00" * 64)


def test_an_interlaced_png_is_refused_by_name():
    """**Adam7 reorders the whole image into seven passes.** A reader that ignored the
    flag would return a picture built from the first pass — recognisable, wrong, and
    silent, which is the one outcome worth spending an exception on.

    **PIL will not write one**, measured: `interlace=True` and `interlace=1` both
    produce a file whose IHDR flag is 0. So the flag is set by hand on a normal file.
    That leaves the IHDR's CRC stale — this reader does not check CRCs, and the flag is
    the only field under test, so the file is invalid in a way the branch never reaches.
    Said out loud because a fixture that is not what it claims to be is worse than none.
    """
    buf = io.BytesIO()
    Image.fromarray(_gradient(16, 16)).save(buf, format="PNG")
    raw = bytearray(buf.getvalue())
    assert raw[28] == 0, "the fixture must start from a non-interlaced file"
    raw[28] = 1                                  # IHDR's last byte is the flag
    with pytest.raises(NotImplementedError, match="interlac"):
        V._png_read(bytes(raw))


class _LooseOmniglot(T.Omniglot):
    """torchvision's, with the zip digest stood down — the folders are written here."""

    def _check_integrity(self):
        return True


@pytest.fixture
def omniglot_root(tmp_path):
    """Two alphabets, two characters each, three drawings each — enough that the class
    is visibly *alphabet and character* and that the ordering rule matters."""
    rng = np.random.default_rng(11)
    base = tmp_path / "omniglot-py" / "images_background"
    for alphabet in ("Latin", "Greek"):
        for character in ("character01", "character02"):
            here = base / alphabet / character
            here.mkdir(parents=True)
            for k in range(3):
                art = rng.random((105, 105)) > 0.85
                Image.fromarray(art).convert("1").save(here / f"{k}_{alphabet}.png")
    return tmp_path


def test_omniglot_matches_torchvision_item_for_item(omniglot_root):
    ours = V.datasets.Omniglot(str(omniglot_root), background=True)
    theirs = _LooseOmniglot(str(omniglot_root), background=True)
    assert len(ours) == len(theirs) == 12
    for i in range(len(ours)):
        picture, label = ours[i]
        their_picture, their_label = theirs[i]
        assert label == their_label, (
            f"item {i}: class {label} here and {their_label} in torchvision.\n"
            "  The order is the filesystem's on both sides — sorting is a better rule\n"
            "  and a different one, and every label moves under it.")
        assert np.array_equal(np.asarray(picture), np.asarray(their_picture))


def test_omniglots_class_is_alphabet_and_character(omniglot_root):
    """Four classes from two alphabets. Folding by alphabet would give two, and an
    accuracy over two classes is not a worse measurement of the same thing."""
    ours = V.datasets.Omniglot(str(omniglot_root), background=True)
    assert len(ours._characters) == 4
    assert all(os.sep in c for c in ours._characters)


@pytest.fixture
def gtsrb_root(tmp_path):
    """`00000` and `00007` — **not contiguous, on purpose.** The real dataset's forty
    three folders run `00000` to `00042`, so the folder's name and its position are
    the same number and no complete input can tell the two rules apart."""
    rng = np.random.default_rng(5)
    base = tmp_path / "gtsrb" / "GTSRB"
    for cls in ("00000", "00007"):
        here = base / "Training" / cls
        here.mkdir(parents=True)
        for k in range(2):
            arr = rng.integers(0, 256, (12, 15, 3)).astype(np.uint8)
            Image.fromarray(arr).save(here / f"{cls}_{k}.ppm")
    test = base / "Final_Test" / "Images"
    test.mkdir(parents=True)
    rows = []
    for k, cls in enumerate((3, 11, 3)):
        name = f"{k:05d}.ppm"
        Image.fromarray(rng.integers(0, 256, (10, 13, 3)).astype(np.uint8)).save(
            test / name)
        rows.append({"Filename": name, "ClassId": cls})
    rows = [rows[2], rows[0], rows[1]]           # out of the order the files sort in
    with open(tmp_path / "gtsrb" / "GT-final_test.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Filename", "ClassId"],
                                delimiter=";")
        writer.writeheader()
        writer.writerows(rows)
    return tmp_path


@pytest.mark.parametrize("split", ["train", "test"])
def test_gtsrb_matches_torchvision_item_for_item(gtsrb_root, split):
    ours = V.datasets.GTSRB(str(gtsrb_root), split=split)
    theirs = T.GTSRB(str(gtsrb_root), split=split)
    assert len(ours) == len(theirs)
    for i in range(len(ours)):
        picture, label = ours[i]
        their_picture, their_label = theirs[i]
        assert label == their_label
        assert np.array_equal(np.asarray(picture), np.asarray(their_picture))


def test_gtsrbs_train_label_is_the_folders_position(gtsrb_root):
    """The finding this fixture exists for: `00007` is class **1**, not 7."""
    ours = V.datasets.GTSRB(str(gtsrb_root), split="train")
    assert sorted({label for _, label in ours._samples}) == [0, 1], (
        "the label came from the folder's name rather than its position. On the real "
        "dataset\n  those are the same number, so only a subset can tell them apart.")


def test_gtsrbs_test_labels_come_from_the_csv(gtsrb_root):
    """Written out of file order, so a reader that walks the folder gets them wrong."""
    ours = V.datasets.GTSRB(str(gtsrb_root), split="test")
    assert [label for _, label in ours._samples] == [3, 3, 11]


def test_an_unknown_gtsrb_split_is_refused(gtsrb_root):
    with pytest.raises(ValueError, match="split"):
        V.datasets.GTSRB(str(gtsrb_root), split="validation")
