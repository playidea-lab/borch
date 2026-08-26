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


def _write_16bit_png(path, arr):
    """PIL cannot **save** sixteen-bit colour, so the fixture is written by hand.

    That is not a detour: the files this exists for — KITTI's flow and disparity —
    were written by something that is not PIL either.
    """
    import struct
    import zlib
    height, width = arr.shape[:2]
    channels = 1 if arr.ndim == 2 else arr.shape[2]
    rows = arr.reshape(height, width * channels)
    raw = b"".join(b"\x00" + rows[y].astype(">u2").tobytes() for y in range(height))

    def chunk(kind, body):
        return (struct.pack(">I", len(body)) + kind + body
                + struct.pack(">I", zlib.crc32(kind + body)))

    header = struct.pack(">IIBBBBB", width, height, 16,
                         {1: 0, 3: 2, 4: 6}[channels], 0, 0, 0)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header)
                     + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


@pytest.mark.parametrize("channels", [1, 3, 4])
def test_sixteen_bit_png_follows_pil_by_default(channels, tmp_path):
    """**PIL keeps sixteen bits for grey and drops them for colour**, and the reader
    stands in PIL's place, so it does the same — including the part that looks like a
    bug in PIL. The grey half is what `Kitti2012Stereo` divides by 256.
    """
    shape = (5, 7) if channels == 1 else (5, 7, channels)
    arr = (np.arange(int(np.prod(shape))).reshape(shape) * 4021 % 65535).astype(np.uint16)
    path = tmp_path / "wide.png"
    _write_16bit_png(path, arr)

    want = np.asarray(Image.open(path))
    got = V._png_read(path.read_bytes())
    assert got.dtype == want.dtype, (
        f"{got.dtype} where PIL gives {want.dtype} — for grey that factor of 256 "
        "reaches the disparity maps and every number still looks like a distance")
    assert np.array_equal(got, want)


@pytest.mark.parametrize("channels", [1, 3, 4])
def test_keep_depth_gives_what_decode_png_gives(channels, tmp_path):
    """The other contract. KITTI's flow is sixteen-bit **colour** and is read as
    `(x - 2**15) / 64`; on PIL's eight-bit answer that arithmetic still runs and still
    returns a flow field of the right shape, pointing somewhere else.
    """
    from torchvision.io import decode_png, read_file
    shape = (5, 7) if channels == 1 else (5, 7, channels)
    arr = (np.arange(int(np.prod(shape))).reshape(shape) * 4021 % 65535).astype(np.uint16)
    path = tmp_path / "wide.png"
    _write_16bit_png(path, arr)

    want = decode_png(read_file(str(path))).numpy().transpose(1, 2, 0)
    if want.shape[2] == 1:
        want = want[:, :, 0]
    got = V._png_read(path.read_bytes(), keep_depth=True)
    assert got.dtype == np.uint16 and np.array_equal(got, want)
    assert np.array_equal(got, arr), "the samples are not the ones that were written"


@pytest.mark.parametrize("size", [(6, 8), (5, 7), (3, 13)])
@pytest.mark.parametrize("mode", ["L", "RGB"])
def test_the_bmp_reader_agrees_with_pil(mode, size, tmp_path):
    """**Widths that are not a multiple of four are the point.** Every BMP row is
    padded to four bytes, and a reader that ignored it shifts each row a little
    further than the last — a sheared picture that still looks like the original. 8
    and 13 are both wrong-length widths; only 8 is a multiple of four.
    """
    height, width = size
    rng = np.random.default_rng(4)
    shape = (height, width) if mode == "L" else (height, width, 3)
    img = Image.fromarray(rng.integers(0, 256, shape).astype(np.uint8), mode)
    path = tmp_path / "sheet.bmp"
    img.save(path, format="BMP")

    want = np.asarray(Image.open(path))
    got = V._bmp_read(path.read_bytes())
    assert got.shape == want.shape, (
        f"{got.shape} against PIL's {want.shape} — a grey palette is one channel")
    assert np.array_equal(got, want), (
        "same shape, different pixels: the rows run bottom to top, they are padded to "
        "four bytes, and the samples are BGR. Each of those alone returns a picture.")


def test_a_compressed_bmp_is_refused_by_name(tmp_path):
    """`BI_RLE8` exists and is not written here, so it is refused rather than read as
    if the bytes were raw — which would return noise shaped like a picture."""
    path = tmp_path / "plain.bmp"
    Image.fromarray(np.zeros((4, 4), np.uint8), "L").save(path, format="BMP")
    raw = bytearray(path.read_bytes())
    raw[30] = 1                                   # compression field: BI_RLE8
    with pytest.raises(ValueError, match="compressed"):
        V._bmp_read(bytes(raw))


@pytest.fixture
def kitti_root(tmp_path):
    base = tmp_path / "Kitti" / "raw" / "training"
    (base / "image_2").mkdir(parents=True)
    (base / "label_2").mkdir(parents=True)
    for i in (1, 2):
        Image.fromarray(np.full((3, 4, 3), i * 10, np.uint8)).save(
            base / "image_2" / f"{i:06d}.png")
        (base / "label_2" / f"{i:06d}.txt").write_text(
            f"Car 0.00 {i} -1.57 1.1 2.2 3.3 4.4 1.5 1.6 4.0 1.8 1.9 8.0 -1.2\n"
            f"Pedestrian 0.5 {i % 3} 0.1 5.5 6.6 7.7 8.8 1.7 0.6 0.9 2.8 1.5 9.0 0.4\n")
    return tmp_path


def test_kitti_targets_match_torchvisions_including_the_types(kitti_root):
    """The golden case carries the numbers; **the class name and `occluded`'s type are
    what it cannot hold.** `occluded` is a level, not a fraction, and reading it as a
    float agrees on every value and on nothing else until something groups by it.
    """
    ours = V.datasets.Kitti(str(kitti_root))
    theirs = T.Kitti(str(kitti_root))
    assert len(ours) == len(theirs)
    for i in range(len(ours)):
        assert ours[i][1] == theirs[i][1]
        for box in ours[i][1]:
            assert isinstance(box["occluded"], int) and not isinstance(
                box["occluded"], bool)
            assert isinstance(box["type"], str)


def test_kitti_without_the_tree_says_so(tmp_path):
    with pytest.raises(RuntimeError, match="Dataset not found"):
        V.datasets.Kitti(str(tmp_path))
