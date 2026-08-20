"""Makes the subset of data the tutorials use.

    python3 site/fetch_data.py              # from the CIFAR binaries at the repository root
    python3 site/fetch_data.py --download   # or fetch the official distribution first

It leaves one sprite and one label file in `site/assets/data/`. **That folder is in the
repository now** — this said "it is gitignored, the same place as `vendor/pyodide`",
that place moved, and measuring showed what it was worth: of one deploy's 34m20s,
**33m56s** was this script. Fetching 170MB of originals and processing sixty thousand
images to produce 1.1MB, repeated on every deploy.

So this script is **not normally run.** It is run to change the subset size or to pick
the quality again, and what comes out is committed. There is no randomness, so the same
input gives the same bytes (confirmed by regenerating).

The original batches (`cifar-batch*.bin`, 29MB) stay gitignored.

## Why a subset rather than all of it

One CIFAR-10 batch is 29MB. That is not a thing to make someone download to open one
tutorial page. Two thousand images is enough for a small CNN to visibly learn, and more
than that does not change the question this page answers — what is measured here is not
the absolute accuracy but whether training happens.

## Why JPEG — chosen by measuring

The same two thousand images are 5.9MB raw, 4.1MB as PNG, 0.84MB as JPEG (q88). Seven
times smaller decided it, and **the pixels are not exactly the originals.** The
tutorials say so — accuracy from here is not to be compared against a paper's numbers.
Anywhere the values have to be exact, such as the golden comparison, does not use this
file.
"""

import argparse
import io
import json
import pathlib
import sys
import tarfile
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "site" / "assets" / "data"
RECORD = 3073                      # one label byte + 32×32×3
COLS = 50                          # images per row of the sprite
QUALITY = 88

SOURCE = "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"
CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]


def local_batches():
    """What is already at the repository root — the files the benchmark uses."""
    train = ROOT / "cifar-batch1.bin"
    test = ROOT / "cifar-batch-test.bin"
    if train.exists() and test.exists():
        return train.read_bytes(), test.read_bytes()
    return None, None


def downloaded():
    """Pulls from the official distribution. The route CI takes."""
    print(f"fetching: {SOURCE}")
    with urllib.request.urlopen(SOURCE, timeout=300) as res:
        blob = res.read()
    print(f"  {len(blob) / 1048576:.0f}MB fetched. unpacking…")
    train = test = None
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
        for member in tar.getmembers():
            if member.name.endswith("data_batch_1.bin"):
                train = tar.extractfile(member).read()
            elif member.name.endswith("test_batch.bin"):
                test = tar.extractfile(member).read()
    if train is None or test is None:
        raise SystemExit("could not find data_batch_1.bin and test_batch.bin in the distribution.")
    return train, test


def write_split(raw, count, name, Image, np):
    """Records into one sprite plus one label file."""
    have = len(raw) // RECORD
    if have < count:
        raise SystemExit(f"{name}: needs {count} images and there are only {have}.")
    rows = (count + COLS - 1) // COLS
    sheet = np.zeros((rows * 32, COLS * 32, 3), dtype=np.uint8)
    labels = []
    for i in range(count):
        rec = raw[i * RECORD:(i + 1) * RECORD]
        labels.append(int(rec[0]))
        px = np.frombuffer(rec[1:], dtype=np.uint8).reshape(3, 32, 32).transpose(1, 2, 0)
        r, c = divmod(i, COLS)
        sheet[r * 32:(r + 1) * 32, c * 32:(c + 1) * 32] = px

    OUT.mkdir(parents=True, exist_ok=True)
    image = OUT / f"cifar-{name}.jpg"
    Image.fromarray(sheet).save(image, "JPEG", quality=QUALITY)
    meta = {
        "count": count, "cols": COLS, "tile": 32, "classes": CLASSES,
        "labels": labels,
        # **Written in both languages.** The tutorials print this sentence when they run
        # (`log(train.note)` in `04-image-classifier.html`), so in one language alone the
        # person who opened the English page receives a sentence they cannot read. That
        # is what happened.
        "note": {
            "en": "a JPEG-compressed subset of CIFAR-10 — the pixels are not the originals.",
            "ko": "JPEG 로 압축한 CIFAR-10 부분집합 — 픽셀이 원본과 같지 않다.",
        },
        "source": SOURCE,
    }
    (OUT / f"cifar-{name}.json").write_text(
        json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    size = (image.stat().st_size + (OUT / f"cifar-{name}.json").stat().st_size) / 1024
    print(f"  {image.relative_to(ROOT)} — {count} images, {size:.0f}KB")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true",
                        help="fetch the official distribution if the local files are missing")
    parser.add_argument("--train", type=int, default=2000)
    parser.add_argument("--test", type=int, default=500)
    args = parser.parse_args(argv)

    try:
        import numpy as np
        from PIL import Image
    except ImportError:
        raise SystemExit(
            "numpy and pillow are needed:\n"
            "  uv run --with numpy --with pillow python site/fetch_data.py")

    train, test = local_batches()
    if train is None:
        if not args.download:
            raise SystemExit(
                "no CIFAR binaries — put cifar-batch1.bin and cifar-batch-test.bin\n"
                "at the repository root, or fetch them with --download.")
        train, test = downloaded()

    print(f"writing into {OUT.relative_to(ROOT)}:")
    write_split(train, args.train, "train", Image, np)
    write_split(test, args.test, "test", Image, np)
    print("done. Tutorials 4 and 5 will run now.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
