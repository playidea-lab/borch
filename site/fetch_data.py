"""튜토리얼이 쓰는 데이터 부분집합을 만든다.

    python3 site/fetch_data.py              # 저장소 루트의 CIFAR 바이너리에서
    python3 site/fetch_data.py --download   # 없으면 공식 배포본을 받아서

`site/assets/data/` 에 스프라이트 한 장과 라벨 파일을 남긴다. **그 폴더는 이제
저장소에 있다** — 여기 "`.gitignore` 다, `vendor/pyodide` 와 같은 자리다" 라고 적혀
있었는데 그쪽이 옮겨 갔고, 재 보니 값이 컸다: 배포 한 판 34분 20초 중 **33분 56초**가
이 스크립트였다. 원본 170MB 를 받아 6만 장을 처리해서 1.1MB 를 내는 일을 배포마다
되풀이한 것이다.

그러니 이 스크립트를 **평소에는 안 돌린다.** 돌릴 때는 부분집합 크기를 바꾸거나
품질을 다시 고를 때이고, 그때 나온 것을 커밋한다. 무작위가 없어 같은 입력이면 같은
바이트다(재생성해서 확인했다).

원본 배치(`cifar-batch*.bin`, 29MB)는 계속 `.gitignore` 다.

## 왜 전부가 아니라 부분집합인가

CIFAR-10 한 배치가 29MB 다. 튜토리얼 한 장을 열자고 그것을 받게 할 수는 없다.
2,000 장이면 작은 CNN 이 눈에 보이게 배우고, 그 이상은 이 페이지가 답하는 질문을
바꾸지 않는다 — 여기서 재는 것은 정확도의 절대값이 아니라 **학습이 되는가** 다.

## 왜 JPEG 인가 — 재서 골랐다

같은 2,000 장이 원본 5.9MB · PNG 4.1MB · JPEG(q88) 0.84MB 다. 7 배 차이라 이쪽을
골랐고, **픽셀이 원본과 완전히 같지는 않다.** 튜토리얼에는 그 사실을 적어 둔다 —
여기서 나온 정확도를 논문의 수와 비교하면 안 된다. 골든 대조처럼 값이 정확해야 하는
자리에는 이 파일을 쓰지 않는다.
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
RECORD = 3073                      # 라벨 1 바이트 + 32×32×3
COLS = 50                          # 스프라이트 한 줄에 몇 장
QUALITY = 88

SOURCE = "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"
CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]


def local_batches():
    """저장소 루트에 이미 있는 것. 벤치가 쓰는 그 파일들이다."""
    train = ROOT / "cifar-batch1.bin"
    test = ROOT / "cifar-batch-test.bin"
    if train.exists() and test.exists():
        return train.read_bytes(), test.read_bytes()
    return None, None


def downloaded():
    """공식 배포본에서 뽑는다. CI 가 쓰는 길이다."""
    print(f"받는 중: {SOURCE}")
    with urllib.request.urlopen(SOURCE, timeout=300) as res:
        blob = res.read()
    print(f"  {len(blob) / 1048576:.0f}MB 받았다. 푸는 중…")
    train = test = None
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
        for member in tar.getmembers():
            if member.name.endswith("data_batch_1.bin"):
                train = tar.extractfile(member).read()
            elif member.name.endswith("test_batch.bin"):
                test = tar.extractfile(member).read()
    if train is None or test is None:
        raise SystemExit("배포본 안에서 data_batch_1.bin·test_batch.bin 을 못 찾았다.")
    return train, test


def write_split(raw, count, name, Image, np):
    """레코드를 스프라이트 한 장 + 라벨 파일로."""
    have = len(raw) // RECORD
    if have < count:
        raise SystemExit(f"{name}: {count} 장이 필요한데 {have} 장뿐이다.")
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
        "note": "JPEG 로 압축한 CIFAR-10 부분집합 — 픽셀이 원본과 같지 않다.",
        "source": SOURCE,
    }
    (OUT / f"cifar-{name}.json").write_text(
        json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    size = (image.stat().st_size + (OUT / f"cifar-{name}.json").stat().st_size) / 1024
    print(f"  {image.relative_to(ROOT)} — {count}장, {size:.0f}KB")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true",
                        help="로컬 파일이 없으면 공식 배포본을 받는다")
    parser.add_argument("--train", type=int, default=2000)
    parser.add_argument("--test", type=int, default=500)
    args = parser.parse_args(argv)

    try:
        import numpy as np
        from PIL import Image
    except ImportError:
        raise SystemExit(
            "numpy 와 pillow 가 필요하다:\n"
            "  uv run --with numpy --with pillow python site/fetch_data.py")

    train, test = local_batches()
    if train is None:
        if not args.download:
            raise SystemExit(
                "CIFAR 바이너리가 없다 — 저장소 루트에 cifar-batch1.bin 과\n"
                "cifar-batch-test.bin 을 두거나, --download 로 받아라.")
        train, test = downloaded()

    print(f"{OUT.relative_to(ROOT)} 에 쓴다:")
    write_split(train, args.train, "train", Image, np)
    write_split(test, args.test, "test", Image, np)
    print("끝. 튜토리얼 4·5 가 이제 돈다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
