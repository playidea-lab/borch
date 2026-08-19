"""링크를 붙였을 때 뜨는 그림을 만든다.

    uv run --with pillow python site/make_og.py

`site/assets/og.png`(1200×630) 하나를 만든다. 소셜 미리보기는 SVG 를 안 받으므로
그림이어야 하고, 손으로 그린 그림을 저장소에 두면 문구가 바뀔 때 조용히 낡는다 —
그래서 **문구를 코드에 두고 그림은 여기서 나온다.**

## 왜 필요한가

이 사이트는 "URL 하나가 곧 배포다" 라고 말한다. 그 말이 사실이려면 URL 을 붙였을 때
무엇이 뜨는지가 그 주장의 일부다. 지금까지 아무것도 안 떴다.

## 왜 커밋하는가

`api.json` 과 같은 이유다 — 정적 배포에는 생성기를 돌릴 자리가 없다. 대신 60KB
아래로 유지하고, 바꿀 일이 있으면 이 파일을 고쳐 다시 만든다.
"""

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "site" / "assets" / "og.png"

W, H = 1200, 630
BG = (10, 12, 16)              # 어두운 쪽 --bg
FG = (232, 236, 243)           # --fg
DIM = (163, 174, 192)          # --fg-dim
EMBER = (255, 122, 61)         # --ember

TITLE = "Machine Learning,\nnative to the Web."
UNDER = "A PyTorch-like TypeScript runtime on WebGPU."
FOOT = "Learn · Tutorials · API · Playground — all of it runs in the page"


def font(size, mono=False, bold=False):
    """이 기계에 있는 것 중에서 고른다. 없으면 기본 글꼴로 떨어진다 —
    그림이 안 나오는 것보다 못생긴 편이 낫다."""
    from PIL import ImageFont
    names = (["Menlo.ttc", "DejaVuSansMono.ttf"] if mono else
             ["HelveticaNeue.ttc", "Helvetica.ttc", "DejaVuSans.ttf"])
    for base in ("/System/Library/Fonts", "/System/Library/Fonts/Supplemental",
                 "/usr/share/fonts/truetype/dejavu"):
        for name in names:
            path = pathlib.Path(base) / name
            if path.exists():
                try:
                    return ImageFont.truetype(str(path), size, index=1 if bold and path.suffix == ".ttc" else 0)
                except OSError:
                    continue
    return ImageFont.load_default(size)


def main():
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        raise SystemExit("pillow 가 필요하다:\n  uv run --with pillow python site/make_og.py")

    image = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(image)

    # 왼쪽 위에서 번지는 주황. 사이트의 히어로와 같은 자리다.
    for i in range(240, 0, -6):
        alpha = (240 - i) / 240 * 0.22
        draw.ellipse((-360 - i, -300 - i, 620 + i, 420 + i),
                     fill=tuple(int(b + (e - b) * alpha) for b, e in zip(BG, EMBER)))

    draw.ellipse((80, 78, 104, 102), fill=EMBER)
    draw.text((120, 74), "borch", font=font(34, mono=True), fill=FG)

    draw.multiline_text((80, 190), TITLE, font=font(76, bold=True), fill=FG, spacing=16)
    draw.text((80, 400), UNDER, font=font(32), fill=DIM)
    draw.text((80, H - 96), FOOT, font=font(24, mono=True), fill=EMBER)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    image.save(OUT, "PNG", optimize=True)
    size = OUT.stat().st_size / 1024
    print(f"{OUT.relative_to(ROOT)} — {W}×{H}, {size:.0f}KB")
    if size > 200:
        print("  경고: 200KB 를 넘는다. 소셜 쪽에서 안 받아 갈 수 있다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
