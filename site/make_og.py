"""Makes the image that appears when the link is pasted somewhere.

    uv run --with pillow python site/make_og.py

It makes one file, `site/assets/og.png` (1200×630). Social previews do not take SVG,
so it has to be a raster, and a hand-drawn image sitting in the repository goes quietly
stale the moment the wording changes — so **the wording lives in code and the image
comes out of here.**

## Why it is needed

This site says a single URL is the whole deployment. For that to be true, what appears
when the URL is pasted is part of the claim. Until now, nothing appeared.

## Why it is committed

The same reason as `api.json` — a static deployment has nowhere to run a generator.
Instead it is kept under 60KB, and anything that needs changing is changed here and
regenerated.
"""

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "site" / "assets" / "og.png"

W, H = 1200, 630
BG = (10, 12, 16)              # --bg on the dark side
FG = (232, 236, 243)           # --fg
DIM = (163, 174, 192)          # --fg-dim
EMBER = (255, 122, 61)         # --ember

TITLE = "Machine Learning,\nnative to the Web."
UNDER = "A PyTorch-like TypeScript runtime on WebGPU."
FOOT = "Learn · Tutorials · API · Playground — all of it runs in the page"


def font(size, mono=False, bold=False):
    """Picks from what this machine has. With none of them it falls back to the
    default font — ugly beats no image at all."""
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
        raise SystemExit("pillow is needed:\n  uv run --with pillow python site/make_og.py")

    image = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(image)

    # Orange bleeding in from the top left, where the site's hero has it.
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
        print("  warning: over 200KB. Some social crawlers will not fetch it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
