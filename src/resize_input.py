"""
One-off helper: resize/compress data/input.jpg to fit SerpAPI's image
upload limits (JPG/PNG/WebP, max 500KB) using high-quality Lanczos
resampling -- sharper than a quick `sips -Z` downscale.

Run with:  python -m src.resize_input
"""

from pathlib import Path
from PIL import Image

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
INPUT_IMAGE = DATA_DIR / "input.jpg"

MAX_DIM = 1280
MAX_BYTES = 480_000  # a little under SerpAPI's 500KB cap


def resize_input():
    img = Image.open(INPUT_IMAGE).convert("RGB")

    if max(img.size) > MAX_DIM:
        ratio = MAX_DIM / max(img.size)
        new_size = (int(img.width * ratio), int(img.height * ratio))
        img = img.resize(new_size, Image.LANCZOS)

    quality = 90
    while quality >= 40:
        img.save(INPUT_IMAGE, "JPEG", quality=quality)
        size = INPUT_IMAGE.stat().st_size
        if size <= MAX_BYTES:
            print(f"Saved {INPUT_IMAGE} at {img.size[0]}x{img.size[1]}, "
                  f"quality={quality}, size={size/1024:.1f} KB")
            return
        quality -= 10

    print(f"Warning: still {INPUT_IMAGE.stat().st_size/1024:.1f} KB "
          f"after max compression attempts.")


if __name__ == "__main__":
    resize_input()