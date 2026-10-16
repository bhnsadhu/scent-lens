"""Real photo ingest: parse names, decode (HEIC too), strip metadata, resize."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps

CONDITIONS = ("clean", "angled", "glare", "dim", "cluttered", "partial")
EXTENSIONS = {".heic", ".heif", ".jpg", ".jpeg", ".png"}
NAME_RE = re.compile(r"^(?P<bottle>b\d{3})__(?P<cond>[a-z]+)__(?P<n>\d+)$")
MAX_SIDE = 1024


@dataclass(frozen=True)
class PhotoName:
    bottle_id: str
    condition: str
    n: int

    @property
    def stem(self) -> str:
        return f"{self.bottle_id}__{self.condition}__{self.n}"


def parse_name(path: str | Path) -> PhotoName | None:
    p = Path(path)
    if p.suffix.lower() not in EXTENSIONS:
        return None
    m = NAME_RE.match(p.stem.lower())
    if not m or m["cond"] not in CONDITIONS:
        return None
    return PhotoName(m["bottle"], m["cond"], int(m["n"]))


def register_heif() -> None:
    try:
        import pillow_heif

        pillow_heif.register_heif_opener()
    except ImportError:  # CI has no pillow_heif; HEIC files then fail to open
        pass


def clean_image(src: str | Path, max_side: int = MAX_SIDE) -> Image.Image:
    """Decode, apply the EXIF orientation, then rebuild from pixels only.

    The returned image carries no EXIF, GPS, XMP, ICC, or other metadata: it is
    a fresh image made from the pixel data alone.
    """
    with Image.open(src) as img:
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGB")
        img.thumbnail((max_side, max_side), Image.LANCZOS)
        clean = Image.frombytes("RGB", img.size, img.tobytes())
    return clean


def save_clean(img: Image.Image, dst: str | Path) -> None:
    img.save(dst, format="JPEG", quality=92, exif=b"", optimize=True)
