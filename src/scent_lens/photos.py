"""Real photo ingest: parse names, decode (HEIC too), strip metadata, resize."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps

CONDITIONS = ("clean", "angled", "glare", "dim", "cluttered", "partial")
ONLINE = "online"  # one audited online product image per bottle; not a phone condition
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
        if img.mode in ("RGBA", "LA", "P"):  # transparent product shots go on white, not black
            rgba = img.convert("RGBA")
            img = Image.new("RGBA", rgba.size, "white")
            img.alpha_composite(rgba)
        img = img.convert("RGB")
        img.thumbnail((max_side, max_side), Image.LANCZOS)
        clean = Image.frombytes("RGB", img.size, img.tobytes())
    return clean


def save_clean(img: Image.Image, dst: str | Path) -> None:
    img.save(dst, format="JPEG", quality=92, exif=b"", optimize=True)


def conditions_for_folder(n_files: int) -> list[tuple[str, int]] | None:
    """Condition and shot number for a bottle folder shot in protocol order.

    Six photos are one round (clean, angled, glare, dim, cluttered, partial);
    twelve are two rounds. Any other count is ambiguous and returns None.
    """
    if n_files not in (len(CONDITIONS), 2 * len(CONDITIONS)):
        return None
    return [(CONDITIONS[i % len(CONDITIONS)], i // len(CONDITIONS) + 1) for i in range(n_files)]


def capture_time(path: str | Path) -> str:
    """EXIF DateTimeOriginal (or DateTime), else the file modification time.

    Read before metadata is stripped; used only to order a bottle's photos.
    """
    try:
        with Image.open(path) as img:
            exif = img.getexif()
            stamp = exif.get_ifd(0x8769).get(0x9003) or exif.get(0x0132)
            if stamp:
                return str(stamp)
    except Exception:
        pass
    from datetime import datetime

    return datetime.fromtimestamp(Path(path).stat().st_mtime).strftime("%Y:%m:%d %H:%M:%S.%f")
