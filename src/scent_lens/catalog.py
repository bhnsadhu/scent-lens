"""Catalog loading: column mapping and image access straight from images.zip."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path, PurePosixPath

import pandas as pd
from PIL import Image

from scent_lens.normalize import line_key, product_key

# Raw csv column -> ScentLens field. Recorded in docs/data_notes.md.
COLUMN_MAP = {
    "brand": "brand",
    "name_perfume": "name",
    "family": "family",
    "subfamily": "subfamily",
    "fragrances": "accords",
    "ingredients": "notes",
    "origin": "origin",
    "gender": "gender",
    "years": "year",
    "description": "description",
    "image_name": "image_name",
    "image_id": "image_id",
}


def load_catalog(csv_path: str | Path) -> pd.DataFrame:
    """Read perfumes.csv, rename columns, and add row, product, and line keys."""
    raw = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    missing = [c for c in COLUMN_MAP if c not in raw.columns]
    if missing:
        raise KeyError(f"perfumes.csv lacks expected columns: {missing}; found {list(raw.columns)}")
    df = raw.rename(columns=COLUMN_MAP)[list(COLUMN_MAP.values())].copy()
    df.insert(0, "row_id", [f"r{i:05d}" for i in range(len(df))])
    df["product_key"] = [product_key(b, n) for b, n in zip(df["brand"], df["name"], strict=True)]
    df["line_key"] = [line_key(b, n) for b, n in zip(df["brand"], df["name"], strict=True)]
    return df


class ZipImages:
    """Read catalog images from the archive by file name, without extracting."""

    def __init__(self, zip_path: str | Path):
        self.zf = zipfile.ZipFile(zip_path)
        self.members: dict[str, str] = {}
        for info in self.zf.infolist():
            if info.is_dir():
                continue
            name = PurePosixPath(info.filename).name
            if name.startswith("._"):
                continue
            self.members.setdefault(name, info.filename)

    def __contains__(self, image_name: str) -> bool:
        return image_name in self.members

    def read_bytes(self, image_name: str) -> bytes:
        return self.zf.read(self.members[image_name])

    def open(self, image_name: str, max_side: int | None = 512) -> Image.Image:
        """Decode to RGB. JPEG draft mode decodes at reduced size to save memory."""
        img = Image.open(io.BytesIO(self.read_bytes(image_name)))
        if max_side and img.format == "JPEG":
            img.draft("RGB", (max_side, max_side))
        img = _to_rgb(img)
        if max_side:
            img.thumbnail((max_side, max_side))
        return img


def _to_rgb(img: Image.Image) -> Image.Image:
    """Flatten transparency onto white, since catalog shots sit on white."""
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        bg = Image.new("RGB", rgba.size, (255, 255, 255))
        bg.paste(rgba, mask=rgba.split()[-1])
        return bg
    return img.convert("RGB")
