"""Ingest the web set (version 2 fresh test): Wikimedia Commons and Flickr photos.

Reads data/real/web/manifest.csv (one verified photo per row: source page,
license, author, catalog product, and whether it is used), strips metadata
and resizes like phone photos, and drops any photo that is a near copy of
one of its own product's catalog images (phash Hamming distance at most 4,
the catalog duplicate rule). Writes data/real/web/photos.csv and the
attribution list results/web_sources.csv.
"""

from __future__ import annotations

import sys
from pathlib import Path

import imagehash
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scent_lens.assets import catalog  # noqa: E402
from scent_lens.catalog import ZipImages  # noqa: E402
from scent_lens.photos import clean_image, save_clean  # noqa: E402

WEB = ROOT / "data" / "real" / "web"
NEAR_DUP_BITS = 4


def main() -> int:
    m = pd.read_csv(WEB / "manifest.csv", dtype=str, keep_default_na=False)
    gal = catalog()
    store = ZipImages(ROOT / "data" / "raw" / "images.zip")
    out = WEB / "processed"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in m.itertuples():
        dst = out / f"{r.web_id}.jpg"
        img = clean_image(WEB / "raw" / r.file)
        save_clean(img, dst)
        h = imagehash.phash(img)
        refs = gal.loc[gal.product_id == r.product_id, "image_name"]
        dist = min((h - imagehash.phash(store.open(n)) for n in refs), default=64)
        use = r.use if dist > NEAR_DUP_BITS else f"excluded: near copy of a catalog image (distance {dist})"
        rows.append({"photo_id": r.web_id, "bottle_id": r.product_id, "condition": "web", "n": 1, "status": "matched",
                     "product_id": r.product_id, "path": str(dst.relative_to(ROOT)), "use": use, "phash_distance": dist})
    photos = pd.DataFrame(rows)
    photos.to_csv(WEB / "photos.csv", index=False)
    src = m[["web_id", "title", "page", "license", "artist", "brand", "name"]].merge(
        photos[["photo_id", "use", "phash_distance"]], left_on="web_id", right_on="photo_id").drop(columns="photo_id")
    src.to_csv(ROOT / "results" / "web_sources.csv", index=False)
    print(photos.use.value_counts().to_string())
    print("smallest phash distance among used photos:", int(photos[photos.use == "test"].phash_distance.min()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
