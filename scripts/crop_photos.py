"""Crop every processed real photo to its bottle (see scent_lens.crop).

Writes data/real/processed_crop/<photo_id>.jpg and artifacts/real/crops.csv
(box, detector score, whether the full image was kept). Uses no labels, so
it runs on all photos; scoring stays on the development pool.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from scent_lens.crop import BottleDetector, apply_crop  # noqa: E402
from scent_lens.photos import save_clean  # noqa: E402

OUT = ROOT / "data" / "real" / "processed_crop"


def main() -> int:
    photos = pd.read_csv(ROOT / "data" / "real" / "photos.csv", dtype=str)
    OUT.mkdir(parents=True, exist_ok=True)
    det = BottleDetector()
    rows = []
    for i, r in enumerate(photos.itertuples(), start=1):
        img = Image.open(ROOT / r.path).convert("RGB")
        box, score = det.detect(img)
        save_clean(apply_crop(img, box), OUT / f"{r.photo_id}.jpg")
        x0, y0, x1, y1 = box or (0, 0, *img.size)
        rows.append({"photo_id": r.photo_id, "x0": x0, "y0": y0, "x1": x1, "y1": y1, "score": round(score, 4),
                     "full": box is None, "area": round((x1 - x0) * (y1 - y0) / (img.size[0] * img.size[1]), 4)})
        if i % 25 == 0:
            print(f"{i}/{len(photos)}", flush=True)
    crops = pd.DataFrame(rows)
    crops.to_csv(ROOT / "artifacts" / "real" / "crops.csv", index=False)
    print(f"{len(crops)} photos, full image kept for {int(crops.full.sum())}, median crop area {crops.area.median():.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
