"""Ingest real photos: data/real/raw -> data/real/processed (metadata stripped, 1024 px).

Validates names against data/real/matches.csv, reports missing conditions per
bottle, and writes data/real/photos.csv (one row per processed photo) and
results/real_photo_counts.json (counts only). Resumable: processed files are
skipped.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scent_lens.photos import CONDITIONS, clean_image, parse_name, register_heif, save_clean  # noqa: E402

REAL = ROOT / "data" / "real"


def main() -> int:
    register_heif()
    raw, out = REAL / "raw", REAL / "processed"
    out.mkdir(parents=True, exist_ok=True)
    matches = pd.read_csv(REAL / "matches.csv", dtype=str, keep_default_na=False)
    known = matches.set_index("bottle_id")
    rows, bad, failed = [], [], []
    seen: Counter = Counter()
    for src in sorted(p for p in raw.rglob("*") if p.is_file() and not p.name.startswith(".")):
        name = parse_name(src)
        if name is None:
            bad.append(f"{src.name}: name does not follow bottle_id__condition__n")
            continue
        if name.bottle_id not in known.index:
            bad.append(f"{src.name}: bottle {name.bottle_id} is not in matches.csv")
            continue
        if known.loc[name.bottle_id, "status"] == "ambiguous":
            bad.append(f"{src.name}: bottle {name.bottle_id} is ambiguous and excluded")
            continue
        seen[name.stem] += 1
        if seen[name.stem] > 1:
            bad.append(f"{src.name}: duplicate of another file named {name.stem}")
            continue
        dst = out / f"{name.stem}.jpg"
        if not dst.exists():
            try:
                save_clean(clean_image(src), dst)
            except Exception as e:  # unreadable file
                failed.append(f"{src.name}: {type(e).__name__}: {e}")
                continue
        rows.append(
            {
                "photo_id": name.stem, "bottle_id": name.bottle_id, "condition": name.condition, "n": name.n,
                "status": known.loc[name.bottle_id, "status"], "product_id": known.loc[name.bottle_id, "product_id"],
                "path": str(dst.relative_to(ROOT)),
            }
        )
    photos = pd.DataFrame(rows)
    photos.to_csv(REAL / "photos.csv", index=False)

    missing = defaultdict(list)
    for b in matches[matches.status.isin(["matched", "absent"])].bottle_id:
        have = set(photos[photos.bottle_id == b].condition) if len(photos) else set()
        for c in CONDITIONS:
            if c not in have:
                missing[b].append(c)
    per_bottle = photos.groupby("bottle_id").size() if len(photos) else pd.Series(dtype=int)
    counts = {
        "photos": int(len(photos)),
        "bottles_with_photos": int(photos.bottle_id.nunique()) if len(photos) else 0,
        "matched_bottles_with_photos": int(photos[photos.status == "matched"].bottle_id.nunique()) if len(photos) else 0,
        "absent_bottles_with_photos": int(photos[photos.status == "absent"].bottle_id.nunique()) if len(photos) else 0,
        "photos_by_condition": photos.condition.value_counts().to_dict() if len(photos) else {},
        "photos_per_bottle": {"min": int(per_bottle.min()), "max": int(per_bottle.max())} if len(per_bottle) else {},
        "bottles_missing_some_condition": len(missing),
        "rejected_files": len(bad),
        "unreadable_files": len(failed),
    }
    (ROOT / "results" / "real_photo_counts.json").write_text(json.dumps(counts, indent=2) + "\n")
    print(json.dumps(counts, indent=2))
    for line in bad + failed:
        print("problem:", line)
    for b, conds in sorted(missing.items()):
        print(f"missing {b}: {', '.join(conds)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
