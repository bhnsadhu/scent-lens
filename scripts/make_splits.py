"""Write the bottle level split manifest: development pool (5 folds) and test split.

Refuses to run if results/split_manifest.csv exists, so the split is fixed
once and committed before any evaluation on real photos.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scent_lens.splits import Bottle, assign_folds, check_manifest, split_bottles  # noqa: E402

SEED = 20261015
MANIFEST = ROOT / "results" / "split_manifest.csv"


def main() -> int:
    if MANIFEST.exists():
        print(f"{MANIFEST.name} exists; the split is fixed and will not be redrawn")
        return 1
    photos = pd.read_csv(ROOT / "data" / "real" / "photos.csv", dtype=str)
    bottles_df = photos.drop_duplicates("bottle_id")[["bottle_id", "status", "product_id"]]
    matches = pd.read_csv(ROOT / "data" / "real" / "matches.csv", dtype=str, keep_default_na=False).set_index("bottle_id")
    bottles = [
        Bottle(r.bottle_id, matches.loc[r.bottle_id, "brand"], r.status == "matched")
        for r in bottles_df.itertuples()
    ]
    pool = split_bottles(bottles, seed=SEED, test_fraction=0.5)
    folds = assign_folds([b for b in bottles if pool[b.bottle_id] == "dev"], seed=SEED, n_folds=5)
    check_manifest(pool, folds)
    m = bottles_df.copy()
    m["pool"] = m.bottle_id.map(pool)
    m["fold"] = m.bottle_id.map(folds).astype("Int64")
    m = m.sort_values("bottle_id")
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    m.to_csv(MANIFEST, index=False)
    summary = {
        "seed": SEED,
        "dev_bottles": int((m.pool == "dev").sum()),
        "test_bottles": int((m.pool == "test").sum()),
        "dev_matched": int(((m.pool == "dev") & (m.status == "matched")).sum()),
        "test_matched": int(((m.pool == "test") & (m.status == "matched")).sum()),
        "dev_absent": int(((m.pool == "dev") & (m.status == "absent")).sum()),
        "test_absent": int(((m.pool == "test") & (m.status == "absent")).sum()),
        "fold_sizes": m[m.pool == "dev"].fold.value_counts().sort_index().astype(int).to_dict(),
    }
    summary["fold_sizes"] = {str(k): v for k, v in summary["fold_sizes"].items()}
    (ROOT / "results" / "split_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
