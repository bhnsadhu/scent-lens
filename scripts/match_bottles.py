"""Match the user's bottle list to the cleaned catalog.

draft: reads data/real/bottle_list.csv (brand, name, expected) and writes
  data/real/matches.csv (draft) and data/real/match_review.csv for the user to
  check, plus local copies of each candidate's catalog image under
  data/real/review_images/ so the user can open them (gitignored, never shared).
apply: reads the edited match_review.csv and rebuilds data/real/matches.csv.

Review columns the user edits:
  decision: ok (the suggested match is right), pick2 or pick3 (an alternative
    candidate is right), absent (not in the catalog), ambiguous (cannot tell).
  product_id_override: optional catalog product id, found in
    data/real/catalog_names.csv, when none of the candidates is right.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scent_lens.catalog import ZipImages  # noqa: E402
from scent_lens.matching import match_bottle, match_confidence  # noqa: E402

REAL = ROOT / "data" / "real"
GALLERY = ROOT / "artifacts" / "catalog" / "gallery.csv"
DECISIONS = {"ok", "pick2", "pick3", "absent", "ambiguous"}


def products() -> pd.DataFrame:
    gal = pd.read_csv(GALLERY, dtype=str, keep_default_na=False)
    return gal.drop_duplicates("product_id")


def draft() -> int:
    bl = pd.read_csv(REAL / "bottle_list.csv", dtype=str, keep_default_na=False)
    bl.columns = [c.strip().lower() for c in bl.columns]
    if not {"brand", "name"} <= set(bl.columns):
        raise SystemExit("bottle_list.csv needs columns brand,name (and optionally expected)")
    if "expected" not in bl.columns:
        bl["expected"] = "in"
    bl = bl[(bl.brand.str.strip() != "") & (bl.name.str.strip() != "")].reset_index(drop=True)
    bl["expected"] = bl["expected"].str.strip().str.lower().replace({"": "in"})
    bl.insert(0, "bottle_id", [f"b{i + 1:03d}" for i in range(len(bl))])

    prods = products()
    prods[["product_id", "brand", "name", "family"]].to_csv(REAL / "catalog_names.csv", index=False)
    first_image = prods.set_index("product_id")["image_name"]
    store = ZipImages(ROOT / "data" / "raw" / "images.zip")
    img_dir = REAL / "review_images"
    img_dir.mkdir(parents=True, exist_ok=True)

    rows, draft_rows = [], []
    for b in bl.itertuples():
        cands = match_bottle(
            b.brand, b.name, prods["brand"].tolist(), prods["name"].tolist(), prods["product_id"].tolist(), k=3
        )
        conf = match_confidence(cands)
        row = {
            "bottle_id": b.bottle_id,
            "your_brand": b.brand.strip(),
            "your_name": b.name.strip(),
            "you_expected": b.expected,
            "status": ("check, you expected this to be absent" if b.expected == "out" else conf),
        }
        for j, c in enumerate(cands, start=1):
            path = img_dir / f"{b.bottle_id}_cand{j}_{c.product_id}.jpg"
            if not path.exists():
                path.write_bytes(store.read_bytes(first_image[c.product_id]))
            row |= {
                f"cand{j}_brand": c.brand,
                f"cand{j}_name": c.name,
                f"cand{j}_score": round(c.score, 1),
                f"cand{j}_product_id": c.product_id,
                f"cand{j}_image": str(path),
            }
        row |= {"decision": "", "product_id_override": ""}
        rows.append(row)
        top = cands[0] if cands else None
        draft_rows.append(
            {
                "bottle_id": b.bottle_id, "brand": b.brand.strip(), "name": b.name.strip(), "expected": b.expected,
                "product_id": top.product_id if top else "", "score": round(top.score, 1) if top else "",
                "confidence": conf,
            }
        )
    review = pd.DataFrame(rows)
    review.to_csv(REAL / "match_review.csv", index=False)
    pd.DataFrame(draft_rows).to_csv(REAL / "matches.csv", index=False)
    n_high = (review["status"] == "high").sum()
    print(f"{len(review)} bottles: {n_high} high confidence, {len(review) - n_high} to check")
    return 0


def apply() -> int:
    review = pd.read_csv(REAL / "match_review.csv", dtype=str, keep_default_na=False)
    prods = products().set_index("product_id")
    out, problems = [], []
    for r in review.itertuples():
        d = r.decision.strip().lower()
        override = r.product_id_override.strip()
        if override:
            if override not in prods.index:
                problems.append(f"{r.bottle_id}: unknown product id {override}")
                continue
            pid, status = override, "matched"
        elif d == "ok":
            pid, status = r.cand1_product_id, "matched"
        elif d in {"pick2", "pick3"}:
            pid, status = getattr(r, f"cand{d[-1]}_product_id"), "matched"
        elif d == "absent":
            pid, status = "", "absent"
        elif d == "ambiguous":
            pid, status = "", "ambiguous"
        else:
            problems.append(f"{r.bottle_id}: decision {r.decision!r} is not one of {sorted(DECISIONS)}")
            continue
        row = {"bottle_id": r.bottle_id, "brand": r.your_brand, "name": r.your_name, "status": status, "product_id": pid}
        if pid:
            row |= {"catalog_brand": prods.loc[pid, "brand"], "catalog_name": prods.loc[pid, "name"]}
        out.append(row)
    if problems:
        print("fix these rows in match_review.csv, nothing was written:")
        print("\n".join(problems))
        return 2
    m = pd.DataFrame(out)
    dup = m[m.status == "matched"].duplicated("product_id", keep=False)
    if dup.any():
        print("note: several bottles map to one catalog product:", m[m.status == "matched"][dup].bottle_id.tolist())
    m.to_csv(REAL / "matches.csv", index=False)
    counts = m.status.value_counts().to_dict()
    print(counts)
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["draft", "apply"])
    raise SystemExit(draft() if ap.parse_args().mode == "draft" else apply())
