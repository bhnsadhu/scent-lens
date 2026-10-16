"""Filter the catalog to clean product photos of single perfume bottles.

Inputs: perfumes.csv, the inventory records, and the CLIP gallery embeddings
(artifacts/emb/clip, all decodable images; reused later as the CLIP baseline).

Steps, each logged with counts in results/clean_counts.json:
1. metadata keyword filter (explicit terms, non bottle products)
2. zero shot CLIP check (keep single bottle product shots; flag people and
   explicit content). Flagged images are never displayed or saved.
3. duplicates: an image byte identical across different products is a generic
   brand image and is dropped; within a product, byte identical and near
   duplicate (phash) copies keep one.
4. products: rows grouped by product key into product ids.

Outputs: artifacts/catalog/gallery.csv (kept images with metadata and the row
index into the CLIP embedding matrix) and results/gallery_manifest.csv (image
ids and product ids only, committed).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scent_lens.catalog import load_catalog  # noqa: E402

RAW = ROOT / "data" / "raw"
EMB = ROOT / "artifacts" / "emb" / "clip"
OUT = ROOT / "artifacts" / "catalog"

EXPLICIT_TERMS = r"\b(porn\w*|xxx|erotic\w*|fetish\w*|nsfw|orgasm\w*|bdsm|sex toys?)\b"
NON_BOTTLE_TERMS = (
    r"\b(candles?|soaps?|shower gel|bath gel|body lotion|body cream|body butter|deodorant\w*|"
    r"coffret|gift set|discovery set|sample set|diffuser|incense)\b"
)

KEEP_PROMPTS = [
    "a product photo of a perfume bottle",
    "a photo of a fragrance bottle on a white background",
    "a photo of a perfume bottle next to its box",
]
PERSON_PROMPTS = ["a photo of a person", "a photo of a fashion model posing"]
EXPLICIT_PROMPTS = ["a nude or sexually explicit photo", "a photo of lingerie or underwear"]
OTHER_PROMPTS = [
    "an illustration or drawing",
    "a logo or brand name text",
    "a blank white image",
    "a photo of a candle",
    "a photo of a cosmetic cream jar or lotion tube",
    "a photo of flowers or fruit",
    "a photo of many different perfume bottles together",
]
KEEP_MIN = 0.5      # total probability on keep prompts
PERSON_MAX = 0.3    # probability on person prompts
EXPLICIT_MAX = 0.05  # probability on explicit prompts; deliberately strict
NEAR_DUP_BITS = 4


def zero_shot(emb: np.ndarray) -> pd.DataFrame:
    from scent_lens.embed import load_encoder

    enc = load_encoder("clip")
    prompts = KEEP_PROMPTS + PERSON_PROMPTS + EXPLICIT_PROMPTS + OTHER_PROMPTS
    text = enc.encode_texts(prompts)
    logits = 100.0 * emb @ text.T
    logits -= logits.max(axis=1, keepdims=True)
    p = np.exp(logits)
    p /= p.sum(axis=1, keepdims=True)
    k, pe, ex = len(KEEP_PROMPTS), len(PERSON_PROMPTS), len(EXPLICIT_PROMPTS)
    return pd.DataFrame(
        {
            "p_keep": p[:, :k].sum(1),
            "p_person": p[:, k : k + pe].sum(1),
            "p_explicit": p[:, k + pe : k + pe + ex].sum(1),
            "top_prompt": [prompts[i] for i in p.argmax(1)],
        }
    )


def main() -> int:
    df = load_catalog(RAW / "perfumes.csv")
    with (ROOT / "artifacts" / "inventory" / "images.jsonl").open() as f:
        inv = pd.DataFrame([json.loads(line) for line in f])
    names = (EMB / "gallery_names.txt").read_text().split()
    emb = np.load(EMB / "gallery.npy")
    counts: dict[str, int] = {"rows": len(df)}

    df = df[df["image_name"] != ""]
    counts["rows_with_image"] = len(df)
    df = df.merge(inv[inv["ok"]][["image_name", "md5", "phash"]], on="image_name", how="inner")
    counts["rows_with_decodable_image"] = len(df)
    pos = {n: i for i, n in enumerate(names)}
    df["emb_index"] = df["image_name"].map(pos)
    assert df["emb_index"].notna().all()
    df["emb_index"] = df["emb_index"].astype(int)

    # 1. metadata keywords
    text = (df["brand"] + " " + df["name"] + " " + df["description"]).str.lower()
    explicit_meta = text.str.contains(EXPLICIT_TERMS, regex=True)
    non_bottle_meta = (df["name"].str.lower()).str.contains(NON_BOTTLE_TERMS, regex=True)
    counts["dropped_metadata_explicit"] = int(explicit_meta.sum())
    counts["dropped_metadata_non_bottle"] = int((non_bottle_meta & ~explicit_meta).sum())
    df = df[~explicit_meta & ~non_bottle_meta].copy()

    # 2. zero shot image check
    zs = zero_shot(emb[df["emb_index"].to_numpy()])
    zs.index = df.index
    df = pd.concat([df, zs], axis=1)
    flagged_explicit = df["p_explicit"] > EXPLICIT_MAX
    flagged_person = (df["p_person"] > PERSON_MAX) & ~flagged_explicit
    not_bottle = (df["p_keep"] < KEEP_MIN) & ~flagged_explicit & ~flagged_person
    counts["dropped_image_flagged_explicit"] = int(flagged_explicit.sum())
    counts["dropped_image_person"] = int(flagged_person.sum())
    counts["dropped_image_not_bottle_photo"] = int(not_bottle.sum())
    counts["not_bottle_top_prompt"] = df[not_bottle]["top_prompt"].value_counts().to_dict()
    df = df[~flagged_explicit & ~flagged_person & ~not_bottle].copy()

    # 3. duplicates
    shared = df.groupby("md5")["product_key"].nunique()
    generic = df["md5"].isin(shared[shared > 1].index)
    counts["dropped_generic_brand_image"] = int(generic.sum())
    df = df[~generic]
    before = len(df)
    df = df.drop_duplicates(["product_key", "md5"])
    counts["dropped_same_product_identical"] = before - len(df)
    keep = []
    for _, grp in df.groupby("product_key", sort=False):
        bits: list[int] = []
        for idx, h in zip(grp.index, grp["phash"], strict=True):
            v = int(h, 16)
            if all(bin(v ^ b).count("1") > NEAR_DUP_BITS for b in bits):
                bits.append(v)
                keep.append(idx)
    counts["dropped_same_product_near_duplicate"] = len(df) - len(keep)
    df = df.loc[sorted(keep)].copy()

    # 4. product ids
    keys = sorted(df["product_key"].unique())
    pid = {k: f"p{i:05d}" for i, k in enumerate(keys)}
    df["product_id"] = df["product_key"].map(pid)
    counts["kept_images"] = len(df)
    counts["kept_products"] = len(keys)
    counts["kept_brands"] = int(df["brand"].str.lower().nunique())
    per = df.groupby("product_id").size()
    counts["kept_products_with_multiple_images"] = int((per > 1).sum())

    cols = [
        "row_id", "product_id", "product_key", "line_key", "brand", "name", "family", "subfamily",
        "accords", "notes", "gender", "year", "description", "image_name", "image_id", "emb_index",
        "p_keep", "p_person", "p_explicit",
    ]
    OUT.mkdir(parents=True, exist_ok=True)
    df[cols].to_csv(OUT / "gallery.csv", index=False)
    df[["image_id", "product_id", "row_id"]].to_csv(ROOT / "results" / "gallery_manifest.csv", index=False)
    (ROOT / "results" / "clean_counts.json").write_text(json.dumps(counts, indent=2) + "\n")
    print(json.dumps(counts, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
