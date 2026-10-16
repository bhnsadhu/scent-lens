"""Embed the synthetic query set: 2,000 catalog images under each condition.

The query sample is drawn once with a fixed seed from the cleaned gallery and
saved to results/synthetic_query_ids.csv. Every augmented image is built once
and embedded by all backbones, so they see identical pixels. Output chunks go
to artifacts/synth/<backbone>/<condition>_XXXX.npy and are skipped if present.
Also embeds product text prompts "{brand} {name}" for the zero shot baseline.
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from scent_lens.augment import CONDITIONS, apply  # noqa: E402
from scent_lens.catalog import ZipImages  # noqa: E402
from scent_lens.embed import load_encoder  # noqa: E402

SEED = 20261015
N_QUERIES = 2000
CHUNK = 250
GALLERY = ROOT / "artifacts" / "catalog" / "gallery.csv"
QUERY_IDS = ROOT / "results" / "synthetic_query_ids.csv"
OUT = ROOT / "artifacts" / "synth"


def query_sample(gal: pd.DataFrame) -> pd.DataFrame:
    if QUERY_IDS.exists():
        ids = pd.read_csv(QUERY_IDS)
        return ids.merge(gal, on=["image_id", "product_id"], how="left", validate="one_to_one")
    rng = np.random.default_rng(SEED)
    pick = np.sort(rng.choice(len(gal), size=N_QUERIES, replace=False))
    q = gal.iloc[pick].reset_index(drop=True)
    q.insert(0, "query_index", np.arange(len(q)))
    q[["query_index", "image_id", "product_id"]].to_csv(QUERY_IDS, index=False)
    return q


def embed_text(encoders: dict, gal: pd.DataFrame) -> None:
    products = gal.drop_duplicates("product_id").sort_values("product_id")
    texts = (products["brand"] + " " + products["name"]).tolist()
    for name, enc in encoders.items():
        if name == "dinov2":
            continue
        path = ROOT / "artifacts" / "emb" / name / "text.npy"
        if path.exists():
            continue
        parts = [enc.encode_texts(texts[i : i + 256]) for i in range(0, len(texts), 256)]
        np.save(path, np.concatenate(parts).astype(np.float32))
        (path.parent / "text_products.txt").write_text("\n".join(products["product_id"]) + "\n")
        print(f"text {name}: {len(texts)} products", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbones", default="clip,siglip2,dinov2")
    args = ap.parse_args()
    names = args.backbones.split(",")
    gal = pd.read_csv(GALLERY, dtype={"year": str}, keep_default_na=False)
    q = query_sample(gal)
    store = ZipImages(ROOT / "data" / "raw" / "images.zip")
    encoders = {n: load_encoder(n) for n in names}
    embed_text(encoders, gal)

    conditions = list(CONDITIONS)
    n_chunks = (len(q) + CHUNK - 1) // CHUNK
    with ThreadPoolExecutor(max_workers=6) as pool:
        for ci, cond in enumerate(conditions):
            for c in range(n_chunks):
                paths = {n: OUT / n / f"{cond}_{c:04d}.npy" for n in names}
                if all(p.exists() for p in paths.values()):
                    continue
                rows = q.iloc[c * CHUNK : (c + 1) * CHUNK]

                def make(row, ci=ci, cond=cond):
                    img = store.open(row.image_name, max_side=448)
                    return apply(cond, img, seed=SEED + 100 * int(row.query_index) + ci)

                imgs = list(pool.map(make, rows.itertuples()))
                for n, enc in encoders.items():
                    if paths[n].exists():
                        continue
                    paths[n].parent.mkdir(parents=True, exist_ok=True)
                    e = np.concatenate([enc.encode_images(imgs[i : i + 64]) for i in range(0, len(imgs), 64)])
                    np.save(paths[n].with_suffix(".tmp.npy"), e.astype(np.float32))
                    paths[n].with_suffix(".tmp.npy").rename(paths[n])
                print(f"{cond} chunk {c + 1}/{n_chunks}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
