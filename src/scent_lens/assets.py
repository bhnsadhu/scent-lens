"""Locate galleries and query embeddings for a method name.

Image methods ("dinov2", "siglip2", "clip", "finetuned") match photos against
catalog image embeddings. Text methods ("clip_text", "siglip2_text") match the
same photo embeddings against "{brand} {name}" text embeddings.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scent_lens.retrieval import Gallery

ROOT = Path(__file__).resolve().parents[2]
IMAGE_METHODS = ("dinov2", "siglip2", "clip")
TEXT_METHODS = ("siglip2_text", "clip_text")


def catalog() -> pd.DataFrame:
    return pd.read_csv(ROOT / "artifacts" / "catalog" / "gallery.csv", dtype={"year": str}, keep_default_na=False)


def products(gal: pd.DataFrame) -> pd.DataFrame:
    return gal.drop_duplicates("product_id").set_index("product_id")


def backbone_of(method: str) -> str:
    return method.removesuffix("_text")


def gallery_for(method: str, gal: pd.DataFrame) -> Gallery:
    if method in TEXT_METHODS:
        d = ROOT / "artifacts" / "emb" / backbone_of(method)
        return Gallery(np.load(d / "text.npy"), (d / "text_products.txt").read_text().split())
    emb = np.load(ROOT / "artifacts" / "emb" / method / "gallery.npy")
    if method == "finetuned":
        names = (ROOT / "artifacts" / "emb" / method / "gallery_names.txt").read_text().split()
        pos = {n: i for i, n in enumerate(names)}
        idx = gal["image_name"].map(pos).to_numpy()
    else:
        idx = gal["emb_index"].to_numpy()
    return Gallery(emb[idx], gal["product_id"].tolist())


def photo_embeddings(method: str, photo_ids: list[str]) -> np.ndarray:
    name = backbone_of(method)
    d = ROOT / "artifacts" / "real"
    ids = (d / f"{name}_ids.txt").read_text().split()
    emb = np.load(d / f"{name}.npy")
    pos = {p: i for i, p in enumerate(ids)}
    return emb[[pos[p] for p in photo_ids]]
