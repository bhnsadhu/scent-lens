"""Per photo retrieval records and summary metrics (no torch needed)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from scent_lens.calibration import confidence_features
from scent_lens.metrics import bootstrap_by_group, rank_of_truth
from scent_lens.retrieval import Gallery


def photo_records(
    gallery: Gallery, products: pd.DataFrame, emb: np.ndarray, photos: pd.DataFrame, k: int = 5
) -> pd.DataFrame:
    """One row per photo: top k predictions, correctness, confidence features.

    products: indexed by product_id with brand and line_key columns.
    photos: photo_id, bottle_id, condition, status, product_id (empty when the
    bottle is not in the catalog), in the same order as emb.
    """
    truth_ids = [p if isinstance(p, str) and p else None for p in photos["product_id"]]
    truth = gallery.truth_index(truth_ids)
    scores = gallery.scores(emb)
    ranks = rank_of_truth(scores, truth)
    idx = np.argsort(-scores, axis=1, kind="stable")[:, :k]
    top_scores = np.take_along_axis(scores, idx, axis=1)
    top_ids = gallery.products[idx]
    brands = products["brand"].str.lower()
    top_brands = [[brands.loc[p] for p in row] for row in top_ids]
    feats = confidence_features(top_scores, top_brands)

    out = photos[["photo_id", "bottle_id", "condition", "status", "product_id"]].copy().reset_index(drop=True)
    for j in range(k):
        out[f"pred{j + 1}"] = top_ids[:, j]
        out[f"score{j + 1}"] = top_scores[:, j]
    out["rank"] = np.where(ranks > 10**9, -1, ranks)
    out["top1"] = (ranks < 1).astype(float)
    out["top5"] = (ranks < 5).astype(float)
    true_line = [products.loc[p, "line_key"] if p else None for p in truth_ids]
    pred_line = products.loc[top_ids[:, 0], "line_key"].to_numpy()
    out["lenient1"] = [float(t is not None and t == p) for t, p in zip(true_line, pred_line, strict=True)]
    out["top1_sim"], out["margin"], out["brand_agree5"] = feats[:, 0], feats[:, 1], feats[:, 2]
    return out


def summarize(rec: pd.DataFrame, n_boot: int = 2000, seed: int = 0, matched_only: bool = True) -> dict:
    """Top 1, top 5, lenient top 1 with bottle level bootstrap intervals."""
    r = rec[rec.status == "matched"] if matched_only else rec
    out = {"photos": int(len(r)), "bottles": int(r.bottle_id.nunique())}
    for col in ("top1", "top5", "lenient1"):
        point, lo, hi = bootstrap_by_group(r.bottle_id, r[col], n_boot=n_boot, seed=seed)
        out[col] = {"value": point, "lo": lo, "hi": hi}
    return out


def by_condition(rec: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> dict:
    r = rec[rec.status == "matched"]
    return {c: summarize(g, n_boot, seed) for c, g in r.groupby("condition")}
