"""Product level retrieval over image embeddings (no torch needed)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


class Gallery:
    """Gallery of reference embeddings grouped by product.

    A product's score for a query is the best cosine similarity over its
    reference images.
    """

    def __init__(self, embeddings: np.ndarray, product_ids: Sequence[str]):
        product_ids = np.asarray(product_ids)
        order = np.argsort(product_ids, kind="stable")
        self.emb = np.ascontiguousarray(embeddings[order], dtype=np.float32)
        sorted_ids = product_ids[order]
        self.products, self.starts = np.unique(sorted_ids, return_index=True)
        self.index = {p: i for i, p in enumerate(self.products)}
        self.single = len(self.products) == len(sorted_ids)

    def __len__(self) -> int:
        return len(self.products)

    def scores(self, queries: np.ndarray) -> np.ndarray:
        """(n_queries, n_products) product scores."""
        sims = np.asarray(queries, dtype=np.float32) @ self.emb.T
        if self.single:
            return sims
        return np.maximum.reduceat(sims, self.starts, axis=1)

    def truth_index(self, product_ids: Sequence[str | None]) -> np.ndarray:
        """Column index per query, or -1 for products not in the gallery."""
        return np.array([self.index.get(p, -1) if p is not None else -1 for p in product_ids], dtype=np.int64)

    def topk(self, queries: np.ndarray, k: int = 5) -> tuple[np.ndarray, np.ndarray]:
        """Top k product indices and scores per query, best first."""
        s = self.scores(queries)
        k = min(k, s.shape[1])
        idx = np.argpartition(-s, k - 1, axis=1)[:, :k]
        part = np.take_along_axis(s, idx, axis=1)
        order = np.argsort(-part, axis=1, kind="stable")
        return np.take_along_axis(idx, order, axis=1), np.take_along_axis(part, order, axis=1)
