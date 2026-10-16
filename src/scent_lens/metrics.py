"""Retrieval and selective prediction metrics, with bottle level bootstrap."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

# Rank given to queries whose truth is not in the gallery: larger than any k.
MISSING_RANK = np.iinfo(np.int64).max


def rank_of_truth(scores: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Rank (0 based) of the true product per query.

    scores: (n_queries, n_products) similarity. truth: (n_queries,) column
    index of the true product, or -1 when the truth is not in the gallery
    (rank is then MISSING_RANK, never a hit for any k). Ties count against
    the truth: rank = number of products scoring at least as high, minus one.
    """
    n_q, _ = scores.shape
    ranks = np.full(n_q, MISSING_RANK, dtype=np.int64)
    ok = truth >= 0
    true_scores = scores[np.where(ok)[0], truth[ok]]
    ranks[ok] = (scores[ok] >= true_scores[:, None]).sum(axis=1) - 1
    return ranks


def topk_from_ranks(ranks: np.ndarray, k: int) -> np.ndarray:
    return (ranks < k).astype(np.float64)


def subsampled_ranks(
    scores: np.ndarray, truth: np.ndarray, gallery_size: int, n_draws: int, seed: int
) -> np.ndarray:
    """Ranks of the truth inside smaller galleries.

    Each gallery holds the true product plus gallery_size minus one distractors
    drawn without replacement, independently per query and per draw. Returns
    (n_draws, n_queries) ranks.

    The rank inside such a gallery is the number of drawn distractors that
    score at least as high as the truth. With b such products among the n - 1
    distractors, that count is hypergeometric(b, n - 1 - b, gallery_size - 1),
    so it is sampled directly; this is exactly the distribution of drawing the
    gallery explicitly, without materializing it.
    """
    n_q, n_p = scores.shape
    if (truth < 0).any():
        raise ValueError("subsampled galleries need the truth in the gallery")
    full = rank_of_truth(scores, truth)
    if gallery_size >= n_p:
        return np.tile(full, (n_draws, 1))
    rng = np.random.default_rng(seed)
    good = full  # distractors scoring at least as high as the truth
    bad = (n_p - 1) - full
    return rng.hypergeometric(good, bad, gallery_size - 1, size=(n_draws, n_q)).astype(np.int64)


def ece(probs: Sequence[float], correct: Sequence[float], n_bins: int = 10) -> float:
    """Expected calibration error with equal width bins on [0, 1]."""
    p = np.asarray(probs, dtype=np.float64)
    y = np.asarray(correct, dtype=np.float64)
    if p.size == 0:
        return float("nan")
    bins = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            total += m.sum() * abs(p[m].mean() - y[m].mean())
    return float(total / p.size)


def brier(probs: Sequence[float], correct: Sequence[float]) -> float:
    p = np.asarray(probs, dtype=np.float64)
    y = np.asarray(correct, dtype=np.float64)
    return float(np.mean((p - y) ** 2))


def selective(confidence: Sequence[float], correct: Sequence[float], threshold: float) -> tuple[float, float]:
    """Return (selective accuracy, coverage) when answering iff confidence >= threshold.

    Selective accuracy is nan when nothing is answered.
    """
    c = np.asarray(confidence, dtype=np.float64)
    y = np.asarray(correct, dtype=np.float64)
    answered = c >= threshold
    coverage = float(answered.mean()) if c.size else float("nan")
    acc = float(y[answered].mean()) if answered.any() else float("nan")
    return acc, coverage


def risk_coverage(confidence: Sequence[float], correct: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    """Coverage and risk (1 minus selective accuracy) at every distinct threshold."""
    c = np.asarray(confidence, dtype=np.float64)
    y = np.asarray(correct, dtype=np.float64)
    order = np.argsort(-c, kind="stable")
    y_sorted = y[order]
    n = np.arange(1, c.size + 1)
    coverage = n / c.size
    risk = 1.0 - np.cumsum(y_sorted) / n
    return coverage, risk


def bootstrap_by_group(
    groups: Sequence[str],
    values: Sequence[float] | np.ndarray,
    statistic: Callable[[np.ndarray], float] | None = None,
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Point estimate and percentile interval, resampling groups (bottles).

    values may be 1 dimensional (one value per photo) or 2 dimensional (one row
    per photo, for paired statistics). The statistic receives the stacked rows
    of the resampled groups; default is the mean of a 1 dimensional value.
    """
    vals = np.asarray(values, dtype=np.float64)
    stat = statistic or (lambda v: float(np.nanmean(v)))
    g = np.asarray(groups)
    uniq = np.unique(g)
    idx_by_group = [np.where(g == u)[0] for u in uniq]
    point = stat(vals)
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.integers(0, len(uniq), size=len(uniq))
        rows = np.concatenate([idx_by_group[j] for j in pick])
        boots[i] = stat(vals[rows])
    lo, hi = np.nanpercentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(point), float(lo), float(hi)
