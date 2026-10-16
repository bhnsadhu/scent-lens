"""Confidence features, calibrators, and the abstention threshold."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.stats import beta
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

FEATURES = ("top1_sim", "margin", "brand_agree5")


def confidence_features(sorted_scores: np.ndarray, top_brands: Sequence[Sequence[str]]) -> np.ndarray:
    """Build (n, 3) features from per query product scores sorted descending.

    top1_sim: best score. margin: best minus second best. brand_agree5: share
    of the top 5 products that have the top 1 brand.
    """
    s = np.asarray(sorted_scores, dtype=np.float64)
    top1 = s[:, 0]
    margin = s[:, 0] - s[:, 1]
    agree = np.array(
        [sum(b == brands[0] for b in brands[:5]) / min(5, len(brands)) for brands in top_brands],
        dtype=np.float64,
    )
    return np.column_stack([top1, margin, agree])


@dataclass
class Calibrator:
    kind: str  # "logistic" or "isotonic"
    model: object
    feature_mean: np.ndarray
    feature_std: np.ndarray

    def predict(self, X: np.ndarray) -> np.ndarray:
        Z = (np.asarray(X, dtype=np.float64) - self.feature_mean) / self.feature_std
        if self.kind == "logistic":
            return self.model.predict_proba(Z)[:, 1]
        # isotonic works on one score: the logistic combination of features
        lin, iso = self.model
        return iso.predict(lin.decision_function(Z))


def fit_calibrator(X: np.ndarray, y: np.ndarray, kind: str = "logistic", seed: int = 0) -> Calibrator:
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    Z = (X - mean) / std
    if len(np.unique(y)) < 2:
        raise ValueError("calibrator needs both correct and incorrect examples")
    lin = LogisticRegression(C=1.0, random_state=seed, max_iter=1000).fit(Z, y)
    if kind == "logistic":
        return Calibrator("logistic", lin, mean, std)
    if kind == "isotonic":
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(lin.decision_function(Z), y)
        return Calibrator("isotonic", (lin, iso), mean, std)
    raise ValueError(f"unknown calibrator kind {kind!r}")


def out_of_fold_probs(
    X: np.ndarray, y: np.ndarray, folds: Sequence[int], kind: str = "logistic", seed: int = 0
) -> np.ndarray:
    """Calibrated probabilities where each photo is scored by a model that never saw its fold."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    f = np.asarray(folds)
    out = np.full(len(y), np.nan)
    for k in np.unique(f):
        train = f != k
        out[f == k] = fit_calibrator(X[train], y[train], kind, seed).predict(X[f == k])
    return out


def error_upper_bound(n_errors: int, n: int, confidence: float = 0.95) -> float:
    """One sided Clopper Pearson upper bound on an error rate."""
    if n == 0:
        return 1.0
    if n_errors >= n:
        return 1.0
    return float(beta.ppf(confidence, n_errors + 1, n - n_errors))


def choose_threshold(
    probs: Sequence[float], correct: Sequence[float], target_accuracy: float, confidence: float = 0.95
) -> float:
    """Lowest threshold whose accepted set has error upper bound at most 1 minus target.

    Returns inf (always abstain) if no threshold qualifies.
    """
    p = np.asarray(probs, dtype=np.float64)
    y = np.asarray(correct, dtype=np.int64)
    max_err = 1.0 - target_accuracy
    best = float("inf")
    for t in np.unique(p):
        accepted = p >= t
        n = int(accepted.sum())
        k = int(n - y[accepted].sum())
        if error_upper_bound(k, n, confidence) <= max_err:
            best = min(best, float(t))
    return best
