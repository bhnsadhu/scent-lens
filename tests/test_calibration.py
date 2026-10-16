import numpy as np
import pytest

from scent_lens.calibration import (
    choose_threshold,
    confidence_features,
    error_upper_bound,
    fit_calibrator,
    out_of_fold_probs,
)
from scent_lens.metrics import ece


def _toy(n=300, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    logit = 2.0 * X[:, 0] + 1.0 * X[:, 1]
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    folds = np.arange(n) % 5
    return X, y, folds


def test_confidence_features():
    scores = np.array([[0.9, 0.7, 0.5, 0.4, 0.3], [0.6, 0.59, 0.5, 0.4, 0.3]])
    brands = [["dior", "dior", "chanel", "dior", "creed"], ["a", "b", "c", "d", "e"]]
    f = confidence_features(scores, brands)
    assert np.allclose(f[:, 0], [0.9, 0.6])
    assert np.allclose(f[:, 1], [0.2, 0.01])
    assert np.allclose(f[:, 2], [0.6, 0.2])


@pytest.mark.parametrize("kind", ["logistic", "isotonic"])
def test_calibrators_fit_and_are_reasonable(kind):
    X, y, folds = _toy()
    probs = out_of_fold_probs(X, y, folds, kind=kind)
    assert not np.isnan(probs).any()
    assert ((probs >= 0) & (probs <= 1)).all()
    assert ece(probs, y) < 0.15


def test_out_of_fold_never_scores_with_own_fold(monkeypatch):
    """Each fold's photos are scored by a model fit only on the other folds."""
    import scent_lens.calibration as cal

    X, y, folds = _toy(n=50)
    seen = []
    real_fit = cal.fit_calibrator

    def spy(Xt, yt, kind="logistic", seed=0):
        seen.append({tuple(r) for r in np.round(Xt, 12)})
        return real_fit(Xt, yt, kind, seed)

    monkeypatch.setattr(cal, "fit_calibrator", spy)
    cal.out_of_fold_probs(X, y, folds)
    for k, train_rows in zip(np.unique(folds), seen, strict=True):
        held = {tuple(r) for r in np.round(X[folds == k], 12)}
        assert not held & train_rows


def test_calibrator_needs_both_classes():
    with pytest.raises(ValueError):
        fit_calibrator(np.zeros((4, 3)), np.ones(4))


def test_error_upper_bound():
    assert error_upper_bound(0, 0) == 1.0
    ub = error_upper_bound(0, 30)
    assert 0.09 < ub < 0.1  # rule of three, about 3 / n
    assert error_upper_bound(5, 50) > 0.1


def test_choose_threshold_is_conservative():
    probs = np.linspace(0, 1, 200)
    correct = (probs > 0.3).astype(int)
    t = choose_threshold(probs, correct, target_accuracy=0.9)
    accepted = probs >= t
    n, k = int(accepted.sum()), int((1 - correct[accepted]).sum())
    # the accepted set meets the target with the 95% error upper bound
    assert error_upper_bound(k, n) <= 0.1
    assert correct[accepted].mean() >= 0.9
    # and t is the lowest threshold that does: every lower one fails the bound
    for lower in probs[probs < t]:
        acc = probs >= lower
        n2, k2 = int(acc.sum()), int((1 - correct[acc]).sum())
        assert error_upper_bound(k2, n2) > 0.1
    # impossible target means always abstain
    assert choose_threshold([0.5, 0.6], [0, 0], target_accuracy=0.9) == float("inf")
