import numpy as np

from scent_lens.metrics import (
    MISSING_RANK,
    bootstrap_by_group,
    brier,
    ece,
    rank_of_truth,
    risk_coverage,
    selective,
    subsampled_ranks,
    topk_from_ranks,
)


def test_rank_of_truth_with_ties_and_missing():
    scores = np.array([[0.9, 0.5, 0.1], [0.2, 0.8, 0.8], [0.3, 0.2, 0.1]])
    truth = np.array([0, 2, -1])
    ranks = rank_of_truth(scores, truth)
    assert ranks.tolist()[:2] == [0, 1]
    assert ranks[2] == MISSING_RANK
    assert topk_from_ranks(ranks, 1).tolist() == [1.0, 0.0, 0.0]
    assert topk_from_ranks(ranks, 5).tolist() == [1.0, 1.0, 0.0]


def test_subsampled_gallery_always_contains_truth():
    rng = np.random.default_rng(0)
    scores = rng.random((20, 50))
    truth = rng.integers(0, 50, size=20)
    full = rank_of_truth(scores, truth)
    sub = subsampled_ranks(scores, truth, gallery_size=10, n_draws=3, seed=1)
    assert sub.shape == (3, 20)
    assert (sub <= 9).all() and (sub >= 0).all()
    # a smaller gallery can only make the rank better or equal
    assert (sub <= full[None, :]).all()
    same = subsampled_ranks(scores, truth, gallery_size=50, n_draws=2, seed=1)
    assert (same == full[None, :]).all()


def test_ece_and_brier():
    assert ece([1.0, 1.0, 0.0, 0.0], [1, 1, 0, 0]) == 0.0
    assert abs(ece([0.8] * 10, [1] * 5 + [0] * 5) - 0.3) < 1e-12
    assert brier([1.0, 0.0], [1, 0]) == 0.0
    assert abs(brier([0.5, 0.5], [1, 0]) - 0.25) < 1e-12


def test_selective_and_risk_coverage():
    acc, cov = selective([0.9, 0.8, 0.2, 0.1], [1, 0, 1, 0], threshold=0.5)
    assert acc == 0.5 and cov == 0.5
    acc, cov = selective([0.1], [1], threshold=0.5)
    assert np.isnan(acc) and cov == 0.0
    coverage, risk = risk_coverage([0.9, 0.8, 0.2], [1, 0, 1])
    assert coverage.tolist() == [1 / 3, 2 / 3, 1.0]
    assert np.allclose(risk, [0.0, 0.5, 1 / 3])


def test_bootstrap_resamples_groups():
    # two bottles: one always right (3 photos), one always wrong (1 photo)
    groups = ["a", "a", "a", "b"]
    values = [1, 1, 1, 0]
    point, lo, hi = bootstrap_by_group(groups, values, n_boot=500, seed=0)
    assert point == 0.75
    assert lo <= point <= hi
    # resampled sets are whole bottles, so 0 and 1 are both reachable
    assert lo == 0.0 and hi == 1.0
    p2 = bootstrap_by_group(groups, values, n_boot=500, seed=0)
    assert p2 == (point, lo, hi)


def test_bootstrap_paired_statistic():
    groups = ["a", "b", "c", "d"]
    pairs = np.array([[1, 0], [1, 0], [1, 1], [0, 0]])
    point, lo, hi = bootstrap_by_group(
        groups, pairs, statistic=lambda v: float(v[:, 0].mean() - v[:, 1].mean()), n_boot=300
    )
    assert point == 0.5
    assert lo <= point <= hi
