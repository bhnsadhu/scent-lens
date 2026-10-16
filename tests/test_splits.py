import pytest

from scent_lens.splits import Bottle, assign_folds, check_manifest, split_bottles


def _bottles(n_in=40, n_out=6):
    brands = ["Dior", "Chanel", "Creed", "YSL", "Hermes", "Tom Ford", "Armani"]
    out = [Bottle(f"b{i:02d}", brands[i % len(brands)], True) for i in range(n_in)]
    out += [Bottle(f"x{i:02d}", "Indie", False) for i in range(n_out)]
    return out


def test_split_is_deterministic_and_balanced():
    bottles = _bottles()
    a = split_bottles(bottles, seed=13)
    assert a == split_bottles(bottles, seed=13)
    assert set(a) == {b.bottle_id for b in bottles}
    n_test = sum(v == "test" for v in a.values())
    assert abs(n_test - len(bottles) / 2) <= 1
    # unmatched bottles go to both pools
    out_pools = {a[b.bottle_id] for b in bottles if not b.in_catalog}
    assert out_pools == {"dev", "test"}


def test_split_stratifies_by_brand():
    bottles = _bottles()
    a = split_bottles(bottles, seed=13)
    for brand in {b.brand for b in bottles if b.in_catalog}:
        pools = [a[b.bottle_id] for b in bottles if b.brand == brand]
        assert abs(pools.count("test") - pools.count("dev")) <= 1


def test_no_bottle_in_both_pools_or_two_folds():
    bottles = _bottles()
    pool = split_bottles(bottles, seed=7)
    dev = [b for b in bottles if pool[b.bottle_id] == "dev"]
    folds = assign_folds(dev, seed=7, n_folds=5)
    check_manifest(pool, folds)
    # each dev bottle has exactly one fold, test bottles have none
    assert set(folds) == {b.bottle_id for b in dev}
    assert all(0 <= f < 5 for f in folds.values())
    assert not set(folds) & {b for b, p in pool.items() if p == "test"}
    sizes = [list(folds.values()).count(k) for k in range(5)]
    assert max(sizes) - min(sizes) <= 1


def test_check_manifest_catches_leaks():
    with pytest.raises(ValueError):
        check_manifest({"a": "test"}, {"a": 0})
    with pytest.raises(ValueError):
        check_manifest({"a": "dev"}, {})
    with pytest.raises(ValueError):
        split_bottles([Bottle("a", "x", True), Bottle("a", "y", True)], seed=0)
