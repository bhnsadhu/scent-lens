"""Bottle level splits: development pool, test split, and grouped folds."""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class Bottle:
    bottle_id: str
    brand: str
    in_catalog: bool


def _brand_interleaved(bottles: list[Bottle], rng: random.Random) -> list[Bottle]:
    """Order bottles brand by brand (brands and members shuffled).

    Assigning pools or folds round robin over this order spreads each brand
    across them, which is the stratification by brand.
    """
    by_brand: dict[str, list[Bottle]] = defaultdict(list)
    for b in sorted(bottles, key=lambda b: b.bottle_id):
        by_brand[b.brand.lower()].append(b)
    brands = sorted(by_brand)
    rng.shuffle(brands)
    ordered: list[Bottle] = []
    for brand in brands:
        members = by_brand[brand]
        rng.shuffle(members)
        ordered.extend(members)
    return ordered


def split_bottles(bottles: Iterable[Bottle], seed: int, test_fraction: float = 0.5) -> dict[str, str]:
    """Assign each bottle to "dev" or "test".

    Catalog and non catalog bottles are split separately so both pools get
    some of each. Within a group, bottles are taken in brand interleaved
    order and each is sent to whichever pool is furthest below its target.
    """
    bottles = list(bottles)
    ids = [b.bottle_id for b in bottles]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate bottle_id in split input")
    rng = random.Random(seed)
    assignment: dict[str, str] = {}
    for in_catalog in (True, False):
        group = [b for b in bottles if b.in_catalog == in_catalog]
        ordered = _brand_interleaved(group, rng)
        n_test = 0
        flip = rng.random() < 0.5
        for i, b in enumerate(ordered, start=1):
            # send to test when test is more than half a bottle behind target;
            # an exact tie goes to test only if flip, so odd groups vary by seed
            gap = test_fraction * i - n_test
            if gap > 0.5 or (gap == 0.5 and flip):
                assignment[b.bottle_id] = "test"
                n_test += 1
            else:
                assignment[b.bottle_id] = "dev"
    return assignment


def assign_folds(bottles: Iterable[Bottle], seed: int, n_folds: int = 5) -> dict[str, int]:
    """Assign development bottles to grouped folds, round robin in brand order."""
    bottles = list(bottles)
    rng = random.Random(seed)
    folds: dict[str, int] = {}
    offset = 0
    for in_catalog in (True, False):
        ordered = _brand_interleaved([b for b in bottles if b.in_catalog == in_catalog], rng)
        for i, b in enumerate(ordered):
            folds[b.bottle_id] = (i + offset) % n_folds
        offset += len(ordered)
    return folds


def check_manifest(pool: dict[str, str], folds: dict[str, int]) -> None:
    """Raise if the manifest leaks a bottle across pools or folds."""
    for bottle_id, p in pool.items():
        if p not in {"dev", "test"}:
            raise ValueError(f"bad pool {p!r} for {bottle_id}")
    for bottle_id in folds:
        if pool.get(bottle_id) != "dev":
            raise ValueError(f"fold assigned to non development bottle {bottle_id}")
    missing = [b for b, p in pool.items() if p == "dev" and b not in folds]
    if missing:
        raise ValueError(f"development bottles without a fold: {missing}")
