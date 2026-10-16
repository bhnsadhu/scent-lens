"""Correctness rules and fuzzy matching of a bottle list to the catalog."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from rapidfuzz import fuzz, process

from scent_lens.normalize import line_key, normalize_brand, normalize_name, product_key


def strict_correct(pred_product_id: str | None, true_product_id: str | None) -> bool:
    """Strict: the prediction is the same product group as the truth.

    A missing truth (bottle not in the catalog) makes every answer incorrect.
    """
    if pred_product_id is None or true_product_id is None:
        return False
    return pred_product_id == true_product_id


def lenient_correct(
    pred_brand: str | None, pred_name: str | None, true_brand: str | None, true_name: str | None
) -> bool:
    """Lenient: same brand and same line key (flanker words removed)."""
    if not true_brand or not true_name or not pred_brand or not pred_name:
        return False
    return line_key(pred_brand, pred_name) == line_key(true_brand, true_name)


@dataclass(frozen=True)
class Candidate:
    product_id: str
    brand: str
    name: str
    score: float


def match_bottle(
    brand: str,
    name: str,
    catalog_brands: Sequence[str],
    catalog_names: Sequence[str],
    catalog_ids: Sequence[str],
    k: int = 3,
    brand_cutoff: float = 80.0,
) -> list[Candidate]:
    """Return the top k catalog candidates for one bottle.

    Brand is matched first (token set ratio on normalized brand, cutoff
    brand_cutoff); names are then scored within the matching brands. If no
    brand passes the cutoff, names are scored against the whole catalog with
    the brand folded into the string, so the reviewer still sees the nearest
    entries.
    """
    q_brand = normalize_brand(brand)
    q_name = normalize_name(name, brand)
    norm_brands = [normalize_brand(b) for b in catalog_brands]

    brand_scores = {
        b: fuzz.token_set_ratio(q_brand, b) for b in set(norm_brands)
    }
    good = {b for b, s in brand_scores.items() if s >= brand_cutoff}

    scored: list[Candidate] = []
    if good:
        for i, b in enumerate(norm_brands):
            if b not in good:
                continue
            c_name = normalize_name(catalog_names[i], catalog_brands[i])
            name_score = fuzz.token_sort_ratio(q_name, c_name)
            score = 0.25 * brand_scores[b] + 0.75 * name_score
            scored.append(Candidate(catalog_ids[i], catalog_brands[i], catalog_names[i], score))
    else:
        choices = [f"{b} {normalize_name(n, cb)}" for b, n, cb in zip(norm_brands, catalog_names, catalog_brands, strict=True)]
        hits = process.extract(f"{q_brand} {q_name}", choices, scorer=fuzz.token_sort_ratio, limit=k * 5)
        for _, score, i in hits:
            scored.append(Candidate(catalog_ids[i], catalog_brands[i], catalog_names[i], float(score)))

    best: dict[str, Candidate] = {}
    for c in scored:
        if c.product_id not in best or c.score > best[c.product_id].score:
            best[c.product_id] = c
    ranked = sorted(best.values(), key=lambda c: (-c.score, c.product_id))
    return ranked[:k]


def match_confidence(candidates: Sequence[Candidate], high: float = 92.0, gap: float = 5.0) -> str:
    """Label a match "high" when the top score is high and clearly ahead, else "check"."""
    if not candidates:
        return "check"
    top = candidates[0].score
    second = candidates[1].score if len(candidates) > 1 else 0.0
    return "high" if top >= high and top - second >= gap else "check"


__all__ = [
    "Candidate",
    "lenient_correct",
    "match_bottle",
    "match_confidence",
    "product_key",
    "strict_correct",
]
