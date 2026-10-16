from scent_lens.matching import lenient_correct, match_bottle, match_confidence, strict_correct


def test_strict_rule():
    assert strict_correct("p1", "p1")
    assert not strict_correct("p1", "p2")
    # bottle not in catalog: any answer is wrong
    assert not strict_correct("p1", None)
    assert not strict_correct(None, "p1")


def test_lenient_rule():
    assert lenient_correct("Dior", "Sauvage Elixir", "Dior", "Sauvage")
    assert lenient_correct("dior", "Sauvage EDP", "Dior", "Sauvage EDT")
    assert not lenient_correct("Dior", "Sauvage", "Chanel", "Bleu de Chanel")
    assert not lenient_correct("Dior", "Sauvage", None, None)


def _match(toy_catalog, brand, name):
    return match_bottle(
        brand,
        name,
        toy_catalog["brand"].tolist(),
        toy_catalog["name"].tolist(),
        toy_catalog["product_id"].tolist(),
    )


def test_match_exact_and_variants(toy_catalog):
    assert _match(toy_catalog, "Dior", "Sauvage EDT")[0].product_id == "p1"
    assert _match(toy_catalog, "dior", "sauvage elixir")[0].product_id == "p2"
    assert _match(toy_catalog, "Hermes", "Terre d Hermes")[0].product_id == "p10"
    assert _match(toy_catalog, "YSL Yves Saint Laurent", "Libre")[0].product_id == "p5"


def test_match_unknown_brand_returns_nearest(toy_catalog):
    cands = _match(toy_catalog, "Nobody Perfumery", "Imaginary Scent")
    assert len(cands) == 3
    assert match_confidence(cands) == "check"


def test_match_confidence_labels(toy_catalog):
    assert match_confidence(_match(toy_catalog, "Creed", "Aventus")) == "high"
    assert match_confidence([]) == "check"


def test_brand_aliases_and_prefixed_names():
    import pandas as pd

    from scent_lens.matching import canonical_brand

    assert canonical_brand("YSL") == "yves saint laurent"
    assert canonical_brand("D&G") == "dolce and gabbana"
    cat = pd.DataFrame(
        {
            "product_id": ["m1", "m2", "y1"],
            "brand": ["Maison Margiela", "Maison Margiela", "Yves Saint Laurent"],
            "name": ["Replica : Jazz Club", "Replica : Beach Walk", "Libre"],
        }
    )
    assert _match(cat, "YSL", "Libre")[0].product_id == "y1"
    assert _match(cat, "Maison Margiela", "Jazz Club")[0].product_id == "m1"
