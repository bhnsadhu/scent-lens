import pandas as pd

from scent_lens.ocr import TextMatcher, text_tokens


def _matcher():
    prods = pd.DataFrame(
        {
            "brand": ["Burberry", "Burberry", "Gucci", "Gucci", "Jo Malone London"],
            "name": ["Hero Eau de Parfum", "Mr. Burberry", "Guilty Pour Homme", "Gucci by Gucci", "Cypress & Grapevine Cologne Intense"],
        },
        index=["p1", "p2", "p3", "p4", "p5"],
    )
    return TextMatcher(prods, ["p1", "p2", "p3", "p4", "p5"])


def test_text_tokens_drop_generic_words():
    assert text_tokens("GUCCI GUILTY Eau de Parfum POUR HOMME 100 ml") == {"gucci", "guilty"}


def test_brand_and_name_words_both_count():
    s = _matcher().text_score(["BURBERRY", "HERO", "EAU DE PARFUM"])
    assert s[0] == 1.0  # Burberry Hero: brand and name read
    assert s[1] == 0.5  # Mr. Burberry: brand only, its name is just the brand


def test_fuzzy_misreads_still_match():
    s = _matcher().text_score(["JO MALONE", "CYPRESS", "& GRAPEVINE", "COLOGHE", "INTANSE"])
    assert s[4] == 1.0
    assert s[2] == 0.0


def test_explained_share_drops_with_extra_words():
    m = _matcher()
    assert m.explained(["BURBERRY", "HERO"], 0) == 1.0
    assert m.explained(["BURBERRY", "HERO", "ORCHID", "SANDALWOOD"], 0) == 0.5
    assert m.explained([], 0) == 0.0


def test_nothing_read_scores_zero():
    assert _matcher().text_score([]).sum() == 0.0
