import pytest

from scent_lens.normalize import line_key, line_name, normalize_name, normalize_text, product_key


def test_normalize_text_strips_accents_and_punctuation():
    assert normalize_text("Terre d'Hermès!") == "terre d hermes"
    assert normalize_text("  Dolce & Gabbana ") == "dolce and gabbana"
    assert normalize_text(None) == ""


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Sauvage Eau de Toilette", "sauvage"),
        ("Sauvage EDT", "sauvage"),
        ("Sauvage Eau de Parfum 2019", "sauvage"),
        ("Bleu de Chanel Parfum", "bleu de chanel"),
        ("Acqua di Gio Cologne", "acqua di gio"),
        ("La Nuit de L'Homme Intense", "la nuit de l homme"),
        ("Aventus (2010)", "aventus"),
        ("Black Opium Extrait de Parfum", "black opium"),
    ],
)
def test_normalize_name_removes_concentration_and_years(raw, expected):
    assert normalize_name(raw) == expected


def test_normalize_name_drops_repeated_brand_prefix():
    assert normalize_name("Dior Sauvage", "Dior") == "sauvage"
    # a name that is only the brand keeps it
    assert normalize_name("Chanel", "Chanel") == "chanel"


def test_normalize_name_never_empty():
    assert normalize_name("Parfum") == "parfum"
    assert normalize_name("2012") == "2012"


def test_product_key_groups_concentrations():
    assert product_key("Dior", "Sauvage EDT") == product_key("DIOR", "Sauvage Eau de Toilette")
    assert product_key("Dior", "Sauvage") != product_key("Dior", "Sauvage Elixir")


def test_line_key_merges_flankers():
    assert line_name("Sauvage Elixir") == "sauvage"
    assert line_key("Dior", "Sauvage Elixir") == line_key("Dior", "Sauvage")
    assert line_key("Paco Rabanne", "1 Million Noir") == line_key("Paco Rabanne", "1 Million")
    assert line_key("Dior", "Sauvage") != line_key("Chanel", "Sauvage")


def test_line_key_never_empty():
    assert line_name("Noir") == "noir"
