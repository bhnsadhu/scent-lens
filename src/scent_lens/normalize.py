"""Text normalization for brand and fragrance names.

Two keys are derived from a catalog row:

* product key: (brand, normalized name). Listings that share it are the same
  product group. Used for the strict correctness rule.
* line key: (brand, name with flanker words removed). Used for the lenient
  correctness rule, so "Sauvage Elixir" and "Sauvage" share a line.
"""

from __future__ import annotations

import re
import unicodedata

# Concentration words removed from names before grouping. Multi word phrases
# are listed first so they are removed before their single word parts.
CONCENTRATION_PHRASES: tuple[str, ...] = (
    "extrait de parfum",
    "eau de parfum",
    "eau de toilette",
    "eau de cologne",
    "parfum intense",
)
CONCENTRATION_WORDS: frozenset[str] = frozenset(
    {"edt", "edp", "edc", "parfum", "perfume", "cologne", "intense", "extrait"}
)

# Flanker words removed when building the line key for the lenient rule.
FLANKER_WORDS: frozenset[str] = frozenset(
    {
        "noir", "sport", "extreme", "absolu", "absolute", "elixir", "fraiche",
        "fresh", "night", "nuit", "summer", "winter", "edition", "limited",
        "collector", "platinum", "black", "rouge", "blanc", "le", "la", "l",
        "eau", "de", "du", "the", "privee", "prive", "pure", "essence",
        "concentree", "legere", "leger", "light", "acqua", "aqua", "cologne",
    }
)

YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def strip_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def normalize_text(text: str | None) -> str:
    """Lowercase, strip accents and punctuation, collapse whitespace."""
    if text is None:
        return ""
    text = strip_accents(str(text)).lower().replace("&", " and ")
    text = NON_ALNUM_RE.sub(" ", text)
    return " ".join(text.split())


def normalize_brand(brand: str | None) -> str:
    return normalize_text(brand)


def normalize_name(name: str | None, brand: str | None = None) -> str:
    """Normalize a fragrance name for product grouping.

    Removes concentration words and four digit years. If the brand is given
    and the name starts with it, the brand prefix is removed too, since some
    listings repeat the brand inside the name. An empty result falls back to
    the plain normalized name so that a product called "Parfum" keeps a key.
    """
    base = normalize_text(name)
    text = f" {YEAR_RE.sub(' ', base)} "
    for phrase in CONCENTRATION_PHRASES:
        text = text.replace(f" {phrase} ", " ")
    tokens = [t for t in text.split() if t not in CONCENTRATION_WORDS]
    if brand:
        brand_tokens = normalize_brand(brand).split()
        if brand_tokens and tokens[: len(brand_tokens)] == brand_tokens and len(tokens) > len(brand_tokens):
            tokens = tokens[len(brand_tokens):]
    result = " ".join(tokens)
    return result if result else base


def line_name(name: str | None, brand: str | None = None) -> str:
    """Name with flanker words removed, for the lenient rule."""
    normalized = normalize_name(name, brand)
    tokens = [t for t in normalized.split() if t not in FLANKER_WORDS]
    return " ".join(tokens) if tokens else normalized


def product_key(brand: str | None, name: str | None) -> str:
    return f"{normalize_brand(brand)}|{normalize_name(name, brand)}"


def line_key(brand: str | None, name: str | None) -> str:
    return f"{normalize_brand(brand)}|{line_name(name, brand)}"
