"""Label reading for version 2: OCR text from a query photo, matched to catalog names.

Text is read with the macOS Vision framework (ocrmac, every reported result) or,
off macOS, RapidOCR (READERS). Words are normalized
like catalog names; generic words (concentrations, "pour homme", sizes,
cities) are dropped, so a match rests on brand and fragrance name words.
Each OCR word fuzzy matches catalog vocabulary words, which tolerates
misreads such as "COLOGHE" for "COLOGNE".
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

from scent_lens.normalize import CONCENTRATION_WORDS, normalize_text, product_key

STOP: frozenset[str] = CONCENTRATION_WORDS | frozenset(
    {
        "eau", "de", "du", "des", "la", "le", "les", "the", "and", "pour", "homme", "femme", "for", "men",
        "women", "him", "her", "spray", "vaporisateur", "natural", "ml", "oz", "fl", "paris", "london",
        "new", "york", "italia", "italy", "france", "made", "toilette", "fragrance", "edition",
    }
)
MIN_LEN = 3
CUTOFF = 85


def text_tokens(text: str) -> set[str]:
    return {t for t in normalize_text(text).split() if len(t) >= MIN_LEN and t not in STOP and not t.isdigit()}


def read_text(image, min_confidence: float = 0.3) -> list[str]:
    """Lines of text the macOS Vision framework finds in a PIL image."""
    from ocrmac import ocrmac

    return [t for t, conf, _ in ocrmac.OCR(image, recognition_level="accurate").recognize() if conf >= min_confidence]


def rapid_reader():
    """A cross platform reader (RapidOCR: PaddleOCR models on ONNX Runtime, CPU), library defaults."""
    from rapidocr_onnxruntime import RapidOCR

    engine = RapidOCR()

    def read(image) -> list[str]:
        result, _ = engine(np.asarray(image.convert("RGB"))[:, :, ::-1])  # RapidOCR takes BGR
        return [t for _, t, _ in result or []]

    return read


READERS = {"vision": lambda: read_text, "rapidocr": rapid_reader}


class TextMatcher:
    """Share of each product's brand words and name words found in OCR text, in gallery column order."""

    def __init__(self, products: pd.DataFrame, order: Sequence[str], cutoff: int = CUTOFF):
        self.cutoff = cutoff
        self.n = len(order)
        self.brand_n = np.zeros(self.n)
        self.name_n = np.zeros(self.n)
        self.where: dict[str, list[tuple[int, int]]] = defaultdict(list)  # word -> (column, 0 brand / 1 name)
        self.words: list[set[str]] = []
        for col, pid in enumerate(order):
            brand, _, name = product_key(products.loc[pid, "brand"], products.loc[pid, "name"]).partition("|")
            brand_toks = text_tokens(brand)
            self.words.append(brand_toks | text_tokens(name))
            for part, toks in ((0, brand_toks), (1, text_tokens(name) - brand_toks)):  # "Gucci by Gucci" has no name words
                (self.brand_n if part == 0 else self.name_n)[col] = len(toks)
                for t in toks:
                    self.where[t].append((col, part))
        self.vocab = sorted(self.where)

    def scores(self, lines: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
        """(brand fraction, name fraction) per product."""
        hits = set()
        for t in text_tokens(" ".join(lines)):
            hits.update(v for v, _, _ in process.extract(t, self.vocab, scorer=fuzz.ratio, score_cutoff=self.cutoff, limit=None))
        brand, name = np.zeros(self.n), np.zeros(self.n)
        for v in hits:
            for col, part in self.where[v]:
                (brand if part == 0 else name)[col] += 1
        return (np.divide(brand, self.brand_n, out=np.zeros(self.n), where=self.brand_n > 0),
                np.divide(name, self.name_n, out=np.zeros(self.n), where=self.name_n > 0))

    def explained(self, lines: Sequence[str], col: int) -> float:
        """Share of the words read that match a word of product col; 0 when nothing was read.

        Low when the label says more than the product name (Vanilla Orchid
        Sandalwood read, Vanilla guessed).
        """
        read = text_tokens(" ".join(lines))
        if not read:
            return 0.0
        words = self.words[col]
        return sum(any(fuzz.ratio(t, w) >= self.cutoff for w in words) for t in read) / len(read)

    def text_score(self, lines: Sequence[str]) -> np.ndarray:
        """Mean of the brand and name fractions; 0 when nothing was read."""
        b, n = self.scores(lines)
        return (b + n) / 2
