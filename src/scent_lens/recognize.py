"""The version 2 system as one call: a photo in, an answer or "not sure" out.

    rec = Recognizer("artifacts/export")
    result = rec(Image.open("bottle.jpg"))

Steps, as fixed in results/calibrator_v2.json and docs/decisions.md: crop the
bottle (OWLv2), embed the crop at 0, 90 and 270 degrees with the fine tuned
SigLIP 2 and keep each product's best score, read the label and add
text_weight times the label match, then answer when the label clearly names
one product, or when the calibrated confidence reaches the threshold and the
label does not contradict the guess. The decision (decide) needs no torch.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from scent_lens.calibration import confidence_features, predict_from_dict
from scent_lens.ocr import READERS, TextMatcher, text_tokens

ROTATIONS = {90: Image.Transpose.ROTATE_90, 270: Image.Transpose.ROTATE_270}


@dataclass
class Match:
    product_id: str
    brand: str
    name: str
    score: float
    details: dict = field(default_factory=dict)  # family, notes and the other products.csv columns


@dataclass
class Result:
    answered: bool
    reason: str  # "clear label", "confident", "low confidence", "label disagrees"
    confidence: float  # calibrated chance the top match is right (1.0 on a clear label)
    matches: list[Match]
    text: list[str]
    box: tuple[int, int, int, int] | None
    seconds: float = 0.0
    timings: dict = field(default_factory=dict)  # seconds per step: crop, embed, read, match

    @property
    def best(self) -> Match | None:
        return self.matches[0] if self.answered else None

    def to_dict(self) -> dict:
        """JSON ready: product_id is the stable catalog id (artifacts/export/products.csv)."""
        return {"answered": self.answered, "reason": self.reason, "confidence": round(self.confidence, 4),
                "product_id": self.best.product_id if self.best else None,
                "matches": [{"product_id": m.product_id, "brand": m.brand, "name": m.name, "score": round(m.score, 4)} | m.details
                            for m in self.matches],
                "text_read": self.text, "box": list(self.box) if self.box else None, "seconds": round(self.seconds, 3),
                "timings": {k: round(v, 3) for k, v in self.timings.items()}}


def decide(scores: np.ndarray, text: np.ndarray, explained: float, read: bool, brands: Sequence[str],
           settings: dict) -> tuple[bool, str, float, np.ndarray]:
    """Answer or abstain for one photo.

    scores: image score per product (best over rotations); text: label match
    per product; explained: share of the words read that match the top
    product; read: whether any word was read; brands: lower case brand per
    product. Returns (answered, reason, confidence, top 5 product indices).
    """
    total = scores + settings["text_weight"] * text
    top = np.argsort(-total, kind="stable")[:5]
    feats = confidence_features(total[top][None, :], [[brands[i] for i in top]])
    conf = float(predict_from_dict(settings["calibrator"], feats)[0])
    share = settings["guard_share"]
    if text[top[0]] >= 1.0 and explained >= share and int((text >= 1.0).sum()) == 1:
        return True, "clear label", 1.0, top
    if read and explained < share:
        return False, "label disagrees", conf, top
    if conf >= settings["threshold"]:
        return True, "confident", conf, top
    return False, "low confidence", conf, top


def default_reader() -> str:
    """The label reader name: macOS Vision when ocrmac is installed, else RapidOCR.

    Every version 2 setting was chosen with macOS Vision; RapidOCR runs with the
    same settings (results/recognizer_rapidocr.md).
    """
    try:
        import ocrmac  # noqa: F401
    except ImportError:
        return "rapidocr"
    return "vision"


class Recognizer:
    def __init__(self, export_dir: str | Path, device: str | None = None,
                 reader: str | Callable[[Image.Image], list[str]] | None = None, detector=None):
        import torch
        from transformers import SiglipVisionConfig, SiglipVisionModel

        from scent_lens.crop import BottleDetector
        from scent_lens.embed import Encoder, pick_device
        from scent_lens.finetune import EmbeddingModel
        from scent_lens.retrieval import Gallery

        d = Path(export_dir)
        self.settings = json.loads((d / "settings.json").read_text())
        self.device = device or pick_device()
        cfg = SiglipVisionConfig.from_json_file(d / "vision_config.json")
        model = EmbeddingModel("siglip2", SiglipVisionModel(cfg), cfg.hidden_size)
        model.load_state_dict({k: v.float() for k, v in torch.load(d / "model.pt", map_location="cpu").items()})
        e = json.loads((d / "encoder.json").read_text())
        self.encoder = Encoder("finetuned", model.eval().to(self.device), e["size"], tuple(e["mean"]), tuple(e["std"]), self.device)
        ids = (d / "gallery_products.txt").read_text().split()
        self.gallery = Gallery(np.load(d / "gallery.npy").astype(np.float32), ids)
        self.products = pd.read_csv(d / "products.csv", index_col="product_id", dtype=str, keep_default_na=False)
        self.brands = self.products.loc[self.gallery.products, "brand"].str.lower().tolist()
        self.matcher = TextMatcher(self.products, list(self.gallery.products))
        self.reader_name = reader if isinstance(reader, str) else ("custom" if reader else default_reader())
        self.reader = READERS[self.reader_name]() if self.reader_name in READERS else reader
        self.detector = detector or BottleDetector(self.device)

    def __call__(self, image: Image.Image | str | Path) -> Result:
        from scent_lens.crop import apply_crop
        from scent_lens.photos import as_saved, clean_image, clean_pil

        t0 = time.perf_counter()
        marks = {}
        # the same path as the measured photos: cleaned and saved, then cropped and saved
        img = as_saved(clean_image(image) if isinstance(image, str | Path) else clean_pil(image))
        box, _ = self.detector.detect(img)
        crop = as_saved(apply_crop(img, box))
        marks["crop"] = time.perf_counter()
        views = [crop] + [crop.transpose(r) for r in ROTATIONS.values()]
        scores = self.gallery.scores(self.encoder.encode_images(views)).max(axis=0)
        marks["embed"] = time.perf_counter()
        lines = [t for v in views for t in self.reader(v)]
        marks["read"] = time.perf_counter()
        text = self.matcher.text_score(lines)
        top0 = int(np.argmax(scores + self.settings["text_weight"] * text))
        explained = self.matcher.explained(lines, top0)
        read = bool(text_tokens(" ".join(lines)))
        answered, reason, conf, top = decide(scores, text, explained, read, self.brands, self.settings)
        total = scores + self.settings["text_weight"] * text
        matches = []
        for i in top[:3]:
            pid = self.gallery.products[i]
            row = self.products.loc[pid]
            matches.append(Match(pid, row.brand, row["name"], float(total[i]), row.drop(["brand", "name"]).to_dict()))
        marks["match"] = time.perf_counter()
        prev, timings = t0, {}
        for step, t in marks.items():
            timings[step], prev = t - prev, t
        return Result(answered, reason, conf, matches, lines, box, marks["match"] - t0, timings)
