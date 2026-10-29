"""Query crop: find the bottle with a zero shot detector and cut the photo to it.

Fixed in the preregistration note of 2026-10-29: OWLv2 base with the query
"a perfume bottle", the best box widened by 15% per side, and the full image
when no box scores at least 0.1.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from PIL import Image

MODEL = "google/owlv2-base-patch16-ensemble"
QUERY = "a perfume bottle"
MIN_SCORE = 0.1
PAD = 0.15


def padded_box(box: Sequence[float], size: tuple[int, int], pad: float = PAD) -> tuple[int, int, int, int]:
    """Widen (x0, y0, x1, y1) by pad times its width and height on every side, clipped to size (w, h)."""
    x0, y0, x1, y1 = (float(v) for v in box)
    w, h = size
    dx, dy = pad * (x1 - x0), pad * (y1 - y0)
    return (
        max(0, round(x0 - dx)),
        max(0, round(y0 - dy)),
        min(w, round(x1 + dx)),
        min(h, round(y1 + dy)),
    )


def choose_box(
    boxes: Sequence[Sequence[float]], scores: Sequence[float], size: tuple[int, int],
    min_score: float = MIN_SCORE, pad: float = PAD,
) -> tuple[int, int, int, int] | None:
    """The best box, padded; None when nothing scores min_score (the caller then uses the full image)."""
    s = np.asarray(scores, dtype=np.float64)
    if s.size == 0 or s.max() < min_score:
        return None
    box = padded_box(boxes[int(s.argmax())], size, pad)
    return box if box[2] > box[0] and box[3] > box[1] else None


class BottleDetector:
    def __init__(self, device: str | None = None):
        import torch
        from transformers import Owlv2ForObjectDetection, Owlv2Processor

        self.torch = torch
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.processor = Owlv2Processor.from_pretrained(MODEL)
        self.model = Owlv2ForObjectDetection.from_pretrained(MODEL).to(self.device).eval()

    def detect(self, img: Image.Image) -> tuple[tuple[int, int, int, int] | None, float]:
        """Padded best box (or None) and its score."""
        inputs = self.processor(text=[[QUERY]], images=img, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            out = self.model(**inputs)
        side = max(img.size)  # OWLv2 pads to a square at the bottom and right, so boxes are in that frame
        res = self.processor.post_process_grounded_object_detection(
            out, threshold=0.0, target_sizes=self.torch.tensor([[side, side]])
        )[0]
        boxes, scores = res["boxes"].cpu().numpy(), res["scores"].cpu().numpy()
        best = float(scores.max()) if scores.size else 0.0
        return choose_box(boxes, scores, img.size), best


def apply_crop(img: Image.Image, box: tuple[int, int, int, int] | None) -> Image.Image:
    return img if box is None else img.crop(box)
