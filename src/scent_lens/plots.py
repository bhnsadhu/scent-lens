"""Shared chart style: reference categorical palette in fixed order, neutral ink."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
# Fixed slot order; a method keeps its color in every chart.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
METHOD_COLORS = {
    "dinov2": SERIES[0],
    "siglip2": SERIES[1],
    "clip": SERIES[2],
    "siglip2_text": SERIES[3],
    "clip_text": SERIES[4],
    "finetuned": SERIES[6],
}
METHOD_LABELS = {
    "dinov2": "DINOv2 base (image)",
    "siglip2": "SigLIP 2 base (image)",
    "clip": "CLIP ViT B 32 (image)",
    "siglip2_text": "SigLIP 2 (text prompt)",
    "clip_text": "CLIP (text prompt)",
    "random": "Random",
    "finetuned": "Fine tuned",
}


def style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.titleweight": "bold",
            "axes.titlesize": 12,
            "axes.titlelocation": "left",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "text.color": INK,
            "legend.frameon": False,
            "legend.labelcolor": INK_2,
            "font.size": 10,
            "lines.linewidth": 2,
            "lines.markersize": 6,
        }
    )
