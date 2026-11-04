"""Assemble the Hugging Face Space folder at artifacts/space (nothing is uploaded here).

Contents: the demo app, the scent_lens modules it needs, the export
(artifacts/export: merged model, float16 gallery embeddings, product names),
the measured results page, and only the sample photos the user approved,
listed one per line in app/samples.txt (photo ids, read from
data/real/processed). No catalog image is included.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "space"
MODULES = ["__init__", "calibration", "crop", "embed", "finetune", "normalize", "ocr", "photos", "recognize", "retrieval"]
REQUIREMENTS = """torch==2.14.1
torchvision==0.29.1
transformers==5.20.0
sentencepiece==0.2.2
protobuf==7.36.2
numpy==2.4.6
pandas==3.0.6
scikit-learn==1.9.1
scipy==1.17.1
rapidfuzz==3.14.6
pillow==12.3.0
pillow-heif==1.8.1
rapidocr_onnxruntime==1.4.4
"""
README = """---
title: ScentLens
emoji: 🔎
colorFrom: pink
colorTo: gray
sdk: gradio
sdk_version: 6.30.0
app_file: app.py
pinned: false
license: mit
short_description: Identify a fragrance from a photo of its bottle
---

ScentLens identifies a fragrance from a photo of its bottle across 24,642 catalog fragrances and says "not sure" when it cannot tell. Code, method and measured results: https://github.com/{github}/scent-lens

This Space holds the merged fine tuned model, float16 catalog image embeddings and product names derived from the doevent/perfume dataset (rights to its catalog images are unverified; no catalog image is included), and sample photos shared with permission. A free Space sleeps when idle, so the first request can be slow.
"""


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    github = args[0] if args else "<user>"
    # --code-only: everything that lives in git, without the export and sample photos (which do not).
    # The Modal deploy in GitHub Actions uses it; those two folders live in a Modal Volume there.
    code_only = "--code-only" in sys.argv
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "src" / "scent_lens").mkdir(parents=True)
    for m in MODULES:
        shutil.copy(ROOT / "src" / "scent_lens" / f"{m}.py", OUT / "src" / "scent_lens" / f"{m}.py")
    shutil.copy(ROOT / "app" / "app.py", OUT / "app.py")
    shutil.copy(ROOT / "app" / "results.md", OUT / "results.md")
    shutil.copy(ROOT / "app" / "mark.svg", OUT / "mark.svg")
    approved = ROOT / "app" / "samples.txt"
    ids = [s for s in approved.read_text().split() if s] if approved.exists() and not code_only else []
    if not code_only:
        shutil.copytree(ROOT / "artifacts" / "export", OUT / "export")
        (OUT / "samples").mkdir()
    for i, pid in enumerate(ids, 1):
        shutil.copy(ROOT / "data" / "real" / "processed" / f"{pid}.jpg", OUT / "samples" / f"sample_{i}.jpg")
    (OUT / "requirements.txt").write_text(REQUIREMENTS)
    (OUT / "packages.txt").write_text("libgl1\nlibglib2.0-0\n")  # OpenCV, used by RapidOCR
    (OUT / "README.md").write_text(README.replace("{github}", github))
    size = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file())
    print(f"{OUT}: {size / 1e6:.0f} MB, {len(ids)} sample photos")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
