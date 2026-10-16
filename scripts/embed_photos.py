"""Embed processed real photos with one or more backbones.

Writes artifacts/real/<backbone>.npy in data/real/photos.csv order, plus the
photo ids alongside. Cheap (a few hundred photos), so it simply recomputes.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from scent_lens.embed import load_encoder  # noqa: E402

OUT = ROOT / "artifacts" / "real"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbones", default="dinov2,siglip2,clip")
    ap.add_argument("--finetuned", default=None, help="path to a fine tuned checkpoint directory")
    args = ap.parse_args()
    photos = pd.read_csv(ROOT / "data" / "real" / "photos.csv", dtype=str)
    imgs = [Image.open(ROOT / p).convert("RGB") for p in photos.path]
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(b, None) for b in args.backbones.split(",") if b]
    if args.finetuned:
        jobs.append(("finetuned", args.finetuned))
    for name, ckpt in jobs:
        if ckpt:
            from scent_lens.finetune import load_finetuned

            enc = load_finetuned(ckpt)
        else:
            enc = load_encoder(name)
        emb = np.concatenate([enc.encode_images(imgs[i : i + 32]) for i in range(0, len(imgs), 32)])
        np.save(OUT / f"{name}.npy", emb.astype(np.float32))
        (OUT / f"{name}_ids.txt").write_text("\n".join(photos.photo_id) + "\n")
        print(f"{name}: {emb.shape}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
