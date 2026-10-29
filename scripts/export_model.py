"""Export the version 2 system for the demo and for use in other apps.

Writes artifacts/export/:
  model.pt             fine tuned SigLIP 2 vision tower with the LoRA weights
                       merged in, plus the projection head (float16)
  vision_config.json   the vision tower config, so loading needs no weight download
  encoder.json         input size and normalization
  gallery.npy          catalog image embeddings (float16), one row per image
  gallery_products.txt the product id of each row
  products.csv         names and notes shown with a match (no catalog images)
  settings.json        the kept version 2 settings (results/calibrator_v2.json)

The OWLv2 detector is not exported; it loads from the Hugging Face hub.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from scent_lens.assets import catalog, gallery_for, products  # noqa: E402
from scent_lens.embed import BACKBONES  # noqa: E402
from scent_lens.finetune import load_finetuned, merge_for_export  # noqa: E402

CKPT = ROOT / "artifacts" / "finetune" / "siglip2_lr1e-4_s1.0" / "epoch1"
OUT = ROOT / "artifacts" / "export"
PRODUCT_COLS = ["brand", "name", "family", "subfamily", "accords", "notes", "gender", "year"]


def main() -> int:
    from transformers import AutoConfig

    OUT.mkdir(parents=True, exist_ok=True)
    enc = load_finetuned(CKPT, device="cpu")
    model = merge_for_export(enc)
    torch.save({k: v.half() for k, v in model.state_dict().items()}, OUT / "model.pt")
    AutoConfig.from_pretrained(BACKBONES["siglip2"]).vision_config.to_json_file(OUT / "vision_config.json")
    (OUT / "encoder.json").write_text(json.dumps({"size": enc.size, "mean": list(enc.mean), "std": list(enc.std)}, indent=2) + "\n")

    gal = catalog()
    g = gallery_for("finetuned", gal)
    np.save(OUT / "gallery.npy", g.emb.astype(np.float16))
    rows = np.repeat(g.products, np.diff(np.append(g.starts, len(g.emb))))
    (OUT / "gallery_products.txt").write_text("\n".join(rows) + "\n")
    products(gal)[PRODUCT_COLS].to_csv(OUT / "products.csv")
    shutil.copy(ROOT / "results" / "calibrator_v2.json", OUT / "settings.json")
    for f in sorted(OUT.iterdir()):
        print(f"{f.name:22s} {f.stat().st_size / 1e6:8.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
