"""Score fine tune checkpoints on the development pool with a proxy gallery.

The proxy gallery is every catalog image of the products of development
bottles plus all images of 5,000 other products drawn with a fixed seed.
Each checkpoint and the frozen backbone are scored on the same proxy gallery
and the same development photos; one line per checkpoint goes to
results/runs.jsonl (config, seed, git sha, metrics). Test photos are never
loaded.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from scent_lens.assets import catalog, products  # noqa: E402
from scent_lens.evaluate import by_condition, photo_records, select_pool, summarize  # noqa: E402
from scent_lens.lock import git_sha  # noqa: E402
from scent_lens.retrieval import Gallery  # noqa: E402

SEED = 20261015
N_DISTRACTORS = 5000
PROXY = ROOT / "artifacts" / "finetune" / "proxy_names.txt"


def dev_photos() -> pd.DataFrame:
    photos = pd.read_csv(ROOT / "data" / "real" / "photos.csv", dtype=str, keep_default_na=False)
    manifest = pd.read_csv(ROOT / "results" / "split_manifest.csv", dtype={"fold": "Int64"}, keep_default_na=False)
    return select_pool(photos, manifest, "dev").reset_index(drop=True)


def proxy_gallery(gal: pd.DataFrame, dev: pd.DataFrame) -> pd.DataFrame:
    dev_products = set(dev.loc[dev.status == "matched", "product_id"])
    others = np.sort(gal.loc[~gal.product_id.isin(dev_products), "product_id"].unique())
    rng = np.random.default_rng(SEED)
    pick = set(rng.choice(others, size=min(N_DISTRACTORS, len(others)), replace=False)) | dev_products
    sub = gal[gal.product_id.isin(pick)]
    if not PROXY.exists():
        PROXY.parent.mkdir(parents=True, exist_ok=True)
        PROXY.write_text("\n".join(sub.image_name) + "\n")
    return sub


def score(name: str, gallery_emb: np.ndarray, sub: pd.DataFrame, photo_emb: np.ndarray, dev: pd.DataFrame, prods) -> dict:
    g = Gallery(gallery_emb, sub.product_id.tolist())
    rec = photo_records(g, prods, photo_emb, dev)
    return {"overall": summarize(rec, n_boot=500, seed=SEED), "by_condition": by_condition(rec, n_boot=200, seed=SEED)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True, help="checkpoint dirs, e.g. artifacts/finetune/r1/epoch1")
    ap.add_argument("--frozen", required=True, choices=["dinov2", "siglip2"])
    args = ap.parse_args()

    gal = catalog()
    prods = products(gal)
    dev = dev_photos()
    sub = proxy_gallery(gal, dev)
    imgs = [Image.open(ROOT / p).convert("RGB") for p in dev.path]
    sha = git_sha(ROOT)
    runs = ROOT / "results" / "runs.jsonl"

    # frozen reference on the same proxy gallery
    full = np.load(ROOT / "artifacts" / "emb" / args.frozen / "gallery.npy")
    from scent_lens.assets import photo_embeddings

    frozen = score(args.frozen, full[sub.emb_index.to_numpy()], sub, photo_embeddings(args.frozen, dev.photo_id.tolist()), dev, prods)
    line = {"run": f"frozen_{args.frozen}", "checkpoint": None, "git_sha": sha, "seed": SEED,
            "proxy_gallery_products": int(sub.product_id.nunique()), "dev_photos": int(len(dev)), "metrics": frozen,
            "time_utc": datetime.now(UTC).isoformat(timespec="seconds")}
    with runs.open("a") as f:
        f.write(json.dumps(line) + "\n")
    print(f"frozen {args.frozen}: top1 {frozen['overall']['top1']['value']:.3f}", flush=True)

    from scent_lens.finetune import load_finetuned

    for ck in args.ckpts:
        ck = Path(ck)
        out_name = f"ft_{ck.parent.name}_{ck.name}_proxy"
        emb_dir = ROOT / "artifacts" / "emb" / out_name
        if not (emb_dir / "gallery.npy").exists():
            subprocess.run([sys.executable, str(ROOT / "scripts" / "embed_catalog.py"), "--backbone", "finetuned",
                            "--ckpt", str(ck), "--out-name", out_name, "--subset", str(PROXY)], check=True)
        names = (emb_dir / "gallery_names.txt").read_text().split()
        pos = {n: i for i, n in enumerate(names)}
        g_emb = np.load(emb_dir / "gallery.npy")[[pos[n] for n in sub.image_name]]
        enc = load_finetuned(ck)
        p_emb = np.concatenate([enc.encode_images(imgs[i : i + 32]) for i in range(0, len(imgs), 32)])
        m = score("finetuned", g_emb, sub, p_emb, dev, prods)
        config = json.loads((ck.parent / "config.json").read_text())
        line = {"run": ck.parent.name, "checkpoint": str(ck.relative_to(ROOT)), "config": config, "git_sha": sha,
                "seed": config["seed"], "proxy_gallery_products": int(sub.product_id.nunique()),
                "dev_photos": int(len(dev)), "metrics": m, "time_utc": datetime.now(UTC).isoformat(timespec="seconds")}
        with runs.open("a") as f:
            f.write(json.dumps(line) + "\n")
        print(f"{ck}: top1 {m['overall']['top1']['value']:.3f} (frozen {frozen['overall']['top1']['value']:.3f})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
