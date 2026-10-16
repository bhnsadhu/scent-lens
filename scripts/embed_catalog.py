"""Embed catalog images with a frozen backbone, in resumable chunks.

Usage: python scripts/embed_catalog.py --backbone clip [--limit N]
       python scripts/embed_catalog.py --backbone finetuned --ckpt DIR --out-name NAME [--subset FILE]

The image order is artifacts/catalog/embed_order.txt (one image_name per line,
written on first run from the inventory: decodable images in csv row order).
Chunks go to artifacts/emb/<backbone>/chunk_XXXXX.npy; existing chunks are
skipped, so a stopped run continues where it left off. When all chunks exist
they are concatenated into artifacts/emb/<backbone>/gallery.npy.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from scent_lens.catalog import ZipImages, load_catalog  # noqa: E402
from scent_lens.embed import load_encoder  # noqa: E402

RAW = ROOT / "data" / "raw"
ORDER = ROOT / "artifacts" / "catalog" / "embed_order.txt"
CHUNK = 512


def embed_order() -> list[str]:
    if ORDER.exists():
        return ORDER.read_text().split()
    df = load_catalog(RAW / "perfumes.csv")
    with (ROOT / "artifacts" / "inventory" / "images.jsonl").open() as f:
        ok = {r["image_name"] for r in map(json.loads, f) if r.get("ok")}
    names = [n for n in df["image_name"] if n in ok]
    ORDER.parent.mkdir(parents=True, exist_ok=True)
    ORDER.write_text("\n".join(names) + "\n")
    return names


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True, choices=["clip", "dinov2", "siglip2", "finetuned"])
    ap.add_argument("--ckpt", default=None, help="fine tuned checkpoint directory (with --backbone finetuned)")
    ap.add_argument("--out-name", default=None, help="output folder under artifacts/emb (default: backbone)")
    ap.add_argument("--subset", default=None, help="file of image names to embed instead of the full order")
    ap.add_argument("--limit", type=int, default=None, help="embed only the first N images (timing)")
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    names = embed_order()
    if args.subset:
        wanted = set(Path(args.subset).read_text().split())
        names = [n for n in names if n in wanted]
    if args.limit:
        names = names[: args.limit]
    out = ROOT / "artifacts" / "emb" / (args.out_name or args.backbone)
    out.mkdir(parents=True, exist_ok=True)
    store = ZipImages(RAW / "images.zip")
    if args.backbone == "finetuned":
        from scent_lens.finetune import load_finetuned

        enc = load_finetuned(args.ckpt)
    else:
        enc = load_encoder(args.backbone)
    print(f"{args.backbone} on {enc.device}: {len(names)} images", flush=True)

    n_chunks = (len(names) + CHUNK - 1) // CHUNK
    t0, done_now = time.time(), 0
    with ThreadPoolExecutor(max_workers=6) as pool:
        for c in range(n_chunks):
            path = out / f"chunk_{c:05d}.npy"
            if path.exists():
                continue
            chunk_names = names[c * CHUNK : (c + 1) * CHUNK]
            parts = []
            for b in range(0, len(chunk_names), args.batch):
                imgs = list(pool.map(lambda n: store.open(n, max_side=448), chunk_names[b : b + args.batch]))
                parts.append(enc.encode_images(imgs))
            np.save(path.with_suffix(".tmp.npy"), np.concatenate(parts).astype(np.float32))
            path.with_suffix(".tmp.npy").rename(path)
            done_now += len(chunk_names)
            rate = done_now / (time.time() - t0)
            left = (len(names) - (c + 1) * CHUNK) / max(rate, 1e-9)
            print(f"chunk {c + 1}/{n_chunks}  {rate:.1f} img/s  eta {max(left, 0) / 60:.1f} min", flush=True)

    if args.limit:
        print(f"timing run done: {done_now / max(time.time() - t0, 1e-9):.1f} img/s")
        return 0
    emb = np.concatenate([np.load(out / f"chunk_{c:05d}.npy") for c in range(n_chunks)])
    assert emb.shape[0] == len(names), (emb.shape, len(names))
    np.save(out / "gallery.npy", emb)
    (out / "gallery_names.txt").write_text("\n".join(names) + "\n")
    print(f"wrote gallery {emb.shape}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
