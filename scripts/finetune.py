"""Fine tune a backbone contrastively on augmented catalog images.

Trains on a fixed sample of catalog products that excludes every product in
the user's bottle list (data/real/matches.csv, matched bottles) and every
product sharing a line key with one, so evaluation is on unseen products.
Never touches real photos.

Resumable: a checkpoint (trainable weights, optimizer, scheduler, step) is
written every --ckpt-every steps to artifacts/finetune/<run>/last; rerunning
the same command continues from it. Batch order is fixed per epoch by the seed.
At each epoch end the weights are copied to artifacts/finetune/<run>/epoch<N>.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from scent_lens.augment import random_mix  # noqa: E402
from scent_lens.catalog import ZipImages  # noqa: E402
from scent_lens.embed import Encoder, pick_device  # noqa: E402
from scent_lens.finetune import brand_batches, build_model, info_nce, save_checkpoint  # noqa: E402


def training_products(gal: pd.DataFrame, n_products: int, seed: int) -> tuple[pd.DataFrame, dict]:
    matches = pd.read_csv(ROOT / "data" / "real" / "matches.csv", dtype=str, keep_default_na=False)
    held_ids = set(matches.loc[matches.status == "matched", "product_id"])
    held_lines = set(gal.loc[gal.product_id.isin(held_ids), "line_key"])
    pool = gal[~gal.product_id.isin(held_ids) & ~gal.line_key.isin(held_lines)]
    prods = np.sort(pool.product_id.unique())
    rng = np.random.default_rng(seed)
    pick = set(rng.choice(prods, size=min(n_products, len(prods)), replace=False))
    train = pool[pool.product_id.isin(pick)]
    info = {
        "held_out_products": len(held_ids),
        "held_out_lines": len(held_lines),
        "excluded_gallery_images": int(len(gal) - len(pool)),
        "train_products": int(train.product_id.nunique()),
        "train_images": int(len(train)),
    }
    return train, info


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True, choices=["dinov2", "siglip2"])
    ap.add_argument("--run", required=True)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--strength", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--products", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--lora-rank", type=int, default=8)
    ap.add_argument("--last-blocks", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=20261015)
    ap.add_argument("--ckpt-every", type=int, default=50)
    ap.add_argument("--max-steps", type=int, default=None, help="stop early (timing runs)")
    args = ap.parse_args()

    out = ROOT / "artifacts" / "finetune" / args.run
    out.mkdir(parents=True, exist_ok=True)
    config = vars(args) | {"device": pick_device()}
    (out / "config.json").write_text(json.dumps(config, indent=2))
    torch.manual_seed(args.seed)

    gal = pd.read_csv(ROOT / "artifacts" / "catalog" / "gallery.csv", dtype=str, keep_default_na=False)
    train, info = training_products(gal, args.products, args.seed)
    (out / "data_info.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info), flush=True)
    by_product = train.groupby("product_id")["image_name"].apply(list)
    prod_ids = by_product.index.tolist()
    brands = train.drop_duplicates("product_id").set_index("product_id").loc[prod_ids, "brand"].str.lower().tolist()

    device = pick_device()
    model, mean, std = build_model(args.backbone, args.lora_rank, args.last_blocks)
    model.to(device)
    prep = Encoder("finetuned", model, 224, mean, std, device)  # used for preprocessing only
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    steps_per_epoch = len(brand_batches(prod_ids, brands, args.batch, np.random.default_rng(args.seed)))
    total = steps_per_epoch * args.epochs
    warm = min(50, total // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(warm, 1)) * 0.5 * (1 + np.cos(np.pi * min(s, total) / total))
    )
    step = 0
    last = out / "last"
    if (last / "state.pt").exists():
        state = torch.load(last / "state.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(torch.load(last / "trainable.pt", map_location="cpu"), strict=False)
        opt.load_state_dict(state["opt"])
        sched.load_state_dict(state["sched"])
        step = state["step"]
        print(f"resumed at step {step}", flush=True)

    store = ZipImages(ROOT / "data" / "raw" / "images.zip")
    log = (out / "train_log.jsonl").open("a")

    def make_pair(args_):
        pid, sample_seed = args_
        rng = np.random.default_rng(sample_seed)
        imgs = by_product[pid]
        pos_name = imgs[rng.integers(len(imgs))]
        q_name = imgs[rng.integers(len(imgs))]
        pos = store.open(pos_name, max_side=448)
        q = random_mix(store.open(q_name, max_side=448), rng, args.strength)
        return q, pos

    model.train()
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=6) as pool:
        for epoch in range(args.epochs):
            batches = brand_batches(prod_ids, brands, args.batch, np.random.default_rng(args.seed + epoch))
            for b, idx in enumerate(batches):
                global_step = epoch * steps_per_epoch + b
                if global_step < step:
                    continue
                pids = [prod_ids[i] for i in idx]
                pairs = list(pool.map(make_pair, [(p, args.seed * 1000 + global_step * 997 + j) for j, p in enumerate(pids)]))
                q = prep.preprocess([p[0] for p in pairs]).to(device)
                pos = prep.preprocess([p[1] for p in pairs]).to(device)
                loss = info_nce(model(q), model(pos), pids, args.temperature)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sched.step()
                step = global_step + 1
                log.write(json.dumps({"step": step, "epoch": epoch, "loss": float(loss), "lr": sched.get_last_lr()[0]}) + "\n")
                if step % 10 == 0:
                    log.flush()
                    print(f"step {step}/{total} loss {float(loss):.3f} {time.time() - t0:.0f}s", flush=True)
                if step % args.ckpt_every == 0 or step == (epoch + 1) * steps_per_epoch:
                    meta = {"backbone": args.backbone, "lora_rank": args.lora_rank, "last_blocks": args.last_blocks, "step": step}
                    save_checkpoint(model, last, meta)
                    torch.save({"opt": opt.state_dict(), "sched": sched.state_dict(), "step": step}, last / "state.pt")
                if args.max_steps and step >= args.max_steps:
                    print(f"timing: {step} steps in {time.time() - t0:.0f}s, {steps_per_epoch} steps per epoch", flush=True)
                    return 0
            ep_dir = out / f"epoch{epoch + 1}"
            if not ep_dir.exists():
                shutil.copytree(last, ep_dir, ignore=shutil.ignore_patterns("state.pt"))
            print(f"epoch {epoch + 1} done at step {step}", flush=True)
    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
