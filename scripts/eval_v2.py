"""Version 2 experiments, run after the one time test (not a preregistered result).

Each variant changes how a cropped query is scored. The calibrator and
threshold are refitted on development photos only, exactly as rules 5 and 6,
then applied to the test photos. Test numbers here were measured after the
test split had been seen once, so they are optimistic and are reported as
such (results/v2_results.md).

Variants:
  base      the version 1 final system (fine tuned SigLIP 2, cropped query)
  rotate    best product score over the crop at 0, 90 and 270 degrees
  ocr       rotate plus a label text bonus: image score + w * text score, where
            the text score is the share of a product's brand and name words
            read on the crop (any rotation); w is chosen on development photos
  guard     ocr, answering only when the label does not contradict the guess:
            nothing was read, or at least half the words read match the
            guessed product (GUARD_SHARE); the threshold is chosen with the
            guard applied, on development photos
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from scent_lens.assets import catalog, gallery_for, photo_embeddings, products  # noqa: E402
from scent_lens.calibration import (  # noqa: E402
    FEATURES,
    choose_threshold,
    fit_calibrator,
    out_of_fold_probs,
)
from scent_lens.evaluate import records_from_scores, select_pool  # noqa: E402
from scent_lens.metrics import brier, selective  # noqa: E402

SEED = 20261015
CKPT = ROOT / "artifacts" / "finetune" / "siglip2_lr1e-4_s1.0" / "epoch1"
V2 = ROOT / "artifacts" / "real" / "v2"
ROTATIONS = {90: Image.Transpose.ROTATE_90, 270: Image.Transpose.ROTATE_270}


def rotated_embeddings(photos: pd.DataFrame) -> dict[int, np.ndarray]:
    """Fine tuned embeddings of every cropped photo at 90 and 270 degrees, cached."""
    V2.mkdir(parents=True, exist_ok=True)
    out, todo = {}, []
    for deg in ROTATIONS:
        f = V2 / f"finetuned_crop_r{deg}.npy"
        ids = V2 / f"finetuned_crop_r{deg}_ids.txt"
        if f.exists() and ids.read_text().split() == photos.photo_id.tolist():
            out[deg] = np.load(f)
        else:
            todo.append(deg)
    if todo:
        from scent_lens.finetune import load_finetuned

        enc = load_finetuned(CKPT)
        imgs = [Image.open(ROOT / "data" / "real" / "processed_crop" / f"{p}.jpg").convert("RGB") for p in photos.photo_id]
        for deg in todo:
            rot = [im.transpose(ROTATIONS[deg]) for im in imgs]
            emb = np.concatenate([enc.encode_images(rot[i : i + 32]) for i in range(0, len(rot), 32)])
            np.save(V2 / f"finetuned_crop_r{deg}.npy", emb.astype(np.float32))
            (V2 / f"finetuned_crop_r{deg}_ids.txt").write_text("\n".join(photos.photo_id) + "\n")
            out[deg] = emb
    return out


def ocr_lines(photos: pd.DataFrame) -> dict[str, list[str]]:
    """Text read on each cropped photo at 0, 90 and 270 degrees, cached."""
    from scent_lens.ocr import read_text

    cache = V2 / "ocr.json"
    lines = json.loads(cache.read_text()) if cache.exists() else {}
    for p in photos.photo_id:
        if p in lines:
            continue
        img = Image.open(ROOT / "data" / "real" / "processed_crop" / f"{p}.jpg").convert("RGB")
        lines[p] = read_text(img) + [t for deg in ROTATIONS for t in read_text(img.transpose(ROTATIONS[deg]))]
    V2.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(lines, indent=1) + "\n")
    return lines


GUARD_SHARE = 0.5
WEIGHTS = [0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0]


def evaluate(name: str, rec: pd.DataFrame, manifest: pd.DataFrame, features: list[str] | None = None,
             guard: np.ndarray | None = None) -> dict:
    """Accuracy per pool and source, and abstention fitted on development photos."""
    features = features or list(FEATURES)
    pool = rec.bottle_id.map(manifest.set_index("bottle_id")["pool"])
    dev, test = rec[pool == "dev"].copy(), rec[pool == "test"].copy()
    dev["fold"] = dev.bottle_id.map(manifest.set_index("bottle_id")["fold"])
    X = lambda r: r[features].to_numpy(float)  # noqa: E731
    y_dev = dev.top1.to_numpy(int)
    oof = {k: out_of_fold_probs(X(dev), y_dev, dev.fold.to_numpy(), kind=k, seed=SEED) for k in ("logistic", "isotonic")}
    kind = "isotonic" if brier(oof["logistic"], y_dev) - brier(oof["isotonic"], y_dev) > 0.01 else "logistic"
    is_dev = (pool == "dev").to_numpy()
    g_dev = np.ones(len(dev), bool) if guard is None else guard[is_dev]
    g_test = np.ones(len(test), bool) if guard is None else guard[(pool == "test").to_numpy()]
    oof = {k: np.where(g_dev, v, -1.0) for k, v in oof.items()}  # guarded photos are never answered
    thr = choose_threshold(oof[kind], y_dev, 0.90)
    cal = fit_calibrator(X(dev), y_dev, kind, SEED)
    p_test = np.where(g_test, cal.predict(X(test)), -1.0)
    out = {"variant": name, "kind": kind, "threshold": None if not np.isfinite(thr) else float(thr)}
    for pname, r in (("dev", dev), ("test", test)):
        m = r[r.status == "matched"]
        ph = m.condition != "online"
        out[pname] = {"top1": float(m.top1.mean()), "top5": float(m.top5.mean()),
                      "phone_top1": float(m[ph].top1.mean()), "online_top1": float(m[~ph].top1.mean()),
                      "front_phone_top1": float(m[m.condition.isin(["clean", "cluttered"])].top1.mean())}
    if np.isfinite(thr):
        sa_dev, cov_dev = selective(oof[kind], y_dev, thr)
        sa, cov = selective(p_test, test.top1.to_numpy(), thr)
        absent = (test.status == "absent").to_numpy()
        out["dev"].update({"selective_accuracy": sa_dev, "coverage": cov_dev})
        out["test"].update({"selective_accuracy": sa, "coverage": cov,
                            "absent_answered": int((p_test[absent] >= thr).sum()), "absent_photos": int(absent.sum())})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="base,rotate,ocr,guard")
    args = ap.parse_args()
    gal = catalog()
    prods = products(gal)
    photos = pd.read_csv(ROOT / "data" / "real" / "photos.csv", dtype=str, keep_default_na=False)
    manifest = pd.read_csv(ROOT / "results" / "split_manifest.csv", dtype={"fold": "Int64"}, keep_default_na=False)
    photos = pd.concat([select_pool(photos, manifest, "dev"), select_pool(photos, manifest, "test")]).reset_index(drop=True)
    g = gallery_for("finetuned_crop", gal)
    base = g.scores(photo_embeddings("finetuned_crop", photos.photo_id.tolist()))
    results = []
    for v in args.variants.split(","):
        if v == "base":
            scores = base
        elif v in ("rotate", "ocr", "guard"):
            rot = rotated_embeddings(photos)
            scores = np.maximum.reduce([base] + [g.scores(e) for e in rot.values()])
        else:
            raise ValueError(v)
        extra = {}
        if v in ("ocr", "guard"):
            from scent_lens.ocr import TextMatcher

            tm = TextMatcher(prods, g.products)
            lines = ocr_lines(photos)
            text = np.stack([tm.text_score(lines[p]) for p in photos.photo_id])
            is_dev = (photos.bottle_id.map(manifest.set_index("bottle_id")["pool"]) == "dev").to_numpy()
            dev_top1 = {}
            for w in WEIGHTS:  # chosen on development photos only
                r = records_from_scores(g, prods, scores[is_dev] + w * text[is_dev], photos[is_dev].reset_index(drop=True))
                dev_top1[w] = float(r[r.status == "matched"].top1.mean())
            w = max(WEIGHTS, key=lambda k: (dev_top1[k], -k))
            scores = scores + w * text
            extra = {"weight": w, "dev_top1_by_weight": dev_top1,
                     "photos_with_text": int(sum(bool(lines[p]) for p in photos.photo_id))}
        rec = records_from_scores(g, prods, scores, photos)
        features, guard = list(FEATURES), None
        if v in ("ocr", "guard"):
            col = {p: i for i, p in enumerate(g.products)}
            rec["text1"] = text[np.arange(len(rec)), rec.pred1.map(col).to_numpy()]
            rec["text_agree"] = ((rec.text1 > 0) & (rec.text1 >= text.max(axis=1))).astype(float)
            rec["text_explained"] = [tm.explained(lines[p], col[q]) for p, q in zip(rec.photo_id, rec.pred1, strict=True)]
            if v == "ocr":  # the calibrator also sees the label signals
                features += ["text1", "text_agree", "text_explained"]
            else:
                from scent_lens.ocr import text_tokens

                read = np.array([bool(text_tokens(" ".join(lines[p]))) for p in rec.photo_id])
                guard = ~read | (rec.text_explained.to_numpy() >= GUARD_SHARE)
                extra["guard_share"] = GUARD_SHARE
        rec.to_csv(V2 / f"records_{v}.csv", index=False)
        results.append({**evaluate(v, rec, manifest, features, guard), **extra})
        r = results[-1]
        print(f"{v:8s} dev top1 {r['dev']['top1']:.3f} phone {r['dev']['phone_top1']:.3f} | test top1 {r['test']['top1']:.3f} "
              f"phone {r['test']['phone_top1']:.3f} front {r['test']['front_phone_top1']:.3f} | thr {r['threshold']} "
              f"test sel {r['test'].get('selective_accuracy')} cov {r['test'].get('coverage')}", flush=True)
    (ROOT / "results" / "v2_results.json").write_text(json.dumps(results, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
