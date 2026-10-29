"""Version 2 experiments, run after the one time test (not a preregistered result).

Each variant changes how a cropped query is scored. The calibrator and
threshold are refitted on development photos only, exactly as rules 5 and 6,
then applied unchanged to the test photos and to the web set. Test numbers
were measured after the test split had been seen once, so they are optimistic.
The web set (data/real/web, Wikimedia Commons photos of products that were
never trained on, tested, or used for any choice; scripts/ingest_web.py) is
the fresh check. Writes results/v2_results.json and results/v2_results.md.

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
  ocr_full, guard_full
            the same, with the label read on the crop cut from the full
            resolution original (phone photos are about 4 times larger than
            the 1024 pixel processed copy)
  label, label_full
            guard, and also always answering on a clear label match: every
            brand and name word of the guessed product was read, at least
            half the words read belong to it, and no other catalog product
            has all its words read (a label reading only "Miss Dior" fits
            dozens of products); the threshold is chosen with these answers
            included
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
from scent_lens.metrics import bootstrap_by_group, brier, selective  # noqa: E402

SEED = 20261015
CKPT = ROOT / "artifacts" / "finetune" / "siglip2_lr1e-4_s1.0" / "epoch1"
V2 = ROOT / "artifacts" / "real" / "v2"
WEB = ROOT / "data" / "real" / "web"
ROTATIONS = {90: Image.Transpose.ROTATE_90, 270: Image.Transpose.ROTATE_270}
GUARD_SHARE = 0.5
WEIGHTS = [0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0]
NAMES = {"base": "Version 1 final system", "rotate": "+ rotations (0, 90, 270)",
         "ocr": "+ label text bonus (w chosen on dev)", "guard": "+ label guard on answering",
         "ocr_full": "+ label read at full resolution", "guard_full": "+ full resolution label and guard",
         "label": "+ answer on a clear label match", "label_full": "+ clear label match, full resolution"}


def query_photos(manifest: pd.DataFrame) -> pd.DataFrame:
    """Development, test and web photos with pool, fold and the path of the cropped query."""
    photos = pd.read_csv(ROOT / "data" / "real" / "photos.csv", dtype=str, keep_default_na=False)
    user = pd.concat([select_pool(photos, manifest, p).assign(pool=p) for p in ("dev", "test")])
    user["crop"] = [ROOT / "data" / "real" / "processed_crop" / f"{p}.jpg" for p in user.photo_id]
    web = pd.read_csv(WEB / "photos.csv", dtype=str, keep_default_na=False)
    web = web[web.use == "test"].assign(pool="web", fold=pd.NA)
    web["crop"] = [WEB / "crop" / f"{p}.jpg" for p in web.photo_id]
    missing = [r for r in web.itertuples() if not r.crop.exists()]
    if missing:
        from scent_lens.crop import BottleDetector, apply_crop
        from scent_lens.photos import save_clean

        (WEB / "crop").mkdir(exist_ok=True)
        det = BottleDetector()
        for r in missing:
            img = Image.open(ROOT / r.path).convert("RGB")
            save_clean(apply_crop(img, det.detect(img)[0]), r.crop)
    cols = ["photo_id", "bottle_id", "condition", "status", "product_id", "pool", "fold", "crop"]
    return pd.concat([user[cols], web[cols]]).reset_index(drop=True)


def crop_embeddings(photos: pd.DataFrame, deg: int, enc_holder: dict) -> np.ndarray:
    """Fine tuned embeddings of the cropped queries rotated by deg, cached per photo.

    At 0 degrees the development and test photos reuse the version 1 embeddings
    (artifacts/real/finetuned_crop.npy), so the base variant reproduces it exactly.
    """
    cache = V2 / f"emb_r{deg}.npz"
    have = dict(np.load(cache, allow_pickle=False)) if cache.exists() else {}
    if deg == 0:
        user = photos[photos.pool != "web"].photo_id.tolist()
        have.update(zip(user, photo_embeddings("finetuned_crop", user), strict=True))
    todo = [r for r in photos.itertuples() if r.photo_id not in have]
    if todo:
        if "enc" not in enc_holder:
            from scent_lens.finetune import load_finetuned

            enc_holder["enc"] = load_finetuned(CKPT)
        imgs = [Image.open(r.crop).convert("RGB") for r in todo]
        if deg:
            imgs = [im.transpose(ROTATIONS[deg]) for im in imgs]
        emb = np.concatenate([enc_holder["enc"].encode_images(imgs[i : i + 32]) for i in range(0, len(imgs), 32)])
        have.update({r.photo_id: e for r, e in zip(todo, emb, strict=True)})
        V2.mkdir(parents=True, exist_ok=True)
        np.savez(cache, **have)
    return np.stack([have[p] for p in photos.photo_id]).astype(np.float32)


def full_res_crop(photo_id: str, crop_path: Path, boxes: pd.DataFrame) -> Image.Image:
    """The query crop cut from the full resolution original; web photos only exist at crop size."""
    from scent_lens.photos import clean_image, register_heif

    register_heif()
    raw = sorted((ROOT / "data" / "real" / "raw").glob(f"{photo_id}.*"))
    if "__online__" in photo_id:
        bottle, _, n = photo_id.split("__")
        pack = ROOT / "data" / "real" / "online_refs" / "scentlens_audited"
        entry = next(b for b in json.loads((pack / "photo_manifest.json").read_text())["bottles"] if b["id"] == bottle)
        raw = [pack / entry["photos"][int(n) - 1]["file"]]
    if not raw or photo_id not in boxes.index:
        return Image.open(crop_path).convert("RGB")
    big = clean_image(raw[0], max_side=100_000)  # orientation and transparency handled, size kept
    small = Image.open(ROOT / "data" / "real" / "processed" / f"{photo_id}.jpg")
    k = big.size[0] / small.size[0]
    b = boxes.loc[photo_id]
    return big.crop(tuple(round(k * v) for v in (b.x0, b.y0, b.x1, b.y1)))


def ocr_lines(photos: pd.DataFrame, full: bool = False) -> dict[str, list[str]]:
    """Text read on each cropped query at 0, 90 and 270 degrees, cached."""
    from scent_lens.ocr import read_text

    cache = V2 / ("ocr_full.json" if full else "ocr.json")
    lines = json.loads(cache.read_text()) if cache.exists() else {}
    boxes = pd.read_csv(ROOT / "artifacts" / "real" / "crops.csv").set_index("photo_id")
    for r in photos.itertuples():
        if r.photo_id not in lines:
            img = full_res_crop(r.photo_id, r.crop, boxes) if full else Image.open(r.crop).convert("RGB")
            lines[r.photo_id] = read_text(img) + [t for deg in ROTATIONS for t in read_text(img.transpose(ROTATIONS[deg]))]
    V2.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(lines, indent=1) + "\n")
    return lines


def interval(groups, values) -> dict:
    point, lo, hi = bootstrap_by_group(groups, values, n_boot=2000, seed=SEED)
    return {"value": point, "lo": lo, "hi": hi}


def evaluate(name: str, rec: pd.DataFrame, features: list[str], guard: np.ndarray | None,
             strong: np.ndarray | None = None) -> dict:
    """Accuracy per pool, and abstention fitted on development photos then applied unchanged.

    guard: photos that may be answered at all. strong: photos always answered
    (a clear label match), scored above any calibrated probability.
    """
    guard = np.ones(len(rec), bool) if guard is None else guard
    strong = np.zeros(len(rec), bool) if strong is None else strong
    is_dev = (rec.pool == "dev").to_numpy()
    dev = rec[is_dev]
    X = lambda r: r[features].to_numpy(float)  # noqa: E731
    y_dev = dev.top1.to_numpy(int)
    oof = {k: out_of_fold_probs(X(dev), y_dev, dev.fold.astype(int).to_numpy(), kind=k, seed=SEED) for k in ("logistic", "isotonic")}
    kind = "isotonic" if brier(oof["logistic"], y_dev) - brier(oof["isotonic"], y_dev) > 0.01 else "logistic"
    p_dev = np.where(strong[is_dev], 2.0, np.where(guard[is_dev], oof[kind], -1.0))  # guarded photos are never answered
    thr = choose_threshold(p_dev, y_dev, 0.90)
    cal = fit_calibrator(X(dev), y_dev, kind, SEED)
    out = {"variant": name, "kind": kind, "threshold": None if not np.isfinite(thr) else float(thr)}
    for pool in ("dev", "test", "web"):
        sel = (rec.pool == pool).to_numpy()
        r = rec[sel]
        m = r[r.status == "matched"]
        ph = ~m.condition.isin(["online", "web"])
        d = {"photos": int(len(r)), "top1": float(m.top1.mean()), "top5": float(m.top5.mean())}
        if pool == "web":
            d["top1_interval"] = interval(m.bottle_id, m.top1)
        else:
            d.update({"phone_top1": float(m[ph].top1.mean()), "online_top1": float(m[~ph].top1.mean()),
                      "front_phone_top1": float(m[m.condition.isin(["clean", "cluttered"])].top1.mean())})
        if np.isfinite(thr):
            p = p_dev if pool == "dev" else np.where(strong[sel], 2.0, np.where(guard[sel], cal.predict(X(r)), -1.0))
            sa, cov = selective(p, r.top1.to_numpy(), thr)
            absent = (r.status == "absent").to_numpy()
            d.update({"selective_accuracy": sa, "coverage": cov, "answered": int((p >= thr).sum()),
                      "absent_answered": int((p[absent] >= thr).sum()), "absent_photos": int(absent.sum())})
        out[pool] = d
    return out


def report(results: list[dict], n_web_products: int) -> str:
    f = lambda x: "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.1f}"  # noqa: E731
    L = ["# Version 2 experiments (after the test)", "",
         "Run after the one time test of version 1 (results/test_results.md). Every choice (rotation set, text weight, guard share, calibrator, threshold) was made on development photos only; the calibrator and threshold were refitted on development photos exactly as rules 5 and 6 and applied unchanged elsewhere.", "",
         "* Test: the version 1 test photos. These bottles were seen before the changes were made, so these columns are optimistic.",
         f"* Web: a fresh set of {results[0]['web']['photos']} Wikimedia Commons photos of {n_web_products} catalog products that were never in the fine tune training sample, the bottle list, or any choice (sources and licenses in results/web_sources.csv). Most are product style shots, so they are easier than phone photos; none has a not in catalog bottle, so abstention there only measures wrong answers on matched products.", "",
         "| Variant | Dev top 1 | Test top 1 | Test phone top 1 | Test front phone top 1 | Web top 1 [95% interval] | Web top 5 |", "|---|---|---|---|---|---|---|"]
    for r in results:
        t, w = r["test"], r["web"]
        wi = w["top1_interval"]
        L.append(f"| {NAMES[r['variant']]} | {f(r['dev']['top1'])} | {f(t['top1'])} | {f(t['phone_top1'])} | {f(t['front_phone_top1'])} | "
                 f"{f(wi['value'])} [{f(wi['lo'])}, {f(wi['hi'])}] | {f(w['top5'])} |")
    L += ["", "## Answering (\"not sure\" otherwise)", "",
          "| Variant | Threshold | Dev answered, accuracy | Test answered, accuracy | Test absent photos answered | Web answered, accuracy |", "|---|---|---|---|---|---|"]
    for r in results:
        if r["threshold"] is None:
            L.append(f"| {NAMES[r['variant']]} | none qualifies | | | | |")
            continue
        d, t, w = r["dev"], r["test"], r["web"]
        L.append(f"| {NAMES[r['variant']]} | {r['threshold']:.3f} | {f(d['coverage'])}%, {f(d['selective_accuracy'])}% | "
                 f"{f(t['coverage'])}%, {f(t['selective_accuracy'])}% | {t['absent_answered']} of {t['absent_photos']} | "
                 f"{f(w['coverage'])}%, {f(w['selective_accuracy'])}% |")
    o = next((r for r in results if "weight" in r), None)
    if o:
        L += ["", f"Text weight chosen on development photos: {o['weight']} (development top 1 by weight: "
              + ", ".join(f"{k} {100 * v:.1f}" for k, v in o["dev_top1_by_weight"].items()) + ")."]
    L += ["", "With the text bonus, the three feature calibrator ranks one confident error early (Victoria's Secret Vanilla Orchid Sandalwood, not in the catalog, read as Victoria's Secret Vanilla), and adding label features to the calibrator did not fix it on 101 development photos. The guard does: it answers only when nothing was read or at least half the words read match the guessed product. Both guard shares tried (0.34 and 0.5) gave the same results. Angled (mostly rear views) and partial photos remain the weak cases."]
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="base,rotate,ocr,guard,ocr_full,guard_full,label,label_full")
    args = ap.parse_args()
    gal = catalog()
    prods = products(gal)
    manifest = pd.read_csv(ROOT / "results" / "split_manifest.csv", dtype={"fold": "Int64"}, keep_default_na=False)
    photos = query_photos(manifest)
    g = gallery_for("finetuned_crop", gal)
    enc: dict = {}
    base = g.scores(crop_embeddings(photos, 0, enc))
    is_dev = (photos.pool == "dev").to_numpy()
    results = []
    for v in args.variants.split(","):
        if v == "base":
            scores = base
        elif v in ("rotate", "ocr", "guard", "ocr_full", "guard_full", "label", "label_full"):
            scores = np.maximum.reduce([base] + [g.scores(crop_embeddings(photos, d, enc)) for d in ROTATIONS])
        else:
            raise ValueError(v)
        extra = {}
        if v.startswith(("ocr", "guard", "label")):
            from scent_lens.ocr import TextMatcher

            tm = TextMatcher(prods, g.products)
            lines = ocr_lines(photos, full=v.endswith("_full"))
            text = np.stack([tm.text_score(lines[p]) for p in photos.photo_id])
            dev_top1 = {}
            for w in WEIGHTS:  # chosen on development photos only
                r = records_from_scores(g, prods, scores[is_dev] + w * text[is_dev], photos[is_dev].reset_index(drop=True))
                dev_top1[w] = float(r[r.status == "matched"].top1.mean())
            w = max(WEIGHTS, key=lambda k: (dev_top1[k], -k))
            scores = scores + w * text
            extra = {"weight": w, "dev_top1_by_weight": dev_top1}
        rec = records_from_scores(g, prods, scores, photos)
        rec["pool"], rec["fold"] = photos.pool.to_numpy(), photos.fold.to_numpy()
        features, guard, strong = list(FEATURES), None, None
        if v.startswith(("ocr", "guard", "label")):
            from scent_lens.ocr import text_tokens

            col = {p: i for i, p in enumerate(g.products)}
            rec["text1"] = text[np.arange(len(rec)), rec.pred1.map(col).to_numpy()]
            rec["text_agree"] = ((rec.text1 > 0) & (rec.text1 >= text.max(axis=1))).astype(float)
            rec["text_explained"] = [tm.explained(lines[p], col[q]) for p, q in zip(rec.photo_id, rec.pred1, strict=True)]
            if v.startswith("ocr"):  # the calibrator also sees the label signals
                features += ["text1", "text_agree", "text_explained"]
            else:
                read = np.array([bool(text_tokens(" ".join(lines[p]))) for p in rec.photo_id])
                guard = ~read | (rec.text_explained.to_numpy() >= GUARD_SHARE)
                extra["guard_share"] = GUARD_SHARE
                if v.startswith("label"):
                    unique = (text >= 1.0).sum(axis=1) == 1
                    strong = ((rec.text1 >= 1.0) & (rec.text_explained >= GUARD_SHARE)).to_numpy() & unique
        rec.to_csv(V2 / f"records_{v}.csv", index=False)
        results.append({**evaluate(v, rec, features, guard, strong), **extra})
        r = results[-1]
        print(f"{v:7s} dev {r['dev']['top1']:.3f} | test {r['test']['top1']:.3f} phone {r['test']['phone_top1']:.3f} | "
              f"web {r['web']['top1']:.3f} | thr {r['threshold']} | test sel {r['test'].get('selective_accuracy')} "
              f"cov {r['test'].get('coverage')} | web sel {r['web'].get('selective_accuracy')} cov {r['web'].get('coverage')}", flush=True)
    (ROOT / "results" / "v2_results.json").write_text(json.dumps(results, indent=2, default=float) + "\n")
    n_web = int(photos[photos.pool == "web"].bottle_id.nunique())
    (ROOT / "results" / "v2_results.md").write_text(report(results, n_web))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
