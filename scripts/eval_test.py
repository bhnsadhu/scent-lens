"""The one time test evaluation (preregistration analysis plan, steps 3 and 4).

Refuses to run if results/test_lock.json exists or the working tree has
uncommitted changes; the lock is written before any test photo is scored and
there is no override. --dry-run runs the same code on the development pool,
writes to artifacts/dryrun, and never touches the lock (used to check this
script before the real run).

Writes results/test_results.json, results/test_results.md,
results/test_risk_coverage.png and results/test_gallery_size.png.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from scent_lens.assets import catalog, gallery_for, photo_embeddings, products  # noqa: E402
from scent_lens.calibration import FEATURES, predict_from_dict  # noqa: E402
from scent_lens.evaluate import photo_records, select_pool  # noqa: E402
from scent_lens.lock import acquire, git_clean, git_sha  # noqa: E402
from scent_lens.metrics import (  # noqa: E402
    bootstrap_by_group,
    brier,
    ece,
    rank_of_truth,
    risk_coverage,
    subsampled_ranks,
)
from scent_lens.plots import INK, INK_2, METHOD_LABELS, SERIES, style  # noqa: E402

SEED = 20261015
N_BOOT = 2000
FINAL, FINETUNED, FROZEN, TEXT = "finetuned_crop", "finetuned", "siglip2", "siglip2_text"
METHODS = ["clip_text", "siglip2_text", "dinov2", "siglip2", "clip", "finetuned", "finetuned_crop", "siglip2_crop"]
LABELS = {
    **METHOD_LABELS,
    "finetuned": "Fine tuned, no crop",
    "finetuned_crop": "Final system (fine tuned, cropped)",
    "siglip2_crop": "SigLIP 2 base, cropped (descriptive)",
}
SIZES = [150, 1000, 5000]
N_DRAWS = 10
CKPT = ROOT / "artifacts" / "finetune" / "siglip2_lr1e-4_s1.0" / "epoch1"
LOCK = ROOT / "results" / "test_lock.json"


def interval(groups, values, stat=None) -> dict:
    point, lo, hi = bootstrap_by_group(groups, values, stat, n_boot=N_BOOT, seed=SEED)
    return {"value": point, "lo": lo, "hi": hi}


def summary(rec: pd.DataFrame) -> dict:
    r = rec[rec.status == "matched"]
    out = {"photos": int(len(r)), "bottles": int(r.bottle_id.nunique())}
    if len(r):
        out.update({c: interval(r.bottle_id, r[c]) for c in ("top1", "top5", "lenient1")})
    return out


def breakdown(rec: pd.DataFrame) -> dict:
    phone = rec.condition != "online"
    return {
        "overall": summary(rec),
        "phone": summary(rec[phone]),
        "online": summary(rec[~phone]),
        "by_condition": {c: summary(g) for c, g in rec.groupby("condition")},
    }


def paired(a: pd.DataFrame, b: pd.DataFrame, mask=None, col: str = "top1") -> dict:
    """a minus b over the same matched photos, bottle level interval."""
    keep = (a.status == "matched").to_numpy() if mask is None else ((a.status == "matched") & mask).to_numpy()
    if not keep.any():
        return {"photos": 0}
    diff = a[col].to_numpy()[keep] - b[col].to_numpy()[keep]
    return {"photos": int(keep.sum()), **interval(a.bottle_id.to_numpy()[keep], diff)}


def random_baseline(rec: pd.DataFrame, prods: pd.DataFrame) -> dict:
    n = len(prods)
    r = rec[rec.status == "matched"]
    line_size = prods.line_key.value_counts()
    lenient = r.product_id.map(prods.line_key).map(line_size) / n
    return {"top1": 1 / n, "top5": 5 / n, "lenient1": float(lenient.mean()) if len(r) else None}


def gallery_sweep(gal, method: str, rec: pd.DataFrame, photos: pd.DataFrame) -> tuple[dict, np.ndarray, np.ndarray]:
    g = gallery_for(method, gal)
    m = (photos.status == "matched").to_numpy()
    emb = photo_embeddings(method, photos.photo_id.tolist())[m]
    truth = g.truth_index(photos.product_id[m].tolist())
    scores = g.scores(emb)
    groups = photos.bottle_id.to_numpy()[m]
    full = rank_of_truth(scores, truth)
    out, per_photo_top1 = {}, {}
    for size in SIZES:
        ranks = subsampled_ranks(scores, truth, size, N_DRAWS, SEED + size)
        t1, t5 = (ranks < 1).mean(axis=0), (ranks < 5).mean(axis=0)
        per_photo_top1[size] = t1
        out[str(size)] = {"top1": interval(groups, t1), "top5": interval(groups, t5)}
    out["full"] = {"top1": interval(groups, (full < 1).astype(float)), "top5": interval(groups, (full < 5).astype(float))}
    return out, per_photo_top1[150], (full < 1).astype(float)


def selective_stats(groups, answered, correct) -> dict:
    if len(groups) == 0:
        return {"photos": 0, "answered": 0, "coverage": {"value": None}, "selective_accuracy": {"value": None}}
    v = np.column_stack([answered, correct]).astype(float)

    def sel_acc(x):
        return float((x[:, 0] * x[:, 1]).sum() / x[:, 0].sum()) if x[:, 0].sum() else float("nan")

    return {
        "photos": int(len(v)),
        "answered": int(answered.sum()),
        "coverage": interval(groups, v[:, 0]),
        "selective_accuracy": interval(groups, v, sel_acc) if answered.any() else {"value": None},
    }


def latency(photo_path: Path, gal, prods, cal: dict) -> dict:
    """Median CPU seconds for one query: crop, embed, search, calibrate (models already loaded)."""
    from PIL import Image

    from scent_lens.crop import BottleDetector, apply_crop
    from scent_lens.finetune import load_finetuned

    det, enc = BottleDetector(device="cpu"), load_finetuned(CKPT, device="cpu")
    g = gallery_for(FINAL, gal)
    img = Image.open(photo_path).convert("RGB")
    one = pd.DataFrame([{"photo_id": "q", "bottle_id": "q", "condition": "x", "status": "absent", "product_id": ""}])

    def run():
        box, _ = det.detect(img)
        emb = enc.encode_images([apply_crop(img, box)])
        rec = photo_records(g, prods, emb, one)
        return predict_from_dict(cal["calibrator"], rec[list(FEATURES)].to_numpy(float))

    run()
    times = []
    for _ in range(5):
        t = time.perf_counter()
        run()
        times.append(time.perf_counter() - t)
    return {"median_seconds": float(np.median(times)), "runs": times, "device": "cpu", "torch_threads": __import__("torch").get_num_threads()}


def judge_interval(lo: float, hi: float) -> str:
    return "supported" if lo > 0 else "refuted" if hi < 0 else "inconclusive"


def pct(d: dict | None, key: str = "value") -> str:
    if not d or d.get(key) is None or (isinstance(d.get(key), float) and np.isnan(d[key])):
        return "n/a"
    if "lo" in d:
        return f"{100 * d['value']:.1f} [{100 * d['lo']:.1f}, {100 * d['hi']:.1f}]"
    return f"{100 * d[key]:.1f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="run on the development pool; no lock, scratch output")
    args = ap.parse_args()
    pool = "dev" if args.dry_run else "test"
    out_dir = ROOT / "artifacts" / "dryrun" if args.dry_run else ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    sha = git_sha(ROOT)
    if not args.dry_run:
        if not git_clean(ROOT):
            print("working tree has uncommitted changes; commit first so the run is reproducible")
            return 1
        acquire(LOCK, sha)  # raises if the test split was already evaluated

    gal = catalog()
    prods = products(gal)
    all_photos = pd.read_csv(ROOT / "data" / "real" / "photos.csv", dtype=str, keep_default_na=False)
    manifest = pd.read_csv(ROOT / "results" / "split_manifest.csv", dtype={"fold": "Int64"}, keep_default_na=False)
    photos = select_pool(all_photos, manifest, pool).reset_index(drop=True)
    cal = json.loads((ROOT / "results" / "calibrator.json").read_text())
    assert cal["method"] == FINAL, cal["method"]
    synth = pd.DataFrame(json.loads((ROOT / "results" / "synthetic_eval.json").read_text())["results"])
    synth_mean = float(synth[(synth.method == FROZEN) & (synth.metric == "top1") & (synth.condition != "none")].value.mean())

    recs = {m: photo_records(gallery_for(m, gal), prods, photo_embeddings(m, photos.photo_id.tolist()), photos)
            for m in METHODS}
    res = {
        "pool": pool, "git_sha": sha, "seed": SEED, "n_boot": N_BOOT,
        "photos": int(len(photos)), "bottles": int(photos.bottle_id.nunique()),
        "matched_photos": int((photos.status == "matched").sum()), "matched_bottles": int(photos[photos.status == "matched"].bottle_id.nunique()),
        "absent_photos": int((photos.status == "absent").sum()), "absent_bottles": int(photos[photos.status == "absent"].bottle_id.nunique()),
        "photos_by_condition": photos.condition.value_counts().to_dict(),
        "random": random_baseline(photos, prods),
        "methods": {m: breakdown(r) for m, r in recs.items()},
    }
    (out_dir / "test_records").mkdir(exist_ok=True)
    for m, r in recs.items():
        r.to_csv(out_dir / "test_records" / f"{m}.csv", index=False)

    # gallery size sweep, final system and frozen backbone
    sweep_final, t1_150, t1_full = gallery_sweep(gal, FINAL, recs[FINAL], photos)
    sweep_frozen, _, _ = gallery_sweep(gal, FROZEN, recs[FROZEN], photos)
    res["gallery_sweep"] = {FINAL: sweep_final, FROZEN: sweep_frozen}
    mg = photos.bottle_id.to_numpy()[(photos.status == "matched").to_numpy()]
    h4_diff = interval(mg, t1_150 - t1_full)

    # calibration and abstention, final system, all photos (absent photos count as incorrect)
    fr = recs[FINAL]
    probs = predict_from_dict(cal["calibrator"], fr[list(FEATURES)].to_numpy(float))
    correct = fr.top1.to_numpy()
    thr = cal["threshold"]
    answered = probs >= thr if thr is not None else np.zeros(len(fr), bool)
    groups = fr.bottle_id.to_numpy()
    phone = (fr.condition != "online").to_numpy()
    absent = (fr.status == "absent").to_numpy()
    cov, risk = risk_coverage(probs, correct)
    res["calibration"] = {
        "threshold": thr, "kind": cal["kind"],
        "ece_all": ece(probs, correct), "brier_all": brier(probs, correct),
        "ece_matched": ece(probs[~absent], correct[~absent]), "brier_matched": brier(probs[~absent], correct[~absent]),
        "selective": {
            "all": selective_stats(groups, answered, correct),
            "phone": selective_stats(groups[phone], answered[phone], correct[phone]),
            "online": selective_stats(groups[~phone], answered[~phone], correct[~phone]),
        },
        "rejection_rate": {"absent": float(1 - answered[absent].mean()) if absent.any() else None,
                           "matched": float(1 - answered[~absent].mean()) if (~absent).any() else None},
        "risk_coverage": {"coverage": cov.tolist(), "risk": risk.tolist()},
    }

    def h7_stat(x):
        rej, ab = x[:, 0], x[:, 1].astype(bool)
        return float(rej[ab].mean() - rej[~ab].mean()) if ab.any() and (~ab).any() else float("nan")

    h7 = interval(groups, np.column_stack([1 - answered, absent]).astype(float), h7_stat) if absent.any() else {"value": None}

    # most confused pairs, final system
    def name(pid):
        return f"{prods.loc[pid, 'brand']} {prods.loc[pid, 'name']}"

    wrong = fr[(fr.status == "matched") & (fr.top1 == 0)]
    res["confused_pairs"] = [{"truth": name(t), "predicted": name(p), "photos": n}
                             for (t, p), n in Counter(zip(wrong.product_id, wrong.pred1, strict=True)).most_common(10)]
    res["answered_absent"] = [{"photo_id": r.photo_id, "predicted": name(r.pred1)}
                              for r, a in zip(fr.itertuples(), answered, strict=True) if a and r.status == "absent"]

    # hypotheses
    h1 = paired(recs[FROZEN], recs[TEXT])
    m_phone = recs[FROZEN].status.eq("matched") & recs[FROZEN].condition.ne("online")
    real_phone = summary(recs[FROZEN][recs[FROZEN].condition != "online"])
    h5_all = paired(recs[FINETUNED], recs[FROZEN])
    h5_clean = paired(recs[FINETUNED], recs[FROZEN], mask=recs[FROZEN].condition.eq("clean"))
    top5_full = sweep_final["full"]["top5"]["value"]
    sel = res["calibration"]["selective"]["all"]
    sa, cv = sel["selective_accuracy"].get("value"), sel["coverage"]["value"]
    damage = {}
    pc = recs[FINAL][recs[FINAL].status.eq("matched") & recs[FINAL].condition.ne("online")]
    if (pc.condition == "clean").any():
        clean = pc[pc.condition == "clean"].top1.mean()
        damage = {c: float(clean - g.top1.mean()) for c, g in pc.groupby("condition") if c != "clean"}

    def h2_status():
        if real_phone.get("top1") is None:
            return "not tested"
        cut = synth_mean - 0.15
        return "supported" if real_phone["top1"]["hi"] < cut else "refuted" if real_phone["top1"]["lo"] > cut else "inconclusive"

    def h4_status():
        if h4_diff["lo"] > 0 and top5_full >= 0.5:
            return "supported"
        if (h4_diff["lo"] <= 0 and h4_diff["value"] <= 0) or top5_full < 0.5:
            return "refuted"
        return "inconclusive"

    def h5_status():
        d, c = h5_all["value"], h5_clean.get("value")
        if d < 0 or (c is not None and c < -0.05):
            return "refuted"
        if d >= 0.05 and h5_all["lo"] > 0 and (c is None or c >= -0.05):
            return "supported"
        return "inconclusive"

    res["hypotheses"] = {
        "H1": {"status": judge_interval(h1["lo"], h1["hi"]), "frozen_minus_text_top1": h1},
        "H2": {"status": h2_status(), "phone_frozen_top1": real_phone.get("top1"), "synthetic_mean_top1": synth_mean,
               "cut": synth_mean - 0.15, "phone_photos": int(m_phone.sum()),
               "online_frozen_top1_descriptive": summary(recs[FROZEN][recs[FROZEN].condition == "online"]).get("top1")},
        "H3": {"status": "not tested", "reason": "no glare or dim phone photos (preregistration note of 2026-10-28)",
               "damage_vs_clean_final_phone_descriptive": damage},
        "H4": {"status": h4_status(), "top1_150_minus_full": h4_diff, "top5_full": top5_full},
        "H5": {"status": h5_status(), "finetuned_minus_frozen_top1": h5_all, "clean_phone_difference": h5_clean},
        "H6": {"status": "supported" if (sa is not None and not np.isnan(sa) and sa >= 0.85 and cv >= 0.25) else "refuted",
               "selective_accuracy": sa, "coverage": cv},
        "H7": {"status": ("supported" if h7.get("value") is not None and h7["value"] > 0 else "refuted") if absent.any() else "not tested",
               "absent_minus_matched_rejection": h7},
    }
    res["latency"] = latency(ROOT / photos.path.iloc[0], gal, prods, cal)
    res["finished_utc"] = datetime.now(UTC).isoformat(timespec="seconds")
    (out_dir / "test_results.json").write_text(json.dumps(res, indent=2, default=float) + "\n")
    write_report(res, out_dir)
    plot(res, out_dir)
    if not args.dry_run:
        lock = json.loads(LOCK.read_text())
        LOCK.write_text(json.dumps({**lock, "finished_utc": res["finished_utc"]}, indent=2) + "\n")
    print((out_dir / "test_results.md").read_text())
    return 0


def write_report(res: dict, out_dir: Path) -> None:
    L = [f"# Test results ({res['pool']} pool)", "",
         f"One time evaluation, git {res['git_sha'][:7]}, seed {res['seed']}, {res['n_boot']} bottle level bootstrap resamples. "
         f"{res['photos']} photos of {res['bottles']} bottles: {res['matched_photos']} photos of {res['matched_bottles']} matched bottles, "
         f"{res['absent_photos']} photos of {res['absent_bottles']} absent bottles. Full gallery of 24,642 products. Percent with 95% intervals.", "",
         "## Hypotheses", "", "| | Result | Evidence |", "|---|---|---|"]
    h = res["hypotheses"]
    L.append(f"| H1 frozen image beats text | {h['H1']['status']} | difference {pct(h['H1']['frozen_minus_text_top1'])} |")
    L.append(f"| H2 real phone photos more than 15 points below synthetic | {h['H2']['status']} | phone top 1 {pct(h['H2']['phone_frozen_top1'])} vs cut {100 * h['H2']['cut']:.1f} (synthetic {100 * h['H2']['synthetic_mean_top1']:.1f}) |")
    L.append(f"| H3 damage order by condition | {h['H3']['status']} | {h['H3']['reason']} |")
    L.append(f"| H4 gallery size | {h['H4']['status']} | top 1 at 150 minus full {pct(h['H4']['top1_150_minus_full'])}; top 5 at full {100 * h['H4']['top5_full']:.1f} |")
    L.append(f"| H5 fine tuning gain at least 5 points | {h['H5']['status']} | overall {pct(h['H5']['finetuned_minus_frozen_top1'])}; clean phone {pct(h['H5']['clean_phone_difference'])} |")
    sa = h["H6"]["selective_accuracy"]
    L.append(f"| H6 abstention, at least 85% at 25% coverage | {h['H6']['status']} | selective accuracy {'n/a' if sa is None or np.isnan(sa) else f'{100 * sa:.1f}'}, coverage {100 * h['H6']['coverage']:.1f} |")
    L.append(f"| H7 absent bottles rejected more (exploratory) | {h['H7']['status']} | difference {pct(h['H7']['absent_minus_matched_rejection'])} |")
    L += ["", "## Accuracy, matched bottles", "", "| Method | Top 1 | Top 5 | Lenient top 1 | Phone top 1 | Online top 1 |", "|---|---|---|---|---|---|"]
    rnd = res["random"]
    rnd_len = "n/a" if rnd["lenient1"] is None else f"{100 * rnd['lenient1']:.3f}"
    L.append(f"| Random | {100 * rnd['top1']:.3f} | {100 * rnd['top5']:.3f} | {rnd_len} | | |")
    for m, r in res["methods"].items():
        o = r["overall"]
        L.append(f"| {LABELS.get(m, m)} | {pct(o.get('top1'))} | {pct(o.get('top5'))} | {pct(o.get('lenient1'))} | {pct(r['phone'].get('top1'))} | {pct(r['online'].get('top1'))} |")
    conds = [c for c in ["clean", "angled", "glare", "dim", "cluttered", "partial", "online"] if c in res["photos_by_condition"]]
    L += ["", "## Top 1 by condition (matched photos)", "", "| Method | " + " | ".join(f"{c} ({res['methods'][FINAL]['by_condition'].get(c, {}).get('photos', 0)})" for c in conds) + " |", "|---" * (len(conds) + 1) + "|"]
    for m, r in res["methods"].items():
        L.append(f"| {LABELS.get(m, m)} | " + " | ".join(pct(r["by_condition"].get(c, {}).get("top1")) for c in conds) + " |")
    L += ["", "## Gallery size (matched photos, 10 draws per size)", "", "| Method | Size | Top 1 | Top 5 |", "|---|---|---|---|"]
    for m, sw in res["gallery_sweep"].items():
        for size, v in sw.items():
            L.append(f"| {LABELS.get(m, m)} | {size} | {pct(v['top1'])} | {pct(v['top5'])} |")
    c = res["calibration"]
    L += ["", "## Confidence and abstention (final system)", "",
          f"Threshold {c['threshold']} ({c['kind']} calibrator, fixed on development photos). ECE {c['ece_all']:.3f} and Brier {c['brier_all']:.3f} over all photos (absent photos count as incorrect); matched photos only: ECE {c['ece_matched']:.3f}, Brier {c['brier_matched']:.3f}.", "",
          "| Set | Photos | Answered | Coverage | Selective accuracy |", "|---|---|---|---|---|"]
    for k, v in c["selective"].items():
        L.append(f"| {k} | {v['photos']} | {v['answered']} | {pct(v['coverage'])} | {pct(v['selective_accuracy'])} |")
    rr = c["rejection_rate"]
    L.append("")
    L.append(f"Rejection rate: absent bottles {pct(rr, 'absent') if rr['absent'] is not None else 'n/a'}, matched bottles {pct(rr, 'matched') if rr['matched'] is not None else 'n/a'}. Absent photos answered: {len(res['answered_absent'])}.")
    L += ["", "## Most confused pairs (final system)", "", "| Truth | Predicted | Photos |", "|---|---|---|"]
    L += [f"| {p['truth']} | {p['predicted']} | {p['photos']} |" for p in res["confused_pairs"]]
    d = res["hypotheses"]["H3"]["damage_vs_clean_final_phone_descriptive"]
    L += ["", "## Descriptive", "", "Damage versus clean phone photos, final system (clean top 1 minus condition top 1): " + (", ".join(f"{k} {100 * v:.1f}" for k, v in d.items()) or "n/a") + ". The angled photos are mostly rear views.",
          f"CPU latency for one query (crop, embed, search, calibrate; models loaded): median {res['latency']['median_seconds']:.2f} s over 5 runs, {res['latency']['torch_threads']} threads."]
    (out_dir / "test_results.md").write_text("\n".join(L) + "\n")


def plot(res: dict, out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    style()
    rc = res["calibration"]["risk_coverage"]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(rc["coverage"], np.array(rc["risk"]) * 100, color=SERIES[6], lw=2, label=LABELS[FINAL])
    ax.axhline(15, color=INK_2, ls="--", lw=1, label="15% risk (85% selective accuracy)")
    ax.set_xlabel("Coverage (share of photos answered)")
    ax.set_ylabel("Risk, % wrong among answered")
    ax.set_title(f"Risk and coverage, {res['pool']} photos", color=INK)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "test_risk_coverage.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    sizes = [150, 1000, 5000, 24642]
    for m, sw in res["gallery_sweep"].items():
        for metric, ls in (("top1", "-"), ("top5", ":")):
            vals = [sw[k][metric]["value"] * 100 for k in ("150", "1000", "5000", "full")]
            ax.plot(sizes, vals, ls=ls, marker="o", color=SERIES[6] if m == FINAL else SERIES[1],
                    label=f"{LABELS.get(m, m)}, {metric.replace('top', 'top ')}")
    ax.set_xscale("log")
    ax.set_xlabel("Gallery size (products)")
    ax.set_ylabel("% of matched photos")
    ax.set_title(f"Gallery size, {res['pool']} photos", color=INK)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "test_gallery_size.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
