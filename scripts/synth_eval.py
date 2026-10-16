"""Synthetic robustness check: augmented catalog images against the full gallery.

Optimistic by construction: each query is derived from an image that is itself
in the gallery. Reports, per method and condition, top 1 and top 5 (strict) and
lenient top 1 over the full gallery with 95% bootstrap intervals over queries
(one query per product), plus a gallery size sweep (150, 1k, 5k, full) where
each smaller gallery holds the true product and random distractors.

Methods: B0 random, B1 zero shot text prompts (CLIP, SigLIP 2 text towers),
B2 nearest neighbor on frozen image embeddings (DINOv2, SigLIP 2, CLIP).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scent_lens.augment import CONDITIONS  # noqa: E402
from scent_lens.metrics import bootstrap_by_group, rank_of_truth, subsampled_ranks  # noqa: E402
from scent_lens.plots import METHOD_COLORS, METHOD_LABELS, style  # noqa: E402
from scent_lens.retrieval import Gallery  # noqa: E402

SEED = 20261015
SIZES = [150, 1000, 5000]
N_DRAWS = 10
N_BOOT = 1000
IMAGE_METHODS = ["dinov2", "siglip2", "clip"]
TEXT_METHODS = ["siglip2_text", "clip_text"]
SYNTH = ROOT / "artifacts" / "synth"


def load_gallery(method: str, gal: pd.DataFrame) -> Gallery:
    if method.endswith("_text"):
        bb = method.removesuffix("_text")
        emb = np.load(ROOT / "artifacts" / "emb" / bb / "text.npy")
        pids = (ROOT / "artifacts" / "emb" / bb / "text_products.txt").read_text().split()
        return Gallery(emb, pids)
    emb = np.load(ROOT / "artifacts" / "emb" / method / "gallery.npy")
    return Gallery(emb[gal["emb_index"].to_numpy()], gal["product_id"].tolist())


def load_queries(backbone: str, cond: str) -> np.ndarray:
    files = sorted(SYNTH.glob(f"{backbone}/{cond}_*.npy"))
    files = [f for f in files if ".tmp" not in f.name]
    return np.concatenate([np.load(f) for f in files])


def main() -> int:
    gal = pd.read_csv(ROOT / "artifacts" / "catalog" / "gallery.csv", dtype={"year": str}, keep_default_na=False)
    q = pd.read_csv(ROOT / "results" / "synthetic_query_ids.csv")
    line_of = gal.drop_duplicates("product_id").set_index("product_id")["line_key"]
    conds = list(CONDITIONS)
    rows, sweep = [], []
    n_products = gal["product_id"].nunique()

    for method in IMAGE_METHODS + TEXT_METHODS:
        g = load_gallery(method, gal)
        assert len(g) == n_products, (method, len(g), n_products)
        truth = g.truth_index(q["product_id"].tolist())
        true_line = line_of.loc[q["product_id"]].to_numpy()
        backbone = method.removesuffix("_text")
        for cond in conds:
            emb = load_queries(backbone, cond)
            assert emb.shape[0] == len(q)
            s = g.scores(emb)
            ranks = rank_of_truth(s, truth)
            pred = g.products[s.argmax(1)]
            lenient = (line_of.loc[pred].to_numpy() == true_line).astype(float)
            for name, vals in [
                ("top1", (ranks < 1).astype(float)),
                ("top5", (ranks < 5).astype(float)),
                ("lenient_top1", lenient),
            ]:
                point, lo, hi = bootstrap_by_group(q["product_id"], vals, n_boot=N_BOOT, seed=SEED)
                rows.append(dict(method=method, condition=cond, metric=name, value=point, lo=lo, hi=hi))
            for size in SIZES + [n_products]:
                sub = subsampled_ranks(s, truth, min(size, n_products), N_DRAWS, seed=SEED + size)
                for k in (1, 5):
                    per_draw = (sub < k).mean(axis=1)
                    sweep.append(
                        dict(method=method, condition=cond, gallery_size=int(min(size, n_products)), k=k,
                             mean=float(per_draw.mean()), sd_over_draws=float(per_draw.std()))
                    )
            print(method, cond, f"top1={rows[-3]['value']:.3f} top5={rows[-2]['value']:.3f}", flush=True)

    for cond in conds:  # B0 random, exact expectation
        for k in (1, 5):
            rows.append(dict(method="random", condition=cond, metric=f"top{k}", value=k / n_products, lo=k / n_products, hi=k / n_products))

    res = pd.DataFrame(rows)
    sw = pd.DataFrame(sweep)
    out = {
        "label": "OPTIMISTIC SYNTHETIC ROBUSTNESS CHECK, NOT A REAL RESULT: each query is an augmented copy of an image that is in the gallery.",
        "seed": SEED,
        "n_queries": int(len(q)),
        "n_gallery_products": int(n_products),
        "n_gallery_images": int(len(gal)),
        "bootstrap": f"{N_BOOT} resamples over queries (one query per product), percentile 95% interval",
        "gallery_sweep": f"{N_DRAWS} random distractor draws per query and size; hypergeometric sampling",
        "results": rows,
        "gallery_sweep_results": sweep,
    }
    (ROOT / "results" / "synthetic_eval.json").write_text(json.dumps(out, indent=2) + "\n")
    write_table(res, sw, n_products, len(q))
    plot(res, sw, n_products)
    return 0


def fmt(r) -> str:
    return f"{100 * r.value:.1f} [{100 * r.lo:.1f}, {100 * r.hi:.1f}]"


def write_table(res: pd.DataFrame, sw: pd.DataFrame, n_products: int, n_q: int) -> None:
    conds = list(CONDITIONS)
    lines = [
        "# Synthetic robustness check",
        "",
        "Optimistic synthetic robustness check, not a real result. Each query is an augmented copy of a catalog image that is itself in the gallery, so these numbers bound what real photos can reach rather than estimate it.",
        "",
        f"{n_q} queries, full gallery of {n_products} products. Values are percent with 95% bootstrap intervals over queries. Seed {SEED}.",
        "",
    ]
    for metric, title in [("top1", "Top 1, strict"), ("top5", "Top 5, strict"), ("lenient_top1", "Top 1, lenient")]:
        lines += [f"## {title}", "", "| Method | " + " | ".join(conds) + " |", "|---" * (len(conds) + 1) + "|"]
        for m in IMAGE_METHODS + TEXT_METHODS + (["random"] if metric != "lenient_top1" else []):
            sub = res[(res.method == m) & (res.metric == metric)].set_index("condition")
            if sub.empty:
                continue
            lines.append(f"| {METHOD_LABELS[m]} | " + " | ".join(fmt(sub.loc[c]) for c in conds) + " |")
        lines.append("")
    distorted = [c for c in conds if c != "none"]
    lines += ["## Gallery size sweep, top 1 and top 5, mean over the seven distorted conditions", "",
              "| Method | k | " + " | ".join(str(s) for s in sorted(sw.gallery_size.unique())) + " |",
              "|---|---" + "|---" * sw.gallery_size.nunique() + "|"]
    for m in IMAGE_METHODS + TEXT_METHODS:
        for k in (1, 5):
            part = sw[(sw.method == m) & (sw.k == k) & sw.condition.isin(distorted)].groupby("gallery_size")["mean"].mean()
            lines.append(f"| {METHOD_LABELS[m]} | {k} | " + " | ".join(f"{100 * v:.1f}" for v in part) + " |")
    lines.append("")
    (ROOT / "results" / "synthetic_eval.md").write_text("\n".join(lines))


def plot(res: pd.DataFrame, sw: pd.DataFrame, n_products: int) -> None:
    import matplotlib.pyplot as plt

    style()
    conds = list(CONDITIONS)
    methods = IMAGE_METHODS + TEXT_METHODS
    fig, ax = plt.subplots(figsize=(10, 4.2))
    width = 0.8 / len(methods)
    for i, m in enumerate(methods):
        sub = res[(res.method == m) & (res.metric == "top1")].set_index("condition").loc[conds]
        x = np.arange(len(conds)) + (i - (len(methods) - 1) / 2) * width
        ax.bar(x, 100 * sub.value, width=width * 0.85, color=METHOD_COLORS[m], label=METHOD_LABELS[m])
        ax.errorbar(x, 100 * sub.value, yerr=[100 * (sub.value - sub.lo), 100 * (sub.hi - sub.value)],
                    fmt="none", ecolor="#52514e", elinewidth=0.8, capsize=0)
    ax.set_xticks(np.arange(len(conds)), [c.replace("_", " ") for c in conds])
    ax.set_ylabel("Top 1 accuracy, %")
    ax.set_ylim(0, 100)
    ax.set_title("Synthetic check (optimistic): top 1 by condition, full gallery")
    ax.legend(ncols=3, loc="upper center", bbox_to_anchor=(0.5, -0.1), fontsize=9)
    fig.tight_layout()
    fig.savefig(ROOT / "results" / "synthetic_by_condition.png", dpi=160)
    plt.close(fig)

    distorted = [c for c in conds if c != "none"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, k in zip(axes, (1, 5), strict=True):
        for m in methods:
            part = sw[(sw.method == m) & (sw.k == k) & sw.condition.isin(distorted)].groupby("gallery_size")["mean"].mean()
            ax.plot(part.index, 100 * part.values, marker="o", color=METHOD_COLORS[m], label=METHOD_LABELS[m])
        ax.set_xscale("log")
        ax.set_xticks(sorted(sw.gallery_size.unique()), [f"{s:,}" for s in sorted(sw.gallery_size.unique())])
        ax.minorticks_off()
        ax.set_xlabel("Gallery size, products")
        ax.set_title(f"Top {k}, mean over distorted conditions")
        ax.set_ylim(0, 100)
    axes[0].set_ylabel("Accuracy, %")
    axes[1].legend(fontsize=8, loc="lower left")
    fig.suptitle("Synthetic check (optimistic): accuracy falls as the gallery grows", x=0.01, ha="left", fontsize=12, fontweight="semibold")
    fig.tight_layout()
    fig.savefig(ROOT / "results" / "synthetic_gallery_size.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
