"""Baselines on the development pool only (never the test split).

Writes results/dev_baselines.json and results/dev_baselines.md, and per photo
records to artifacts/dev/<method>.csv for the calibrator. Used to choose the
frozen backbone and, later, the final system.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scent_lens.assets import (  # noqa: E402
    IMAGE_METHODS,
    TEXT_METHODS,
    catalog,
    gallery_for,
    photo_embeddings,
    products,
)
from scent_lens.evaluate import by_condition, photo_records, select_pool, summarize  # noqa: E402
from scent_lens.plots import METHOD_LABELS  # noqa: E402

SEED = 20261015


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default=",".join(IMAGE_METHODS + TEXT_METHODS))
    ap.add_argument("--tag", default="dev_baselines")
    args = ap.parse_args()
    gal = catalog()
    prods = products(gal)
    photos = pd.read_csv(ROOT / "data" / "real" / "photos.csv", dtype=str, keep_default_na=False)
    manifest = pd.read_csv(ROOT / "results" / "split_manifest.csv", dtype={"fold": "Int64"}, keep_default_na=False)
    dev = select_pool(photos, manifest, "dev").reset_index(drop=True)
    out_dir = ROOT / "artifacts" / "dev"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {"pool": "dev", "seed": SEED, "photos": int(len(dev)), "bottles": int(dev.bottle_id.nunique()), "methods": {}}
    for method in args.methods.split(","):
        g = gallery_for(method, gal)
        rec = photo_records(g, prods, photo_embeddings(method, dev.photo_id.tolist()), dev)
        rec["fold"] = dev["fold"].to_numpy()
        rec.to_csv(out_dir / f"{method}.csv", index=False)
        results["methods"][method] = {"overall": summarize(rec, seed=SEED), "by_condition": by_condition(rec, seed=SEED)}
        o = results["methods"][method]["overall"]
        print(f"{method:14s} top1 {o['top1']['value']:.3f} top5 {o['top5']['value']:.3f} lenient {o['lenient1']['value']:.3f}", flush=True)
    (ROOT / "results" / f"{args.tag}.json").write_text(json.dumps(results, indent=2) + "\n")

    def f(d):
        return f"{100 * d['value']:.1f} [{100 * d['lo']:.1f}, {100 * d['hi']:.1f}]"

    conds = ["clean", "angled", "glare", "dim", "cluttered", "partial"]
    lines = [
        f"# Development pool results ({args.tag})", "",
        f"Development pool only: {results['photos']} photos of {results['bottles']} bottles. Matched bottles only; "
        "values are percent with 95% bootstrap intervals resampling bottles. Seed "
        f"{SEED}.", "",
        "| Method | Top 1 | Top 5 | Lenient top 1 |", "|---|---|---|---|",
    ]
    for m, r in results["methods"].items():
        o = r["overall"]
        lines.append(f"| {METHOD_LABELS.get(m, m)} | {f(o['top1'])} | {f(o['top5'])} | {f(o['lenient1'])} |")
    lines += ["", "Top 1 by condition", "", "| Method | " + " | ".join(conds) + " |", "|---" * (len(conds) + 1) + "|"]
    for m, r in results["methods"].items():
        bc = r["by_condition"]
        lines.append(f"| {METHOD_LABELS.get(m, m)} | " + " | ".join(
            f"{100 * bc[c]['top1']['value']:.1f}" if c in bc else "n/a" for c in conds) + " |")
    (ROOT / "results" / f"{args.tag}.md").write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
