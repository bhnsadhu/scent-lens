"""Fit the confidence calibrator and abstention threshold (rules 5 and 6).

Reads the development pool records of the final system from
artifacts/dev/<method>.csv (written by eval_dev.py), fits logistic and
isotonic calibrators with 5 fold grouped cross validation (folds from the
split manifest, so all photos of a bottle share a fold), keeps logistic unless
isotonic's out of fold Brier is lower by more than 0.01, and picks the lowest
threshold whose accepted out of fold photos have a 95% upper bound on error at
or below 10%. Absent bottle photos count as incorrect. The calibrator refit on
all development photos and the threshold go to results/calibrator.json.
Test photos are never loaded.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scent_lens.calibration import (  # noqa: E402
    FEATURES,
    calibrator_to_dict,
    choose_threshold,
    fit_dev_calibrator,
    out_of_fold_probs,
)
from scent_lens.metrics import brier, selective  # noqa: E402

SEED = 20261015
TARGET = 0.90


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, help="records file artifacts/dev/<method>.csv of the final system")
    args = ap.parse_args()
    rec = pd.read_csv(ROOT / "artifacts" / "dev" / f"{args.method}.csv", dtype={"bottle_id": str})
    manifest = pd.read_csv(ROOT / "results" / "split_manifest.csv", dtype=str, keep_default_na=False)
    pool = dict(zip(manifest.bottle_id, manifest.pool, strict=True))

    X = rec[list(FEATURES)].to_numpy(dtype=np.float64)
    y = rec["top1"].to_numpy(dtype=np.int64)  # absent photos are 0: answering them is always wrong
    folds = rec["fold"].to_numpy()

    oof = {k: out_of_fold_probs(X, y, folds, kind=k, seed=SEED) for k in ("logistic", "isotonic")}
    scores = {k: brier(p, y) for k, p in oof.items()}
    kind = "isotonic" if scores["logistic"] - scores["isotonic"] > 0.01 else "logistic"
    threshold = choose_threshold(oof[kind], y, TARGET)
    sel_acc, coverage = selective(oof[kind], y, threshold) if np.isfinite(threshold) else (float("nan"), 0.0)

    cal = fit_dev_calibrator(rec, pool, kind=kind, seed=SEED)
    out = {
        "method": args.method,
        "seed": SEED,
        "dev_photos": int(len(rec)),
        "dev_correct": int(y.sum()),
        "oof_brier": scores,
        "kind": kind,
        "target_selective_accuracy": TARGET,
        "threshold": threshold if np.isfinite(threshold) else None,
        "always_abstain": not np.isfinite(threshold),
        "oof_selective_accuracy": None if np.isnan(sel_acc) else sel_acc,
        "oof_coverage": coverage,
        "calibrator": calibrator_to_dict(cal),
    }
    (ROOT / "results" / "calibrator.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "calibrator"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
