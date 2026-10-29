"""Run the exported Recognizer end to end on every development, test and web photo.

This checks the shipped system (artifacts/export, scent_lens.recognize), not
cached features: each photo goes from its original file through cleaning,
crop, embedding, label reading and the answer decision. All settings are the
kept version 2 settings (results/calibrator_v2.json); nothing is fitted here.
With --reader vision the top match must agree with scripts/eval_v2.py
(label variant). Writes results/recognizer_<reader>.json and .md.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from eval_v2 import query_photos  # noqa: E402
from scent_lens.photos import register_heif  # noqa: E402
from scent_lens.recognize import Recognizer  # noqa: E402

WEB = ROOT / "data" / "real" / "web"
PACK = ROOT / "data" / "real" / "online_refs" / "scentlens_audited"


def originals() -> Callable[[str], Path]:
    """photo id -> original file (phone HEIC, audited online image, or web download)."""
    web = pd.read_csv(WEB / "manifest.csv", dtype=str).set_index("web_id").file
    online = {b["id"]: b for b in json.loads((PACK / "photo_manifest.json").read_text())["bottles"]}

    def path(pid: str) -> Path:
        if pid in web.index:
            return WEB / "raw" / web[pid]
        if "__online__" in pid:
            bottle, _, n = pid.split("__")
            return PACK / online[bottle]["photos"][int(n) - 1]["file"]
        return sorted((ROOT / "data" / "real" / "raw").glob(f"{pid}.*"))[0]

    return path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reader", default="vision", choices=["vision", "rapidocr"])
    ap.add_argument("--device", default=None, help="cpu to measure the latency a free CPU host would see")
    args = ap.parse_args()
    register_heif()
    manifest = pd.read_csv(ROOT / "results" / "split_manifest.csv", dtype={"fold": "Int64"}, keep_default_na=False)
    photos = query_photos(manifest)
    ref = pd.read_csv(ROOT / "artifacts" / "real" / "v2" / "records_label.csv", dtype=str, keep_default_na=False).set_index("photo_id")
    rec = Recognizer(ROOT / "artifacts" / "export", device=args.device, reader=args.reader)
    rec(originals()(photos.photo_id.iloc[0]))  # warm up
    path, rows = originals(), []
    for r in photos.itertuples():
        out = rec(path(r.photo_id))
        ids = [m.product_id for m in out.matches]
        rows.append({"photo_id": r.photo_id, "pool": r.pool, "status": r.status, "truth": r.product_id,
                     "pred1": ids[0], "top1": ids[0] == r.product_id, "top3": r.product_id in ids,
                     "answered": out.answered, "reason": out.reason, "confidence": out.confidence,
                     "same_as_eval": ids[0] == ref.pred1[r.photo_id], "seconds": out.seconds})
    d = pd.DataFrame(rows)
    res = {"reader": args.reader, "device": rec.device, "settings": "results/calibrator_v2.json",
           "median_seconds": float(d.seconds.median()), "p90_seconds": float(d.seconds.quantile(0.9))}
    lines = [f"# Exported recognizer, label read by {args.reader}", "",
             f"End to end from original files through artifacts/export with the kept version 2 settings (results/calibrator_v2.json), on {rec.device}. Nothing was fitted. Test bottles were seen before version 2, so test columns are optimistic; web photos are the fresh check.", "",
             "| Pool | Photos | Top 1 | Top 3 | Answered | Correct when answered | Absent photos answered | Same top 1 as eval_v2 |", "|---|---|---|---|---|---|---|---|"]
    for pool in ("dev", "test", "web"):
        g = d[d.pool == pool]
        m, a = g[g.status == "matched"], g[g.answered]
        absent = g[g.status == "absent"]
        res[pool] = {"photos": len(g), "top1": float(m.top1.mean()), "top3": float(m.top3.mean()), "answered": len(a),
                     "coverage": len(a) / len(g), "selective_accuracy": float(a.top1.mean()) if len(a) else None,
                     "absent_answered": int(absent.answered.sum()), "absent_photos": len(absent),
                     "same_top1_as_eval": float(g.same_as_eval.mean()),
                     "wrong_answers": [{"photo_id": x.photo_id, "status": x.status, "predicted": x.pred1} for x in a[~a.top1].itertuples()]}
        s = res[pool]
        lines.append(f"| {pool} | {s['photos']} | {100 * s['top1']:.1f} | {100 * s['top3']:.1f} | {s['answered']} ({100 * s['coverage']:.1f}%) | "
                     f"{len(a) - len(s['wrong_answers'])} of {len(a)} | {s['absent_answered']} of {s['absent_photos']} | {100 * s['same_top1_as_eval']:.1f}% |")
    wrong = [w | {"pool": p} for p in ("dev", "test", "web") for w in res[p]["wrong_answers"]]
    lines += ["", f"Latency per photo: median {res['median_seconds']:.2f} s, 90th percentile {res['p90_seconds']:.2f} s.", "",
              "Wrong answers: " + ("none." if not wrong else "; ".join(
                  f"{w['photo_id']} ({w['pool']}, {w['status']}) answered as {w['predicted']} {rec.products.loc[w['predicted'], 'brand']} {rec.products.loc[w['predicted'], 'name']}" for w in wrong) + ".")]
    (ROOT / "results" / f"recognizer_{args.reader}.json").write_text(json.dumps(res, indent=2, default=float) + "\n")
    (ROOT / "results" / f"recognizer_{args.reader}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
