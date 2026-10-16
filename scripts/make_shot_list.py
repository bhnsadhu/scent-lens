"""Write data/real/shot_list.md and the data/real/labels.csv template from matches.csv.

Matched and absent bottles get all six conditions with two slots each; the
first slot is required, the second optional. Ambiguous bottles are excluded.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REAL = ROOT / "data" / "real"
CONDITIONS = ["clean", "angled", "glare", "dim", "cluttered", "partial"]


def main() -> int:
    m = pd.read_csv(REAL / "matches.csv", dtype=str, keep_default_na=False)
    m = m[m.status.isin(["matched", "absent"])]
    lines = [
        "# ScentLens shot list",
        "",
        "One photo per condition is required; a second is optional but helps. See docs/photo_protocol.md.",
        "File name: bottle_id__condition__n, for example b001__clean__1.heic.",
        "",
    ]
    labels = []
    for r in m.itertuples():
        tag = "" if r.status == "matched" else " (not in catalog: tests the not sure answer)"
        lines.append(f"## {r.bottle_id}: {r.brand} {r.name}{tag}")
        lines.append("")
        for c in CONDITIONS:
            lines.append(f"- [ ] {c} 1")
            lines.append(f"- [ ] {c} 2 (optional)")
            for n in (1, 2):
                labels.append(
                    {"filename": f"{r.bottle_id}__{c}__{n}", "bottle_id": r.bottle_id, "brand": r.brand,
                     "name": r.name, "condition": c, "n": n, "taken": "no", "notes": ""}
                )
        lines.append("")
    (REAL / "shot_list.md").write_text("\n".join(lines))
    labels_path = REAL / "labels.csv"
    if labels_path.exists():
        print("labels.csv exists; not overwriting")
    else:
        pd.DataFrame(labels).to_csv(labels_path, index=False)
    print(f"{len(m)} bottles, {len(m) * len(CONDITIONS)} required photos, up to {len(labels)} total")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
