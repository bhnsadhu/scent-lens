"""Download the doevent/perfume metadata and image archive at a pinned revision.

Checks free disk against the archive size before downloading. Files land in
data/raw/ (gitignored). Safe to rerun: existing complete files are skipped.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

REPO_ID = "doevent/perfume"
REVISION = "9d2db452d8f0d649c4bfb0189a2fcc5282028a3c"
FILES = ("README.md", "perfumes.csv", "images.zip")
HEADROOM_BYTES = 8 * 1024**3  # keep room for weights, venv growth, embeddings

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    info = HfApi().dataset_info(REPO_ID, revision=REVISION, files_metadata=True)
    sizes = {s.rfilename: s.size for s in info.siblings}
    need = sum(sizes[f] for f in FILES if not (OUT / f).exists() or (OUT / f).stat().st_size != sizes[f])
    free = shutil.disk_usage(OUT).free
    print(f"need {need / 1e9:.2f} GB, free {free / 1e9:.2f} GB, headroom {HEADROOM_BYTES / 1e9:.1f} GB")
    if need + HEADROOM_BYTES > free:
        print("not enough disk; stopping before download", file=sys.stderr)
        return 2
    for f in FILES:
        target = OUT / f
        if target.exists() and target.stat().st_size == sizes[f]:
            print(f"have {f}")
            continue
        hf_hub_download(REPO_ID, f, repo_type="dataset", revision=REVISION, local_dir=OUT)
        print(f"downloaded {f} ({sizes[f] / 1e6:.1f} MB)")
    (OUT / "download_manifest.json").write_text(
        json.dumps({"repo_id": REPO_ID, "revision": REVISION, "sizes": {f: sizes[f] for f in FILES}}, indent=2)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
