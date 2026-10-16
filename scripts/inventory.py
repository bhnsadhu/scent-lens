"""Catalog inventory: rows, products, images per product, corrupt files, sizes, near duplicates.

Pass 1 decodes every referenced image at reduced size and appends one JSON line
per image to artifacts/inventory/images.jsonl (resumable: done images are
skipped). Pass 2 summarizes into results/data_inventory.json.
"""

from __future__ import annotations

import hashlib
import io
import json
import sys
from pathlib import Path

import imagehash
import numpy as np
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scent_lens.catalog import ZipImages, load_catalog  # noqa: E402

RAW = ROOT / "data" / "raw"
OUT_DIR = ROOT / "artifacts" / "inventory"
RECORDS = OUT_DIR / "images.jsonl"
RESULT = ROOT / "results" / "data_inventory.json"
NEAR_DUP_BITS = 4  # phash Hamming distance at or below this counts as a near duplicate


def scan(df, store: ZipImages) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    done = set()
    if RECORDS.exists():
        with RECORDS.open() as f:
            done = {json.loads(line)["image_name"] for line in f if line.strip()}
    todo = [n for n in df["image_name"] if n and n not in done]
    print(f"{len(done)} done, {len(todo)} to scan", flush=True)
    with RECORDS.open("a") as f:
        for i, name in enumerate(tqdm(todo, mininterval=10)):
            rec = {"image_name": name}
            if name not in store:
                rec.update(ok=False, error="missing from zip")
            else:
                try:
                    data = store.read_bytes(name)
                    with Image.open(io.BytesIO(data)) as header:
                        width, height, fmt = header.width, header.height, header.format
                    img = store.open(name, max_side=256)
                    rec.update(
                        ok=True,
                        bytes=len(data),
                        width=width,
                        height=height,
                        format=fmt,
                        md5=hashlib.md5(data).hexdigest(),
                        phash=str(imagehash.phash(img)),
                    )
                except Exception as e:  # corrupt or unreadable image
                    rec.update(ok=False, error=f"{type(e).__name__}: {e}"[:200])
            f.write(json.dumps(rec) + "\n")
            if i % 500 == 0:
                f.flush()


def near_duplicate_pairs(hashes: list[str]) -> list[tuple[int, int, int]]:
    bits = np.array([int(h, 16) for h in hashes], dtype=np.uint64)
    pairs = []
    for start in range(0, len(bits), 1024):
        block = bits[start : start + 1024]
        dist = np.bitwise_count(block[:, None] ^ bits[None, :])
        for i, j in zip(*np.nonzero(dist <= NEAR_DUP_BITS), strict=True):
            a = start + int(i)
            if a < int(j):
                pairs.append((a, int(j), int(dist[i, j])))
    return pairs


def summarize(df) -> dict:
    with RECORDS.open() as f:
        recs = {r["image_name"]: r for r in map(json.loads, f)}
    ok = [r for r in recs.values() if r.get("ok")]
    bad = [r for r in recs.values() if not r.get("ok")]
    names = [r["image_name"] for r in ok]
    pairs = near_duplicate_pairs([r["phash"] for r in ok])
    key_of = dict(zip(df["image_name"], df["product_key"], strict=True))
    same_product = sum(key_of[names[a]] == key_of[names[b]] for a, b, _ in pairs)
    exact = sum(d == 0 for _, _, d in pairs)

    md5_groups: dict[str, set[str]] = {}
    md5_count: dict[str, int] = {}
    for r in ok:
        md5_groups.setdefault(r["md5"], set()).add(key_of[r["image_name"]])
        md5_count[r["md5"]] = md5_count.get(r["md5"], 0) + 1
    byte_dup_groups = [m for m, c in md5_count.items() if c > 1]
    shared_across = [m for m in byte_dup_groups if len(md5_groups[m]) > 1]

    per_product = df[df["image_name"] != ""].groupby("product_key").size()
    w = np.array([r["width"] for r in ok])
    h = np.array([r["height"] for r in ok])
    b = np.array([r["bytes"] for r in ok])

    def dist(a):
        return {q: float(np.percentile(a, p)) for q, p in [("min", 0), ("p05", 5), ("median", 50), ("p95", 95), ("max", 100)]}

    return {
        "rows": int(len(df)),
        "rows_without_image_name": int((df["image_name"] == "").sum()),
        "unique_brands": int(df["brand"].str.lower().nunique()),
        "unique_products": int(df["product_key"].nunique()),
        "unique_lines": int(df["line_key"].nunique()),
        "images_referenced": int(len(recs)),
        "images_ok": int(len(ok)),
        "images_failed": int(len(bad)),
        "failure_examples": [r["error"] for r in bad[:10]],
        "images_per_product": {str(k): int(v) for k, v in per_product.value_counts().sort_index().items()},
        "width_px": dist(w),
        "height_px": dist(h),
        "file_bytes": dist(b),
        "formats": {k: int(v) for k, v in zip(*np.unique([r["format"] for r in ok], return_counts=True), strict=True)},
        "byte_identical_groups": len(byte_dup_groups),
        "byte_identical_images": int(sum(md5_count[m] for m in byte_dup_groups)),
        "byte_identical_groups_shared_by_different_products": len(shared_across),
        "byte_identical_images_shared_by_different_products": int(sum(md5_count[m] for m in shared_across)),
        "near_duplicate_rule": f"phash 64 bit, Hamming distance <= {NEAR_DUP_BITS}",
        "near_duplicate_pairs": len(pairs),
        "near_duplicate_pairs_identical_hash": int(exact),
        "near_duplicate_pairs_same_product": int(same_product),
        "near_duplicate_pairs_different_product": int(len(pairs) - same_product),
        "images_in_any_near_duplicate_pair": int(len({a for a, _, _ in pairs} | {b for _, b, _ in pairs})),
    }


def main() -> int:
    df = load_catalog(RAW / "perfumes.csv")
    store = ZipImages(RAW / "images.zip")
    scan(df, store)
    summary = summarize(df)
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
