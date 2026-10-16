# ScentLens decisions

Each entry: date, decision, reason.

- 2026-10-15. Dataset files observed on the hub (revision 9d2db452): perfumes.csv (12.1 MB, the file the card's data_files points to), images.zip (875 MB), plus parquet, xls, xlsx, sqlite, json, and jsonl copies of the metadata. Use perfumes.csv as the metadata source. Card tags: license mit, not-for-all-audiences.
- 2026-10-15. Read catalog images straight from images.zip instead of extracting, to save about 0.9 GB of disk.
- 2026-10-15. Disk budget estimate about 10 GB working set, 15 GB free requested for headroom: venv about 3 GB, model weights about 2.5 GB (dinov2 base, siglip 2 base, open_clip ViT B 32), dataset about 0.9 GB, embeddings and caches under 0.5 GB, fine tune checkpoints about 1 GB, real photos 1 to 2 GB, transient pip and build space about 2 GB (installs use no pip cache).
