# ScentLens progress

Read docs/PLAN.md and this file before doing anything at every resume.

## Status

- [x] Repo check: origin bhnsadhu/scent-lens, public, default branch main, only README.md present
- [x] docs/PLAN.md written verbatim, MIT license, .gitignore (data/ and artifacts/ ignored); pushed
- [x] Phase 0: preflight and setup (STOP A cleared 2026-10-15; venv, pins, layout, 36 unit tests, CI)
- [x] Phase 1: data (download rev 9d2db452, inventory, cleaning to 25,748 images / 24,642 products, data_notes.md, spot check 50)
- [x] Phase 2: embeddings and synthetic baselines (results/synthetic_eval.md, charts in results/)
- [x] Phase 3: bottle coverage (47 bottles: 36 matched, 11 absent; results/bottle_manifest.csv)
- [x] Phase 4: preregistration (frozen by test), photo protocol, shot list, labels template
- [ ] Phase 5: ingest and splits
- [ ] Phase 6: fine tune
- [ ] Phase 7: confidence and abstention
- [ ] Phase 8: final evaluation, once
- [ ] Phase 9: demo (STOP E, STOP F)
- [ ] Phase 10: readme and finish

## Current stop

STOP D: user takes photos into data/real/raw (one folder per bottle, 6 or 12 photos in protocol order, or exact file names). Reply word: photos.
On reply: `.venv/bin/python scripts/ingest_photos.py`, view a contact sheet per bottle locally to catch ordering mistakes, gate at 30 matched bottles with photos (else STOP D again), then `scripts/make_splits.py`, commit the manifest, then `scripts/embed_photos.py` and `scripts/eval_dev.py`.

## Next action

Phase 2 is running in the background: `scripts/embed_catalog.py --backbone dinov2`, then `scripts/synth_embed.py`, then `scripts/synth_eval.py` (log logs/phase2.log). All resumable: rerun the same commands; done chunks are skipped.
After it: read results/synthetic_eval.md, commit results and charts, record headline decisions, then STOP C (bottle list).
Already written for later phases: match_bottles.py (draft/apply), make_shot_list.py, ingest_photos.py, make_splits.py, embed_photos.py, eval_dev.py, finetune module, calibration, lock.

Tools: ~/.local/bin/uv and ~/.local/bin/gh (not on PATH). Run python as .venv/bin/python. Set PYTORCH_ENABLE_MPS_FALLBACK=1, NO_ALBUMENTATIONS_UPDATE=1, and HF_HUB_OFFLINE=1 (weights cached; without it the hub check can stall a run for minutes).
Disk: about 8.5 GB free. Watch it.

## Seeds

- 20261015: synthetic query sample (results/synthetic_query_ids.csv), augmentation seeds (20261015 + 100 * query_index + condition index), bootstrap, gallery sweep draws (20261015 + size), split manifest.
- 7: pandas random_state for the 50 image spot check.
