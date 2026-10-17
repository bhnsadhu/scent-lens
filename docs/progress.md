# ScentLens progress

Read docs/PLAN.md and this file before doing anything at every resume.

## Status

- [x] Repo check: origin bhnsadhu/scent-lens, public, default branch main, only README.md present
- [x] docs/PLAN.md written verbatim, MIT license, .gitignore (data/ and artifacts/ ignored); pushed
- [x] Phase 0: preflight and setup (STOP A cleared 2026-10-15; venv, pins, layout, 36 unit tests, CI)
- [x] Phase 1: data (download rev 9d2db452, inventory, cleaning to 25,748 images / 24,642 products, data_notes.md, spot check 50)
- [x] Phase 2: embeddings and synthetic baselines (results/synthetic_eval.md, charts in results/)
- [ ] Phase 3: bottle coverage (STOP C done: 43 bottles; at STOP C2)
- [ ] Phase 4: preregistration and photo protocol (STOP D)
- [ ] Phase 5: ingest and splits
- [ ] Phase 6: fine tune
- [ ] Phase 7: confidence and abstention
- [ ] Phase 8: final evaluation, once
- [ ] Phase 9: demo (STOP E, STOP F)
- [ ] Phase 10: readme and finish

## Current stop

STOP C2: user reviews data/real/match_review.csv (decisions prefilled with suggestions; b042 needs pick1 or pick2; b016 ambiguous unless user confirms). Reply word: matches.
On reply: `.venv/bin/python scripts/match_bottles.py apply`, report counts (gate: at least 30 matched), then phase 4 (preregistration, photo protocol, shot list), then STOP D.

## Next action

Phase 2 is running in the background: `scripts/embed_catalog.py --backbone dinov2`, then `scripts/synth_embed.py`, then `scripts/synth_eval.py` (log logs/phase2.log). All resumable: rerun the same commands; done chunks are skipped.
After it: read results/synthetic_eval.md, commit results and charts, record headline decisions, then STOP C (bottle list).
Already written for later phases: match_bottles.py (draft/apply), make_shot_list.py, ingest_photos.py, make_splits.py, embed_photos.py, eval_dev.py, finetune module, calibration, lock.

Tools: ~/.local/bin/uv and ~/.local/bin/gh (not on PATH). Run python as .venv/bin/python. Set PYTORCH_ENABLE_MPS_FALLBACK=1, NO_ALBUMENTATIONS_UPDATE=1, and HF_HUB_OFFLINE=1 (weights cached; without it the hub check can stall a run for minutes).
Disk: about 8.5 GB free. Watch it.

## Seeds

- 20261015: synthetic query sample (results/synthetic_query_ids.csv), augmentation seeds (20261015 + 100 * query_index + condition index), bootstrap, gallery sweep draws (20261015 + size), split manifest.
- 7: pandas random_state for the 50 image spot check.
