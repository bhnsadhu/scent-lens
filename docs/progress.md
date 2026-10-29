# ScentLens progress

Read docs/PLAN.md and this file before doing anything at every resume.

## Status

- [x] Repo check: origin bhnsadhu/scent-lens, public, default branch main, only README.md present
- [x] docs/PLAN.md written verbatim, MIT license, .gitignore (data/ and artifacts/ ignored); pushed
- [x] Phase 0: preflight and setup (STOP A cleared 2026-10-15; venv, pins, layout, 36 unit tests, CI)
- [x] Phase 1: data (download rev 9d2db452, inventory, cleaning to 25,748 images / 24,642 products, data_notes.md, spot check 50)
- [x] Phase 2: embeddings and synthetic baselines (results/synthetic_eval.md, charts in results/)
- [x] Phase 3: bottle coverage (63 bottles: 49 matched, 14 absent; results/bottle_manifest.csv)
- [x] Phase 4: preregistration (frozen by test), photo protocol, shot list, labels template
- [x] Phase 5: ingest and splits (185 images of 55 bottles: 141 phone photos in two rounds plus 44 online product images; split 27 dev / 28 test bottles, results/split_manifest.csv, extended for the second round; deviations noted in the preregistration 2026-10-28)
- [x] Phase 6: fine tune (grid of 4 runs, results/runs.jsonl; rule 4 picks lr 1e-4 strength 1.0 epoch 1: proxy top 1 48.1 vs frozen 34.6); rule 7 adds a bottle crop of queries (53.1 vs 34.6 full gallery)
- [x] Phase 7: confidence and abstention (logistic calibrator, threshold 0.641: answers 33 of 101 development photos, all correct, abstains on every absent bottle photo; results/calibrator.json)
- [x] Phase 8: final evaluation, once (2026-10-29, results/test_results.md: final system 66.7 strict top 1, H2 H4 H5 H7 supported, H1 inconclusive, H3 not tested, H6 refuted at 83.0 selective accuracy)
- [x] Version 2 after the test (results/v2_results.md: rotations, label reading, answer guard and clear label rule; web set of 42 openly licensed photos)
- [x] Phase 9: demo, live on Modal's free tier (deploy/modal_app.py); recognizer API in docs/integration.md
- [x] Phase 10: README, pushed 2026-10-29, CI green

## Current stop

ScentLens is complete. Next, outside this repo: Accord, which calls the recognizer as described in docs/integration.md.

## Next action

None. Redeploy the demo after changes: `.venv/bin/python scripts/build_space.py bhnsadhu && ~/.local/bin/modal deploy deploy/modal_app.py`.

Tools: ~/.local/bin/uv, ~/.local/bin/gh and ~/.local/bin/modal (not on PATH). Run python as .venv/bin/python. Set PYTORCH_ENABLE_MPS_FALLBACK=1, NO_ALBUMENTATIONS_UPDATE=1, and HF_HUB_OFFLINE=1 (weights cached; without it the hub check can stall a run for minutes).

## Seeds

- 20261015: synthetic query sample (results/synthetic_query_ids.csv), augmentation seeds (20261015 + 100 * query_index + condition index), bootstrap, gallery sweep draws (20261015 + size), split manifest.
- 7: pandas random_state for the 50 image spot check.
