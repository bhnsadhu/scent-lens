# ScentLens progress

Read docs/PLAN.md and this file before doing anything at every resume.

## November 14, 2026: conserve the free hosting allowance

- Modal's public endpoint returns HTTP 404 with `workspace ... is disabled`. The custom domain continues to serve the paused-demo page. No paid resource or billing change was introduced.
- The demo now initializes recognition on the first submitted photo rather than on page load. The UI and public API share one initialization and serialize inference.
- A bounded in-memory cache reuses eight recent results for identical cleaned pixels for up to ten minutes. It retains a digest and result metadata, never photo pixels. Failed calls are retried normally. The container idle window is reduced from 300 to 30 seconds.
- The recognizer's model, calibration, thresholds, frozen test split and published accuracy numbers are unchanged. Runtime tests use synthetic fixtures. Hosted validation remains unavailable while the workspace is disabled.
- Validation: lint and 70 non-torch unit tests passed; two torch-marked tests were deselected. The deploy bundle includes the runtime. Importing the demo with a deliberately absent export succeeded without constructing a recognizer. A CPU/RapidOCR regression on development photo `b006__online__1` produced the same decision and matches as the direct recognizer, with only one model call across two identical requests. This check reused the photo's recorded detection box because the local OWLv2 cache was unavailable; it is not a new detector or accuracy evaluation. Environment: Python 3.11.13, torch 2.14.1, transformers 5.20.0, Gradio 6.30.0 and Pillow 12.3.0.

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

The recognition project is complete, but live hosting is paused by the free allowance. Runtime improvements can be validated locally. The separate private Accord scanner shares the same Modal workspace and is also subject to that allowance.

## Next action

Verify the runtime deployment after Modal accepts work again, then restore the custom domain's live rewrite only after an identification request succeeds. The deployment workflow runs on qualifying pushes to main, but cannot reactivate a disabled workspace. The model export and sample photos are not in git: after a new export or new samples, run `scripts/upload_demo_data.sh` to put them in the Modal Volume `scent-lens-data`. By hand: `.venv/bin/python scripts/build_space.py bhnsadhu && .venv/bin/python -m modal deploy deploy/modal_app.py`.

Tools: ~/.local/bin/uv, ~/.local/bin/gh and ~/.local/bin/modal (not on PATH). Run python as .venv/bin/python. Set PYTORCH_ENABLE_MPS_FALLBACK=1, NO_ALBUMENTATIONS_UPDATE=1, and HF_HUB_OFFLINE=1 (weights cached; without it the hub check can stall a run for minutes).

## Seeds

- 20261015: synthetic query sample (results/synthetic_query_ids.csv), augmentation seeds (20261015 + 100 * query_index + condition index), bootstrap, gallery sweep draws (20261015 + size), split manifest.
- 7: pandas random_state for the 50 image spot check.
