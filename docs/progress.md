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
- [ ] Phase 9: demo. Done: export (scripts/export_model.py), recognizer (scent_lens.recognize, docs/integration.md), end to end check (results/recognizer_*.md), RapidOCR for non macOS hosts, Gradio app (app/app.py, API endpoint identify), Space bundler (scripts/build_space.py). Waiting on STOP E and STOP F.
- [ ] Phase 10: readme written (README.md); finish after the Space is live (verify a sample on the Space, CPU latency there, push to GitHub with approval)

## Current stop

STOP E done (app/samples.txt). STOP F done: Hugging Face now charges for Gradio Spaces, so the demo runs on Modal's free tier (deploy/modal_app.py, https://bhnsadhu--scent-lens-web.us-east.modal.run). Remaining: push to GitHub with the user's approval.

## Next action

Write the README (phase 10 spec in docs/PLAN.md), with ScentLens as a standalone project whose recognizer will later serve Accord (docs/integration.md). Run the demo locally: `SCENTLENS_READER=rapidocr .venv/bin/python app/app.py`. Bundle the Space: `.venv/bin/python scripts/build_space.py <github user>`, which writes artifacts/space.
