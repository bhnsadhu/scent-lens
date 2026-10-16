# ScentLens progress

Read docs/PLAN.md and this file before doing anything at every resume.

## Status

- [x] Repo check: origin bhnsadhu/scent-lens, public, default branch main, only README.md present
- [x] docs/PLAN.md written verbatim, MIT license, .gitignore (data/ and artifacts/ ignored); pushed
- [x] Phase 0: preflight and setup (STOP A cleared 2026-10-15; venv, pins, layout, 36 unit tests, CI)
- [ ] Phase 1: data (IN PROGRESS: downloading via scripts/download_data.py, log logs/download.log)
- [ ] Phase 2: embeddings and synthetic baselines
- [ ] Phase 3: bottle coverage (STOP C, STOP C2)
- [ ] Phase 4: preregistration and photo protocol (STOP D)
- [ ] Phase 5: ingest and splits
- [ ] Phase 6: fine tune
- [ ] Phase 7: confidence and abstention
- [ ] Phase 8: final evaluation, once
- [ ] Phase 9: demo (STOP E, STOP F)
- [ ] Phase 10: readme and finish

## Current stop

None. STOP A cleared.

## Next action

Phase 1. Download is resumable: rerun `.venv/bin/python scripts/download_data.py` (skips complete files). Then inspect perfumes.csv columns, write the column mapping, inventory script, cleaning.

Tools: ~/.local/bin/uv and ~/.local/bin/gh (not on PATH). Run python as .venv/bin/python. Set PYTORCH_ENABLE_MPS_FALLBACK=1 and NO_ALBUMENTATIONS_UPDATE=1.

## Seeds

None used yet.
