# ScentLens progress

Read docs/PLAN.md and this file before doing anything at every resume.

## Status

- [x] Repo check: origin bhnsadhu/scent-lens, public, default branch main, only README.md present
- [x] docs/PLAN.md written verbatim, MIT license, .gitignore (data/ and artifacts/ ignored); pushed
- [ ] Phase 0: preflight and setup (BLOCKED at STOP A)
- [ ] Phase 1: data
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

STOP A, waiting for reply word: ready.

Preflight findings (2026-10-15):
- Free disk: 456 MiB on the data volume. Need about 15 GB free.
- Python: only /usr/bin/python3 3.9.6. Need 3.11 or newer.
- gh: binary not installed (a ~/.config/gh/hosts.yml exists from an earlier install).
- Hugging Face: no token file, no HF_TOKEN, huggingface_hub not installed.
- git push over https works (osxkeychain).
- Hardware: Apple M3, 16 GB RAM, MPS expected.

## Next action

After "ready": recheck disk, python, gh auth, HF whoami with write role. If the user chose the self install route, install uv into ~/.local/bin, then Python 3.11 via uv, and gh from the official release zip into ~/.local/bin. Then create .venv, pinned requirements, layout, CI.

## Seeds

None used yet.
