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
- Free disk: was 456 MiB; user deleted old display logs, now 16 GiB free. Resolved.
- Python: installed uv 0.8.22 and CPython 3.11.13 into ~/.local (uv managed). Resolved.
- gh: installed gh 2.80.0 into ~/.local/bin; logged in as bhnsadhu, scopes repo and workflow; gh repo view says PUBLIC, main. Resolved.
- Hugging Face: no token file yet. Still open; only the user can create a write token.
- git push over https works (osxkeychain).
- Hardware: Apple M3, 16 GB RAM, MPS expected.

## Next action

After "ready": recheck disk, python, gh auth, HF whoami with write role. Use ~/.local/bin/uv and ~/.local/bin/gh (not on PATH). Then create .venv, pinned requirements, layout, CI.

## Seeds

None used yet.
