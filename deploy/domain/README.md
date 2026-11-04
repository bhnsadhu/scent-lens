# scentlens.bhanusadhu.com

A Vercel project with no build: one rewrite that serves the Modal demo
(`deploy/modal_app.py`) under scentlens.bhanusadhu.com, so the address stays on the
domain. The Vercel project `scentlens` is connected to this repository with its Root
Directory set to `deploy/domain`, so every push to `main` redeploys the rewrite; with the
root left at `.`, a push would deploy the whole repository and the domain would 404.
