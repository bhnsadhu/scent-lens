# scentlens.bhanusadhu.com

A Vercel project with no build. The project `scentlens` is connected to this repository with
its Root Directory set to `deploy/domain`, so every push to `main` redeploys it; with the
root left at `.`, a push would deploy the whole repository and the domain would 404.

## Right now: a status page

The Modal demo (`deploy/modal_app.py`) is paused because the free Modal credit for the month
is used up, so `index.html` explains that and shows the recorded demo (`demo.gif`) instead of
the raw hosting error. `vercel.json` has no rewrite.

## Turning the live demo back on

When Modal is running again, serve it under this domain by restoring the rewrite:

```sh
cp vercel.live.json vercel.json
git add vercel.json && git commit -m "deploy: serve the live demo again" && git push
```

`vercel.live.json` is the one rewrite that serves the Modal demo under scentlens.bhanusadhu.com,
so the address stays on the domain. To pause again, put the empty `vercel.json` back.
