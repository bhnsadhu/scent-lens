#!/usr/bin/env bash
# Upload the demo's model export and approved sample photos to the Modal Volume that the
# deployed app reads (deploy/modal_app.py). Neither is in git, so GitHub Actions cannot
# ship them; it deploys only the code. Run from a Mac with the artifacts, once, and again
# whenever the export or the approved samples change (a redeploy is not needed after).
set -euo pipefail
cd "$(dirname "$0")/.."

MODAL=(.venv/bin/python -m modal)
BUNDLE=artifacts/space

.venv/bin/python scripts/build_space.py bhnsadhu
"${MODAL[@]}" volume create scent-lens-data 2>/dev/null || true  # already there after the first run
"${MODAL[@]}" volume put --force scent-lens-data "$BUNDLE/export" /export
"${MODAL[@]}" volume put --force scent-lens-data "$BUNDLE/samples" /samples
echo "Uploaded. The app reads them on its next start."
