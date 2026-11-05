#!/usr/bin/env bash
# One-time setup for automatic demo deploys (.github/workflows/deploy.yml), run on the Mac
# that holds the artifacts and is logged in to Modal (~/.modal.toml) and GitHub (gh):
#   1. uploads the model export and sample photos to the Modal Volume (upload_demo_data.sh)
#   2. stores this Mac's Modal token as the repository secrets MODAL_TOKEN_ID / MODAL_TOKEN_SECRET
#      (piped straight from ~/.modal.toml to gh, never printed)
#   3. pushes this branch to main, which runs the first automatic deploy
set -euo pipefail
cd "$(dirname "$0")/.."
REPO=bhnsadhu/scent-lens

GH=$(command -v gh || echo "$HOME/.local/bin/gh")
[ -x "$GH" ] || { echo "gh (GitHub CLI) not found. Install it: brew install gh, then gh auth login" >&2; exit 1; }
"$GH" auth status >/dev/null 2>&1 || { echo "gh is not logged in. Run: $GH auth login" >&2; exit 1; }

echo "1/3 Uploading the model and sample photos to Modal..."
scripts/upload_demo_data.sh

echo "2/3 Saving the Modal token as GitHub secrets..."
modal_token() {  # $1: token_id or token_secret, from the active profile in ~/.modal.toml
  .venv/bin/python - "$1" <<'PY'
import sys, tomllib
from pathlib import Path
profiles = tomllib.loads((Path.home() / ".modal.toml").read_text())
active = next((p for p in profiles.values() if p.get("active")), next(iter(profiles.values())))
sys.stdout.write(active[sys.argv[1]])
PY
}
modal_token token_id | "$GH" secret set MODAL_TOKEN_ID --repo "$REPO"
modal_token token_secret | "$GH" secret set MODAL_TOKEN_SECRET --repo "$REPO"

echo "3/3 Pushing to main (this starts the first automatic deploy)..."
git push origin HEAD:main
echo "Done. Watch the deploy: https://github.com/$REPO/actions/workflows/deploy.yml"
