"""One shot guard for the final test evaluation.

The lock file records the git commit and time of the single test run. If it
exists, the evaluation refuses to run. There is deliberately no override flag;
rerunning requires the project owner to authorize it and remove the file.
"""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path


class SplitAlreadyEvaluated(RuntimeError):
    pass


def git_sha(repo: Path) -> str:
    out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True)
    return out.stdout.strip()


def git_clean(repo: Path) -> bool:
    out = subprocess.run(["git", "status", "--porcelain"], cwd=repo, capture_output=True, text=True, check=True)
    return out.stdout.strip() == ""


def acquire(lock_path: Path, sha: str) -> dict:
    """Create the lock or raise SplitAlreadyEvaluated if it already exists."""
    lock_path = Path(lock_path)
    record = {"git_sha": sha, "started_utc": datetime.now(UTC).isoformat(timespec="seconds")}
    try:
        with lock_path.open("x") as f:  # exclusive create: fails if present
            json.dump(record, f, indent=2)
            f.write("\n")
    except FileExistsError as e:
        existing = lock_path.read_text()
        raise SplitAlreadyEvaluated(f"test split already evaluated: {existing.strip()}") from e
    return record
