import json

import pytest

from scent_lens.lock import SplitAlreadyEvaluated, acquire


def test_lock_allows_exactly_one_run(tmp_path):
    lock = tmp_path / "test_lock.json"
    rec = acquire(lock, "abc123")
    assert json.loads(lock.read_text())["git_sha"] == "abc123"
    assert rec["git_sha"] == "abc123"
    with pytest.raises(SplitAlreadyEvaluated):
        acquire(lock, "def456")
    # the first record is untouched by the refused attempt
    assert json.loads(lock.read_text())["git_sha"] == "abc123"
