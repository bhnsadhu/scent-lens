"""The preregistration is frozen: its original text may only be followed by appended correction notes."""

import hashlib
from pathlib import Path

PREREG = Path(__file__).resolve().parents[1] / "docs" / "experiments" / "preregistration.md"
ORIGINAL_BYTES = 12024
ORIGINAL_SHA256 = "42fa9c48b12eb38b0f4e2cbf24632cd4fdc0caf9f23a999b6a62efdb30884373"


def test_original_text_unchanged():
    data = PREREG.read_bytes()
    assert len(data) >= ORIGINAL_BYTES
    assert hashlib.sha256(data[:ORIGINAL_BYTES]).hexdigest() == ORIGINAL_SHA256
