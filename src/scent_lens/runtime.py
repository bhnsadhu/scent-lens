"""Load the demo model on demand and reuse recent results without retaining photos."""

from __future__ import annotations

import hashlib
import time
from collections import OrderedDict
from collections.abc import Callable
from copy import deepcopy
from dataclasses import replace
from threading import Lock
from typing import TYPE_CHECKING

from PIL import Image

from scent_lens.photos import clean_pil

if TYPE_CHECKING:
    from scent_lens.recognize import Result


class RecognitionRuntime:
    """One lazy recognizer per process, with bounded, expiring result reuse.

    Only a digest and the result are kept in memory, never the uploaded pixels.
    A lock shares initialization and inference across the Gradio UI and API.
    The underlying recognizer and its decision settings are unchanged.
    """

    def __init__(self, load: Callable[[], Callable[[Image.Image], Result]], *,
                 max_results: int = 8, ttl_seconds: float = 600,
                 clock: Callable[[], float] = time.monotonic):
        if max_results < 1 or ttl_seconds <= 0:
            raise ValueError("Result limits must be positive")
        self.load = load
        self.max_results = max_results
        self.ttl_seconds = ttl_seconds
        self.clock = clock
        self._recognizer = None
        self._results: OrderedDict[str, tuple[float, Result]] = OrderedDict()
        self._lock = Lock()

    def __call__(self, image: Image.Image) -> Result:
        started = self.clock()
        # Hash the same normalized pixels the recognizer consumes. Orientation
        # and transparency are handled, metadata is stripped, and size is bounded.
        cleaned = clean_pil(image)
        digest = hashlib.sha256(f"{cleaned.size[0]}x{cleaned.size[1]}:RGB:".encode())
        digest.update(cleaned.tobytes())
        key = digest.hexdigest()
        with self._lock:
            now = self.clock()
            for old_key, (expires, _) in list(self._results.items()):
                if expires <= now:
                    del self._results[old_key]
            hit = self._results.get(key)
            if hit is not None:
                self._results.move_to_end(key)
                elapsed = self.clock() - started
                return replace(deepcopy(hit[1]), seconds=elapsed, timings={"reuse": elapsed})
            if self._recognizer is None:
                self._recognizer = self.load()
            result = self._recognizer(cleaned)
            self._results[key] = (self.clock() + self.ttl_seconds, deepcopy(result))
            if len(self._results) > self.max_results:
                self._results.popitem(last=False)
            return result
