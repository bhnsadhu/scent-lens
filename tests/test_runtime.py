from concurrent.futures import ThreadPoolExecutor
from threading import Event
from unittest.mock import Mock

import pytest
from PIL import Image

from scent_lens.recognize import Match, Result
from scent_lens.runtime import RecognitionRuntime


def result():
    return Result(True, "confident", 0.8, [Match("p1", "Brand", "Name", 0.9)],
                  ["Brand", "Name"], (0, 0, 10, 10), seconds=6.0, timings={"match": 6.0})


def photo(color="red", size=(20, 20)):
    return Image.new("RGB", size, color)


def test_model_loads_only_when_a_photo_is_submitted():
    recognize = Mock(side_effect=lambda _: result())
    load = Mock(return_value=recognize)
    runtime = RecognitionRuntime(load)
    load.assert_not_called()
    first = runtime(photo())
    assert first.to_dict() == result().to_dict()
    runtime(photo("blue"))
    load.assert_called_once()
    assert recognize.call_count == 2


def test_identical_pixels_reuse_a_result_without_mutable_cross_request_state():
    recognize = Mock(side_effect=lambda _: result())
    runtime = RecognitionRuntime(lambda: recognize)
    first = runtime(photo())
    first.matches[0].name = "Changed by caller"
    reused = runtime(photo())
    assert reused.matches[0].name == "Name"
    assert reused.seconds < 1
    assert "reuse" in reused.timings
    reused.text.append("Changed too")
    assert runtime(photo()).text == ["Brand", "Name"]
    recognize.assert_called_once()


def test_shape_is_part_of_the_key_even_when_pixel_bytes_are_identical():
    recognize = Mock(side_effect=lambda _: result())
    runtime = RecognitionRuntime(lambda: recognize)
    runtime(photo(size=(10, 20)))
    runtime(photo(size=(20, 10)))
    assert recognize.call_count == 2


def test_cache_expires_and_evicts_the_least_recent_result():
    now = [0.0]
    recognize = Mock(side_effect=lambda _: result())
    runtime = RecognitionRuntime(lambda: recognize, max_results=2, ttl_seconds=10,
                                 clock=lambda: now[0])
    runtime(photo("red"))
    runtime(photo("blue"))
    runtime(photo("red"))  # Red remains in use; blue should be evicted first.
    runtime(photo("green"))
    runtime(photo("red"))
    assert recognize.call_count == 3
    runtime(photo("blue"))
    assert recognize.call_count == 4
    now[0] = 10.0
    runtime(photo("blue"))
    assert recognize.call_count == 5


def test_duplicate_concurrent_requests_share_initialization_and_inference():
    entered, release = Event(), Event()

    def recognize(_):
        entered.set()
        assert release.wait(timeout=3)
        return result()

    load = Mock(return_value=Mock(side_effect=recognize))
    runtime = RecognitionRuntime(load)
    with ThreadPoolExecutor(max_workers=2) as threads:
        first = threads.submit(runtime, photo())
        assert entered.wait(timeout=3)
        second = threads.submit(runtime, photo())
        release.set()
        assert first.result(timeout=3).best.product_id == "p1"
        assert second.result(timeout=3).best.product_id == "p1"
    load.assert_called_once()
    load.return_value.assert_called_once()


def test_failed_load_or_inference_can_be_retried_without_caching_an_error():
    recognize = Mock(side_effect=[RuntimeError("inference failed"), result()])
    load = Mock(side_effect=[RuntimeError("load failed"), recognize])
    runtime = RecognitionRuntime(load)
    with pytest.raises(RuntimeError, match="load failed"):
        runtime(photo())
    with pytest.raises(RuntimeError, match="inference failed"):
        runtime(photo())
    assert runtime(photo()).best.product_id == "p1"
    assert load.call_count == 2
    assert recognize.call_count == 2


def test_model_receives_cleaned_pixels_without_private_metadata():
    recognize = Mock(side_effect=lambda _: result())
    runtime = RecognitionRuntime(lambda: recognize)
    image = photo(size=(2000, 1000))
    image.getexif()[270] = "private description"
    runtime(image)
    cleaned = recognize.call_args.args[0]
    assert cleaned.size == (1024, 512)
    assert not cleaned.getexif()
    assert cleaned.info == {}
