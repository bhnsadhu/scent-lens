import numpy as np
from PIL import Image

from scent_lens.photos import as_saved, clean_pil
from scent_lens.recognize import decide

SETTINGS = {
    "text_weight": 0.3,
    "guard_share": 0.5,
    "threshold": 0.67,
    "calibrator": {"kind": "logistic", "feature_mean": [0.0, 0.0, 0.0], "feature_std": [1.0, 1.0, 1.0],
                   "coef": [0.0, 50.0, 0.0], "intercept": -2.0},  # confidence grows with the margin
}
BRANDS = ["a", "a", "b", "c", "d", "e"]


def _decide(scores, text=None, explained=0.0, read=False):
    text = np.zeros(len(scores)) if text is None else np.asarray(text, float)
    return decide(np.asarray(scores, float), text, explained, read, BRANDS, SETTINGS)


def test_confident_image_match_is_answered():
    answered, reason, conf, top = _decide([0.9, 0.5, 0.4, 0.3, 0.2, 0.1])
    assert (answered, reason, int(top[0])) == (True, "confident", 0)
    assert conf > SETTINGS["threshold"]


def test_close_call_abstains():
    answered, reason, conf, _ = _decide([0.80, 0.79, 0.4, 0.3, 0.2, 0.1])
    assert (answered, reason) == (False, "low confidence")
    assert conf < SETTINGS["threshold"]


def test_label_naming_another_product_blocks_a_confident_guess():
    answered, reason, _, _ = _decide([0.9, 0.5, 0.4, 0.3, 0.2, 0.1], explained=0.2, read=True)
    assert (answered, reason) == (False, "label disagrees")


def test_clear_unique_label_is_answered_even_on_a_close_call():
    text = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    answered, reason, conf, top = _decide([0.80, 0.79, 0.4, 0.3, 0.2, 0.1], text, explained=1.0, read=True)
    assert (answered, reason, conf, int(top[0])) == (True, "clear label", 1.0, 1)


def test_label_that_fits_several_products_is_not_a_clear_label():
    text = [1.0, 1.0, 0.0, 0.0, 0.0, 0.0]  # "Miss Dior" fits many products
    answered, reason, _, _ = _decide([0.80, 0.79, 0.4, 0.3, 0.2, 0.1], text, explained=1.0, read=True)
    assert reason != "clear label"
    assert not answered


def test_live_upload_takes_the_saved_photo_path():
    img = Image.new("RGB", (3000, 1500), (200, 100, 50))
    clean = clean_pil(img)
    assert max(clean.size) == 1024
    saved = as_saved(clean)
    assert saved.size == clean.size and saved.mode == "RGB"
