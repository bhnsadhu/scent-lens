from PIL import Image

from scent_lens.crop import apply_crop, choose_box, padded_box


def test_padded_box_widens_and_clips():
    assert padded_box((100, 200, 200, 400), (1000, 1000)) == (85, 170, 215, 430)
    assert padded_box((0, 0, 100, 100), (110, 105)) == (0, 0, 110, 105)


def test_choose_box_takes_best_or_falls_back():
    boxes = [(0, 0, 10, 10), (100, 200, 200, 400)]
    assert choose_box(boxes, [0.3, 0.6], (1000, 1000)) == (85, 170, 215, 430)
    assert choose_box(boxes, [0.05, 0.09], (1000, 1000)) is None
    assert choose_box([], [], (1000, 1000)) is None


def test_apply_crop():
    img = Image.new("RGB", (50, 40))
    assert apply_crop(img, None) is img
    assert apply_crop(img, (10, 5, 30, 25)).size == (20, 20)
