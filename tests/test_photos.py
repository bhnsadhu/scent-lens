from PIL import Image

from scent_lens.photos import clean_image, parse_name, save_clean


def test_parse_name():
    assert parse_name("b001__clean__1.HEIC").stem == "b001__clean__1"
    p = parse_name("raw/b014__partial__2.jpg")
    assert (p.bottle_id, p.condition, p.n) == ("b014", "partial", 2)
    assert parse_name("b001_clean_1.jpg") is None
    assert parse_name("b001__sunny__1.jpg") is None
    assert parse_name("b001__clean__1.txt") is None


def test_exif_and_gps_are_stripped(tmp_path):
    src = tmp_path / "in.jpg"
    img = Image.new("RGB", (3000, 2000), (200, 100, 50))
    exif = Image.Exif()
    exif[0x010F] = "PhoneMaker"  # Make
    exif[0x0112] = 6  # orientation: rotate 90
    exif[0x8825] = {1: "N", 2: (41.0, 52.0, 30.0), 3: "W", 4: (87.0, 37.0, 0.0)}  # GPS
    img.save(src, exif=exif)
    assert Image.open(src).getexif()

    out = clean_image(src)
    assert max(out.size) == 1024
    assert out.size == (683, 1024)  # orientation applied before stripping
    dst = tmp_path / "out.jpg"
    save_clean(out, dst)
    reread = Image.open(dst)
    assert not reread.getexif()
    assert "exif" not in reread.info
    assert "icc_profile" not in reread.info
