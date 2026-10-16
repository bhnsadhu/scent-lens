"""Synthetic photo conditions applied to catalog images.

Each function maps (PIL image, numpy Generator, strength) to a PIL image and is
deterministic given the generator state. strength 1.0 is the evaluation
setting; the fine tune samples strengths around it.
"""

from __future__ import annotations

import io
from collections.abc import Callable

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def _arr(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("RGB"), dtype=np.float32)


def _img(a: np.ndarray) -> Image.Image:
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def foreground_mask(img: Image.Image, thresh: float = 245.0) -> np.ndarray:
    """Pixels that are not near white background. Catalog shots sit on white."""
    a = _arr(img)
    mask = (a.min(axis=2) < thresh).astype(np.float32)
    m = Image.fromarray((mask * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(5))
    m = m.filter(ImageFilter.GaussianBlur(2))
    return np.asarray(m, dtype=np.float32) / 255.0


def _fg_box(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask > 0.5)
    if len(xs) == 0:
        h, w = mask.shape
        return 0, 0, w, h
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def glare(img: Image.Image, rng: np.random.Generator, strength: float = 1.0) -> Image.Image:
    """Specular highlight: one or two bright elongated blobs over the bottle."""
    a = _arr(img)
    h, w = a.shape[:2]
    x0, y0, x1, y1 = _fg_box(foreground_mask(img))
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    light = np.zeros((h, w), np.float32)
    for _ in range(rng.integers(1, 3)):
        cx = rng.uniform(x0, x1)
        cy = rng.uniform(y0, y0 + 0.7 * (y1 - y0))
        sx = rng.uniform(0.06, 0.14) * w * (0.7 + 0.3 * strength)
        sy = sx * rng.uniform(1.5, 3.5)
        theta = rng.uniform(-0.6, 0.6)
        dx, dy = xx - cx, yy - cy
        u = dx * np.cos(theta) + dy * np.sin(theta)
        v = -dx * np.sin(theta) + dy * np.cos(theta)
        light += np.exp(-0.5 * ((u / sx) ** 2 + (v / sy) ** 2))
    light = np.clip(light, 0, 1) * min(1.0, 0.85 * strength)
    out = a + (255.0 - a) * light[..., None]
    return _img(out)


def blur(img: Image.Image, rng: np.random.Generator, strength: float = 1.0) -> Image.Image:
    """Motion or defocus blur."""
    if rng.random() < 0.5:
        return img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(2.0, 4.0) * strength))
    k = max(3, int(round(rng.uniform(9, 17) * strength)) | 1)
    angle = rng.uniform(0, 180)
    kernel = Image.new("L", (k, k), 0)
    ImageDraw.Draw(kernel).line([(0, k // 2), (k - 1, k // 2)], fill=255, width=1)
    kernel = np.asarray(kernel.rotate(angle), dtype=np.float32)
    kernel /= kernel.sum()
    a = _arr(img)
    pad = k // 2
    p = np.pad(a, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
    out = np.zeros_like(a)
    for dy, dx in zip(*np.nonzero(kernel), strict=True):
        out += kernel[dy, dx] * p[dy : dy + a.shape[0], dx : dx + a.shape[1]]
    return _img(out)


def low_light(img: Image.Image, rng: np.random.Generator, strength: float = 1.0) -> Image.Image:
    """Dim exposure: darker, lower contrast, warm cast, sensor noise."""
    a = _arr(img) / 255.0
    gain = 1.0 - rng.uniform(0.55, 0.75) * min(strength, 1.2)
    gamma = 1.0 + rng.uniform(0.3, 0.7) * strength
    a = (a**gamma) * gain
    a = a * np.array([1.0, rng.uniform(0.88, 0.97), rng.uniform(0.7, 0.9)], np.float32)
    noise = rng.normal(0, 0.03 * strength, a.shape).astype(np.float32)
    return _img((a + noise) * 255.0)


def perspective(img: Image.Image, rng: np.random.Generator, strength: float = 1.0) -> Image.Image:
    """Viewing angle: random perspective warp and slight rotation."""
    w, h = img.size
    d = 0.18 * strength

    def j(v: float, s: float) -> float:
        return v + rng.uniform(-d, d) * s

    quad = (j(0, w), j(0, h), j(0, w), j(h, h), j(w, w), j(h, h), j(w, w), j(0, h))
    out = img.convert("RGB").transform((w, h), Image.QUAD, quad, Image.BICUBIC, fillcolor=(255, 255, 255))
    return out.rotate(rng.uniform(-12, 12) * strength, Image.BICUBIC, fillcolor=(255, 255, 255))


def clutter(img: Image.Image, rng: np.random.Generator, strength: float = 1.0) -> Image.Image:
    """Busy background: procedural shelf or desk scene behind the bottle."""
    a = _arr(img)
    h, w = a.shape[:2]
    mask = foreground_mask(img)
    base = rng.uniform(40, 200, size=3)
    grad = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
    bg = base * (0.7 + 0.5 * grad) + rng.normal(0, 8, (h, w, 3))
    canvas = _img(bg)
    draw = ImageDraw.Draw(canvas)
    for _ in range(int(rng.integers(8, 20) * strength)):
        x, y = rng.uniform(0, w), rng.uniform(0, h)
        rw, rh = rng.uniform(0.05, 0.4) * w, rng.uniform(0.05, 0.5) * h
        color = tuple(int(c) for c in rng.uniform(0, 255, 3))
        if rng.random() < 0.6:
            draw.rectangle([x, y, x + rw, y + rh], fill=color)
        else:
            draw.ellipse([x, y, x + rw, y + rh], fill=color)
    # shelf edge
    sy = rng.uniform(0.6, 0.9) * h
    draw.rectangle([0, sy, w, h], fill=tuple(int(c) for c in rng.uniform(60, 180, 3)))
    bg = np.asarray(canvas.filter(ImageFilter.GaussianBlur(rng.uniform(0.5, 2.5))), np.float32)
    out = a * mask[..., None] + bg * (1 - mask[..., None])
    return _img(out)


def occlusion(img: Image.Image, rng: np.random.Generator, strength: float = 1.0) -> Image.Image:
    """Partial cover: a hand like blob or a solid object over part of the label."""
    out = img.convert("RGB").copy()
    x0, y0, x1, y1 = _fg_box(foreground_mask(img))
    bw, bh = x1 - x0, y1 - y0
    frac = min(0.6, rng.uniform(0.25, 0.4) * strength)
    ow, oh = bw * rng.uniform(0.5, 1.1), bh * frac
    ox = x0 + rng.uniform(-0.2, 0.6) * bw
    oy = y0 + rng.uniform(0.2, 0.8) * bh - oh / 2
    if rng.random() < 0.6:
        color = (int(rng.uniform(150, 230)), int(rng.uniform(100, 180)), int(rng.uniform(80, 150)))
    else:
        color = tuple(int(c) for c in rng.uniform(0, 255, 3))
    layer = Image.new("L", out.size, 0)
    ImageDraw.Draw(layer).rounded_rectangle([ox, oy, ox + ow, oy + oh], radius=int(oh / 2), fill=255)
    layer = layer.filter(ImageFilter.GaussianBlur(3))
    out.paste(Image.new("RGB", out.size, color), mask=layer)
    return out


def jpeg(img: Image.Image, rng: np.random.Generator, strength: float = 1.0) -> Image.Image:
    """Heavy compression after downscaling, like a forwarded chat photo."""
    w, h = img.size
    scale = max(0.25, 1.0 - 0.45 * strength)
    small = img.convert("RGB").resize((max(32, int(w * scale)), max(32, int(h * scale))), Image.BILINEAR)
    quality = int(np.clip(rng.uniform(8, 20) / max(strength, 0.3), 5, 60))
    buf = io.BytesIO()
    small.save(buf, format="JPEG", quality=quality)
    return Image.open(io.BytesIO(buf.getvalue())).convert("RGB").resize((w, h), Image.BILINEAR)


def identity(img: Image.Image, rng: np.random.Generator, strength: float = 1.0) -> Image.Image:
    return img.convert("RGB")


CONDITIONS: dict[str, Callable[[Image.Image, np.random.Generator, float], Image.Image]] = {
    "none": identity,
    "glare": glare,
    "blur": blur,
    "low_light": low_light,
    "perspective": perspective,
    "clutter": clutter,
    "occlusion": occlusion,
    "jpeg": jpeg,
}


def apply(name: str, img: Image.Image, seed: int, strength: float = 1.0) -> Image.Image:
    return CONDITIONS[name](img, np.random.default_rng(seed), strength)


def random_mix(img: Image.Image, rng: np.random.Generator, strength: float = 1.0, max_ops: int = 3) -> Image.Image:
    """One to max_ops random conditions in a row, for training queries."""
    names = [n for n in CONDITIONS if n != "none"]
    for name in rng.choice(names, size=int(rng.integers(1, max_ops + 1)), replace=False):
        img = CONDITIONS[str(name)](img, rng, float(strength * rng.uniform(0.6, 1.2)))
    return img
