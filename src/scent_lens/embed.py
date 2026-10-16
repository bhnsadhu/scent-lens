"""Frozen image (and text) encoders behind one interface.

Every backbone sees the same input: the image padded to a square with white
and resized, never center cropped, so tall bottles keep their cap and base.
Embeddings are L2 normalized float32.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

BACKBONES = {
    "dinov2": "facebook/dinov2-base",
    "siglip2": "google/siglip2-base-patch16-224",
    "clip": "hf-hub:laion/CLIP-ViT-B-32-laion2B-s34B-b79K",
}


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def pad_to_square(img: Image.Image, fill: tuple[int, int, int] = (255, 255, 255)) -> Image.Image:
    w, h = img.size
    side = max(w, h)
    if w == h:
        return img
    canvas = Image.new("RGB", (side, side), fill)
    canvas.paste(img, ((side - w) // 2, (side - h) // 2))
    return canvas


def _normalize(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(x.float(), dim=-1)


@dataclass
class Encoder:
    name: str
    model: torch.nn.Module
    size: int
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    device: str
    tokenizer: object | None = None

    def preprocess(self, images: list[Image.Image]) -> torch.Tensor:
        arrs = []
        for img in images:
            img = pad_to_square(img.convert("RGB")).resize((self.size, self.size), Image.BICUBIC)
            arrs.append(np.asarray(img, dtype=np.float32) / 255.0)
        x = torch.from_numpy(np.stack(arrs)).permute(0, 3, 1, 2)
        mean = torch.tensor(self.mean).view(1, 3, 1, 1)
        std = torch.tensor(self.std).view(1, 3, 1, 1)
        return (x - mean) / std

    @torch.no_grad()
    def encode_pixels(self, pixels: torch.Tensor) -> torch.Tensor:
        pixels = pixels.to(self.device)
        if self.name == "clip":
            out = self.model.encode_image(pixels)
        elif self.name == "dinov2":
            out = self.model(pixel_values=pixels).pooler_output
        elif self.name == "siglip2":
            out = self.model.get_image_features(pixel_values=pixels)
            out = getattr(out, "pooler_output", out)
        else:
            out = self.model(pixels)
        return _normalize(out)

    def encode_images(self, images: list[Image.Image]) -> np.ndarray:
        return self.encode_pixels(self.preprocess(images)).cpu().numpy()

    @torch.no_grad()
    def encode_texts(self, texts: list[str]) -> np.ndarray:
        if self.name == "clip":
            tokens = self.tokenizer(texts).to(self.device)
            out = self.model.encode_text(tokens)
        elif self.name == "siglip2":
            # SigLIP 2 text is trained with max_length 64 padding and lowercased text
            enc = self.tokenizer(
                [t.lower() for t in texts], padding="max_length", max_length=64,
                truncation=True, return_tensors="pt",
            ).to(self.device)
            out = self.model.get_text_features(**enc)
            out = getattr(out, "pooler_output", out)
        else:
            raise ValueError(f"{self.name} has no text tower")
        return _normalize(out).cpu().numpy()


def load_encoder(name: str, device: str | None = None) -> Encoder:
    device = device or pick_device()
    if name == "clip":
        import open_clip

        model, _, preprocess = open_clip.create_model_and_transforms(BACKBONES["clip"])
        tokenizer = open_clip.get_tokenizer(BACKBONES["clip"])
        norm = next(t for t in preprocess.transforms if t.__class__.__name__ == "Normalize")
        enc = Encoder("clip", model, 224, tuple(norm.mean), tuple(norm.std), device, tokenizer)
    elif name == "dinov2":
        from transformers import AutoImageProcessor, AutoModel

        proc = AutoImageProcessor.from_pretrained(BACKBONES["dinov2"])
        model = AutoModel.from_pretrained(BACKBONES["dinov2"])
        enc = Encoder("dinov2", model, 224, tuple(proc.image_mean), tuple(proc.image_std), device)
    elif name == "siglip2":
        from transformers import AutoImageProcessor, AutoModel, AutoTokenizer

        proc = AutoImageProcessor.from_pretrained(BACKBONES["siglip2"])
        model = AutoModel.from_pretrained(BACKBONES["siglip2"])
        tokenizer = AutoTokenizer.from_pretrained(BACKBONES["siglip2"])
        enc = Encoder("siglip2", model, 224, tuple(proc.image_mean), tuple(proc.image_std), device, tokenizer)
    else:
        raise ValueError(f"unknown backbone {name!r}")
    enc.model.eval().to(device)
    return enc
