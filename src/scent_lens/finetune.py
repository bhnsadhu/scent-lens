"""Contrastive fine tune of a frozen backbone with LoRA adapters and a projection head.

Queries are augmented catalog images, positives are clean catalog images of
the same product, negatives are the other products in the batch. Batches are
built brand by brand so same brand products act as hard negatives. Only LoRA
adapters on the attention projections of the last blocks and a projection
head (initialized to the identity) are trained.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn

from scent_lens.embed import BACKBONES, Encoder, pick_device

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# attention projection module names and the block path per backbone
LORA_TARGETS = {
    "dinov2": (["q_proj", "v_proj"], "encoder.layer"),
    "siglip2": (["q_proj", "v_proj"], "encoder.layers"),
}


class EmbeddingModel(nn.Module):
    def __init__(self, backbone_name: str, vision: nn.Module, dim: int):
        super().__init__()
        self.backbone_name = backbone_name
        self.vision = vision
        self.head = nn.Linear(dim, dim, bias=False)
        with torch.no_grad():
            self.head.weight.copy_(torch.eye(dim))

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        out = self.vision(pixel_values=pixels).pooler_output
        return nn.functional.normalize(self.head(out), dim=-1)


def _vision_tower(name: str) -> tuple[nn.Module, int, tuple, tuple]:
    from transformers import AutoImageProcessor, AutoModel

    proc = AutoImageProcessor.from_pretrained(BACKBONES[name])
    model = AutoModel.from_pretrained(BACKBONES[name])
    if name == "siglip2":
        vision = model.vision_model
        dim = model.config.vision_config.hidden_size
    elif name == "dinov2":
        vision = model
        dim = model.config.hidden_size
    else:
        raise ValueError(f"fine tuning supports dinov2 and siglip2, not {name}")
    return vision, dim, tuple(proc.image_mean), tuple(proc.image_std)


def build_model(name: str, lora_rank: int = 8, last_blocks: int = 4) -> tuple[EmbeddingModel, tuple, tuple]:
    from peft import LoraConfig, get_peft_model

    vision, dim, mean, std = _vision_tower(name)
    targets, block_path = LORA_TARGETS[name]
    n_blocks = len(vision.get_submodule(block_path))
    keep = range(n_blocks - last_blocks, n_blocks)
    modules = [
        n for n, _ in vision.named_modules()
        if any(n.endswith(t) for t in targets) and any(f"{block_path}.{i}." in n for i in keep)
    ]
    cfg = LoraConfig(r=lora_rank, lora_alpha=2 * lora_rank, lora_dropout=0.05, target_modules=modules)
    vision = get_peft_model(vision, cfg)
    model = EmbeddingModel(name, vision, dim)
    for p in model.parameters():
        p.requires_grad = False
    for n, p in model.named_parameters():
        if "lora_" in n or n.startswith("head."):
            p.requires_grad = True
    return model, mean, std


def info_nce(q: torch.Tensor, p: torch.Tensor, product_ids: list[str], temperature: float = 0.05) -> torch.Tensor:
    """Symmetric InfoNCE; entries of the same product are not counted as negatives."""
    logits = q @ p.T / temperature
    ids = np.asarray(product_ids)
    same = torch.from_numpy(ids[:, None] == ids[None, :]).to(logits.device)
    eye = torch.eye(len(ids), dtype=torch.bool, device=logits.device)
    logits = logits.masked_fill(same & ~eye, float("-inf"))
    target = torch.arange(len(ids), device=logits.device)
    return 0.5 * (nn.functional.cross_entropy(logits, target) + nn.functional.cross_entropy(logits.T, target))


def brand_batches(products: list[str], brands: list[str], batch: int, rng: np.random.Generator) -> list[list[int]]:
    """Index batches built from shuffled brands, up to 4 products per brand per batch."""
    by_brand: dict[str, list[int]] = {}
    for i, b in enumerate(brands):
        by_brand.setdefault(b, []).append(i)
    chunks = []
    for idx in by_brand.values():
        idx = list(rng.permutation(idx))
        chunks += [idx[i : i + 4] for i in range(0, len(idx), 4)]
    order = rng.permutation(len(chunks))
    batches, cur = [], []
    for c in order:
        for i in chunks[c]:
            cur.append(int(i))
            if len(cur) == batch:
                batches.append(cur)
                cur = []
    return batches


def save_checkpoint(model: EmbeddingModel, path: Path, meta: dict) -> None:
    path.mkdir(parents=True, exist_ok=True)
    trainable = {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}
    torch.save(trainable, path / "trainable.pt")
    (path / "meta.json").write_text(json.dumps(meta, indent=2))


def load_finetuned(path: str | Path, device: str | None = None) -> Encoder:
    path = Path(path)
    meta = json.loads((path / "meta.json").read_text())
    model, mean, std = build_model(meta["backbone"], meta["lora_rank"], meta["last_blocks"])
    state = torch.load(path / "trainable.pt", map_location="cpu")
    missing = set(state) - {n for n, _ in model.named_parameters()}
    if missing:
        raise KeyError(f"checkpoint has unknown parameters: {sorted(missing)[:3]}")
    model.load_state_dict(state, strict=False)
    device = device or pick_device()
    model.eval().to(device)
    return Encoder("finetuned", model, 224, mean, std, device)


def merge_for_export(enc: Encoder) -> nn.Module:
    """Fold LoRA weights into the base model, returning a plain module."""
    model: EmbeddingModel = enc.model
    model.vision = model.vision.merge_and_unload()
    return model
