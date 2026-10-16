import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.torch


@pytest.mark.parametrize("name", ["dinov2", "siglip2"])
def test_untrained_finetune_model_equals_frozen_backbone(name):
    from PIL import Image

    from scent_lens.embed import Encoder, load_encoder
    from scent_lens.finetune import build_model, info_nce

    rng = np.random.default_rng(0)
    imgs = [Image.fromarray(rng.integers(0, 255, (300, 200, 3), dtype=np.uint8)) for _ in range(3)]
    frozen = load_encoder(name, device="cpu")
    model, mean, std = build_model(name)
    ft = Encoder("finetuned", model.eval(), 224, mean, std, "cpu")
    a, b = frozen.encode_images(imgs), ft.encode_images(imgs)
    assert np.allclose(a, b, atol=1e-4)
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert trainable and all("lora_" in n or n.startswith("head.") for n in trainable)
    q = torch.nn.functional.normalize(torch.randn(4, 8), dim=-1)
    loss = info_nce(q, q, ["a", "b", "c", "a"])
    assert torch.isfinite(loss)
