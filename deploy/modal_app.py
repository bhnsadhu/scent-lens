"""Serve the ScentLens demo on Modal (free Starter credits), CPU only.

    .venv/bin/python scripts/build_space.py bhnsadhu   # writes artifacts/space
    modal deploy deploy/modal_app.py

The bundle in artifacts/space (app, scent_lens modules, export, approved
samples, results page) is the same one built for a Hugging Face Space. The
OWLv2 detector weights are baked into the image so a cold start downloads
nothing. One container at most, shut down after 5 idle minutes, which keeps
the cost to cents per hour of use.
"""

from __future__ import annotations

from pathlib import Path

import modal

BUNDLE = Path(__file__).resolve().parents[1] / "artifacts" / "space"
HF_HOME = "/cache/hf"


def _fetch_detector() -> None:
    from huggingface_hub import snapshot_download

    snapshot_download("google/owlv2-base-patch16-ensemble")


image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install("torch==2.14.1", "torchvision==0.29.1", index_url="https://download.pytorch.org/whl/cpu")
    .pip_install_from_requirements(str(BUNDLE / "requirements.txt"))
    .pip_install("gradio==6.30.0", "fastapi==0.143.2")
    .env({"HF_HOME": HF_HOME, "SCENTLENS_READER": "rapidocr", "GRADIO_ANALYTICS_ENABLED": "False",
          "OMP_NUM_THREADS": "8", "MKL_NUM_THREADS": "8"})  # match the 4 physical cores; the host has many more
    .run_function(_fetch_detector)
    .env({"HF_HUB_OFFLINE": "1"})
    .add_local_dir(BUNDLE, "/app")
)

app = modal.App("scent-lens", image=image)


@app.function(cpu=4.0, memory=4096, max_containers=1, scaledown_window=300, timeout=600)
@modal.concurrent(max_inputs=4)
@modal.asgi_app()
def web():
    import os
    import sys

    os.chdir("/app")
    sys.path.insert(0, "/app")
    from fastapi import FastAPI
    from gradio.routes import mount_gradio_app

    import app as demo  # the Gradio Blocks in artifacts/space/app.py

    return mount_gradio_app(FastAPI(), demo.demo, path="/", max_file_size=demo.MAX_UPLOAD,
                            allowed_paths=["/app/samples"])
