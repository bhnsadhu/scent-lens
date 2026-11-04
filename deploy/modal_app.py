"""Serve the ScentLens demo on Modal (free Starter credits), CPU only.

    .venv/bin/python scripts/build_space.py bhnsadhu   # writes artifacts/space
    modal deploy deploy/modal_app.py

The bundle in artifacts/space (app, scent_lens modules, export, approved
samples, results page) is the same one built for a Hugging Face Space. The
OWLv2 detector weights are baked into the image so a cold start downloads
nothing. One container at most, shut down after 5 idle minutes, which keeps
the cost to cents per hour of use.

A redeploy replaces the running container: calls in flight on the old one are
cancelled, and the new one needs about a minute to load before /identify
answers again, so Accord's scanner can time out briefly. Deploy when it's quiet.
"""

from __future__ import annotations

from pathlib import Path

import modal

BUNDLE = Path(__file__).resolve().parents[1] / "artifacts" / "space"
BRAND = Path(__file__).resolve().parent / "brand"  # mark, icons and link-preview image
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
    .add_local_dir(BRAND, "/brand")
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
    from fastapi.staticfiles import StaticFiles
    from gradio.routes import mount_gradio_app

    import app as demo  # the Gradio Blocks in artifacts/space/app.py

    api = FastAPI()
    api.mount("/brand", StaticFiles(directory="/brand"), name="brand")
    api.middleware("http")(_brand_page)
    return mount_gradio_app(api, demo.demo, path="/", max_file_size=demo.MAX_UPLOAD,
                            allowed_paths=["/app/samples"], favicon_path="/brand/icon-512.png")


SITE = "https://bhnsadhu--scent-lens-web.us-east.modal.run"
DESCRIPTION = ("Names a fragrance from one photo of its bottle across 24,642 fragrances, "
               "or says it isn't sure. The bottle scanner behind Accord.")
HEAD = f"""<title>ScentLens · identify a fragrance from a photo</title>
<meta name="description" content="{DESCRIPTION}">
<meta name="theme-color" content="#f0f1f3">
<meta property="og:type" content="website">
<meta property="og:url" content="{SITE}/">
<meta property="og:site_name" content="ScentLens">
<meta property="og:title" content="ScentLens · identify a fragrance from a photo">
<meta property="og:description" content="{DESCRIPTION}">
<meta property="og:image" content="{SITE}/brand/og.png">
<meta property="og:image:width" content="1200">
<meta property="og:image:height" content="630">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:title" content="ScentLens · identify a fragrance from a photo">
<meta name="twitter:description" content="{DESCRIPTION}">
<meta name="twitter:image" content="{SITE}/brand/og.png">
<link rel="icon" type="image/svg+xml" href="/brand/mark.svg">
<link rel="icon" type="image/png" sizes="32x32" href="/brand/favicon-32.png">
<link rel="apple-touch-icon" href="/brand/apple-touch-icon.png">"""


async def _brand_page(request, call_next):
    """Swap Gradio's default page tags ("Gradio", its banner, its icon) for ScentLens's own."""
    import re

    from fastapi.responses import HTMLResponse

    response = await call_next(request)
    html_page = response.headers.get("content-type", "").startswith("text/html")
    if request.method != "GET" or request.url.path != "/" or not html_page or response.headers.get("content-encoding"):
        return response
    body = b"".join([chunk async for chunk in response.body_iterator]).decode("utf-8")
    body = re.sub(r'<meta[^>]*?(?:property|name)="(?:og|twitter):[^"]*"[^>]*>'
                  r'|<meta[^>]*?name="(?:description|theme-color)"[^>]*>'
                  r'|<link[^>]*?rel="(?:icon|shortcut icon|apple-touch-icon)"[^>]*>'
                  r'|<title>.*?</title>', "", body, flags=re.S)
    body = re.sub(r"<head[^>]*>", lambda m: m.group(0) + HEAD, body, count=1)
    headers = {k: v for k, v in response.headers.items() if k.lower() != "content-length"}
    return HTMLResponse(body, status_code=response.status_code, headers=headers)
