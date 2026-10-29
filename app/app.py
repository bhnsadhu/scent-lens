"""ScentLens demo: upload a perfume bottle photo, get the fragrance or an honest "not sure".

Runs locally from the repo (artifacts/export) or as a Hugging Face Space
(export/ next to this file). The same function is exposed as an API endpoint,
"identify", returning the JSON of scent_lens.recognize.Result.to_dict().
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import gradio as gr
from gradio.data_classes import FileData
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent if (HERE.parent / "src" / "scent_lens").exists() else HERE
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from scent_lens.photos import clean_pil  # noqa: E402
from scent_lens.recognize import Recognizer  # noqa: E402

EXPORT = Path(os.environ.get("SCENTLENS_EXPORT", HERE / "export" if (HERE / "export").exists() else ROOT / "artifacts" / "export"))
SAMPLES = sorted((HERE / "samples").glob("*.jpg"))
MAX_UPLOAD = "10mb"
ADVICE = ("**Tips for a better photo:** show the front label straight on, fill most of the frame with the bottle, "
          "avoid glare from lights or windows, and keep other bottles out of the shot.")
REASONS = {
    "low confidence": "The closest catalog matches are too close to call.",
    "label disagrees": "The text on the label does not match the closest catalog fragrance, so this bottle may not be in the catalog.",
}

rec = Recognizer(EXPORT, reader=os.environ.get("SCENTLENS_READER"))  # RapidOCR off macOS


def _details(m) -> str:
    bits = [m.details.get("family", "").title(), m.details.get("accords", "")]
    notes = m.details.get("notes", "")
    line = " · ".join(b for b in bits if b)
    return line + (f"<br><small>Notes: {notes}</small>" if notes else "")


def identify(image: Image.Image | None):
    if image is None:
        return None, "Upload a photo of a perfume bottle.", ""
    r = rec(image)
    shown = clean_pil(image)  # the frame the box was found in
    if r.box:
        ImageDraw.Draw(shown).rectangle(r.box, outline=(214, 64, 120), width=4)
    if r.answered:
        m = r.matches[0]
        how = ("the label names this fragrance clearly" if r.reason == "clear label"
               else f"estimated confidence {r.confidence:.2f} (a calibrated estimate from 101 development photos, not a guarantee)")
        verdict = f"## {m.brand} {m.name}\n\nIdentified: {how}. {_details(m)}"
    else:
        verdict = f"## Not sure\n\n{REASONS[r.reason]} ScentLens would rather say so than guess.\n\n{ADVICE}"
    rows = "\n".join(f"| {i + 1} | {m.brand} | {m.name} | {m.details.get('family', '').title()} | {m.score:.3f} |"
                     for i, m in enumerate(r.matches))
    title = "Top 3 catalog matches" if r.answered else "Closest catalog matches (not an answer)"
    table = (f"### {title}\n\n| | Brand | Fragrance | Family | Match score |\n|---|---|---|---|---|\n{rows}\n\n"
             f"<small>Match score is visual similarity plus a label bonus, not a probability. "
             f"Label text read: {', '.join(dict.fromkeys(r.text)) or 'none'}. Time: {r.seconds:.1f} s.</small>")
    return shown, verdict, table


def identify_api(image: FileData) -> dict:
    """Structured result for other apps (see docs/integration.md)."""
    path = image["path"] if isinstance(image, dict) else image.path
    with Image.open(path) as img:
        return rec(img).to_dict()


with gr.Blocks(title="ScentLens") as demo:
    gr.Markdown("# ScentLens\nIdentify a fragrance from a photo of its bottle, across 24,642 catalog fragrances. "
                "When it is not sure, it says so.")
    with gr.Tab("Identify"):
        with gr.Row():
            with gr.Column():
                inp = gr.Image(type="pil", label="Bottle photo", sources=["upload", "webcam", "clipboard"])
                btn = gr.Button("Identify", variant="primary")
                if SAMPLES:
                    gr.Examples([[str(p)] for p in SAMPLES], inputs=inp, label="Sample photos")
            with gr.Column():
                verdict = gr.Markdown()
                crop = gr.Image(label="Bottle found", interactive=False)
        table = gr.Markdown()
        btn.click(identify, inp, [crop, verdict, table], api_name="ui_identify", api_visibility="private")
        inp.upload(identify, inp, [crop, verdict, table], api_name="ui_upload", api_visibility="private")
    with gr.Tab("Measured results"):
        gr.Markdown((HERE / "results.md").read_text() if (HERE / "results.md").exists() else "Results not bundled.")
    gr.api(identify_api, api_name="identify")

if __name__ == "__main__":
    demo.launch(max_file_size=MAX_UPLOAD)
