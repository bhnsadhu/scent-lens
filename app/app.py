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

# Accord's world: a silver evaluation bench, ink type, Schibsted Grotesk with Geist Mono for
# figures. ScentLens is the lens behind Accord's scanner, so the two read as one family.
INK, INK_2, BENCH, LINE = "#171b1f", "#4a5058", "#f0f1f3", "#d9dce1"
# The same bench after dark: Gradio follows the viewer's system setting (or its Settings
# panel), so every light colour above has a dark twin, or text vanishes into the background.
INK_D, INK_2_D, BENCH_D, BLOCK_D, LINE_D = "#f0f1f3", "#a9b0b8", "#121518", "#1b1f23", "#30353c"
THEME = gr.themes.Base(
    font=[gr.themes.GoogleFont("Schibsted Grotesk"), "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("Geist Mono"), "ui-monospace", "monospace"],
    radius_size=gr.themes.sizes.radius_sm,
).set(
    body_background_fill=BENCH, body_text_color=INK, body_text_color_subdued=INK_2,
    block_background_fill="#ffffff", block_border_color=LINE, block_label_text_color=INK_2,
    block_title_text_color=INK, border_color_primary=LINE, color_accent=INK, color_accent_soft="#e2e4e7",
    button_primary_background_fill=INK, button_primary_background_fill_hover="#2b3036",
    button_primary_text_color="#f8f9fa", button_primary_border_color=INK,
    link_text_color=INK, link_text_color_hover=INK, link_text_color_visited=INK,
    input_background_fill="#ffffff", background_fill_secondary="#f7f8f9",
    body_background_fill_dark=BENCH_D, body_text_color_dark=INK_D, body_text_color_subdued_dark=INK_2_D,
    block_background_fill_dark=BLOCK_D, block_border_color_dark=LINE_D, block_label_text_color_dark=INK_2_D,
    block_title_text_color_dark=INK_D, border_color_primary_dark=LINE_D, color_accent_soft_dark="#2a2f35",
    button_primary_background_fill_dark=INK_D, button_primary_background_fill_hover_dark="#d9dce1",
    button_primary_text_color_dark=INK, button_primary_border_color_dark=INK_D,
    link_text_color_dark=INK_D, link_text_color_hover_dark=INK_D, link_text_color_visited_dark=INK_D,
    input_background_fill_dark=BLOCK_D, background_fill_secondary_dark="#171a1e",
)
CSS = """
/* Colours come from variables so the header, pills and tabs follow Gradio's light/dark switch
   (it puts .dark on <body>), whatever the system setting says. */
.gradio-container { --sl-ink: #171b1f; --sl-ink-2: #4a5058; --sl-on-ink: #f8f9fa; --sl-rule: #b9bec5; }
.dark .gradio-container, .gradio-container.dark { --sl-ink: #f0f1f3; --sl-ink-2: #a9b0b8; --sl-on-ink: #171b1f; --sl-rule: #5a616a;
  --color-accent: #f0f1f3; }  /* the theme has no dark accent setting; this lights the selected upload icon */
/* The header sits flush with the blocks below it: Gradio's HTML block padding is removed,
   the mark's top lines up with the title's cap height, and every line of text shares one edge. */
.sl-header, .sl-header > div { padding: 0 !important; }
.sl-head { display: flex; align-items: flex-start; gap: 18px; margin: 4px 0 6px; }
.sl-head svg { width: 56px; height: 56px; flex: none; margin-top: 2px; }
.sl-head svg .i { fill: var(--sl-ink) !important; }
.sl-head svg .s { stroke: var(--sl-ink) !important; }
.sl-head__text p { margin: 4px 0 0; }
.sl-head h1 { margin: 0; font-size: 34px; letter-spacing: -0.03em; font-weight: 600; color: var(--sl-ink); }
.sl-head p { color: var(--sl-ink-2); font-size: 15px; text-wrap: pretty; }
.sl-head .sl-links { display: flex; flex-wrap: wrap; gap: 4px 22px; margin-top: 10px; font: 500 12px/1.6 'Geist Mono', ui-monospace, monospace; letter-spacing: .1em; text-transform: uppercase; color: var(--sl-ink-2); }
.sl-links span { white-space: nowrap; }
.sl-links a { color: var(--sl-ink) !important; text-decoration: none !important; background: none !important; padding: 0 !important;
  border-radius: 0 !important; border-bottom: 1px solid var(--sl-rule); }
.sl-links a:hover { border-color: var(--sl-ink); }
.sl-pill { display: inline-block; font: 500 12px/1 'Geist Mono', ui-monospace, monospace; letter-spacing: .12em;
  text-transform: uppercase; padding: 9px 13px; border-radius: 999px; background: var(--sl-ink); color: var(--sl-on-ink); margin-bottom: 4px; }
.sl-pill--unsure { background: transparent; color: var(--sl-ink); box-shadow: inset 0 0 0 1px var(--sl-ink); }
/* Gradio draws its image toolbar icons at 20px; WCAG 2.2 asks for 24. */
.icon-button-wrapper .icon-button, .icon-button-wrapper .download-link, button[aria-label="Download"] {
  min-width: 24px !important; min-height: 24px !important; display: inline-flex; align-items: center; justify-content: center; }
@media (max-width: 600px) {
  .sl-head { gap: 14px; }
  .sl-head svg { width: 40px; height: 40px; margin-top: 2px; }
  .sl-head h1 { font-size: 28px; }
  .sl-head .sl-links { gap: 2px 18px; }
}
button[role="tab"] { color: var(--sl-ink-2) !important; }
.tab-nav button.selected, button[role="tab"][aria-selected="true"] { color: var(--sl-ink) !important; border-color: var(--sl-ink) !important; }
"""
# Gradio renders the sample photos as unnamed buttons; name each one for screen readers.
HEAD = """<script>
new MutationObserver(() => {
  document.querySelectorAll('.gallery-item:not([aria-label])').forEach((b, i) => b.setAttribute('aria-label', `Try sample photo ${i + 1}`));
  // An empty image slot carries a label but no role; give it one so the label is read.
  document.querySelectorAll('div[aria-label]:not([role])').forEach((d) => d.setAttribute('role', 'img'));
  // Icon-only buttons Gradio draws around images (enlarge, download) get their title as a name.
  document.querySelectorAll('button:not([aria-label])').forEach((x) => {
    if (!x.textContent.trim()) x.setAttribute('aria-label', x.title || 'Enlarge image');
  });
}).observe(document.documentElement, { childList: true, subtree: true });
</script>"""
MARK = (Path(__file__).resolve().parent / "mark.svg")
MARK_SVG = MARK.read_text() if MARK.exists() else ""
HEADER = f"""<div class="sl-head">{MARK_SVG}<div class="sl-head__text"><h1>ScentLens</h1>
<p>Names a fragrance from one photo of its bottle, across 24,642 catalog fragrances. When it is not sure, it says so.</p>
<p class="sl-links"><span>The bottle scanner in <a href="https://accord.bhanusadhu.com">Accord</a></span>
<span><a href="https://github.com/bhnsadhu/scent-lens">Repository</a></span>
<span>Designed and built by <a href="https://bhanusadhu.com">Bhanu Sadhu</a></span></p></div></div>"""
WAITING = '<span class="sl-pill sl-pill--unsure">Waiting for a photo</span>\n\nUpload a bottle photo, or try one of the samples.'


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
        pill = "Match · clear label" if r.reason == "clear label" else f"Match · confidence {r.confidence:.2f}"
        verdict = f'<span class="sl-pill">{pill}</span>\n\n## {m.brand} {m.name}\n\nIdentified: {how}. {_details(m)}'
    else:
        verdict = f'<span class="sl-pill sl-pill--unsure">Not sure</span>\n\n## Not sure\n\n{REASONS[r.reason]} ScentLens would rather say so than guess.\n\n{ADVICE}'
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
    gr.HTML(HEADER, elem_classes=["sl-header"])
    with gr.Tab("Identify"):
        with gr.Row():
            with gr.Column():
                inp = gr.Image(type="pil", label="Bottle photo", sources=["upload", "webcam", "clipboard"])
                btn = gr.Button("Identify", variant="primary")
                if SAMPLES:
                    gr.Examples([[str(p)] for p in SAMPLES], inputs=inp, label="Sample photos")
            with gr.Column():
                verdict = gr.Markdown(WAITING)
                # Only enlarge: downloading or sharing the annotated photo is not part of the job.
                crop = gr.Image(label="Bottle found", interactive=False, buttons=["fullscreen"])
        table = gr.Markdown()
        btn.click(identify, inp, [crop, verdict, table], api_name="ui_identify", api_visibility="private")
        inp.upload(identify, inp, [crop, verdict, table], api_name="ui_upload", api_visibility="private")
    with gr.Tab("Measured results"):
        gr.Markdown((HERE / "results.md").read_text() if (HERE / "results.md").exists() else "Results not bundled.")
    gr.api(identify_api, api_name="identify")

if __name__ == "__main__":
    demo.launch(max_file_size=MAX_UPLOAD, theme=THEME, css=CSS, head=HEAD, footer_links=["api", "settings"])
