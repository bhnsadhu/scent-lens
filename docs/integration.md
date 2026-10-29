# Using the ScentLens recognizer from another app

The recognizer is one Python call. Another app can use it in two ways.

## 1. In Python

```python
from PIL import Image
from scent_lens.recognize import Recognizer

rec = Recognizer("artifacts/export")          # loads once, a few seconds
result = rec(Image.open("bottle.jpg"))        # or a file path
result.to_dict()
```

`Recognizer(export_dir, device=None, reader=None)` takes these options:

- `device`: `"cpu"`, `"mps"` or `"cuda"`. When not given, it picks the best one available.
- `reader`: `"vision"` uses macOS Vision (macOS only). Every version 2 setting was chosen with it. `"rapidocr"` is cross platform. When not given, it uses Vision if `ocrmac` is installed and RapidOCR otherwise.

The export folder is written by `scripts/export_model.py`.

## 2. Over HTTP, through the Hugging Face Space

The Space serves the same function as the API endpoint `identify`. From Python:

```python
from gradio_client import Client, handle_file

client = Client("<user>/scent-lens")
result = client.predict(handle_file("bottle.jpg"), api_name="/identify")
```

From JavaScript or TypeScript (for example a Next.js server route), use `@gradio/client`:

```ts
import { Client, handle_file } from "@gradio/client";

const client = await Client.connect("<user>/scent-lens");
const { data } = await client.predict("/identify", { image: handle_file(blob) });
```

A free Space sleeps when idle, so the first request after a pause can take a minute or more. It also runs on a shared CPU. Measured CPU latency is in `results/recognizer_rapidocr.md`.

## The result

```json
{
  "answered": true,
  "reason": "confident",
  "confidence": 0.8123,
  "product_id": "p05819",
  "matches": [
    {"product_id": "p05819", "brand": "Chanel", "name": "Chance Eau De Parfum", "score": 1.0412,
     "family": "FLORAL", "subfamily": "...", "accords": "...", "notes": "...", "gender": "...", "year": "..."}
  ],
  "text_read": ["CHANCE", "CHANEL"],
  "box": [120, 40, 610, 980],
  "seconds": 1.16
}
```

| Field | Meaning |
|---|---|
| `answered` | Whether the system names a fragrance. When false, show the top matches as suggestions, not as an answer. |
| `reason` | Why it answered or abstained (see the next table). |
| `confidence` | A calibrated chance that the top match is right, fitted on 101 development photos. It is a coarse estimate from a small sample, not a guarantee. It is 1.0 for `"clear label"`. |
| `product_id` | The answer, or null. |
| `matches` | The top 3 catalog products. `score` is image similarity plus 0.3 times the label match, which ranks matches but is not a probability. |
| `text_read` | The label text that was read. |
| `box` | The bottle crop, in the frame of the photo resized to at most 1024 pixels on its longer side. |
| `seconds` | Time taken by the call. |

`reason` is one of the following:

| Reason | Answered | Meaning |
|---|---|---|
| `clear label` | yes | Every brand and name word of exactly one catalog product was read, and at least half the words read belong to it. |
| `confident` | yes | Calibrated confidence ≥ 0.670, and the label read nothing or agrees. |
| `low confidence` | no | The best matches are too close to call. |
| `label disagrees` | no | Most of the words read do not belong to the best guess. |

## Stable ids

`product_id` (for example `p05819`) is the id from the catalog `doevent/perfume`, as cleaned by `scripts/clean_catalog.py`, and it stays the same across exports. `artifacts/export/products.csv` maps each id to its brand, name, family, accords, notes, gender and year. To link to another database, match on this id, or on brand and name through `scent_lens.normalize.product_key`.

## Known limits

- The system names only the 24,642 catalog fragrances. A flanker that is missing from the catalog can be answered as its catalog sibling. On test photos this happened twice: Glossier You Fleur was answered as Glossier You, and Good Girl was answered as Good Girl Légère.
- It reads the front label best. Rear views, side views and photos where a hand covers the bottle are rarely answered.
- One bottle per photo.
