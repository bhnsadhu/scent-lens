import numpy as np
import pandas as pd

from scent_lens.evaluate import photo_records, summarize
from scent_lens.retrieval import Gallery


def test_photo_records_and_absent_bottles():
    emb = np.eye(6, dtype=np.float32)
    pids = ["p1", "p2", "p3", "p4", "p5", "p6"]
    g = Gallery(emb, pids)
    products = pd.DataFrame(
        {"brand": ["Dior", "Dior", "Chanel", "Creed", "Creed", "Ysl"],
         "line_key": ["dior|sauvage", "dior|sauvage", "chanel|bleu", "creed|aventus", "creed|viking", "ysl|y"]},
        index=pids,
    )
    q = np.array([[0.9, 0.5, 0, 0, 0, 0], [0.2, 0.9, 0, 0, 0, 0], [0, 0, 0, 0.8, 0.7, 0]], dtype=np.float32)
    photos = pd.DataFrame(
        {"photo_id": ["a", "b", "c"], "bottle_id": ["b1", "b1", "b2"], "condition": ["clean", "glare", "clean"],
         "status": ["matched", "matched", "absent"], "product_id": ["p2", "p2", ""]}
    )
    r = photo_records(g, products, q, photos)
    assert r.top1.tolist() == [0.0, 1.0, 0.0]
    assert r.top5.tolist() == [1.0, 1.0, 0.0]
    # p1 and p2 share a line, so the first photo is lenient correct
    assert r.lenient1.tolist() == [1.0, 1.0, 0.0]
    assert r["rank"].tolist() == [1, 0, -1]
    assert np.isclose(r.margin[0], 0.4)
    s = summarize(r, n_boot=50)
    assert s["photos"] == 2 and s["bottles"] == 1
    assert s["top1"]["value"] == 0.5
