import numpy as np

from scent_lens.retrieval import Gallery


def _unit(a):
    a = np.asarray(a, dtype=np.float32)
    return a / np.linalg.norm(a, axis=1, keepdims=True)


def test_gallery_takes_best_reference_per_product():
    emb = _unit([[1, 0], [0, 1], [0.6, 0.8], [-1, 0]])
    g = Gallery(emb, ["b", "a", "a", "c"])
    assert list(g.products) == ["a", "b", "c"]
    q = _unit([[0.6, 0.8]])
    s = g.scores(q)
    assert np.isclose(s[0, g.index["a"]], 1.0)  # best of its two images
    assert np.isclose(s[0, g.index["b"]], 0.6)
    idx, top = g.topk(q, k=2)
    assert [g.products[i] for i in idx[0]] == ["a", "b"]
    assert top[0, 0] >= top[0, 1]


def test_truth_index_marks_missing():
    g = Gallery(_unit([[1, 0], [0, 1]]), ["a", "b"])
    assert g.truth_index(["b", "zzz", None]).tolist() == [1, -1, -1]
