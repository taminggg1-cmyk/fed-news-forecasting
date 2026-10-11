import numpy as np
import pandas as pd
import pytest
import torch

from fed_news import decision_embed as de


def test_window_bounds_cover_every_token_once():
    assert de.window_bounds(0) == [(0, 0)]
    assert de.window_bounds(1020, 512) == [(0, 510), (510, 1020)]
    b = de.window_bounds(1021, 512)
    assert len(b) == 3 and b[-1] == (1020, 1021)


def test_mean_pool_ignores_padding():
    h = torch.tensor([[[1.0, 1.0], [3.0, 3.0], [100.0, 100.0]]])
    m = torch.tensor([[1, 1, 0]])
    assert torch.allclose(de.mean_pool(h, m), torch.tensor([[2.0, 2.0]]))


def test_needed_docs_cover_every_year_using_the_vintage():
    docs = pd.DataFrame({"doc_id": list("abcd"), "text": "x",
                         "release": pd.to_datetime(["1996-05-01", "2000-05-01", "2001-05-01", "2026-01-01"])})
    assert de.needed_docs(docs, 1999)["doc_id"].tolist() == ["a", "b"]
    assert de.needed_docs(docs, 2000)["doc_id"].tolist() == ["a", "b", "c"]
    assert de.needed_docs(docs, 2015)["doc_id"].tolist() == ["a", "b", "c"]
    assert de.needed_docs(docs, 2024)["doc_id"].tolist() == list("abcd")
    assert de.needed_docs(docs, 1990).empty


def fake_factory(calls):
    def fake(texts, vintage):
        calls.append(list(texts))
        return np.arange(len(texts) * 3, dtype=np.float32).reshape(len(texts), 3) + vintage
    return fake


def small_docs():
    return pd.DataFrame({"doc_id": ["a", "b"], "text": ["one", "two"],
                         "release": pd.to_datetime(["1999-01-01", "2000-01-01"])})


def test_cache_reuses_and_recomputes_when_text_changes(tmp_path):
    calls, docs = [], small_docs()
    first = de.load_or_embed(docs, 1999, fake_factory(calls), tmp_path)
    again = de.load_or_embed(docs, 1999, fake_factory(calls), tmp_path)
    assert len(calls) == 1 and np.array_equal(first["b"], again["b"])
    docs.loc[1, "text"] = "changed"
    de.load_or_embed(docs, 1999, fake_factory(calls), tmp_path)
    assert len(calls) == 2


def test_duplicate_doc_ids_are_refused(tmp_path):
    docs = pd.concat([small_docs(), small_docs().iloc[:1]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        de.load_or_embed(docs, 1999, fake_factory([]), tmp_path)


def test_cached_embedder_returns_rows_in_requested_order(tmp_path):
    calls, docs = [], small_docs()
    table = de.load_or_embed(docs, 1999, fake_factory(calls), tmp_path)
    out = de.cached_embedder(docs, fake_factory(calls), tmp_path)(1999, ["b", "a"])
    assert out.shape == (2, 3)
    assert np.array_equal(out[0], table["b"]) and np.array_equal(out[1], table["a"])


@pytest.mark.gpu
def test_embed_texts_gpu_smoke():
    out = de.embed_texts(["The Committee decided to raise the target range.", "word " * 1500], 1999, batch=4)
    assert out.shape == (2, 768) and np.isfinite(out).all()
