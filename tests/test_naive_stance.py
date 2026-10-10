import numpy as np
import pandas as pd

from fed_news import naive_stance as ns, stance


def test_score_naive_uses_named_model_and_restores(monkeypatch, tmp_path):
    docs = pd.DataFrame({"doc_id": ["a", "b"], "type": ["speech", "minutes"],
                         "release": pd.to_datetime(["2005-01-03", "2005-01-04"]),
                         "text": ["Rates may rise again later this year if inflation stays high. "
                                  "Inflation remains well above the committee objective.",
                                  "Policy will remain accommodative for an extended period of time."]})
    labels = pd.DataFrame({"sentence": ["x", "y", "z"], "label": [0, 1, 2], "year": [2001, 2010, 2020]})
    seen = []

    def fake(train, texts, vintage, seed):
        seen.append((stance.model_name(vintage), len(train)))
        out = np.zeros((len(texts), 3), dtype=np.float32)
        out[:, 1] = 0.7
        out[:, 0] = 0.2
        out[:, 2] = 0.1
        return out

    monkeypatch.setattr(ns, "build_corpus", lambda: docs)
    monkeypatch.setattr(ns, "load_labels", lambda: labels)
    monkeypatch.setattr(stance, "train_and_score", fake)
    monkeypatch.setattr(ns, "OUT_PATH", tmp_path / "naive.pkl")
    monkeypatch.setattr(ns, "OUT", tmp_path)
    out = ns.score_naive("some/model")
    assert all(name == "some/model" and n == 3 for name, n in seen)       # all labels, the named model
    assert stance.model_name(2010) == "manelalab/chrono-bert-v1-20101231"  # restored
    assert np.allclose(out["stance"], 0.5) and list(out["n_sent"]) == [2, 1]


def test_model_name_restored_when_training_fails(monkeypatch, tmp_path):
    monkeypatch.setattr(ns, "build_corpus", lambda: pd.DataFrame(
        {"doc_id": ["a"], "type": ["speech"], "release": pd.to_datetime(["2005-01-03"]),
         "text": ["Policy will remain accommodative for an extended period of time."]}))
    monkeypatch.setattr(ns, "load_labels", lambda: pd.DataFrame({"sentence": ["x"], "label": [0], "year": [2001]}))

    def boom(*a, **k):
        raise RuntimeError("gpu")
    monkeypatch.setattr(stance, "train_and_score", boom)
    try:
        ns.score_naive("some/model")
    except RuntimeError:
        pass
    assert stance.model_name(2010) == "manelalab/chrono-bert-v1-20101231"


def test_naive_panel_replaces_only_stance_columns(monkeypatch, tmp_path):
    idx = pd.date_range("2005-01-01", periods=4, freq="MS")
    rows = pd.DataFrame({"win_start": idx - pd.offsets.MonthBegin(1), "cutoff": idx}, index=idx)
    full = rows.assign(n_stance=9.0, n_stance_chg=9.0, n_beige=1.0, s_rate_lag=2.0)
    rows.to_pickle(tmp_path / "t1.pkl")
    full.to_pickle(tmp_path / "t1_full.pkl")
    monkeypatch.setattr(ns.config, "DATA", tmp_path)
    ds = pd.DataFrame({"doc_id": ["a", "b"], "type": ["speech", "speech"],
                       "release": pd.to_datetime(["2004-12-15", "2005-02-10"]), "stance": [0.2, 0.6], "n_sent": [1, 1]})
    out = ns.naive_panel("t1", ds)
    assert np.allclose(out["n_stance"], [0.2, 0.2, 0.6, 0.6])             # carried forward when a month is empty
    assert np.isnan(out["n_stance_chg"].iloc[0]) and np.allclose(out["n_stance_chg"].iloc[1:], [0.0, 0.4, 0.0])
    assert (out["n_beige"] == 1.0).all()
