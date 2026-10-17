import numpy as np
import pandas as pd

from fed_news.studies.plan5_event_study import naive_stance as ns
from fed_news.features import stance


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


def _events(n=1500, slope=2.0, seed=0):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("1998-01-01", periods=n)
    s = rng.normal(size=n)
    return pd.DataFrame({"surprise": s, "dy2": slope * s + rng.normal(size=n), "dy2_before": rng.normal(size=n),
                         "category": "speech", "stance": s, "n_docs": 1}, index=pd.Index(days, name="day"))


def test_event_compare_columns():
    a, b = _events(), _events(seed=1)
    out = ns.event_compare(a, b, test_start="2001-01-01")
    for c in ("n_common", "mse_small_chrono", "mse_large_naive", "mse_ratio_chrono", "mse_ratio_naive",
              "dm_p_naive_vs_chrono"):
        assert c in out.columns
    assert 0 <= out["dm_p_naive_vs_chrono"].iloc[0] <= 1


def test_compare_pipelines_writes_direct_column_and_no_verdict(monkeypatch, tmp_path):
    dates = pd.date_range("2005-01-01", periods=30, freq="MS")
    rng = np.random.default_rng(0)

    def preds(noise):
        return pd.DataFrame({"date": dates, "y_true": 1.0 + np.arange(30) * 0.01,
                             "y_pred": 1.0 + np.arange(30) * 0.01 + rng.normal(scale=noise, size=30),
                             "lag": 1.0, "class_true": 1, "class_pred": 1})
    for task in ("T1", "T2"):
        for n in ("S", "S+N"):
            preds(0.1).to_csv(tmp_path / f"preds_{task}_{n}.csv", index=False)
    pd.DataFrame({"doc_id": ["a"]}).to_pickle(tmp_path / "naive.pkl")
    pd.DataFrame({"y2": [1.0]}).to_pickle(tmp_path / "market_daily.pkl")
    monkeypatch.setattr(ns.config, "RESULTS", tmp_path)
    monkeypatch.setattr(ns.config, "DATA", tmp_path)
    monkeypatch.setattr(ns, "OUT", tmp_path)
    monkeypatch.setattr(ns, "OUT_PATH", tmp_path / "naive.pkl")
    monkeypatch.setattr(ns, "naive_panel", lambda name, d: None)
    monkeypatch.setattr(ns, "common_rows", lambda x: x)
    monkeypatch.setattr(ns, "run_spec", lambda *a, **k: preds(0.2))
    monkeypatch.setattr(ns, "build_events", lambda d, y: _events())
    monkeypatch.setattr(ns, "load_events", lambda: _events(seed=3))
    table = ns.compare_pipelines()
    assert "dm_p_naive_vs_chrono" in table.columns
    assert "verdict" not in pd.read_csv(tmp_path / "naive_event_study.csv").columns
    assert (tmp_path / "lookahead_event_compare.csv").exists()
