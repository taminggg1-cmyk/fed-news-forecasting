import numpy as np
import pandas as pd
import pytest

from fed_news import models, neural_walk as nw, panel


def synthetic(n=90, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("1995-01-01", periods=n, freq="MS")
    df = pd.DataFrame(rng.normal(size=(n, len(panel.S_COLS))), index=idx, columns=panel.S_COLS)
    for c in panel.M_COLS + panel.N_COLS:
        df[c] = rng.normal(size=n)
    df["t_u3"] = rng.normal(size=n)
    df["s_rate_lag"] = 2.0
    change = 0.25 * np.sign(df["n_stance"].to_numpy()) * (np.abs(df["n_stance"].to_numpy()) > 0.5)
    df["y_change"], df["y_level"] = change, 2.0 + change
    df["y_class"] = np.where(change > 0.02, 2, np.where(change < -0.02, 0, 1))
    return df


def spec(task, name):
    return {s.name: s for s in nw.neural_ladder(task)}[name]


def run(df, sp, start, end=None, **kw):
    return nw.run_neural_spec(df, sp, 0, start, end, device="cpu", require_prereg=False, epochs=kw.pop("epochs", 3), **kw)


def test_ladder_matches_preregistered_settings():
    t1 = {s.name: s for s in nw.neural_ladder("T1")}
    assert (t1["lstm-S"].lookback, t1["lstm-S"].hidden, t1["lstm-S"].lr, t1["lstm-S"].epochs, t1["lstm-S"].refit_every) == (12, 240, 1e-4, 300, 1)
    assert (t1["gru-S+N"].lookback, t1["gru-S+N"].hidden, t1["gru-S+N"].lr, t1["gru-S+N"].epochs) == (12, 188, 2e-4, 449)
    assert (t1["tft-S"].hidden, t1["tft-S"].dropout, t1["tft-S"].lookback, t1["tft-S"].refit_every, t1["tft-S"].target) == (32, 0.2, 12, 12, "change")
    assert t1["hybrid-S+N"].anchor and t1["hybrid-S+N"].cols == panel.S_COLS + panel.N_COLS
    t2 = {s.name: s for s in nw.neural_ladder("T2")}
    assert (t2["mlstm-S"].lookback, t2["mlstm-S"].refit_every, t2["mlstm-S"].target) == (4, 8, "level")
    assert (t2["tft-S+N"].lookback, t2["tft-S+N"].refit_every) == (8, 8)
    assert sorted(t2) == ["hybrid-S", "hybrid-S+N", "mlstm-S", "mlstm-S+N", "tft-S", "tft-S+N"]
    assert nw.neural_ladder("T1", paper_refit_every=3)[0].refit_every == 3


def test_one_forecast_per_test_row():
    df = synthetic()
    for task, name in (("T1", "lstm-S"), ("T2", "tft-S"), ("T2", "hybrid-S")):
        out = run(df, spec(task, name), "2001-01-01")
        assert len(out) == int((df.index >= "2001-01-01").sum())
        assert np.isfinite(out["y_pred"]).all()


def test_t1_forecast_ignores_rows_outside_training_window():
    df = synthetic()
    i = int(np.flatnonzero(df.index >= "2001-01-01")[0])
    day = df.index[i]
    base = run(df, spec("T1", "lstm-S+N"), day, day)
    pert = df.copy()
    pert.iloc[i - 1:i + 1, pert.columns.get_indexer(["y_level", "y_change"])] = 7.0
    pert.iloc[i - 1:i + 1, pert.columns.get_loc("y_class")] = 2
    pert.iloc[i + 1:, :] = pert.iloc[i + 1:, :] * -3 + 5
    again = run(pert, spec("T1", "lstm-S+N"), day, day)
    assert again["y_pred"].iloc[0] == base["y_pred"].iloc[0]


def test_t2_forecast_ignores_current_and_future_targets():
    df = synthetic()
    i = int(np.flatnonzero(df.index >= "2001-01-01")[0])
    day = df.index[i]
    for name in ("tft-S+N", "hybrid-S+N", "mlstm-S+N"):
        base = run(df, spec("T2", name), day, day)
        pert = df.copy()
        pert.iloc[i, pert.columns.get_indexer(["y_level", "y_change"])] = 7.0
        pert.iloc[i, pert.columns.get_loc("y_class")] = 2
        pert.iloc[i + 1:, :] = pert.iloc[i + 1:, :] * -3 + 5
        again = run(pert, spec("T2", name), day, day)
        assert again["y_pred"].iloc[0] == base["y_pred"].iloc[0], name


def test_refit_schedule_builds_one_model_per_block(monkeypatch):
    df = synthetic()
    calls = []
    real = nw.build_model
    monkeypatch.setattr(nw, "build_model", lambda *a, **k: calls.append(1) or real(*a, **k))
    n_test = int((df.index >= "2001-01-01").sum())   # 18
    run(df, spec("T2", "tft-S"), "2001-01-01")
    assert len(calls) == -(-n_test // 8)


def test_oof_residuals_come_from_anchors_fit_on_earlier_rows(monkeypatch):
    df = synthetic()
    X = df[panel.S_COLS].to_numpy(float)
    rows = np.arange(60)
    seen = []
    real = nw._anchor
    monkeypatch.setattr(nw, "_anchor", lambda Z, yc, ycl, r, task: seen.append(np.asarray(r)) or real(Z, yc, ycl, r, task))
    res_rows, res = nw.oof_residuals(X, df["y_change"].to_numpy(), df["y_class"].to_numpy(), rows, "T1")
    folds = [(tr, va) for tr, va in models.expanding_folds(60) if len(tr) >= nw.MIN_ANCHOR_ROWS]
    assert len(seen) == len(folds) > 0
    for used, (tr, va) in zip(seen, folds):
        assert used.max() < va.min()               # each anchor saw only rows before the ones it scores
    assert np.array_equal(res_rows, np.concatenate([va for _, va in folds]))
    assert len(res) == len(res_rows)


def test_hybrid_runs_at_earliest_refit():
    df = synthetic()
    out = run(df, spec("T2", "hybrid-S+N"), "1999-01-01", "1999-02-01", epochs=5)
    assert len(out) == 2 and np.isfinite(out["y_pred"]).all()


def test_rejects_non_finite_inputs():
    df = synthetic()
    df.iloc[5, df.columns.get_loc("n_epu")] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        run(df, spec("T1", "tft-S+N"), "2001-01-01")


def test_rejects_unsorted_index():
    df = synthetic().iloc[::-1]
    with pytest.raises(ValueError, match="chronological"):
        run(df, spec("T1", "tft-S"), "2001-01-01")


def test_text_spec_refused_without_h2a_tag(monkeypatch):
    monkeypatch.setattr(nw, "prereg_ok", lambda repo=None: True)
    monkeypatch.setattr(nw, "prereg_h2a_ok", lambda repo=None: False)
    with pytest.raises(PermissionError):
        nw.run_neural_spec(synthetic(), spec("T1", "lstm-S+N"), 0, "2001-01-01", device="cpu", epochs=1)
    out = nw.run_neural_spec(synthetic(), spec("T1", "lstm-S"), 0, "2002-05-01", device="cpu", epochs=1)
    assert len(out) == 2
