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


def test_text_spec_refused_without_both_tags(monkeypatch):
    calls = []
    real = nw.build_model
    monkeypatch.setattr(nw, "build_model", lambda *a, **k: calls.append(1) or real(*a, **k))
    for prereg, h2a in ((True, False), (False, True)):
        monkeypatch.setattr(nw, "prereg_ok", lambda repo=None, v=prereg: v)
        monkeypatch.setattr(nw, "prereg_h2a_ok", lambda repo=None, v=h2a: v)
        with pytest.raises(PermissionError):
            nw.run_neural_spec(synthetic(), spec("T1", "lstm-S+N"), 0, "2001-01-01", device="cpu", epochs=1)
    assert calls == []
    out = nw.run_neural_spec(synthetic(), spec("T1", "lstm-S"), 0, "2002-05-01", device="cpu", epochs=1)
    assert len(out) == 2


def test_stale_model_rows_ignore_their_own_and_later_targets():
    df = synthetic()
    i = int(np.flatnonzero(df.index >= "2001-01-01")[0])
    end = df.index[i + 7]
    sp = spec("T2", "tft-S+N")
    base = run(df, sp, df.index[i], end)
    pert = df.copy()
    pert.iloc[i:, pert.columns.get_indexer(["y_level", "y_change", "y_class"])] = [7.0, 7.0, 2]
    again = run(pert, sp, df.index[i], end)
    assert len(base) == 8
    assert np.array_equal(base["y_pred"].to_numpy(), again["y_pred"].to_numpy())


def test_t1_hybrid_ignores_rows_outside_training_window():
    df = synthetic()
    i = int(np.flatnonzero(df.index >= "2001-01-01")[0])
    day = df.index[i]
    sp = spec("T1", "hybrid-S+N")
    base = run(df, sp, day, day)
    pert = df.copy()
    pert.iloc[i - 1:i + 1, pert.columns.get_indexer(["y_level", "y_change"])] = 7.0
    pert.iloc[i - 1:i + 1, pert.columns.get_loc("y_class")] = 2
    pert.iloc[i + 1:, :] = pert.iloc[i + 1:, :] * -3 + 5
    assert run(pert, sp, day, day)["y_pred"].iloc[0] == base["y_pred"].iloc[0]


def test_hybrid_anchor_refits_at_every_row(monkeypatch):
    df = synthetic()
    seen = []
    real = nw._anchor
    monkeypatch.setattr(nw, "_anchor", lambda Z, yc, ycl, r, task: seen.append(len(r)) or real(Z, yc, ycl, r, task))
    run(df, spec("T2", "hybrid-S"), "2001-01-01", "2001-03-01")
    # per-row anchors (3 rows) have strictly growing training sets; the rest are out-of-fold anchors
    per_row = seen[-3:]
    assert per_row == sorted(set(per_row)) and len(per_row) == 3


def test_rejects_nan_target():
    df = synthetic()
    df.iloc[5, df.columns.get_loc("y_change")] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        run(df, spec("T1", "tft-S"), "2001-01-01")


def test_empty_test_window_returns_empty_frame():
    out = run(synthetic(), spec("T1", "lstm-S"), "2030-01-01")
    assert len(out) == 0
    assert list(out.columns) == ["date", "y_true", "y_pred", "lag", "class_true", "class_pred"]


def test_level_target_learns_constant_level():
    df = synthetic()
    df["y_level"], df["y_change"], df["y_class"], df["s_rate_lag"] = 5.0, 0.0, 1, 5.0
    out = nw.run_neural_spec(df, spec("T1", "lstm-S"), 0, "2001-01-01", "2001-02-01", device="cpu",
                             require_prereg=False)
    print("constant-level forecasts:", out["y_pred"].tolist())
    assert len(out) == 2 and (out["y_pred"] - 5.0).abs().max() < 0.15
