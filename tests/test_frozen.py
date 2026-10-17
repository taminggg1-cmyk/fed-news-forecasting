import numpy as np
import pandas as pd

from fed_news import config
from fed_news.studies.plan2_neural_original import frozen, neural_walk as nw, run_neural
from fed_news.features import panel


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


def test_frozen_linear_never_sees_test_period_targets(monkeypatch):
    monkeypatch.setattr(frozen, "prereg_ok", lambda repo=None: True)
    monkeypatch.setattr(frozen, "prereg_h2a_ok", lambda repo=None: True)
    df = synthetic()
    start = "2000-01-01"
    i0 = int(np.flatnonzero(df.index >= start)[0])
    for task in ("T1", "T2"):
        first_unseen = i0 - 1 if task == "T1" else i0   # T1 trains on rows <= i0-2, T2 on rows < i0
        base = frozen.frozen_linear(df, panel.S_COLS + panel.N_COLS, task, start)
        pert = df.copy()
        cols = pert.columns.get_indexer(["y_level", "y_change", "y_class"])
        pert.iloc[first_unseen:, cols] = pert.iloc[first_unseen:, cols] * -3 + 5
        again = frozen.frozen_linear(pert, panel.S_COLS + panel.N_COLS, task, start)
        assert len(base) == int((df.index >= start).sum())
        assert np.allclose(base["y_pred"], again["y_pred"])


def test_frozen_rw_predicts_no_change():
    out = frozen.frozen_rw(synthetic(), "2001-01-01")
    assert (out["y_pred"] == out["lag"]).all() and (out["class_pred"] == 1).all()


def test_frozen_neural_fits_exactly_once(monkeypatch):
    df = synthetic()
    calls = []
    real = nw.build_model
    monkeypatch.setattr(nw, "build_model", lambda *a, **k: calls.append(1) or real(*a, **k))
    sp = {s.name: s for s in nw.neural_ladder("T1")}["tft-S"]
    from dataclasses import replace
    out = nw.run_neural_spec(df, replace(sp, refit_every=frozen.NEVER), 0, "2000-01-01", device="cpu",
                             require_prereg=False, epochs=3)
    assert len(calls) == 1 and len(out) == int((df.index >= "2000-01-01").sum())


def test_run_all_and_report_with_fakes(tmp_path, monkeypatch):
    monkeypatch.setattr(frozen, "OUT", tmp_path / "frozen")
    monkeypatch.setattr(config, "RESULTS", tmp_path)
    monkeypatch.setattr(config, "REPO", tmp_path)
    monkeypatch.setattr(run_neural, "OUT", tmp_path / "h2a")
    df = synthetic(n=400)
    monkeypatch.setattr(frozen, "load_task", lambda task: df)
    monkeypatch.setattr(frozen, "prereg_ok", lambda repo=None: True)
    monkeypatch.setattr(frozen, "prereg_h2a_ok", lambda repo=None: True)

    def fake_neural(d, spec, seed, start, device="cuda", **kw):
        assert kw == {"anchor_every_row": False}
        rw = frozen.frozen_rw(d, start)
        rw["y_pred"] = rw["y_pred"] + 0.01 * seed
        return rw

    monkeypatch.setattr(frozen, "run_neural_spec", fake_neural)
    frozen.run_all(device="cpu", seeds=run_neural.SEEDS)
    for task in ("T1", "T2"):
        for yyyy in ("2016", "2020"):
            assert (tmp_path / "frozen" / f"preds_{task}_linear-S+N_{yyyy}.csv").exists()
            assert (tmp_path / "frozen" / f"preds_{task}_tft-S_{yyyy}.csv").exists()
    table = frozen.report()
    assert {"task", "start", "model", "rmse", "mae", "accuracy", "p_vs_no_change"} <= set(table.columns)
    assert (tmp_path / "docs" / "reports" / "RESULTS_FROZEN.md").exists()


def test_frozen_hybrid_never_sees_test_period_targets():
    from dataclasses import replace
    df = synthetic()
    start = "2000-01-01"
    i0 = int(np.flatnonzero(df.index >= start)[0])
    changed = {}
    for task in ("T1", "T2"):
        first_unseen = i0 - 1 if task == "T1" else i0
        sp = replace({s.name: s for s in nw.neural_ladder(task)}["hybrid-S"], refit_every=frozen.NEVER)
        pert = df.copy()
        cols = pert.columns.get_indexer(["y_level", "y_change", "y_class"])
        pert.iloc[first_unseen:, cols] = pert.iloc[first_unseen:, cols] * -3 + 5
        kw = dict(device="cpu", require_prereg=False, epochs=3)
        a = nw.run_neural_spec(df, sp, 0, start, anchor_every_row=False, **kw)
        b = nw.run_neural_spec(pert, sp, 0, start, anchor_every_row=False, **kw)
        assert np.allclose(a["y_pred"], b["y_pred"])
        if task == "T1":   # perturbed T2 classes would be invalid labels for a per-row refit
            c = nw.run_neural_spec(df, sp, 0, start, anchor_every_row=True, **kw)
            d = nw.run_neural_spec(pert, sp, 0, start, anchor_every_row=True, **kw)
            changed[task] = not np.allclose(c["y_pred"], d["y_pred"])
    assert changed["T1"]
