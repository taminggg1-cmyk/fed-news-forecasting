import json

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from fed_news import poc_stmt as ps


def toy_meetings():
    idx = pd.to_datetime(["2000-02-02", "2000-03-21", "2000-04-10", "2000-05-16", "2000-06-28"])
    return pd.DataFrame({"cutoff": idx - pd.Timedelta(days=1),
                         "target_before": [5.50, 5.75, 6.00, 5.75, 6.25],
                         "target_after": [5.75, 6.00, 5.75, 6.25, 6.25],
                         "y_change": [0.25, 0.25, -0.25, 0.5, 0.0], "y_class": [2, 2, 0, 2, 1],
                         "unscheduled": [0, 0, 1, 0, 0]}, index=idx)


def test_lead_meetings_keeps_scheduled_rows_with_cutoff_seven_days_before():
    out = ps.lead_meetings(toy_meetings(), days=7)
    assert list(out.index) == list(pd.to_datetime(["2000-02-02", "2000-03-21", "2000-05-16", "2000-06-28"]))
    assert (out.index - out["cutoff"]).dt.days.tolist() == [7, 7, 7, 7]


def test_prev_change_spans_unscheduled_moves_between_scheduled_meetings():
    out = ps.lead_meetings(toy_meetings(), days=7)
    assert np.isnan(out["p_prev_change"].iloc[0])
    assert out["p_prev_change"].iloc[1:].tolist() == pytest.approx([0.25, 0.0, 0.5])


def test_proba3_puts_absent_classes_at_the_floor_and_rows_sum_to_one():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    model = LogisticRegression().fit(X, np.array([1, 1, 2, 2]))
    p = ps.proba3(model, X)
    assert p.shape == (4, 3)
    assert np.allclose(p.sum(1), 1.0)
    assert (p[:, 0] < 1e-5).all()


def synthetic(n=90, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 2))
    y = np.where(x[:, 0] > 0.7, 2, np.where(x[:, 0] < -0.7, 0, 1))
    df = pd.DataFrame({"a": x[:, 0], "b": x[:, 1], "y_class": y, "y_change": (y - 1) * 0.25},
                      index=pd.date_range("1996-01-01", periods=n, freq="45D"))
    return df


def test_cv_logloss_scales_inside_each_fold():
    df = synthetic()
    X, y = df[["a", "b"]].to_numpy() * [1.0, 100.0], df["y_class"].to_numpy()
    folds = [(tr, va) for tr, va in ps.inner_folds(len(y))]
    manual = []
    for tr, va in folds:
        sc = StandardScaler().fit(X[tr])
        m = LogisticRegression(C=0.5, max_iter=5000).fit(sc.transform(X[tr]), y[tr])
        manual.append(ps.log_loss_rows(ps.proba3(m, sc.transform(X[va])), y[va]).mean())
    assert ps.cv_logloss(X, y, 0.5) == pytest.approx(np.mean(manual))


def test_inner_folds_skip_folds_with_fewer_than_20_training_rows():
    assert all(len(tr) >= 20 for tr, _ in ps.inner_folds(45))
    assert len(ps.inner_folds(45)) == 3


def test_walk_forward_forecast_does_not_depend_on_later_rows():
    df = synthetic()
    start = df.index[60]
    a = ps.walk_forward(df, ["a", "b"], start, df.index[65])
    later = df.copy()
    later.iloc[61:, :2] = 99.0
    later.iloc[61:, 2] = 0
    b = ps.walk_forward(later, ["a", "b"], start, df.index[65])
    assert np.allclose(a[ps.PCOLS].iloc[0], b[ps.PCOLS].iloc[0])


def test_log_loss_and_brier_rows():
    p = np.array([[0.2, 0.5, 0.3], [0.1, 0.1, 0.8]])
    y = np.array([1, 2])
    assert ps.log_loss_rows(p, y) == pytest.approx([-np.log(0.5), -np.log(0.8)])
    assert ps.brier_rows(p, y) == pytest.approx([0.04 + 0.25 + 0.09, 0.01 + 0.01 + 0.04])


def test_ece_is_zero_when_confidence_matches_hit_rate():
    p = np.tile([0.25, 0.5, 0.25], (4, 1))
    y = np.array([1, 1, 0, 2])
    assert ps.ece(p, y) == pytest.approx(0.0)


def test_block_bootstrap_is_reproducible_and_brackets_the_mean():
    d = np.random.default_rng(1).normal(0.1, 0.05, 120)
    lo, hi = ps.block_bootstrap_ci(d, seed=0)
    assert (lo, hi) == ps.block_bootstrap_ci(d, seed=0)
    assert lo < d.mean() < hi


def test_stage2_refuses_without_a_stage1_pass(tmp_path, monkeypatch):
    monkeypatch.setattr(ps, "OUT", tmp_path)
    with pytest.raises(PermissionError):
        ps.require_stage1()
    (tmp_path / "stage1.json").write_text(json.dumps({"stage1_passed": False}))
    with pytest.raises(PermissionError):
        ps.require_stage1()
    (tmp_path / "stage1.json").write_text(json.dumps({"stage1_passed": True}))
    ps.require_stage1()
