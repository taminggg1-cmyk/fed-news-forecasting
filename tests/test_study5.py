import numpy as np
import pandas as pd
import pytest

from fed_news import study5 as s5


def _meetings():
    d = pd.to_datetime(["2005-01-10", "2005-02-20", "2005-03-01", "2005-04-10", "2005-05-20", "2005-07-01"])
    return pd.DataFrame({"cutoff": d - pd.Timedelta(days=1),
                         "target_before": [1.00, 1.25, 1.50, 1.75, 1.75, 1.50],
                         "target_after": [1.25, 1.50, 1.75, 1.75, 1.50, 1.75],
                         "unscheduled": [0, 0, 1, 0, 0, 0]}, index=d)


def test_horizon_target_is_net_change_over_scheduled_meetings_including_unscheduled_moves():
    t = s5.horizon_targets(_meetings(), 2)
    assert list(t.index) == list(_meetings().index[[0, 1, 3, 4, 5]])  # unscheduled meeting is not a row
    # row 2005-02-20: before 1.25, after the next scheduled meeting (04-10) 1.75; the unscheduled +0.25 counts
    assert t.loc["2005-02-20", "change"] == pytest.approx(0.50) and t.loc["2005-02-20", "y_class"] == 2
    assert t.loc["2005-02-20", "end_date"] == pd.Timestamp("2005-04-10")
    assert t.loc["2005-02-20", "days"] == (pd.Timestamp("2005-04-10") - pd.Timestamp("2005-02-19")).days
    # row 2005-04-10: hold then cut to 1.50 -> net -0.25
    assert t.loc["2005-04-10", "change"] == pytest.approx(-0.25) and t.loc["2005-04-10", "y_class"] == 0
    assert np.isnan(t.loc["2005-07-01", "change"])  # horizon runs past the data


def test_net_change_of_a_hike_then_equal_cut_is_a_hold():
    m = _meetings()
    m["target_after"] = [1.25, 1.00, 1.00, 1.00, 1.00, 1.00]
    m["target_before"] = [1.00, 1.25, 1.00, 1.00, 1.00, 1.00]
    t = s5.horizon_targets(m, 2)
    assert t.loc["2005-01-10", "change"] == pytest.approx(0.0) and t.loc["2005-01-10", "y_class"] == 1


def test_training_rows_need_their_whole_horizon_decided_before_the_cutoff():
    cutoffs = pd.to_datetime(["2005-01-09", "2005-02-19", "2005-04-09", "2005-05-19", "2005-06-30"])
    ends = pd.to_datetime(["2005-04-10", "2005-05-20", "2005-07-01", "2005-08-10", "2005-09-20"])
    mask = s5.train_rows(ends, cutoffs, 4)
    assert mask.tolist() == [True, True, False, False, False]  # row 2 ends 07-01, after the 06-30 cutoff


def test_purged_folds_drop_the_rows_whose_horizons_overlap_validation():
    for tr, va in s5.purged_folds(40, k=4, gap=2):
        assert tr.max() < va.min() - 2 and len(set(tr) & set(va)) == 0


def test_stance_window_starts_at_the_previous_scheduled_cutoff():
    rows = pd.DataFrame({"cutoff": pd.to_datetime(["2005-01-09", "2005-02-19", "2005-04-09"])},
                        index=pd.to_datetime(["2005-01-10", "2005-02-20", "2005-04-10"]))
    docs = pd.DataFrame({"type": ["speech", "speech", "speech"], "stance": [1.0, -1.0, 0.5],
                         "release": pd.to_datetime(["2005-01-05", "2005-03-01", "2005-04-09"])})
    f = s5.stance_features(docs, rows, first_start=pd.Timestamp("2004-12-01"))
    # row 2's window is empty, so row 1's value carries forward; the 04-09 document sits on row 3's cutoff, so
    # it is not before it and only the 03-01 document counts
    assert f["n_stance"].tolist() == [1.0, 1.0, -1.0]
    assert f["n_stance_chg"].tolist()[1:] == [0.0, -2.0]


def test_newey_west_interval_agrees_with_the_one_sided_test():
    rng = np.random.default_rng(0)
    for shift in (0.0, 0.05, 0.2):
        r = s5.nw_summary(shift + rng.normal(0, 0.3, 200), lags=8)
        assert (r["p_one_sided"] < 0.05) == (r["ci90"][0] > 0)


def test_smallest_effect_statements():
    assert "ruled out" in s5.effect_statement([-0.01, 0.015], supported=False)
    assert s5.effect_statement([-0.01, 0.05], supported=False) == "inconclusive"
    assert "smaller than the threshold" in s5.effect_statement([0.005, 0.05], supported=True)
    assert "at least the threshold" in s5.effect_statement([0.03, 0.08], supported=True)


def test_fallback_check_refuses_a_series_whose_vintages_start_too_late():
    first = {"m2": pd.Timestamp("1980-02-08"), "u6": pd.Timestamp("2011-03-04")}
    s5.check_vintages(first, ["m2"], pd.Timestamp("1995-03-27"))
    with pytest.raises(ValueError):
        s5.check_vintages(first, ["m2", "u6"], pd.Timestamp("1995-03-27"))


def test_walk_forward_trains_only_on_decided_rows_and_returns_probabilities():
    rng = np.random.default_rng(0)
    n = 60
    idx = pd.date_range("1998-01-01", periods=n, freq="45D")
    df = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)}, index=idx)
    df["cutoff"] = idx - pd.Timedelta(days=1)
    df["end_date"] = idx + pd.Timedelta(days=90)
    df["y_change"] = 0.25 * np.sign(df["a"]) * (df["a"].abs() > 0.5)
    df["y_class"] = np.where(df["y_change"] > 0.1, 2, np.where(df["y_change"] < -0.1, 0, 1))
    df["s_rate_lag"] = 2.0
    seen = []
    out = s5.walk_forward(df, ["a", "b"], h=3, test_start=idx[40], on_fit=lambda i, tr: seen.append((i, tr)))
    assert len(out) == 20 and np.allclose(out[["p_cut", "p_hold", "p_hike"]].sum(axis=1), 1)
    for i, tr in seen:
        assert (df["end_date"].to_numpy()[tr] < df["cutoff"].to_numpy()[i]).all()
