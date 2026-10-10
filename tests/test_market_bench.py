import numpy as np
import pandas as pd

from fed_news import market_bench as mb


def panel(n=60, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("1996-01-01", periods=n, freq="MS")
    y = 3 + np.cumsum(rng.normal(scale=0.1, size=n))
    return pd.DataFrame({"cutoff": idx, "y_level": y, "s_rate_lag": np.r_[y[0], y[:-1]],
                         "y_class": 1}, index=idx)


def test_bill_forecast_is_bill_plus_training_spread():
    df = panel()
    bill = pd.Series(df["y_level"].to_numpy() - 0.25, index=df.index - pd.Timedelta(days=1))  # dated before cutoff
    out = mb.bill_forecast(df, bill, "T1", "1998-01-01")
    assert np.allclose(out["y_pred"], out["y_true"])            # constant spread 0.25 recovered exactly
    assert {"date", "y_true", "y_pred", "lag", "class_true", "class_pred"} == set(out.columns)


def test_spread_uses_only_training_rows():
    df = panel()
    bill = pd.Series(df["y_level"].to_numpy() - 0.25, index=df.index - pd.Timedelta(days=1))
    base = mb.bill_forecast(df, bill, "T1", "1998-01-01")
    pert = df.copy()
    k = 40
    pert.iloc[k:, pert.columns.get_loc("y_level")] += 5.0      # later targets change
    again = mb.bill_forecast(pert, bill, "T1", "1998-01-01")
    upto = df.index[k + 1]                                       # T1 trains on rows <= i-2
    assert np.allclose(base.set_index("date").loc[:upto, "y_pred"], again.set_index("date").loc[:upto, "y_pred"])


def test_rows_without_bill_are_skipped():
    df = panel()
    bill = pd.Series(df["y_level"].to_numpy() - 0.25, index=df.index - pd.Timedelta(days=1))
    bill = bill[bill.index >= pd.Timestamp("1999-01-01")]       # no bill before 1999
    out = mb.bill_forecast(df, bill, "T1", "1998-01-01")
    assert out["date"].min() > pd.Timestamp("1999-12-01")        # needs a bill and 12 training rows with bills


def test_fetch_error_hides_url(monkeypatch):
    def boom(url, **kw):
        raise FileNotFoundError(url)
    monkeypatch.setattr(mb.http, "get", boom)
    monkeypatch.setattr(mb, "api_key", lambda: "SECRETKEY")
    try:
        mb.fetch_series("DTB4WK")
    except RuntimeError as err:
        assert "SECRETKEY" not in str(err) and err.__cause__ is None and err.__suppress_context__
    else:
        raise AssertionError("must raise")


def test_bill_dated_on_cutoff_is_not_used():
    df = panel()
    bill = pd.Series(df["y_level"].to_numpy() - 0.25, index=df.index - pd.Timedelta(days=1))
    base = mb.bill_forecast(df, bill, "T1", "1998-01-01")
    spiked = pd.concat([bill, pd.Series(99.0, index=df["cutoff"])]).sort_index()   # sentinel dated exactly on cutoffs
    spiked = spiked[~spiked.index.duplicated()]
    again = mb.bill_forecast(df, spiked, "T1", "1998-01-01")
    assert (again["y_pred"] < 50).all() and np.allclose(base["y_pred"], again["y_pred"])
