import numpy as np
import pandas as pd
import pytest

from fed_news import study3 as s3


def preds(y_pred, y_true, seed=0):
    dates = pd.date_range("2001-01-01", periods=len(y_true), freq="45D")
    cls = lambda v: np.where(v > 0.125, 2, np.where(v < -0.125, 0, 1))
    return pd.DataFrame({"date": dates, "y_true": y_true, "y_pred": y_pred, "lag": 0.0,
                         "class_true": cls(y_true), "class_pred": cls(y_pred)})


def fake_forecasts(n=200, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.choice([-0.25, 0.0, 0.25], n, p=[0.2, 0.6, 0.2])
    base = y + rng.normal(0, 0.2, n)
    good = y + rng.normal(0, 0.1, n)
    noise = lambda: base + rng.normal(0, 0.02, n)
    return {"S": preds(base, y), "S+D": preds(good, y), "placebo": preds(noise(), y),
            "shuffled-text": preds(noise(), y), "S+M": preds(base, y), "S+M+D": preds(good, y),
            "S+N": preds(noise(), y)}


def test_decide_h7():
    assert s3.decide_h7(0.01, 0.5, 0.5) == "supported"
    assert s3.decide_h7(0.20, 0.5, 0.5) == "not supported"
    assert s3.decide_h7(0.01, 0.04, 0.5) == "void"
    assert s3.decide_h7(0.01, 0.5, 0.04) == "void"
    with pytest.raises(ValueError):
        s3.decide_h7(float("nan"), 0.5, 0.5)


def test_hypothesis_tests_names_and_direction():
    table = s3.hypothesis_tests(fake_forecasts()).set_index("test")
    assert list(table.index) == ["H7 Clark-West", "placebo Clark-West", "shuffled-text Clark-West",
                                 "S+M+D vs S+M Clark-West", "decisions McNemar", "DM S+D vs S", "DM S+D vs S+N"]
    assert table.loc["H7 Clark-West", "p"] < 0.01             # S+D is much better by construction
    assert table["p"].between(0, 1).all()
    assert (table["n"] == 200).all()


def test_same_rows_intersects_indices():
    a = pd.DataFrame({"x": range(4)}, index=[1, 2, 3, 4])
    b = pd.DataFrame({"x": range(3)}, index=[2, 3, 5])
    ra, rb = s3.same_rows(a, b)
    assert list(ra.index) == [2, 3] and list(rb.index) == [2, 3]


def test_power_statement_formula():
    y = np.zeros(100)
    base, new = preds(np.r_[np.ones(50), -np.ones(50)], y), preds(np.zeros(100), y)
    p = s3.power_statement(base, new)
    d = np.ones(100)                                          # every loss differential equals 1
    assert p["mse_base"] == pytest.approx(1.0) and p["sd_loss_diff"] == pytest.approx(np.std(d, ddof=1))
    assert p["mde_mse"] == pytest.approx(s3.Z80 * p["sd_loss_diff"] / 10)
    assert "100 meetings" in s3.power_sentence(p)
