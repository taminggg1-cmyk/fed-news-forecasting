import numpy as np
import pytest
import optuna
import pandas as pd
import torch

from fed_news import neural, tune


def _df(n=60, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("1996-01-01", periods=n, freq="45D")
    df = pd.DataFrame(rng.normal(size=(n, 3)), index=idx, columns=["a", "b", "c"])
    df["y_change"] = 0.25 * np.sign(df["a"]) * (df["a"].abs() > 0.8)
    df["s_rate_lag"] = 3.0
    df["y_level"] = df["s_rate_lag"] + df["y_change"]
    df["y_class"] = np.where(df["y_change"] > 0.1, 2, np.where(df["y_change"] < -0.1, 0, 1))
    return df


def test_rnn_can_stack_layers_with_dropout():
    m = neural.build_model("gru", 3, 8, 0.3, seed=0, layers=2)
    assert m.rnn.num_layers == 2 and m.rnn.dropout == 0.3
    assert m(torch.zeros(5, 4, 3)).shape == (5,)
    assert neural.build_model("lstm", 3, 8, 0.3, seed=0).rnn.num_layers == 1


def test_fit_validate_reports_the_best_validation_epoch():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 2, 3)).astype(np.float32)
    y = X[:, -1, 0] * 2
    m = neural.build_model("lstm", 3, 8, 0.0, seed=0)
    best, epoch = neural.fit_validate(m, X[:60], y[:60], X[60:], y[60:], lr=1e-2, weight_decay=0.0, batch_size=16,
                                      max_epochs=50, patience=10, device="cpu", seed=0)
    assert 1 <= epoch <= 50 and best < float(np.mean(y[60:] ** 2))


def test_tuning_points_are_the_first_test_meeting_of_each_year():
    dates = pd.to_datetime(["2000-12-01", "2001-02-01", "2001-03-01", "2005-01-20", "2009-06-01"])
    assert tune.tuning_points(dates, years=(2001, 2005, 2009), test_start="2001-01-01") == [1, 3, 4]
    assert tune.point_for(2, [1, 3, 4]) == 1 and tune.point_for(3, [1, 3, 4]) == 3


def test_search_space_gives_valid_settings_for_each_architecture():
    for arch in tune.ARCHS:
        study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
        p = tune.suggest(study.ask(), arch)
        assert p["lookback"] in (2, 4, 8, 12) and 1e-4 <= p["lr"] <= 1e-2
        assert ("heads" in p) == (arch == "tft") and ("layers" in p) == (arch != "tft")


def test_cv_score_never_reads_rows_at_or_after_the_tuning_point():
    df = _df()
    df.iloc[40:, :3] = np.nan  # anything after the point would turn the score into NaN
    p = {"hidden": 8, "dropout": 0.0, "lr": 1e-2, "weight_decay": 0.0, "lookback": 2, "batch": 16, "layers": 1}
    score, epoch = tune.cv_score(df[["a", "b", "c"]].to_numpy(float), df["y_change"].to_numpy(float), 40, "lstm",
                                 p, device="cpu", max_epochs=5)
    assert np.isfinite(score) and epoch >= 1


def test_run_tuned_uses_the_settings_of_the_latest_point_and_forecasts_every_test_row():
    df = _df()
    p = {"hidden": 8, "dropout": 0.0, "lr": 1e-2, "weight_decay": 0.0, "lookback": 2, "batch": 16, "layers": 1,
         "epochs": 3}
    settings = {30: dict(p), 45: dict(p, lookback=4)}
    used = []
    out = tune.run_tuned(df, ["a", "b", "c"], "gru", settings, seed=0, device="cpu", test_start=df.index[30],
                         on_refit=lambda i, s: used.append((i, s["lookback"])))
    assert len(out) == 30 and np.isfinite(out["y_pred"]).all()
    assert (30, 2) in used and (45, 4) in used and all(i < 45 or lb == 4 for i, lb in used)


def test_placebo_panel_shifts_only_the_stance_columns_by_half_the_rows():
    df = _df(10).assign(n_stance=np.arange(10.0), n_stance_chg=np.arange(10.0) * 2)
    out = tune.placebo_panel(df)
    assert out["n_stance"].tolist() == [5.0, 6, 7, 8, 9, 0, 1, 2, 3, 4]
    assert out["n_stance_chg"].tolist() == (np.roll(np.arange(10.0) * 2, 5)).tolist()
    assert out.drop(columns=["n_stance", "n_stance_chg"]).equals(df.drop(columns=["n_stance", "n_stance_chg"]))


def test_placebo_arms_use_the_real_columns_and_the_comparisons_cover_both_questions():
    for reader in ("C", "H"):
        assert tune.FEATURES[f"placebo-{reader}"][1] == tune.FEATURES[f"S+M+{reader}"][1]
        assert tune.FEATURES[f"placebo-{reader}"][0] == reader + "p"
        assert (f"placebo-{reader}", "S+M") in tune.PAIRS and (f"S+M+{reader}", f"placebo-{reader}") in tune.PAIRS


def test_monthly_task_uses_monthly_lookbacks_and_the_paper_inputs():
    cfg = tune.TASKS["T1"]
    assert cfg.lookbacks == (3, 6, 12, 24) and cfg.refit_every == 12 and cfg.gap == 1 and cfg.base == "S"
    assert cfg.features["S"][1] == tune.S_COLS and cfg.features["placebo-H"][1] == tune.S_COLS + tune.D
    assert ("S+C", "placebo-C") in cfg.pairs and ("placebo-H", "S") in cfg.pairs
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
    assert tune.suggest(study.ask(), "lstm", cfg.lookbacks)["lookback"] in (3, 6, 12, 24)


def test_monthly_tuning_skips_the_month_before_the_point():
    df = _df()
    df.iloc[39, :3] = np.nan  # month t-1 of a point at 40 is not known yet
    p = {"hidden": 8, "dropout": 0.0, "lr": 1e-2, "weight_decay": 0.0, "lookback": 3, "batch": 16, "layers": 1}
    score, _ = tune.cv_score(df[["a", "b", "c"]].to_numpy(float), df["y_change"].to_numpy(float), 40, "lstm", p,
                             device="cpu", max_epochs=3, gap=1)
    assert np.isfinite(score)


def test_monthly_walk_forward_trains_on_rows_up_to_t_minus_2():
    df = _df()
    df.iloc[29, df.columns.get_loc("y_change")] = np.nan  # the row just before the first forecast
    p = {"hidden": 8, "dropout": 0.0, "lr": 1e-2, "weight_decay": 0.0, "lookback": 3, "batch": 16, "layers": 1,
         "epochs": 2}
    out = tune.run_tuned(df, ["a", "b", "c"], "gru", {30: p}, seed=0, device="cpu", test_start=df.index[30],
                         task="T1", refit_every=100)
    assert len(out) == 30
    with pytest.raises(ValueError):
        tune.run_tuned(df, ["a", "b", "c"], "gru", {30: p}, seed=0, device="cpu", test_start=df.index[30],
                       task="T2", refit_every=100)
