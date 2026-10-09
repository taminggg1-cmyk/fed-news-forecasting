import json

import numpy as np
import pandas as pd
import pytest

from fed_news import run_neural


def preds(dates, y_pred, lag=2.0):
    return pd.DataFrame({"date": pd.to_datetime(dates), "y_true": 2.0, "y_pred": y_pred, "lag": lag,
                         "class_true": 1, "class_pred": 1})


def test_ensemble_is_seed_mean_with_band_class():
    d = ["2001-01-01", "2001-02-01"]
    out = run_neural.ensemble([preds(d, [2.0, 2.3]), preds(d, [2.2, 2.3])])
    assert out["y_pred"].tolist() == pytest.approx([2.1, 2.3])
    assert out["class_pred"].tolist() == [1, 2]       # changes 0.1 (hold) and 0.3 (> 0.125, hike)


def test_ensemble_refuses_mismatched_dates():
    with pytest.raises(ValueError, match="dates"):
        run_neural.ensemble([preds(["2001-01-01"], [2.0]), preds(["2001-02-01"], [2.0])])


def test_too_slow_is_more_than_one_hour():
    assert run_neural.too_slow(3600.1) and not run_neural.too_slow(3600.0)


def test_main_runs_selected_specs_resumes_and_ensembles(tmp_path, monkeypatch):
    monkeypatch.setattr(run_neural, "OUT", tmp_path)
    idx = pd.date_range("2001-01-01", periods=3, freq="MS")
    monkeypatch.setattr(run_neural, "load_task", lambda task: pd.DataFrame(index=idx))
    calls = []

    def fake(df, spec, seed, start, device):
        calls.append((spec.name, seed))
        return preds(idx, [2.0 + 0.01 * seed] * 3)

    monkeypatch.setattr(run_neural, "run_neural_spec", lambda df, spec, seed, start, device="cuda": fake(df, spec, seed, start, device))
    run_neural.main(["--task", "T2", "--only", "tft-S", "--seeds", "0-7", "--device", "cpu"])
    assert len(calls) == 8
    assert run_neural.ensemble_path("T2", "tft-S").exists()
    log = json.loads((tmp_path / "run_log.json").read_text())
    assert log["T2/tft-S/seed3"]["refit_every"] == 8
    run_neural.main(["--task", "T2", "--only", "tft-S", "--seeds", "0-7", "--device", "cpu"])
    assert len(calls) == 8                                 # resumed: nothing rerun


def test_no_ensemble_until_all_eight_seeds(tmp_path, monkeypatch):
    monkeypatch.setattr(run_neural, "OUT", tmp_path)
    idx = pd.date_range("2001-01-01", periods=2, freq="MS")
    monkeypatch.setattr(run_neural, "load_task", lambda task: pd.DataFrame(index=idx))
    monkeypatch.setattr(run_neural, "run_neural_spec", lambda df, spec, seed, start, device="cuda": preds(idx, [2.0, 2.0]))
    run_neural.main(["--task", "T1", "--only", "lstm-S", "--seeds", "0-2", "--device", "cpu"])
    assert not run_neural.ensemble_path("T1", "lstm-S").exists()
