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


# ---- fix round 1 ----
def _setup(tmp_path, monkeypatch, n=3):
    monkeypatch.setattr(run_neural, "OUT", tmp_path)
    idx = pd.date_range("2001-01-01", periods=n, freq="MS")
    monkeypatch.setattr(run_neural, "load_task", lambda task: pd.DataFrame(index=idx))
    calls = []

    def fake(df, spec, seed, start, **kwargs):
        calls.append((spec.name, seed, start, kwargs))
        return preds(idx, [2.0] * n)

    monkeypatch.setattr(run_neural, "run_neural_spec", fake)
    return idx, calls


ARGS = ["--task", "T1", "--only", "lstm-S", "--device", "cpu"]


def test_paper_refit_every_reaches_ladder(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    seen = []
    real = run_neural.neural_ladder
    monkeypatch.setattr(run_neural, "neural_ladder", lambda task, n=1: seen.append(n) or real(task, n))
    run_neural.main(ARGS + ["--seeds", "0", "--paper-refit-every", "3"])
    assert seen == [3]
    log = json.loads((tmp_path / "run_log.json").read_text())
    assert log["T1/lstm-S/seed0"]["refit_every"] == 3


def test_run_neural_spec_called_with_exact_args(tmp_path, monkeypatch):
    _, calls = _setup(tmp_path, monkeypatch)
    run_neural.main(ARGS + ["--seeds", "3"])
    assert [(c[0], c[1]) for c in calls] == [("lstm-S", 3)]          # single integer = that seed only
    assert calls[0][2] == run_neural.config.TEST_START
    assert calls[0][3] == {"device": "cpu"}


def _first_run(tmp_path, monkeypatch):
    idx, calls = _setup(tmp_path, monkeypatch)
    run_neural.main(ARGS + ["--seeds", "0"])
    assert len(calls) == 1
    return idx, calls


def test_resume_skips_when_valid(tmp_path, monkeypatch):
    _, calls = _first_run(tmp_path, monkeypatch)
    run_neural.main(ARGS + ["--seeds", "0"])
    assert len(calls) == 1


def test_resume_recomputes_missing_log_entry(tmp_path, monkeypatch):
    _, calls = _first_run(tmp_path, monkeypatch)
    (tmp_path / "run_log.json").write_text("{}")
    run_neural.main(ARGS + ["--seeds", "0"])
    assert len(calls) == 2


def test_resume_recomputes_other_refit_every(tmp_path, monkeypatch):
    _, calls = _first_run(tmp_path, monkeypatch)
    run_neural.main(ARGS + ["--seeds", "0", "--paper-refit-every", "5"])
    assert len(calls) == 2
    assert json.loads((tmp_path / "run_log.json").read_text())["T1/lstm-S/seed0"]["refit_every"] == 5


def test_resume_recomputes_truncated_csv(tmp_path, monkeypatch):
    idx, calls = _first_run(tmp_path, monkeypatch)
    p = run_neural.seed_path("T1", "lstm-S", 0)
    pd.read_csv(p).iloc[:2].to_csv(p, index=False)
    run_neural.main(ARGS + ["--seeds", "0"])
    assert len(calls) == 2 and len(pd.read_csv(p)) == len(idx)


def test_resume_recomputes_nan_prediction(tmp_path, monkeypatch):
    _, calls = _first_run(tmp_path, monkeypatch)
    p = run_neural.seed_path("T1", "lstm-S", 0)
    f = pd.read_csv(p)
    f.loc[1, "y_pred"] = np.nan
    f.to_csv(p, index=False)
    run_neural.main(ARGS + ["--seeds", "0"])
    assert len(calls) == 2 and np.isfinite(pd.read_csv(p)["y_pred"]).all()


def test_partial_seeds_print_waiting_and_write_no_ensemble(tmp_path, monkeypatch, capsys):
    _setup(tmp_path, monkeypatch)
    run_neural.main(ARGS + ["--seeds", "0-2"])
    assert "waiting for seeds [3, 4, 5, 6, 7]" in capsys.readouterr().out
    assert not run_neural.ensemble_path("T1", "lstm-S").exists()


D2 = ["2001-01-01", "2001-02-01"]


def test_ensemble_rejects_y_true_mismatch():
    b = preds(D2, [2.0, 2.0])
    b["y_true"] = 3.0
    with pytest.raises(ValueError, match="y_true"):
        run_neural.ensemble([preds(D2, [2.0, 2.0]), b])


def test_ensemble_rejects_lag_mismatch():
    with pytest.raises(ValueError, match="lag"):
        run_neural.ensemble([preds(D2, [2.0, 2.0]), preds(D2, [2.0, 2.0], lag=1.0)])


def test_ensemble_rejects_class_true_mismatch():
    b = preds(D2, [2.0, 2.0])
    b["class_true"] = 0
    with pytest.raises(ValueError, match="class_true"):
        run_neural.ensemble([preds(D2, [2.0, 2.0]), b])


def test_ensemble_rejects_non_finite_and_empty():
    with pytest.raises(ValueError, match="finite"):
        run_neural.ensemble([preds(D2, [2.0, 2.0]), preds(D2, [2.0, np.nan])])
    with pytest.raises(ValueError):
        run_neural.ensemble([])
