import numpy as np
import pandas as pd
import pytest

from fed_news import config, evaluate_h2a as ev, run_neural


def test_variance_rule():
    assert ev.variance_rule([0.20, 0.21, 0.19, 0.20], [0.15, 0.16, 0.14, 0.15])
    assert not ev.variance_rule([0.20, 0.25, 0.15, 0.20], [0.18, 0.23, 0.13, 0.18])


def fake_preds(dates, err, cls_ok):
    y = np.full(len(dates), 2.0)
    return pd.DataFrame({"date": dates, "y_true": y, "y_pred": y + err, "lag": 2.0,
                         "class_true": 1, "class_pred": np.where(cls_ok, 1, 2)})


def write_all(tmp_path, rng):
    for task, archs in ev.ARCHS.items():
        dates = pd.date_range("2016-01-01", periods=40, freq="MS")   # inside the paper window
        for arch in archs:
            for info, scale in (("S", 0.3), ("S+N", 0.1)):
                for s in run_neural.SEEDS:
                    err = rng.normal(scale=scale, size=40)
                    fake_preds(dates, err, rng.random(40) < (0.5 if info == "S" else 0.9)).to_csv(
                        run_neural.seed_path(task, f"{arch}-{info}", s), index=False)
                err = rng.normal(scale=scale, size=40)
                fake_preds(dates, err, rng.random(40) < (0.5 if info == "S" else 0.9)).to_csv(
                    run_neural.ensemble_path(task, f"{arch}-{info}"), index=False)


def test_pair_tests_t2_has_dm_and_mcnemar(tmp_path, monkeypatch):
    monkeypatch.setattr(run_neural, "OUT", tmp_path)
    write_all(tmp_path, np.random.default_rng(0))
    out = ev.pair_tests("T2", "tft")
    assert set(out) == {"H2a T2 tft DM", "H2a T2 tft McNemar"}
    assert all(0 <= p <= 1 for p in out.values())
    assert set(ev.pair_tests("T1", "lstm")) == {"H2a T1 lstm DM"}


def test_main_writes_tables_and_report(tmp_path, monkeypatch):
    monkeypatch.setattr(run_neural, "OUT", tmp_path)
    monkeypatch.setattr(config, "RESULTS", tmp_path)
    monkeypatch.setattr(config, "REPO", tmp_path)
    write_all(tmp_path, np.random.default_rng(1))
    pd.DataFrame({"hypothesis": ["H2c T1 S+N vs rw"], "p": [0.05], "holm_within_plan1": [0.05]}).to_csv(
        tmp_path / "h2_pvalues.csv", index=False)
    ev.main()
    for name in ("seed_metrics.csv", "ensemble_metrics.csv", "tests.csv"):
        assert (tmp_path / name).exists()
    tests = pd.read_csv(tmp_path / "tests.csv")
    assert {"hypothesis", "p", "holm_h2_so_far", "variance_rule"} <= set(tests.columns)
    assert "H2c T1 S+N vs rw" in set(tests["hypothesis"])
    report = (tmp_path / "RESULTS_H2A.md").read_text(encoding="utf-8")
    assert "Holm" in report and "variance rule" in report.lower()
