import numpy as np
import pandas as pd

from fed_news.studies.plan2_neural_original import diag_tft, neural_walk, run_neural
from test_neural_walk import synthetic


def test_diag_records_epochs_writes_rerun_files_and_metrics(tmp_path, monkeypatch):
    out, committed = tmp_path / "diag", tmp_path / "h2a"
    out.mkdir(), committed.mkdir()
    monkeypatch.setattr(run_neural, "OUT", committed)
    df = synthetic()
    names = {"tft-S", "hybrid-S", "hybrid-S+N"}
    specs = {"T1": [s for s in diag_tft.diag_specs("T1") if s.name in names]}
    assert {s.name for s in specs["T1"]} == names and all(s.early_stop for s in specs["T1"])
    table = diag_tft.run_diag({"T1": df}, device="cpu", seeds=(0,), out=out, test_start="2001-01-01", epochs=3,
                              require_prereg=False, specs=specs)
    n_refits = len(range(0, int((df.index >= "2001-01-01").sum()), 12))
    assert list(table.columns) == diag_tft.EPOCH_COLS
    assert len(table) == 3 * n_refits and table["best_epoch"].between(0, 3).all()
    assert set(table["spec"]) == names
    for name in names:
        assert diag_tft.rerun_path(out, "T1", name, 0).exists()
    assert not list(committed.iterdir())          # nothing written to the committed directory

    # committed forecasts = the re-run ones, so the nondeterminism bound is exactly zero
    for name in names:
        frame = pd.read_csv(diag_tft.rerun_path(out, "T1", name, 0), parse_dates=["date"])
        frame.to_csv(run_neural.seed_path("T1", name, 0), index=False)
    metrics = diag_tft.rerun_metrics(tasks=("T1",), seeds=(0,), out=out, specs=specs)
    assert set(metrics["spec"]) == names
    assert (metrics["max_abs_forecast_diff"] == 0).all()
    assert np.allclose(metrics["committed_rmse"], metrics["rerun_rmse"])
    assert (out / "rerun_metrics.csv").exists()
    row = metrics[metrics["spec"] == "hybrid-S+N"].iloc[0]
    assert row["committed_dm_p"] == row["rerun_dm_p"]

    # a second call resumes: nothing is re-run
    def boom(*a, **k):
        raise AssertionError("re-ran a finished seed")
    monkeypatch.setattr(diag_tft, "run_neural_spec", boom)
    again = diag_tft.run_diag({"T1": df}, device="cpu", seeds=(0,), out=out, test_start="2001-01-01", epochs=3,
                              require_prereg=False, specs=specs)
    assert len(again) == len(table)
