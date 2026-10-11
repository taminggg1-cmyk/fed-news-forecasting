import numpy as np
import pandas as pd

from fed_news import evaluate_study3 as ev
from fed_news import study3 as s3


def write_results(folder, p_main, p_placebo):
    folder.mkdir()
    n = 30
    y = np.zeros(n)
    for m in ev.MODELS:
        pd.DataFrame({"date": pd.date_range("2001-01-01", periods=n, freq="45D"), "y_true": y,
                      "y_pred": np.full(n, 0.1), "lag": 0.0, "class_true": 1, "class_pred": 1}
                     ).to_csv(folder / f"preds_T2_{m}.csv", index=False)
    names = ["H7 Clark-West", "placebo Clark-West", "shuffled-text Clark-West", "S+M+D vs S+M Clark-West",
             "decisions McNemar", "DM S+D vs S", "DM S+D vs S+N"]
    p = [p_main, p_placebo, 0.5, 0.5, 0.5, 0.5, 0.5]
    pd.DataFrame({"test": names, "p": p, "stat": 0.0}).to_csv(folder / "tests.csv", index=False)
    pd.DataFrame({"year": [2001], "accuracy": [0.6]}).to_csv(folder / "probe_quality.csv", index=False)
    pd.DataFrame([s3.power_statement(
        pd.read_csv(folder / "preds_T2_S.csv"), pd.read_csv(folder / "preds_T2_S+D.csv").assign(y_pred=0.0))]
    ).to_csv(folder / "power.csv", index=False)


def test_report_states_verdict_and_registration(tmp_path):
    write_results(tmp_path / "res", 0.2, 0.5)
    (tmp_path / "PREREG_STUDY3_OSF.txt").write_text("https://osf.io/abcd1/", encoding="utf-8")
    text = ev.report(results=tmp_path / "res", repo=tmp_path, data_path=tmp_path / "missing.pkl")
    assert "**H7 verdict: not supported.**" in text
    assert "https://osf.io/abcd1/" in text and "30 meetings" in text


def test_report_void_when_placebo_passes(tmp_path):
    write_results(tmp_path / "res", 0.01, 0.01)
    text = ev.report(results=tmp_path / "res", repo=tmp_path, data_path=tmp_path / "missing.pkl")
    assert "**H7 verdict: void.**" in text
