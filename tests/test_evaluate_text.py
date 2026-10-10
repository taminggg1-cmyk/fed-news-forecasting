import pandas as pd

from fed_news import config, evaluate_text as et


def test_report_with_all_inputs(tmp_path, monkeypatch):
    out = tmp_path / "text"
    out.mkdir()
    pd.DataFrame([{"pipeline": "chronobert", "n_test": 1800, "cw_t": 2.5, "cw_p": 0.006, "void_cw_t": 0.1,
                   "void_cw_p": 0.46, "verdict": "supported", "sign_accuracy": 0.55}]).to_csv(out / "event_study.csv", index=False)
    pd.DataFrame([{"task": "T1", "pipeline": "naive", "rmse": 0.15}]).to_csv(out / "naive_comparison.csv", index=False)
    monkeypatch.setattr(config, "RESULTS", tmp_path)
    monkeypatch.setattr(config, "REPO", tmp_path)
    monkeypatch.setattr(et, "_git", lambda *a: "abc123")
    et.main()
    text = (tmp_path / "RESULTS_TEXT.md").read_text(encoding="utf-8")
    assert "**H5 verdict: supported.**" in text
    assert "exploratory" in text.lower() and "not run" in text      # missing ablation / market sections
