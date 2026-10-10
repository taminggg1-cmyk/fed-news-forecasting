import numpy as np
import pandas as pd

from fed_news import config, evaluate_study2 as ev2, study2


def test_event_rows_split_by_source(monkeypatch):
    seen = []

    def fake_build(ds, y2):
        seen.append(sorted(set(ds["source"])))
        return pd.DataFrame({"x": [1]})

    def fake_run(events, label="", **kw):
        return {"pipeline": label, "cw_p": 0.5, "void_cw_p": 0.5, "verdict": "not supported"}, pd.DataFrame({"y": [0]})

    monkeypatch.setattr(study2, "build_events", fake_build)
    monkeypatch.setattr(study2, "run_event_study", fake_run)
    ds = pd.DataFrame({"source": ["corpus", "fraser", "bis"]})
    rows, wf = study2.event_rows(ds, pd.Series(dtype=float))
    assert seen == [["bis", "corpus", "fraser"], ["bis", "fraser"], ["corpus"]]
    assert list(rows["pipeline"]) == ["H6: all documents", "secondary: regional speeches only",
                                      "secondary: Plan 5 documents, new classifier"]
    assert list(rows["role"]) == ["confirmatory", "secondary", "secondary"]


def test_report(tmp_path, monkeypatch):
    out = tmp_path / "study2"
    out.mkdir()
    pd.DataFrame([{"pipeline": "H6: all documents", "role": "confirmatory", "n_test": 3300, "cw_p": 0.03,
                   "void_cw_p": 0.4, "verdict": "supported"}]).to_csv(out / "h6.csv", index=False)
    monkeypatch.setattr(config, "RESULTS", tmp_path)
    monkeypatch.setattr(config, "REPO", tmp_path)
    monkeypatch.setattr(ev2, "_git", lambda *a: "abc")
    monkeypatch.setattr(ev2, "_sha", lambda p: "deadbeef")
    ev2.main()
    text = (tmp_path / "RESULTS_STUDY2.md").read_text(encoding="utf-8")
    assert "**H6 verdict: supported.**" in text and "not run" in text and "conservative" in text
