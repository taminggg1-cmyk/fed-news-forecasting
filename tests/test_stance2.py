import numpy as np
import pandas as pd

from fed_news import stance, stance2


def docs():
    t = "Rates may rise again later this year if inflation stays high. Inflation remains well above the objective."
    return pd.DataFrame({"doc_id": ["a", "b", "c"], "type": ["speech", "minutes", "speech"],
                         "release": pd.to_datetime(["2005-03-01", "2005-06-01", "2006-02-01"]),
                         "text": [t, t, t], "source": ["corpus", "corpus", "fraser"]})


def labels():
    return pd.DataFrame({"sentence": [f"s{i}" for i in range(200)], "label": [i % 3 for i in range(200)],
                         "year": [1999 + (i % 10) for i in range(200)]})


def test_run_uses_all_eligible_labels_and_resumes(tmp_path, monkeypatch):
    seen = []

    def fake(train, texts, vintage, seed):
        seen.append((vintage, len(train)))
        out = np.zeros((len(texts), 3), dtype=np.float32)
        out[:, 1], out[:, 0] = 0.6, 0.1
        return out

    monkeypatch.setattr(stance, "train_and_score", fake)
    monkeypatch.setattr(stance2, "OUT_PATH", tmp_path / "ds2.pkl")
    lab = labels()
    out = stance2.run(docs(), lab, parts_dir=tmp_path / "parts")
    n2005 = int((lab["year"] < 2005).sum())
    n2006 = int((lab["year"] < 2006).sum())
    assert (2004, n2005) in seen and (2005, n2006) in seen                 # vintage y-1, ALL labels before y
    assert np.allclose(out["stance"], 0.5) and set(out["source"]) == {"corpus", "fraser"}
    assert list(out.columns) == ["doc_id", "type", "release", "stance", "n_sent", "source"]
    calls = len(seen)
    stance2.run(docs(), lab, parts_dir=tmp_path / "parts")
    assert len(seen) == calls                                              # resumed from parts, no retraining
