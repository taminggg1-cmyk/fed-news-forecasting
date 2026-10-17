import subprocess

import numpy as np
import pandas as pd

from fed_news.studies.study3_decision_text import decision_text as dt
from fed_news.features.stance import vintage_year


def meetings():
    idx = pd.to_datetime(["2000-03-21", "2000-06-28", "2001-01-31"])
    return pd.DataFrame({"y_class": [2, 1, 0]}, index=pd.Index(idx, name="date"))


def test_label_is_first_meeting_strictly_after_release():
    docs = pd.DataFrame({"doc_id": ["a", "b", "c", "d"],
                         "release": pd.to_datetime(["2000-01-05", "2000-03-21", "2000-06-27", "2001-02-01"])})
    out = dt.next_decision(docs, meetings())
    assert out["label"].tolist() == [2, 1, 1, -1]          # b is released on a meeting day: next meeting's label
    assert out["label_date"].iloc[1] == pd.Timestamp("2000-06-28")
    assert pd.isna(out["label_date"].iloc[3])


def test_training_docs_use_only_labels_known_before_the_year():
    docs = pd.DataFrame({"doc_id": list("abc"),
                         "release": pd.to_datetime(["2000-01-05", "2000-12-20", "2000-12-30"])})
    tr = dt.training_docs(dt.next_decision(docs, meetings()), 2001)
    assert tr["doc_id"].tolist() == ["a"]                  # b and c are labelled by the 2001-01-31 meeting


def test_enough_needs_100_docs_and_10_per_class():
    ok = pd.DataFrame({"label": [0] * 10 + [1] * 80 + [2] * 10})
    assert dt.enough(ok)
    assert not dt.enough(ok.iloc[1:])                      # 99 documents and 9 cuts
    assert not dt.enough(pd.DataFrame({"label": [1] * 95 + [2] * 15}))


def test_vintage_rule():
    assert [vintage_year(y) for y in (1996, 2000, 2001, 2025, 2026)] == [1999, 1999, 2000, 2024, 2024]


def separable(n=300, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 3, n)
    X = rng.normal(size=(n, 5))
    X[:, 0] += 3 * (y - 1)
    return X, y


def test_fit_probe_learns_signal_and_picks_c_from_grid():
    X, y = separable()
    model, mu, sd, c = dt.fit_probe(X, y)
    assert c in dt.C_GRID
    assert np.corrcoef(dt.doc_scores(model, mu, sd, X), y)[0, 1] > 0.8


def test_fit_probe_falls_back_when_no_fold_has_all_classes():
    y = np.array([1] * 90 + [0] * 10 + [2] * 10)           # every expanding training fold lacks the hikes
    X = np.random.default_rng(1).normal(size=(len(y), 3))
    _, _, _, c = dt.fit_probe(X, y)
    assert c == dt.FALLBACK_C


def world(seed=0, n=1200):
    rng = np.random.default_rng(seed)
    mdates = pd.date_range("1995-01-15", "2002-12-15", freq="45D")
    meet = pd.DataFrame({"y_class": np.resize([0, 1, 2, 1], len(mdates))},           # every class every year
                        index=pd.Index(mdates, name="date"))
    lo, hi = pd.Timestamp("1995-01-01").value // 10 ** 9, pd.Timestamp("2002-12-31").value // 10 ** 9
    release = pd.to_datetime(rng.integers(lo, hi, n), unit="s").normalize()
    docs = pd.DataFrame({"doc_id": [f"d{i}" for i in range(n)], "type": "speech", "release": release,
                         "source": "corpus"})
    table = {d: rng.normal(size=4) for d in docs["doc_id"]}
    return docs, meet, (lambda v, ids: np.stack([table[i] for i in ids]))


def test_score_all_scores_only_years_with_enough_labels():
    docs, meet, emb = world()
    scored, quality = dt.score_all(dt.next_decision(docs, meet), emb)
    assert scored.loc[scored["release"].dt.year == 1995, "stance"].isna().all()
    assert scored.loc[scored["release"].dt.year >= 1996, "stance"].notna().all()
    assert scored["stance"].dropna().between(-1, 1).all()
    assert {"year", "n_train", "classifier", "C", "accuracy", "majority_accuracy"} <= set(quality.columns)


def test_future_labels_do_not_change_past_scores():
    docs, meet, emb = world()
    a, _ = dt.score_all(dt.next_decision(docs, meet), emb)
    meet2 = meet.copy()
    later = meet2.index >= pd.Timestamp("2000-01-01")
    meet2.loc[later, "y_class"] = (meet2.loc[later, "y_class"] + 1) % 3
    b, _ = dt.score_all(dt.next_decision(docs, meet2), emb)
    early = (a["release"].dt.year < 2000).to_numpy()
    np.testing.assert_allclose(a["stance"].to_numpy()[early], b["stance"].to_numpy()[early])


def test_shuffled_labels_change_scores_but_not_coverage():
    docs, meet, emb = world()
    lab = dt.next_decision(docs, meet)
    real, _ = dt.score_all(lab, emb)
    shuf, _ = dt.score_all(lab, emb, shuffle_seed=0)
    assert (real["stance"].isna() == shuf["stance"].isna()).all()
    assert not np.allclose(real["stance"].dropna(), shuf["stance"].dropna())


def git(repo, *args):
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def test_gate_needs_tag_and_osf_url(tmp_path):
    git(tmp_path, "init", "-q")
    git(tmp_path, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "--allow-empty", "-m", "x")
    assert "tag" in dt.gate(tmp_path)
    git(tmp_path, "tag", dt.TAG)
    assert "OSF" in dt.gate(tmp_path)
    (tmp_path / dt.OSF_NAME).write_text("https://osf.io/abcd1/\n", encoding="utf-8")
    assert dt.gate(tmp_path) is None


import pytest


def test_run_refuses_without_registration(tmp_path, monkeypatch):
    monkeypatch.setattr(dt, "gate", lambda repo=None: "tag prereg-study3 missing")
    with pytest.raises(PermissionError, match="refused"):
        dt.run(out_path=tmp_path / "a.pkl", shuffled_path=tmp_path / "b.pkl", results=tmp_path)


def test_run_writes_real_and_shuffled_scores(tmp_path, monkeypatch):
    monkeypatch.setattr(dt, "gate", lambda repo=None: None)
    docs, meet, emb = world()
    dt.run(docs=docs.assign(text="x"), meetings=meet, embed=emb, out_path=tmp_path / "real.pkl",
           shuffled_path=tmp_path / "shuf.pkl", results=tmp_path / "res")
    real, shuf = pd.read_pickle(tmp_path / "real.pkl"), pd.read_pickle(tmp_path / "shuf.pkl")
    assert {"doc_id", "type", "release", "source", "label", "label_date", "year", "stance"} <= set(real.columns)
    assert "text" not in real.columns
    assert (real["doc_id"] == shuf["doc_id"]).all()
    assert (tmp_path / "res" / "probe_quality.csv").exists()
    assert (tmp_path / "res" / "probe_quality_shuffled.csv").exists()


def test_run_refuses_duplicate_doc_ids(tmp_path, monkeypatch):
    monkeypatch.setattr(dt, "gate", lambda repo=None: None)
    docs, meet, emb = world()
    docs = pd.concat([docs, docs.iloc[:1]], ignore_index=True).assign(text="x")
    with pytest.raises(ValueError, match="duplicate"):
        dt.run(docs=docs, meetings=meet, embed=emb, out_path=tmp_path / "a.pkl",
               shuffled_path=tmp_path / "b.pkl", results=tmp_path)


def test_poc_labels_never_keep_test_window_outcomes():
    docs = pd.DataFrame({"doc_id": list("abc"),
                         "release": pd.to_datetime(["2000-01-05", "2000-12-20", "2001-02-01"])})
    out = dt.poc_labels(dt.next_decision(docs, meetings()))
    assert out["doc_id"].tolist() == ["a", "b"]             # 2001 releases dropped
    assert out["label"].tolist() == [2, -1]                 # b's label is the 2001-01-31 decision: removed
    assert pd.isna(out["label_date"].iloc[1])


def test_poc_summary_pass_mark():
    quality = pd.DataFrame({"year": [1997, 1998], "classifier": [True, True], "n_scored_labelled": [10, 30],
                            "accuracy": [0.6, 0.5], "majority_accuracy": [0.5, 0.4]})
    scored = pd.DataFrame({"label": [0, 1, 2, 2, -1], "stance": [-0.5, 0.0, 0.4, 0.6, 0.9]})
    s = dt.poc_summary(scored, quality)
    assert s["accuracy"] == pytest.approx(21 / 40) and s["majority_accuracy"] == pytest.approx(17 / 40)
    assert s["spearman"] > 0 and s["passed"]
    assert not dt.poc_summary(scored, quality.assign(accuracy=[0.4, 0.3]))["passed"]


def test_poc_runs_without_registration(tmp_path, monkeypatch):
    monkeypatch.setattr(dt, "gate", lambda repo=None: "tag prereg-study3 missing")
    docs, meet, emb = world()
    s = dt.poc(docs=docs.assign(text="x"), meetings=meet, embed=emb, results=tmp_path)
    assert (tmp_path / "poc_quality.csv").exists() and (tmp_path / "poc_summary.csv").exists()
    assert set(s) >= {"n_docs", "accuracy", "majority_accuracy", "spearman", "passed"}


def test_centred_embedder_subtracts_mean_of_recent_earlier_documents():
    docs = pd.DataFrame({"doc_id": list("abcde"),
                         "release": pd.to_datetime(["2000-01-01", "2000-02-01", "2000-03-01", "2000-03-01",
                                                    "2001-06-01"])})
    table = {"a": np.array([1.0, 0.0]), "b": np.array([3.0, 0.0]), "c": np.array([10.0, 1.0]),
             "d": np.array([0.0, 5.0]), "e": np.array([7.0, 7.0])}
    raw = lambda v, ids: np.stack([table[i] for i in ids])
    f = dt.centred_embedder(docs, raw, window_days=180, min_docs=2)
    out = f(1999, ["c", "d", "a", "e"])
    np.testing.assert_allclose(out[0], [10.0 - 2.0, 1.0])  # c: mean of a, b (d is same day, not earlier)
    np.testing.assert_allclose(out[1], [0.0 - 2.0, 5.0])
    assert np.isnan(out[2]).all()                          # a: no earlier documents
    assert np.isnan(out[3]).all()                          # e: nothing in the previous 180 days


def test_score_all_skips_documents_without_a_finite_embedding():
    docs, meet, emb = world()
    blank = set(docs["doc_id"].iloc[::7])
    holey = lambda v, ids: np.stack([np.full(4, np.nan) if i in blank else emb(v, [i])[0] for i in ids])
    scored, _ = dt.score_all(dt.next_decision(docs, meet), holey)
    assert scored.loc[scored["doc_id"].isin(blank), "stance"].isna().all()
    assert scored.loc[~scored["doc_id"].isin(blank) & (scored["release"].dt.year >= 1996), "stance"].notna().all()


def test_poc_compares_with_the_old_score():
    docs, meet, emb = world()
    old = pd.DataFrame({"doc_id": docs["doc_id"], "stance": np.random.default_rng(3).normal(size=len(docs))})
    s = dt.poc(docs=docs.assign(text="x"), meetings=meet, embed=emb, results=None, old_scores=old)
    assert {"spearman_common", "spearman_old", "passed_v2"} <= set(s)
    assert s["passed_v2"] == (s["spearman_common"] > 0 and s["spearman_common"] > s["spearman_old"])
