import subprocess

import numpy as np
import pandas as pd
import pytest

from fed_news import panel, walkforward as wf


def synthetic(n=80, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("1995-01-01", periods=n, freq="MS")
    df = pd.DataFrame(rng.normal(size=(n, len(panel.S_COLS))), index=idx, columns=panel.S_COLS)
    for c in panel.M_COLS + panel.N_COLS:
        df[c] = rng.normal(size=n)
    df["t_u3"] = rng.normal(size=n)
    df["s_rate_lag"] = 2.0
    change = 0.25 * np.sign(df["n_stance"].to_numpy()) * (np.abs(df["n_stance"].to_numpy()) > 0.5)
    df["y_change"], df["y_level"] = change, 2.0 + change
    df["y_class"] = np.where(change > 0.02, 2, np.where(change < -0.02, 0, 1))
    return df


def test_train_mask_t1_ends_two_rows_back_and_t2_one():
    assert np.flatnonzero(wf.train_mask(10, 6, "T1")).tolist() == [0, 1, 2, 3, 4]
    assert np.flatnonzero(wf.train_mask(10, 6, "T2")).tolist() == [0, 1, 2, 3, 4, 5]


def test_common_rows_drops_any_missing():
    df = synthetic()
    df.iloc[3, df.columns.get_loc("n_epu")] = np.nan
    df.iloc[5, df.columns.get_loc("t_u3")] = np.nan
    out = wf.common_rows(df)
    assert len(out) == len(df) - 2 and df.index[3] not in out.index


def test_placebo_touches_only_n_columns_and_uses_training_rows():
    X = np.arange(20.0).reshape(5, 4)
    train = np.array([True, True, True, False, False])
    out = wf.placebo_matrix(X, train, 3, [2, 3], np.random.default_rng(0))
    assert np.array_equal(out[:, :2], X[:, :2])
    assert sorted(out[:3, 2].tolist()) == sorted(X[:3, 2].tolist())
    assert out[3, 2] in X[:3, 2]
    assert np.array_equal(out[4], X[4])


def test_ladder_names_and_blocks():
    names = [s.name for s in wf.ladder("T2")]
    assert names == ["rw", "taylor", "S", "S+N", "placebo", "S+M", "S+M+N"]
    spec = {s.name: s for s in wf.ladder("T2")}
    assert spec["S+N"].cols == panel.S_COLS + panel.N_COLS and spec["placebo"].placebo


def test_text_spec_refused_without_prereg(monkeypatch):
    monkeypatch.setattr(wf, "prereg_ok", lambda repo=None: False)
    spec = {s.name: s for s in wf.ladder("T2")}["S+N"]
    with pytest.raises(PermissionError):
        wf.run_spec(synthetic(), spec, "2000-01-01")


def test_primary_t2_learns_signal_and_rw_holds():
    df = synthetic(120)
    spec = {s.name: s for s in wf.ladder("T2")}
    sn = wf.run_spec(df, spec["S+N"], "2002-01-01", require_prereg=False)
    rw = wf.run_spec(df, spec["rw"], "2002-01-01", require_prereg=False)
    assert (sn["class_pred"] == sn["class_true"]).mean() > 0.6
    assert (rw["class_pred"] == 1).all() and (rw["y_pred"] == 2.0).all()


def test_prereg_ok_reads_git_tag(tmp_path):
    def git(*args):
        subprocess.run(["git", "-C", str(tmp_path), "-c", "user.name=t", "-c", "user.email=t@t", *args],
                       check=True, capture_output=True)
    git("init", "-q")
    git("commit", "-q", "--allow-empty", "-m", "x")
    assert wf.prereg_ok(tmp_path) is False
    git("tag", "prereg")
    assert wf.prereg_ok(tmp_path) is True
