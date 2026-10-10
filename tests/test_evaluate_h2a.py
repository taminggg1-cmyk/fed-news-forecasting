import numpy as np
import pandas as pd
import pytest

from fed_news import config, evaluate_h2a as ev, neural_walk, run_neural


def test_variance_rule():
    assert ev.variance_rule([0.20, 0.21, 0.19, 0.20], [0.15, 0.16, 0.14, 0.15])
    assert not ev.variance_rule([0.20, 0.25, 0.15, 0.20], [0.18, 0.23, 0.13, 0.18])


def test_variance_rule_uses_sample_sd_ddof_1():
    # gap 0.25; threshold 0.2 with ddof 0 (would pass) and 0.283 with ddof 1 (fails)
    base, new = [0.9, 1.1], [0.75, 0.75]
    assert 0.25 > 2 * np.std(base, ddof=0)
    assert not ev.variance_rule(base, new)


def fake_preds(dates, err, cls_ok):
    y = np.full(len(dates), 2.0)
    return pd.DataFrame({"date": dates, "y_true": y, "y_pred": y + err, "lag": 2.0,
                         "class_true": 1, "class_pred": np.where(cls_ok, 1, 2)})


def write_all(tmp_path, rng):
    for task in ev.ARCHS:   # Plan 1 context rows
        dates = pd.date_range("2016-01-01", periods=40, freq="MS")
        for name in ("rw", "S", "S+N"):
            fake_preds(dates, rng.normal(scale=0.2, size=40), rng.random(40) < 0.7).to_csv(
                tmp_path / f"preds_{task}_{name}.csv", index=False)
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


def write_ensembles(tmp_path, rng, scale_s, scale_sn):
    dates = pd.date_range("2016-01-01", periods=60, freq="MS")
    for name, scale in (("S", scale_s), ("S+N", scale_sn)):
        fake_preds(dates, rng.normal(scale=scale, size=60), np.ones(60, bool)).to_csv(
            run_neural.ensemble_path("T1", f"tft-{name}"), index=False)


def test_pair_tests_one_sided_direction(tmp_path, monkeypatch):
    monkeypatch.setattr(run_neural, "OUT", tmp_path)
    write_ensembles(tmp_path, np.random.default_rng(5), 0.4, 0.1)      # S+N clearly better
    assert ev.pair_tests("T1", "tft")["H2a T1 tft DM"] < 0.05
    write_ensembles(tmp_path, np.random.default_rng(5), 0.1, 0.4)      # S+N clearly worse
    assert ev.pair_tests("T1", "tft")["H2a T1 tft DM"] > 0.5


def test_no_neural_spec_uses_a_target_column():
    for task in ("T1", "T2"):
        for spec in neural_walk.neural_ladder(task):
            assert not any(c.startswith("y_") for c in spec.cols), spec.name


def test_reading_of_tests_counts_come_from_the_table():
    tests = pd.DataFrame({
        "hypothesis": ["H2c T1 S+N vs rw", "H2a T1 a DM", "H2a T1 b DM", "H2a T2 c DM", "H2a T2 c McNemar"],
        "p": [0.001, 0.01, 0.04, 0.5, 0.2], "holm_h2_so_far": [0.005, 0.06, 0.12, 0.5, 0.4],
        "variance_rule": ["", True, False, False, False]})
    text = ev.reading_of_tests(tests)
    assert "4 H2a tests" in text and "0 of 4 have an adjusted p" in text
    assert "2 of 4 have a raw one-sided p" in text and "1 of those 2 also pass the variance rule" in text
    assert "H2a T1 a DM (p = 0.0100)" in text and "lower bounds" in text and "no per-architecture verdict" in text


def test_deviations_section_reads_the_plan2_block(tmp_path):
    (tmp_path / "DEVIATIONS.md").write_text(
        "### Other\n\n- x\n\n### Plan 2 (H2a) deviations\n\n- first\n- second\n\n### Later\n\n- y\n",
        encoding="utf-8")
    text = ev.deviations_section(tmp_path)
    assert "- first" in text and "- second" in text and "- x" not in text and "- y" not in text
    assert ev.deviations_section(tmp_path / "missing") == "None recorded."


def test_provenance_section_uses_git_helper(monkeypatch):
    answers = {"rev-list": "e931dfc", "rev-parse": "abc1234", "diff": " src/x.py | 2 +-"}
    monkeypatch.setattr(ev, "_git", lambda repo, *a: answers[a[0]])
    text = ev.provenance_section()
    assert "e931dfc" in text and "abc1234" in text and "src/x.py" in text
    assert "evaluate_h2a.py was committed (0554690) before the first forecast file was written. Later edits" in text


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
    (tmp_path / "DEVIATIONS.md").write_text("### Plan 2 (H2a) deviations\n\n- marker line\n\n", encoding="utf-8")
    ev.main()
    for name in ("seed_metrics.csv", "ensemble_metrics.csv", "tests.csv"):
        assert (tmp_path / name).exists()
    tests = pd.read_csv(tmp_path / "tests.csv")
    assert {"hypothesis", "p", "holm_h2_so_far", "variance_rule"} <= set(tests.columns)
    assert "H2c T1 S+N vs rw" in set(tests["hypothesis"])
    report = (tmp_path / "RESULTS_H2A.md").read_text(encoding="utf-8")
    assert "Holm" in report and "variance rule" in report.lower()
    for head in ("## Reading of the tests", "## Deviations", "## Provenance", "## Exploratory analyses",
                 "best-epoch diagnostic not run", "re-run not done"):
        assert head in report
    assert "- marker line" in report and "(not in paper" in report
    ens = pd.read_csv(tmp_path / "ensemble_metrics.csv")
    assert {"rw", "linear"} <= set(ens["arch"])
