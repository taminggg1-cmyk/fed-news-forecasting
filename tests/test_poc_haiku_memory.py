import numpy as np
import pandas as pd
import pytest

from fed_news.studies.poc import poc_haiku_memory as hm


def test_statement_text_is_cut_at_the_vote_and_masked():
    text = ("Release Date: May 3, 2005. The Committee decided today to raise its target by 25 basis points to "
            "3 percent. Voting for the FOMC monetary policy action were: Alan Greenspan, Chairman.")
    out = hm.prepare(text, {"greenspan"})
    assert "voting" not in out and "greenspan" not in out and "2005" not in out and "raise its target" in out


def test_parse_next_letters():
    assert hm.parse_next('{"1": "C", "2": "h", "3": "R"}', 3) == [-1, 0, 1]
    with pytest.raises(ValueError):
        hm.parse_next('{"1": "X"}', 1)


def test_direction_uses_the_class_band():
    assert hm.direction(np.array([-0.5, -0.1, 0.0, 0.25])).tolist() == [-1, 0, 0, 1]


def test_verdict_passes_when_haiku_is_no_better_than_persistence_and_not_date_dependent():
    truth = np.array([0, 0, 1, -1] * 25)
    persistence = truth.copy()
    haiku = truth.copy()
    dated = np.array([True, False] * 50)
    assert hm.verdict(truth, haiku, persistence, dated)["passed"] is True


def test_verdict_fails_when_accuracy_depends_on_knowing_the_date():
    rng = np.random.default_rng(0)
    truth = rng.choice([-1, 0, 1], 200)
    dated = np.arange(200) < 100
    haiku = np.where(dated, truth, rng.choice([-1, 0, 1], 200))
    persistence = np.where(rng.random(200) < 0.6, truth, 0)
    res = hm.verdict(truth, haiku, persistence, dated)
    assert res["date_dependence_flag"] is True and res["passed"] is False
