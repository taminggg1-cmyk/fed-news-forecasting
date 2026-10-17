import numpy as np
import pandas as pd
import pytest

from fed_news.studies.poc import poc_haiku_leak as hl

NAMES = {"bernanke", "kohn"}


def test_mask_hides_years_months_numbers_names_and_events():
    s = ("in september 2008, after lehman failed, chairman bernanke said the rate of 2 percent "
         "and inflation of 3-1/2 percent were consistent with easing.")
    m = hl.mask(s, NAMES)
    for gone in ("2008", "september", "lehman", "bernanke", "3-1/2", " 2 "):
        assert gone not in m
    assert "inflation" in m and "easing" in m and "[YEAR]" in m and "[NAME]" in m and "[EVENT]" in m


def test_mask_keeps_may_and_march_as_ordinary_words_but_not_as_dates():
    assert hl.mask("the committee may decide to march ahead", NAMES) == "the committee may decide to march ahead"
    assert "[MONTH]" in hl.mask("in may the committee met", NAMES)
    assert "[MONTH]" in hl.mask("since march [NUM]", NAMES)
    assert "pandemic" not in hl.mask("pandemic-related disruptions eased", NAMES)


def test_parse_years():
    assert hl.parse_years('{"1": 2004, "2": "1999"}', 2) == [2004, 1999]
    with pytest.raises(ValueError):
        hl.parse_years('{"1": 2004}', 2)


def test_next_decision_is_the_first_meeting_after_release():
    meetings = pd.DataFrame({"y_change": [0.25, 0.0, -0.5]},
                            index=pd.to_datetime(["2005-02-02", "2005-03-22", "2005-05-03"]))
    rel = pd.to_datetime(["2005-02-02", "2005-01-10", "2005-04-01"])
    assert hl.next_decision(rel, meetings).tolist() == [0, 1, -1]


def _frame(n=1500, seed=0, leak=True):
    rng = np.random.default_rng(seed)
    meeting = rng.integers(0, 150, n)
    nxt = np.sign(rng.normal(size=150))[meeting]
    human = np.sign(nxt + rng.normal(0, 1.5, n))
    chrono = human * 0.5 + rng.normal(0, 1, n)
    haiku = np.sign(nxt + rng.normal(0, 0.5, n)) if leak else np.sign(human + rng.normal(0, 0.5, n))
    return pd.DataFrame({"next": nxt, "human": human, "haiku": haiku, "chrono": chrono, "meeting": meeting})


def test_hindsight_flag_fires_when_haiku_tracks_the_next_decision():
    assert hl.hindsight(_frame(leak=True))["flag"] is True


def test_hindsight_flag_quiet_when_haiku_only_reads_the_text():
    assert hl.hindsight(_frame(leak=False))["flag"] is False
