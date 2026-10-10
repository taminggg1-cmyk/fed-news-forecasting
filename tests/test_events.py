import numpy as np
import pandas as pd

from fed_news import events as ev


def docs(rows):
    return pd.DataFrame(rows, columns=["doc_id", "type", "release", "stance", "n_sent"]).assign(
        release=lambda d: pd.to_datetime(d["release"]))


def test_surprise_uses_30_day_window_before_the_day():
    d = ev._docs(docs([("a", "speech", "2005-01-01", 0.9, 5),      # 40 days before: outside window
                       ("b", "speech", "2005-01-20", 0.2, 5),
                       ("c", "minutes", "2005-02-01", 0.4, 5),
                       ("d", "statement", "2005-02-10", 0.1, 5),   # event day
                       ("e", "speech", "2005-02-10", 0.3, 5)]))    # same day: excluded from the baseline
    assert np.isclose(ev.surprise_at(d, pd.Timestamp("2005-02-10")), 0.2 - 0.3)


def test_surprise_falls_back_to_latest_earlier_document():
    d = ev._docs(docs([("a", "speech", "2005-01-01", 0.5, 5), ("b", "speech", "2005-03-15", 0.1, 5)]))
    assert np.isclose(ev.surprise_at(d, pd.Timestamp("2005-03-15")), 0.1 - 0.5)


def test_surprise_ignores_later_documents():
    rows = [("a", "speech", "2005-01-10", 0.5, 5), ("b", "speech", "2005-01-20", 0.1, 5)]
    base = ev.surprise_at(ev._docs(docs(rows)), pd.Timestamp("2005-01-20"))
    later = ev.surprise_at(ev._docs(docs(rows + [("c", "speech", "2005-01-25", 9.0, 5)])), pd.Timestamp("2005-01-20"))
    assert base == later


def test_yield_change_weekend_and_day_before():
    y = pd.Series([1.00, 1.05, 1.20], index=pd.to_datetime(["2005-01-06", "2005-01-07", "2005-01-10"]))
    assert np.isclose(ev.yield_change(y, pd.Timestamp("2005-01-08")), 15.0)   # Saturday: Monday - Friday
    assert np.isclose(ev.yield_change(y, pd.Timestamp("2005-01-07")), 5.0)
    assert np.isclose(ev.yield_change_before(y, pd.Timestamp("2005-01-10")), 5.0)


def test_day_category_priority():
    assert ev.day_category(["speech", "minutes"]) == "minutes"
    assert ev.day_category(["speech", "presconf"]) == "statement"
    assert ev.day_category(["speech"]) == "speech"


def test_build_events_one_row_per_day_and_drops_nan():
    d = docs([("a", "speech", "2005-01-03", 0.5, 5), ("b", "speech", "2005-01-04", 0.1, 5),
              ("c", "minutes", "2005-01-04", 0.3, 5), ("z", "beige", "2005-01-04", 0.9, 5)])
    y = pd.Series([1.0, 1.1, 1.3, 1.2], index=pd.to_datetime(["2004-12-31", "2005-01-03", "2005-01-04", "2005-01-05"]))
    out = ev.build_events(d, y)
    assert list(out.index) == [pd.Timestamp("2005-01-04")]              # first day has no surprise
    row = out.iloc[0]
    assert np.isclose(row["stance"], 0.2) and row["n_docs"] == 2 and row["category"] == "minutes"
    assert np.isclose(row["dy2"], 20.0) and np.isclose(row["dy2_before"], 10.0)
