import numpy as np
import pandas as pd

from fed_news.studies.plan5_event_study import events as ev


def docs(rows):
    return pd.DataFrame(rows, columns=["doc_id", "type", "release", "stance", "n_sent"]).assign(
        release=lambda d: pd.to_datetime(d["release"]))


def yields(pairs):
    return pd.Series([v for _, v in pairs], index=pd.to_datetime([d for d, _ in pairs]))


Y = yields([("2005-01-03", 1.00), ("2005-01-04", 1.10), ("2005-01-05", 1.30), ("2005-01-06", 1.20),
            ("2005-01-07", 1.25), ("2005-01-10", 1.40), ("2005-02-07", 1.50), ("2005-02-08", 1.45),
            ("2005-02-09", 1.60), ("2005-02-10", 1.70)])


def test_weekend_release_joins_next_trading_day():
    d = docs([("a", "speech", "2005-01-04", 0.5, 5), ("s", "speech", "2005-01-08", 0.1, 5),   # Saturday
              ("m", "minutes", "2005-01-10", 0.3, 5)])                                       # Monday
    out = ev.build_events(d, Y)
    row = out.loc[pd.Timestamp("2005-01-10")]
    assert row["n_docs"] == 2 and np.isclose(row["stance"], 0.2) and row["category"] == "minutes"
    assert np.isclose(row["dy2"], 15.0) and np.isclose(row["dy2_before"], 5.0)               # Mon-Fri, Fri-Thu
    assert pd.Timestamp("2005-01-08") not in out.index
    assert np.isclose(row["surprise"], 0.2 - 0.5)       # a weekend document must not leak into its own baseline


def test_surprise_window_excludes_event_documents_and_old_ones():
    d = docs([("o", "speech", "2005-01-04", 0.9, 5),                                          # > 30 days before
              ("b", "speech", "2005-02-07", 0.2, 5), ("c", "minutes", "2005-02-08", 0.4, 5),
              ("e1", "statement", "2005-02-10", 0.1, 5), ("e2", "speech", "2005-02-10", 0.3, 5)])
    out = ev.build_events(d, Y)
    assert np.isclose(out.loc[pd.Timestamp("2005-02-10"), "surprise"], 0.2 - 0.3)


def test_fallback_is_mean_of_latest_earlier_event():
    d = docs([("a", "speech", "2005-01-04", 0.9, 5), ("x", "speech", "2005-01-05", 0.4, 5),
              ("y", "minutes", "2005-01-05", 0.6, 5), ("z", "speech", "2005-02-09", 0.1, 5)])
    out = ev.build_events(d, Y)
    assert np.isclose(out.loc[pd.Timestamp("2005-02-09"), "surprise"], 0.1 - 0.5)            # mean of x and y


def test_later_documents_do_not_change_earlier_surprise():
    rows = [("a", "speech", "2005-01-04", 0.5, 5), ("b", "speech", "2005-01-06", 0.1, 5)]
    base = ev.build_events(docs(rows), Y).loc[pd.Timestamp("2005-01-06"), "surprise"]
    later = ev.build_events(docs(rows + [("c", "speech", "2005-01-07", 9.0, 5)]), Y).loc[pd.Timestamp("2005-01-06"), "surprise"]
    assert base == later


def test_first_event_and_non_event_types_dropped():
    d = docs([("a", "speech", "2005-01-05", 0.5, 5), ("z", "beige", "2005-01-06", 0.9, 5),
              ("b", "speech", "2005-01-06", 0.1, 5)])
    out = ev.build_events(d, Y)
    assert list(out.index) == [pd.Timestamp("2005-01-06")] and out.iloc[0]["n_docs"] == 1


def test_day_category_priority():
    assert ev.day_category(["speech", "minutes"]) == "minutes"
    assert ev.day_category(["speech", "presconf"]) == "statement"
    assert ev.day_category(["speech"]) == "speech"
