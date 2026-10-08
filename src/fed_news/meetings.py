"""One row per FOMC decision from 1994 (spec section 4, T2).

Decision dates are the union of minutes-link and statement dates (dates within two days are
merged to the later one), plus any target change more than three days after every such date.
The decision is the target in force up to four days after the meeting (or the day before the
next meeting) minus the target in force before it. Cutoff = the day before the decision date.
"""
import numpy as np
import pandas as pd

from fed_news import config, market
from fed_news.fomc_calendar import load_calendar


def _decision_dates(cal: pd.DataFrame, target: pd.Series) -> list[pd.Timestamp]:
    dates = sorted(set(cal.loc[cal["kind"].isin(["minutes", "statement"]), "date"]))
    merged: list[pd.Timestamp] = []
    for day in dates:
        if merged and (day - merged[-1]).days <= 2:
            merged[-1] = day
        else:
            merged.append(day)
    steps = target.diff()
    for day in steps.index[steps.abs() > 1e-9]:
        if not any(0 <= (day - m).days <= 3 for m in merged):
            merged.append(day)
    return sorted(merged)


def build_meetings(cal: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
    minutes_dates = set(cal.loc[cal["kind"] == "minutes", "date"])
    dates = [d for d in _decision_dates(cal, target)
             if d >= pd.Timestamp(config.SAMPLE_START) and d + pd.Timedelta(days=4) <= target.index.max()]
    rows = []
    for i, day in enumerate(dates):
        nxt = dates[i + 1] if i + 1 < len(dates) else day + pd.Timedelta(days=60)
        horizon = min(day + pd.Timedelta(days=5), nxt)
        before = market.asof(target, day)
        after = market.asof(target, horizon)
        change = after - before
        rows.append({
            "date": day, "cutoff": day - pd.Timedelta(days=1),
            "target_before": before, "target_after": after, "y_change": change,
            "y_class": 0 if change < -config.CLASS_EPS else (2 if change > config.CLASS_EPS else 1),
            "unscheduled": int(not any(abs((day - m).days) <= 2 for m in minutes_dates)),
        })
    return pd.DataFrame(rows).set_index("date")


def load_meetings(refresh: bool = False) -> pd.DataFrame:
    path = config.DATA / "meetings.pkl"
    if path.exists() and not refresh:
        return pd.read_pickle(path)
    frame = build_meetings(load_calendar(), market.load_daily()["target"])
    frame.to_pickle(path)
    return frame


if __name__ == "__main__":
    m = load_meetings(refresh=True)
    print(f"meetings {len(m)}  {m.index.min().date()} .. {m.index.max().date()}")
    print(m["y_class"].value_counts().sort_index().to_string())
    print("unscheduled:", list(m.index[m["unscheduled"] == 1].date))
