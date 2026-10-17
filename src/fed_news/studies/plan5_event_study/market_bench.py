"""Plan 5 market benchmark (descriptive): Treasury-bill forecasts of the policy rate.

forecast(row) = bill yield at the last date strictly before the row's cutoff + spread, where spread is the
mean of (y_level - bill) over the row's training rows (walkforward.train_mask) that have a bill value; at
least 12 such rows are required. Bills: bill3m = market_daily.pkl tbill_3m (FRED DTB3) and bill4w = FRED
DTB4WK (from 2001-07-31). Rows without a bill value get no forecast.
"""
import json

import numpy as np
import pandas as pd
import requests

from fed_news import config, http
from fed_news.models import linear as models
from fed_news.collect.alfred import api_key
from fed_news.evaluation.clark_west import compare
from fed_news.evaluation.evaluate import metrics
from fed_news.collect.market import asof
from fed_news.studies.plan1_linear.run_models import load_task
from fed_news.models.walkforward import train_mask

FRED_OBS = "https://api.stlouisfed.org/fred/series/observations?series_id={}&api_key={}&file_type=json"
BILLS = ("bill3m", "bill4w")
MIN_SPREAD_ROWS = 12
COLUMNS = ["date", "y_true", "y_pred", "lag", "class_true", "class_pred"]
OUT = config.RESULTS / "text"


def fetch_series(sid: str) -> pd.Series:
    url = FRED_OBS.format(sid, api_key())
    try:
        raw = http.get(url, cache=False)
    except (requests.RequestException, FileNotFoundError):
        raise RuntimeError(f"FRED request failed for {sid}") from None  # never echo the URL: it holds the key
    payload = json.loads(raw)
    obs = pd.DataFrame(payload["observations"])
    return pd.Series(pd.to_numeric(obs["value"], errors="coerce").to_numpy(),
                     index=pd.to_datetime(obs["date"]), name=sid).dropna()


def load_bills(refresh: bool = False) -> pd.DataFrame:
    path = config.DATA / "market_bill.pkl"
    if path.exists() and not refresh:
        return pd.read_pickle(path)
    daily = pd.read_pickle(config.DATA / "market_daily.pkl")
    bills = daily[["tbill_3m"]].rename(columns={"tbill_3m": "bill3m"}).join(
        fetch_series("DTB4WK").rename("bill4w"), how="outer")
    bills.to_pickle(path)
    return bills


def bill_forecast(df: pd.DataFrame, bill: pd.Series, task: str, test_start: str) -> pd.DataFrame:
    b = np.array([asof(bill, c) for c in df["cutoff"]])
    y, lag = df["y_level"].to_numpy(float), df["s_rate_lag"].to_numpy(float)
    gap = y - b
    rows = []
    for i in np.flatnonzero(df.index >= pd.Timestamp(test_start)):
        tr = train_mask(len(df), i, task) & ~np.isnan(gap)
        if np.isnan(b[i]) or tr.sum() < MIN_SPREAD_ROWS:
            continue
        pred = float(b[i] + gap[tr].mean())
        rows.append({"date": df.index[i], "y_true": y[i], "y_pred": pred, "lag": lag[i],
                     "class_true": int(df["y_class"].iloc[i]),
                     "class_pred": int(models.classes_from_change(np.array([pred - lag[i]]))[0])})
    return pd.DataFrame(rows, columns=COLUMNS)


def _on(dates, frame):
    return frame[frame["date"].isin(dates)].sort_values("date").reset_index(drop=True)


def main() -> pd.DataFrame:
    OUT.mkdir(parents=True, exist_ok=True)
    bills = load_bills()
    load = lambda task, name: pd.read_csv(config.RESULTS / f"preds_{task}_{name}.csv", parse_dates=["date"])
    rows = []
    for task in ("T1", "T2"):
        df = load_task(task)
        preds = {name: load(task, name) for name in ("rw", "S", "S+N", "S+M")}
        for bill in BILLS:
            p = bill_forecast(df, bills[bill], task, config.TEST_START)
            p.to_csv(OUT / f"preds_{task}_{bill}.csv", index=False)
            preds[bill] = p
        windows = {"full": None, "paper": ("2016-01-01", "2025-04-01")} if task == "T1" else {"full": None}
        for wname, w in windows.items():
            for bill in BILLS:
                dates = set(preds[bill]["date"])
                if w:
                    dates = {d for d in dates if pd.Timestamp(w[0]) <= d <= pd.Timestamp(w[1])}
                for name, p in preds.items():
                    if name in BILLS and name != bill:
                        continue
                    pw, rw, bw = _on(dates, p), _on(dates, preds["rw"]), _on(dates, preds[bill])
                    rows.append({"task": task, "window": wname, "bill_dates": bill, "model": name, **metrics(pw),
                                 "p_vs_no_change": compare(rw, pw)["dm_hln_p"] if name != "rw" else np.nan,
                                 "p_vs_bill": compare(bw, pw)["dm_hln_p"] if name != bill else np.nan})
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "market_benchmark.csv", index=False)
    print(table[["task", "window", "bill_dates", "model", "n", "rmse", "mae", "accuracy", "p_vs_no_change",
                 "p_vs_bill"]].round(4).to_string(index=False))
    return table


if __name__ == "__main__":
    main()
