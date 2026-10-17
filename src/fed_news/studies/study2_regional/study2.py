"""Study 2 (H6, PREREG_STUDY2.md): the Plan 5 event study on the expanded document table, plus secondary
analyses (regional speeches only; Plan 5 documents with the new classifier; forecasting with rebuilt
Fed-stance features). Only the H6 row decides; secondary rows are reported, not decisive."""
import pandas as pd

from fed_news import config
from fed_news.clark_west import compare
from fed_news.evaluate import metrics
from fed_news.event_study import run as run_event_study
from fed_news.events import build_events
from fed_news.naive_stance import naive_panel
from fed_news.panel import N_COLS, S_COLS
from fed_news.walkforward import Spec, common_rows, run_spec

OUT = config.RESULTS / "study2"
DS_PATH = config.DATA / "doc_stance2.pkl"


def event_rows(ds: pd.DataFrame, y2: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    sets = [("H6: all documents", "confirmatory", ds),
            ("secondary: regional speeches only", "secondary", ds[ds["source"] != "corpus"]),
            ("secondary: Plan 5 documents, new classifier", "secondary", ds[ds["source"] == "corpus"])]
    rows, h6_wf = [], None
    for label, role, sub in sets:
        if role == "confirmatory":
            summary, h6_wf = run_event_study(build_events(sub, y2), label=label)
        else:
            try:  # a failing secondary set never costs the confirmatory H6 result
                summary, _ = run_event_study(build_events(sub, y2), label=label)
                summary.pop("verdict", None)
            except (ValueError, KeyError) as err:
                summary = {"error": f"failed: {err}"}
        rows.append({"pipeline": label, "role": role, **{k: v for k, v in summary.items() if k != "pipeline"}})
    return pd.DataFrame(rows), h6_wf


def forecasting(ds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for task, name in (("T1", "t1"), ("T2", "t2")):
        p = run_spec(common_rows(naive_panel(name, ds)), Spec("S+N study2", task, "primary", S_COLS + N_COLS),
                     config.TEST_START)
        p.to_csv(OUT / f"preds_{task}_S+N_study2.csv", index=False)
        base = pd.read_csv(config.RESULTS / f"preds_{task}_S.csv", parse_dates=["date"])
        dates = set(base["date"]) & set(p["date"])
        on = lambda f: f[f["date"].isin(dates)].sort_values("date").reset_index(drop=True)
        rmse_s, m = metrics(on(base))["rmse"], metrics(on(p))
        rows.append({"task": task, "n": m["n"], "rmse": m["rmse"], "rmse_S": rmse_s,
                     "change_vs_S_pct": 100 * (m["rmse"] / rmse_s - 1), **compare(on(base), on(p))})
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ds = pd.read_pickle(DS_PATH)
    y2 = pd.read_pickle(config.DATA / "market_daily.pkl")["y2"]
    rows, wf = event_rows(ds, y2)
    rows[rows["role"] == "confirmatory"].dropna(axis=1, how="all").to_csv(OUT / "h6.csv", index=False)
    wf.to_csv(OUT / "h6_forecasts.csv")
    rows[rows["role"] != "confirmatory"].dropna(axis=1, how="all").to_csv(OUT / "secondary_events.csv", index=False)
    forecasting(ds).to_csv(OUT / "forecasting.csv", index=False)
    print(rows.T.to_string())


if __name__ == "__main__":
    main()
