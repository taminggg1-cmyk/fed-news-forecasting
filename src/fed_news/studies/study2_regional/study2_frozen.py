"""Study 2 frozen analysis (exploratory, added after tag prereg-study2 and before any Study 2 result).

Mirrors the paper's fixed-rule exercise: (1) the event model dy2 = a + b * surprise is fitted once on events
before a cutoff (2016-01-01 or 2020-01-01) and never updated; its out-of-sample Clark-West test is reported
next to the walk-forward test on the same days; (2) the Plan 1 primary model with S + the Study 2 Fed-stance
features is fitted once before the cutoff (Plan 2 frozen.frozen_linear) and compared with frozen S and no
change. Descriptive only: not part of H6 and cannot change its verdict.
"""
import numpy as np
import pandas as pd

from fed_news import config
from fed_news.evaluation.clark_west import clark_west, compare
from fed_news.evaluation.evaluate import metrics
from fed_news.studies.plan5_event_study.event_study import MIN_TRAIN, walk_forward
from fed_news.studies.plan5_event_study.events import TEST_END, build_events
from fed_news.studies.plan2_neural_original.frozen import STARTS, frozen_linear, frozen_rw
from fed_news.studies.plan5_event_study.naive_stance import naive_panel
from fed_news.features.panel import N_COLS, S_COLS
from fed_news.models.walkforward import common_rows

OUT = config.RESULTS / "study2"
DS_PATH = config.DATA / "doc_stance2.pkl"


def frozen_event(events: pd.DataFrame, cutoff: str, outcome: str = "dy2", regressor: str = "surprise",
                 test_end: str = TEST_END) -> dict:
    ev = events.sort_index()
    train = ev[ev.index < pd.Timestamp(cutoff)]
    test = ev[(ev.index >= pd.Timestamp(cutoff)) & (ev.index <= pd.Timestamp(test_end))]
    if len(train) < MIN_TRAIN:
        raise ValueError(f"only {len(train)} training events before {cutoff}")
    s, y = train[regressor].to_numpy(float), train[outcome].to_numpy(float)
    a, b = np.linalg.lstsq(np.column_stack([np.ones(len(s)), s]), y, rcond=None)[0]
    yt, st = test[outcome].to_numpy(float), test[regressor].to_numpy(float)
    t, p = clark_west(yt, np.full(len(yt), y.mean()), a + b * st)
    wf = walk_forward(ev, outcome, regressor, cutoff, test_end)
    tw, pw = clark_west(wf["y"], wf["pred_small"], wf["pred_large"])
    return {"cutoff": cutoff, "n_train": len(train), "n_test": len(test), "frozen_slope": float(b),
            "frozen_cw_t": t, "frozen_cw_p": p, "walkforward_cw_t": tw, "walkforward_cw_p": pw}


def frozen_forecasting(ds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for task, name in (("T1", "t1"), ("T2", "t2")):
        df = common_rows(naive_panel(name, ds))
        for start in STARTS:
            base, news, rw = (frozen_linear(df, S_COLS, task, start), frozen_linear(df, S_COLS + N_COLS, task, start),
                              frozen_rw(df, start))
            for label, p in (("no change", rw), ("frozen linear S", base), ("frozen linear S+N (Study 2 stance)", news)):
                rows.append({"task": task, "start": start[:4], "model": label, **metrics(p),
                             **({k: v for k, v in compare(base, p).items() if k != "n"} if p is news else {})})
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ds = pd.read_pickle(DS_PATH)
    events = build_events(ds, pd.read_pickle(config.DATA / "market_daily.pkl")["y2"])
    ev_rows = pd.DataFrame([frozen_event(events, c) for c in STARTS])
    ev_rows.to_csv(OUT / "frozen_events.csv", index=False)
    fc = frozen_forecasting(ds)
    fc.to_csv(OUT / "frozen_forecasting.csv", index=False)
    lines = ["# Study 2 frozen analysis (exploratory, added before any Study 2 result)", "",
             "Models fitted once before the cutoff and never updated; descriptive, not part of H6.", "",
             "## Event model", "", ev_rows.round(6).to_markdown(index=False), "",
             "## Forecasting", "", fc.round(4).to_markdown(index=False), ""]
    config.report_path("RESULTS_STUDY2_FROZEN.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
