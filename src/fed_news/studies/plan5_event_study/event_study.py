"""Plan 5 event study (H5): does the Fed stance surprise predict the same-day 2-year yield move, out of sample?

Primary test (PREREG_TEXT.md): walk-forward OLS on all strictly earlier event days; small model dy2 = a,
large model dy2 = a + b * surprise; one-sided Clark-West, p < 0.05. Void if the same test on the day-before
yield change gives p < 0.05. Secondary: full-sample OLS with Newey-West errors (5 lags), Clark-West by day
category, out-of-sample sign accuracy.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from fed_news import config
from fed_news.clark_west import clark_west
from fed_news.events import TEST_END, TEST_START, load_events

ALPHA = 0.05
MIN_TRAIN = 50
OUT = config.RESULTS / "text"


def walk_forward(events: pd.DataFrame, outcome: str, regressor: str = "surprise",
                 test_start: str = TEST_START, test_end: str = TEST_END) -> pd.DataFrame:
    ev = events.sort_index()
    days = ev.index
    y, s = ev[outcome].to_numpy(float), ev[regressor].to_numpy(float)
    rows = []
    for i in np.flatnonzero((days >= pd.Timestamp(test_start)) & (days <= pd.Timestamp(test_end))):
        tr = np.flatnonzero(days < days[i])
        if len(tr) < MIN_TRAIN:
            raise ValueError(f"only {len(tr)} training events before {days[i].date()}")
        a, b = np.linalg.lstsq(np.column_stack([np.ones(len(tr)), s[tr]]), y[tr], rcond=None)[0]
        rows.append({"day": days[i], "y": y[i], "pred_small": float(y[tr].mean()),
                     "pred_large": float(a + b * s[i]), "category": ev["category"].iloc[i]})
    return pd.DataFrame(rows).set_index("day")


def decide(primary_p: float, void_p: float, alpha: float = ALPHA) -> str:
    if np.isnan(primary_p) or np.isnan(void_p):
        raise ValueError("NaN p-value in the H5 decision")
    if void_p < alpha:
        return "void"
    return "supported" if primary_p < alpha else "not supported"


def full_sample_ols(events: pd.DataFrame, regressor: str = "surprise", test_start: str = TEST_START,
                    test_end: str = TEST_END) -> dict:
    ev = events[(events.index >= pd.Timestamp(test_start)) & (events.index <= pd.Timestamp(test_end))]
    fit = sm.OLS(ev["dy2"].to_numpy(float), sm.add_constant(ev[regressor].to_numpy(float))).fit(
        cov_type="HAC", cov_kwds={"maxlags": 5})
    t = float(fit.tvalues[1])
    return {"ols_n": len(ev), "ols_slope": float(fit.params[1]), "ols_t_hac": t,
            "ols_p_one_sided": float(1 - stats.norm.cdf(t))}


def sign_accuracy(wf: pd.DataFrame) -> float:
    nz = wf[wf["y"] != 0]
    return float((np.sign(nz["pred_large"]) == np.sign(nz["y"])).mean())


def run(events: pd.DataFrame, regressor: str = "surprise", label: str = "chronobert",
        test_start: str = TEST_START, test_end: str = TEST_END) -> tuple[dict, pd.DataFrame]:
    wf = walk_forward(events, "dy2", regressor, test_start, test_end)
    wf_void = walk_forward(events, "dy2_before", regressor, test_start, test_end)
    t, p = clark_west(wf["y"], wf["pred_small"], wf["pred_large"])
    tv, pv = clark_west(wf_void["y"], wf_void["pred_small"], wf_void["pred_large"])
    summary = {"pipeline": label, "n_test": len(wf), "cw_t": t, "cw_p": p, "void_cw_t": tv, "void_cw_p": pv,
               "verdict": decide(p, pv), "sign_accuracy": sign_accuracy(wf)}
    for cat, g in wf.groupby("category"):
        summary[f"cw_p_{cat}"] = clark_west(g["y"], g["pred_small"], g["pred_large"])[1]
        summary[f"n_{cat}"] = len(g)
    summary.update(full_sample_ols(events, regressor, test_start, test_end))
    return summary, wf


def main() -> pd.DataFrame:
    OUT.mkdir(parents=True, exist_ok=True)
    events = load_events()
    events.to_csv(OUT / "events.csv")
    summary, wf = run(events)
    wf.to_csv(OUT / "event_forecasts.csv")
    table = pd.DataFrame([summary])
    table.to_csv(OUT / "event_study.csv", index=False)
    print(table.T.to_string())
    return table


if __name__ == "__main__":
    main()
