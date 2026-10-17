"""Exploratory checks added after an external review of the results (2026-10-11).

None of these is pre-registered and none changes a pre-registered verdict. Each reads stored forecasts or
cached data and writes one CSV to results/exploratory/:

- linear_vs_neural.csv: one-sided DM-HLN tests that the linear model beats each neural ensemble with the
  same inputs (the "simple beats deep" comparison; non-nested, so DM rather than Clark-West).
- text_sign_test.csv: every model-task pair with and without the text block (11 pairs, including the
  market models), with a one-sided sign test of "text lowers RMSE".
- ex_zlb.csv: the text comparisons on rows whose last observed rate is above the zero lower bound
  (lag > 0.25), where "no change" is not nearly perfect by construction.
- event_slope_per_sd.csv: the full-sample event-study slope per standard deviation of the stance surprise.
- stance_majority_baseline.csv: the majority-class accuracy of each year's stance hold-out set, rebuilt
  with the same seeded split as stance.run_stance.
- fallback_rows.csv: test rows in which at least one statistic came from the ALFRED fallback (later
  revised data) because no real-time vintage existed.
"""
import numpy as np
import pandas as pd
from scipy import stats

from fed_news import config
from fed_news.evaluation.clark_west import compare
from fed_news.evaluation.stats_tests import dm_hln

OUT = config.RESULTS / "exploratory"
ZLB = 0.25
NEURAL = {"T1": ("lstm", "gru", "tft", "hybrid"), "T2": ("mlstm", "tft", "hybrid")}
PLAN1 = ("S", "S+M")


def load(task: str, name: str) -> pd.DataFrame:
    """Stored forecasts: Plan 1 rows by info set ('S', 'S+N', ...), neural ensembles as 'arch-info'."""
    sub = "h2a/" if "-" in name else ""
    return pd.read_csv(config.RESULTS / f"{sub}preds_{task}_{name}.csv", parse_dates=["date"])


def rmse(p: pd.DataFrame) -> float:
    return float(np.sqrt(((p["y_pred"] - p["y_true"]) ** 2).mean()))


def text_pairs(task: str) -> list[tuple[str, str, str]]:
    """(label, without-text name, with-text name) for every model of a task."""
    linear = [("linear", "S", "S+N"), ("linear+market", "S+M", "S+M+N")]
    return linear + [(arch, f"{arch}-S", f"{arch}-S+N") for arch in NEURAL[task]]


def sign_test(n_better: int, n: int) -> float:
    """One-sided binomial p of at least n_better successes out of n under p = 0.5."""
    return float(stats.binomtest(n_better, n, 0.5, alternative="greater").pvalue) if n else 1.0


def above_zlb(p: pd.DataFrame, bound: float = ZLB) -> pd.DataFrame:
    return p[p["lag"] > bound].reset_index(drop=True)


def linear_vs_neural() -> pd.DataFrame:
    rows = []
    for task, archs in NEURAL.items():
        for info in ("S", "S+N"):
            lin = load(task, info)
            for arch in archs:
                net = load(task, f"{arch}-{info}")
                if not (lin["date"].to_numpy() == net["date"].to_numpy()).all():
                    raise ValueError(f"{task} {arch}-{info}: forecast dates differ")
                y = lin["y_true"].to_numpy(float)
                stat, p = dm_hln((y - net["y_pred"]) ** 2, (y - lin["y_pred"]) ** 2)
                rows.append({"task": task, "info": info, "neural": arch, "n": len(y), "rmse_linear": rmse(lin),
                             "rmse_neural": rmse(net), "dm_stat": stat, "p_linear_better": p})
    return pd.DataFrame(rows)


def text_sign_table() -> pd.DataFrame:
    rows = []
    for task in ("T1", "T2"):
        for label, base, text in text_pairs(task):
            a, b = load(task, base), load(task, text)
            rows.append({"task": task, "model": label, "rmse_without": rmse(a), "rmse_with": rmse(b),
                         "change_pct": 100 * (rmse(b) / rmse(a) - 1), "text_better": rmse(b) < rmse(a)})
    table = pd.DataFrame(rows)
    k, n = int(table["text_better"].sum()), len(table)
    summary = {"task": "all", "model": f"{k} of {n} pairs improve", "rmse_without": np.nan, "rmse_with": np.nan,
               "change_pct": np.nan, "text_better": np.nan, "sign_test_p": sign_test(k, n)}
    return pd.concat([table, pd.DataFrame([summary])], ignore_index=True)


def ex_zlb() -> pd.DataFrame:
    rows = []
    for task in ("T1", "T2"):
        rw = above_zlb(load(task, "rw"))
        rows.append({"task": task, "model": "no change", "n": len(rw), "rmse_without": rmse(rw)})
        for label, base, text in text_pairs(task):
            a, b = above_zlb(load(task, base)), above_zlb(load(task, text))
            rows.append({"task": task, "model": label, "n": len(a), "rmse_without": rmse(a), "rmse_with": rmse(b),
                         **{k: v for k, v in compare(a, b).items() if k != "n"}})
    return pd.DataFrame(rows)


def event_slopes() -> pd.DataFrame:
    import statsmodels.api as sm
    from fed_news.studies.plan5_event_study.events import TEST_END, TEST_START, build_events

    sets = {"H5 (Plan 5 documents)": pd.read_csv(config.RESULTS / "text" / "events.csv", parse_dates=["day"],
                                                 index_col="day")}
    ds2 = config.DATA / "doc_stance2.pkl"
    if ds2.exists():
        sets["H6 (with regional speeches)"] = build_events(pd.read_pickle(ds2),
                                                           pd.read_pickle(config.DATA / "market_daily.pkl")["y2"])
    rows = []
    for name, ev in sets.items():
        ev = ev[(ev.index >= pd.Timestamp(TEST_START)) & (ev.index <= pd.Timestamp(TEST_END))]
        s = ev["surprise"].to_numpy(float)
        fit = sm.OLS(ev["dy2"].to_numpy(float), sm.add_constant(s)).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
        rows.append({"event_set": name, "n": len(ev), "sd_surprise": s.std(ddof=1),
                     "slope_bp_per_unit": float(fit.params[1]), "slope_bp_per_sd": float(fit.params[1]) * s.std(ddof=1),
                     "sd_dy2_bp": float(ev["dy2"].std(ddof=1)), "t_hac": float(fit.tvalues[1])})
    return pd.DataFrame(rows)


def stance_majority() -> pd.DataFrame:
    from fed_news.features.stance import classifier_year, eligible

    labels = pd.read_pickle(config.DATA / "labels.pkl")
    acc = pd.read_csv(config.RESULTS / "stance_accuracy.csv")
    rows = []
    for cyear in sorted(acc["cyear"].astype(int)):
        pool = eligible(labels, cyear)
        hold = np.random.default_rng(0).random(len(pool)) < config.STANCE_HOLDOUT
        h, train = pool[hold], pool[~hold]
        majority = train["label"].mode().iloc[0]
        rec = acc[acc["cyear"] == cyear].iloc[0]
        if len(h) != int(rec["n_holdout"]):
            raise ValueError(f"{cyear}: rebuilt hold-out has {len(h)} rows, stance_accuracy.csv {rec['n_holdout']}")
        rows.append({"cyear": cyear, "n_holdout": len(h), "classifier_accuracy": float(rec["accuracy"]),
                     "majority_label": int(majority), "majority_accuracy": float((h["label"] == majority).mean()),
                     "top_share_in_holdout": float(h["label"].value_counts(normalize=True).iloc[0])})
    return pd.DataFrame(rows)


def fallback_rows() -> pd.DataFrame:
    rows = []
    for task, name in (("T1", "t1_full.pkl"), ("T2", "t2_full.pkl")):
        panel = pd.read_pickle(config.DATA / name)
        test = panel.loc[panel.index.isin(load(task, "S")["date"])]
        rows.append({"task": task, "test_rows": len(test), "rows_with_fallback": int((test["fallback_n"] > 0).sum()),
                     "share": float((test["fallback_n"] > 0).mean())})
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in (("linear_vs_neural", linear_vs_neural), ("text_sign_test", text_sign_table),
                     ("ex_zlb", ex_zlb), ("event_slope_per_sd", event_slopes),
                     ("stance_majority_baseline", stance_majority), ("fallback_rows", fallback_rows)):
        table = fn()
        table.to_csv(OUT / f"{name}.csv", index=False)
        print(f"\n== {name}\n{table.round(4).to_string(index=False)}")


if __name__ == "__main__":
    main()
