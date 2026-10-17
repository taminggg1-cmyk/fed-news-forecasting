"""Study 5: does Fed communication help forecast the policy path three months ahead? (exploratory)
(docs/superpowers/specs/2026-10-12-study5-three-month-horizon-design.md).

One row per scheduled FOMC meeting, forecast at its cutoff (the day before the decision). Target: the net
change in the federal funds target from before meeting t to after scheduled meeting t+h-1 (unscheduled moves
in between included). Real-time inputs only. A row trains a forecast only once its whole horizon has been
decided before that forecast's cutoff; the penalty CV is purged of overlapping horizons. Primary: h = 3,
no news (B) against B + ChronoBERT stance, mean log-loss difference with a Newey-West test (8 lags).
"""
import json

import numpy as np
import pandas as pd
from scipy import stats

from fed_news import alfred, config, models
from fed_news.evaluate import decide, metrics
from fed_news.haiku_compare import brier, log_loss
from fed_news.haiku_index import OUT_PATH as HAIKU_PATH
from fed_news.panel import M_COLS
from fed_news.stats_tests import dm_hln, holm
from fed_news.study4_sensitivity import block_bootstrap, class_counts
from fed_news.text_features import fed_stance
from fed_news.walkforward import _standardise, placebo_matrix

OUT = config.RESULTS / "study5"
REALTIME_S = ["s_rate_lag", "s_m2_mom", "s_m2_yoy", "s_payems_chg", "s_gdp_yoy", "s_cpi_yoy",
              "s_slope_10y2y", "s_slope_1y6m"]
KEPT_SERIES = ("m2", "payems", "gdp", "cpi")
B = REALTIME_S + M_COLS
NW_LAGS, THRESHOLD, LAST_DECISION = 8, 0.02, "2026-09-16"


def horizon_targets(meetings: pd.DataFrame, h: int) -> pd.DataFrame:
    m = meetings.sort_index()
    sched = m[m["unscheduled"] == 0]
    rows = []
    for k, d in enumerate(sched.index):
        if k + h - 1 < len(sched):
            end = sched.index[k + h - 1]
            change = float(m.loc[end, "target_after"] - m.loc[d, "target_before"])
        else:
            end, change = pd.NaT, np.nan
        rows.append({"cutoff": sched.loc[d, "cutoff"], "end_date": end, "change": change})
    out = pd.DataFrame(rows, index=sched.index)
    out["days"] = (out["end_date"] - out["cutoff"]).dt.days
    out["y_class"] = np.where(out["change"].isna(), np.nan, models.classes_from_change(out["change"].fillna(0)))
    return out


def train_rows(ends, cutoffs, i: int) -> np.ndarray:
    ends, cutoffs = pd.DatetimeIndex(ends), pd.DatetimeIndex(cutoffs)
    return np.asarray((ends < cutoffs[i]) & (np.arange(len(ends)) < i))


def purged_folds(n: int, k: int = config.CV_FOLDS, gap: int = 0):
    out = []
    for tr, va in models.expanding_folds(n, k):
        tr = tr[tr < va.min() - gap]
        if len(tr):
            out.append((tr, va))
    return out


def choose_penalty_purged(X, y, grid, gap: int) -> float:
    best, best_score = grid[0], np.inf
    for value in grid:
        s = np.mean([models.logloss_score(models.OrderedLogit(value).fit(X[tr], y[tr]), X[va], y[va])
                     for tr, va in purged_folds(len(y), gap=gap)])
        if not np.isfinite(s):
            raise ValueError(f"non-finite CV score for penalty {value}")
        if s < best_score:
            best, best_score = value, s
    return best


def stance_features(docs: pd.DataFrame, rows: pd.DataFrame, first_start: pd.Timestamp) -> pd.DataFrame:
    starts = [first_start] + list(rows["cutoff"].iloc[:-1])
    st = pd.Series([fed_stance(docs, s, c) for s, c in zip(starts, rows["cutoff"])], index=rows.index).ffill()
    return pd.DataFrame({"n_stance": st, "n_stance_chg": st.diff()})


def nw_summary(d, lags: int = NW_LAGS) -> dict:
    import statsmodels.api as sm
    d = np.asarray(d, float)
    fit = sm.OLS(d, np.ones(len(d))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    mean, se = float(fit.params[0]), float(fit.bse[0])
    return {"n": int(len(d)), "mean": mean, "se": se, "p_one_sided": float(1 - stats.norm.cdf(mean / se)),
            "ci90": [mean - 1.645 * se, mean + 1.645 * se]}


def effect_statement(ci, supported: bool, threshold: float = THRESHOLD) -> str:
    if ci[1] < threshold:
        return f"gains of {threshold} or more are ruled out by the stated threshold"
    if not supported:
        return "inconclusive"
    if ci[0] >= threshold:
        return f"the gain is at least the threshold ({threshold})"
    return f"significant, but the gain may be smaller than the threshold ({threshold})"


def check_vintages(first: dict, kept, first_cutoff: pd.Timestamp) -> None:
    late = [k for k in kept if first[k] > first_cutoff - pd.Timedelta(days=1)]
    if late:
        raise ValueError(f"series would use the ALFRED fallback: {late}")


def walk_forward(df: pd.DataFrame, cols, h: int, test_start, placebo_cols=(), shuffle: bool = False,
                 seed: int = 0, on_fit=None) -> pd.DataFrame:
    X = df[cols].to_numpy(float)
    ycl, yc = df["y_class"].to_numpy(int), df["y_change"].to_numpy(float)
    lag, ends, cutoffs, dates = df["s_rate_lag"].to_numpy(float), df["end_date"], df["cutoff"], df.index
    ncols = [cols.index(c) for c in placebo_cols]
    rng = np.random.default_rng(seed)
    out = []
    for i in np.flatnonzero(dates >= pd.Timestamp(test_start)):
        train = train_rows(ends, cutoffs, i)
        rows = np.flatnonzero(train)
        if on_fit:
            on_fit(i, rows)
        Xi = placebo_matrix(X, train, i, ncols, rng) if ncols else X
        y_cl, y_ch = ycl, yc
        if shuffle:
            perm = rng.permutation(rows)
            y_cl, y_ch = ycl.copy(), yc.copy()
            y_cl[rows], y_ch[rows] = ycl[perm], yc[perm]
        Z = _standardise(Xi, train)
        lam = choose_penalty_purged(Z[rows], y_cl[rows], config.ORDLOGIT_GRID, gap=h - 1)
        p = models.OrderedLogit(lam).fit(Z[rows], y_cl[rows]).predict_proba(Z[i:i + 1])[0]
        change = float(p @ models.move_sizes(y_cl[rows], y_ch[rows]))
        out.append({"date": dates[i], "y_true": lag[i] + yc[i], "y_pred": lag[i] + change, "lag": lag[i],
                    "class_true": int(ycl[i]), "class_pred": int(p.argmax()),
                    "p_cut": float(p[0]), "p_hold": float(p[1]), "p_hike": float(p[2])})
    return pd.DataFrame(out)


def build_panel(h: int, readers: dict) -> pd.DataFrame:
    m = pd.read_pickle(config.DATA / "meetings.pkl").sort_index()
    t = horizon_targets(m, h)
    full = pd.read_pickle(config.DATA / "t2_full.pkl")
    rows = full.loc[t.index, B + ["n_beige"]].assign(cutoff=t["cutoff"])
    before = m.index[m.index < t.index[0]]
    first_start = m.loc[before[-1], "cutoff"] if len(before) else t["cutoff"].iloc[0] - pd.Timedelta(days=60)
    for tag, docs in readers.items():
        f = stance_features(docs, rows, first_start)
        rows[f"n_stance_{tag}"], rows[f"n_stance_chg_{tag}"] = f["n_stance"], f["n_stance_chg"]
    rows = rows.assign(y_change=t["change"], y_class=t["y_class"], end_date=t["end_date"], days=t["days"])
    return rows.dropna()


def stance(tag: str) -> list[str]:
    return [f"n_stance_{tag}", f"n_stance_chg_{tag}"]


SPECS = {  # name: (columns, placebo columns, shuffle)
    "B": (B, (), False),
    "B+C": (B + stance("C"), (), False),
    "placebo-C": (B + stance("C"), tuple(stance("C")), False),
    "shuffled-C": (B + stance("C"), (), True),
    "B+H": (B + stance("H"), (), False),
    "B+C+Beige": (B + stance("C") + ["n_beige"], (), False),
}
RUNS = {3: list(SPECS), 2: ["B", "B+C", "B+H"], 1: ["B", "B+C"]}


def run() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    obs = alfred.load_all()
    first = {k: v["realtime_start"].min() for k, v in obs.items()}
    readers = {"C": pd.read_pickle(config.DATA / "doc_stance.pkl"), "H": pd.read_pickle(HAIKU_PATH)}
    preds, info = {}, {}
    for h, names in RUNS.items():
        df = build_panel(h, readers)
        check_vintages(first, KEPT_SERIES, df["cutoff"].iloc[0])
        test = df[df.index >= pd.Timestamp(config.TEST_START)]
        info[h] = {"n_rows": int(len(df)), "n_test": int(len(test)), "first_test": str(test.index[0].date()),
                   "last_test": str(test.index[-1].date()),
                   "classes": {int(k): int(v) for k, v in test["y_class"].value_counts().sort_index().items()},
                   "days_min_median_max": [int(test["days"].min()), float(test["days"].median()),
                                           int(test["days"].max())]}
        for name in names:
            cols, pcols, shuffle = SPECS[name]
            path = OUT / f"preds_h{h}_{name}.csv"
            if not path.exists():
                walk_forward(df, cols, h, config.TEST_START, pcols, shuffle).to_csv(path, index=False)
                print(f"done h={h} {name}", flush=True)
            preds[(h, name)] = pd.read_csv(path, parse_dates=["date"])

    ll = {k: log_loss(v) for k, v in preds.items()}
    gain = lambda h, name: ll[(h, "B")] - ll[(h, name)]
    rows = [{"h": h, "model": name, **metrics(preds[(h, name)]), "log_loss": float(ll[(h, name)].mean()),
             "brier": float(brier(preds[(h, name)]).mean()),
             **({} if name == "B" else {f"nw_{k}": v for k, v in nw_summary(gain(h, name)).items()})}
            for h, names in RUNS.items() for name in names]
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "comparison.csv", index=False)

    primary = nw_summary(gain(3, "B+C"))
    p_placebo, p_shuffled = nw_summary(gain(3, "placebo-C"))["p_one_sided"], nw_summary(gain(3, "shuffled-C"))["p_one_sided"]
    decision = decide([primary["p_one_sided"]], [p_placebo], [p_shuffled])
    supported = decision == "supported"
    wording = ("public Fed communication, read by a leak-free model, improves three-month forecasts of the policy "
               "path over a model with real-time economic data and Treasury yields (exploratory); this is not a "
               "win over market expectations measured by fed funds futures" if supported else
               "no detectable improvement at three months in this data, with this reader and horizon"
               if decision == "not supported" else "a control beat the baseline, so the comparison is void")
    secondary = {"B+H h3": nw_summary(gain(3, "B+H"))["p_one_sided"],
                 "B+C+Beige h3": nw_summary(gain(3, "B+C+Beige"))["p_one_sided"],
                 "B+C h2": nw_summary(gain(2, "B+C"))["p_one_sided"],
                 "B+H h2": nw_summary(gain(2, "B+H"))["p_one_sided"]}
    res = {"panel": info, "primary": primary, "placebo_p": p_placebo, "shuffled_p": p_shuffled,
           "decision": decision, "wording": wording,
           "effect": effect_statement(primary["ci90"], supported) if decision != "void" else None,
           "robustness": {"hln_dm_h3_p": dm_hln(ll[(3, "B")], ll[(3, "B+C")], h=3)[1],
                          "block_bootstrap_8": block_bootstrap(gain(3, "B+C"), 8, 10_000, 0)},
           "secondary_p": secondary, "secondary_holm": holm(secondary),
           "h1_reference": nw_summary(gain(1, "B+C"))}
    class_counts(preds[(3, "B")], preds[(3, "B+C")]).to_csv(OUT / "class_counts_h3.csv", index=False)
    (OUT / "verdict.json").write_text(json.dumps(res, indent=2))
    pd.set_option("display.width", 220)
    print(table[["h", "model", "n", "accuracy", "log_loss", "brier", "nw_mean", "nw_p_one_sided"]].round(4)
          .to_string(index=False))
    print(json.dumps(res, indent=2), flush=True)
    return res


if __name__ == "__main__":
    run()
