"""Proof of concept (exploratory), stages 2 and 3: does what changed in FOMC statements forecast the next decision
(docs/superpowers/specs/2026-10-12-poc-statement-change-design.md, sections 3-6)?

Scheduled meetings only, cutoff seven days before the decision. Multinomial logistic regression with the scaler
inside a Pipeline, C chosen by inner expanding-fold log loss on the training rows; refit at every meeting.
Stage 2 runs only after stage 1 (stmt_change) has passed.
"""
import argparse
import json

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from fed_news.collect import alfred, market
from fed_news import config
from fed_news.models import linear as models
from fed_news.collect.meetings import load_meetings
from fed_news.features.panel import M_COLS, S_COLS, _row
from fed_news.studies.poc.stmt_change import OUT, WORK, meeting_features
from fed_news.models.walkforward import placebo_matrix

LEAD_DAYS = 7
GRID = tuple(float(c) for c in np.logspace(-3, 2, 9))
C_FALLBACK = 1.0
MIN_FOLD_ROWS = 20
EPS = 1e-6
PCOLS = ["p_cut", "p_hold", "p_hike"]
X_COLS = ["x_shift", "x_dissent_n", "x_dissent_dir"]
P_COLS = ["p_prev_change"]
STAGE2 = ("2001-01-01", "2015-12-31")
STAGE3 = ("2016-01-01", "2026-09-16")


def lead_meetings(meet: pd.DataFrame, days: int = LEAD_DAYS) -> pd.DataFrame:
    sched = meet[meet["unscheduled"] == 0].copy()
    sched["cutoff"] = sched.index - pd.Timedelta(days=days)
    sched["p_prev_change"] = sched["target_before"] - sched["target_before"].shift(1)
    return sched


def proba3(model, X) -> np.ndarray:
    p = np.zeros((len(X), 3))
    p[:, np.asarray(model.classes_, dtype=int)] = model.predict_proba(X)
    p = np.clip(p, EPS, None)
    return p / p.sum(1, keepdims=True)


def _pipe(C: float):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000))


def inner_folds(n: int) -> list:
    return [(tr, va) for tr, va in models.expanding_folds(n) if len(tr) >= MIN_FOLD_ROWS]


def log_loss_rows(p: np.ndarray, y) -> np.ndarray:
    return -np.log(p[np.arange(len(p)), np.asarray(y, dtype=int)])


def brier_rows(p: np.ndarray, y) -> np.ndarray:
    onehot = np.eye(3)[np.asarray(y, dtype=int)]
    return ((p - onehot) ** 2).sum(1)


def cv_logloss(X, y, C: float) -> float:
    scores = [log_loss_rows(proba3(_pipe(C).fit(X[tr], y[tr]), X[va]), y[va]).mean()
              for tr, va in inner_folds(len(y)) if len(np.unique(y[tr])) > 1]
    return float(np.mean(scores)) if scores else np.nan


def choose_C(X, y) -> float:
    scores = [cv_logloss(X, y, C) for C in GRID]
    return C_FALLBACK if np.all(np.isnan(scores)) else GRID[int(np.nanargmin(scores))]


def walk_forward(df: pd.DataFrame, cols: list, start, end, placebo: bool = False, seed: int = 0) -> pd.DataFrame:
    """Forecast each meeting in [start, end] from all earlier meetings only."""
    X0, y = df[cols].to_numpy(float), df["y_class"].to_numpy(int)
    ncols = [cols.index(c) for c in X_COLS if c in cols]
    rng = np.random.default_rng(seed)
    test = np.flatnonzero((df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end)))
    out = []
    for i in test:
        train = np.arange(len(df)) < i
        X = placebo_matrix(X0, train, i, ncols, rng) if placebo else X0
        C = choose_C(X[train], y[train])
        p = proba3(_pipe(C).fit(X[train], y[train]), X[i:i + 1])[0]
        out.append({"date": df.index[i], "class_true": int(y[i]), "y_change": float(df["y_change"].iloc[i]),
                    **dict(zip(PCOLS, p)), "C": C})
    return pd.DataFrame(out)


def climatology(df: pd.DataFrame, start, end) -> pd.DataFrame:
    y = df["y_class"].to_numpy(int)
    test = np.flatnonzero((df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end)))
    out = []
    for i in test:
        p = np.clip(np.bincount(y[:i], minlength=3) / i, EPS, None)
        out.append({"date": df.index[i], "class_true": int(y[i]), "y_change": float(df["y_change"].iloc[i]),
                    **dict(zip(PCOLS, p / p.sum()))})
    return pd.DataFrame(out)


def always_hold(ref: pd.DataFrame) -> pd.DataFrame:
    p = np.array([EPS, 1.0, EPS]) / (1 + 2 * EPS)
    return ref[["date", "class_true", "y_change"]].assign(**dict(zip(PCOLS, p)))


def ece(p: np.ndarray, y, bins: int = 5) -> float:
    conf, hit = p.max(1), (p.argmax(1) == np.asarray(y)).astype(float)
    idx = np.minimum((conf * bins).astype(int), bins - 1)
    return float(sum(abs(hit[idx == b].mean() - conf[idx == b].mean()) * (idx == b).mean()
                     for b in range(bins) if (idx == b).any()))


def block_bootstrap_ci(d, block: int = 8, draws: int = 10_000, seed: int = 0, alpha: float = 0.05):
    d = np.asarray(d, float)
    n, rng = len(d), np.random.default_rng(seed)
    k = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(draws, k))
    means = d[(starts[:, :, None] + np.arange(block)).reshape(draws, -1)[:, :n]].mean(1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def metrics(pred: pd.DataFrame) -> dict:
    p, y = pred[PCOLS].to_numpy(), pred["class_true"].to_numpy(int)
    yhat = p.argmax(1)
    rec = recall_score(y, yhat, labels=[0, 1, 2], average=None, zero_division=0)
    return {"n": len(y), "n_cut": int((y == 0).sum()), "n_hold": int((y == 1).sum()), "n_hike": int((y == 2).sum()),
            "log_loss": float(log_loss_rows(p, y).mean()), "brier": float(brier_rows(p, y).mean()),
            "accuracy": float((yhat == y).mean()),
            "macro_f1": float(f1_score(y, yhat, labels=[0, 1, 2], average="macro", zero_division=0)),
            "recall_cut": float(rec[0]), "recall_hold": float(rec[1]), "recall_hike": float(rec[2]),
            "ece": ece(p, y)}


def require_stage1() -> None:
    path = OUT / "stage1.json"
    if not path.exists() or not json.loads(path.read_text()).get("stage1_passed"):
        raise PermissionError("stage 1 has not passed (results/poc_stmt/stage1.json); stage 2 is not run")


def build_panel() -> pd.DataFrame:
    lead = lead_meetings(load_meetings())
    daily, alfred_obs = market.load_daily(), alfred.load_all()
    rows, prev = {}, None
    for day, m in lead.iterrows():
        win_start = prev if prev is not None else m["cutoff"] - pd.Timedelta(days=45)
        rows[day] = _row(m["cutoff"], win_start, m["target_after"], m["target_before"], daily, alfred_obs)
        rows[day].update(y_class=int(m["y_class"]), p_prev_change=m["p_prev_change"])
        prev = m["cutoff"]
    df = pd.DataFrame.from_dict(rows, orient="index").sort_index()
    df = df.join(meeting_features(pd.read_pickle(WORK / "statements.pkl"), df))
    df = df[df[S_COLS + M_COLS + P_COLS + X_COLS + ["y_class"]].notna().all(axis=1)]
    df.to_pickle(WORK / "t2_lead7.pkl")
    return df


SPECS = {  # name: (columns, placebo)
    "S+P": (S_COLS + P_COLS, False), "S+P+X": (S_COLS + P_COLS + X_COLS, False),
    "S+M+P": (S_COLS + M_COLS + P_COLS, False), "S+M+P+X": (S_COLS + M_COLS + P_COLS + X_COLS, False),
    "S": (S_COLS, False), "S+X": (S_COLS + X_COLS, False),
    "S+M": (S_COLS + M_COLS, False), "S+M+X": (S_COLS + M_COLS + X_COLS, False),
    "S+P+X placebo": (S_COLS + P_COLS + X_COLS, True), "S+M+P+X placebo": (S_COLS + M_COLS + P_COLS + X_COLS, True),
}
PAIRS = {"P1": ("S+P", "S+P+X"), "P2": ("S+M+P", "S+M+P+X"),
         "S vs S+X (descriptive)": ("S", "S+X"), "S+M vs S+M+X (descriptive)": ("S+M", "S+M+X"),
         "P1 placebo (descriptive)": ("S+P", "S+P+X placebo"), "P2 placebo (descriptive)": ("S+M+P", "S+M+P+X placebo")}


def run_stage2() -> dict:
    require_stage1()
    df = build_panel()
    start, end = STAGE2
    preds = {name: walk_forward(df, cols, start, end, placebo=pl) for name, (cols, pl) in SPECS.items()}
    preds["climatology"] = climatology(df, start, end)
    preds["always hold"] = always_hold(preds["S+P"])
    (OUT / "stage2").mkdir(parents=True, exist_ok=True)
    for name, p in preds.items():
        p.to_csv(OUT / "stage2" / f"preds_{name.replace(' ', '_')}.csv", index=False)
    table = pd.DataFrame({name: metrics(p) for name, p in preds.items()}).T
    table.to_csv(OUT / "stage2" / "metrics.csv")
    comp = []
    for label, (a, b) in PAIRS.items():
        ya = preds[a]["class_true"].to_numpy()
        d = log_loss_rows(preds[a][PCOLS].to_numpy(), ya) - log_loss_rows(preds[b][PCOLS].to_numpy(), ya)
        lo, hi = block_bootstrap_ci(d)
        comp.append({"pair": label, "without": a, "with": b, "logloss_without": table.loc[a, "log_loss"],
                     "logloss_with": table.loc[b, "log_loss"], "mean_gain": float(d.mean()), "ci_low": lo, "ci_high": hi})
    comp = pd.DataFrame(comp)
    comp.to_csv(OUT / "stage2" / "comparisons.csv", index=False)
    rows = df[(df.index >= start) & (df.index <= end)]
    rho = float(stats.spearmanr(rows["x_shift"], rows["y_change"]).statistic)
    gains = comp.set_index("pair")["mean_gain"]
    res = {"period": STAGE2, "n_meetings": int(len(rows)), "spearman_shift_vs_change": rho,
           "P1_logloss_lower_with_X": bool(gains["P1"] > 0), "P2_logloss_lower_with_X": bool(gains["P2"] > 0)}
    res["stage2_passed"] = bool(rho > 0 and (res["P1_logloss_lower_with_X"] or res["P2_logloss_lower_with_X"]))
    res["pairs_for_stage3"] = [p for p in ("P1", "P2") if res[f"{p}_logloss_lower_with_X"]] if res["stage2_passed"] else []
    (OUT / "stage2.json").write_text(json.dumps(res, indent=2))
    print(table.round(4).to_string(), "\n", comp.round(4).to_string(), "\n", json.dumps(res, indent=2), flush=True)
    return res


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["stage2"])
    ap.parse_args(argv)
    run_stage2()


if __name__ == "__main__":
    main()
