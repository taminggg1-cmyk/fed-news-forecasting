"""Audit recomputation (2026-10-13): every text-vs-no-text comparison, recomputed from the saved prediction files.

Nothing here retrains a model or edits an existing result. For each pair it checks that the two files cover the
same dates and the same outcomes, then reports the metric with and without text, the difference, a one-sided
Newey-West p-value on the per-row loss difference, and a paired moving-block bootstrap 90% interval
(blocks = the refit interval: 8 meetings or 12 months; 10,000 draws; seed 0; blocks drawn with replacement
from the row order, so the time order inside a block is kept). Accuracy differences get the same bootstrap and
an exact McNemar count, never the squared-error p-value.

Run from the repository root:  PYTHONPATH=src python audit/2026-10-13/recompute.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

R = Path("results")
OUT = Path(__file__).parent
DRAWS, SEED = 10_000, 0
PROBS = ["p_cut", "p_hold", "p_hike"]


def nw_p(d, lags):
    fit = sm.OLS(np.asarray(d, float), np.ones(len(d))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return float(1 - stats.norm.cdf(fit.params[0] / fit.bse[0]))


def boot_ci(d, block):
    d = np.asarray(d, float)
    n, rng = len(d), np.random.default_rng(SEED)
    k = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(DRAWS, k))
    idx = (starts[:, :, None] + np.arange(block)).reshape(DRAWS, -1)[:, :n]
    means = d[idx].mean(1)
    return float(np.percentile(means, 5)), float(np.percentile(means, 95))


def mcnemar(ok_b, ok_n):
    fixes, breaks = int((~ok_b & ok_n).sum()), int((ok_b & ~ok_n).sum())
    p = float(stats.binomtest(fixes, fixes + breaks, 0.5, alternative="greater").pvalue) if fixes + breaks else 1.0
    return fixes, breaks, p


def load(path):
    return pd.read_csv(R / path, parse_dates=["date"])


def logloss(p):
    q = np.clip(p[PROBS].to_numpy(float), 1e-12, None)
    return -np.log(q[np.arange(len(q)), p["class_true"].to_numpy(int)])


def compare(group, study, role, task, label, base_f, new_f, block, metric, mask=None, subset="all"):
    b, n = load(base_f), load(new_f)
    if mask is not None:
        b, n = b[b["date"].isin(mask)].reset_index(drop=True), n[n["date"].isin(mask)].reset_index(drop=True)
    assert (b["date"].values == n["date"].values).all(), f"dates differ: {base_f} {new_f}"
    assert np.allclose(b["y_true"], n["y_true"]), f"outcomes differ: {base_f} {new_f}"
    y = b["y_true"].to_numpy(float)
    se_b, se_n = (y - b["y_pred"]) ** 2, (y - n["y_pred"]) ** 2
    ok_b = (b["class_pred"] == b["class_true"]).to_numpy()
    ok_n = (n["class_pred"] == n["class_true"]).to_numpy()
    row = {"group": group, "study": study, "role": role, "task": task, "comparison": label, "subset": subset,
           "n": len(b), "first": b["date"].min().date(), "last": b["date"].max().date(), "block": block,
           "rmse_base": float(np.sqrt(se_b.mean())), "rmse_text": float(np.sqrt(se_n.mean())),
           "mse_diff": float((se_b - se_n).mean()), "mse_diff_ci90": boot_ci(se_b - se_n, block),
           "mse_nw_p": nw_p(se_b - se_n, block),
           "acc_base": float(ok_b.mean()), "acc_text": float(ok_n.mean()),
           "acc_diff": float(ok_n.mean() - ok_b.mean()), "acc_diff_ci90": boot_ci(ok_n.astype(float) - ok_b, block)}
    row["mcnemar_fixes"], row["mcnemar_breaks"], row["mcnemar_p"] = mcnemar(ok_b, ok_n)
    if metric == "log loss":
        ll_b, ll_n = logloss(b), logloss(n)
        row.update({"ll_base": float(ll_b.mean()), "ll_text": float(ll_n.mean()),
                    "ll_diff": float((ll_b - ll_n).mean()), "ll_diff_ci90": boot_ci(ll_b - ll_n, block),
                    "ll_nw_p": nw_p(ll_b - ll_n, block)})
    row["tested_metric"] = metric
    row["base_file"], row["text_file"] = base_f, new_f
    return row


def main():
    meet = pd.read_pickle("D:/fed_news/meetings.pkl")
    scheduled = meet.index[meet["unscheduled"] != 1]
    rows = []
    # --- Group 1: mixed vintage (revised-data fallback in some S columns) -------------------------------
    for t, blk in (("T1", 12), ("T2", 8)):
        rows.append(compare("mixed", "Plan 1 (ordered logit)", "primary (H1)", t, "S -> S+N",
                            f"preds_{t}_S.csv", f"preds_{t}_S+N.csv", blk, "MSE"))
        rows.append(compare("mixed", "Plan 1 (ordered logit)", "exploratory", t, "S+M -> S+M+N",
                            f"preds_{t}_S+M.csv", f"preds_{t}_S+M+N.csv", blk, "MSE"))
    for new in ("S+M+H", "S+M+C"):
        rows.append(compare("mixed", "Study 4 (ordered logit)", "primary" if new == "S+M+H" else "secondary", "T2",
                            f"S+M -> {new}", "study4/preds_T2_S+M.csv", f"study4/preds_T2_{new}.csv", 8,
                            "log loss"))
    for arch in ("lstm", "gru", "tft"):
        for new in ("S+M+C", "S+M+H"):
            rows.append(compare("mixed", "Study 4 v2 networks", "primary (Holm family of 6)", "T2",
                                f"{arch.upper()} S+M -> {new}", f"study4/neural/preds_{arch}_S+M.csv",
                                f"study4/neural/preds_{arch}_{new}.csv", 8, "MSE", scheduled, "scheduled"))
        for new in ("S+C", "S+H"):
            rows.append(compare("mixed", "Study 4 v2 networks", "primary (Holm family of 6)", "T1",
                                f"{arch.upper()} S -> {new}", f"study4/neural_t1/preds_{arch}_S.csv",
                                f"study4/neural_t1/preds_{arch}_{new}.csv", 12, "MSE"))
    # --- Group 2: fallback columns dropped, but fallback values remain in pre-2001 training rows -------
    for new in ("S+M+H", "S+M+C"):
        rows.append(compare("partial", "Study 4 section 8.1 (ordered logit)", "exploratory", "T2",
                            f"rt S+M -> rt {new}", "study4/sensitivity/preds_rt_S+M.csv",
                            f"study4/sensitivity/preds_rt_{new}.csv", 8, "log loss"))
    # --- Group 3: strict real-time (only series with complete vintages; no fallback anywhere) ----------
    for arch in ("lstm", "gru", "tft"):
        for new in ("strict+C", "strict+H"):
            rows.append(compare("strict", "Study 4 v2 networks", "robustness", "T2", f"{arch.upper()} strict -> {new}",
                                f"study4/neural/preds_{arch}_strict.csv", f"study4/neural/preds_{arch}_{new}.csv",
                                8, "MSE", scheduled, "scheduled"))
            rows.append(compare("strict", "Study 4 v2 networks", "robustness", "T1", f"{arch.upper()} strict -> {new}",
                                f"study4/neural_t1/preds_{arch}_strict.csv",
                                f"study4/neural_t1/preds_{arch}_{new}.csv", 12, "MSE"))
    for h, new, role in ((3, "B+C", "primary"), (3, "B+H", "secondary"), (2, "B+C", "secondary"),
                         (2, "B+H", "secondary"), (1, "B+C", "reference")):
        rows.append(compare("strict", "Study 5 (ordered logit)", role, f"h={h}", f"B -> {new}",
                            f"study5/preds_h{h}_B.csv", f"study5/preds_h{h}_{new}.csv", 8, "log loss"))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "recomputed_comparisons.csv", index=False)
    pd.set_option("display.width", 250)
    cols = ["group", "study", "task", "comparison", "n", "rmse_base", "rmse_text", "mse_nw_p", "acc_base",
            "acc_text", "mcnemar_p", "ll_base", "ll_text", "ll_nw_p"]
    print(df[[c for c in cols if c in df]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
