"""Study 3 forecasting and tests (PREREG_STUDY3.md, hypothesis H7).

The decision-trained document scores (decision_text) become the meeting signal D: naive_stance.naive_panel
writes the mean of group means over [previous cutoff, cutoff), carried forward, into the `n_stance` column of a
copy of the T2 panel. Every model is the Plan 1 penalised ordered logit (walkforward.run_spec, unchanged) on
exactly the same rows. `--power` writes the pre-registration power statement from the committed Plan 1
forecasts and needs no Study 3 output.
"""
import argparse

import numpy as np
import pandas as pd
from scipy import stats

from fed_news import config
from fed_news.clark_west import compare
from fed_news.decision_text import OUT_PATH, RESULTS, SHUFFLED_PATH, gate
from fed_news.evaluate import metrics
from fed_news.naive_stance import naive_panel
from fed_news.panel import M_COLS, N_COLS, S_COLS
from fed_news.stats_tests import dm_hln, mcnemar_greater
from fed_news.walkforward import Spec, common_rows, run_spec

D = ["n_stance"]  # the decision signal lives in this column of the Study 3 panel
Z80 = float(stats.norm.ppf(0.95) + stats.norm.ppf(0.80))
CW_TESTS = (("H7 Clark-West", "primary", "S", "S+D"),
            ("placebo Clark-West", "control", "S", "placebo"),
            ("shuffled-text Clark-West", "control", "S", "shuffled-text"),
            ("S+M+D vs S+M Clark-West", "secondary", "S+M", "S+M+D"))


def same_rows(*frames: pd.DataFrame) -> list[pd.DataFrame]:
    rows = frames[0].index
    for f in frames[1:]:
        rows = rows.intersection(f.index)
    return [f.loc[rows] for f in frames]


def panels(ds: pd.DataFrame, ds_shuffled: pd.DataFrame) -> list[pd.DataFrame]:
    """Decision-signal panel, shuffled-label-signal panel and the original Plan 1 panel, on the same rows."""
    return same_rows(common_rows(naive_panel("t2", ds)), common_rows(naive_panel("t2", ds_shuffled)),
                     common_rows(pd.read_pickle(config.DATA / "t2_full.pkl")))


def forecasts(dec: pd.DataFrame, shu: pd.DataFrame, orig: pd.DataFrame) -> dict[str, pd.DataFrame]:
    def run(df, name, cols, **kw):
        return run_spec(df, Spec(name, "T2", "primary", cols, **kw), config.TEST_START)
    return {"S": run(dec, "S", S_COLS),
            "S+D": run(dec, "S+D", S_COLS + D),
            "placebo": run(dec, "placebo", S_COLS + D, placebo=True),
            "shuffled-text": run(shu, "shuffled-text", S_COLS + D),
            "S+M": run(dec, "S+M", S_COLS + M_COLS),
            "S+M+D": run(dec, "S+M+D", S_COLS + M_COLS + D),
            "S+N": run(orig, "S+N", S_COLS + N_COLS)}


def _correct(p: pd.DataFrame) -> np.ndarray:
    return (p["class_pred"] == p["class_true"]).to_numpy()


def _sq(p: pd.DataFrame) -> np.ndarray:
    return ((p["y_pred"] - p["y_true"]) ** 2).to_numpy(float)


def hypothesis_tests(f: dict[str, pd.DataFrame]) -> pd.DataFrame:
    def base_row(test, role, base, model):
        a, b = metrics(f[base]), metrics(f[model])
        return {"test": test, "role": role, "base": base, "model": model, "n": a["n"], "rmse_base": a["rmse"],
                "rmse_model": b["rmse"], "accuracy_base": a["accuracy"], "accuracy_model": b["accuracy"]}
    rows = []
    for test, role, base, model in CW_TESTS:
        c = compare(f[base], f[model])
        rows.append({**base_row(test, role, base, model), "stat": c["clark_west_t"], "p": c["clark_west_p"]})
    compare(f["S"], f["S+D"])                                   # raises if the forecast dates differ
    fixes, breaks, p = mcnemar_greater(_correct(f["S"]), _correct(f["S+D"]))
    rows.append({**base_row("decisions McNemar", "secondary", "S", "S+D"), "stat": fixes - breaks, "p": p})
    for test, base in (("DM S+D vs S", "S"), ("DM S+D vs S+N", "S+N")):
        compare(f[base], f["S+D"])
        stat, p = dm_hln(_sq(f[base]), _sq(f["S+D"]))
        rows.append({**base_row(test, "secondary", base, "S+D"), "stat": stat, "p": p})
    return pd.DataFrame(rows)


def decide_h7(p_main: float, p_placebo: float, p_shuffled: float, alpha: float = config.ALPHA) -> str:
    if any(np.isnan(p) for p in (p_main, p_placebo, p_shuffled)):
        raise ValueError("NaN p-value in the H7 decision")
    if p_placebo < alpha or p_shuffled < alpha:
        return "void"
    return "supported" if p_main < alpha else "not supported"


def power_statement(base: pd.DataFrame, new: pd.DataFrame) -> dict:
    """MSE improvement a one-sided 5% test detects with 80% power: Z80 * sd(loss differential) / sqrt(n)."""
    y = base["y_true"].to_numpy(float)
    e_base = (y - base["y_pred"].to_numpy(float)) ** 2
    d = e_base - (y - new["y_pred"].to_numpy(float)) ** 2
    mse, sd = float(e_base.mean()), float(np.std(d, ddof=1))
    mde = Z80 * sd / np.sqrt(len(d))
    return {"n": len(d), "mse_base": mse, "sd_loss_diff": sd, "z": Z80, "mde_mse": mde,
            "mde_pct_of_mse": 100 * mde / mse, "rmse_base": float(np.sqrt(mse)),
            "rmse_at_mde": float(np.sqrt(max(mse - mde, 0.0)))}


def power_sentence(p: dict) -> str:
    return (f"With {int(p['n'])} meetings, a one-sided 5% test has about 80% power only if the text model lowers "
            f"the mean squared error by at least {p['mde_mse']:.4f} ({p['mde_pct_of_mse']:.1f}% of the "
            f"statistics-only MSE {p['mse_base']:.4f}; RMSE from {p['rmse_base']:.3f} to {p['rmse_at_mde']:.3f}). "
            f"Computed from the spread of the squared-error differential of the committed Plan 1 per-meeting "
            f"forecasts (S vs S+N, sd {p['sd_loss_diff']:.4f}), normal approximation.")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--power", action="store_true", help="write the registration power statement only")
    args = ap.parse_args(argv)
    RESULTS.mkdir(parents=True, exist_ok=True)
    if args.power:
        load = lambda n: pd.read_csv(config.RESULTS / f"preds_T2_{n}.csv", parse_dates=["date"])
        p = power_statement(load("S"), load("S+N"))
        pd.DataFrame([p]).to_csv(RESULTS / "power.csv", index=False)
        print(power_sentence(p))
        return
    reason = gate()
    if reason:
        raise PermissionError(f"Study 3 forecasting refused: {reason}")
    dec, shu, orig = panels(pd.read_pickle(OUT_PATH), pd.read_pickle(SHUFFLED_PATH))
    print(f"{len(dec)} common rows, {dec.index.min().date()} .. {dec.index.max().date()}", flush=True)
    f = forecasts(dec, shu, orig)
    for name, p in f.items():
        p.to_csv(RESULTS / f"preds_T2_{name}.csv", index=False)
    table = hypothesis_tests(f)
    table.to_csv(RESULTS / "tests.csv", index=False)
    print(table.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
