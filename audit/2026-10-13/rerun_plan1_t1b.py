"""Post-audit methodological correction (exploratory): Plan 1 monthly task (T1) on the corrected cutoff.

Defect: the Plan 1 monthly panel `t1` dates each month's forecast at the first calendar day. At that time
FEDFUNDS(t-1) and other inputs are not guaranteed to be published (design section 12.1 of the Study 4 spec).
Panel `t1b` (built 2026-10-13, commit e2b74a8; cutoff = start of the second US business day; verified in
results/study4/v2_verification.json) fixes this.

This script reruns the Plan 1 T1 ladder (rw, taylor, S, S+N, placebo, shuffled, S+M, S+M+N) on `t1b_full`.
Model settings, penalty grids, the t-2 training rule, the seed and the test window are unchanged
(walkforward.ladder / run_spec). Evaluation uses the registered tests: DM-HLN on squared error, the placebo and
shuffled controls of the H1 rule, and H2c/H2d. It adds the audit's Newey-West p and the paired block-bootstrap
interval (12 months, 10,000 draws, seed 0). The original results in results/ are untouched. The meeting task
(T2) is unaffected and is not rerun.

Run from the repository root:  PYTHONPATH=src python audit/2026-10-13/rerun_plan1_t1b.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from fed_news import config
from fed_news.evaluate import metrics
from fed_news.stats_tests import dm_hln
from fed_news.walkforward import common_rows, ladder, run_spec

sys.path.insert(0, str(Path(__file__).parent))
from recompute import boot_ci, nw_p  # noqa: E402  (the audit's own Newey-West and bootstrap)

OUT = config.RESULTS / "post_audit_t1b"


def sq(p):
    return ((p["y_pred"] - p["y_true"]) ** 2).to_numpy()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = common_rows(pd.read_pickle(config.DATA / "t1b_full.pkl"))
    print(f"t1b: {len(df)} common rows, {df.index.min().date()} .. {df.index.max().date()}", flush=True)
    preds = {}
    for spec in ladder("T1"):
        path = OUT / f"preds_T1_{spec.name}.csv"
        if not path.exists():
            run_spec(df, spec, config.TEST_START).to_csv(path, index=False)
        preds[spec.name] = pd.read_csv(path, parse_dates=["date"])
    old = {n: pd.read_csv(config.RESULTS / f"preds_T1_{n}.csv", parse_dates=["date"]) for n in preds}
    for n, p in preds.items():
        assert p["date"].equals(preds["S"]["date"]), f"{n}: dates differ from S"
        assert p["date"].equals(old[n]["date"]), f"{n}: dates differ from the original run"

    rows = []
    for n, p in preds.items():
        m_new, m_old = metrics(p), metrics(old[n])
        rows.append({"model": n, "n": m_new["n"], "rmse_t1b": m_new["rmse"], "rmse_t1_original": m_old["rmse"],
                     "mae_t1b": m_new["mae"], "mbe_t1b": m_new["mbe"], "accuracy_t1b": m_new["accuracy"]})
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "metrics_T1_t1b.csv", index=False)

    def pair(new, base):
        d = sq(preds[base]) - sq(preds[new])
        return {"comparison": f"{base} -> {new}", "n": len(d), "rmse_base": float(np.sqrt(sq(preds[base]).mean())),
                "rmse_new": float(np.sqrt(sq(preds[new]).mean())), "mse_diff": float(d.mean()),
                "mse_diff_ci90": boot_ci(d, 12), "dm_hln_p": float(dm_hln(sq(preds[base]), sq(preds[new]))[1]),
                "nw_p": nw_p(d, 12)}

    tests = {"H1 T1 (S+N vs S)": pair("S+N", "S"), "H1 control: placebo vs S": pair("placebo", "S"),
             "H1 control: shuffled vs rw": pair("shuffled", "rw"), "H2c T1 (S+N vs rw)": pair("S+N", "rw"),
             "H2d T1 (S+M+N vs S+M)": pair("S+M+N", "S+M")}
    out = {"label": "post-audit methodological correction, exploratory", "panel": "t1b_full.pkl",
           "test_window": [str(preds["S"]["date"].min().date()), str(preds["S"]["date"].max().date())],
           "tests": tests, "metrics": table.to_dict(orient="records")}
    (OUT / "tests_T1_t1b.json").write_text(json.dumps(out, indent=2, default=float))
    pd.set_option("display.width", 200)
    print(table.round(4).to_string(index=False))
    for k, v in tests.items():
        print(f"{k:32s} RMSE {v['rmse_base']:.4f} -> {v['rmse_new']:.4f}  dMSE {v['mse_diff']:+.4f} "
              f"[{v['mse_diff_ci90'][0]:+.4f}, {v['mse_diff_ci90'][1]:+.4f}]  DM-HLN p {v['dm_hln_p']:.4f}  "
              f"NW p {v['nw_p']:.4f}")


if __name__ == "__main__":
    main()
