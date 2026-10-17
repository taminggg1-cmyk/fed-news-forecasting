"""Plan 5 ablation (exploratory): which text features carry the signal?

The Plan 1 primary model (T1 ridge, T2 ordered logit) with S plus one feature group, and with S+N minus one
group. Each is compared with the stored S forecasts: RMSE, change against S, DM-HLN p and Clark-West p.
"""
import pandas as pd

from fed_news import config
from fed_news.evaluation.clark_west import compare
from fed_news.evaluation.evaluate import metrics
from fed_news.features.panel import N_COLS, S_COLS
from fed_news.studies.plan1_linear.run_models import load_task
from fed_news.models.walkforward import Spec, run_spec

GROUPS = {"stance": ["n_stance", "n_stance_chg"], "beige": ["n_beige"], "newspapers": ["n_news", "n_gpr", "n_epu"]}
OUT = config.RESULTS / "text"


def specs(task: str) -> list[Spec]:
    out = []
    for group, cols in GROUPS.items():
        out.append(Spec(f"S+{group}", task, "primary", S_COLS + cols))
        out.append(Spec(f"S+N-{group}", task, "primary", S_COLS + [c for c in N_COLS if c not in cols]))
    return out


def main() -> pd.DataFrame:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for task in ("T1", "T2"):
        df = load_task(task)
        base = pd.read_csv(config.RESULTS / f"preds_{task}_S.csv", parse_dates=["date"])
        full = pd.read_csv(config.RESULTS / f"preds_{task}_S+N.csv", parse_dates=["date"])
        results = {"S+N (all)": full}
        for spec in specs(task):
            p = run_spec(df, spec, config.TEST_START)
            p.to_csv(OUT / f"preds_{task}_{spec.name}.csv", index=False)
            results[spec.name] = p
        rmse_s = metrics(base)["rmse"]
        for name, p in results.items():
            m = metrics(p)
            rows.append({"task": task, "model": name, "rmse": m["rmse"], "accuracy": m["accuracy"],
                         "change_vs_S_pct": 100 * (m["rmse"] / rmse_s - 1), **compare(base, p)})
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "ablation.csv", index=False)
    print(table.round(4).to_string(index=False))
    return table


if __name__ == "__main__":
    main()
