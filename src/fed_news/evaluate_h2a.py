"""Plan 2 evaluation (spec sections 3 H2a and 8): metrics, the variance rule, tests, Holm, RESULTS_H2A.md.

Tests compare S+N with S for each architecture on the seed-mean ensembles (one-sided DM-HLN; McNemar
for T2). The variance rule uses the per-seed RMSEs. Holm runs across every H2 p-value so far (Plan 1
plus these); the final family is fixed after Plan 3. Nothing here touches the H1 verdict.
"""
import numpy as np
import pandas as pd

from fed_news import config, run_neural
from fed_news.evaluate import PAPER_ROWS, metrics
from fed_news.stats_tests import dm_hln, holm, mcnemar_greater

ARCHS = {"T1": ["lstm", "gru", "tft", "hybrid"], "T2": ["mlstm", "tft", "hybrid"]}
INFOS = ("S", "S+N")


def variance_rule(rmse_base, rmse_new) -> bool:
    base, new = np.asarray(rmse_base, float), np.asarray(rmse_new, float)
    return bool(base.mean() - new.mean() > 2 * max(base.std(ddof=1), new.std(ddof=1)))


def _load(path) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["date"])


def _seed_rows(task, arch) -> list[dict]:
    rows = []
    for info in INFOS:
        for s in run_neural.SEEDS:
            rows.append({"task": task, "arch": arch, "info": info, "seed": s,
                         **metrics(_load(run_neural.seed_path(task, f"{arch}-{info}", s)))})
    return rows


def pair_tests(task, arch) -> dict:
    base = _load(run_neural.ensemble_path(task, f"{arch}-S"))
    new = _load(run_neural.ensemble_path(task, f"{arch}-S+N"))
    if len(base) != len(new) or not (base["date"].to_numpy() == new["date"].to_numpy()).all():
        raise ValueError(f"prediction dates differ: {task} {arch}")
    sq = lambda p: ((p["y_pred"] - p["y_true"]) ** 2).to_numpy()
    out = {f"H2a {task} {arch} DM": dm_hln(sq(base), sq(new))[1]}
    if task == "T2":
        ok = lambda p: (p["class_pred"] == p["class_true"]).to_numpy()
        out[f"H2a {task} {arch} McNemar"] = mcnemar_greater(ok(base), ok(new))[2]
    return out


def main() -> None:
    out_dir = run_neural.OUT
    seeds = pd.DataFrame([r for t, archs in ARCHS.items() for a in archs for r in _seed_rows(t, a)])
    seeds.to_csv(out_dir / "seed_metrics.csv", index=False)
    summary = (seeds.groupby(["task", "arch", "info"])[["rmse", "mae", "accuracy"]]
               .agg(["mean", "std"]).round(4))
    summary.columns = [f"{m} {s}" for m, s in summary.columns]
    summary = summary.reset_index()

    ens_rows, paper_rows = [], []
    for task, archs in ARCHS.items():
        for arch in archs:
            for info in INFOS:
                p = _load(run_neural.ensemble_path(task, f"{arch}-{info}"))
                ens_rows.append({"task": task, "arch": arch, "info": info, **metrics(p)})
                if task == "T1" and arch in ("lstm", "gru"):
                    w = p[(p["date"] >= config.PAPER_T1[0]) & (p["date"] <= config.PAPER_T1[1])]
                    paper_rows.append({"model": f"our {arch.upper()} replication, {info}", **metrics(w)})
    ens = pd.DataFrame(ens_rows)
    ens.to_csv(out_dir / "ensemble_metrics.csv", index=False)
    paper = pd.concat([pd.DataFrame(PAPER_ROWS, columns=["model", "mse", "rmse", "mae", "mbe"]),
                       pd.DataFrame(paper_rows)[["model", "mse", "rmse", "mae", "mbe"]]], ignore_index=True)

    h2a, rule = {}, {}
    for task, archs in ARCHS.items():
        for arch in archs:
            tests = pair_tests(task, arch)
            h2a.update(tests)
            s = seeds[(seeds.task == task) & (seeds.arch == arch)]
            passed = variance_rule(s[s["info"] == "S"]["rmse"], s[s["info"] == "S+N"]["rmse"])
            rule.update({k: passed for k in tests})
    plan1 = pd.read_csv(config.RESULTS / "h2_pvalues.csv")
    family = dict(zip(plan1["hypothesis"], plan1["p"]))
    family.update(h2a)
    adjusted = holm(family)
    tests = pd.DataFrame({"hypothesis": list(family), "p": list(family.values()),
                          "holm_h2_so_far": [adjusted[k] for k in family],
                          "variance_rule": [rule.get(k, "") for k in family]})
    tests.to_csv(out_dir / "tests.csv", index=False)

    lines = [
        "# Results: Plan 2 (H2a, neural architectures)", "",
        "Secondary analyses; they cannot change the H1 verdict in RESULTS.md. "
        "Settings: PREREG_H2A.md (tag prereg-h2a).", "",
        "## Ensemble metrics (mean of 8 seeds' forecasts), test windows from 2001", "",
        ens.to_markdown(index=False, floatfmt=".4f"), "",
        "## Per-seed metrics, mean and standard deviation over 8 seeds", "",
        summary.to_markdown(index=False, floatfmt=".4f"), "",
        "## Paper replication on the paper's window (Jan 2016 - Apr 2025), T1", "",
        paper.to_markdown(index=False, floatfmt=".4f"), "",
        "## Tests: S+N vs S per architecture (one-sided), Holm across all H2 so far", "",
        "The variance rule: a gain counts only if mean RMSE(S) - mean RMSE(S+N) exceeds twice the "
        "larger seed standard deviation. The final Holm family is fixed after Plan 3.", "",
        tests.to_markdown(index=False, floatfmt=".4f"), "",
    ]
    (config.REPO / "RESULTS_H2A.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:4]))


if __name__ == "__main__":
    main()
