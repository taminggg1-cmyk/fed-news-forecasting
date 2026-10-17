"""Plan 2 evaluation (spec sections 3 H2a and 8): metrics, the variance rule, tests, Holm, RESULTS_H2A.md.

Tests compare S+N with S for each architecture on the seed-mean ensembles (one-sided DM-HLN; McNemar
for T2). The variance rule uses the per-seed RMSEs. Holm runs across every H2 p-value so far (Plan 1
plus these); the final family is fixed after Plan 3. Nothing here touches the H1 verdict.
The exploratory sections (hybrid against its own anchor, MSE decomposition, best epochs, GPU re-run)
were added after the results were known and are labelled so in the report.
"""
import numpy as np
import pandas as pd

from fed_news import config
from fed_news.studies.plan2_neural_original import run_neural
from fed_news.evaluation.evaluate import PAPER_ROWS, _git, metrics
from fed_news.studies.plan2_neural_original.neural_walk import H2A_TAG
from fed_news.evaluation.stats_tests import dm_hln, holm, mcnemar_greater

ARCHS = {"T1": ["lstm", "gru", "tft", "hybrid"], "T2": ["mlstm", "tft", "hybrid"]}
INFOS = ("S", "S+N")
PLAN2_HEADING = "### Plan 2 (H2a) deviations"
EVAL_COMMIT = "0554690"


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


def deviations_section(repo=None) -> str:
    """The Plan 2 block of DEVIATIONS.md, read at run time (its heading to the next ### heading)."""
    path = (repo or config.REPO) / "DEVIATIONS.md"
    if not path.exists():
        return "None recorded."
    text = path.read_text(encoding="utf-8")
    start = text.find(PLAN2_HEADING)
    if start < 0:
        return "None recorded."
    nxt = text.find("\n### ", start + len(PLAN2_HEADING))
    return text[start: nxt if nxt >= 0 else len(text)].strip()


def provenance_section(repo=None) -> str:
    repo = repo or config.REPO
    tag = _git(repo, "rev-list", "-n", "1", H2A_TAG) or "no tag"
    diff = _git(repo, "diff", "--stat", H2A_TAG, "--", "src") if tag != "no tag" else ""
    return "\n".join([f"- prereg-h2a tag commit: {tag}", f"- HEAD: {_git(repo, 'rev-parse', '--short', 'HEAD')}",
                      "- `git diff --stat prereg-h2a -- src`:", "", "```", diff, "```", "",
                      f"evaluate_h2a.py was committed ({EVAL_COMMIT}) before the first forecast file was written. "
                      "Later edits add report sections only; the pre-registered test code (pair_tests, "
                      "variance_rule, Holm) is unchanged."])


def _plan1(task, name) -> pd.DataFrame:
    return _load(config.RESULTS / f"preds_{task}_{name}.csv")


def _sq(p) -> np.ndarray:
    return ((p["y_pred"] - p["y_true"]) ** 2).to_numpy()


def reading_of_tests(tests: pd.DataFrame) -> str:
    """A generated paragraph; every count comes from the tests table."""
    h = tests[tests["hypothesis"].str.startswith("H2a")]
    n = len(h)
    passes = h["variance_rule"].apply(lambda v: isinstance(v, (bool, np.bool_)) and bool(v))
    holm_sig = h[h["holm_h2_so_far"] < config.ALPHA]
    raw = h[h["p"] < config.ALPHA]
    both = raw[passes.loc[raw.index]]
    fmt = lambda f: "; ".join(f"{r.hypothesis} (p = {r.p:.4f})" for r in f.itertuples()) or "none"
    best = h.loc[h["holm_h2_so_far"].idxmin()]
    return (f"There are {n} H2a tests (a one-sided DM test per architecture, plus a McNemar test for each T2 "
            f"architecture). After Holm across all H2 p-values so far, {len(holm_sig)} of {n} have an adjusted "
            f"p below {config.ALPHA} (smallest adjusted p: {best.holm_h2_so_far:.4f}, {best.hypothesis}). "
            f"{len(raw)} of {n} have a raw one-sided p below {config.ALPHA}: {fmt(raw)}. "
            f"{len(both)} of those {len(raw)} also pass the variance rule: {fmt(both)}. "
            "The holm_h2_so_far values can only rise when Plan 3 adds hypotheses, so they are lower bounds of "
            "the final adjusted p-values. PREREG_H2A.md defines the per-test p-value, the variance rule and "
            "the Holm correction, but no per-architecture verdict: these are counts of tests, not of models.")


def anchor_comparison() -> pd.DataFrame:
    rows = []
    for task in ARCHS:
        for info in INFOS:
            hyb = _load(run_neural.ensemble_path(task, f"hybrid-{info}"))
            anchor, rw = _plan1(task, info), _plan1(task, "rw")
            for other in (anchor, rw):
                if len(other) != len(hyb) or not (other["date"].to_numpy() == hyb["date"].to_numpy()).all():
                    raise ValueError(f"dates differ: {task} hybrid-{info} vs Plan 1")
            rows.append({"task": task, "info": info, "hybrid rmse": metrics(hyb)["rmse"],
                         "own anchor rmse": metrics(anchor)["rmse"], "rw rmse": metrics(rw)["rmse"],
                         "p anchor beats hybrid": dm_hln(_sq(hyb), _sq(anchor))[1],
                         "p rw beats hybrid": dm_hln(_sq(hyb), _sq(rw))[1]})
    return pd.DataFrame(rows)


def mse_decomposition() -> pd.DataFrame:
    """Hybrid MSE(S) - MSE(S+N) = (linear MSE(S) - linear MSE(S+N)) + remainder (the TFT corrections)."""
    rows = []
    for task in ARCHS:
        hyb = {i: metrics(_load(run_neural.ensemble_path(task, f"hybrid-{i}")))["mse"] for i in INFOS}
        lin = {i: metrics(_plan1(task, i))["mse"] for i in INFOS}
        gap, lin_gap = hyb["S"] - hyb["S+N"], lin["S"] - lin["S+N"]
        rows.append({"task": task, "hybrid mse gap": gap, "linear mse gap (Plan 1)": lin_gap,
                     "remainder (TFT corrections)": gap - lin_gap})
    return pd.DataFrame(rows)


def epochs_table(diag) -> str:
    path = diag / "epochs.csv"
    if not path.exists():
        return "best-epoch diagnostic not run"
    e = pd.read_csv(path)
    g = e.groupby(["task", "spec"])["best_epoch"]
    t = pd.DataFrame({"seeds": e.groupby(["task", "spec"])["seed"].nunique(), "refits": g.size(), "share best epoch 0": g.apply(lambda x: float((x == 0).mean())),
                      "median best epoch": g.median()}).reset_index()
    return t.to_markdown(index=False, floatfmt=".3f")


def exploratory_section() -> list[str]:
    diag = config.RESULTS / "h2a_diag"
    lines = [
        "## Exploratory analyses (exploratory, added after results)", "",
        "Added after the results were known, following the final review. Not pre-registered; they do not "
        "change any test above.", "",
        "### Hybrid against its own linear anchor and against the random walk (exploratory, added after results)",
        "", "One-sided DM p-values on the seed-mean ensembles; the anchor is the Plan 1 forecast "
        "results/preds_<task>_<S|S+N>.csv. A small p means the anchor (or the random walk) is better.", "",
        anchor_comparison().to_markdown(index=False, floatfmt=".3g"), "",
        "### MSE decomposition of the hybrid S vs S+N gap (exploratory, added after results)", "",
        mse_decomposition().to_markdown(index=False, floatfmt=".5f"), "",
        "### Best epoch kept by the TFT early stopping (exploratory, added after results)", "",
        "Epoch 0 means the untrained weights were kept. Every refit of every seed listed in the seeds column.", "", epochs_table(diag), "",
        "### GPU re-run of the TFT and hybrid specs (exploratory, added after results)", "",
    ]
    rerun = diag / "rerun_metrics.csv"
    if rerun.exists():
        lines += ["The same code and seeds re-run once more, to bound GPU floating-point nondeterminism. The "
                  "committed forecasts in results/h2a/ are unchanged.", "",
                  pd.read_csv(rerun).to_markdown(index=False, floatfmt=".4f"), ""]
    else:
        lines += ["re-run not done", ""]
    return lines


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
                    label = ("S (refit every 3 months; paper: monthly; real-time data)" if info == "S"
                             else "S+N (not in paper; refit every 3 months, real-time data)")
                    paper_rows.append({"model": f"our {arch.upper()}, {label}", **metrics(w)})
    for task in ARCHS:  # context rows: random walk and the Plan 1 linear models
        for arch, info, name in (("rw", "", "rw"), ("linear", "S", "S"), ("linear", "S+N", "S+N")):
            ens_rows.append({"task": task, "arch": arch, "info": info, **metrics(_plan1(task, name))})
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
        "Rows with arch rw and linear are the Plan 1 random walk and linear S / S+N models, for context.", "",
        ens.to_markdown(index=False, floatfmt=".4f"), "",
        "## Per-seed metrics, mean and standard deviation over 8 seeds", "",
        summary.to_markdown(index=False, floatfmt=".4f"), "",
        "## Paper replication on the paper's window (Jan 2016 - Apr 2025), T1", "",
        paper.to_markdown(index=False, floatfmt=".4f"), "",
        "## Reading of the tests", "", reading_of_tests(tests), "",
        "## Tests: S+N vs S per architecture (one-sided), Holm across all H2 so far", "",
        "The variance rule: a gain counts only if mean RMSE(S) - mean RMSE(S+N) exceeds twice the "
        "larger seed standard deviation. The final Holm family is fixed after Plan 3.", "",
        tests.to_markdown(index=False, floatfmt=".4f"), "",
        *exploratory_section(),
        "## Deviations", "", deviations_section(), "",
        "## Provenance", "", provenance_section(), "",
    ]
    (config.REPO / "RESULTS_H2A.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:4]))


if __name__ == "__main__":
    main()
