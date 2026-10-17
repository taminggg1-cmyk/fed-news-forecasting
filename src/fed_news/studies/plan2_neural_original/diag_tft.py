"""Exploratory diagnostics for the TFT and hybrid specs (added after results, following the final review).

Re-runs every TFT/hybrid spec of neural_ladder("T1") and ("T2") for seeds 0-7 with the unchanged code and
records, for every refit, the epoch the early stopping kept (0 = the untrained weights). The re-run
forecasts are written to results/h2a_diag/ and never replace results/h2a/. Comparing them with the
committed forecasts bounds GPU floating-point nondeterminism: rerun_metrics.csv holds, per spec, the
committed and re-run ensemble RMSE and the largest forecast difference; for the hybrid S+N vs S pair it also
holds the DM p-value and the variance-rule outcome of both runs.

    PYTHONPATH=src python -m fed_news.diag_tft [--device cuda]
"""
import argparse
import os
from unittest import mock

import numpy as np
import pandas as pd

from fed_news import config, neural_walk, run_neural
from fed_news.evaluate import metrics
from fed_news.evaluate_h2a import variance_rule
from fed_news.neural_walk import neural_ladder, run_neural_spec
from fed_news.run_models import load_task
from fed_news.stats_tests import dm_hln

OUT = config.RESULTS / "h2a_diag"
EPOCH_COLS = ["task", "spec", "seed", "refit_date", "best_epoch"]


def diag_specs(task: str) -> list:
    return [s for s in neural_ladder(task) if s.early_stop]


def _write(frame: pd.DataFrame, path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def rerun_path(out, task, spec, seed):
    return out / f"preds_{task}_{spec}_seed{seed}.csv"


def _recording(original, sink: list):
    def wrapper(model, *args, **kwargs):
        result = original(model, *args, **kwargs)
        sink.append(int(result.best_epoch_))
        return result
    return wrapper


def run_diag(frames: dict, device: str = "cuda", seeds=run_neural.SEEDS, out=OUT, test_start=config.TEST_START,
             epochs: int | None = None, require_prereg: bool = True, specs: dict | None = None) -> pd.DataFrame:
    """Re-run the early-stopping specs; return the epochs table (also written to out/epochs.csv)."""
    out.mkdir(parents=True, exist_ok=True)
    epochs_path = out / "epochs.csv"
    table = pd.read_csv(epochs_path) if epochs_path.exists() else pd.DataFrame(columns=EPOCH_COLS)
    for task, df in frames.items():
        test_dates = df.index[df.index >= pd.Timestamp(test_start)]
        for spec in (specs[task] if specs else diag_specs(task)):
            for seed in seeds:
                done = ((table["task"] == task) & (table["spec"] == spec.name) & (table["seed"] == seed)).any()
                path = rerun_path(out, task, spec.name, seed)
                if done and path.exists():
                    continue
                sink: list = []
                with mock.patch.object(neural_walk, "fit_early_stopping",
                                       _recording(neural_walk.fit_early_stopping, sink)):
                    preds = run_neural_spec(df, spec, seed, test_start, device=device, epochs=epochs,
                                            require_prereg=require_prereg)
                refits = test_dates[:: spec.refit_every]
                if len(sink) != len(refits):
                    raise RuntimeError(f"{task}/{spec.name}/seed{seed}: {len(sink)} fits, {len(refits)} refits")
                new = pd.DataFrame({"task": task, "spec": spec.name, "seed": seed,
                                    "refit_date": [d.date().isoformat() for d in refits], "best_epoch": sink})
                table = pd.concat([table[~((table["task"] == task) & (table["spec"] == spec.name)
                                           & (table["seed"] == seed))], new], ignore_index=True)
                _write(preds, path)
                _write(table[EPOCH_COLS], epochs_path)
                print(f"{task} {spec.name} seed {seed}: kept epoch 0 in {sum(e == 0 for e in sink)}/{len(sink)}",
                      flush=True)
    return table


def rerun_metrics(tasks=("T1", "T2"), seeds=run_neural.SEEDS, out=OUT, specs: dict | None = None) -> pd.DataFrame:
    """Committed vs re-run ensemble RMSE and largest forecast difference, per spec (written to out)."""
    rows, per_seed = [], {}
    for task in tasks:
        for spec in (specs[task] if specs else diag_specs(task)):
            files = [rerun_path(out, task, spec.name, s) for s in seeds]
            if not all(f.exists() for f in files):
                continue
            new = [pd.read_csv(f, parse_dates=["date"]) for f in files]
            old = [pd.read_csv(run_neural.seed_path(task, spec.name, s), parse_dates=["date"]) for s in seeds]
            diff = max(float(np.abs(a["y_pred"].to_numpy() - b["y_pred"].to_numpy()).max()) for a, b in zip(new, old))
            ens_new, ens_old = run_neural.ensemble(new), run_neural.ensemble(old)
            per_seed[task, spec.name] = ([metrics(f)["rmse"] for f in old], [metrics(f)["rmse"] for f in new],
                                         ens_old, ens_new)
            rows.append({"task": task, "spec": spec.name, "committed_rmse": metrics(ens_old)["rmse"],
                         "rerun_rmse": metrics(ens_new)["rmse"], "max_abs_forecast_diff": diff})
    table = pd.DataFrame(rows)
    for col in ("committed_dm_p", "rerun_dm_p", "committed_variance_rule", "rerun_variance_rule"):
        table[col] = np.nan if col.endswith("_p") else ""
    sq = lambda p: ((p["y_pred"] - p["y_true"]) ** 2).to_numpy()
    for task in tasks:
        if (task, "hybrid-S") in per_seed and (task, "hybrid-S+N") in per_seed:
            s_old, s_new, e_s_old, e_s_new = per_seed[task, "hybrid-S"]
            n_old, n_new, e_n_old, e_n_new = per_seed[task, "hybrid-S+N"]
            idx = table.index[(table["task"] == task) & (table["spec"] == "hybrid-S+N")][0]
            table.loc[idx, "committed_dm_p"] = dm_hln(sq(e_s_old), sq(e_n_old))[1]
            table.loc[idx, "rerun_dm_p"] = dm_hln(sq(e_s_new), sq(e_n_new))[1]
            table.loc[idx, "committed_variance_rule"] = str(variance_rule(s_old, n_old))
            table.loc[idx, "rerun_variance_rule"] = str(variance_rule(s_new, n_new))
    _write(table, out / "rerun_metrics.csv")
    return table


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    frames = {task: load_task(task) for task in ("T1", "T2")}
    run_diag(frames, device=args.device)
    print(rerun_metrics().round(4).to_string(index=False))


if __name__ == "__main__":
    main()
