"""Run the Plan 2 neural ladder: every spec x 8 seeds (resumable), then seed-mean ensembles.

Per-seed predictions: results/h2a/preds_<task>_<name>_seed<k>.csv; ensembles (mean of the 8 seeds'
forecasts, class from the +/-0.125 band): results/h2a/preds_<task>_<name>.csv; timings and refit
schedule: results/h2a/run_log.json (the one-hour rule for the paper's LSTM/GRU is read from it).
"""
import argparse
import json
import time

import numpy as np
import pandas as pd

from fed_news import config, models
from fed_news.neural_walk import neural_ladder, run_neural_spec
from fed_news.run_models import load_task

OUT = config.RESULTS / "h2a"
SEEDS = tuple(range(8))
ONE_HOUR = 3600.0


def seed_path(task: str, name: str, seed: int):
    return OUT / f"preds_{task}_{name}_seed{seed}.csv"


def ensemble_path(task: str, name: str):
    return OUT / f"preds_{task}_{name}.csv"


def ensemble(frames: list[pd.DataFrame]) -> pd.DataFrame:
    base = frames[0][["date", "y_true", "lag", "class_true"]].copy()
    for f in frames[1:]:
        if len(f) != len(base) or not (f["date"].to_numpy() == base["date"].to_numpy()).all():
            raise ValueError("seed files cover different dates")
    base["y_pred"] = np.mean([f["y_pred"].to_numpy(float) for f in frames], axis=0)
    base["class_pred"] = models.classes_from_change(base["y_pred"].to_numpy() - base["lag"].to_numpy(float))
    return base[["date", "y_true", "y_pred", "lag", "class_true", "class_pred"]]


def too_slow(seconds: float) -> bool:
    return seconds > ONE_HOUR


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["T1", "T2", "all"], default="all")
    ap.add_argument("--only", default="")
    ap.add_argument("--seeds", default="0-7")
    ap.add_argument("--paper-refit-every", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    lo, hi = (int(x) for x in args.seeds.split("-"))
    only = {s for s in args.only.split(",") if s}
    OUT.mkdir(parents=True, exist_ok=True)
    log_path = OUT / "run_log.json"
    log = json.loads(log_path.read_text()) if log_path.exists() else {}
    for task in (["T1", "T2"] if args.task == "all" else [args.task]):
        df = load_task(task)
        for spec in neural_ladder(task, args.paper_refit_every):
            if only and spec.name not in only:
                continue
            for seed in range(lo, hi + 1):
                path = seed_path(task, spec.name, seed)
                if path.exists():
                    continue
                t0 = time.time()
                preds = run_neural_spec(df, spec, seed, config.TEST_START, device=args.device)
                seconds = time.time() - t0
                preds.to_csv(path, index=False)
                log[f"{task}/{spec.name}/seed{seed}"] = {"seconds": round(seconds, 1), "refit_every": spec.refit_every}
                log_path.write_text(json.dumps(log, indent=1))
                flag = "  SLOWER THAN ONE HOUR" if too_slow(seconds) else ""
                print(f"{task} {spec.name} seed {seed}: {len(preds)} forecasts, {seconds:.0f}s{flag}", flush=True)
            files = [seed_path(task, spec.name, s) for s in SEEDS]
            if all(p.exists() for p in files):
                frames = [pd.read_csv(p, parse_dates=["date"]) for p in files]
                ensemble(frames).to_csv(ensemble_path(task, spec.name), index=False)


if __name__ == "__main__":
    main()
