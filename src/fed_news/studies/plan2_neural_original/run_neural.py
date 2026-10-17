"""Run the Plan 2 neural ladder: every spec x 8 seeds (resumable), then seed-mean ensembles.

Per-seed predictions: results/h2a/preds_<task>_<name>_seed<k>.csv; ensembles (mean of the 8 seeds'
forecasts, class from the +/-0.125 band): results/h2a/preds_<task>_<name>.csv; timings and refit
schedule: results/h2a/run_log.json (the one-hour rule for the paper's LSTM/GRU is read from it).
A seed counts as done only if its CSV is complete and finite and run_log.json has a matching entry
(written after the CSV, so the log entry is the commit marker).
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from fed_news import config
from fed_news.models import linear as models
from fed_news.studies.plan2_neural_original.neural_walk import neural_ladder, run_neural_spec
from fed_news.studies.plan1_linear.run_models import load_task

OUT = config.RESULTS / "h2a"
SEEDS = tuple(range(8))
ONE_HOUR = 3600.0


def seed_path(task: str, name: str, seed: int):
    return OUT / f"preds_{task}_{name}_seed{seed}.csv"


def ensemble_path(task: str, name: str):
    return OUT / f"preds_{task}_{name}.csv"


def ensemble(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        raise ValueError("no seed frames to ensemble")
    base = frames[0][["date", "y_true", "lag", "class_true"]].copy()
    for f in frames:
        if len(f) != len(base) or not (f["date"].to_numpy() == base["date"].to_numpy()).all():
            raise ValueError("seed files cover different dates")
        for col in ("y_true", "lag", "class_true"):
            if not (f[col].to_numpy() == base[col].to_numpy()).all():
                raise ValueError(f"seed files disagree on {col}")
        if not np.isfinite(f["y_pred"].to_numpy(float)).all():
            raise ValueError("non-finite y_pred in a seed file")
    base["y_pred"] = np.mean([f["y_pred"].to_numpy(float) for f in frames], axis=0)
    base["class_pred"] = models.classes_from_change(base["y_pred"].to_numpy() - base["lag"].to_numpy(float))
    return base[["date", "y_true", "y_pred", "lag", "class_true", "class_pred"]]


def too_slow(seconds: float) -> bool:
    return seconds > ONE_HOUR


def read_log() -> dict:
    p = OUT / "run_log.json"
    return json.loads(p.read_text()) if p.exists() else {}


def seed_done(task: str, spec, seed: int, dates) -> str:
    """Return "" if this seed's CSV and log entry are valid, else the reason it must be recomputed."""
    path = seed_path(task, spec.name, seed)
    if not path.exists():
        return "no CSV"
    entry = read_log().get(f"{task}/{spec.name}/seed{seed}")
    if entry is None:
        return "no run_log entry"
    if entry.get("refit_every") != spec.refit_every:
        return f"refit_every {entry.get('refit_every')} != {spec.refit_every}"
    expected = dates[dates >= pd.Timestamp(config.TEST_START)]
    try:
        f = pd.read_csv(path, parse_dates=["date"])
        n = len(f)
        if n != len(expected):
            return f"{n} rows, expected {len(expected)}"
        if not (f["date"].to_numpy() == expected.to_numpy()).all():
            return "dates differ from the test window"
        if not np.isfinite(f["y_pred"].to_numpy(float)).all():
            return "non-finite y_pred"
    except Exception as e:
        return f"unreadable CSV ({e})"
    return ""


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["T1", "T2", "all"], default="all")
    ap.add_argument("--only", default="")
    ap.add_argument("--seeds", default="0-7")
    # Binding value is 3 (PREREG_H2A.md; DEVIATIONS.md, one-hour rule): a plain rerun keeps the committed seeds.
    ap.add_argument("--paper-refit-every", type=int, default=3)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    lo_s, _, hi_s = args.seeds.partition("-")
    lo = int(lo_s)
    hi = int(hi_s) if hi_s else lo
    only = {s for s in args.only.split(",") if s}
    OUT.mkdir(parents=True, exist_ok=True)
    log_path = OUT / "run_log.json"
    for task in (["T1", "T2"] if args.task == "all" else [args.task]):
        df = load_task(task)
        for spec in neural_ladder(task, args.paper_refit_every):
            if only and spec.name not in only:
                continue
            for seed in range(lo, hi + 1):
                path = seed_path(task, spec.name, seed)
                reason = seed_done(task, spec, seed, df.index)
                if not reason:
                    continue
                if path.exists():
                    print(f"recomputing {task}/{spec.name}/seed{seed}: {reason}", flush=True)
                t0 = time.time()
                preds = run_neural_spec(df, spec, seed, config.TEST_START, device=args.device)
                seconds = time.time() - t0
                tmp = path.with_name(path.name + ".tmp")
                preds.to_csv(tmp, index=False)
                os.replace(tmp, path)
                log = read_log()
                log[f"{task}/{spec.name}/seed{seed}"] = {"seconds": round(seconds, 1), "refit_every": spec.refit_every}
                tmp = log_path.with_name(log_path.name + ".tmp")
                tmp.write_text(json.dumps(log, indent=1))
                os.replace(tmp, log_path)
                flag = "  SLOWER THAN ONE HOUR" if too_slow(seconds) else ""
                print(f"{task} {spec.name} seed {seed}: {len(preds)} forecasts, {seconds:.0f}s{flag}", flush=True)
            waiting = [s for s in SEEDS if seed_done(task, spec, s, df.index)]
            if waiting:
                print(f"ensemble {task}/{spec.name}: waiting for seeds {waiting}", flush=True)
                continue
            frames = [pd.read_csv(seed_path(task, spec.name, s), parse_dates=["date"]) for s in SEEDS]
            ens = ensemble_path(task, spec.name)
            tmp = ens.with_name(ens.name + ".tmp")
            ensemble(frames).to_csv(tmp, index=False)
            os.replace(tmp, ens)


if __name__ == "__main__":
    main()
