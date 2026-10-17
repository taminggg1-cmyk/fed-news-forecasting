"""Frozen-model analysis (PREREG_H2A.md; the paper's section 4.1 fixed-rule exercise).

Every model is trained once on what was available before 2016-01-01 (or 2020-01-01) and then forecasts
every later month or meeting without updating. Compared with the same models in the main walk-forward
analysis ('updated'), this shows whether rules learned before a date still hold afterwards. 'Updated'
keeps each model's pre-registered schedule: linear models refit every month/meeting; the paper's LSTM/GRU
every 3 months (one-hour rule, DEVIATIONS.md); the TFT and the hybrid's TFT every 12 months / 8 meetings
(the hybrid's linear anchor at every row); the meeting LSTM every 8 meetings. Descriptive: not part of the
H2 Holm family.
"""
import argparse
import os
from dataclasses import replace

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from fed_news import config
from fed_news.models import linear as models
from fed_news.studies.plan2_neural_original import run_neural
from fed_news.evaluation.evaluate import metrics
from fed_news.studies.plan2_neural_original.neural_walk import (neural_ladder, prereg_h2a_ok, run_neural_spec,
                                                               standardise)
from fed_news.features.panel import N_COLS, S_COLS
from fed_news.studies.plan1_linear.run_models import load_task
from fed_news.evaluation.stats_tests import dm_hln
from fed_news.models.walkforward import prereg_ok, train_mask

STARTS = ("2016-01-01", "2020-01-01")
NEVER = 10 ** 9
OUT = config.RESULTS / "frozen"
COLUMNS = ["date", "y_true", "y_pred", "lag", "class_true", "class_pred"]


def _arrays(df):
    return (df["y_level"].to_numpy(float), df["y_change"].to_numpy(float), df["y_class"].to_numpy(int),
            df["s_rate_lag"].to_numpy(float))


def _frame(df, test, y_pred, class_pred) -> pd.DataFrame:
    y_level, _, y_class, lag = _arrays(df)
    return pd.DataFrame({"date": df.index[test], "y_true": y_level[test], "y_pred": y_pred, "lag": lag[test],
                         "class_true": y_class[test], "class_pred": class_pred})[COLUMNS]


def frozen_rw(df, test_start) -> pd.DataFrame:
    test = np.flatnonzero(df.index >= pd.Timestamp(test_start))
    lag = df["s_rate_lag"].to_numpy(float)
    return _frame(df, test, lag[test], np.ones(len(test), dtype=int))


def frozen_linear(df, cols, task, test_start) -> pd.DataFrame:
    if any(c in N_COLS for c in cols) and not (prereg_ok() and prereg_h2a_ok()):
        raise PermissionError("news columns need tags 'prereg' and 'prereg-h2a'")
    test = np.flatnonzero(df.index >= pd.Timestamp(test_start))
    train = train_mask(len(df), test[0], task)
    rows = np.flatnonzero(train)
    Z = standardise(df[cols].to_numpy(float), train)
    _, y_change, y_class, lag = _arrays(df)
    if task == "T1":
        alpha = models.choose_penalty(lambda a: Ridge(alpha=a), models.ridge_score, Z[rows], y_change[rows],
                                      config.RIDGE_GRID)
        change = Ridge(alpha=alpha).fit(Z[rows], y_change[rows]).predict(Z[test])
        cls = models.classes_from_change(change)
    else:
        lam = models.choose_penalty(models.OrderedLogit, models.logloss_score, Z[rows], y_class[rows],
                                    config.ORDLOGIT_GRID)
        proba = models.OrderedLogit(lam).fit(Z[rows], y_class[rows]).predict_proba(Z[test])
        change = proba @ models.move_sizes(y_class[rows], y_change[rows])
        cls = proba.argmax(axis=1)
    return _frame(df, test, lag[test] + change, cls)


def _write(frame, path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def _path(task, model, start, seed=None):
    suffix = "" if seed is None else f"_seed{seed}"
    return OUT / f"preds_{task}_{model}_{start[:4]}{suffix}.csv"


def run_all(device: str = "cuda", seeds=run_neural.SEEDS) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for task in ("T1", "T2"):
        df = load_task(task)
        for start in STARTS:
            _write(frozen_rw(df, start), _path(task, "rw", start))
            for label, cols in (("S", S_COLS), ("S+N", S_COLS + N_COLS)):
                _write(frozen_linear(df, cols, task, start), _path(task, f"linear-{label}", start))
            for spec in neural_ladder(task):
                for seed in seeds:
                    path = _path(task, spec.name, start, seed)
                    if not path.exists():
                        _write(run_neural_spec(df, replace(spec, refit_every=NEVER), seed, start, device=device,
                                               anchor_every_row=False), path)
                files = [_path(task, spec.name, start, s) for s in run_neural.SEEDS]
                if all(p.exists() for p in files):
                    frames = [pd.read_csv(p, parse_dates=["date"]) for p in files]
                    _write(run_neural.ensemble(frames), _path(task, spec.name, start))


def _updated(task, model, start) -> pd.DataFrame | None:
    """The same model's walk-forward forecasts, cut to the frozen test months."""
    if model.startswith("linear-"):
        path = config.RESULTS / f"preds_{task}_{model.split('-', 1)[1]}.csv"
    elif model == "rw":
        path = config.RESULTS / f"preds_{task}_rw.csv"
    else:
        path = run_neural.ensemble_path(task, model)
    if not path.exists():
        return None
    p = pd.read_csv(path, parse_dates=["date"])
    return p[p["date"] >= pd.Timestamp(start)]


def report() -> pd.DataFrame:
    rows = []
    for task in ("T1", "T2"):
        names = ["rw", "linear-S", "linear-S+N"] + [s.name for s in neural_ladder(task)]
        for start in STARTS:
            rw = pd.read_csv(_path(task, "rw", start), parse_dates=["date"])
            for model in names:
                path = _path(task, model, start)
                if not path.exists():
                    continue
                p = pd.read_csv(path, parse_dates=["date"])
                sq = lambda f: ((f["y_pred"] - f["y_true"]) ** 2).to_numpy()
                p_rw = np.nan if model == "rw" else dm_hln(sq(rw), sq(p))[1]
                row = {"task": task, "start": start[:4], "model": model, "version": "frozen",
                       **metrics(p), "p_vs_no_change": p_rw}
                rows.append(row)
                upd = _updated(task, model, start)
                if upd is not None and len(upd):
                    rows.append({"task": task, "start": start[:4], "model": model, "version": "updated",
                                 **metrics(upd), "p_vs_no_change": np.nan})
    table = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / "metrics.csv", index=False)
    lines = [
        "# Results: frozen models (train once, never update)", "",
        "Descriptive analysis from PREREG_H2A.md, modelled on the paper's fixed-rule exercise. Each model is "
        "trained once on data before 2016 (or 2020) and forecasts every later month or meeting; 'updated' is "
        "the same model in the main walk-forward analysis with its pre-registered schedule, over the same "
        "period: linear models refit every month or meeting; the paper's LSTM/GRU every 3 months (one-hour "
        "rule); the TFT and the hybrid's TFT every 12 months / 8 meetings (the hybrid's linear anchor at every "
        "row); the meeting LSTM every 8 meetings. "
        "p_vs_no_change is a one-sided DM test against the no-change forecast; it is not part of any "
        "pre-registered decision.", "",
        table.round(4).to_markdown(index=False), "",
    ]
    config.report_path("RESULTS_FROZEN.md").write_text("\n".join(lines), encoding="utf-8")
    return table


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args(argv)
    if not args.report_only:
        run_all(device=args.device)
    print(report().round(4).to_string(index=False))


if __name__ == "__main__":
    main()
