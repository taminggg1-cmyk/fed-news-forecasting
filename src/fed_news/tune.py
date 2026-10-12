"""Study 4, section 5: LSTM, GRU and TFT tuned on our own data (secondary)
(docs/superpowers/specs/2026-10-12-study4-haiku-index-tuned-design.md).

At each tuning point (the first test meeting of 2001, 2005, ..., 2025) Optuna's TPE sampler searches the
settings on meetings before that point only, scored by mean validation MSE over 4 expanding time-series folds.
The chosen settings forecast every test meeting until the next point: walk-forward refits every 8 meetings
(and at each point), 8 seeds, seed-mean forecasts. Every feature set gets the same search space and budget.
"""
import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import optuna
import pandas as pd

from fed_news import config, models
from fed_news.clark_west import compare
from fed_news.evaluate import metrics
from fed_news.neural import build_model, fit_fixed, fit_validate, make_sequences, predict
from fed_news.neural_walk import standardise
from fed_news.panel import M_COLS, S_COLS
from fed_news.stats_tests import mcnemar_greater
from fed_news.walkforward import train_mask

ARCHS = ("lstm", "gru", "tft")
YEARS = (2001, 2005, 2009, 2013, 2017, 2021, 2025)
N_TRIALS, N_FOLDS, MAX_EPOCHS, PATIENCE, REFIT_EVERY, SEEDS = 30, 4, 300, 20, 8, tuple(range(8))
D = ["n_stance", "n_stance_chg"]
FEATURES = {"S+M": ("H", S_COLS + M_COLS), "S+M+H": ("H", S_COLS + M_COLS + D), "S+M+C": ("C", S_COLS + M_COLS + D),
            # design section 10: the same columns with the stance shifted circularly by half the panel
            "placebo-C": ("Cp", S_COLS + M_COLS + D), "placebo-H": ("Hp", S_COLS + M_COLS + D)}
PAIRS = [("S+M+C", "S+M"), ("S+M+H", "S+M"), ("placebo-C", "S+M"), ("placebo-H", "S+M"),
         ("S+M+C", "placebo-C"), ("S+M+H", "placebo-H")]  # (model, the model it is tested against)
OUT = config.RESULTS / "study4" / "neural"
# design section 11: the monthly task with Plan 2's inputs (the paper's Table 1 statistics) and the stance columns
FEATURES_T1 = {"S": ("C", S_COLS), "S+C": ("C", S_COLS + D), "S+H": ("H", S_COLS + D),
               "placebo-C": ("Cp", S_COLS + D), "placebo-H": ("Hp", S_COLS + D)}
PAIRS_T1 = [("S+C", "S"), ("S+H", "S"), ("placebo-C", "S"), ("placebo-H", "S"), ("S+C", "placebo-C"),
            ("S+H", "placebo-H")]


@dataclass(frozen=True)
class TaskCfg:
    name: str
    panel: str
    out: Path
    features: dict
    pairs: list
    lookbacks: tuple
    refit_every: int
    gap: int  # rows before a tuning point that are not yet known (T1: month t-1)
    base: str


TASKS = {"T2": TaskCfg("T2", "t2", OUT, FEATURES, PAIRS, (2, 4, 8, 12), REFIT_EVERY, 0, "S+M"),
         "T1": TaskCfg("T1", "t1", config.RESULTS / "study4" / "neural_t1", FEATURES_T1, PAIRS_T1, (3, 6, 12, 24),
                       12, 1, "S")}


def placebo_panel(df: pd.DataFrame, cols=D) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        out[c] = np.roll(df[c].to_numpy(), len(df) // 2)
    return out


def tuning_points(dates, years=YEARS, test_start=config.TEST_START) -> list[int]:
    dates = pd.DatetimeIndex(dates)
    out = []
    for y in years:
        start = max(pd.Timestamp(f"{y}-01-01"), pd.Timestamp(test_start))
        hits = np.flatnonzero(dates >= start)
        if len(hits) and int(hits[0]) not in out:
            out.append(int(hits[0]))
    return out


def point_for(i: int, points) -> int:
    return max(p for p in points if p <= i)


def suggest(trial, arch: str, lookbacks=(2, 4, 8, 12)) -> dict:
    p = {"hidden": trial.suggest_categorical("hidden", [8, 16, 32, 64] if arch == "tft" else [8, 16, 32, 64, 128, 256]),
         "dropout": trial.suggest_float("dropout", 0.0, 0.5),
         "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
         "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
         "lookback": trial.suggest_categorical("lookback", list(lookbacks)),
         "batch": trial.suggest_categorical("batch", [16, 32, 64])}
    if arch == "tft":
        p["heads"] = trial.suggest_categorical("heads", [1, 2, 4])
    else:
        p["layers"] = trial.suggest_categorical("layers", [1, 2])
    return p


def _model(arch: str, n_features: int, p: dict, seed: int):
    return build_model(arch, n_features, p["hidden"], p["dropout"], seed, layers=p.get("layers", 1),
                       heads=p.get("heads", 4))


def _target(y, rows):
    mu, sd = float(y[rows].mean()), float(y[rows].std())
    return mu, sd if sd > 0 else 1.0


def cv_score(X, y_change, point: int, arch: str, p: dict, device: str, max_epochs: int = MAX_EPOCHS,
             seed: int = 0, gap: int = 0) -> tuple[float, int]:
    """Mean validation MSE (standardised target) over expanding folds of rows 0..point-gap-1, and the median best epoch."""
    rows, lb = np.arange(point - gap), p["lookback"]
    vals, epochs = [], []
    for tr, va in models.expanding_folds(len(rows), N_FOLDS):
        tr, va = rows[tr][rows[tr] >= lb - 1], rows[va][rows[va] >= lb - 1]
        if len(tr) < 2 or len(va) < 1:
            continue
        mask = np.zeros(len(X), bool)
        mask[tr] = True
        Z = standardise(X, mask)
        mu, sd = _target(y_change, tr)
        model = _model(arch, X.shape[1], p, seed)
        v, e = fit_validate(model, make_sequences(Z, tr, lb), (y_change[tr] - mu) / sd, make_sequences(Z, va, lb),
                            (y_change[va] - mu) / sd, p["lr"], p["weight_decay"], p["batch"], max_epochs, PATIENCE,
                            device, seed)
        vals.append(v)
        epochs.append(e)
    if not vals:
        return np.inf, 1
    return float(np.mean(vals)), max(1, int(np.median(epochs)))


def tune_point(X, y_change, point: int, arch: str, device: str, n_trials: int = N_TRIALS, lookbacks=(2, 4, 8, 12),
               gap: int = 0) -> dict:
    def objective(trial):
        p = suggest(trial, arch, lookbacks)
        score, epochs = cv_score(X, y_change, point, arch, p, device, gap=gap)
        trial.set_user_attr("epochs", epochs)
        return score

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(objective, n_trials=n_trials)
    best = study.best_trial
    return {**best.params, "epochs": int(best.user_attrs["epochs"]), "cv_mse": float(best.value)}


def run_tuned(df: pd.DataFrame, cols, arch: str, settings: dict, seed: int, device: str, test_start,
              on_refit=None, task: str = "T2", refit_every: int = REFIT_EVERY) -> pd.DataFrame:
    X = df[cols].to_numpy(float)
    y_change, lag, y_level = (df[c].to_numpy(float) for c in ("y_change", "s_rate_lag", "y_level"))
    y_class, dates, n = df["y_class"].to_numpy(int), df.index, len(df)
    test, points = np.flatnonzero(dates >= pd.Timestamp(test_start)), sorted(settings)
    out, model, current = [], None, None
    for k, i in enumerate(test):
        pt = point_for(i, points)
        if k % refit_every == 0 or pt != current:
            current, p = pt, settings[pt]
            train = train_mask(n, i, task)
            Z = standardise(X, train)
            rows = np.flatnonzero(train)
            rows = rows[rows >= p["lookback"] - 1]
            mu, sd = _target(y_change, rows)
            model = _model(arch, len(cols), p, seed)
            fit_fixed(model, make_sequences(Z, rows, p["lookback"]), (y_change[rows] - mu) / sd, p["lr"],
                      p["epochs"], device, batch_size=p["batch"], seed=seed, weight_decay=p["weight_decay"])
            if on_refit:
                on_refit(i, p)
        change = float(predict(model, make_sequences(Z, np.array([i]), p["lookback"]), device)[0]) * sd + mu
        if not np.isfinite(change):
            raise ValueError(f"{arch}: non-finite forecast at {dates[i].date()}")
        out.append({"date": dates[i], "y_true": y_level[i], "y_pred": lag[i] + change, "lag": lag[i],
                    "class_true": int(y_class[i]),
                    "class_pred": int(models.classes_from_change(np.array([change]))[0])})
    return pd.DataFrame(out)


def _settings_path(arch, fs, out=OUT):
    return out / f"settings_{arch}_{fs}.json"


def tune_all(pan: dict, device: str, n_trials: int = N_TRIALS, archs=ARCHS, features=None,
             cfg: TaskCfg = TASKS["T2"]) -> None:
    cfg.out.mkdir(parents=True, exist_ok=True)
    for arch in archs:
        for fs in features or tuple(cfg.features):
            which, cols = cfg.features[fs]
            df = pan[which]
            path = _settings_path(arch, fs, cfg.out)
            done = {int(k): v for k, v in json.loads(path.read_text()).items()} if path.exists() else {}
            X, y = df[cols].to_numpy(float), df["y_change"].to_numpy(float)
            for pt in tuning_points(df.index):
                if pt in done:
                    continue
                t0 = time.time()
                done[pt] = (tune_point(X, y, pt, arch, device, n_trials, cfg.lookbacks, cfg.gap)
                            | {"date": str(df.index[pt].date())})
                path.write_text(json.dumps(done, indent=2))
                print(f"{arch} {fs} {df.index[pt].date()}: {done[pt]} ({time.time() - t0:.0f}s)", flush=True)


def time_trial(pan: dict, device: str) -> dict:
    """Seconds for one trial per architecture at the last tuning point (the most rows), and the projection."""
    df = pan["C"]  # timing only: the ChronoBERT panel exists before the Haiku index does
    X, y = df[S_COLS + M_COLS + D].to_numpy(float), df["y_change"].to_numpy(float)
    pts = tuning_points(df.index)
    secs = {}
    for arch in ARCHS:
        t0 = time.time()
        tune_point(X, y, pts[-1], arch, device, n_trials=3)
        secs[arch] = (time.time() - t0) / 3
    # upper bound: every point costs as much as the last one
    hours = sum(secs.values()) * N_TRIALS * len(pts) * len(FEATURES) / 3600
    res = {"seconds_per_trial_at_last_point": secs, "projected_hours_upper_bound": hours, "n_points": len(pts)}
    print(json.dumps(res, indent=2), flush=True)
    return res


def forecast_all(pan: dict, device: str, archs=ARCHS, cfg: TaskCfg = TASKS["T2"]) -> pd.DataFrame:
    rows, comps, preds = [], [], {}
    out = cfg.out
    for arch in archs:
        for fs, (which, cols) in cfg.features.items():
            if not _settings_path(arch, fs, out).exists():
                continue
            df = pan[which]
            settings = {int(k): v for k, v in json.loads(_settings_path(arch, fs, out).read_text()).items()}
            frames = []
            for seed in SEEDS:
                path = out / f"preds_{arch}_{fs}_seed{seed}.csv"
                if not path.exists():
                    run_tuned(df, cols, arch, settings, seed, device, config.TEST_START, task=cfg.name,
                              refit_every=cfg.refit_every).to_csv(path, index=False)
                frames.append(pd.read_csv(path, parse_dates=["date"]))
            ens = frames[0][["date", "y_true", "lag", "class_true"]].copy()
            ens["y_pred"] = np.mean([f["y_pred"].to_numpy() for f in frames], axis=0)
            ens["class_pred"] = models.classes_from_change(ens["y_pred"].to_numpy() - ens["lag"].to_numpy())
            ens.to_csv(out / f"preds_{arch}_{fs}.csv", index=False)
            preds[(arch, fs)] = ens
            rows.append({"arch": arch, "features": fs, **metrics(ens)})
        for new, base in cfg.pairs:
            if (arch, new) in preds and (arch, base) in preds:
                b, p = preds[(arch, base)], preds[(arch, new)]
                fixes, breaks, p_mc = mcnemar_greater(b["class_pred"] == b["class_true"], p["class_pred"] == p["class_true"])
                comps.append({"arch": arch, "model": new, "against": base, **compare(b, p),
                              "mcnemar_fixes": fixes, "mcnemar_breaks": breaks, "mcnemar_p": p_mc})
    table, comp = pd.DataFrame(rows), pd.DataFrame(comps)
    if cfg.name == "T1":
        comp = pd.concat([comp, paper_settings_rows(preds, archs)], ignore_index=True)
    table.to_csv(out / f"metrics_neural_{cfg.name}.csv", index=False)
    comp.to_csv(out / f"comparison_neural_{cfg.name}.csv", index=False)
    pd.set_option("display.width", 200)
    print(table.round(4).to_string(index=False))
    print(comp.round(4).to_string(index=False), flush=True)
    return comp


def paper_settings_rows(preds: dict, archs) -> pd.DataFrame:
    """Section 11 (a): tuned S against Plan 2's paper-settings S forecasts, all test months and the paper window."""
    rows = []
    for arch in archs:
        if (arch, "S") not in preds:
            continue
        old = pd.read_csv(config.RESULTS / "h2a" / f"preds_T1_{arch}-S.csv", parse_dates=["date"])
        new = preds[(arch, "S")]
        for window, (lo, hi) in (("all test months", (config.TEST_START, "2100-01-01")),
                                 ("paper window", config.PAPER_T1)):
            o = old[(old["date"] >= lo) & (old["date"] <= hi)]
            n = new[new["date"].isin(o["date"])].sort_values("date").reset_index(drop=True)
            o = o[o["date"].isin(n["date"])].sort_values("date").reset_index(drop=True)
            fixes, breaks, p_mc = mcnemar_greater(o["class_pred"] == o["class_true"],
                                                  n["class_pred"] == n["class_true"])
            rows.append({"arch": arch, "model": f"S tuned ({window})", "against": "S paper settings",
                         "rmse_against": metrics(o)["rmse"], "rmse_model": metrics(n)["rmse"],
                         "accuracy_against": metrics(o)["accuracy"], "accuracy_model": metrics(n)["accuracy"],
                         **compare(o, n), "mcnemar_fixes": fixes, "mcnemar_breaks": breaks, "mcnemar_p": p_mc})
    return pd.DataFrame(rows)


def panels_for(cfg: TaskCfg) -> dict:
    if cfg.name == "T2":
        from fed_news.haiku_compare import panels
        pan = panels()
    else:
        from fed_news.haiku_compare import same_rows
        from fed_news.haiku_index import OUT_PATH as HAIKU_PATH
        from fed_news.naive_stance import naive_panel
        from fed_news.walkforward import common_rows
        c = common_rows(naive_panel(cfg.panel, pd.read_pickle(config.DATA / "doc_stance.pkl")))
        h = common_rows(naive_panel(cfg.panel, pd.read_pickle(HAIKU_PATH)))
        c, h = same_rows(c, h)
        pan = {"C": c, "H": h}
    pan["Cp"], pan["Hp"] = placebo_panel(pan["C"]), placebo_panel(pan["H"])
    return pan


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["time", "tune", "forecast"])
    ap.add_argument("--trials", type=int, default=N_TRIALS)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--archs", nargs="+", default=list(ARCHS), choices=ARCHS)
    ap.add_argument("--features", nargs="+", default=None, choices=sorted(set(FEATURES) | set(FEATURES_T1)))
    ap.add_argument("--task", default="T2", choices=sorted(TASKS))
    args = ap.parse_args(argv)
    if args.step == "time":
        from fed_news.naive_stance import naive_panel
        from fed_news.walkforward import common_rows
        time_trial({"C": common_rows(naive_panel("t2", pd.read_pickle(config.DATA / "doc_stance.pkl")))}, args.device)
        return
    cfg = TASKS[args.task]
    pan = panels_for(cfg)
    if args.step == "tune":
        tune_all(pan, args.device, args.trials, args.archs, args.features, cfg)
    else:
        forecast_all(pan, args.device, args.archs, cfg)


if __name__ == "__main__":
    main()
