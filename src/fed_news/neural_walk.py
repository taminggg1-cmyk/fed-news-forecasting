"""Walk-forward for the neural rows of the ladder (spec section 7, secondary architectures).

Same rows, cutoffs and training masks as the primary engine (walkforward.train_mask): T1 trains on
rows up to t-2, T2 on all earlier meetings. Features are standardised on each refit's training rows.
The paper's LSTM/GRU and the meeting LSTM predict the level; the TFT predicts the change; the hybrid
adds a TFT correction, trained on out-of-fold residuals, to the primary linear model's change.
Between refits the last model (and its standardisation) forecasts the following rows;
the hybrid's linear anchor is refit at every row like the primary model. Targets are standardised on the fit rows.
"""
import subprocess
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from fed_news import config, models
from fed_news.neural import build_model, fit_early_stopping, fit_fixed, make_sequences, predict
from fed_news.panel import N_COLS, S_COLS
from fed_news.walkforward import prereg_ok, train_mask

H2A_TAG = "prereg-h2a"
TFT_HIDDEN, TFT_DROPOUT, TFT_LR, TFT_MAX_EPOCHS, TFT_PATIENCE, TFT_VAL_FRAC = 32, 0.2, 1e-3, 300, 20, 0.2
MIN_ANCHOR_ROWS = 20
TARGET_COLS = ["y_level", "y_change", "y_class", "s_rate_lag"]


@dataclass
class NeuralSpec:
    name: str
    task: str
    arch: str
    cols: list
    lookback: int
    target: str
    hidden: int
    lr: float
    epochs: int
    refit_every: int
    early_stop: bool = False
    dropout: float = 0.0
    anchor: bool = False


def neural_ladder(task: str, paper_refit_every: int = 1) -> list[NeuralSpec]:
    out = []
    for label, cols in (("S", S_COLS), ("S+N", S_COLS + N_COLS)):
        if task == "T1":
            out += [
                NeuralSpec(f"lstm-{label}", "T1", "lstm", cols, 12, "level", 240, 1e-4, 300, paper_refit_every),
                NeuralSpec(f"gru-{label}", "T1", "gru", cols, 12, "level", 188, 2e-4, 449, paper_refit_every),
                NeuralSpec(f"tft-{label}", "T1", "tft", cols, 12, "change", TFT_HIDDEN, TFT_LR, TFT_MAX_EPOCHS, 12,
                           True, TFT_DROPOUT),
                NeuralSpec(f"hybrid-{label}", "T1", "tft", cols, 12, "change", TFT_HIDDEN, TFT_LR, TFT_MAX_EPOCHS, 12,
                           True, TFT_DROPOUT, True),
            ]
        else:
            out += [
                NeuralSpec(f"mlstm-{label}", "T2", "lstm", cols, 4, "level", 240, 1e-4, 300, 8),
                NeuralSpec(f"tft-{label}", "T2", "tft", cols, 8, "change", TFT_HIDDEN, TFT_LR, TFT_MAX_EPOCHS, 8,
                           True, TFT_DROPOUT),
                NeuralSpec(f"hybrid-{label}", "T2", "tft", cols, 8, "change", TFT_HIDDEN, TFT_LR, TFT_MAX_EPOCHS, 8,
                           True, TFT_DROPOUT, True),
            ]
    return out


def prereg_h2a_ok(repo=config.REPO) -> bool:
    out = subprocess.run(["git", "-C", str(repo), "tag", "-l", H2A_TAG], capture_output=True, text=True)
    return out.stdout.strip() == H2A_TAG


def standardise(X, train) -> np.ndarray:
    mu, sd = X[train].mean(0), X[train].std(0)
    return (X - mu) / np.where(sd > 0, sd, 1.0)


def _anchor(Z, y_change, y_class, rows, task):
    """Fit the primary linear model on `rows`; return a function giving its forecast of the change."""
    rows = np.asarray(rows)
    if task == "T1":
        alpha = models.choose_penalty(lambda a: Ridge(alpha=a), models.ridge_score, Z[rows], y_change[rows],
                                      config.RIDGE_GRID)
        fitted = Ridge(alpha=alpha).fit(Z[rows], y_change[rows])
        return lambda idx: fitted.predict(Z[idx])
    lam = models.choose_penalty(models.OrderedLogit, models.logloss_score, Z[rows], y_class[rows],
                                config.ORDLOGIT_GRID)
    fitted = models.OrderedLogit(lam).fit(Z[rows], y_class[rows])
    sizes = models.move_sizes(y_class[rows], y_change[rows])
    return lambda idx: fitted.predict_proba(Z[idx]) @ sizes


def oof_residuals(Z, y_change, y_class, rows, task) -> tuple[np.ndarray, np.ndarray]:
    """y_change minus the anchor's forecast, each row's anchor fit only on strictly earlier training rows."""
    rows = np.asarray(rows)
    res_rows, res = [], []
    for tr, va in models.expanding_folds(len(rows)):
        if len(tr) < MIN_ANCHOR_ROWS:
            continue
        f = _anchor(Z, y_change, y_class, rows[tr], task)
        res_rows.append(rows[va])
        res.append(y_change[rows[va]] - f(rows[va]))
    if not res_rows:
        return np.array([], dtype=int), np.array([], dtype=float)
    return np.concatenate(res_rows), np.concatenate(res)


def _check_inputs(df: pd.DataFrame, spec: NeuralSpec) -> None:
    if not df.index.is_monotonic_increasing or not df.index.is_unique:
        raise ValueError("rows must be in chronological order with a unique index")
    values = df[list(dict.fromkeys(spec.cols + TARGET_COLS))].to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError("non-finite value in features or targets")


def run_neural_spec(df: pd.DataFrame, spec: NeuralSpec, seed: int, test_start, test_end=None, device: str = "cuda",
                    require_prereg: bool = True, epochs: int | None = None) -> pd.DataFrame:
    uses_text = any(c in N_COLS for c in spec.cols)
    if require_prereg and uses_text and not (prereg_ok() and prereg_h2a_ok()):
        raise PermissionError(f"{spec.name} uses news columns; tags 'prereg' and '{H2A_TAG}' must exist first")
    _check_inputs(df, spec)
    dates, n = df.index, len(df)
    in_window = dates >= pd.Timestamp(test_start)
    if test_end is not None:
        in_window &= dates <= pd.Timestamp(test_end)
    test = np.flatnonzero(in_window)
    X = df[spec.cols].to_numpy(float)
    y_level, y_change, lag = (df[c].to_numpy(float) for c in ("y_level", "y_change", "s_rate_lag"))
    y_class = df["y_class"].to_numpy(int)
    n_epochs = spec.epochs if epochs is None else epochs
    cols_out = ["date", "y_true", "y_pred", "lag", "class_true", "class_pred"]
    out, model, Z, t_mu, t_sd = [], None, None, 0.0, 1.0
    for k, i in enumerate(test):
        if k % spec.refit_every == 0:
            train = train_mask(n, i, spec.task)
            Z = standardise(X, train)
            rows = np.flatnonzero(train)
            if spec.anchor:
                fit_rows, target = oof_residuals(Z, y_change, y_class, rows, spec.task)
            else:
                fit_rows = rows
                target = (y_level if spec.target == "level" else y_change)[rows]
            keep = fit_rows >= spec.lookback - 1
            fit_rows, target = fit_rows[keep], target[keep]
            if len(fit_rows) < 2:
                raise ValueError(f"{spec.name}: fewer than 2 training sequences at {dates[i].date()}")
            t_mu, t_sd = float(target.mean()), float(target.std())
            t_sd = t_sd if t_sd > 0 else 1.0
            target = (target - t_mu) / t_sd
            Xs = make_sequences(Z, fit_rows, spec.lookback)
            model = build_model(spec.arch, len(spec.cols), spec.hidden, spec.dropout, seed)
            if spec.early_stop:
                fit_early_stopping(model, Xs, target, spec.lr, n_epochs, TFT_PATIENCE, TFT_VAL_FRAC, device)
            else:
                fit_fixed(model, Xs, target, spec.lr, n_epochs, device, seed=seed)
        raw = float(predict(model, make_sequences(Z, np.array([i]), spec.lookback), device)[0]) * t_sd + t_mu
        if spec.target == "level":
            change = raw - lag[i]
        else:
            change = raw
            if spec.anchor:
                # The anchor refits at every row, exactly as the primary linear model does.
                train_i = train_mask(n, i, spec.task)
                Z_i = standardise(X, train_i)
                change += float(_anchor(Z_i, y_change, y_class, np.flatnonzero(train_i), spec.task)(np.array([i]))[0])
        if not np.isfinite(change):
            raise ValueError(f"{spec.name}: non-finite forecast at {dates[i].date()}")
        out.append({"date": dates[i], "y_true": y_level[i], "y_pred": lag[i] + change, "lag": lag[i],
                    "class_true": int(y_class[i]),
                    "class_pred": int(models.classes_from_change(np.array([change]))[0])})
    return pd.DataFrame(out, columns=cols_out)
