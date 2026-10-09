"""Neural forecasters for Plan 2 (spec section 7, secondary architectures).

The paper's LSTM/GRU and the meeting LSTM train full-batch Adam on MSE for a fixed number of
epochs; the TFT (vendored in tft.py) trains with early stopping on the chronologically last part
of its training rows and keeps the best epoch's weights.
"""
import copy

import numpy as np
import torch
import torch.nn as nn

from fed_news.tft import CompactTFT


class RNNRegressor(nn.Module):
    """One recurrent layer (LSTM or GRU) and a linear head on the last time step."""

    def __init__(self, n_features: int, hidden: int, cell: str):
        super().__init__()
        rnn = nn.LSTM if cell == "lstm" else nn.GRU
        self.rnn = rnn(n_features, hidden, num_layers=1, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.head(out[:, -1, :]).squeeze(-1)


def build_model(arch: str, n_features: int, hidden: int, dropout: float, seed: int) -> nn.Module:
    torch.manual_seed(seed)
    if arch in ("lstm", "gru"):
        return RNNRegressor(n_features, hidden, arch)
    if arch == "tft":
        return CompactTFT(n_features, hidden=hidden, n_heads=4, dropout=dropout)
    raise ValueError(f"unknown arch {arch}")


def make_sequences(Z: np.ndarray, rows, lookback: int) -> np.ndarray:
    rows = np.asarray(rows, dtype=int)
    if len(rows) == 0:
        return np.empty((0, lookback, Z.shape[1]), dtype=np.float32)
    if rows.min() < lookback - 1:
        raise ValueError(f"row {rows.min()} has less than {lookback} rows of history")
    return np.stack([Z[j - lookback + 1: j + 1] for j in rows]).astype(np.float32)


def _tensor(a, device: str) -> torch.Tensor:
    return torch.tensor(np.asarray(a), dtype=torch.float32, device=device)


def fit_fixed(model, X, y, lr: float, epochs: int, device: str):
    model.to(device).train()
    xt, yt = _tensor(X, device), _tensor(y, device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        loss = nn.functional.mse_loss(model(xt), yt)
        loss.backward()
        opt.step()
    return model


def fit_early_stopping(model, X, y, lr: float, max_epochs: int, patience: int, val_frac: float, device: str):
    n = len(y)
    n_val = max(1, int(round(n * val_frac)))
    if n - n_val < 1:
        raise ValueError("too few rows for an early-stopping split")
    model.to(device)
    xt, yt = _tensor(X[: n - n_val], device), _tensor(y[: n - n_val], device)
    xv, yv = _tensor(X[n - n_val:], device), _tensor(y[n - n_val:], device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    def val_loss() -> float:
        model.eval()
        with torch.no_grad():
            v = float(nn.functional.mse_loss(model(xv), yv))
        if not np.isfinite(v):
            raise ValueError("non-finite validation loss")
        return v

    # The untrained weights count as epoch 0, so the result is never worse than the start.
    best, best_state, wait = val_loss(), copy.deepcopy(model.state_dict()), 0
    for _ in range(max_epochs):
        model.train()
        opt.zero_grad()
        loss = nn.functional.mse_loss(model(xt), yt)
        loss.backward()
        opt.step()
        v = val_loss()
        if v < best - 1e-12:
            best, best_state, wait = v, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    return model


def predict(model, X, device: str) -> np.ndarray:
    model.to(device).eval()
    with torch.no_grad():
        return model(_tensor(X, device)).float().cpu().numpy()
