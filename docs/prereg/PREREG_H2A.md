# Pre-registration addendum: Plan 2 neural rows (H2a)

Date: 2026-10-09. Code commit: 0206ad2.
Binding: spec sections 3 (H2a), 7 (Secondary architectures), 8 (variance rule) and PREREG.md (tag `prereg`).
This file fixes the settings the spec leaves open, before any neural forecast that uses news.

## Settings
- Paper LSTM (T1): 12 steps, 240 units, 1 layer, no dropout, Adam lr 1e-4, 300 epochs, batch 32 shuffled per epoch (seeded), MSE on the standardised target, predicts the level, refit every month.
- Paper GRU (T1): 12 steps, 188 units, 1 layer, no dropout, Adam lr 2e-4, 449 epochs, batch 32 shuffled per epoch (seeded), MSE on the standardised target, predicts the level, refit every month.
- One-hour rule: `T1/lstm-S/seed0` is timed first, then (only if it took at most 3,600 s) `T1/gru-S/seed0`; if either takes longer than 3,600 s, all paper LSTM/GRU T1 specs are run with refit every 3 months instead, recorded in DEVIATIONS.md.
- Meeting LSTM (T2): lookback 4 meetings, refit every 8 meetings, otherwise as the paper LSTM.
- TFT: CompactTFT (gold_gpt2/src/tft.py, vendored unchanged), hidden 32, 4 heads, dropout 0.2, lookback 12 months / 8 meetings, Adam lr 1e-3, full batch, at most 300 epochs, early stopping on the chronologically last 20% of training rows with patience 20 (best epoch kept; the untrained weights count as epoch 0), refit every 12 months / 8 meetings, predicts the change.
- Hybrid: anchor = Plan 1 primary model (T1 ridge on the change, T2 penalised ordered logit expected change, penalties by 5-fold expanding CV), refitted at every test row as in Plan 1; TFT (refitted on its schedule) trained on out-of-fold anchor residuals (5 expanding folds inside the training window, folds whose training part has fewer than 20 rows skipped); forecast = lag + anchor change + TFT correction.
- Training rows: walkforward.train_mask (T1 rows <= t-2, T2 earlier meetings); features standardised on each refit's training rows; the model and standardisation of the last refit forecast the rows until the next refit.
- All neural targets standardised with the fit rows' mean and sd and inverted at prediction. Seeds 0-7 per spec (initialisation and batch order); ensemble = mean of the 8 seeds' forecasts; classes from the +/-0.125 band.

## Tests and decision rule (secondary; cannot change the H1 verdict)
- Per architecture (T1: lstm, gru, tft, hybrid; T2: mlstm, tft, hybrid), S+N vs S on the seed-mean ensembles: one-sided DM-HLN on squared error (T1, T2); exact one-sided McNemar on correct classes (T2).
- Variance rule: a gain counts only if mean RMSE(S) - mean RMSE(S+N) > 2 x max(sd RMSE(S), sd RMSE(S+N)) over the 8 seeds (ddof 1).
- Holm adjustment across all H2 p-values (Plan 1 results/h2_pvalues.csv plus these); the final family is fixed after Plan 3.
- Reported for every spec: per-seed mean +/- sd and ensemble RMSE, MAE, accuracy; the paper LSTM/GRU replication also on the paper window (Jan 2016 - Apr 2025) next to the paper's published values.

## Frozen-model analysis (descriptive, as the paper's section 4.1 fixed-rule exercise)
- Two cutoffs: train once on everything available before 2016-01-01 (T1: rows up to 2015-11, the t-2 rule; T2: meetings before 2016) and forecast every later month/meeting to Sep 2026 with no updating; the same with 2020-01-01.
- Models: "no change" (random walk / always-hold); the Plan 1 linear primary model with S and with S+N; every Plan 2 neural spec (T1 lstm, gru, tft, hybrid; T2 mlstm, tft, hybrid) with S and S+N, 8 seeds, seed-mean ensembles; neural settings as above except that the model is fitted once.
- Reported: RMSE, MAE, accuracy for each, next to the same model's walk-forward (updated) forecasts over the same months, and a one-sided DM p-value against "no change". These are descriptive: they are not part of the H2 Holm family and cannot change any verdict.
