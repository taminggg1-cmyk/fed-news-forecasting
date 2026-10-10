# Plan 4 design: faithful replication of the paper's LSTM/GRU and a market benchmark

Date: 2026-10-10. Status: exploratory, added after the H1 and H2a results were known. Binding settings
are copied into `PREREG_REP.md` and tagged `prereg-rep` before any run. Nothing here changes the H1
verdict (RESULTS.md) or the H2a tests (RESULTS_H2A.md).

## 1. Purpose

Two questions a referee of the paper will ask:

1. **Why is our LSTM worse than the paper's?** Our real-time LSTM scored RMSE 0.415 on the paper's window
   (Jan 2016 - Apr 2025) against the paper's 0.238 (GRU: 0.647 against 0.271). Two suspects: the data
   (the paper appears to use today's revised values indexed by observation month; we use first-release
   vintages with release lags) and the refit schedule (the paper refits monthly; our run refit every 3
   months under the one-hour rule). A third setting, a model trained once and never updated, mirrors the
   paper's own fixed-rule exercise.
2. **Does any model beat what the market expected?** Fed funds futures history is not free, so we use the
   Treasury bill yield as the market's expected policy rate.

## 2. Replication grid (task T1 only)

Design: 2 data types x 3 refit schedules, for the paper's LSTM and GRU, statistics only (no news).

| | Monthly refit | Refit every 3 months | Frozen (one fit) |
|---|---|---|---|
| Revised data | cell R1 (paper-faithful) | cell R3 | cell RF |
| Real-time data | cell V1 | cell V3 | cell VF |

Extra rows: frozen at 2020 for both data types (one fit at the first test month 2020-01, training rows up
to 2019-11; test 2020-01 to 2025-04), matching the paper's second fixed-rule cutoff.

### Shared settings

- Sample: rows from January 1998 (the paper's start), both data types.
- Target: monthly average effective federal funds rate (`FEDFUNDS`) in month t.
- Timing: fit on rows up to t-2, forecast month t from the row for t-1 (the paper's protocol, our
  `walkforward.train_mask` for T1).
- Test window: January 2016 to April 2025 (112 months). Frozen-2020 rows: January 2020 to April 2025.
- Features: the 16 S columns (`panel.S_COLS`): the paper's Table 1 without the commercial property index,
  CRB replaced by PPI, as in Plans 1-2.
- Models: the PREREG_H2A.md paper LSTM (12 steps, 240 units, lr 1e-4, 300 epochs) and GRU (12 steps,
  188 units, lr 2e-4, 449 epochs), level target, standardised inputs and targets, batch 32 shuffled per
  epoch (seeded), via `neural_walk.run_neural_spec` with `refit_every` = 1, 3 or a value larger than the
  window (frozen).
- Seeds 0-7; ensemble = mean of the 8 seeds' forecasts.

### Data types

- **Real-time (V cells):** the existing T1 panel (`t1_full`, ALFRED vintages released before each cutoff),
  restricted to rows from 1998.
- **Revised (R cells):** for every macroeconomic series, the latest vintage in the cached ALFRED files
  (`D:/fed_news/alfred/*.pkl`, downloaded 2026-10-09) for each observation date, indexed by observation
  period with no release lag: the row for month t-1 carries the values for observation month t-1
  (quarterly GDP: the value of the quarter containing month t-1). The same transformations as
  `panel.macro_at` (monthly and 12-month changes, 4-quarter GDP growth, U-6 differences). Market columns
  (lagged rate, yield slopes) are the same as the real-time panel, since market data are not revised.
  This panel is deliberately look-ahead: it is the paper's likely protocol, not ours.

### Reading rule (fixed in advance)

- For each model, report every cell's 8-seed mean RMSE +/- sd, the ensemble RMSE, MAE and MBE, next to
  the paper's published values and the no-change forecast on the same months.
- "Reproduced" means the paper's published RMSE lies within the cell's seed mean +/- 2 sd.
- Data effect = RMSE(V1) - RMSE(R1); refit effect = RMSE(V3) - RMSE(V1) and RMSE(R3) - RMSE(R1);
  updating effect = RMSE(VF) - RMSE(V1) and RMSE(RF) - RMSE(R1). Reported as differences of ensemble RMSE
  with a one-sided DM p-value for each pair (descriptive; no multiplicity claim).

## 3. Market benchmark (T-bill proxy)

- Bill yield: FRED `DTB4WK` (4-week bill, from 2001-07-31); before that date `DTB3` (3-month bill).
  Fetched once through the FRED API and cached under `D:/fed_news/market_bill.pkl` with its hash recorded.
- **T1:** forecast(t) = bill yield on the last business day of month t-1 + spread, where spread = mean over
  training months s <= t-2 of (FEDFUNDS(s) - bill on the last business day of month s-1), expanding window
  from 1994.
- **T2:** forecast(meeting) = bill yield on the last business day before the meeting + spread, where spread
  = mean over earlier meetings of (target - bill on the last business day before that meeting). Class from
  the +/-0.125 band on (forecast - standing target).
- Reported on both test windows (T1 2001-01 to 2026-09 and the paper window; T2 212 meetings) next to no
  change, linear S, linear S+N and linear S+M, with one-sided DM p-values against no change and against the
  bill (descriptive).

## 4. Code (new modules only; Plan 1-2 files unchanged)

| file | responsibility |
|---|---|
| `src/fed_news/revised_panel.py` | latest-vintage S columns by observation month; `build_revised_t1() -> DataFrame` with the same index and target columns as `t1_full` |
| `src/fed_news/market_bench.py` | fetch/cache bill yields; `bill_forecast_t1(df, bills)`, `bill_forecast_t2(df, bills)` returning the standard prediction frame |
| `src/fed_news/replication.py` | the 2x3 grid plus frozen-2020 rows, resumable per seed (atomic writes, run log), writes `results/replication/` |
| `src/fed_news/evaluate_rep.py` | tables, reading rule, DM p-values, `RESULTS_REPLICATION.md` |

Tests (offline, synthetic): the revised panel uses observation-period values with no release lag and the
latest vintage; the T1/T2 bill spread uses training rows only (perturbing later targets leaves earlier
forecasts unchanged); the frozen cell fits once; the grid writes one file per cell x model x seed; the
reading-rule arithmetic.

## 5. Compute and order

1. Bill benchmark (seconds). 2. Frozen cells (minutes). 3. Refit-every-3 cells (about 2.5 GPU hours each
data type). 4. Monthly cells (about 7.5 GPU hours each data type). Total about 20 GPU hours, background
runs with a harness waiter. GRU and LSTM of a cell run before the next cell.

## 6. Out of scope

News features in the replication (the paper's RNNs had none); the paper's LLM; buying futures data;
changing any pre-registered H1/H2a setting or result.
