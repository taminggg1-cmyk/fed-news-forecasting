# Results: Study 5 (three-month policy path, real-time inputs) - exploratory

Design: `docs/superpowers/specs/2026-10-12-study5-three-month-horizon-design.md` (committed, revised twice
after reviews, all before any code ran on outcomes). Run once; no bug found after the run.

**Verdict: not supported.** No detectable improvement at three months in this data, with this reader and
horizon. Gains of 0.02 or more in mean log loss are ruled out by the stated threshold (upper bound of the 90%
interval 0.018).

## Setup

- One row per scheduled meeting, forecast the day before its decision. Target: net change in the federal funds
  target over the next h scheduled meetings (unscheduled moves included), cut / hold / hike at +/-0.125.
- Inputs real-time by construction: economic series with complete ALFRED vintages only (M2, payrolls, GDP,
  CPI; checked in code), Treasury-market prices, and Fed communication stance by release date. No SF Fed news
  sentiment, GPR or EPU, and no series that needed the ALFRED fallback.
- Training rows only once their whole horizon was decided before the forecast's cutoff; penalty CV purged of
  overlapping horizons.

| horizon | test rows | first - last | cuts / holds / hikes | days to last decision (min, median, max) |
|---|---:|---|---|---|
| h = 3 (primary) | 204 | 2001-01-31 - 2026-06-17 | 44 / 105 / 55 | 71, 92, 107 |
| h = 2 | 205 | 2001-01-31 - 2026-07-29 | 36 / 119 / 50 | 35, 44, 59 |
| h = 1 (reference) | 206 | 2001-01-31 - 2026-09-16 | 28 / 140 / 38 | 1, 1, 1 |

## Results

d = log loss of the no-news model B minus the news model (positive means news helps); one-sided p from a
Newey-West test with 8 lags (8 scheduled meetings, about one year).

| h | model | accuracy | log loss | Brier | mean d | p |
|---|---|---:|---:|---:|---:|---:|
| 3 | B (no news) | 0.716 | 0.779 | 0.441 | | |
| 3 | **B+C (ChronoBERT, primary)** | 0.691 | 0.839 | 0.466 | **-0.060** | **0.898** |
| 3 | placebo-C | 0.711 | 0.810 | 0.448 | -0.031 | 0.888 |
| 3 | shuffled-C | 0.461 | 1.065 | 0.642 | -0.287 | 0.998 |
| 3 | B+H (Haiku) | 0.696 | 0.769 | 0.441 | 0.010 | 0.300 |
| 3 | B+C+Beige | 0.711 | 0.790 | 0.448 | -0.012 | 0.696 |
| 2 | B | 0.805 | 0.520 | 0.307 | | |
| 2 | B+C | 0.800 | 0.515 | 0.307 | 0.005 | 0.323 |
| 2 | B+H | 0.820 | 0.482 | 0.287 | 0.039 | 0.014 |
| 1 | B | 0.879 | 0.315 | 0.184 | | |
| 1 | B+C | 0.869 | 0.328 | 0.188 | -0.013 | 0.782 |

- Primary: mean d -0.060, 90% interval [-0.138, 0.018]; the ChronoBERT stance made three-month forecasts
  worse on average. Robustness: HLN-DM (h = 3) p 0.919; block bootstrap (8 rows) 90% interval
  [-0.140, 0.004].
- Controls do not void the comparison (placebo p 0.888, shuffled p 0.998).
- Secondary, Holm-adjusted across four: B+H at h = 2 p 0.014 raw, 0.054 adjusted; the others 0.90. Haiku is
  the one reader whose memory of decisions could produce exactly this pattern, so this is not evidence that
  public text helps.
- Correct forecasts at h = 3, B against B+C: cuts 20 against 16, holds 89 against 88, hikes 37 against 37,
  turning points 10 against 10, all 146 against 141.

## Reading

Across one-meeting and three-month horizons, with real-time inputs and a leak-free reader, Fed communication
adds nothing detectable to a model with economic data and Treasury yields; at three months gains of 0.02 or
more in log loss are ruled out by the stated threshold. This is a statement about a baseline with Treasury
yields, not about fed funds futures, and about the 2001-2026 sample already used by earlier studies.
