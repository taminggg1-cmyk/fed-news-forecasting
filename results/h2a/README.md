# ORIGINAL neural study: Plan 2, hypothesis H2a

**Status: registered** (tag `prereg-h2a`, e931dfc; run fb78233, evaluated 1593530, 2026-10-10).
**Not** the amended Study 4 version 2 networks, which are in `results/study4/neural/` (meetings) and
`results/study4/neural_t1/` (months).

- Code: `src/fed_news/studies/plan2_neural_original/` (run_neural.py, neural_walk.py, evaluate_h2a.py).
- Comparison: statistics S against S + Plan 1 news block N (`-S` vs `-S+N` in the file names).
- Inputs: mixed vintage (revised fallback for wages, U-6 and house prices before 2011-12/2012-05).
- Monthly cutoff: the ORIGINAL first-calendar-day cutoff (panel t1), later found not to guarantee that
  FEDFUNDS(t-1) was published. Meetings: the day before the meeting.
- Paper LSTM (`lstm-*`): settings of the July 2025 preliminary AMRO paper (240 units, 1 layer, 300 epochs), not
  the September 2025 working paper (DEVIATIONS.md, 2026-10-13 and 2026-10-17). GRU (`gru-*`): 188 units, 449
  epochs. Both refit every 3 months (one-hour rule). `mlstm-*`: meeting LSTM. `tft-*`, `hybrid-*`: early stopping,
  refit every 12 months or 8 meetings.
- Files: `preds_<task>_<spec>_seed<k>.csv` (one seed), `preds_<task>_<spec>.csv` (mean of 8 seeds, the reported
  forecast), `ensemble_metrics.csv` (RMSE, MAE, accuracy), `tests.csv` (DM and McNemar p, Holm over the 15
  registered H2 tests), `run_log.json`. Report: `docs/reports/RESULTS_H2A.md`.
