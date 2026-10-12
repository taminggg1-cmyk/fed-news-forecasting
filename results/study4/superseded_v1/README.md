# Superseded: Study 4 network results, version 1 (stopped 2026-10-13, about 00:35)

Kept as exploratory, superseded results. An external review of the methods note found problems that affect
both tasks: validation folds that were not identical across candidate lookbacks, validation loss measured in
fold-specific standardised units instead of rate units, no t-2 gap inside the monthly folds, a monthly cutoff
(first calendar day) at which the previous month's average rate was not guaranteed to be published, and
Clark-West tests applied to non-nested, separately tuned networks. The corrected protocol is section 12 of
docs/superpowers/specs/2026-10-12-study4-haiku-index-tuned-design.md.

Contents: `neural/` (per-meeting task: complete tuning and forecasts for LSTM and GRU; TFT tuning complete
for the real-news arms, TFT placebo tuning partial; LSTM and GRU placebo tuning complete, no placebo
forecasts), `neural_t1/` (monthly task: LSTM tuning partial when stopped), `logs/` (run logs).
