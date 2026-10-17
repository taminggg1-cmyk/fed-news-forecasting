### Deviations from PREREG.md

- None in the 17 Plan 1 modules: `git diff --stat prereg -- <Plan 1 files>` is empty; Plan 2 added new modules only. The H1 run used the tagged code (HEAD 8c8d3c2).

### Errata

- PREREG.md line 11 states the median held-out stance accuracy as 0.661; the correct value from results/stance_accuracy.csv is 0.655 (mean 0.644). This does not affect H1.

### Limitations (in addition to PREREG.md "Known limitations")

- T1 timing: s_rate_lag = FEDFUNDS(t-1) with the cutoff on the 1st of month t; the monthly average needs the last day's effective rate, published the next business day (up to about 1-3 days of lookahead). It enters S, S+N and the random walk identically and follows the paper's protocol, so it cannot favour S+N.
- ALFRED fallback for earn/u6 before 2011-03-04 gates periods by a 9-day release lag; the employment report is sometimes released on the 10th, so in fallback windows a release-day value could be visible. Identical for S and S+N.
- The McNemar leg of the shuffled-label control has no power by construction (the shuffled model predicts hold at all 212 meetings, 0 discordant pairs, p = 1.0); only the two Diebold-Mariano legs test for leakage.
- Secondary results: H2c T2 (S+N vs random walk, p = 0.0003) mostly reflects S beating the random walk (RMSE 0.2052 vs 0.2411), not the news block. H2d McNemar (S+M+N vs S+M, p = 0.0112) is Holm-adjusted only within Plan 1 (0.045); the final H2 family (Plans 2-3) is larger and the spec expected H2d to be null. Neither is evidence that news beats the random walk or the market.
- Environment: the venv uses system site-packages (include-system-site-packages = true); versions matched results/pip_freeze.txt at run time.
- Reproduction: an independent re-run of the full ladder from the hashed data and tagged code reproduced all 16 prediction files (max |difference| 8.9e-16) and the H1 p-values (max difference 8.3e-17).

### Plan 2 (H2a) deviations

- Plan 2: T1/gru-S/seed0 took more than 3,600 s (the timing run was stopped 3,720 s after the process started, so the seed-pass exceeded 3,600 s); T1/lstm-S/seed0 took 3,039 s. Per PREREG_H2A.md all paper LSTM/GRU T1 specs run with refit every 3 months. The refit-every-month pilot forecasts were deleted unused. LSTM pilot 2026-10-09 17:44:49-18:35:34 (3,038.6 s seed-pass); GRU pilot started 18:35:46 and was stopped by a watchdog at 19:37:46 (3,720 s after start; 3,600 s plus 120 s for imports/data loading, which the runner's timer excludes). GRU at refit every 3 months later took 1,222-1,280 s per seed-pass (gru-S, the piloted spec: 1,262-1,280 s), about 3,790-3,840 s for gru-S if scaled to monthly refits.
- Plan 2: after tag `prereg-h2a`, `neural_walk.run_neural_spec` gained a keyword `anchor_every_row` (default True, the pre-registered behaviour, unchanged). Only the descriptive frozen-model analysis passes False, so a frozen hybrid's linear anchor is fitted once, like every other frozen model.
- Plan 2: run order changed to TFT and hybrid specs first, then the rest; each spec's forecasts depend only on its own settings and seed, so the order changes no result, up to GPU floating-point nondeterminism (see RESULTS_H2A.md exploratory section).
- Plan 2: run command: `PYTHONPATH=src python -m fed_news.run_neural --task all --paper-refit-every 3 --device cuda` (TFT/hybrid specs were run first with `--only`).
- Plan 2: after the results were known (final-review fixes, exploratory): `neural.fit_early_stopping` records the kept epoch as `best_epoch_` (training, return value and numbers unchanged); `run_neural --paper-refit-every` now defaults to 3; `evaluate_h2a.py` and `frozen.py` report text was extended and results/h2a/ensemble_metrics.csv gained six context rows (random walk and Plan 1 linear); `diag_tft.py` is new. No forecast file was changed or regenerated. Exploratory additions are labelled "exploratory, added after results" in RESULTS_H2A.md.

### Post-results exploratory analyses

- Clark-West (2007) test for nested models, linear S vs S+N (`src/fed_news/clark_west.py`, output `results/exploratory/clark_west.csv`), chosen after the results were known because the pre-registered Diebold-Mariano test is undersized for nested comparisons: T1 t = 2.04, p = 0.021 (308 months); T2 t = 3.03, p = 0.001 (212 meetings). Exploratory only; it does not change the H1 verdict.

### Plan 5 (H5) notes

- No deviation after tag `prereg-text`. Before the tag, code review led to keying events by their outcome trading day (weekend and holiday releases join the next trading day) so that no two events share an outcome; the spec and PREREG_TEXT.md were updated before any event-day yield change was computed.
- The look-ahead comparison changes two things at once: the language model (ModernBERT-base instead of the per-year ChronoBERT vintages) and the label dating (all labels at once). It shows what a typical non-time-consistent pipeline would report, not the separate effect of each choice.

### Plan 5 post-results notes (final review)

- Look-ahead comparison (exploratory, added after results): the naive pipeline differs from the ChronoBERT pipeline in at least four ways: the language model (one ModernBERT instead of 27 ChronoBERT vintages), label dating (all labels at once), label count (3,074 against 418-2,438 after the 20% hold-out) and overlap (its training sentences come from the documents it scores). The comparison illustrates what a typical pipeline reports; it does not isolate look-ahead. The direct tests of the naive pipeline against ChronoBERT on the same dates (naive_comparison.csv column dm_p_naive_vs_chrono, lookahead_event_compare.csv) are not significant. naive_event_study.csv carries no H5 verdict: the H5 rule applies only to the pre-registered ChronoBERT pipeline.
- Release-time caveat: the corpus has release dates only, and the 2-year yield is the afternoon fixing. Releases after that quote (late speeches, the end of press conferences) are matched to a yield fixed before they happened, so the daily event test is conservative: "not supported" means no detectable effect in a close-to-close daily window, not that Fed text carries no event-day information.
- Market benchmark: two separate bills (3-month bill on the full sample, 4-week bill from 2001-07-31), each with its own spread, replace the spec's spliced series. This was decided before the tag (plan, PREREG_TEXT.md). The expanding-mean spread drifts, so the bill is a weak benchmark and the table is descriptive.
- Void rule: about 30% of test events follow an event on the previous trading day, so there the placebo outcome is the previous event's own outcome. Disclosure only; the rule is as pre-registered.
- Method details not in the spec: press conferences are grouped with statements; MIN_TRAIN never binds; the first events (no earlier trading day pair) are dropped; zero-change days are excluded from sign accuracy.
- Commit 658f1a0 also regenerated RESULTS.md and RESULTS_H2A.md: only the provenance and the mirrored deviation notes changed, no number.
- Pre-registration timestamps are local git metadata (lightweight tags, no remote).

### Study 2 notes

- Added after tag `prereg-study2` and before any Study 2 result (while the FRASER texts were still downloading; no regional speech had been scored and no new event-day yield change computed): an exploratory frozen analysis (`src/fed_news/study2_frozen.py`, report `RESULTS_STUDY2_FROZEN.md`). The event model and the Plan 1 primary model with the Study 2 Fed-stance features are fitted once before 2016-01-01 or 2020-01-01 and never updated. Descriptive only; it is not part of H6 and cannot change its verdict.
- Before the tag, review-driven fixes changed the regional corpus (bank of the speaker taken from the "President ... of the Federal Reserve Bank of X" phrase; all vice-president variants excluded) and `stance2` (full part fingerprint, unique document ids) and made `study2` keep the H6 result if a secondary set fails.
- Study 2 run (operational, no analysis change): the stance re-training was stopped once by Claude Code's memory guard (year 2000 saved), then by a controller watchdog whose 11 GB limit was below the job's normal ~12.5 GB working set (40 restarts, no progress, 21:54-00:36 on 2026-10-10); it was restarted on 2026-10-11 05:33 with a 20 GB safety limit and completed at 18:41. Saved yearly parts were reused, so every document was scored exactly once by its year's classifier.

### Post-review notes (2026-10-11)

- Plan 3 (H2b ChronoGPT scorer, H2e paper's LLM re-run, H2f LLM news reader, H2g instruction arm), registered in PREREG.md, was not run; the work moved to Plan 5 and Study 2. The final H2 Holm family is therefore the 15 H2 tests that ran (H2a 10, H2c 2, H2d 3), with the adjusted p-values reported as "holm_h2_so_far" in RESULTS_H2A.md.
- The Plan 4 design (docs/superpowers/specs/2026-10-10-replication-market-design.md) was never tagged `prereg-rep` and never run.
- An external review of the README led to reporting corrections (README only; no forecast or pre-registered result changed): the replication failure on the paper's window is reported, the headline no longer says the paper's models (incl. its LLM) lose to no change, "look-ahead-free" became "look-ahead-controlled" with residual leaks listed, the text result is described as inconclusive, and the use of an AI coding agent (Claude Code) for code, analyses, reviews and reproduction runs is disclosed.
- New exploratory module `robustness.py` (after the review): linear vs neural DM tests, 11-pair sign test (6 of 11, p = 0.5), ex-zero-lower-bound subsample, event slope per standard deviation, stance majority-class baseline, ALFRED fallback counts (136 of 308 T1 and 96 of 212 T2 test rows). New `plots.py` draws figures from the committed forecasts.

### Post-review notes (2026-10-12)

A second external review (code and saved results) led to the disclosures below. Reporting only: no forecast file, test or verdict changed, and nothing was rerun. The only code change is a comment in `neural.fit_early_stopping`.

- Hybrid and epoch 0: when the hybrid's early stopping keeps epoch 0 (42-45% of T1 hybrid refits, 32-33% of T2; results/h2a_diag/epochs.csv), the forecast is the anchor plus the untrained TFT's output, rescaled by the residual sd and shifted by the mean out-of-fold residual. It is not the anchor alone, and the pre-registered design has no zero-correction option, so a hybrid can be worse than its own anchor even when training finds nothing. Every hybrid is worse than its anchor (RMSE T1 0.2034 vs 0.1798 with S, 0.1812 vs 0.1664 with S+N; T2 0.2417 vs 0.2052 with S, 0.2252 vs 0.1994 with S+N). The first T2 hybrid refit trains the 100,555-parameter TFT on 19 sequences and stops on 5.
- Standardisation inside the training window: features are standardised on all training rows of a refit, including the rows that are later held out inside that window (the last 20% used for TFT and hybrid early stopping, the hybrid's out-of-fold residual folds, and the expanding folds that choose the ridge and ordered-logit penalties of the Plan 1 primary model and the hybrid anchor). Neural targets are likewise standardised on all fit rows, early-stopping rows included. Test rows never enter. The inner validation is therefore slightly optimistic, which can affect only the kept epoch and the chosen penalty, and it applies identically to S and S+N.
- January 2004: the T1 panel has no row for 2004-01 (s_pce_yoy and s_pce_core_yoy are missing in the real-time data), so it is neither trained on nor forecast (308 test months, not 309). Sequence models index by row, so the 11 test sequences ending 2004-02 to 2004-12 (lookback 12) span 13 calendar months, and refit schedules counted in rows shift by one month (the T1 TFT refit dated 2004-02-01 instead of 2004-01-01). Identical for S and S+N.
- Model sizes against sample size (context for the neural results): at the first test forecast the paper LSTM (253,681 parameters with S+N) trains on 58 monthly sequences and the meeting LSTM on 45 meeting sequences, for a fixed 300 epochs without dropout or early stopping. These are the paper's LSTM settings (PREREG_H2A.md; the meeting LSTM reuses them), not tuned for this sample.

### Study 4 notes (2026-10-12)

- Amendment 7.1: the Haiku token projection (23,980,280) exceeded the 20M limit; nothing was sent until the user
  allowed the full corpus; the limit became 25M. Actual use 23,395,731 tokens.
- Design section 8 (sensitivity checks) was added after the primary result, at an external reviewer's request,
  and recorded before those runs: real-time-data reruns, a news-column ablation, class counts, HAC and
  block-bootstrap versions of the primary test, two-sided McNemar. All exploratory; RESULTS_STUDY4.md.
- The July and September 2026 meetings are a retrospective post-cutoff evaluation, not live forecasts
  (Haiku 5.5 was released on 2026-10-07).
- Archived copies of the daily news-index files could not be retrieved (archive.org offline on 2026-10-12);
  their revision behaviour is unverified.
- Found 2026-10-12 while writing the methods note: the Plan 2 "paper LSTM" (240 units, 1 layer, 300 epochs;
  PREREG_H2A.md) does not match the LSTM settings in the AMRO WP/25-10 version read on 2026-10-12 (109 units,
  two layers, no dropout, learning rate 1e-4, 644 epochs). The GRU (188 units, 1 layer, 2e-4, 449 epochs)
  matches. The source of the 240/300 setting is not recorded; the Plan 2 LSTM rows should not be described as
  the paper's LSTM settings until this is resolved.
- Resolved 2026-10-17: the 240-unit, one-layer, 300-epoch LSTM and the published RMSE 0.2376 / 0.2705 come from
  the July 2025 preliminary version of the paper (SSRN 5401236, `ssrn-5401236.pdf`, section 3.2.2 and Table 2).
  The September 2025 AMRO WP/25-10 version changes the LSTM to 109 units, two layers, 644 epochs, and its Table 2
  reports RMSE 0.1868 (LSTM) and 0.2672 (GRU), MAE 0.1164 and 0.1476. The Plan 2 LSTM therefore follows the
  preliminary version, not the working paper cited in the references. The README and the draft paper now quote
  both versions. With the working-paper numbers, the no-change forecast (0.192) no longer beats the paper's LSTM.
  No result was rerun.
- Study 4 v2 compute rule (design 12.8), applied 2026-10-13 before any v2 tuning: timed upper bounds were
  about 14.3 h (CPU: LSTM and GRU, both tasks) and 25.6 h (GPU: TFT, both tasks), above 16 h, so every v2 arm
  uses 20 Optuna trials instead of 30 (results/study4/v2_timing/).

### Exploratory pooled news test (recorded 2026-10-13, about 14:00, before running it)

Added after seeing the v2 LSTM and GRU results (the user noted that the LSTM's news versions mostly point the
right way), so it is exploratory, chosen post hoc, and outside the Holm family of the primary comparisons.
- For each network (LSTM, GRU) and task, the per-row loss differential is the squared error of the no-news
  model minus that of the news model, in rate units, for the four news-vs-no-news pairs: normal + ChronoBERT,
  normal + Haiku (against the normal no-news model), strict + ChronoBERT, strict + Haiku (against the strict
  no-news model). The pooled differential is the row-wise mean of the four; positive favours news.
- Test: one-sided Newey-West test of its mean (lags = the refit interval: 8 meetings, 12 months) and a
  moving-block bootstrap 90% interval (same block, 10,000 draws, seed 0). Also by reader (the two ChronoBERT
  pairs, the two Haiku pairs) and the count of pairs whose mean differential is positive.
- Subsets: scheduled meetings (main) and all meetings for the meeting task; all test months (main) and the
  2016-2025 paper window for the monthly task; plus test rows before 2016 vs from 2016 for both tasks.
- TFT is excluded until its forecasts finish. Results: results/study4/exploratory_pooled_news.json.

### Post-audit corrections (2026-10-13, after the audit in audit/2026-10-13/)

- Plan 1 monthly task rerun on panel t1b (second-business-day cutoff), because the original t1 cutoff (first
  calendar day) does not guarantee that inputs such as FEDFUNDS(t-1) were published. This is a post-audit
  methodological correction, labelled exploratory. Settings, tests and the 308 test months are unchanged.
  Outputs: results/post_audit_t1b/. The original results/preds_T1_*.csv, h1_tests.csv and metrics.csv are
  unchanged. H1 remains not supported (T1 DM-HLN p 0.103 on t1b, was 0.077). Details: audit/2026-10-13/AUDIT.md,
  section 6.1.
- tune.forecast_all now takes the Holm family from the registered specification (tune.registered_holm). It
  reports no adjustment while the family is incomplete. The four-test adjustment printed in
  logs/v2_fc_t2_rnn.log (LSTM + Haiku 0.0135) is provisional and superseded; the committed six-test values are
  unchanged (0.020).
- The within-Plan-1 Holm value for H2d McNemar (0.045, results/h2_pvalues.csv) is an interim family. The final
  15-test H2 family gives 0.146 (results/h2a/tests.csv).
