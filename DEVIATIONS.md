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
