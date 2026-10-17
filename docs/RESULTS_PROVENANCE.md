# Results provenance

Every headline number of the draft paper (`paper/main.tex`, draft of 17 October 2026) traced to the saved
predictions, the code that evaluates them, the test, the data-vintage treatment, the commit that produced the
result file, and its status. Code paths are relative to `src/fed_news/` in the 2026-10-17 layout
([`REVIEW_MAP.md`](REVIEW_MAP.md) maps the old flat names).

**Status words.** *Registered*: hypothesis and test fixed under a git tag before the run. *Amended*: fixed in a
dated amendment written after earlier results of the same study had been seen. *Exploratory*: chosen or added
after results, or part of a study that re-used data already examined (fixed design committed before the run,
but no tag). *Superseded*: replaced by a corrected version; kept for the record, not cited.

**Vintage words.** *Mixed*: ALFRED vintages where they exist; four statistics (average earnings, U-6 monthly and
yearly change, house prices) fall back to revised values before 2011-12 / 2012-05, so 136 of 308 test months and
96 of 212 test meetings contain at least one revised value (`results/exploratory/fallback_rows.csv`). *Partial*:
those four columns dropped from the inputs. *Strict*: only the eight fully vintaged statistics (plus markets in
the meeting task). Cutoffs: T1 original = first calendar day of the month (panel t1); T1 corrected = start of
the second US business day (panel t1b); T2 = the day before the meeting.

## Abstract and headline results

| Number in the draft | Value | Prediction files | Evaluation code | Test | Vintage, cutoff | Commit | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Monthly ridge, S → S+N, RMSE (pp) | 0.1919 → 0.1675 (−13%) | `results/post_audit_t1b/preds_T1_S.csv`, `preds_T1_S+N.csv` | `audit/2026-10-13/rerun_plan1_t1b.py` (uses `models/walkforward.py`, `evaluation/evaluate.py`) | one-sided DM-HLN on squared error, p 0.103 (`tests_T1_t1b.json`) | mixed, T1 corrected | run 2026-10-13, committed 0f00878 | Registered test (H1) re-run on the corrected cutoff after the audit: exploratory correction |
| 90% interval of that gain (ΔMSE, pp²) | +0.0088 [−0.0001, +0.0219] | same | same | paired moving-block bootstrap, 12-month blocks, 10,000 draws, seed 0 | mixed, T1 corrected | 0f00878 | Exploratory (audit addition) |
| Share of the monthly gain from June 2020 | 78% (67% on the original cutoff) | same; `results/preds_T1_S*.csv` | `paper/make_figures.py` (`fig_cumgain`), `AUDIT.md` 6.5 | descriptive | mixed, T1 corrected | 0f00878 | Exploratory, chosen after seeing the forecasts |
| Network comparisons favouring text | 10 of 12 (squared error) | `results/study4/neural/preds_*_S+M*.csv`, `results/study4/neural_t1/preds_*_S*.csv` | `studies/study4_networks_v2/tune.py` (`forecast` step) | count, descriptive (pairs share data) | mixed; T1 corrected, T2 | 1a31c73 | Amended (v2 protocol 6048e0d) |
| LSTM + Haiku, meetings, RMSE (pp) | 0.1420 → 0.1333 | `results/study4/neural/preds_lstm_S+M.csv`, `preds_lstm_S+M+H.csv` | `tune.py`; `comparison_neural_T2.csv` | one-sided Newey–West on ΔMSE, p 0.0034; Holm over 6, p 0.020 | mixed, T2, 206 scheduled meetings | 1a31c73 | **Amended**, not registered |
| LSTM + Haiku, decisions correct | 78.6% → 84.5% (15 fixed / 3 broken) | same | `tune.py` (`metrics_neural_T2.csv`, `comparison_neural_T2.csv`) | exact McNemar, p 0.0038 | mixed, T2 | 1a31c73 | Amended |
| LSTM + Haiku, strict real-time | ΔMSE +0.0001 [−0.0016, +0.0024], p 0.461 | `results/study4/neural/preds_lstm_strict.csv`, `preds_lstm_strict+H.csv` | `tune.py`; interval `audit/2026-10-13/recompute.py` | Newey–West; bootstrap | **strict**, T2 | 1a31c73; interval 0f00878 | Amended robustness arm, outside Holm |
| LSTM + Haiku vs its own placebo | ΔMSE +0.0031, p 0.0008 | `preds_lstm_S+M+H.csv`, `preds_lstm_placebo-H.csv` | `tune.py` | Newey–West | mixed, T2 | 1a31c73 | Amended control |
| Haiku training cutoff covers | 204 of 206 test meetings | `results/study4/post_cutoff_T2.csv` (the 2 later meetings) | `studies/study4_haiku_logit/haiku_compare.py` (`CUTOFF`) | count | – | 7879a70 | Descriptive; cutoff from Anthropic's models overview (Jun 2026) |
| Text with markets, monthly RMSE change | 0.1333 → 0.1316 (H2d) | `results/post_audit_t1b/preds_T1_S+M.csv`, `preds_T1_S+M+N.csv` | `rerun_plan1_t1b.py` | DM-HLN p 0.251 | mixed, T1 corrected | 0f00878 | Registered test re-run: exploratory correction |
| No pre-registered text hypothesis passes | H1 0.103 / 0.061 / 0.227; H2a 0 of 10; H5 0.129; H6 0.369 | see rows below | – | – | – | – | Registered |

## Linear models (paper table `tab:linear`, `tab:cutoff`)

| Number | Value | Prediction files | Code | Test | Vintage, cutoff | Commit | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Meeting logit S → S+N, RMSE | 0.2052 → 0.1994 | `results/preds_T2_S.csv`, `preds_T2_S+N.csv` | `evaluation/evaluate.py` | DM-HLN p 0.061 (`results/h1_tests.csv`) | mixed, T2 | 8c8d3c2 | Registered (H1) |
| H1 decision accuracy | 67.9% → 69.8%, McNemar p 0.227 | same | `evaluation/evaluate.py` | exact McNemar | mixed, T2 | 8c8d3c2 | Registered (H1) |
| Monthly ridge S → S+N, original cutoff | 0.1798 → 0.1664, p 0.077 | `results/preds_T1_S.csv`, `preds_T1_S+N.csv` | `evaluation/evaluate.py` | DM-HLN | mixed, **T1 original** | 8c8d3c2 | Registered; **superseded** for T1 by the corrected cutoff |
| Meeting logit S+M → S+M+N, accuracy | 86.8% → 91.0% (11/2), McNemar p 0.011, Holm 0.146 | `results/preds_T2_S+M.csv`, `preds_T2_S+M+N.csv` | `evaluation/evaluate.py`; Holm in `studies/plan2_neural_original/evaluate_h2a.py` | exact McNemar; Holm over the 15 H2 tests (`results/h2a/tests.csv`) | mixed, T2 | 8c8d3c2; Holm 1593530 | Registered (H2d) |
| Study 4 logit, S+M → S+M+H, log loss | 0.2938 → 0.2721, p 0.144 | `results/study4/preds_T2_S+M.csv`, `preds_T2_S+M+H.csv` | `studies/study4_haiku_logit/haiku_compare.py` | DM-HLN on log loss (`verdict_T2.json`) | mixed, T2 | 7879a70 | Exploratory study, primary |
| Study 4 logit, S+M → S+M+C | 0.2938 → 0.2814, p 0.207 | `preds_T2_S+M+C.csv` | same | DM-HLN on log loss | mixed, T2 | 7879a70 | Exploratory study, secondary |
| Study 4 logit, accuracy with Haiku | 86.8% → 89.2% (6/1), p 0.062 | `preds_T2_S+M+H.csv` | same (`comparison_T2.csv`) | exact McNemar | mixed, T2 | 7879a70 | Exploratory |
| Study 5, three meetings ahead, B → B+C | log loss 0.779 → 0.839, p 0.898 | `results/study5/preds_h3_B.csv`, `preds_h3_B+C.csv` | `studies/study5_horizon/study5.py` (`comparison.csv`, `verdict.json`) | Newey–West on log loss | **strict**, T2 | 074d2b9 | Exploratory study, primary |
| Placebo, monthly ridge | 0.1919 → 0.1917, p 0.439 | `results/post_audit_t1b/preds_T1_placebo.csv` | `rerun_plan1_t1b.py` | DM-HLN | mixed, T1 corrected | 0f00878 | Registered control, re-run |
| Placebo, Study 4 Haiku | p 0.441 | `results/study4/preds_T2_placebo-H.csv` | `haiku_compare.py` | DM-HLN on log loss | mixed, T2 | 7879a70 | Exploratory study control |
| Best monthly model | ridge S+M+N 0.1316; S+M 0.1333; no-change 0.1813 | `results/post_audit_t1b/preds_T1_*.csv` | `metrics_T1_t1b.csv` | not tested | mixed, T1 corrected | 0f00878 | Descriptive |
| Meeting logit S+M+N, scheduled meetings | 90.8% (206) | `results/preds_T2_S+M+N.csv` | not in a result file; recompute: restrict to `unscheduled == 0` in `D:/fed_news/t2.pkl` (checked 2026-10-17) | descriptive | mixed, T2 | 8c8d3c2 | Descriptive |

## Networks (paper tables `tab:nets`, `tab:strict`; figure `fig:forest`)

| Number | Value | Files | Code | Test | Vintage | Commit | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Twelve primary v2 comparisons (RMSE, ΔMSE, NW p, Holm p, accuracy) | `tab:nets` rows | `results/study4/neural/`, `results/study4/neural_t1/` (`preds_<arch>_<features>.csv`, `comparison_neural_T*.csv`, `metrics_neural_T*.csv`) | `studies/study4_networks_v2/tune.py`; intervals `audit/2026-10-13/recompute.py` → `recomputed_comparisons.csv` | Newey–West (lags 8 / 12), Holm over 6 per task, bootstrap | mixed; T2 scheduled, T1 corrected | 1a31c73; intervals 0f00878 | Amended |
| Twelve strict comparisons | `tab:strict` rows | same folders, `strict*` files | same | same, no Holm | **strict** | 1a31c73 | Amended robustness |
| Forest plot | `fig:forest` | `recomputed_comparisons.csv`, `comparison_neural_T*.csv` | `paper/make_figures.py` | – | mixed (filled) and strict (hollow) | 0f00878 | – |
| Interim Holm p 0.014 (LSTM + Haiku) | superseded by 0.020 | `logs/v2_fc_t2_rnn.log` (not committed) | `tune.py` before the fix in 0f00878 | Holm over 4 (TFT not finished) | – | – | **Superseded** |
| Version 1 networks (LSTM + Haiku DM p 0.018; GRU + ChronoBERT p 0.009) | not cited | `results/study4/superseded_v1/` | `tune.py` at 74ab18f | – | mixed, T1 original | 74ab18f | **Superseded** |
| Plan 2 original networks: 0 of 10 after Holm; hybrid raw p 0.0084 / 0.0115 | `docs/reports/RESULTS_H2A.md` | `results/h2a/preds_*.csv` | `studies/plan2_neural_original/evaluate_h2a.py` (`tests.csv`, `ensemble_metrics.csv`) | DM, McNemar, Holm over 15 | mixed, T1 original | fb78233, 1593530 | Registered (H2a) |
| Replication window, our LSTM / GRU RMSE | 0.415 / 0.647 | `results/h2a/preds_T1_lstm-S.csv`, `preds_T1_gru-S.csv`, Jan 2016–Apr 2025 | `studies/plan2_neural_original/evaluate_h2a.py` (paper-window table in `docs/reports/RESULTS_H2A.md`: 0.4154) | descriptive | mixed, T1 original | 1593530 | Registered run, descriptive window |
| No-change on that window | 0.1925 | `results/preds_T1_rw.csv` | `results/metrics.csv` ("paper window") | descriptive | – | 8c8d3c2 | Descriptive |
| AMRO published numbers | WP/25-10: LSTM 0.1868, GRU 0.2672; preliminary: 0.2376, 0.2705 | – | – | – | revised data (Haver) | – | External (`DEVIATIONS.md`, 2026-10-17) |

## Event studies, stance measurement, Haiku memory

| Number | Value | File | Code | Status | Commit |
| --- | --- | --- | --- | --- | --- |
| H5 Clark–West p, 1,778 events | 0.129 | `results/text/`, `docs/reports/RESULTS_TEXT.md` | `studies/plan5_event_study/event_study.py` | Registered (`prereg-text`) | 658f1a0 |
| H6 Clark–West p, 3,158 events | 0.369 | `results/study2/`, `docs/reports/RESULTS_STUDY2.md` | `studies/study2_regional/study2.py` | Registered (`prereg-study2`) | ae94496 |
| ChronoBERT held-out accuracy | 0.56 (2000) → 0.70 (2025), mean 0.644 | `results/stance_accuracy.csv` | `features/stance.py` | Descriptive | 68b5f08 |
| Majority-class baseline | 0.42–0.56 | `results/exploratory/stance_majority_baseline.csv` | `evaluation/robustness.py` | Exploratory | 1f31475 |
| Haiku sentences labelled; hawkish / dovish shares | 343,717 of 343,742; 8.5% / 12.4% | `results/study4/haiku_index.json` | `features/haiku_index.py` | Descriptive | 3faa7c4 |
| Haiku agreement with human labels | 67.3% (300 sentences) | `results/poc_llm/pilot.json` | `studies/poc/poc_llm_labels.py` | Proof of concept | 2c8cdfc |
| Haiku recalls decisions from dates; dates masked statements within 1 year | 87.8%; 89.5% | `results/poc_haiku/memory_exam.json` | `studies/poc/poc_haiku_memory.py` | Proof of concept | 31002fd |
| Memory-flag regression | coef 0.010, p 0.378, 190 of 211 recalled | `results/study4/verdict_T2.json` | `haiku_compare.py` | Exploratory study | 7879a70 |
| Revised-value rows | 136 of 308 months, 96 of 212 meetings | `results/exploratory/fallback_rows.csv` | `evaluation/robustness.py` | Descriptive | 1f31475 |
