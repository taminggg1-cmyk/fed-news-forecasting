# Audit of the "Fed Communication and Rate Forecasts" report (2026-10-13)

Scope: every text-versus-no-text result in the report of 2026-10-13, checked against the code, the saved prediction
files, the run logs and the git history. Nothing was retrained. Every number below was recomputed from saved
prediction files by `audit/2026-10-13/recompute.py` (output `recomputed_comparisons.csv`). The original reports,
predictions and logs are unchanged.

Recomputation method (one rule for every row, so rows can be compared):
- Paired by date. The script stops if the two files differ in dates or outcomes; no pair failed this check.
- Loss difference d = loss without text minus loss with text, per row. Positive means text helped.
- p-value: one-sided Newey-West test that the mean of d is above zero, with lags equal to the refit interval
  (8 meetings, 12 months). This is the Study 4 v2 protocol test (design section 12.6). Applied to Plan 1 and to
  the Study 4 ordered logit, it is an audit addition (post-results, exploratory). Those studies' registered
  tests are listed separately.
- 90% interval: paired moving-block bootstrap of the mean of d (the section 12.6 rule). Block length = refit
  interval (8 meetings / 12 months); 10,000 draws; seed 0. Each draw takes ceil(n / block) block starts uniformly
  from 0..n-block, keeps each block's rows in time order, and truncates to n rows. Every difference uses the same
  resampled rows for both models. For Plan 1, Study 4 section 4 and Study 5 this interval is post-results
  exploratory unless the study's own report already gave it (Study 4 section 8 and Study 5 did, with the same
  rule; the recomputed intervals match theirs).
- Accuracy is tested only by exact one-sided McNemar on the discordant meetings, plus the same bootstrap
  interval. No squared-error or log-loss p-value is used for an accuracy claim.

Check of the stored values: every recomputed p-value and interval matches the value stored in the study's own
results file to the printed precision. Examples: v2 T2 LSTM + Haiku Newey-West p 0.0034, interval
[+0.0010, +0.0039]; Study 4 log-loss interval [-0.008, +0.061]; Study 5 primary interval [-0.140, +0.004].

## 1. Result provenance

| Result in the report | Files | Code | Commit (date) | Protocol status | Before or after the corrections | Verified |
|---|---|---|---|---|---|---|
| Plan 1 H1 tests (DM-HLN T1 p 0.077, T2 p 0.061, McNemar p 0.227) | results/h1_tests.csv, results/preds_T{1,2}_{S,S+N,placebo,shuffled}.csv | walkforward.py, evaluate.py | 8c8d3c2 (2026-10-09 13:11), after tag `prereg` (5d0c31c, 13:00) | Primary, pre-registered | Before. T1 uses the v1 monthly cutoff (first calendar day), which section 12.1 found does not guarantee that FEDFUNDS(t-1) was published | p-values: verified. T1 information availability: **unverified** (see section 5) |
| Plan 1 RMSE (T1 0.180 to 0.166 pp, T2 0.205 to 0.199 pp) | results/metrics.csv | evaluate.py | 8c8d3c2 | Primary metrics | Same as above | Verified by recomputation |
| Clark-West T1 p 0.021, T2 p 0.001 | results/exploratory/clark_west.csv | clark_west.py | 9c2bd82 (2026-10-10) | Exploratory, chosen after results | Before | Present in file; not recomputed |
| Plan 2 networks (0 of 10 after Holm; hybrid raw p 0.008, 0.011) | results/h2a/ | neural_walk.py, evaluate_h2a.py | 6a24e0a (2026-10-10), after tag `prereg-h2a` | Secondary, pre-registered | Before: v1 monthly cutoff, untuned paper-style settings; the LSTM settings do not match the paper (DEVIATIONS.md) | File values: verified. Not recomputed |
| Event studies H5 p 0.129, H6 p 0.369 | RESULTS_TEXT.md, RESULTS_STUDY2.md | event_study.py | tags `prereg-text`, `prereg-study2` | Primary, pre-registered | Not affected | File values: verified. Not recomputed |
| Study 4 ordered logit (log loss 0.294 to 0.272, DM p 0.144) | results/study4/preds_T2_*.csv, verdict_T2.json | haiku_compare.py | 7879a70 (2026-10-12 19:34); design b4a00fd (19:07), committed before the run, no git tag | Primary of Study 4 (exploratory study: the meetings were used before) | Not affected by v2 (section 12 covers networks only) | Verified by recomputation |
| Study 4 real-time sensitivity | results/study4/sensitivity/ | study4_sensitivity.py | b7cda28 | Exploratory (section 8, added after results) | Drops 4 fallback columns. Pre-2001 training rows still use fallback for PCE, core PCE, PPI and core CPI (first vintages 1996-2000) | Verified |
| Study 4 networks, **v2** (final) | results/study4/neural/, neural_t1/ (comparison_*, preds_*, settings_*) | tune.py | 1a31c73 (2026-10-13 15:25); protocol 6048e0d (00:51) | Secondary. Protocol written after the v1 results were seen (disclosed) | After. Corrected cutoff (t1b), identical folds, rate-unit scoring, strict arms | Verified by recomputation, including Holm over 6 |
| Study 4 networks, v1 | results/study4/superseded_v1/ | tune.py at 74ab18f | 74ab18f, c45c7b5, 26d9011 | Superseded | Before | Kept for audit only; **must not be cited** |
| Exploratory pooled news test | results/study4/exploratory_pooled_news.json | study4_pooled.py | 1a31c73 | Post hoc exploratory (DEVIATIONS.md) | After | File present; not recomputed (**unverified**) |
| Study 5 (log loss 0.779 to 0.839, p 0.898) | results/study5/ | study5.py | 074d2b9 (2026-10-12 21:03); design 85de61f, committed before the run | Primary of Study 5 (exploratory study) | Strict real-time by construction, checked in code | Verified by recomputation |
| ChronoBERT classifier accuracy 0.644 | results/stance_accuracy.csv | stance.py | 68b5f08 | Descriptive | Not affected | Verified (mean of the file) |
| Haiku label agreement 0.673 | results/poc_llm/pilot.json, RESULTS_POC.md | poc_llm_labels.py | 2c8cdfc | Proof of concept | Not affected | File values: verified |

## 2. Corrected results by data-vintage treatment

Units: RMSE in percentage points of the policy rate. ΔMSE = MSE without text minus MSE with text (positive
favours text), in squared percentage points. Δ log loss is defined the same way. p = one-sided Newey-West on the
metric named in the "tested" column. Accuracy rows give McNemar fixes/breaks and an exact one-sided p.

### 2a. Mixed vintage (revised-data fallback in some economic inputs)

Information rules: decisions are forecast with data dated before the day-before-meeting cutoff (T2). Months use
the first calendar day (Plan 1, v1) or the second business day (v2 networks, panel t1b). Fallback columns:
earnings, U6 and house prices before 2011-12 to 2012-05. PCE, PPI and core CPI are revised before their first
vintages (1996-2000).

| Study / model | Task, sample | Inputs (without text) | Without text | With text | Difference [90% CI] | p (tested) | Accuracy without → with, McNemar | Role |
|---|---|---|---|---|---|---|---|---|
| Plan 1 ridge, S → S+N (original t1 cutoff; superseded by section 6) | T1, 308 months 2001-01..2026-09 | 16 S columns | RMSE 0.1798 | RMSE 0.1664 | ΔMSE +0.0046 [−0.0002, +0.0106] | 0.081 (MSE; registered DM-HLN 0.077) | 55.8% → 53.9%; 15/21, p 0.88 | Primary H1 |
| Plan 1 ordered logit, S → S+N | T2, 212 meetings 2001-01..2026-09 | 16 S | RMSE 0.2052 | RMSE 0.1994 | ΔMSE +0.0023 [−0.0018, +0.0048] | 0.167 (MSE; registered DM-HLN 0.061) | 67.9% → 69.8%; 10/6, p 0.23 (registered) | Primary H1 |
| Plan 1 ridge, S+M → S+M+N (original t1 cutoff; superseded by section 6; N includes current-vintage news, GPR, EPU indices) | T1, 308 | 16 S + 8 M | RMSE 0.1311 | RMSE 0.1332 | ΔMSE −0.0006 [−0.0028, +0.0017] | 0.659 (MSE; registered H2d DM-HLN 0.695) | 66.2% → 66.9%; 13/11, p 0.42 | Secondary H2d |
| Plan 1, S+M → S+M+N | T2, 212 | 16 S + 8 M | RMSE 0.1321 | RMSE 0.1323 | ΔMSE −0.0000 [−0.0012, +0.0009] | 0.526 (MSE; registered H2d DM-HLN 0.520) | 86.8% → 91.0%; 11/2, p 0.011 (registered H2d McNemar; Holm 0.146 in the final 15-test H2 family) | Secondary H2d |
| Study 4 ordered logit, S+M → S+M+H | T2, 212 | 16 S + 8 M | log loss 0.2938 | 0.2721 | Δ +0.0217 [−0.0085, +0.0611] | 0.154 (log loss; registered DM-HLN 0.144) | 86.8% → 89.2%; 6/1, p 0.062 | Primary |
| Study 4 ordered logit, S+M → S+M+C | T2, 212 | 16 S + 8 M | log loss 0.2938 | 0.2814 | Δ +0.0123 [−0.0079, +0.0391] | 0.197 (log loss) | 86.8% → 87.3%; 1/0, p 0.50 | Secondary |
| v2 LSTM, S+M → S+M+H | T2, 206 scheduled meetings | 16 S + 8 M | RMSE 0.1420 | 0.1333 | ΔMSE +0.0024 [+0.0010, +0.0039] | 0.0034 (MSE); **Holm over 6: 0.020** | 78.6% → 84.5%; 15/3, p 0.004 | Primary |
| v2 LSTM, S+M → S+M+C | T2, 206 | 16 S + 8 M | 0.1420 | 0.1406 | +0.0004 [−0.0012, +0.0028] | 0.375; Holm 0.85 | 78.6% → 83.5%; 15/5, p 0.021 | Primary |
| v2 GRU, S+M → S+M+H | T2, 206 | 16 S + 8 M | 0.1445 | 0.1379 | +0.0019 [−0.0007, +0.0063] | 0.200; Holm 0.85 | 80.6% → 82.0%; 13/10, p 0.34 | Primary |
| v2 GRU, S+M → S+M+C | T2, 206 | 16 S + 8 M | 0.1445 | 0.1612 | −0.0051 [−0.0089, −0.0019] | 0.992; Holm 0.99 | 80.6% → 80.1%; 6/7, p 0.71 | Primary |
| v2 TFT-inspired, S+M → S+M+H | T2, 206 | 16 S + 8 M | 0.1415 | 0.1342 | +0.0020 [−0.0008, +0.0061] | 0.170; Holm 0.85 | 86.4% → 86.9%; 8/7, p 0.50 | Primary |
| v2 TFT-inspired, S+M → S+M+C | T2, 206 | 16 S + 8 M | 0.1415 | 0.1361 | +0.0015 [−0.0009, +0.0051] | 0.210; Holm 0.85 | 86.4% → 85.9%; 7/8, p 0.70 | Primary |
| v2 LSTM, S → S+H | T1, 308 months (t1b) | 16 S | RMSE 0.1878 | 0.1722 | +0.0056 [+0.0007, +0.0089] | 0.026; Holm over 6: 0.156 | 59.1% → 65.3%; 30/11, p 0.002 | Primary |
| v2 LSTM, S → S+C | T1, 308 | 16 S | 0.1878 | 0.1979 | −0.0039 [−0.0100, +0.0007] | 0.878; Holm 1.00 | 59.1% → 53.9%; 26/42, p 0.98 | Primary |
| v2 GRU, S → S+H | T1, 308 | 16 S | 0.1803 | 0.1779 | +0.0008 [−0.0105, +0.0089] | 0.447; Holm 1.00 | 60.4% → 59.1%; 37/41, p 0.71 | Primary |
| v2 GRU, S → S+C | T1, 308 | 16 S | 0.1803 | 0.1782 | +0.0008 [−0.0027, +0.0042] | 0.360; Holm 1.00 | 60.4% → 62.3%; 25/19, p 0.23 | Primary |
| v2 TFT-inspired, S → S+H | T1, 308 | 16 S | 0.1911 | 0.1726 | +0.0067 [+0.0018, +0.0137] | 0.034; Holm 0.168 | 54.9% → 59.4%; 26/12, p 0.017 | Primary |
| v2 TFT-inspired, S → S+C | T1, 308 | 16 S | 0.1911 | 0.1766 | +0.0053 [+0.0008, +0.0107] | 0.042; Holm 0.169 | 54.9% → 56.5%; 24/19, p 0.27 | Primary |

T1 accuracy is the direction of the monthly average-rate change, not a policy decision (design 12.6).

### 2b. Fallback columns dropped from the inputs, but fallback values remain in pre-2001 training rows

| Study / model | Task, sample | Inputs | Without text | With text | Difference [90% CI] | p (tested) | Accuracy, McNemar | Role |
|---|---|---|---|---|---|---|---|---|
| Study 4 rt, S+M → S+M+H | T2, 212 | 12 S + 8 M | log loss 0.2975 | 0.2760 | +0.0216 [−0.0006, +0.0517] | 0.091 (log loss, NW; RESULTS_STUDY4.md gives DM-HLN 0.038) | 88.7% → 90.1%; 5/2, p 0.23 | Exploratory |
| Study 4 rt, S+M → S+M+C | T2, 212 | 12 S + 8 M | 0.2975 | 0.2816 | +0.0159 [−0.0112, +0.0566] | 0.225 | 88.7% → 88.2%; 0/1, p 1.00 | Exploratory |

### 2c. Strict real-time (only series with complete ALFRED vintages; no fallback in tuning, training or prediction)

Retained series: policy rate, M2 monthly and yearly change, payroll change, GDP, CPI, 10y-2y and 1y-6m slopes
(8 columns), plus the 8 market columns in the meeting tasks. They differ from 2a's inputs: 8 S columns are
dropped (PCE, core PCE, earnings, U6 monthly and yearly, PPI, core CPI, house prices). Differences between 2a and
2c therefore cannot be attributed to vintage treatment alone.

| Study / model | Task, sample | Without text | With text | Difference [90% CI] | p (tested) | Accuracy, McNemar | Role |
|---|---|---|---|---|---|---|---|
| v2 LSTM, strict → +H | T2, 206 | RMSE 0.1251 | 0.1246 | ΔMSE +0.0001 [−0.0016, +0.0024] | 0.461 (MSE) | 81.6% → 85.4%; 12/4, p 0.038 | Robustness |
| v2 LSTM, strict → +C | T2, 206 | 0.1251 | 0.1247 | +0.0001 [−0.0018, +0.0026] | 0.467 | 81.6% → 84.5%; 13/7, p 0.13 | Robustness |
| v2 GRU, strict → +H | T2, 206 | 0.1222 | 0.1323 | −0.0026 [−0.0058, −0.0001] | 0.925 | 83.0% → 86.4%; 12/5, p 0.072 | Robustness |
| v2 GRU, strict → +C | T2, 206 | 0.1222 | 0.1306 | −0.0021 [−0.0058, +0.0010] | 0.849 | 83.0% → 85.9%; 10/4, p 0.090 | Robustness |
| v2 TFT-inspired, strict → +H | T2, 206 | 0.1358 | 0.1405 | −0.0013 [−0.0065, +0.0023] | 0.688 | 87.4% → 87.9%; 9/8, p 0.50 | Robustness |
| v2 TFT-inspired, strict → +C | T2, 206 | 0.1358 | 0.1534 | −0.0051 [−0.0118, −0.0007] | 0.935 | 87.4% → 86.9%; 12/13, p 0.66 | Robustness |
| v2 LSTM, strict → +H | T1, 308 | 0.1972 | 0.1736 | +0.0087 [−0.0051, +0.0182] | 0.117 | 53.6% → 58.4%; 48/33, p 0.060 | Robustness |
| v2 LSTM, strict → +C | T1, 308 | 0.1972 | 0.1854 | +0.0045 [−0.0026, +0.0106] | 0.132 | 53.6% → 55.5%; 27/21, p 0.24 | Robustness |
| v2 GRU, strict → +H | T1, 308 | 0.1841 | 0.1645 | +0.0068 [+0.0015, +0.0115] | 0.014 | 59.4% → 63.6%; 37/24, p 0.062 | Robustness |
| v2 GRU, strict → +C | T1, 308 | 0.1841 | 0.1885 | −0.0017 [−0.0082, +0.0042] | 0.669 | 59.4% → 59.4%; 21/21, p 0.56 | Robustness |
| v2 TFT-inspired, strict → +H | T1, 308 | 0.1859 | 0.1731 | +0.0046 [−0.0010, +0.0100] | 0.084 | 53.6% → 56.5%; 25/16, p 0.11 | Robustness |
| v2 TFT-inspired, strict → +C | T1, 308 | 0.1859 | 0.1806 | +0.0020 [−0.0046, +0.0079] | 0.305 | 53.6% → 50.0%; 17/28, p 0.96 | Robustness |
| Study 5, B → B+C, 3 meetings ahead | 204 meetings 2001-01-31..2026-06-17 | log loss 0.7788 | 0.8389 | −0.0601 [−0.1397, +0.0044] | 0.898 (log loss) | 71.6% → 69.1%; 1/6, p 0.99 | Primary |
| Study 5, B → B+H, 3 ahead | 204 | 0.7788 | 0.7691 | +0.0097 [−0.0100, +0.0464] | 0.300 | 71.6% → 69.6%; 2/6, p 0.97 | Secondary |
| Study 5, B → B+C, 2 ahead | 205 | 0.5202 | 0.5153 | +0.0049 [−0.0161, +0.0163] | 0.323 | 80.5% → 80.0%; 2/3, p 0.81 | Secondary |
| Study 5, B → B+H, 2 ahead | 205 | 0.5202 | 0.4815 | +0.0387 [+0.0092, +0.0645] | 0.0135 raw; Holm over 4: 0.054 | 80.5% → 82.0%; 5/2, p 0.23 | Secondary |
| Study 5, B → B+C, next meeting | 206 | 0.3154 | 0.3280 | −0.0126 [−0.0475, +0.0062] | 0.782 | 87.9% → 86.9%; 2/4, p 0.89 | Reference |

Counts: text lowered MSE in 14 of 18 mixed-vintage pairs, 1 of 2 partial and 11 of 17 strict pairs. These counts
are descriptive. The pairs share data and are not independent, so the counts are not a test.

## 3. Holm families and analysis roles

| Study | Family | Size | Test | Survivors at 0.05 |
|---|---|---|---|---|
| Plan 1 H1 | No Holm. All three registered tests had to pass (conjunctive rule) | 3 | DM-HLN on MSE (T1, T2), McNemar (T2) | None pass |
| Plan 1 H2c, H2d and Plan 2 H2a | Holm across the final H2 family (15 tests that ran: H2a 10, H2c 2, H2d 3; README) | 15 | One-sided DM on MSE; McNemar on accuracy | 1: H2c T2 (S+N vs random walk, 0.0044), which is not a text-increment test. H2d McNemar 0.146. The within-Plan-1 value 0.045 was an interim family |
| Study 4 section 4 | Single primary | 1 | DM-HLN on log loss | Not significant (0.144) |
| Study 4 section 8 | Holm across the twelve news rows (stated in RESULTS_STUDY4.md) | 12 | DM-HLN on log loss | 0 |
| Study 4 v2, T2 | Holm across 3 architectures × 2 readers (design 12.6) | 6 | Newey-West on MSE, scheduled meetings | 1: LSTM + Haiku (adjusted 0.020) |
| Study 4 v2, T1 | Same | 6 | Newey-West on MSE, all test months | 0 (smallest adjusted 0.156) |
| Study 5 | Single primary; Holm across four secondaries | 1 + 4 | Newey-West on log loss | 0 (smallest secondary adjusted 0.054) |
| Exploratory pooled news test | Outside any family (post hoc) | – | Newey-West on the pooled MSE difference | Not a confirmatory result |

Audit additions, all post-results exploratory: the Newey-West p and the bootstrap intervals for Plan 1 and for
Study 4 section 4. These were not in those studies' protocols.

## 4. Discrepancies between earlier reports, explained

1. **"Holm-adjusted p = 0.014 for LSTM + Haiku" against "no significant network results after Holm".**
   - The 0.014 is 0.0135, printed in `logs/v2_fc_t2_rnn.log` (11:45 on 2026-10-13). That evaluation ran when
     only the LSTM and GRU forecasts existed. `tune.forecast_all` builds the Holm family from whichever primary
     pairs have predictions, so the family had 4 members: 0.0034 × 4 = 0.0135.
   - After the TFT forecasts finished (`logs/v2_eval_t2.log`, 15:21, committed in 1a31c73), the family has the
     6 members fixed in design 12.6, and the adjusted p is 0.020. The final value is 0.020, and it is below 0.05.
   - "No significant network results after Holm" matches Plan 2 H2a (RESULTS_H2A.md: 0 of 10). Those are
     different networks (untuned, paper-style settings, v1 monthly cutoff, S vs S+N) and a different family.
     It also matches v2 T1 (0 of 6). It does not match v2 T2.
   - The PDF itself was not found on this machine, so which of these it reported is **unverified**.
   - A second 0.014 exists: Study 5, B → B+H at 2 meetings ahead, raw p 0.0135 on log loss (Holm-adjusted 0.054).
     It is a different model and should not be confused with the LSTM result.
2. **Six comparisons against four.**
   - Six is the registered v2 family per task (design 12.6: LSTM, GRU, TFT × ChronoBERT, Haiku).
   - Four arises in two places: (a) the interim RNN-only evaluation above (TFT not finished); (b) the exploratory
     pooled test, which pools four pairs per network (mixed + ChronoBERT, mixed + Haiku, strict + ChronoBERT,
     strict + Haiku).
   - The TFT is in the final six-member families for both tasks. DEVIATIONS.md says the pooled test excluded TFT
     "until its forecasts finish". `exploratory_pooled_news.json` now contains TFT rows, so the note is out of
     date.
3. **v1 against v2 network results.**
   - Commit 74ab18f ("LSTM gains with Haiku, DM p 0.018; GRU with ChronoBERT, DM p 0.009") reports v1 results:
     non-identical validation folds, standardised-unit scoring and the first-calendar-day monthly cutoff. They
     are superseded and kept only in `results/study4/superseded_v1/`.
   - The GRU + ChronoBERT gain does not survive in v2: GRU S+M → S+M+C ΔMSE −0.0051, p 0.992.
4. **Evaluation samples differ across studies.**
   - Ordered logit T2: all 212 meetings.
   - v2 networks T2: 206 scheduled meetings (primary), with 212 reported as well.
   - Study 5: 204-206 rows starting 2001-01-31, by horizon.
   - T1: 308 months in both Plan 1 (panel t1, first-calendar-day cutoff) and v2 (panel t1b, second-business-day
     cutoff). The inputs are therefore not identical across these studies.
5. **Placebo constructions differ.**
   - Plan 1 and Study 5: text columns permuted within each training window.
   - Study 4 v2 networks: circular shift by half the panel.
   - Study 4 section 4: permutation (placebo-H).
   - Passing a placebo shows that a shuffled signal does not help. It does not show that the real signal is free
     of leakage.
6. **Errors in the 2026-10-13 report (corrected in the revised report).**
   - "Lowered forecast error in almost every study": not supported. The README count is 6 of 11 model-task pairs
     (sign test p 0.5), and Study 5 at 3 meetings ahead was worse with text.
   - "The gain largely disappears once macro and market data are in": true for the ordered logit (MSE). It is
     false for the v2 LSTM, which gains with Haiku on top of S+M (T2). Plan 1 S+M → S+M+N accuracy also rose
     from 86.8% to 91.0% (McNemar p 0.011, exploratory, current-vintage indices).
   - The v2 network results were missing from the report entirely, including the one primary comparison that
     survives Holm.
   - The placebo p-values were quoted side by side although they come from different tests and metrics.
   - The monthly chart (Plan 1 T1) uses the v1 cutoff without saying so.

## 5. Unresolved or unverifiable

1. The PDF that reports "no significant network results after Holm" was not found; its source is **unverified**.
2. Plan 1 T1 (and the report's monthly chart) uses the first-calendar-day cutoff. Section 12.1 established that
   FEDFUNDS(t-1) is not guaranteed to be published by then. This affects both arms, but the size of any
   look-ahead is **unverified**. A targeted fix exists: rerun only the Plan 1 T1 ordered logit (S, S+N) on panel
   t1b, labelled post-results exploratory. It has not been run.
3. Haiku memory: Haiku recalls 87.8% of decisions from dates and dates 89.5% of masked statements. The only
   primary network survivor (LSTM + Haiku, T2) is Haiku-based. It does not survive in the strict arm (p 0.461 on
   MSE), and the strict LSTM without text has a lower RMSE (0.1251) than the mixed LSTM with Haiku (0.1333).
   Whether the gain is memory is not identified.
4. Current-vintage daily indices (SF Fed news sentiment, GPR, EPU) in Plan 1 N: their revision behaviour is
   **unverified** (archive copies not retrieved).
5. The exploratory pooled test (`exploratory_pooled_news.json`, code `study4_pooled.py`) was not recomputed
   (**unverified**).
6. Clark-West p-values (Plan 1, exploratory) and the H2a, H5 and H6 results were checked against their files,
   not recomputed from predictions.
7. The v2 protocol (section 12) was written after the v1 network results were seen. Its comparisons are
   prespecified relative to v2 only.
8. The strict and mixed arms differ in their feature sets (16 vs 24 inputs in T2, 8 vs 16 in T1). No arm
   isolates vintage treatment with identical columns.
9. Code issue (no result changed): `tune.forecast_all` builds the Holm family from the primary pairs that have
   predictions, so an interim run prints a smaller family (4 instead of 6 on 2026-10-13 at 11:45). A guard that
   requires all six registered pairs before computing `nw_p_holm` would prevent this. Not applied.

## 6. Post-audit corrections (2026-10-13, after review of this audit)

### 6.1 Plan 1 monthly task rerun on the corrected cutoff (post-audit methodological correction, exploratory)

- **Defect.** Panel `t1` dates each month's forecast at the first calendar day. At that time inputs such as
  FEDFUNDS(t-1) are not guaranteed to be published (Study 4 design 12.1). Both arms sharing the problem does
  not make their comparison safe.
- **Fix.** `audit/2026-10-13/rerun_plan1_t1b.py` reruns the full Plan 1 T1 ladder (rw, taylor, S, S+N, placebo,
  shuffled, S+M, S+M+N) on `t1b_full.pkl`. That panel was built 2026-10-13 00:53 (commit e2b74a8), with the
  cutoff at the start of the second US business day; cutoff check: 393 months, 0 violations in
  `results/study4/v2_verification.json`. Model settings, grids, the t-2 rule, the seed and the 308 test months
  (2001-01 to 2026-09, same dates) are unchanged. Outputs are in `results/post_audit_t1b/`; the original
  `results/preds_T1_*.csv` are untouched.
- **Correction to earlier text.** The Plan 1 monthly model is ridge regression on the rate change, not an
  ordered logit; only the meeting task uses the ordered logit.
- **Reproducibility.** Rerunning the original `t1` ladder in the project environment (`D:/gold_lm/venv`)
  reproduces the committed forecasts to 9e-16. The `t1b` forecasts are identical in the venv and in the global
  Python 3.13 (same library versions).
- **Unaffected and not rerun.** The meeting task (T2), all networks, Studies 4 and 5, and the event studies.

| Comparison (T1, 308 months) | RMSE without → with (pp), t1b | Original t1 | ΔMSE [90% CI], t1b | DM-HLN p (registered test) | NW p | Accuracy without → with, McNemar |
|---|---|---|---|---|---|---|
| H1: S → S+N | 0.1919 → 0.1675 | 0.1798 → 0.1664 | +0.0088 [−0.0001, +0.0219] | 0.103 (was 0.077) | 0.099 | 52.9% → 54.2%; 22/18, p 0.32 |
| Control: S → placebo | 0.1919 → 0.1917 | 0.1798 → 0.1890 | +0.0001 [−0.0005, +0.0012] | 0.439 (was 0.858) | 0.446 | 52.9% → 53.6%; 6/4, p 0.38 |
| Control: rw → shuffled | 0.1813 → 0.1824 | 0.1813 → 0.1823 | −0.0004 [−0.0020, +0.0013] | 0.750 (was 0.737) | 0.664 | 52.6% → 52.6%; 0/0 |
| H2c: rw → S+N | 0.1813 → 0.1675 | 0.1813 → 0.1664 | +0.0048 [−0.0042, +0.0107] | 0.074 (was 0.056) | 0.183 | 52.6% → 54.2%; 36/31, p 0.31 |
| H2d: S+M → S+M+N | 0.1333 → 0.1316 | 0.1311 → 0.1332 | +0.0004 [−0.0011, +0.0025] | 0.251 (was 0.695) | 0.348 | 66.2% → 67.2%; 15/12, p 0.35 |

- **H1 under the corrected cutoff.** The registered rule needs all three tests at p < 0.05: T1 0.103, T2 0.061
  (unchanged), McNemar 0.227 (unchanged). H1 is still not supported, and the controls do not void it
  (placebo 0.439 and 0.808 against 0.05; shuffled 0.750 and 0.731 against 0.01).
- **What changed.** The corrected cutoff makes S worse (RMSE 0.1798 to 0.1919) and leaves S+N almost unchanged
  (0.1664 to 0.1675). The estimated text gain is therefore larger, but less precise. With markets in the
  model, text now points slightly toward a gain (H2d ΔMSE +0.0004) instead of a loss. Neither change is
  significant.
- Tables 2a and 3: the Plan 1 T1 rows above supersede the t1 rows. Those rows are kept for comparison and
  marked superseded. Full output: `recomputed_t1b_rerun.csv`, `results/post_audit_t1b/tests_T1_t1b.json`.

### 6.2 Holm implementation fixed

- `tune.registered_holm` builds the family from the registered specification (all of `ARCHS` × the task's
  primary pairs, design 12.6). While any member is missing it reports no adjusted p-value.
- `tune.forecast_all` uses it. New tests in `tests/test_tune.py`:
  - an interim LSTM + GRU family returns no adjustment;
  - the function reproduces the committed `nw_p_holm` columns for both tasks.
- **Superseded:** the four-test adjustment printed at 11:45 (`logs/v2_fc_t2_rnn.log`: LSTM + Haiku 0.0135, GRU
  + Haiku 0.5995, LSTM + ChronoBERT 0.7494, GRU + ChronoBERT 0.9925). These values are provisional and must not
  be cited. The final values (six-test family) are unchanged: LSTM + Haiku 0.020.
- **The same issue in Plan 1.** The H2d McNemar adjustment 0.045 (`results/h2_pvalues.csv`, "holm_within_plan1")
  is also an interim family. The final registered H2 family of 15 tests gives 0.146 (`results/h2a/tests.csv`).

### 6.3 Statements corrected

1. **"No pre-registered hypothesis passed" and "one pre-specified primary comparison survives".** They do not
   conflict once their sources are separated:
   - The pre-registrations under git tags (`prereg`, `prereg-h2a`, `prereg-text`, `prereg-study2`) test text
     in H1, H2a, H2d, H5 and H6. None passed in its final family. H2c T2 passes (adjusted 0.0044), but it
     tests S+N against the random walk, not what text adds.
   - The surviving LSTM + Haiku comparison belongs to the **dated amendment** of Study 4 (design section 12,
     2026-10-13 00:45). That amendment was written after the v1 network results were seen, in a study that
     re-used meetings from earlier studies and carries no git tag. It is prespecified relative to v2 only, not
     pre-registered.
2. **"The gain disappears with strict inputs".** This was too strong. Correct wording: the improvement was not
   statistically established in the strict-input analysis (LSTM + Haiku, meetings, ΔMSE +0.0001,
   90% CI [−0.0016, +0.0024], p 0.461; mixed: +0.0024 [+0.0010, +0.0039]). A significant result in one setting
   and a non-significant one in another does not by itself establish a difference between them. No test of
   that difference was run.
3. **The RMSE comparison across strict and mixed inputs is descriptive.** Strict LSTM without text 0.1251 pp;
   mixed LSTM with Haiku 0.1333 pp. The strict arm has 16 inputs against 24, so the difference cannot be
   attributed solely to vintage quality.

### 6.4 Environment note

- The project environment is `D:/gold_lm/venv` (logs/run_study2_v2.sh). All post-audit runs and tests were
  repeated there.
- During the audit, `pyarrow==25.0.1` and `pytest==9.1.1` (with `pluggy==1.6.0`, `iniconfig==2.3.1`, the
  versions in results/pip_freeze.txt) were installed into the global Python 3.13. That environment is not the
  project's. The installs are harmless but unnecessary.

### 6.5 Influence of one month on the monthly gain (descriptive, post-results)

| Panel | ΔMSE, all 308 months | ΔMSE without 2020-06 | Share of the gain from 2020-06 | RMSE S → S+N without 2020-06 (pp) |
|---|---|---|---|---|
| t1b (corrected cutoff) | +0.0088 | +0.0019 | 78% | 0.1723 → 0.1666 |
| t1 (original) | +0.0046 | +0.0015 | 67% | 0.1707 → 0.1661 |

In June 2020 the statistics-only model forecast −1.41% on t1b (−0.93% on t1), against an actual 0.08%. Most of
the monthly squared-error gain from text comes from avoiding that one error. This is a descriptive check chosen
after seeing the forecasts, not a test.
