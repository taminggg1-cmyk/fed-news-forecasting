# Plan 5 design: evidence for look-ahead-free text features as a forecasting method

Date: 2026-10-10. Status: new work after the H1 and H2a results were known. The event study (section 2)
is confirmatory and is pre-registered in `PREREG_TEXT.md`, tagged `prereg-text`, before any event-day
yield change is computed. Sections 3-5 are exploratory and labelled so. Nothing here changes the H1 or
H2a verdicts. The Plan 4 replication grid (`2026-10-10-replication-market-design.md`, section 2) is
deferred and optional; its T-bill benchmark (section 3 there) moves into this plan as section 4.

## 1. Purpose and framing

The paper is reframed as a methods paper: Fed documents scored by a language model that cannot see the
future (ChronoBERT, one version per year), combined with real-time statistics, evaluated walk-forward
with leakage controls and pre-registration. Existing evidence: text lowers error in 6 of 9 models, the
placebo shows the gain comes from content, and the exploratory Clark-West test is significant
(p = 0.021 monthly, 0.001 per meeting). This plan adds:

1. a confirmatory test with enough observations that text carries information (event study);
2. evidence that the time-consistent design matters (look-ahead comparison);
3. the market benchmark (T-bill proxy);
4. which text features carry the signal (ablation).

## 2. Event study (H5, confirmatory)

**Question.** Does the stance of Fed communication predict the move in the 2-year Treasury yield on the
day it is released, out of sample?

**Events.** Every release day from 2001-01-01 to 2026-09-30 with at least one statement, set of minutes,
press-conference transcript or speech in the corpus (1,800 days, 2,248 documents). Documents from
1994-2000 are used only for training and for the surprise baseline.

**Stance.** Document stance from `D:/fed_news/doc_stance.pkl` (each document scored by the classifier of
its year, as in Plan 1). Day stance = mean stance of that day's documents.

**Regressor: stance surprise.** Day stance minus the mean stance of all corpus documents released in the
30 calendar days before the event day (the event day excluded); if there are none, minus the stance of
the most recent earlier document.

**Outcome.** Change in the 2-year constant-maturity Treasury yield (`market_daily.pkl` column `y2`, FRED
`DGS2`), in basis points: yield on the event day minus yield on the previous business day. If the event
day has no yield (weekend or holiday), the next business day's yield minus the last yield before the event
day.

**Primary test.** Walk-forward out-of-sample forecasts. For each test event day d (2001-2026), fit by OLS
on all earlier event days (from 1994): small model `dy2 = a`, large model `dy2 = a + b * surprise`.
One-sided Clark-West test that the large model adds predictive information, horizon 1. H5 is supported if
p < 0.05. Single primary test, so no multiplicity adjustment.

**Void rule (anticipation or leakage control).** The same test with the yield change on the business day
before the event day as the outcome. If its Clark-West p < 0.05, H5 is reported as void.

**Secondary (reported, not decisive).** Full-sample OLS of dy2 on surprise over the 1,800 test days with
Newey-West standard errors (5 lags), one-sided t-test of b > 0; the primary test split by document type
(statement, minutes, press conference, speech); out-of-sample sign accuracy.

## 3. Look-ahead comparison (exploratory)

**Question.** Would a standard language model, one that has read about later events and is trained on all
labels at once, make text look more useful than it really is?

**Naive pipeline.** Fine-tune `answerdotai/ModernBERT-base` (pre-trained on text up to 2024) once on all
Trillion Dollar Words and World Central Banks labels with no date restriction, with the Plan 1 settings
(learning rate 2e-5, 3 epochs, batch 16, length 256, three seeds, averaged probabilities). Score every
corpus document and rebuild the two Fed-stance features with the Plan 1 feature code. If ModernBERT is
not supported by the installed `transformers`, use `roberta-base` and record the substitution.

**Comparison.** Rerun the Plan 1 primary linear models (T1 ridge, T2 ordered logit) with S + naive N
against S, and the event study with naive stance. Report RMSE, the change against S, Clark-West p and
DM p for both pipelines side by side. Interpretation: a larger apparent gain with the naive pipeline is
evidence of look-ahead inflation, which the time-consistent design removes.

## 4. Market benchmark (descriptive)

As in the Plan 4 spec, section 3: the 4-week Treasury bill (`DTB4WK`, the 3-month bill `DTB3` before
2001-07-31) plus a spread estimated on training rows only, for T1 and T2. Reported next to no change,
linear S, S+N and S+M with one-sided DM p-values.

## 5. Ablation (exploratory)

Feature groups: Fed stance (`n_stance`, `n_stance_chg`); Beige Book tone (`n_beige`); newspaper indices
(`n_news`, `n_gpr`, `n_epu`). For T1 and T2, the primary linear model with S + one group, and with S+N
minus one group. Report RMSE, change against S, and Clark-West p against S.

## 6. Code (new modules only)

| file | responsibility |
|---|---|
| `src/fed_news/events.py` | event days, day stance, stance surprise, yield changes (event day and day before) |
| `src/fed_news/event_study.py` | walk-forward OLS forecasts, Clark-West, void rule, secondary tests |
| `src/fed_news/naive_stance.py` | ModernBERT classifier on all labels, document scores, naive N features |
| `src/fed_news/market_bench.py` | bill yields, T1/T2 bill forecasts |
| `src/fed_news/ablation.py` | linear runs per feature group |
| `src/fed_news/evaluate_text.py` | `RESULTS_TEXT.md` with every table and the H5 verdict |

Plan 1 modules are imported, never edited. Tests (offline, synthetic): the surprise uses only documents
before the event day; walk-forward fits use only earlier events (perturbing later outcomes leaves earlier
forecasts unchanged); the weekend/holiday rule; the void-rule outcome is the day before; the bill spread
uses training rows only; ablation column sets.

## 7. Order and compute

1. `PREREG_TEXT.md` and tag `prereg-text`. 2. Event study (minutes, CPU). 3. Market benchmark and
ablation (minutes). 4. Naive stance classifier and scoring (a few GPU hours, background), then the
look-ahead comparison (minutes). 5. `RESULTS_TEXT.md`, then the paper rewrite around the method.

## 8. Out of scope

Intraday data; new label sets; other countries (follow-up paper); changing any pre-registered H1/H2a
setting or result; the Plan 4 replication grid unless requested.
