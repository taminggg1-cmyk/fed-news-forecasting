# Design: News + Statistics Forecasts of the Federal Funds Rate, Without Lookahead

Date: 2026-10-08. Status: approved section by section in chat; this document awaits review.

## 1. Purpose

**What the user asked for.** Predict what Chan-Lau, Quach and Shi (2025, "Forecasting Federal
Funds Rates with AI", AMRO WP/25-10, SSRN 5401236) predict, with a "news + statistics"
architecture that produces a positive result for the user's professor. The data used in the
earlier projects (FNSPID, Guardian, GDELT) was judged not good enough. Only free data is
available (no WRDS subscription).

**What "works" means here (assumption the user accepted).** On the paper's own task, adding
news text to the paper's own statistics improves out-of-sample forecasts, with a significance
test fixed before the run, and with every input point-in-time (no lookahead).

**What it does not mean.** It does not mean beating market prices. The earlier project
`fedfunds_gpt2` showed a six-feature yield-curve logit predicts 0.911 of decisions and that
FOMC text adds nothing beyond it. That result stays in every table as a ceiling.

## 2. Why this design

Four earlier projects (THB, `fedfunds_gpt2`, `gold_gpt2`, stock round S) found the same thing:
text loses once the benchmark contains prices that have already absorbed the text. So the
comparison here is the one the paper actually makes, statistics against statistics plus text,
and the work fixes the paper's three weaknesses:

1. **Lookahead.** The paper's LLM is Claude 3.5 Sonnet, trained on text written after the
   forecast dates. The paper concedes this (its footnote citing Sarkar and Vafa 2024). Here every
   language model is a ChronoBERT or ChronoGPT vintage trained only on text before the forecast
   year (He, Lv, Manela and Wu; `manelalab/*` on Hugging Face, yearly vintages 1999-2024).
2. **No real text.** The paper's LLM receives tables of numbers and a prompt. Here it receives
   the Fed's own documents and newspaper-based indices.
3. **Weak benchmarks and revised data.** The paper reports no random walk and no market
   benchmark, and uses revised macro data. Here macro data are real-time ALFRED vintages and the
   ladder includes a random walk, a Taylor rule and the market ceiling.

## 3. Hypotheses

- **H1 (primary).** On both tasks below, statistics plus news (S+N) forecasts better than
  statistics alone (S) with the same model.
- **H2 (secondary, pre-registered, not part of the win rule).**
  a. H1 holds with the paper's LSTM/GRU architecture as well.
  b. H1 holds with ChronoGPT zero-shot scores in place of the stance classifier.
  c. S+N beats the random walk.
  d. Text adds to the market model (expected null; reported either way).
  e. The paper's LLM result is partly lookahead: a modern LLM beats a vintage LLM by more before
     its release date than after it.

## 4. Tasks

### T1: the paper's RNN task (monthly)

- Target: FRED `FEDFUNDS` (monthly average effective rate) in month t.
- Timing, following the paper: models are fit on rows ending t-2 and predict month t from
  inputs available at the end of month t-1.
- Primary test window: January 2016 to April 2025 (112 months), the paper's window. Secondary:
  extended to the last complete month.
- Metrics: MSE, RMSE, MAE, MBE (the paper's Table 2 metrics).
- Point forecast for change models: `FEDFUNDS[t-1] + predicted change`.

### T2: the paper's LLM task (per FOMC meeting)

- Target: the federal funds target set at the meeting, as the midpoint of the target range
  (`DFEDTAR` spliced to the `DFEDTARL`/`DFEDTARU` midpoint, reusing `fedfunds_gpt2`).
- Class label: cut / hold / hike from the sign of the change.
- Meeting set: identical to `fedfunds_gpt2/data/meetings.pkl` (235 meetings, scheduled and
  unscheduled), so the market ceiling is directly comparable.
- Cutoff: inputs dated strictly before the first day of the meeting.
- Primary test window: meetings from January 2004 to the latest meeting (191 meetings), chosen
  for statistical power. Secondary: the paper's window, January 2016 to April 2025.
- Metrics: MSE, RMSE, MAE, MBE on the rate, plus accuracy and macro-F1 on the class.

### Training rows

Text features start in 2000 because the first vintage is 1999. Both S and S+N therefore train
on rows from January 2000 onward, so the paired comparison uses identical rows.

## 5. Data (all free)

### S: the paper's statistics (Table 1)

Lagged policy rate (T1: last published `FEDFUNDS`; T2: standing target), M2, PCE and core PCE
price index, CPI and core CPI, change in nonfarm payrolls, average weekly earnings of production
and nonsupervisory workers, nominal GDP, U-6, FHFA purchase-only house prices, 10y-2y slope,
1y-6m slope, with the paper's transformations. Two Haver series have no free equivalent: the
Green Street commercial property index is dropped and the CRB spot index is replaced by
`PPIACO`, as in `fedfunds_gpt2`.

Real-time rule: each value is the latest ALFRED vintage released strictly before the cutoff.
Where ALFRED has no vintage for a date, the earliest available vintage is used with the release
lag from `fedfunds_gpt2/src/build_meetings.py`, and the coverage table records it.

### N: news text and text-derived indices

| source | content | where |
|---|---|---|
| Trillion Dollar Words (Shah, Paturi and Chava, ACL 2023) | FOMC minutes, speeches and press-conference transcripts 1996 to October 2022, with human hawkish/dovish/neutral sentence labels | gtfintechlab GitHub / Hugging Face (CC BY-NC 4.0) |
| Fed documents after October 2022 | minutes, Board speeches, press-conference transcripts | scraped from federalreserve.gov |
| FOMC statements | 235 statements | `fedfunds_gpt2/data/fomc_statements.jsonl` |
| Beige Book | regional business reports, two weeks before each meeting | federalreserve.gov (Minneapolis Fed archive for older issues) |
| SF Fed Daily News Sentiment Index | economic news sentiment from 24 US newspapers, daily since 1980 | frbsf.org CSV |
| Economic Policy Uncertainty, daily | newspaper-based uncertainty | policyuncertainty.com (robustness only) |

Every document carries a date. Trillion Dollar Words labelled sentences are dated by exact text
match to their source document; sentences that do not match are dropped from classifier
training. Regional Reserve Bank presidents' speeches are out of scope (Section 11).

## 6. Text features

### Vintage rule

A row dated in year y uses language models trained through December 31 of year y-1. Rows in
2025 and 2026 use the 2024 vintage. Every row, including rows inside training windows, carries
the features that were computable on its own date.

### Stance classifier

For each year y from 2000 onward: fine-tune `manelalab/chrono-bert-v1-{y-1}1231` with a
three-class head (dovish, hawkish, neutral) on Trillion Dollar Words labels whose source
document is dated before January 1 of year y. Fixed settings: learning rate 2e-5, 3 epochs,
batch 16, maximum length 256, three seeds with averaged probabilities. A random 20% of the
eligible labels is held out to report accuracy and macro-F1 per vintage.

Sentence stance is `P(hawkish) - P(dovish)`. Document stance is the mean over its sentences.

### Features (four, fixed)

| | feature | T2 definition (window: since the previous meeting's start date) | T1 definition |
|---|---|---|---|
| F1 | Fed stance | mean of three group means: statement plus press conference, minutes, speeches (empty groups skipped) | same, over month t-1; carried forward if month t-1 has no documents |
| F2 | stance change | F1 minus F1 of the previous meeting | F1(t-1) minus F1(t-2) |
| F3 | Beige Book tone | Loughran-McDonald net tone `(pos - neg) / (pos + neg)` of the latest Beige Book before the cutoff | same, as of the end of month t-1 |
| F4 | newspaper sentiment | SF Fed index three days before the cutoff | same, three days before the end of month t-1 |

## 7. Models and results ladder

All models are refit walk-forward on an expanding window: every month for T1, every meeting
for T2. Features are standardised on training rows only. Penalties are chosen by five-fold
expanding-window cross-validation inside each training window.

- **T2 primary model:** penalised ordered (cumulative) logit for cut < hold < hike, written with
  `scipy.optimize`, L2 penalty on slopes only, grid {0.01, 0.03, 0.1, 0.3, 1, 3}. The rate
  forecast is the standing target plus the probability-weighted mean move, with mean move sizes
  per class taken from the training window (hold = 0). The class forecast is the most probable
  class.
- **T1 primary model:** ridge regression on the monthly change in `FEDFUNDS`, penalty grid of 7
  log-spaced values from 0.1 to 1000.
- **Paper replication (T1):** LSTM (12 steps, 240 units, learning rate 1e-4, 300 epochs) and GRU
  (12 steps, 188 units, learning rate 2e-4, 449 epochs) predicting the level, refit monthly as in
  the paper, 8 seeds each. If one seed-pass takes longer than one hour, refit every 3 months and
  record the deviation in `RESULTS.md`.
- **T2 neural extension:** LSTM on meeting sequences, lookback 4, refit every 8 meetings,
  8 seeds.

| row | content | T1 | T2 |
|---|---|---|---|
| 0 | random walk (T1) / always-hold (T2) | yes | yes |
| 1 | paper's published LSTM/GRU (Table 2) and LLM (Table 3), LLM flagged for lookahead | yes | yes |
| 2 | S, neural: paper's LSTM/GRU replicated (T1); meeting LSTM (T2) | yes | yes |
| 3 | inertial Taylor rule: OLS of the rate on its lag, real-time core PCE inflation and real-time U-3, walk-forward | yes | yes |
| 4 | S, primary model | yes | yes |
| 5 | **S+N, primary model (H1: row 5 against row 4)** | yes | yes |
| 6 | S+N, neural (H2a: row 6 against row 2) | yes | yes |
| 7 | placebo: S plus N with text features permuted across rows within each training window | yes | yes |
| 8 | market ceiling: primary model on the six `fedfunds_gpt2` yield-curve features, with and without N | - | yes |

Continuous forecasts are turned into classes with a ±0.125 percentage-point band.

## 8. Evaluation and win rule

All of this is committed in `PREREG.md` and tagged before any S+N forecast is made on a test
window. Each primary test is run once.

**H1 is supported only if all three hold:**
1. T1, paper window: S+N beats S on squared error, Harvey-Leybourne-Newbold corrected
   Diebold-Mariano, horizon 1, one-sided p < 0.05.
2. T2, 2004 onward: S+N beats S on squared rate error, same test, one-sided p < 0.05.
3. T2, 2004 onward: S+N beats S on class correctness, exact McNemar on discordant pairs,
   one-sided p < 0.05.

Requiring all three together means no multiplicity correction is needed for H1.

**Voiding rule.** If the placebo (row 7) also beats S at p < 0.05 on any of the three tests,
the H1 result is void and reported as a leakage or overfitting signal.

**Variance rule.** Every neural configuration is reported as mean ± standard deviation over
8 seeds. A neural comparison counts only if its gap exceeds two seed standard deviations.

**Null rule.** If H1 fails, the result is reported as a null. The target, windows and features
are not changed after seeing results.

## 9. Secondary analyses

- **H2b, ChronoGPT scorer:** `manelalab/chrono-gpt-v1-{y-1}1231` conditioned on each statement
  and minutes document, scoring length-normalised log-likelihoods of "decided to raise / lower /
  maintain the target range", replacing F1 and F2. Same tests as H1, reported separately.
- **H2e, lookahead measurement:** the paper's prompt ("As an AI central bank advisor, review the
  past n months of U.S. economic data and predict the Fed Funds Target Rate...", n = 12, same
  output format) with the S table, given to `manelalab/chrono-gpt-instruct-v1-{y-1}1231` and to
  `Qwen/Qwen2.5-3B-Instruct` (cached), 10 replications each at temperature 0.7. The lookahead gap
  is the modern model's error advantage on meetings before 2024-09-19 (Qwen2.5's release) minus
  its advantage after it.
- **Robustness:** EPU as a fifth text feature; the extended T1 window; the paper's T2 window.

## 10. Code layout and run order

Repository `C:\Users\User\fed_news`. Large data under `D:/fed_news`. Models from `D:/hf_cache`.
Python: the CUDA virtual environment at `D:/gold_lm/venv`.

| step | file (`src/fed_news/`) | output |
|---|---|---|
| 1 | `fetch_macro_realtime.py` | ALFRED vintages with release dates |
| 2 | `fetch_target.py` (copied from `fedfunds_gpt2`) | daily target rate |
| 3 | `fetch_text.py` | corpus table: doc id, type, speaker, date, text |
| 4 | `fetch_indices.py` | SF Fed sentiment, EPU |
| 5 | `build_panel.py` | T1 and T2 rows with S features and cutoffs |
| 6 | `stance_train.py` | one classifier per year, with held-out accuracy |
| 7 | `text_features.py` | F1-F4 per row |
| 8 | `run_models.py` | ladder rows 0-8, both tasks |
| 9 | `evaluate.py` | tests, tables in the paper's layout, `RESULTS.md` |
| 10 | `chrono_scorer.py`, `paper_llm_rerun.py` | H2b, H2e |

Shared pieces: `config.py` (paths, windows, seeds, grids), `models.py` (ordered logit, ridge
wrapper, RNNs), `stats_tests.py` (DM-HLN, McNemar).

## 11. Testing and failure handling

- **Coverage table** of documents per type per year. If any test year lacks statements or
  minutes, stop and report before modelling.
- **pytest unit tests:**
  - no feature uses a document or data vintage dated on or after its row's cutoff;
  - every language-model vintage year is earlier than its row's year;
  - columns starting `y_` never enter a feature matrix;
  - the ordered logit at zero penalty matches `statsmodels` `OrderedModel`;
  - DM-HLN and McNemar match hand-computed values.
- **Leak controls:** placebo row 7 and a shuffled-label run, which must fall to chance.

## 12. Risks

- **Power.** T2 has 191 test meetings and T1 112 months. A small true gain can still fail the
  tests. That outcome is reported as a null, not retried.
- **ALFRED coverage** may not reach back to 2000 for every series; the fallback is in Section 5.
- **Label dating.** If many Trillion Dollar Words sentences cannot be matched to documents,
  early vintages train on very few labels. Per-vintage accuracy is reported.
- **SF Fed index revisions.** The published history may have been recomputed. The three-day lag
  is a partial guard, and the limitation is stated.
- **Speech coverage after October 2022** is Board speeches only.

## 13. Out of scope

- Regional Reserve Bank presidents' speeches.
- The Fed's numeric projections (SEP dot plot).
- Beating market prices as a primary goal.
- An out-of-country exam (for example another central bank); possible follow-up.
- Earnings-call transcripts (approach C in the brainstorm).
