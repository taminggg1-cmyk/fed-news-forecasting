# Design: News + Statistics Forecasts of the Federal Funds Rate, Without Lookahead

Date: 2026-10-08. Revision 2 (adds the longer sample, geopolitical and newspaper news, the M
block, TFT, and an LLM news reader). Status: awaiting the user's review.

## 1. Purpose

**What the user asked for.** Predict what Chan-Lau, Quach and Shi (2025, "Forecasting Federal
Funds Rates with AI", AMRO WP/25-10, SSRN 5401236) predict, with a "news + statistics"
architecture that produces a positive result for the user's professor. Revision 2 requests: a
longer sample; the paper's data plus our own; geopolitical and other news; TFT; a modern LLM
(Qwen, DeepSeek or Gemma) in case ChronoBERT cannot read news. Only free data is available, and
the GPU is an RTX 2060 with 6 GB.

**What "works" means here (assumption the user accepted).** On the paper's own task, adding
news to the paper's own statistics improves out-of-sample forecasts, under a significance test
fixed before the run, with every input point-in-time (no lookahead).

**What it does not mean.** It does not mean beating market prices. `fedfunds_gpt2` showed a
six-feature yield-curve logit predicts 0.911 of decisions and FOMC text adds nothing beyond
it. That comparison is kept (block M) and expected to be null.

## 2. Why this design

Four earlier projects (THB, `fedfunds_gpt2`, `gold_gpt2`, stock round S) found the same thing:
text loses once the benchmark contains prices that have already absorbed the text. So the
primary comparison is the one the paper makes, statistics against statistics plus news, and the
work fixes the paper's three weaknesses:

1. **Lookahead.** The paper's LLM is Claude 3.5 Sonnet, trained on text written after the
   forecast dates; the paper concedes this (footnote citing Sarkar and Vafa 2024). Primary text
   features here come only from ChronoBERT and ChronoGPT vintages trained on text before the
   forecast year (He, Lv, Manela and Wu; `manelalab/*` on Hugging Face, yearly 1999-2024).
2. **No real text.** The paper's LLM receives number tables and a prompt. Here models read the
   Fed's documents, newspaper headlines, and newspaper-based indices.
3. **Weak benchmarks and revised data.** The paper reports no random walk and no market
   benchmark, and uses revised data. Here macro data are real-time ALFRED vintages, and the
   ladder includes a random walk, a Taylor rule and the market block.

Modern LLMs (Qwen2.5) and TFT are included, but as secondary analyses: a modern LLM's knowledge
of later events would reproduce the paper's flaw, and TFT has overfit every small sample tried
so far (dates memorised on gold with 2,517 days; LSTM lost to a logit on 115 meetings).

## 3. Hypotheses

- **H1 (primary).** On both tasks, statistics plus news (S+N) forecasts better than statistics
  alone (S) with the primary model.
- **H2 (secondary, pre-registered, not part of the win rule).**
  a. H1 holds with the paper's LSTM/GRU, with TFT, and with the linear-anchor + TFT-correction
     hybrid.
  b. H1 holds with ChronoGPT zero-shot scores in place of the stance classifier.
  c. S+N beats the random walk.
  d. News adds to statistics plus market data: S+M+N against S+M (expected null).
  e. The paper's LLM result is partly lookahead.
  f. An LLM news reader (Qwen2.5-7B-Instruct, with ChronoGPT-instruct as its lookahead-free
     twin) improves on S, and its edge over the twin is larger before its release date than
     after it.
  g. Telling the modern LLM to use only information up to the forecast date (the paper's own
     safeguard, and the user's proposal) does not remove its lookahead.

## 4. Tasks

### Sample

All series start in January 1994, the month the FOMC began issuing statements (the paper
starts in 1998). Rows dated 1994-1999 are only ever training rows.

### T1: the paper's RNN task (monthly)

- Target: FRED `FEDFUNDS` (monthly average effective rate) in month t.
- Timing, following the paper: models are fit on rows ending t-2 and predict month t from
  inputs available at the end of month t-1.
- Primary test window: January 2001 to the latest complete month (about 308 months).
  Comparison window: January 2016 to April 2025 (the paper's 112 months), reported, not tested
  for the win rule.
- Metrics: MSE, RMSE, MAE, MBE (the paper's Table 2 metrics).
- Point forecast for change models: `FEDFUNDS[t-1] + predicted change`.

### T2: the paper's LLM task (per FOMC meeting)

- Target: the federal funds target set at the meeting, as the midpoint of the target range
  (`DFEDTAR` spliced to the `DFEDTARL`/`DFEDTARU` midpoint, reusing `fedfunds_gpt2`).
- Class label: cut / hold / hike from the sign of the change.
- Meeting set: every minutes-link date on the Fed's calendar pages from 1994 (one per scheduled
  meeting; dates within two days merged to the later one), plus every target change that is not
  within three days after one of those dates (an unscheduled move), dated by its statement when a
  statement link falls on the change date or the day before. Statement-only dates are not
  meetings: on the calendar pages they include press releases without a rate decision (liquidity
  facilities, swap lines, asset purchases, the 2020 framework statement). This differs from
  `fedfunds_gpt2/data/meetings.pkl`, which counted those releases as hold decisions.
- Cutoff: inputs dated strictly before the first day of the meeting, implemented as the day
  before the decision date (exact for two-day meetings, one day conservative for one-day ones).
- Primary test window: meetings from January 2001 to the latest meeting (about 210 meetings).
  Comparison window: the paper's January 2016 to April 2025.
- Metrics: MSE, RMSE, MAE, MBE on the rate, plus accuracy and macro-F1 on the class.

Both S and S+N train on identical rows, so every comparison is paired.

## 5. Data (all free)

### S: the paper's statistics (Table 1)

Lagged policy rate (T1: last published `FEDFUNDS`; T2: standing target), M2, PCE and core PCE
price index, CPI and core CPI, change in nonfarm payrolls, average weekly earnings of production
and nonsupervisory workers, nominal GDP, U-6, FHFA purchase-only house prices, 10y-2y slope,
1y-6m slope, with the paper's transformations. The Green Street commercial property index has
no free equivalent and is dropped; the CRB spot index is replaced by `PPIACO`, as in
`fedfunds_gpt2`.

Average weekly earnings of production and nonsupervisory workers is FRED `CES0500000030`. ALFRED
is read through the FRED API, which needs a free API key from the user.

Real-time rule: each value is the latest ALFRED vintage released strictly before the cutoff.
Where ALFRED has no vintage for a date, the earliest available vintage is used with the release
lag from `fedfunds_gpt2/src/build_meetings.py`, and the coverage table records it.

### M: our additional statistics

The six `fedfunds_gpt2` yield-curve features (3-month bill, 6-month, 1-year and 2-year yield
spreads to the standing target, target level, zero-lower-bound flag), VIX, and WTI oil
(`DCOILWTICO`), each as of the cutoff. Used only in H2d and the market ceiling.

### N: news

| source | content | where |
|---|---|---|
| Trillion Dollar Words (Shah, Paturi and Chava, ACL 2023) | FOMC minutes, speeches and press-conference transcripts 1996 to October 2022, with human hawkish/dovish/neutral sentence labels | gtfintechlab GitHub / Hugging Face (CC BY-NC 4.0) |
| Fed documents outside that corpus | minutes 1994-1995 and after October 2022, Board speeches and press conferences after October 2022 | federalreserve.gov |
| FOMC statements | 1994 onward | `fedfunds_gpt2/data/fomc_statements.jsonl` (from 1998) plus federalreserve.gov (1994-1998) |
| Beige Book | regional business reports, two weeks before each meeting | federalreserve.gov, Minneapolis Fed archive |
| SF Fed Daily News Sentiment Index | economic news sentiment, 24 US newspapers, daily since 1980 | frbsf.org CSV |
| Geopolitical Risk Index, daily (Caldara and Iacoviello) | newspaper counts of geopolitical tension, daily since 1985 | matteoiacoviello.com; reuse `gold_gpt2/src/fetch_gpr.py` |
| Economic Policy Uncertainty, daily (Baker, Bloom and Davis) | newspaper-based policy uncertainty, daily since 1985 | policyuncertainty.com |
| Guardian headlines (H2f only) | headline and trail text from the business, us-news and world sections, filtered by a fixed keyword list (Fed, interest rates, inflation, jobs, recession, banks, oil, war, sanctions, tariffs, terrorism, pandemic), 1999 onward | Guardian Open Platform; key in `gold_gpt2/secrets/guardian_key.txt`; reuse `gold_gpt2/src/fetch_guardian.py` |

Every document carries a date. Statements are dated by the date in their own text header, or by
the calendar link date if that is later (one 2007 statement is linked under the wrong date).
Labelled sentences come from Trillion Dollar Words (2,480) and the 1,000 Federal Reserve sentences
of the World Central Banks dataset (`gtfintechlab/federal_reserve_system`, CC BY-NC-SA 4.0), which
use the same three stance labels; duplicates are removed. The datasets' `year` fields are NOT
document years (for Trillion Dollar Words they match the source document's year in about 5% of
cases), so each label is dated by the earliest release date of a corpus document that contains it,
and unmatched labels are dropped. The classifier for year y trains on labels dated before
January 1 of year y. Regional
Reserve Bank presidents' speeches are out of scope (Section 13).

## 6. Primary text features

### Vintage rule

A row dated in year y (y >= 2000) uses language models trained through December 31 of year y-1.
Rows in 2025 and 2026 use the 2024 vintage. Rows dated 1994-1999 use the 1999 vintage, the
earliest that exists. Those rows are training-only, so every test row obeys the rule strictly;
the exception is stated in the paper. Each row carries the features computable on its own date.

### Stance classifier

For each year y from 2000 onward: fine-tune `manelalab/chrono-bert-v1-{y-1}1231` with a
three-class head (dovish, hawkish, neutral) on Trillion Dollar Words labels whose source
document is dated before January 1 of year y. Rows dated 1994-1999 use the year-2000
classifier. Fixed settings: learning rate 2e-5, 3 epochs, batch 16, maximum length 256, three
seeds with averaged probabilities. A random 20% of the eligible labels is held out to report
accuracy and macro-F1 per vintage.

Sentence stance is `P(hawkish) - P(dovish)`. Document stance is the mean over its sentences.

### Features (six, fixed)

| | feature | T2 definition (window: since the previous meeting's start date) | T1 definition |
|---|---|---|---|
| F1 | Fed stance | mean of three group means: statement plus press conference, minutes, speeches (empty groups skipped); carried forward if the window has no documents | same, over month t-1; carried forward if month t-1 has no documents |
| F2 | stance change | F1 minus F1 of the previous meeting | F1(t-1) minus F1(t-2) |
| F3 | Beige Book tone | Loughran-McDonald net tone `(pos - neg) / (pos + neg)` of the latest Beige Book before the cutoff | same, as of the end of month t-1 |
| F4 | newspaper sentiment | SF Fed index, mean over the 30 days ending three days before the cutoff | same, ending three days before the end of month t-1 |
| F5 | geopolitical risk | log daily GPR, mean over the same 30 days (days with GPR <= 0 treated as missing) | same |
| F6 | policy uncertainty | log daily EPU, mean over the same 30 days (days with EPU <= 0 treated as missing) | same |

## 7. Models and results ladder

All models are refit walk-forward on an expanding window. Features are standardised on training
rows only. Penalties are chosen by five-fold expanding-window cross-validation inside each
training window.

### Primary models

- **T2:** penalised ordered (cumulative) logit for cut < hold < hike, written with
  `scipy.optimize`, L2 penalty on slopes only, grid {0.01, 0.03, 0.1, 0.3, 1, 3}, refit every
  meeting. The rate forecast is the standing target plus the probability-weighted mean move, with
  mean move sizes per class from the training window (hold = 0). The class forecast is the most
  probable class.
- **T1:** ridge regression on the monthly change in `FEDFUNDS`, penalty grid of 7 log-spaced
  values from 0.1 to 1000, refit every month.

### Secondary architectures (H2a), 8 seeds each

- **Paper replication (T1):** LSTM (12 steps, 240 units, learning rate 1e-4, 300 epochs) and GRU
  (12 steps, 188 units, learning rate 2e-4, 449 epochs) predicting the level, refit monthly as in
  the paper. If one seed-pass takes longer than one hour, refit every 3 months and record the
  deviation in `RESULTS.md`.
- **Meeting LSTM (T2):** lookback 4 meetings, refit every 8 meetings.
- **TFT:** `gold_gpt2/src/tft.py`, hidden size 32, dropout 0.2, lookback 12 months (T1) or
  8 meetings (T2), early stopping on the last 20% of each training window, refit every 12 months
  (T1) or 8 meetings (T2). It predicts the change in the rate.
- **Hybrid (the `gold_gpt2` V3c design):** the primary linear model is the anchor; a TFT with
  the same settings is trained on the anchor's residuals, computed out-of-fold by expanding-window
  cross-validation inside the training window. Forecast = anchor + correction.

### Ladder

| row | content | T1 | T2 |
|---|---|---|---|
| 0 | random walk (T1) / always-hold (T2) | yes | yes |
| 1 | paper's published LSTM/GRU (Table 2) and LLM (Table 3), LLM flagged for lookahead | yes | yes |
| 2 | S, neural: paper's LSTM/GRU (T1); meeting LSTM (T2) | yes | yes |
| 3 | inertial Taylor rule: OLS of the rate on its lag, real-time core PCE inflation and real-time U-3, walk-forward | yes | yes |
| 4 | S, primary model | yes | yes |
| 5 | **S+N, primary model (H1: row 5 against row 4)** | yes | yes |
| 6 | S+N, neural (H2a: row 6 against row 2) | yes | yes |
| 7 | S and S+N, TFT; S and S+N, hybrid (H2a) | yes | yes |
| 8 | placebo: S plus N with the text features permuted across rows within each training window | yes | yes |
| 9 | S+M and S+M+N, primary model (H2d; S+M is the market ceiling) | yes | yes |
| 10 | S plus LLM-reader features, Qwen and ChronoGPT-instruct (H2f) | yes | yes |

Continuous forecasts are turned into classes with a ±0.125 percentage-point band.

## 8. Evaluation and win rule

All of this is committed in `PREREG.md` and tagged before any forecast that uses N is made on a
test window. Each primary test is run once.

**H1 is supported only if all three hold:**
1. T1, 2001 onward: S+N beats S on squared error, Harvey-Leybourne-Newbold corrected
   Diebold-Mariano, horizon 1, one-sided p < 0.05.
2. T2, 2001 onward: S+N beats S on squared rate error, same test, one-sided p < 0.05.
3. T2, 2001 onward: S+N beats S on class correctness, exact McNemar on discordant pairs,
   one-sided p < 0.05.

Requiring all three together means no multiplicity correction is needed for H1. Secondary
analyses use the same tests, are reported with Holm-adjusted p-values across the H2 family,
and cannot rescue a failed H1.

**Voiding rule.** If the placebo (row 8) also beats S at p < 0.05 on any of the three tests,
the H1 result is void and reported as a leakage or overfitting signal. The same holds for the
shuffled-label control (S+N columns, training targets permuted within each training window): if it
beats the random walk / always-hold at p < 0.05 on any of the three tests, H1 is void.

**Variance rule.** Every neural configuration is reported as mean ± standard deviation over
8 seeds. A neural comparison counts only if its gap exceeds two seed standard deviations.

**Null rule.** If H1 fails, the result is reported as a null. The target, windows and features
are not changed after seeing results.

## 9. Secondary analyses in detail

- **H2b, ChronoGPT scorer:** `manelalab/chrono-gpt-v1-{y-1}1231` conditioned on each statement
  and minutes document, scoring length-normalised log-likelihoods of "decided to raise / lower /
  maintain the target range", replacing F1 and F2.
- **H2e, the paper's LLM, re-run:** the paper's prompt ("As an AI central bank advisor, review the
  past n months of U.S. economic data and predict the Fed Funds Target Rate...", n = 12, same
  output format) with the S table, given to `manelalab/chrono-gpt-instruct-v1-{y-1}1231` and to
  `Qwen/Qwen2.5-7B-Instruct` (4-bit), 10 replications each at temperature 0.7.
- **H2f, LLM news reader:** documents are split into chunks that fit the smaller model's context
  window, with explicit years and month-year dates replaced by placeholders; the same chunks go to
  both models. Fed documents are read chunk by chunk; Guardian headlines are read as one batch of
  up to 30 per day. Each prompt asks whether, at the next FOMC meeting, the Committee will raise,
  lower or maintain the target rate, and the answer is scored by the log-probabilities of the three
  answer words (no free generation). Chunk scores are averaged into document and day scores, then
  over the row's window into two features: `P(raise) - P(lower)` and `1 - P(maintain)`.
- **H2g, instruction arm:** every Qwen prompt in H2e and H2f is run twice, plain and with the
  instruction "Today is {date}. You must use only information available up to {date}. Do not
  use any knowledge of events after {date}." Both arms get the lookahead diagnostic below.
- **Recall audit (before any Qwen feature is used):** for every test meeting, ask Qwen the
  target-range upper bound in force after that meeting, with no context. Report recall accuracy
  by year against the always-same-as-before baseline. Years where recall beats that baseline are
  flagged as memorised in every Qwen table (method of Lopez-Lira, Tang and Zhu 2025).
- **Lookahead diagnostic (H2e, H2f, H2g):** for each model pair, the modern model's error
  advantage over the ChronoGPT twin on test rows before 2024-09-19 (Qwen2.5's release) minus its
  advantage after it. A positive gap is evidence of lookahead. Qwen results are always reported
  next to this diagnostic and the recall audit.
- **Pilot evidence (2026-10-08, throwaway, five cases, Qwen2.5-3B-Instruct):** the model's
  recall of past rates was mostly wrong (3 of 5 off by more than a point), so a small model has
  little to leak. In the one case it knew (2022 hikes, recall 4.25), the instructed forecast made
  "as of December 1, 2021" with no current rate given was 4.50, the exact outcome, against a
  then-current 0.25. When the current rate was supplied it anchored on it (0.50). The instruction
  did not prevent the leak where the knowledge existed.

## 10. Code layout and run order

Repository `C:\Users\User\fed_news`. Large data under `D:/fed_news`. Models from `D:/hf_cache`.
Python: the CUDA virtual environment at `D:/gold_lm/venv`.

| step | file (`src/fed_news/`) | output |
|---|---|---|
| 1 | `fetch_macro_realtime.py` | ALFRED vintages with release dates; M block series |
| 2 | `fetch_target.py` (copied from `fedfunds_gpt2`) | daily target rate |
| 3 | `fetch_text.py` | Fed corpus: doc id, type, speaker, date, text |
| 4 | `fetch_indices.py` | SF Fed sentiment, GPR, EPU |
| 5 | `fetch_guardian.py` (adapted from `gold_gpt2`) | dated headlines (H2f) |
| 6 | `build_panel.py` | T1 and T2 rows from 1994, S and M features, cutoffs |
| 7 | `stance_train.py` | one classifier per year, with held-out accuracy |
| 8 | `text_features.py` | F1-F6 per row |
| 9 | `run_models.py` | ladder rows 0-9 |
| 10 | `evaluate.py` | tests, tables in the paper's layout, `RESULTS.md` |
| 11 | `chrono_scorer.py`, `paper_llm_rerun.py`, `llm_reader.py` | H2b, H2e, H2f, row 10 |

Shared pieces: `config.py` (paths, windows, seeds, grids), `models.py` (ordered logit, ridge
wrapper, RNNs, TFT and hybrid wrappers), `stats_tests.py` (DM-HLN, McNemar, Holm).

The implementation plans fix the final module names (for example `alfred.py` for step 1 and
`fomc_calendar.py` plus `corpus.py` for step 3); the step order above is unchanged. Plan 1 covers
H1 with rows 0, 3-5, 8 and 9; Plan 2 the neural rows (2, 6, 7); Plan 3 the language-model
analyses (H2b, H2e, H2f, H2g, row 10).

## 11. Testing and failure handling

- **Coverage table** of documents per type per year. If any test year lacks statements or
  minutes, stop and report before modelling.
- **pytest unit tests:**
  - no feature uses a document, index value or data vintage dated on or after its row's cutoff;
  - every language-model vintage year is earlier than its row's year for rows from 2000 on;
  - columns starting `y_` never enter a feature matrix;
  - the ordered logit at zero penalty matches `statsmodels` `OrderedModel`;
  - the hybrid's residuals are out-of-fold (no residual from a row the anchor was fit on);
  - DM-HLN, McNemar and Holm match hand-computed values.
- **Leak controls:** placebo row 8 and a shuffled-label run, which must fall to chance.

## 12. Risks

- **Power.** About 308 test months and 210 test meetings. A small true gain can still fail the
  tests; that outcome is reported as a null, not retried.
- **ALFRED coverage** may not reach back to 1994 for every series; the fallback is in Section 5.
- **Label dating.** If many Trillion Dollar Words sentences cannot be matched to documents, early
  vintages train on few labels; per-vintage accuracy is reported.
- **SF Fed index revisions.** The published history may have been recomputed; the three-day lag
  is a partial guard and the limitation is stated.
- **GPU.** Qwen2.5-7B in 4-bit needs bitsandbytes on Windows within 6 GB. If it does not load,
  fall back to the cached `Qwen/Qwen2.5-3B-Instruct` and record the change.
- **Guardian quota.** The developer key allows a limited number of calls per day; the headline
  fetch may take several days and runs in the background while the primary analysis proceeds.
- **Speech coverage after October 2022** is Board speeches only.

## 13. Out of scope

- Regional Reserve Bank presidents' speeches.
- The Fed's numeric projections (SEP dot plot).
- Beating market prices as a primary goal (tested only as H2d).
- An out-of-country exam (for example another central bank); possible follow-up.
- Earnings-call transcripts.
