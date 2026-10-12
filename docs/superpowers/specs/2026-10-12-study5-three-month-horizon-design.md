# Study 5: does Fed communication help forecast the policy path three months ahead? (exploratory)

Date: 2026-10-12. Requested by the user after Study 4. Fixed and committed before any Study 5 code runs on
outcomes. The 2001-2026 meetings were used by every earlier study (at the one-meeting horizon), so Study 5 is
exploratory; its horizon has not been tested before.

## 1. Why this horizon

The day before a meeting, Treasury-market prices already imply the decision (an informal check during Study 4:
a one-line rule on the 6-month spread to the target, with a band chosen by hand, was right on 87.3% of the 212
test meetings, against 86.8% for the full no-news model). The literature finds that market expectations are least
complete at horizons of three months and more: fed funds futures are the best market forecast but their errors
grow with the horizon (Gurkaynak, Sack and Swanson 2007), and text from FOMC materials improved on market
expectations at three to nine months (Fischer, McCaughrin, Prazad and Vandergon 2023, NY Fed Staff Report
1081, using confidential materials released with a five-year lag; we use only public text).

## 2. Rows and target

- One row per **scheduled** FOMC meeting (262 of 270 decisions; the 8 unscheduled decisions are not forecast
  rows, but their rate moves count inside the horizons).
- Forecast made at the cutoff of scheduled meeting t (the day before its decision), as in Plans 1-4.
- Target at horizon h: the federal funds target after scheduled meeting t+h-1 minus the target before meeting t
  (any unscheduled moves in between included). Class: cut if below -0.125, hike if above +0.125, else hold.
- **Primary horizon h = 3** (meetings t, t+1, t+2; about three months). Test rows: scheduled meetings from
  2001-01-01 whose horizon ends by 2026-09-16 (204 rows, to 2026-06-17; 44 cuts, 105 holds, 55 hikes).
  Secondary horizon h = 2 (205 rows). The one-meeting horizon (h = 1) on the same features is run as a
  descriptive reference only.

## 3. Information set: real-time by construction

Every input is the value that existed on the cutoff date; inputs without that guarantee are left out.

- **Economic data**: only series whose ALFRED vintages cover every row: `s_m2_mom`, `s_m2_yoy`,
  `s_payems_chg`, `s_gdp_yoy`, `s_cpi_yoy`, with `s_rate_lag`. Left out: PCE, core PCE, average hourly
  earnings, U6, PPI, core CPI and house prices (their first vintages start between 1996 and 2012). The code
  checks that no kept series uses the ALFRED fallback in any row.
- **Market prices** (not revised): `s_slope_10y2y`, `s_slope_1y6m` and the eight M columns (3-month bill,
  6-month, 1-year and 2-year Treasury spreads to the target, the target level, the zero-bound flag, VIX,
  WTI), values dated strictly before the cutoff.
- **Fed communication**: stance of statements, press conferences, minutes and speeches released in
  [previous scheduled meeting's cutoff, this cutoff), by release date; mean of group means as in Plan 1;
  carried forward when the window is empty; `n_stance_chg` is the change from the previous scheduled row.
  Primary reader **ChronoBERT** (per-year checkpoint and classifier, leak-free). Secondary reader Haiku 5.5
  (the Study 4 labels; memory caveats apply and are larger at longer horizons). Secondary add-on: Beige Book
  tone (computed by us from the text with a fixed dictionary, by release date).
- **Left out entirely**: the SF Fed news sentiment, GPR and EPU indices (2026 vintages; revision behaviour
  unverified).

## 4. Training respects the horizon

- A row's outcome is known only after scheduled meeting t+h-1. The forecast at row t trains on rows j whose
  horizon ended before the cutoff of t (decision date of j+h-1 earlier than cutoff t; for h = 3 this is
  j <= t-3 when no unscheduled meeting intervenes).
- Features standardised on those training rows only.
- Penalty chosen from `config.ORDLOGIT_GRID` by expanding-window CV inside the training rows, with the last
  h-1 training rows before each validation block removed (purged), so no fold trains on outcomes that overlap
  its validation rows.

## 5. Models and comparisons

Penalised ordered logit (`models.OrderedLogit`), walk-forward, refit at every row; class probabilities saved.
Let B = real-time economic data + market prices (no news).

| name | features | role |
|---|---|---|
| B | economic + market | baseline |
| B+C | B + ChronoBERT stance (2 columns) | **primary** |
| placebo-C | B+C, stance columns permuted across training rows | control |
| shuffled-C | B+C, training targets permuted | control |
| B+H | B + Haiku stance | secondary |
| B+C+Beige | B+C + Beige Book tone | secondary |

All at h = 3; B, B+C and B+H also at h = 2; B and B+C at h = 1 (descriptive).

## 6. Tests and verdict

- Loss: per-row log loss of the true class. **Primary test**: one-sided Harvey-Leybourne-Newbold
  Diebold-Mariano with horizon h = 3 (overlapping forecasts), B+C against B.
- **Verdict** (`evaluate.decide`): "void" if placebo-C beats B at p < 0.05 or shuffled-C at p < 0.01; otherwise
  "supported" if the primary p < 0.05, else "not supported".
- Secondary tests (B+H, B+C+Beige at h = 3; B+C and B+H at h = 2) with the same DM test and their own h;
  Holm-adjusted across these four and reported; they do not change the verdict.
- Reported without rules: accuracy, Brier, correct counts by class and at turning points, Newey-West
  (lags 4 and 8) and moving-block bootstrap (blocks of 8 rows, 10,000 draws, seed 0) versions of the primary
  comparison, and the h = 1 reference.
- **Wording fixed now.** Supported: "public Fed communication, read by a leak-free model, improves
  three-month forecasts of the policy path over real-time economic data and Treasury-market prices
  (exploratory; the market baseline is Treasury yields, not fed funds futures)". Not supported: "no detectable
  improvement at three months either".

## 7. Limitations stated before the run

- No fed funds futures (paid data); Treasury yields carry term premia, so the market baseline is weaker than
  futures and the news gain may be overstated relative to a futures benchmark.
- Same 2001-2026 sample as all earlier studies; 204 overlapping three-month horizons, so the effective sample
  is smaller than 204.
- Haiku's memory of decisions (87.8% recall from dates) matters more at longer horizons; Haiku is secondary for
  that reason.
- Human stance labels were made after the events.
- Leaving out seven economic series removes information both models would otherwise share.

## 8. Order of work

1. This document is committed. 2. Code with tests (target, horizon-aware training mask, purged CV, feature
build, fallback check). 3. Run once. 4. RESULTS_STUDY5.md with every number and any deviation.
