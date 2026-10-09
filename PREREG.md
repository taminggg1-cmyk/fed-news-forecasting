# Pre-registration: news + statistics forecasts of the federal funds rate

Date: 2026-10-09. Code commit: 68b5f08.
Design: docs/superpowers/specs/2026-10-08-fed-news-design.md (all sections binding).

## Data as built
- Meetings: 270 from 1995-03-28 to 2026-09-16; unscheduled: 2007-08-10, 2008-03-11, 2010-05-09, 2019-10-11, 2020-03-23, 2020-03-31, 2020-08-27, 2025-08-22, 2007-06-18 (9 non-decision press releases / mislinks excluded).
- T1 common rows: 378, first 1995-03-01; T2 common rows: 260, first 1995-03-28.
- ALFRED first vintages: m2 unknown, pce/pce_core 2000-08-01, ppi/cpi_core 1996-12, earn 2011-03-04, u6 2011-03-04, hpi 2012-05-23; rows using the fallback: T1 137, T2 96 (within test windows; revision lookahead shared by S and S+N).
- Corpus documents by type: statement 240, minutes 263, presconf 94, beige 262, speech 1981 (total 2,840); dropped: 5 speech documents.
- Stance classifier held-out accuracy by year: see results/stance_accuracy.csv (min 0.560 (year 2000), max 0.701 (year 2025), median 0.661).
- Label matching rates: TDW 0.893, WCB 0.983 (gate 0.85); eligible for training: 2000 410, 2001 522, 2005 983, 2010 1522, 2020 2542, 2026 3074.

## Primary hypothesis H1 and win rule
S = the 16 columns in panel.S_COLS. N = n_stance, n_stance_chg, n_beige, n_news, n_gpr, n_epu.
Test windows: T1 2001-01 to 2026-09; T2 meetings 2001-01-25 to 2026-09-16.
H1 holds only if all three are one-sided p < 0.05:
1. T1: S+N beats S on squared error, DM-HLN, h = 1.
2. T2: S+N beats S on squared rate error, DM-HLN, h = 1.
3. T2: S+N beats S on class correctness, exact McNemar.
Void if the placebo beats S at p < 0.05 on any of the three. Void if shuffled beats rw at p < 0.01 on any of the three (leak signal). If H1 fails it is reported as a null; target, windows and features are not changed afterwards.

## Fixed settings
Ordered logit L2 grid {0.01, 0.03, 0.1, 0.3, 1, 3}; ridge grid 7 log-spaced 0.1-1000; 5-fold expanding inner CV; T1 trains on rows up to t-2, T2 on earlier meetings; class band ±0.125.

## Secondary hypotheses (Plans 2 and 3)
H2a-H2g exactly as in spec sections 3, 7 and 9, with Holm adjustment across the H2 family.
They cannot rescue a failed H1.

## Amendments before the tag
- Labels dated by the earliest containing corpus document (statement/minutes/presconf/speech) because TDW year differs from document year; match rates TDW 0.893, WCB 0.983, gate 0.85.
- Statements dated by header ("Release Date: Month D, YYYY") when later than URL date (one: 2007-06-18 moved to 2007-06-28).
- GPR/EPU <= 0 treated as missing before log-transformation.
- Meeting set built from minutes dates (merged within 2 days) plus unscheduled target changes with statement-only rows filtered (9 non-decision press releases excluded).
- Beige Books dated from Fed archive pages by release table date (1996+); 1994-1995 from calendar files.
- Shuffled-label control added (spec §11): training targets permuted within expanding window to test leak presence.

## Known limitations
- ALFRED fallback revision lookahead to 2012 for earn/u6/hpi (identical for S and S+N; shared by both models).
- SF Fed index vintage downloaded 2026-10-09.
- No statements for 1994-1996 (before web publication).
- T1 2004-01 row dropped (PCE vintage 2003-12-23 mostly NaN).
- Stance classifier holdout is sentence-level (same-meeting leakage possible; optimistic accuracy).
- 1994-1999 documents scored by 2000 classifier trained on labels including those years (training-only rows in-sample).
- Beige text type differs by era (full report vs summary).
- TDW/WCB labels created after sample (human annotation of historical text).

## Data and environment
- Data hashes (SHA-256) in results/data_hashes.txt (12 root pickles + 13 alfred/*.pkl): `333cea866038c88175f86f28a8378903c5799ecd117f7c8ef6914f8b48cf88fd`.
- Environment pinned in results/pip_freeze.txt (Python 3.13.7 + torch CUDA, transformers, sklearn, statsmodels, scipy, pandas 3.0, numpy 2.3, bs4, lxml, pypdf): `61bdcd5142bc459f8685e791038287076925c3d9911018fd26dd6e06fb974854`.
