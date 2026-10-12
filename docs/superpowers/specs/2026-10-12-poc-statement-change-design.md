# Proof of concept: what changed in FOMC statements (exploratory)

Date: 2026-10-12. Status: design approved in conversation; this file fixes the details before any stage is run.
Exploratory: it does not replace or reinterpret any pre-registered result (PREREG.md, PREREG_H2A.md,
PREREG_TEXT.md, PREREG_STUDY2.md, Study 3).

## 1. Purpose

Earlier text features average the hawkish/dovish tone of all Fed documents in a window. This proof of concept
asks, cheaply and before any long training, whether measuring what *changed* in the FOMC statement carries a
signal about the next decision. Study 3 (document-level classifiers trained on decisions) failed its direction
check, so no model is trained on decisions here and no language model is trained at all.

## 2. Features X (computed at each meeting's cutoff)

Statements: corpus documents of type `statement` released within one day of a decision date in `meetings.pkl`
(scheduled or unscheduled). Other Fed press releases (liquidity facilities, framework reviews) are excluded.
Each statement is compared with the previous such statement.

- `x_shift`. Sentences and their stance are the saved Plan 1 sentence scores (`<data>/stance/<year>.pkl`;
  classifier year = release year, ChronoBERT vintage year - 1). Excluded before comparison: every sentence from
  the first vote, implementation-note, media or web-footer sentence onward, and sentences about the policy
  action itself or the discount rate (regular expressions in `stmt_change.py`). Remaining sentences are
  normalised (month names, digits, whitespace, release-header text) and compared as sets.
  shift = sum of stance of added sentences - sum of stance of removed sentences.
  A meeting's `x_shift` is the sum over the statements released in [previous cutoff, cutoff); 0 if none.
- `x_dissent_n`: number of members voting against, from the latest statement in that window (0 if none, and
  before March 2002, when statements did not report votes).
- `x_dissent_dir`: dissenters preferring tighter policy minus those preferring easier policy, relative to the
  committee's action, by fixed cue rules: "lower/reduce/cut/decrease/ease" or "more/additional accommodation"
  = easier; "raise/increase/tighten/firm" or "less accommodation" or "did not support additional policy
  accommodation" = tighter; "no change/maintain/keep" = tighter after a cut, easier after a hike, no direction
  after a hold. A clause without a cue counts 0.

## 3. Panel

Scheduled meetings only (an unscheduled meeting cannot be known seven days ahead). Cutoff = decision date - 7
days. S and M are computed at that cutoff with `panel._row` (ALFRED vintages published by cutoff - 1 day,
market closes strictly before the cutoff). `p_prev_change` = target before this meeting - target before the
previous scheduled meeting (includes unscheduled moves; known at the cutoff). Target: `y_class` (cut / hold /
hike, +/-0.125 band). Rows with a missing S or M value are dropped; every model uses the same rows.

## 4. Model

Multinomial logistic regression (scikit-learn LogisticRegression, lbfgs, L2) with StandardScaler inside a
Pipeline, so scaling is fitted on training folds only. C is chosen from logspace(-3, 2, 9) by mean log loss over
the 5 expanding inner folds of `models.expanding_folds` inside the training rows; folds with fewer than 20
training rows are skipped. No class weights. Probabilities are mapped to (cut, hold, hike) with absent classes
at 0, clipped at 1e-6 and renormalised. Refit at every test meeting on all earlier meetings.

## 5. Comparisons (identical rows, grid, folds, training procedure)

- P1: S + P vs S + P + X (beyond economic statistics)
- P2: S + M + P vs S + M + P + X (beyond statistics and markets)

P = `p_prev_change`, included in both arms so that the text does not get credit for rate momentum.
Descriptive only: the same pairs without P; a placebo (X columns permuted within the training rows, the test
row's X taken from a random training row); climatology (expanding class frequencies) and always-hold.

## 6. Stages and pass rules (fixed before any stage runs)

**Stage 1 - measurement (no forecast outcome is used).**
- 40 statement pairs with at least one changed sentence, sampled with seed 0 from all pairs, shown without
  dates or scores in random order. The user labels each H (more hawkish), D (more dovish) or N (no clear shift).
- Pass: among pairs labelled H or D, sign(`x_shift`) agrees with the label on at least 70% (a zero shift counts
  as disagreement) and above the share of the more common of H and D among them.
- Dissent: 20 statements with a "Voting against" passage, seed 0. The parsed count must equal the count in the
  text for all 20, and the direction must follow the rule in section 2 for all 20.
- If either part fails, stop and report.

**Stage 2 - selection period: test meetings 2001-01-01 to 2015-12-31.**
- Runs only after the stage 1 pass file exists.
- Pass if both: Spearman(`x_shift`, `y_change`) > 0 over the stage 2 test rows, and mean log loss with X is
  below that without X in at least one of P1, P2.
- Reported: class counts, accuracy, macro-F1, recall per class, log loss, Brier score, ECE (5 bins on the
  largest probability), and a paired moving-block bootstrap (block 8 meetings, 10,000 draws, seed 0) 95%
  interval for the mean log-loss difference. No p-value decides anything at stage 2.

**Stage 3 - test meetings 2016-01-01 to 2026-09-16, once.** Only pairs that passed stage 2. A written
pre-registration and git tag come first. One-sided Diebold-Mariano on per-meeting log loss with a Newey-West
variance (lag 4) and the HLN correction, plus the bootstrap interval; Holm across the pairs tested. This
period was used in earlier studies with other features, so it is not untouched data.

## 7. Known limitations

- Adjacent statements across a year boundary are scored by different yearly classifiers, so an unchanged
  sentence is never compared with itself under two classifiers, but added and removed sentences can be.
- The saved sentences are lowercased and split at initials, which fragments vote lists; the truncation rule
  in section 2 removes them.
- Before May 1999 statements were issued only when policy changed (training rows only).
- Statements are public at release and priced by markets, so P2 is expected to be hard.
- Stage 1 labels come from the user, who knows FOMC history; dates are hidden, but content can reveal the era.

## 8. Compute

CPU only. Feature building and each stage take minutes; the stage 1 labelling takes the user about 45 minutes.
