# Proofs of concept after the pre-registered studies (exploratory)

Date: 2026-10-12. Branch `poc-stmt`. Nothing here changes a pre-registered result; every rule below was
committed before the step it governs ran.

## 1. What changed in FOMC statements

Design: `docs/superpowers/specs/2026-10-12-poc-statement-change-design.md` (sections 9 and 10 are amendments made
before any score or forecast existed).

Question: does measuring what *changed* in the FOMC statement (instead of the average tone of all Fed documents)
carry a signal about the next decision? Features: the stance of sentences added minus sentences removed
relative to the previous statement (saved ChronoBERT sentence scores, no retraining), and the number and
direction of dissents. Three stages, each gated: (1) measurement check, (2) forecasts on 2001-2015,
(3) one confirmation run on 2016-2026.

**Stage 1 result: not passed. Stages 2 and 3 were not run.**

| Check | Rule | Result |
|---|---|---|
| Shift direction vs human labels | at least 70% sign agreement over at least 20 pairs, and above the majority share | 30 of 44 pairs (0.68); majority share 0.50 |
| Dissent count and direction | all 20 sampled statements correct under the rule | 20 of 20 |

Human reference: TDW/WCB sentence labels on changed sentences that the scoring classifier never saw (63 pairs
carry such labels, 44 with a non-zero human shift; files `results/poc_stmt/stage1.json`,
`results/poc_stmt/stage1_human_pairs.csv`). The user could not hand-label the planned 40-pair sheet, so this
source replaced it by amendment (design section 10).

Reading: the agreement is above chance (one-sided binomial p = 0.011 against 0.5), so the measure carries real
information, but it misses the pre-set bar by one pair. The limit is the sentence scorer (held-out accuracy of
the yearly classifiers about 0.66-0.69): with scores this noisy, any summary of Fed text, average or change,
inherits the noise. Moving the bar after seeing 0.68 would make any stage 2 result uninterpretable, so the
study stops here.

The dissent rule leaves direction at 0 when a dissent gives no explicit rate or accommodation cue (9 of the 20
sampled dissenters), so `x_dissent_dir` is a partial measure.

## 2. Is ChronoGPT a better scorer than ChronoBERT?

Design: `docs/superpowers/specs/2026-10-12-poc-scorer-design.md`. Result: pending.
