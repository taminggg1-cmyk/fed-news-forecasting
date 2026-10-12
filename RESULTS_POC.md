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

Design: `docs/superpowers/specs/2026-10-12-poc-scorer-design.md`. Both models are the 2011 vintage, frozen;
mean-pooled hidden states of one layer feed a standardised multinomial logistic regression. Layer and penalty
were chosen by cross-validation on the 1,722 labels dated before 2012; the 1,352 labels from 2012 on were scored
once (class shares: neutral 0.42, dovish 0.32, hawkish 0.26).

**Result: not passed** (rule: ChronoGPT accuracy at least 0.75 and at least 0.05 above ChronoBERT).

| Test labels 2012+ | ChronoBERT (layer 17 of 22) | ChronoGPT (layer 26 of 52) |
|---|---|---|
| Accuracy (3 classes) | 0.520 | 0.561 |
| Macro-F1 | 0.484 | 0.537 |
| Recall dovish / hawkish / neutral | 0.36 / 0.35 / 0.75 | 0.47 / 0.40 / 0.72 |
| Log loss | 0.956 | 0.936 |
| Hawkish-vs-dovish sign agreement | 0.632 | 0.671 |

ChronoGPT is better on every measure, and the gain is unlikely to be chance (exact McNemar: right where
ChronoBERT was wrong 183 times, the reverse 127 times; one-sided p = 0.0009). But the gain is 0.041, below the
pre-set 0.05 margin, and accuracy is far from 0.75.

Reading: on later years, a frozen chronologically consistent model reaches about 0.52-0.56 on this
three-class task, below the 0.66-0.69 that the pipeline's fine-tuned classifiers report on random hold-outs
(an easier test, since it shares years with training). A larger model helps a little; it does not lift the
scorer to the level at which change-based text features passed the stage 1 check. Fine-tuning the 1.5B model
might do better but does not fit the 6 GB GPU, and this test cannot say how much better it would be.

Operational note: the authors' code precomputes rotary tables for 65,536 positions in every layer (about 1.7 GB
in float32); they were trimmed to the 256 positions used, which leaves every result unchanged (unit test) and
made the run fit in GPU memory.

## 3. Fine-tuned ChronoGPT vs fine-tuned ChronoBERT

Design: `docs/superpowers/specs/2026-10-12-poc-finetune-design.md`. Same split. ChronoBERT fine-tuned exactly as
the pipeline does (`stance.train_and_score`); ChronoGPT's encoder half (26 blocks) fine-tuned with LoRA adapters
(rank 16) and a linear head, base weights frozen in bfloat16. Three seeds each, probabilities averaged.

**Result: not passed** (same bar: at least 0.75 and at least 0.05 above ChronoBERT).

| Test labels 2012+ (3-seed ensembles) | Fine-tuned ChronoBERT | LoRA ChronoGPT |
|---|---|---|
| Accuracy (3 classes) | 0.592 | 0.581 |
| Seed accuracies | 0.621, 0.561, 0.559 | 0.554, 0.561, 0.554 |
| Macro-F1 | 0.571 | 0.561 |
| Recall dovish / hawkish / neutral | 0.51 / 0.43 / 0.76 | 0.58 / 0.39 / 0.70 |
| Log loss | 0.993 | 0.965 |
| Hawkish-vs-dovish sign agreement | 0.748 | 0.720 |

The two are level: McNemar finds ChronoGPT right where ChronoBERT was wrong 147 times and the reverse 162 times
(one-sided p = 0.82). Fine-tuning lifts both models over their frozen versions (ChronoBERT 0.520 to 0.592,
ChronoGPT 0.561 to 0.581), and on future years both land near 0.58-0.59 three-class accuracy (chance 0.33,
always-neutral 0.42) and 0.72-0.75 hawkish-vs-dovish sign agreement.

## 4. Conclusion of the proofs of concept

Within these budgets, the text route is limited by how well a look-ahead-free model reads Fed sentences, not by
how the scores are summarised: the change-based features failed their measurement check by one pair, and neither
a larger chronologically consistent model (frozen or fine-tuned) nor fine-tuning lifted the scorer much above
0.6 on later years. The human sentence labels are themselves noisy, which caps any scorer. No further forecasting
run was made. Separately, the pre-registered results show that markets already carry most of what Fed text could
add for the next decision, which a better scorer would not change.

## 5. Can Claude Haiku 5.5 label Fed sentences like the human annotators?

Design: `docs/superpowers/specs/2026-10-12-poc-llm-labels-design.md`. 300 human-labelled sentences dated before
2012, text only, zero-shot, 25 per request (19,559 input and 1,704 output tokens).

**Result: not passed** (rule: three-class agreement with the human labels at least 0.70).

| | |
|---|---|
| Three-class agreement | 0.673 |
| Recall dovish / hawkish / neutral | 0.61 / 0.59 / 0.74 |
| Hawkish-vs-dovish agreement (both directional) | 0.853 |

Confusion (rows human, columns Haiku; dovish, hawkish, neutral): dovish 46 / 6 / 24, hawkish 9 / 41 / 19,
neutral 23 / 17 / 115. Almost all disagreement is about whether a sentence is neutral (83 of 98 errors);
when both call a sentence directional, they agree on the direction 85% of the time. The neutral boundary of
the labels is the fuzzy part, which also caps the trained scorers. LLM labelling is dropped, as fixed in advance.

**Attempt 2** (one change, at the user's request: effort medium with adaptive thinking allowed; fresh sample of
300 other pre-2012 sentences; same bar): **not passed**, agreement 0.657 (recall dovish / hawkish / neutral
0.71 / 0.42 / 0.75; hawkish-vs-dovish agreement 0.888; 90 of 103 errors involve the neutral class). Output was
1,776 tokens for 300 sentences, about the same as attempt 1 (1,704): with adaptive thinking Haiku chose not to
think on this task, so attempt 2 effectively re-measured the same setup on new sentences. Over both samples
(600 sentences) agreement is about 0.665.
