# Proof of concept: is ChronoGPT a better hawk/dove scorer than ChronoBERT? (exploratory)

Date: 2026-10-12. Status: approved in conversation ("try both"); this file fixes the details before anything runs.
Exploratory; no forecast is made and no pre-registered result changes.

## 1. Why

The statement-change proof of concept stopped at stage 1: on changed sentences with unseen human labels, the
sign of the ChronoBERT-based shift agreed with the human shift in 30 of 44 pairs (0.68 < 0.70). The bottleneck
is the sentence scorer (held-out accuracy about 0.66-0.69). Before any further forecasting work, this test asks
only whether a larger chronologically consistent model scores sentences clearly better.

## 2. Data and split

All TDW/WCB human sentence labels (`labels.load_labels`, 3,074 sentences; dovish, hawkish, neutral), dated by
their earliest containing corpus document. One split, fixed now: training = label year < 2012 (1,722),
test = label year >= 2012 (1,352). Both models are the 2011 vintage (`chrono-bert-v1-20111231`,
`chrono-gpt-v1-20111231`), so neither has seen text from the test years and no test label is used for training
or selection.

## 3. Methods (identical for both models)

Frozen model, mean-pooled hidden states over the sentence's tokens (truncated at 256 tokens), then multinomial
logistic regression with standardisation in a Pipeline. The layer (quarter points: ChronoBERT hidden states 6,
11, 17, 22 of 22; ChronoGPT layer outputs 13, 26, 39, 52 of 52) and C (logspace(-3, 2, 9)) are chosen together
by mean 3-class log loss over 5 stratified folds of the training labels (shuffled, seed 0). The chosen
combination is refitted on all training labels and applied once to the test labels.

ChronoGPT runs from a reviewed, vendored copy of the authors' `ChronoGPT_inference.py` (MIT licence) in
bfloat16; weights are loaded with `torch.load(weights_only=True)`. Tokeniser: GPT-2 BPE.

## 4. Pass rule (fixed before running)

ChronoGPT passes if its 3-class test accuracy is at least 0.75 and at least 0.05 above frozen ChronoBERT's.
Reported as well: macro-F1, recall per class, log loss, the hawkish-vs-dovish sign agreement on the test
sentences labelled hawkish or dovish, and an exact McNemar test (descriptive). For context, the fine-tuned
ChronoBERT classifiers of the pipeline report 0.66-0.69 on random 20% hold-outs (a different and more
favourable test).

If ChronoGPT passes, the next step is a written plan to score all sentences with yearly ChronoGPT vintages and
re-run the statement-change stage 1 with a fresh gate. If it fails, the text-measurement route is reported as
not improved by a larger chronologically consistent model under this budget.

## 5. Limits

One split and one vintage; frozen features only (fine-tuning a 1.5B model does not fit the 6 GB GPU). Labels
are sentence-level and noisy, which caps any scorer's accuracy.

## 6. Compute

Download about 6.2 GB once. Embedding 3,074 sentences: minutes on the RTX 2060 for ChronoBERT, about 10-30
minutes for ChronoGPT. Logistic regression: CPU, minutes.
