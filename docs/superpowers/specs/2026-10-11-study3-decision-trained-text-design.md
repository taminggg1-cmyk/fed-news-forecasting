# Study 3 design: ChronoBERT trained on Fed decisions, combined with statistics

Date: 2026-10-11. Status: approved in conversation; to be registered on OSF before any real-label run.

## 1. Purpose

The professor asked for a better AI architecture that uses ChronoBERT together with the economic statistics,
compared against the same model without news. Earlier studies used ChronoBERT only through a hawk/dove
classifier trained on about 3,000 human-labelled sentences (held-out accuracy about 0.65); the resulting
stance features gave small, non-significant gains. Study 3 changes one step: the text model learns directly
from the outcome being forecast, the next FOMC decision, using about 6,140 Fed documents.

Success means a clean, registered test of "statistics + decision-trained text" against "statistics only" on
the per-meeting task. A positive result is not assumed.

## 2. Hypothesis H7 (primary, one test)

On the FOMC meetings from 2001-01-03 to 2026-09-16 that have a text signal (expected: all 212), the penalised ordered logit with the 16 real-time
statistics plus the decision-trained text signal (`S+D`) forecasts the target rate better than the same model
with the statistics only (`S`): one-sided Clark-West test on squared target-rate errors, p < 0.05.

**Void** (H7 counts as not supported) if either control passes the same test at p < 0.05 against `S`:

- *Placebo:* `S+D` with the text signal permuted across rows inside each training window
  (`walkforward.Spec(placebo=True)`).
- *Shuffled-label text model:* the whole text pipeline re-run with document labels permuted inside each
  year's training set (seed 0); its signal enters `S+D` in place of the real one.

## 3. Architecture

```
Fed documents (6,140: statements, minutes, press conferences, Board + regional speeches)
   -> ChronoBERT vintage v(y), frozen: token windows of up to 512 tokens, mean-pooled last hidden state,
      document embedding = mean over its windows (768 numbers)
   -> per-year classifier: multinomial logistic regression on standardised embeddings,
      labels = decision at the next FOMC meeting after the document (cut / hold / hike)
   -> document score = P(hike) - P(cut)
   -> meeting signal D = mean of group means of document scores released in [previous cutoff, cutoff)
      (groups: statements + press conferences, minutes, speeches; carried forward when empty)
   -> penalised ordered logit on S + D (existing Plan 1 model and walk-forward engine)
No-news model: the same ordered logit on S only, on exactly the same rows.
```

ChronoBERT stays frozen; fine-tuning it (about 50 GPU hours, high overfitting risk on a few thousand noisy
document labels) is out of scope.

## 4. Details

**Documents.** `stance2.all_docs()`: the Fed corpus types `statement`, `minutes`, `presconf`, `speech` plus
the regional corpus (FRASER, BIS), 6,140 documents, 1994-01 to 2026-10.

**Document label.** The `y_class` (0 cut, 1 hold, 2 hike; +/-0.125 band) of the first meeting in
`meetings.pkl` whose decision date is strictly after the document's release date. Documents with no later
meeting are unlabelled (scored, never trained on).

**Year rule (no look-ahead).** Each document is scored by the classifier of its release year y. That
classifier is trained only on documents whose label meeting date is before 1 January of year y, with
embeddings from ChronoBERT vintage v(y) = min(max(y - 1, 1999), 2024)
(`manelalab/chrono-bert-v1-{v}1231`). Every score is therefore out of sample with respect to its own label.
A year gets a classifier only if its training set has at least 100 documents and at least 10 per class;
documents of earlier years stay unscored, and meetings without a signal are dropped from both models.
Documents released before 2000 use vintage 1999 (a language-model look-ahead that affects training rows only;
all test rows from 2001 use v(y) <= y - 1).

**Embedding.** Tokenise the full text; split into consecutive windows of at most 512 tokens (or the model's
maximum if smaller); mean-pool the last hidden state over each window's tokens; average the windows. fp16
autocast on the GPU; embeddings cached per vintage in `<data>/decision_emb/{v}.npz` with a fingerprint of
document ids, texts and model name.

**Classifier.** scikit-learn `LogisticRegression` (multinomial, lbfgs, `class_weight="balanced"`,
max_iter 2000), embeddings standardised with the training documents' mean and standard deviation. The inverse
penalty C is chosen from {0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1} by 5-fold expanding-window cross-validation
over training documents sorted by label meeting date (`models.expanding_folds`), maximising mean
log-likelihood.

**Meeting signal and forecasting.** The signal is written into the `n_stance` column of a copy of the T2 panel
with `naive_stance.naive_panel("t2", docs)` (as in Study 2), and the models use
`Spec(..., cols=S_COLS + ["n_stance"])` with `walkforward.run_spec` (Plan 1 code, unchanged). `S` and every
other comparison model are re-run on the same rows (`common_rows` of that panel).

## 5. Secondary analyses (reported, not decisive)

1. Decisions: exact McNemar, `S+D` vs `S` (correct cut / hold / hike).
2. DM-HLN, `S+D` vs `S`.
3. With market data: Clark-West, `S+M+D` vs `S+M`.
4. New vs old text: DM-HLN, `S+D` vs `S+N` (the Plan 1 six news features), same rows.
5. Text-model quality: per-year out-of-sample document accuracy and macro-F1 against the majority class of
   the training labels.

## 6. Power statement (before registration)

From the committed Plan 1 T2 forecasts (`S` vs `S+N`, already public), compute the standard deviation of the
squared-error differential and report the mean improvement in MSE that a one-sided test at 5% would detect with
80% power on 212 meetings (2.49 standard errors). This is written into the registration.

## 7. Registration and gating

1. Code is written and unit-tested on synthetic data. Embeddings may be computed before registration (they
   use no outcome).
2. `PREREG_STUDY3.md` (hypothesis, rules, controls, secondary list, power statement) is committed and tagged
   `prereg-study3`.
3. The author registers that file on OSF and records the registration URL in `PREREG_STUDY3_OSF.txt`.
4. The runner refuses to train any classifier on real labels unless the tag exists and the OSF file holds a
   URL starting with `https://osf.io/`.

## 8. Outputs

- `<data>/doc_decision.pkl`: doc_id, type, release, source, year, label, stance (= score).
- `results/study3/`: forecasts, `probe_quality.csv`, `tests.csv`.
- `RESULTS_STUDY3.md`: verdict, tests, controls, secondary analyses, provenance (tag, OSF URL, hashes).
- README and DEVIATIONS updated.

## 9. Compute

About 2-3 GPU hours for embeddings (26 vintages, cumulative document sets, about 0.6 million windows);
classifiers and forecasting take minutes. The shuffled-label control reuses the embeddings.

## 10. Testing

Unit tests on synthetic data: label assignment (strictly next meeting, unlabelled tail); year rule (no training
document with label meeting on or after 1 January of its year; vintage mapping); minimum-label rule; window
splitting and pooling shape; classifier chooses C only from training documents and changing future labels
leaves past scores unchanged; signal aggregation reuses `fed_stance`; gating refuses without tag or OSF URL;
shuffled-label pipeline permutes within training sets only.

## 11. Known limitations

Document labels are noisy (most documents precede a hold; the next decision depends mostly on data, not on one
document). Pre-2000 training rows use the 1999 language model. The test window is the same 2001-2026 sample
already used for earlier models, which will be disclosed. Release times are dates only.
