# Proof of concept: fine-tuned ChronoGPT vs fine-tuned ChronoBERT as hawk/dove scorers (exploratory)

Date: 2026-10-12. Status: chosen by the user after the frozen-scorer proof of concept (RESULTS_POC.md section 2);
this file fixes the details before anything runs. Exploratory; no forecast is made.

## 1. Question

Frozen 2011-vintage features gave 0.520 (ChronoBERT) and 0.561 (ChronoGPT) three-class accuracy on labels from
2012 on. Does fine-tuning, the way the pipeline already fine-tunes ChronoBERT, lift ChronoGPT clearly above
fine-tuned ChronoBERT?

## 2. Data

Same split as the frozen test: TDW/WCB labels dated before 2012 for training (1,722), from 2012 on for testing
(1,352). 2011 vintages of both models. Nothing is tuned on the test labels.

## 3. Methods

- **ChronoBERT (baseline):** `stance.train_and_score`, unchanged: full fine-tuning, AdamW lr 2e-5, 3 epochs,
  batch 16, max 256 tokens, gradient clipping 1.0. Seeds 0, 1, 2; probabilities averaged over seeds.
- **ChronoGPT:** the 26 encoder blocks of the 52-block model (layer 26 was the best layer in the frozen test,
  chosen on training labels only); decoder blocks and the language-model head are not used. LoRA adapters
  (rank 16, alpha 32, dropout 0.05) on every linear map of those blocks (query, key, value, attention output,
  MLP in and out); base weights frozen in bfloat16; a linear head on the mean-pooled normalised layer-26 states.
  AdamW lr 2e-4, weight decay 0.01, 3 epochs, batch 16, max 256 tokens, gradient clipping 1.0, gradient
  checkpointing per block. Seeds 0, 1, 2; probabilities averaged over seeds.
- To train, the `@torch.inference_mode()` decorators are removed from the vendored model code; the forward
  arithmetic is unchanged (unit test against the decorated version's outputs).

## 4. Pass rule (fixed before running; the same bar as the frozen test)

Fine-tuned ChronoGPT passes if its three-class test accuracy is at least 0.75 and at least 0.05 above
fine-tuned ChronoBERT's. Also reported: macro-F1, recall per class, log loss, hawkish-vs-dovish sign agreement,
exact McNemar (descriptive), and each seed's accuracy.

If it passes, the next step is a written plan for yearly ChronoGPT vintages and a fresh statement-change stage 1
gate. If it fails, the text route is closed for this project's budget.

## 5. Compute

RTX 2060 (6 GB). ChronoBERT: a few minutes per seed. ChronoGPT: roughly 20-30 minutes per seed. About 1.5-2
hours in total.
