# Proof of concept: can Claude Haiku 5.5 label Fed sentences like the human annotators? (exploratory)

Date: 2026-10-12. Status: chosen by the user; this file fixes the details before any API call.

## Question

The sentence scorers reach about 0.59 three-class accuracy on later years with 1,722 human labels. More labels
could come from an LLM, but a student model trained on LLM labels can hardly beat the LLM's own agreement with
the human labels. This pilot measures that agreement first.

## Pilot

- 300 human-labelled sentences (TDW/WCB) dated before 2012, sampled with seed 0. Text only: no date, speaker,
  document type or source is sent.
- Model `claude-haiku-5-5`, thinking disabled, effort low, 25 sentences per request, fixed zero-shot
  instructions (definitions of hawkish, dovish and neutral; no dataset sentences as examples). Answers are JSON
  `{"1": "H", ...}`; a malformed answer is retried once, then counted as unlabelled (and as wrong).
- Pass: three-class agreement with the human labels at least 0.70 (clearly above the scorers' 0.59). Also
  reported: recall per class, confusion matrix, hawkish-vs-dovish sign agreement, and the tokens used.

If it passes, the next step is a written plan: label about 20,000 unlabelled pre-2012 sentences, fine-tune
ChronoBERT on human plus LLM labels, test once on the 2012+ human labels (bar: at least 0.642), and check for
hindsight contamination (LLM labels agreeing with the next decision more than human labels do). If it fails,
LLM labelling is dropped.

## Limits

Haiku 5.5 was trained on text written long after these sentences and may label with hindsight; the pilot
measures agreement with humans, not the absence of hindsight. 300 sentences give a standard error of about 0.026.

## Compute

About 17,000 input and 2,000 output tokens (well under one US cent at $0.10 / $0.50 per million). API key read
from `ANTHROPIC_API_KEY` or `<data>/secrets/anthropic_api_key.txt` (never committed).
