# Pre-registration: Study 2 (H6, regional Fed speeches)

Date: 2026-10-10. Code commit: e8467d4.
Binding: docs/superpowers/specs/2026-10-10-study2-regional-speeches-design.md; code src/fed_news/{fraser,
regional_corpus,stance2,study2,evaluate_study2}.py at this commit, with the unchanged Plan 5 event code
(events.py, event_study.py at tag prereg-text). No regional speech text had been scored and no new event-day
yield change had been computed before this tag; only FRASER item metadata (dates, titles) and BIS speech
counts had been examined.

## H6
On the event set expanded with regional Federal Reserve Bank presidents' speeches (FRASER titles of 35
presidents plus BIS speeches, tenure years and de-duplication as in the spec), with documents re-scored by
per-year ChronoBERT classifiers trained on all eligible labels, the stance surprise predicts the same-day
2-year Treasury yield change out of sample: one-sided Clark-West p < 0.05 (Plan 5 walk-forward OLS, events
keyed by outcome trading day, 30-day document-mean baseline, test outcome days 2001-01-01 to 2026-09-30).
Void if the same test on the previous trading day's change gives p < 0.05.

## Secondary (not decisive)
Regional speeches only; Plan 5 documents with the new classifier; full-sample OLS with Newey-West errors;
monthly and per-meeting forecasting (Plan 1 primary model, S + rebuilt N against S).
