# Pre-registration addendum: Plan 5 event study (H5)

Date: 2026-10-10. Code commit: fbebb73.
Binding: docs/superpowers/specs/2026-10-10-text-methodology-design.md, section 2; code src/fed_news/events.py
and src/fed_news/event_study.py at this commit. No event-day yield change had been computed against stance
before this tag.

## H5
The Fed stance surprise predicts the same-day change in the 2-year Treasury yield out of sample.

- Events: each statement, minutes, press-conference transcript or speech (document stance from
  D:/fed_news/doc_stance.pkl, the Plan 1 time-consistent classifiers) is keyed to its outcome trading day,
  the first date with a 2-year yield on or after its release day; documents with the same key form one
  event (weekend and holiday releases join the next trading day). Test events: outcome days 2001-01-01 to
  2026-09-30. Event stance = mean document stance.
- Surprise: event stance minus the document-mean stance of earlier events' documents released within the
  30 calendar days before the outcome day; if none, minus the mean stance of the latest earlier event's
  documents.
- Outcome: 2-year constant-maturity yield (FRED DGS2, market_daily.pkl y2), basis points, outcome day minus
  previous trading day. Void-rule outcome: the change on the previous trading day.
- Test: walk-forward OLS on all strictly earlier event days (from 1994); small model dy2 = a; large model
  dy2 = a + b * surprise; one-sided Clark-West; supported if p < 0.05.
- Void: the same test with the day-before change as outcome; if p < 0.05, H5 is void.
- Secondary (not decisive): full-sample OLS with Newey-West (5 lags), Clark-West by day category
  (statement > minutes > speech), sign accuracy.

## Exploratory analyses (not pre-registered tests)
Look-ahead comparison (answerdotai/ModernBERT-base on all labels), T-bill market benchmark, feature ablation,
as in the spec sections 3-5. The market benchmark is descriptive: both bills are shown with no selection,
the 3-month bill (full sample) is the headline row, and its DM p-values are not tests.
