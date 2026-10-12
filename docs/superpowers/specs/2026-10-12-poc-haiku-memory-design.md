# Does Claude Haiku 5.5 remember what the Fed did next? (exploratory memory exam)

Date: 2026-10-12. Status: requested by the user after the hindsight flag (RESULTS_POC.md section 6). Fixed before
any call. If Haiku passes, a Haiku-predictor backtest gets its own written rules next, and the paper reports
the method and the leakage concern; if it fails, Haiku is used only for live forecasts of meetings after its
training cutoff.

## Items

Every decision-day FOMC statement in the corpus (release within one day of a decision in `meetings.pkl`). Text
is cut at the vote paragraph (the vote lists name members and reveal the era), then masked with
`poc_haiku_leak.mask` (years, months, numbers, official names, named events). Target: the direction of the
first decision strictly after the statement (cut, hold, hike; +/-0.125 band). Five statements per request, in
random order; Haiku settings as before (no thinking, effort low).

## Exams

- **Exam 0 (descriptive): pure recall.** Only the date of each decision, no text: "what did the FOMC decide at
  its meeting on this date?" Measures whether Haiku remembers decisions at all; it does not decide anything.
- **Exam 1: next decision from the masked statement**, compared on the same statements with the leak-free
  persistence rule (next decision = the direction of this statement's own decision). One-sided exact McNemar:
  is Haiku right more often than persistence?
- **Exam 2: date dependence.** Separately, Haiku guesses the year of each masked statement. Correctly dated =
  within 1 year. Better reading should not depend on knowing the date; memory does. Date-dependence flag if
  Haiku's Exam 1 accuracy on correctly dated statements is at least 0.10 higher than on the others and a
  one-sided Fisher exact test gives p < 0.05.

## Pass rule ("no detectable memory")

Haiku passes only if Exam 1 is not significant (McNemar p >= 0.05) and Exam 2 raises no date-dependence flag.
This is conservative: Haiku can fail Exam 1 by reading better than persistence, not only by remembering, and
that is reported as such.

## Cost

About 160K input and a few thousand output tokens (around 2 US cents).
