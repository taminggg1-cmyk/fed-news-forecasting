# Haiku as a direct sentence scorer: leak mitigation and leak tests, step 1 (exploratory)

Date: 2026-10-12. Status: approved by the user, who accepts that remaining look-ahead risk is reported as a
limitation. This file fixes step 1 before any API call. Step 2 (scoring all 343,742 sentences, about 20M tokens)
and step 3 (forecasts) get their own written rules later and run only if step 1 allows it.

## Why

Claude Haiku 5.5 was trained on text written long after most Fed documents, so scores it gives to old sentences
can carry knowledge of what the Fed did next. That cannot be removed, only reduced and measured.

## Mitigation (applied to every sentence sent)

`mask()` replaces four-digit years, month names, all numbers (levels, fractions, percentages), the surnames of
Fed officials in the corpus speaker field plus Greenspan, Bernanke, Yellen and Powell, and named events
(Lehman, Bear Stearns, AIG, Fannie Mae, Freddie Mac, Katrina, Rita, September 11, Enron, WorldCom, Y2K, COVID,
coronavirus, pandemic, Omicron, Brexit, Ukraine, Silicon Valley Bank, SVB) with placeholders. Sentences are sent
with no date, speaker or source, 25 per request, in random order so one request mixes years. Haiku settings as
the labelling pilot attempt 1 (no thinking, effort low).

## Test A: can Haiku still date masked sentences? (decides step 2)

300 human-labelled sentences from all years (seed 0), masked; Haiku is asked for its best guess of the year
each was written. **Masking fails to hide the era if more than 50% of guesses are within 2 years of the true
year.** Reported next to a constant-guess baseline (always the median year). If masking fails, step 2 does
not run.

## Test B: hindsight in labels (reported as a flag)

All 3,074 human-labelled sentences, masked, labelled by Haiku (H = +1, D = -1, N = 0). For each sentence, the
next FOMC decision after its release date gives `next` (cut -1, hold 0, hike +1). The leak-free comparison is
the saved ChronoBERT sentence score (P(hawkish) - P(dovish), from the classifier of the sentence's year, trained
only on earlier labels). OLS of `next` on the human label, Haiku's label and ChronoBERT's score (the last two
standardised), standard errors clustered by next meeting. **Hindsight flag if Haiku's coefficient is positive
and exceeds ChronoBERT's with p < 0.05 (Wald test of the difference).** A flag means Haiku's labels track what
the Fed did next beyond what the human reading and a leak-free model of the same text show; it cannot separate
hindsight from better reading, so a flag is reported prominently with any later result.

## Cost

Test A: about 25K tokens. Test B: about 220K tokens. Under 5 US cents in total.
