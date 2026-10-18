# Fed News Forecasting

**Does what the Federal Reserve says help forecast what it does?** This project tests whether the hawkish or
dovish tone of Fed communication improves out-of-sample forecasts of the US federal funds rate, beyond the
economic statistics and market interest rates that were available at the time.

Every forecast uses only information that existed on the day it was made: data as first published (ALFRED
vintages), documents dated by their release, and a language model (ChronoBERT) retrained each year on earlier text
only. Hypotheses were registered with git tags before the main runs.

*Draft research, October 2026. The code, analyses and documents were written with an AI coding agent
(see [How this was made](#how-this-was-made)).*

## Main result: the final study

The main result is the **final study**: Study 4 (simple model), Study 4 version 2 (tuned LSTM, GRU and TFT) and
Study 5 (three meetings ahead). The author's briefing calls these Tests 4–6. Everything before them was an
earlier design (see [Earlier designs](#earlier-designs-trial-and-error)).

**Short answer.** News improves most models: Fed tone lowers the forecast error in 12 of 14 main comparisons (10
of 12 for the networks, and both versions of the simple model). Only one gain is statistically significant, LSTM +
Claude Haiku on meetings, and it is not confirmed on strict real-time data. Not significant does not mean news
fails: with about 200 meetings, gains this small cannot yet be told apart from chance.

**Design.** FOMC meetings and months, 2001–2026. Every forecast uses only information available before it: 16
economic statistics as first published (ALFRED), plus 8 market signals for meetings. Fed tone is read by ChronoBERT
(retrained each year on earlier text only) or by Claude Haiku 5.5 (training data to June 2026). Models: a simple
model (penalised ordered logit for meetings), LSTM, GRU and TFT. The networks are tuned with Optuna inside each
training window and averaged over 8 seeds.

**Status.** None of the final study is a pre-registration. Study 4 and Study 5 are exploratory studies whose design
was committed before the first run. Study 4 version 2 is a dated amendment (6048e0d), written after the version 1
network results had been seen.

**Per meeting (206 scheduled meetings, 2001–2026): decisions called correctly, with p for the news versions**

| Model | No news | + ChronoBERT | + Haiku |
| --- | --- | --- | --- |
| Simple model (ordered logit) | 86.4% | 86.9%, p 0.21 | 88.8%, p 0.14 |
| LSTM, normal data | 78.6% | 83.5%, p 0.37 | **84.5%, p 0.003 (Holm p 0.020)** |
| GRU, normal data | 80.6% | 80.1%, p 0.99 | 82.0%, p 0.20 |
| TFT, normal data | 86.4% | 85.9%, p 0.21 | 86.9%, p 0.17 |
| LSTM, strict data | 81.6% | 84.5%, p 0.47 | 85.4%, p 0.46 |
| GRU, strict data | 83.0% | 85.9%, p 0.85 | 86.4%, p 0.93 |
| TFT, strict data | 87.4% | 86.9%, p 0.94 | 87.9%, p 0.69 |

Only one gain is significant: LSTM + Haiku on normal data (RMSE 0.1420 → 0.1333 percentage points). The strict-data
accuracy gains are within chance: a 3-point gain is about 6 meetings, and the LSTM + Haiku RMSE barely moved
(0.1251 → 0.1246). Three meetings ahead (Study 5, simple model, strict real-time inputs, 204 meetings), ChronoBERT
makes log loss 7.7% worse (p 0.898) and Haiku lowers it 1.2% (p 0.300).

**Monthly (308 months, 2001–2026): RMSE in percentage points, with p for the news versions**

| Model | No news | + ChronoBERT | + Haiku |
| --- | --- | --- | --- |
| "Rate stays the same" | 0.181 | | |
| LSTM, normal data | 0.188 | 0.198, p 0.88 | 0.172, p 0.026 (Holm p 0.156) |
| GRU, normal data | 0.180 | 0.178, p 0.36 | 0.178, p 0.45 |
| TFT, normal data | 0.191 | 0.177, p 0.042 (Holm p 0.169) | 0.173, p 0.034 (Holm p 0.168) |
| LSTM, strict data | 0.197 | 0.185, p 0.13 | 0.174, p 0.12 |
| GRU, strict data | 0.184 | 0.189, p 0.67 | 0.165, p 0.014 (secondary check) |
| TFT, strict data | 0.186 | 0.181, p 0.31 | 0.173, p 0.084 |

How to read the p-values: each is one-sided (does news lower the error?). Network p-values are Newey–West tests
on squared error. Simple-model p-values are HLN Diebold–Mariano tests on log loss over all 212 meetings. Holm p
corrects over the 6 network comparisons per task (LSTM, GRU and TFT × ChronoBERT and Haiku) on normal data. Strict
data = only the fully vintaged statistics; it is a robustness check outside that family. Sources:
`results/study4/neural/metrics_neural_T2.csv`, `comparison_neural_T2.csv`,
`results/study4/neural_t1/metrics_neural_T1.csv`, `comparison_neural_T1.csv`, `results/study4/verdict_T2.json`.

**Checks on the news signal.** A placebo (tone shifted by half the sample, about 16 years) and shuffled training
labels never beat the base model significantly. LSTM + Haiku beats its own placebo (p 0.0008). Fake tone did lower
meeting RMSE for the GRU (0.1445 → 0.1345) and the TFT (0.1415 → 0.1355), so network gains of this size can come
from noise. Haiku recalls 87.8% of decisions from the date alone, and its labels track the next decision more than
ChronoBERT's (p 0.00005), so a Haiku gain may be memory. Only 2 test meetings fall after Haiku's training cutoff.

**Why the simple model beats the bigger networks.** On normal data the ordered logit (RMSE 0.127 percentage points)
has lower RMSE than every network (0.133–0.161), and calls at least as many decisions right as the best network
in each column. Descriptive diagnostics
([`audit/2026-10-18/why_simple_model_wins.txt`](audit/2026-10-18/why_simple_model_wins.txt)):

- Each refit trains on 48 to 246 meetings. The tuned LSTM and GRU have a median of about 211,000 and 217,000
  parameters; the logit has 26.
- The networks call moves that never happen. On the 140 holds, the logit called a move 12 times, against 28 (LSTM),
  25 (GRU) and 18 (TFT). Holds give 95% of the LSTM's extra squared error.
- A few unusual meetings decide it. In early 2009, with the rate at 0.125%, the LSTM and GRU forecast −0.39% and
  −0.48%. In March 2023 they forecast cuts while the Fed hiked.
- Tuned settings jump between refits (LSTM hidden size 256 → 16 → 128), and single-seed RMSE ranges 0.140–0.176
  across the networks.
- Market yields already price the expected move, and the ordered logit turns them straight into cut / hold / hike
  probabilities.
- With strict data (16 inputs instead of 24), the GRU and LSTM reach lower RMSE (0.122, 0.125) but call fewer
  decisions right (83.0%, 81.6%). No network beats the simple model on both measures.

## Earlier designs (trial and error)

These runs came before the final study. They used a different news input (a six-column news block), untuned
paper-style networks, or a different question, so they are not the main result. The registered results stand as
reported.

| Briefing name | Repository name | Status | Result |
| --- | --- | --- | --- |
| Test 1, simple models | Plan 1, H1 | Registered (tag `prereg`) | News lowered monthly RMSE 0.1919 → 0.1675 (−13%, corrected cutoff), p 0.103; 78% of the gain is June 2020. Meetings p 0.061, decisions p 0.227 |
| Test 1, networks | Plan 2, H2a | Registered (`prereg-h2a`) | News lowered error in 6 of 9 models; 0 of 10 tests after Holm |
| Test 2, bond yields | Plan 5, H5; Study 2, H6 | Registered (`prereg-text`, `prereg-study2`) | No effect: Clark–West p 0.129 (1,778 releases), 0.369 (3,158) |
| Test 3, tone readers | Proofs of concept (`results/poc_*`), Study 3 | Exploratory | No reader reached 70% agreement with human labels (52–68%) |

**Why the early LSTM and GRU did worse.** Plan 2 copied the paper's hyperparameters (preliminary SSRN version: 240
units, 1 layer, 300 epochs) without tuning. Tuning them with Optuna in Study 4 version 2 lowered monthly squared
error by 0.158 squared percentage points for the LSTM (Newey–West p 0.000007) and 0.224 for the GRU (p 0.012); the
TFT gain was 0.0006 (p 0.43). The replication of Chan-Lau, Quach and Shi (AMRO WP/25-10) was not achieved: our
untuned LSTM and GRU reach RMSE 0.415 and 0.647 on their window, against 0.1868 and 0.2672 published.

Numbers in the paper draft are traced to their files in [`docs/RESULTS_PROVENANCE.md`](docs/RESULTS_PROVENANCE.md).

## Where to look

| To see | Open |
| --- | --- |
| The paper draft | [`paper/main.pdf`](paper/main.pdf) |
| How the code fits together, with one forecast traced from raw data to p-value | [`docs/REVIEW_MAP.md`](docs/REVIEW_MAP.md) |
| Where every number in the paper comes from | [`docs/RESULTS_PROVENANCE.md`](docs/RESULTS_PROVENANCE.md) |
| The audit of every text result (2026-10-13) | [`audit/2026-10-13/AUDIT.md`](audit/2026-10-13/AUDIT.md) |
| Why the simple model beats the networks (2026-10-18 diagnostics) | [`audit/2026-10-18/`](audit/2026-10-18/) |
| Every deviation from the plans, with dates | [`DEVIATIONS.md`](DEVIATIONS.md) |
| The pre-registrations | [`docs/prereg/`](docs/prereg/) |
| The report of each study | [`docs/reports/`](docs/reports/) |
| Full technical description: data, stance classifier, models, tests | [`docs/README_DETAILED.md`](docs/README_DETAILED.md) |
| Saved forecasts and test tables | [`results/`](results/) (each folder states its study and status) |

## Repository layout

```
src/fed_news/
  collect/      raw sources -> dated tables (ALFRED, markets, FOMC calendar, Fed documents, labels)
  features/     stance scores (ChronoBERT, Haiku) and the monthly / per-meeting feature panels
  models/       ridge, ordered logit, Taylor rule, walk-forward engine, LSTM / GRU / TFT
  evaluation/   metrics, registered tests (DM-HLN, McNemar, Holm), figures
  studies/      one folder per study (plan1_linear, plan2_neural_original, ..., study4_networks_v2, study5_horizon)
tests/          369 offline unit tests
results/        all saved forecasts, metrics and test tables
audit/          the 2026-10-13 audit, a read-only reproduction check, 2026-10-18 diagnostics
paper/          draft paper (LaTeX, PDF, figures)
docs/           review map, provenance, pre-registrations, study reports, design records
DEVIATIONS.md   every deviation, erratum and post-results note
```

## Reproduce

- Python 3.13 environment; exact versions in [`results/pip_freeze.txt`](results/pip_freeze.txt). Raw data and
  caches live outside the repository (`D:/fed_news`, or set `FED_NEWS_DATA`); API keys go in `<data root>/secrets/`.
- The data folder (everything except API keys, 0.93 GB) is attached to the release
  [`data-2026-10-17`](https://github.com/taminggg1-cmyk/fed-news-forecasting/releases/tag/data-2026-10-17)
  as `fed_news_data.zip`. Unzip it and point `FED_NEWS_DATA` at the folder.
- From the repository root, with `PYTHONPATH=src`:
  - `python -m pytest -q` runs the offline tests (about 2 minutes).
  - `python audit/verify_saved_outputs.py` re-runs the linear models from the cached panels and checks that every
    saved forecast is reproduced byte for byte (CPU, about 8 minutes; writes nothing).
- The full run order, including the GPU and paid-API steps, is in
  [`docs/README_DETAILED.md`, section 13](docs/README_DETAILED.md#13-reproducing-the-study).
- Module names used before the 2026-10-17 reorganisation (`python -m fed_news.run_models`,
  `python -m fed_news.tune forecast`, ...) still work.

## How this was made

All code, analyses, commits and documents in this repository were written by an AI coding agent (Claude, by
Anthropic) working with the project's author. The author's decisions appear where commit messages and design
documents record them. Every commit carries a `Co-Authored-By: Claude` line. Results that were added after
earlier results had been seen are labelled exploratory, and the corrections found in the audit are listed in
[`DEVIATIONS.md`](DEVIATIONS.md) and [`audit/2026-10-13/AUDIT.md`](audit/2026-10-13/AUDIT.md).

## License

MIT, see [`LICENSE`](LICENSE).
