# Fed News Forecasting

**Does what the Federal Reserve says help forecast what it does?** This project tests whether the hawkish or
dovish tone of Fed communication improves out-of-sample forecasts of the US federal funds rate, beyond the
economic statistics and market interest rates that were available at the time.

Every forecast uses only information that existed on the day it was made: data as first published (ALFRED
vintages), documents dated by their release, and a language model (ChronoBERT) retrained each year on earlier text
only. Hypotheses were registered with git tags before the main runs.

*Draft research, October 2026. The code, analyses and documents were written with an AI coding agent
(see [How this was made](#how-this-was-made)).*

## Short answer

Fed text looks like a **promising but not yet established** complement to market prices.

- **Monthly forecasts.** Adding Fed text to a statistics-only ridge model lowers the RMSE from 0.1919 to 0.1675
  percentage points (−13%, 308 months, corrected cutoff). The registered test gives p = 0.103, and 78% of the
  gain comes from one month, June 2020.
- **Neural networks.** Text lowers squared error in 10 of 12 comparisons. An LSTM reading Claude Haiku's stance
  scores improves meeting forecasts: RMSE 0.1420 → 0.1333, decisions correct 78.6% → 84.5%, Holm-adjusted
  p = 0.020. Limits: this comparison comes from a protocol written after the first network results; it is not
  established with strictly real-time inputs (p = 0.461); Haiku's training data cover 204 of the 206 test meetings.
- **Markets dominate.** Ridge with market rates has monthly RMSE 0.1333; adding text moves it by 0.0017.
- **No pre-registered text hypothesis passed** (details below).
- **Replication of Chan-Lau, Quach and Shi (AMRO WP/25-10) not achieved:** our LSTM and GRU reach RMSE 0.415 and
  0.647 on their window, against 0.1868 and 0.2672 published.

RMSE is in percentage points of the policy rate. Every number on this page is traced to its file in
[`docs/RESULTS_PROVENANCE.md`](docs/RESULTS_PROVENANCE.md).

## Results at a glance

| Study | Question | Status | Result |
| --- | --- | --- | --- |
| Plan 1, H1 | Does text help the linear models? | Registered (tag `prereg`) | Not supported: monthly p 0.103 (0.077 on the original cutoff), meetings p 0.061, decisions p 0.227 |
| Plan 2, H2a | Does text help paper-style LSTM, GRU, TFT, hybrid? | Registered (`prereg-h2a`) | Not supported: 0 of 10 tests after Holm |
| Plan 5, H5 | Does the stance surprise move the 2-year yield that day? | Registered (`prereg-text`) | Not supported: Clark–West p 0.129, 1,778 releases |
| Study 2, H6 | Same, adding regional Fed presidents' speeches | Registered (`prereg-study2`) | Not supported: p 0.369, 3,158 releases |
| Study 4 | Haiku stance index in the meeting model | Exploratory study | Not supported: log loss 0.294 → 0.272, p 0.144 |
| Study 4, version 2 | Tuned LSTM / GRU / TFT with Haiku or ChronoBERT stance | Amended protocol | 1 of 12 survives Holm: LSTM + Haiku, meetings, p 0.020 |
| Study 5 | Policy path three meetings ahead, strictly real-time | Exploratory study | Not supported: log loss 0.779 → 0.839, p 0.898 |

## Where to look

| To see | Open |
| --- | --- |
| The paper draft | [`paper/main.pdf`](paper/main.pdf) |
| How the code fits together, with one forecast traced from raw data to p-value | [`docs/REVIEW_MAP.md`](docs/REVIEW_MAP.md) |
| Where every number in the paper comes from | [`docs/RESULTS_PROVENANCE.md`](docs/RESULTS_PROVENANCE.md) |
| The audit of every text result (2026-10-13) | [`audit/2026-10-13/AUDIT.md`](audit/2026-10-13/AUDIT.md) |
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
audit/          the 2026-10-13 audit and a read-only reproduction check
paper/          draft paper (LaTeX, PDF, figures)
docs/           review map, provenance, pre-registrations, study reports, design records
DEVIATIONS.md   every deviation, erratum and post-results note
```

## Reproduce

- Python 3.13 environment; exact versions in [`results/pip_freeze.txt`](results/pip_freeze.txt). Raw data and
  caches live outside the repository (`D:/fed_news`, or set `FED_NEWS_DATA`); API keys go in `<data root>/secrets/`.
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
