# Study reports

| Report | Study | Made by |
| --- | --- | --- |
| `RESULTS.md` | Plan 1 linear models (H1, H2c, H2d) | `src/fed_news/evaluation/evaluate.py` |
| `RESULTS_H2A.md` | Plan 2 original networks (H2a) | `src/fed_news/studies/plan2_neural_original/evaluate_h2a.py` |
| `RESULTS_FROZEN.md` | Plan 2 frozen-model analysis | `src/fed_news/studies/plan2_neural_original/frozen.py` |
| `RESULTS_TEXT.md` | Plan 5 event study (H5) and look-ahead comparison | `src/fed_news/studies/plan5_event_study/evaluate_text.py` |
| `RESULTS_STUDY2.md`, `RESULTS_STUDY2_FROZEN.md` | Study 2 (H6) | `src/fed_news/studies/study2_regional/evaluate_study2.py`, `study2_frozen.py` |
| `RESULTS_POC.md` | Proofs of concept of 2026-10-12 | written by hand (by the AI agent) from `results/poc_*` |
| `RESULTS_STUDY4.md` | Study 4: Haiku index, sensitivity checks, version 2 networks | written by hand from `results/study4/` |
| `RESULTS_STUDY5.md` | Study 5 | written by hand from `results/study5/` |

These reports were written when each study finished. Corrections found later (the 2026-10-13 audit, the corrected
monthly cutoff, the AMRO paper versions) are in [`../../audit/2026-10-13/AUDIT.md`](../../audit/2026-10-13/AUDIT.md),
[`../../DEVIATIONS.md`](../../DEVIATIONS.md) and [`../RESULTS_PROVENANCE.md`](../RESULTS_PROVENANCE.md); the reports
themselves are left as written. Until 2026-10-18 they were at the repository root, and the generators now write
here (`config.report_path`).
