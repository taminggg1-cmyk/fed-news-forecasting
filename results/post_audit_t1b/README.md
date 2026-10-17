# Plan 1 monthly ladder on the corrected cutoff (post-audit, exploratory)

Made by `audit/2026-10-13/rerun_plan1_t1b.py` on 2026-10-13 (committed 0f00878) from `D:/fed_news/t1b_full.pkl`.
Same specs, grids, t-2 rule, seed and 308 test months as Plan 1; only the cutoff differs (start of the second US
business day instead of the first calendar day). These rows supersede `results/preds_T1_*.csv` for the monthly
task. Mixed vintage. `tests_T1_t1b.json`: DM-HLN (the registered test) plus the audit's Newey-West p and
block-bootstrap interval. Re-check without writing anything: `python audit/verify_saved_outputs.py --quick`.
