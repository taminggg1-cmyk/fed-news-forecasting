### Deviations from PREREG.md

- None in code: `git diff --stat prereg -- src` is empty; the H1 run used the tagged code.

### Errata

- PREREG.md line 11 states the median held-out stance accuracy as 0.661; the correct value from results/stance_accuracy.csv is 0.655 (mean 0.644). This does not affect H1.

### Limitations (in addition to PREREG.md "Known limitations")

- T1 timing: s_rate_lag = FEDFUNDS(t-1) with the cutoff on the 1st of month t; the monthly average needs the last day's effective rate, published the next business day (up to about 1-3 days of lookahead). It enters S, S+N and the random walk identically and follows the paper's protocol, so it cannot favour S+N.
- ALFRED fallback for earn/u6 before 2011-03-04 gates periods by a 9-day release lag; the employment report is sometimes released on the 10th, so in fallback windows a release-day value could be visible. Identical for S and S+N.
- The McNemar leg of the shuffled-label control has no power by construction (the shuffled model predicts hold at all 212 meetings, 0 discordant pairs, p = 1.0); only the two Diebold-Mariano legs test for leakage.
- Secondary results: H2c T2 (S+N vs random walk, p = 0.0003) mostly reflects S beating the random walk (RMSE 0.2052 vs 0.2411), not the news block. H2d McNemar (S+M+N vs S+M, p = 0.0112) is Holm-adjusted only within Plan 1 (0.045); the final H2 family (Plans 2-3) is larger and the spec expected H2d to be null. Neither is evidence that news beats the random walk or the market.
- Environment: the venv uses system site-packages (include-system-site-packages = true); versions matched results/pip_freeze.txt at run time.
- Reproduction: an independent re-run of the full ladder from the hashed data and tagged code reproduced all 16 prediction files (max |difference| 8.9e-16) and the H1 p-values (max difference 8.3e-17).
