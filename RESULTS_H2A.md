# Results: Plan 2 (H2a, neural architectures)

Secondary analyses; they cannot change the H1 verdict in RESULTS.md. Settings: PREREG_H2A.md (tag prereg-h2a).

## Ensemble metrics (mean of 8 seeds' forecasts), test windows from 2001

Rows with arch rw and linear are the Plan 1 random walk and linear S / S+N models, for context.

| task   | arch   | info   |   n |    mse |   rmse |    mae |     mbe |   accuracy |   macro_f1 |
|:-------|:-------|:-------|----:|-------:|-------:|-------:|--------:|-----------:|-----------:|
| T1     | lstm   | S      | 308 | 0.1930 | 0.4393 | 0.3119 |  0.1616 |     0.3896 |     0.3614 |
| T1     | lstm   | S+N    | 308 | 0.1760 | 0.4196 | 0.2875 |  0.1275 |     0.4026 |     0.3604 |
| T1     | gru    | S      | 308 | 0.2569 | 0.5069 | 0.3161 |  0.1910 |     0.3799 |     0.3538 |
| T1     | gru    | S+N    | 308 | 0.2001 | 0.4474 | 0.2894 |  0.1581 |     0.3506 |     0.3212 |
| T1     | tft    | S      | 308 | 0.0372 | 0.1927 | 0.1203 | -0.0255 |     0.5227 |     0.2900 |
| T1     | tft    | S+N    | 308 | 0.0383 | 0.1957 | 0.1265 | -0.0414 |     0.5260 |     0.2906 |
| T1     | hybrid | S      | 308 | 0.0414 | 0.2034 | 0.1406 |  0.0023 |     0.5292 |     0.4487 |
| T1     | hybrid | S+N    | 308 | 0.0328 | 0.1812 | 0.1251 | -0.0373 |     0.5292 |     0.3985 |
| T2     | mlstm  | S      | 212 | 0.8073 | 0.8985 | 0.6180 |  0.3643 |     0.3208 |     0.3158 |
| T2     | mlstm  | S+N    | 212 | 0.8499 | 0.9219 | 0.6562 |  0.3552 |     0.2736 |     0.2824 |
| T2     | tft    | S      | 212 | 0.0615 | 0.2480 | 0.1635 | -0.0370 |     0.5519 |     0.3042 |
| T2     | tft    | S+N    | 212 | 0.0618 | 0.2486 | 0.1731 | -0.0560 |     0.5566 |     0.3641 |
| T2     | hybrid | S      | 212 | 0.0584 | 0.2417 | 0.1611 | -0.0186 |     0.6415 |     0.5176 |
| T2     | hybrid | S+N    | 212 | 0.0507 | 0.2252 | 0.1572 | -0.0323 |     0.5991 |     0.4278 |
| T1     | rw     |        | 308 | 0.0329 | 0.1813 | 0.0917 |  0.0087 |     0.5260 |     0.2298 |
| T1     | linear | S      | 308 | 0.0323 | 0.1798 | 0.1127 | -0.0304 |     0.5584 |     0.4147 |
| T1     | linear | S+N    | 308 | 0.0277 | 0.1664 | 0.1096 | -0.0115 |     0.5390 |     0.4233 |
| T2     | rw     |        | 212 | 0.0582 | 0.2411 | 0.1256 |  0.0124 |     0.6604 |     0.2652 |
| T2     | linear | S      | 212 | 0.0421 | 0.2052 | 0.1354 | -0.0309 |     0.6792 |     0.4969 |
| T2     | linear | S+N    | 212 | 0.0397 | 0.1994 | 0.1298 | -0.0220 |     0.6981 |     0.5431 |

## Per-seed metrics, mean and standard deviation over 8 seeds

| task   | arch   | info   |   rmse mean |   rmse std |   mae mean |   mae std |   accuracy mean |   accuracy std |
|:-------|:-------|:-------|------------:|-----------:|-----------:|----------:|----------------:|---------------:|
| T1     | gru    | S      |      0.5145 |     0.0358 |     0.3200 |    0.0132 |          0.3839 |         0.0131 |
| T1     | gru    | S+N    |      0.4541 |     0.0298 |     0.2932 |    0.0151 |          0.3575 |         0.0102 |
| T1     | hybrid | S      |      0.2243 |     0.0112 |     0.1617 |    0.0112 |          0.4623 |         0.0478 |
| T1     | hybrid | S+N    |      0.1953 |     0.0129 |     0.1406 |    0.0088 |          0.4846 |         0.0231 |
| T1     | lstm   | S      |      0.4462 |     0.0116 |     0.3161 |    0.0079 |          0.3807 |         0.0199 |
| T1     | lstm   | S+N    |      0.4240 |     0.0086 |     0.2909 |    0.0060 |          0.3920 |         0.0202 |
| T1     | tft    | S      |      0.2045 |     0.0152 |     0.1361 |    0.0126 |          0.4903 |         0.0277 |
| T1     | tft    | S+N    |      0.2057 |     0.0099 |     0.1393 |    0.0094 |          0.4748 |         0.0251 |
| T2     | hybrid | S      |      0.2575 |     0.0140 |     0.1829 |    0.0087 |          0.5430 |         0.0423 |
| T2     | hybrid | S+N    |      0.2422 |     0.0124 |     0.1779 |    0.0106 |          0.5336 |         0.0522 |
| T2     | mlstm  | S      |      0.9042 |     0.0377 |     0.6193 |    0.0196 |          0.3213 |         0.0162 |
| T2     | mlstm  | S+N    |      0.9277 |     0.0283 |     0.6608 |    0.0185 |          0.2795 |         0.0128 |
| T2     | tft    | S      |      0.2630 |     0.0202 |     0.1857 |    0.0150 |          0.5153 |         0.0394 |
| T2     | tft    | S+N    |      0.2607 |     0.0135 |     0.1865 |    0.0127 |          0.5236 |         0.0493 |

## Paper replication on the paper's window (Jan 2016 - Apr 2025), T1

| model                                                              |    mse |   rmse |    mae |      mbe |
|:-------------------------------------------------------------------|-------:|-------:|-------:|---------:|
| paper LSTM (Table 2)                                               | 0.0565 | 0.2376 | 0.1495 |  -0.0006 |
| paper GRU (Table 2)                                                | 0.0732 | 0.2705 | 0.1327 |   0.0091 |
| paper LLM, lookahead (Table 3)                                     | 0.0311 | 0.1762 | 0.1483 | nan      |
| our LSTM, S (refit every 3 months; paper: monthly; real-time data) | 0.1725 | 0.4154 | 0.2925 |   0.1028 |
| our LSTM, S+N (not in paper; refit every 3 months, real-time data) | 0.1850 | 0.4301 | 0.2848 |  -0.0164 |
| our GRU, S (refit every 3 months; paper: monthly; real-time data)  | 0.4181 | 0.6466 | 0.3550 |   0.1887 |
| our GRU, S+N (not in paper; refit every 3 months, real-time data)  | 0.3172 | 0.5632 | 0.3326 |   0.0733 |

## Reading of the tests

There are 10 H2a tests (a one-sided DM test per architecture, plus a McNemar test for each T2 architecture). After Holm across all H2 p-values so far, 0 of 10 have an adjusted p below 0.05 (smallest adjusted p: 0.1177, H2a T1 hybrid DM). 2 of 10 have a raw one-sided p below 0.05: H2a T1 hybrid DM (p = 0.0084); H2a T2 hybrid DM (p = 0.0115). 1 of those 2 also pass the variance rule: H2a T1 hybrid DM (p = 0.0084). The holm_h2_so_far values can only rise when Plan 3 adds hypotheses, so they are lower bounds of the final adjusted p-values. PREREG_H2A.md defines the per-test p-value, the variance rule and the Holm correction, but no per-architecture verdict: these are counts of tests, not of models.

## Tests: S+N vs S per architecture (one-sided), Holm across all H2 so far

The variance rule: a gain counts only if mean RMSE(S) - mean RMSE(S+N) exceeds twice the larger seed standard deviation. The final Holm family is fixed after Plan 3.

| hypothesis                  |      p |   holm_h2_so_far | variance_rule   |
|:----------------------------|-------:|-----------------:|:----------------|
| H2c T1 S+N vs rw            | 0.0558 |           0.5849 |                 |
| H2c T2 S+N vs rw            | 0.0003 |           0.0044 |                 |
| H2d T1 S+M+N vs S+M         | 0.6947 |           1.0000 |                 |
| H2d T2 S+M+N vs S+M         | 0.5202 |           1.0000 |                 |
| H2d T2 McNemar S+M+N vs S+M | 0.0112 |           0.1460 |                 |
| H2a T1 lstm DM              | 0.0532 |           0.5849 | False           |
| H2a T1 gru DM               | 0.0721 |           0.6493 | False           |
| H2a T1 tft DM               | 0.8296 |           1.0000 | False           |
| H2a T1 hybrid DM            | 0.0084 |           0.1177 | True            |
| H2a T2 mlstm DM             | 0.7119 |           1.0000 | False           |
| H2a T2 mlstm McNemar        | 0.9405 |           1.0000 | False           |
| H2a T2 tft DM               | 0.5753 |           1.0000 | False           |
| H2a T2 tft McNemar          | 0.5000 |           1.0000 | False           |
| H2a T2 hybrid DM            | 0.0115 |           0.1460 | False           |
| H2a T2 hybrid McNemar       | 0.9414 |           1.0000 | False           |

## Exploratory analyses (exploratory, added after results)

Added after the results were known, following the final review. Not pre-registered; they do not change any test above.

### Hybrid against its own linear anchor and against the random walk (exploratory, added after results)

One-sided DM p-values on the seed-mean ensembles; the anchor is the Plan 1 forecast results/preds_<task>_<S|S+N>.csv. A small p means the anchor (or the random walk) is better.

| task   | info   |   hybrid rmse |   own anchor rmse |   rw rmse |   p anchor beats hybrid |   p rw beats hybrid |
|:-------|:-------|--------------:|------------------:|----------:|------------------------:|--------------------:|
| T1     | S      |         0.203 |             0.18  |     0.181 |                1.2e-08  |              0.0343 |
| T1     | S+N    |         0.181 |             0.166 |     0.181 |                3.99e-05 |              0.504  |
| T2     | S      |         0.242 |             0.205 |     0.241 |                9.11e-06 |              0.484  |
| T2     | S+N    |         0.225 |             0.199 |     0.241 |                5.48e-07 |              0.91   |

### MSE decomposition of the hybrid S vs S+N gap (exploratory, added after results)

| task   |   hybrid mse gap |   linear mse gap (Plan 1) |   remainder (TFT corrections) |
|:-------|-----------------:|--------------------------:|------------------------------:|
| T1     |          0.00854 |                   0.00465 |                       0.00390 |
| T2     |          0.00769 |                   0.00235 |                       0.00534 |

### Best epoch kept by the TFT early stopping (exploratory, added after results)

Epoch 0 means the untrained weights were kept. Every refit of every seed listed in the seeds column.

| task   | spec       |   seeds |   refits |   share best epoch 0 |   median best epoch |
|:-------|:-----------|--------:|---------:|---------------------:|--------------------:|
| T1     | hybrid-S   |       8 |      208 |                0.423 |               1.000 |
| T1     | hybrid-S+N |       8 |      208 |                0.452 |               1.000 |
| T1     | tft-S      |       8 |      208 |                0.207 |               6.500 |
| T1     | tft-S+N    |       8 |      208 |                0.245 |               6.000 |
| T2     | hybrid-S   |       8 |      216 |                0.315 |               3.000 |
| T2     | hybrid-S+N |       8 |      216 |                0.329 |               3.500 |
| T2     | tft-S      |       8 |      216 |                0.222 |               8.000 |
| T2     | tft-S+N    |       8 |      216 |                0.218 |               8.000 |

### GPU re-run of the TFT and hybrid specs (exploratory, added after results)

The same code and seeds re-run once more, to bound GPU floating-point nondeterminism. The committed forecasts in results/h2a/ are unchanged.

| task   | spec       |   committed_rmse |   rerun_rmse |   max_abs_forecast_diff |   committed_dm_p |   rerun_dm_p |   committed_variance_rule |   rerun_variance_rule |
|:-------|:-----------|-----------------:|-------------:|------------------------:|-----------------:|-------------:|--------------------------:|----------------------:|
| T1     | tft-S      |           0.1927 |       0.1927 |                  0.0000 |         nan      |     nan      |                  nan      |              nan      |
| T1     | hybrid-S   |           0.2034 |       0.2034 |                  0.0000 |         nan      |     nan      |                  nan      |              nan      |
| T1     | tft-S+N    |           0.1957 |       0.1957 |                  0.0000 |         nan      |     nan      |                  nan      |              nan      |
| T1     | hybrid-S+N |           0.1812 |       0.1812 |                  0.0000 |           0.0084 |       0.0084 |                    1.0000 |                1.0000 |
| T2     | tft-S      |           0.2480 |       0.2480 |                  0.0000 |         nan      |     nan      |                  nan      |              nan      |
| T2     | hybrid-S   |           0.2417 |       0.2417 |                  0.0000 |         nan      |     nan      |                  nan      |              nan      |
| T2     | tft-S+N    |           0.2486 |       0.2486 |                  0.0000 |         nan      |     nan      |                  nan      |              nan      |
| T2     | hybrid-S+N |           0.2252 |       0.2252 |                  0.0000 |           0.0115 |       0.0115 |                    0.0000 |                0.0000 |

## Deviations

### Plan 2 (H2a) deviations

- Plan 2: T1/gru-S/seed0 took more than 3,600 s (the timing run was stopped 3,720 s after the process started, so the seed-pass exceeded 3,600 s); T1/lstm-S/seed0 took 3,039 s. Per PREREG_H2A.md all paper LSTM/GRU T1 specs run with refit every 3 months. The refit-every-month pilot forecasts were deleted unused. LSTM pilot 2026-10-09 17:44:49-18:35:34 (3,038.6 s seed-pass); GRU pilot started 18:35:46 and was stopped by a watchdog at 19:37:46 (3,720 s after start; 3,600 s plus 120 s for imports/data loading, which the runner's timer excludes). GRU at refit every 3 months later took 1,222-1,280 s per seed-pass (gru-S, the piloted spec: 1,262-1,280 s), about 3,790-3,840 s for gru-S if scaled to monthly refits.
- Plan 2: after tag `prereg-h2a`, `neural_walk.run_neural_spec` gained a keyword `anchor_every_row` (default True, the pre-registered behaviour, unchanged). Only the descriptive frozen-model analysis passes False, so a frozen hybrid's linear anchor is fitted once, like every other frozen model.
- Plan 2: run order changed to TFT and hybrid specs first, then the rest; each spec's forecasts depend only on its own settings and seed, so the order changes no result, up to GPU floating-point nondeterminism (see RESULTS_H2A.md exploratory section).
- Plan 2: run command: `PYTHONPATH=src python -m fed_news.run_neural --task all --paper-refit-every 3 --device cuda` (TFT/hybrid specs were run first with `--only`).
- Plan 2: after the results were known (final-review fixes, exploratory): `neural.fit_early_stopping` records the kept epoch as `best_epoch_` (training, return value and numbers unchanged); `run_neural --paper-refit-every` now defaults to 3; `evaluate_h2a.py` and `frozen.py` report text was extended and results/h2a/ensemble_metrics.csv gained six context rows (random walk and Plan 1 linear); `diag_tft.py` is new. No forecast file was changed or regenerated. Exploratory additions are labelled "exploratory, added after results" in RESULTS_H2A.md.

## Provenance

- prereg-h2a tag commit: e931dfc3d5142e506330a35d7c4f005d4f7593af
- HEAD: d76b80d
- `git diff --stat prereg-h2a -- src`:

```
src/fed_news/diag_tft.py     | 128 +++++++++++++++++++++++
 src/fed_news/evaluate_h2a.py | 241 +++++++++++++++++++++++++++++++++++++++++++
 src/fed_news/frozen.py       | 167 ++++++++++++++++++++++++++++++
 src/fed_news/neural.py       |   6 +-
 src/fed_news/neural_walk.py  |  19 ++--
 src/fed_news/run_neural.py   |   3 +-
 6 files changed, 555 insertions(+), 9 deletions(-)
```

evaluate_h2a.py was committed (0554690) before the first forecast file was written. Later edits add report sections only; the pre-registered test code (pair_tests, variance_rule, Holm) is unchanged.
