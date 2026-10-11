# Study 2 frozen analysis (exploratory, added before any Study 2 result)

Models fitted once before the cutoff and never updated; descriptive, not part of H6.

## Event model

| cutoff     |   n_train |   n_test |   frozen_slope |   frozen_cw_t |   frozen_cw_p |   walkforward_cw_t |   walkforward_cw_p |
|:-----------|----------:|---------:|---------------:|--------------:|--------------:|-------------------:|-------------------:|
| 2016-01-01 |      2598 |     1243 |        3.90493 |      0.214315 |      0.415151 |           0.103136 |           0.458927 |
| 2020-01-01 |      3086 |      755 |        3.73237 |     -0.198037 |      0.578492 |          -0.400149 |           0.655477 |

## Forecasting

| task   |   start | model                              |   n |    mse |   rmse |    mae |     mbe |   accuracy |   macro_f1 |   dm_hln_p |   clark_west_t |   clark_west_p |
|:-------|--------:|:-----------------------------------|----:|-------:|-------:|-------:|--------:|-----------:|-----------:|-----------:|---------------:|---------------:|
| T1     |    2016 | no change                          | 129 | 0.0331 | 0.1819 | 0.0925 | -0.0272 |     0.5349 |     0.2323 |   nan      |       nan      |       nan      |
| T1     |    2016 | frozen linear S                    | 129 | 0.0471 | 0.217  | 0.1266 | -0.0724 |     0.5349 |     0.2335 |   nan      |       nan      |       nan      |
| T1     |    2016 | frozen linear S+N (Study 2 stance) | 129 | 0.0486 | 0.2204 | 0.1359 | -0.0867 |     0.5194 |     0.2363 |     0.6832 |        -0.1032 |         0.5411 |
| T1     |    2020 | no change                          |  81 | 0.0468 | 0.2163 | 0.1084 | -0.0272 |     0.5679 |     0.2415 |   nan      |       nan      |       nan      |
| T1     |    2020 | frozen linear S                    |  81 | 0.1128 | 0.3358 | 0.2309 | -0.1869 |     0.3086 |     0.2363 |   nan      |       nan      |       nan      |
| T1     |    2020 | frozen linear S+N (Study 2 stance) |  81 | 0.0987 | 0.3142 | 0.2349 | -0.1882 |     0.2716 |     0.2091 |     0.1992 |         1.0326 |         0.1509 |
| T2     |    2016 | no change                          |  87 | 0.0647 | 0.2543 | 0.1322 | -0.0402 |     0.6437 |     0.2611 |   nan      |       nan      |       nan      |
| T2     |    2016 | frozen linear S                    |  87 | 0.0705 | 0.2656 | 0.1633 | -0.0809 |     0.6437 |     0.2611 |   nan      |       nan      |       nan      |
| T2     |    2016 | frozen linear S+N (Study 2 stance) |  87 | 0.0773 | 0.278  | 0.1765 | -0.1004 |     0.6207 |     0.2553 |     0.9993 |        -2.8167 |         0.9976 |
| T2     |    2020 | no change                          |  55 | 0.0898 | 0.2996 | 0.1591 | -0.0409 |     0.6364 |     0.2593 |   nan      |       nan      |       nan      |
| T2     |    2020 | frozen linear S                    |  55 | 0.1026 | 0.3203 | 0.2203 | -0.1191 |     0.5636 |     0.2403 |   nan      |       nan      |       nan      |
| T2     |    2020 | frozen linear S+N (Study 2 stance) |  55 | 0.1307 | 0.3615 | 0.2633 | -0.1786 |     0.4727 |     0.2512 |     0.9999 |        -3.2267 |         0.9994 |
