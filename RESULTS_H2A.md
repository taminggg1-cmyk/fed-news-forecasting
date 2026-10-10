# Results: Plan 2 (H2a, neural architectures)

Secondary analyses; they cannot change the H1 verdict in RESULTS.md. Settings: PREREG_H2A.md (tag prereg-h2a).

## Ensemble metrics (mean of 8 seeds' forecasts), test windows from 2001

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

| model                          |    mse |   rmse |    mae |      mbe |
|:-------------------------------|-------:|-------:|-------:|---------:|
| paper LSTM (Table 2)           | 0.0565 | 0.2376 | 0.1495 |  -0.0006 |
| paper GRU (Table 2)            | 0.0732 | 0.2705 | 0.1327 |   0.0091 |
| paper LLM, lookahead (Table 3) | 0.0311 | 0.1762 | 0.1483 | nan      |
| our LSTM replication, S        | 0.1725 | 0.4154 | 0.2925 |   0.1028 |
| our LSTM replication, S+N      | 0.1850 | 0.4301 | 0.2848 |  -0.0164 |
| our GRU replication, S         | 0.4181 | 0.6466 | 0.3550 |   0.1887 |
| our GRU replication, S+N       | 0.3172 | 0.5632 | 0.3326 |   0.0733 |

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
