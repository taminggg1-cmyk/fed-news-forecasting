# Results: Plan 5 (text as a forecasting method)

**H5 verdict: not supported.**

H5 (pre-registered in PREREG_TEXT.md, tag prereg-text): the Fed stance surprise predicts the same-day 2-year Treasury yield change out of sample (one-sided Clark-West, p < 0.05); void if the same test on the day-before change gives p < 0.05. It does not change the H1 or H2a verdicts.

## Event study (confirmatory)

| pipeline   |   n_test |   cw_t |   cw_p |   void_cw_t |   void_cw_p | verdict       |   sign_accuracy |   cw_p_minutes |   n_minutes |   cw_p_speech |   n_speech |   cw_p_statement |   n_statement |   ols_n |   ols_slope |   ols_t_hac |   ols_p_one_sided |
|:-----------|---------:|-------:|-------:|------------:|------------:|:--------------|----------------:|---------------:|------------:|--------------:|-----------:|-----------------:|--------------:|--------:|------------:|------------:|------------------:|
| chronobert |     1778 | 1.1307 | 0.1291 |     -0.2066 |      0.5818 | not supported |          0.5047 |         0.1452 |         206 |        0.3799 |       1350 |           0.1686 |           222 |    1778 |      3.3912 |      1.5859 |            0.0564 |

The daily test is conservative for releases after the 2-year yield's afternoon quote (late speeches, the end of press conferences): those are matched to a yield fixed before they happened, which biases the test toward no effect.

About 30% of test events follow an event on the previous trading day, so the void-rule outcome there is the previous event's own outcome.

Method details: press conferences are grouped with statements; MIN_TRAIN never binds; the first events (no earlier trading day pair) are dropped; zero-change days are excluded from sign accuracy.

## Look-ahead comparison (exploratory, added after results)

Direct tests of the naive pipeline against ChronoBERT on the same dates (column dm_p_naive_vs_chrono: DM-HLN p that the naive forecast is better; Clark-West is not meaningful because the models are not nested):

| task   | pipeline                       |   n |   rmse |   rmse_S |   change_vs_S_pct |   dm_hln_p |   clark_west_t |   clark_west_p |   dm_p_naive_vs_chrono |
|:-------|:-------------------------------|----:|-------:|---------:|------------------:|-----------:|---------------:|---------------:|-----------------------:|
| T1     | chronobert (time-consistent)   | 308 | 0.1664 |   0.1798 |           -7.4599 |     0.0769 |         2.0398 |         0.0207 |               nan      |
| T1     | naive (ModernBERT, all labels) | 308 | 0.1627 |   0.1798 |           -9.5261 |     0.0429 |         2.1468 |         0.0159 |                 0.0552 |
| T2     | chronobert (time-consistent)   | 212 | 0.1994 |   0.2052 |           -2.8301 |     0.0609 |         3.0307 |         0.0012 |               nan      |
| T2     | naive (ModernBERT, all labels) | 212 | 0.1988 |   0.2052 |           -3.0885 |     0.0657 |         2.9405 |         0.0016 |                 0.4028 |

Event study, both pipelines on their common test days (exploratory, added after results; MSE ratio = large model over small model):

|   n_common |   mse_small_chrono |   mse_large_chrono |   mse_ratio_chrono |   mse_small_naive |   mse_large_naive |   mse_ratio_naive |   dm_p_naive_vs_chrono |
|-----------:|-------------------:|-------------------:|-------------------:|------------------:|------------------:|------------------:|-----------------------:|
|       1778 |            33.9075 |            33.9367 |             1.0009 |           33.9075 |           33.8544 |            0.9984 |                 0.1361 |

The naive pipeline differs in at least four ways: the language model (one ModernBERT instead of 27 ChronoBERT vintages), label dating (all labels at once), label count (3,074 against 418-2,438 after the 20% hold-out) and overlap (its training sentences come from the documents it scores). Its stance correlates 0.58 with ChronoBERT's. The comparison illustrates a typical pipeline; it does not isolate look-ahead, and the gap is not statistically significant.

Naive model: answerdotai/ModernBERT-base
revision 8949b909ec900327062f0ebf497f51aef5e6f0c8 (local Hugging Face cache, refs/main at run time), fine-tuned on all labels at once (deliberate look-ahead).

Event study with the naive stance (exploratory, added after results; no H5 verdict, which applies only to the pre-registered ChronoBERT pipeline):

| pipeline   |   n_test |   cw_t |   cw_p |   void_cw_t |   void_cw_p |   sign_accuracy |   cw_p_minutes |   n_minutes |   cw_p_speech |   n_speech |   cw_p_statement |   n_statement |   ols_n |   ols_slope |   ols_t_hac |   ols_p_one_sided |
|:-----------|---------:|-------:|-------:|------------:|------------:|----------------:|---------------:|------------:|--------------:|-----------:|-----------------:|--------------:|--------:|------------:|------------:|------------------:|
| naive      |     1778 | 1.6662 | 0.0478 |     -0.4425 |      0.6709 |          0.4987 |         0.5728 |         206 |        0.1032 |       1350 |           0.0974 |           222 |    1778 |      3.1422 |      2.2335 |            0.0128 |

## Market benchmark (descriptive)

Descriptive. Headline row: the 3-month bill on the full sample. Its DM p-values are not tests. The expanding-mean spread drifts over time (the bill-to-rate gap shrank from the 1990s to the 2020s), so the bill forecast is biased upward and is a weak benchmark; do not read this table as text beating the market.

| task   | window   | bill_dates   | model   |   n |    mse |   rmse |    mae |     mbe |   accuracy |   macro_f1 |   p_vs_no_change |   p_vs_bill |
|:-------|:---------|:-------------|:--------|----:|-------:|-------:|-------:|--------:|-----------:|-----------:|-----------------:|------------:|
| T1     | full     | bill3m       | rw      | 308 | 0.0329 | 0.1813 | 0.0917 |  0.0087 |     0.526  |     0.2298 |         nan      |      0.0001 |
| T1     | full     | bill3m       | S       | 308 | 0.0323 | 0.1798 | 0.1127 | -0.0304 |     0.5584 |     0.4147 |           0.4554 |      0.0001 |
| T1     | full     | bill3m       | S+N     | 308 | 0.0277 | 0.1664 | 0.1096 | -0.0115 |     0.539  |     0.4233 |           0.0558 |      0      |
| T1     | full     | bill3m       | S+M     | 308 | 0.0172 | 0.1311 | 0.0889 |  0.0026 |     0.6623 |     0.6024 |           0      |      0      |
| T1     | full     | bill3m       | bill3m  | 308 | 0.0545 | 0.2335 | 0.1961 |  0.1438 |     0.4838 |     0.4996 |           0.9999 |    nan      |
| T1     | full     | bill4w       | rw      | 288 | 0.0279 | 0.167  | 0.0816 | -0.0069 |     0.5451 |     0.2352 |         nan      |      0.0018 |
| T1     | full     | bill4w       | S       | 288 | 0.0299 | 0.1729 | 0.1053 | -0.0304 |     0.5486 |     0.3731 |           0.6772 |      0.007  |
| T1     | full     | bill4w       | S+N     | 288 | 0.0262 | 0.1618 | 0.1044 | -0.0115 |     0.5278 |     0.3836 |           0.2668 |      0.0013 |
| T1     | full     | bill4w       | S+M     | 288 | 0.0154 | 0.1242 | 0.0824 |  0.0066 |     0.6562 |     0.5663 |           0.0002 |      0      |
| T1     | full     | bill4w       | bill4w  | 288 | 0.0538 | 0.232  | 0.1615 |  0.0264 |     0.4722 |     0.445  |           0.9982 |    nan      |
| T1     | paper    | bill3m       | rw      | 112 | 0.037  | 0.1925 | 0.0992 | -0.0365 |     0.5179 |     0.2275 |         nan      |      0.4082 |
| T1     | paper    | bill3m       | S       | 112 | 0.0447 | 0.2115 | 0.1192 | -0.0555 |     0.4821 |     0.2182 |           0.7952 |      0.706  |
| T1     | paper    | bill3m       | S+N     | 112 | 0.0384 | 0.196  | 0.1214 | -0.0335 |     0.4375 |     0.2275 |           0.7059 |      0.4774 |
| T1     | paper    | bill3m       | S+M     | 112 | 0.0207 | 0.1438 | 0.0954 |  0.0043 |     0.6429 |     0.5556 |           0.0035 |      0.0005 |
| T1     | paper    | bill3m       | bill3m  | 112 | 0.0388 | 0.197  | 0.1632 |  0.1563 |     0.625  |     0.6393 |           0.5918 |    nan      |
| T1     | paper    | bill4w       | rw      | 112 | 0.037  | 0.1925 | 0.0992 | -0.0365 |     0.5179 |     0.2275 |         nan      |      0.8534 |
| T1     | paper    | bill4w       | S       | 112 | 0.0447 | 0.2115 | 0.1192 | -0.0555 |     0.4821 |     0.2182 |           0.7952 |      0.9204 |
| T1     | paper    | bill4w       | S+N     | 112 | 0.0384 | 0.196  | 0.1214 | -0.0335 |     0.4375 |     0.2275 |           0.7059 |      0.9002 |
| T1     | paper    | bill4w       | S+M     | 112 | 0.0207 | 0.1438 | 0.0954 |  0.0043 |     0.6429 |     0.5556 |           0.0035 |      0.1263 |
| T1     | paper    | bill4w       | bill4w  | 112 | 0.0284 | 0.1684 | 0.1201 |  0.0647 |     0.5625 |     0.5011 |           0.1466 |    nan      |
| T2     | full     | bill3m       | rw      | 212 | 0.0582 | 0.2411 | 0.1256 |  0.0124 |     0.6604 |     0.2652 |         nan      |      0.5398 |
| T2     | full     | bill3m       | S       | 212 | 0.0421 | 0.2052 | 0.1354 | -0.0309 |     0.6792 |     0.4969 |           0.0007 |      0.0307 |
| T2     | full     | bill3m       | S+N     | 212 | 0.0397 | 0.1994 | 0.1298 | -0.022  |     0.6981 |     0.5431 |           0.0003 |      0.0164 |
| T2     | full     | bill3m       | S+M     | 212 | 0.0174 | 0.1321 | 0.0737 |  0.0166 |     0.8679 |     0.8328 |           0      |      0      |
| T2     | full     | bill3m       | bill3m  | 212 | 0.0572 | 0.2391 | 0.2028 |  0.1439 |     0.4387 |     0.5118 |           0.4602 |    nan      |
| T2     | full     | bill4w       | rw      | 194 | 0.051  | 0.2258 | 0.1102 | -0.0135 |     0.6907 |     0.2724 |         nan      |      0.0894 |
| T2     | full     | bill4w       | S       | 194 | 0.0377 | 0.1942 | 0.1239 | -0.0437 |     0.7062 |     0.4927 |           0.0039 |      0.0055 |
| T2     | full     | bill4w       | S+N     | 194 | 0.038  | 0.195  | 0.1233 | -0.0323 |     0.7062 |     0.5211 |           0.0063 |      0.007  |
| T2     | full     | bill4w       | S+M     | 194 | 0.0161 | 0.1267 | 0.0664 |  0.0109 |     0.8711 |     0.8182 |           0      |      0      |
| T2     | full     | bill4w       | bill4w  | 194 | 0.0663 | 0.2575 | 0.1947 |  0.0338 |     0.4948 |     0.506  |           0.9106 |    nan      |

## Ablation (exploratory, added after results)

Unadjusted p-values from 12 exploratory comparisons; descriptive only.

| task   | model          |   rmse |   accuracy |   change_vs_S_pct |   n |   dm_hln_p |   clark_west_t |   clark_west_p |
|:-------|:---------------|-------:|-----------:|------------------:|----:|-----------:|---------------:|---------------:|
| T1     | S+N (all)      | 0.1664 |     0.539  |           -7.4599 | 308 |     0.0769 |         2.0398 |         0.0207 |
| T1     | S+stance       | 0.1823 |     0.5227 |            1.3562 | 308 |     0.6698 |         0.5969 |         0.2753 |
| T1     | S+N-stance     | 0.1691 |     0.5714 |           -5.9509 | 308 |     0.1059 |         1.8669 |         0.031  |
| T1     | S+beige        | 0.1747 |     0.5617 |           -2.8702 | 308 |     0.0977 |         2.46   |         0.0069 |
| T1     | S+N-beige      | 0.1694 |     0.5357 |           -5.8077 | 308 |     0.0452 |         2.7649 |         0.0028 |
| T1     | S+newspapers   | 0.1761 |     0.5584 |           -2.0914 | 308 |     0.1509 |         2.5547 |         0.0053 |
| T1     | S+N-newspapers | 0.1739 |     0.5487 |           -3.3282 | 308 |     0.1412 |         2.9315 |         0.0017 |
| T2     | S+N (all)      | 0.1994 |     0.6981 |           -2.8301 | 212 |     0.0609 |         3.0307 |         0.0012 |
| T2     | S+stance       | 0.2021 |     0.7028 |           -1.4823 | 212 |     0.0166 |         3.5918 |         0.0002 |
| T2     | S+N-stance     | 0.2037 |     0.6981 |           -0.6925 | 212 |     0.3655 |         1.7657 |         0.0387 |
| T2     | S+beige        | 0.2042 |     0.7028 |           -0.4516 | 212 |     0.2674 |         1.9891 |         0.0233 |
| T2     | S+N-beige      | 0.1994 |     0.684  |           -2.8257 | 212 |     0.0561 |         2.9892 |         0.0014 |
| T2     | S+newspapers   | 0.2055 |     0.6792 |            0.1453 | 212 |     0.5312 |         1.3721 |         0.085  |
| T2     | S+N-newspapers | 0.2011 |     0.717  |           -1.9663 | 212 |     0.0059 |         4.06   |         0      |

## Provenance

- prereg-text tag commit: 8d2c5c197d2b6dbaa131053067d246e8a966dd7a
- HEAD: f8c8abf
- sha256 (first 16 hex) doc_stance.pkl: 511dd312f363069b
- sha256 (first 16 hex) doc_stance_naive.pkl: d990364251528f69
- sha256 (first 16 hex) market_bill.pkl: 0bdc7577a91e2d15
- naive_model.txt: answerdotai/ModernBERT-base
revision 8949b909ec900327062f0ebf497f51aef5e6f0c8 (local Hugging Face cache, refs/main at run time)
