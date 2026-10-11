# Results: Study 2 (regional Fed speeches, H6)

**H6 verdict: not supported.**

H6 (PREREG_STUDY2.md, tag prereg-study2): on the event set expanded with regional Federal Reserve Bank presidents' speeches, the stance surprise predicts the same-day 2-year Treasury yield change out of sample (one-sided Clark-West, p < 0.05); void if the same test on the previous trading day's change gives p < 0.05.

## H6 (confirmatory)

| pipeline          | role         |   n_test |    cw_t |     cw_p |   void_cw_t |   void_cw_p | verdict       |   sign_accuracy |   cw_p_minutes |   n_minutes |   cw_p_speech |   n_speech |   cw_p_statement |   n_statement |   ols_n |   ols_slope |   ols_t_hac |   ols_p_one_sided |
|:------------------|:-------------|---------:|--------:|---------:|------------:|------------:|:--------------|----------------:|---------------:|------------:|--------------:|-----------:|-----------------:|--------------:|--------:|------------:|------------:|------------------:|
| H6: all documents | confirmatory |     3158 | 0.33574 | 0.368534 |   -0.702441 |    0.758798 | not supported |        0.496549 |        0.29976 |         206 |      0.586106 |       2730 |         0.276762 |           222 |    3158 |     2.19805 |     1.19174 |          0.116682 |

The daily test is conservative for releases after the 2-year yield's afternoon quote: those are matched to a yield fixed before they happened, which biases the test toward no effect.

## Secondary event sets (reported, not decisive)

| pipeline                                    | role      |   n_test |     cw_t |     cw_p |   void_cw_t |   void_cw_p |   sign_accuracy |   cw_p_minutes |   n_minutes |   cw_p_speech |   n_speech |   cw_p_statement |   n_statement |   ols_n |   ols_slope |   ols_t_hac |   ols_p_one_sided |
|:--------------------------------------------|:----------|---------:|---------:|---------:|------------:|------------:|----------------:|---------------:|------------:|--------------:|-----------:|-----------------:|--------------:|--------:|------------:|------------:|------------------:|
| secondary: regional speeches only           | secondary |     2080 | -0.57201 | 0.716342 |    0.108202 |    0.456918 |        0.49556  |     nan        |         nan |      0.716342 |       2080 |       nan        |           nan |    2080 |     1.88155 |     0.86768 |          0.192785 |
| secondary: Plan 5 documents, new classifier | secondary |     1778 |  0.40385 | 0.343161 |   -1.24748  |    0.89389  |        0.496676 |       0.293949 |         206 |      0.411216 |       1350 |         0.431831 |           222 |    1778 |     2.40027 |     1.18626 |          0.11776  |

## Forecasting with rebuilt Fed-stance features (secondary)

| task   |   n |    rmse |   rmse_S |   change_vs_S_pct |   dm_hln_p |   clark_west_t |   clark_west_p |
|:-------|----:|--------:|---------:|------------------:|-----------:|---------------:|---------------:|
| T1     | 308 | 0.17342 | 0.179836 |         -3.56747  |   0.22758  |        1.36663 |       0.08587  |
| T2     | 212 | 0.20643 | 0.205171 |          0.613624 |   0.618919 |        1.18055 |       0.118891 |

## Provenance

- prereg-study2 tag commit: b967d1796cd95c2468e3dc1f6f71d42905349fda
- HEAD: dbad5e1
- sha256 (first 16 hex) regional_corpus.pkl: 4b372a350ef55a55
- sha256 (first 16 hex) doc_stance2.pkl: d99d469f6550bd3c
