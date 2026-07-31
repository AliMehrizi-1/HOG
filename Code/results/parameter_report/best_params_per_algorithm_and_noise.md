# Best parameters by algorithm and noise scenario

> This analysis evaluates a separable 1-D Taylor model applied independently along x and y. It is not the joint full 2-D Taylor model.

The table searches the seven evaluated one-factor-at-a-time pairs `(4,1), (8,1), (16,1), (32,1), (8,2), (8,3), (8,4)`. It does not represent a full Cartesian `n x h` grid.

| Noise | Algorithm | Criterion | n | h | RMSE | HOG error | Cosine | Runtime (s) | Iterations | Residual |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| S&P 5% | Direct squared loss | Minimum gradient RMSE | 32 | 1 | 1.6182 | 0.5563 | 0.8453 | 0.293 | 1 | 45.2371 |
| S&P 5% | Direct squared loss | Minimum HOG error | 8 | 4 | 3.0778 | 0.5134 | 0.8682 | 0.212 | 1 | 41.3520 |
| S&P 5% | Direct squared loss | Maximum cosine | 8 | 4 | 3.0778 | 0.5134 | 0.8682 | 0.212 | 1 | 41.3520 |
| S&P 5% | Direct squared loss | Minimum runtime | 4 | 1 | 40.0644 | 0.5427 | 0.8527 | 0.207 | 1 | 23.6715 |
| S&P 5% | Iterative squared loss | Minimum gradient RMSE | 32 | 1 | 1.5723 | 0.5567 | 0.8451 | 1.378 | 39 | 45.2393 |
| S&P 5% | Iterative squared loss | Minimum HOG error | 8 | 4 | 2.9866 | 0.5133 | 0.8683 | 0.759 | 41 | 41.3562 |
| S&P 5% | Iterative squared loss | Maximum cosine | 8 | 4 | 2.9866 | 0.5133 | 0.8683 | 0.759 | 41 | 41.3562 |
| S&P 5% | Iterative squared loss | Minimum runtime | 4 | 1 | 43.6230 | 0.5543 | 0.8464 | 0.634 | 29 | 23.7243 |
| S&P 5% | Correntropy loss | Minimum gradient RMSE | 32 | 1 | 0.8991 | 0.2027 | 0.9795 | 20.071 | 184 | 46.5870 |
| S&P 5% | Correntropy loss | Minimum HOG error | 32 | 1 | 0.8991 | 0.2027 | 0.9795 | 20.071 | 184 | 46.5870 |
| S&P 5% | Correntropy loss | Maximum cosine | 32 | 1 | 0.8991 | 0.2027 | 0.9795 | 20.071 | 184 | 46.5870 |
| S&P 5% | Correntropy loss | Minimum runtime | 4 | 1 | 40.0644 | 0.5427 | 0.8527 | 0.416 | 3 | 23.6763 |
| S&P 10% | Direct squared loss | Minimum gradient RMSE | 32 | 1 | 2.3225 | 0.6135 | 0.8118 | 0.289 | 1 | 57.6964 |
| S&P 10% | Direct squared loss | Minimum HOG error | 16 | 1 | 6.6628 | 0.5896 | 0.8262 | 0.245 | 1 | 52.9483 |
| S&P 10% | Direct squared loss | Maximum cosine | 16 | 1 | 6.6628 | 0.5896 | 0.8262 | 0.245 | 1 | 52.9483 |
| S&P 10% | Direct squared loss | Minimum runtime | 4 | 1 | 56.4351 | 0.6633 | 0.7800 | 0.208 | 1 | 33.0282 |
| S&P 10% | Iterative squared loss | Minimum gradient RMSE | 32 | 1 | 2.2573 | 0.6139 | 0.8115 | 1.377 | 39 | 57.6982 |
| S&P 10% | Iterative squared loss | Minimum HOG error | 16 | 1 | 6.4727 | 0.5899 | 0.8260 | 0.990 | 39 | 52.9503 |
| S&P 10% | Iterative squared loss | Maximum cosine | 16 | 1 | 6.4727 | 0.5899 | 0.8260 | 0.990 | 39 | 52.9503 |
| S&P 10% | Iterative squared loss | Minimum runtime | 4 | 1 | 61.4583 | 0.6675 | 0.7772 | 0.633 | 29 | 33.0973 |
| S&P 10% | Correntropy loss | Minimum gradient RMSE | 32 | 1 | 1.3680 | 0.3321 | 0.9449 | 25.201 | 246 | 59.3599 |
| S&P 10% | Correntropy loss | Minimum HOG error | 32 | 1 | 1.3680 | 0.3321 | 0.9449 | 25.201 | 246 | 59.3599 |
| S&P 10% | Correntropy loss | Maximum cosine | 32 | 1 | 1.3680 | 0.3321 | 0.9449 | 25.201 | 246 | 59.3599 |
| S&P 10% | Correntropy loss | Minimum runtime | 4 | 1 | 56.4351 | 0.6633 | 0.7800 | 0.415 | 3 | 33.0349 |
