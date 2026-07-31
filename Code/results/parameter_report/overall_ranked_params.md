# Overall noisy-condition parameter ranking

> This analysis evaluates a separable 1-D Taylor model applied independently along x and y. It is not the joint full 2-D Taylor model.

Only the 42 unique noisy rows are ranked. The shared `n=8,h=1` baseline is deduplicated and no-noise rows are excluded.

For a metric `x`, higher-is-better min-max normalization is `(x-min)/(max-min)`. For RMSE, HOG error, and runtime the benefit is inverted as `(max-x)/(max-min)`. A zero-range component contributes 1 equally to every candidate. Global bounds are computed across all 42 noisy rows.

Composite score = `0.45*inverse(HOG error) + 0.30*cosine + 0.20*inverse(RMSE) + 0.05*inverse(runtime)`.

Because both HOG error and cosine are included, HOG stability has 75% of the declared weight. The global ranking also compares different noise severities, so 5% rows tend to outrank otherwise comparable 10% rows. Scores are candidate-set dependent.

Normalization bounds:

- `hog_relative_error`: min=0.2027307972, max=0.6674980913
- `cosine_similarity`: min=0.777223149, max=0.9794501119
- `gradient_RMSE`: min=0.8991260483, max=61.45829701
- `runtime`: min=0.2067654, max=25.2011675

| Rank | Noise | Algorithm | n | h | Score | RMSE | HOG error | Cosine | Runtime (s) | Iterations | Residual |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | S&P 5% | Correntropy loss | 32 | 1 | 0.9603 | 0.8991 | 0.2027 | 0.9795 | 20.071 | 184 | 46.5870 |
| 2 | S&P 5% | Correntropy loss | 16 | 1 | 0.9048 | 2.2623 | 0.2526 | 0.9681 | 13.015 | 160 | 42.0264 |
| 3 | S&P 10% | Correntropy loss | 32 | 1 | 0.7719 | 1.3680 | 0.3321 | 0.9449 | 25.201 | 246 | 59.3599 |
| 4 | S&P 5% | Correntropy loss | 8 | 4 | 0.6704 | 2.2877 | 0.4120 | 0.9151 | 13.706 | 310 | 42.3983 |
| 5 | S&P 5% | Correntropy loss | 8 | 3 | 0.6533 | 2.9177 | 0.4237 | 0.9102 | 11.862 | 197 | 40.3368 |
| 6 | S&P 10% | Correntropy loss | 16 | 1 | 0.6229 | 3.9420 | 0.4333 | 0.9061 | 17.735 | 225 | 55.4069 |
| 7 | S&P 5% | Correntropy loss | 8 | 2 | 0.6156 | 4.3320 | 0.4431 | 0.9018 | 12.785 | 195 | 38.0398 |
| 8 | S&P 5% | Correntropy loss | 8 | 1 | 0.5378 | 8.1962 | 0.4762 | 0.8866 | 18.011 | 344 | 35.1593 |
| 9 | S&P 5% | Direct squared loss | 8 | 4 | 0.5270 | 3.0778 | 0.5134 | 0.8682 | 0.212 | 1 | 41.3520 |
| 10 | S&P 5% | Iterative squared loss | 8 | 4 | 0.5264 | 2.9866 | 0.5133 | 0.8683 | 0.759 | 41 | 41.3562 |
| 11 | S&P 5% | Direct squared loss | 8 | 3 | 0.4926 | 4.1905 | 0.5310 | 0.8590 | 0.215 | 1 | 39.3021 |
| 12 | S&P 5% | Iterative squared loss | 8 | 3 | 0.4914 | 4.0670 | 0.5313 | 0.8589 | 0.801 | 41 | 39.3061 |
| 13 | S&P 10% | Correntropy loss | 8 | 4 | 0.4893 | 3.4862 | 0.5243 | 0.8626 | 8.903 | 164 | 52.6128 |
| 14 | S&P 10% | Correntropy loss | 8 | 3 | 0.4627 | 4.5876 | 0.5341 | 0.8574 | 11.788 | 197 | 51.1903 |
| 15 | S&P 5% | Direct squared loss | 32 | 1 | 0.4561 | 1.6182 | 0.5563 | 0.8453 | 0.293 | 1 | 45.2371 |
| 16 | S&P 5% | Iterative squared loss | 32 | 1 | 0.4534 | 1.5723 | 0.5567 | 0.8451 | 1.378 | 39 | 45.2393 |
| 17 | S&P 5% | Direct squared loss | 16 | 1 | 0.4510 | 4.6858 | 0.5535 | 0.8468 | 0.245 | 1 | 39.9367 |
| 18 | S&P 5% | Iterative squared loss | 16 | 1 | 0.4494 | 4.5518 | 0.5538 | 0.8466 | 0.980 | 39 | 39.9389 |
| 19 | S&P 10% | Correntropy loss | 8 | 2 | 0.4348 | 6.8604 | 0.5413 | 0.8535 | 15.682 | 278 | 49.6517 |
| 20 | S&P 5% | Direct squared loss | 8 | 2 | 0.4306 | 6.4528 | 0.5617 | 0.8423 | 0.220 | 1 | 36.9966 |
| 21 | S&P 5% | Iterative squared loss | 8 | 2 | 0.4300 | 6.2634 | 0.5617 | 0.8422 | 0.824 | 41 | 37.0001 |
| 22 | S&P 10% | Direct squared loss | 8 | 4 | 0.3798 | 4.3308 | 0.5933 | 0.8240 | 0.213 | 1 | 51.3635 |
| 23 | S&P 10% | Direct squared loss | 16 | 1 | 0.3789 | 6.6628 | 0.5896 | 0.8262 | 0.245 | 1 | 52.9483 |
| 24 | S&P 10% | Iterative squared loss | 8 | 4 | 0.3783 | 4.2031 | 0.5937 | 0.8237 | 0.791 | 41 | 51.3678 |
| 25 | S&P 10% | Iterative squared loss | 16 | 1 | 0.3775 | 6.4727 | 0.5899 | 0.8260 | 0.990 | 39 | 52.9503 |
| 26 | S&P 10% | Direct squared loss | 8 | 3 | 0.3652 | 5.9266 | 0.5984 | 0.8210 | 0.215 | 1 | 49.9138 |
| 27 | S&P 10% | Iterative squared loss | 8 | 3 | 0.3635 | 5.7521 | 0.5989 | 0.8207 | 0.854 | 41 | 49.9179 |
| 28 | S&P 10% | Correntropy loss | 8 | 1 | 0.3579 | 13.3623 | 0.5659 | 0.8399 | 21.340 | 313 | 47.8784 |
| 29 | S&P 5% | Direct squared loss | 4 | 1 | 0.3535 | 40.0644 | 0.5427 | 0.8527 | 0.207 | 1 | 23.6715 |
| 30 | S&P 5% | Correntropy loss | 4 | 1 | 0.3531 | 40.0644 | 0.5427 | 0.8527 | 0.416 | 3 | 23.6763 |
| 31 | S&P 10% | Direct squared loss | 32 | 1 | 0.3487 | 2.3225 | 0.6135 | 0.8118 | 0.289 | 1 | 57.6964 |
| 32 | S&P 10% | Iterative squared loss | 32 | 1 | 0.3459 | 2.2573 | 0.6139 | 0.8115 | 1.377 | 39 | 57.6982 |
| 33 | S&P 10% | Direct squared loss | 8 | 2 | 0.3365 | 9.1231 | 0.6081 | 0.8151 | 0.222 | 1 | 48.3499 |
| 34 | S&P 10% | Iterative squared loss | 8 | 2 | 0.3353 | 8.8547 | 0.6085 | 0.8149 | 0.900 | 41 | 48.3537 |
| 35 | S&P 5% | Iterative squared loss | 8 | 1 | 0.3244 | 12.9798 | 0.6071 | 0.8157 | 0.875 | 41 | 34.0946 |
| 36 | S&P 5% | Direct squared loss | 8 | 1 | 0.3232 | 13.3740 | 0.6077 | 0.8153 | 0.220 | 1 | 34.0917 |
| 37 | S&P 5% | Iterative squared loss | 4 | 1 | 0.3203 | 43.6230 | 0.5543 | 0.8464 | 0.634 | 29 | 23.7243 |
| 38 | S&P 10% | Iterative squared loss | 8 | 1 | 0.2609 | 18.2652 | 0.6315 | 0.8006 | 0.874 | 41 | 46.5314 |
| 39 | S&P 10% | Direct squared loss | 8 | 1 | 0.2601 | 18.8205 | 0.6316 | 0.8005 | 0.229 | 1 | 46.5281 |
| 40 | S&P 10% | Direct squared loss | 4 | 1 | 0.0748 | 56.4351 | 0.6633 | 0.7800 | 0.208 | 1 | 33.0282 |
| 41 | S&P 10% | Correntropy loss | 4 | 1 | 0.0744 | 56.4351 | 0.6633 | 0.7800 | 0.415 | 3 | 33.0349 |
| 42 | S&P 10% | Iterative squared loss | 4 | 1 | 0.0491 | 61.4583 | 0.6675 | 0.7772 | 0.633 | 29 | 33.0973 |
