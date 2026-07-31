# Parameter sensitivity analysis

> This experiment evaluates a separable 1-D Taylor model applied independently along x and y. It is not the joint full 2-D Taylor model.

## Experimental controls

- Polynomial order: `3`
- Symmetric samples per axis: `[4, 8, 16, 32]`
- Sample spacing values: `[1, 2, 3, 4]`
- Random seed: `42`
- Noise scenarios: no noise, 5% salt-and-pepper, and 10% salt-and-pepper
- Common valid-image padding: `16` pixels
- Solver tolerance: `0.001`
- Maximum solver iterations: `500`
- Converged rows: `72/72`; iteration ranges: direct `1–1`, iterative `29–41`, correntropy `3–344`
- HOG normalization: `L2-Hys`
- Fixed HOG weighting: `huber` (delta=2.0, window=5)
- Runtime: median of `3` noisy inference runs; per-configuration matrix setup and clean-reference computation are excluded

Salt-and-pepper density denotes the fraction of selected pixel locations. A selected endpoint-valued pixel can already equal its assigned impulse value, so the realized fraction of changed numeric values can be slightly smaller.

For every `(n, h, algorithm)` combination, the clean reference uses the identical estimator and HOG parameters as its noisy counterpart. The same precomputed noisy image is reused across all parameter comparisons in a scenario.

The reported final residual is the unweighted RMS Taylor-fit residual over the horizontal and vertical per-axis systems on the same common image ROI used for all metrics. It is a fit diagnostic, not the correntropy objective, and its absolute scale can change with n and h. Direct loss reports one closed-form solve; iterative squared loss and correntropy report the maximum actual batch-iteration count across x and y. `converged=false` and `(limit)` in a panel mean that at least one axis reached `max_iter` before the strict all-pixel stopping rule.

## Per-axis design matrices

`n_points` counts the symmetric samples along one axis, including both negative and positive offsets. It does not count a full 2-D neighbourhood.

| Sweep | n | h | Matrix shape | Rank | Condition number |
|---|---:|---:|---:|---:|---:|
| n | 4 | 1 | 4x3 | 3 | 6.655 |
| n | 8 | 1 | 8x3 | 3 | 6.654 |
| n | 16 | 1 | 16x3 | 3 | 20.020 |
| n | 32 | 1 | 32x3 | 3 | 74.369 |
| h | 8 | 1 | 8x3 | 3 | 6.654 |
| h | 8 | 2 | 8x3 | 3 | 22.557 |
| h | 8 | 3 | 8x3 | 3 | 50.277 |
| h | 8 | 4 | 8x3 | 3 | 89.239 |

## Descriptive results

The table identifies the setting with the lowest gradient RMSE and, separately, settings with the highest HOG cosine similarity and lowest HOG relative error. These optima are descriptive rather than inferential.

| Sweep | Noise | Algorithm | Min-RMSE setting | RMSE | Max-cosine setting | Cosine | Min-error setting | Relative error |
|---|---|---|---:|---:|---:|---:|---:|---:|
| n | saltpepper_5 | Direct squared loss | n=32 | 1.6182 | n=4 | 0.8527 | n=4 | 0.5427 |
| n | saltpepper_5 | Iterative squared loss | n=32 | 1.5723 | n=16 | 0.8466 | n=16 | 0.5538 |
| n | saltpepper_5 | Correntropy loss | n=32 | 0.8991 | n=32 | 0.9795 | n=32 | 0.2027 |
| n | saltpepper_10 | Direct squared loss | n=32 | 2.3225 | n=16 | 0.8262 | n=16 | 0.5896 |
| n | saltpepper_10 | Iterative squared loss | n=32 | 2.2573 | n=16 | 0.8260 | n=16 | 0.5899 |
| n | saltpepper_10 | Correntropy loss | n=32 | 1.3680 | n=32 | 0.9449 | n=32 | 0.3321 |
| h | saltpepper_5 | Direct squared loss | h=4 | 3.0778 | h=4 | 0.8682 | h=4 | 0.5134 |
| h | saltpepper_5 | Iterative squared loss | h=4 | 2.9866 | h=4 | 0.8683 | h=4 | 0.5133 |
| h | saltpepper_5 | Correntropy loss | h=4 | 2.2877 | h=4 | 0.9151 | h=4 | 0.4120 |
| h | saltpepper_10 | Direct squared loss | h=4 | 4.3308 | h=4 | 0.8240 | h=4 | 0.5933 |
| h | saltpepper_10 | Iterative squared loss | h=4 | 4.2031 | h=4 | 0.8237 | h=4 | 0.5937 |
| h | saltpepper_10 | Correntropy loss | h=4 | 3.4862 | h=4 | 0.8626 | h=4 | 0.5243 |

## Interpretation limits

Gradient RMSE here compares each noisy result with its matched clean estimator output. It measures noise stability, not absolute derivative accuracy against analytical ground truth. The no-noise rows are regression checks and must yield RMSE=0, cosine=1, and relative error=0.

This deterministic single-image, single-seed experiment supports parameter sensitivity analysis only. Population-level claims require multiple images and seeds; absolute accuracy claims require an analytical synthetic-gradient reference.

## Generated artifacts

- [Fixed h, varying n](n_analysis.png)
- [Fixed n, varying h](h_analysis.png)
- [All numerical measurements](parameter_analysis_results.csv)
- [Reproducibility configuration](parameter_analysis_config.json)
